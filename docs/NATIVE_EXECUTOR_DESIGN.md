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
| `runtime/gpu/native_exec/reflection_model.h` | Reflection loader: the `.reflect.json` sidecar (v1..v3), `layout-manifest.json`, registry lookups, merged bind group layouts, reflected vertex layout (`reflection_loader.js`). No WebGPU types; unit-tested in CI. |
| `runtime/gpu/native_exec/replay_frames.h` | `splitFrames`, frame digest and sampled stats of `tools/replay/replay.html`. No WebGPU types. |
| `runtime/gpu/native_exec/native_executor.{h,cpp}` | Dawn device, swapchain (offscreen texture or surface), packet handlers, Present. |
| `runtime/gpu/native_exec/native_window.{h,cpp}` | Optional Win32 window (other hosts: headless only for now). |
| `runtime/gpu/native_exec/live_pump.h` | Bounded queue + executor thread for live play (`--window`); executor-agnostic, unit-tested. |
| `runtime/gpu/native_exec/pg6_native_exec_main.cpp` | Trace replay tool. |
| `tests/test_native_exec_model.cpp` | Model tests (CI). |
| `tests/test_native_exec_reflection.cpp` | Reflection loader model tests (CI). |
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
- A pipeline whose explicit layout does not satisfy the statically referenced WGSL `@group(0)` bindings
  gets Dawn's `layout: auto` plus the worker's group 0 bind-group remap (`layout_model.h`:
  `parseWgslGroup0Bindings`, `isExplicitLayoutCompatibleWithWgsl`, the resource matching and dynamic
  offset folding of `_getOrCreateRemappedBindGroup`, dummies, default samplers; applied while the pass
  commands run, like `_preparePassCommandsForFlush`). Counted as `auto_layout_remap_pipelines`,
  `remapped_bind_groups`, `remap_*`. Not ported: the PresentX plane bind-group cache (a cache only), the
  async-compile retry with `layout:'auto'`, v1 `SetBindGroup` on a remapped pipeline, indirect draws
  (the worker does not remap them either). `pipeline_layout_id == 0` pipelines are not remapped (as in the
  worker); `auto_layout_pipelines` counts them.
- Reflection loader (`reflection_model.h`, a port of `web/workers/reflection_loader.js`):
  `NativeExecutor::load_reflection` (`pg6_native_exec --reflection`) takes a `playgta6-stage-reflection`
  sidecar (versions 1..3) or an S6 `layout-manifest.json`; `load_wgsl_pack` also loads an existing
  `<pack>.reflect.json` (the boot orchestrator's `deriveReflectionSidecarUrl`; WGPK v1 and v2 packs alike,
  the sidecar is keyed by the shader's pack hash). Validation is the loader's: a rejected document registers
  nothing (`reflection_load_errors`). Binding sources (user SGPRs, heaps, static samplers, ...) are
  validated and dropped, they never reach WebGPU; `diagnoseStageBindings`, the root-signature join and
  `synthesizeMissingStaticSamplers` are producer-side and not ported. `_resolveShaderReflection` looks a
  shader up by its hash (the pack hash, else FNV-1a 64 of the WGSL text, the worker's `hashHex`), then by
  `"<stage>:<entryPoint>"` (manifest shaders); `registerShaderReflectionKey` has no native producer.
  A record gives a v1 pipeline its layout (`buildPipelineLayout`: the stages' layouts merged per
  group, conflicts and groups >= 4 are op errors that create no pipeline), its entry points and, with a
  non-zero stride, the vertex layout of `buildVertexBufferLayouts` (a rejected one keeps the single
  attribute default: `reflection_vertex_fallbacks`). v1 bind groups take the reflected group layout and
  synthesize default samplers for unbound sampler slots (`v1_synth_samplers`, also for v2 layout ids
  with known entries, like `_getBindGroupLayoutEntriesForSynthesis`). `ManifestReference` layouts
  (flag `0x1`) resolve to the loaded manifest's `bind_group_layouts` / `pipeline_layouts` under `content_id`
  (`web/` leaves the `resolveManifestLayout` hook unbound; the manifest tables are what its doc comment
  names); without a manifest they stay unavailable (`manifest_layouts_unavailable`). Not reproduced: an
  `externalTexture` layout entry (the pinned Dawn has no `ExternalTextureBindingLayout`:
  `reflection_layout_unsupported`) and a storage texture format Dawn does not know by name.
- v1 pipelines (`8`/`19`) and bind groups (`15`/`16`) of a shader without a registry record take the
  worker's path for "no reflection": layout `auto`, entry points `vs_main` /
  `fs_main` / `cs_main`, one `float32xN` attribute at location 0 for a non-zero stride, the canvas format
  as the only target, bind group layouts from `pipeline.getBindGroupLayout(group)`, a missing
  `resource_id == 0` sampler replaced by the default linear sampler. Counted as `reflection_unavailable`.
  The G34 fallback shader profiles (the worker's shader router) are not reproduced.
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
- Non-canvas stream swapchain formats (sRGB, packed, others) use the worker's intermediate texture + the
  worker's fullscreen blit (`blit_model.h`, `blit_frames`); the canvas keeps its format and an sRGB source is
  decoded by the sampler and written as is, exactly like the worker. Beyond the worker, a stream size that
  differs from the canvas (`present_width/height` headless, `fixed_stream_size` with a window) is
  letterboxed into the canvas (`window_layout.h` `letterbox_rect`, the window's `present_rect`), bars black.
  Only depth/stencil and block-compressed codes still count as `swapchain_format_unsupported`.
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
| the layout:auto bind-group remap | `_getOrCreateRemappedBindGroup`, `_preparePassCommandsForFlush` | **M2b (done)** |
| `layout-manifest.json` layouts, the reflection loader (`.reflect.json` v1..v3 sidecar incl. WGPK v2 packs) | `reflection_loader.js` | **M2b (done)** |
| `requestNegotiatedDevice` limit negotiation | `requestNegotiatedDevice` | **M2b (done)** |
| Swapchain blit fallback (+ letterbox) | `wgpu_swapchain.js` | **M2b (done)** |
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

- **Headless live: `--live-executor`** runs exactly this path (same LivePump, executor, async pipelines, blob
  cache, `--live-digest`, `--screenshot-at`, `--gpu-validation`, gpu_time, `--gpu-capture`, `--host-input`) but
  presents into the executor's offscreen swapchain texture (BGRA8Unorm, `--window-size`, the same blit and
  letterbox) and opens no window, so agents and CI-like runs can test the live executor without taking over the
  screen. Differences: no window messages (no F12, no `--window-script`, no present-mode / title; `--max-fps`
  still applies), nothing can close it (end it with `--max-ms` or the guest), a guest fault is not held open.
  `tools/run_native_boot.py --live-executor` builds with the native executor and passes it; a `--window` in its
  extra options is turned into `--live-executor` unless `--show-window` is given (only `--play` and
  `--show-window` open a window). `--window` and `--live-executor` are mutually exclusive.
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

## On-screen overlay (F3)

An optional stats box for the live window, because the title bar is not visible in fullscreen. Toggle with the
`overlay` hotkey (default F3, `runtime/os/window_hotkeys.h`, rebindable in `keymap.ini` `[hotkeys]`), start with it on
via `playgta6_boot --overlay` / `ExecutorOptions::overlay`, switch at run time with `NativeExecutor::set_overlay()`.
Content: fps and frame time, a graph of the last 120 frames (green, orange above 20 ms, red above 33 ms, 50 ms at the
top), present mode, draws per frame, GPU ms per frame (gpu_time table; "N/A" without `timestamp-query`), audio
underruns (host feed, `set_overlay_extra`) and, once the guest stopped, `STOPPED: <summary>` in red.

It is digest-neutral by construction: `present()` encodes the digest copy, the screenshot copy and the probe copy
first and the overlay pass last (blend over the canvas, premultiplied, nearest-scaled 1x/2x/3x by canvas height), and
nothing reads the canvas after it. With the overlay on in a window the frame always goes through the intermediate
texture, so after a fault (no more presents) `poll()` redraws the kept last frame with the fault text. The font
(5x7, ASCII, own glyph table), history, text and the CPU twin of the composite are `gpu/native_exec/overlay_model.h`
(`tests/test_overlay_model.cpp`); `test_native_executor` checks that digests are identical with the overlay on and the
box is on the canvas.

## Present any target: `--show-target`

A read-only diagnostic for black or garbled screens: instead of the swapchain image the window (or the
headless offscreen swapchain) shows any render target or texture of the stream, so one can see which
pass produced nothing. `pg6_native_exec --show-target <selector>` and `playgta6_boot --window --show-target
<selector>` (through `run_native_boot.py --play ... -- --show-target ...`) take it; `ExecutorOptions::show_target`
and `NativeExecutor::set_show_target()` set it in code. The model (selector grammar, resolution, cycling, which
textures can be shown, the shader and its CPU twin) is `runtime/gpu/native_exec/show_target_model.h`, unit
tested without a GPU in `tests/test_native_exec_model.cpp`.

| Selector | Shows |
|---|---|
| `<id>`, `tex:<id>` | the texture with that stream id as colour (rgb, alpha forced to 1) |
| `depth:<id>` | a depth texture as grey, through its r32float depth twin (backlog 35's `depth_twin`; created on demand, refreshed before the frame's passes and the display pass) |
| `pass:<N>` / `pass:<N>:depth` | the colour (else depth) / depth target of the N-th render pass of the presented frame, 0-based in encode order; the attachment's own mip and layer are the default |
| `last` | the colour target of the last pass that rendered to a texture |
| `cycle` | nothing until a hotkey picks |

Suffixes: `@layer=L`, `@mip=M` (clamped into the texture; the mip is shown at its own size), `@range=LO:HI`
(value to 0..1 as `(v-LO)/(HI-LO)`: reverse-Z depth, HDR colour). Hotkeys (only when `--show-target` is given;
they are swallowed by the window): **Ctrl+F10** next / **Ctrl+F9** previous render target of the last frame (sorted by id, colour
before depth, then back to the swapchain image), **Ctrl+F8** next array layer, **Ctrl+F7** next mip (rebindable: keymap.ini `[hotkeys]`). The title bar names what
is shown (`tex 12 depth 1280x720 depth32float mip 0/1 layer 0/1`) or why it cannot be (`unavailable: pass 9 does not
exist`), also in the live window where the boot title tick appends it.

How it stays read-only: the frame is encoded exactly as without it, the `--live-digest` copy of the stream image is
taken first, and only then one extra pass draws the chosen texture over the canvas (`textureLoad` with nearest
sampling, letterboxed like the normal blit, black bars; multisampled targets show sample 0). The guest's resources,
its pass list and its present are untouched; the depth twin is the only object created for the guest's textures
and is not read by the guest unless it already would be. Not shown, reported in the label and counted in
`show_target_unavailable`: 3D, integer, stencil-only and multisampled-depth textures and textures created without
`TextureBinding` (they cannot be sampled). Stats: `show_target_frames`, `show_target_unavailable`.
GPU tests: `tests/test_native_executor.cpp` (`test_show_target*`: colour, layers, mips, msaa, depth ranges,
pass selectors, hotkey cycling, digest equality with and without the option).

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

## Device loss

A driver reset, TDR or GPU removal loses the Dawn device. The browser worker (`wgpu_worker.js`, TM9) handles
`device.lost` with a watchdog and a leak-free teardown; the native executor mirrors the semantics and goes
one step further, it rebuilds.

- **Detect.** The device-lost callback (any reason but our own `Destroyed`) sets `lost` and counts
  `device_lost_events`. The executor reacts at the next packet, at the end of a batch, and in `poll()`,
  so a loss between batches is repaired without waiting for the guest. Nothing is submitted to a lost
  device: the existing `lost` gates stay.
- **Journal.** Every packet that creates a resource or fills one (buffers, shader modules, textures,
  samplers, layouts, pipelines v1 / v2, bind groups, `WriteBuffer`, `WriteTexture`) is copied into a
  journal. `DestroyResource` prunes that id's entries; a write of the same range replaces the older one.
  Budget: `ExecutorOptions::recovery_journal_bytes` (default 512 MiB).
- **Recover.** `recover()` fails the read-backs of the frame under construction (their fence counters
  advance as failed), destroys the old device, requests adapter + device again (the same feature / limit
  negotiation, the blob cache stays open, up to 5 attempts 250 ms apart), unconfigures the surface (it is
  reconfigured by the next present), and re-dispatches the journal in order. Pipelines are rebuilt through
  the blob cache, shader modules from the pack. The new device must offer the features the old one had.
- **What is not rebuilt.** Resources the GPU wrote itself (render targets, storage output) come back
  empty and are counted (`recovery_gpu_resources_reset`); the next frame writes them again. A frame that
  was being built when the device died is dropped (`recovery_frames_dropped`, also `present_failures`);
  the clear colour carries over. Nothing is faked: the output after recovery is what the same packets
  produce on a healthy device, a differing replay is visible as `recovery_replay_errors`.
- **Cannot rebuild, so end cleanly.** Journal disabled or over budget, no device after the retries, a
  device with fewer features, or a loss during the replay on every attempt: `faulted()` becomes true,
  `stats_json()` carries `"fault_record": {"kind": "gpu_device_lost", "reason", "device_error", counters}`,
  later batches are dropped (`dropped_while_lost`) and `poll()` returns false, which ends the live run
  the same way closing the window does. No crash.
- **Counters** (`stats_json`): `device_lost_events`, `recoveries`, `recovery_failures`, `recovery_ms`,
  `recovery_replayed_packets`, `recovery_replay_errors`, `recovery_frames_dropped`,
  `recovery_gpu_resources_reset`, `journal_bytes`, `journal_overflows`.
- **Test.** `test_native_executor` loses the device with `NativeExecutor::debug_force_device_lost`
  (Dawn `Device::ForceLoss`): buffer and texture contents compare equal after the rebuild, a pruned
  resource stays gone, a queued frame is dropped, and the no-journal / over-budget cases end with the
  fault record.

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

4. **Live vs replay (`tools/live_vs_replay.py`):** one live `playgta6_boot --live-executor` (headless; `--window` with `--show-window`) run writes a GPU
   capture (`--gpu-capture-payloads`, not thinned) and, from the live executor itself, one replay digest
   per presented frame (`--live-digest FILE`, `ExecutorOptions::frame_digest_path`: the stream image
   before the blit, hashed exactly like `pg6_native_exec --digest-at`). The script then replays the capture
   headless with `pg6_native_exec` at the live stream size and compares the digests frame by frame
   (capture frame i = live present `first_present + i`), printing the first mismatch. The live run
   exercises the paths a replay never takes (guest-memory read-backs, lazy shader compile, async pipelines
   + blob cache, the blit), so a mismatch bisects to one of them. Example (local, GPU, game files outside
   the repo; the capture is large and is deleted afterwards unless `--keep`):
   `python tools/live_vs_replay.py --exe <exe> --game-dir <dir> --pack <std.wgsl.pack> --preset menu-flow
   --capture-start 22000 --max-ms 40000`. Result on NVIDIA / D3D12 (1280x720 `rgba8unorm`, real title):
   600 captured frames starting at live present 1136 (307 distinct images, 21 k draws, 3.9 k dispatches,
   14 async pipelines, 7 first-use waits), all 600 digests identical; and the first 600 frames from boot
   (4 distinct images) identical. The blit path was not exercised (window stream size == swapchain size).
   Note: under `menu-flow` the guest itself faults at about 45 s (present ~2243, null call), so keep
   `--max-ms` below that.

5. **Window test hook (`--window-script`):** `playgta6_boot --window --window-script "4000:resize=1000x700,5000:probe,7000:minimize,10000:restore,13000:fullscreen,15000:probe,17000:windowed"` drives the live window from the executor's poll (times in ms from the first presented frame; `resize=WxH` client pixels, `minimize`, `restore`, `maximize`, `fullscreen`, `windowed`, `probe`). `probe` reads the presented canvas back after the blit and logs `window probe OK|FAIL`: the letterbox bars must be black. The live window keeps the stream at its fixed size (`fixed_stream_size`), so a resize letterboxes instead of stretching. Surface reconfiguration is the pure `SurfaceState` machine (`window_script.h`, unit tested): a zero-size client area (minimized) skips the present and keeps the configuration, a size change reconfigures, Outdated / Lost retry once. Stats: `surface_configures`, `surface_minimized_skips`, `surface_acquire_retries`, `window_actions`, `probes`, `probes_failed`. NVIDIA / D3D12 run: resize to 1000x700 and 700x900, minimize 3 s, restore, fullscreen, windowed and maximize, no device error, bars black on every probe.

## Shader quarantine (diagnostic)

`--skip-shader <hash>[,<hash>]`, `--skip-shader-file <path>` and the repo list `runtime/gpu/native_exec/shader_quarantine.txt`
(empty by default; format `<hash> <reason> <issue link>`, `--no-quarantine-list` ignores it) make the executor skip every
draw and dispatch (indirect ones too) of the listed shaders, so a program that hangs or crashes the GPU can be bisected
without a rebuild. Each skip is counted per phase in the boot report's drop table as `quarantined` with the hash; no
output is substituted. Hair-strand skip-by-name is not possible on the native side: the executor only sees shader
hashes, not program names (use `tools/program_census.py` to map names to hashes, then list the hash).
