# GPU Command Stream & Shader Index (`runtime/gpu` ↔ `web/workers/wgpu_worker.js`)

## Command Packet Layout
Little-endian binary packets matching `playgta6::GpuPacketHeader` (`runtime/abi/playgta6_abi.h`, 8 bytes) followed by `payload_bytes`:

| Offset | Type | Field |
| :--- | :--- | :--- |
| `0..2` | `u16` | `opcode` (`GpuOpcode`) |
| `2..4` | `u16` | `flags` |
| `4..8` | `u32` | `payload_bytes` |
| `8..8+N` | `[u8; N]` | Opcode payload |

### Opcodes (v1)
All multi-byte integers and IEEE-754 floats are **little-endian**. Every struct below matches its exact `static_assert` size in `runtime/abi/playgta6_abi.h`.

| Opcode | Name | Payload Size | Payload Layout |
| :--- | :--- | :--- | :--- |
| `0` | `Nop` | `0` | Empty |
| `1` | `BeginFrame` | `0` | Empty — resets per-frame clear color and queued render-pass commands |
| `2` | `ClearColor` | `16` | `{ f32 r, f32 g, f32 b, f32 a }` (`GpuClearColorPayload`) |
| `3` | `CreateBuffer` | `16` | `{ u32 id, u32 usage_flags, u64 byte_size }` (`GpuCreateBufferPayload`) |
| `4` | `WriteBuffer` | `16 + data_len` | `{ u32 id, u32 dst_offset, u32 data_len, u32 reserved }` (`GpuWriteBufferPayload`) followed by `data_len` raw bytes |
| `5` | `CreateShaderModule` | `16 + code_len` | `{ u32 id, u32 stage, u32 code_len, u32 reserved }` (`GpuCreateShaderModulePayload`) followed by `code_len` UTF-8 WGSL bytes |
| `6` | `Draw` | `16` | `{ u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance }` (`GpuDrawPayload`) |
| `7` | `Present` | `0` | Empty — encodes and submits the current frame's WebGPU passes (`ClearColor` + render/compute passes) |
| `8` | `CreateRenderPipeline` | `24` | `{ u32 id, u32 vs_shader_id, u32 fs_shader_id, u32 vertex_stride, u32 topology, u32 flags }` (`GpuCreateRenderPipelinePayload`) |
| `9` | `SetPipeline` | `4` | `{ u32 pipeline_id }` (`GpuSetPipelinePayload`) |
| `10` | `SetVertexBuffer` | `16` | `{ u32 slot, u32 buffer_id, u64 offset }` (`GpuSetVertexBufferPayload`) |
| `11` | `DestroyResource` | `4` | `{ u32 id }` (`GpuDestroyResourcePayload`) — destroys buffer/texture (if present) and removes `id` from all resource tables |
| `12` | `CreateTexture` | `32` | `{ u32 id, u32 width, u32 height, u32 depth_or_layers, u32 mip_levels, u32 format, u32 usage_flags, u32 dimension }` (`GpuCreateTexturePayload`) |
| `13` | `WriteTexture` | `48 + data_len` | `{ u32 texture_id, u32 mip_level, u32 origin_x, u32 origin_y, u32 origin_z, u32 width, u32 height, u32 depth_or_layers, u32 bytes_per_row, u32 rows_per_image, u32 data_len, u32 reserved }` (`GpuWriteTexturePayload`) followed by `data_len` raw texel bytes |
| `14` | `CreateSampler` | `32` | `{ u32 id, u32 address_u, u32 address_v, u32 address_w, u32 mag_filter, u32 min_filter, u32 mipmap_filter, u32 max_anisotropy }` (`GpuCreateSamplerPayload`) |
| `15` | `CreateBindGroup` | `16 + 32 * entry_count` | `{ u32 id, u32 pipeline_id, u32 group_index, u32 entry_count }` (`GpuCreateBindGroupPayload`) followed by `entry_count` packed `GpuBindGroupEntry` structs (`{ u32 binding, u32 resource_type, u32 resource_id, u32 reserved, u64 offset, u64 size }`, 32 bytes each; for `Buffer` entries, `size == 0` binds from `offset` to buffer end) |
| `16` | `SetBindGroup` | `8` | `{ u32 group_index, u32 bind_group_id }` (`GpuSetBindGroupPayload`) — binds to the currently active render or compute pass |
| `17` | `SetIndexBuffer` | `16` | `{ u32 buffer_id, u32 index_format, u64 offset }` (`GpuSetIndexBufferPayload`) |
| `18` | `DrawIndexed` | `20` | `{ u32 index_count, u32 instance_count, u32 first_index, i32 base_vertex, u32 first_instance }` (`GpuDrawIndexedPayload`) |
| `19` | `CreateComputePipeline` | `16` | `{ u32 id, u32 cs_shader_id, u32 flags, u32 reserved }` (`GpuCreateComputePipelinePayload`) |
| `20` | `SetComputePipeline` | `4` | `{ u32 pipeline_id }` (`GpuSetComputePipelinePayload`) — ends any open render pass and begins a compute pass |
| `21` | `DispatchCompute` | `12` | `{ u32 workgroup_x, u32 workgroup_y, u32 workgroup_z }` (`GpuDispatchComputePayload`) |

### Enum Values

#### `usage_flags` (`GpuBufferUsage`, matches WebGPU `GPUBufferUsage`)
- `0x0001`: `MAP_READ`
- `0x0002`: `MAP_WRITE`
- `0x0004`: `COPY_SRC`
- `0x0008`: `COPY_DST`
- `0x0010`: `INDEX`
- `0x0020`: `VERTEX`
- `0x0040`: `UNIFORM`
- `0x0080`: `STORAGE`
- `0x0100`: `INDIRECT`
- `0x0200`: `QUERY_RESOLVE`

#### `usage_flags` (`GpuTextureUsage`, matches WebGPU `GPUTextureUsage`)
- `0x0001`: `COPY_SRC`
- `0x0002`: `COPY_DST`
- `0x0004`: `TEXTURE_BINDING`
- `0x0008`: `STORAGE_BINDING`
- `0x0010`: `RENDER_ATTACHMENT`

#### `format` (`GpuTextureFormat`)
- `0`: `Rgba8Unorm` (`"rgba8unorm"`)
- `1`: `Bgra8Unorm` (`"bgra8unorm"`)
- `2`: `Rgba16Float` (`"rgba16float"`)
- `3`: `Depth24Plus` (`"depth24plus"`)
- `4`: `Depth32Float` (`"depth32float"`)
- `5`: `Bc7RgbaUnorm` (`"bc7-rgba-unorm"`)

#### `dimension` (`GpuTextureDimension`)
- `0`: `Dim1D` (`"1d"`)
- `1`: `Dim2D` (`"2d"`)
- `2`: `Dim3D` (`"3d"`)

#### `address_u` / `address_v` / `address_w` (`GpuAddressMode`)
- `0`: `ClampToEdge` (`"clamp-to-edge"`)
- `1`: `Repeat` (`"repeat"`)
- `2`: `MirrorRepeat` (`"mirror-repeat"`)

#### `mag_filter` / `min_filter` / `mipmap_filter` (`GpuFilterMode`)
- `0`: `Nearest` (`"nearest"`)
- `1`: `Linear` (`"linear"`)

#### `resource_type` (`GpuBindResourceType`)
- `0`: `Buffer` (`GPUBufferBinding { buffer, offset, size? }`)
- `1`: `TextureView` (`GPUTextureView` created from texture `resource_id`)
- `2`: `Sampler` (`GPUSampler`)

#### `index_format` (`GpuIndexFormat`)
- `0`: `Uint16` (`"uint16"`)
- `1`: `Uint32` (`"uint32"`)

#### `stage` (`GpuShaderStage`)
- `0`: `Vertex` (`vs_main`)
- `1`: `Fragment` (`fs_main`)
- `2`: `Compute` (`cs_main`)

#### `topology` (`GpuPrimitiveTopology`)
- `0`: `TriangleList` (`"triangle-list"`)
- `1`: `TriangleStrip` (`"triangle-strip"`)
- `2`: `LineList` (`"line-list"`)
- `3`: `LineStrip` (`"line-strip"`)
- `4`: `PointList` (`"point-list"`)

### Pass Switching Semantics
Within a frame (between `BeginFrame` and `Present`), the WebGPU worker lazily opens render and compute passes on a single `GPUCommandEncoder`:
- Any compute command (`SetComputePipeline`, `DispatchCompute`) ends an open render pass before calling `beginComputePass()`.
- Any render command (`SetPipeline`, `SetVertexBuffer`, `SetIndexBuffer`, `Draw`, `DrawIndexed`) ends an open compute pass before calling `beginRenderPass()`. The first render pass in a frame uses `loadOp: "clear"` with the active `ClearColor`; any resumed render pass in the same frame uses `loadOp: "load"` to preserve prior drawing.
- `Present` ends any open pass (opening a clearing render pass if no render pass ran yet) and submits the finished `GPUCommandBuffer`.

## Opcodes (v2)
v2 adds explicit bind group / pipeline layouts, full render pipeline keys, explicit render passes and dynamic render state. It is **additive**: opcodes `0..21` and their payloads keep their exact v1 meaning, a stream may mix v1 and v2 packets, and new opcodes are only ever appended (a consumer skips opcodes it does not know by `payload_bytes`). `kAbiVersion` (the Wasm ABI) is unchanged.

| Opcode | Name | Payload Size | Payload Layout |
| :--- | :--- | :--- | :--- |
| `22` | `CreateBindGroupLayout` | `24 + 24 * entry_count` | `{ u32 id, u32 entry_count, u32 flags, u32 reserved, u64 content_id }` (`GpuCreateBindGroupLayoutPayload`) followed by `entry_count` `GpuBindGroupLayoutEntry` (24 bytes each) |
| `23` | `CreatePipelineLayout` | `24 + 4 * bind_group_layout_count` | `{ u32 id, u32 bind_group_layout_count, u32 flags, u32 reserved, u64 content_id }` (`GpuCreatePipelineLayoutPayload`) followed by `bind_group_layout_count` `u32` bind group layout ids for groups `0..n-1` (`0` = empty layout) |
| `24` | `CreateRenderPipeline2` | `392` | `{ u32 id, u32 vs_module_id, u32 fs_module_id, u32 pipeline_layout_id, GpuRenderPipelineKey key }` (`GpuCreateRenderPipeline2Payload`, key at offset 16) |
| `25` | `CreateComputePipeline2` | `32` | `{ u32 id, u32 cs_module_id, u32 pipeline_layout_id, u32 reserved, GpuComputePipelineKey key }` (`GpuCreateComputePipeline2Payload`) |
| `26` | `BeginRenderPass` | `368` | `{ u32 color_count, u32 reserved, GpuRenderPassColorAttachment color[8], GpuRenderPassDepthStencilAttachment depth_stencil }` (`GpuBeginRenderPassPayload`, `depth_stencil` at offset 328) |
| `27` | `EndRenderPass` | `0` | Empty |
| `28` | `SetViewport` | `24` | `{ f32 x, f32 y, f32 width, f32 height, f32 min_depth, f32 max_depth }` (`GpuSetViewportPayload`) |
| `29` | `SetScissorRect` | `16` | `{ u32 x, u32 y, u32 width, u32 height }` (`GpuSetScissorRectPayload`) |
| `30` | `SetBlendConstant` | `16` | `{ f32 r, f32 g, f32 b, f32 a }` (`GpuSetBlendConstantPayload`) |
| `31` | `SetStencilReference` | `4` | `{ u32 reference }` (`GpuSetStencilReferencePayload`) |
| `32` | `CreateBindGroup2` | `16 + 48 * entry_count` | `{ u32 id, u32 layout_id, u32 entry_count, u32 flags }` (`GpuCreateBindGroup2Payload`; `flags & BindingArrays (0x1)`: entries hold binding-array runs, set exactly when the layout has `GpuLayoutFlags::BindingArrays`) followed by `entry_count` `GpuBindGroupEntry2` (48 bytes each) |
| `33` | `SetBindGroup2` | `16 + 4 * dynamic_offset_count` | `{ u32 group_index, u32 bind_group_id, u32 dynamic_offset_count, u32 reserved }` (`GpuSetBindGroup2Payload`) followed by `dynamic_offset_count` `u32` dynamic offsets |
| `34` | `CreateSampler2` | `24` | `{ u32 id, u8 address_u, u8 address_v, u8 address_w, u8 mag_filter, u8 min_filter, u8 mipmap_filter, u8 compare, u8 reserved0, u16 max_anisotropy, u16 reserved1, f32 lod_min_clamp, f32 lod_max_clamp }` (`GpuCreateSampler2Payload`) |
| `35` | `CreateTexture2` | `40` | `{ u32 id, u32 width, u32 height, u32 depth_or_layers, u32 mip_levels, u32 sample_count, u32 usage_flags, u8 dimension, u8 format, u8 view_format_count, u8 reserved, u8 view_formats[8] }` (`GpuCreateTexture2Payload`) |
| `36` | `DrawIndirect` | `16` | `{ u32 buffer_id, u32 reserved, u64 offset }` (`GpuIndirectPayload`): `drawIndirect(buffer, offset)`, a 16-byte `{ vertex_count, instance_count, first_vertex, first_instance }` record |
| `37` | `DrawIndexedIndirect` | `16` | `GpuIndirectPayload`: `drawIndexedIndirect(buffer, offset)`, a 20-byte `{ index_count, instance_count, first_index, base_vertex, first_instance }` record |
| `38` | `DispatchIndirect` | `16` | `GpuIndirectPayload`: `dispatchWorkgroupsIndirect(buffer, offset)`, a 12-byte `{ x, y, z }` record |
| `39` | `ReadbackBuffer` | `40` | `{ u32 seq, u32 buffer_id, u64 src_offset, u64 size, u64 dst_addr, u64 done_addr }` (`GpuReadbackBufferPayload`) |
| `40` | `ReadbackTexture` | `72` | `{ u32 seq, u32 texture_id, u32 mip_level, u32 origin_x, u32 origin_y, u32 origin_z, u32 width, u32 height, u32 depth_or_layers, u32 bytes_per_row, u32 rows_per_image, u32 reserved, u64 dst_addr, u64 done_addr, u64 dst_size }` (`GpuReadbackTexturePayload`) |

### Ids
- All ids share the v1 `u32` resource id namespace (`0` = none). `DestroyResource` also destroys bind group layouts, pipeline layouts and v2 bind groups, samplers and pipelines.
- `kGpuSwapchainTextureId = 0xFFFFFFFF` names the canvas texture of the current frame (`getCurrentTexture()`) wherever a texture id is expected.
- **Layout ids.** S6 (`shaders/src/reflect.rs`, `layout-manifest.json`) identifies layouts by 64-bit content ids (`"0x"` + 16 hex digits, FNV-1a 64). Packets reference layouts by a `u32` runtime id; `content_id` carries the S6 id parsed as a `u64` (`0` when the layout did not come from the manifest). The producer allocates one runtime id per distinct `content_id` and never reuses it for another `content_id`. The worker keeps `content_id -> layout` so a layout may come from a packet or from the manifest:
  - `flags = 0`: the packet carries the entries; the worker creates the layout from them (and may reuse an identical object registered under the same non-zero `content_id`).
  - `flags & ManifestReference (0x1)`: the packet carries no entries (`entry_count` / `bind_group_layout_count` must be `0`); the worker binds `id` to the layout it loaded from `layout-manifest.json` under `content_id`.
  - `CreateBindGroupLayout` `flags & BindingArrays (0x2)`: some entries are binding arrays (`binding_array_size >= 1`). Emitted only when the producer's device capability `BridgeOptions::binding_arrays` is set (the consumer supports `GPUBindGroupLayoutEntry.bindingArraySize`); off by default, a consumer that did not opt in never sees such a layout.

### `GpuBindGroupLayoutEntry` (24 bytes, one `GPUBindGroupLayoutEntry`)
Same information as an S6 manifest entry, so manifest JSON and packets convert 1:1.

| Offset | Type | Field |
| :--- | :--- | :--- |
| `0` | `u32` | `binding` |
| `4` | `u32` | `visibility` (`GPUShaderStage` bits: vertex `0x1`, fragment `0x2`, compute `0x4`) |
| `8` | `u8` | `type` (`GpuBindingLayoutType`): `0` `buffer`, `1` `sampler`, `2` `texture`, `3` `storageTexture` |
| `9` | `u8` | `binding_type`, by `type`: buffer `type` (`0` `"uniform"`, `1` `"storage"`, `2` `"read-only-storage"`); sampler `type` (`0` `"filtering"`, `1` `"non-filtering"`, `2` `"comparison"`); texture `sampleType` (`0` `"float"`, `1` `"unfilterable-float"`, `2` `"depth"`, `3` `"sint"`, `4` `"uint"`); storageTexture `access` (`0` `"write-only"`, `1` `"read-only"`, `2` `"read-write"`) |
| `10` | `u8` | `view_dimension` (`GpuTextureViewDimension`, texture / storageTexture; `0` = `"2d"`) |
| `11` | `u8` | `flags`: `0x1` buffer `hasDynamicOffset`, `0x2` texture `multisampled` |
| `12` | `u8` | `storage_format` (storageTexture: texture format code) |
| `13` | `u8` | reserved (0) |
| `14` | `u16` | `binding_array_size`: `0` = not an array (the former reserved bytes); `N >= 1` = `GPUBindGroupLayoutEntry.bindingArraySize` N (WGSL `binding_array<T, N>`), only in layouts with `BindingArrays`; `N <= kGpuMaxBindingArraySize` (256) |
| `16` | `u64` | `min_binding_size` (buffer `minBindingSize`) |

### `GpuTextureViewDesc` (16 bytes, a `GPUTextureViewDescriptor`)
| Offset | Type | Field |
| :--- | :--- | :--- |
| `0` | `u8` | `format` (texture format code; `0` = the texture's format) |
| `1` | `u8` | `dimension` (`GpuTextureViewDimension`): `0` default, `1` `"1d"`, `2` `"2d"`, `3` `"2d-array"`, `4` `"cube"`, `5` `"cube-array"`, `6` `"3d"` |
| `2` | `u8` | `aspect` (`GpuTextureAspect`): `0` `"all"`, `1` `"depth-only"`, `2` `"stencil-only"` |
| `3` | `u8` | reserved (0) |
| `4` | `u16` | `base_mip_level` |
| `6` | `u16` | `mip_level_count` (`0` = all remaining) |
| `8` | `u32` | `base_array_layer` |
| `12` | `u32` | `array_layer_count` (`0` = all remaining) |

An all-zero descriptor is the texture's default view.

### `GpuBindGroupEntry2` (48 bytes)
The v1 `GpuBindGroupEntry` (offsets `0..32`: `binding`, `resource_type`, `resource_id`, `reserved`, `u64 offset`, `u64 size`) followed by a `GpuTextureViewDesc view` at offset `32` that describes the view for `TextureView` entries (sampled and storage textures). `resource_id == 0` is a null binding (D3D12 null descriptor): the worker binds a zero-filled resource compatible with the layout entry (buffer / 1×1 texture of the entry's sample type and view dimension / default sampler).

Offset `12` (v1 `reserved`) is `u16 array_index` + `u16 reserved`. A binding array of `N` elements is a run of `N` consecutive entries with the same `binding` and `array_index` `0..N-1` (element `i` is `GPUBindGroupEntry` `binding + i` in WebGPU's binding-array numbering); each element may be null (`resource_id == 0`). `entry_count` counts every element. Outside binding arrays `array_index` is `0`.

### `CreateRenderPipeline2` and the render pipeline key
`vs_module_id` / `fs_module_id` are `CreateShaderModule` ids (entry points `vs_main` / `fs_main` as in v1; `fs_module_id == 0` = no fragment shader, in which case the worker binds an empty fragment stage when `color_target_count > 0`). `pipeline_layout_id` is a `CreatePipelineLayout` id (`0` = `layout: "auto"`).

`key` is `GpuRenderPipelineKey`, which is **byte-for-byte** `playgta6::gpu::pso::GraphicsPipelineKey` (`runtime/gpu/pso_cache.h`, pinned by `static_assert`s in `tests/test_runtime_abi.cpp`). The key is canonical: reserved bytes and every field of a null color target / absent depth-stencil state are zero. It maps to `GPURenderPipelineDescriptor` as follows (offsets from the start of the key):

| Offset | Type | Field | `GPURenderPipelineDescriptor` |
| :--- | :--- | :--- | :--- |
| `0` | `u64` | `vs_shader` | informational shader identity |
| `8` | `u64` | `ps_shader` | informational (`0` = no pixel shader) |
| `16` | `u32` | `root_signature_id` | informational |
| `20` | `u32` | `color_target_count` | `fragment.targets.length` |
| `24 + 12*i` | `GpuColorTargetKey` | `color_targets[i]`, `i < 8` | `fragment.targets[i]` |
| `120` | `GpuDepthStencilKey` | `depth_stencil` | `depthStencil` |
| `152` | `GpuPrimitiveKey` | `primitive` | `primitive` |
| `160` | `GpuMultisampleKey` | `multisample` | `multisample` |
| `172` | `u32` | `vertex_buffer_count` | `vertex.buffers.length` |
| `176` | `u32` | `vertex_attribute_count` | number of used `vertex_attributes` |
| `180 + 8*i` | `GpuVertexBufferKey` | `vertex_buffers[i]`, `i < 8` | `vertex.buffers[i]` |
| `244 + 8*i` | `GpuVertexAttributeKey` | `vertex_attributes[i]`, `i < 16` | attributes |
| `372` | `u32` | `reserved` | — |

- `GpuColorTargetKey` (12 bytes): `+0 u8 format` (texture format code; `0` = `null` target), `+1 u8 blend_enabled` (`0` = omit `blend`), `+2 u8 write_mask` (`GPUColorWrite` bits), `+3 u8 reserved`, `+4 GpuBlendComponentKey color`, `+8 GpuBlendComponentKey alpha`.
- `GpuBlendComponentKey` (4 bytes): `+0 u8 operation` (blend operation code), `+1 u8 src_factor`, `+2 u8 dst_factor` (blend factor codes), `+3 u8 reserved`.
- `GpuDepthStencilKey` (32 bytes): `+0 u8 format` (texture format code; `0` = omit `depthStencil`), `+1 u8 depth_write_enabled`, `+2 u8 depth_compare` (compare code, `0` = omit), `+3 u8 reserved`, `+4 GpuStencilFaceKey stencil_front`, `+8 GpuStencilFaceKey stencil_back`, `+12 u32 stencil_read_mask`, `+16 u32 stencil_write_mask`, `+20 i32 depth_bias`, `+24 u32 depth_bias_slope_scale` (IEEE-754 `f32` bits), `+28 u32 depth_bias_clamp` (`f32` bits).
- `GpuStencilFaceKey` (4 bytes): `+0 u8 compare` (compare code), `+1 u8 fail_op`, `+2 u8 depth_fail_op`, `+3 u8 pass_op` (stencil operation codes).
- `GpuPrimitiveKey` (8 bytes): `+0 u8 topology` (`GpuPrimitiveTopology`), `+1 u8 strip_index_format` (index format code, `0` = omit), `+2 u8 front_face`, `+3 u8 cull_mode`, `+4 u8 unclipped_depth`, `+5 u8[3] reserved`.
- `GpuMultisampleKey` (12 bytes): `+0 u32 count` (`1` or `4`), `+4 u32 mask`, `+8 u8 alpha_to_coverage_enabled`, `+9 u8[3] reserved`.
- `GpuVertexBufferKey` (8 bytes): `+0 u32 array_stride`, `+4 u8 step_mode` (`0` = unused slot, emitted as `null`; `1` `"vertex"`, `2` `"instance"`), `+5 u8 attribute_count`, `+6 u8[2] reserved`. Buffer `i` owns the next `attribute_count` entries of `vertex_attributes` (attributes are sorted by `(buffer_index, shader_location)`).
- `GpuVertexAttributeKey` (8 bytes): `+0 u32 offset`, `+4 u8 format` (vertex format code), `+5 u8 shader_location`, `+6 u8 buffer_index`, `+7 u8 reserved`.

`CreateComputePipeline2.key` is `GpuComputePipelineKey` = byte-for-byte `pso::ComputePipelineKey`: `{ u64 cs_shader, u32 root_signature_id, u32 reserved }` (informational; the pipeline is `cs_module_id` with entry point `cs_main` and `pipeline_layout_id`).

### WebGPU enum codes
Byte codes used inside the v2 payloads (the numbering of the `pso::Wgpu*` enums in `runtime/gpu/pso_cache.h`; frozen by this document):
- **Texture format:** `0` `undefined`, `1` `r8unorm`, `2` `r8snorm`, `3` `r8uint`, `4` `r8sint`, `5` `r16unorm`, `6` `r16snorm`, `7` `r16uint`, `8` `r16sint`, `9` `r16float`, `10` `rg8unorm`, `11` `rg8snorm`, `12` `rg8uint`, `13` `rg8sint`, `14` `r32uint`, `15` `r32sint`, `16` `r32float`, `17` `rg16unorm`, `18` `rg16snorm`, `19` `rg16uint`, `20` `rg16sint`, `21` `rg16float`, `22` `rgba8unorm`, `23` `rgba8unorm-srgb`, `24` `rgba8snorm`, `25` `rgba8uint`, `26` `rgba8sint`, `27` `bgra8unorm`, `28` `bgra8unorm-srgb`, `29` `rgb9e5ufloat`, `30` `rgb10a2uint`, `31` `rgb10a2unorm`, `32` `rg11b10ufloat`, `33` `rg32uint`, `34` `rg32sint`, `35` `rg32float`, `36` `rgba16unorm`, `37` `rgba16snorm`, `38` `rgba16uint`, `39` `rgba16sint`, `40` `rgba16float`, `41` `rgba32uint`, `42` `rgba32sint`, `43` `rgba32float`, `44` `stencil8`, `45` `depth16unorm`, `46` `depth24plus`, `47` `depth24plus-stencil8`, `48` `depth32float`, `49` `depth32float-stencil8`, `50` `bc1-rgba-unorm`, `51` `bc1-rgba-unorm-srgb`, `52` `bc2-rgba-unorm`, `53` `bc2-rgba-unorm-srgb`, `54` `bc3-rgba-unorm`, `55` `bc3-rgba-unorm-srgb`, `56` `bc4-r-unorm`, `57` `bc4-r-snorm`, `58` `bc5-rg-unorm`, `59` `bc5-rg-snorm`, `60` `bc6h-rgb-ufloat`, `61` `bc6h-rgb-float`, `62` `bc7-rgba-unorm`, `63` `bc7-rgba-unorm-srgb`. (Distinct from the v1 `GpuTextureFormat` used by `CreateTexture`.)
- **Vertex format:** `0` `undefined`, `1` `uint8`, `2` `uint8x2`, `3` `uint8x4`, `4` `sint8`, `5` `sint8x2`, `6` `sint8x4`, `7` `unorm8`, `8` `unorm8x2`, `9` `unorm8x4`, `10` `snorm8`, `11` `snorm8x2`, `12` `snorm8x4`, `13` `uint16`, `14` `uint16x2`, `15` `uint16x4`, `16` `sint16`, `17` `sint16x2`, `18` `sint16x4`, `19` `unorm16`, `20` `unorm16x2`, `21` `unorm16x4`, `22` `snorm16`, `23` `snorm16x2`, `24` `snorm16x4`, `25` `float16`, `26` `float16x2`, `27` `float16x4`, `28` `float32`, `29` `float32x2`, `30` `float32x3`, `31` `float32x4`, `32` `uint32`, `33` `uint32x2`, `34` `uint32x3`, `35` `uint32x4`, `36` `sint32`, `37` `sint32x2`, `38` `sint32x3`, `39` `sint32x4`, `40` `unorm10-10-10-2`, `41` `unorm8x4-bgra`.
- **Blend factor:** `0` `zero`, `1` `one`, `2` `src`, `3` `one-minus-src`, `4` `src-alpha`, `5` `one-minus-src-alpha`, `6` `dst`, `7` `one-minus-dst`, `8` `dst-alpha`, `9` `one-minus-dst-alpha`, `10` `src-alpha-saturated`, `11` `constant`, `12` `one-minus-constant`, `13` `src1`, `14` `one-minus-src1`, `15` `src1-alpha`, `16` `one-minus-src1-alpha`.
- **Blend operation:** `0` `add`, `1` `subtract`, `2` `reverse-subtract`, `3` `min`, `4` `max`.
- **Compare function:** `0` `undefined`, `1` `never`, `2` `less`, `3` `equal`, `4` `less-equal`, `5` `greater`, `6` `not-equal`, `7` `greater-equal`, `8` `always`.
- **Stencil operation:** `0` `keep`, `1` `zero`, `2` `replace`, `3` `invert`, `4` `increment-clamp`, `5` `decrement-clamp`, `6` `increment-wrap`, `7` `decrement-wrap`.
- **Cull mode:** `0` `none`, `1` `front`, `2` `back`. **Front face:** `0` `ccw`, `1` `cw`.
- **Index format (key):** `0` `undefined`, `1` `uint16`, `2` `uint32` (unlike v1 `GpuIndexFormat` of `SetIndexBuffer`).
- **Load op** (`GpuLoadOp`): `0` undefined, `1` `load`, `2` `clear`. **Store op** (`GpuStoreOp`): `0` undefined, `1` `store`, `2` `discard`.

### `BeginRenderPass` attachments
- `GpuRenderPassColorAttachment` (40 bytes): `+0 u32 texture_id` (`0` = `null` attachment at this index), `+4 u8 load_op`, `+5 u8 store_op`, `+6 u16 reserved`, `+8 GpuTextureViewDesc view`, `+24 f32 clear_value[4]` (RGBA, used with `load_op == clear`). Only `color[0..color_count)` are read.
- `GpuRenderPassDepthStencilAttachment` (40 bytes): `+0 u32 texture_id` (`0` = no `depthStencilAttachment`), `+4 u8 depth_load_op`, `+5 u8 depth_store_op`, `+6 u8 stencil_load_op`, `+7 u8 stencil_store_op` (each `0` = omitted, required when that aspect is read-only or absent from the format), `+8 GpuTextureViewDesc view`, `+24 f32 depth_clear_value`, `+28 u32 stencil_clear_value`, `+32 u8 depth_read_only`, `+33 u8 stencil_read_only`, `+34 u16 reserved`, `+36 u32 reserved`.

### `CreateSampler2`
`address_*` are v1 `GpuAddressMode`, `*_filter` are v1 `GpuFilterMode`, `compare` is a compare function code (`0` = not a comparison sampler), `max_anisotropy >= 1`. Superset of v1 `CreateSampler` (adds `compare` and LOD clamps).

### `CreateTexture2`
A `GPUTextureDescriptor` for textures the v1 `CreateTexture` cannot describe (D3D12 resources created by the `d3d12_xs` layer). `format` and `view_formats` are v2 texture format codes ("WebGPU enum codes" below; `0` is invalid for `format`), `dimension` is `GpuTextureDimension`, `usage_flags` is `GpuTextureUsage`, `sample_count` is `1` or `4` (`0` is read as `1`), `depth_or_layers` is the depth of a 3D texture and the array layer count otherwise. `viewFormats` is `view_formats[0..view_format_count)` (`view_format_count <= 8`; the remaining entries are `0`). The texture lives in the same `u32` id namespace as v1 textures: `WriteTexture`, `DestroyResource`, bind group entries and render pass attachments accept the id, and the default view is created the same way as for `CreateTexture`. Re-creating an existing id replaces the texture. A consumer that cannot create the texture (unsupported format or feature) reports it and leaves the id unbound.

### Read-back (`ReadbackBuffer`, `ReadbackTexture`)
GPU -> CPU copies for memory the title reads on the CPU (readback-heap buffers, texture regions copied into a placed footprint). The consumer ends any open pass, copies the source (buffer range, or texture region at `mip_level` / origin with the given size) into a `MAP_READ` staging buffer in stream order, maps it, and writes the bytes to `dst_addr` in the producer's memory (wasm linear-memory offset; texture rows land `bytes_per_row` apart, `rows_per_image` block rows per image, `dst_size` bytes in all). After the bytes are written it does `Atomics.add(u32 at done_addr, 1)`, also when the copy failed. The producer counts issued read-backs and holds every queue fence signal issued after one until the counter has caught up, so the title never sees the fence complete before the data is in its memory. `seq` is the 1-based issue number, for diagnostics only.

### `ClearTexture` (opcode `63`, native executor extension: `kGpuOpcodeClearTexture`, not in `GpuOpcode`)
Payload `{ u32 texture_id, u32 reserved }` (8 bytes, `GpuClearTexturePayload`): every texel of every mip and layer / slice of a single-sample color `COPY_DST` texture becomes zero, so a recycled texture (`runtime/gpu/small_texture_pool.h`) reads exactly like a new one. Earlier `WriteTexture`s of the texture are wiped, later ones land on the zeroed texture; like a write it is not ordered against passes recorded earlier in the frame, so the producer clears only textures the frame being recorded has not used. The native executor encodes copies from a shared zero buffer at the start of the next encoder it submits and lets later writes that cover whole rows / slices shrink the copies (`runtime/gpu/native_exec/clear_model.h`). The producer emits it only when `gpu_texture_clear()` is on (the live native executor without `--gpu-capture`); otherwise it zero-fills with CPU `WriteTexture`s. The browser worker does not implement it yet (a port would do the same: `copyBufferToTexture` from a zero buffer, in a command encoder submitted before the next upload of the texture).

### `CopyTexture` (opcode `62`, native executor extension: `kGpuOpcodeCopyTexture`, not in `GpuOpcode`)
Payload `GpuCopyTexturePayload` (56 bytes: `u32 dst_id, src_id, dst_mip, src_mip, dst_x, dst_y, dst_z, src_x, src_y, src_z, width, height, depth_or_layers, reserved`): WebGPU `copyTextureToTexture` of that region (`*_z` is the array layer of a 2D texture and the depth slice of a 3D one). It serves D3D12 `CopyTextureRegion` / `CopyResource` between textures whose destination is neither a render target nor a UAV (the render-blit and compute-storage routes cannot write it), which used to be skipped with "GPU texture-to-texture copy requires render-attachment or storage destination". The producer emits it only for same-dimension (2D or 3D), single-sample, copy-compatible (same format up to sRGB-ness) non-swapchain textures, with a depth/stencil texture copied whole, and only when `gpu_texture_copy()` is on (the live native executor); otherwise the copy is still skipped and counted. The consumer ends any open pass and encodes the copy in stream order. It creates every texture with `CopySrc` so any texture can be a source. The browser worker does not implement it yet (a port: add `COPY_SRC` to every created texture, end the pass, `encoder.copyTextureToTexture(...)`).

### Command-list slot 101 (unidentified, kept an E_NOTIMPL stub)
A late browser boot (issue 663) logged `unimplemented ID3D12GraphicsCommandList slot 101` (`[vtable+0x328]`). The method is not identified, and the evidence says the hit is not a missing command-list method:
- Public headers: `ID3D12GraphicsCommandList` through `...List10` have 86 slots (0..85), so no public method has index 101. The guest vtable (`runtime/gpu/d3d12_guest_command_list.cpp`) follows the Xbox order, which differs from public after slot 27 (`RSSetShadingRate` 126 vs public 77, `Barrier` 146 vs 80), so public indices cannot be mapped across anyway.
- Call sites (static scan of the exe, read-only): 27 `call/jmp [reg+0x328]` sites in 22 functions; no `mov reg,[reg+0x328]` + indirect-call forms; no such sites in the bundled DLLs (FidelityFX, Bink2, BVH builder). None acts on a command list: the receivers are gameplay objects with live fields at +0x20..+0x1250 (a command list's guest payload is zero there), a 240-slot serializer interface, handle-table entities, and one is the device (a global the device-creation function fills; device slot 101 = `CreateStateObject`, implemented). Arg shapes vary per site (`(this)`, `(this, ptr)`, `(this, u32, float, u8, u32, ptr)`, `(this, 2, 0|1)`, a handle-returning `(this, u32, ptr)`), as expected for unrelated classes.
- So a hit means the title called a title-class virtual on a pointer that was one of our command lists (stale or confused object), and the fix is upstream of the vtable. The stub's log line carries `this` and the return address; the call sites' return addresses are `0x140123e76 0x140123f48 0x140dac6ba 0x140dadd10 0x141c78bb2 0x1424813f5 0x1426afbbf 0x142a8bb5f 0x142d7c025 0x142f858f9 0x1436da518 0x1437d5497 0x143c47163 0x143c47210 0x14403b9d8 0x1445b6165 0x1445ba7bb 0x1445bea6f 0x1445beae0 0x1445beb1e 0x1445f5ad7 0x144a395f9 0x144a39627 0x1450a6657 0x1450a6670 0x145716902 0x145b51c3c`. Match a reported return address against this list to find the class and trace where its pointer came from.

### Explicit Pass Semantics (v2)
- `BeginRenderPass` ends any open pass (v1 lazy render pass, compute pass or explicit render pass) and begins a render pass with exactly the given attachments; `EndRenderPass` ends it. `ClearColor` only affects v1 lazy passes.
- While an explicit pass is open, `SetPipeline`, `SetVertexBuffer`, `SetIndexBuffer`, `SetBindGroup`, `SetBindGroup2`, `Draw`, `DrawIndexed` and the dynamic state packets (`28..31`) record into it. WebGPU resets viewport, scissor, blend constant, stencil reference and pipeline at every pass begin, so producers re-emit them after each `BeginRenderPass`. Dynamic state packets outside a render pass are ignored.
- A v1 render command outside an explicit pass keeps the v1 behaviour (lazy swapchain pass with `loadOp: "load"` if a render pass already ran this frame). A compute command ends an open explicit pass (v1 rule); producers should emit `EndRenderPass` first.
- `SetBindGroup2` applies to the open render or compute pass; offsets follow the layout's dynamic bindings in binding order.
- `Present` ends any open pass; the v1 clearing swapchain pass only runs when no render pass (lazy or explicit) ran in the frame.

### Mesh pipelines (G66: mesh shaders as compute + draw)
There is no mesh opcode. A D3D12 mesh pipeline state is two ordinary pipelines (`create_mesh_pso`, `runtime/gpu/d3d12_guest_resources.cpp`; `bridge::MeshPipelineState`, `runtime/gpu/state_bridge.h`), and `DispatchMesh(x, 1, 1)` becomes a `DispatchIndirect` + `DrawIndirect` pair (plain `Dispatch` + `Draw` when not clamped), so every consumer needs only the v2 packets above.
- **Mesh stage**: the translated mesh shader is a WGSL compute module, pack stage `compute`, pack key `bridge::mesh_compute_pack_key(mesh_key)` (`mesh_key ^ 0x4d53435300000000`, "MSCS"), entry point `cs_main`, one workgroup per mesh group (`workgroup_id.x` is the group index; `x <= mesh_max_groups`, `y == z == 1`). Its pipeline layout is the stage's own groups `0..2` (bound from the graphics root arguments) plus group `3` (`kMeshOutputGroup`), binding `0`: the runtime output buffer (storage, `read_write`; 32 MiB, written from offset `0`, GPU-ordered).
- **Output layout** (reflection `mesh_output`: `max_vertices` 1..256, `max_primitives` 1..256, `vertex_words = 4 + 4n`, `n <= 15`): one record per group of `4 + max_vertices * vertex_words + max_primitives * 3` u32 words: word 0 vertex count, word 1 primitive count (`SetMeshOutputCounts`), words 2..3 zero, then `max_vertices` vertices (SV_Position xyzw as f32 bits, then one `vec4<f32>` per attribute location `0..n-1`), then `max_primitives` triangles of three u32 vertex indices local to the group.
- **Draw**: the pixel shader is the title's; the vertex shader is the runtime's `bridge::mesh_draw_vertex_wgsl(layout)` (entry `vs_main`, group `3` binding `0` read-only storage). It draws `groups * max_primitives * 3` vertices as a triangle list whatever topology the list set; a primitive past the group's count, with an index `>= vertex count`, or whose record lies outside the buffer collapses to a point outside the depth range (no fragments).
- Coverage: `tests/test_native_executor.cpp` `test_mesh_pipeline` runs a synthetic MSCS shader on the real Dawn device (writes a triangle's records, draws it, reads back pixels); packets and reflection are covered by `test_state_bridge` / `test_native_exec_reflection`.

### Deferred (not in v2)
- Partial-rect clears (D3D12 `Clear*View` with rects, G7 `ClearFallback`) have no packet yet; they need a worker-side clear-draw helper.
- Copies, resolves, queries and indirect draws (textures with v2 format codes, sample counts and view formats: `CreateTexture2`).
- Pipeline-overridable constants and per-pipeline entry point names (entry points stay `vs_main` / `fs_main` / `cs_main`).

## `shader-index.json` (Generated by `shaders/`)
Maps Xbox shader hashes to lowered WGSL modules:

```json
{
  "abi_version": 1,
  "shaders": {
    "0xdeadbeef": {
      "stage": "compute",
      "wgsl": "shaders/deadbeef.wgsl"
    }
  }
}
```
