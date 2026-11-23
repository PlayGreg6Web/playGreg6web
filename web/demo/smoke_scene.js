// Self-contained WebGPU G5 binary command-stream smoke scene generator.
// Encodes exact little-endian GpuPacketHeader packets matching runtime/abi/playgta6_abi.h
// and docs/gpu_commands.md for execution by web/workers/wgpu_worker.js over GpuRingBuffer.
// Zero external dependencies; ES module compatible with both Browser and Node.js.

export const GPU_OPCODES = Object.freeze({
  NOP: 0,
  BEGIN_FRAME: 1,
  CLEAR_COLOR: 2,
  CREATE_BUFFER: 3,
  WRITE_BUFFER: 4,
  CREATE_SHADER_MODULE: 5,
  DRAW: 6,
  PRESENT: 7,
  CREATE_RENDER_PIPELINE: 8,
  SET_PIPELINE: 9,
  SET_VERTEX_BUFFER: 10,
  DESTROY_RESOURCE: 11,
  CREATE_TEXTURE: 12,
  WRITE_TEXTURE: 13,
  CREATE_SAMPLER: 14,
  CREATE_BIND_GROUP: 15,
  SET_BIND_GROUP: 16,
  SET_INDEX_BUFFER: 17,
  DRAW_INDEXED: 18,
  CREATE_COMPUTE_PIPELINE: 19,
  SET_COMPUTE_PIPELINE: 20,
  DISPATCH_COMPUTE: 21,
  CREATE_BIND_GROUP_LAYOUT: 22,
  CREATE_PIPELINE_LAYOUT: 23,
  CREATE_RENDER_PIPELINE2: 24,
  CREATE_COMPUTE_PIPELINE2: 25,
  BEGIN_RENDER_PASS: 26,
  END_RENDER_PASS: 27,
  SET_VIEWPORT: 28,
  SET_SCISSOR_RECT: 29,
  SET_BLEND_CONSTANT: 30,
  SET_STENCIL_REFERENCE: 31,
  CREATE_BIND_GROUP2: 32,
  SET_BIND_GROUP2: 33,
  CREATE_SAMPLER2: 34,
});

export const GPU_BUFFER_USAGE = Object.freeze({
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
});

export const GPU_TEXTURE_USAGE = Object.freeze({
  COPY_SRC: 0x0001,
  COPY_DST: 0x0002,
  TEXTURE_BINDING: 0x0004,
  STORAGE_BINDING: 0x0008,
  RENDER_ATTACHMENT: 0x0010,
});

export const GPU_TEXTURE_FORMAT = Object.freeze({
  RGBA8_UNORM: 0,
  BGRA8_UNORM: 1,
  RGBA16_FLOAT: 2,
  DEPTH24_PLUS: 3,
  DEPTH32_FLOAT: 4,
  BC7_RGBA_UNORM: 5,
});

export const GPU_TEXTURE_DIMENSION = Object.freeze({
  DIM_1D: 0,
  DIM_2D: 1,
  DIM_3D: 2,
});

export const GPU_ADDRESS_MODE = Object.freeze({
  CLAMP_TO_EDGE: 0,
  REPEAT: 1,
  MIRROR_REPEAT: 2,
});

export const GPU_FILTER_MODE = Object.freeze({
  NEAREST: 0,
  LINEAR: 1,
});

export const GPU_BIND_RESOURCE_TYPE = Object.freeze({
  BUFFER: 0,
  TEXTURE_VIEW: 1,
  SAMPLER: 2,
});

export const GPU_INDEX_FORMAT = Object.freeze({
  UINT16: 0,
  UINT32: 1,
});

export const GPU_SHADER_STAGE = Object.freeze({
  VERTEX: 0,
  FRAGMENT: 1,
  COMPUTE: 2,
});

export const GPU_PRIMITIVE_TOPOLOGY = Object.freeze({
  TRIANGLE_LIST: 0,
  TRIANGLE_STRIP: 1,
  LINE_LIST: 2,
  LINE_STRIP: 3,
  POINT_LIST: 4,
});

export const DEFAULT_RESOURCE_IDS = Object.freeze({
  TEXTURE: 1,
  SAMPLER: 2,
  VERTEX_BUFFER: 3,
  INDEX_BUFFER: 4,
  UNIFORM_BUFFER: 5,
  STORAGE_BUFFER: 6,
  VERTEX_SHADER: 7,
  FRAGMENT_SHADER: 8,
  COMPUTE_SHADER: 9,
  RENDER_PIPELINE: 10,
  COMPUTE_PIPELINE: 11,
  RENDER_BIND_GROUP: 12,
  COMPUTE_BIND_GROUP: 13,
});

export const UNIFORM_BYTE_SIZE = 64; // 4 x vec4<f32>
export const STORAGE_BYTE_SIZE = 48; // 3 x vec4<f32>
export const VERTEX_STRIDE_BYTES = 12; // vec3<f32> matching wgpu_worker.js float32x3 layout
export const DEFAULT_TEXTURE_SIZE = 64; // 64x64 RGBA8 => bytesPerRow=256, 16 KiB total

// Real WebGPU WGSL compute shader updating transform & color state in a storage buffer.
export const SMOKE_COMPUTE_WGSL = `struct SmokeUniforms {
  time_and_frame : vec4<f32>,
  input_axes     : vec4<f32>,
  tint_color     : vec4<f32>,
  extra          : vec4<f32>,
};

struct SmokeComputeState {
  transform : vec4<f32>,
  offset    : vec4<f32>,
  color_mod : vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : SmokeUniforms;
@group(0) @binding(1) var<storage, read_write> compute_state : SmokeComputeState;

@compute @workgroup_size(1, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
    return;
  }
  let t = uniforms.time_and_frame.x;
  let base_scale = uniforms.time_and_frame.z;
  let rot_speed = 0.75 + uniforms.input_axes.z * 1.5;
  let angle = t * rot_speed;
  let zoom = clamp(base_scale + uniforms.input_axes.w * 0.25, 0.35, 1.65);
  let pulse = 0.85 + 0.15 * sin(t * 2.5);

  compute_state.transform = vec4<f32>(cos(angle), sin(angle), zoom, fract(t * 0.12));
  compute_state.offset = vec4<f32>(
    clamp(uniforms.input_axes.x * 0.45, -0.65, 0.65),
    clamp(uniforms.input_axes.y * 0.45, -0.65, 0.65),
    uniforms.time_and_frame.y,
    1.0
  );
  compute_state.color_mod = vec4<f32>(
    uniforms.tint_color.r * pulse,
    uniforms.tint_color.g * (0.9 + 0.1 * cos(t * 1.7)),
    uniforms.tint_color.b * pulse,
    1.0
  );
}
`;

// Real WebGPU WGSL vertex shader transforming a textured quad using compute-updated state.
export const SMOKE_VERTEX_WGSL = `struct SmokeUniforms {
  time_and_frame : vec4<f32>,
  input_axes     : vec4<f32>,
  tint_color     : vec4<f32>,
  extra          : vec4<f32>,
};

struct SmokeComputeState {
  transform : vec4<f32>,
  offset    : vec4<f32>,
  color_mod : vec4<f32>,
};

struct VertexOutput {
  @builtin(position) clip_pos : vec4<f32>,
  @location(0) uv             : vec2<f32>,
  @location(1) accent         : vec3<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : SmokeUniforms;
@group(0) @binding(1) var<storage, read> compute_state : SmokeComputeState;

@vertex
fn vs_main(@location(0) position : vec3<f32>) -> VertexOutput {
  let c = compute_state.transform.x;
  let s = compute_state.transform.y;
  let scale = compute_state.transform.z;
  let aspect = max(0.1, uniforms.extra.w);

  let scaled = position.xy * scale;
  let rotated = vec2<f32>(
    scaled.x * c - scaled.y * s,
    scaled.x * s + scaled.y * c
  );
  let world = vec2<f32>(rotated.x / aspect, rotated.y) + compute_state.offset.xy;

  var out : VertexOutput;
  out.clip_pos = vec4<f32>(world, position.z, 1.0);
  out.uv = vec2<f32>(position.x + 0.5, 0.5 - position.y);
  out.accent = compute_state.color_mod.rgb;
  return out;
}
`;

// Real WebGPU WGSL fragment shader sampling the procedural checkerboard/gradient texture.
export const SMOKE_FRAGMENT_WGSL = `struct SmokeUniforms {
  time_and_frame : vec4<f32>,
  input_axes     : vec4<f32>,
  tint_color     : vec4<f32>,
  extra          : vec4<f32>,
};

struct SmokeComputeState {
  transform : vec4<f32>,
  offset    : vec4<f32>,
  color_mod : vec4<f32>,
};

struct FragmentInput {
  @location(0) uv     : vec2<f32>,
  @location(1) accent : vec3<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : SmokeUniforms;
@group(0) @binding(1) var<storage, read> compute_state : SmokeComputeState;
@group(0) @binding(2) var smoke_sampler : sampler;
@group(0) @binding(3) var smoke_texture : texture_2d<f32>;

@fragment
fn fs_main(in : FragmentInput) -> @location(0) vec4<f32> {
  let scroll = compute_state.transform.w;
  let sample_uv = vec2<f32>(in.uv.x + scroll, in.uv.y);
  let texel = textureSample(smoke_texture, smoke_sampler, sample_uv);
  let tint = uniforms.tint_color.rgb * in.accent;
  let rgb = texel.rgb * tint;
  return vec4<f32>(rgb, texel.a);
}
`;

const textEncoder = new TextEncoder();

function toUint8Array(data, label = 'data') {
  if (data instanceof Uint8Array) {
    return data;
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  if (typeof SharedArrayBuffer !== 'undefined' && data instanceof SharedArrayBuffer) {
    return new Uint8Array(data);
  }
  throw new TypeError(`${label} must be a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`);
}

function toBigUint64(value) {
  return typeof value === 'bigint' ? value : BigInt(value >>> 0 === value ? value : Math.trunc(value));
}

function finiteOr(value, fallback) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/**
 * Binary packet encoder matching runtime/abi/playgta6_abi.h opcodes 0..21.
 */
export class SmokeCommandBuilder {
  constructor() {
    this._chunks = [];
    this._byteLength = 0;
  }

  get byteLength() {
    return this._byteLength;
  }

  get packetCount() {
    return this._chunks.length;
  }

  reset() {
    this._chunks = [];
    this._byteLength = 0;
    return this;
  }

  _pushPacket(opcode, payloadBytes, flags = 0) {
    const payloadLen = payloadBytes ? payloadBytes.byteLength : 0;
    const buf = new ArrayBuffer(8 + payloadLen);
    const view = new DataView(buf);
    view.setUint16(0, opcode, true);
    view.setUint16(2, flags, true);
    view.setUint32(4, payloadLen, true);
    if (payloadLen > 0) {
      new Uint8Array(buf, 8).set(payloadBytes);
    }
    const packet = new Uint8Array(buf);
    this._chunks.push(packet);
    this._byteLength += packet.byteLength;
    return this;
  }

  // Opcode 0: Nop (0 bytes)
  nop(flags = 0) {
    return this._pushPacket(GPU_OPCODES.NOP, null, flags);
  }

  // Opcode 1: BeginFrame (0 bytes)
  beginFrame(flags = 0) {
    return this._pushPacket(GPU_OPCODES.BEGIN_FRAME, null, flags);
  }

  // Opcode 2: ClearColor (16 bytes: f32 r, g, b, a)
  clearColor(r, g, b, a = 1.0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat32(0, r, true);
    view.setFloat32(4, g, true);
    view.setFloat32(8, b, true);
    view.setFloat32(12, a, true);
    return this._pushPacket(GPU_OPCODES.CLEAR_COLOR, new Uint8Array(buf));
  }

  // Opcode 3: CreateBuffer (16 bytes: u32 id, u32 usage_flags, u64 byte_size)
  createBuffer(id, usageFlags, byteSize) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, usageFlags, true);
    view.setBigUint64(8, toBigUint64(byteSize), true);
    return this._pushPacket(GPU_OPCODES.CREATE_BUFFER, new Uint8Array(buf));
  }

  // Opcode 4: WriteBuffer (16 + data_len bytes)
  writeBuffer(id, dstOffset, data) {
    const src = toUint8Array(data, 'writeBuffer data');
    const buf = new ArrayBuffer(16 + src.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, dstOffset, true);
    view.setUint32(8, src.byteLength, true);
    view.setUint32(12, 0, true);
    if (src.byteLength > 0) {
      new Uint8Array(buf, 16).set(src);
    }
    return this._pushPacket(GPU_OPCODES.WRITE_BUFFER, new Uint8Array(buf));
  }

  // Opcode 5: CreateShaderModule (16 + code_len bytes)
  createShaderModule(id, stage, wgslCode) {
    const encoded =
      typeof wgslCode === 'string'
        ? textEncoder.encode(wgslCode)
        : toUint8Array(wgslCode, 'createShaderModule wgslCode');
    const buf = new ArrayBuffer(16 + encoded.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, encoded.byteLength, true);
    view.setUint32(12, 0, true);
    if (encoded.byteLength > 0) {
      new Uint8Array(buf, 16).set(encoded);
    }
    return this._pushPacket(GPU_OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
  }

  // Opcode 6: Draw (16 bytes: u32 vertex_count, instance_count, first_vertex, first_instance)
  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, vertexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstVertex, true);
    view.setUint32(12, firstInstance, true);
    return this._pushPacket(GPU_OPCODES.DRAW, new Uint8Array(buf));
  }

  // Opcode 7: Present (0 bytes)
  present(flags = 0) {
    return this._pushPacket(GPU_OPCODES.PRESENT, null, flags);
  }

  // Opcode 8: CreateRenderPipeline (24 bytes)
  createRenderPipeline(
    id,
    vsShaderId,
    fsShaderId,
    vertexStride = 0,
    topology = GPU_PRIMITIVE_TOPOLOGY.TRIANGLE_LIST,
    flags = 0
  ) {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, vsShaderId, true);
    view.setUint32(8, fsShaderId, true);
    view.setUint32(12, vertexStride, true);
    view.setUint32(16, topology, true);
    view.setUint32(20, flags, true);
    return this._pushPacket(GPU_OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
  }

  // Opcode 9: SetPipeline (4 bytes: u32 pipeline_id)
  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    return this._pushPacket(GPU_OPCODES.SET_PIPELINE, new Uint8Array(buf));
  }

  // Opcode 10: SetVertexBuffer (16 bytes: u32 slot, u32 buffer_id, u64 offset)
  setVertexBuffer(slot, bufferId, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, slot, true);
    view.setUint32(4, bufferId, true);
    view.setBigUint64(8, toBigUint64(offset), true);
    return this._pushPacket(GPU_OPCODES.SET_VERTEX_BUFFER, new Uint8Array(buf));
  }

  // Opcode 11: DestroyResource (4 bytes: u32 id)
  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this._pushPacket(GPU_OPCODES.DESTROY_RESOURCE, new Uint8Array(buf));
  }

  // Opcode 12: CreateTexture (32 bytes)
  createTexture(
    id,
    width,
    height,
    depthOrLayers = 1,
    mipLevels = 1,
    format = GPU_TEXTURE_FORMAT.RGBA8_UNORM,
    usageFlags = GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
    dimension = GPU_TEXTURE_DIMENSION.DIM_2D
  ) {
    const buf = new ArrayBuffer(32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, width, true);
    view.setUint32(8, height, true);
    view.setUint32(12, depthOrLayers, true);
    view.setUint32(16, mipLevels, true);
    view.setUint32(20, format, true);
    view.setUint32(24, usageFlags, true);
    view.setUint32(28, dimension, true);
    return this._pushPacket(GPU_OPCODES.CREATE_TEXTURE, new Uint8Array(buf));
  }

  // Opcode 13: WriteTexture (48 + data_len bytes)
  writeTexture(
    textureId,
    mipLevel,
    originX,
    originY,
    originZ,
    width,
    height,
    depthOrLayers,
    bytesPerRow,
    rowsPerImage,
    data
  ) {
    const src = toUint8Array(data, 'writeTexture data');
    const buf = new ArrayBuffer(48 + src.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, textureId, true);
    view.setUint32(4, mipLevel, true);
    view.setUint32(8, originX, true);
    view.setUint32(12, originY, true);
    view.setUint32(16, originZ, true);
    view.setUint32(20, width, true);
    view.setUint32(24, height, true);
    view.setUint32(28, depthOrLayers, true);
    view.setUint32(32, bytesPerRow, true);
    view.setUint32(36, rowsPerImage, true);
    view.setUint32(40, src.byteLength, true);
    view.setUint32(44, 0, true);
    if (src.byteLength > 0) {
      new Uint8Array(buf, 48).set(src);
    }
    return this._pushPacket(GPU_OPCODES.WRITE_TEXTURE, new Uint8Array(buf));
  }

  // Opcode 14: CreateSampler (32 bytes)
  createSampler(
    id,
    addressU = GPU_ADDRESS_MODE.CLAMP_TO_EDGE,
    addressV = GPU_ADDRESS_MODE.CLAMP_TO_EDGE,
    addressW = GPU_ADDRESS_MODE.CLAMP_TO_EDGE,
    magFilter = GPU_FILTER_MODE.LINEAR,
    minFilter = GPU_FILTER_MODE.LINEAR,
    mipmapFilter = GPU_FILTER_MODE.LINEAR,
    maxAnisotropy = 1
  ) {
    const buf = new ArrayBuffer(32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, addressU, true);
    view.setUint32(8, addressV, true);
    view.setUint32(12, addressW, true);
    view.setUint32(16, magFilter, true);
    view.setUint32(20, minFilter, true);
    view.setUint32(24, mipmapFilter, true);
    view.setUint32(28, maxAnisotropy, true);
    return this._pushPacket(GPU_OPCODES.CREATE_SAMPLER, new Uint8Array(buf));
  }

  // Opcode 15: CreateBindGroup (16 + 32 * entry_count bytes)
  createBindGroup(id, pipelineId, groupIndex, entries = []) {
    const buf = new ArrayBuffer(16 + entries.length * 32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, pipelineId, true);
    view.setUint32(8, groupIndex, true);
    view.setUint32(12, entries.length, true);
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const off = 16 + i * 32;
      view.setUint32(off, entry.binding, true);
      view.setUint32(off + 4, entry.resourceType, true);
      view.setUint32(off + 8, entry.resourceId, true);
      view.setUint32(off + 12, 0, true);
      view.setBigUint64(off + 16, toBigUint64(entry.offset || 0), true);
      view.setBigUint64(off + 24, toBigUint64(entry.size || 0), true);
    }
    return this._pushPacket(GPU_OPCODES.CREATE_BIND_GROUP, new Uint8Array(buf));
  }

  // Opcode 16: SetBindGroup (8 bytes: u32 group_index, u32 bind_group_id)
  setBindGroup(groupIndex, bindGroupId) {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, groupIndex, true);
    view.setUint32(4, bindGroupId, true);
    return this._pushPacket(GPU_OPCODES.SET_BIND_GROUP, new Uint8Array(buf));
  }

  // Opcode 17: SetIndexBuffer (16 bytes: u32 buffer_id, u32 index_format, u64 offset)
  setIndexBuffer(bufferId, indexFormat = GPU_INDEX_FORMAT.UINT16, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, bufferId, true);
    view.setUint32(4, indexFormat, true);
    view.setBigUint64(8, toBigUint64(offset), true);
    return this._pushPacket(GPU_OPCODES.SET_INDEX_BUFFER, new Uint8Array(buf));
  }

  // Opcode 18: DrawIndexed (20 bytes: u32 index_count, instance_count, first_index, i32 base_vertex, u32 first_instance)
  drawIndexed(indexCount, instanceCount = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(20);
    const view = new DataView(buf);
    view.setUint32(0, indexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstIndex, true);
    view.setInt32(12, baseVertex, true);
    view.setUint32(16, firstInstance, true);
    return this._pushPacket(GPU_OPCODES.DRAW_INDEXED, new Uint8Array(buf));
  }

  // Opcode 19: CreateComputePipeline (16 bytes: u32 id, u32 cs_shader_id, u32 flags, u32 reserved)
  createComputePipeline(id, csShaderId, flags = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, csShaderId, true);
    view.setUint32(8, flags, true);
    view.setUint32(12, 0, true);
    return this._pushPacket(GPU_OPCODES.CREATE_COMPUTE_PIPELINE, new Uint8Array(buf));
  }

  // Opcode 20: SetComputePipeline (4 bytes: u32 pipeline_id)
  setComputePipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    return this._pushPacket(GPU_OPCODES.SET_COMPUTE_PIPELINE, new Uint8Array(buf));
  }

  // Opcode 21: DispatchCompute (12 bytes: u32 workgroup_x, workgroup_y, workgroup_z)
  dispatchCompute(workgroupX, workgroupY = 1, workgroupZ = 1) {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, workgroupX, true);
    view.setUint32(4, workgroupY, true);
    view.setUint32(8, workgroupZ, true);
    return this._pushPacket(GPU_OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf));
  }

  finish() {
    const out = new Uint8Array(this._byteLength);
    let offset = 0;
    for (let i = 0; i < this._chunks.length; i++) {
      const chunk = this._chunks[i];
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return out;
  }

  toArrayBuffer() {
    return this.finish().buffer;
  }
}

/**
 * Generate a procedural RGBA8 checkerboard + gradient texture.
 */
export function generateProceduralTexture(width = DEFAULT_TEXTURE_SIZE, height = DEFAULT_TEXTURE_SIZE) {
  if (!Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1) {
    throw new RangeError(`Invalid texture dimensions: ${width}x${height}`);
  }
  const pixels = new Uint8Array(width * height * 4);
  const tileSize = Math.max(1, Math.floor(Math.min(width, height) / 8));

  for (let y = 0; y < height; y++) {
    const v = height > 1 ? y / (height - 1) : 0;
    for (let x = 0; x < width; x++) {
      const u = width > 1 ? x / (width - 1) : 0;
      const checker = (((Math.floor(x / tileSize) + Math.floor(y / tileSize)) & 1) === 0);
      const isBorder = x < 2 || y < 2 || x >= width - 2 || y >= height - 2;

      let r;
      let g;
      let b;
      if (isBorder) {
        r = 255;
        g = 220;
        b = 120;
      } else if (checker) {
        r = Math.round(40 + u * 215);
        g = Math.round(180 - v * 90);
        b = Math.round(210 + (1 - u) * 45);
      } else {
        r = Math.round(24 + u * 80);
        g = Math.round(28 + v * 120);
        b = Math.round(52 + (1 - v) * 90);
      }

      const idx = (y * width + x) * 4;
      pixels[idx] = Math.min(255, Math.max(0, r));
      pixels[idx + 1] = Math.min(255, Math.max(0, g));
      pixels[idx + 2] = Math.min(255, Math.max(0, b));
      pixels[idx + 3] = 255;
    }
  }
  return pixels;
}

/**
 * Normalize optional inputState (plain JS object with camelCase or snake_case fields,
 * or raw 40-byte playgta6::InputState buffer/SharedArrayBuffer).
 */
export function normalizeInputState(inputState) {
  const normalized = {
    buttons: 0,
    mouseDx: 0,
    mouseDy: 0,
    leftStickX: 0,
    leftStickY: 0,
    rightStickX: 0,
    rightStickY: 0,
    leftTrigger: 0,
    rightTrigger: 0,
  };

  if (!inputState) {
    return normalized;
  }

  if (
    inputState instanceof ArrayBuffer ||
    (typeof SharedArrayBuffer !== 'undefined' && inputState instanceof SharedArrayBuffer) ||
    ArrayBuffer.isView(inputState)
  ) {
    const view = ArrayBuffer.isView(inputState)
      ? new DataView(inputState.buffer, inputState.byteOffset, inputState.byteLength)
      : new DataView(inputState);
    if (view.byteLength >= 40) {
      normalized.buttons = view.getUint32(4, true);
      normalized.mouseDx = view.getInt32(8, true);
      normalized.mouseDy = view.getInt32(12, true);
      normalized.leftStickX = finiteOr(view.getFloat32(16, true), 0);
      normalized.leftStickY = finiteOr(view.getFloat32(20, true), 0);
      normalized.rightStickX = finiteOr(view.getFloat32(24, true), 0);
      normalized.rightStickY = finiteOr(view.getFloat32(28, true), 0);
      normalized.leftTrigger = finiteOr(view.getFloat32(32, true), 0);
      normalized.rightTrigger = finiteOr(view.getFloat32(36, true), 0);
    }
    return normalized;
  }

  if (typeof inputState === 'object') {
    normalized.buttons = (inputState.buttons ?? 0) >>> 0;
    normalized.mouseDx = finiteOr(inputState.mouseDx ?? inputState.mouse_dx, 0);
    normalized.mouseDy = finiteOr(inputState.mouseDy ?? inputState.mouse_dy, 0);
    normalized.leftStickX = finiteOr(inputState.leftStickX ?? inputState.left_stick_x, 0);
    normalized.leftStickY = finiteOr(inputState.leftStickY ?? inputState.left_stick_y, 0);
    normalized.rightStickX = finiteOr(inputState.rightStickX ?? inputState.right_stick_x, 0);
    normalized.rightStickY = finiteOr(inputState.rightStickY ?? inputState.right_stick_y, 0);
    normalized.leftTrigger = finiteOr(inputState.leftTrigger ?? inputState.left_trigger, 0);
    normalized.rightTrigger = finiteOr(inputState.rightTrigger ?? inputState.right_trigger, 0);
  }

  return normalized;
}

/**
 * Helper to submit an encoded binary command stream cleanly into a GpuRingBuffer.
 *
 * @param {{ writeBytes: (bytes: Uint8Array) => boolean }} gpuRingBuffer
 * @param {Uint8Array|ArrayBufferView|ArrayBuffer|SharedArrayBuffer} bytes
 * @returns {boolean} True when all bytes were committed to the ring buffer.
 */
export function submitToRing(gpuRingBuffer, bytes) {
  if (!gpuRingBuffer || typeof gpuRingBuffer.writeBytes !== 'function') {
    throw new TypeError('submitToRing requires a GpuRingBuffer instance with writeBytes()');
  }
  const payload = toUint8Array(bytes, 'command buffer bytes');
  if (payload.byteLength === 0) {
    return true;
  }
  return Boolean(gpuRingBuffer.writeBytes(payload));
}

function buildQuadVertices() {
  // 4 vertices * vec3<f32> = 48 bytes (stride = 12 bytes)
  return new Float32Array([
    -0.5, -0.5, 0.0, // bottom-left
     0.5, -0.5, 0.0, // bottom-right
     0.5,  0.5, 0.0, // top-right
    -0.5,  0.5, 0.0, // top-left
  ]);
}

function buildQuadIndices() {
  // 2 triangles = 6 uint16 indices = 12 bytes (4-byte aligned)
  return new Uint16Array([0, 1, 2, 0, 2, 3]);
}

function buildInitialStorageState() {
  // Matches SmokeComputeState (3 x vec4<f32> = 48 bytes)
  return new Float32Array([
    1.0, 0.0, 1.0, 0.0, // transform: cos=1, sin=0, scale=1, scroll=0
    0.0, 0.0, 0.0, 1.0, // offset: x=0, y=0, framePhase=0, w=1
    1.0, 1.0, 1.0, 1.0, // color_mod: white
  ]);
}

/**
 * High-level controller producing init, per-frame animated, and teardown G5 binary command buffers.
 */
export class SmokeSceneController {
  constructor(options = {}) {
    const textureSize = options.textureSize !== undefined ? options.textureSize : DEFAULT_TEXTURE_SIZE;
    if (!Number.isInteger(textureSize) || textureSize < 1) {
      throw new RangeError(`textureSize must be a positive integer, got ${textureSize}`);
    }

    const aspect = options.aspect !== undefined ? options.aspect : 16 / 9;
    if (typeof aspect !== 'number' || !Number.isFinite(aspect) || aspect <= 0) {
      throw new RangeError(`aspect must be a positive finite number, got ${aspect}`);
    }

    this.textureSize = textureSize;
    this.aspect = aspect;
    this.resourceIds = Object.freeze({
      ...DEFAULT_RESOURCE_IDS,
      ...(options.resourceIds || {}),
    });

    // Teardown order: bind groups -> pipelines -> shader modules -> buffers -> sampler -> texture.
    this.allocatedIds = Object.freeze([
      this.resourceIds.COMPUTE_BIND_GROUP,
      this.resourceIds.RENDER_BIND_GROUP,
      this.resourceIds.COMPUTE_PIPELINE,
      this.resourceIds.RENDER_PIPELINE,
      this.resourceIds.COMPUTE_SHADER,
      this.resourceIds.FRAGMENT_SHADER,
      this.resourceIds.VERTEX_SHADER,
      this.resourceIds.STORAGE_BUFFER,
      this.resourceIds.UNIFORM_BUFFER,
      this.resourceIds.INDEX_BUFFER,
      this.resourceIds.VERTEX_BUFFER,
      this.resourceIds.SAMPLER,
      this.resourceIds.TEXTURE,
    ]);

    this.initialized = false;
    this.frameCount = 0;
  }

  _packUniforms(frameIndex, timeSec, inputState) {
    const t = finiteOr(timeSec, 0);
    const frame = finiteOr(frameIndex, 0);
    const inp = normalizeInputState(inputState);

    // Subtle hue shift + button/trigger response
    const boost = (inp.buttons & 0x100) !== 0 || (inp.buttons & 0x10) !== 0 ? 1.2 : 1.0;
    const tintR = Math.min(1.5, (0.95 + 0.05 * Math.sin(t * 1.1)) * boost);
    const tintG = Math.min(1.5, (0.92 + 0.08 * Math.cos(t * 0.9)) * boost);
    const tintB = Math.min(1.5, (0.98 + 0.02 * Math.sin(t * 1.9)) * boost);
    const baseScale = 1.0;
    const pulse = 0.5 + 0.5 * Math.sin(t * 2.0);

    const uniforms = new Float32Array(UNIFORM_BYTE_SIZE / 4);
    // time_and_frame
    uniforms[0] = t;
    uniforms[1] = frame;
    uniforms[2] = baseScale;
    uniforms[3] = pulse;
    // input_axes
    uniforms[4] = inp.leftStickX;
    uniforms[5] = inp.leftStickY;
    uniforms[6] = inp.rightStickX;
    uniforms[7] = inp.rightTrigger - inp.leftTrigger;
    // tint_color
    uniforms[8] = tintR;
    uniforms[9] = tintG;
    uniforms[10] = tintB;
    uniforms[11] = 1.0;
    // extra
    uniforms[12] = inp.buttons;
    uniforms[13] = inp.mouseDx;
    uniforms[14] = inp.mouseDy;
    uniforms[15] = this.aspect;

    return {
      bytes: new Uint8Array(uniforms.buffer),
      input: inp,
    };
  }

  /**
   * Build one-time resource initialization command buffer:
   * procedural RGBA8 texture + upload, sampler, vertex/index/uniform/storage buffers,
   * VS/FS/CS WGSL shader modules, render & compute pipelines, and bind groups.
   *
   * @returns {Uint8Array}
   */
  buildInitCommandBuffer() {
    const ids = this.resourceIds;
    const size = this.textureSize;
    const texels = generateProceduralTexture(size, size);
    const vertices = buildQuadVertices();
    const indices = buildQuadIndices();
    const { bytes: initialUniforms } = this._packUniforms(0, 0, null);
    const initialStorage = buildInitialStorageState();

    const builder = new SmokeCommandBuilder();

    // 1. Procedural texture + upload + sampler
    builder
      .createTexture(
        ids.TEXTURE,
        size,
        size,
        1,
        1,
        GPU_TEXTURE_FORMAT.RGBA8_UNORM,
        GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST,
        GPU_TEXTURE_DIMENSION.DIM_2D
      )
      .writeTexture(
        ids.TEXTURE,
        0,
        0,
        0,
        0,
        size,
        size,
        1,
        size * 4,
        size,
        texels
      )
      .createSampler(
        ids.SAMPLER,
        GPU_ADDRESS_MODE.REPEAT,
        GPU_ADDRESS_MODE.REPEAT,
        GPU_ADDRESS_MODE.CLAMP_TO_EDGE,
        GPU_FILTER_MODE.LINEAR,
        GPU_FILTER_MODE.LINEAR,
        GPU_FILTER_MODE.LINEAR,
        1
      );

    // 2. Buffers (vertex, index, uniform, compute storage)
    builder
      .createBuffer(
        ids.VERTEX_BUFFER,
        GPU_BUFFER_USAGE.VERTEX | GPU_BUFFER_USAGE.COPY_DST,
        vertices.byteLength
      )
      .writeBuffer(ids.VERTEX_BUFFER, 0, vertices)
      .createBuffer(
        ids.INDEX_BUFFER,
        GPU_BUFFER_USAGE.INDEX | GPU_BUFFER_USAGE.COPY_DST,
        indices.byteLength
      )
      .writeBuffer(ids.INDEX_BUFFER, 0, indices)
      .createBuffer(
        ids.UNIFORM_BUFFER,
        GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
        UNIFORM_BYTE_SIZE
      )
      .writeBuffer(ids.UNIFORM_BUFFER, 0, initialUniforms)
      .createBuffer(
        ids.STORAGE_BUFFER,
        GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
        STORAGE_BYTE_SIZE
      )
      .writeBuffer(ids.STORAGE_BUFFER, 0, initialStorage);

    // 3. Shader modules (VS, FS, CS)
    builder
      .createShaderModule(ids.VERTEX_SHADER, GPU_SHADER_STAGE.VERTEX, SMOKE_VERTEX_WGSL)
      .createShaderModule(ids.FRAGMENT_SHADER, GPU_SHADER_STAGE.FRAGMENT, SMOKE_FRAGMENT_WGSL)
      .createShaderModule(ids.COMPUTE_SHADER, GPU_SHADER_STAGE.COMPUTE, SMOKE_COMPUTE_WGSL);

    // 4. Pipelines (Render + Compute)
    builder
      .createRenderPipeline(
        ids.RENDER_PIPELINE,
        ids.VERTEX_SHADER,
        ids.FRAGMENT_SHADER,
        VERTEX_STRIDE_BYTES,
        GPU_PRIMITIVE_TOPOLOGY.TRIANGLE_LIST,
        0
      )
      .createComputePipeline(ids.COMPUTE_PIPELINE, ids.COMPUTE_SHADER, 0);

    // 5. Bind groups (Compute group 0 + Render group 0)
    builder
      .createBindGroup(ids.COMPUTE_BIND_GROUP, ids.COMPUTE_PIPELINE, 0, [
        {
          binding: 0,
          resourceType: GPU_BIND_RESOURCE_TYPE.BUFFER,
          resourceId: ids.UNIFORM_BUFFER,
          offset: 0,
          size: UNIFORM_BYTE_SIZE,
        },
        {
          binding: 1,
          resourceType: GPU_BIND_RESOURCE_TYPE.BUFFER,
          resourceId: ids.STORAGE_BUFFER,
          offset: 0,
          size: STORAGE_BYTE_SIZE,
        },
      ])
      .createBindGroup(ids.RENDER_BIND_GROUP, ids.RENDER_PIPELINE, 0, [
        {
          binding: 0,
          resourceType: GPU_BIND_RESOURCE_TYPE.BUFFER,
          resourceId: ids.UNIFORM_BUFFER,
          offset: 0,
          size: UNIFORM_BYTE_SIZE,
        },
        {
          binding: 1,
          resourceType: GPU_BIND_RESOURCE_TYPE.BUFFER,
          resourceId: ids.STORAGE_BUFFER,
          offset: 0,
          size: STORAGE_BYTE_SIZE,
        },
        {
          binding: 2,
          resourceType: GPU_BIND_RESOURCE_TYPE.SAMPLER,
          resourceId: ids.SAMPLER,
        },
        {
          binding: 3,
          resourceType: GPU_BIND_RESOURCE_TYPE.TEXTURE_VIEW,
          resourceId: ids.TEXTURE,
        },
      ]);

    this.initialized = true;
    return builder.finish();
  }

  /**
   * Build one animated frame command buffer:
   * BeginFrame -> WriteBuffer (uniforms) -> Compute pass (SetComputePipeline, SetBindGroup,
   * DispatchCompute(1,1,1)) -> ClearColor -> Render pass (SetPipeline, SetBindGroup,
   * SetVertexBuffer, SetIndexBuffer, DrawIndexed) -> Present.
   *
   * @param {number} frameIndex
   * @param {number} timeSec
   * @param {object|ArrayBuffer|SharedArrayBuffer|ArrayBufferView|null} [inputState]
   * @returns {Uint8Array}
   */
  buildFrameCommandBuffer(frameIndex = 0, timeSec = 0, inputState = null) {
    const ids = this.resourceIds;
    const t = finiteOr(timeSec, 0);
    const { bytes: uniformBytes, input } = this._packUniforms(frameIndex, t, inputState);

    const triggerBoost = Math.max(0, Math.min(1, input.rightTrigger)) * 0.12;
    const clearR = 0.06 + 0.04 * Math.sin(t * 0.8) + triggerBoost;
    const clearG = 0.08 + 0.03 * Math.cos(t * 0.6);
    const clearB = 0.16 + 0.06 * Math.sin(t * 0.5 + 1.2);

    const builder = new SmokeCommandBuilder();
    builder
      .beginFrame()
      .writeBuffer(ids.UNIFORM_BUFFER, 0, uniformBytes)
      .setComputePipeline(ids.COMPUTE_PIPELINE)
      .setBindGroup(0, ids.COMPUTE_BIND_GROUP)
      .dispatchCompute(1, 1, 1)
      .clearColor(clearR, clearG, clearB, 1.0)
      .setPipeline(ids.RENDER_PIPELINE)
      .setBindGroup(0, ids.RENDER_BIND_GROUP)
      .setVertexBuffer(0, ids.VERTEX_BUFFER, 0)
      .setIndexBuffer(ids.INDEX_BUFFER, GPU_INDEX_FORMAT.UINT16, 0)
      .drawIndexed(6, 1, 0, 0, 0)
      .present();

    this.frameCount += 1;
    return builder.finish();
  }

  /**
   * Build teardown command buffer emitting DestroyResource for all allocated IDs.
   *
   * @returns {Uint8Array}
   */
  buildTeardownCommandBuffer() {
    const builder = new SmokeCommandBuilder();
    for (let i = 0; i < this.allocatedIds.length; i++) {
      builder.destroyResource(this.allocatedIds[i]);
    }
    this.initialized = false;
    return builder.finish();
  }

  /**
   * Write an encoded command buffer into a GpuRingBuffer.
   *
   * @param {{ writeBytes: (bytes: Uint8Array) => boolean }} gpuRingBuffer
   * @param {Uint8Array|ArrayBufferView|ArrayBuffer|SharedArrayBuffer} bytes
   * @returns {boolean}
   */
  submitToRing(gpuRingBuffer, bytes) {
    return submitToRing(gpuRingBuffer, bytes);
  }

  static submitToRing(gpuRingBuffer, bytes) {
    return submitToRing(gpuRingBuffer, bytes);
  }
}
