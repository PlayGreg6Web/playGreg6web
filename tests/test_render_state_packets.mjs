import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';
import {
  SHADER_STAGE as WGPK_SHADER_STAGE,
  buildWgslPackBuffer,
  fnv1a64,
  formatHashHex,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, OPCODES, buildGpuTelemetrySnapshot } = require('../web/workers/wgpu_worker.js');
const rs = require('../web/workers/render_state_packets.js');
const {
  installRenderStatePackets,
  clampViewportToTarget,
  clampScissorToTarget,
  decodeRenderPipelineKey,
  resolveShaderEntryPoint,
  findWgslStageEntryPoints,
  validateVertexBufferLayouts,
  V2_OPCODES,
  SWAPCHAIN_TEXTURE_ID,
} = rs;

const repoFile = (rel) => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');

// ---------------------------------------------------------------------------------------------
// Enum tables: the JS mirrors must match runtime/gpu/pso_cache.{h,cpp} and the ABI header.
// ---------------------------------------------------------------------------------------------

function stripComments(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

function cppEnumMembers(text, name) {
  const match = new RegExp(`enum class ${name}\\s*:\\s*\\w+\\s*\\{([\\s\\S]*?)\\}`).exec(stripComments(text));
  assert.ok(match, `enum ${name} not found`);
  const members = [];
  let next = 0;
  for (const raw of match[1].split(',')) {
    const item = raw.trim();
    if (!item) continue;
    const [member, value] = item.split('=').map((s) => s.trim());
    if (value !== undefined && /^\d+$/.test(value)) next = Number(value);
    members.push({ name: member, value: next });
    next += 1;
  }
  return members;
}

// Members before `Count`, checked to be numbered 0..n-1.
function cppEnumNames(text, name) {
  const members = cppEnumMembers(text, name).filter((m) => m.name !== 'Count');
  members.forEach((m, i) => assert.equal(m.value, i, `${name}::${m.name} is not ${i}`));
  return members.map((m) => m.name);
}

function cppStringArray(text, arrayName) {
  const match = new RegExp(`${arrayName}\\[\\]\\s*=\\s*\\{([\\s\\S]*?)\\};`).exec(stripComments(text));
  assert.ok(match, `${arrayName} not found`);
  return [...match[1].matchAll(/"([^"]*)"/g)].map((m) => m[1]);
}

// Table rows like {"rgba8unorm", ...}: the first string of each row.
function cppTableNames(text, arrayName) {
  const match = new RegExp(`${arrayName}\\[\\]\\s*=\\s*\\{([\\s\\S]*?)\\n\\};`).exec(stripComments(text));
  assert.ok(match, `${arrayName} not found`);
  return [...match[1].matchAll(/\{\s*"([^"]*)"/g)].map((m) => m[1]);
}

const kebab = (s) => s.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase();

test('JS enum tables match the pso_cache *_name() tables and enum numbering', () => {
  const header = repoFile('runtime/gpu/pso_cache.h');
  const source = repoFile('runtime/gpu/pso_cache.cpp');
  const tables = [
    ['WgpuTextureFormat', cppTableNames(source, 'kTextureFormatInfo'), rs.TEXTURE_FORMAT_NAMES],
    ['WgpuVertexFormat', cppTableNames(source, 'kVertexFormatInfo'), rs.VERTEX_FORMAT_NAMES],
    ['WgpuBlendFactor', cppStringArray(source, 'kBlendFactorNames'), rs.BLEND_FACTOR_NAMES],
    ['WgpuBlendOperation', cppStringArray(source, 'kBlendOperationNames'), rs.BLEND_OPERATION_NAMES],
    ['WgpuCompareFunction', cppStringArray(source, 'kCompareNames'), rs.COMPARE_FUNCTION_NAMES],
    ['WgpuStencilOperation', cppStringArray(source, 'kStencilOpNames'), rs.STENCIL_OPERATION_NAMES],
    ['WgpuCullMode', cppStringArray(source, 'kCullNames'), rs.CULL_MODE_NAMES],
    ['WgpuFrontFace', cppStringArray(source, 'kFrontFaceNames'), rs.FRONT_FACE_NAMES],
    ['WgpuPrimitiveTopology', cppStringArray(source, 'kTopologyNames'), rs.PRIMITIVE_TOPOLOGY_NAMES],
    ['WgpuIndexFormat', cppStringArray(source, 'kIndexFormatNames'), rs.INDEX_FORMAT_NAMES],
    ['WgpuLoadOp', cppStringArray(source, 'kLoadOpNames'), rs.LOAD_OP_NAMES],
    ['WgpuStoreOp', cppStringArray(source, 'kStoreOpNames'), rs.STORE_OP_NAMES],
  ];
  for (const [enumName, cppNames, jsNames] of tables) {
    const members = cppEnumNames(header, enumName);
    assert.equal(cppNames.length, members.length, `${enumName}: C++ name table vs enum size`);
    assert.deepEqual([...jsNames], cppNames, `${enumName}: JS table drifted from pso_cache.cpp`);
  }
  // Enums without a *_name() function: compare member spellings.
  assert.deepEqual(cppEnumNames(header, 'WgpuVertexStepMode').map(kebab), [...rs.VERTEX_STEP_MODE_NAMES]);
  assert.deepEqual(cppEnumNames(header, 'WgpuTextureAspect').map(kebab), [...rs.TEXTURE_ASPECT_NAMES]);
});

test('packet-level enums and opcodes match runtime/abi/playgta6_abi.h', () => {
  const abi = repoFile('runtime/abi/playgta6_abi.h');
  const opcodes = Object.fromEntries(cppEnumMembers(abi, 'GpuOpcode').map((m) => [m.name, m.value]));
  for (const [name, value] of Object.entries(V2_OPCODES)) {
    const pascal = name.toLowerCase().replace(/(^|_)([a-z0-9])/g, (_, __, c) => c.toUpperCase());
    assert.equal(opcodes[pascal], value, `GpuOpcode::${pascal}`);
    assert.equal(OPCODES[name], value, `wgpu_worker OPCODES.${name}`);
  }
  const viewDims = cppEnumMembers(abi, 'GpuTextureViewDimension');
  assert.deepEqual(
    viewDims.map((m) => m.value),
    rs.VIEW_DIMENSION_NAMES.map((_, i) => i)
  );
  assert.deepEqual(
    viewDims.map((m) => m.name),
    ['Undefined', 'Dim1D', 'Dim2D', 'Dim2DArray', 'Cube', 'CubeArray', 'Dim3D']
  );
  assert.deepEqual(cppEnumMembers(abi, 'GpuLoadOp').map((m) => m.name.toLowerCase()), [...rs.LOAD_OP_NAMES]);
  assert.deepEqual(cppEnumMembers(abi, 'GpuStoreOp').map((m) => m.name.toLowerCase()), [...rs.STORE_OP_NAMES]);
  assert.match(abi, /kGpuSwapchainTextureId = 0xFFFFFFFFu/);
  assert.equal(SWAPCHAIN_TEXTURE_ID, 0xffffffff);
});

// ---------------------------------------------------------------------------------------------
// WebGPU mock (same approach as tests/test_wgpu_worker.mjs, plus the v2 surface)
// ---------------------------------------------------------------------------------------------

function createMockWebGpu() {
  const calls = [];
  const passCalls = [];
  let nextHandle = 1;
  const swapchainView = { kind: 'GPUTextureView', id: 'swapchain-view' };
  const obj = (kind, extra = {}) => ({ kind, handle: nextHandle++, ...extra });

  const ctx = {
    getCurrentTexture() {
      return { createView: () => swapchainView };
    },
  };

  const makePass = (kind) => {
    const record = (op, args) => passCalls.push({ pass: kind, op, args });
    return {
      setPipeline: (p) => record('setPipeline', [p]),
      setBindGroup: (i, bg, offsets) => record('setBindGroup', offsets === undefined ? [i, bg] : [i, bg, offsets]),
      setVertexBuffer: (s, b, o) => record('setVertexBuffer', [s, b, o]),
      setIndexBuffer: (b, f, o) => record('setIndexBuffer', [b, f, o]),
      setViewport: (...a) => record('setViewport', a),
      setScissorRect: (...a) => record('setScissorRect', a),
      setBlendConstant: (c) => record('setBlendConstant', [c]),
      setStencilReference: (r) => record('setStencilReference', [r]),
      draw: (...a) => record('draw', a),
      drawIndexed: (...a) => record('drawIndexed', a),
      dispatchWorkgroups: (...a) => record('dispatchWorkgroups', a),
      end: () => record('end', []),
    };
  };

  const device = {
    createBuffer(desc) {
      const buffer = obj('GPUBuffer', { size: desc.size, destroy() {} });
      calls.push({ op: 'createBuffer', desc: { ...desc }, result: buffer });
      return buffer;
    },
    createShaderModule(desc) {
      const module = obj('GPUShaderModule', { code: desc.code });
      calls.push({ op: 'createShaderModule', desc, result: module });
      return module;
    },
    createTexture(desc) {
      const texture = obj('GPUTexture', {
        desc,
        createView(viewDesc) {
          const view = obj('GPUTextureView', { textureHandle: texture.handle, viewDesc });
          calls.push({ op: 'texture.createView', texture, desc: viewDesc, result: view });
          return view;
        },
        destroy() {},
      });
      calls.push({ op: 'createTexture', desc, result: texture });
      return texture;
    },
    createSampler(desc) {
      const sampler = obj('GPUSampler', { desc });
      calls.push({ op: 'createSampler', desc, result: sampler });
      return sampler;
    },
    createBindGroupLayout(desc) {
      const layout = obj('GPUBindGroupLayout', { desc });
      calls.push({ op: 'createBindGroupLayout', desc, result: layout });
      return layout;
    },
    createPipelineLayout(desc) {
      const layout = obj('GPUPipelineLayout', { desc });
      calls.push({ op: 'createPipelineLayout', desc, result: layout });
      return layout;
    },
    createRenderPipeline(desc) {
      const pipeline = obj('GPURenderPipeline', { desc });
      calls.push({ op: 'createRenderPipeline', desc, result: pipeline });
      return pipeline;
    },
    createComputePipeline(desc) {
      const pipeline = obj('GPUComputePipeline', { desc });
      calls.push({ op: 'createComputePipeline', desc, result: pipeline });
      return pipeline;
    },
    createBindGroup(desc) {
      const bindGroup = obj('GPUBindGroup', { desc });
      calls.push({ op: 'createBindGroup', desc, result: bindGroup });
      return bindGroup;
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          passCalls.push({ pass: 'render', op: 'beginRenderPass', args: [desc] });
          return makePass('render');
        },
        beginComputePass() {
          passCalls.push({ pass: 'compute', op: 'beginComputePass', args: [] });
          return makePass('compute');
        },
        finish: () => obj('GPUCommandBuffer'),
      };
    },
    queue: {
      writeBuffer(buffer, dstOffset, data) {
        calls.push({ op: 'writeBuffer', buffer, dstOffset, data: new Uint8Array(data) });
      },
      writeTexture() {},
      submit(buffers) {
        calls.push({ op: 'submit', buffers });
      },
    },
  };
  const results = (op) => calls.filter((c) => c.op === op);
  return { device, ctx, calls, passCalls, swapchainView, results };
}

// v1 setup packets the d3d12_xs device would have sent before the recorded list.
class Stream {
  constructor() {
    this.parts = [];
  }
  packet(opcode, payload = new Uint8Array(0)) {
    const out = new Uint8Array(8 + payload.byteLength);
    const view = new DataView(out.buffer);
    view.setUint16(0, opcode, true);
    view.setUint32(4, payload.byteLength, true);
    out.set(payload, 8);
    this.parts.push(out);
    return this;
  }
  raw(bytes) {
    this.parts.push(bytes);
    return this;
  }
  u32s(opcode, values) {
    const payload = new Uint8Array(values.length * 4);
    const view = new DataView(payload.buffer);
    values.forEach((v, i) => view.setUint32(i * 4, v, true));
    return this.packet(opcode, payload);
  }
  shader(id, stage, code) {
    const text = new TextEncoder().encode(code);
    const payload = new Uint8Array(16 + text.byteLength);
    const view = new DataView(payload.buffer);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, text.byteLength, true);
    payload.set(text, 16);
    return this.packet(OPCODES.CREATE_SHADER_MODULE, payload);
  }
  buffer(id, usage, size) {
    const payload = new Uint8Array(16);
    const view = new DataView(payload.buffer);
    view.setUint32(0, id, true);
    view.setUint32(4, usage, true);
    view.setBigUint64(8, BigInt(size), true);
    return this.packet(OPCODES.CREATE_BUFFER, payload);
  }
  texture(id, width, height, format, usage = 0x14) {
    return this.u32s(OPCODES.CREATE_TEXTURE, [id, width, height, 1, 1, format, usage, 1]);
  }
  bytes() {
    const total = this.parts.reduce((n, p) => n + p.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const p of this.parts) {
      out.set(p, offset);
      offset += p.byteLength;
    }
    return out;
  }
}

function fixtureBytes() {
  const hex = repoFile('tests/fixtures/state_bridge_frame.hex').replace(/\s+/g, '');
  return new Uint8Array(hex.match(/../g).map((h) => parseInt(h, 16)));
}

// ---------------------------------------------------------------------------------------------
// End-to-end: the C++ bridge's packet stream (tests/test_state_bridge.cpp) drives WebGPU exactly.
// ---------------------------------------------------------------------------------------------

test('state bridge fixture stream produces the exact WebGPU calls', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  const setup = new Stream()
    .shader(1, 0, '@vertex fn vs_main() {}')
    .shader(2, 1, '@fragment fn fs_main() {}')
    .texture(101, 640, 360, 1) // offscreen bgra8unorm
    .texture(102, 1280, 720, 4) // depth32float
    .texture(103, 256, 256, 0) // albedo rgba8unorm
    .buffer(200, 0x20 | 0x08, 36)
    .packet(OPCODES.BEGIN_FRAME)
    .raw(fixtureBytes())
    .packet(OPCODES.PRESENT);
  const result = processor.executeCommandBuffer(setup.bytes());
  assert.equal(result.presents, 1);
  assert.deepEqual(handler.stats, { packets: 18, errors: 0 }, handler.lastError); // 27 packets, 9 of them v1

  const [vsModule, fsModule] = gpu.results('createShaderModule').map((c) => c.result);
  const [bglCall] = gpu.results('createBindGroupLayout');
  assert.deepEqual(bglCall.desc, {
    entries: [
      { binding: 0, visibility: 3, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 0 } },
      { binding: 1, visibility: 2, texture: { sampleType: 'float', viewDimension: '2d', multisampled: false } },
      { binding: 2, visibility: 2, sampler: { type: 'filtering' } },
    ],
  });
  const [plCall] = gpu.results('createPipelineLayout');
  assert.deepEqual(plCall.desc, { bindGroupLayouts: [bglCall.result] });

  const pipelines = gpu.results('createRenderPipeline');
  assert.equal(pipelines.length, 2);
  const stencilFace = { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' };
  const common = {
    layout: plCall.result,
    vertex: {
      module: vsModule,
      entryPoint: 'vs_main',
      buffers: [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }] }],
    },
    primitive: { topology: 'triangle-list', frontFace: 'cw', cullMode: 'back' },
    multisample: { count: 1, mask: 1, alphaToCoverageEnabled: false },
    fragment: { module: fsModule, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm', writeMask: 15 }] },
  };
  assert.deepEqual(pipelines[0].desc, {
    ...common,
    depthStencil: {
      format: 'depth32float',
      depthWriteEnabled: true,
      depthCompare: 'less',
      stencilFront: stencilFace,
      stencilBack: stencilFace,
      stencilReadMask: 0,
      stencilWriteMask: 0,
      depthBias: 0,
      depthBiasSlopeScale: 0,
      depthBiasClamp: 0,
    },
  });
  assert.deepEqual(pipelines[1].desc, common); // offscreen variant: no depth-stencil state

  const [sampler] = gpu.results('createSampler');
  assert.deepEqual(sampler.desc, {
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    addressModeW: 'repeat',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    lodMinClamp: 0,
    lodMaxClamp: 32,
    maxAnisotropy: 1,
  });

  const ring = gpu.results('createBuffer').find((c) => c.desc.size === 1 << 20);
  assert.deepEqual(ring.desc, { size: 1 << 20, usage: 0x0048 });
  // Views created from packet view descriptors (v1 CreateTexture also makes a default view).
  const views = gpu.results('texture.createView').filter((v) => v.desc !== undefined);
  const viewOf = (textureHandle) => views.find((v) => v.texture.handle === textureHandle);
  const textures = gpu.results('createTexture').map((c) => c.result);
  const albedoView = viewOf(textures[2].handle);
  assert.deepEqual(albedoView.desc, { format: 'rgba8unorm', dimension: '2d', arrayLayerCount: 1 });
  const depthView = viewOf(textures[1].handle);
  assert.deepEqual(depthView.desc, { format: 'depth32float', dimension: '2d', mipLevelCount: 1, arrayLayerCount: 1 });
  const offscreenView = viewOf(textures[0].handle);

  const [bindGroup] = gpu.results('createBindGroup');
  assert.deepEqual(bindGroup.desc, {
    layout: bglCall.result,
    entries: [
      { binding: 0, resource: { buffer: ring.result, offset: 0, size: 256 } },
      { binding: 1, resource: albedoView.result },
      { binding: 2, resource: sampler.result },
    ],
  });
  const writes = gpu.results('writeBuffer').filter((c) => c.buffer === ring.result);
  assert.deepEqual(writes.map((w) => w.dstOffset), [0, 256]);
  // The whole 256-byte ring slot is written: the 4 constants, then zeros.
  assert.deepEqual([...new Float32Array(writes[0].data.buffer)], [1, 0.5, 0.25, 1, ...new Array(60).fill(0)]);

  const vb = gpu.results('createBuffer').find((c) => c.desc.size === 36).result;
  const f32 = Math.fround;
  assert.deepEqual(gpu.passCalls, [
    {
      pass: 'render',
      op: 'beginRenderPass',
      args: [
        {
          colorAttachments: [
            { view: gpu.swapchainView, loadOp: 'clear', storeOp: 'store', clearValue: { r: f32(0.1), g: f32(0.2), b: f32(0.3), a: 1 } },
          ],
          depthStencilAttachment: { view: depthView.result, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
        },
      ],
    },
    { pass: 'render', op: 'setPipeline', args: [pipelines[0].result] },
    { pass: 'render', op: 'setViewport', args: [10, 20, 640, 360, 0, 1] },
    { pass: 'render', op: 'setScissorRect', args: [8, 8, 1264, 704] },
    { pass: 'render', op: 'setStencilReference', args: [128] },
    { pass: 'render', op: 'setBindGroup', args: [0, bindGroup.result, [0]] },
    { pass: 'render', op: 'setVertexBuffer', args: [0, vb, 0] },
    { pass: 'render', op: 'draw', args: [3, 1, 0, 0] },
    { pass: 'render', op: 'end', args: [] },
    {
      pass: 'render',
      op: 'beginRenderPass',
      args: [
        {
          colorAttachments: [
            { view: offscreenView.result, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } },
          ],
        },
      ],
    },
    { pass: 'render', op: 'setPipeline', args: [pipelines[1].result] },
    { pass: 'render', op: 'setViewport', args: [10, 20, 640, 360, 0, 1] },
    { pass: 'render', op: 'setScissorRect', args: [8, 8, 632, 352] },
    { pass: 'render', op: 'setStencilReference', args: [128] },
    { pass: 'render', op: 'setBindGroup', args: [0, bindGroup.result, [256]] },
    { pass: 'render', op: 'setVertexBuffer', args: [0, vb, 0] },
    { pass: 'render', op: 'draw', args: [6, 2, 0, 0] },
    { pass: 'render', op: 'end', args: [] },
  ]);
  assert.equal(gpu.results('submit').length, 1);
});

// ---------------------------------------------------------------------------------------------
// Unit coverage of the individual packets
// ---------------------------------------------------------------------------------------------

function header(opcode, payload) {
  return new Stream().packet(opcode, payload).bytes();
}

function bindGroupLayoutPacket(id, entries, { contentId = 0n, flags = 0 } = {}) {
  const payload = new Uint8Array(24 + entries.length * 24);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, entries.length, true);
  view.setUint32(8, flags, true);
  view.setBigUint64(16, contentId, true);
  entries.forEach((e, i) => {
    const o = 24 + i * 24;
    view.setUint32(o, e.binding, true);
    view.setUint32(o + 4, e.visibility, true);
    view.setUint8(o + 8, e.type);
    view.setUint8(o + 9, e.subtype || 0);
    view.setUint8(o + 10, e.dimension || 0);
    view.setUint8(o + 11, e.flags || 0);
    view.setUint8(o + 12, e.storageFormat || 0);
    view.setBigUint64(o + 16, BigInt(e.minBindingSize || 0), true);
  });
  return header(V2_OPCODES.CREATE_BIND_GROUP_LAYOUT, payload);
}

function pipelineLayoutPacket(id, groupIds, { contentId = 0n, flags = 0 } = {}) {
  const payload = new Uint8Array(24 + groupIds.length * 4);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, groupIds.length, true);
  view.setUint32(8, flags, true);
  view.setBigUint64(16, contentId, true);
  groupIds.forEach((g, i) => view.setUint32(24 + i * 4, g, true));
  return header(V2_OPCODES.CREATE_PIPELINE_LAYOUT, payload);
}

function bindGroup2Packet(id, layoutId, entries) {
  const payload = new Uint8Array(16 + entries.length * 48);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, layoutId, true);
  view.setUint32(8, entries.length, true);
  entries.forEach((e, i) => {
    const o = 16 + i * 48;
    view.setUint32(o, e.binding, true);
    view.setUint32(o + 4, e.type, true);
    view.setUint32(o + 8, e.id, true);
    view.setBigUint64(o + 16, BigInt(e.offset || 0), true);
    view.setBigUint64(o + 24, BigInt(e.size || 0), true);
    if (e.view) payload.set(e.view, o + 32);
  });
  return header(V2_OPCODES.CREATE_BIND_GROUP2, payload);
}

function setBindGroup2Packet(index, id, offsets = []) {
  const payload = new Uint8Array(16 + offsets.length * 4);
  const view = new DataView(payload.buffer);
  view.setUint32(0, index, true);
  view.setUint32(4, id, true);
  view.setUint32(8, offsets.length, true);
  offsets.forEach((o, i) => view.setUint32(16 + i * 4, o, true));
  return header(V2_OPCODES.SET_BIND_GROUP2, payload);
}

function viewDesc({ format = 0, dimension = 0, aspect = 0, baseMip = 0, mipCount = 0, baseLayer = 0, layerCount = 0 }) {
  const out = new Uint8Array(16);
  const view = new DataView(out.buffer);
  view.setUint8(0, format);
  view.setUint8(1, dimension);
  view.setUint8(2, aspect);
  view.setUint16(4, baseMip, true);
  view.setUint16(6, mipCount, true);
  view.setUint32(8, baseLayer, true);
  view.setUint32(12, layerCount, true);
  return out;
}

function renderPipeline2Packet(id, vs, fs, layoutId, keyBytes) {
  const payload = new Uint8Array(392);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, vs, true);
  view.setUint32(8, fs, true);
  view.setUint32(12, layoutId, true);
  payload.set(keyBytes, 16);
  return header(V2_OPCODES.CREATE_RENDER_PIPELINE2, payload);
}

function concat(...arrays) {
  const s = new Stream();
  arrays.forEach((a) => s.raw(a));
  return s.bytes();
}

test('decodeRenderPipelineKey maps every key field to GPURenderPipelineDescriptor members', () => {
  const key = new Uint8Array(376);
  const v = new DataView(key.buffer);
  v.setUint32(20, 3, true); // color targets: rgba16float + blend, hole, r32float
  key.set([40, 1, 0x7, 0, 0, 11, 12, 0, 4, 1, 0, 0], 24); // add constant/one-minus-constant, max one/zero
  key.set([16, 0, 0xf, 0], 48);
  key.set([47, 0, 7, 0, 5, 1, 2, 3, 8, 0, 0, 0], 120); // depth24plus-stencil8, greater-equal, stencil
  v.setUint32(132, 0xff, true);
  v.setUint32(136, 0x0f, true);
  v.setInt32(140, -4, true);
  v.setFloat32(144, 2.5, true);
  v.setFloat32(148, 0.125, true);
  key.set([1, 1, 0, 1, 1], 152); // triangle-strip, uint16 strip index, ccw, front, unclipped
  v.setUint32(160, 4, true);
  v.setUint32(164, 0xf, true);
  key[168] = 1;
  v.setUint32(172, 3, true); // buffers: slot0 vertex, slot1 hole, slot2 instance
  v.setUint32(176, 2, true);
  v.setUint32(180, 16, true);
  key.set([1, 1], 184);
  v.setUint32(196, 4, true);
  key.set([2, 1], 200);
  v.setUint32(244, 4, true);
  key.set([29, 3, 0], 248); // float32x2 @3 in buffer 0
  v.setUint32(252, 0, true);
  key.set([41, 5, 2], 256); // unorm8x4-bgra @5 in buffer 2

  const d = decodeRenderPipelineKey(new DataView(key.buffer), 0);
  assert.deepEqual(d.targets, [
    {
      format: 'rgba16float',
      writeMask: 7,
      blend: {
        color: { operation: 'add', srcFactor: 'constant', dstFactor: 'one-minus-constant' },
        alpha: { operation: 'max', srcFactor: 'one', dstFactor: 'zero' },
      },
    },
    null,
    { format: 'r32float', writeMask: 15 },
  ]);
  assert.deepEqual(d.depthStencil, {
    format: 'depth24plus-stencil8',
    depthWriteEnabled: false,
    depthCompare: 'greater-equal',
    stencilFront: { compare: 'greater', failOp: 'zero', depthFailOp: 'replace', passOp: 'invert' },
    stencilBack: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
    stencilReadMask: 0xff,
    stencilWriteMask: 0x0f,
    depthBias: -4,
    depthBiasSlopeScale: 2.5,
    depthBiasClamp: 0.125,
  });
  assert.deepEqual(d.primitive, {
    topology: 'triangle-strip',
    stripIndexFormat: 'uint16',
    frontFace: 'ccw',
    cullMode: 'front',
    unclippedDepth: true,
  });
  assert.deepEqual(d.multisample, { count: 4, mask: 0xf, alphaToCoverageEnabled: true });
  assert.deepEqual(d.buffers, [
    { arrayStride: 16, stepMode: 'vertex', attributes: [{ format: 'float32x2', offset: 4, shaderLocation: 3 }] },
    null,
    { arrayStride: 4, stepMode: 'instance', attributes: [{ format: 'unorm8x4-bgra', offset: 0, shaderLocation: 5 }] },
  ]);
});

test('pipeline without a pixel shader gets an empty fragment stage that writes nothing', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = installRenderStatePackets(processor);
  const key = new Uint8Array(376);
  new DataView(key.buffer).setUint32(20, 1, true);
  key.set([22, 0, 0xf, 0], 24);
  new DataView(key.buffer).setUint32(160, 1, true);
  processor.executeCommandBuffer(
    concat(new Stream().shader(1, 0, 'vs').bytes(), renderPipeline2Packet(5, 1, 0, 0, key))
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  const [pipeline] = gpu.results('createRenderPipeline');
  assert.equal(pipeline.desc.layout, 'auto');
  assert.equal(pipeline.desc.fragment.entryPoint, 'fs_main');
  assert.match(pipeline.desc.fragment.module.code, /@fragment fn fs_main\(\) \{\}/);
  assert.deepEqual(pipeline.desc.fragment.targets, [{ format: 'rgba8unorm', writeMask: 0 }]);
  assert.equal(processor.pipelines.get(5), pipeline.result);

  // Unknown vertex module: counted as an error, nothing created.
  processor.executeCommandBuffer(renderPipeline2Packet(6, 99, 0, 0, key));
  assert.equal(handler.stats.errors, 1);
  assert.match(handler.lastError, /unknown vertex shader module 99/);
});

test('bind groups: texture view descriptors, null bindings, manifest layouts and retirement', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const manifestLayout = { kind: 'GPUBindGroupLayout', from: 'manifest' };
  const handler = installRenderStatePackets(processor, {
    resolveManifestLayout: (kind, contentId) => (kind === 'bindGroupLayout' && contentId === 0xabcn ? manifestLayout : null),
  });
  const entries = [
    { binding: 0, visibility: 4, type: 0, subtype: 2, minBindingSize: 64 }, // read-only storage
    { binding: 1, visibility: 2, type: 2, subtype: 2, dimension: 4 }, // depth cube
    { binding: 2, visibility: 2, type: 1, subtype: 2 }, // comparison sampler
    { binding: 3, visibility: 4, type: 3, subtype: 0, dimension: 3, storageFormat: 16 }, // r32float 2d-array
    { binding: 4, visibility: 2, type: 2, subtype: 0, dimension: 2, flags: 0x2 }, // multisampled float
  ];
  const stream = concat(
    new Stream().texture(50, 64, 64, 0).bytes(),
    bindGroupLayoutPacket(10, entries, { contentId: 0x1234n }),
    bindGroupLayoutPacket(11, [], { contentId: 0xabcn, flags: 1 }),
    pipelineLayoutPacket(12, [10, 0, 11]),
    bindGroup2Packet(20, 10, [
      { binding: 0, type: 0, id: 0 },
      { binding: 1, type: 1, id: 0 },
      { binding: 2, type: 2, id: 0 },
      { binding: 3, type: 1, id: 50, view: viewDesc({ format: 16, dimension: 3, baseMip: 1, mipCount: 1, baseLayer: 2, layerCount: 3 }) },
      { binding: 4, type: 1, id: 0 },
    ])
  );
  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  const [bgl] = gpu.results('createBindGroupLayout');
  assert.deepEqual(bgl.desc.entries, [
    { binding: 0, visibility: 4, buffer: { type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 64 } },
    { binding: 1, visibility: 2, texture: { sampleType: 'depth', viewDimension: 'cube', multisampled: false } },
    { binding: 2, visibility: 2, sampler: { type: 'comparison' } },
    { binding: 3, visibility: 4, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '2d-array' } },
    { binding: 4, visibility: 2, texture: { sampleType: 'float', viewDimension: '2d', multisampled: true } },
  ]);
  const layouts = gpu.results('createBindGroupLayout');
  assert.equal(layouts.length, 2); // ours + the empty layout for group 1; group 2 came from the manifest
  assert.deepEqual(layouts[1].desc, { entries: [] });
  const [pl] = gpu.results('createPipelineLayout');
  assert.deepEqual(pl.desc.bindGroupLayouts, [bgl.result, layouts[1].result, manifestLayout]);

  const [bg] = gpu.results('createBindGroup');
  const resources = bg.desc.entries.map((e) => e.resource);
  const dummyBuffer = gpu.results('createBuffer')[0];
  assert.deepEqual(dummyBuffer.desc, { size: 256, usage: 0x40 | 0x80 | 0x08 });
  assert.deepEqual(resources[0], { buffer: dummyBuffer.result });
  const dummyTextures = gpu.results('createTexture').slice(1);
  assert.deepEqual(dummyTextures[0].desc, {
    size: { width: 1, height: 1, depthOrArrayLayers: 6 },
    format: 'depth32float',
    usage: 0x04,
    dimension: '2d',
    sampleCount: 1,
  });
  assert.deepEqual(dummyTextures[1].desc.sampleCount, 4);
  assert.deepEqual(gpu.results('createSampler')[0].desc, { compare: 'always' });
  const storageView = gpu.results('texture.createView').find((v) => v.result === resources[3]);
  assert.deepEqual(storageView.desc, {
    format: 'r32float',
    dimension: '2d-array',
    baseMipLevel: 1,
    mipLevelCount: 1,
    baseArrayLayer: 2,
    arrayLayerCount: 3,
  });

  // SetBindGroup2 resolves the bind group when parsed, so retiring it later in the frame is safe.
  const frame = concat(
    new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.CREATE_COMPUTE_PIPELINE, [30, 0, 0, 0]).bytes(),
    setBindGroup2Packet(1, 20, [512, 1024]),
    new Stream().u32s(OPCODES.DESTROY_RESOURCE, [20]).u32s(OPCODES.DESTROY_RESOURCE, [10]).packet(OPCODES.PRESENT).bytes()
  );
  processor.executeCommandBuffer(frame);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  // No pass was open for it: the bind group is not applied, and nothing throws.
  assert.deepEqual(
    gpu.passCalls.filter((c) => c.op === 'setBindGroup'),
    []
  );
  assert.equal(processor.bindGroups.has(20), false);
  assert.equal(handler.bindGroupLayouts.has(10), false);

  // ManifestReference without a resolver result is an error.
  processor.executeCommandBuffer(bindGroupLayoutPacket(13, [], { contentId: 0x999n, flags: 1 }));
  assert.equal(handler.stats.errors, 1);
  assert.match(handler.lastError, /manifest bind group layout 0x999 unavailable/);
});

test('explicit passes: dynamic state, v2 compute pipelines and v1 interop', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = installRenderStatePackets(processor);
  const pass = new Uint8Array(368);
  const pv = new DataView(pass.buffer);
  pv.setUint32(0, 2, true);
  pv.setUint32(8, 60, true); // color 0: texture 60, load/discard
  pass.set([1, 2], 12);
  pass.set(viewDesc({ format: 22, dimension: 2, mipCount: 1, layerCount: 1 }), 16);
  // color 1: hole; depth-stencil: read-only depth, stencil clear to 7
  pv.setUint32(328, 61, true);
  pass.set([0, 0, 2, 1], 332);
  pass.set(viewDesc({ aspect: 0 }), 336);
  pv.setUint32(356, 7, true);
  pass[360] = 1;

  const blend = new Uint8Array(16);
  new DataView(blend.buffer).setFloat32(0, 0.5, true);
  const compute = new Uint8Array(32);
  const cv = new DataView(compute.buffer);
  cv.setUint32(0, 70, true);
  cv.setUint32(4, 3, true);
  const stream = concat(
    new Stream()
      .shader(3, 2, 'cs')
      .texture(60, 32, 32, 0)
      .texture(61, 32, 32, 3)
      .packet(OPCODES.BEGIN_FRAME)
      .bytes(),
    header(V2_OPCODES.SET_VIEWPORT, new Uint8Array(24)), // outside a pass: ignored
    header(V2_OPCODES.BEGIN_RENDER_PASS, pass),
    header(V2_OPCODES.SET_BLEND_CONSTANT, blend),
    new Stream().u32s(V2_OPCODES.SET_SCISSOR_RECT, [1, 2, 3, 4]).bytes(),
    header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)),
    header(V2_OPCODES.CREATE_COMPUTE_PIPELINE2, compute),
    new Stream()
      .u32s(OPCODES.SET_COMPUTE_PIPELINE, [70])
      .u32s(OPCODES.DISPATCH_COMPUTE, [2, 1, 1])
      .packet(OPCODES.PRESENT)
      .bytes()
  );
  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  // (v1 CreateTexture also creates a default view without a descriptor.)
  const views = gpu.results('texture.createView').filter((v) => v.desc !== undefined);
  assert.deepEqual(gpu.passCalls.map((c) => c.op), [
    'beginRenderPass',
    'setBlendConstant',
    'setScissorRect',
    'end',
    'beginComputePass',
    'setPipeline',
    'dispatchWorkgroups',
    'end',
  ]);
  assert.deepEqual(gpu.passCalls[0].args[0], {
    colorAttachments: [{ view: views[0].result, loadOp: 'load', storeOp: 'discard' }, null],
    depthStencilAttachment: { view: views[1].result, stencilLoadOp: 'clear', stencilStoreOp: 'store', stencilClearValue: 7, depthReadOnly: true },
  });
  assert.deepEqual(views[0].desc, { format: 'rgba8unorm', dimension: '2d', mipLevelCount: 1, arrayLayerCount: 1 });
  assert.deepEqual(views[1].desc, {});
  assert.deepEqual(gpu.passCalls[1].args, [{ r: 0.5, g: 0, b: 0, a: 0 }]);
  assert.deepEqual(gpu.passCalls[2].args, [1, 2, 3, 4]);
  const [cp] = gpu.results('createComputePipeline');
  assert.deepEqual(cp.desc, { layout: 'auto', compute: { module: gpu.results('createShaderModule')[0].result, entryPoint: 'cs_main' } });
  assert.equal(gpu.passCalls[5].args[0], cp.result);
});

test('v2 packets are skipped by a processor without the extension and malformed packets are counted', () => {
  const gpu = createMockWebGpu();
  const plain = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const result = plain.executeCommandBuffer(concat(fixtureBytes(), new Stream().packet(OPCODES.PRESENT).bytes()));
  assert.equal(result.presents, 1);
  assert.equal(gpu.results('createRenderPipeline').length, 0);

  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = installRenderStatePackets(processor);
  processor.executeCommandBuffer(header(V2_OPCODES.BEGIN_RENDER_PASS, new Uint8Array(100)));
  processor.executeCommandBuffer(bindGroup2Packet(1, 404, []));
  assert.deepEqual(handler.stats, { packets: 2, errors: 2 });
  assert.match(handler.lastError, /unknown bind group layout 404/);
});

// ---------------------------------------------------------------------------------------------
// G33: Swapchain & offscreen render-pass viewport/scissor clamping
// ---------------------------------------------------------------------------------------------

function beginRenderPassPacket({ colors = [], depthStencil = null } = {}) {
  const payload = new Uint8Array(368);
  const view = new DataView(payload.buffer);
  view.setUint32(0, colors.length, true);
  colors.forEach((c, i) => {
    if (!c) return;
    const o = 8 + 40 * i;
    view.setUint32(o, c.textureId, true);
    view.setUint8(o + 4, c.loadOp ?? 2); // clear
    view.setUint8(o + 5, c.storeOp ?? 1); // store
    if (c.view) payload.set(c.view, o + 8);
  });
  if (depthStencil) {
    const d = 328;
    view.setUint32(d, depthStencil.textureId, true);
    view.setUint8(d + 4, depthStencil.depthLoadOp ?? 2); // clear
    view.setUint8(d + 5, depthStencil.depthStoreOp ?? 1); // store
    if (depthStencil.view) payload.set(depthStencil.view, d + 8);
    view.setFloat32(d + 24, depthStencil.depthClearValue ?? 1, true);
  }
  return header(V2_OPCODES.BEGIN_RENDER_PASS, payload);
}

function setViewportPacket(x, y, width, height, minDepth, maxDepth) {
  const payload = new Uint8Array(24);
  const view = new DataView(payload.buffer);
  view.setFloat32(0, x, true);
  view.setFloat32(4, y, true);
  view.setFloat32(8, width, true);
  view.setFloat32(12, height, true);
  view.setFloat32(16, minDepth, true);
  view.setFloat32(20, maxDepth, true);
  return header(V2_OPCODES.SET_VIEWPORT, payload);
}

function setScissorPacket(x, y, width, height) {
  return new Stream().u32s(V2_OPCODES.SET_SCISSOR_RECT, [x, y, width, height]).bytes();
}

test('G33 clampViewportToTarget clamps oversized, offset, negative, and depth-out-of-range viewports', () => {
  // 4K 3840x2160 guest viewport onto 1280x800 windowed swapchain
  assert.deepEqual(
    clampViewportToTarget({ x: 0, y: 0, width: 3840, height: 2160, minDepth: 0, maxDepth: 1 }, { width: 1280, height: 800 }),
    { x: 0, y: 0, width: 1280, height: 800, minDepth: 0, maxDepth: 1 }
  );

  // Single-axis oversized full-origin viewport
  assert.deepEqual(
    clampViewportToTarget({ x: 0, y: 0, width: 3840, height: 600, minDepth: 0, maxDepth: 1 }, { width: 1280, height: 800 }),
    { x: 0, y: 0, width: 1280, height: 600, minDepth: 0, maxDepth: 1 }
  );

  // In-bounds viewport preserved verbatim
  assert.deepEqual(
    clampViewportToTarget({ x: 10, y: 20, width: 640, height: 360, minDepth: 0.125, maxDepth: 0.875 }, { width: 1280, height: 800 }),
    { x: 10, y: 20, width: 640, height: 360, minDepth: 0.125, maxDepth: 0.875 }
  );

  // Offset out-of-bounds viewport clamped to exact valid sub-rect
  assert.deepEqual(
    clampViewportToTarget({ x: 1000, y: 700, width: 640, height: 360, minDepth: 0, maxDepth: 1 }, { width: 1280, height: 800 }),
    { x: 1000, y: 700, width: 280, height: 100, minDepth: 0, maxDepth: 1 }
  );

  // Offset completely beyond target bounds
  assert.deepEqual(
    clampViewportToTarget({ x: 2000, y: 1500, width: 640, height: 360, minDepth: 0, maxDepth: 1 }, { width: 1280, height: 800 }),
    { x: 1280, y: 800, width: 0, height: 0, minDepth: 0, maxDepth: 1 }
  );

  // Negative origin/dimensions clamped to valid non-negative bounds
  assert.deepEqual(
    clampViewportToTarget({ x: -20, y: -10, width: 1400, height: -5, minDepth: 0, maxDepth: 1 }, { width: 1280, height: 800 }),
    { x: 0, y: 0, width: 1280, height: 0, minDepth: 0, maxDepth: 1 }
  );

  // Depth range clamping: outside [0, 1], reversed min > max, and non-finite inputs
  assert.deepEqual(
    clampViewportToTarget({ x: 0, y: 0, width: 640, height: 360, minDepth: -0.5, maxDepth: 2.5 }, { width: 1280, height: 800 }),
    { x: 0, y: 0, width: 640, height: 360, minDepth: 0, maxDepth: 1 }
  );
  assert.deepEqual(
    clampViewportToTarget({ x: 0, y: 0, width: 640, height: 360, minDepth: 0.8, maxDepth: 0.2 }, { width: 1280, height: 800 }),
    { x: 0, y: 0, width: 640, height: 360, minDepth: 0.2, maxDepth: 0.8 }
  );
  assert.deepEqual(
    clampViewportToTarget({ x: Number.NaN, y: -0, width: 640, height: 360, minDepth: Number.NaN, maxDepth: Number.POSITIVE_INFINITY }, null),
    { x: 0, y: 0, width: 640, height: 360, minDepth: 0, maxDepth: 1 }
  );

  // Missing or non-positive targetSize preserves positive dimensions while sanitizing depth
  assert.deepEqual(
    clampViewportToTarget({ x: 0, y: 0, width: 3840, height: 2160, minDepth: -1, maxDepth: 2 }, null),
    { x: 0, y: 0, width: 3840, height: 2160, minDepth: 0, maxDepth: 1 }
  );
  assert.deepEqual(
    clampViewportToTarget({ x: 0, y: 0, width: 3840, height: 2160, minDepth: 1, maxDepth: 0 }, { width: 0, height: 800 }),
    { x: 0, y: 0, width: 3840, height: 2160, minDepth: 0, maxDepth: 1 }
  );
});

test('G33 clampScissorToTarget clamps oversized, offset, and fractional scissor rects', () => {
  // 4K 3840x2160 onto 1280x800 swapchain
  assert.deepEqual(
    clampScissorToTarget({ x: 0, y: 0, width: 3840, height: 2160 }, { width: 1280, height: 800 }),
    { x: 0, y: 0, width: 1280, height: 800 }
  );

  // In-bounds scissor preserved verbatim
  assert.deepEqual(
    clampScissorToTarget({ x: 8, y: 8, width: 1264, height: 704 }, { width: 1280, height: 720 }),
    { x: 8, y: 8, width: 1264, height: 704 }
  );

  // Offset out-of-bounds scissor clamped to exact valid sub-rect
  assert.deepEqual(
    clampScissorToTarget({ x: 1000, y: 700, width: 500, height: 300 }, { width: 1280, height: 800 }),
    { x: 1000, y: 700, width: 280, height: 100 }
  );

  // Completely out-of-bounds offset
  assert.deepEqual(
    clampScissorToTarget({ x: 2000, y: 1500, width: 500, height: 300 }, { width: 1280, height: 800 }),
    { x: 1280, y: 800, width: 0, height: 0 }
  );

  // Negative/fractional values floored and clamped
  assert.deepEqual(
    clampScissorToTarget({ x: -5.4, y: 10.9, width: 1300.7, height: 900.2 }, { width: 1280, height: 800 }),
    { x: 0, y: 10, width: 1280, height: 790 }
  );

  // Missing targetSize preserves non-negative integer rect
  assert.deepEqual(
    clampScissorToTarget({ x: 0, y: 0, width: 3840, height: 2160 }, null),
    { x: 0, y: 0, width: 3840, height: 2160 }
  );
});

test('G33 swapchain render pass clamps 4K 3840x2160 viewport/scissor to 1280x800 canvas bounds', () => {
  const gpu = createMockWebGpu();
  gpu.ctx.canvas = { width: 1280, height: 800 };
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  const stream = concat(
    new Stream().packet(OPCODES.BEGIN_FRAME).bytes(),
    beginRenderPassPacket({ colors: [{ textureId: SWAPCHAIN_TEXTURE_ID }] }),
    // Oversized 4K guest viewport/scissor + out-of-range depth
    setViewportPacket(0, 0, 3840, 2160, -0.25, 1.5),
    setScissorPacket(0, 0, 3840, 2160),
    // In-bounds viewport/scissor preserved verbatim
    setViewportPacket(16, 24, 640, 360, 0, 1),
    setScissorPacket(16, 24, 640, 360),
    // Offset out-of-bounds viewport/scissor + reversed depth
    setViewportPacket(1000, 700, 640, 360, 0.75, 0.25),
    setScissorPacket(1000, 700, 640, 360),
    header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)),
    new Stream().packet(OPCODES.PRESENT).bytes()
  );

  const result = processor.executeCommandBuffer(stream);
  assert.equal(result.presents, 1);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(handler._activePassTargetSize, null, 'END_RENDER_PASS must clear active pass bounds');

  const viewports = gpu.passCalls.filter((c) => c.op === 'setViewport').map((c) => c.args);
  const scissors = gpu.passCalls.filter((c) => c.op === 'setScissorRect').map((c) => c.args);
  assert.deepEqual(viewports, [
    [0, 0, 1280, 800, 0, 1],
    [16, 24, 640, 360, 0, 1],
    [1000, 700, 280, 100, 0.25, 0.75],
  ]);
  assert.deepEqual(scissors, [
    [0, 0, 1280, 800],
    [16, 24, 640, 360],
    [1000, 700, 280, 100],
  ]);
});

test('G33 swapchain size resolves via ctl.canvasWidth/canvasHeight, ctl.getCanvasTexture(), processor.canvas, and ctx.getCurrentTexture()', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  // 1. ctl.canvasWidth / ctl.canvasHeight
  assert.deepEqual(handler._resolveSwapchainSize({ canvasWidth: 1280, canvasHeight: 800 }), { width: 1280, height: 800 });

  // 2. ctl.getCanvasTexture()
  assert.deepEqual(
    handler._resolveSwapchainSize({ getCanvasTexture: () => ({ width: 1920, height: 1080 }) }),
    { width: 1920, height: 1080 }
  );

  // 3. processor.canvas
  processor.canvas = { width: 1024, height: 768 };
  assert.deepEqual(handler._resolveSwapchainSize({}), { width: 1024, height: 768 });
  delete processor.canvas;

  // 4. ctx.getCurrentTexture() returning { width, height }
  gpu.ctx.getCurrentTexture = () => ({ width: 1600, height: 900, createView: () => gpu.swapchainView });
  assert.deepEqual(handler._resolveSwapchainSize({}), { width: 1600, height: 900 });
});

test('G33 offscreen texture attachment bounds clamp viewport/scissor for color, mip, and depth-only passes', () => {
  const gpu = createMockWebGpu();
  // Expose WebGPU-spec texture.width / texture.height on created offscreen textures.
  const origCreateTexture = gpu.device.createTexture.bind(gpu.device);
  gpu.device.createTexture = (desc) => {
    const tex = origCreateTexture(desc);
    tex.width = desc.size.width;
    tex.height = desc.size.height;
    return tex;
  };
  gpu.ctx.canvas = { width: 1280, height: 800 };

  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);

  const stream = concat(
    new Stream()
      .texture(500, 512, 256, 0) // offscreen color 512x256
      .texture(501, 1024, 1024, 4) // depth-only shadow map 1024x1024
      .packet(OPCODES.BEGIN_FRAME)
      .bytes(),
    // Pass 1: offscreen color attachment (512x256)
    beginRenderPassPacket({ colors: [{ textureId: 500 }] }),
    setViewportPacket(0, 0, 3840, 2160, 0, 1),
    setScissorPacket(256, 128, 1024, 512),
    header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)),
    // Pass 2: offscreen color attachment at mip level 1 (256x128)
    beginRenderPassPacket({
      colors: [{ textureId: 500, view: viewDesc({ baseMip: 1, mipCount: 1 }) }],
    }),
    setViewportPacket(0, 0, 512, 256, 0, 1),
    setScissorPacket(0, 0, 512, 256),
    header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)),
    // Pass 3: depth-only pass (1024x1024)
    beginRenderPassPacket({ colors: [], depthStencil: { textureId: 501 } }),
    setViewportPacket(0, 0, 3840, 2160, 0, 1),
    setScissorPacket(512, 512, 2048, 2048),
    header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)),
    // Pass 4: swapchain pass (1280x800) verifies no stale offscreen bounds leaked
    beginRenderPassPacket({ colors: [{ textureId: SWAPCHAIN_TEXTURE_ID }] }),
    setViewportPacket(0, 0, 3840, 2160, 0, 1),
    setScissorPacket(0, 0, 3840, 2160),
    header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)),
    new Stream().packet(OPCODES.PRESENT).bytes()
  );

  const result = processor.executeCommandBuffer(stream);
  assert.equal(result.presents, 1);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(handler._activePassTargetSize, null);

  const viewports = gpu.passCalls.filter((c) => c.op === 'setViewport').map((c) => c.args);
  const scissors = gpu.passCalls.filter((c) => c.op === 'setScissorRect').map((c) => c.args);
  assert.deepEqual(viewports, [
    [0, 0, 512, 256, 0, 1],
    [0, 0, 256, 128, 0, 1],
    [0, 0, 1024, 1024, 0, 1],
    [0, 0, 1280, 800, 0, 1],
  ]);
  assert.deepEqual(scissors, [
    [256, 128, 256, 128],
    [0, 0, 256, 128],
    [512, 512, 512, 512],
    [0, 0, 1280, 800],
  ]);
});

// ---------------------------------------------------------------------------------------------
// W81: Frame-1 WebGPU pipeline hardening (entry points, root-sig group-0 layouts, depth24+s8)
// ---------------------------------------------------------------------------------------------

function buildFrame1PipelineKey({
  colorFormat = 22, // rgba8unorm
  depthFormat = 47, // depth24plus-stencil8
  depthWriteEnabled = 0,
  depthCompare = 0, // 0 = disabled / unset -> must default to 'always'
  multisampleCount = 0, // 0 -> must default to 1
} = {}) {
  const key = new Uint8Array(376);
  const v = new DataView(key.buffer);
  // Color target 0: rgba8unorm, blend off, writeMask 0xf
  v.setUint32(20, 1, true);
  key.set([colorFormat, 0, 0x0f, 0], 24);
  // Depth-stencil state
  key[120] = depthFormat;
  key[121] = depthWriteEnabled;
  key[122] = depthCompare;
  // Primitive: triangle-list (0), no strip index, ccw (0), cull none (0)
  key.set([0, 0, 0, 0, 0], 152);
  // Multisample
  v.setUint32(160, multisampleCount, true);
  v.setUint32(164, 0xffffffff, true);
  key[168] = 0;
  // Vertex layout: 1 buffer, 24B stride, 3 attributes:
  //   loc 0: float32x3 (30) @ 0
  //   loc 1: unorm8x4  (9)  @ 12
  //   loc 2: float32x2 (29) @ 16
  v.setUint32(172, 1, true); // vertex_buffer_count
  v.setUint32(176, 3, true); // vertex_attribute_count
  v.setUint32(180, 24, true); // buffer[0].array_stride
  key.set([1, 3], 184); // step_mode = vertex (1), attribute_count = 3
  // attr 0: offset 0, format 30 (float32x3), loc 0, buf 0
  v.setUint32(244, 0, true);
  key.set([30, 0, 0], 248);
  // attr 1: offset 12, format 9 (unorm8x4), loc 1, buf 0
  v.setUint32(252, 12, true);
  key.set([9, 1, 0], 256);
  // attr 2: offset 16, format 29 (float32x2), loc 2, buf 0
  v.setUint32(260, 16, true);
  key.set([29, 2, 0], 264);
  return key;
}

test('W81 Frame-1 CreateRenderPipeline2 produces WebGPU-spec-valid rgba8unorm + depth24plus-stencil8 + 24B 3-attribute descriptor', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  const frame1Key = buildFrame1PipelineKey({
    colorFormat: 22, // rgba8unorm
    depthFormat: 47, // depth24plus-stencil8
    depthWriteEnabled: 0,
    depthCompare: 0, // disabled -> must become 'always', never undefined
    multisampleCount: 0, // unset -> must become 1
  });
  const depth32OffKey = buildFrame1PipelineKey({
    colorFormat: 22,
    depthFormat: 48, // depth32float
    depthWriteEnabled: 0,
    depthCompare: 0,
    multisampleCount: 1,
  });
  const stencilOnlyKey = buildFrame1PipelineKey({
    colorFormat: 22,
    depthFormat: 44, // stencil8
    depthWriteEnabled: 1, // illegal on stencil8 -> must clamp to false
    depthCompare: 0,
    multisampleCount: 1,
  });

  const stream = concat(
    new Stream()
      .shader(
        1,
        0,
        '@vertex fn vs_main(@location(0) p: vec3<f32>, @location(1) c: vec4<f32>, @location(2) uv: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 1.0); }'
      )
      .shader(2, 1, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }')
      .bytes(),
    renderPipeline2Packet(10, 1, 2, 0, frame1Key),
    renderPipeline2Packet(11, 1, 2, 0, depth32OffKey),
    renderPipeline2Packet(12, 1, 2, 0, stencilOnlyKey)
  );

  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  const pipelines = gpu.results('createRenderPipeline');
  assert.equal(pipelines.length, 3);

  const d0 = pipelines[0].desc;
  assert.equal(d0.layout, 'auto');
  assert.equal(d0.vertex.entryPoint, 'vs_main');
  assert.equal(d0.fragment.entryPoint, 'fs_main');
  assert.deepEqual(d0.primitive, { topology: 'triangle-list', frontFace: 'ccw', cullMode: 'none' });
  assert.deepEqual(d0.multisample, { count: 1, mask: 0xffffffff, alphaToCoverageEnabled: false });
  assert.deepEqual(d0.fragment.targets, [{ format: 'rgba8unorm', writeMask: 0x0f }]);
  assert.deepEqual(d0.vertex.buffers, [
    {
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [
        { format: 'float32x3', offset: 0, shaderLocation: 0 },
        { format: 'unorm8x4', offset: 12, shaderLocation: 1 },
        { format: 'float32x2', offset: 16, shaderLocation: 2 },
      ],
    },
  ]);
  assert.equal(d0.depthStencil.format, 'depth24plus-stencil8');
  assert.strictEqual(d0.depthStencil.depthWriteEnabled, false);
  assert.strictEqual(d0.depthStencil.depthCompare, 'always');
  assert.deepEqual(d0.depthStencil.stencilFront, {
    compare: 'always',
    failOp: 'keep',
    depthFailOp: 'keep',
    passOp: 'keep',
  });
  assert.deepEqual(d0.depthStencil.stencilBack, {
    compare: 'always',
    failOp: 'keep',
    depthFailOp: 'keep',
    passOp: 'keep',
  });

  // depth32float depth-off also receives depthCompare: 'always' and boolean depthWriteEnabled: false
  const d1 = pipelines[1].desc;
  assert.equal(d1.depthStencil.format, 'depth32float');
  assert.strictEqual(d1.depthStencil.depthWriteEnabled, false);
  assert.strictEqual(d1.depthStencil.depthCompare, 'always');

  // stencil8 clamps depthWriteEnabled to false and omits depthCompare
  const d2 = pipelines[2].desc;
  assert.equal(d2.depthStencil.format, 'stencil8');
  assert.strictEqual(d2.depthStencil.depthWriteEnabled, false);
  assert.equal('depthCompare' in d2.depthStencil, false);

  assert.equal(handler.telemetry.depthCompareDefaults, 2);
  assert.equal(handler.telemetry.autoLayoutFallbacks, 3);
  assert.equal(handler.telemetry.legacyEntryPointRemaps, 0);
});

test('W81 validateVertexBufferLayouts validates Frame-1 24B layout and rejects invalid/dropped layouts', () => {
  assert.deepEqual(
    validateVertexBufferLayouts([
      {
        arrayStride: 24,
        stepMode: 'vertex',
        attributes: [
          { format: 'float32x3', offset: 0, shaderLocation: 0 },
          { format: 'unorm8x4', offset: 12, shaderLocation: 1 },
          { format: 'float32x2', offset: 16, shaderLocation: 2 },
        ],
      },
    ]),
    { valid: true }
  );

  // Unaligned stride (not multiple of 4)
  assert.equal(
    validateVertexBufferLayouts([
      { arrayStride: 22, stepMode: 'vertex', attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }] },
    ]).valid,
    false
  );

  // Attribute extends past arrayStride
  assert.equal(
    validateVertexBufferLayouts([
      { arrayStride: 16, stepMode: 'vertex', attributes: [{ format: 'float32x2', offset: 12, shaderLocation: 0 }] },
    ]).valid,
    false
  );

  // Duplicate shaderLocation
  assert.equal(
    validateVertexBufferLayouts([
      {
        arrayStride: 24,
        stepMode: 'vertex',
        attributes: [
          { format: 'float32x3', offset: 0, shaderLocation: 1 },
          { format: 'float32x2', offset: 12, shaderLocation: 1 },
        ],
      },
    ]).valid,
    false
  );

  // Orphaned attribute (bufferIndex points to unused slot) is rejected instead of silently dropped
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = installRenderStatePackets(processor);
  const badKey = buildFrame1PipelineKey();
  badKey[266] = 3; // attr 2 bufferIndex = 3 (>= bufferCount 1)
  processor.executeCommandBuffer(
    concat(
      new Stream().shader(1, 0, '@vertex fn vs_main() {}').shader(2, 1, '@fragment fn fs_main() {}').bytes(),
      renderPipeline2Packet(99, 1, 2, 0, badKey)
    )
  );
  assert.equal(handler.stats.errors, 1);
  assert.match(handler.lastError, /dropped attributes/);
  assert.equal(gpu.results('createRenderPipeline').length, 0);
});

test('W81 entry point resolution defaults to fs_main and remaps legacy ps_main with sanitized telemetry', () => {
  // Pure unit checks for WGSL comment stripping and stage matching
  const commentedWgsl = [
    '// @fragment fn commented_out() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }',
    '/* @fragment /* nested */ fn block_commented() {} */',
    'fn helper_color() -> vec4<f32> { return vec4<f32>(1.0); }',
    '@fragment',
    'fn ps_main() -> @location(0) vec4<f32> { return helper_color(); }',
  ].join('\n');
  assert.deepEqual(findWgslStageEntryPoints(commentedWgsl, 'fragment'), ['ps_main']);
  assert.deepEqual(
    resolveShaderEntryPoint({ code: commentedWgsl }, 'fragment'),
    { entryPoint: 'ps_main', defaultEntryPoint: 'fs_main', remapped: true, reason: 'source-inspect' }
  );
  // Stale metadata claiming fs_main when WGSL only declares ps_main is corrected to ps_main
  assert.deepEqual(
    resolveShaderEntryPoint({ code: commentedWgsl, entryPoint: 'fs_main' }, 'fragment'),
    { entryPoint: 'ps_main', defaultEntryPoint: 'fs_main', remapped: true, reason: 'source-override' }
  );
  // Standard fs_main is NOT remapped
  assert.deepEqual(
    resolveShaderEntryPoint({ code: '@fragment fn fs_main() {}', entryPoint: 'fs_main' }, 'fragment'),
    { entryPoint: 'fs_main', defaultEntryPoint: 'fs_main', remapped: false, reason: 'default' }
  );

  // End-to-end through CreateRenderPipeline2 & CreateComputePipeline2
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);
  const key = buildFrame1PipelineKey();

  // Case 1: Standard vs_main / fs_main -> zero remaps
  processor.executeCommandBuffer(
    concat(
      new Stream()
        .shader(1, 0, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }')
        .shader(2, 1, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }')
        .bytes(),
      renderPipeline2Packet(10, 1, 2, 0, key)
    )
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(gpu.results('createRenderPipeline')[0].desc.fragment.entryPoint, 'fs_main');
  assert.equal(handler.telemetry.legacyEntryPointRemaps, 0);
  assert.equal(handler.getTelemetry().warnings.length, 0);

  // Case 2: Legacy pack fragment shader with `@fragment fn ps_main`
  processor.executeCommandBuffer(
    concat(
      new Stream().shader(3, 1, commentedWgsl).bytes(),
      renderPipeline2Packet(11, 1, 3, 0, key)
    )
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(gpu.results('createRenderPipeline')[1].desc.vertex.entryPoint, 'vs_main');
  assert.equal(gpu.results('createRenderPipeline')[1].desc.fragment.entryPoint, 'ps_main');
  assert.equal(handler.telemetry.legacyEntryPointRemaps, 1);
  assert.equal(handler.getStats().legacyEntryPointRemaps, 1);

  // Case 3: Shader module carrying explicit entryPoint metadata without WGSL source
  const metaMod = gpu.device.createShaderModule({ code: '' });
  processor.shaderModules.set(4, { module: metaMod, stage: 1, entryPoint: 'ps_main' });
  processor.executeCommandBuffer(renderPipeline2Packet(12, 1, 4, 0, key));
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(gpu.results('createRenderPipeline')[2].desc.fragment.entryPoint, 'ps_main');
  assert.equal(handler.telemetry.legacyEntryPointRemaps, 2);

  // Case 4: Legacy compute shader `@compute @workgroup_size(8, 8, 1) fn main()`
  const compPayload = new Uint8Array(32);
  const cv = new DataView(compPayload.buffer);
  cv.setUint32(0, 20, true);
  cv.setUint32(4, 5, true);
  processor.executeCommandBuffer(
    concat(
      new Stream().shader(5, 2, '@compute @workgroup_size(8, 8, 1) fn main() {}').bytes(),
      header(V2_OPCODES.CREATE_COMPUTE_PIPELINE2, compPayload)
    )
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(gpu.results('createComputePipeline')[0].desc.compute.entryPoint, 'main');
  assert.equal(handler.telemetry.legacyEntryPointRemaps, 3);

  // Warnings are sanitized (no raw WGSL or file paths)
  const warnings = handler.getTelemetry().warnings;
  assert.equal(warnings.length, 3);
  assert.match(warnings[0], /stage=fragment default=fs_main resolved=ps_main/);
  assert.match(warnings[1], /stage=fragment default=fs_main resolved=ps_main/);
  assert.match(warnings[2], /stage=compute default=cs_main resolved=main/);
  for (const w of warnings) {
    assert.equal(w.includes('helper_color'), false);
    assert.equal(w.includes('file://'), false);
  }
});

test('W81 root-signature group-0 sequential binding layouts (cb0@0, tex0@1, samp0@2, non-filtering, storage) + auto fallback', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);
  const key = buildFrame1PipelineKey();

  // Matches runtime commit 91ad39f layout_from_root_signature: single group @group(0),
  // sequential bindings 0..4, tagged Vertex|Fragment (0x3).
  const rootSigEntries = [
    { binding: 0, visibility: 3, type: 0, subtype: 0, minBindingSize: 0 }, // cb0: uniform buffer
    { binding: 1, visibility: 3, type: 2, subtype: 0, dimension: 2 }, // tex0: sampled 2D float texture
    { binding: 2, visibility: 3, type: 1, subtype: 0 }, // samp0: filtering sampler
    { binding: 3, visibility: 3, type: 1, subtype: 1 }, // samp1: non-filtering sampler
    { binding: 4, visibility: 3, type: 0, subtype: 1, minBindingSize: 0 }, // uav0: storage buffer (VERTEX bit must strip to FRAGMENT 0x2!)
  ];

  const stream = concat(
    new Stream()
      .shader(1, 0, '@vertex fn vs_main() {}')
      .shader(2, 1, '@fragment fn fs_main() {}')
      .buffer(100, 0x0040 | 0x0008, 256) // cb0
      .texture(101, 64, 64, 0, 0x04) // tex0
      .buffer(104, 0x0080 | 0x0008, 256) // uav0
      .bytes(),
    header(
      V2_OPCODES.CREATE_SAMPLER2,
      (() => {
        const p = new Uint8Array(24);
        const dv = new DataView(p.buffer);
        dv.setUint32(0, 102, true); // samp0 filtering
        p.set([0, 0, 0, 1, 1, 1, 0], 4);
        dv.setUint16(12, 1, true);
        dv.setFloat32(16, 0, true);
        dv.setFloat32(20, 32, true);
        return p;
      })()
    ),
    header(
      V2_OPCODES.CREATE_SAMPLER2,
      (() => {
        const p = new Uint8Array(24);
        const dv = new DataView(p.buffer);
        dv.setUint32(0, 103, true); // samp1 non-filtering
        p.set([0, 0, 0, 0, 0, 0, 0], 4);
        dv.setUint16(12, 1, true);
        dv.setFloat32(16, 0, true);
        dv.setFloat32(20, 32, true);
        return p;
      })()
    ),
    bindGroupLayoutPacket(200, rootSigEntries),
    pipelineLayoutPacket(201, [200]), // single @group(0) pipeline layout
    renderPipeline2Packet(300, 1, 2, 201, key), // explicit root-sig layout
    renderPipeline2Packet(301, 1, 2, 0, key), // layout: 'auto' fallback
    bindGroup2Packet(400, 200, [
      { binding: 0, type: 0, id: 100, offset: 0, size: 256 },
      { binding: 1, type: 1, id: 101 },
      { binding: 2, type: 2, id: 102 },
      { binding: 3, type: 2, id: 103 },
      { binding: 4, type: 0, id: 104, offset: 0, size: 256 },
    ])
  );

  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  const [bgl] = gpu.results('createBindGroupLayout');
  assert.deepEqual(bgl.desc.entries, [
    { binding: 0, visibility: 3, buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 0 } },
    { binding: 1, visibility: 3, texture: { sampleType: 'float', viewDimension: '2d', multisampled: false } },
    { binding: 2, visibility: 3, sampler: { type: 'filtering' } },
    { binding: 3, visibility: 3, sampler: { type: 'non-filtering' } },
    { binding: 4, visibility: 2, buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 0 } },
  ]);
  const [pl] = gpu.results('createPipelineLayout');
  assert.deepEqual(pl.desc, { bindGroupLayouts: [bgl.result] });

  const pipelines = gpu.results('createRenderPipeline');
  assert.equal(pipelines.length, 2);
  assert.equal(pipelines[0].desc.layout, pl.result);
  assert.equal(pipelines[1].desc.layout, 'auto');
  assert.equal(pipelines[0].desc.vertex.buffers[0].attributes.length, 3);
  assert.equal(handler.telemetry.rootSigGroup0Layouts, 1);
  assert.equal(handler.telemetry.autoLayoutFallbacks, 1);

  const [bg] = gpu.results('createBindGroup');
  assert.equal(bg.desc.layout, bgl.result);
  assert.deepEqual(
    bg.desc.entries.map((e) => e.binding),
    [0, 1, 2, 3, 4]
  );

  // Duplicate binding numbers in CreateBindGroupLayout are rejected cleanly
  processor.executeCommandBuffer(
    bindGroupLayoutPacket(202, [
      { binding: 0, visibility: 3, type: 0, subtype: 0 },
      { binding: 0, visibility: 2, type: 1, subtype: 0 },
    ])
  );
  assert.equal(handler.stats.errors, 1);
  assert.match(handler.lastError, /duplicate binding 0/);
});

test('W81 G31 render_state_packets.js has zero ES module syntax and reloads cleanly in classic Worker VM', () => {
  const source = repoFile('web/workers/render_state_packets.js');
  assert.equal(/^\s*(?:import|export)\b/m.test(source), false, 'must have zero top-level ES import/export syntax');

  const sandbox = { console };
  sandbox.self = sandbox;
  sandbox.globalThis = sandbox;
  const ctx = vm.createContext(sandbox);

  assert.doesNotThrow(() => {
    vm.runInContext(source, ctx, { filename: 'render_state_packets.js' });
    // Simulate a second importScripts('./render_state_packets.js') call in the same WorkerGlobalScope
    vm.runInContext(source, ctx, { filename: 'render_state_packets.js' });
  });

  assert.ok(ctx.PlayGta6RenderState, 'PlayGta6RenderState must be published on WorkerGlobalScope');
  assert.equal(typeof ctx.PlayGta6RenderState.installRenderStatePackets, 'function');
  assert.equal(typeof ctx.PlayGta6RenderState.resolveShaderEntryPoint, 'function');
  assert.equal(typeof ctx.PlayGta6RenderState.validateVertexBufferLayouts, 'function');
  assert.equal(typeof ctx.PlayGta6RenderState.decodeCreateTexture2, 'function');
  assert.equal(typeof ctx.PlayGta6RenderState.sanitizeTextureDescriptor, 'function');
  assert.equal(typeof ctx.PlayGta6RenderState.validateComputeDispatch, 'function');
  assert.equal(typeof ctx.PlayGta6RenderState.resolveTextureSampleType, 'function');
  assert.equal(typeof ctx.PlayGta6RenderState.parseWgslGroup0Bindings, 'function');
  assert.equal(typeof ctx.PlayGta6RenderState.isExplicitLayoutCompatibleWithWgsl, 'function');

  // Verify zero leaked top-level lexical declarations
  assert.doesNotThrow(() => {
    vm.runInContext(
      'const TEXTURE_FORMAT_NAMES = []; const DEFAULT_ENTRY_POINTS = {}; class RenderStatePackets {}',
      ctx
    );
  });
});

// ---------------------------------------------------------------------------------------------
// W89: Compute dispatches (cs_main), @group(0) storage/S22 bindings, and G48/R32 texture formats
// ---------------------------------------------------------------------------------------------

function createTexture2Packet({
  id,
  width,
  height,
  depthOrLayers = 1,
  mipLevels = 1,
  sampleCount = 1,
  usage = 0x04,
  dimension = 1, // 2d
  format,
  viewFormats = [],
}) {
  const payload = new Uint8Array(40);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, width, true);
  view.setUint32(8, height, true);
  view.setUint32(12, depthOrLayers, true);
  view.setUint32(16, mipLevels, true);
  view.setUint32(20, sampleCount, true);
  view.setUint32(24, usage, true);
  view.setUint8(28, dimension);
  view.setUint8(29, format);
  view.setUint8(30, viewFormats.length);
  viewFormats.forEach((f, i) => view.setUint8(32 + i, f));
  return header(V2_OPCODES.CREATE_TEXTURE2, payload);
}

function createComputePipeline2Packet(id, csModuleId, layoutId = 0, csShaderHash = 0n, rootSigId = 0) {
  const payload = new Uint8Array(32);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, csModuleId, true);
  view.setUint32(8, layoutId, true);
  view.setBigUint64(16, csShaderHash, true);
  view.setUint32(24, rootSigId, true);
  return header(V2_OPCODES.CREATE_COMPUTE_PIPELINE2, payload);
}

function createSampler2Packet(id, { filter = 0, compare = 0 } = {}) {
  const payload = new Uint8Array(24);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  payload.set([0, 0, 0, filter, filter, filter, compare], 4);
  view.setUint16(12, 1, true);
  view.setFloat32(16, 0, true);
  view.setFloat32(20, 32, true);
  return header(V2_OPCODES.CREATE_SAMPLER2, payload);
}

function formatCodeOf(name) {
  const idx = rs.TEXTURE_FORMAT_NAMES.indexOf(name);
  assert.ok(idx > 0, `expected format ${name} in TEXTURE_FORMAT_NAMES`);
  return idx;
}

test('W89 CreateComputePipeline2 (cs_main) + @group(0) storage/uniform/texture bind group + Dispatch(8,4,1) + zero/oversized guards', () => {
  const gpu = createMockWebGpu();
  gpu.device.limits = { maxComputeWorkgroupsPerDimension: 256 };
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  const computeWgsl = [
    '@group(0) @binding(0) var<uniform> cb0: vec4<f32>;',
    '@group(0) @binding(1) var<storage, read> srcBuf: array<u32>;',
    '@group(0) @binding(2) var<storage, read_write> dstBuf: array<u32>;',
    '@group(0) @binding(3) var inTex: texture_2d<f32>;',
    '@group(0) @binding(4) var inSamp: sampler;',
    '@group(0) @binding(5) var outTex: texture_storage_2d<rgba32float, write>;',
    '@compute @workgroup_size(8, 8, 1)',
    'fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {}',
  ].join('\n');

  const group0Entries = [
    { binding: 0, visibility: 4, type: 0, subtype: 0, minBindingSize: 16 }, // uniform (16B tight S22 CBV)
    { binding: 1, visibility: 4, type: 0, subtype: 2, minBindingSize: 64 }, // read-only-storage
    { binding: 2, visibility: 4, type: 0, subtype: 1, minBindingSize: 64 }, // storage
    { binding: 3, visibility: 4, type: 2, subtype: 1, dimension: 2 }, // unfilterable-float 2d
    { binding: 4, visibility: 4, type: 1, subtype: 1 }, // non-filtering sampler
    { binding: 5, visibility: 4, type: 3, subtype: 0, dimension: 2, storageFormat: formatCodeOf('rgba32float') }, // write-only storage texture
  ];

  const stream = concat(
    new Stream()
      .shader(10, 2, computeWgsl)
      .shader(11, 2, 'fn not_a_compute_entry() {}') // lacks @compute / cs_main -> sanitized warning
      .buffer(100, 0x0040 | 0x0008, 16) // tight 16B CBV
      .buffer(101, 0x0080 | 0x0008, 256) // read-only storage
      .buffer(102, 0x0080 | 0x0008, 256) // read-write storage
      .bytes(),
    createTexture2Packet({
      id: 103,
      width: 64,
      height: 64,
      format: formatCodeOf('rgba32float'),
      usage: 0x04 | 0x02, // TEXTURE_BINDING | COPY_DST
    }),
    createSampler2Packet(104, { filter: 0 }),
    createTexture2Packet({
      id: 105,
      width: 64,
      height: 64,
      format: formatCodeOf('rgba32float'),
      usage: 0x08 | 0x04, // STORAGE_BINDING | TEXTURE_BINDING
    }),
    bindGroupLayoutPacket(200, group0Entries),
    pipelineLayoutPacket(201, [200]),
    createComputePipeline2Packet(300, 10, 201, 0xdeadbeefn, 7),
    createComputePipeline2Packet(301, 11, 0), // missing @compute warning
    bindGroup2Packet(400, 200, [
      { binding: 0, type: 0, id: 100, offset: 0, size: 16 },
      { binding: 1, type: 0, id: 101, offset: 0, size: 256 },
      { binding: 2, type: 0, id: 102, offset: 0, size: 256 },
      { binding: 3, type: 1, id: 103 },
      { binding: 4, type: 2, id: 104 },
      { binding: 5, type: 1, id: 105 },
    ]),
    new Stream()
      .packet(OPCODES.BEGIN_FRAME)
      .u32s(OPCODES.SET_COMPUTE_PIPELINE, [300])
      .bytes(),
    setBindGroup2Packet(0, 400, []),
    new Stream()
      .u32s(OPCODES.DISPATCH_COMPUTE, [8, 4, 1]) // valid dispatch
      .u32s(OPCODES.DISPATCH_COMPUTE, [0, 4, 1]) // zero X -> skipped
      .u32s(OPCODES.DISPATCH_COMPUTE, [8, 0, 1]) // zero Y -> skipped
      .u32s(OPCODES.DISPATCH_COMPUTE, [8, 4, 0]) // zero Z -> skipped
      .u32s(OPCODES.DISPATCH_COMPUTE, [1024, 4, 512]) // exceeds limit 256 -> clamped to 256x4x256
      .u32s(OPCODES.SET_COMPUTE_PIPELINE, [9999]) // missing pipeline -> skipped
      .u32s(OPCODES.DISPATCH_COMPUTE, [2, 2, 1]) // dispatch on missing pipeline -> skipped
      .packet(OPCODES.PRESENT)
      .bytes()
  );

  const result = processor.executeCommandBuffer(stream);
  assert.equal(result.presents, 1);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  const [cp0, cp1] = gpu.results('createComputePipeline');
  const [pl] = gpu.results('createPipelineLayout');
  const [bg] = gpu.results('createBindGroup');
  assert.equal(cp0.desc.layout, pl.result);
  assert.equal(cp0.desc.compute.entryPoint, 'cs_main');
  assert.equal(cp1.desc.layout, 'auto');
  assert.equal(cp1.desc.compute.entryPoint, 'cs_main');

  // Verify exact compute pass call sequence: beginComputePass -> setPipeline -> setBindGroup(0) -> dispatchWorkgroups -> end
  const computeCalls = gpu.passCalls.filter((c) => c.pass === 'compute');
  assert.deepEqual(
    computeCalls.map((c) => c.op),
    ['beginComputePass', 'setPipeline', 'setBindGroup', 'dispatchWorkgroups', 'dispatchWorkgroups', 'end']
  );
  assert.equal(computeCalls[1].args[0], cp0.result);
  assert.equal(computeCalls[2].args[0], 0);
  assert.equal(computeCalls[2].args[1], bg.result);
  assert.deepEqual(computeCalls[3].args, [8, 4, 1]);
  assert.deepEqual(computeCalls[4].args, [256, 4, 256]);

  // Verify compute telemetry counters in extension stats snapshot
  const stats = handler.getStats();
  assert.equal(stats.computePipelineCount, 2);
  assert.equal(stats.computeDispatchCount, 2);
  assert.equal(stats.skippedComputeDispatchCount, 4);
  assert.equal(stats.clampedComputeDispatchCount, 1);
  assert.equal(stats.rootSigGroup0Layouts, 1);

  // Verify sanitized warnings (no raw WGSL identifiers leaked)
  const warnings = handler.getTelemetry().warnings;
  assert.ok(warnings.some((w) => /lacks @compute/.test(w)), 'must warn when compute shader lacks @compute');
  assert.ok(warnings.some((w) => /zero or invalid workgroups/.test(w)), 'must warn on zero dispatch');
  assert.ok(warnings.some((w) => /clamped compute dispatch.*256x4x256/.test(w)), 'must warn on oversized clamp');
  assert.ok(warnings.some((w) => /compute pipeline 9999/.test(w)), 'must warn on missing compute pipeline');
  for (const w of warnings) {
    assert.equal(w.includes('not_a_compute_entry'), false, 'warning must not leak WGSL identifiers');
  }

  // Pure validateComputeDispatch checks
  assert.deepEqual(rs.validateComputeDispatch(8, 4, 1, gpu.device), {
    skip: false,
    clamped: false,
    x: 8,
    y: 4,
    z: 1,
    maxWorkgroupsPerDimension: 256,
    reason: null,
  });
  assert.equal(rs.validateComputeDispatch(0, 4, 1, gpu.device).skip, true);
  assert.equal(rs.validateComputeDispatch(1, -2, 1, gpu.device).skip, true);
  assert.deepEqual(rs.validateComputeDispatch(70000, 1, 1, null), {
    skip: false,
    clamped: true,
    x: 65535,
    y: 1,
    z: 1,
    maxWorkgroupsPerDimension: 65535,
    reason: 'exceeds-limit',
  });
});

test('W89 R32G32B32A32_FLOAT (rgba32float) texture creation & unfilterable-float binding without WebGPU validation errors', () => {
  // Strict WebGPU mock enforcing float32-filterable rules on createBindGroup
  function createStrictFloat32Mock(enableFloat32Filterable) {
    const gpu = createMockWebGpu();
    gpu.device.features = enableFloat32Filterable ? new Set(['float32-filterable']) : new Set();
    const origCreateBg = gpu.device.createBindGroup.bind(gpu.device);
    const origCreateTex = gpu.device.createTexture.bind(gpu.device);
    const viewFormats = new Map();
    gpu.device.createTexture = (desc) => {
      const tex = origCreateTex(desc);
      tex.format = desc.format;
      const origView = tex.createView.bind(tex);
      tex.createView = (viewDesc) => {
        const v = origView(viewDesc);
        viewFormats.set(v, (viewDesc && viewDesc.format) || desc.format);
        return v;
      };
      return tex;
    };
    gpu.device.createBindGroup = (desc) => {
      const layoutEntries = (desc.layout && desc.layout.desc && desc.layout.desc.entries) || [];
      for (const e of desc.entries) {
        const le = layoutEntries.find((x) => x.binding === e.binding);
        if (le && le.texture && le.texture.sampleType === 'float') {
          const fmt = viewFormats.get(e.resource);
          if (
            (fmt === 'rgba32float' || fmt === 'rg32float' || fmt === 'r32float') &&
            !gpu.device.features.has('float32-filterable')
          ) {
            throw new Error(`GPUValidationError: texture format ${fmt} is not filterable`);
          }
        }
      }
      return origCreateBg(desc);
    };
    return { gpu, viewFormats };
  }

  // Case 1: Device WITHOUT float32-filterable
  assert.equal(rs.resolveTextureSampleType('rgba32float', { features: new Set() }), 'unfilterable-float');
  assert.equal(rs.resolveTextureSampleType('r32float', { features: new Set() }), 'unfilterable-float');
  assert.equal(rs.resolveTextureSampleType('rg32float', { features: new Set() }), 'unfilterable-float');
  assert.equal(rs.resolveTextureSampleType('r32uint', { features: new Set() }), 'uint');
  assert.equal(rs.resolveTextureSampleType('depth24plus-stencil8', { features: new Set() }), 'depth');
  assert.equal(rs.resolveTextureSampleType('rgba16float', { features: new Set() }), 'float');
  assert.equal(
    rs.resolveTextureSampleType('rgba32float', { features: new Set(['float32-filterable']) }),
    'float'
  );

  const strictNoF32 = createStrictFloat32Mock(false);
  const procNoF32 = new WgpuCommandProcessor(strictNoF32.gpu.device, strictNoF32.gpu.ctx);
  const handlerNoF32 = installRenderStatePackets(procNoF32);

  procNoF32.executeCommandBuffer(
    concat(
      createTexture2Packet({
        id: 50,
        width: 128,
        height: 128,
        format: formatCodeOf('rgba32float'),
        usage: 0x04 | 0x08 | 0x10 | 0x02, // TEXTURE_BINDING | STORAGE_BINDING | RENDER_ATTACHMENT | COPY_DST (all valid for rgba32float!)
      }),
      // Layout A: unfilterable-float (matches rgba32float directly)
      bindGroupLayoutPacket(10, [{ binding: 0, visibility: 6, type: 2, subtype: 1, dimension: 2 }]),
      bindGroup2Packet(20, 10, [{ binding: 0, type: 1, id: 50 }]),
      // Layout B: filterable float (must NOT trigger GPUValidationError without float32-filterable!)
      bindGroupLayoutPacket(11, [{ binding: 0, visibility: 2, type: 2, subtype: 0, dimension: 2 }]),
      bindGroup2Packet(21, 11, [{ binding: 0, type: 1, id: 50 }])
    )
  );
  assert.equal(handlerNoF32.stats.errors, 0, handlerNoF32.lastError);

  const createdTex = strictNoF32.gpu.results('createTexture')[0];
  assert.equal(createdTex.desc.format, 'rgba32float');
  assert.equal(createdTex.desc.usage, 0x04 | 0x08 | 0x10 | 0x02);

  const [bgUnfilt, bgFilt] = strictNoF32.gpu.results('createBindGroup');
  // Unfilterable layout binds the real rgba32float texture view directly!
  assert.equal(strictNoF32.viewFormats.get(bgUnfilt.desc.entries[0].resource), 'rgba32float');
  // Filterable layout substituted a filterable fallback instead of failing WebGPU validation!
  assert.equal(strictNoF32.viewFormats.get(bgFilt.desc.entries[0].resource), 'rgba8unorm');
  assert.equal(handlerNoF32.getStats().unfilterableFloatFallbacks, 1);

  // Case 2: Device WITH float32-filterable binds real rgba32float to 'float' directly
  const strictWithF32 = createStrictFloat32Mock(true);
  const procWithF32 = new WgpuCommandProcessor(strictWithF32.gpu.device, strictWithF32.gpu.ctx);
  const handlerWithF32 = installRenderStatePackets(procWithF32);
  procWithF32.executeCommandBuffer(
    concat(
      createTexture2Packet({ id: 50, width: 64, height: 64, format: formatCodeOf('rgba32float'), usage: 0x04 }),
      bindGroupLayoutPacket(11, [{ binding: 0, visibility: 2, type: 2, subtype: 0, dimension: 2 }]),
      bindGroup2Packet(21, 11, [{ binding: 0, type: 1, id: 50 }])
    )
  );
  assert.equal(handlerWithF32.stats.errors, 0, handlerWithF32.lastError);
  const [bgDirect] = strictWithF32.gpu.results('createBindGroup');
  assert.equal(strictWithF32.viewFormats.get(bgDirect.desc.entries[0].resource), 'rgba32float');
  assert.equal(handlerWithF32.getStats().unfilterableFloatFallbacks, 0);
});

test('W89 G48/R32 texture format & usage compatibility strips invalid STORAGE_BINDING / RENDER_ATTACHMENT and falls back cleanly without texture-compression-bc', () => {
  const allG48Formats = [
    'rgba8unorm',
    'rgba8unorm-srgb',
    'bgra8unorm',
    'rgba16float',
    'rgba32float',
    'r32float',
    'rg32float',
    'r32uint',
    'rg11b10ufloat',
    'rgb10a2unorm',
    'depth24plus-stencil8',
    'depth32float',
    'bc1-rgba-unorm',
    'bc3-rgba-unorm',
    'bc4-r-unorm',
    'bc5-rg-unorm',
    'bc6h-rgb-ufloat',
    'bc7-rgba-unorm',
  ];

  // 1. Device WITHOUT texture-compression-bc / bgra8unorm-storage / rg11b10ufloat-renderable
  const gpuNoFeatures = createMockWebGpu();
  gpuNoFeatures.device.features = new Set();
  const origCreateTex = gpuNoFeatures.device.createTexture.bind(gpuNoFeatures.device);
  gpuNoFeatures.device.createTexture = (desc) => {
    if (desc.format.startsWith('bc')) {
      throw new Error(`GPUValidationError: ${desc.format} requires texture-compression-bc`);
    }
    return origCreateTex(desc);
  };

  const processor = new WgpuCommandProcessor(gpuNoFeatures.device, gpuNoFeatures.ctx);
  const handler = installRenderStatePackets(processor);

  const packets = allG48Formats.map((fmt, idx) =>
    createTexture2Packet({
      id: 1000 + idx,
      width: 64,
      height: 64,
      format: formatCodeOf(fmt),
      usage: 0x04 | 0x08 | 0x10 | 0x02, // TEXTURE_BINDING | STORAGE_BINDING | RENDER_ATTACHMENT | COPY_DST
    })
  );
  // Also emit a second bc7-rgba-unorm texture to verify warning deduplication, plus a view with bc7 format code
  packets.push(
    createTexture2Packet({
      id: 2000,
      width: 32,
      height: 32,
      format: formatCodeOf('bc7-rgba-unorm'),
      usage: 0x04 | 0x08 | 0x10,
      viewFormats: [formatCodeOf('bc7-rgba-unorm')],
    }),
    bindGroupLayoutPacket(50, [{ binding: 0, visibility: 2, type: 2, subtype: 0, dimension: 2 }]),
    bindGroup2Packet(51, 50, [
      { binding: 0, type: 1, id: 2000, view: viewDesc({ format: formatCodeOf('bc7-rgba-unorm'), dimension: 2 }) },
    ])
  );

  processor.executeCommandBuffer(concat(...packets));
  assert.equal(handler.stats.errors, 0, handler.lastError);

  const created = gpuNoFeatures.results('createTexture');
  const byIndex = (name) => created[allG48Formats.indexOf(name)].desc;

  // sRGB: STORAGE_BINDING stripped, RENDER_ATTACHMENT kept
  assert.equal(byIndex('rgba8unorm-srgb').format, 'rgba8unorm-srgb');
  assert.equal(byIndex('rgba8unorm-srgb').usage & 0x08, 0);
  assert.equal(byIndex('rgba8unorm-srgb').usage & 0x10, 0x10);

  // bgra8unorm without bgra8unorm-storage: STORAGE_BINDING stripped, RENDER_ATTACHMENT kept
  assert.equal(byIndex('bgra8unorm').format, 'bgra8unorm');
  assert.equal(byIndex('bgra8unorm').usage & 0x08, 0);
  assert.equal(byIndex('bgra8unorm').usage & 0x10, 0x10);

  // depth formats: STORAGE_BINDING stripped, RENDER_ATTACHMENT kept
  assert.equal(byIndex('depth24plus-stencil8').usage & 0x08, 0);
  assert.equal(byIndex('depth24plus-stencil8').usage & 0x10, 0x10);
  assert.equal(byIndex('depth32float').usage & 0x08, 0);
  assert.equal(byIndex('depth32float').usage & 0x10, 0x10);

  // core storage+render formats: both STORAGE_BINDING and RENDER_ATTACHMENT kept
  for (const fmt of ['rgba8unorm', 'rgba16float', 'rgba32float', 'r32float', 'rg32float', 'r32uint']) {
    assert.equal(byIndex(fmt).format, fmt);
    assert.equal(byIndex(fmt).usage & 0x08, 0x08, `${fmt} must preserve STORAGE_BINDING`);
    assert.equal(byIndex(fmt).usage & 0x10, 0x10, `${fmt} must preserve RENDER_ATTACHMENT`);
  }

  // rg11b10ufloat without feature: both STORAGE_BINDING and RENDER_ATTACHMENT stripped
  assert.equal(byIndex('rg11b10ufloat').usage & 0x08, 0);
  assert.equal(byIndex('rg11b10ufloat').usage & 0x10, 0);

  // rgb10a2unorm: STORAGE_BINDING stripped (tier1), RENDER_ATTACHMENT kept
  assert.equal(byIndex('rgb10a2unorm').usage & 0x08, 0);
  assert.equal(byIndex('rgb10a2unorm').usage & 0x10, 0x10);

  // BC formats without texture-compression-bc: fell back to rgba8unorm with STORAGE_BINDING & RENDER_ATTACHMENT stripped
  for (const bcFmt of ['bc1-rgba-unorm', 'bc3-rgba-unorm', 'bc4-r-unorm', 'bc5-rg-unorm', 'bc6h-rgb-ufloat', 'bc7-rgba-unorm']) {
    assert.equal(byIndex(bcFmt).format, 'rgba8unorm', `${bcFmt} must fall back to rgba8unorm`);
    assert.equal(byIndex(bcFmt).usage & 0x08, 0, `${bcFmt} must strip STORAGE_BINDING`);
    assert.equal(byIndex(bcFmt).usage & 0x10, 0, `${bcFmt} must strip RENDER_ATTACHMENT`);
  }

  // View on fallback BC7 texture remaps requested bc7-rgba-unorm view format to rgba8unorm
  const bc7ViewCall = gpuNoFeatures
    .results('texture.createView')
    .find((v) => v.desc && v.desc.dimension === '2d');
  assert.equal(bc7ViewCall.desc.format, 'rgba8unorm');

  // Telemetry counts & deduplicated BC7 diagnostic (two bc7 textures -> only ONE bc7-unsupported warning)
  const stats = handler.getStats();
  assert.equal(stats.texture2Count, allG48Formats.length + 1);
  assert.equal(stats.bcFormatFallbacks, 7);
  assert.ok(stats.usageStrips > 0);
  const bc7Warnings = handler
    .getTelemetry()
    .warnings.filter((w) => w.includes('bc7-rgba-unorm unavailable'));
  assert.equal(bc7Warnings.length, 1, 'BC7 fallback diagnostic must be deduplicated');

  // 2. Device WITH texture-compression-bc and bgra8unorm-storage keeps native BC formats & bgra8 storage,
  //    while still stripping STORAGE_BINDING and RENDER_ATTACHMENT on BC formats!
  const gpuWithBc = createMockWebGpu();
  gpuWithBc.device.features = new Set(['texture-compression-bc', 'bgra8unorm-storage']);
  const procWithBc = new WgpuCommandProcessor(gpuWithBc.device, gpuWithBc.ctx);
  const handlerWithBc = installRenderStatePackets(procWithBc);
  procWithBc.executeCommandBuffer(
    concat(
      createTexture2Packet({
        id: 1,
        width: 128,
        height: 128,
        format: formatCodeOf('bc7-rgba-unorm'),
        usage: 0x04 | 0x08 | 0x10,
      }),
      createTexture2Packet({
        id: 2,
        width: 128,
        height: 128,
        format: formatCodeOf('bgra8unorm'),
        usage: 0x04 | 0x08 | 0x10,
      })
    )
  );
  assert.equal(handlerWithBc.stats.errors, 0, handlerWithBc.lastError);
  const [bcNative, bgraStorage] = gpuWithBc.results('createTexture');
  assert.equal(bcNative.desc.format, 'bc7-rgba-unorm');
  assert.equal(bcNative.desc.usage, 0x04); // STORAGE_BINDING & RENDER_ATTACHMENT stripped even with BC feature
  assert.equal(bgraStorage.desc.format, 'bgra8unorm');
  assert.equal(bgraStorage.desc.usage, 0x04 | 0x08 | 0x10);
  assert.equal(handlerWithBc.getStats().bcFormatFallbacks, 0);
});

test('W89 S22 tight 16-byte & 48-byte CBV @group(0) bindings plus full storage/texture/sampler layout alignment', () => {
  const gpu = createMockWebGpu();
  gpu.device.features = new Set();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  // Full S22 @group(0) layout: 16B CBV, 48B CBV, read-only storage, read-write storage,
  // float 2D texture, unfilterable-float 2D texture, uint 2D texture, depth 2D texture,
  // filtering/non-filtering/comparison samplers, and write-only storage texture.
  const s22Entries = [
    { binding: 0, visibility: 7, type: 0, subtype: 0, minBindingSize: 16 }, // tight 16B CBV
    { binding: 1, visibility: 7, type: 0, subtype: 0, minBindingSize: 48 }, // tight 48B CBV
    { binding: 2, visibility: 7, type: 0, subtype: 2, minBindingSize: 32 }, // read-only-storage
    { binding: 3, visibility: 7, type: 0, subtype: 1, minBindingSize: 64 }, // storage (VERTEX stripped -> 6)
    { binding: 4, visibility: 6, type: 2, subtype: 0, dimension: 2 }, // float 2d
    { binding: 5, visibility: 6, type: 2, subtype: 1, dimension: 2 }, // unfilterable-float 2d
    { binding: 6, visibility: 6, type: 2, subtype: 4, dimension: 2 }, // uint 2d
    { binding: 7, visibility: 6, type: 2, subtype: 2, dimension: 2 }, // depth 2d
    { binding: 8, visibility: 6, type: 1, subtype: 0 }, // filtering sampler
    { binding: 9, visibility: 6, type: 1, subtype: 1 }, // non-filtering sampler
    { binding: 10, visibility: 6, type: 1, subtype: 2 }, // comparison sampler
    { binding: 11, visibility: 7, type: 3, subtype: 0, dimension: 2, storageFormat: formatCodeOf('rgba8unorm') }, // write-only storage texture (VERTEX stripped -> 6)
  ];

  const stream = concat(
    new Stream()
      .buffer(1, 0x0040 | 0x0008, 16) // exact 16B tight CBV buffer
      .buffer(2, 0x0040 | 0x0008, 48) // exact 48B tight CBV buffer
      .buffer(3, 0x0080 | 0x0008, 128) // read-only storage
      .buffer(4, 0x0080 | 0x0008, 128) // read-write storage
      .buffer(5, 0x0040 | 0x0008, 256) // larger buffer bound with undersized range (16B < 48B) -> widened to 48B
      .bytes(),
    createTexture2Packet({ id: 10, width: 16, height: 16, format: formatCodeOf('rgba8unorm'), usage: 0x04 }),
    createTexture2Packet({ id: 11, width: 16, height: 16, format: formatCodeOf('rgba32float'), usage: 0x04 }),
    createTexture2Packet({ id: 12, width: 16, height: 16, format: formatCodeOf('r32uint'), usage: 0x04 }),
    createTexture2Packet({ id: 13, width: 16, height: 16, format: formatCodeOf('depth32float'), usage: 0x04 }),
    createTexture2Packet({ id: 14, width: 16, height: 16, format: formatCodeOf('rgba8unorm'), usage: 0x08 }),
    createSampler2Packet(20, { filter: 1 }),
    createSampler2Packet(21, { filter: 0 }),
    createSampler2Packet(22, { filter: 1, compare: 2 }), // less
    bindGroupLayoutPacket(100, s22Entries),
    bindGroup2Packet(200, 100, [
      { binding: 0, type: 0, id: 1, offset: 0, size: 16 },
      { binding: 1, type: 0, id: 2, offset: 0, size: 48 },
      { binding: 2, type: 0, id: 3, offset: 0, size: 128 },
      { binding: 3, type: 0, id: 4, offset: 0, size: 128 },
      { binding: 4, type: 1, id: 10 },
      { binding: 5, type: 1, id: 11 },
      { binding: 6, type: 1, id: 12 },
      { binding: 7, type: 1, id: 13 },
      { binding: 8, type: 2, id: 20 },
      { binding: 9, type: 2, id: 21 },
      { binding: 10, type: 2, id: 22 },
      { binding: 11, type: 1, id: 14 },
    ]),
    // Second bind group tests widening a 16B range on a 256B buffer up to the 48B layout minBindingSize
    bindGroupLayoutPacket(101, [{ binding: 0, visibility: 7, type: 0, subtype: 0, minBindingSize: 48 }]),
    bindGroup2Packet(201, 101, [{ binding: 0, type: 0, id: 5, offset: 0, size: 16 }])
  );

  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  const [bgl] = gpu.results('createBindGroupLayout');
  assert.deepEqual(bgl.desc.entries[0], {
    binding: 0,
    visibility: 7,
    buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 16 },
  });
  assert.deepEqual(bgl.desc.entries[1], {
    binding: 1,
    visibility: 7,
    buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 48 },
  });
  assert.equal(bgl.desc.entries[3].visibility, 6); // VERTEX stripped from read-write storage
  assert.equal(bgl.desc.entries[11].visibility, 6); // VERTEX stripped from write-only storageTexture
  assert.deepEqual(bgl.desc.entries[11].storageTexture, {
    access: 'write-only',
    format: 'rgba8unorm',
    viewDimension: '2d',
  });

  const [bg0, bg1] = gpu.results('createBindGroup');
  assert.equal(bg0.desc.entries[0].resource.size, 16);
  assert.equal(bg0.desc.entries[1].resource.size, 48);
  assert.equal(bg0.desc.entries.length, 12);
  // Widened to minBindingSize 48 so WebGPU never rejects the binding
  assert.equal(bg1.desc.entries[0].resource.size, 48);

  // Duplicate binding in CreateBindGroup2 is rejected cleanly
  processor.executeCommandBuffer(
    bindGroup2Packet(202, 101, [
      { binding: 0, type: 0, id: 5, offset: 0, size: 48 },
      { binding: 0, type: 0, id: 5, offset: 0, size: 48 },
    ])
  );
  assert.equal(handler.stats.errors, 1);
  assert.match(handler.lastError, /duplicate binding 0/);
});

function makeBasicRenderKey({ formatCode = 27, vertexStride = 0 } = {}) {
  const key = new Uint8Array(376);
  const v = new DataView(key.buffer);
  v.setUint32(20, 1, true); // 1 color target
  key.set([formatCode, 0, 0xf, 0], 24);
  v.setUint32(160, 1, true); // sampleCount = 1
  if (vertexStride > 0) {
    v.setUint32(172, 1, true); // 1 vertex buffer
    v.setUint32(176, 1, true); // 1 attribute
    v.setUint32(180, vertexStride, true);
    key.set([1, 1], 184); // stepMode = vertex, attrCount = 1
    v.setUint32(244, 0, true);
    key.set([30, 0, 0], 248); // float32x3 @0 in buffer 0
  }
  return key;
}

test('W93 CreateRenderPipeline2 populates _pipelineMeta directly and destroyResource cleans it up', () => {
  const gpu = createMockWebGpu();
  const vsPackWgsl =
    '@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }';
  const fsPackWgsl =
    '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.7, 0.4, 1.0); }';
  const vsGroup0PackWgsl =
    '@group(0) @binding(0) var<uniform> u_frame: vec4<f32>; @vertex fn vs_main() -> @builtin(position) vec4<f32> { return u_frame; }';
  const fsFallbackWgsl =
    '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 1.0, 1.0); }';
  const vsBuiltinWgsl =
    '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }';

  const standaloneProcessor = {
    device: gpu.device,
    ctx: gpu.ctx,
    shaderModules: new Map([
      [1, { module: gpu.device.createShaderModule({ code: vsPackWgsl }), code: vsPackWgsl, source: 'pack' }],
      [2, { module: gpu.device.createShaderModule({ code: fsPackWgsl }), code: fsPackWgsl, source: 'pack' }],
      [3, { module: gpu.device.createShaderModule({ code: vsGroup0PackWgsl }), code: vsGroup0PackWgsl, source: 'pack' }],
      [4, { module: gpu.device.createShaderModule({ code: fsFallbackWgsl }), code: fsFallbackWgsl, source: 'fallback' }],
      [
        5,
        {
          module: gpu.device.createShaderModule({ code: vsBuiltinWgsl }),
          code: vsBuiltinWgsl,
          source: 'builtin',
          fallbackProfile: 'fullscreen_triangle_cbv',
        },
      ],
    ]),
    pipelines: new Map(),
    computePipelines: new Map(),
    buffers: new Map(),
    textures: new Map(),
    samplers: new Map(),
    bindGroups: new Map(),
    _pipelineMeta: new Map(),
    _passCommands: [],
  };

  const handler = new rs.RenderStatePackets(standaloneProcessor);
  const dispatchRaw = (pkt) => {
    const view = new DataView(pkt.buffer, pkt.byteOffset, pkt.byteLength);
    const opcode = view.getUint16(0, true);
    const payloadBytes = view.getUint32(4, true);
    return handler.handlePacket(opcode, view, 8, payloadBytes);
  };

  const keyNoVb = makeBasicRenderKey({ formatCode: 27, vertexStride: 0 });
  const keyWithVb = makeBasicRenderKey({ formatCode: 27, vertexStride: 12 });

  // 1. Real-pack VS + FS without @group(0)
  assert.equal(dispatchRaw(renderPipeline2Packet(10, 1, 2, 0, keyWithVb)), true);
  // 2. Real-pack VS only (fsId = 0) declaring @group(0)
  assert.equal(dispatchRaw(renderPipeline2Packet(11, 3, 0, 0, keyNoVb)), true);
  // 3. Fallback pack FS (source === 'fallback')
  assert.equal(dispatchRaw(renderPipeline2Packet(12, 1, 4, 0, keyNoVb)), true);
  // 4. Builtin fallback VS (source === 'builtin')
  assert.equal(dispatchRaw(renderPipeline2Packet(13, 5, 2, 0, keyNoVb)), true);
  // 5. Real-pack VS + FS with explicit group-0 pipeline layout
  assert.equal(
    dispatchRaw(bindGroupLayoutPacket(50, [{ binding: 0, visibility: 3, type: 0, subtype: 0, minBindingSize: 16 }])),
    true
  );
  assert.equal(dispatchRaw(pipelineLayoutPacket(51, [50])), true);
  assert.equal(dispatchRaw(renderPipeline2Packet(14, 1, 2, 51, keyNoVb)), true);

  assert.equal(handler.stats.errors, 0, handler.lastError);

  assert.deepEqual(standaloneProcessor._pipelineMeta.get(10), {
    id: 10,
    vsShaderId: 1,
    fsShaderId: 2,
    layoutId: 0,
    vertexStride: 12,
    topology: 'triangle-list',
    isV2: true,
    hasReflectedLayout: false,
    isRealPack: true,
    isFallback: false,
    usesFallbackShader: false,
    expectsGroup0: false,
    fallbackProfile: null,
  });

  const meta11 = standaloneProcessor._pipelineMeta.get(11);
  assert.equal(meta11.isRealPack, true);
  assert.equal(meta11.isFallback, false);
  assert.equal(meta11.usesFallbackShader, false);
  assert.equal(meta11.expectsGroup0, true);

  const meta12 = standaloneProcessor._pipelineMeta.get(12);
  assert.equal(meta12.isRealPack, false);
  assert.equal(meta12.isFallback, false);
  assert.equal(meta12.usesFallbackShader, true);
  assert.equal(meta12.expectsGroup0, false);

  const meta13 = standaloneProcessor._pipelineMeta.get(13);
  assert.equal(meta13.isRealPack, false);
  assert.equal(meta13.isFallback, true);
  assert.equal(meta13.usesFallbackShader, true);
  assert.equal(meta13.fallbackProfile, 'fullscreen_triangle_cbv');

  const meta14 = standaloneProcessor._pipelineMeta.get(14);
  assert.equal(meta14.isRealPack, true);
  assert.equal(meta14.hasReflectedLayout, true);
  assert.equal(meta14.expectsGroup0, true);

  // destroyResource removes pipeline metadata from processor._pipelineMeta
  handler.destroyResource(10);
  assert.equal(standaloneProcessor._pipelineMeta.has(10), false);
  assert.equal(standaloneProcessor._pipelineMeta.has(11), true);
});

test('W93 CreateRenderPipeline2 with real-pack and fallback shaders increments realPackDrawCount, fallbackDrawCount, and telemetry snapshot', () => {
  const vsWgsl =
    '@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }';
  const fsWgsl =
    '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.7, 0.4, 1.0); }';
  const fallbackFsWgsl =
    '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.9, 0.1, 0.5, 1.0); }';

  const vsHash = formatHashHex(fnv1a64(vsWgsl));
  const fsHash = formatHashHex(fnv1a64(fsWgsl));
  const fallbackFsHash = '0x000000000000f093';
  const missingFsHash = '0xdeadbeef00009301';

  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: vsWgsl },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fsWgsl },
    { hash: fallbackFsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fallbackFsWgsl },
  ]);

  const key = makeBasicRenderKey({ formatCode: 27, vertexStride: 0 });

  // Case 1: Real-pack VS/FS + fallback-pack FS via CreateRenderPipeline2 (opcode 24)
  const gpu1 = createMockWebGpu();
  const processor1 = new WgpuCommandProcessor(gpu1.device, gpu1.ctx, {
    format: 'bgra8unorm',
    wgslPackBuffer: packBuf,
    fallbackShaderHashes: { fragment: fallbackFsHash },
  });
  const handler1 = installRenderStatePackets(processor1);

  const indexBufPayload = new Uint8Array(16);
  const indexBufView = new DataView(indexBufPayload.buffer);
  indexBufView.setUint32(0, 40, true); // bufferId = 40
  indexBufView.setUint32(4, 1, true); // uint16
  indexBufView.setBigUint64(8, 0n, true);

  const stream1 = concat(
    new Stream()
      .packet(OPCODES.BEGIN_FRAME)
      .shader(1, 0, vsHash)
      .shader(2, 1, fsHash)
      .shader(3, 1, missingFsHash)
      .buffer(40, 0x10 | 0x08, 16)
      .bytes(),
    renderPipeline2Packet(10, 1, 2, 0, key),
    renderPipeline2Packet(11, 1, 3, 0, key),
    new Stream()
      .u32s(OPCODES.SET_PIPELINE, [10])
      .u32s(OPCODES.DRAW, [3, 1, 0, 0])
      .packet(OPCODES.SET_INDEX_BUFFER, indexBufPayload)
      .u32s(OPCODES.DRAW_INDEXED, [3, 1, 0, 0, 0])
      .u32s(OPCODES.SET_PIPELINE, [11])
      .u32s(OPCODES.DRAW, [3, 1, 0, 0])
      .packet(OPCODES.PRESENT)
      .bytes()
  );

  const res1 = processor1.executeCommandBuffer(stream1);
  assert.equal(res1.presents, 1);
  assert.equal(handler1.stats.errors, 0, handler1.lastError);
  assert.equal(processor1.shaderModules.get(1).source, 'pack');
  assert.equal(processor1.shaderModules.get(2).source, 'pack');
  assert.equal(processor1.shaderModules.get(3).source, 'fallback');
  assert.equal(processor1._pipelineMeta.get(10).isRealPack, true);
  assert.equal(processor1._pipelineMeta.get(11).isRealPack, false);
  assert.equal(processor1._pipelineMeta.get(11).usesFallbackShader, true);

  assert.equal(processor1.realPackDrawCount, 2);
  assert.equal(processor1.fallbackDrawCount, 1);

  const snap1 = buildGpuTelemetrySnapshot(processor1, { timestamp: 9300 });
  assert.equal(snap1.frameStats.framesPresented, 1);
  assert.equal(snap1.frameStats.drawCalls, 3);
  assert.equal(snap1.frameStats.realPackDrawCount, 2);
  assert.equal(snap1.frameStats.fallbackDrawCount, 1);

  // DestroyResource cleans up _pipelineMeta
  processor1.executeCommandBuffer(new Stream().u32s(OPCODES.DESTROY_RESOURCE, [10]).bytes());
  assert.equal(processor1.pipelines.has(10), false);
  assert.equal(processor1._pipelineMeta.has(10), false);

  // Case 2: Builtin fallback shader module (no fallbackShaderHashes configured) increments fallbackDrawCount
  const gpu2 = createMockWebGpu();
  const processor2 = new WgpuCommandProcessor(gpu2.device, gpu2.ctx, {
    format: 'bgra8unorm',
    wgslPackBuffer: packBuf,
  });
  const handler2 = installRenderStatePackets(processor2);

  const stream2 = concat(
    new Stream()
      .packet(OPCODES.BEGIN_FRAME)
      .shader(1, 0, vsHash)
      .shader(2, 1, missingFsHash)
      .bytes(),
    renderPipeline2Packet(20, 1, 2, 0, key),
    new Stream()
      .u32s(OPCODES.SET_PIPELINE, [20])
      .u32s(OPCODES.DRAW, [3, 1, 0, 0])
      .packet(OPCODES.PRESENT)
      .bytes()
  );

  processor2.executeCommandBuffer(stream2);
  assert.equal(handler2.stats.errors, 0, handler2.lastError);
  assert.equal(processor2.shaderModules.get(2).source, 'builtin');
  assert.equal(processor2._pipelineMeta.get(20).isRealPack, false);
  assert.equal(processor2._pipelineMeta.get(20).isFallback, true);
  assert.equal(processor2._pipelineMeta.get(20).usesFallbackShader, true);
  assert.equal(processor2.realPackDrawCount, 0);
  assert.equal(processor2.fallbackDrawCount, 1);

  const snap2 = buildGpuTelemetrySnapshot(processor2, { timestamp: 9301 });
  assert.equal(snap2.frameStats.realPackDrawCount, 0);
  assert.equal(snap2.frameStats.fallbackDrawCount, 1);
});

// ---------------------------------------------------------------------------------------------
// W97: WGSL @group(0) layout compatibility check, layout:'auto' fallback, and bind-group remap
// ---------------------------------------------------------------------------------------------

test('W97 parseWgslGroup0Bindings parses multi-line Naga WGSL, reversed attributes, storage/texture kinds, and skips unused bindings', () => {
  const nagaWgsl = [
    '// @group(0) @binding(99) var<uniform> commented_out: vec4<f32>;',
    '/* @group(0) @binding(98) var<uniform> block_commented: vec4<f32>; */',
    'struct Rdna2Cbv { v: vec4<f32> };',
    '@group(0) @binding(0)',
    'var<uniform> cb0_: Rdna2Cbv;',
    '@binding(2) @group(0)',
    'var samp0_: sampler;',
    '@group(0) @binding(1)',
    'var tex0_: texture_2d<f32>;',
    '@group(0) @binding(3)',
    'var<storage, read> src_buf: array<u32>;',
    '@group(0) @binding(4)',
    'var<storage, read_write> dst_buf: array<u32>;',
    '@group(0) @binding(5)',
    'var out_tex: texture_storage_2d<rgba8unorm,write>;',
    '@group(0) @binding(6)',
    'var shadow_tex: texture_depth_cube;',
    '@group(0) @binding(7)',
    'var shadow_samp: sampler_comparison;',
    '@group(0) @binding(8)',
    'var msaa_tex: texture_multisampled_2d<f32>;',
    '@group(0) @binding(9)',
    'var arr_storage: texture_storage_2d_array<rgba16float, write>;',
    '@group(0) @binding(10)',
    'var<uniform> unused_cb_: vec4<f32>;',
    '@group(1) @binding(0)',
    'var<uniform> group1_cb_: vec4<f32>;',
    '',
    '@compute @workgroup_size(1, 1, 1)',
    'fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {',
    '  let a = cb0_.v;',
    '  let b = textureSampleLevel(tex0_, samp0_, vec2<f32>(0.0), 0.0);',
    '  let c = src_buf[0];',
    '  dst_buf[0] = c;',
    '  textureStore(out_tex, vec2<i32>(0, 0), a + b);',
    '  let d = textureSampleCompareLevel(shadow_tex, shadow_samp, vec3<f32>(0.0), 0.5);',
    '  let e = textureLoad(msaa_tex, vec2<i32>(0, 0), 0);',
    '  textureStore(arr_storage, vec2<i32>(0, 0), 0, e + vec4<f32>(d));',
    '  _ = group1_cb_;',
    '}',
  ].join('\n');

  const parsed = rs.parseWgslGroup0Bindings(nagaWgsl);
  assert.deepEqual(parsed, [
    { binding: 0, kind: 'uniform' },
    { binding: 1, kind: 'texture', viewDimension: '2d' },
    { binding: 2, kind: 'sampler' },
    { binding: 3, kind: 'read-only-storage' },
    { binding: 4, kind: 'storage' },
    { binding: 5, kind: 'storage-texture', viewDimension: '2d', storageFormat: 'rgba8unorm' },
    { binding: 6, kind: 'texture', viewDimension: 'cube', sampleType: 'depth' },
    { binding: 7, kind: 'sampler', comparison: true },
    { binding: 8, kind: 'texture', viewDimension: '2d', multisampled: true },
    { binding: 9, kind: 'storage-texture', viewDimension: '2d-array', storageFormat: 'rgba16float' },
  ]);

  // Empty / non-string / zero-used-bindings inputs return []
  assert.deepEqual(rs.parseWgslGroup0Bindings(''), []);
  assert.deepEqual(rs.parseWgslGroup0Bindings(null), []);
  assert.deepEqual(
    rs.parseWgslGroup0Bindings(
      '@group(0) @binding(0) var<uniform> dead_cb: vec4<f32>;\n@fragment fn fs_main() {}'
    ),
    []
  );
  // Header-only snippet (no fn body) still extracts declared bindings
  assert.deepEqual(
    rs.parseWgslGroup0Bindings('@group(0) @binding(0)\nvar<uniform> cb0_: vec4<f32>;'),
    [{ binding: 0, kind: 'uniform' }]
  );
  // Member access `.unused_cb` must not count as a static reference to global `unused_cb`
  assert.deepEqual(
    rs.parseWgslGroup0Bindings(
      'struct S { unused_cb: f32 }; @group(0) @binding(0) var<uniform> unused_cb: vec4<f32>; fn fs_main(s: S) -> f32 { return s.unused_cb; }'
    ),
    []
  );
});

test('W97 isExplicitLayoutCompatibleWithWgsl detects type mismatches, missing bindings, and accepts compatible/manifest layouts', () => {
  const wgsl3 = [
    { binding: 0, kind: 'uniform' },
    { binding: 1, kind: 'texture', viewDimension: '2d' },
    { binding: 2, kind: 'sampler' },
  ];
  const fullGroup0 = [
    { binding: 0, visibility: 3, buffer: { type: 'uniform' } },
    { binding: 1, visibility: 3, texture: { sampleType: 'float', viewDimension: '2d' } },
    { binding: 2, visibility: 3, sampler: { type: 'filtering' } },
    { binding: 3, visibility: 6, buffer: { type: 'storage' } },
    { binding: 4, visibility: 6, storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' } },
  ];

  // Exact/superset match
  assert.equal(rs.isExplicitLayoutCompatibleWithWgsl(fullGroup0, wgsl3), true);
  // Empty WGSL bindings or null manifest group0Entries are treated as compatible
  assert.equal(rs.isExplicitLayoutCompatibleWithWgsl(fullGroup0, []), true);
  assert.equal(rs.isExplicitLayoutCompatibleWithWgsl(null, wgsl3), true);
  // RS-4 style: layout only has binding 0 CBV while WGSL needs bindings 0, 1, 2
  assert.equal(
    rs.isExplicitLayoutCompatibleWithWgsl([{ binding: 0, visibility: 3, buffer: { type: 'uniform' } }], wgsl3),
    false
  );
  // RS-8 style: layout has texture at 0 while WGSL has uniform at 0
  assert.equal(
    rs.isExplicitLayoutCompatibleWithWgsl(
      [
        { binding: 0, visibility: 4, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 1, visibility: 4, buffer: { type: 'uniform' } },
        { binding: 2, visibility: 4, sampler: { type: 'filtering' } },
      ],
      wgsl3
    ),
    false
  );
  // Storage vs read-only-storage rules:
  // 'storage' layout entry satisfies 'read-only-storage' WGSL binding, but NOT vice versa.
  assert.equal(
    rs.isExplicitLayoutCompatibleWithWgsl(
      [{ binding: 0, visibility: 4, buffer: { type: 'storage' } }],
      [{ binding: 0, kind: 'read-only-storage' }]
    ),
    true
  );
  assert.equal(
    rs.isExplicitLayoutCompatibleWithWgsl(
      [{ binding: 0, visibility: 4, buffer: { type: 'read-only-storage' } }],
      [{ binding: 0, kind: 'storage' }]
    ),
    false
  );
  // Storage texture & storage buffer match
  assert.equal(
    rs.isExplicitLayoutCompatibleWithWgsl(fullGroup0, [
      { binding: 3, kind: 'storage' },
      { binding: 4, kind: 'storage-texture', storageFormat: 'rgba8unorm', viewDimension: '2d' },
    ]),
    true
  );
  assert.equal(
    rs.isExplicitLayoutCompatibleWithWgsl(fullGroup0, [{ binding: 1, kind: 'storage-texture' }]),
    false
  );
});

test('W97 compute pipeline with RS-8 mismatching layout falls back to layout:auto and remaps setBindGroup(0) with usage-safe resources and dummies', () => {
  const gpu = createMockWebGpu();
  const autoBglMap = new Map();
  const origCreateCompute = gpu.device.createComputePipeline.bind(gpu.device);
  gpu.device.createComputePipeline = (desc) => {
    const pipe = origCreateCompute(desc);
    const autoBgl = { kind: 'GPUBindGroupLayout', autoFor: pipe.handle };
    autoBglMap.set(pipe, autoBgl);
    pipe.getBindGroupLayout = (groupIndex) => {
      if (groupIndex !== 0) throw new Error(`invalid group ${groupIndex}`);
      return autoBgl;
    };
    return pipe;
  };

  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  // Naga-style compute WGSL: cb0@0, tex0@1, samp0@2, dstBuf@3, outTex@4, unused@5 (stripped!)
  const csWgsl = [
    'struct Rdna2Cbv { params: vec4<f32> };',
    '@group(0) @binding(0)',
    'var<uniform> cb0_: Rdna2Cbv;',
    '@group(0) @binding(1)',
    'var tex0_: texture_2d<f32>;',
    '@group(0) @binding(2)',
    'var samp0_: sampler;',
    '@group(0) @binding(3)',
    'var<storage, read_write> dst_buf: array<u32>;',
    '@group(0) @binding(4)',
    'var out_tex: texture_storage_2d<rgba8unorm,write>;',
    '@group(0) @binding(5)',
    'var<uniform> unused_cb: vec4<f32>;',
    '@compute @workgroup_size(8, 8, 1)',
    'fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {',
    '  let t = textureSampleLevel(tex0_, samp0_, vec2<f32>(0.0), 0.0);',
    '  dst_buf[gid.x] = u32(cb0_.params.x);',
    '  textureStore(out_tex, vec2<i32>(0, 0), t + cb0_.params);',
    '}',
  ].join('\n');

  // RS-8 style mismatched layout: SRV texture at 0, STORAGE-only buffer at 1, UNIFORM buffer at 2, sampler at 3
  // (No storage texture in RS-8 -> binding 4 must fall back to dummy storage texture!)
  const rs8Entries = [
    { binding: 0, visibility: 4, type: 2, subtype: 0, dimension: 2 }, // texture at 0 (mismatches WGSL uniform@0!)
    { binding: 1, visibility: 4, type: 0, subtype: 1, minBindingSize: 64 }, // storage buffer at 1
    { binding: 2, visibility: 4, type: 0, subtype: 0, minBindingSize: 64 }, // uniform buffer at 2
    { binding: 3, visibility: 4, type: 1, subtype: 0 }, // sampler at 3
  ];

  const stream = concat(
    new Stream()
      .shader(10, 2, csWgsl)
      .buffer(101, 0x0080 | 0x0008, 256) // STORAGE-only buffer (NO UNIFORM bit!)
      .buffer(102, 0x0040 | 0x0008, 256) // UNIFORM-only buffer (NO STORAGE bit!)
      .bytes(),
    createTexture2Packet({ id: 100, width: 32, height: 32, format: formatCodeOf('rgba8unorm'), usage: 0x04 }),
    createSampler2Packet(103, { filter: 1 }),
    bindGroupLayoutPacket(200, rs8Entries),
    pipelineLayoutPacket(201, [200]),
    createComputePipeline2Packet(300, 10, 201),
    bindGroup2Packet(400, 200, [
      { binding: 0, type: 1, id: 100 },
      { binding: 1, type: 0, id: 101, offset: 0, size: 256 },
      { binding: 2, type: 0, id: 102, offset: 0, size: 256 },
      { binding: 3, type: 2, id: 103 },
    ]),
    new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [300]).bytes(),
    setBindGroup2Packet(0, 400, []),
    new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [4, 4, 1]).bytes(),
    // Second dispatch with same bind group should reuse cached remapped bind group
    setBindGroup2Packet(0, 400, []),
    new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [2, 2, 1]).packet(OPCODES.PRESENT).bytes()
  );

  const res = processor.executeCommandBuffer(stream);
  assert.equal(res.presents, 1);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  const [cp] = gpu.results('createComputePipeline');
  assert.equal(cp.desc.layout, 'auto', 'mismatched RS-8 compute pipeline must fall back to layout:auto');
  assert.equal(handler.telemetry.autoLayoutFallbacks, 1);
  assert.equal(handler.telemetry.rootSigGroup0Layouts, 0);
  assert.ok(
    handler.getTelemetry().warnings.some((w) => /compute pipeline 300.*incompatible.*layout:'auto'/.test(w))
  );
  const cpMeta = handler.computePipelineMeta.get(300);
  assert.equal(cpMeta.autoLayoutRemap, true);
  assert.deepEqual(
    cpMeta.wgslBindings.map((b) => b.binding),
    [0, 1, 2, 3, 4],
    'unused binding 5 must be excluded from wgslBindings'
  );

  // createBindGroup called twice total: 1) original RS-8 bindGroup, 2) cached remapped bindGroup
  const bgs = gpu.results('createBindGroup');
  assert.equal(bgs.length, 2, 'remapped bind group must be cached across repeated setBindGroup(0) calls');
  const remappedBg = bgs[1];
  assert.equal(remappedBg.desc.layout, autoBglMap.get(cp.result));
  assert.deepEqual(
    remappedBg.desc.entries.map((e) => e.binding),
    [0, 1, 2, 3, 4]
  );

  const storageBuf = processor.buffers.get(101);
  const uniformBuf = processor.buffers.get(102);
  const realSampler = processor.samplers.get(103);
  // Binding 0 (uniform) got real UNIFORM buffer (102), NOT storage-only buffer (101)!
  assert.equal(remappedBg.desc.entries[0].resource.buffer, uniformBuf);
  // Binding 1 (texture) got real texture 100 view
  assert.equal(remappedBg.desc.entries[1].resource.textureHandle, processor.textures.get(100).handle);
  // Binding 2 (sampler) got real sampler 103
  assert.equal(remappedBg.desc.entries[2].resource, realSampler);
  // Binding 3 (storage) got real STORAGE buffer (101), NOT uniform-only buffer (102)!
  assert.equal(remappedBg.desc.entries[3].resource.buffer, storageBuf);
  // Binding 4 (storage-texture) fell back to dummy storage texture (different handle from real texture 100)
  assert.notEqual(remappedBg.desc.entries[4].resource.textureHandle, processor.textures.get(100).handle);

  // Verify compute pass actually bound the remapped bind group (not the original RS-8 bind group)
  const setBgCalls = gpu.passCalls.filter((c) => c.pass === 'compute' && c.op === 'setBindGroup');
  assert.equal(setBgCalls.length, 2);
  assert.equal(setBgCalls[0].args[0], 0);
  assert.equal(setBgCalls[0].args[1], remappedBg.result);
  assert.equal(setBgCalls[1].args[1], remappedBg.result);

  // Destroying bind group 400 clears bindGroupDescriptors and invalidates remapped cache
  processor.executeCommandBuffer(new Stream().u32s(OPCODES.DESTROY_RESOURCE, [400]).bytes());
  assert.equal(handler.bindGroupDescriptors.has(400), false);
  assert.equal(handler.remappedBindGroups.has('c:300:400'), false);

  // Dispatching WITHOUT a prior setBindGroup(0) synthesizes a dummy-filled remapped bind group
  const noBgFrame = concat(
    new Stream()
      .packet(OPCODES.BEGIN_FRAME)
      .u32s(OPCODES.SET_COMPUTE_PIPELINE, [300])
      .u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1])
      .packet(OPCODES.PRESENT)
      .bytes()
  );
  processor.executeCommandBuffer(noBgFrame);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(gpu.results('createBindGroup').length, 3);
  const autoDummyBg = gpu.results('createBindGroup')[2];
  assert.equal(autoDummyBg.desc.layout, autoBglMap.get(cp.result));
  assert.deepEqual(
    autoDummyBg.desc.entries.map((e) => e.binding),
    [0, 1, 2, 3, 4]
  );
});

test('W97 compute remap respects buffer usage flags (storage-only never bound to uniform) and handles throwing or absent getBindGroupLayout', () => {
  // Case A: Bind group only provides a STORAGE-only buffer; WGSL wants uniform@0 -> must use dummy buffer!
  const gpu = createMockWebGpu();
  const origCreateCompute = gpu.device.createComputePipeline.bind(gpu.device);
  gpu.device.createComputePipeline = (desc) => {
    const pipe = origCreateCompute(desc);
    pipe.getBindGroupLayout = () => ({ kind: 'GPUBindGroupLayout', autoFor: pipe.handle });
    return pipe;
  };
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = installRenderStatePackets(processor);

  const csUniformOnly = [
    '@group(0) @binding(0) var<uniform> cb0: vec4<f32>;',
    '@compute @workgroup_size(1, 1, 1) fn cs_main() { let x = cb0; _ = x; }',
  ].join('\n');

  processor.executeCommandBuffer(
    concat(
      new Stream()
        .shader(1, 2, csUniformOnly)
        .buffer(10, 0x0080 | 0x0008, 256) // STORAGE-only, NO UNIFORM usage!
        .bytes(),
      bindGroupLayoutPacket(20, [{ binding: 0, visibility: 4, type: 0, subtype: 1, minBindingSize: 16 }]),
      pipelineLayoutPacket(21, [20]),
      createComputePipeline2Packet(30, 1, 21),
      bindGroup2Packet(40, 20, [{ binding: 0, type: 0, id: 10, offset: 0, size: 256 }]),
      new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [30]).bytes(),
      setBindGroup2Packet(0, 40, []),
      new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1]).packet(OPCODES.PRESENT).bytes()
    )
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  const remapped = gpu.results('createBindGroup')[1];
  assert.notEqual(
    remapped.desc.entries[0].resource.buffer,
    processor.buffers.get(10),
    'storage-only buffer must not be bound to a uniform binding'
  );

  // Case B: getBindGroupLayout(0) throws (Dawn stripped group 0) -> setBindGroup(0) is skipped cleanly!
  const gpuThrow = createMockWebGpu();
  const origCreateCompute2 = gpuThrow.device.createComputePipeline.bind(gpuThrow.device);
  gpuThrow.device.createComputePipeline = (desc) => {
    const pipe = origCreateCompute2(desc);
    pipe.getBindGroupLayout = () => {
      throw new Error('Binding group index 0 does not exist');
    };
    return pipe;
  };
  const procThrow = new WgpuCommandProcessor(gpuThrow.device, gpuThrow.ctx);
  const handlerThrow = installRenderStatePackets(procThrow);
  procThrow.executeCommandBuffer(
    concat(
      new Stream()
        .shader(1, 2, csUniformOnly)
        .buffer(10, 0x0080 | 0x0008, 256)
        .bytes(),
      bindGroupLayoutPacket(20, [{ binding: 0, visibility: 4, type: 0, subtype: 1, minBindingSize: 16 }]),
      pipelineLayoutPacket(21, [20]),
      createComputePipeline2Packet(30, 1, 21),
      bindGroup2Packet(40, 20, [{ binding: 0, type: 0, id: 10, offset: 0, size: 256 }]),
      new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [30]).bytes(),
      setBindGroup2Packet(0, 40, []),
      new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1]).packet(OPCODES.PRESENT).bytes()
    )
  );
  assert.equal(handlerThrow.stats.errors, 0, handlerThrow.lastError);
  assert.equal(
    gpuThrow.passCalls.filter((c) => c.op === 'setBindGroup').length,
    0,
    'setBindGroup(0) must be skipped when getBindGroupLayout(0) throws'
  );

  // Case C: Mock device without getBindGroupLayout on pipeline -> does not crash!
  const gpuNoGetBgl = createMockWebGpu();
  const procNoGetBgl = new WgpuCommandProcessor(gpuNoGetBgl.device, gpuNoGetBgl.ctx);
  const handlerNoGetBgl = installRenderStatePackets(procNoGetBgl);
  assert.doesNotThrow(() => {
    procNoGetBgl.executeCommandBuffer(
      concat(
        new Stream()
          .shader(1, 2, csUniformOnly)
          .buffer(10, 0x0080 | 0x0008, 256)
          .bytes(),
        bindGroupLayoutPacket(20, [{ binding: 0, visibility: 4, type: 0, subtype: 1, minBindingSize: 16 }]),
        pipelineLayoutPacket(21, [20]),
        createComputePipeline2Packet(30, 1, 21),
        bindGroup2Packet(40, 20, [{ binding: 0, type: 0, id: 10, offset: 0, size: 256 }]),
        new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [30]).bytes(),
        setBindGroup2Packet(0, 40, []),
        new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1]).packet(OPCODES.PRESENT).bytes()
      )
    );
  });
  assert.equal(handlerNoGetBgl.stats.errors, 0, handlerNoGetBgl.lastError);
});

test('W97 render pipeline: RS-4 mismatch falls back to layout:auto and remaps group 0, while compatible PSO #50450 preserves explicit PipelineLayout untouched', () => {
  const gpu = createMockWebGpu();
  const autoBglMap = new Map();
  const origCreateRender = gpu.device.createRenderPipeline.bind(gpu.device);
  gpu.device.createRenderPipeline = (desc) => {
    const pipe = origCreateRender(desc);
    const autoBgl = { kind: 'GPUBindGroupLayout', autoFor: pipe.handle };
    autoBglMap.set(pipe, autoBgl);
    pipe.getBindGroupLayout = (groupIndex) => {
      if (groupIndex !== 0) throw new Error(`invalid group ${groupIndex}`);
      return autoBgl;
    };
    return pipe;
  };

  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);
  const key = buildFrame1PipelineKey();

  const vsWgsl = [
    'struct Rdna2Cbv { m: vec4<f32> };',
    '@group(0) @binding(0)',
    'var<uniform> cb0_: Rdna2Cbv;',
    '@vertex fn vs_main(@location(0) p: vec3<f32>, @location(1) c: vec4<f32>, @location(2) uv: vec2<f32>) -> @builtin(position) vec4<f32> {',
    '  return vec4<f32>(p + cb0_.m.xyz, 1.0);',
    '}',
  ].join('\n');
  const fsWgsl = [
    '@group(0) @binding(1)',
    'var tex0_: texture_2d<f32>;',
    '@group(0) @binding(2)',
    'var samp0_: sampler;',
    '@fragment fn fs_main() -> @location(0) vec4<f32> {',
    '  return textureSample(tex0_, samp0_, vec2<f32>(0.5, 0.5));',
    '}',
  ].join('\n');

  // RS-4 layout: only 1 CBV (binding 0, with dynamic offset) -> INCOMPATIBLE with tex0@1 & samp0@2!
  const rs4Entries = [{ binding: 0, visibility: 3, type: 0, subtype: 0, flags: 0x1, minBindingSize: 16 }];
  // PSO #50450 layout: CBV@0 (dynamic offset), tex0@1, samp0@2 -> COMPATIBLE!
  const pso50450Entries = [
    { binding: 0, visibility: 3, type: 0, subtype: 0, flags: 0x1, minBindingSize: 16 },
    { binding: 1, visibility: 2, type: 2, subtype: 0, dimension: 2 },
    { binding: 2, visibility: 2, type: 1, subtype: 0 },
  ];

  const stream = concat(
    new Stream()
      .shader(1, 0, vsWgsl)
      .shader(2, 1, fsWgsl)
      .buffer(100, 0x0040 | 0x0008, 256) // CBV
      .buffer(105, 0x0020 | 0x0008, 72) // VB
      .bytes(),
    createTexture2Packet({ id: 101, width: 32, height: 32, format: formatCodeOf('rgba8unorm'), usage: 0x04 }),
    createSampler2Packet(102, { filter: 1 }),
    bindGroupLayoutPacket(200, rs4Entries),
    pipelineLayoutPacket(201, [200]),
    bindGroupLayoutPacket(210, pso50450Entries),
    pipelineLayoutPacket(211, [210]),
    // PSO #50451 with RS-4 (mismatched -> auto fallback)
    renderPipeline2Packet(50451, 1, 2, 201, key),
    // PSO #50450 with matching explicit layout (untouched!)
    renderPipeline2Packet(50450, 1, 2, 211, key),
    bindGroup2Packet(400, 200, [{ binding: 0, type: 0, id: 100, offset: 0, size: 256 }]),
    bindGroup2Packet(410, 210, [
      { binding: 0, type: 0, id: 100, offset: 0, size: 256 },
      { binding: 1, type: 1, id: 101 },
      { binding: 2, type: 2, id: 102 },
    ]),
    new Stream().packet(OPCODES.BEGIN_FRAME).bytes(),
    beginRenderPassPacket({ colors: [{ textureId: SWAPCHAIN_TEXTURE_ID }] }),
    // Draw 1: PSO #50451 (autoLayoutRemap) with RS-4 bind group (dynamic offset [0] must be dropped on auto layout!)
    new Stream().u32s(OPCODES.SET_PIPELINE, [50451]).bytes(),
    setBindGroup2Packet(0, 400, [0]),
    new Stream().u32s(OPCODES.DRAW, [3, 1, 0, 0]).bytes(),
    // Draw 2: PSO #50450 (explicit layout preserved untouched with dynamic offset [0]!)
    new Stream().u32s(OPCODES.SET_PIPELINE, [50450]).bytes(),
    setBindGroup2Packet(0, 410, [0]),
    new Stream().u32s(OPCODES.DRAW, [3, 1, 0, 0]).bytes(),
    header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)),
    new Stream().packet(OPCODES.PRESENT).bytes()
  );

  const res = processor.executeCommandBuffer(stream);
  assert.equal(res.presents, 1);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(processor.skippedDrawCount, 0, 'neither PSO #50451 nor PSO #50450 draw should be skipped');

  const [plRs4, pl50450] = gpu.results('createPipelineLayout');
  const [pipe50451, pipe50450] = gpu.results('createRenderPipeline');

  // PSO #50451 fell back to layout:'auto'
  assert.equal(pipe50451.desc.layout, 'auto');
  assert.equal(handler.renderPipelineAutoLayoutMeta.has(50451), true);
  assert.equal(handler.renderPipelineAutoLayoutMeta.get(50451).autoLayoutRemap, true);
  assert.deepEqual(
    handler.renderPipelineAutoLayoutMeta.get(50451).wgslBindings.map((b) => b.binding),
    [0, 1, 2]
  );

  // PSO #50450 kept its explicit PipelineLayout untouched!
  assert.equal(pipe50450.desc.layout, pl50450.result);
  assert.notEqual(plRs4.result, pl50450.result);
  assert.equal(handler.renderPipelineAutoLayoutMeta.has(50450), false);
  assert.equal(handler.telemetry.autoLayoutFallbacks, 1);
  assert.equal(handler.telemetry.rootSigGroup0Layouts, 1);

  // Verify bind groups: [0]=bg400(RS-4), [1]=bg410(50450), [2]=remapped for 50451
  const bgs = gpu.results('createBindGroup');
  assert.equal(bgs.length, 3);
  const remapped50451 = bgs[2];
  assert.equal(remapped50451.desc.layout, autoBglMap.get(pipe50451.result));
  assert.deepEqual(
    remapped50451.desc.entries.map((e) => e.binding),
    [0, 1, 2]
  );
  // Real CBV buffer preserved at binding 0; dummy texture & dummy sampler synthesized for 1 & 2
  assert.equal(remapped50451.desc.entries[0].resource.buffer, processor.buffers.get(100));
  assert.equal(remapped50451.desc.entries[1].resource.kind, 'GPUTextureView');
  assert.equal(remapped50451.desc.entries[2].resource.kind, 'GPUSampler');

  const renderSetBgs = gpu.passCalls.filter((c) => c.pass === 'render' && c.op === 'setBindGroup');
  assert.equal(renderSetBgs.length, 2);
  // PSO #50451 received remapped bind group WITHOUT dynamic offsets
  assert.deepEqual(renderSetBgs[0].args, [0, remapped50451.result]);
  // PSO #50450 received original bind group 410 WITH dynamic offset [0] untouched
  assert.deepEqual(renderSetBgs[1].args, [0, bgs[1].result, [0]]);
});

// ---------------------------------------------------------------------------------------------
// W98: Bind group resource-kind guard & runtime shape/format guards in auto-layout remapper
// ---------------------------------------------------------------------------------------------

test('W98 BUFFER bound to layoutEntry.texture/storageTexture substitutes dummy view for record.layout while preserving real buffer for W97 auto-layout remap', () => {
  const gpu = createMockWebGpu();
  const autoBglMap = new Map();
  const origCreateCompute = gpu.device.createComputePipeline.bind(gpu.device);
  gpu.device.createComputePipeline = (desc) => {
    const pipe = origCreateCompute(desc);
    const autoBgl = { kind: 'GPUBindGroupLayout', autoFor: pipe.handle };
    autoBglMap.set(pipe, autoBgl);
    pipe.getBindGroupLayout = (groupIndex) => {
      if (groupIndex !== 0) throw new Error(`invalid group ${groupIndex}`);
      return autoBgl;
    };
    return pipe;
  };

  // Strict WebGPU createBindGroup validator mimicking Dawn's resource-kind checks.
  const origCreateBg = gpu.device.createBindGroup.bind(gpu.device);
  gpu.device.createBindGroup = (desc) => {
    const layoutEntries = (desc.layout && desc.layout.desc && desc.layout.desc.entries) || [];
    for (const e of desc.entries) {
      const le = layoutEntries.find((x) => x.binding === e.binding);
      if (!le) continue;
      if (le.texture || le.storageTexture) {
        if (!e.resource || (typeof e.resource === 'object' && 'buffer' in e.resource) || e.resource.kind !== 'GPUTextureView') {
          throw new Error(`Binding entry textureView not set at binding ${e.binding}`);
        }
      } else if (le.buffer) {
        if (!e.resource || typeof e.resource !== 'object' || !('buffer' in e.resource) || !e.resource.buffer) {
          throw new Error(`Binding entry buffer not set at binding ${e.binding}`);
        }
      } else if (le.sampler) {
        if (!e.resource || e.resource.kind !== 'GPUSampler') {
          throw new Error(`Binding entry sampler not set at binding ${e.binding}`);
        }
      }
    }
    return origCreateBg(desc);
  };

  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  // Compute WGSL with read-only storage buffer @0, texture_2d<f32> sampled via textureLoad @1,
  // and uniform buffer @2.
  const csWgsl = [
    '@group(0) @binding(0) var<storage, read> structured_srv: array<u32>;',
    '@group(0) @binding(1) var tex_unfilt: texture_2d<f32>;',
    '@group(0) @binding(2) var<uniform> cb0: vec4<f32>;',
    '@compute @workgroup_size(8, 1, 1)',
    'fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {',
    '  let v = structured_srv[gid.x];',
    '  let t = textureLoad(tex_unfilt, vec2<i32>(i32(v), 0), 0);',
    '  _ = t + cb0;',
    '}',
  ].join('\n');

  // Root-signature layout declares SRV binding 4 as texture_2d (matching D3D12 SRV -> texture_2d)
  // and binding 5 as storageTexture, plus binding 0 as uniform buffer.
  const bglEntries = [
    { binding: 0, visibility: 4, type: 0, subtype: 0, minBindingSize: 16 }, // uniform buffer
    { binding: 4, visibility: 4, type: 2, subtype: 0, dimension: 2 }, // texture (StructuredBuffer SRV bound here!)
    { binding: 5, visibility: 4, type: 3, subtype: 0, dimension: 2, storageFormat: formatCodeOf('rgba8unorm') }, // storageTexture (RWStructuredBuffer UAV bound here!)
  ];

  const stream = concat(
    new Stream()
      .shader(10, 2, csWgsl)
      .buffer(101, 0x0080 | 0x0008, 512) // STORAGE-only StructuredBuffer SRV
      .buffer(102, 0x0040 | 0x0080 | 0x0008, 256) // UNIFORM | STORAGE buffer
      .buffer(103, 0x0080 | 0x0008, 128) // STORAGE-only buffer bound to uniform slot 0 (usage mismatch!)
      .bytes(),
    bindGroupLayoutPacket(200, bglEntries),
    pipelineLayoutPacket(201, [200]),
    createComputePipeline2Packet(300, 10, 201),
    // Bind STORAGE-only buffer 103 to uniform@0 (usage mismatch),
    // StructuredBuffer SRV 101 to texture@4 (kind mismatch!),
    // UNIFORM|STORAGE buffer 102 to storageTexture@5 (kind mismatch!).
    bindGroup2Packet(400, 200, [
      { binding: 0, type: 0, id: 103, offset: 0, size: 128 },
      { binding: 4, type: 0, id: 101, offset: 0, size: 512 },
      { binding: 5, type: 0, id: 102, offset: 0, size: 256 },
    ]),
    new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [300]).bytes(),
    setBindGroup2Packet(0, 400, []),
    new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1]).packet(OPCODES.PRESENT).bytes()
  );

  const res = processor.executeCommandBuffer(stream);
  assert.equal(res.presents, 1);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  const bgs = gpu.results('createBindGroup');
  assert.equal(bgs.length, 2, 'expected original bind group + auto-layout remapped bind group');

  // 1) Original bind group against record.layout must have substituted dummies for all mismatched slots!
  const origBg = bgs[0];
  assert.notEqual(
    origBg.desc.entries[0].resource.buffer,
    processor.buffers.get(103),
    'STORAGE-only buffer bound to uniform layout slot must receive dummy buffer'
  );
  assert.equal(
    origBg.desc.entries[1].resource.kind,
    'GPUTextureView',
    'BUFFER bound to layoutEntry.texture slot must receive dummy GPUTextureView, never { buffer }'
  );
  assert.equal('buffer' in origBg.desc.entries[1].resource, false);
  assert.equal(
    origBg.desc.entries[2].resource.kind,
    'GPUTextureView',
    'BUFFER bound to layoutEntry.storageTexture slot must receive dummy GPUTextureView'
  );

  // 2) descriptorResources must preserve the real buffers classified by their ACTUAL type/usage!
  const bgDesc = handler.bindGroupDescriptors.get(400);
  assert.ok(bgDesc);
  assert.equal(bgDesc.resources.length, 3);
  assert.equal(bgDesc.resources[0].kind, 'storage');
  assert.equal(bgDesc.resources[0].isDummy, false);
  assert.equal(bgDesc.resources[0].resource.buffer, processor.buffers.get(103));
  assert.equal(bgDesc.resources[1].kind, 'storage');
  assert.equal(bgDesc.resources[1].isDummy, false);
  assert.equal(bgDesc.resources[1].resource.buffer, processor.buffers.get(101));
  assert.equal(bgDesc.resources[2].kind, 'uniform');
  assert.equal(bgDesc.resources[2].isDummy, false);
  assert.equal(bgDesc.resources[2].resource.buffer, processor.buffers.get(102));

  // 3) W97 auto-layout remapped bind group:
  //    - binding 0 (var<storage, read>) gets a real STORAGE buffer (103 or 101)
  //    - binding 1 (texture_2d<f32>) NEVER gets { buffer }; receives a valid dummy GPUTextureView!
  //    - binding 2 (var<uniform>) gets real UNIFORM|STORAGE buffer 102!
  const remappedBg = bgs[1];
  assert.equal(remappedBg.desc.layout, autoBglMap.get(gpu.results('createComputePipeline')[0].result));
  assert.equal(remappedBg.desc.entries[0].resource.buffer, processor.buffers.get(103));
  assert.equal(remappedBg.desc.entries[1].resource.kind, 'GPUTextureView');
  assert.equal('buffer' in remappedBg.desc.entries[1].resource, false);
  assert.equal(remappedBg.desc.entries[2].resource.buffer, processor.buffers.get(102));

  // Deduplicated telemetry warnings emitted for kind mismatch and buffer usage mismatch
  const warnings = handler.getTelemetry().warnings;
  assert.ok(warnings.some((w) => /binding 4 buffer resource mismatches layout entry/.test(w)));
  assert.ok(warnings.some((w) => /binding 0 buffer usage .* lacks required uniform usage/.test(w)));
});

test('W98 TEXTURE_VIEW and SAMPLER bound to layoutEntry.buffer substitute dummy buffers while preserving real texture/sampler for auto-layout remap', () => {
  const gpu = createMockWebGpu();
  const origCreateCompute = gpu.device.createComputePipeline.bind(gpu.device);
  gpu.device.createComputePipeline = (desc) => {
    const pipe = origCreateCompute(desc);
    pipe.getBindGroupLayout = () => ({ kind: 'GPUBindGroupLayout', autoFor: pipe.handle });
    return pipe;
  };
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  const csWgsl = [
    '@group(0) @binding(0) var tex0: texture_2d<f32>;',
    '@group(0) @binding(1) var samp0: sampler;',
    '@compute @workgroup_size(1, 1, 1)',
    'fn cs_main() { let c = textureSampleLevel(tex0, samp0, vec2<f32>(0.5), 0.0); _ = c; }',
  ].join('\n');

  // Layout expects two uniform buffers at 0 and 1, but packet binds a TEXTURE_VIEW at 0 and SAMPLER at 1.
  const bglEntries = [
    { binding: 0, visibility: 4, type: 0, subtype: 0, minBindingSize: 16 },
    { binding: 1, visibility: 4, type: 0, subtype: 0, minBindingSize: 16 },
  ];

  const stream = concat(
    new Stream().shader(10, 2, csWgsl).bytes(),
    createTexture2Packet({ id: 50, width: 16, height: 16, format: formatCodeOf('rgba8unorm'), usage: 0x04 }),
    createSampler2Packet(51, { filter: 1 }),
    bindGroupLayoutPacket(200, bglEntries),
    pipelineLayoutPacket(201, [200]),
    createComputePipeline2Packet(300, 10, 201),
    bindGroup2Packet(400, 200, [
      { binding: 0, type: 1, id: 50 },
      { binding: 1, type: 2, id: 51 },
    ]),
    new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [300]).bytes(),
    setBindGroup2Packet(0, 400, []),
    new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1]).packet(OPCODES.PRESENT).bytes()
  );

  processor.executeCommandBuffer(stream);
  assert.equal(handler.stats.errors, 0, handler.lastError);

  const [origBg, remappedBg] = gpu.results('createBindGroup');
  // Original bind group received dummy { buffer } objects at both buffer slots
  assert.ok(origBg.desc.entries[0].resource && 'buffer' in origBg.desc.entries[0].resource);
  assert.ok(origBg.desc.entries[1].resource && 'buffer' in origBg.desc.entries[1].resource);

  // descriptorResources preserved real GPUTextureView as 'texture' and real GPUSampler as 'sampler'
  const bgDesc = handler.bindGroupDescriptors.get(400);
  assert.equal(bgDesc.resources[0].kind, 'texture');
  assert.equal(bgDesc.resources[0].isDummy, false);
  assert.equal(bgDesc.resources[0].resource.textureHandle, processor.textures.get(50).handle);
  assert.equal(bgDesc.resources[1].kind, 'sampler');
  assert.equal(bgDesc.resources[1].isDummy, false);
  assert.equal(bgDesc.resources[1].resource, processor.samplers.get(51));

  // Remapped bind group bound the REAL texture 50 and REAL sampler 51!
  assert.equal(remappedBg.desc.entries[0].resource.textureHandle, processor.textures.get(50).handle);
  assert.equal(remappedBg.desc.entries[1].resource, processor.samplers.get(51));
});

test('W98 pickResourceForBinding shape guards reject poisoned resources and enforce texture format/sampleType compatibility', () => {
  const gpu = createMockWebGpu();
  gpu.device.features = new Set(); // no float32-filterable
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor);

  const bufObj = gpu.device.createBuffer({ size: 256, usage: 0x0040 | 0x0080 | 0x0008 });
  const texR32F = gpu.device.createTexture({
    size: { width: 8, height: 8, depthOrArrayLayers: 1 },
    format: 'r32float',
    usage: 0x04,
  });
  const texDepth = gpu.device.createTexture({
    size: { width: 8, height: 8, depthOrArrayLayers: 1 },
    format: 'depth32float',
    usage: 0x04,
  });
  const texUint = gpu.device.createTexture({
    size: { width: 8, height: 8, depthOrArrayLayers: 1 },
    format: 'r32uint',
    usage: 0x04,
  });
  const viewR32F = texR32F.createView({ dimension: '2d' });
  const viewDepth = texDepth.createView({ dimension: '2d' });
  const viewUint = texUint.createView({ dimension: '2d' });
  const samplerObj = gpu.device.createSampler({});

  const autoBgl = { kind: 'GPUBindGroupLayout', autoFor: 'poison-test' };
  const fakePipe = { getBindGroupLayout: () => autoBgl };

  // Poisoned descriptor: a { buffer } tagged as 'texture', a textureView tagged as 'storage',
  // a { buffer } tagged as 'sampler', plus unfilterable r32float / depth32float / r32uint views.
  const poisonedDesc = {
    id: 777,
    layoutId: 1,
    resources: [
      { binding: 0, kind: 'texture', resource: { buffer: bufObj, offset: 0, size: 256 }, isDummy: false },
      { binding: 1, kind: 'storage', resource: viewR32F, bufferUsage: 0x0080, isDummy: false },
      { binding: 2, kind: 'sampler', resource: { buffer: bufObj, offset: 0, size: 256 }, isDummy: false },
      { binding: 3, kind: 'uniform', resource: samplerObj, bufferUsage: 0x0040, isDummy: false },
      { binding: 4, kind: 'texture', resource: viewR32F, textureUsage: 0x04, textureFormat: 'r32float', viewDimension: '2d', isDummy: false },
      { binding: 5, kind: 'texture', resource: viewDepth, textureUsage: 0x04, textureFormat: 'depth32float', viewDimension: '2d', isDummy: false },
      { binding: 6, kind: 'texture', resource: viewUint, textureUsage: 0x04, textureFormat: 'r32uint', viewDimension: '2d', isDummy: false },
    ],
  };

  const wgslBindings = [
    { binding: 0, kind: 'texture', viewDimension: '2d' }, // f32 texture: must reject { buffer }, r32float (unfilterable), depth32float, r32uint -> dummy rgba8unorm!
    { binding: 1, kind: 'read-only-storage' }, // must reject viewR32F -> dummy buffer!
    { binding: 2, kind: 'sampler' }, // must reject { buffer } -> dummy sampler!
    { binding: 3, kind: 'uniform' }, // must reject samplerObj -> dummy buffer!
    { binding: 4, kind: 'texture', viewDimension: '2d', sampleType: 'depth' }, // matches real viewDepth!
    { binding: 5, kind: 'texture', viewDimension: '2d', sampleType: 'uint' }, // matches real viewUint!
  ];

  const remapped = handler._getOrCreateRemappedBindGroup(true, 999, fakePipe, wgslBindings, 777, poisonedDesc);
  assert.equal(remapped.status, 'ok');

  const [bg] = gpu.results('createBindGroup');
  // binding 0: dummy rgba8unorm GPUTextureView (NEVER { buffer }, NEVER r32float/depth/uint!)
  assert.equal(bg.desc.entries[0].resource.kind, 'GPUTextureView');
  assert.equal('buffer' in bg.desc.entries[0].resource, false);
  assert.notEqual(bg.desc.entries[0].resource, viewR32F);
  assert.notEqual(bg.desc.entries[0].resource, viewDepth);
  assert.notEqual(bg.desc.entries[0].resource, viewUint);
  // binding 1: dummy { buffer } (NEVER viewR32F!)
  assert.ok(bg.desc.entries[1].resource && 'buffer' in bg.desc.entries[1].resource && bg.desc.entries[1].resource.buffer);
  assert.notEqual(bg.desc.entries[1].resource, viewR32F);
  // binding 2: dummy GPUSampler (NEVER { buffer }!)
  assert.equal(bg.desc.entries[2].resource.kind, 'GPUSampler');
  assert.equal('buffer' in bg.desc.entries[2].resource, false);
  // binding 3: dummy { buffer } (NEVER samplerObj!)
  assert.ok(bg.desc.entries[3].resource && 'buffer' in bg.desc.entries[3].resource && bg.desc.entries[3].resource.buffer);
  assert.notEqual(bg.desc.entries[3].resource, samplerObj);
  // binding 4: real depth32float view matched sampleType:'depth'
  assert.equal(bg.desc.entries[4].resource, viewDepth);
  // binding 5: real r32uint view matched sampleType:'uint'
  assert.equal(bg.desc.entries[5].resource, viewUint);
});

// ---------------------------------------------------------------------------------------------
// W101: root-constant dynamicOffsets survive layout:'auto' bind group remapping
// ---------------------------------------------------------------------------------------------

const W101_VS = [
  'struct Rdna2Cbv { m: vec4<f32> };',
  '@group(0) @binding(0)',
  'var<uniform> cb0_: Rdna2Cbv;',
  '@vertex fn vs_main(@location(0) p: vec3<f32>, @location(1) c: vec4<f32>, @location(2) uv: vec2<f32>) -> @builtin(position) vec4<f32> {',
  '  return vec4<f32>(p + cb0_.m.xyz, 1.0);',
  '}',
].join('\n');
const W101_FS = [
  '@group(0) @binding(1)',
  'var tex0_: texture_2d<f32>;',
  '@group(0) @binding(2)',
  'var samp0_: sampler;',
  '@fragment fn fs_main() -> @location(0) vec4<f32> {',
  '  return textureSample(tex0_, samp0_, vec2<f32>(0.5, 0.5));',
  '}',
].join('\n');

// Root-constant ring (buffer 100, `ringSize` bytes) bound through an RS-4-style layout whose only
// entry is a dynamic-offset CBV, used by PSOs 600/601 which fall back to layout:'auto'.
function w101RenderSetup({ ringSize = 1024, bindingSize = 256, options = {} } = {}) {
  const gpu = createMockWebGpu();
  const autoBglMap = new Map();
  const origCreateRender = gpu.device.createRenderPipeline.bind(gpu.device);
  gpu.device.createRenderPipeline = (desc) => {
    const pipe = origCreateRender(desc);
    const autoBgl = { kind: 'GPUBindGroupLayout', autoFor: pipe.handle };
    autoBglMap.set(pipe, autoBgl);
    pipe.getBindGroupLayout = (groupIndex) => {
      if (groupIndex !== 0) throw new Error(`invalid group ${groupIndex}`);
      return autoBgl;
    };
    return pipe;
  };
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'rgba8unorm' });
  const handler = installRenderStatePackets(processor, options);
  const key = buildFrame1PipelineKey();
  processor.executeCommandBuffer(
    concat(
      new Stream()
        .shader(1, 0, W101_VS)
        .shader(2, 1, W101_FS)
        .buffer(100, 0x0040 | 0x0008, ringSize)
        .bytes(),
      bindGroupLayoutPacket(200, [
        { binding: 0, visibility: 3, type: 0, subtype: 0, flags: 0x1, minBindingSize: 16 },
      ]),
      pipelineLayoutPacket(201, [200]),
      renderPipeline2Packet(600, 1, 2, 201, key),
      renderPipeline2Packet(601, 1, 2, 201, key),
      bindGroup2Packet(400, 200, [{ binding: 0, type: 0, id: 100, offset: 0, size: bindingSize }])
    )
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(handler.renderPipelineAutoLayoutMeta.get(600).autoLayoutRemap, true);
  return { gpu, processor, handler, autoBglMap };
}

function w101DrawFrame(processor, draws) {
  const parts = [
    new Stream().packet(OPCODES.BEGIN_FRAME).bytes(),
    beginRenderPassPacket({ colors: [{ textureId: SWAPCHAIN_TEXTURE_ID }] }),
  ];
  for (const d of draws) {
    if (d.pipeline !== undefined) parts.push(new Stream().u32s(OPCODES.SET_PIPELINE, [d.pipeline]).bytes());
    if (d.offsets !== undefined) parts.push(setBindGroup2Packet(0, 400, d.offsets));
    parts.push(new Stream().u32s(OPCODES.DRAW, [3, 1, 0, 0]).bytes());
  }
  parts.push(header(V2_OPCODES.END_RENDER_PASS, new Uint8Array(0)));
  parts.push(new Stream().packet(OPCODES.PRESENT).bytes());
  return processor.executeCommandBuffer(concat(...parts));
}

test('W101 bind group layouts/descriptors record dynamicOffsetIndex in ascending binding order', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = installRenderStatePackets(processor);
  processor.executeCommandBuffer(
    concat(
      new Stream().buffer(100, 0x0040 | 0x0008, 1024).buffer(101, 0x0040 | 0x0008, 512).bytes(),
      // Declared out of order: binding 3 (dynamic), 1 (dynamic), 2 (static).
      bindGroupLayoutPacket(200, [
        { binding: 3, visibility: 3, type: 0, subtype: 0, flags: 0x1, minBindingSize: 16 },
        { binding: 1, visibility: 3, type: 0, subtype: 0, flags: 0x1, minBindingSize: 16 },
        { binding: 2, visibility: 3, type: 0, subtype: 0, flags: 0, minBindingSize: 16 },
      ]),
      bindGroup2Packet(400, 200, [
        { binding: 3, type: 0, id: 100, offset: 0, size: 256 },
        { binding: 1, type: 0, id: 101, offset: 0, size: 256 },
        { binding: 2, type: 0, id: 100, offset: 256, size: 256 },
      ])
    )
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.deepEqual([...handler.bindGroupLayouts.get(200).dynamicOffsetIndices], [
    [1, 0],
    [3, 1],
  ]);
  const byBinding = new Map(handler.bindGroupDescriptors.get(400).resources.map((r) => [r.binding, r]));
  assert.equal(byBinding.get(1).hasDynamicOffset, true);
  assert.equal(byBinding.get(1).dynamicOffsetIndex, 0);
  assert.equal(byBinding.get(1).bufferSize, 512);
  assert.equal(byBinding.get(3).hasDynamicOffset, true);
  assert.equal(byBinding.get(3).dynamicOffsetIndex, 1);
  assert.equal(byBinding.get(3).bufferSize, 1024);
  assert.equal(byBinding.get(2).hasDynamicOffset, undefined);
  assert.equal(byBinding.get(2).dynamicOffsetIndex, undefined);
});

test('W101 auto-layout remap folds per-draw root-constant dynamicOffsets into the static buffer offset', () => {
  const { gpu, processor, handler, autoBglMap } = w101RenderSetup();
  const ring = processor.buffers.get(100);
  const res = w101DrawFrame(processor, [
    { pipeline: 600, offsets: [0] },
    { offsets: [256] },
    { offsets: [256] }, // same offset -> cached remapped bind group reused
    { offsets: [0] }, // back to 0 -> cached base-key bind group reused
  ]);
  assert.equal(res.presents, 1);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(processor.skippedDrawCount, 0);

  const pipe600 = gpu.results('createRenderPipeline')[0];
  const bgs = gpu.results('createBindGroup');
  // [0] = bg 400 (explicit RS-4 layout), [1] = remap @0, [2] = remap @256.
  assert.equal(bgs.length, 3);
  const [, remap0, remap256] = bgs;
  for (const bg of [remap0, remap256]) {
    assert.equal(bg.desc.layout, autoBglMap.get(pipe600.result));
    assert.deepEqual(bg.desc.entries.map((e) => e.binding), [0, 1, 2]);
  }
  assert.deepEqual(remap0.desc.entries[0].resource, { buffer: ring, offset: 0, size: 256 });
  assert.deepEqual(remap256.desc.entries[0].resource, { buffer: ring, offset: 256, size: 256 });

  // Each draw binds the remapped group matching its own offset (no dynamic offsets on auto layout).
  const renderSetBgs = gpu.passCalls.filter((c) => c.pass === 'render' && c.op === 'setBindGroup');
  assert.deepEqual(
    renderSetBgs.map((c) => c.args),
    [
      [0, remap0.result],
      [0, remap256.result],
      [0, remap256.result],
      [0, remap0.result],
    ]
  );
  assert.equal(handler.remappedBindGroups.has('r:600:400'), true);
  assert.equal(handler.remappedBindGroups.has('r:600:400@0=256'), true);
  assert.equal(handler.getTelemetry().dynamicOffsetRemaps, 1);

  // Destroying bind group 400 invalidates every dynamic-offset variant too.
  handler.destroyResource(400);
  assert.equal(handler.remappedBindGroups.size, 0);
});

test('W101 last group-0 dynamicOffsets are preserved when re-binding across a pipeline change', () => {
  const { gpu, processor, handler } = w101RenderSetup();
  const ring = processor.buffers.get(100);
  // setBindGroup(0, 400, [512]) under PSO 600, then switch to PSO 601 and draw without re-binding.
  w101DrawFrame(processor, [{ pipeline: 600, offsets: [512] }, { pipeline: 601 }]);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  const bgs = gpu.results('createBindGroup');
  assert.equal(bgs.length, 3);
  assert.deepEqual(bgs[1].desc.entries[0].resource, { buffer: ring, offset: 512, size: 256 });
  assert.deepEqual(bgs[2].desc.entries[0].resource, { buffer: ring, offset: 512, size: 256 });
  assert.equal(handler.remappedBindGroups.has('r:601:400@0=512'), true);
});

test('W101 dynamic offsets are clamped to the GPUBuffer size and out-of-bounds offsets bind a dummy buffer', () => {
  const { gpu, processor, handler } = w101RenderSetup({ ringSize: 1024, bindingSize: 256 });
  const ring = processor.buffers.get(100);
  w101DrawFrame(processor, [
    { pipeline: 600, offsets: [768] }, // 768 + 256 == 1024: fits exactly
    { offsets: [896] }, // 896 + 256 > 1024 -> size shrinks to 128
    { offsets: [1024] }, // offset == buffer size -> dummy buffer
  ]);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  const bgs = gpu.results('createBindGroup');
  assert.equal(bgs.length, 4);
  assert.deepEqual(bgs[1].desc.entries[0].resource, { buffer: ring, offset: 768, size: 256 });
  assert.deepEqual(bgs[2].desc.entries[0].resource, { buffer: ring, offset: 896, size: 128 });
  const oob = bgs[3].desc.entries[0].resource;
  assert.notEqual(oob.buffer, ring, 'out-of-bounds dynamic offset must fall back to a dummy buffer');
  for (const bg of bgs.slice(1)) {
    const r = bg.desc.entries[0].resource;
    if (r.buffer === ring) assert.ok(r.offset + r.size <= ring.size);
  }
  const t = handler.getTelemetry();
  assert.equal(t.dynamicOffsetRemaps, 2);
  assert.equal(t.dynamicOffsetClamps, 2);
  assert.ok(t.warnings.some((w) => w.includes('size clamped to 128')));
  assert.ok(t.warnings.some((w) => w.includes('exceeds buffer size 1024')));
});

test('W101 remappedBindGroups cache is LRU-bounded so rotating ring offsets never leak bind groups', () => {
  const { gpu, processor, handler } = w101RenderSetup({
    ringSize: 4096,
    options: { remappedBindGroupCacheLimit: 3 },
  });
  assert.equal(handler.remappedBindGroupCacheLimit, 3);
  const draws = [{ pipeline: 600, offsets: [0] }];
  for (let i = 1; i < 8; i += 1) draws.push({ offsets: [i * 256] });
  w101DrawFrame(processor, draws);
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(gpu.results('createBindGroup').length, 1 + 8);
  assert.equal(handler.remappedBindGroups.size, 3);
  assert.equal(handler.getTelemetry().remappedBindGroupEvictions, 5);
  assert.deepEqual(
    [...handler.remappedBindGroups.keys()],
    ['r:600:400@0=1280', 'r:600:400@0=1536', 'r:600:400@0=1792']
  );

  // A cache hit refreshes recency: re-using 1280 then adding 0 evicts 1536 (the LRU entry).
  w101DrawFrame(processor, [{ pipeline: 600, offsets: [1280] }, { offsets: [0] }]);
  assert.equal(gpu.results('createBindGroup').length, 1 + 8 + 1);
  assert.deepEqual(
    [...handler.remappedBindGroups.keys()],
    ['r:600:400@0=1792', 'r:600:400@0=1280', 'r:600:400']
  );

  // Default limit applies when no option is passed.
  const dflt = installRenderStatePackets(new WgpuCommandProcessor(gpu.device, gpu.ctx));
  assert.equal(dflt.remappedBindGroupCacheLimit, 2048);
});

test('W101 compute auto-layout remap honours root-constant dynamicOffsets on dispatch', () => {
  const gpu = createMockWebGpu();
  const origCreateCompute = gpu.device.createComputePipeline.bind(gpu.device);
  gpu.device.createComputePipeline = (desc) => {
    const pipe = origCreateCompute(desc);
    pipe.getBindGroupLayout = () => ({ kind: 'GPUBindGroupLayout', autoFor: pipe.handle });
    return pipe;
  };
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = installRenderStatePackets(processor);
  const cs = [
    '@group(0) @binding(0) var<uniform> cb0: vec4<f32>;',
    '@group(0) @binding(1) var<storage, read_write> out0: array<vec4<f32>>;',
    '@compute @workgroup_size(1, 1, 1) fn cs_main() { out0[0] = cb0; }',
  ].join('\n');
  processor.executeCommandBuffer(
    concat(
      new Stream()
        .shader(1, 2, cs)
        .buffer(10, 0x0040 | 0x0008, 1024)
        .bytes(),
      // Root-signature layout only carries the dynamic CBV -> WGSL storage@1 forces layout:'auto'.
      bindGroupLayoutPacket(20, [
        { binding: 0, visibility: 4, type: 0, subtype: 0, flags: 0x1, minBindingSize: 16 },
      ]),
      pipelineLayoutPacket(21, [20]),
      createComputePipeline2Packet(30, 1, 21),
      bindGroup2Packet(40, 20, [{ binding: 0, type: 0, id: 10, offset: 0, size: 256 }]),
      new Stream().packet(OPCODES.BEGIN_FRAME).u32s(OPCODES.SET_COMPUTE_PIPELINE, [30]).bytes(),
      setBindGroup2Packet(0, 40, [256]),
      new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1]).bytes(),
      setBindGroup2Packet(0, 40, [512]),
      new Stream().u32s(OPCODES.DISPATCH_COMPUTE, [1, 1, 1]).packet(OPCODES.PRESENT).bytes()
    )
  );
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(handler.computePipelineMeta.get(30).autoLayoutRemap, true);
  const bgs = gpu.results('createBindGroup');
  assert.equal(bgs.length, 3);
  const ring = processor.buffers.get(10);
  assert.deepEqual(bgs[1].desc.entries[0].resource, { buffer: ring, offset: 256, size: 256 });
  assert.deepEqual(bgs[2].desc.entries[0].resource, { buffer: ring, offset: 512, size: 256 });
  const computeSetBgs = gpu.passCalls.filter((c) => c.pass === 'compute' && c.op === 'setBindGroup');
  assert.deepEqual(
    computeSetBgs.map((c) => c.args),
    [
      [0, bgs[1].result],
      [0, bgs[2].result],
    ]
  );
});
