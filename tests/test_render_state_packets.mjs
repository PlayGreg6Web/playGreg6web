import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, OPCODES } = require('../web/workers/wgpu_worker.js');
const rs = require('../web/workers/render_state_packets.js');
const {
  installRenderStatePackets,
  decodeRenderPipelineKey,
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
      { binding: 0, resource: { buffer: ring.result, offset: 0, size: 16 } },
      { binding: 1, resource: albedoView.result },
      { binding: 2, resource: sampler.result },
    ],
  });
  const writes = gpu.results('writeBuffer').filter((c) => c.buffer === ring.result);
  assert.deepEqual(writes.map((w) => w.dstOffset), [0, 256]);
  assert.deepEqual([...new Float32Array(writes[0].data.buffer)], [1, 0.5, 0.25, 1]);

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
