import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const WORKERS_DIR = fileURLToPath(new URL('../web/workers/', import.meta.url));

/**
 * Exact set of classic-worker helper scripts loaded via `importScripts(...)`
 * by `web/workers/wgpu_worker.js` (plus `texture_formats.js`), in boot order.
 */
const WGPU_WORKER_HELPER_SCRIPTS = Object.freeze([
  './staging_pool.js',
  './uniform_ring.js',
  './gpu_timer.js',
  './reflection_loader.js',
  './resource_registry.js',
  './rt_pool.js',
  './shader_fallback_router.js',
  './render_state_packets.js',
  './texture_packets.js',
  './gpu_ring.js',
  './texture_formats.js',
]);

/**
 * Build a mock WebGPU adapter/device/canvas/context suitable for driving
 * `wgpu_worker.js` `defaultInitGpu()` and Packet ABI v1/v2 streams inside a VM context.
 */
function createMockWebGpuEnv() {
  const calls = {
    createBuffer: [],
    createTexture: [],
    createSampler: [],
    createShaderModule: [],
    createBindGroupLayout: [],
    createPipelineLayout: [],
    createBindGroup: [],
    createRenderPipeline: [],
    createComputePipeline: [],
    submit: 0,
    setViewport: [],
  };
  let nextHandle = 1;

  const device = {
    features: new Set(),
    createBuffer(desc) {
      const buf = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        mapState: desc.mappedAtCreation ? 'mapped' : 'unmapped',
        getMappedRange: () => new ArrayBuffer(desc.size),
        unmap() {
          buf.mapState = 'unmapped';
        },
        destroy() {
          buf.destroyed = true;
        },
      };
      calls.createBuffer.push({ desc, result: buf });
      return buf;
    },
    createTexture(desc) {
      const tex = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        desc,
        destroyed: false,
        createView(viewDesc) {
          return { kind: 'GPUTextureView', textureHandle: tex.handle, viewDesc };
        },
        destroy() {
          tex.destroyed = true;
        },
      };
      calls.createTexture.push({ desc, result: tex });
      return tex;
    },
    createSampler(desc) {
      const sampler = { kind: 'GPUSampler', handle: nextHandle++, desc };
      calls.createSampler.push({ desc, result: sampler });
      return sampler;
    },
    createShaderModule(desc) {
      const mod = { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
      calls.createShaderModule.push({ desc, result: mod });
      return mod;
    },
    createBindGroupLayout(desc) {
      const bgl = { kind: 'GPUBindGroupLayout', handle: nextHandle++, desc };
      calls.createBindGroupLayout.push({ desc, result: bgl });
      return bgl;
    },
    createPipelineLayout(desc) {
      const pl = { kind: 'GPUPipelineLayout', handle: nextHandle++, desc };
      calls.createPipelineLayout.push({ desc, result: pl });
      return pl;
    },
    createBindGroup(desc) {
      const bg = { kind: 'GPUBindGroup', handle: nextHandle++, desc };
      calls.createBindGroup.push({ desc, result: bg });
      return bg;
    },
    createRenderPipeline(desc) {
      const rp = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout: (i) => ({ kind: 'autoBgl', index: i }),
      };
      calls.createRenderPipeline.push({ desc, result: rp });
      return rp;
    },
    createComputePipeline(desc) {
      const cp = {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout: (i) => ({ kind: 'autoBgl', index: i }),
      };
      calls.createComputePipeline.push({ desc, result: cp });
      return cp;
    },
    createCommandEncoder() {
      return {
        beginRenderPass: () => ({
          setPipeline() {},
          setVertexBuffer() {},
          setIndexBuffer() {},
          setBindGroup() {},
          setViewport(...args) {
            calls.setViewport.push(args);
          },
          setScissorRect() {},
          setBlendConstant() {},
          setStencilReference() {},
          draw() {},
          drawIndexed() {},
          end() {},
        }),
        beginComputePass: () => ({
          setPipeline() {},
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {},
        }),
        copyBufferToBuffer() {},
        copyBufferToTexture() {},
        finish: () => ({ kind: 'GPUCommandBuffer' }),
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {
        calls.submit += 1;
      },
      onSubmittedWorkDone: () => Promise.resolve(),
    },
  };

  const ctx = {
    configured: null,
    configure(opts) {
      this.configured = opts;
    },
    getCurrentTexture: () => ({
      createView: () => ({ kind: 'swapchainView' }),
    }),
  };

  const canvas = {
    width: 0,
    height: 0,
    getContext(kind) {
      assert.equal(kind, 'webgpu');
      return ctx;
    },
  };

  const adapter = {
    info: { vendor: 'mock-gpu', architecture: 'classic-worker-test' },
    async requestDevice() {
      return device;
    },
  };

  const navigator = {
    gpu: {
      async requestAdapter() {
        return adapter;
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    },
  };

  return { device, ctx, canvas, adapter, navigator, calls };
}

/**
 * Create a Node `vm` context simulating a browser classic `WorkerGlobalScope`:
 * - `self === globalThis`
 * - `importScripts(...urls)` executes scripts sequentially in the shared global lexical scope
 * - No CommonJS `module`, `exports`, or `require` (forces the real browser `importScripts` path!)
 */
function createClassicWorkerContext(mockGpu) {
  const importedScripts = [];
  const postedMessages = [];

  const sandbox = {
    console,
    TextEncoder,
    TextDecoder,
    URL,
    performance,
    setTimeout,
    clearTimeout,
    navigator: mockGpu.navigator,
    postMessage(msg) {
      postedMessages.push(msg);
    },
  };

  const vmCtx = vm.createContext(sandbox);
  vmCtx.self = vmCtx;

  vmCtx.importScripts = (...urls) => {
    for (const rel of urls) {
      const fullPath = path.resolve(WORKERS_DIR, rel);
      const source = readFileSync(fullPath, 'utf8');
      importedScripts.push(rel);
      vm.runInContext(source, vmCtx, { filename: fullPath });
    }
  };

  const loadScript = (rel) => {
    const fullPath = path.resolve(WORKERS_DIR, rel);
    const source = readFileSync(fullPath, 'utf8');
    vm.runInContext(source, vmCtx, { filename: fullPath });
  };

  return { vmCtx, importedScripts, postedMessages, loadScript };
}

/**
 * Build a command packet buffer inside the VM realm (to avoid cross-realm ArrayBufferView issues).
 */
function buildPacketStreamInVm(vmCtx, packets) {
  const totalBytes = packets.reduce((acc, p) => acc + 8 + p.payload.byteLength, 0);
  const vmBytes = vm.runInContext(`new Uint8Array(${totalBytes})`, vmCtx);
  const view = new DataView(vmBytes.buffer, vmBytes.byteOffset, vmBytes.byteLength);
  let offset = 0;
  for (const p of packets) {
    view.setUint16(offset, p.opcode, true);
    view.setUint16(offset + 2, 0, true);
    view.setUint32(offset + 4, p.payload.byteLength, true);
    vmBytes.set(p.payload, offset + 8);
    offset += 8 + p.payload.byteLength;
  }
  return vmBytes;
}

function makeCreateBindGroupLayoutPayload(id, binding = 0, visibility = 0x3) {
  const buf = new ArrayBuffer(24 + 24);
  const view = new DataView(buf);
  view.setUint32(0, id, true);
  view.setUint32(4, 1, true); // count = 1
  view.setUint32(8, 0, true); // flags = 0
  view.setBigUint64(16, 0x12345678n, true); // contentId
  // Entry 0: uniform buffer binding
  view.setUint32(24, binding, true);
  view.setUint32(28, visibility, true);
  view.setUint8(32, 0); // BUFFER
  view.setUint8(33, 0); // uniform
  view.setBigUint64(40, 256n, true); // minBindingSize
  return new Uint8Array(buf);
}

function makeCreateSampler2Payload(id) {
  const buf = new ArrayBuffer(24);
  const view = new DataView(buf);
  view.setUint32(0, id, true);
  view.setUint8(4, 1); // repeat
  view.setUint8(5, 1); // repeat
  view.setUint8(6, 0); // clamp-to-edge
  view.setUint8(7, 1); // linear
  view.setUint8(8, 1); // linear
  view.setUint8(9, 1); // linear
  view.setUint8(10, 2); // compare: less (code 2)
  view.setUint16(12, 4, true); // maxAnisotropy
  view.setFloat32(16, 0, true);
  view.setFloat32(20, 32, true);
  return new Uint8Array(buf);
}

function makeCreateTexture2Payload({ id, width, height, samples = 4, format = 40, usage = 0x10 }) {
  const buf = new ArrayBuffer(40);
  const view = new DataView(buf);
  view.setUint32(0, id, true);
  view.setUint32(4, width, true);
  view.setUint32(8, height, true);
  view.setUint32(12, 1, true); // layers
  view.setUint32(16, 1, true); // mips
  view.setUint32(20, samples, true);
  view.setUint32(24, usage, true);
  view.setUint8(28, 1); // 2d
  view.setUint8(29, format); // 40 = rgba16float
  view.setUint8(30, 0); // viewFormats count
  return new Uint8Array(buf);
}

function makeHashOnlyCreateShaderModulePayload(id, stage, hashText) {
  const encoded = new TextEncoder().encode(hashText);
  const buf = new ArrayBuffer(16 + encoded.byteLength);
  const view = new DataView(buf);
  view.setUint32(0, id, true);
  view.setUint32(4, stage, true);
  view.setUint32(8, encoded.byteLength, true);
  view.setUint32(12, 0, true);
  new Uint8Array(buf, 16).set(encoded);
  return new Uint8Array(buf);
}

function makeSetViewportPayload(x, y, w, h, minDepth, maxDepth) {
  const buf = new ArrayBuffer(24);
  const view = new DataView(buf);
  view.setFloat32(0, x, true);
  view.setFloat32(4, y, true);
  view.setFloat32(8, w, true);
  view.setFloat32(12, h, true);
  view.setFloat32(16, minDepth, true);
  view.setFloat32(20, maxDepth, true);
  return new Uint8Array(buf);
}

test('G31 classic WorkerGlobalScope boot loads all wgpu_worker importScripts helpers without collisions and activates ABI v2 + texture_packets + shader_fallback_router', async () => {
  const mockGpu = createMockWebGpuEnv();
  const { vmCtx, importedScripts, postedMessages, loadScript } = createClassicWorkerContext(mockGpu);

  assert.equal(typeof vmCtx.module, 'undefined');
  assert.equal(typeof vmCtx.require, 'undefined');

  // 1. Load wgpu_worker.js as the classic Worker entry script.
  loadScript('./wgpu_worker.js');
  assert.equal(typeof vmCtx.self.onmessage, 'function');
  assert.ok(vmCtx.PlayGta6Wgpu, 'wgpu_worker.js must expose globalThis.PlayGta6Wgpu');

  // 2. Boot the worker with ALL optional subsystems + SharedArrayBuffer ring enabled so every
  //    importScripts(...) path in wgpu_worker.js executes in the exact live Chrome order.
  const ringSab = vm.runInContext('new SharedArrayBuffer(64 + 4096)', vmCtx);
  // Initialize GPUR ring header (magic 'GPUR', version 1, capacity 4096)
  vm.runInContext(
    `(() => {
      const h = new Int32Array(this.__ringSab, 0, 16);
      h[0] = 0x47505552;
      h[1] = 1;
      h[2] = 4096;
    })()`,
    Object.assign(vmCtx, { __ringSab: ringSab })
  );

  await vmCtx.self.onmessage({
    data: {
      init: true,
      canvas: mockGpu.canvas,
      width: 1280,
      height: 720,
      enableStagingPool: true,
      enableUniformRing: true,
      enablePassTimer: true,
      enableReflectionRegistry: true,
      enableResourceRegistry: true,
      enableRtPool: true,
      enableShaderFallback: true,
      ringSab,
    },
  });

  // Also load texture_formats.js into the same classic WorkerGlobalScope.
  vmCtx.importScripts('./texture_formats.js');

  // Zero errors (previously failed with: packet ABI v2 unavailable: SyntaxError: Identifier 'TEXTURE_FORMAT_NAMES' has already been declared)
  const errors = postedMessages.filter((m) => m && m.error);
  assert.deepEqual(errors, [], `unexpected worker error messages: ${JSON.stringify(errors)}`);
  assert.ok(
    postedMessages.some((m) => m && m.ready === true),
    'worker must post { ready: true }'
  );
  assert.ok(
    postedMessages.some((m) => m && m.ringAttached === true),
    'worker must post { ringAttached: true }'
  );

  // Confirm every expected helper script was loaded via importScripts.
  for (const script of WGPU_WORKER_HELPER_SCRIPTS) {
    assert.ok(
      importedScripts.includes(script),
      `expected importScripts to load ${script}; loaded: ${JSON.stringify(importedScripts)}`
    );
  }

  // Confirm all globals are populated in WorkerGlobalScope.
  assert.ok(vmCtx.PlayGta6RenderState && typeof vmCtx.PlayGta6RenderState.installRenderStatePackets === 'function');
  assert.ok(vmCtx.PlayGta6TexturePackets && typeof vmCtx.PlayGta6TexturePackets.installTexturePackets === 'function');
  assert.ok(vmCtx.PlayGta6TextureFormats && typeof vmCtx.PlayGta6TextureFormats.mapDxgiFormatToWebGpu === 'function');
  assert.ok(vmCtx.PlayGta6ShaderFallbackRouter && typeof vmCtx.PlayGta6ShaderFallbackRouter.ShaderFallbackRouter === 'function');
  assert.ok(vmCtx.PlayGta6GpuRing && typeof vmCtx.PlayGta6GpuRing.GpuRingBuffer === 'function');

  // 3. Drive live Packet ABI v2 (opcodes 22, 28, 34), CreateTexture2 (opcode 35),
  //    and ShaderFallbackRouter (hash-only CreateShaderModule opcode 5) + Present (opcode 7).
  const cmdStream = buildPacketStreamInVm(vmCtx, [
    { opcode: 1, payload: new Uint8Array(0) }, // BEGIN_FRAME
    { opcode: 22, payload: makeCreateBindGroupLayoutPayload(101) }, // CREATE_BIND_GROUP_LAYOUT (render_state_packets.js)
    { opcode: 34, payload: makeCreateSampler2Payload(202) }, // CREATE_SAMPLER2 (render_state_packets.js)
    { opcode: 35, payload: makeCreateTexture2Payload({ id: 303, width: 640, height: 360, samples: 4, format: 40 }) }, // CREATE_TEXTURE2 (texture_packets.js)
    { opcode: 5, payload: makeHashOnlyCreateShaderModulePayload(404, 1, '0xdeadbeefcafebabe') }, // CREATE_SHADER_MODULE fallback (shader_fallback_router.js)
    { opcode: 28, payload: makeSetViewportPayload(0, 0, 1280, 720, 0, 1) }, // SET_VIEWPORT (render_state_packets.js)
    { opcode: 7, payload: new Uint8Array(0) }, // PRESENT
  ]);

  await vmCtx.self.onmessage({ data: { cmdBuffer: cmdStream } });

  const postErrors = postedMessages.filter((m) => m && m.error);
  assert.deepEqual(postErrors, [], `unexpected runtime errors: ${JSON.stringify(postErrors)}`);

  // Verify render_state_packets.js executed opcodes 22 & 34.
  assert.equal(mockGpu.calls.createBindGroupLayout.length, 1, 'ABI v2 CreateBindGroupLayout must be active');
  assert.equal(mockGpu.calls.createSampler.length, 1, 'ABI v2 CreateSampler2 must be active');
  assert.equal(mockGpu.calls.createSampler[0].desc.compare, 'less');

  // Verify texture_packets.js executed opcode 35 (CreateTexture2).
  const tex2Call = mockGpu.calls.createTexture.find(
    (c) => c.desc && c.desc.format === 'rgba16float' && c.desc.sampleCount === 4
  );
  assert.ok(tex2Call, 'texture_packets.js CreateTexture2 (opcode 35) must be active');

  // Verify shader_fallback_router.js resolved the missing hash-only shader and tracked telemetry.
  assert.equal(mockGpu.calls.createShaderModule.length, 1, 'shader_fallback_router.js must resolve hash-only shader');
  assert.match(mockGpu.calls.createShaderModule[0].desc.code, /@fragment/);

  await vmCtx.self.onmessage({ data: { id: 'missing-check', missingShaders: true } });
  const missingReply = postedMessages.find((m) => m && m.id === 'missing-check');
  assert.ok(missingReply, 'worker must reply to missingShaders request');
  assert.ok(missingReply.shaderFallbackStats, 'shaderFallbackStats must be active');
  assert.equal(missingReply.shaderFallbackStats.totalLookups, 1);
  assert.equal(missingReply.missingShaders.length, 1);
  assert.equal(missingReply.missingShaders[0].hash, '0xdeadbeefcafebabe');
});

test('G31 importing all classic-worker helper scripts twice in the same WorkerGlobalScope throws zero SyntaxErrors', () => {
  const mockGpu = createMockWebGpuEnv();
  const { vmCtx, loadScript } = createClassicWorkerContext(mockGpu);

  // Load wgpu_worker.js first (mirrors live WorkerGlobalScope with wgpu_worker's top-level consts).
  loadScript('./wgpu_worker.js');

  // Pass 1: import every helper script in exact order.
  assert.doesNotThrow(() => {
    vmCtx.importScripts(...WGPU_WORKER_HELPER_SCRIPTS);
  }, 'first pass of importScripts must succeed without SyntaxError');

  // Pass 2: import every helper script a SECOND time (catches ANY leaked top-level const/let/class).
  assert.doesNotThrow(() => {
    vmCtx.importScripts(...WGPU_WORKER_HELPER_SCRIPTS);
  }, 'second pass of importScripts must succeed without SyntaxError');

  // Pass 3: import the three texture/render-state helpers individually in reverse order.
  assert.doesNotThrow(() => {
    vmCtx.importScripts('./texture_packets.js');
    vmCtx.importScripts('./render_state_packets.js');
    vmCtx.importScripts('./texture_formats.js');
  }, 'repeated individual importScripts calls must succeed without SyntaxError');

  // Verify WgpuCommandProcessor + chained extensions + shader fallback router work together cleanly.
  const processor = new vmCtx.PlayGta6Wgpu.WgpuCommandProcessor(mockGpu.device, mockGpu.ctx, {
    enableShaderFallback: true,
  });
  const rsHandler = vmCtx.PlayGta6RenderState.installRenderStatePackets(processor);
  const texHandler = vmCtx.PlayGta6TexturePackets.installTexturePackets(processor);

  assert.equal(processor.extension, texHandler, 'TexturePackets must be outer extension');
  assert.equal(texHandler.next, rsHandler, 'TexturePackets must chain to RenderStatePackets');
  assert.ok(processor.shaderFallbackRouter, 'ShaderFallbackRouter must be initialized');

  // Execute both a render_state_packets opcode (22) and a texture_packets opcode (35).
  const stream = buildPacketStreamInVm(vmCtx, [
    { opcode: 22, payload: makeCreateBindGroupLayoutPayload(1) },
    { opcode: 35, payload: makeCreateTexture2Payload({ id: 2, width: 32, height: 32, samples: 1, format: 22 }) },
  ]);
  processor.executeCommandBuffer(stream);

  assert.equal(rsHandler.stats.packets, 1);
  assert.equal(rsHandler.stats.errors, 0);
  assert.equal(texHandler.stats.created, 1);
  assert.equal(texHandler.stats.errors, 0);
});

test('G31 texture_formats.js, texture_packets.js, and render_state_packets.js leak zero top-level const/let/class bindings', () => {
  const mockGpu = createMockWebGpuEnv();
  const { vmCtx } = createClassicWorkerContext(mockGpu);

  vmCtx.importScripts(
    './texture_formats.js',
    './render_state_packets.js',
    './texture_packets.js'
  );

  // If any of these identifiers leaked as a top-level lexical binding, re-declaring them here throws SyntaxError.
  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const TEXTURE_FORMAT_NAMES = [];
      const TEXTURE_DIMENSION_NAMES = [];
      const INDEX_FORMAT_NAMES = [];
      const ADDRESS_MODE_NAMES = [];
      const FILTER_MODE_NAMES = [];
      const BIND_RESOURCE_TYPE = {};
      const CREATE_TEXTURE2 = 0;
      const V2_OPCODES = {};
      const DXGI_FORMAT = {};
      const exportedApi = {};
      const exportsObj = {};
      class TexturePackets {}
      class RenderStatePackets {}
      `,
      vmCtx
    );
  }, 'helper scripts must not leak top-level lexical declarations into WorkerGlobalScope');

  // Globals remain intact and unaffected by subsequent script lexical declarations.
  assert.equal(vmCtx.PlayGta6TexturePackets.CREATE_TEXTURE2, 35);
  assert.equal(vmCtx.PlayGta6RenderState.V2_OPCODES.CREATE_BIND_GROUP_LAYOUT, 22);
  assert.equal(vmCtx.PlayGta6TextureFormats.WEBGPU_BYTES_PER_ROW_ALIGNMENT, 256);
});

test('G31 CommonJS module.exports and globalThis.PlayGta6* exports remain intact', () => {
  const tf = require('../web/workers/texture_formats.js');
  const tp = require('../web/workers/texture_packets.js');
  const rs = require('../web/workers/render_state_packets.js');

  assert.equal(globalThis.PlayGta6TextureFormats, tf);
  assert.equal(globalThis.PlayGta6TexturePackets, tp);
  assert.equal(globalThis.PlayGta6RenderState, rs);

  assert.equal(typeof tf.mapDxgiFormatToWebGpu, 'function');
  assert.equal(typeof tf.decodeBc1ToRgba8, 'function');
  assert.equal(typeof tp.installTexturePackets, 'function');
  assert.equal(typeof tp.decodeCreateTexture2, 'function');
  assert.equal(tp.CREATE_TEXTURE2, 35);
  assert.equal(typeof rs.installRenderStatePackets, 'function');
  assert.equal(typeof rs.decodeRenderPipelineKey, 'function');
  assert.equal(rs.V2_OPCODES.CREATE_SAMPLER2, 34);
});
