import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  bootstrapPlayGta6App,
  extractBootAssetUrls,
  buildWorkerPerfInitPayloads,
} from '../web/app_bootstrap.js';
import {
  SHADER_STAGE as PACK_STAGE,
  WGSL_PACK_VARIANT,
  buildWgslPackBuffer,
  fnv1a64,
  resolveWgslPackSelection,
  selectWgslPackUrl,
  WgslPackCache,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const wgpuMod = require('../web/workers/wgpu_worker.js');
const ringMod = require('../web/workers/gpu_ring.js');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  OPTIONAL_ADAPTER_FEATURES,
  SUBGROUPS_FEATURE_NAME,
  adapterSupportsSubgroups,
  extractSubgroupCapabilities,
  requestNegotiatedDevice,
  MAX_CACHED_VIEWS_PER_TEXTURE,
  OPCODES,
  BUFFER_USAGE,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  SHADER_STAGE,
  BIND_RESOURCE_TYPE,
} = wgpuMod;

const { GpuRingBuffer } = ringMod;

class StreamBuilder {
  constructor() {
    this.chunks = [];
  }

  packet(opcode, payload = new Uint8Array(0)) {
    const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
    const out = new Uint8Array(8 + bytes.byteLength);
    const view = new DataView(out.buffer);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, bytes.byteLength, true);
    out.set(bytes, 8);
    this.chunks.push(out);
    return this;
  }

  beginFrame() {
    return this.packet(OPCODES.BEGIN_FRAME);
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const dv = new DataView(buf);
    dv.setFloat32(0, r, true);
    dv.setFloat32(4, g, true);
    dv.setFloat32(8, b, true);
    dv.setFloat32(12, a, true);
    return this.packet(OPCODES.SET_CLEAR_COLOR, new Uint8Array(buf));
  }

  present() {
    return this.packet(OPCODES.PRESENT);
  }

  createBuffer(id, size, usage = BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST) {
    const buf = new ArrayBuffer(16);
    const dv = new DataView(buf);
    dv.setUint32(0, id, true);
    dv.setUint32(4, usage, true);
    dv.setBigUint64(8, BigInt(size), true);
    return this.packet(OPCODES.CREATE_BUFFER, new Uint8Array(buf));
  }

  createTexture(
    id,
    width,
    height,
    usage = TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    format = TEXTURE_FORMAT.RGBA8_UNORM,
    dimension = TEXTURE_DIMENSION.DIM_2D
  ) {
    const buf = new ArrayBuffer(32);
    const dv = new DataView(buf);
    dv.setUint32(0, id, true);
    dv.setUint32(4, width, true);
    dv.setUint32(8, height, true);
    dv.setUint32(12, 1, true);
    dv.setUint32(16, 1, true);
    dv.setUint32(20, format, true);
    dv.setUint32(24, usage, true);
    dv.setUint32(28, dimension, true);
    return this.packet(OPCODES.CREATE_TEXTURE, new Uint8Array(buf));
  }

  createSampler(id, filter = 1) {
    const buf = new ArrayBuffer(32);
    const dv = new DataView(buf);
    dv.setUint32(0, id, true);
    dv.setUint32(4, 0, true);
    dv.setUint32(8, 0, true);
    dv.setUint32(12, 0, true);
    dv.setUint32(16, filter, true);
    dv.setUint32(20, filter, true);
    dv.setUint32(24, filter, true);
    dv.setUint32(28, 1, true);
    return this.packet(OPCODES.CREATE_SAMPLER, new Uint8Array(buf));
  }

  createShaderModule(id, stage, code) {
    const enc = new TextEncoder().encode(code);
    const buf = new ArrayBuffer(16 + enc.byteLength);
    const dv = new DataView(buf);
    dv.setUint32(0, id, true);
    dv.setUint32(4, stage, true);
    dv.setUint32(8, enc.byteLength, true);
    dv.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(enc);
    return this.packet(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
  }

  createRenderPipeline(id, vsId, fsId = 0, vertexStride = 0, topology = 0, flags = 0) {
    const buf = new ArrayBuffer(24);
    const dv = new DataView(buf);
    dv.setUint32(0, id, true);
    dv.setUint32(4, vsId, true);
    dv.setUint32(8, fsId, true);
    dv.setUint32(12, vertexStride, true);
    dv.setUint32(16, topology, true);
    dv.setUint32(20, flags, true);
    return this.packet(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
  }

  createBindGroup(id, pipelineId, groupIndex, entries) {
    const buf = new ArrayBuffer(16 + entries.length * 32);
    const dv = new DataView(buf);
    dv.setUint32(0, id, true);
    dv.setUint32(4, pipelineId, true);
    dv.setUint32(8, groupIndex, true);
    dv.setUint32(12, entries.length, true);
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i];
      const base = 16 + i * 32;
      dv.setUint32(base, e.binding, true);
      dv.setUint32(base + 4, e.type ?? BIND_RESOURCE_TYPE.BUFFER, true);
      dv.setUint32(base + 8, e.resourceId, true);
      dv.setUint32(base + 12, 0, true);
      dv.setBigUint64(base + 16, BigInt(e.offset ?? 0), true);
      dv.setBigUint64(base + 24, BigInt(e.size ?? 0), true);
    }
    return this.packet(OPCODES.CREATE_BIND_GROUP, new Uint8Array(buf));
  }

  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this.packet(OPCODES.DESTROY_RESOURCE, new Uint8Array(buf));
  }

  build() {
    let total = 0;
    for (const c of this.chunks) total += c.byteLength;
    const out = new Uint8Array(total);
    let off = 0;
    for (const c of this.chunks) {
      out.set(c, off);
      off += c.byteLength;
    }
    return out;
  }
}

function createTrackingMockGpu(options = {}) {
  let nextId = 1;
  const createViewCalls = [];
  const createBindGroupCalls = [];
  const createTextureCalls = [];
  const sharedLayouts = new Map();

  const device = {
    features: new Set(options.features || []),
    limits: options.limits || {},
    adapterInfo: options.adapterInfo,
    createBuffer(desc) {
      return {
        kind: 'GPUBuffer',
        id: nextId++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
    },
    createTexture(desc) {
      const size = Array.isArray(desc.size)
        ? { width: desc.size[0], height: desc.size[1] || 1, depthOrArrayLayers: desc.size[2] || 1 }
        : desc.size;
      const tex = {
        kind: 'GPUTexture',
        id: nextId++,
        width: size.width,
        height: size.height,
        depthOrArrayLayers: size.depthOrArrayLayers || 1,
        format: desc.format,
        usage: desc.usage,
        dimension: desc.dimension || '2d',
        viewFormats: desc.viewFormats,
        desc,
        destroyed: false,
        createView(viewDesc) {
          const viewObj = {
            kind: 'GPUTextureView',
            viewId: nextId++,
            textureId: tex.id,
            desc: viewDesc ? { ...viewDesc } : null,
            dimension: (viewDesc && viewDesc.dimension) || tex.dimension,
            format: (viewDesc && viewDesc.format) || tex.format,
          };
          createViewCalls.push({ tex, viewDesc: viewDesc ? { ...viewDesc } : null, viewObj });
          return viewObj;
        },
        destroy() {
          this.destroyed = true;
        },
      };
      createTextureCalls.push(tex);
      return tex;
    },
    createSampler(desc) {
      return { kind: 'GPUSampler', id: nextId++, desc };
    },
    createBindGroupLayout(desc) {
      return { kind: 'GPUBindGroupLayout', id: nextId++, desc };
    },
    createPipelineLayout(desc) {
      return { kind: 'GPUPipelineLayout', id: nextId++, desc };
    },
    createShaderModule(desc) {
      return { kind: 'GPUShaderModule', id: nextId++, code: desc.code, label: desc.label };
    },
    createRenderPipeline(desc) {
      const pipeId = nextId++;
      const pipe = {
        kind: 'GPURenderPipeline',
        id: pipeId,
        desc,
        getBindGroupLayout(idx) {
          const key = `${pipeId}:${idx}`;
          if (!sharedLayouts.has(key)) {
            sharedLayouts.set(key, { kind: 'GPUBindGroupLayout', pipeId, idx });
          }
          return sharedLayouts.get(key);
        },
      };
      return pipe;
    },
    createComputePipeline(desc) {
      const pipeId = nextId++;
      return {
        kind: 'GPUComputePipeline',
        id: pipeId,
        desc,
        getBindGroupLayout(idx) {
          const key = `c:${pipeId}:${idx}`;
          if (!sharedLayouts.has(key)) {
            sharedLayouts.set(key, { kind: 'GPUBindGroupLayout', pipeId, idx });
          }
          return sharedLayouts.get(key);
        },
      };
    },
    createBindGroup(desc) {
      const bg = { kind: 'GPUBindGroup', id: nextId++, desc };
      createBindGroupCalls.push({ desc, bg });
      return bg;
    },
    createCommandEncoder() {
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            setViewport() {},
            setScissorRect() {},
            draw() {},
            drawIndexed() {},
            end() {},
          };
        },
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        copyBufferToBuffer() {},
        copyBufferToTexture() {},
        copyTextureToBuffer() {},
        copyTextureToTexture() {},
        finish() {
          return { kind: 'GPUCommandBuffer' };
        },
      };
    },
    queue: {
      submit() {},
      writeBuffer() {},
      writeTexture() {},
      onSubmittedWorkDone: async () => {},
    },
    addEventListener() {},
    removeEventListener() {},
    lost: new Promise(() => {}),
  };

  const ctx = {
    configure() {},
    getCurrentTexture() {
      return {
        kind: 'GPUTexture',
        width: 1280,
        height: 720,
        createView() {
          return { kind: 'SwapchainView' };
        },
      };
    },
  };

  return { device, ctx, createViewCalls, createBindGroupCalls, createTextureCalls };
}

async function withMockNavigatorGpu(gpu, fn) {
  const prevDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    value: { gpu },
    configurable: true,
    writable: true,
  });
  try {
    return await fn();
  } finally {
    if (prevDescriptor) {
      Object.defineProperty(globalThis, 'navigator', prevDescriptor);
    } else {
      delete globalThis.navigator;
    }
  }
}

const SIMPLE_VS = `
@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
`;
const SIMPLE_FS = `
@group(0) @binding(0) var<uniform> u0: vec4<f32>;
@group(0) @binding(1) var t0: texture_2d<f32>;
@group(0) @binding(2) var s0: sampler;
@fragment fn fs_main() -> @location(0) vec4<f32> {
  return u0 + textureSample(t0, s0, vec2<f32>(0.5, 0.5));
}
`;

// ---------------------------------------------------------------------------
// 1. WebGPU Hardware `subgroups` Auto-Detection, Fallback & Telemetry
// ---------------------------------------------------------------------------

test('#2191 subgroups adapter auto-detection requests subgroups opportunistically and surfaces telemetry', async () => {
  assert.equal(SUBGROUPS_FEATURE_NAME, 'subgroups');
  assert.equal(Object.isFrozen(OPTIONAL_ADAPTER_FEATURES), true);
  assert.equal(OPTIONAL_ADAPTER_FEATURES.includes('subgroups'), false);

  // 1a. Adapter WITH 'subgroups' + adapter.info subgroupMinSize/MaxSize
  {
    const requestCalls = [];
    const { device, ctx } = createTrackingMockGpu({
      features: ['texture-compression-bc', 'subgroups'],
      adapterInfo: { subgroupMinSize: 32, subgroupMaxSize: 64 },
    });
    const mockAdapter = {
      info: { vendor: 'amd', architecture: 'rdna2', subgroupMinSize: 32, subgroupMaxSize: 64 },
      features: new Set(['texture-compression-bc', 'subgroups']),
      limits: {},
      async requestDevice(desc) {
        requestCalls.push(desc);
        return device;
      },
    };
    assert.equal(adapterSupportsSubgroups(mockAdapter), true);

    const negotiated = await requestNegotiatedDevice(mockAdapter);
    assert.equal(requestCalls.length, 1);
    assert.deepEqual(requestCalls[0].requiredFeatures, ['texture-compression-bc', 'subgroups']);
    assert.equal(negotiated.hasSubgroups, true);
    assert.equal(negotiated.subgroupMinSize, 32);
    assert.equal(negotiated.subgroupMaxSize, 64);
    assert.ok(negotiated.enabledFeatures.includes('subgroups'));

    const posted = [];
    const mockGpu = {
      async requestAdapter() {
        return mockAdapter;
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });
    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({
        init: true,
        canvas: { width: 0, height: 0, getContext: () => ctx },
        width: 640,
        height: 360,
      });
    });

    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg);
    assert.equal(readyMsg.hasSubgroups, true);
    assert.equal(readyMsg.subgroupMinSize, 32);
    assert.equal(readyMsg.subgroupMaxSize, 64);
    assert.ok(readyMsg.features.includes('subgroups'));

    const stats = handler.processor.getStats();
    assert.equal(stats.hasSubgroups, true);
    assert.equal(stats.subgroupMinSize, 32);
    assert.equal(stats.subgroupMaxSize, 64);
    assert.deepEqual(handler.processor.getDeviceCapabilities(), {
      hasSubgroups: true,
      subgroupMinSize: 32,
      subgroupMaxSize: 64,
    });

    const snap = buildGpuTelemetrySnapshot(handler.processor);
    assert.equal(snap.hasSubgroups, true);
    assert.equal(snap.subgroupMinSize, 32);
    assert.equal(snap.subgroupMaxSize, 64);

    handler.processor.destroy();
  }

  // 1b. Adapter WITHOUT 'subgroups' -> never requests 'subgroups'
  {
    const requestCalls = [];
    const { device } = createTrackingMockGpu({
      features: ['texture-compression-bc'],
    });
    const mockAdapter = {
      info: { vendor: 'minimal', architecture: 'v1' },
      features: new Set(['texture-compression-bc']),
      limits: {},
      async requestDevice(desc) {
        requestCalls.push(desc);
        return device;
      },
    };
    assert.equal(adapterSupportsSubgroups(mockAdapter), false);
    const negotiated = await requestNegotiatedDevice(mockAdapter);
    assert.equal(requestCalls.length, 1);
    assert.deepEqual(requestCalls[0].requiredFeatures, ['texture-compression-bc']);
    assert.equal(negotiated.hasSubgroups, false);
    assert.equal(negotiated.subgroupMinSize, null);
    assert.equal(negotiated.subgroupMaxSize, null);
  }

  // 1c. Adapter advertising 'subgroups' that REJECTS requestDevice when 'subgroups' is requested ->
  //     falls back cleanly without 'subgroups' while keeping other features & elevated limits!
  {
    const requestCalls = [];
    const { device } = createTrackingMockGpu({
      features: ['texture-compression-bc', 'float32-filterable'],
    });
    const mockAdapter = {
      info: { vendor: 'flaky', architecture: 'gen1' },
      features: new Set(['texture-compression-bc', 'float32-filterable', 'subgroups']),
      limits: { maxStorageBuffersPerShaderStage: 16 },
      async requestDevice(desc) {
        requestCalls.push(desc);
        if (desc && Array.isArray(desc.requiredFeatures) && desc.requiredFeatures.includes('subgroups')) {
          throw new Error('subgroups unsupported by driver policy');
        }
        return device;
      },
    };
    const negotiated = await requestNegotiatedDevice(mockAdapter);
    assert.equal(requestCalls.length, 2);
    assert.ok(requestCalls[0].requiredFeatures.includes('subgroups'));
    assert.deepEqual(requestCalls[1].requiredFeatures, ['texture-compression-bc', 'float32-filterable']);
    assert.deepEqual(requestCalls[1].requiredLimits, { maxStorageBuffersPerShaderStage: 16 });
    assert.equal(negotiated.hasSubgroups, false);
    assert.deepEqual(negotiated.enabledFeatures, ['texture-compression-bc', 'float32-filterable']);
  }

  // 1d. Limits-based fallback extraction (minSubgroupSize / maxSubgroupSize)
  {
    const caps = extractSubgroupCapabilities({
      adapter: { limits: { minSubgroupSize: 8, maxSubgroupSize: 32 } },
      device: { features: new Set(['subgroups']) },
    });
    assert.deepEqual(caps, {
      hasSubgroups: true,
      subgroupMinSize: 8,
      subgroupMaxSize: 32,
    });
  }
});

// ---------------------------------------------------------------------------
// 2. Dual-Mode WGSL Pack Selection & app_bootstrap.js Forwarding
// ---------------------------------------------------------------------------

test('#2191 dual-mode WGSL pack selection picks subgroups pack when supported and falls back cleanly', async () => {
  const defaultUrl = '/shaders/workgroup_lds.wgsl.pack';
  const subgroupsUrl = '/shaders/subgroups.wgsl.pack';

  assert.equal(
    selectWgslPackUrl({ defaultPackUrl: defaultUrl, subgroupsPackUrl: subgroupsUrl, hasSubgroups: true }),
    subgroupsUrl
  );
  assert.equal(
    selectWgslPackUrl({ defaultPackUrl: defaultUrl, subgroupsPackUrl: subgroupsUrl, hasSubgroups: false }),
    defaultUrl
  );
  assert.equal(
    selectWgslPackUrl({ defaultPackUrl: defaultUrl, subgroupsPackUrl: null, hasSubgroups: true }),
    defaultUrl
  );
  assert.equal(
    selectWgslPackUrl({
      defaultPackUrl: defaultUrl,
      subgroupsPackUrl: subgroupsUrl,
      hasSubgroups: true,
      preferSubgroups: false,
    }),
    defaultUrl
  );
  assert.equal(selectWgslPackUrl(defaultUrl, subgroupsUrl, true), subgroupsUrl);
  assert.equal(selectWgslPackUrl(defaultUrl, subgroupsUrl, false), defaultUrl);

  const selSub = resolveWgslPackSelection({
    defaultPackUrl: defaultUrl,
    subgroupsPackUrl: subgroupsUrl,
    hasSubgroups: true,
  });
  assert.equal(selSub.mode, WGSL_PACK_VARIANT.SUBGROUPS);
  assert.equal(selSub.usesSubgroupsPack, true);
  assert.equal(selSub.fallbackUrl, defaultUrl);

  const selDefault = resolveWgslPackSelection({
    defaultPackUrl: defaultUrl,
    subgroupsPackUrl: subgroupsUrl,
    hasSubgroups: false,
  });
  assert.equal(selDefault.mode, WGSL_PACK_VARIANT.WORKGROUP_LDS);
  assert.equal(selDefault.usesSubgroupsPack, false);

  const hash = fnv1a64('compute_subgroup_test');
  const ldsWgsl = '// workgroup_lds\n@compute @workgroup_size(64) fn cs_main() {}';
  const subWgsl = '// subgroups\nenable subgroups;\n@compute @workgroup_size(64) fn cs_main() {}';
  const ldsPackBuf = buildWgslPackBuffer([{ hash, stage: PACK_STAGE.COMPUTE, wgsl: ldsWgsl }]);
  const subPackBuf = buildWgslPackBuffer([{ hash, stage: PACK_STAGE.COMPUTE, wgsl: subWgsl }]);

  const toArrayBuffer = (bufOrView) =>
    bufOrView instanceof ArrayBuffer
      ? bufOrView.slice(0)
      : bufOrView.buffer.slice(bufOrView.byteOffset, bufOrView.byteOffset + bufOrView.byteLength);

  // 2a. WgslPackCache loads subgroups pack when hasSubgroups=true
  {
    const fetchedUrls = [];
    const mockFetch = async (url) => {
      fetchedUrls.push(url);
      const bytes = url === subgroupsUrl ? subPackBuf : ldsPackBuf;
      return { ok: true, status: 200, arrayBuffer: async () => toArrayBuffer(bytes) };
    };
    const cache = new WgslPackCache({ hasSubgroups: true, subgroupsPackUrl: subgroupsUrl, fetch: mockFetch });
    await cache.loadPackUrl(defaultUrl);
    assert.deepEqual(fetchedUrls, [subgroupsUrl]);
    assert.equal(cache.getWgsl(hash), subWgsl);
    assert.equal(cache.getStats().subgroupsPackActive, true);
    assert.equal(cache.getStats().packVariant, 'subgroups');
    assert.equal(cache.getStats().subgroupsPackFallbackCount, 0);
  }

  // 2b. WgslPackCache falls back cleanly with ZERO error when subgroups pack returns 404
  {
    const fetchedUrls = [];
    const mockFetch = async (url) => {
      fetchedUrls.push(url);
      if (url === subgroupsUrl) {
        return { ok: false, status: 404, statusText: 'Not Found' };
      }
      return { ok: true, status: 200, arrayBuffer: async () => toArrayBuffer(ldsPackBuf) };
    };
    const cache = new WgslPackCache({ hasSubgroups: true, fetch: mockFetch });
    await cache.loadPackUrl(defaultUrl, { subgroupsPackUrl: subgroupsUrl });
    assert.deepEqual(fetchedUrls, [subgroupsUrl, defaultUrl]);
    assert.equal(cache.getWgsl(hash), ldsWgsl);
    assert.equal(cache.getStats().subgroupsPackActive, false);
    assert.equal(cache.getStats().packVariant, 'workgroup-lds');
    assert.equal(cache.getStats().subgroupsPackFallbackCount, 1);
  }

  // 2c. End-to-end wgpu_worker init selects subgroups pack when adapter supports subgroups
  {
    const fetchedUrls = [];
    const mockFetch = async (url) => {
      fetchedUrls.push(url);
      const bytes = url === subgroupsUrl ? subPackBuf : ldsPackBuf;
      return { ok: true, status: 200, arrayBuffer: async () => toArrayBuffer(bytes) };
    };
    const { device, ctx } = createTrackingMockGpu({
      features: ['subgroups'],
      adapterInfo: { subgroupMinSize: 32, subgroupMaxSize: 64 },
    });
    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'amd', architecture: 'rdna3', subgroupMinSize: 32, subgroupMaxSize: 64 },
          features: new Set(['subgroups']),
          async requestDevice() {
            return device;
          },
        };
      },
      getPreferredCanvasFormat: () => 'bgra8unorm',
    };
    const handler = createWgpuWorkerHandler({
      postMessage() {},
      enableSwapchain: false,
      fetch: mockFetch,
    });
    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({
        init: true,
        canvas: { width: 0, height: 0, getContext: () => ctx },
        width: 640,
        height: 360,
        shaderPackUrl: defaultUrl,
        wgslSubgroupsPackUrl: subgroupsUrl,
      });
    });
    assert.deepEqual(fetchedUrls, [subgroupsUrl]);
    assert.equal(handler.processor.wgslPackCache.getShaderSource(hash), subWgsl);
    handler.processor.destroy();
  }
});

test('#2191 app_bootstrap extracts and forwards wgslSubgroupsPackUrl and exposes subgroup telemetry', async () => {
  const extracted = extractBootAssetUrls({
    shaderPackUrl: '/shaders/default.wgsl.pack',
    wgslSubgroupsPackUrl: '/shaders/subgroups.wgsl.pack',
  });
  assert.equal(extracted.shaderPackUrl, '/shaders/default.wgsl.pack');
  assert.equal(extracted.wgslSubgroupsPackUrl, '/shaders/subgroups.wgsl.pack');

  const workerInit = buildWorkerPerfInitPayloads(
    { name: 'high', textureMipCap: 0 },
    { maxDisplayWidth: 1920, maxDisplayHeight: 1080 },
    { wgslSubgroupsPackUrl: '/shaders/subgroups.wgsl.pack', preferSubgroups: true }
  );
  assert.equal(workerInit.gpuWorker.wgslSubgroupsPackUrl, '/shaders/subgroups.wgsl.pack');
  assert.equal(workerInit.gpuWorker.preferSubgroups, true);

  let capturedCreateOpts = null;
  const postedToWorker = [];
  let workerMessageHandler = null;
  const fakeWorker = {
    addEventListener(type, fn) {
      if (type === 'message') workerMessageHandler = fn;
    },
    removeEventListener() {},
    postMessage(msg) {
      postedToWorker.push(msg);
    },
    terminate() {},
  };

  const fakeCanvas = {
    width: 640,
    height: 360,
    clientWidth: 640,
    clientHeight: 360,
    style: {},
    setAttribute() {},
    addEventListener() {},
    removeEventListener() {},
  };

  const handle = bootstrapPlayGta6App({
    canvas: fakeCanvas,
    windowObj: {
      crossOriginIsolated: true,
      SharedArrayBuffer,
      devicePixelRatio: 1,
      location: { origin: 'http://localhost:8080', search: '', href: 'http://localhost:8080/' },
      addEventListener() {},
      removeEventListener() {},
    },
    navigatorObj: { gpu: {} },
    attachGlobalHook: false,
    wgslSubgroupsPackUrl: '/shaders/subgroups.wgsl.pack',
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        hasWasmBinary: true,
        shaderPackUrl: '/shaders/default.wgsl.pack',
        wgslSubgroupsPackUrl: '/shaders/subgroups.wgsl.pack',
      }),
    }),
    createWasmWorker: async (opts) => {
      capturedCreateOpts = opts;
      return fakeWorker;
    },
  });

  const startRes = await handle.start();
  assert.equal(startRes.ok, true);
  assert.equal(capturedCreateOpts.wgslSubgroupsPackUrl, '/shaders/subgroups.wgsl.pack');
  assert.equal(postedToWorker.length >= 1, true);
  assert.equal(postedToWorker[0].wgslSubgroupsPackUrl, '/shaders/subgroups.wgsl.pack');

  // Simulate GPU telemetry reporting hardware subgroups capabilities
  assert.equal(typeof workerMessageHandler, 'function');
  workerMessageHandler({
    data: {
      type: 'gpu-telemetry',
      hasSubgroups: true,
      subgroupMinSize: 32,
      subgroupMaxSize: 64,
      frameStats: { framesPresented: 1, drawCalls: 1 },
    },
  });

  const diag = handle.getDiagnosticsSnapshot();
  assert.equal(diag.hasSubgroups, true);
  assert.equal(diag.subgroupMinSize, 32);
  assert.equal(diag.subgroupMaxSize, 64);

  const bootTel = handle.getBootTelemetry();
  assert.equal(bootTel.hasSubgroups, true);
  assert.equal(bootTel.subgroupMinSize, 32);
  assert.equal(bootTel.subgroupMaxSize, 64);
  assert.equal(bootTel.gpu.hasSubgroups, true);
  assert.equal(bootTel.gpu.subgroupMinSize, 32);
  assert.equal(bootTel.gpu.subgroupMaxSize, 64);

  handle.destroy();
});

// ---------------------------------------------------------------------------
// 3. Wave-5 gpu_ring.js `drainInto` Single-Pass Zero-Copy Fast Path
// ---------------------------------------------------------------------------

test('#2191 attachRing & drainRing support gpu_ring.js drainInto including wrapping SAB packets', () => {
  // 3a. Real GpuRingBuffer with small 128-byte capacity so packets wrap repeatedly across the SAB boundary!
  const gpu = createTrackingMockGpu();
  const proc = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    requireNonBlackFrame: false,
    preferDrainInto: true,
  });

  const producer = GpuRingBuffer.create(128);
  const consumer = GpuRingBuffer.attach(producer.sab);
  let drainIntoCalls = 0;
  const origDrainInto = consumer.drainInto.bind(consumer);
  consumer.drainInto = (...args) => {
    drainIntoCalls += 1;
    return origDrainInto(...args);
  };
  consumer.drainBatchZeroAlloc = () => {
    throw new Error('drainBatchZeroAlloc must not be called when preferDrainInto is true');
  };
  consumer.drainTo = () => {
    throw new Error('drainTo must not be called when drainInto is active');
  };

  proc.attachRing(consumer);

  // Frame stream is 8 (beginFrame) + 24 (clearColor) + 8 (present) = 40 bytes.
  // Over 10 iterations (400 bytes total on a 128-byte ring), packets wrap around the SAB boundary multiple times!
  const frameStream = new StreamBuilder().beginFrame().clearColor(0.2, 0.4, 0.6, 1.0).present().build();
  assert.equal(frameStream.byteLength, 40);

  let wrappedDrains = 0;
  for (let i = 0; i < 10; i += 1) {
    const startOff = consumer.readPos & (consumer.capacityBytes - 1);
    if (startOff + frameStream.byteLength > consumer.capacityBytes) {
      wrappedDrains += 1;
    }
    assert.equal(producer.writeBytes(frameStream), true);
    const res = proc.drainRing({ reuseResult: true });
    assert.equal(res.frameCount, i + 1);
    assert.equal(res.presents, 1);
    assert.equal(res.bytesDrained, frameStream.byteLength);
  }

  assert.ok(wrappedDrains >= 2, `expected >= 2 wrapping SAB drains, saw ${wrappedDrains}`);
  assert.equal(drainIntoCalls, 10);
  assert.equal(proc.getStats().drainIntoDrains, 10);
  assert.equal(proc.getStats().zeroAllocDrains, 10);
  assert.equal(proc.getStats().frameCount, 10);
  proc.destroy();

  // 3b. Custom ring object exposing ONLY `drainInto` (recognized by attachRing directly)
  const gpu2 = createTrackingMockGpu();
  const proc2 = new WgpuCommandProcessor(gpu2.device, gpu2.ctx, { requireNonBlackFrame: false });
  const backingRing = GpuRingBuffer.create(256);
  let onlyDrainIntoCalls = 0;
  const drainIntoOnlyRing = {
    capacityBytes: backingRing.capacityBytes,
    drainInto(cb, opts) {
      onlyDrainIntoCalls += 1;
      return backingRing.drainInto(cb, opts);
    },
  };
  const attached = proc2.attachRing(drainIntoOnlyRing);
  assert.equal(attached, drainIntoOnlyRing);
  assert.equal(backingRing.writeBytes(frameStream), true);
  const res2 = proc2.drainRing();
  assert.equal(onlyDrainIntoCalls, 1);
  assert.equal(res2.frameCount, 1);
  assert.equal(res2.presents, 1);
  assert.equal(res2.bytesDrained, frameStream.byteLength);
  proc2.destroy();
});

// ---------------------------------------------------------------------------
// 4. Bounded GPUTextureView Caching & Safety Preservation
// ---------------------------------------------------------------------------

test('#2191 _createSafeTextureView caches views per texture, preserves 1D/sRGB safety, bounds capacity, and invalidates on destroy', () => {
  const gpu = createTrackingMockGpu();
  const proc = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    promote1DTo2D: false,
    requireNonBlackFrame: false,
  });

  // Create normal 2D texture (id=10) + storage texture (id=11) + 1D texture (id=12)
  const setup = new StreamBuilder()
    .createTexture(10, 64, 64, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_DIMENSION.DIM_2D)
    .createTexture(11, 64, 64, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.STORAGE_BINDING, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_DIMENSION.DIM_2D)
    .createTexture(12, 64, 1, TEXTURE_USAGE.TEXTURE_BINDING, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_DIMENSION.DIM_1D)
    .build();
  proc.executeCommandBuffer(setup);

  const tex10 = proc.textures.get(10);
  const tex11 = proc.textures.get(11);
  const tex12 = proc.textures.get(12);
  const initialCreateViewCount = gpu.createViewCalls.length;
  assert.equal(initialCreateViewCount, 3); // 1 default view per created texture

  // Repeated default view (null descriptor) -> cache HIT!
  const v10DefaultA = proc._createSafeTextureView(tex10, null, 10);
  const v10DefaultB = proc._createSafeTextureView(tex10, null, 10);
  assert.equal(v10DefaultA, proc.textureViews.get(10));
  assert.equal(v10DefaultB, v10DefaultA);
  assert.equal(gpu.createViewCalls.length, initialCreateViewCount);

  // Explicit descriptor miss -> then hit on identical values from a brand-new descriptor object
  const v10Mip0A = proc._createSafeTextureView(tex10, { format: 'rgba8unorm', dimension: '2d', baseMipLevel: 0, mipLevelCount: 1 }, 10);
  const v10Mip0B = proc._createSafeTextureView(tex10, { format: 'rgba8unorm', dimension: '2d', baseMipLevel: 0, mipLevelCount: 1 }, 10);
  assert.equal(v10Mip0A, v10Mip0B);
  assert.equal(gpu.createViewCalls.length, initialCreateViewCount + 1);

  // Distinct descriptor -> miss & distinct view
  const v10Mip1 = proc._createSafeTextureView(tex10, { format: 'rgba8unorm', dimension: '2d', baseMipLevel: 1, mipLevelCount: 1 }, 10);
  assert.notEqual(v10Mip1, v10Mip0A);
  assert.equal(gpu.createViewCalls.length, initialCreateViewCount + 2);

  // sRGB safety rewrite on STORAGE_BINDING texture (id=11) is preserved AND cached
  const v11SrgbA = proc._createSafeTextureView(tex11, { format: 'rgba8unorm-srgb', dimension: '2d' }, 11);
  const v11SrgbB = proc._createSafeTextureView(tex11, { format: 'rgba8unorm-srgb', dimension: '2d' }, 11);
  assert.equal(v11SrgbA.format, 'rgba8unorm');
  assert.equal(v11SrgbB, v11SrgbA);
  assert.equal(gpu.createViewCalls.length, initialCreateViewCount + 3);

  // 1D safety coercion on 1D texture (id=12) is preserved AND cached
  const v12CoercedA = proc._createSafeTextureView(tex12, { dimension: '2d' }, 12);
  const v12CoercedB = proc._createSafeTextureView(tex12, { dimension: '2d' }, 12);
  assert.equal(v12CoercedA.dimension, '1d');
  assert.equal(v12CoercedB, v12CoercedA);
  assert.equal(gpu.createViewCalls.length, initialCreateViewCount + 4);

  // Telemetry counters reflect hits & misses
  const statsMid = proc.getStats();
  assert.equal(statsMid.textureViewCacheHits, 5);
  assert.equal(statsMid.textureViewCacheMisses, 7);
  const snapMid = buildGpuTelemetrySnapshot(proc);
  assert.equal(snapMid.textureViewCacheHits, 5);
  assert.equal(snapMid.textureViewCacheMisses, 7);

  // Bounded eviction: inserting > MAX_CACHED_VIEWS_PER_TEXTURE distinct descriptors caps map size cleanly
  for (let layer = 0; layer < MAX_CACHED_VIEWS_PER_TEXTURE + 8; layer += 1) {
    proc._createSafeTextureView(tex10, { baseArrayLayer: layer, arrayLayerCount: 1 }, 10);
  }
  assert.equal(proc._textureViewCache.get(tex10).size, MAX_CACHED_VIEWS_PER_TEXTURE);

  // Invalidation on destroy + recreate (even if a mock reuses the same texture object reference!)
  const countBeforeDestroy = gpu.createViewCalls.length;
  proc.executeCommandBuffer(new StreamBuilder().destroyResource(10).build());
  assert.equal(proc._textureViewCache.has(tex10), false);

  // Calling _createSafeTextureView on tex10 again after destroy must miss and call createView() fresh
  const v10AfterDestroy = proc._createSafeTextureView(tex10, null, 10);
  assert.notEqual(v10AfterDestroy, v10DefaultA);
  assert.equal(gpu.createViewCalls.length, countBeforeDestroy + 1);

  // Recreating texture ID 11 invalidates old tex11 entry too
  proc.executeCommandBuffer(
    new StreamBuilder()
      .createTexture(11, 32, 32, TEXTURE_USAGE.TEXTURE_BINDING)
      .build()
  );
  assert.equal(proc._textureViewCache.has(tex11), false);
  proc.destroy();
});

// ---------------------------------------------------------------------------
// 5. GPUBindGroup Reuse Caching & Strict Resource/Layout Invalidation
// ---------------------------------------------------------------------------

test('#2191 CreateBindGroup reuses identical live bind groups and strictly invalidates on resource or layout change', () => {
  const gpu = createTrackingMockGpu();
  const proc = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    enableAsyncPipelines: false,
    requireNonBlackFrame: false,
  });

  const setup = new StreamBuilder()
    .createBuffer(1, 256, BUFFER_USAGE.UNIFORM)
    .createTexture(2, 16, 16, TEXTURE_USAGE.TEXTURE_BINDING)
    .createSampler(3, 1)
    .createShaderModule(10, SHADER_STAGE.VERTEX, SIMPLE_VS)
    .createShaderModule(11, SHADER_STAGE.FRAGMENT, SIMPLE_FS)
    .createRenderPipeline(100, 10, 11)
    .createBindGroup(200, 100, 0, [
      { binding: 0, type: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
      { binding: 1, type: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 2 },
      { binding: 2, type: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 3 },
    ])
    .build();
  proc.executeCommandBuffer(setup);

  assert.equal(gpu.createBindGroupCalls.length, 1);
  const bg200First = proc.bindGroups.get(200);
  assert.ok(bg200First);
  assert.equal(proc.getStats().bindGroupReuseHits, 0);

  // 5a. Re-creating same ID (200) and another ID (201) with identical live bindings -> BOTH reuse!
  const reuseStream = new StreamBuilder()
    .createBindGroup(200, 100, 0, [
      { binding: 0, type: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
      { binding: 1, type: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 2 },
      { binding: 2, type: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 3 },
    ])
    .createBindGroup(201, 100, 0, [
      { binding: 0, type: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
      { binding: 1, type: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 2 },
      { binding: 2, type: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 3 },
    ])
    .build();
  proc.executeCommandBuffer(reuseStream);

  assert.equal(gpu.createBindGroupCalls.length, 1, 'identical live bind groups must reuse without calling device.createBindGroup');
  assert.equal(proc.bindGroups.get(200), bg200First);
  assert.equal(proc.bindGroups.get(201), bg200First);
  assert.equal(proc.getStats().bindGroupReuseHits, 2);
  assert.equal(buildGpuTelemetrySnapshot(proc).bindGroupReuseHits, 2);

  // 5b. Different buffer offset -> cache miss, creates a new bind group!
  proc.executeCommandBuffer(
    new StreamBuilder()
      .createBindGroup(202, 100, 0, [
        { binding: 0, type: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 64, size: 128 },
        { binding: 1, type: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 2 },
        { binding: 2, type: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 3 },
      ])
      .build()
  );
  assert.equal(gpu.createBindGroupCalls.length, 2);
  assert.notEqual(proc.bindGroups.get(202), bg200First);

  // 5c. Overwriting Buffer #1 -> strictly invalidates cached bind groups referencing Buffer #1!
  proc.executeCommandBuffer(
    new StreamBuilder()
      .createBuffer(1, 256, BUFFER_USAGE.UNIFORM)
      .createBindGroup(200, 100, 0, [
        { binding: 0, type: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
        { binding: 1, type: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 2 },
        { binding: 2, type: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 3 },
      ])
      .build()
  );
  assert.equal(gpu.createBindGroupCalls.length, 3);
  const bgAfterBufReplace = proc.bindGroups.get(200);
  assert.notEqual(bgAfterBufReplace, bg200First);

  // 5d. Overwriting Texture #2 -> strictly invalidates!
  proc.executeCommandBuffer(
    new StreamBuilder()
      .createTexture(2, 32, 32, TEXTURE_USAGE.TEXTURE_BINDING)
      .createBindGroup(200, 100, 0, [
        { binding: 0, type: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
        { binding: 1, type: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 2 },
        { binding: 2, type: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 3 },
      ])
      .build()
  );
  assert.equal(gpu.createBindGroupCalls.length, 4);
  const bgAfterTexReplace = proc.bindGroups.get(200);
  assert.notEqual(bgAfterTexReplace, bgAfterBufReplace);

  // 5e. Overwriting Sampler #3 -> strictly invalidates!
  proc.executeCommandBuffer(
    new StreamBuilder()
      .createSampler(3, 0)
      .createBindGroup(200, 100, 0, [
        { binding: 0, type: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
        { binding: 1, type: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 2 },
        { binding: 2, type: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 3 },
      ])
      .build()
  );
  assert.equal(gpu.createBindGroupCalls.length, 5);
  const bgAfterSamplerReplace = proc.bindGroups.get(200);
  assert.notEqual(bgAfterSamplerReplace, bgAfterTexReplace);

  // 5f. Overwriting Pipeline #100 -> strictly invalidates!
  proc.executeCommandBuffer(
    new StreamBuilder()
      .createRenderPipeline(100, 10, 11)
      .createBindGroup(200, 100, 0, [
        { binding: 0, type: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
        { binding: 1, type: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 2 },
        { binding: 2, type: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 3 },
      ])
      .build()
  );
  assert.equal(gpu.createBindGroupCalls.length, 6);
  const bgAfterPipeReplace = proc.bindGroups.get(200);
  assert.notEqual(bgAfterPipeReplace, bgAfterSamplerReplace);

  // 5g. Destroying BindGroup #200 -> evicts reuse entry so recreating #200 builds fresh!
  proc.executeCommandBuffer(
    new StreamBuilder()
      .destroyResource(200)
      .createBindGroup(200, 100, 0, [
        { binding: 0, type: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
        { binding: 1, type: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 2 },
        { binding: 2, type: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 3 },
      ])
      .build()
  );
  assert.equal(gpu.createBindGroupCalls.length, 7);
  assert.notEqual(proc.bindGroups.get(200), bgAfterPipeReplace);

  // 5h. Rebind device clears cache completely
  const gpu2 = createTrackingMockGpu();
  proc.rebindDevice(gpu2.device, gpu2.ctx);
  assert.equal(proc._bindGroupReuseCache.size, 0);
  proc.destroy();
});
