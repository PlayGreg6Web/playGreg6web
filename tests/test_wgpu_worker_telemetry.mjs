import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  SHADER_STAGE as WGPK_SHADER_STAGE,
  buildWgslPackBuffer,
  fnv1a64,
  formatHashHex,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const wgpuMod = require('../web/workers/wgpu_worker.js');
const gpuRingMod = require('../web/workers/gpu_ring.js');
const panelMod = require('../web/ui/gpu_boot_panel.js');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  DEFAULT_TELEMETRY_INTERVAL_MS,
  DEFAULT_TELEMETRY_MISSING_LIMIT,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
} = wgpuMod;

const { GpuRingBuffer } = gpuRingMod;
const { GpuBootPanelController } = panelMod;

const PACK_VS_WGSL = [
  '@vertex',
  'fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {',
  '    return vec4<f32>(0.0, 0.0, 0.0, 1.0);',
  '}',
  '',
].join('\n');

const PACK_FS_WGSL = [
  '@fragment',
  'fn fs_main() -> @location(0) vec4<f32> {',
  '    return vec4<f32>(0.2, 0.6, 0.9, 1.0);',
  '}',
  '',
].join('\n');

const S9_FALLBACK_FS_WGSL = [
  '@fragment',
  'fn fs_main() -> @location(0) vec4<f32> {',
  '    return vec4<f32>(0.9, 0.1, 0.5, 1.0);',
  '}',
  '',
].join('\n');

class PacketBuilder {
  constructor() {
    this.chunks = [];
  }

  _push(opcode, payloadBytes) {
    const buf = new ArrayBuffer(8 + payloadBytes.byteLength);
    const view = new DataView(buf);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, payloadBytes.byteLength, true);
    new Uint8Array(buf, 8).set(payloadBytes);
    this.chunks.push(new Uint8Array(buf));
  }

  beginFrame() {
    this._push(OPCODES.BEGIN_FRAME, new Uint8Array(0));
    return this;
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat32(0, r, true);
    view.setFloat32(4, g, true);
    view.setFloat32(8, b, true);
    view.setFloat32(12, a, true);
    this._push(OPCODES.CLEAR_COLOR, new Uint8Array(buf));
    return this;
  }

  createShaderModule(id, stage, wgslOrHash) {
    const encoded = new TextEncoder().encode(wgslOrHash);
    const buf = new ArrayBuffer(16 + encoded.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, encoded.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(encoded);
    this._push(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
    return this;
  }

  createRenderPipeline(id, vsId, fsId, vertexStride = 0, topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST) {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, vsId, true);
    view.setUint32(8, fsId, true);
    view.setUint32(12, vertexStride, true);
    view.setUint32(16, topology, true);
    view.setUint32(20, 0, true);
    this._push(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
    return this;
  }

  createComputePipeline(id, csId) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, csId, true);
    view.setUint32(8, 0, true);
    view.setUint32(12, 0, true);
    this._push(OPCODES.CREATE_COMPUTE_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._push(OPCODES.SET_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setComputePipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._push(OPCODES.SET_COMPUTE_PIPELINE, new Uint8Array(buf));
    return this;
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, vertexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstVertex, true);
    view.setUint32(12, firstInstance, true);
    this._push(OPCODES.DRAW, new Uint8Array(buf));
    return this;
  }

  dispatchCompute(x, y = 1, z = 1) {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, x, true);
    view.setUint32(4, y, true);
    view.setUint32(8, z, true);
    this._push(OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf));
    return this;
  }

  present() {
    this._push(OPCODES.PRESENT, new Uint8Array(0));
    return this;
  }

  finish() {
    const total = this.chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return out.buffer;
  }
}

function createMockWebGpu() {
  let nextHandle = 1;
  const ctx = {
    getCurrentTexture() {
      const view = { kind: 'GPUTextureView', id: `swap-${nextHandle++}` };
      return {
        createView() {
          return view;
        },
      };
    },
  };

  const device = {
    createShaderModule(desc) {
      return { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
    },
    createRenderPipeline(desc) {
      return {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(idx) {
          return { kind: 'GPUBindGroupLayout', idx };
        },
      };
    },
    createComputePipeline(desc) {
      return {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(idx) {
          return { kind: 'GPUBindGroupLayout', idx };
        },
      };
    },
    createCommandEncoder() {
      const cmdBuf = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            draw() {},
            end() {},
          };
        },
        beginComputePass() {
          return {
            setPipeline() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        finish() {
          return cmdBuf;
        },
      };
    },
    queue: {
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
  };

  return { device, ctx };
}

class MockElement {
  constructor(tagName) {
    this.tagName = String(tagName).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.attributes = new Map();
    this.style = { display: '', width: '' };
    this.hidden = false;
    this.textContent = '';
    this.className = '';
  }

  setAttribute(name, value) {
    this.attributes.set(String(name), String(value));
    if (name === 'class') this.className = String(value);
  }

  getAttribute(name) {
    return this.attributes.has(String(name)) ? this.attributes.get(String(name)) : null;
  }

  appendChild(child) {
    if (child.parentNode && child.parentNode !== this) {
      child.parentNode.removeChild(child);
    }
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx >= 0) {
      this.children.splice(idx, 1);
      child.parentNode = null;
    }
    return child;
  }

  querySelectorAll(selector) {
    const out = [];
    const match = (el) => {
      if (selector.startsWith('.')) {
        return String(el.className || '')
          .split(/\s+/)
          .includes(selector.slice(1));
      }
      return el.tagName === selector.toUpperCase();
    };
    const walk = (nodes) => {
      for (const child of nodes) {
        if (match(child)) out.push(child);
        walk(child.children);
      }
    };
    walk(this.children);
    return out;
  }

  querySelector(selector) {
    const found = this.querySelectorAll(selector);
    return found.length > 0 ? found[0] : null;
  }
}

function createMockDocument() {
  const head = new MockElement('head');
  const body = new MockElement('body');
  const documentElement = new MockElement('html');
  documentElement.appendChild(head);
  documentElement.appendChild(body);
  return {
    head,
    body,
    documentElement,
    createElement(tag) {
      return new MockElement(tag);
    },
    querySelector(selector) {
      return head.querySelector(selector) || body.querySelector(selector);
    },
  };
}

// ---------------------------------------------------------------------------
// 1. buildGpuTelemetrySnapshot shape matches GpuBootPanelController.update(...)
// ---------------------------------------------------------------------------

test('G30 buildGpuTelemetrySnapshot returns exact shape compatible with GpuBootPanelController.update()', () => {
  assert.equal(DEFAULT_TELEMETRY_INTERVAL_MS, 250);
  assert.equal(DEFAULT_TELEMETRY_MISSING_LIMIT, 64);

  const mock = createMockWebGpu();
  const vsHash = formatHashHex(fnv1a64(PACK_VS_WGSL));
  const fallbackFsHash = '0x000000000000f001';
  const missingFsHash = '0xdeadbeef00000002';
  const missingCsHash = '0xfeedface00000003';

  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL },
    { hash: fallbackFsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: S9_FALLBACK_FS_WGSL },
  ]);

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    wgslPackBuffer: packBuf,
    fallbackShaderHashes: { fragment: fallbackFsHash },
  });

  const frameStream = new PacketBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsHash)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, missingFsHash)
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, missingFsHash)
    .createShaderModule(4, SHADER_STAGE.COMPUTE, missingCsHash)
    .createRenderPipeline(10, 1, 2, 0)
    .createComputePipeline(11, 4)
    .setComputePipeline(11)
    .dispatchCompute(4, 1, 1)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  processor.executeCommandBuffer(frameStream);

  const snapshot = buildGpuTelemetrySnapshot(processor, { now: () => 1700000001234 });

  assert.equal(snapshot.type, 'gpu-telemetry');
  assert.equal(snapshot.timestamp, 1700000001234);

  assert.deepEqual(snapshot.frameStats, {
    framesPresented: 1,
    droppedFrames: 0,
    drawCalls: 1,
    computeDispatches: 1,
    packetsProcessed: 12,
  });

  assert.deepEqual(snapshot.fallbackStats, {
    totalLookups: 4,
    totalResolves: 4,
    packHits: 1,
    fallbackHits: 2,
    builtinHits: 1,
    hitRatePct: 25,
    uniqueMissing: 2,
    missingCount: 2,
    overflowMissingCount: 0,
  });

  assert.equal(snapshot.missingShaders.length, 2);
  assert.equal(snapshot.missingShaders[0].hash, missingFsHash);
  assert.equal(snapshot.missingShaders[0].stage, 'fragment');
  assert.equal(snapshot.missingShaders[0].hits, 2);
  assert.equal(snapshot.missingShaders[0].hitCount, 2);
  assert.equal(snapshot.missingShaders[1].hash, missingCsHash);
  assert.equal(snapshot.missingShaders[1].stage, 'compute');
  assert.equal(snapshot.missingShaders[1].hits, 1);
  assert.equal(snapshot.missingShaders[1].hitCount, 1);

  // Feed telemetry snapshot directly into GpuBootPanelController.update(...)
  const doc = createMockDocument();
  const panel = new GpuBootPanelController({ documentObj: doc, now: () => 1700000001234 });
  const state = panel.update(snapshot);

  assert.equal(state.frameStats.framesPresented, 1);
  assert.equal(state.frameStats.droppedFrames, 0);
  assert.equal(state.fallbackStats.packHits, 1);
  assert.equal(state.fallbackStats.fallbackHits, 2);
  assert.equal(state.fallbackStats.builtinHits, 1);
  assert.equal(state.fallbackStats.totalLookups, 4);
  assert.equal(state.fallbackStats.hitRatePct, 25);
  assert.equal(state.hitRateText, '25.0%');
  assert.equal(state.missingShaders.length, 2);
  assert.equal(state.missingShaders[0].hash, missingFsHash);
  assert.equal(state.missingShaders[0].hits, 2);
  assert.equal(state.missingShaders[1].hash, missingCsHash);
  assert.equal(state.missingShaders[1].hits, 1);

  const exported = JSON.parse(panel.exportMissingShadersJson());
  assert.equal(exported.gpu.framesPresented, 1);
  assert.equal(exported.gpu.packHits, 1);
  assert.equal(exported.gpu.fallbackHits, 2);
  assert.equal(exported.gpu.builtinHits, 1);
  assert.equal(exported.missingShaderCount, 2);

  panel.destroy();
});

// ---------------------------------------------------------------------------
// 2. buildGpuTelemetrySnapshot sanitization, limit, and null/disabled processor
// ---------------------------------------------------------------------------

test('G30 buildGpuTelemetrySnapshot sanitizes missingShaders, enforces limit, and handles null/disabled processor', () => {
  const nullSnap = buildGpuTelemetrySnapshot(null, { timestamp: 42 });
  assert.deepEqual(nullSnap, {
    type: 'gpu-telemetry',
    frameStats: {
      framesPresented: 0,
      droppedFrames: 0,
      drawCalls: 0,
      computeDispatches: 0,
      packetsProcessed: 0,
    },
    fallbackStats: {
      totalLookups: 0,
      totalResolves: 0,
      packHits: 0,
      fallbackHits: 0,
      builtinHits: 0,
      hitRatePct: 0,
      uniqueMissing: 0,
      missingCount: 0,
      overflowMissingCount: 0,
    },
    missingShaders: [],
    timestamp: 42,
  });

  // Disabled ShaderFallbackRouter returns clean zeroed fallbackStats.
  const mock = createMockWebGpu();
  const disabledProc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    enableShaderFallback: false,
  });
  const disabledSnap = buildGpuTelemetrySnapshot(disabledProc, { timestamp: 99 });
  assert.equal(disabledSnap.fallbackStats.totalResolves, 0);
  assert.equal(disabledSnap.fallbackStats.missingCount, 0);
  assert.deepEqual(disabledSnap.missingShaders, []);

  // Synthetic dirty processor must have host paths, raw buffers, and invalid hashes stripped.
  const dirtyProcessor = {
    getStats() {
      return {
        frameCount: 3,
        droppedFrames: 1,
        drawCalls: 10,
        computeDispatches: 2,
        packetCount: 45,
      };
    },
    getShaderFallbackStats() {
      return {
        totalLookups: 5,
        packHits: 2,
        fallbackHits: 1,
        builtinHits: 2,
        uniqueMissing: 3,
        overflowMissingCount: 0,
        hitRatePct: 40,
      };
    },
    getMissingShaders({ limit }) {
      assert.equal(limit, 2);
      return [
        {
          hash: 'C:\\Users\\secret\\game_scarlett_master.exe',
          stage: 'vertex',
          hitCount: 99,
        },
        {
          hash: '0x1234',
          stage: 'PS',
          hitCount: 7,
          firstSeenMs: 10,
          lastSeenMs: 20,
          hostPath: '/Users/secret/gta6.rpf',
          rawBytes: new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
        },
        {
          hash: '0xabcdef0123456789',
          stage: 'compute',
          hits: 3,
          wgsl: '@compute fn leak() {}',
        },
        {
          hash: '0x00000000000000ff',
          stage: 'vertex',
          hitCount: 1,
        },
      ];
    },
  };

  const cleanSnap = buildGpuTelemetrySnapshot(dirtyProcessor, { limit: 2, timestamp: 500 });
  assert.equal(cleanSnap.frameStats.framesPresented, 3);
  assert.equal(cleanSnap.frameStats.droppedFrames, 1);
  assert.equal(cleanSnap.missingShaders.length, 2);
  assert.deepEqual(cleanSnap.missingShaders[0], {
    hash: '0x0000000000001234',
    stage: 'fragment',
    hits: 7,
    hitCount: 7,
    firstSeenMs: 10,
    lastSeenMs: 20,
  });
  assert.deepEqual(cleanSnap.missingShaders[1], {
    hash: '0xabcdef0123456789',
    stage: 'compute',
    hits: 3,
    hitCount: 3,
  });

  const serialized = JSON.stringify(cleanSnap);
  assert.equal(serialized.includes('secret'), false);
  assert.equal(serialized.includes('rawBytes'), false);
  assert.equal(serialized.includes('hostPath'), false);
  assert.equal(serialized.includes('leak'), false);
});

// ---------------------------------------------------------------------------
// 3. Frame-1 immediate emission, throttling of rapid frames, and request-gpu-telemetry
// ---------------------------------------------------------------------------

test('G30 createWgpuWorkerHandler emits immediately on Frame 1 Present, throttles rapid frames, and responds to request-gpu-telemetry', async () => {
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);
  const posted = [];
  let fakeNow = 1000;

  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (msg) => posted.push(msg),
    now: () => fakeNow,
  });

  // Batch without Present must NOT emit gpu-telemetry.
  const noPresentBuf = new PacketBuilder().beginFrame().clearColor(0.1, 0.2, 0.3, 1.0).finish();
  await handler.handleMessage({ cmdBuffer: noPresentBuf });
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 0);

  // Frame 1 Present (opcode 7) emits immediately with zero delay.
  const frameBuf = () => new PacketBuilder().beginFrame().clearColor(0.2, 0.4, 0.6, 1.0).present().finish();
  await handler.handleMessage({ cmdBuffer: frameBuf() });

  const telemetryAfterFrame1 = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(telemetryAfterFrame1.length, 1);
  assert.equal(telemetryAfterFrame1[0].frameStats.framesPresented, 1);
  assert.equal(telemetryAfterFrame1[0].timestamp, 1000);
  assert.ok(posted.some((m) => m.frame === 1));

  // Rapid subsequent frames within default 250ms throttle window do NOT spam gpu-telemetry.
  fakeNow = 1016;
  await handler.handleMessage({ cmdBuffer: frameBuf() });
  fakeNow = 1033;
  await handler.handleMessage({ cmdBuffer: frameBuf() });
  fakeNow = 1249;
  await handler.handleMessage({ cmdBuffer: frameBuf() });

  assert.equal(processor.frameCount, 4);
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 1);

  // On-demand { type: 'request-gpu-telemetry' } responds immediately even while throttled!
  fakeNow = 1249;
  await handler.handleMessage({ type: 'request-gpu-telemetry' });
  const afterRequest = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(afterRequest.length, 2);
  assert.equal(afterRequest[1].frameStats.framesPresented, 4);
  assert.equal(afterRequest[1].timestamp, 1249);

  // Once >= 250ms elapsed since Frame 1 (1000 -> 1250), next Present emits telemetry again.
  fakeNow = 1250;
  await handler.handleMessage({ cmdBuffer: frameBuf() });
  const afterThrottleWindow = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(afterThrottleWindow.length, 3);
  assert.equal(afterThrottleWindow[2].frameStats.framesPresented, 5);
  assert.equal(afterThrottleWindow[2].timestamp, 1250);

  // Subsequent frame 10ms later is throttled again.
  fakeNow = 1260;
  await handler.handleMessage({ cmdBuffer: frameBuf() });
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 3);
});

// ---------------------------------------------------------------------------
// 4. Configurable telemetryIntervalMs, drainRing, submit/executeBatch, and pre-init request
// ---------------------------------------------------------------------------

test('G30 createWgpuWorkerHandler supports custom telemetryIntervalMs, drainRing, submit/executeBatch, and pre-init telemetry request', async () => {
  const postedPreInit = [];
  const preInitHandler = createWgpuWorkerHandler({
    postMessage: (m) => postedPreInit.push(m),
    now: () => 500,
  });
  await preInitHandler.handleMessage({ type: 'request-gpu-telemetry', id: 9 });
  assert.equal(postedPreInit.length, 1);
  assert.equal(postedPreInit[0].type, 'gpu-telemetry');
  assert.equal(postedPreInit[0].id, 9);
  assert.equal(postedPreInit[0].frameStats.framesPresented, 0);
  assert.deepEqual(postedPreInit[0].missingShaders, []);

  // Custom telemetryIntervalMs = 50 via drainRing + submit/executeBatch
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);
  const producer = GpuRingBuffer.create(512);
  processor.attachRing(producer.sab, { GpuRingBuffer });

  const posted = [];
  let fakeNow = 2000;
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    telemetryIntervalMs: 50,
    now: () => fakeNow,
  });

  const frameStream = new PacketBuilder().beginFrame().present().finish();

  // Frame 1 via drainRing -> emits immediately.
  assert.equal(producer.writeBytes(new Uint8Array(frameStream)), true);
  await handler.handleMessage({ drainRing: true });
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 1);
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry')[0].frameStats.framesPresented, 1);

  // Frame 2 via { type: 'submit', buffer } 20ms later -> throttled.
  fakeNow = 2020;
  await handler.handleMessage({ type: 'submit', buffer: frameStream });
  assert.equal(processor.frameCount, 2);
  assert.equal(posted.filter((m) => m.type === 'gpu-telemetry').length, 1);

  // Frame 3 via { type: 'executeBatch', buffer } 50ms after Frame 1 -> emits!
  fakeNow = 2050;
  await handler.handleMessage({ type: 'executeBatch', buffer: frameStream });
  const telemetries = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(telemetries.length, 2);
  assert.equal(telemetries[1].frameStats.framesPresented, 3);
  assert.equal(telemetries[1].timestamp, 2050);
});

// ---------------------------------------------------------------------------
// 5. Missing-shader records from hash-only CreateShaderModule packets appear in telemetry
// ---------------------------------------------------------------------------

test('G30 hash-only CreateShaderModule misses populate missingShaders and fallbackStats in emitted gpu-telemetry', async () => {
  const mock = createMockWebGpu();
  const packVsHash = formatHashHex(fnv1a64(PACK_VS_WGSL));
  const packFsHash = formatHashHex(fnv1a64(PACK_FS_WGSL));
  const missVsHash = '0xaaaaaaaaaaaaaaaa';
  const missFsHash = '0x00000000deadbeef';
  const missCsHash = '0x123456789abcdef0';

  const packBuf = buildWgslPackBuffer([
    { hash: packVsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL },
    { hash: packFsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: PACK_FS_WGSL },
  ]);

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    wgslPackBuffer: packBuf,
  });

  const posted = [];
  let fakeNow = 5000;
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (msg) => posted.push(msg),
    now: () => fakeNow,
  });

  const frame1 = new PacketBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, packVsHash)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, `hash:${packFsHash}`)
    .createShaderModule(3, SHADER_STAGE.VERTEX, missVsHash)
    .createShaderModule(4, SHADER_STAGE.FRAGMENT, 'hash:0xDEADBEEF')
    .createShaderModule(5, SHADER_STAGE.FRAGMENT, 'wgpk:00000000deadbeef')
    .createShaderModule(6, SHADER_STAGE.COMPUTE, '123456789abcdef0')
    .createRenderPipeline(10, 1, 4, 0)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  await handler.handleMessage({ cmdBuffer: frame1 });

  const telemetryMsgs = posted.filter((m) => m.type === 'gpu-telemetry');
  assert.equal(telemetryMsgs.length, 1);
  const t1 = telemetryMsgs[0];

  assert.equal(t1.frameStats.framesPresented, 1);
  assert.equal(t1.frameStats.drawCalls, 1);
  assert.equal(t1.fallbackStats.packHits, 2);
  assert.equal(t1.fallbackStats.fallbackHits, 0);
  assert.equal(t1.fallbackStats.builtinHits, 4);
  assert.equal(t1.fallbackStats.totalResolves, 6);
  assert.equal(t1.fallbackStats.missingCount, 3);
  assert.equal(t1.fallbackStats.hitRatePct, 33.33);

  assert.equal(t1.missingShaders.length, 3);
  // Sorted by hitCount descending: missFsHash (2) first, then missCsHash & missVsHash (1 each).
  assert.equal(t1.missingShaders[0].hash, missFsHash);
  assert.equal(t1.missingShaders[0].stage, 'fragment');
  assert.equal(t1.missingShaders[0].hits, 2);
  assert.equal(t1.missingShaders[0].hitCount, 2);

  const byHash = new Map(t1.missingShaders.map((r) => [r.hash, r]));
  assert.equal(byHash.get(missVsHash).stage, 'vertex');
  assert.equal(byHash.get(missVsHash).hits, 1);
  assert.equal(byHash.get(missCsHash).stage, 'compute');
  assert.equal(byHash.get(missCsHash).hits, 1);
});
