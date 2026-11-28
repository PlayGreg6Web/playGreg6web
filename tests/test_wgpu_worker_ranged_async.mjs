import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

import {
  WgslPackCache,
  buildWgslPackBuffer,
  fnv1a64 as wgpkFnv1a64,
  formatHashHex as wgpkFormatHashHex,
  SHADER_STAGE as WGPK_SHADER_STAGE,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const WORKERS_DIR = fileURLToPath(new URL('../web/workers/', import.meta.url));
const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES,
  DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES,
  DEFAULT_MISSING_SUMMARY_LIMIT,
  DEFAULT_MAX_CONCURRENT_PIPELINE_COMPILES,
  sanitizeMissingShaderSummary,
  resolveShaderPackStreamThresholdBytes,
  OPCODES,
  SHADER_STAGE,
  BIND_RESOURCE_TYPE,
} = require('../web/workers/wgpu_worker.js');
const {
  installRenderStatePackets,
  TEXTURE_FORMAT_NAMES: RS_TEXTURE_FORMAT_NAMES,
} = require('../web/workers/render_state_packets.js');
const {
  BinkVideoPresenter,
  UNIFORM_FLAG_SWAP_CHROMA,
} = require('../web/workers/bink_presenter.js');
const { WebGpuPipelineCache } = require('../web/workers/pipeline_cache.js');
const {
  ReflectionRegistry,
  computeBindGroupLayoutHash,
  computePipelineLayoutHash,
  SHADER_STAGE_BITS,
} = require('../web/workers/reflection_loader.js');

class BinaryCommandBuilder {
  constructor() {
    this._chunks = [];
  }

  _pushPacket(opcode, payloadBytes) {
    const payload = payloadBytes instanceof Uint8Array ? payloadBytes : new Uint8Array(payloadBytes);
    const buf = new ArrayBuffer(8 + payload.byteLength);
    const view = new DataView(buf);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, payload.byteLength, true);
    new Uint8Array(buf, 8).set(payload);
    this._chunks.push(new Uint8Array(buf));
    return this;
  }

  beginFrame() {
    return this._pushPacket(OPCODES.BEGIN_FRAME, new Uint8Array(0));
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setFloat32(0, r, true);
    v.setFloat32(4, g, true);
    v.setFloat32(8, b, true);
    v.setFloat32(12, a, true);
    return this._pushPacket(OPCODES.CLEAR_COLOR, new Uint8Array(buf));
  }

  createShaderModule(id, stage, code) {
    const encoded = new TextEncoder().encode(code);
    const buf = new ArrayBuffer(16 + encoded.byteLength);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, stage, true);
    v.setUint32(8, encoded.byteLength, true);
    v.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(encoded);
    return this._pushPacket(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
  }

  createRenderPipeline(id, vsShaderId, fsShaderId, vertexStride = 0, topology = 0) {
    const buf = new ArrayBuffer(24);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, vsShaderId, true);
    v.setUint32(8, fsShaderId, true);
    v.setUint32(12, vertexStride, true);
    v.setUint32(16, topology, true);
    v.setUint32(20, 0, true);
    return this._pushPacket(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
  }

  createComputePipeline(id, csShaderId) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, csShaderId, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
    return this._pushPacket(OPCODES.CREATE_COMPUTE_PIPELINE, new Uint8Array(buf));
  }

  createRenderPipeline2(id, vsShaderId, fsShaderId, layoutId = 0, colorFormat = 'bgra8unorm') {
    const buf = new ArrayBuffer(16 + 376);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, vsShaderId, true);
    v.setUint32(8, fsShaderId, true);
    v.setUint32(12, layoutId, true);
    // 376-byte RenderPipelineKey starting at byte 16:
    // colorTargetCount at key+20 (u32), target[0].format at key+24 (u8), target[0].writeMask at key+26 (u8)
    const keyBase = 16;
    v.setUint32(keyBase + 20, 1, true);
    const fmtCode = RS_TEXTURE_FORMAT_NAMES.indexOf(colorFormat);
    v.setUint8(keyBase + 24, fmtCode >= 0 ? fmtCode : 27);
    v.setUint8(keyBase + 26, 0x0f);
    // multisample count = 1 at keyBase + 164 (u32), mask = 0xffffffff at keyBase + 168 (u32)
    v.setUint32(keyBase + 164, 1, true);
    v.setUint32(keyBase + 168, 0xffffffff, true);
    return this._pushPacket(OPCODES.CREATE_RENDER_PIPELINE2, new Uint8Array(buf));
  }

  createComputePipeline2(id, csShaderId, layoutId = 0) {
    const buf = new ArrayBuffer(32);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, csShaderId, true);
    v.setUint32(8, layoutId, true);
    return this._pushPacket(OPCODES.CREATE_COMPUTE_PIPELINE2, new Uint8Array(buf));
  }

  setPipeline(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this._pushPacket(OPCODES.SET_PIPELINE, new Uint8Array(buf));
  }

  setComputePipeline(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this._pushPacket(OPCODES.SET_COMPUTE_PIPELINE, new Uint8Array(buf));
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, vertexCount, true);
    v.setUint32(4, instanceCount, true);
    v.setUint32(8, firstVertex, true);
    v.setUint32(12, firstInstance, true);
    return this._pushPacket(OPCODES.DRAW, new Uint8Array(buf));
  }

  createBuffer(id, size, usage = 0x30) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, usage, true);
    v.setBigUint64(8, BigInt(size), true);
    return this._pushPacket(OPCODES.CREATE_BUFFER, new Uint8Array(buf));
  }

  setIndexBuffer(bufferId, formatCode = 0, offset = 0) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, bufferId, true);
    v.setUint32(4, formatCode, true);
    v.setBigUint64(8, BigInt(offset), true);
    return this._pushPacket(OPCODES.SET_INDEX_BUFFER, new Uint8Array(buf));
  }

  drawIndexed(indexCount, instanceCount = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(20);
    const v = new DataView(buf);
    v.setUint32(0, indexCount, true);
    v.setUint32(4, instanceCount, true);
    v.setUint32(8, firstIndex, true);
    v.setInt32(12, baseVertex, true);
    v.setUint32(16, firstInstance, true);
    return this._pushPacket(OPCODES.DRAW_INDEXED, new Uint8Array(buf));
  }

  dispatchCompute(x, y = 1, z = 1) {
    const buf = new ArrayBuffer(12);
    const v = new DataView(buf);
    v.setUint32(0, x, true);
    v.setUint32(4, y, true);
    v.setUint32(8, z, true);
    return this._pushPacket(OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf));
  }

  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this._pushPacket(OPCODES.DESTROY_RESOURCE, new Uint8Array(buf));
  }

  present() {
    return this._pushPacket(OPCODES.PRESENT, new Uint8Array(0));
  }

  finish() {
    const total = this._chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const c of this._chunks) {
      out.set(c, offset);
      offset += c.byteLength;
    }
    return out.buffer;
  }
}

function createMockWebGpu() {
  const calls = [];
  let nextId = 1;

  const device = {
    features: new Set(),
    createShaderModule(desc) {
      const mod = { kind: 'GPUShaderModule', id: nextId++, code: desc.code };
      calls.push({ op: 'createShaderModule', desc, mod });
      return mod;
    },
    createRenderPipeline(desc) {
      const pipe = {
        kind: 'GPURenderPipeline',
        id: nextId++,
        desc,
        getBindGroupLayout(index) {
          return { kind: 'GPUBindGroupLayout', pipelineId: pipe.id, index };
        },
      };
      calls.push({ op: 'createRenderPipeline', desc, pipe });
      return pipe;
    },
    createComputePipeline(desc) {
      const pipe = {
        kind: 'GPUComputePipeline',
        id: nextId++,
        desc,
        getBindGroupLayout(index) {
          return { kind: 'GPUBindGroupLayout', pipelineId: pipe.id, index };
        },
      };
      calls.push({ op: 'createComputePipeline', desc, pipe });
      return pipe;
    },
    createBindGroupLayout(desc) {
      return { kind: 'GPUBindGroupLayout', id: nextId++, desc };
    },
    createPipelineLayout(desc) {
      return { kind: 'GPUPipelineLayout', id: nextId++, desc };
    },
    createBindGroup(desc) {
      return { kind: 'GPUBindGroup', id: nextId++, desc };
    },
    createBuffer(desc) {
      return { kind: 'GPUBuffer', id: nextId++, size: desc.size, usage: desc.usage, destroy() {} };
    },
    createCommandEncoder() {
      const commands = [];
      return {
        beginRenderPass(desc) {
          commands.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline(pipeline) {
              commands.push({ op: 'setPipeline', pipeline });
            },
            setBindGroup(index, bg) {
              commands.push({ op: 'setBindGroup', index, bg });
            },
            setVertexBuffer(slot, buf, offset) {
              commands.push({ op: 'setVertexBuffer', slot, buf, offset });
            },
            setIndexBuffer(buf, format, offset) {
              commands.push({ op: 'setIndexBuffer', buf, format, offset });
            },
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              commands.push({ op: 'draw', vertexCount, instanceCount, firstVertex, firstInstance });
            },
            drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
              commands.push({
                op: 'drawIndexed',
                indexCount,
                instanceCount,
                firstIndex,
                baseVertex,
                firstInstance,
              });
            },
            end() {
              commands.push({ op: 'endRenderPass' });
            },
          };
        },
        beginComputePass() {
          commands.push({ op: 'beginComputePass' });
          return {
            setPipeline(pipeline) {
              commands.push({ op: 'setComputePipeline', pipeline });
            },
            setBindGroup(index, bg) {
              commands.push({ op: 'setComputeBindGroup', index, bg });
            },
            dispatchWorkgroups(x, y, z) {
              commands.push({ op: 'dispatchWorkgroups', x, y, z });
            },
            end() {
              commands.push({ op: 'endComputePass' });
            },
          };
        },
        finish() {
          const cb = { kind: 'GPUCommandBuffer', commands };
          calls.push({ op: 'finishCommandBuffer', commands });
          return cb;
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit(cbs) {
        calls.push({ op: 'submit', count: cbs.length, cbs });
      },
    },
  };

  const ctx = {
    configure() {},
    getCurrentTexture() {
      return {
        createView() {
          return { kind: 'GPUTextureView' };
        },
      };
    },
  };

  return { device, ctx, calls };
}

/**
 * Synthetic HTTP Range server backed by a compact WGPK pack buffer, optionally reporting a
 * large virtual Content-Length (e.g., 1.83 GiB) on HEAD and Content-Range so tests exercise
 * the >128 MiB auto-streaming path without allocating gigabytes of RAM.
 */
function createSyntheticRangePackServer(packBuffer, options = {}) {
  const packBytes = new Uint8Array(packBuffer);
  const reportedTotalBytes =
    options.reportedTotalBytes !== undefined ? options.reportedTotalBytes : packBytes.byteLength;
  const acceptRanges = options.acceptRanges !== undefined ? options.acceptRanges : 'bytes';
  const deferDataRanges = Boolean(options.deferDataRanges);
  const dataStartOffset = options.dataStartOffset || 0;
  const rejectFullGet = Boolean(options.rejectFullGet);

  const requests = [];
  const deferredRanges = [];

  const fetchImpl = async (url, init = {}) => {
    const method = (init && init.method ? String(init.method) : 'GET').toUpperCase();
    const headers = init && init.headers ? init.headers : {};
    const rangeHeader = headers.Range || headers.range || null;
    const req = { url, method, rangeHeader, start: null, end: null };
    requests.push(req);

    if (method === 'HEAD') {
      const respHeaders = new Map([['content-length', String(reportedTotalBytes)]]);
      if (acceptRanges !== null) {
        respHeaders.set('accept-ranges', acceptRanges);
      }
      return {
        ok: true,
        status: 200,
        headers: respHeaders,
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    if (rangeHeader) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      assert.ok(m, `Unexpected Range header: ${rangeHeader}`);
      const start = Number(m[1]);
      const requestedEnd = Number(m[2]);
      const clampedEnd = Math.min(requestedEnd, packBytes.byteLength - 1);
      req.start = start;
      req.end = clampedEnd;

      const isDataRange = dataStartOffset > 0 && start >= dataStartOffset;
      if (isDataRange && deferDataRanges) {
        await new Promise((resolve, reject) => {
          deferredRanges.push({ req, start, end: clampedEnd, resolve, reject });
        });
      }

      const slice = packBytes.slice(start, clampedEnd + 1);
      const totalForHeader = Math.max(reportedTotalBytes, clampedEnd + 1);
      return {
        ok: true,
        status: 206,
        headers: new Map([
          ['content-range', `bytes ${start}-${clampedEnd}/${totalForHeader}`],
          ['content-length', String(slice.byteLength)],
          ['accept-ranges', 'bytes'],
        ]),
        async arrayBuffer() {
          return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
        },
      };
    }

    if (rejectFullGet) {
      throw new Error(`Unexpected full-buffer GET for ${url} (expected HTTP 206 Range streaming)`);
    }

    return {
      ok: true,
      status: 200,
      headers: new Map([['content-length', String(packBytes.byteLength)]]),
      async arrayBuffer() {
        return packBuffer.slice(0);
      },
    };
  };

  return {
    requests,
    deferredRanges,
    fetchImpl,
    resolveNextDeferred() {
      const next = deferredRanges.shift();
      assert.ok(next, 'Expected a deferred range fetch');
      next.resolve();
      return next;
    },
    resolveAllDeferred() {
      while (deferredRanges.length > 0) {
        deferredRanges.shift().resolve();
      }
    },
  };
}

const VS_WGSL = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.2, 0.4, 0.0, 1.0); }\n';
const FS_WGSL = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.9, 0.3, 0.2, 1.0); }\n';
const CS_WGSL = '@compute @workgroup_size(8, 1, 1) fn cs_main() {}\n';
const VS_HASH = wgpkFormatHashHex(wgpkFnv1a64(VS_WGSL));
const FS_HASH = wgpkFormatHashHex(wgpkFnv1a64(FS_WGSL));
const CS_HASH = wgpkFormatHashHex(wgpkFnv1a64(CS_WGSL));

function buildSamplePack() {
  const buffer = buildWgslPackBuffer([
    { hash: VS_HASH, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: VS_WGSL },
    { hash: FS_HASH, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: FS_WGSL },
    { hash: CS_HASH, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl: CS_WGSL },
  ]);
  const dataStart = 16 + 3 * 20;
  return { buffer, dataStart };
}

// ---------------------------------------------------------------------------
// 1. Threshold constant export & resolution precedence
// ---------------------------------------------------------------------------

test('W159 exports DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES (128 MiB) and resolves threshold options cleanly', () => {
  assert.equal(DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES, 128 * 1024 * 1024);
  assert.equal(DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES, 64 * 1024 * 1024);

  // Default auto-threshold when omitted everywhere.
  assert.equal(resolveShaderPackStreamThresholdBytes(null, null), 128 * 1024 * 1024);
  assert.equal(resolveShaderPackStreamThresholdBytes({}, {}), 128 * 1024 * 1024);

  // Explicit 0 or streamShaders: false disables ranged streaming.
  assert.equal(resolveShaderPackStreamThresholdBytes({ streamThresholdBytes: 0 }, null), 0);
  assert.equal(resolveShaderPackStreamThresholdBytes({ streamShaders: false }, null), 0);
  assert.equal(
    resolveShaderPackStreamThresholdBytes(
      { streamThresholdBytes: 0 },
      { streamThresholdBytes: 128 * 1024 * 1024 }
    ),
    0
  );
  assert.equal(
    resolveShaderPackStreamThresholdBytes(
      { streamShaders: false },
      { streamThresholdBytes: 64 * 1024 * 1024 }
    ),
    0
  );
  assert.equal(
    resolveShaderPackStreamThresholdBytes({ streamShaders: false, streamThresholdBytes: 1024 }, null),
    0
  );

  // Explicit streamShaders: true uses 64 MiB; explicit streamThresholdBytes overrides it.
  assert.equal(resolveShaderPackStreamThresholdBytes({ streamShaders: true }, null), 64 * 1024 * 1024);
  assert.equal(
    resolveShaderPackStreamThresholdBytes({ streamShaders: true, streamThresholdBytes: 32 * 1024 * 1024 }, null),
    32 * 1024 * 1024
  );

  // Fallback & tertiary precedence.
  assert.equal(
    resolveShaderPackStreamThresholdBytes({}, { streamThresholdBytes: 0 }, { streamThresholdBytes: 128 }),
    0
  );
  assert.equal(
    resolveShaderPackStreamThresholdBytes({}, {}, { streamThresholdBytes: 0 }),
    0
  );
});

// ---------------------------------------------------------------------------
// 2. Auto-activation for >128 MiB shader packs by default (simulating 1.83 GiB pack)
// ---------------------------------------------------------------------------

test('W159 auto-enables HTTP 206 Range streaming by default for >128 MiB packs on both owned InlineWgslPackCache and WgslPackCache', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const FULL_PACK_SIZE_1_83_GIB = 1_964_947_456; // 1.83 GiB (> 128 MiB)

  // 2a. Default WgpuCommandProcessor (owned InlineWgslPackCache -> auto-upgrades to WgslPackCache).
  const netOwned = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: FULL_PACK_SIZE_1_83_GIB,
    dataStartOffset: dataStart,
    rejectFullGet: true,
  });
  const mockOwned = createMockWebGpu();
  const procOwned = new WgpuCommandProcessor(mockOwned.device, mockOwned.ctx, {
    format: 'bgra8unorm',
  });

  // Before loading: default threshold is 128 MiB, ranged pack not yet active.
  assert.equal(procOwned.rangedPackActive, false);
  assert.equal(procOwned.rangedPackThresholdBytes, DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES);
  assert.equal(procOwned.asyncShaderPendingCount, 0);

  await procOwned.loadShaderPackUrl('/shaders/full-18504.wgsl.pack', {
    fetch: netOwned.fetchImpl,
    initialRangeBytes: dataStart,
  });

  // Zero full-buffer 200 GETs; HEAD + HTTP 206 Range index load only.
  assert.equal(netOwned.requests.filter((r) => r.method === 'GET' && r.rangeHeader === null).length, 0);
  assert.equal(netOwned.requests.filter((r) => r.method === 'HEAD').length, 1);
  assert.equal(netOwned.requests.filter((r) => r.rangeHeader !== null).length, 1);

  assert.equal(procOwned.rangedPackActive, true);
  assert.equal(procOwned.rangedPackThresholdBytes, DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES);
  assert.equal(procOwned.asyncShaderPendingCount, 0);

  const ownedStats = procOwned.getStats();
  assert.equal(ownedStats.rangedPackActive, true);
  assert.equal(ownedStats.rangedPackThresholdBytes, DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES);
  assert.equal(ownedStats.asyncShaderPendingCount, 0);

  const ownedSnap = buildGpuTelemetrySnapshot(procOwned, { timestamp: 1001 });
  assert.equal(ownedSnap.shaderPackStats.mode, 'ranged');
  assert.equal(ownedSnap.shaderPackStats.rangedPackActive, true);
  assert.equal(
    ownedSnap.shaderPackStats.rangedPackThresholdBytes,
    DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES
  );
  assert.equal(ownedSnap.shaderPackStats.asyncShaderPendingCount, 0);
  assert.equal(ownedSnap.shaderPackStats.shaderCount, 3);

  // 2b. Worker handler init with default options (no streamThresholdBytes specified) + gpu-stats reply.
  const netWorker = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: FULL_PACK_SIZE_1_83_GIB,
    dataStartOffset: dataStart,
    rejectFullGet: true,
  });
  const mockWorker = createMockWebGpu();
  const procWorker = new WgpuCommandProcessor(mockWorker.device, mockWorker.ctx, {
    format: 'bgra8unorm',
    wgslPackCache: new WgslPackCache(),
  });
  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor: procWorker,
    fetch: netWorker.fetchImpl,
    postMessage: (m) => posted.push(m),
  });

  await handler.handleMessage({
    init: true,
    shaderPackUrl: '/shaders/full-18504.wgsl.pack',
    initialRangeBytes: dataStart,
  });

  const packWorkerRequests = netWorker.requests.filter(
    (r) => r.url === '/shaders/full-18504.wgsl.pack'
  );
  assert.equal(
    packWorkerRequests.filter((r) => r.method === 'GET' && r.rangeHeader === null).length,
    0
  );
  assert.ok(packWorkerRequests.some((r) => r.rangeHeader !== null));
  assert.equal(procWorker.rangedPackActive, true);

  await handler.handleMessage({ id: 'stats-1', type: 'request-gpu-stats' });
  const gpuStatsMsg = posted.find((m) => m && m.id === 'stats-1' && m.type === 'gpu-stats');
  assert.ok(gpuStatsMsg);
  assert.equal(gpuStatsMsg.rangedPackActive, true);
  assert.equal(gpuStatsMsg.rangedPackThresholdBytes, DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES);
  assert.equal(gpuStatsMsg.asyncShaderPendingCount, 0);

  procOwned.destroy();
  procWorker.destroy();
});

// ---------------------------------------------------------------------------
// 3. Explicit opt-out (`streamThresholdBytes: 0` or `streamShaders: false`)
// ---------------------------------------------------------------------------

test('W159 explicit streamThresholdBytes: 0 and streamShaders: false disable auto-streaming and force full-buffer GET', async () => {
  const { buffer } = buildSamplePack();
  const LARGE_REPORTED_SIZE = 512 * 1024 * 1024; // 512 MiB (> 128 MiB)

  // 3a. Explicit streamThresholdBytes: 0 on loadShaderPackUrl.
  const netZero = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: LARGE_REPORTED_SIZE,
  });
  const mockZero = createMockWebGpu();
  const procZero = new WgpuCommandProcessor(mockZero.device, mockZero.ctx, { format: 'bgra8unorm' });

  await procZero.loadShaderPackUrl('/shaders/full-optout-0.wgsl.pack', {
    fetch: netZero.fetchImpl,
    streamThresholdBytes: 0,
  });

  assert.equal(netZero.requests.length, 1);
  assert.equal(netZero.requests[0].method, 'GET');
  assert.equal(netZero.requests[0].rangeHeader, null);
  assert.equal(procZero.rangedPackActive, false);
  assert.equal(procZero.rangedPackThresholdBytes, 0);
  assert.equal(procZero.getStats().rangedPackActive, false);
  assert.equal(procZero.getStats().rangedPackThresholdBytes, 0);
  assert.equal(buildGpuTelemetrySnapshot(procZero).shaderPackStats.rangedPackActive, false);
  assert.equal(buildGpuTelemetrySnapshot(procZero).shaderPackStats.rangedPackThresholdBytes, 0);

  // Shaders resolve synchronously from full buffer with zero ranged fetches.
  procZero.executeCommandBuffer(
    new BinaryCommandBuilder().createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH).finish()
  );
  assert.equal(procZero.shaderModules.get(1).source, 'pack');
  assert.equal(procZero.getStats().rangedShaderFetches, 0);

  // 3b. Explicit streamShaders: false on processor constructor & worker handler message.
  const netFalse = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: LARGE_REPORTED_SIZE,
  });
  const mockFalse = createMockWebGpu();
  const procFalse = new WgpuCommandProcessor(mockFalse.device, mockFalse.ctx, {
    format: 'bgra8unorm',
    wgslPackCache: new WgslPackCache(),
    streamShaders: false,
  });
  const handlerFalse = createWgpuWorkerHandler({
    processor: procFalse,
    fetch: netFalse.fetchImpl,
  });

  await handlerFalse.handleMessage({
    init: true,
    shaderPackUrl: '/shaders/full-optout-false.wgsl.pack',
  });

  const packFalseRequests = netFalse.requests.filter(
    (r) => r.url === '/shaders/full-optout-false.wgsl.pack'
  );
  assert.equal(packFalseRequests.length, 1);
  assert.equal(packFalseRequests[0].method, 'GET');
  assert.equal(packFalseRequests[0].rangeHeader, null);
  assert.equal(procFalse.rangedPackActive, false);
  assert.equal(procFalse.rangedPackThresholdBytes, 0);
  assert.equal(procFalse.wgslPackCache.getStats().mode, 'buffer');

  procZero.destroy();
  procFalse.destroy();
});

// ---------------------------------------------------------------------------
// 4. Small-pack full-buffer fast path (< 128 MiB) & Accept-Ranges: none fallback
// ---------------------------------------------------------------------------

test('W159 preserves full-buffer loading for small (<128 MiB) packs and Accept-Ranges: none servers', async () => {
  const { buffer } = buildSamplePack();

  // 4a. Small pack (< 128 MiB) with default auto-threshold stays in buffer mode with zero Range requests.
  const netSmall = createSyntheticRangePackServer(buffer);
  const mockSmall = createMockWebGpu();
  const procSmall = new WgpuCommandProcessor(mockSmall.device, mockSmall.ctx, {
    format: 'bgra8unorm',
  });

  await procSmall.loadShaderPackUrl('/shaders/fallback.wgsl.pack', {
    fetch: netSmall.fetchImpl,
  });

  assert.equal(netSmall.requests.filter((r) => r.rangeHeader !== null).length, 0);
  assert.equal(procSmall.rangedPackActive, false);
  assert.equal(procSmall.rangedPackThresholdBytes, DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES);
  assert.equal(buildGpuTelemetrySnapshot(procSmall).shaderPackStats.mode, 'buffer');
  assert.equal(buildGpuTelemetrySnapshot(procSmall).shaderPackStats.rangedPackActive, false);

  procSmall.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
      .createRenderPipeline(10, 1, 2, 0)
      .beginFrame()
      .setPipeline(10)
      .draw(3, 1, 0, 0)
      .present()
      .finish()
  );
  assert.equal(procSmall.getStats().realPackDrawCount, 1);
  assert.equal(procSmall.getStats().rangedShaderFetches, 0);
  assert.equal(procSmall.getStats().skippedDrawCount, 0);

  // 4b. Large Content-Length but Accept-Ranges: none -> gracefully uses full-buffer GET.
  const netNoRange = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 256 * 1024 * 1024,
    acceptRanges: 'none',
  });
  const mockNoRange = createMockWebGpu();
  const procNoRange = new WgpuCommandProcessor(mockNoRange.device, mockNoRange.ctx, {
    format: 'bgra8unorm',
  });

  await procNoRange.loadShaderPackUrl('/shaders/no-range.wgsl.pack', {
    fetch: netNoRange.fetchImpl,
  });

  assert.equal(netNoRange.requests.filter((r) => r.rangeHeader !== null).length, 0);
  assert.equal(procNoRange.rangedPackActive, false);
  assert.equal(procNoRange.wgslPackCache.getStats().shaderCount, 3);

  procSmall.destroy();
  procNoRange.destroy();
});

// ---------------------------------------------------------------------------
// 5. Concurrent async getWgslAsync resolution + pending pipeline creation (zero dropped draws)
// ---------------------------------------------------------------------------

test('W159 concurrent async getWgslAsync resolution upgrades pending v1/v2 render and compute pipelines seamlessly with zero dropped draws', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const FULL_PACK_SIZE_1_83_GIB = 1_964_947_456;

  const net = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: FULL_PACK_SIZE_1_83_GIB,
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    deferRangedFallbackPipelines: false,
  });
  installRenderStatePackets(processor);

  // Auto-stream via default 128 MiB threshold (no explicit streamThresholdBytes).
  await processor.loadShaderPackUrl('/shaders/full-18504.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });
  assert.equal(processor.rangedPackActive, true);

  // Frame 1: create VS, FS, CS from ranged hashes + v1 & v2 pipelines + draw + compute dispatch while fetches are in flight.
  const frame1 = new BinaryCommandBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
    .createShaderModule(3, SHADER_STAGE.COMPUTE, CS_HASH)
    .createRenderPipeline(10, 1, 2, 0)
    .createComputePipeline(20, 3)
    .createRenderPipeline2(30, 1, 2, 0)
    .createComputePipeline2(40, 3, 0)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .setPipeline(30)
    .draw(3, 1, 0, 0)
    .setComputePipeline(20)
    .dispatchCompute(1, 1, 1)
    .present()
    .finish();

  const frame1Res = processor.executeCommandBuffer(frame1);
  assert.deepEqual(frame1Res, { frameCount: 1, presents: 1 });

  // Zero dropped draws on Frame 1 while all 3 ranged shaders are still pending!
  const midStats = processor.getStats();
  assert.equal(midStats.drawCalls, 2);
  assert.equal(midStats.fallbackDrawCount, 2);
  assert.equal(midStats.realPackDrawCount, 0);
  assert.equal(midStats.skippedDrawCount, 0);
  assert.equal(midStats.validationErrorCount, 0);
  assert.equal(midStats.asyncShaderPendingCount, 3);
  assert.equal(midStats.rangedShaderFetches, 3);
  assert.equal(midStats.rangedShaderReady, 0);
  assert.equal(processor.asyncShaderPendingCount, 3);

  const midSnap = buildGpuTelemetrySnapshot(processor, { timestamp: 2001 });
  assert.equal(midSnap.shaderPackStats.rangedPackActive, true);
  assert.equal(midSnap.shaderPackStats.asyncShaderPendingCount, 3);

  // Pipelines 10 & 30 are currently fallback pipelines while fetches are in flight.
  const fallbackPipe10 = processor.pipelines.get(10);
  const fallbackPipe30 = processor.pipelines.get(30);
  const fallbackCompute20 = processor.computePipelines.get(20);
  const fallbackCompute40 = processor.computePipelines.get(40);
  assert.ok(fallbackPipe10 && fallbackPipe30 && fallbackCompute20 && fallbackCompute40);
  assert.equal(processor._pipelineMeta.get(10).isRealPack, false);
  assert.equal(processor._pipelineMeta.get(30).isRealPack, false);

  // Resolve VS (1st of 3) -> render pipelines wait until FS also settles (no wasteful intermediate compile).
  assert.equal(net.deferredRanges.length, 3);
  net.resolveNextDeferred();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(processor.asyncShaderPendingCount, 2);
  assert.equal(processor.pipelines.get(10), fallbackPipe10, 'Wait for both VS+FS before rebuilding pipeline 10');

  // Resolve FS (2nd of 3) and CS (3rd of 3) concurrently -> all pending pipelines upgrade automatically!
  net.resolveAllDeferred();
  await processor.flushPendingRangedShaders();

  const afterStats = processor.getStats();
  assert.equal(afterStats.asyncShaderPendingCount, 0);
  assert.equal(afterStats.rangedShaderFetches, 3);
  assert.equal(afterStats.rangedShaderReady, 3);
  assert.equal(afterStats.rangedShaderErrors, 0);
  assert.equal(afterStats.validationErrorCount, 0);

  // Verify v1 & v2 render and compute pipelines were upgraded to the real pack shader modules!
  const realPipe10 = processor.pipelines.get(10);
  const realPipe30 = processor.pipelines.get(30);
  const realCompute20 = processor.computePipelines.get(20);
  const realCompute40 = processor.computePipelines.get(40);

  assert.notEqual(realPipe10, fallbackPipe10);
  assert.notEqual(realPipe30, fallbackPipe30);
  assert.notEqual(realCompute20, fallbackCompute20);
  assert.notEqual(realCompute40, fallbackCompute40);

  assert.equal(realPipe10.desc.vertex.module.code, VS_WGSL);
  assert.equal(realPipe10.desc.fragment.module.code, FS_WGSL);
  assert.equal(realPipe30.desc.vertex.module.code, VS_WGSL);
  assert.equal(realPipe30.desc.fragment.module.code, FS_WGSL);
  assert.equal(realCompute20.desc.compute.module.code, CS_WGSL);
  assert.equal(realCompute40.desc.compute.module.code, CS_WGSL);

  assert.equal(processor._pipelineMeta.get(10).isRealPack, true);
  assert.equal(processor._pipelineMeta.get(10).isFallback, false);
  assert.equal(processor._pipelineMeta.get(30).isRealPack, true);
  assert.equal(processor._pipelineMeta.get(30).isFallback, false);

  // Frame 2: draw with the same pipeline IDs -> now draws with REAL pack shaders, zero skipped draws!
  const frame2 = new BinaryCommandBuilder()
    .beginFrame()
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .setPipeline(30)
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  processor.executeCommandBuffer(frame2);

  const finalStats = processor.getStats();
  assert.equal(finalStats.drawCalls, 4);
  assert.equal(finalStats.realPackDrawCount, 2);
  assert.equal(finalStats.fallbackDrawCount, 2);
  assert.equal(finalStats.skippedDrawCount, 0);
  assert.equal(finalStats.validationErrorCount, 0);

  processor.destroy();
});

test('W159 deferred pipeline creation with enableShaderFallback: false creates pipeline once async ranged shaders settle without errors', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const net = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 256 * 1024 * 1024,
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: false,
  });

  await processor.loadShaderPackUrl('/shaders/full-nofb.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });
  assert.equal(processor.rangedPackActive, true);

  // Issue shader + pipeline creation + draw while fetches are still in flight.
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
      .createRenderPipeline(10, 1, 2, 0)
      .setPipeline(10)
      .draw(3, 1, 0, 0)
      .present()
      .finish()
  );

  // Zero validation errors while pipeline is pending ranged shaders.
  assert.equal(processor.getStats().validationErrorCount, 0);
  assert.equal(processor.isAsyncPipelinePending(10), true);
  assert.equal(processor.pipelines.has(10), false);
  assert.equal(processor.asyncShaderPendingCount, 2);

  // Resolve both ranged fetches -> pipeline 10 is created automatically!
  net.resolveAllDeferred();
  await processor.flushPendingRangedShaders();

  assert.equal(processor.asyncShaderPendingCount, 0);
  assert.equal(processor.isAsyncPipelinePending(10), false);
  assert.equal(processor.pipelines.has(10), true);
  assert.equal(processor.pipelines.get(10).desc.vertex.module.code, VS_WGSL);
  assert.equal(processor.pipelines.get(10).desc.fragment.module.code, FS_WGSL);
  assert.equal(processor.getStats().validationErrorCount, 0);

  // Frame 2 draws cleanly with the real pipeline!
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .setPipeline(10)
      .draw(3, 1, 0, 0)
      .present()
      .finish()
  );
  assert.equal(processor.getStats().realPackDrawCount, 1);
  assert.equal(processor.getStats().skippedDrawCount, 0);
  assert.equal(processor.getStats().validationErrorCount, 0);

  processor.destroy();
});

test('W159 destroying a pending pipeline before ranged shaders resolve prevents stale pipeline resurrection', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const net = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 256 * 1024 * 1024,
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    deferRangedFallbackPipelines: false,
  });

  await processor.loadShaderPackUrl('/shaders/full-destroy.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });

  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
      .createRenderPipeline(10, 1, 2, 0)
      .destroyResource(10)
      .finish()
  );
  assert.equal(processor.pipelines.has(10), false);

  net.resolveAllDeferred();
  await processor.flushPendingRangedShaders();

  // Shader modules upgraded, but destroyed pipeline 10 was NOT recreated.
  assert.equal(processor.shaderModules.get(1).source, 'pack');
  assert.equal(processor.shaderModules.get(2).source, 'pack');
  assert.equal(processor.pipelines.has(10), false);
  assert.equal(processor.getStats().asyncShaderPendingCount, 0);

  processor.destroy();
});

// ===========================================================================
// W185: Live wiring of W165 (Bink2 chromaOrder/loadOp), W167 (fallback sampler
// synthesis), W168 (graphics-over-compute async pipeline priority & compile
// latency telemetry), and W179 (categorized missing-shader summary export).
// ===========================================================================

function createRichW185MockWebGpu(options = {}) {
  let nextId = 1;
  const samplerDescs = [];
  const bindGroupDescs = [];
  const renderPasses = [];
  const writeBufferCalls = [];
  const writeTextureCalls = [];
  const asyncCompiles = [];

  const queue = {
    writeBuffer(buffer, offset, data) {
      const bytes =
        data instanceof Uint8Array
          ? new Uint8Array(data)
          : ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))
            : new Uint8Array(data.slice(0));
      writeBufferCalls.push({ buffer, offset, data: bytes });
    },
    writeTexture(dest, data, dataLayout, size) {
      writeTextureCalls.push({ dest, dataLayout: { ...dataLayout }, size: { ...size } });
    },
    submit() {},
    onSubmittedWorkDone() {
      return Promise.resolve();
    },
  };

  const makePipeline = (kind, desc) => {
    const pipe = {
      kind,
      id: nextId++,
      desc,
      getBindGroupLayout(index) {
        return { kind: 'GPUBindGroupLayout', pipelineId: pipe.id, index };
      },
    };
    return pipe;
  };

  const device = {
    features: new Set(),
    queue,
    createShaderModule(desc) {
      if (typeof options.onCreateShaderModule === 'function') {
        return options.onCreateShaderModule(desc, () => nextId++);
      }
      return { kind: 'GPUShaderModule', id: nextId++, code: desc.code };
    },
    createBuffer(desc) {
      return {
        kind: 'GPUBuffer',
        id: nextId++,
        size: desc.size,
        usage: desc.usage,
        destroy() {},
      };
    },
    createTexture(desc) {
      const id = nextId++;
      return {
        kind: 'GPUTexture',
        id,
        desc,
        width: desc.size.width,
        height: desc.size.height,
        createView() {
          return { kind: 'GPUTextureView', id: `view-${id}`, textureId: id };
        },
        destroy() {},
      };
    },
    createBindGroupLayout(desc) {
      return { kind: 'GPUBindGroupLayout', id: nextId++, desc };
    },
    createPipelineLayout(desc) {
      return { kind: 'GPUPipelineLayout', id: nextId++, desc };
    },
    createBindGroup(desc) {
      const bg = { kind: 'GPUBindGroup', id: nextId++, desc };
      bindGroupDescs.push(desc);
      return bg;
    },
    createRenderPipeline(desc) {
      return makePipeline('GPURenderPipeline', desc);
    },
    createComputePipeline(desc) {
      return makePipeline('GPUComputePipeline', desc);
    },
    createCommandEncoder() {
      return {
        beginRenderPass(passDesc) {
          const rec = { passDesc, draws: [], ended: false };
          renderPasses.push(rec);
          return {
            setPipeline() {},
            setViewport() {},
            setScissorRect() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            draw(vc, ic, fv, fi) {
              rec.draws.push({ vc, ic, fv, fi });
            },
            drawIndexed() {},
            end() {
              rec.ended = true;
            },
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
        finish() {
          return { kind: 'GPUCommandBuffer', id: nextId++ };
        },
      };
    },
  };

  if (options.omitCreateSampler !== true) {
    device.createSampler = (desc = {}) => {
      samplerDescs.push({ ...desc });
      return { kind: 'GPUSampler', id: nextId++, desc: { ...desc } };
    };
  }

  if (options.asyncPipelines === true) {
    const enqueueAsync = (kind, gpuKind) => (desc) =>
      new Promise((resolve, reject) => {
        const entry = {
          kind,
          desc,
          pipeline: null,
          resolve() {
            entry.pipeline = makePipeline(gpuKind, desc);
            resolve(entry.pipeline);
          },
          reject(err) {
            reject(err instanceof Error ? err : new Error(String(err || 'compile failed')));
          },
        };
        asyncCompiles.push(entry);
      });
    device.createRenderPipelineAsync = enqueueAsync('render', 'GPURenderPipeline');
    device.createComputePipelineAsync = enqueueAsync('compute', 'GPUComputePipeline');
  }

  const swapWidth = options.swapWidth || 1280;
  const swapHeight = options.swapHeight || 720;
  const ctx = {
    canvas: { width: swapWidth, height: swapHeight },
    configure() {},
    getCurrentTexture() {
      const view = { kind: 'GPUTextureView', id: `swap-${nextId++}` };
      return {
        width: swapWidth,
        height: swapHeight,
        createView() {
          return view;
        },
      };
    },
  };

  const canvas = {
    width: swapWidth,
    height: swapHeight,
    getContext() {
      return ctx;
    },
  };

  const adapter = {
    info: { vendor: 'mock-gpu', architecture: 'w185-test' },
    async requestDevice() {
      return device;
    },
  };

  const navigatorObj = {
    gpu: {
      async requestAdapter() {
        return adapter;
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    },
  };

  return {
    device,
    ctx,
    canvas,
    adapter,
    navigator: navigatorObj,
    samplerDescs,
    bindGroupDescs,
    renderPasses,
    writeBufferCalls,
    writeTextureCalls,
    asyncCompiles,
  };
}

function makeYuv420FramePlanes(width, height) {
  const uvW = Math.ceil(width / 2);
  const uvH = Math.ceil(height / 2);
  return {
    y: new Uint8Array(width * height).fill(180),
    u: new Uint8Array(uvW * uvH).fill(96),
    v: new Uint8Array(uvW * uvH).fill(160),
  };
}

function buildCreateTexturePacket(id, width, height, formatCode = 0, usage = 0x06) {
  const buf = new ArrayBuffer(32);
  const v = new DataView(buf);
  v.setUint32(0, id, true);
  v.setUint32(4, width, true);
  v.setUint32(8, height, true);
  v.setUint32(12, 1, true);
  v.setUint32(16, 1, true);
  v.setUint32(20, formatCode, true);
  v.setUint32(24, usage, true);
  v.setUint32(28, 1, true); // 2d
  return new BinaryCommandBuilder()._pushPacket(OPCODES.CREATE_TEXTURE, new Uint8Array(buf));
}

function buildCreateBindGroupPacket(id, pipelineId, groupIndex, entries) {
  const buf = new ArrayBuffer(16 + entries.length * 32);
  const v = new DataView(buf);
  v.setUint32(0, id, true);
  v.setUint32(4, pipelineId, true);
  v.setUint32(8, groupIndex, true);
  v.setUint32(12, entries.length, true);
  for (let i = 0; i < entries.length; i += 1) {
    const e = entries[i];
    const base = 16 + i * 32;
    v.setUint32(base, e.binding, true);
    v.setUint32(base + 4, e.resourceType, true);
    v.setUint32(base + 8, e.resourceId, true);
    v.setBigUint64(base + 16, BigInt(e.offset || 0), true);
    v.setBigUint64(base + 24, BigInt(e.size || 0), true);
  }
  return new BinaryCommandBuilder()._pushPacket(OPCODES.CREATE_BIND_GROUP, new Uint8Array(buf));
}

// ---------------------------------------------------------------------------
// 6. W185 / W165: Bink2 chromaOrder ('yuv' | 'yvu') & loadOp ('clear' | 'load')
// ---------------------------------------------------------------------------

test('W185 (W165) forwards chromaOrder and loadOp from init config and bink-frame / present-splash-frame messages', async () => {
  // 6a. Constructor init config (binkChromaOrder: 'yvu', binkLoadOp: 'load') + per-frame toggle via worker messages.
  const mock = createRichW185MockWebGpu({ swapWidth: 1920, swapHeight: 1080 });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    width: 1920,
    height: 1080,
    binkChromaOrder: 'yvu',
    binkLoadOp: 'load',
  });
  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    now: () => 1000,
  });

  // Frame 1 inherits init chromaOrder='yvu' and loadOp='load' (menu underlay).
  const planes1 = makeYuv420FramePlanes(256, 16);
  await handler.handleMessage({
    type: 'bink-frame',
    id: 'w165-1',
    frame: {
      width: 256,
      height: 16,
      format: 'yuv420p',
      planes: planes1,
      frameIndex: 0,
    },
  });

  const ack1 = posted.find((m) => m && m.type === 'splash-frame-presented' && m.id === 'w165-1');
  assert.ok(ack1 && ack1.ok === true);
  assert.equal(processor.binkPresenter.chromaOrder, 'yvu');
  assert.equal(processor.binkPresenter.loadOp, 'load');
  assert.equal(mock.renderPasses.length, 1);
  assert.equal(mock.renderPasses[0].passDesc.colorAttachments[0].loadOp, 'load');
  assert.equal(mock.renderPasses[0].passDesc.colorAttachments[0].clearValue, undefined);

  // Uniform buffer flags lane (u32[3]) has UNIFORM_FLAG_SWAP_CHROMA set for 'yvu'.
  assert.ok(mock.writeBufferCalls.length >= 1);
  const u32Frame1 = new Uint32Array(
    mock.writeBufferCalls[0].data.buffer,
    mock.writeBufferCalls[0].data.byteOffset,
    4
  );
  assert.equal((u32Frame1[3] & UNIFORM_FLAG_SWAP_CHROMA) !== 0, true);

  // Frame 2 switches per-message to chromaOrder='yuv' and loadOp='clear'.
  const planes2 = makeYuv420FramePlanes(256, 16);
  await handler.handleMessage({
    type: 'present-splash-frame',
    id: 'w165-2',
    chromaOrder: 'yuv',
    loadOp: 'clear',
    frame: {
      width: 256,
      height: 16,
      format: 'yuv420p',
      planes: planes2,
      frameIndex: 1,
    },
  });

  const ack2 = posted.find((m) => m && m.type === 'splash-frame-presented' && m.id === 'w165-2');
  assert.ok(ack2 && ack2.ok === true);
  assert.equal(processor.binkPresenter.chromaOrder, 'yuv');
  assert.equal(processor.binkPresenter.loadOp, 'clear');
  assert.equal(mock.renderPasses.length, 2);
  assert.equal(mock.renderPasses[1].passDesc.colorAttachments[0].loadOp, 'clear');
  assert.ok(mock.renderPasses[1].passDesc.colorAttachments[0].clearValue);

  const u32Frame2 = new Uint32Array(
    mock.writeBufferCalls[mock.writeBufferCalls.length - 1].data.buffer,
    mock.writeBufferCalls[mock.writeBufferCalls.length - 1].data.byteOffset,
    4
  );
  assert.equal((u32Frame2[3] & UNIFORM_FLAG_SWAP_CHROMA) !== 0, false);

  // 6b. Direct BinkVideoPresenter fallback (SplashFrameBridge: null) forwards chromaOrder + loadOp directly.
  const directMock = createRichW185MockWebGpu();
  const directPresenter = new BinkVideoPresenter({ device: directMock.device });
  const directProc = new WgpuCommandProcessor(directMock.device, directMock.ctx, {
    binkPresenter: directPresenter,
    SplashFrameBridge: null,
  });

  const directRes = directProc.presentSplashFrame(
    {
      width: 256,
      height: 8,
      format: 'yuv420p',
      planes: makeYuv420FramePlanes(256, 8),
      chromaOrder: 'yvu',
      loadOp: 'load',
      frameIndex: 0,
    }
  );
  assert.equal(directRes.ok, true);
  assert.equal(directRes.chromaOrder, 'yvu');
  assert.equal(directRes.loadOp, 'load');
  assert.equal(directPresenter.chromaOrder, 'yvu');
  assert.equal(directPresenter.loadOp, 'load');
  assert.equal(directMock.renderPasses[0].passDesc.colorAttachments[0].loadOp, 'load');
  assert.equal(directProc.getStats().binkPresenter.chromaOrder, 'yvu');
  assert.equal(directProc.getStats().binkPresenter.loadOp, 'load');

  // 6c. Minimal mock presenter lacking W165 setChromaOrder/setLoadOp keeps working safely.
  const minimalMock = createRichW185MockWebGpu();
  const minimalPresenter = {
    uploadYuv420Frame() {},
    presentFrame() {
      return { presented: true };
    },
  };
  const minimalProc = new WgpuCommandProcessor(minimalMock.device, minimalMock.ctx, {
    binkPresenter: minimalPresenter,
    SplashFrameBridge: null,
    binkChromaOrder: 'yvu',
    binkLoadOp: 'load',
  });
  const minRes = minimalProc.presentSplashFrame({
    width: 16,
    height: 8,
    format: 'yuv420p',
    planes: makeYuv420FramePlanes(16, 8),
    chromaOrder: 'yvu',
    loadOp: 'load',
  });
  assert.equal(minRes.ok, true);

  processor.destroy();
  directProc.destroy();
  minimalProc.destroy();
});

// ---------------------------------------------------------------------------
// 7. W185 / W167: Default fallback sampler synthesis on bind-group creation
// ---------------------------------------------------------------------------

test('W185 (W167) synthesizes default fallback samplers via pipelineCache when reflected or sentinel sampler bindings are missing', async () => {
  const mock = createRichW185MockWebGpu();
  const pipelineCache = new WebGpuPipelineCache({ device: mock.device });
  const reflectionRegistry = new ReflectionRegistry();

  const TEX_FS_WGSL = [
    '@group(0) @binding(0) var u_tex: texture_2d<f32>;',
    '@group(0) @binding(1) var u_smp: sampler;',
    '@fragment fn fs_main() -> @location(0) vec4<f32> { return textureSample(u_tex, u_smp, vec2<f32>(0.5)); }',
  ].join('\n');

  const fsHash = wgpkFormatHashHex(wgpkFnv1a64(TEX_FS_WGSL));
  const fsBglEntries = [
    {
      binding: 0,
      visibility: SHADER_STAGE_BITS.FRAGMENT,
      texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
    },
    {
      binding: 1,
      visibility: SHADER_STAGE_BITS.FRAGMENT,
      sampler: { type: 'filtering' },
    },
  ];
  const fsBglId = computeBindGroupLayoutHash({ entries: fsBglEntries });
  const fsPlId = computePipelineLayoutHash([fsBglId]);

  reflectionRegistry.loadManifest({
    schema_version: 1,
    register_convention: { source: 'dxc-spirv', shifts: { b: 0, t: 0, u: 0, s: 0 } },
    bind_group_layouts: {
      [fsBglId]: { entries: fsBglEntries },
    },
    pipeline_layouts: {
      [fsPlId]: { bindGroupLayouts: [fsBglId] },
    },
    shaders: {
      [fsHash]: {
        stage: 'fragment',
        entry_point: 'fs_main',
        name: 'w185_fs',
        pipeline_layout: fsPlId,
        bind_group_layouts: [fsBglId],
        layout_complete: true,
        fragment_outputs: [{ location: 0, type: 'vec4<f32>' }],
        bindings: [
          { group: 0, binding: 0, name: 'u_tex', resource: 'texture', in_layout: true },
          { group: 0, binding: 1, name: 'u_smp', resource: 'sampler', in_layout: true },
        ],
      },
    },
  });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    pipelineCache,
    reflectionRegistry,
  });

  // Create VS + FS + reflected pipeline 10 + texture 100, then CreateBindGroup with ONLY texture @0 (no sampler @1!).
  const builder = new BinaryCommandBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, VS_WGSL)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, TEX_FS_WGSL)
    .createRenderPipeline(10, 1, 2, 0);
  const texChunk = buildCreateTexturePacket(100, 16, 16).finish();
  const bgChunk = buildCreateBindGroupPacket(50, 10, 0, [
    { binding: 0, resourceType: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 100 },
  ]).finish();

  processor.executeCommandBuffer(builder.finish());
  processor.executeCommandBuffer(texChunk);
  processor.executeCommandBuffer(bgChunk);

  // Zero validation errors, bind group created with synthesized filtering sampler at binding 1!
  assert.equal(processor.validationErrorCount, 0);
  assert.equal(processor.bindGroups.has(50), true);
  assert.equal(mock.samplerDescs.length, 1);
  assert.equal(mock.samplerDescs[0].magFilter, 'linear');
  assert.equal(mock.samplerDescs[0].minFilter, 'linear');
  assert.equal(mock.samplerDescs[0].addressModeU, 'clamp-to-edge');

  const createdBg = mock.bindGroupDescs[0];
  assert.equal(createdBg.entries.length, 2);
  assert.equal(createdBg.entries[0].binding, 0);
  assert.equal(createdBg.entries[1].binding, 1);
  assert.equal(createdBg.entries[1].resource.kind, 'GPUSampler');

  // Second bind group reuses the cached default sampler (zero extra createSampler calls).
  processor.executeCommandBuffer(
    buildCreateBindGroupPacket(51, 10, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 100 },
    ]).finish()
  );
  assert.equal(mock.samplerDescs.length, 1);
  assert.equal(processor.getStats().fallbackSamplersCreated, 1);
  assert.equal(processor.getStats().fallbackSamplerBindingsSynthesized, 2);

  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 3000 });
  assert.deepEqual(snap.fallbackSamplerStats, {
    fallbackSamplersCreated: 1,
    fallbackSamplerBindingsSynthesized: 2,
    fallbackSamplerCount: 1,
  });

  // Sentinel sampler resourceId 0 also synthesizes a default filtering sampler without error;
  // unknown non-zero sampler IDs still record a validation error.
  processor.executeCommandBuffer(
    buildCreateBindGroupPacket(52, 10, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 100 },
      { binding: 1, resourceType: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 0 },
    ]).finish()
  );
  assert.equal(processor.bindGroups.has(52), true);
  assert.equal(processor.validationErrorCount, 0);

  processor.executeCommandBuffer(
    buildCreateBindGroupPacket(53, 10, 0, [
      { binding: 1, resourceType: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 9999 },
    ]).finish()
  );
  assert.equal(processor.bindGroups.has(53), false);
  assert.equal(processor.validationErrorCount, 1);

  // Device without createSampler skips fallback synthesis gracefully without throwing.
  const noSamplerMock = createRichW185MockWebGpu({ omitCreateSampler: true });
  const noSamplerProc = new WgpuCommandProcessor(noSamplerMock.device, noSamplerMock.ctx, {
    format: 'bgra8unorm',
  });
  assert.equal(noSamplerProc.getOrCreateDefaultSampler('filtering'), null);

  processor.destroy();
  noSamplerProc.destroy();
});

// ---------------------------------------------------------------------------
// 8. W185 / W168: Graphics-over-compute async pipeline priority & latency telemetry
// ---------------------------------------------------------------------------

test('W185 (W168) configures AsyncPipelineCache with prioritizeGraphics and maxConcurrentCompiles and surfaces compile latency telemetry', async () => {
  assert.equal(DEFAULT_MAX_CONCURRENT_PIPELINE_COMPILES, 2);

  let fakeNow = 100;
  const mock = createRichW185MockWebGpu({ asyncPipelines: true });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    maxConcurrentCompiles: 2,
    prioritizeGraphics: true,
    now: () => fakeNow,
  });
  installRenderStatePackets(processor);

  assert.equal(processor.asyncPipelineCache.prioritizeGraphics, true);
  assert.equal(processor.asyncPipelineCache.maxConcurrentCompiles, 2);
  assert.equal(processor.asyncPipelineCache.maxConcurrent, 2);

  // Enqueue 4 background compute pipelines first (2 start immediately, 2 queue), then 2 UI render pipelines!
  const stream = new BinaryCommandBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, VS_WGSL)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_WGSL)
    .createShaderModule(3, SHADER_STAGE.COMPUTE, CS_WGSL)
    .createComputePipeline2(101, 3, 0)
    .createComputePipeline2(102, 3, 0)
    .createComputePipeline2(103, 3, 0)
    .createComputePipeline2(104, 3, 0)
    .createRenderPipeline2(201, 1, 2, 0)
    .createRenderPipeline2(202, 1, 2, 0)
    .finish();

  processor.executeCommandBuffer(stream);

  // At most 2 compiles in flight; both initial slots are the first two compute jobs.
  assert.equal(mock.asyncCompiles.length, 2);
  assert.deepEqual(
    mock.asyncCompiles.map((c) => c.kind),
    ['compute', 'compute']
  );

  // Complete the first two compute jobs after 15ms and 35ms -> next 2 dispatched MUST be the 2 render pipelines!
  fakeNow = 115;
  mock.asyncCompiles[0].resolve();
  await new Promise((r) => setImmediate(r));
  fakeNow = 135;
  mock.asyncCompiles[1].resolve();
  await new Promise((r) => setImmediate(r));

  assert.equal(mock.asyncCompiles.length, 4);
  assert.deepEqual(
    mock.asyncCompiles.slice(2, 4).map((c) => c.kind),
    ['render', 'render'],
    'graphics render pipelines must be promoted ahead of queued background compute pipelines'
  );

  // Resolve both render pipelines (10ms each) and the remaining 2 compute pipelines (20ms each).
  fakeNow = 145;
  mock.asyncCompiles[2].resolve();
  mock.asyncCompiles[3].resolve();
  await new Promise((r) => setImmediate(r));

  assert.equal(mock.asyncCompiles.length, 6);
  assert.deepEqual(
    mock.asyncCompiles.slice(4, 6).map((c) => c.kind),
    ['compute', 'compute']
  );
  fakeNow = 165;
  mock.asyncCompiles[4].resolve();
  mock.asyncCompiles[5].resolve();
  await processor.asyncPipelineCache.whenIdle();
  await new Promise((r) => setImmediate(r));

  // Verify compile latency & priority telemetry on getStats(), gpu-stats reply, and buildGpuTelemetrySnapshot().
  const stats = processor.getStats();
  assert.ok(stats.asyncPipelineStats);
  assert.equal(stats.asyncPipelineStats.compiled, 6);
  assert.equal(stats.asyncPipelineStats.prioritizeGraphics, true);
  assert.equal(stats.asyncPipelineStats.maxConcurrentCompiles, 2);
  assert.equal(stats.asyncPipelineStats.peakConcurrentCompiles, 2);
  assert.ok(stats.asyncPipelineStats.priorityPromotions >= 2);
  assert.ok(stats.asyncPipelineStats.avgCompileMs > 0);
  assert.equal(stats.asyncPipelineStats.maxCompileMs, 35);

  assert.ok(stats.asyncPipelineCompileStats);
  assert.equal(stats.asyncPipelineCompileStats.prioritizeGraphics, true);
  assert.equal(stats.asyncPipelineCompileStats.maxConcurrentCompiles, 2);
  assert.equal(stats.asyncPipelineCompileStats.peakConcurrentCompiles, 2);
  assert.equal(stats.asyncPipelineCompileStats.maxCompileMs, 35);
  assert.ok(stats.asyncPipelineCompileStats.priorityPromotions >= 2);

  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 4000 });
  assert.equal(snap.asyncPipelineStats.compiled, 6);
  assert.equal(snap.asyncPipelineStats.maxConcurrent, 2);
  assert.equal(snap.asyncPipelineCompileStats.prioritizeGraphics, true);
  assert.equal(snap.asyncPipelineCompileStats.maxConcurrentCompiles, 2);
  assert.equal(snap.asyncPipelineCompileStats.peakConcurrentCompiles, 2);
  assert.equal(snap.asyncPipelineCompileStats.maxCompileMs, 35);
  assert.ok(snap.asyncPipelineCompileStats.avgCompileMs > 0);
  assert.ok(snap.asyncPipelineCompileStats.priorityPromotions >= 2);

  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
  });
  await handler.handleMessage({ id: 'w168-stats', type: 'request-gpu-stats' });
  const reply = posted.find((m) => m && m.id === 'w168-stats' && m.type === 'gpu-stats');
  assert.ok(reply);
  assert.equal(reply.asyncPipelineCompileStats.maxCompileMs, 35);
  assert.equal(reply.asyncPipelineCompileStats.prioritizeGraphics, true);
  assert.equal(reply.asyncPipelineStats.compiled, 6);

  // Also verify maxConcurrentCompiles: 0 (unlimited) and prioritizeGraphics: false options.
  const unlimitedProc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    maxConcurrentCompiles: 0,
    prioritizeGraphics: false,
  });
  assert.equal(unlimitedProc.asyncPipelineCache.maxConcurrentCompiles, 0);
  assert.equal(unlimitedProc.asyncPipelineCache.maxConcurrent, 0);
  assert.equal(unlimitedProc.asyncPipelineCache.prioritizeGraphics, false);

  processor.destroy();
  unlimitedProc.destroy();
});

// ---------------------------------------------------------------------------
// 9. W185 / W179: Categorized missing-shader summary export & sanitization
// ---------------------------------------------------------------------------

test('W185 (W179) exports bounded, sanitized missingShaderSummary on getStats, gpu-stats, missingShaders, and gpu-telemetry', async () => {
  assert.equal(DEFAULT_MISSING_SUMMARY_LIMIT, 32);

  const badCompileWgsl = '@fragment fn broken_fs() { invalid_token; }';
  const mock = createRichW185MockWebGpu({
    onCreateShaderModule(desc, nextId) {
      if (desc.code === badCompileWgsl) {
        throw new Error('WGSL parse error at C:\\Users\\secret\\shader.wgsl');
      }
      return { kind: 'GPUShaderModule', id: nextId(), code: desc.code };
    },
  });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    deferRangedFallbackPipelines: false,
  });

  // Trigger not_in_pack misses + compile_failed + reflection_failed + real-shader guard.
  const missVsHash = '0x1111222233334444';
  const missFsHash = '0x5555666677778888';
  const reflFailHash = '0x9999aaaabbbbcccc';
  const realHash = '0xdeadbeef00001234';

  processor.shaderFallbackRouter.registerRealShader(realHash, 'fragment', {
    wgsl: FS_WGSL,
    source: 'real',
  });
  processor.shaderFallbackRouter.recordMissingShader(reflFailHash, 'compute', 'reflection_failed');

  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(1, SHADER_STAGE.VERTEX, missVsHash)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, missFsHash)
      .createShaderModule(3, SHADER_STAGE.FRAGMENT, missFsHash)
      .createShaderModule(4, SHADER_STAGE.FRAGMENT, badCompileWgsl)
      .createShaderModule(5, SHADER_STAGE.FRAGMENT, realHash)
      .finish()
  );

  const summary = processor.getMissingShaderSummary({ limit: 2 });
  assert.ok(summary);
  assert.equal(summary.totalMissingEvents, 5); // 3 not_in_pack + 1 reflection_failed + 1 compile_failed
  assert.equal(summary.countsByReason.not_in_pack, 3);
  assert.equal(summary.countsByReason.reflection_failed, 1);
  assert.equal(summary.countsByReason.compile_failed, 1);
  assert.equal(summary.countsByReason.other, 0);
  assert.equal(summary.countsByStage.vertex, 1);
  assert.equal(summary.countsByStage.fragment, 3);
  assert.equal(summary.countsByStage.compute, 1);
  assert.equal(summary.realShaderGuard.registeredRealShaders, 1);
  assert.equal(summary.realShaderGuard.blockedOverrideCount, 1);
  assert.equal(summary.topMissing.length, 2, 'limit must cap topMissing array');
  assert.equal(summary.topMissing[0].hashHex, missFsHash);
  assert.equal(summary.topMissing[0].hitCount, 2);

  // Verify getStats(), buildGpuTelemetrySnapshot(), request-gpu-stats, and missingShaders worker replies.
  const stats = processor.getStats();
  assert.ok(stats.missingShaderSummary);
  assert.equal(stats.missingShaderSummary.countsByReason.compile_failed, 1);
  assert.equal(stats.missingShaderSummary.countsByReason.not_in_pack, 3);

  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 5000, limit: 2 });
  assert.ok(snap.missingShaderSummary);
  assert.equal(snap.missingShaderSummary.totalMissingEvents, 5);
  assert.equal(snap.missingShaderSummary.topMissing.length, 2);

  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
  });
  await handler.handleMessage({ id: 'w179-stats', type: 'request-gpu-stats' });
  await handler.handleMessage({ id: 'w179-miss', missingShaders: true, limit: 2 });

  const statsReply = posted.find((m) => m && m.id === 'w179-stats' && m.type === 'gpu-stats');
  assert.ok(statsReply && statsReply.missingShaderSummary);
  assert.equal(statsReply.missingShaderSummary.countsByReason.compile_failed, 1);

  const missReply = posted.find((m) => m && m.id === 'w179-miss');
  assert.ok(missReply && missReply.missingShaderSummary);
  assert.equal(missReply.missingShaderSummary.topMissing.length, 2);

  // Sanitization strips host paths, raw WGSL, raw buffers, and invalid hashes from untrusted summaries.
  const sanitized = sanitizeMissingShaderSummary(
    {
      totalMissingEvents: 9,
      uniqueMissingHashes: 4,
      maxTrackedHashes: 8192,
      overflowCount: 0,
      countsByReason: { not_in_pack: 9, evilKey: 'C:\\Users\\secret\\game.exe' },
      countsByStage: { vertex: 9 },
      topMissing: [
        { hashHex: 'C:\\Users\\secret\\bad.rpf', stage: 'vertex', reason: 'not_in_pack', hitCount: 99 },
        {
          hashHex: '0x000000000000abcd',
          stage: 'ps',
          reason: 'compile_failed',
          hitCount: 5,
          hostPath: '/Users/secret/gta6.rpf',
          wgsl: '@fragment fn leaked() {}',
          rawBytes: new Uint8Array([1, 2, 3]),
        },
        { hashHex: '0x000000000000beef', stage: 'vs', reason: 'not_in_pack', hitCount: 1 },
      ],
    },
    1
  );
  assert.equal(sanitized.topMissing.length, 1);
  assert.equal(sanitized.topMissing[0].hashHex, '0x000000000000abcd');
  assert.equal(sanitized.topMissing[0].stage, 'fragment');
  assert.equal(sanitized.topMissing[0].reason, 'compile_failed');
  const json = JSON.stringify(sanitized);
  assert.equal(json.includes('secret'), false);
  assert.equal(json.includes('leaked'), false);
  assert.equal(json.includes('rawBytes'), false);

  // Legacy minimal mock shaderFallbackRouter without getMissingShaderSummary keeps working safely.
  const legacyProc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    shaderFallbackRouter: {
      resolve() {
        return { wgsl: VS_WGSL, source: 'builtin' };
      },
    },
  });
  assert.equal(legacyProc.getMissingShaderSummary(), null);
  assert.equal(buildGpuTelemetrySnapshot(legacyProc, { timestamp: 5001 }).missingShaderSummary, null);

  processor.destroy();
  legacyProc.destroy();
});

// ---------------------------------------------------------------------------
// 10. W185 classic WorkerGlobalScope importScripts smoke test
// ---------------------------------------------------------------------------

test('W185 classic WorkerGlobalScope importScripts wires W165/W167/W168/W179 cleanly with zero errors', async () => {
  const mock = createRichW185MockWebGpu({ asyncPipelines: true, swapWidth: 1280, swapHeight: 720 });
  const posted = [];
  const sandbox = {
    console,
    TextEncoder,
    TextDecoder,
    URL,
    performance,
    setTimeout,
    clearTimeout,
    navigator: mock.navigator,
    postMessage(msg) {
      posted.push(msg);
    },
  };
  const vmCtx = vm.createContext(sandbox);
  vmCtx.self = vmCtx;
  vmCtx.importScripts = (...urls) => {
    for (const rel of urls) {
      const fullPath = path.resolve(WORKERS_DIR, rel);
      vm.runInContext(readFileSync(fullPath, 'utf8'), vmCtx, { filename: fullPath });
    }
  };

  vm.runInContext(
    readFileSync(path.resolve(WORKERS_DIR, './wgpu_worker.js'), 'utf8'),
    vmCtx,
    { filename: path.resolve(WORKERS_DIR, './wgpu_worker.js') }
  );

  await vmCtx.self.onmessage({
    data: {
      init: true,
      canvas: mock.canvas,
      width: 1280,
      height: 720,
      enableSplashBridge: true,
      enablePipelineCache: true,
      binkChromaOrder: 'yvu',
      binkLoadOp: 'load',
      maxConcurrentCompiles: 2,
      prioritizeGraphics: true,
    },
  });

  const vmY = vm.runInContext('new Uint8Array(256 * 8).fill(190)', vmCtx);
  const vmU = vm.runInContext('new Uint8Array(128 * 4).fill(120)', vmCtx);
  const vmV = vm.runInContext('new Uint8Array(128 * 4).fill(140)', vmCtx);
  await vmCtx.self.onmessage({
    data: {
      type: 'bink-frame',
      id: 'vm-w185-bink',
      frame: {
        width: 256,
        height: 8,
        format: 'yuv420p',
        planes: { y: vmY, u: vmU, v: vmV },
        frameIndex: 0,
      },
    },
  });

  await vmCtx.self.onmessage({ data: { id: 'vm-w185-stats', type: 'request-gpu-stats' } });
  await vmCtx.self.onmessage({ data: { id: 'vm-w185-tel', type: 'request-gpu-telemetry' } });

  const errors = posted.filter((m) => m && m.error);
  assert.deepEqual(errors, [], `unexpected classic worker errors: ${JSON.stringify(errors)}`);

  const binkAck = posted.find((m) => m && m.id === 'vm-w185-bink');
  assert.ok(binkAck && binkAck.ok === true);
  assert.equal(mock.renderPasses[0].passDesc.colorAttachments[0].loadOp, 'load');

  const statsReply = posted.find((m) => m && m.id === 'vm-w185-stats' && m.type === 'gpu-stats');
  assert.ok(statsReply);
  assert.equal(statsReply.stats.binkPresenter.chromaOrder, 'yvu');
  assert.equal(statsReply.stats.binkPresenter.loadOp, 'load');
  assert.equal(statsReply.asyncPipelineCompileStats.prioritizeGraphics, true);
  assert.equal(statsReply.asyncPipelineCompileStats.maxConcurrentCompiles, 2);
  assert.ok(statsReply.missingShaderSummary);

  const telReply = posted.find((m) => m && m.id === 'vm-w185-tel' && m.type === 'gpu-telemetry');
  assert.ok(telReply);
  assert.equal(telReply.asyncPipelineCompileStats.prioritizeGraphics, true);
  assert.equal(telReply.asyncPipelineCompileStats.maxConcurrentCompiles, 2);
  assert.ok(telReply.missingShaderSummary);
});

// ---------------------------------------------------------------------------
// 11. Regression: pending ranged-streaming pipelines skip SetPipeline/DrawIndexed/Dispatch
//     cleanly with 0 validation errors, materialize on resolve, and still error on unknown IDs
// ---------------------------------------------------------------------------

test('pending ranged-streaming render pipelines skip SetPipeline/DrawIndexed with 0 validation errors and materialize once shaders settle', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const net = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 7_510_000_000, // 7.51 GB WGPK v2 scale
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: false,
    streamShaders: true,
  });
  const rsHandler = installRenderStatePackets(processor);

  await processor.loadShaderPackUrl('/shaders/full-7gb.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });
  assert.equal(processor.rangedPackActive, true);

  // Frame 1: create v1 pipeline 12285 and v2 pipeline 94611 referencing streaming ranged shaders,
  // then immediately issue SetPipeline + DrawIndexed while HTTP 206 range fetches are in flight.
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .createBuffer(50, 64, 0x10)
      .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
      .createRenderPipeline(12285, 1, 2, 0)
      .createRenderPipeline2(94611, 1, 2, 0)
      .setIndexBuffer(50, 0, 0)
      .setPipeline(12285)
      .drawIndexed(6, 1, 0, 0, 0)
      .setPipeline(94611)
      .drawIndexed(6, 1, 0, 0, 0)
      .present()
      .finish()
  );

  // Zero validation errors while both pipelines are pending async ranged shader fetches.
  assert.equal(processor.validationErrorCount, 0);
  assert.equal(processor.getValidationErrors().length, 0);
  assert.equal(processor.isAsyncPipelinePending(12285), true);
  assert.equal(processor.isAsyncPipelinePending(94611), true);
  assert.equal(rsHandler.isAsyncPipelinePending(12285, false), true);
  assert.equal(rsHandler.isAsyncPipelinePending(94611, false), true);
  assert.equal(processor.pipelines.has(12285), false);
  assert.equal(processor.pipelines.has(94611), false);
  assert.equal(processor.getAsyncPipelineStats().pendingAsyncPipelineSkips, 2);

  // Resolve ranged fetches -> both pipelines materialize automatically with real pack shaders.
  net.resolveAllDeferred();
  await processor.flushPendingRangedShaders();

  assert.equal(processor.asyncShaderPendingCount, 0);
  assert.equal(processor.isAsyncPipelinePending(12285), false);
  assert.equal(processor.isAsyncPipelinePending(94611), false);
  assert.equal(rsHandler.isAsyncPipelinePending(12285, false), false);
  assert.equal(rsHandler.isAsyncPipelinePending(94611, false), false);
  assert.equal(processor.pipelines.has(12285), true);
  assert.equal(processor.pipelines.has(94611), true);
  assert.equal(processor.pipelines.get(12285).desc.vertex.module.code, VS_WGSL);
  assert.equal(processor.pipelines.get(94611).desc.fragment.module.code, FS_WGSL);
  assert.equal(processor.validationErrorCount, 0);

  // Frame 2: SetPipeline + DrawIndexed now execute cleanly with real pipelines.
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .setIndexBuffer(50, 0, 0)
      .setPipeline(12285)
      .drawIndexed(6, 1, 0, 0, 0)
      .setPipeline(94611)
      .drawIndexed(6, 1, 0, 0, 0)
      .present()
      .finish()
  );

  assert.equal(processor.getStats().realPackDrawCount, 2);
  assert.equal(processor.getStats().skippedDrawCount, 0);
  assert.equal(processor.validationErrorCount, 0);

  processor.destroy();
});

test('live Chrome streamShaders=1 + asyncPipelines: rejected fallback PSO stays pending with 0 errors and upgrades cleanly; dedupe microtask gap stays pending', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const net = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 7_510_000_000,
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mock = createRichW185MockWebGpu({ asyncPipelines: true });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    streamShaders: true,
    enableShaderFallback: true,
    deferRangedFallbackPipelines: false,
  });
  const rsHandler = installRenderStatePackets(processor);
  processor.asyncPipelineCache.coalesceByDescriptor = true;

  await processor.loadShaderPackUrl('/shaders/full-live.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });

  // Create streaming VS/FS + CreateRenderPipeline2(12285) while fetches are in flight.
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .createBuffer(50, 64, 0x10)
      .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
      .createRenderPipeline2(12285, 1, 2, 0)
      .finish()
  );

  // Simulate Chrome Dawn rejecting the temporary fallback PSO (e.g. MRT/target mismatch).
  assert.equal(mock.asyncCompiles.length, 1);
  mock.asyncCompiles[0].reject('GPUValidationError: fallback shader incompatible with target format');
  await processor.asyncPipelineCache.whenIdle();
  await new Promise((r) => setImmediate(r));

  // Even after fallback compile rejected, 12285 is still waiting on ranged shaders -> 0 errors!
  assert.equal(processor.validationErrorCount, 0);
  assert.equal(rsHandler.telemetry.asyncPipelineFailures, 0);
  assert.equal(rsHandler.stats.errors, 0);
  assert.equal(processor.isAsyncPipelinePending(12285), true);
  assert.equal(rsHandler.isAsyncPipelinePending(12285, false), true);

  // DrawIndexed while ranged shaders are still in flight skips cleanly with 0 validation errors.
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .setIndexBuffer(50, 0, 0)
      .setPipeline(12285)
      .drawIndexed(6, 1, 0, 0, 0)
      .present()
      .finish()
  );
  assert.equal(processor.validationErrorCount, 0);
  assert.equal(processor.getStats().skippedDrawCount, 0);
  assert.equal(processor.getAsyncPipelineStats().pendingAsyncPipelineSkips, 1);

  // Now ranged shaders resolve -> triggers async compile of the REAL pack pipeline!
  net.resolveAllDeferred();
  await processor.flushPendingRangedShaders();
  assert.equal(mock.asyncCompiles.length, 2);
  assert.equal(mock.asyncCompiles[1].desc.vertex.module.code, VS_WGSL);
  assert.equal(mock.asyncCompiles[1].desc.fragment.module.code, FS_WGSL);
  assert.equal(processor.isAsyncPipelinePending(12285), true);

  // Resolve real async pipeline compile -> 12285 materializes!
  mock.asyncCompiles[1].resolve();
  await processor.asyncPipelineCache.whenIdle();
  await new Promise((r) => setImmediate(r));
  assert.equal(processor.pipelines.has(12285), true);
  assert.equal(processor.isAsyncPipelinePending(12285), false);

  // Create second pipeline 56795 with identical descriptor -> hits AsyncPipelineCache dedupe
  // synchronously (asyncCache.isPending(56795) is false before .then() runs). Verify
  // processor.isAsyncPipelinePending(56795) stays true during the microtask gap (0 errors)!
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .createRenderPipeline2(56795, 1, 2, 0)
      .setIndexBuffer(50, 0, 0)
      .setPipeline(56795)
      .drawIndexed(6, 1, 0, 0, 0)
      .present()
      .finish()
  );
  assert.equal(processor.validationErrorCount, 0);
  assert.equal(processor.asyncPipelineCache.isPending(56795), false);
  assert.equal(processor.pipelines.has(56795), false);
  assert.equal(processor.isAsyncPipelinePending(56795), true);

  await new Promise((r) => setImmediate(r));
  assert.equal(processor.pipelines.has(56795), true);
  assert.equal(processor.isAsyncPipelinePending(56795), false);

  // Frame 3: both 12285 and 56795 draw cleanly with real pack shaders.
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .setIndexBuffer(50, 0, 0)
      .setPipeline(12285)
      .drawIndexed(6, 1, 0, 0, 0)
      .setPipeline(56795)
      .drawIndexed(6, 1, 0, 0, 0)
      .present()
      .finish()
  );
  assert.equal(processor.getStats().realPackDrawCount, 2);
  assert.equal(processor.validationErrorCount, 0);

  processor.destroy();
});

test('pending ranged compute pipeline skips Dispatch cleanly with 0 errors and pipelineCache fast-key upgrades on shader resolve', async () => {
  const { buffer, dataStart } = buildSamplePack();
  // Part A: deferred compute pipeline (enableShaderFallback: false) without and with render_state_packets.
  const netA = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 256 * 1024 * 1024,
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mockA = createMockWebGpu();
  const procA = new WgpuCommandProcessor(mockA.device, mockA.ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: false,
  });
  await procA.loadShaderPackUrl('/shaders/comp-a.wgsl.pack', {
    fetch: netA.fetchImpl,
    initialRangeBytes: dataStart,
  });

  procA.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .createShaderModule(3, SHADER_STAGE.COMPUTE, CS_HASH)
      .createComputePipeline(18466, 3)
      .setComputePipeline(18466)
      .dispatchCompute(4, 1, 1)
      .present()
      .finish()
  );

  // Must skip cleanly without dispatching on an unbound compute pass or recording validation errors.
  assert.equal(procA.validationErrorCount, 0);
  assert.equal(procA.getStats().computeDispatches, 0);
  assert.equal(procA.isAsyncPipelinePending(18466), true);
  assert.equal(procA.getAsyncPipelineStats().pendingAsyncPipelineSkips, 1);

  netA.resolveAllDeferred();
  await procA.flushPendingRangedShaders();
  assert.equal(procA.computePipelines.has(18466), true);
  assert.equal(procA.computePipelines.get(18466).desc.compute.module.code, CS_WGSL);

  procA.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .setComputePipeline(18466)
      .dispatchCompute(4, 1, 1)
      .present()
      .finish()
  );
  assert.equal(procA.getStats().computeDispatches, 1);
  assert.equal(procA.validationErrorCount, 0);
  procA.destroy();

  // Part B: with pipelineCache enabled, ranged shader upgrade bumps shaderModuleGenerations so
  // getFastComputePipeline compiles the real shader instead of returning the cached fallback!
  const netB = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 256 * 1024 * 1024,
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mockB = createMockWebGpu();
  const procB = new WgpuCommandProcessor(mockB.device, mockB.ctx, {
    format: 'bgra8unorm',
    enablePipelineCache: true,
    enableShaderFallback: true,
    deferRangedFallbackPipelines: false,
  });
  await procB.loadShaderPackUrl('/shaders/comp-b.wgsl.pack', {
    fetch: netB.fetchImpl,
    initialRangeBytes: dataStart,
  });

  procB.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .createShaderModule(3, SHADER_STAGE.COMPUTE, CS_HASH)
      .createComputePipeline(20, 3)
      .present()
      .finish()
  );
  const fallbackComp = procB.computePipelines.get(20);
  assert.ok(fallbackComp);
  assert.notEqual(fallbackComp.desc.compute.module.code, CS_WGSL);

  netB.resolveAllDeferred();
  await procB.flushPendingRangedShaders();
  const realComp = procB.computePipelines.get(20);
  assert.ok(realComp);
  assert.notEqual(realComp, fallbackComp);
  assert.equal(realComp.desc.compute.module.code, CS_WGSL);
  assert.equal(procB.validationErrorCount, 0);
  procB.destroy();
});

test('genuinely unknown and failed pipeline IDs still record SetPipeline and DrawIndexed validation errors', async () => {
  const { buffer, dataStart } = buildSamplePack();
  const net = createSyntheticRangePackServer(buffer, {
    reportedTotalBytes: 256 * 1024 * 1024,
    dataStartOffset: dataStart,
    deferDataRanges: true,
    rejectFullGet: true,
  });
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: false,
  });
  installRenderStatePackets(processor);

  await processor.loadShaderPackUrl('/shaders/full-err.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });

  // 1. Genuinely unknown pipeline ID 77777 (never created) MUST still record validation errors!
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .createBuffer(50, 64, 0x10)
      .setIndexBuffer(50, 0, 0)
      .setPipeline(77777)
      .drawIndexed(6, 1, 0, 0, 0)
      .present()
      .finish()
  );

  const unknownErrors = processor.getValidationErrors().map((e) => e.message);
  assert.equal(processor.validationErrorCount, 2);
  assert.ok(unknownErrors.some((m) => /render pipeline 77777 not found/.test(m)));
  assert.ok(unknownErrors.some((m) => /skipped draw: render pipeline 77777 is unavailable/.test(m)));

  // 2. Pipeline whose ranged shader fetch genuinely fails MUST record validation errors after failure settles!
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(1, SHADER_STAGE.VERTEX, VS_HASH)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, FS_HASH)
      .createRenderPipeline(88888, 1, 2, 0)
      .finish()
  );
  assert.equal(processor.isAsyncPipelinePending(88888), true);

  // Reject all deferred range fetches with a non-retryable HTTP 404 error.
  while (net.deferredRanges.length > 0) {
    const err = new Error('HTTP 404 range fetch error');
    err.httpStatus = 404;
    net.deferredRanges.shift().reject(err);
  }
  await processor.flushPendingRangedShaders();

  assert.equal(processor.isAsyncPipelinePending(88888), false);
  assert.equal(processor.pipelines.has(88888), false);

  const countBeforeDraw = processor.validationErrorCount;
  processor.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .setIndexBuffer(50, 0, 0)
      .setPipeline(88888)
      .drawIndexed(6, 1, 0, 0, 0)
      .present()
      .finish()
  );
  assert.equal(processor.validationErrorCount, countBeforeDraw + 2);
  const allErrors = processor.getValidationErrors().map((e) => e.message);
  assert.ok(allErrors.some((m) => /render pipeline 88888 not found/.test(m)));
  assert.ok(allErrors.some((m) => /skipped draw: render pipeline 88888 is unavailable/.test(m)));

  processor.destroy();
});
