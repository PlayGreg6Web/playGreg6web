import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  WgslPackCache,
  buildWgslPackBuffer,
  loadIndexFromUrl,
  fnv1a64 as wgpkFnv1a64,
  formatHashHex as wgpkFormatHashHex,
  SHADER_STAGE as WGPK_SHADER_STAGE,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const {
  WgpuCommandProcessor,
  buildGpuTelemetrySnapshot,
  OPCODES,
  SHADER_STAGE,
} = require('../web/workers/wgpu_worker.js');
const { ShaderFallbackRouter } = require('../web/workers/shader_fallback_router.js');
const { installRenderStatePackets } = require('../web/workers/render_state_packets.js');

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

  setPipeline(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this._pushPacket(OPCODES.SET_PIPELINE, new Uint8Array(buf));
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

function createMockWebGpu(options = {}) {
  let nextId = 1;
  const shaderModulesCreated = [];
  const renderPipelinesCreated = [];

  const device = {
    features: new Set(),
    createShaderModule(desc) {
      if (typeof options.onCreateShaderModule === 'function') {
        const custom = options.onCreateShaderModule(desc, nextId);
        if (custom !== undefined) {
          shaderModulesCreated.push(custom);
          return custom;
        }
      }
      const mod = {
        kind: 'GPUShaderModule',
        id: nextId++,
        code: desc.code,
      };
      if (typeof options.makeCompilationInfo === 'function') {
        mod.getCompilationInfo = () => options.makeCompilationInfo(desc, mod);
      }
      shaderModulesCreated.push(mod);
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
      renderPipelinesCreated.push(pipe);
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
      return pipe;
    },
    createBindGroupLayout(desc) {
      return { kind: 'GPUBindGroupLayout', id: nextId++, desc };
    },
    createPipelineLayout(desc) {
      return { kind: 'GPUPipelineLayout', id: nextId++, desc };
    },
    createCommandEncoder() {
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
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
        finish() {
          return { kind: 'GPUCommandBuffer' };
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {},
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

  return { device, ctx, shaderModulesCreated, renderPipelinesCreated };
}

function createSyntheticRangePackServer(packBuffer, options = {}) {
  const packBytes = new Uint8Array(packBuffer);
  const reportedTotalBytes =
    options.reportedTotalBytes !== undefined ? options.reportedTotalBytes : 256 * 1024 * 1024;
  const deferDataRanges = Boolean(options.deferDataRanges);
  const dataStartOffset = options.dataStartOffset || 0;
  const deferredRanges = [];
  let dataFetchCount = 0;

  const fetchImpl = async (url, init = {}) => {
    const method = (init && init.method ? String(init.method) : 'GET').toUpperCase();
    const headers = init && init.headers ? init.headers : {};
    const rangeHeader = headers.Range || headers.range || null;

    if (method === 'HEAD') {
      return {
        ok: true,
        status: 200,
        headers: new Map([
          ['content-length', String(reportedTotalBytes)],
          ['accept-ranges', 'bytes'],
        ]),
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
      const isDataRange = dataStartOffset > 0 && start >= dataStartOffset;
      if (isDataRange) {
        dataFetchCount += 1;
        if (typeof options.onDataRange === 'function') {
          const override = await options.onDataRange({ url, start, end: clampedEnd, init });
          if (override !== undefined) {
            return override;
          }
        }
        if (deferDataRanges) {
          await new Promise((resolve, reject) => {
            deferredRanges.push({ start, end: clampedEnd, resolve, reject });
          });
        }
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

    throw new Error(`Unexpected non-Range GET for ${url}`);
  };

  return {
    fetchImpl,
    deferredRanges,
    get dataFetchCount() {
      return dataFetchCount;
    },
    resolveAllDeferred() {
      while (deferredRanges.length > 0) {
        deferredRanges.shift().resolve();
      }
    },
  };
}

// ---------------------------------------------------------------------------
// 1. Ranged-pack TOC hits must NEVER pollute missingShaders / builtinHits
// ---------------------------------------------------------------------------

test('ranged-pack TOC hits do NOT record as missingShaders, overflowMissingCount, or builtinHits, and credit packHits once compiled', async () => {
  const HASH_UI_VS = '0x006d0f6d4b911457';
  const HASH_SPLASH_FS = '0x000127b60500fe72';
  const UI_VS_WGSL = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }\n';
  const SPLASH_FS_WGSL = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.8, 0.5, 0.2, 1.0); }\n';

  const entries = [
    { hash: HASH_UI_VS, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: UI_VS_WGSL },
    { hash: HASH_SPLASH_FS, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: SPLASH_FS_WGSL },
  ];
  const extraHashes = [];
  for (let i = 0; i < 120; i += 1) {
    const wgsl = `@compute @workgroup_size(${(i % 8) + 1}) fn cs_${i}() {}\n`;
    const h = wgpkFormatHashHex(wgpkFnv1a64(wgsl));
    extraHashes.push(h);
    entries.push({ hash: h, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl });
  }

  const packBuf = buildWgslPackBuffer(entries);
  const dataStart = 16 + entries.length * 20;
  const net = createSyntheticRangePackServer(packBuf, {
    dataStartOffset: dataStart,
    deferDataRanges: true,
  });

  const mock = createMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: true,
    lazyRangedShaderCompile: true,
  });

  await proc.loadShaderPackUrl('/shaders/splash.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
  });
  assert.equal(proc.rangedPackActive, true);

  // Also verify direct ShaderFallbackRouter behavior on ranged TOC hits.
  assert.equal(proc.shaderFallbackRouter.hasRealShader(HASH_UI_VS, 'vertex'), true);
  assert.equal(proc.shaderFallbackRouter.hasRealShader(HASH_SPLASH_FS, 'fragment'), true);

  // Submit 122 ranged TOC shaders + 2 genuinely missing shaders.
  const GENUINE_MISS_VS = '0xdeadbeef00000001';
  const GENUINE_MISS_FS = '0xdeadbeef00000002';
  const preloadBuilder = new BinaryCommandBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, HASH_UI_VS)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, HASH_SPLASH_FS)
    .createShaderModule(500, SHADER_STAGE.VERTEX, GENUINE_MISS_VS)
    .createShaderModule(501, SHADER_STAGE.FRAGMENT, GENUINE_MISS_FS);
  for (let i = 0; i < extraHashes.length; i += 1) {
    preloadBuilder.createShaderModule(10 + i, SHADER_STAGE.COMPUTE, extraHashes[i]);
  }
  proc.executeCommandBuffer(preloadBuilder.finish());

  // Lazy-deferred ranged shaders must NOT be counted as asyncShaderPendingCount, builtinHits, or missingShaders!
  assert.equal(proc.asyncShaderPendingCount, 0, 'Lazy-deferred shaders are not in-flight');
  assert.equal(proc.deferredShaderCount, 122, 'All 122 unused ranged TOC shaders are tracked as deferredShaderCount');

  const fbStatsPre = proc.getShaderFallbackStats();
  assert.equal(fbStatsPre.builtinHits, 2, 'Only genuinely missing hashes increment builtinHits');
  assert.equal(fbStatsPre.uniqueMissing, 2, 'Only genuinely missing hashes are recorded in missing table');
  assert.equal(fbStatsPre.overflowMissingCount, 0);

  const missingHashesPre = proc.getMissingShaders().map((r) => r.hash);
  assert.deepEqual(missingHashesPre.sort(), [GENUINE_MISS_VS, GENUINE_MISS_FS]);
  assert.equal(missingHashesPre.includes(HASH_UI_VS), false, 'TOC shader 0x006d0f6d4b911457 must not appear in missingShaders');
  assert.equal(missingHashesPre.includes(HASH_SPLASH_FS), false, 'TOC shader 0x000127b60500fe72 must not appear in missingShaders');

  // Destroying a lazy-deferred shader decrements deferredShaderCount cleanly without fetching.
  proc.executeCommandBuffer(new BinaryCommandBuilder().destroyResource(10).finish());
  assert.equal(proc.deferredShaderCount, 121);
  assert.equal(proc.asyncShaderPendingCount, 0);

  // Create a pipeline referencing HASH_UI_VS (id=1) + HASH_SPLASH_FS (id=2) -> triggers ONLY those 2 fetches.
  proc.executeCommandBuffer(
    new BinaryCommandBuilder().createRenderPipeline(100, 1, 2, 0).finish()
  );
  assert.equal(proc.deferredShaderCount, 119);
  assert.equal(proc.asyncShaderPendingCount, 2);
  assert.equal(proc.rangedShaderFetches, 2);

  // Resolve deferred HTTP 206 fetches -> shaders compile, pipeline rebuilds, packHits credits +2!
  net.resolveAllDeferred();
  await proc.flushPendingRangedShaders();

  assert.equal(proc.asyncShaderPendingCount, 0);
  assert.equal(proc.rangedShaderReady, 2);
  assert.equal(proc.rangedShaderErrors, 0);
  assert.equal(proc.deferredShaderCount, 119);
  assert.equal(proc.pipelines.has(100), true, 'Pending render pipeline must build once ranged shaders settle');
  assert.equal(proc.pipelines.get(100).desc.vertex.module.code, UI_VS_WGSL);
  assert.equal(proc.pipelines.get(100).desc.fragment.module.code, SPLASH_FS_WGSL);

  const fbStatsPost = proc.getShaderFallbackStats();
  assert.equal(fbStatsPost.packHits, 2, 'Compiled ranged shaders must credit packHits');
  assert.equal(fbStatsPost.builtinHits, 2);
  assert.equal(fbStatsPost.hitRatePct, 50);

  const snap = buildGpuTelemetrySnapshot(proc, { timestamp: 9001 });
  assert.equal(snap.fallbackStats.packHits, 2);
  assert.equal(snap.fallbackStats.builtinHits, 2);
  assert.equal(snap.shaderPackStats.asyncShaderPendingCount, 0);
  assert.equal(snap.shaderPackStats.deferredShaderCount, 119);
  assert.equal(snap.shaderPackStats.rangedShaderReady, 2);

  // Execute draw with the rebuilt pipeline -> realPackDrawCount increments cleanly with 0 errors.
  proc.executeCommandBuffer(
    new BinaryCommandBuilder()
      .beginFrame()
      .setPipeline(100)
      .draw(3, 1, 0, 0)
      .present()
      .finish()
  );
  assert.equal(proc.getStats().realPackDrawCount, 1);
  assert.equal(proc.validationErrorCount, 0);

  // Direct router test with a minimal custom ranged loader ({ has, getWgslAsync }).
  const customRouter = new ShaderFallbackRouter({
    packLoader: {
      has: (k) => k === HASH_UI_VS || k === BigInt(HASH_UI_VS),
      async getWgslAsync() {
        return UI_VS_WGSL;
      },
    },
  });
  const pendingRes = customRouter.resolve(HASH_UI_VS, 'vertex');
  assert.equal(pendingRes.source, 'builtin');
  assert.equal(pendingRes.rangedPending, true);
  assert.equal(customRouter.getStats().builtinHits, 0);
  assert.equal(customRouter.getStats().uniqueMissing, 0);
  assert.equal(customRouter.getMissingReport().length, 0);

  proc.destroy();
});

// ---------------------------------------------------------------------------
// 2. 520+ queued shaders drain cleanly under oversized, throws, hung getCompilationInfo,
//    rejecting getCompilationInfo, and superseded generations (zero stalls / zero leaks)
// ---------------------------------------------------------------------------

test('_rangedShaderCompileQueue drains 520 queued shaders cleanly despite hung/failing getCompilationInfo, oversized entries, throws, and superseded generations', async () => {
  const TOTAL = 520;
  const wgslByHash = new Map();
  const hashes = [];

  for (let i = 0; i < TOTAL; i += 1) {
    let wgsl;
    if (i % 50 === 0) {
      // Oversized (> 256 KiB cap when cap=4096 in test)
      wgsl = `@fragment fn fs_huge_${i}() -> @location(0) vec4<f32> {\n// ${'z'.repeat(6000)}\nreturn vec4<f32>(1.0);\n}`;
    } else if (i % 37 === 0) {
      // Triggers synchronous createShaderModule throw
      wgsl = `@fragment fn fs_throw_${i}() -> @location(0) vec4<f32> { /* THROW_SYNC */ return vec4<f32>(1.0); }`;
    } else if (i % 41 === 0) {
      // Triggers a NEVER-RESOLVING getCompilationInfo() (Dawn stall!)
      wgsl = `@vertex fn vs_hang_${i}() -> @builtin(position) vec4<f32> { /* HANG_INFO */ return vec4<f32>(0.0); }`;
    } else if (i % 43 === 0) {
      // Triggers a rejecting getCompilationInfo()
      wgsl = `@vertex fn vs_reject_${i}() -> @builtin(position) vec4<f32> { /* REJECT_INFO */ return vec4<f32>(0.0); }`;
    } else {
      wgsl =
        i % 2 === 0
          ? `@vertex fn vs_ok_${i}() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }`
          : `@compute @workgroup_size(1) fn cs_ok_${i}() { /* ${i} */ }`;
    }
    const hash = wgpkFormatHashHex(wgpkFnv1a64(wgsl));
    hashes.push(hash);
    wgslByHash.set(hash, { wgsl, entryPoint: null });
  }

  let hungInfoCount = 0;
  const mock = createMockWebGpu({
    onCreateShaderModule(desc, nextId) {
      if (desc.code.includes('THROW_SYNC')) {
        throw new Error('simulated Dawn sync createShaderModule failure');
      }
      const mod = { kind: 'GPUShaderModule', id: nextId, code: desc.code };
      if (desc.code.includes('HANG_INFO')) {
        hungInfoCount += 1;
        mod.getCompilationInfo = () => new Promise(() => {}); // Never resolves!
      } else if (desc.code.includes('REJECT_INFO')) {
        mod.getCompilationInfo = () => Promise.reject(new Error('simulated getCompilationInfo rejection'));
      } else {
        mod.getCompilationInfo = () => Promise.resolve({ messages: [] });
      }
      return mod;
    },
  });

  const proc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: true,
    lazyRangedShaderCompile: false,
    maxWgslCompileBytes: 4096,
    shaderCompileInfoTimeoutMs: 15,
    shaderCompileSliceBudgetMs: 8,
  });

  // Wire ranged pack cache directly (no getShaderMetadata so oversized shaders hit post-fetch quarantine too).
  proc.wgslPackCache = {
    mode: 'ranged',
    has: (h) => wgslByHash.has(h),
    async getWgslAsync(h) {
      return wgslByHash.get(h) || null;
    },
  };

  // Enqueue all 520 ranged shaders in one burst!
  const builder = new BinaryCommandBuilder();
  for (let i = 0; i < TOTAL; i += 1) {
    const stage =
      i % 50 === 0 || i % 37 === 0
        ? SHADER_STAGE.FRAGMENT
        : i % 41 === 0 || i % 43 === 0 || i % 2 === 0
          ? SHADER_STAGE.VERTEX
          : SHADER_STAGE.COMPUTE;
    builder.createShaderModule(i + 1, stage, hashes[i]);
  }
  // Supersede shader IDs 5, 15, 25 with inline shaders and destroy shader ID 35 while queued!
  builder.createShaderModule(
    5,
    SHADER_STAGE.VERTEX,
    '@vertex fn vs_superseded() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }'
  );
  builder.destroyResource(35);

  proc.executeCommandBuffer(builder.finish());

  // Flush must settle completely without hanging!
  await proc.flushPendingRangedShaders();

  const stats = proc.getStats();
  assert.equal(proc.asyncShaderPendingCount, 0, 'asyncShaderPendingCount must drain to 0');
  assert.equal(stats.asyncShaderPendingCount, 0);
  assert.equal(stats.activeRangedShaderCompiles, 0, 'Compile slot must never leak');
  assert.equal(stats.rangedShaderQueuedCount, 0, 'Compile queue must drain completely');
  assert.ok(hungInfoCount > 0, 'Test must exercise hung getCompilationInfo() shaders');
  assert.equal(
    stats.rangedShaderCompileTimeouts,
    hungInfoCount,
    'Every hung getCompilationInfo() must release its slot via timeout'
  );
  assert.ok(stats.oversizedWgslFallbacks > 0, 'Oversized shaders must be quarantined cleanly');
  assert.ok(stats.rangedShaderReady > 450, `Expected >450 ready shaders, got ${stats.rangedShaderReady}`);

  // Verify a hung-info shader still finalized its valid GPUShaderModule handle!
  const hungShaderEntry = proc.shaderModules.get(42); // i=41 => id=42 (41 % 41 === 0)
  assert.ok(hungShaderEntry && hungShaderEntry.module, 'Hung getCompilationInfo shader still commits its GPUShaderModule');
  assert.equal(hungShaderEntry.source, 'pack');

  // Verify superseded shader 5 kept its newer inline module and destroyed shader 35 stayed absent.
  assert.equal(proc.shaderModules.get(5).entryPoint, 'vs_superseded');
  assert.equal(proc.shaderModules.has(35), false);

  proc.destroy();
});

// ---------------------------------------------------------------------------
// 3. Saturated/stalled AsyncPipelineCache cannot permanently starve ranged shaders
// ---------------------------------------------------------------------------

test('stalled AsyncPipelineCache at capacity cannot permanently starve _rangedShaderCompileQueue', async () => {
  const mock = createMockWebGpu();
  // Simulate permanently stalled async pipeline compiles (never resolve).
  mock.device.createRenderPipelineAsync = () => new Promise(() => {});

  const proc = new WgpuCommandProcessor(mock.device, mock.ctx, {
    format: 'bgra8unorm',
    enableShaderFallback: false,
    lazyRangedShaderCompile: false,
    maxPipelineCapacityYieldMs: 25,
  });
  installRenderStatePackets(proc);

  const vsCode = '@vertex fn vs_s() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const fsCode = '@fragment fn fs_s() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';
  const rangedCode = '@vertex fn vs_unstarved() -> @builtin(position) vec4<f32> { return vec4<f32>(0.5); }';
  const rangedHash = wgpkFormatHashHex(wgpkFnv1a64(rangedCode));

  // Saturate AsyncPipelineCache (2/2 in-flight forever).
  const p2Buf = new ArrayBuffer(16 + 376);
  const p2v = new DataView(p2Buf);
  const makeP2 = (pid) => {
    const out = new ArrayBuffer(8 + 16 + 376);
    const v = new DataView(out);
    v.setUint16(0, OPCODES.CREATE_RENDER_PIPELINE2, true);
    v.setUint32(4, 16 + 376, true);
    v.setUint32(8, pid, true);
    v.setUint32(12, 1, true);
    v.setUint32(16, 2, true);
    v.setUint32(24 + 20, 1, true);
    v.setUint8(24 + 24, 27);
    v.setUint8(24 + 26, 0x0f);
    v.setUint32(24 + 164, 1, true);
    v.setUint32(24 + 168, 0xffffffff, true);
    return new Uint8Array(out);
  };
  proc.executeCommandBuffer(
    new BinaryCommandBuilder()
      .createShaderModule(1, SHADER_STAGE.VERTEX, vsCode)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, fsCode)
      .finish()
  );
  proc.executeCommandBuffer(makeP2(101).buffer);
  proc.executeCommandBuffer(makeP2(102).buffer);
  assert.equal(proc._isAsyncPipelineCacheAtCapacity(), true);

  proc.wgslPackCache = {
    mode: 'ranged',
    has: (h) => h === rangedHash,
    async getWgslAsync() {
      return { wgsl: rangedCode, entryPoint: 'vs_unstarved' };
    },
  };

  proc.executeCommandBuffer(
    new BinaryCommandBuilder().createShaderModule(3, SHADER_STAGE.VERTEX, rangedHash).finish()
  );

  // Must NOT hang forever even though AsyncPipelineCache stays at 2/2 capacity!
  await proc.flushPendingRangedShaders();
  assert.equal(proc.asyncShaderPendingCount, 0);
  assert.equal(proc.shaderModules.get(3).entryPoint, 'vs_unstarved');
  assert.equal(proc.shaderModules.get(3).source, 'pack');

  proc.destroy();
});

// ---------------------------------------------------------------------------
// 4. wgsl_pack_loader ranged HTTP 206 fetch timeout releases slot and prevents queue deadlock
// ---------------------------------------------------------------------------

test('wgsl_pack_loader rangeFetchTimeoutMs aborts hung HTTP 206 fetches, retries cleanly, and never leaks fetch slots', async () => {
  const wgslA = '@vertex fn vs_a() -> @builtin(position) vec4<f32> { return vec4<f32>(0.1); }\n';
  const wgslB = '@fragment fn fs_b() -> @location(0) vec4<f32> { return vec4<f32>(0.2); }\n';
  const wgslC = '@compute @workgroup_size(1) fn cs_c() {}\n';
  const hashA = wgpkFnv1a64(wgslA);
  const hashB = wgpkFnv1a64(wgslB);
  const hashC = wgpkFnv1a64(wgslC);

  const packBuf = buildWgslPackBuffer([
    { hash: hashA, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: wgslA },
    { hash: hashB, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: wgslB },
    { hash: hashC, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl: wgslC },
  ]);
  const dataStart = 16 + 3 * 20;

  // Map offsets to identify which shader is requested.
  const attemptsByStart = new Map();
  let hashBStart = -1;

  const net = createSyntheticRangePackServer(packBuf, {
    dataStartOffset: dataStart,
    async onDataRange({ start }) {
      const n = (attemptsByStart.get(start) || 0) + 1;
      attemptsByStart.set(start, n);
      if (start === dataStart && n === 1) {
        // First attempt for shader A hangs forever!
        return new Promise(() => {});
      }
      if (start === hashBStart) {
        // All attempts for shader B hang forever!
        return new Promise(() => {});
      }
      return undefined;
    },
  });

  const pack = await loadIndexFromUrl('/shaders/timeout-test.wgsl.pack', {
    fetch: net.fetchImpl,
    initialRangeBytes: dataStart,
    maxConcurrentRangeFetches: 1,
    maxRangeFetchRetries: 1,
    rangeRetryDelayMs: 5,
    rangeFetchTimeoutMs: 25,
  });

  hashBStart = pack.entries.get(hashB).offset;

  // Launch A (hangs once then retries & succeeds), B (hangs all attempts -> rejects), and C (must not deadlock!).
  const [resA, resB, resC] = await Promise.allSettled([
    pack.getWgslAsync(hashA),
    pack.getWgslAsync(hashB),
    pack.getWgslAsync(hashC),
  ]);

  assert.equal(resA.status, 'fulfilled');
  assert.equal(resA.value, wgslA);

  assert.equal(resB.status, 'rejected');
  assert.equal(resB.reason.code, 'WGPK_RANGE_FETCH_TIMEOUT');

  assert.equal(resC.status, 'fulfilled');
  assert.equal(resC.value, wgslC);

  const packStats = pack.getStats();
  assert.equal(packStats.activeRangeFetches, 0, 'All range fetch slots must be released after timeouts');
  assert.ok(packStats.rangeFetchRetries >= 2, 'Timed-out range fetches must retry cleanly');
});
