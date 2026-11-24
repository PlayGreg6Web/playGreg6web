import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  SHADER_STAGE as WGPK_SHADER_STAGE,
  buildWgslPackBuffer,
  WgslPackCache,
  fnv1a64,
  formatHashHex,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const wgpuMod = require('../web/workers/wgpu_worker.js');
const fallbackMod = require('../web/workers/shader_fallback_router.js');
const reflectionMod = require('../web/workers/reflection_loader.js');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
} = wgpuMod;

const { ShaderFallbackRouter, BUILTIN_FALLBACK_WGSL } = fallbackMod;
const {
  SHADER_STAGE_BITS,
  computeBindGroupLayoutHash,
  computePipelineLayoutHash,
} = reflectionMod;

const PACK_VS_WGSL = [
  '@vertex',
  'fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {',
  '    return vec4<f32>(pos, 1.0);',
  '}',
  '',
].join('\n');

const PACK_FS_WGSL = [
  '@fragment',
  'fn fs_main() -> @location(0) vec4<f32> {',
  '    return vec4<f32>(0.15, 0.75, 0.95, 1.0);',
  '}',
  '',
].join('\n');

const PACK_CS_WGSL = [
  '@compute @workgroup_size(8, 1, 1)',
  'fn cs_main() {}',
  '',
].join('\n');

const S9_VS_WGSL = [
  '@vertex',
  'fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {',
  '    return vec4<f32>(0.0, 0.0, 0.0, 1.0);',
  '}',
  '',
].join('\n');

const S9_FS_WGSL = [
  '@fragment',
  'fn fs_main() -> @location(0) vec4<f32> {',
  '    return vec4<f32>(0.9, 0.2, 0.6, 1.0);',
  '}',
  '',
].join('\n');

const S9_CS_WGSL = [
  '@compute @workgroup_size(16, 1, 1)',
  'fn cs_main() {}',
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
  const calls = [];
  let nextHandle = 1;

  const ctx = {
    configure(desc) {
      calls.push({ op: 'ctx.configure', desc });
    },
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
      if (typeof desc.code !== 'string' || desc.code.trim().length === 0) {
        throw new Error('WGSL shader module code must be non-empty');
      }
      const mod = { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
      calls.push({ op: 'createShaderModule', desc: { ...desc }, result: mod });
      return mod;
    },
    createBindGroupLayout(desc) {
      const bgl = { kind: 'GPUBindGroupLayout', handle: nextHandle++, desc };
      calls.push({ op: 'createBindGroupLayout', desc, result: bgl });
      return bgl;
    },
    createPipelineLayout(desc) {
      const pl = { kind: 'GPUPipelineLayout', handle: nextHandle++, desc };
      calls.push({ op: 'createPipelineLayout', desc, result: pl });
      return pl;
    },
    createRenderPipeline(desc) {
      const pipe = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(idx) {
          return { kind: 'GPUBindGroupLayout', idx };
        },
      };
      calls.push({ op: 'createRenderPipeline', desc, result: pipe });
      return pipe;
    },
    createComputePipeline(desc) {
      const pipe = {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(idx) {
          return { kind: 'GPUBindGroupLayout', idx };
        },
      };
      calls.push({ op: 'createComputePipeline', desc, result: pipe });
      return pipe;
    },
    createCommandEncoder() {
      const cmdBuf = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass(desc) {
          calls.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline(p) {
              calls.push({ op: 'renderPass.setPipeline', pipeline: p });
            },
            draw(vc, ic, fv, fi) {
              calls.push({ op: 'renderPass.draw', vc, ic, fv, fi });
            },
            end() {
              calls.push({ op: 'renderPass.end' });
            },
          };
        },
        beginComputePass(desc) {
          calls.push({ op: 'beginComputePass', desc });
          return {
            setPipeline(p) {
              calls.push({ op: 'computePass.setPipeline', pipeline: p });
            },
            dispatchWorkgroups(x, y, z) {
              calls.push({ op: 'computePass.dispatchWorkgroups', x, y, z });
            },
            end() {
              calls.push({ op: 'computePass.end' });
            },
          };
        },
        finish() {
          calls.push({ op: 'encoder.finish', result: cmdBuf });
          return cmdBuf;
        },
      };
    },
    queue: {
      submit(cmdBuffers) {
        calls.push({ op: 'queue.submit', count: cmdBuffers.length });
      },
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
  };

  return { device, ctx, calls };
}

function buildReflectionManifestForHash(vsHash, fsHash) {
  const vsEntries = [
    {
      binding: 0,
      visibility: SHADER_STAGE_BITS.VERTEX,
      buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 64 },
    },
  ];
  const fsEntries = [
    {
      binding: 0,
      visibility: SHADER_STAGE_BITS.FRAGMENT,
      buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 64 },
    },
  ];
  const vsBglId = computeBindGroupLayoutHash({ entries: vsEntries });
  const fsBglId = computeBindGroupLayoutHash({ entries: fsEntries });
  const vsPlId = computePipelineLayoutHash([vsBglId]);
  const fsPlId = computePipelineLayoutHash([fsBglId]);

  return {
    schema_version: 1,
    register_convention: { source: 'dxc-spirv', shifts: { b: 0, t: 0, u: 0, s: 0 } },
    bind_group_layouts: {
      [vsBglId]: { entries: vsEntries },
      [fsBglId]: { entries: fsEntries },
    },
    pipeline_layouts: {
      [vsPlId]: { bindGroupLayouts: [vsBglId] },
      [fsPlId]: { bindGroupLayouts: [fsBglId] },
    },
    shaders: {
      [vsHash]: {
        stage: 'vertex',
        entry_point: 'vs_main',
        name: 'preloaded_vs',
        pipeline_layout: vsPlId,
        bind_group_layouts: [vsBglId],
        layout_complete: true,
        vertex_inputs: [{ location: 0, type: 'vec3<f32>', format: 'float32x3' }],
        bindings: [{ group: 0, binding: 0, name: 'uCam', resource: 'buffer', in_layout: true }],
      },
      [fsHash]: {
        stage: 'fragment',
        entry_point: 'fs_main',
        name: 'preloaded_fs',
        pipeline_layout: fsPlId,
        bind_group_layouts: [fsBglId],
        layout_complete: true,
        fragment_outputs: [{ location: 0, type: 'vec4<f32>' }],
        bindings: [{ group: 0, binding: 0, name: 'uCam', resource: 'buffer', in_layout: true }],
      },
    },
  };
}

// ---------------------------------------------------------------------------
// 1. Empty WGSL resolves cleanly to built-in stage fallbacks & records telemetry
// ---------------------------------------------------------------------------

test('G28 empty CreateShaderModule payloads resolve to valid built-in WGSL for VS, FS, and CS', () => {
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);

  assert.ok(processor.shaderFallbackRouter instanceof ShaderFallbackRouter);
  assert.ok(processor.wgslPackCache);

  const buf = new PacketBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, '')
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, '   \n\t  ')
    .createShaderModule(3, SHADER_STAGE.COMPUTE, '')
    .finish();

  processor.executeCommandBuffer(buf);

  const shaderCalls = mock.calls.filter((c) => c.op === 'createShaderModule');
  assert.equal(shaderCalls.length, 3);
  assert.equal(shaderCalls[0].desc.code, BUILTIN_FALLBACK_WGSL.vertex);
  assert.equal(shaderCalls[1].desc.code, BUILTIN_FALLBACK_WGSL.fragment);
  assert.equal(shaderCalls[2].desc.code, BUILTIN_FALLBACK_WGSL.compute);

  assert.equal(processor.shaderModules.get(1).source, 'builtin');
  assert.equal(processor.shaderModules.get(2).source, 'builtin');
  assert.equal(processor.shaderModules.get(3).source, 'builtin');

  const stats = processor.getShaderFallbackStats();
  assert.equal(stats.totalLookups, 3);
  assert.equal(stats.packHits, 0);
  assert.equal(stats.builtinHits, 3);
  assert.equal(stats.uniqueMissing, 3);

  const missing = processor.getMissingShaders();
  assert.equal(missing.length, 3);
  const stages = new Set(missing.map((m) => m.stage));
  assert.deepEqual(stages, new Set(['vertex', 'fragment', 'compute']));
});

// ---------------------------------------------------------------------------
// 2. Hash-only lookup misses resolve via fallback & record canonical missing hash
// ---------------------------------------------------------------------------

test('G28 hash-only CreateShaderModule misses resolve cleanly for 0x, hash:, wgpk:, and bare hex', () => {
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);

  const h1 = '0xdeadbeefcafebabe';
  const h2 = '0x000000001234abcd';
  const h3 = '0xfeedface01020304';
  const h4 = '0xabcdef0123456789';

  const buf = new PacketBuilder()
    .createShaderModule(10, SHADER_STAGE.VERTEX, h1)
    .createShaderModule(11, SHADER_STAGE.FRAGMENT, 'hash:0x1234ABCD')
    .createShaderModule(12, SHADER_STAGE.COMPUTE, 'wgpk:feedface01020304')
    .createShaderModule(13, SHADER_STAGE.VERTEX, 'abcdef0123456789')
    .finish();

  processor.executeCommandBuffer(buf);

  assert.equal(processor.shaderModules.get(10).code, BUILTIN_FALLBACK_WGSL.vertex);
  assert.equal(processor.shaderModules.get(11).code, BUILTIN_FALLBACK_WGSL.fragment);
  assert.equal(processor.shaderModules.get(12).code, BUILTIN_FALLBACK_WGSL.compute);
  assert.equal(processor.shaderModules.get(13).code, BUILTIN_FALLBACK_WGSL.vertex);

  const missing = processor.getMissingShaders();
  assert.equal(missing.length, 4);
  const byHash = new Map(missing.map((m) => [m.hash, m]));
  assert.equal(byHash.get(h1).stage, 'vertex');
  assert.equal(byHash.get(h2).stage, 'fragment');
  assert.equal(byHash.get(h3).stage, 'compute');
  assert.equal(byHash.get(h4).stage, 'vertex');
});

// ---------------------------------------------------------------------------
// 3. Pack hit precedence over fallback without recording missing telemetry
// ---------------------------------------------------------------------------

test('G28 loaded WGSL pack hits take precedence over fallbacks and do not record missing shaders', () => {
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);

  const vsHash = formatHashHex(fnv1a64(PACK_VS_WGSL));
  const fsHash = formatHashHex(fnv1a64(PACK_FS_WGSL));
  const csHash = formatHashHex(fnv1a64(PACK_CS_WGSL));

  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: PACK_FS_WGSL },
    { hash: csHash, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl: PACK_CS_WGSL },
  ]);

  processor.loadWgslPack(packBuf, 'boot.wgsl.pack');

  const stream = new PacketBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsHash)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, `hash:${fsHash}`)
    .createShaderModule(3, SHADER_STAGE.COMPUTE, `wgpk:${csHash}`)
    .finish();

  processor.executeCommandBuffer(stream);

  assert.equal(processor.shaderModules.get(1).code, PACK_VS_WGSL);
  assert.equal(processor.shaderModules.get(1).source, 'pack');
  assert.equal(processor.shaderModules.get(1).resolvedHash, vsHash);
  assert.equal(processor.shaderModules.get(2).code, PACK_FS_WGSL);
  assert.equal(processor.shaderModules.get(2).source, 'pack');
  assert.equal(processor.shaderModules.get(3).code, PACK_CS_WGSL);
  assert.equal(processor.shaderModules.get(3).source, 'pack');

  assert.deepEqual(processor.getMissingShaders(), []);
  const stats = processor.getShaderFallbackStats();
  assert.equal(stats.totalLookups, 3);
  assert.equal(stats.packHits, 3);
  assert.equal(stats.fallbackHits, 0);
  assert.equal(stats.builtinHits, 0);
  assert.equal(stats.hitRatePct, 100);
});

// ---------------------------------------------------------------------------
// 4. Injected ESM WgslPackCache compatibility
// ---------------------------------------------------------------------------

test('G28 injected ESM WgslPackCache resolves hits and routes misses through ShaderFallbackRouter', () => {
  const mock = createMockWebGpu();
  const packCache = new WgslPackCache();
  const vsHash = '0x1111222233334444';
  const missingHash = '0x9999888877776666';

  packCache.loadPackBuffer(
    buildWgslPackBuffer([{ hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL }]),
    'esm-pack'
  );

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    wgslPackCache: packCache,
  });

  const stream = new PacketBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsHash)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, missingHash)
    .finish();
  processor.executeCommandBuffer(stream);

  assert.equal(processor.shaderModules.get(1).code, PACK_VS_WGSL);
  assert.equal(processor.shaderModules.get(1).source, 'pack');
  assert.equal(processor.shaderModules.get(2).code, BUILTIN_FALLBACK_WGSL.fragment);
  assert.equal(processor.shaderModules.get(2).source, 'builtin');

  const stats = processor.getShaderFallbackStats();
  assert.equal(stats.packHits, 1);
  assert.equal(stats.builtinHits, 1);
  assert.equal(stats.hitRatePct, 50);
  assert.equal(processor.getMissingShaders().length, 1);
  assert.equal(processor.getMissingShaders()[0].hash, missingHash);
});

// ---------------------------------------------------------------------------
// 5. Stage-matched S9 fallbackShaderHashes from pack
// ---------------------------------------------------------------------------

test('G28 fallbackShaderHashes resolves misses to stage-matched S9 fallback WGSL in pack', () => {
  const mock = createMockWebGpu();
  const fallbackHashes = {
    vertex: '0x0000000000000101',
    fragment: '0x0000000000000202',
    compute: '0x0000000000000303',
  };

  const packBuf = buildWgslPackBuffer([
    { hash: fallbackHashes.vertex, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: S9_VS_WGSL },
    { hash: fallbackHashes.fragment, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: S9_FS_WGSL },
    { hash: fallbackHashes.compute, stage: WGPK_SHADER_STAGE.COMPUTE, wgsl: S9_CS_WGSL },
  ]);

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    fallbackShaderHashes: fallbackHashes,
    wgslPackBuffer: packBuf,
  });

  const stream = new PacketBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, '0xaaaaaaaaaaaaaaaa')
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, '0xbbbbbbbbbbbbbbbb')
    .createShaderModule(3, SHADER_STAGE.COMPUTE, '0xcccccccccccccccc')
    .finish();
  processor.executeCommandBuffer(stream);

  assert.equal(processor.shaderModules.get(1).code, S9_VS_WGSL);
  assert.equal(processor.shaderModules.get(1).source, 'fallback');
  assert.equal(processor.shaderModules.get(1).resolvedHash, fallbackHashes.vertex);
  assert.equal(processor.shaderModules.get(2).code, S9_FS_WGSL);
  assert.equal(processor.shaderModules.get(2).source, 'fallback');
  assert.equal(processor.shaderModules.get(3).code, S9_CS_WGSL);
  assert.equal(processor.shaderModules.get(3).source, 'fallback');

  const stats = processor.getShaderFallbackStats();
  assert.equal(stats.fallbackHits, 3);
  assert.equal(stats.builtinHits, 0);
  assert.equal(stats.uniqueMissing, 3);
});

// ---------------------------------------------------------------------------
// 6. Injected custom ShaderFallbackRouter is used without overwriting packLoader
// ---------------------------------------------------------------------------

test('G28 injected shaderFallbackRouter is preserved and exposed via telemetry accessors', () => {
  const mock = createMockWebGpu();
  const customHash = '0x5555555555555555';
  const customPack = new Map([[customHash, PACK_FS_WGSL]]);
  const router = new ShaderFallbackRouter({ packLoader: customPack });

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    shaderFallbackRouter: router,
  });
  assert.equal(processor.shaderFallbackRouter, router);

  const stream = new PacketBuilder()
    .createShaderModule(1, SHADER_STAGE.FRAGMENT, customHash)
    .createShaderModule(2, SHADER_STAGE.VERTEX, '0x6666666666666666')
    .finish();
  processor.executeCommandBuffer(stream);

  assert.equal(processor.shaderModules.get(1).code, PACK_FS_WGSL);
  assert.equal(processor.shaderModules.get(1).source, 'pack');
  assert.equal(processor.shaderModules.get(2).code, BUILTIN_FALLBACK_WGSL.vertex);
  assert.equal(processor.shaderModules.get(2).source, 'builtin');
  assert.deepEqual(processor.getShaderFallbackStats(), router.getStats());
  assert.deepEqual(processor.getMissingShaders(), router.getMissingReport());
});

// ---------------------------------------------------------------------------
// 7. enableShaderFallback: false disables router cleanly
// ---------------------------------------------------------------------------

test('G28 enableShaderFallback: false disables ShaderFallbackRouter and returns safe empty telemetry', () => {
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    enableShaderFallback: false,
  });

  assert.equal(processor.shaderFallbackRouter, null);
  assert.deepEqual(processor.getMissingShaders(), []);
  assert.equal(processor.getShaderFallbackStats(), null);
  assert.equal(processor.getStats().shaderFallback, null);

  // Valid inline WGSL still compiles cleanly.
  const okStream = new PacketBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, PACK_VS_WGSL)
    .finish();
  processor.executeCommandBuffer(okStream);
  assert.equal(processor.shaderModules.get(1).code, PACK_VS_WGSL);

  // Empty WGSL reaches device.createShaderModule directly (throws in our strict mock).
  const emptyStream = new PacketBuilder()
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, '')
    .finish();
  assert.throws(() => processor.executeCommandBuffer(emptyStream), /must be non-empty/);
});

// ---------------------------------------------------------------------------
// 8. Inline WGSL bypasses fallback router with zero telemetry pollution
// ---------------------------------------------------------------------------

test('G28 normal inline WGSL bypasses ShaderFallbackRouter with zero lookup overhead', () => {
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);

  const stream = new PacketBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, PACK_VS_WGSL)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, PACK_FS_WGSL)
    .createShaderModule(3, SHADER_STAGE.COMPUTE, PACK_CS_WGSL)
    .finish();
  processor.executeCommandBuffer(stream);

  assert.equal(processor.shaderModules.get(1).source, 'inline');
  assert.equal(processor.shaderModules.get(2).source, 'inline');
  assert.equal(processor.shaderModules.get(3).source, 'inline');
  assert.equal(processor.getShaderFallbackStats().totalLookups, 0);
  assert.deepEqual(processor.getMissingShaders(), []);
});

// ---------------------------------------------------------------------------
// 9. Missing telemetry sorting, deduplication, limit option, and getStats()
// ---------------------------------------------------------------------------

test('G28 getMissingShaders({ limit }) sorts by hitCount desc and getStats() includes shaderFallback', () => {
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);

  const hot = '0x00000000000000aa';
  const warm = '0x00000000000000bb';
  const cold = '0x00000000000000cc';

  const stream = new PacketBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, cold)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, warm)
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, warm)
    .createShaderModule(4, SHADER_STAGE.VERTEX, hot)
    .createShaderModule(5, SHADER_STAGE.VERTEX, hot)
    .createShaderModule(6, SHADER_STAGE.VERTEX, hot)
    .finish();
  processor.executeCommandBuffer(stream);

  const all = processor.getMissingShaders();
  assert.equal(all.length, 3);
  assert.equal(all[0].hash, hot);
  assert.equal(all[0].hitCount, 3);
  assert.equal(all[1].hash, warm);
  assert.equal(all[1].hitCount, 2);
  assert.equal(all[2].hash, cold);
  assert.equal(all[2].hitCount, 1);

  const top2 = processor.getMissingShaders({ limit: 2 });
  assert.equal(top2.length, 2);
  assert.equal(top2[0].hash, hot);
  assert.equal(top2[1].hash, warm);

  const aggStats = processor.getStats();
  assert.deepEqual(aggStats.shaderFallback, processor.getShaderFallbackStats());
  assert.equal(aggStats.shaderFallback.totalLookups, 6);
  assert.equal(aggStats.shaderFallback.uniqueMissing, 3);
});

// ---------------------------------------------------------------------------
// 10. Full render + compute frame succeeds using fallback shaders
// ---------------------------------------------------------------------------

test('G28 render and compute pipelines + draw/dispatch succeed with missing fallback shaders', () => {
  const mock = createMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);

  const stream = new PacketBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, '')
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, '0xdead000000000001')
    .createShaderModule(3, SHADER_STAGE.COMPUTE, '0xdead000000000002')
    .createRenderPipeline(10, 1, 2, 0)
    .createComputePipeline(11, 3)
    .setComputePipeline(11)
    .dispatchCompute(1, 1, 1)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  const res = processor.executeCommandBuffer(stream);
  assert.deepEqual(res, { frameCount: 1, presents: 1 });
  assert.equal(processor.pipelines.has(10), true);
  assert.equal(processor.computePipelines.has(11), true);
  assert.equal(processor.getShaderFallbackStats().builtinHits, 3);
});

// ---------------------------------------------------------------------------
// 11. createWgpuWorkerHandler init preloads shaderPackUrl + reflectionManifestUrl
// ---------------------------------------------------------------------------

test('G28 createWgpuWorkerHandler init preloads shaderPackUrl and reflectionManifestUrl before ready', async () => {
  const mock = createMockWebGpu();
  const vsHash = formatHashHex(fnv1a64(PACK_VS_WGSL));
  const fsHash = formatHashHex(fnv1a64(PACK_FS_WGSL));

  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: PACK_FS_WGSL },
  ]);
  const manifest = buildReflectionManifestForHash(vsHash, fsHash);

  const fetchedUrls = [];
  const mockFetch = async (url) => {
    fetchedUrls.push(url);
    if (url === '/shaders/fallback.wgsl.pack') {
      return {
        ok: true,
        status: 200,
        async arrayBuffer() {
          return packBuf.slice(0);
        },
      };
    }
    if (url === '/shaders/reflection-manifest.json') {
      return {
        ok: true,
        status: 200,
        async text() {
          return JSON.stringify(manifest);
        },
      };
    }
    return { ok: false, status: 404 };
  };

  // Temporarily stub navigator.gpu so defaultInitGpu executes the full browser init path.
  const prevNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          return {
            info: { vendor: 'mock', architecture: 'g28' },
            async requestDevice() {
              return mock.device;
            },
          };
        },
        getPreferredCanvasFormat() {
          return 'bgra8unorm';
        },
      },
    },
  });

  const posted = [];
  let readySawLoadedPack = false;
  let readySawLoadedReflection = false;
  let handler;
  try {
    handler = createWgpuWorkerHandler({
      fetch: mockFetch,
      postMessage: (m) => {
        if (m && m.ready && handler && handler.processor) {
          readySawLoadedPack = handler.processor.wgslPackCache.getStats().shaderCount === 2;
          readySawLoadedReflection = handler.processor.getStats().reflectionShaderCount === 2;
        }
        posted.push(m);
      },
    });

    const fakeCanvas = {
      width: 0,
      height: 0,
      getContext(kind) {
        assert.equal(kind, 'webgpu');
        return mock.ctx;
      },
    };

    await handler.handleMessage({
      init: true,
      canvas: fakeCanvas,
      width: 1280,
      height: 720,
      shaderPackUrl: '/shaders/fallback.wgsl.pack',
      reflectionManifestUrl: '/shaders/reflection-manifest.json',
    });
  } finally {
    if (prevNavigator) {
      Object.defineProperty(globalThis, 'navigator', prevNavigator);
    } else {
      delete globalThis.navigator;
    }
  }

  assert.deepEqual(fetchedUrls.sort(), [
    '/shaders/fallback.wgsl.pack',
    '/shaders/reflection-manifest.json',
  ]);
  assert.equal(readySawLoadedPack, true, 'shaderPackUrl must be loaded before ready message');
  assert.equal(
    readySawLoadedReflection,
    true,
    'reflectionManifestUrl must be loaded before ready message'
  );
  assert.ok(posted.some((m) => m.ready === true));

  // Execute hash-only CreateShaderModule + CreateRenderPipeline and verify both pack + reflection work!
  const stream = new PacketBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsHash)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, fsHash)
    .createRenderPipeline(3, 1, 2, 12)
    .setPipeline(3)
    .draw(3, 1, 0, 0)
    .present()
    .finish();
  await handler.handleMessage({ cmdBuffer: stream });

  const proc = handler.processor;
  assert.equal(proc.shaderModules.get(1).source, 'pack');
  assert.equal(proc.shaderModules.get(2).source, 'pack');
  assert.equal(proc.getShaderFallbackStats().packHits, 2);
  assert.deepEqual(proc.getMissingShaders(), []);

  const rpCall = mock.calls.find((c) => c.op === 'createRenderPipeline');
  assert.equal(rpCall.desc.layout.kind, 'GPUPipelineLayout', 'preloaded S6 reflection should build explicit layout');
});

// ---------------------------------------------------------------------------
// 12. Custom wgslPackCache implementing loadFromBuffer or parse during init
// ---------------------------------------------------------------------------

test('G28 init preloading supports custom wgslPackCache with loadFromBuffer() or parse()', async () => {
  const mock = createMockWebGpu();
  const vsHash = '0x1234000000005678';
  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL },
  ]);

  let loadFromBufferCalled = 0;
  const customCacheA = {
    _map: new Map(),
    loadFromBuffer(buf) {
      loadFromBufferCalled += 1;
      assert.equal(buf.byteLength, packBuf.byteLength);
      this._map.set(vsHash, PACK_VS_WGSL);
    },
    getSource(h) {
      return this._map.get(h);
    },
  };

  let parseCalled = 0;
  const customCacheB = {
    _map: new Map(),
    parse(buf) {
      parseCalled += 1;
      assert.equal(buf.byteLength, packBuf.byteLength);
      this._map.set(vsHash, PACK_VS_WGSL);
    },
    getWgsl(h) {
      return this._map.get(h);
    },
  };

  const mockFetch = async () => ({
    ok: true,
    status: 200,
    async arrayBuffer() {
      return packBuf.slice(0);
    },
  });

  const procA = new WgpuCommandProcessor(mock.device, mock.ctx, { wgslPackCache: customCacheA });
  const handlerA = createWgpuWorkerHandler({
    processor: procA,
    initGpu: async () => procA,
    fetch: mockFetch,
  });
  await handlerA.handleMessage({ init: true, canvas: {}, shaderPackUrl: '/shaders/a.wgsl.pack' });
  assert.equal(loadFromBufferCalled, 1);

  procA.executeCommandBuffer(new PacketBuilder().createShaderModule(1, SHADER_STAGE.VERTEX, vsHash).finish());
  assert.equal(procA.shaderModules.get(1).source, 'pack');
  assert.equal(procA.shaderModules.get(1).code, PACK_VS_WGSL);

  const procB = new WgpuCommandProcessor(mock.device, mock.ctx, { wgslPackCache: customCacheB });
  const handlerB = createWgpuWorkerHandler({
    processor: procB,
    fetch: mockFetch,
  });
  await handlerB.handleMessage({ init: true, shaderPackUrl: '/shaders/b.wgsl.pack' });
  assert.equal(parseCalled, 1);

  procB.executeCommandBuffer(new PacketBuilder().createShaderModule(1, SHADER_STAGE.VERTEX, vsHash).finish());
  assert.equal(procB.shaderModules.get(1).source, 'pack');
  assert.equal(procB.shaderModules.get(1).code, PACK_VS_WGSL);
});

// ---------------------------------------------------------------------------
// 13. Graceful resilience when shaderPackUrl or reflectionManifestUrl fails
// ---------------------------------------------------------------------------

test('G28 init gracefully ignores 404, network errors, corrupt WGPK, and bad reflection JSON', async () => {
  const scenarios = [
    {
      name: 'HTTP 404 for both URLs',
      fetch: async () => ({ ok: false, status: 404 }),
    },
    {
      name: 'Network rejection for both URLs',
      fetch: async () => {
        throw new Error('ECONNREFUSED');
      },
    },
    {
      name: 'Corrupt WGPK magic + invalid JSON',
      fetch: async (url) => ({
        ok: true,
        status: 200,
        async arrayBuffer() {
          return new Uint8Array([0x00, 0x01, 0x02, 0x03]).buffer;
        },
        async text() {
          return url.endsWith('.json') ? '{broken-json' : '';
        },
      }),
    },
    {
      name: 'Invalid reflection schema version',
      fetch: async () => ({
        ok: true,
        status: 200,
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
        async text() {
          return JSON.stringify({ schema_version: 999 });
        },
      }),
    },
  ];

  for (const scenario of scenarios) {
    const mock = createMockWebGpu();
    const posted = [];
    const processor = new WgpuCommandProcessor(mock.device, mock.ctx);
    const handler = createWgpuWorkerHandler({
      processor,
      initGpu: async () => {
        posted.push({ ready: true });
        return processor;
      },
      fetch: scenario.fetch,
      postMessage: (m) => posted.push(m),
    });

    await handler.handleMessage({
      init: true,
      canvas: {},
      shaderPackUrl: '/shaders/bad.wgsl.pack',
      reflectionManifestUrl: '/shaders/bad-reflection.json',
    });

    assert.equal(
      posted.some((m) => m.error),
      false,
      `${scenario.name}: init must not emit fatal error`
    );
    assert.equal(
      posted.some((m) => m.ready === true),
      true,
      `${scenario.name}: init must still complete ready`
    );

    // Built-in fallback still keeps shader creation alive!
    processor.executeCommandBuffer(
      new PacketBuilder().createShaderModule(1, SHADER_STAGE.FRAGMENT, '').finish()
    );
    assert.equal(processor.shaderModules.get(1).code, BUILTIN_FALLBACK_WGSL.fragment);
  }
});

// ---------------------------------------------------------------------------
// 14. Partial preload failure: valid reflection still loads when pack 404s (and vice versa)
// ---------------------------------------------------------------------------

test('G28 partial init preload failure isolates shaderPackUrl and reflectionManifestUrl independently', async () => {
  const mock = createMockWebGpu();
  const vsHash = formatHashHex(fnv1a64(PACK_VS_WGSL));
  const fsHash = formatHashHex(fnv1a64(PACK_FS_WGSL));
  const manifest = buildReflectionManifestForHash(vsHash, fsHash);
  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL },
  ]);

  // Case A: pack 404s, reflection succeeds
  const procA = new WgpuCommandProcessor(mock.device, mock.ctx);
  const handlerA = createWgpuWorkerHandler({
    processor: procA,
    initGpu: async () => procA,
    fetch: async (url) => {
      if (url.endsWith('.wgsl.pack')) return { ok: false, status: 404 };
      return { ok: true, status: 200, async json() { return manifest; } };
    },
  });
  await handlerA.handleMessage({
    init: true,
    canvas: {},
    shaderPackUrl: '/shaders/missing.wgsl.pack',
    reflectionManifestUrl: '/shaders/reflection-manifest.json',
  });
  assert.equal(procA.wgslPackCache.getStats().shaderCount, 0);
  assert.equal(procA.getStats().reflectionShaderCount, 2);

  // Case B: pack succeeds, reflection rejects
  const procB = new WgpuCommandProcessor(mock.device, mock.ctx);
  const handlerB = createWgpuWorkerHandler({
    processor: procB,
    initGpu: async () => procB,
    fetch: async (url) => {
      if (url.endsWith('.json')) throw new Error('offline');
      return { ok: true, status: 200, async arrayBuffer() { return packBuf.slice(0); } };
    },
  });
  await handlerB.handleMessage({
    init: true,
    canvas: {},
    shaderPackUrl: '/shaders/fallback.wgsl.pack',
    reflectionManifestUrl: '/shaders/reflection-manifest.json',
  });
  assert.equal(procB.wgslPackCache.getStats().shaderCount, 1);
  assert.equal(procB.getStats().reflectionShaderCount, 0);
});

// ---------------------------------------------------------------------------
// 15. Worker missingShaders telemetry message & destroy cleanup
// ---------------------------------------------------------------------------

test('G28 worker missingShaders message returns telemetry and destroy clears owned wgslPackCache', async () => {
  const mock = createMockWebGpu();
  const posted = [];
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx);
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
  });

  processor.loadWgslPack(
    buildWgslPackBuffer([
      { hash: '0x0000000000000001', stage: WGPK_SHADER_STAGE.VERTEX, wgsl: PACK_VS_WGSL },
    ])
  );
  assert.equal(processor.wgslPackCache.getStats().shaderCount, 1);

  const stream = new PacketBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, '0x0000000000000001')
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, '0x00000000000000ff')
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, '0x00000000000000ff')
    .finish();
  await handler.handleMessage({ cmdBuffer: stream });

  await handler.handleMessage({ id: 77, missingShaders: true, limit: 5 });
  const msg = posted.find((m) => m.id === 77);
  assert.ok(msg);
  assert.equal(msg.missingShaders.length, 1);
  assert.equal(msg.missingShaders[0].hash, '0x00000000000000ff');
  assert.equal(msg.missingShaders[0].hitCount, 2);
  assert.equal(msg.shaderFallbackStats.packHits, 1);
  assert.equal(msg.shaderFallbackStats.builtinHits, 2);

  await handler.handleMessage({ id: 78, destroy: true });
  assert.equal(processor.destroyed, true);
  assert.equal(processor.wgslPackCache.getStats().shaderCount, 0);
});
