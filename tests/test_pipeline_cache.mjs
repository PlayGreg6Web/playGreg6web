import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const pipelineCacheMod = require('../web/workers/pipeline_cache.js');
const shaderFallbackMod = require('../web/workers/shader_fallback_router.js');
const {
  DEFAULT_MAX_ERROR_HISTORY,
  MAX_PIPELINE_ERROR_MESSAGE_LENGTH,
  FRAME1_VERTEX_STRIDE_24B,
  FRAME1_CBV_MIN_BINDING_SIZE,
  FRAME1_VERTEX_LAYOUT_24B,
  SHADER_STAGE_BITS,
  hashDescriptorKey,
  registerShaderModuleId,
  normalizeFallbackProfile,
  getVertexFormatByteSize,
  buildVertexBufferLayoutForStride,
  normalizeVertexBufferLayouts,
  buildFallbackBindGroupLayoutDescriptor,
  normalizeRenderPipelineDescriptor,
  sanitizePipelineErrorMessage,
  WebGpuPipelineCache,
} = pipelineCacheMod;
const { ShaderFallbackRouter } = shaderFallbackMod;

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createMockGpuDevice({
  withAsync = true,
  withPipelineLayout = true,
  limits = undefined,
  advanceClockOnCompile = null,
  renderSyncImpl = null,
  renderAsyncImpl = null,
  computeSyncImpl = null,
  computeAsyncImpl = null,
  bglImpl = null,
  pipelineLayoutImpl = null,
} = {}) {
  let nextId = 1;
  const calls = {
    createRenderPipeline: [],
    createRenderPipelineAsync: [],
    createComputePipeline: [],
    createComputePipelineAsync: [],
    createBindGroupLayout: [],
    createPipelineLayout: [],
    createShaderModule: [],
  };

  const device = {
    calls,
    ...(limits ? { limits } : {}),
    createShaderModule(desc) {
      const id = nextId++;
      const mod = {
        label: desc.label || '',
        getCompilationInfo: async () => ({ messages: [] }),
      };
      calls.createShaderModule.push({ id, desc, mod });
      return mod;
    },
    createBindGroupLayout(desc) {
      if (advanceClockOnCompile) advanceClockOnCompile(1.5);
      const id = nextId++;
      if (bglImpl) {
        const layout = bglImpl(desc, id);
        calls.createBindGroupLayout.push({ id, desc, layout });
        return layout;
      }
      const layout = { kind: 'GPUBindGroupLayout', id, desc };
      calls.createBindGroupLayout.push({ id, desc, layout });
      return layout;
    },
    createRenderPipeline(desc) {
      if (advanceClockOnCompile) advanceClockOnCompile(4.0);
      const id = nextId++;
      if (renderSyncImpl) {
        const pipeline = renderSyncImpl(desc, id);
        calls.createRenderPipeline.push({ id, desc, pipeline });
        return pipeline;
      }
      const pipeline = { kind: 'GPURenderPipeline', id, desc };
      calls.createRenderPipeline.push({ id, desc, pipeline });
      return pipeline;
    },
    createComputePipeline(desc) {
      if (advanceClockOnCompile) advanceClockOnCompile(2.5);
      const id = nextId++;
      if (computeSyncImpl) {
        const pipeline = computeSyncImpl(desc, id);
        calls.createComputePipeline.push({ id, desc, pipeline });
        return pipeline;
      }
      const pipeline = { kind: 'GPUComputePipeline', id, desc };
      calls.createComputePipeline.push({ id, desc, pipeline });
      return pipeline;
    },
  };

  if (withPipelineLayout) {
    device.createPipelineLayout = (desc) => {
      const id = nextId++;
      if (pipelineLayoutImpl) {
        const layout = pipelineLayoutImpl(desc, id);
        calls.createPipelineLayout.push({ id, desc, layout });
        return layout;
      }
      const layout = { kind: 'GPUPipelineLayout', id, desc };
      calls.createPipelineLayout.push({ id, desc, layout });
      return layout;
    };
  }

  if (withAsync) {
    device.createRenderPipelineAsync = async (desc) => {
      const id = nextId++;
      const record = { id, desc };
      calls.createRenderPipelineAsync.push(record);
      if (renderAsyncImpl) {
        return renderAsyncImpl(desc, id);
      }
      if (advanceClockOnCompile) advanceClockOnCompile(5.0);
      return { kind: 'GPURenderPipeline', id, async: true, desc };
    };

    device.createComputePipelineAsync = async (desc) => {
      const id = nextId++;
      const record = { id, desc };
      calls.createComputePipelineAsync.push(record);
      if (computeAsyncImpl) {
        return computeAsyncImpl(desc, id);
      }
      if (advanceClockOnCompile) advanceClockOnCompile(3.0);
      return { kind: 'GPUComputePipeline', id, async: true, desc };
    };
  }

  return device;
}

test('UMD module exposes globalThis.PlayGta6PipelineCache and exports expected API', () => {
  assert.equal(globalThis.PlayGta6PipelineCache, pipelineCacheMod);
  assert.equal(typeof hashDescriptorKey, 'function');
  assert.equal(typeof WebGpuPipelineCache, 'function');
  assert.equal(SHADER_STAGE_BITS.VERTEX, 0x1);
  assert.equal(SHADER_STAGE_BITS.FRAGMENT, 0x2);
  assert.equal(SHADER_STAGE_BITS.COMPUTE, 0x4);
});

test('hashDescriptorKey is deterministic, key-order invariant, and handles nested objects and BigInt', () => {
  const h1 = hashDescriptorKey({ a: 1, b: 2 });
  const h2 = hashDescriptorKey({ b: 2, a: 1 });
  assert.match(h1, /^0x[0-9a-f]{16}$/);
  assert.equal(h1, h2);

  // Nested objects with different key insertion orders + undefined omission
  const nestedA = {
    layout: 'auto',
    vertex: {
      entryPoint: 'vs_main',
      buffers: [{ arrayStride: 16, attributes: [{ format: 'float32x4', offset: 0, shaderLocation: 0 }] }],
    },
    fragment: {
      targets: [{ format: 'bgra8unorm', writeMask: 0xf }],
      entryPoint: 'fs_main',
    },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: undefined,
  };
  const nestedB = {
    primitive: { cullMode: 'back', topology: 'triangle-list' },
    fragment: {
      entryPoint: 'fs_main',
      targets: [{ writeMask: 0xf, format: 'bgra8unorm' }],
    },
    vertex: {
      buffers: [{ attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }], arrayStride: 16 }],
      entryPoint: 'vs_main',
    },
    layout: 'auto',
  };
  assert.equal(hashDescriptorKey(nestedA), hashDescriptorKey(nestedB));

  // Array order IS significant
  assert.notEqual(hashDescriptorKey({ list: [1, 2] }), hashDescriptorKey({ list: [2, 1] }));

  // BigInt support (including 64-bit WGPK hashes) and type separation
  const wgpkHashA = 0xdeadbeefcafebaben;
  const wgpkHashB = 0xdeadbeefcafebabfn;
  assert.equal(
    hashDescriptorKey({ vs: wgpkHashA, fs: wgpkHashB }),
    hashDescriptorKey({ fs: wgpkHashB, vs: wgpkHashA })
  );
  assert.notEqual(hashDescriptorKey({ id: wgpkHashA }), hashDescriptorKey({ id: wgpkHashB }));
  assert.notEqual(hashDescriptorKey({ id: 1n }), hashDescriptorKey({ id: 1 }));
  assert.notEqual(hashDescriptorKey({ id: 1n }), hashDescriptorKey({ id: '1' }));

  // -0 normalizes to 0
  assert.equal(hashDescriptorKey({ v: -0 }), hashDescriptorKey({ v: 0 }));

  // Circular references throw cleanly
  const circular = {};
  circular.self = circular;
  assert.throws(() => hashDescriptorKey(circular), /Circular reference/);
});

test('hashDescriptorKey distinguishes opaque GPUShaderModules and honors WGPK shader IDs', () => {
  const device = createMockGpuDevice();
  const modA = device.createShaderModule({ code: 'fn a() {}' });
  const modB = device.createShaderModule({ code: 'fn b() {}' });

  const descA1 = { layout: 'auto', vertex: { module: modA, entryPoint: 'vs_main' } };
  const descA2 = { vertex: { entryPoint: 'vs_main', module: modA }, layout: 'auto' };
  const descB = { layout: 'auto', vertex: { module: modB, entryPoint: 'vs_main' } };

  assert.equal(hashDescriptorKey(descA1), hashDescriptorKey(descA2));
  assert.notEqual(hashDescriptorKey(descA1), hashDescriptorKey(descB));

  // WGPK-labeled shader modules with identical hash produce identical descriptor keys
  const wgpkMod1 = device.createShaderModule({
    label: 'wgpk:vertex:0x0123456789abcdef',
    code: 'fn vs() {}',
  });
  const wgpkMod2 = device.createShaderModule({
    label: 'wgpk:vertex:0x0123456789abcdef',
    code: 'fn vs() {}',
  });
  assert.equal(
    hashDescriptorKey({ vertex: { module: wgpkMod1, entryPoint: 'vs_main' } }),
    hashDescriptorKey({ vertex: { module: wgpkMod2, entryPoint: 'vs_main' } })
  );

  // Explicit registerShaderModuleId binds opaque module objects to WGPK hashes
  const opaque1 = {};
  const opaque2 = {};
  assert.notEqual(
    hashDescriptorKey({ vertex: { module: opaque1 } }),
    hashDescriptorKey({ vertex: { module: opaque2 } })
  );
  registerShaderModuleId(opaque1, 0xfeedface12345678n);
  registerShaderModuleId(opaque2, 0xfeedface12345678n);
  assert.equal(
    hashDescriptorKey({ vertex: { module: opaque1 } }),
    hashDescriptorKey({ vertex: { module: opaque2 } })
  );
});

test('sync render, compute, and bindGroupLayout cache hits, misses, and telemetry', () => {
  let clock = 100;
  const device = createMockGpuDevice({
    advanceClockOnCompile: (dt) => {
      clock += dt;
    },
  });
  const cache = new WebGpuPipelineCache({ device, now: () => clock });

  const vs = device.createShaderModule({ label: 'wgpk:vertex:0x0000000000000001', code: 'vs' });
  const fs = device.createShaderModule({ label: 'wgpk:fragment:0x0000000000000002', code: 'fs' });
  const cs = device.createShaderModule({ label: 'wgpk:compute:0x0000000000000003', code: 'cs' });

  // BindGroupLayout miss then hit (with reordered descriptor keys)
  const bgl1 = cache.getOrCreateBindGroupLayout({
    entries: [{ binding: 0, visibility: SHADER_STAGE_BITS.VERTEX, buffer: { type: 'uniform' } }],
  });
  const bgl2 = cache.getOrCreateBindGroupLayout({
    entries: [{ buffer: { type: 'uniform' }, visibility: SHADER_STAGE_BITS.VERTEX, binding: 0 }],
  });
  assert.equal(bgl1, bgl2);
  assert.equal(device.calls.createBindGroupLayout.length, 1);

  // RenderPipeline miss then hit (reordered keys)
  const rp1 = cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_main' },
    fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  const rp2 = cache.getOrCreateRenderPipeline({
    fragment: { targets: [{ format: 'bgra8unorm' }], entryPoint: 'fs_main', module: fs },
    vertex: { entryPoint: 'vs_main', module: vs },
    layout: 'auto',
  });
  assert.equal(rp1, rp2);
  assert.equal(device.calls.createRenderPipeline.length, 1);

  // ComputePipeline miss then hit
  const cp1 = cache.getOrCreateComputePipeline({
    layout: 'auto',
    compute: { module: cs, entryPoint: 'cs_main' },
  });
  const cp2 = cache.getOrCreateComputePipeline({
    compute: { entryPoint: 'cs_main', module: cs },
    layout: 'auto',
  });
  assert.equal(cp1, cp2);
  assert.equal(device.calls.createComputePipeline.length, 1);

  // KeyHint override allows precomputed BigInt/string keys
  const rpHint1 = cache.getOrCreateRenderPipeline(
    { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_alt' } },
    0xabcdef01n
  );
  const rpHint2 = cache.getOrCreateRenderPipeline(
    { layout: 'auto', vertex: { module: vs, entryPoint: 'completely_different_desc' } },
    0xabcdef01n
  );
  assert.equal(rpHint1, rpHint2);
  assert.equal(device.calls.createRenderPipeline.length, 2);

  const stats = cache.getStats();
  assert.deepEqual(stats, {
    renderHits: 2,
    renderMisses: 2,
    computeHits: 1,
    computeMisses: 1,
    layoutHits: 1,
    layoutMisses: 1,
    inFlightDedupes: 0,
    evictions: 0,
    renderCount: 2,
    computeCount: 1,
    layoutCount: 1,
    totalCompileMs: 1.5 + 4.0 + 2.5 + 4.0,
    fallbackNormalizedCount: 0,
    vertexLayoutNormalizedCount: 0,
    layoutNormalizedCount: 0,
    fallbackRecoveryCount: 0,
    pipelineErrorCount: 0,
    pipelineErrorHistorySize: 0,
    maxErrorHistory: DEFAULT_MAX_ERROR_HISTORY,
  });
});

test('concurrent async render & compute pipeline compilations deduplicate in-flight promises', async () => {
  const renderDeferred = createDeferred();
  const computeDeferred = createDeferred();
  let clock = 0;

  const device = createMockGpuDevice({
    renderAsyncImpl: async (desc, id) => {
      const res = await renderDeferred.promise;
      clock += 7.5;
      return { kind: 'GPURenderPipeline', id, tag: res, desc };
    },
    computeAsyncImpl: async (desc, id) => {
      const res = await computeDeferred.promise;
      clock += 4.5;
      return { kind: 'GPUComputePipeline', id, tag: res, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device, now: () => clock });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  const renderDesc1 = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_main' } };
  const renderDesc2 = { vertex: { entryPoint: 'vs_main', module: vs }, layout: 'auto' };

  // Two simultaneous render async requests for the same PSO
  const rpPromise1 = cache.getOrCreateRenderPipelineAsync(renderDesc1);
  const rpPromise2 = cache.getOrCreateRenderPipelineAsync(renderDesc2);
  assert.equal(device.calls.createRenderPipelineAsync.length, 1);

  // Two simultaneous compute async requests for the same PSO
  const computeDesc = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_main' } };
  const cpPromise1 = cache.getOrCreateComputePipelineAsync(computeDesc);
  const cpPromise2 = cache.getOrCreateComputePipelineAsync(computeDesc);
  assert.equal(device.calls.createComputePipelineAsync.length, 1);

  renderDeferred.resolve('ok-render');
  const [rp1, rp2] = await Promise.all([rpPromise1, rpPromise2]);

  computeDeferred.resolve('ok-compute');
  const [cp1, cp2] = await Promise.all([cpPromise1, cpPromise2]);

  assert.equal(rp1, rp2);
  assert.equal(cp1, cp2);

  // Subsequent call hits the populated cache directly
  const rp3 = await cache.getOrCreateRenderPipelineAsync(renderDesc1);
  const cp3 = await cache.getOrCreateComputePipelineAsync(computeDesc);
  assert.equal(rp3, rp1);
  assert.equal(cp3, cp1);
  assert.equal(device.calls.createRenderPipelineAsync.length, 1);
  assert.equal(device.calls.createComputePipelineAsync.length, 1);

  const stats = cache.getStats();
  assert.equal(stats.renderMisses, 1);
  assert.equal(stats.renderHits, 1);
  assert.equal(stats.computeMisses, 1);
  assert.equal(stats.computeHits, 1);
  assert.equal(stats.inFlightDedupes, 2);
  assert.equal(stats.renderCount, 1);
  assert.equal(stats.computeCount, 1);
  // Render compiled 0 -> 7.5ms (7.5ms); compute compiled 0 -> 12.0ms (12.0ms)
  assert.equal(stats.totalCompileMs, 19.5);
});

test('rejected async pipeline promises are cleaned up and do not poison the cache', async () => {
  let renderAttempts = 0;
  let computeAttempts = 0;

  const device = createMockGpuDevice({
    renderAsyncImpl: async (desc, id) => {
      renderAttempts += 1;
      if (renderAttempts === 1) {
        throw new Error('transient render compile failure');
      }
      return { kind: 'GPURenderPipeline', id, attempt: renderAttempts, desc };
    },
    computeAsyncImpl: async (desc, id) => {
      computeAttempts += 1;
      if (computeAttempts === 1) {
        throw new Error('transient compute compile failure');
      }
      return { kind: 'GPUComputePipeline', id, attempt: computeAttempts, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });
  const renderDesc = { layout: 'auto', vertex: { module: vs, entryPoint: 'vs_main' } };
  const computeDesc = { layout: 'auto', compute: { module: cs, entryPoint: 'cs_main' } };

  // Two concurrent callers observe the first failure
  const p1 = cache.getOrCreateRenderPipelineAsync(renderDesc);
  const p2 = cache.getOrCreateRenderPipelineAsync(renderDesc);
  await assert.rejects(p1, /transient render compile failure/);
  await assert.rejects(p2, /transient render compile failure/);
  assert.equal(cache.getStats().renderCount, 0);

  // Retry succeeds and populates the cache
  const recoveredRender = await cache.getOrCreateRenderPipelineAsync(renderDesc);
  assert.equal(recoveredRender.attempt, 2);
  assert.equal(cache.getStats().renderCount, 1);

  // Same for compute pipeline
  await assert.rejects(
    cache.getOrCreateComputePipelineAsync(computeDesc),
    /transient compute compile failure/
  );
  assert.equal(cache.getStats().computeCount, 0);

  const recoveredCompute = await cache.getOrCreateComputePipelineAsync(computeDesc);
  assert.equal(recoveredCompute.attempt, 2);
  assert.equal(cache.getStats().computeCount, 1);
});

test('async methods fall back to synchronous createRenderPipeline / createComputePipeline when async APIs are absent', async () => {
  const device = createMockGpuDevice({ withAsync: false });
  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  const rp = await cache.getOrCreateRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_main' },
  });
  const cp = await cache.getOrCreateComputePipelineAsync({
    layout: 'auto',
    compute: { module: cs, entryPoint: 'cs_main' },
  });

  assert.equal(rp.kind, 'GPURenderPipeline');
  assert.equal(cp.kind, 'GPUComputePipeline');
  assert.equal(device.calls.createRenderPipeline.length, 1);
  assert.equal(device.calls.createComputePipeline.length, 1);
});

test('LRU eviction drops the least recently used entries when max capacity is exceeded', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({
    device,
    maxRenderPipelines: 2,
    maxComputePipelines: 2,
    maxBindGroupLayouts: 2,
  });

  const vs = device.createShaderModule({ code: 'vs' });
  const makeRenderDesc = (entryPoint) => ({
    layout: 'auto',
    vertex: { module: vs, entryPoint },
  });

  const pA = cache.getOrCreateRenderPipeline(makeRenderDesc('vs_a'));
  const pB = cache.getOrCreateRenderPipeline(makeRenderDesc('vs_b'));
  assert.equal(cache.getStats().renderCount, 2);
  assert.equal(cache.getStats().evictions, 0);

  // Touch A so B becomes the least recently used entry
  const pAHit = cache.getOrCreateRenderPipeline(makeRenderDesc('vs_a'));
  assert.equal(pAHit, pA);

  // Insert C -> evicts B, keeps A and C
  const pC = cache.getOrCreateRenderPipeline(makeRenderDesc('vs_c'));
  assert.equal(cache.getStats().renderCount, 2);
  assert.equal(cache.getStats().evictions, 1);
  assert.equal(cache.getOrCreateRenderPipeline(makeRenderDesc('vs_a')), pA);
  assert.equal(cache.getOrCreateRenderPipeline(makeRenderDesc('vs_c')), pC);

  // Requesting B again must recompile (miss) and evict A (since C was touched after A)
  const pBRecompiled = cache.getOrCreateRenderPipeline(makeRenderDesc('vs_b'));
  assert.notEqual(pBRecompiled, pB);
  assert.equal(cache.getStats().evictions, 2);
  assert.equal(device.calls.createRenderPipeline.length, 4);

  // Compute pipeline LRU eviction
  const cs = device.createShaderModule({ code: 'cs' });
  cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'c1' } });
  cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'c2' } });
  cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'c3' } });
  assert.equal(cache.getStats().computeCount, 2);
  assert.equal(cache.getStats().evictions, 3);

  // BindGroupLayout LRU eviction
  cache.getOrCreateBindGroupLayout({ entries: [{ binding: 0, visibility: 1, buffer: {} }] });
  cache.getOrCreateBindGroupLayout({ entries: [{ binding: 1, visibility: 1, buffer: {} }] });
  cache.getOrCreateBindGroupLayout({ entries: [{ binding: 2, visibility: 1, buffer: {} }] });
  assert.equal(cache.getStats().layoutCount, 2);
  assert.equal(cache.getStats().evictions, 4);
});

test('createBindGroupLayoutFromReflection normalizes S6 reflection metadata and deduplicates layouts', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });

  const reflectionGroupA = {
    group: 0,
    entries: [
      {
        binding: 2,
        visibility: ['fragment'],
        texture: { sample_type: 'float', view_dimension: '2d', multisampled: false },
      },
      {
        binding: 0,
        visibility: 'vertex | fragment',
        buffer: { type: 'uniform', has_dynamic_offset: true, min_binding_size: 256 },
      },
      {
        binding: 1,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: { type: 'filtering' },
      },
      {
        binding: 3,
        visibility: 'compute',
        storage_texture: { access: 'write-only', format: 'rgba8unorm', view_dimension: '2d' },
      },
    ],
  };

  // Same schema provided as a plain array in a different binding order using camelCase
  const reflectionGroupB = [
    {
      binding: 0,
      visibility: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT,
      buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 256 },
    },
    {
      binding: 1,
      visibility: 'fs',
      sampler: { type: 'filtering' },
    },
    {
      binding: 2,
      visibility: 'fragment',
      texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
    },
    {
      binding: 3,
      visibility: ['cs'],
      storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' },
    },
  ];

  const layoutA = cache.createBindGroupLayoutFromReflection(reflectionGroupA);
  const layoutB = cache.createBindGroupLayoutFromReflection(reflectionGroupB);

  assert.equal(layoutA, layoutB);
  assert.equal(device.calls.createBindGroupLayout.length, 1);

  const submittedDesc = device.calls.createBindGroupLayout[0].desc;
  assert.deepEqual(submittedDesc, {
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 256 },
      },
      {
        binding: 1,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: { type: 'filtering' },
      },
      {
        binding: 2,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
      },
      {
        binding: 3,
        visibility: SHADER_STAGE_BITS.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' },
      },
    ],
  });

  // Duplicate binding index is rejected
  assert.throws(
    () =>
      cache.createBindGroupLayoutFromReflection([
        { binding: 0, visibility: 'vertex', buffer: {} },
        { binding: 0, visibility: 'fragment', buffer: {} },
      ]),
    /Duplicate reflection binding index 0/
  );

  // Unknown stage name is rejected
  assert.throws(
    () =>
      cache.createBindGroupLayoutFromReflection([
        { binding: 0, visibility: 'geometry', buffer: {} },
      ]),
    /Unknown shader stage visibility/
  );
});

test('prewarmRenderPipelines and prewarmComputePipelines bound concurrency and deduplicate', async () => {
  let activeRender = 0;
  let peakRender = 0;
  let activeCompute = 0;
  let peakCompute = 0;

  const device = createMockGpuDevice({
    renderAsyncImpl: async (desc, id) => {
      activeRender += 1;
      peakRender = Math.max(peakRender, activeRender);
      await new Promise((r) => setTimeout(r, 8));
      activeRender -= 1;
      return { kind: 'GPURenderPipeline', id, desc };
    },
    computeAsyncImpl: async (desc, id) => {
      activeCompute += 1;
      peakCompute = Math.max(peakCompute, activeCompute);
      await new Promise((r) => setTimeout(r, 8));
      activeCompute -= 1;
      return { kind: 'GPUComputePipeline', id, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  // 8 render descriptors with 1 duplicate -> 7 unique compilations, max concurrency = 3
  const renderDescriptors = [];
  for (let i = 0; i < 7; i += 1) {
    renderDescriptors.push({
      layout: 'auto',
      vertex: { module: vs, entryPoint: `vs_${i}` },
    });
  }
  renderDescriptors.push({
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'vs_0' },
  });

  const renderSummary = await cache.prewarmRenderPipelines(renderDescriptors, { concurrency: 3 });
  assert.equal(renderSummary.total, 8);
  assert.equal(renderSummary.pipelines.length, 8);
  assert.equal(renderSummary.pipelines[0], renderSummary.pipelines[7]);
  assert.equal(device.calls.createRenderPipelineAsync.length, 7);
  assert.ok(peakRender <= 3, `expected peak render concurrency <= 3, got ${peakRender}`);
  assert.ok(peakRender >= 2, `expected parallel prewarm execution, got peak ${peakRender}`);

  // 6 compute descriptors, max concurrency = 2
  const computeDescriptors = Array.from({ length: 6 }, (_, i) => ({
    layout: 'auto',
    compute: { module: cs, entryPoint: `cs_${i}` },
  }));

  const computeSummary = await cache.prewarmComputePipelines(computeDescriptors, { concurrency: 2 });
  assert.equal(computeSummary.total, 6);
  assert.equal(computeSummary.compiled, 6);
  assert.equal(computeSummary.cached, 0);
  assert.equal(device.calls.createComputePipelineAsync.length, 6);
  assert.ok(peakCompute <= 2, `expected peak compute concurrency <= 2, got ${peakCompute}`);

  // Second prewarm hits cache without compiling anything
  const cachedSummary = await cache.prewarmComputePipelines(computeDescriptors, { concurrency: 2 });
  assert.equal(cachedSummary.cached, 6);
  assert.equal(cachedSummary.compiled, 0);
  assert.equal(device.calls.createComputePipelineAsync.length, 6);
});

test('evictOlderThan and clear manage lifecycle and isolate stale in-flight compilations', async () => {
  let clock = 1000;
  const pendingRender = createDeferred();

  const device = createMockGpuDevice({
    renderAsyncImpl: async (desc, id) => {
      await pendingRender.promise;
      return { kind: 'GPURenderPipeline', id, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device, now: () => clock });
  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  // Populate entries at t = 1000
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'old_vs' } });
  cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'old_cs' } });
  cache.getOrCreateBindGroupLayout({ entries: [{ binding: 0, visibility: 1, buffer: {} }] });

  // Advance clock to t = 2500 and touch compute + add fresh render
  clock = 2500;
  cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'old_cs' } });
  cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'new_vs' } });

  // At t = 2600, evict anything older than 1000ms (cutoff = 1600):
  // old_vs (t=1000) and layout (t=1000) are evicted; old_cs (t=2500) and new_vs (t=2500) survive.
  clock = 2600;
  const evicted = cache.evictOlderThan(1000);
  assert.equal(evicted, 2);
  assert.equal(cache.getStats().renderCount, 1);
  assert.equal(cache.getStats().computeCount, 1);
  assert.equal(cache.getStats().layoutCount, 0);
  assert.equal(cache.getStats().evictions, 2);

  // Start an in-flight async compile, then clear() before it resolves
  const inFlightPromise = cache.getOrCreateRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: vs, entryPoint: 'inflight_vs' },
  });
  cache.clear();
  assert.equal(cache.getStats().renderCount, 0);
  assert.equal(cache.getStats().computeCount, 0);
  assert.equal(cache.getStats().layoutCount, 0);

  // Resolving the pre-clear compilation should still return to caller but NOT repopulate cache
  pendingRender.resolve();
  const stalePipeline = await inFlightPromise;
  assert.equal(stalePipeline.kind, 'GPURenderPipeline');
  assert.equal(cache.getStats().renderCount, 0);

  // clear({ resetStats: true }) resets telemetry counters
  cache.clear({ resetStats: true });
  assert.deepEqual(cache.getStats(), {
    renderHits: 0,
    renderMisses: 0,
    computeHits: 0,
    computeMisses: 0,
    layoutHits: 0,
    layoutMisses: 0,
    inFlightDedupes: 0,
    evictions: 0,
    renderCount: 0,
    computeCount: 0,
    layoutCount: 0,
    totalCompileMs: 0,
    fallbackNormalizedCount: 0,
    vertexLayoutNormalizedCount: 0,
    layoutNormalizedCount: 0,
    fallbackRecoveryCount: 0,
    pipelineErrorCount: 0,
    pipelineErrorHistorySize: 0,
    maxErrorHistory: DEFAULT_MAX_ERROR_HISTORY,
  });
});

// --- G36: Fallback-Safe Pipeline & 24B Vertex/CBV Layout Normalization Tests ---

test('G36 preserves classic-worker importScripts IIFE isolation with zero global leakage', () => {
  const scriptPath = path.resolve(
    path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')),
    '../web/workers/pipeline_cache.js'
  );
  const source = fs.readFileSync(scriptPath, 'utf8');
  const sandbox = {
    TextEncoder,
    performance,
    Date,
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);

  assert.equal(typeof sandbox.PlayGta6PipelineCache, 'object');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.WebGpuPipelineCache, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.normalizeVertexBufferLayouts, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.normalizeRenderPipelineDescriptor, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.buildFallbackBindGroupLayoutDescriptor, 'function');
  assert.equal(typeof sandbox.PlayGta6PipelineCache.sanitizePipelineErrorMessage, 'function');

  // Verify no internal constants or helpers leaked onto global scope.
  assert.equal(sandbox.SHADER_STAGE_BITS, undefined);
  assert.equal(sandbox.FRAME1_VERTEX_LAYOUT_24B, undefined);
  assert.equal(sandbox.normalizeVertexBufferLayouts, undefined);
  assert.equal(sandbox.WebGpuPipelineCache, undefined);
});

test('G36 normalizeFallbackProfile and getVertexFormatByteSize handle aliases and invalid inputs', () => {
  assert.equal(normalizeFallbackProfile('default'), 'default');
  assert.equal(normalizeFallbackProfile('minimal'), 'default');
  assert.equal(normalizeFallbackProfile('fullscreen_triangle_cbv'), 'fullscreen_triangle_cbv');
  assert.equal(normalizeFallbackProfile('fullscreen-cbv'), 'fullscreen_triangle_cbv');
  assert.equal(normalizeFallbackProfile('FULLSCREEN_CBV'), 'fullscreen_triangle_cbv');
  assert.equal(normalizeFallbackProfile('textured_quad_24b'), 'textured_quad_24b');
  assert.equal(normalizeFallbackProfile('textured-24b'), 'textured_quad_24b');
  assert.equal(normalizeFallbackProfile('unknown-profile'), 'default');
  assert.equal(normalizeFallbackProfile(null), 'default');
  assert.equal(normalizeFallbackProfile(undefined), 'default');

  assert.equal(getVertexFormatByteSize('float32x4'), 16);
  assert.equal(getVertexFormatByteSize('float32x3'), 12);
  assert.equal(getVertexFormatByteSize('float32x2'), 8);
  assert.equal(getVertexFormatByteSize('float32'), 4);
  assert.equal(getVertexFormatByteSize('uint8x2'), 2);
  assert.equal(getVertexFormatByteSize('uint8'), 1);
  assert.equal(getVertexFormatByteSize('unorm10-10-10-2'), 4);
  assert.equal(getVertexFormatByteSize('not-a-format'), 0);
  assert.equal(getVertexFormatByteSize(null), 0);
});

test('G36 normalizeVertexBufferLayouts normalizes Frame-1 24B DrawInstanced(3, 1) and G34 profiles', () => {
  // 1. Frame-1 24-byte-stride buffer via stride number or missing/empty attributes -> canonical 24B layout
  const expected24B = [
    {
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    },
  ];
  assert.equal(FRAME1_VERTEX_STRIDE_24B, 24);
  assert.deepEqual(buildVertexBufferLayoutForStride(24), expected24B);
  assert.deepEqual(normalizeVertexBufferLayouts([{ arrayStride: 24 }]), expected24B);
  assert.deepEqual(normalizeVertexBufferLayouts([{ arrayStride: 24, attributes: [] }]), expected24B);
  assert.deepEqual(normalizeVertexBufferLayouts([], { vertexStride: 24 }), expected24B);
  assert.deepEqual(normalizeVertexBufferLayouts(24), expected24B);
  assert.equal(FRAME1_VERTEX_LAYOUT_24B.arrayStride, 24);

  // 2. fullscreen_triangle_cbv & default (@builtin(vertex_index)) ALWAYS normalize to empty buffers: []
  assert.deepEqual(
    normalizeVertexBufferLayouts(expected24B, { fallbackProfile: 'fullscreen_triangle_cbv' }),
    []
  );
  assert.deepEqual(
    normalizeVertexBufferLayouts(expected24B, { fallbackProfile: 'fullscreen-cbv' }),
    []
  );
  assert.deepEqual(
    normalizeVertexBufferLayouts(expected24B, { fallbackProfile: 'default' }),
    []
  );
  assert.deepEqual(
    buildVertexBufferLayoutForStride(24, { fallbackProfile: 'fullscreen_triangle_cbv' }),
    []
  );

  // 3. textured_quad_24b enforces canonical 24B layout even if input was missing or single-attribute
  assert.deepEqual(
    normalizeVertexBufferLayouts([], { fallbackProfile: 'textured_quad_24b' }),
    expected24B
  );
  assert.deepEqual(
    normalizeVertexBufferLayouts(
      [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }],
      { fallbackProfile: 'textured-24b' }
    ),
    expected24B
  );

  // 4. Clamping invalid offsets, unaligned strides, oversized strides, unknown formats, duplicate locations
  const clamped = normalizeVertexBufferLayouts([
    {
      arrayStride: 24,
      stepMode: 'bogus-step',
      attributes: [
        { shaderLocation: 0, offset: -8, format: 'float32x4' },
        { shaderLocation: 0, offset: 22, format: 'float32x2' }, // duplicate loc 0 & offset 22+8 > 24 -> loc 1, offset 16
      ],
    },
    {
      arrayStride: 4096, // exceeds WebGPU default 2048 -> clamps to 2048
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 99, offset: 3000, format: 'float32x4' }, // loc -> 2, offset -> 2032
      ],
    },
    {
      arrayStride: -16, // invalid negative stride -> derived from attributes
      attributes: [
        { shaderLocation: 5, offset: 3, format: 'invalid_fmt' }, // fallback float32x4, offset aligned to 0 -> stride 16
      ],
    },
    {
      arrayStride: 8, // stride smaller than float32x4 (16B) -> expands to 16B, offset 0
      attributes: [
        { shaderLocation: 6, offset: 4, format: 'float32x4' },
      ],
    },
  ]);

  assert.deepEqual(clamped, [
    {
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    },
    {
      arrayStride: 2048,
      stepMode: 'instance',
      attributes: [
        { shaderLocation: 2, offset: 2032, format: 'float32x4' },
      ],
    },
    {
      arrayStride: 16,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 5, offset: 0, format: 'float32x4' },
      ],
    },
    {
      arrayStride: 16,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 6, offset: 0, format: 'float32x4' },
      ],
    },
  ]);

  // 5. Unaligned positive stride rounds up to multiple of 4 (e.g., 22 -> 24)
  assert.deepEqual(buildVertexBufferLayoutForStride(22), expected24B);
  assert.deepEqual(buildVertexBufferLayoutForStride(0), []);
  assert.deepEqual(buildVertexBufferLayoutForStride(-4), []);
  assert.deepEqual(buildVertexBufferLayoutForStride(NaN), []);
});

test('G36 fallback bind-group and pipeline layout helpers match G34 CBV and textured_quad_24b profiles', () => {
  assert.equal(FRAME1_CBV_MIN_BINDING_SIZE, 256);

  // fullscreen_triangle_cbv -> group(0) binding(0) uniform CBV (VS | FS, minBindingSize 256)
  const cbvBglDesc = buildFallbackBindGroupLayoutDescriptor('fullscreen_triangle_cbv');
  assert.deepEqual(cbvBglDesc, {
    label: 'fallback-bgl:fullscreen_triangle_cbv:group0',
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT,
        buffer: {
          type: 'uniform',
          hasDynamicOffset: false,
          minBindingSize: 256,
        },
      },
    ],
  });

  // textured_quad_24b -> group(0) binding(0) 2d float texture + binding(1) filtering sampler
  const texBglDesc = buildFallbackBindGroupLayoutDescriptor('textured_quad_24b');
  assert.deepEqual(texBglDesc, {
    label: 'fallback-bgl:textured_quad_24b:group0',
    entries: [
      {
        binding: 0,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        texture: {
          sampleType: 'float',
          viewDimension: '2d',
          multisampled: false,
        },
      },
      {
        binding: 1,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: {
          type: 'filtering',
        },
      },
    ],
  });

  // default -> empty entries
  const defBglDesc = buildFallbackBindGroupLayoutDescriptor('default');
  assert.deepEqual(defBglDesc, {
    label: 'fallback-bgl:default:group0',
    entries: [],
  });

  // Mutating returned descriptor does not affect subsequent calls
  cbvBglDesc.entries[0].buffer.minBindingSize = 9999;
  assert.equal(
    buildFallbackBindGroupLayoutDescriptor('fullscreen_triangle_cbv').entries[0].buffer.minBindingSize,
    256
  );

  // Cached fallback BindGroupLayout & PipelineLayout deduplication
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });

  const cbvPl1 = cache.getOrCreateFallbackPipelineLayout('fullscreen_triangle_cbv');
  const cbvPl2 = cache.getOrCreateFallbackPipelineLayout('fullscreen-cbv');
  assert.equal(cbvPl1, cbvPl2);
  assert.equal(cbvPl1.kind, 'GPUPipelineLayout');
  assert.equal(cbvPl1.desc.bindGroupLayouts.length, 1);
  assert.equal(device.calls.createBindGroupLayout.length, 1);
  assert.equal(device.calls.createPipelineLayout.length, 1);

  const texPl = cache.getOrCreateFallbackPipelineLayout('textured_quad_24b');
  assert.equal(texPl.kind, 'GPUPipelineLayout');
  assert.equal(texPl.desc.bindGroupLayouts.length, 1);
  assert.equal(device.calls.createBindGroupLayout.length, 2);
  assert.equal(device.calls.createPipelineLayout.length, 2);

  const defaultPl = cache.getOrCreateFallbackPipelineLayout('default');
  assert.equal(defaultPl.kind, 'GPUPipelineLayout');
  assert.equal(defaultPl.desc.bindGroupLayouts.length, 0);

  // Deterministic hash stability for fallback pipeline layouts
  const vs = device.createShaderModule({ label: 'fallback:fullscreen_triangle_cbv', code: 'vs' });
  assert.equal(
    hashDescriptorKey({ layout: cbvPl1, vertex: { module: vs, entryPoint: 'vs_main' } }),
    hashDescriptorKey({ vertex: { entryPoint: 'vs_main', module: vs }, layout: cbvPl2 })
  );

  // Graceful fallback to 'auto' when device.createPipelineLayout is unavailable
  const noPlDevice = createMockGpuDevice({ withPipelineLayout: false });
  const noPlCache = new WebGpuPipelineCache({ device: noPlDevice });
  assert.equal(noPlCache.getOrCreateFallbackPipelineLayout('fullscreen_triangle_cbv'), 'auto');

  // Graceful fallback to 'auto' + recorded sanitized error when device.createPipelineLayout throws
  const failingPlDevice = createMockGpuDevice({
    pipelineLayoutImpl: () => {
      const err = new Error('GPUValidationError: layout failed at C:\\Users\\dev\\secret\\p.wgsl');
      err.name = 'GPUValidationError';
      throw err;
    },
  });
  const failingPlCache = new WebGpuPipelineCache({ device: failingPlDevice, now: () => 500 });
  assert.equal(failingPlCache.getOrCreateFallbackPipelineLayout('fullscreen_triangle_cbv'), 'auto');
  assert.equal(failingPlCache.getStats().pipelineErrorCount, 1);
  const plErrors = failingPlCache.getPipelineErrors();
  assert.equal(plErrors.length, 1);
  assert.equal(plErrors[0].kind, 'pipeline-layout');
  assert.equal(plErrors[0].errorType, 'validation');
  assert.equal(plErrors[0].recovered, true);
  assert.doesNotMatch(plErrors[0].message, /Users\\dev/);
  assert.match(plErrors[0].message, /<path>/);
});

test('G36 normalizeRenderPipelineDescriptor and PipelineCache normalize fallback layouts without mutating caller descriptors', () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device, defaultColorFormat: 'rgba8unorm' });
  const router = new ShaderFallbackRouter({ defaultProfile: 'fullscreen_triangle_cbv' });

  // Resolve real G34 fallback shaders for Frame-1 DrawInstanced(3, 1)
  const resolvedVs = router.resolve('0x1111222233334444', 'vertex', {
    profile: 'fullscreen_triangle_cbv',
  });
  const resolvedFs = router.resolve('0x5555666677778888', 'fragment', {
    profile: 'fullscreen_triangle_cbv',
  });
  const vsMod = device.createShaderModule({
    label: `fallback:${resolvedVs.profile}:vs`,
    code: resolvedVs.wgsl,
  });
  const fsMod = device.createShaderModule({
    label: `fallback:${resolvedFs.profile}:fs`,
    code: resolvedFs.wgsl,
  });

  // Mismatched custom pipeline layout + 24B vertex buffer + missing fragment targets + invalid topology
  const mismatchedLayout = { kind: 'GPUPipelineLayout', id: 999, custom: true };
  const rawInput = {
    layout: mismatchedLayout,
    vertex: {
      module: vsMod,
      entryPoint: '',
      buffers: [
        {
          arrayStride: 24,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
        },
      ],
    },
    fragment: {
      module: fsMod,
      entryPoint: '',
      targets: [],
    },
    primitive: {
      topology: 'triangle-fan-invalid',
    },
  };
  const inputSnapshot = JSON.stringify({
    vertexBuffers: rawInput.vertex.buffers,
    targets: rawInput.fragment.targets,
    topology: rawInput.primitive.topology,
  });

  // 1. Default auto-layout normalization for fullscreen_triangle_cbv (detected from module label!)
  const rpAuto = cache.getOrCreateRenderPipeline(rawInput);
  assert.equal(rpAuto.desc.layout, 'auto');
  assert.deepEqual(rpAuto.desc.vertex.buffers, []);
  assert.equal(rpAuto.desc.vertex.entryPoint, 'vs_main');
  assert.equal(rpAuto.desc.fragment.entryPoint, 'fs_main');
  assert.deepEqual(rpAuto.desc.fragment.targets, [{ format: 'rgba8unorm' }]);
  assert.equal(rpAuto.desc.primitive.topology, 'triangle-list');

  // Verify caller's raw descriptor was NEVER mutated
  assert.equal(rawInput.layout, mismatchedLayout);
  assert.equal(
    JSON.stringify({
      vertexBuffers: rawInput.vertex.buffers,
      targets: rawInput.fragment.targets,
      topology: rawInput.primitive.topology,
    }),
    inputSnapshot
  );

  // 2. Explicit CBV layout mode binds cached group(0) binding(0) CBV GPUPipelineLayout
  const rpExplicit = cache.getOrCreateFallbackRenderPipeline(rawInput, {
    fallbackProfile: 'fullscreen_triangle_cbv',
    layoutMode: 'explicit',
  });
  assert.equal(rpExplicit.desc.layout.kind, 'GPUPipelineLayout');
  assert.equal(rpExplicit.desc.layout.desc.label, 'fallback-pl:fullscreen_triangle_cbv');
  assert.deepEqual(rpExplicit.desc.vertex.buffers, []);

  // Passing that exact compatible fallback GPUPipelineLayout back preserves it untouched
  const compatNorm = normalizeRenderPipelineDescriptor(
    {
      layout: rpExplicit.desc.layout,
      vertex: { module: vsMod, entryPoint: 'vs_main', buffers: [] },
      fragment: { module: fsMod, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
    },
    { fallbackProfile: 'fullscreen_triangle_cbv' }
  );
  assert.equal(compatNorm.layoutNormalized, false);
  assert.equal(compatNorm.descriptor.layout, rpExplicit.desc.layout);

  // 3. textured_quad_24b fallback normalizes to canonical 24B vertex layout + group(0) tex/sampler layout
  const quadVs = device.createShaderModule({
    label: 'fallback:textured_quad_24b:vs',
    code: router.resolve('0xaaaa000000000001', 'vertex', { profile: 'textured_quad_24b' }).wgsl,
  });
  const quadFs = device.createShaderModule({
    label: 'fallback:textured_quad_24b:fs',
    code: router.resolve('0xaaaa000000000002', 'fragment', { profile: 'textured_quad_24b' }).wgsl,
  });
  const rpQuad = cache.getOrCreateFallbackRenderPipeline(
    {
      // omitted layout + helper metadata stripped before WebGPU call
      fallbackProfile: 'textured_quad_24b',
      vertexStride: 24,
      vertex: { module: quadVs, entryPoint: 'vs_main' },
      fragment: { module: quadFs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
    },
    { layoutMode: 'explicit' }
  );
  assert.equal('fallbackProfile' in rpQuad.desc, false);
  assert.equal('vertexStride' in rpQuad.desc, false);
  assert.equal(rpQuad.desc.layout.desc.label, 'fallback-pl:textured_quad_24b');
  assert.deepEqual(rpQuad.desc.vertex.buffers, [
    {
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    },
  ]);

  // 4. Compute pipeline with missing layout normalizes to 'auto'
  const csMod = device.createShaderModule({ label: 'wgpk:compute:0x0000000000000099', code: 'cs' });
  const cpAuto = cache.getOrCreateComputePipeline({
    compute: { module: csMod, entryPoint: 'cs_main' },
  });
  assert.equal(cpAuto.desc.layout, 'auto');

  const stats = cache.getStats();
  assert.equal(stats.fallbackNormalizedCount, 4);
  assert.equal(stats.vertexLayoutNormalizedCount, 3);
  assert.equal(stats.layoutNormalizedCount, 4);
  assert.equal(stats.pipelineErrorCount, 0);
});

test('G36 auto-layout retry recovery works for sync, async, and deduplicated concurrent fallback pipelines', async () => {
  let syncCalls = 0;
  let asyncCalls = 0;

  const device = createMockGpuDevice({
    renderSyncImpl: (desc, id) => {
      syncCalls += 1;
      if (desc.layout !== 'auto') {
        const err = new Error('BindGroupLayout mismatch on group(0) binding(0)');
        err.name = 'GPUValidationError';
        throw err;
      }
      return { kind: 'GPURenderPipeline', id, recovered: true, desc };
    },
    renderAsyncImpl: async (desc, id) => {
      asyncCalls += 1;
      await new Promise((r) => setTimeout(r, 4));
      if (desc.layout !== 'auto') {
        const err = new Error('Async BindGroupLayout mismatch on group(0)');
        err.name = 'GPUValidationError';
        throw err;
      }
      return { kind: 'GPURenderPipeline', id, asyncRecovered: true, desc };
    },
  });

  const cache = new WebGpuPipelineCache({ device, now: () => 1234 });
  const vs = device.createShaderModule({ label: 'fallback:fullscreen_triangle_cbv:vs', code: 'vs' });
  const fs = device.createShaderModule({ label: 'fallback:fullscreen_triangle_cbv:fs', code: 'fs' });

  // Sync recovery: explicit layout fails -> automatically retried with layout: 'auto'
  const syncRecovered = cache.getOrCreateFallbackRenderPipeline(
    {
      vertex: { module: vs, entryPoint: 'vs_main' },
      fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
    },
    { fallbackProfile: 'fullscreen_triangle_cbv', layoutMode: 'explicit' }
  );
  assert.equal(syncRecovered.recovered, true);
  assert.equal(syncRecovered.desc.layout, 'auto');
  assert.equal(syncCalls, 2);

  // Async recovery with concurrent in-flight dedupe: both callers receive recovered pipeline
  const vs2 = device.createShaderModule({ label: 'fallback:textured_quad_24b:vs', code: 'vs2' });
  const fs2 = device.createShaderModule({ label: 'fallback:textured_quad_24b:fs', code: 'fs2' });
  const asyncDesc = {
    vertex: { module: vs2, entryPoint: 'vs_main' },
    fragment: { module: fs2, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  };
  const [a1, a2] = await Promise.all([
    cache.getOrCreateFallbackRenderPipelineAsync(asyncDesc, {
      fallbackProfile: 'textured_quad_24b',
      layoutMode: 'explicit',
    }),
    cache.getOrCreateFallbackRenderPipelineAsync(asyncDesc, {
      fallbackProfile: 'textured_quad_24b',
      layoutMode: 'explicit',
    }),
  ]);
  assert.equal(a1, a2);
  assert.equal(a1.asyncRecovered, true);
  assert.equal(a1.desc.layout, 'auto');
  assert.equal(asyncCalls, 2);

  const stats = cache.getStats();
  assert.equal(stats.fallbackRecoveryCount, 2);
  assert.equal(stats.inFlightDedupes, 1);
  assert.equal(stats.pipelineErrorCount, 2);

  const errors = cache.getPipelineErrors();
  assert.equal(errors.length, 2);
  assert.equal(errors[0].recovered, true);
  assert.equal(errors[0].fallbackProfile, 'fullscreen_triangle_cbv');
  assert.equal(errors[1].recovered, true);
  assert.equal(errors[1].fallbackProfile, 'textured_quad_24b');
});

test('G36 structured pipeline creation error tracking is sanitized, bounded, and isolates callbacks', async () => {
  let clock = 100;
  const callbackRecords = [];

  const device = createMockGpuDevice({
    renderSyncImpl: () => {
      const err = new Error(
        'Validation failed at C:\\Users\\alice\\local-data\\gta6\\sh.wgsl and "D:\\My Games\\GTA6\\x.rpf" and \\\\srv01\\share\\a.bin and file:///Users/alice/a.wgsl and /home/alice/dev/p.wgsl (rgba8unorm/bgra8unorm) hash 0x123456789abcdef0'
      );
      err.name = 'GPUValidationError';
      throw err;
    },
    computeSyncImpl: () => {
      const hexBlob = 'cd'.repeat(40);
      const spacedHex = Array.from({ length: 20 }, (_, i) => i.toString(16).padStart(2, '0')).join(' ');
      const byteList = Array.from({ length: 20 }, (_, i) => String(i)).join(', ');
      const b64Blob = 'QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFB';
      const err = new Error(
        `OOM:\nhex=${hexBlob}\tbytes=[${byteList}] dump=${spacedHex} uri=data:application/octet-stream;base64,${b64Blob} raw=${b64Blob}\x00\x07`
      );
      err.name = 'GPUOutOfMemoryError';
      throw err;
    },
    bglImpl: () => {
      const err = new Error(`Long failure: ${'binding_conflict '.repeat(50)}`);
      err.name = 'GPUInternalError';
      throw err;
    },
  });

  const cache = new WebGpuPipelineCache({
    device,
    maxErrorHistory: 3,
    now: () => {
      clock += 10;
      return clock;
    },
    onPipelineError: (rec) => {
      callbackRecords.push({ ...rec });
      rec.message = 'MUTATED_BY_CALLBACK';
      if (rec.seq === 2) {
        throw new Error('user callback exception must be isolated');
      }
    },
  });

  const vs = device.createShaderModule({ code: 'vs' });
  const cs = device.createShaderModule({ code: 'cs' });

  // 1. Render error (host paths stripped, WGPK hash + format token preserved)
  assert.throws(
    () => cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs' } }, 'C:\\Users\\alice\\key.pso'),
    /Validation failed/
  );

  // 2. Compute error (raw buffers/hex/base64/control chars stripped; throwing callback isolated)
  assert.throws(
    () => cache.getOrCreateComputePipeline({ layout: 'auto', compute: { module: cs, entryPoint: 'cs' } }),
    /OOM/
  );

  // 3. BindGroupLayout error (>512 chars capped cleanly)
  assert.throws(
    () => cache.getOrCreateBindGroupLayout({ entries: [{ binding: 0, visibility: 1, buffer: {} }] }),
    /Long failure/
  );

  // 4. Fourth error evicts seq 1 from ring (maxErrorHistory = 3)
  assert.throws(
    () => cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs2' } }),
    /Validation failed/
  );

  const stats = cache.getStats();
  assert.equal(stats.pipelineErrorCount, 4);
  assert.equal(stats.pipelineErrorHistorySize, 3);
  assert.equal(stats.maxErrorHistory, 3);

  // Check callback captured seq 1 before eviction and sanitized paths + key
  assert.equal(callbackRecords.length, 4);
  assert.equal(callbackRecords[0].seq, 1);
  assert.equal(callbackRecords[0].kind, 'render');
  assert.equal(callbackRecords[0].errorType, 'validation');
  assert.doesNotMatch(callbackRecords[0].key, /alice/i);
  assert.match(callbackRecords[0].key, /<path>/);
  assert.doesNotMatch(callbackRecords[0].message, /alice/i);
  assert.doesNotMatch(callbackRecords[0].message, /C:\\/);
  assert.doesNotMatch(callbackRecords[0].message, /\\\\srv01/);
  assert.doesNotMatch(callbackRecords[0].message, /file:\/\//);
  assert.doesNotMatch(callbackRecords[0].message, /\/home\//);
  assert.match(callbackRecords[0].message, /rgba8unorm\/bgra8unorm/);
  assert.match(callbackRecords[0].message, /0x123456789abcdef0/);

  // Ring retains seq [2, 3, 4]
  const history = cache.getPipelineErrors();
  assert.deepEqual(
    history.map((e) => e.seq),
    [2, 3, 4]
  );
  assert.equal(history[0].kind, 'compute');
  assert.equal(history[0].errorType, 'out-of-memory');
  assert.doesNotMatch(history[0].message, /cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd/);
  assert.doesNotMatch(history[0].message, /QUFBQUFBQUFBQUFB/);
  assert.doesNotMatch(history[0].message, /[\x00-\x1F\x7F]/);
  assert.match(history[0].message, /<buffer>/);

  assert.equal(history[1].kind, 'bind-group-layout');
  assert.equal(history[1].errorType, 'internal');
  assert.equal(history[1].message.length, MAX_PIPELINE_ERROR_MESSAGE_LENGTH);
  assert.ok(history[1].message.endsWith('...'));

  // JSON-serializable & defensive copy isolation
  assert.equal(JSON.parse(JSON.stringify(history)).length, 3);
  history[0].message = 'CORRUPTED';
  assert.notEqual(cache.getPipelineErrors()[0].message, 'CORRUPTED');

  // Direct sanitizePipelineErrorMessage helper checks
  assert.equal(sanitizePipelineErrorMessage(null), '');
  assert.equal(sanitizePipelineErrorMessage(undefined), '');

  // clearPipelineErrors clears ring but keeps cumulative count and monotonic seq
  cache.clearPipelineErrors();
  assert.deepEqual(cache.getPipelineErrors(), []);
  assert.equal(cache.getStats().pipelineErrorHistorySize, 0);
  assert.equal(cache.getStats().pipelineErrorCount, 4);

  assert.throws(
    () => cache.getOrCreateRenderPipeline({ layout: 'auto', vertex: { module: vs, entryPoint: 'vs3' } }),
    /Validation failed/
  );
  assert.equal(cache.getPipelineErrors()[0].seq, 5);
  assert.equal(cache.getStats().pipelineErrorCount, 5);

  // clear({ resetStats: true }) resets error counts, seq, and ring
  cache.clear({ resetStats: true });
  assert.equal(cache.getStats().pipelineErrorCount, 0);
  assert.equal(cache.getStats().pipelineErrorHistorySize, 0);
  assert.deepEqual(cache.getPipelineErrors(), []);
});

test('G36 prewarmRenderPipelines deduplicates normalized fallback descriptors with subsequent cache lookups', async () => {
  const device = createMockGpuDevice();
  const cache = new WebGpuPipelineCache({ device });
  const vs = device.createShaderModule({ label: 'fallback:fullscreen_triangle_cbv:vs', code: 'vs' });
  const fs = device.createShaderModule({ label: 'fallback:fullscreen_triangle_cbv:fs', code: 'fs' });

  const summary = await cache.prewarmRenderPipelines([
    {
      descriptor: {
        // missing layout + 24B buffers -> normalizes to layout: 'auto', buffers: []
        vertex: { module: vs, entryPoint: 'vs_main', buffers: [{ arrayStride: 24 }] },
        fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
      },
      options: { fallbackProfile: 'fullscreen_triangle_cbv' },
    },
    {
      descriptor: {
        layout: 'auto',
        vertex: { module: vs, entryPoint: 'vs_main', buffers: [] },
        fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
      },
      options: { fallbackProfile: 'fullscreen_triangle_cbv' },
    },
  ]);

  assert.equal(summary.total, 2);
  assert.equal(summary.pipelines[0], summary.pipelines[1]);
  assert.equal(device.calls.createRenderPipelineAsync.length, 1);

  // Subsequent sync lookup hits the prewarmed normalized entry directly
  const hit = cache.getOrCreateFallbackRenderPipeline(
    {
      vertex: { module: vs, entryPoint: 'vs_main', buffers: [{ arrayStride: 24 }] },
      fragment: { module: fs, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
    },
    { fallbackProfile: 'fullscreen_triangle_cbv' }
  );
  assert.equal(hit, summary.pipelines[0]);
  assert.equal(device.calls.createRenderPipeline.length, 0);
});
