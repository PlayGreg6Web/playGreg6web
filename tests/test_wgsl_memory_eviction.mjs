import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import {
  SHADER_STAGE,
  WGPK_HEADER_BYTES,
  WGPK_ENTRY_BYTES,
  DEFAULT_MAX_COMPILED_MODULES,
  fnv1a64,
  formatHashHex,
  buildWgslPackBuffer,
  parseWgslPack,
  loadIndexFromUrl,
  WgslPackCache,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const pipelineCacheMod = require('../web/workers/pipeline_cache.js');
const reflectionLoaderMod = require('../web/workers/reflection_loader.js');

const {
  DEFAULT_MAX_RETAINED_WGSL_SOURCE_BYTES,
  getRegisteredShaderModuleSource,
  getRegisteredShaderModuleSummary,
  releaseShaderModuleSource,
  analyzeInterstageCompatibility,
  WebGpuPipelineCache,
} = pipelineCacheMod;
const { ReflectionRegistry } = reflectionLoaderMod;

function createMockGpuDevice(options = {}) {
  const calls = {
    createShaderModule: [],
    createRenderPipeline: [],
    createRenderPipelineAsync: [],
  };
  let nextModId = 1;
  let nextPipeId = 1;

  const device = {
    createShaderModule(desc) {
      if (options.failNextShaderCompile && options.failNextShaderCompile()) {
        throw new Error('Mock shader compilation error');
      }
      const mod = {
        kind: 'GPUShaderModule',
        id: nextModId++,
        label: desc.label || `mod-${nextModId}`,
        // Intentionally do NOT attach `.code` when options.opaqueModules is true so
        // getRegisteredShaderModuleSource() cannot silently read mock.code after eviction.
        ...(options.opaqueModules ? {} : { code: desc.code }),
        getCompilationInfo: async () => ({ messages: [] }),
      };
      calls.createShaderModule.push({ desc, mod });
      return mod;
    },
    createRenderPipeline(desc) {
      calls.createRenderPipeline.push(desc);
      return { kind: 'GPURenderPipeline', id: nextPipeId++, descriptor: desc };
    },
    async createRenderPipelineAsync(desc) {
      calls.createRenderPipelineAsync.push(desc);
      return { kind: 'GPURenderPipeline', id: nextPipeId++, descriptor: desc };
    },
  };

  return { device, calls };
}

function createRangeMockFetch(filesByUrl) {
  const requests = [];
  const fetchImpl = async (url, init = {}) => {
    const raw = filesByUrl[url];
    if (!raw) {
      return { ok: false, status: 404, statusText: 'Not Found', headers: new Headers() };
    }
    const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
    const headersIn = init.headers || {};
    const rangeHeader =
      typeof headersIn.get === 'function'
        ? headersIn.get('Range') || headersIn.get('range')
        : headersIn.Range || headersIn.range || null;

    if (rangeHeader) {
      const m = /^bytes=(\d+)-(\d+)$/i.exec(String(rangeHeader).trim());
      assert.ok(m, `unexpected Range header: ${rangeHeader}`);
      const start = Number(m[1]);
      const end = Math.min(Number(m[2]), bytes.byteLength - 1);
      const slice = bytes.slice(start, end + 1);
      requests.push({ url, range: rangeHeader, start, end, bytesSent: slice.byteLength });
      return {
        ok: true,
        status: 206,
        statusText: 'Partial Content',
        headers: new Headers({
          'Content-Range': `bytes ${start}-${end}/${bytes.byteLength}`,
          'Content-Length': String(slice.byteLength),
          'Accept-Ranges': 'bytes',
        }),
        async arrayBuffer() {
          return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
        },
      };
    }

    requests.push({ url, range: null, bytesSent: bytes.byteLength });
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: new Headers({
        'Content-Length': String(bytes.byteLength),
        'Accept-Ranges': 'bytes',
      }),
      async arrayBuffer() {
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      },
    };
  };
  return { fetch: fetchImpl, requests };
}

const VS_3_LOCS = [
  'struct VsIn {',
  '    @location(0) attr_0_: vec4<f32>,',
  '}',
  'struct VsOut {',
  '    @builtin(position) position: vec4<f32>,',
  '    @location(0) param_0_: vec4<f32>,',
  '    @location(1) param_1_: vec4<f32>,',
  '    @location(2) param_2_: vec4<f32>,',
  '}',
  '@vertex',
  'fn vs_main(vs_in: VsIn) -> VsOut {',
  '    var vs_out: VsOut;',
  '    vs_out = VsOut();',
  '    vs_out.position = vs_in.attr_0_;',
  '    vs_out.param_0_ = vec4<f32>(1.0, 0.0, 0.0, 1.0);',
  '    vs_out.param_1_ = vec4<f32>(0.0, 1.0, 0.0, 1.0);',
  '    vs_out.param_2_ = vec4<f32>(0.0, 0.0, 1.0, 1.0);',
  '    return vs_out;',
  '}',
].join('\n');

const PS_MATCHING = [
  'struct FragIn {',
  '    @location(0) attr_0_: vec4<f32>,',
  '    @location(1) attr_1_: vec4<f32>,',
  '}',
  'struct PsOut {',
  '    @location(0) mrt_0_: vec4<f32>,',
  '}',
  '@fragment',
  'fn fs_main(frag_in: FragIn) -> PsOut {',
  '    var ps_out: PsOut;',
  '    ps_out.mrt_0_ = frag_in.attr_0_ + frag_in.attr_1_;',
  '    return ps_out;',
  '}',
].join('\n');

const PS_FLAT_LOC2 = [
  'struct FragIn {',
  '    @location(0) attr_0_: vec4<f32>,',
  '    @location(2) @interpolate(flat) attr_2_: vec4<f32>,',
  '}',
  'struct PsOut {',
  '    @location(0) mrt_0_: vec4<f32>,',
  '}',
  '@fragment',
  'fn fs_main(frag_in: FragIn) -> PsOut {',
  '    var ps_out: PsOut;',
  '    ps_out.mrt_0_ = frag_in.attr_0_ + frag_in.attr_2_;',
  '    return ps_out;',
  '}',
].join('\n');

const PS_FSSM_REMAP_FLAT = [
  'struct FragIn {',
  '    @location(0) @interpolate(flat) attr_0_: vec4<f32>,',
  '}',
  'struct PsOut {',
  '    @location(0) mrt_0_: vec4<f32>,',
  '}',
  '@fragment',
  'fn fs_main(frag_in: FragIn) -> PsOut {',
  '    var ps_out: PsOut;',
  '    ps_out.mrt_0_ = frag_in.attr_0_;',
  '    return ps_out;',
  '}',
].join('\n');

test('#2162 wgsl_pack_loader: post-compile WGSL text eviction drops retained string bytes to 0 and lazily re-decodes on getWgsl/getEntry', () => {
  const enc = new TextEncoder();
  const hVs = fnv1a64('evict-buf-vs');
  const hFs = fnv1a64('evict-buf-fs');
  const vsLen = enc.encode(VS_3_LOCS).byteLength;
  const fsLen = enc.encode(PS_FLAT_LOC2).byteLength;

  const packBuf = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: VS_3_LOCS },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: PS_FLAT_LOC2 },
  ]);

  const cache = new WgslPackCache();
  const pack = cache.loadPackBuffer(packBuf, 'boot.wgsl.pack');
  const gpu = createMockGpuDevice();

  // Default lazyDecode=true -> 0 retained decoded WGSL text bytes before compile.
  assert.equal(cache.getStats().releaseSourceAfterCompile, 'auto');
  assert.equal(cache.getStats().wgslTextBytesRetained, 0);
  assert.equal(cache.getStats().wgslTextBytesReleased, 0);
  assert.equal(cache.hasCachedWgslText(hVs), false);

  // Compile both shaders: decoded WGSL text is automatically released immediately after compile!
  const modVs = cache.getOrCreateShaderModule(gpu.device, hVs, 'gta6');
  const modFs = cache.compileShaderModule(gpu.device, hFs, 'gta6');
  assert.equal(gpu.calls.createShaderModule.length, 2);
  assert.equal(modVs.code, VS_3_LOCS);
  assert.equal(modFs.code, PS_FLAT_LOC2);

  const afterCompile = cache.getStats();
  assert.equal(afterCompile.wgslTextBytesRetained, 0);
  assert.equal(afterCompile.wgslTextEntriesRetained, 0);
  assert.equal(afterCompile.wgslTextBytesReleased, vsLen + fsLen);
  assert.equal(afterCompile.wgslTextReleases, 2);
  assert.equal(afterCompile.wgslTextLazyRedecodes, 0);
  assert.equal(cache.hasCachedWgslText(hVs), false);
  assert.equal(pack.hasCachedWgslText(hFs), false);

  // Repeat getOrCreateShaderModule hits cached GPUShaderModule with 0 compiles and 0 re-decodes!
  assert.equal(cache.getOrCreateShaderModule(gpu.device, hVs), modVs);
  assert.equal(cache.getOrCreateShaderModule(gpu.device, hFs), modFs);
  assert.equal(gpu.calls.createShaderModule.length, 2);
  assert.equal(cache.getStats().wgslTextLazyRedecodes, 0);
  assert.equal(cache.getStats().wgslTextBytesRetained, 0);

  // Explicit getWgsl / getEntry after release lazily re-decodes cleanly from backing pack bytes!
  assert.equal(cache.getWgsl(hVs), VS_3_LOCS);
  assert.equal(cache.getEntry(hFs).wgsl, PS_FLAT_LOC2);
  const afterRedecode = cache.getStats();
  assert.equal(afterRedecode.wgslTextLazyRedecodes, 2);
  assert.equal(afterRedecode.wgslTextBytesRetained, vsLen + fsLen);
  assert.equal(cache.hasCachedWgslText(hVs), true);

  // Compact binary index metadata remains intact.
  const entryMeta = pack.getEntry(hVs);
  assert.equal(entryMeta.hash, hVs);
  assert.equal(entryMeta.stage, 'vertex');
  assert.equal(entryMeta.length, vsLen);
  assert.ok(entryMeta.offset >= WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES);

  // Explicit releaseWgslText + releaseAllWgslText drop retained bytes back to 0.
  assert.equal(cache.releaseWgslText(hVs), true);
  assert.equal(cache.releaseWgslText(hVs), false); // already released
  assert.equal(cache.getStats().wgslTextBytesRetained, fsLen);
  const relAll = cache.releaseAllWgslText();
  assert.deepEqual(relAll, { releasedCount: 1, bytesReleased: fsLen });
  assert.equal(cache.getStats().wgslTextBytesRetained, 0);
  assert.equal(cache.getStats().wgslTextBytesReleased, 2 * (vsLen + fsLen));

  // Eager parseWgslPack({ lazyDecode: false }) also supports releaseAllWgslText + lazy re-decode!
  const eagerPack = parseWgslPack(packBuf, { lazyDecode: false });
  assert.equal(eagerPack.stats().wgslTextBytesRetained, vsLen + fsLen);
  assert.equal(eagerPack.stats().decodedCount, 2);
  assert.deepEqual(eagerPack.releaseAllWgslText(), { releasedCount: 2, bytesReleased: vsLen + fsLen });
  assert.equal(eagerPack.stats().wgslTextBytesRetained, 0);
  assert.equal(eagerPack.getWgsl(hVs), VS_3_LOCS);
  assert.equal(eagerPack.stats().wgslTextLazyRedecodes, 1);

  // Opt-out releaseSourceAfterCompile: false retains decoded WGSL text until manually released.
  const retainCache = new WgslPackCache({ releaseSourceAfterCompile: false });
  retainCache.loadPackBuffer(packBuf, 'retain.wgsl.pack');
  retainCache.getOrCreateShaderModule(gpu.device, hVs);
  assert.equal(retainCache.getStats().wgslTextBytesRetained, vsLen);
  assert.equal(retainCache.getStats().wgslTextBytesReleased, 0);
  assert.equal(retainCache.releaseWgslText(hVs), true);
  assert.equal(retainCache.getStats().wgslTextBytesRetained, 0);
});

test('#2162 wgsl_pack_loader: ranged post-compile eviction respects pins, lazily re-fetches, and enforces bounded compiled module LRU', async () => {
  const enc = new TextEncoder();
  const entries = [];
  for (let i = 0; i < 6; i += 1) {
    const wgsl = `// shader-${i} ${'X'.repeat(120)}\n@compute @workgroup_size(1) fn cs_${i}() {}`;
    entries.push({
      hash: fnv1a64(`ranged-evict-${i}`),
      stage: SHADER_STAGE.COMPUTE,
      wgsl,
      len: enc.encode(wgsl).byteLength,
    });
  }
  const packBuf = buildWgslPackBuffer(entries);
  const tableEnd = WGPK_HEADER_BYTES + entries.length * WGPK_ENTRY_BYTES;
  const mock = createRangeMockFetch({ '/packs/ranged-evict.wgsl.pack': packBuf });

  assert.equal(DEFAULT_MAX_COMPILED_MODULES, 4096);
  const cache = new WgslPackCache({
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    releaseSourceAfterCompile: true,
    maxCompiledModules: 3,
  });
  const pack = await cache.loadIndexFromUrl('/packs/ranged-evict.wgsl.pack');
  const gpu = createMockGpuDevice();

  // Pin entry 0 so post-compile auto-release keeps it in pinned cache while unpinned entries drop.
  cache.pinHashes([entries[0].hash]);

  const mod0 = await cache.getOrCreateShaderModuleAsync(gpu.device, entries[0].hash);
  const mod1 = await cache.getOrCreateShaderModuleAsync(gpu.device, entries[1].hash);
  const mod2 = await cache.getOrCreateShaderModuleAsync(gpu.device, entries[2].hash);

  assert.equal(cache.hasCachedWgslText(entries[0].hash), true, 'pinned shader stays resident');
  assert.equal(cache.hasCachedWgslText(entries[1].hash), false, 'unpinned compiled shader released');
  assert.equal(cache.hasCachedWgslText(entries[2].hash), false, 'unpinned compiled shader released');
  assert.equal(cache.getStats().wgslTextBytesRetained, entries[0].len);
  assert.equal(cache.getStats().wgslTextBytesReleased, entries[1].len + entries[2].len);

  // Lazy re-fetch of released entry 1 works seamlessly and increments wgslTextLazyRedecodes.
  const refetched1 = await cache.getWgslAsync(entries[1].hash);
  assert.equal(refetched1, entries[1].wgsl);
  assert.equal(pack.getStats().wgslTextLazyRedecodes, 1);
  assert.equal(cache.getStats().wgslTextLazyRedecodes, 1);

  // Explicit releaseAllWgslText({ includePinned: true }) drops pinned text too.
  const relAll = cache.releaseAllWgslText({ includePinned: true });
  assert.equal(relAll.releasedCount, 2); // entry 0 (pinned) + entry 1 (re-fetched)
  assert.equal(cache.getStats().wgslTextBytesRetained, 0);

  // Touch mod0 so mod1 becomes the oldest compiled module, then compile entry 3 -> evicts mod1!
  assert.equal(await cache.getOrCreateShaderModuleAsync(gpu.device, entries[0].hash), mod0);
  await cache.getOrCreateShaderModuleAsync(gpu.device, entries[3].hash);
  assert.equal(cache.getStats().cachedModules, 3);
  assert.equal(cache.getStats().moduleLruEvictions, 1);
  assert.equal(cache.getStats().evictions, 1);
  // mod0 and mod2 are still cached (0 extra compiles); mod1 was evicted so recompiles cleanly.
  assert.equal(await cache.getOrCreateShaderModuleAsync(gpu.device, entries[0].hash), mod0);
  assert.equal(await cache.getOrCreateShaderModuleAsync(gpu.device, entries[2].hash), mod2);
  const compilesBeforeMod1 = cache.getStats().compiles;
  const mod1Recompiled = await cache.getOrCreateShaderModuleAsync(gpu.device, entries[1].hash);
  assert.notEqual(mod1Recompiled, mod1);
  assert.equal(cache.getStats().compiles, compilesBeforeMod1 + 1);
});

test('#2162 pipeline_cache: bounded raw WGSL source LRU stays strictly under cap over 250 shaders while preserving compact interstage summaries', () => {
  assert.equal(DEFAULT_MAX_RETAINED_WGSL_SOURCE_BYTES, 8 * 1024 * 1024);

  const CAP_BYTES = 32 * 1024; // 32 KiB tight cap
  const { device, calls } = createMockGpuDevice({ opaqueModules: true });
  const cache = new WebGpuPipelineCache({
    device,
    maxRetainedWgslSourceBytes: CAP_BYTES,
  });

  const padComment = `// ${'P'.repeat(4000)}\n`; // ~4 KiB per shader -> 250 shaders = ~1 MiB raw WGSL
  const vsModules = [];
  const psModules = [];

  for (let i = 0; i < 125; i += 1) {
    const vsHex = `0x${(0x10000n + BigInt(i)).toString(16).padStart(16, '0')}`;
    const psHex = `0x${(0x20000n + BigInt(i)).toString(16).padStart(16, '0')}`;
    const vsMod = { label: `wgpk:vertex:${vsHex}`, hashHex: vsHex };
    const psMod = { label: `wgpk:fragment:${psHex}`, hashHex: psHex };

    cache.registerShaderModule(vsMod, vsHex, `${padComment}// vs-${i}\n${VS_3_LOCS}`);
    cache.registerShaderModule(psMod, psHex, `${padComment}// ps-${i}\n${PS_MATCHING}`);

    const st = cache.getStats();
    assert.ok(
      st.wgslSourceBytesRetained <= CAP_BYTES,
      `retained bytes ${st.wgslSourceBytesRetained} exceeded cap ${CAP_BYTES} at iter ${i}`
    );

    vsModules.push(vsMod);
    psModules.push(psMod);
  }

  const midStats = cache.getStats();
  assert.ok(midStats.wgslSourceBytesRetained <= CAP_BYTES);
  assert.ok(midStats.wgslSourceEvictions >= 230, `expected >= 230 evictions, got ${midStats.wgslSourceEvictions}`);
  assert.ok(midStats.wgslSourceBytesReleased > 800 * 1024);

  // Oldest shader's raw WGSL source was evicted from BOTH cache LRU and global WeakMap!
  assert.equal(getRegisteredShaderModuleSource(vsModules[0]), null);
  assert.equal(getRegisteredShaderModuleSource(psModules[0]), null);

  // Compact inter-stage interface summaries are STILL present for evicted shaders!
  const vs0Summary = cache.getShaderModuleSummary(vsModules[0]);
  const ps0Summary = getRegisteredShaderModuleSummary(psModules[0]);
  assert.ok(vs0Summary);
  assert.deepEqual(vs0Summary.vsOutputLocations, [0, 1, 2]);
  assert.equal(vs0Summary.hasFlatNeeded, false);
  assert.ok(ps0Summary);
  assert.deepEqual(ps0Summary.psInputLocations, [0, 1]);
  assert.equal(ps0Summary.psLocationTypes[0], 'vec4<f32>');
  assert.equal(ps0Summary.psLocationTypes[1], 'vec4<f32>');
  assert.equal(ps0Summary.hasFlatNeeded, false);

  // Linking evicted compatible VS[0] + PS[0] succeeds with ZERO lazy WGSL fetches and ZERO extra shader compiles!
  const pipe = cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsModules[0], entryPoint: 'vs_main' },
    fragment: { module: psModules[0], entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  assert.equal(pipe.kind, 'GPURenderPipeline');
  assert.equal(calls.createShaderModule.length, 0);
  assert.equal(cache.getStats().wgslSourceLazyFetches, 0);

  // Explicit releaseAllShaderModuleSources drops retained bytes to 0; clear({ resetStats: true }) resets counters.
  const rel = cache.releaseAllShaderModuleSources();
  assert.ok(rel.releasedCount > 0 && rel.bytesReleased > 0);
  assert.equal(cache.getStats().wgslSourceBytesRetained, 0);

  cache.clear({ resetStats: true });
  assert.equal(cache.getStats().wgslSourceBytesRetained, 0);
  assert.equal(cache.getStats().wgslSourceBytesReleased, 0);
  assert.equal(cache.getStats().wgslSourceEvictions, 0);
});

test('#2162 pipeline_cache: interstage patching (@interpolate(flat) + #2113 FSSM @location remap) works seamlessly from both cached source and lazy re-decode after eviction', async () => {
  const vsHash = 0x5fe7a88d9cb680b3n;
  const psHash = 0x56214b8d3b10a108n;
  const vsHex = formatHashHex(vsHash);
  const psHex = formatHashHex(psHash);

  const registry = new ReflectionRegistry();
  registry.loadStageReflection({
    schema: 'playgta6-stage-reflection',
    version: 1,
    stages: {
      [vsHex]: {
        stage: 'vertex',
        bindings: [],
        vs_outputs: [
          { semantic_reg: 1, location: 0 },
          { semantic_reg: 2, location: 1 },
          { semantic_reg: 3, location: 2 },
        ],
      },
      [psHex]: {
        stage: 'fragment',
        bindings: [],
        // PS reads semantic_reg=2 at @location(0) with @interpolate(flat) -> must remap PS @0->@1 AND patch VS @1 to @interpolate(flat)!
        ps_inputs: [{ semantic_reg: 2, location: 0, mask: 3 }],
      },
    },
  });

  // --- Subtest A: Cached source + releaseWgslSourceAfterLink: true ---
  {
    const { device, calls } = createMockGpuDevice({ opaqueModules: true });
    const cache = new WebGpuPipelineCache({
      device,
      reflectionRegistry: registry,
      releaseWgslSourceAfterLink: true,
    });
    const vsMod = { label: `wgpk:vertex:${vsHex}`, hashHex: vsHex };
    const psMod = { label: `wgpk:fragment:${psHex}`, hashHex: psHex };
    cache.registerShaderModule(vsMod, vsHex, VS_3_LOCS);
    cache.registerShaderModule(psMod, psHex, PS_FSSM_REMAP_FLAT);

    assert.ok(cache.getStats().wgslSourceBytesRetained > 0);
    assert.equal(cache.getShaderModuleSummary(psMod).hasFlatNeeded, true);

    const pipe = cache.getOrCreateRenderPipeline({
      layout: 'auto',
      vertex: { module: vsMod, entryPoint: 'vs_main' },
      fragment: { module: psMod, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
    });
    assert.equal(pipe.kind, 'GPURenderPipeline');
    assert.equal(calls.createShaderModule.length, 2); // 1 remapped PS + 1 flat-patched VS
    const remappedPsCode = calls.createShaderModule[0].desc.code;
    const patchedVsCode = calls.createShaderModule[1].desc.code;
    assert.match(remappedPsCode, /@location\(1\)\s+@interpolate\(flat\)\s+attr_0_/);
    assert.match(patchedVsCode, /@location\(1\)\s+@interpolate\(flat\)\s+param_1_/);
    assert.equal(analyzeInterstageCompatibility(patchedVsCode, remappedPsCode).compatible, true);

    // Raw WGSL sources were automatically released after link!
    assert.equal(cache.getStats().wgslSourceBytesRetained, 0);
    assert.ok(cache.getStats().wgslSourceBytesReleased > 0);
    assert.equal(cache.getStats().wgslSourceLazyFetches, 0);
    assert.equal(getRegisteredShaderModuleSource(vsMod), null);
    assert.equal(getRegisteredShaderModuleSource(psMod), null);
    assert.equal(cache.getInterstageStats().locationRemapFixes, 1);
    assert.equal(cache.getInterstageStats().flatFixes, 1);
  }

  // --- Subtest B: End-to-end zero-retention integration with WgslPackCache lazy re-decode ---
  {
    const packBuf = buildWgslPackBuffer([
      { hash: vsHash, stage: SHADER_STAGE.VERTEX, wgsl: VS_3_LOCS },
      { hash: psHash, stage: SHADER_STAGE.FRAGMENT, wgsl: PS_FSSM_REMAP_FLAT },
    ]);
    const packCache = new WgslPackCache(); // releaseSourceAfterCompile: 'auto' (enabled for buffer packs)
    packCache.loadPackBuffer(packBuf, 'fssm-evict.wgsl.pack');

    const { device, calls } = createMockGpuDevice({ opaqueModules: true });
    // Compile via WgslPackCache -> decoded WGSL text immediately released in WgslPackCache!
    const vsMod = packCache.getOrCreateShaderModule(device, vsHash, 'wgpk');
    const psMod = packCache.getOrCreateShaderModule(device, psHash, 'wgpk');
    assert.equal(packCache.getStats().wgslTextBytesRetained, 0);
    assert.equal(packCache.getStats().wgslTextLazyRedecodes, 0);

    // Register on WebGpuPipelineCache with maxRetainedWgslSourceBytes: 0 (immediate eviction after summary)
    // and wire packCache as the lazy wgslSourceProvider!
    const pipelineCache = new WebGpuPipelineCache({
      device,
      reflectionRegistry: registry,
      maxRetainedWgslSourceBytes: 0,
      wgslSourceProvider: packCache,
    });
    // Temporarily decode to register summaries, then immediately release in packCache too -> 0 retained bytes everywhere!
    pipelineCache.registerShaderModule(vsMod, vsHex, packCache.getWgsl(vsHash));
    pipelineCache.registerShaderModule(psMod, psHex, packCache.getWgsl(psHash));
    packCache.releaseAllWgslText();

    assert.equal(packCache.getStats().wgslTextBytesRetained, 0);
    assert.equal(pipelineCache.getStats().wgslSourceBytesRetained, 0);
    assert.equal(pipelineCache.getStats().wgslSourceEvictions, 2);
    assert.equal(getRegisteredShaderModuleSource(vsMod), null);
    assert.equal(getRegisteredShaderModuleSource(psMod), null);
    const redecodesBeforeLink = packCache.getStats().wgslTextLazyRedecodes;

    // Create pipeline asynchronously -> pipelineCache uses summaries, detects BOTH FSSM @location remap
    // AND @interpolate(flat) mismatch, lazily re-decodes both WGSL sources from binary WGPK buffer,
    // patches & compiles cleanly, and keeps pipelineCache retained WGSL bytes at 0!
    const pipe = await pipelineCache.getOrCreateRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: vsMod, entryPoint: 'vs_main' },
      fragment: { module: psMod, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
    });
    assert.equal(pipe.kind, 'GPURenderPipeline');
    assert.notEqual(pipe.descriptor.vertex.module, vsMod);
    assert.notEqual(pipe.descriptor.fragment.module, psMod);

    // 2 original compiles + 1 remapped PS + 1 flat-patched VS = 4 total shader compiles
    assert.equal(calls.createShaderModule.length, 4);
    const remappedPsCode = calls.createShaderModule[2].desc.code;
    const patchedVsCode = calls.createShaderModule[3].desc.code;
    assert.match(remappedPsCode, /@location\(1\)\s+@interpolate\(flat\)\s+attr_0_/);
    assert.match(patchedVsCode, /@location\(1\)\s+@interpolate\(flat\)\s+param_1_/);
    assert.equal(analyzeInterstageCompatibility(patchedVsCode, remappedPsCode).compatible, true);

    assert.equal(pipelineCache.getStats().wgslSourceBytesRetained, 0);
    assert.equal(pipelineCache.getStats().wgslSourceLazyFetches, 2);
    assert.equal(packCache.getStats().wgslTextLazyRedecodes, redecodesBeforeLink + 2);
    assert.equal(pipelineCache.getStats().interstagePatchedPipelines, 1);
    assert.equal(pipelineCache.getStats().interstageLocationRemapFixes, 1);
    assert.equal(pipelineCache.getStats().interstageFlatFixes, 1);

    // Second pipeline with different RT format hits patched VS module cache cleanly without re-compiling!
    const pipe2 = pipelineCache.getOrCreateRenderPipeline({
      layout: 'auto',
      vertex: { module: vsMod, entryPoint: 'vs_main' },
      fragment: { module: psMod, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
    });
    assert.equal(calls.createShaderModule.length, 4, 'both patched PS and VS modules reused from LRU');
    assert.equal(pipe2.descriptor.vertex.module, pipe.descriptor.vertex.module);
    assert.equal(pipe2.descriptor.fragment.module, pipe.descriptor.fragment.module);
    assert.equal(pipelineCache.getInterstageStats().moduleCacheHits, 2);
  }
});

test('#2162 pipeline_cache: custom lazy wgslSourceProvider callback handles flat-only patching after explicit releaseShaderModuleSource', () => {
  const { device, calls } = createMockGpuDevice({ opaqueModules: true });
  const fetchedStages = [];
  const vsMod = { label: 'wgpk:vertex:0x000000000000aa01', hashHex: '0x000000000000aa01' };
  const psMod = { label: 'wgpk:fragment:0x000000000000aa02', hashHex: '0x000000000000aa02' };

  const cache = new WebGpuPipelineCache({
    device,
    wgslSourceProvider(mod, ctx) {
      fetchedStages.push({ stage: ctx.stage, hash: ctx.hash });
      if (mod === vsMod) return VS_3_LOCS;
      if (mod === psMod) return PS_FLAT_LOC2;
      return null;
    },
  });

  cache.registerShaderModule(vsMod, '0x000000000000aa01', VS_3_LOCS);
  cache.registerShaderModule(psMod, '0x000000000000aa02', PS_FLAT_LOC2);

  // Release both raw WGSL strings explicitly.
  assert.equal(cache.releaseShaderModuleSource(vsMod), true);
  assert.equal(cache.releaseShaderModuleSource(psMod), true);
  assert.equal(cache.releaseShaderModuleSource(psMod), false);
  assert.equal(releaseShaderModuleSource(vsMod), false);
  assert.equal(cache.getStats().wgslSourceBytesRetained, 0);

  // Only VS needs rewriting (@interpolate(flat) on @location(2)) -> PS raw WGSL is NEVER even fetched!
  const pipe = cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: psMod, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  assert.equal(pipe.kind, 'GPURenderPipeline');
  assert.equal(calls.createShaderModule.length, 1);
  assert.match(calls.createShaderModule[0].desc.code, /@location\(2\)\s+@interpolate\(flat\)\s+param_2_/);
  assert.deepEqual(fetchedStages, [{ stage: 'vertex', hash: '0x000000000000aa01' }]);
  assert.equal(cache.getStats().wgslSourceLazyFetches, 1);
  assert.equal(cache.getStats().wgslSourceBytesRetained, 0);

  // Pairing with a different RT format hits _interstagePatchedModules -> ZERO lazy fetches!
  cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: psMod, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
  });
  assert.equal(fetchedStages.length, 1);
  assert.equal(cache.getStats().wgslSourceLazyFetches, 1);
  assert.equal(calls.createShaderModule.length, 1);
});
