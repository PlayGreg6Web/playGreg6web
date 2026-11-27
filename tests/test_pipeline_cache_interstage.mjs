import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const pipelineCacheMod = require('../web/workers/pipeline_cache.js');
const routerMod = require('../web/workers/shader_fallback_router.js');
const reflectionLoaderMod = require('../web/workers/reflection_loader.js');

const {
  DEFAULT_MAX_INTERSTAGE_MODULES,
  DEFAULT_MAX_INTERSTAGE_INTERFACE_CACHE,
  registerShaderModuleSource,
  getRegisteredShaderModuleSource,
  registerShaderModuleReflection,
  parseInterstageInterface,
  computeInterstageLocationRemaps,
  remapPsInputLocationsWgsl,
  analyzeInterstageCompatibility,
  patchInterstageWgsl,
  WebGpuPipelineCache,
} = pipelineCacheMod;
const { ReflectionRegistry } = reflectionLoaderMod;

function assertBalancedWgsl(code, label = 'wgsl') {
  const stack = [];
  const pairs = { '(': ')', '{': '}', '[': ']' };
  for (let i = 0; i < code.length; i += 1) {
    const ch = code[i];
    if (pairs[ch]) {
      stack.push(pairs[ch]);
    } else if (ch === ')' || ch === '}' || ch === ']') {
      assert.equal(stack.pop(), ch, `${label}: unmatched closing ${ch} at offset ${i}`);
    }
  }
  assert.equal(stack.length, 0, `${label}: unclosed delimiters remaining: ${stack.join('')}`);
}

function createMockWebGpuDevice(options = {}) {
  const calls = {
    createShaderModule: [],
    createRenderPipeline: [],
    createRenderPipelineAsync: [],
  };
  let nextModId = 1;
  let nextPipeId = 1;

  const device = {
    createShaderModule(desc) {
      if (options.failCreateShaderModule) {
        throw new Error('Mock createShaderModule validation error');
      }
      const mod = {
        kind: 'GPUShaderModule',
        id: nextModId++,
        label: desc.label || `mod-${nextModId}`,
        code: desc.code,
        getCompilationInfo: async () => ({ messages: [] }),
      };
      calls.createShaderModule.push({ desc, mod });
      return mod;
    },
    createRenderPipeline(desc) {
      calls.createRenderPipeline.push(desc);
      if (typeof options.validateRenderPipeline === 'function') {
        options.validateRenderPipeline(desc);
      }
      return {
        kind: 'GPURenderPipeline',
        id: nextPipeId++,
        descriptor: desc,
      };
    },
  };

  if (options.withAsync !== false) {
    device.createRenderPipelineAsync = async (desc) => {
      calls.createRenderPipelineAsync.push(desc);
      if (typeof options.validateRenderPipeline === 'function') {
        options.validateRenderPipeline(desc);
      }
      return {
        kind: 'GPURenderPipeline',
        id: nextPipeId++,
        descriptor: desc,
      };
    };
  }

  return { device, calls };
}

function makeOpaqueShaderModule(label, hashHex) {
  return {
    label,
    hashHex,
    getCompilationInfo: async () => ({ messages: [] }),
  };
}

// Naga-30 / playgta6-shaders canonical VS output (all perspective default, vec4<f32>)
const NAGA_VS_3_LOCS = [
  'struct VsIn {',
  '    @location(0) attr_0_: vec4<f32>,',
  '    @location(1) attr_1_: vec4<u32>,',
  '}',
  '',
  'struct VsOut {',
  '    @builtin(position) position: vec4<f32>,',
  '    @location(0) param_0_: vec4<f32>,',
  '    @location(1) param_1_: vec4<f32>,',
  '    @location(2) param_2_: vec4<f32>,',
  '}',
  '',
  '@vertex',
  'fn vs_main(vs_in: VsIn) -> VsOut {',
  '    var vs_out: VsOut;',
  '    vs_out = VsOut();',
  '    vs_out.position = vs_in.attr_0_;',
  '    vs_out.param_0_ = vec4<f32>(1.0, 0.0, 0.0, 1.0);',
  '    vs_out.param_1_ = vec4<f32>(0.0, 1.0, 0.0, 1.0);',
  '    vs_out.param_2_ = vec4<f32>(0.0, 0.0, 1.0, 1.0);',
  '    let _e1 = vs_out;',
  '    return _e1;',
  '}',
].join('\n');

// Naga-30 PS with @interpolate(flat) on @location(2) (matching the 3,121 PS in splash.wgsl.pack)
const NAGA_PS_FLAT_LOC2 = [
  'struct FragIn {',
  '    @location(0) attr_0_: vec4<f32>,',
  '    @location(1) attr_1_: vec4<f32>,',
  '    @location(2) @interpolate(flat) attr_2_: vec4<f32>,',
  '}',
  '',
  'struct PsOut {',
  '    @location(0) mrt_0_: vec4<f32>,',
  '}',
  '',
  '@fragment',
  'fn fs_main(frag_in: FragIn) -> PsOut {',
  '    var ps_out: PsOut;',
  '    ps_out.mrt_0_ = frag_in.attr_0_ + frag_in.attr_2_;',
  '    return ps_out;',
  '}',
].join('\n');

// Naga-30 PS matching VS_3_LOCS with zero mismatches
const NAGA_PS_MATCHING = [
  'struct FragIn {',
  '    @location(0) attr_0_: vec4<f32>,',
  '    @location(1) attr_1_: vec4<f32>,',
  '    @location(2) attr_2_: vec4<f32>,',
  '}',
  '',
  'struct PsOut {',
  '    @location(0) mrt_0_: vec4<f32>,',
  '}',
  '',
  '@fragment',
  'fn fs_main(frag_in: FragIn) -> PsOut {',
  '    var ps_out: PsOut;',
  '    ps_out.mrt_0_ = frag_in.attr_0_;',
  '    return ps_out;',
  '}',
].join('\n');

test('P1 #9: VS perspective @location(2) + PS @interpolate(flat) @location(2) patches cleanly & caches on reuse', () => {
  const { device, calls } = createMockWebGpuDevice({
    validateRenderPipeline(desc) {
      // Simulate strict Dawn/Tint inter-stage validation on createRenderPipeline.
      const vsCode = getRegisteredShaderModuleSource(desc.vertex.module);
      const fsCode = getRegisteredShaderModuleSource(desc.fragment.module);
      assert.ok(vsCode && fsCode, 'mock validator expects shader sources');
      const check = analyzeInterstageCompatibility(vsCode, fsCode);
      assert.equal(check.compatible, true, `Dawn validation failed: ${check.patchSignature}`);
    },
  });

  const cache = new WebGpuPipelineCache({ device });
  const vsMod = makeOpaqueShaderModule('wgpk:vertex:0x0000000000001001', '0x0000000000001001');
  const psModA = makeOpaqueShaderModule('wgpk:fragment:0x0000000000002001', '0x0000000000002001');
  const psModB = makeOpaqueShaderModule('wgpk:fragment:0x0000000000002002', '0x0000000000002002');

  cache.registerShaderModule(vsMod, '0x0000000000001001', NAGA_VS_3_LOCS);
  cache.registerShaderModuleSource(psModA, NAGA_PS_FLAT_LOC2);
  // Distinct PS with reversed attribute ordering `@interpolate(flat) @location(2)` -> same patch signature!
  const psFlatReversed = NAGA_PS_FLAT_LOC2.replace(
    '@location(2) @interpolate(flat) attr_2_: vec4<f32>,',
    '@interpolate(flat) @location(2) attr_2_: vec4<f32>,'
  );
  cache.registerShaderModuleSource(psModB, psFlatReversed);

  const descA = {
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: psModA, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  };
  const pipeA = cache.getOrCreateRenderPipeline(descA);
  assert.equal(pipeA.kind, 'GPURenderPipeline');
  assert.equal(calls.createShaderModule.length, 1);

  const patchedVsCode = calls.createShaderModule[0].desc.code;
  assertBalancedWgsl(patchedVsCode, 'patched flat loc2');
  assert.match(patchedVsCode, /@location\(2\)\s+@interpolate\(flat\)\s+param_2_:\s*vec4<f32>/);
  // VsIn must never be modified (especially integer attr_1_ in VsIn!).
  assert.match(patchedVsCode, /struct VsIn \{\s*@location\(0\) attr_0_: vec4<f32>,\s*@location\(1\) attr_1_: vec4<u32>,\s*\}/);

  // Repeat call with same descriptor -> pipeline cache hit, ZERO additional shader compilation!
  const pipeA2 = cache.getOrCreateRenderPipeline(descA);
  assert.equal(pipeA2, pipeA);
  assert.equal(calls.createShaderModule.length, 1);

  // Second pipeline pairing the SAME VS with psModB (same flat@2 signature) -> reuses cached patched VS module!
  const descB = {
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: psModB, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
  };
  const pipeB = cache.getOrCreateRenderPipeline(descB);
  assert.equal(pipeB.kind, 'GPURenderPipeline');
  assert.equal(calls.createShaderModule.length, 1, 'patched VS module must be reused from LRU cache');
  assert.equal(pipeA.descriptor.vertex.module, pipeB.descriptor.vertex.module);

  const stats = cache.getStats();
  assert.equal(stats.interstagePatchedPipelines, 2);
  assert.equal(stats.interstageFlatFixes, 2);
  assert.equal(stats.interstageMissingLocationFixes, 0);

  const interStats = cache.getInterstageStats();
  assert.equal(interStats.patchedModuleCacheSize, 1);
  assert.equal(interStats.moduleCacheMisses, 1);
  assert.equal(interStats.moduleCacheHits, 1);
});

test('P1 #9: Missing VS @location(3) & @location(5) read by PS are synthesized cleanly in VsOut', () => {
  const vsNoTrailingComma = [
    'struct VsOut {',
    '    @builtin(position) position: vec4<f32>,',
    '    @location(0) param_0_: vec4<f32>',
    '}',
    '@vertex',
    'fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {',
    '    var vs_out: VsOut;',
    '    vs_out = VsOut();',
    '    vs_out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);',
    '    vs_out.param_0_ = vec4<f32>(1.0);',
    '    return vs_out;',
    '}',
  ].join('\n');

  const psReadsMissing = [
    'struct FragIn {',
    '    @location(0) attr_0_: vec4<f32>,',
    '    @location(3) attr_3_: vec4<f32>,',
    '    @location(5) @interpolate(flat) attr_5_: vec4<f32>,',
    '}',
    '@fragment',
    'fn fs_main(frag_in: FragIn) -> @location(0) vec4<f32> {',
    '    return frag_in.attr_0_ + frag_in.attr_3_ + frag_in.attr_5_;',
    '}',
  ].join('\n');

  const res = patchInterstageWgsl(vsNoTrailingComma, psReadsMissing);
  assert.equal(res.patched, true);
  assert.equal(res.missingLocationFixes, 2);
  assert.equal(res.flatFixes, 0);
  assertBalancedWgsl(res.vsCode, 'synthesized missing locs');
  // Comma must have been inserted after param_0_ before synthesized fields.
  assert.match(res.vsCode, /@location\(0\) param_0_: vec4<f32>,/);
  assert.match(res.vsCode, /@location\(3\) _pg6_link_loc_3: vec4<f32>,/);
  assert.match(res.vsCode, /@location\(5\) @interpolate\(flat\) _pg6_link_loc_5: vec4<f32>,/);

  // Re-analysis must report 100% compatible.
  const verify = analyzeInterstageCompatibility(res.vsCode, psReadsMissing);
  assert.equal(verify.compatible, true);
  assert.equal(verify.needsPatch, false);

  // Also test through WebGpuPipelineCache with descriptor helper vsCode/fsCode stripped cleanly.
  const { device, calls } = createMockWebGpuDevice();
  const cache = new WebGpuPipelineCache({ device });
  const vsMod = makeOpaqueShaderModule('vs-missing', '0x0000000000003001');
  const psMod = makeOpaqueShaderModule('ps-missing', '0x0000000000003002');

  const pipe = cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vsCode: vsNoTrailingComma,
    fsCode: psReadsMissing,
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: psMod, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  assert.equal(pipe.kind, 'GPURenderPipeline');
  assert.equal('vsCode' in calls.createRenderPipeline[0], false, 'helper vsCode must be stripped');
  assert.equal('fsCode' in calls.createRenderPipeline[0], false, 'helper fsCode must be stripped');
  assert.equal(cache.getStats().interstagePatchedPipelines, 1);
  assert.equal(cache.getStats().interstageMissingLocationFixes, 2);
});

test('P1 #9: Already-matching VS+PS is a zero-overhead passthrough with zero shader re-compilation', () => {
  const { device, calls } = createMockWebGpuDevice();
  const cache = new WebGpuPipelineCache({ device });
  const vsMod = makeOpaqueShaderModule('vs-match', '0x0000000000004001');
  const psMod = makeOpaqueShaderModule('ps-match', '0x0000000000004002');
  registerShaderModuleSource(vsMod, NAGA_VS_3_LOCS);
  registerShaderModuleSource(psMod, NAGA_PS_MATCHING);

  const directPatch = patchInterstageWgsl(NAGA_VS_3_LOCS, NAGA_PS_MATCHING);
  assert.equal(directPatch.patched, false);
  assert.equal(directPatch.vsCode, NAGA_VS_3_LOCS);

  // Explicit @interpolate(perspective) and @interpolate(perspective, center) match omitted default.
  const psExplicitPerspective = NAGA_PS_MATCHING.replace(
    '@location(1) attr_1_: vec4<f32>,',
    '@location(1) @interpolate(perspective, center) attr_1_: vec4<f32>,'
  );
  assert.equal(analyzeInterstageCompatibility(NAGA_VS_3_LOCS, psExplicitPerspective).compatible, true);

  // Comments containing fake @location/@interpolate must be ignored.
  const psWithComments = [
    'struct FragIn {',
    '    // @location(9) @interpolate(flat) fake_line: vec4<f32>,',
    '    /* outer /* nested @location(8) @interpolate(flat) fake_nested: vec4<f32>, */ still comment */',
    '    @location(0) attr_0_: vec4<f32>,',
    '}',
    '@fragment fn fs_main(frag_in: FragIn) -> @location(0) vec4<f32> { return frag_in.attr_0_; }',
  ].join('\n');
  assert.equal(analyzeInterstageCompatibility(NAGA_VS_3_LOCS, psWithComments).compatible, true);

  const pipe = cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: psMod, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  assert.equal(pipe.descriptor.vertex.module, vsMod);
  assert.equal(calls.createShaderModule.length, 0);
  assert.equal(cache.getStats().interstagePatchedPipelines, 0);
  assert.equal(cache.getStats().interstageFlatFixes, 0);
  assert.equal(cache.getStats().interstageMissingLocationFixes, 0);
});

test('P1 #9: Integer VS output without @interpolate(flat), flat/either, linear/centroid, and removing flat on float', () => {
  // 1. Integer VS output missing @interpolate(flat) (even when PS does not read @location(1))
  const vsIntMissingFlat = [
    'struct VsOut {',
    '    @builtin(position) position: vec4<f32>,',
    '    @location(0) uv: vec4<f32>,',
    '    @location(1) mat_id: vec4<u32>,',
    '}',
    '@vertex fn vs_main() -> VsOut { var o: VsOut; return o; }',
  ].join('\n');
  const psReadsOnly0 = [
    '@fragment fn fs_main(@location(0) uv: vec4<f32>) -> @location(0) vec4<f32> { return uv; }',
  ].join('\n');

  const intRes = patchInterstageWgsl(vsIntMissingFlat, psReadsOnly0);
  assert.equal(intRes.patched, true);
  assert.equal(intRes.integerFlatFixes, 1);
  assert.equal(intRes.flatFixes, 1);
  assert.match(intRes.vsCode, /@location\(1\)\s+@interpolate\(flat\)\s+mat_id:\s*vec4<u32>/);
  assert.equal(analyzeInterstageCompatibility(intRes.vsCode, psReadsOnly0).compatible, true);

  // 2. @interpolate(flat, either) and @interpolate(linear, centroid)
  const psFancy = [
    'struct FragIn {',
    '    @location(0) @interpolate(linear, centroid) a0: vec4<f32>,',
    '    @location(1) @interpolate(flat, either) a1: vec4<f32>,',
    '    @location(2) @interpolate(perspective, centroid) a2: vec4<f32>,',
    '}',
    '@fragment fn fs_main(i: FragIn) -> @location(0) vec4<f32> { return i.a0 + i.a1 + i.a2; }',
  ].join('\n');
  const fancyRes = patchInterstageWgsl(NAGA_VS_3_LOCS, psFancy);
  assert.equal(fancyRes.patched, true);
  assert.equal(fancyRes.interpolateFixes, 3);
  assert.equal(fancyRes.flatFixes, 1);
  assert.match(fancyRes.vsCode, /@location\(0\)\s+@interpolate\(linear, centroid\)\s+param_0_/);
  assert.match(fancyRes.vsCode, /@location\(1\)\s+@interpolate\(flat, either\)\s+param_1_/);
  assert.match(fancyRes.vsCode, /@location\(2\)\s+@interpolate\(perspective, centroid\)\s+param_2_/);
  assert.equal(analyzeInterstageCompatibility(fancyRes.vsCode, psFancy).compatible, true);

  // 3. VS has @interpolate(flat) on float, PS expects default perspective -> strips @interpolate(flat)
  const vsFlatFloat = NAGA_VS_3_LOCS.replace(
    '@location(2) param_2_: vec4<f32>,',
    '@location(2) @interpolate(flat) param_2_: vec4<f32>,'
  );
  const stripRes = patchInterstageWgsl(vsFlatFloat, NAGA_PS_MATCHING);
  assert.equal(stripRes.patched, true);
  assert.equal(stripRes.flatFixes, 1);
  assert.doesNotMatch(stripRes.vsCode, /@interpolate\(flat\)/);
  assert.equal(analyzeInterstageCompatibility(stripRes.vsCode, NAGA_PS_MATCHING).compatible, true);
});

test('P1 #9: Wrapper mode handles direct @builtin(position) VS, positional struct ctors, and vector/scalar type mismatches', () => {
  // 1. Direct @builtin(position) VS (BUILTIN_FALLBACK_WGSL) paired with real PS
  const fallbackVs = routerMod.BUILTIN_FALLBACK_WGSL.vertex;
  const directRes = patchInterstageWgsl(fallbackVs, NAGA_PS_FLAT_LOC2);
  assert.equal(directRes.patched, true);
  assert.equal(directRes.missingLocationFixes, 3);
  assertBalancedWgsl(directRes.vsCode, 'wrapped direct builtin VS');
  assert.match(directRes.vsCode, /fn _pg6_orig_vs_main\(vertex_index: u32\) -> vec4<f32>/);
  assert.match(directRes.vsCode, /@location\(2\)\s+@interpolate\(flat\)/);
  assert.equal(analyzeInterstageCompatibility(directRes.vsCode, NAGA_PS_FLAT_LOC2).compatible, true);

  // 2. Type mismatch: textured_quad_24b VS outputs @location(0) vec2<f32>, PS expects vec4<f32> + flat@2
  const quadVs = routerMod.BUILTIN_FALLBACK_PROFILES.textured_quad_24b.vertex;
  const typeRes = patchInterstageWgsl(quadVs, NAGA_PS_FLAT_LOC2);
  assert.equal(typeRes.patched, true);
  assert.equal(typeRes.typeFixes, 1);
  assert.equal(typeRes.missingLocationFixes, 2);
  assertBalancedWgsl(typeRes.vsCode, 'wrapped type mismatch VS');
  assert.match(typeRes.vsCode, /vec4<f32>\(_pg6_raw\.uv, 0\.0, 0\.0\)/);
  assert.equal(analyzeInterstageCompatibility(typeRes.vsCode, NAGA_PS_FLAT_LOC2).compatible, true);

  // 3. Narrowing + scalar bitcast: VS outputs vec4<u32> @location(0), PS expects vec2<f32> @location(0)
  const vsU32 = [
    'struct VsOut {',
    '    @builtin(position) position: vec4<f32>,',
    '    @location(0) @interpolate(flat) packed: vec4<u32>,',
    '}',
    '@vertex fn vs_main() -> VsOut { var o: VsOut; return o; }',
  ].join('\n');
  const psVec2F32 = '@fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { return vec4<f32>(uv, 0.0, 1.0); }';
  const bitcastRes = patchInterstageWgsl(vsU32, psVec2F32);
  assert.equal(bitcastRes.patched, true);
  assert.equal(bitcastRes.typeFixes, 1);
  assertBalancedWgsl(bitcastRes.vsCode, 'wrapped bitcast VS');
  assert.match(bitcastRes.vsCode, /bitcast<vec2<f32>>\(\(_pg6_raw\.packed\)\.xy\)/);
  assert.equal(analyzeInterstageCompatibility(bitcastRes.vsCode, psVec2F32).compatible, true);

  // 4. Positional struct constructor VsOut(pos, uv) with missing location -> uses wrapper so ctor stays valid!
  const vsPosCtor = [
    'struct VsOut {',
    '    @builtin(position) position: vec4<f32>,',
    '    @location(0) uv: vec4<f32>,',
    '}',
    '@vertex fn vs_main() -> VsOut { return VsOut(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(0.5)); }',
  ].join('\n');
  const posRes = patchInterstageWgsl(vsPosCtor, NAGA_PS_FLAT_LOC2);
  assert.equal(posRes.patched, true);
  assert.equal(posRes.missingLocationFixes, 2);
  assertBalancedWgsl(posRes.vsCode, 'wrapped positional ctor VS');
  assert.match(posRes.vsCode, /return VsOut\(vec4<f32>\(0\.0, 0\.0, 0\.0, 1\.0\), vec4<f32>\(0\.5\)\);/);
  assert.equal(analyzeInterstageCompatibility(posRes.vsCode, NAGA_PS_FLAT_LOC2).compatible, true);
});

test('P1 #9: Bounded LRU eviction (#1077) for patched shader modules, async path, and cache lifecycle', async () => {
  assert.equal(DEFAULT_MAX_INTERSTAGE_MODULES, 512);
  assert.equal(DEFAULT_MAX_INTERSTAGE_INTERFACE_CACHE, 2048);

  const { device, calls } = createMockWebGpuDevice({ withAsync: true });
  const cache = new WebGpuPipelineCache({
    device,
    maxInterstageModules: 2,
    maxInterstageInterfaceCache: 3,
  });

  const vsMod = makeOpaqueShaderModule('vs-lru', '0x0000000000005000');
  cache.registerShaderModule(vsMod, '0x0000000000005000', NAGA_VS_3_LOCS);

  const makePsFlatAt = (loc) =>
    [
      'struct FragIn {',
      `    @location(${loc}) @interpolate(flat) a: vec4<f32>,`,
      '}',
      '@fragment fn fs_main(i: FragIn) -> @location(0) vec4<f32> { return i.a; }',
    ].join('\n');

  const ps0 = makeOpaqueShaderModule('ps-0', '0x0000000000005001');
  const ps1 = makeOpaqueShaderModule('ps-1', '0x0000000000005002');
  const ps2 = makeOpaqueShaderModule('ps-2', '0x0000000000005003');
  const ps0Dup = makeOpaqueShaderModule('ps-0-dup', '0x0000000000005004');
  cache.registerShaderModuleSource(ps0, makePsFlatAt(0));
  cache.registerShaderModuleSource(ps1, makePsFlatAt(1));
  cache.registerShaderModuleSource(ps2, makePsFlatAt(2));
  cache.registerShaderModuleSource(ps0Dup, makePsFlatAt(0));

  // Compile loc0 (sync) and loc1 (async) -> cache has [loc0, loc1]
  cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: ps0, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  await cache.getOrCreateRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: ps1, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  assert.equal(cache.getInterstageStats().patchedModuleCacheSize, 2);
  assert.equal(cache.getInterstageStats().moduleCacheEvictions, 0);

  // Touch loc0 again via ps0Dup -> refreshes loc0 recency so loc1 becomes oldest!
  cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: ps0Dup, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  assert.equal(cache.getInterstageStats().moduleCacheHits, 1);
  assert.equal(calls.createShaderModule.length, 2);

  // Insert loc2 -> evicts loc1 (NOT loc0!), keeping size capped at 2.
  cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: ps2, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  assert.equal(cache.getInterstageStats().patchedModuleCacheSize, 2);
  assert.equal(cache.getInterstageStats().moduleCacheEvictions, 1);
  assert.ok(cache.getInterstageStats().interfaceCacheSize <= 3);

  // Reusing loc0 with a different target format still HITS the patched module cache (was protected by LRU)!
  cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: ps0, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
  });
  assert.equal(cache.getInterstageStats().moduleCacheHits, 2);
  assert.equal(calls.createShaderModule.length, 3);

  // clear({ resetStats: true }) empties patched modules, interface cache, and counters.
  cache.clear({ resetStats: true });
  assert.equal(cache.getInterstageStats().patchedModuleCacheSize, 0);
  assert.equal(cache.getInterstageStats().interfaceCacheSize, 0);
  assert.equal(cache.getStats().interstagePatchedPipelines, 0);
  assert.equal(cache.getStats().interstageFlatFixes, 0);
  assert.equal(cache.getStats().interstageMissingLocationFixes, 0);
});

test('P1 #9: Built-in fallback VS module auto-resolves WGSL and graceful fallback when createShaderModule throws', () => {
  // 1. Unregistered fallback:default:vs module paired with real PS auto-resolves BUILTIN_FALLBACK_WGSL.vertex
  const { device, calls } = createMockWebGpuDevice();
  const cache = new WebGpuPipelineCache({ device });
  const fallbackVsMod = makeOpaqueShaderModule('fallback:default:vs', '0x0000000000006001');
  const realPsMod = makeOpaqueShaderModule('wgpk:fragment:0x0000000000006002', '0x0000000000006002');
  cache.registerShaderModuleSource(realPsMod, NAGA_PS_FLAT_LOC2);

  const pipe = cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: fallbackVsMod, entryPoint: 'vs_main' },
    fragment: { module: realPsMod, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  assert.equal(pipe.kind, 'GPURenderPipeline');
  assert.equal(calls.createShaderModule.length, 1);
  assert.equal(cache.getStats().interstagePatchedPipelines, 1);
  assert.equal(cache.getStats().interstageMissingLocationFixes, 3);

  // 2. If device.createShaderModule throws, pipeline creation gracefully uses original module without crashing.
  const { device: failDevice, calls: failCalls } = createMockWebGpuDevice({
    failCreateShaderModule: true,
  });
  const failCache = new WebGpuPipelineCache({ device: failDevice });
  const vsMod = makeOpaqueShaderModule('vs-fail', '0x0000000000007001');
  const psMod = makeOpaqueShaderModule('ps-fail', '0x0000000000007002');
  failCache.registerShaderModuleSource(vsMod, NAGA_VS_3_LOCS);
  failCache.registerShaderModuleSource(psMod, NAGA_PS_FLAT_LOC2);

  const fallbackPipe = failCache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: psMod, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  assert.equal(fallbackPipe.descriptor.vertex.module, vsMod);
  assert.equal(failCalls.createRenderPipeline.length, 1);
  assert.equal(failCache.getStats().interstagePatchedPipelines, 0);

  // 3. parseInterstageInterface handles direct PS parameters cleanly
  const psDirectIface = parseInterstageInterface(
    '@fragment fn fs_main(@location(0) a: vec4<f32>, @interpolate(flat) @location(4u) b: vec4<u32>) -> @location(0) vec4<f32> { return a; }',
    'fragment'
  );
  assert.ok(psDirectIface);
  assert.equal(psDirectIface.locations.size, 2);
  assert.equal(psDirectIface.locations.get(4).interpolate.type, 'flat');
  assert.equal(psDirectIface.locations.get(4).typeInfo.canonical, 'vec4<u32>');
});

test('P1 #9 review hardening: fn-containing entry point names, @invariant, enable f16, fast-lookup preservation, and #1077 memory caps', () => {
  // 1. Entry point name containing `fn` (e.g. `vs_fn` / `fn_main`) must never corrupt wrapper output.
  const vsFnName = [
    '@vertex',
    'fn vs_fn(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {',
    '    return vec4<f32>(0.0, 0.0, 0.0, 1.0);',
    '}',
  ].join('\n');
  const fnRes = patchInterstageWgsl(vsFnName, NAGA_PS_FLAT_LOC2);
  assert.equal(fnRes.patched, true);
  assertBalancedWgsl(fnRes.vsCode, 'vs_fn wrapper');
  assert.match(fnRes.vsCode, /fn _pg6_orig_vs_fn\(vertex_index|fn _pg6_orig_vs_fn\(vid: u32\)/);
  assert.match(fnRes.vsCode, /@vertex\s+fn vs_fn\(@builtin\(vertex_index\) vid: u32\) -> _Pg6LinkVsOut/);
  assert.equal(analyzeInterstageCompatibility(fnRes.vsCode, NAGA_PS_FLAT_LOC2).compatible, true);

  // 2. @invariant on @builtin(position) is stripped from helper struct and preserved on wrapper struct.
  const vsInvariant = [
    'struct VsOut {',
    '    @invariant @builtin(position) position: vec4<f32>,',
    '    @location(0) uv: vec2<f32>,',
    '}',
    '@vertex fn vs_main() -> VsOut { var o: VsOut; return o; }',
  ].join('\n');
  const invRes = patchInterstageWgsl(vsInvariant, NAGA_PS_FLAT_LOC2);
  assert.equal(invRes.patched, true);
  assertBalancedWgsl(invRes.vsCode, 'invariant wrapper');
  const origStructBlock = /struct VsOut \{[^}]*\}/.exec(invRes.vsCode);
  assert.ok(origStructBlock);
  assert.doesNotMatch(origStructBlock[0], /@invariant|@builtin|@location/);
  const wrapStructBlock = /struct _Pg6LinkVsOut \{[^}]*\}/.exec(invRes.vsCode);
  assert.ok(wrapStructBlock);
  assert.match(wrapStructBlock[0], /@invariant\s+@builtin\(position\)\s+position:\s*vec4<f32>/);
  assert.equal(analyzeInterstageCompatibility(invRes.vsCode, NAGA_PS_FLAT_LOC2).compatible, true);

  // 3. Introducing f16 inter-stage outputs into a VS without `enable f16;` prepends `enable f16;`.
  const psF16 = [
    'enable f16;',
    'struct FragIn {',
    '    @location(0) a0: vec4<f32>,',
    '    @location(3) a3: vec4<f16>,',
    '}',
    '@fragment fn fs_main(i: FragIn) -> @location(0) vec4<f32> { return i.a0; }',
  ].join('\n');
  const f16Res = patchInterstageWgsl(NAGA_VS_3_LOCS, psF16);
  assert.equal(f16Res.patched, true);
  assert.match(f16Res.vsCode, /^enable f16;\n/);
  assert.equal(analyzeInterstageCompatibility(f16Res.vsCode, psF16).compatible, true);

  // 4. Inter-stage patching an unrelated pipeline must NOT invalidate warm fast-lookup caches for other pipelines.
  const { device } = createMockWebGpuDevice();
  const cache = new WebGpuPipelineCache({ device });
  const vsWarm = makeOpaqueShaderModule('vs-warm', '0x0000000000008001');
  const psWarm = makeOpaqueShaderModule('ps-warm', '0x0000000000008002');
  const descWarm = {
    layout: 'auto',
    vertex: { module: vsWarm, entryPoint: 'vs_main' },
    fragment: { module: psWarm, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  };
  cache.getOrCreateRenderPipeline(descWarm);
  cache.getOrCreateRenderPipeline(descWarm);
  const hitsBefore = cache.getStats().fastLookupHits;
  assert.equal(hitsBefore, 1);

  const vsPatched = makeOpaqueShaderModule('vs-patch', '0x0000000000008003');
  const psPatched = makeOpaqueShaderModule('ps-patch', '0x0000000000008004');
  cache.registerShaderModuleSource(vsPatched, NAGA_VS_3_LOCS);
  cache.registerShaderModuleSource(psPatched, NAGA_PS_FLAT_LOC2);
  cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsPatched, entryPoint: 'vs_main' },
    fragment: { module: psPatched, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });

  // Warm descWarm must STILL hit the fast lookup cache!
  cache.getOrCreateRenderPipeline(descWarm);
  assert.equal(cache.getStats().fastLookupHits, hitsBefore + 1);

  // 5. #1077: _interstageInterfaceCache must NOT retain raw WGSL strings, and trimTo(0) clears interstage caches.
  for (const entry of cache._interstageInterfaceCache.values()) {
    assert.equal('wgslCode' in entry, false, 'interface cache must not retain full WGSL strings');
  }
  cache.trimTo(0);
  assert.equal(cache.getInterstageStats().patchedModuleCacheSize, 0);
  assert.equal(cache.getInterstageStats().interfaceCacheSize, 0);
});

test('#2085 SPI_PS_INPUT_CNTL remap: Brightness Calibration VS_Blit (COLOR0@0, TEXCOORD0@1) -> PS_BlitCalibration (TEXCOORD0@0 -> @1)', async () => {
  // Exact PIPE 55065 VS_Blit / PS_BlitCalibration inter-stage structure from issue #2085:
  // VS exports COLOR0 (semantic_reg=1) at @location(0) and TEXCOORD0 (semantic_reg=2) at @location(1).
  // PS leaves COLOR0 unused (ISG1 rw=0x0) and reads TEXCOORD0 (semantic_reg=2, mask=3) at @location(0),
  // while also outputting fragment MRT at @location(0) (PsOut must NEVER be rewritten!).
  const vsBlitWgsl = [
    'struct VsOut {',
    '    @builtin(position) position: vec4<f32>,',
    '    @location(0) param_0_: vec4<f32>,',
    '    @location(1) param_1_: vec4<f32>,',
    '}',
    '@vertex fn vs_main(@location(0) pos: vec3<f32>, @location(1) col: vec4<f32>, @location(2) uv: vec2<f32>) -> VsOut {',
    '    var out: VsOut;',
    '    out.position = vec4<f32>(pos, 1.0);',
    '    out.param_0_ = col;',
    '    out.param_1_ = vec4<f32>(uv, 0.0, 1.0);',
    '    return out;',
    '}',
  ].join('\n');

  const psBlitCalibrationWgsl = [
    'struct FragIn {',
    '    // @location(0) comment_ignored: vec4<f32>,',
    '    @location(0) attr_0_: vec4<f32>,',
    '}',
    'struct PsOut {',
    '    @location(0) mrt_0_: vec4<f32>,',
    '}',
    '@fragment fn fs_main(frag_in: FragIn) -> PsOut {',
    '    var ps_out: PsOut;',
    '    ps_out.mrt_0_ = frag_in.attr_0_;',
    '    return ps_out;',
    '}',
  ].join('\n');

  const vsOutputs = [
    { semantic_reg: 1, location: 0 },
    { semantic_reg: 2, location: 1 },
  ];
  const psInputsWithUnusedColor = [
    { semantic_reg: 1, location: 0, mask: 0 },
    { semantic_reg: 2, location: 0, mask: 3 },
  ];

  // 1. Pure remap computation skips unused mask=0 entry and maps PS @location(0) -> VS @location(1)
  const remaps = computeInterstageLocationRemaps(vsOutputs, psInputsWithUnusedColor);
  assert.deepEqual(remaps, [
    {
      semanticReg: 2,
      semantic_reg: 2,
      psLocation: 0,
      ps_location: 0,
      vsLocation: 1,
      vs_location: 1,
    },
  ]);

  // 2. Direct patchInterstageWgsl rewrites FragIn @location(0) -> @location(1), leaves PsOut & VS untouched
  const patched = patchInterstageWgsl(vsBlitWgsl, psBlitCalibrationWgsl, {
    vsOutputs,
    psInputs: psInputsWithUnusedColor,
  });
  assert.equal(patched.patched, true);
  assert.equal(patched.patchedStage, 'fragment');
  assert.equal(patched.locationRemapFixes, 1);
  assert.equal(patched.vsCode, vsBlitWgsl);
  assertBalancedWgsl(patched.psCode, 'remapped PS_BlitCalibration');
  assert.match(patched.psCode, /struct FragIn \{[\s\S]*@location\(1\) attr_0_: vec4<f32>,[\s\S]*\}/);
  assert.match(patched.psCode, /struct PsOut \{\s*@location\(0\) mrt_0_: vec4<f32>,\s*\}/);
  assert.match(patched.psCode, /\/\/ @location\(0\) comment_ignored/);
  assert.equal(analyzeInterstageCompatibility(patched.vsCode, patched.psCode).compatible, true);

  // 3. Identity mapping is left completely untouched
  const identityPsInputs = [
    { semantic_reg: 1, location: 0, mask: 15 },
    { semantic_reg: 2, location: 1, mask: 3 },
  ];
  assert.deepEqual(computeInterstageLocationRemaps(vsOutputs, identityPsInputs), []);
  const identityRes = patchInterstageWgsl(vsBlitWgsl, psBlitCalibrationWgsl, {
    vsOutputs,
    psInputs: identityPsInputs,
  });
  assert.equal(identityRes.patched, false);
  assert.equal(identityRes.locationRemapFixes, 0);
  assert.equal(identityRes.psCode, psBlitCalibrationWgsl);

  // 4. WebGpuPipelineCache end-to-end auto-lookup via ReflectionRegistry + caching + async path
  const registry = new ReflectionRegistry();
  registry.loadStageReflection({
    schema: 'playgta6-stage-reflection',
    version: 1,
    stages: {
      '0x5fe7a88d9cb680b3': {
        stage: 'vertex',
        bindings: [],
        vs_outputs: vsOutputs,
      },
      '0x56214b8d3b10a108': {
        stage: 'fragment',
        bindings: [],
        ps_inputs: [{ semantic_reg: 2, location: 0, mask: 3 }],
      },
    },
  });

  const { device, calls } = createMockWebGpuDevice({ withAsync: true });
  const cache = new WebGpuPipelineCache({ device, reflectionRegistry: registry });
  const vsMod = makeOpaqueShaderModule('wgpk:vertex:0x5fe7a88d9cb680b3', '0x5fe7a88d9cb680b3');
  const psMod = makeOpaqueShaderModule('wgpk:fragment:0x56214b8d3b10a108', '0x56214b8d3b10a108');
  cache.registerShaderModule(vsMod, '0x5fe7a88d9cb680b3', vsBlitWgsl);
  cache.registerShaderModule(psMod, '0x56214b8d3b10a108', psBlitCalibrationWgsl);

  const pipe1 = cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: psMod, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
  });
  assert.equal(calls.createShaderModule.length, 1);
  assert.equal(pipe1.descriptor.vertex.module, vsMod);
  assert.notEqual(pipe1.descriptor.fragment.module, psMod);
  assert.match(calls.createShaderModule[0].desc.code, /@location\(1\) attr_0_: vec4<f32>/);
  assert.equal(cache.getInterstageStats().locationRemapFixes, 1);
  assert.equal(cache.getStats().interstageLocationRemapFixes, 1);
  assert.equal(cache.getStats().interstagePatchedPipelines, 1);

  // Second pipeline with different RT format reuses cached remapped PS module (zero extra createShaderModule!)
  const pipe2 = await cache.getOrCreateRenderPipelineAsync({
    layout: 'auto',
    vertex: { module: vsMod, entryPoint: 'vs_main' },
    fragment: { module: psMod, entryPoint: 'fs_main', targets: [{ format: 'bgra8unorm' }] },
  });
  assert.equal(calls.createShaderModule.length, 1);
  assert.equal(pipe2.descriptor.fragment.module, pipe1.descriptor.fragment.module);
  assert.equal(cache.getInterstageStats().moduleCacheHits, 1);
  assert.equal(cache.getInterstageStats().locationRemapFixes, 2);

  // Descriptor helper vsOutputs/psInputs are stripped before reaching device.createRenderPipeline
  const vsMod2 = makeOpaqueShaderModule('vs-helper', '0x0000000000009001');
  const psMod2 = makeOpaqueShaderModule('ps-helper', '0x0000000000009002');
  registerShaderModuleReflection(vsMod2, { vs_outputs: vsOutputs });
  const pipe3 = cache.getOrCreateRenderPipeline({
    layout: 'auto',
    vsCode: vsBlitWgsl,
    fsCode: psBlitCalibrationWgsl,
    psInputs: [{ semantic_reg: 2, location: 0, mask: 3 }],
    vertex: { module: vsMod2, entryPoint: 'vs_main' },
    fragment: { module: psMod2, entryPoint: 'fs_main', targets: [{ format: 'rgba8unorm' }] },
  });
  assert.equal('psInputs' in pipe3.descriptor, false);
  assert.equal('vsOutputs' in pipe3.descriptor, false);
  assert.notEqual(pipe3.descriptor.fragment.module, psMod2);
});

test('#2085 SPI_PS_INPUT_CNTL remap handles swaps/cycles, direct PS params, combined VS flat fix, and unmapped collision avoidance', () => {
  // 1. 2-way swap (0->1, 1->0) and 3-way cycle (0->1, 1->2, 2->0) in one pass without clobbering
  const psCycleDirect =
    '@fragment fn fs_main(@location(0) a: vec4<f32>, @location(1) b: vec4<f32>, @location(2) c: vec4<f32>) -> @location(0) vec4<f32> { return a + b + c; }';
  const cycleRemap = remapPsInputLocationsWgsl(psCycleDirect, [
    { semanticReg: 2, psLocation: 0, vsLocation: 1 },
    { semanticReg: 3, psLocation: 1, vsLocation: 2 },
    { semanticReg: 1, psLocation: 2, vsLocation: 0 },
  ]);
  assert.equal(cycleRemap.patched, true);
  assert.equal(cycleRemap.locationRemapFixes, 3);
  assert.equal(cycleRemap.remapSignature, 'r:0->1;1->2;2->0');
  assertBalancedWgsl(cycleRemap.psCode, '3-cycle direct PS params');
  assert.match(
    cycleRemap.psCode,
    /@location\(1\) a: vec4<f32>, @location\(2\) b: vec4<f32>, @location\(0\) c: vec4<f32>/
  );
  assert.match(cycleRemap.psCode, /-> @location\(0\) vec4<f32>/);

  // 2. Combined PS remap + VS @interpolate(flat) fix targets the REMAPPED VS location (loc 1, NOT loc 0!)
  const psFlatRemapped = [
    'struct FragIn {',
    '    @location(0) @interpolate(flat) uv: vec4<f32>,',
    '}',
    '@fragment fn fs_main(i: FragIn) -> @location(0) vec4<f32> { return i.uv; }',
  ].join('\n');
  const combined = patchInterstageWgsl(NAGA_VS_3_LOCS, psFlatRemapped, {
    vsOutputs: [
      { semantic_reg: 1, location: 0 },
      { semantic_reg: 2, location: 1 },
    ],
    psInputs: [{ semantic_reg: 2, location: 0, mask: 3 }],
  });
  assert.equal(combined.patched, true);
  assert.equal(combined.patchedStage, 'both');
  assert.equal(combined.locationRemapFixes, 1);
  assert.equal(combined.flatFixes, 1);
  assert.match(combined.psCode, /@location\(1\) @interpolate\(flat\) uv: vec4<f32>/);
  // VS @location(1) gets @interpolate(flat), while VS @location(0) stays perspective!
  assert.match(combined.vsCode, /@location\(0\) param_0_: vec4<f32>/);
  assert.match(combined.vsCode, /@location\(1\) @interpolate\(flat\) param_1_: vec4<f32>/);
  assert.equal(analyzeInterstageCompatibility(combined.vsCode, combined.psCode).compatible, true);

  // 3. Unmapped PS input sitting at remapped target location is cleanly relocated (no duplicate @location(1))
  const psCollision = [
    'struct FragIn {',
    '    @location(0) mapped_uv: vec4<f32>,',
    '    @location(1) unmapped_extra: vec4<f32>,',
    '}',
    '@fragment fn fs_main(i: FragIn) -> @location(0) vec4<f32> { return i.mapped_uv + i.unmapped_extra; }',
  ].join('\n');
  const colRes = patchInterstageWgsl(NAGA_VS_3_LOCS, psCollision, {
    vsOutputs: [
      { semantic_reg: 1, location: 0 },
      { semantic_reg: 2, location: 1 },
    ],
    psInputs: [
      { semantic_reg: 2, location: 0, mask: 3 },
      { semantic_reg: 99, location: 1, mask: 15 },
    ],
  });
  assert.equal(colRes.patched, true);
  assertBalancedWgsl(colRes.psCode, 'collision-free PS');
  assertBalancedWgsl(colRes.vsCode, 'collision-free VS');
  assert.match(colRes.psCode, /@location\(1\) mapped_uv: vec4<f32>/);
  // NAGA_VS_3_LOCS already exports @location(0, 1, 2), so displaced unmapped input moves to @location(3)
  assert.match(colRes.psCode, /@location\(3\) unmapped_extra: vec4<f32>/);
  assert.match(colRes.vsCode, /@location\(3\) _pg6_link_loc_3: vec4<f32>/);
  assert.equal(analyzeInterstageCompatibility(colRes.vsCode, colRes.psCode).compatible, true);
});
