import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const routerMod = require('../web/workers/shader_fallback_router.js');

const {
  SHADER_STAGE,
  STAGE_NAMES,
  BUILTIN_FALLBACK_WGSL,
  DEFAULT_ENTRY_POINTS,
  DEFAULT_MAX_TRACKED_MISSING,
  normalizeShaderHash,
  formatShaderHash,
  normalizeStage,
  ShaderFallbackRouter,
} = routerMod;

const S9_FALLBACK_HASHES = Object.freeze({
  vertex: '0x1111111111111111',
  fragment: '0x2222222222222222',
  compute: '0x3333333333333333',
});

const S9_FALLBACK_WGSL = Object.freeze({
  vertex: [
    '@vertex',
    'fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {',
    '    return vec4<f32>(0.0, 0.0, 0.0, 1.0);',
    '}',
    '',
  ].join('\n'),
  fragment: [
    '@fragment',
    'fn fs_main() -> @location(0) vec4<f32> {',
    '    return vec4<f32>(0.2, 0.85, 0.45, 1.0);',
    '}',
    '',
  ].join('\n'),
  compute: [
    '@compute @workgroup_size(64)',
    'fn cs_main() {}',
    '',
  ].join('\n'),
});

const PACK_VS_WGSL =
  '@vertex fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(pos, 1.0); }';
const PACK_FS_WGSL =
  '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.25, 0.5, 1.0); }';
const PACK_CS_WGSL = '@compute @workgroup_size(1) fn cs_main() {}';

function createMapPackLoader(entries) {
  const byHash = new Map();
  for (const [hash, wgsl] of Object.entries(entries)) {
    const key = normalizeShaderHash(hash);
    byHash.set(key, wgsl);
    byHash.set(formatShaderHash(key), wgsl);
  }
  return {
    get(hash) {
      if (typeof hash === 'bigint' || typeof hash === 'number' || typeof hash === 'string') {
        try {
          const key = normalizeShaderHash(hash);
          return byHash.get(key) ?? byHash.get(formatShaderHash(key));
        } catch (_err) {
          return byHash.get(hash);
        }
      }
      return undefined;
    },
  };
}

function createGetSourcePackLoader(entries) {
  const map = createMapPackLoader(entries);
  return {
    getSource(hash) {
      return map.get(hash);
    },
  };
}

function createGetWgslPackLoader(entries) {
  const map = createMapPackLoader(entries);
  return {
    getWgsl(hash) {
      return map.get(hash);
    },
  };
}

function createLookupPackLoader(entries) {
  const map = createMapPackLoader(entries);
  return {
    lookup(hash) {
      const wgsl = map.get(hash);
      if (wgsl === undefined) return undefined;
      return { wgsl, entryPoint: null };
    },
  };
}

function createS9PackLoader() {
  return createGetSourcePackLoader({
    [S9_FALLBACK_HASHES.vertex]: S9_FALLBACK_WGSL.vertex,
    [S9_FALLBACK_HASHES.fragment]: S9_FALLBACK_WGSL.fragment,
    [S9_FALLBACK_HASHES.compute]: S9_FALLBACK_WGSL.compute,
  });
}

function assertValidWgsl(wgsl, stage) {
  assert.equal(typeof wgsl, 'string');
  assert.ok(wgsl.length > 0, 'WGSL must be non-empty');
  assert.match(wgsl, new RegExp(`@${stage}\\b`));
  assert.match(wgsl, /\bfn\s+[A-Za-z_]/);
}

test('UMD export attaches to globalThis.PlayGta6ShaderFallbackRouter and module.exports', () => {
  assert.equal(globalThis.PlayGta6ShaderFallbackRouter, routerMod);
  assert.equal(typeof ShaderFallbackRouter, 'function');
  assert.deepEqual(SHADER_STAGE, { VERTEX: 1, FRAGMENT: 2, COMPUTE: 3 });
  assert.deepEqual(STAGE_NAMES, { 1: 'vertex', 2: 'fragment', 3: 'compute' });
  assert.equal(DEFAULT_MAX_TRACKED_MISSING, 512);
  assert.deepEqual(DEFAULT_ENTRY_POINTS, {
    vertex: 'vs_main',
    fragment: 'fs_main',
    compute: 'cs_main',
  });
});

test('normalizeShaderHash accepts BigInt, 0x hex, bare hex, and safe integers', () => {
  const expected = 0xdeadbeefcafebaben;
  assert.equal(normalizeShaderHash(expected), expected);
  assert.equal(normalizeShaderHash('0xdeadbeefcafebabe'), expected);
  assert.equal(normalizeShaderHash('0XDEADBEEFCAFEBABE'), expected);
  assert.equal(normalizeShaderHash('deadbeefcafebabe'), expected);
  assert.equal(normalizeShaderHash(42), 42n);
  assert.equal(formatShaderHash(expected), '0xdeadbeefcafebabe');
  assert.equal(formatShaderHash(1), '0x0000000000000001');

  assert.throws(() => normalizeShaderHash(-1n), /u64 range/);
  assert.throws(() => normalizeShaderHash(1.5), /safe integer/);
  assert.throws(() => normalizeShaderHash('0xzz'), /Invalid hex/);
  assert.throws(() => normalizeShaderHash({}), /BigInt, hex string/);
});

test('normalizeStage accepts WGPK codes and stage name aliases', () => {
  assert.equal(normalizeStage(1), 'vertex');
  assert.equal(normalizeStage(2), 'fragment');
  assert.equal(normalizeStage(3), 'compute');
  assert.equal(normalizeStage('VERTEX'), 'vertex');
  assert.equal(normalizeStage('fs'), 'fragment');
  assert.equal(normalizeStage('ps'), 'fragment');
  assert.equal(normalizeStage('cs'), 'compute');
  assert.throws(() => normalizeStage(0), /Invalid shader stage code/);
  assert.throws(() => normalizeStage('geometry'), /Invalid shader stage name/);
});

test('BUILTIN_FALLBACK_WGSL provides minimal valid WGSL for every stage', () => {
  for (const stage of ['vertex', 'fragment', 'compute']) {
    assertValidWgsl(BUILTIN_FALLBACK_WGSL[stage], stage);
    assert.equal(
      Object.isFrozen(BUILTIN_FALLBACK_WGSL) || typeof BUILTIN_FALLBACK_WGSL[stage] === 'string',
      true
    );
  }
  assert.ok(BUILTIN_FALLBACK_WGSL.vertex.includes('vs_main'));
  assert.ok(BUILTIN_FALLBACK_WGSL.fragment.includes('fs_main'));
  assert.ok(BUILTIN_FALLBACK_WGSL.compute.includes('cs_main'));
});

test('resolve returns exact pack hit via getSource without recording missing telemetry', () => {
  const packHash = '0xaaaaaaaaaaaaaaaa';
  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({ [packHash]: PACK_VS_WGSL }),
    fallbackHashes: S9_FALLBACK_HASHES,
  });

  const result = router.resolve(packHash, 'vertex');
  assert.equal(result.source, 'pack');
  assert.equal(result.wgsl, PACK_VS_WGSL);
  assert.equal(result.requestedHash, packHash);
  assert.equal(result.resolvedHash, packHash);
  assert.equal(result.stage, 'vertex');
  assert.equal(result.entryPoint, 'vs_main');

  assert.deepEqual(router.getMissingReport(), []);
  assert.equal(router.getStats().packHits, 1);
  assert.equal(router.getStats().uniqueMissing, 0);
});

test('resolve finds exact pack hit via getWgsl, lookup, and Map.get interfaces', () => {
  const hash = 0xbbbbbbbbbbbbbbbbn;

  const viaGetWgsl = new ShaderFallbackRouter({
    packLoader: createGetWgslPackLoader({ [formatShaderHash(hash)]: PACK_FS_WGSL }),
  });
  assert.equal(viaGetWgsl.resolve(hash, SHADER_STAGE.FRAGMENT).source, 'pack');
  assert.equal(viaGetWgsl.resolve(hash, 'fragment').wgsl, PACK_FS_WGSL);

  const viaLookup = new ShaderFallbackRouter({
    packLoader: createLookupPackLoader({ [formatShaderHash(hash)]: PACK_CS_WGSL }),
  });
  const lookupHit = viaLookup.resolve(formatShaderHash(hash), 'compute');
  assert.equal(lookupHit.source, 'pack');
  assert.equal(lookupHit.wgsl, PACK_CS_WGSL);

  const map = new Map([
    [hash, PACK_VS_WGSL],
    [formatShaderHash(hash), PACK_VS_WGSL],
  ]);
  const viaMap = new ShaderFallbackRouter({ packLoader: map });
  assert.equal(viaMap.resolve(hash, 'vertex').source, 'pack');
});

test('resolve uses stage-matched S9 fallback hash from pack when exact miss', () => {
  const missingHash = '0xcafebabedeadbeef';
  const router = new ShaderFallbackRouter({
    packLoader: createS9PackLoader(),
    fallbackHashes: S9_FALLBACK_HASHES,
  });

  const vs = router.resolve(missingHash, 'vertex');
  assert.equal(vs.source, 'fallback');
  assert.equal(vs.wgsl, S9_FALLBACK_WGSL.vertex);
  assert.equal(vs.requestedHash, missingHash);
  assert.equal(vs.resolvedHash, S9_FALLBACK_HASHES.vertex);
  assert.equal(vs.entryPoint, 'vs_main');

  const fs = router.resolve('0x0102030405060708', SHADER_STAGE.FRAGMENT);
  assert.equal(fs.source, 'fallback');
  assert.equal(fs.resolvedHash, S9_FALLBACK_HASHES.fragment);
  assert.equal(fs.wgsl, S9_FALLBACK_WGSL.fragment);

  const cs = router.resolve(0x9999999999999999n, 'compute');
  assert.equal(cs.source, 'fallback');
  assert.equal(cs.resolvedHash, S9_FALLBACK_HASHES.compute);

  const stats = router.getStats();
  assert.equal(stats.fallbackHits, 3);
  assert.equal(stats.packHits, 0);
  assert.equal(stats.uniqueMissing, 3);
});

test('resolve falls back to builtin WGSL when pack and S9 fallback are absent', () => {
  const router = new ShaderFallbackRouter();
  const vs = router.resolve('0xabcdef0123456789', 'vertex');
  assert.equal(vs.source, 'builtin');
  assert.equal(vs.resolvedHash, null);
  assert.equal(vs.wgsl, BUILTIN_FALLBACK_WGSL.vertex);
  assert.equal(vs.entryPoint, 'vs_main');
  assertValidWgsl(vs.wgsl, 'vertex');

  const fs = router.resolve(1n, 2);
  assert.equal(fs.source, 'builtin');
  assert.equal(fs.wgsl, BUILTIN_FALLBACK_WGSL.fragment);
  assert.equal(fs.entryPoint, 'fs_main');

  const cs = router.resolve(2, 'COMPUTE');
  assert.equal(cs.source, 'builtin');
  assert.equal(cs.wgsl, BUILTIN_FALLBACK_WGSL.compute);
  assert.equal(cs.entryPoint, 'cs_main');

  assert.equal(router.getStats().builtinHits, 3);
  assert.equal(router.getStats().packHits, 0);
  assert.equal(router.getStats().fallbackHits, 0);
});

test('resolve uses builtin when fallbackHashes point at hashes missing from pack', () => {
  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({}),
    fallbackHashes: S9_FALLBACK_HASHES,
  });
  const result = router.resolve('0xfeedfacefeedface', 'fragment');
  assert.equal(result.source, 'builtin');
  assert.equal(result.wgsl, BUILTIN_FALLBACK_WGSL.fragment);
  assert.equal(router.getStats().builtinHits, 1);
  assert.equal(router.getStats().fallbackHits, 0);
});

test('missing telemetry deduplicates by hash+stage and sorts by hit count descending', () => {
  let tick = 1000;
  const router = new ShaderFallbackRouter({
    packLoader: createS9PackLoader(),
    fallbackHashes: S9_FALLBACK_HASHES,
    now: () => {
      tick += 10;
      return tick;
    },
  });

  const a = '0xaaaaaaaaaaaaaaaa';
  const b = '0xbbbbbbbbbbbbbbbb';
  const c = '0xcccccccccccccccc';

  router.resolve(a, 'vertex');
  router.resolve(a, 'vertex');
  router.resolve(a, 'vertex');
  router.resolve(b, 'fragment');
  router.resolve(b, 'fragment');
  router.resolve(c, 'compute');
  // Same hash, different stage — separate telemetry row.
  router.resolve(a, 'fragment');

  const report = router.getMissingReport();
  assert.equal(report.length, 4);
  assert.equal(report[0].hash, a);
  assert.equal(report[0].stage, 'vertex');
  assert.equal(report[0].hitCount, 3);
  assert.equal(report[1].hash, b);
  assert.equal(report[1].hitCount, 2);
  assert.ok(report[0].firstSeenMs <= report[0].lastSeenMs);
  assert.ok(report.every((row) => typeof row.hash === 'string' && row.hash.startsWith('0x')));
  assert.equal(JSON.parse(JSON.stringify(report)).length, 4);

  const limited = router.getMissingReport({ limit: 2 });
  assert.equal(limited.length, 2);
  assert.equal(limited[0].hitCount, 3);
  assert.equal(limited[1].hitCount, 2);
});

test('maxTrackedMissing caps unique rows and increments overflowMissingCount', () => {
  const router = new ShaderFallbackRouter({
    maxTrackedMissing: 2,
    now: () => 42,
  });

  router.resolve('0x0000000000000001', 'vertex');
  router.resolve('0x0000000000000002', 'fragment');
  router.resolve('0x0000000000000003', 'compute');
  router.resolve('0x0000000000000004', 'vertex');
  // Duplicate of an overflowed hash still overflows (not tracked).
  router.resolve('0x0000000000000003', 'compute');

  const stats = router.getStats();
  assert.equal(stats.uniqueMissing, 2);
  assert.equal(stats.overflowMissingCount, 3);
  assert.equal(router.getMissingReport().length, 2);
  assert.equal(router.getMissingReport()[0].hitCount, 1);
});

test('getStats reports hitRatePct from pack hits and reset clears telemetry', () => {
  const packHash = '0xdddddddddddddddd';
  const router = new ShaderFallbackRouter({
    packLoader: createGetSourcePackLoader({
      [packHash]: PACK_VS_WGSL,
      [S9_FALLBACK_HASHES.vertex]: S9_FALLBACK_WGSL.vertex,
    }),
    fallbackHashes: { vertex: S9_FALLBACK_HASHES.vertex },
  });

  router.resolve(packHash, 'vertex');
  router.resolve(packHash, 'vertex');
  router.resolve('0xeeeeeeeeeeeeeeee', 'vertex');
  router.resolve('0xffffffffffffffff', 'fragment');

  let stats = router.getStats();
  assert.equal(stats.totalLookups, 4);
  assert.equal(stats.packHits, 2);
  assert.equal(stats.fallbackHits, 1);
  assert.equal(stats.builtinHits, 1);
  assert.equal(stats.uniqueMissing, 2);
  assert.equal(stats.overflowMissingCount, 0);
  assert.equal(stats.hitRatePct, 50);

  router.reset();
  stats = router.getStats();
  assert.deepEqual(stats, {
    totalLookups: 0,
    packHits: 0,
    fallbackHits: 0,
    builtinHits: 0,
    uniqueMissing: 0,
    overflowMissingCount: 0,
    hitRatePct: 0,
  });
  assert.deepEqual(router.getMissingReport(), []);

  const after = router.resolve('0x1111222233334444', 'compute');
  assert.equal(after.source, 'builtin');
  assert.equal(router.getStats().totalLookups, 1);
  assert.equal(router.getStats().uniqueMissing, 1);
});

test('BigInt and hex forms of the same hash share pack hits and missing telemetry', () => {
  const value = 0x1234567890abcdefn;
  const hex = formatShaderHash(value);
  const router = new ShaderFallbackRouter({
    packLoader: createMapPackLoader({ [hex]: PACK_FS_WGSL }),
  });

  assert.equal(router.resolve(value, 'fragment').source, 'pack');
  assert.equal(router.resolve(hex, 'fragment').source, 'pack');
  assert.equal(router.resolve(hex.toUpperCase().replace('0X', '0x'), 'fragment').source, 'pack');

  const missRouter = new ShaderFallbackRouter({ now: () => 7 });
  missRouter.resolve(value, 'vertex');
  missRouter.resolve(hex, 'vertex');
  missRouter.resolve('1234567890abcdef', 'vertex');
  const report = missRouter.getMissingReport();
  assert.equal(report.length, 1);
  assert.equal(report[0].hash, hex);
  assert.equal(report[0].hitCount, 3);
});

test('constructor validates options and setPackLoader / setFallbackHashes update resolution', () => {
  assert.throws(() => new ShaderFallbackRouter('nope'), /options must be an object/);
  assert.throws(() => new ShaderFallbackRouter({ maxTrackedMissing: -1 }), /maxTrackedMissing/);
  assert.throws(() => new ShaderFallbackRouter({ maxTrackedMissing: 1.5 }), /maxTrackedMissing/);

  const router = new ShaderFallbackRouter();
  let hit = router.resolve('0x0101010101010101', 'vertex');
  assert.equal(hit.source, 'builtin');

  router.setPackLoader(
    createGetSourcePackLoader({
      [S9_FALLBACK_HASHES.vertex]: S9_FALLBACK_WGSL.vertex,
    })
  );
  router.setFallbackHashes({ vertex: S9_FALLBACK_HASHES.vertex });
  hit = router.resolve('0x0202020202020202', 'vertex');
  assert.equal(hit.source, 'fallback');
  assert.equal(hit.wgsl, S9_FALLBACK_WGSL.vertex);

  router.setFallbackHashes(null);
  hit = router.resolve('0x0303030303030303', 'vertex');
  assert.equal(hit.source, 'builtin');
});
