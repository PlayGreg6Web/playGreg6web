import test from 'node:test';
import assert from 'node:assert/strict';

import {
  WGPK_MAGIC,
  WGPK_VERSION,
  WGPK_HEADER_BYTES,
  WGPK_ENTRY_BYTES,
  SHADER_STAGE,
  STAGE_NAMES,
  fnv1a64,
  normalizeHash,
  formatHashHex,
  buildWgslPackBuffer,
  parseWgslPack,
  WgslPack,
  WgslPackCache,
} from '../web/workers/wgsl_pack_loader.js';

const SAMPLE_VS_WGSL =
  '@vertex fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(pos, 1.0); }';
const SAMPLE_FS_WGSL =
  '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.25, 0.5, 1.0); }';
const SAMPLE_CS_WGSL = '@compute @workgroup_size(1) fn main() {}';

function createMockGpuDevice(options = {}) {
  const calls = [];
  let nextHandle = 1;
  let activeCompilations = 0;
  let peakCompilations = 0;
  const compileDelayMs = options.compileDelayMs || 0;
  const withCompilationInfo = Boolean(options.withCompilationInfo);

  const device = {
    createShaderModule(desc) {
      const mod = {
        kind: 'GPUShaderModule',
        handle: nextHandle++,
        label: desc.label,
        code: desc.code,
      };
      if (withCompilationInfo) {
        mod.getCompilationInfo = async () => {
          activeCompilations += 1;
          if (activeCompilations > peakCompilations) {
            peakCompilations = activeCompilations;
          }
          await new Promise((resolve) => setTimeout(resolve, compileDelayMs));
          activeCompilations -= 1;
          return { messages: [] };
        };
      }
      calls.push({ op: 'createShaderModule', desc: { ...desc }, result: mod });
      return mod;
    },
  };

  return {
    device,
    calls,
    getPeakCompilations: () => peakCompilations,
  };
}

test('constants match shaders/src/pack.rs and shaders/src/convert.rs', () => {
  assert.equal(WGPK_MAGIC, 'WGPK');
  assert.equal(WGPK_VERSION, 1);
  assert.equal(WGPK_HEADER_BYTES, 16);
  assert.equal(WGPK_ENTRY_BYTES, 20);
  assert.deepEqual(SHADER_STAGE, {
    VERTEX: 1,
    FRAGMENT: 2,
    COMPUTE: 3,
  });
  assert.deepEqual(STAGE_NAMES, {
    1: 'vertex',
    2: 'fragment',
    3: 'compute',
  });
});

test('fnv1a64 and formatHashHex match exact test vectors from shaders/src/convert.rs', () => {
  const enc = new TextEncoder();

  // Exact vectors from shaders/src/convert.rs::tests::fnv1a_64_matches_standard_test_vectors
  assert.equal(fnv1a64(new Uint8Array(0)), 0xcbf29ce484222325n);
  assert.equal(fnv1a64(enc.encode('a')), 0xaf63dc4c8601ec8cn);
  assert.equal(fnv1a64(enc.encode('foobar')), 0x85944171f73967e8n);

  // String, ArrayBuffer, and sliced Uint8Array inputs produce identical hashes
  assert.equal(fnv1a64(''), 0xcbf29ce484222325n);
  assert.equal(fnv1a64('a'), 0xaf63dc4c8601ec8cn);
  assert.equal(fnv1a64('foobar'), 0x85944171f73967e8n);

  const padded = new Uint8Array([0xaa, 0xbb, ...enc.encode('foobar'), 0xcc]);
  const slice = padded.subarray(2, 8);
  assert.equal(fnv1a64(slice), 0x85944171f73967e8n);
  assert.equal(fnv1a64(enc.encode('foobar').buffer), 0x85944171f73967e8n);

  // formatHashHex canonical 0x + 16 lowercase hex digits
  assert.equal(formatHashHex(0xdeadbeefn), '0x00000000deadbeef');
  assert.equal(formatHashHex(0xdeadbeef), '0x00000000deadbeef');
  assert.equal(formatHashHex('0xDEADBEEF'), '0x00000000deadbeef');
  assert.equal(formatHashHex('deadbeef'), '0x00000000deadbeef');
  assert.equal(formatHashHex(0n), '0x0000000000000000');
  assert.equal(formatHashHex(0xffffffffffffffffn), '0xffffffffffffffff');

  assert.equal(normalizeHash('0x00000000deadbeef'), 0xdeadbeefn);
  assert.throws(() => normalizeHash(-1n), RangeError);
  assert.throws(() => normalizeHash(1n << 64n), RangeError);
  assert.throws(() => normalizeHash('0xnothex'), RangeError);
  assert.throws(() => normalizeHash('0x1234567890abcdef0'), RangeError);
});

test('binary fixture matches shaders/src/pack.rs byte-for-byte', () => {
  // Exact byte layout produced by shaders/src/pack.rs::write_pack for:
  // hash = 0x1122_3344_5566_7788, stage = Compute (3), flags = 0,
  // wgsl = "@compute @workgroup_size(1) fn main() {}" (40 ASCII bytes)
  const payloadBytes = new TextEncoder().encode(SAMPLE_CS_WGSL);
  assert.equal(payloadBytes.byteLength, 40);

  const expectedBytes = new Uint8Array([
    // Header (16 bytes): "WGPK", version=1 LE, entry_count=1 LE, flags=0 LE
    0x57, 0x47, 0x50, 0x4b,
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    // Entry 0 (20 bytes): hash=0x1122334455667788 LE, stage=3, reserved=[0,0,0], offset=36, length=40
    0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
    0x03, 0x00, 0x00, 0x00,
    0x24, 0x00, 0x00, 0x00,
    0x28, 0x00, 0x00, 0x00,
    // Payload (40 bytes)
    ...payloadBytes,
  ]);
  assert.equal(expectedBytes.byteLength, 76);

  const built = buildWgslPackBuffer([
    {
      hash: 0x1122334455667788n,
      stage: SHADER_STAGE.COMPUTE,
      wgsl: SAMPLE_CS_WGSL,
    },
  ]);
  assert.deepEqual(new Uint8Array(built), expectedBytes);

  const pack = parseWgslPack(expectedBytes);
  assert.ok(pack instanceof WgslPack);
  assert.equal(pack.version, 1);
  assert.equal(pack.flags, 0);
  assert.equal(pack.size, 1);
  assert.ok(pack.entries instanceof Map);
  assert.equal(pack.has(0x1122334455667788n), true);
  assert.equal(pack.has('0x1122334455667788'), true);
  assert.equal(pack.has('1122334455667788'), true);
  assert.equal(pack.has(0x9999n), false);

  const entry = pack.getByHash(0x1122334455667788n);
  assert.deepEqual(entry, {
    hash: 0x1122334455667788n,
    hashHex: '0x1122334455667788',
    stage: 'compute',
    stageCode: 3,
    offset: 36,
    length: 40,
    wgsl: SAMPLE_CS_WGSL,
  });
  assert.equal(pack.getSource('0x1122334455667788'), SAMPLE_CS_WGSL);
  assert.equal(pack.getSource(0xdeadbeefn), undefined);
});

test('parseWgslPack handles multiple stages, UTF-8 multibyte text, non-zero byteOffset, and lazyDecode vs eagerDecode', () => {
  const utf8Fragment = `// 日本語コメント & rocket 🚀\n${SAMPLE_FS_WGSL}`;
  const vsHash = fnv1a64('synthetic-vs-spv');
  const fsHash = fnv1a64('synthetic-fs-spv');
  const csHash = fnv1a64('synthetic-cs-spv');

  const packBuf = buildWgslPackBuffer(
    [
      { hash: vsHash, stage: 'vertex', wgsl: SAMPLE_VS_WGSL },
      { hash: fsHash, stage: SHADER_STAGE.FRAGMENT, wgsl: utf8Fragment },
      { hash: csHash, stage: 'compute', wgsl: SAMPLE_CS_WGSL },
    ],
    { flags: 0x0000_0005 }
  );

  // Set non-zero reserved bytes on entry 0 to confirm forward compatibility (ignored on read).
  const packBytes = new Uint8Array(packBuf);
  packBytes[WGPK_HEADER_BYTES + 9] = 0xab;
  packBytes[WGPK_HEADER_BYTES + 10] = 0xcd;
  packBytes[WGPK_HEADER_BYTES + 11] = 0xef;

  // Embed inside a larger buffer at a non-zero offset to test view slicing.
  const outer = new Uint8Array(32 + packBytes.byteLength + 16);
  outer.set(packBytes, 32);
  const subView = outer.subarray(32, 32 + packBytes.byteLength);

  // 1. Eager decode (default)
  const eagerPack = parseWgslPack(subView);
  assert.equal(eagerPack.flags, 5);
  assert.equal(eagerPack.lazyDecode, false);
  const eagerStats = eagerPack.stats();
  assert.equal(eagerStats.entryCount, 3);
  assert.equal(eagerStats.decodedCount, 3);
  assert.deepEqual(eagerStats.stageCounts, { vertex: 1, fragment: 1, compute: 1 });
  assert.equal(eagerPack.getSource(vsHash), SAMPLE_VS_WGSL);
  assert.equal(eagerPack.getSource(fsHash), utf8Fragment);
  assert.equal(eagerPack.getSource(csHash), SAMPLE_CS_WGSL);

  // Verify UTF-8 byte length (multibyte chars > JS string length)
  const fsEntry = eagerPack.getByHash(fsHash);
  assert.equal(fsEntry.length, new TextEncoder().encode(utf8Fragment).byteLength);
  assert.ok(fsEntry.length > utf8Fragment.length);

  // 2. Lazy decode
  const lazyPack = parseWgslPack(subView, { lazyDecode: true });
  assert.equal(lazyPack.lazyDecode, true);
  assert.equal(lazyPack.stats().decodedCount, 0);

  // Listing entries and checking existence does NOT decode strings
  const listed = lazyPack.listEntries();
  assert.equal(listed.length, 3);
  assert.equal(listed[0].stage, 'vertex');
  assert.equal(listed[1].stage, 'fragment');
  assert.equal(listed[2].stage, 'compute');
  assert.equal(lazyPack.has(fsHash), true);
  assert.equal(lazyPack.stats().decodedCount, 0);

  // Reading one shader decodes only that shader, and caches the string on subsequent reads
  assert.equal(lazyPack.getSource(fsHash), utf8Fragment);
  assert.equal(lazyPack.stats().decodedCount, 1);
  assert.equal(lazyPack.getByHash(fsHash).wgsl, utf8Fragment);
  assert.equal(lazyPack.stats().decodedCount, 1);

  // Reading a second shader increments decodedCount to 2
  assert.equal(lazyPack.getSource(csHash), SAMPLE_CS_WGSL);
  assert.equal(lazyPack.stats().decodedCount, 2);
});

test('parseWgslPack supports empty packs (0 entries), empty payloads, and SharedArrayBuffer', () => {
  const emptyBuf = buildWgslPackBuffer([]);
  assert.equal(emptyBuf.byteLength, WGPK_HEADER_BYTES);
  const emptyPack = parseWgslPack(emptyBuf);
  assert.equal(emptyPack.size, 0);
  assert.deepEqual(emptyPack.listEntries(), []);
  assert.equal(emptyPack.stats().payloadBytes, 0);

  const zeroPayloadBuf = buildWgslPackBuffer([
    { hash: 1n, stage: SHADER_STAGE.VERTEX, wgsl: '' },
    { hash: 2n, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);
  const zeroPack = parseWgslPack(zeroPayloadBuf);
  assert.equal(zeroPack.getSource(1n), '');
  assert.equal(zeroPack.getSource(2n), SAMPLE_FS_WGSL);

  if (typeof SharedArrayBuffer !== 'undefined') {
    const src = new Uint8Array(zeroPayloadBuf);
    const sab = new SharedArrayBuffer(src.byteLength);
    new Uint8Array(sab).set(src);
    const sabPack = parseWgslPack(sab, { lazyDecode: true });
    assert.equal(sabPack.getSource(2n), SAMPLE_FS_WGSL);
  }
});

test('parseWgslPack rejects malformed, truncated, out-of-bounds, overlapping, and invalid UTF-8 inputs', () => {
  const validBuf = buildWgslPackBuffer([
    { hash: 0x1122334455667788n, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
    { hash: 0x8877665544332211n, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
  ]);
  const validBytes = new Uint8Array(validBuf);

  // Non-buffer input
  assert.throws(() => parseWgslPack(null), TypeError);
  assert.throws(() => parseWgslPack('WGPK'), TypeError);

  // Truncated header
  assert.throws(() => parseWgslPack(validBytes.subarray(0, 8)), /truncated/i);
  assert.throws(() => parseWgslPack(new Uint8Array(0)), /truncated/i);

  // Bad magic
  const badMagic = validBytes.slice();
  badMagic.set(new TextEncoder().encode('NOPE'), 0);
  assert.throws(() => parseWgslPack(badMagic), /invalid wgpk pack magic/i);

  // Unsupported version
  const badVersion = validBytes.slice();
  new DataView(badVersion.buffer).setUint32(4, 99, true);
  assert.throws(() => parseWgslPack(badVersion), /unsupported wgpk pack version 99/i);

  // Truncated entry table (claims 2 entries, only header + 1 entry bytes exist)
  const truncatedTable = validBytes.slice(0, WGPK_HEADER_BYTES + WGPK_ENTRY_BYTES);
  assert.throws(() => parseWgslPack(truncatedTable), /truncated/i);

  // Invalid stage code (0 and 9)
  const badStage0 = validBytes.slice();
  badStage0[WGPK_HEADER_BYTES + 8] = 0;
  assert.throws(() => parseWgslPack(badStage0), /invalid wgpk shader stage value 0/i);

  const badStage9 = validBytes.slice();
  badStage9[WGPK_HEADER_BYTES + 8] = 9;
  assert.throws(() => parseWgslPack(badStage9), /invalid wgpk shader stage value 9/i);

  // Offset pointing inside header/table (< dataStart)
  const headerOverlap = validBytes.slice();
  new DataView(headerOverlap.buffer).setUint32(WGPK_HEADER_BYTES + 12, 4, true);
  assert.throws(() => parseWgslPack(headerOverlap), /out of bounds/i);

  // Out-of-bounds offset
  const oobOffset = validBytes.slice();
  new DataView(oobOffset.buffer).setUint32(WGPK_HEADER_BYTES + 12, 99999, true);
  assert.throws(() => parseWgslPack(oobOffset), /out of bounds/i);

  // Out-of-bounds length (extends past buffer end)
  const oobLength = validBytes.slice();
  new DataView(oobLength.buffer).setUint32(WGPK_HEADER_BYTES + 16, 99999, true);
  assert.throws(() => parseWgslPack(oobLength), /out of bounds/i);

  // uint32 overflow on offset + length
  const overflowRange = validBytes.slice();
  const overflowView = new DataView(overflowRange.buffer);
  overflowView.setUint32(WGPK_HEADER_BYTES + 12, 0xffffffff, true);
  overflowView.setUint32(WGPK_HEADER_BYTES + 16, 16, true);
  assert.throws(() => parseWgslPack(overflowRange), /out of bounds/i);

  // Overlapping payload ranges between entry 0 and entry 1
  const overlapEntries = validBytes.slice();
  const overlapView = new DataView(overlapEntries.buffer);
  const entry0Offset = overlapView.getUint32(WGPK_HEADER_BYTES + 12, true);
  // Point entry 1 into the middle of entry 0's payload
  overlapView.setUint32(WGPK_HEADER_BYTES + WGPK_ENTRY_BYTES + 12, entry0Offset + 8, true);
  overlapView.setUint32(WGPK_HEADER_BYTES + WGPK_ENTRY_BYTES + 16, 16, true);
  assert.throws(() => parseWgslPack(overlapEntries), /overlapping payload/i);

  // Invalid UTF-8 bytes inside payload
  const badUtf8 = validBytes.slice();
  const dataStart = WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES;
  badUtf8[dataStart] = 0xff;
  assert.throws(() => parseWgslPack(badUtf8), /not valid utf-8/i);

  // In lazyDecode mode, parsing succeeds and accessing the corrupted entry throws
  const lazyBadUtf8 = parseWgslPack(badUtf8, { lazyDecode: true });
  assert.equal(lazyBadUtf8.getSource(0x8877665544332211n), SAMPLE_VS_WGSL);
  assert.throws(() => lazyBadUtf8.getSource(0x1122334455667788n), /not valid utf-8/i);
});

test('WgslPackCache deduplicates GPUShaderModule compilation, tracks stats, and isolates devices', () => {
  const hVs = fnv1a64('shader-vs-1');
  const hFs = fnv1a64('shader-fs-1');
  const hCs = fnv1a64('shader-cs-1');

  const pack1 = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);
  const pack2 = buildWgslPackBuffer([
    { hash: hCs, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
  ]);

  const cache = new WgslPackCache({ lazyDecode: true });
  cache.loadPackBuffer(pack1, 'base.wgsl.pack');
  cache.loadPackBuffer(pack2, 'compute.wgsl.pack');

  assert.equal(cache.hasShader(hVs), true);
  assert.equal(cache.hasShader(formatHashHex(hCs)), true);
  assert.equal(cache.getShaderSource(hFs), SAMPLE_FS_WGSL);
  assert.equal(cache.getShaderSource(0x1234n), undefined);

  const mock1 = createMockGpuDevice();
  const mock2 = createMockGpuDevice();

  // First lookup compiles and caches
  const modVs1 = cache.getOrCreateShaderModule(mock1.device, hVs, 'gta6');
  assert.equal(mock1.calls.length, 1);
  assert.equal(mock1.calls[0].desc.code, SAMPLE_VS_WGSL);
  assert.equal(mock1.calls[0].desc.label, `gta6:vertex:${formatHashHex(hVs)}`);

  // Second lookup (even via hex string) returns identical cached GPUShaderModule
  const modVs2 = cache.getOrCreateShaderModule(mock1.device, formatHashHex(hVs), 'gta6');
  assert.equal(modVs2, modVs1);
  assert.equal(mock1.calls.length, 1);

  // Different device compiles its own module instead of returning mock1's module
  const modVsOtherDevice = cache.getOrCreateShaderModule(mock2.device, hVs);
  assert.notEqual(modVsOtherDevice, modVs1);
  assert.equal(mock2.calls.length, 1);
  assert.equal(mock2.calls[0].desc.label, `wgpk:vertex:${formatHashHex(hVs)}`);

  // Unknown shader throws and counts as a miss without incrementing compiles
  assert.throws(
    () => cache.getOrCreateShaderModule(mock1.device, 0xdeadbeefn),
    /not found in any loaded wgsl pack/i
  );

  const stats = cache.getStats();
  assert.equal(stats.packCount, 2);
  assert.equal(stats.shaderCount, 3);
  assert.equal(stats.cachedModules, 2);
  assert.equal(stats.hits, 1);
  assert.equal(stats.misses, 3);
  assert.equal(stats.compiles, 2);
  assert.equal(stats.evictions, 0);
  assert.equal(stats.hitRate, 1 / 4);
  assert.equal(stats.totalPackBytes, pack1.byteLength + pack2.byteLength);

  // Evict removes cached modules across devices and forces recompilation on next access
  assert.equal(cache.evict(hVs), true);
  assert.equal(cache.evict(hVs), false);
  assert.equal(cache.getStats().cachedModules, 0);
  assert.equal(cache.getStats().evictions, 2);

  const modVsRecompiled = cache.getOrCreateShaderModule(mock1.device, hVs);
  assert.notEqual(modVsRecompiled, modVs1);
  assert.equal(mock1.calls.length, 2);

  // Evict with removeSource drops the pack entry too
  assert.equal(cache.evict(hCs, { removeSource: true }), true);
  assert.equal(cache.hasShader(hCs), false);

  // Clear empties packs, shaders, and cached modules
  cache.clear({ resetStats: true });
  const cleared = cache.getStats();
  assert.equal(cleared.packCount, 0);
  assert.equal(cleared.shaderCount, 0);
  assert.equal(cleared.cachedModules, 0);
  assert.equal(cleared.hits, 0);
  assert.equal(cleared.misses, 0);
  assert.equal(cleared.compiles, 0);
});

test('WgslPackCache.prewarm batch-compiles with bounded concurrency and skips cached shaders', async () => {
  const entries = [];
  for (let i = 0; i < 10; i += 1) {
    const wgsl = `@compute @workgroup_size(${i + 1}) fn main() {}`;
    entries.push({
      hash: fnv1a64(`compute-shader-${i}`),
      stage: SHADER_STAGE.COMPUTE,
      wgsl,
    });
  }

  const cache = new WgslPackCache();
  cache.loadPackBuffer(buildWgslPackBuffer(entries), 'batch.wgsl.pack');

  const mock = createMockGpuDevice({ withCompilationInfo: true, compileDelayMs: 5 });

  // Pre-compile entry 0 first
  const preExisting = cache.getOrCreateShaderModule(mock.device, entries[0].hash);

  // Prewarm first 8 hashes (with a duplicate) at concurrency = 3
  const subset = [
    entries[0].hash,
    entries[1].hash,
    entries[2].hash,
    entries[2].hash, // duplicate should be deduplicated
    entries[3].hash,
    entries[4].hash,
    entries[5].hash,
    entries[6].hash,
    entries[7].hash,
  ];
  const res = await cache.prewarm(mock.device, subset, { concurrency: 3, labelPrefix: 'warm' });
  assert.equal(res.total, 8);
  assert.equal(res.cached, 1);
  assert.equal(res.compiled, 7);
  assert.equal(res.modules.get(entries[0].hash), preExisting);
  assert.equal(mock.calls.length, 8);
  assert.ok(mock.getPeakCompilations() <= 3, `expected peak concurrency <= 3, got ${mock.getPeakCompilations()}`);
  assert.ok(mock.getPeakCompilations() >= 2, `expected concurrent execution, got peak ${mock.getPeakCompilations()}`);

  // Prewarming without an explicit hash list compiles the remaining 2 shaders
  const resAll = await cache.prewarm(mock.device, undefined, { concurrency: 2 });
  assert.equal(resAll.total, 10);
  assert.equal(resAll.cached, 8);
  assert.equal(resAll.compiled, 2);
  assert.equal(mock.calls.length, 10);
});

test('WgslPackCache supports loadPackUrl, pack replacement, and shader-index.json metadata', async () => {
  const hVs = 0x00000000deadbeefn;
  const hFs = 0xfeedfacecafebaben;

  const packV1 = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);

  const shaderIndex = {
    abi_version: 1,
    pack: 'shaders.wgsl.pack',
    shaders: {
      '0x00000000deadbeef': {
        stage: 'vertex',
        wgsl: 'shaders/00000000deadbeef.wgsl',
        entry_point: 'vs_main',
        pack_offset: WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES,
        pack_length: new TextEncoder().encode(SAMPLE_VS_WGSL).byteLength,
      },
      '0xfeedfacecafebabe': {
        stage: 'fragment',
        wgsl: 'shaders/feedfacecafebabe.wgsl',
        entry_point: 'fs_main',
      },
    },
  };

  const fetchedUrls = [];
  const mockFetch = async (url) => {
    fetchedUrls.push(url);
    if (url === '/assets/shaders.wgsl.pack') {
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        async arrayBuffer() {
          return packV1;
        },
      };
    }
    if (url === '/assets/shader-index.json') {
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        async json() {
          return shaderIndex;
        },
      };
    }
    return {
      ok: false,
      status: 404,
      statusText: 'Not Found',
      async arrayBuffer() {
        return new ArrayBuffer(0);
      },
    };
  };

  const cache = new WgslPackCache();
  const loadedPack = await cache.loadPackUrl('/assets/shaders.wgsl.pack', mockFetch);
  assert.equal(loadedPack.size, 2);

  const indexInfo = await cache.loadShaderIndexUrl('/assets/shader-index.json', mockFetch);
  assert.deepEqual(indexInfo, {
    abiVersion: 1,
    pack: 'shaders.wgsl.pack',
    count: 2,
  });
  assert.deepEqual(fetchedUrls, ['/assets/shaders.wgsl.pack', '/assets/shader-index.json']);

  const meta = cache.getShaderMetadata(hVs);
  assert.equal(meta.hash, hVs);
  assert.equal(meta.hashHex, '0x00000000deadbeef');
  assert.equal(meta.stage, 'vertex');
  assert.equal(meta.stageCode, 1);
  assert.equal(meta.entryPoint, 'vs_main');
  assert.equal(meta.wgslPath, 'shaders/00000000deadbeef.wgsl');
  assert.equal(meta.packName, '/assets/shaders.wgsl.pack');

  // HTTP 404 rejects with clear error
  await assert.rejects(
    () => cache.loadPackUrl('/assets/missing.wgsl.pack', mockFetch),
    /HTTP 404 Not Found/
  );

  // Unsupported shader-index.json abi_version rejects
  assert.throws(
    () => cache.loadShaderIndex({ abi_version: 2, pack: null, shaders: {} }),
    /unsupported shader-index\.json abi_version 2/i
  );

  // Reloading the same pack name with updated WGSL invalidates stale compiled modules
  const mock = createMockGpuDevice();
  const modOld = cache.getOrCreateShaderModule(mock.device, hVs);
  const updatedVs = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const packV2 = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: updatedVs },
  ]);
  cache.loadPackBuffer(packV2, '/assets/shaders.wgsl.pack');
  assert.equal(cache.hasShader(hFs), false);
  assert.equal(cache.getShaderSource(hVs), updatedVs);

  const modNew = cache.getOrCreateShaderModule(mock.device, hVs);
  assert.notEqual(modNew, modOld);
  assert.equal(modNew.code, updatedVs);
});
