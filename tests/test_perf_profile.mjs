import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const WORKERS_DIR = fileURLToPath(new URL('../web/workers/', import.meta.url));

const perfProfileMod = require('../web/workers/perf_profile.js');
const rtPoolMod = require('../web/workers/rt_pool.js');
const registryMod = require('../web/workers/resource_registry.js');
const ioWorkerMod = require('../web/workers/io_worker.js');

const {
  MiB,
  GiB,
  PERF_PROFILES,
  PERF_PROFILE_NAMES,
  parsePerfProfileFromQuery,
  classifyAdapterInfo,
  resolvePerfProfile,
  coercePerfProfile,
} = perfProfileMod;

const { RenderTargetPool, DEFAULT_MAX_POOL_BYTES, DEFAULT_IDLE_FRAME_TTL } = rtPoolMod;
const {
  GpuResourceRegistry,
  RESOURCE_KIND,
  VRAM_PRESSURE_TIER,
  DEFAULT_FRAMES_IN_FLIGHT,
  DEFAULT_SOFT_BUDGET_BYTES,
} = registryMod;
const {
  ImmutableHttpfsCache,
  IoWorkerController,
  createIoWorkerHandler,
  DEFAULT_SLAB_MAX_BYTES,
  DEFAULT_MAX_CONCURRENT_SLAB_FETCHES,
} = ioWorkerMod;

function createMockGpuDevice() {
  let nextTexId = 1;
  let nextViewId = 1;
  const createdTextures = [];
  const device = {
    createdTextures,
    createTexture(descriptor) {
      const texId = nextTexId++;
      const texture = {
        mockId: texId,
        label: descriptor.label,
        descriptor: { ...descriptor, size: { ...descriptor.size } },
        destroyed: false,
        destroyCount: 0,
        createView(viewDesc) {
          return {
            kind: 'GPUTextureView',
            viewId: nextViewId++,
            textureMockId: texId,
            viewDesc: viewDesc || null,
          };
        },
        destroy() {
          this.destroyed = true;
          this.destroyCount += 1;
        },
      };
      createdTextures.push(texture);
      return texture;
    },
  };
  return { device, createdTextures };
}

function createDestroyableMock(name) {
  return {
    name,
    destroyCount: 0,
    destroy() {
      this.destroyCount += 1;
    },
  };
}

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createRangeFetch(urlBytesMap) {
  return async function fetchImpl(url, init = {}) {
    const data = urlBytesMap.get(url);
    if (!data) {
      return {
        ok: false,
        status: 404,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
        async json() {
          throw new Error('404');
        },
      };
    }
    const headers = new Map();
    const rangeHeader = init && init.headers && (init.headers.Range || init.headers.range);
    if (!rangeHeader) {
      headers.set('Content-Length', String(data.byteLength));
      return {
        ok: true,
        status: 200,
        headers,
        async arrayBuffer() {
          return data.slice().buffer;
        },
        async json() {
          return JSON.parse(new TextDecoder().decode(data));
        },
      };
    }
    const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
    const start = Number(m[1]);
    const end = Math.min(Number(m[2]), data.byteLength - 1);
    const slice = data.subarray(start, end + 1);
    headers.set('Content-Range', `bytes ${start}-${end}/${data.byteLength}`);
    headers.set('Content-Length', String(slice.byteLength));
    return {
      ok: true,
      status: 206,
      headers,
      async arrayBuffer() {
        return slice.slice().buffer;
      },
    };
  };
}

// ---------------------------------------------------------------------------
// 1. Module exports, classic-worker isolation, and concrete preset values
// ---------------------------------------------------------------------------

test('UMD export attaches PlayGta6PerfProfile and loads cleanly under classic-worker importScripts', () => {
  assert.equal(globalThis.PlayGta6PerfProfile, perfProfileMod);
  assert.ok(Object.isFrozen(perfProfileMod));
  assert.ok(Object.isFrozen(PERF_PROFILES));
  assert.deepEqual(PERF_PROFILE_NAMES, ['low', 'balanced', 'high']);

  // Verify classic worker IIFE isolation (loading twice without top-level collisions).
  const sandbox = { console, URLSearchParams };
  const ctx = vm.createContext(sandbox);
  ctx.self = ctx;
  ctx.importScripts = (...urls) => {
    for (const rel of urls) {
      const fullPath = path.resolve(WORKERS_DIR, rel);
      const src = readFileSync(fullPath, 'utf8');
      vm.runInContext(src, ctx, { filename: fullPath });
    }
  };
  assert.doesNotThrow(() => {
    ctx.importScripts('./perf_profile.js');
    ctx.importScripts('./perf_profile.js');
  });
  assert.equal(typeof ctx.PlayGta6PerfProfile.resolvePerfProfile, 'function');
  assert.equal(ctx.PlayGta6PerfProfile.PERF_PROFILES.low.rtPoolMaxBytes, 128 * MiB);
});

test('PERF_PROFILES defines exact bounded parameters for low, balanced, and high', () => {
  assert.deepEqual(PERF_PROFILES.low, {
    name: 'low',
    rtPoolMaxBytes: 128 * MiB,
    rtPoolMaxIdleFrames: 8,
    vramBudgetBytes: 1536 * MiB,
    vramReclaimTriggerRatio: 0.72,
    vramReclaimTargetRatio: 0.58,
    gcDeferFrames: 2,
    ioSlabCacheBytes: 64 * MiB,
    ioMaxConcurrentLanes: 4,
    textureMipCap: 512,
    dprCap: 1.0,
    prewarmMaxInFlight: 1,
    shaderCompileConcurrency: 1,
  });

  assert.deepEqual(PERF_PROFILES.balanced, {
    name: 'balanced',
    rtPoolMaxBytes: 256 * MiB,
    rtPoolMaxIdleFrames: 16,
    vramBudgetBytes: 3072 * MiB,
    vramReclaimTriggerRatio: 0.78,
    vramReclaimTargetRatio: 0.65,
    gcDeferFrames: 2,
    ioSlabCacheBytes: 128 * MiB,
    ioMaxConcurrentLanes: 6,
    textureMipCap: 1024,
    dprCap: 1.0,
    prewarmMaxInFlight: 2,
    shaderCompileConcurrency: 2,
  });

  assert.deepEqual(PERF_PROFILES.high, {
    name: 'high',
    rtPoolMaxBytes: 512 * MiB,
    rtPoolMaxIdleFrames: 30,
    vramBudgetBytes: 6144 * MiB,
    vramReclaimTriggerRatio: 0.85,
    vramReclaimTargetRatio: 0.72,
    gcDeferFrames: 3,
    ioSlabCacheBytes: 192 * MiB,
    ioMaxConcurrentLanes: 8,
    textureMipCap: 0,
    dprCap: 2.0,
    prewarmMaxInFlight: 4,
    shaderCompileConcurrency: 4,
  });
});

// ---------------------------------------------------------------------------
// 2. Auto-detection across hardware classes
// ---------------------------------------------------------------------------

test('auto detection classifies low-end laptop, mid-range PC, and high-end workstation accurately', () => {
  // Low-end laptop: 4 cores / 8 GB RAM
  const lowLaptop = resolvePerfProfile({
    preset: 'auto',
    deviceMemoryGb: 8,
    hardwareConcurrency: 4,
    adapterLimits: { maxBufferSize: 512 * MiB, maxTextureDimension2D: 8192 },
  });
  assert.equal(lowLaptop.name, 'low');
  assert.equal(lowLaptop.source, 'auto');
  assert.equal(lowLaptop.rtPoolMaxBytes, 128 * MiB);
  assert.equal(lowLaptop.ioSlabCacheBytes, 64 * MiB);
  assert.equal(lowLaptop.ioMaxConcurrentLanes, 4);
  assert.ok(lowLaptop.reasons.includes('deviceMemoryGb<=8'));
  assert.ok(lowLaptop.reasons.includes('hardwareConcurrency<=4'));

  // Software / SwiftShader adapter forces low even on a high-core/high-RAM host
  const swiftShader = resolvePerfProfile({
    preset: 'auto',
    deviceMemoryGb: 32,
    hardwareConcurrency: 24,
    adapterLimits: { maxBufferSize: 2 * GiB, maxTextureDimension2D: 16384 },
    adapterInfo: { description: 'Google SwiftShader WebGPU' },
  });
  assert.equal(swiftShader.name, 'low');
  assert.equal(swiftShader.hardware.isSoftwareAdapter, true);
  assert.ok(swiftShader.reasons.includes('softwareAdapter'));

  // Fallback adapter flag forces low
  const fallbackAdapter = resolvePerfProfile({
    preset: 'auto',
    deviceMemoryGb: 32,
    hardwareConcurrency: 24,
    adapterInfo: { isFallbackAdapter: true },
  });
  assert.equal(fallbackAdapter.name, 'low');

  // Small maxBufferSize (<= 256 MiB) forces low
  const tinyBufferLimit = resolvePerfProfile({
    preset: 'auto',
    deviceMemoryGb: 32,
    hardwareConcurrency: 16,
    adapterLimits: { maxBufferSize: 256 * MiB },
  });
  assert.equal(tinyBufferLimit.name, 'low');
  assert.ok(tinyBufferLimit.reasons.includes('maxBufferSize<=256MiB'));

  // Mid-range PC (16 GB RAM / 12 threads / 1 GiB maxBufferSize — e.g. 6 GB VRAM desktop) -> balanced
  const midRangePc = resolvePerfProfile({
    preset: 'auto',
    deviceMemoryGb: 16,
    hardwareConcurrency: 12,
    adapterLimits: {
      maxBufferSize: 1 * GiB,
      maxStorageBufferBindingSize: 512 * MiB,
      maxTextureDimension2D: 16384,
    },
    adapterInfo: { description: 'Discrete 6GB GPU', architecture: 'desktop-discrete' },
  });
  assert.equal(midRangePc.name, 'balanced');
  assert.equal(midRangePc.source, 'auto');
  assert.equal(midRangePc.vramBudgetBytes, 3072 * MiB);
  assert.equal(midRangePc.ioSlabCacheBytes, 128 * MiB);
  assert.equal(midRangePc.ioMaxConcurrentLanes, 6);
  assert.equal(midRangePc.textureMipCap, 1024);
  assert.equal(midRangePc.dprCap, 1.0);

  // Integrated / low-power adapter on a 32 GB laptop caps auto to balanced (not high)
  const integratedLaptop = resolvePerfProfile({
    preset: 'auto',
    deviceMemoryGb: 32,
    hardwareConcurrency: 16,
    adapterLimits: { maxBufferSize: 2 * GiB, maxTextureDimension2D: 16384 },
    adapterInfo: { type: 'integrated', powerPreference: 'low-power' },
  });
  assert.equal(integratedLaptop.name, 'balanced');
  assert.equal(integratedLaptop.hardware.isLowPowerAdapter, true);
  assert.ok(integratedLaptop.reasons.includes('lowPowerAdapter'));

  // High-end workstation: 32 GB RAM / 24 threads / 2 GiB maxBufferSize -> high
  const workstation = resolvePerfProfile({
    preset: 'auto',
    deviceMemoryGb: 32,
    hardwareConcurrency: 24,
    adapterLimits: {
      maxBufferSize: 2 * GiB,
      maxStorageBufferBindingSize: 1 * GiB,
      maxTextureDimension2D: 16384,
    },
    adapterInfo: { description: 'Desktop Workstation GPU' },
  });
  assert.equal(workstation.name, 'high');
  assert.equal(workstation.source, 'auto');
  assert.equal(workstation.rtPoolMaxBytes, 512 * MiB);
  assert.equal(workstation.vramBudgetBytes, 6144 * MiB);
  assert.equal(workstation.ioSlabCacheBytes, 192 * MiB);
  assert.equal(workstation.ioMaxConcurrentLanes, 8);
  assert.equal(workstation.textureMipCap, 0);
  assert.equal(workstation.dprCap, 2.0);

  // Navigator fallback when explicit signals are omitted
  const fromNavigator = resolvePerfProfile({
    preset: 'auto',
    navigator: { deviceMemory: 8, hardwareConcurrency: 4 },
  });
  assert.equal(fromNavigator.name, 'low');

  // Absent hardware signals fall back safely to balanced
  const noSignals = resolvePerfProfile({ preset: 'auto', navigator: null });
  assert.equal(noSignals.name, 'balanced');
  assert.ok(noSignals.reasons.includes('noHardwareSignals:safeDefault'));

  // classifyAdapterInfo helper sanity checks
  assert.deepEqual(classifyAdapterInfo({ description: 'llvmpipe (LLVM 17)' }), {
    isSoftwareAdapter: true,
    isLowPowerAdapter: false,
  });
  assert.deepEqual(classifyAdapterInfo({ description: 'Integrated GPU' }), {
    isSoftwareAdapter: false,
    isLowPowerAdapter: true,
  });
  assert.deepEqual(classifyAdapterInfo(null), {
    isSoftwareAdapter: false,
    isLowPowerAdapter: false,
  });
});

// ---------------------------------------------------------------------------
// 3. Query-string parsing, explicit preset precedence, and numeric overrides
// ---------------------------------------------------------------------------

test('query-string parsing and explicit/numeric overrides take proper precedence', () => {
  assert.equal(parsePerfProfileFromQuery('?quality=low'), 'low');
  assert.equal(parsePerfProfileFromQuery('?foo=1&quality=BALANCED&bar=2'), 'balanced');
  assert.equal(parsePerfProfileFromQuery('https://localhost:8080/?perfProfile=high#play'), 'high');
  assert.equal(parsePerfProfileFromQuery(new URLSearchParams('quality=auto')), 'auto');
  assert.equal(parsePerfProfileFromQuery('?quality=ultra-mega'), null);
  assert.equal(parsePerfProfileFromQuery('?quality='), null);
  assert.equal(parsePerfProfileFromQuery('not-a-query'), null);
  assert.equal(parsePerfProfileFromQuery(null), null);

  // Query string overrides auto-detection on high-spec hardware
  const queryLow = resolvePerfProfile({
    preset: 'auto',
    queryString: '?quality=low',
    deviceMemoryGb: 64,
    hardwareConcurrency: 32,
  });
  assert.equal(queryLow.name, 'low');
  assert.equal(queryLow.source, 'query');

  // Invalid query string falls back safely to auto-detection
  const invalidQuery = resolvePerfProfile({
    queryString: '?quality=invalid_preset',
    deviceMemoryGb: 32,
    hardwareConcurrency: 24,
  });
  assert.equal(invalidQuery.name, 'high');
  assert.equal(invalidQuery.source, 'auto');

  // Explicit preset overrides both auto hardware detection and query string
  const explicitHigh = resolvePerfProfile({
    preset: 'high',
    queryString: '?quality=low',
    deviceMemoryGb: 4,
    hardwareConcurrency: 2,
    adapterInfo: { description: 'SwiftShader' },
  });
  assert.equal(explicitHigh.name, 'high');
  assert.equal(explicitHigh.source, 'explicit');

  // Individual numeric overrides (both canonical and alias keys) take precedence over preset defaults
  const overridden = resolvePerfProfile({
    preset: 'low',
    slabCacheBytes: 96 * MiB,
    rtPoolMaxBytes: 192 * MiB,
    vramBudgetBytes: 2048 * MiB,
    overrides: {
      gcDeferFrames: 4,
      vramReclaimTriggerRatio: 0.6,
      vramReclaimTargetRatio: 0.8, // clamped to <= triggerRatio
    },
  });
  assert.equal(overridden.name, 'low');
  assert.equal(overridden.ioSlabCacheBytes, 96 * MiB);
  assert.equal(overridden.rtPoolMaxBytes, 192 * MiB);
  assert.equal(overridden.vramBudgetBytes, 2048 * MiB);
  assert.equal(overridden.gcDeferFrames, 4);
  assert.equal(overridden.vramReclaimTriggerRatio, 0.6);
  assert.equal(overridden.vramReclaimTargetRatio, 0.6);
  assert.ok(overridden.overridesApplied.includes('ioSlabCacheBytes'));
  assert.ok(overridden.overridesApplied.includes('rtPoolMaxBytes'));
  assert.ok(overridden.overridesApplied.includes('vramBudgetBytes'));

  // coercePerfProfile validation
  assert.equal(coercePerfProfile('low').name, 'low');
  assert.equal(coercePerfProfile('?quality=balanced').name, 'balanced');
  assert.throws(() => coercePerfProfile('unknown-tier'), RangeError);
  assert.throws(() => coercePerfProfile(null), TypeError);
});

// ---------------------------------------------------------------------------
// 4. RenderTargetPool integration (`rt_pool.js`)
// ---------------------------------------------------------------------------

test('rt_pool.js preserves defaults without perfProfile and applies low profile with immediate idle eviction', () => {
  // Backward compatibility: omitting perfProfile preserves legacy defaults
  const { device: devDefault } = createMockGpuDevice();
  const defaultPool = new RenderTargetPool({ device: devDefault });
  assert.equal(defaultPool.maxPoolBytes, DEFAULT_MAX_POOL_BYTES);
  assert.equal(defaultPool.idleFrameTtl, DEFAULT_IDLE_FRAME_TTL);
  assert.equal(defaultPool.perfProfile, null);
  assert.equal(defaultPool.getStats().perfProfile, null);
  defaultPool.destroy();

  // Constructor perfProfile option + explicit override precedence
  const { device: devCtor } = createMockGpuDevice();
  const balancedPool = new RenderTargetPool({ device: devCtor, perfProfile: 'balanced' });
  assert.equal(balancedPool.maxPoolBytes, 256 * MiB);
  assert.equal(balancedPool.idleFrameTtl, 16);
  assert.equal(balancedPool.getStats().perfProfile, 'balanced');
  balancedPool.destroy();

  const { device: devOverride } = createMockGpuDevice();
  const overridePool = new RenderTargetPool({
    device: devOverride,
    perfProfile: 'low',
    maxPoolBytes: 200 * MiB,
  });
  assert.equal(overridePool.maxPoolBytes, 200 * MiB);
  assert.equal(overridePool.idleFrameTtl, 8);
  overridePool.destroy();

  // Runtime applyPerfProfile('low') tightens cap from 512 MiB -> 128 MiB and evicts excess idle RTs
  const { device, createdTextures } = createMockGpuDevice();
  const pool = new RenderTargetPool({ device, perfProfile: 'high' });
  assert.equal(pool.maxPoolBytes, 512 * MiB);
  assert.equal(pool.idleFrameTtl, 30);

  // Each 2048x2048 rgba16float target = 2048 * 2048 * 8 = 32 MiB.
  // Allocate 5 distinct targets (160 MiB total), keep 1 active (32 MiB) and release 4 idle (128 MiB).
  const formats = ['rgba16float', 'rgba16unorm', 'rgba16uint', 'rgba16sint', 'rg32float'];
  const targets = formats.map((format, idx) =>
    pool.acquire({ width: 2048, height: 2048, format }, `rt-${idx}`)
  );
  // Release first 4 targets -> 128 MiB idle + 32 MiB active = 160 MiB total.
  for (let i = 0; i < 4; i += 1) {
    pool.release(targets[i]);
  }
  assert.equal(pool.getStats().totalBytes, 160 * MiB);
  assert.equal(pool.getStats().idleCount, 4);
  assert.equal(pool.getStats().activeCount, 1);

  // Switch to 'low' (128 MiB cap, 8 idle frames): must evict oldest idle target (32 MiB) so totalBytes <= 128 MiB!
  const applied = pool.applyPerfProfile('low');
  assert.equal(applied.profile, 'low');
  assert.equal(applied.maxPoolBytes, 128 * MiB);
  assert.equal(applied.idleFrameTtl, 8);
  assert.equal(applied.evictedCount, 1);
  assert.equal(applied.evictedBytes, 32 * MiB);

  // Oldest idle target destroyed; active target[4] remains untouched!
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(createdTextures[4].destroyed, false);
  const stats = pool.getStats();
  assert.equal(stats.maxPoolBytes, 128 * MiB);
  assert.equal(stats.idleFrameTtl, 8);
  assert.equal(stats.totalBytes, 128 * MiB);
  assert.equal(stats.activeCount, 1);
  assert.equal(stats.idleCount, 3);
  assert.equal(stats.perfProfile, 'low');

  pool.release(targets[4]);
  pool.destroy();
});

// ---------------------------------------------------------------------------
// 5. GpuResourceRegistry integration (`resource_registry.js`)
// ---------------------------------------------------------------------------

test('resource_registry.js preserves defaults without perfProfile and tunes VRAM pressure & deferred GC on applyPerfProfile', async () => {
  // Backward compatibility: omitting perfProfile preserves legacy defaults
  const defaultReg = new GpuResourceRegistry();
  assert.equal(defaultReg.framesInFlight, DEFAULT_FRAMES_IN_FLIGHT);
  assert.equal(defaultReg.softBudgetBytes, DEFAULT_SOFT_BUDGET_BYTES);
  assert.equal(defaultReg.perfProfile, null);
  assert.equal(defaultReg.getStats().perfProfile, null);

  // Start on 'high' (6144 MiB budget, trigger 0.85, target 0.72, gcDeferFrames 3)
  const pressureEvents = [];
  const reg = new GpuResourceRegistry({
    perfProfile: 'high',
    onPressureChange: (ev) => pressureEvents.push(ev),
  });
  assert.equal(reg.softBudgetBytes, 6144 * MiB);
  assert.equal(reg.framesInFlight, 3);
  assert.equal(reg.vramReclaimTriggerRatio, 0.85);
  assert.equal(reg.vramReclaimTargetRatio, 0.72);
  assert.equal(reg.getVramPressure().tier, VRAM_PRESSURE_TIER.NORMAL);

  // Register 1200 MiB active texture + 100 MiB retired buffer (frame 0) + 50 MiB fence-gated retired buffer
  const bigTex = createDestroyableMock('world-atlas');
  const retiredBuf = createDestroyableMock('old-staging');
  const fencedBuf = createDestroyableMock('fenced-staging');
  const pendingFence = createDeferred();

  reg.beginFrame(0);
  reg.register(RESOURCE_KIND.TEXTURE, 1, bigTex, { byteSize: 1200 * MiB, label: 'atlas' });
  reg.register(RESOURCE_KIND.BUFFER, 2, retiredBuf, { byteSize: 100 * MiB, label: 'old-buf' });
  reg.retire(RESOURCE_KIND.BUFFER, 2);
  reg.endFrame(); // seal retiredBuf as epoch-based on frame 0

  reg.beginFrame(1);
  reg.register(RESOURCE_KIND.BUFFER, 3, fencedBuf, { byteSize: 50 * MiB, label: 'fenced-buf' });
  reg.retire(RESOURCE_KIND.BUFFER, 3);
  reg.endFrame(pendingFence.promise);

  // Advance to frame 2: under high (framesInFlight=3), retiredBuf has elapsed 2 < 3 -> NOT yet destroyed.
  reg.beginFrame(2);
  assert.equal(retiredBuf.destroyCount, 0);
  assert.equal(fencedBuf.destroyCount, 0);
  // Under 6144 MiB budget, 1350 MiB is ~22% -> normal pressure.
  assert.equal(reg.getVramPressure().tier, VRAM_PRESSURE_TIER.NORMAL);

  // Apply 'low' profile: budget -> 1536 MiB, triggerRatio -> 0.72 (1105.92 MiB), targetRatio -> 0.58, gcDeferFrames -> 2.
  // retiredBuf (elapsed 2 >= 2) should be collected immediately, while fencedBuf stays gated by pendingFence!
  const applied = reg.applyPerfProfile('low');
  assert.equal(applied.profile, 'low');
  assert.equal(applied.softBudgetBytes, 1536 * MiB);
  assert.equal(applied.framesInFlight, 2);
  assert.equal(applied.vramReclaimTriggerRatio, 0.72);
  assert.equal(applied.vramReclaimTargetRatio, 0.58);
  assert.equal(applied.collected.destroyedCount, 1);
  assert.equal(applied.collected.freedBytes, 100 * MiB);
  assert.equal(retiredBuf.destroyCount, 1);
  assert.equal(fencedBuf.destroyCount, 0);

  // Remaining VRAM: 1200 MiB active + 50 MiB pending = 1250 MiB / 1536 MiB (~81.38% >= 72% trigger, < 100% hard) -> SOFT pressure!
  const softPressure = reg.getVramPressure();
  assert.equal(softPressure.tier, VRAM_PRESSURE_TIER.SOFT);
  assert.equal(softPressure.shouldReclaim, true);
  assert.ok(softPressure.utilization > 0.81 && softPressure.utilization < 0.82);
  assert.ok(softPressure.bytesOverTarget > 0);
  assert.equal(reg.getStats().vramPressureTier, VRAM_PRESSURE_TIER.SOFT);
  assert.equal(reg.getStats().perfProfile, 'low');
  assert.equal(reg.getVramBreakdown().vramPressureTier, VRAM_PRESSURE_TIER.SOFT);
  assert.ok(pressureEvents.some((ev) => ev.tier === VRAM_PRESSURE_TIER.SOFT));

  // Push VRAM over 1536 MiB -> HARD pressure
  const extraTex = createDestroyableMock('extra-rt');
  reg.register(RESOURCE_KIND.TEXTURE, 4, extraTex, { byteSize: 400 * MiB, label: 'extra' });
  const hardPressure = reg.getVramPressure();
  assert.equal(hardPressure.tier, VRAM_PRESSURE_TIER.HARD);
  assert.equal(hardPressure.shouldReclaim, true);
  assert.equal(reg.getStats().budgetExceeded, true);
  assert.ok(pressureEvents.some((ev) => ev.tier === VRAM_PRESSURE_TIER.HARD));

  // Upgrade to 'balanced' (3072 MiB budget) -> utilization drops back to ~53.7% -> NORMAL pressure
  const balancedRes = reg.applyPerfProfile('balanced');
  assert.equal(balancedRes.profile, 'balanced');
  assert.equal(balancedRes.softBudgetBytes, 3072 * MiB);
  assert.equal(balancedRes.framesInFlight, 2);
  assert.equal(reg.getVramPressure().tier, VRAM_PRESSURE_TIER.NORMAL);
  assert.equal(reg.getStats().budgetExceeded, false);

  pendingFence.resolve();
  await pendingFence.promise;
  await Promise.resolve();
  reg.destroyAll();
});

// ---------------------------------------------------------------------------
// 6. IoWorkerController & ImmutableHttpfsCache integration (`io_worker.js`)
// ---------------------------------------------------------------------------

test('io_worker.js preserves defaults without perfProfile and applies low/balanced/high slab cache & lane budgets', async () => {
  // Backward compatibility: no perfProfile leaves existing defaults untouched
  const defaultCtrl = new IoWorkerController();
  const defaultImm = defaultCtrl.ensureImmutableCache();
  assert.equal(defaultImm.slabMaxBytes, DEFAULT_SLAB_MAX_BYTES);
  assert.equal(defaultImm.maxConcurrentSlabFetches, DEFAULT_MAX_CONCURRENT_SLAB_FETCHES);
  assert.equal(defaultCtrl.perfProfile, null);
  assert.equal(defaultCtrl.getStats().perfProfile, null);

  // ImmutableHttpfsCache direct constructor & runtime applyPerfProfile with immediate slab eviction
  const fileUrl = 'http://localhost/game/world.rpf';
  const payload = new Uint8Array(6 * MiB);
  payload.fill(0x5a);
  const fetchImpl = createRangeFetch(new Map([[fileUrl, payload]]));

  const imm = new ImmutableHttpfsCache({
    perfProfile: 'high',
    slabSize: 1 * MiB,
    prefetchAheadSlabs: 0,
    opfsL2Enabled: false,
    fetchImpl,
  });
  assert.equal(imm.slabMaxBytes, 192 * MiB);
  assert.equal(imm.maxConcurrentSlabFetches, 8);
  assert.equal(imm.getStats().perfProfile, 'high');

  // Populate 4 slabs (4 MiB) in RAM
  for (let i = 0; i < 4; i += 1) {
    await imm.read(fileUrl, i * MiB, 64);
  }
  assert.equal(imm.getStats().slabBytesCached, 4 * MiB);

  // Apply custom tight profile -> evicts cached slabs immediately down to 2 MiB
  const customProfile = resolvePerfProfile({
    preset: 'low',
    slabCacheBytes: 2 * MiB,
    ioMaxConcurrentLanes: 3,
  });
  const immRes = imm.applyPerfProfile(customProfile);
  assert.equal(immRes.profile, 'low');
  assert.equal(immRes.slabMaxBytes, 2 * MiB);
  assert.equal(immRes.maxConcurrentSlabFetches, 3);
  assert.equal(imm.getStats().slabBytesCached, 2 * MiB);

  // IoWorkerController init() with low / balanced / high perfProfile
  for (const [tier, expectedSlabBytes, expectedLanes] of [
    ['low', 64 * MiB, 4],
    ['balanced', 128 * MiB, 6],
    ['high', 192 * MiB, 8],
  ]) {
    const ctrl = new IoWorkerController({ fetchImpl });
    const ready = await ctrl.init({
      dataBase: '/data/',
      opfsDir: null,
      opfsL2Enabled: false,
      bootset: [],
      perfProfile: tier,
    });
    assert.equal(ready.ready, true);
    assert.equal(ready.perfProfile, tier);
    assert.equal(ready.maxConcurrentLanes, expectedLanes);
    assert.equal(ctrl.immutableCache.slabMaxBytes, expectedSlabBytes);
    assert.equal(ctrl.immutableCache.maxConcurrentSlabFetches, expectedLanes);
    assert.equal(ctrl.rangeCache.maxPrefetchConcurrency, expectedLanes);
    if (ctrl.scheduler) {
      assert.equal(ctrl.scheduler.maxConcurrency, expectedLanes);
    }
    const stats = ctrl.getStats();
    assert.equal(stats.perfProfile, tier);
    assert.equal(stats.maxConcurrentLanes, expectedLanes);
    assert.equal(stats.immutableCache.slabMaxBytes, expectedSlabBytes);
    assert.equal(stats.immutableCache.maxConcurrentSlabFetches, expectedLanes);
  }

  // Explicit init option overrides perfProfile default
  const overrideCtrl = new IoWorkerController({ fetchImpl });
  await overrideCtrl.init({
    dataBase: '/data/',
    opfsDir: null,
    opfsL2Enabled: false,
    bootset: [],
    perfProfile: 'low',
    slabMaxBytes: 96 * MiB,
  });
  assert.equal(overrideCtrl.immutableCache.slabMaxBytes, 96 * MiB);
  assert.equal(overrideCtrl.immutableCache.maxConcurrentSlabFetches, 4);

  // Worker message handler supports runtime `{ applyPerfProfile: 'low' }`
  const replies = [];
  const handler = createIoWorkerHandler({
    fetchImpl,
    postMessage: (msg) => replies.push(msg),
  });
  await handler({
    init: true,
    dataBase: '/data/',
    opfsDir: null,
    opfsL2Enabled: false,
    bootset: [],
    perfProfile: 'high',
  });
  assert.equal(handler.controller.immutableCache.slabMaxBytes, 192 * MiB);
  assert.equal(handler.controller.maxConcurrentLanes, 8);

  await handler({ id: 42, applyPerfProfile: 'low' });
  const applyReply = replies.find((r) => r.id === 42);
  assert.ok(applyReply);
  assert.equal(applyReply.ok, true);
  assert.equal(applyReply.perfProfile.profile, 'low');
  assert.equal(applyReply.perfProfile.slabMaxBytes, 64 * MiB);
  assert.equal(applyReply.perfProfile.maxConcurrentLanes, 4);
  assert.equal(handler.controller.immutableCache.slabMaxBytes, 64 * MiB);
  assert.equal(handler.controller.immutableCache.maxConcurrentSlabFetches, 4);
  assert.equal(handler.controller.rangeCache.maxPrefetchConcurrency, 4);
  if (handler.controller.scheduler) {
    assert.equal(handler.controller.scheduler.maxConcurrency, 4);
  }
});
