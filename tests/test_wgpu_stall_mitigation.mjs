import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  OPCODES,
  SHADER_STAGE,
} = require('../web/workers/wgpu_worker.js');
const { AsyncPipelineCache } = require('../web/workers/async_pipeline_cache.js');
const {
  parseWgslGroup0Bindings,
  findWgslStageEntryPoints,
} = require('../web/workers/render_state_packets.js');

function buildCreateShaderModulePacket(id, stage, wgsl) {
  const codeBytes = new TextEncoder().encode(wgsl);
  const payloadBytes = 16 + codeBytes.byteLength;
  const buf = new ArrayBuffer(8 + payloadBytes);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.CREATE_SHADER_MODULE, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, payloadBytes, true);
  view.setUint32(8, id, true);
  view.setUint32(12, stage, true);
  view.setUint32(16, codeBytes.byteLength, true);
  view.setUint32(20, 0, true);
  new Uint8Array(buf, 24).set(codeBytes);
  return new Uint8Array(buf);
}

function buildCreateRenderPipelinePacket(id, vsId, fsId) {
  const payloadBytes = 24;
  const buf = new ArrayBuffer(8 + payloadBytes);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.CREATE_RENDER_PIPELINE, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, payloadBytes, true);
  view.setUint32(8, id, true);
  view.setUint32(12, vsId, true);
  view.setUint32(16, fsId, true);
  view.setUint32(20, 0, true);
  view.setUint32(24, 0, true);
  view.setUint32(28, 0, true);
  return new Uint8Array(buf);
}

function buildCreateComputePipelinePacket(id, csId) {
  const payloadBytes = 16;
  const buf = new ArrayBuffer(8 + payloadBytes);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.CREATE_COMPUTE_PIPELINE, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, payloadBytes, true);
  view.setUint32(8, id, true);
  view.setUint32(12, csId, true);
  view.setUint32(16, 0, true);
  view.setUint32(20, 0, true);
  return new Uint8Array(buf);
}

function buildDestroyResourcePacket(id) {
  const payloadBytes = 4;
  const buf = new ArrayBuffer(8 + payloadBytes);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.DESTROY_RESOURCE, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, payloadBytes, true);
  view.setUint32(8, id, true);
  return new Uint8Array(buf);
}

function buildCreateTexturePacket(id, width, height, formatCode = 1, usage = 0x10) {
  const payloadBytes = 32;
  const buf = new ArrayBuffer(8 + payloadBytes);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.CREATE_TEXTURE, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, payloadBytes, true);
  view.setUint32(8, id, true);
  view.setUint32(12, width, true);
  view.setUint32(16, height, true);
  view.setUint32(20, 1, true);
  view.setUint32(24, 1, true);
  view.setUint32(28, formatCode, true);
  view.setUint32(32, usage, true);
  view.setUint32(36, 1, true); // 2d
  return new Uint8Array(buf);
}

function buildWriteTexturePacket({
  textureId,
  mipLevel = 0,
  originX = 0,
  originY = 0,
  originZ = 0,
  width,
  height,
  depth = 1,
  bytesPerRow,
  rowsPerImage,
  data,
}) {
  const payloadBytes = 48 + data.byteLength;
  const buf = new ArrayBuffer(8 + payloadBytes);
  const view = new DataView(buf);
  view.setUint16(0, OPCODES.WRITE_TEXTURE, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, payloadBytes, true);
  view.setUint32(8, textureId, true);
  view.setUint32(12, mipLevel, true);
  view.setUint32(16, originX, true);
  view.setUint32(20, originY, true);
  view.setUint32(24, originZ, true);
  view.setUint32(28, width, true);
  view.setUint32(32, height, true);
  view.setUint32(36, depth, true);
  view.setUint32(40, bytesPerRow, true);
  view.setUint32(44, rowsPerImage, true);
  view.setUint32(48, data.byteLength, true);
  view.setUint32(52, 0, true);
  new Uint8Array(buf, 56).set(data);
  return new Uint8Array(buf);
}

function createMockDevice() {
  const shaderModules = [];
  const renderPipelines = [];
  const computePipelines = [];
  const writeTextures = [];
  let nextHandle = 1;
  const device = {
    features: new Set(['texture-compression-bc']),
    createShaderModule(desc) {
      const mod = {
        kind: 'GPUShaderModule',
        handle: nextHandle++,
        label: desc.label,
        code: desc.code,
      };
      shaderModules.push(mod);
      return mod;
    },
    createRenderPipeline(desc) {
      const pipe = { kind: 'GPURenderPipeline', handle: nextHandle++, desc };
      renderPipelines.push(pipe);
      return pipe;
    },
    createComputePipeline(desc) {
      const pipe = { kind: 'GPUComputePipeline', handle: nextHandle++, desc };
      computePipelines.push(pipe);
      return pipe;
    },
    createTexture(desc) {
      return {
        kind: 'GPUTexture',
        handle: nextHandle++,
        format: desc.format,
        width: desc.size.width,
        height: desc.size.height,
        destroy() {},
      };
    },
    createCommandEncoder() {
      return { finish: () => ({ kind: 'GPUCommandBuffer' }) };
    },
    queue: {
      submit() {},
      writeTexture(dst, data, layout, size) {
        writeTextures.push({
          texture: dst.texture,
          mipLevel: dst.mipLevel,
          origin: { x: dst.origin.x, y: dst.origin.y, z: dst.origin.z },
          dataByteLength: data.byteLength,
          firstByte: data[0],
          lastByte: data[data.byteLength - 1],
          layout: { ...layout },
          size: { ...size },
        });
      },
    },
  };
  const ctx = {
    getCurrentTexture: () => ({ createView: () => ({ kind: 'swapchainView' }) }),
  };
  return { device, ctx, shaderModules, renderPipelines, computePipelines, writeTextures };
}

test('lazyRangedShaderCompile defers HTTP 206 fetch + compile until pipeline or ensureShaderModule references the shader', async () => {
  const { device, ctx, shaderModules, renderPipelines, computePipelines } = createMockDevice();

  const hashes = {
    unused: '0x0000000000000101',
    vs: '0x0000000000000102',
    fs: '0x0000000000000103',
    cs: '0x0000000000000104',
    destroyed: '0x0000000000000105',
  };
  const wgslByHash = {
    [hashes.unused]: '@compute @workgroup_size(64) fn main() { /* heavy unused preload */ }',
    [hashes.vs]: '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }',
    [hashes.fs]: '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }',
    [hashes.cs]: '@compute @workgroup_size(1) fn cs_main() {}',
    [hashes.destroyed]: '@compute @workgroup_size(1) fn dead_main() {}',
  };
  const fetchedKeys = [];
  const fakeRangedPack = {
    has(key) {
      return Object.prototype.hasOwnProperty.call(wgslByHash, key);
    },
    async getWgslAsync(key) {
      fetchedKeys.push(key);
      return wgslByHash[key] || null;
    },
  };

  const processor = new WgpuCommandProcessor(device, ctx, {
    lazyShaderCompile: 'pack',
    lazyRangedShaderCompile: true,
    enableShaderFallback: false,
    wgslPackCache: fakeRangedPack,
  });

  // Preload 5 ranged shaders — NONE should fetch or compile yet.
  processor.executeCommandBuffer(buildCreateShaderModulePacket(1, SHADER_STAGE.COMPUTE, hashes.unused));
  processor.executeCommandBuffer(buildCreateShaderModulePacket(2, SHADER_STAGE.VERTEX, hashes.vs));
  processor.executeCommandBuffer(buildCreateShaderModulePacket(3, SHADER_STAGE.FRAGMENT, hashes.fs));
  processor.executeCommandBuffer(buildCreateShaderModulePacket(4, SHADER_STAGE.COMPUTE, hashes.cs));
  processor.executeCommandBuffer(buildCreateShaderModulePacket(5, SHADER_STAGE.COMPUTE, hashes.destroyed));

  await processor.flushPendingRangedShaders();
  assert.deepEqual(fetchedKeys, [], 'Preloaded ranged shaders must not fetch before a pipeline references them');
  assert.equal(shaderModules.length, 0, 'Preloaded ranged shaders must not call device.createShaderModule');

  // Destroying an unreferenced lazy ranged shader drops it cleanly without ever fetching.
  processor.executeCommandBuffer(buildDestroyResourcePacket(5));
  await processor.flushPendingRangedShaders();
  assert.deepEqual(fetchedKeys, []);

  // Referencing VS + FS in CreateRenderPipeline triggers fetch + compile ONLY for those two.
  processor.executeCommandBuffer(buildCreateRenderPipelinePacket(10, 2, 3));
  await processor.flushPendingRangedShaders();
  assert.deepEqual(fetchedKeys.sort(), [hashes.vs, hashes.fs].sort());
  assert.equal(shaderModules.length, 2);
  assert.equal(renderPipelines.length, 1);

  // Referencing CS in CreateComputePipeline triggers fetch + compile for CS only.
  processor.executeCommandBuffer(buildCreateComputePipelinePacket(11, 4));
  await processor.flushPendingRangedShaders();
  assert.ok(fetchedKeys.includes(hashes.cs));
  assert.ok(!fetchedKeys.includes(hashes.unused), 'Unreferenced preload shader must still never be fetched');
  assert.ok(!fetchedKeys.includes(hashes.destroyed), 'Destroyed preload shader must never be fetched');
  assert.equal(computePipelines.length, 1);
});

test('createWgpuWorkerHandler defaultInitGpu enables lazyRangedShaderCompile by default and respects explicit override', async () => {
  const { device, ctx } = createMockDevice();
  device.lost = new Promise(() => {});
  const posted = [];
  const fakeCanvas = {
    width: 640,
    height: 360,
    getContext: () => ({
      ...ctx,
      configure() {},
    }),
  };
  const origNavDesc = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      gpu: {
        getPreferredCanvasFormat: () => 'bgra8unorm',
        requestAdapter: async () => ({
          features: new Set(),
          limits: {},
          requestDevice: async () => device,
        }),
      },
    },
  });
  try {
    const defaultHandler = createWgpuWorkerHandler({
      enableSwapchain: false,
      postMessage: (m) => posted.push(m),
    });
    await defaultHandler.handleMessage({ init: true, canvas: fakeCanvas });
    assert.equal(defaultHandler.processor._lazyRangedShaderCompile, true);

    const overrideHandler = createWgpuWorkerHandler({
      enableSwapchain: false,
      postMessage: (m) => posted.push(m),
    });
    await overrideHandler.handleMessage({
      init: true,
      canvas: fakeCanvas,
      lazyRangedShaderCompile: false,
    });
    assert.equal(overrideHandler.processor._lazyRangedShaderCompile, false);
  } finally {
    if (origNavDesc) {
      Object.defineProperty(globalThis, 'navigator', origNavDesc);
    } else {
      delete globalThis.navigator;
    }
  }
});

test('builtin GPUShaderModule cache reuses a single compiled module across oversized fallbacks and clears on reset', () => {
  const { device, ctx, shaderModules } = createMockDevice();
  const processor = new WgpuCommandProcessor(device, ctx, {
    maxWgslCompileBytes: 128,
    enableShaderFallback: true,
  });

  const oversized = `@compute @workgroup_size(64) fn main() { /* ${'x'.repeat(256)} */ }`;
  for (let id = 100; id < 110; id += 1) {
    processor.executeCommandBuffer(buildCreateShaderModulePacket(id, SHADER_STAGE.COMPUTE, oversized));
  }

  assert.equal(processor._oversizedWgslFallbacks, 10);
  assert.equal(
    shaderModules.length,
    1,
    '10 oversized compute fallbacks should reuse 1 cached builtin GPUShaderModule instead of compiling 10 duplicates'
  );
  const firstMod = processor.shaderModules.get(100).module;
  assert.strictEqual(processor.shaderModules.get(109).module, firstMod);

  // Device reset must clear cached builtin modules so stale handles are never reused.
  processor._resetDeviceResources();
  assert.equal(processor._cachedBuiltinShaderModules.size, 0);
  processor.executeCommandBuffer(buildCreateShaderModulePacket(200, SHADER_STAGE.COMPUTE, oversized));
  assert.equal(shaderModules.length, 2, 'New device compiles a fresh builtin GPUShaderModule after reset');
  assert.notStrictEqual(processor.shaderModules.get(200).module, firstMod);
});

test('_handleWriteTexture row-chunks >2 MiB uncompressed and BC-compressed 2D uploads with exact byte/row alignment', () => {
  const { device, ctx, writeTextures } = createMockDevice();
  const processor = new WgpuCommandProcessor(device, ctx, {
    enableStagingPool: false,
  });

  // 1) Uncompressed 1080p r8unorm-like upload (bytesPerRow=2048, height=1080 => 2,211,840 B > 2 MiB)
  // TEXTURE_FORMAT_NAMES[1] === 'rgba8unorm' (uncompressed, blockHeight=1)
  processor.executeCommandBuffer(buildCreateTexturePacket(1, 1920, 1080, 1));
  const bpr = 2048;
  const height = 1080;
  const fullData = new Uint8Array(bpr * height);
  for (let row = 0; row < height; row += 1) {
    fullData[row * bpr] = row & 0xff;
    fullData[(row + 1) * bpr - 1] = (row + 7) & 0xff;
  }
  processor.executeCommandBuffer(
    buildWriteTexturePacket({
      textureId: 1,
      originY: 0,
      width: 1920,
      height,
      bytesPerRow: bpr,
      rowsPerImage: height,
      data: fullData,
    })
  );

  assert.equal(writeTextures.length, 2, '2,211,840 B 2D upload should split into 2 <=2 MiB row bands');
  assert.equal(writeTextures[0].origin.y, 0);
  assert.equal(writeTextures[0].size.height, 1024);
  assert.equal(writeTextures[0].dataByteLength, 1024 * 2048);
  assert.equal(writeTextures[0].firstByte, 0);
  assert.equal(writeTextures[1].origin.y, 1024);
  assert.equal(writeTextures[1].size.height, 56);
  assert.equal(writeTextures[1].dataByteLength, 56 * 2048);
  assert.equal(writeTextures[1].firstByte, 1024 & 0xff);
  assert.equal(writeTextures[1].lastByte, (1079 + 7) & 0xff);

  // 2) Small <=2 MiB upload stays a single queue.writeTexture call
  writeTextures.length = 0;
  const smallData = new Uint8Array(256 * 64);
  processor.executeCommandBuffer(
    buildWriteTexturePacket({
      textureId: 1,
      width: 64,
      height: 64,
      bytesPerRow: 256,
      rowsPerImage: 64,
      data: smallData,
    })
  );
  assert.equal(writeTextures.length, 1);

  // 3) BC-compressed texture (bc1-rgba-unorm, blockHeight=4): chunk origins & heights must stay 4-row aligned.
  writeTextures.length = 0;
  // Find bc1-rgba-unorm via CreateTexture2 or directly register texture meta + handle
  const bcTex = device.createTexture({
    size: { width: 2048, height: 2048, depthOrArrayLayers: 1 },
    format: 'bc1-rgba-unorm',
    usage: 0x10,
  });
  processor.textures.set(2, bcTex);
  processor._textureMeta.set(2, {
    width: 2048,
    height: 2048,
    depthOrArrayLayers: 1,
    format: 'bc1-rgba-unorm',
  });
  // 2048x2048 BC1: 512 block-rows * 4096 bytesPerRow = 2,097,152 B; use 520 block-rows (height=2080) => 2,129,920 B > 2 MiB
  const bcBpr = 4096;
  const bcBlockRows = 520;
  const bcHeight = bcBlockRows * 4;
  const bcData = new Uint8Array(bcBpr * bcBlockRows);
  bcData[0] = 0xaa;
  bcData[512 * bcBpr] = 0xbb;
  processor.executeCommandBuffer(
    buildWriteTexturePacket({
      textureId: 2,
      originY: 0,
      width: 2048,
      height: bcHeight,
      bytesPerRow: bcBpr,
      rowsPerImage: bcBlockRows,
      data: bcData,
    })
  );
  assert.equal(writeTextures.length, 2, 'BC1 >2 MiB upload should split on 4-row block boundaries');
  assert.equal(writeTextures[0].origin.y % 4, 0);
  assert.equal(writeTextures[0].size.height % 4, 0);
  assert.equal(writeTextures[1].origin.y, writeTextures[0].size.height);
  assert.equal(writeTextures[1].origin.y % 4, 0);
  assert.equal(writeTextures[0].size.height + writeTextures[1].size.height, bcHeight);
  assert.equal(writeTextures[0].firstByte, 0xaa);
  assert.equal(writeTextures[1].firstByte, 0xbb);
});

test('executeCommandBuffer stall warning includes per-opcode topOps breakdown', () => {
  const { device, ctx } = createMockDevice();
  const warnings = [];
  let fakeNow = 1000;
  device.createTexture = (desc) => {
    fakeNow += 1500;
    return { kind: 'GPUTexture', format: desc.format, width: desc.size.width, height: desc.size.height, destroy() {} };
  };

  const processor = new WgpuCommandProcessor(device, ctx, {
    gpuStallWarnMs: 500,
    gpuStallWarnMinIntervalMs: 0,
    now: () => fakeNow,
    logger: {
      warn: (msg) => warnings.push(msg),
      log: () => {},
      error: () => {},
    },
  });

  processor.executeCommandBuffer(buildCreateTexturePacket(1, 64, 64, 1));
  const stallWarn = warnings.find((m) => m.includes('op=executeCommandBuffer'));
  assert.ok(stallWarn, 'Expected executeCommandBuffer GPU stall warning');
  assert.match(stallWarn, /packets=1\b/);
  assert.match(stallWarn, /bytes=40\b/);
  assert.match(stallWarn, /topOps=\[CreateTexture:1\//);
});

test('AsyncPipelineCache treats Dawn external Instance reference lost error as device_lost without fallback retry', async () => {
  let createCalls = 0;
  const device = {
    async createRenderPipelineAsync() {
      createCalls += 1;
      throw new Error('A valid external Instance reference no longer exists.');
    },
  };

  const cache = new AsyncPipelineCache({ device, maxConcurrent: 2 });
  const outcome = await cache.compileRenderPipeline(
    42,
    { label: 'primary' },
    { fallbackDescriptor: { label: 'fallback' } }
  );

  assert.equal(outcome.status, 'failed');
  assert.equal(outcome.reason, 'device_lost');
  assert.equal(outcome.retried, false);
  assert.equal(
    createCalls,
    1,
    'Device-lost error must NOT retry fallbackDescriptor'
  );
  const telemetry = cache.getTelemetry();
  assert.equal(telemetry.failuresByReason.device_lost, 1);
  assert.equal(telemetry.deviceLostRejections, 1);
  assert.equal(telemetry.quarantinedCount, 0, 'device_lost failures must not be quarantined');
});

test('render_state_packets WGSL comment stripper and parse caches return identical results across repeated calls', () => {
  const wgsl = [
    '// @group(0) @binding(99) var<uniform> ignored_line: vec4<f32>;',
    '/* outer /* nested @vertex fn fake_vs() {} */ comment */',
    '@group(0) @binding(0) var<uniform> u_cam: vec4<f32>;',
    '@group(0) @binding(2) var t_albedo: texture_2d<f32>;',
    '@group(0) @binding(3) var s_linear: sampler;',
    '@fragment fn fs_main() -> @location(0) vec4<f32> {',
    '  let c = textureSample(t_albedo, s_linear, u_cam.xy);',
    '  return c;',
    '}',
  ].join('\n');

  const b1 = parseWgslGroup0Bindings(wgsl);
  const b2 = parseWgslGroup0Bindings(wgsl);
  assert.deepEqual(b1, [
    { binding: 0, kind: 'uniform' },
    { binding: 2, kind: 'texture', viewDimension: '2d' },
    { binding: 3, kind: 'sampler' },
  ]);
  assert.deepEqual(b2, b1);

  const ep1 = findWgslStageEntryPoints(wgsl, 'fragment');
  const ep2 = findWgslStageEntryPoints(wgsl, 'fragment');
  assert.deepEqual([...ep1], ['fs_main']);
  assert.strictEqual(ep2, ep1);
  assert.deepEqual([...findWgslStageEntryPoints(wgsl, 'vertex')], []);
});
