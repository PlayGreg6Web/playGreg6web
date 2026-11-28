import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
} = require('../web/workers/wgpu_worker.js');
const { WebGpuPipelineCache } = require('../web/workers/pipeline_cache.js');
const { computePipelineLayoutHash } = require('../web/workers/reflection_loader.js');

const textEncoder = new TextEncoder();

function buildCommandBuffer(packets) {
  let totalBytes = 0;
  for (const pkt of packets) {
    totalBytes += 8 + pkt.payload.byteLength;
  }
  const out = new Uint8Array(totalBytes);
  const view = new DataView(out.buffer);
  let offset = 0;
  for (const pkt of packets) {
    view.setUint16(offset, pkt.opcode, true);
    view.setUint16(offset + 2, 0, true);
    view.setUint32(offset + 4, pkt.payload.byteLength, true);
    out.set(pkt.payload, offset + 8);
    offset += 8 + pkt.payload.byteLength;
  }
  return out;
}

function pktCreateShaderModule(id, stage, codeStr) {
  const codeBytes = textEncoder.encode(codeStr);
  const payload = new Uint8Array(16 + codeBytes.byteLength);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, stage, true);
  view.setUint32(8, codeBytes.byteLength, true);
  view.setUint32(12, 0, true);
  payload.set(codeBytes, 16);
  return { opcode: OPCODES.CREATE_SHADER_MODULE, payload };
}

function pktCreateRenderPipeline(
  id,
  vsShaderId,
  fsShaderId,
  vertexStride = 0,
  topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST
) {
  const payload = new Uint8Array(24);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, vsShaderId, true);
  view.setUint32(8, fsShaderId, true);
  view.setUint32(12, vertexStride, true);
  view.setUint32(16, topology, true);
  view.setUint32(20, 0, true);
  return { opcode: OPCODES.CREATE_RENDER_PIPELINE, payload };
}

function buildWgpkV1(entries) {
  const HEADER_BYTES = 16;
  const ENTRY_BYTES = 20;
  const encoded = entries.map((e) => ({
    hash: typeof e.hash === 'bigint' ? e.hash : BigInt(e.hash),
    stageCode: e.stageCode,
    bytes: textEncoder.encode(e.wgsl),
  }));
  const dataStart = HEADER_BYTES + encoded.length * ENTRY_BYTES;
  let totalBytes = dataStart;
  for (const item of encoded) {
    totalBytes += item.bytes.byteLength;
  }
  const buf = new Uint8Array(totalBytes);
  const view = new DataView(buf.buffer);
  buf[0] = 0x57;
  buf[1] = 0x47;
  buf[2] = 0x50;
  buf[3] = 0x4b;
  view.setUint32(4, 1, true);
  view.setUint32(8, encoded.length, true);
  view.setUint32(12, 0, true);

  let cursor = dataStart;
  for (let i = 0; i < encoded.length; i += 1) {
    const base = HEADER_BYTES + i * ENTRY_BYTES;
    const item = encoded[i];
    view.setBigUint64(base, item.hash, true);
    buf[base + 8] = item.stageCode;
    view.setUint32(base + 12, cursor, true);
    view.setUint32(base + 16, item.bytes.byteLength, true);
    buf.set(item.bytes, cursor);
    cursor += item.bytes.byteLength;
  }
  return buf;
}

function createMockWebGpu() {
  const calls = {
    createShaderModule: [],
    createRenderPipeline: [],
  };
  let nextHandle = 1;

  const device = {
    features: new Set(),
    createShaderModule(desc) {
      const mod = {
        kind: 'GPUShaderModule',
        handle: nextHandle++,
        label: desc.label || '',
        async getCompilationInfo() {
          return { messages: [] };
        },
      };
      calls.createShaderModule.push({ desc: { ...desc }, result: mod });
      return mod;
    },
    createBindGroupLayout(desc) {
      return { kind: 'GPUBindGroupLayout', handle: nextHandle++, desc };
    },
    createPipelineLayout(desc) {
      return { kind: 'GPUPipelineLayout', handle: nextHandle++, desc };
    },
    createRenderPipeline(desc) {
      const rp = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout: (i) => ({ kind: 'autoBgl', index: i }),
      };
      calls.createRenderPipeline.push({ desc, result: rp });
      return rp;
    },
    createBuffer(desc) {
      return { kind: 'GPUBuffer', handle: nextHandle++, size: desc.size, destroy() {} };
    },
    createCommandEncoder() {
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            setBindGroup() {},
            draw() {},
            drawIndexed() {},
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
      onSubmittedWorkDone: () => Promise.resolve(),
    },
  };

  const ctx = {
    configure() {},
    getCurrentTexture: () => ({
      width: 1280,
      height: 720,
      createView: () => ({ kind: 'swapchainView' }),
    }),
  };

  return { device, ctx, calls };
}

const VS_BLIT_HASH = '0x5fe7a88d9cb680b3';
const PS_BLIT_HASH = '0x56214b8d3b10a108';
const LEGACY_DXC_HASH = '0x000000000000d001';

const VS_BLIT_WGSL = [
  'struct VsOut {',
  '    @builtin(position) position: vec4<f32>,',
  '    @location(0) param_0_: vec4<f32>,',
  '    @location(1) param_1_: vec4<f32>,',
  '}',
  '@vertex fn vs_main(@location(0) pos: vec3<f32>) -> VsOut {',
  '    var out: VsOut;',
  '    out.position = vec4<f32>(pos, 1.0);',
  '    out.param_0_ = vec4<f32>(1.0, 1.0, 1.0, 1.0);',
  '    out.param_1_ = vec4<f32>(0.5, 0.5, 0.0, 1.0);',
  '    return out;',
  '}',
].join('\n');

const PS_BLIT_CALIBRATION_WGSL = [
  'struct FragIn {',
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

// Compact playgta6-stage-reflection v3 interstage sidecar (exact shape emitted by
// tools/stage_reflection_sidecar.py and server/serve_local.py).
const COMPACT_INTERSTAGE_V3 = {
  schema: 'playgta6-stage-reflection',
  version: 3,
  stages: {
    [VS_BLIT_HASH]: {
      stage: 'vertex',
      bindings: [],
      vs_outputs: [
        { semantic_reg: 1, location: 0 },
        { semantic_reg: 2, location: 1 },
      ],
    },
    [PS_BLIT_HASH]: {
      stage: 'fragment',
      bindings: [],
      ps_inputs: [{ semantic_reg: 2, location: 0, mask: 3 }],
    },
  },
};

// Legacy S6 dxc-spirv reflection-manifest.json (0 RDNA2 stages, 1 legacy DXC shader).
const EMPTY_PL_ID = computePipelineLayoutHash([]);
const LEGACY_REFLECTION_MANIFEST = {
  schema_version: 1,
  register_convention: { source: 'dxc-spirv', shifts: { b: 0, t: 0, u: 0, s: 0 } },
  bind_group_layouts: {},
  pipeline_layouts: {
    [EMPTY_PL_ID]: { bindGroupLayouts: [] },
  },
  shaders: {
    [LEGACY_DXC_HASH]: {
      stage: 'fragment',
      entry_point: 'legacy_fs_main',
      name: 'legacy_ui_ps',
      pipeline_layout: EMPTY_PL_ID,
      bind_group_layouts: [],
      bindings: [],
    },
  },
  pipelines: {},
};

test('preloadBootShaderAssets fetches derived <pack>.interstage.json alongside reflection-manifest.json, merges both, and remaps VS_Blit/PS_BlitCalibration @location(0) -> @location(1)', async () => {
  const { device, ctx, calls } = createMockWebGpu();
  const pipelineCache = new WebGpuPipelineCache({ device });
  const wgpk = buildWgpkV1([
    { hash: VS_BLIT_HASH, stageCode: 1, wgsl: VS_BLIT_WGSL },
    { hash: PS_BLIT_HASH, stageCode: 2, wgsl: PS_BLIT_CALIBRATION_WGSL },
  ]);

  const processor = new WgpuCommandProcessor(device, ctx, {
    pipelineCache,
    enableShaderFallback: false,
    lazyShaderCompile: false,
  });

  const fetchedUrls = [];
  const mockFetch = async (url) => {
    fetchedUrls.push(url);
    if (url === '/shaders/splash.wgsl.pack?v=42') {
      return {
        ok: true,
        status: 200,
        async arrayBuffer() {
          return wgpk.buffer.slice(wgpk.byteOffset, wgpk.byteOffset + wgpk.byteLength);
        },
      };
    }
    if (url === '/shaders/reflection-manifest.json') {
      return {
        ok: true,
        status: 200,
        async text() {
          return JSON.stringify(LEGACY_REFLECTION_MANIFEST);
        },
      };
    }
    if (url === '/shaders/splash.wgsl.pack.interstage.json?v=42') {
      return {
        ok: true,
        status: 200,
        async text() {
          return JSON.stringify(COMPACT_INTERSTAGE_V3);
        },
      };
    }
    return { ok: false, status: 404 };
  };

  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor,
    initGpu: async () => {
      posted.push({ ready: true });
      return processor;
    },
    fetch: mockFetch,
    postMessage: (m) => posted.push(m),
  });

  await handler.handleMessage({
    init: true,
    canvas: {},
    shaderPackUrl: '/shaders/splash.wgsl.pack?v=42',
    reflectionManifestUrl: '/shaders/reflection-manifest.json',
  });

  // Verify all 3 resources were fetched with query string preserved on .interstage.json
  assert.ok(fetchedUrls.includes('/shaders/splash.wgsl.pack?v=42'));
  assert.ok(fetchedUrls.includes('/shaders/reflection-manifest.json'));
  assert.ok(fetchedUrls.includes('/shaders/splash.wgsl.pack.interstage.json?v=42'));

  // Both legacy dxc-spirv shader AND RDNA2 FSSM stages must coexist in the SAME ReflectionRegistry!
  assert.ok(processor.reflectionRegistry, 'ReflectionRegistry must be initialized');
  assert.equal(pipelineCache._reflectionRegistry, processor.reflectionRegistry);
  assert.ok(
    processor.reflectionRegistry.getStageReflection(LEGACY_DXC_HASH),
    'legacy reflection-manifest.json shader must not be clobbered'
  );
  assert.deepEqual(
    processor.reflectionRegistry
      .getStageReflection(VS_BLIT_HASH)
      .vs_outputs.map(({ semantic_reg, location }) => ({ semantic_reg, location })),
    COMPACT_INTERSTAGE_V3.stages[VS_BLIT_HASH].vs_outputs
  );
  assert.deepEqual(
    processor.reflectionRegistry
      .getStageReflection(PS_BLIT_HASH)
      .ps_inputs.map(({ semantic_reg, location, mask }) => ({ semantic_reg, location, mask })),
    COMPACT_INTERSTAGE_V3.stages[PS_BLIT_HASH].ps_inputs
  );
  assert.equal(processor.getStats().reflectionShaderCount, 3);

  // Execute CreateShaderModule + CreateRenderPipeline through WgpuCommandProcessor
  processor.executeCommandBuffer(
    buildCommandBuffer([
      pktCreateShaderModule(1, SHADER_STAGE.VERTEX, VS_BLIT_HASH),
      pktCreateShaderModule(2, SHADER_STAGE.FRAGMENT, PS_BLIT_HASH),
      pktCreateRenderPipeline(10, 1, 2, 12),
    ])
  );

  // Expect 3 createShaderModule calls: VS_Blit, PS_BlitCalibration, and remapped PS_BlitCalibration!
  assert.equal(calls.createShaderModule.length, 3);
  const remappedPsCall = calls.createShaderModule[2];
  assert.match(remappedPsCall.desc.label, /:interstage-remap$/);
  assert.match(
    remappedPsCall.desc.code,
    /struct FragIn \{[\s\S]*@location\(1\) attr_0_: vec4<f32>,[\s\S]*\}/
  );
  assert.match(
    remappedPsCall.desc.code,
    /struct PsOut \{[\s\S]*@location\(0\) mrt_0_: vec4<f32>,[\s\S]*\}/
  );

  assert.equal(calls.createRenderPipeline.length, 1);
  assert.equal(calls.createRenderPipeline[0].desc.vertex.module, calls.createShaderModule[0].result);
  assert.equal(calls.createRenderPipeline[0].desc.fragment.module, remappedPsCall.result);
  assert.equal(pipelineCache.getInterstageStats().locationRemapFixes, 1);

  processor.destroy();
});

test('WgpuCommandProcessor.loadStageReflection merges cleanly with loadReflectionManifest in reverse order too', () => {
  const { device, ctx } = createMockWebGpu();
  const pipelineCache = new WebGpuPipelineCache({ device });
  const processor = new WgpuCommandProcessor(device, ctx, { pipelineCache });

  // Load compact interstage sidecar FIRST, then legacy reflection manifest SECOND
  processor.loadStageReflection(COMPACT_INTERSTAGE_V3);
  processor.loadReflectionManifest(LEGACY_REFLECTION_MANIFEST);

  assert.equal(pipelineCache._reflectionRegistry, processor.reflectionRegistry);
  assert.ok(processor.reflectionRegistry.getStageReflection(LEGACY_DXC_HASH));
  assert.ok(processor.reflectionRegistry.getStageReflection(VS_BLIT_HASH));
  assert.ok(processor.reflectionRegistry.getStageReflection(PS_BLIT_HASH));
  assert.equal(processor.getStats().reflectionShaderCount, 3);

  processor.destroy();
});

test('interstageReflectionUrl supports explicit override, opt-out (false), and non-fatal 404/network errors', async () => {
  const wgpk = buildWgpkV1([
    { hash: VS_BLIT_HASH, stageCode: 1, wgsl: VS_BLIT_WGSL },
    { hash: PS_BLIT_HASH, stageCode: 2, wgsl: PS_BLIT_CALIBRATION_WGSL },
  ]);

  // 1. Explicit interstageReflectionUrl override
  {
    const { device, ctx } = createMockWebGpu();
    const processor = new WgpuCommandProcessor(device, ctx, { enableShaderFallback: false });
    const seen = [];
    const handler = createWgpuWorkerHandler({
      processor,
      fetch: async (url) => {
        seen.push(url);
        if (url === '/custom/fssm.interstage.json') {
          return { ok: true, status: 200, async text() { return JSON.stringify(COMPACT_INTERSTAGE_V3); } };
        }
        return { ok: false, status: 404 };
      },
    });
    await handler.handleMessage({
      init: true,
      interstageReflectionUrl: '/custom/fssm.interstage.json',
    });
    assert.deepEqual(seen, ['/custom/fssm.interstage.json']);
    assert.ok(processor.reflectionRegistry.getStageReflection(VS_BLIT_HASH));
    processor.destroy();
  }

  // 2. Opt-out via interstageReflectionUrl: false skips auto-derived sidecar fetch
  {
    const { device, ctx } = createMockWebGpu();
    const processor = new WgpuCommandProcessor(device, ctx, { enableShaderFallback: false });
    const seen = [];
    const handler = createWgpuWorkerHandler({
      processor,
      fetch: async (url) => {
        seen.push(url);
        return {
          ok: true,
          status: 200,
          async arrayBuffer() {
            return wgpk.buffer.slice(wgpk.byteOffset, wgpk.byteOffset + wgpk.byteLength);
          },
        };
      },
    });
    await handler.handleMessage({
      init: true,
      shaderPackUrl: '/shaders/splash.wgsl.pack',
      interstageReflectionUrl: false,
    });
    assert.deepEqual(seen, ['/shaders/splash.wgsl.pack']);
    processor.destroy();
  }

  // 3. 404 or network failure on .interstage.json is strictly non-fatal
  for (const mode of ['404', 'throw', 'bad-json']) {
    const { device, ctx } = createMockWebGpu();
    const posted = [];
    const processor = new WgpuCommandProcessor(device, ctx, { enableShaderFallback: false });
    const handler = createWgpuWorkerHandler({
      processor,
      initGpu: async () => {
        posted.push({ ready: true });
        return processor;
      },
      postMessage: (m) => posted.push(m),
      fetch: async (url) => {
        if (url.endsWith('.interstage.json')) {
          if (mode === '404') return { ok: false, status: 404 };
          if (mode === 'throw') throw new Error('ECONNRESET');
          return { ok: true, status: 200, async text() { return '{corrupt'; } };
        }
        if (url.endsWith('.wgsl.pack')) {
          return {
            ok: true,
            status: 200,
            async arrayBuffer() {
              return wgpk.buffer.slice(wgpk.byteOffset, wgpk.byteOffset + wgpk.byteLength);
            },
          };
        }
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify(LEGACY_REFLECTION_MANIFEST);
          },
        };
      },
    });
    await handler.handleMessage({
      init: true,
      canvas: {},
      shaderPackUrl: '/shaders/splash.wgsl.pack',
      reflectionManifestUrl: '/shaders/reflection-manifest.json',
    });
    assert.equal(posted.some((m) => m.error), false, `${mode}: must not emit fatal error`);
    assert.equal(posted.some((m) => m.ready === true), true, `${mode}: must still emit ready`);
    assert.equal(processor.wgslPackCache.getStats().shaderCount, 2);
    assert.equal(processor.getStats().reflectionShaderCount, 1);
    processor.destroy();
  }
});
