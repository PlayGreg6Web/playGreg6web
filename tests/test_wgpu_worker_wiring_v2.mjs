import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const WORKERS_DIR = fileURLToPath(new URL('../web/workers/', import.meta.url));

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
} = require('../web/workers/wgpu_worker.js');
const { WebGpuPipelineCache } = require('../web/workers/pipeline_cache.js');
const { ReflectionRegistry } = require('../web/workers/reflection_loader.js');
const {
  PREWARM_KIND_GRAPHICS,
  PREWARM_KIND_COMPUTE,
  PREWARM_MSG_REQUEST,
  PREWARM_MSG_RESULT,
  formatShaderKeyHex,
} = require('../web/workers/prewarm_bridge.js');
const { GpuPassTimer, MAP_MODE, GPU_TIME_SUPPORT } = require('../web/workers/gpu_timer.js');

const textEncoder = new TextEncoder();

// ============================================================================
// Binary Packet & WGPK Helpers
// ============================================================================

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

function pktBeginFrame() {
  return { opcode: OPCODES.BEGIN_FRAME, payload: new Uint8Array(0) };
}

function pktPresent() {
  return { opcode: OPCODES.PRESENT, payload: new Uint8Array(0) };
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

function pktCreateRenderPipeline(id, vsShaderId, fsShaderId, vertexStride = 0, topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST) {
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

function pktSetPipeline(pipelineId) {
  const payload = new Uint8Array(4);
  new DataView(payload.buffer).setUint32(0, pipelineId, true);
  return { opcode: OPCODES.SET_PIPELINE, payload };
}

function pktDraw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
  const payload = new Uint8Array(16);
  const view = new DataView(payload.buffer);
  view.setUint32(0, vertexCount, true);
  view.setUint32(4, instanceCount, true);
  view.setUint32(8, firstVertex, true);
  view.setUint32(12, firstInstance, true);
  return { opcode: OPCODES.DRAW, payload };
}

function pktCreateComputePipeline(id, csShaderId) {
  const payload = new Uint8Array(16);
  const view = new DataView(payload.buffer);
  view.setUint32(0, id, true);
  view.setUint32(4, csShaderId, true);
  view.setUint32(8, 0, true);
  view.setUint32(12, 0, true);
  return { opcode: OPCODES.CREATE_COMPUTE_PIPELINE, payload };
}

function pktSetComputePipeline(pipelineId) {
  const payload = new Uint8Array(4);
  new DataView(payload.buffer).setUint32(0, pipelineId, true);
  return { opcode: OPCODES.SET_COMPUTE_PIPELINE, payload };
}

function pktDispatchCompute(x, y = 1, z = 1) {
  const payload = new Uint8Array(12);
  const view = new DataView(payload.buffer);
  view.setUint32(0, x, true);
  view.setUint32(4, y, true);
  view.setUint32(8, z, true);
  return { opcode: OPCODES.DISPATCH_COMPUTE, payload };
}

/**
 * Build a minimal WGPK v1 binary buffer matching shaders/src/pack.rs & InlineWgslPackCache.
 * stageCode: 1 = vertex, 2 = fragment, 3 = compute.
 */
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
  // Magic 'WGPK'
  buf[0] = 0x57;
  buf[1] = 0x47;
  buf[2] = 0x50;
  buf[3] = 0x4b;
  view.setUint32(4, 1, true); // version 1
  view.setUint32(8, encoded.length, true);
  view.setUint32(12, 0, true);

  let dataCursor = dataStart;
  for (let i = 0; i < encoded.length; i += 1) {
    const base = HEADER_BYTES + i * ENTRY_BYTES;
    const item = encoded[i];
    view.setBigUint64(base, item.hash, true);
    buf[base + 8] = item.stageCode;
    view.setUint32(base + 12, dataCursor, true);
    view.setUint32(base + 16, item.bytes.byteLength, true);
    buf.set(item.bytes, dataCursor);
    dataCursor += item.bytes.byteLength;
  }
  return buf;
}

// ============================================================================
// Mock WebGPU Environment (supports timestamp-query, async compiles, order log)
// ============================================================================

function createMockWebGpu({ timestampQuery = false, timestamps = [] } = {}) {
  const calls = {
    createShaderModule: [],
    createRenderPipeline: [],
    createComputePipeline: [],
    createComputePipelineAsync: [],
    beginRenderPass: [],
    beginComputePass: [],
    resolveQuerySet: [],
    copyBufferToBuffer: [],
    encoderFinish: 0,
    queueSubmit: 0,
    order: [],
  };
  let nextHandle = 1;
  let nextTimestamps = timestamps.slice();

  const device = {
    features: new Set(timestampQuery ? ['timestamp-query'] : []),
    createQuerySet(desc) {
      return {
        kind: 'GPUQuerySet',
        handle: nextHandle++,
        desc: { ...desc },
        values: new BigUint64Array(desc.count),
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
    },
    createBuffer(desc) {
      const storage = new ArrayBuffer(desc.size);
      const buf = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        _storage: storage,
        mapState: desc.mappedAtCreation ? 'mapped' : 'unmapped',
        destroyed: false,
        mapAsync(mode) {
          if (mode !== MAP_MODE.READ) {
            return Promise.reject(new Error(`bad map mode ${mode}`));
          }
          this.mapState = 'pending';
          return Promise.resolve().then(() => {
            this.mapState = 'mapped';
          });
        },
        getMappedRange(offset = 0, size = this.size - offset) {
          return this._storage.slice(offset, offset + size);
        },
        unmap() {
          this.mapState = 'unmapped';
        },
        destroy() {
          this.destroyed = true;
        },
      };
      return buf;
    },
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
    createComputePipeline(desc) {
      const cp = {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout: (i) => ({ kind: 'autoBgl', index: i }),
      };
      calls.createComputePipeline.push({ desc, result: cp });
      return cp;
    },
    async createComputePipelineAsync(desc) {
      const cp = {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout: (i) => ({ kind: 'autoBgl', index: i }),
      };
      calls.createComputePipelineAsync.push({ desc, result: cp });
      return cp;
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          calls.beginRenderPass.push(desc);
          calls.order.push('beginRenderPass');
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
        beginComputePass(desc) {
          calls.beginComputePass.push(desc);
          calls.order.push('beginComputePass');
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        resolveQuerySet(querySet, firstQuery, queryCount, destination, destinationOffset) {
          calls.resolveQuerySet.push({
            querySet,
            firstQuery,
            queryCount,
            destination,
            destinationOffset,
          });
          calls.order.push('resolveQuerySet');
          for (let i = 0; i < queryCount; i += 1) {
            querySet.values[firstQuery + i] =
              i < nextTimestamps.length ? BigInt(nextTimestamps[i]) : 0n;
          }
          const dst = new BigUint64Array(destination._storage, destinationOffset, queryCount);
          for (let i = 0; i < queryCount; i += 1) {
            dst[i] = querySet.values[firstQuery + i];
          }
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          calls.copyBufferToBuffer.push({
            source,
            sourceOffset,
            destination,
            destinationOffset,
            size,
          });
          calls.order.push('copyBufferToBuffer');
          new Uint8Array(destination._storage, destinationOffset, size).set(
            new Uint8Array(source._storage, sourceOffset, size)
          );
        },
        finish() {
          calls.encoderFinish += 1;
          calls.order.push('encoder.finish');
          return { kind: 'GPUCommandBuffer' };
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {
        calls.queueSubmit += 1;
        calls.order.push('queue.submit');
      },
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

  return {
    device,
    ctx,
    calls,
    setTimestamps(list) {
      nextTimestamps = list.slice();
    },
  };
}

// Canonical #2085 VS_Blit / PS_BlitCalibration WGSL fixtures.
const VS_BLIT_HASH = '0x5fe7a88d9cb680b3';
const PS_BLIT_HASH = '0x56214b8d3b10a108';
const CS_SAMPLE_HASH = '0x1122334455667788';

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

const CS_SAMPLE_WGSL = '@compute @workgroup_size(1) fn cs_main() {}';

const FSSM_SIDECAR = {
  schema: 'playgta6-stage-reflection',
  version: 1,
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
      ps_inputs: [
        { semantic_reg: 1, location: 0, mask: 0 },
        { semantic_reg: 2, location: 0, mask: 3 },
      ],
    },
  },
};

// ============================================================================
// 1. FSSM VS/PS @location Remap Integration (#2085 / #2113)
// ============================================================================

test('#2119 FSSM remap: WgpuCommandProcessor applies VS_Blit param1 -> attr0 @location remap automatically from .reflect.json sidecar', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const pipelineCache = new WebGpuPipelineCache({ device });
  const wgpk = buildWgpkV1([
    { hash: VS_BLIT_HASH, stageCode: 1, wgsl: VS_BLIT_WGSL },
    { hash: PS_BLIT_HASH, stageCode: 2, wgsl: PS_BLIT_CALIBRATION_WGSL },
  ]);

  const processor = new WgpuCommandProcessor(device, ctx, {
    pipelineCache,
    wgslPackBuffer: wgpk,
    reflectionManifest: FSSM_SIDECAR,
    enableShaderFallback: false,
    lazyShaderCompile: false,
  });

  // Confirm pipelineCache received the ReflectionRegistry via setReflectionRegistry().
  assert.equal(pipelineCache._reflectionRegistry, processor.reflectionRegistry);

  const stream = buildCommandBuffer([
    pktCreateShaderModule(1, SHADER_STAGE.VERTEX, VS_BLIT_HASH),
    pktCreateShaderModule(2, SHADER_STAGE.FRAGMENT, PS_BLIT_HASH),
    pktCreateRenderPipeline(10, 1, 2, 12),
  ]);
  processor.executeCommandBuffer(stream);

  // Expect 3 createShaderModule calls: original VS, original PS, and remapped PS!
  assert.equal(calls.createShaderModule.length, 3);
  const remappedCall = calls.createShaderModule[2];
  assert.match(remappedCall.desc.label, /:interstage-remap$/);
  assert.match(
    remappedCall.desc.code,
    /struct FragIn \{[\s\S]*@location\(1\) attr_0_: vec4<f32>,[\s\S]*\}/
  );
  // Fragment MRT output @location(0) must never be rewritten.
  assert.match(
    remappedCall.desc.code,
    /struct PsOut \{[\s\S]*@location\(0\) mrt_0_: vec4<f32>,[\s\S]*\}/
  );

  // Render pipeline uses the remapped PS module while keeping the original VS module.
  assert.equal(calls.createRenderPipeline.length, 1);
  const pipeDesc = calls.createRenderPipeline[0].desc;
  assert.equal(pipeDesc.vertex.module, calls.createShaderModule[0].result);
  assert.equal(pipeDesc.fragment.module, remappedCall.result);

  const interstageStats = pipelineCache.getInterstageStats();
  assert.equal(interstageStats.locationRemapFixes, 1);
  assert.equal(pipelineCache.getStats().interstagePatchedPipelines, 1);

  processor.destroy();
});

test('#2119 FSSM remap: loading .reflect.json sidecar AFTER CreateShaderModule still triggers remap, and absent reflectionManifest/pipelineCache never crashes', () => {
  // Part 1: Sidecar loaded after shader creation (with inline WGSL + explicit reflection key)
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, {
    enablePipelineCache: true,
    enableShaderFallback: false,
    lazyShaderCompile: false,
  });

  processor.executeCommandBuffer(
    buildCommandBuffer([
      pktCreateShaderModule(11, SHADER_STAGE.VERTEX, VS_BLIT_WGSL),
      pktCreateShaderModule(12, SHADER_STAGE.FRAGMENT, PS_BLIT_CALIBRATION_WGSL),
    ])
  );
  processor.registerShaderReflectionKey(11, VS_BLIT_HASH);
  processor.registerShaderReflectionKey(12, PS_BLIT_HASH);
  processor.loadReflectionManifest(FSSM_SIDECAR);

  processor.executeCommandBuffer(
    buildCommandBuffer([pktCreateRenderPipeline(20, 11, 12, 12)])
  );

  assert.equal(calls.createShaderModule.length, 3);
  assert.match(
    calls.createShaderModule[2].desc.code,
    /@location\(1\) attr_0_: vec4<f32>/
  );
  assert.equal(processor.pipelineCache.getInterstageStats().locationRemapFixes, 1);
  processor.destroy();

  // Part 2: Absent reflectionManifest AND absent pipelineCache degrades cleanly
  const bare = createMockWebGpu();
  const bareProc = new WgpuCommandProcessor(bare.device, bare.ctx, {
    enablePipelineCache: false,
    enableShaderFallback: false,
  });
  assert.equal(bareProc.pipelineCache, null);
  assert.equal(bareProc.reflectionRegistry, null);
  assert.doesNotThrow(() => {
    bareProc.executeCommandBuffer(
      buildCommandBuffer([
        pktCreateShaderModule(1, SHADER_STAGE.VERTEX, VS_BLIT_WGSL),
        pktCreateShaderModule(2, SHADER_STAGE.FRAGMENT, PS_BLIT_CALIBRATION_WGSL),
        pktCreateRenderPipeline(1, 1, 2, 12),
      ])
    );
  });
  assert.equal(bare.calls.createRenderPipeline.length, 1);
  bareProc.destroy();
});

// ============================================================================
// 2. Async Pipeline Prewarm Integration (#1613 / #2054 / #2116)
// ============================================================================

test('#2119 prewarm_bridge: worker routes prewarm-task, yields while busy, resumes on Present, exposes prewarmStats, and detaches on destroy', async () => {
  const { device, ctx, calls } = createMockWebGpu();
  const wgpk = buildWgpkV1([
    { hash: VS_BLIT_HASH, stageCode: 1, wgsl: VS_BLIT_WGSL },
    { hash: PS_BLIT_HASH, stageCode: 2, wgsl: PS_BLIT_CALIBRATION_WGSL },
    { hash: CS_SAMPLE_HASH, stageCode: 3, wgsl: CS_SAMPLE_WGSL },
  ]);

  const processor = new WgpuCommandProcessor(device, ctx, {
    wgslPackBuffer: wgpk,
    enableShaderFallback: false,
  });

  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    prewarmBridgeOptions: {
      // Use a long retry delay so resumption is proven to be triggered immediately by Present's pump()!
      yieldRetryMs: 60_000,
    },
  });

  assert.ok(handler.prewarmBridge, 'worker handler must attach prewarmBridge');
  assert.equal(processor.prewarmBridge, handler.prewarmBridge);

  // 1. Make the processor busy by recording a live frame without PRESENT yet.
  await handler.handleMessage({
    cmdBuffer: buildCommandBuffer([pktBeginFrame(), pktDraw(3, 1, 0, 0)]),
  });
  assert.equal(processor.isPrewarmIdle(), false);

  // 2. Send a graphics prewarm-task while busy -> must yield and queue!
  await handler.handleMessage({
    type: PREWARM_MSG_REQUEST,
    task: {
      id: 42,
      kind: PREWARM_KIND_GRAPHICS,
      key0: BigInt(VS_BLIT_HASH),
      key1: BigInt(PS_BLIT_HASH),
    },
  });
  await Promise.resolve();

  const busyStats = processor.getStats();
  assert.ok(busyStats.prewarmStats, 'getStats() must expose prewarmStats when attached');
  assert.equal(busyStats.prewarmStats.queued, 1);
  assert.ok(busyStats.prewarmStats.yieldCount >= 1, 'prewarm task must yield while processor is busy');
  assert.equal(
    posted.filter((m) => m && m.type === PREWARM_MSG_RESULT).length,
    0,
    'prewarm task must not complete while live frame commands are pending'
  );

  // 3. Complete the frame with PRESENT -> _flushPresent() calls prewarmBridge.pump() -> resumes immediately!
  await handler.handleMessage({
    cmdBuffer: buildCommandBuffer([pktPresent()]),
  });

  for (let i = 0; i < 20 && !posted.some((m) => m && m.type === PREWARM_MSG_RESULT && m.id === 42); i += 1) {
    await new Promise((r) => setTimeout(r, 5));
  }

  const gfxResult = posted.find((m) => m && m.type === PREWARM_MSG_RESULT && m.id === 42);
  assert.ok(gfxResult, 'deferred graphics prewarm-task must complete after frame PRESENT');
  assert.equal(gfxResult.ok, true);
  assert.equal(typeof gfxResult.buildNs, 'number');
  assert.equal(gfxResult.error, null);

  // 4. Send a compute prewarm-task while idle -> executes directly and compiles compute pipeline.
  await handler.handleMessage({
    type: PREWARM_MSG_REQUEST,
    task: {
      id: 43,
      kind: PREWARM_KIND_COMPUTE,
      key0: BigInt(CS_SAMPLE_HASH),
      key1: 0n,
    },
  });
  for (let i = 0; i < 20 && !posted.some((m) => m && m.type === PREWARM_MSG_RESULT && m.id === 43); i += 1) {
    await new Promise((r) => setTimeout(r, 5));
  }
  const csResult = posted.find((m) => m && m.type === PREWARM_MSG_RESULT && m.id === 43);
  assert.ok(csResult, 'compute prewarm-task must post prewarm-task-result');
  assert.equal(csResult.ok, true);
  assert.equal(calls.createComputePipelineAsync.length, 1);

  // 5. Verify prewarmStats is surfaced in processor.getStats(), telemetry snapshot, and gpu-stats reply.
  const idleStats = processor.getStats();
  assert.equal(idleStats.prewarmStats.completed, 2);
  assert.equal(idleStats.prewarmStats.failed, 0);
  assert.ok(idleStats.prewarmStats.yieldCount >= 1);

  const snap = buildGpuTelemetrySnapshot(processor);
  assert.ok(snap.prewarmStats);
  assert.equal(snap.prewarmStats.completed, 2);

  await handler.handleMessage({ id: 'stats-1', type: 'gpu-stats' });
  const gpuStatsMsg = posted.find((m) => m && m.id === 'stats-1');
  assert.ok(gpuStatsMsg && gpuStatsMsg.prewarmStats);
  assert.equal(gpuStatsMsg.prewarmStats.completed, 2);

  // 6. Destroy worker -> cleanly detaches prewarmBridge.
  const bridgeRef = handler.prewarmBridge;
  await handler.handleMessage({ id: 'destroy-1', destroy: true });
  assert.equal(handler.prewarmBridge, null);
  assert.equal(bridgeRef.getStats().detached, true);
});

test('#2119 prewarm_bridge: classic WorkerGlobalScope loads ./prewarm_bridge.js via importScripts and processes prewarm-task via self.onmessage', async () => {
  const { device, ctx } = createMockWebGpu();
  const wgpk = buildWgpkV1([
    { hash: CS_SAMPLE_HASH, stageCode: 3, wgsl: CS_SAMPLE_WGSL },
  ]);

  const importedScripts = [];
  const postedMessages = [];
  const canvas = {
    width: 0,
    height: 0,
    getContext: () => ctx,
  };
  const sandbox = {
    console,
    TextEncoder,
    TextDecoder,
    URL,
    performance,
    setTimeout,
    clearTimeout,
    navigator: {
      gpu: {
        requestAdapter: async () => ({
          info: { vendor: 'mock', architecture: 'test' },
          requestDevice: async () => device,
        }),
        getPreferredCanvasFormat: () => 'bgra8unorm',
      },
    },
    postMessage: (m) => postedMessages.push(m),
  };
  const vmCtx = vm.createContext(sandbox);
  vmCtx.self = vmCtx;
  vmCtx.importScripts = (...urls) => {
    for (const rel of urls) {
      const fullPath = path.resolve(WORKERS_DIR, rel);
      importedScripts.push(rel);
      vm.runInContext(readFileSync(fullPath, 'utf8'), vmCtx, { filename: fullPath });
    }
  };

  const workerPath = path.resolve(WORKERS_DIR, './wgpu_worker.js');
  vm.runInContext(readFileSync(workerPath, 'utf8'), vmCtx, { filename: workerPath });

  // Copy WGPK bytes into VM realm to avoid cross-realm TypedArray issues
  const vmWgpk = vm.runInContext(`new Uint8Array(${wgpk.byteLength})`, vmCtx);
  vmWgpk.set(wgpk);

  await vmCtx.self.onmessage({
    data: {
      init: true,
      canvas,
      width: 640,
      height: 360,
      wgslPackBuffer: vmWgpk,
      enableShaderFallback: false,
    },
  });

  assert.ok(
    importedScripts.includes('./prewarm_bridge.js'),
    `expected ./prewarm_bridge.js via importScripts; got ${JSON.stringify(importedScripts)}`
  );
  assert.ok(vmCtx.PlayGta6PrewarmBridge, 'globalThis.PlayGta6PrewarmBridge must be defined');

  await vmCtx.self.onmessage({
    data: {
      type: PREWARM_MSG_REQUEST,
      task: {
        id: 7,
        kind: PREWARM_KIND_COMPUTE,
        key0: formatShaderKeyHex(BigInt(CS_SAMPLE_HASH)),
        key1: '0x0000000000000000',
      },
    },
  });

  for (let i = 0; i < 20 && !postedMessages.some((m) => m && m.type === PREWARM_MSG_RESULT && m.id === 7); i += 1) {
    await new Promise((r) => setTimeout(r, 5));
  }

  const res = postedMessages.find((m) => m && m.type === PREWARM_MSG_RESULT && m.id === 7);
  assert.ok(res, 'classic worker self.onmessage must reply with prewarm-task-result');
  assert.equal(res.ok, true);
});

// ============================================================================
// 3. WebGPU Pass Timestamps Integration (#1608 / #2116)
// ============================================================================

test('#2119 gpu_timer: render & compute passes receive timestampWrites, resolveFrame precedes finish/submit, flushReadbacks populates getStats().gpuTime & telemetry', async () => {
  // Pass 0 (render): 2.5 ms (2_500_000 ns), Pass 1 (compute): 1.25 ms (1_250_000 ns)
  const { device, ctx, calls } = createMockWebGpu({
    timestampQuery: true,
    timestamps: [
      1_000_000n,
      3_500_000n, // render pass = 2_500_000 ns
      4_000_000n,
      5_250_000n, // compute pass = 1_250_000 ns
    ],
  });

  const wgpk = buildWgpkV1([
    { hash: VS_BLIT_HASH, stageCode: 1, wgsl: VS_BLIT_WGSL },
    { hash: PS_BLIT_HASH, stageCode: 2, wgsl: PS_BLIT_CALIBRATION_WGSL },
    { hash: CS_SAMPLE_HASH, stageCode: 3, wgsl: CS_SAMPLE_WGSL },
  ]);

  const processor = new WgpuCommandProcessor(device, ctx, {
    wgslPackBuffer: wgpk,
    enableShaderFallback: false,
    lazyShaderCompile: false,
  });

  // Automatically attached GpuPassTimer because device.features.has('timestamp-query') is true!
  assert.ok(processor.passTimer instanceof GpuPassTimer);
  assert.equal(processor.gpuTimer, processor.passTimer);
  assert.equal(processor.passTimer.supported, true);

  let flushReadbacksCalls = 0;
  const origFlush = processor.passTimer.flushReadbacks.bind(processor.passTimer);
  processor.passTimer.flushReadbacks = (...args) => {
    flushReadbacksCalls += 1;
    calls.order.push('flushReadbacks');
    return origFlush(...args);
  };

  const frameStream = buildCommandBuffer([
    pktCreateShaderModule(1, SHADER_STAGE.VERTEX, VS_BLIT_HASH),
    pktCreateShaderModule(2, SHADER_STAGE.FRAGMENT, PS_BLIT_HASH),
    pktCreateRenderPipeline(10, 1, 2, 12),
    pktCreateShaderModule(3, SHADER_STAGE.COMPUTE, CS_SAMPLE_HASH),
    pktCreateComputePipeline(20, 3),
    pktBeginFrame(),
    pktSetPipeline(10),
    pktDraw(3, 1, 0, 0),
    pktSetComputePipeline(20),
    pktDispatchCompute(4, 1, 1),
    pktPresent(),
  ]);

  processor.executeCommandBuffer(frameStream);

  // Verify timestampWrites were passed to BOTH render and compute passes.
  assert.equal(calls.beginRenderPass.length, 1);
  assert.ok(calls.beginRenderPass[0].timestampWrites, 'render pass must receive timestampWrites');
  assert.equal(calls.beginRenderPass[0].timestampWrites.beginningOfPassWriteIndex, 0);
  assert.equal(calls.beginRenderPass[0].timestampWrites.endOfPassWriteIndex, 1);

  assert.equal(calls.beginComputePass.length, 1);
  assert.ok(
    calls.beginComputePass[0] && calls.beginComputePass[0].timestampWrites,
    'compute pass must receive timestampWrites'
  );
  assert.equal(calls.beginComputePass[0].timestampWrites.beginningOfPassWriteIndex, 2);
  assert.equal(calls.beginComputePass[0].timestampWrites.endOfPassWriteIndex, 3);

  // Verify exact command order: resolveQuerySet + copyBufferToBuffer BEFORE encoder.finish() BEFORE queue.submit() BEFORE flushReadbacks()!
  const idxResolve = calls.order.indexOf('resolveQuerySet');
  const idxCopy = calls.order.indexOf('copyBufferToBuffer');
  const idxFinish = calls.order.indexOf('encoder.finish');
  const idxSubmit = calls.order.indexOf('queue.submit');
  const idxFlush = calls.order.indexOf('flushReadbacks');
  assert.ok(idxResolve >= 0 && idxCopy > idxResolve);
  assert.ok(idxFinish > idxCopy, 'resolveFrame must run before encoder.finish()');
  assert.ok(idxSubmit > idxFinish, 'encoder.finish() must precede queue.submit()');
  assert.ok(idxFlush > idxSubmit, 'flushReadbacks() must run after queue.submit()');
  assert.equal(flushReadbacksCalls, 1);

  // Wait for non-blocking readback promise to settle.
  await processor.collectPassTimings();

  const stats = processor.getStats();
  assert.ok(stats.gpuTime, 'getStats().gpuTime must be populated');
  assert.equal(stats.gpuTime.available, true);
  assert.equal(stats.gpuTime.support, GPU_TIME_SUPPORT.AVAILABLE);
  assert.equal(stats.gpuTime.frames_measured, 1);
  assert.equal(stats.gpuTime.frame_gpu_ns, 3_750_000);
  assert.equal(stats.gpuTime.total_ns, 3_750_000);
  assert.equal(stats.gpuTime.tracked, 2);
  assert.equal(stats.gpuTime.top.length, 2);

  // Render pass (2.5 ms) attributed to PS_BLIT_HASH, compute pass (1.25 ms) attributed to CS_SAMPLE_HASH.
  assert.equal(stats.gpuTime.top[0].kind, 'render');
  assert.equal(stats.gpuTime.top[0].hash, PS_BLIT_HASH);
  assert.equal(stats.gpuTime.top[0].total_ns, 2_500_000);
  assert.equal(stats.gpuTime.top[1].kind, 'compute');
  assert.equal(stats.gpuTime.top[1].hash, CS_SAMPLE_HASH);
  assert.equal(stats.gpuTime.top[1].total_ns, 1_250_000);

  // Verify buildGpuTelemetrySnapshot and worker gpu-stats reply also surface gpuTime!
  const snap = buildGpuTelemetrySnapshot(processor);
  assert.ok(snap.gpuTime);
  assert.equal(snap.gpuTime.frame_gpu_ns, 3_750_000);

  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
  });
  await handler.handleMessage({ id: 'gt-1', type: 'gpu-stats' });
  const reply = posted.find((m) => m && m.id === 'gt-1');
  assert.ok(reply && reply.gpuTime);
  assert.equal(reply.gpuTime.frames_measured, 1);
  assert.equal(reply.gpuTime.frame_gpu_ns, 3_750_000);

  await handler.handleMessage({ destroy: true });
});

test('#2119 gpu_timer: degrades cleanly with zero errors and omits timestampWrites when timestamp-query is unavailable', async () => {
  // Case A: Unsupported device with default auto-detect (passTimer stays null, zero timestampWrites)
  const mockA = createMockWebGpu({ timestampQuery: false });
  const procA = new WgpuCommandProcessor(mockA.device, mockA.ctx, {
    enableShaderFallback: false,
  });
  assert.equal(procA.passTimer, null);

  const stream = buildCommandBuffer([
    pktCreateShaderModule(1, SHADER_STAGE.VERTEX, VS_BLIT_WGSL),
    pktCreateShaderModule(2, SHADER_STAGE.FRAGMENT, PS_BLIT_CALIBRATION_WGSL),
    pktCreateRenderPipeline(10, 1, 2, 12),
    pktCreateShaderModule(3, SHADER_STAGE.COMPUTE, CS_SAMPLE_WGSL),
    pktCreateComputePipeline(20, 3),
    pktBeginFrame(),
    pktSetPipeline(10),
    pktDraw(3, 1, 0, 0),
    pktSetComputePipeline(20),
    pktDispatchCompute(2, 1, 1),
    pktPresent(),
  ]);

  assert.doesNotThrow(() => procA.executeCommandBuffer(stream));
  assert.equal(mockA.calls.beginRenderPass.length, 1);
  assert.equal('timestampWrites' in mockA.calls.beginRenderPass[0], false);
  assert.equal(mockA.calls.beginComputePass.length, 1);
  assert.equal(mockA.calls.beginComputePass[0], undefined);
  assert.equal(mockA.calls.resolveQuerySet.length, 0);
  assert.equal(procA.getStats().validationErrorCount, 0);
  assert.equal(procA.getStats().gpuTime, null);
  procA.destroy();

  // Case B: Unsupported device with injected GpuPassTimer -> omits timestampWrites, reports gpuTime.available=false
  const mockB = createMockWebGpu({ timestampQuery: false });
  const injectedTimer = new GpuPassTimer({ device: mockB.device });
  assert.equal(injectedTimer.supported, false);

  const procB = new WgpuCommandProcessor(mockB.device, mockB.ctx, {
    gpuTimer: injectedTimer,
    enableShaderFallback: false,
  });
  assert.doesNotThrow(() => procB.executeCommandBuffer(stream));
  assert.equal(mockB.calls.beginRenderPass.length, 1);
  assert.equal('timestampWrites' in mockB.calls.beginRenderPass[0], false);
  assert.equal(mockB.calls.beginComputePass.length, 1);
  assert.equal(mockB.calls.beginComputePass[0], undefined);
  assert.equal(mockB.calls.resolveQuerySet.length, 0);

  const statsB = procB.getStats();
  assert.equal(statsB.validationErrorCount, 0);
  assert.ok(statsB.gpuTime);
  assert.equal(statsB.gpuTime.available, false);
  assert.equal(statsB.gpuTime.support, GPU_TIME_SUPPORT.UNAVAILABLE);
  assert.equal(statsB.gpuTime.frames_measured, 0);
  assert.equal(statsB.gpuTime.total_ns, 0);
  procB.destroy();
});
