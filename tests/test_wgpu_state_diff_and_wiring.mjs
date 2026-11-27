import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  bootstrapPlayGta6App,
  resolveDisplayBounds,
  resolveBootstrapPerfProfile,
  buildWorkerPerfInitPayloads,
} from '../web/app_bootstrap.js';
import {
  SHADER_STAGE as PACK_STAGE,
  fnv1a64,
  buildWgslPackBuffer,
  WgslPackCache,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const wgpuMod = require('../web/workers/wgpu_worker.js');
const swapchainMod = require('../web/workers/wgpu_swapchain.js');
const rtPoolMod = require('../web/workers/rt_pool.js');
const resourceRegistryMod = require('../web/workers/resource_registry.js');
const pipelineCacheMod = require('../web/workers/pipeline_cache.js');
const rsMod = require('../web/workers/render_state_packets.js');

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  OPCODES,
  OPCODE_COPY_TEXTURE,
  BUFFER_USAGE,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  SHADER_STAGE,
  BIND_RESOURCE_TYPE,
  INDEX_FORMAT,
} = wgpuMod;

const { WgpuSwapchain } = swapchainMod;
const { RenderTargetPool } = rtPoolMod;
const { GpuResourceRegistry } = resourceRegistryMod;
const { WebGpuPipelineCache } = pipelineCacheMod;
const { installRenderStatePackets } = rsMod;

class StreamBuilder {
  constructor() {
    this.chunks = [];
  }

  packet(opcode, payload = new Uint8Array(0)) {
    const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
    const out = new Uint8Array(8 + bytes.byteLength);
    const view = new DataView(out.buffer);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, bytes.byteLength, true);
    out.set(bytes, 8);
    this.chunks.push(out);
    return this;
  }

  beginFrame() {
    return this.packet(OPCODES.BEGIN_FRAME);
  }

  present() {
    return this.packet(OPCODES.PRESENT);
  }

  createBuffer(id, size, usage = BUFFER_USAGE.VERTEX | BUFFER_USAGE.INDEX | BUFFER_USAGE.UNIFORM) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, usage, true);
    view.setBigUint64(8, BigInt(size), true);
    return this.packet(OPCODES.CREATE_BUFFER, new Uint8Array(buf));
  }

  createTexture(
    id,
    width,
    height,
    usage = TEXTURE_USAGE.COPY_SRC | TEXTURE_USAGE.COPY_DST | TEXTURE_USAGE.TEXTURE_BINDING
  ) {
    const buf = new ArrayBuffer(32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, width, true);
    view.setUint32(8, height, true);
    view.setUint32(12, 1, true);
    view.setUint32(16, 1, true);
    view.setUint32(20, TEXTURE_FORMAT.RGBA8_UNORM, true);
    view.setUint32(24, usage, true);
    view.setUint32(28, TEXTURE_DIMENSION.DIM_2D, true);
    return this.packet(OPCODES.CREATE_TEXTURE, new Uint8Array(buf));
  }

  createShaderModule(id, stage, code) {
    const enc = new TextEncoder().encode(code);
    const buf = new ArrayBuffer(16 + enc.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, enc.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(enc);
    return this.packet(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
  }

  createRenderPipeline(id, vsId, fsId = 0, vertexStride = 0, topology = 0, flags = 0) {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, vsId, true);
    view.setUint32(8, fsId, true);
    view.setUint32(12, vertexStride, true);
    view.setUint32(16, topology, true);
    view.setUint32(20, flags, true);
    return this.packet(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
  }

  createComputePipeline(id, csId, flags = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, csId, true);
    view.setUint32(8, flags, true);
    view.setUint32(12, 0, true);
    return this.packet(OPCODES.CREATE_COMPUTE_PIPELINE, new Uint8Array(buf));
  }

  createBindGroup(id, pipelineId, groupIndex, entries) {
    const buf = new ArrayBuffer(16 + entries.length * 32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, pipelineId, true);
    view.setUint32(8, groupIndex, true);
    view.setUint32(12, entries.length, true);
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i];
      const base = 16 + i * 32;
      view.setUint32(base, e.binding, true);
      view.setUint32(base + 4, e.type ?? BIND_RESOURCE_TYPE.BUFFER, true);
      view.setUint32(base + 8, e.resourceId, true);
      view.setUint32(base + 12, 0, true);
      view.setBigUint64(base + 16, BigInt(e.offset ?? 0), true);
      view.setBigUint64(base + 24, BigInt(e.size ?? 0), true);
    }
    return this.packet(OPCODES.CREATE_BIND_GROUP, new Uint8Array(buf));
  }

  setPipeline(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this.packet(OPCODES.SET_PIPELINE, new Uint8Array(buf));
  }

  setComputePipeline(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this.packet(OPCODES.SET_COMPUTE_PIPELINE, new Uint8Array(buf));
  }

  setBindGroup(groupIndex, bindGroupId) {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, groupIndex, true);
    view.setUint32(4, bindGroupId, true);
    return this.packet(OPCODES.SET_BIND_GROUP, new Uint8Array(buf));
  }

  setVertexBuffer(slot, bufferId, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, slot, true);
    view.setUint32(4, bufferId, true);
    view.setBigUint64(8, BigInt(offset), true);
    return this.packet(OPCODES.SET_VERTEX_BUFFER, new Uint8Array(buf));
  }

  setIndexBuffer(bufferId, format = INDEX_FORMAT.UINT16, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, bufferId, true);
    view.setUint32(4, format, true);
    view.setBigUint64(8, BigInt(offset), true);
    return this.packet(OPCODES.SET_INDEX_BUFFER, new Uint8Array(buf));
  }

  setViewport(x, y, width, height, minDepth = 0, maxDepth = 1) {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setFloat32(0, x, true);
    view.setFloat32(4, y, true);
    view.setFloat32(8, width, true);
    view.setFloat32(12, height, true);
    view.setFloat32(16, minDepth, true);
    view.setFloat32(20, maxDepth, true);
    return this.packet(OPCODES.SET_VIEWPORT, new Uint8Array(buf));
  }

  setScissorRect(x, y, width, height) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, x, true);
    view.setUint32(4, y, true);
    view.setUint32(8, width, true);
    view.setUint32(12, height, true);
    return this.packet(OPCODES.SET_SCISSOR_RECT, new Uint8Array(buf));
  }

  setBlendConstant(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat32(0, r, true);
    view.setFloat32(4, g, true);
    view.setFloat32(8, b, true);
    view.setFloat32(12, a, true);
    return this.packet(OPCODES.SET_BLEND_CONSTANT, new Uint8Array(buf));
  }

  setStencilReference(ref) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, ref >>> 0, true);
    return this.packet(OPCODES.SET_STENCIL_REFERENCE, new Uint8Array(buf));
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, vertexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstVertex, true);
    view.setUint32(12, firstInstance, true);
    return this.packet(OPCODES.DRAW, new Uint8Array(buf));
  }

  drawIndexed(indexCount, instanceCount = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(20);
    const view = new DataView(buf);
    view.setUint32(0, indexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstIndex, true);
    view.setInt32(12, baseVertex, true);
    view.setUint32(16, firstInstance, true);
    return this.packet(OPCODES.DRAW_INDEXED, new Uint8Array(buf));
  }

  dispatchCompute(x, y = 1, z = 1) {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, x, true);
    view.setUint32(4, y, true);
    view.setUint32(8, z, true);
    return this.packet(OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf));
  }

  copyTexture(dstId, srcId, width = 16, height = 16) {
    const buf = new ArrayBuffer(56);
    const view = new DataView(buf);
    view.setUint32(0, dstId, true);
    view.setUint32(4, srcId, true);
    view.setUint32(40, width, true);
    view.setUint32(44, height, true);
    view.setUint32(48, 1, true);
    return this.packet(OPCODE_COPY_TEXTURE, new Uint8Array(buf));
  }

  build() {
    let total = 0;
    for (const c of this.chunks) total += c.byteLength;
    const out = new Uint8Array(total);
    let offset = 0;
    for (const c of this.chunks) {
      out.set(c, offset);
      offset += c.byteLength;
    }
    return out;
  }
}

function createTrackingMockGpu(options = {}) {
  const renderPasses = [];
  const computePasses = [];
  const copyTextureCalls = [];
  const createShaderModuleCalls = [];
  const createRenderPipelineCalls = [];
  let nextId = 1;

  const device = {
    features: new Set(),
    limits: { maxTextureDimension2D: 8192 },
    createBuffer(desc) {
      return {
        kind: 'GPUBuffer',
        id: nextId++,
        size: desc.size,
        usage: desc.usage,
        destroy() {},
      };
    },
    createTexture(desc) {
      const size = Array.isArray(desc.size)
        ? { width: desc.size[0], height: desc.size[1] || 1, depthOrArrayLayers: desc.size[2] || 1 }
        : desc.size;
      return {
        kind: 'GPUTexture',
        id: nextId++,
        width: size.width,
        height: size.height,
        depthOrArrayLayers: size.depthOrArrayLayers || 1,
        format: desc.format,
        usage: desc.usage,
        createView(viewDesc) {
          return { kind: 'GPUTextureView', textureId: nextId, desc: viewDesc || {} };
        },
        destroy() {},
      };
    },
    createSampler(desc) {
      return { kind: 'GPUSampler', id: nextId++, desc };
    },
    createBindGroupLayout(desc) {
      return { kind: 'GPUBindGroupLayout', id: nextId++, entries: desc.entries };
    },
    createPipelineLayout(desc) {
      return { kind: 'GPUPipelineLayout', id: nextId++, desc };
    },
    createShaderModule(desc) {
      const mod = {
        kind: 'GPUShaderModule',
        id: nextId++,
        label: desc.label || `mod-${nextId}`,
        ...(options.opaqueModules ? {} : { code: desc.code }),
        getCompilationInfo: async () => ({ messages: [] }),
      };
      createShaderModuleCalls.push({ desc, mod });
      return mod;
    },
    createRenderPipeline(desc) {
      const pipe = {
        kind: 'GPURenderPipeline',
        id: nextId++,
        desc,
        getBindGroupLayout(idx) {
          return { kind: 'GPUBindGroupLayout', fromPipeline: true, idx };
        },
      };
      createRenderPipelineCalls.push({ desc, pipe });
      return pipe;
    },
    createComputePipeline(desc) {
      return {
        kind: 'GPUComputePipeline',
        id: nextId++,
        desc,
        getBindGroupLayout(idx) {
          return { kind: 'GPUBindGroupLayout', fromPipeline: true, idx };
        },
      };
    },
    createBindGroup(desc) {
      return { kind: 'GPUBindGroup', id: nextId++, desc };
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          const log = {
            index: renderPasses.length,
            desc,
            setPipeline: [],
            setBindGroup: [],
            setVertexBuffer: [],
            setIndexBuffer: [],
            setViewport: [],
            setScissorRect: [],
            setStencilReference: [],
            setBlendConstant: [],
            draw: [],
            drawIndexed: [],
            ended: false,
          };
          renderPasses.push(log);
          return {
            setPipeline(p) {
              log.setPipeline.push(p);
            },
            setBindGroup(slot, bg, dyn) {
              log.setBindGroup.push({ slot, bg, dyn: dyn ? Array.from(dyn) : undefined });
            },
            setVertexBuffer(slot, buf, off, size) {
              log.setVertexBuffer.push({ slot, buf, off, size });
            },
            setIndexBuffer(buf, fmt, off, size) {
              log.setIndexBuffer.push({ buf, fmt, off, size });
            },
            setViewport(x, y, w, h, minD, maxD) {
              log.setViewport.push({ x, y, w, h, minD, maxD });
            },
            setScissorRect(x, y, w, h) {
              log.setScissorRect.push({ x, y, w, h });
            },
            setStencilReference(ref) {
              log.setStencilReference.push(ref);
            },
            setBlendConstant(c) {
              log.setBlendConstant.push({ ...c });
            },
            draw(vc, ic, fv, fi) {
              log.draw.push({ vc, ic, fv, fi });
            },
            drawIndexed(ic, inst, fi, bv, finst) {
              log.drawIndexed.push({ ic, inst, fi, bv, finst });
            },
            end() {
              log.ended = true;
            },
          };
        },
        beginComputePass(desc) {
          const log = {
            index: computePasses.length,
            desc,
            setPipeline: [],
            setBindGroup: [],
            dispatchWorkgroups: [],
            ended: false,
          };
          computePasses.push(log);
          return {
            setPipeline(p) {
              log.setPipeline.push(p);
            },
            setBindGroup(slot, bg, dyn) {
              log.setBindGroup.push({ slot, bg, dyn: dyn ? Array.from(dyn) : undefined });
            },
            dispatchWorkgroups(x, y, z) {
              log.dispatchWorkgroups.push({ x, y, z });
            },
            end() {
              log.ended = true;
            },
          };
        },
        copyTextureToTexture(src, dst, size) {
          copyTextureCalls.push({ src, dst, size });
        },
        copyBufferToBuffer() {},
        finish() {
          return { kind: 'GPUCommandBuffer' };
        },
      };
    },
    queue: {
      submit() {},
      writeBuffer() {},
      writeTexture() {},
      onSubmittedWorkDone: async () => {},
    },
  };

  const ctxConfigs = [];
  const ctx = {
    configure(cfg) {
      ctxConfigs.push({ ...cfg });
    },
    getCurrentTexture() {
      return {
        kind: 'GPUTexture',
        width: 1280,
        height: 720,
        createView() {
          return { kind: 'SwapchainView' };
        },
      };
    },
  };

  return {
    device,
    ctx,
    ctxConfigs,
    renderPasses,
    computePasses,
    copyTextureCalls,
    createShaderModuleCalls,
    createRenderPipelineCalls,
  };
}

const SIMPLE_VS = `
@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
`;
const SIMPLE_FS = `
@group(0) @binding(0) var<uniform> u0: vec4<f32>;
@fragment fn fs_main() -> @location(0) vec4<f32> { return u0; }
`;
const SIMPLE_CS = `
@group(0) @binding(0) var<uniform> u0: vec4<f32>;
@compute @workgroup_size(1) fn cs_main() { _ = u0; }
`;

test('#2177 render pass state-diffing eliminates redundant calls across 100 draws and re-binds on every state change or pass boundary', () => {
  const gpu = createTrackingMockGpu();
  const proc = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    enableAsyncPipelines: false,
    requireNonBlackFrame: false,
  });

  // Setup resources
  const setup = new StreamBuilder()
    .createBuffer(1, 1024, BUFFER_USAGE.VERTEX)
    .createBuffer(2, 1024, BUFFER_USAGE.INDEX)
    .createBuffer(3, 256, BUFFER_USAGE.UNIFORM)
    .createBuffer(4, 256, BUFFER_USAGE.UNIFORM)
    .createTexture(10, 16, 16)
    .createTexture(11, 16, 16)
    .createShaderModule(1, SHADER_STAGE.VERTEX, SIMPLE_VS)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, SIMPLE_FS)
    .createRenderPipeline(100, 1, 2)
    .createRenderPipeline(101, 1, 2)
    .createBindGroup(200, 100, 0, [{ binding: 0, resourceId: 3, size: 256 }])
    .createBindGroup(201, 100, 0, [{ binding: 0, resourceId: 4, size: 256 }]);
  proc.executeCommandBuffer(setup.build());

  // Frame 1: 100 identical draws sharing pipeline/bindGroup/VB/IB/viewport/scissor/stencil/blend
  const f1 = new StreamBuilder().beginFrame();
  for (let i = 0; i < 100; i += 1) {
    f1.setPipeline(100)
      .setBindGroup(0, 200)
      .setVertexBuffer(0, 1, 0)
      .setIndexBuffer(2, INDEX_FORMAT.UINT16, 0)
      .setViewport(0, 0, 1280, 720, 0, 1)
      .setScissorRect(0, 0, 1280, 720)
      .setStencilReference(7)
      .setBlendConstant(0.25, 0.5, 0.75, 1.0)
      .drawIndexed(6, 1, 0, 0, 0);
  }
  f1.present();
  proc.executeCommandBuffer(f1.build());

  assert.equal(gpu.renderPasses.length, 1);
  const pass0 = gpu.renderPasses[0];
  assert.equal(pass0.drawIndexed.length, 100);
  assert.equal(pass0.setPipeline.length, 1);
  assert.equal(pass0.setBindGroup.length, 1);
  assert.equal(pass0.setVertexBuffer.length, 1);
  assert.equal(pass0.setIndexBuffer.length, 1);
  assert.equal(pass0.setViewport.length, 1);
  assert.equal(pass0.setScissorRect.length, 1);
  assert.equal(pass0.setStencilReference.length, 1);
  assert.equal(pass0.setBlendConstant.length, 1);

  const stats1 = proc.getStats();
  assert.equal(stats1.redundantStateCallsSkipped, 99 * 8);
  assert.deepEqual(stats1.redundantStateSkippedBreakdown, {
    pipeline: 99,
    bindGroup: 99,
    vertexBuffer: 99,
    indexBuffer: 99,
    viewport: 99,
    scissor: 99,
    stencilRef: 99,
    blendConstant: 99,
  });

  const telemetry1 = buildGpuTelemetrySnapshot(proc);
  assert.equal(telemetry1.redundantStateCallsSkipped, 99 * 8);

  // Frame 2: test state mutation (every field changing triggers an immediate re-bind)
  // AND pass split via CopyTexture resets ALL tracked state on the new render pass!
  const f2 = new StreamBuilder()
    .beginFrame()
    .setPipeline(100)
    .setBindGroup(0, 200)
    .setVertexBuffer(0, 1, 0)
    .setIndexBuffer(2, INDEX_FORMAT.UINT16, 0)
    .setViewport(0, 0, 1280, 720, 0, 1)
    .setScissorRect(0, 0, 1280, 720)
    .setStencilReference(1)
    .setBlendConstant(1, 1, 1, 1)
    .drawIndexed(3)
    // Change each individual field once -> MUST emit every single one
    .setPipeline(101)
    .setBindGroup(0, 201)
    .setVertexBuffer(0, 1, 64) // different offset
    .setIndexBuffer(2, INDEX_FORMAT.UINT32, 0) // different format
    .setViewport(0, 0, 640, 360, 0, 1) // different viewport
    .setScissorRect(10, 20, 300, 200) // different scissor
    .setStencilReference(2) // different stencil ref
    .setBlendConstant(0, 0, 0, 1) // different blend constant
    .drawIndexed(3)
    // Split pass with CopyTexture -> ends pass #1, starts pass #2 with clean reset state!
    .copyTexture(11, 10, 16, 16)
    .setPipeline(101)
    .setBindGroup(0, 201)
    .setVertexBuffer(0, 1, 64)
    .setIndexBuffer(2, INDEX_FORMAT.UINT32, 0)
    .setViewport(0, 0, 640, 360, 0, 1)
    .setScissorRect(10, 20, 300, 200)
    .setStencilReference(2)
    .setBlendConstant(0, 0, 0, 1)
    .drawIndexed(3)
    .present();

  proc.executeCommandBuffer(f2.build());

  assert.equal(gpu.renderPasses.length, 3);
  assert.equal(gpu.copyTextureCalls.length, 1);

  const pass1 = gpu.renderPasses[1];
  assert.equal(pass1.setPipeline.length, 2);
  assert.equal(pass1.setBindGroup.length, 2);
  assert.equal(pass1.setVertexBuffer.length, 2);
  assert.equal(pass1.setVertexBuffer[1].off, 64);
  assert.equal(pass1.setIndexBuffer.length, 2);
  assert.equal(pass1.setIndexBuffer[1].fmt, 'uint32');
  assert.equal(pass1.setViewport.length, 2);
  assert.equal(pass1.setScissorRect.length, 2);
  assert.equal(pass1.setStencilReference.length, 2);
  assert.equal(pass1.setBlendConstant.length, 2);

  // Pass #2 after CopyTexture MUST re-bind ALL initial state even though values equal end of pass #1!
  const pass2 = gpu.renderPasses[2];
  assert.equal(pass2.setPipeline.length, 1);
  assert.equal(pass2.setBindGroup.length, 1);
  assert.equal(pass2.setVertexBuffer.length, 1);
  assert.equal(pass2.setIndexBuffer.length, 1);
  assert.equal(pass2.setViewport.length, 1);
  assert.equal(pass2.setScissorRect.length, 1);
  assert.equal(pass2.setStencilReference.length, 1);
  assert.equal(pass2.setBlendConstant.length, 1);
  assert.equal(pass2.drawIndexed.length, 1);

  // No redundant skips occurred in Frame 2 because every call was either a change or after a pass reset.
  assert.equal(proc.getStats().redundantStateCallsSkipped, 99 * 8);
});

test('#2177 _passSetBindGroup never skips when dynamicOffsets differ and skips when identical', () => {
  const gpu = createTrackingMockGpu();
  const proc = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    enableAsyncPipelines: false,
  });

  const calls = [];
  const fakePass = {
    setBindGroup(slot, bg, dyn) {
      calls.push({ slot, bg, dyn: dyn ? Array.from(dyn) : undefined });
    },
  };
  const bgA = { id: 'bgA' };
  const bgB = { id: 'bgB' };

  proc._resetActivePassState();
  assert.equal(proc._passSetBindGroup(fakePass, 0, bgA, [0, 256]), true);
  assert.equal(proc._passSetBindGroup(fakePass, 0, bgA, [0, 256]), false); // identical dynamicOffsets -> skip!
  assert.equal(proc._passSetBindGroup(fakePass, 0, bgA, new Uint32Array([0, 256])), false); // typed array identical -> skip!
  assert.equal(proc._passSetBindGroup(fakePass, 0, bgA, [0, 512]), true); // dynamic offset changed -> MUST re-bind!
  assert.equal(proc._passSetBindGroup(fakePass, 0, bgA, [0, 512]), false); // identical again -> skip!
  assert.equal(proc._passSetBindGroup(fakePass, 0, bgA), true); // count changed (2 -> 0) -> MUST re-bind!
  assert.equal(proc._passSetBindGroup(fakePass, 0, bgA), false); // identical no-offset -> skip!
  assert.equal(proc._passSetBindGroup(fakePass, 0, bgB), true); // different bindGroup -> MUST re-bind!

  assert.equal(calls.length, 4);
  assert.deepEqual(calls[0].dyn, [0, 256]);
  assert.deepEqual(calls[1].dyn, [0, 512]);
  assert.equal(calls[2].dyn, undefined);
  assert.equal(calls[3].bg, bgB);
  assert.equal(proc.getStats().redundantStateSkippedBreakdown.bindGroup, 4);

  // Reset pass state -> identical bgB MUST re-bind on the new pass
  proc._resetActivePassState();
  assert.equal(proc._passSetBindGroup(fakePass, 0, bgB), true);
  assert.equal(calls.length, 5);
});

test('#2177 compute pass state-diffing eliminates redundant calls across dispatches and resets across CopyTexture boundaries', () => {
  const gpu = createTrackingMockGpu();
  const proc = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    enableAsyncPipelines: false,
    requireNonBlackFrame: false,
  });

  const setup = new StreamBuilder()
    .createBuffer(1, 256, BUFFER_USAGE.UNIFORM)
    .createTexture(10, 16, 16)
    .createTexture(11, 16, 16)
    .createShaderModule(1, SHADER_STAGE.COMPUTE, SIMPLE_CS)
    .createComputePipeline(50, 1)
    .createBindGroup(60, 50, 0, [{ binding: 0, resourceId: 1, size: 256 }]);
  proc.executeCommandBuffer(setup.build());

  const frame = new StreamBuilder().beginFrame();
  // 25 dispatches in compute pass #0 with identical pipeline + bindGroup
  for (let i = 0; i < 25; i += 1) {
    frame.setComputePipeline(50).setBindGroup(0, 60).dispatchCompute(4, 1, 1);
  }
  // CopyTexture forces compute pass #0 to end!
  frame.copyTexture(11, 10, 16, 16);
  // 10 more dispatches in compute pass #1 with the same pipeline + bindGroup
  for (let i = 0; i < 10; i += 1) {
    frame.setComputePipeline(50).setBindGroup(0, 60).dispatchCompute(2, 1, 1);
  }
  frame.present();

  proc.executeCommandBuffer(frame.build());

  assert.equal(gpu.computePasses.length, 2);
  assert.equal(gpu.copyTextureCalls.length, 1);

  assert.equal(gpu.computePasses[0].setPipeline.length, 1);
  assert.equal(gpu.computePasses[0].setBindGroup.length, 1);
  assert.equal(gpu.computePasses[0].dispatchWorkgroups.length, 25);

  // Pass #1 MUST re-bind pipeline + bindGroup once after CopyTexture pass break!
  assert.equal(gpu.computePasses[1].setPipeline.length, 1);
  assert.equal(gpu.computePasses[1].setBindGroup.length, 1);
  assert.equal(gpu.computePasses[1].dispatchWorkgroups.length, 10);

  // 24 redundant pairs in pass #0 + 9 redundant pairs in pass #1 = 66 skipped calls
  const stats = proc.getStats();
  assert.equal(stats.redundantStateSkippedBreakdown.pipeline, 33);
  assert.equal(stats.redundantStateSkippedBreakdown.bindGroup, 33);
  assert.equal(stats.redundantStateCallsSkipped, 66);
});

test('#2177 wgpu_worker perfProfile + displayBounds wiring forwards to rtPool, resourceRegistry, prewarmBridge, and swapchain', async () => {
  const gpu = createTrackingMockGpu();
  const canvas = {
    width: 3840,
    height: 2160,
    getContext() {
      return gpu.ctx;
    },
  };
  const swapchain = new WgpuSwapchain({
    canvas,
    device: gpu.device,
    context: gpu.ctx,
    format: 'bgra8unorm',
  });
  swapchain.configure({ width: 3840, height: 2160, devicePixelRatio: 1 });
  assert.equal(swapchain.width, 3840);
  assert.equal(swapchain.height, 2160);

  const rtPool = new RenderTargetPool({ device: gpu.device });
  const resourceRegistry = new GpuResourceRegistry();
  const prewarmApplied = [];
  const mockPrewarmBridge = {
    maxInFlight: 4,
    setMaxInFlight(n) {
      this.maxInFlight = n;
      prewarmApplied.push({ kind: 'setMaxInFlight', n });
    },
    getStats() {
      return { maxInFlight: this.maxInFlight };
    },
    detach() {},
  };

  const proc = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    swapchain,
    rtPool,
    resourceRegistry,
    prewarmBridge: mockPrewarmBridge,
    perfProfile: 'low',
    displayBounds: { screenWidth: 1920, screenHeight: 1080, devicePixelRatio: 1 },
    enableAsyncPipelines: false,
  });

  // Constructor forwarded low profile + 1080p display bounds across every subsystem!
  assert.equal(proc.getStats().perfProfile, 'low');
  assert.equal(proc.getStats().displayBounds.maxDisplayWidth, 1920);
  assert.equal(proc.getStats().displayBounds.maxDisplayHeight, 1080);
  assert.equal(rtPool.getStats().maxPoolBytes, 128 * 1024 * 1024);
  assert.equal(rtPool.getStats().idleFrameTtl, 8);
  assert.equal(resourceRegistry.getStats().softBudgetBytes, 1536 * 1024 * 1024);
  assert.equal(mockPrewarmBridge.maxInFlight, 1);
  // 4K backing store clamped losslessly to 1080p native physical display pixels!
  assert.equal(swapchain.width, 1920);
  assert.equal(swapchain.height, 1080);

  // Worker handler runtime messages update perfProfile and displayBounds dynamically!
  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor: proc,
    canvas,
    swapchain,
    postMessage: (m) => posted.push(m),
  });

  await handler.handleMessage({
    id: 1,
    type: 'apply-perf-profile',
    perfProfile: 'high',
  });
  assert.equal(proc.getStats().perfProfile, 'high');
  assert.equal(rtPool.getStats().maxPoolBytes, 512 * 1024 * 1024);
  assert.equal(resourceRegistry.getStats().softBudgetBytes, 6144 * 1024 * 1024);
  assert.equal(mockPrewarmBridge.maxInFlight, 4);
  assert.equal(posted[0].type, 'perf-profile-applied');
  assert.equal(posted[0].profile, 'high');
  assert.equal(posted[0].perfProfile.textureMipCap, 0);

  await handler.handleMessage({
    id: 2,
    type: 'set-display-bounds',
    screenWidth: 1440,
    screenHeight: 900,
    devicePixelRatio: 2,
  });
  assert.equal(posted[1].type, 'display-bounds-updated');
  assert.equal(posted[1].displayBounds.maxDisplayWidth, 2880);
  assert.equal(posted[1].displayBounds.maxDisplayHeight, 1800);
  // 3840x2160 (16:9) aspect-preserving clamped into 2880x1800 (16:10) display bounds -> 2880x1620
  assert.equal(swapchain.width, 2880);
  assert.equal(swapchain.height, 1620);
});

test('#2177 wgpu_worker wires wgslPackCache as wgslSourceProvider on WebGpuPipelineCache for post-compile eviction + lazy inter-stage patching', () => {
  const VS_INTERSTAGE = [
    'struct VsOut {',
    '  @builtin(position) pos: vec4<f32>,',
    '  @location(0) param_0: vec4<f32>,',
    '  @location(1) param_1: vec4<f32>,',
    '}',
    '@vertex fn vs_main() -> VsOut {',
    '  var out: VsOut;',
    '  out.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);',
    '  out.param_0 = vec4<f32>(1.0);',
    '  out.param_1 = vec4<f32>(2.0);',
    '  return out;',
    '}',
  ].join('\n');

  const PS_FLAT_MISMATCH = [
    'struct FsIn {',
    '  @location(0) a0: vec4<f32>,',
    '  @location(1) @interpolate(flat) a1: vec4<f32>,',
    '}',
    '@fragment fn fs_main(in: FsIn) -> @location(0) vec4<f32> {',
    '  return in.a0 + in.a1;',
    '}',
  ].join('\n');

  const hVs = fnv1a64('e2e-evict-vs');
  const hPs = fnv1a64('e2e-evict-ps');
  const packBuf = buildWgslPackBuffer([
    { hash: hVs, stage: PACK_STAGE.VERTEX, wgsl: VS_INTERSTAGE },
    { hash: hPs, stage: PACK_STAGE.FRAGMENT, wgsl: PS_FLAT_MISMATCH },
  ]);

  const wgslPackCache = new WgslPackCache();
  wgslPackCache.loadPackBuffer(packBuf, 'e2e.wgsl.pack');

  // Opaque GPUShaderModules (no `.code` property) + zero retained WGSL source bytes in pipelineCache!
  const gpu = createTrackingMockGpu({ opaqueModules: true });
  const pipelineCache = new WebGpuPipelineCache({
    device: gpu.device,
    maxRetainedWgslSourceBytes: 0,
  });

  const proc = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    pipelineCache,
    wgslPackCache,
    enableAsyncPipelines: false,
  });

  // Verify wgpu_worker automatically wired wgslPackCache as wgslSourceProvider on pipelineCache!
  assert.equal(pipelineCache._wgslSourceProvider, wgslPackCache);

  // Compile shaders through wgslPackCache (auto-evicts decoded WGSL text immediately after compile!)
  const modVs = wgslPackCache.getOrCreateShaderModule(gpu.device, hVs, 'e2e');
  const modPs = wgslPackCache.getOrCreateShaderModule(gpu.device, hPs, 'e2e');
  assert.equal(wgslPackCache.getStats().wgslTextBytesRetained, 0);

  // Register modules with pipelineCache (retains compact inter-stage summaries only, 0 WGSL bytes!)
  const vsHex = `0x${hVs.toString(16).padStart(16, '0')}`;
  const psHex = `0x${hPs.toString(16).padStart(16, '0')}`;
  pipelineCache.registerShaderModule(modVs, vsHex, VS_INTERSTAGE);
  pipelineCache.registerShaderModule(modPs, psHex, PS_FLAT_MISMATCH);
  assert.equal(pipelineCache.getStats().wgslSourceBytesRetained, 0);

  // Creating a render pipeline detects the flat interpolation mismatch from compact summaries,
  // lazily re-decodes raw WGSL via wgslPackCache, patches the VS module, and compiles cleanly!
  const pipe = pipelineCache.getOrCreateRenderPipeline({
    vertex: { module: modVs, entryPoint: 'vs_main' },
    fragment: {
      module: modPs,
      entryPoint: 'fs_main',
      targets: [{ format: 'bgra8unorm' }],
    },
  });
  assert.ok(pipe);
  assert.ok(
    pipelineCache.getStats().wgslSourceLazyFetches >= 1,
    `expected lazy WGSL fetch from wgslPackCache, got ${pipelineCache.getStats().wgslSourceLazyFetches}`
  );
  assert.ok(wgslPackCache.getStats().wgslTextLazyRedecodes >= 1);
  // Patched VS shader module was compiled with `@interpolate(flat)` on `@location(1)`.
  const patchedCompile = gpu.createShaderModuleCalls.find(
    (c) => c.desc && typeof c.desc.code === 'string' && c.desc.code.includes('@interpolate(flat)') && c.desc.code.includes('vs_main')
  );
  assert.ok(patchedCompile, 'expected inter-stage patched VS module with @interpolate(flat)');
  assert.ok(proc.wgslPackCache === wgslPackCache);
});

test('#2177 app_bootstrap resolves full-quality perfProfile + displayBounds safely in Node and mock browsers and updates on resize/DPR', async () => {
  // 1. Safe in bare Node environment with no window/screen/navigator
  const nodeResolved = resolveBootstrapPerfProfile({
    windowObj: null,
    navigatorObj: null,
    urlSearch: '',
  });
  assert.equal(nodeResolved.displayBounds, null);
  assert.equal(nodeResolved.perfProfile.name, 'balanced');
  assert.equal(nodeResolved.perfProfile.textureMipCap, 0);
  assert.equal(nodeResolved.perfProfile.preserveFullQuality, true);
  assert.equal(resolveDisplayBounds({ windowObj: null }), null);

  // 2. Mock browser with low-spec hardware + Retina display + URL query override
  const mockScreen = { width: 1440, height: 900 };
  const listeners = new Map();
  const mockWindow = {
    innerWidth: 1440,
    innerHeight: 900,
    devicePixelRatio: 2,
    crossOriginIsolated: true,
    SharedArrayBuffer,
    screen: mockScreen,
    location: { origin: 'https://playgta6.local', search: '?quality=low' },
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      if (listeners.has(type)) listeners.get(type).delete(fn);
    },
    dispatchResize() {
      for (const fn of listeners.get('resize') || []) fn();
    },
  };
  const mockNavigator = {
    deviceMemory: 8,
    hardwareConcurrency: 4,
    gpu: {},
  };

  const browserResolved = resolveBootstrapPerfProfile({
    windowObj: mockWindow,
    navigatorObj: mockNavigator,
    urlSearch: mockWindow.location.search,
  });
  assert.equal(browserResolved.perfProfile.name, 'low');
  assert.equal(browserResolved.perfProfile.textureMipCap, 0);
  assert.equal(browserResolved.perfProfile.preserveFullQuality, true);
  assert.equal(browserResolved.displayBounds.maxDisplayWidth, 2880);
  assert.equal(browserResolved.displayBounds.maxDisplayHeight, 1800);
  assert.equal(browserResolved.displayBounds.devicePixelRatio, 2);

  const workerPayloads = buildWorkerPerfInitPayloads(
    browserResolved.perfProfile,
    browserResolved.displayBounds
  );
  assert.equal(workerPayloads.ioWorker.perfProfile.name, 'low');
  assert.equal(workerPayloads.ioWorker.perfProfile.textureMipCap, 0);
  assert.equal(workerPayloads.gpuWorker.perfProfile.name, 'low');
  assert.equal(workerPayloads.gpuWorker.displayBounds.maxDisplayWidth, 2880);
  assert.equal(workerPayloads.gpuWorker.displayBounds.maxDisplayHeight, 1800);

  // 3. Full bootstrapPlayGta6App lifecycle forwards perfProfile + displayBounds to loader, ioWorker, gpuWorker
  // and updates displayBounds cleanly when monitor screen/DPR changes!
  const loaderPosted = [];
  const ioPosted = [];
  const gpuPosted = [];
  const offscreen = { kind: 'OffscreenCanvas' };
  const mockCanvas = {
    width: 1440,
    height: 900,
    clientWidth: 1440,
    clientHeight: 900,
    transferControlToOffscreen: () => offscreen,
  };
  const mockIoWorker = {
    addEventListener() {},
    removeEventListener() {},
    postMessage(m) {
      ioPosted.push(m);
    },
  };
  const mockGpuWorker = {
    addEventListener() {},
    removeEventListener() {},
    postMessage(m) {
      gpuPosted.push(m);
    },
  };

  const app = bootstrapPlayGta6App({
    windowObj: mockWindow,
    navigatorObj: mockNavigator,
    documentObj: null,
    canvas: mockCanvas,
    ioWorker: mockIoWorker,
    gpuWorker: mockGpuWorker,
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      async json() {
        return { hasWasmBinary: true, runtimeWasmReady: true, wasmUrl: '/dist/runtime.wasm' };
      },
    }),
    createWasmWorker: () => ({
      addEventListener() {},
      removeEventListener() {},
      postMessage(payload, transfer) {
        loaderPosted.push({ payload, transfer });
      },
      terminate() {},
    }),
  });

  try {
    assert.equal(app.perfProfile.name, 'low');
    assert.equal(app.perfProfile.textureMipCap, 0);
    assert.equal(app.displayBounds.maxDisplayWidth, 2880);
    assert.equal(app.displayBounds.maxDisplayHeight, 1800);

    const startRes = await app.start({ preferWasm: true });
    assert.equal(startRes.ok, true);

    // Loader init payload contains full-quality perfProfile + display-native displayBounds
    assert.equal(loaderPosted[0].payload.perfProfile.name, 'low');
    assert.equal(loaderPosted[0].payload.perfProfile.textureMipCap, 0);
    assert.equal(loaderPosted[0].payload.displayBounds.maxDisplayWidth, 2880);
    assert.equal(loaderPosted[0].payload.displayBounds.maxDisplayHeight, 1800);

    // External ioWorker and gpuWorker also received perfProfile / displayBounds
    assert.equal(ioPosted.length, 1);
    assert.equal(ioPosted[0].type, 'apply-perf-profile');
    assert.equal(ioPosted[0].perfProfile.name, 'low');
    assert.equal(gpuPosted.length, 2);
    assert.equal(gpuPosted[0].type, 'apply-perf-profile');
    assert.equal(gpuPosted[1].type, 'set-display-bounds');
    assert.equal(gpuPosted[1].displayBounds.maxDisplayWidth, 2880);

    // Simulate moving browser window to a 1080p 1x monitor and firing window resize
    mockScreen.width = 1920;
    mockScreen.height = 1080;
    mockWindow.devicePixelRatio = 1;
    mockCanvas.clientWidth = 1920;
    mockCanvas.clientHeight = 1080;
    mockWindow.dispatchResize();

    assert.equal(app.displayBounds.maxDisplayWidth, 1920);
    assert.equal(app.displayBounds.maxDisplayHeight, 1080);
    const boundsMsg = loaderPosted.find((m) => m.payload && m.payload.type === 'set-display-bounds');
    assert.ok(boundsMsg, 'expected set-display-bounds message to loader worker on monitor change');
    assert.equal(boundsMsg.payload.displayBounds.maxDisplayWidth, 1920);
    assert.equal(boundsMsg.payload.displayBounds.maxDisplayHeight, 1080);

    const gpuBoundsMsgs = gpuPosted.filter((m) => m.type === 'set-display-bounds');
    assert.equal(gpuBoundsMsgs.length, 2);
    assert.equal(gpuBoundsMsgs[1].displayBounds.maxDisplayWidth, 1920);
  } finally {
    await app.destroy();
  }
});

test('#2177 render_state_packets v2 state elision aggregates into wgpu_worker redundantStateCallsSkipped without double counting', () => {
  const gpu = createTrackingMockGpu();
  const proc = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    enableAsyncPipelines: false,
    requireNonBlackFrame: false,
  });
  installRenderStatePackets(proc);

  const setup = new StreamBuilder()
    .createShaderModule(1, SHADER_STAGE.VERTEX, SIMPLE_VS)
    .createShaderModule(
      2,
      SHADER_STAGE.FRAGMENT,
      '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }'
    )
    .createRenderPipeline(100, 1, 2);
  proc.executeCommandBuffer(setup.build());

  const frame = new StreamBuilder().beginFrame().setPipeline(100);
  for (let i = 0; i < 10; i += 1) {
    frame
      .setPipeline(100)
      .setViewport(0, 0, 1280, 720, 0, 1)
      .setScissorRect(0, 0, 1280, 720)
      .setStencilReference(5)
      .setBlendConstant(0.5, 0.5, 0.5, 1)
      .draw(3);
  }
  frame.present();
  proc.executeCommandBuffer(frame.build());

  const pass = gpu.renderPasses[0];
  assert.equal(pass.setPipeline.length, 1);
  assert.equal(pass.setViewport.length, 1);
  assert.equal(pass.setScissorRect.length, 1);
  assert.equal(pass.setStencilReference.length, 1);
  assert.equal(pass.setBlendConstant.length, 1);
  assert.equal(pass.draw.length, 10);

  const stats = proc.getStats();
  // 10 repeated setPipeline (after initial 1 -> 10 skips) + 9 skips each for viewport/scissor/stencil/blend = 46!
  assert.equal(stats.redundantStateSkippedBreakdown.pipeline, 10);
  assert.equal(stats.redundantStateSkippedBreakdown.viewport, 9);
  assert.equal(stats.redundantStateSkippedBreakdown.scissor, 9);
  assert.equal(stats.redundantStateSkippedBreakdown.stencilRef, 9);
  assert.equal(stats.redundantStateSkippedBreakdown.blendConstant, 9);
  assert.equal(stats.redundantStateCallsSkipped, 46);
});
