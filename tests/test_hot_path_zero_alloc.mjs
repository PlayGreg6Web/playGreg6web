import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const stagingMod = require('../web/workers/staging_pool.js');
const renderStateMod = require('../web/workers/render_state_packets.js');
const wgpuMod = require('../web/workers/wgpu_worker.js');

const { GpuStagingPool, STAGING_MODE, MAP_MODE } = stagingMod;
const {
  installRenderStatePackets,
  V2_OPCODES,
  SWAPCHAIN_TEXTURE_ID,
  validateIndirectCommand,
  sanitizeRenderPassAttachments,
} = renderStateMod;
const { WgpuCommandProcessor, OPCODES } = wgpuMod;

// ---------------------------------------------------------------------------
// Helpers: deterministic mock WebGPU devices & binary packet builders
// ---------------------------------------------------------------------------

function createStagingMockDevice(options = {}) {
  let nextBufferId = 1;
  const createdBuffers = [];
  const deferredFences = [];
  let writeBufferCalls = 0;

  const queue = {
    writeBuffer(buffer, bufferOffset, data) {
      writeBufferCalls += 1;
      const bytes =
        data instanceof Uint8Array
          ? data
          : ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            : new Uint8Array(data);
      if (buffer._data) {
        buffer._data.set(bytes, bufferOffset);
      }
    },
  };

  if (options.enableWorkDoneFence) {
    queue.onSubmittedWorkDone = () => {
      let resolve;
      const promise = new Promise((res) => {
        resolve = res;
      });
      deferredFences.push({ promise, resolve });
      return promise;
    };
  }

  const device = {
    queue,
    createBuffer(descriptor) {
      const id = nextBufferId++;
      const keepBacking = options.keepBacking !== false;
      const backing = keepBacking ? new ArrayBuffer(descriptor.size) : null;
      const dataView = backing ? new Uint8Array(backing) : null;
      const buffer = {
        id,
        label: descriptor.label || `buf-${id}`,
        size: descriptor.size,
        usage: descriptor.usage,
        mappedAtCreation: Boolean(descriptor.mappedAtCreation),
        _data: dataView,
        _mapped: Boolean(descriptor.mappedAtCreation),
        unmapCount: 0,
        mapAsyncCount: 0,
        destroyCount: 0,
        getMappedRange(offset = 0, size = descriptor.size - offset) {
          if (!this._mapped) {
            throw new Error(`Buffer ${this.id} is not mapped`);
          }
          if (!backing) {
            throw new Error('Backing disabled');
          }
          if (offset === 0 && size === descriptor.size) {
            return backing;
          }
          return backing.slice(offset, offset + size);
        },
        unmap() {
          if (!this._mapped) {
            throw new Error(`Buffer ${this.id} is already unmapped`);
          }
          this._mapped = false;
          this.unmapCount += 1;
        },
        async mapAsync(mode) {
          assert.equal(mode, MAP_MODE.WRITE);
          this.mapAsyncCount += 1;
          this._mapped = true;
        },
        destroy() {
          this._mapped = false;
          this.destroyCount += 1;
        },
      };
      createdBuffers.push(buffer);
      return buffer;
    },
  };

  return {
    device,
    createdBuffers,
    deferredFences,
    getWriteBufferCalls: () => writeBufferCalls,
  };
}

function createStagingMockEncoder(options = {}) {
  const captureSnapshots = options.captureSnapshots === true;
  const bufferCopies = [];
  const textureCopies = [];
  let bufferCopyCount = 0;
  let textureCopyCount = 0;
  return {
    bufferCopies,
    textureCopies,
    getBufferCopyCount: () => bufferCopyCount,
    getTextureCopyCount: () => textureCopyCount,
    copyBufferToBuffer(srcBuffer, srcOffset, dstBuffer, dstOffset, size) {
      bufferCopyCount += 1;
      if (captureSnapshots) {
        bufferCopies.push({
          srcBuffer,
          srcOffset,
          dstBuffer,
          dstOffset,
          size,
          snapshot: srcBuffer._data ? srcBuffer._data.slice(srcOffset, srcOffset + size) : null,
        });
      }
    },
    copyBufferToTexture(source, destination, copySize) {
      textureCopyCount += 1;
      if (captureSnapshots) {
        const totalBytes = source.bytesPerRow * source.rowsPerImage * copySize.depthOrArrayLayers;
        textureCopies.push({
          source: { ...source },
          destination: { ...destination, origin: { ...destination.origin } },
          copySize: { ...copySize },
          snapshot: source.buffer._data
            ? source.buffer._data.slice(source.offset, source.offset + totalBytes)
            : null,
        });
      }
    },
  };
}

function createRenderMockGpu(options = {}) {
  const recordPassOps = options.recordPassOps !== false;
  const passOps = [];
  let passOpCount = 0;
  let nextHandle = 1;
  let createBindGroupCalls = 0;
  let createViewCalls = 0;
  const swapchainView = { kind: 'GPUTextureView', id: 'swapchain-view', width: 1280, height: 720 };
  const swapchainTexture = {
    kind: 'GPUTexture',
    width: 1280,
    height: 720,
    format: 'bgra8unorm',
    createView: () => swapchainView,
  };
  const ctx = {
    canvas: { width: 1280, height: 720 },
    getCurrentTexture: () => swapchainTexture,
  };

  const makePass = (kind) => {
    const rec = (op, a0, a1, a2, a3, a4, a5) => {
      passOpCount += 1;
      if (recordPassOps) {
        passOps.push([kind, op, a0, a1, a2, a3, a4, a5]);
      }
    };
    return {
      setPipeline: (p) => rec('setPipeline', p && p.handle),
      setBindGroup: (i, bg, offsets) =>
        rec('setBindGroup', i, bg && bg.handle, offsets ? Array.from(offsets) : null),
      setVertexBuffer: (s, b, o) => rec('setVertexBuffer', s, b && b.handle, o),
      setIndexBuffer: (b, f, o) => rec('setIndexBuffer', b && b.handle, f, o),
      setViewport: (x, y, w, h, minD, maxD) => rec('setViewport', x, y, w, h, minD, maxD),
      setScissorRect: (x, y, w, h) => rec('setScissorRect', x, y, w, h),
      setBlendConstant: (c) => rec('setBlendConstant', c.r, c.g, c.b, c.a),
      setStencilReference: (r) => rec('setStencilReference', r),
      draw: (vc, ic, fv, fi) => rec('draw', vc, ic, fv, fi),
      drawIndexed: (ic, inst, fi, bv, finst) => rec('drawIndexed', ic, inst, fi, bv, finst),
      drawIndirect: (b, off) => rec('drawIndirect', b && b.handle, off),
      drawIndexedIndirect: (b, off) => rec('drawIndexedIndirect', b && b.handle, off),
      dispatchWorkgroups: (x, y, z) => rec('dispatchWorkgroups', x, y, z),
      dispatchWorkgroupsIndirect: (b, off) => rec('dispatchWorkgroupsIndirect', b && b.handle, off),
      end: () => rec('end'),
    };
  };

  const device = {
    limits: {
      maxComputeWorkgroupsPerDimension: 65535,
      minUniformBufferOffsetAlignment: 256,
      minStorageBufferOffsetAlignment: 256,
      maxUniformBufferBindingSize: 65536,
      maxStorageBufferBindingSize: 134217728,
    },
    createBuffer(desc) {
      return {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        desc,
        destroy() {},
      };
    },
    createShaderModule(desc) {
      return { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
    },
    createTexture(desc) {
      const handle = nextHandle++;
      const tex = {
        kind: 'GPUTexture',
        handle,
        width: desc.size.width,
        height: desc.size.height,
        depthOrArrayLayers: desc.size.depthOrArrayLayers || 1,
        mipLevelCount: desc.mipLevelCount || 1,
        sampleCount: desc.sampleCount || 1,
        dimension: desc.dimension || '2d',
        format: desc.format,
        usage: desc.usage,
        desc,
        createView(viewDesc) {
          createViewCalls += 1;
          return {
            kind: 'GPUTextureView',
            handle: nextHandle++,
            textureHandle: handle,
            viewDesc: viewDesc ? { ...viewDesc } : undefined,
          };
        },
        destroy() {},
      };
      return tex;
    },
    createSampler(desc) {
      return { kind: 'GPUSampler', handle: nextHandle++, desc };
    },
    createBindGroupLayout(desc) {
      return { kind: 'GPUBindGroupLayout', handle: nextHandle++, desc };
    },
    createPipelineLayout(desc) {
      return { kind: 'GPUPipelineLayout', handle: nextHandle++, desc };
    },
    createRenderPipeline(desc) {
      const handle = nextHandle++;
      const autoBgl = { kind: 'GPUBindGroupLayout', handle: nextHandle++, autoFor: handle };
      return {
        kind: 'GPURenderPipeline',
        handle,
        desc,
        getBindGroupLayout(idx) {
          if (idx === 0) return autoBgl;
          throw new Error('unsupported group');
        },
      };
    },
    createComputePipeline(desc) {
      const handle = nextHandle++;
      const autoBgl = { kind: 'GPUBindGroupLayout', handle: nextHandle++, autoFor: handle };
      return {
        kind: 'GPUComputePipeline',
        handle,
        desc,
        getBindGroupLayout(idx) {
          if (idx === 0) return autoBgl;
          throw new Error('unsupported group');
        },
      };
    },
    createBindGroup(desc) {
      createBindGroupCalls += 1;
      return { kind: 'GPUBindGroup', handle: nextHandle++, desc };
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          passOpCount += 1;
          if (recordPassOps) {
            passOps.push([
              'render',
              'beginRenderPass',
              desc.colorAttachments.length,
              Boolean(desc.depthStencilAttachment),
            ]);
          }
          return makePass('render');
        },
        beginComputePass() {
          passOpCount += 1;
          if (recordPassOps) {
            passOps.push(['compute', 'beginComputePass']);
          }
          return makePass('compute');
        },
        copyBufferToBuffer() {},
        finish: () => ({ kind: 'GPUCommandBuffer', handle: nextHandle++ }),
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {},
    },
  };

  return {
    device,
    ctx,
    passOps,
    getPassOpCount: () => passOpCount,
    getCreateBindGroupCalls: () => createBindGroupCalls,
    getCreateViewCalls: () => createViewCalls,
  };
}

class PacketBuilder {
  constructor() {
    this.chunks = [];
    this.totalBytes = 0;
  }
  pushPacket(opcode, payloadBytes, writer) {
    const buf = new Uint8Array(8 + payloadBytes);
    const dv = new DataView(buf.buffer);
    dv.setUint16(0, opcode, true);
    dv.setUint32(4, payloadBytes, true);
    if (writer) writer(dv, 8, buf);
    this.chunks.push(buf);
    this.totalBytes += buf.byteLength;
    return this;
  }
  toUint8Array() {
    const out = new Uint8Array(this.totalBytes);
    let off = 0;
    for (let i = 0; i < this.chunks.length; i += 1) {
      out.set(this.chunks[i], off);
      off += this.chunks[i].byteLength;
    }
    return out;
  }
}

function appendCreateShaderModule(pb, id, stage, wgsl) {
  const textBytes = new TextEncoder().encode(wgsl);
  return pb.pushPacket(OPCODES.CREATE_SHADER_MODULE, 16 + textBytes.byteLength, (dv, o, buf) => {
    dv.setUint32(o, id, true);
    dv.setUint32(o + 4, stage, true);
    dv.setUint32(o + 8, textBytes.byteLength, true);
    buf.set(textBytes, o + 16);
  });
}

function appendCreateBuffer(pb, id, size, usage) {
  return pb.pushPacket(OPCODES.CREATE_BUFFER, 16, (dv, o) => {
    dv.setUint32(o, id, true);
    dv.setUint32(o + 4, usage, true);
    dv.setBigUint64(o + 8, BigInt(size), true);
  });
}

function appendCreateTexture2(pb, { id, width, height, depth = 1, mips = 1, format = 22, usage = 0x15 }) {
  return pb.pushPacket(V2_OPCODES.CREATE_TEXTURE2, 40, (dv, o) => {
    dv.setUint32(o, id, true);
    dv.setUint32(o + 4, width, true);
    dv.setUint32(o + 8, height, true);
    dv.setUint32(o + 12, depth, true);
    dv.setUint32(o + 16, mips, true);
    dv.setUint32(o + 20, 1, true); // sampleCount
    dv.setUint32(o + 24, usage, true);
    dv.setUint8(o + 28, 1); // 2d
    dv.setUint8(o + 29, format);
    dv.setUint8(o + 30, 0); // viewFormatCount
  });
}

function appendCreateSampler2(pb, id) {
  return pb.pushPacket(V2_OPCODES.CREATE_SAMPLER2, 24, (dv, o) => {
    dv.setUint32(o, id, true);
    dv.setUint8(o + 4, 0);
    dv.setUint8(o + 5, 0);
    dv.setUint8(o + 6, 0);
    dv.setUint8(o + 7, 1); // linear
    dv.setUint8(o + 8, 1);
    dv.setUint8(o + 9, 1);
    dv.setUint16(o + 12, 1, true);
    dv.setFloat32(o + 16, 0, true);
    dv.setFloat32(o + 20, 32, true);
  });
}

function appendCreateBindGroupLayout(pb, id, entries) {
  return pb.pushPacket(V2_OPCODES.CREATE_BIND_GROUP_LAYOUT, 24 + entries.length * 24, (dv, o) => {
    dv.setUint32(o, id, true);
    dv.setUint32(o + 4, entries.length, true);
    dv.setUint32(o + 8, 0, true);
    dv.setBigUint64(o + 16, 0n, true);
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i];
      const base = o + 24 + i * 24;
      dv.setUint32(base, e.binding, true);
      dv.setUint32(base + 4, e.visibility ?? 0x7, true);
      dv.setUint8(base + 8, e.type);
      dv.setUint8(base + 9, e.subtype ?? 0);
      dv.setUint8(base + 10, e.dimension ?? 2);
      dv.setUint8(base + 11, e.flags ?? 0);
      dv.setUint8(base + 12, e.storageFormat ?? 0);
      dv.setBigUint64(base + 16, BigInt(e.minBindingSize ?? 0), true);
    }
  });
}

function appendCreatePipelineLayout(pb, id, bglIds) {
  return pb.pushPacket(V2_OPCODES.CREATE_PIPELINE_LAYOUT, 24 + bglIds.length * 4, (dv, o) => {
    dv.setUint32(o, id, true);
    dv.setUint32(o + 4, bglIds.length, true);
    dv.setUint32(o + 8, 0, true);
    dv.setBigUint64(o + 16, 0n, true);
    for (let i = 0; i < bglIds.length; i += 1) {
      dv.setUint32(o + 24 + i * 4, bglIds[i], true);
    }
  });
}

function appendCreateRenderPipeline2(pb, { id, vsId, fsId, layoutId, colorFormat = 22, depthFormat = 0 }) {
  return pb.pushPacket(V2_OPCODES.CREATE_RENDER_PIPELINE2, 16 + 376, (dv, o) => {
    dv.setUint32(o, id, true);
    dv.setUint32(o + 4, vsId, true);
    dv.setUint32(o + 8, fsId, true);
    dv.setUint32(o + 12, layoutId, true);
    const key = o + 16;
    dv.setUint32(key + 20, 1, true); // colorTargetCount
    dv.setUint8(key + 24, colorFormat);
    dv.setUint8(key + 24 + 2, 0x0f);
    if (depthFormat !== 0) {
      dv.setUint8(key + 120, depthFormat);
      dv.setUint8(key + 121, 1); // depthWriteEnabled
      dv.setUint8(key + 122, 2); // less
    }
    dv.setUint32(key + 160, 1, true); // sampleCount
    dv.setUint32(key + 164, 0xffffffff, true);
  });
}

function appendCreateComputePipeline2(pb, id, csId, layoutId) {
  return pb.pushPacket(V2_OPCODES.CREATE_COMPUTE_PIPELINE2, 32, (dv, o) => {
    dv.setUint32(o, id, true);
    dv.setUint32(o + 4, csId, true);
    dv.setUint32(o + 8, layoutId, true);
  });
}

function appendCreateBindGroup2(pb, id, layoutId, entries) {
  return pb.pushPacket(V2_OPCODES.CREATE_BIND_GROUP2, 16 + entries.length * 48, (dv, o) => {
    dv.setUint32(o, id, true);
    dv.setUint32(o + 4, layoutId, true);
    dv.setUint32(o + 8, entries.length, true);
    dv.setUint32(o + 12, 0, true);
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i];
      const base = o + 16 + i * 48;
      dv.setUint32(base, e.binding, true);
      dv.setUint32(base + 4, e.type, true);
      dv.setUint32(base + 8, e.resourceId, true);
      if (e.type === 0) {
        dv.setBigUint64(base + 16, BigInt(e.offset ?? 0), true);
        dv.setBigUint64(base + 24, BigInt(e.size ?? 0), true);
      }
    }
  });
}

function appendBeginRenderPass(pb, colorTexId, depthTexId = 0) {
  return pb.pushPacket(V2_OPCODES.BEGIN_RENDER_PASS, 368, (dv, o) => {
    dv.setUint32(o, 1, true); // colorCount = 1
    dv.setUint32(o + 4, depthTexId !== 0 ? 1 : 0, true); // hasDepth
    // Color attachment 0 at offset + 8 (40 bytes)
    dv.setUint32(o + 8, colorTexId, true);
    dv.setUint8(o + 12, 2); // clear
    dv.setUint8(o + 13, 1); // store
    dv.setFloat32(o + 32, 0.1, true);
    dv.setFloat32(o + 36, 0.2, true);
    dv.setFloat32(o + 40, 0.3, true);
    dv.setFloat32(o + 44, 1.0, true);
    if (depthTexId !== 0) {
      // Depth-stencil attachment at offset + 328 (40 bytes)
      const ds = o + 328;
      dv.setUint32(ds, depthTexId, true);
      dv.setUint8(ds + 4, 2); // depthLoadOp = clear
      dv.setUint8(ds + 5, 1); // depthStoreOp = store
      dv.setFloat32(ds + 28, 1.0, true); // depthClearValue
    }
  });
}

function appendSetViewport(pb, x, y, w, h, minD = 0, maxD = 1) {
  return pb.pushPacket(V2_OPCODES.SET_VIEWPORT, 24, (dv, o) => {
    dv.setFloat32(o, x, true);
    dv.setFloat32(o + 4, y, true);
    dv.setFloat32(o + 8, w, true);
    dv.setFloat32(o + 12, h, true);
    dv.setFloat32(o + 16, minD, true);
    dv.setFloat32(o + 20, maxD, true);
  });
}

function appendSetScissor(pb, x, y, w, h) {
  return pb.pushPacket(V2_OPCODES.SET_SCISSOR_RECT, 16, (dv, o) => {
    dv.setUint32(o, x, true);
    dv.setUint32(o + 4, y, true);
    dv.setUint32(o + 8, w, true);
    dv.setUint32(o + 12, h, true);
  });
}

function appendSetBlendConstant(pb, r, g, b, a) {
  return pb.pushPacket(V2_OPCODES.SET_BLEND_CONSTANT, 16, (dv, o) => {
    dv.setFloat32(o, r, true);
    dv.setFloat32(o + 4, g, true);
    dv.setFloat32(o + 8, b, true);
    dv.setFloat32(o + 12, a, true);
  });
}

function appendSetStencilRef(pb, ref) {
  return pb.pushPacket(V2_OPCODES.SET_STENCIL_REFERENCE, 4, (dv, o) => {
    dv.setUint32(o, ref, true);
  });
}

function appendSetBindGroup2(pb, groupIndex, bgId, dynOffsets = []) {
  return pb.pushPacket(V2_OPCODES.SET_BIND_GROUP2, 16 + dynOffsets.length * 4, (dv, o) => {
    dv.setUint32(o, groupIndex, true);
    dv.setUint32(o + 4, bgId, true);
    dv.setUint32(o + 8, dynOffsets.length, true);
    for (let i = 0; i < dynOffsets.length; i += 1) {
      dv.setUint32(o + 16 + i * 4, dynOffsets[i], true);
    }
  });
}

function appendSetPipeline(pb, pipelineId) {
  return pb.pushPacket(OPCODES.SET_PIPELINE, 4, (dv, o) => dv.setUint32(o, pipelineId, true));
}

function appendSetComputePipeline(pb, pipelineId) {
  return pb.pushPacket(OPCODES.SET_COMPUTE_PIPELINE, 4, (dv, o) => dv.setUint32(o, pipelineId, true));
}

function appendSetIndexBuffer(pb, bufferId, format = 1, offset = 0) {
  return pb.pushPacket(OPCODES.SET_INDEX_BUFFER, 16, (dv, o) => {
    dv.setUint32(o, bufferId, true);
    dv.setUint32(o + 4, format, true);
    dv.setBigUint64(o + 8, BigInt(offset), true);
  });
}

function appendDrawIndexed(pb, indexCount, instanceCount = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0) {
  return pb.pushPacket(OPCODES.DRAW_INDEXED, 20, (dv, o) => {
    dv.setUint32(o, indexCount, true);
    dv.setUint32(o + 4, instanceCount, true);
    dv.setUint32(o + 8, firstIndex, true);
    dv.setInt32(o + 12, baseVertex, true);
    dv.setUint32(o + 16, firstInstance, true);
  });
}

function appendDrawIndexedIndirect(pb, bufferId, offset) {
  return pb.pushPacket(OPCODES.DRAW_INDEXED_INDIRECT, 16, (dv, o) => {
    dv.setUint32(o, bufferId, true);
    dv.setBigUint64(o + 8, BigInt(offset), true);
  });
}

function appendDispatchCompute(pb, x, y, z) {
  return pb.pushPacket(OPCODES.DISPATCH_COMPUTE, 12, (dv, o) => {
    dv.setUint32(o, x, true);
    dv.setUint32(o + 4, y, true);
    dv.setUint32(o + 8, z, true);
  });
}

function appendDispatchIndirect(pb, bufferId, offset) {
  return pb.pushPacket(OPCODES.DISPATCH_INDIRECT, 16, (dv, o) => {
    dv.setUint32(o, bufferId, true);
    dv.setBigUint64(o + 8, BigInt(offset), true);
  });
}

// ---------------------------------------------------------------------------
// 1. GpuStagingPool O(1) ring-queue FIFO recycling, wrap-around, and churn tests
// ---------------------------------------------------------------------------

test('#2139 staging_pool: strict FIFO chunk recycling across 3,000 frames and ring wrap-arounds with zero steady-state GPUBuffer churn', () => {
  const { device, createdBuffers } = createStagingMockDevice({ keepBacking: false });
  const pool = new GpuStagingPool({
    device,
    chunkSize: 4096,
    maxPoolBytes: 64 * 1024,
    framesInFlight: 3,
  });
  const encoder = createStagingMockEncoder();
  const dstBuffer = { id: 'dst-vb' };
  const payload = new Uint8Array(512).fill(0x3c);
  const scratchOut = {};

  // Warmup 3 frames -> 3 chunks created (ids 1, 2, 3).
  const warmupOrder = [];
  for (let f = 0; f < 3; f += 1) {
    pool.beginFrame(f);
    pool.stageBufferUpload(encoder, dstBuffer, 0, payload, scratchOut);
    warmupOrder.push(scratchOut.buffer.id);
    pool.endFrame();
  }
  assert.deepEqual(warmupOrder, [1, 2, 3]);
  assert.equal(createdBuffers.length, 3);

  // Run 3,000 steady-state frames (wrapping the circular queue > 180 times) and verify strict FIFO!
  for (let f = 3; f < 3003; f += 1) {
    const info = pool.beginFrame(f);
    assert.equal(info.recycledCount, 1);
    pool.stageBufferUpload(encoder, dstBuffer, 0, payload, scratchOut);
    assert.equal(
      scratchOut.buffer.id,
      warmupOrder[f % 3],
      `frame ${f} must recycle in strict FIFO order`
    );
    pool.endFrame();
  }

  assert.equal(createdBuffers.length, 3, 'zero new GPUBuffers after 3-frame warmup');
  const stats = pool.getStats();
  assert.equal(stats.framesCompleted, 3003);
  assert.equal(stats.bufferReuseHits, 3000);
  assert.equal(stats.bufferReuseMisses, 3);
  assert.equal(stats.zeroAllocAcquires, 3003);
  assert.equal(stats.inFlightChunks, 3);
  assert.equal(stats.recycledChunks, 0);
  assert.equal(stats.allocatedBytes, 3 * 4096);
  assert.equal(encoder.getBufferCopyCount(), 3003);

  pool.destroy();
  assert.equal(pool.getStats().allocatedBytes, 0);
  assert.equal(pool.getStats().inFlightChunks, 0);
  assert.equal(pool.getStats().recycledChunks, 0);
});

test('#2139 staging_pool: burst > 64 chunks forces ring growth, out-of-order fence reclaim, mapped mode, budget eviction, and reset/destroy invariants', async () => {
  const { device, createdBuffers, deferredFences } = createStagingMockDevice({
    enableWorkDoneFence: true,
  });
  const pool = new GpuStagingPool({
    device,
    chunkSize: 1024,
    maxPoolBytes: 128 * 1024,
    framesInFlight: 8,
    mode: STAGING_MODE.MAPPED,
  });

  // Burst 40 distinct chunks in frame 0 (exceeds initial ring capacity 16 -> exercises _repack growth).
  pool.beginFrame(0);
  const frame0BufIds = [];
  const out = {};
  for (let i = 0; i < 40; i += 1) {
    pool.allocate(900, 4, out);
    frame0BufIds.push(out.buffer.id);
    if (i % 5 === 0) {
      // Explicit mid-frame release retires individual chunks with single-chunk fence path!
      assert.equal(pool.release(out), true);
    }
  }
  const end0 = pool.endFrame();
  assert.equal(end0.chunksRetired, 32); // 40 - 8 already released
  assert.equal(createdBuffers.length, 40);
  assert.equal(pool.getStats().inFlightChunks, 40);
  assert.equal(deferredFences.length, 9); // 8 single releases + 1 endFrame batch

  // Resolve fences OUT OF ORDER (middle fences first, then front, then batch) to exercise tombstones + O(1) remove!
  deferredFences[4].resolve();
  deferredFences[2].resolve();
  deferredFences[7].resolve();
  await Promise.all([
    deferredFences[4].promise,
    deferredFences[2].promise,
    deferredFences[7].promise,
  ]);
  await pool.waitForPendingMaps();

  const reclaimedMid = pool.recycleCompleted();
  assert.equal(reclaimedMid, 3);
  assert.equal(pool.getStats().recycledChunks, 3);
  assert.equal(pool.getStats().inFlightChunks, 37);

  // Calling recycleCompleted again when no new fences resolved takes O(1) _gpuDoneInFlightCount === 0 fast path.
  assert.equal(pool.recycleCompleted(), 0);

  // Resolve all remaining fences and verify all 40 chunks recycle cleanly.
  for (let i = 0; i < deferredFences.length; i += 1) {
    deferredFences[i].resolve();
  }
  await Promise.all(deferredFences.map((f) => f.promise));
  await pool.waitForPendingMaps();

  const begin1 = pool.beginFrame(1);
  assert.equal(begin1.recycledCount, 37);
  assert.equal(pool.getStats().recycledChunks, 40);
  assert.equal(pool.getStats().inFlightChunks, 0);

  // Allocate an oversized chunk that requires budget eviction of idle standard chunks.
  const huge = pool.allocate(100 * 1024, 4);
  assert.equal(huge.size, 100 * 1024);
  assert.ok(pool.getStats().allocatedBytes <= 128 * 1024);
  assert.ok(pool.getStats().evictedCount > 0);

  // Verify OOM throws cleanly without corrupting invariants.
  const statsBeforeOom = pool.getStats();
  assert.throws(() => pool.allocate(64 * 1024, 4), RangeError);
  const statsAfterOom = pool.getStats();
  assert.equal(statsAfterOom.allocatedBytes, statsBeforeOom.allocatedBytes);
  assert.equal(statsAfterOom.recycledChunks, statsBeforeOom.recycledChunks);
  assert.equal(statsAfterOom.inFlightChunks, statsBeforeOom.inFlightChunks);

  // resetStats re-baselines telemetry while keeping live buffers intact.
  const reset = pool.resetStats();
  assert.equal(reset.allocationCount, 0);
  assert.equal(reset.peakAllocatedBytes, reset.allocatedBytes);

  pool.endFrame();
  pool.destroy();
  pool.destroy(); // idempotent
  assert.equal(pool.getStats().allocatedBytes, 0);
  for (const buf of createdBuffers) {
    assert.equal(buf.destroyCount, 1);
  }
});

test('#2139 staging_pool: stageTextureUpload & stageBufferUpload bit-identical snapshots across small/large rows, SAB, chunked uploads, and scratch reuse', () => {
  const { device } = createStagingMockDevice({ keepBacking: true });
  const pool = new GpuStagingPool({
    device,
    chunkSize: 64 * 1024,
    maxPoolBytes: 256 * 1024,
    maxWriteChunkBytes: 4096,
    framesInFlight: 2,
  });
  const encoder = createStagingMockEncoder({ captureSnapshots: true });
  const dstTex = { id: 'tex-0' };
  const dstBuf = { id: 'buf-0' };
  const texOut = {};
  const bufOut = {};

  pool.beginFrame(0);

  // 1. Small unaligned rows (<= 64B, uses zero-subarray byte-copy fast path): 12 bytes/row x 5 rows.
  const smallSrc = new Uint8Array(12 * 5);
  for (let i = 0; i < smallSrc.length; i += 1) smallSrc[i] = (i * 7 + 3) & 0xff;
  pool.stageTextureUpload(
    encoder,
    dstTex,
    smallSrc,
    { width: 3, height: 5, bytesPerRow: 12, origin: { x: 2, y: 4, z: 0 } },
    texOut
  );
  assert.equal(texOut.bytesPerRow, 256);
  assert.equal(texOut.size, 256 * 5);

  // 2. Large unaligned rows (> 64B, uses memcpy subarray path): 100 bytes/row x 4 rows.
  const largeSrc = new Uint8Array(100 * 4);
  for (let i = 0; i < largeSrc.length; i += 1) largeSrc[i] = (i * 13 + 11) & 0xff;
  pool.stageTextureUpload(
    encoder,
    dstTex,
    largeSrc,
    { width: 25, height: 4, bytesPerRow: 100, mipLevel: 2, origin: [1, 2, 0] },
    texOut
  );

  // 3. SharedArrayBuffer aligned texture upload (no extra subarray when exact length).
  const sab = new SharedArrayBuffer(512);
  const sabView = new Uint8Array(sab);
  for (let i = 0; i < 512; i += 1) sabView[i] = (0xa0 + i) & 0xff;
  pool.stageTextureUpload(
    encoder,
    dstTex,
    sabView,
    { width: 64, height: 2, bytesPerRow: 256 },
    texOut
  );

  // 4. Chunked unaligned buffer upload (> maxWriteChunkBytes = 4096): 5003 bytes -> pads to 5004.
  const bigBuf = new Uint8Array(5003);
  for (let i = 0; i < bigBuf.length; i += 1) bigBuf[i] = (i * 3 + 1) & 0xff;
  pool.stageBufferUpload(encoder, dstBuf, 16, bigBuf, bufOut);
  assert.equal(bufOut.size, 5004);
  assert.equal(bufOut.srcByteLength, 5003);

  // Verify every single captured snapshot & descriptor field has zero aliasing or corruption!
  assert.equal(encoder.textureCopies.length, 3);
  const c0 = encoder.textureCopies[0];
  assert.deepEqual(c0.destination.origin, { x: 2, y: 4, z: 0 });
  assert.equal(c0.destination.mipLevel, 0);
  for (let r = 0; r < 5; r += 1) {
    for (let col = 0; col < 12; col += 1) {
      assert.equal(c0.snapshot[r * 256 + col], smallSrc[r * 12 + col]);
    }
    for (let pad = 12; pad < 256; pad += 1) {
      assert.equal(c0.snapshot[r * 256 + pad], 0);
    }
  }

  const c1 = encoder.textureCopies[1];
  assert.deepEqual(c1.destination.origin, { x: 1, y: 2, z: 0 });
  assert.equal(c1.destination.mipLevel, 2);
  for (let r = 0; r < 4; r += 1) {
    for (let col = 0; col < 100; col += 1) {
      assert.equal(c1.snapshot[r * 256 + col], largeSrc[r * 100 + col]);
    }
    for (let pad = 100; pad < 256; pad += 1) {
      assert.equal(c1.snapshot[r * 256 + pad], 0);
    }
  }

  const c2 = encoder.textureCopies[2];
  assert.deepEqual(Array.from(c2.snapshot), Array.from(sabView));

  assert.equal(encoder.bufferCopies.length, 2);
  assert.equal(encoder.bufferCopies[0].size, 4096);
  assert.equal(encoder.bufferCopies[1].size, 908);
  assert.deepEqual(Array.from(encoder.bufferCopies[0].snapshot), Array.from(bigBuf.subarray(0, 4096)));
  assert.deepEqual(
    Array.from(encoder.bufferCopies[1].snapshot.subarray(0, 907)),
    Array.from(bigBuf.subarray(4096, 5003))
  );
  assert.equal(encoder.bufferCopies[1].snapshot[907], 0);

  pool.endFrame();
  pool.destroy();
});

// ---------------------------------------------------------------------------
// 2. RenderStatePackets heavy 3D scene draw stream (10,000+ ops) & parity tests
// ---------------------------------------------------------------------------

test('#2139 render_state_packets: 10,000+ state-change + indexed-draw + indirect + compute + staging ops produce bit-identical commands and exact telemetry', () => {
  const gpu = createRenderMockGpu({ recordPassOps: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    width: 1280,
    height: 720,
    enableResourceReplay: false,
    requireNonBlackFrame: false,
  });
  const handler = installRenderStatePackets(processor);

  const stagingMock = createStagingMockDevice({ keepBacking: false });
  const stagingPool = new GpuStagingPool({
    device: stagingMock.device,
    chunkSize: 64 * 1024,
    maxPoolBytes: 512 * 1024,
    framesInFlight: 3,
  });
  const stagingEnc = createStagingMockEncoder();
  const uploadDst = { id: 'per-draw-ubo' };
  const uploadBytes = new Uint8Array(64).fill(0x55);
  const uploadOut = {};

  // Setup resources once: VS, FS, CS, UBO, IB, indirect buffer, offscreen color + depth textures, sampler, layouts, pipelines, bind groups.
  const setup = new PacketBuilder();
  appendCreateShaderModule(
    setup,
    1,
    0,
    '@group(0) @binding(0) var<uniform> ubo: vec4<f32>; @vertex fn vs_main() -> @builtin(position) vec4<f32> { return ubo; }'
  );
  appendCreateShaderModule(
    setup,
    2,
    1,
    '@group(0) @binding(1) var tex0: texture_2d<f32>; @group(0) @binding(2) var smp0: sampler; @fragment fn fs_main() -> @location(0) vec4<f32> { return textureSample(tex0, smp0, vec2<f32>(0.5, 0.5)); }'
  );
  appendCreateShaderModule(
    setup,
    3,
    2,
    '@group(0) @binding(0) var<storage, read_write> outBuf: array<u32>; @compute @workgroup_size(1) fn cs_main() { outBuf[0] = 1u; }'
  );
  // Buffers: 10=UBO (UNIFORM|COPY_DST), 11=IB (INDEX), 12=Indirect (INDIRECT|STORAGE), 13=Storage
  appendCreateBuffer(setup, 10, 65536, 0x0040 | 0x0008);
  appendCreateBuffer(setup, 11, 4096, 0x0010);
  appendCreateBuffer(setup, 12, 4096, 0x0100 | 0x0080);
  appendCreateBuffer(setup, 13, 4096, 0x0080);
  // Textures: 20=Color (2D, 2 mips, RENDER_ATTACHMENT|TEXTURE_BINDING), 21=Depth (depth24plus, RENDER_ATTACHMENT)
  appendCreateTexture2(setup, { id: 20, width: 1280, height: 720, mips: 2, format: 22, usage: 0x15 });
  appendCreateTexture2(setup, { id: 21, width: 1280, height: 720, mips: 1, format: 46, usage: 0x10 });
  appendCreateSampler2(setup, 30);
  // Render BGL 40: binding 0 dynamic uniform, binding 1 texture, binding 2 sampler
  appendCreateBindGroupLayout(setup, 40, [
    { binding: 0, type: 0, subtype: 0, flags: 0x1, minBindingSize: 256 },
    { binding: 1, type: 2, subtype: 0, dimension: 2 },
    { binding: 2, type: 1, subtype: 0 },
  ]);
  // Compute BGL 41: binding 0 storage buffer
  appendCreateBindGroupLayout(setup, 41, [{ binding: 0, visibility: 0x4, type: 0, subtype: 1 }]);
  appendCreatePipelineLayout(setup, 50, [40]);
  appendCreatePipelineLayout(setup, 51, [41]);
  appendCreateRenderPipeline2(setup, { id: 60, vsId: 1, fsId: 2, layoutId: 50, colorFormat: 22, depthFormat: 46 });
  appendCreateComputePipeline2(setup, 61, 3, 51);
  appendCreateBindGroup2(setup, 70, 40, [
    { binding: 0, type: 0, resourceId: 10, offset: 0, size: 256 },
    { binding: 1, type: 1, resourceId: 20 },
    { binding: 2, type: 2, resourceId: 30 },
  ]);
  appendCreateBindGroup2(setup, 71, 41, [{ binding: 0, type: 0, resourceId: 13, offset: 0, size: 256 }]);

  processor.executeCommandBuffer(setup.toUint8Array());
  assert.equal(handler.stats.errors, 0, handler.lastError);

  // Build one heavy 3D frame: 600 draws per frame with viewport/scissor/blend/stencil/bindgroup/drawIndexed/indirect/compute (~4,800 commands per frame).
  // Across 3 frames = 14,400+ commands (> 10,000 ops)!
  const DRAWS_PER_FRAME = 600;
  const framePb = new PacketBuilder();
  appendBeginRenderPass(framePb, 20, 21);
  appendSetPipeline(framePb, 60);
  appendSetIndexBuffer(framePb, 11, 1, 0);
  for (let d = 0; d < DRAWS_PER_FRAME; d += 1) {
    // Alternate redundant & changing state to exercise elision + state updates.
    appendSetViewport(framePb, 0, 0, 1280, 720, 0, 1);
    appendSetScissor(framePb, 0, 0, 1280, 720);
    appendSetBlendConstant(framePb, 1, 1, 1, 1);
    appendSetStencilRef(framePb, d & 1);
    appendSetBindGroup2(framePb, 0, 70, [(d % 8) * 256]);
    appendDrawIndexed(framePb, 36, 1, 0, 0, 0);
    if (d % 50 === 0) {
      appendDrawIndexedIndirect(framePb, 12, (d % 4) * 20);
    }
  }
  framePb.pushPacket(V2_OPCODES.END_RENDER_PASS, 0);
  // Also record a swapchain pass + compute pass each frame.
  appendBeginRenderPass(framePb, SWAPCHAIN_TEXTURE_ID, 0);
  appendSetPipeline(framePb, 60);
  appendSetBindGroup2(framePb, 0, 70, [0]);
  appendDrawIndexed(framePb, 6, 1, 0, 0, 0);
  framePb.pushPacket(V2_OPCODES.END_RENDER_PASS, 0);
  appendSetComputePipeline(framePb, 61);
  appendSetBindGroup2(framePb, 0, 71, []);
  appendDispatchCompute(framePb, 8, 4, 1);
  appendDispatchIndirect(framePb, 12, 0);
  framePb.pushPacket(OPCODES.PRESENT, 0);
  const frameBytes = framePb.toUint8Array();

  const perFrameSnapshots = [];
  const viewCallsAfterFrame0 = { count: 0 };
  for (let f = 0; f < 3; f += 1) {
    stagingPool.beginFrame(f);
    for (let u = 0; u < 100; u += 1) {
      stagingPool.stageBufferUpload(stagingEnc, uploadDst, (u * 64) & 0xffc, uploadBytes, uploadOut);
    }
    const startIdx = gpu.passOps.length;
    const res = processor.executeCommandBuffer(frameBytes);
    assert.equal(res.presents, 1);
    stagingPool.endFrame();

    perFrameSnapshots.push(gpu.passOps.slice(startIdx));
    if (f === 0) {
      viewCallsAfterFrame0.count = gpu.getCreateViewCalls();
    }
  }

  assert.equal(handler.stats.errors, 0, handler.lastError);
  // Every steady-state frame must record bit-identical WebGPU commands!
  assert.deepEqual(perFrameSnapshots[1], perFrameSnapshots[0]);
  assert.deepEqual(perFrameSnapshots[2], perFrameSnapshots[0]);

  // Zero additional GPUTexture.createView calls on frames 1 & 2 (100% view cache hits!).
  assert.equal(gpu.getCreateViewCalls(), viewCallsAfterFrame0.count);

  // Verify exact telemetry counters across the 3 frames.
  const t = handler.getTelemetry();
  assert.equal(t.elidedViewportSets, 3 * (DRAWS_PER_FRAME - 1));
  assert.equal(t.elidedScissorSets, 3 * (DRAWS_PER_FRAME - 1));
  assert.equal(t.elidedBlendConstantSets, 3 * (DRAWS_PER_FRAME - 1));
  assert.equal(t.elidedStencilRefSets, 0); // alternates 0, 1 every draw
  assert.equal(t.computeDispatchCount, 6); // 2 dispatches x 3 frames
  assert.equal(t.skippedInvalidIndirectOps, 0);
  assert.ok(t.textureViewCacheHits >= 4, 'offscreen color/depth views must hit cache on steady frames');

  // Verify frame command pool retained all 600 commands/frame without truncation churn.
  assert.equal(handler._vpCmdPool.length, DRAWS_PER_FRAME);
  assert.equal(handler._sbg2CmdPool.length, DRAWS_PER_FRAME + 2);

  stagingPool.destroy();
  handler.destroy();
});

test('#2139 render_state_packets: _preparePassCommandsForFlush in-place compaction, scratch-spill insertion, and O(1) warning ring buffer', () => {
  const gpu = createRenderMockGpu({ recordPassOps: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    width: 1280,
    height: 720,
    enableResourceReplay: false,
    requireNonBlackFrame: false,
  });
  const handler = installRenderStatePackets(processor);

  // Create an auto-layout fallback pipeline (explicit layout has binding 0 storage, WGSL declares uniform at binding 0).
  const setup = new PacketBuilder();
  appendCreateShaderModule(
    setup,
    101,
    0,
    '@group(0) @binding(0) var<uniform> u: vec4<f32>; @vertex fn vs_main() -> @builtin(position) vec4<f32> { return u; }'
  );
  appendCreateShaderModule(setup, 102, 1, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }');
  appendCreateBuffer(setup, 200, 1024, 0x0040 | 0x0008); // uniform
  appendCreateBuffer(setup, 201, 64, 0x0080); // missing INDIRECT usage -> triggers invalid indirect skip (drop!)
  appendCreateBindGroupLayout(setup, 300, [{ binding: 0, visibility: 0x2, type: 0, subtype: 1 }]); // storage -> mismatches WGSL uniform -> layout:'auto'
  appendCreatePipelineLayout(setup, 301, [300]);
  appendCreateRenderPipeline2(setup, { id: 400, vsId: 101, fsId: 102, layoutId: 301, colorFormat: 22 });
  appendCreateBindGroup2(setup, 500, 300, [{ binding: 0, type: 0, resourceId: 200, offset: 0, size: 256 }]);
  processor.executeCommandBuffer(setup.toUint8Array());
  assert.equal(handler.stats.errors, 0, handler.lastError);

  // Frame 1: tests insertion path (draw on auto-layout pipeline WITHOUT prior setBindGroup injects a RemappedSetBindGroupCommand!)
  // AND tests in-place compaction on drop (invalid drawIndirect is skipped).
  const passCmdArrayBefore = processor._passCommands;
  const f1 = new PacketBuilder();
  appendBeginRenderPass(f1, SWAPCHAIN_TEXTURE_ID, 0);
  appendSetPipeline(f1, 400);
  // No SetBindGroup2 here -> _preparePassCommandsForFlush must inject RemappedSetBindGroupCommand before DrawIndexed (insertion path)!
  appendDrawIndexed(f1, 3, 1, 0, 0, 0);
  // Invalid indirect buffer -> dropped by in-place compaction!
  appendDrawIndexedIndirect(f1, 201, 0);
  f1.pushPacket(V2_OPCODES.END_RENDER_PASS, 0);
  f1.pushPacket(OPCODES.PRESENT, 0);
  processor.executeCommandBuffer(f1.toUint8Array());

  // processor._passCommands identity must be preserved (never swapped to a freshly allocated array).
  assert.equal(processor._passCommands, passCmdArrayBefore);
  assert.equal(handler.getTelemetry().skippedInvalidIndirectOps, 1);
  const bgOps = gpu.passOps.filter((op) => op[1] === 'setBindGroup');
  const drawOps = gpu.passOps.filter((op) => op[1] === 'drawIndexed');
  assert.equal(bgOps.length, 1, 'auto-layout draw must receive injected remapped bind group');
  assert.equal(drawOps.length, 1);

  // Verify validateIndirectCommand & sanitizeRenderPassAttachments zero-alloc out scratch parity.
  const scratchInd = {};
  const validBuf = { size: 128, usage: 0x0100 };
  const indFresh = validateIndirectCommand('drawIndexedIndirect', validBuf, 20);
  const indScratch = validateIndirectCommand('drawIndexedIndirect', validBuf, 20, null, scratchInd);
  assert.equal(indScratch, scratchInd);
  assert.equal(indScratch.valid, indFresh.valid);
  assert.equal(indScratch.requiredBytes, indFresh.requiredBytes);

  const scratchRp = {};
  const cleanRpDesc = {
    colorAttachments: [{ view: { width: 640, height: 480 }, loadOp: 'clear', storeOp: 'store' }],
  };
  const rpFresh = sanitizeRenderPassAttachments(cleanRpDesc);
  const rpScratch = sanitizeRenderPassAttachments(cleanRpDesc, null, scratchRp);
  assert.equal(rpScratch, scratchRp);
  assert.equal(rpScratch.descriptor, rpFresh.descriptor);
  assert.equal(rpScratch.sanitizedCount, 0);
  assert.deepEqual(rpScratch.referenceSize, rpFresh.referenceSize);

  // Verify O(1) warning ring buffer preserves exact 32-item FIFO sliding window over 100 warnings.
  for (let i = 0; i < 100; i += 1) {
    handler._recordTelemetryWarning(`warn-${i}`);
  }
  const expectedWarnings = [];
  for (let i = 68; i < 100; i += 1) expectedWarnings.push(`warn-${i}`);
  assert.deepEqual(handler.getTelemetry().warnings, expectedWarnings);
  assert.deepEqual(handler.telemetry.warnings, expectedWarnings);

  // Verify getOrCreatePresentXPlaneBindGroup optional out scratch returns exact cached handles.
  const pxTexPb = new PacketBuilder();
  appendCreateTexture2(pxTexPb, { id: 900, width: 64, height: 64, format: 22, usage: 0x04 });
  appendCreateSampler2(pxTexPb, 901);
  appendCreateBindGroupLayout(pxTexPb, 902, [
    { binding: 0, type: 2, subtype: 0, dimension: 2 },
    { binding: 1, type: 1, subtype: 0 },
  ]);
  processor.executeCommandBuffer(pxTexPb.toUint8Array());
  const pxOut = {};
  const pxMiss = handler.getOrCreatePresentXPlaneBindGroup(
    { planeTextureId: 900, samplerId: 901, layoutId: 902 },
    pxOut
  );
  assert.equal(pxMiss, pxOut);
  assert.equal(pxMiss.hit, false);
  const pxBg = pxMiss.bindGroup;
  const pxHit = handler.getOrCreatePresentXPlaneBindGroup(
    { planeTextureId: 900, samplerId: 901, layoutId: 902 },
    pxOut
  );
  assert.equal(pxHit, pxOut);
  assert.equal(pxHit.hit, true);
  assert.equal(pxHit.bindGroup, pxBg);

  handler.destroy();
});

test('#2139 throughput: 2,000 concurrent in-flight chunks + 12,000 render-state packets complete rapidly with O(1) queue behavior and clean classic-worker IIFE reload', async () => {
  const t0 = performance.now();

  // 1. Stress O(1) _inFlightQueue & _onWorkDoneChunk with 2,000 simultaneously in-flight chunks!
  const { device, deferredFences } = createStagingMockDevice({
    keepBacking: false,
    enableWorkDoneFence: true,
  });
  const pool = new GpuStagingPool({
    device,
    chunkSize: 1024,
    maxPoolBytes: 4 * 1024 * 1024,
    framesInFlight: 3,
  });
  const allocOut = {};
  pool.beginFrame(0);
  for (let i = 0; i < 2000; i += 1) {
    pool.allocate(1024, 4, allocOut);
  }
  pool.endFrame();
  assert.equal(pool.getStats().inFlightChunks, 2000);
  deferredFences[0].resolve();
  await deferredFences[0].promise;
  await Promise.resolve();
  assert.equal(pool.recycleCompleted(), 2000);
  assert.equal(pool.getStats().recycledChunks, 2000);
  // Pop all 2,000 recycled chunks in O(1) FIFO order!
  pool.beginFrame(1);
  for (let i = 0; i < 2000; i += 1) {
    pool.allocate(1024, 4, allocOut);
  }
  assert.equal(pool.getStats().bufferReuseHits, 2000);
  pool.endFrame();
  pool.destroy();

  // 2. Stress 12,000 hot render-state packets across 3 frames (> 2,048/frame to verify MAX_FRAME_CMD_POOL reuse).
  const gpu = createRenderMockGpu({ recordPassOps: false });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, {
    width: 1280,
    height: 720,
    enableResourceReplay: false,
    requireNonBlackFrame: false,
  });
  const handler = installRenderStatePackets(processor);
  const setup = new PacketBuilder();
  appendCreateShaderModule(setup, 1, 0, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }');
  appendCreateShaderModule(setup, 2, 1, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }');
  appendCreateRenderPipeline2(setup, { id: 10, vsId: 1, fsId: 2, layoutId: 0, colorFormat: 22 });
  processor.executeCommandBuffer(setup.toUint8Array());

  const heavyFrame = new PacketBuilder();
  appendBeginRenderPass(heavyFrame, SWAPCHAIN_TEXTURE_ID, 0);
  appendSetPipeline(heavyFrame, 10);
  for (let i = 0; i < 2500; i += 1) {
    appendSetViewport(heavyFrame, 0, 0, 1280, 720, 0, 1);
    appendSetScissor(heavyFrame, 0, 0, 1280, 720);
  }
  heavyFrame.pushPacket(V2_OPCODES.END_RENDER_PASS, 0);
  heavyFrame.pushPacket(OPCODES.PRESENT, 0);
  const heavyBytes = heavyFrame.toUint8Array();

  for (let f = 0; f < 3; f += 1) {
    processor.executeCommandBuffer(heavyBytes);
  }
  assert.equal(handler.stats.errors, 0, handler.lastError);
  assert.equal(handler._vpCmdPool.length, 2500, 'pool keeps 2,500 slots without truncating at 2,048');
  assert.equal(handler._scCmdPool.length, 2500, 'pool keeps 2,500 slots without truncating at 2,048');
  handler.destroy();

  const elapsedMs = performance.now() - t0;
  assert.ok(elapsedMs < 5000, `hot-path stress took ${elapsedMs.toFixed(1)}ms (expected < 5000ms)`);

  // 3. Classic-worker importScripts VM check for both modified files.
  const ctx = { self: null, console, performance };
  ctx.self = ctx;
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(require.resolve('../web/workers/staging_pool.js'), 'utf8'), ctx);
  vm.runInContext(fs.readFileSync(require.resolve('../web/workers/render_state_packets.js'), 'utf8'), ctx);
  assert.equal(typeof ctx.PlayGta6StagingPool.GpuStagingPool, 'function');
  assert.equal(typeof ctx.PlayGta6RenderState.installRenderStatePackets, 'function');
  assert.equal(ctx.ChunkRingQueue, undefined, 'ChunkRingQueue must stay encapsulated inside IIFE');
});
