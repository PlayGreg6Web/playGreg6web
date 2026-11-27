import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import {
  SHADER_STAGE as WGPK_SHADER_STAGE,
  buildWgslPackBuffer,
  fnv1a64,
  formatHashHex,
} from '../web/workers/wgsl_pack_loader.js';

const require = createRequire(import.meta.url);
const wgpuMod = require('../web/workers/wgpu_worker.js');
const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  selectSupportedAdapterFeatures,
  OPTIONAL_ADAPTER_FEATURES,
  selectSupportedAdapterLimits,
  OPTIONAL_ADAPTER_LIMIT_SPECS,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  ADDRESS_MODE,
  FILTER_MODE,
  BIND_RESOURCE_TYPE,
  INDEX_FORMAT,
} = wgpuMod;

class BinaryCommandBuilder {
  constructor() {
    this.chunks = [];
  }

  _pushPacket(opcode, payloadBytes) {
    const buf = new ArrayBuffer(8 + payloadBytes.byteLength);
    const view = new DataView(buf);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, payloadBytes.byteLength, true);
    new Uint8Array(buf, 8).set(payloadBytes);
    this.chunks.push(new Uint8Array(buf));
  }

  nop() {
    this._pushPacket(OPCODES.NOP, new Uint8Array(0));
    return this;
  }

  beginFrame() {
    this._pushPacket(OPCODES.BEGIN_FRAME, new Uint8Array(0));
    return this;
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat32(0, r, true);
    view.setFloat32(4, g, true);
    view.setFloat32(8, b, true);
    view.setFloat32(12, a, true);
    this._pushPacket(OPCODES.CLEAR_COLOR, new Uint8Array(buf));
    return this;
  }

  createBuffer(id, usageFlags, byteSize) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, usageFlags, true);
    view.setBigUint64(8, BigInt(byteSize), true);
    this._pushPacket(OPCODES.CREATE_BUFFER, new Uint8Array(buf));
    return this;
  }

  writeBuffer(id, dstOffset, bytes) {
    const src = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const buf = new ArrayBuffer(16 + src.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, dstOffset, true);
    view.setUint32(8, src.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(src);
    this._pushPacket(OPCODES.WRITE_BUFFER, new Uint8Array(buf));
    return this;
  }

  createShaderModule(id, stage, wgslCode) {
    const encoded = new TextEncoder().encode(wgslCode);
    const buf = new ArrayBuffer(16 + encoded.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, encoded.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(encoded);
    this._pushPacket(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
    return this;
  }

  createRenderPipeline(id, vsShaderId, fsShaderId, vertexStride, topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, flags = 0) {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, vsShaderId, true);
    view.setUint32(8, fsShaderId, true);
    view.setUint32(12, vertexStride, true);
    view.setUint32(16, topology, true);
    view.setUint32(20, flags, true);
    this._pushPacket(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._pushPacket(OPCODES.SET_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setVertexBuffer(slot, bufferId, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, slot, true);
    view.setUint32(4, bufferId, true);
    view.setBigUint64(8, BigInt(offset), true);
    this._pushPacket(OPCODES.SET_VERTEX_BUFFER, new Uint8Array(buf));
    return this;
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, vertexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstVertex, true);
    view.setUint32(12, firstInstance, true);
    this._pushPacket(OPCODES.DRAW, new Uint8Array(buf));
    return this;
  }

  present() {
    this._pushPacket(OPCODES.PRESENT, new Uint8Array(0));
    return this;
  }

  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    this._pushPacket(OPCODES.DESTROY_RESOURCE, new Uint8Array(buf));
    return this;
  }

  createTexture(
    id,
    width,
    height,
    depthOrLayers = 1,
    mipLevels = 1,
    format = TEXTURE_FORMAT.RGBA8_UNORM,
    usageFlags = TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    dimension = TEXTURE_DIMENSION.DIM_2D
  ) {
    const buf = new ArrayBuffer(32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, width, true);
    view.setUint32(8, height, true);
    view.setUint32(12, depthOrLayers, true);
    view.setUint32(16, mipLevels, true);
    view.setUint32(20, format, true);
    view.setUint32(24, usageFlags, true);
    view.setUint32(28, dimension, true);
    this._pushPacket(OPCODES.CREATE_TEXTURE, new Uint8Array(buf));
    return this;
  }

  writeTexture(
    textureId,
    mipLevel,
    originX,
    originY,
    originZ,
    width,
    height,
    depthOrLayers,
    bytesPerRow,
    rowsPerImage,
    bytes
  ) {
    const src = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const buf = new ArrayBuffer(48 + src.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, textureId, true);
    view.setUint32(4, mipLevel, true);
    view.setUint32(8, originX, true);
    view.setUint32(12, originY, true);
    view.setUint32(16, originZ, true);
    view.setUint32(20, width, true);
    view.setUint32(24, height, true);
    view.setUint32(28, depthOrLayers, true);
    view.setUint32(32, bytesPerRow, true);
    view.setUint32(36, rowsPerImage, true);
    view.setUint32(40, src.byteLength, true);
    view.setUint32(44, 0, true);
    new Uint8Array(buf, 48).set(src);
    this._pushPacket(OPCODES.WRITE_TEXTURE, new Uint8Array(buf));
    return this;
  }

  createSampler(
    id,
    addressU = ADDRESS_MODE.CLAMP_TO_EDGE,
    addressV = ADDRESS_MODE.CLAMP_TO_EDGE,
    addressW = ADDRESS_MODE.CLAMP_TO_EDGE,
    magFilter = FILTER_MODE.LINEAR,
    minFilter = FILTER_MODE.LINEAR,
    mipmapFilter = FILTER_MODE.LINEAR,
    maxAnisotropy = 1
  ) {
    const buf = new ArrayBuffer(32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, addressU, true);
    view.setUint32(8, addressV, true);
    view.setUint32(12, addressW, true);
    view.setUint32(16, magFilter, true);
    view.setUint32(20, minFilter, true);
    view.setUint32(24, mipmapFilter, true);
    view.setUint32(28, maxAnisotropy, true);
    this._pushPacket(OPCODES.CREATE_SAMPLER, new Uint8Array(buf));
    return this;
  }

  createBindGroup(id, pipelineId, groupIndex, entries = []) {
    const buf = new ArrayBuffer(16 + entries.length * 32);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, pipelineId, true);
    view.setUint32(8, groupIndex, true);
    view.setUint32(12, entries.length, true);
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      const off = 16 + i * 32;
      view.setUint32(off, e.binding, true);
      view.setUint32(off + 4, e.resourceType, true);
      view.setUint32(off + 8, e.resourceId, true);
      view.setUint32(off + 12, 0, true);
      view.setBigUint64(off + 16, BigInt(e.offset || 0), true);
      view.setBigUint64(off + 24, BigInt(e.size || 0), true);
    }
    this._pushPacket(OPCODES.CREATE_BIND_GROUP, new Uint8Array(buf));
    return this;
  }

  setBindGroup(groupIndex, bindGroupId) {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, groupIndex, true);
    view.setUint32(4, bindGroupId, true);
    this._pushPacket(OPCODES.SET_BIND_GROUP, new Uint8Array(buf));
    return this;
  }

  setIndexBuffer(bufferId, indexFormat = INDEX_FORMAT.UINT16, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, bufferId, true);
    view.setUint32(4, indexFormat, true);
    view.setBigUint64(8, BigInt(offset), true);
    this._pushPacket(OPCODES.SET_INDEX_BUFFER, new Uint8Array(buf));
    return this;
  }

  drawIndexed(indexCount, instanceCount = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(20);
    const view = new DataView(buf);
    view.setUint32(0, indexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstIndex, true);
    view.setInt32(12, baseVertex, true);
    view.setUint32(16, firstInstance, true);
    this._pushPacket(OPCODES.DRAW_INDEXED, new Uint8Array(buf));
    return this;
  }

  createComputePipeline(id, csShaderId, flags = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, csShaderId, true);
    view.setUint32(8, flags, true);
    view.setUint32(12, 0, true);
    this._pushPacket(OPCODES.CREATE_COMPUTE_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setComputePipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._pushPacket(OPCODES.SET_COMPUTE_PIPELINE, new Uint8Array(buf));
    return this;
  }

  dispatchCompute(workgroupX, workgroupY = 1, workgroupZ = 1) {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, workgroupX, true);
    view.setUint32(4, workgroupY, true);
    view.setUint32(8, workgroupZ, true);
    this._pushPacket(OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf));
    return this;
  }

  finish() {
    const total = this.chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return out.buffer;
  }
}

function createMockWebGpu() {
  const calls = [];
  let nextHandle = 1;
  const mockTextureView = { kind: 'GPUTextureView', id: 'swapchain-view-0' };

  const ctx = {
    getCurrentTexture() {
      return {
        createView() {
          return mockTextureView;
        },
      };
    },
  };

  const device = {
    createBuffer(desc) {
      const buffer = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
          calls.push({ op: 'buffer.destroy', buffer });
        },
      };
      calls.push({ op: 'createBuffer', desc: { ...desc }, result: buffer });
      return buffer;
    },

    createShaderModule(desc) {
      const module = {
        kind: 'GPUShaderModule',
        handle: nextHandle++,
        code: desc.code,
      };
      calls.push({ op: 'createShaderModule', desc: { ...desc }, result: module });
      return module;
    },

    createRenderPipeline(desc) {
      const pipeline = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index) {
          const layout = { kind: 'GPUBindGroupLayout', pipelineHandle: this.handle, index };
          calls.push({ op: 'pipeline.getBindGroupLayout', pipeline: this, index, result: layout });
          return layout;
        },
      };
      calls.push({ op: 'createRenderPipeline', desc, result: pipeline });
      return pipeline;
    },

    createComputePipeline(desc) {
      const pipeline = {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index) {
          const layout = { kind: 'GPUBindGroupLayout', pipelineHandle: this.handle, index };
          calls.push({ op: 'computePipeline.getBindGroupLayout', pipeline: this, index, result: layout });
          return layout;
        },
      };
      calls.push({ op: 'createComputePipeline', desc, result: pipeline });
      return pipeline;
    },

    createTexture(desc) {
      const textureHandle = nextHandle++;
      const texture = {
        kind: 'GPUTexture',
        handle: textureHandle,
        desc: { ...desc, size: { ...desc.size } },
        destroyed: false,
        createView() {
          const view = { kind: 'GPUTextureView', textureHandle, handle: nextHandle++ };
          calls.push({ op: 'texture.createView', texture, result: view });
          return view;
        },
        destroy() {
          this.destroyed = true;
          calls.push({ op: 'texture.destroy', texture });
        },
      };
      calls.push({ op: 'createTexture', desc: { ...desc, size: { ...desc.size } }, result: texture });
      return texture;
    },

    createSampler(desc) {
      const sampler = {
        kind: 'GPUSampler',
        handle: nextHandle++,
        desc: { ...desc },
      };
      calls.push({ op: 'createSampler', desc: { ...desc }, result: sampler });
      return sampler;
    },

    createBindGroup(desc) {
      const bindGroup = {
        kind: 'GPUBindGroup',
        handle: nextHandle++,
        desc,
      };
      calls.push({ op: 'createBindGroup', desc, result: bindGroup });
      return bindGroup;
    },

    createCommandEncoder() {
      const commandBuffer = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass(desc) {
          calls.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline(pipeline) {
              calls.push({ op: 'setPipeline', pipeline });
            },
            setBindGroup(index, bindGroup) {
              calls.push({ op: 'renderPass.setBindGroup', index, bindGroup });
            },
            setVertexBuffer(slot, buffer, offset) {
              calls.push({ op: 'setVertexBuffer', slot, buffer, offset });
            },
            setIndexBuffer(buffer, indexFormat, offset) {
              calls.push({ op: 'setIndexBuffer', buffer, indexFormat, offset });
            },
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              calls.push({
                op: 'draw',
                vertexCount,
                instanceCount,
                firstVertex,
                firstInstance,
              });
            },
            drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
              calls.push({
                op: 'drawIndexed',
                indexCount,
                instanceCount,
                firstIndex,
                baseVertex,
                firstInstance,
              });
            },
            end() {
              calls.push({ op: 'renderPass.end' });
            },
          };
        },
        beginComputePass() {
          calls.push({ op: 'beginComputePass' });
          return {
            setPipeline(pipeline) {
              calls.push({ op: 'computePass.setPipeline', pipeline });
            },
            setBindGroup(index, bindGroup) {
              calls.push({ op: 'computePass.setBindGroup', index, bindGroup });
            },
            dispatchWorkgroups(x, y, z) {
              calls.push({ op: 'dispatchWorkgroups', x, y, z });
            },
            end() {
              calls.push({ op: 'computePass.end' });
            },
          };
        },
        finish() {
          calls.push({ op: 'encoder.finish', result: commandBuffer });
          return commandBuffer;
        },
      };
    },

    queue: {
      writeBuffer(buffer, dstOffset, data) {
        calls.push({
          op: 'writeBuffer',
          buffer,
          dstOffset,
          data: new Uint8Array(data),
        });
      },
      writeTexture(destination, data, dataLayout, size) {
        calls.push({
          op: 'writeTexture',
          destination: { ...destination, origin: { ...destination.origin } },
          data: new Uint8Array(data),
          dataLayout: { ...dataLayout },
          size: { ...size },
        });
      },
      submit(commandBuffers) {
        calls.push({ op: 'submit', commandBuffers });
      },
    },
  };

  return { device, ctx, calls, mockTextureView };
}

test('UMD export attaches PlayGta6Wgpu on globalThis', () => {
  assert.equal(globalThis.PlayGta6Wgpu.WgpuCommandProcessor, WgpuCommandProcessor);
  assert.equal(globalThis.PlayGta6Wgpu.OPCODES.CREATE_RENDER_PIPELINE, 8);
  assert.equal(globalThis.PlayGta6Wgpu.OPCODES.DESTROY_RESOURCE, 11);
});

test('full binary command stream drives every WebGPU device, render-pass, and destroy call', () => {
  const { device, ctx, calls, mockTextureView } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });

  const vertexFloats = new Float32Array([
    0.0, 0.5, 0.0,
    -0.5, -0.5, 0.0,
    0.5, -0.5, 0.0,
  ]);
  const vertexBytes = new Uint8Array(vertexFloats.buffer);
  const vsWgsl = '@vertex fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(pos, 1.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.5, 0.0, 1.0); }';

  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, vertexBytes.byteLength)
    .writeBuffer(1, 0, vertexBytes)
    .createShaderModule(2, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, fsWgsl)
    .createRenderPipeline(4, 2, 3, 12, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, 0)
    .clearColor(0.25, 0.5, 0.75, 1.0)
    .setPipeline(4)
    .setVertexBuffer(0, 1, 0)
    .draw(3, 1, 0, 0)
    .present()
    .destroyResource(1)
    .finish();

  const result = processor.execute(stream);
  assert.deepEqual(result, { frameCount: 1, presents: 1 });
  assert.equal(processor.frameCount, 1);

  assert.deepEqual(
    calls.map((c) => c.op),
    [
      'createBuffer',
      'writeBuffer',
      'createShaderModule',
      'createShaderModule',
      'createRenderPipeline',
      'beginRenderPass',
      'setPipeline',
      'setVertexBuffer',
      'draw',
      'renderPass.end',
      'encoder.finish',
      'submit',
      'buffer.destroy',
    ],
  );

  // 1. createBuffer
  const createBufCall = calls[0];
  assert.deepEqual(createBufCall.desc, {
    size: vertexBytes.byteLength,
    usage: BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST,
  });
  const gpuBuffer = createBufCall.result;

  // 2. writeBuffer
  const writeBufCall = calls[1];
  assert.equal(writeBufCall.buffer, gpuBuffer);
  assert.equal(writeBufCall.dstOffset, 0);
  assert.deepEqual(writeBufCall.data, vertexBytes);

  // 3 & 4. createShaderModule (VS & FS)
  const vsCall = calls[2];
  const fsCall = calls[3];
  assert.equal(vsCall.desc.code, vsWgsl);
  assert.equal(fsCall.desc.code, fsWgsl);

  // 5. createRenderPipeline
  const pipelineCall = calls[4];
  assert.equal(pipelineCall.desc.layout, 'auto');
  assert.equal(pipelineCall.desc.vertex.module, vsCall.result);
  assert.equal(pipelineCall.desc.vertex.entryPoint, 'vs_main');
  assert.deepEqual(pipelineCall.desc.vertex.buffers, [
    {
      arrayStride: 12,
      attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
    },
  ]);
  assert.equal(pipelineCall.desc.fragment.module, fsCall.result);
  assert.equal(pipelineCall.desc.fragment.entryPoint, 'fs_main');
  assert.deepEqual(pipelineCall.desc.fragment.targets, [{ format: 'bgra8unorm' }]);
  assert.deepEqual(pipelineCall.desc.primitive, { topology: 'triangle-list' });
  const gpuPipeline = pipelineCall.result;

  // 6. beginRenderPass
  const passCall = calls[5];
  assert.deepEqual(passCall.desc, {
    colorAttachments: [
      {
        view: mockTextureView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.25, g: 0.5, b: 0.75, a: 1.0 },
      },
    ],
  });

  // 7. setPipeline
  assert.equal(calls[6].pipeline, gpuPipeline);

  // 8. setVertexBuffer
  assert.equal(calls[7].slot, 0);
  assert.equal(calls[7].buffer, gpuBuffer);
  assert.equal(calls[7].offset, 0);

  // 9. draw
  assert.deepEqual(calls[8], {
    op: 'draw',
    vertexCount: 3,
    instanceCount: 1,
    firstVertex: 0,
    firstInstance: 0,
  });

  // 11. submit
  assert.deepEqual(calls[11].commandBuffers, [calls[10].result]);

  // 12. destroyResource after present
  assert.equal(calls[12].buffer, gpuBuffer);
  assert.equal(gpuBuffer.destroyed, true);
  assert.equal(processor.buffers.has(1), false);
  assert.equal(processor.shaderModules.has(2), true);
  assert.equal(processor.pipelines.has(4), true);
});

test('DestroyResource safely handles shaders, pipelines, missing IDs, and buffers without destroy()', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx);

  // Inject a buffer object without a .destroy method to confirm safe cleanup.
  processor.buffers.set(99, { kind: 'GPUBufferNoDestroy' });

  const stream = new BinaryCommandBuilder()
    .nop()
    .createShaderModule(10, SHADER_STAGE.VERTEX, 'fn vs_main() {}')
    .createRenderPipeline(11, 10, 0, 0, PRIMITIVE_TOPOLOGY.LINE_STRIP, 0)
    .destroyResource(99)
    .destroyResource(10)
    .destroyResource(11)
    .destroyResource(9999)
    .finish();

  const res = processor.execute(stream);
  assert.deepEqual(res, { frameCount: 0, presents: 0 });
  assert.equal(processor.buffers.size, 0);
  assert.equal(processor.shaderModules.size, 0);
  assert.equal(processor.pipelines.size, 0);
  assert.equal(calls[1].desc.primitive.topology, 'line-strip');
  assert.deepEqual(calls[1].desc.vertex.buffers, []);
  assert.equal(calls[1].desc.fragment, undefined);
});

test('G5 opcodes drive textures, samplers, bind groups, indexed draw, and compute dispatch', () => {
  const { device, ctx, calls, mockTextureView } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });

  assert.equal(OPCODES.CREATE_TEXTURE, 12);
  assert.equal(OPCODES.WRITE_TEXTURE, 13);
  assert.equal(OPCODES.CREATE_SAMPLER, 14);
  assert.equal(OPCODES.CREATE_BIND_GROUP, 15);
  assert.equal(OPCODES.SET_BIND_GROUP, 16);
  assert.equal(OPCODES.SET_INDEX_BUFFER, 17);
  assert.equal(OPCODES.DRAW_INDEXED, 18);
  assert.equal(OPCODES.CREATE_COMPUTE_PIPELINE, 19);
  assert.equal(OPCODES.SET_COMPUTE_PIPELINE, 20);
  assert.equal(OPCODES.DISPATCH_COMPUTE, 21);

  const texels = new Uint8Array([
    255, 0, 0, 255,   0, 255, 0, 255,
    0, 0, 255, 255,   255, 255, 255, 255,
  ]);
  const indices = new Uint16Array([0, 1, 2, 2, 3, 0]);
  const indexBytes = new Uint8Array(indices.buffer);

  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.STORAGE | BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 256)
    .createBuffer(2, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, 64)
    .createBuffer(3, BUFFER_USAGE.INDEX | BUFFER_USAGE.COPY_DST, indexBytes.byteLength)
    .writeBuffer(3, 0, indexBytes)
    .createTexture(
      4,
      2,
      2,
      1,
      1,
      TEXTURE_FORMAT.RGBA8_UNORM,
      TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
      TEXTURE_DIMENSION.DIM_2D
    )
    .writeTexture(4, 0, 0, 0, 0, 2, 2, 1, 8, 2, texels)
    .createSampler(
      5,
      ADDRESS_MODE.REPEAT,
      ADDRESS_MODE.MIRROR_REPEAT,
      ADDRESS_MODE.CLAMP_TO_EDGE,
      FILTER_MODE.LINEAR,
      FILTER_MODE.LINEAR,
      FILTER_MODE.NEAREST,
      8
    )
    .createShaderModule(6, SHADER_STAGE.COMPUTE, '@compute @workgroup_size(8) fn cs_main() {}')
    .createShaderModule(7, SHADER_STAGE.VERTEX, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }')
    .createShaderModule(8, SHADER_STAGE.FRAGMENT, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }')
    .createComputePipeline(9, 6, 0)
    .createRenderPipeline(10, 7, 8, 16, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST, 0)
    .createBindGroup(11, 9, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 64, size: 128 },
    ])
    .createBindGroup(12, 10, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 0 },
      { binding: 1, resourceType: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 4 },
      { binding: 2, resourceType: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 5 },
    ])
    .clearColor(0.25, 0.5, 0.75, 1.0)
    .setComputePipeline(9)
    .setBindGroup(0, 11)
    .dispatchCompute(4, 2, 1)
    .setPipeline(10)
    .setBindGroup(0, 12)
    .setVertexBuffer(0, 2, 0)
    .setIndexBuffer(3, INDEX_FORMAT.UINT16, 0)
    .drawIndexed(6, 2, 0, -4, 1)
    .present()
    .finish();

  const res = processor.execute(stream);
  assert.deepEqual(res, { frameCount: 1, presents: 1 });

  assert.deepEqual(
    calls.map((c) => c.op),
    [
      'createBuffer',
      'createBuffer',
      'createBuffer',
      'writeBuffer',
      'createTexture',
      'texture.createView',
      'writeTexture',
      'createSampler',
      'createShaderModule',
      'createShaderModule',
      'createShaderModule',
      'createComputePipeline',
      'createRenderPipeline',
      'computePipeline.getBindGroupLayout',
      'createBindGroup',
      'pipeline.getBindGroupLayout',
      'createBindGroup',
      'beginComputePass',
      'computePass.setPipeline',
      'computePass.setBindGroup',
      'dispatchWorkgroups',
      'computePass.end',
      'beginRenderPass',
      'setPipeline',
      'renderPass.setBindGroup',
      'setVertexBuffer',
      'setIndexBuffer',
      'drawIndexed',
      'renderPass.end',
      'encoder.finish',
      'submit',
    ]
  );

  // Verify createTexture & writeTexture
  const createTexCall = calls.find((c) => c.op === 'createTexture');
  assert.deepEqual(createTexCall.desc, {
    size: { width: 2, height: 2, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    format: 'rgba8unorm',
    usage: TEXTURE_USAGE.COPY_SRC | TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    dimension: '2d',
  });
  const writeTexCall = calls.find((c) => c.op === 'writeTexture');
  assert.equal(writeTexCall.destination.texture, createTexCall.result);
  assert.equal(writeTexCall.destination.mipLevel, 0);
  assert.deepEqual(writeTexCall.destination.origin, { x: 0, y: 0, z: 0 });
  assert.deepEqual(writeTexCall.data, texels);
  assert.deepEqual(writeTexCall.dataLayout, { offset: 0, bytesPerRow: 8, rowsPerImage: 2 });
  assert.deepEqual(writeTexCall.size, { width: 2, height: 2, depthOrArrayLayers: 1 });

  // Verify createSampler
  const samplerCall = calls.find((c) => c.op === 'createSampler');
  assert.deepEqual(samplerCall.desc, {
    addressModeU: 'repeat',
    addressModeV: 'mirror-repeat',
    addressModeW: 'clamp-to-edge',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'nearest',
    maxAnisotropy: 8,
  });

  // Verify createBindGroup entries (size > 0 included; size === 0 omitted)
  const bgCalls = calls.filter((c) => c.op === 'createBindGroup');
  assert.equal(bgCalls.length, 2);
  assert.deepEqual(bgCalls[0].desc.entries, [
    { binding: 0, resource: { buffer: processor.buffers.get(1), offset: 64, size: 128 } },
  ]);
  assert.deepEqual(bgCalls[1].desc.entries, [
    { binding: 0, resource: { buffer: processor.buffers.get(1), offset: 0 } },
    { binding: 1, resource: processor.textureViews.get(4) },
    { binding: 2, resource: processor.samplers.get(5) },
  ]);

  // Verify compute dispatch & indexed draw (including negative baseVertex)
  const dispatchCall = calls.find((c) => c.op === 'dispatchWorkgroups');
  assert.deepEqual(dispatchCall, { op: 'dispatchWorkgroups', x: 4, y: 2, z: 1 });

  const renderPassCall = calls.find((c) => c.op === 'beginRenderPass');
  assert.deepEqual(renderPassCall.desc, {
    colorAttachments: [
      {
        view: mockTextureView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0.25, g: 0.5, b: 0.75, a: 1.0 },
      },
    ],
  });

  const setIbCall = calls.find((c) => c.op === 'setIndexBuffer');
  assert.equal(setIbCall.buffer, processor.buffers.get(3));
  assert.equal(setIbCall.indexFormat, 'uint16');
  assert.equal(setIbCall.offset, 0);

  const drawIdxCall = calls.find((c) => c.op === 'drawIndexed');
  assert.deepEqual(drawIdxCall, {
    op: 'drawIndexed',
    indexCount: 6,
    instanceCount: 2,
    firstIndex: 0,
    baseVertex: -4,
    firstInstance: 1,
  });
});

test('Pass switching automatically ends render->compute->render passes and preserves prior render contents', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });

  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, 'fn vs_main() {}')
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, 'fn fs_main() {}')
    .createShaderModule(3, SHADER_STAGE.COMPUTE, 'fn cs_main() {}')
    .createRenderPipeline(4, 1, 2, 0)
    .createComputePipeline(5, 3)
    .createBuffer(6, BUFFER_USAGE.INDEX, 16)
    .clearColor(0.1, 0.2, 0.3, 1.0)
    // Pass 1: Render (clear)
    .setPipeline(4)
    .draw(3, 1, 0, 0)
    // Pass 2: Compute (auto-ends render pass 1)
    .setComputePipeline(5)
    .dispatchCompute(2, 1, 1)
    // Pass 3: Render resumed (auto-ends compute pass, uses loadOp: 'load')
    .setPipeline(4)
    .setIndexBuffer(6, INDEX_FORMAT.UINT32, 4)
    .drawIndexed(3, 1, 0, 0, 0)
    .present()
    // Frame 2: Compute-only before Present (auto-ends compute pass, still clears/presents swapchain)
    .beginFrame()
    .clearColor(0.875, 0.75, 0.625, 1.0)
    .setComputePipeline(5)
    .dispatchCompute(1, 1, 1)
    .present()
    .finish();

  const res = processor.execute(stream);
  assert.deepEqual(res, { frameCount: 2, presents: 2 });

  const passOps = calls
    .filter((c) =>
      [
        'beginRenderPass',
        'renderPass.end',
        'beginComputePass',
        'computePass.end',
        'draw',
        'dispatchWorkgroups',
        'setIndexBuffer',
        'drawIndexed',
        'submit',
      ].includes(c.op)
    )
    .map((c) => c.op);

  assert.deepEqual(passOps, [
    // Frame 1
    'beginRenderPass',
    'draw',
    'renderPass.end',
    'beginComputePass',
    'dispatchWorkgroups',
    'computePass.end',
    'beginRenderPass',
    'setIndexBuffer',
    'drawIndexed',
    'renderPass.end',
    'submit',
    // Frame 2
    'beginComputePass',
    'dispatchWorkgroups',
    'computePass.end',
    'beginRenderPass',
    'renderPass.end',
    'submit',
  ]);

  const renderPasses = calls.filter((c) => c.op === 'beginRenderPass');
  assert.equal(renderPasses.length, 3);
  assert.equal(renderPasses[0].desc.colorAttachments[0].loadOp, 'clear');
  assert.equal(renderPasses[1].desc.colorAttachments[0].loadOp, 'load');
  assert.equal(renderPasses[1].desc.colorAttachments[0].clearValue, undefined);
  assert.equal(renderPasses[2].desc.colorAttachments[0].loadOp, 'clear');
  assert.deepEqual(renderPasses[2].desc.colorAttachments[0].clearValue, {
    r: 0.875,
    g: 0.75,
    b: 0.625,
    a: 1.0,
  });

  const setIbCall = calls.find((c) => c.op === 'setIndexBuffer');
  assert.equal(setIbCall.indexFormat, 'uint32');
  assert.equal(setIbCall.offset, 4);
});

test('DestroyResource cleans up textures, textureViews, samplers, bindGroups, computePipelines, and replaced textures', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx);

  const stream = new BinaryCommandBuilder()
    .createTexture(100, 4, 4, 1, 1, TEXTURE_FORMAT.BC7_RGBA_UNORM, TEXTURE_USAGE.TEXTURE_BINDING)
    // Recreating with same ID must destroy previous texture first.
    .createTexture(100, 8, 8, 1, 2, TEXTURE_FORMAT.RGBA16_FLOAT, TEXTURE_USAGE.TEXTURE_BINDING)
    .createSampler(101)
    .createShaderModule(102, SHADER_STAGE.COMPUTE, 'fn cs_main() {}')
    .createComputePipeline(103, 102)
    .createBindGroup(104, 103, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.TEXTURE_VIEW, resourceId: 100 },
      { binding: 1, resourceType: BIND_RESOURCE_TYPE.SAMPLER, resourceId: 101 },
    ])
    .destroyResource(100)
    .destroyResource(101)
    .destroyResource(102)
    .destroyResource(103)
    .destroyResource(104)
    .finish();

  processor.execute(stream);

  const createdTextures = calls.filter((c) => c.op === 'createTexture').map((c) => c.result);
  assert.equal(createdTextures.length, 2);
  assert.equal(createdTextures[0].desc.format, 'bc7-rgba-unorm');
  assert.equal(createdTextures[1].desc.format, 'rgba16float');
  assert.equal(createdTextures[0].destroyed, true);
  assert.equal(createdTextures[1].destroyed, true);

  assert.equal(processor.textures.size, 0);
  assert.equal(processor.textureViews.size, 0);
  assert.equal(processor.samplers.size, 0);
  assert.equal(processor.bindGroups.size, 0);
  assert.equal(processor.shaderModules.size, 0);
  assert.equal(processor.computePipelines.size, 0);
});

test('W92 selectSupportedAdapterFeatures picks only supported optional features and handles missing adapter.features', () => {
  assert.equal(Object.isFrozen(OPTIONAL_ADAPTER_FEATURES), true);
  assert.deepEqual(OPTIONAL_ADAPTER_FEATURES, [
    'texture-compression-bc',
    'depth32float-stencil8',
    'float32-filterable',
    'bgra8unorm-storage',
    'rg11b10ufloat-renderable',
    'texture-formats-tier1',
    'float32-blendable',
    'dual-source-blending',
    'depth-clip-control',
  ]);
  assert.equal(globalThis.PlayGta6Wgpu.OPTIONAL_ADAPTER_FEATURES, OPTIONAL_ADAPTER_FEATURES);
  assert.equal(
    globalThis.PlayGta6Wgpu.selectSupportedAdapterFeatures,
    selectSupportedAdapterFeatures
  );

  assert.deepEqual(selectSupportedAdapterFeatures(null), []);
  assert.deepEqual(selectSupportedAdapterFeatures(undefined), []);
  assert.deepEqual(selectSupportedAdapterFeatures({}), []);
  assert.deepEqual(selectSupportedAdapterFeatures({ features: null }), []);
  assert.deepEqual(selectSupportedAdapterFeatures({ features: {} }), []);
  assert.deepEqual(
    selectSupportedAdapterFeatures({
      features: {
        has() {
          throw new Error('probe error');
        },
      },
    }),
    []
  );

  const partialAdapter = {
    features: new Set([
      'texture-compression-bc',
      'depth32float-stencil8',
      'rg11b10ufloat-renderable',
      'indirect-first-instance',
      'timestamp-query',
    ]),
  };
  assert.deepEqual(selectSupportedAdapterFeatures(partialAdapter), [
    'texture-compression-bc',
    'depth32float-stencil8',
    'rg11b10ufloat-renderable',
  ]);

  const fullAdapter = {
    features: new Set(OPTIONAL_ADAPTER_FEATURES),
  };
  assert.deepEqual(selectSupportedAdapterFeatures(fullAdapter), [...OPTIONAL_ADAPTER_FEATURES]);
});

async function withMockNavigatorGpu(gpu, fn) {
  const prevDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    value: { gpu },
    configurable: true,
    writable: true,
  });
  try {
    return await fn();
  } finally {
    if (prevDescriptor) {
      Object.defineProperty(globalThis, 'navigator', prevDescriptor);
    } else {
      delete globalThis.navigator;
    }
  }
}

function createMockOffscreenCanvas(ctx) {
  const configured = [];
  const webgpuCtx = {
    ...ctx,
    configure(desc) {
      configured.push(desc);
    },
  };
  return {
    canvas: {
      width: 0,
      height: 0,
      getContext(kind) {
        assert.equal(kind, 'webgpu');
        return webgpuCtx;
      },
    },
    configured,
  };
}

test('W92 defaultInitGpu requests supported optional adapter features and falls back cleanly if requestDevice rejects', async () => {
  // 1. Adapter supports optional features -> passes requiredFeatures to requestDevice.
  {
    const { device, ctx } = createMockWebGpu();
    const supported = [
      'texture-compression-bc',
      'depth32float-stencil8',
      'float32-filterable',
    ];
    device.features = new Set(supported);
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'mock-gpu', architecture: 'rdna2' },
          features: new Set([...supported, 'timestamp-query']),
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 640, height: 360 });
    });

    assert.equal(requestDeviceCalls.length, 1);
    assert.deepEqual(requestDeviceCalls[0], { requiredFeatures: supported });
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg, 'expected ready postMessage');
    assert.equal(readyMsg.adapter, 'mock-gpu rdna2');
    assert.deepEqual(readyMsg.features, supported);
  }

  // 2. requestDevice({ requiredFeatures }) rejects -> falls back to requestDevice() with no args.
  {
    const { device, ctx } = createMockWebGpu();
    device.features = new Set();
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'fallback-gpu', architecture: '' },
          features: new Set(['texture-compression-bc', 'texture-formats-tier1']),
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            if (desc && desc.requiredFeatures) {
              throw new Error('requested feature rejected by driver');
            }
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 640, height: 360 });
    });

    assert.equal(requestDeviceCalls.length, 2);
    assert.deepEqual(requestDeviceCalls[0], {
      requiredFeatures: ['texture-compression-bc', 'texture-formats-tier1'],
    });
    assert.equal(requestDeviceCalls[1], undefined);
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg, 'expected ready postMessage after fallback');
    assert.equal(readyMsg.adapter, 'fallback-gpu');
    assert.deepEqual(readyMsg.features, []);
  }

  // 3. Adapter without features -> calls requestDevice() once with no arguments.
  {
    const { device, ctx } = createMockWebGpu();
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 320, height: 180 });
    });

    assert.equal(requestDeviceCalls.length, 1);
    assert.equal(requestDeviceCalls[0], undefined);
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg);
    assert.deepEqual(readyMsg.features, []);
  }
});

test('W92 buildGpuTelemetrySnapshot exposes realPackDrawCount and fallbackDrawCount in frameStats', () => {
  const { device, ctx } = createMockWebGpu();
  const vsWgsl =
    '@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }';
  const fsWgsl =
    '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.7, 0.4, 1.0); }';
  const fallbackFsWgsl =
    '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.9, 0.1, 0.5, 1.0); }';

  const vsHash = formatHashHex(fnv1a64(vsWgsl));
  const fsHash = formatHashHex(fnv1a64(fsWgsl));
  const fallbackFsHash = '0x000000000000f001';
  const missingFsHash = '0xdeadbeef00009201';

  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: vsWgsl },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fsWgsl },
    { hash: fallbackFsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fallbackFsWgsl },
  ]);

  const processor = new WgpuCommandProcessor(device, ctx, {
    format: 'bgra8unorm',
    wgslPackBuffer: packBuf,
    fallbackShaderHashes: { fragment: fallbackFsHash },
  });

  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createShaderModule(1, SHADER_STAGE.VERTEX, vsHash)
    .createShaderModule(2, SHADER_STAGE.FRAGMENT, fsHash)
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, missingFsHash)
    .createRenderPipeline(10, 1, 2, 0)
    .createRenderPipeline(11, 1, 3, 0)
    .setPipeline(10)
    .draw(3, 1, 0, 0)
    .draw(3, 1, 0, 0)
    .setPipeline(11)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  processor.execute(stream);

  const snapshot = buildGpuTelemetrySnapshot(processor, { timestamp: 12345 });
  assert.equal(snapshot.frameStats.framesPresented, 1);
  assert.equal(snapshot.frameStats.drawCalls, 3);
  assert.equal(snapshot.frameStats.realPackDrawCount, 2);
  assert.equal(snapshot.frameStats.fallbackDrawCount, 1);

  // Fallback to direct processor properties and non-negative integer coercion.
  const directPropSnap = buildGpuTelemetrySnapshot(
    {
      getStats: () => ({ frameCount: 2, drawCalls: 5 }),
      realPackDrawCount: 4,
      fallbackDrawCount: 1,
    },
    { timestamp: 200 }
  );
  assert.equal(directPropSnap.frameStats.realPackDrawCount, 4);
  assert.equal(directPropSnap.frameStats.fallbackDrawCount, 1);

  const coercedSnap = buildGpuTelemetrySnapshot(
    {
      getStats: () => ({
        frameCount: 1,
        realPackDrawCount: -5,
        fallbackDrawCount: Number.NaN,
      }),
    },
    { timestamp: 300 }
  );
  assert.equal(coercedSnap.frameStats.realPackDrawCount, 0);
  assert.equal(coercedSnap.frameStats.fallbackDrawCount, 0);
});

test('W96 selectSupportedAdapterLimits picks elevated limits clamped to caps and ignores core defaults or missing limits', () => {
  assert.equal(Object.isFrozen(OPTIONAL_ADAPTER_LIMIT_SPECS), true);
  assert.equal(globalThis.PlayGta6Wgpu.OPTIONAL_ADAPTER_LIMIT_SPECS, OPTIONAL_ADAPTER_LIMIT_SPECS);
  assert.equal(
    globalThis.PlayGta6Wgpu.selectSupportedAdapterLimits,
    selectSupportedAdapterLimits
  );

  // Missing / invalid / throwing adapter.limits all yield an empty object.
  assert.deepEqual(selectSupportedAdapterLimits(null), {});
  assert.deepEqual(selectSupportedAdapterLimits(undefined), {});
  assert.deepEqual(selectSupportedAdapterLimits({}), {});
  assert.deepEqual(selectSupportedAdapterLimits({ limits: null }), {});
  assert.deepEqual(selectSupportedAdapterLimits({ limits: {} }), {});
  assert.deepEqual(
    selectSupportedAdapterLimits({
      get limits() {
        throw new Error('limits getter error');
      },
    }),
    {}
  );
  assert.deepEqual(
    selectSupportedAdapterLimits({
      limits: {
        get maxSampledTexturesPerShaderStage() {
          throw new Error('property getter error');
        },
        maxSamplersPerShaderStage: Number.NaN,
        maxStorageBuffersPerShaderStage: Number.POSITIVE_INFINITY,
        maxUniformBuffersPerShaderStage: Number.NEGATIVE_INFINITY,
        maxBufferSize: '1073741824',
      },
    }),
    {}
  );

  // Adapter at or below WebGPU core minimums -> returns {} (no keys requested).
  assert.deepEqual(
    selectSupportedAdapterLimits({
      limits: {
        maxSampledTexturesPerShaderStage: 16,
        maxSamplersPerShaderStage: 16,
        maxStorageBuffersPerShaderStage: 8,
        maxUniformBuffersPerShaderStage: 12,
        maxStorageTexturesPerShaderStage: 4,
        maxBindingsPerBindGroup: 1000,
        maxBufferSize: 256 * 1024 * 1024,
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxComputeWorkgroupStorageSize: 16384,
        maxTextureDimension2D: 16384,
      },
    }),
    {}
  );
  assert.deepEqual(
    selectSupportedAdapterLimits({
      limits: {
        maxSampledTexturesPerShaderStage: 8,
        maxBindingsPerBindGroup: 640,
        maxBufferSize: 128 * 1024 * 1024,
      },
    }),
    {}
  );

  // Partial adapter with values between core default and cap -> preserves exact supported values.
  assert.deepEqual(
    selectSupportedAdapterLimits({
      limits: {
        maxSampledTexturesPerShaderStage: 64,
        maxSamplersPerShaderStage: 16,
        maxStorageBuffersPerShaderStage: 16,
        maxUniformBuffersPerShaderStage: 24,
        maxStorageTexturesPerShaderStage: 8,
        maxBindingsPerBindGroup: 1000,
        maxBufferSize: 512 * 1024 * 1024,
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
        maxComputeWorkgroupStorageSize: 32768,
        maxColorAttachments: 8,
      },
    }),
    {
      maxSampledTexturesPerShaderStage: 64,
      maxStorageBuffersPerShaderStage: 16,
      maxUniformBuffersPerShaderStage: 24,
      maxStorageTexturesPerShaderStage: 8,
      maxBufferSize: 512 * 1024 * 1024,
      maxStorageBufferBindingSize: 256 * 1024 * 1024,
      maxComputeWorkgroupStorageSize: 32768,
    }
  );

  // High-end adapter exceeding all caps -> clamps every key to its configured cap.
  assert.deepEqual(
    selectSupportedAdapterLimits({
      limits: {
        maxSampledTexturesPerShaderStage: 1024,
        maxSamplersPerShaderStage: 512,
        maxStorageBuffersPerShaderStage: 128,
        maxUniformBuffersPerShaderStage: 128,
        maxStorageTexturesPerShaderStage: 64,
        maxBindingsPerBindGroup: 65536,
        maxBufferSize: 4 * 1024 * 1024 * 1024,
        maxStorageBufferBindingSize: 2 * 1024 * 1024 * 1024,
        maxComputeWorkgroupStorageSize: 131072,
      },
    }),
    {
      maxSampledTexturesPerShaderStage: 256,
      maxSamplersPerShaderStage: 64,
      maxStorageBuffersPerShaderStage: 32,
      maxUniformBuffersPerShaderStage: 32,
      maxStorageTexturesPerShaderStage: 16,
      maxBindingsPerBindGroup: 4096,
      maxBufferSize: 1024 * 1024 * 1024,
      maxStorageBufferBindingSize: 512 * 1024 * 1024,
      maxComputeWorkgroupStorageSize: 65536,
    }
  );
});

test('W96 defaultInitGpu passes requiredLimits when supported and falls back cleanly on rejection', async () => {
  // 1. Adapter supports both optional features and elevated limits -> single requestDevice({ requiredFeatures, requiredLimits }).
  {
    const { device, ctx } = createMockWebGpu();
    const supportedFeatures = ['texture-compression-bc', 'float32-filterable'];
    device.features = new Set(supportedFeatures);
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'mock-gpu', architecture: 'rdna3' },
          features: new Set(supportedFeatures),
          limits: {
            maxSampledTexturesPerShaderStage: 512,
            maxStorageBuffersPerShaderStage: 16,
            maxBufferSize: 2 * 1024 * 1024 * 1024,
            maxBindingsPerBindGroup: 1000,
          },
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 640, height: 360 });
    });

    assert.equal(requestDeviceCalls.length, 1);
    assert.deepEqual(requestDeviceCalls[0], {
      requiredFeatures: supportedFeatures,
      requiredLimits: {
        maxSampledTexturesPerShaderStage: 256,
        maxStorageBuffersPerShaderStage: 16,
        maxBufferSize: 1024 * 1024 * 1024,
      },
    });
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg, 'expected ready postMessage');
    assert.deepEqual(readyMsg.features, supportedFeatures);
  }

  // 2. Adapter supports elevated limits only (no optional features) -> passes only { requiredLimits }.
  {
    const { device, ctx } = createMockWebGpu();
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'limits-only-gpu', architecture: '' },
          limits: {
            maxUniformBuffersPerShaderStage: 64,
            maxComputeWorkgroupStorageSize: 32768,
          },
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 320, height: 180 });
    });

    assert.equal(requestDeviceCalls.length, 1);
    assert.deepEqual(requestDeviceCalls[0], {
      requiredLimits: {
        maxUniformBuffersPerShaderStage: 32,
        maxComputeWorkgroupStorageSize: 32768,
      },
    });
    assert.equal('requiredFeatures' in requestDeviceCalls[0], false);
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg);
    assert.deepEqual(readyMsg.features, []);
  }

  // 3. Adapter rejects requiredLimits but accepts requiredFeatures -> falls back to { requiredFeatures }.
  {
    const { device, ctx } = createMockWebGpu();
    const supportedFeatures = ['texture-compression-bc', 'depth32float-stencil8'];
    device.features = new Set(supportedFeatures);
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'strict-limits-gpu', architecture: '' },
          features: new Set(supportedFeatures),
          limits: {
            maxSampledTexturesPerShaderStage: 128,
            maxBufferSize: 1024 * 1024 * 1024,
          },
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            if (desc && desc.requiredLimits) {
              throw new Error('driver rejected elevated requiredLimits');
            }
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 640, height: 360 });
    });

    assert.equal(requestDeviceCalls.length, 2);
    assert.deepEqual(requestDeviceCalls[0], {
      requiredFeatures: supportedFeatures,
      requiredLimits: {
        maxSampledTexturesPerShaderStage: 128,
        maxBufferSize: 1024 * 1024 * 1024,
      },
    });
    assert.deepEqual(requestDeviceCalls[1], {
      requiredFeatures: supportedFeatures,
    });
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg);
    assert.deepEqual(readyMsg.features, supportedFeatures);
  }

  // 4. Adapter rejects both { requiredFeatures, requiredLimits } and { requiredFeatures } -> falls back to bare requestDevice().
  {
    const { device, ctx } = createMockWebGpu();
    device.features = new Set();
    const supportedFeatures = ['texture-compression-bc'];
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'bare-fallback-gpu', architecture: '' },
          features: new Set(supportedFeatures),
          limits: {
            maxStorageBufferBindingSize: 512 * 1024 * 1024,
          },
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            if (desc) {
              throw new Error('driver rejected device descriptor');
            }
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 320, height: 180 });
    });

    assert.equal(requestDeviceCalls.length, 3);
    assert.deepEqual(requestDeviceCalls[0], {
      requiredFeatures: supportedFeatures,
      requiredLimits: {
        maxStorageBufferBindingSize: 512 * 1024 * 1024,
      },
    });
    assert.deepEqual(requestDeviceCalls[1], {
      requiredFeatures: supportedFeatures,
    });
    assert.equal(requestDeviceCalls[2], undefined);
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg);
    assert.deepEqual(readyMsg.features, []);
  }

  // 5. Adapter rejects { requiredLimits } when no features are supported -> falls back directly to bare requestDevice().
  {
    const { device, ctx } = createMockWebGpu();
    const requestDeviceCalls = [];

    const mockGpu = {
      async requestAdapter() {
        return {
          limits: {
            maxSamplersPerShaderStage: 64,
          },
          async requestDevice(desc) {
            requestDeviceCalls.push(desc);
            if (desc && desc.requiredLimits) {
              throw new Error('limits rejected');
            }
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };

    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 320, height: 180 });
    });

    assert.equal(requestDeviceCalls.length, 2);
    assert.deepEqual(requestDeviceCalls[0], {
      requiredLimits: {
        maxSamplersPerShaderStage: 64,
      },
    });
    assert.equal(requestDeviceCalls[1], undefined);
    const readyMsg = posted.find((m) => m && m.ready === true);
    assert.ok(readyMsg);
    assert.deepEqual(readyMsg.features, []);
  }
});

function createInstanceLostError() {
  const err = new Error('A valid external Instance reference no longer exists.');
  err.name = 'OperationError';
  return err;
}

async function collectUnhandledRejections(fn) {
  const rejections = [];
  const onUnhandled = (reason) => rejections.push(reason);
  process.on('unhandledRejection', onUnhandled);
  try {
    await fn();
    // Let rejected fences settle and any unhandled-rejection detection run.
    for (let i = 0; i < 3; i += 1) {
      await new Promise((resolve) => setImmediate(resolve));
    }
  } finally {
    process.off('unhandledRejection', onUnhandled);
  }
  return rejections;
}

test('W99 _getWorkDoneFence tolerates synchronous OperationError from onSubmittedWorkDone', async () => {
  const { device, ctx, calls } = createMockWebGpu();
  let fenceCalls = 0;
  device.queue.onSubmittedWorkDone = () => {
    fenceCalls += 1;
    throw createInstanceLostError();
  };
  const processor = new WgpuCommandProcessor(device, ctx, {
    format: 'bgra8unorm',
    resourceRegistry: { endFrame() {} },
  });

  const rejections = await collectUnhandledRejections(async () => {
    assert.equal(processor._getWorkDoneFence(), null);
    // _flushPresent submits and then requests the fence when a fence consumer is active; neither may throw.
    assert.doesNotThrow(() => processor._flushPresent());
    assert.doesNotThrow(() =>
      processor.execute(new BinaryCommandBuilder().beginFrame().present().finish())
    );
  });

  assert.deepEqual(rejections, []);
  assert.ok(fenceCalls >= 3);
  assert.equal(processor.frameCount, 2);
  assert.equal(calls.filter((c) => c.op === 'submit').length, 2);
  assert.equal(processor._workDoneFenceFailures, fenceCalls);
  const fenceErrors = processor
    .getValidationErrors()
    .filter((r) => r.message.startsWith('OnSubmittedWorkDone:'));
  assert.equal(fenceErrors.length, 1);
  assert.match(fenceErrors[0].message, /OperationError: A valid external Instance reference/);
});

test('W99 _getWorkDoneFence converts a rejected onSubmittedWorkDone promise into a resolved fence', async () => {
  const { device, ctx } = createMockWebGpu();
  let fenceCalls = 0;
  device.queue.onSubmittedWorkDone = () => {
    fenceCalls += 1;
    return Promise.reject(createInstanceLostError());
  };
  const processor = new WgpuCommandProcessor(device, ctx, {
    format: 'bgra8unorm',
    resourceRegistry: { endFrame() {} },
  });

  let fenceValue = 'unset';
  const rejections = await collectUnhandledRejections(async () => {
    const fence = processor._getWorkDoneFence();
    assert.ok(fence && typeof fence.then === 'function');
    fenceValue = await fence;
    assert.doesNotThrow(() => processor._flushPresent());
    assert.doesNotThrow(() =>
      processor.execute(new BinaryCommandBuilder().beginFrame().present().finish())
    );
  });

  assert.equal(fenceValue, undefined);
  assert.deepEqual(rejections, []);
  assert.equal(processor.frameCount, 2);
  assert.ok(fenceCalls >= 3);
  assert.equal(processor._workDoneFenceFailures, fenceCalls);
  const fenceErrors = processor
    .getValidationErrors()
    .filter((r) => r.message.startsWith('OnSubmittedWorkDone:'));
  assert.equal(fenceErrors.length, 1);

  // A healthy fence still resolves through the wrapper.
  const healthy = createMockWebGpu();
  healthy.device.queue.onSubmittedWorkDone = () => Promise.resolve('done');
  const healthyProcessor = new WgpuCommandProcessor(healthy.device, healthy.ctx, {});
  assert.equal(await healthyProcessor._getWorkDoneFence(), undefined);
  assert.equal(healthyProcessor._workDoneFenceFailures, 0);
});

test('W99 _flushPresent records torn-down instance errors from finish/submit/encoder without throwing', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });

  // queue.submit throws OperationError: no frame counted, no fence requested.
  let fenceCalls = 0;
  device.queue.onSubmittedWorkDone = () => {
    fenceCalls += 1;
    return Promise.resolve();
  };
  const origSubmit = device.queue.submit;
  device.queue.submit = () => {
    throw createInstanceLostError();
  };
  assert.doesNotThrow(() => processor._flushPresent());
  assert.equal(processor.frameCount, 0);
  assert.equal(fenceCalls, 0);
  device.queue.submit = origSubmit;

  // encoder.finish throws OperationError: submit is skipped.
  const origCreateEncoder = device.createCommandEncoder;
  device.createCommandEncoder = function createThrowingFinishEncoder() {
    const enc = origCreateEncoder.call(this);
    enc.finish = () => {
      throw createInstanceLostError();
    };
    return enc;
  };
  const submitsBefore = calls.filter((c) => c.op === 'submit').length;
  assert.doesNotThrow(() => processor._flushPresent());
  assert.equal(calls.filter((c) => c.op === 'submit').length, submitsBefore);
  assert.equal(processor.frameCount, 0);

  // createCommandEncoder itself throws: frame dropped cleanly.
  device.createCommandEncoder = () => {
    throw createInstanceLostError();
  };
  assert.doesNotThrow(() =>
    processor.execute(new BinaryCommandBuilder().beginFrame().present().finish())
  );
  assert.equal(processor.frameCount, 0);
  assert.equal(processor._presentSubmitFailures, 3);

  const presentErrors = processor
    .getValidationErrors()
    .filter((r) => r.message.startsWith('Present:'));
  assert.ok(presentErrors.length >= 1);
  assert.match(presentErrors[0].message, /OperationError: A valid external Instance reference/);

  // Recovery: a healthy encoder presents normally again. With zero deferred destroys and no
  // fence-consuming subsystem, steady-state _flushPresent skips onSubmittedWorkDone().
  device.createCommandEncoder = origCreateEncoder;
  assert.doesNotThrow(() => processor._flushPresent());
  assert.equal(processor.frameCount, 1);
  assert.equal(fenceCalls, 0);

  // When a deferred GPU destroy is queued, _flushPresent requests exactly one fence.
  processor._deferredGpuDestroys.push({ destroy() {} });
  assert.doesNotThrow(() => processor._flushPresent());
  assert.equal(processor.frameCount, 2);
  assert.equal(fenceCalls, 1);
});

test('W99 buildGpuTelemetrySnapshot surfaces render-state compute/auto-layout counters with 0 defaults', () => {
  const { installRenderStatePackets } = require('../web/workers/render_state_packets.js');
  const zeroCounters = {
    computeDispatchCount: 0,
    computePipelineCount: 0,
    autoLayoutFallbacks: 0,
    bindGroupKindMismatches: 0,
  };

  // No render-state telemetry attached: counters default to 0, existing keys unchanged.
  const bare = buildGpuTelemetrySnapshot({ getStats: () => ({ frameCount: 1 }) }, { timestamp: 1 });
  assert.deepEqual(bare.renderStateStats, zeroCounters);
  assert.deepEqual(Object.keys(bare.frameStats), [
    'framesPresented',
    'droppedFrames',
    'drawCalls',
    'realPackDrawCount',
    'fallbackDrawCount',
    'computeDispatches',
    'packetsProcessed',
  ]);
  // Null processor keeps the pre-W99 empty snapshot shape (no renderStateStats key).
  assert.equal('renderStateStats' in buildGpuTelemetrySnapshot(null, { timestamp: 1 }), false);

  // Live render_state_packets.js handler installed on a real processor.
  const { device, ctx } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });
  const handler = installRenderStatePackets(processor);
  assert.deepEqual(
    buildGpuTelemetrySnapshot(processor, { timestamp: 2 }).renderStateStats,
    zeroCounters
  );
  handler.telemetry.computeDispatchCount = 7;
  handler.telemetry.computePipelineCount = 3;
  handler.telemetry.autoLayoutFallbacks = 2;
  handler.telemetry.bindGroupKindMismatches = 4;
  const live = buildGpuTelemetrySnapshot(processor, { timestamp: 3 });
  assert.deepEqual(live.renderStateStats, {
    computeDispatchCount: 7,
    computePipelineCount: 3,
    autoLayoutFallbacks: 2,
    bindGroupKindMismatches: 4,
  });
  assert.equal(live.frameStats.computeDispatches, 0);

  // Explicit renderStateProcessor / _renderStatePackets objects and coercion of bad values.
  const viaProp = buildGpuTelemetrySnapshot(
    {
      getStats: () => ({}),
      renderStateProcessor: {
        getTelemetry: () => ({
          computeDispatchCount: 5.9,
          computePipelineCount: -1,
          autoLayoutFallbacks: 'nope',
          bindGroupKindMismatches: 3.7,
        }),
      },
    },
    { timestamp: 4 }
  );
  assert.deepEqual(viaProp.renderStateStats, {
    computeDispatchCount: 5,
    computePipelineCount: 0,
    autoLayoutFallbacks: 0,
    bindGroupKindMismatches: 3,
  });
  const viaPrivate = buildGpuTelemetrySnapshot(
    {
      getStats: () => ({}),
      _renderStatePackets: { telemetry: { computePipelineCount: 4 } },
    },
    { timestamp: 5 }
  );
  assert.deepEqual(viaPrivate.renderStateStats, {
    computeDispatchCount: 0,
    computePipelineCount: 4,
    autoLayoutFallbacks: 0,
    bindGroupKindMismatches: 0,
  });

  // Throwing telemetry getters never break the snapshot.
  const throwing = buildGpuTelemetrySnapshot(
    {
      getStats: () => ({}),
      renderStateProcessor: {
        getTelemetry() {
          throw new Error('boom');
        },
      },
    },
    { timestamp: 6 }
  );
  assert.deepEqual(throwing.renderStateStats, zeroCounters);
});

test('W332 buildMergedBinkPresenterOptions defaults enableTripleBuffering: true unless explicitly false (W316)', async () => {
  const { buildMergedBinkPresenterOptions } = wgpuMod;
  const { BinkVideoPresenter } = require('../web/workers/bink_presenter.js');
  assert.equal(typeof buildMergedBinkPresenterOptions, 'function');

  // Default behavior: enableTripleBuffering and allowUnalignedDirectUpload default to true.
  assert.deepEqual(buildMergedBinkPresenterOptions(null, null), {
    enableTripleBuffering: true,
    allowUnalignedDirectUpload: true,
  });
  assert.deepEqual(buildMergedBinkPresenterOptions({}, undefined), {
    enableTripleBuffering: true,
    allowUnalignedDirectUpload: true,
  });
  assert.deepEqual(buildMergedBinkPresenterOptions({ chromaOrder: 'YVU' }, null), {
    chromaOrder: 'yvu',
    enableTripleBuffering: true,
    allowUnalignedDirectUpload: true,
  });

  // Explicit false in base or extra options is preserved.
  assert.equal(
    buildMergedBinkPresenterOptions({ allowUnalignedDirectUpload: false }, null)
      .allowUnalignedDirectUpload,
    false
  );
  assert.equal(
    buildMergedBinkPresenterOptions(
      { allowUnalignedDirectUpload: true },
      { allowUnalignedDirectUpload: false }
    ).allowUnalignedDirectUpload,
    false
  );
  assert.equal(
    buildMergedBinkPresenterOptions({ enableTripleBuffering: false }, null).enableTripleBuffering,
    false
  );
  assert.equal(
    buildMergedBinkPresenterOptions({ enableTripleBuffering: true }, { enableTripleBuffering: false })
      .enableTripleBuffering,
    false
  );
  assert.equal(
    buildMergedBinkPresenterOptions({ enableTripleBuffering: false }, { enableTripleBuffering: true })
      .enableTripleBuffering,
    true
  );

  // Live WgpuCommandProcessor constructs BinkVideoPresenter with 3-slot triple buffering by default.
  const makePresenterMockDevice = () => {
    const base = createMockWebGpu();
    base.device.createBindGroupLayout = (desc) => ({ kind: 'GPUBindGroupLayout', desc });
    base.device.createPipelineLayout = (desc) => ({ kind: 'GPUPipelineLayout', desc });
    return base;
  };
  const envDefault = makePresenterMockDevice();
  const procDefault = new WgpuCommandProcessor(envDefault.device, envDefault.ctx, {
    enableSplashBridge: true,
    BinkVideoPresenter,
  });
  assert.ok(procDefault.binkPresenter instanceof BinkVideoPresenter);
  assert.equal(procDefault.binkPresenter.planeRingDepth, 3);
  procDefault.destroy();

  // Explicit enableTripleBuffering: false falls back to double-buffered 2-slot ring.
  const envOptOut = makePresenterMockDevice();
  const procOptOut = new WgpuCommandProcessor(envOptOut.device, envOptOut.ctx, {
    enableSplashBridge: true,
    BinkVideoPresenter,
    binkPresenterOptions: { enableTripleBuffering: false },
  });
  assert.equal(procOptOut.binkPresenter.planeRingDepth, 2);
  procOptOut.destroy();

  // Worker init also honors explicit enableTripleBuffering: false.
  const envWorker = makePresenterMockDevice();
  const { canvas } = createMockOffscreenCanvas(envWorker.ctx);
  const posted = [];
  const mockGpu = {
    async requestAdapter() {
      return {
        info: { vendor: 'mock', architecture: 'w332' },
        async requestDevice() {
          return envWorker.device;
        },
      };
    },
    getPreferredCanvasFormat() {
      return 'bgra8unorm';
    },
  };
  const handler = createWgpuWorkerHandler({
    postMessage: (m) => posted.push(m),
    enableSwapchain: false,
  });
  await withMockNavigatorGpu(mockGpu, async () => {
    await handler.handleMessage({
      init: true,
      canvas,
      width: 640,
      height: 360,
      enableSplashBridge: true,
      enableTripleBuffering: false,
    });
  });
  assert.equal(handler.processor.binkPresenter.planeRingDepth, 2);
  handler.processor.destroy();
});

test('W332 active uniformRing advances frame slot cleanly on present with function guards (W314)', () => {
  const { DynamicUniformRing } = require('../web/workers/uniform_ring.js');
  const { device, ctx } = createMockWebGpu();
  device.queue.onSubmittedWorkDone = () => Promise.resolve();

  const uniformRing = new DynamicUniformRing({
    device,
    byteSize: 2048,
    framesInFlight: 3,
  });
  const processor = new WgpuCommandProcessor(device, ctx, {
    format: 'bgra8unorm',
    uniformRing,
  });

  // Present without explicit BEGIN_FRAME still rotates slot cleanly (0 -> 1 -> 2 -> 0).
  const presentOnly = new BinaryCommandBuilder().present().finish();
  processor.execute(presentOnly);
  assert.equal(uniformRing.getStats().slotIndex, 1);
  assert.equal(uniformRing.getStats().frameIndex, 1);

  processor.execute(presentOnly);
  assert.equal(uniformRing.getStats().slotIndex, 2);
  assert.equal(uniformRing.getStats().frameIndex, 2);

  // Explicit BEGIN_FRAME matching already-advanced frameIndex must NOT double-rotate.
  const rotationsBefore = uniformRing.getStats().partitionRotations;
  const withBeginAndPresent = new BinaryCommandBuilder().beginFrame().present().finish();
  processor.execute(withBeginAndPresent);
  assert.equal(uniformRing.getStats().slotIndex, 0);
  assert.equal(uniformRing.getStats().frameIndex, 3);
  assert.equal(uniformRing.getStats().partitionRotations, rotationsBefore + 1);
  processor.destroy();

  // Custom mock exposing advanceFrame() + endFrame() is invoked safely.
  const events = [];
  const mockAdvanceRing = {
    flush() {
      events.push('flush');
    },
    endFrame() {
      events.push('endFrame');
    },
    advanceFrame(frameIdx) {
      events.push(`advanceFrame:${frameIdx}`);
    },
  };
  const procMock = new WgpuCommandProcessor(device, ctx, { uniformRing: mockAdvanceRing });
  procMock.execute(presentOnly);
  assert.deepEqual(events, ['flush', 'endFrame', 'advanceFrame:1']);

  // Partial mock lacking beginFrame/advanceFrame/endFrame/flush never throws.
  const procBare = new WgpuCommandProcessor(device, ctx, { uniformRing: { custom: true } });
  assert.doesNotThrow(() => procBare.execute(presentOnly));
  assert.equal(procBare.frameCount, 1);
});

test('W332 trimTransientMemory trims StagingPool slabs (W319) and RT pool unused/oversized targets (W285)', async () => {
  const { GpuStagingPool } = require('../web/workers/staging_pool.js');
  const { RenderTargetPool } = require('../web/workers/rt_pool.js');
  const { device, ctx } = createMockWebGpu();
  device.queue.onSubmittedWorkDone = () => Promise.resolve();

  // 1. Real GpuStagingPool + real RenderTargetPool via trimTransientMemory & scene-transition.
  const stagingPool = new GpuStagingPool({
    device,
    chunkSize: 1024 * 1024, // 1 MiB standard chunk
    maxPoolBytes: 64 * 1024 * 1024,
    framesInFlight: 2,
    mode: 'writeBuffer',
  });
  const rtPool = new RenderTargetPool({
    device,
    maxPoolBytes: 256 * 1024 * 1024,
    idleFrameTtl: 16,
  });

  const processor = new WgpuCommandProcessor(device, ctx, {
    format: 'bgra8unorm',
    stagingPool,
    rtPool,
  });

  // Allocate an oversized 4 MiB staging slab and recycle it across frames so it's idle.
  stagingPool.beginFrame(0);
  stagingPool.allocate(4 * 1024 * 1024);
  stagingPool.endFrame();
  stagingPool.beginFrame(1);
  stagingPool.endFrame();
  stagingPool.beginFrame(2);
  stagingPool.endFrame();

  // Acquire & release a 4K oversized render target (3840x2160 rgba8unorm = 33,177,600B) and age 2 frames.
  const rt4kBytes = 3840 * 2160 * 4;
  const rt4k = processor.acquireRenderTarget(
    { width: 3840, height: 2160, format: 'rgba8unorm' },
    'scene-4k'
  );
  processor.releaseRenderTarget(rt4k);
  rtPool.endFrame();
  rtPool.endFrame();

  const trimmed = processor.trimTransientMemory();
  assert.equal(trimmed.stagingReclaimedSlabs, 1);
  assert.equal(trimmed.stagingReclaimedBytes, 4 * 1024 * 1024);
  assert.equal(trimmed.rtReclaimedCount, 1);
  assert.equal(trimmed.rtReclaimedBytes, rt4kBytes);
  assert.equal(trimmed.reclaimedCount, 2);
  assert.equal(trimmed.reclaimedBytes, 4 * 1024 * 1024 + rt4kBytes);
  assert.equal(stagingPool.getStats().sceneTransitionTrimBytes, 4 * 1024 * 1024);
  processor.destroy();

  // 2. Mock rtPool exposing reclaimUnused() + mock stagingPool exposing trimForSceneTransition()
  //    wired through worker memory-pressure & scene-transition messages.
  let stagingTrimCalls = 0;
  let rtReclaimUnusedCalls = 0;
  const mockProc = new WgpuCommandProcessor(device, ctx, {
    stagingPool: {
      trimForSceneTransition() {
        stagingTrimCalls += 1;
        return { reclaimedSlabs: 2, reclaimedBytes: 8192 };
      },
    },
    rtPool: {
      reclaimUnused() {
        rtReclaimUnusedCalls += 1;
        return { reclaimedCount: 3, reclaimedBytes: 16384 };
      },
    },
  });
  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor: mockProc,
    postMessage: (m) => posted.push(m),
  });
  await handler.handleMessage({ type: 'memory-pressure', id: 'mp-1' });
  const mpReply = posted.find((m) => m.id === 'mp-1');
  assert.ok(mpReply);
  assert.equal(mpReply.type, 'memory-pressure');
  assert.equal(mpReply.stagingReclaimedSlabs, 2);
  assert.equal(mpReply.stagingReclaimedBytes, 8192);
  assert.equal(mpReply.rtReclaimedCount, 3);
  assert.equal(mpReply.rtReclaimedBytes, 16384);
  assert.equal(mpReply.reclaimedCount, 5);
  assert.equal(mpReply.reclaimedBytes, 24576);
  assert.equal(stagingTrimCalls, 1);
  assert.equal(rtReclaimUnusedCalls, 1);

  // Throwing mocks never crash trimTransientMemory().
  const throwingProc = new WgpuCommandProcessor(device, ctx, {
    stagingPool: {
      trimForSceneTransition() {
        throw new Error('staging trim boom');
      },
    },
    rtPool: {
      reclaimUnused() {
        throw new Error('rt reclaim boom');
      },
    },
  });
  assert.deepEqual(throwingProc.trimTransientMemory(), {
    reclaimedCount: 0,
    reclaimedBytes: 0,
    stagingReclaimedSlabs: 0,
    stagingReclaimedBytes: 0,
    rtReclaimedCount: 0,
    rtReclaimedBytes: 0,
  });
});

test('W332 surfaces GpuPassTimer 30 FPS per-pass histogram (W324) and GpuFrameRecorder spike state (W325) in getStats & telemetry', async () => {
  const { GpuPassTimer } = require('../web/workers/gpu_timer.js');
  const { GpuFrameRecorder } = require('../web/workers/gpu_recorder.js');
  const { device, ctx } = createMockWebGpu();

  let fakeNow = 100;
  const nowFn = () => fakeNow;

  // CPU-mode GpuPassTimer (device without timestamp-query still records rolling 30 FPS pass histograms).
  const gpuTimer = new GpuPassTimer({
    device,
    passBudgetMs: 33.333,
    now: nowFn,
  });
  gpuTimer.beginFrame();
  gpuTimer.beginPass('shadow_pass');
  fakeNow = 112;
  gpuTimer.endPass();
  gpuTimer.beginPass('gbuffer_pass');
  fakeNow = 152; // 40ms > 33.333ms budget
  gpuTimer.endPass();
  gpuTimer.endFrame();

  // GpuFrameRecorder with 30 FPS spike auto-capture enabled.
  const gpuRecorder = new GpuFrameRecorder({
    maxFrames: 4,
    maxRecordedBytes: 64 * 1024,
    now: nowFn,
  });
  gpuRecorder.enableSpikeTrigger({ budgetMs: 33.333, maxSnapshots: 2 });

  const processor = new WgpuCommandProcessor(device, ctx, {
    format: 'bgra8unorm',
    gpuTimer,
    gpuRecorder,
    now: nowFn,
  });
  assert.equal(processor.gpuTimer, gpuTimer);
  assert.equal(processor.passTimer, gpuTimer);

  // Attach recorder interceptor so chunks are staged during a frame.
  const interceptor = gpuRecorder.createInterceptor(processor);
  const frameStream = new BinaryCommandBuilder().beginFrame().clearColor(0.2, 0.4, 0.6, 1.0).present().finish();

  // Frame 1: fast 16ms frame (under 33.333ms budget) -> NOT captured as spike.
  fakeNow = 200;
  processor.beginFrame(0);
  fakeNow = 216;
  processor.executeCommandBuffer(frameStream);
  assert.equal(gpuRecorder.getStats().spikeSnapshotsCaptured, 0);

  // Frame 2: slow 45ms frame (over 33.333ms budget) -> auto-captured as spike snapshot!
  fakeNow = 300;
  processor.beginFrame(1);
  fakeNow = 345;
  processor.executeCommandBuffer(frameStream);
  assert.equal(gpuRecorder.getStats().spikeSnapshotsCaptured, 1);
  assert.equal(gpuRecorder.getStats().lastSpikeFrameMs, 45);
  interceptor.detach();

  // Verify getStats() exposes sanitized gpuTimer + gpuRecorder summaries.
  const stats = processor.getStats();
  assert.ok(stats.gpuTimer);
  assert.equal(stats.gpuTimer.passBudgetMs, 33.333);
  assert.equal(stats.gpuTimer.passesOverBudget, 1);
  assert.equal(stats.gpuTimer.maxPassMs, 40);
  assert.ok(stats.gpuTimer.passStats.shadow_pass);
  assert.equal(stats.gpuTimer.passStats.shadow_pass.lastMs, 12);
  assert.ok(stats.gpuTimer.passStats.gbuffer_pass);
  assert.equal(stats.gpuTimer.passStats.gbuffer_pass.passesOverBudget, 1);

  assert.ok(stats.gpuRecorder);
  assert.equal(stats.gpuRecorder.spikeTriggerEnabled, true);
  assert.equal(stats.gpuRecorder.spikeBudgetMs, 33.333);
  assert.equal(stats.gpuRecorder.spikeMaxSnapshots, 2);
  assert.equal(stats.gpuRecorder.spikeSnapshotsCaptured, 1);
  assert.equal(stats.gpuRecorder.lastSpikeFrameMs, 45);

  // Verify buildGpuTelemetrySnapshot() & worker request-gpu-stats surface both when active.
  const snap = buildGpuTelemetrySnapshot(processor, { timestamp: 500 });
  assert.ok(snap.gpuTimer);
  assert.equal(snap.gpuTimer.passesOverBudget, 1);
  assert.equal(snap.gpuTimer.passStats.gbuffer_pass.maxMs, 40);
  assert.ok(snap.gpuRecorder);
  assert.equal(snap.gpuRecorder.spikeTriggerEnabled, true);
  assert.equal(snap.gpuRecorder.spikeSnapshotsCaptured, 1);

  const posted = [];
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
  });
  await handler.handleMessage({ type: 'request-gpu-stats', id: 'stats-w332' });
  const statsReply = posted.find((m) => m.id === 'stats-w332' && m.type === 'gpu-stats');
  assert.ok(statsReply);
  assert.equal(statsReply.gpuTimer.passesOverBudget, 1);
  assert.equal(statsReply.gpuRecorder.spikeSnapshotsCaptured, 1);

  // Inactive processor omits gpuTimer & gpuRecorder from telemetry snapshot cleanly.
  const bareProc = new WgpuCommandProcessor(device, ctx, { enablePassTimer: false });
  assert.equal(bareProc.getStats().gpuTimer, null);
  assert.equal(bareProc.getStats().gpuRecorder, null);
  const bareSnap = buildGpuTelemetrySnapshot(bareProc, { timestamp: 600 });
  assert.equal('gpuTimer' in bareSnap, false);
  assert.equal('gpuRecorder' in bareSnap, false);
  processor.destroy();
});

test('W332 optional requireNonBlackFrame gates first-visible-frame via W307 luminance check for command-buffer & Bink2 frames', async () => {
  const { device, ctx } = createMockWebGpu();
  device.createBindGroupLayout = (desc) => ({ kind: 'GPUBindGroupLayout', desc });
  device.createPipelineLayout = (desc) => ({ kind: 'GPUPipelineLayout', desc });
  const vsWgsl = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }';

  const buildDrawFrame = () =>
    new BinaryCommandBuilder()
      .beginFrame()
      .createShaderModule(1, SHADER_STAGE.VERTEX, vsWgsl)
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, fsWgsl)
      .createRenderPipeline(3, 1, 2, 0)
      .setPipeline(3)
      .draw(3, 1, 0, 0)
      .present()
      .finish();

  // 1. Default requireNonBlackFrame === false: backwards-compatible immediate emit on first draw.
  {
    const procDefault = new WgpuCommandProcessor(device, ctx);
    assert.equal(procDefault.requireNonBlackFrame, false);
    procDefault.execute(buildDrawFrame());
    assert.equal(procDefault.firstVisibleFrame, true);
    assert.equal(procDefault.suppressedBlackFrames, 0);
  }

  // 2. Worker init option requireNonBlackFrame: true gates pure-black draws until non-black pixels arrive.
  {
    const { canvas } = createMockOffscreenCanvas(ctx);
    const posted = [];
    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'mock', architecture: 'w332-luma' },
          async requestDevice() {
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };
    let fakeNow = 1000;
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
      now: () => fakeNow,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({
        init: true,
        canvas,
        width: 16,
        height: 16,
        requireNonBlackFrame: true,
      });
    });

    assert.equal(handler.processor.requireNonBlackFrame, true);

    // Draw 1 with all-black RGBA pixels -> suppressed!
    const blackRgba = new Uint8Array(16 * 16 * 4);
    for (let i = 3; i < blackRgba.length; i += 4) blackRgba[i] = 255;
    await handler.handleMessage({
      cmdBuffer: buildDrawFrame(),
      frameLuminance: { data: blackRgba, width: 16, height: 16 },
    });
    assert.equal(handler.processor.firstVisibleFrame, false);
    assert.equal(handler.processor.suppressedBlackFrames, 1);
    assert.equal(posted.filter((m) => m.type === 'first-visible-frame').length, 0);

    // Draw 2 with bright non-black RGBA pixels -> W307 hasVisibleNonBlackPixels passes, emits ONCE!
    fakeNow = 1250;
    const brightRgba = new Uint8Array(16 * 16 * 4).fill(200);
    await handler.handleMessage({
      cmdBuffer: buildDrawFrame(),
      frameLuminance: { data: brightRgba, width: 16, height: 16 },
    });
    assert.equal(handler.processor.firstVisibleFrame, true);
    assert.equal(handler.processor.suppressedBlackFrames, 1);
    assert.equal(handler.processor.lastSampledMaxLuma, 200);
    const visMsgs = posted.filter((m) => m.type === 'first-visible-frame');
    assert.equal(visMsgs.length, 1);
    assert.equal(visMsgs[0].maxLuma, 200);
    assert.equal(visMsgs[0].suppressedBlackFrames, 1);
    handler.processor.destroy();
  }

  // 3. With requireNonBlackFrame: true, black Bink2 frame is suppressed and bright Bink2 frame emits immediately.
  {
    const binkMock = createMockWebGpu();
    binkMock.device.createBindGroupLayout = (desc) => ({ kind: 'GPUBindGroupLayout', desc });
    binkMock.device.createPipelineLayout = (desc) => ({ kind: 'GPUPipelineLayout', desc });
    const origCreateEncoder = binkMock.device.createCommandEncoder.bind(binkMock.device);
    binkMock.device.createCommandEncoder = () => {
      const enc = origCreateEncoder();
      const origBeginRenderPass = enc.beginRenderPass.bind(enc);
      enc.beginRenderPass = (desc) => {
        const pass = origBeginRenderPass(desc);
        pass.setViewport = () => {};
        pass.setScissorRect = () => {};
        return pass;
      };
      return enc;
    };

    const posted = [];
    const procBink = new WgpuCommandProcessor(binkMock.device, binkMock.ctx, {
      width: 16,
      height: 8,
      requireNonBlackFrame: true,
      enableSplashBridge: true,
    });
    const handler = createWgpuWorkerHandler({
      processor: procBink,
      postMessage: (m) => posted.push(m),
    });

    // Black Bink2 YUV420P frame (Y=16 limited-range black) -> presented, but first-visible-frame suppressed.
    await handler.handleMessage({
      type: 'bink-frame',
      id: 'bink-black',
      frame: {
        width: 16,
        height: 8,
        format: 'yuv420p',
        colorRange: 'limited',
        planes: {
          y: new Uint8Array(16 * 8).fill(16),
          u: new Uint8Array(8 * 4).fill(128),
          v: new Uint8Array(8 * 4).fill(128),
        },
        frameIndex: 0,
      },
    });
    const blackReply = posted.find((m) => m.id === 'bink-black');
    assert.ok(blackReply && blackReply.ok === true);
    assert.equal(procBink.firstVisibleFrame, false);
    assert.ok(procBink.suppressedBlackFrames >= 1);
    assert.equal(posted.filter((m) => m.type === 'first-visible-frame').length, 0);

    // Visible bright Bink2 YUV420P frame (Y=200) -> emits first-visible-frame immediately!
    await handler.handleMessage({
      type: 'bink-frame',
      id: 'bink-bright',
      frame: {
        width: 16,
        height: 8,
        format: 'yuv420p',
        colorRange: 'limited',
        planes: {
          y: new Uint8Array(16 * 8).fill(200),
          u: new Uint8Array(8 * 4).fill(128),
          v: new Uint8Array(8 * 4).fill(128),
        },
        frameIndex: 1,
      },
    });
    const brightReply = posted.find((m) => m.id === 'bink-bright');
    assert.ok(brightReply && brightReply.ok === true);
    assert.equal(procBink.firstVisibleFrame, true);
    const visMsgs = posted.filter((m) => m.type === 'first-visible-frame');
    assert.equal(visMsgs.length, 1);
    assert.ok(visMsgs[0].maxLuma >= 180);
    procBink.destroy();
  }
});

// ---------------------------------------------------------------------------
// W338: Async non-black swapchain luminance probe tests (#1453)
// ---------------------------------------------------------------------------

function createProbeCapableMockWebGpu({
  width = 1280,
  height = 720,
  format = 'bgra8unorm',
  textureUsage = TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.COPY_SRC,
  deferMapAsync = false,
} = {}) {
  const base = createMockWebGpu();
  const pixels = new Uint8Array(width * height * 4);
  // Default: solid black RGB with opaque alpha (255) — alpha must be ignored by probe!
  for (let i = 3; i < pixels.length; i += 4) {
    pixels[i] = 255;
  }

  const copyCalls = [];
  const probeBuffers = [];
  const pendingMaps = [];
  const bufferEvents = [];

  const setPixel = (x, y, c0, c1, c2, a = 255) => {
    const off = (y * width + x) * 4;
    pixels[off] = c0;
    pixels[off + 1] = c1;
    pixels[off + 2] = c2;
    pixels[off + 3] = a;
  };

  const clearPixels = (c0 = 0, c1 = 0, c2 = 0, a = 255) => {
    for (let i = 0; i < pixels.length; i += 4) {
      pixels[i] = c0;
      pixels[i + 1] = c1;
      pixels[i + 2] = c2;
      pixels[i + 3] = a;
    }
  };

  const ctx = {
    configured: [],
    configure(desc) {
      this.configured.push({ ...desc });
    },
    getCurrentTexture() {
      return {
        width,
        height,
        format,
        usage: textureUsage,
        createView() {
          return base.mockTextureView;
        },
      };
    },
  };

  const origCreateBuffer = base.device.createBuffer.bind(base.device);
  base.device.createBuffer = (desc) => {
    const buf = origCreateBuffer(desc);
    const backing = new Uint8Array(desc.size);
    let mapped = false;
    let mapInFlight = false;
    buf._backing = backing;
    buf.mapCalls = 0;
    buf.unmapCalls = 0;
    buf.doubleMapDetected = false;
    buf.mapAsync = (mode) => {
      if (buf.destroyed) {
        return Promise.reject(new Error('cannot map destroyed buffer'));
      }
      if (mapped || mapInFlight) {
        buf.doubleMapDetected = true;
        return Promise.reject(new Error('buffer already mapped or map in flight'));
      }
      buf.mapCalls += 1;
      mapInFlight = true;
      bufferEvents.push({ op: 'mapAsync', handle: buf.handle, mode });
      if (deferMapAsync) {
        return new Promise((resolve, reject) => {
          pendingMaps.push({
            buffer: buf,
            resolve: () => {
              mapInFlight = false;
              if (buf.destroyed) {
                reject(new Error('buffer destroyed while mapping'));
                return;
              }
              mapped = true;
              resolve();
            },
            reject: (err) => {
              mapInFlight = false;
              reject(err || new Error('mapAsync aborted'));
            },
          });
        });
      }
      return Promise.resolve().then(() => {
        mapInFlight = false;
        if (buf.destroyed) {
          throw new Error('buffer destroyed while mapping');
        }
        mapped = true;
      });
    };
    buf.getMappedRange = (offset = 0, size = backing.byteLength - offset) => {
      if (!mapped || buf.destroyed) {
        throw new Error('buffer not mapped');
      }
      bufferEvents.push({ op: 'getMappedRange', handle: buf.handle, offset, size });
      return backing.buffer.slice(offset, offset + size);
    };
    buf.unmap = () => {
      if (buf.destroyed) {
        throw new Error('cannot unmap destroyed buffer');
      }
      mapped = false;
      buf.unmapCalls += 1;
      bufferEvents.push({ op: 'unmap', handle: buf.handle });
    };
    const origDestroy = buf.destroy.bind(buf);
    buf.destroy = () => {
      mapped = false;
      mapInFlight = false;
      bufferEvents.push({ op: 'destroy', handle: buf.handle });
      origDestroy();
    };
    if (desc && desc.label === 'playgta6-swapchain-luma-probe') {
      probeBuffers.push(buf);
    }
    return buf;
  };

  const origCreateEncoder = base.device.createCommandEncoder.bind(base.device);
  base.device.createCommandEncoder = () => {
    const enc = origCreateEncoder();
    enc.copyTextureToBuffer = (source, destination, copySize) => {
      copyCalls.push({
        source: { ...source, origin: source.origin ? { ...source.origin } : { x: 0, y: 0, z: 0 } },
        destination: { ...destination },
        copySize: { ...copySize },
      });
      const dstBacking = destination.buffer && destination.buffer._backing;
      if (dstBacking) {
        const ox = (source.origin && source.origin.x) || 0;
        const oy = (source.origin && source.origin.y) || 0;
        const cw = copySize.width || 0;
        const ch = copySize.height || 1;
        const dstOff = destination.offset || 0;
        const bpr = destination.bytesPerRow || cw * 4;
        for (let row = 0; row < ch; row += 1) {
          const srcStart = ((oy + row) * width + ox) * 4;
          const dstStart = dstOff + row * bpr;
          dstBacking.set(pixels.subarray(srcStart, srcStart + cw * 4), dstStart);
        }
      }
    };
    return enc;
  };

  base.device.createBindGroupLayout = (desc) => ({ kind: 'GPUBindGroupLayout', desc });
  base.device.createPipelineLayout = (desc) => ({ kind: 'GPUPipelineLayout', desc });

  return {
    device: base.device,
    ctx,
    calls: base.calls,
    pixels,
    setPixel,
    clearPixels,
    copyCalls,
    probeBuffers,
    pendingMaps,
    bufferEvents,
  };
}

function buildRealPackDrawStream(vsHash, fsHash, { includeSetup = true } = {}) {
  const b = new BinaryCommandBuilder().beginFrame();
  if (includeSetup) {
    b.createShaderModule(1, SHADER_STAGE.VERTEX, formatHashHex(vsHash))
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, formatHashHex(fsHash))
      .createRenderPipeline(3, 1, 2, 0);
  }
  return b.setPipeline(3).draw(3, 1, 0, 0).present().finish();
}

function createW338PackBuffer() {
  const vsWgsl = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }';
  const vsHash = fnv1a64(vsWgsl);
  const fsHash = fnv1a64(fsWgsl);
  const packBuffer = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: vsWgsl },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fsWgsl },
  ]);
  return { vsHash, fsHash, packBuffer };
}

async function flushMicrotasks(turns = 5) {
  for (let i = 0; i < turns; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

test('W338 all-black swapchain frames hold first-visible-frame until non-black pixel appears (including bottom-right bar at y=92% and center 6-star)', async () => {
  const { vsHash, fsHash, packBuffer } = createW338PackBuffer();

  // 1. bgra8unorm 1280x720: bottom-right OFFLINEPLAYER [A] bar at y=92% triggers first-visible-frame.
  {
    const env = createProbeCapableMockWebGpu({ width: 1280, height: 720, format: 'bgra8unorm' });
    let fakeNow = 1000;
    const posted = [];
    const processor = new WgpuCommandProcessor(env.device, env.ctx, {
      format: 'bgra8unorm',
      width: 1280,
      height: 720,
      now: () => fakeNow,
      wgslPackBuffer: packBuffer,
    });
    const handler = createWgpuWorkerHandler({
      processor,
      postMessage: (m) => posted.push(m),
      now: () => fakeNow,
    });

    // Clear-only frame before any draw: must NOT allocate probe buffer or run copyTextureToBuffer.
    await handler.handleMessage({
      cmdBuffer: new BinaryCommandBuilder().beginFrame().clearColor(0, 0, 0, 1).present().finish(),
    });
    await flushMicrotasks();
    assert.equal(env.copyCalls.length, 0);
    assert.equal(env.probeBuffers.length, 0);
    assert.equal(processor.firstVisibleFrame, false);

    // Draw #1 with solid black RGB (0,0,0,255): probe runs, holds first-visible-frame!
    fakeNow = 1100;
    await handler.handleMessage({
      cmdBuffer: buildRealPackDrawStream(vsHash, fsHash, { includeSetup: true }),
    });
    await flushMicrotasks();
    assert.equal(env.copyCalls.length, 8);
    assert.equal(processor.firstVisibleFrame, false);
    assert.equal(processor.suppressedBlackFrames, 1);
    assert.equal(posted.filter((m) => m.type === 'first-visible-frame').length, 0);

    // Verify sampled strips match 8 horizontal strips at y = [15%, 35%, 50%, 65%, 80%, 89%, 92%, 95%], width <= 64, bytesPerRow = 256.
    assert.deepEqual(
      env.copyCalls.map((c) => c.source.origin.y),
      [
        Math.floor(720 * 0.15),
        Math.floor(720 * 0.35),
        Math.floor(720 * 0.5),
        Math.floor(720 * 0.65),
        Math.floor(720 * 0.8),
        Math.floor(720 * 0.89),
        Math.floor(720 * 0.92),
        Math.floor(720 * 0.95),
      ]
    );
    assert.ok(env.copyCalls.every((c) => c.copySize.width === 64 && c.copySize.height === 1));
    assert.ok(env.copyCalls.every((c) => c.destination.bytesPerRow === 256));
    assert.deepEqual(
      env.copyCalls.map((c) => c.destination.offset),
      [0, 256, 512, 768, 1024, 1280, 1536, 1792]
    );

    // Draw #2 with near-black noise (8,8,8,255) — threshold is strictly > 8, still held!
    fakeNow = 1400;
    env.clearPixels(8, 8, 8, 255);
    await handler.handleMessage({
      cmdBuffer: buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }),
    });
    await flushMicrotasks();
    assert.equal(processor.firstVisibleFrame, false);
    assert.equal(processor.suppressedBlackFrames, 2);
    assert.equal(processor.lastSampledMaxLuma, 8);

    // Draw #3 paints a bright pixel ONLY in the bottom-right OFFLINEPLAYER [A] bar at y=92% (bgra: B=0, G=0, R=220, A=255).
    fakeNow = 1700;
    env.clearPixels(0, 0, 0, 255);
    const barY = Math.floor(720 * 0.92);
    const barX = Math.floor((1280 - 64) * 0.9) + 12;
    env.setPixel(barX, barY, 0, 0, 220, 255);
    await handler.handleMessage({
      cmdBuffer: buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }),
    });
    await flushMicrotasks();

    assert.equal(processor.firstVisibleFrame, true);
    assert.equal(processor.lastSampledMaxLuma, 220);
    const visMsgs = posted.filter((m) => m.type === 'first-visible-frame');
    assert.equal(visMsgs.length, 1);
    assert.equal(visMsgs[0].source, 'swapchain-probe');
    assert.equal(visMsgs[0].maxLuma, 220);
    assert.equal(visMsgs[0].suppressedBlackFrames, 2);
    assert.equal(visMsgs[0].firstVisibleFrameMs, 1700);
    processor.destroy();
  }

  // 2. rgba8unorm: center 6-star loading pixel at y=50% triggers first-visible-frame.
  {
    const env = createProbeCapableMockWebGpu({ width: 1280, height: 720, format: 'rgba8unorm' });
    let fakeNow = 2000;
    const processor = new WgpuCommandProcessor(env.device, env.ctx, {
      format: 'rgba8unorm',
      width: 1280,
      height: 720,
      now: () => fakeNow,
      wgslPackBuffer: packBuffer,
    });
    env.setPixel(640, Math.floor(720 * 0.5), 0, 185, 0, 255);
    processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: true }));
    await flushMicrotasks();
    assert.equal(processor.firstVisibleFrame, true);
    assert.equal(processor.lastSampledMaxLuma, 185);
    processor.destroy();
  }
});

test('W338 throttles probes (>= 250ms) and enforces strictly non-overlapping in-flight guard (never double-maps)', async () => {
  const { vsHash, fsHash, packBuffer } = createW338PackBuffer();
  const env = createProbeCapableMockWebGpu({
    width: 1280,
    height: 720,
    deferMapAsync: true,
  });
  let fakeNow = 1000;
  const processor = new WgpuCommandProcessor(env.device, env.ctx, {
    format: 'bgra8unorm',
    width: 1280,
    height: 720,
    now: () => fakeNow,
    wgslPackBuffer: packBuffer,
  });

  // Frame 1 at t=1000ms starts probe #1 (remains in flight).
  processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: true }));
  assert.equal(processor.swapchainProbeInFlight, true);
  assert.equal(processor.swapchainProbeCount, 1);
  assert.equal(env.copyCalls.length, 8);
  assert.equal(env.pendingMaps.length, 1);
  assert.equal(processor.firstVisibleFrame, false);

  // Frame 2 at t=1016ms (in-flight) and Frame 3 at t=1500ms (still in-flight, even past 250ms!):
  // must NEVER start a second overlapping readback or double-map the buffer!
  fakeNow = 1016;
  processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }));
  fakeNow = 1500;
  processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }));
  assert.equal(env.copyCalls.length, 8);
  assert.equal(env.pendingMaps.length, 1);
  assert.equal(processor.swapchainProbeCount, 1);
  assert.equal(env.probeBuffers[0].doubleMapDetected, false);
  assert.equal(processor.firstVisibleFrame, false);

  // Complete probe #1 as all-black at t=1100ms (reset clock to test throttle window relative to t=1000ms start).
  fakeNow = 1100;
  env.pendingMaps.shift().resolve();
  await flushMicrotasks();
  assert.equal(processor.swapchainProbeInFlight, false);
  assert.equal(processor.suppressedBlackFrames, 1);

  // Frame 4 at t=1200ms (< 250ms since startMs=1000ms): throttled, no copy/mapAsync!
  fakeNow = 1200;
  processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }));
  assert.equal(env.copyCalls.length, 8);
  assert.equal(env.pendingMaps.length, 0);
  assert.equal(processor.swapchainProbeCount, 1);
  assert.equal(processor.firstVisibleFrame, false);

  // Frame 5 at t=1250ms (>= 250ms since startMs=1000ms): starts probe #2 reusing the exact same buffer!
  fakeNow = 1250;
  env.setPixel(640, Math.floor(720 * 0.5), 140, 200, 255, 255);
  processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }));
  assert.equal(env.probeBuffers.length, 1);
  assert.equal(env.copyCalls.length, 16);
  assert.equal(env.pendingMaps.length, 1);
  assert.equal(processor.swapchainProbeCount, 2);

  env.pendingMaps.shift().resolve();
  await flushMicrotasks();
  assert.equal(processor.firstVisibleFrame, true);
  assert.equal(env.probeBuffers[0].doubleMapDetected, false);
  assert.equal(env.probeBuffers[0].mapCalls, 2);
  processor.destroy();
});

test('W338 reuses single 1024-byte probe buffer and destroys it cleanly on first-visible-frame (zero steady-state overhead) and on destroy()', async () => {
  const { vsHash, fsHash, packBuffer } = createW338PackBuffer();

  // 1. Destroyed cleanly right after unmap when first-visible-frame fires; zero overhead afterwards.
  {
    const env = createProbeCapableMockWebGpu({ width: 1280, height: 720 });
    let fakeNow = 1000;
    const processor = new WgpuCommandProcessor(env.device, env.ctx, {
      format: 'bgra8unorm',
      width: 1280,
      height: 720,
      now: () => fakeNow,
      wgslPackBuffer: packBuffer,
    });

    // Black frame allocates one 2048-byte (8 * 256) MAP_READ | COPY_DST buffer.
    processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: true }));
    await flushMicrotasks();
    assert.equal(env.probeBuffers.length, 1);
    const probeBuf = env.probeBuffers[0];
    assert.equal(probeBuf.size, 8 * 256);
    assert.equal(probeBuf.usage, BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST);
    assert.equal(probeBuf.destroyed, false);

    // Second black frame reuses the exact same buffer.
    fakeNow = 1300;
    processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }));
    await flushMicrotasks();
    assert.equal(env.probeBuffers.length, 1);
    assert.equal(probeBuf.destroyed, false);

    // Non-black frame fires first-visible-frame -> unmaps THEN destroys staging buffer.
    fakeNow = 1600;
    env.setPixel(640, Math.floor(720 * 0.15), 255, 255, 255, 255);
    processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }));
    await flushMicrotasks();
    assert.equal(processor.firstVisibleFrame, true);
    assert.equal(probeBuf.destroyed, true);

    const bufOps = env.bufferEvents
      .filter((e) => e.handle === probeBuf.handle)
      .map((e) => e.op);
    assert.equal(bufOps[bufOps.length - 1], 'destroy');
    assert.equal(bufOps[bufOps.length - 2], 'unmap');

    // Subsequent frames have ZERO steady-state probe overhead.
    const copiesAfterVisible = env.copyCalls.length;
    const mapsAfterVisible = probeBuf.mapCalls;
    fakeNow = 2500;
    processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }));
    fakeNow = 3500;
    processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }));
    await flushMicrotasks();
    assert.equal(env.copyCalls.length, copiesAfterVisible);
    assert.equal(env.probeBuffers.length, 1);
    assert.equal(probeBuf.mapCalls, mapsAfterVisible);
    processor.destroy();
  }

  // 2. Worker destroy() while a probe is in flight cleanly destroys the buffer with zero unhandled rejections.
  {
    const env = createProbeCapableMockWebGpu({
      width: 1280,
      height: 720,
      deferMapAsync: true,
    });
    const processor = new WgpuCommandProcessor(env.device, env.ctx, {
      format: 'bgra8unorm',
      width: 1280,
      height: 720,
      wgslPackBuffer: packBuffer,
    });
    const rejections = await collectUnhandledRejections(async () => {
      processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: true }));
      assert.equal(env.probeBuffers.length, 1);
      const probeBuf = env.probeBuffers[0];
      assert.equal(probeBuf.destroyed, false);
      assert.equal(processor.swapchainProbeInFlight, true);

      processor.destroy();
      assert.equal(probeBuf.destroyed, true);
      assert.equal(processor.swapchainProbeInFlight, false);

      // Resolving the stale deferred mapAsync after destroy() must not throw or emit first-visible-frame.
      env.pendingMaps.shift().resolve();
      await flushMicrotasks();
      assert.equal(processor.firstVisibleFrame, false);
    });
    assert.deepEqual(rejections, []);
  }
});

test('W338 configures swapchain context with RENDER_ATTACHMENT | COPY_SRC when GPUTextureUsage is available and falls back to RENDER_ATTACHMENT on error', async () => {
  const { configureCanvasContextWithCopySrc } = wgpuMod;
  const prevGpuTextureUsage = Object.getOwnPropertyDescriptor(globalThis, 'GPUTextureUsage');
  const mockUsage = Object.freeze({
    COPY_SRC: 0x01,
    COPY_DST: 0x02,
    TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08,
    RENDER_ATTACHMENT: 0x10,
  });

  try {
    Object.defineProperty(globalThis, 'GPUTextureUsage', {
      configurable: true,
      writable: true,
      value: mockUsage,
    });

    // 1. Direct helper: passes RENDER_ATTACHMENT | COPY_SRC when accepted.
    const acceptedCalls = [];
    const resOk = configureCanvasContextWithCopySrc(
      {
        configure(cfg) {
          acceptedCalls.push({ ...cfg });
        },
      },
      { device: { id: 'dev' }, format: 'bgra8unorm', alphaMode: 'opaque' }
    );
    assert.deepEqual(resOk, { configured: true, copySrcEnabled: true });
    assert.equal(acceptedCalls.length, 1);
    assert.equal(acceptedCalls[0].usage, mockUsage.RENDER_ATTACHMENT | mockUsage.COPY_SRC);

    // 2. Direct helper: falls back cleanly to RENDER_ATTACHMENT if COPY_SRC throws.
    const fallbackCalls = [];
    const resFallback = configureCanvasContextWithCopySrc(
      {
        configure(cfg) {
          fallbackCalls.push({ ...cfg });
          if (cfg.usage & mockUsage.COPY_SRC) {
            throw new Error('COPY_SRC unsupported on swapchain');
          }
        },
      },
      { device: { id: 'dev' }, format: 'bgra8unorm', alphaMode: 'opaque' }
    );
    assert.deepEqual(resFallback, { configured: true, copySrcEnabled: false });
    assert.equal(fallbackCalls.length, 2);
    assert.equal(fallbackCalls[0].usage, mockUsage.RENDER_ATTACHMENT | mockUsage.COPY_SRC);
    assert.equal(fallbackCalls[1].usage, mockUsage.RENDER_ATTACHMENT);

    // 3. Worker init (with WgpuSwapchain enabled): falls back cleanly from COPY_SRC to RENDER_ATTACHMENT
    //    and does NOT stall first-visible-frame when texture.usage lacks COPY_SRC.
    const env = createProbeCapableMockWebGpu({
      width: 640,
      height: 360,
      textureUsage: mockUsage.RENDER_ATTACHMENT, // no COPY_SRC after fallback!
    });
    const ctxConfigLog = [];
    env.ctx.configure = (cfg) => {
      ctxConfigLog.push({ ...cfg });
      if (cfg.usage && (cfg.usage & mockUsage.COPY_SRC)) {
        throw new Error('driver rejected swapchain COPY_SRC');
      }
    };
    const canvas = {
      width: 0,
      height: 0,
      getContext(kind) {
        assert.equal(kind, 'webgpu');
        return env.ctx;
      },
    };
    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'mock', architecture: 'w338-fallback' },
          async requestDevice() {
            return env.device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: true,
    });
    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 640, height: 360 });
    });

    assert.ok(ctxConfigLog.some((c) => c.usage === (mockUsage.RENDER_ATTACHMENT | mockUsage.COPY_SRC)));
    assert.equal(ctxConfigLog[ctxConfigLog.length - 1].usage, mockUsage.RENDER_ATTACHMENT);

    // Since swapchain fell back to RENDER_ATTACHMENT only, first draw must emit immediately without stalling!
    const { vsHash, fsHash, packBuffer } = createW338PackBuffer();
    handler.processor.loadWgslPack(packBuffer);
    await handler.handleMessage({
      cmdBuffer: buildRealPackDrawStream(vsHash, fsHash, { includeSetup: true }),
    });
    assert.equal(handler.processor.firstVisibleFrame, true);
    assert.equal(posted.filter((m) => m.type === 'first-visible-frame').length, 1);
    handler.processor.destroy();
  } finally {
    if (prevGpuTextureUsage) {
      Object.defineProperty(globalThis, 'GPUTextureUsage', prevGpuTextureUsage);
    } else {
      delete globalThis.GPUTextureUsage;
    }
  }
});

test('W338 preserves backwards compatibility on mock devices, supports explicit requireNonBlackFrame: false opt-out, and emits immediately on non-black Bink2 YUV frames', async () => {
  const { vsHash, fsHash, packBuffer } = createW338PackBuffer();

  // 1. Standard mock device (no copyTextureToBuffer / mapAsync / COPY_SRC): immediate first-visible-frame
  //    and zero extra createBuffer calls.
  {
    const { device, ctx, calls } = createMockWebGpu();
    const processor = new WgpuCommandProcessor(device, ctx, { wgslPackBuffer: packBuffer });
    processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: true }));
    assert.equal(processor.firstVisibleFrame, true);
    assert.equal(processor.swapchainProbeCount, 0);
    assert.equal(calls.filter((c) => c.op === 'createBuffer').length, 0);
    processor.destroy();
  }

  // 2. Real-capability device with explicit requireNonBlackFrame: false opts out of probing completely.
  {
    const env = createProbeCapableMockWebGpu({ width: 1280, height: 720 });
    const processor = new WgpuCommandProcessor(env.device, env.ctx, {
      wgslPackBuffer: packBuffer,
      requireNonBlackFrame: false,
    });
    processor.execute(buildRealPackDrawStream(vsHash, fsHash, { includeSetup: true }));
    assert.equal(processor.firstVisibleFrame, true);
    assert.equal(processor.swapchainProbeCount, 0);
    assert.equal(env.copyCalls.length, 0);
    assert.equal(env.probeBuffers.length, 0);
    processor.destroy();
  }

  // 3. Default options (real-capability device): non-black Bink2 YUV video frame emits first-visible-frame
  //    immediately and destroys any pre-allocated swapchain probe buffer cleanly.
  {
    const env = createProbeCapableMockWebGpu({ width: 16, height: 8, deferMapAsync: true });
    const origCreateEncoder = env.device.createCommandEncoder.bind(env.device);
    env.device.createCommandEncoder = () => {
      const enc = origCreateEncoder();
      const origBeginRenderPass = enc.beginRenderPass.bind(enc);
      enc.beginRenderPass = (desc) => {
        const pass = origBeginRenderPass(desc);
        pass.setViewport = () => {};
        pass.setScissorRect = () => {};
        return pass;
      };
      return enc;
    };

    const posted = [];
    const processor = new WgpuCommandProcessor(env.device, env.ctx, {
      width: 16,
      height: 8,
      wgslPackBuffer: packBuffer,
      enableSplashBridge: true,
    });
    const handler = createWgpuWorkerHandler({
      processor,
      postMessage: (m) => posted.push(m),
    });

    // Kick off a black command-buffer draw so a swapchain probe buffer is allocated and in flight.
    await handler.handleMessage({
      cmdBuffer: buildRealPackDrawStream(vsHash, fsHash, { includeSetup: true }),
    });
    assert.equal(env.probeBuffers.length, 1);
    assert.equal(env.probeBuffers[0].destroyed, false);
    assert.equal(processor.firstVisibleFrame, false);

    // Black Bink2 YUV frame (Y=16 limited-range black) -> still held!
    await handler.handleMessage({
      type: 'bink-frame',
      id: 'w338-bink-black',
      frame: {
        width: 16,
        height: 8,
        format: 'yuv420p',
        colorRange: 'limited',
        planes: {
          y: new Uint8Array(16 * 8).fill(16),
          u: new Uint8Array(8 * 4).fill(128),
          v: new Uint8Array(8 * 4).fill(128),
        },
        frameIndex: 0,
      },
    });
    assert.equal(processor.firstVisibleFrame, false);
    assert.equal(posted.filter((m) => m.type === 'first-visible-frame').length, 0);

    // Bright Bink2 YUV frame (Y=200) -> emits first-visible-frame immediately and destroys probe buffer!
    await handler.handleMessage({
      type: 'bink-frame',
      id: 'w338-bink-bright',
      frame: {
        width: 16,
        height: 8,
        format: 'yuv420p',
        colorRange: 'limited',
        planes: {
          y: new Uint8Array(16 * 8).fill(200),
          u: new Uint8Array(8 * 4).fill(128),
          v: new Uint8Array(8 * 4).fill(128),
        },
        frameIndex: 1,
      },
    });
    assert.equal(processor.firstVisibleFrame, true);
    assert.equal(env.probeBuffers[0].destroyed, true);
    const visMsgs = posted.filter((m) => m.type === 'first-visible-frame');
    assert.equal(visMsgs.length, 1);

    // Settling the stale in-flight swapchain probe after Bink2 emitted must not throw or double-emit.
    env.pendingMaps.shift().resolve();
    await flushMicrotasks();
    assert.equal(posted.filter((m) => m.type === 'first-visible-frame').length, 1);
    processor.destroy();
  }
});

// ============================================================================
// W342: Mid-frame deferred GPU destroy & 1D -> 2D LUT texture promotion
// ============================================================================

function buildCreateTexture2Packet({
  id,
  width,
  height = 1,
  depthOrArrayLayers = 1,
  mipLevelCount = 1,
  sampleCount = 1,
  usage = TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
  dimensionCode = TEXTURE_DIMENSION.DIM_1D,
  formatCode = 22, // rgba8unorm in ABI v2
}) {
  const payload = new ArrayBuffer(40);
  const view = new DataView(payload);
  view.setUint32(0, id, true);
  view.setUint32(4, width, true);
  view.setUint32(8, height, true);
  view.setUint32(12, depthOrArrayLayers, true);
  view.setUint32(16, mipLevelCount, true);
  view.setUint32(20, sampleCount, true);
  view.setUint32(24, usage, true);
  view.setUint8(28, dimensionCode);
  view.setUint8(29, formatCode);
  view.setUint8(30, 0);
  const builder = new BinaryCommandBuilder();
  builder._pushPacket(OPCODES.CREATE_TEXTURE2, new Uint8Array(payload));
  return builder.finish();
}

test('W342: mid-frame _destroyTextureIfPresent and _destroyBufferIfPresent defer .destroy() until queue.submit() + onSubmittedWorkDone()', async () => {
  const { device, ctx } = createMockWebGpu();
  const workDoneResolvers = [];
  const submitSnapshots = [];

  const trackedTextures = [];
  const trackedBuffers = [];
  const origCreateTexture = device.createTexture.bind(device);
  device.createTexture = (desc) => {
    const tex = origCreateTexture(desc);
    tex.destroyCalls = 0;
    const origDestroy = tex.destroy.bind(tex);
    tex.destroy = () => {
      tex.destroyCalls += 1;
      origDestroy();
    };
    trackedTextures.push(tex);
    return tex;
  };
  const origCreateBuffer = device.createBuffer.bind(device);
  device.createBuffer = (desc) => {
    const buf = origCreateBuffer(desc);
    buf.destroyCalls = 0;
    const origDestroy = buf.destroy.bind(buf);
    buf.destroy = () => {
      buf.destroyCalls += 1;
      origDestroy();
    };
    trackedBuffers.push(buf);
    return buf;
  };
  const origSubmit = device.queue.submit.bind(device.queue);
  device.queue.submit = (cmdBufs) => {
    submitSnapshots.push({
      destroyedTexturesAtSubmit: trackedTextures.filter((t) => t.destroyed).length,
      destroyedBuffersAtSubmit: trackedBuffers.filter((b) => b.destroyed).length,
    });
    return origSubmit(cmdBufs);
  };
  device.queue.onSubmittedWorkDone = () =>
    new Promise((resolve) => {
      workDoneResolvers.push(resolve);
    });

  // resourceRegistry is null (live Chrome default configuration).
  const processor = new WgpuCommandProcessor(device, ctx);
  assert.equal(processor.resourceRegistry, null);

  // Mid-frame: create a 256x1 LUT texture and buffer, then replace/destroy them before Present.
  const prePresent = new BinaryCommandBuilder()
    .beginFrame()
    .createTexture(10, 256, 1, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST, TEXTURE_DIMENSION.DIM_2D)
    .createBuffer(20, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, 256)
    // Recreating id 10 and destroying id 20 mid-frame must NOT synchronously call .destroy()!
    .createTexture(10, 256, 1, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST, TEXTURE_DIMENSION.DIM_2D)
    .destroyResource(20)
    .finish();

  processor.execute(prePresent);

  assert.equal(trackedTextures.length, 2);
  assert.equal(trackedBuffers.length, 1);
  // Old texture and destroyed buffer are removed from live maps, but NOT yet destroyed!
  assert.equal(processor.textures.get(10), trackedTextures[1]);
  assert.equal(processor.buffers.has(20), false);
  assert.equal(trackedTextures[0].destroyed, false);
  assert.equal(trackedBuffers[0].destroyed, false);
  assert.equal(processor._deferredGpuDestroys.length, 2);

  // Now submit the frame via Present.
  processor.execute(new BinaryCommandBuilder().present().finish());

  // At queue.submit() time, zero destroyed resources were in use!
  assert.equal(submitSnapshots.length, 1);
  assert.deepEqual(submitSnapshots[0], {
    destroyedTexturesAtSubmit: 0,
    destroyedBuffersAtSubmit: 0,
  });
  // Still awaiting onSubmittedWorkDone() -> must remain alive until GPU work completes.
  assert.equal(trackedTextures[0].destroyed, false);
  assert.equal(trackedBuffers[0].destroyed, false);
  assert.equal(processor._deferredGpuDestroys.length, 0);
  assert.equal(processor._inFlightDeferredDestroys.size, 2);
  assert.equal(workDoneResolvers.length, 1);

  // Resolve GPU work fence -> deferred destroys finalize cleanly.
  workDoneResolvers.shift()();
  await new Promise((r) => setImmediate(r));

  assert.equal(trackedTextures[0].destroyed, true);
  assert.equal(trackedTextures[0].destroyCalls, 1);
  assert.equal(trackedBuffers[0].destroyed, true);
  assert.equal(trackedBuffers[0].destroyCalls, 1);
  assert.equal(trackedTextures[1].destroyed, false);
  assert.equal(processor._inFlightDeferredDestroys.size, 0);
  assert.equal(processor.getStats().deferredGpuDestroysCompleted, 2);

  processor.destroy();
});

test('W342: deferred destroy falls back cleanly right after submit() when onSubmittedWorkDone is absent, drains on _flushPendingUploads(), and drives resourceTable beginFrame/endFrame', () => {
  const { device, ctx } = createMockWebGpu();
  delete device.queue.onSubmittedWorkDone;

  const events = [];
  const origSubmit = device.queue.submit.bind(device.queue);
  device.queue.submit = (cmdBufs) => {
    events.push({
      type: 'submit',
      pendingCount: processor._deferredGpuDestroys.length,
      anyDestroyed: processor._deferredGpuDestroys.some((r) => r.destroyed),
    });
    return origSubmit(cmdBufs);
  };

  const resourceTable = {
    beginFrame() {
      events.push({ type: 'resourceTable.beginFrame' });
    },
    endFrame(queue) {
      events.push({ type: 'resourceTable.endFrame', hasQueue: queue === device.queue });
    },
  };

  const processor = new WgpuCommandProcessor(device, ctx, { resourceTable });

  const frameStream = new BinaryCommandBuilder()
    .beginFrame()
    .createTexture(7, 128, 1, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING)
    .destroyResource(7)
    .present()
    .finish();

  processor.execute(frameStream);

  assert.equal(events[0].type, 'resourceTable.beginFrame');
  assert.equal(events[1].type, 'submit');
  assert.equal(events[1].pendingCount, 1);
  assert.equal(events[1].anyDestroyed, false);
  assert.deepEqual(events[2], { type: 'resourceTable.endFrame', hasQueue: true });
  assert.equal(processor._deferredGpuDestroys.length, 0);
  assert.equal(processor.getStats().deferredGpuDestroysCompleted, 1);

  // Also test _flushPendingUploads() path when an upload encoder is active outside beginFrame().
  processor.execute(
    new BinaryCommandBuilder()
      .createBuffer(33, BUFFER_USAGE.COPY_DST | BUFFER_USAGE.UNIFORM, 64)
      .finish()
  );
  const uploadBuf = processor.buffers.get(33);
  processor._ensurePendingEncoder();
  processor._hasPendingUploadCommands = true;
  processor._destroyBufferIfPresent(33);
  assert.equal(uploadBuf.destroyed, false);
  assert.equal(processor._deferredGpuDestroys.length, 1);

  assert.equal(processor._flushPendingUploads(false), true);
  assert.equal(uploadBuf.destroyed, true);
  assert.equal(processor._deferredGpuDestroys.length, 0);

  processor.destroy();
});

test('W342: _resetDeviceResources() and destroy() flush pending and in-flight deferred destroys without double-destroy', async () => {
  // Part A: _resetDeviceResources()
  {
    const { device, ctx } = createMockWebGpu();
    const resolvers = [];
    device.queue.onSubmittedWorkDone = () => new Promise((r) => resolvers.push(r));

    const origCreateTexture = device.createTexture.bind(device);
    device.createTexture = (desc) => {
      const t = origCreateTexture(desc);
      t.destroyCalls = 0;
      const origDestroy = t.destroy.bind(t);
      t.destroy = () => {
        t.destroyCalls += 1;
        origDestroy();
      };
      return t;
    };

    const processor = new WgpuCommandProcessor(device, ctx);

    // Frame 1: create & mid-frame destroy texture 1 -> ends up in _inFlightDeferredDestroys after Present.
    processor.execute(
      new BinaryCommandBuilder()
        .beginFrame()
        .createTexture(1, 64, 64, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING)
        .finish()
    );
    const inFlightTex = processor.textures.get(1);
    processor.execute(new BinaryCommandBuilder().destroyResource(1).present().finish());
    assert.equal(inFlightTex.destroyed, false);
    assert.equal(processor._inFlightDeferredDestroys.size, 1);

    // Frame 2: create & mid-frame destroy texture 2 (still pending, no Present yet).
    processor.execute(
      new BinaryCommandBuilder()
        .beginFrame()
        .createTexture(2, 64, 64, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING)
        .finish()
    );
    const pendingTex = processor.textures.get(2);
    processor.execute(new BinaryCommandBuilder().destroyResource(2).finish());
    assert.equal(pendingTex.destroyed, false);
    assert.equal(processor._deferredGpuDestroys.length, 1);

    // Device reset must immediately destroy both pending and in-flight deferred textures.
    processor._resetDeviceResources();
    assert.equal(inFlightTex.destroyed, true);
    assert.equal(inFlightTex.destroyCalls, 1);
    assert.equal(pendingTex.destroyed, true);
    assert.equal(pendingTex.destroyCalls, 1);
    assert.equal(processor._deferredGpuDestroys.length, 0);
    assert.equal(processor._inFlightDeferredDestroys.size, 0);

    // Resolving the stale fence afterwards must NOT call destroy() a second time.
    resolvers.shift()();
    await new Promise((r) => setImmediate(r));
    assert.equal(inFlightTex.destroyCalls, 1);

    processor.destroy();
  }

  // Part B: destroy()
  {
    const { device, ctx } = createMockWebGpu();
    const resolvers = [];
    device.queue.onSubmittedWorkDone = () => new Promise((r) => resolvers.push(r));

    const origCreateBuffer = device.createBuffer.bind(device);
    device.createBuffer = (desc) => {
      const b = origCreateBuffer(desc);
      b.destroyCalls = 0;
      const origDestroy = b.destroy.bind(b);
      b.destroy = () => {
        b.destroyCalls += 1;
        origDestroy();
      };
      return b;
    };

    const processor = new WgpuCommandProcessor(device, ctx);

    processor.execute(
      new BinaryCommandBuilder()
        .beginFrame()
        .createBuffer(10, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 128)
        .finish()
    );
    const inFlightBuf = processor.buffers.get(10);
    processor.execute(new BinaryCommandBuilder().destroyResource(10).present().finish());

    processor.execute(
      new BinaryCommandBuilder()
        .beginFrame()
        .createBuffer(11, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 128)
        .finish()
    );
    const pendingBuf = processor.buffers.get(11);
    processor.execute(new BinaryCommandBuilder().destroyResource(11).finish());

    assert.equal(inFlightBuf.destroyed, false);
    assert.equal(pendingBuf.destroyed, false);

    processor.destroy();

    assert.equal(inFlightBuf.destroyed, true);
    assert.equal(inFlightBuf.destroyCalls, 1);
    assert.equal(pendingBuf.destroyed, true);
    assert.equal(pendingBuf.destroyCalls, 1);
    assert.equal(processor._deferredGpuDestroys.length, 0);
    assert.equal(processor._inFlightDeferredDestroys.size, 0);

    resolvers.shift()();
    await new Promise((r) => setImmediate(r));
    assert.equal(inFlightBuf.destroyCalls, 1);
  }
});

test('W342: 1D LUT textures (4096x1 rgba8unorm & r32float) are promoted to WebGPU dimension 2d with valid 2d createView()', () => {
  const { device, ctx } = createMockWebGpu();
  const createdDescs = [];
  const viewDescs = [];

  device.createTexture = (desc) => {
    const copy = { ...desc, size: { ...desc.size } };
    createdDescs.push(copy);
    return {
      kind: 'GPUTexture',
      dimension: copy.dimension,
      desc: copy,
      destroyed: false,
      createView(viewDesc) {
        if (copy.dimension === '1d' && viewDesc && viewDesc.dimension === '2d') {
          throw new Error('WebGPU validation error: cannot create 2d view of 1d texture');
        }
        viewDescs.push(viewDesc ? { ...viewDesc } : null);
        return {
          kind: 'GPUTextureView',
          dimension: (viewDesc && viewDesc.dimension) || copy.dimension,
        };
      },
      destroy() {
        this.destroyed = true;
      },
    };
  };

  const processor = new WgpuCommandProcessor(device, ctx);

  // 1. ABI v1 CREATE_TEXTURE: 4096x1 rgba8unorm, dimension = 1d
  processor.execute(
    new BinaryCommandBuilder()
      .createTexture(
        101,
        4096,
        1,
        1,
        1,
        TEXTURE_FORMAT.RGBA8_UNORM,
        TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
        TEXTURE_DIMENSION.DIM_1D
      )
      .finish()
  );

  // 2. ABI v2 CREATE_TEXTURE2: 4096x1 r32float (code 16), dimension = 1d
  processor.execute(
    buildCreateTexture2Packet({
      id: 102,
      width: 4096,
      height: 1,
      depthOrArrayLayers: 1,
      dimensionCode: TEXTURE_DIMENSION.DIM_1D,
      formatCode: 16, // r32float
    })
  );

  // 3. ABI v2 CREATE_TEXTURE2: 4096x1 rgba8unorm (code 22), dimension = 1d
  processor.execute(
    buildCreateTexture2Packet({
      id: 103,
      width: 4096,
      height: 1,
      depthOrArrayLayers: 1,
      dimensionCode: TEXTURE_DIMENSION.DIM_1D,
      formatCode: 22, // rgba8unorm
    })
  );

  assert.equal(createdDescs.length, 3);
  for (const desc of createdDescs) {
    assert.equal(desc.dimension, '2d');
    assert.deepEqual(desc.size, { width: 4096, height: 1, depthOrArrayLayers: 1 });
    assert.equal(desc.mipLevelCount, 1);
  }
  assert.equal(createdDescs[0].format, 'rgba8unorm');
  assert.equal(createdDescs[1].format, 'r32float');
  assert.equal(createdDescs[2].format, 'rgba8unorm');

  assert.deepEqual(viewDescs, [
    { dimension: '2d' },
    { dimension: '2d' },
    { dimension: '2d' },
  ]);
  assert.equal(processor.textureViews.get(101).dimension, '2d');
  assert.equal(processor.textureViews.get(102).dimension, '2d');
  assert.equal(processor.textureViews.get(103).dimension, '2d');
  assert.equal(processor.getStats().promoted1DTo2DCount, 3);

  processor.destroy();
});

test('W342: promote1DTo2D: false opts out of 1D -> 2D promotion and guards createView() against { dimension: "2d" } on 1d textures', () => {
  const { device, ctx } = createMockWebGpu();
  const createdDescs = [];
  const viewDescs = [];

  device.createTexture = (desc) => {
    const copy = { ...desc, size: { ...desc.size } };
    createdDescs.push(copy);
    return {
      kind: 'GPUTexture',
      dimension: copy.dimension,
      desc: copy,
      destroyed: false,
      createView(viewDesc) {
        if (copy.dimension === '1d' && viewDesc && viewDesc.dimension === '2d') {
          throw new Error('WebGPU validation error: cannot create 2d view of 1d texture');
        }
        viewDescs.push(viewDesc ? { ...viewDesc } : null);
        return {
          kind: 'GPUTextureView',
          dimension: (viewDesc && viewDesc.dimension) || copy.dimension,
        };
      },
      destroy() {
        this.destroyed = true;
      },
    };
  };

  const processor = new WgpuCommandProcessor(device, ctx, { promote1DTo2D: false });

  processor.execute(
    new BinaryCommandBuilder()
      .createTexture(
        201,
        4096,
        1,
        1,
        1,
        TEXTURE_FORMAT.RGBA8_UNORM,
        TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
        TEXTURE_DIMENSION.DIM_1D
      )
      .finish()
  );
  processor.execute(
    buildCreateTexture2Packet({
      id: 202,
      width: 4096,
      height: 1,
      depthOrArrayLayers: 1,
      dimensionCode: TEXTURE_DIMENSION.DIM_1D,
      formatCode: 16, // r32float
    })
  );

  assert.equal(createdDescs.length, 2);
  assert.equal(createdDescs[0].dimension, '1d');
  assert.equal(createdDescs[0].format, 'rgba8unorm');
  assert.deepEqual(createdDescs[0].size, { width: 4096, height: 1, depthOrArrayLayers: 1 });
  assert.equal(createdDescs[1].dimension, '1d');
  assert.equal(createdDescs[1].format, 'r32float');
  assert.deepEqual(createdDescs[1].size, { width: 4096, height: 1, depthOrArrayLayers: 1 });

  // Neither createView call passed { dimension: '2d' }.
  assert.deepEqual(viewDescs, [null, null]);
  assert.equal(processor.textureViews.get(201).dimension, '1d');
  assert.equal(processor.textureViews.get(202).dimension, '1d');
  assert.equal(processor.getStats().promoted1DTo2DCount, 0);

  // Explicit _createSafeTextureView guard prevents passing { dimension: '2d' } to a 1d texture.
  const guardedView = processor._createSafeTextureView(processor.textures.get(201), {
    dimension: '2d',
  });
  assert.equal(guardedView.dimension, '1d');
  assert.deepEqual(viewDescs[2], { dimension: '1d' });

  processor.destroy();
});

test('W342: outside an active frame, _destroyTextureIfPresent and _destroyBufferIfPresent stay synchronous', () => {
  const { device, ctx } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx);

  processor.execute(
    new BinaryCommandBuilder()
      .createTexture(301, 16, 16, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING)
      .createBuffer(302, BUFFER_USAGE.UNIFORM, 64)
      .finish()
  );
  const tex = processor.textures.get(301);
  const buf = processor.buffers.get(302);

  assert.equal(processor._isFrameOrBatchActive(), false);
  processor._destroyTextureIfPresent(301);
  processor._destroyBufferIfPresent(302);

  assert.equal(tex.destroyed, true);
  assert.equal(buf.destroyed, true);
  assert.equal(processor._deferredGpuDestroys.length, 0);

  processor.destroy();
});

// ============================================================================
// W346: Skip redundant _scanCommandBufferFallbackProfiles double-walk &
//       enableResourceReplay: false zero-overhead fast path + init wiring
// ============================================================================

test('W346: steady-state and pack-backed command buffers skip _scanCommandBufferFallbackProfiles completely', () => {
  const { device, ctx } = createMockWebGpu();
  const vsWgsl = '@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.4, 0.8, 1.0); }';
  const vsHash = 0x3460000000000001n;
  const fsHash = 0x3460000000000002n;
  const packBuf = buildWgslPackBuffer([
    { hash: vsHash, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: vsWgsl },
    { hash: fsHash, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: fsWgsl },
  ]);

  const processor = new WgpuCommandProcessor(device, ctx, {
    wgslPackBuffer: packBuf,
  });

  let scanCalls = 0;
  const origScan = processor._scanCommandBufferFallbackProfiles.bind(processor);
  processor._scanCommandBufferFallbackProfiles = (view) => {
    scanCalls += 1;
    return origScan(view);
  };

  // Frame 1: CreateShaderModule hashes hit splash.wgsl.pack -> ZERO fallback profile scans!
  const packSetupAndDraw = new BinaryCommandBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 256)
    .createShaderModule(2, SHADER_STAGE.VERTEX, formatHashHex(vsHash))
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, formatHashHex(fsHash))
    .createRenderPipeline(4, 2, 3, 0)
    .setPipeline(4)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  const r1 = processor.executeCommandBuffer(packSetupAndDraw);
  assert.deepEqual(r1, { frameCount: 1, presents: 1 });
  assert.equal(processor.shaderModules.get(2).source, 'pack');
  assert.equal(processor.shaderModules.get(3).source, 'pack');
  assert.equal(scanCalls, 0);
  assert.equal(processor.getStats().fallbackProfileScanCount, 0);

  // Frame 2: Inline WGSL CreateShaderModule -> ZERO fallback profile scans!
  const inlineShaderDraw = new BinaryCommandBuilder()
    .beginFrame()
    .createShaderModule(12, SHADER_STAGE.VERTEX, vsWgsl)
    .createShaderModule(13, SHADER_STAGE.FRAGMENT, fsWgsl)
    .createRenderPipeline(14, 12, 13, 0)
    .setPipeline(14)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  const r2 = processor.executeCommandBuffer(inlineShaderDraw);
  assert.deepEqual(r2, { frameCount: 2, presents: 1 });
  assert.equal(scanCalls, 0);
  assert.equal(processor.getStats().fallbackProfileScanCount, 0);

  // Frames 3..12: Steady-state draw-only command buffers -> ZERO fallback profile scans!
  const drawOnly = new BinaryCommandBuilder()
    .beginFrame()
    .setPipeline(4)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  for (let i = 0; i < 10; i += 1) {
    processor.executeCommandBuffer(drawOnly);
  }
  assert.equal(processor.frameCount, 12);
  assert.equal(scanCalls, 0);
  assert.equal(processor.getStats().fallbackProfileScanCount, 0);

  processor.destroy();
});

test('W346: missing-shader CreateShaderModule lazily scans command buffer ONCE and still selects exact fallback profile', () => {
  const { device, ctx } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx);

  let scanCalls = 0;
  const origScan = processor._scanCommandBufferFallbackProfiles.bind(processor);
  processor._scanCommandBufferFallbackProfiles = (view) => {
    scanCalls += 1;
    return origScan(view);
  };

  // Batch with TWO missing shaders (VS + FS) + CBV + Draw(3, 1):
  // must lazily scan the command buffer at most ONCE and route both to fullscreen_triangle_cbv.
  const missingBatch = new BinaryCommandBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 256)
    .createShaderModule(2, SHADER_STAGE.VERTEX, '0xdead000000000346')
    .createShaderModule(3, SHADER_STAGE.FRAGMENT, '0xdead000000000347')
    .createRenderPipeline(4, 2, 3, 0)
    .createBindGroup(5, 4, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 256 },
    ])
    .setPipeline(4)
    .setBindGroup(0, 5)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  const res = processor.executeCommandBuffer(missingBatch);
  assert.deepEqual(res, { frameCount: 1, presents: 1 });
  assert.equal(scanCalls, 1, 'should lazily scan command buffer exactly once for both missing shaders');
  assert.equal(processor.getStats().fallbackProfileScanCount, 1);
  assert.equal(processor.shaderModules.get(2).fallbackProfile, 'fullscreen_triangle_cbv');
  assert.equal(processor.shaderModules.get(3).fallbackProfile, 'fullscreen_triangle_cbv');

  // Subsequent steady-state frames using that pipeline must NOT re-scan.
  const drawFrame = new BinaryCommandBuilder()
    .beginFrame()
    .setPipeline(4)
    .setBindGroup(0, 5)
    .draw(3, 1, 0, 0)
    .present()
    .finish();

  processor.executeCommandBuffer(drawFrame);
  processor.executeCommandBuffer(drawFrame);
  assert.equal(scanCalls, 1);
  assert.equal(processor.getStats().fallbackProfileScanCount, 1);

  processor.destroy();
});

test('W346: enableResourceReplay: false bypasses _recordReplayPacket and _recordShadowBufferWrite with zero allocations, while default stays true', () => {
  const { device: devDefault, ctx: ctxDefault } = createMockWebGpu();
  const procDefault = new WgpuCommandProcessor(devDefault, ctxDefault);

  const payloadBytes = new Uint8Array(64).fill(0xab);
  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 64)
    .writeBuffer(1, 0, payloadBytes)
    .createTexture(2, 16, 16, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING)
    .present()
    .finish();

  // Default behavior: resource replay remains enabled for device-loss recovery compatibility.
  assert.equal(procDefault.resourceReplayEnabled, true);
  procDefault.executeCommandBuffer(stream);
  const defaultReplayStats = procDefault.getResourceReplayStats();
  assert.equal(defaultReplayStats.enabled, true);
  assert.equal(defaultReplayStats.recordedResourceCount, 2);
  assert.ok(defaultReplayStats.recordedResourceBytes > 0);
  assert.equal(defaultReplayStats.shadowBufferCount, 1);
  assert.equal(defaultReplayStats.shadowBufferBytes, 64);
  assert.equal(procDefault.getStats().resourceReplayEnabled, true);

  // Runtime disable with { clear: true } drops previously recorded entries & stops recording.
  procDefault.setEnableResourceReplay(false, { clear: true });
  assert.equal(procDefault.resourceReplayEnabled, false);
  const clearedStats = procDefault.getResourceReplayStats();
  assert.equal(clearedStats.enabled, false);
  assert.equal(clearedStats.recordedResourceCount, 0);
  assert.equal(clearedStats.recordedResourceBytes, 0);
  assert.equal(clearedStats.shadowBufferCount, 0);
  assert.equal(clearedStats.shadowBufferBytes, 0);
  procDefault.destroy();

  // Fast-path: enableResourceReplay: false completely skips _recordReplayPacket & _recordShadowBufferWrite.
  const { device, ctx } = createMockWebGpu();
  const procFast = new WgpuCommandProcessor(device, ctx, { enableResourceReplay: false });
  assert.equal(procFast.resourceReplayEnabled, false);

  let recordPacketCalls = 0;
  let recordShadowCalls = 0;
  let recordFromCalls = 0;
  procFast._recordReplayPacket = () => {
    recordPacketCalls += 1;
  };
  procFast._recordShadowBufferWrite = () => {
    recordShadowCalls += 1;
  };
  procFast._recordReplayPacketsFrom = () => {
    recordFromCalls += 1;
  };

  procFast.executeCommandBuffer(stream);
  assert.equal(recordPacketCalls, 0, 'executeCommandBuffer must bypass _recordReplayPacket');
  assert.equal(recordShadowCalls, 0, 'executeCommandBuffer must bypass _recordShadowBufferWrite');
  assert.equal(recordFromCalls, 0, 'executeCommandBuffer must bypass _recordReplayPacketsFrom');

  // Even while recovering, enableResourceReplay: false must not scan/copy packets.
  procFast.beginDeviceRecovery();
  procFast.executeCommandBuffer(stream);
  procFast.endDeviceRecovery(true);
  assert.equal(recordFromCalls, 0);

  const fastStats = procFast.getResourceReplayStats();
  assert.equal(fastStats.enabled, false);
  assert.equal(fastStats.recordedResourceCount, 0);
  assert.equal(fastStats.recordedResourceBytes, 0);
  assert.equal(fastStats.shadowBufferCount, 0);
  assert.equal(fastStats.shadowBufferBytes, 0);
  assert.equal(procFast.getStats().resourceReplayEnabled, false);

  procFast.destroy();
});

test('W346: createWgpuWorkerHandler forwards init.enableResourceReplay: false and options.enableResourceReplay: false', async () => {
  const makeMockEnv = () => {
    const { device, ctx } = createMockWebGpu();
    device.features = new Set();
    const { canvas } = createMockOffscreenCanvas(ctx);
    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'mock', architecture: 'rdna2' },
          features: new Set(),
          limits: {},
          async requestDevice() {
            return device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };
    return { device, ctx, canvas, mockGpu };
  };

  const payloadBytes = new Uint8Array(64).fill(0x55);
  const stream = new BinaryCommandBuilder()
    .beginFrame()
    .createBuffer(1, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST, 64)
    .writeBuffer(1, 0, payloadBytes)
    .present()
    .finish();

  // 1. init message sets enableResourceReplay: false.
  {
    const { canvas, mockGpu } = makeMockEnv();
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
    });
    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({
        init: true,
        canvas,
        width: 640,
        height: 360,
        enableResourceReplay: false,
      });
    });

    assert.ok(handler.processor, 'processor initialized');
    assert.equal(handler.processor.resourceReplayEnabled, false);
    await handler.handleMessage({ cmdBuffer: stream });
    const replayStats = handler.processor.getResourceReplayStats();
    assert.equal(replayStats.enabled, false);
    assert.equal(replayStats.recordedResourceCount, 0);
    assert.equal(replayStats.shadowBufferCount, 0);
    assert.equal(replayStats.shadowBufferBytes, 0);
    handler.processor.destroy();
  }

  // 2. handler options sets enableResourceReplay: false (and subsequent init can toggle it back on).
  {
    const { canvas, mockGpu } = makeMockEnv();
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: false,
      enableResourceReplay: false,
    });
    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({
        init: true,
        canvas,
        width: 640,
        height: 360,
      });
    });

    assert.equal(handler.processor.resourceReplayEnabled, false);
    await handler.handleMessage({ cmdBuffer: stream });
    assert.equal(handler.processor.getResourceReplayStats().recordedResourceCount, 0);

    // Re-enabling via follow-up init message works cleanly.
    await handler.handleMessage({ init: true, enableResourceReplay: true });
    assert.equal(handler.processor.resourceReplayEnabled, true);
    await handler.handleMessage({ cmdBuffer: stream });
    assert.equal(handler.processor.getResourceReplayStats().recordedResourceCount, 1);
    assert.equal(handler.processor.getResourceReplayStats().shadowBufferCount, 1);
    handler.processor.destroy();
  }
});

test('W348 black frames on COPY_SRC device with WgpuSwapchain never leak firstVisibleFrame in postMessage, getStats, swapchainStats, or gpu-telemetry across resizes, clear frames, and throttle windows', async () => {
  const { vsHash, fsHash, packBuffer } = createW338PackBuffer();
  const prevGpuTextureUsage = Object.getOwnPropertyDescriptor(globalThis, 'GPUTextureUsage');
  const mockUsage = Object.freeze({
    COPY_SRC: 0x01,
    COPY_DST: 0x02,
    TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08,
    RENDER_ATTACHMENT: 0x10,
  });

  try {
    Object.defineProperty(globalThis, 'GPUTextureUsage', {
      configurable: true,
      writable: true,
      value: mockUsage,
    });

    const env = createProbeCapableMockWebGpu({
      width: 1280,
      height: 720,
      format: 'bgra8unorm',
      textureUsage: mockUsage.RENDER_ATTACHMENT | mockUsage.COPY_SRC,
    });

    // Make mock ctx dynamically track configured usage & support transient usage override.
    let activeConfiguredUsage = mockUsage.RENDER_ATTACHMENT;
    let overrideTextureUsage = null;
    env.ctx.configure = (cfg) => {
      env.ctx.configured.push({ ...cfg });
      activeConfiguredUsage =
        typeof cfg.usage === 'number' ? cfg.usage : mockUsage.RENDER_ATTACHMENT;
    };
    const origGetCurrentTexture = env.ctx.getCurrentTexture.bind(env.ctx);
    env.ctx.getCurrentTexture = () => {
      const tex = origGetCurrentTexture();
      tex.usage = overrideTextureUsage !== null ? overrideTextureUsage : activeConfiguredUsage;
      return tex;
    };

    const canvas = {
      width: 0,
      height: 0,
      getContext(kind) {
        assert.equal(kind, 'webgpu');
        return env.ctx;
      },
    };
    const mockGpu = {
      async requestAdapter() {
        return {
          info: { vendor: 'mock', architecture: 'w348-probe' },
          async requestDevice() {
            return env.device;
          },
        };
      },
      getPreferredCanvasFormat() {
        return 'bgra8unorm';
      },
    };

    let fakeNow = 1000;
    const posted = [];
    const handler = createWgpuWorkerHandler({
      postMessage: (m) => posted.push(m),
      enableSwapchain: true,
      now: () => fakeNow,
      telemetryIntervalMs: 0,
    });

    await withMockNavigatorGpu(mockGpu, async () => {
      await handler.handleMessage({ init: true, canvas, width: 1280, height: 720 });
    });

    const processor = handler.processor;
    const swapchain = handler.swapchain;
    assert.ok(processor, 'processor initialized');
    assert.ok(swapchain, 'WgpuSwapchain attached');
    assert.equal(
      swapchain.usage,
      mockUsage.RENDER_ATTACHMENT | mockUsage.COPY_SRC,
      'initial configure must enable COPY_SRC'
    );
    processor.loadWgslPack(packBuffer);

    const assertStillNotVisible = (label) => {
      const stats = processor.getStats();
      const snap = buildGpuTelemetrySnapshot(processor, { timestamp: fakeNow });
      assert.equal(processor.firstVisibleFrame, false, `${label}: processor.firstVisibleFrame`);
      assert.equal(swapchain.hasVisibleContent, false, `${label}: swapchain.hasVisibleContent`);
      assert.equal(
        swapchain.visiblePresentsTotal,
        0,
        `${label}: swapchain.visiblePresentsTotal`
      );
      assert.equal(
        swapchain.firstVisiblePresentMs,
        null,
        `${label}: swapchain.firstVisiblePresentMs`
      );
      assert.equal(stats.firstVisibleFrame, false, `${label}: getStats().firstVisibleFrame`);
      assert.equal(
        stats.swapchain.hasVisibleContent,
        false,
        `${label}: getStats().swapchain.hasVisibleContent`
      );
      assert.equal(
        stats.swapchain.visiblePresentsTotal,
        0,
        `${label}: getStats().swapchain.visiblePresentsTotal`
      );
      assert.equal(snap.firstVisibleFrame, false, `${label}: gpu-telemetry.firstVisibleFrame`);
      assert.equal(
        snap.firstVisibleFrameMs,
        null,
        `${label}: gpu-telemetry.firstVisibleFrameMs`
      );
      assert.equal(
        posted.filter((m) => m.type === 'first-visible-frame').length,
        0,
        `${label}: no first-visible-frame postMessage`
      );
      const telemetryMsgs = posted.filter((m) => m.type === 'gpu-telemetry');
      assert.ok(
        telemetryMsgs.every((m) => m.firstVisibleFrame === false),
        `${label}: every posted gpu-telemetry must have firstVisibleFrame === false`
      );
    };

    // Frame 1: real-pack draw onto 100% black (0,0,0,255) canvas -> probe suppresses, zero leaks.
    fakeNow = 1100;
    await handler.handleMessage({
      cmdBuffer: buildRealPackDrawStream(vsHash, fsHash, { includeSetup: true }),
    });
    await flushMicrotasks();
    assert.equal(processor.swapchainProbeCount, 1);
    assert.equal(processor.suppressedBlackFrames, 1);
    assertStillNotVisible('after black real-pack draw #1');

    // Resize swapchain -> COPY_SRC must be preserved across resize!
    await handler.handleMessage({ resize: true, width: 1280, height: 720 });
    await handler.handleMessage({ resize: true, width: 1280, height: 718 });
    assert.equal(
      swapchain.usage,
      mockUsage.RENDER_ATTACHMENT | mockUsage.COPY_SRC,
      'resize must preserve COPY_SRC'
    );
    await handler.handleMessage({ resize: true, width: 1280, height: 720 });
    assert.equal(
      swapchain.usage,
      mockUsage.RENDER_ATTACHMENT | mockUsage.COPY_SRC,
      'second resize must preserve COPY_SRC'
    );

    // Frame 2: real-pack black draw inside 250ms throttle window -> must NOT bypass probe!
    fakeNow = 1116;
    await handler.handleMessage({
      cmdBuffer: buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }),
    });
    await flushMicrotasks();
    assertStillNotVisible('after throttled black draw #2');

    // Frame 3: clear-only frame after black draws -> must NOT bypass probe!
    fakeNow = 1200;
    await handler.handleMessage({
      cmdBuffer: new BinaryCommandBuilder().beginFrame().clearColor(0, 0, 0, 1).present().finish(),
    });
    await flushMicrotasks();
    assertStillNotVisible('after clear-only frame #3');

    // Frame 4: transient missing-COPY_SRC texture glitch with real-pack draw -> sticky probe MUST hold!
    fakeNow = 1400;
    overrideTextureUsage = mockUsage.RENDER_ATTACHMENT;
    await handler.handleMessage({
      cmdBuffer: buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }),
    });
    await flushMicrotasks();
    overrideTextureUsage = null;
    assertStillNotVisible('after transient missing-COPY_SRC frame #4');

    // Frame 5: dim near-black frame (8,8,8,255) at default threshold 8 -> still held!
    fakeNow = 1700;
    env.clearPixels(8, 8, 8, 255);
    await handler.handleMessage({
      cmdBuffer: buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }),
    });
    await flushMicrotasks();
    assert.equal(processor.lastSampledMaxLuma, 8);
    assertStillNotVisible('after luma=8 dim frame #5');

    // Also request explicit gpu-stats while black -> must report firstVisibleFrame: false and swapchain.hasVisibleContent: false.
    await handler.handleMessage({ type: 'gpu-stats', id: 'w348-black-stats' });
    const blackStatsMsg = posted.find((m) => m.id === 'w348-black-stats');
    assert.ok(blackStatsMsg, 'gpu-stats reply received');
    assert.equal(blackStatsMsg.firstVisibleFrame, false);
    assert.equal(blackStatsMsg.stats.firstVisibleFrame, false);
    assert.equal(blackStatsMsg.stats.swapchain.hasVisibleContent, false);

    // Frame 6: Start Screen draws bright pixel (> 12) in bottom-right bar -> fires promptly & destroys staging buffer!
    fakeNow = 2000;
    env.clearPixels(0, 0, 0, 255);
    const barY = Math.floor(720 * 0.92);
    const barX = Math.floor((1280 - 64) * 0.9) + 10;
    env.setPixel(barX, barY, 0, 0, 220, 255);
    await handler.handleMessage({
      cmdBuffer: buildRealPackDrawStream(vsHash, fsHash, { includeSetup: false }),
    });
    await flushMicrotasks();

    assert.equal(processor.firstVisibleFrame, true);
    assert.equal(processor.lastSampledMaxLuma, 220);
    assert.equal(swapchain.hasVisibleContent, true);
    assert.equal(swapchain.firstVisiblePresentMs, 2000);
    assert.equal(env.probeBuffers.length, 1);
    assert.equal(env.probeBuffers[0].destroyed, true, 'staging buffer must be destroyed once visible');

    const visMsgs = posted.filter((m) => m.type === 'first-visible-frame');
    assert.equal(visMsgs.length, 1);
    assert.equal(visMsgs[0].source, 'swapchain-probe');
    assert.equal(visMsgs[0].maxLuma, 220);
    assert.equal(visMsgs[0].firstVisibleFrameMs, 2000);

    const visibleSnap = buildGpuTelemetrySnapshot(processor, { timestamp: 2050 });
    assert.equal(visibleSnap.firstVisibleFrame, true);
    assert.equal(visibleSnap.firstVisibleFrameMs, 2000);

    processor.destroy();
  } finally {
    if (prevGpuTextureUsage) {
      Object.defineProperty(globalThis, 'GPUTextureUsage', prevGpuTextureUsage);
    } else {
      delete globalThis.GPUTextureUsage;
    }
  }
});

// ---------------------------------------------------------------------------
// PR #1868 high @binding(64+) compaction + #1077 GPU/worker memory hardening
// ---------------------------------------------------------------------------

test('#1077 compacts large compiled WGSL to metadata stub preserving PR #1868 @binding(64+) and entry points', () => {
  const { device, ctx } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { lazyShaderCompile: false });

  const padding = '  let _pad = r0 + r1;\n'.repeat(180);
  const largePsWgsl = [
    'struct Cb0 { data: array<vec4<f32>, 16>, };',
    '@group(0) @binding(64) var<uniform> cb0: Cb0;',
    '@group(0) @binding(65) var tex0: texture_2d<f32>;',
    '@group(0) @binding(66) var samp0: sampler;',
    '@group(0) @binding(67) var unused_tex: texture_2d<f32>;',
    '@fragment',
    'fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {',
    '  let r0 = cb0.data[0].x;',
    '  let r1 = textureSample(tex0, samp0, uv).x;',
    padding,
    '  return vec4<f32>(r0, r1, 0.0, 1.0);',
    '}',
  ].join('\n');
  assert.ok(largePsWgsl.length > 3500);

  const stream = new BinaryCommandBuilder()
    .createShaderModule(42, SHADER_STAGE.FRAGMENT, largePsWgsl)
    .finish();
  processor.executeCommandBuffer(stream);

  const entry = processor.shaderModules.get(42);
  assert.ok(entry && entry.module);
  assert.ok(entry.code.length < 400, `expected compacted stub < 400 chars, got ${entry.code.length}`);
  assert.ok(entry.code.includes('@binding(64)'));
  assert.ok(entry.code.includes('@binding(65)'));
  assert.ok(entry.code.includes('@binding(66)'));
  assert.equal(entry.code.includes('unused_tex'), false, 'unreferenced binding dropped from stub');
  assert.ok(entry.code.includes('@fragment fn fs_main'));

  // Verify render_state_packets.js parses the exact same bindings from the compacted stub.
  const rsp = require('../web/workers/render_state_packets.js');
  const parsed = rsp.parseWgslGroup0Bindings(entry.code);
  assert.deepEqual(
    parsed.map((b) => [b.binding, b.kind]),
    [
      [64, 'uniform'],
      [65, 'texture'],
      [66, 'sampler'],
    ]
  );

  processor.destroy();
});

test('#1077 readback staging GPUBuffer is destroyed on mapAsync rejection and on processor destroy/reset', async () => {
  const { device, ctx } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx);

  let rejectedDestroyed = false;
  let pendingDestroyed = false;

  processor._recordedReadbacks = [
    {
      req: { kind: 'buffer', size: 16, slotIndex: 0 },
      layout: { skip: 0, stagingSize: 16 },
      staging: {
        mapAsync: () => Promise.reject(new Error('device lost during mapAsync')),
        destroy: () => {
          rejectedDestroyed = true;
        },
      },
    },
  ];
  processor._startRecordedReadbacks(true);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(rejectedDestroyed, true, 'staging buffer must be destroyed when mapAsync rejects');

  processor._recordedReadbacks = [
    {
      req: { kind: 'buffer', size: 16, slotIndex: 1 },
      layout: { skip: 0, stagingSize: 16 },
      staging: {
        destroy: () => {
          pendingDestroyed = true;
        },
      },
    },
  ];
  processor.destroy();
  assert.equal(pendingDestroyed, true, 'unsubmitted readback staging buffer must be destroyed on destroy()');
  assert.equal(processor._recordedReadbacks.length, 0);
});

test('#1077 _handleWriteTexture recovers from stagingPool RangeError and trimTransientMemory trims pipelines & shadow buffers', () => {
  const { device, ctx } = createMockWebGpu();
  const origCreateEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (...args) => {
    const enc = origCreateEncoder(...args);
    if (typeof enc.copyBufferToTexture !== 'function') {
      enc.copyBufferToTexture = () => {};
    }
    return enc;
  };
  let directWriteTextureCalls = 0;
  const origWriteTexture = device.queue.writeTexture.bind(device.queue);
  device.queue.writeTexture = (...args) => {
    directWriteTextureCalls += 1;
    return origWriteTexture(...args);
  };
  let flushCalled = 0;
  const mockStagingPool = {
    _destroyed: false,
    stageTextureUpload() {
      throw new RangeError('staging pool exhausted');
    },
    trimForSceneTransition() {
      return { reclaimedSlabs: 0, reclaimedBytes: 0 };
    },
    destroy() {},
  };
  let lowPriEvicted = 0;
  const mockPipelineCache = {
    evictLowPriority() {
      lowPriEvicted += 1;
      return 1;
    },
    clear() {},
  };

  const processor = new WgpuCommandProcessor(device, ctx, {
    stagingPool: mockStagingPool,
    pipelineCache: mockPipelineCache,
  });
  const origFlush = processor._flushPendingUploads.bind(processor);
  processor._flushPendingUploads = (force) => {
    flushCalled += 1;
    return origFlush(force);
  };

  // Create a 4x4 rgba8unorm texture and write 64 bytes — stagingPool throws RangeError -> falls back cleanly.
  const texPixels = new Uint8Array(64).fill(0x7f);
  const stream = new BinaryCommandBuilder()
    .createTexture(7, 4, 4, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.COPY_DST | TEXTURE_USAGE.TEXTURE_BINDING)
    .writeTexture(7, 0, 0, 0, 0, 4, 4, 1, 16, 4, texPixels)
    .finish();
  assert.doesNotThrow(() => processor.executeCommandBuffer(stream));
  assert.equal(flushCalled, 1, 'should flush pending uploads on stagingPool RangeError');
  assert.equal(directWriteTextureCalls, 1, 'should fall back to direct queue.writeTexture');

  // Seed 20 MiB of shadow buffers + oversized cached command buffer and invoke trimTransientMemory().
  processor._shadowBuffers.set(1, new Uint8Array(10 * 1024 * 1024));
  processor._shadowBuffers.set(2, new Uint8Array(10 * 1024 * 1024));
  processor._shadowBufferBytes = 20 * 1024 * 1024;
  processor._cachedCmdBuffer = new ArrayBuffer(5 * 1024 * 1024);
  processor._cachedCmdView = new DataView(processor._cachedCmdBuffer);

  processor.trimTransientMemory();
  assert.equal(lowPriEvicted, 1, 'trimTransientMemory should call pipelineCache.evictLowPriority()');
  assert.ok(processor._shadowBufferBytes <= 16 * 1024 * 1024, 'shadow buffers trimmed to <= 16 MiB');
  assert.equal(processor._cachedCmdBuffer, null, 'oversized >4 MiB _cachedCmdBuffer cleared');

  processor.destroy();
});

test('#1077 InlineWgslPackCache lazily decodes entries on demand with bounded decoded LRU cache', () => {
  const wgslA = '@vertex fn vs_a() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const wgslB = '@fragment fn fs_b() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';
  const hashA = fnv1a64('lazy-pack-a');
  const hashB = fnv1a64('lazy-pack-b');
  const packBuf = buildWgslPackBuffer([
    { hash: hashA, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: wgslA },
    { hash: hashB, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: wgslB },
  ]);

  const { device, ctx } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx);
  const packCache = processor.wgslPackCache;
  packCache.loadPackBuffer(packBuf, 'lazy-test');

  // After loadPackBuffer, decoded string cache is empty (0 strings pinned eagerly).
  assert.equal(packCache._decodedCache.size, 0);
  assert.equal(packCache._decodedBytes, 0);

  // Accessing via getWgsl or entry.wgsl decodes on demand and returns exact WGSL.
  assert.equal(packCache.getWgsl(hashA), wgslA);
  assert.equal(packCache._decodedCache.size, 1);
  assert.equal(packCache.getByHash(hashB).wgsl, wgslB);
  assert.equal(packCache._decodedCache.size, 2);

  packCache.clear();
  assert.equal(packCache._decodedCache.size, 0);
  assert.equal(packCache._decodedBytes, 0);
  processor.destroy();
});
