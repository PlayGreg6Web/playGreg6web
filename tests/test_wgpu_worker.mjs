import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const wgpuMod = require('../web/workers/wgpu_worker.js');
const {
  WgpuCommandProcessor,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
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
      };
      calls.push({ op: 'createRenderPipeline', desc, result: pipeline });
      return pipeline;
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
            setVertexBuffer(slot, buffer, offset) {
              calls.push({ op: 'setVertexBuffer', slot, buffer, offset });
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
            end() {
              calls.push({ op: 'renderPass.end' });
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
