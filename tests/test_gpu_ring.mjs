import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ringMod = require('../web/workers/gpu_ring.js');
const wgpuMod = require('../web/workers/wgpu_worker.js');

const {
  GpuRingBuffer,
  GPU_RING_MAGIC,
  GPU_RING_VERSION,
  GPU_RING_HEADER_BYTES,
  HEADER_INDEX,
} = ringMod;

const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
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

  createRenderPipeline(
    id,
    vsShaderId,
    fsShaderId,
    vertexStride,
    topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST,
    flags = 0
  ) {
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
    return out;
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

test('UMD export and 64-byte GpuRingHeader binary layout match runtime/gpu/gpu_ring.h', () => {
  assert.equal(globalThis.PlayGta6GpuRing.GpuRingBuffer, GpuRingBuffer);
  assert.equal(GPU_RING_MAGIC, 0x47505552);
  assert.equal(GPU_RING_VERSION, 1);
  assert.equal(GPU_RING_HEADER_BYTES, 64);

  const ring = GpuRingBuffer.create(128);
  assert.equal(ring.sab.byteLength, 64 + 128);

  const raw = new DataView(ring.sab);
  assert.equal(raw.getUint32(0, true), 0x47505552); // magic
  assert.equal(raw.getUint32(4, true), 1);          // version
  assert.equal(raw.getUint32(8, true), 128);        // capacity_bytes
  assert.equal(raw.getUint32(12, true), 0);         // reserved0
  assert.equal(raw.getUint32(16, true), 0);         // write_pos
  assert.equal(raw.getUint32(20, true), 0);         // read_pos
  assert.equal(raw.getUint32(24, true), 0);         // dropped_packets
  assert.equal(raw.getUint32(28, true), 0);         // notify_seq
  for (let off = 32; off < 64; off += 4) {
    assert.equal(raw.getUint32(off, true), 0);
  }

  assert.throws(() => GpuRingBuffer.create(96), RangeError);
  assert.throws(() => GpuRingBuffer.create(0), RangeError);
  assert.throws(() => GpuRingBuffer.attach(new SharedArrayBuffer(128)), /Invalid GpuRingHeader magic/);
});

test('GpuRingBuffer handles wrap-around, zero-copy contiguous drain, and full-buffer backpressure', () => {
  const producer = GpuRingBuffer.create(64);
  const consumer = GpuRingBuffer.attach(producer.sab);

  assert.equal(producer.availableBytes(), 0);
  assert.equal(producer.freeBytes(), 64);

  // Write 48 bytes -> contiguous drain should be zero-copy directly on SAB.
  const first = Uint8Array.from({ length: 48 }, (_, i) => (i + 1) & 0xff);
  assert.equal(producer.writeBytes(first), true);
  assert.equal(producer.availableBytes(), 48);
  assert.equal(producer.freeBytes(), 16);
  assert.equal(producer.notifySeq, 1);

  // Backpressure: 17 bytes does not fit into 16 free bytes.
  const rejected = new Uint8Array(17).fill(0xee);
  assert.equal(producer.writeBytes(rejected), false);
  assert.equal(producer.droppedPackets, 1);
  assert.equal(producer.availableBytes(), 48);
  assert.equal(producer.freeBytes(), 16);

  // Oversized write (> capacity) returns false without corrupting ring.
  assert.equal(producer.writeBytes(new Uint8Array(65)), false);
  assert.equal(producer.droppedPackets, 2);

  let drainedFirst = null;
  let wasZeroCopy = false;
  const n1 = consumer.drainTo((bytes) => {
    wasZeroCopy = bytes.buffer === producer.sab;
    drainedFirst = bytes.slice();
  });
  assert.equal(n1, 48);
  assert.equal(wasZeroCopy, true);
  assert.deepEqual(drainedFirst, first);
  assert.equal(consumer.availableBytes(), 0);
  assert.equal(producer.freeBytes(), 64);

  // Write 40 bytes at offset 48 -> wraps across [48..64) and [0..24).
  const wrapped = Uint8Array.from({ length: 40 }, (_, i) => (0xa0 + i) & 0xff);
  const tail = Uint8Array.from({ length: 24 }, (_, i) => (0x50 + i) & 0xff);
  assert.equal(producer.writeBytes(wrapped), true);
  assert.equal(producer.writeBytes(tail), true);
  assert.equal(producer.availableBytes(), 64);
  assert.equal(producer.freeBytes(), 0);

  // Full ring rejects even 1 byte and keeps existing data intact.
  assert.equal(producer.writeBytes(Uint8Array.of(0xff)), false);
  assert.equal(producer.droppedPackets, 3);

  let drainedWrapped = null;
  const n2 = consumer.drainTo((bytes) => {
    drainedWrapped = bytes.slice();
  });
  assert.equal(n2, 64);
  assert.deepEqual(drainedWrapped.subarray(0, 40), wrapped);
  assert.deepEqual(drainedWrapped.subarray(40, 64), tail);
  assert.equal(consumer.availableBytes(), 0);
  assert.equal(producer.freeBytes(), 64);

  // Verify 32-bit monotonic counter wraparound across 0xFFFFFFFF -> 0.
  const headerU32 = new Uint32Array(producer.sab, 0, 16);
  Atomics.store(headerU32, HEADER_INDEX.WRITE_POS, 0xfffffff0);
  Atomics.store(headerU32, HEADER_INDEX.READ_POS, 0xfffffff0);

  const wrap32 = Uint8Array.from({ length: 32 }, (_, i) => (i ^ 0x5a) & 0xff);
  assert.equal(producer.writeBytes(wrap32), true);
  assert.equal(producer.writePos, 0x00000010);
  assert.equal(consumer.availableBytes(), 32);

  let wrap32Out = null;
  const n3 = consumer.drainTo((bytes) => {
    wrap32Out = bytes.slice();
  });
  assert.equal(n3, 32);
  assert.equal(consumer.readPos, 0x00000010);
  assert.deepEqual(wrap32Out, wrap32);
});

test('Zero-corruption streaming of wrapped GpuPacketHeader commands into WgpuCommandProcessor', () => {
  const { device, ctx, calls } = createMockWebGpu();
  const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });

  // Small 512-byte ring forces repeated wrap-arounds across multi-packet frames.
  const producer = GpuRingBuffer.create(512);
  processor.attachRing(producer.sab);

  const vertexFloats = new Float32Array([
    0.0, 0.5, 0.0,
    -0.5, -0.5, 0.0,
    0.5, -0.5, 0.0,
  ]);
  const vertexBytes = new Uint8Array(vertexFloats.buffer);
  const vsWgsl = '@vertex fn vs_main(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 1.0); }';
  const fsWgsl = '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.6, 1.0, 1.0); }';

  const frameStream = new BinaryCommandBuilder()
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

  assert.ok(frameStream.byteLength > 300 && frameStream.byteLength < 512);

  // Stream 10 frames so every frame starts at a different ring offset and wraps across 512B.
  for (let frameIdx = 1; frameIdx <= 10; frameIdx++) {
    calls.length = 0;
    assert.equal(producer.writeBytes(frameStream), true);

    // Backpressure: a second full frame cannot fit in the 512B ring while the first is queued.
    assert.equal(producer.writeBytes(frameStream), false);

    const res = processor.drainRing();
    assert.deepEqual(res, {
      frameCount: frameIdx,
      presents: 1,
      bytesDrained: frameStream.byteLength,
    });
    assert.equal(producer.availableBytes(), 0);

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
      ]
    );
    assert.deepEqual(calls[1].data, vertexBytes);
    assert.equal(calls[2].desc.code, vsWgsl);
    assert.equal(calls[3].desc.code, fsWgsl);
    assert.deepEqual(calls[5].desc.colorAttachments[0].clearValue, {
      r: 0.25,
      g: 0.5,
      b: 0.75,
      a: 1.0,
    });
    assert.equal(calls[8].vertexCount, 3);
  }

  assert.equal(producer.droppedPackets, 10);
});

test('wgpu_worker message handler supports attachRing, drainRing, injected GpuRingBuffer, and legacy cmdBuffer', async () => {
  const { device, ctx, calls } = createMockWebGpu();
  const posted = [];
  let injectedConstructed = 0;

  class InjectedGpuRing extends GpuRingBuffer {
    constructor(sab, opts) {
      super(sab, opts);
      injectedConstructed += 1;
    }
  }

  const producer = GpuRingBuffer.create(256);
  const worker = createWgpuWorkerHandler({
    postMessage: (msg) => posted.push(msg),
    GpuRingBuffer: InjectedGpuRing,
    initGpu: async () => new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' }),
  });

  // Attach ring BEFORE init to verify pending ring binding carries over once GPU initializes.
  await worker.handleMessage({ attachRing: true, ringSab: producer.sab });
  assert.equal(injectedConstructed, 1);
  assert.deepEqual(posted.shift(), { ringAttached: true, capacityBytes: 256 });

  await worker.handleMessage({ init: true, canvas: { width: 0, height: 0 }, width: 640, height: 360 });
  assert.ok(worker.processor);
  assert.equal(worker.processor.ring, worker.ring);

  // Stream a ring frame that wraps around the 256B ring.
  const padStream = new BinaryCommandBuilder().beginFrame().clearColor(0.1, 0.1, 0.1, 1.0).finish();
  for (let i = 0; i < 6; i++) {
    assert.equal(producer.writeBytes(padStream), true);
    await worker.handleMessage({ drainRing: true });
  }
  assert.equal(posted.length, 0); // no Present yet

  const presentStream = new BinaryCommandBuilder()
    .beginFrame()
    .clearColor(0.9, 0.2, 0.1, 1.0)
    .present()
    .finish();
  assert.equal(producer.writeBytes(presentStream), true);
  await worker.handleMessage({ drainRing: true });

  assert.deepEqual(posted.shift(), { frame: 1 });
  const frame1Telemetry = posted.shift();
  assert.equal(frame1Telemetry.type, 'gpu-telemetry');
  assert.equal(frame1Telemetry.frameStats.framesPresented, 1);
  assert.equal(worker.processor.frameCount, 1);
  assert.ok(calls.some((c) => c.op === 'submit'));

  // Backward-compatible { cmdBuffer } path still works alongside attached ring.
  const legacyStream = new BinaryCommandBuilder()
    .beginFrame()
    .clearColor(0.0, 1.0, 0.0, 1.0)
    .present()
    .finish();
  await worker.handleMessage({ cmdBuffer: legacyStream.buffer });
  assert.deepEqual(posted.shift(), { frame: 2 });
  assert.equal(worker.processor.frameCount, 2);
});
