import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  OPCODES,
  SHADER_STAGE,
  BUFFER_USAGE,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  BIND_RESOURCE_TYPE,
  INDEX_FORMAT,
  SWAPCHAIN_PROBE_STRIPS,
  SWAPCHAIN_PROBE_BUFFER_SIZE,
} = require('../web/workers/wgpu_worker.js');
const { GpuRingBuffer } = require('../web/workers/gpu_ring.js');
const { installRenderStatePackets } = require('../web/workers/render_state_packets.js');
const { ShaderFallbackRouter } = require('../web/workers/shader_fallback_router.js');
import { WgslPackCache, buildWgslPackBuffer } from '../web/workers/wgsl_pack_loader.js';

const enc = new TextEncoder();

class PacketBuilder {
  constructor() {
    this.chunks = [];
    this.totalBytes = 0;
  }

  _push(opcode, payloadBuf, flags = 0) {
    const payload = payloadBuf ? new Uint8Array(payloadBuf) : new Uint8Array(0);
    const pkt = new Uint8Array(8 + payload.byteLength);
    const view = new DataView(pkt.buffer);
    view.setUint16(0, opcode, true);
    view.setUint16(2, flags, true);
    view.setUint32(4, payload.byteLength, true);
    if (payload.byteLength > 0) pkt.set(payload, 8);
    this.chunks.push(pkt);
    this.totalBytes += pkt.byteLength;
    return this;
  }

  beginFrame() {
    return this._push(OPCODES.BEGIN_FRAME);
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setFloat32(0, r, true);
    v.setFloat32(4, g, true);
    v.setFloat32(8, b, true);
    v.setFloat32(12, a, true);
    return this._push(OPCODES.CLEAR_COLOR, buf);
  }

  createBuffer(id, usage, byteSize) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, usage, true);
    v.setBigUint64(8, BigInt(byteSize), true);
    return this._push(OPCODES.CREATE_BUFFER, buf);
  }

  createTexture(id, width, height, format = TEXTURE_FORMAT.RGBA8_UNORM, usage = TEXTURE_USAGE.TEXTURE_BINDING) {
    const buf = new ArrayBuffer(32);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, width, true);
    v.setUint32(8, height, true);
    v.setUint32(12, 1, true);
    v.setUint32(16, 1, true);
    v.setUint32(20, format, true);
    v.setUint32(24, usage, true);
    v.setUint32(28, TEXTURE_DIMENSION.DIM_2D, true);
    return this._push(OPCODES.CREATE_TEXTURE, buf);
  }

  writeTexture(
    textureId,
    mipLevel,
    originX,
    originY,
    originZ,
    width,
    height,
    depthOrArrayLayers,
    bytesPerRow,
    rowsPerImage,
    data
  ) {
    const src = data instanceof Uint8Array ? data : new Uint8Array(data || 0);
    const buf = new ArrayBuffer(48 + src.byteLength);
    const v = new DataView(buf);
    v.setUint32(0, textureId, true);
    v.setUint32(4, mipLevel, true);
    v.setUint32(8, originX, true);
    v.setUint32(12, originY, true);
    v.setUint32(16, originZ, true);
    v.setUint32(20, width, true);
    v.setUint32(24, height, true);
    v.setUint32(28, depthOrArrayLayers, true);
    v.setUint32(32, bytesPerRow, true);
    v.setUint32(36, rowsPerImage, true);
    v.setUint32(40, src.byteLength, true);
    v.setUint32(44, 0, true);
    if (src.byteLength > 0) new Uint8Array(buf, 48).set(src);
    return this._push(OPCODES.WRITE_TEXTURE, buf);
  }

  createShaderModule(id, stage, wgsl) {
    const src = enc.encode(wgsl);
    const buf = new ArrayBuffer(16 + src.byteLength);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, stage, true);
    v.setUint32(8, src.byteLength, true);
    v.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(src);
    return this._push(OPCODES.CREATE_SHADER_MODULE, buf);
  }

  createRenderPipeline(id, vsId, fsId, vertexStride = 0, topology = 0) {
    const buf = new ArrayBuffer(24);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, vsId, true);
    v.setUint32(8, fsId, true);
    v.setUint32(12, vertexStride, true);
    v.setUint32(16, topology, true);
    v.setUint32(20, 0, true);
    return this._push(OPCODES.CREATE_RENDER_PIPELINE, buf);
  }

  createComputePipeline(id, csId) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, csId, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
    return this._push(OPCODES.CREATE_COMPUTE_PIPELINE, buf);
  }

  createBindGroup(id, pipelineId, groupIndex, entries) {
    const buf = new ArrayBuffer(16 + entries.length * 32);
    const v = new DataView(buf);
    v.setUint32(0, id, true);
    v.setUint32(4, pipelineId, true);
    v.setUint32(8, groupIndex, true);
    v.setUint32(12, entries.length, true);
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i];
      const off = 16 + i * 32;
      v.setUint32(off, e.binding, true);
      v.setUint32(off + 4, e.resourceType, true);
      v.setUint32(off + 8, e.resourceId, true);
      v.setUint32(off + 12, 0, true);
      v.setBigUint64(off + 16, BigInt(e.offset || 0), true);
      v.setBigUint64(off + 24, BigInt(e.size || 0), true);
    }
    return this._push(OPCODES.CREATE_BIND_GROUP, buf);
  }

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    return this._push(OPCODES.SET_PIPELINE, buf);
  }

  setVertexBuffer(slot, bufferId, offset = 0) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, slot, true);
    v.setUint32(4, bufferId, true);
    v.setBigUint64(8, BigInt(offset), true);
    return this._push(OPCODES.SET_VERTEX_BUFFER, buf);
  }

  setIndexBuffer(bufferId, indexFormat = INDEX_FORMAT.UINT16, offset = 0) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, bufferId, true);
    v.setUint32(4, indexFormat, true);
    v.setBigUint64(8, BigInt(offset), true);
    return this._push(OPCODES.SET_INDEX_BUFFER, buf);
  }

  setBindGroup(groupIndex, bindGroupId) {
    const buf = new ArrayBuffer(8);
    const v = new DataView(buf);
    v.setUint32(0, groupIndex, true);
    v.setUint32(4, bindGroupId, true);
    return this._push(OPCODES.SET_BIND_GROUP, buf);
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, vertexCount, true);
    v.setUint32(4, instanceCount, true);
    v.setUint32(8, firstVertex, true);
    v.setUint32(12, firstInstance, true);
    return this._push(OPCODES.DRAW, buf);
  }

  drawIndexed(indexCount, instanceCount = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(20);
    const v = new DataView(buf);
    v.setUint32(0, indexCount, true);
    v.setUint32(4, instanceCount, true);
    v.setUint32(8, firstIndex, true);
    v.setInt32(12, baseVertex, true);
    v.setUint32(16, firstInstance, true);
    return this._push(OPCODES.DRAW_INDEXED, buf);
  }

  setComputePipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    return this._push(OPCODES.SET_COMPUTE_PIPELINE, buf);
  }

  dispatchCompute(x, y, z) {
    const buf = new ArrayBuffer(12);
    const v = new DataView(buf);
    v.setUint32(0, x, true);
    v.setUint32(4, y, true);
    v.setUint32(8, z, true);
    return this._push(OPCODES.DISPATCH_COMPUTE, buf);
  }

  destroyResource(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this._push(OPCODES.DESTROY_RESOURCE, buf);
  }

  present() {
    return this._push(OPCODES.PRESENT);
  }

  beginRenderPassSwapchain(clearRgba = [0, 0, 0, 1]) {
    // ABI v2 BeginRenderPass: 1 color attachment targeting SWAPCHAIN_TEXTURE_ID (0xFFFFFFFF)
    const buf = new ArrayBuffer(4 + 32 + 32);
    const v = new DataView(buf);
    v.setUint32(0, 1, true); // colorCount = 1, hasDepth = 0
    v.setUint32(4, 0xffffffff, true); // SWAPCHAIN_TEXTURE_ID
    v.setUint32(8, 2, true); // loadOp = clear
    v.setUint32(12, 1, true); // storeOp = store
    v.setFloat32(16, clearRgba[0], true);
    v.setFloat32(20, clearRgba[1], true);
    v.setFloat32(24, clearRgba[2], true);
    v.setFloat32(28, clearRgba[3], true);
    return this._push(OPCODES.BEGIN_RENDER_PASS, buf);
  }

  endRenderPass() {
    return this._push(OPCODES.END_RENDER_PASS);
  }

  setBindGroup2(groupIndex, bindGroupId) {
    const buf = new ArrayBuffer(16);
    const v = new DataView(buf);
    v.setUint32(0, groupIndex, true);
    v.setUint32(4, bindGroupId, true);
    v.setUint32(8, 0, true);
    v.setUint32(12, 0, true);
    return this._push(OPCODES.SET_BIND_GROUP2, buf);
  }

  finish() {
    const out = new Uint8Array(this.totalBytes);
    let off = 0;
    for (const c of this.chunks) {
      out.set(c, off);
      off += c.byteLength;
    }
    return out;
  }
}

function createFastMockWebGpu(opts = {}) {
  const counts = {
    beginRenderPass: 0,
    beginComputePass: 0,
    setPipeline: 0,
    setComputePipeline: 0,
    setVertexBuffer: 0,
    setIndexBuffer: 0,
    setBindGroup: 0,
    draw: 0,
    drawIndexed: 0,
    dispatchWorkgroups: 0,
    submits: 0,
    copyTextureToBuffer: 0,
  };
  const clearValues = [];
  const destroyedResources = [];
  const pendingFences = [];
  let nextResId = 1;

  const renderPass = {
    setPipeline() { counts.setPipeline += 1; },
    setVertexBuffer() { counts.setVertexBuffer += 1; },
    setIndexBuffer() { counts.setIndexBuffer += 1; },
    setBindGroup() { counts.setBindGroup += 1; },
    draw() { counts.draw += 1; },
    drawIndexed() { counts.drawIndexed += 1; },
    end() {},
  };

  const computePass = {
    setPipeline() { counts.setComputePipeline += 1; },
    setBindGroup() { counts.setBindGroup += 1; },
    dispatchWorkgroups() { counts.dispatchWorkgroups += 1; },
    end() {},
  };

  const device = {
    createBuffer(desc) {
      const id = nextResId++;
      const buf = {
        _mockId: id,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
          destroyedResources.push({ kind: 'buffer', id });
        },
        mapAsync() {
          return Promise.resolve();
        },
        getMappedRange() {
          const out = new Uint8Array(desc.size);
          if (opts.probeVisibleAfterCopies && counts.copyTextureToBuffer >= opts.probeVisibleAfterCopies) {
            out[0] = 200;
            out[1] = 200;
            out[2] = 200;
            out[3] = 255;
          }
          return out.buffer;
        },
        unmap() {},
      };
      return buf;
    },
    createTexture(desc) {
      const id = nextResId++;
      return {
        _mockId: id,
        width: desc.size.width,
        height: desc.size.height,
        format: desc.format,
        usage: desc.usage,
        destroyed: false,
        createView() {
          return { _viewFor: id };
        },
        destroy() {
          this.destroyed = true;
          destroyedResources.push({ kind: 'texture', id });
        },
      };
    },
    createShaderModule(desc) {
      return { code: desc.code };
    },
    createRenderPipeline(desc) {
      return {
        desc,
        getBindGroupLayout(idx) {
          return { _bglIdx: idx };
        },
      };
    },
    createComputePipeline(desc) {
      return {
        desc,
        getBindGroupLayout(idx) {
          return { _bglIdx: idx };
        },
      };
    },
    createBindGroup(desc) {
      return { desc };
    },
    createCommandEncoder() {
      return {
        beginRenderPass(desc) {
          counts.beginRenderPass += 1;
          if (opts.recordClearValues && desc.colorAttachments && desc.colorAttachments[0]) {
            clearValues.push(desc.colorAttachments[0].clearValue);
          }
          return renderPass;
        },
        beginComputePass() {
          counts.beginComputePass += 1;
          return computePass;
        },
        copyTextureToBuffer() {
          counts.copyTextureToBuffer += 1;
        },
        finish() {
          return { _cmdBuf: true };
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {
        counts.submits += 1;
      },
      onSubmittedWorkDone: opts.manualFence
        ? () =>
            new Promise((resolve) => {
              pendingFences.push(resolve);
            })
        : undefined,
    },
  };

  const swapchainTexture = {
    width: 1280,
    height: 720,
    usage: 0x11, // RENDER_ATTACHMENT | COPY_SRC
    createView() {
      return { _swapView: true };
    },
  };

  const ctx = {
    getCurrentTexture() {
      return swapchainTexture;
    },
  };

  return { device, ctx, counts, clearValues, destroyedResources, pendingFences };
}

test('W359: 120 frames x 50 draws reuses pooled pass commands, binding maps/slots, immutable clearColor, frozen gpuAdapter, and preserves W346 fast-path', () => {
  const mock = createFastMockWebGpu({ recordClearValues: true });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    width: 1280,
    height: 720,
    enableResourceReplay: false,
    requireNonBlackFrame: false,
    gpuAdapter: {
      vendor: 'amd',
      architecture: 'rdna2',
      device: 'series-x',
      description: 'mock-rdna2',
    },
  });

  // Frozen cached gpuAdapter snapshot returns identical object reference with zero spread churn.
  const adapterSnap1 = processor.gpuAdapter;
  const adapterSnap2 = processor.gpuAdapter;
  assert.equal(adapterSnap1, adapterSnap2);
  assert.equal(Object.isFrozen(adapterSnap1), true);
  assert.equal(adapterSnap1.vendor, 'amd');

  // Setup resources once before steady-state loop.
  const setup = new PacketBuilder()
    .createBuffer(1, BUFFER_USAGE.VERTEX, 4096)
    .createBuffer(2, BUFFER_USAGE.INDEX, 2048)
    .createBuffer(3, BUFFER_USAGE.UNIFORM, 256)
    .createShaderModule(10, SHADER_STAGE.VERTEX, '@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0.0); }')
    .createShaderModule(11, SHADER_STAGE.FRAGMENT, '@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }')
    .createShaderModule(12, SHADER_STAGE.COMPUTE, '@compute @workgroup_size(1) fn cs_main() {}')
    .createRenderPipeline(20, 10, 11, 24, 0)
    .createComputePipeline(21, 12)
    .createBindGroup(30, 20, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 3, offset: 0, size: 256 },
    ])
    .finish();
  processor.executeCommandBuffer(setup);

  // Build one steady-state frame with 50 draws (25 non-indexed + 25 indexed) + 1 compute dispatch.
  const buildFrame = (r, g, b, a) => {
    const fb = new PacketBuilder().beginFrame().clearColor(r, g, b, a);
    fb.setPipeline(20).setBindGroup(0, 30).setVertexBuffer(0, 1, 0).setIndexBuffer(2, INDEX_FORMAT.UINT32, 0);
    for (let d = 0; d < 25; d += 1) {
      fb.setVertexBuffer(0, 1, d * 24);
      fb.draw(3, 1, 0, d);
      fb.drawIndexed(6, 1, 0, d, 0);
    }
    fb.setComputePipeline(21).dispatchCompute(4, 1, 1);
    fb.present();
    return fb.finish();
  };

  const steadyFrameBytes = buildFrame(0.125, 0.25, 0.5, 1.0);
  const altColorFrameBytes = buildFrame(0.75, 0.5, 0.25, 1.0);

  const ring = GpuRingBuffer.create(64 * 1024);
  processor.attachRing(ring);

  const initialPassCommandsRef = processor._passCommands;
  const initialBoundBgMapRef = processor._boundBindGroups;
  const initialBoundVbMapRef = processor._boundVertexBuffers;

  let poolSizeAfterFrame1 = 0;
  let firstBgSlotRef = null;
  let firstVbSlotRef = null;

  const t0 = performance.now();
  const FRAMES = 120;
  for (let f = 0; f < FRAMES; f += 1) {
    const frameBytes = f === 60 ? altColorFrameBytes : steadyFrameBytes;
    assert.equal(ring.writeBytes(frameBytes), true);
    const res = processor.drainRing({ reuseResult: true });
    assert.equal(res.presents, 1);
    assert.equal(res.frameCount, f + 1);

    if (f === 0) {
      poolSizeAfterFrame1 = processor._passCommandPool.length;
      firstBgSlotRef = processor._boundBindGroupSlots.get(0);
      firstVbSlotRef = processor._boundVertexBufferSlots.get(0);
      assert.ok(poolSizeAfterFrame1 >= 80, 'pool should accommodate all pass commands of frame 1');
      assert.ok(firstBgSlotRef && firstVbSlotRef);
    } else {
      assert.equal(
        processor._passCommandPool.length,
        poolSizeAfterFrame1,
        `pool must not allocate new command slots on steady-state frame ${f + 1}`
      );
      assert.equal(processor._passCommands, initialPassCommandsRef);
      assert.equal(processor._boundBindGroups, initialBoundBgMapRef);
      assert.equal(processor._boundVertexBuffers, initialBoundVbMapRef);
      assert.equal(processor._boundBindGroupSlots.get(0), firstBgSlotRef);
      assert.equal(processor._boundVertexBufferSlots.get(0), firstVbSlotRef);
    }
  }
  const elapsedMs = performance.now() - t0;

  assert.equal(processor.frameCount, FRAMES);
  assert.equal(mock.counts.draw, FRAMES * 25);
  assert.equal(mock.counts.drawIndexed, FRAMES * 25);
  assert.equal(mock.counts.dispatchWorkgroups, FRAMES);
  assert.equal(mock.counts.setPipeline, FRAMES);
  // #2177: d=0 repeats setVertexBuffer(0, 1, 0) immediately after line 488 -> 1 redundant call elided per frame.
  assert.equal(mock.counts.setVertexBuffer, FRAMES * 25);
  assert.equal(processor.getStats().redundantStateCallsSkipped, FRAMES);
  assert.equal(mock.counts.setIndexBuffer, FRAMES);
  assert.equal(mock.counts.setBindGroup, FRAMES);
  assert.equal(mock.counts.submits, FRAMES);

  // Verify immutable clearColor snapshots: frames 0..59 share one frozen object, frame 60 has alt color,
  // and frame 60's clearValue was NOT mutated by subsequent frames or _resetFrameState().
  assert.equal(mock.clearValues.length, FRAMES);
  assert.equal(mock.clearValues[0], mock.clearValues[59]);
  assert.equal(Object.isFrozen(mock.clearValues[0]), true);
  assert.deepEqual(mock.clearValues[60], { r: 0.75, g: 0.5, b: 0.25, a: 1.0 });
  assert.deepEqual(mock.clearValues[61], { r: 0.125, g: 0.25, b: 0.5, a: 1.0 });

  // Verify pooled slots release GPU object references after frame reset (no retained leaks).
  assert.equal(processor._passCommands.length, 0);
  assert.equal(processor._passCommandPoolUsed, 0);
  for (const slot of processor._passCommandPool) {
    assert.equal(slot.pipeline, null);
    assert.equal(slot.bindGroup, null);
    assert.equal(slot._wgpuMeta, null);
  }
  assert.equal(processor._boundBindGroups.size, 0);
  assert.equal(processor._boundVertexBuffers.size, 0);
  assert.equal(firstBgSlotRef.bindGroup, null);
  assert.equal(firstVbSlotRef.buffer, null);

  // W346 invariants intact: zero fallback lookahead scans and zero replay entries when disabled.
  assert.equal(processor._fallbackProfileScanCount, 0);
  assert.equal(processor.resourceReplayEnabled, false);
  assert.equal(processor.getResourceReplayStats().recordedResourceCount, 0);

  // Throughput sanity check: 120 frames x 50 draws should execute well under 2 seconds (60 FPS budget).
  assert.ok(elapsedMs < 2000, `120x50 draw hot loop took ${elapsedMs.toFixed(1)}ms (> 2000ms)`);
});

test('W359: GpuRingBuffer wrap-around drains reuse persistent wrapScratch + cached DataView with directView zero-subarray mode', () => {
  // Small 1024B ring guarantees wrap-around on nearly every frame.
  const producer = GpuRingBuffer.create(1024);
  const consumer = GpuRingBuffer.attach(producer.sab);

  const initialWrapBuf = consumer._wrapScratch.buffer;
  const initialWrapView = consumer.wrapScratchView;
  const initialDataView = consumer.dataView;
  assert.ok(initialWrapView instanceof DataView);
  assert.ok(initialDataView instanceof DataView);

  const frameStream = new PacketBuilder()
    .beginFrame()
    .clearColor(0.2, 0.4, 0.6, 1.0)
    .draw(3, 1, 0, 0)
    .draw(6, 1, 0, 1)
    .present()
    .finish();

  let wrapCount = 0;
  let nonWrapCount = 0;
  let firstResultRef = null;
  let firstSpanRef = null;

  for (let i = 0; i < 64; i += 1) {
    assert.equal(producer.writeBytes(frameStream), true);
    const res = consumer.drainBatchZeroAlloc(
      (bytesView, byteOffset, byteLength, span) => {
        if (firstSpanRef === null) {
          firstSpanRef = span;
        } else {
          assert.equal(span, firstSpanRef, 'span descriptor must be reused');
        }
        assert.equal(byteLength, frameStream.byteLength);
        if (span.wrapped) {
          wrapCount += 1;
          assert.equal(bytesView, consumer._wrapScratch, 'directView wrap must pass _wrapScratch directly');
          assert.equal(bytesView.buffer, initialWrapBuf);
          assert.equal(byteOffset, 0);
          assert.equal(span.dataView, initialWrapView, 'wrap span must reuse cached wrapScratchView');
          assert.equal(span.viewOffset, 0);
        } else {
          nonWrapCount += 1;
          assert.equal(bytesView, consumer._data, 'directView non-wrap must pass _data directly');
          assert.equal(bytesView.buffer, producer.sab);
          assert.equal(span.dataView, initialDataView, 'non-wrap span must reuse cached dataView');
          assert.equal(span.viewOffset, byteOffset);
        }
        // Verify exact binary integrity of first and last opcodes via span.dataView.
        assert.equal(span.dataView.getUint16(span.viewOffset, true), OPCODES.BEGIN_FRAME);
        assert.equal(
          span.dataView.getUint16(span.viewOffset + byteLength - 8, true),
          OPCODES.PRESENT
        );
      },
      { reuseResult: true, directView: true }
    );
    if (firstResultRef === null) {
      firstResultRef = res;
    } else {
      assert.equal(res, firstResultRef, 'reuseResult must return same result object');
    }
    assert.equal(res.packets, 5);
    assert.equal(res.bytes, frameStream.byteLength);
  }

  assert.ok(wrapCount >= 5, `expected repeated wrap-arounds, got ${wrapCount}`);
  assert.ok(nonWrapCount >= 5, `expected non-wrap drains, got ${nonWrapCount}`);
  assert.equal(consumer._wrapScratch.buffer, initialWrapBuf);
  assert.equal(consumer.wrapScratchView, initialWrapView);

  // Verify drainRing({ reuseResult: true }) reuses result object across empty & non-empty polls
  // while default drainRing() returns distinct objects.
  const mock = createFastMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx, { requireNonBlackFrame: false });
  proc.attachRing(consumer);

  const empty1 = proc.drainRing({ reuseResult: true });
  const empty2 = proc.drainRing({ reuseResult: true });
  assert.equal(empty1, empty2);
  assert.equal(empty1.bytesDrained, 0);

  const fresh1 = proc.drainRing();
  const fresh2 = proc.drainRing();
  assert.notEqual(fresh1, fresh2);

  // Stream 40 wrapped frames end-to-end through proc.drainRing({ reuseResult: true }).
  for (let i = 0; i < 40; i += 1) {
    assert.equal(producer.writeBytes(frameStream), true);
    const r = proc.drainRing({ reuseResult: true });
    assert.equal(r, empty1);
    assert.equal(r.presents, 1);
    assert.equal(r.frameCount, i + 1);
  }
  assert.equal(mock.counts.draw, 80);
  assert.equal(consumer._wrapScratch.buffer, initialWrapBuf);
  assert.equal(consumer.wrapScratchView, initialWrapView);

  // Verify monkey-patched executeCommandBuffer (e.g. GpuFrameRecorder.createInterceptor)
  // still intercepts ring-drained spans with exact byteLength, then returns to fast path on detach.
  const interceptedLengths = [];
  const origExec = proc.executeCommandBuffer;
  proc.executeCommandBuffer = function wrappedExec(bytes, opts) {
    interceptedLengths.push(bytes.byteLength);
    return origExec.call(this, bytes, opts);
  };
  assert.equal(producer.writeBytes(frameStream), true);
  proc.drainRing({ reuseResult: true });
  assert.deepEqual(interceptedLengths, [frameStream.byteLength]);
  delete proc.executeCommandBuffer;
  assert.equal(producer.writeBytes(frameStream), true);
  proc.drainRing({ reuseResult: true });
  assert.deepEqual(interceptedLengths, [frameStream.byteLength]);
});

test('W359: W342 mid-frame deferred destroys remain intact across hot-loop pooled frames', async () => {
  const mock = createFastMockWebGpu({ manualFence: true });
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    requireNonBlackFrame: false,
    enableResourceReplay: false,
  });

  // Create buffer + texture on frame 1, then destroy both mid-frame on frame 2 before PRESENT.
  const frame1 = new PacketBuilder()
    .beginFrame()
    .createBuffer(101, BUFFER_USAGE.VERTEX, 256)
    .createTexture(102, 64, 64)
    .present()
    .finish();
  processor.executeCommandBuffer(frame1);
  assert.equal(mock.destroyedResources.length, 0);

  const frame2 = new PacketBuilder()
    .beginFrame()
    .destroyResource(101)
    .destroyResource(102)
    .present()
    .finish();
  processor.executeCommandBuffer(frame2);

  // Resources must NOT be destroyed yet while the submitted fence is pending.
  // Frame 1 had zero deferred destroys (so zero queue.onSubmittedWorkDone() stall);
  // frame 2 queued 2 destroys and requested exactly 1 fence.
  assert.equal(mock.destroyedResources.length, 0);
  assert.equal(mock.pendingFences.length, 1);

  // Resolve fences and yield microtasks -> deferred destroys execute exactly once.
  for (const resolve of mock.pendingFences) resolve();
  await new Promise((r) => setImmediate(r));

  assert.equal(mock.destroyedResources.length, 2);
  assert.equal(processor._deferredDestroyCount, 2);
});

test('W359: worker handler throttles telemetry posts and preserves W350 8-strip swapchain probe + render_state_packets compatibility', async () => {
  assert.equal(SWAPCHAIN_PROBE_STRIPS.length, 8);
  assert.equal(SWAPCHAIN_PROBE_BUFFER_SIZE, 2048);

  const mock = createFastMockWebGpu({ probeVisibleAfterCopies: 8 });
  let fakeNow = 1000;
  const posted = [];
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    width: 1280,
    height: 720,
    now: () => fakeNow,
    swapchainProbeIntervalMs: 250,
    enableResourceReplay: false,
    gpuAdapter: { vendor: 'amd', architecture: 'rdna2', device: 'test', description: 'test' },
  });
  installRenderStatePackets(processor);

  const ring = GpuRingBuffer.create(32 * 1024);
  processor.attachRing(ring);

  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (m) => posted.push(m),
    telemetryIntervalMs: 250,
    now: () => fakeNow,
  });

  // Setup shader + pipeline + bind group so draws count as non-clear candidates for W350 probe.
  const setup = new PacketBuilder()
    .createBuffer(1, BUFFER_USAGE.UNIFORM, 64)
    .createShaderModule(10, SHADER_STAGE.VERTEX, '@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0.0); }')
    .createShaderModule(11, SHADER_STAGE.FRAGMENT, '@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }')
    .createRenderPipeline(20, 10, 11, 0, 0)
    .createBindGroup(30, 20, 0, [
      { binding: 0, resourceType: BIND_RESOURCE_TYPE.BUFFER, resourceId: 1, offset: 0, size: 64 },
    ])
    .finish();
  processor.executeCommandBuffer(setup);

  const frameWithExt = new PacketBuilder()
    .beginFrame()
    .beginRenderPassSwapchain([0.1, 0.2, 0.3, 1.0])
    .setPipeline(20)
    .setBindGroup2(0, 30)
    .draw(3, 1, 0, 0)
    .endRenderPass()
    .present()
    .finish();

  // Run 30 frames at 16ms (~60 Hz) = 480ms total -> telemetry should fire on frame 1 + at ~250ms only (<= 3 times, NOT 30!).
  for (let f = 0; f < 30; f += 1) {
    fakeNow = 1000 + f * 16;
    assert.equal(ring.writeBytes(frameWithExt), true);
    await handler.handleMessage({ drainRing: true });
    if (f === 0) {
      // Allow W350 8-strip probe async readback to settle after frame 1.
      await new Promise((r) => setImmediate(r));
    }
  }

  const frameMsgs = posted.filter((m) => typeof m.frame === 'number');
  const telemetryMsgs = posted.filter((m) => m.type === 'gpu-telemetry');
  const visibleMsgs = posted.filter((m) => m.type === 'first-visible-frame');

  assert.equal(frameMsgs.length, 30);
  assert.ok(
    telemetryMsgs.length >= 2 && telemetryMsgs.length <= 3,
    `expected throttled telemetry (2-3 posts over 480ms), got ${telemetryMsgs.length}`
  );
  assert.equal(visibleMsgs.length, 1, 'W350 8-strip swapchain probe must fire first-visible-frame once');
  assert.equal(mock.counts.copyTextureToBuffer, 8, 'W350 probe must encode 8 strips then stop once visible');
  assert.equal(processor._setBindGroup2MetaPool.length, 1, 'setBindGroup2 _wgpuMeta pool must reuse single slot');
});

test('G75: wgpu_worker cmdBuffer sub-range byteLength and recycle-cmd-buffer round-trip', async () => {
  const mock = createFastMockWebGpu();
  const posted = [];
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    width: 1280,
    height: 720,
    requireNonBlackFrame: false,
    enableResourceReplay: false,
  });
  const handler = createWgpuWorkerHandler({
    processor,
    postMessage: (msg, transfer) => posted.push({ msg, transfer }),
  });

  const frameBytes = new Uint8Array(
    new PacketBuilder()
      .beginFrame()
      .clearColor(0.2, 0.4, 0.6, 1.0)
      .draw(3, 1, 0, 0)
      .present()
      .finish()
  );

  // Oversized 512 KiB warm buffer with a poisoned extra PRESENT packet past `byteLength`.
  const oversized = new ArrayBuffer(512 * 1024);
  const dst = new Uint8Array(oversized);
  dst.set(frameBytes, 0);
  dst.set(frameBytes, frameBytes.byteLength); // must NOT be executed because byteLength stops before it

  await handler.handleMessage({
    cmdBuffer: oversized,
    byteLength: frameBytes.byteLength,
    recycleBuffer: true,
  });

  assert.equal(processor.frameCount, 1, 'only packets within [0, byteLength) must execute');
  assert.equal(mock.counts.draw, 1);
  assert.equal(
    processor._cachedCmdBuffer,
    null,
    'cached DataView reference must be cleared before transferring recycled cmdBuffer'
  );
  const recyclePosts = posted.filter((p) => p.msg && p.msg.type === 'recycle-cmd-buffer');
  assert.equal(recyclePosts.length, 1);
  assert.equal(recyclePosts[0].msg.cmdBuffer, oversized);
  assert.deepEqual(recyclePosts[0].transfer, [oversized]);

  // Small (< 256 KiB) buffer must NOT be recycled even if recycleBuffer is true, and omitted byteLength works.
  posted.length = 0;
  const smallBuf = frameBytes.slice().buffer;
  await handler.handleMessage({
    cmdBuffer: smallBuf,
    recycleBuffer: true,
  });
  assert.equal(processor.frameCount, 2);
  assert.equal(
    posted.filter((p) => p.msg && p.msg.type === 'recycle-cmd-buffer').length,
    0,
    'buffers below 256 KiB must not be recycled'
  );

  // SharedArrayBuffer must never be recycled (not transferable).
  posted.length = 0;
  const sab = new SharedArrayBuffer(512 * 1024);
  new Uint8Array(sab).set(frameBytes, 0);
  await handler.handleMessage({
    cmdBuffer: sab,
    byteLength: frameBytes.byteLength,
    recycleBuffer: true,
  });
  assert.equal(processor.frameCount, 3);
  assert.equal(
    posted.filter((p) => p.msg && p.msg.type === 'recycle-cmd-buffer').length,
    0,
    'SharedArrayBuffer must never emit recycle-cmd-buffer'
  );
});

test('W363: _handleWriteTexture reuses scratch descriptors with exact WebGPU args and zero stale field bleed', () => {
  const mock = createFastMockWebGpu();
  const captured = [];
  const rawRefs = [];
  mock.device.queue.writeTexture = (dst, data, dataLayout, size) => {
    rawRefs.push({ dst, origin: dst.origin, dataLayout, size });
    captured.push({
      texture: dst.texture,
      mipLevel: dst.mipLevel,
      origin: { ...dst.origin },
      data: new Uint8Array(data),
      dataLayout: { ...dataLayout },
      layoutKeys: Object.keys(dataLayout).sort(),
      size: { ...size },
    });
  };

  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    width: 1280,
    height: 720,
    enableStagingPool: false,
    enableResourceReplay: false,
    requireNonBlackFrame: false,
  });

  const texelsA = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
  const texelsB = new Uint8Array([21, 22, 23, 24, 25, 26, 27, 28]);
  const texelsC = new Uint8Array([31, 32, 33, 34]);
  const texelsD = new Uint8Array([41, 42, 43, 44, 45, 46, 47, 48]);

  // Exercise all 4 combinations of optional (bytesPerRow, rowsPerImage) in sequence so any
  // stale field bleed between consecutive calls is immediately caught.
  const stream = new PacketBuilder()
    .createTexture(10, 64, 64)
    .createTexture(11, 32, 32)
    // 1. Both bytesPerRow > 0 and rowsPerImage > 0
    .writeTexture(10, 2, 4, 8, 1, 2, 2, 3, 256, 4, texelsA)
    // 2. bytesPerRow === 0 and rowsPerImage === 0 (neither optional key may bleed!)
    .writeTexture(11, 0, 1, 2, 0, 2, 1, 1, 0, 0, texelsB)
    // 3. bytesPerRow > 0 only (rowsPerImage must be absent!)
    .writeTexture(10, 1, 0, 3, 2, 1, 1, 1, 64, 0, texelsC)
    // 4. rowsPerImage > 0 only (bytesPerRow must be absent!)
    .writeTexture(11, 3, 5, 6, 7, 2, 1, 2, 0, 2, texelsD)
    // 5. Back to both > 0 with different values
    .writeTexture(10, 0, 0, 0, 0, 4, 4, 1, 16, 4, texelsA)
    .finish();

  processor.executeCommandBuffer(stream);

  const tex10 = processor.textures.get(10);
  const tex11 = processor.textures.get(11);
  assert.equal(captured.length, 5);

  // Verify exact scratch object reuse across all 5 calls (zero per-packet dst/origin/size alloc).
  for (let i = 0; i < rawRefs.length; i += 1) {
    assert.equal(rawRefs[i].dst, processor._wtDstScratch);
    assert.equal(rawRefs[i].origin, processor._wtOriginScratch);
    assert.equal(rawRefs[i].size, processor._wtSizeScratch);
  }
  assert.equal(rawRefs[0].dataLayout, processor._wtLayoutFullScratch);
  assert.equal(rawRefs[1].dataLayout, processor._wtLayoutOffsetOnlyScratch);
  assert.equal(rawRefs[2].dataLayout, processor._wtLayoutBprScratch);
  assert.equal(rawRefs[3].dataLayout, processor._wtLayoutRpiScratch);
  assert.equal(rawRefs[4].dataLayout, processor._wtLayoutFullScratch);

  // Verify dst.texture reference is cleared after _handleWriteTexture (never retained).
  assert.equal(processor._wtDstScratch.texture, null);

  // Verify exact values & own-property shapes for every single call.
  assert.equal(captured[0].texture, tex10);
  assert.equal(captured[0].mipLevel, 2);
  assert.deepEqual(captured[0].origin, { x: 4, y: 8, z: 1 });
  assert.deepEqual(captured[0].dataLayout, { offset: 0, bytesPerRow: 256, rowsPerImage: 4 });
  assert.deepEqual(captured[0].layoutKeys, ['bytesPerRow', 'offset', 'rowsPerImage']);
  assert.deepEqual(captured[0].size, { width: 2, height: 2, depthOrArrayLayers: 3 });
  assert.deepEqual(Array.from(captured[0].data), Array.from(texelsA));

  assert.equal(captured[1].texture, tex11);
  assert.equal(captured[1].mipLevel, 0);
  assert.deepEqual(captured[1].origin, { x: 1, y: 2, z: 0 });
  assert.deepEqual(captured[1].dataLayout, { offset: 0 });
  assert.deepEqual(captured[1].layoutKeys, ['offset']);
  assert.deepEqual(captured[1].size, { width: 2, height: 1, depthOrArrayLayers: 1 });
  assert.deepEqual(Array.from(captured[1].data), Array.from(texelsB));

  assert.equal(captured[2].texture, tex10);
  assert.equal(captured[2].mipLevel, 1);
  assert.deepEqual(captured[2].origin, { x: 0, y: 3, z: 2 });
  assert.deepEqual(captured[2].dataLayout, { offset: 0, bytesPerRow: 64 });
  assert.deepEqual(captured[2].layoutKeys, ['bytesPerRow', 'offset']);
  assert.deepEqual(captured[2].size, { width: 1, height: 1, depthOrArrayLayers: 1 });

  assert.equal(captured[3].texture, tex11);
  assert.equal(captured[3].mipLevel, 3);
  assert.deepEqual(captured[3].origin, { x: 5, y: 6, z: 7 });
  assert.deepEqual(captured[3].dataLayout, { offset: 0, rowsPerImage: 2 });
  assert.deepEqual(captured[3].layoutKeys, ['offset', 'rowsPerImage']);
  assert.deepEqual(captured[3].size, { width: 2, height: 1, depthOrArrayLayers: 2 });

  assert.equal(captured[4].texture, tex10);
  assert.equal(captured[4].mipLevel, 0);
  assert.deepEqual(captured[4].origin, { x: 0, y: 0, z: 0 });
  assert.deepEqual(captured[4].dataLayout, { offset: 0, bytesPerRow: 16, rowsPerImage: 4 });
  assert.deepEqual(captured[4].size, { width: 4, height: 4, depthOrArrayLayers: 1 });

  // Also verify hot cmdBuffer execution in worker handler runs synchronously and returns
  // a shared resolved Promise reference (zero per-message Promise allocation).
  const handler = createWgpuWorkerHandler({ processor, postMessage: () => {} });
  const framePkt = new PacketBuilder().beginFrame().present().finish();
  const ret1 = handler.handleMessage({
    cmdBuffer: framePkt.buffer,
    byteLength: framePkt.byteLength,
  });
  assert.equal(processor.frameCount, 1, 'cmdBuffer must execute synchronously before awaiting');
  const ret2 = handler.handleMessage({
    cmdBuffer: framePkt.buffer,
    byteLength: framePkt.byteLength,
  });
  assert.equal(processor.frameCount, 2);
  assert.equal(ret1, ret2, 'hot sync handleMessage paths must reuse shared resolved Promise');
});

test('W363: cached ShaderFallbackRouter.getMissingShaderSummary stays accurate across mutations and preserves mutation isolation', () => {
  let tick = 1000;
  const router = new ShaderFallbackRouter({ now: () => ++tick });

  const s0 = router.getMissingShaderSummary();
  assert.equal(s0.totalMissingEvents, 0);
  assert.equal(s0.uniqueMissingHashes, 0);
  assert.equal(s0.realShaderGuard.registeredRealShaders, 0);
  assert.equal(router._summaryCacheVersion, router._summaryVersion);

  // Record missing shaders -> bumps _summaryVersion and updates cached summary immediately.
  const vBefore = router._summaryVersion;
  router.recordMissingShader('0x1111', 'vertex', 'not_in_pack');
  router.recordMissingShader('0x1111', 'vertex', 'not_in_pack');
  router.recordMissingShader('0x2222', 'fragment', 'compile_failed');
  assert.ok(router._summaryVersion > vBefore);

  const s1 = router.getMissingShaderSummary({ limit: 8 });
  assert.equal(s1.totalMissingEvents, 3);
  assert.equal(s1.uniqueMissingHashes, 2);
  assert.equal(s1.countsByReason.not_in_pack, 2);
  assert.equal(s1.countsByReason.compile_failed, 1);
  assert.equal(s1.topMissing.length, 2);
  assert.equal(s1.topMissing[0].hitCount, 2);

  // Repeated steady-state call hits cache (version unchanged) and stays mutation-isolated.
  const versionSteady = router._summaryVersion;
  s1.countsByReason.not_in_pack = 9999;
  s1.topMissing[0].hitCount = 9999;
  const s2 = router.getMissingShaderSummary({ limit: 8 });
  assert.equal(router._summaryVersion, versionSteady);
  assert.equal(s2.countsByReason.not_in_pack, 2);
  assert.equal(s2.topMissing[0].hitCount, 2);

  // Filtered and limit=0 queries stay accurate on the same version.
  const fragOnly = router.getMissingShaderSummary({ stage: 'fragment', limit: 4 });
  assert.equal(fragOnly.topMissing.length, 1);
  assert.equal(fragOnly.topMissing[0].stage, 'fragment');
  assert.deepEqual(router.getMissingShaderSummary({ limit: 0 }).topMissing, []);

  // Real-shader guard mutations invalidate cache cleanly.
  router.registerRealShader('0x3333', 'fragment', '@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }');
  assert.equal(router.getMissingShaderSummary().realShaderGuard.registeredRealShaders, 1);
  router.unregisterRealShader('0x3333', 'fragment');
  assert.equal(router.getMissingShaderSummary().realShaderGuard.registeredRealShaders, 0);

  // resetMissingTelemetry() / reset() invalidate cache cleanly.
  router.resetMissingTelemetry();
  const afterReset = router.getMissingShaderSummary({ limit: 8 });
  assert.equal(afterReset.totalMissingEvents, 0);
  assert.equal(afterReset.uniqueMissingHashes, 0);
  assert.deepEqual(afterReset.topMissing, []);
});

test('W363: cached WgslPackCache.getStats and WgslPack.stats stay accurate across all state mutations', () => {
  const vsCode = '@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0.0); }';
  const fsCode = '@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }';
  const csCode = '@compute @workgroup_size(1) fn cs_main() {}';
  const packBuf = buildWgslPackBuffer([
    { hash: 0xaaa1n, stage: 'vertex', wgsl: vsCode },
    { hash: 0xaaa2n, stage: 'fragment', wgsl: fsCode },
  ]);

  const cache = new WgslPackCache({ lazyDecode: true });
  const initial = cache.getStats();
  assert.equal(initial.packCount, 0);
  assert.equal(initial.shaderCount, 0);

  const v0 = cache._statsVersion;
  const pack = cache.loadPackBuffer(packBuf, 'main');
  assert.ok(cache._statsVersion > v0);

  const st1 = cache.getStats();
  assert.equal(st1.packCount, 1);
  assert.equal(st1.shaderCount, 2);
  assert.equal(st1.staticShaderCount, 2);
  assert.deepEqual(st1.stageCounts, { vertex: 1, fragment: 1, compute: 0 });
  assert.equal(st1.packs.main.decodedCount, 0);

  // Mutation isolation on cached steady-state snapshot.
  const cachedObj = cache._cachedStats;
  st1.stageCounts.vertex = 999;
  st1.packs.main.decodedCount = 999;
  const st1Again = cache.getStats();
  assert.equal(cache._cachedStats, cachedObj, 'steady-state getStats must hit cached snapshot');
  assert.equal(st1Again.stageCounts.vertex, 1);
  assert.equal(st1Again.packs.main.decodedCount, 0);

  // Lazy decode of static entry inside pack must invalidate cached stats via decodedCount stamp.
  assert.equal(pack.getSource(0xaaa1n), vsCode);
  const stAfterDecode = cache.getStats();
  assert.equal(stAfterDecode.packs.main.decodedCount, 1);

  // Compiling & caching a shader module bumps _statsVersion without re-scanning static entries.
  const entriesVerBeforeCompile = cache._entriesVersion;
  const mockDevice = { createShaderModule: (d) => ({ label: d.label, code: d.code }) };
  cache.getOrCreateShaderModule(mockDevice, 0xaaa1n);
  cache.getOrCreateShaderModule(mockDevice, 0xaaa1n);
  assert.equal(cache._entriesVersion, entriesVerBeforeCompile);
  const stAfterCompile = cache.getStats();
  assert.equal(stAfterCompile.compiles, 1);
  assert.equal(stAfterCompile.misses, 1);
  assert.equal(stAfterCompile.hits, 1);
  assert.equal(stAfterCompile.cachedModules, 1);

  // Dynamic shader registration, pinning, unpinning, eviction, and clear all stay 100% accurate.
  cache.registerDynamicShader(0xaaa3n, 'compute', csCode);
  let st = cache.getStats();
  assert.equal(st.dynamicCount, 1);
  assert.equal(st.shaderCount, 3);
  assert.deepEqual(st.stageCounts, { vertex: 1, fragment: 1, compute: 1 });

  cache.pinHashes([0xaaa1n]);
  assert.equal(cache.getStats().pinnedCount, 1);
  cache.unpinHashes([0xaaa1n]);
  assert.equal(cache.getStats().pinnedCount, 0);

  cache.unregisterDynamicShader(0xaaa3n);
  st = cache.getStats();
  assert.equal(st.dynamicCount, 0);
  assert.equal(st.shaderCount, 2);

  cache.evict(0xaaa1n);
  st = cache.getStats();
  assert.equal(st.cachedModules, 0);
  assert.equal(st.evictions, 1);

  cache.clear({ resetStats: true });
  st = cache.getStats();
  assert.equal(st.packCount, 0);
  assert.equal(st.shaderCount, 0);
  assert.equal(st.hits, 0);
});

test('W363: _invalidateRemappedBindGroupsForBindGroup uses O(affected) reverse index and handles LRU eviction cleanly', () => {
  const mock = createFastMockWebGpu();
  const processor = new WgpuCommandProcessor(mock.device, mock.ctx, {
    width: 1280,
    height: 720,
    enableResourceReplay: false,
    requireNonBlackFrame: false,
  });
  const handler = installRenderStatePackets(processor, { remappedBindGroupCacheLimit: 8 });

  // Populate multiple remapped bind groups across render/compute pipelines and dynamic offsets.
  handler._cacheRemappedBindGroup('r:100:400', { id: 'r100-400' }, 'r:100', 400);
  handler._cacheRemappedBindGroup('r:100:400@0=256', { id: 'r100-400-256' }, 'r:100', 400);
  handler._cacheRemappedBindGroup('c:200:400', { id: 'c200-400' }, 'c:200', 400);
  handler._cacheRemappedBindGroup('r:100:401', { id: 'r100-401' }, 'r:100', 401);
  handler._cacheRemappedBindGroup('c:200:402@0=512', { id: 'c200-402-512' }, 'c:200', 402);

  assert.equal(handler.remappedBindGroups.size, 5);
  assert.equal(handler._remappedBgByBindGroupId.get(400).size, 3);
  assert.equal(handler._remappedBgByPipelineKey.get('r:100').size, 3);

  // Invalidating an unrelated bindGroupId is O(1) no-op and touches nothing.
  handler._invalidateRemappedBindGroupsForBindGroup(999);
  assert.equal(handler.remappedBindGroups.size, 5);

  // Invalidating bindGroupId 400 removes all 3 affected entries across render + compute + dynOffsets.
  handler._invalidateRemappedBindGroupsForBindGroup(400);
  assert.equal(handler.remappedBindGroups.size, 2);
  assert.equal(handler.remappedBindGroups.has('r:100:400'), false);
  assert.equal(handler.remappedBindGroups.has('r:100:400@0=256'), false);
  assert.equal(handler.remappedBindGroups.has('c:200:400'), false);
  assert.equal(handler.remappedBindGroups.has('r:100:401'), true);
  assert.equal(handler.remappedBindGroups.has('c:200:402@0=512'), true);
  assert.equal(handler._remappedBgByBindGroupId.has(400), false);
  assert.equal(handler._remappedBgByPipelineKey.get('r:100').size, 1);
  assert.equal(handler._remappedBgByPipelineKey.get('c:200').size, 1);

  // Invalidating compute pipeline 200 removes only its entries and cleans up reverse index sets.
  handler._invalidateRemappedBindGroupsForPipeline(200, true);
  assert.equal(handler.remappedBindGroups.size, 1);
  assert.equal(handler.remappedBindGroups.has('c:200:402@0=512'), false);
  assert.equal(handler._remappedBgByBindGroupId.has(402), false);
  assert.equal(handler._remappedBgByPipelineKey.has('c:200'), false);
  assert.equal(handler.remappedBindGroups.has('r:100:401'), true);

  // Verify LRU eviction keeps both reverse indexes completely in sync (zero leaked sets).
  for (let i = 0; i < 10; i += 1) {
    handler._cacheRemappedBindGroup(`r:300:${500 + i}`, { id: i });
  }
  assert.equal(handler.remappedBindGroups.size, 8);
  assert.equal(handler.remappedBindGroups.has('r:100:401'), false);
  assert.equal(handler._remappedBgByBindGroupId.has(401), false);
  assert.equal(handler._remappedBgByPipelineKey.has('r:100'), false);
  assert.equal(handler._remappedBgByPipelineKey.get('r:300').size, 8);
});
