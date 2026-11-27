import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const uniformRingMod = require('../web/workers/uniform_ring.js');
const indirectMod = require('../web/workers/indirect_draw.js');
const texturePacketsMod = require('../web/workers/texture_packets.js');
const { WgpuCommandProcessor, OPCODES } = require('../web/workers/wgpu_worker.js');
const { GpuStagingPool } = require('../web/workers/staging_pool.js');

const { DynamicUniformRing, coalesceDirtyRanges } = uniformRingMod;
const {
  U32_MAX,
  BUFFER_USAGE,
  INDIRECT_KIND,
  INDIRECT_CLAMP_KIND,
  IndirectDrawError,
  normalizeRecord,
  encodeRecord,
  decodeRecord,
  encodeDrawIndirectInto,
  encodeDrawIndexedIndirectInto,
  encodeDispatchIndirectInto,
  decodeDrawIndirect,
  decodeDrawIndexedIndirect,
  decodeDispatchIndirect,
  encodeBatch,
  computeIndirectBatchStepPlan,
  validateIndirectBatch,
  validateDispatchCounts,
  encodeDispatches,
  encodeDispatchForItems,
  encodeMultiDraw,
  encodeMultiDispatchIndirect,
  executeIndirect,
  dispatchIndirectDraw,
  dispatchIndexedIndirectDraw,
  dispatchIndirectBatch,
  GpuIndirectClampPass,
  encodeMeshComputeThenDraw,
  IndirectArgsBuffer,
  getStats: getIndirectStats,
  resetStats: resetIndirectStats,
} = indirectMod;
const {
  TextureScratchPool,
  installTexturePackets,
  repackRowsTo256,
  expandA8ToRgba8,
  copyFromSharedIfNeeded,
  resolveCompatibleViewFormats,
} = texturePacketsMod;

// ---------------------------------------------------------------------------
// Helpers: deterministic mock WebGPU devices & binary packet writers
// ---------------------------------------------------------------------------

function createUniformMockDevice(options = {}) {
  let nextBufferId = 1;
  const createdBuffers = [];
  const queueWrites = [];
  let writeBufferCount = 0;
  const recordWrites = options.recordWrites !== false;

  const queue = {
    writeBuffer(buffer, bufferOffset, data) {
      if (
        typeof SharedArrayBuffer !== 'undefined' &&
        (data instanceof SharedArrayBuffer ||
          (ArrayBuffer.isView(data) && data.buffer instanceof SharedArrayBuffer))
      ) {
        throw new TypeError(
          "Failed to execute 'writeBuffer' on 'GPUQueue': The provided ArrayBufferView value must not be shared."
        );
      }
      const bytes =
        data instanceof Uint8Array
          ? data
          : ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            : new Uint8Array(data);
      assert.equal(bytes.byteOffset % 4, 0, 'source byteOffset must be 4B aligned');
      assert.equal(bufferOffset % 4, 0, 'bufferOffset must be 4B aligned');
      assert.equal(bytes.byteLength % 4, 0, 'byteLength must be 4B aligned');
      assert.ok(bufferOffset + bytes.byteLength <= buffer.size, 'writeBuffer within bounds');
      buffer._data.set(bytes, bufferOffset);
      writeBufferCount += 1;
      if (recordWrites) {
        queueWrites.push({
          buffer,
          bufferOffset,
          byteLength: bytes.byteLength,
          dataRef: bytes,
        });
      }
    },
  };

  const device = {
    limits: options.limits || {
      minUniformBufferOffsetAlignment: 256,
      maxUniformBufferBindingSize: 65536,
    },
    queue,
    createBuffer(desc) {
      const id = nextBufferId++;
      const buffer = {
        id,
        label: desc.label || `ubuf-${id}`,
        size: desc.size,
        usage: desc.usage,
        _data: new Uint8Array(desc.size),
        destroyCount: 0,
        destroy() {
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
    queueWrites,
    getWriteBufferCount: () => writeBufferCount,
  };
}

function createIndirectMockDevice(options = {}) {
  const buffers = [];
  const writes = [];
  const bindGroups = [];
  let createBindGroupCalls = 0;
  let createPipelineCalls = 0;
  let nextId = 1;
  const recordWrites = options.recordWrites === true;

  const queue = {
    writeBuffer(buffer, offset, data) {
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      if (buffer._data) {
        buffer._data.set(bytes, offset);
      }
      if (recordWrites) {
        writes.push({ buffer, offset, length: bytes.byteLength, snapshot: bytes.slice() });
      }
    },
    submit(cmds) {
      for (let i = 0; i < cmds.length; i += 1) {
        const copies = cmds[i].copies;
        if (!copies) continue;
        for (let j = 0; j < copies.length; j += 1) {
          const c = copies[j];
          c.dst._data.set(c.src._data.subarray(c.srcOffset, c.srcOffset + c.size), c.dstOffset);
        }
      }
    },
  };

  const device = {
    features: new Set(options.features || []),
    limits: {
      maxComputeWorkgroupsPerDimension: options.maxComputeWorkgroupsPerDimension || 65535,
    },
    queue,
    createBuffer(desc) {
      const buf = {
        id: nextId++,
        label: desc.label,
        size: desc.size,
        usage: desc.usage,
        _data: new Uint8Array(desc.size),
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      buffers.push(buf);
      return buf;
    },
    createShaderModule(desc) {
      return { id: nextId++, code: desc.code, label: desc.label };
    },
    createBindGroupLayout(desc) {
      return { id: nextId++, desc };
    },
    createPipelineLayout(desc) {
      return { id: nextId++, desc };
    },
    createComputePipeline(desc) {
      createPipelineCalls += 1;
      return {
        id: nextId++,
        desc,
        destroyed: false,
        getBindGroupLayout: () => ({ id: nextId++ }),
      };
    },
    createBindGroup(desc) {
      createBindGroupCalls += 1;
      const bg = { id: nextId++, layout: desc.layout, destroyed: false };
      bindGroups.push(bg);
      return bg;
    },
    createCommandEncoder() {
      const copies = [];
      return {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          copies.push({ src, srcOffset, dst, dstOffset, size });
        },
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            dispatchWorkgroupsIndirect() {},
            end() {},
          };
        },
        beginRenderPass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            drawIndirect() {},
            drawIndexedIndirect() {},
            end() {},
          };
        },
        finish() {
          return { copies };
        },
      };
    },
  };

  return {
    device,
    buffers,
    writes,
    bindGroups,
    getCreateBindGroupCalls: () => createBindGroupCalls,
    getCreatePipelineCalls: () => createPipelineCalls,
  };
}

function createTextureMockGpu(options = {}) {
  const rejectSabWriteTexture = options.rejectSabWriteTexture === true;
  const recordSnapshots = options.recordSnapshots !== false;
  let nextHandle = 1;
  let writeTextureCount = 0;
  let copyBufferToTextureCount = 0;
  let writeBufferCount = 0;
  const createdTextures = [];
  const writeTextureCalls = [];
  const copyBufferToTextureCalls = [];

  const device = {
    features: new Set(options.features || []),
    createBuffer(desc) {
      const backing = new Uint8Array(desc.size);
      return {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        backing,
        _data: backing,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
    },
    createTexture(desc) {
      const tex = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        width: desc.size.width,
        height: desc.size.height,
        depthOrArrayLayers: desc.size.depthOrArrayLayers || 1,
        mipLevelCount: desc.mipLevelCount || 1,
        sampleCount: desc.sampleCount || 1,
        dimension: desc.dimension || '2d',
        format: desc.format,
        usage: desc.usage,
        viewFormats: desc.viewFormats ? desc.viewFormats.slice() : undefined,
        desc,
        destroyed: false,
        createView() {
          return { kind: 'GPUTextureView', textureHandle: tex.handle };
        },
        destroy() {
          tex.destroyed = true;
        },
      };
      createdTextures.push(tex);
      return tex;
    },
    createCommandEncoder() {
      return {
        copyBufferToBuffer(srcBuf, srcOff, dstBuf, dstOff, size) {
          if (srcBuf.backing && dstBuf.backing) {
            dstBuf.backing.set(srcBuf.backing.subarray(srcOff, srcOff + size), dstOff);
          }
        },
        copyBufferToTexture(source, destination, copySize) {
          assert.equal(
            source.bytesPerRow % 256,
            0,
            `copyBufferToTexture bytesPerRow (${source.bytesPerRow}) must be 256B aligned`
          );
          copyBufferToTextureCount += 1;
          if (recordSnapshots) {
            const total =
              source.bytesPerRow *
              (source.rowsPerImage || copySize.height) *
              (copySize.depthOrArrayLayers || 1);
            copyBufferToTextureCalls.push({
              source: { ...source },
              destination: { ...destination, origin: { ...destination.origin } },
              copySize: { ...copySize },
              bytes: source.buffer.backing.slice(source.offset || 0, (source.offset || 0) + total),
            });
          }
        },
        beginRenderPass: () => ({ end() {} }),
        beginComputePass: () => ({ end() {} }),
        finish: () => ({ kind: 'GPUCommandBuffer' }),
      };
    },
    queue: {
      writeBuffer(buffer, bufferOffset, data) {
        if (
          typeof SharedArrayBuffer !== 'undefined' &&
          data &&
          data.buffer instanceof SharedArrayBuffer
        ) {
          throw new TypeError('SharedArrayBuffer disallowed in writeBuffer');
        }
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        if (buffer.backing) {
          buffer.backing.set(bytes, bufferOffset);
        }
        writeBufferCount += 1;
      },
      writeTexture(destination, data, dataLayout, size) {
        const isSab =
          typeof SharedArrayBuffer !== 'undefined' &&
          data &&
          data.buffer instanceof SharedArrayBuffer;
        if (rejectSabWriteTexture && isSab) {
          throw new TypeError('Legacy WebGPU: SharedArrayBuffer rejected in writeTexture');
        }
        writeTextureCount += 1;
        if (recordSnapshots) {
          const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
          writeTextureCalls.push({
            destination: {
              ...destination,
              origin: destination.origin ? { ...destination.origin } : destination.origin,
            },
            dataLayout: { ...dataLayout },
            size: { ...size },
            isSab,
            byteOffset: data.byteOffset,
            byteLength: data.byteLength,
            bytes: bytes.slice(),
          });
        }
      },
      submit() {},
    },
  };

  const ctx = {
    getCurrentTexture: () => ({ createView: () => ({ kind: 'swapchain' }) }),
  };

  return {
    device,
    ctx,
    createdTextures,
    writeTextureCalls,
    copyBufferToTextureCalls,
    getWriteTextureCount: () => writeTextureCount,
    getCopyBufferToTextureCount: () => copyBufferToTextureCount,
    getWriteBufferCount: () => writeBufferCount,
  };
}

function buildCreateTexture2Packet({
  id,
  width,
  height,
  layers = 1,
  mips = 1,
  samples = 1,
  usage = 0x06,
  dimension = 1,
  format = 22,
  viewFormats = [],
}) {
  const out = new Uint8Array(8 + 40);
  const dv = new DataView(out.buffer);
  dv.setUint16(0, 35, true);
  dv.setUint32(4, 40, true);
  dv.setUint32(8, id, true);
  dv.setUint32(12, width, true);
  dv.setUint32(16, height, true);
  dv.setUint32(20, layers, true);
  dv.setUint32(24, mips, true);
  dv.setUint32(28, samples, true);
  dv.setUint32(32, usage, true);
  dv.setUint8(36, dimension);
  dv.setUint8(37, format);
  dv.setUint8(38, viewFormats.length);
  for (let i = 0; i < viewFormats.length; i += 1) {
    dv.setUint8(40 + i, viewFormats[i]);
  }
  return out;
}

function buildWriteTexturePacket({
  textureId,
  mipLevel = 0,
  originX = 0,
  originY = 0,
  originZ = 0,
  width,
  height,
  layers = 1,
  bytesPerRow = 0,
  rowsPerImage = 0,
  data,
  useSab = false,
}) {
  const totalBytes = 8 + 48 + data.byteLength;
  const backing = useSab ? new SharedArrayBuffer(totalBytes) : new ArrayBuffer(totalBytes);
  const out = new Uint8Array(backing);
  const dv = new DataView(backing);
  dv.setUint16(0, OPCODES.WRITE_TEXTURE, true);
  dv.setUint32(4, 48 + data.byteLength, true);
  dv.setUint32(8, textureId, true);
  dv.setUint32(12, mipLevel, true);
  dv.setUint32(16, originX, true);
  dv.setUint32(20, originY, true);
  dv.setUint32(24, originZ, true);
  dv.setUint32(28, width, true);
  dv.setUint32(32, height, true);
  dv.setUint32(36, layers, true);
  dv.setUint32(40, bytesPerRow, true);
  dv.setUint32(44, rowsPerImage, true);
  dv.setUint32(48, data.byteLength, true);
  dv.setUint32(52, 0, true);
  out.set(data, 56);
  return out;
}

// ===========================================================================
// 1. DynamicUniformRing zero-alloc hot paths (12,000+ allocations)
// ===========================================================================

test('uniform_ring: 12,000+ zero-alloc allocations across multi-frame rotations, wrap-around, and dynamic offsets', () => {
  const { device, queueWrites, getWriteBufferCount } = createUniformMockDevice();
  const ring = new DynamicUniformRing({
    device,
    byteSize: 256 * 1024, // 256 KiB = 1024 draws/slot
    maxByteSize: 1024 * 1024,
    framesInFlight: 3,
  });

  const frameOut = {};
  const allocOut = {};
  const flushOut = {};
  const endOut = {};
  const bgOut = { binding: 0, resource: { buffer: null, offset: 0, size: 0 } };
  const payloadU32 = new Uint32Array(16); // 64 bytes
  const payloadDvBuf = new ArrayBuffer(20);
  const payloadDv = new DataView(payloadDvBuf);
  const payloadClamped = new Uint8ClampedArray(12);

  const FRAMES = 20;
  const DRAWS_PER_FRAME = 640; // 20 * 640 = 12,800 draws!

  for (let frame = 0; frame < FRAMES; frame += 1) {
    const fInfo = ring.beginFrame(frame, frameOut);
    assert.equal(fInfo, frameOut, 'beginFrame reuses caller out object');
    assert.equal(fInfo.frameIndex, frame);
    assert.equal(fInfo.slotIndex, frame % 3);

    const bgEntry = ring.createBindGroupEntry(0, 64, fInfo.slotIndex, bgOut);
    assert.equal(bgEntry, bgOut);
    assert.equal(bgEntry.resource.buffer, fInfo.buffer);
    assert.equal(bgEntry.resource.size, 64);

    for (let d = 0; d < DRAWS_PER_FRAME; d += 1) {
      let res;
      if ((d & 3) === 0) {
        // allocateInto + direct shadow write
        res = ring.allocateInto(48, allocOut);
        assert.equal(res, allocOut);
        const off = res.dynamicOffset;
        res.shadow[off] = frame & 0xff;
        res.shadow[off + 1] = d & 0xff;
        res.shadow[off + 2] = (d >>> 8) & 0xff;
        res.shadow[off + 3] = 0xa5;
      } else if ((d & 3) === 1) {
        // pushConstantsInto with Uint32Array
        payloadU32[0] = (frame << 16) | d;
        payloadU32[1] = 0xdead0000 ^ d;
        payloadU32[15] = frame * 1000 + d;
        res = ring.pushConstantsInto(payloadU32, allocOut);
        assert.equal(res, allocOut);
      } else if ((d & 3) === 2) {
        // pushConstantsInto with DataView (zero Uint8Array wrapper fast path)
        payloadDv.setUint32(0, (frame << 16) | d, true);
        payloadDv.setUint32(4, 0xcafe0000 ^ d, true);
        payloadDv.setUint32(16, 0x13572468, true);
        res = ring.pushConstantsInto(payloadDv, allocOut);
        assert.equal(res, allocOut);
      } else {
        // pushCbvInto with Uint8ClampedArray
        payloadClamped[0] = frame & 0xff;
        payloadClamped[1] = d & 0xff;
        payloadClamped[11] = 0x7f;
        res = ring.pushCbvInto(payloadClamped, allocOut);
        assert.equal(res, allocOut);
      }

      assert.equal(res.dynamicOffset, d * 256);
      assert.equal(res.dynamicOffset % 256, 0);
      assert.equal(res.alignedSize, 256);

      const dynOffsets = ring.getDynamicOffsetArray(res.dynamicOffset);
      assert.equal(dynOffsets.length, 1);
      assert.equal(dynOffsets[0], d * 256);
    }

    const flushed = ring.flush(flushOut);
    assert.equal(flushed, flushOut);
    assert.equal(flushed.uploadedBytes, DRAWS_PER_FRAME * 256);
    assert.equal(flushed.drawCount, DRAWS_PER_FRAME);

    // Verify exact uploaded bytes in slot GPU buffer for every draw type.
    const gpuBytes = fInfo.buffer._data;
    const gpuU32 = new Uint32Array(gpuBytes.buffer);
    for (const checkDraw of [0, 1, 2, 3, 128, 319, 636, 637, 638, 639]) {
      const byteOff = checkDraw * 256;
      const wordOff = byteOff >>> 2;
      if ((checkDraw & 3) === 0) {
        assert.equal(gpuBytes[byteOff], frame & 0xff);
        assert.equal(gpuBytes[byteOff + 1], checkDraw & 0xff);
        assert.equal(gpuBytes[byteOff + 2], (checkDraw >>> 8) & 0xff);
        assert.equal(gpuBytes[byteOff + 3], 0xa5);
        assert.equal(gpuBytes[byteOff + 48], 0);
        assert.equal(gpuBytes[byteOff + 255], 0);
      } else if ((checkDraw & 3) === 1) {
        assert.equal(gpuU32[wordOff], (frame << 16) | checkDraw);
        assert.equal(gpuU32[wordOff + 1], (0xdead0000 ^ checkDraw) >>> 0);
        assert.equal(gpuU32[wordOff + 15], frame * 1000 + checkDraw);
        assert.equal(gpuBytes[byteOff + 64], 0);
      } else if ((checkDraw & 3) === 2) {
        assert.equal(gpuU32[wordOff], (frame << 16) | checkDraw);
        assert.equal(gpuU32[wordOff + 1], (0xcafe0000 ^ checkDraw) >>> 0);
        assert.equal(gpuU32[wordOff + 4], 0x13572468);
        assert.equal(gpuBytes[byteOff + 20], 0);
      } else {
        assert.equal(gpuBytes[byteOff], frame & 0xff);
        assert.equal(gpuBytes[byteOff + 1], checkDraw & 0xff);
        assert.equal(gpuBytes[byteOff + 11], 0x7f);
        assert.equal(gpuBytes[byteOff + 12], 0);
      }
    }

    const ended = ring.endFrame(frame, { out: endOut });
    assert.equal(ended, endOut);
    assert.equal(ended.frameIndex, frame);
    assert.equal(ended.frameBytes, DRAWS_PER_FRAME * 256);
  }

  assert.equal(getWriteBufferCount(), FRAMES, '1 coalesced writeBuffer per frame');
  // Steady-state per-slot flush view cache: frame 3..19 reuse exact same Uint8Array view as slot's first frame!
  assert.equal(queueWrites[3].dataRef, queueWrites[0].dataRef);
  assert.equal(queueWrites[4].dataRef, queueWrites[1].dataRef);
  assert.equal(queueWrites[5].dataRef, queueWrites[2].dataRef);

  const stats = ring.getStats();
  assert.equal(stats.totalDrawCount, FRAMES * DRAWS_PER_FRAME);
  assert.equal(stats.zeroAllocAllocs, FRAMES * DRAWS_PER_FRAME);
  assert.equal(stats.writeBufferCalls, FRAMES);
  ring.destroy();
});

test('uniform_ring: wrap-around, out-of-order dirty coalescing (no splice), batch/reset out reuse, and SAB safety', () => {
  const { device } = createUniformMockDevice({ recordWrites: false });
  const ring = new DynamicUniformRing({
    device,
    byteSize: 1024, // 4 x 256B slices
    maxByteSize: 1024,
    framesInFlight: 2,
    allowWrap: true,
  });

  const batchOut = {};
  const endBatchOut = {};
  const allocOut = {};
  const resetOut = {};
  const u32 = new Uint32Array(4);

  ring.beginFrame(0);
  assert.equal(ring.beginBatch(batchOut), batchOut);

  // Write out-of-order into offsets 768, 256, 512, 0 -> coalesces into single [0, 1024) range without Array.splice!
  for (const [idx, off] of [
    [1, 768],
    [2, 256],
    [3, 512],
    [4, 0],
  ]) {
    u32[0] = idx * 100;
    u32[1] = off;
    ring.writeStaged(off, u32, { out: allocOut });
    assert.equal(allocOut.dynamicOffset, off);
  }
  const ranges = ring.getDirtyRanges();
  assert.equal(ranges.length, 1);
  assert.deepEqual(ranges[0], { start: 0, end: 1024, byteLength: 1024, writeCount: 4 });

  const endRes = ring.endBatch({ out: endBatchOut });
  assert.equal(endRes, endBatchOut);
  assert.equal(endRes.uploadedBytes, 1024);
  assert.equal(endRes.writeBufferCalls, 1);
  assert.equal(endRes.coalescedWrites, 3);

  const gpuU32 = new Uint32Array(ring.currentBuffer._data.buffer);
  assert.equal(gpuU32[0], 400);
  assert.equal(gpuU32[256 / 4], 200);
  assert.equal(gpuU32[512 / 4], 300);
  assert.equal(gpuU32[768 / 4], 100);

  // Wrap-around across 2,000 pushes with SharedArrayBuffer source view.
  const sab = new SharedArrayBuffer(16);
  const sabU32 = new Uint32Array(sab);
  for (let i = 0; i < 2000; i += 1) {
    sabU32[0] = 0x5ab00000 + i;
    sabU32[1] = i;
    ring.pushConstantsInto(sabU32, allocOut);
    assert.equal(allocOut.dynamicOffset, (i % 4) * 256);
    if ((i & 3) === 3) {
      ring.flush(endBatchOut);
    }
  }
  const lastGpuU32 = new Uint32Array(ring.currentBuffer._data.buffer);
  assert.equal(lastGpuU32[0], 0x5ab00000 + 1996);
  assert.equal(lastGpuU32[256 / 4], 0x5ab00000 + 1997);
  assert.equal(lastGpuU32[512 / 4], 0x5ab00000 + 1998);
  assert.equal(lastGpuU32[768 / 4], 0x5ab00000 + 1999);

  // coalesceDirtyRanges pure helper preserves exact merging with out-of-order intervals.
  const pureMerged = coalesceDirtyRanges([
    { start: 512, end: 768 },
    { start: 0, end: 256 },
    { start: 256, end: 512 },
    { start: 1024, end: 1280 },
  ]);
  assert.deepEqual(pureMerged, [
    { start: 0, end: 768, byteLength: 768, writeCount: 3 },
    { start: 1024, end: 1280, byteLength: 256, writeCount: 1 },
  ]);

  // Legacy allocate() / pushConstants() / flush() still return distinct objects.
  const a1 = ring.allocate(32);
  const a2 = ring.allocate(32);
  assert.notEqual(a1, a2);
  const f1 = ring.flush();
  const f2 = ring.flush();
  assert.notEqual(f1, f2);

  assert.equal(ring.reset(resetOut), resetOut);
  assert.equal(resetOut.reclaimedPages, 0);
  ring.destroy();
});

// ===========================================================================
// 2. indirect_draw.js zero-alloc hot paths (10,000+ encodings & dispatches)
// ===========================================================================

test('indirect_draw: 12,000+ draw, indexed-draw, and dispatch indirect encodings with byte-exact verification', () => {
  resetIndirectStats();
  const COUNT = 4000; // 4,000 x 3 kinds = 12,000 records!

  const drawBuf = new Uint8Array(COUNT * 16);
  const idxBuf = new Uint8Array(COUNT * 20);
  const dispBuf = new Uint8Array(COUNT * 12);
  const drawU32 = new Uint32Array(COUNT * 4);

  const drawRec = { vertexCount: 0, instanceCount: 1, firstVertex: 0, firstInstance: 0 };
  const idxRec = { indexCount: 0, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0 };
  const dispRec = { x: 1, y: 1, z: 1 };
  const decOut = {};

  for (let i = 0; i < COUNT; i += 1) {
    drawRec.vertexCount = 3 + i * 3;
    drawRec.instanceCount = 1 + (i & 7);
    drawRec.firstVertex = i * 16;
    drawRec.firstInstance = i & 0xff;
    assert.equal(encodeDrawIndirectInto(drawRec, drawBuf, i * 16), 16);
    assert.equal(encodeDrawIndirectInto(drawRec, drawU32, i * 16), 16);

    idxRec.indexCount = 6 + i * 6;
    idxRec.instanceCount = 1 + (i & 3);
    idxRec.firstIndex = i * 12;
    idxRec.baseVertex = (i & 1) === 0 ? i : -i - 1;
    idxRec.firstInstance = (i * 5) & 0xffff;
    assert.equal(encodeDrawIndexedIndirectInto(idxRec, idxBuf, i * 20), 20);

    dispRec.x = 1 + (i % 65535);
    dispRec.y = 1 + (i & 15);
    dispRec.z = 1 + (i & 3);
    assert.equal(encodeDispatchIndirectInto(dispRec, dispBuf, i * 12), 12);
  }

  // Verify bit-exact round-trip across Uint8Array, Uint32Array, and DataView with zero-alloc decode out.
  const idxDv = new DataView(idxBuf.buffer);
  for (let i = 0; i < COUNT; i += 1) {
    const d = decodeDrawIndirect(drawBuf, i * 16, decOut);
    assert.equal(d, decOut);
    assert.equal(d.vertexCount, 3 + i * 3);
    assert.equal(d.instanceCount, 1 + (i & 7));
    assert.equal(d.firstVertex, i * 16);
    assert.equal(d.firstInstance, i & 0xff);

    const du32 = decodeDrawIndirect(drawU32, i * 16, decOut);
    assert.equal(du32.vertexCount, 3 + i * 3);
    assert.equal(du32.firstInstance, i & 0xff);

    const idx = decodeDrawIndexedIndirect(idxDv, i * 20, decOut);
    assert.equal(idx.indexCount, 6 + i * 6);
    assert.equal(idx.instanceCount, 1 + (i & 3));
    assert.equal(idx.firstIndex, i * 12);
    assert.equal(idx.baseVertex, (i & 1) === 0 ? i : -i - 1);
    assert.equal(idx.firstInstance, (i * 5) & 0xffff);

    const disp = decodeDispatchIndirect(dispBuf, i * 12, decOut);
    assert.equal(disp.x, 1 + (i % 65535));
    assert.equal(disp.y, 1 + (i & 15));
    assert.equal(disp.z, 1 + (i & 3));
  }

  // Zero-alloc encodeBatch with caller out, patchedOut, and resultOut.
  const batchRecords = [];
  for (let i = 0; i < 64; i += 1) {
    batchRecords.push({
      indexCount: 36 + i,
      instanceCount: 1,
      firstIndex: i * 36,
      baseVertex: -i,
      firstInstance: i % 4 === 0 ? 7 : 0,
    });
  }
  const batchBytes = new Uint8Array(64 * 24);
  const patchedOut = [];
  const batchResultOut = {};
  const batchOpts = {
    stride: 24,
    patchFirstInstance: true,
    out: batchBytes,
    outOffset: 0,
    patchedOut,
    resultOut: batchResultOut,
  };
  for (let rep = 0; rep < 200; rep += 1) {
    const b = encodeBatch(INDIRECT_KIND.DRAW_INDEXED, batchRecords, batchOpts);
    assert.equal(b, batchResultOut);
    assert.equal(b.bytes, batchBytes);
    assert.equal(b.patched, patchedOut);
    assert.equal(b.patched.length, 16);
  }
  assert.equal(decodeDrawIndexedIndirect(batchBytes, 0, decOut).firstInstance, 0);
  assert.equal(decodeDrawIndexedIndirect(batchBytes, 63 * 24, decOut).baseVertex, -63);

  // Zero-alloc normalizeRecord, validateDispatchCounts, encodeDispatchForItems.
  const normOut = {};
  assert.equal(normalizeRecord(INDIRECT_KIND.DRAW, { vertexCount: 3, instanceCount: 1 }, normOut), normOut);
  assert.equal(normOut.firstVertex, 0);
  const vcOut = {};
  assert.equal(validateDispatchCounts({ x: 100 }, 65535, vcOut), vcOut);
  assert.deepEqual(vcOut, { x: 100, y: 1, z: 1 });
  const itemBuf = new Uint8Array(12);
  assert.equal(encodeDispatchForItems(1000, 64, 65535, itemBuf, 0), itemBuf);
  assert.equal(decodeDispatchIndirect(itemBuf, 0, decOut).x, 16);

  // Bounds/type/range validation still rejects invalid records identically.
  assert.throws(() => encodeDrawIndirectInto({ vertexCount: U32_MAX + 1, instanceCount: 1 }, drawBuf), RangeError);
  assert.throws(() => encodeDrawIndexedIndirectInto({ indexCount: 1, instanceCount: 1, baseVertex: 0x80000000 }, idxBuf), RangeError);
  assert.throws(() => encodeRecord(INDIRECT_KIND.DISPATCH, { x: 1, y: 1, z: 1 }, new Uint8Array(8)), RangeError);
});

test('indirect_draw: IndirectArgsBuffer, multi-draw/dispatch, ExecuteIndirect, and GpuIndirectClampPass hot loops', () => {
  resetIndirectStats();
  const env = createIndirectMockDevice({ recordWrites: true });

  // 1. IndirectArgsBuffer: 4,000 setArgs + out-of-order dirty coalescing + flush + encodeDraws/executeDraws.
  const args = new IndirectArgsBuffer({
    device: env.device,
    kind: INDIRECT_KIND.DRAW_INDEXED,
    capacity: 256,
    mergeGap: 0,
    patchFirstInstance: true,
  });
  const slots = [];
  for (let i = 0; i < 256; i += 1) {
    slots.push(args.allocate({ indexCount: 3 + i, instanceCount: 1, firstIndex: 0, baseVertex: -i, firstInstance: 5 }));
  }
  assert.equal(args.flush(), 1, '256 sequential allocations coalesce into 1 writeBuffer');
  assert.equal(env.writes.length, 1);
  assert.equal(env.writes[0].offset, 0);
  assert.equal(env.writes[0].length, 256 * 20);

  // Out-of-order updates + repeated mutations reuse pooled dirty ranges without allocating new arrays.
  env.writes.length = 0;
  const recScratch = { indexCount: 12, instanceCount: 2, firstIndex: 0, baseVertex: -7, firstInstance: 3 };
  for (let iter = 0; iter < 500; iter += 1) {
    recScratch.indexCount = 12 + iter;
    args.setArgs(200, recScratch);
    args.setArgs(10, recScratch);
    args.setArgs(11, recScratch);
    args.setArgs(201, recScratch);
    const writesIssued = args.flush();
    assert.equal(writesIssued, 2, 'two disjoint 2-slot ranges flush in 2 writeBuffer calls');
  }
  const argOut = {};
  args.getArgs(201, argOut);
  assert.equal(argOut.indexCount, 12 + 499);
  assert.equal(argOut.baseVertex, -7);
  assert.equal(argOut.firstInstance, 0, 'patchFirstInstance zeroed firstInstance');

  let drawIndexedCalls = 0;
  let dispatchIndirectCalls = 0;
  const pass = {
    drawIndirect() {},
    drawIndexedIndirect() {
      drawIndexedCalls += 1;
    },
    dispatchWorkgroups() {},
    dispatchWorkgroupsIndirect() {
      dispatchIndirectCalls += 1;
    },
  };

  const drawOut = {};
  const execOut = {};
  const planOut = {};
  const valBatchOut = {};
  const batchDispatchOut = {};
  const multiDispOut = {};

  for (let i = 0; i < 200; i += 1) {
    assert.equal(args.encodeDraws(pass, { count: 16, out: drawOut }), drawOut);
    assert.equal(drawOut.draws, 16);
    assert.equal(args.executeDraws(pass, { count: 16, out: execOut }), execOut);
    assert.equal(execOut.executed, 16);
  }
  assert.equal(drawIndexedCalls, 200 * 32);

  // Direct hot loop on dispatchIndirectDraw / dispatchIndexedIndirectDraw / executeIndirect / encodeMultiDispatchIndirect.
  const gpuBuf = { usage: BUFFER_USAGE.INDIRECT | BUFFER_USAGE.STORAGE, size: 4096 };
  const singleOut = {};
  for (let i = 0; i < 4000; i += 1) {
    const offset = (i % 100) * 20;
    assert.equal(dispatchIndexedIndirectDraw(pass, gpuBuf, offset, null, singleOut), singleOut);
    assert.equal(singleOut.dispatched, true);
    assert.equal(executeIndirect(pass, INDIRECT_KIND.DISPATCH, gpuBuf, (i % 100) * 12, null, execOut), execOut);
    assert.equal(execOut.executed, 1);
  }
  assert.equal(dispatchIndirectCalls, 4000);

  // Multi-draw / multi-dispatch / step plan / batch validation with out targets.
  assert.equal(
    computeIndirectBatchStepPlan(gpuBuf, { kind: INDIRECT_KIND.DRAW_INDEXED, count: 10 }, planOut),
    planOut
  );
  assert.equal(planOut.valid, true);
  assert.equal(planOut.effectiveCount, 10);

  assert.equal(
    validateIndirectBatch(gpuBuf, { kind: INDIRECT_KIND.DISPATCH, count: 8 }, valBatchOut),
    valBatchOut
  );
  assert.equal(valBatchOut.requiredBytes, 8 * 12);

  assert.equal(
    encodeMultiDraw(pass, gpuBuf, { indexed: true, count: 4 }, drawOut),
    drawOut
  );
  assert.equal(drawOut.draws, 4);

  assert.equal(
    encodeMultiDispatchIndirect(pass, gpuBuf, { count: 4 }, multiDispOut),
    multiDispOut
  );
  assert.equal(multiDispOut.dispatches, 4);

  assert.equal(
    dispatchIndirectBatch(pass, gpuBuf, { indexed: true, count: 4 }, batchDispatchOut),
    batchDispatchOut
  );
  assert.equal(batchDispatchOut.dispatched, 4);

  // encodeDispatches without Array.forEach closure.
  let directWg = 0;
  const directPass = { dispatchWorkgroups: (x) => { directWg += x; } };
  encodeDispatches(directPass, [{ x: 10, y: 1, z: 1 }, { x: 5, y: 1, z: 1 }]);
  assert.equal(directWg, 15);

  // 2. GpuIndirectClampPass: 2,000 hot clamp passes across all 4 kinds hitting pipeline & direct-ref bind-group caches.
  const clampPass = new GpuIndirectClampPass({ device: env.device });
  const srcBuf = env.device.createBuffer({
    size: 4096,
    usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.INDIRECT,
  });
  const dstBuf = env.device.createBuffer({
    size: 4096,
    usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.INDIRECT,
  });
  const computePass = env.device.createCommandEncoder().beginComputePass();
  const clampOpts = {
    srcBuffer: srcBuf,
    srcOffset: 0,
    dstBuffer: dstBuf,
    dstOffset: 0,
    commandCount: 4,
    maxVertexCount: 1024,
    maxIndexCount: 2048,
    maxInstanceCount: 64,
  };
  const clampOut = {};
  const kinds = [
    INDIRECT_CLAMP_KIND.DRAW,
    INDIRECT_CLAMP_KIND.DRAW_INDEXED,
    INDIRECT_CLAMP_KIND.DISPATCH,
    INDIRECT_CLAMP_KIND.DISPATCH_MESH,
  ];
  for (let k = 0; k < kinds.length; k += 1) {
    for (let i = 0; i < 500; i += 1) {
      assert.equal(clampPass.encodeClamp(computePass, kinds[k], clampOpts, clampOut), clampOut);
      assert.equal(clampOut.ok, true);
      assert.equal(clampOut.encoded, true);
    }
  }
  const cStats = clampPass.getStats();
  assert.equal(cStats.gpuClampPassesEncoded, 2000);
  assert.equal(cStats.gpuClampPipelinesCreated, 4, 'only 4 pipelines created across 2000 passes');
  assert.equal(cStats.gpuClampPipelineCacheHits, 1996);
  assert.equal(env.getCreateBindGroupCalls(), 4, 'only 4 bind groups created (1 per kind)');
  assert.equal(cStats.bindGroupCacheHits, 1996);

  // 3. encodeMeshComputeThenDraw multi-draw (drawCount > 1) zero-alloc batch options scratch.
  const meshOut = {};
  const encoder = env.device.createCommandEncoder();
  const meshRes = encodeMeshComputeThenDraw(
    encoder,
    {
      device: env.device,
      clampExecutor: clampPass,
      meshPipeline: { id: 1 },
      renderPipeline: { id: 2 },
      workgroupCountX: 4,
      indexed: true,
      drawCount: 3,
      drawIndirectBuffer: srcBuf,
      drawIndirectOffset: 0,
      clampedDrawBuffer: dstBuf,
      clampedDrawOffset: 0,
    },
    meshOut
  );
  assert.equal(meshRes, meshOut);
  assert.equal(meshRes.ok, true);
  assert.equal(meshRes.drawsIssued, 3);

  assert.ok(getIndirectStats().indirectIndexedDrawsEncoded > 10000);
  clampPass.destroy();
  args.destroy();
});

// ===========================================================================
// 3. texture_packets.js zero-alloc hot paths, word-copy repack, #2094 SAB, #2114 COPY_SRC
// ===========================================================================

test('texture_packets: thousands of direct, word-repacked, small-row, sub-region, and BCn uploads are bit-exact', () => {
  const gpu = createTextureMockGpu({ recordSnapshots: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  processor.stagingPool = new GpuStagingPool({
    device: gpu.device,
    chunkSize: 4 * 1024 * 1024,
  });
  const handler = installTexturePackets(processor);

  // Create textures via CreateTexture2 (v2) and verify #2114 COPY_SRC + sRGB sibling viewFormats.
  // 1) r8unorm (code 1) 1920x64 (unaligned row pitch 1920 -> 2048, hits multi-row 4B-aligned word-copy fast path!)
  // 2) rgba8unorm (code 22) 64x64 (256B aligned direct fast path)
  // 3) r8unorm (code 1) 300x12 (unaligned 300B -> 512B, exercises 4B-aligned word-copy)
  // 4) r8unorm (code 1) 19x8 (odd unaligned <=64B, exercises direct byte-copy fast path)
  // 5) bc7-rgba-unorm (code 62) 16x16 with 5 mips (tests 4x4 tail mips + clamping)
  processor.executeCommandBuffer(buildCreateTexture2Packet({ id: 1, width: 1920, height: 64, format: 1, usage: 0x06 }));
  processor.executeCommandBuffer(buildCreateTexture2Packet({ id: 2, width: 64, height: 64, format: 22, usage: 0x06 }));
  processor.executeCommandBuffer(buildCreateTexture2Packet({ id: 3, width: 300, height: 12, format: 1, usage: 0x06 }));
  processor.executeCommandBuffer(buildCreateTexture2Packet({ id: 4, width: 19, height: 8, format: 1, usage: 0x06 }));
  processor.executeCommandBuffer(buildCreateTexture2Packet({ id: 5, width: 16, height: 16, mips: 5, format: 62, usage: 0x06 }));

  // #2114: every created texture must include COPY_SRC (0x01).
  assert.equal(gpu.createdTextures.length, 5);
  for (const tex of gpu.createdTextures) {
    assert.notEqual(tex.usage & 0x01, 0, 'COPY_SRC must be set on every created texture (#2114)');
  }
  assert.deepEqual(gpu.createdTextures[1].viewFormats, ['rgba8unorm-srgb']);
  assert.deepEqual(gpu.createdTextures[4].viewFormats, ['bc7-rgba-unorm-srgb']);

  // Verify resolveCompatibleViewFormats without Set allocation handles duplicates and STORAGE_BINDING.
  assert.deepEqual(
    resolveCompatibleViewFormats('rgba8unorm', ['rgba8unorm', 'rgba8unorm', 'bgra8unorm', 'rgba8unorm-srgb']),
    { viewFormats: ['rgba8unorm', 'rgba8unorm-srgb'], siblingAdded: false }
  );
  assert.deepEqual(
    resolveCompatibleViewFormats('rgba8unorm', ['rgba8unorm-srgb'], { hasStorageBinding: true }),
    { viewFormats: undefined, siblingAdded: false }
  );

  // 1) 1920x64 r8unorm word-copy repacking (1920 -> 2048): bit-exact check + repeated scratch reuse!
  const wideSrc = new Uint8Array(1920 * 64);
  for (let y = 0; y < 64; y += 1) {
    for (let x = 0; x < 1920; x += 1) {
      wideSrc[y * 1920 + x] = (x * 7 + y * 13 + 1) & 0xff;
    }
  }
  const widePkt = buildWriteTexturePacket({
    textureId: 1,
    width: 1920,
    height: 64,
    bytesPerRow: 1920,
    rowsPerImage: 64,
    data: wideSrc,
  });
  processor.executeCommandBuffer(widePkt);
  processor._flushPendingUploads(false);
  assert.equal(gpu.copyBufferToTextureCalls.length, 1);
  const wideCall = gpu.copyBufferToTextureCalls[0];
  assert.equal(wideCall.source.bytesPerRow, 2048);
  assert.equal(wideCall.bytes.byteLength, 2048 * 64);
  for (let y = 0; y < 64; y += 1) {
    const dstRow = y * 2048;
    const srcRow = y * 1920;
    for (let x = 0; x < 1920; x += 1) {
      assert.equal(wideCall.bytes[dstRow + x], wideSrc[srcRow + x], `mismatch at (${x},${y})`);
    }
    for (let p = 1920; p < 2048; p += 1) {
      assert.equal(wideCall.bytes[dstRow + p], 0, `non-zero pad at (${p},${y})`);
    }
  }

  // Dirty scratch buffer with 0xff to verify reused-scratch zero-padding works identically on subsequent repacks!
  handler._scratchBuffer.fill(0xff);

  // 2) Small odd-width 19x8 r8unorm (<=64B byte-copy fast path) + 300x12 r8unorm (word-copy).
  const oddSrc = new Uint8Array(19 * 8);
  for (let i = 0; i < oddSrc.length; i += 1) oddSrc[i] = (i * 29 + 3) & 0xff;
  processor.executeCommandBuffer(
    buildWriteTexturePacket({ textureId: 4, width: 19, height: 8, bytesPerRow: 19, rowsPerImage: 8, data: oddSrc })
  );
  const midSrc = new Uint8Array(300 * 12);
  for (let i = 0; i < midSrc.length; i += 1) midSrc[i] = (i * 11 + 5) & 0xff;
  processor.executeCommandBuffer(
    buildWriteTexturePacket({ textureId: 3, width: 300, height: 12, bytesPerRow: 300, rowsPerImage: 12, data: midSrc })
  );
  // BC7 tail mip level 4 (logical 1x1 -> physical 4x4 = 16 bytes) + sub-region clamping on mip 0!
  const bc7Tail = new Uint8Array(16);
  for (let i = 0; i < 16; i += 1) bc7Tail[i] = 0xa0 + i;
  processor.executeCommandBuffer(
    buildWriteTexturePacket({ textureId: 5, mipLevel: 4, width: 1, height: 1, bytesPerRow: 16, rowsPerImage: 1, data: bc7Tail })
  );
  // Out-of-bounds sub-region on rgba8unorm 64x64: origin (48, 48), size (32, 32) -> clamped to (16, 16)
  const subSrc = new Uint8Array(32 * 32 * 4).fill(0x42);
  processor.executeCommandBuffer(
    buildWriteTexturePacket({
      textureId: 2,
      originX: 48,
      originY: 48,
      width: 32,
      height: 32,
      bytesPerRow: 128,
      rowsPerImage: 32,
      data: subSrc,
    })
  );
  processor._flushPendingUploads(false);

  const oddCall = gpu.copyBufferToTextureCalls[1];
  assert.equal(oddCall.source.bytesPerRow, 256);
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 19; x += 1) {
      assert.equal(oddCall.bytes[y * 256 + x], oddSrc[y * 19 + x]);
    }
    for (let p = 19; p < 256; p += 1) {
      assert.equal(oddCall.bytes[y * 256 + p], 0, 'reused scratch padding must be zeroed');
    }
  }

  const midCall = gpu.copyBufferToTextureCalls[2];
  assert.equal(midCall.source.bytesPerRow, 512);
  for (let y = 0; y < 12; y += 1) {
    for (let x = 0; x < 300; x += 1) {
      assert.equal(midCall.bytes[y * 512 + x], midSrc[y * 300 + x]);
    }
    for (let p = 300; p < 512; p += 1) {
      assert.equal(midCall.bytes[y * 512 + p], 0, 'reused scratch word-copy padding must be zeroed');
    }
  }

  const bc7Call = gpu.copyBufferToTextureCalls[3];
  assert.deepEqual(bc7Call.copySize, { width: 4, height: 4, depthOrArrayLayers: 1 });
  assert.deepEqual(Array.from(bc7Call.bytes.subarray(0, 16)), Array.from(bc7Tail));

  const clampedCall = gpu.copyBufferToTextureCalls[4];
  assert.deepEqual(clampedCall.copySize, { width: 16, height: 16, depthOrArrayLayers: 1 });
  assert.equal(handler.getStats().clampedSubRegionCount, 1);

  // Heavy synthetic hot loop: 2,500 direct + staged/repacked uploads verifying telemetry & zero corruption!
  const gpuFast = createTextureMockGpu({ recordSnapshots: false });
  const procFast = new WgpuCommandProcessor(gpuFast.device, gpuFast.ctx);
  procFast.stagingPool = new GpuStagingPool({ device: gpuFast.device, chunkSize: 8 * 1024 * 1024 });
  const handlerFast = installTexturePackets(procFast);
  procFast.executeCommandBuffer(buildCreateTexture2Packet({ id: 10, width: 64, height: 16, format: 22, usage: 0x06 })); // 256B aligned direct
  procFast.executeCommandBuffer(buildCreateTexture2Packet({ id: 11, width: 300, height: 16, format: 1, usage: 0x06 })); // repacked staged

  const directPkt = buildWriteTexturePacket({
    textureId: 10,
    width: 64,
    height: 16,
    bytesPerRow: 256,
    rowsPerImage: 16,
    data: new Uint8Array(256 * 16).fill(0x5a),
  });
  const repackPkt = buildWriteTexturePacket({
    textureId: 11,
    width: 300,
    height: 16,
    bytesPerRow: 300,
    rowsPerImage: 16,
    data: new Uint8Array(300 * 16).fill(0x3c),
  });

  const directView = new DataView(directPkt.buffer, directPkt.byteOffset, directPkt.byteLength);
  const repackView = new DataView(repackPkt.buffer, repackPkt.byteOffset, repackPkt.byteLength);
  for (let i = 0; i < 1250; i += 1) {
    handlerFast.handlePacket(OPCODES.WRITE_TEXTURE, directView, 8, directPkt.byteLength - 8);
    handlerFast.handlePacket(OPCODES.WRITE_TEXTURE, repackView, 8, repackPkt.byteLength - 8);
    if ((i & 31) === 31) {
      procFast._flushPendingUploads(false);
      procFast.stagingPool.beginFrame();
    }
  }
  procFast._flushPendingUploads(false);

  const fastStats = handlerFast.getStats();
  assert.equal(fastStats.uploadCount, 2500);
  assert.equal(fastStats.directZeroCopyUploads, 1250);
  assert.equal(fastStats.repackedUploads, 1250);
  assert.equal(fastStats.scratchBufferAllocations, 1, 'scratch buffer allocated once and reused 1249 times');
  assert.equal(fastStats.scratchBufferReuses, 1249);
  assert.equal(gpuFast.getWriteTextureCount(), 1250);
  assert.equal(gpuFast.getCopyBufferToTextureCount(), 1250);

  // Scratch pool acquireInto & helper `out` options.
  const pool = new TextureScratchPool({ maxBytes: 65536 });
  const acqOut = {};
  const repackOut = {};
  const copyOut = {};
  const a8Out = {};
  const src16 = new Uint8Array(16 * 4);
  for (let i = 0; i < src16.length; i += 1) src16[i] = i;
  for (let i = 0; i < 100; i += 1) {
    assert.equal(pool.acquireInto(acqOut, 1024), acqOut);
    assert.equal(repackRowsTo256(src16, 16, 4, 1, { scratchPool: pool, out: repackOut }), repackOut);
    assert.equal(repackOut.bytesPerRow, 256);
    assert.equal(copyFromSharedIfNeeded(src16, { scratchPool: pool, out: copyOut }), copyOut);
    assert.equal(expandA8ToRgba8(src16, 16, 4, { scratchPool: pool, out: a8Out }), a8Out);
    assert.equal(a8Out.bytesPerRow, 256);
    assert.equal(a8Out.data[3], 0);
    assert.equal(a8Out.data[7], 1);
  }
  assert.equal(pool.getStats().allocations, 1);

  // writeTexture & writeA8Texture zero-alloc `out` targets.
  const wtOut = {};
  const wa8Out = {};
  const tex2 = processor.textures.get(2);
  assert.equal(
    handler.writeTexture({ texture: tex2 }, new Uint8Array(64 * 64 * 4), { bytesPerRow: 256, rowsPerImage: 64 }, { width: 64, height: 64 }, 2, wtOut),
    wtOut
  );
  assert.equal(wtOut.uploaded, true);
  assert.equal(
    handler.writeA8Texture({ texture: tex2 }, new Uint8Array(16 * 4).fill(0x88), { width: 16, height: 4 }, { preferDirect: true, out: wa8Out }),
    wa8Out
  );
  assert.equal(wa8Out.uploaded, true);
  assert.equal(wa8Out.bytesPerRow, 64);

  processor.destroy();
  procFast.destroy();
});

test('texture_packets: preserves #2094 direct SharedArrayBuffer queue.writeTexture fast path and legacy fallback', () => {
  // Modern WebGPU: accepts SharedArrayBuffer views directly with zero copies.
  const modernGpu = createTextureMockGpu({ rejectSabWriteTexture: false, recordSnapshots: true });
  const modernProc = new WgpuCommandProcessor(modernGpu.device, modernGpu.ctx);
  const modernHandler = installTexturePackets(modernProc);
  modernProc.executeCommandBuffer(buildCreateTexture2Packet({ id: 1, width: 64, height: 8, format: 22, usage: 0x06 }));

  const payload = new Uint8Array(64 * 8 * 4);
  for (let i = 0; i < payload.length; i += 1) payload[i] = (i * 3 + 7) & 0xff;
  const sabPkt = buildWriteTexturePacket({
    textureId: 1,
    width: 64,
    height: 8,
    bytesPerRow: 256,
    rowsPerImage: 8,
    data: payload,
    useSab: true,
  });

  for (let i = 0; i < 200; i += 1) {
    modernProc.executeCommandBuffer(sabPkt);
  }
  assert.equal(modernGpu.getWriteTextureCount(), 200);
  assert.equal(modernGpu.writeTextureCalls[0].isSab, true, '#2094 direct SAB write preserved');
  assert.equal(modernGpu.writeTextureCalls[0].byteOffset, 56);
  assert.equal(modernGpu.writeTextureCalls[0].byteLength, payload.byteLength);
  assert.deepEqual(Array.from(modernGpu.writeTextureCalls[0].bytes), Array.from(payload));
  assert.equal(modernHandler.getStats().sabCopyCount, 0);
  assert.equal(modernHandler.getStats().directZeroCopyUploads, 200);

  // Legacy WebGPU: rejects SAB with TypeError on first probe, caches fallback, copies via pooled scratch.
  const legacyGpu = createTextureMockGpu({ rejectSabWriteTexture: true, recordSnapshots: true });
  const legacyProc = new WgpuCommandProcessor(legacyGpu.device, legacyGpu.ctx);
  const legacyHandler = installTexturePackets(legacyProc);
  legacyProc.executeCommandBuffer(buildCreateTexture2Packet({ id: 1, width: 64, height: 8, format: 22, usage: 0x06 }));

  for (let i = 0; i < 200; i += 1) {
    legacyProc.executeCommandBuffer(sabPkt);
  }
  assert.equal(legacyGpu.getWriteTextureCount(), 200);
  assert.equal(legacyGpu.writeTextureCalls[0].isSab, false);
  assert.deepEqual(Array.from(legacyGpu.writeTextureCalls[0].bytes), Array.from(payload));
  assert.equal(legacyHandler.getStats().sabCopyCount, 200);
  assert.equal(legacyHandler.getStats().scratchBufferAllocations, 1);
  assert.equal(legacyHandler.getStats().scratchBufferReuses, 199);

  modernProc.destroy();
  legacyProc.destroy();
});

test('classic worker importScripts IIFE isolation holds across uniform_ring, indirect_draw, and texture_packets', () => {
  const sandbox = {
    self: null,
    console,
    Uint8Array,
    Uint8ClampedArray,
    Uint16Array,
    Uint32Array,
    Int32Array,
    Float32Array,
    ArrayBuffer,
    SharedArrayBuffer,
    DataView,
    Map,
    Set,
    WeakMap,
    WeakSet,
    Object,
    Array,
    Math,
    Number,
    String,
    Boolean,
    Error,
    TypeError,
    RangeError,
    Symbol,
  };
  sandbox.self = sandbox;
  sandbox.globalThis = sandbox;
  const ctx = vm.createContext(sandbox);

  for (const rel of [
    '../web/workers/uniform_ring.js',
    '../web/workers/indirect_draw.js',
    '../web/workers/texture_packets.js',
  ]) {
    const src = readFileSync(new URL(rel, import.meta.url), 'utf8');
    vm.runInContext(src, ctx, { filename: rel });
  }

  assert.equal(typeof ctx.PlayGta6UniformRing.DynamicUniformRing, 'function');
  assert.equal(typeof ctx.PlayGta6IndirectDraw.encodeDrawIndirectInto, 'function');
  assert.equal(typeof ctx.PlayGta6TexturePackets.TexturePackets, 'function');
});
