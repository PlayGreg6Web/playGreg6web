import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, OPCODES } = require('../web/workers/wgpu_worker.js');
const { installRenderStatePackets, TEXTURE_FORMAT_NAMES: RS_FORMATS } = require('../web/workers/render_state_packets.js');
const { GpuStagingPool } = require('../web/workers/staging_pool.js');
const tf = require('../web/workers/texture_formats.js');
const tp = require('../web/workers/texture_packets.js');

const WORKERS_DIR = fileURLToPath(new URL('../web/workers/', import.meta.url));
const repoFile = (rel) => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');

function createMockWebGpu({ strictWebGpu = false, features = undefined } = {}) {
  const calls = [];
  const buffers = new Map();
  let nextHandle = 1;

  const device = {
    ...(features !== undefined ? { features } : {}),
    createBuffer(desc) {
      const backing = new Uint8Array(desc.size);
      const buffer = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        desc,
        backing,
        destroyed: false,
        destroy() {
          buffer.destroyed = true;
        },
      };
      buffers.set(buffer.handle, buffer);
      calls.push({ op: 'createBuffer', desc, result: buffer });
      return buffer;
    },
    createTexture(desc) {
      if (desc.format === 'bc7-rgba-unorm' && desc.sampleCount > 1) {
        throw new Error('not multisampleable');
      }
      if (
        strictWebGpu &&
        desc.dimension === '3d' &&
        typeof desc.format === 'string' &&
        desc.format.startsWith('bc') &&
        !(features && typeof features.has === 'function' && features.has('texture-compression-bc-sliced-3d'))
      ) {
        throw new Error(
          `WebGPU validation: 3d texture format ${desc.format} requires texture-compression-bc-sliced-3d`
        );
      }
      const texDimension = desc.dimension || '2d';
      const texWidth = Array.isArray(desc.size) ? desc.size[0] : desc.size && desc.size.width;
      const texHeight = Array.isArray(desc.size)
        ? (desc.size[1] !== undefined ? desc.size[1] : 1)
        : (desc.size && desc.size.height !== undefined ? desc.size.height : 1);
      const texDepth = Array.isArray(desc.size)
        ? (desc.size[2] !== undefined ? desc.size[2] : 1)
        : (desc.size && desc.size.depthOrArrayLayers !== undefined ? desc.size.depthOrArrayLayers : 1);
      const texture = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        format: desc.format,
        dimension: texDimension,
        width: texWidth,
        height: texHeight,
        depthOrArrayLayers: texDepth,
        mipLevelCount: desc.mipLevelCount || 1,
        desc,
        destroyed: false,
        createView(viewDesc) {
          const requestedViewDim =
            viewDesc && typeof viewDesc.dimension === 'string' ? viewDesc.dimension : null;
          if (strictWebGpu && requestedViewDim) {
            const validDimsByTexDim = {
              '1d': ['1d'],
              '2d': ['2d', '2d-array', 'cube', 'cube-array'],
              '3d': ['3d'],
            };
            const compatible = validDimsByTexDim[texDimension] || ['2d'];
            if (!compatible.includes(requestedViewDim)) {
              throw new Error(
                `WebGPU validation: The dimension (${requestedViewDim}) of the texture view is not compatible with the dimension (${texDimension}) of texture (${texWidth} px, ${desc.format})`
              );
            }
          }
          if (
            strictWebGpu &&
            viewDesc &&
            viewDesc.format &&
            viewDesc.format !== desc.format
          ) {
            const allowed = Array.isArray(desc.viewFormats) ? desc.viewFormats : [];
            if (!allowed.includes(viewDesc.format)) {
              throw new Error(
                `WebGPU validation: texture view format ${viewDesc.format} is not in texture viewFormats for ${desc.format}`
              );
            }
          }
          const resolvedDimension =
            requestedViewDim ||
            (texDimension === '1d'
              ? '1d'
              : texDimension === '3d'
                ? '3d'
                : texDepth > 1
                  ? '2d-array'
                  : '2d');
          return {
            kind: 'GPUTextureView',
            textureHandle: texture.handle,
            dimension: resolvedDimension,
            viewDesc,
          };
        },
        destroy() {
          texture.destroyed = true;
        },
      };
      calls.push({ op: 'createTexture', desc, result: texture });
      return texture;
    },
    createCommandEncoder() {
      const encoder = {
        beginRenderPass: () => ({ end() {}, setPipeline() {}, draw() {} }),
        beginComputePass: () => ({ end() {} }),
        copyBufferToBuffer(srcBuf, srcOff, dstBuf, dstOff, size) {
          if (strictWebGpu && size % 4 !== 0) {
            throw new Error(`WebGPU validation: copyBufferToBuffer size (${size}) not 4B aligned`);
          }
          if (srcBuf.backing && dstBuf.backing) {
            dstBuf.backing.set(srcBuf.backing.subarray(srcOff, srcOff + size), dstOff);
          }
          calls.push({ op: 'copyBufferToBuffer', srcBuf, srcOff, dstBuf, dstOff, size });
        },
        copyBufferToTexture(source, destination, copySize) {
          if (
            strictWebGpu &&
            source.bytesPerRow !== undefined &&
            source.bytesPerRow % 256 !== 0
          ) {
            throw new Error(
              `WebGPU validation: copyBufferToTexture bytesPerRow (${source.bytesPerRow}) must be a multiple of 256`
            );
          }
          const snapshot =
            source.buffer && source.buffer.backing
              ? source.buffer.backing.slice(source.offset || 0)
              : null;
          calls.push({
            op: 'copyBufferToTexture',
            source: { ...source },
            destination: {
              ...destination,
              origin: destination.origin ? { ...destination.origin } : destination.origin,
            },
            copySize: { ...copySize },
            sourceRef: source,
            destinationRef: destination,
            originRef: destination.origin,
            copySizeRef: copySize,
            bytes: snapshot,
          });
        },
        finish: () => ({ kind: 'GPUCommandBuffer' }),
      };
      return encoder;
    },
    queue: {
      writeBuffer(buffer, bufferOffset, data) {
        if (
          strictWebGpu &&
          typeof SharedArrayBuffer !== 'undefined' &&
          data &&
          data.buffer instanceof SharedArrayBuffer
        ) {
          throw new TypeError('WebGPU validation: SharedArrayBuffer view disallowed in writeBuffer');
        }
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        if (buffer && buffer.backing) {
          buffer.backing.set(bytes, bufferOffset);
        }
        calls.push({
          op: 'writeBuffer',
          buffer,
          bufferOffset,
          byteLength: bytes.byteLength,
          isSab: typeof SharedArrayBuffer !== 'undefined' && data.buffer instanceof SharedArrayBuffer,
          bytes: bytes.slice(),
        });
      },
      writeTexture(destination, data, dataLayout, size) {
        if (
          strictWebGpu &&
          typeof SharedArrayBuffer !== 'undefined' &&
          data &&
          data.buffer instanceof SharedArrayBuffer
        ) {
          throw new TypeError('WebGPU validation: SharedArrayBuffer view disallowed in writeTexture');
        }
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        calls.push({
          op: 'writeTexture',
          destination: {
            ...destination,
            origin: destination.origin ? { ...destination.origin } : destination.origin,
          },
          dataLayout: { ...dataLayout },
          size: { ...size },
          destinationRef: destination,
          originRef: destination.origin,
          dataLayoutRef: dataLayout,
          sizeRef: size,
          isSab: typeof SharedArrayBuffer !== 'undefined' && data.buffer instanceof SharedArrayBuffer,
          dataRef: data,
          bufferRef: data.buffer,
          byteOffset: data.byteOffset,
          byteLength: data.byteLength,
          bytes: bytes.slice(),
        });
      },
      submit() {
        calls.push({ op: 'submit' });
      },
    },
  };
  const ctx = { getCurrentTexture: () => ({ createView: () => ({ kind: 'swapchain' }) }) };
  return { device, ctx, calls };
}

function createTexture1Packet({
  id,
  width,
  height,
  layers = 1,
  mips = 1,
  format = 0,
  usage = 0x06,
  dimension = 1,
}) {
  const out = new Uint8Array(8 + 32);
  const view = new DataView(out.buffer);
  view.setUint16(0, OPCODES.CREATE_TEXTURE, true);
  view.setUint32(4, 32, true);
  const p = 8;
  view.setUint32(p, id, true);
  view.setUint32(p + 4, width, true);
  view.setUint32(p + 8, height, true);
  view.setUint32(p + 12, layers, true);
  view.setUint32(p + 16, mips, true);
  view.setUint32(p + 20, format, true);
  view.setUint32(p + 24, usage, true);
  view.setUint32(p + 28, dimension, true);
  return out;
}

function createTexture2Packet({
  id,
  width,
  height,
  layers = 1,
  mips = 1,
  samples = 1,
  usage = 0x14,
  dimension = 1,
  format,
  viewFormats = [],
}) {
  const out = new Uint8Array(8 + 40);
  const view = new DataView(out.buffer);
  view.setUint16(0, 35, true);
  view.setUint32(4, 40, true);
  const p = 8;
  view.setUint32(p, id, true);
  view.setUint32(p + 4, width, true);
  view.setUint32(p + 8, height, true);
  view.setUint32(p + 12, layers, true);
  view.setUint32(p + 16, mips, true);
  view.setUint32(p + 20, samples, true);
  view.setUint32(p + 24, usage, true);
  view.setUint8(p + 28, dimension);
  view.setUint8(p + 29, format);
  view.setUint8(p + 30, viewFormats.length);
  viewFormats.forEach((f, i) => view.setUint8(p + 32 + i, f));
  return out;
}

function writeTexturePacket({
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
  const view = new DataView(backing);
  view.setUint16(0, OPCODES.WRITE_TEXTURE, true);
  view.setUint32(4, 48 + data.byteLength, true);
  const p = 8;
  view.setUint32(p, textureId, true);
  view.setUint32(p + 4, mipLevel, true);
  view.setUint32(p + 8, originX, true);
  view.setUint32(p + 12, originY, true);
  view.setUint32(p + 16, originZ, true);
  view.setUint32(p + 20, width, true);
  view.setUint32(p + 24, height, true);
  view.setUint32(p + 28, layers, true);
  view.setUint32(p + 32, bytesPerRow, true);
  view.setUint32(p + 36, rowsPerImage, true);
  view.setUint32(p + 40, data.byteLength, true);
  view.setUint32(p + 44, 0, true);
  out.set(data, p + 48);
  return out;
}

function destroyPacket(id) {
  const out = new Uint8Array(12);
  const view = new DataView(out.buffer);
  view.setUint16(0, OPCODES.DESTROY_RESOURCE, true);
  view.setUint32(4, 4, true);
  view.setUint32(8, id, true);
  return out;
}

test('CreateTexture2 opcode and payload size match the ABI header', () => {
  assert.equal(OPCODES.CREATE_TEXTURE2, 35);
  assert.equal(tp.CREATE_TEXTURE2, 35);
  assert.equal(tp.CREATE_TEXTURE2_BYTES, 40);
  assert.equal(tp.CREATE_TEXTURE, 12);
  assert.equal(tp.WRITE_TEXTURE, 13);
  assert.equal(tp.WEBGPU_BYTES_PER_ROW_ALIGNMENT, 256);
  const header = repoFile('runtime/abi/playgta6_abi.h');
  assert.match(header, /CreateTexture2 = 35,/);
  assert.match(header, /sizeof\(GpuCreateTexture2Payload\) == 40/);
  // Same v2 format code table as the render state handler (docs/gpu_commands.md).
  assert.deepEqual([...tp.TEXTURE_FORMAT_NAMES], [...RS_FORMATS]);
});

test('CreateTexture2 creates a texture with format, samples and view formats in the shared tables', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  installRenderStatePackets(processor);
  const handler = tp.installTexturePackets(processor);

  // rgba16float (40) 4x MSAA render target; bgra8unorm (27) with an sRGB view format (28).
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 7, width: 640, height: 360, samples: 4, usage: 0x10, format: 40 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 8,
      width: 256,
      height: 128,
      layers: 6,
      mips: 9,
      usage: 0x06,
      format: 27,
      viewFormats: [28],
    })
  );
  assert.deepEqual(handler.getStats(), {
    created: 2,
    errors: 0,
    createTexture2Count: 2,
    rowPitchRepackCount: 0,
    sabCopyCount: 0,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 0,
    uploadsCount: 0,
    uploadBytes: 0,
    uploadBytesTotal: 0,
    reusedDescriptorCount: 0,
    clampedSubRegionCount: 0,
    zeroCopyUploads: 0,
    scratchRepackUploads: 0,
    directZeroCopyUploads: 0,
    repackedUploads: 0,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 0,
    scratchBufferBytes: 0,
    r8unormUploads: 0,
    bytesUploaded: 0,
    srgbSiblingViewFormatsAdded: 0,
    promoted1DTo2DCount: 0,
    chunkedWrites: 0,
    trimmedStagingBytes: 0,
    destroyedTextures: 0,
  });
  const [rt, tex] = gpu.calls.filter((c) => c.op === 'createTexture').map((c) => c.desc);
  assert.deepEqual(rt, {
    size: { width: 640, height: 360, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 4,
    dimension: '2d',
    format: 'rgba16float',
    usage: 0x10 | 0x01,
  });
  assert.equal(tex.format, 'bgra8unorm');
  assert.deepEqual(tex.viewFormats, ['bgra8unorm-srgb']);
  assert.equal(tex.size.depthOrArrayLayers, 6);
  assert.equal(tex.mipLevelCount, 9);
  assert.ok(processor.textures.get(7));
  assert.ok(processor.textureViews.get(8));

  // Re-creating an id replaces (and destroys) the previous texture; DestroyResource removes it.
  const first = processor.textures.get(7);
  processor.executeCommandBuffer(createTexture2Packet({ id: 7, width: 64, height: 64, format: 22 }));
  assert.equal(first.destroyed, true);
  assert.equal(processor.textures.get(7).desc.format, 'rgba8unorm');
  assert.equal(handler.stats.createTexture2Count, 3);
  processor.executeCommandBuffer(destroyPacket(7));
  assert.equal(processor.textures.has(7), false);
});

test('invalid CreateTexture2 payloads are counted and leave the id unbound', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);
  processor.executeCommandBuffer(createTexture2Packet({ id: 1, width: 4, height: 4, format: 0 }));
  processor.executeCommandBuffer(createTexture2Packet({ id: 2, width: 4, height: 4, format: 200 }));
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 3, width: 4, height: 4, format: 22, dimension: 9 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 4, width: 4, height: 4, format: 62, samples: 4 })
  );
  assert.equal(handler.stats.errors, 4);
  assert.equal(handler.stats.createTexture2Count, 0);
  assert.match(handler.lastError, /not multisampleable/);
  for (const id of [1, 2, 3, 4]) assert.equal(processor.textures.has(id), false);

  // Truncated payload decoded directly returns a clean error instead of throwing RangeError.
  const truncated = tp.decodeCreateTexture2(new DataView(new ArrayBuffer(20)), 0);
  assert.match(truncated.error, /payload 20 < 40 bytes/);
});

test('the texture handler chains other opcodes to the previously installed extension', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const seen = [];
  processor.extension = {
    handlePacket: (opcode) => seen.push(['packet', opcode]),
    destroyResource: (id) => seen.push(['destroy', id]),
  };
  tp.installTexturePackets(processor);
  const unknown = new Uint8Array(8);
  new DataView(unknown.buffer).setUint16(0, 77, true);
  processor.executeCommandBuffer(unknown);
  processor.executeCommandBuffer(destroyPacket(5));
  assert.deepEqual(seen, [
    ['packet', 77],
    ['destroy', 5],
  ]);
});

test('CreateTexture2 and CreateTexture safely clamp mipLevels, sampleCount, and dimensions', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  // Excessive mip levels (99 -> 9 for 256x128) and MSAA with mips > 1 (4 -> 1).
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 10, width: 256, height: 128, mips: 99, samples: 4, format: 22 })
  );
  // 8x MSAA on single-mip 2D -> clamped to 4.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 11, width: 64, height: 64, mips: 0, samples: 8, format: 22 })
  );
  // 2x MSAA (unsupported in WebGPU) -> clamped to 1.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 12, width: 64, height: 64, mips: 1, samples: 2, format: 22 })
  );
  // 4x MSAA on 2D array (layers=6) -> clamped to 1.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 13, width: 64, height: 64, layers: 6, mips: 1, samples: 4, format: 22 })
  );
  // 3D texture: depth participates in max mip calculation (16x16x64 -> 7 mips), MSAA -> 1.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 14,
      width: 16,
      height: 16,
      layers: 64,
      mips: 20,
      samples: 4,
      dimension: 2,
      format: 22,
    })
  );
  // 1D texture: height=1, depthOrArrayLayers=1, mipLevelCount=1, sampleCount=1.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 15,
      width: 128,
      height: 8,
      layers: 4,
      mips: 5,
      samples: 4,
      dimension: 0,
      format: 22,
    })
  );
  // v1 CreateTexture alongside CreateTexture2 also clamps excessive mipLevels (50 -> 6 for 32x16).
  processor.executeCommandBuffer(
    createTexture1Packet({ id: 16, width: 32, height: 16, mips: 50, format: 0, dimension: 1 })
  );

  const byId = (id) => processor.textures.get(id).desc;
  assert.equal(byId(10).mipLevelCount, 9);
  assert.equal(byId(10).sampleCount, 1);

  assert.equal(byId(11).mipLevelCount, 1);
  assert.equal(byId(11).sampleCount, 4);

  assert.equal(byId(12).sampleCount, 1);
  assert.equal(byId(13).sampleCount, 1);

  assert.equal(byId(14).dimension, '3d');
  assert.equal(byId(14).mipLevelCount, 7);
  assert.equal(byId(14).sampleCount, 1);

  assert.deepEqual(byId(15).size, { width: 128, height: 1, depthOrArrayLayers: 1 });
  assert.equal(byId(15).mipLevelCount, 1);
  assert.equal(byId(15).sampleCount, 1);

  assert.equal(byId(16).mipLevelCount, 6);
  assert.equal(byId(16).format, 'rgba8unorm');

  // createTexture2Count counts only v2 CreateTexture2 (6), while created counts both v1 + v2 (7).
  assert.equal(handler.stats.created, 7);
  assert.equal(handler.stats.createTexture2Count, 6);
  assert.equal(handler.stats.errors, 0);
});

test('alignBytesPerRow256 and repackRowsTo256 enforce WebGPU 256-byte row pitch alignment', () => {
  assert.equal(tp.alignBytesPerRow256(0), 0);
  assert.equal(tp.alignBytesPerRow256(1), 256);
  assert.equal(tp.alignBytesPerRow256(12), 256);
  assert.equal(tp.alignBytesPerRow256(256), 256);
  assert.equal(tp.alignBytesPerRow256(257), 512);
  assert.equal(tp.alignBytesPerRow256(768), 768);
  assert.throws(() => tp.alignBytesPerRow256(-4), RangeError);
  assert.throws(() => tp.alignBytesPerRow256(12.5), RangeError);

  // Unaligned 3 rows x 12 bytes -> 3 rows x 256 bytes, zero-padded tails.
  const srcUnaligned = new Uint8Array(36);
  for (let i = 0; i < 36; i += 1) srcUnaligned[i] = i + 1;
  const repacked = tp.repackRowsTo256(srcUnaligned, 12, 3);
  assert.equal(repacked.repacked, true);
  assert.equal(repacked.copiedFromSab, false);
  assert.equal(repacked.bytesPerRow, 256);
  assert.equal(repacked.originalBytesPerRow, 12);
  assert.equal(repacked.rowCount, 3);
  assert.equal(repacked.data.byteLength, 768);
  for (let row = 0; row < 3; row += 1) {
    const rowStart = row * 256;
    assert.deepEqual(
      Array.from(repacked.data.subarray(rowStart, rowStart + 12)),
      Array.from(srcUnaligned.subarray(row * 12, row * 12 + 12))
    );
    for (let pad = 12; pad < 256; pad += 1) {
      assert.equal(repacked.data[rowStart + pad], 0);
    }
  }

  // Already 256-byte aligned plain ArrayBuffer view -> zero-copy, repacked: false.
  const srcAligned = new Uint8Array(512);
  srcAligned[0] = 0xaa;
  srcAligned[256] = 0xbb;
  const alignedRes = tp.repackRowsTo256(srcAligned, 256, 2);
  assert.equal(alignedRes.repacked, false);
  assert.equal(alignedRes.copiedFromSab, false);
  assert.equal(alignedRes.bytesPerRow, 256);
  assert.equal(alignedRes.data, srcAligned);

  // Multi-layer (2D array / 3D) repacking preserves byteOffset on subarray input.
  const outer = new Uint8Array(16 + 16 * 2 * 3);
  outer.fill(0xff, 0, 16);
  for (let i = 0; i < 96; i += 1) outer[16 + i] = (i * 7) & 0xff;
  const sub = outer.subarray(16);
  const multi = tp.repackRowsTo256(sub, 16, 2, 3);
  assert.equal(multi.repacked, true);
  assert.equal(multi.rowCount, 6);
  assert.equal(multi.data.byteLength, 6 * 256);
  assert.equal(multi.data[0], 0);
  assert.equal(multi.data[5 * 256 + 15], (95 * 7) & 0xff);

  // SharedArrayBuffer-backed views are safely copied out even when already aligned.
  const sabAligned = new SharedArrayBuffer(512);
  const sabAlignedView = new Uint8Array(sabAligned);
  sabAlignedView[0] = 42;
  sabAlignedView[256] = 99;
  const sabRes = tp.repackRowsTo256(sabAlignedView, 256, 2);
  assert.equal(sabRes.repacked, false);
  assert.equal(sabRes.copiedFromSab, true);
  assert.equal(sabRes.data.buffer instanceof ArrayBuffer, true);
  assert.equal(sabRes.data[0], 42);
  assert.equal(sabRes.data[256], 99);

  // SharedArrayBuffer-backed unaligned view is both repacked and copied out.
  const sabUnaligned = new SharedArrayBuffer(32);
  const sabUnalignedView = new Uint8Array(sabUnaligned);
  for (let i = 0; i < 32; i += 1) sabUnalignedView[i] = i + 10;
  const sabRepacked = tp.repackRowsTo256(sabUnalignedView, 16, 2);
  assert.equal(sabRepacked.repacked, true);
  assert.equal(sabRepacked.copiedFromSab, true);
  assert.equal(sabRepacked.data.buffer instanceof ArrayBuffer, true);
  assert.equal(sabRepacked.bytesPerRow, 256);
  assert.equal(sabRepacked.data[256 + 15], 41);

  // Insufficient bytes throw RangeError.
  assert.throws(() => tp.repackRowsTo256(new Uint8Array(10), 16, 2), RangeError);
});

test('staged WriteTexture & copyBufferToTexture repack unaligned bytesPerRow to 256B and copy SAB views', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  // Create a 4x4 rgba8unorm texture via ABI v2 CreateTexture2.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 50, width: 4, height: 4, usage: 0x06, format: 22 })
  );

  // Upload 4x4 RGBA8 (bytesPerRow = 16, unaligned!) from a SharedArrayBuffer command packet.
  const texels = new Uint8Array(64);
  for (let i = 0; i < 64; i += 1) texels[i] = i + 1;

  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 50,
      width: 4,
      height: 4,
      bytesPerRow: 16,
      rowsPerImage: 4,
      data: texels,
      useSab: true,
    })
  );

  // Verify copyBufferToTexture received 256-byte aligned bytesPerRow and zero SAB views reached queue.writeBuffer.
  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(c2tCalls.length, 1);
  assert.equal(c2tCalls[0].source.bytesPerRow, 256);
  assert.equal(c2tCalls[0].source.rowsPerImage, 4);
  for (let row = 0; row < 4; row += 1) {
    assert.deepEqual(
      Array.from(c2tCalls[0].bytes.subarray(row * 256, row * 256 + 16)),
      Array.from(texels.subarray(row * 16, row * 16 + 16))
    );
  }
  const wbCalls = gpu.calls.filter((c) => c.op === 'writeBuffer');
  assert.ok(wbCalls.length >= 1);
  assert.ok(wbCalls.every((c) => c.isSab === false));

  assert.deepEqual(handler.getStats(), {
    created: 1,
    errors: 0,
    createTexture2Count: 1,
    rowPitchRepackCount: 1,
    sabCopyCount: 1,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 1,
    uploadsCount: 1,
    uploadBytes: 1024,
    uploadBytesTotal: 1024,
    reusedDescriptorCount: 1,
    clampedSubRegionCount: 0,
    zeroCopyUploads: 0,
    scratchRepackUploads: 1,
    directZeroCopyUploads: 0,
    repackedUploads: 1,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 1,
    scratchBufferBytes: 1024,
    r8unormUploads: 0,
    bytesUploaded: 1024,
    srgbSiblingViewFormatsAdded: 1,
    promoted1DTo2DCount: 0,
    chunkedWrites: 0,
    trimmedStagingBytes: 0,
    destroyedTextures: 0,
  });
});

test('direct queue.writeTexture safely copies SharedArrayBuffer views and tracks sabCopyCount', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 60, width: 2, height: 2, usage: 0x06, format: 22 })
  );

  const texels = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

  // Upload 1: SharedArrayBuffer-backed WriteTexture packet -> must copy out of SAB.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 60,
      width: 2,
      height: 2,
      bytesPerRow: 8,
      rowsPerImage: 2,
      data: texels,
      useSab: true,
    })
  );

  // Upload 2: plain ArrayBuffer-backed WriteTexture packet -> no SAB copy needed.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 60,
      width: 2,
      height: 2,
      bytesPerRow: 8,
      rowsPerImage: 2,
      data: texels,
      useSab: false,
    })
  );

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 2);
  assert.equal(wtCalls[0].isSab, false);
  assert.deepEqual(Array.from(wtCalls[0].bytes), Array.from(texels));
  assert.equal(wtCalls[1].isSab, false);

  assert.deepEqual(handler.getStats(), {
    created: 1,
    errors: 0,
    createTexture2Count: 1,
    rowPitchRepackCount: 0,
    sabCopyCount: 1,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 2,
    uploadsCount: 2,
    uploadBytes: 32,
    uploadBytesTotal: 32,
    reusedDescriptorCount: 2,
    clampedSubRegionCount: 0,
    zeroCopyUploads: 1,
    scratchRepackUploads: 0,
    directZeroCopyUploads: 1,
    repackedUploads: 0,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 1,
    scratchBufferBytes: 16,
    r8unormUploads: 0,
    bytesUploaded: 32,
    srgbSiblingViewFormatsAdded: 1,
    promoted1DTo2DCount: 0,
    chunkedWrites: 0,
    trimmedStagingBytes: 0,
    destroyedTextures: 0,
  });
});

test('handler.copyBufferToTexture repacks both CPU-backed and GPU-buffer-only unaligned copies', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 70, width: 4, height: 2, usage: 0x06, format: 22 })
  );
  const texture = processor.textures.get(70);
  const encoder = gpu.device.createCommandEncoder();

  // Case 1: CPU-backed SAB bytes with unaligned bytesPerRow = 16.
  const sab = new SharedArrayBuffer(32);
  const sabView = new Uint8Array(sab);
  for (let i = 0; i < 32; i += 1) sabView[i] = 100 + i;

  const resCpu = handler.copyBufferToTexture(
    encoder,
    { bytesPerRow: 16, rowsPerImage: 2 },
    { texture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 4, height: 2, depthOrArrayLayers: 1 },
    sabView
  );
  assert.equal(resCpu.repacked, true);
  assert.equal(resCpu.copiedFromSab, true);
  assert.equal(resCpu.bytesPerRow, 256);

  // Case 2: GPU-buffer-only copy with unaligned 4-byte-multiple bytesPerRow = 16.
  const gpuBuf = gpu.device.createBuffer({ size: 32, usage: 0x0004 | 0x0008 });
  for (let i = 0; i < 32; i += 1) gpuBuf.backing[i] = 200 + i;
  const resGpu = handler.copyBufferToTexture(
    encoder,
    { buffer: gpuBuf, offset: 0, bytesPerRow: 16, rowsPerImage: 2 },
    { texture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 4, height: 2, depthOrArrayLayers: 1 }
  );
  assert.equal(resGpu.repacked, true);
  assert.equal(resGpu.copiedFromSab, false);
  assert.equal(resGpu.bytesPerRow, 256);

  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(c2tCalls.length, 2);
  assert.equal(c2tCalls[0].source.bytesPerRow, 256);
  assert.equal(c2tCalls[1].source.bytesPerRow, 256);
  assert.deepEqual(Array.from(c2tCalls[1].bytes.subarray(0, 16)), Array.from(gpuBuf.backing.subarray(0, 16)));
  assert.deepEqual(
    Array.from(c2tCalls[1].bytes.subarray(256, 272)),
    Array.from(gpuBuf.backing.subarray(16, 32))
  );

  // Case 3: Already-aligned GPU buffer passes straight through without repacking.
  const alignedBuf = gpu.device.createBuffer({ size: 512, usage: 0x0004 });
  const resAligned = handler.copyBufferToTexture(
    encoder,
    { buffer: alignedBuf, offset: 0, bytesPerRow: 256, rowsPerImage: 2 },
    { texture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 4, height: 2, depthOrArrayLayers: 1 }
  );
  assert.equal(resAligned.repacked, false);
  assert.equal(resAligned.bytesPerRow, 256);

  const stats = handler.getStats();
  assert.deepEqual(stats, {
    created: 1,
    errors: 0,
    createTexture2Count: 1,
    rowPitchRepackCount: 2,
    sabCopyCount: 1,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 3,
    uploadsCount: 3,
    uploadBytes: 1536,
    uploadBytesTotal: 1536,
    reusedDescriptorCount: 3,
    clampedSubRegionCount: 0,
    zeroCopyUploads: 0,
    scratchRepackUploads: 1,
    directZeroCopyUploads: 0,
    repackedUploads: 2,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 1,
    scratchBufferBytes: 512,
    r8unormUploads: 0,
    bytesUploaded: 1536,
    srgbSiblingViewFormatsAdded: 1,
    promoted1DTo2DCount: 0,
    chunkedWrites: 0,
    trimmedStagingBytes: 0,
    destroyedTextures: 0,
  });
  for (const [key, val] of Object.entries(stats)) {
    assert.equal(typeof val, 'number', `stat ${key} must be numeric`);
    assert.ok(Number.isInteger(val) && val >= 0, `stat ${key} must be a non-negative integer`);
  }
});

test('G31 classic-worker importScripts IIFE isolation preserves zero top-level lexical leaks', () => {
  const ctx = vm.createContext({
    console,
    Uint8Array,
    DataView,
    ArrayBuffer,
    SharedArrayBuffer,
    Map,
    Set,
  });
  ctx.globalThis = ctx;
  ctx.self = ctx;

  const load = (file) => {
    const src = readFileSync(path.join(WORKERS_DIR, file), 'utf8');
    vm.runInContext(src, ctx, { filename: file });
  };

  // Load texture_formats.js, render_state_packets.js, and texture_packets.js twice without SyntaxError.
  assert.doesNotThrow(() => {
    load('texture_formats.js');
    load('render_state_packets.js');
    load('texture_packets.js');
    load('texture_packets.js');
  });

  // Verify no internal identifiers leaked into global lexical scope.
  assert.doesNotThrow(() => {
    vm.runInContext(
      `
      const CREATE_TEXTURE2 = 0;
      const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 0;
      const TEXTURE_FORMAT_NAMES = [];
      const alignBytesPerRow256 = null;
      const repackRowsTo256 = null;
      const isBcCompressedFormat = null;
      const deviceSupportsBcSliced3d = null;
      const SRGB_SIBLING_FORMATS = null;
      const DEFAULT_MAX_SCRATCH_BYTES = 0;
      const getSrgbSiblingFormat = null;
      const resolveCompatibleViewFormats = null;
      const WEBGPU_FORMAT_BLOCK_INFO = null;
      const WEBGPU_FORMAT_INFO = null;
      const expandA8ToRgba8 = null;
      const alignCopySizeForFormat = null;
      const calcMipUploadLayout = null;
      const calcMipDimension = null;
      const clampSubRegionToMipBounds = null;
      class TexturePackets {}
      class TextureScratchPool {}
      `,
      ctx
    );
  });

  assert.equal(typeof ctx.PlayGta6TexturePackets.alignBytesPerRow256, 'function');
  assert.equal(typeof ctx.PlayGta6TexturePackets.repackRowsTo256, 'function');
  assert.equal(typeof ctx.PlayGta6TexturePackets.expandA8ToRgba8, 'function');
  assert.equal(typeof ctx.PlayGta6TexturePackets.clampSubRegionToMipBounds, 'function');
  assert.equal(typeof ctx.PlayGta6TexturePackets.calcMipDimension, 'function');
  assert.equal(typeof ctx.PlayGta6TexturePackets.TextureScratchPool, 'function');
  assert.equal(typeof ctx.PlayGta6TextureFormats.expandA8ToRgba8, 'function');
  assert.equal(typeof ctx.PlayGta6TextureFormats.TextureScratchPool, 'function');
  assert.equal(ctx.PlayGta6TexturePackets.alignBytesPerRow256(16), 256);
});

test('W108 (#734): 3D BC textures fall back to rgba8unorm without bc-sliced-3d and skip compressed uploads', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const rsHandler = installRenderStatePackets(processor);
  const handler = tp.installTexturePackets(processor);

  // CreateTexture2: 3D bc5-rg-unorm (code 58) with incompatible bc5 viewFormats.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 201,
      width: 16,
      height: 16,
      layers: 4,
      mips: 1,
      dimension: 2, // 3d
      format: 58, // bc5-rg-unorm
      usage: 0x06,
      viewFormats: [58, 59], // bc5-rg-unorm, bc5-rg-snorm
    })
  );
  // CreateTexture2: 3D bc7-rgba-unorm (code 62) with bc7-rgba-unorm-srgb (code 63) viewFormat.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 202,
      width: 8,
      height: 8,
      layers: 4,
      mips: 1,
      dimension: 2, // 3d
      format: 62, // bc7-rgba-unorm
      usage: 0x06,
      viewFormats: [63], // bc7-rgba-unorm-srgb
    })
  );
  // v1 CreateTexture: 3D bc7-rgba-unorm (v1 format code 5).
  processor.executeCommandBuffer(
    createTexture1Packet({
      id: 203,
      width: 8,
      height: 8,
      layers: 2,
      mips: 1,
      dimension: 2, // 3d
      format: 5, // bc7-rgba-unorm
      usage: 0x06,
    })
  );

  const tex201 = processor.textures.get(201);
  const tex202 = processor.textures.get(202);
  const tex203 = processor.textures.get(203);
  assert.ok(tex201 && tex202 && tex203);
  assert.equal(tex201.desc.format, 'rgba8unorm');
  assert.equal(tex201.desc.dimension, '3d');
  assert.equal(tex201.desc.viewFormats, undefined);
  assert.equal(tex202.desc.format, 'rgba8unorm');
  assert.equal(tex202.desc.viewFormats, undefined);
  assert.equal(tex203.desc.format, 'rgba8unorm');
  assert.equal(tex203.desc.dimension, '3d');
  assert.equal(handler._bc3dFallbackTextures.has(201), true);
  assert.equal(handler._bc3dFallbackTextures.has(202), true);
  assert.equal(handler._bc3dFallbackTextures.has(203), true);
  assert.equal(handler.stats.bc3dFallbacks, 3);
  assert.equal(handler.stats.errors, 0);

  // Chained RenderStatePackets view creation must not pass bc* format against the rgba8unorm fallback.
  const view201 = rsHandler._textureView(201, {
    format: 'bc5-rg-unorm',
    dimension: '3d',
    baseMipLevel: 0,
    mipLevelCount: 1,
    baseArrayLayer: 0,
    arrayLayerCount: 1,
    aspect: 'all',
  });
  assert.ok(view201);
  assert.equal(view201.viewDesc.format, 'rgba8unorm');

  // Compressed WriteTexture uploads to fallback 3D textures must be skipped cleanly (both staged and direct).
  const bc5Bytes = new Uint8Array(4 * 4 * 16 * 4); // 16x16x4 BC5 blocks
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 201,
      width: 16,
      height: 16,
      layers: 4,
      bytesPerRow: 64,
      rowsPerImage: 4,
      data: bc5Bytes,
    })
  );
  // Direct writeTexture() and copyBufferToTexture() wrappers also skip fallback texture objects.
  const directWt = handler.writeTexture(
    { texture: tex202, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    new Uint8Array(64),
    { offset: 0, bytesPerRow: 32, rowsPerImage: 2 },
    { width: 8, height: 8, depthOrArrayLayers: 4 }
  );
  assert.equal(directWt.uploaded, false);
  assert.equal(directWt.skippedBc3dFallback, true);

  const encoder = gpu.device.createCommandEncoder();
  const directC2t = handler.copyBufferToTexture(
    encoder,
    { bytesPerRow: 32, rowsPerImage: 2 },
    { texture: tex202, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 8, height: 8, depthOrArrayLayers: 4 },
    new Uint8Array(256)
  );
  assert.equal(directC2t.skippedBc3dFallback, true);

  assert.equal(gpu.calls.filter((c) => c.op === 'writeTexture').length, 0);
  assert.equal(gpu.calls.filter((c) => c.op === 'copyBufferToTexture').length, 0);
  assert.deepEqual(handler.getStats(), {
    created: 3,
    errors: 0,
    createTexture2Count: 2,
    rowPitchRepackCount: 0,
    sabCopyCount: 0,
    bc3dFallbacks: 3,
    bc3dFallbackUploadsSkipped: 3,
    uploadCount: 0,
    uploadsCount: 0,
    uploadBytes: 0,
    uploadBytesTotal: 0,
    reusedDescriptorCount: 0,
    clampedSubRegionCount: 0,
    zeroCopyUploads: 0,
    scratchRepackUploads: 0,
    directZeroCopyUploads: 0,
    repackedUploads: 0,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 0,
    scratchBufferBytes: 0,
    r8unormUploads: 0,
    bytesUploaded: 0,
    srgbSiblingViewFormatsAdded: 0,
    promoted1DTo2DCount: 0,
    chunkedWrites: 0,
    trimmedStagingBytes: 0,
    destroyedTextures: 0,
  });
});

test('W108 (#734): 3D BC textures preserve original format and upload normally when bc-sliced-3d is present', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc', 'texture-compression-bc-sliced-3d']),
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 210,
      width: 16,
      height: 16,
      layers: 4,
      mips: 1,
      dimension: 2, // 3d
      format: 58, // bc5-rg-unorm
      usage: 0x06,
    })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 211,
      width: 8,
      height: 8,
      layers: 2,
      mips: 1,
      dimension: 2, // 3d
      format: 62, // bc7-rgba-unorm
      usage: 0x06,
      viewFormats: [63], // bc7-rgba-unorm-srgb
    })
  );

  assert.equal(processor.textures.get(210).desc.format, 'bc5-rg-unorm');
  assert.equal(processor.textures.get(211).desc.format, 'bc7-rgba-unorm');
  assert.deepEqual(processor.textures.get(211).desc.viewFormats, ['bc7-rgba-unorm-srgb']);
  assert.equal(handler._bc3dFallbackTextures.size, 0);
  assert.equal(handler.stats.bc3dFallbacks, 0);

  const bc5Bytes = new Uint8Array(4 * 4 * 16 * 4);
  bc5Bytes[0] = 0x5a;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 210,
      width: 16,
      height: 16,
      layers: 4,
      bytesPerRow: 64,
      rowsPerImage: 4,
      data: bc5Bytes,
    })
  );

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 1);
  assert.equal(wtCalls[0].bytes[0], 0x5a);
  assert.equal(handler.stats.bc3dFallbackUploadsSkipped, 0);
  assert.equal(handler.stats.errors, 0);
});

test('W108 (#734): 2D BC textures remain unaffected when bc-sliced-3d is absent', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 220,
      width: 16,
      height: 16,
      layers: 1,
      mips: 1,
      dimension: 1, // 2d
      format: 58, // bc5-rg-unorm
      usage: 0x06,
    })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 221,
      width: 16,
      height: 16,
      layers: 6,
      mips: 1,
      dimension: 1, // 2d array
      format: 62, // bc7-rgba-unorm
      usage: 0x06,
      viewFormats: [63],
    })
  );

  assert.equal(processor.textures.get(220).desc.format, 'bc5-rg-unorm');
  assert.equal(processor.textures.get(221).desc.format, 'bc7-rgba-unorm');
  assert.deepEqual(processor.textures.get(221).desc.viewFormats, ['bc7-rgba-unorm-srgb']);
  assert.equal(handler._bc3dFallbackTextures.size, 0);
  assert.equal(handler.stats.bc3dFallbacks, 0);

  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 220,
      width: 16,
      height: 16,
      layers: 1,
      bytesPerRow: 64,
      rowsPerImage: 4,
      data: new Uint8Array(256),
    })
  );
  assert.equal(gpu.calls.filter((c) => c.op === 'writeTexture').length, 1);
  assert.equal(handler.stats.bc3dFallbackUploadsSkipped, 0);
});

test('W108 (#734): destroying or recreating a fallback texture ID clears fallback tracking', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  // Create two 3D BC fallback textures.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 230, width: 8, height: 8, layers: 4, dimension: 2, format: 58, usage: 0x06 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 231, width: 8, height: 8, layers: 4, dimension: 2, format: 62, usage: 0x06 })
  );
  assert.equal(handler._bc3dFallbackTextures.has(230), true);
  assert.equal(handler._bc3dFallbackTextures.has(231), true);

  // 1) DestroyResource removes ID 230 from the fallback tracking set.
  processor.executeCommandBuffer(destroyPacket(230));
  assert.equal(handler._bc3dFallbackTextures.has(230), false);
  assert.equal(handler._bc3dFallbackTextures.has(231), true);

  // 2) Recreating ID 231 with a non-3D/non-BC format clears it and allows normal uploads.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 231, width: 4, height: 4, layers: 1, dimension: 1, format: 22, usage: 0x06 })
  );
  assert.equal(handler._bc3dFallbackTextures.has(231), false);
  assert.equal(processor.textures.get(231).desc.format, 'rgba8unorm');

  const rgbaBytes = new Uint8Array(4 * 4 * 4);
  rgbaBytes[0] = 0x77;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 231,
      width: 4,
      height: 4,
      layers: 1,
      bytesPerRow: 16,
      rowsPerImage: 4,
      data: rgbaBytes,
    })
  );
  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 1);
  assert.equal(wtCalls[0].bytes[0], 0x77);
  assert.equal(handler.stats.bc3dFallbacks, 2);
  assert.equal(handler.stats.bc3dFallbackUploadsSkipped, 0);
});

test('W164 (#871): 4K Bink2 3840x2160 r8unorm Y plane takes zero-copy direct queue.writeTexture path even with stagingPool', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 4 * 1024 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  const W = 3840;
  const H = 2160;
  assert.equal(W % 256, 0);

  // Create 4K r8unorm (format code 1) Y plane texture.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 300, width: W, height: H, usage: 0x06, format: 1 })
  );
  assert.equal(processor.textures.get(300).desc.format, 'r8unorm');
  assert.equal(processor.textures.get(300).desc.viewFormats, undefined);

  const yBytes = new Uint8Array(W * H);
  yBytes[0] = 0x11;
  yBytes[W - 1] = 0x22;
  yBytes[(H - 1) * W] = 0x33;
  yBytes[W * H - 1] = 0x44;

  const pkt = writeTexturePacket({
    textureId: 300,
    width: W,
    height: H,
    bytesPerRow: W,
    rowsPerImage: H,
    data: yBytes,
    useSab: false,
  });

  processor.executeCommandBuffer(pkt);

  // Must go directly to queue.writeTexture with the exact command packet ArrayBuffer (zero-copy).
  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(wtCalls.length, 1);
  assert.equal(c2tCalls.length, 0);
  assert.equal(wtCalls[0].bufferRef, pkt.buffer, '3840x2160 aligned r8unorm upload must be zero-copy');
  assert.equal(wtCalls[0].byteOffset, 8 + 48);
  assert.equal(wtCalls[0].byteLength, W * H);
  assert.equal(wtCalls[0].dataLayout.bytesPerRow, W);
  assert.equal(wtCalls[0].dataLayout.rowsPerImage, H);
  assert.equal(wtCalls[0].bytes[0], 0x11);
  assert.equal(wtCalls[0].bytes[W - 1], 0x22);
  assert.equal(wtCalls[0].bytes[(H - 1) * W], 0x33);
  assert.equal(wtCalls[0].bytes[W * H - 1], 0x44);

  // No staging buffer chunks allocated and no scratch buffer touched.
  assert.equal(stagingPool.getStats().totalChunksAllocated, 0);
  assert.deepEqual(handler.getStats(), {
    created: 1,
    errors: 0,
    createTexture2Count: 1,
    rowPitchRepackCount: 0,
    sabCopyCount: 0,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 1,
    uploadsCount: 1,
    uploadBytes: W * H,
    uploadBytesTotal: W * H,
    reusedDescriptorCount: 1,
    clampedSubRegionCount: 0,
    zeroCopyUploads: 1,
    scratchRepackUploads: 0,
    directZeroCopyUploads: 1,
    repackedUploads: 0,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 0,
    scratchBufferBytes: 0,
    r8unormUploads: 1,
    bytesUploaded: W * H,
    srgbSiblingViewFormatsAdded: 0,
    promoted1DTo2DCount: 0,
    chunkedWrites: 0,
    trimmedStagingBytes: 0,
    destroyedTextures: 0,
  });
});

test('W164 (#871): 4K Bink2 1920x1080 r8unorm Cr/Cb planes reuse persistent scratch Uint8Array across frames and zero padding tails', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({
    device: gpu.device,
    chunkSize: 4 * 1024 * 1024,
    maxPoolBytes: 64 * 1024 * 1024,
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  const YW = 3840;
  const YH = 2160;
  const CW = 1920;
  const CH = 1080;
  const ALIGNED_CW = 2048;
  assert.equal(tp.alignBytesPerRow256(CW), ALIGNED_CW);

  // Create Y, Cr, Cb r8unorm planes.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 310, width: YW, height: YH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 311, width: CW, height: CH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 312, width: CW, height: CH, usage: 0x06, format: 1 })
  );

  const yData = new Uint8Array(YW * YH);
  const crData = new Uint8Array(CW * CH);
  const cbData = new Uint8Array(CW * CH);

  // Simulate 3 frames at 30 FPS cadence: 3x Y (zero-copy) + 3x Cr + 3x Cb (scratch-reused repack).
  const FRAMES = 3;
  let firstScratchRef = null;
  for (let frame = 0; frame < FRAMES; frame += 1) {
    stagingPool.beginFrame(frame);
    yData[0] = 0x10 + frame;
    crData.fill(0xa0 + frame);
    cbData.fill(0x50 + frame);
    // Mark first and last active bytes of row 0 and last row.
    crData[0] = 0xc0 + frame;
    crData[CW - 1] = 0xd0 + frame;
    cbData[0] = 0x60 + frame;
    cbData[CW - 1] = 0x70 + frame;
    cbData[(CH - 1) * CW + (CW - 1)] = 0x80 + frame;

    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 310,
        width: YW,
        height: YH,
        bytesPerRow: YW,
        rowsPerImage: YH,
        data: yData,
      })
    );
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 311,
        width: CW,
        height: CH,
        bytesPerRow: CW,
        rowsPerImage: CH,
        data: crData,
      })
    );
    if (frame === 0) {
      firstScratchRef = handler._scratchBuffer;
      assert.ok(firstScratchRef instanceof Uint8Array);
      assert.equal(firstScratchRef.byteLength, ALIGNED_CW * CH);
      // Poison padding tail in scratch buffer to verify next upload strictly zeroes it.
      firstScratchRef.fill(0xff, CW, ALIGNED_CW);
    }
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 312,
        width: CW,
        height: CH,
        bytesPerRow: CW,
        rowsPerImage: CH,
        data: cbData,
      })
    );
    assert.equal(
      handler._scratchBuffer,
      firstScratchRef,
      `Frame ${frame}: scratch Uint8Array must be reused without reallocating`
    );
    stagingPool.endFrame();
  }

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(wtCalls.length, FRAMES, 'Y plane uses direct zero-copy writeTexture each frame');
  assert.equal(c2tCalls.length, FRAMES * 2, 'Cr + Cb planes stage repacked 2048B rows');

  // Verify Cb frame 0 repacked rows have exact active pixels and strictly zeroed padding [1920..2048).
  const cbFrame0 = c2tCalls[1];
  assert.equal(cbFrame0.source.bytesPerRow, ALIGNED_CW);
  assert.equal(cbFrame0.source.rowsPerImage, CH);
  assert.equal(cbFrame0.bytes[0], 0x60);
  assert.equal(cbFrame0.bytes[CW - 1], 0x70);
  for (let p = CW; p < ALIGNED_CW; p += 1) {
    assert.equal(cbFrame0.bytes[p], 0, `padding byte at offset ${p} must be zeroed`);
  }
  assert.equal(cbFrame0.bytes[(CH - 1) * ALIGNED_CW + (CW - 1)], 0x80);

  const expectedBytes = FRAMES * (YW * YH) + FRAMES * 2 * (ALIGNED_CW * CH);
  assert.deepEqual(handler.getStats(), {
    created: 3,
    errors: 0,
    createTexture2Count: 3,
    rowPitchRepackCount: FRAMES * 2,
    sabCopyCount: 0,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: FRAMES * 3,
    uploadsCount: FRAMES * 3,
    uploadBytes: expectedBytes,
    uploadBytesTotal: expectedBytes,
    reusedDescriptorCount: FRAMES * 3,
    clampedSubRegionCount: 0,
    zeroCopyUploads: FRAMES,
    scratchRepackUploads: FRAMES * 2,
    directZeroCopyUploads: FRAMES,
    repackedUploads: FRAMES * 2,
    scratchBufferReuses: FRAMES * 2 - 1,
    scratchBufferAllocations: 1,
    scratchBufferBytes: ALIGNED_CW * CH,
    r8unormUploads: FRAMES * 3,
    bytesUploaded: expectedBytes,
    srgbSiblingViewFormatsAdded: 0,
    promoted1DTo2DCount: 0,
    chunkedWrites: 0,
    trimmedStagingBytes: 0,
    destroyedTextures: 0,
  });
});

test('W164 (#871): scratch buffer grows on demand, respects maxScratchBytes cap, and handles SAB-backed aligned uploads', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  // Small cap (4096 bytes) to test growth + cap enforcement deterministically.
  const handler = tp.installTexturePackets(processor, { maxScratchBytes: 4096 });

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 320, width: 16, height: 4, usage: 0x06, format: 1 }) // r8unorm
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 321, width: 16, height: 8, usage: 0x06, format: 1 }) // r8unorm
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 322, width: 16, height: 32, usage: 0x06, format: 1 }) // r8unorm -> 256*32=8192 > 4096 cap
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 323, width: 256, height: 8, usage: 0x06, format: 1 }) // aligned 256x8 = 2048B SAB
  );

  // 1) Initial 16x4 -> repacked 256*4 = 1024 bytes (alloc #1).
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 320, width: 16, height: 4, bytesPerRow: 16, rowsPerImage: 4, data: new Uint8Array(64) })
  );
  assert.equal(handler.getStats().scratchBufferAllocations, 1);
  assert.equal(handler.getStats().scratchBufferReuses, 0);
  assert.equal(handler.getStats().scratchBufferBytes, 1024);

  // 2) Same size -> reuses 1024B scratch (reuse #1).
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 320, width: 16, height: 4, bytesPerRow: 16, rowsPerImage: 4, data: new Uint8Array(64) })
  );
  assert.equal(handler.getStats().scratchBufferAllocations, 1);
  assert.equal(handler.getStats().scratchBufferReuses, 1);

  // 3) Larger 16x8 -> repacked 256*8 = 2048 bytes <= 4096 cap -> grows (alloc #2).
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 321, width: 16, height: 8, bytesPerRow: 16, rowsPerImage: 8, data: new Uint8Array(128) })
  );
  assert.equal(handler.getStats().scratchBufferAllocations, 2);
  assert.equal(handler.getStats().scratchBufferBytes, 2048);

  // 4) Oversized 16x32 -> repacked 256*32 = 8192 > 4096 cap -> one-off alloc, persistent scratch stays 2048B.
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 322, width: 16, height: 32, bytesPerRow: 16, rowsPerImage: 32, data: new Uint8Array(512) })
  );
  assert.equal(handler.getStats().scratchBufferAllocations, 2);
  assert.equal(handler.getStats().scratchBufferBytes, 2048);

  // 5) SAB-backed aligned 256x8 (2048B) -> copies out of SAB using existing 2048B scratch (reuse #2), direct writeTexture!
  const sabData = new Uint8Array(256 * 8);
  sabData[0] = 0xab;
  sabData[2047] = 0xcd;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 323,
      width: 256,
      height: 8,
      bytesPerRow: 256,
      rowsPerImage: 8,
      data: sabData,
      useSab: true,
    })
  );
  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 1);
  assert.equal(wtCalls[0].isSab, false);
  assert.equal(wtCalls[0].bytes[0], 0xab);
  assert.equal(wtCalls[0].bytes[2047], 0xcd);
  assert.equal(handler.getStats().sabCopyCount, 1);
  assert.equal(handler.getStats().scratchBufferReuses, 2);
  assert.equal(handler.getStats().directZeroCopyUploads, 0, 'SAB copy must not count as zero-copy');
  assert.equal(handler.getStats().r8unormUploads, 5);
});

test('W164 (#871): CreateTexture and CreateTexture2 attach compatible sRGB/linear sibling viewFormats safely', () => {
  // Unit check helper exports directly.
  assert.equal(tp.getSrgbSiblingFormat('rgba8unorm'), 'rgba8unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('rgba8unorm-srgb'), 'rgba8unorm');
  assert.equal(tp.getSrgbSiblingFormat('bgra8unorm'), 'bgra8unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('bgra8unorm-srgb'), 'bgra8unorm');
  assert.equal(tp.getSrgbSiblingFormat('bc1-rgba-unorm'), 'bc1-rgba-unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('bc1-rgba-unorm-srgb'), 'bc1-rgba-unorm');
  assert.equal(tp.getSrgbSiblingFormat('bc2-rgba-unorm'), 'bc2-rgba-unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('bc2-rgba-unorm-srgb'), 'bc2-rgba-unorm');
  assert.equal(tp.getSrgbSiblingFormat('bc3-rgba-unorm'), 'bc3-rgba-unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('bc3-rgba-unorm-srgb'), 'bc3-rgba-unorm');
  assert.equal(tp.getSrgbSiblingFormat('bc7-rgba-unorm'), 'bc7-rgba-unorm-srgb');
  assert.equal(tp.getSrgbSiblingFormat('bc7-rgba-unorm-srgb'), 'bc7-rgba-unorm');
  assert.equal(tp.getSrgbSiblingFormat('r8unorm'), null);
  assert.equal(tp.getSrgbSiblingFormat('bc4-r-unorm'), null);
  assert.equal(tp.getSrgbSiblingFormat('bc5-rg-unorm'), null);
  assert.equal(tp.getSrgbSiblingFormat('bc6h-rgb-ufloat'), null);
  assert.equal(tp.getSrgbSiblingFormat('rgba16float'), null);

  // resolveCompatibleViewFormats strips incompatible formats and deduplicates.
  assert.deepEqual(tp.resolveCompatibleViewFormats('rgba8unorm', ['rgba16float', 'rgba8unorm']), {
    viewFormats: ['rgba8unorm', 'rgba8unorm-srgb'],
    siblingAdded: true,
  });
  assert.deepEqual(tp.resolveCompatibleViewFormats('bgra8unorm', ['bgra8unorm-srgb', 'bgra8unorm-srgb']), {
    viewFormats: ['bgra8unorm-srgb'],
    siblingAdded: false,
  });

  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  // v1 CreateTexture: rgba8unorm (0), bgra8unorm (1), rgba16float (2), bc7-rgba-unorm (5).
  processor.executeCommandBuffer(createTexture1Packet({ id: 400, width: 64, height: 64, format: 0 }));
  processor.executeCommandBuffer(createTexture1Packet({ id: 401, width: 64, height: 64, format: 1 }));
  processor.executeCommandBuffer(createTexture1Packet({ id: 402, width: 64, height: 64, format: 2 }));
  processor.executeCommandBuffer(createTexture1Packet({ id: 403, width: 64, height: 64, format: 5 }));

  assert.deepEqual(processor.textures.get(400).desc.viewFormats, ['rgba8unorm-srgb']);
  assert.deepEqual(processor.textures.get(401).desc.viewFormats, ['bgra8unorm-srgb']);
  assert.equal(processor.textures.get(402).desc.viewFormats, undefined);
  assert.deepEqual(processor.textures.get(403).desc.viewFormats, ['bc7-rgba-unorm-srgb']);

  // Creating sRGB views on v1 linear textures must pass strict WebGPU validation.
  assert.doesNotThrow(() => processor.textures.get(400).createView({ format: 'rgba8unorm-srgb' }));
  assert.doesNotThrow(() => processor.textures.get(401).createView({ format: 'bgra8unorm-srgb' }));
  assert.doesNotThrow(() => processor.textures.get(403).createView({ format: 'bc7-rgba-unorm-srgb' }));

  // v2 CreateTexture2: all sRGB-capable pairs without explicit viewFormats.
  const v2Pairs = [
    [410, 22, 'rgba8unorm', 'rgba8unorm-srgb'],
    [411, 23, 'rgba8unorm-srgb', 'rgba8unorm'],
    [412, 27, 'bgra8unorm', 'bgra8unorm-srgb'],
    [413, 28, 'bgra8unorm-srgb', 'bgra8unorm'],
    [414, 50, 'bc1-rgba-unorm', 'bc1-rgba-unorm-srgb'],
    [415, 51, 'bc1-rgba-unorm-srgb', 'bc1-rgba-unorm'],
    [416, 52, 'bc2-rgba-unorm', 'bc2-rgba-unorm-srgb'],
    [417, 53, 'bc2-rgba-unorm-srgb', 'bc2-rgba-unorm'],
    [418, 54, 'bc3-rgba-unorm', 'bc3-rgba-unorm-srgb'],
    [419, 55, 'bc3-rgba-unorm-srgb', 'bc3-rgba-unorm'],
    [420, 62, 'bc7-rgba-unorm', 'bc7-rgba-unorm-srgb'],
    [421, 63, 'bc7-rgba-unorm-srgb', 'bc7-rgba-unorm'],
  ];
  for (const [id, code, expectedFmt, expectedSibling] of v2Pairs) {
    processor.executeCommandBuffer(
      createTexture2Packet({ id, width: 16, height: 16, format: code, usage: 0x06 })
    );
    const tex = processor.textures.get(id);
    assert.equal(tex.desc.format, expectedFmt);
    assert.deepEqual(tex.desc.viewFormats, [expectedSibling]);
    assert.doesNotThrow(() => tex.createView({ format: expectedSibling }));
  }

  // Incompatible viewFormat in packet payload is stripped safely while valid sibling is attached.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 430,
      width: 16,
      height: 16,
      format: 22, // rgba8unorm
      viewFormats: [40], // incompatible rgba16float
      usage: 0x06,
    })
  );
  assert.deepEqual(processor.textures.get(430).desc.viewFormats, ['rgba8unorm-srgb']);

  // Formats without sRGB siblings (r8unorm, bc4, bc5, bc6h) keep viewFormats omitted.
  for (const [id, code] of [
    [440, 1], // r8unorm
    [441, 56], // bc4-r-unorm
    [442, 58], // bc5-rg-unorm
    [443, 60], // bc6h-rgb-ufloat
  ]) {
    processor.executeCommandBuffer(
      createTexture2Packet({ id, width: 16, height: 16, format: code, usage: 0x06 })
    );
    assert.equal(processor.textures.get(id).desc.viewFormats, undefined);
  }

  // 3 v1 textures + 12 v2 pairs + 1 stripped-incompatible v2 = 16 sibling viewFormats added.
  assert.equal(handler.getStats().srgbSiblingViewFormatsAdded, 16);
  assert.equal(handler.getStats().errors, 0);
});

test('W164 (#871): telemetry resets cleanly on reset(), destroy(), and processor.destroy()', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 500, width: 256, height: 4, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 501, width: 16, height: 4, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 502, width: 8, height: 8, usage: 0x06, format: 22 })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 500, width: 256, height: 4, bytesPerRow: 256, rowsPerImage: 4, data: new Uint8Array(1024) })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 501, width: 16, height: 4, bytesPerRow: 16, rowsPerImage: 4, data: new Uint8Array(64) })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 501, width: 16, height: 4, bytesPerRow: 16, rowsPerImage: 4, data: new Uint8Array(64) })
  );

  const beforeReset = handler.getStats();
  assert.equal(beforeReset.created, 3);
  assert.equal(beforeReset.directZeroCopyUploads, 1);
  assert.equal(beforeReset.repackedUploads, 2);
  assert.equal(beforeReset.scratchBufferAllocations, 1);
  assert.equal(beforeReset.scratchBufferReuses, 1);
  assert.equal(beforeReset.scratchBufferBytes, 1024);
  assert.equal(beforeReset.r8unormUploads, 3);
  assert.equal(beforeReset.bytesUploaded, 1024 + 1024 + 1024);
  assert.equal(beforeReset.srgbSiblingViewFormatsAdded, 1);
  for (const [key, val] of Object.entries(beforeReset)) {
    assert.equal(typeof val, 'number', `stat ${key} must be numeric`);
    assert.ok(Number.isInteger(val) && val >= 0, `stat ${key} must be a non-negative integer`);
  }

  const expectedZeroStats = {
    created: 0,
    errors: 0,
    createTexture2Count: 0,
    rowPitchRepackCount: 0,
    sabCopyCount: 0,
    bc3dFallbacks: 0,
    bc3dFallbackUploadsSkipped: 0,
    uploadCount: 0,
    uploadsCount: 0,
    uploadBytes: 0,
    uploadBytesTotal: 0,
    reusedDescriptorCount: 0,
    clampedSubRegionCount: 0,
    zeroCopyUploads: 0,
    scratchRepackUploads: 0,
    directZeroCopyUploads: 0,
    repackedUploads: 0,
    scratchBufferReuses: 0,
    scratchBufferAllocations: 0,
    scratchBufferBytes: 0,
    r8unormUploads: 0,
    bytesUploaded: 0,
    srgbSiblingViewFormatsAdded: 0,
    promoted1DTo2DCount: 0,
    chunkedWrites: 0,
    trimmedStagingBytes: 0,
    destroyedTextures: 0,
  };

  // 1) reset() clears counters and releases scratch buffer, while handler stays usable.
  handler.reset();
  assert.deepEqual(handler.getStats(), expectedZeroStats);
  assert.equal(handler._scratchBuffer, null);
  assert.equal(handler.lastError, null);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 510, width: 256, height: 2, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({ textureId: 510, width: 256, height: 2, bytesPerRow: 256, rowsPerImage: 2, data: new Uint8Array(512) })
  );
  assert.equal(handler.getStats().directZeroCopyUploads, 1);
  assert.equal(handler.getStats().r8unormUploads, 1);
  assert.equal(handler.getStats().bytesUploaded, 512);

  // 2) processor.destroy() triggers handler.destroy() cleanly and idempotently.
  processor.destroy();
  assert.deepEqual(handler.getStats(), expectedZeroStats);
  assert.equal(handler._scratchBuffer, null);
  handler.destroy();
  assert.deepEqual(handler.getStats(), expectedZeroStats);
});

test('W191: tightly packed and already-aligned WriteTexture payloads pass exact zero-copy ArrayBuffer subarray views', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  const YW = 3840;
  const YH = 2160;
  const CW = 1920;
  const CH = 1080;

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 600, width: YW, height: YH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 601, width: CW, height: CH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 602, width: CW, height: CH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 603, width: 8, height: 4, layers: 2, usage: 0x06, format: 1 })
  );

  // Build one combined multi-packet command buffer with a leading non-zero prefix to verify
  // exact view.buffer identity and byteOffset preservation without any .slice() / copy.
  const yData = new Uint8Array(YW * YH);
  yData[0] = 0x1a;
  yData[yData.length - 1] = 0x2b;

  const uData = new Uint8Array(CW * CH);
  uData[0] = 0x3c;
  uData[uData.length - 1] = 0x4d;

  const vData = new Uint8Array(CW * CH);
  vData[0] = 0x5e;
  vData[vData.length - 1] = 0x6f;

  const layerData = new Uint8Array(8 * 4 * 2);
  layerData[0] = 0x71;
  layerData[layerData.length - 1] = 0x82;

  // Packet 1: 4K Y with implicit tightly-packed bytesPerRow=0 (inferred 3840, 256B aligned).
  const pktY = writeTexturePacket({
    textureId: 600,
    width: YW,
    height: YH,
    bytesPerRow: 0,
    rowsPerImage: 0,
    data: yData,
  });
  // Packet 2: 1920x1080 U with explicit tightly-packed bytesPerRow=1920.
  const pktU = writeTexturePacket({
    textureId: 601,
    width: CW,
    height: CH,
    bytesPerRow: CW,
    rowsPerImage: CH,
    data: uData,
  });
  // Packet 3: 1920x1080 V with implicit tightly-packed bytesPerRow=0.
  const pktV = writeTexturePacket({
    textureId: 602,
    width: CW,
    height: CH,
    bytesPerRow: 0,
    rowsPerImage: 0,
    data: vData,
  });
  // Packet 4: multi-layer texture with rowsPerImage=0 (must populate rowsPerImage=height).
  const pktLayers = writeTexturePacket({
    textureId: 603,
    width: 8,
    height: 4,
    layers: 2,
    bytesPerRow: 8,
    rowsPerImage: 0,
    data: layerData,
  });

  const prefixPad = 64;
  const combined = new Uint8Array(
    prefixPad + pktY.byteLength + pktU.byteLength + pktV.byteLength + pktLayers.byteLength
  );
  let cursor = prefixPad;
  const offY = cursor;
  combined.set(pktY, cursor);
  cursor += pktY.byteLength;
  const offU = cursor;
  combined.set(pktU, cursor);
  cursor += pktU.byteLength;
  const offV = cursor;
  combined.set(pktV, cursor);
  cursor += pktV.byteLength;
  const offLayers = cursor;
  combined.set(pktLayers, cursor);

  const cmdSlice = combined.subarray(prefixPad);
  processor.executeCommandBuffer(cmdSlice);

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 4);

  // Every single upload must reference the exact underlying combined.buffer (zero copies!).
  assert.equal(wtCalls[0].bufferRef, combined.buffer);
  assert.equal(wtCalls[0].byteOffset, offY + 8 + 48);
  assert.equal(wtCalls[0].byteLength, YW * YH);
  assert.equal(wtCalls[0].dataLayout.bytesPerRow, YW);
  assert.equal(wtCalls[0].dataLayout.rowsPerImage, YH);
  assert.equal(wtCalls[0].bytes[0], 0x1a);
  assert.equal(wtCalls[0].bytes[YW * YH - 1], 0x2b);

  assert.equal(wtCalls[1].bufferRef, combined.buffer);
  assert.equal(wtCalls[1].byteOffset, offU + 8 + 48);
  assert.equal(wtCalls[1].byteLength, CW * CH);
  assert.equal(wtCalls[1].dataLayout.bytesPerRow, CW);
  assert.equal(wtCalls[1].bytes[0], 0x3c);
  assert.equal(wtCalls[1].bytes[CW * CH - 1], 0x4d);

  assert.equal(wtCalls[2].bufferRef, combined.buffer);
  assert.equal(wtCalls[2].byteOffset, offV + 8 + 48);
  assert.equal(wtCalls[2].byteLength, CW * CH);
  assert.equal(wtCalls[2].dataLayout.bytesPerRow, CW);
  assert.equal(wtCalls[2].bytes[0], 0x5e);
  assert.equal(wtCalls[2].bytes[CW * CH - 1], 0x6f);

  assert.equal(wtCalls[3].bufferRef, combined.buffer);
  assert.equal(wtCalls[3].byteOffset, offLayers + 8 + 48);
  assert.equal(wtCalls[3].byteLength, 64);
  assert.equal(wtCalls[3].dataLayout.bytesPerRow, 8);
  assert.equal(wtCalls[3].dataLayout.rowsPerImage, 4);
  assert.equal(wtCalls[3].bytes[0], 0x71);
  assert.equal(wtCalls[3].bytes[63], 0x82);

  const expectedTotalBytes = YW * YH + CW * CH * 2 + 64;
  const uploadStats = handler.getUploadStats();
  assert.deepEqual(uploadStats, {
    uploadCount: 4,
    uploadBytes: expectedTotalBytes,
    zeroCopyUploads: 4,
    scratchRepackUploads: 0,
    scratchBufferAllocations: 0,
    scratchBufferReuses: 0,
    scratchBufferBytes: 0,
    sabCopyCount: 0,
  });
  assert.deepEqual(handler.uploadStats, uploadStats);
  assert.equal(handler._scratchBuffer, null);
});

test('W191: upload telemetry distinguishes zero-copy, scratch-repack, oversized repack, SAB copy, and resets cleanly', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({
    device: gpu.device,
    chunkSize: 4 * 1024 * 1024,
    maxPoolBytes: 64 * 1024 * 1024,
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  // Cap persistent scratch at 4096 bytes to verify bounded growth and oversized fallback telemetry.
  const handler = tp.installTexturePackets(processor, { maxScratchBytes: 4096 });

  // Verify default cap constant is 16 MiB.
  assert.equal(tp.DEFAULT_MAX_SCRATCH_BYTES, 16 * 1024 * 1024);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 700, width: 256, height: 4, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 701, width: 20, height: 4, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 702, width: 20, height: 32, usage: 0x06, format: 1 })
  );

  // 1) Aligned non-SAB upload -> zero-copy direct writeTexture.
  const alignedPkt = writeTexturePacket({
    textureId: 700,
    width: 256,
    height: 4,
    bytesPerRow: 256,
    rowsPerImage: 4,
    data: new Uint8Array(1024),
    useSab: false,
  });
  processor.executeCommandBuffer(alignedPkt);

  // 2) Three repeated padded uploads (20x4 -> 256x4 = 1024B <= 4096B) -> 1 alloc + 2 reuses.
  let scratchRef = null;
  for (let i = 0; i < 3; i += 1) {
    const rowData = new Uint8Array(80);
    rowData.fill(0x20 + i);
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 701,
        width: 20,
        height: 4,
        bytesPerRow: 20,
        rowsPerImage: 4,
        data: rowData,
      })
    );
    if (i === 0) {
      scratchRef = handler._scratchBuffer;
      assert.ok(scratchRef instanceof Uint8Array);
      assert.equal(scratchRef.byteLength, 1024);
    } else {
      assert.equal(handler._scratchBuffer, scratchRef, 'Scratch Uint8Array must be reused');
    }
  }

  // 3) Oversized padded upload (20x32 -> 256x32 = 8192B > 4096B cap) -> one-off repack, NOT scratchRepackUploads.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 702,
      width: 20,
      height: 32,
      bytesPerRow: 20,
      rowsPerImage: 32,
      data: new Uint8Array(640),
    })
  );
  assert.equal(handler._scratchBuffer, scratchRef, 'Oversized upload must not replace persistent scratch');

  // 4) Aligned SAB upload (256x4 = 1024B) -> copies out of SAB using existing scratch (reuse #3), NOT zeroCopy.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 700,
      width: 256,
      height: 4,
      bytesPerRow: 256,
      rowsPerImage: 4,
      data: new Uint8Array(1024),
      useSab: true,
    })
  );

  // Verify copyFromSharedIfNeeded optional scratch reuse helper as well.
  const sabBuf = new SharedArrayBuffer(64);
  const sabView = new Uint8Array(sabBuf);
  sabView[0] = 0xaa;
  sabView[63] = 0xbb;
  const externalScratch = new Uint8Array(128);
  const sabCopied = tp.copyFromSharedIfNeeded(sabView, externalScratch);
  assert.equal(sabCopied.copiedFromSab, true);
  assert.equal(sabCopied.reusedScratch, true);
  assert.equal(sabCopied.data.buffer, externalScratch.buffer);
  assert.equal(sabCopied.data.byteLength, 64);
  assert.equal(sabCopied.data[0], 0xaa);
  assert.equal(sabCopied.data[63], 0xbb);

  const expectedBytes = 1024 + 3 * 1024 + 8192 + 1024;
  assert.deepEqual(handler.getUploadStats(), {
    uploadCount: 6,
    uploadBytes: expectedBytes,
    zeroCopyUploads: 1,
    scratchRepackUploads: 3,
    scratchBufferAllocations: 1,
    scratchBufferReuses: 3,
    scratchBufferBytes: 1024,
    sabCopyCount: 1,
  });
  assert.equal(handler.getStats().repackedUploads, 4);
  assert.equal(handler.getStats().scratchRepackUploads, 3);
  assert.equal(handler.getStats().uploadCount, 6);
  assert.equal(handler.getStats().uploadBytes, expectedBytes);
  assert.equal(handler.getStats().bytesUploaded, expectedBytes);

  // Reset zeros all upload telemetry cleanly.
  handler.reset();
  assert.deepEqual(handler.getUploadStats(), {
    uploadCount: 0,
    uploadBytes: 0,
    zeroCopyUploads: 0,
    scratchRepackUploads: 0,
    scratchBufferAllocations: 0,
    scratchBufferReuses: 0,
    scratchBufferBytes: 0,
    sabCopyCount: 0,
  });
});

test('W265 (#1127): BC1-BC7 mip layout & 4x4 block-extent alignment cover all 14 formats across 1x1, 2x2, 3x3, 4x4, and multi-row mips', () => {
  const allBcFormats = [
    ['bc1-rgba-unorm', 8],
    ['bc1-rgba-unorm-srgb', 8],
    ['bc2-rgba-unorm', 16],
    ['bc2-rgba-unorm-srgb', 16],
    ['bc3-rgba-unorm', 16],
    ['bc3-rgba-unorm-srgb', 16],
    ['bc4-r-unorm', 8],
    ['bc4-r-snorm', 8],
    ['bc5-rg-unorm', 16],
    ['bc5-rg-snorm', 16],
    ['bc6h-rgb-ufloat', 16],
    ['bc6h-rgb-float', 16],
    ['bc7-rgba-unorm', 16],
    ['bc7-rgba-unorm-srgb', 16],
  ];

  for (const [format, bytesPerBlock] of allBcFormats) {
    // 1x1, 2x2, 3x3 tail mips must clamp copy extents to 4x4 and align bytesPerRow to 256B.
    for (const [w, h] of [
      [1, 1],
      [2, 2],
      [3, 3],
      [1, 3],
      [2, 1],
    ]) {
      const tpLayout = tp.calcMipUploadLayout(format, w, h, 1);
      const tfLayout = tf.calcMipUploadLayout(format, w, h, 1);
      assert.deepEqual(tpLayout, tfLayout, `${format} ${w}x${h} layout must match across modules`);
      assert.equal(tpLayout.copyWidth, 4);
      assert.equal(tpLayout.copyHeight, 4);
      assert.equal(tpLayout.blocksX, 1);
      assert.equal(tpLayout.blocksY, 1);
      assert.equal(tpLayout.bytesPerBlock, bytesPerBlock);
      assert.equal(tpLayout.unalignedBytesPerRow, bytesPerBlock);
      assert.equal(tpLayout.bytesPerRow, 256);
      assert.equal(tpLayout.rowsPerImage, 1);
      assert.equal(tpLayout.tightTotalBytes, bytesPerBlock);
      assert.equal(tpLayout.totalBytes, 256);
      assert.equal(tpLayout.isCompressed, true);
      assert.equal(tpLayout.isTailMip, true);
      assert.equal(tpLayout.wasBlockClamped, true);
      assert.equal(tpLayout.needsRowPadding, true);

      assert.deepEqual(tp.alignCopySizeForFormat(format, { width: w, height: h, depthOrArrayLayers: 1 }), {
        width: 4,
        height: 4,
        depthOrArrayLayers: 1,
      });
      const tfExtent = tf.alignCopyExtentForFormat(format, w, h, 1);
      assert.equal(tfExtent.width, 4);
      assert.equal(tfExtent.height, 4);
      assert.equal(tfExtent.wasClamped, true);
    }

    // 4x4 single-block mip: already 4x4 aligned (not clamped, not tail mip), 1 block row -> 256B padded.
    const mip4 = tp.calcMipUploadLayout(format, 4, 4, 1);
    assert.equal(mip4.copyWidth, 4);
    assert.equal(mip4.copyHeight, 4);
    assert.equal(mip4.blocksX, 1);
    assert.equal(mip4.blocksY, 1);
    assert.equal(mip4.unalignedBytesPerRow, bytesPerBlock);
    assert.equal(mip4.bytesPerRow, 256);
    assert.equal(mip4.isTailMip, false);
    assert.equal(mip4.wasBlockClamped, false);

    // 5x3 non-multiple-of-4 sub-mip -> 2x1 blocks -> 8x4 copy extent.
    const mip5x3 = tp.calcMipUploadLayout(format, 5, 3, 1);
    assert.equal(mip5x3.copyWidth, 8);
    assert.equal(mip5x3.copyHeight, 4);
    assert.equal(mip5x3.blocksX, 2);
    assert.equal(mip5x3.blocksY, 1);
    assert.equal(mip5x3.unalignedBytesPerRow, bytesPerBlock * 2);
    assert.equal(mip5x3.bytesPerRow, 256);
    assert.equal(mip5x3.wasBlockClamped, true);

    // 8x8 multi-row mip -> 2x2 blocks -> 2 rows x 256B = 512B.
    const mip8 = tp.calcMipUploadLayout(format, 8, 8, 1);
    assert.equal(mip8.copyWidth, 8);
    assert.equal(mip8.copyHeight, 8);
    assert.equal(mip8.blocksX, 2);
    assert.equal(mip8.blocksY, 2);
    assert.equal(mip8.unalignedBytesPerRow, bytesPerBlock * 2);
    assert.equal(mip8.bytesPerRow, 256);
    assert.equal(mip8.rowsPerImage, 2);
    assert.equal(mip8.tightTotalBytes, bytesPerBlock * 4);
    assert.equal(mip8.totalBytes, 512);

    // Exact 256-byte-aligned row width (128px for 8B/block BC1/BC4; 64px for 16B/block BC2/3/5/6H/7).
    const alignedW = (256 / bytesPerBlock) * 4;
    const exactMip = tp.calcMipUploadLayout(format, alignedW, 8, 1);
    assert.equal(exactMip.unalignedBytesPerRow, 256);
    assert.equal(exactMip.bytesPerRow, 256);
    assert.equal(exactMip.needsRowPadding, false);
    assert.equal(exactMip.totalBytes, 512);

    // texture_formats.js calcMipLayout accepts WebGPU format names directly.
    const baseLayout = tf.calcMipLayout(format, 2, 2, 6);
    assert.equal(baseLayout.blocksX, 1);
    assert.equal(baseLayout.blocksY, 1);
    assert.equal(baseLayout.unalignedBytesPerRow, bytesPerBlock);
    assert.equal(baseLayout.bytesPerRow, 256);
    assert.equal(baseLayout.totalBytes, 256 * 6);
  }
});

test('W265 (#1127): streamed BC1-BC7 mip chains clamp 1x1/2x2/3x3 tail mips to 4x4 and pad 256B rows under strict WebGPU validation', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });

  // Layer strict WebGPU BC copy validation on top of mock device.
  const origWriteTexture = gpu.device.queue.writeTexture;
  gpu.device.queue.writeTexture = (destination, data, dataLayout, size) => {
    const fmt = destination.texture && destination.texture.format;
    const info = tp.getTextureFormatBlockInfo(fmt);
    if (info && info.isCompressed) {
      assert.equal(size.width % 4, 0, `writeTexture BC copySize.width (${size.width}) must be a multiple of 4`);
      assert.equal(size.height % 4, 0, `writeTexture BC copySize.height (${size.height}) must be a multiple of 4`);
      const blocksX = size.width / 4;
      if (dataLayout && dataLayout.bytesPerRow !== undefined) {
        assert.ok(
          dataLayout.bytesPerRow >= blocksX * info.bytesPerBlock,
          `writeTexture BC bytesPerRow (${dataLayout.bytesPerRow}) < ${blocksX * info.bytesPerBlock}`
        );
      }
    }
    return origWriteTexture(destination, data, dataLayout, size);
  };
  const origCreateCommandEncoder = gpu.device.createCommandEncoder;
  gpu.device.createCommandEncoder = () => {
    const enc = origCreateCommandEncoder();
    const origC2t = enc.copyBufferToTexture;
    enc.copyBufferToTexture = (source, destination, copySize) => {
      const fmt = destination.texture && destination.texture.format;
      const info = tp.getTextureFormatBlockInfo(fmt);
      if (info && info.isCompressed) {
        assert.equal(copySize.width % 4, 0, `copyBufferToTexture BC copySize.width (${copySize.width}) must be a multiple of 4`);
        assert.equal(copySize.height % 4, 0, `copyBufferToTexture BC copySize.height (${copySize.height}) must be a multiple of 4`);
        assert.equal(source.bytesPerRow % 256, 0, 'copyBufferToTexture bytesPerRow must be 256B aligned');
        const blocksX = copySize.width / 4;
        assert.ok(source.bytesPerRow >= blocksX * info.bytesPerBlock);
      }
      return origC2t(source, destination, copySize);
    };
    return enc;
  };

  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  // Create 16x16 5-mip BC1 (code 50, 8B/block) and BC7 (code 62, 16B/block), plus 6-layer BC3 cubemap (code 54).
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 801, width: 16, height: 16, mips: 5, usage: 0x06, format: 50 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 802, width: 16, height: 16, mips: 5, usage: 0x06, format: 62 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 803, width: 16, height: 16, layers: 6, mips: 5, usage: 0x06, format: 54 })
  );

  // Stream complete BC1 mip chain: 16x16 (mip0), 8x8 (mip1), 4x4 (mip2), 2x2 (mip3), 1x1 (mip4).
  const bc1Mips = [
    { mip: 0, w: 16, h: 16, bpr: 32, rows: 4, bytes: 128 },
    { mip: 1, w: 8, h: 8, bpr: 0, rows: 0, bytes: 32 }, // implicit tight packing (2x2 blocks * 8B)
    { mip: 2, w: 4, h: 4, bpr: 8, rows: 1, bytes: 8 },
    { mip: 3, w: 2, h: 2, bpr: 8, rows: 2, bytes: 8 }, // texel-height rowsPerImage=2 safely clamped to 1 block row
    { mip: 4, w: 1, h: 1, bpr: 0, rows: 0, bytes: 8 }, // 1x1 tail mip -> 4x4 copySize
  ];

  for (const m of bc1Mips) {
    const payload = new Uint8Array(m.bytes);
    for (let i = 0; i < payload.length; i += 1) payload[i] = ((m.mip + 1) * 17 + i) & 0xff;
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 801,
        mipLevel: m.mip,
        width: m.w,
        height: m.h,
        bytesPerRow: m.bpr,
        rowsPerImage: m.rows,
        data: payload,
        useSab: m.mip >= 3, // exercise SAB copy on 2x2 and 1x1 tail mips
      })
    );
  }

  // Stream BC7 2x2 and 1x1 tail mips (16B/block) + 6-layer BC3 2x2 cubemap tail mip (6 * 16B = 96B).
  const bc7Tail = new Uint8Array(16);
  for (let i = 0; i < 16; i += 1) bc7Tail[i] = 0xc0 + i;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 802,
      mipLevel: 3,
      width: 2,
      height: 2,
      bytesPerRow: 16,
      rowsPerImage: 1,
      data: bc7Tail,
    })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 802,
      mipLevel: 4,
      width: 1,
      height: 1,
      bytesPerRow: 0,
      rowsPerImage: 0,
      data: bc7Tail,
    })
  );

  const cubeTail = new Uint8Array(6 * 16);
  for (let face = 0; face < 6; face += 1) {
    for (let b = 0; b < 16; b += 1) cubeTail[face * 16 + b] = ((face + 1) << 4) | b;
  }
  // Poison persistent scratch buffer before cubemap upload to confirm zeroed padding tails on all 6 faces.
  assert.ok(handler._scratchBuffer instanceof Uint8Array);
  handler._scratchBuffer.fill(0xee);
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 803,
      mipLevel: 3,
      width: 2,
      height: 2,
      layers: 6,
      bytesPerRow: 16,
      rowsPerImage: 1,
      data: cubeTail,
    })
  );

  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(c2tCalls.length, 8);

  // Verify BC1 mip 1 (8x8 implicit tight packing): 2 block rows x 256B, copySize 8x8.
  assert.deepEqual(c2tCalls[1].copySize, { width: 8, height: 8, depthOrArrayLayers: 1 });
  assert.equal(c2tCalls[1].source.bytesPerRow, 256);
  assert.equal(c2tCalls[1].source.rowsPerImage, 2);
  for (let p = 16; p < 256; p += 1) {
    assert.equal(c2tCalls[1].bytes[p], 0, 'BC1 mip1 row 0 padding must be zero');
    assert.equal(c2tCalls[1].bytes[256 + p], 0, 'BC1 mip1 row 1 padding must be zero');
  }

  // Verify BC1 mip 3 (2x2) & mip 4 (1x1) clamped to 4x4 copy extents and 1 block row.
  assert.deepEqual(c2tCalls[3].copySize, { width: 4, height: 4, depthOrArrayLayers: 1 });
  assert.equal(c2tCalls[3].source.rowsPerImage, 1);
  assert.deepEqual(c2tCalls[4].copySize, { width: 4, height: 4, depthOrArrayLayers: 1 });
  assert.equal(c2tCalls[4].source.rowsPerImage, 1);

  // Verify BC7 tail mips (2x2 and 1x1) clamped to 4x4 with exact 16-byte block and zeroed padding.
  assert.deepEqual(c2tCalls[5].copySize, { width: 4, height: 4, depthOrArrayLayers: 1 });
  assert.deepEqual(Array.from(c2tCalls[5].bytes.subarray(0, 16)), Array.from(bc7Tail));
  assert.deepEqual(c2tCalls[6].copySize, { width: 4, height: 4, depthOrArrayLayers: 1 });
  assert.deepEqual(Array.from(c2tCalls[6].bytes.subarray(0, 16)), Array.from(bc7Tail));

  // Verify 6-face BC3 2x2 cubemap tail mip: 6 slices x 256B, copySize 4x4x6, zeroed tails.
  const cubeCall = c2tCalls[7];
  assert.deepEqual(cubeCall.copySize, { width: 4, height: 4, depthOrArrayLayers: 6 });
  assert.equal(cubeCall.source.bytesPerRow, 256);
  assert.equal(cubeCall.source.rowsPerImage, 1);
  for (let face = 0; face < 6; face += 1) {
    const base = face * 256;
    assert.deepEqual(
      Array.from(cubeCall.bytes.subarray(base, base + 16)),
      Array.from(cubeTail.subarray(face * 16, face * 16 + 16))
    );
    for (let p = 16; p < 256; p += 1) {
      assert.equal(cubeCall.bytes[base + p], 0, `cubemap face ${face} padding byte ${p} must be zero`);
    }
  }

  // Also test direct queue.writeTexture path (no stagingPool) for 1x1 & 2x2 BC7 tail mips.
  const directProcessor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const directHandler = tp.installTexturePackets(directProcessor);
  directProcessor.executeCommandBuffer(
    createTexture2Packet({ id: 810, width: 8, height: 8, mips: 4, usage: 0x06, format: 62 })
  );
  directProcessor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 810,
      mipLevel: 3,
      width: 1,
      height: 1,
      bytesPerRow: 0,
      rowsPerImage: 0,
      data: bc7Tail,
    })
  );
  // Direct handler.writeTexture also clamps 2x2 BC7 copySize to 4x4.
  directHandler.writeTexture(
    { texture: directProcessor.textures.get(810), mipLevel: 2, origin: { x: 0, y: 0, z: 0 } },
    bc7Tail,
    { offset: 0, bytesPerRow: 16, rowsPerImage: 1 },
    { width: 2, height: 2, depthOrArrayLayers: 1 },
    810
  );
  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 2);
  assert.deepEqual(wtCalls[0].size, { width: 4, height: 4, depthOrArrayLayers: 1 });
  assert.deepEqual(wtCalls[1].size, { width: 4, height: 4, depthOrArrayLayers: 1 });
  assert.equal(handler.getStats().errors, 0);
  assert.equal(directHandler.getStats().errors, 0);
});

test('W265 (#1127 / #973): A8_UNORM -> rgba8unorm font atlas expansion widens alpha to (0,0,0,a), aligns 256B rows, and reuses scratch', () => {
  // 1) Unit test tight & 256B-aligned expansion in both texture_formats.js and texture_packets.js.
  const width = 5;
  const height = 3;
  // Use strided source (srcBytesPerRow = 8 > width = 5) with non-zero slack bytes to ensure slack is ignored.
  const srcStride = 8;
  const a8Src = new Uint8Array(srcStride * height);
  a8Src.fill(0xee);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      a8Src[y * srcStride + x] = (y * 32 + x * 7 + 11) & 0xff;
    }
  }

  // Tight expansion (texture_formats.expandA8ToRgba8 default alignBytesPerRow = 1).
  const tight = tf.expandA8ToRgba8(a8Src, width, height, { srcBytesPerRow: srcStride });
  assert.equal(tight.byteLength, width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const off = (y * width + x) * 4;
      assert.equal(tight[off], 0);
      assert.equal(tight[off + 1], 0);
      assert.equal(tight[off + 2], 0);
      assert.equal(tight[off + 3], a8Src[y * srcStride + x]);
    }
  }

  // 256-byte aligned expansion into a poisoned reusable scratch pool (SAB input!).
  const sab = new SharedArrayBuffer(a8Src.byteLength);
  new Uint8Array(sab).set(a8Src);
  const pool = new tp.TextureScratchPool({ maxBytes: 4096 });

  const first = tp.expandA8ToRgba8(sab, width, height, {
    srcBytesPerRow: srcStride,
    scratchPool: pool,
  });
  assert.equal(first.bytesPerRow, 256);
  assert.equal(first.unalignedBytesPerRow, 20);
  assert.equal(first.rowsPerImage, 3);
  assert.equal(first.totalBytes, 768);
  assert.equal(first.copiedFromSab, true);
  assert.equal(first.reusedScratch, false);
  assert.equal(first.pooledScratch, true);
  assert.equal(first.data.buffer instanceof ArrayBuffer, true);

  // Poison entire pool buffer with 0xff and expand again -> must reuse buffer AND strictly zero RGB + padding tails!
  pool.currentBuffer.fill(0xff);
  const second = tp.expandA8ToRgba8(sab, width, height, {
    srcBytesPerRow: srcStride,
    scratchPool: pool,
  });
  assert.equal(second.reusedScratch, true);
  assert.equal(second.data.buffer, pool.currentBuffer.buffer);
  for (let y = 0; y < height; y += 1) {
    const rowBase = y * 256;
    for (let x = 0; x < width; x += 1) {
      assert.equal(second.data[rowBase + x * 4 + 0], 0);
      assert.equal(second.data[rowBase + x * 4 + 1], 0);
      assert.equal(second.data[rowBase + x * 4 + 2], 0);
      assert.equal(second.data[rowBase + x * 4 + 3], a8Src[y * srcStride + x]);
    }
    for (let p = 20; p < 256; p += 1) {
      assert.equal(second.data[rowBase + p], 0, `A8 row ${y} padding byte ${p} must be zeroed`);
    }
  }
  assert.equal(pool.getStats().allocations, 1);
  assert.equal(pool.getStats().reuses, 1);

  // 2) End-to-end handler.writeA8Texture staged + direct uploads with zero per-frame scratch churn.
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 128 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  // Font atlas target is rgba8unorm (code 22) per #973.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 850, width: 100, height: 32, usage: 0x06, format: 22 })
  );
  const atlasTex = processor.textures.get(850);
  const glyphData = new Uint8Array(100 * 32);

  let scratchRef = null;
  for (let frame = 0; frame < 4; frame += 1) {
    stagingPool.beginFrame(frame);
    glyphData.fill(0x40 + frame);
    glyphData[0] = 0x90 + frame;
    glyphData[99] = 0xa0 + frame;
    const res = handler.writeA8Texture(
      { texture: atlasTex, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
      glyphData,
      { width: 100, height: 32, depthOrArrayLayers: 1 }
    );
    assert.equal(res.uploaded, true);
    assert.equal(res.staged, true);
    assert.equal(res.unalignedBytesPerRow, 400);
    assert.equal(res.bytesPerRow, 512);
    assert.equal(res.pooledScratch, true);
    if (frame === 0) {
      assert.equal(res.reusedScratch, false);
      scratchRef = handler._scratchBuffer;
      assert.equal(scratchRef.byteLength, 512 * 32);
      scratchRef.fill(0xdd, 400, 512); // poison padding tail
    } else {
      assert.equal(res.reusedScratch, true);
      assert.equal(handler._scratchBuffer, scratchRef);
    }
    stagingPool.endFrame();
  }

  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(c2tCalls.length, 4);
  const frame1Bytes = c2tCalls[1].bytes;
  assert.deepEqual(Array.from(frame1Bytes.subarray(0, 4)), [0, 0, 0, 0x91]);
  assert.deepEqual(Array.from(frame1Bytes.subarray(99 * 4, 100 * 4)), [0, 0, 0, 0xa1]);
  for (let p = 400; p < 512; p += 1) {
    assert.equal(frame1Bytes[p], 0, `staged A8 atlas padding byte ${p} must be zero`);
  }
  assert.equal(handler.getStats().scratchBufferAllocations, 1);
  assert.equal(handler.getStats().scratchBufferReuses, 3);
  assert.equal(handler.getStats().scratchRepackUploads, 4);

  // Direct mode (preferDirect: true) expands tightly (400B/row) with SAB input and zero SAB views reaching WebGPU.
  const sabGlyph = new SharedArrayBuffer(100 * 32);
  new Uint8Array(sabGlyph).fill(0x7f);
  const directRes = handler.writeA8Texture(
    { texture: atlasTex, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    sabGlyph,
    { width: 100, height: 32, depthOrArrayLayers: 1 },
    { preferDirect: true }
  );
  assert.equal(directRes.uploaded, true);
  assert.equal(directRes.staged, false);
  assert.equal(directRes.bytesPerRow, 400);
  assert.equal(directRes.copiedFromSab, true);
  assert.equal(directRes.reusedScratch, true);
  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 1);
  assert.equal(wtCalls[0].isSab, false);
  assert.deepEqual(Array.from(wtCalls[0].bytes.subarray(0, 4)), [0, 0, 0, 0x7f]);
});

test('W265 (#1127): TextureScratchPool enforces upper cap, multi-slot isolation, SAB rejection, and opt-in vs safe-copy anti-aliasing', () => {
  // 1) Bounded upper-cap safety: oversized one-off uploads must NEVER pin giant RAM.
  const pool = new tf.TextureScratchPool({ maxBytes: 2048 });
  const a1 = pool.acquire(1024);
  assert.equal(a1.pooled, true);
  assert.equal(a1.reused, false);
  assert.equal(a1.oversized, false);
  const retainedRef = pool.currentBuffer;
  assert.equal(retainedRef.byteLength, 1024);

  const a2 = pool.acquire(512);
  assert.equal(a2.pooled, true);
  assert.equal(a2.reused, true);
  assert.equal(a2.buffer, retainedRef);
  assert.equal(a2.view.byteLength, 512);

  // Oversized 8192B (> 2048B cap): one-off standalone buffer, retained buffer stays 1024B.
  const huge = pool.acquire(8192);
  assert.equal(huge.pooled, false);
  assert.equal(huge.reused, false);
  assert.equal(huge.oversized, true);
  assert.equal(huge.view.byteLength, 8192);
  assert.equal(pool.currentBuffer, retainedRef);
  assert.equal(pool.retainedBytes, 1024);
  assert.deepEqual(pool.getStats(), {
    allocations: 1,
    reuses: 1,
    oversizedFallbacks: 1,
    retainedBytes: 1024,
    maxBytes: 2048,
    maxSlots: 1,
    activeSlots: 1,
  });

  // 2) Safe copy semantics (anti-aliasing): default calls and copyResult:true never alias stored slices!
  const rowA = new Uint8Array(16).fill(0x11);
  const rowB = new Uint8Array(16).fill(0x22);

  // padTextureRowsForWriteTexture default (no scratch) -> independent buffers.
  const padSafeA = tf.padTextureRowsForWriteTexture(rowA, 8, 256, 2);
  const padSafeB = tf.padTextureRowsForWriteTexture(rowB, 8, 256, 2);
  assert.notEqual(padSafeA.buffer, padSafeB.buffer);
  assert.equal(padSafeA[0], 0x11);
  assert.equal(padSafeB[0], 0x22);

  // padTextureRowsForWriteTexture with pool + copyResult:true -> owned non-aliased copy.
  const padPoolOwnedA = tf.padTextureRowsForWriteTexture(rowA, 8, 256, 2, {
    scratchPool: pool,
    copyResult: true,
  });
  const padPoolReusedB = tf.padTextureRowsForWriteTexture(rowB, 8, 256, 2, pool);
  assert.notEqual(padPoolOwnedA.buffer, padPoolReusedB.buffer);
  assert.equal(padPoolOwnedA[0], 0x11);
  assert.equal(padPoolReusedB[0], 0x22);

  // expandA8ToRgba8 with copyResult:true & pool.snapshot() prevent aliasing across consecutive frames.
  const a8First = new Uint8Array([0x12, 0x34, 0x56, 0x78]);
  const a8Second = new Uint8Array([0xaa, 0xbb, 0xcc, 0xdd]);
  const expOwned = tp.expandA8ToRgba8(a8First, 2, 2, { scratchPool: pool, copyResult: true });
  const expScratch = tp.expandA8ToRgba8(a8Second, 2, 2, { scratchPool: pool });
  const snap = pool.snapshot(expScratch.data);
  // Overwrite scratch with new data and verify both expOwned and snap stayed intact.
  tp.expandA8ToRgba8(new Uint8Array([0xff, 0xff, 0xff, 0xff]), 2, 2, { scratchPool: pool });
  assert.equal(expOwned.data[3], 0x12);
  assert.equal(expOwned.data[256 + 7], 0x78);
  assert.equal(snap[3], 0xaa);
  assert.equal(snap[256 + 7], 0xdd);

  // repackRowsTo256 with copyResult:true detaches even already-256B-aligned input.
  const alignedIn = new Uint8Array(256).fill(0x55);
  const detached = tp.repackRowsTo256(alignedIn, 256, 1, 1, { copyResult: true });
  assert.notEqual(detached.data.buffer, alignedIn.buffer);
  alignedIn[0] = 0x99;
  assert.equal(detached.data[0], 0x55);

  // 3) Multi-slot pool isolates concurrent slots while enforcing combined maxBytes cap.
  const multiPool = new tp.TextureScratchPool({ maxBytes: 1024, maxSlots: 2 });
  const slot0 = multiPool.acquire(512, { slot: 0 });
  const slot1 = multiPool.acquire(512, { slot: 1 });
  assert.equal(slot0.pooled, true);
  assert.equal(slot1.pooled, true);
  assert.notEqual(slot0.buffer, slot1.buffer);
  slot0.view.fill(0x33);
  slot1.view.fill(0x77);
  assert.equal(slot0.view[0], 0x33);
  assert.equal(slot1.view[0], 0x77);
  // Growing slot 1 to 768B would push total retained bytes (512 + 768 = 1280) > 1024 cap -> oversized fallback.
  const overCap = multiPool.acquire(768, { slot: 1 });
  assert.equal(overCap.oversized, true);
  assert.equal(overCap.pooled, false);
  assert.equal(multiPool.retainedBytes, 1024);

  // 4) SAB-backed scratch buffers are rejected safely (never returned as WebGPU destination).
  const sabScratch = new Uint8Array(new SharedArrayBuffer(1024));
  const paddedSafe = tf.padTextureRowsForWriteTexture(rowA, 8, 256, 2, sabScratch);
  assert.equal(paddedSafe.buffer instanceof ArrayBuffer, true);
  const expandedSafe = tp.expandA8ToRgba8(a8First, 2, 2, { scratchBuffer: sabScratch });
  assert.equal(expandedSafe.data.buffer instanceof ArrayBuffer, true);
  assert.equal(expandedSafe.reusedScratch, false);

  // 5) Reset releases retained RAM cleanly.
  pool.reset();
  multiPool.destroy();
  assert.equal(pool.currentBuffer, null);
  assert.equal(pool.retainedBytes, 0);
  assert.equal(multiPool.retainedBytes, 0);
});

test('W303 (#1219): pre-allocated WebGPU descriptor structs are reused across consecutive writeTexture & copyBufferToTexture uploads with every field overwritten', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 901, width: 64, height: 64, layers: 4, mips: 4, usage: 0x06, format: 22 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 902, width: 128, height: 128, layers: 1, mips: 1, usage: 0x06, format: 1 })
  );
  const texA = processor.textures.get(901);
  const texB = processor.textures.get(902);

  // 1) Consecutive direct queue.writeTexture calls with differing fields (including aspect & rowsPerImage).
  const res1 = handler.writeTexture(
    { texture: texA, mipLevel: 2, origin: { x: 4, y: 8, z: 2 }, aspect: 'depth-only' },
    new Uint8Array(256 * 4),
    { offset: 16, bytesPerRow: 256, rowsPerImage: 4 },
    { width: 8, height: 4, depthOrArrayLayers: 1 },
    901
  );
  const res2 = handler.writeTexture(
    { texture: texB, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    new Uint8Array(128),
    { offset: 0, bytesPerRow: 128 },
    { width: 128, height: 1, depthOrArrayLayers: 1 },
    902
  );
  assert.equal(res1.uploaded, true);
  assert.equal(res2.uploaded, true);

  // Also issue two consecutive WriteTexture command packets on the fast path.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 901,
      mipLevel: 0,
      originX: 0,
      originY: 0,
      originZ: 1,
      width: 64,
      height: 2,
      layers: 1,
      bytesPerRow: 256,
      rowsPerImage: 2,
      data: new Uint8Array(512),
    })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 901,
      mipLevel: 1,
      originX: 8,
      originY: 4,
      originZ: 0,
      width: 16,
      height: 2,
      layers: 1,
      bytesPerRow: 256,
      rowsPerImage: 0,
      data: new Uint8Array(512),
    })
  );

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 4);

  // Every writeTexture call reuses the exact same pre-allocated descriptor instances.
  for (let i = 1; i < wtCalls.length; i += 1) {
    assert.equal(wtCalls[i].destinationRef, wtCalls[0].destinationRef, 'GPUTexelCopyTextureInfo must be reused');
    assert.equal(wtCalls[i].originRef, wtCalls[0].originRef, 'GPUOrigin3D must be reused');
    assert.equal(wtCalls[i].dataLayoutRef, wtCalls[0].dataLayoutRef, 'GPUTexelCopyBufferLayout must be reused');
    assert.equal(wtCalls[i].sizeRef, wtCalls[0].sizeRef, 'GPUExtent3DDict must be reused');
  }

  // Verify every field was overwritten cleanly on call 2 (no stale aspect, mipLevel, origin, offset, or rowsPerImage).
  assert.equal(wtCalls[0].destination.texture, texA);
  assert.equal(wtCalls[0].destination.mipLevel, 2);
  assert.deepEqual(wtCalls[0].destination.origin, { x: 4, y: 8, z: 2 });
  assert.equal(wtCalls[0].destination.aspect, 'depth-only');
  assert.equal(wtCalls[0].dataLayout.offset, 16);
  assert.equal(wtCalls[0].dataLayout.bytesPerRow, 256);
  assert.equal(wtCalls[0].dataLayout.rowsPerImage, 4);
  assert.deepEqual(wtCalls[0].size, { width: 8, height: 4, depthOrArrayLayers: 1 });

  assert.equal(wtCalls[1].destination.texture, texB);
  assert.equal(wtCalls[1].destination.mipLevel, 0);
  assert.deepEqual(wtCalls[1].destination.origin, { x: 0, y: 0, z: 0 });
  assert.equal(wtCalls[1].destination.aspect, 'all');
  assert.equal(wtCalls[1].dataLayout.offset, 0);
  assert.equal(wtCalls[1].dataLayout.bytesPerRow, 128);
  assert.equal(wtCalls[1].dataLayout.rowsPerImage, undefined);
  assert.deepEqual(wtCalls[1].size, { width: 128, height: 1, depthOrArrayLayers: 1 });

  assert.equal(wtCalls[3].destination.mipLevel, 1);
  assert.deepEqual(wtCalls[3].destination.origin, { x: 8, y: 4, z: 0 });
  assert.deepEqual(wtCalls[3].size, { width: 16, height: 2, depthOrArrayLayers: 1 });

  // 2) Consecutive copyBufferToTexture calls reuse GPUTexelCopyBufferInfo, GPUTexelCopyTextureInfo, and GPUExtent3DDict.
  const encoder = gpu.device.createCommandEncoder();
  const buf1 = gpu.device.createBuffer({ size: 1024, usage: 0x0004 | 0x0008 });
  const buf2 = gpu.device.createBuffer({ size: 1024, usage: 0x0004 | 0x0008 });

  handler.copyBufferToTexture(
    encoder,
    { buffer: buf1, offset: 256, bytesPerRow: 256, rowsPerImage: 2 },
    { texture: texA, mipLevel: 1, origin: { x: 4, y: 4, z: 1 }, aspect: 'stencil-only' },
    { width: 8, height: 2, depthOrArrayLayers: 1 }
  );
  handler.copyBufferToTexture(
    encoder,
    { buffer: buf2, offset: 0, bytesPerRow: 256, rowsPerImage: 1 },
    { texture: texB, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 64, height: 1, depthOrArrayLayers: 1 }
  );

  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(c2tCalls.length, 2);
  assert.equal(c2tCalls[1].sourceRef, c2tCalls[0].sourceRef, 'GPUTexelCopyBufferInfo must be reused');
  assert.equal(c2tCalls[1].destinationRef, c2tCalls[0].destinationRef, 'GPUTexelCopyTextureInfo must be reused');
  assert.equal(c2tCalls[1].originRef, c2tCalls[0].originRef, 'GPUOrigin3D must be reused');
  assert.equal(c2tCalls[1].copySizeRef, c2tCalls[0].copySizeRef, 'GPUExtent3DDict must be reused');

  assert.equal(c2tCalls[0].source.buffer, buf1);
  assert.equal(c2tCalls[0].source.offset, 256);
  assert.equal(c2tCalls[0].destination.aspect, 'stencil-only');
  assert.deepEqual(c2tCalls[0].destination.origin, { x: 4, y: 4, z: 1 });
  assert.equal(c2tCalls[1].source.buffer, buf2);
  assert.equal(c2tCalls[1].source.offset, 0);
  assert.equal(c2tCalls[1].destination.aspect, 'all');
  assert.deepEqual(c2tCalls[1].destination.origin, { x: 0, y: 0, z: 0 });
  assert.deepEqual(c2tCalls[1].copySize, { width: 64, height: 1, depthOrArrayLayers: 1 });

  const stats = handler.getStats();
  assert.equal(stats.uploadsCount, 6);
  assert.equal(stats.uploadCount, 6);
  assert.equal(stats.reusedDescriptorCount, 6);
  assert.equal(stats.clampedSubRegionCount, 0);
  assert.equal(stats.uploadBytesTotal, stats.uploadBytes);
  assert.equal(stats.uploadBytesTotal, stats.bytesUploaded);
});

test('W303 (#1219): sub-region dirty-rect bounds clamping validates mip dimensions across 1D/2D/2D-array/3D/UI-atlas paths without WebGPU out-of-bounds errors', () => {
  // 1) Pure helper checks for calcMipDimension & clampSubRegionToMipBounds.
  assert.equal(tp.calcMipDimension(256, 0), 256);
  assert.equal(tp.calcMipDimension(256, 2), 64);
  assert.equal(tp.calcMipDimension(256, 12), 1);
  assert.equal(tp.calcMipDimension(256, 40), 1);

  // 2D mip 2 of 256x128 -> 64x32; dirty rect at (48, 24) size 32x16 -> clamped to 16x8.
  const c2d = tp.clampSubRegionToMipBounds(
    { width: 256, height: 128, depthOrArrayLayers: 1, mipLevelCount: 9, dimension: '2d', format: 'rgba8unorm' },
    2,
    { x: 48, y: 24, z: 0 },
    { width: 32, height: 16, depthOrArrayLayers: 1 }
  );
  assert.equal(c2d.valid, true);
  assert.equal(c2d.clamped, true);
  assert.equal(c2d.outOfBounds, false);
  assert.equal(c2d.width, 16);
  assert.equal(c2d.height, 8);
  assert.equal(c2d.depthOrArrayLayers, 1);
  assert.equal(c2d.mipWidth, 64);
  assert.equal(c2d.mipHeight, 32);

  // 2D array: layers (6) do NOT shrink at mip 2, so z=4 depth=4 clamps to depth=2.
  const cArr = tp.clampSubRegionToMipBounds(
    { width: 64, height: 64, depthOrArrayLayers: 6, mipLevelCount: 5, dimension: '2d', format: 'rgba8unorm' },
    2,
    { x: 0, y: 0, z: 4 },
    { width: 16, height: 16, depthOrArrayLayers: 4 }
  );
  assert.equal(cArr.valid, true);
  assert.equal(cArr.clamped, true);
  assert.equal(cArr.depthOrArrayLayers, 2);
  assert.equal(cArr.mipDepth, 6);

  // 3D texture: depth (8) DOES shrink at mip 2 -> mipDepth=2; z=1 depth=4 clamps to depth=1.
  const c3d = tp.clampSubRegionToMipBounds(
    { width: 64, height: 64, depthOrArrayLayers: 8, mipLevelCount: 5, dimension: '3d', format: 'rgba8unorm' },
    2,
    { x: 0, y: 0, z: 1 },
    { width: 16, height: 16, depthOrArrayLayers: 4 }
  );
  assert.equal(c3d.valid, true);
  assert.equal(c3d.clamped, true);
  assert.equal(c3d.depthOrArrayLayers, 1);
  assert.equal(c3d.mipDepth, 2);

  // Completely out-of-bounds dirty rect (origin.x >= mipWidth) -> valid=false, clamped=true, outOfBounds=true.
  const cOob = tp.clampSubRegionToMipBounds(
    { width: 64, height: 64, depthOrArrayLayers: 1, mipLevelCount: 4, dimension: '2d', format: 'rgba8unorm' },
    1,
    { x: 32, y: 0, z: 0 },
    { width: 8, height: 8, depthOrArrayLayers: 1 }
  );
  assert.equal(cOob.valid, false);
  assert.equal(cOob.clamped, true);
  assert.equal(cOob.outOfBounds, true);
  assert.equal(cOob.invalidMipLevel, false);

  // Invalid mipLevel >= mipLevelCount -> invalidMipLevel=true.
  const cBadMip = tp.clampSubRegionToMipBounds(
    { width: 64, height: 64, depthOrArrayLayers: 1, mipLevelCount: 3, dimension: '2d', format: 'rgba8unorm' },
    5,
    { x: 0, y: 0, z: 0 },
    { width: 4, height: 4, depthOrArrayLayers: 1 }
  );
  assert.equal(cBadMip.valid, false);
  assert.equal(cBadMip.invalidMipLevel, true);

  // 2) End-to-end strict WebGPU bounds validation across WriteTexture (direct + staged),
  //    copyBufferToTexture, and writeA8Texture.
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const assertInMipBounds = (destination, size, label) => {
    const desc = destination.texture.desc;
    const mip = destination.mipLevel || 0;
    assert.ok(mip >= 0 && mip < desc.mipLevelCount, `${label}: mipLevel ${mip} out of range`);
    const info = tp.getTextureFormatBlockInfo(desc.format);
    const bw = info && info.isCompressed ? info.blockWidth : 1;
    const bh = info && info.isCompressed ? info.blockHeight : 1;
    const mipW = Math.ceil(tp.calcMipDimension(desc.size.width, mip) / bw) * bw;
    const mipH = desc.dimension === '1d' ? 1 : Math.ceil(tp.calcMipDimension(desc.size.height, mip) / bh) * bh;
    const mipD =
      desc.dimension === '1d'
        ? 1
        : desc.dimension === '3d'
          ? tp.calcMipDimension(desc.size.depthOrArrayLayers, mip)
          : desc.size.depthOrArrayLayers;
    const ox = (destination.origin && destination.origin.x) || 0;
    const oy = (destination.origin && destination.origin.y) || 0;
    const oz = (destination.origin && destination.origin.z) || 0;
    assert.ok(ox >= 0 && size.width >= 1 && ox + size.width <= mipW, `${label}: X ${ox}+${size.width} > ${mipW}`);
    assert.ok(oy >= 0 && size.height >= 1 && oy + size.height <= mipH, `${label}: Y ${oy}+${size.height} > ${mipH}`);
    assert.ok(oz >= 0 && size.depthOrArrayLayers >= 1 && oz + size.depthOrArrayLayers <= mipD, `${label}: Z ${oz}+${size.depthOrArrayLayers} > ${mipD}`);
  };
  const origWt = gpu.device.queue.writeTexture;
  gpu.device.queue.writeTexture = (dest, data, layout, size) => {
    assertInMipBounds(dest, size, 'writeTexture');
    return origWt(dest, data, layout, size);
  };
  const origEnc = gpu.device.createCommandEncoder;
  gpu.device.createCommandEncoder = () => {
    const enc = origEnc();
    const origC2t = enc.copyBufferToTexture;
    enc.copyBufferToTexture = (src, dest, size) => {
      assertInMipBounds(dest, size, 'copyBufferToTexture');
      return origC2t(src, dest, size);
    };
    return enc;
  };

  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 128 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  // Create 64x64 4-mip rgba8unorm UI atlas (mip 0 = 64x64, mip 1 = 32x32, mip 2 = 16x16, mip 3 = 8x8).
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 920, width: 64, height: 64, mips: 4, usage: 0x06, format: 22 })
  );
  const atlasTex = processor.textures.get(920);

  // (a) Staged WriteTexture packet: partial UI dirty rect at (48, 48) size 32x32 (overflows 64x64 -> clamps to 16x16).
  const dirtyPixels = new Uint8Array(32 * 32 * 4);
  for (let y = 0; y < 32; y += 1) {
    dirtyPixels[y * 128] = 0x10 + y;
  }
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 920,
      mipLevel: 0,
      originX: 48,
      originY: 48,
      width: 32,
      height: 32,
      bytesPerRow: 128,
      rowsPerImage: 32,
      data: dirtyPixels,
    })
  );

  // (b) Direct 256B-aligned WriteTexture packet at mip 1 (32x32): origin (20, 20), size 64x20 -> clamps to 12x12.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 920,
      mipLevel: 1,
      originX: 20,
      originY: 20,
      width: 64,
      height: 20,
      bytesPerRow: 256,
      rowsPerImage: 20,
      data: new Uint8Array(256 * 20),
    })
  );

  // (c) Completely out-of-bounds dirty rect at mip 2 (16x16): origin (24, 0) -> cleanly skipped, no GPU call, no error.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 920,
      mipLevel: 2,
      originX: 24,
      originY: 0,
      width: 8,
      height: 8,
      bytesPerRow: 32,
      rowsPerImage: 8,
      data: new Uint8Array(256),
    })
  );

  // (d) Direct writeA8Texture partial glyph dirty rect at (56, 56) size 16x16 -> clamps to 8x8 preserving row stride 16.
  const glyphSrc = new Uint8Array(16 * 16);
  glyphSrc[0] = 0xa1;
  glyphSrc[7] = 0xa7;
  glyphSrc[16] = 0xb1; // row 1 col 0 at stride 16
  const a8Res = handler.writeA8Texture(
    { texture: atlasTex, mipLevel: 0, origin: { x: 56, y: 56, z: 0 } },
    glyphSrc,
    { width: 16, height: 16, depthOrArrayLayers: 1 },
    { preferDirect: true }
  );
  assert.equal(a8Res.uploaded, true);
  assert.equal(a8Res.clamped, true);
  assert.equal(a8Res.width, 8);
  assert.equal(a8Res.height, 8);

  // (e) GPU-buffer copyBufferToTexture partial dirty rect at mip 2 (16x16): origin (10, 12) size 12x8 -> clamps to 6x4.
  const encoder = gpu.device.createCommandEncoder();
  const srcBuf = gpu.device.createBuffer({ size: 256 * 8, usage: 0x0004 });
  const c2tRes = handler.copyBufferToTexture(
    encoder,
    { buffer: srcBuf, offset: 0, bytesPerRow: 256, rowsPerImage: 8 },
    { texture: atlasTex, mipLevel: 2, origin: { x: 10, y: 12, z: 0 } },
    { width: 12, height: 8, depthOrArrayLayers: 1 }
  );
  assert.equal(c2tRes.clamped, true);

  // (f) Invalid mipLevel = 9 (>= mipLevelCount 4) -> counted as error, zero GPU calls.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 920,
      mipLevel: 9,
      width: 4,
      height: 4,
      bytesPerRow: 16,
      rowsPerImage: 4,
      data: new Uint8Array(64),
    })
  );

  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(c2tCalls.length, 2);
  assert.deepEqual(c2tCalls[0].copySize, { width: 16, height: 16, depthOrArrayLayers: 1 });
  assert.deepEqual(c2tCalls[0].destination.origin, { x: 48, y: 48, z: 0 });
  assert.equal(c2tCalls[0].bytes[0], 0x10);
  assert.equal(c2tCalls[0].bytes[256], 0x11);
  assert.deepEqual(c2tCalls[1].copySize, { width: 6, height: 4, depthOrArrayLayers: 1 });
  assert.deepEqual(c2tCalls[1].destination.origin, { x: 10, y: 12, z: 0 });

  assert.equal(wtCalls.length, 2);
  assert.deepEqual(wtCalls[0].size, { width: 12, height: 12, depthOrArrayLayers: 1 });
  assert.deepEqual(wtCalls[0].destination.origin, { x: 20, y: 20, z: 0 });
  assert.deepEqual(wtCalls[1].size, { width: 8, height: 8, depthOrArrayLayers: 1 });
  assert.deepEqual(wtCalls[1].destination.origin, { x: 56, y: 56, z: 0 });
  // Verify A8 row 0 col 0, row 0 col 7, and row 1 col 0 (using preserved srcBytesPerRow=16).
  assert.deepEqual(Array.from(wtCalls[1].bytes.subarray(0, 4)), [0, 0, 0, 0xa1]);
  assert.deepEqual(Array.from(wtCalls[1].bytes.subarray(7 * 4, 8 * 4)), [0, 0, 0, 0xa7]);
  assert.deepEqual(Array.from(wtCalls[1].bytes.subarray(8 * 4, 9 * 4)), [0, 0, 0, 0xb1]);

  const stats = handler.getStats();
  assert.equal(stats.uploadsCount, 4);
  assert.equal(stats.reusedDescriptorCount, 4);
  assert.equal(stats.clampedSubRegionCount, 5); // (a) + (b) + (c OOB) + (d) + (e)
  assert.equal(stats.errors, 1); // (f) invalid mipLevel
});

test('W303 (#1219): BC1-BC7 tail-mip parity preserved alongside block-aligned BC sub-region clamping and resetStats()', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  // 16x16 5-mip BC7 texture (mips: 16x16, 8x8, 4x4, 2x2->phys 4x4, 1x1->phys 4x4).
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 940, width: 16, height: 16, mips: 5, usage: 0x06, format: 62 })
  );
  const bc7Tex = processor.textures.get(940);

  // 1) Normal W265 2x2 (mip 3) and 1x1 (mip 4) BC7 tail mips must upload as 4x4 with clampedSubRegionCount === 0.
  const singleBlock = new Uint8Array(16).fill(0x5c);
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 940,
      mipLevel: 3,
      width: 2,
      height: 2,
      bytesPerRow: 16,
      rowsPerImage: 1,
      data: singleBlock,
    })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 940,
      mipLevel: 4,
      width: 1,
      height: 1,
      bytesPerRow: 0,
      rowsPerImage: 0,
      data: singleBlock,
    })
  );
  assert.equal(handler.getStats().clampedSubRegionCount, 0, 'W265 BC tail mips must not count as OOB clamps');
  assert.equal(handler.getStats().uploadsCount, 2);
  assert.equal(handler.getStats().reusedDescriptorCount, 2);
  assert.equal(handler.getStats().uploadBytesTotal, 512);
  assert.equal(handler.getStats().scratchBufferAllocations, 1);
  assert.equal(handler.getStats().scratchBufferReuses, 1);

  // 2) Oversized BC7 dirty rect on mip 0 with non-block-aligned origin (10, 14) -> snaps to (8, 12) and clamps to 8x4!
  const bcBlocks = new Uint8Array(3 * 2 * 16).fill(0x7e); // 3x2 blocks (12x8 texels)
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 940,
      mipLevel: 0,
      originX: 10,
      originY: 14,
      width: 12,
      height: 8,
      bytesPerRow: 48,
      rowsPerImage: 2,
      data: bcBlocks,
    })
  );
  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(c2tCalls.length, 3);
  assert.deepEqual(c2tCalls[0].copySize, { width: 4, height: 4, depthOrArrayLayers: 1 });
  assert.deepEqual(c2tCalls[1].copySize, { width: 4, height: 4, depthOrArrayLayers: 1 });
  assert.deepEqual(c2tCalls[2].destination.origin, { x: 8, y: 12, z: 0 });
  assert.deepEqual(c2tCalls[2].copySize, { width: 8, height: 4, depthOrArrayLayers: 1 });
  assert.equal(handler.getStats().clampedSubRegionCount, 1);
  assert.equal(handler.getStats().uploadsCount, 3);
  assert.equal(handler.getStats().reusedDescriptorCount, 3);

  // 3) resetStats() zeroes all telemetry counters cleanly while preserving live texture metadata & scratch pool.
  const scratchRefBefore = handler._scratchBuffer;
  assert.ok(scratchRefBefore instanceof Uint8Array);

  handler.resetStats();
  const afterResetStats = handler.getStats();
  assert.equal(afterResetStats.uploadsCount, 0);
  assert.equal(afterResetStats.uploadCount, 0);
  assert.equal(afterResetStats.uploadBytesTotal, 0);
  assert.equal(afterResetStats.uploadBytes, 0);
  assert.equal(afterResetStats.bytesUploaded, 0);
  assert.equal(afterResetStats.reusedDescriptorCount, 0);
  assert.equal(afterResetStats.clampedSubRegionCount, 0);
  assert.equal(afterResetStats.created, 0);
  assert.equal(afterResetStats.errors, 0);
  assert.equal(afterResetStats.scratchBufferAllocations, 0);
  assert.equal(afterResetStats.scratchBufferReuses, 0);
  assert.equal(handler._scratchBuffer, scratchRefBefore, 'resetStats() must preserve persistent scratch buffer');
  for (const [key, val] of Object.entries(afterResetStats)) {
    assert.equal(typeof val, 'number', `stat ${key} must be numeric`);
    assert.ok(Number.isInteger(val) && val >= 0, `stat ${key} must be a non-negative integer`);
  }

  // Subsequent upload to already-created texture 940 STILL knows its BC7 mip bounds and reuses scratch!
  const postResetRes = handler.writeTexture(
    { texture: bc7Tex, mipLevel: 2, origin: { x: 0, y: 0, z: 0 } },
    new Uint8Array(64),
    { offset: 0, bytesPerRow: 32, rowsPerImage: 2 },
    { width: 8, height: 8, depthOrArrayLayers: 1 }, // mip 2 is 4x4 -> clamps to 4x4!
    940
  );
  assert.equal(postResetRes.uploaded, true);
  assert.equal(postResetRes.clamped, true);
  assert.equal(handler.getStats().uploadsCount, 1);
  assert.equal(handler.getStats().uploadBytesTotal, 64);
  assert.equal(handler.getStats().reusedDescriptorCount, 1);
  assert.equal(handler.getStats().clampedSubRegionCount, 1);

  // 4) reset() and destroy() clear reusable descriptor object references so GPU resources can be GC'd.
  assert.equal(handler._reusableCopyTexture.texture, bc7Tex);
  handler.reset();
  assert.equal(handler._reusableCopyTexture.texture, null);
  assert.equal(handler._reusableBufferSource.buffer, null);
  assert.equal(handler._scratchBuffer, null);
  assert.equal(handler.getStats().uploadsCount, 0);
  assert.equal(handler.getStats().reusedDescriptorCount, 0);
  assert.equal(handler.getStats().clampedSubRegionCount, 0);
});

test('W341 (#1460): 4096x1 rgba8unorm 1D LUT promotes to WebGPU 2D (4096x1x1) with valid 2D view, sRGB viewFormats, and metadata', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const rsHandler = installRenderStatePackets(processor);
  const handler = tp.installTexturePackets(processor);

  assert.equal(handler.promote1DTo2D, true);

  // ABI v2 CreateTexture2: 4096x1 rgba8unorm (format 22) 1D LUT (dimension 0).
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 1460,
      width: 4096,
      height: 1,
      layers: 1,
      mips: 1,
      samples: 1,
      usage: 0x06, // COPY_DST | TEXTURE_BINDING
      dimension: 0, // 1d
      format: 22, // rgba8unorm
    })
  );
  // ABI v1 CreateTexture: 4096x1 rgba8unorm (v1 format 0) 1D LUT (dimension 0).
  processor.executeCommandBuffer(
    createTexture1Packet({
      id: 1461,
      width: 4096,
      height: 1,
      layers: 1,
      mips: 1,
      usage: 0x06,
      dimension: 0,
      format: 0,
    })
  );

  for (const id of [1460, 1461]) {
    const tex = processor.textures.get(id);
    assert.ok(tex, `texture ${id} must be created`);
    assert.equal(tex.desc.dimension, '2d');
    assert.deepEqual(tex.desc.size, { width: 4096, height: 1, depthOrArrayLayers: 1 });
    assert.equal(tex.desc.mipLevelCount, 1);
    assert.equal(tex.desc.format, 'rgba8unorm');
    assert.deepEqual(tex.desc.viewFormats, ['rgba8unorm-srgb']);

    // Default textureView created by handler must be a valid '2d' GPUTextureView.
    const defaultView = processor.textureViews.get(id);
    assert.ok(defaultView);
    assert.equal(defaultView.dimension, '2d');

    // Explicit 2D views (linear and sRGB sibling) must pass strict WebGPU validation.
    const view2d = tex.createView({ dimension: '2d', format: 'rgba8unorm' });
    assert.equal(view2d.dimension, '2d');
    const view2dSrgb = tex.createView({ dimension: '2d', format: 'rgba8unorm-srgb' });
    assert.equal(view2dSrgb.dimension, '2d');

    // Chained RenderStatePackets._textureView requesting a 2D view must succeed cleanly.
    const rsView = rsHandler._textureView(id, {
      dimension: '2d',
      format: 'rgba8unorm',
      baseMipLevel: 0,
      mipLevelCount: 1,
      baseArrayLayer: 0,
      arrayLayerCount: 1,
      aspect: 'all',
    });
    assert.ok(rsView);
    assert.equal(rsView.dimension, '2d');

    // Metadata preserves requestedDimension: '1d' while recording actual WebGPU dimension: '2d'.
    const meta = handler._getTextureMeta(tex, id);
    assert.equal(meta.dimension, '2d');
    assert.equal(meta.requestedDimension, '1d');
    assert.equal(meta.promoted1DTo2D, true);
    assert.equal(meta.width, 4096);
    assert.equal(meta.height, 1);
    assert.equal(meta.depthOrArrayLayers, 1);

    const rsMeta = rsHandler.textureMeta.get(id);
    assert.ok(rsMeta);
    assert.equal(rsMeta.dimension, '2d');
    assert.equal(rsMeta.requestedDimension, '1d');
    assert.equal(rsMeta.promoted1DTo2D, true);
  }

  assert.equal(processor.textures.get(1460).desc.sampleCount, 1);
  assert.equal(handler.getStats().promoted1DTo2DCount, 2);
  assert.equal(handler.getStats().srgbSiblingViewFormatsAdded, 2);
  assert.equal(handler.getStats().errors, 0);
});

test('W341 (#1460): 4096x1 r32float 1D LUT promotes to WebGPU 2D (4096x1x1) with valid 2D view compatibility', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  // 4096x1 r32float (format 16) 1D LUT with oversized height/layers/mips/samples safely clamped.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 1462,
      width: 4096,
      height: 4,
      layers: 2,
      mips: 8,
      samples: 4,
      usage: 0x06,
      dimension: 0, // 1d
      format: 16, // r32float
    })
  );

  const tex = processor.textures.get(1462);
  assert.ok(tex);
  assert.equal(tex.desc.dimension, '2d');
  assert.deepEqual(tex.desc.size, { width: 4096, height: 1, depthOrArrayLayers: 1 });
  assert.equal(tex.desc.mipLevelCount, 1);
  assert.equal(tex.desc.sampleCount, 1);
  assert.equal(tex.desc.format, 'r32float');
  assert.equal(tex.desc.viewFormats, undefined);

  const defaultView = processor.textureViews.get(1462);
  assert.ok(defaultView);
  assert.equal(defaultView.dimension, '2d');
  assert.doesNotThrow(() => tex.createView({ dimension: '2d', format: 'r32float' }));

  const meta = handler._getTextureMeta(tex, 1462);
  assert.equal(meta.dimension, '2d');
  assert.equal(meta.requestedDimension, '1d');
  assert.equal(meta.promoted1DTo2D, true);
  assert.equal(handler.getStats().promoted1DTo2DCount, 1);
  assert.equal(handler.getStats().errors, 0);
});

test('W341 (#1460): direct writeTexture uploads into promoted 4096x1 rgba8unorm and r32float 2D LUTs (plain, SAB, and height=0)', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1463, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 22 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1464, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 16 })
  );

  const rgbaLut = new Uint8Array(4096 * 4);
  rgbaLut[0] = 0x11;
  rgbaLut[1] = 0x22;
  rgbaLut[4096 * 4 - 1] = 0xee;

  const r32Lut = new Uint8Array(4096 * 4);
  new Float32Array(r32Lut.buffer)[0] = 1.5;
  new Float32Array(r32Lut.buffer)[4095] = -3.25;

  // 1) Plain ArrayBuffer 4096x1 rgba8unorm upload (16384B/row, 256B-aligned zero-copy fast path).
  const pktRgba = writeTexturePacket({
    textureId: 1463,
    width: 4096,
    height: 1,
    layers: 1,
    bytesPerRow: 4096 * 4,
    rowsPerImage: 1,
    data: rgbaLut,
    useSab: false,
  });
  processor.executeCommandBuffer(pktRgba);

  // 2) SAB-backed 4096x1 r32float upload with height=0 (1D packet omitting height -> normalized to 1).
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 1464,
      width: 4096,
      height: 0,
      layers: 1,
      bytesPerRow: 4096 * 4,
      rowsPerImage: 0,
      data: r32Lut,
      useSab: true,
    })
  );

  // 3) Direct handler.writeTexture call omitting height in GPUExtent3D.
  const directRes = handler.writeTexture(
    { texture: processor.textures.get(1463), mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    rgbaLut,
    { offset: 0, bytesPerRow: 4096 * 4, rowsPerImage: 1 },
    { width: 4096 },
    1463
  );
  assert.equal(directRes.uploaded, true);
  assert.equal(directRes.clamped, false);

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 3);

  assert.deepEqual(wtCalls[0].size, { width: 4096, height: 1, depthOrArrayLayers: 1 });
  assert.deepEqual(wtCalls[0].destination.origin, { x: 0, y: 0, z: 0 });
  assert.equal(wtCalls[0].dataLayout.bytesPerRow, 16384);
  assert.equal(wtCalls[0].bufferRef, pktRgba.buffer);
  assert.equal(wtCalls[0].bytes[0], 0x11);
  assert.equal(wtCalls[0].bytes[16383], 0xee);

  assert.deepEqual(wtCalls[1].size, { width: 4096, height: 1, depthOrArrayLayers: 1 });
  assert.equal(wtCalls[1].isSab, false);
  const uploadedFloats = new Float32Array(
    wtCalls[1].bytes.buffer,
    wtCalls[1].bytes.byteOffset,
    4096
  );
  assert.equal(uploadedFloats[0], 1.5);
  assert.equal(uploadedFloats[4095], -3.25);

  assert.deepEqual(wtCalls[2].size, { width: 4096, height: 1, depthOrArrayLayers: 1 });
  assert.equal(handler.getStats().promoted1DTo2DCount, 2);
  assert.equal(handler.getStats().sabCopyCount, 1);
  assert.equal(handler.getStats().errors, 0);
});

test('W341 (#1460): staged WriteTexture & copyBufferToTexture uploads and sub-rect originX/originY clamping on promoted 1D->2D LUTs', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 128 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1465, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 22 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1466, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 16 })
  );
  const rgbaTex = processor.textures.get(1465);
  const r32Tex = processor.textures.get(1466);

  // 1) Staged partial sub-rect WriteTexture with unaligned 128B row pitch at originX=128, originY=0.
  const subBytes = new Uint8Array(32 * 4);
  for (let i = 0; i < subBytes.length; i += 1) subBytes[i] = (i + 5) & 0xff;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 1465,
      originX: 128,
      originY: 0,
      width: 32,
      height: 1,
      bytesPerRow: 128,
      rowsPerImage: 1,
      data: subBytes,
    })
  );

  // 2) Right-edge partial sub-rect clamp: originX=4080, width=32 (4080+32=4112 > 4096 -> clamps to width=16).
  const edgeBytes = new Uint8Array(32 * 4);
  for (let i = 0; i < 32; i += 1) {
    edgeBytes[i * 4] = i + 1;
    edgeBytes[i * 4 + 3] = 0xff;
  }
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 1465,
      originX: 4080,
      originY: 0,
      width: 32,
      height: 1,
      bytesPerRow: 128,
      rowsPerImage: 1,
      data: edgeBytes,
    })
  );

  // 3) Out-of-bounds originY=1 on promoted 4096x1 texture -> safely skipped, zero GPU calls, zero errors.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 1465,
      originX: 0,
      originY: 1,
      width: 16,
      height: 1,
      bytesPerRow: 64,
      rowsPerImage: 1,
      data: new Uint8Array(64),
    })
  );

  // 4) copyBufferToTexture CPU-backed SAB and GPU-buffer-only unaligned 64B rows into promoted r32float LUT.
  const encoder = gpu.device.createCommandEncoder();
  const sabSub = new Uint8Array(new SharedArrayBuffer(16 * 4));
  new Float32Array(sabSub.buffer)[0] = 42.0;
  new Float32Array(sabSub.buffer)[15] = 99.5;
  const c2tCpu = handler.copyBufferToTexture(
    encoder,
    { bytesPerRow: 64, rowsPerImage: 1 },
    { texture: r32Tex, mipLevel: 0, origin: { x: 64, y: 0, z: 0 } },
    { width: 16, height: 1, depthOrArrayLayers: 1 },
    sabSub
  );
  assert.equal(c2tCpu.repacked, true);
  assert.equal(c2tCpu.copiedFromSab, true);
  assert.equal(c2tCpu.bytesPerRow, 256);
  assert.equal(c2tCpu.clamped, false);

  const gpuBuf = gpu.device.createBuffer({ size: 64, usage: 0x0004 | 0x0008 });
  for (let i = 0; i < 64; i += 1) gpuBuf.backing[i] = 0x80 + i;
  const c2tGpu = handler.copyBufferToTexture(
    encoder,
    { buffer: gpuBuf, offset: 0, bytesPerRow: 64, rowsPerImage: 1 },
    { texture: r32Tex, mipLevel: 0, origin: { x: 256, y: 0, z: 0 } },
    { width: 16, height: 1, depthOrArrayLayers: 1 }
  );
  assert.equal(c2tGpu.repacked, true);
  assert.equal(c2tGpu.bytesPerRow, 256);
  assert.equal(c2tGpu.clamped, false);

  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(c2tCalls.length, 4);

  // Verify call 0: exact sub-rect at (128, 0), 32x1x1, 256B aligned with zeroed padding.
  assert.equal(c2tCalls[0].destination.texture, rgbaTex);
  assert.deepEqual(c2tCalls[0].destination.origin, { x: 128, y: 0, z: 0 });
  assert.deepEqual(c2tCalls[0].copySize, { width: 32, height: 1, depthOrArrayLayers: 1 });
  assert.equal(c2tCalls[0].source.bytesPerRow, 256);
  assert.deepEqual(Array.from(c2tCalls[0].bytes.subarray(0, 128)), Array.from(subBytes));
  for (let p = 128; p < 256; p += 1) {
    assert.equal(c2tCalls[0].bytes[p], 0, `padding byte ${p} must be zero`);
  }

  // Verify call 1: right-edge clamped to (4080, 0), 16x1x1, preserving unclipped left-16 texels.
  assert.deepEqual(c2tCalls[1].destination.origin, { x: 4080, y: 0, z: 0 });
  assert.deepEqual(c2tCalls[1].copySize, { width: 16, height: 1, depthOrArrayLayers: 1 });
  assert.equal(c2tCalls[1].bytes[0], 1);
  assert.equal(c2tCalls[1].bytes[15 * 4], 16);

  // Verify calls 2 & 3 on r32float LUT.
  assert.equal(c2tCalls[2].destination.texture, r32Tex);
  assert.deepEqual(c2tCalls[2].destination.origin, { x: 64, y: 0, z: 0 });
  assert.deepEqual(c2tCalls[2].copySize, { width: 16, height: 1, depthOrArrayLayers: 1 });
  const c2tFloats = new Float32Array(c2tCalls[2].bytes.buffer, c2tCalls[2].bytes.byteOffset, 16);
  assert.equal(c2tFloats[0], 42.0);
  assert.equal(c2tFloats[15], 99.5);

  assert.deepEqual(c2tCalls[3].destination.origin, { x: 256, y: 0, z: 0 });
  assert.deepEqual(c2tCalls[3].copySize, { width: 16, height: 1, depthOrArrayLayers: 1 });
  assert.deepEqual(Array.from(c2tCalls[3].bytes.subarray(0, 64)), Array.from(gpuBuf.backing));

  const stats = handler.getStats();
  assert.equal(stats.promoted1DTo2DCount, 2);
  assert.equal(stats.clampedSubRegionCount, 2); // right-edge clamp + OOB originY=1
  assert.equal(stats.rowPitchRepackCount, 4);
  assert.equal(stats.errors, 0);
});

test('W341 (#1460): explicit promote1DTo2D: false opt-out preserves native 1D dimension and demonstrates 2D-view mismatch', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor, { promote1DTo2D: false });

  assert.equal(handler.promote1DTo2D, false);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1470, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 22 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1471, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 16 })
  );

  for (const id of [1470, 1471]) {
    const tex = processor.textures.get(id);
    assert.ok(tex);
    assert.equal(tex.desc.dimension, '1d');
    assert.deepEqual(tex.desc.size, { width: 4096, height: 1, depthOrArrayLayers: 1 });
    assert.equal(processor.textureViews.get(id).dimension, '1d');
    const meta = handler._getTextureMeta(tex, id);
    assert.equal(meta.dimension, '1d');
    assert.equal(meta.requestedDimension, '1d');
    assert.equal(meta.promoted1DTo2D, false);

    // Creating a 2D view against an unpromoted 1D texture fails strict WebGPU validation
    // (reproducing the t81/t82 Chrome WebGPU validation error).
    assert.throws(
      () => tex.createView({ dimension: '2d' }),
      /not compatible with the dimension \(1d\)/
    );
    assert.doesNotThrow(() => tex.createView({ dimension: '1d' }));
  }

  assert.equal(handler.getStats().promoted1DTo2DCount, 0);
  assert.equal(handler.getStats().created, 2);
  assert.equal(handler.getStats().errors, 0);

  // Alias option promote1DTexturesTo2D: false also opts out cleanly.
  const gpu2 = createMockWebGpu({ strictWebGpu: true });
  const processor2 = new WgpuCommandProcessor(gpu2.device, gpu2.ctx);
  const handler2 = tp.installTexturePackets(processor2, { promote1DTexturesTo2D: false });
  assert.equal(handler2.promote1DTo2D, false);
  processor2.executeCommandBuffer(
    createTexture2Packet({ id: 1472, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 16 })
  );
  assert.equal(processor2.textures.get(1472).desc.dimension, '1d');
  assert.equal(handler2.getStats().promoted1DTo2DCount, 0);
});

test('W341 (#1460): promoted1DTo2DCount ignores failed creations, updates on recreate/destroy, and resets cleanly', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  // Invalid format code 0 with dimension 0 must NOT increment promoted1DTo2DCount.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1480, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 0 })
  );
  assert.equal(handler.getStats().errors, 1);
  assert.equal(handler.getStats().promoted1DTo2DCount, 0);

  // Valid 1D texture increments promoted1DTo2DCount to 1.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1481, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 22 })
  );
  assert.equal(handler.getStats().promoted1DTo2DCount, 1);
  assert.equal(handler._getTextureMeta(processor.textures.get(1481), 1481).requestedDimension, '1d');

  // Re-creating id 1481 as a native 2D texture refreshes metadata cleanly.
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1481, width: 64, height: 64, usage: 0x06, dimension: 1, format: 22 })
  );
  assert.equal(handler.getStats().promoted1DTo2DCount, 1);
  const recreatedMeta = handler._getTextureMeta(processor.textures.get(1481), 1481);
  assert.equal(recreatedMeta.dimension, '2d');
  assert.equal(recreatedMeta.requestedDimension, '2d');
  assert.equal(recreatedMeta.promoted1DTo2D, false);

  // DestroyResource clears tracked metadata for that texture ID.
  processor.executeCommandBuffer(destroyPacket(1481));
  assert.equal(handler._textureMeta.has(1481), false);

  // resetStats(), reset(), and destroy() zero promoted1DTo2DCount cleanly.
  handler.resetStats();
  assert.equal(handler.getStats().promoted1DTo2DCount, 0);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1482, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 16 })
  );
  assert.equal(handler.getStats().promoted1DTo2DCount, 1);
  handler.reset();
  assert.equal(handler.getStats().promoted1DTo2DCount, 0);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 1483, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 16 })
  );
  assert.equal(handler.getStats().promoted1DTo2DCount, 1);
  handler.destroy();
  assert.equal(handler.getStats().promoted1DTo2DCount, 0);
});

test('W344: 4K Bink2 SAB Y+Cr+Cb multi-frame uploads reuse a single persistent scratch buffer with zero per-frame allocations', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({
    device: gpu.device,
    chunkSize: 4 * 1024 * 1024,
    maxPoolBytes: 64 * 1024 * 1024,
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  const YW = 3840;
  const YH = 2160;
  const CW = 1920;
  const CH = 1080;
  const ALIGNED_CW = 2048;

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 3440, width: YW, height: YH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 3441, width: CW, height: CH, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 3442, width: CW, height: CH, usage: 0x06, format: 1 })
  );

  const yData = new Uint8Array(YW * YH);
  const crData = new Uint8Array(CW * CH);
  const cbData = new Uint8Array(CW * CH);

  const FRAMES = 4;
  let stableScratchRef = null;
  for (let frame = 0; frame < FRAMES; frame += 1) {
    stagingPool.beginFrame(frame);
    yData[0] = 0x10 + frame;
    yData[YW * YH - 1] = 0x20 + frame;
    crData[0] = 0x30 + frame;
    crData[CW - 1] = 0x40 + frame;
    cbData[0] = 0x50 + frame;
    cbData[(CH - 1) * CW + (CW - 1)] = 0x60 + frame;

    // All three planes arrive from SharedArrayBuffer (real Wasm64 command ring).
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 3440,
        width: YW,
        height: YH,
        bytesPerRow: YW,
        rowsPerImage: YH,
        data: yData,
        useSab: true,
      })
    );
    if (frame === 0) {
      stableScratchRef = handler._scratchBuffer;
      assert.ok(stableScratchRef instanceof Uint8Array);
      assert.equal(stableScratchRef.byteLength, YW * YH);
      // Poison chroma padding zone inside the shared scratch buffer to prove strict zero-padding on reuse.
      stableScratchRef.fill(0xee, CW, ALIGNED_CW);
    }
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 3441,
        width: CW,
        height: CH,
        bytesPerRow: CW,
        rowsPerImage: CH,
        data: crData,
        useSab: true,
      })
    );
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 3442,
        width: CW,
        height: CH,
        bytesPerRow: CW,
        rowsPerImage: CH,
        data: cbData,
        useSab: true,
      })
    );
    assert.equal(
      handler._scratchBuffer,
      stableScratchRef,
      `Frame ${frame}: single 8.3 MiB scratch buffer must serve Y, Cr, and Cb with zero reallocations`
    );
    stagingPool.endFrame();
  }

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(wtCalls.length, FRAMES);
  assert.equal(c2tCalls.length, FRAMES * 2);
  assert.ok(wtCalls.every((c) => c.isSab === false), 'no SAB views may reach queue.writeTexture');

  // Verify frame 0 Y plane bytes and Cr plane zero-padded tail [1920..2048).
  assert.equal(wtCalls[0].bytes[0], 0x10);
  assert.equal(wtCalls[0].bytes[YW * YH - 1], 0x20);
  assert.equal(c2tCalls[0].bytes[0], 0x30);
  assert.equal(c2tCalls[0].bytes[CW - 1], 0x40);
  for (let p = CW; p < ALIGNED_CW; p += 1) {
    assert.equal(c2tCalls[0].bytes[p], 0, `Cr padding byte ${p} must be zeroed on reused scratch`);
  }

  const stats = handler.getStats();
  assert.equal(stats.scratchBufferAllocations, 1, 'only 1 scratch allocation across 12 SAB plane uploads');
  assert.equal(stats.scratchBufferReuses, FRAMES * 3 - 1, 'remaining 11 uploads must reuse scratch');
  assert.equal(stats.sabCopyCount, FRAMES * 3);
  assert.equal(stats.reusedDescriptorCount, FRAMES * 3);
  // Internal reusable holders must not pin buffer references after uploads finish.
  assert.equal(handler._reusablePreparedData.data, null);
  assert.equal(handler._reusableRepackResult.data, null);
});

test('W344: TextureScratchPool geometric growth amortizes ramping upload sizes up to maxBytes cap', () => {
  const pool = new tp.TextureScratchPool({ maxBytes: 4096 });

  // Initial allocation stays tightly sized to requiredBytes.
  const a1 = pool.acquire(512);
  assert.equal(a1.pooled, true);
  assert.equal(a1.reused, false);
  assert.equal(a1.buffer.byteLength, 512);
  assert.equal(a1.view.byteLength, 512);

  // Ramping to 700B grows geometrically (512 * 2 = 1024B) so 900B hits without reallocating.
  const a2 = pool.acquire(700);
  assert.equal(a2.pooled, true);
  assert.equal(a2.reused, false);
  assert.equal(a2.buffer.byteLength, 1024);
  assert.equal(a2.view.byteLength, 700);

  const a3 = pool.acquire(900);
  assert.equal(a3.pooled, true);
  assert.equal(a3.reused, true);
  assert.equal(a3.buffer, a2.buffer);
  assert.equal(a3.view.byteLength, 900);

  // Ramping to 3000B grows to 3000B (max(3000, 2048) clamped to 4096B cap).
  const a4 = pool.acquire(3000);
  assert.equal(a4.pooled, true);
  assert.equal(a4.reused, false);
  assert.equal(a4.buffer.byteLength, 3000);
  assert.equal(a4.view.byteLength, 3000);

  // Ramping to 3500B doubles clamped to maxBytes (4096B).
  const a5 = pool.acquire(3500);
  assert.equal(a5.pooled, true);
  assert.equal(a5.reused, false);
  assert.equal(a5.buffer.byteLength, 4096);
  assert.equal(a5.view.byteLength, 3500);

  // Requesting 4096B reuses the capped buffer.
  const a6 = pool.acquire(4096);
  assert.equal(a6.pooled, true);
  assert.equal(a6.reused, true);
  assert.equal(a6.buffer, a5.buffer);

  // Oversized 8192B falls back to standalone without replacing retained 4096B slot.
  const a7 = pool.acquire(8192);
  assert.equal(a7.pooled, false);
  assert.equal(a7.oversized, true);
  assert.equal(pool.retainedBytes, 4096);
  assert.equal(pool.getStats().allocations, 4);
  assert.equal(pool.getStats().reuses, 2);
  assert.equal(pool.getStats().oversizedFallbacks, 1);
});

test('W344: handleWriteTexture pre-clamped fast path avoids duplicate clamping and preserves small-row & 1D->2D LUT parity', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  processor.executeCommandBuffer(
    createTexture2Packet({ id: 3450, width: 256, height: 4, usage: 0x06, format: 1 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 3451, width: 4096, height: 1, usage: 0x06, dimension: 0, format: 22 })
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 3452, width: 4, height: 4, usage: 0x06, format: 22 })
  );

  let clampCalls = 0;
  const origClamp = handler._clampSubRegion.bind(handler);
  handler._clampSubRegion = function spiedClamp(...args) {
    clampCalls += 1;
    return origClamp(...args);
  };

  // 1) Aligned direct upload (256x4 r8unorm) -> exactly 1 clamp call (no duplicate clamp in writeTexture).
  const alignedData = new Uint8Array(256 * 4);
  alignedData[0] = 0x7a;
  alignedData[1023] = 0x8b;
  const alignedPkt = writeTexturePacket({
    textureId: 3450,
    width: 256,
    height: 4,
    bytesPerRow: 256,
    rowsPerImage: 4,
    data: alignedData,
  });
  processor.executeCommandBuffer(alignedPkt);
  assert.equal(clampCalls, 1, 'aligned direct WriteTexture must clamp sub-region exactly once');

  // 2) Promoted 1D->2D 4096x1 rgba8unorm LUT (bytesPerRow = 16384, aligned) -> 1 clamp call, zero-copy!
  const lutData = new Uint8Array(4096 * 4);
  lutData[0] = 0xaa;
  lutData[lutData.length - 1] = 0xbb;
  const lutPkt = writeTexturePacket({
    textureId: 3451,
    width: 4096,
    height: 0,
    bytesPerRow: 4096 * 4,
    rowsPerImage: 1,
    data: lutData,
  });
  processor.executeCommandBuffer(lutPkt);
  assert.equal(clampCalls, 2);

  // 3) Small-row (<=32B) staged repack (4x4 rgba8unorm, 16B/row) -> verifies inline small-row repack & zeroed tails.
  const smallPixels = new Uint8Array(64);
  for (let i = 0; i < 64; i += 1) smallPixels[i] = i + 5;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 3452,
      width: 4,
      height: 4,
      bytesPerRow: 16,
      rowsPerImage: 4,
      data: smallPixels,
    })
  );
  // Poison scratch and re-upload to verify zero-padding on small-row scratch reuse.
  handler._scratchBuffer.fill(0xcc);
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 3452,
      width: 4,
      height: 4,
      bytesPerRow: 16,
      rowsPerImage: 4,
      data: smallPixels,
    })
  );
  assert.equal(clampCalls, 4);

  const wtCalls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(wtCalls.length, 2);
  assert.equal(wtCalls[0].bufferRef, alignedPkt.buffer);
  assert.equal(wtCalls[1].bufferRef, lutPkt.buffer);
  assert.deepEqual(wtCalls[1].size, { width: 4096, height: 1, depthOrArrayLayers: 1 });

  const c2tCalls = gpu.calls.filter((c) => c.op === 'copyBufferToTexture');
  assert.equal(c2tCalls.length, 2);
  for (let row = 0; row < 4; row += 1) {
    assert.deepEqual(
      Array.from(c2tCalls[1].bytes.subarray(row * 256, row * 256 + 16)),
      Array.from(smallPixels.subarray(row * 16, row * 16 + 16))
    );
    for (let pad = 16; pad < 256; pad += 1) {
      assert.equal(c2tCalls[1].bytes[row * 256 + pad], 0);
    }
  }
});

test('W373: zero-alloc _dispatchDirectWriteTexture result reuse, scratch-backed decodeCreateTexture/decodeCreateTexture2, and writeA8Texture slot0 path', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  // 1) Pure scratch-backed decodeCreateTexture / decodeCreateTexture2 reuse the exact scratch object
  const tex1A = createTexture1Packet({ id: 4001, width: 128, height: 64, format: 0, usage: 0x06 }); // rgba8unorm (has sRGB sibling)
  const tex1B = createTexture1Packet({ id: 4002, width: 32, height: 32, format: 2, usage: 0x06 }); // rgba16float (no sRGB sibling)
  const scratch1 = {};
  const d1A = tp.decodeCreateTexture(new DataView(tex1A.buffer, 8), 0, 32, scratch1);
  assert.equal(d1A, scratch1);
  assert.equal(d1A.id, 4001);
  assert.deepEqual(d1A.descriptor.viewFormats, ['rgba8unorm-srgb']);
  const descRef1 = d1A.descriptor;
  const sizeRef1 = d1A.descriptor.size;
  const d1B = tp.decodeCreateTexture(new DataView(tex1B.buffer, 8), 0, 32, scratch1);
  assert.equal(d1B, scratch1);
  assert.equal(d1B.descriptor, descRef1);
  assert.equal(d1B.descriptor.size, sizeRef1);
  assert.equal(d1B.id, 4002);
  assert.equal('viewFormats' in d1B.descriptor, false, 'stale viewFormats removed on scratch reuse');

  const tex2A = createTexture2Packet({
    id: 4010,
    width: 256,
    height: 128,
    mips: 4,
    format: 62, // bc7-rgba-unorm
    usage: 0x06,
    viewFormats: [63],
  });
  const tex2B = createTexture2Packet({
    id: 4011,
    width: 64,
    height: 64,
    mips: 1,
    samples: 4,
    format: 16, // r32float
    usage: 0x14,
  });
  const scratch2 = {};
  const d2A = tp.decodeCreateTexture2(new DataView(tex2A.buffer, 8), 0, 40, scratch2);
  assert.equal(d2A, scratch2);
  assert.deepEqual(d2A.descriptor.viewFormats, ['bc7-rgba-unorm-srgb']);
  const descRef2 = d2A.descriptor;
  const sizeRef2 = d2A.descriptor.size;
  const d2B = tp.decodeCreateTexture2(new DataView(tex2B.buffer, 8), 0, 40, scratch2);
  assert.equal(d2B, scratch2);
  assert.equal(d2B.descriptor, descRef2);
  assert.equal(d2B.descriptor.size, sizeRef2);
  assert.equal(d2B.descriptor.sampleCount, 4);
  assert.equal('viewFormats' in d2B.descriptor, false);

  // 2) Consecutive CREATE_TEXTURE / CREATE_TEXTURE2 packets must not alias texture.desc in mocks
  processor.executeCommandBuffer(tex1A);
  processor.executeCommandBuffer(tex1B);
  processor.executeCommandBuffer(tex2A);
  processor.executeCommandBuffer(tex2B);
  assert.equal(processor.textures.get(4001).desc.format, 'rgba8unorm');
  assert.deepEqual(processor.textures.get(4001).desc.size, { width: 128, height: 64, depthOrArrayLayers: 1 });
  assert.equal(processor.textures.get(4002).desc.format, 'rgba16float');
  assert.deepEqual(processor.textures.get(4002).desc.size, { width: 32, height: 32, depthOrArrayLayers: 1 });
  assert.equal(processor.textures.get(4010).desc.format, 'bc7-rgba-unorm');
  assert.equal(processor.textures.get(4011).desc.format, 'r32float');

  // 3) _dispatchDirectWriteTexture returns the pre-allocated _reusableDispatchResult across uploads
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 4020, width: 256, height: 4, format: 1, usage: 0x06 }) // r8unorm
  );
  const dispatchReturnRefs = [];
  const origDispatch = handler._dispatchDirectWriteTexture.bind(handler);
  handler._dispatchDirectWriteTexture = function spiedDispatch(...args) {
    const res = origDispatch(...args);
    dispatchReturnRefs.push(res);
    return res;
  };

  const rowData = new Uint8Array(256 * 4);
  rowData.fill(0x5a);
  const pkt = writeTexturePacket({
    textureId: 4020,
    width: 256,
    height: 4,
    bytesPerRow: 256,
    rowsPerImage: 4,
    data: rowData,
  });
  processor.executeCommandBuffer(pkt);
  processor.executeCommandBuffer(pkt);
  assert.equal(dispatchReturnRefs.length, 2);
  assert.equal(dispatchReturnRefs[0], handler._reusableDispatchResult);
  assert.equal(dispatchReturnRefs[1], handler._reusableDispatchResult);

  // Public writeTexture() still returns a fresh caller-safe object distinct from _reusableDispatchResult
  const pubRes1 = handler.writeTexture(
    { texture: processor.textures.get(4020), mipLevel: 0 },
    rowData,
    { bytesPerRow: 256, rowsPerImage: 4 },
    { width: 256, height: 4, depthOrArrayLayers: 1 },
    4020
  );
  const pubRes2 = handler.writeTexture(
    { texture: processor.textures.get(4020), mipLevel: 0 },
    rowData,
    { bytesPerRow: 256, rowsPerImage: 4 },
    { width: 256, height: 4, depthOrArrayLayers: 1 },
    4020
  );
  assert.notEqual(pubRes1, handler._reusableDispatchResult);
  assert.notEqual(pubRes1, pubRes2);
  assert.equal(pubRes1.uploaded, true);
  assert.equal(pubRes1.zeroCopy, true);

  // 4) writeA8Texture uses _acquireScratchSlot0 directly (never calls _acquireScratchBuffer wrapper)
  let wrapperCalls = 0;
  handler._acquireScratchBuffer = () => {
    wrapperCalls += 1;
    throw new Error('writeA8Texture must use _acquireScratchSlot0 directly');
  };
  const a8Res = handler.writeA8Texture(
    { texture: processor.textures.get(4001), mipLevel: 0 },
    new Uint8Array([0x11, 0x22, 0x33, 0x44]),
    { width: 2, height: 2, depthOrArrayLayers: 1 },
    { preferDirect: true, textureId: 4001 }
  );
  assert.equal(wrapperCalls, 0);
  assert.equal(a8Res.uploaded, true);
});

test('W373: 3D scene texture uploads — BC7 full mip chain (incl. 2x2 & 1x1 tail mips), 6-face cubemap / 2D array, 3D volume mip depth halving, and depth/stencil', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 256 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  // 1) BC7 16x16 full 5-mip chain (16x16, 8x8, 4x4, 2x2 -> 4x4, 1x1 -> 4x4)
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 4100, width: 16, height: 16, mips: 5, format: 62, usage: 0x06 })
  );
  const bc7Mips = [
    { mip: 0, w: 16, h: 16, bpr: 64, rpi: 4, expW: 16, expH: 16 },
    { mip: 1, w: 8, h: 8, bpr: 32, rpi: 2, expW: 8, expH: 8 },
    { mip: 2, w: 4, h: 4, bpr: 16, rpi: 1, expW: 4, expH: 4 },
    { mip: 3, w: 2, h: 2, bpr: 16, rpi: 1, expW: 4, expH: 4 }, // tail mip clamped to 4x4
    { mip: 4, w: 1, h: 1, bpr: 16, rpi: 1, expW: 4, expH: 4 }, // tail mip clamped to 4x4
  ];
  for (const m of bc7Mips) {
    const data = new Uint8Array(m.bpr * m.rpi);
    data.fill(0x20 + m.mip);
    processor.executeCommandBuffer(
      writeTexturePacket({
        textureId: 4100,
        mipLevel: m.mip,
        width: m.w,
        height: m.h,
        bytesPerRow: m.bpr,
        rowsPerImage: m.rpi,
        data,
      })
    );
  }
  const bc7Copies = gpu.calls.filter(
    (c) => c.op === 'copyBufferToTexture' && c.destination.texture === processor.textures.get(4100)
  );
  assert.equal(bc7Copies.length, 5);
  bc7Mips.forEach((m, idx) => {
    assert.equal(bc7Copies[idx].destination.mipLevel, m.mip);
    assert.deepEqual(bc7Copies[idx].copySize, { width: m.expW, height: m.expH, depthOrArrayLayers: 1 });
    assert.equal(bc7Copies[idx].source.bytesPerRow % 256, 0);
  });

  // 2) 6-face cubemap / 2D array (64x64x6 rgba8unorm, 3 mips): array layers stay 6 at every mip
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 4101, width: 64, height: 64, layers: 6, mips: 3, format: 22, usage: 0x06 })
  );
  // Mip 0: 64*4 = 256B/row (aligned -> direct zero-copy), 6 faces
  const cubeMip0 = new Uint8Array(256 * 64 * 6);
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 4101,
      mipLevel: 0,
      width: 64,
      height: 64,
      layers: 6,
      bytesPerRow: 256,
      rowsPerImage: 64,
      data: cubeMip0,
    })
  );
  // Mip 2: 16x16x6 (64B/row unaligned -> staged repack), all 6 faces preserved
  const cubeMip2 = new Uint8Array(64 * 16 * 6);
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 4101,
      mipLevel: 2,
      width: 16,
      height: 16,
      layers: 6,
      bytesPerRow: 64,
      rowsPerImage: 16,
      data: cubeMip2,
    })
  );
  const cubeDirect = gpu.calls.filter(
    (c) => c.op === 'writeTexture' && c.destination.texture === processor.textures.get(4101)
  );
  const cubeStaged = gpu.calls.filter(
    (c) => c.op === 'copyBufferToTexture' && c.destination.texture === processor.textures.get(4101)
  );
  assert.equal(cubeDirect.length, 1);
  assert.deepEqual(cubeDirect[0].size, { width: 64, height: 64, depthOrArrayLayers: 6 });
  assert.equal(cubeStaged.length, 1);
  assert.equal(cubeStaged[0].destination.mipLevel, 2);
  assert.deepEqual(cubeStaged[0].copySize, { width: 16, height: 16, depthOrArrayLayers: 6 });

  // 3) 3D volume texture (16x16x8 rgba16float, 4 mips): depth halves per mip (8 -> 4 -> 2 -> 1)
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 4102,
      width: 16,
      height: 16,
      layers: 8,
      mips: 4,
      dimension: 2, // 3d
      format: 40, // rgba16float (8 Bpp)
      usage: 0x06,
    })
  );
  // Mip 2 has logical dimensions 4x4x2; request depth=4 (OOB) -> clamped safely to depth=2
  const volMip2Data = new Uint8Array(32 * 4 * 4);
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 4102,
      mipLevel: 2,
      width: 4,
      height: 4,
      layers: 4,
      bytesPerRow: 32,
      rowsPerImage: 4,
      data: volMip2Data,
    })
  );
  // Mip 3 has logical dimensions 2x2x1
  const volMip3Data = new Uint8Array(16 * 2 * 1);
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 4102,
      mipLevel: 3,
      width: 2,
      height: 2,
      layers: 1,
      bytesPerRow: 16,
      rowsPerImage: 2,
      data: volMip3Data,
    })
  );
  const volCopies = gpu.calls.filter(
    (c) => c.op === 'copyBufferToTexture' && c.destination.texture === processor.textures.get(4102)
  );
  assert.equal(volCopies.length, 2);
  assert.deepEqual(volCopies[0].copySize, { width: 4, height: 4, depthOrArrayLayers: 2 });
  assert.deepEqual(volCopies[1].copySize, { width: 2, height: 2, depthOrArrayLayers: 1 });

  // 4) Depth/stencil formats (depth32float=48, stencil8=44) upload cleanly via direct & staged paths
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 4103, width: 64, height: 4, format: 48, usage: 0x16 }) // depth32float (64*4=256B aligned)
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 4104, width: 32, height: 4, format: 44, usage: 0x16 }) // stencil8 (32B unaligned)
  );
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 4103,
      width: 64,
      height: 4,
      bytesPerRow: 256,
      rowsPerImage: 4,
      data: new Uint8Array(256 * 4),
    })
  );
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 4104,
      width: 32,
      height: 4,
      bytesPerRow: 32,
      rowsPerImage: 4,
      data: new Uint8Array(32 * 4),
    })
  );
  assert.equal(handler.getStats().errors, 0, handler.lastError);
});

test('Console texture fidelity: tight-last-row BC7 single/multi-slice uploads and GPU copyBufferToBuffer clamp', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']),
  });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 128 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  // 1) 16x16 BC7-sRGB (code 63) 6-face cubemap with D3D12 tight final row:
  //    4 block rows/slice, unaligned bytesPerRow = 128 (or aligned 256), tight last row = 64 bytes,
  //    caller supplies rowsPerImage = height = 16 (pixel rows instead of 4 block rows).
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 5001,
      width: 16,
      height: 16,
      layers: 6,
      mips: 1,
      format: 63, // bc7-rgba-unorm-srgb
      usage: 0x06,
    })
  );
  // 5 full slices * 4 block rows * 128B + 3 full block rows * 128B + 64B tight last row = 3008 bytes (< 3072 full).
  const tightCubeBytes = new Uint8Array(5 * 4 * 128 + 3 * 128 + 64);
  tightCubeBytes.fill(0x5a);
  tightCubeBytes[tightCubeBytes.length - 1] = 0xc3;
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 5001,
      width: 16,
      height: 16,
      layers: 6,
      bytesPerRow: 128,
      rowsPerImage: 16, // pixel rows -> must auto-correct to 4 block rows even on tight last row!
      data: tightCubeBytes,
    })
  );
  const cubeCopy = gpu.calls.find(
    (c) => c.op === 'copyBufferToTexture' && c.destination.texture === processor.textures.get(5001)
  );
  assert.ok(cubeCopy, 'tight-last-row multi-slice BC7 cubemap must stage via copyBufferToTexture');
  assert.equal(cubeCopy.source.bytesPerRow, 256);
  assert.equal(
    cubeCopy.source.rowsPerImage,
    4,
    'multi-slice BC7 rowsPerImage must be corrected from 16 texel rows to 4 block rows'
  );
  assert.deepEqual(cubeCopy.copySize, { width: 16, height: 16, depthOrArrayLayers: 6 });

  // 2) Tight-last-row copyBufferToTexture (CPU bytes) for 16x16 BC7 (bytesPerRow=128, tight last row=64 -> 448B).
  const enc = gpu.device.createCommandEncoder();
  const tightSingle = new Uint8Array(3 * 128 + 64);
  tightSingle.fill(0x33);
  const c2tRes = handler.copyBufferToTexture(
    enc,
    { bytesPerRow: 128, rowsPerImage: 4 },
    { texture: processor.textures.get(5001), mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 16, height: 16, depthOrArrayLayers: 1 },
    tightSingle
  );
  assert.equal(c2tRes.error, undefined);
  assert.equal(c2tRes.staged, true);
  assert.equal(c2tRes.bytesPerRow, 256);

  // 3) GPU-buffer-only unaligned copyBufferToTexture clamps final row to srcBuffer.size (never reads past EOF).
  const tightGpuBuf = { id: 'gpu-tight', size: 3 * 128 + 64 };
  const encGpu = gpu.device.createCommandEncoder();
  const c2tGpu = handler.copyBufferToTexture(
    encGpu,
    { buffer: tightGpuBuf, offset: 0, bytesPerRow: 128, rowsPerImage: 4 },
    { texture: processor.textures.get(5001), mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
    { width: 16, height: 16, depthOrArrayLayers: 1 }
  );
  assert.equal(c2tGpu.error, undefined);
  const b2bCalls = gpu.calls.filter((c) => c.op === 'copyBufferToBuffer' && c.srcBuf === tightGpuBuf);
  assert.equal(b2bCalls.length, 4);
  assert.equal(b2bCalls[0].size, 128);
  assert.equal(b2bCalls[3].size, 64, 'final GPU copyBufferToBuffer row must clamp to remaining srcBuffer.size');
  assert.equal(handler.getStats().errors, 0, handler.lastError);
});

test('Console texture fidelity: direct queue.writeTexture flushes earlier staged write on same texture (not other textures)', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const stagingPool = new GpuStagingPool({ device: gpu.device, chunkSize: 64 * 1024 });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { stagingPool });
  const handler = tp.installTexturePackets(processor);

  const eventLog = [];
  const origFlush = processor._flushPendingUploads.bind(processor);
  processor._flushPendingUploads = function trackedFlush(retire) {
    eventLog.push('flushPendingUploads');
    return origFlush(retire);
  };
  const origWriteTexture = gpu.device.queue.writeTexture.bind(gpu.device.queue);
  gpu.device.queue.writeTexture = function trackedWriteTexture(dest, data, layout, size) {
    eventLog.push(`writeTexture:${dest.texture === processor.textures.get(5010) ? 'A' : 'B'}`);
    return origWriteTexture(dest, data, layout, size);
  };

  // Texture A (UI/font atlas) and Texture B (separate plane).
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 5010, width: 64, height: 4, format: 22, usage: 0x06 }) // rgba8unorm
  );
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 5011, width: 64, height: 4, format: 22, usage: 0x06 }) // rgba8unorm
  );

  // 1) Unaligned dirty-rect write to Texture A -> staged onto _pendingEncoder.
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 5010,
      width: 8,
      height: 2,
      bytesPerRow: 32,
      rowsPerImage: 2,
      data: new Uint8Array(64).fill(1),
    })
  );
  assert.equal(processor._hasPendingUploadCommands, true);

  // 2) 256B-aligned direct write to Texture B (DIFFERENT texture) -> must NOT flush A's pending encoder!
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 5011,
      width: 64,
      height: 2,
      bytesPerRow: 256,
      rowsPerImage: 2,
      data: new Uint8Array(512).fill(2),
    })
  );
  assert.deepEqual(eventLog, ['writeTexture:B']);
  assert.equal(processor._hasPendingUploadCommands, true);

  // 3) 256B-aligned direct write to Texture A (SAME texture!) -> MUST flush pending staged copy BEFORE queue.writeTexture!
  processor.executeCommandBuffer(
    writeTexturePacket({
      textureId: 5010,
      width: 64,
      height: 2,
      bytesPerRow: 256,
      rowsPerImage: 2,
      data: new Uint8Array(512).fill(3),
    })
  );
  assert.deepEqual(eventLog, ['writeTexture:B', 'flushPendingUploads', 'writeTexture:A']);
  assert.equal(handler.getStats().errors, 0, handler.lastError);
});

test('Console texture fidelity: 3D BC-sRGB fallback preserves sRGB and _create sanitizes illegal WebGPU usages/MSAA', () => {
  const gpu = createMockWebGpu({
    strictWebGpu: true,
    features: new Set(['texture-compression-bc']), // no bc-sliced-3d, no bgra8unorm-storage
  });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);

  // 1) 3D bc7-rgba-unorm-srgb (code 63) without bc-sliced-3d falls back to rgba8unorm-srgb (preserves sRGB).
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 5020,
      width: 8,
      height: 8,
      layers: 4,
      dimension: 2, // 3d
      format: 63, // bc7-rgba-unorm-srgb
      usage: 0x06 | 0x08 | 0x10, // COPY_DST | TEXTURE_BINDING | illegal STORAGE_BINDING & RENDER_ATTACHMENT
    })
  );
  const tex5020 = processor.textures.get(5020);
  assert.ok(tex5020);
  assert.equal(tex5020.desc.format, 'rgba8unorm-srgb');
  assert.equal(tex5020.desc.usage & 0x08, 0, 'sRGB fallback must strip STORAGE_BINDING');

  // 2) 2D rgba8unorm-srgb (code 23) with STORAGE_BINDING strips STORAGE_BINDING while keeping RENDER_ATTACHMENT.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 5021,
      width: 32,
      height: 32,
      format: 23, // rgba8unorm-srgb
      usage: 0x06 | 0x08 | 0x10,
    })
  );
  const tex5021 = processor.textures.get(5021);
  assert.equal(tex5021.desc.format, 'rgba8unorm-srgb');
  assert.equal(tex5021.desc.usage & 0x08, 0, 'rgba8unorm-srgb must strip STORAGE_BINDING');
  assert.notEqual(tex5021.desc.usage & 0x10, 0, 'rgba8unorm-srgb keeps RENDER_ATTACHMENT');

  // 3) 2D bgra8unorm (code 27) without bgra8unorm-storage strips STORAGE_BINDING.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 5022,
      width: 32,
      height: 32,
      format: 27, // bgra8unorm
      usage: 0x06 | 0x08 | 0x10,
    })
  );
  const tex5022 = processor.textures.get(5022);
  assert.equal(tex5022.desc.usage & 0x08, 0, 'bgra8unorm without bgra8unorm-storage must strip STORAGE_BINDING');

  // 4) 2D bc7-rgba-unorm (code 62) strips both STORAGE_BINDING and RENDER_ATTACHMENT.
  processor.executeCommandBuffer(
    createTexture2Packet({
      id: 5023,
      width: 16,
      height: 16,
      format: 62, // bc7-rgba-unorm
      usage: 0x06 | 0x08 | 0x10,
    })
  );
  const tex5023 = processor.textures.get(5023);
  assert.equal(tex5023.desc.usage & (0x08 | 0x10), 0, 'BC7 must strip STORAGE_BINDING and RENDER_ATTACHMENT');

  assert.equal(handler.getStats().errors, 0, handler.lastError);
});

test('modern direct SAB writes avoid app snapshots and retain synchronous source bytes', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);
  processor.executeCommandBuffer(createTexture2Packet({ id: 8801, width: 256, height: 2, usage: 0x06, format: 1 }));
  const data = new Uint8Array(512).fill(37);
  const pkt = writeTexturePacket({ textureId: 8801, width: 256, height: 2, bytesPerRow: 256, rowsPerImage: 2, data, useSab: true });
  processor.executeCommandBuffer(pkt);
  // The API's content-timeline copy must precede producer reuse of the ring bytes.
  new Uint8Array(pkt.buffer, pkt.byteOffset + 56, 512).fill(91);
  const call = gpu.calls.find((c) => c.op === 'writeTexture');
  assert.equal(call.isSab, true);
  assert.equal(call.bufferRef, pkt.buffer);
  assert.ok(call.bytes.every((byte) => byte === 37));
  assert.equal(handler.getStats().sabCopyCount, 0);
  assert.equal(handler.getStats().scratchBufferAllocations, 0);
  assert.equal(handler.getStats().directZeroCopyUploads, 1);
  assert.equal(handler._reusablePreparedData.data, null);
});

test('legacy SAB TypeError is probed once per queue binding and fallback telemetry stays accurate', () => {
  const gpu = createMockWebGpu({ strictWebGpu: true });
  const original = gpu.device.queue.writeTexture;
  let sharedAttempts = 0;
  gpu.device.queue.writeTexture = function (...args) {
    if (args[1].buffer instanceof SharedArrayBuffer) sharedAttempts++;
    return original.apply(this, args);
  };
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);
  processor.executeCommandBuffer(createTexture2Packet({ id: 8802, width: 256, height: 2, usage: 0x06, format: 1 }));
  const pkt = writeTexturePacket({ textureId: 8802, width: 256, height: 2, bytesPerRow: 256, rowsPerImage: 2, data: new Uint8Array(512).fill(52), useSab: true });
  for (let i = 0; i < 3; i++) processor.executeCommandBuffer(pkt);
  assert.equal(sharedAttempts, 1);
  const calls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(calls.length, 3);
  assert.ok(calls.every((call) => !call.isSab && call.bytes[0] === 52));
  assert.equal(handler.getStats().sabCopyCount, 3);
  assert.equal(handler.getStats().scratchBufferAllocations, 1);
  assert.equal(handler.getStats().scratchBufferReuses, 2);
  // A replacement/recovered device queue gets its own capability probe.
  const modern = createMockWebGpu();
  processor.device = modern.device;
  processor.executeCommandBuffer(pkt);
  assert.equal(modern.calls.find((c) => c.op === 'writeTexture').isSab, true);
  assert.equal(handler.getStats().sabCopyCount, 3);
});

test('direct SAB writes propagate non-TypeError failures without copying or retrying', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);
  const texture = gpu.device.createTexture({ size: { width: 256, height: 2 }, format: 'r8unorm', usage: 6 });
  let attempts = 0;
  const failure = new RangeError('synthetic write failure');
  gpu.device.queue.writeTexture = () => { attempts++; throw failure; };
  const shared = new Uint8Array(new SharedArrayBuffer(512));
  assert.throws(() => handler.writeTexture({ texture }, shared, { bytesPerRow: 256, rowsPerImage: 2 }, { width: 256, height: 2 }), (error) => error === failure);
  assert.equal(attempts, 1);
  assert.equal(handler.getStats().sabCopyCount, 0);
  assert.equal(handler.getStats().scratchBufferAllocations, 0);
});

test('shared uploads above the chunk cap retain bounded snapshots on modern queues', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor, { maxWriteChunkBytes: 256 });
  processor.executeCommandBuffer(createTexture2Packet({ id: 8803, width: 256, height: 2, usage: 0x06, format: 1 }));
  const pkt = writeTexturePacket({ textureId: 8803, width: 256, height: 2, bytesPerRow: 256, rowsPerImage: 2, data: new Uint8Array(512).fill(63), useSab: true });
  processor.executeCommandBuffer(pkt);
  const calls = gpu.calls.filter((c) => c.op === 'writeTexture');
  assert.equal(calls.length, 2);
  assert.ok(calls.every((call) => !call.isSab && call.byteLength === 256 && call.bytes[0] === 63));
  assert.equal(handler.getStats().sabCopyCount, 1);
  assert.equal(handler.getStats().chunkedWrites, 1);
});


test('cross-realm TypeError selects legacy fallback and counts snapshots even when retry fails', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);
  const texture = gpu.device.createTexture({ size: { width: 256, height: 2 }, format: 'r8unorm', usage: 6 });
  const crossRealmTypeError = vm.runInNewContext('new TypeError("shared views unsupported")');
  const retryError = new RangeError('retry failed');
  let sharedAttempts = 0;
  gpu.device.queue.writeTexture = (_destination, data) => {
    if (data.buffer instanceof SharedArrayBuffer) { sharedAttempts++; throw crossRealmTypeError; }
    throw retryError;
  };
  const shared = new Uint8Array(new SharedArrayBuffer(512));
  for (let i = 0; i < 2; i++) {
    assert.throws(() => handler.writeTexture({ texture }, shared, { bytesPerRow: 256, rowsPerImage: 2 }, { width: 256, height: 2 }), (error) => error === retryError);
  }
  assert.equal(sharedAttempts, 1);
  assert.equal(handler.getStats().sabCopyCount, 2);
});
