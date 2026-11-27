import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, OPCODES } = require('../web/workers/wgpu_worker.js');
const rs = require('../web/workers/render_state_packets.js');
const tp = require('../web/workers/texture_packets.js');
const { DrawExecutor } = require('../web/workers/draw_executor.js');

const {
  installRenderStatePackets,
  sanitizeTextureDescriptor,
  resolveCompatibleViewFormats: rsResolveCompatibleViewFormats,
  V2_OPCODES,
  TEXTURE_FORMAT_NAMES,
} = rs;

const USAGE_COPY_SRC = 0x0001;
const USAGE_COPY_DST = 0x0002;
const USAGE_TEXTURE_BINDING = 0x0004;
const USAGE_STORAGE_BINDING = 0x0008;
const USAGE_RENDER_ATTACHMENT = 0x0010;
const USAGE_ALL =
  USAGE_COPY_SRC |
  USAGE_COPY_DST |
  USAGE_TEXTURE_BINDING |
  USAGE_STORAGE_BINDING |
  USAGE_RENDER_ATTACHMENT; // 0x1f — matches live Chrome Report 18

const formatCodeOf = (name) => {
  const code = TEXTURE_FORMAT_NAMES.indexOf(name);
  assert.ok(code > 0, `unknown texture format ${name}`);
  return code;
};

function header(opcode, payload) {
  const out = new Uint8Array(8 + payload.byteLength);
  const v = new DataView(out.buffer);
  v.setUint32(0, opcode, true);
  v.setUint32(4, payload.byteLength, true);
  out.set(payload, 8);
  return out;
}

function concat(...parts) {
  const total = parts.reduce((n, p) => n + p.byteLength, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.byteLength;
  }
  return out;
}

function createTexture1Packet({
  id,
  width = 64,
  height = 64,
  depth = 1,
  mips = 1,
  format = 0, // rgba8unorm
  usage = USAGE_COPY_DST | USAGE_TEXTURE_BINDING,
  dimension = 1, // 2d
}) {
  const buf = new Uint8Array(32);
  const v = new DataView(buf.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, width, true);
  v.setUint32(8, height, true);
  v.setUint32(12, depth, true);
  v.setUint32(16, mips, true);
  v.setUint32(20, format, true);
  v.setUint32(24, usage, true);
  v.setUint32(28, dimension, true);
  return header(OPCODES.CREATE_TEXTURE, buf);
}

function createTexture2Packet({
  id,
  width = 2048,
  height = 2048,
  depth = 1,
  mips = 1,
  sampleCount = 1,
  usage = USAGE_ALL,
  dimension = 1, // 2d
  format,
  viewFormats = [],
}) {
  const buf = new Uint8Array(40);
  const v = new DataView(buf.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, width, true);
  v.setUint32(8, height, true);
  v.setUint32(12, depth, true);
  v.setUint32(16, mips, true);
  v.setUint32(20, sampleCount, true);
  v.setUint32(24, usage, true);
  v.setUint8(28, dimension);
  v.setUint8(29, format);
  v.setUint8(30, viewFormats.length);
  for (let i = 0; i < viewFormats.length; i += 1) {
    v.setUint8(32 + i, viewFormats[i]);
  }
  return header(V2_OPCODES.CREATE_TEXTURE2, buf);
}

function bindGroupLayoutPacket(id, entries, { contentId = 0n, flags = 0 } = {}) {
  const b = new Uint8Array(24 + entries.length * 24);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, entries.length, true);
  v.setUint32(8, flags, true);
  v.setBigUint64(16, contentId, true);
  entries.forEach((e, i) => {
    const o = 24 + i * 24;
    v.setUint32(o, e.binding, true);
    v.setUint32(o + 4, e.visibility ?? 0x7, true);
    b[o + 8] = e.type;
    b[o + 9] = e.subtype || 0;
    b[o + 10] = e.dimension || 0;
    b[o + 11] = e.flags || 0;
    b[o + 12] = e.storageFormat || 0;
    v.setUint16(o + 14, e.bindingArraySize || 0, true);
    v.setBigUint64(o + 16, BigInt(e.minBindingSize || 0), true);
  });
  return header(V2_OPCODES.CREATE_BIND_GROUP_LAYOUT, b);
}

function viewDesc({
  format = 0,
  dimension = 2,
  aspect = 0,
  baseMip = 0,
  mipCount = 0,
  baseLayer = 0,
  layerCount = 0,
} = {}) {
  const b = new Uint8Array(16);
  const v = new DataView(b.buffer);
  b[0] = format;
  b[1] = dimension;
  b[2] = aspect;
  v.setUint16(4, baseMip, true);
  v.setUint16(6, mipCount, true);
  v.setUint16(8, baseLayer, true);
  v.setUint16(10, layerCount, true);
  return b;
}

function bindGroup2Packet(id, layoutId, entries, { flags = 0 } = {}) {
  const b = new Uint8Array(16 + entries.length * 48);
  const v = new DataView(b.buffer);
  v.setUint32(0, id, true);
  v.setUint32(4, layoutId, true);
  v.setUint32(8, entries.length, true);
  v.setUint32(12, flags, true);
  entries.forEach((e, i) => {
    const o = 16 + i * 48;
    v.setUint32(o, e.binding, true);
    v.setUint32(o + 4, e.type, true);
    v.setUint32(o + 8, e.id, true);
    v.setUint16(o + 12, e.arrayIndex || 0, true);
    v.setBigUint64(o + 16, BigInt(e.offset || 0), true);
    v.setBigUint64(o + 24, BigInt(e.size || 0), true);
    if (e.view) b.set(e.view, o + 32);
  });
  return header(V2_OPCODES.CREATE_BIND_GROUP2, b);
}

/**
 * Strict Dawn-like WebGPU mock enforcing:
 * 1) createTexture rejects STORAGE_BINDING + `-srgb` format or `-srgb` viewFormats
 * 2) texture.createView rejects `-srgb` format when parent texture has STORAGE_BINDING
 *    OR when requested view format != base format and not listed in texture.viewFormats
 */
function createStrictSrgbStorageMockWebGpu({ features = [] } = {}) {
  const createdTextures = [];
  const createdViews = [];
  const createdBindGroups = [];
  const validationErrors = [];

  const device = {
    features: new Set(features),
    limits: {
      maxTextureDimension2D: 8192,
      maxBindGroups: 4,
    },
    createTexture(desc) {
      const hasStorage = (desc.usage & USAGE_STORAGE_BINDING) !== 0;
      if (hasStorage && typeof desc.format === 'string' && desc.format.endsWith('-srgb')) {
        const msg = `createTexture: STORAGE_BINDING incompatible with format ${desc.format}`;
        validationErrors.push(msg);
        throw new Error(msg);
      }
      if (hasStorage && Array.isArray(desc.viewFormats)) {
        const bad = desc.viewFormats.find((vf) => typeof vf === 'string' && vf.endsWith('-srgb'));
        if (bad) {
          const msg = `createTexture: STORAGE_BINDING incompatible with viewFormat ${bad}`;
          validationErrors.push(msg);
          throw new Error(msg);
        }
      }
      const tex = {
        kind: 'GPUTexture',
        desc: {
          ...desc,
          size: desc.size ? { ...desc.size } : undefined,
          viewFormats: Array.isArray(desc.viewFormats) ? desc.viewFormats.slice() : undefined,
        },
        width: desc.size?.width ?? 1,
        height: desc.size?.height ?? 1,
        depthOrArrayLayers: desc.size?.depthOrArrayLayers ?? 1,
        mipLevelCount: desc.mipLevelCount ?? 1,
        sampleCount: desc.sampleCount ?? 1,
        dimension: desc.dimension ?? '2d',
        format: desc.format,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
        createView(viewDescriptor) {
          const reqFmt = viewDescriptor && viewDescriptor.format ? viewDescriptor.format : desc.format;
          if ((desc.usage & USAGE_STORAGE_BINDING) !== 0 && typeof reqFmt === 'string' && reqFmt.endsWith('-srgb')) {
            const msg =
              `The texture usage (0x${desc.usage.toString(16)}) includes StorageBinding, ` +
              `which is incompatible with the format (${reqFmt}).`;
            validationErrors.push(msg);
            throw new Error(msg);
          }
          if (reqFmt !== desc.format) {
            const allowed = Array.isArray(desc.viewFormats) ? desc.viewFormats : [];
            if (!allowed.includes(reqFmt)) {
              const msg = `createView: format ${reqFmt} is not in texture viewFormats for ${desc.format}`;
              validationErrors.push(msg);
              throw new Error(msg);
            }
          }
          const view = {
            kind: 'GPUTextureView',
            texture: tex,
            format: reqFmt,
            desc: viewDescriptor ? { ...viewDescriptor } : null,
            rawDescRef: viewDescriptor ?? null,
          };
          createdViews.push(view);
          return view;
        },
      };
      createdTextures.push(tex);
      return tex;
    },
    createBindGroupLayout(desc) {
      return { kind: 'GPUBindGroupLayout', desc };
    },
    createBindGroup(desc) {
      const bg = { kind: 'GPUBindGroup', desc };
      createdBindGroups.push(bg);
      return bg;
    },
    createBuffer(desc) {
      return { kind: 'GPUBuffer', desc, size: desc.size, usage: desc.usage, destroy() {} };
    },
    createSampler(desc) {
      return { kind: 'GPUSampler', desc };
    },
    createCommandEncoder() {
      return {
        beginRenderPass() {
          return { end() {} };
        },
        finish() {
          return { kind: 'GPUCommandBuffer' };
        },
      };
    },
    queue: {
      submit() {},
      writeBuffer() {},
      writeTexture() {},
    },
  };

  const ctx = {
    configure() {},
    getCurrentTexture() {
      return {
        kind: 'GPUTexture',
        width: 1280,
        height: 720,
        format: 'bgra8unorm',
        usage: USAGE_RENDER_ATTACHMENT | USAGE_COPY_SRC,
        createView(desc) {
          const view = { kind: 'GPUTextureView', swapchain: true, desc: desc ? { ...desc } : null };
          createdViews.push(view);
          return view;
        },
      };
    },
  };

  return { device, ctx, createdTextures, createdViews, createdBindGroups, validationErrors };
}

test('createTexture strips -srgb viewFormats on linear STORAGE_BINDING textures across wgpu_worker, render_state_packets, and texture_packets', () => {
  // 1) render_state_packets: resolveCompatibleViewFormats & sanitizeTextureDescriptor
  const vfStorage = rsResolveCompatibleViewFormats('rgba8unorm', ['rgba8unorm-srgb'], {
    usage: USAGE_ALL,
  });
  assert.deepEqual(vfStorage.viewFormats, []);
  assert.deepEqual(vfStorage.dropped, ['rgba8unorm-srgb']);

  const vfNonStorage = rsResolveCompatibleViewFormats('rgba8unorm', ['rgba8unorm-srgb'], {
    usage: USAGE_TEXTURE_BINDING | USAGE_RENDER_ATTACHMENT,
  });
  assert.deepEqual(vfNonStorage.viewFormats, ['rgba8unorm-srgb']);
  assert.deepEqual(vfNonStorage.dropped, []);

  const mockDevice = { features: new Set(['bgra8unorm-storage']) };
  const sanitizedRgba = sanitizeTextureDescriptor(
    {
      size: { width: 2048, height: 2048, depthOrArrayLayers: 1 },
      mipLevelCount: 1,
      sampleCount: 1,
      dimension: '2d',
      format: 'rgba8unorm',
      usage: USAGE_ALL,
      viewFormats: ['rgba8unorm-srgb'],
    },
    mockDevice
  );
  assert.equal(sanitizedRgba.descriptor.format, 'rgba8unorm');
  assert.equal(sanitizedRgba.descriptor.usage & USAGE_STORAGE_BINDING, USAGE_STORAGE_BINDING);
  assert.equal(sanitizedRgba.descriptor.viewFormats, undefined);

  const sanitizedBgraStorage = sanitizeTextureDescriptor(
    {
      size: { width: 1024, height: 1024, depthOrArrayLayers: 1 },
      mipLevelCount: 1,
      sampleCount: 1,
      dimension: '2d',
      format: 'bgra8unorm',
      usage: USAGE_TEXTURE_BINDING | USAGE_STORAGE_BINDING,
      viewFormats: ['bgra8unorm-srgb'],
    },
    mockDevice
  );
  assert.equal(sanitizedBgraStorage.descriptor.usage & USAGE_STORAGE_BINDING, USAGE_STORAGE_BINDING);
  assert.equal(sanitizedBgraStorage.descriptor.viewFormats, undefined);

  // sRGB base format with STORAGE_BINDING strips STORAGE_BINDING and keeps compatible linear sibling
  const sanitizedSrgbBase = sanitizeTextureDescriptor(
    {
      size: { width: 512, height: 512, depthOrArrayLayers: 1 },
      mipLevelCount: 1,
      sampleCount: 1,
      dimension: '2d',
      format: 'rgba8unorm-srgb',
      usage: USAGE_ALL,
    },
    mockDevice
  );
  assert.equal(sanitizedSrgbBase.descriptor.format, 'rgba8unorm-srgb');
  assert.equal(sanitizedSrgbBase.descriptor.usage & USAGE_STORAGE_BINDING, 0);
  assert.deepEqual(sanitizedSrgbBase.descriptor.viewFormats, ['rgba8unorm']);

  // 2) texture_packets: resolveCompatibleViewFormats
  assert.deepEqual(
    tp.resolveCompatibleViewFormats('rgba8unorm', ['rgba8unorm-srgb'], { usage: USAGE_ALL }),
    { viewFormats: undefined, siblingAdded: false }
  );
  assert.deepEqual(
    tp.resolveCompatibleViewFormats('rgba8unorm', undefined, { usage: USAGE_ALL }),
    { viewFormats: undefined, siblingAdded: false }
  );
  assert.deepEqual(
    tp.resolveCompatibleViewFormats('rgba8unorm', ['rgba8unorm-srgb'], {
      usage: USAGE_TEXTURE_BINDING | USAGE_COPY_DST,
    }),
    { viewFormats: ['rgba8unorm-srgb'], siblingAdded: false }
  );

  // 3) wgpu_worker: _normalizeTextureDescriptor (zero-alloc on clean path, strips -srgb viewFormats on storage)
  const gpu = createStrictSrgbStorageMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const cleanDesc = {
    size: { width: 64, height: 64, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: '2d',
    format: 'rgba8unorm',
    usage: USAGE_COPY_SRC | USAGE_TEXTURE_BINDING | USAGE_RENDER_ATTACHMENT,
    viewFormats: ['rgba8unorm-srgb'],
  };
  assert.equal(
    processor._normalizeTextureDescriptor(cleanDesc),
    cleanDesc,
    'clean non-storage descriptor must return original object with zero allocation'
  );

  const storageDesc = {
    size: { width: 2048, height: 2048, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: '2d',
    format: 'rgba8unorm',
    usage: USAGE_ALL,
    viewFormats: ['rgba8unorm-srgb'],
  };
  const normalizedStorage = processor._normalizeTextureDescriptor(storageDesc);
  assert.equal(normalizedStorage.usage & USAGE_STORAGE_BINDING, USAGE_STORAGE_BINDING);
  assert.equal(normalizedStorage.viewFormats, undefined);

  const srgbBaseStorage = processor._normalizeTextureDescriptor({
    ...storageDesc,
    format: 'rgba8unorm-srgb',
    viewFormats: ['rgba8unorm'],
  });
  assert.equal(srgbBaseStorage.usage & USAGE_STORAGE_BINDING, 0);
  assert.deepEqual(srgbBaseStorage.viewFormats, ['rgba8unorm']);
});

test('live Chrome Report 18 regression: 2048x2048 rgba8unorm STORAGE_BINDING texture + sampled rgba8unorm-srgb view succeeds end-to-end with RenderStatePackets + TexturePackets', () => {
  const gpu = createStrictSrgbStorageMockWebGpu({ features: ['bgra8unorm-storage'] });
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const rsHandler = installRenderStatePackets(processor);
  const tpHandler = tp.installTexturePackets(processor);

  const rgbaCode = formatCodeOf('rgba8unorm');
  const rgbaSrgbCode = formatCodeOf('rgba8unorm-srgb');
  const bgraCode = formatCodeOf('bgra8unorm');
  const bgraSrgbCode = formatCodeOf('bgra8unorm-srgb');

  const stream = concat(
    // 1) Guest 2048x2048 rgba8unorm texture with USAGE_ALL (0x1f) + explicit rgba8unorm-srgb viewFormat
    createTexture2Packet({
      id: 100,
      width: 2048,
      height: 2048,
      format: rgbaCode,
      usage: USAGE_ALL,
      viewFormats: [rgbaSrgbCode],
    }),
    // 2) Normal (non-storage) rgba8unorm texture with rgba8unorm-srgb viewFormat
    createTexture2Packet({
      id: 101,
      width: 512,
      height: 512,
      format: rgbaCode,
      usage: USAGE_COPY_DST | USAGE_TEXTURE_BINDING | USAGE_RENDER_ATTACHMENT,
      viewFormats: [rgbaSrgbCode],
    }),
    // 3) bgra8unorm texture with STORAGE_BINDING + explicit bgra8unorm-srgb viewFormat
    createTexture2Packet({
      id: 102,
      width: 256,
      height: 256,
      format: bgraCode,
      usage: USAGE_TEXTURE_BINDING | USAGE_STORAGE_BINDING,
      viewFormats: [bgraSrgbCode],
    }),
    // 4) v1 CreateTexture rgba8unorm with STORAGE_BINDING
    createTexture1Packet({
      id: 103,
      width: 128,
      height: 128,
      format: 0,
      usage: USAGE_ALL,
    }),
    // Sampled texture BGL (NOT storageTexture — this is the exact Report 18 path!)
    bindGroupLayoutPacket(200, [
      { binding: 0, visibility: 0x2, type: 2, subtype: 0, dimension: 2 },
      { binding: 1, visibility: 0x2, type: 2, subtype: 0, dimension: 2 },
      { binding: 2, visibility: 0x2, type: 2, subtype: 0, dimension: 2 },
      { binding: 3, visibility: 0x2, type: 2, subtype: 0, dimension: 2 },
    ]),
    // BindGroup2 requesting -srgb views on all four textures
    bindGroup2Packet(300, 200, [
      { binding: 0, type: 1, id: 100, view: viewDesc({ format: rgbaSrgbCode, dimension: 2 }) },
      { binding: 1, type: 1, id: 101, view: viewDesc({ format: rgbaSrgbCode, dimension: 2 }) },
      { binding: 2, type: 1, id: 102, view: viewDesc({ format: bgraSrgbCode, dimension: 2 }) },
      { binding: 3, type: 1, id: 103, view: viewDesc({ format: rgbaSrgbCode, dimension: 2 }) },
    ])
  );

  processor.executeCommandBuffer(stream);

  assert.deepEqual(gpu.validationErrors, [], 'strict WebGPU mock must report zero validation errors');
  assert.equal(rsHandler.stats.errors, 0, rsHandler.lastError || '');
  assert.equal(tpHandler.stats.errors, 0, tpHandler.lastError || '');

  // Verify created textures kept STORAGE_BINDING and omitted -srgb viewFormats where incompatible
  const tex100 = processor.textures.get(100);
  assert.equal(tex100.desc.format, 'rgba8unorm');
  assert.equal(tex100.desc.usage, USAGE_ALL);
  assert.equal(tex100.desc.viewFormats, undefined);

  const tex101 = processor.textures.get(101);
  assert.equal(tex101.desc.format, 'rgba8unorm');
  assert.equal(tex101.desc.usage & USAGE_STORAGE_BINDING, 0);
  assert.deepEqual(tex101.desc.viewFormats, ['rgba8unorm-srgb']);

  const tex102 = processor.textures.get(102);
  assert.equal(tex102.desc.format, 'bgra8unorm');
  assert.equal(tex102.desc.usage & USAGE_STORAGE_BINDING, USAGE_STORAGE_BINDING);
  assert.equal(tex102.desc.viewFormats, undefined);

  const tex103 = processor.textures.get(103);
  assert.equal(tex103.desc.format, 'rgba8unorm');
  assert.equal(tex103.desc.usage, USAGE_ALL);
  assert.equal(tex103.desc.viewFormats, undefined);

  // Verify bind group entries got real views (not dummies) with exact expected view formats
  assert.equal(gpu.createdBindGroups.length, 1);
  const bgEntries = gpu.createdBindGroups[0].desc.entries;
  assert.equal(bgEntries[0].resource.texture, tex100);
  assert.equal(
    bgEntries[0].resource.format,
    'rgba8unorm',
    'STORAGE_BINDING rgba8unorm texture requesting rgba8unorm-srgb view must fall back to rgba8unorm'
  );
  assert.equal(bgEntries[1].resource.texture, tex101);
  assert.equal(
    bgEntries[1].resource.format,
    'rgba8unorm-srgb',
    'non-storage rgba8unorm texture with rgba8unorm-srgb viewFormat must preserve rgba8unorm-srgb'
  );
  assert.equal(bgEntries[2].resource.texture, tex102);
  assert.equal(
    bgEntries[2].resource.format,
    'bgra8unorm',
    'STORAGE_BINDING bgra8unorm texture requesting bgra8unorm-srgb view must fall back to bgra8unorm'
  );
  assert.equal(bgEntries[3].resource.texture, tex103);
  assert.equal(
    bgEntries[3].resource.format,
    'rgba8unorm',
    'v1 STORAGE_BINDING rgba8unorm texture requesting rgba8unorm-srgb view must fall back to rgba8unorm'
  );
});

test('_createSafeTextureView, getOrCreateTextureView, and DrawExecutor guard STORAGE_BINDING & missing viewFormats across all metadata sources while preserving non-storage sRGB and swapchain views', () => {
  const gpu = createStrictSrgbStorageMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx, { format: 'bgra8unorm' });
  const rsHandler = installRenderStatePackets(processor);
  const executor = new DrawExecutor(gpu.device);

  // A) _createSafeTextureView on a texture with STORAGE_BINDING (via texture.usage) -> rewrites to linear
  const storageTex = gpu.device.createTexture({
    size: { width: 2048, height: 2048, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: '2d',
    format: 'rgba8unorm',
    usage: USAGE_ALL,
  });
  const safeView1 = processor._createSafeTextureView(storageTex, {
    format: 'rgba8unorm-srgb',
    dimension: '2d',
  });
  assert.equal(safeView1.format, 'rgba8unorm');

  // B) _createSafeTextureView when usage is only tracked in processor._textureMeta (minimal texture without .usage/.desc)
  let minimalViewDesc = null;
  const minimalStorageTex = {
    format: 'bgra8unorm',
    createView(d) {
      minimalViewDesc = d;
      return { kind: 'GPUTextureView', format: d?.format || 'bgra8unorm' };
    },
  };
  processor._textureMeta.set(777, {
    texture: minimalStorageTex,
    format: 'bgra8unorm',
    usage: USAGE_TEXTURE_BINDING | USAGE_STORAGE_BINDING,
    viewFormats: null,
  });
  const safeView2 = processor._createSafeTextureView(
    minimalStorageTex,
    { format: 'bgra8unorm-srgb' },
    777
  );
  assert.equal(safeView2.format, 'bgra8unorm');
  assert.equal(minimalViewDesc.format, 'bgra8unorm');

  // C) _createSafeTextureView on a normal non-storage texture with viewFormats: ['rgba8unorm-srgb'] -> preserves sRGB with zero rewrite allocation
  const nonStorageTex = gpu.device.createTexture({
    size: { width: 256, height: 256, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: '2d',
    format: 'rgba8unorm',
    usage: USAGE_COPY_DST | USAGE_TEXTURE_BINDING,
    viewFormats: ['rgba8unorm-srgb'],
  });
  const originalViewDesc = { format: 'rgba8unorm-srgb', dimension: '2d' };
  const safeView3 = processor._createSafeTextureView(nonStorageTex, originalViewDesc);
  assert.equal(safeView3.format, 'rgba8unorm-srgb');
  assert.equal(
    safeView3.rawDescRef,
    originalViewDesc,
    'non-storage valid sRGB view must pass descriptor through without allocating a copy'
  );

  // D) Non-storage linear texture WITHOUT rgba8unorm-srgb in viewFormats -> rewrites to linear rgba8unorm
  const noSrgbVfTex = gpu.device.createTexture({
    size: { width: 128, height: 128, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: '2d',
    format: 'rgba8unorm',
    usage: USAGE_TEXTURE_BINDING,
  });
  assert.equal(
    processor._createSafeTextureView(noSrgbVfTex, { format: 'rgba8unorm-srgb' }).format,
    'rgba8unorm'
  );
  assert.equal(
    rsHandler.getOrCreateTextureView(noSrgbVfTex, { format: 'rgba8unorm-srgb' }).format,
    'rgba8unorm'
  );
  assert.equal(
    executor.getOrCreateTextureView(noSrgbVfTex, { format: 'rgba8unorm-srgb' }).format,
    'rgba8unorm'
  );

  // E) DrawExecutor on storage texture vs non-storage sRGB texture
  assert.equal(
    executor.getOrCreateTextureView(storageTex, { format: 'rgba8unorm-srgb' }).format,
    'rgba8unorm'
  );
  const execSrgbView = executor.getOrCreateTextureView(nonStorageTex, originalViewDesc);
  assert.equal(execSrgbView.format, 'rgba8unorm-srgb');
  assert.equal(execSrgbView.rawDescRef, originalViewDesc);

  // F) Swapchain canvas texture (no STORAGE_BINDING, no .viewFormats, no .desc, no meta) preserves -srgb view!
  const swapTex = gpu.ctx.getCurrentTexture();
  const swapSrgbDesc = { format: 'bgra8unorm-srgb' };
  const swapView = rsHandler.getOrCreateTextureView(swapTex, swapSrgbDesc, null, 0);
  assert.deepEqual(swapView.desc, { format: 'bgra8unorm-srgb' });

  assert.deepEqual(gpu.validationErrors, []);
});
