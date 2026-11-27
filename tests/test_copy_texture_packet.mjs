import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const wgpuMod = require('../web/workers/wgpu_worker.js');
const rsMod = require('../web/workers/render_state_packets.js');
const tpMod = require('../web/workers/texture_packets.js');

const {
  WgpuCommandProcessor,
  OPCODES,
  OPCODE_COPY_TEXTURE,
  COPY_TEXTURE_PAYLOAD_BYTES,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  SHADER_STAGE,
} = wgpuMod;

const {
  installRenderStatePackets,
  V2_OPCODES,
  OPCODE_COPY_TEXTURE: RS_OPCODE_COPY_TEXTURE,
  COPY_TEXTURE_PAYLOAD_BYTES: RS_COPY_TEXTURE_PAYLOAD_BYTES,
  TEXTURE_FORMAT_NAMES: V2_TEXTURE_FORMAT_NAMES,
} = rsMod;

const {
  installTexturePackets,
  OPCODE_COPY_TEXTURE: TP_OPCODE_COPY_TEXTURE,
  COPY_TEXTURE_PAYLOAD_BYTES: TP_COPY_TEXTURE_PAYLOAD_BYTES,
} = tpMod;

function v2FormatCode(name) {
  const idx = V2_TEXTURE_FORMAT_NAMES.indexOf(name);
  assert.ok(idx > 0, `unknown v2 format ${name}`);
  return idx;
}

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

  createTexture(
    id,
    width,
    height,
    depthOrLayers = 1,
    mipLevels = 1,
    format = TEXTURE_FORMAT.RGBA8_UNORM,
    usage = TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
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
    view.setUint32(24, usage, true);
    view.setUint32(28, dimension, true);
    return this.packet(OPCODES.CREATE_TEXTURE, new Uint8Array(buf));
  }

  createTexture2({
    id,
    width,
    height,
    depthOrLayers = 1,
    mipLevels = 1,
    sampleCount = 1,
    usage = TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    dimension = TEXTURE_DIMENSION.DIM_2D,
    format = 'rgba8unorm',
    viewFormats = [],
  }) {
    const buf = new ArrayBuffer(40);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, width, true);
    view.setUint32(8, height, true);
    view.setUint32(12, depthOrLayers, true);
    view.setUint32(16, mipLevels, true);
    view.setUint32(20, sampleCount, true);
    view.setUint32(24, usage, true);
    view.setUint8(28, dimension);
    view.setUint8(29, typeof format === 'number' ? format : v2FormatCode(format));
    view.setUint8(30, viewFormats.length);
    view.setUint8(31, 0);
    for (let i = 0; i < Math.min(8, viewFormats.length); i += 1) {
      view.setUint8(
        32 + i,
        typeof viewFormats[i] === 'number' ? viewFormats[i] : v2FormatCode(viewFormats[i])
      );
    }
    return this.packet(OPCODES.CREATE_TEXTURE2, new Uint8Array(buf));
  }

  writeTexture({
    textureId,
    mipLevel = 0,
    originX = 0,
    originY = 0,
    originZ = 0,
    width,
    height,
    depthOrLayers = 1,
    bytesPerRow = 0,
    rowsPerImage = 0,
    bytes,
  }) {
    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const buf = new ArrayBuffer(48 + data.byteLength);
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
    view.setUint32(40, data.byteLength, true);
    view.setUint32(44, 0, true);
    new Uint8Array(buf, 48).set(data);
    return this.packet(OPCODES.WRITE_TEXTURE, new Uint8Array(buf));
  }

  copyTexture({
    dstId,
    srcId,
    dstMip = 0,
    srcMip = 0,
    dstX = 0,
    dstY = 0,
    dstZ = 0,
    srcX = 0,
    srcY = 0,
    srcZ = 0,
    width,
    height,
    depthOrLayers = 1,
    reserved = 0,
  }) {
    const buf = new ArrayBuffer(56);
    const view = new DataView(buf);
    view.setUint32(0, dstId, true);
    view.setUint32(4, srcId, true);
    view.setUint32(8, dstMip, true);
    view.setUint32(12, srcMip, true);
    view.setUint32(16, dstX, true);
    view.setUint32(20, dstY, true);
    view.setUint32(24, dstZ, true);
    view.setUint32(28, srcX, true);
    view.setUint32(32, srcY, true);
    view.setUint32(36, srcZ, true);
    view.setUint32(40, width, true);
    view.setUint32(44, height, true);
    view.setUint32(48, depthOrLayers, true);
    view.setUint32(52, reserved, true);
    return this.packet(OPCODE_COPY_TEXTURE, new Uint8Array(buf));
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

  setPipeline(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this.packet(OPCODES.SET_PIPELINE, new Uint8Array(buf));
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

  createComputePipeline(id, csId, flags = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, csId, true);
    view.setUint32(8, flags, true);
    view.setUint32(12, 0, true);
    return this.packet(OPCODES.CREATE_COMPUTE_PIPELINE, new Uint8Array(buf));
  }

  setComputePipeline(id) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, id, true);
    return this.packet(OPCODES.SET_COMPUTE_PIPELINE, new Uint8Array(buf));
  }

  dispatchCompute(x, y = 1, z = 1) {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setUint32(0, x, true);
    view.setUint32(4, y, true);
    view.setUint32(8, z, true);
    return this.packet(OPCODES.DISPATCH_COMPUTE, new Uint8Array(buf));
  }

  finish() {
    const total = this.chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const c of this.chunks) {
      out.set(c, off);
      off += c.byteLength;
    }
    return out.buffer;
  }
}

function createMockWebGpu({ features = [] } = {}) {
  const events = [];
  const createdTextures = [];
  const copyTextureToTextures = [];
  const submits = [];
  let nextHandle = 1;
  const swapchainView = { kind: 'GPUTextureView', id: 'swapchain-view' };

  const ctx = {
    getCurrentTexture() {
      return {
        createView() {
          return swapchainView;
        },
      };
    },
  };

  const device = {
    features: new Set(features),
    createShaderModule(desc) {
      return { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
    },
    createRenderPipeline(desc) {
      return { kind: 'GPURenderPipeline', handle: nextHandle++, desc };
    },
    createComputePipeline(desc) {
      return { kind: 'GPUComputePipeline', handle: nextHandle++, desc };
    },
    createBuffer(desc) {
      return {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
    },
    createTexture(desc) {
      const size = desc.size || {};
      const tex = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        desc: { ...desc, size: { ...size } },
        width: size.width || 1,
        height: size.height || 1,
        depthOrArrayLayers: size.depthOrArrayLayers || 1,
        mipLevelCount: desc.mipLevelCount || 1,
        sampleCount: desc.sampleCount || 1,
        format: desc.format,
        usage: desc.usage,
        dimension: desc.dimension || '2d',
        destroyed: false,
        createView(viewDesc) {
          return {
            kind: 'GPUTextureView',
            handle: nextHandle++,
            textureHandle: tex.handle,
            viewDesc: viewDesc ? { ...viewDesc } : null,
          };
        },
        destroy() {
          this.destroyed = true;
        },
      };
      createdTextures.push(tex);
      events.push({ op: 'createTexture', handle: tex.handle, desc: tex.desc });
      return tex;
    },
    createCommandEncoder() {
      const encId = nextHandle++;
      const encOps = [];
      return {
        encId,
        beginRenderPass(desc) {
          encOps.push({ op: 'beginRenderPass' });
          events.push({ op: 'beginRenderPass', encId, attachments: (desc.colorAttachments || []).length });
          return {
            setPipeline() {
              encOps.push({ op: 'render.setPipeline' });
            },
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            draw(vc) {
              encOps.push({ op: 'render.draw', vc });
              events.push({ op: 'render.draw', encId, vc });
            },
            drawIndexed() {},
            end() {
              encOps.push({ op: 'renderPass.end' });
              events.push({ op: 'renderPass.end', encId });
            },
          };
        },
        beginComputePass() {
          encOps.push({ op: 'beginComputePass' });
          events.push({ op: 'beginComputePass', encId });
          return {
            setPipeline() {
              encOps.push({ op: 'compute.setPipeline' });
            },
            setBindGroup() {},
            dispatchWorkgroups(x, y, z) {
              encOps.push({ op: 'compute.dispatch', x, y, z });
              events.push({ op: 'compute.dispatch', encId, x, y, z });
            },
            end() {
              encOps.push({ op: 'computePass.end' });
              events.push({ op: 'computePass.end', encId });
            },
          };
        },
        copyBufferToTexture(src, dst, copySize) {
          encOps.push({ op: 'copyBufferToTexture', textureHandle: dst.texture.handle });
          events.push({
            op: 'copyBufferToTexture',
            encId,
            textureHandle: dst.texture.handle,
            copySize: { ...copySize },
          });
        },
        copyTextureToTexture(src, dst, copySize) {
          const rec = {
            src: {
              texture: src.texture,
              mipLevel: src.mipLevel,
              origin: { ...src.origin },
              aspect: src.aspect,
            },
            dst: {
              texture: dst.texture,
              mipLevel: dst.mipLevel,
              origin: { ...dst.origin },
              aspect: dst.aspect,
            },
            copySize: { ...copySize },
          };
          copyTextureToTextures.push(rec);
          encOps.push({ op: 'copyTextureToTexture', ...rec });
          events.push({ op: 'copyTextureToTexture', encId, ...rec });
        },
        finish() {
          const cb = { kind: 'GPUCommandBuffer', handle: nextHandle++, encId, ops: encOps };
          events.push({ op: 'encoder.finish', encId, handle: cb.handle, ops: encOps.map((o) => o.op) });
          return cb;
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture(dst, data, dataLayout, size) {
        events.push({
          op: 'queue.writeTexture',
          textureHandle: dst.texture.handle,
          size: { ...size },
        });
      },
      submit(buffers) {
        submits.push(buffers);
        events.push({
          op: 'queue.submit',
          handles: buffers.map((b) => b.handle),
          ops: buffers.flatMap((b) => b.ops.map((o) => o.op)),
        });
      },
    },
  };

  return { device, ctx, events, createdTextures, copyTextureToTextures, submits };
}

test('CopyTexture opcode 62 matches playgta6_abi.h and keeps OPCODES/V2_OPCODES parity intact', () => {
  assert.equal(OPCODE_COPY_TEXTURE, 62);
  assert.equal(RS_OPCODE_COPY_TEXTURE, 62);
  assert.equal(TP_OPCODE_COPY_TEXTURE, 62);
  assert.equal(COPY_TEXTURE_PAYLOAD_BYTES, 56);
  assert.equal(RS_COPY_TEXTURE_PAYLOAD_BYTES, 56);
  assert.equal(TP_COPY_TEXTURE_PAYLOAD_BYTES, 56);
  assert.equal('COPY_TEXTURE' in OPCODES, false, 'must stay outside OPCODES for GpuOpcode parity');
  assert.equal('COPY_TEXTURE' in V2_OPCODES, false, 'must stay outside V2_OPCODES for enum parity');

  const abi = readFileSync(new URL('../runtime/abi/playgta6_abi.h', import.meta.url), 'utf8');
  assert.match(abi, /kGpuOpcodeCopyTexture\s*=\s*static_cast<GpuOpcode>\(62\)/);
  assert.match(abi, /static_assert\(sizeof\(GpuCopyTexturePayload\)\s*==\s*56/);
});

test('CreateTexture and CreateTexture2 always include COPY_SRC across processor, RenderStatePackets, and TexturePackets', () => {
  // 1) Raw WgpuCommandProcessor (v1 CreateTexture + v2 CreateTexture2)
  {
    const mock = createMockWebGpu();
    const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
    const stream = new StreamBuilder()
      .createTexture(1, 16, 16, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING)
      .createTexture2({
        id: 2,
        width: 32,
        height: 32,
        format: 'rgba8unorm',
        usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
      })
      .finish();
    proc.executeCommandBuffer(stream);
    assert.equal(mock.createdTextures.length, 2);
    assert.notEqual(mock.createdTextures[0].usage & TEXTURE_USAGE.COPY_SRC, 0);
    assert.notEqual(mock.createdTextures[1].usage & TEXTURE_USAGE.COPY_SRC, 0);
  }

  // 2) RenderStatePackets installed
  {
    const mock = createMockWebGpu();
    const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
    installRenderStatePackets(proc);
    const stream = new StreamBuilder()
      .createTexture2({
        id: 3,
        width: 64,
        height: 64,
        format: 'rgba8unorm',
        usage: TEXTURE_USAGE.TEXTURE_BINDING,
      })
      .finish();
    proc.executeCommandBuffer(stream);
    assert.equal(mock.createdTextures.length, 1);
    assert.notEqual(mock.createdTextures[0].usage & TEXTURE_USAGE.COPY_SRC, 0);
  }

  // 3) TexturePackets installed in front of RenderStatePackets
  {
    const mock = createMockWebGpu();
    const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
    installRenderStatePackets(proc);
    installTexturePackets(proc);
    const stream = new StreamBuilder()
      .createTexture(4, 16, 16, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING)
      .createTexture2({
        id: 5,
        width: 16,
        height: 16,
        format: 'rgba8unorm',
        usage: TEXTURE_USAGE.COPY_DST,
      })
      .finish();
    proc.executeCommandBuffer(stream);
    assert.equal(mock.createdTextures.length, 2);
    assert.notEqual(mock.createdTextures[0].usage & TEXTURE_USAGE.COPY_SRC, 0);
    assert.notEqual(mock.createdTextures[1].usage & TEXTURE_USAGE.COPY_SRC, 0);
  }
});

test('CopyTexture encodes 2D sub-region copy including sRGB <-> linear copy-compatible formats', () => {
  const mock = createMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  const rs = installRenderStatePackets(proc);

  const stream = new StreamBuilder()
    .createTexture2({
      id: 10,
      width: 64,
      height: 32,
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    })
    .createTexture2({
      id: 11,
      width: 64,
      height: 32,
      format: 'rgba8unorm-srgb',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    })
    .copyTexture({
      dstId: 11,
      srcId: 10,
      dstX: 8,
      dstY: 4,
      srcX: 2,
      srcY: 6,
      width: 16,
      height: 12,
      depthOrLayers: 1,
    })
    .finish();

  proc.executeCommandBuffer(stream);

  assert.equal(proc.getValidationErrors().length, 0);
  assert.equal(rs.stats.errors, 0);
  assert.equal(rs.telemetry.textureCopies, 1);
  assert.equal(proc.getStats().copyTextureStats.copied, 1);
  assert.equal(mock.copyTextureToTextures.length, 1);
  const c = mock.copyTextureToTextures[0];
  assert.equal(c.src.texture, proc.textures.get(10));
  assert.equal(c.src.mipLevel, 0);
  assert.deepEqual(c.src.origin, { x: 2, y: 6, z: 0 });
  assert.equal(c.dst.texture, proc.textures.get(11));
  assert.equal(c.dst.mipLevel, 0);
  assert.deepEqual(c.dst.origin, { x: 8, y: 4, z: 0 });
  assert.deepEqual(c.copySize, { width: 16, height: 12, depthOrArrayLayers: 1 });
});

test('CopyTexture encodes 2D array layer copy (dst_z / src_z) and 3D depth slice copy', () => {
  const mock = createMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  installRenderStatePackets(proc);
  installTexturePackets(proc);

  const stream = new StreamBuilder()
    // 2D array textures (6 layers)
    .createTexture2({
      id: 20,
      width: 32,
      height: 32,
      depthOrLayers: 6,
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
      dimension: TEXTURE_DIMENSION.DIM_2D,
    })
    .createTexture2({
      id: 21,
      width: 32,
      height: 32,
      depthOrLayers: 6,
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
      dimension: TEXTURE_DIMENSION.DIM_2D,
    })
    .copyTexture({
      dstId: 21,
      srcId: 20,
      dstX: 0,
      dstY: 0,
      dstZ: 3,
      srcX: 4,
      srcY: 4,
      srcZ: 1,
      width: 16,
      height: 16,
      depthOrLayers: 2,
    })
    // 3D textures (16x16x8, 2 mips -> mip 1 is 8x8x4)
    .createTexture2({
      id: 30,
      width: 16,
      height: 16,
      depthOrLayers: 8,
      mipLevels: 2,
      format: 'rgba16float',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
      dimension: TEXTURE_DIMENSION.DIM_3D,
    })
    .createTexture2({
      id: 31,
      width: 16,
      height: 16,
      depthOrLayers: 8,
      mipLevels: 2,
      format: 'rgba16float',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
      dimension: TEXTURE_DIMENSION.DIM_3D,
    })
    .copyTexture({
      dstId: 31,
      srcId: 30,
      dstMip: 1,
      srcMip: 1,
      dstX: 0,
      dstY: 0,
      dstZ: 1,
      srcX: 2,
      srcY: 2,
      srcZ: 2,
      width: 4,
      height: 4,
      depthOrLayers: 2,
    })
    .finish();

  proc.executeCommandBuffer(stream);

  assert.equal(proc.getValidationErrors().length, 0);
  assert.equal(mock.copyTextureToTextures.length, 2);

  // 2D array layer copy
  const c2d = mock.copyTextureToTextures[0];
  assert.equal(c2d.src.texture, proc.textures.get(20));
  assert.deepEqual(c2d.src.origin, { x: 4, y: 4, z: 1 });
  assert.equal(c2d.dst.texture, proc.textures.get(21));
  assert.deepEqual(c2d.dst.origin, { x: 0, y: 0, z: 3 });
  assert.deepEqual(c2d.copySize, { width: 16, height: 16, depthOrArrayLayers: 2 });

  // 3D depth slice copy at mip 1
  const c3d = mock.copyTextureToTextures[1];
  assert.equal(c3d.src.texture, proc.textures.get(30));
  assert.equal(c3d.src.mipLevel, 1);
  assert.deepEqual(c3d.src.origin, { x: 2, y: 2, z: 2 });
  assert.equal(c3d.dst.texture, proc.textures.get(31));
  assert.equal(c3d.dst.mipLevel, 1);
  assert.deepEqual(c3d.dst.origin, { x: 0, y: 0, z: 1 });
  assert.deepEqual(c3d.copySize, { width: 4, height: 4, depthOrArrayLayers: 2 });
});

test('CopyTexture encodes mip-to-mip copy and BC7 block-aligned tail mip copy', () => {
  const mock = createMockWebGpu({ features: ['texture-compression-bc'] });
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  installRenderStatePackets(proc);

  const stream = new StreamBuilder()
    .createTexture2({
      id: 40,
      width: 64,
      height: 32,
      mipLevels: 4,
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    })
    .createTexture2({
      id: 41,
      width: 32,
      height: 16,
      mipLevels: 3,
      format: 'rgba8unorm',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    })
    // src mip 2 (16x8) -> dst mip 1 (16x8)
    .copyTexture({
      dstId: 41,
      srcId: 40,
      dstMip: 1,
      srcMip: 2,
      width: 16,
      height: 8,
      depthOrLayers: 1,
    })
    // BC7 8x8 with 4 mips -> mip 3 logical 1x1, physical 4x4
    .createTexture2({
      id: 42,
      width: 8,
      height: 8,
      mipLevels: 4,
      format: 'bc7-rgba-unorm',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    })
    .createTexture2({
      id: 43,
      width: 8,
      height: 8,
      mipLevels: 4,
      format: 'bc7-rgba-unorm-srgb',
      usage: TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST,
    })
    .copyTexture({
      dstId: 43,
      srcId: 42,
      dstMip: 3,
      srcMip: 3,
      width: 4,
      height: 4,
      depthOrLayers: 1,
    })
    .finish();

  proc.executeCommandBuffer(stream);

  assert.equal(proc.getValidationErrors().length, 0);
  assert.equal(mock.copyTextureToTextures.length, 2);
  assert.equal(mock.copyTextureToTextures[0].src.mipLevel, 2);
  assert.equal(mock.copyTextureToTextures[0].dst.mipLevel, 1);
  assert.deepEqual(mock.copyTextureToTextures[0].copySize, { width: 16, height: 8, depthOrArrayLayers: 1 });
  assert.equal(mock.copyTextureToTextures[1].src.mipLevel, 3);
  assert.equal(mock.copyTextureToTextures[1].dst.mipLevel, 3);
  assert.deepEqual(mock.copyTextureToTextures[1].copySize, { width: 4, height: 4, depthOrArrayLayers: 1 });
});

test('CopyTexture ends active render/compute pass and flushes staged uploads in stream order', () => {
  // 1) Active render pass ended before copyTextureToTexture on frame encoder
  {
    const mock = createMockWebGpu();
    const proc = new WgpuCommandProcessor(mock.device, mock.ctx);

    const stream = new StreamBuilder()
      .createTexture(50, 16, 16, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.COPY_DST)
      .createTexture(51, 16, 16, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST)
      .createShaderModule(1, SHADER_STAGE.VERTEX, '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }')
      .createShaderModule(2, SHADER_STAGE.FRAGMENT, '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }')
      .createRenderPipeline(100, 1, 2)
      .beginFrame()
      .setPipeline(100)
      .draw(3)
      // Active render pass is still open here!
      .copyTexture({ dstId: 51, srcId: 50, width: 16, height: 16, depthOrLayers: 1 })
      .present()
      .finish();

    proc.executeCommandBuffer(stream);

    assert.equal(proc.getValidationErrors().length, 0);
    assert.equal(mock.copyTextureToTextures.length, 1);
    const submitOps = mock.submits[mock.submits.length - 1][0].ops.map((o) => o.op);
    const drawIdx = submitOps.indexOf('render.draw');
    const endIdx = submitOps.indexOf('renderPass.end');
    const copyIdx = submitOps.indexOf('copyTextureToTexture');
    assert.ok(drawIdx >= 0 && endIdx > drawIdx && copyIdx > endIdx, `expected draw -> endPass -> copy, got ${submitOps}`);
  }

  // 2) Active compute pass ended before copyTextureToTexture
  {
    const mock = createMockWebGpu();
    const proc = new WgpuCommandProcessor(mock.device, mock.ctx);

    const stream = new StreamBuilder()
      .createTexture(52, 16, 16, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.COPY_DST)
      .createTexture(53, 16, 16, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST)
      .createShaderModule(3, SHADER_STAGE.COMPUTE, '@compute @workgroup_size(1) fn cs_main() {}')
      .createComputePipeline(101, 3)
      .beginFrame()
      .setComputePipeline(101)
      .dispatchCompute(2, 2, 1)
      // Active compute pass is still open here!
      .copyTexture({ dstId: 53, srcId: 52, width: 8, height: 8, depthOrLayers: 1 })
      .present()
      .finish();

    proc.executeCommandBuffer(stream);

    assert.equal(proc.getValidationErrors().length, 0);
    assert.equal(mock.copyTextureToTextures.length, 1);
    const submitOps = mock.submits[mock.submits.length - 1][0].ops.map((o) => o.op);
    const dispatchIdx = submitOps.indexOf('compute.dispatch');
    const endIdx = submitOps.indexOf('computePass.end');
    const copyIdx = submitOps.indexOf('copyTextureToTexture');
    assert.ok(
      dispatchIdx >= 0 && endIdx > dispatchIdx && copyIdx > endIdx,
      `expected dispatch -> computePass.end -> copy, got ${submitOps}`
    );
  }

  // 3) Staged WriteTexture upload via stagingPool is flushed BEFORE copyTextureToTexture
  {
    const mock = createMockWebGpu();
    let prepareSubmitCalls = 0;
    const fakeStagingPool = {
      _destroyed: false,
      stageTextureUpload(encoder, texture, data, layout) {
        encoder.copyBufferToTexture(
          { buffer: { kind: 'GPUBuffer' }, offset: 0, bytesPerRow: layout.bytesPerRow || 256 },
          { texture, mipLevel: layout.mipLevel || 0, origin: { x: 0, y: 0, z: 0 } },
          { width: layout.width, height: layout.height, depthOrArrayLayers: layout.depthOrArrayLayers || 1 }
        );
      },
      prepareSubmit() {
        prepareSubmitCalls += 1;
      },
    };
    const proc = new WgpuCommandProcessor(mock.device, mock.ctx, { stagingPool: fakeStagingPool });

    const pixels = new Uint8Array(256 * 4).fill(0xab);
    const stream = new StreamBuilder()
      .createTexture(60, 4, 4, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST)
      .createTexture(61, 4, 4, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST)
      .writeTexture({ textureId: 60, width: 4, height: 4, bytesPerRow: 256, rowsPerImage: 4, bytes: pixels })
      .copyTexture({ dstId: 61, srcId: 60, width: 4, height: 4, depthOrLayers: 1 })
      .finish();

    proc.executeCommandBuffer(stream);

    assert.equal(proc.getValidationErrors().length, 0);
    assert.ok(prepareSubmitCalls >= 1, 'stagingPool.prepareSubmit should be called when flushing staged uploads');
    assert.equal(mock.submits.length, 2, 'staged upload submit followed by copyTextureToTexture submit');
    assert.deepEqual(mock.submits[0][0].ops.map((o) => o.op), ['copyBufferToTexture']);
    assert.deepEqual(mock.submits[1][0].ops.map((o) => o.op), ['copyTextureToTexture']);
  }
});

test('CopyTexture rejects missing textures, short payloads, invalid bounds, misaligned BC regions, and partial depth/stencil copies; accepts zero-size no-ops and whole depth copies', () => {
  const mock = createMockWebGpu({ features: ['texture-compression-bc'] });
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);
  const rs = installRenderStatePackets(proc);

  const stream = new StreamBuilder()
    .createTexture2({ id: 70, width: 8, height: 8, mipLevels: 2, format: 'rgba8unorm', usage: TEXTURE_USAGE.COPY_DST })
    .createTexture2({ id: 71, width: 8, height: 8, mipLevels: 2, format: 'rgba8unorm', usage: TEXTURE_USAGE.COPY_DST })
    .createTexture2({ id: 72, width: 8, height: 8, format: 'r8unorm', usage: TEXTURE_USAGE.COPY_DST })
    .createTexture2({ id: 73, width: 8, height: 8, depthOrLayers: 4, format: 'rgba8unorm', dimension: TEXTURE_DIMENSION.DIM_3D, usage: TEXTURE_USAGE.COPY_DST })
    .createTexture2({ id: 74, width: 8, height: 8, format: 'depth32float', usage: TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.COPY_DST })
    .createTexture2({ id: 75, width: 8, height: 8, format: 'depth32float', usage: TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.COPY_DST })
    .createTexture2({ id: 76, width: 8, height: 8, format: 'bc7-rgba-unorm', usage: TEXTURE_USAGE.COPY_DST })
    .createTexture2({ id: 77, width: 8, height: 8, format: 'bc7-rgba-unorm', usage: TEXTURE_USAGE.COPY_DST })
    // 1) Truncated payload (12B < 56B)
    .packet(OPCODE_COPY_TEXTURE, new Uint8Array(12))
    // 2) Missing source texture
    .copyTexture({ dstId: 71, srcId: 9999, width: 4, height: 4 })
    // 3) Missing destination texture
    .copyTexture({ dstId: 9998, srcId: 70, width: 4, height: 4 })
    // 4) Incompatible formats (rgba8unorm vs r8unorm)
    .copyTexture({ dstId: 71, srcId: 72, width: 4, height: 4 })
    // 5) Dimension mismatch (2D vs 3D)
    .copyTexture({ dstId: 71, srcId: 73, width: 4, height: 4 })
    // 6) Mip out of range
    .copyTexture({ dstId: 71, srcId: 70, dstMip: 2, srcMip: 0, width: 2, height: 2 })
    // 7) Region out of bounds (dstX 6 + width 4 = 10 > 8)
    .copyTexture({ dstId: 71, srcId: 70, dstX: 6, dstY: 0, width: 4, height: 4 })
    // 8) Compressed region not block-aligned (srcX = 2)
    .copyTexture({ dstId: 77, srcId: 76, srcX: 2, srcY: 0, width: 4, height: 4 })
    // 9) Partial depth/stencil copy rejected
    .copyTexture({ dstId: 75, srcId: 74, width: 4, height: 4 })
    // 10) Zero-size no-op (valid, emits no copy and no error)
    .copyTexture({ dstId: 71, srcId: 70, width: 0, height: 4, depthOrLayers: 1 })
    // 11) Whole depth/stencil copy accepted
    .copyTexture({ dstId: 75, srcId: 74, width: 8, height: 8, depthOrLayers: 1 })
    .finish();

  proc.executeCommandBuffer(stream);

  // Only the whole depth copy (#11) should have been encoded.
  assert.equal(mock.copyTextureToTextures.length, 1);
  assert.equal(mock.copyTextureToTextures[0].src.texture, proc.textures.get(74));
  assert.equal(mock.copyTextureToTextures[0].dst.texture, proc.textures.get(75));

  const stats = proc.getStats().copyTextureStats;
  assert.equal(stats.copied, 1);
  assert.equal(stats.noop, 1);
  assert.equal(stats.missing, 2);
  assert.equal(stats.invalid, 6);
  assert.equal(rs.telemetry.textureCopies, 1);
  assert.equal(rs.telemetry.textureCopiesMissing, 2);
  // 1 truncated + 2 missing + 6 invalid = 9 validation errors
  assert.equal(proc.getValidationErrors().length, 9);
});

test('CopyTexture survives coalesced intermediate Present', () => {
  const mock = createMockWebGpu();
  const proc = new WgpuCommandProcessor(mock.device, mock.ctx);

  const stream = new StreamBuilder()
    .createTexture(80, 8, 8, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST)
    .createTexture(81, 8, 8, 1, 1, TEXTURE_FORMAT.RGBA8_UNORM, TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_DST)
    .createShaderModule(10, SHADER_STAGE.COMPUTE, '@compute @workgroup_size(1) fn cs_main() {}')
    .createComputePipeline(110, 10)
    // Frame 1 (will be coalesced because Frame 2 Present is in the same batch)
    .beginFrame()
    .setComputePipeline(110)
    .dispatchCompute(1, 1, 1)
    .copyTexture({ dstId: 81, srcId: 80, width: 8, height: 8, depthOrLayers: 1 })
    .present()
    // Frame 2 (final Present)
    .beginFrame()
    .present()
    .finish();

  proc.executeCommandBuffer(stream);

  assert.equal(proc.getValidationErrors().length, 0);
  assert.equal(mock.copyTextureToTextures.length, 1);
  assert.equal(mock.copyTextureToTextures[0].src.texture, proc.textures.get(80));
  assert.equal(mock.copyTextureToTextures[0].dst.texture, proc.textures.get(81));
});
