import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { WgpuCommandProcessor, OPCODES } = require('../web/workers/wgpu_worker.js');
const { installRenderStatePackets, TEXTURE_FORMAT_NAMES: RS_FORMATS } = require('../web/workers/render_state_packets.js');
const tp = require('../web/workers/texture_packets.js');

const repoFile = (rel) => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');

function createMockWebGpu() {
  const calls = [];
  let nextHandle = 1;
  const device = {
    createTexture(desc) {
      if (desc.format === 'bc7-rgba-unorm' && desc.sampleCount > 1) throw new Error('not multisampleable');
      const texture = {
        kind: 'GPUTexture',
        handle: nextHandle++,
        desc,
        destroyed: false,
        createView(viewDesc) {
          return { kind: 'GPUTextureView', textureHandle: texture.handle, viewDesc };
        },
        destroy() {
          texture.destroyed = true;
        },
      };
      calls.push({ op: 'createTexture', desc, result: texture });
      return texture;
    },
    createCommandEncoder() {
      return {
        beginRenderPass: () => ({ end() {}, setPipeline() {}, draw() {} }),
        beginComputePass: () => ({ end() {} }),
        finish: () => ({}),
      };
    },
    queue: { writeBuffer() {}, writeTexture() {}, submit() {} },
  };
  const ctx = { getCurrentTexture: () => ({ createView: () => ({ kind: 'swapchain' }) }) };
  return { device, ctx, calls };
}

function createTexture2Packet({ id, width, height, layers = 1, mips = 1, samples = 1, usage = 0x14, dimension = 1, format, viewFormats = [] }) {
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
  processor.executeCommandBuffer(createTexture2Packet({ id: 7, width: 640, height: 360, samples: 4, usage: 0x10, format: 40 }));
  processor.executeCommandBuffer(
    createTexture2Packet({ id: 8, width: 256, height: 128, layers: 6, mips: 9, usage: 0x06, format: 27, viewFormats: [28] })
  );
  assert.deepEqual(handler.stats, { created: 2, errors: 0 });
  const [rt, tex] = gpu.calls.map((c) => c.desc);
  assert.deepEqual(rt, {
    size: { width: 640, height: 360, depthOrArrayLayers: 1 },
    mipLevelCount: 1,
    sampleCount: 4,
    dimension: '2d',
    format: 'rgba16float',
    usage: 0x10,
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
  processor.executeCommandBuffer(destroyPacket(7));
  assert.equal(processor.textures.has(7), false);
});

test('invalid CreateTexture2 payloads are counted and leave the id unbound', () => {
  const gpu = createMockWebGpu();
  const processor = new WgpuCommandProcessor(gpu.device, gpu.ctx);
  const handler = tp.installTexturePackets(processor);
  processor.executeCommandBuffer(createTexture2Packet({ id: 1, width: 4, height: 4, format: 0 }));
  processor.executeCommandBuffer(createTexture2Packet({ id: 2, width: 4, height: 4, format: 200 }));
  processor.executeCommandBuffer(createTexture2Packet({ id: 3, width: 4, height: 4, format: 22, dimension: 9 }));
  processor.executeCommandBuffer(createTexture2Packet({ id: 4, width: 4, height: 4, format: 62, samples: 4 }));
  assert.equal(handler.stats.errors, 4);
  assert.match(handler.lastError, /not multisampleable/);
  for (const id of [1, 2, 3, 4]) assert.equal(processor.textures.has(id), false);
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
