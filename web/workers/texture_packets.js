'use strict';

// GPU packet ABI v2 CreateTexture2 (opcode 35, docs/gpu_commands.md "CreateTexture2") for
// WgpuCommandProcessor: textures with any v2 texture format code, a sample count and view formats,
// as created by the d3d12_xs layer (runtime/gpu). The texture lands in the processor's own tables
// (textures, textureViews, resource registry) exactly like a v1 CreateTexture, so WriteTexture,
// DestroyResource, bind groups and render pass attachments work on it unchanged.
//
// Install with installTexturePackets(processor) after the v2 handler (render_state_packets.js): it
// chains in front of processor.extension and forwards every other opcode to it.

const CREATE_TEXTURE2 = 35;
const CREATE_TEXTURE2_BYTES = 40;
const MAX_VIEW_FORMATS = 8;

// v2 texture format codes ("WebGPU enum codes" in docs/gpu_commands.md).
const TEXTURE_FORMAT_NAMES = Object.freeze([
  'undefined', 'r8unorm', 'r8snorm', 'r8uint', 'r8sint', 'r16unorm', 'r16snorm', 'r16uint',
  'r16sint', 'r16float', 'rg8unorm', 'rg8snorm', 'rg8uint', 'rg8sint', 'r32uint', 'r32sint',
  'r32float', 'rg16unorm', 'rg16snorm', 'rg16uint', 'rg16sint', 'rg16float', 'rgba8unorm',
  'rgba8unorm-srgb', 'rgba8snorm', 'rgba8uint', 'rgba8sint', 'bgra8unorm', 'bgra8unorm-srgb',
  'rgb9e5ufloat', 'rgb10a2uint', 'rgb10a2unorm', 'rg11b10ufloat', 'rg32uint', 'rg32sint',
  'rg32float', 'rgba16unorm', 'rgba16snorm', 'rgba16uint', 'rgba16sint', 'rgba16float',
  'rgba32uint', 'rgba32sint', 'rgba32float', 'stencil8', 'depth16unorm', 'depth24plus',
  'depth24plus-stencil8', 'depth32float', 'depth32float-stencil8', 'bc1-rgba-unorm',
  'bc1-rgba-unorm-srgb', 'bc2-rgba-unorm', 'bc2-rgba-unorm-srgb', 'bc3-rgba-unorm',
  'bc3-rgba-unorm-srgb', 'bc4-r-unorm', 'bc4-r-snorm', 'bc5-rg-unorm', 'bc5-rg-snorm',
  'bc6h-rgb-ufloat', 'bc6h-rgb-float', 'bc7-rgba-unorm', 'bc7-rgba-unorm-srgb',
]);
const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);

// Decodes a CreateTexture2 payload into { id, descriptor } (descriptor = GPUTextureDescriptor), or
// { id, error } when it cannot be honoured.
function decodeCreateTexture2(view, offset) {
  const id = view.getUint32(offset, true);
  const width = view.getUint32(offset + 4, true);
  const height = view.getUint32(offset + 8, true);
  const depthOrArrayLayers = Math.max(1, view.getUint32(offset + 12, true));
  const mipLevelCount = Math.max(1, view.getUint32(offset + 16, true));
  const sampleCount = Math.max(1, view.getUint32(offset + 20, true));
  const usage = view.getUint32(offset + 24, true);
  const dimensionCode = view.getUint8(offset + 28);
  const formatCode = view.getUint8(offset + 29);
  const viewFormatCount = view.getUint8(offset + 30);
  const format = TEXTURE_FORMAT_NAMES[formatCode];
  if (formatCode === 0 || format === undefined) {
    return { id, error: `CreateTexture2 ${id}: invalid texture format code ${formatCode}` };
  }
  const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode];
  if (dimension === undefined) {
    return { id, error: `CreateTexture2 ${id}: invalid dimension ${dimensionCode}` };
  }
  if (viewFormatCount > MAX_VIEW_FORMATS) {
    return { id, error: `CreateTexture2 ${id}: view_format_count ${viewFormatCount} > ${MAX_VIEW_FORMATS}` };
  }
  const viewFormats = [];
  for (let i = 0; i < viewFormatCount; i += 1) {
    const code = view.getUint8(offset + 32 + i);
    const name = TEXTURE_FORMAT_NAMES[code];
    if (code === 0 || name === undefined) {
      return { id, error: `CreateTexture2 ${id}: invalid view format code ${code}` };
    }
    viewFormats.push(name);
  }
  const descriptor = {
    size: { width, height, depthOrArrayLayers },
    mipLevelCount,
    sampleCount,
    dimension,
    format,
    usage,
  };
  if (viewFormats.length > 0) descriptor.viewFormats = viewFormats;
  return { id, descriptor };
}

class TexturePackets {
  constructor(processor, next) {
    this.processor = processor;
    this.next = next || null;
    this.stats = { created: 0, errors: 0 };
    this.lastError = null;
  }

  handlePacket(opcode, view, payloadOffset, payloadBytes) {
    if (opcode !== CREATE_TEXTURE2) {
      return this.next ? this.next.handlePacket(opcode, view, payloadOffset, payloadBytes) : false;
    }
    if (payloadBytes < CREATE_TEXTURE2_BYTES) {
      this._error(`CreateTexture2: payload ${payloadBytes} < ${CREATE_TEXTURE2_BYTES} bytes`);
      return true;
    }
    const decoded = decodeCreateTexture2(view, payloadOffset);
    if (decoded.error) {
      this._error(decoded.error);
      return true;
    }
    this._create(decoded.id, decoded.descriptor);
    return true;
  }

  destroyResource(id) {
    if (this.next && typeof this.next.destroyResource === 'function') this.next.destroyResource(id);
  }

  _create(id, descriptor) {
    const p = this.processor;
    if (typeof p._destroyTextureIfPresent === 'function') p._destroyTextureIfPresent(id);
    // Cached v2 views of a previous texture with this id must not survive.
    if (this.next && typeof this.next.destroyResource === 'function') this.next.destroyResource(id);
    let texture;
    try {
      texture = p.device.createTexture(descriptor);
    } catch (err) {
      this._error(`CreateTexture2 ${id}: ${String(err && err.message ? err.message : err)}`);
      return;
    }
    p.textures.set(id, texture);
    let textureView = null;
    if (texture && typeof texture.createView === 'function') {
      textureView = texture.createView();
      p.textureViews.set(id, textureView);
    }
    if (p.resourceRegistry && typeof p.resourceRegistry.register === 'function') {
      p.resourceRegistry.register('texture', id, texture, { byteSize: 0, label: `texture:${id}` });
      if (textureView) {
        p.resourceRegistry.register('textureView', id, textureView, { byteSize: 0, label: `textureView:${id}` });
      }
    }
    this.stats.created += 1;
  }

  _error(message) {
    this.stats.errors += 1;
    this.lastError = message;
  }
}

function installTexturePackets(processor) {
  const handler = new TexturePackets(processor, processor.extension || null);
  processor.extension = handler;
  return handler;
}

const exportedApi = {
  TexturePackets,
  installTexturePackets,
  decodeCreateTexture2,
  CREATE_TEXTURE2,
  CREATE_TEXTURE2_BYTES,
  TEXTURE_FORMAT_NAMES,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6TexturePackets = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
