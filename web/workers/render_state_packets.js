// GPU packet ABI v2 (opcodes 22..34, docs/gpu_commands.md "Opcodes (v2)") for WgpuCommandProcessor:
// explicit bind group / pipeline layouts, pipelines built from the 376-byte render pipeline key,
// explicit render passes, dynamic render state, bind groups with texture view descriptors and
// dynamic offsets, and comparison samplers.
//
// Install with installRenderStatePackets(processor). Objects land in the processor's own tables
// (pipelines, computePipelines, samplers, bindGroups) so v1 SetPipeline / SetComputePipeline /
// SetBindGroup / DestroyResource keep working on them. Pass-level packets are queued in the
// processor's per-frame command list and replayed at Present through the `run(passControl)` hook.

(function (globalScope) {
  'use strict';

// "WebGPU enum codes" (docs/gpu_commands.md): numbering of the pso::Wgpu* enums and the *_name()
// tables in runtime/gpu/pso_cache.{h,cpp}. tests/test_render_state_packets.mjs cross-checks them.
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
const VERTEX_FORMAT_NAMES = Object.freeze([
  'undefined', 'uint8', 'uint8x2', 'uint8x4', 'sint8', 'sint8x2', 'sint8x4', 'unorm8', 'unorm8x2',
  'unorm8x4', 'snorm8', 'snorm8x2', 'snorm8x4', 'uint16', 'uint16x2', 'uint16x4', 'sint16',
  'sint16x2', 'sint16x4', 'unorm16', 'unorm16x2', 'unorm16x4', 'snorm16', 'snorm16x2', 'snorm16x4',
  'float16', 'float16x2', 'float16x4', 'float32', 'float32x2', 'float32x3', 'float32x4', 'uint32',
  'uint32x2', 'uint32x3', 'uint32x4', 'sint32', 'sint32x2', 'sint32x3', 'sint32x4',
  'unorm10-10-10-2', 'unorm8x4-bgra',
]);
const BLEND_FACTOR_NAMES = Object.freeze([
  'zero', 'one', 'src', 'one-minus-src', 'src-alpha', 'one-minus-src-alpha', 'dst', 'one-minus-dst',
  'dst-alpha', 'one-minus-dst-alpha', 'src-alpha-saturated', 'constant', 'one-minus-constant',
  'src1', 'one-minus-src1', 'src1-alpha', 'one-minus-src1-alpha',
]);
const BLEND_OPERATION_NAMES = Object.freeze(['add', 'subtract', 'reverse-subtract', 'min', 'max']);
const COMPARE_FUNCTION_NAMES = Object.freeze([
  'undefined', 'never', 'less', 'equal', 'less-equal', 'greater', 'not-equal', 'greater-equal', 'always',
]);
const STENCIL_OPERATION_NAMES = Object.freeze([
  'keep', 'zero', 'replace', 'invert', 'increment-clamp', 'decrement-clamp', 'increment-wrap',
  'decrement-wrap',
]);
const CULL_MODE_NAMES = Object.freeze(['none', 'front', 'back']);
const FRONT_FACE_NAMES = Object.freeze(['ccw', 'cw']);
const PRIMITIVE_TOPOLOGY_NAMES = Object.freeze([
  'triangle-list', 'triangle-strip', 'line-list', 'line-strip', 'point-list',
]);
const INDEX_FORMAT_NAMES = Object.freeze(['undefined', 'uint16', 'uint32']);
const VERTEX_STEP_MODE_NAMES = Object.freeze(['unused', 'vertex', 'instance']);
const LOAD_OP_NAMES = Object.freeze(['undefined', 'load', 'clear']);
const STORE_OP_NAMES = Object.freeze(['undefined', 'store', 'discard']);
const TEXTURE_ASPECT_NAMES = Object.freeze(['all', 'depth-only', 'stencil-only']);

// Packet-level enums defined in runtime/abi/playgta6_abi.h.
const VIEW_DIMENSION_NAMES = Object.freeze([
  'undefined', '1d', '2d', '2d-array', 'cube', 'cube-array', '3d',
]);
const BINDING_LAYOUT_TYPE = Object.freeze({ BUFFER: 0, SAMPLER: 1, TEXTURE: 2, STORAGE_TEXTURE: 3 });
const BUFFER_BINDING_TYPE_NAMES = Object.freeze(['uniform', 'storage', 'read-only-storage']);
const SAMPLER_BINDING_TYPE_NAMES = Object.freeze(['filtering', 'non-filtering', 'comparison']);
const TEXTURE_SAMPLE_TYPE_NAMES = Object.freeze(['float', 'unfilterable-float', 'depth', 'sint', 'uint']);
const STORAGE_TEXTURE_ACCESS_NAMES = Object.freeze(['write-only', 'read-only', 'read-write']);
const ADDRESS_MODE_NAMES = Object.freeze(['clamp-to-edge', 'repeat', 'mirror-repeat']);
const FILTER_MODE_NAMES = Object.freeze(['nearest', 'linear']);
const BIND_RESOURCE_TYPE = Object.freeze({ BUFFER: 0, TEXTURE_VIEW: 1, SAMPLER: 2 });

const V2_OPCODES = Object.freeze({
  CREATE_BIND_GROUP_LAYOUT: 22,
  CREATE_PIPELINE_LAYOUT: 23,
  CREATE_RENDER_PIPELINE2: 24,
  CREATE_COMPUTE_PIPELINE2: 25,
  BEGIN_RENDER_PASS: 26,
  END_RENDER_PASS: 27,
  SET_VIEWPORT: 28,
  SET_SCISSOR_RECT: 29,
  SET_BLEND_CONSTANT: 30,
  SET_STENCIL_REFERENCE: 31,
  CREATE_BIND_GROUP2: 32,
  SET_BIND_GROUP2: 33,
  CREATE_SAMPLER2: 34,
});

const SWAPCHAIN_TEXTURE_ID = 0xffffffff;
const MANIFEST_REFERENCE_FLAG = 0x1;
const KEY_OFFSET = 16; // CreateRenderPipeline2: header, then the 376-byte key
const KEY_SIZE = 376;
const GPU_BUFFER_USAGE_UNIFORM_STORAGE_COPY_DST = 0x0040 | 0x0080 | 0x0008;
const GPU_TEXTURE_USAGE_TEXTURE_BINDING = 0x0004;
const GPU_TEXTURE_USAGE_STORAGE_BINDING = 0x0008;
const EMPTY_FRAGMENT_WGSL = '@fragment fn fs_main() {}\n';

function nameOf(table, code) {
  return code < table.length ? table[code] : undefined;
}

// ---- Decoders (pure; also used by the tests) ----

function decodeTextureViewDesc(view, offset) {
  const desc = {};
  const format = view.getUint8(offset);
  const dimension = view.getUint8(offset + 1);
  const aspect = view.getUint8(offset + 2);
  const baseMipLevel = view.getUint16(offset + 4, true);
  const mipLevelCount = view.getUint16(offset + 6, true);
  const baseArrayLayer = view.getUint32(offset + 8, true);
  const arrayLayerCount = view.getUint32(offset + 12, true);
  if (format !== 0) desc.format = nameOf(TEXTURE_FORMAT_NAMES, format);
  if (dimension !== 0) desc.dimension = nameOf(VIEW_DIMENSION_NAMES, dimension);
  if (aspect !== 0) desc.aspect = nameOf(TEXTURE_ASPECT_NAMES, aspect);
  if (baseMipLevel !== 0) desc.baseMipLevel = baseMipLevel;
  if (mipLevelCount !== 0) desc.mipLevelCount = mipLevelCount;
  if (baseArrayLayer !== 0) desc.baseArrayLayer = baseArrayLayer;
  if (arrayLayerCount !== 0) desc.arrayLayerCount = arrayLayerCount;
  return desc;
}

function decodeBindGroupLayoutEntry(view, offset) {
  const entry = {
    binding: view.getUint32(offset, true),
    visibility: view.getUint32(offset + 4, true),
  };
  const type = view.getUint8(offset + 8);
  const subtype = view.getUint8(offset + 9);
  const dimension = view.getUint8(offset + 10);
  const flags = view.getUint8(offset + 11);
  const storageFormat = view.getUint8(offset + 12);
  const viewDimension = dimension !== 0 ? nameOf(VIEW_DIMENSION_NAMES, dimension) : '2d';
  if (type === BINDING_LAYOUT_TYPE.BUFFER) {
    entry.buffer = {
      type: nameOf(BUFFER_BINDING_TYPE_NAMES, subtype),
      hasDynamicOffset: (flags & 0x1) !== 0,
      minBindingSize: Number(view.getBigUint64(offset + 16, true)),
    };
  } else if (type === BINDING_LAYOUT_TYPE.SAMPLER) {
    entry.sampler = { type: nameOf(SAMPLER_BINDING_TYPE_NAMES, subtype) };
  } else if (type === BINDING_LAYOUT_TYPE.TEXTURE) {
    entry.texture = {
      sampleType: nameOf(TEXTURE_SAMPLE_TYPE_NAMES, subtype),
      viewDimension,
      multisampled: (flags & 0x2) !== 0,
    };
  } else if (type === BINDING_LAYOUT_TYPE.STORAGE_TEXTURE) {
    entry.storageTexture = {
      access: nameOf(STORAGE_TEXTURE_ACCESS_NAMES, subtype),
      format: nameOf(TEXTURE_FORMAT_NAMES, storageFormat),
      viewDimension,
    };
  } else {
    return null;
  }
  return entry;
}

function decodeBlendComponent(view, offset) {
  return {
    operation: nameOf(BLEND_OPERATION_NAMES, view.getUint8(offset)),
    srcFactor: nameOf(BLEND_FACTOR_NAMES, view.getUint8(offset + 1)),
    dstFactor: nameOf(BLEND_FACTOR_NAMES, view.getUint8(offset + 2)),
  };
}

function decodeStencilFace(view, offset) {
  return {
    compare: nameOf(COMPARE_FUNCTION_NAMES, view.getUint8(offset)),
    failOp: nameOf(STENCIL_OPERATION_NAMES, view.getUint8(offset + 1)),
    depthFailOp: nameOf(STENCIL_OPERATION_NAMES, view.getUint8(offset + 2)),
    passOp: nameOf(STENCIL_OPERATION_NAMES, view.getUint8(offset + 3)),
  };
}

function f32FromBits(bits) {
  const scratch = new DataView(new ArrayBuffer(4));
  scratch.setUint32(0, bits, true);
  return scratch.getFloat32(0, true);
}

// Decodes a GpuRenderPipelineKey at `base` into the GPURenderPipelineDescriptor parts it determines.
// Returns { primitive, multisample, depthStencil?, targets, buffers, colorTargetCount }.
function decodeRenderPipelineKey(view, base) {
  const colorTargetCount = Math.min(view.getUint32(base + 20, true), 8);
  const targets = [];
  for (let i = 0; i < colorTargetCount; i++) {
    const o = base + 24 + 12 * i;
    const format = view.getUint8(o);
    if (format === 0) {
      targets.push(null);
      continue;
    }
    const target = { format: nameOf(TEXTURE_FORMAT_NAMES, format), writeMask: view.getUint8(o + 2) };
    if (view.getUint8(o + 1) !== 0) {
      target.blend = { color: decodeBlendComponent(view, o + 4), alpha: decodeBlendComponent(view, o + 8) };
    }
    targets.push(target);
  }

  const ds = base + 120;
  let depthStencil;
  const dsFormat = view.getUint8(ds);
  if (dsFormat !== 0) {
    depthStencil = {
      format: nameOf(TEXTURE_FORMAT_NAMES, dsFormat),
      depthWriteEnabled: view.getUint8(ds + 1) !== 0,
      stencilFront: decodeStencilFace(view, ds + 4),
      stencilBack: decodeStencilFace(view, ds + 8),
      stencilReadMask: view.getUint32(ds + 12, true),
      stencilWriteMask: view.getUint32(ds + 16, true),
      depthBias: view.getInt32(ds + 20, true),
      depthBiasSlopeScale: f32FromBits(view.getUint32(ds + 24, true)),
      depthBiasClamp: f32FromBits(view.getUint32(ds + 28, true)),
    };
    const depthCompare = view.getUint8(ds + 2);
    if (depthCompare !== 0) depthStencil.depthCompare = nameOf(COMPARE_FUNCTION_NAMES, depthCompare);
  }

  const pr = base + 152;
  const primitive = {
    topology: nameOf(PRIMITIVE_TOPOLOGY_NAMES, view.getUint8(pr)),
    frontFace: nameOf(FRONT_FACE_NAMES, view.getUint8(pr + 2)),
    cullMode: nameOf(CULL_MODE_NAMES, view.getUint8(pr + 3)),
  };
  const stripIndexFormat = view.getUint8(pr + 1);
  if (stripIndexFormat !== 0) primitive.stripIndexFormat = nameOf(INDEX_FORMAT_NAMES, stripIndexFormat);
  if (view.getUint8(pr + 4) !== 0) primitive.unclippedDepth = true;

  const ms = base + 160;
  const multisample = {
    count: view.getUint32(ms, true),
    mask: view.getUint32(ms + 4, true),
    alphaToCoverageEnabled: view.getUint8(ms + 8) !== 0,
  };

  const bufferCount = Math.min(view.getUint32(base + 172, true), 8);
  const attributeCount = Math.min(view.getUint32(base + 176, true), 16);
  const attributes = [];
  for (let i = 0; i < attributeCount; i++) {
    const o = base + 244 + 8 * i;
    attributes.push({
      bufferIndex: view.getUint8(o + 6),
      attribute: {
        format: nameOf(VERTEX_FORMAT_NAMES, view.getUint8(o + 4)),
        offset: view.getUint32(o, true),
        shaderLocation: view.getUint8(o + 5),
      },
    });
  }
  const buffers = [];
  for (let i = 0; i < bufferCount; i++) {
    const o = base + 180 + 8 * i;
    const stepMode = view.getUint8(o + 4);
    if (stepMode === 0) {
      buffers.push(null);
      continue;
    }
    buffers.push({
      arrayStride: view.getUint32(o, true),
      stepMode: nameOf(VERTEX_STEP_MODE_NAMES, stepMode),
      attributes: attributes.filter((a) => a.bufferIndex === i).map((a) => a.attribute),
    });
  }
  return { primitive, multisample, depthStencil, targets, buffers, colorTargetCount };
}

function decodeBeginRenderPass(view, offset) {
  const colorCount = Math.min(view.getUint32(offset, true), 8);
  const colors = [];
  for (let i = 0; i < colorCount; i++) {
    const o = offset + 8 + 40 * i;
    const textureId = view.getUint32(o, true);
    if (textureId === 0) {
      colors.push(null);
      continue;
    }
    colors.push({
      textureId,
      loadOp: nameOf(LOAD_OP_NAMES, view.getUint8(o + 4)),
      storeOp: nameOf(STORE_OP_NAMES, view.getUint8(o + 5)),
      view: decodeTextureViewDesc(view, o + 8),
      clearValue: {
        r: view.getFloat32(o + 24, true),
        g: view.getFloat32(o + 28, true),
        b: view.getFloat32(o + 32, true),
        a: view.getFloat32(o + 36, true),
      },
    });
  }
  const d = offset + 328;
  let depthStencil = null;
  const dsTexture = view.getUint32(d, true);
  if (dsTexture !== 0) {
    depthStencil = {
      textureId: dsTexture,
      depthLoadOp: view.getUint8(d + 4),
      depthStoreOp: view.getUint8(d + 5),
      stencilLoadOp: view.getUint8(d + 6),
      stencilStoreOp: view.getUint8(d + 7),
      view: decodeTextureViewDesc(view, d + 8),
      depthClearValue: view.getFloat32(d + 24, true),
      stencilClearValue: view.getUint32(d + 28, true),
      depthReadOnly: view.getUint8(d + 32) !== 0,
      stencilReadOnly: view.getUint8(d + 33) !== 0,
    };
  }
  return { colors, depthStencil };
}

// ---- Packet handler ----

class RenderStatePackets {
  constructor(processor, options = {}) {
    this.processor = processor;
    this.device = processor.device;
    // Optional: (kind 'bindGroupLayout' | 'pipelineLayout', contentId: bigint) => GPU object, for
    // ManifestReference layouts loaded from layout-manifest.json (e.g. by reflection_loader.js).
    this.resolveManifestLayout =
      typeof options.resolveManifestLayout === 'function' ? options.resolveManifestLayout : null;
    this.bindGroupLayouts = new Map(); // id -> { layout, entries }
    this.pipelineLayouts = new Map(); // id -> GPUPipelineLayout
    this.layoutsByContent = new Map(); // 'bgl:<hex>' / 'pl:<hex>' -> object
    this.textureViews = new Map(); // textureId -> Map(descKey -> GPUTextureView)
    this.dummies = new Map();
    this.emptyLayout = null;
    this.emptyFragment = null;
    this.stats = { packets: 0, errors: 0 };
    this.lastError = null;
    this._packetError = null;
  }

  // Returns true when the opcode belongs to ABI v2 (handled, possibly with an error recorded).
  handlePacket(opcode, view, offset, payloadBytes) {
    this._packetError = null;
    switch (opcode) {
      case V2_OPCODES.CREATE_BIND_GROUP_LAYOUT:
        return this._count(this._createBindGroupLayout(view, offset, payloadBytes));
      case V2_OPCODES.CREATE_PIPELINE_LAYOUT:
        return this._count(this._createPipelineLayout(view, offset, payloadBytes));
      case V2_OPCODES.CREATE_RENDER_PIPELINE2:
        return this._count(payloadBytes >= KEY_OFFSET + KEY_SIZE && this._createRenderPipeline(view, offset));
      case V2_OPCODES.CREATE_COMPUTE_PIPELINE2:
        return this._count(payloadBytes >= 32 && this._createComputePipeline(view, offset));
      case V2_OPCODES.BEGIN_RENDER_PASS:
        return this._count(payloadBytes >= 368 && this._beginRenderPass(view, offset));
      case V2_OPCODES.END_RENDER_PASS:
        this._queue((ctl) => {
          if (ctl.passKind === 'render') ctl.endPass();
        });
        return this._count(true);
      case V2_OPCODES.SET_VIEWPORT: {
        if (payloadBytes < 24) return this._count(false);
        const a = [0, 4, 8, 12, 16, 20].map((o) => view.getFloat32(offset + o, true));
        this._queueRender((pass) => pass.setViewport(a[0], a[1], a[2], a[3], a[4], a[5]));
        return this._count(true);
      }
      case V2_OPCODES.SET_SCISSOR_RECT: {
        if (payloadBytes < 16) return this._count(false);
        const r = [0, 4, 8, 12].map((o) => view.getUint32(offset + o, true));
        this._queueRender((pass) => pass.setScissorRect(r[0], r[1], r[2], r[3]));
        return this._count(true);
      }
      case V2_OPCODES.SET_BLEND_CONSTANT: {
        if (payloadBytes < 16) return this._count(false);
        const c = [0, 4, 8, 12].map((o) => view.getFloat32(offset + o, true));
        this._queueRender((pass) => pass.setBlendConstant({ r: c[0], g: c[1], b: c[2], a: c[3] }));
        return this._count(true);
      }
      case V2_OPCODES.SET_STENCIL_REFERENCE: {
        if (payloadBytes < 4) return this._count(false);
        const reference = view.getUint32(offset, true);
        this._queueRender((pass) => pass.setStencilReference(reference));
        return this._count(true);
      }
      case V2_OPCODES.CREATE_BIND_GROUP2:
        return this._count(this._createBindGroup(view, offset, payloadBytes));
      case V2_OPCODES.SET_BIND_GROUP2:
        return this._count(this._setBindGroup(view, offset, payloadBytes));
      case V2_OPCODES.CREATE_SAMPLER2:
        return this._count(payloadBytes >= 24 && this._createSampler(view, offset));
      default:
        return false;
    }
  }

  _count(ok) {
    this.stats.packets += 1;
    if (!ok) {
      this.stats.errors += 1;
      this.lastError = this._packetError || 'malformed or truncated v2 packet';
    }
    return true;
  }

  // DestroyResource hook: forget layouts and cached views of `id`.
  destroyResource(id) {
    this.bindGroupLayouts.delete(id);
    this.pipelineLayouts.delete(id);
    this.textureViews.delete(id);
  }

  _queue(run) {
    this.processor._passCommands.push({ type: 'renderState', run });
  }

  _queueRender(fn) {
    this._queue((ctl) => {
      if (ctl.passKind === 'render') fn(ctl.pass);
    });
  }

  // ---- Layouts ----

  _createBindGroupLayout(view, offset, payloadBytes) {
    if (payloadBytes < 24) return false;
    const id = view.getUint32(offset, true);
    const count = view.getUint32(offset + 4, true);
    const flags = view.getUint32(offset + 8, true);
    const contentId = view.getBigUint64(offset + 16, true);
    if ((flags & MANIFEST_REFERENCE_FLAG) !== 0) {
      const layout = this.resolveManifestLayout && this.resolveManifestLayout('bindGroupLayout', contentId);
      if (!layout) {
        this._packetError = `manifest bind group layout 0x${contentId.toString(16)} unavailable`;
        return false;
      }
      this.bindGroupLayouts.set(id, { layout, entries: null });
      return true;
    }
    if (24 + count * 24 > payloadBytes) return false;
    const entries = [];
    for (let i = 0; i < count; i++) {
      const entry = decodeBindGroupLayoutEntry(view, offset + 24 + i * 24);
      if (!entry) return false;
      entries.push(entry);
    }
    const contentKey = contentId !== 0n ? `bgl:${contentId.toString(16)}` : null;
    let layout = contentKey ? this.layoutsByContent.get(contentKey) : undefined;
    if (!layout) {
      layout = this.device.createBindGroupLayout({ entries });
      if (contentKey) this.layoutsByContent.set(contentKey, layout);
    }
    this.bindGroupLayouts.set(id, { layout, entries });
    return true;
  }

  _emptyBindGroupLayout() {
    if (!this.emptyLayout) this.emptyLayout = this.device.createBindGroupLayout({ entries: [] });
    return this.emptyLayout;
  }

  _createPipelineLayout(view, offset, payloadBytes) {
    if (payloadBytes < 24) return false;
    const id = view.getUint32(offset, true);
    const count = view.getUint32(offset + 4, true);
    const flags = view.getUint32(offset + 8, true);
    const contentId = view.getBigUint64(offset + 16, true);
    let layout;
    if ((flags & MANIFEST_REFERENCE_FLAG) !== 0) {
      layout = this.resolveManifestLayout && this.resolveManifestLayout('pipelineLayout', contentId);
      if (!layout) {
        this._packetError = `manifest pipeline layout 0x${contentId.toString(16)} unavailable`;
        return false;
      }
    } else {
      if (24 + count * 4 > payloadBytes) return false;
      const bindGroupLayouts = [];
      for (let g = 0; g < count; g++) {
        const bglId = view.getUint32(offset + 24 + g * 4, true);
        const record = bglId !== 0 ? this.bindGroupLayouts.get(bglId) : null;
        if (bglId !== 0 && !record) {
          this._packetError = `pipeline layout ${id}: unknown bind group layout ${bglId}`;
          return false;
        }
        bindGroupLayouts.push(record ? record.layout : this._emptyBindGroupLayout());
      }
      layout = this.device.createPipelineLayout({ bindGroupLayouts });
    }
    this.pipelineLayouts.set(id, layout);
    return true;
  }

  _layoutFor(layoutId) {
    if (layoutId === 0) return 'auto';
    const layout = this.pipelineLayouts.get(layoutId);
    if (!layout) this._packetError = `unknown pipeline layout ${layoutId}`;
    return layout || null;
  }

  // ---- Pipelines ----

  // The exact GPURenderPipelineDescriptor for a CreateRenderPipeline2 payload at `offset`.
  buildRenderPipelineDescriptor(view, offset) {
    const vsId = view.getUint32(offset + 4, true);
    const fsId = view.getUint32(offset + 8, true);
    const layout = this._layoutFor(view.getUint32(offset + 12, true));
    const vs = this.processor.shaderModules.get(vsId);
    if (!vs || !layout) {
      if (!vs) this._packetError = `unknown vertex shader module ${vsId}`;
      return null;
    }
    const key = decodeRenderPipelineKey(view, offset + KEY_OFFSET);
    const descriptor = {
      layout,
      vertex: { module: vs.module, entryPoint: 'vs_main', buffers: key.buffers },
      primitive: key.primitive,
      multisample: key.multisample,
    };
    if (key.depthStencil) descriptor.depthStencil = key.depthStencil;
    let fragmentModule = null;
    let targets = key.targets;
    if (fsId !== 0) {
      const fs = this.processor.shaderModules.get(fsId);
      if (!fs) {
        this._packetError = `unknown fragment shader module ${fsId}`;
        return null;
      }
      fragmentModule = fs.module;
    } else if (key.colorTargetCount > 0) {
      // No pixel shader but color targets: an empty fragment stage that writes nothing.
      if (!this.emptyFragment) this.emptyFragment = this.device.createShaderModule({ code: EMPTY_FRAGMENT_WGSL });
      fragmentModule = this.emptyFragment;
      targets = targets.map((t) => (t ? { format: t.format, writeMask: 0 } : null));
    }
    if (fragmentModule) {
      descriptor.fragment = { module: fragmentModule, entryPoint: 'fs_main', targets };
    }
    return descriptor;
  }

  _createRenderPipeline(view, offset) {
    const descriptor = this.buildRenderPipelineDescriptor(view, offset);
    if (!descriptor) return false;
    this.processor.pipelines.set(view.getUint32(offset, true), this.device.createRenderPipeline(descriptor));
    return true;
  }

  _createComputePipeline(view, offset) {
    const id = view.getUint32(offset, true);
    const csId = view.getUint32(offset + 4, true);
    const layout = this._layoutFor(view.getUint32(offset + 8, true));
    const cs = this.processor.shaderModules.get(csId);
    if (!cs || !layout) {
      if (!cs) this._packetError = `unknown compute shader module ${csId}`;
      return false;
    }
    this.processor.computePipelines.set(
      id,
      this.device.createComputePipeline({ layout, compute: { module: cs.module, entryPoint: 'cs_main' } })
    );
    return true;
  }

  // ---- Textures and views ----

  _textureView(textureId, desc) {
    const texture = this.processor.textures.get(textureId);
    if (!texture || typeof texture.createView !== 'function') return null;
    const key = JSON.stringify(desc);
    let views = this.textureViews.get(textureId);
    if (!views) {
      views = new Map();
      this.textureViews.set(textureId, views);
    }
    let textureView = views.get(key);
    if (!textureView) {
      textureView = texture.createView(desc);
      views.set(key, textureView);
    }
    return textureView;
  }

  // Attachment view resolver; the canvas texture only exists at Present (ctl.swapchainView).
  _attachmentView(textureId, desc) {
    if (textureId === SWAPCHAIN_TEXTURE_ID) return (ctl) => ctl.swapchainView;
    const textureView = this._textureView(textureId, desc);
    return textureView ? () => textureView : null;
  }

  // The exact GPURenderPassDescriptor for a decoded BeginRenderPass (views resolved via `ctl`).
  buildRenderPassDescriptor(decoded, resolvers, ctl) {
    const colorAttachments = decoded.colors.map((c, i) => {
      if (!c) return null;
      const attachment = { view: resolvers.colors[i](ctl), loadOp: c.loadOp, storeOp: c.storeOp };
      if (c.loadOp === 'clear') attachment.clearValue = c.clearValue;
      return attachment;
    });
    const descriptor = { colorAttachments };
    const d = decoded.depthStencil;
    if (d) {
      const ds = { view: resolvers.depthStencil(ctl) };
      if (d.depthLoadOp !== 0) {
        ds.depthLoadOp = nameOf(LOAD_OP_NAMES, d.depthLoadOp);
        ds.depthStoreOp = nameOf(STORE_OP_NAMES, d.depthStoreOp);
        if (ds.depthLoadOp === 'clear') ds.depthClearValue = d.depthClearValue;
      }
      if (d.stencilLoadOp !== 0) {
        ds.stencilLoadOp = nameOf(LOAD_OP_NAMES, d.stencilLoadOp);
        ds.stencilStoreOp = nameOf(STORE_OP_NAMES, d.stencilStoreOp);
        if (ds.stencilLoadOp === 'clear') ds.stencilClearValue = d.stencilClearValue;
      }
      if (d.depthReadOnly) ds.depthReadOnly = true;
      if (d.stencilReadOnly) ds.stencilReadOnly = true;
      descriptor.depthStencilAttachment = ds;
    }
    return descriptor;
  }

  _beginRenderPass(view, offset) {
    const decoded = decodeBeginRenderPass(view, offset);
    const resolvers = { colors: [], depthStencil: null };
    for (const c of decoded.colors) {
      const resolver = c ? this._attachmentView(c.textureId, c.view) : null;
      if (c && !resolver) {
        this._packetError = `render pass: unknown texture ${c.textureId}`;
        return false;
      }
      resolvers.colors.push(resolver);
    }
    if (decoded.depthStencil) {
      resolvers.depthStencil = this._attachmentView(decoded.depthStencil.textureId, decoded.depthStencil.view);
      if (!resolvers.depthStencil) {
        this._packetError = `render pass: unknown texture ${decoded.depthStencil.textureId}`;
        return false;
      }
    }
    this._queue((ctl) => ctl.beginRenderPass(this.buildRenderPassDescriptor(decoded, resolvers, ctl)));
    return true;
  }

  // ---- Bind groups ----

  _dummyResource(layoutEntry) {
    if (layoutEntry.buffer) {
      const size = Math.max(256, Math.ceil(layoutEntry.buffer.minBindingSize / 4) * 4);
      const key = `buffer:${size}`;
      if (!this.dummies.has(key)) {
        this.dummies.set(key, this.device.createBuffer({ size, usage: GPU_BUFFER_USAGE_UNIFORM_STORAGE_COPY_DST }));
      }
      return { buffer: this.dummies.get(key) };
    }
    if (layoutEntry.sampler) {
      const key = `sampler:${layoutEntry.sampler.type}`;
      if (!this.dummies.has(key)) {
        const desc = layoutEntry.sampler.type === 'comparison' ? { compare: 'always' } : {};
        this.dummies.set(key, this.device.createSampler(desc));
      }
      return this.dummies.get(key);
    }
    const tex = layoutEntry.texture || layoutEntry.storageTexture;
    const viewDimension = tex.viewDimension || '2d';
    let format = tex.format;
    if (!format) {
      format = { float: 'rgba8unorm', 'unfilterable-float': 'r32float', depth: 'depth32float', sint: 'rgba8sint', uint: 'rgba8uint' }[
        tex.sampleType
      ];
    }
    const multisampled = Boolean(layoutEntry.texture && layoutEntry.texture.multisampled);
    const key = `texture:${format}:${viewDimension}:${multisampled}:${Boolean(layoutEntry.storageTexture)}`;
    if (!this.dummies.has(key)) {
      const layers = viewDimension === 'cube' || viewDimension === 'cube-array' ? 6 : 1;
      const texture = this.device.createTexture({
        size: { width: 1, height: 1, depthOrArrayLayers: layers },
        format,
        usage: layoutEntry.storageTexture ? GPU_TEXTURE_USAGE_STORAGE_BINDING : GPU_TEXTURE_USAGE_TEXTURE_BINDING,
        dimension: viewDimension === '1d' ? '1d' : viewDimension === '3d' ? '3d' : '2d',
        sampleCount: multisampled ? 4 : 1,
      });
      this.dummies.set(key, texture.createView({ dimension: viewDimension }));
    }
    return this.dummies.get(key);
  }

  _createBindGroup(view, offset, payloadBytes) {
    if (payloadBytes < 16) return false;
    const id = view.getUint32(offset, true);
    const layoutId = view.getUint32(offset + 4, true);
    const count = view.getUint32(offset + 8, true);
    if (16 + count * 48 > payloadBytes) return false;
    const record = this.bindGroupLayouts.get(layoutId);
    if (!record) {
      this._packetError = `bind group ${id}: unknown bind group layout ${layoutId}`;
      return false;
    }
    const entries = [];
    for (let i = 0; i < count; i++) {
      const o = offset + 16 + i * 48;
      const binding = view.getUint32(o, true);
      const type = view.getUint32(o + 4, true);
      const resourceId = view.getUint32(o + 8, true);
      let resource = null;
      if (resourceId === 0) {
        const layoutEntry = record.entries && record.entries.find((e) => e.binding === binding);
        resource = layoutEntry ? this._dummyResource(layoutEntry) : null;
      } else if (type === BIND_RESOURCE_TYPE.BUFFER) {
        const buffer = this.processor.buffers.get(resourceId);
        if (buffer) {
          resource = { buffer, offset: Number(view.getBigUint64(o + 16, true)) };
          const size = Number(view.getBigUint64(o + 24, true));
          if (size > 0) resource.size = size;
        }
      } else if (type === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
        resource = this._textureView(resourceId, decodeTextureViewDesc(view, o + 32));
      } else if (type === BIND_RESOURCE_TYPE.SAMPLER) {
        resource = this.processor.samplers.get(resourceId) || null;
      }
      if (!resource) {
        this._packetError = `bind group ${id}: binding ${binding} resource ${resourceId} unavailable`;
        return false;
      }
      entries.push({ binding, resource });
    }
    this.processor.bindGroups.set(id, this.device.createBindGroup({ layout: record.layout, entries }));
    return true;
  }

  _setBindGroup(view, offset, payloadBytes) {
    if (payloadBytes < 16) return false;
    const index = view.getUint32(offset, true);
    const bindGroupId = view.getUint32(offset + 4, true);
    const count = view.getUint32(offset + 8, true);
    if (16 + count * 4 > payloadBytes) return false;
    const offsets = [];
    for (let i = 0; i < count; i++) offsets.push(view.getUint32(offset + 16 + i * 4, true));
    // Resolved now: a later DestroyResource (bind group retired by the runtime) must not affect it.
    const bindGroup = this.processor.bindGroups.get(bindGroupId);
    if (!bindGroup) {
      this._packetError = `unknown bind group ${bindGroupId}`;
      return false;
    }
    this._queue((ctl) => {
      if (ctl.pass) ctl.pass.setBindGroup(index, bindGroup, offsets);
    });
    return true;
  }

  _createSampler(view, offset) {
    const id = view.getUint32(offset, true);
    const desc = {
      addressModeU: nameOf(ADDRESS_MODE_NAMES, view.getUint8(offset + 4)) || 'clamp-to-edge',
      addressModeV: nameOf(ADDRESS_MODE_NAMES, view.getUint8(offset + 5)) || 'clamp-to-edge',
      addressModeW: nameOf(ADDRESS_MODE_NAMES, view.getUint8(offset + 6)) || 'clamp-to-edge',
      magFilter: nameOf(FILTER_MODE_NAMES, view.getUint8(offset + 7)) || 'nearest',
      minFilter: nameOf(FILTER_MODE_NAMES, view.getUint8(offset + 8)) || 'nearest',
      mipmapFilter: nameOf(FILTER_MODE_NAMES, view.getUint8(offset + 9)) || 'nearest',
      lodMinClamp: view.getFloat32(offset + 16, true),
      lodMaxClamp: view.getFloat32(offset + 20, true),
      maxAnisotropy: Math.max(1, view.getUint16(offset + 12, true)),
    };
    const compare = view.getUint8(offset + 10);
    if (compare !== 0) desc.compare = nameOf(COMPARE_FUNCTION_NAMES, compare);
    this.processor.samplers.set(id, this.device.createSampler(desc));
    return true;
  }
}

function installRenderStatePackets(processor, options = {}) {
  const handler = new RenderStatePackets(processor, options);
  processor.extension = handler;
  return handler;
}

const exportedApi = {
  RenderStatePackets,
  installRenderStatePackets,
  decodeRenderPipelineKey,
  decodeBeginRenderPass,
  decodeTextureViewDesc,
  decodeBindGroupLayoutEntry,
  V2_OPCODES,
  SWAPCHAIN_TEXTURE_ID,
  TEXTURE_FORMAT_NAMES,
  VERTEX_FORMAT_NAMES,
  BLEND_FACTOR_NAMES,
  BLEND_OPERATION_NAMES,
  COMPARE_FUNCTION_NAMES,
  STENCIL_OPERATION_NAMES,
  CULL_MODE_NAMES,
  FRONT_FACE_NAMES,
  PRIMITIVE_TOPOLOGY_NAMES,
  INDEX_FORMAT_NAMES,
  VERTEX_STEP_MODE_NAMES,
  LOAD_OP_NAMES,
  STORE_OP_NAMES,
  TEXTURE_ASPECT_NAMES,
  VIEW_DIMENSION_NAMES,
};

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6RenderState = exportedApi;
  } else if (globalScope) {
    globalScope.PlayGta6RenderState = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
