// GPU packet ABI v2 (opcodes 22..35, docs/gpu_commands.md "Opcodes (v2)") for WgpuCommandProcessor:
// explicit bind group / pipeline layouts, pipelines built from the 376-byte render pipeline key,
// explicit render passes, dynamic render state, bind groups with texture view descriptors and
// dynamic offsets, comparison samplers, v2 textures (CreateTexture2), and hardened compute dispatches.
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
  CREATE_TEXTURE2: 35,
});

// Native-executor extension opcode outside GpuOpcode (runtime/abi/playgta6_abi.h kGpuOpcodeClearTexture).
// Kept separate from V2_OPCODES so enum parity tests against GpuOpcode stay exact.
const OPCODE_CLEAR_TEXTURE = 63;
const CLEAR_TEXTURE_PAYLOAD_BYTES = 8;

const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);

const SWAPCHAIN_TEXTURE_ID = 0xffffffff;
const MANIFEST_REFERENCE_FLAG = 0x1;
const BINDING_ARRAYS_LAYOUT_FLAG = 0x2;
const BINDING_ARRAYS_BG_FLAG = 0x1;
const RT_ARENA_GROUP_INDEX = 3;
const RT_ARENA_BINDING_MIN = 16;
const RT_ARENA_BINDING_MAX = 21;
const KEY_OFFSET = 16; // CreateRenderPipeline2: header, then the 376-byte key
const KEY_SIZE = 376;
const CREATE_TEXTURE2_BYTES = 40;
const MAX_TEXTURE_VIEW_FORMATS = 8;
const MAX_TEXTURE_GENERATIONS = 16384;
const MAX_SEEN_WARNING_KEYS = 4096;
const MAX_LAYOUTS_BY_CONTENT = 4096;
const GPU_BUFFER_USAGE_INDEX = 0x0010;
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;
const GPU_BUFFER_USAGE_INDIRECT = 0x0100;
const GPU_BUFFER_USAGE_UNIFORM_STORAGE_COPY_DST = GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_STORAGE | 0x0008;
const INDIRECT_DRAW_BYTES = 16;
const INDIRECT_DRAW_INDEXED_BYTES = 20;
const INDIRECT_DISPATCH_BYTES = 12;
const DEFAULT_MAX_VERTEX_BUFFER_ARRAY_STRIDE = 2048;
const GPU_TEXTURE_USAGE_COPY_SRC = 0x0001;
const GPU_TEXTURE_USAGE_COPY_DST = 0x0002;
const GPU_TEXTURE_USAGE_TEXTURE_BINDING = 0x0004;
const GPU_TEXTURE_USAGE_STORAGE_BINDING = 0x0008;
const GPU_TEXTURE_USAGE_RENDER_ATTACHMENT = 0x0010;
const SRGB_SIBLING_FORMATS = Object.freeze({
  rgba8unorm: 'rgba8unorm-srgb',
  'rgba8unorm-srgb': 'rgba8unorm',
  bgra8unorm: 'bgra8unorm-srgb',
  'bgra8unorm-srgb': 'bgra8unorm',
  'bc1-rgba-unorm': 'bc1-rgba-unorm-srgb',
  'bc1-rgba-unorm-srgb': 'bc1-rgba-unorm',
  'bc2-rgba-unorm': 'bc2-rgba-unorm-srgb',
  'bc2-rgba-unorm-srgb': 'bc2-rgba-unorm',
  'bc3-rgba-unorm': 'bc3-rgba-unorm-srgb',
  'bc3-rgba-unorm-srgb': 'bc3-rgba-unorm',
  'bc7-rgba-unorm': 'bc7-rgba-unorm-srgb',
  'bc7-rgba-unorm-srgb': 'bc7-rgba-unorm',
});
// #1937/#1944 compute float-texture readback formats that may be read back via ReadbackTexture.
const FLOAT_READBACK_STORAGE_FORMATS = Object.freeze(
  new Set(['r16float', 'rg16float', 'rgba16float', 'r32float', 'rg32float', 'rgba32float'])
);
const NORM16_TO_FLOAT16_FALLBACK = Object.freeze({
  r16unorm: 'r16float',
  r16snorm: 'r16float',
  rg16unorm: 'rg16float',
  rg16snorm: 'rg16float',
  rgba16unorm: 'rgba16float',
  rgba16snorm: 'rgba16float',
});
const DUAL_SOURCE_BLEND_FACTORS = Object.freeze(
  new Set(['src1', 'one-minus-src1', 'src1-alpha', 'one-minus-src1-alpha'])
);
const ALPHA_LESS_COLOR_FORMATS = Object.freeze(
  new Set([
    'r8unorm',
    'r8snorm',
    'r8uint',
    'r8sint',
    'r16unorm',
    'r16snorm',
    'r16uint',
    'r16sint',
    'r16float',
    'rg8unorm',
    'rg8snorm',
    'rg8uint',
    'rg8sint',
    'rg16unorm',
    'rg16snorm',
    'rg16uint',
    'rg16sint',
    'rg16float',
    'r32uint',
    'r32sint',
    'r32float',
    'rg32uint',
    'rg32sint',
    'rg32float',
    'rg11b10ufloat',
    'rgb9e5ufloat',
  ])
);
const EMPTY_FRAGMENT_WGSL = '@fragment fn fs_main() {}\n';
const GPU_SHADER_STAGE_VERTEX = 0x1;
const GPU_SHADER_STAGE_ALL_MASK = 0x7;
const MAX_TELEMETRY_WARNINGS = 32;
// W101: upper bound on cached layout:'auto' remapped bind groups. Root-constant ring offsets rotate
// per draw, so the (pipeline, bind group, dynamic offsets) key space is unbounded; evict LRU-first.
const REMAPPED_BIND_GROUP_CACHE_LIMIT = 2048;
// W186: upper bound on cached PresentX multi-plane (planeTextureId, viewFormat, sampler, pipeline/layout)
// texture views & bind groups across frames; LRU-evicted and invalidated on handle/device/reset changes.
const PRESENTX_BIND_GROUP_CACHE_LIMIT = 256;
// W104: bound on distinct missing render-pass textures remembered for error deduplication.
const MISSING_TEXTURE_REPORT_LIMIT = 4096;
// W354: per-texture WeakMap GPUTextureView cache cap and fast pass state elision limits.
const MAX_VIEWS_PER_TEXTURE = 64;
const MAX_FAST_PASS_BG_SLOTS = 8;
const MAX_FAST_PASS_DYN_OFFSETS = 8;
const EMPTY_DYNAMIC_OFFSETS = Object.freeze([]);
const EMPTY_WARNINGS = Object.freeze([]);
const SWAPCHAIN_VIEW_SENTINEL = Symbol('swapchainView');

let _cachedTextureFormatsModule = null;
const _gpuObjectIds = new WeakMap();
let _nextGpuObjectId = 1;

function gpuObjectId(obj) {
  if (!obj || (typeof obj !== 'object' && typeof obj !== 'function')) return 0;
  let id = _gpuObjectIds.get(obj);
  if (id === undefined) {
    id = _nextGpuObjectId;
    _nextGpuObjectId += 1;
    _gpuObjectIds.set(obj, id);
  }
  return id;
}

function buildTextureViewKey(desc) {
  if (!desc || typeof desc !== 'object') return '';
  const fmt = desc.format || '';
  const dim = desc.dimension || '';
  const aspect = desc.aspect || '';
  const baseMip = desc.baseMipLevel || 0;
  const mipCount = desc.mipLevelCount || 0;
  const baseLayer = desc.baseArrayLayer || 0;
  const layerCount = desc.arrayLayerCount || 0;
  if (!fmt && !dim && !aspect && !baseMip && !mipCount && !baseLayer && !layerCount) {
    return '';
  }
  return `${fmt}|${dim}|${aspect}|${baseMip}|${mipCount}|${baseLayer}|${layerCount}`;
}

function getResolveStorageCompatibleFormatFn(options) {
  if (options && typeof options.resolveStorageCompatibleFormat === 'function') {
    return options.resolveStorageCompatibleFormat;
  }
  if (
    options &&
    options.textureFormats &&
    typeof options.textureFormats.resolveStorageCompatibleFormat === 'function'
  ) {
    return options.textureFormats.resolveStorageCompatibleFormat;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis.PlayGta6TextureFormats &&
    typeof globalThis.PlayGta6TextureFormats.resolveStorageCompatibleFormat === 'function'
  ) {
    return globalThis.PlayGta6TextureFormats.resolveStorageCompatibleFormat;
  }
  if (
    globalScope &&
    globalScope.PlayGta6TextureFormats &&
    typeof globalScope.PlayGta6TextureFormats.resolveStorageCompatibleFormat === 'function'
  ) {
    return globalScope.PlayGta6TextureFormats.resolveStorageCompatibleFormat;
  }
  if (_cachedTextureFormatsModule === null && typeof require === 'function') {
    try {
      _cachedTextureFormatsModule = require('./texture_formats.js') || false;
    } catch (_) {
      _cachedTextureFormatsModule = false;
    }
  }
  if (
    _cachedTextureFormatsModule &&
    typeof _cachedTextureFormatsModule.resolveStorageCompatibleFormat === 'function'
  ) {
    return _cachedTextureFormatsModule.resolveStorageCompatibleFormat;
  }
  return null;
}
const DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION = 65535;
const DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT = 256;
const DEFAULT_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT = 256;
const DEFAULT_MAX_UNIFORM_BUFFER_BINDING_SIZE = 65536;
const DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE = 134217728;
const CORE_READ_WRITE_STORAGE_FORMATS = Object.freeze(new Set(['r32uint', 'r32sint', 'r32float']));
const TIER2_READ_WRITE_STORAGE_FORMATS = Object.freeze(
  new Set([
    'r8unorm',
    'r8uint',
    'r8sint',
    'rgba8unorm',
    'rgba8uint',
    'rgba8sint',
    'r16uint',
    'r16sint',
    'r16float',
    'rgba16uint',
    'rgba16sint',
    'rgba16float',
    'rgba32uint',
    'rgba32sint',
    'rgba32float',
  ])
);
const VALID_WGSL_IDENT_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const WGSL_GROUP0_RE = /@group\(\s*0\s*\)/;

const HOOK_ORIG_FLUSH_PRESENT = '__playgta6RsOrigFlushPresent';
const HOOK_ORIG_CREATE_COMPUTE_PIPELINE = '__playgta6RsOrigCreateComputePipeline';
const HOOK_ACTIVE_HANDLER = '__playgta6RsActiveHandler';

const DEFAULT_ENTRY_POINTS = Object.freeze({
  vertex: 'vs_main',
  fragment: 'fs_main',
  compute: 'cs_main',
});

const DEPTH_TEXTURE_FORMATS = Object.freeze(
  new Set([
    'depth16unorm',
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
  ])
);

const FLOAT32_TEXTURE_FORMATS = Object.freeze(
  new Set(['r32float', 'rg32float', 'rgba32float'])
);

// Core WebGPU storage-compatible texture formats ("Texture Format Capabilities").
const CORE_STORAGE_TEXTURE_FORMATS = Object.freeze(
  new Set([
    'rgba8unorm',
    'rgba8snorm',
    'rgba8uint',
    'rgba8sint',
    'rgba16uint',
    'rgba16sint',
    'rgba16float',
    'r32uint',
    'r32sint',
    'r32float',
    'rg32uint',
    'rg32sint',
    'rg32float',
    'rgba32uint',
    'rgba32sint',
    'rgba32float',
  ])
);

const TIER1_STORAGE_TEXTURE_FORMATS = Object.freeze(
  new Set([
    'r8unorm',
    'r8snorm',
    'r8uint',
    'r8sint',
    'r16unorm',
    'r16snorm',
    'r16uint',
    'r16sint',
    'r16float',
    'rg8unorm',
    'rg8snorm',
    'rg8uint',
    'rg8sint',
    'rg16unorm',
    'rg16snorm',
    'rg16uint',
    'rg16sint',
    'rg16float',
    'rgb10a2uint',
    'rgb10a2unorm',
    'rg11b10ufloat',
    'rgba16unorm',
    'rgba16snorm',
  ])
);

const NON_RENDERABLE_CORE_FORMATS = Object.freeze(
  new Set(['rgb9e5ufloat', 'r8snorm', 'rg8snorm', 'rgba8snorm'])
);

const TIER1_RENDERABLE_FORMATS = Object.freeze(
  new Set(['r16unorm', 'r16snorm', 'rg16unorm', 'rg16snorm', 'rgba16unorm', 'rgba16snorm'])
);

// Core WebGPU blendable color target formats ("Texture Format Capabilities").
const BLENDABLE_CORE_FORMATS = Object.freeze(
  new Set([
    'r8unorm',
    'r16float',
    'rg8unorm',
    'rg16float',
    'rgba8unorm',
    'rgba8unorm-srgb',
    'bgra8unorm',
    'bgra8unorm-srgb',
    'rgb10a2unorm',
    'rgba16float',
  ])
);

const NON_MULTISAMPLEABLE_CORE_FORMATS = Object.freeze(
  new Set([
    'r32uint',
    'r32sint',
    'r32float',
    'rg32uint',
    'rg32sint',
    'rg32float',
    'rgba32uint',
    'rgba32sint',
    'rgba32float',
    'rgb10a2uint',
  ])
);

const VERTEX_FORMAT_BYTE_SIZES = Object.freeze({
  uint8: 1,
  uint8x2: 2,
  uint8x4: 4,
  sint8: 1,
  sint8x2: 2,
  sint8x4: 4,
  unorm8: 1,
  unorm8x2: 2,
  unorm8x4: 4,
  snorm8: 1,
  snorm8x2: 2,
  snorm8x4: 4,
  uint16: 2,
  uint16x2: 4,
  uint16x4: 8,
  sint16: 2,
  sint16x2: 4,
  sint16x4: 8,
  unorm16: 2,
  unorm16x2: 4,
  unorm16x4: 8,
  snorm16: 2,
  snorm16x2: 4,
  snorm16x4: 8,
  float16: 2,
  float16x2: 4,
  float16x4: 8,
  float32: 4,
  float32x2: 8,
  float32x3: 12,
  float32x4: 16,
  uint32: 4,
  uint32x2: 8,
  uint32x3: 12,
  uint32x4: 16,
  sint32: 4,
  sint32x2: 8,
  sint32x3: 12,
  sint32x4: 16,
  'unorm10-10-10-2': 4,
  'unorm8x4-bgra': 4,
});

function isDepthTextureFormat(format) {
  return typeof format === 'string' && DEPTH_TEXTURE_FORMATS.has(format);
}

function isDepthOrStencilTextureFormat(format) {
  return typeof format === 'string' && (DEPTH_TEXTURE_FORMATS.has(format) || format === 'stencil8');
}

function isBcCompressedFormat(format) {
  return typeof format === 'string' && format.startsWith('bc');
}

function isSrgbTextureFormat(format) {
  return typeof format === 'string' && format.endsWith('-srgb');
}

function getSrgbSiblingFormat(format) {
  if (typeof format !== 'string' || format.length === 0) return null;
  const normalized = format.trim().toLowerCase();
  return Object.prototype.hasOwnProperty.call(SRGB_SIBLING_FORMATS, normalized)
    ? SRGB_SIBLING_FORMATS[normalized]
    : null;
}

function isViewFormatCompatible(textureFormat, viewFormat) {
  if (typeof textureFormat !== 'string' || typeof viewFormat !== 'string') return true;
  const base = textureFormat.trim().toLowerCase();
  const target = viewFormat.trim().toLowerCase();
  if (!base || !target || base === 'undefined' || target === 'undefined') return true;
  if (base === target) return true;
  const sibling = getSrgbSiblingFormat(base);
  return sibling !== null && target === sibling;
}

// Resolves Dawn-valid `viewFormats` for `effectiveFormat`, mapping any fallback formats, stripping
// incompatible formats that Dawn would reject, and ensuring linear/sRGB siblings (#1896/#1904) are
// always registered at texture creation time.
function resolveCompatibleViewFormats(effectiveFormat, rawViewFormats, options = null) {
  const bcFallback = Boolean(options && options.bcFallback);
  const depthFallback = Boolean(options && options.depthFallback);
  const norm16Fallback = Boolean(options && options.norm16Fallback);
  const sibling = getSrgbSiblingFormat(effectiveFormat);
  const clean = [];
  const dropped = [];
  if (Array.isArray(rawViewFormats)) {
    for (let i = 0; i < rawViewFormats.length; i += 1) {
      let vf = rawViewFormats[i];
      if (typeof vf !== 'string' || vf.length === 0 || vf === 'undefined') continue;
      if (bcFallback && isBcCompressedFormat(vf)) {
        vf = isSrgbTextureFormat(vf) ? 'rgba8unorm-srgb' : 'rgba8unorm';
      } else if (depthFallback && vf === 'depth32float-stencil8' && effectiveFormat === 'depth24plus-stencil8') {
        vf = 'depth24plus-stencil8';
      } else if (norm16Fallback && Object.prototype.hasOwnProperty.call(NORM16_TO_FLOAT16_FALLBACK, vf)) {
        vf = NORM16_TO_FLOAT16_FALLBACK[vf];
      }
      if (!isViewFormatCompatible(effectiveFormat, vf)) {
        dropped.push(vf);
        continue;
      }
      if (!clean.includes(vf)) {
        clean.push(vf);
      }
    }
  }
  if (sibling !== null && !clean.includes(sibling)) {
    clean.push(sibling);
  }
  return { viewFormats: clean, dropped };
}

function hasDepthAspect(format) {
  return isDepthTextureFormat(format);
}

function hasStencilAspect(format) {
  return typeof format === 'string' && (format === 'stencil8' || isCombinedDepthStencilFormat(format));
}

function isFloat32TextureFormat(format) {
  return typeof format === 'string' && FLOAT32_TEXTURE_FORMATS.has(format);
}

function isIntegerTextureFormat(format) {
  if (typeof format !== 'string' || format.length === 0) return false;
  const normalized = format.trim().toLowerCase();
  return normalized.endsWith('uint') || normalized.endsWith('sint');
}

// Validates a GPU indirect draw/dispatch command against WebGPU/Dawn spec rules:
// live buffer, GPUBufferUsage.INDIRECT (0x0100), 4-byte offset alignment, and 16/20/12B bounds.
function validateIndirectCommand(kind, buffer, offset, bufMeta = null) {
  const requiredBytes =
    kind === 'drawIndexedIndirect'
      ? INDIRECT_DRAW_INDEXED_BYTES
      : kind === 'dispatchIndirect'
        ? INDIRECT_DISPATCH_BYTES
        : INDIRECT_DRAW_BYTES;
  if (!buffer || buffer.destroyed === true) {
    return { valid: false, reason: 'missing-or-destroyed', requiredBytes };
  }
  let usage = null;
  if (typeof buffer.usage === 'number') {
    usage = buffer.usage;
  } else if (bufMeta && typeof bufMeta.usage === 'number') {
    usage = bufMeta.usage;
  } else if (buffer.desc && typeof buffer.desc.usage === 'number') {
    usage = buffer.desc.usage;
  }
  if (usage !== null && (usage & GPU_BUFFER_USAGE_INDIRECT) === 0) {
    return { valid: false, reason: 'missing-indirect-usage', requiredBytes, usage };
  }
  if (
    typeof offset !== 'number' ||
    !Number.isFinite(offset) ||
    !Number.isInteger(offset) ||
    offset < 0 ||
    offset % 4 !== 0
  ) {
    return { valid: false, reason: 'unaligned-or-invalid-offset', requiredBytes };
  }
  let bufferSize = null;
  if (typeof buffer.size === 'number' && Number.isFinite(buffer.size) && buffer.size >= 0) {
    bufferSize = buffer.size;
  } else if (bufMeta && typeof bufMeta.byteSize === 'number' && Number.isFinite(bufMeta.byteSize) && bufMeta.byteSize >= 0) {
    bufferSize = bufMeta.byteSize;
  } else if (buffer.desc && typeof buffer.desc.size === 'number' && Number.isFinite(buffer.desc.size) && buffer.desc.size >= 0) {
    bufferSize = buffer.desc.size;
  }
  if (bufferSize !== null && offset + requiredBytes > bufferSize) {
    return { valid: false, reason: 'offset-out-of-bounds', requiredBytes, bufferSize };
  }
  return { valid: true, requiredBytes };
}

// Sanitizes a GPUSamplerDescriptor against Dawn validation rules:
// finite non-negative lodMinClamp <= lodMaxClamp, and maxAnisotropy > 1 only when all filters are 'linear'.
function sanitizeSamplerDescriptor(desc) {
  if (!desc || typeof desc !== 'object') return desc;
  const out = { ...desc };
  const lodMin = Math.max(0, finiteOr(out.lodMinClamp, 0)) || 0;
  const rawMax = finiteOr(out.lodMaxClamp, 32);
  const lodMax = Math.max(lodMin, rawMax) || 0;
  out.lodMinClamp = lodMin;
  out.lodMaxClamp = lodMax;
  const rawAniso = Math.floor(finiteOr(out.maxAnisotropy, 1));
  const allLinear =
    out.magFilter === 'linear' && out.minFilter === 'linear' && out.mipmapFilter === 'linear';
  out.maxAnisotropy = allLinear ? Math.min(16, Math.max(1, rawAniso)) : 1;
  if ('compare' in out && (!out.compare || out.compare === 'undefined')) {
    delete out.compare;
  }
  return out;
}

// W266: WebGPU forbids blending on integer (*uint/*sint), depth/stencil, compressed, and
// non-blendable float32 formats (r32float/rg32float/rgba32float without 'float32-blendable').
function isBlendableTextureFormat(format, device) {
  if (typeof format !== 'string' || format.length === 0 || format === 'undefined') return false;
  const normalized = format.trim().toLowerCase();
  if (isIntegerTextureFormat(normalized)) return false;
  if (isDepthOrStencilTextureFormat(normalized) || isBcCompressedFormat(normalized)) return false;
  if (BLENDABLE_CORE_FORMATS.has(normalized)) return true;
  if (FLOAT32_TEXTURE_FORMATS.has(normalized)) {
    return deviceHasFeature(device, 'float32-blendable');
  }
  if (normalized === 'rg11b10ufloat') {
    return deviceHasFeature(device, 'rg11b10ufloat-renderable');
  }
  if (TIER1_RENDERABLE_FORMATS.has(normalized)) {
    return deviceHasFeature(device, 'texture-formats-tier1');
  }
  return false;
}

function blendComponentUsesDualSource(comp) {
  return Boolean(
    comp &&
      (DUAL_SOURCE_BLEND_FACTORS.has(comp.srcFactor) || DUAL_SOURCE_BLEND_FACTORS.has(comp.dstFactor))
  );
}

// W266 + Dawn parity: strips `.blend` when format is not blendable or dual-source blending is
// unsupported, and forces srcFactor='one', dstFactor='one' for 'min'/'max' blend operations.
function sanitizeColorTargetBlendState(target, device, colorTargetCount = 1) {
  if (!target || typeof target !== 'object' || !target.blend) {
    return { target, stripped: false, reason: null };
  }
  const format = typeof target.format === 'string' ? target.format.trim().toLowerCase() : '';
  if (!isBlendableTextureFormat(format, device)) {
    const cleanTarget = { ...target };
    delete cleanTarget.blend;
    let reason = 'format-not-blendable';
    if (isIntegerTextureFormat(format)) {
      reason = 'integer-format-not-blendable';
    } else if (FLOAT32_TEXTURE_FORMATS.has(format)) {
      reason = 'float32-format-not-blendable';
    }
    return { target: cleanTarget, stripped: true, reason };
  }
  const blend = target.blend;
  const usesDualSource =
    blendComponentUsesDualSource(blend.color) || blendComponentUsesDualSource(blend.alpha);
  if (usesDualSource && (!deviceHasFeature(device, 'dual-source-blending') || colorTargetCount > 1)) {
    const cleanTarget = { ...target };
    delete cleanTarget.blend;
    return { target: cleanTarget, stripped: true, reason: 'dual-source-blending-unsupported' };
  }
  let colorComp = blend.color;
  let alphaComp = blend.alpha;
  let adjusted = false;
  if (
    colorComp &&
    (colorComp.operation === 'min' || colorComp.operation === 'max') &&
    (colorComp.srcFactor !== 'one' || colorComp.dstFactor !== 'one')
  ) {
    colorComp = { ...colorComp, srcFactor: 'one', dstFactor: 'one' };
    adjusted = true;
  }
  if (
    alphaComp &&
    (alphaComp.operation === 'min' || alphaComp.operation === 'max') &&
    (alphaComp.srcFactor !== 'one' || alphaComp.dstFactor !== 'one')
  ) {
    alphaComp = { ...alphaComp, srcFactor: 'one', dstFactor: 'one' };
    adjusted = true;
  }
  if (adjusted) {
    return {
      target: { ...target, blend: { color: colorComp, alpha: alphaComp } },
      stripped: false,
      reason: 'min-max-blend-factors-normalized',
    };
  }
  return { target, stripped: false, reason: null };
}

function deviceHasFeature(device, featureName) {
  const features = device && device.features;
  if (!features || !featureName) return false;
  if (typeof features.has === 'function') return Boolean(features.has(featureName));
  if (Array.isArray(features)) return features.includes(featureName);
  if (typeof features[Symbol.iterator] === 'function') {
    for (const item of features) {
      if (item === featureName) return true;
    }
    return false;
  }
  if (typeof features === 'object') return Boolean(features[featureName]);
  return false;
}

// W163/W186: resolves storage-texture compatibility using texture_formats.js's
// resolveStorageCompatibleFormat when loaded, falling back to built-in tables in isolated VMs.
function resolveStorageTextureFormat(format, device, options) {
  if (typeof format !== 'string' || format.length === 0 || format === 'undefined') {
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'invalid-format',
    };
  }
  const helper = getResolveStorageCompatibleFormatFn(options);
  if (helper) {
    try {
      const resolved = helper(format, { deviceFeatures: device && device.features });
      if (resolved && typeof resolved === 'object') {
        return {
          storageFormat: typeof resolved.storageFormat === 'string' ? resolved.storageFormat : null,
          supportsStorageBinding: Boolean(resolved.supportsStorageBinding),
          needsLinearSiblingView: Boolean(resolved.needsLinearSiblingView),
          reason: typeof resolved.reason === 'string' ? resolved.reason : null,
        };
      }
    } catch (_) {
      // Fall through to built-in format tables if an injected helper throws.
    }
  }
  const normalized = format.trim().toLowerCase();
  if (isDepthOrStencilTextureFormat(normalized)) {
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'depth-stencil-not-storage-compatible',
    };
  }
  if (isBcCompressedFormat(normalized)) {
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'compressed-not-storage-compatible',
    };
  }
  if (CORE_STORAGE_TEXTURE_FORMATS.has(normalized)) {
    return {
      storageFormat: normalized,
      supportsStorageBinding: true,
      needsLinearSiblingView: false,
      reason: null,
    };
  }
  if (normalized === 'bgra8unorm') {
    if (deviceHasFeature(device, 'bgra8unorm-storage')) {
      return {
        storageFormat: 'bgra8unorm',
        supportsStorageBinding: true,
        needsLinearSiblingView: false,
        reason: null,
      };
    }
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'missing-bgra8unorm-storage',
    };
  }
  if (TIER1_STORAGE_TEXTURE_FORMATS.has(normalized)) {
    if (deviceHasFeature(device, 'texture-formats-tier1')) {
      return {
        storageFormat: normalized,
        supportsStorageBinding: true,
        needsLinearSiblingView: false,
        reason: null,
      };
    }
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'missing-texture-formats-tier1',
    };
  }
  if (normalized === 'rgba8unorm-srgb') {
    return {
      storageFormat: 'rgba8unorm',
      supportsStorageBinding: true,
      needsLinearSiblingView: true,
      reason: 'srgb-requires-linear-view',
    };
  }
  if (normalized === 'bgra8unorm-srgb') {
    if (deviceHasFeature(device, 'bgra8unorm-storage')) {
      return {
        storageFormat: 'bgra8unorm',
        supportsStorageBinding: true,
        needsLinearSiblingView: true,
        reason: 'srgb-requires-linear-view',
      };
    }
    return {
      storageFormat: null,
      supportsStorageBinding: false,
      needsLinearSiblingView: false,
      reason: 'missing-bgra8unorm-storage',
    };
  }
  return {
    storageFormat: null,
    supportsStorageBinding: false,
    needsLinearSiblingView: false,
    reason: 'unsupported-storage-format',
  };
}

function isStorageBindingSupported(format, device, options) {
  const resolved = resolveStorageTextureFormat(format, device, options);
  return Boolean(resolved.supportsStorageBinding && !resolved.needsLinearSiblingView);
}

function isRenderAttachmentSupported(format, device) {
  if (typeof format !== 'string' || format.length === 0 || format === 'undefined') return false;
  if (isBcCompressedFormat(format) || NON_RENDERABLE_CORE_FORMATS.has(format)) return false;
  if (format === 'rg11b10ufloat') return deviceHasFeature(device, 'rg11b10ufloat-renderable');
  if (TIER1_RENDERABLE_FORMATS.has(format)) {
    return deviceHasFeature(device, 'texture-formats-tier1');
  }
  return true;
}

function isMultisampleSupported(format, device) {
  if (!isRenderAttachmentSupported(format, device)) return false;
  if (NON_MULTISAMPLEABLE_CORE_FORMATS.has(format)) return false;
  return true;
}

// Resolves the WebGPU GPUTextureSampleType compatible with `format` on `device`.
// Notably maps rgba32float / rg32float / r32float to 'unfilterable-float' when 'float32-filterable' is absent.
function resolveTextureSampleType(format, device) {
  if (typeof format !== 'string' || format.length === 0 || format === 'undefined') return 'float';
  if (isDepthTextureFormat(format)) return 'depth';
  if (format === 'stencil8' || format.endsWith('uint')) return 'uint';
  if (format.endsWith('sint')) return 'sint';
  if (isFloat32TextureFormat(format)) {
    return deviceHasFeature(device, 'float32-filterable') ? 'float' : 'unfilterable-float';
  }
  return 'float';
}

function getDeviceLimit(device, limitName, fallback) {
  const limits = device && typeof device === 'object' ? device.limits : null;
  const raw = limits && typeof limits === 'object' ? limits[limitName] : undefined;
  if (typeof raw === 'number' && Number.isFinite(raw) && raw > 0) {
    return Math.floor(raw);
  }
  return fallback;
}

function isCombinedDepthStencilFormat(format) {
  return format === 'depth24plus-stencil8' || format === 'depth32float-stencil8';
}

function resolveEffectiveViewFormat(format, aspect) {
  if (typeof format !== 'string' || format.length === 0) return format;
  if (aspect === 'depth-only') {
    if (format === 'depth32float-stencil8') return 'depth32float';
    if (format === 'depth24plus-stencil8') return 'depth24plus';
  } else if (aspect === 'stencil-only') {
    if (format === 'depth32float-stencil8' || format === 'depth24plus-stencil8') return 'stencil8';
  }
  return format;
}

function isTextureSampleTypeCompatible(format, expectedSampleType, device) {
  if (typeof format !== 'string' || format.length === 0 || format === 'undefined') return true;
  const expected = expectedSampleType || 'float';
  const actual = resolveTextureSampleType(format, device);
  if (expected === 'unfilterable-float') {
    return actual === 'float' || actual === 'unfilterable-float' || actual === 'depth';
  }
  return actual === expected;
}

function isStorageAccessCompatible(format, access, device) {
  if (access !== 'read-write') return true;
  if (typeof format !== 'string' || format.length === 0) return false;
  const normalized = format.trim().toLowerCase();
  if (CORE_READ_WRITE_STORAGE_FORMATS.has(normalized)) return true;
  if (
    deviceHasFeature(device, 'texture-formats-tier2') ||
    deviceHasFeature(device, 'chromium-experimental-read-write-storage-texture')
  ) {
    return TIER2_READ_WRITE_STORAGE_FORMATS.has(normalized);
  }
  return false;
}

function isSamplerFilteringDesc(desc) {
  if (!desc || typeof desc !== 'object') return false;
  return (
    desc.magFilter === 'linear' ||
    desc.minFilter === 'linear' ||
    desc.mipmapFilter === 'linear' ||
    (typeof desc.maxAnisotropy === 'number' && desc.maxAnisotropy > 1)
  );
}

function computeMaxMipLevels(width, height, depthOrArrayLayers = 1, dimension = '2d') {
  if (dimension === '1d') return 1;
  const w = Math.max(1, Math.floor(Number(width) || 1));
  const h = Math.max(1, Math.floor(Number(height) || 1));
  const d = Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
  const maxDim = dimension === '3d' ? Math.max(w, h, d) : Math.max(w, h);
  return Math.floor(Math.log2(maxDim)) + 1;
}

function getMaxComputeWorkgroupsPerDimension(deviceOrLimits) {
  const limits =
    deviceOrLimits && typeof deviceOrLimits === 'object'
      ? deviceOrLimits.limits && typeof deviceOrLimits.limits === 'object'
        ? deviceOrLimits.limits
        : deviceOrLimits
      : null;
  const raw = limits ? limits.maxComputeWorkgroupsPerDimension : undefined;
  if (typeof raw === 'number' && Number.isFinite(raw) && raw >= 1) {
    return Math.floor(raw);
  }
  return DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION;
}

// Validates and clamps compute dispatch workgroup counts against WebGPU limits.
// Zero or invalid dimensions are skipped; oversized dimensions are clamped to maxComputeWorkgroupsPerDimension.
function validateComputeDispatch(x, y, z, deviceOrLimits, out = null) {
  const maxDim = getMaxComputeWorkgroupsPerDimension(deviceOrLimits);
  const fx = typeof x === 'number' && Number.isFinite(x) ? Math.floor(x) : 0;
  const fy = typeof y === 'number' && Number.isFinite(y) ? Math.floor(y) : 0;
  const fz = typeof z === 'number' && Number.isFinite(z) ? Math.floor(z) : 0;
  const dst = out || {
    skip: false,
    clamped: false,
    x: 0,
    y: 0,
    z: 0,
    maxWorkgroupsPerDimension: maxDim,
    reason: null,
  };
  dst.maxWorkgroupsPerDimension = maxDim;
  if (fx <= 0 || fy <= 0 || fz <= 0) {
    dst.skip = true;
    dst.clamped = false;
    dst.x = 0;
    dst.y = 0;
    dst.z = 0;
    dst.reason = 'zero-dimension';
    return dst;
  }
  if (fx > maxDim || fy > maxDim || fz > maxDim) {
    dst.skip = false;
    dst.clamped = true;
    dst.x = Math.min(fx, maxDim);
    dst.y = Math.min(fy, maxDim);
    dst.z = Math.min(fz, maxDim);
    dst.reason = 'exceeds-limit';
    return dst;
  }
  dst.skip = false;
  dst.clamped = false;
  dst.x = fx;
  dst.y = fy;
  dst.z = fz;
  dst.reason = null;
  return dst;
}

// Decodes an ABI v2 CreateTexture2 payload (opcode 35, 40-byte GpuCreateTexture2Payload).
// Supports an optional pre-allocated `out` scratch target (as 4th arg, or 3rd arg when payloadBytes is omitted).
function decodeCreateTexture2(view, offset = 0, payloadBytes, out = null) {
  let targetOut = out;
  let explicitPayloadBytes = payloadBytes;
  if (typeof payloadBytes === 'object' && payloadBytes !== null && out === null) {
    targetOut = payloadBytes;
    explicitPayloadBytes = undefined;
  }
  const writeError = (id, message) => {
    if (targetOut) {
      targetOut.id = id;
      targetOut.error = message;
      if ('descriptor' in targetOut) delete targetOut.descriptor;
      return targetOut;
    }
    return { id, error: message };
  };
  if (!view) {
    return writeError(0, `CreateTexture2: payload 0 < ${CREATE_TEXTURE2_BYTES} bytes`);
  }
  const available =
    explicitPayloadBytes !== undefined ? explicitPayloadBytes : view.byteLength - offset;
  if (
    offset < 0 ||
    available < CREATE_TEXTURE2_BYTES ||
    offset + CREATE_TEXTURE2_BYTES > view.byteLength
  ) {
    return writeError(0, `CreateTexture2: payload ${available} < ${CREATE_TEXTURE2_BYTES} bytes`);
  }
  const id = view.getUint32(offset, true);
  const rawWidth = view.getUint32(offset + 4, true);
  const rawHeight = view.getUint32(offset + 8, true);
  const rawDepth = view.getUint32(offset + 12, true);
  const rawMipLevels = view.getUint32(offset + 16, true);
  const rawSampleCount = view.getUint32(offset + 20, true);
  const usage = view.getUint32(offset + 24, true);
  const dimensionCode = view.getUint8(offset + 28);
  const formatCode = view.getUint8(offset + 29);
  const viewFormatCount = view.getUint8(offset + 30);

  const format = nameOf(TEXTURE_FORMAT_NAMES, formatCode);
  if (formatCode === 0 || !format || format === 'undefined') {
    return writeError(id, `CreateTexture2 ${id}: invalid texture format code ${formatCode}`);
  }
  const dimension = nameOf(TEXTURE_DIMENSION_NAMES, dimensionCode);
  if (!dimension) {
    return writeError(id, `CreateTexture2 ${id}: invalid dimension ${dimensionCode}`);
  }
  if (viewFormatCount > MAX_TEXTURE_VIEW_FORMATS) {
    return writeError(
      id,
      `CreateTexture2 ${id}: view_format_count ${viewFormatCount} > ${MAX_TEXTURE_VIEW_FORMATS}`
    );
  }

  let pool = targetOut ? targetOut._pool : null;
  if (targetOut && !pool) {
    const sizeObj =
      targetOut.descriptor && targetOut.descriptor.size && typeof targetOut.descriptor.size === 'object'
        ? targetOut.descriptor.size
        : { width: 1, height: 1, depthOrArrayLayers: 1 };
    const descObj =
      targetOut.descriptor && typeof targetOut.descriptor === 'object'
        ? targetOut.descriptor
        : {
            size: sizeObj,
            mipLevelCount: 1,
            sampleCount: 1,
            dimension: '2d',
            format: 'rgba8unorm',
            usage: 0,
          };
    descObj.size = sizeObj;
    pool = {
      descriptor: descObj,
      size: sizeObj,
      viewFormats: Array.isArray(descObj.viewFormats) ? descObj.viewFormats : [],
    };
    Object.defineProperty(targetOut, '_pool', {
      value: pool,
      enumerable: false,
      configurable: false,
    });
  }

  const viewFormats = pool ? pool.viewFormats : [];
  viewFormats.length = 0;
  for (let i = 0; i < viewFormatCount; i += 1) {
    const code = view.getUint8(offset + 32 + i);
    const vf = nameOf(TEXTURE_FORMAT_NAMES, code);
    if (code === 0 || !vf || vf === 'undefined') {
      viewFormats.length = 0;
      return writeError(id, `CreateTexture2 ${id}: invalid view format code ${code}`);
    }
    viewFormats.push(vf);
  }

  const width = Math.max(1, rawWidth);
  const height = dimension === '1d' ? 1 : Math.max(1, rawHeight);
  const depthOrArrayLayers = dimension === '1d' ? 1 : Math.max(1, rawDepth);
  const maxMips = computeMaxMipLevels(width, height, depthOrArrayLayers, dimension);
  const mipLevelCount =
    Number.isFinite(rawMipLevels) && rawMipLevels >= 1
      ? Math.min(maxMips, Math.floor(rawMipLevels))
      : 1;
  const geometryAllowsMsaa =
    dimension === '2d' && mipLevelCount <= 1 && depthOrArrayLayers === 1;
  const sampleCount = rawSampleCount >= 4 && geometryAllowsMsaa ? 4 : 1;

  if (targetOut) {
    const size = pool.size;
    size.width = width;
    size.height = height;
    size.depthOrArrayLayers = depthOrArrayLayers;
    const descriptor = pool.descriptor;
    descriptor.size = size;
    descriptor.mipLevelCount = mipLevelCount;
    descriptor.sampleCount = sampleCount;
    descriptor.dimension = dimension;
    descriptor.format = format;
    descriptor.usage = usage;
    if (viewFormats.length > 0) {
      descriptor.viewFormats = viewFormats;
    } else if ('viewFormats' in descriptor) {
      delete descriptor.viewFormats;
    }
    targetOut.id = id;
    targetOut.descriptor = descriptor;
    if ('error' in targetOut) delete targetOut.error;
    return targetOut;
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

// Sanitizes a GPUTextureDescriptor against WebGPU format & device capability constraints
// (stripping illegal STORAGE_BINDING / RENDER_ATTACHMENT usages, falling back cleanly on
// unsupported BC compressed / optional depth / norm16 formats, preserving COPY_SRC on float
// storage textures for #1937/#1944 readbacks, and ensuring compatible sRGB/linear viewFormats
// for #1896/#1904).
function sanitizeTextureDescriptor(rawDescriptor, device, options) {
  const originalFormat = rawDescriptor.format;
  const wasBc = isBcCompressedFormat(originalFormat);
  let effectiveFormat = originalFormat;
  let effectiveDimension = rawDescriptor.dimension || '2d';
  let usage = rawDescriptor.usage >>> 0;
  let sampleCount = rawDescriptor.sampleCount || 1;
  let bcFallback = false;
  let formatFallback = false;
  let depthFallback = false;
  let norm16Fallback = false;
  const strippedUsages = [];
  const warnings = [];

  const is1d = effectiveDimension === '1d';
  const is3d = effectiveDimension === '3d';
  const texWidth = rawDescriptor.size && typeof rawDescriptor.size.width === 'number' ? rawDescriptor.size.width : 1;
  const texHeight = rawDescriptor.size && typeof rawDescriptor.size.height === 'number' ? rawDescriptor.size.height : 1;
  if (wasBc && !deviceHasFeature(device, 'texture-compression-bc')) {
    bcFallback = true;
    formatFallback = true;
    effectiveFormat = isSrgbTextureFormat(originalFormat) ? 'rgba8unorm-srgb' : 'rgba8unorm';
    warnings.push({
      key: `bc-unsupported:${originalFormat}`,
      message: `texture format ${originalFormat} unavailable without texture-compression-bc; falling back to ${effectiveFormat}`,
    });
  } else if (wasBc && is3d && !deviceHasFeature(device, 'texture-compression-bc-sliced-3d')) {
    // W107 (#695): WebGPU rejects 3D BC-compressed textures without texture-compression-bc-sliced-3d.
    bcFallback = true;
    formatFallback = true;
    effectiveFormat = isSrgbTextureFormat(originalFormat) ? 'rgba8unorm-srgb' : 'rgba8unorm';
    warnings.push({
      key: `bc-3d-unsupported:${originalFormat}`,
      message: `3d texture format ${originalFormat} unavailable without texture-compression-bc-sliced-3d; falling back to ${effectiveFormat}`,
    });
  } else if (wasBc && (is1d || texWidth % 4 !== 0 || texHeight % 4 !== 0)) {
    // WebGPU rejects BC compressed textures on 1D or non-multiple-of-4 dimensions.
    bcFallback = true;
    formatFallback = true;
    effectiveFormat = isSrgbTextureFormat(originalFormat) ? 'rgba8unorm-srgb' : 'rgba8unorm';
    warnings.push({
      key: `bc-dims-invalid:${originalFormat}:${effectiveDimension}`,
      message: `BC texture format ${originalFormat} invalid for ${effectiveDimension} ${texWidth}x${texHeight}; falling back to ${effectiveFormat}`,
    });
  } else if (
    originalFormat === 'depth32float-stencil8' &&
    !deviceHasFeature(device, 'depth32float-stencil8')
  ) {
    formatFallback = true;
    depthFallback = true;
    effectiveFormat = 'depth24plus-stencil8';
    warnings.push({
      key: 'depth32float-stencil8-unsupported',
      message: 'texture format depth32float-stencil8 unavailable; falling back to depth24plus-stencil8',
    });
  } else if (
    Object.prototype.hasOwnProperty.call(NORM16_TO_FLOAT16_FALLBACK, originalFormat) &&
    !deviceHasFeature(device, 'texture-formats-tier1') &&
    !deviceHasFeature(device, 'chromium-experimental-unorm16-texture-formats')
  ) {
    formatFallback = true;
    norm16Fallback = true;
    effectiveFormat = NORM16_TO_FLOAT16_FALLBACK[originalFormat];
    warnings.push({
      key: `norm16-unsupported:${originalFormat}`,
      message: `texture format ${originalFormat} unavailable without texture-formats-tier1; falling back to ${effectiveFormat}`,
    });
  }

  // WebGPU forbids 1D and 3D depth/stencil textures.
  if (isDepthOrStencilTextureFormat(effectiveFormat) && effectiveDimension !== '2d') {
    warnings.push({
      key: `depth-dim-coerce:${effectiveDimension}:${effectiveFormat}`,
      message: `coerced ${effectiveDimension} depth/stencil texture (${effectiveFormat}) to 2d`,
    });
    effectiveDimension = '2d';
  }

  if (
    (usage & GPU_TEXTURE_USAGE_STORAGE_BINDING) !== 0 &&
    (wasBc || !isStorageBindingSupported(effectiveFormat, device, options))
  ) {
    usage &= ~GPU_TEXTURE_USAGE_STORAGE_BINDING;
    strippedUsages.push('STORAGE_BINDING');
    warnings.push({
      key: `strip-storage:${originalFormat}`,
      message: `stripped unsupported STORAGE_BINDING usage for texture format ${originalFormat}`,
    });
  }

  if (
    (usage & GPU_TEXTURE_USAGE_RENDER_ATTACHMENT) !== 0 &&
    (effectiveDimension === '1d' || wasBc || !isRenderAttachmentSupported(effectiveFormat, device))
  ) {
    usage &= ~GPU_TEXTURE_USAGE_RENDER_ATTACHMENT;
    strippedUsages.push('RENDER_ATTACHMENT');
    warnings.push({
      key: `strip-render-attachment:${originalFormat}`,
      message: `stripped unsupported RENDER_ATTACHMENT usage for texture format ${originalFormat}`,
    });
  }

  if (sampleCount > 1) {
    // WebGPU forbids STORAGE_BINDING on multisampled textures and requires RENDER_ATTACHMENT.
    if ((usage & GPU_TEXTURE_USAGE_STORAGE_BINDING) !== 0) {
      usage &= ~GPU_TEXTURE_USAGE_STORAGE_BINDING;
      strippedUsages.push('STORAGE_BINDING');
      warnings.push({
        key: `strip-msaa-storage:${originalFormat}`,
        message: `stripped unsupported STORAGE_BINDING usage on multisampled texture (${originalFormat})`,
      });
    }
    if (
      wasBc ||
      effectiveDimension !== '2d' ||
      (usage & GPU_TEXTURE_USAGE_RENDER_ATTACHMENT) === 0 ||
      !isMultisampleSupported(effectiveFormat, device)
    ) {
      sampleCount = 1;
    }
  }

  // #1937/#1944: address-backed compute UAV float textures (ensure_address_texture: STORAGE_BINDING
  // without RENDER_ATTACHMENT) may be read back via ReadbackTexture; ensure COPY_SRC is present so
  // copyTextureToBuffer never fails Dawn validation.
  if (
    (usage & GPU_TEXTURE_USAGE_STORAGE_BINDING) !== 0 &&
    (usage & GPU_TEXTURE_USAGE_RENDER_ATTACHMENT) === 0 &&
    effectiveDimension === '2d' &&
    sampleCount === 1 &&
    FLOAT_READBACK_STORAGE_FORMATS.has(effectiveFormat)
  ) {
    usage |= GPU_TEXTURE_USAGE_COPY_SRC;
  }

  if (usage === 0) {
    usage = GPU_TEXTURE_USAGE_TEXTURE_BINDING;
  }

  const descriptor = {
    size: { ...rawDescriptor.size },
    mipLevelCount: rawDescriptor.mipLevelCount || 1,
    sampleCount,
    dimension: effectiveDimension,
    format: effectiveFormat,
    usage,
  };

  // #1896/#1904: strip Dawn-incompatible viewFormats and auto-register sRGB/linear siblings so
  // PresentX linear sibling views and GPU-to-GPU bit-exact copies never fail createView validation.
  const vfResolved = resolveCompatibleViewFormats(effectiveFormat, rawDescriptor.viewFormats, {
    bcFallback,
    depthFallback,
    norm16Fallback,
  });
  if (vfResolved.dropped.length > 0) {
    warnings.push({
      key: `strip-view-formats:${effectiveFormat}:${vfResolved.dropped.join(',')}`,
      message: `stripped incompatible viewFormats [${vfResolved.dropped.join(', ')}] for texture format ${effectiveFormat}`,
    });
  }
  if (vfResolved.viewFormats.length > 0) {
    descriptor.viewFormats = vfResolved.viewFormats;
  }

  return {
    descriptor,
    originalFormat,
    effectiveFormat,
    strippedUsages,
    bcFallback,
    formatFallback,
    warnings,
  };
}

const MAX_PARSED_WGSL_CACHE_ENTRIES = 512;
const _parsedWgslGroup0Cache = new Map();
const _parsedWgslStageEntriesCache = new Map();

function _setBoundedCache(map, key, value) {
  if (map.has(key)) {
    map.delete(key);
  } else if (map.size >= MAX_PARSED_WGSL_CACHE_ENTRIES) {
    const oldest = map.keys().next().value;
    if (oldest !== undefined) {
      map.delete(oldest);
    }
  }
  map.set(key, value);
}

// Strips WGSL line (//) and nestable block (/* ... */) comments so commented-out entry points are ignored.
function stripWgslComments(code) {
  if (typeof code !== 'string' || code.length === 0) return '';
  if (code.indexOf('//') === -1 && code.indexOf('/*') === -1) return code;
  const chunks = [];
  let chunkStart = 0;
  let blockDepth = 0;
  const len = code.length;
  for (let i = 0; i < len; i += 1) {
    const ch = code.charCodeAt(i);
    if (blockDepth > 0) {
      if (ch === 47 /* '/' */ && i + 1 < len && code.charCodeAt(i + 1) === 42 /* '*' */) {
        blockDepth += 1;
        i += 1;
      } else if (ch === 42 /* '*' */ && i + 1 < len && code.charCodeAt(i + 1) === 47 /* '/' */) {
        blockDepth -= 1;
        i += 1;
        chunks.push(' ');
        if (blockDepth === 0) {
          chunkStart = i + 1;
        }
      } else if (ch === 10 /* '\n' */) {
        chunks.push('\n');
      }
      continue;
    }
    if (ch === 47 /* '/' */ && i + 1 < len) {
      const next = code.charCodeAt(i + 1);
      if (next === 47 /* '/' */) {
        if (i > chunkStart) {
          chunks.push(code.slice(chunkStart, i));
        }
        i += 2;
        while (i < len && code.charCodeAt(i) !== 10 /* '\n' */) i += 1;
        if (i < len && code.charCodeAt(i) === 10 /* '\n' */) {
          chunks.push('\n');
        }
        chunkStart = i + 1;
        continue;
      }
      if (next === 42 /* '*' */) {
        if (i > chunkStart) {
          chunks.push(code.slice(chunkStart, i));
        }
        blockDepth = 1;
        i += 1;
        chunkStart = len;
        continue;
      }
    }
  }
  if (blockDepth === 0 && chunkStart < len) {
    chunks.push(code.slice(chunkStart));
  }
  return chunks.join('');
}

function findWgslStageEntryPoints(code, stage) {
  if (typeof code !== 'string' || code.length === 0) return [];
  const stageAttr =
    stage === 'vertex' || stage === 'fragment' || stage === 'compute' ? stage : null;
  if (!stageAttr) return [];
  const cacheKey = `${stageAttr}\0${code}`;
  const cached = _parsedWgslStageEntriesCache.get(cacheKey);
  if (cached !== undefined) return cached;
  const cleaned = stripWgslComments(code);
  const re = new RegExp(
    `@${stageAttr}\\b(?:\\s*@[A-Za-z_][A-Za-z0-9_]*(?:\\s*\\([^)]*\\))?)*\\s*fn\\s+([A-Za-z_][A-Za-z0-9_]*)`,
    'g'
  );
  const found = [];
  let match;
  while ((match = re.exec(cleaned)) !== null) {
    if (!found.includes(match[1])) {
      found.push(match[1]);
    }
  }
  const frozen = Object.freeze(found);
  _setBoundedCache(_parsedWgslStageEntriesCache, cacheKey, frozen);
  return frozen;
}

// #1953 parity: extracts the set of `@location(N)` color output indices written by a WGSL
// `@fragment` entry point. Returns a Set<number> when the fragment return signature is complete
// and parseable, or `null` when unavailable/compacted/unparseable (fail-open to avoid false zeroes).
function parseWgslFragmentOutputLocations(wgslSource, entryPointName = null) {
  if (typeof wgslSource !== 'string' || wgslSource.length === 0) return null;
  const cleaned = stripWgslComments(wgslSource);
  if (cleaned.length === 0) return null;
  const fragFnRe =
    /@fragment\b(?:\s*@[A-Za-z_][A-Za-z0-9_]*(?:\s*\([^)]*\))?)*\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g;
  let chosenMatch = null;
  let match;
  while ((match = fragFnRe.exec(cleaned)) !== null) {
    if (!entryPointName || match[1] === entryPointName) {
      chosenMatch = match;
      break;
    }
    if (!chosenMatch) chosenMatch = match;
  }
  if (!chosenMatch) return null;
  let i = chosenMatch.index + chosenMatch[0].length;
  let parenDepth = 1;
  while (i < cleaned.length && parenDepth > 0) {
    if (cleaned[i] === '(') parenDepth += 1;
    else if (cleaned[i] === ')') parenDepth -= 1;
    i += 1;
  }
  const braceIdx = cleaned.indexOf('{', i);
  if (braceIdx < 0) return null;
  const sigTail = cleaned.slice(i, braceIdx).trim();
  if (!sigTail.startsWith('->')) {
    // Void fragment shader writes zero color locations (e.g. depth-only or empty fs_main).
    return new Set();
  }
  const retClause = sigTail.slice(2).trim();
  const locations = new Set();
  const directLocRe = /@location\s*\(\s*(\d+)[ui]?\s*\)/g;
  let lm;
  while ((lm = directLocRe.exec(retClause)) !== null) {
    const idx = parseInt(lm[1], 10);
    if (Number.isInteger(idx) && idx >= 0 && idx < 8) locations.add(idx);
  }
  if (locations.size > 0) return locations;
  if (/@builtin\s*\(/.test(retClause)) {
    // e.g. `-> @builtin(frag_depth) f32` writes no color targets.
    return new Set();
  }
  // Check for a named return struct `-> FsOut`.
  const strippedAttrs = retClause.replace(/@[A-Za-z_][A-Za-z0-9_]*(?:\s*\([^)]*\))?/g, '').trim();
  const structNameMatch = /^([A-Za-z_][A-Za-z0-9_]*)$/.exec(strippedAttrs);
  if (!structNameMatch) return null;
  const structName = structNameMatch[1];
  const structRe = new RegExp(`\\bstruct\\s+${structName}\\s*\\{([\\s\\S]*?)\\}`);
  const sm = structRe.exec(cleaned);
  if (!sm) return null;
  directLocRe.lastIndex = 0;
  while ((lm = directLocRe.exec(sm[1])) !== null) {
    const idx = parseInt(lm[1], 10);
    if (Number.isInteger(idx) && idx >= 0 && idx < 8) locations.add(idx);
  }
  return locations;
}

function normalizeEntryPointName(raw) {
  if (typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  if (!trimmed || !VALID_WGSL_IDENT_RE.test(trimmed)) return null;
  return trimmed;
}

function getShaderSourceText(shaderEntry) {
  if (!shaderEntry || typeof shaderEntry !== 'object') return null;
  if (typeof shaderEntry.code === 'string' && shaderEntry.code.length > 0) return shaderEntry.code;
  if (typeof shaderEntry.wgsl === 'string' && shaderEntry.wgsl.length > 0) return shaderEntry.wgsl;
  if (
    shaderEntry.module &&
    typeof shaderEntry.module.code === 'string' &&
    shaderEntry.module.code.length > 0
  ) {
    return shaderEntry.module.code;
  }
  return null;
}

function wgslDeclaresGroup0(code) {
  return typeof code === 'string' && code.length > 0 && WGSL_GROUP0_RE.test(code);
}

const WGSL_GROUP0_VAR_RE =
  /(?:@group\s*\(\s*0[ui]?\s*\)\s*@binding\s*\(\s*(\d+)[ui]?\s*\)|@binding\s*\(\s*(\d+)[ui]?\s*\)\s*@group\s*\(\s*0[ui]?\s*\))\s*var(?:\s*<\s*([^>]+)\s*>)?\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^;]+);/g;

function classifyWgslTextureDim(typePrefix) {
  if (typePrefix.endsWith('_2d_array')) return '2d-array';
  if (typePrefix.endsWith('_cube_array')) return 'cube-array';
  if (typePrefix.endsWith('_cube')) return 'cube';
  if (typePrefix.endsWith('_3d')) return '3d';
  if (typePrefix.endsWith('_1d')) return '1d';
  if (typePrefix.endsWith('_2d') || typePrefix === 'texture_external') return '2d';
  return '2d';
}

function extractWgslFunctionBodies(cleaned) {
  const fnRe = /\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*\(/g;
  let bodies = '';
  let foundFn = false;
  let match;
  while ((match = fnRe.exec(cleaned)) !== null) {
    let i = match.index + match[0].length;
    let parenDepth = 1;
    while (i < cleaned.length && parenDepth > 0) {
      if (cleaned[i] === '(') parenDepth += 1;
      else if (cleaned[i] === ')') parenDepth -= 1;
      i += 1;
    }
    while (i < cleaned.length && cleaned[i] !== '{') {
      if (cleaned[i] === ';') break;
      i += 1;
    }
    if (i >= cleaned.length || cleaned[i] !== '{') continue;
    foundFn = true;
    const bodyStart = i + 1;
    let braceDepth = 1;
    i += 1;
    while (i < cleaned.length && braceDepth > 0) {
      if (cleaned[i] === '{') braceDepth += 1;
      else if (cleaned[i] === '}') braceDepth -= 1;
      i += 1;
    }
    bodies += cleaned.slice(bodyStart, Math.max(bodyStart, i - 1)) + '\n';
    fnRe.lastIndex = i;
  }
  return { hasFunction: foundFn, bodies };
}

// Parses statically referenced @group(0) @binding(N) (or @binding(N) @group(0)) var declarations
// in WGSL source, returning entries sorted and deduplicated by binding index.
function parseWgslGroup0Bindings(wgslSource) {
  if (typeof wgslSource !== 'string' || wgslSource.length === 0) return [];
  const cached = _parsedWgslGroup0Cache.get(wgslSource);
  if (cached !== undefined) return cached;
  const cleaned = stripWgslComments(wgslSource);
  if (cleaned.length === 0) return [];

  const matches = [];
  WGSL_GROUP0_VAR_RE.lastIndex = 0;
  let match;
  while ((match = WGSL_GROUP0_VAR_RE.exec(cleaned)) !== null) {
    const bindingStr = match[1] !== undefined ? match[1] : match[2];
    const binding = parseInt(bindingStr, 10);
    if (!Number.isInteger(binding) || binding < 0) continue;
    matches.push({
      binding,
      addrRaw: match[3] ? match[3].trim() : '',
      varName: match[4],
      typeRaw: match[5].trim(),
    });
  }
  if (matches.length === 0) return [];

  const { hasFunction, bodies: usageSource } = extractWgslFunctionBodies(cleaned);

  const byBinding = new Map();
  for (let i = 0; i < matches.length; i += 1) {
    const m = matches[i];
    if (byBinding.has(m.binding)) continue;
    if (hasFunction) {
      const refRe = new RegExp(`(?:^|[^A-Za-z0-9_.])${m.varName}(?![A-Za-z0-9_])`);
      if (!refRe.test(usageSource)) continue;
    }

    let entry = null;
    if (m.addrRaw.length > 0) {
      const parts = m.addrRaw.split(',').map((s) => s.trim().toLowerCase());
      if (parts[0] === 'uniform') {
        entry = { binding: m.binding, kind: 'uniform' };
      } else if (parts[0] === 'storage') {
        entry = {
          binding: m.binding,
          kind: parts[1] === 'read' ? 'read-only-storage' : 'storage',
        };
      }
    } else if (/^sampler_comparison\b/.test(m.typeRaw)) {
      entry = { binding: m.binding, kind: 'sampler', comparison: true };
    } else if (/^sampler\b/.test(m.typeRaw)) {
      entry = { binding: m.binding, kind: 'sampler' };
    } else if (/^texture_storage_/.test(m.typeRaw)) {
      const prefixMatch = /^(texture_storage_[A-Za-z0-9_]+)/.exec(m.typeRaw);
      const prefix = prefixMatch ? prefixMatch[1] : 'texture_storage_2d';
      const fmtMatch = /^texture_storage_[A-Za-z0-9_]+\s*<\s*([A-Za-z0-9_-]+)/.exec(m.typeRaw);
      entry = {
        binding: m.binding,
        kind: 'storage-texture',
        viewDimension: classifyWgslTextureDim(prefix),
      };
      if (fmtMatch && fmtMatch[1]) {
        entry.storageFormat = fmtMatch[1];
      }
    } else if (/^texture_/.test(m.typeRaw)) {
      const prefixMatch = /^(texture_[A-Za-z0-9_]+)/.exec(m.typeRaw);
      const prefix = prefixMatch ? prefixMatch[1] : 'texture_2d';
      entry = {
        binding: m.binding,
        kind: 'texture',
        viewDimension: classifyWgslTextureDim(prefix),
      };
      if (prefix.includes('_multisampled_')) {
        entry.multisampled = true;
      }
      if (prefix.startsWith('texture_depth_')) {
        entry.sampleType = 'depth';
      } else {
        const genericMatch = /<\s*(u32|i32|f32)\s*>/.exec(m.typeRaw);
        if (genericMatch) {
          if (genericMatch[1] === 'u32') entry.sampleType = 'uint';
          else if (genericMatch[1] === 'i32') entry.sampleType = 'sint';
        }
      }
    }

    if (entry) {
      byBinding.set(m.binding, entry);
    }
  }

  const result = Array.from(byBinding.values()).sort((a, b) => a.binding - b.binding);
  _setBoundedCache(_parsedWgslGroup0Cache, wgslSource, result);
  return result;
}

function mergeWgslGroup0Bindings(vsBindings, fsBindings) {
  const byBinding = new Map();
  if (Array.isArray(vsBindings)) {
    for (let i = 0; i < vsBindings.length; i += 1) {
      byBinding.set(vsBindings[i].binding, vsBindings[i]);
    }
  }
  if (Array.isArray(fsBindings)) {
    for (let i = 0; i < fsBindings.length; i += 1) {
      const fb = fsBindings[i];
      const existing = byBinding.get(fb.binding);
      if (!existing) {
        byBinding.set(fb.binding, fb);
      } else if (existing.kind === 'read-only-storage' && fb.kind === 'storage') {
        byBinding.set(fb.binding, fb);
      }
    }
  }
  return Array.from(byBinding.values()).sort((a, b) => a.binding - b.binding);
}

// Checks whether an explicit @group(0) BindGroupLayout's entries satisfy every binding declared
// and statically referenced by a WGSL shader (or VS+FS pair).
function isExplicitLayoutCompatibleWithWgsl(group0Entries, wgslBindings) {
  if (!Array.isArray(wgslBindings) || wgslBindings.length === 0) return true;
  if (!Array.isArray(group0Entries)) return true;

  for (let i = 0; i < wgslBindings.length; i += 1) {
    const b = wgslBindings[i];
    const e = group0Entries.find((x) => x && x.binding === b.binding);
    if (!e) return false;
    switch (b.kind) {
      case 'uniform':
        if (!(e.buffer && (!e.buffer.type || e.buffer.type === 'uniform'))) return false;
        break;
      case 'read-only-storage':
        if (!(e.buffer && (e.buffer.type === 'read-only-storage' || e.buffer.type === 'storage'))) {
          return false;
        }
        break;
      case 'storage':
        if (!(e.buffer && e.buffer.type === 'storage')) return false;
        break;
      case 'texture':
        if (!e.texture) return false;
        break;
      case 'storage-texture':
        if (!e.storageTexture) return false;
        break;
      case 'sampler':
        if (!e.sampler) return false;
        break;
      default:
        return false;
    }
  }
  return true;
}

function isBufferBindingResource(res) {
  return Boolean(res && typeof res === 'object' && 'buffer' in res && res.buffer);
}

function isNonBufferBindingResource(res) {
  return Boolean(res && !(typeof res === 'object' && 'buffer' in res));
}

function isSamplerResource(res) {
  return Boolean(
    res &&
      ((typeof GPUSampler === 'function' && res instanceof GPUSampler) ||
        (typeof res === 'object' && res.kind === 'GPUSampler'))
  );
}

function isTextureViewResource(res) {
  return Boolean(
    res &&
      ((typeof GPUTextureView === 'function' && res instanceof GPUTextureView) ||
        (typeof res === 'object' && res.kind === 'GPUTextureView'))
  );
}

function isRtArenaBinding(binding) {
  return binding >= RT_ARENA_BINDING_MIN && binding <= RT_ARENA_BINDING_MAX;
}

// W101/W354: normalize a setBindGroup dynamicOffsets argument (Array / typed array / null) into a
// plain array of uint32 offsets; empty/invalid input returns shared frozen EMPTY_DYNAMIC_OFFSETS.
function normalizeDynamicOffsets(raw) {
  if (!raw || !(Array.isArray(raw) || ArrayBuffer.isView(raw)) || raw.length === 0) {
    return EMPTY_DYNAMIC_OFFSETS;
  }
  const len = raw.length;
  const out = new Array(len);
  for (let i = 0; i < len; i += 1) {
    const v = Number(raw[i]);
    out[i] = Number.isFinite(v) && v > 0 ? Math.floor(v) >>> 0 : 0;
  }
  return out;
}

// W101: WebGPU orders setBindGroup dynamicOffsets by ascending layout binding number over the
// entries with buffer.hasDynamicOffset. Returns Map(binding -> dynamicOffsetIndex).
function dynamicOffsetIndicesForLayoutEntries(entries) {
  const indices = new Map();
  if (!Array.isArray(entries)) return indices;
  const dynBindings = entries
    .filter((e) => e && e.buffer && e.buffer.hasDynamicOffset)
    .map((e) => e.binding)
    .sort((a, b) => a - b);
  for (let i = 0; i < dynBindings.length; i += 1) indices.set(dynBindings[i], i);
  return indices;
}

function synthLayoutEntryForWgslBinding(b) {
  if (!b) return null;
  if (b.kind === 'uniform') {
    return {
      binding: b.binding,
      buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 4096 },
    };
  }
  if (b.kind === 'read-only-storage') {
    return {
      binding: b.binding,
      buffer: { type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 4096 },
    };
  }
  if (b.kind === 'storage') {
    return {
      binding: b.binding,
      buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 4096 },
    };
  }
  if (b.kind === 'sampler') {
    return {
      binding: b.binding,
      sampler: { type: b.comparison ? 'comparison' : 'filtering' },
    };
  }
  if (b.kind === 'storage-texture') {
    return {
      binding: b.binding,
      storageTexture: {
        access: 'write-only',
        format: b.storageFormat || 'rgba8unorm',
        viewDimension: b.viewDimension || '2d',
      },
    };
  }
  return {
    binding: b.binding,
    texture: {
      sampleType: b.sampleType || 'float',
      viewDimension: b.viewDimension || '2d',
      multisampled: Boolean(b.multisampled),
    },
  };
}

// Resolves the WebGPU shader stage entry point: defaults to vs_main/fs_main/cs_main per
// docs/gpu_commands.md, while honoring explicit entryPoint metadata and WGSL source declarations
// (such as legacy `@fragment fn ps_main` from older packs).
function resolveShaderEntryPoint(shaderEntry, stage) {
  const defaultEntryPoint = DEFAULT_ENTRY_POINTS[stage] || 'vs_main';
  const explicit =
    normalizeEntryPointName(shaderEntry && shaderEntry.entryPoint) ||
    normalizeEntryPointName(shaderEntry && shaderEntry.entry_point);
  const sourceText = getShaderSourceText(shaderEntry);
  const declared = sourceText ? findWgslStageEntryPoints(sourceText, stage) : [];

  let entryPoint = defaultEntryPoint;
  let reason = 'default';

  if (declared.length > 0) {
    if (explicit && declared.includes(explicit)) {
      entryPoint = explicit;
      reason = explicit === defaultEntryPoint ? 'default' : 'metadata+source';
    } else if (declared.includes(defaultEntryPoint)) {
      entryPoint = defaultEntryPoint;
      reason = 'default';
    } else {
      entryPoint = declared[0];
      reason = explicit ? 'source-override' : 'source-inspect';
    }
  } else if (explicit) {
    entryPoint = explicit;
    reason = explicit === defaultEntryPoint ? 'default' : 'metadata';
  }

  return {
    entryPoint,
    defaultEntryPoint,
    remapped: entryPoint !== defaultEntryPoint,
    reason,
  };
}

// Validates decoded GPURenderPipelineDescriptor.vertex.buffers against WebGPU layout rules.
function validateVertexBufferLayouts(buffers, device = null) {
  if (!Array.isArray(buffers)) {
    return { valid: false, error: 'vertex buffers must be an array' };
  }
  const maxStride = getDeviceLimit(
    device,
    'maxVertexBufferArrayStride',
    DEFAULT_MAX_VERTEX_BUFFER_ARRAY_STRIDE
  );
  const usedLocations = new Set();
  for (let i = 0; i < buffers.length; i += 1) {
    const slot = buffers[i];
    if (slot === null) continue;
    if (!slot || typeof slot !== 'object') {
      return { valid: false, error: `vertex buffer slot ${i}: invalid layout object` };
    }
    if (slot.stepMode !== 'vertex' && slot.stepMode !== 'instance') {
      return { valid: false, error: `vertex buffer slot ${i}: invalid stepMode ${String(slot.stepMode)}` };
    }
    if (
      !Number.isInteger(slot.arrayStride) ||
      slot.arrayStride < 0 ||
      slot.arrayStride % 4 !== 0 ||
      slot.arrayStride > maxStride
    ) {
      return { valid: false, error: `vertex buffer slot ${i}: invalid arrayStride ${slot.arrayStride}` };
    }
    if (!Array.isArray(slot.attributes)) {
      return { valid: false, error: `vertex buffer slot ${i}: attributes must be an array` };
    }
    for (let j = 0; j < slot.attributes.length; j += 1) {
      const attr = slot.attributes[j];
      if (!attr || typeof attr !== 'object') {
        return { valid: false, error: `vertex buffer slot ${i} attr ${j}: invalid attribute` };
      }
      const byteSize = VERTEX_FORMAT_BYTE_SIZES[attr.format];
      if (!byteSize) {
        return { valid: false, error: `vertex buffer slot ${i} attr ${j}: invalid format ${String(attr.format)}` };
      }
      if (!Number.isInteger(attr.offset) || attr.offset < 0) {
        return { valid: false, error: `vertex buffer slot ${i} attr ${j}: invalid offset ${attr.offset}` };
      }
      const offsetAlign = Math.min(4, byteSize);
      if (attr.offset % offsetAlign !== 0) {
        return {
          valid: false,
          error: `vertex buffer slot ${i} attr ${j}: offset ${attr.offset} not aligned to ${offsetAlign}`,
        };
      }
      if (!Number.isInteger(attr.shaderLocation) || attr.shaderLocation < 0 || attr.shaderLocation >= 16) {
        return {
          valid: false,
          error: `vertex buffer slot ${i} attr ${j}: invalid shaderLocation ${attr.shaderLocation}`,
        };
      }
      if (usedLocations.has(attr.shaderLocation)) {
        return {
          valid: false,
          error: `vertex buffer slot ${i} attr ${j}: duplicate shaderLocation ${attr.shaderLocation}`,
        };
      }
      usedLocations.add(attr.shaderLocation);
      if (slot.arrayStride > 0 && attr.offset + byteSize > slot.arrayStride) {
        return {
          valid: false,
          error: `vertex buffer slot ${i} attr ${j}: offset ${attr.offset} + ${byteSize} exceeds arrayStride ${slot.arrayStride}`,
        };
      }
      if (slot.arrayStride === 0 && attr.offset + byteSize > maxStride) {
        return {
          valid: false,
          error: `vertex buffer slot ${i} attr ${j}: offset ${attr.offset} + ${byteSize} exceeds maxVertexBufferArrayStride ${maxStride}`,
        };
      }
    }
  }
  return { valid: true };
}

function nameOf(table, code) {
  return code < table.length ? table[code] : undefined;
}

function finiteOr(value, fallback) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function isValidTargetDims(width, height) {
  return (
    typeof width === 'number' &&
    typeof height === 'number' &&
    Number.isFinite(width) &&
    Number.isFinite(height) &&
    width > 0 &&
    height > 0
  );
}

function validTargetSize(width, height, out = null) {
  if (isValidTargetDims(width, height)) {
    if (out) {
      out.width = width;
      out.height = height;
      return out;
    }
    return { width, height };
  }
  return null;
}

// Clamps a guest viewport to the active render target bounds and WebGPU [0, 1] depth range.
// When `out` is provided, writes directly into `out` with zero heap allocations (safe when `out === viewport`).
function clampViewportToTarget(viewport, targetSize, out = null) {
  const rawX = finiteOr(viewport && viewport.x, 0);
  const rawY = finiteOr(viewport && viewport.y, 0);
  const rawW = finiteOr(viewport && viewport.width, 0);
  const rawH = finiteOr(viewport && viewport.height, 0);

  let minDepth = Math.min(1, Math.max(0, finiteOr(viewport && viewport.minDepth, 0)));
  let maxDepth = Math.min(1, Math.max(0, finiteOr(viewport && viewport.maxDepth, 1)));
  if (minDepth > maxDepth) {
    const tmp = minDepth;
    minDepth = maxDepth;
    maxDepth = tmp;
  }
  minDepth = minDepth || 0;
  maxDepth = maxDepth || 0;

  const hasTarget = Boolean(targetSize && isValidTargetDims(targetSize.width, targetSize.height));
  if (!hasTarget) {
    const safeX = Number.isFinite(Math.fround(rawX)) ? Math.max(0, rawX) || 0 : 0;
    const safeY = Number.isFinite(Math.fround(rawY)) ? Math.max(0, rawY) || 0 : 0;
    const safeW = Number.isFinite(Math.fround(rawW)) ? Math.max(0, rawW) || 0 : 0;
    const safeH = Number.isFinite(Math.fround(rawH)) ? Math.max(0, rawH) || 0 : 0;
    if (out) {
      out.x = safeX;
      out.y = safeY;
      out.width = safeW;
      out.height = safeH;
      out.minDepth = minDepth;
      out.maxDepth = maxDepth;
      return out;
    }
    return {
      x: safeX,
      y: safeY,
      width: safeW,
      height: safeH,
      minDepth,
      maxDepth,
    };
  }

  const targetWidth = Math.floor(targetSize.width);
  const targetHeight = Math.floor(targetSize.height);
  if (targetWidth <= 0 || targetHeight <= 0) {
    const safeX = Number.isFinite(Math.fround(rawX)) ? Math.max(0, rawX) || 0 : 0;
    const safeY = Number.isFinite(Math.fround(rawY)) ? Math.max(0, rawY) || 0 : 0;
    const safeW = Number.isFinite(Math.fround(rawW)) ? Math.max(0, rawW) || 0 : 0;
    const safeH = Number.isFinite(Math.fround(rawH)) ? Math.max(0, rawH) || 0 : 0;
    if (out) {
      out.x = safeX;
      out.y = safeY;
      out.width = safeW;
      out.height = safeH;
      out.minDepth = minDepth;
      out.maxDepth = maxDepth;
      return out;
    }
    return { x: safeX, y: safeY, width: safeW, height: safeH, minDepth, maxDepth };
  }
  const f32RawX = Math.fround(rawX);
  const f32RawY = Math.fround(rawY);
  const f32RawW = Math.fround(rawW);
  const f32RawH = Math.fround(rawH);
  if (
    rawX >= 0 &&
    rawY >= 0 &&
    rawW >= 0 &&
    rawH >= 0 &&
    rawX + rawW <= targetWidth &&
    rawY + rawH <= targetHeight &&
    f32RawX + f32RawW <= targetWidth &&
    f32RawY + f32RawH <= targetHeight &&
    Math.fround(f32RawX + f32RawW) <= targetWidth &&
    Math.fround(f32RawY + f32RawH) <= targetHeight
  ) {
    const x = rawX || 0;
    const y = rawY || 0;
    const width = rawW || 0;
    const height = rawH || 0;
    if (out) {
      out.x = x;
      out.y = y;
      out.width = width;
      out.height = height;
      out.minDepth = minDepth;
      out.maxDepth = maxDepth;
      return out;
    }
    return {
      x,
      y,
      width,
      height,
      minDepth,
      maxDepth,
    };
  }

  let x = Math.min(Math.max(0, rawX), targetWidth) || 0;
  if (Math.fround(x) > targetWidth) {
    x = Math.max(0, Math.floor(targetWidth)) || 0;
  }
  let y = Math.min(Math.max(0, rawY), targetHeight) || 0;
  if (Math.fround(y) > targetHeight) {
    y = Math.max(0, Math.floor(targetHeight)) || 0;
  }
  let width = Math.min(Math.max(0, rawW), targetWidth - x) || 0;
  let height = Math.min(Math.max(0, rawH), targetHeight - y) || 0;
  if (
    x + width > targetWidth ||
    Math.fround(x) + Math.fround(width) > targetWidth ||
    Math.fround(Math.fround(x) + Math.fround(width)) > targetWidth
  ) {
    width = Math.max(0, Math.floor(targetWidth - Math.fround(x))) || 0;
  }
  if (
    y + height > targetHeight ||
    Math.fround(y) + Math.fround(height) > targetHeight ||
    Math.fround(Math.fround(y) + Math.fround(height)) > targetHeight
  ) {
    height = Math.max(0, Math.floor(targetHeight - Math.fround(y))) || 0;
  }
  if (out) {
    out.x = x;
    out.y = y;
    out.width = width;
    out.height = height;
    out.minDepth = minDepth;
    out.maxDepth = maxDepth;
    return out;
  }
  return { x, y, width, height, minDepth, maxDepth };
}

function isViewportClamped(raw, clamped) {
  if (!raw || typeof raw !== 'object') return true;
  return (
    !Number.isFinite(raw.x) ||
    !Number.isFinite(raw.y) ||
    !Number.isFinite(raw.width) ||
    !Number.isFinite(raw.height) ||
    !Number.isFinite(raw.minDepth) ||
    !Number.isFinite(raw.maxDepth) ||
    clamped.x !== raw.x ||
    clamped.y !== raw.y ||
    clamped.width !== raw.width ||
    clamped.height !== raw.height ||
    clamped.minDepth !== raw.minDepth ||
    clamped.maxDepth !== raw.maxDepth
  );
}

const MAX_SCISSOR_UINT32 = 0xffffffff;

// Clamps a guest integer scissor rect to [0..targetWidth] x [0..targetHeight].
// When `out` is provided, writes directly into `out` with zero heap allocations (safe when `out === scissor`).
function clampScissorToTarget(scissor, targetSize, out = null) {
  const rawX = Math.min(MAX_SCISSOR_UINT32, Math.max(0, Math.floor(finiteOr(scissor && scissor.x, 0)))) || 0;
  const rawY = Math.min(MAX_SCISSOR_UINT32, Math.max(0, Math.floor(finiteOr(scissor && scissor.y, 0)))) || 0;
  const rawW = Math.min(MAX_SCISSOR_UINT32, Math.max(0, Math.floor(finiteOr(scissor && scissor.width, 0)))) || 0;
  const rawH = Math.min(MAX_SCISSOR_UINT32, Math.max(0, Math.floor(finiteOr(scissor && scissor.height, 0)))) || 0;

  const hasTarget = Boolean(targetSize && isValidTargetDims(targetSize.width, targetSize.height));
  if (!hasTarget) {
    if (out) {
      out.x = rawX;
      out.y = rawY;
      out.width = rawW;
      out.height = rawH;
      return out;
    }
    return { x: rawX, y: rawY, width: rawW, height: rawH };
  }

  const targetWidth = Math.floor(targetSize.width);
  const targetHeight = Math.floor(targetSize.height);
  if (targetWidth <= 0 || targetHeight <= 0) {
    if (out) {
      out.x = rawX;
      out.y = rawY;
      out.width = rawW;
      out.height = rawH;
      return out;
    }
    return { x: rawX, y: rawY, width: rawW, height: rawH };
  }

  if (rawX + rawW <= targetWidth && rawY + rawH <= targetHeight) {
    if (out) {
      out.x = rawX;
      out.y = rawY;
      out.width = rawW;
      out.height = rawH;
      return out;
    }
    return { x: rawX, y: rawY, width: rawW, height: rawH };
  }

  const x = Math.min(rawX, targetWidth) || 0;
  const y = Math.min(rawY, targetHeight) || 0;
  const width = Math.min(rawW, targetWidth - x) || 0;
  const height = Math.min(rawH, targetHeight - y) || 0;
  if (out) {
    out.x = x;
    out.y = y;
    out.width = width;
    out.height = height;
    return out;
  }
  return { x, y, width, height };
}

function isScissorClamped(raw, clamped) {
  if (!raw || typeof raw !== 'object') return true;
  return (
    !Number.isFinite(raw.x) ||
    !Number.isFinite(raw.y) ||
    !Number.isFinite(raw.width) ||
    !Number.isFinite(raw.height) ||
    clamped.x !== raw.x ||
    clamped.y !== raw.y ||
    clamped.width !== raw.width ||
    clamped.height !== raw.height
  );
}

function resolveAttachmentViewSize(view, explicitSize) {
  if (explicitSize) {
    const valid = validTargetSize(explicitSize.width, explicitSize.height);
    if (valid) return valid;
  }
  if (!view || typeof view !== 'object') return null;
  const fromView = validTargetSize(view.width, view.height);
  if (fromView) return fromView;
  if (view.texture && typeof view.texture === 'object') {
    const fromTex = validTargetSize(view.texture.width, view.texture.height);
    if (fromTex) {
      const rawMip =
        typeof view.baseMipLevel === 'number'
          ? view.baseMipLevel
          : view.desc && typeof view.desc.baseMipLevel === 'number'
            ? view.desc.baseMipLevel
            : view.viewDesc && typeof view.viewDesc.baseMipLevel === 'number'
              ? view.viewDesc.baseMipLevel
              : 0;
      const mip = Number.isFinite(rawMip) && rawMip > 0 ? Math.floor(rawMip) : 0;
      if (mip > 0) {
        const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
        return {
          width: scale === Infinity ? 1 : Math.max(1, Math.floor(fromTex.width / scale)),
          height: scale === Infinity ? 1 : Math.max(1, Math.floor(fromTex.height / scale)),
        };
      }
      return fromTex;
    }
  }
  return null;
}

function resolveAttachmentSampleCount(view, explicitSampleCount) {
  if (typeof explicitSampleCount === 'number' && Number.isFinite(explicitSampleCount) && explicitSampleCount >= 1) {
    return Math.floor(explicitSampleCount);
  }
  if (!view || typeof view !== 'object') return null;
  if (typeof view.sampleCount === 'number' && Number.isFinite(view.sampleCount) && view.sampleCount >= 1) {
    return Math.floor(view.sampleCount);
  }
  if (view.texture && typeof view.texture === 'object') {
    if (typeof view.texture.sampleCount === 'number' && view.texture.sampleCount >= 1) {
      return Math.floor(view.texture.sampleCount);
    }
    if (view.texture.desc && typeof view.texture.desc.sampleCount === 'number' && view.texture.desc.sampleCount >= 1) {
      return Math.floor(view.texture.desc.sampleCount);
    }
  }
  return null;
}

function resolveAttachmentFormat(view, explicitFormat) {
  if (typeof explicitFormat === 'string' && explicitFormat.length > 0 && explicitFormat !== 'undefined') {
    return explicitFormat;
  }
  if (!view || typeof view !== 'object') return null;
  if (typeof view.format === 'string' && view.format.length > 0 && view.format !== 'undefined') {
    return view.format;
  }
  if (view.viewDesc && typeof view.viewDesc.format === 'string' && view.viewDesc.format !== 'undefined') {
    return view.viewDesc.format;
  }
  if (view.texture && typeof view.texture === 'object') {
    if (typeof view.texture.format === 'string' && view.texture.format !== 'undefined') {
      return view.texture.format;
    }
    if (view.texture.desc && typeof view.texture.desc.format === 'string' && view.texture.desc.format !== 'undefined') {
      return view.texture.desc.format;
    }
  }
  return null;
}

function resolveAttachmentUsage(view, explicitUsage) {
  if (typeof explicitUsage === 'number' && Number.isFinite(explicitUsage)) return explicitUsage;
  if (!view || typeof view !== 'object') return null;
  if (view.texture && typeof view.texture === 'object') {
    if (typeof view.texture.usage === 'number') return view.texture.usage;
    if (view.texture.desc && typeof view.texture.desc.usage === 'number') return view.texture.desc.usage;
  }
  return null;
}

// W266/W354 + Dawn parity: normalizes a GPURenderPassDescriptor so all attachments share consistent
// (width, height) and sampleCount, validates resolveTarget invariants, sanitizes color clearValue,
// and enforces depth/stencil aspect + readOnly vs loadOp/storeOp/clearValue rules before Dawn sees
// the pass descriptor. Avoids .map()/.slice()/array allocations on the common clean path.
function sanitizeRenderPassAttachments(descriptor, attachmentSizes = null) {
  if (!descriptor || typeof descriptor !== 'object') {
    return { descriptor, referenceSize: null, sanitizedCount: 0, warnings: EMPTY_WARNINGS };
  }
  const colorSizesInput =
    attachmentSizes && Array.isArray(attachmentSizes.colors) ? attachmentSizes.colors : null;
  const colorSampleCountsInput =
    attachmentSizes && Array.isArray(attachmentSizes.colorSampleCounts)
      ? attachmentSizes.colorSampleCounts
      : null;
  const colorFormatsInput =
    attachmentSizes && Array.isArray(attachmentSizes.colorFormats)
      ? attachmentSizes.colorFormats
      : null;
  const rawColors = Array.isArray(descriptor.colorAttachments) ? descriptor.colorAttachments : null;
  const colorCount = rawColors ? rawColors.length : 0;

  let referenceSize = null;
  let referenceSampleCount = null;
  // Pass 1: locate referenceSize and referenceSampleCount from the first non-null attachment.
  for (let i = 0; i < colorCount; i += 1) {
    const att = rawColors[i];
    if (!att) continue;
    if (!referenceSize) {
      const sz = resolveAttachmentViewSize(att.view, colorSizesInput ? colorSizesInput[i] || null : null);
      if (sz) referenceSize = sz;
    }
    if (referenceSampleCount === null) {
      const sc = resolveAttachmentSampleCount(
        att.view,
        colorSampleCountsInput ? colorSampleCountsInput[i] : null
      );
      if (sc !== null) referenceSampleCount = sc;
    }
    if (referenceSize && referenceSampleCount !== null) break;
  }
  const rawDs = descriptor.depthStencilAttachment || null;
  const dsSize = rawDs
    ? resolveAttachmentViewSize(rawDs.view, attachmentSizes ? attachmentSizes.depthStencil : null)
    : null;
  const dsSampleCount = rawDs
    ? resolveAttachmentSampleCount(rawDs.view, attachmentSizes ? attachmentSizes.depthSampleCount : null)
    : null;
  if (!referenceSize && dsSize) {
    referenceSize = dsSize;
  }
  if (referenceSampleCount === null && dsSampleCount !== null) {
    referenceSampleCount = dsSampleCount;
  }

  let sanitizedCount = 0;
  let warnings = null;
  let outColors = null;
  const pushWarning = (key, message) => {
    if (!warnings) warnings = [];
    warnings.push({ key, message });
  };

  for (let i = 0; i < colorCount; i += 1) {
    const att = rawColors[i];
    if (!att) continue;
    const sz = resolveAttachmentViewSize(att.view, colorSizesInput ? colorSizesInput[i] || null : null);
    if (referenceSize && sz && (sz.width !== referenceSize.width || sz.height !== referenceSize.height)) {
      if (!outColors) outColors = rawColors.slice();
      outColors[i] = null;
      sanitizedCount += 1;
      pushWarning(
        `rp-color-size-mismatch:${i}:${sz.width}x${sz.height}:${referenceSize.width}x${referenceSize.height}`,
        `render pass: dropped colorAttachments[${i}] (${sz.width}x${sz.height}) mismatched with pass target (${referenceSize.width}x${referenceSize.height})`
      );
      continue;
    }
    const attSampleCount = resolveAttachmentSampleCount(
      att.view,
      colorSampleCountsInput ? colorSampleCountsInput[i] : null
    );
    if (
      referenceSampleCount !== null &&
      attSampleCount !== null &&
      attSampleCount !== referenceSampleCount
    ) {
      if (!outColors) outColors = rawColors.slice();
      outColors[i] = null;
      sanitizedCount += 1;
      pushWarning(
        `rp-color-sample-mismatch:${i}:${attSampleCount}:${referenceSampleCount}`,
        `render pass: dropped colorAttachments[${i}] (sampleCount=${attSampleCount}) mismatched with pass sampleCount=${referenceSampleCount}`
      );
      continue;
    }

    let cleanAtt = null;
    if (att.loadOp !== 'load' && att.loadOp !== 'clear') {
      cleanAtt = { ...att, loadOp: 'load' };
      sanitizedCount += 1;
      pushWarning(
        `rp-color-loadop:${i}`,
        `render pass: normalized invalid colorAttachments[${i}].loadOp to 'load'`
      );
    }
    const effLoadOp = cleanAtt ? cleanAtt.loadOp : att.loadOp;
    if (att.storeOp !== 'store' && att.storeOp !== 'discard') {
      if (!cleanAtt) cleanAtt = { ...att };
      cleanAtt.storeOp = 'store';
      sanitizedCount += 1;
      pushWarning(
        `rp-color-storeop:${i}`,
        `render pass: normalized invalid colorAttachments[${i}].storeOp to 'store'`
      );
    }
    if (effLoadOp === 'clear' && att.clearValue && typeof att.clearValue === 'object') {
      const cv = att.clearValue;
      const isArr = Array.isArray(cv);
      const r0 = isArr ? cv[0] : cv.r;
      const g0 = isArr ? cv[1] : cv.g;
      const b0 = isArr ? cv[2] : cv.b;
      const a0 = isArr ? cv[3] : cv.a;
      const r = finiteOr(r0, 0) || 0;
      const g = finiteOr(g0, 0) || 0;
      const b = finiteOr(b0, 0) || 0;
      const a = finiteOr(a0, 1) || 0;
      if (
        !Number.isFinite(r0) ||
        !Number.isFinite(g0) ||
        !Number.isFinite(b0) ||
        !Number.isFinite(a0)
      ) {
        if (!cleanAtt) cleanAtt = { ...att };
        cleanAtt.clearValue = { r, g, b, a };
        sanitizedCount += 1;
        pushWarning(
          `rp-color-clearvalue:${i}`,
          `render pass: sanitized non-finite colorAttachments[${i}].clearValue`
        );
      }
    } else if (effLoadOp !== 'clear' && att.clearValue !== undefined) {
      if (!cleanAtt) cleanAtt = { ...att };
      delete cleanAtt.clearValue;
      sanitizedCount += 1;
    }

    if (att.resolveTarget) {
      const resolveExplicit =
        attachmentSizes && Array.isArray(attachmentSizes.resolveTargets)
          ? attachmentSizes.resolveTargets[i] || null
          : null;
      const resolveSz = resolveAttachmentViewSize(att.resolveTarget, resolveExplicit);
      const resolveSc = resolveAttachmentSampleCount(att.resolveTarget, null);
      const viewFmt = resolveAttachmentFormat(att.view, colorFormatsInput ? colorFormatsInput[i] : null);
      const resolveFmt = resolveAttachmentFormat(att.resolveTarget, null);
      const resolveUsage = resolveAttachmentUsage(att.resolveTarget, null);
      let dropResolveReason = null;
      if (
        referenceSize &&
        resolveSz &&
        (resolveSz.width !== referenceSize.width || resolveSz.height !== referenceSize.height)
      ) {
        dropResolveReason = `size ${resolveSz.width}x${resolveSz.height} != ${referenceSize.width}x${referenceSize.height}`;
      } else if (attSampleCount !== null && attSampleCount <= 1) {
        dropResolveReason = `attachment is single-sampled (sampleCount=${attSampleCount})`;
      } else if (resolveSc !== null && resolveSc !== 1) {
        dropResolveReason = `resolveTarget must be single-sampled (sampleCount=${resolveSc})`;
      } else if (viewFmt && resolveFmt && viewFmt !== resolveFmt) {
        dropResolveReason = `format ${resolveFmt} != ${viewFmt}`;
      } else if (viewFmt && isIntegerTextureFormat(viewFmt)) {
        dropResolveReason = `integer format ${viewFmt} is not resolve-capable`;
      } else if (resolveUsage !== null && (resolveUsage & GPU_TEXTURE_USAGE_RENDER_ATTACHMENT) === 0) {
        dropResolveReason = 'resolveTarget lacks RENDER_ATTACHMENT usage';
      }
      if (dropResolveReason) {
        if (!cleanAtt) cleanAtt = { ...att };
        delete cleanAtt.resolveTarget;
        sanitizedCount += 1;
        pushWarning(
          `rp-resolve-invalid:${i}:${dropResolveReason}`,
          `render pass: dropped colorAttachments[${i}].resolveTarget (${dropResolveReason})`
        );
      }
    }

    if (cleanAtt) {
      if (!outColors) outColors = rawColors.slice();
      outColors[i] = cleanAtt;
    }
  }

  let outDs = rawDs;
  let dsChanged = false;
  if (rawDs) {
    const dsFormat = resolveAttachmentFormat(rawDs.view, attachmentSizes ? attachmentSizes.depthFormat : null);
    const dsUsage = resolveAttachmentUsage(rawDs.view, attachmentSizes ? attachmentSizes.depthUsage : null);
    if (referenceSize && dsSize && (dsSize.width !== referenceSize.width || dsSize.height !== referenceSize.height)) {
      outDs = null;
      dsChanged = true;
      sanitizedCount += 1;
      pushWarning(
        `rp-depth-size-mismatch:${dsSize.width}x${dsSize.height}:${referenceSize.width}x${referenceSize.height}`,
        `render pass: dropped depthStencilAttachment (${dsSize.width}x${dsSize.height}) mismatched with pass target (${referenceSize.width}x${referenceSize.height})`
      );
    } else if (
      referenceSampleCount !== null &&
      dsSampleCount !== null &&
      dsSampleCount !== referenceSampleCount
    ) {
      outDs = null;
      dsChanged = true;
      sanitizedCount += 1;
      pushWarning(
        `rp-depth-sample-mismatch:${dsSampleCount}:${referenceSampleCount}`,
        `render pass: dropped depthStencilAttachment (sampleCount=${dsSampleCount}) mismatched with pass sampleCount=${referenceSampleCount}`
      );
    } else if (dsFormat && !isDepthOrStencilTextureFormat(dsFormat)) {
      outDs = null;
      dsChanged = true;
      sanitizedCount += 1;
      pushWarning(
        `rp-depth-non-ds-format:${dsFormat}`,
        `render pass: dropped depthStencilAttachment with non-depth-stencil format ${dsFormat}`
      );
    } else if (dsUsage !== null && (dsUsage & GPU_TEXTURE_USAGE_RENDER_ATTACHMENT) === 0) {
      outDs = null;
      dsChanged = true;
      sanitizedCount += 1;
      pushWarning(
        'rp-depth-missing-ra-usage',
        'render pass: dropped depthStencilAttachment lacking RENDER_ATTACHMENT usage'
      );
    } else {
      let cleanDs = null;
      const ensureCleanDs = () => {
        if (!cleanDs) cleanDs = { ...rawDs };
        return cleanDs;
      };
      const hasDepth = dsFormat ? hasDepthAspect(dsFormat) : null;
      const hasStencil = dsFormat ? hasStencilAspect(dsFormat) : null;

      // Depth aspect rules:
      // 1. Stencil-only format (stencil8): forbid ALL depth fields.
      // 2. depthReadOnly: true -> forbid depthLoadOp / depthStoreOp / depthClearValue.
      // 3. hasDepth && !depthReadOnly -> require valid depthLoadOp ('load'|'clear') & depthStoreOp ('store'|'discard').
      if (hasDepth === false) {
        if (
          rawDs.depthLoadOp !== undefined ||
          rawDs.depthStoreOp !== undefined ||
          rawDs.depthClearValue !== undefined ||
          rawDs.depthReadOnly !== undefined
        ) {
          const dst = ensureCleanDs();
          delete dst.depthLoadOp;
          delete dst.depthStoreOp;
          delete dst.depthClearValue;
          delete dst.depthReadOnly;
          dsChanged = true;
          sanitizedCount += 1;
          pushWarning(
            `rp-depth-ops-on-stencil-only:${dsFormat}`,
            `render pass: stripped depth fields on stencil-only attachment (${dsFormat})`
          );
        }
      } else if (rawDs.depthReadOnly) {
        if (
          rawDs.depthLoadOp !== undefined ||
          rawDs.depthStoreOp !== undefined ||
          rawDs.depthClearValue !== undefined
        ) {
          const dst = ensureCleanDs();
          delete dst.depthLoadOp;
          delete dst.depthStoreOp;
          delete dst.depthClearValue;
          dsChanged = true;
          sanitizedCount += 1;
          pushWarning(
            'rp-depth-readonly-ops',
            'render pass: stripped depthLoadOp/depthStoreOp/depthClearValue on depthReadOnly attachment'
          );
        }
      } else {
        const hasAnyDepthOp =
          rawDs.depthLoadOp !== undefined ||
          rawDs.depthStoreOp !== undefined ||
          rawDs.depthClearValue !== undefined;
        if (hasDepth === true || hasAnyDepthOp) {
          const rawLoad = rawDs.depthLoadOp;
          const rawStore = rawDs.depthStoreOp;
          const normLoad = rawLoad === 'clear' || rawLoad === 'load' ? rawLoad : 'load';
          const normStore = rawStore === 'discard' || rawStore === 'store' ? rawStore : 'store';
          if (rawLoad !== normLoad || rawStore !== normStore) {
            const dst = ensureCleanDs();
            dst.depthLoadOp = normLoad;
            dst.depthStoreOp = normStore;
            dsChanged = true;
            sanitizedCount += 1;
            pushWarning(
              `rp-depth-ops-norm:${String(rawLoad)}:${String(rawStore)}`,
              `render pass: normalized depthLoadOp/depthStoreOp to '${normLoad}'/'${normStore}'`
            );
          }
          if (normLoad === 'clear') {
            const rawClear = rawDs.depthClearValue;
            const clampedClear = Math.min(1, Math.max(0, finiteOr(rawClear, 1))) || 0;
            if (!Number.isFinite(rawClear) || clampedClear !== rawClear) {
              const dst = ensureCleanDs();
              dst.depthClearValue = clampedClear;
              dsChanged = true;
              sanitizedCount += 1;
              pushWarning(
                `rp-depth-clear-clamp:${rawClear}`,
                `render pass: clamped depthClearValue ${rawClear} to ${clampedClear}`
              );
            }
          } else if (rawDs.depthClearValue !== undefined) {
            const dst = ensureCleanDs();
            delete dst.depthClearValue;
            dsChanged = true;
            sanitizedCount += 1;
          }
        }
      }

      // Stencil aspect rules (exact symmetric WebGPU invariants):
      // 1. Depth-only format (depth32float, depth24plus, depth16unorm): forbid ALL stencil fields.
      // 2. stencilReadOnly: true -> forbid stencilLoadOp / stencilStoreOp / stencilClearValue.
      // 3. hasStencil && !stencilReadOnly -> require valid stencilLoadOp & stencilStoreOp.
      if (hasStencil === false) {
        if (
          rawDs.stencilLoadOp !== undefined ||
          rawDs.stencilStoreOp !== undefined ||
          rawDs.stencilClearValue !== undefined ||
          rawDs.stencilReadOnly !== undefined
        ) {
          const dst = ensureCleanDs();
          delete dst.stencilLoadOp;
          delete dst.stencilStoreOp;
          delete dst.stencilClearValue;
          delete dst.stencilReadOnly;
          dsChanged = true;
          sanitizedCount += 1;
          pushWarning(
            `rp-stencil-ops-on-depth-only:${dsFormat}`,
            `render pass: stripped stencil fields on depth-only attachment (${dsFormat})`
          );
        }
      } else if (rawDs.stencilReadOnly) {
        if (
          rawDs.stencilLoadOp !== undefined ||
          rawDs.stencilStoreOp !== undefined ||
          rawDs.stencilClearValue !== undefined
        ) {
          const dst = ensureCleanDs();
          delete dst.stencilLoadOp;
          delete dst.stencilStoreOp;
          delete dst.stencilClearValue;
          dsChanged = true;
          sanitizedCount += 1;
          pushWarning(
            'rp-stencil-readonly-ops',
            'render pass: stripped stencilLoadOp/stencilStoreOp/stencilClearValue on stencilReadOnly attachment'
          );
        }
      } else {
        const hasAnyStencilOp =
          rawDs.stencilLoadOp !== undefined ||
          rawDs.stencilStoreOp !== undefined ||
          rawDs.stencilClearValue !== undefined;
        if (hasStencil === true || hasAnyStencilOp) {
          const rawSLoad = rawDs.stencilLoadOp;
          const rawSStore = rawDs.stencilStoreOp;
          const normSLoad = rawSLoad === 'clear' || rawSLoad === 'load' ? rawSLoad : 'load';
          const normSStore = rawSStore === 'discard' || rawSStore === 'store' ? rawSStore : 'store';
          if (rawSLoad !== normSLoad || rawSStore !== normSStore) {
            const dst = ensureCleanDs();
            dst.stencilLoadOp = normSLoad;
            dst.stencilStoreOp = normSStore;
            dsChanged = true;
            sanitizedCount += 1;
            pushWarning(
              `rp-stencil-ops-norm:${String(rawSLoad)}:${String(rawSStore)}`,
              `render pass: normalized stencilLoadOp/stencilStoreOp to '${normSLoad}'/'${normSStore}'`
            );
          }
          if (normSLoad === 'clear') {
            const rawSClear = rawDs.stencilClearValue;
            const normSClear = finiteOr(rawSClear, 0) >>> 0;
            if (rawSClear !== normSClear) {
              const dst = ensureCleanDs();
              dst.stencilClearValue = normSClear;
              dsChanged = true;
              sanitizedCount += 1;
            }
          } else if (rawDs.stencilClearValue !== undefined) {
            const dst = ensureCleanDs();
            delete dst.stencilClearValue;
            dsChanged = true;
            sanitizedCount += 1;
          }
        }
      }
      if (cleanDs) outDs = cleanDs;
    }
  }

  if (outColors === null && !dsChanged) {
    return { descriptor, referenceSize, sanitizedCount: 0, warnings: EMPTY_WARNINGS };
  }
  const outDescriptor = { ...descriptor, colorAttachments: outColors || rawColors || [] };
  if (outDs) {
    outDescriptor.depthStencilAttachment = outDs;
  } else {
    delete outDescriptor.depthStencilAttachment;
  }
  return { descriptor: outDescriptor, referenceSize, sanitizedCount, warnings: warnings || EMPTY_WARNINGS };
}

// ---- Decoders (pure; also used by the tests; W302 zero-alloc scratch reuse) ----

const VIEWPORT_PACKET_BYTES = 24;
const SCISSOR_PACKET_BYTES = 16;
const BLEND_CONSTANTS_PACKET_BYTES = 16;
const STENCIL_REF_PACKET_BYTES = 4;
const PRIMITIVE_TOPOLOGY_PACKET_BYTES = 5;
const SAMPLER2_PACKET_BYTES = 24;
const TEXTURE_VIEW_DESC_BYTES = 16;
const BIND_GROUP_LAYOUT_ENTRY_BYTES = 24;
const BEGIN_RENDER_PASS_BYTES = 368;

// Pre-allocated 4-byte DataView so f32FromBits performs zero heap allocations.
const _f32ScratchView = new DataView(new ArrayBuffer(4));

function f32FromBits(bits) {
  _f32ScratchView.setUint32(0, bits, true);
  return _f32ScratchView.getFloat32(0, true);
}

function attachNonEnumerablePool(obj, pool) {
  Object.defineProperty(obj, '_pool', {
    value: pool,
    enumerable: false,
    configurable: false,
  });
  return pool;
}

// W302: factory returning pre-allocated reusable scratch structs & typed buffers for all
// hot render-state packet decoders so callers (e.g. wgpu_worker.js) can decode thousands of
// per-frame state packets with zero per-packet object allocations.
function createRenderStateScratch() {
  const bindGroupLayoutEntry = { binding: 0, visibility: 0 };
  attachNonEnumerablePool(bindGroupLayoutEntry, {
    buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 0 },
    sampler: { type: 'filtering' },
    texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
    storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' },
  });

  const createTexture2 = {
    id: 0,
    descriptor: {
      size: { width: 1, height: 1, depthOrArrayLayers: 1 },
      mipLevelCount: 1,
      sampleCount: 1,
      dimension: '2d',
      format: 'rgba8unorm',
      usage: 0,
    },
  };
  attachNonEnumerablePool(createTexture2, {
    descriptor: createTexture2.descriptor,
    size: createTexture2.descriptor.size,
    viewFormats: [],
  });

  const beginRenderPassColors = new Array(8);
  for (let i = 0; i < 8; i += 1) {
    beginRenderPassColors[i] = {
      textureId: 0,
      loadOp: 'undefined',
      storeOp: 'undefined',
      view: {},
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
    };
  }
  const beginRenderPass = { colors: [], depthStencil: null };
  attachNonEnumerablePool(beginRenderPass, {
    colors: beginRenderPassColors,
    depthStencil: {
      textureId: 0,
      depthLoadOp: 0,
      depthStoreOp: 0,
      stencilLoadOp: 0,
      stencilStoreOp: 0,
      view: {},
      depthClearValue: 1,
      stencilClearValue: 0,
      depthReadOnly: false,
      stencilReadOnly: false,
    },
  });

  const pipelineTargets = new Array(8);
  const pipelineBlends = new Array(8);
  const pipelineBuffers = new Array(8);
  for (let i = 0; i < 8; i += 1) {
    pipelineTargets[i] = { format: 'rgba8unorm', writeMask: 0x0f };
    pipelineBlends[i] = {
      color: { operation: 'add', srcFactor: 'one', dstFactor: 'zero' },
      alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'zero' },
    };
    pipelineBuffers[i] = { arrayStride: 0, stepMode: 'vertex', attributes: [] };
  }
  const pipelineAttrs = new Array(16);
  for (let i = 0; i < 16; i += 1) {
    pipelineAttrs[i] = { format: 'float32', offset: 0, shaderLocation: 0 };
  }
  const renderPipelineKey = {
    primitive: { topology: 'triangle-list', frontFace: 'ccw', cullMode: 'none' },
    multisample: { count: 1, mask: 0xffffffff, alphaToCoverageEnabled: false },
    depthStencil: undefined,
    targets: [],
    buffers: [],
    colorTargetCount: 0,
  };
  attachNonEnumerablePool(renderPipelineKey, {
    targets: pipelineTargets,
    blends: pipelineBlends,
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: false,
      stencilFront: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
      stencilBack: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
      stencilReadMask: 0,
      stencilWriteMask: 0,
      depthBias: 0,
      depthBiasSlopeScale: 0,
      depthBiasClamp: 0,
    },
    buffers: pipelineBuffers,
    attributes: pipelineAttrs,
  });

  return {
    viewport: { x: 0, y: 0, width: 0, height: 0, minDepth: 0, maxDepth: 1 },
    viewportClamped: { x: 0, y: 0, width: 0, height: 0, minDepth: 0, maxDepth: 1 },
    scissor: { x: 0, y: 0, width: 0, height: 0 },
    scissorClamped: { x: 0, y: 0, width: 0, height: 0 },
    blendConstants: { r: 0, g: 0, b: 0, a: 0 },
    stencilRef: { reference: 0 },
    primitiveTopology: { topology: 'triangle-list', frontFace: 'ccw', cullMode: 'none' },
    blendComponent: { operation: 'add', srcFactor: 'one', dstFactor: 'zero' },
    stencilFace: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
    textureViewDesc: {},
    bindGroupLayoutEntry,
    setBindGroup2: { groupIndex: 0, bindGroupId: 0, dynamicOffsetCount: 0, dynamicOffsets: [] },
    sampler2: {
      id: 0,
      descriptor: {
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
        addressModeW: 'clamp-to-edge',
        magFilter: 'nearest',
        minFilter: 'nearest',
        mipmapFilter: 'nearest',
        lodMinClamp: 0,
        lodMaxClamp: 32,
        maxAnisotropy: 1,
      },
    },
    createTexture2,
    beginRenderPass,
    renderPipelineKey,
    computeDispatch: {
      skip: false,
      clamped: false,
      x: 0,
      y: 0,
      z: 0,
      maxWorkgroupsPerDimension: DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
      reason: null,
    },
    targetSize: { width: 0, height: 0 },
    viewportF32: new Float32Array(6),
    scissorU32: new Uint32Array(4),
    blendConstantsF32: new Float32Array(4),
    dynamicOffsetsU32: new Uint32Array(16),
  };
}

// Decodes a 24-byte SET_VIEWPORT (opcode 28) payload.
// Optionally writes into pre-allocated `out` and clamps to `targetSize` in-place.
function decodeViewportPacket(view, offset = 0, out = null, targetSize = null) {
  if (!view || offset < 0 || offset + VIEWPORT_PACKET_BYTES > view.byteLength) {
    return null;
  }
  const dst = out || { x: 0, y: 0, width: 0, height: 0, minDepth: 0, maxDepth: 1 };
  dst.x = view.getFloat32(offset, true);
  dst.y = view.getFloat32(offset + 4, true);
  dst.width = view.getFloat32(offset + 8, true);
  dst.height = view.getFloat32(offset + 12, true);
  dst.minDepth = view.getFloat32(offset + 16, true);
  dst.maxDepth = view.getFloat32(offset + 20, true);
  if (targetSize) {
    return clampViewportToTarget(dst, targetSize, dst);
  }
  return dst;
}

// Decodes a 16-byte SET_SCISSOR_RECT (opcode 29) payload.
// Optionally writes into pre-allocated `out` and clamps to `targetSize` in-place.
function decodeScissorPacket(view, offset = 0, out = null, targetSize = null) {
  if (!view || offset < 0 || offset + SCISSOR_PACKET_BYTES > view.byteLength) {
    return null;
  }
  const dst = out || { x: 0, y: 0, width: 0, height: 0 };
  dst.x = view.getUint32(offset, true);
  dst.y = view.getUint32(offset + 4, true);
  dst.width = view.getUint32(offset + 8, true);
  dst.height = view.getUint32(offset + 12, true);
  if (targetSize) {
    return clampScissorToTarget(dst, targetSize, dst);
  }
  return dst;
}

// Decodes a 16-byte SET_BLEND_CONSTANT (opcode 30) payload with W266 NaN/Infinity/-0 sanitization.
function decodeBlendConstantsPacket(view, offset = 0, out = null) {
  if (!view || offset < 0 || offset + BLEND_CONSTANTS_PACKET_BYTES > view.byteLength) {
    return null;
  }
  const dst = out || { r: 0, g: 0, b: 0, a: 0 };
  dst.r = finiteOr(view.getFloat32(offset, true), 0) || 0;
  dst.g = finiteOr(view.getFloat32(offset + 4, true), 0) || 0;
  dst.b = finiteOr(view.getFloat32(offset + 8, true), 0) || 0;
  dst.a = finiteOr(view.getFloat32(offset + 12, true), 0) || 0;
  return dst;
}

// Decodes a 4-byte SET_STENCIL_REFERENCE (opcode 31) payload.
function decodeStencilRefPacket(view, offset = 0, out = null) {
  if (!view || offset < 0 || offset + STENCIL_REF_PACKET_BYTES > view.byteLength) {
    return null;
  }
  const dst = out || { reference: 0 };
  dst.reference = view.getUint32(offset, true);
  return dst;
}

// Decodes the 5-byte primitive state block (topology, stripIndexFormat, frontFace, cullMode, unclippedDepth).
function decodePrimitiveTopologyPacket(view, offset = 0, out = null) {
  if (!view || offset < 0 || offset + PRIMITIVE_TOPOLOGY_PACKET_BYTES > view.byteLength) {
    return null;
  }
  const topology = nameOf(PRIMITIVE_TOPOLOGY_NAMES, view.getUint8(offset)) || 'triangle-list';
  const stripIndexFormat = view.getUint8(offset + 1);
  const frontFace = nameOf(FRONT_FACE_NAMES, view.getUint8(offset + 2)) || 'ccw';
  const cullMode = nameOf(CULL_MODE_NAMES, view.getUint8(offset + 3)) || 'none';
  const unclippedDepth = view.getUint8(offset + 4) !== 0;
  const dst = out || { topology, frontFace, cullMode };
  dst.topology = topology;
  dst.frontFace = frontFace;
  dst.cullMode = cullMode;
  if (stripIndexFormat !== 0 && (topology === 'triangle-strip' || topology === 'line-strip')) {
    const idxFormat = nameOf(INDEX_FORMAT_NAMES, stripIndexFormat);
    if (idxFormat && idxFormat !== 'undefined') {
      dst.stripIndexFormat = idxFormat;
    } else if ('stripIndexFormat' in dst) {
      delete dst.stripIndexFormat;
    }
  } else if ('stripIndexFormat' in dst) {
    delete dst.stripIndexFormat;
  }
  if (unclippedDepth) {
    dst.unclippedDepth = true;
  } else if ('unclippedDepth' in dst) {
    delete dst.unclippedDepth;
  }
  return dst;
}

// Decodes a SET_BIND_GROUP2 (opcode 33) payload (16-byte header + dynamicOffsetCount * 4 bytes).
// Supports optional `out` (as 4th arg, or 3rd arg when payloadBytes is omitted).
function decodeSetBindGroup2Packet(view, offset = 0, payloadBytes, out = null) {
  let targetOut = out;
  let explicitPayloadBytes = payloadBytes;
  if (typeof payloadBytes === 'object' && payloadBytes !== null && out === null) {
    targetOut = payloadBytes;
    explicitPayloadBytes = undefined;
  }
  if (!view || offset < 0) return null;
  const available =
    explicitPayloadBytes !== undefined ? explicitPayloadBytes : view.byteLength - offset;
  if (available < 16 || offset + 16 > view.byteLength) return null;
  const groupIndex = view.getUint32(offset, true);
  const bindGroupId = view.getUint32(offset + 4, true);
  const count = view.getUint32(offset + 8, true);
  const required = 16 + count * 4;
  if (required > available || offset + required > view.byteLength) return null;
  const dst = targetOut || {
    groupIndex: 0,
    bindGroupId: 0,
    dynamicOffsetCount: 0,
    dynamicOffsets: [],
  };
  dst.groupIndex = groupIndex;
  dst.bindGroupId = bindGroupId;
  dst.dynamicOffsetCount = count;
  if (!Array.isArray(dst.dynamicOffsets)) {
    dst.dynamicOffsets = [];
  }
  const offsets = dst.dynamicOffsets;
  offsets.length = count;
  for (let i = 0; i < count; i += 1) {
    offsets[i] = view.getUint32(offset + 16 + i * 4, true);
  }
  return dst;
}

// Decodes a 24-byte CREATE_SAMPLER2 (opcode 34) payload.
function decodeSampler2Packet(view, offset = 0, out = null) {
  if (!view || offset < 0 || offset + SAMPLER2_PACKET_BYTES > view.byteLength) {
    return null;
  }
  const id = view.getUint32(offset, true);
  const dst = out || {
    id: 0,
    descriptor: {
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
      magFilter: 'nearest',
      minFilter: 'nearest',
      mipmapFilter: 'nearest',
      lodMinClamp: 0,
      lodMaxClamp: 32,
      maxAnisotropy: 1,
    },
  };
  if (!dst.descriptor || typeof dst.descriptor !== 'object') {
    dst.descriptor = {};
  }
  const desc = dst.descriptor;
  desc.addressModeU = nameOf(ADDRESS_MODE_NAMES, view.getUint8(offset + 4)) || 'clamp-to-edge';
  desc.addressModeV = nameOf(ADDRESS_MODE_NAMES, view.getUint8(offset + 5)) || 'clamp-to-edge';
  desc.addressModeW = nameOf(ADDRESS_MODE_NAMES, view.getUint8(offset + 6)) || 'clamp-to-edge';
  desc.magFilter = nameOf(FILTER_MODE_NAMES, view.getUint8(offset + 7)) || 'nearest';
  desc.minFilter = nameOf(FILTER_MODE_NAMES, view.getUint8(offset + 8)) || 'nearest';
  desc.mipmapFilter = nameOf(FILTER_MODE_NAMES, view.getUint8(offset + 9)) || 'nearest';
  desc.lodMinClamp = view.getFloat32(offset + 16, true);
  desc.lodMaxClamp = view.getFloat32(offset + 20, true);
  desc.maxAnisotropy = Math.max(1, view.getUint16(offset + 12, true));
  const compare = view.getUint8(offset + 10);
  if (compare !== 0) {
    desc.compare = nameOf(COMPARE_FUNCTION_NAMES, compare);
  } else if ('compare' in desc) {
    delete desc.compare;
  }
  dst.id = id;
  return dst;
}

function decodeTextureViewDesc(view, offset = 0, out = null) {
  if (!view || offset < 0 || offset + TEXTURE_VIEW_DESC_BYTES > view.byteLength) {
    return null;
  }
  const desc = out || {};
  const format = view.getUint8(offset);
  const dimension = view.getUint8(offset + 1);
  const aspect = view.getUint8(offset + 2);
  const baseMipLevel = view.getUint16(offset + 4, true);
  const mipLevelCount = view.getUint16(offset + 6, true);
  const baseArrayLayer = view.getUint32(offset + 8, true);
  const arrayLayerCount = view.getUint32(offset + 12, true);
  if (format !== 0) desc.format = nameOf(TEXTURE_FORMAT_NAMES, format);
  else if (out && 'format' in desc) delete desc.format;
  if (dimension !== 0) desc.dimension = nameOf(VIEW_DIMENSION_NAMES, dimension);
  else if (out && 'dimension' in desc) delete desc.dimension;
  if (aspect !== 0) desc.aspect = nameOf(TEXTURE_ASPECT_NAMES, aspect);
  else if (out && 'aspect' in desc) delete desc.aspect;
  if (baseMipLevel !== 0) desc.baseMipLevel = baseMipLevel;
  else if (out && 'baseMipLevel' in desc) delete desc.baseMipLevel;
  if (mipLevelCount !== 0) desc.mipLevelCount = mipLevelCount;
  else if (out && 'mipLevelCount' in desc) delete desc.mipLevelCount;
  if (baseArrayLayer !== 0) desc.baseArrayLayer = baseArrayLayer;
  else if (out && 'baseArrayLayer' in desc) delete desc.baseArrayLayer;
  if (arrayLayerCount !== 0) desc.arrayLayerCount = arrayLayerCount;
  else if (out && 'arrayLayerCount' in desc) delete desc.arrayLayerCount;
  return desc;
}

function decodeBindGroupLayoutEntry(view, offset = 0, out = null) {
  if (!view || offset < 0 || offset + BIND_GROUP_LAYOUT_ENTRY_BYTES > view.byteLength) {
    return null;
  }
  const binding = view.getUint32(offset, true);
  let visibility = view.getUint32(offset + 4, true) & GPU_SHADER_STAGE_ALL_MASK;
  const type = view.getUint8(offset + 8);
  const subtype = view.getUint8(offset + 9);
  const dimension = view.getUint8(offset + 10);
  const flags = view.getUint8(offset + 11);
  const storageFormat = view.getUint8(offset + 12);
  const bindingArraySize = view.getUint16(offset + 14, true);
  const viewDimension = dimension !== 0 ? nameOf(VIEW_DIMENSION_NAMES, dimension) : '2d';

  let bufferType;
  let samplerType;
  let sampleType;
  let access;
  let format;
  if (type === BINDING_LAYOUT_TYPE.BUFFER) {
    bufferType = nameOf(BUFFER_BINDING_TYPE_NAMES, subtype);
    if (!bufferType) return null;
    // WebGPU forbids read-write 'storage' buffers in the vertex stage (e.g., root-signature layouts
    // that tag all entries Vertex|Fragment). Strip the illegal VERTEX visibility bit cleanly.
    if (bufferType === 'storage' && (visibility & GPU_SHADER_STAGE_VERTEX) !== 0) {
      visibility &= ~GPU_SHADER_STAGE_VERTEX;
    }
  } else if (type === BINDING_LAYOUT_TYPE.SAMPLER) {
    samplerType = nameOf(SAMPLER_BINDING_TYPE_NAMES, subtype);
    if (!samplerType) return null;
  } else if (type === BINDING_LAYOUT_TYPE.TEXTURE) {
    sampleType = nameOf(TEXTURE_SAMPLE_TYPE_NAMES, subtype);
    if (!sampleType || !viewDimension || viewDimension === 'undefined') return null;
  } else if (type === BINDING_LAYOUT_TYPE.STORAGE_TEXTURE) {
    access = nameOf(STORAGE_TEXTURE_ACCESS_NAMES, subtype);
    format = storageFormat !== 0 ? nameOf(TEXTURE_FORMAT_NAMES, storageFormat) : undefined;
    if (!access || !format || format === 'undefined' || !viewDimension || viewDimension === 'undefined') {
      return null;
    }
    if ((access === 'write-only' || access === 'read-write') && (visibility & GPU_SHADER_STAGE_VERTEX) !== 0) {
      visibility &= ~GPU_SHADER_STAGE_VERTEX;
    }
  } else {
    return null;
  }

  let pool = null;
  let entry;
  if (out) {
    entry = out;
    pool = out._pool || attachNonEnumerablePool(out, {
      buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 0 },
      sampler: { type: 'filtering' },
      texture: { sampleType: 'float', viewDimension: '2d', multisampled: false },
      storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' },
    });
    if ('buffer' in entry) delete entry.buffer;
    if ('sampler' in entry) delete entry.sampler;
    if ('texture' in entry) delete entry.texture;
    if ('storageTexture' in entry) delete entry.storageTexture;
    entry.binding = binding;
    entry.visibility = visibility;
  } else {
    entry = { binding, visibility };
  }

  if (type === BINDING_LAYOUT_TYPE.BUFFER) {
    const minBindingSize = Number(view.getBigUint64(offset + 16, true));
    if (pool) {
      const b = pool.buffer;
      b.type = bufferType;
      b.hasDynamicOffset = (flags & 0x1) !== 0;
      b.minBindingSize = minBindingSize;
      entry.buffer = b;
    } else {
      entry.buffer = {
        type: bufferType,
        hasDynamicOffset: (flags & 0x1) !== 0,
        minBindingSize,
      };
    }
  } else if (type === BINDING_LAYOUT_TYPE.SAMPLER) {
    if (pool) {
      pool.sampler.type = samplerType;
      entry.sampler = pool.sampler;
    } else {
      entry.sampler = { type: samplerType };
    }
  } else if (type === BINDING_LAYOUT_TYPE.TEXTURE) {
    if (pool) {
      const t = pool.texture;
      t.sampleType = sampleType;
      t.viewDimension = viewDimension;
      t.multisampled = (flags & 0x2) !== 0;
      entry.texture = t;
    } else {
      entry.texture = {
        sampleType,
        viewDimension,
        multisampled: (flags & 0x2) !== 0,
      };
    }
  } else {
    if (pool) {
      const st = pool.storageTexture;
      st.access = access;
      st.format = format;
      st.viewDimension = viewDimension;
      entry.storageTexture = st;
    } else {
      entry.storageTexture = {
        access,
        format,
        viewDimension,
      };
    }
  }
  if (bindingArraySize >= 1) {
    entry.bindingArraySize = bindingArraySize;
  } else if (out && 'bindingArraySize' in entry) {
    delete entry.bindingArraySize;
  }
  return entry;
}

function decodeBlendComponent(view, offset = 0, out = null) {
  if (!view || offset < 0 || offset + 3 > view.byteLength) {
    return null;
  }
  const operation = nameOf(BLEND_OPERATION_NAMES, view.getUint8(offset)) || 'add';
  const srcFactor = nameOf(BLEND_FACTOR_NAMES, view.getUint8(offset + 1)) || 'one';
  const dstFactor = nameOf(BLEND_FACTOR_NAMES, view.getUint8(offset + 2)) || 'zero';
  if (out) {
    out.operation = operation;
    out.srcFactor = srcFactor;
    out.dstFactor = dstFactor;
    return out;
  }
  return { operation, srcFactor, dstFactor };
}

function decodeStencilFace(view, offset = 0, out = null) {
  if (!view || offset < 0 || offset + 4 > view.byteLength) {
    return null;
  }
  const compareCode = view.getUint8(offset);
  const compareRaw = compareCode !== 0 ? nameOf(COMPARE_FUNCTION_NAMES, compareCode) : undefined;
  const compare = compareRaw && compareRaw !== 'undefined' ? compareRaw : 'always';
  const failOp = nameOf(STENCIL_OPERATION_NAMES, view.getUint8(offset + 1)) || 'keep';
  const depthFailOp = nameOf(STENCIL_OPERATION_NAMES, view.getUint8(offset + 2)) || 'keep';
  const passOp = nameOf(STENCIL_OPERATION_NAMES, view.getUint8(offset + 3)) || 'keep';
  if (out) {
    out.compare = compare;
    out.failOp = failOp;
    out.depthFailOp = depthFailOp;
    out.passOp = passOp;
    return out;
  }
  return { compare, failOp, depthFailOp, passOp };
}

// Decodes a GpuRenderPipelineKey at `base` into the GPURenderPipelineDescriptor parts it determines.
// Returns { primitive, multisample, depthStencil?, targets, buffers, colorTargetCount }.
// When `out` is provided, reuses pooled sub-objects and resets all dirty fields for zero-alloc decoding.
function decodeRenderPipelineKey(view, base = 0, out = null) {
  if (!view || base < 0 || base + KEY_SIZE > view.byteLength) {
    return null;
  }
  let pool = out ? out._pool : null;
  if (out && !pool) {
    const pTargets = new Array(8);
    const pBlends = new Array(8);
    const pBuffers = new Array(8);
    for (let i = 0; i < 8; i += 1) {
      pTargets[i] = { format: 'rgba8unorm', writeMask: 0x0f };
      pBlends[i] = {
        color: { operation: 'add', srcFactor: 'one', dstFactor: 'zero' },
        alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'zero' },
      };
      pBuffers[i] = { arrayStride: 0, stepMode: 'vertex', attributes: [] };
    }
    const pAttrs = new Array(16);
    for (let i = 0; i < 16; i += 1) {
      pAttrs[i] = { format: 'float32', offset: 0, shaderLocation: 0 };
    }
    pool = attachNonEnumerablePool(out, {
      targets: pTargets,
      blends: pBlends,
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: false,
        stencilFront: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
        stencilBack: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
        stencilReadMask: 0,
        stencilWriteMask: 0,
        depthBias: 0,
        depthBiasSlopeScale: 0,
        depthBiasClamp: 0,
      },
      buffers: pBuffers,
      attributes: pAttrs,
    });
  }

  const colorTargetCount = Math.min(view.getUint32(base + 20, true), 8);
  let targets;
  if (out) {
    targets = Array.isArray(out.targets) ? out.targets : [];
    targets.length = colorTargetCount;
  } else {
    targets = new Array(colorTargetCount);
  }
  for (let i = 0; i < colorTargetCount; i += 1) {
    const o = base + 24 + 12 * i;
    const format = view.getUint8(o);
    if (format === 0) {
      targets[i] = null;
      continue;
    }
    const targetFormat = nameOf(TEXTURE_FORMAT_NAMES, format);
    const writeMask = view.getUint8(o + 2) & 0x0f;
    const blendEnable = view.getUint8(o + 1) !== 0;
    if (pool) {
      const target = pool.targets[i];
      target.format = targetFormat;
      target.writeMask = writeMask;
      if (blendEnable) {
        const blend = pool.blends[i];
        decodeBlendComponent(view, o + 4, blend.color);
        decodeBlendComponent(view, o + 8, blend.alpha);
        target.blend = blend;
      } else if ('blend' in target) {
        delete target.blend;
      }
      targets[i] = target;
    } else {
      const target = { format: targetFormat, writeMask };
      if (blendEnable) {
        target.blend = {
          color: decodeBlendComponent(view, o + 4),
          alpha: decodeBlendComponent(view, o + 8),
        };
      }
      targets[i] = target;
    }
  }

  const ds = base + 120;
  let depthStencil;
  const dsFormat = view.getUint8(ds);
  const dsFormatName = dsFormat !== 0 ? nameOf(TEXTURE_FORMAT_NAMES, dsFormat) : undefined;
  if (dsFormatName && dsFormatName !== 'undefined') {
    const hasDepth = isDepthTextureFormat(dsFormatName);
    const rawDepthCompare = view.getUint8(ds + 2);
    const mappedCompare = rawDepthCompare !== 0 ? nameOf(COMPARE_FUNCTION_NAMES, rawDepthCompare) : undefined;
    const validCompare = mappedCompare && mappedCompare !== 'undefined' ? mappedCompare : undefined;
    if (pool) {
      depthStencil = pool.depthStencil;
      depthStencil.format = dsFormatName;
      depthStencil.depthWriteEnabled = hasDepth ? Boolean(view.getUint8(ds + 1) !== 0) : false;
      decodeStencilFace(view, ds + 4, depthStencil.stencilFront);
      decodeStencilFace(view, ds + 8, depthStencil.stencilBack);
      depthStencil.stencilReadMask = view.getUint32(ds + 12, true);
      depthStencil.stencilWriteMask = view.getUint32(ds + 16, true);
      depthStencil.depthBias = view.getInt32(ds + 20, true);
      depthStencil.depthBiasSlopeScale = finiteOr(f32FromBits(view.getUint32(ds + 24, true)), 0);
      depthStencil.depthBiasClamp = finiteOr(f32FromBits(view.getUint32(ds + 28, true)), 0);
      if (hasDepth) {
        depthStencil.depthCompare = validCompare || 'always';
      } else if (validCompare === 'always') {
        depthStencil.depthCompare = 'always';
      } else if ('depthCompare' in depthStencil) {
        delete depthStencil.depthCompare;
      }
    } else {
      depthStencil = {
        format: dsFormatName,
        depthWriteEnabled: hasDepth ? Boolean(view.getUint8(ds + 1) !== 0) : false,
        stencilFront: decodeStencilFace(view, ds + 4),
        stencilBack: decodeStencilFace(view, ds + 8),
        stencilReadMask: view.getUint32(ds + 12, true),
        stencilWriteMask: view.getUint32(ds + 16, true),
        depthBias: view.getInt32(ds + 20, true),
        depthBiasSlopeScale: finiteOr(f32FromBits(view.getUint32(ds + 24, true)), 0),
        depthBiasClamp: finiteOr(f32FromBits(view.getUint32(ds + 28, true)), 0),
      };
      if (hasDepth) {
        // Depth-containing formats (e.g. depth24plus-stencil8, depth32float) must always carry a
        // valid GPUCompareFunction; default to 'always' when depth test is disabled (code 0).
        depthStencil.depthCompare = validCompare || 'always';
      } else if (validCompare === 'always') {
        depthStencil.depthCompare = 'always';
      }
    }
  }

  const primitiveOut = out && out.primitive && typeof out.primitive === 'object' ? out.primitive : null;
  const primitive = decodePrimitiveTopologyPacket(view, base + 152, primitiveOut);

  const ms = base + 160;
  let multisample;
  if (out && out.multisample && typeof out.multisample === 'object') {
    multisample = out.multisample;
    multisample.count = view.getUint32(ms, true) || 1;
    multisample.mask = view.getUint32(ms + 4, true);
    multisample.alphaToCoverageEnabled = view.getUint8(ms + 8) !== 0;
  } else {
    multisample = {
      count: view.getUint32(ms, true) || 1,
      mask: view.getUint32(ms + 4, true),
      alphaToCoverageEnabled: view.getUint8(ms + 8) !== 0,
    };
  }

  const bufferCount = Math.min(view.getUint32(base + 172, true), 8);
  const attributeCount = Math.min(view.getUint32(base + 176, true), 16);
  let buffers;
  if (out) {
    buffers = Array.isArray(out.buffers) ? out.buffers : [];
    buffers.length = bufferCount;
  } else {
    buffers = new Array(bufferCount);
  }
  for (let i = 0; i < bufferCount; i += 1) {
    const o = base + 180 + 8 * i;
    const stepMode = view.getUint8(o + 4);
    if (stepMode === 0) {
      buffers[i] = null;
      continue;
    }
    if (pool) {
      const slot = pool.buffers[i];
      slot.arrayStride = view.getUint32(o, true);
      slot.stepMode = nameOf(VERTEX_STEP_MODE_NAMES, stepMode);
      slot.attributes.length = 0;
      buffers[i] = slot;
    } else {
      buffers[i] = {
        arrayStride: view.getUint32(o, true),
        stepMode: nameOf(VERTEX_STEP_MODE_NAMES, stepMode),
        attributes: [],
      };
    }
  }
  for (let i = 0; i < attributeCount; i += 1) {
    const o = base + 244 + 8 * i;
    const bufferIndex = view.getUint8(o + 6);
    if (bufferIndex >= bufferCount) continue;
    const slot = buffers[bufferIndex];
    if (!slot) continue;
    if (pool) {
      const attr = pool.attributes[i];
      attr.format = nameOf(VERTEX_FORMAT_NAMES, view.getUint8(o + 4));
      attr.offset = view.getUint32(o, true);
      attr.shaderLocation = view.getUint8(o + 5);
      slot.attributes.push(attr);
    } else {
      slot.attributes.push({
        format: nameOf(VERTEX_FORMAT_NAMES, view.getUint8(o + 4)),
        offset: view.getUint32(o, true),
        shaderLocation: view.getUint8(o + 5),
      });
    }
  }

  if (out) {
    out.primitive = primitive;
    out.multisample = multisample;
    out.depthStencil = depthStencil;
    out.targets = targets;
    out.buffers = buffers;
    out.colorTargetCount = colorTargetCount;
    return out;
  }
  return { primitive, multisample, depthStencil, targets, buffers, colorTargetCount };
}

function decodeBeginRenderPass(view, offset = 0, out = null) {
  if (!view || offset < 0 || offset + BEGIN_RENDER_PASS_BYTES > view.byteLength) {
    return null;
  }
  let pool = out ? out._pool : null;
  if (out && !pool) {
    const pColors = new Array(8);
    for (let i = 0; i < 8; i += 1) {
      pColors[i] = {
        textureId: 0,
        loadOp: 'undefined',
        storeOp: 'undefined',
        view: {},
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      };
    }
    pool = attachNonEnumerablePool(out, {
      colors: pColors,
      depthStencil: {
        textureId: 0,
        depthLoadOp: 0,
        depthStoreOp: 0,
        stencilLoadOp: 0,
        stencilStoreOp: 0,
        view: {},
        depthClearValue: 1,
        stencilClearValue: 0,
        depthReadOnly: false,
        stencilReadOnly: false,
      },
    });
  }

  const colorCount = Math.min(view.getUint32(offset, true), 8);
  let colors;
  if (out) {
    colors = Array.isArray(out.colors) ? out.colors : [];
    colors.length = colorCount;
  } else {
    colors = new Array(colorCount);
  }
  for (let i = 0; i < colorCount; i += 1) {
    const o = offset + 8 + 40 * i;
    const textureId = view.getUint32(o, true);
    if (textureId === 0) {
      colors[i] = null;
      continue;
    }
    if (pool) {
      const slot = pool.colors[i];
      slot.textureId = textureId;
      slot.loadOp = nameOf(LOAD_OP_NAMES, view.getUint8(o + 4));
      slot.storeOp = nameOf(STORE_OP_NAMES, view.getUint8(o + 5));
      decodeTextureViewDesc(view, o + 8, slot.view);
      slot.clearValue.r = view.getFloat32(o + 24, true);
      slot.clearValue.g = view.getFloat32(o + 28, true);
      slot.clearValue.b = view.getFloat32(o + 32, true);
      slot.clearValue.a = view.getFloat32(o + 36, true);
      colors[i] = slot;
    } else {
      colors[i] = {
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
      };
    }
  }
  const d = offset + 328;
  let depthStencil = null;
  const dsTexture = view.getUint32(d, true);
  if (dsTexture !== 0) {
    if (pool) {
      depthStencil = pool.depthStencil;
      depthStencil.textureId = dsTexture;
      depthStencil.depthLoadOp = view.getUint8(d + 4);
      depthStencil.depthStoreOp = view.getUint8(d + 5);
      depthStencil.stencilLoadOp = view.getUint8(d + 6);
      depthStencil.stencilStoreOp = view.getUint8(d + 7);
      decodeTextureViewDesc(view, d + 8, depthStencil.view);
      depthStencil.depthClearValue = view.getFloat32(d + 24, true);
      depthStencil.stencilClearValue = view.getUint32(d + 28, true);
      depthStencil.depthReadOnly = view.getUint8(d + 32) !== 0;
      depthStencil.stencilReadOnly = view.getUint8(d + 33) !== 0;
    } else {
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
  }
  if (out) {
    out.colors = colors;
    out.depthStencil = depthStencil;
    return out;
  }
  return { colors, depthStencil };
}

// W302/W354: lightweight monomorphic pass-command objects for hot dynamic render-state packets,
// eliminating per-packet closures and redundant WebGPU state calls within the same render pass.
class SetViewportCommand {
  constructor(handler, x, y, width, height, minDepth, maxDepth) {
    this.type = 'renderState';
    this._handler = handler;
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
    this.minDepth = minDepth;
    this.maxDepth = maxDepth;
  }
  run(ctl) {
    if (ctl.passKind !== 'render') return;
    const h = this._handler;
    const targetSize = h._getActivePassTargetSize(ctl, h._scratch.targetSize);
    const raw = h._scratch.viewport;
    raw.x = this.x;
    raw.y = this.y;
    raw.width = this.width;
    raw.height = this.height;
    raw.minDepth = this.minDepth;
    raw.maxDepth = this.maxDepth;
    const vp = clampViewportToTarget(raw, targetSize, h._scratch.viewportClamped);
    const wasClamped = isViewportClamped(raw, vp);
    if (wasClamped) {
      h.telemetry.clampedViewports += 1;
      const targetTag = targetSize ? `${targetSize.width}x${targetSize.height}` : 'unbounded';
      h._recordTelemetryWarningOnce(
        `clamp-viewport:${targetTag}`,
        `clamped viewport (${raw.x},${raw.y},${raw.width}x${raw.height},d=${raw.minDepth}..${raw.maxDepth}) to (${vp.x},${vp.y},${vp.width}x${vp.height},d=${vp.minDepth}..${vp.maxDepth}) for target ${targetTag}`
      );
    }
    const ps = h._syncPassState(ctl.pass);
    if (
      h.elideRedundantState &&
      ps.hasViewport &&
      ps.vpX === vp.x &&
      ps.vpY === vp.y &&
      ps.vpW === vp.width &&
      ps.vpH === vp.height &&
      ps.vpMinD === vp.minDepth &&
      ps.vpMaxD === vp.maxDepth &&
      (!wasClamped ||
        (ps.vpRawX === this.x &&
          ps.vpRawY === this.y &&
          ps.vpRawW === this.width &&
          ps.vpRawH === this.height &&
          ps.vpRawMinD === this.minDepth &&
          ps.vpRawMaxD === this.maxDepth))
    ) {
      h.telemetry.elidedViewportSets += 1;
      return;
    }
    ctl.pass.setViewport(vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth);
    ps.hasViewport = true;
    ps.vpX = vp.x;
    ps.vpY = vp.y;
    ps.vpW = vp.width;
    ps.vpH = vp.height;
    ps.vpMinD = vp.minDepth;
    ps.vpMaxD = vp.maxDepth;
    ps.vpRawX = this.x;
    ps.vpRawY = this.y;
    ps.vpRawW = this.width;
    ps.vpRawH = this.height;
    ps.vpRawMinD = this.minDepth;
    ps.vpRawMaxD = this.maxDepth;
  }
}

class SetScissorRectCommand {
  constructor(handler, x, y, width, height) {
    this.type = 'renderState';
    this._handler = handler;
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
  }
  run(ctl) {
    if (ctl.passKind !== 'render') return;
    const h = this._handler;
    const targetSize = h._getActivePassTargetSize(ctl, h._scratch.targetSize);
    const raw = h._scratch.scissor;
    raw.x = this.x;
    raw.y = this.y;
    raw.width = this.width;
    raw.height = this.height;
    const sc = clampScissorToTarget(raw, targetSize, h._scratch.scissorClamped);
    const wasClamped = isScissorClamped(raw, sc);
    if (wasClamped) {
      h.telemetry.clampedScissors += 1;
      const targetTag = targetSize ? `${targetSize.width}x${targetSize.height}` : 'unbounded';
      h._recordTelemetryWarningOnce(
        `clamp-scissor:${targetTag}`,
        `clamped scissor (${raw.x},${raw.y},${raw.width}x${raw.height}) to (${sc.x},${sc.y},${sc.width}x${sc.height}) for target ${targetTag}`
      );
    }
    const ps = h._syncPassState(ctl.pass);
    if (
      h.elideRedundantState &&
      ps.hasScissor &&
      ps.scX === sc.x &&
      ps.scY === sc.y &&
      ps.scW === sc.width &&
      ps.scH === sc.height &&
      (!wasClamped ||
        (ps.scRawX === this.x &&
          ps.scRawY === this.y &&
          ps.scRawW === this.width &&
          ps.scRawH === this.height))
    ) {
      h.telemetry.elidedScissorSets += 1;
      return;
    }
    ctl.pass.setScissorRect(sc.x, sc.y, sc.width, sc.height);
    ps.hasScissor = true;
    ps.scX = sc.x;
    ps.scY = sc.y;
    ps.scW = sc.width;
    ps.scH = sc.height;
    ps.scRawX = this.x;
    ps.scRawY = this.y;
    ps.scRawW = this.width;
    ps.scRawH = this.height;
  }
}

class SetBlendConstantCommand {
  constructor(handler, r, g, b, a) {
    this.type = 'renderState';
    this._handler = handler;
    this.r = r;
    this.g = g;
    this.b = b;
    this.a = a;
  }
  run(ctl) {
    if (ctl.passKind !== 'render') return;
    const h = this._handler;
    if (h) {
      const ps = h._syncPassState(ctl.pass);
      if (
        h.elideRedundantState &&
        ps.hasBlend &&
        ps.bcR === this.r &&
        ps.bcG === this.g &&
        ps.bcB === this.b &&
        ps.bcA === this.a
      ) {
        h.telemetry.elidedBlendConstantSets += 1;
        return;
      }
      ctl.pass.setBlendConstant({ r: this.r, g: this.g, b: this.b, a: this.a });
      ps.hasBlend = true;
      ps.bcR = this.r;
      ps.bcG = this.g;
      ps.bcB = this.b;
      ps.bcA = this.a;
      return;
    }
    ctl.pass.setBlendConstant({ r: this.r, g: this.g, b: this.b, a: this.a });
  }
}

class SetStencilReferenceCommand {
  constructor(handler, reference) {
    this.type = 'renderState';
    this._handler = handler;
    this.reference = reference;
  }
  run(ctl) {
    if (ctl.passKind !== 'render') return;
    const h = this._handler;
    if (h) {
      const ps = h._syncPassState(ctl.pass);
      if (h.elideRedundantState && ps.hasStencil && ps.stencilRef === this.reference) {
        h.telemetry.elidedStencilRefSets += 1;
        return;
      }
      ctl.pass.setStencilReference(this.reference);
      ps.hasStencil = true;
      ps.stencilRef = this.reference;
      return;
    }
    ctl.pass.setStencilReference(this.reference);
  }
}

// W354: monomorphic SetBindGroup2 command (self-serves _rsSetBindGroup & _wgpuMeta without extra objects).
class SetBindGroup2Command {
  constructor(handler, groupIndex, bindGroupId, bindGroup, bindGroupDesc, offsets) {
    this.type = 'renderState';
    this._handler = handler;
    this.kind = 'setBindGroup2';
    this.groupIndex = groupIndex;
    this.bindGroupId = bindGroupId;
    this.bindGroup = bindGroup;
    this.bindGroupDesc = bindGroupDesc;
    this.offsets = offsets;
    this._rsSetBindGroup = this;
    this._wgpuMeta = this;
  }
  run(ctl) {
    if (!ctl.pass) return;
    const h = this._handler;
    let effectiveOffsets = this.offsets;
    // #1941 ExecuteIndirect parity: validate dynamic offset alignment, buffer bounds, and dummy
    // fallbacks on the explicit-layout pass path (without disturbing layout:'auto' remapping).
    if (
      h &&
      effectiveOffsets &&
      effectiveOffsets.length > 0 &&
      this.bindGroupDesc &&
      this.bindGroupDesc.dynamicSlotMeta
    ) {
      effectiveOffsets = h._sanitizeExplicitDynamicOffsets(
        this.bindGroupId,
        this.bindGroupDesc,
        effectiveOffsets
      );
    }
    if (
      h &&
      ctl.passKind === 'render' &&
      h.elideRedundantState &&
      !h._flushHasV1BindGroup &&
      h._isPassBindGroupRedundant(ctl.pass, this.groupIndex, this.bindGroup, effectiveOffsets)
    ) {
      h.telemetry.elidedBindGroupSets += 1;
      return;
    }
    ctl.pass.setBindGroup(this.groupIndex, this.bindGroup, effectiveOffsets);
    if (h && ctl.passKind === 'render') {
      h._recordPassBindGroup(ctl.pass, this.groupIndex, this.bindGroup, effectiveOffsets);
    }
  }
}

class RemappedSetBindGroupCommand {
  constructor(handler, bindGroup, bindGroupId) {
    this.type = 'renderState';
    this._handler = handler;
    this.kind = 'setBindGroup2';
    this.groupIndex = 0;
    this.bindGroupId = bindGroupId || 0;
    this.bindGroup = bindGroup;
    this._wgpuMeta = this;
  }
  run(ctl) {
    if (!ctl.pass) return;
    const h = this._handler;
    if (
      h &&
      ctl.passKind === 'render' &&
      h.elideRedundantState &&
      h.elideRemappedBindGroups &&
      !h._flushHasV1BindGroup &&
      h._isPassBindGroupRedundant(ctl.pass, 0, this.bindGroup, null)
    ) {
      h.telemetry.elidedBindGroupSets += 1;
      return;
    }
    ctl.pass.setBindGroup(0, this.bindGroup);
    if (h && ctl.passKind === 'render') {
      h._recordPassBindGroup(ctl.pass, 0, this.bindGroup, null);
    }
  }
}

class NoopRenderGroup0Command {
  constructor(bindGroupId) {
    this.type = 'renderState';
    this.kind = 'setBindGroup2';
    this.groupIndex = 0;
    this.bindGroupId = bindGroupId || 0;
    this._wgpuMeta = this;
  }
  run() {}
}

class BeginRenderPassCommand {
  constructor(handler, decoded, resolvers, targetAttachmentIsSwapchain, staticTargetSize) {
    this.type = 'renderState';
    this._handler = handler;
    this.kind = 'beginRenderPass';
    this._rsPassOp = 'beginRenderPass';
    this._wgpuMeta = this;
    this._decoded = decoded || { colors: [], depthStencil: null };
    this._resolvers = resolvers || { colors: [], depthStencil: null };
    this._targetIsSwapchain = Boolean(targetAttachmentIsSwapchain);
    this._staticTargetSizeSlot = { width: 0, height: 0 };
    this._staticTargetSize = staticTargetSize || null;
  }
  run(ctl) {
    const h = this._handler;
    const desc = h.buildRenderPassDescriptor(this._decoded, this._resolvers, ctl);
    const refSize = h._lastPassReferenceSize;
    h._lastPassReferenceSize = null;
    h._activePass = null;
    h._activePassTargetSize = null;
    const hasColor =
      Array.isArray(desc.colorAttachments) && desc.colorAttachments.some((c) => Boolean(c && c.view));
    const hasDepthStencil = Boolean(desc.depthStencilAttachment && desc.depthStencilAttachment.view);
    if (!hasColor && !hasDepthStencil) {
      if (ctl.passKind === 'render' && typeof ctl.endPass === 'function') {
        ctl.endPass();
      }
      h._resetPassState(null);
      h._recordTelemetryWarningOnce(
        'rp-empty-attachments',
        'render pass: skipped beginRenderPass with zero valid color or depthStencil attachments'
      );
      return;
    }
    const pass = ctl.beginRenderPass(desc);
    const activePass = pass || ctl.pass || null;
    h._activePass = activePass;
    h._resetPassState(activePass);
    h._activePassTargetSize =
      refSize ||
      (this._targetIsSwapchain ? h._resolveSwapchainSize(ctl) : this._staticTargetSize);
  }
}

// ---- Packet handler ----

class RenderStatePackets {
  constructor(processor, options = {}) {
    this.processor = processor;
    this.device = processor.device;
    this._cachedDeviceGeneration =
      processor && Number.isInteger(processor._deviceGeneration) ? processor._deviceGeneration : 0;
    // Optional: (kind 'bindGroupLayout' | 'pipelineLayout', contentId: bigint) => GPU object, for
    // ManifestReference layouts loaded from layout-manifest.json (e.g. by reflection_loader.js).
    this.resolveManifestLayout =
      typeof options.resolveManifestLayout === 'function' ? options.resolveManifestLayout : null;
    // W167/W186: optional pipelineCache override (falls back to processor.pipelineCache).
    this.pipelineCache =
      options.pipelineCache && typeof options.pipelineCache === 'object' ? options.pipelineCache : null;
    // W163/W186: optional textureFormats / resolveStorageCompatibleFormat override.
    this._textureFormatsOptions =
      typeof options.resolveStorageCompatibleFormat === 'function' ||
      (options.textureFormats && typeof options.textureFormats === 'object')
        ? {
            resolveStorageCompatibleFormat: options.resolveStorageCompatibleFormat,
            textureFormats: options.textureFormats,
          }
        : null;
    this.bindGroupLayouts = new Map(); // id -> { layout, entries, sequentialBindings }
    this.pipelineLayouts = new Map(); // id -> GPUPipelineLayout
    this.pipelineLayoutMeta = new Map(); // id -> { groupCount, isSingleGroup0, sequentialGroup0, group0HasEntries, group0Entries }
    this.computePipelineMeta = new Map(); // id -> { id, csId, layoutId, entryPoint, hasComputeAttr, autoLayoutRemap, wgslBindings }
    this.renderPipelineAutoLayoutMeta = new Map(); // id -> { id, layoutId, autoLayoutRemap, wgslBindings }
    this.bindGroupDescriptors = new Map(); // id -> { id, layoutId, resources }
    this.samplerMeta = new Map(); // id -> { comparison, filtering }
    // 'c|r:<pipeId>:<bgId>' (+ '@<dynOffsets>' when root-constant dynamic offsets are non-zero)
    // -> GPUBindGroup | null. LRU-bounded by remappedBindGroupCacheLimit.
    this.remappedBindGroups = new Map();
    // W363: O(affected) reverse indexes for bindGroupId / pipeline invalidation.
    this._remappedBgByBindGroupId = new Map();
    this._remappedBgByPipelineKey = new Map();
    this.remappedBindGroupCacheLimit =
      Number.isInteger(options.remappedBindGroupCacheLimit) && options.remappedBindGroupCacheLimit > 0
        ? options.remappedBindGroupCacheLimit
        : REMAPPED_BIND_GROUP_CACHE_LIMIT;
    // W363: Per-frame pooled v2 pass command objects (released by WgpuCommandProcessor._clearPassCommands).
    this._vpCmdPool = [];
    this._vpCmdPoolUsed = 0;
    this._scCmdPool = [];
    this._scCmdPoolUsed = 0;
    this._bcCmdPool = [];
    this._bcCmdPoolUsed = 0;
    this._srCmdPool = [];
    this._srCmdPoolUsed = 0;
    this._sbg2CmdPool = [];
    this._sbg2CmdPoolUsed = 0;
    this._remappedSbgCmdPool = [];
    this._remappedSbgCmdPoolUsed = 0;
    this._noopGroup0CmdPool = [];
    this._noopGroup0CmdPoolUsed = 0;
    // W373: Per-frame pooled BeginRenderPassCommand + SetBindGroup2 dynamic-offsets arrays.
    this._brpCmdPool = [];
    this._brpCmdPoolUsed = 0;
    this._dynOffsetsPool = [];
    this._dynOffsetsPoolUsed = 0;
    // W186: hot-path PresentX multi-plane cache keyed by (planeTextureId, viewFormat, sampler, pipeline/layout).
    this.presentxBindGroupCache = new Map();
    this.presentxBindGroupCacheLimit =
      Number.isInteger(options.presentxBindGroupCacheLimit) && options.presentxBindGroupCacheLimit > 0
        ? options.presentxBindGroupCacheLimit
        : PRESENTX_BIND_GROUP_CACHE_LIMIT;
    this._textureGenerations = new Map(); // textureId -> monotonic generation (LRU-bounded)
    this._nextTextureGeneration = 0;
    this._textureViewOwners = new Map(); // textureId -> GPUTexture handle for silent-replacement detection
    this._destroyed = false;
    this._lastBuiltRenderPipelineAutoLayout = null;
    // W103: pipeline id -> token for compiles still running in processor.asyncPipelineCache. A
    // pipeline stays "pending" here until its result is installed, so flushes never see a gap.
    this.pendingAsyncRenderPipelines = new Map();
    this.pendingAsyncComputePipelines = new Map();
    this.layoutsByContent = new Map(); // 'bgl:<hex>' / 'pl:<hex>' -> object
    this.textureViews = new Map(); // textureId -> Map(descKey -> GPUTextureView)
    // W354: WeakMap<GPUTexture, { defaultView, byKey }> caches GPUTextureViews by live texture object
    // identity (automatic GC when textures are replaced/collected, zero Dawn IPC spam).
    this._textureViewWeakCache = new WeakMap();
    // W354: WeakMap<GPUPipeline, GPUBindGroupLayout> caches group-0 layout lookups on PresentX / auto-remap paths.
    this._pipelineBglCache = new WeakMap();
    // W354: per-pass state-change filter (default enabled) to elide redundant WebGPU state calls within a pass.
    this.elideRedundantState = options.elideRedundantState !== false;
    this.elideRemappedBindGroups = options.elideRemappedBindGroups === true;
    this._flushHasV1BindGroup = false;
    this._passState = {
      pass: null,
      hasViewport: false,
      vpX: 0,
      vpY: 0,
      vpW: 0,
      vpH: 0,
      vpMinD: 0,
      vpMaxD: 1,
      vpRawX: 0,
      vpRawY: 0,
      vpRawW: 0,
      vpRawH: 0,
      vpRawMinD: 0,
      vpRawMaxD: 1,
      hasScissor: false,
      scX: 0,
      scY: 0,
      scW: 0,
      scH: 0,
      scRawX: 0,
      scRawY: 0,
      scRawW: 0,
      scRawH: 0,
      hasBlend: false,
      bcR: 0,
      bcG: 0,
      bcB: 0,
      bcA: 0,
      hasStencil: false,
      stencilRef: 0,
      bgMask: 0,
      bgGroups: new Array(MAX_FAST_PASS_BG_SLOTS).fill(null),
      bgHasOffsets: new Uint8Array(MAX_FAST_PASS_BG_SLOTS),
      bgOffsetsLen: new Int32Array(MAX_FAST_PASS_BG_SLOTS),
      bgOffsetsData: new Uint32Array(MAX_FAST_PASS_BG_SLOTS * MAX_FAST_PASS_DYN_OFFSETS),
      bgOffsetsOverflow: new Array(MAX_FAST_PASS_BG_SLOTS).fill(null),
      bgOverflow: null,
    };
    this._scratchSwapchainFormatDesc = { format: '' };
    this._scratchLastRenderGroup0 = { bindGroupId: 0, bindGroupDesc: null, dynamicOffsets: EMPTY_DYNAMIC_OFFSETS };
    this._scratchLastComputeGroup0 = { bindGroupId: 0, bindGroupDesc: null, dynamicOffsets: EMPTY_DYNAMIC_OFFSETS };
    this._scratchRemapResult = { status: 'empty', bindGroup: null };
    this._scratchColorSizes = [];
    this._scratchColorSizeSlots = [
      { width: 0, height: 0 },
      { width: 0, height: 0 },
      { width: 0, height: 0 },
      { width: 0, height: 0 },
      { width: 0, height: 0 },
      { width: 0, height: 0 },
      { width: 0, height: 0 },
      { width: 0, height: 0 },
    ];
    this._scratchDepthSizeSlot = { width: 0, height: 0 };
    this._scratchColorSampleCounts = [];
    this._scratchColorFormats = [];
    this._scratchAttachmentSizes = {
      colors: this._scratchColorSizes,
      colorSampleCounts: this._scratchColorSampleCounts,
      colorFormats: this._scratchColorFormats,
      depthStencil: null,
      depthSampleCount: null,
      depthFormat: null,
      depthUsage: null,
    };
    this.textureMeta = new Map(); // textureId -> { id, requestedFormat, format, bcFallback, usage, width, height }
    this.dummies = new Map();
    this.emptyLayout = null;
    this.emptyFragment = null;
    this.stats = { packets: 0, errors: 0 };
    this.telemetry = {
      legacyEntryPointRemaps: 0,
      depthCompareDefaults: 0,
      rootSigGroup0Layouts: 0,
      autoLayoutFallbacks: 0,
      bindGroupKindMismatches: 0,
      computePipelineCount: 0,
      computeDispatchCount: 0,
      skippedComputeDispatchCount: 0,
      clampedComputeDispatchCount: 0,
      texture2Count: 0,
      textureClears: 0,
      textureClearsMissing: 0,
      bcFormatFallbacks: 0,
      usageStrips: 0,
      unfilterableFloatFallbacks: 0,
      dynamicOffsetRemaps: 0,
      dynamicOffsetClamps: 0,
      remappedBindGroupEvictions: 0,
      asyncPipelineCompiles: 0,
      asyncPipelineFailures: 0,
      pendingAsyncPipelineSkips: 0,
      suppressedMissingTextureErrors: 0,
      presentxBindGroupCacheHits: 0,
      presentxBindGroupCacheMisses: 0,
      fallbackSamplersBound: 0,
      clampedViewports: 0,
      clampedScissors: 0,
      strippedNonBlendableBlends: 0,
      sanitizedAttachments: 0,
      elidedViewportSets: 0,
      elidedScissorSets: 0,
      elidedBlendConstantSets: 0,
      elidedStencilRefSets: 0,
      elidedBindGroupSets: 0,
      textureViewCacheHits: 0,
      textureViewCacheMisses: 0,
      bindingArrayLayouts: 0,
      bindingArrayBindGroups: 0,
      rtArenaGroup3Binds: 0,
      missingResourceFallbacks: 0,
      bindGroupValidationFallbacks: 0,
      unknownBindGroupErrors: 0,
      skippedUnboundPassOps: 0,
      skippedInvalidIndirectOps: 0,
      warnings: [],
    };
    this._seenWarningKeys = new Set();
    this.lastError = null;
    this._packetError = null;
    // W104: 'gen:textureId' keys of missing render-pass attachments already reported once.
    this._reportedMissingTextures = new Set();
    this._suppressPacketError = false;
    this._activePass = null;
    this._activePassTargetSize = null;
    this._lastPassReferenceSize = null;
    // W302: pre-allocated scratch bag and reusable END_RENDER_PASS command for zero-alloc hot paths.
    this._scratch = createRenderStateScratch();
    this._endRenderPassCmd = {
      type: 'renderState',
      _rsPassOp: 'endRenderPass',
      _wgpuMeta: { kind: 'endRenderPass' },
      run: (ctl) => {
        if (ctl.passKind === 'render') ctl.endPass();
        this._activePass = null;
        this._activePassTargetSize = null;
        this._lastPassReferenceSize = null;
        this._resetPassState(null);
      },
    };
  }

  _resetPassState(pass) {
    const ps = this._passState;
    ps.pass = pass;
    ps.hasViewport = false;
    ps.hasScissor = false;
    ps.hasBlend = false;
    ps.hasStencil = false;
    let mask = ps.bgMask;
    while (mask !== 0) {
      const bit = mask & -mask;
      const slot = 31 - Math.clz32(bit);
      mask ^= bit;
      ps.bgGroups[slot] = null;
      ps.bgHasOffsets[slot] = 0;
      ps.bgOffsetsLen[slot] = 0;
    }
    ps.bgMask = 0;
    if (ps.bgOverflow !== null && ps.bgOverflow.size > 0) {
      ps.bgOverflow.clear();
    }
    return ps;
  }

  _syncPassState(pass) {
    if (this._passState.pass !== pass) {
      return this._resetPassState(pass);
    }
    return this._passState;
  }

  _isPassBindGroupRedundant(pass, groupIndex, bindGroup, offsets) {
    if (!bindGroup || bindGroup.destroyed === true) return false;
    const ps = this._syncPassState(pass);
    const hasOff = offsets !== null && offsets !== undefined ? 1 : 0;
    const len = hasOff === 1 ? offsets.length : 0;
    if (groupIndex >= 0 && groupIndex < MAX_FAST_PASS_BG_SLOTS) {
      const bit = 1 << groupIndex;
      if ((ps.bgMask & bit) === 0 || ps.bgGroups[groupIndex] !== bindGroup) {
        return false;
      }
      if (ps.bgHasOffsets[groupIndex] !== hasOff) {
        // Treat null/undefined and empty [] offsets identically on the same bind group only when both have len === 0?
        // Preserve strict match between explicit offsets array presence and length.
        if (!(len === 0 && ps.bgOffsetsLen[groupIndex] === 0)) return false;
      }
      if (ps.bgOffsetsLen[groupIndex] !== len) return false;
      if (len === 0) return true;
      if (len <= MAX_FAST_PASS_DYN_OFFSETS) {
        const base = groupIndex * MAX_FAST_PASS_DYN_OFFSETS;
        const data = ps.bgOffsetsData;
        for (let i = 0; i < len; i += 1) {
          if (data[base + i] !== offsets[i]) return false;
        }
        return true;
      }
      const ov = ps.bgOffsetsOverflow[groupIndex];
      if (!ov || ov.length < len) return false;
      for (let i = 0; i < len; i += 1) {
        if (ov[i] !== offsets[i]) return false;
      }
      return true;
    }
    if (ps.bgOverflow === null) return false;
    const rec = ps.bgOverflow.get(groupIndex);
    if (!rec || rec.bindGroup !== bindGroup || rec.len !== len) return false;
    if (len === 0) return true;
    for (let i = 0; i < len; i += 1) {
      if (rec.offsets[i] !== offsets[i]) return false;
    }
    return true;
  }

  _recordPassBindGroup(pass, groupIndex, bindGroup, offsets) {
    const ps = this._syncPassState(pass);
    const hasOff = offsets !== null && offsets !== undefined ? 1 : 0;
    const len = hasOff === 1 ? offsets.length : 0;
    if (groupIndex >= 0 && groupIndex < MAX_FAST_PASS_BG_SLOTS) {
      const bit = 1 << groupIndex;
      ps.bgMask |= bit;
      ps.bgGroups[groupIndex] = bindGroup;
      ps.bgHasOffsets[groupIndex] = hasOff;
      ps.bgOffsetsLen[groupIndex] = len;
      if (len > 0) {
        if (len <= MAX_FAST_PASS_DYN_OFFSETS) {
          const base = groupIndex * MAX_FAST_PASS_DYN_OFFSETS;
          const data = ps.bgOffsetsData;
          for (let i = 0; i < len; i += 1) {
            data[base + i] = offsets[i];
          }
        } else {
          let ov = ps.bgOffsetsOverflow[groupIndex];
          if (!ov || ov.length < len) {
            ov = new Uint32Array(len);
            ps.bgOffsetsOverflow[groupIndex] = ov;
          }
          for (let i = 0; i < len; i += 1) {
            ov[i] = offsets[i];
          }
        }
      }
      return;
    }
    if (ps.bgOverflow === null) ps.bgOverflow = new Map();
    let rec = ps.bgOverflow.get(groupIndex);
    if (!rec) {
      rec = { bindGroup, len, offsets: len > 0 ? new Uint32Array(len) : null };
      ps.bgOverflow.set(groupIndex, rec);
    } else {
      rec.bindGroup = bindGroup;
      rec.len = len;
      if (len > 0 && (!rec.offsets || rec.offsets.length < len)) {
        rec.offsets = new Uint32Array(len);
      }
    }
    for (let i = 0; i < len; i += 1) {
      rec.offsets[i] = offsets[i];
    }
  }

  _getPipelineGroup0Layout(pipeline) {
    if (!pipeline || typeof pipeline.getBindGroupLayout !== 'function') return null;
    if (this._pipelineBglCache.has(pipeline)) {
      return this._pipelineBglCache.get(pipeline);
    }
    let layout = null;
    try {
      layout = pipeline.getBindGroupLayout(0) || null;
    } catch (_) {
      layout = null;
    }
    if (layout) {
      this._pipelineBglCache.set(pipeline, layout);
    }
    return layout;
  }

  getStats() {
    return {
      packets: this.stats.packets,
      errors: this.stats.errors,
      legacyEntryPointRemaps: this.telemetry.legacyEntryPointRemaps,
      depthCompareDefaults: this.telemetry.depthCompareDefaults,
      rootSigGroup0Layouts: this.telemetry.rootSigGroup0Layouts,
      autoLayoutFallbacks: this.telemetry.autoLayoutFallbacks,
      bindGroupKindMismatches: this.telemetry.bindGroupKindMismatches,
      computePipelineCount: this.telemetry.computePipelineCount,
      computeDispatchCount: this.telemetry.computeDispatchCount,
      skippedComputeDispatchCount: this.telemetry.skippedComputeDispatchCount,
      clampedComputeDispatchCount: this.telemetry.clampedComputeDispatchCount,
      texture2Count: this.telemetry.texture2Count,
      bcFormatFallbacks: this.telemetry.bcFormatFallbacks,
      usageStrips: this.telemetry.usageStrips,
      unfilterableFloatFallbacks: this.telemetry.unfilterableFloatFallbacks,
      presentxBindGroupCacheHits: this.telemetry.presentxBindGroupCacheHits,
      presentxBindGroupCacheMisses: this.telemetry.presentxBindGroupCacheMisses,
      fallbackSamplersBound: this.telemetry.fallbackSamplersBound,
      clampedViewports: this.telemetry.clampedViewports,
      clampedScissors: this.telemetry.clampedScissors,
      strippedNonBlendableBlends: this.telemetry.strippedNonBlendableBlends,
      sanitizedAttachments: this.telemetry.sanitizedAttachments,
      elidedViewportSets: this.telemetry.elidedViewportSets,
      elidedScissorSets: this.telemetry.elidedScissorSets,
      elidedBlendConstantSets: this.telemetry.elidedBlendConstantSets,
      elidedStencilRefSets: this.telemetry.elidedStencilRefSets,
      elidedBindGroupSets: this.telemetry.elidedBindGroupSets,
      textureViewCacheHits: this.telemetry.textureViewCacheHits,
      textureViewCacheMisses: this.telemetry.textureViewCacheMisses,
      bindingArrayLayouts: this.telemetry.bindingArrayLayouts,
      bindingArrayBindGroups: this.telemetry.bindingArrayBindGroups,
      rtArenaGroup3Binds: this.telemetry.rtArenaGroup3Binds,
      missingResourceFallbacks: this.telemetry.missingResourceFallbacks,
      bindGroupValidationFallbacks: this.telemetry.bindGroupValidationFallbacks,
      unknownBindGroupErrors: this.telemetry.unknownBindGroupErrors,
      skippedUnboundPassOps: this.telemetry.skippedUnboundPassOps,
      skippedInvalidIndirectOps: this.telemetry.skippedInvalidIndirectOps,
    };
  }

  getTelemetry() {
    return {
      legacyEntryPointRemaps: this.telemetry.legacyEntryPointRemaps,
      depthCompareDefaults: this.telemetry.depthCompareDefaults,
      rootSigGroup0Layouts: this.telemetry.rootSigGroup0Layouts,
      autoLayoutFallbacks: this.telemetry.autoLayoutFallbacks,
      bindGroupKindMismatches: this.telemetry.bindGroupKindMismatches,
      computePipelineCount: this.telemetry.computePipelineCount,
      computeDispatchCount: this.telemetry.computeDispatchCount,
      skippedComputeDispatchCount: this.telemetry.skippedComputeDispatchCount,
      clampedComputeDispatchCount: this.telemetry.clampedComputeDispatchCount,
      texture2Count: this.telemetry.texture2Count,
      bcFormatFallbacks: this.telemetry.bcFormatFallbacks,
      usageStrips: this.telemetry.usageStrips,
      unfilterableFloatFallbacks: this.telemetry.unfilterableFloatFallbacks,
      dynamicOffsetRemaps: this.telemetry.dynamicOffsetRemaps,
      dynamicOffsetClamps: this.telemetry.dynamicOffsetClamps,
      remappedBindGroupEvictions: this.telemetry.remappedBindGroupEvictions,
      asyncPipelineCompiles: this.telemetry.asyncPipelineCompiles,
      asyncPipelineFailures: this.telemetry.asyncPipelineFailures,
      pendingAsyncPipelineSkips: this.telemetry.pendingAsyncPipelineSkips,
      suppressedMissingTextureErrors: this.telemetry.suppressedMissingTextureErrors,
      presentxBindGroupCacheHits: this.telemetry.presentxBindGroupCacheHits,
      presentxBindGroupCacheMisses: this.telemetry.presentxBindGroupCacheMisses,
      fallbackSamplersBound: this.telemetry.fallbackSamplersBound,
      clampedViewports: this.telemetry.clampedViewports,
      clampedScissors: this.telemetry.clampedScissors,
      strippedNonBlendableBlends: this.telemetry.strippedNonBlendableBlends,
      sanitizedAttachments: this.telemetry.sanitizedAttachments,
      elidedViewportSets: this.telemetry.elidedViewportSets,
      elidedScissorSets: this.telemetry.elidedScissorSets,
      elidedBlendConstantSets: this.telemetry.elidedBlendConstantSets,
      elidedStencilRefSets: this.telemetry.elidedStencilRefSets,
      elidedBindGroupSets: this.telemetry.elidedBindGroupSets,
      textureViewCacheHits: this.telemetry.textureViewCacheHits,
      textureViewCacheMisses: this.telemetry.textureViewCacheMisses,
      bindingArrayLayouts: this.telemetry.bindingArrayLayouts,
      bindingArrayBindGroups: this.telemetry.bindingArrayBindGroups,
      rtArenaGroup3Binds: this.telemetry.rtArenaGroup3Binds,
      missingResourceFallbacks: this.telemetry.missingResourceFallbacks,
      bindGroupValidationFallbacks: this.telemetry.bindGroupValidationFallbacks,
      unknownBindGroupErrors: this.telemetry.unknownBindGroupErrors,
      skippedUnboundPassOps: this.telemetry.skippedUnboundPassOps,
      skippedInvalidIndirectOps: this.telemetry.skippedInvalidIndirectOps,
      warnings: this.telemetry.warnings.slice(),
    };
  }

  _recordTelemetryWarning(message) {
    if (typeof message !== 'string' || message.length === 0) return;
    if (this.telemetry.warnings.length >= MAX_TELEMETRY_WARNINGS) {
      this.telemetry.warnings.shift();
    }
    this.telemetry.warnings.push(message);
  }

  _recordTelemetryWarningOnce(key, message) {
    if (typeof key === 'string' && key.length > 0) {
      if (this._seenWarningKeys.has(key)) return;
      if (this._seenWarningKeys.size >= MAX_SEEN_WARNING_KEYS) {
        const oldestKey = this._seenWarningKeys.values().next().value;
        if (oldestKey !== undefined) {
          this._seenWarningKeys.delete(oldestKey);
        }
      }
      this._seenWarningKeys.add(key);
    }
    this._recordTelemetryWarning(message);
  }

  _resolveEntryPoint(shaderEntry, stage) {
    const resolved = resolveShaderEntryPoint(shaderEntry, stage);
    if (resolved.remapped) {
      this.telemetry.legacyEntryPointRemaps += 1;
      this._recordTelemetryWarning(
        `legacy entry point remapped: stage=${stage} default=${resolved.defaultEntryPoint} resolved=${resolved.entryPoint} reason=${resolved.reason}`
      );
    }
    return resolved.entryPoint;
  }

  // Returns true when the opcode belongs to ABI v2 (handled, possibly with an error recorded).
  handlePacket(opcode, view, offset, payloadBytes) {
    this._syncDeviceGeneration();
    this._packetError = null;
    this._suppressPacketError = false;
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
      case V2_OPCODES.END_RENDER_PASS: {
        this.processor._passCommands.push(this._endRenderPassCmd);
        return this._count(true);
      }
      case V2_OPCODES.SET_VIEWPORT: {
        if (payloadBytes < VIEWPORT_PACKET_BYTES) return this._count(false);
        const vp = decodeViewportPacket(view, offset, this._scratch.viewport);
        if (!vp) return this._count(false);
        this.processor._passCommands.push(
          this._acquireSetViewportCommand(vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth)
        );
        return this._count(true);
      }
      case V2_OPCODES.SET_SCISSOR_RECT: {
        if (payloadBytes < SCISSOR_PACKET_BYTES) return this._count(false);
        const sc = decodeScissorPacket(view, offset, this._scratch.scissor);
        if (!sc) return this._count(false);
        this.processor._passCommands.push(
          this._acquireSetScissorRectCommand(sc.x, sc.y, sc.width, sc.height)
        );
        return this._count(true);
      }
      case V2_OPCODES.SET_BLEND_CONSTANT: {
        if (payloadBytes < BLEND_CONSTANTS_PACKET_BYTES) return this._count(false);
        const bc = decodeBlendConstantsPacket(view, offset, this._scratch.blendConstants);
        if (!bc) return this._count(false);
        this.processor._passCommands.push(
          this._acquireSetBlendConstantCommand(bc.r, bc.g, bc.b, bc.a)
        );
        return this._count(true);
      }
      case V2_OPCODES.SET_STENCIL_REFERENCE: {
        if (payloadBytes < STENCIL_REF_PACKET_BYTES) return this._count(false);
        const sr = decodeStencilRefPacket(view, offset, this._scratch.stencilRef);
        if (!sr) return this._count(false);
        this.processor._passCommands.push(this._acquireSetStencilReferenceCommand(sr.reference));
        return this._count(true);
      }
      case V2_OPCODES.CREATE_BIND_GROUP2:
        return this._count(this._createBindGroup(view, offset, payloadBytes));
      case V2_OPCODES.SET_BIND_GROUP2:
        return this._count(this._setBindGroup(view, offset, payloadBytes));
      case V2_OPCODES.CREATE_SAMPLER2:
        return this._count(payloadBytes >= 24 && this._createSampler(view, offset));
      case V2_OPCODES.CREATE_TEXTURE2:
        return this._count(this._createTexture2(view, offset, payloadBytes));
      case OPCODE_CLEAR_TEXTURE:
        return this._count(this._clearTexture(view, offset, payloadBytes));
      default:
        return false;
    }
  }

  _count(ok) {
    this.stats.packets += 1;
    if (!ok && this._suppressPacketError) {
      // W104: repeated error already reported once for this device generation.
      this._suppressPacketError = false;
      this.telemetry.suppressedMissingTextureErrors += 1;
      return true;
    }
    if (!ok) {
      this.stats.errors += 1;
      this.lastError = this._packetError || 'malformed or truncated v2 packet';
    }
    return true;
  }

  _acquireSetViewportCommand(x, y, width, height, minDepth, maxDepth) {
    const idx = this._vpCmdPoolUsed;
    this._vpCmdPoolUsed = idx + 1;
    let cmd;
    if (idx < this._vpCmdPool.length) {
      cmd = this._vpCmdPool[idx];
      cmd.x = x;
      cmd.y = y;
      cmd.width = width;
      cmd.height = height;
      cmd.minDepth = minDepth;
      cmd.maxDepth = maxDepth;
    } else {
      cmd = new SetViewportCommand(this, x, y, width, height, minDepth, maxDepth);
      this._vpCmdPool.push(cmd);
    }
    return cmd;
  }

  _acquireSetScissorRectCommand(x, y, width, height) {
    const idx = this._scCmdPoolUsed;
    this._scCmdPoolUsed = idx + 1;
    let cmd;
    if (idx < this._scCmdPool.length) {
      cmd = this._scCmdPool[idx];
      cmd.x = x;
      cmd.y = y;
      cmd.width = width;
      cmd.height = height;
    } else {
      cmd = new SetScissorRectCommand(this, x, y, width, height);
      this._scCmdPool.push(cmd);
    }
    return cmd;
  }

  _acquireSetBlendConstantCommand(r, g, b, a) {
    const idx = this._bcCmdPoolUsed;
    this._bcCmdPoolUsed = idx + 1;
    let cmd;
    if (idx < this._bcCmdPool.length) {
      cmd = this._bcCmdPool[idx];
      cmd.r = r;
      cmd.g = g;
      cmd.b = b;
      cmd.a = a;
    } else {
      cmd = new SetBlendConstantCommand(this, r, g, b, a);
      this._bcCmdPool.push(cmd);
    }
    return cmd;
  }

  _acquireSetStencilReferenceCommand(reference) {
    const idx = this._srCmdPoolUsed;
    this._srCmdPoolUsed = idx + 1;
    let cmd;
    if (idx < this._srCmdPool.length) {
      cmd = this._srCmdPool[idx];
      cmd.reference = reference;
    } else {
      cmd = new SetStencilReferenceCommand(this, reference);
      this._srCmdPool.push(cmd);
    }
    return cmd;
  }

  _acquireSetBindGroup2Command(groupIndex, bindGroupId, bindGroup, bindGroupDesc, offsets) {
    const idx = this._sbg2CmdPoolUsed;
    this._sbg2CmdPoolUsed = idx + 1;
    let cmd;
    if (idx < this._sbg2CmdPool.length) {
      cmd = this._sbg2CmdPool[idx];
      cmd.groupIndex = groupIndex;
      cmd.bindGroupId = bindGroupId;
      cmd.bindGroup = bindGroup;
      cmd.bindGroupDesc = bindGroupDesc;
      cmd.offsets = offsets;
      cmd._wgpuMeta = cmd;
    } else {
      cmd = new SetBindGroup2Command(this, groupIndex, bindGroupId, bindGroup, bindGroupDesc, offsets);
      this._sbg2CmdPool.push(cmd);
    }
    return cmd;
  }

  _acquireRemappedSetBindGroupCommand(bindGroup, bindGroupId) {
    const idx = this._remappedSbgCmdPoolUsed;
    this._remappedSbgCmdPoolUsed = idx + 1;
    let cmd;
    if (idx < this._remappedSbgCmdPool.length) {
      cmd = this._remappedSbgCmdPool[idx];
      cmd.bindGroup = bindGroup;
      cmd.bindGroupId = bindGroupId || 0;
    } else {
      cmd = new RemappedSetBindGroupCommand(this, bindGroup, bindGroupId);
      this._remappedSbgCmdPool.push(cmd);
    }
    return cmd;
  }

  _acquireNoopRenderGroup0Command(bindGroupId) {
    const idx = this._noopGroup0CmdPoolUsed;
    this._noopGroup0CmdPoolUsed = idx + 1;
    let cmd;
    if (idx < this._noopGroup0CmdPool.length) {
      cmd = this._noopGroup0CmdPool[idx];
      cmd.bindGroupId = bindGroupId || 0;
    } else {
      cmd = new NoopRenderGroup0Command(bindGroupId);
      this._noopGroup0CmdPool.push(cmd);
    }
    return cmd;
  }

  _acquireDynamicOffsetsArray(count) {
    const idx = this._dynOffsetsPoolUsed;
    this._dynOffsetsPoolUsed = idx + 1;
    let arr;
    if (idx < this._dynOffsetsPool.length) {
      arr = this._dynOffsetsPool[idx];
      arr.length = count;
    } else {
      arr = new Array(count);
      this._dynOffsetsPool.push(arr);
    }
    return arr;
  }

  _peekBeginRenderPassCommand() {
    const idx = this._brpCmdPoolUsed;
    if (idx < this._brpCmdPool.length) {
      return this._brpCmdPool[idx];
    }
    const cmd = new BeginRenderPassCommand(this, null, null, false, null);
    this._brpCmdPool.push(cmd);
    return cmd;
  }

  _releaseFramePools() {
    for (let i = 0; i < this._sbg2CmdPoolUsed; i += 1) {
      const cmd = this._sbg2CmdPool[i];
      cmd.bindGroup = null;
      cmd.bindGroupDesc = null;
      cmd.offsets = EMPTY_DYNAMIC_OFFSETS;
      cmd._wgpuMeta = cmd;
    }
    for (let i = 0; i < this._remappedSbgCmdPoolUsed; i += 1) {
      this._remappedSbgCmdPool[i].bindGroup = null;
    }
    for (let i = 0; i < this._brpCmdPoolUsed; i += 1) {
      const cmd = this._brpCmdPool[i];
      cmd._resolvers.colors.length = 0;
      cmd._resolvers.depthStencil = null;
      cmd._decoded.colors.length = 0;
      cmd._decoded.depthStencil = null;
      cmd._targetIsSwapchain = false;
      cmd._staticTargetSize = null;
    }
    this._vpCmdPoolUsed = 0;
    this._scCmdPoolUsed = 0;
    this._bcCmdPoolUsed = 0;
    this._srCmdPoolUsed = 0;
    this._sbg2CmdPoolUsed = 0;
    this._remappedSbgCmdPoolUsed = 0;
    this._noopGroup0CmdPoolUsed = 0;
    this._brpCmdPoolUsed = 0;
    this._dynOffsetsPoolUsed = 0;
    const MAX_POOL = 2048;
    if (this._vpCmdPool.length > MAX_POOL) this._vpCmdPool.length = MAX_POOL;
    if (this._scCmdPool.length > MAX_POOL) this._scCmdPool.length = MAX_POOL;
    if (this._bcCmdPool.length > MAX_POOL) this._bcCmdPool.length = MAX_POOL;
    if (this._srCmdPool.length > MAX_POOL) this._srCmdPool.length = MAX_POOL;
    if (this._sbg2CmdPool.length > MAX_POOL) this._sbg2CmdPool.length = MAX_POOL;
    if (this._remappedSbgCmdPool.length > MAX_POOL) this._remappedSbgCmdPool.length = MAX_POOL;
    if (this._noopGroup0CmdPool.length > MAX_POOL) this._noopGroup0CmdPool.length = MAX_POOL;
    if (this._brpCmdPool.length > MAX_POOL) this._brpCmdPool.length = MAX_POOL;
    if (this._dynOffsetsPool.length > MAX_POOL) this._dynOffsetsPool.length = MAX_POOL;
  }

  _parseRemappedBgKeyPipeline(key) {
    const firstColon = key.indexOf(':');
    if (firstColon < 0) return '';
    const secondColon = key.indexOf(':', firstColon + 1);
    return secondColon > firstColon ? key.slice(0, secondColon) : '';
  }

  _parseRemappedBgKeyBindGroupId(key) {
    const at = key.indexOf('@');
    const baseEnd = at >= 0 ? at : key.length;
    const lastColon = key.lastIndexOf(':', baseEnd - 1);
    if (lastColon < 0) return 0;
    const raw = key.slice(lastColon + 1, baseEnd);
    const num = Number(raw);
    return Number.isNaN(num) ? raw : num;
  }

  _addRemappedBgIndexes(cacheKey, pipelineKey, bindGroupId) {
    const bgKey = bindGroupId !== undefined ? bindGroupId : this._parseRemappedBgKeyBindGroupId(cacheKey);
    const pKey = pipelineKey !== undefined ? pipelineKey : this._parseRemappedBgKeyPipeline(cacheKey);
    let bgSet = this._remappedBgByBindGroupId.get(bgKey);
    if (!bgSet) {
      bgSet = new Set();
      this._remappedBgByBindGroupId.set(bgKey, bgSet);
    }
    bgSet.add(cacheKey);
    if (pKey) {
      let pipeSet = this._remappedBgByPipelineKey.get(pKey);
      if (!pipeSet) {
        pipeSet = new Set();
        this._remappedBgByPipelineKey.set(pKey, pipeSet);
      }
      pipeSet.add(cacheKey);
    }
  }

  _removeRemappedBgFromPipelineIndex(cacheKey) {
    const pKey = this._parseRemappedBgKeyPipeline(cacheKey);
    if (!pKey) return;
    const pipeSet = this._remappedBgByPipelineKey.get(pKey);
    if (pipeSet) {
      pipeSet.delete(cacheKey);
      if (pipeSet.size === 0) {
        this._remappedBgByPipelineKey.delete(pKey);
      }
    }
  }

  _removeRemappedBgFromBindGroupIndex(cacheKey) {
    const bgKey = this._parseRemappedBgKeyBindGroupId(cacheKey);
    const bgSet = this._remappedBgByBindGroupId.get(bgKey);
    if (bgSet) {
      bgSet.delete(cacheKey);
      if (bgSet.size === 0) {
        this._remappedBgByBindGroupId.delete(bgKey);
      }
    }
  }

  _invalidateRemappedBindGroupsForPipeline(pipelineId, isCompute) {
    if (this.remappedBindGroups.size === 0) return;
    const pipelineKey = `${isCompute ? 'c' : 'r'}:${pipelineId}`;
    const keys = this._remappedBgByPipelineKey.get(pipelineKey);
    if (!keys) return;
    this._remappedBgByPipelineKey.delete(pipelineKey);
    for (const key of keys) {
      this.remappedBindGroups.delete(key);
      this._removeRemappedBgFromBindGroupIndex(key);
    }
  }

  _invalidateRemappedBindGroupsForBindGroup(bindGroupId) {
    if (this.remappedBindGroups.size === 0) return;
    const keys = this._remappedBgByBindGroupId.get(bindGroupId);
    if (!keys) return;
    this._remappedBgByBindGroupId.delete(bindGroupId);
    for (const key of keys) {
      this.remappedBindGroups.delete(key);
      this._removeRemappedBgFromPipelineIndex(key);
    }
  }

  // LRU insert: re-inserting moves the key to the most-recently-used end; the oldest entries are
  // evicted once the cache exceeds remappedBindGroupCacheLimit.
  _cacheRemappedBindGroup(cacheKey, bindGroup, pipelineKey, bindGroupId) {
    const existed = this.remappedBindGroups.has(cacheKey);
    if (existed) {
      this.remappedBindGroups.delete(cacheKey);
    }
    this.remappedBindGroups.set(cacheKey, bindGroup);
    if (!existed) {
      this._addRemappedBgIndexes(cacheKey, pipelineKey, bindGroupId);
    }
    while (this.remappedBindGroups.size > this.remappedBindGroupCacheLimit) {
      const oldest = this.remappedBindGroups.keys().next().value;
      this.remappedBindGroups.delete(oldest);
      this._removeRemappedBgFromBindGroupIndex(oldest);
      this._removeRemappedBgFromPipelineIndex(oldest);
      this.telemetry.remappedBindGroupEvictions += 1;
    }
  }

  // W186/W354: detect GPUDevice replacement or device-generation bump and flush device-bound caches.
  _syncDeviceGeneration() {
    const liveDevice = (this.processor && this.processor.device) || this.device;
    const liveGen =
      this.processor && Number.isInteger(this.processor._deviceGeneration)
        ? this.processor._deviceGeneration
        : this._cachedDeviceGeneration;
    if (liveDevice !== this.device || liveGen !== this._cachedDeviceGeneration) {
      this.invalidatePresentXCache();
      this.textureViews.clear();
      this._textureViewWeakCache = new WeakMap();
      this._pipelineBglCache = new WeakMap();
      this._textureViewOwners.clear();
      this.dummies.clear();
      this.layoutsByContent.clear();
      this.remappedBindGroups.clear();
      this._remappedBgByBindGroupId.clear();
      this._remappedBgByPipelineKey.clear();
      this._releaseFramePools();
      this.emptyLayout = null;
      this.emptyFragment = null;
      this._resetPassState(null);
      this.device = liveDevice;
      this._cachedDeviceGeneration = liveGen;
      return true;
    }
    return false;
  }

  _bumpTextureGeneration(textureId) {
    this._nextTextureGeneration += 1;
    const next = this._nextTextureGeneration;
    if (this._textureGenerations.has(textureId)) {
      this._textureGenerations.delete(textureId);
    } else if (this._textureGenerations.size >= MAX_TEXTURE_GENERATIONS) {
      const oldestTextureId = this._textureGenerations.keys().next().value;
      if (oldestTextureId !== undefined) {
        this._textureGenerations.delete(oldestTextureId);
      }
    }
    this._textureGenerations.set(textureId, next);
    return next;
  }

  _getTextureGeneration(textureId, liveTexture) {
    const prevOwner = this._textureViewOwners.get(textureId);
    if (liveTexture && prevOwner && prevOwner !== liveTexture) {
      this._textureViewWeakCache.delete(prevOwner);
      this.textureViews.delete(textureId);
      this._invalidatePresentXCacheForResource(textureId);
      this._textureViewOwners.set(textureId, liveTexture);
      return this._bumpTextureGeneration(textureId);
    }
    if (liveTexture && !prevOwner) {
      this._textureViewOwners.set(textureId, liveTexture);
    }
    return this._textureGenerations.get(textureId) || 0;
  }

  _getPipelineCache() {
    if (this.pipelineCache && typeof this.pipelineCache === 'object') {
      return this.pipelineCache;
    }
    if (this.processor && this.processor.pipelineCache && typeof this.processor.pipelineCache === 'object') {
      return this.processor.pipelineCache;
    }
    return null;
  }

  _resolveDefaultSamplerFromCache(samplerKind) {
    const cache = this._getPipelineCache();
    if (!cache || typeof cache.getOrCreateDefaultSampler !== 'function') {
      return null;
    }
    const kind =
      samplerKind === 'comparison' || samplerKind === 'non-filtering' || samplerKind === 'filtering'
        ? samplerKind
        : 'filtering';
    try {
      const sampler = cache.getOrCreateDefaultSampler(kind);
      return sampler || null;
    } catch (err) {
      this._recordTelemetryWarningOnce(
        `fallback-sampler-error:${kind}`,
        `pipelineCache.getOrCreateDefaultSampler(${kind}) failed: ${String(err && err.message ? err.message : err)}`
      );
      return null;
    }
  }

  _resolveStorageFormat(format) {
    return resolveStorageTextureFormat(format, this.device, this._textureFormatsOptions);
  }

  _cachePresentXBindGroup(cacheKey, entry) {
    if (this.presentxBindGroupCache.has(cacheKey)) {
      this.presentxBindGroupCache.delete(cacheKey);
    }
    this.presentxBindGroupCache.set(cacheKey, entry);
    while (this.presentxBindGroupCache.size > this.presentxBindGroupCacheLimit) {
      const oldest = this.presentxBindGroupCache.keys().next().value;
      this.presentxBindGroupCache.delete(oldest);
    }
  }

  _invalidatePresentXCacheForResource(resourceId) {
    if (this.presentxBindGroupCache.size === 0) return 0;
    let removed = 0;
    for (const [key, entry] of this.presentxBindGroupCache.entries()) {
      if (
        entry.planeTextureId === resourceId ||
        entry.samplerId === resourceId ||
        entry.layoutId === resourceId ||
        entry.pipelineId === resourceId
      ) {
        this.presentxBindGroupCache.delete(key);
        removed += 1;
      }
    }
    return removed;
  }

  invalidatePresentXCache(filter = null) {
    if (!filter || typeof filter !== 'object') {
      const count = this.presentxBindGroupCache.size;
      this.presentxBindGroupCache.clear();
      return count;
    }
    let removed = 0;
    for (const [key, entry] of this.presentxBindGroupCache.entries()) {
      if (
        (filter.planeTextureId !== undefined && entry.planeTextureId === filter.planeTextureId) ||
        (filter.samplerId !== undefined && entry.samplerId === filter.samplerId) ||
        (filter.sampler !== undefined && entry.sampler === filter.sampler) ||
        (filter.layoutId !== undefined && entry.layoutId === filter.layoutId) ||
        (filter.layout !== undefined && entry.layout === filter.layout) ||
        (filter.pipelineId !== undefined && entry.pipelineId === filter.pipelineId) ||
        (filter.pipeline !== undefined && entry.pipeline === filter.pipeline)
      ) {
        this.presentxBindGroupCache.delete(key);
        removed += 1;
      }
    }
    return removed;
  }

  // W186/W354: hot-path PresentX multi-plane bind-group + texture-view cache.
  // Caches per-(planeTextureId, viewFormat, sampler, pipeline/layout) across frames and invalidates
  // automatically on texture handle/generation/destroyed change, device change, reset, or destroy.
  getOrCreatePresentXPlaneBindGroup(params) {
    if (this._destroyed || !params || typeof params !== 'object') return null;
    this._syncDeviceGeneration();
    if (!this.device || typeof this.device.createBindGroup !== 'function') return null;

    const planeTextureId = params.planeTextureId;
    if (!Number.isInteger(planeTextureId) || planeTextureId <= 0 || planeTextureId === SWAPCHAIN_TEXTURE_ID) {
      return null;
    }
    const texture =
      this.processor && this.processor.textures ? this.processor.textures.get(planeTextureId) : null;
    if (!texture || texture.destroyed === true || typeof texture.createView !== 'function') return null;

    const texGen = this._getTextureGeneration(planeTextureId, texture);
    const hasParamViewDesc = Boolean(params.viewDesc && typeof params.viewDesc === 'object');
    const hasParamViewFormat =
      typeof params.viewFormat === 'string' &&
      params.viewFormat.length > 0 &&
      params.viewFormat !== 'undefined';
    let baseViewDesc;
    if (hasParamViewDesc) {
      if (hasParamViewFormat && !params.viewDesc.format) {
        baseViewDesc = { ...params.viewDesc, format: params.viewFormat };
      } else {
        baseViewDesc = params.viewDesc;
      }
    } else if (hasParamViewFormat) {
      const scratchDesc = this._scratchSwapchainFormatDesc;
      scratchDesc.format = params.viewFormat;
      baseViewDesc = scratchDesc;
    } else {
      baseViewDesc = null;
    }
    const resolvedViewFormat =
      this._resolveTextureFormat(planeTextureId, baseViewDesc) ||
      (hasParamViewFormat ? params.viewFormat : 'default');
    const viewDescKey = buildTextureViewKey(baseViewDesc);

    const samplerId = Number.isInteger(params.samplerId) && params.samplerId > 0 ? params.samplerId : 0;
    let samplerObj = params.sampler || null;
    let usedFallbackSampler = false;
    if (!samplerObj && samplerId > 0 && this.processor && this.processor.samplers) {
      samplerObj = this.processor.samplers.get(samplerId) || null;
    }
    if (!samplerObj || samplerObj.destroyed === true) {
      const samplerKind =
        params.samplerKind === 'comparison' || params.samplerKind === 'non-filtering'
          ? params.samplerKind
          : 'filtering';
      samplerObj = this._resolveDefaultSamplerFromCache(samplerKind);
      if (samplerObj) {
        usedFallbackSampler = true;
      } else {
        samplerObj = this._dummyResource({ sampler: { type: samplerKind } });
      }
    }
    if (!samplerObj) return null;

    const pipelineId = Number.isInteger(params.pipelineId) && params.pipelineId > 0 ? params.pipelineId : 0;
    let pipelineObj = params.pipeline || null;
    if (!pipelineObj && pipelineId > 0 && this.processor && this.processor.pipelines) {
      pipelineObj = this.processor.pipelines.get(pipelineId) || null;
    }

    const layoutId = Number.isInteger(params.layoutId) && params.layoutId > 0 ? params.layoutId : 0;
    let bindGroupLayout = params.layout || null;
    if (!bindGroupLayout && layoutId > 0) {
      const bglRec = this.bindGroupLayouts.get(layoutId);
      bindGroupLayout = bglRec ? bglRec.layout : null;
    }
    if (!bindGroupLayout && pipelineObj) {
      bindGroupLayout = this._getPipelineGroup0Layout(pipelineObj);
    }
    if (!bindGroupLayout) return null;

    const textureBinding = Number.isInteger(params.textureBinding) ? params.textureBinding : 0;
    const samplerBinding = Number.isInteger(params.samplerBinding) ? params.samplerBinding : 1;
    const cacheKey = `px:${planeTextureId}:${texGen}:${resolvedViewFormat}:${viewDescKey}:s=${gpuObjectId(
      samplerObj
    )}:p=${gpuObjectId(pipelineObj)}:l=${gpuObjectId(bindGroupLayout)}:b=${textureBinding},${samplerBinding}`;

    const cached = this.presentxBindGroupCache.get(cacheKey);
    if (
      cached &&
      cached.device === this.device &&
      cached.deviceGeneration === this._cachedDeviceGeneration &&
      cached.texture === texture &&
      cached.texture.destroyed !== true &&
      cached.view &&
      cached.view.destroyed !== true &&
      cached.bindGroup &&
      cached.bindGroup.destroyed !== true &&
      cached.textureGen === texGen &&
      cached.sampler === samplerObj &&
      cached.sampler.destroyed !== true &&
      cached.layout === bindGroupLayout &&
      cached.pipeline === (pipelineObj || null)
    ) {
      this._cachePresentXBindGroup(cacheKey, cached);
      this.telemetry.presentxBindGroupCacheHits += 1;
      if (usedFallbackSampler) {
        this.telemetry.fallbackSamplersBound += 1;
      }
      return { view: cached.view, bindGroup: cached.bindGroup, hit: true, cacheKey };
    }

    const view = this._textureView(planeTextureId, baseViewDesc || {});
    if (!view) return null;

    let bindGroup;
    try {
      bindGroup = this.device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: textureBinding, resource: view },
          { binding: samplerBinding, resource: samplerObj },
        ],
      });
    } catch (err) {
      this._packetError = `PresentX bind group (texture ${planeTextureId}): ${String(
        err && err.message ? err.message : err
      )}`;
      return null;
    }

    if (usedFallbackSampler) {
      this.telemetry.fallbackSamplersBound += 1;
    }
    this.telemetry.presentxBindGroupCacheMisses += 1;
    const entry = {
      view,
      bindGroup,
      planeTextureId,
      texture,
      textureGen: texGen,
      viewFormat: resolvedViewFormat,
      viewDescKey,
      samplerId,
      sampler: samplerObj,
      layoutId,
      layout: bindGroupLayout,
      pipelineId,
      pipeline: pipelineObj || null,
      device: this.device,
      deviceGeneration: this._cachedDeviceGeneration,
    };
    this._cachePresentXBindGroup(cacheKey, entry);
    return { view, bindGroup, hit: false, cacheKey };
  }

  reset() {
    this.invalidatePresentXCache();
    this.textureViews.clear();
    this._textureViewWeakCache = new WeakMap();
    this._pipelineBglCache = new WeakMap();
    this._textureViewOwners.clear();
    this.remappedBindGroups.clear();
    this._remappedBgByBindGroupId.clear();
    this._remappedBgByPipelineKey.clear();
    this._releaseFramePools();
    this.dummies.clear();
    this.emptyLayout = null;
    this.emptyFragment = null;
    this._activePass = null;
    this._activePassTargetSize = null;
    this._lastPassReferenceSize = null;
    this._resetPassState(null);
  }

  destroy() {
    this.reset();
    this.bindGroupLayouts.clear();
    this.pipelineLayouts.clear();
    this.pipelineLayoutMeta.clear();
    this.computePipelineMeta.clear();
    this.renderPipelineAutoLayoutMeta.clear();
    this.pendingAsyncRenderPipelines.clear();
    this.pendingAsyncComputePipelines.clear();
    this.bindGroupDescriptors.clear();
    this.samplerMeta.clear();
    this.layoutsByContent.clear();
    this.textureMeta.clear();
    this._textureGenerations.clear();
    this._reportedMissingTextures.clear();
    this._destroyed = true;
  }

  // DestroyResource hook: forget layouts, metadata and cached views of `id`.
  destroyResource(id) {
    this.bindGroupLayouts.delete(id);
    this.pipelineLayouts.delete(id);
    this.pipelineLayoutMeta.delete(id);
    this.computePipelineMeta.delete(id);
    this.renderPipelineAutoLayoutMeta.delete(id);
    this.pendingAsyncRenderPipelines.delete(id);
    this.pendingAsyncComputePipelines.delete(id);
    this.bindGroupDescriptors.delete(id);
    this.samplerMeta.delete(id);
    this._invalidateRemappedBindGroupsForPipeline(id, true);
    this._invalidateRemappedBindGroupsForPipeline(id, false);
    this._invalidateRemappedBindGroupsForBindGroup(id);
    const wasTrackedTexture =
      this.textureMeta.has(id) ||
      this.textureViews.has(id) ||
      this._textureViewOwners.has(id) ||
      this._textureGenerations.has(id) ||
      Boolean(this.processor && this.processor.textures && this.processor.textures.has(id));
    this._invalidatePresentXCacheForResource(id);
    if (wasTrackedTexture) {
      this._bumpTextureGeneration(id);
    }
    const prevOwner = this._textureViewOwners.get(id);
    if (prevOwner) {
      this._textureViewWeakCache.delete(prevOwner);
      this._textureViewOwners.delete(id);
    }
    const liveTex = this.processor && this.processor.textures ? this.processor.textures.get(id) : null;
    if (liveTex) {
      this._textureViewWeakCache.delete(liveTex);
    }
    this.textureViews.delete(id);
    this.textureMeta.delete(id);
    if (
      this.processor &&
      this.processor._pipelineMeta &&
      typeof this.processor._pipelineMeta.delete === 'function'
    ) {
      this.processor._pipelineMeta.delete(id);
    }
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
    const hasBindingArrays = (flags & BINDING_ARRAYS_LAYOUT_FLAG) !== 0;
    if ((flags & MANIFEST_REFERENCE_FLAG) !== 0) {
      const layout = this.resolveManifestLayout && this.resolveManifestLayout('bindGroupLayout', contentId);
      if (!layout) {
        this._packetError = `manifest bind group layout 0x${contentId.toString(16)} unavailable`;
        return false;
      }
      this.bindGroupLayouts.set(id, {
        layout,
        entries: null,
        entryByBinding: null,
        hasBindingArrays,
        hasRtArenaBindings: false,
      });
      if (hasBindingArrays) this.telemetry.bindingArrayLayouts += 1;
      return true;
    }
    if (24 + count * 24 > payloadBytes) return false;
    const entries = [];
    const entryByBinding = new Map();
    let sequentialBindings = true;
    let hasRtArenaBindings = false;
    for (let i = 0; i < count; i++) {
      const entry = decodeBindGroupLayoutEntry(view, offset + 24 + i * 24);
      if (!entry) return false;
      if (entryByBinding.has(entry.binding)) {
        this._packetError = `bind group layout ${id}: duplicate binding ${entry.binding}`;
        return false;
      }
      if (entry.texture && entry.texture.multisampled) {
        // WebGPU requires multisampled texture bindings to use viewDimension '2d' and non-filtering sampleType.
        if (entry.texture.viewDimension !== '2d') {
          const origDim = entry.texture.viewDimension;
          entry.texture.viewDimension = '2d';
          this._recordTelemetryWarningOnce(
            `bgl-msaa-dim:${origDim}`,
            `bind group layout multisampled texture viewDimension ${origDim} invalid; coercing to 2d`
          );
        }
        if (entry.texture.sampleType === 'float') {
          entry.texture.sampleType = 'unfilterable-float';
          this._recordTelemetryWarningOnce(
            'bgl-msaa-sample-type',
            'bind group layout multisampled texture sampleType float invalid; coercing to unfilterable-float'
          );
        }
      }
      if (entry.storageTexture) {
        const origFmt = entry.storageTexture.format;
        const resolvedStorage = this._resolveStorageFormat(origFmt);
        if (resolvedStorage.supportsStorageBinding && resolvedStorage.storageFormat) {
          if (resolvedStorage.storageFormat !== origFmt) {
            entry.storageTexture.format = resolvedStorage.storageFormat;
            this._recordTelemetryWarningOnce(
              `bgl-storage-format:${origFmt}`,
              `bind group layout storageTexture format ${origFmt} remapped to linear sibling ${resolvedStorage.storageFormat}`
            );
          }
        } else {
          entry.storageTexture.format = 'rgba8unorm';
          this._recordTelemetryWarningOnce(
            `bgl-storage-format:${origFmt}`,
            `bind group layout storageTexture format ${origFmt} unsupported${
              resolvedStorage.reason ? ` (${resolvedStorage.reason})` : ''
            }; falling back to rgba8unorm`
          );
        }
        if (
          entry.storageTexture.access === 'read-write' &&
          !isStorageAccessCompatible(entry.storageTexture.format, 'read-write', this.device)
        ) {
          const fmt = entry.storageTexture.format;
          entry.storageTexture.access = 'write-only';
          this._recordTelemetryWarningOnce(
            `bgl-storage-rw:${fmt}`,
            `bind group layout storageTexture format ${fmt} does not support read-write access; downgrading to write-only`
          );
        }
        if (
          entry.storageTexture.viewDimension === 'cube' ||
          entry.storageTexture.viewDimension === 'cube-array'
        ) {
          const origDim = entry.storageTexture.viewDimension;
          entry.storageTexture.viewDimension = '2d-array';
          this._recordTelemetryWarningOnce(
            `bgl-storage-dim:${origDim}`,
            `bind group layout storageTexture dimension ${origDim} unsupported; remapping to 2d-array`
          );
        }
      }
      entryByBinding.set(entry.binding, entry);
      if (entry.binding !== i) sequentialBindings = false;
      if (isRtArenaBinding(entry.binding)) hasRtArenaBindings = true;
      entries.push(entry);
    }
    const contentKey = contentId !== 0n ? `bgl:${contentId.toString(16)}` : null;
    let layout = contentKey ? this.layoutsByContent.get(contentKey) : undefined;
    if (!layout) {
      try {
        layout = this.device.createBindGroupLayout({ entries });
      } catch (err) {
        this._packetError = `bind group layout ${id}: ${String(err && err.message ? err.message : err)}`;
        return false;
      }
      if (contentKey) {
        if (this.layoutsByContent.has(contentKey)) {
          this.layoutsByContent.delete(contentKey);
        } else if (this.layoutsByContent.size >= MAX_LAYOUTS_BY_CONTENT) {
          const oldestContentKey = this.layoutsByContent.keys().next().value;
          if (oldestContentKey !== undefined) {
            this.layoutsByContent.delete(oldestContentKey);
          }
        }
        this.layoutsByContent.set(contentKey, layout);
      }
    }
    this.bindGroupLayouts.set(id, {
      layout,
      entries,
      entryByBinding,
      hasBindingArrays,
      hasRtArenaBindings,
      sequentialBindings,
      dynamicOffsetIndices: dynamicOffsetIndicesForLayoutEntries(entries),
    });
    if (hasBindingArrays) this.telemetry.bindingArrayLayouts += 1;
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
    let meta = {
      groupCount: count,
      isSingleGroup0: false,
      sequentialGroup0: false,
      group0HasEntries: false,
      group0Entries: null,
      group3HasEntries: false,
      hasRtArenaGroup3: false,
      requiredGroupsMask: 0,
      groupBglIds: null,
    };
    if ((flags & MANIFEST_REFERENCE_FLAG) !== 0) {
      layout = this.resolveManifestLayout && this.resolveManifestLayout('pipelineLayout', contentId);
      if (!layout) {
        this._packetError = `manifest pipeline layout 0x${contentId.toString(16)} unavailable`;
        return false;
      }
    } else {
      if (24 + count * 4 > payloadBytes) return false;
      const explicitMaxGroups =
        this.device &&
        this.device.limits &&
        typeof this.device.limits.maxBindGroups === 'number' &&
        this.device.limits.maxBindGroups >= 1
          ? Math.floor(this.device.limits.maxBindGroups)
          : null;
      const effectiveGroupCount =
        explicitMaxGroups !== null && count > explicitMaxGroups ? explicitMaxGroups : count;
      if (effectiveGroupCount < count) {
        this._recordTelemetryWarningOnce(
          `pl-max-bind-groups:${count}:${explicitMaxGroups}`,
          `pipeline layout ${id}: clamped group count ${count} to device limit maxBindGroups=${explicitMaxGroups}`
        );
      }
      const bindGroupLayouts = [];
      const groupBglIds = new Array(effectiveGroupCount);
      let requiredGroupsMask = 0;
      let group0Record = null;
      let group3Record = null;
      for (let g = 0; g < effectiveGroupCount; g++) {
        const bglId = view.getUint32(offset + 24 + g * 4, true);
        const record = bglId !== 0 ? this.bindGroupLayouts.get(bglId) : null;
        if (bglId !== 0 && !record) {
          this._packetError = `pipeline layout ${id}: unknown bind group layout ${bglId}`;
          return false;
        }
        groupBglIds[g] = bglId;
        if (
          g < 31 &&
          record &&
          (record.entries === null || (Array.isArray(record.entries) && record.entries.length > 0))
        ) {
          requiredGroupsMask |= 1 << g;
        }
        if (g === 0) group0Record = record;
        else if (g === RT_ARENA_GROUP_INDEX) group3Record = record;
        bindGroupLayouts.push(record ? record.layout : this._emptyBindGroupLayout());
      }
      try {
        layout = this.device.createPipelineLayout({ bindGroupLayouts });
      } catch (err) {
        this._packetError = `pipeline layout ${id}: ${String(err && err.message ? err.message : err)}`;
        return false;
      }
      meta = {
        groupCount: effectiveGroupCount,
        isSingleGroup0: effectiveGroupCount === 1 && group0Record !== null,
        sequentialGroup0: Boolean(effectiveGroupCount === 1 && group0Record && group0Record.sequentialBindings),
        group0HasEntries: Boolean(
          group0Record &&
            (group0Record.entries === null ||
              (Array.isArray(group0Record.entries) && group0Record.entries.length > 0))
        ),
        group0Entries: group0Record ? group0Record.entries : null,
        group3HasEntries: Boolean(
          group3Record &&
            (group3Record.entries === null ||
              (Array.isArray(group3Record.entries) && group3Record.entries.length > 0))
        ),
        hasRtArenaGroup3: Boolean(group3Record && group3Record.hasRtArenaBindings),
        requiredGroupsMask,
        groupBglIds,
      };
    }
    this.pipelineLayouts.set(id, layout);
    this.pipelineLayoutMeta.set(id, meta);
    return true;
  }

  _layoutFor(layoutId) {
    if (layoutId === 0) return 'auto';
    const layout = this.pipelineLayouts.get(layoutId);
    if (!layout) this._packetError = `unknown pipeline layout ${layoutId}`;
    return layout || null;
  }

  // ---- Pipelines ----

  // W106: materialize lazy GPUShaderModule on first pipeline reference.
  _ensureShaderModule(id) {
    if (this.processor && typeof this.processor.ensureShaderModule === 'function') {
      return this.processor.ensureShaderModule(id);
    }
    return this.processor && this.processor.shaderModules
      ? this.processor.shaderModules.get(id) || null
      : null;
  }

  // The exact GPURenderPipelineDescriptor for a CreateRenderPipeline2 payload at `offset`.
  buildRenderPipelineDescriptor(view, offset) {
    this._lastBuiltRenderPipelineAutoLayout = null;
    const pipeId = view.getUint32(offset, true);
    const vsId = view.getUint32(offset + 4, true);
    const fsId = view.getUint32(offset + 8, true);
    const layoutId = view.getUint32(offset + 12, true);
    let layout = this._layoutFor(layoutId);
    const vs = this._ensureShaderModule(vsId);
    if (!vs || !layout) {
      if (!vs) this._packetError = `unknown vertex shader module ${vsId}`;
      return null;
    }
    const keyBase = offset + KEY_OFFSET;
    const key = decodeRenderPipelineKey(view, keyBase);
    const vertexCheck = validateVertexBufferLayouts(key.buffers, this.device);
    if (!vertexCheck.valid) {
      this._packetError = `render pipeline vertex layout invalid: ${vertexCheck.error}`;
      return null;
    }
    const expectedAttrs = Math.min(view.getUint32(keyBase + 176, true), 16);
    const attachedAttrs = key.buffers.reduce((n, b) => n + (b ? b.attributes.length : 0), 0);
    if (attachedAttrs !== expectedAttrs) {
      this._packetError = `render pipeline vertex layout dropped attributes (${attachedAttrs}/${expectedAttrs})`;
      return null;
    }
    for (let i = 0; i < key.targets.length; i += 1) {
      const target = key.targets[i];
      if (target && (!target.format || target.format === 'undefined')) {
        this._packetError = `render pipeline target ${i}: invalid texture format`;
        return null;
      }
    }
    let fs = null;
    let fragmentModule = null;
    let targets = key.targets;
    const fsEntryPoint = fsId !== 0 ? null : DEFAULT_ENTRY_POINTS.fragment;
    let resolvedFsEntryPoint = fsEntryPoint;
    if (fsId !== 0) {
      fs = this._ensureShaderModule(fsId);
      if (!fs) {
        this._packetError = `unknown fragment shader module ${fsId}`;
        return null;
      }
      fragmentModule = fs.module;
      resolvedFsEntryPoint = this._resolveEntryPoint(fs, 'fragment');
      if (Array.isArray(targets) && targets.length > 0) {
        const targetCount = targets.length;
        targets = targets.map((t, idx) => {
          if (!t) return null;
          const checked = sanitizeColorTargetBlendState(t, this.device, targetCount);
          if (checked.stripped) {
            this.telemetry.strippedNonBlendableBlends += 1;
            this._recordTelemetryWarningOnce(
              `strip-blend:${t.format}:${idx}`,
              `render pipeline ${pipeId} target ${idx}: stripped blend state on ${t.format} (${checked.reason})`
            );
          }
          return checked.target;
        });
      }
    } else if (key.colorTargetCount > 0) {
      // No pixel shader but color targets: an empty fragment stage that writes nothing.
      if (!this.emptyFragment) this.emptyFragment = this.device.createShaderModule({ code: EMPTY_FRAGMENT_WGSL });
      fragmentModule = this.emptyFragment;
      targets = targets.map((t) => (t ? { format: t.format, writeMask: 0 } : null));
    }

    // Dawn-strict primitive sanitization: gate unclippedDepth on depth-clip-control.
    if (key.primitive && key.primitive.unclippedDepth && !deviceHasFeature(this.device, 'depth-clip-control')) {
      delete key.primitive.unclippedDepth;
      this._recordTelemetryWarningOnce(
        'rp-unclipped-depth-unsupported',
        `render pipeline ${pipeId}: stripped unclippedDepth without depth-clip-control feature`
      );
    }

    // Dawn-strict depthStencil sanitization (#1904/#1940 parity).
    let depthStencil = key.depthStencil;
    if (depthStencil) {
      if (!isDepthOrStencilTextureFormat(depthStencil.format)) {
        this._recordTelemetryWarningOnce(
          `rp-invalid-ds-format:${depthStencil.format}`,
          `render pipeline ${pipeId}: dropped non-depth-stencil format ${depthStencil.format}`
        );
        depthStencil = undefined;
      } else {
        if (
          depthStencil.format === 'depth32float-stencil8' &&
          !deviceHasFeature(this.device, 'depth32float-stencil8')
        ) {
          depthStencil = { ...depthStencil, format: 'depth24plus-stencil8' };
          this._recordTelemetryWarningOnce(
            'rp-depth32float-stencil8-fallback',
            `render pipeline ${pipeId}: downgraded depth32float-stencil8 to depth24plus-stencil8`
          );
        }
        const topo = key.primitive ? key.primitive.topology : 'triangle-list';
        if (
          (topo === 'point-list' || topo === 'line-list' || topo === 'line-strip') &&
          (depthStencil.depthBias !== 0 ||
            depthStencil.depthBiasSlopeScale !== 0 ||
            depthStencil.depthBiasClamp !== 0)
        ) {
          depthStencil = {
            ...depthStencil,
            depthBias: 0,
            depthBiasSlopeScale: 0,
            depthBiasClamp: 0,
          };
          this._recordTelemetryWarningOnce(
            `rp-depth-bias-nontriangle:${topo}`,
            `render pipeline ${pipeId}: zeroed depthBias on non-triangle topology ${topo}`
          );
        }
        if (!hasStencilAspect(depthStencil.format)) {
          const sf = depthStencil.stencilFront;
          const sb = depthStencil.stencilBack;
          const nonDefaultStencil =
            (sf &&
              (sf.compare !== 'always' ||
                sf.failOp !== 'keep' ||
                sf.depthFailOp !== 'keep' ||
                sf.passOp !== 'keep')) ||
            (sb &&
              (sb.compare !== 'always' ||
                sb.failOp !== 'keep' ||
                sb.depthFailOp !== 'keep' ||
                sb.passOp !== 'keep'));
          if (nonDefaultStencil) {
            depthStencil = {
              ...depthStencil,
              stencilFront: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
              stencilBack: { compare: 'always', failOp: 'keep', depthFailOp: 'keep', passOp: 'keep' },
            };
          }
        }
      }
    }

    // Dawn-strict multisample sanitization: count must be 1 or 4 (and all targets must support MSAA),
    // and alphaToCoverageEnabled requires count > 1 plus an alpha-capable target 0.
    const multisample = key.multisample ? { ...key.multisample } : { count: 1, mask: 0xffffffff, alphaToCoverageEnabled: false };
    if (multisample.count > 1) {
      let msaaOk = multisample.count >= 4;
      if (msaaOk && Array.isArray(targets)) {
        for (let i = 0; i < targets.length; i += 1) {
          if (targets[i] && !isMultisampleSupported(targets[i].format, this.device)) {
            msaaOk = false;
            break;
          }
        }
      }
      if (msaaOk && depthStencil && !isMultisampleSupported(depthStencil.format, this.device)) {
        msaaOk = false;
      }
      multisample.count = msaaOk ? 4 : 1;
    } else {
      multisample.count = 1;
    }
    if (multisample.alphaToCoverageEnabled) {
      const target0 = Array.isArray(targets) && targets.length > 0 ? targets[0] : null;
      if (
        multisample.count <= 1 ||
        !target0 ||
        ALPHA_LESS_COLOR_FORMATS.has(target0.format)
      ) {
        multisample.alphaToCoverageEnabled = false;
      }
    }

    const vsSource = getShaderSourceText(vs);
    const fsSource = fs ? getShaderSourceText(fs) : null;
    const wgslBindings = mergeWgslGroup0Bindings(
      vsSource ? parseWgslGroup0Bindings(vsSource) : [],
      fsSource ? parseWgslGroup0Bindings(fsSource) : []
    );
    let autoLayoutRemap = false;
    const layoutMeta = layoutId !== 0 ? this.pipelineLayoutMeta.get(layoutId) : null;
    if (
      layoutId !== 0 &&
      layoutMeta &&
      !isExplicitLayoutCompatibleWithWgsl(layoutMeta.group0Entries, wgslBindings)
    ) {
      layout = 'auto';
      autoLayoutRemap = true;
      this._lastBuiltRenderPipelineAutoLayout = { autoLayoutRemap: true, wgslBindings };
      this._recordTelemetryWarningOnce(
        `auto-layout-remap:render:${pipeId}:${layoutId}`,
        `render pipeline ${pipeId}: explicit layout ${layoutId} incompatible with WGSL @group(0) bindings; falling back to layout:'auto'`
      );
    } else {
      this._lastBuiltRenderPipelineAutoLayout = { autoLayoutRemap: false, wgslBindings };
    }

    if (layoutId === 0 || autoLayoutRemap) {
      this.telemetry.autoLayoutFallbacks += 1;
    } else if (layoutMeta && layoutMeta.isSingleGroup0) {
      this.telemetry.rootSigGroup0Layouts += 1;
    }
    if (depthStencil && isDepthTextureFormat(depthStencil.format) && view.getUint8(keyBase + 122) === 0) {
      this.telemetry.depthCompareDefaults += 1;
    }

    const vsEntryPoint = this._resolveEntryPoint(vs, 'vertex');
    const descriptor = {
      layout,
      vertex: { module: vs.module, entryPoint: vsEntryPoint, buffers: key.buffers },
      primitive: key.primitive,
      multisample,
    };
    if (depthStencil) descriptor.depthStencil = depthStencil;
    if (fragmentModule) {
      descriptor.fragment = {
        module: fragmentModule,
        entryPoint: resolvedFsEntryPoint || DEFAULT_ENTRY_POINTS.fragment,
        targets,
      };
    }
    return descriptor;
  }

  _createRenderPipeline(view, offset) {
    const descriptor = this.buildRenderPipelineDescriptor(view, offset);
    if (!descriptor) return false;
    const autoLayoutInfo = this._lastBuiltRenderPipelineAutoLayout;
    this._lastBuiltRenderPipelineAutoLayout = null;
    const id = view.getUint32(offset, true);
    const vsId = view.getUint32(offset + 4, true);
    const fsId = view.getUint32(offset + 8, true);
    const layoutId = view.getUint32(offset + 12, true);
    const info = {
      vsId,
      fsId,
      layoutId,
      autoLayoutRemap: Boolean(autoLayoutInfo && autoLayoutInfo.autoLayoutRemap),
      wgslBindings: autoLayoutInfo && Array.isArray(autoLayoutInfo.wgslBindings) ? autoLayoutInfo.wgslBindings : [],
    };
    const asyncCache = this._asyncPipelineCacheFor('render');
    if (asyncCache) {
      return this._compileRenderPipelineAsync(asyncCache, id, descriptor, info);
    }
    let pipeline;
    try {
      pipeline = this.device.createRenderPipeline(descriptor);
    } catch (err) {
      this._packetError = `render pipeline ${id}: ${String(
        err && err.message ? err.message : err
      )}`;
      return false;
    }
    this._installRenderPipeline(id, pipeline, descriptor, info);
    return true;
  }

  // ---- W103: async pipeline compilation (processor.asyncPipelineCache) ----

  /**
   * The processor's AsyncPipelineCache when `kind` pipelines can compile asynchronously on this
   * device, else null (mock devices without create*PipelineAsync keep the synchronous path).
   */
  _asyncPipelineCacheFor(kind) {
    const cache = this.processor && this.processor.asyncPipelineCache;
    if (!cache || typeof cache.compileRenderPipeline !== 'function' || typeof cache.compileComputePipeline !== 'function') {
      return null;
    }
    const method = kind === 'compute' ? 'createComputePipelineAsync' : 'createRenderPipelineAsync';
    return this.device && typeof this.device[method] === 'function' ? cache : null;
  }

  isAsyncPipelinePending(id, isCompute) {
    if ((isCompute ? this.pendingAsyncComputePipelines : this.pendingAsyncRenderPipelines).has(id)) {
      return true;
    }
    const proc = this.processor;
    if (proc && proc._pendingRangedPipelines && proc._pendingRangedPipelines.has(id)) {
      const live = isCompute ? proc.computePipelines : proc.pipelines;
      if (!live || !live.has(id)) {
        const rec = proc._pendingRangedPipelines.get(id);
        const recIsCompute = Boolean(rec && (rec.kind === 'compute' || rec.kind === 'compute2'));
        if (!rec || recIsCompute === Boolean(isCompute)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Queue `descriptor` on the async cache; the pipeline is installed when the compile resolves.
   * Returns true: the packet is well formed and compile failures are reported asynchronously.
   *
   * @param {object} asyncCache
   * @param {'render' | 'compute'} kind
   * @param {number} id
   * @param {object} descriptor
   * @param {object | null} fallbackDescriptor - retried once (layout:'auto') if `descriptor` rejects.
   * @param {(pipeline: object, retried: boolean) => void} install
   */
  _compilePipelineAsync(asyncCache, kind, id, descriptor, fallbackDescriptor, install) {
    const isCompute = kind === 'compute';
    const pending = isCompute ? this.pendingAsyncComputePipelines : this.pendingAsyncRenderPipelines;
    const live = isCompute ? this.processor.computePipelines : this.processor.pipelines;
    // A re-created id must never draw with its previous pipeline or receive an older compile.
    if (typeof asyncCache.delete === 'function') asyncCache.delete(id);
    if (live && typeof live.delete === 'function') live.delete(id);
    const token = {};
    pending.set(id, token);
    this.telemetry.asyncPipelineCompiles += 1;
    const compile = isCompute ? asyncCache.compileComputePipeline : asyncCache.compileRenderPipeline;
    let promise;
    try {
      promise = Promise.resolve(compile.call(asyncCache, id, descriptor, { fallbackDescriptor }));
    } catch (err) {
      promise = Promise.resolve({ id, status: 'failed', error: err });
    }
    promise.then(
      (result) => this._onAsyncPipelineSettled(kind, id, token, result, install),
      (err) => this._onAsyncPipelineSettled(kind, id, token, { id, status: 'failed', error: err }, install)
    );
    return true;
  }

  _onAsyncPipelineSettled(kind, id, token, result, install) {
    const pending = kind === 'compute' ? this.pendingAsyncComputePipelines : this.pendingAsyncRenderPipelines;
    if (pending.get(id) !== token) return; // destroyed or re-created meanwhile
    pending.delete(id);
    const status = result && result.status;
    if (status === 'stale' || status === 'deleted') return; // device reset / resource destroyed
    if (status === 'ready' && result.pipeline) {
      try {
        install(result.pipeline, Boolean(result.retried));
      } catch (err) {
        this._recordAsyncPipelineFailure(kind, id, err);
      }
      return;
    }
    if (
      this.processor &&
      this.processor._pendingRangedPipelines &&
      this.processor._pendingRangedPipelines.has(id)
    ) {
      // Temporary fallback compile failed while real ranged shaders are still streaming in;
      // wait cleanly for the ranged shader settlement to materialize the real pipeline.
      return;
    }
    this._recordAsyncPipelineFailure(kind, id, result && result.error);
  }

  _recordAsyncPipelineFailure(kind, id, err) {
    const message = `${kind} pipeline ${id}: ${String(err && err.message ? err.message : err || 'async compile failed')}`;
    this.telemetry.asyncPipelineFailures += 1;
    this.stats.errors += 1;
    this.lastError = message;
    this._recordTelemetryWarningOnce(`async-pipeline-failed:${kind}:${id}`, message);
  }

  _compileRenderPipelineAsync(asyncCache, id, descriptor, info) {
    // Explicit root-signature layout first; if Dawn rejects it asynchronously, retry with
    // layout:'auto' and remap group 0 at flush time like the synchronous mismatch path does.
    const fallbackDescriptor =
      !info.autoLayoutRemap && descriptor.layout !== 'auto'
        ? { ...descriptor, layout: 'auto' }
        : null;
    return this._compilePipelineAsync(asyncCache, 'render', id, descriptor, fallbackDescriptor, (pipeline, retried) => {
      const usedFallback = retried && fallbackDescriptor !== null;
      if (usedFallback) {
        this.telemetry.autoLayoutFallbacks += 1;
        this._recordTelemetryWarningOnce(
          `async-auto-layout:render:${id}:${info.layoutId}`,
          `render pipeline ${id}: explicit layout ${info.layoutId} rejected by async compile; using layout:'auto'`
        );
      }
      this._installRenderPipeline(id, pipeline, usedFallback ? fallbackDescriptor : descriptor, {
        ...info,
        autoLayoutRemap: info.autoLayoutRemap || usedFallback,
      });
    });
  }

  _installRenderPipeline(id, pipeline, descriptor, info) {
    const { vsId, fsId, layoutId } = info;
    this.processor.pipelines.set(id, pipeline);
    this._invalidateRemappedBindGroupsForPipeline(id, false);
    this._invalidatePresentXCacheForResource(id);
    if (info.autoLayoutRemap) {
      this.renderPipelineAutoLayoutMeta.set(id, {
        id,
        layoutId,
        autoLayoutRemap: true,
        wgslBindings: info.wgslBindings,
      });
    } else {
      this.renderPipelineAutoLayoutMeta.delete(id);
    }
    if (this.processor._pipelineMeta && typeof this.processor._pipelineMeta.set === 'function') {
      const vs = this.processor.shaderModules.get(vsId) || null;
      const fs = fsId !== 0 ? this.processor.shaderModules.get(fsId) || null : null;
      const isFallback = Boolean(
        (vs && vs.source === 'builtin') || (fs && fs.source === 'builtin')
      );
      const usesFallbackShader = Boolean(
        (vs && (vs.source === 'builtin' || vs.source === 'fallback')) ||
          (fs && (fs.source === 'builtin' || fs.source === 'fallback'))
      );
      const isRealPack = Boolean(
        vs && vs.source === 'pack' && (fsId === 0 || (fs && fs.source === 'pack'))
      );
      let expectsGroup0 = Boolean(
        wgslDeclaresGroup0(getShaderSourceText(vs)) || wgslDeclaresGroup0(getShaderSourceText(fs))
      );
      if (!expectsGroup0 && layoutId !== 0) {
        const layoutMeta = this.pipelineLayoutMeta.get(layoutId);
        if (layoutMeta && layoutMeta.group0HasEntries) {
          expectsGroup0 = true;
        }
      }
      const effectiveFallbackProfile =
        vs && vs.fallbackProfile && vs.fallbackProfile !== 'default'
          ? vs.fallbackProfile
          : fs && fs.fallbackProfile && fs.fallbackProfile !== 'default'
            ? fs.fallbackProfile
            : isFallback
              ? 'default'
              : null;
      const firstBuffer =
        descriptor.vertex && Array.isArray(descriptor.vertex.buffers) && descriptor.vertex.buffers.length > 0
          ? descriptor.vertex.buffers[0]
          : null;
      const vertexStride =
        firstBuffer && typeof firstBuffer.arrayStride === 'number' ? firstBuffer.arrayStride : 0;
      const topology =
        (descriptor.primitive && descriptor.primitive.topology) || 'triangle-list';
      this.processor._pipelineMeta.set(id, {
        id,
        vsShaderId: vsId,
        fsShaderId: fsId,
        layoutId,
        vertexStride,
        topology,
        isV2: true,
        hasReflectedLayout: layoutId !== 0,
        isRealPack,
        isFallback,
        usesFallbackShader,
        expectsGroup0,
        fallbackProfile: effectiveFallbackProfile,
      });
    }
    if (typeof this.processor._invalidateSpecializedPipelinesFor === 'function') {
      this.processor._invalidateSpecializedPipelinesFor(id);
    }
  }

  _inspectComputeShaderEntryPoint(cs, csId) {
    const sourceText = getShaderSourceText(cs);
    const declared = sourceText ? findWgslStageEntryPoints(sourceText, 'compute') : [];
    const hasComputeAttr = declared.length > 0;
    if (sourceText && !hasComputeAttr) {
      this._recordTelemetryWarningOnce(
        `missing-compute-entry:${csId}`,
        `compute shader module ${csId} lacks @compute / cs_main declaration; defaulting to ${DEFAULT_ENTRY_POINTS.compute}`
      );
      return { entryPoint: DEFAULT_ENTRY_POINTS.compute, hasComputeAttr: false };
    }
    const csEntryPoint = this._resolveEntryPoint(cs, 'compute');
    return { entryPoint: csEntryPoint, hasComputeAttr };
  }

  _createComputePipeline(view, offset) {
    const id = view.getUint32(offset, true);
    const csId = view.getUint32(offset + 4, true);
    const layoutId = view.getUint32(offset + 8, true);
    let layout = this._layoutFor(layoutId);
    const cs = this._ensureShaderModule(csId);
    if (!cs || !layout) {
      if (!cs) this._packetError = `unknown compute shader module ${csId}`;
      return false;
    }
    const { entryPoint: csEntryPoint, hasComputeAttr } = this._inspectComputeShaderEntryPoint(cs, csId);
    const csSource = getShaderSourceText(cs);
    const wgslBindings = csSource ? parseWgslGroup0Bindings(csSource) : [];
    let autoLayoutRemap = false;
    const layoutMeta = layoutId !== 0 ? this.pipelineLayoutMeta.get(layoutId) : null;
    if (
      layoutId !== 0 &&
      layoutMeta &&
      !isExplicitLayoutCompatibleWithWgsl(layoutMeta.group0Entries, wgslBindings)
    ) {
      layout = 'auto';
      autoLayoutRemap = true;
      this._recordTelemetryWarningOnce(
        `auto-layout-remap:compute:${id}:${layoutId}`,
        `compute pipeline ${id}: explicit layout ${layoutId} incompatible with WGSL @group(0) bindings; falling back to layout:'auto'`
      );
    }
    const descriptor = { layout, compute: { module: cs.module, entryPoint: csEntryPoint } };
    const info = {
      csId,
      layoutId,
      entryPoint: csEntryPoint,
      hasComputeAttr,
      autoLayoutRemap,
      wgslBindings,
      isSingleGroup0Layout: Boolean(layoutMeta && layoutMeta.isSingleGroup0),
    };
    const asyncCache = this._asyncPipelineCacheFor('compute');
    if (asyncCache) {
      const fallbackDescriptor =
        !autoLayoutRemap && layout !== 'auto'
          ? { ...descriptor, layout: 'auto' }
          : null;
      return this._compilePipelineAsync(asyncCache, 'compute', id, descriptor, fallbackDescriptor, (pipeline, retried) => {
        const usedFallback = retried && fallbackDescriptor !== null;
        if (usedFallback) {
          this._recordTelemetryWarningOnce(
            `async-auto-layout:compute:${id}:${layoutId}`,
            `compute pipeline ${id}: explicit layout ${layoutId} rejected by async compile; using layout:'auto'`
          );
        }
        this._installComputePipeline(id, pipeline, { ...info, autoLayoutRemap: autoLayoutRemap || usedFallback });
      });
    }
    let pipeline;
    try {
      pipeline = this.device.createComputePipeline(descriptor);
    } catch (err) {
      this._packetError = `compute pipeline ${id}: ${String(err && err.message ? err.message : err)}`;
      return false;
    }
    this._installComputePipeline(id, pipeline, info);
    return true;
  }

  _installComputePipeline(id, pipeline, info) {
    if (info.layoutId === 0 || info.autoLayoutRemap) {
      this.telemetry.autoLayoutFallbacks += 1;
    } else if (info.isSingleGroup0Layout) {
      this.telemetry.rootSigGroup0Layouts += 1;
    }
    this.processor.computePipelines.set(id, pipeline);
    this._invalidateRemappedBindGroupsForPipeline(id, true);
    this.computePipelineMeta.set(id, {
      id,
      csId: info.csId,
      layoutId: info.layoutId,
      entryPoint: info.entryPoint,
      hasComputeAttr: info.hasComputeAttr,
      autoLayoutRemap: info.autoLayoutRemap,
      wgslBindings: info.wgslBindings,
    });
    this.telemetry.computePipelineCount += 1;
  }

  _handleV1CreateComputePipeline(origHandler, view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const csId = view.getUint32(payloadOffset + 4, true);
    const cs = this._ensureShaderModule(csId);
    let inspected = null;
    if (cs) {
      inspected = this._inspectComputeShaderEntryPoint(cs, csId);
    }
    if (cs && this._handleV1CreateComputePipelineAsync(id, csId, cs, inspected, view, payloadOffset)) {
      return;
    }
    const prevPipeline =
      this.processor && this.processor.computePipelines ? this.processor.computePipelines.get(id) : undefined;
    try {
      origHandler.call(this.processor, view, payloadOffset);
    } catch (err) {
      this._packetError = `compute pipeline ${id}: ${String(err && err.message ? err.message : err)}`;
      this._recordTelemetryWarningOnce(
        `v1-compute-create-fail:${id}`,
        `compute pipeline ${id} creation failed`
      );
      return;
    }
    const nextPipeline =
      this.processor && this.processor.computePipelines ? this.processor.computePipelines.get(id) : undefined;
    if (nextPipeline && nextPipeline !== prevPipeline) {
      this._invalidateRemappedBindGroupsForPipeline(id, true);
      this.computePipelineMeta.set(id, {
        id,
        csId,
        layoutId: 0,
        entryPoint: inspected ? inspected.entryPoint : DEFAULT_ENTRY_POINTS.compute,
        hasComputeAttr: Boolean(inspected && inspected.hasComputeAttr),
        autoLayoutRemap: false,
        wgslBindings: [],
      });
      this.telemetry.computePipelineCount += 1;
    }
  }

  /**
   * W103: compile a v1 CreateComputePipeline through the async cache. Only pipelines with a
   * reflected explicit layout go async (v1 CreateBindGroup resolves their layouts while pending);
   * 'auto'-layout ones stay synchronous. Returns true when the packet was handled here.
   */
  _handleV1CreateComputePipelineAsync(id, csId, cs, inspected, view, payloadOffset) {
    const proc = this.processor;
    const asyncCache = this._asyncPipelineCacheFor('compute');
    if (
      !asyncCache ||
      typeof proc._buildComputePipelineDescriptorV1 !== 'function' ||
      typeof proc._registerComputePipelineV1 !== 'function'
    ) {
      return false;
    }
    let built;
    try {
      built = proc._buildComputePipelineDescriptorV1(view, payloadOffset);
    } catch (err) {
      this._packetError = `compute pipeline ${id}: ${String(err && err.message ? err.message : err)}`;
      this._recordTelemetryWarningOnce(`v1-compute-create-fail:${id}`, `compute pipeline ${id} creation failed`);
      return true;
    }
    if (!built || !built.reflected) return false;
    const csSource = getShaderSourceText(cs);
    const wgslBindings = csSource ? parseWgslGroup0Bindings(csSource) : [];
    const descriptor = built.descriptor;
    const fallbackDescriptor = wgslBindings.length > 0 ? { ...descriptor, layout: 'auto' } : null;
    this._compilePipelineAsync(asyncCache, 'compute', id, descriptor, fallbackDescriptor, (pipeline, retried) => {
      const usedFallback = retried && fallbackDescriptor !== null;
      if (usedFallback) {
        // The reflected layouts no longer describe this pipeline; use its 'auto' layouts.
        if (proc._computePipelineBindGroupLayouts) proc._computePipelineBindGroupLayouts.delete(id);
        this.telemetry.autoLayoutFallbacks += 1;
        this._recordTelemetryWarningOnce(
          `async-auto-layout:compute:${id}:v1`,
          `compute pipeline ${id}: reflected layout rejected by async compile; using layout:'auto'`
        );
      }
      proc._registerComputePipelineV1(id, pipeline);
      this._invalidateRemappedBindGroupsForPipeline(id, true);
      this.computePipelineMeta.set(id, {
        id,
        csId,
        layoutId: 0,
        entryPoint: inspected ? inspected.entryPoint : DEFAULT_ENTRY_POINTS.compute,
        hasComputeAttr: Boolean(inspected && inspected.hasComputeAttr),
        autoLayoutRemap: usedFallback,
        wgslBindings: usedFallback ? wgslBindings : [],
      });
      this.telemetry.computePipelineCount += 1;
    });
    return true;
  }

  _resolveBindGroupDescriptor(bindGroupId) {
    if (!bindGroupId) return null;
    const desc = this.bindGroupDescriptors.get(bindGroupId);
    if (desc) return desc;
    const v1Meta =
      this.processor &&
      this.processor._bindGroupMeta &&
      typeof this.processor._bindGroupMeta.get === 'function'
        ? this.processor._bindGroupMeta.get(bindGroupId)
        : null;
    if (!v1Meta || !Array.isArray(v1Meta.entries)) return null;
    const resources = [];
    for (let i = 0; i < v1Meta.entries.length; i += 1) {
      const e = v1Meta.entries[i];
      if (!e || !e.resource) continue;
      if (e.resource.buffer) {
        const buf = e.resource.buffer;
        let bufferUsage = null;
        if (typeof buf.usage === 'number') {
          bufferUsage = buf.usage;
        } else if (buf.desc && typeof buf.desc.usage === 'number') {
          bufferUsage = buf.desc.usage;
        }
        const kind =
          bufferUsage !== null &&
          (bufferUsage & GPU_BUFFER_USAGE_STORAGE) !== 0 &&
          (bufferUsage & GPU_BUFFER_USAGE_UNIFORM) === 0
            ? 'storage'
            : 'uniform';
        resources.push({ binding: e.binding, kind, resource: e.resource, bufferUsage, isDummy: false });
      } else if (
        e.resource.kind === 'GPUSampler' ||
        (e.resource.desc && ('magFilter' in e.resource.desc || 'compare' in e.resource.desc))
      ) {
        resources.push({
          binding: e.binding,
          kind: 'sampler',
          resource: e.resource,
          comparison: Boolean(e.resource.desc && e.resource.desc.compare),
          isDummy: false,
        });
      } else {
        resources.push({ binding: e.binding, kind: 'texture', resource: e.resource, isDummy: false });
      }
    }
    return { id: bindGroupId, layoutId: v1Meta.layoutId || 0, resources };
  }

  // `dynamicOffsets` are the setBindGroup(0, ..., dynamicOffsets) values (root-constant ring
  // offsets). layout:'auto' bind group layouts never have hasDynamicOffset, so each offset is folded
  // into the static offset of the buffer entry whose dynamicOffsetIndex it belongs to.
  _getOrCreateRemappedBindGroup(
    isCompute,
    pipelineId,
    pipeline,
    wgslBindings,
    bindGroupId,
    bindGroupDesc,
    dynamicOffsets,
    out = null
  ) {
    if (!pipeline || typeof pipeline.getBindGroupLayout !== 'function') {
      if (out) {
        out.status = 'unsupported';
        out.bindGroup = null;
        return out;
      }
      return { status: 'unsupported', bindGroup: null };
    }
    if (!Array.isArray(wgslBindings) || wgslBindings.length === 0) {
      if (out) {
        out.status = 'empty';
        out.bindGroup = null;
        return out;
      }
      return { status: 'empty', bindGroup: null };
    }

    const available =
      bindGroupDesc && Array.isArray(bindGroupDesc.resources) ? bindGroupDesc.resources : EMPTY_DYNAMIC_OFFSETS;

    // W186/W373: fast path for 2-binding PresentX plane blits (texture_2d<f32> + sampler) on
    // layout:'auto' render pipelines without allocating .find() closures.
    if (!isCompute && wgslBindings.length === 2) {
      let texBinding = null;
      let smpBinding = null;
      for (let i = 0; i < 2; i += 1) {
        const b = wgslBindings[i];
        if (!b) continue;
        if (
          !texBinding &&
          b.kind === 'texture' &&
          !b.multisampled &&
          (b.viewDimension || '2d') === '2d' &&
          (!b.sampleType || b.sampleType === 'float')
        ) {
          texBinding = b;
        } else if (!smpBinding && b.kind === 'sampler') {
          smpBinding = b;
        }
      }
      if (texBinding && smpBinding) {
        let texRes = null;
        let smpRes = null;
        for (let i = 0; i < available.length; i += 1) {
          const r = available[i];
          if (!r || r.isDummy) continue;
          if (!texRes && r.kind === 'texture' && Number.isInteger(r.planeTextureId) && r.planeTextureId > 0) {
            texRes = r;
          } else if (!smpRes && r.kind === 'sampler' && r.resource) {
            smpRes = r;
          }
        }
        if (texRes) {
          const cachedPx = this.getOrCreatePresentXPlaneBindGroup({
            planeTextureId: texRes.planeTextureId,
            viewDesc: texRes.viewDesc || null,
            viewFormat: texRes.textureFormat || null,
            sampler: smpRes ? smpRes.resource : null,
            samplerId: smpRes && Number.isInteger(smpRes.samplerId) ? smpRes.samplerId : 0,
            samplerKind: smpBinding.comparison ? 'comparison' : 'filtering',
            pipeline,
            pipelineId,
            textureBinding: texBinding.binding,
            samplerBinding: smpBinding.binding,
          });
          if (cachedPx && cachedPx.bindGroup) {
            if (out) {
              out.status = 'ok';
              out.bindGroup = cachedPx.bindGroup;
              return out;
            }
            return { status: 'ok', bindGroup: cachedPx.bindGroup };
          }
        }
      }
    }

    // W373: Zero-alloc cache-hit fast path: compute dynKey without normalizing into a fresh array,
    // without allocating `parts`, and without defining per-call closures before checking the cache.
    const hasRawOffsets =
      dynamicOffsets !== null &&
      dynamicOffsets !== undefined &&
      (Array.isArray(dynamicOffsets) || ArrayBuffer.isView(dynamicOffsets)) &&
      dynamicOffsets.length > 0;
    let dynKey = '';
    let hasAnyNonZeroDyn = false;
    if (hasRawOffsets) {
      const dynLen = dynamicOffsets.length;
      for (let i = 0; i < dynLen; i += 1) {
        const v = Number(dynamicOffsets[i]);
        if (Number.isFinite(v) && v > 0) {
          hasAnyNonZeroDyn = true;
          break;
        }
      }
      if (hasAnyNonZeroDyn) {
        // Only offsets that land on a real dynamic-offset buffer entry split the cache; all-zero
        // offsets share the plain '<c|r>:<pipeId>:<bgId>' key.
        let anyMatchedNonZero = false;
        let keyStr = '';
        for (let i = 0; i < available.length; i += 1) {
          const r = available[i];
          if (!r || !r.hasDynamicOffset || r.isDummy) continue;
          let v = 0;
          const dIdx = r.dynamicOffsetIndex;
          if (Number.isInteger(dIdx) && dIdx >= 0 && dIdx < dynLen) {
            const rawV = Number(dynamicOffsets[dIdx]);
            v = Number.isFinite(rawV) && rawV > 0 ? Math.floor(rawV) >>> 0 : 0;
          }
          if (v > 0) anyMatchedNonZero = true;
          keyStr = keyStr.length > 0 ? `${keyStr},${r.binding}=${v}` : `${r.binding}=${v}`;
        }
        if (anyMatchedNonZero) dynKey = keyStr;
      }
    }

    const effectiveBgId = bindGroupId || 0;
    const isLiveDesc =
      effectiveBgId === 0 ||
      (bindGroupDesc && this.bindGroupDescriptors.get(effectiveBgId) === bindGroupDesc);
    const pipelineKey = `${isCompute ? 'c' : 'r'}:${pipelineId}`;
    const cacheKey = dynKey.length > 0 ? `${pipelineKey}:${effectiveBgId}@${dynKey}` : `${pipelineKey}:${effectiveBgId}`;
    if (isLiveDesc && this.remappedBindGroups.has(cacheKey)) {
      const cached = this.remappedBindGroups.get(cacheKey);
      this._cacheRemappedBindGroup(cacheKey, cached, pipelineKey, effectiveBgId);
      if (out) {
        out.status = cached ? 'ok' : 'empty';
        out.bindGroup = cached || null;
        return out;
      }
      return cached ? { status: 'ok', bindGroup: cached } : { status: 'empty', bindGroup: null };
    }

    let autoBgl;
    try {
      autoBgl = pipeline.getBindGroupLayout(0);
    } catch (_) {
      if (isLiveDesc) this._cacheRemappedBindGroup(cacheKey, null, pipelineKey, effectiveBgId);
      if (out) {
        out.status = 'empty';
        out.bindGroup = null;
        return out;
      }
      return { status: 'empty', bindGroup: null };
    }
    if (!autoBgl) {
      if (isLiveDesc) this._cacheRemappedBindGroup(cacheKey, null, pipelineKey, effectiveBgId);
      if (out) {
        out.status = 'empty';
        out.bindGroup = null;
        return out;
      }
      return { status: 'empty', bindGroup: null };
    }

    const dynOffsets = hasAnyNonZeroDyn ? normalizeDynamicOffsets(dynamicOffsets) : EMPTY_DYNAMIC_OFFSETS;
    const usedIndices = new Set();

    // Folds r's dynamic offset into a static { buffer, offset, size } binding, clamped so that
    // offset + size never exceeds the GPUBuffer size; an out-of-bounds offset binds a dummy.
    const bufferResourceWithDynamicOffset = (r, b) => {
      const base = r.resource;
      if (!base || !base.buffer || base.buffer.destroyed === true) {
        return this._dummyResource(synthLayoutEntryForWgslBinding(b));
      }
      const dIdx = r && r.hasDynamicOffset ? r.dynamicOffsetIndex : -1;
      const dyn =
        Number.isInteger(dIdx) && dIdx >= 0 && dIdx < dynOffsets.length ? dynOffsets[dIdx] : 0;
      const isUniform = b.kind === 'uniform';
      const maxBindingSize = isUniform
        ? getDeviceLimit(this.device, 'maxUniformBufferBindingSize', DEFAULT_MAX_UNIFORM_BUFFER_BINDING_SIZE)
        : getDeviceLimit(this.device, 'maxStorageBufferBindingSize', DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE);
      const effectiveOffset = (Number(base.offset) || 0) + (dyn > 0 ? dyn : 0);
      const explicitAlignKey = isUniform ? 'minUniformBufferOffsetAlignment' : 'minStorageBufferOffsetAlignment';
      const explicitAlign =
        this.device &&
        this.device.limits &&
        typeof this.device.limits[explicitAlignKey] === 'number' &&
        this.device.limits[explicitAlignKey] > 1
          ? Math.floor(this.device.limits[explicitAlignKey])
          : 0;
      if (effectiveOffset > 0 && explicitAlign > 1 && effectiveOffset % explicitAlign !== 0) {
        this.telemetry.dynamicOffsetClamps += 1;
        this._recordTelemetryWarningOnce(
          `remap-dyn-offset-unaligned:${effectiveBgId}:${r.binding}:${explicitAlign}`,
          `auto-layout remap: bind group ${effectiveBgId} binding ${r.binding} offset ${effectiveOffset} not aligned to ${explicitAlign}B; using dummy buffer`
        );
        return this._dummyResource(synthLayoutEntryForWgslBinding(b));
      }
      const bufferSize =
        typeof r.bufferSize === 'number' && Number.isFinite(r.bufferSize) && r.bufferSize > 0
          ? r.bufferSize
          : base.buffer && typeof base.buffer.size === 'number' && base.buffer.size > 0
            ? base.buffer.size
            : null;
      let size = typeof base.size === 'number' && base.size > 0 ? base.size : null;
      if (dyn <= 0) {
        const boundSize = size !== null ? size : bufferSize !== null ? bufferSize - effectiveOffset : null;
        const needsClamp = boundSize !== null && boundSize > maxBindingSize;
        const needsStorageAlign = !isUniform && boundSize !== null && (boundSize < 4 || boundSize % 4 !== 0);
        if (!needsClamp && !needsStorageAlign) return base;
      }
      if (bufferSize !== null) {
        if (effectiveOffset >= bufferSize) {
          this.telemetry.dynamicOffsetClamps += 1;
          this._recordTelemetryWarningOnce(
            `remap-dyn-offset-oob:${effectiveBgId}:${r.binding}`,
            `auto-layout remap: bind group ${effectiveBgId} binding ${r.binding} dynamic offset ${dyn} exceeds buffer size ${bufferSize}; using dummy buffer`
          );
          return this._dummyResource(synthLayoutEntryForWgslBinding(b));
        }
        const remaining = bufferSize - effectiveOffset;
        if (size !== null && size > remaining) {
          size = remaining;
          this.telemetry.dynamicOffsetClamps += 1;
          this._recordTelemetryWarningOnce(
            `remap-dyn-offset-clamp:${effectiveBgId}:${r.binding}`,
            `auto-layout remap: bind group ${effectiveBgId} binding ${r.binding} size clamped to ${remaining} at dynamic offset ${dyn}`
          );
        }
        const effBound = size !== null ? size : remaining;
        if (effBound > maxBindingSize) {
          size = isUniform ? maxBindingSize : Math.floor(maxBindingSize / 4) * 4;
        }
        if (!isUniform) {
          const storageBound = size !== null ? size : remaining;
          if (storageBound < 4) {
            return this._dummyResource(synthLayoutEntryForWgslBinding(b));
          }
          if (storageBound % 4 !== 0) {
            size = Math.floor(storageBound / 4) * 4;
          }
        }
      } else if (size !== null && size > maxBindingSize) {
        size = isUniform ? maxBindingSize : Math.floor(maxBindingSize / 4) * 4;
      }
      if (dyn > 0) this.telemetry.dynamicOffsetRemaps += 1;
      const folded = { buffer: base.buffer, offset: effectiveOffset };
      if (size !== null) folded.size = size;
      return folded;
    };

    const pickResourceForBinding = (b) => {
      for (let pass = 0; pass < 2; pass += 1) {
        for (let i = 0; i < available.length; i += 1) {
          if (usedIndices.has(i)) continue;
          const r = available[i];
          if (!r || !r.resource || r.isDummy || r.resource.destroyed === true) continue;
          if (b.kind === 'uniform') {
            if (!isBufferBindingResource(r.resource) || r.resource.buffer.destroyed === true) continue;
            const isBuf =
              r.kind === 'uniform' || r.kind === 'storage' || r.kind === 'read-only-storage';
            if (!isBuf) continue;
            if (pass === 0 && r.kind !== 'uniform') continue;
            if (r.bufferUsage !== null && r.bufferUsage !== undefined) {
              if ((r.bufferUsage & GPU_BUFFER_USAGE_UNIFORM) === 0) continue;
            } else if (r.kind !== 'uniform') {
              continue;
            }
            usedIndices.add(i);
            return bufferResourceWithDynamicOffset(r, b);
          }
          if (b.kind === 'read-only-storage' || b.kind === 'storage') {
            if (!isBufferBindingResource(r.resource) || r.resource.buffer.destroyed === true) continue;
            const isBuf =
              r.kind === 'uniform' || r.kind === 'storage' || r.kind === 'read-only-storage';
            if (!isBuf) continue;
            if (
              pass === 0 &&
              !(r.kind === b.kind || (b.kind === 'read-only-storage' && r.kind === 'storage'))
            ) {
              continue;
            }
            if (r.bufferUsage !== null && r.bufferUsage !== undefined) {
              if ((r.bufferUsage & GPU_BUFFER_USAGE_STORAGE) === 0) continue;
            } else if (
              !(r.kind === 'storage' || (b.kind === 'read-only-storage' && r.kind === 'read-only-storage'))
            ) {
              continue;
            }
            usedIndices.add(i);
            return bufferResourceWithDynamicOffset(r, b);
          }
          if (pass > 0) continue;
          if (b.kind === 'texture') {
            if (r.kind !== 'texture') continue;
            if (!isNonBufferBindingResource(r.resource) || isSamplerResource(r.resource)) continue;
            if (
              r.textureUsage !== null &&
              r.textureUsage !== undefined &&
              (r.textureUsage & GPU_TEXTURE_USAGE_TEXTURE_BINDING) === 0
            ) {
              continue;
            }
            const expectedDim = b.viewDimension || '2d';
            if (r.viewDimension && r.viewDimension !== expectedDim) continue;
            if (
              r.multisampled !== undefined &&
              r.multisampled !== null &&
              Boolean(r.multisampled) !== Boolean(b.multisampled)
            ) {
              continue;
            }
            if (r.textureFormat) {
              const expectedSampleType = b.sampleType || 'float';
              if (!isTextureSampleTypeCompatible(r.textureFormat, expectedSampleType, this.device)) {
                continue;
              }
            }
            usedIndices.add(i);
            return r.resource;
          }
          if (b.kind === 'storage-texture') {
            if (r.kind !== 'storage-texture') continue;
            if (!isNonBufferBindingResource(r.resource) || isSamplerResource(r.resource)) continue;
            if (r.multisampled === true) continue;
            if (
              r.textureUsage !== null &&
              r.textureUsage !== undefined &&
              (r.textureUsage & GPU_TEXTURE_USAGE_STORAGE_BINDING) === 0
            ) {
              continue;
            }
            const expectedDim = b.viewDimension || '2d';
            if (r.viewDimension && r.viewDimension !== expectedDim) continue;
            if (r.textureFormat && b.storageFormat && r.textureFormat !== b.storageFormat) {
              continue;
            }
            usedIndices.add(i);
            return r.resource;
          }
          if (b.kind === 'sampler') {
            if (r.kind !== 'sampler') continue;
            if (!isNonBufferBindingResource(r.resource) || isTextureViewResource(r.resource)) {
              continue;
            }
            if (
              r.comparison !== undefined &&
              r.comparison !== null &&
              Boolean(r.comparison) !== Boolean(b.comparison)
            ) {
              continue;
            }
            usedIndices.add(i);
            return r.resource;
          }
        }
      }
      if (b.kind === 'sampler') {
        const samplerKind = b.comparison ? 'comparison' : 'filtering';
        const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
        if (fallbackSampler) {
          this.telemetry.fallbackSamplersBound += 1;
          return fallbackSampler;
        }
      }
      return this._dummyResource(synthLayoutEntryForWgslBinding(b));
    };

    const entries = wgslBindings.map((b) => ({
      binding: b.binding,
      resource: pickResourceForBinding(b),
    }));

    let remapped = null;
    try {
      remapped = this.device.createBindGroup({ layout: autoBgl, entries });
    } catch (_) {
      try {
        const fallbackEntries = wgslBindings.map((b) => {
          if (b.kind === 'sampler') {
            const samplerKind = b.comparison ? 'comparison' : 'filtering';
            const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
            if (fallbackSampler) {
              this.telemetry.fallbackSamplersBound += 1;
              return { binding: b.binding, resource: fallbackSampler };
            }
          }
          return {
            binding: b.binding,
            resource: this._dummyResource(synthLayoutEntryForWgslBinding(b)),
          };
        });
        remapped = this.device.createBindGroup({ layout: autoBgl, entries: fallbackEntries });
      } catch (_err2) {
        remapped = null;
      }
    }

    if (isLiveDesc) {
      this._cacheRemappedBindGroup(cacheKey, remapped, pipelineKey, effectiveBgId);
    }
    if (out) {
      out.status = remapped ? 'ok' : 'empty';
      out.bindGroup = remapped || null;
      return out;
    }
    return remapped ? { status: 'ok', bindGroup: remapped } : { status: 'empty', bindGroup: null };
  }

  // Guards queued compute dispatches against zero/oversized workgroup dimensions and missing
  // compute pipelines, and remaps @group(0) bind groups for pipelines that fell back to layout:'auto',
  // immediately before WgpuCommandProcessor._flushPresent() executes them.
  _preparePassCommandsForFlush() {
    this._syncDeviceGeneration();
    this._flushHasV1BindGroup = false;
    this._resetPassState(null);
    const cmds = this.processor && this.processor._passCommands;
    if (!Array.isArray(cmds) || cmds.length === 0) return;

    let activePassMode = null; // 'render' | 'compute' | null
    let activeComputePipelineId = null;
    let activeComputePipelineValid = false;
    let activeComputeAutoLayout = null;
    let activeComputeHasKnownLayout = false;
    let activeComputeRequiredGroupsMask = 0;
    let boundComputeGroupsMask = 0;
    let invalidComputeGroupsMask = 0;
    const boundComputeGroupBglIds = new Int32Array(8);
    boundComputeGroupBglIds.fill(-1);
    let hadExplicitSetComputePipeline = false;
    let group0BoundForCompute = false;
    let hasLastComputeGroup0 = false;
    const lastComputeGroup0 = this._scratchLastComputeGroup0;

    let activeRenderPipelineId = null;
    let activeRenderAutoLayout = null;
    let activeRenderHasKnownLayout = false;
    let activeRenderRequiredGroupsMask = 0;
    let boundRenderGroupsMask = 0;
    let invalidRenderGroupsMask = 0;
    let hasIndexBufferInPass = false;
    let group0BoundForRender = false;
    let hasLastRenderGroup0 = false;
    const lastRenderGroup0 = this._scratchLastRenderGroup0;
    // W103: the bound pipeline is still compiling asynchronously; skip its draws/dispatches.
    let activeRenderPending = false;
    let activeComputePending = false;

    // W354: lazily allocate `filtered` ONLY if the command stream actually needs modification
    // (zero array allocation on steady-state frames!).
    let filtered = null;
    const ensureFiltered = (copyUpTo) => {
      if (filtered === null) {
        filtered = new Array(copyUpTo);
        for (let k = 0; k < copyUpTo; k += 1) {
          filtered[k] = cmds[k];
        }
      }
      return filtered;
    };

    for (let i = 0; i < cmds.length; i += 1) {
      const cmd = cmds[i];
      if (!cmd) {
        ensureFiltered(i);
        continue;
      }

      if (cmd._rsInvalidBindGroup) {
        const gIdx = cmd._rsInvalidBindGroup.groupIndex;
        const bit = gIdx >= 0 && gIdx < 31 ? 1 << gIdx : 0;
        if (bit !== 0) {
          if (activePassMode === 'compute') {
            boundComputeGroupsMask &= ~bit;
            invalidComputeGroupsMask |= bit;
            if (gIdx === 0) {
              group0BoundForCompute = false;
              hasLastComputeGroup0 = false;
            }
          } else if (activePassMode === 'render') {
            boundRenderGroupsMask &= ~bit;
            invalidRenderGroupsMask |= bit;
            if (gIdx === 0) {
              group0BoundForRender = false;
              hasLastRenderGroup0 = false;
            }
          } else {
            boundComputeGroupsMask &= ~bit;
            invalidComputeGroupsMask |= bit;
            boundRenderGroupsMask &= ~bit;
            invalidRenderGroupsMask |= bit;
            if (gIdx === 0) {
              group0BoundForCompute = false;
              hasLastComputeGroup0 = false;
              group0BoundForRender = false;
              hasLastRenderGroup0 = false;
            }
          }
        }
        ensureFiltered(i);
        continue;
      }

      const passOpKind =
        (cmd._wgpuMeta && cmd._wgpuMeta.kind) || cmd._rsPassOp || null;
      if (passOpKind === 'beginRenderPass' || passOpKind === 'endRenderPass') {
        activePassMode = passOpKind === 'beginRenderPass' ? 'render' : null;
        activeComputePipelineId = null;
        activeComputePipelineValid = false;
        activeComputeAutoLayout = null;
        activeComputeHasKnownLayout = false;
        activeComputeRequiredGroupsMask = 0;
        boundComputeGroupsMask = 0;
        invalidComputeGroupsMask = 0;
        boundComputeGroupBglIds.fill(-1);
        hadExplicitSetComputePipeline = false;
        group0BoundForCompute = false;
        hasLastComputeGroup0 = false;
        activeRenderPipelineId = null;
        activeRenderAutoLayout = null;
        activeRenderHasKnownLayout = false;
        activeRenderRequiredGroupsMask = 0;
        boundRenderGroupsMask = 0;
        invalidRenderGroupsMask = 0;
        hasIndexBufferInPass = false;
        group0BoundForRender = false;
        hasLastRenderGroup0 = false;
        activeRenderPending = false;
        activeComputePending = false;
        if (filtered !== null) filtered.push(cmd);
        continue;
      }

      if (cmd.type === 'setIndexBuffer') {
        const idxBuf = this.processor.buffers ? this.processor.buffers.get(cmd.bufferId) : null;
        const idxMeta =
          this.processor._bufferMeta && typeof this.processor._bufferMeta.get === 'function'
            ? this.processor._bufferMeta.get(cmd.bufferId)
            : null;
        const idxUsage =
          idxBuf && typeof idxBuf.usage === 'number'
            ? idxBuf.usage
            : idxMeta && typeof idxMeta.usage === 'number'
              ? idxMeta.usage
              : idxBuf && idxBuf.desc && typeof idxBuf.desc.usage === 'number'
                ? idxBuf.desc.usage
                : null;
        hasIndexBufferInPass = Boolean(
          idxBuf &&
            idxBuf.destroyed !== true &&
            (idxUsage === null || (idxUsage & GPU_BUFFER_USAGE_INDEX) !== 0)
        );
        if (filtered !== null) filtered.push(cmd);
        continue;
      }

      if (cmd.type === 'setPipeline') {
        if (activePassMode === 'compute') {
          boundComputeGroupsMask = 0;
          invalidComputeGroupsMask = 0;
          boundComputeGroupBglIds.fill(-1);
        }
        activePassMode = 'render';
        activeComputePipelineId = null;
        activeComputePipelineValid = false;
        activeComputeAutoLayout = null;
        activeComputeHasKnownLayout = false;
        activeComputeRequiredGroupsMask = 0;
        hadExplicitSetComputePipeline = false;
        group0BoundForCompute = false;
        hasLastComputeGroup0 = false;
        activeComputePending = false;
        activeRenderPipelineId = cmd.pipelineId;
        activeRenderPending = this.isAsyncPipelinePending(cmd.pipelineId, false);
        if (activeRenderPending) {
          // Never bind an undefined pipeline; its draws are dropped until the compile resolves.
          activeRenderAutoLayout = null;
          activeRenderHasKnownLayout = false;
          activeRenderRequiredGroupsMask = 0;
          group0BoundForRender = false;
          ensureFiltered(i);
          continue;
        }
        const rMeta = this.renderPipelineAutoLayoutMeta.get(cmd.pipelineId) || null;
        activeRenderAutoLayout = rMeta && rMeta.autoLayoutRemap ? rMeta : null;
        if (activeRenderAutoLayout) {
          activeRenderHasKnownLayout = true;
          activeRenderRequiredGroupsMask = 0;
        } else {
          const rPipeMeta =
            this.processor && this.processor._pipelineMeta
              ? this.processor._pipelineMeta.get(cmd.pipelineId) || null
              : null;
          const rLayoutMeta =
            rPipeMeta && rPipeMeta.layoutId ? this.pipelineLayoutMeta.get(rPipeMeta.layoutId) || null : null;
          activeRenderHasKnownLayout = Boolean(rLayoutMeta);
          activeRenderRequiredGroupsMask = rLayoutMeta ? rLayoutMeta.requiredGroupsMask || 0 : 0;
        }
        group0BoundForRender = false;
        if (filtered !== null) filtered.push(cmd);
        continue;
      }

      if (cmd.type === 'setComputePipeline') {
        if (activePassMode === 'render') {
          boundRenderGroupsMask = 0;
          invalidRenderGroupsMask = 0;
          hasIndexBufferInPass = false;
        }
        activePassMode = 'compute';
        activeRenderPipelineId = null;
        activeRenderAutoLayout = null;
        activeRenderHasKnownLayout = false;
        activeRenderRequiredGroupsMask = 0;
        group0BoundForRender = false;
        hasLastRenderGroup0 = false;
        activeRenderPending = false;
        hadExplicitSetComputePipeline = true;
        activeComputePipelineId = cmd.pipelineId;
        activeComputePending = this.isAsyncPipelinePending(cmd.pipelineId, true);
        if (activeComputePending) {
          activeComputePipelineValid = false;
          activeComputeAutoLayout = null;
          activeComputeHasKnownLayout = false;
          activeComputeRequiredGroupsMask = 0;
          group0BoundForCompute = false;
          ensureFiltered(i);
          continue;
        }
        activeComputePipelineValid = Boolean(
          this.processor.computePipelines && this.processor.computePipelines.has(cmd.pipelineId)
        );
        if (!activeComputePipelineValid) {
          activeComputeAutoLayout = null;
          activeComputeHasKnownLayout = false;
          activeComputeRequiredGroupsMask = 0;
          group0BoundForCompute = false;
          this._recordTelemetryWarningOnce(
            `missing-compute-pipeline:${cmd.pipelineId}`,
            `compute pipeline ${cmd.pipelineId} unavailable; subsequent dispatches will be skipped`
          );
          ensureFiltered(i);
          continue;
        }
        const cMeta = this.computePipelineMeta.get(cmd.pipelineId) || null;
        activeComputeAutoLayout = cMeta && cMeta.autoLayoutRemap ? cMeta : null;
        if (activeComputeAutoLayout) {
          activeComputeHasKnownLayout = true;
          activeComputeRequiredGroupsMask = 0;
        } else {
          const cLayoutMeta =
            cMeta && cMeta.layoutId ? this.pipelineLayoutMeta.get(cMeta.layoutId) || null : null;
          activeComputeHasKnownLayout = Boolean(cLayoutMeta);
          activeComputeRequiredGroupsMask = cLayoutMeta ? cLayoutMeta.requiredGroupsMask || 0 : 0;
          // #1941 ExecuteIndirect parity: switching compute pipelines (e.g. indirect_constants ->
          // indirect_clamp -> user compute) invalidates any previously bound group whose BGL differs
          // from the new pipeline's expected group BGL so stale groups never mask unbound slots.
          if (cLayoutMeta && Array.isArray(cLayoutMeta.groupBglIds)) {
            const expectedBgls = cLayoutMeta.groupBglIds;
            for (let g = 0; g < expectedBgls.length && g < 8; g += 1) {
              const expectedBgl = expectedBgls[g] || 0;
              const boundBgl = boundComputeGroupBglIds[g];
              if (expectedBgl !== 0 && boundBgl > 0 && boundBgl !== expectedBgl) {
                const gBit = 1 << g;
                boundComputeGroupsMask &= ~gBit;
                invalidComputeGroupsMask &= ~gBit;
                boundComputeGroupBglIds[g] = -1;
              }
            }
          }
        }
        group0BoundForCompute = false;
        if (filtered !== null) filtered.push(cmd);
        continue;
      }

      const isSetBgV1 = cmd.type === 'setBindGroup';
      if (isSetBgV1) {
        this._flushHasV1BindGroup = true;
      }
      const isSetBgV2 = Boolean(cmd._rsSetBindGroup || passOpKind === 'setBindGroup2');
      if (isSetBgV1 || isSetBgV2) {
        let groupIndex;
        let bindGroupId;
        let bindGroupDesc;
        let dynamicOffsets;
        if (cmd instanceof SetBindGroup2Command) {
          groupIndex = cmd.groupIndex;
          bindGroupId = cmd.bindGroupId;
          bindGroupDesc = cmd.bindGroupDesc || this._resolveBindGroupDescriptor(bindGroupId);
          // W373: SetBindGroup2Command.offsets comes directly from _setBindGroup (already normalized uint32 array or EMPTY_DYNAMIC_OFFSETS).
          dynamicOffsets = cmd.offsets;
        } else if (cmd._rsSetBindGroup) {
          groupIndex = cmd._rsSetBindGroup.groupIndex;
          bindGroupId = cmd._rsSetBindGroup.bindGroupId;
          bindGroupDesc =
            cmd._rsSetBindGroup.bindGroupDesc || this._resolveBindGroupDescriptor(bindGroupId);
          const rawOff = cmd._rsSetBindGroup.offsets || cmd._rsSetBindGroup.dynamicOffsets;
          dynamicOffsets = rawOff === EMPTY_DYNAMIC_OFFSETS ? EMPTY_DYNAMIC_OFFSETS : normalizeDynamicOffsets(rawOff);
        } else if (isSetBgV1) {
          groupIndex = cmd.groupIndex;
          bindGroupId = cmd.bindGroupId;
          bindGroupDesc = this._resolveBindGroupDescriptor(bindGroupId);
          dynamicOffsets = normalizeDynamicOffsets(cmd.dynamicOffsets || cmd.offsets);
        } else {
          groupIndex = cmd._wgpuMeta.groupIndex;
          bindGroupId = cmd._wgpuMeta.bindGroupId;
          bindGroupDesc = this._resolveBindGroupDescriptor(bindGroupId);
          dynamicOffsets = normalizeDynamicOffsets(
            cmd._wgpuMeta.dynamicOffsets || cmd._wgpuMeta.offsets
          );
        }

        const bgBit = groupIndex >= 0 && groupIndex < 31 ? 1 << groupIndex : 0;
        const bgLayoutId =
          bindGroupDesc && typeof bindGroupDesc.layoutId === 'number' ? bindGroupDesc.layoutId : 0;
        if (bgBit !== 0) {
          if (activePassMode === 'compute') {
            boundComputeGroupsMask |= bgBit;
            invalidComputeGroupsMask &= ~bgBit;
            if (groupIndex >= 0 && groupIndex < 8) {
              boundComputeGroupBglIds[groupIndex] = bgLayoutId;
            }
          } else if (activePassMode === 'render') {
            boundRenderGroupsMask |= bgBit;
            invalidRenderGroupsMask &= ~bgBit;
          } else {
            // Note: SetBindGroup2Command.run() is a no-op when ctl.pass is null, whereas v1
            // setBindGroup caches state on the processor. Only mark v1 binds outside an active pass.
            if (isSetBgV1) {
              boundComputeGroupsMask |= bgBit;
              invalidComputeGroupsMask &= ~bgBit;
              boundRenderGroupsMask |= bgBit;
              invalidRenderGroupsMask &= ~bgBit;
              if (groupIndex >= 0 && groupIndex < 8) {
                boundComputeGroupBglIds[groupIndex] = bgLayoutId;
              }
            }
          }
        }

        // #1977 parity: if a compute pipeline fell back to layout:'auto', an explicit-layout
        // RT arena bind group at group 3 is incompatible with its implicit layout; drop safely.
        if (
          activePassMode === 'compute' &&
          activeComputePipelineValid &&
          activeComputeAutoLayout &&
          groupIndex === RT_ARENA_GROUP_INDEX &&
          bindGroupDesc &&
          bindGroupDesc.hasRtArenaBindings
        ) {
          this._recordTelemetryWarningOnce(
            `skip-rt-group3-auto:${activeComputePipelineId || 0}`,
            `skipped explicit RT arena group 3 bind on layout:'auto' compute pipeline ${activeComputePipelineId || 0}`
          );
          ensureFiltered(i);
          continue;
        }

        if (groupIndex === 0) {
          if (activePassMode === 'compute') {
            lastComputeGroup0.bindGroupId = bindGroupId;
            lastComputeGroup0.bindGroupDesc = bindGroupDesc;
            lastComputeGroup0.dynamicOffsets = dynamicOffsets;
            hasLastComputeGroup0 = true;
            if (activeComputePipelineValid && activeComputeAutoLayout) {
              const pipeline = this.processor.computePipelines.get(activeComputePipelineId);
              const res = this._getOrCreateRemappedBindGroup(
                true,
                activeComputePipelineId,
                pipeline,
                activeComputeAutoLayout.wgslBindings,
                bindGroupId,
                bindGroupDesc,
                dynamicOffsets,
                this._scratchRemapResult
              );
              group0BoundForCompute = true;
              if (res.status === 'ok') {
                ensureFiltered(i).push(this._acquireRemappedSetBindGroupCommand(res.bindGroup, bindGroupId));
                continue;
              }
              if (res.status === 'empty') {
                ensureFiltered(i);
                continue;
              }
            }
          } else if (activePassMode === 'render') {
            lastRenderGroup0.bindGroupId = bindGroupId;
            lastRenderGroup0.bindGroupDesc = bindGroupDesc;
            lastRenderGroup0.dynamicOffsets = dynamicOffsets;
            hasLastRenderGroup0 = true;
            if (activeRenderAutoLayout) {
              const pipeline = this.processor.pipelines.get(activeRenderPipelineId);
              const res = this._getOrCreateRemappedBindGroup(
                false,
                activeRenderPipelineId,
                pipeline,
                activeRenderAutoLayout.wgslBindings,
                bindGroupId,
                bindGroupDesc,
                dynamicOffsets,
                this._scratchRemapResult
              );
              group0BoundForRender = true;
              if (res.status === 'ok') {
                ensureFiltered(i).push(this._acquireRemappedSetBindGroupCommand(res.bindGroup, bindGroupId));
                continue;
              }
              if (res.status === 'empty') {
                ensureFiltered(i).push(this._acquireNoopRenderGroup0Command(bindGroupId));
                continue;
              }
            }
          } else {
            lastRenderGroup0.bindGroupId = bindGroupId;
            lastRenderGroup0.bindGroupDesc = bindGroupDesc;
            lastRenderGroup0.dynamicOffsets = dynamicOffsets;
            hasLastRenderGroup0 = true;
            lastComputeGroup0.bindGroupId = bindGroupId;
            lastComputeGroup0.bindGroupDesc = bindGroupDesc;
            lastComputeGroup0.dynamicOffsets = dynamicOffsets;
            hasLastComputeGroup0 = true;
          }
        }
        if (filtered !== null) filtered.push(cmd);
        continue;
      }

      if (
        cmd.type === 'draw' ||
        cmd.type === 'drawIndexed' ||
        cmd.type === 'drawIndirect' ||
        cmd.type === 'drawIndexedIndirect'
      ) {
        if (activePassMode === 'compute') {
          boundComputeGroupsMask = 0;
          invalidComputeGroupsMask = 0;
        }
        activePassMode = 'render';
        activeComputePipelineId = null;
        activeComputePipelineValid = false;
        activeComputeAutoLayout = null;
        activeComputeRequiredGroupsMask = 0;
        hadExplicitSetComputePipeline = false;
        group0BoundForCompute = false;
        hasLastComputeGroup0 = false;
        activeComputePending = false;
        if (activeRenderPending) {
          this.telemetry.pendingAsyncPipelineSkips += 1;
          ensureFiltered(i);
          continue;
        }
        if (cmd.type === 'drawIndirect' || cmd.type === 'drawIndexedIndirect') {
          const indirectBuf = this.processor.buffers ? this.processor.buffers.get(cmd.bufferId) : null;
          const indirectMeta =
            this.processor._bufferMeta && typeof this.processor._bufferMeta.get === 'function'
              ? this.processor._bufferMeta.get(cmd.bufferId)
              : null;
          const indirectCheck = validateIndirectCommand(cmd.type, indirectBuf, cmd.offset, indirectMeta);
          if (!indirectCheck.valid) {
            this.telemetry.skippedInvalidIndirectOps += 1;
            this._recordTelemetryWarningOnce(
              `skip-indirect:${cmd.type}:${indirectCheck.reason}:${cmd.bufferId}`,
              `skipped ${cmd.type} on buffer ${cmd.bufferId} offset ${cmd.offset}: ${indirectCheck.reason}`
            );
            ensureFiltered(i);
            continue;
          }
          if (cmd.type === 'drawIndexedIndirect' && !hasIndexBufferInPass) {
            this.telemetry.skippedInvalidIndirectOps += 1;
            this._recordTelemetryWarningOnce(
              `skip-indirect:drawIndexedIndirect:no-index-buffer:${activeRenderPipelineId || 0}`,
              `skipped drawIndexedIndirect on pipeline ${activeRenderPipelineId || 0}: no valid index buffer bound in pass`
            );
            ensureFiltered(i);
            continue;
          }
        }
        if (activeRenderAutoLayout && !group0BoundForRender) {
          const pipeline = this.processor.pipelines.get(activeRenderPipelineId);
          const srcBgId = hasLastRenderGroup0 ? lastRenderGroup0.bindGroupId : 0;
          const srcBgDesc = hasLastRenderGroup0 ? lastRenderGroup0.bindGroupDesc : null;
          const srcDynOff = hasLastRenderGroup0 ? lastRenderGroup0.dynamicOffsets : EMPTY_DYNAMIC_OFFSETS;
          const res = this._getOrCreateRemappedBindGroup(
            false,
            activeRenderPipelineId,
            pipeline,
            activeRenderAutoLayout.wgslBindings,
            srcBgId,
            srcBgDesc,
            srcDynOff,
            this._scratchRemapResult
          );
          group0BoundForRender = true;
          if (res.status === 'ok') {
            ensureFiltered(i).push(this._acquireRemappedSetBindGroupCommand(res.bindGroup, srcBgId));
          } else if (res.status === 'empty') {
            ensureFiltered(i).push(this._acquireNoopRenderGroup0Command(srcBgId));
          }
        } else if (
          !activeRenderAutoLayout &&
          (activeRenderHasKnownLayout
            ? (activeRenderRequiredGroupsMask & invalidRenderGroupsMask) !== 0
            : invalidRenderGroupsMask !== 0)
        ) {
          const badMask = activeRenderHasKnownLayout
            ? activeRenderRequiredGroupsMask & invalidRenderGroupsMask
            : invalidRenderGroupsMask;
          this.telemetry.skippedUnboundPassOps += 1;
          this._recordTelemetryWarningOnce(
            `skip-draw-unbound:${activeRenderPipelineId || 0}:${badMask}`,
            `skipped render draw on pipeline ${activeRenderPipelineId || 0}: required bind group failed to create or is unbound`
          );
          ensureFiltered(i);
          continue;
        }
        if (filtered !== null) filtered.push(cmd);
        continue;
      }

      if (cmd.type === 'dispatchCompute' || cmd.type === 'dispatchIndirect') {
        if (activeComputePending) {
          this.telemetry.pendingAsyncPipelineSkips += 1;
          ensureFiltered(i);
          continue;
        }
        if (hadExplicitSetComputePipeline && !activeComputePipelineValid) {
          this.telemetry.skippedComputeDispatchCount += 1;
          this._recordTelemetryWarningOnce(
            `skip-dispatch-no-pipeline:${activeComputePipelineId}`,
            `skipped compute dispatch: compute pipeline ${activeComputePipelineId} is unavailable`
          );
          ensureFiltered(i);
          continue;
        }
        if (cmd.type === 'dispatchIndirect') {
          const indirectBuf = this.processor.buffers ? this.processor.buffers.get(cmd.bufferId) : null;
          const indirectMeta =
            this.processor._bufferMeta && typeof this.processor._bufferMeta.get === 'function'
              ? this.processor._bufferMeta.get(cmd.bufferId)
              : null;
          const indirectCheck = validateIndirectCommand(
            'dispatchIndirect',
            indirectBuf,
            cmd.offset,
            indirectMeta
          );
          if (!indirectCheck.valid) {
            this.telemetry.skippedInvalidIndirectOps += 1;
            this.telemetry.skippedComputeDispatchCount += 1;
            this._recordTelemetryWarningOnce(
              `skip-indirect:dispatchIndirect:${indirectCheck.reason}:${cmd.bufferId}`,
              `skipped dispatchIndirect on buffer ${cmd.bufferId} offset ${cmd.offset}: ${indirectCheck.reason}`
            );
            ensureFiltered(i);
            continue;
          }
        }
        if (cmd.type === 'dispatchCompute') {
          const rawX = cmd.workgroupX;
          const rawY = cmd.workgroupY;
          const rawZ = cmd.workgroupZ;
          const check = validateComputeDispatch(rawX, rawY, rawZ, this.device, this._scratch.computeDispatch);
          if (check.skip) {
            this.telemetry.skippedComputeDispatchCount += 1;
            this._recordTelemetryWarningOnce(
              `skip-dispatch-zero:${rawX}x${rawY}x${rawZ}`,
              `skipped compute dispatch with zero or invalid workgroups (${rawX}x${rawY}x${rawZ})`
            );
            ensureFiltered(i);
            continue;
          }
          if (check.clamped) {
            cmd.workgroupX = check.x;
            cmd.workgroupY = check.y;
            cmd.workgroupZ = check.z;
            this.telemetry.clampedComputeDispatchCount += 1;
            this._recordTelemetryWarningOnce(
              `clamp-dispatch:${rawX}x${rawY}x${rawZ}:${check.maxWorkgroupsPerDimension}`,
              `clamped compute dispatch from ${rawX}x${rawY}x${rawZ} to ${check.x}x${check.y}x${check.z} (limit=${check.maxWorkgroupsPerDimension})`
            );
          }
        }
        if (activeComputeAutoLayout && !group0BoundForCompute) {
          const pipeline = this.processor.computePipelines.get(activeComputePipelineId);
          const srcBgId = hasLastComputeGroup0 ? lastComputeGroup0.bindGroupId : 0;
          const srcBgDesc = hasLastComputeGroup0 ? lastComputeGroup0.bindGroupDesc : null;
          const srcDynOff = hasLastComputeGroup0 ? lastComputeGroup0.dynamicOffsets : EMPTY_DYNAMIC_OFFSETS;
          const res = this._getOrCreateRemappedBindGroup(
            true,
            activeComputePipelineId,
            pipeline,
            activeComputeAutoLayout.wgslBindings,
            srcBgId,
            srcBgDesc,
            srcDynOff,
            this._scratchRemapResult
          );
          group0BoundForCompute = true;
          if (res.status === 'ok') {
            ensureFiltered(i).push(this._acquireRemappedSetBindGroupCommand(res.bindGroup, srcBgId));
          }
        } else if (
          !activeComputeAutoLayout &&
          (activeComputeHasKnownLayout
            ? (activeComputeRequiredGroupsMask & ~boundComputeGroupsMask) !== 0
            : invalidComputeGroupsMask !== 0)
        ) {
          const missingMask = activeComputeHasKnownLayout
            ? activeComputeRequiredGroupsMask & ~boundComputeGroupsMask
            : invalidComputeGroupsMask;
          this.telemetry.skippedUnboundPassOps += 1;
          this.telemetry.skippedComputeDispatchCount += 1;
          this._recordTelemetryWarningOnce(
            `skip-dispatch-unbound:${activeComputePipelineId || 0}:${missingMask}`,
            `skipped compute dispatch on pipeline ${activeComputePipelineId || 0}: required bind group mask 0x${missingMask.toString(16)} is unbound`
          );
          ensureFiltered(i);
          continue;
        }
        this.telemetry.computeDispatchCount += 1;
        if (filtered !== null) filtered.push(cmd);
        continue;
      }

      if (filtered !== null) filtered.push(cmd);
    }

    if (filtered !== null) {
      this.processor._passCommands = filtered;
    }
  }

  // ---- Textures and views ----

  _createTexture2(view, offset, payloadBytes) {
    const decoded = decodeCreateTexture2(view, offset, payloadBytes, this._scratch.createTexture2);
    if (decoded.error) {
      this._packetError = decoded.error;
      return false;
    }
    const sanitized = sanitizeTextureDescriptor(
      decoded.descriptor,
      this.device,
      this._textureFormatsOptions
    );
    for (let i = 0; i < sanitized.warnings.length; i += 1) {
      const w = sanitized.warnings[i];
      this._recordTelemetryWarningOnce(w.key, w.message);
    }
    if (sanitized.bcFallback) {
      this.telemetry.bcFormatFallbacks += 1;
    }
    if (sanitized.strippedUsages.length > 0) {
      this.telemetry.usageStrips += sanitized.strippedUsages.length;
    }

    const id = decoded.id;
    const prevTexture = this.processor && this.processor.textures ? this.processor.textures.get(id) : null;
    if (prevTexture) {
      this._textureViewWeakCache.delete(prevTexture);
    }
    const prevOwner = this._textureViewOwners.get(id);
    if (prevOwner) {
      this._textureViewWeakCache.delete(prevOwner);
    }
    if (typeof this.processor._destroyTextureIfPresent === 'function') {
      this.processor._destroyTextureIfPresent(id);
    }
    this._invalidatePresentXCacheForResource(id);
    this._bumpTextureGeneration(id);
    this._textureViewOwners.delete(id);
    this.textureViews.delete(id);
    this.textureMeta.delete(id);

    let texture;
    try {
      texture = this.device.createTexture(sanitized.descriptor);
    } catch (err) {
      this._packetError = `CreateTexture2 ${id}: ${String(err && err.message ? err.message : err)}`;
      return false;
    }
    this.processor.textures.set(id, texture);
    this._textureViewOwners.set(id, texture);
    let defaultView = null;
    if (texture && typeof texture.createView === 'function') {
      try {
        defaultView = texture.createView();
      } catch (_) {
        defaultView = null;
      }
      if (defaultView && this.processor.textureViews) {
        this.processor.textureViews.set(id, defaultView);
      }
    }
    if (this.processor.resourceRegistry && typeof this.processor.resourceRegistry.register === 'function') {
      this.processor.resourceRegistry.register('texture', id, texture, {
        byteSize: 0,
        label: `texture:${id}`,
      });
      if (defaultView) {
        this.processor.resourceRegistry.register('textureView', id, defaultView, {
          byteSize: 0,
          label: `textureView:${id}`,
        });
      }
    }
    this.textureMeta.set(id, {
      id,
      requestedFormat: sanitized.originalFormat,
      format: sanitized.effectiveFormat,
      bcFallback: sanitized.bcFallback,
      formatFallback: sanitized.formatFallback,
      usage: sanitized.descriptor.usage,
      dimension: sanitized.descriptor.dimension,
      mipLevelCount: sanitized.descriptor.mipLevelCount,
      sampleCount: sanitized.descriptor.sampleCount,
      width: sanitized.descriptor.size.width,
      height: sanitized.descriptor.size.height,
      depthOrArrayLayers: sanitized.descriptor.size.depthOrArrayLayers,
      viewFormats: sanitized.descriptor.viewFormats || null,
    });
    this.telemetry.texture2Count += 1;
    return true;
  }

  _clearTexture(view, offset, payloadBytes) {
    if (
      !view ||
      payloadBytes < CLEAR_TEXTURE_PAYLOAD_BYTES ||
      offset + CLEAR_TEXTURE_PAYLOAD_BYTES > view.byteLength
    ) {
      this._packetError = `ClearTexture: payload ${payloadBytes} < ${CLEAR_TEXTURE_PAYLOAD_BYTES} bytes`;
      return false;
    }
    const textureId = view.getUint32(offset, true);
    const meta = this.textureMeta.get(textureId) || null;
    if (this.processor && typeof this.processor._clearTextureById === 'function') {
      const result = this.processor._clearTextureById(textureId, meta);
      if (result.status === 'cleared') {
        this.telemetry.textureClears += 1;
        return true;
      }
      if (result.status === 'missing') {
        this.telemetry.textureClearsMissing += 1;
        return true;
      }
      this._packetError = result.error || `ClearTexture ${textureId}: invalid texture`;
      return false;
    }
    this.telemetry.textureClearsMissing += 1;
    return true;
  }

  _resolveTextureMetaFull(textureId) {
    const meta = this.textureMeta.get(textureId) || null;
    const texture =
      this.processor && this.processor.textures ? this.processor.textures.get(textureId) || null : null;
    const desc = texture && texture.desc ? texture.desc : null;
    const width =
      (texture && typeof texture.width === 'number' && texture.width > 0 ? texture.width : null) ||
      (meta && typeof meta.width === 'number' && meta.width > 0 ? meta.width : null);
    const height =
      (texture && typeof texture.height === 'number' && texture.height > 0 ? texture.height : null) ||
      (meta && typeof meta.height === 'number' && meta.height > 0 ? meta.height : null);
    const depthOrArrayLayers =
      (texture && typeof texture.depthOrArrayLayers === 'number' && texture.depthOrArrayLayers > 0
        ? texture.depthOrArrayLayers
        : null) ||
      (meta && typeof meta.depthOrArrayLayers === 'number' && meta.depthOrArrayLayers > 0
        ? meta.depthOrArrayLayers
        : 1);
    const mipLevelCount =
      (texture && typeof texture.mipLevelCount === 'number' && texture.mipLevelCount >= 1
        ? texture.mipLevelCount
        : null) ||
      (meta && typeof meta.mipLevelCount === 'number' && meta.mipLevelCount >= 1
        ? meta.mipLevelCount
        : null);
    const sampleCount =
      (texture && typeof texture.sampleCount === 'number' && texture.sampleCount >= 1
        ? texture.sampleCount
        : null) ||
      (meta && typeof meta.sampleCount === 'number' && meta.sampleCount >= 1
        ? meta.sampleCount
        : null) ||
      (desc && typeof desc.sampleCount === 'number' && desc.sampleCount >= 1
        ? desc.sampleCount
        : null);
    const format =
      (meta && typeof meta.format === 'string' ? meta.format : null) ||
      (texture && typeof texture.format === 'string' ? texture.format : null) ||
      (desc && typeof desc.format === 'string' ? desc.format : null);
    const usage =
      meta && typeof meta.usage === 'number'
        ? meta.usage
        : texture && typeof texture.usage === 'number'
          ? texture.usage
          : desc && typeof desc.usage === 'number'
            ? desc.usage
            : null;
    const dimension =
      (texture && typeof texture.dimension === 'string' ? texture.dimension : null) ||
      (meta && typeof meta.dimension === 'string' ? meta.dimension : null) ||
      (desc && typeof desc.dimension === 'string' ? desc.dimension : null);
    return {
      texture,
      meta,
      width,
      height,
      depthOrArrayLayers,
      mipLevelCount,
      sampleCount,
      format,
      usage,
      dimension,
    };
  }

  _resolveTextureFormat(textureId, viewDesc) {
    if (viewDesc && typeof viewDesc.format === 'string' && viewDesc.format !== 'undefined') {
      const meta = this.textureMeta.get(textureId);
      if (meta && meta.bcFallback && isBcCompressedFormat(viewDesc.format)) {
        return meta.format;
      }
      if (
        meta &&
        meta.formatFallback &&
        viewDesc.format === 'depth32float-stencil8' &&
        meta.format === 'depth24plus-stencil8'
      ) {
        return meta.format;
      }
      return viewDesc.format;
    }
    const meta = this.textureMeta.get(textureId);
    if (meta && meta.format) return meta.format;
    const texture = this.processor.textures.get(textureId);
    if (!texture) return null;
    if (typeof texture.format === 'string') return texture.format;
    if (texture.desc && typeof texture.desc.format === 'string') return texture.desc.format;
    return null;
  }

  _resolveActualTextureDimension(texture, meta) {
    if (texture && typeof texture.dimension === 'string') return texture.dimension;
    if (texture && texture.desc && typeof texture.desc.dimension === 'string') {
      return texture.desc.dimension;
    }
    if (meta && typeof meta.dimension === 'string') return meta.dimension;
    return null;
  }

  // W354: WeakMap<GPUTexture, { defaultView, byKey }> texture-view cache respecting W347 1D->2D
  // promoted LUT coercion and reverse 1D guard, with zero descriptor allocation on cache hits.
  getOrCreateTextureView(texture, desc = null, meta = null, textureIdForError = 0) {
    if (!texture || typeof texture !== 'object') return null;
    if (texture.destroyed === true) {
      this._textureViewWeakCache.delete(texture);
      return null;
    }
    if (typeof texture.createView !== 'function') return null;

    const hasDesc = Boolean(desc && typeof desc === 'object');
    let effFormat = hasDesc && typeof desc.format === 'string' && desc.format !== '' ? desc.format : '';
    let effDim = hasDesc && typeof desc.dimension === 'string' && desc.dimension !== '' ? desc.dimension : '';
    let effAspect = hasDesc && typeof desc.aspect === 'string' && desc.aspect !== '' ? desc.aspect : '';
    let effBaseMip = hasDesc && desc.baseMipLevel !== undefined ? desc.baseMipLevel : undefined;
    let effMipCount = hasDesc && desc.mipLevelCount !== undefined ? desc.mipLevelCount : undefined;
    let effBaseLayer = hasDesc && desc.baseArrayLayer !== undefined ? desc.baseArrayLayer : undefined;
    let effLayerCount = hasDesc && desc.arrayLayerCount !== undefined ? desc.arrayLayerCount : undefined;
    let modified = false;

    const actualTexFormat =
      (meta && typeof meta.format === 'string' ? meta.format : null) ||
      (typeof texture.format === 'string' ? texture.format : null) ||
      (texture.desc && typeof texture.desc.format === 'string' ? texture.desc.format : null);

    if (effFormat && meta) {
      if (meta.bcFallback && isBcCompressedFormat(effFormat)) {
        effFormat = meta.format;
        modified = true;
      } else if (
        meta.formatFallback &&
        effFormat === 'depth32float-stencil8' &&
        meta.format === 'depth24plus-stencil8'
      ) {
        effFormat = meta.format;
        modified = true;
      } else if (
        meta.formatFallback &&
        Object.prototype.hasOwnProperty.call(NORM16_TO_FLOAT16_FALLBACK, effFormat) &&
        NORM16_TO_FLOAT16_FALLBACK[effFormat] === meta.format
      ) {
        effFormat = meta.format;
        modified = true;
      }
    }

    // Sanitize aspect against actual texture format (e.g. stencil-only on depth-only or depth/stencil aspect on color).
    if (effAspect && actualTexFormat) {
      if (effAspect === 'depth-only' && !hasDepthAspect(actualTexFormat)) {
        effAspect = 'all';
        modified = true;
      } else if (effAspect === 'stencil-only' && !hasStencilAspect(actualTexFormat)) {
        effAspect = 'all';
        modified = true;
      }
    }

    if (effAspect === 'depth-only') {
      if (effFormat === 'depth32float-stencil8') {
        effFormat = 'depth32float';
        modified = true;
      } else if (effFormat === 'depth24plus-stencil8') {
        effFormat = 'depth24plus';
        modified = true;
      }
    } else if (effAspect === 'stencil-only') {
      if (effFormat === 'depth32float-stencil8' || effFormat === 'depth24plus-stencil8') {
        effFormat = 'stencil8';
        modified = true;
      }
    }

    // Reject incompatible view format before Dawn creates an error view (except valid depth/stencil sub-aspects).
    if (effFormat && actualTexFormat) {
      const effectiveBaseForAspect = resolveEffectiveViewFormat(actualTexFormat, effAspect || 'all');
      if (
        !isViewFormatCompatible(actualTexFormat, effFormat) &&
        !isViewFormatCompatible(effectiveBaseForAspect, effFormat)
      ) {
        if (textureIdForError) {
          this._packetError = `texture ${textureIdForError} createView: incompatible view format ${effFormat} for ${actualTexFormat}`;
        }
        return null;
      }
    }

    const actualTexDim = this._resolveActualTextureDimension(texture, meta);
    const isPromoted1DTo2D = Boolean(
      (meta && meta.promoted1DTo2D === true) ||
        texture.promoted1DTo2D === true ||
        (texture.desc && texture.desc.promoted1DTo2D === true)
    );
    const texMipCount =
      typeof texture.mipLevelCount === 'number' && texture.mipLevelCount >= 1
        ? texture.mipLevelCount
        : texture.desc &&
            typeof texture.desc.mipLevelCount === 'number' &&
            texture.desc.mipLevelCount >= 1
          ? texture.desc.mipLevelCount
          : meta && typeof meta.mipLevelCount === 'number' && meta.mipLevelCount >= 1
            ? meta.mipLevelCount
            : isPromoted1DTo2D
              ? 1
              : null;
    const texLayers =
      typeof texture.depthOrArrayLayers === 'number' && texture.depthOrArrayLayers >= 1
        ? texture.depthOrArrayLayers
        : texture.desc &&
            texture.desc.size &&
            typeof texture.desc.size.depthOrArrayLayers === 'number' &&
            texture.desc.size.depthOrArrayLayers >= 1
          ? texture.desc.size.depthOrArrayLayers
          : meta && typeof meta.depthOrArrayLayers === 'number' && meta.depthOrArrayLayers >= 1
            ? meta.depthOrArrayLayers
            : null;

    if (hasDesc && effDim === '1d' && (actualTexDim === '2d' || isPromoted1DTo2D)) {
      // W347: W341/W342 promote 1D LUT textures (e.g. 4096x1) to WebGPU dimension '2d';
      // coerce explicit '1d' view descriptors to '2d' with valid single-layer/mip bounds.
      effDim = '2d';
      effBaseLayer = 0;
      effLayerCount = 1;
      if (isPromoted1DTo2D || texMipCount === 1) {
        if (effBaseMip !== undefined && effBaseMip !== 0) {
          effBaseMip = 0;
        }
        effMipCount = 1;
      }
      modified = true;
    } else if (
      actualTexDim === '1d' &&
      hasDesc &&
      effDim &&
      effDim !== 'undefined' &&
      effDim !== '1d'
    ) {
      // W347: Reverse guard — real WebGPU '1d' textures only allow '1d' views (1 mip, 1 layer).
      effDim = '1d';
      if (effBaseLayer !== undefined && effBaseLayer !== 0) effBaseLayer = 0;
      if (effLayerCount !== undefined && effLayerCount !== 1) effLayerCount = 1;
      if (effBaseMip !== undefined && effBaseMip !== 0) effBaseMip = 0;
      if (effMipCount !== undefined && effMipCount !== 1) effMipCount = 1;
      modified = true;
    }

    // Clamp mip/layer ranges against known texture bounds so Dawn never rejects OOB views.
    if (texMipCount !== null) {
      if (effBaseMip !== undefined && effBaseMip >= texMipCount) {
        effBaseMip = Math.max(0, texMipCount - 1);
        modified = true;
      }
      const bMipVal = effBaseMip !== undefined ? effBaseMip : 0;
      if (effMipCount !== undefined && bMipVal + effMipCount > texMipCount) {
        effMipCount = Math.max(1, texMipCount - bMipVal);
        modified = true;
      }
    }
    if (effDim === '3d' || actualTexDim === '3d') {
      if (effBaseLayer !== undefined && effBaseLayer !== 0) {
        effBaseLayer = undefined;
        modified = true;
      }
      if (effLayerCount !== undefined && effLayerCount !== 1) {
        effLayerCount = undefined;
        modified = true;
      }
    } else {
      if (texLayers !== null) {
        if (effBaseLayer !== undefined && effBaseLayer >= texLayers) {
          effBaseLayer = Math.max(0, texLayers - 1);
          modified = true;
        }
        const bLayVal = effBaseLayer !== undefined ? effBaseLayer : 0;
        if (effLayerCount !== undefined && bLayVal + effLayerCount > texLayers) {
          effLayerCount = Math.max(1, texLayers - bLayVal);
          modified = true;
        }
      }
      if (effDim === '2d' && effLayerCount !== undefined && effLayerCount > 1) {
        effLayerCount = 1;
        modified = true;
      }
    }

    const bMip = effBaseMip || 0;
    const mCnt = effMipCount || 0;
    const bLay = effBaseLayer || 0;
    const lCnt = effLayerCount || 0;
    const isDefaultKey = !effFormat && !effDim && !effAspect && !bMip && !mCnt && !bLay && !lCnt;

    let record = this._textureViewWeakCache.get(texture);
    if (isDefaultKey) {
      if (record && record.defaultView) {
        if (record.defaultView.destroyed !== true) {
          this.telemetry.textureViewCacheHits += 1;
          return record.defaultView;
        }
        record.defaultView = null;
      }
      this.telemetry.textureViewCacheMisses += 1;
      let view;
      try {
        view = hasDesc ? texture.createView(desc) : texture.createView();
      } catch (err) {
        if (textureIdForError) {
          this._packetError = `texture ${textureIdForError} createView: ${String(
            err && err.message ? err.message : err
          )}`;
        }
        return null;
      }
      if (view && typeof view === 'object') {
        if (!record) {
          this._textureViewWeakCache.set(texture, { defaultView: view, byKey: null });
        } else {
          record.defaultView = view;
        }
      }
      return view;
    }

    const key = `${effFormat}|${effDim}|${effAspect}|${bMip}|${mCnt}|${bLay}|${lCnt}`;
    if (record && record.byKey !== null) {
      const cached = record.byKey.get(key);
      if (cached) {
        if (cached.destroyed !== true) {
          this.telemetry.textureViewCacheHits += 1;
          return cached;
        }
        record.byKey.delete(key);
      }
    }

    this.telemetry.textureViewCacheMisses += 1;
    let createDesc;
    if (!modified && hasDesc) {
      createDesc = desc;
    } else {
      createDesc = hasDesc ? { ...desc } : {};
      if (effFormat) createDesc.format = effFormat;
      if (effDim) createDesc.dimension = effDim;
      if (effAspect) createDesc.aspect = effAspect;
      if (effBaseMip !== undefined) createDesc.baseMipLevel = effBaseMip;
      else if ('baseMipLevel' in createDesc) delete createDesc.baseMipLevel;
      if (effMipCount !== undefined) createDesc.mipLevelCount = effMipCount;
      else if ('mipLevelCount' in createDesc) delete createDesc.mipLevelCount;
      if (effBaseLayer !== undefined) createDesc.baseArrayLayer = effBaseLayer;
      else if ('baseArrayLayer' in createDesc) delete createDesc.baseArrayLayer;
      if (effLayerCount !== undefined) createDesc.arrayLayerCount = effLayerCount;
      else if ('arrayLayerCount' in createDesc) delete createDesc.arrayLayerCount;
    }

    let textureView;
    try {
      textureView = texture.createView(createDesc);
    } catch (err) {
      if (textureIdForError) {
        this._packetError = `texture ${textureIdForError} createView: ${String(
          err && err.message ? err.message : err
        )}`;
      }
      return null;
    }
    if (textureView && typeof textureView === 'object') {
      if (!record) {
        const byKey = new Map();
        byKey.set(key, textureView);
        this._textureViewWeakCache.set(texture, { defaultView: null, byKey });
      } else {
        if (record.byKey === null) {
          record.byKey = new Map();
        } else if (record.byKey.size >= MAX_VIEWS_PER_TEXTURE) {
          const oldestKey = record.byKey.keys().next().value;
          record.byKey.delete(oldestKey);
        }
        record.byKey.set(key, textureView);
      }
    }
    return textureView;
  }

  _textureView(textureId, desc) {
    this._syncDeviceGeneration();
    const texture = this.processor.textures.get(textureId);
    if (!texture || texture.destroyed === true || typeof texture.createView !== 'function') {
      if (texture && texture.destroyed === true) {
        this._textureViewWeakCache.delete(texture);
        this.textureViews.delete(textureId);
      }
      return null;
    }
    this._getTextureGeneration(textureId, texture);
    const meta = this.textureMeta.get(textureId) || null;
    return this.getOrCreateTextureView(texture, desc, meta, textureId);
  }

  // Attachment view resolver; the canvas texture only exists at Present (ctl.swapchainView).
  // W354 + Dawn parity: non-swapchain attachments enforce single-mip/single-layer views when the
  // underlying texture is mipmapped/layered/3D, and depth-stencil attachments always use aspect 'all'.
  _attachmentView(textureId, desc, options = null) {
    if (textureId === SWAPCHAIN_TEXTURE_ID) {
      const requestedFormat =
        desc && typeof desc.format === 'string' && desc.format !== 'undefined' ? desc.format : null;
      if (!requestedFormat) {
        return SWAPCHAIN_VIEW_SENTINEL;
      }
      return (ctl) => {
        if (
          ctl &&
          ctl.swapchainTexture &&
          ctl.swapchainTexture.destroyed !== true &&
          typeof ctl.swapchainTexture.createView === 'function' &&
          this.processor &&
          this.processor._swapchainViewFormat &&
          this.processor._swapchainViewFormat !== requestedFormat &&
          (!this.processor._swapchain || !this.processor._swapchain.usesBlitFallback)
        ) {
          const swapBaseFmt =
            typeof ctl.swapchainTexture.format === 'string'
              ? ctl.swapchainTexture.format
              : this.processor._swapchainFormat || null;
          // #1896 parity: never attempt a cross-channel view (e.g. rgba8unorm on bgra8unorm).
          if (!swapBaseFmt || isViewFormatCompatible(swapBaseFmt, requestedFormat)) {
            const formatDesc = this._scratchSwapchainFormatDesc;
            formatDesc.format = requestedFormat;
            const cachedView = this.getOrCreateTextureView(ctl.swapchainTexture, formatDesc, null, 0);
            if (cachedView) return cachedView;
          }
        }
        return ctl.swapchainView;
      };
    }
    const isDepthStencil = Boolean(options && options.isDepthStencil);
    const full = this._resolveTextureMetaFull(textureId);
    const hasDesc = Boolean(desc && typeof desc === 'object');
    const actualDim = full.dimension || '2d';
    const is3d = actualDim === '3d';
    const isMultiMip = full.mipLevelCount !== null && full.mipLevelCount > 1;
    const isLayered = !is3d && full.depthOrArrayLayers > 1;
    const badAspect =
      isDepthStencil && hasDesc && (desc.aspect === 'depth-only' || desc.aspect === 'stencil-only');
    const badMipCount = hasDesc && desc.mipLevelCount !== undefined && desc.mipLevelCount !== 1;
    const badLayerCount = !is3d && hasDesc && desc.arrayLayerCount !== undefined && desc.arrayLayerCount !== 1;
    const badDim =
      hasDesc &&
      desc.dimension &&
      desc.dimension !== 'undefined' &&
      (is3d ? desc.dimension !== '3d' : desc.dimension !== '2d');

    if (isMultiMip || isLayered || is3d || badAspect || badMipCount || badLayerCount || badDim) {
      const attDesc = hasDesc ? { ...desc } : {};
      if (badAspect) {
        attDesc.aspect = 'all';
        if (full.format && attDesc.format && attDesc.format !== full.format) {
          delete attDesc.format;
        }
      }
      const rawBaseMip = typeof attDesc.baseMipLevel === 'number' && attDesc.baseMipLevel > 0 ? Math.floor(attDesc.baseMipLevel) : 0;
      const clampedMip =
        full.mipLevelCount !== null ? Math.min(full.mipLevelCount - 1, rawBaseMip) : rawBaseMip;
      if (clampedMip > 0) attDesc.baseMipLevel = clampedMip;
      else if ('baseMipLevel' in attDesc) delete attDesc.baseMipLevel;
      attDesc.mipLevelCount = 1;
      if (is3d) {
        attDesc.dimension = '3d';
        delete attDesc.baseArrayLayer;
        delete attDesc.arrayLayerCount;
      } else {
        attDesc.dimension = '2d';
        const rawBaseLayer =
          typeof attDesc.baseArrayLayer === 'number' && attDesc.baseArrayLayer > 0
            ? Math.floor(attDesc.baseArrayLayer)
            : 0;
        const clampedLayer = Math.min(Math.max(0, full.depthOrArrayLayers - 1), rawBaseLayer);
        if (clampedLayer > 0) attDesc.baseArrayLayer = clampedLayer;
        else if ('baseArrayLayer' in attDesc) delete attDesc.baseArrayLayer;
        attDesc.arrayLayerCount = 1;
      }
      return this._textureView(textureId, attDesc);
    }
    return this._textureView(textureId, desc);
  }

  _resolveAttachmentEntry(entry, ctl) {
    if (entry === SWAPCHAIN_VIEW_SENTINEL) return ctl.swapchainView;
    if (typeof entry === 'function') return entry(ctl);
    return entry;
  }

  _textureAttachmentSize(textureId, viewDesc, out = null) {
    const full = this._resolveTextureMetaFull(textureId);
    const base = validTargetSize(full.width, full.height, out);
    if (!base) return null;
    const rawMip =
      viewDesc && typeof viewDesc.baseMipLevel === 'number' && viewDesc.baseMipLevel > 0
        ? Math.floor(viewDesc.baseMipLevel)
        : 0;
    const mip =
      full.mipLevelCount !== null ? Math.min(full.mipLevelCount - 1, rawMip) : rawMip;
    if (mip === 0) return base;
    const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
    base.width = scale === Infinity ? 1 : Math.max(1, Math.floor(base.width / scale));
    base.height = scale === Infinity ? 1 : Math.max(1, Math.floor(base.height / scale));
    return base;
  }

  _resolveSwapchainSize(ctl, out = null) {
    if (ctl) {
      if (ctl.swapchainTexture) {
        const fromTex = validTargetSize(ctl.swapchainTexture.width, ctl.swapchainTexture.height, out);
        if (fromTex) return fromTex;
      }
      const fromCtl =
        validTargetSize(ctl.swapchainWidth, ctl.swapchainHeight, out) ||
        validTargetSize(ctl.canvasWidth, ctl.canvasHeight, out);
      if (fromCtl) return fromCtl;
      if (typeof ctl.getCanvasTexture === 'function') {
        try {
          const tex = ctl.getCanvasTexture();
          if (tex) {
            const fromGetter = validTargetSize(tex.width, tex.height, out);
            if (fromGetter) return fromGetter;
          }
        } catch (_) {
          // Fall through to remaining swapchain size probes.
        }
      }
      if (ctl.swapchainView) {
        const fromView =
          validTargetSize(ctl.swapchainView.width, ctl.swapchainView.height, out) ||
          (ctl.swapchainView.texture &&
            validTargetSize(ctl.swapchainView.texture.width, ctl.swapchainView.texture.height, out));
        if (fromView) return fromView;
      }
    }
    const proc = this.processor;
    if (proc) {
      if (proc._swapchain && !proc._swapchain.isDestroyed) {
        const fromSwapchain = validTargetSize(proc._swapchain.width, proc._swapchain.height, out);
        if (fromSwapchain) return fromSwapchain;
      }
      if (proc.canvas) {
        const fromProcCanvas = validTargetSize(proc.canvas.width, proc.canvas.height, out);
        if (fromProcCanvas) return fromProcCanvas;
      }
      if (proc.ctx) {
        if (proc.ctx.canvas) {
          const fromCtxCanvas = validTargetSize(proc.ctx.canvas.width, proc.ctx.canvas.height, out);
          if (fromCtxCanvas) return fromCtxCanvas;
        }
        if (typeof proc.ctx.getCurrentTexture === 'function') {
          try {
            const current = proc.ctx.getCurrentTexture();
            if (current) {
              const fromCurrent = validTargetSize(current.width, current.height, out);
              if (fromCurrent) return fromCurrent;
            }
          } catch (_) {
            // Fall back to null when swapchain texture dimensions cannot be queried.
          }
        }
      }
    }
    return null;
  }

  _getActivePassTargetSize(ctl, out = null) {
    if (this._activePass !== null && ctl.pass === this._activePass) {
      return this._activePassTargetSize;
    }
    this._activePass = null;
    this._activePassTargetSize = null;
    return this._resolveSwapchainSize(ctl, out);
  }

  // The exact GPURenderPassDescriptor for a decoded BeginRenderPass (views resolved via `ctl`).
  buildRenderPassDescriptor(decoded, resolvers, ctl) {
    const colorCount = decoded.colors.length;
    const colorSizes = this._scratchColorSizes;
    const colorSampleCounts = this._scratchColorSampleCounts;
    const colorFormats = this._scratchColorFormats;
    colorSizes.length = colorCount;
    colorSampleCounts.length = colorCount;
    colorFormats.length = colorCount;
    const colorSizeSlots = this._scratchColorSizeSlots;
    const colorAttachments = new Array(colorCount);
    for (let i = 0; i < colorCount; i += 1) {
      const c = decoded.colors[i];
      if (!c) {
        colorSizes[i] = null;
        colorSampleCounts[i] = null;
        colorFormats[i] = null;
        colorAttachments[i] = null;
        continue;
      }
      const isSwap = c.textureId === SWAPCHAIN_TEXTURE_ID;
      const full = isSwap ? null : this._resolveTextureMetaFull(c.textureId);
      // Drop color attachments lacking RENDER_ATTACHMENT usage or carrying a depth/stencil format.
      if (!isSwap && full) {
        const cFmt = (c.view && c.view.format) || full.format;
        if (full.usage !== null && (full.usage & GPU_TEXTURE_USAGE_RENDER_ATTACHMENT) === 0) {
          this.telemetry.sanitizedAttachments += 1;
          this._recordTelemetryWarningOnce(
            `rp-color-no-ra:${c.textureId}`,
            `render pass: dropped colorAttachments[${i}] (texture ${c.textureId}) lacking RENDER_ATTACHMENT usage`
          );
          colorSizes[i] = null;
          colorSampleCounts[i] = null;
          colorFormats[i] = null;
          colorAttachments[i] = null;
          continue;
        }
        if (cFmt && isDepthOrStencilTextureFormat(cFmt)) {
          this.telemetry.sanitizedAttachments += 1;
          this._recordTelemetryWarningOnce(
            `rp-color-ds-format:${c.textureId}:${cFmt}`,
            `render pass: dropped colorAttachments[${i}] with depth-stencil format ${cFmt}`
          );
          colorSizes[i] = null;
          colorSampleCounts[i] = null;
          colorFormats[i] = null;
          colorAttachments[i] = null;
          continue;
        }
      }
      const sizeSlot = i < colorSizeSlots.length ? colorSizeSlots[i] : null;
      colorSizes[i] = isSwap
        ? this._resolveSwapchainSize(ctl, sizeSlot)
        : this._textureAttachmentSize(c.textureId, c.view, sizeSlot);
      colorSampleCounts[i] = isSwap ? 1 : full && full.sampleCount !== null ? full.sampleCount : null;
      colorFormats[i] = isSwap
        ? (c.view && c.view.format) || (this.processor && this.processor._swapchainFormat) || null
        : (c.view && c.view.format) || (full ? full.format : null);
      const loadOp = c.loadOp === 'clear' || c.loadOp === 'load' ? c.loadOp : 'load';
      const storeOp = c.storeOp === 'discard' || c.storeOp === 'store' ? c.storeOp : 'store';
      const attachment = {
        view: this._resolveAttachmentEntry(resolvers.colors[i], ctl),
        loadOp,
        storeOp,
      };
      if (loadOp === 'clear') {
        attachment.clearValue = {
          r: finiteOr(c.clearValue && c.clearValue.r, 0) || 0,
          g: finiteOr(c.clearValue && c.clearValue.g, 0) || 0,
          b: finiteOr(c.clearValue && c.clearValue.b, 0) || 0,
          a: finiteOr(c.clearValue && c.clearValue.a, 1) || 0,
        };
      }
      // WebGPU requires depthSlice on 3D color attachments.
      if (!isSwap && full && full.dimension === '3d') {
        const mip =
          c.view && typeof c.view.baseMipLevel === 'number' && c.view.baseMipLevel > 0
            ? Math.floor(c.view.baseMipLevel)
            : 0;
        const mipDepth = Math.max(1, Math.floor(full.depthOrArrayLayers / Math.pow(2, Math.min(30, mip))));
        const rawSlice =
          c.view && typeof c.view.baseArrayLayer === 'number' && c.view.baseArrayLayer > 0
            ? Math.floor(c.view.baseArrayLayer)
            : 0;
        attachment.depthSlice = Math.min(mipDepth - 1, Math.max(0, rawSlice));
      }
      colorAttachments[i] = attachment;
    }
    const descriptor = { colorAttachments };
    const d = decoded.depthStencil;
    let depthStencilSize = null;
    let depthSampleCount = null;
    let depthFormat = null;
    let depthUsage = null;
    if (d && d.textureId !== SWAPCHAIN_TEXTURE_ID) {
      const dsFull = this._resolveTextureMetaFull(d.textureId);
      depthFormat = dsFull.format || (d.view && d.view.format) || null;
      depthUsage = dsFull.usage;
      depthSampleCount = dsFull.sampleCount;
      depthStencilSize = this._textureAttachmentSize(d.textureId, d.view, this._scratchDepthSizeSlot);
      const ds = { view: this._resolveAttachmentEntry(resolvers.depthStencil, ctl) };
      const hasDepth = depthFormat ? hasDepthAspect(depthFormat) : null;
      const hasStencil = depthFormat ? hasStencilAspect(depthFormat) : null;

      // WebGPU depth-stencil rules:
      // - Forbid depthLoadOp/depthStoreOp/depthClearValue when depthReadOnly: true or !hasDepth.
      // - Require valid depthLoadOp ('load'|'clear') & depthStoreOp ('store'|'discard') when !depthReadOnly && hasDepth.
      if (hasDepth !== false) {
        if (d.depthReadOnly) {
          ds.depthReadOnly = true;
        } else if (hasDepth === true || d.depthLoadOp !== 0 || d.depthStoreOp !== 0) {
          const rawDL = d.depthLoadOp !== 0 ? nameOf(LOAD_OP_NAMES, d.depthLoadOp) : undefined;
          const rawDS = d.depthStoreOp !== 0 ? nameOf(STORE_OP_NAMES, d.depthStoreOp) : undefined;
          ds.depthLoadOp = rawDL === 'clear' || rawDL === 'load' ? rawDL : 'load';
          ds.depthStoreOp = rawDS === 'discard' || rawDS === 'store' ? rawDS : 'store';
          if (ds.depthLoadOp === 'clear') {
            // Pass raw depthClearValue so sanitizeRenderPassAttachments clamps and accounts for it in telemetry.
            ds.depthClearValue = d.depthClearValue;
          }
        }
      }
      if (hasStencil !== false) {
        if (d.stencilReadOnly) {
          ds.stencilReadOnly = true;
        } else if (hasStencil === true || d.stencilLoadOp !== 0 || d.stencilStoreOp !== 0) {
          const rawSL = d.stencilLoadOp !== 0 ? nameOf(LOAD_OP_NAMES, d.stencilLoadOp) : undefined;
          const rawSS = d.stencilStoreOp !== 0 ? nameOf(STORE_OP_NAMES, d.stencilStoreOp) : undefined;
          ds.stencilLoadOp = rawSL === 'clear' || rawSL === 'load' ? rawSL : 'load';
          ds.stencilStoreOp = rawSS === 'discard' || rawSS === 'store' ? rawSS : 'store';
          if (ds.stencilLoadOp === 'clear') {
            ds.stencilClearValue = finiteOr(d.stencilClearValue, 0) >>> 0;
          }
        }
      }
      descriptor.depthStencilAttachment = ds;
    } else if (d && d.textureId === SWAPCHAIN_TEXTURE_ID) {
      this.telemetry.sanitizedAttachments += 1;
      this._recordTelemetryWarningOnce(
        'rp-depth-swapchain',
        'render pass: dropped swapchain texture bound as depthStencilAttachment'
      );
    }
    this._scratchAttachmentSizes.colors = colorSizes;
    this._scratchAttachmentSizes.colorSampleCounts = colorSampleCounts;
    this._scratchAttachmentSizes.colorFormats = colorFormats;
    this._scratchAttachmentSizes.depthStencil = depthStencilSize;
    this._scratchAttachmentSizes.depthSampleCount = depthSampleCount;
    this._scratchAttachmentSizes.depthFormat = depthFormat;
    this._scratchAttachmentSizes.depthUsage = depthUsage;
    const sanitized = sanitizeRenderPassAttachments(descriptor, this._scratchAttachmentSizes);
    colorSizes.length = 0;
    colorSampleCounts.length = 0;
    colorFormats.length = 0;
    this._scratchAttachmentSizes.depthStencil = null;
    this._scratchAttachmentSizes.depthSampleCount = null;
    this._scratchAttachmentSizes.depthFormat = null;
    this._scratchAttachmentSizes.depthUsage = null;
    this._lastPassReferenceSize = sanitized.referenceSize;
    if (sanitized.sanitizedCount > 0) {
      this.telemetry.sanitizedAttachments += sanitized.sanitizedCount;
      for (let i = 0; i < sanitized.warnings.length; i += 1) {
        const w = sanitized.warnings[i];
        this._recordTelemetryWarningOnce(w.key, w.message);
      }
    }
    return sanitized.descriptor;
  }

  // W104: report a missing render-pass attachment once per texture id and device generation
  // instead of once per frame.
  _missingRenderPassTexture(textureId) {
    this._packetError = `render pass: unknown texture ${textureId}`;
    const generation =
      this.processor && Number.isInteger(this.processor._deviceGeneration)
        ? this.processor._deviceGeneration
        : 0;
    const key = `${generation}:${textureId}`;
    if (this._reportedMissingTextures.has(key)) {
      this._suppressPacketError = true;
    } else {
      if (this._reportedMissingTextures.size >= MISSING_TEXTURE_REPORT_LIMIT) {
        this._reportedMissingTextures.clear();
      }
      this._reportedMissingTextures.add(key);
    }
    return false;
  }

  _beginRenderPass(view, offset) {
    const cmd = this._peekBeginRenderPassCommand();
    const decoded = decodeBeginRenderPass(view, offset, cmd._decoded);
    const colorCount = decoded.colors.length;
    const colorResolvers = cmd._resolvers.colors;
    colorResolvers.length = colorCount;
    cmd._resolvers.depthStencil = null;
    let targetAttachment = null;
    for (let i = 0; i < colorCount; i += 1) {
      const c = decoded.colors[i];
      if (!c) {
        colorResolvers[i] = null;
        continue;
      }
      if (!targetAttachment) targetAttachment = c;
      if (
        c.textureId === SWAPCHAIN_TEXTURE_ID &&
        c.view &&
        c.view.format &&
        c.view.format !== 'undefined' &&
        typeof this.processor.setSwapchainFormat === 'function'
      ) {
        // #709: match canvas swapchain format to the stream's declared attachment format.
        this.processor.setSwapchainFormat(c.view.format);
      }
      const resolver = this._attachmentView(c.textureId, c.view);
      if (!resolver) {
        colorResolvers.length = 0;
        cmd._resolvers.depthStencil = null;
        decoded.colors.length = 0;
        decoded.depthStencil = null;
        return this._missingRenderPassTexture(c.textureId);
      }
      colorResolvers[i] = resolver;
    }
    let dsResolver = null;
    if (decoded.depthStencil) {
      if (!targetAttachment) targetAttachment = decoded.depthStencil;
      dsResolver = this._attachmentView(decoded.depthStencil.textureId, decoded.depthStencil.view, {
        isDepthStencil: true,
      });
      if (!dsResolver) {
        colorResolvers.length = 0;
        cmd._resolvers.depthStencil = null;
        decoded.colors.length = 0;
        decoded.depthStencil = null;
        return this._missingRenderPassTexture(decoded.depthStencil.textureId);
      }
    }
    cmd._resolvers.depthStencil = dsResolver;
    const targetIsSwapchain = Boolean(targetAttachment && targetAttachment.textureId === SWAPCHAIN_TEXTURE_ID);
    const staticTargetSize =
      targetAttachment && !targetIsSwapchain
        ? this._textureAttachmentSize(
            targetAttachment.textureId,
            targetAttachment.view,
            cmd._staticTargetSizeSlot
          )
        : null;
    cmd._targetIsSwapchain = targetIsSwapchain;
    cmd._staticTargetSize = staticTargetSize;
    this._brpCmdPoolUsed += 1;
    this.processor._passCommands.push(cmd);
    return true;
  }

  // ---- Bind groups ----

  _dummyResource(layoutEntry, ownerKey = '') {
    if (layoutEntry.buffer) {
      const rawMin =
        typeof layoutEntry.buffer.minBindingSize === 'number' &&
        Number.isFinite(layoutEntry.buffer.minBindingSize) &&
        layoutEntry.buffer.minBindingSize > 0
          ? layoutEntry.buffer.minBindingSize
          : 0;
      const bufType = layoutEntry.buffer.type || 'uniform';
      const isUniform = bufType === 'uniform';
      const isWritableStorage = bufType === 'storage';
      const maxBindingSize = isUniform
        ? getDeviceLimit(this.device, 'maxUniformBufferBindingSize', DEFAULT_MAX_UNIFORM_BUFFER_BINDING_SIZE)
        : getDeviceLimit(this.device, 'maxStorageBufferBindingSize', DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE);
      const size = Math.min(maxBindingSize, Math.max(256, Math.ceil(rawMin / 4) * 4));
      // #1977/#1892 parity: split writable 'storage' dummy buffers from read-only/uniform dummy
      // buffers so null RT arena slots (17..19) or SM 6.6 UAVs never alias read-only/uniform
      // bindings in the same pass (which Dawn rejects as a buffer usage-scope conflict).
      const accessTag = isWritableStorage ? 'rw' : 'ro';
      const key = `buffer:${accessTag}:${size}`;
      if (!this.dummies.has(key)) {
        this.dummies.set(key, this.device.createBuffer({ size, usage: GPU_BUFFER_USAGE_UNIFORM_STORAGE_COPY_DST }));
      }
      return { buffer: this.dummies.get(key) };
    }
    if (layoutEntry.sampler) {
      const samplerType = layoutEntry.sampler.type || 'filtering';
      const key = `sampler:${samplerType}`;
      if (!this.dummies.has(key)) {
        const desc = samplerType === 'comparison' ? { compare: 'always' } : {};
        this.dummies.set(key, this.device.createSampler(desc));
      }
      return this.dummies.get(key);
    }
    const tex = layoutEntry.texture || layoutEntry.storageTexture;
    let viewDimension = (tex && tex.viewDimension) || '2d';
    let format = tex && tex.format;
    if (!format) {
      format =
        { float: 'rgba8unorm', 'unfilterable-float': 'r32float', depth: 'depth32float', sint: 'rgba8sint', uint: 'rgba8uint' }[
          tex && tex.sampleType
        ] || 'rgba8unorm';
    }
    if (layoutEntry.storageTexture) {
      const resolvedStorage = this._resolveStorageFormat(format);
      format =
        resolvedStorage.supportsStorageBinding && resolvedStorage.storageFormat
          ? resolvedStorage.storageFormat
          : 'rgba8unorm';
      if (viewDimension === 'cube' || viewDimension === 'cube-array') {
        viewDimension = '2d-array';
      }
    }
    const multisampled = Boolean(layoutEntry.texture && layoutEntry.texture.multisampled);
    if (multisampled) {
      viewDimension = '2d';
      if (!isMultisampleSupported(format, this.device)) {
        format = 'rgba8unorm';
      }
    }
    // #1953/#1977/#1892 parity: give distinct storageTexture bindings separate dummy GPUTextures
    // keyed by ownerKey (layoutId:binding) so 2+ null UAVs never trigger Dawn's "writable storage
    // texture used twice" validation error.
    const isStorage = Boolean(layoutEntry.storageTexture);
    const storageSlotSuffix = isStorage && ownerKey ? `:${ownerKey}` : '';
    const key = `texture:${format}:${viewDimension}:${multisampled}:${isStorage}${storageSlotSuffix}`;
    if (!this.dummies.has(key)) {
      if (this.dummies.size >= 2048) {
        const oldestKey = this.dummies.keys().next().value;
        this.dummies.delete(oldestKey);
      }
      const layers = viewDimension === 'cube' || viewDimension === 'cube-array' ? 6 : 1;
      const requestedDim = viewDimension === '1d' ? '1d' : viewDimension === '3d' ? '3d' : '2d';
      const effectiveSampleCount = multisampled && requestedDim === '2d' && layers === 1 ? 4 : 1;
      // WebGPU requires RENDER_ATTACHMENT usage on all multisampled textures.
      const dummyUsage = isStorage
        ? GPU_TEXTURE_USAGE_STORAGE_BINDING
        : effectiveSampleCount > 1
          ? GPU_TEXTURE_USAGE_TEXTURE_BINDING | GPU_TEXTURE_USAGE_RENDER_ATTACHMENT
          : GPU_TEXTURE_USAGE_TEXTURE_BINDING;
      const texture = this.device.createTexture({
        size: { width: 1, height: 1, depthOrArrayLayers: layers },
        format,
        usage: dummyUsage,
        dimension: requestedDim,
        sampleCount: effectiveSampleCount,
      });
      const actualDummyDim = typeof texture.dimension === 'string' ? texture.dimension : requestedDim;
      const effectiveViewDim =
        viewDimension === '1d' && actualDummyDim === '2d'
          ? '2d'
          : actualDummyDim === '1d' && viewDimension !== '1d'
            ? '1d'
            : viewDimension;
      this.dummies.set(key, texture.createView({ dimension: effectiveViewDim }));
    }
    return this.dummies.get(key);
  }

  _fallbackResourceForLayoutEntry(layoutEntry, countSamplerFallback = true, ownerKey = '') {
    if (!layoutEntry) return null;
    if (layoutEntry.sampler) {
      const samplerKind = layoutEntry.sampler.type || 'filtering';
      const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
      if (fallbackSampler) {
        if (countSamplerFallback) this.telemetry.fallbackSamplersBound += 1;
        return fallbackSampler;
      }
    }
    return this._dummyResource(layoutEntry, ownerKey);
  }

  _buildAllDummyEntriesForLayout(layoutEntries, layoutId = 0) {
    const fallbackEntries = [];
    for (let i = 0; i < layoutEntries.length; i += 1) {
      const le = layoutEntries[i];
      if (!le) continue;
      if (le.bindingArraySize >= 1) {
        for (let ai = 0; ai < le.bindingArraySize; ai += 1) {
          const slot = le.binding + ai;
          fallbackEntries.push({
            binding: slot,
            resource: this._fallbackResourceForLayoutEntry(le, false, `${layoutId}:${slot}`),
          });
        }
      } else {
        fallbackEntries.push({
          binding: le.binding,
          resource: this._fallbackResourceForLayoutEntry(le, false, `${layoutId}:${le.binding}`),
        });
      }
    }
    return fallbackEntries;
  }

  _createBindGroup(view, offset, payloadBytes) {
    if (payloadBytes < 16) return false;
    const id = view.getUint32(offset, true);
    const layoutId = view.getUint32(offset + 4, true);
    const count = view.getUint32(offset + 8, true);
    const flags = view.getUint32(offset + 12, true);
    const isBindingArrayBg = (flags & BINDING_ARRAYS_BG_FLAG) !== 0;
    if (16 + count * 48 > payloadBytes) return false;
    const record = this.bindGroupLayouts.get(layoutId);
    if (!record) {
      this._packetError = `bind group ${id}: unknown bind group layout ${layoutId}`;
      return false;
    }
    const entries = [];
    const descriptorResources = [];
    const dynamicSlotMeta =
      record.dynamicOffsetIndices && record.dynamicOffsetIndices.size > 0
        ? new Array(record.dynamicOffsetIndices.size).fill(null)
        : null;
    const seenBindings = new Set();
    const entryByBinding = record.entryByBinding || null;
    for (let i = 0; i < count; i++) {
      const o = offset + 16 + i * 48;
      const binding = view.getUint32(o, true);
      const arrayIndex = isBindingArrayBg ? view.getUint16(o + 12, true) : 0;
      const effectiveBinding = isBindingArrayBg ? binding + arrayIndex : binding;
      const dummyOwner = `${layoutId}:${effectiveBinding}`;
      const type = view.getUint32(o + 4, true);
      const resourceId = view.getUint32(o + 8, true);
      const layoutEntry = entryByBinding
        ? entryByBinding.get(binding) || null
        : record.entries
          ? record.entries.find((e) => e.binding === binding) || null
          : null;
      if (!layoutEntry && Array.isArray(record.entries)) {
        this.telemetry.bindGroupValidationFallbacks += 1;
        this._recordTelemetryWarningOnce(
          `bg-extra-binding:${layoutId}:${effectiveBinding}`,
          `bind group ${id}: skipping binding ${effectiveBinding} not declared in layout ${layoutId}`
        );
        continue;
      }
      if (isBindingArrayBg && layoutEntry) {
        const maxArraySize = layoutEntry.bindingArraySize >= 1 ? layoutEntry.bindingArraySize : 1;
        if (arrayIndex >= maxArraySize) {
          this.telemetry.bindGroupValidationFallbacks += 1;
          this._recordTelemetryWarningOnce(
            `bg-array-oob:${layoutId}:${binding}:${arrayIndex}`,
            `bind group ${id}: skipping binding ${binding} arrayIndex ${arrayIndex} >= bindingArraySize ${maxArraySize}`
          );
          continue;
        }
      }
      if (seenBindings.has(effectiveBinding)) {
        this._packetError = `bind group ${id}: duplicate binding ${effectiveBinding}`;
        return false;
      }
      seenBindings.add(effectiveBinding);
      const layoutExpectsBuffer = Boolean(layoutEntry && layoutEntry.buffer);
      const layoutExpectsTexture = Boolean(
        layoutEntry && (layoutEntry.texture || layoutEntry.storageTexture)
      );
      const layoutExpectsSampler = Boolean(layoutEntry && layoutEntry.sampler);
      const kindMismatch =
        resourceId !== 0 &&
        Boolean(layoutEntry) &&
        ((layoutExpectsBuffer && type !== BIND_RESOURCE_TYPE.BUFFER) ||
          (layoutExpectsTexture && type !== BIND_RESOURCE_TYPE.TEXTURE_VIEW) ||
          (layoutExpectsSampler && type !== BIND_RESOURCE_TYPE.SAMPLER));

      let realResource = null;
      let entryResource = null;
      let entryUsedDummy = false;
      let bufferUsage = null;
      let bufferSizeKnown = null;
      let textureUsage = null;
      let textureFormat = null;
      let textureViewDimension = null;
      let textureMultisampled = null;
      let textureViewDescUsed = null;
      let samplerComparison = null;
      if (resourceId === 0) {
        if (layoutExpectsSampler || type === BIND_RESOURCE_TYPE.SAMPLER) {
          const samplerKind =
            (layoutEntry && layoutEntry.sampler && layoutEntry.sampler.type) || 'filtering';
          const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
          if (fallbackSampler) {
            entryResource = fallbackSampler;
            entryUsedDummy = true;
            samplerComparison = samplerKind === 'comparison';
            this.telemetry.fallbackSamplersBound += 1;
          } else if (layoutEntry) {
            entryResource = this._dummyResource(layoutEntry, dummyOwner);
            if (entryResource) entryUsedDummy = true;
          }
        } else {
          entryResource = layoutEntry ? this._dummyResource(layoutEntry, dummyOwner) : null;
          if (entryResource) entryUsedDummy = true;
        }
      } else if (type === BIND_RESOURCE_TYPE.BUFFER) {
        const buffer = this.processor.buffers.get(resourceId);
        const bufferLive = Boolean(buffer && buffer.destroyed !== true);
        if (bufferLive) {
          const bufMeta =
            this.processor &&
            this.processor._bufferMeta &&
            typeof this.processor._bufferMeta.get === 'function'
              ? this.processor._bufferMeta.get(resourceId)
              : null;
          if (typeof buffer.usage === 'number') {
            bufferUsage = buffer.usage;
          } else if (bufMeta && typeof bufMeta.usage === 'number') {
            bufferUsage = bufMeta.usage;
          } else if (buffer.desc && typeof buffer.desc.usage === 'number') {
            bufferUsage = buffer.desc.usage;
          }
          const bufOffset = Number(view.getBigUint64(o + 16, true));
          const requestedSizeBig = view.getBigUint64(o + 24, true);
          const requestedSize =
            requestedSizeBig === 0xffffffffffffffffn || requestedSizeBig >= BigInt(Number.MAX_SAFE_INTEGER)
              ? 0
              : Number(requestedSizeBig);
          const bufferSize =
            typeof buffer.size === 'number' && Number.isFinite(buffer.size) && buffer.size >= 0
              ? buffer.size
              : bufMeta && typeof bufMeta.byteSize === 'number' && Number.isFinite(bufMeta.byteSize) && bufMeta.byteSize >= 0
                ? bufMeta.byteSize
                : buffer.desc && typeof buffer.desc.size === 'number' && Number.isFinite(buffer.desc.size) && buffer.desc.size >= 0
                  ? buffer.desc.size
                  : null;
          bufferSizeKnown = bufferSize;
          const bufLayoutType = layoutExpectsBuffer ? layoutEntry.buffer.type || 'uniform' : null;
          const isStorageBufLayout =
            bufLayoutType === 'storage' || bufLayoutType === 'read-only-storage';
          const requiredBufferUsage =
            bufLayoutType === 'uniform'
              ? GPU_BUFFER_USAGE_UNIFORM
              : bufLayoutType
                ? GPU_BUFFER_USAGE_STORAGE
                : 0;
          const bufferUsageMismatch =
            layoutExpectsBuffer &&
            bufferUsage !== null &&
            requiredBufferUsage !== 0 &&
            (bufferUsage & requiredBufferUsage) === 0;
          const minBindingSize =
            layoutExpectsBuffer &&
            !bufferUsageMismatch &&
            typeof layoutEntry.buffer.minBindingSize === 'number' &&
            layoutEntry.buffer.minBindingSize > 0
              ? layoutEntry.buffer.minBindingSize
              : 0;
          const offsetAlignment =
            bufLayoutType === 'uniform'
              ? getDeviceLimit(
                  this.device,
                  'minUniformBufferOffsetAlignment',
                  DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT
                )
              : isStorageBufLayout
                ? getDeviceLimit(
                    this.device,
                    'minStorageBufferOffsetAlignment',
                    DEFAULT_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT
                  )
                : 1;
          const maxBindingSize =
            bufLayoutType === 'uniform'
              ? getDeviceLimit(
                  this.device,
                  'maxUniformBufferBindingSize',
                  DEFAULT_MAX_UNIFORM_BUFFER_BINDING_SIZE
                )
              : isStorageBufLayout
                ? getDeviceLimit(
                    this.device,
                    'maxStorageBufferBindingSize',
                    DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE
                  )
                : Infinity;

          if (
            !kindMismatch &&
            !bufferUsageMismatch &&
            layoutExpectsBuffer &&
            bufOffset > 0 &&
            offsetAlignment > 1 &&
            bufOffset % offsetAlignment !== 0
          ) {
            this.telemetry.bindGroupValidationFallbacks += 1;
            this._recordTelemetryWarningOnce(
              `bg-buffer-unaligned-offset:${layoutId}:${effectiveBinding}:${offsetAlignment}`,
              `bind group ${id}: binding ${effectiveBinding} offset ${bufOffset} not aligned to ${offsetAlignment}B; using dummy buffer`
            );
            entryResource = this._dummyResource(layoutEntry);
            entryUsedDummy = true;
          } else if (bufferSize !== null && bufOffset >= bufferSize) {
            if (layoutEntry) {
              this.telemetry.bindGroupValidationFallbacks += 1;
              this._recordTelemetryWarningOnce(
                `bg-buffer-oob:${layoutId}:${binding}`,
                `bind group ${id}: binding ${binding} offset ${bufOffset} exceeds buffer size ${bufferSize}; using dummy buffer`
              );
              entryResource = this._dummyResource(layoutEntry);
              entryUsedDummy = true;
            }
          } else {
            let effectiveSize = requestedSize;
            if (bufferSize !== null && effectiveSize > 0 && bufOffset + effectiveSize > bufferSize) {
              effectiveSize = bufferSize - bufOffset;
            }
            const remaining = bufferSize !== null ? bufferSize - bufOffset : null;
            let undersizedForLayout = false;
            if (!kindMismatch && !bufferUsageMismatch && layoutExpectsBuffer) {
              const rawBoundSize = effectiveSize > 0 ? effectiveSize : remaining;
              if (rawBoundSize !== null && rawBoundSize > maxBindingSize) {
                effectiveSize = isStorageBufLayout
                  ? Math.floor(maxBindingSize / 4) * 4
                  : maxBindingSize;
                this._recordTelemetryWarningOnce(
                  `bg-buffer-max-size:${layoutId}:${effectiveBinding}:${maxBindingSize}`,
                  `bind group ${id}: binding ${effectiveBinding} size ${rawBoundSize} clamped to maxBindingSize ${effectiveSize}`
                );
              }
              if (isStorageBufLayout) {
                const curStorageSize = effectiveSize > 0 ? effectiveSize : remaining;
                if (curStorageSize !== null && curStorageSize < 4) {
                  undersizedForLayout = true;
                  this._recordTelemetryWarningOnce(
                    `bg-storage-size-too-small:${layoutId}:${effectiveBinding}`,
                    `bind group ${id}: binding ${effectiveBinding} storage buffer size ${curStorageSize} < 4B; using dummy buffer`
                  );
                } else if (effectiveSize > 0 && effectiveSize % 4 !== 0) {
                  const alignedUp = Math.ceil(effectiveSize / 4) * 4;
                  if (
                    (remaining === null || alignedUp <= remaining) &&
                    alignedUp <= maxBindingSize
                  ) {
                    effectiveSize = alignedUp;
                  } else {
                    effectiveSize = Math.floor(effectiveSize / 4) * 4;
                  }
                } else if (effectiveSize === 0 && remaining !== null && remaining % 4 !== 0) {
                  effectiveSize = Math.floor(remaining / 4) * 4;
                }
              }
            }
            if (!undersizedForLayout && minBindingSize > 0) {
              const neededMin = isStorageBufLayout
                ? Math.ceil(minBindingSize / 4) * 4
                : minBindingSize;
              const boundSize = effectiveSize > 0 ? effectiveSize : remaining;
              if (boundSize !== null && boundSize < neededMin) {
                if (
                  remaining !== null &&
                  remaining >= neededMin &&
                  neededMin <= maxBindingSize
                ) {
                  effectiveSize = neededMin;
                  this._recordTelemetryWarningOnce(
                    `bg-buffer-widen:${layoutId}:${binding}:${neededMin}`,
                    `bind group ${id}: binding ${binding} size widened to layout minBindingSize ${neededMin}`
                  );
                } else if (layoutEntry) {
                  this._recordTelemetryWarningOnce(
                    `bg-buffer-undersized:${layoutId}:${binding}:${neededMin}`,
                    `bind group ${id}: binding ${binding} buffer smaller than minBindingSize ${neededMin}; using dummy buffer`
                  );
                  undersizedForLayout = true;
                }
              }
            }
            if (undersizedForLayout) {
              this.telemetry.bindGroupValidationFallbacks += 1;
              entryResource = this._dummyResource(layoutEntry);
              entryUsedDummy = true;
            } else {
              realResource = { buffer, offset: bufOffset };
              if (effectiveSize > 0) realResource.size = effectiveSize;
              if (kindMismatch) {
                this.telemetry.bindGroupKindMismatches += 1;
                this._recordTelemetryWarningOnce(
                  `bg-kind-mismatch:${layoutId}:${binding}:buffer`,
                  `bind group ${id}: binding ${binding} buffer resource mismatches layout entry; substituting dummy resource`
                );
                entryResource = this._fallbackResourceForLayoutEntry(layoutEntry, false);
                entryUsedDummy = true;
              } else if (bufferUsageMismatch) {
                this.telemetry.bindGroupValidationFallbacks += 1;
                this._recordTelemetryWarningOnce(
                  `bg-buffer-usage-mismatch:${layoutId}:${binding}:${bufLayoutType}`,
                  `bind group ${id}: binding ${binding} buffer usage 0x${bufferUsage.toString(16)} lacks required ${bufLayoutType} usage; using dummy buffer`
                );
                entryResource = this._dummyResource(layoutEntry);
                entryUsedDummy = true;
              } else {
                entryResource = realResource;
              }
            }
          }
        } else if (layoutEntry) {
          this.telemetry.missingResourceFallbacks += 1;
          this._recordTelemetryWarningOnce(
            `bg-missing-buffer:${layoutId}:${effectiveBinding}:${resourceId}`,
            `bind group ${id}: binding ${effectiveBinding} buffer ${resourceId} unavailable; substituting dummy resource`
          );
          entryResource = this._fallbackResourceForLayoutEntry(layoutEntry, true, dummyOwner);
          entryUsedDummy = true;
        }
      } else if (type === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
        const viewDesc = decodeTextureViewDesc(view, o + 32);
        const texFormat = this._resolveTextureFormat(resourceId, viewDesc);
        const texMeta = this.textureMeta.get(resourceId) || null;
        const texObj = this.processor.textures.get(resourceId) || null;
        const texLive = Boolean(
          texObj && texObj.destroyed !== true && typeof texObj.createView === 'function'
        );
        if (!texLive) {
          if (layoutEntry) {
            this.telemetry.missingResourceFallbacks += 1;
            this._recordTelemetryWarningOnce(
              `bg-missing-texture:${layoutId}:${effectiveBinding}:${resourceId}`,
              `bind group ${id}: binding ${effectiveBinding} texture ${resourceId} unavailable; substituting dummy resource`
            );
            entryResource = this._fallbackResourceForLayoutEntry(layoutEntry, true, dummyOwner);
            entryUsedDummy = true;
          }
        } else {
          if (texMeta && typeof texMeta.usage === 'number') {
            textureUsage = texMeta.usage;
          } else if (texObj && typeof texObj.usage === 'number') {
            textureUsage = texObj.usage;
          } else if (texObj && texObj.desc && typeof texObj.desc.usage === 'number') {
            textureUsage = texObj.desc.usage;
          }
          textureFormat = texFormat || null;
          const actualTexDim = this._resolveActualTextureDimension(texObj, texMeta);
          const isPromoted1DTo2D = Boolean(texMeta && texMeta.promoted1DTo2D === true);
          const baseLayers =
            (texMeta && typeof texMeta.depthOrArrayLayers === 'number' ? texMeta.depthOrArrayLayers : null) ||
            (texObj && typeof texObj.depthOrArrayLayers === 'number' ? texObj.depthOrArrayLayers : null) ||
            (texObj && texObj.desc && texObj.desc.size && typeof texObj.desc.size.depthOrArrayLayers === 'number'
              ? texObj.desc.size.depthOrArrayLayers
              : 1);
          if (viewDesc && typeof viewDesc.dimension === 'string' && viewDesc.dimension !== 'undefined') {
            if (viewDesc.dimension === '1d' && (actualTexDim === '2d' || isPromoted1DTo2D)) {
              textureViewDimension = '2d';
            } else if (actualTexDim === '1d' && viewDesc.dimension !== '1d') {
              textureViewDimension = '1d';
            } else {
              textureViewDimension = viewDesc.dimension;
            }
          } else {
            const baseDim = actualTexDim || '2d';
            textureViewDimension = baseDim === '2d' && baseLayers > 1 ? '2d-array' : baseDim;
          }
          const sampleCount =
            (texMeta && typeof texMeta.sampleCount === 'number' ? texMeta.sampleCount : null) ||
            (texObj && typeof texObj.sampleCount === 'number' ? texObj.sampleCount : null) ||
            (texObj && texObj.desc && typeof texObj.desc.sampleCount === 'number' ? texObj.desc.sampleCount : null);
          if (sampleCount !== null) {
            textureMultisampled = sampleCount > 1;
          }
          textureViewDescUsed = viewDesc;
          if (kindMismatch) {
            const resolvedView = this._textureView(resourceId, viewDesc);
            if (resolvedView) {
              realResource = resolvedView;
            } else {
              this._packetError = null;
            }
            this.telemetry.bindGroupKindMismatches += 1;
            this._recordTelemetryWarningOnce(
              `bg-kind-mismatch:${layoutId}:${binding}:texture`,
              `bind group ${id}: binding ${binding} texture resource mismatches layout entry; substituting dummy resource`
            );
            entryResource = this._fallbackResourceForLayoutEntry(layoutEntry, false);
            entryUsedDummy = true;
          } else if (layoutEntry && layoutEntry.storageTexture) {
            const expectedStorageFormat = layoutEntry.storageTexture.format || null;
            const expectedStorageDim = layoutEntry.storageTexture.viewDimension || '2d';
            const expectedStorageAccess = layoutEntry.storageTexture.access || 'write-only';
            const resolvedStorage = texFormat ? this._resolveStorageFormat(texFormat) : null;
            if (resolvedStorage && (!resolvedStorage.supportsStorageBinding || !resolvedStorage.storageFormat)) {
              this.telemetry.bindGroupValidationFallbacks += 1;
              this._recordTelemetryWarningOnce(
                `bg-storage-format-unsupported:${texFormat}:${binding}`,
                `bind group ${id}: binding ${binding} storageTexture format ${texFormat} unsupported${
                  resolvedStorage.reason ? ` (${resolvedStorage.reason})` : ''
                }; substituting dummy storage texture`
              );
              entryResource = this._dummyResource(layoutEntry, dummyOwner);
              entryUsedDummy = true;
            } else if (textureMultisampled === true) {
              this.telemetry.bindGroupValidationFallbacks += 1;
              this._recordTelemetryWarningOnce(
                `bg-storage-multisampled:${layoutId}:${effectiveBinding}`,
                `bind group ${id}: binding ${effectiveBinding} multisampled texture cannot bind as storageTexture; substituting dummy storage texture`
              );
              entryResource = this._dummyResource(layoutEntry, dummyOwner);
              entryUsedDummy = true;
            } else if (
              textureUsage !== null &&
              (textureUsage & GPU_TEXTURE_USAGE_STORAGE_BINDING) === 0
            ) {
              this.telemetry.bindGroupValidationFallbacks += 1;
              this._recordTelemetryWarningOnce(
                `bg-storage-usage-mismatch:${layoutId}:${effectiveBinding}`,
                `bind group ${id}: binding ${effectiveBinding} texture usage 0x${textureUsage.toString(16)} lacks STORAGE_BINDING; substituting dummy storage texture`
              );
              entryResource = this._dummyResource(layoutEntry, dummyOwner);
              entryUsedDummy = true;
            } else {
              const effectiveStorageFormat = resolvedStorage ? resolvedStorage.storageFormat : texFormat;
              if (
                expectedStorageFormat &&
                effectiveStorageFormat &&
                effectiveStorageFormat !== expectedStorageFormat
              ) {
                this.telemetry.bindGroupValidationFallbacks += 1;
                this._recordTelemetryWarningOnce(
                  `bg-storage-format-mismatch:${layoutId}:${effectiveBinding}:${effectiveStorageFormat}:${expectedStorageFormat}`,
                  `bind group ${id}: binding ${effectiveBinding} storageTexture format ${effectiveStorageFormat} mismatches layout ${expectedStorageFormat}; substituting dummy storage texture`
                );
                entryResource = this._dummyResource(layoutEntry, dummyOwner);
                entryUsedDummy = true;
              } else if (
                effectiveStorageFormat &&
                !isStorageAccessCompatible(effectiveStorageFormat, expectedStorageAccess, this.device)
              ) {
                this.telemetry.bindGroupValidationFallbacks += 1;
                this._recordTelemetryWarningOnce(
                  `bg-storage-access-unsupported:${layoutId}:${effectiveBinding}:${effectiveStorageFormat}:${expectedStorageAccess}`,
                  `bind group ${id}: binding ${effectiveBinding} storageTexture format ${effectiveStorageFormat} unsupported for ${expectedStorageAccess}; substituting dummy storage texture`
                );
                entryResource = this._dummyResource(layoutEntry, dummyOwner);
                entryUsedDummy = true;
              } else {
                let storageViewDesc =
                  resolvedStorage && resolvedStorage.needsLinearSiblingView
                    ? { ...(viewDesc || {}), format: resolvedStorage.storageFormat, mipLevelCount: 1 }
                    : viewDesc && viewDesc.mipLevelCount === 1
                      ? viewDesc
                      : { ...(viewDesc || {}), mipLevelCount: 1 };
                const texDimForCheck = actualTexDim || '2d';
                let dimCompatible =
                  texDimForCheck === '1d'
                    ? expectedStorageDim === '1d'
                    : texDimForCheck === '3d'
                      ? expectedStorageDim === '3d'
                      : expectedStorageDim === '2d' || expectedStorageDim === '2d-array';
                if (dimCompatible && textureViewDimension && textureViewDimension !== expectedStorageDim) {
                  if (texDimForCheck === '2d' && expectedStorageDim === '2d') {
                    storageViewDesc = { ...storageViewDesc, dimension: '2d', arrayLayerCount: 1 };
                    textureViewDimension = '2d';
                  } else if (texDimForCheck === '2d' && expectedStorageDim === '2d-array') {
                    storageViewDesc = { ...storageViewDesc, dimension: '2d-array' };
                    textureViewDimension = '2d-array';
                  } else {
                    dimCompatible = false;
                  }
                }
                if (!dimCompatible) {
                  this.telemetry.bindGroupValidationFallbacks += 1;
                  this._recordTelemetryWarningOnce(
                    `bg-storage-dim-mismatch:${layoutId}:${effectiveBinding}:${textureViewDimension}:${expectedStorageDim}`,
                    `bind group ${id}: binding ${effectiveBinding} storageTexture dimension ${textureViewDimension} mismatches layout ${expectedStorageDim}; substituting dummy storage texture`
                  );
                  entryResource = this._dummyResource(layoutEntry, dummyOwner);
                  entryUsedDummy = true;
                } else {
                  const resolvedView = this._textureView(resourceId, storageViewDesc);
                  if (resolvedView) {
                    realResource = resolvedView;
                    entryResource = resolvedView;
                    if (resolvedStorage && resolvedStorage.needsLinearSiblingView) {
                      textureFormat = resolvedStorage.storageFormat;
                    }
                    textureViewDescUsed = storageViewDesc;
                  } else {
                    this._packetError = null;
                    this.telemetry.bindGroupValidationFallbacks += 1;
                    this._recordTelemetryWarningOnce(
                      resolvedStorage && resolvedStorage.needsLinearSiblingView
                        ? `bg-storage-linear-view-fallback:${texFormat}:${binding}`
                        : `bg-storage-view-fallback:${layoutId}:${effectiveBinding}:${resourceId}`,
                      resolvedStorage && resolvedStorage.needsLinearSiblingView
                        ? `bind group ${id}: binding ${binding} linear sibling view ${resolvedStorage.storageFormat} unavailable for ${texFormat}; substituting dummy storage texture`
                        : `bind group ${id}: binding ${effectiveBinding} storageTexture view unavailable for texture ${resourceId}; substituting dummy storage texture`
                    );
                    entryResource = this._dummyResource(layoutEntry, dummyOwner);
                    entryUsedDummy = true;
                  }
                }
              }
            }
          } else if (layoutEntry && layoutEntry.texture) {
            const expectedSampleType = layoutEntry.texture.sampleType || 'float';
            const expectedDim = layoutEntry.texture.viewDimension || '2d';
            const expectedMultisampled = Boolean(layoutEntry.texture.multisampled);
            let sampledViewDesc = viewDesc;
            let effectiveAspect = (viewDesc && viewDesc.aspect) || 'all';
            if (texFormat && isCombinedDepthStencilFormat(texFormat) && effectiveAspect === 'all') {
              effectiveAspect = expectedSampleType === 'uint' ? 'stencil-only' : 'depth-only';
              sampledViewDesc = { ...(sampledViewDesc || {}), aspect: effectiveAspect };
            }
            const effectiveTexFormat = resolveEffectiveViewFormat(texFormat, effectiveAspect);
            if (effectiveTexFormat) textureFormat = effectiveTexFormat;

            if (
              textureUsage !== null &&
              (textureUsage & GPU_TEXTURE_USAGE_TEXTURE_BINDING) === 0
            ) {
              this.telemetry.bindGroupValidationFallbacks += 1;
              this._recordTelemetryWarningOnce(
                `bg-texture-usage-mismatch:${layoutId}:${effectiveBinding}`,
                `bind group ${id}: binding ${effectiveBinding} texture usage 0x${textureUsage.toString(16)} lacks TEXTURE_BINDING; substituting dummy texture`
              );
              entryResource = this._dummyResource(layoutEntry);
              entryUsedDummy = true;
            } else if (
              textureMultisampled !== null &&
              textureMultisampled !== expectedMultisampled
            ) {
              this.telemetry.bindGroupValidationFallbacks += 1;
              this._recordTelemetryWarningOnce(
                `bg-texture-multisample-mismatch:${layoutId}:${effectiveBinding}:${textureMultisampled}:${expectedMultisampled}`,
                `bind group ${id}: binding ${effectiveBinding} texture multisampled=${textureMultisampled} mismatches layout multisampled=${expectedMultisampled}; substituting dummy texture`
              );
              entryResource = this._dummyResource(layoutEntry);
              entryUsedDummy = true;
            } else if (
              expectedSampleType === 'float' &&
              effectiveTexFormat &&
              isFloat32TextureFormat(effectiveTexFormat) &&
              !deviceHasFeature(this.device, 'float32-filterable')
            ) {
              this.telemetry.unfilterableFloatFallbacks += 1;
              this._recordTelemetryWarningOnce(
                `bg-unfilterable-float:${effectiveTexFormat}:${binding}`,
                `bind group ${id}: binding ${binding} format ${effectiveTexFormat} requires unfilterable-float without float32-filterable; substituting filterable fallback`
              );
              entryResource = this._dummyResource(layoutEntry);
              entryUsedDummy = true;
            } else if (
              effectiveTexFormat &&
              !isTextureSampleTypeCompatible(effectiveTexFormat, expectedSampleType, this.device)
            ) {
              this.telemetry.bindGroupValidationFallbacks += 1;
              this._recordTelemetryWarningOnce(
                `bg-sample-type-mismatch:${layoutId}:${effectiveBinding}:${effectiveTexFormat}:${expectedSampleType}`,
                `bind group ${id}: binding ${effectiveBinding} format ${effectiveTexFormat} incompatible with layout sampleType ${expectedSampleType}; substituting dummy texture`
              );
              entryResource = this._dummyResource(layoutEntry);
              entryUsedDummy = true;
            } else {
              const texDimForCheck = actualTexDim || '2d';
              let dimCompatible =
                texDimForCheck === '1d'
                  ? expectedDim === '1d'
                  : texDimForCheck === '3d'
                    ? expectedDim === '3d'
                    : expectedDim === '2d' ||
                      expectedDim === '2d-array' ||
                      expectedDim === 'cube' ||
                      expectedDim === 'cube-array';
              if (dimCompatible && textureViewDimension && textureViewDimension !== expectedDim) {
                if (texDimForCheck === '2d' && !expectedMultisampled) {
                  if (expectedDim === '2d' && textureViewDimension === '2d-array') {
                    sampledViewDesc = { ...(sampledViewDesc || {}), dimension: '2d', arrayLayerCount: 1 };
                    textureViewDimension = '2d';
                  } else if (expectedDim === '2d-array' && textureViewDimension === '2d') {
                    sampledViewDesc = { ...(sampledViewDesc || {}), dimension: '2d-array' };
                    textureViewDimension = '2d-array';
                  } else if (expectedDim === 'cube' && baseLayers >= 6) {
                    sampledViewDesc = {
                      ...(sampledViewDesc || {}),
                      dimension: 'cube',
                      baseArrayLayer: 0,
                      arrayLayerCount: 6,
                    };
                    textureViewDimension = 'cube';
                  } else if (expectedDim === 'cube-array' && baseLayers >= 6) {
                    const cubeLayers = Math.floor(baseLayers / 6) * 6;
                    sampledViewDesc = {
                      ...(sampledViewDesc || {}),
                      dimension: 'cube-array',
                      baseArrayLayer: 0,
                      arrayLayerCount: cubeLayers,
                    };
                    textureViewDimension = 'cube-array';
                  } else {
                    dimCompatible = false;
                  }
                } else {
                  dimCompatible = false;
                }
              }
              if (
                dimCompatible &&
                (expectedDim === 'cube' || expectedDim === 'cube-array')
              ) {
                const texWidth =
                  (texMeta && typeof texMeta.width === 'number' ? texMeta.width : null) ||
                  (texObj && typeof texObj.width === 'number' ? texObj.width : null) ||
                  (texObj && texObj.desc && texObj.desc.size && typeof texObj.desc.size.width === 'number'
                    ? texObj.desc.size.width
                    : null);
                const texHeight =
                  (texMeta && typeof texMeta.height === 'number' ? texMeta.height : null) ||
                  (texObj && typeof texObj.height === 'number' ? texObj.height : null) ||
                  (texObj && texObj.desc && texObj.desc.size && typeof texObj.desc.size.height === 'number'
                    ? texObj.desc.size.height
                    : null);
                const viewBaseLayer =
                  sampledViewDesc && typeof sampledViewDesc.baseArrayLayer === 'number'
                    ? sampledViewDesc.baseArrayLayer
                    : 0;
                const availLayers = Math.max(0, baseLayers - viewBaseLayer);
                const viewLayerCount =
                  sampledViewDesc && typeof sampledViewDesc.arrayLayerCount === 'number' && sampledViewDesc.arrayLayerCount > 0
                    ? sampledViewDesc.arrayLayerCount
                    : availLayers;
                if (
                  baseLayers < 6 ||
                  (texWidth !== null && texHeight !== null && texWidth !== texHeight) ||
                  (expectedDim === 'cube' && viewLayerCount !== 6) ||
                  (expectedDim === 'cube-array' && (viewLayerCount < 6 || viewLayerCount % 6 !== 0))
                ) {
                  dimCompatible = false;
                }
              }
              if (!dimCompatible) {
                this.telemetry.bindGroupValidationFallbacks += 1;
                this._recordTelemetryWarningOnce(
                  `bg-texture-dim-mismatch:${layoutId}:${effectiveBinding}:${textureViewDimension}:${expectedDim}`,
                  `bind group ${id}: binding ${effectiveBinding} texture viewDimension ${textureViewDimension} (layers=${baseLayers}) incompatible with layout ${expectedDim}; substituting dummy texture`
                );
                entryResource = this._dummyResource(layoutEntry);
                entryUsedDummy = true;
              } else {
                const resolvedView = this._textureView(resourceId, sampledViewDesc);
                if (resolvedView) {
                  realResource = resolvedView;
                  entryResource = resolvedView;
                  textureViewDescUsed = sampledViewDesc;
                } else {
                  this._packetError = null;
                  this.telemetry.bindGroupValidationFallbacks += 1;
                  this._recordTelemetryWarningOnce(
                    `bg-texture-view-fallback:${layoutId}:${effectiveBinding}:${resourceId}`,
                    `bind group ${id}: binding ${effectiveBinding} texture ${resourceId} createView failed; substituting dummy texture`
                  );
                  entryResource = this._dummyResource(layoutEntry);
                  entryUsedDummy = true;
                }
              }
            }
          } else {
            const resolvedView = this._textureView(resourceId, viewDesc);
            if (resolvedView) {
              realResource = resolvedView;
              entryResource = resolvedView;
            }
          }
        }
      } else if (type === BIND_RESOURCE_TYPE.SAMPLER) {
        const sampler = this.processor.samplers.get(resourceId) || null;
        const samplerLive = Boolean(sampler && sampler.destroyed !== true);
        if (samplerLive) {
          realResource = sampler;
          const sMeta = this.samplerMeta.get(resourceId) || null;
          let samplerFiltering = null;
          if (sMeta) {
            samplerComparison = sMeta.comparison;
            samplerFiltering = sMeta.filtering;
          } else if (sampler.desc) {
            samplerComparison = sampler.desc.compare !== undefined;
            samplerFiltering = isSamplerFilteringDesc(sampler.desc);
          }
          if (kindMismatch) {
            this.telemetry.bindGroupKindMismatches += 1;
            this._recordTelemetryWarningOnce(
              `bg-kind-mismatch:${layoutId}:${binding}:sampler`,
              `bind group ${id}: binding ${binding} sampler resource mismatches layout entry; substituting dummy resource`
            );
            entryResource = this._fallbackResourceForLayoutEntry(layoutEntry, false);
            entryUsedDummy = true;
          } else if (layoutExpectsSampler) {
            const expectedSamplerType = layoutEntry.sampler.type || 'filtering';
            const comparisonMismatch =
              (expectedSamplerType === 'comparison' && samplerComparison === false) ||
              (expectedSamplerType !== 'comparison' && samplerComparison === true);
            const filteringMismatch =
              expectedSamplerType === 'non-filtering' && samplerFiltering === true;
            if (comparisonMismatch || filteringMismatch) {
              this.telemetry.bindGroupValidationFallbacks += 1;
              this._recordTelemetryWarningOnce(
                `bg-sampler-type-mismatch:${layoutId}:${effectiveBinding}:${expectedSamplerType}`,
                `bind group ${id}: binding ${effectiveBinding} sampler ${resourceId} incompatible with layout type ${expectedSamplerType}; substituting default sampler`
              );
              entryResource = this._fallbackResourceForLayoutEntry(layoutEntry);
              entryUsedDummy = true;
            } else {
              entryResource = sampler;
            }
          } else {
            entryResource = sampler;
          }
        } else if (layoutEntry) {
          this.telemetry.missingResourceFallbacks += 1;
          if (!kindMismatch) {
            const samplerKind =
              (layoutEntry.sampler && layoutEntry.sampler.type) || 'filtering';
            const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
            if (fallbackSampler) {
              entryResource = fallbackSampler;
              samplerComparison = samplerKind === 'comparison';
              this.telemetry.fallbackSamplersBound += 1;
            } else {
              entryResource = this._dummyResource(layoutEntry);
              samplerComparison = samplerKind === 'comparison';
            }
            this._recordTelemetryWarningOnce(
              `bg-missing-sampler:${layoutId}:${effectiveBinding}:${resourceId}`,
              `bind group ${id}: binding ${effectiveBinding} sampler ${resourceId} unavailable; bound default ${samplerKind} sampler`
            );
          } else {
            entryResource = this._dummyResource(layoutEntry);
            this._recordTelemetryWarningOnce(
              `bg-missing-sampler:${layoutId}:${effectiveBinding}:${resourceId}`,
              `bind group ${id}: binding ${effectiveBinding} sampler ${resourceId} unavailable; substituting dummy resource`
            );
          }
          entryUsedDummy = true;
        } else if (!kindMismatch) {
          const fallbackSampler = this._resolveDefaultSamplerFromCache('filtering');
          if (fallbackSampler) {
            entryResource = fallbackSampler;
            entryUsedDummy = true;
            samplerComparison = false;
            this.telemetry.fallbackSamplersBound += 1;
            this.telemetry.missingResourceFallbacks += 1;
            this._recordTelemetryWarningOnce(
              `bg-missing-sampler:${layoutId}:${effectiveBinding}:${resourceId}`,
              `bind group ${id}: binding ${effectiveBinding} sampler ${resourceId} unavailable; bound default filtering sampler`
            );
          }
        }
      }
      if (!entryResource && layoutEntry) {
        this.telemetry.missingResourceFallbacks += 1;
        this._recordTelemetryWarningOnce(
          `bg-missing-resource:${layoutId}:${effectiveBinding}:${resourceId}`,
          `bind group ${id}: binding ${effectiveBinding} resource ${resourceId} unavailable; substituting dummy resource`
        );
        entryResource = this._fallbackResourceForLayoutEntry(layoutEntry, true, dummyOwner);
        entryUsedDummy = true;
      }
      if (!entryResource) {
        this._packetError = `bind group ${id}: binding ${effectiveBinding} resource ${resourceId} unavailable`;
        return false;
      }
      entries.push({ binding: effectiveBinding, resource: entryResource });

      // #1941 parity: track dynamic-offset buffer metadata per dynamicOffsetIndex so SetBindGroup2
      // can validate alignment/bounds and zero offsets on dummy fallback buffers before Dawn sees them.
      if (dynamicSlotMeta && layoutExpectsBuffer && record.dynamicOffsetIndices) {
        const dynIdx = record.dynamicOffsetIndices.get(binding);
        if (dynIdx !== undefined) {
          const boundBaseOffset =
            !entryUsedDummy && entryResource && typeof entryResource.offset === 'number'
              ? entryResource.offset
              : 0;
          const boundSize =
            !entryUsedDummy && entryResource && typeof entryResource.size === 'number'
              ? entryResource.size
              : layoutEntry.buffer.minBindingSize || 0;
          dynamicSlotMeta[dynIdx] = {
            type: layoutEntry.buffer.type || 'uniform',
            baseOffset: boundBaseOffset,
            bindingSize: boundSize,
            bufferSize: entryUsedDummy ? null : bufferSizeKnown,
            isDummy: entryUsedDummy,
          };
        }
      }

      const descResource = realResource || entryResource;
      const descIsDummy = !realResource;
      let resourceKind = null;
      if (descIsDummy) {
        if (layoutEntry) {
          if (layoutEntry.buffer) {
            resourceKind = layoutEntry.buffer.type || 'uniform';
            bufferUsage = GPU_BUFFER_USAGE_UNIFORM_STORAGE_COPY_DST;
          } else if (layoutEntry.storageTexture) {
            resourceKind = 'storage-texture';
            textureUsage = GPU_TEXTURE_USAGE_STORAGE_BINDING;
          } else if (layoutEntry.texture) {
            resourceKind = 'texture';
            textureUsage = GPU_TEXTURE_USAGE_TEXTURE_BINDING;
          } else if (layoutEntry.sampler) {
            resourceKind = 'sampler';
            if (samplerComparison === null) {
              samplerComparison = layoutEntry.sampler.type === 'comparison';
            }
          }
        } else if (type === BIND_RESOURCE_TYPE.SAMPLER) {
          resourceKind = 'sampler';
          if (samplerComparison === null) {
            samplerComparison = false;
          }
        }
      } else if (type === BIND_RESOURCE_TYPE.BUFFER) {
        if (!kindMismatch && !entryUsedDummy && layoutEntry && layoutEntry.buffer) {
          resourceKind = layoutEntry.buffer.type || 'uniform';
        } else {
          resourceKind =
            bufferUsage !== null &&
            (bufferUsage & GPU_BUFFER_USAGE_STORAGE) !== 0 &&
            (bufferUsage & GPU_BUFFER_USAGE_UNIFORM) === 0
              ? 'storage'
              : 'uniform';
        }
      } else if (type === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
        if (!kindMismatch && !entryUsedDummy && layoutEntry && layoutEntry.storageTexture) {
          resourceKind = 'storage-texture';
        } else if (!kindMismatch && !entryUsedDummy && layoutEntry && layoutEntry.texture) {
          resourceKind = 'texture';
        } else {
          resourceKind =
            textureUsage !== null &&
            (textureUsage & GPU_TEXTURE_USAGE_STORAGE_BINDING) !== 0 &&
            (textureUsage & GPU_TEXTURE_USAGE_TEXTURE_BINDING) === 0
              ? 'storage-texture'
              : 'texture';
        }
      } else if (type === BIND_RESOURCE_TYPE.SAMPLER) {
        resourceKind = 'sampler';
        if (!kindMismatch && layoutEntry && layoutEntry.sampler && samplerComparison === null) {
          samplerComparison = layoutEntry.sampler.type === 'comparison';
        }
      }

      if (resourceKind) {
        const descEntry = { binding: effectiveBinding, kind: resourceKind, resource: descResource, isDummy: descIsDummy };
        if (bufferUsage !== null) descEntry.bufferUsage = bufferUsage;
        if (textureUsage !== null) descEntry.textureUsage = textureUsage;
        if (textureFormat !== null && !descIsDummy) descEntry.textureFormat = textureFormat;
        if (textureViewDimension !== null && !descIsDummy) descEntry.viewDimension = textureViewDimension;
        if (textureMultisampled !== null && !descIsDummy) descEntry.multisampled = textureMultisampled;
        if (type === BIND_RESOURCE_TYPE.TEXTURE_VIEW && !descIsDummy && resourceId !== 0) {
          descEntry.planeTextureId = resourceId;
          if (textureViewDescUsed) descEntry.viewDesc = textureViewDescUsed;
        }
        if (type === BIND_RESOURCE_TYPE.SAMPLER && !descIsDummy && resourceId !== 0) {
          descEntry.samplerId = resourceId;
        }
        if (samplerComparison !== null) descEntry.comparison = samplerComparison;
        if (type === BIND_RESOURCE_TYPE.BUFFER && !descIsDummy) {
          if (bufferSizeKnown !== null) descEntry.bufferSize = bufferSizeKnown;
          // W101: remember which setBindGroup dynamicOffsets slot (root-constant ring offset)
          // applies to this buffer so layout:'auto' remapping can fold it into the static offset.
          const dynIndex =
            record.dynamicOffsetIndices && layoutExpectsBuffer
              ? record.dynamicOffsetIndices.get(binding)
              : undefined;
          if (dynIndex !== undefined) {
            descEntry.hasDynamicOffset = true;
            descEntry.dynamicOffsetIndex = dynIndex;
          }
        }
        descriptorResources.push(descEntry);
      }
    }

    // W167/W186 + Orch-02: synthesize safe fallback entries for any layout bindings omitted from
    // CreateBindGroup2 (samplers, buffers, textures, storageTextures, and binding-array slots)
    // so device.createBindGroup never fails with entry count mismatches.
    if (Array.isArray(record.entries)) {
      for (let i = 0; i < record.entries.length; i += 1) {
        const le = record.entries[i];
        if (!le) continue;
        if (le.bindingArraySize >= 1) {
          for (let ai = 0; ai < le.bindingArraySize; ai += 1) {
            const slot = le.binding + ai;
            if (seenBindings.has(slot)) continue;
            seenBindings.add(slot);
            const fallbackRes = this._fallbackResourceForLayoutEntry(le, true, `${layoutId}:${slot}`);
            entries.push({ binding: slot, resource: fallbackRes });
            const kind = le.buffer
              ? le.buffer.type || 'uniform'
              : le.storageTexture
                ? 'storage-texture'
                : le.texture
                  ? 'texture'
                  : 'sampler';
            const descObj = { binding: slot, kind, resource: fallbackRes, isDummy: true };
            if (le.sampler) descObj.comparison = le.sampler.type === 'comparison';
            descriptorResources.push(descObj);
            this.telemetry.bindGroupValidationFallbacks += 1;
            this._recordTelemetryWarningOnce(
              `bg-omitted-array-slot:${layoutId}:${slot}`,
              `bind group ${id}: omitted binding-array slot ${slot}; synthesized dummy resource`
            );
          }
          continue;
        }
        if (seenBindings.has(le.binding)) continue;
        seenBindings.add(le.binding);
        if (le.sampler) {
          const samplerKind = le.sampler.type || 'filtering';
          const fallbackSampler = this._resolveDefaultSamplerFromCache(samplerKind);
          const samplerRes = fallbackSampler || this._dummyResource(le, `${layoutId}:${le.binding}`);
          entries.push({ binding: le.binding, resource: samplerRes });
          descriptorResources.push({
            binding: le.binding,
            kind: 'sampler',
            resource: samplerRes,
            isDummy: true,
            comparison: samplerKind === 'comparison',
          });
          if (fallbackSampler) {
            this.telemetry.fallbackSamplersBound += 1;
          } else {
            this.telemetry.bindGroupValidationFallbacks += 1;
          }
          this._recordTelemetryWarningOnce(
            `bg-omitted-sampler:${layoutId}:${le.binding}`,
            `bind group ${id}: omitted sampler binding ${le.binding}; bound default ${samplerKind} sampler`
          );
        } else {
          const fallbackRes = this._dummyResource(le, `${layoutId}:${le.binding}`);
          entries.push({ binding: le.binding, resource: fallbackRes });
          if (dynamicSlotMeta && le.buffer && record.dynamicOffsetIndices) {
            const dynIdx = record.dynamicOffsetIndices.get(le.binding);
            if (dynIdx !== undefined) {
              dynamicSlotMeta[dynIdx] = {
                type: le.buffer.type || 'uniform',
                baseOffset: 0,
                bindingSize: 0,
                bufferSize: null,
                isDummy: true,
              };
            }
          }
          const kind = le.buffer
            ? le.buffer.type || 'uniform'
            : le.storageTexture
              ? 'storage-texture'
              : 'texture';
          descriptorResources.push({
            binding: le.binding,
            kind,
            resource: fallbackRes,
            isDummy: true,
          });
          this.telemetry.bindGroupValidationFallbacks += 1;
          this._recordTelemetryWarningOnce(
            `bg-omitted-binding:${layoutId}:${le.binding}`,
            `bind group ${id}: omitted ${kind} binding ${le.binding}; synthesized dummy resource`
          );
        }
      }
    }

    // W186: hot-path PresentX multi-plane bind-group cache for 2-entry (2D float texture + sampler)
    // plane-blit bind groups, keyed by (planeTextureId, viewFormat, sampler, layout) and invalidated
    // on texture/sampler/layout/device changes.
    let presentxCacheKey = null;
    let presentxCacheMeta = null;
    if (
      !isBindingArrayBg &&
      !record.hasBindingArrays &&
      entries.length === 2 &&
      Array.isArray(record.entries) &&
      record.entries.length === 2 &&
      descriptorResources.length === 2
    ) {
      const texLayout = record.entries.find(
        (le) =>
          le &&
          le.texture &&
          !le.texture.multisampled &&
          le.texture.viewDimension === '2d' &&
          le.texture.sampleType === 'float'
      );
      const smpLayout = record.entries.find((le) => le && le.sampler);
      if (texLayout && smpLayout) {
        const texDesc = descriptorResources.find(
          (r) =>
            r &&
            r.binding === texLayout.binding &&
            r.kind === 'texture' &&
            !r.isDummy &&
            Number.isInteger(r.planeTextureId) &&
            r.planeTextureId > 0
        );
        const smpDesc = descriptorResources.find(
          (r) => r && r.binding === smpLayout.binding && r.kind === 'sampler' && r.resource
        );
        if (texDesc && smpDesc) {
          const liveTexture = this.processor.textures.get(texDesc.planeTextureId) || null;
          if (liveTexture) {
            const texGen = this._getTextureGeneration(texDesc.planeTextureId, liveTexture);
            const viewFormat = texDesc.textureFormat || 'default';
            const viewDescKey = buildTextureViewKey(texDesc.viewDesc);
            presentxCacheKey = `px:${texDesc.planeTextureId}:${texGen}:${viewFormat}:${viewDescKey}:s=${gpuObjectId(
              smpDesc.resource
            )}:p=0:l=${gpuObjectId(record.layout)}:b=${texDesc.binding},${smpDesc.binding}`;
            presentxCacheMeta = {
              view: texDesc.resource,
              planeTextureId: texDesc.planeTextureId,
              texture: liveTexture,
              textureGen: texGen,
              viewFormat,
              viewDescKey,
              samplerId: Number.isInteger(smpDesc.samplerId) ? smpDesc.samplerId : 0,
              sampler: smpDesc.resource,
              layoutId,
              layout: record.layout,
              pipelineId: 0,
              pipeline: null,
              device: this.device,
              deviceGeneration: this._cachedDeviceGeneration,
            };
          }
        }
      }
    }

    let bindGroup = null;
    if (presentxCacheKey && presentxCacheMeta) {
      const cached = this.presentxBindGroupCache.get(presentxCacheKey);
      if (
        cached &&
        cached.device === this.device &&
        cached.deviceGeneration === this._cachedDeviceGeneration &&
        cached.texture === presentxCacheMeta.texture &&
        cached.texture.destroyed !== true &&
        cached.view &&
        cached.view.destroyed !== true &&
        cached.bindGroup &&
        cached.bindGroup.destroyed !== true &&
        cached.textureGen === presentxCacheMeta.textureGen &&
        cached.sampler === presentxCacheMeta.sampler &&
        cached.sampler.destroyed !== true &&
        cached.layout === presentxCacheMeta.layout
      ) {
        this._cachePresentXBindGroup(presentxCacheKey, cached);
        this.telemetry.presentxBindGroupCacheHits += 1;
        bindGroup = cached.bindGroup;
      }
    }

    if (!bindGroup) {
      try {
        bindGroup = this.device.createBindGroup({ layout: record.layout, entries });
      } catch (err) {
        if (Array.isArray(record.entries)) {
          presentxCacheKey = null;
          try {
            const fallbackEntries = this._buildAllDummyEntriesForLayout(record.entries, layoutId);
            bindGroup = this.device.createBindGroup({ layout: record.layout, entries: fallbackEntries });
            if (dynamicSlotMeta) {
              for (let di = 0; di < dynamicSlotMeta.length; di += 1) {
                if (dynamicSlotMeta[di]) dynamicSlotMeta[di].isDummy = true;
              }
            }
            this.telemetry.bindGroupValidationFallbacks += 1;
            this._recordTelemetryWarningOnce(
              `bg-create-retry-dummy:${layoutId}`,
              `bind group ${id}: createBindGroup threw (${String(err && err.message ? err.message : err)}); retried with all-dummy entries`
            );
          } catch (err2) {
            if (
              this.processor &&
              typeof this.processor._maybeSignalDeviceLostFromError === 'function'
            ) {
              this.processor._maybeSignalDeviceLostFromError(err2, 'createBindGroup');
            }
            this._packetError = `bind group ${id}: ${String(err2 && err2.message ? err2.message : err2)}`;
            return false;
          }
        } else {
          if (
            this.processor &&
            typeof this.processor._maybeSignalDeviceLostFromError === 'function'
          ) {
            this.processor._maybeSignalDeviceLostFromError(err, 'createBindGroup');
          }
          this._packetError = `bind group ${id}: ${String(err && err.message ? err.message : err)}`;
          return false;
        }
      }
      if (presentxCacheKey && presentxCacheMeta) {
        this.telemetry.presentxBindGroupCacheMisses += 1;
        this._cachePresentXBindGroup(presentxCacheKey, {
          ...presentxCacheMeta,
          bindGroup,
        });
      }
    }
    this.processor.bindGroups.set(id, bindGroup);
    this.bindGroupDescriptors.set(id, {
      id,
      layoutId,
      resources: descriptorResources,
      hasRtArenaBindings: Boolean(record.hasRtArenaBindings),
      dynamicSlotMeta,
    });
    if (isBindingArrayBg) this.telemetry.bindingArrayBindGroups += 1;
    this._invalidateRemappedBindGroupsForBindGroup(id);
    return true;
  }

  _sanitizeExplicitDynamicOffsets(bindGroupId, bindGroupDesc, rawOffsets) {
    const slotMetaList = bindGroupDesc && bindGroupDesc.dynamicSlotMeta ? bindGroupDesc.dynamicSlotMeta : null;
    if (!slotMetaList || !rawOffsets || rawOffsets.length === 0) return rawOffsets;
    const len = rawOffsets.length;
    const uniformAlign = getDeviceLimit(
      this.device,
      'minUniformBufferOffsetAlignment',
      DEFAULT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT
    );
    const storageAlign = getDeviceLimit(
      this.device,
      'minStorageBufferOffsetAlignment',
      DEFAULT_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT
    );
    let out = null;
    for (let i = 0; i < len; i += 1) {
      const slot = i < slotMetaList.length ? slotMetaList[i] : null;
      if (!slot) continue;
      const rawOff = rawOffsets[i];
      if (slot.isDummy) {
        if (rawOff !== 0) {
          if (!out) out = rawOffsets.slice();
          out[i] = 0;
          this.telemetry.dynamicOffsetClamps += 1;
          this._recordTelemetryWarningOnce(
            `sbg-dynoffset-dummy:${bindGroupDesc.layoutId}:${i}`,
            `SetBindGroup2: zeroed dynamicOffset[${i}]=${rawOff} for dummy fallback buffer in bind group ${bindGroupId}`
          );
        }
        continue;
      }
      const align = slot.type === 'uniform' ? uniformAlign : storageAlign;
      let effOff = rawOff;
      let adjusted = false;
      if (align > 1 && effOff % align !== 0) {
        effOff = Math.floor(effOff / align) * align;
        adjusted = true;
      }
      if (
        slot.bufferSize !== null &&
        slot.baseOffset + effOff + slot.bindingSize > slot.bufferSize
      ) {
        const maxAllowed = slot.bufferSize - slot.baseOffset - slot.bindingSize;
        effOff = maxAllowed > 0 && align > 0 ? Math.floor(maxAllowed / align) * align : 0;
        adjusted = true;
      }
      if (adjusted) {
        if (!out) out = rawOffsets.slice();
        out[i] = effOff;
        this.telemetry.dynamicOffsetClamps += 1;
        this._recordTelemetryWarningOnce(
          `sbg-dynoffset-clamp:${bindGroupDesc.layoutId}:${i}:${align}`,
          `SetBindGroup2: clamped/aligned dynamicOffset[${i}] from ${rawOff} to ${effOff} (align=${align})`
        );
      }
    }
    return out || rawOffsets;
  }

  _setBindGroup(view, offset, payloadBytes) {
    if (payloadBytes < 16) return false;
    const index = view.getUint32(offset, true);
    const bindGroupId = view.getUint32(offset + 4, true);
    const count = view.getUint32(offset + 8, true);
    if (16 + count * 4 > payloadBytes) return false;
    // Resolved now: a later DestroyResource (bind group retired by the runtime) must not affect it.
    const bindGroup = this.processor.bindGroups.get(bindGroupId);
    if (!bindGroup) {
      // Mark this group slot invalid in the pass command stream so _preparePassCommandsForFlush
      // clears any stale prior binding and skips subsequent draws/dispatches on an unbound group.
      this.processor._passCommands.push({
        type: 'renderState',
        _rsInvalidBindGroup: { groupIndex: index, bindGroupId },
        run() {},
      });
      this.telemetry.unknownBindGroupErrors += 1;
      const replayEvictions =
        this.processor && typeof this.processor._replayBindGroupEvictions === 'number'
          ? this.processor._replayBindGroupEvictions
          : 0;
      const replayOverflows =
        this.processor && typeof this.processor._replayLogOverflowCount === 'number'
          ? this.processor._replayLogOverflowCount
          : 0;
      if (replayEvictions > 0 || replayOverflows > 0) {
        this._recordTelemetryWarningOnce(
          `sbg-unknown-after-replay-pressure:${this._cachedDeviceGeneration}`,
          `SetBindGroup2: unknown bind group ${bindGroupId} observed with replay pressure (bindGroupEvictions=${replayEvictions}, logOverflows=${replayOverflows})`
        );
      }
      this._packetError = `unknown bind group ${bindGroupId}`;
      return false;
    }
    if (this.processor && typeof this.processor._touchReplayBindGroup === 'function') {
      this.processor._touchReplayBindGroup(bindGroupId);
    }
    const bindGroupDesc = this.bindGroupDescriptors.get(bindGroupId) || null;
    const layoutRec =
      bindGroupDesc && bindGroupDesc.layoutId
        ? this.bindGroupLayouts.get(bindGroupDesc.layoutId) || null
        : null;
    const expectedDynCount =
      layoutRec && layoutRec.dynamicOffsetIndices ? layoutRec.dynamicOffsetIndices.size : null;
    const effectiveCount = expectedDynCount !== null ? expectedDynCount : count;
    if (expectedDynCount !== null && count !== expectedDynCount) {
      this._recordTelemetryWarningOnce(
        `sbg-dynoffset-count:${bindGroupDesc.layoutId}:${count}:${expectedDynCount}`,
        `SetBindGroup2: bind group ${bindGroupId} dynamicOffset count ${count} normalized to layout expected ${expectedDynCount}`
      );
    }
    let offsets;
    if (effectiveCount === 0) {
      offsets = EMPTY_DYNAMIC_OFFSETS;
    } else {
      offsets = this._acquireDynamicOffsetsArray(effectiveCount);
      const copyLen = Math.min(count, effectiveCount);
      for (let i = 0; i < copyLen; i += 1) {
        offsets[i] = view.getUint32(offset + 16 + i * 4, true);
      }
      for (let i = copyLen; i < effectiveCount; i += 1) {
        offsets[i] = 0;
      }
    }
    if (index === RT_ARENA_GROUP_INDEX && bindGroupDesc && bindGroupDesc.hasRtArenaBindings) {
      this.telemetry.rtArenaGroup3Binds += 1;
    }
    this.processor._passCommands.push(
      this._acquireSetBindGroup2Command(index, bindGroupId, bindGroup, bindGroupDesc, offsets)
    );
    return true;
  }

  _createSampler(view, offset) {
    const decoded = decodeSampler2Packet(view, offset);
    if (!decoded) return false;
    const id = decoded.id;
    const desc = sanitizeSamplerDescriptor(decoded.descriptor);
    let sampler;
    try {
      sampler = this.device.createSampler(desc);
    } catch (err) {
      this._packetError = `sampler ${id}: ${String(err && err.message ? err.message : err)}`;
      return false;
    }
    this.processor.samplers.set(id, sampler);
    this.samplerMeta.set(id, {
      comparison: desc.compare !== undefined,
      filtering: isSamplerFilteringDesc(desc),
    });
    this._invalidatePresentXCacheForResource(id);
    return true;
  }
}

function installRenderStatePackets(processor, options = {}) {
  const handler = new RenderStatePackets(processor, options);
  processor.extension = handler;

  if (processor && typeof processor === 'object') {
    processor[HOOK_ACTIVE_HANDLER] = handler;

    if (typeof processor._flushPresent === 'function' && !processor[HOOK_ORIG_FLUSH_PRESENT]) {
      const origFlushPresent = processor._flushPresent;
      processor[HOOK_ORIG_FLUSH_PRESENT] = origFlushPresent;
      processor._flushPresent = function hardenedFlushPresent() {
        const activeHandler = this[HOOK_ACTIVE_HANDLER];
        if (activeHandler && typeof activeHandler._preparePassCommandsForFlush === 'function') {
          activeHandler._preparePassCommandsForFlush();
        }
        return origFlushPresent.apply(this, arguments);
      };
    }

    if (
      typeof processor._handleCreateComputePipeline === 'function' &&
      !processor[HOOK_ORIG_CREATE_COMPUTE_PIPELINE]
    ) {
      const origCreateComputePipeline = processor._handleCreateComputePipeline;
      processor[HOOK_ORIG_CREATE_COMPUTE_PIPELINE] = origCreateComputePipeline;
      processor._handleCreateComputePipeline = function hardenedHandleCreateComputePipeline(
        view,
        payloadOffset
      ) {
        const activeHandler = this[HOOK_ACTIVE_HANDLER];
        if (activeHandler && typeof activeHandler._handleV1CreateComputePipeline === 'function') {
          activeHandler._handleV1CreateComputePipeline(origCreateComputePipeline, view, payloadOffset);
          return;
        }
        origCreateComputePipeline.call(this, view, payloadOffset);
      };
    }
  }

  return handler;
}

const exportedApi = {
  RenderStatePackets,
  installRenderStatePackets,
  createRenderStateScratch,
  decodeViewportPacket,
  decodeScissorPacket,
  decodeBlendConstantsPacket,
  decodeStencilRefPacket,
  decodePrimitiveTopologyPacket,
  decodeBlendComponent,
  decodeStencilFace,
  decodeSetBindGroup2Packet,
  decodeSampler2Packet,
  clampViewportToTarget,
  clampScissorToTarget,
  decodeRenderPipelineKey,
  decodeBeginRenderPass,
  decodeTextureViewDesc,
  decodeBindGroupLayoutEntry,
  decodeCreateTexture2,
  sanitizeTextureDescriptor,
  sanitizeColorTargetBlendState,
  sanitizeRenderPassAttachments,
  sanitizeSamplerDescriptor,
  validateIndirectCommand,
  validateComputeDispatch,
  getMaxComputeWorkgroupsPerDimension,
  deviceHasFeature,
  isBcCompressedFormat,
  isSrgbTextureFormat,
  getSrgbSiblingFormat,
  isViewFormatCompatible,
  resolveCompatibleViewFormats,
  hasDepthAspect,
  hasStencilAspect,
  isDepthOrStencilTextureFormat,
  isFloat32TextureFormat,
  isIntegerTextureFormat,
  isBlendableTextureFormat,
  isStorageBindingSupported,
  resolveStorageTextureFormat,
  isRenderAttachmentSupported,
  isMultisampleSupported,
  resolveTextureSampleType,
  resolveEffectiveViewFormat,
  isTextureSampleTypeCompatible,
  isStorageAccessCompatible,
  computeMaxMipLevels,
  resolveShaderEntryPoint,
  findWgslStageEntryPoints,
  parseWgslFragmentOutputLocations,
  parseWgslGroup0Bindings,
  isExplicitLayoutCompatibleWithWgsl,
  validateVertexBufferLayouts,
  GPU_BUFFER_USAGE_INDEX,
  GPU_BUFFER_USAGE_INDIRECT,
  DEFAULT_ENTRY_POINTS,
  DEPTH_TEXTURE_FORMATS,
  BLENDABLE_CORE_FORMATS,
  VERTEX_FORMAT_BYTE_SIZES,
  V2_OPCODES,
  OPCODE_CLEAR_TEXTURE,
  CLEAR_TEXTURE_PAYLOAD_BYTES,
  SWAPCHAIN_TEXTURE_ID,
  BINDING_ARRAYS_LAYOUT_FLAG,
  BINDING_ARRAYS_BG_FLAG,
  RT_ARENA_GROUP_INDEX,
  RT_ARENA_BINDING_MIN,
  RT_ARENA_BINDING_MAX,
  TEXTURE_FORMAT_NAMES,
  TEXTURE_DIMENSION_NAMES,
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
