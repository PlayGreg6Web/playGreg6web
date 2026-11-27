'use strict';

// Dedicated OffscreenCanvas WebGPU worker consuming binary GpuPacketHeader streams.
// Opcode contract: docs/gpu_commands.md & runtime/abi/playgta6_abi.h.
const OPCODES = Object.freeze({
  NOP: 0,
  BEGIN_FRAME: 1,
  CLEAR_COLOR: 2,
  CREATE_BUFFER: 3,
  WRITE_BUFFER: 4,
  CREATE_SHADER_MODULE: 5,
  DRAW: 6,
  PRESENT: 7,
  CREATE_RENDER_PIPELINE: 8,
  SET_PIPELINE: 9,
  SET_VERTEX_BUFFER: 10,
  DESTROY_RESOURCE: 11,
  CREATE_TEXTURE: 12,
  WRITE_TEXTURE: 13,
  CREATE_SAMPLER: 14,
  CREATE_BIND_GROUP: 15,
  SET_BIND_GROUP: 16,
  SET_INDEX_BUFFER: 17,
  DRAW_INDEXED: 18,
  CREATE_COMPUTE_PIPELINE: 19,
  SET_COMPUTE_PIPELINE: 20,
  DISPATCH_COMPUTE: 21,
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
  DRAW_INDIRECT: 36,
  DRAW_INDEXED_INDIRECT: 37,
  DISPATCH_INDIRECT: 38,
  READBACK_BUFFER: 39,
  READBACK_TEXTURE: 40,
});

// Native-executor extension opcode outside GpuOpcode (runtime/abi/playgta6_abi.h kGpuOpcodeClearTexture).
// Kept separate from OPCODES so tools/repo_health.py GpuOpcode enum parity stays exact.
const OPCODE_CLEAR_TEXTURE = 63;
const CLEAR_TEXTURE_PAYLOAD_BYTES = 8;
const CLEAR_TEXTURE_ROW_ALIGN = 256;
const CLEAR_TEXTURE_ZERO_BUFFER_MIN_BYTES = 64 * 1024;
const CLEAR_TEXTURE_ZERO_BUFFER_BYTES = 4 * 1024 * 1024;
const CLEAR_TEXTURE_ZERO_SCRATCH_MAX_BYTES = 4 * 1024 * 1024;
const CLEAR_TEXTURE_ZERO_COLOR = Object.freeze({ r: 0, g: 0, b: 0, a: 0 });
const CLEAR_TEXTURE_DEPTH_STENCIL_FORMATS = Object.freeze(
  new Set([
    'stencil8',
    'depth16unorm',
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
  ])
);
const CLEAR_TEXTURE_NON_RENDERABLE_FORMATS = Object.freeze(
  new Set(['r8snorm', 'rg8snorm', 'rgba8snorm', 'rgb9e5ufloat'])
);

const SHADER_STAGE = Object.freeze({
  VERTEX: 0,
  FRAGMENT: 1,
  COMPUTE: 2,
});

const PRIMITIVE_TOPOLOGY = Object.freeze({
  TRIANGLE_LIST: 0,
  TRIANGLE_STRIP: 1,
  LINE_LIST: 2,
  LINE_STRIP: 3,
  POINT_LIST: 4,
});

const TOPOLOGY_NAMES = Object.freeze([
  'triangle-list',
  'triangle-strip',
  'line-list',
  'line-strip',
  'point-list',
]);

const BUFFER_USAGE = Object.freeze({
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
});

const TEXTURE_USAGE = Object.freeze({
  COPY_SRC: 0x0001,
  COPY_DST: 0x0002,
  TEXTURE_BINDING: 0x0004,
  STORAGE_BINDING: 0x0008,
  RENDER_ATTACHMENT: 0x0010,
});

const TEXTURE_FORMAT = Object.freeze({
  RGBA8_UNORM: 0,
  BGRA8_UNORM: 1,
  RGBA16_FLOAT: 2,
  DEPTH24_PLUS: 3,
  DEPTH32_FLOAT: 4,
  BC7_RGBA_UNORM: 5,
});

const TEXTURE_FORMAT_NAMES = Object.freeze([
  'rgba8unorm',
  'bgra8unorm',
  'rgba16float',
  'depth24plus',
  'depth32float',
  'bc7-rgba-unorm',
]);

// ABI v2 CreateTexture2 WebGPU texture format codes (docs/gpu_commands.md).
const TEXTURE_FORMAT_NAMES_V2 = Object.freeze([
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

const TEXTURE_DIMENSION = Object.freeze({
  DIM_1D: 0,
  DIM_2D: 1,
  DIM_3D: 2,
});

const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);

const ADDRESS_MODE = Object.freeze({
  CLAMP_TO_EDGE: 0,
  REPEAT: 1,
  MIRROR_REPEAT: 2,
});

const ADDRESS_MODE_NAMES = Object.freeze(['clamp-to-edge', 'repeat', 'mirror-repeat']);

const FILTER_MODE = Object.freeze({
  NEAREST: 0,
  LINEAR: 1,
});

const FILTER_MODE_NAMES = Object.freeze(['nearest', 'linear']);

const BIND_RESOURCE_TYPE = Object.freeze({
  BUFFER: 0,
  TEXTURE_VIEW: 1,
  SAMPLER: 2,
});

const INDEX_FORMAT = Object.freeze({
  UINT16: 0,
  UINT32: 1,
});

const INDEX_FORMAT_NAMES = Object.freeze(['uint16', 'uint32']);

const DEFAULT_CLEAR_COLOR = Object.freeze({ r: 0.05, g: 0.07, b: 0.11, a: 1.0 });

const MAX_PASS_COMMAND_POOL_SIZE = 16384;
const WGPU_META_BEGIN_RENDER_PASS = Object.freeze({ kind: 'beginRenderPass' });
const WGPU_META_END_RENDER_PASS = Object.freeze({ kind: 'endRenderPass' });
const SENTINEL_BOUND_BIND_GROUP = Object.freeze({ bound: true });
const DRAIN_DEFERRED_OPTS_IMMEDIATE = Object.freeze({ immediate: true });
const DRAIN_DEFERRED_OPTS_DEFERRED = Object.freeze({ immediate: false });
const DRAIN_RING_REUSE_OPTS = Object.freeze({ reuseResult: true });
const DRAIN_RING_REUSE_COALESCE_OPTS = Object.freeze({ reuseResult: true, coalescePresents: true });
const DRAIN_RING_REUSE_NO_COALESCE_OPTS = Object.freeze({ reuseResult: true, coalescePresents: false });

function createPassCommandSlot() {
  return {
    type: '',
    pipelineId: 0,
    pipeline: null,
    slot: 0,
    bufferId: 0,
    offset: 0,
    indexFormat: 'uint16',
    groupIndex: 0,
    bindGroupId: 0,
    bindGroup: null,
    vertexCount: 0,
    instanceCount: 0,
    firstVertex: 0,
    firstInstance: 0,
    indexCount: 0,
    firstIndex: 0,
    baseVertex: 0,
    workgroupX: 0,
    workgroupY: 0,
    workgroupZ: 0,
    _wgpuMeta: null,
  };
}

function isPlainArrayBuffer(value) {
  if (!value || typeof value !== 'object') {
    return false;
  }
  if (
    typeof SharedArrayBuffer !== 'undefined' &&
    (value instanceof SharedArrayBuffer ||
      Object.prototype.toString.call(value) === '[object SharedArrayBuffer]')
  ) {
    return false;
  }
  return (
    value instanceof ArrayBuffer ||
    Object.prototype.toString.call(value) === '[object ArrayBuffer]'
  );
}

function readUint64AsNumber(view, offset) {
  const lo = view.getUint32(offset, true);
  const hi = view.getUint32(offset + 4, true);
  return hi === 0 ? lo : hi * 4294967296 + lo;
}

const DEFAULT_MAX_VALIDATION_ERROR_HISTORY = 32;
const DEFAULT_MAX_VALIDATION_ERROR_POSTS = 64;
const MAX_VALIDATION_ERROR_MESSAGE_LENGTH = 512;
const MAX_GPU_ADAPTER_FIELD_LENGTH = 96;
const MAX_DIRECT_WRITE_BUFFER_CHUNK_BYTES = 4 * 1024 * 1024;
const MAX_DIRECT_WRITE_TEXTURE_CHUNK_BYTES = 2 * 1024 * 1024;
const DEFAULT_MAX_DEFERRED_GPU_DESTROYS = 2048;
const DEFAULT_MAX_DEFERRED_GPU_DESTROY_BYTES = 256 * 1024 * 1024;
const MAX_SPECIALIZED_FALLBACK_PIPELINES = 512;
const MAX_SPECIALIZED_FALLBACK_BIND_GROUPS = 2048;
const MAX_VALIDATION_ERROR_DEDUP_ENTRIES = 1024;
const RECYCLE_CMD_BUFFER_MIN_BYTES = 256 * 1024;
const RECYCLE_CMD_BUFFER_MAX_BYTES = 32 * 1024 * 1024;
const DEFAULT_OVERSIZED_RT_RECLAIM_IDLE_FRAMES = 2;
const DEFAULT_OVERSIZED_RT_RECLAIM_INTERVAL_FRAMES = 2;
const MAX_ZERO_ALLOC_RING_DRAIN_BATCHES = 64;
const DEFAULT_MAX_WGSL_COMPILE_BYTES = 256 * 1024;
const DEFAULT_MAX_CONCURRENT_SHADER_COMPILES = 1;
const DEFAULT_SHADER_COMPILE_SLICE_BUDGET_MS = 8;
const DEFAULT_MAX_PENDING_RANGED_PIPELINES = 2048;
const DEFAULT_HEAVY_SHADER_YIELD_BYTES = 64 * 1024;
const DEFAULT_HEAVY_SHADER_YIELD_MS = 4;
const MAX_OVERSIZED_SHADER_HASHES = 8192;
const WGSL_GROUP0_RE = /@group\(\s*0\s*\)/;

// W338 / W350: Throttled non-black swapchain luminance probe constants.
// Samples an 8-strip horizontal grid (<= 64 px wide, bytesPerRow = 256 -> 2048 bytes total,
// < 0.02 ms) across y = [15%, 35%, 50%, 65%, 80%, 89%, 92%, 95%] spanning center, left,
// right, and bottom-right bands so it reliably detects full-screen video, center 6-star
// loading animation, sparse menus, AND the bottom-right OFFLINEPLAYER [A] bar.
const SWAPCHAIN_PROBE_DEFAULT_INTERVAL_MS = 250;
const SWAPCHAIN_PROBE_DEFAULT_TIMEOUT_MS = 5000;
const SWAPCHAIN_PROBE_STRIP_MAX_WIDTH = 64;
const SWAPCHAIN_PROBE_BYTES_PER_ROW = 256;
const SWAPCHAIN_PROBE_DEFAULT_LUMA_THRESHOLD = 8;
const SWAPCHAIN_PROBE_MAX_CONSECUTIVE_ERRORS = 3;
const DEFAULT_GPU_STALL_WARN_MS = 1000;
const DEFAULT_GPU_STALL_WARN_MIN_INTERVAL_MS = 1000;
const SWAPCHAIN_PROBE_STRIPS = Object.freeze([
  Object.freeze({ yFraction: 0.15, xFraction: 0.5 }),
  Object.freeze({ yFraction: 0.35, xFraction: 0.25 }),
  Object.freeze({ yFraction: 0.5, xFraction: 0.5 }),
  Object.freeze({ yFraction: 0.65, xFraction: 0.75 }),
  Object.freeze({ yFraction: 0.8, xFraction: 0.5 }),
  Object.freeze({ yFraction: 0.89, xFraction: 0.9 }),
  Object.freeze({ yFraction: 0.92, xFraction: 0.9 }),
  Object.freeze({ yFraction: 0.95, xFraction: 0.9 }),
]);
const SWAPCHAIN_PROBE_BUFFER_SIZE =
  SWAPCHAIN_PROBE_STRIPS.length * SWAPCHAIN_PROBE_BYTES_PER_ROW;

function resolveSwapchainCopySrcUsage() {
  if (
    typeof GPUTextureUsage !== 'undefined' &&
    typeof GPUTextureUsage.RENDER_ATTACHMENT === 'number' &&
    typeof GPUTextureUsage.COPY_SRC === 'number'
  ) {
    return {
      withCopySrc: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      renderOnly: GPUTextureUsage.RENDER_ATTACHMENT,
    };
  }
  return null;
}

const SWAPCHAIN_SRGB_TO_BASE = Object.freeze({
  'bgra8unorm-srgb': 'bgra8unorm',
  'rgba8unorm-srgb': 'rgba8unorm',
});

const TEXTURE_SRGB_TO_LINEAR = Object.freeze({
  'rgba8unorm-srgb': 'rgba8unorm',
  'bgra8unorm-srgb': 'bgra8unorm',
  'bc1-rgba-unorm-srgb': 'bc1-rgba-unorm',
  'bc2-rgba-unorm-srgb': 'bc2-rgba-unorm',
  'bc3-rgba-unorm-srgb': 'bc3-rgba-unorm',
  'bc7-rgba-unorm-srgb': 'bc7-rgba-unorm',
});

const SWAPCHAIN_BASE_TO_SRGB = Object.freeze({
  bgra8unorm: 'bgra8unorm-srgb',
  rgba8unorm: 'rgba8unorm-srgb',
});

const SWAPCHAIN_LINEAR_HDR_FORMATS = new Set(['rgba16float', 'rg11b10ufloat', 'rgba32float']);
const SWAPCHAIN_SDR_BASE_FORMATS = new Set(['bgra8unorm', 'rgba8unorm']);

function normalizeSwapchainAlphaMode(alphaMode) {
  if (typeof alphaMode === 'string') {
    const lower = alphaMode.trim().toLowerCase();
    if (lower === 'premultiplied' || lower === 'opaque') {
      return lower;
    }
  }
  return 'opaque';
}

function normalizeSwapchainToneMapMode(mode) {
  if (mode === undefined || mode === null || mode === '' || mode === 'auto') {
    return null;
  }
  if (typeof mode === 'string') {
    const lower = mode.trim().toLowerCase();
    if (lower === 'none' || lower === 'passthrough') {
      return 'none';
    }
    if (lower === 'reinhard' || lower === 'aces') {
      return lower;
    }
  }
  return null;
}

function normalizeSwapchainHdrMode(mode, defaultMode = 'auto') {
  if (typeof mode === 'string') {
    const lower = mode.trim().toLowerCase();
    if (lower === 'extended' || lower === 'sdr-tonemap' || lower === 'auto') {
      return lower;
    }
  }
  return defaultMode;
}

function normalizeSwapchainExposure(value, fallback = 1.0) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback;
}

function configureCanvasContextWithCopySrc(ctx, baseConfig) {
  if (!ctx || typeof ctx.configure !== 'function') {
    return { configured: false, copySrcEnabled: false };
  }
  const usageSpec = resolveSwapchainCopySrcUsage();
  if (!usageSpec || (baseConfig && baseConfig.usage !== undefined)) {
    ctx.configure(baseConfig);
    const hasCopySrc = Boolean(
      usageSpec &&
        baseConfig &&
        typeof baseConfig.usage === 'number' &&
        (baseConfig.usage & usageSpec.withCopySrc) === usageSpec.withCopySrc
    );
    return { configured: true, copySrcEnabled: hasCopySrc };
  }
  try {
    ctx.configure({ ...baseConfig, usage: usageSpec.withCopySrc });
    return { configured: true, copySrcEnabled: true };
  } catch (_) {
    ctx.configure({ ...baseConfig, usage: usageSpec.renderOnly });
    return { configured: true, copySrcEnabled: false };
  }
}

function configureCanvasContextWithViewFormats(ctx, baseConfig) {
  if (!ctx || typeof ctx.configure !== 'function') {
    return { configured: false, copySrcEnabled: false, viewFormatsApplied: false };
  }
  const cfg = baseConfig && typeof baseConfig === 'object' ? baseConfig : {};
  const explicitViews = Array.isArray(cfg.viewFormats) && cfg.viewFormats.length > 0;
  const srgbSibling =
    !explicitViews && typeof cfg.format === 'string' ? SWAPCHAIN_BASE_TO_SRGB[cfg.format] : null;
  if (explicitViews || srgbSibling) {
    const withViews = explicitViews ? cfg : { ...cfg, viewFormats: [srgbSibling] };
    try {
      const res = configureCanvasContextWithCopySrc(ctx, withViews);
      return { ...res, viewFormatsApplied: true };
    } catch (_) {
      const { viewFormats: _ignored, ...withoutViews } = cfg;
      const res = configureCanvasContextWithCopySrc(ctx, withoutViews);
      return { ...res, viewFormatsApplied: false };
    }
  }
  const res = configureCanvasContextWithCopySrc(ctx, cfg);
  return { ...res, viewFormatsApplied: false };
}

function configureSwapchainWithCopySrc(targetSwapchain, config) {
  const usageSpec = resolveSwapchainCopySrcUsage();
  if (!usageSpec || (config && config.usage !== undefined)) {
    targetSwapchain.configure(config);
    const hasCopySrc = Boolean(
      usageSpec &&
        config &&
        typeof config.usage === 'number' &&
        (config.usage & usageSpec.withCopySrc) === usageSpec.withCopySrc
    );
    return { copySrcEnabled: hasCopySrc };
  }
  try {
    targetSwapchain.configure({ ...config, usage: usageSpec.withCopySrc });
    return { copySrcEnabled: true };
  } catch (_) {
    targetSwapchain.configure({ ...config, usage: usageSpec.renderOnly });
    return { copySrcEnabled: false };
  }
}

const OPCODE_NAMES = Object.freeze({
  [OPCODES.NOP]: 'Nop',
  [OPCODES.BEGIN_FRAME]: 'BeginFrame',
  [OPCODES.CLEAR_COLOR]: 'ClearColor',
  [OPCODES.CREATE_BUFFER]: 'CreateBuffer',
  [OPCODES.WRITE_BUFFER]: 'WriteBuffer',
  [OPCODES.CREATE_SHADER_MODULE]: 'CreateShaderModule',
  [OPCODES.DRAW]: 'Draw',
  [OPCODES.PRESENT]: 'Present',
  [OPCODES.CREATE_RENDER_PIPELINE]: 'CreateRenderPipeline',
  [OPCODES.SET_PIPELINE]: 'SetPipeline',
  [OPCODES.SET_VERTEX_BUFFER]: 'SetVertexBuffer',
  [OPCODES.DESTROY_RESOURCE]: 'DestroyResource',
  [OPCODES.CREATE_TEXTURE]: 'CreateTexture',
  [OPCODES.WRITE_TEXTURE]: 'WriteTexture',
  [OPCODES.CREATE_SAMPLER]: 'CreateSampler',
  [OPCODES.CREATE_BIND_GROUP]: 'CreateBindGroup',
  [OPCODES.SET_BIND_GROUP]: 'SetBindGroup',
  [OPCODES.SET_INDEX_BUFFER]: 'SetIndexBuffer',
  [OPCODES.DRAW_INDEXED]: 'DrawIndexed',
  [OPCODES.CREATE_COMPUTE_PIPELINE]: 'CreateComputePipeline',
  [OPCODES.SET_COMPUTE_PIPELINE]: 'SetComputePipeline',
  [OPCODES.DISPATCH_COMPUTE]: 'DispatchCompute',
  [OPCODES.CREATE_BIND_GROUP_LAYOUT]: 'CreateBindGroupLayout',
  [OPCODES.CREATE_PIPELINE_LAYOUT]: 'CreatePipelineLayout',
  [OPCODES.CREATE_RENDER_PIPELINE2]: 'CreateRenderPipeline2',
  [OPCODES.CREATE_COMPUTE_PIPELINE2]: 'CreateComputePipeline2',
  [OPCODES.BEGIN_RENDER_PASS]: 'BeginRenderPass',
  [OPCODES.END_RENDER_PASS]: 'EndRenderPass',
  [OPCODES.SET_VIEWPORT]: 'SetViewport',
  [OPCODES.SET_SCISSOR_RECT]: 'SetScissorRect',
  [OPCODES.SET_BLEND_CONSTANT]: 'SetBlendConstant',
  [OPCODES.SET_STENCIL_REFERENCE]: 'SetStencilReference',
  [OPCODES.CREATE_BIND_GROUP2]: 'CreateBindGroup2',
  [OPCODES.SET_BIND_GROUP2]: 'SetBindGroup2',
  [OPCODES.CREATE_SAMPLER2]: 'CreateSampler2',
  [OPCODES.CREATE_TEXTURE2]: 'CreateTexture2',
  [OPCODES.DRAW_INDIRECT]: 'DrawIndirect',
  [OPCODES.DRAW_INDEXED_INDIRECT]: 'DrawIndexedIndirect',
  [OPCODES.DISPATCH_INDIRECT]: 'DispatchIndirect',
  [OPCODES.READBACK_BUFFER]: 'ReadbackBuffer',
  [OPCODES.READBACK_TEXTURE]: 'ReadbackTexture',
  [OPCODE_CLEAR_TEXTURE]: 'ClearTexture',
});

function wgslDeclaresGroup0(code) {
  return typeof code === 'string' && code.length > 0 && WGSL_GROUP0_RE.test(code);
}

function opcodeLabel(opcode) {
  return OPCODE_NAMES[opcode] || `Opcode(${opcode})`;
}

const FALLBACK_PROFILE = Object.freeze({
  DEFAULT: 'default',
  FULLSCREEN_TRIANGLE_CBV: 'fullscreen_triangle_cbv',
  TEXTURED_QUAD_24B: 'textured_quad_24b',
  SPLASH_RECT_24B: 'splash_rect_24b',
});

const FALLBACK_PROFILE_ALIASES = Object.freeze({
  default: FALLBACK_PROFILE.DEFAULT,
  minimal: FALLBACK_PROFILE.DEFAULT,
  fullscreen_triangle_cbv: FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
  fullscreen_cbv: FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
  'fullscreen-triangle-cbv': FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
  'fullscreen-cbv': FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV,
  textured_quad_24b: FALLBACK_PROFILE.TEXTURED_QUAD_24B,
  textured_24b: FALLBACK_PROFILE.TEXTURED_QUAD_24B,
  'textured-quad-24b': FALLBACK_PROFILE.TEXTURED_QUAD_24B,
  'textured-24b': FALLBACK_PROFILE.TEXTURED_QUAD_24B,
  splash_rect_24b: FALLBACK_PROFILE.SPLASH_RECT_24B,
  'splash-rect-24b': FALLBACK_PROFILE.SPLASH_RECT_24B,
  rectlist_24b: FALLBACK_PROFILE.SPLASH_RECT_24B,
  'rectlist-24b': FALLBACK_PROFILE.SPLASH_RECT_24B,
  frame1_splash_24b: FALLBACK_PROFILE.SPLASH_RECT_24B,
  'frame1-splash-24b': FALLBACK_PROFILE.SPLASH_RECT_24B,
});

const TEXTURED_QUAD_24B_STRIDE = 24;

const FALLBACK_VERTEX_LAYOUT_24B = Object.freeze({
  arrayStride: TEXTURED_QUAD_24B_STRIDE,
  stepMode: 'vertex',
  attributes: Object.freeze([
    Object.freeze({ shaderLocation: 0, offset: 0, format: 'float32x4' }),
    Object.freeze({ shaderLocation: 1, offset: 16, format: 'float32x2' }),
  ]),
});

const FALLBACK_VERTEX_LAYOUT_SPLASH_RECT_24B = Object.freeze({
  arrayStride: TEXTURED_QUAD_24B_STRIDE,
  stepMode: 'vertex',
  attributes: Object.freeze([
    Object.freeze({ shaderLocation: 0, offset: 0, format: 'float32x3' }),
    Object.freeze({ shaderLocation: 1, offset: 12, format: 'unorm8x4' }),
    Object.freeze({ shaderLocation: 2, offset: 16, format: 'float32x2' }),
  ]),
});

const COMPACT_WGSL_THRESHOLD_CHARS = 2048;
const COMPACT_WGSL_GROUP0_VAR_RE =
  /(?:@group\s*\(\s*0[ui]?\s*\)\s*@binding\s*\(\s*\d+[ui]?\s*\)|@binding\s*\(\s*\d+[ui]?\s*\)\s*@group\s*\(\s*0[ui]?\s*\))\s*var(?:\s*<\s*[^>]+\s*>)?\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[^;]+;/g;
const COMPACT_WGSL_STAGE_ENTRY_RE =
  /@(vertex|fragment|compute)\b(?:\s*@[A-Za-z_][A-Za-z0-9_]*(?:\s*\([^)]*\))?)*\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)/g;

function stripWgslCommentsCompact(code) {
  if (code.indexOf('//') === -1 && code.indexOf('/*') === -1) {
    return code;
  }
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

/**
 * #1077: Compact a compiled WGSL source string down to a minimal stub that preserves exact
 * `@group(0) @binding(N)` statically-referenced declarations and stage entry-point names so
 * `render_state_packets.js` (`parseWgslGroup0Bindings`, `findWgslStageEntryPoints`,
 * `wgslDeclaresGroup0`) yields identical metadata while freeing 95-99% of worker JS heap.
 */
function compactWgslForMetadata(wgslSource) {
  if (typeof wgslSource !== 'string' || wgslSource.length === 0) {
    return wgslSource;
  }
  const cleaned = stripWgslCommentsCompact(wgslSource);
  const cleanedLen = cleaned.length;
  const fnRe = /\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*\(/g;
  const bodyChunks = [];
  let hasFunction = false;
  let match;
  while ((match = fnRe.exec(cleaned)) !== null) {
    let i = match.index + match[0].length;
    let parenDepth = 1;
    while (i < cleanedLen && parenDepth > 0) {
      const cc = cleaned.charCodeAt(i);
      if (cc === 40 /* '(' */) parenDepth += 1;
      else if (cc === 41 /* ')' */) parenDepth -= 1;
      i += 1;
    }
    while (i < cleanedLen && cleaned.charCodeAt(i) !== 123 /* '{' */) {
      if (cleaned.charCodeAt(i) === 59 /* ';' */) break;
      i += 1;
    }
    if (i >= cleanedLen || cleaned.charCodeAt(i) !== 123 /* '{' */) continue;
    hasFunction = true;
    const bodyStart = i + 1;
    let braceDepth = 1;
    i += 1;
    while (i < cleanedLen && braceDepth > 0) {
      const cc = cleaned.charCodeAt(i);
      if (cc === 123 /* '{' */) braceDepth += 1;
      else if (cc === 125 /* '}' */) braceDepth -= 1;
      i += 1;
    }
    bodyChunks.push(cleaned.slice(bodyStart, Math.max(bodyStart, i - 1)));
    fnRe.lastIndex = i;
  }
  const bodies = bodyChunks.length > 0 ? bodyChunks.join('\n') : '';

  const declLines = [];
  const usedVarNames = [];
  COMPACT_WGSL_GROUP0_VAR_RE.lastIndex = 0;
  while ((match = COMPACT_WGSL_GROUP0_VAR_RE.exec(cleaned)) !== null) {
    const varName = match[1];
    if (hasFunction) {
      const refRe = new RegExp(`(?:^|[^A-Za-z0-9_.])${varName}(?![A-Za-z0-9_])`);
      if (!refRe.test(bodies)) {
        continue;
      }
    }
    declLines.push(match[0]);
    usedVarNames.push(varName);
  }

  const entryLines = [];
  const seenEntries = new Set();
  COMPACT_WGSL_STAGE_ENTRY_RE.lastIndex = 0;
  const bodyRefLine = usedVarNames.length > 0 ? ` _ = ${usedVarNames.join('; _ = ')}; ` : '';
  while ((match = COMPACT_WGSL_STAGE_ENTRY_RE.exec(cleaned)) !== null) {
    const stageAttr = match[1];
    const fnName = match[2];
    const key = `${stageAttr}:${fnName}`;
    if (seenEntries.has(key)) continue;
    seenEntries.add(key);
    const body = entryLines.length === 0 ? bodyRefLine : '';
    entryLines.push(`@${stageAttr} fn ${fnName}() {${body}}`);
  }

  if (entryLines.length === 0 && usedVarNames.length > 0) {
    entryLines.push(`fn _meta_ref() {${bodyRefLine}}`);
  }

  const parts = declLines.concat(entryLines);
  if (parts.length === 0) {
    // Preserve @group(0) signal if present in unusual syntax so wgslDeclaresGroup0 stays accurate.
    return WGSL_GROUP0_RE.test(cleaned) ? '@group(0) // compacted' : '';
  }
  const stub = parts.join('\n');
  return stub.length < wgslSource.length ? stub : wgslSource;
}

const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
const FNV1A_64_PRIME = 0x00000100000001b3n;
const U64_MASK = 0xffffffffffffffffn;

function defaultNowMs() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function classifyGpuErrorType(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (
    normalized === 'out-of-memory' ||
    normalized === 'out_of_memory' ||
    normalized === 'outofmemory' ||
    normalized === 'oom' ||
    normalized.includes('outofmemory') ||
    normalized.includes('out-of-memory') ||
    normalized.includes('out_of_memory')
  ) {
    return 'out-of-memory';
  }
  if (normalized === 'validation' || normalized.includes('validation')) {
    return 'validation';
  }
  if (normalized === 'internal' || normalized.includes('internal')) {
    return 'internal';
  }
  return null;
}

function normalizeValidationErrorType(error, event) {
  if (error && typeof error === 'object') {
    if (typeof GPUOutOfMemoryError === 'function' && error instanceof GPUOutOfMemoryError) {
      return 'out-of-memory';
    }
    if (typeof GPUValidationError === 'function' && error instanceof GPUValidationError) {
      return 'validation';
    }
    if (typeof GPUInternalError === 'function' && error instanceof GPUInternalError) {
      return 'internal';
    }
    const fromType = classifyGpuErrorType(error.type);
    if (fromType) {
      return fromType;
    }
    const fromName = classifyGpuErrorType(error.name);
    if (fromName) {
      return fromName;
    }
    if (error.constructor && typeof error.constructor.name === 'string') {
      const fromCtor = classifyGpuErrorType(error.constructor.name);
      if (fromCtor) {
        return fromCtor;
      }
    }
  }
  if (event && typeof event === 'object') {
    const fromEventType = classifyGpuErrorType(event.errorType);
    if (fromEventType) {
      return fromEventType;
    }
  }
  return 'unknown';
}

function sanitizeValidationErrorMessage(rawMessage) {
  let msg;
  if (typeof rawMessage === 'string') {
    msg = rawMessage;
  } else if (rawMessage === null || rawMessage === undefined) {
    return '';
  } else {
    msg = String(rawMessage);
  }

  msg = msg.replace(/[\r\n\t]+/g, ' ').replace(/[\x00-\x1F\x7F]+/g, '');

  // Strip file:// URLs and host filesystem paths (Windows drive, UNC, and Unix paths).
  msg = msg.replace(/\bfile:\/\/[^\s"'`<>)\]]+/gi, '<path>');
  msg = msg.replace(/\\\\[^\s\\"'`<>)\]]+\\[^\s"'`<>)\]]+/g, '<path>');
  msg = msg.replace(/(["'`])[A-Za-z]:[\\\/][^"'`\r\n]+\1/g, '$1<path>$1');
  msg = msg.replace(/\b[A-Za-z]:[\\\/][^\s"'`<>)\],;]+/g, '<path>');
  msg = msg.replace(
    /(["'`])\/(?:Users|home|var|tmp|private|etc|opt|usr|Volumes|mnt|root|Library|Applications|System)\/[^"'`\r\n]+\1/gi,
    '$1<path>$1'
  );
  msg = msg.replace(
    /(^|[\s"'`(\[=,:])\/(?:[A-Za-z0-9._-]+\/)+[A-Za-z0-9._~+-]+/g,
    '$1<path>'
  );

  // Strip long embedded buffers (data URIs, byte arrays, hex dumps, base64 blobs).
  msg = msg.replace(/\bdata:[^\s;,'"`<>]+;base64,[A-Za-z0-9+/=]{16,}/gi, '<buffer>');
  msg = msg.replace(
    /(?:\b(?:0x)?[0-9a-fA-F]{1,4}\s*,\s*){16,}(?:0x)?[0-9a-fA-F]{1,4}\b/g,
    '<buffer>'
  );
  msg = msg.replace(/(?:\b[0-9a-fA-F]{2}\s+){16,}[0-9a-fA-F]{2}\b/g, '<buffer>');
  msg = msg.replace(/\b(?:0x)?[0-9a-fA-F]{48,}\b/g, '<buffer>');
  msg = msg.replace(
    /(^|[\s"'`(\[=,:])[A-Za-z0-9+/]{64,}={0,2}(?=$|[\s"'`)\]=,.;:])/g,
    '$1<buffer>'
  );

  msg = msg.replace(/\s{2,}/g, ' ').trim();

  if (msg.length > MAX_VALIDATION_ERROR_MESSAGE_LENGTH) {
    msg = `${msg.slice(0, MAX_VALIDATION_ERROR_MESSAGE_LENGTH - 3)}...`;
  }
  return msg;
}

function sanitizeGpuAdapterField(rawValue) {
  if (typeof rawValue !== 'string') {
    return '';
  }
  let str = rawValue
    .replace(/[\r\n\t]+/g, ' ')
    .replace(/[\x00-\x1F\x7F]+/g, '');
  // Strip file:// URLs and host filesystem paths (Windows drive, UNC, and Unix paths).
  str = str.replace(/\bfile:\/\/[^\s"'`<>)\]]+/gi, '');
  str = str.replace(/\\\\[^\s\\"'`<>)\]]+\\[^\s"'`<>)\]]+/g, '');
  str = str.replace(/(["'`])[A-Za-z]:[\\\/][^"'`\r\n]+\1/g, '');
  str = str.replace(/\b[A-Za-z]:[\\\/][^\s"'`<>)\],;]+/g, '');
  str = str.replace(
    /(["'`])\/(?:Users|home|var|tmp|private|etc|opt|usr|Volumes|mnt|root|Library|Applications|System)\/[^"'`\r\n]+\1/gi,
    ''
  );
  str = str.replace(
    /(^|[\s"'`(\[=,:])\/(?:[A-Za-z0-9._-]+\/)+[A-Za-z0-9._~+-]+/g,
    '$1'
  );
  str = str.replace(/[^\x20-\x7E]/g, '');
  str = str.replace(/\s{2,}/g, ' ').trim().toLowerCase();
  if (str.length > MAX_GPU_ADAPTER_FIELD_LENGTH) {
    str = str.slice(0, MAX_GPU_ADAPTER_FIELD_LENGTH).trimEnd();
  }
  return str;
}

function sanitizeGpuAdapterInfo(rawInfo, adapter = null) {
  const src = rawInfo && typeof rawInfo === 'object' ? rawInfo : {};
  const vendor = sanitizeGpuAdapterField(src.vendor);
  const architecture = sanitizeGpuAdapterField(src.architecture);
  const device = sanitizeGpuAdapterField(src.device);
  const description = sanitizeGpuAdapterField(src.description);
  let label = sanitizeGpuAdapterField(src.label);
  if (!label) {
    const parts = [];
    for (const part of [vendor, architecture, device, description]) {
      if (part && !parts.includes(part)) {
        parts.push(part);
      }
    }
    label = sanitizeGpuAdapterField(parts.join(' ')) || 'unknown';
  }
  const isFallbackAdapter = Boolean(
    src.isFallbackAdapter === true ||
      (adapter && typeof adapter === 'object' && adapter.isFallbackAdapter === true)
  );
  return {
    vendor,
    architecture,
    device,
    description,
    label,
    isFallbackAdapter,
  };
}

/**
 * Extract and sanitize WebGPU adapter metadata from `adapter.info` or async
 * `adapter.requestAdapterInfo()` (W257), stripping host paths, control characters,
 * and bounding every string field to `MAX_GPU_ADAPTER_FIELD_LENGTH` (96) chars.
 *
 * @param {GPUAdapter | object | null | undefined} adapter
 * @returns {Promise<{
 *   vendor: string,
 *   architecture: string,
 *   device: string,
 *   description: string,
 *   label: string,
 *   isFallbackAdapter: boolean,
 * }>}
 */
async function extractSanitizedGpuAdapter(adapter) {
  if (!adapter || typeof adapter !== 'object') {
    return sanitizeGpuAdapterInfo(null, null);
  }
  let rawInfo = null;
  try {
    if (adapter.info && typeof adapter.info === 'object') {
      rawInfo = adapter.info;
    }
  } catch (_) {
    rawInfo = null;
  }
  const hasPopulatedField = Boolean(
    rawInfo &&
      ((typeof rawInfo.vendor === 'string' && rawInfo.vendor.trim().length > 0) ||
        (typeof rawInfo.architecture === 'string' && rawInfo.architecture.trim().length > 0) ||
        (typeof rawInfo.device === 'string' && rawInfo.device.trim().length > 0) ||
        (typeof rawInfo.description === 'string' && rawInfo.description.trim().length > 0) ||
        (typeof rawInfo.label === 'string' && rawInfo.label.trim().length > 0))
  );
  if (!hasPopulatedField && typeof adapter.requestAdapterInfo === 'function') {
    try {
      const asyncInfo = await adapter.requestAdapterInfo();
      if (asyncInfo && typeof asyncInfo === 'object') {
        rawInfo = rawInfo ? { ...rawInfo, ...asyncInfo } : asyncInfo;
      }
    } catch (_) {
      // Gracefully ignore requestAdapterInfo rejections or unsupported environments.
    }
  }
  return sanitizeGpuAdapterInfo(rawInfo, adapter);
}

function normalizeFallbackProfileName(profile) {
  if (typeof profile !== 'string') {
    return FALLBACK_PROFILE.DEFAULT;
  }
  const key = profile.trim().toLowerCase();
  if (key.length === 0) {
    return FALLBACK_PROFILE.DEFAULT;
  }
  if (Object.prototype.hasOwnProperty.call(FALLBACK_PROFILE_ALIASES, key)) {
    return FALLBACK_PROFILE_ALIASES[key];
  }
  return FALLBACK_PROFILE.DEFAULT;
}

function iterateDrawCollectionValues(collection) {
  if (!collection) {
    return [];
  }
  if (collection instanceof Map) {
    return Array.from(collection.values());
  }
  if (Array.isArray(collection)) {
    return collection;
  }
  if (typeof collection === 'object') {
    return Object.values(collection);
  }
  return [];
}

function hasAnyTruthyResource(collection) {
  const values = iterateDrawCollectionValues(collection);
  for (let i = 0; i < values.length; i += 1) {
    const item = values[i];
    if (item !== null && item !== undefined && item !== false) {
      if (typeof item === 'object') {
        if (item.destroyed === true) continue;
        if ('buffer' in item && item.buffer === null && item.present !== true) continue;
        if ('texture' in item && item.texture === null && !item.textureView && item.present !== true) continue;
        if ('sampler' in item && item.sampler === null && item.present !== true) continue;
      }
      return true;
    }
  }
  return false;
}

function inspectBindGroupCollection(bindGroups) {
  let hasCbv = false;
  let hasTexture = false;
  let hasSampler = false;

  const groups = iterateDrawCollectionValues(bindGroups);
  for (let i = 0; i < groups.length; i += 1) {
    const group = groups[i];
    if (!group || typeof group !== 'object') {
      continue;
    }
    if (group.hasCbv === true || group.hasBoundCbv === true || group.cbv || group.cbvBuffer || group.uniformBuffer) {
      hasCbv = true;
    }
    if (
      group.hasTexture === true ||
      group.hasBoundTexture === true ||
      group.hasSampledTexture === true ||
      group.hasSrv === true ||
      group.texture ||
      group.textureView ||
      group.srv
    ) {
      hasTexture = true;
    }
    if (group.hasSampler === true || group.hasBoundSampler === true || group.sampler) {
      hasSampler = true;
    }
    if (group.meta && typeof group.meta === 'object') {
      if (group.meta.hasCbv === true) hasCbv = true;
      if (group.meta.hasTexture === true) hasTexture = true;
      if (group.meta.hasSampler === true) hasSampler = true;
    }

    const entries = Array.isArray(group.entries)
      ? group.entries
      : group.meta && Array.isArray(group.meta.entries)
        ? group.meta.entries
        : group.desc && Array.isArray(group.desc.entries)
          ? group.desc.entries
          : null;
    if (!entries) {
      continue;
    }
    for (let j = 0; j < entries.length; j += 1) {
      const entry = entries[j];
      if (!entry || typeof entry !== 'object') {
        continue;
      }
      const type = typeof entry.type === 'string' ? entry.type.toLowerCase() : '';
      const kind = typeof entry.kind === 'string' ? entry.kind.toLowerCase() : '';
      const resourceType = entry.resourceType;
      const resource = entry.resource;
      const resKind =
        resource && typeof resource === 'object' && typeof resource.kind === 'string'
          ? resource.kind.toLowerCase()
          : '';

      if (
        type === 'uniform' ||
        type === 'cbv' ||
        kind === 'cbv' ||
        kind === 'uniform' ||
        entry.isCbv === true ||
        resourceType === BIND_RESOURCE_TYPE.BUFFER ||
        resourceType === 'buffer' ||
        (resource && typeof resource === 'object' && resource.buffer)
      ) {
        hasCbv = true;
      }
      if (
        type === 'texture' ||
        type === 'srv' ||
        type === 'sampled-texture' ||
        kind === 'srv' ||
        kind === 'texture' ||
        kind === 'gputextureview' ||
        resKind === 'gputextureview' ||
        resKind === 'gputexture' ||
        entry.isTexture === true ||
        resourceType === BIND_RESOURCE_TYPE.TEXTURE_VIEW ||
        resourceType === 'textureView' ||
        resourceType === 'texture' ||
        entry.textureView ||
        entry.texture
      ) {
        hasTexture = true;
      }
      if (
        type === 'sampler' ||
        kind === 'sampler' ||
        kind === 'gpusampler' ||
        resKind === 'gpusampler' ||
        entry.isSampler === true ||
        resourceType === BIND_RESOURCE_TYPE.SAMPLER ||
        resourceType === 'sampler' ||
        entry.sampler
      ) {
        hasSampler = true;
      }
    }
  }

  return { hasCbv, hasTexture, hasSampler };
}

function hasVertexStrideInState(drawState, targetStride) {
  if (
    drawState.vertexStride === targetStride ||
    drawState.arrayStride === targetStride ||
    drawState.vertexBufferStride === targetStride
  ) {
    return true;
  }

  const vbValues = iterateDrawCollectionValues(drawState.vertexBuffers);
  for (let i = 0; i < vbValues.length; i += 1) {
    const vb = vbValues[i];
    if (!vb || typeof vb !== 'object') {
      continue;
    }
    if (vb.stride === targetStride || vb.arrayStride === targetStride) {
      return true;
    }
    if (
      vb.buffer &&
      typeof vb.buffer === 'object' &&
      (vb.buffer.stride === targetStride || vb.buffer.arrayStride === targetStride)
    ) {
      return true;
    }
  }

  const layoutValues = iterateDrawCollectionValues(
    drawState.vertexBufferLayouts || drawState.vertexLayouts
  );
  for (let i = 0; i < layoutValues.length; i += 1) {
    const layout = layoutValues[i];
    if (!layout || typeof layout !== 'object') {
      continue;
    }
    if (layout.arrayStride === targetStride || layout.stride === targetStride) {
      return true;
    }
  }

  return false;
}

/**
 * Detect the real Frame-1 draw pattern and select the matching G34 fallback WGSL profile:
 * - `vertexCount === 3` + bound CBV and no bound texture -> `'fullscreen_triangle_cbv'`
 * - 24-byte-stride VB + bound texture -> `'textured_quad_24b'`
 * - Otherwise -> `'default'`
 *
 * @param {object | null | undefined} drawState
 * @returns {'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b'}
 */
function selectFallbackShaderProfile(drawState) {
  if (!drawState || typeof drawState !== 'object') {
    return FALLBACK_PROFILE.DEFAULT;
  }

  const rawVertexCount = drawState.vertexCount;
  const vertexCount =
    typeof rawVertexCount === 'number' && Number.isFinite(rawVertexCount) && rawVertexCount >= 0
      ? Math.trunc(rawVertexCount)
      : 0;
  const bgInfo = inspectBindGroupCollection(drawState.bindGroups);

  const hasBoundCbv =
    drawState.hasBoundCbv === true ||
    drawState.hasCbv === true ||
    (typeof drawState.cbvCount === 'number' && drawState.cbvCount > 0) ||
    Boolean(drawState.cbv || drawState.cbvBuffer || drawState.uniformBuffer) ||
    hasAnyTruthyResource(drawState.cbvs) ||
    hasAnyTruthyResource(drawState.uniformBuffers) ||
    bgInfo.hasCbv;

  const hasBoundTexture =
    drawState.hasBoundTexture === true ||
    drawState.hasTexture === true ||
    drawState.hasSampledTexture === true ||
    drawState.hasBoundSampledTexture === true ||
    drawState.hasBoundSrv === true ||
    (typeof drawState.textureCount === 'number' && drawState.textureCount > 0) ||
    (typeof drawState.srvCount === 'number' && drawState.srvCount > 0) ||
    Boolean(drawState.texture || drawState.textureView || drawState.srv) ||
    hasAnyTruthyResource(drawState.textures) ||
    hasAnyTruthyResource(drawState.textureViews) ||
    hasAnyTruthyResource(drawState.srvs) ||
    bgInfo.hasTexture;

  if (vertexCount === 3 && hasBoundCbv && !hasBoundTexture) {
    return FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV;
  }

  if (hasVertexStrideInState(drawState, TEXTURED_QUAD_24B_STRIDE) && hasBoundTexture) {
    return FALLBACK_PROFILE.TEXTURED_QUAD_24B;
  }

  return FALLBACK_PROFILE.DEFAULT;
}

function normalizeFallbackRouterResult(raw, lookupHashHex, stageName) {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (typeof raw === 'string' && raw.length > 0) {
    return {
      wgsl: raw,
      source: 'builtin',
      requestedHash: lookupHashHex,
      resolvedHash: null,
      stage: stageName,
      entryPoint: null,
      profile: null,
    };
  }
  if (typeof raw === 'object') {
    const wgsl =
      typeof raw.wgsl === 'string'
        ? raw.wgsl
        : typeof raw.code === 'string'
          ? raw.code
          : typeof raw.source === 'string' && raw.source.includes('fn ')
            ? raw.source
            : null;
    if (!wgsl || wgsl.length === 0) {
      return null;
    }
    const source =
      typeof raw.source === 'string' && !raw.source.includes('fn ') ? raw.source : 'fallback';
    return {
      wgsl,
      source,
      requestedHash: raw.requestedHash || lookupHashHex,
      resolvedHash: raw.resolvedHash !== undefined ? raw.resolvedHash : null,
      stage: raw.stage || stageName,
      entryPoint: typeof raw.entryPoint === 'string' ? raw.entryPoint : null,
      profile:
        typeof raw.profile === 'string' && raw.profile.length > 0
          ? normalizeFallbackProfileName(raw.profile)
          : null,
    };
  }
  return null;
}

/**
 * Call `shaderFallbackRouter.resolve(hash, stage, { profile })` while staying 100%
 * backwards-compatible with legacy 2-arg routers (including strict mocks that reject a 3rd arg).
 */
function callFallbackRouterResolve(router, lookupHashHex, stageName, profile) {
  if (!router || typeof router.resolve !== 'function') {
    return null;
  }
  const canonicalProfile = normalizeFallbackProfileName(profile);
  if (canonicalProfile !== FALLBACK_PROFILE.DEFAULT) {
    try {
      const withProfile = router.resolve(lookupHashHex, stageName, { profile: canonicalProfile });
      const normalized = normalizeFallbackRouterResult(withProfile, lookupHashHex, stageName);
      if (normalized) {
        return normalized;
      }
    } catch (_) {
      // Legacy 2-arg router threw on the 3rd options argument — retry with 2 args.
    }
  }
  const legacy = router.resolve(lookupHashHex, stageName);
  return normalizeFallbackRouterResult(legacy, lookupHashHex, stageName);
}

const TEXTURE_FORMAT_BPP = Object.freeze({
  r8unorm: 1,
  r8snorm: 1,
  r8uint: 1,
  r8sint: 1,
  stencil8: 1,
  r16unorm: 2,
  r16snorm: 2,
  r16uint: 2,
  r16sint: 2,
  r16float: 2,
  rg8unorm: 2,
  rg8snorm: 2,
  rg8uint: 2,
  rg8sint: 2,
  depth16unorm: 2,
  r32uint: 4,
  r32sint: 4,
  r32float: 4,
  rg16unorm: 4,
  rg16snorm: 4,
  rg16uint: 4,
  rg16sint: 4,
  rg16float: 4,
  rgba8unorm: 4,
  'rgba8unorm-srgb': 4,
  rgba8snorm: 4,
  rgba8uint: 4,
  rgba8sint: 4,
  bgra8unorm: 4,
  'bgra8unorm-srgb': 4,
  rgb9e5ufloat: 4,
  rgb10a2uint: 4,
  rgb10a2unorm: 4,
  rg11b10ufloat: 4,
  depth24plus: 4,
  'depth24plus-stencil8': 4,
  depth32float: 4,
  rg32uint: 8,
  rg32sint: 8,
  rg32float: 8,
  rgba16unorm: 8,
  rgba16snorm: 8,
  rgba16uint: 8,
  rgba16sint: 8,
  rgba16float: 8,
  'depth32float-stencil8': 8,
  rgba32uint: 16,
  rgba32sint: 16,
  rgba32float: 16,
});

const TEXTURE_BLOCK_FORMAT_BYTES = Object.freeze({
  'bc1-rgba-unorm': 8,
  'bc1-rgba-unorm-srgb': 8,
  'bc2-rgba-unorm': 16,
  'bc2-rgba-unorm-srgb': 16,
  'bc3-rgba-unorm': 16,
  'bc3-rgba-unorm-srgb': 16,
  'bc4-r-unorm': 8,
  'bc4-r-snorm': 8,
  'bc5-rg-unorm': 16,
  'bc5-rg-snorm': 16,
  'bc6h-rgb-ufloat': 16,
  'bc6h-rgb-float': 16,
  'bc7-rgba-unorm': 16,
  'bc7-rgba-unorm-srgb': 16,
});

function planClearTextureCopies(geom, maxChunkBytes = CLEAR_TEXTURE_ZERO_BUFFER_BYTES) {
  const width = Math.max(1, Math.floor(Number(geom && geom.width) || 1));
  const dimension = geom && typeof geom.dimension === 'string' ? geom.dimension : '2d';
  const is1D = dimension === '1d';
  const is3D = dimension === '3d';
  const height = is1D ? 1 : Math.max(1, Math.floor(Number(geom && geom.height) || 1));
  const depthOrLayers = is1D
    ? 1
    : Math.max(1, Math.floor(Number(geom && geom.depthOrArrayLayers) || 1));
  const mips = is1D ? 1 : Math.max(1, Math.floor(Number(geom && geom.mipLevelCount) || 1));
  const blockW = Math.max(1, Math.floor(Number(geom && geom.blockW) || 1));
  const blockH = Math.max(1, Math.floor(Number(geom && geom.blockH) || 1));
  const blockBytes = Math.max(1, Math.floor(Number(geom && geom.blockBytes) || 4));
  const roundUp = (v, a) => Math.ceil(v / a) * a;
  const level = (base, m) => (m >= 31 ? 1 : Math.max(1, base >>> m));

  const widestBlocksX = roundUp(width, blockW) / blockW;
  const widestRow = roundUp(widestBlocksX * blockBytes, CLEAR_TEXTURE_ROW_ALIGN);
  const cap = Math.max(
    widestRow,
    Math.floor(Number(maxChunkBytes) || CLEAR_TEXTURE_ZERO_BUFFER_BYTES)
  );

  const copies = [];
  let maxRequiredBytes = 0;
  for (let m = 0; m < mips; m += 1) {
    const mw = roundUp(level(width, m), blockW);
    const mh = is1D ? 1 : roundUp(level(height, m), blockH);
    const md = is3D ? level(depthOrLayers, m) : depthOrLayers;
    const blocksX = mw / blockW;
    const rowsFull = mh / blockH;
    const tightRow = blocksX * blockBytes;
    const bytesPerRow = roundUp(tightRow, CLEAR_TEXTURE_ROW_ALIGN);
    const sliceBytes = rowsFull * bytesPerRow;
    if (sliceBytes <= cap) {
      const per = Math.max(1, Math.min(md, Math.floor(cap / sliceBytes)));
      for (let s = 0; s < md; s += per) {
        const n = Math.min(per, md - s);
        const copyBytes = sliceBytes * n;
        if (copyBytes > maxRequiredBytes) {
          maxRequiredBytes = copyBytes;
        }
        copies.push({
          mip: m,
          y: 0,
          z: s,
          width: mw,
          height: mh,
          depth: n,
          bytesPerRow,
          rowsPerImage: rowsFull,
          copyBytes,
          tightBytes: tightRow * rowsFull * n,
        });
      }
    } else {
      const perRows = Math.max(1, Math.floor(cap / bytesPerRow));
      for (let s = 0; s < md; s += 1) {
        for (let r = 0; r < rowsFull; r += perRows) {
          const nRows = Math.min(perRows, rowsFull - r);
          const copyBytes = nRows * bytesPerRow;
          if (copyBytes > maxRequiredBytes) {
            maxRequiredBytes = copyBytes;
          }
          copies.push({
            mip: m,
            y: r * blockH,
            z: s,
            width: mw,
            height: nRows * blockH,
            depth: 1,
            bytesPerRow,
            rowsPerImage: nRows,
            copyBytes,
            tightBytes: tightRow * nRows,
          });
        }
      }
    }
  }
  return { copies, maxRequiredBytes, widestRow };
}

function fnv1a64Hex(text) {
  const bytes = new TextEncoder().encode(String(text));
  let hash = FNV1A_64_OFFSET_BASIS;
  for (let i = 0; i < bytes.length; i += 1) {
    hash ^= BigInt(bytes[i]);
    hash = (hash * FNV1A_64_PRIME) & U64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, '0')}`;
}

function estimateTextureBytes(
  width,
  height,
  depthOrArrayLayers,
  mipLevelCount,
  format,
  dimension,
  sampleCount = 1
) {
  const w0 = Math.max(1, Math.floor(Number(width) || 1));
  const is1D = dimension === '1d';
  const is3D = dimension === '3d';
  const h0 = is1D ? 1 : Math.max(1, Math.floor(Number(height) || 1));
  const d0 = is1D ? 1 : Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
  const mips = Math.max(1, Math.floor(Number(mipLevelCount) || 1));
  const samples = Math.max(1, Math.floor(Number(sampleCount) || 1));
  const fmtKey = typeof format === 'string' ? format.trim().toLowerCase() : 'rgba8unorm';
  const bcBlockBytes = TEXTURE_BLOCK_FORMAT_BYTES[fmtKey] || 0;
  const bpp = bcBlockBytes > 0 ? 0 : TEXTURE_FORMAT_BPP[fmtKey] || 4;

  let total = 0;
  for (let mip = 0; mip < mips; mip += 1) {
    const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
    const mw = scale === Infinity ? 1 : Math.max(1, Math.floor(w0 / scale));
    const mh = is1D ? 1 : scale === Infinity ? 1 : Math.max(1, Math.floor(h0 / scale));
    const md = is3D ? (scale === Infinity ? 1 : Math.max(1, Math.floor(d0 / scale))) : d0;
    if (bcBlockBytes > 0) {
      total += Math.ceil(mw / 4) * Math.ceil(mh / 4) * bcBlockBytes * md * samples;
    } else {
      total += mw * mh * md * bpp * samples;
    }
  }
  return total;
}

function extractWgslEntryPoint(code, stage) {
  if (typeof code !== 'string' || code.length === 0) {
    return null;
  }
  const attr =
    stage === SHADER_STAGE.VERTEX
      ? 'vertex'
      : stage === SHADER_STAGE.FRAGMENT
        ? 'fragment'
        : stage === SHADER_STAGE.COMPUTE
          ? 'compute'
          : null;
  if (attr) {
    const re = new RegExp(`@${attr}\\b[\\s\\S]*?\\bfn\\s+([A-Za-z_][A-Za-z0-9_]*)`);
    const m = re.exec(code);
    if (m) return m[1];
  }
  const fallback = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/.exec(code);
  return fallback ? fallback[1] : null;
}

function stageCodeToName(stage) {
  if (stage === SHADER_STAGE.VERTEX) return 'vertex';
  if (stage === SHADER_STAGE.FRAGMENT) return 'fragment';
  if (stage === SHADER_STAGE.COMPUTE) return 'compute';
  return null;
}

function resolveSubsystemCtor(injectedCtor, globalNamespace, exportName, scriptRelPath) {
  if (typeof injectedCtor === 'function') {
    return injectedCtor;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis[globalNamespace] &&
    typeof globalThis[globalNamespace][exportName] === 'function'
  ) {
    return globalThis[globalNamespace][exportName];
  }
  if (typeof importScripts === 'function') {
    try {
      importScripts(scriptRelPath);
      if (
        typeof globalThis !== 'undefined' &&
        globalThis[globalNamespace] &&
        typeof globalThis[globalNamespace][exportName] === 'function'
      ) {
        return globalThis[globalNamespace][exportName];
      }
    } catch (_) {
      // Optional helper script unavailable in this worker environment.
    }
  }
  if (typeof require === 'function') {
    try {
      const mod = require(scriptRelPath);
      if (mod && typeof mod[exportName] === 'function') {
        return mod[exportName];
      }
    } catch (_) {
      // Optional helper module unavailable.
    }
  }
  return null;
}

// Synchronous classic-worker fallback matching web/workers/rt_bvh_prelude.js
// (and runtime/gpu/rt_bvh.cpp::prelude_wgsl()) byte-for-byte so RT shaders calling
// pg6_bvh_intersect() / pg6_load32() can NEVER compile un-prefixed if dynamic
// import('./rt_bvh_prelude.js') is still pending or unavailable.
const INLINE_RT_BVH_PRELUDE_VERSION = 1;

const INLINE_RT_BVH_PRELUDE_WGSL = `// pg6 runtime prelude, version 1 (rt_bvh.h kPreludeVersion). Group 3, bindings 16-21: contract C3.
@group(3) @binding(16) var<storage, read_write> pg6_arena0: array<u32>;
@group(3) @binding(17) var<storage, read_write> pg6_arena1: array<u32>;
@group(3) @binding(18) var<storage, read_write> pg6_arena2: array<u32>;
@group(3) @binding(19) var<storage, read_write> pg6_arena3: array<u32>;
@group(3) @binding(20) var<storage, read> pg6_arena_dir: array<u32>;
@group(3) @binding(21) var<storage, read_write> pg6_rt_stats: array<atomic<u32>>;

// A non-resident access: bumps rt_fault (word 0) and logs the page for the runtime to make resident
// (word 1 = log attempts, words 2..31 = faulting page + 1, first 30 only). rt_residency.h FaultTracker.
fn pg6_fault(page: u32) {
    atomicAdd(&pg6_rt_stats[0], 1u);
    let i = atomicAdd(&pg6_rt_stats[1], 1u);
    if (i < 30u) {
        atomicStore(&pg6_rt_stats[2u + i], page + 1u);
    }
}

// Resolves a 40-bit address through the two-level page directory once. Returns 0u on fault.
fn pg6_resolve(addr: vec2<u32>) -> u32 {
    let page = (addr.x >> 16u) | ((addr.y & 0xffu) << 16u);
    let t = pg6_arena_dir[page >> 12u];
    if (t == 0u) {
        pg6_fault(page);
        return 0u;
    }
    let e = pg6_arena_dir[4096u + (t - 1u) * 4096u + (page & 0xfffu)];
    if ((e & 0x80000000u) == 0u) {
        pg6_fault(page);
        return 0u;
    }
    return e;
}

fn pg6_read_resolved(e: u32, page_byte_off: u32) -> u32 {
    let w = (e & 0x1fffffffu) * 16384u + ((page_byte_off & 0xffffu) >> 2u);
    switch ((e >> 29u) & 3u) {
        case 0u: { return pg6_arena0[w]; }
        case 1u: { return pg6_arena1[w]; }
        case 2u: { return pg6_arena2[w]; }
        default: { return pg6_arena3[w]; }
    }
}

// Dword at a 40-bit address (x = low 32 bits, y = bits 39:32). Unmapped: counts a fault, reads 0.
fn pg6_load32(addr: vec2<u32>) -> u32 {
    let e = pg6_resolve(addr);
    if (e == 0u) {
        return 0u;
    }
    return pg6_read_resolved(e, addr.x);
}

fn pg6_add40(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    var carry = 0u;
    if (lo < a.x) { carry = 1u; }
    return vec2<u32>(lo, (a.y + b.y + carry) & 0xffu);
}

fn pg6_addr_off(a: vec2<u32>, off: u32) -> vec2<u32> {
    return pg6_add40(a, vec2<u32>(off, 0u));
}

fn pg6_f(a: vec2<u32>, off: u32) -> f32 {
    return bitcast<f32>(pg6_load32(pg6_addr_off(a, off)));
}

fn pg6_f_resolved(e: u32, page_off: u32, off: u32) -> f32 {
    return bitcast<f32>(pg6_read_resolved(e, page_off + off));
}

// Bit-exact IEEE-754 NaN test (immune to fast-math folding \`x != x\` to false on Metal/HLSL).
fn pg6_isnan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

// Slab test of one box: .x = 1.0 when entered, .y = the entry distance.
fn pg6_slab(mn: vec3<f32>, mx: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, extent: f32) -> vec2<f32> {
    var lo = 0.0;
    var hi = extent;
    for (var a = 0u; a < 3u; a = a + 1u) {
        if (pg6_isnan(mn[a]) || pg6_isnan(mx[a]) || mn[a] > mx[a]) {
            return vec2<f32>(0.0, 0.0);
        }
        let t0 = (mn[a] - o[a]) * inv[a];
        let t1 = (mx[a] - o[a]) * inv[a];
        if (pg6_isnan(t0) || pg6_isnan(t1)) {
            continue;
        }
        var a0 = t1;
        var a1 = t0;
        if (t0 < t1) {
            a0 = t0;
            a1 = t1;
        }
        if (a0 > lo) { lo = a0; }
        if (a1 < hi) { hi = a1; }
    }
    if (lo <= hi) {
        return vec2<f32>(1.0, lo);
    }
    return vec2<f32>(0.0, 0.0);
}

fn pg6_tri_miss() -> vec4<u32> {
    return vec4<u32>(0x7f800000u, 0x3f800000u, 0u, 0u);
}

fn pg6_vec_nan(v: vec3<f32>) -> bool {
    return pg6_isnan(v.x) || pg6_isnan(v.y) || pg6_isnan(v.z);
}

fn pg6_triangle(e: u32, page_off: u32, kind: u32, o: vec3<f32>, d: vec3<f32>) -> vec4<u32> {
    var vi = vec3<u32>(0u, 1u, 2u);
    if (kind == 1u) { vi = vec3<u32>(1u, 3u, 2u); }
    if (kind == 2u) { vi = vec3<u32>(2u, 3u, 4u); }
    if (kind == 3u) { vi = vec3<u32>(2u, 4u, 0u); }
    var v: array<vec3<f32>, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let b = vi[i] * 12u;
        v[i] = vec3<f32>(pg6_f_resolved(e, page_off, b), pg6_f_resolved(e, page_off, b + 4u),
                         pg6_f_resolved(e, page_off, b + 8u)) - o;
    }
    var kz = 0u;
    var az = abs(d.x);
    if (abs(d.y) > az) { kz = 1u; az = abs(d.y); }
    if (abs(d.z) > az) { kz = 2u; }
    let dz = d[kz];
    if (dz == 0.0) {
        return pg6_tri_miss();
    }
    var kx = (kz + 1u) % 3u;
    var ky = (kx + 1u) % 3u;
    if (dz < 0.0) {
        let t = kx;
        kx = ky;
        ky = t;
    }
    let sx = d[kx] / dz;
    let sy = d[ky] / dz;
    let sz = 1.0 / dz;
    var px: array<f32, 3>;
    var py: array<f32, 3>;
    var pz: array<f32, 3>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        px[i] = v[i][kx] - sx * v[i][kz];
        py[i] = v[i][ky] - sy * v[i][kz];
        pz[i] = sz * v[i][kz];
    }
    let u = px[2] * py[1] - py[2] * px[1];
    let vv = px[0] * py[2] - py[0] * px[2];
    let w = px[1] * py[0] - py[1] * px[0];
    if ((u < 0.0 || vv < 0.0 || w < 0.0) && (u > 0.0 || vv > 0.0 || w > 0.0)) {
        return pg6_tri_miss();
    }
    let det = u + vv + w;
    let t_num = u * pz[0] + vv * pz[1] + w * pz[2];
    if (det == 0.0 || pg6_isnan(det) || pg6_isnan(t_num) || pg6_isnan(u) || pg6_isnan(vv) || pg6_isnan(w)) {
        return pg6_tri_miss();
    }
    let sel = (pg6_read_resolved(e, page_off + 60u) >> (8u * kind)) & 0xffu;
    var wt = array<f32, 4>(u, vv, w, 0.0);
    return vec4<u32>(bitcast<u32>(t_num), bitcast<u32>(det), bitcast<u32>(wt[sel & 3u]),
                     bitcast<u32>(wt[(sel >> 2u) & 3u]));
}

// C4: the console's image_bvh(64)_intersect_ray node test. desc = the 128-bit BVH descriptor,
// node = the node pointer (x low, y high; high 0 for the 32-bit form).
fn pg6_bvh_intersect(desc: vec4<u32>, node: vec2<u32>, extent: f32, origin: vec3<f32>, dir: vec3<f32>,
                     inv_dir: vec3<f32>) -> vec4<u32> {
    let kind = node.x & 7u;
    if (kind >= 6u) {
        return pg6_tri_miss();
    }
    let nan_ray = pg6_isnan(extent) || pg6_vec_nan(origin) || pg6_vec_nan(dir) || pg6_vec_nan(inv_dir);
    // address = (base256 << 8) + ((ptr >> 3) << 6), 40 bits (guaranteed 64-byte aligned)
    let base = vec2<u32>(desc.x << 8u, ((desc.y & 0xffu) << 8u) | (desc.x >> 24u));
    let idx = vec2<u32>((node.x >> 3u) | (node.y << 29u), node.y >> 3u);
    let off = vec2<u32>(idx.x << 6u, (idx.y << 6u) | (idx.x >> 26u));
    let addr = pg6_add40(base, off);
    let page_off = addr.x & 0xffffu;
    if (kind < 4u) {
        if (nan_ray) {
            return pg6_tri_miss();
        }
        let e0 = pg6_resolve(addr);
        if (e0 == 0u) {
            return pg6_tri_miss();
        }
        return pg6_triangle(e0, page_off, kind, origin, dir);
    }
    if (nan_ray) {
        return vec4<u32>(0xffffffffu);
    }
    let e0 = pg6_resolve(addr);
    if (e0 == 0u) {
        return vec4<u32>(0xffffffffu);
    }
    let single_page = kind == 4u || page_off <= 0xff80u;
    var ptr: array<u32, 4>;
    var key: array<f32, 4>;
    var hit: array<bool, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        ptr[i] = pg6_read_resolved(e0, page_off + 4u * i);
        key[i] = 0.0;
        hit[i] = false;
        if (ptr[i] == 0xffffffffu) {
            continue;
        }
        var mn: vec3<f32>;
        var mx: vec3<f32>;
        if (kind == 5u) {
            let b = 16u + 24u * i;
            if (single_page) {
                mn = vec3<f32>(pg6_f_resolved(e0, page_off, b), pg6_f_resolved(e0, page_off, b + 4u),
                               pg6_f_resolved(e0, page_off, b + 8u));
                mx = vec3<f32>(pg6_f_resolved(e0, page_off, b + 12u), pg6_f_resolved(e0, page_off, b + 16u),
                               pg6_f_resolved(e0, page_off, b + 20u));
            } else {
                mn = vec3<f32>(pg6_f(addr, b), pg6_f(addr, b + 4u), pg6_f(addr, b + 8u));
                mx = vec3<f32>(pg6_f(addr, b + 12u), pg6_f(addr, b + 16u), pg6_f(addr, b + 20u));
            }
        } else {
            let b = 16u + 12u * i;
            let a = unpack2x16float(pg6_read_resolved(e0, page_off + b));
            let bb = unpack2x16float(pg6_read_resolved(e0, page_off + b + 4u));
            let c = unpack2x16float(pg6_read_resolved(e0, page_off + b + 8u));
            mn = vec3<f32>(a.x, a.y, bb.x);
            mx = vec3<f32>(bb.y, c.x, c.y);
        }
        let s = pg6_slab(mn, mx, origin, inv_dir, extent);
        hit[i] = s.x > 0.5;
        key[i] = s.y;
    }
    var out = vec4<u32>(0xffffffffu);
    let sort = (desc.y >> 31u) != 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (!hit[i]) {
            continue;
        }
        var rank = i;
        if (sort) {
            rank = 0u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                if (hit[j] && (key[j] < key[i] || (key[j] == key[i] && j < i))) {
                    rank = rank + 1u;
                }
            }
        }
        out[rank] = ptr[i];
    }
    return out;
}
`;

const RT_BVH_HELPER_CALL_RE = /\b(?:pg6_bvh_intersect|pg6_load32)\s*\(/;
const RT_BVH_HELPER_DEF_RE = /\bfn\s+(?:pg6_load32|pg6_bvh_intersect)\s*\(/;

function isRtWgslWhitespaceCode(ch) {
  return (
    ch === 0x20 ||
    ch === 0x09 ||
    ch === 0x0a ||
    ch === 0x0d ||
    ch === 0x0b ||
    ch === 0x0c ||
    ch === 0xfeff
  );
}

function skipRtWgslCommentAt(wgsl, pos) {
  const len = wgsl.length;
  if (pos + 1 >= len || wgsl.charCodeAt(pos) !== 0x2f) {
    return pos;
  }
  const next = wgsl.charCodeAt(pos + 1);
  if (next === 0x2f) {
    let i = pos + 2;
    while (i < len && wgsl.charCodeAt(i) !== 0x0a) {
      i += 1;
    }
    return i;
  }
  if (next === 0x2a) {
    let i = pos + 2;
    let depth = 1;
    while (i < len && depth > 0) {
      const c = wgsl.charCodeAt(i);
      if (c === 0x2f && i + 1 < len && wgsl.charCodeAt(i + 1) === 0x2a) {
        depth += 1;
        i += 2;
      } else if (c === 0x2a && i + 1 < len && wgsl.charCodeAt(i + 1) === 0x2f) {
        depth -= 1;
        i += 2;
      } else {
        i += 1;
      }
    }
    return i;
  }
  return pos;
}

function skipRtWgslLeadingTrivia(wgsl, start) {
  const len = wgsl.length;
  let i = start;
  while (i < len) {
    const ch = wgsl.charCodeAt(i);
    if (isRtWgslWhitespaceCode(ch)) {
      i += 1;
      continue;
    }
    const afterComment = skipRtWgslCommentAt(wgsl, i);
    if (afterComment !== i) {
      i = afterComment;
      continue;
    }
    break;
  }
  return i;
}

function matchRtWgslDirectiveKeyword(wgsl, pos) {
  const len = wgsl.length;
  if (wgsl.startsWith('enable', pos)) {
    const after = pos + 6;
    if (after < len) {
      const ch = wgsl.charCodeAt(after);
      if (isRtWgslWhitespaceCode(ch) || ch === 0x2f) {
        return after;
      }
    }
  } else if (wgsl.startsWith('requires', pos)) {
    const after = pos + 8;
    if (after < len) {
      const ch = wgsl.charCodeAt(after);
      if (isRtWgslWhitespaceCode(ch) || ch === 0x2f) {
        return after;
      }
    }
  } else if (wgsl.startsWith('diagnostic', pos)) {
    const after = pos + 10;
    if (after < len) {
      const ch = wgsl.charCodeAt(after);
      if (isRtWgslWhitespaceCode(ch) || ch === 0x2f || ch === 0x28) {
        return after;
      }
    }
  }
  return -1;
}

function findRtWgslGlobalDirectivesEnd(wgsl) {
  const len = wgsl.length;
  let pos = 0;
  let lastDirectiveEnd = 0;

  while (pos < len) {
    const kwStart = skipRtWgslLeadingTrivia(wgsl, pos);
    if (kwStart >= len) {
      break;
    }
    const afterKw = matchRtWgslDirectiveKeyword(wgsl, kwStart);
    if (afterKw < 0) {
      break;
    }

    let i = afterKw;
    let parenDepth = 0;
    let foundSemi = false;
    while (i < len) {
      const afterComment = skipRtWgslCommentAt(wgsl, i);
      if (afterComment !== i) {
        i = afterComment;
        continue;
      }
      const ch = wgsl.charCodeAt(i);
      if (ch === 0x28) {
        parenDepth += 1;
        i += 1;
      } else if (ch === 0x29) {
        if (parenDepth > 0) {
          parenDepth -= 1;
        }
        i += 1;
      } else if (ch === 0x3b && parenDepth === 0) {
        i += 1;
        foundSemi = true;
        break;
      } else {
        i += 1;
      }
    }

    if (!foundSemi) {
      break;
    }

    let tail = i;
    while (tail < len && (wgsl.charCodeAt(tail) === 0x20 || wgsl.charCodeAt(tail) === 0x09)) {
      tail += 1;
    }
    if (
      tail + 1 < len &&
      wgsl.charCodeAt(tail) === 0x2f &&
      wgsl.charCodeAt(tail + 1) === 0x2f
    ) {
      tail = skipRtWgslCommentAt(wgsl, tail);
    }
    if (tail < len && wgsl.charCodeAt(tail) === 0x0d) {
      tail += 1;
      if (tail < len && wgsl.charCodeAt(tail) === 0x0a) {
        tail += 1;
      }
      i = tail;
    } else if (tail < len && wgsl.charCodeAt(tail) === 0x0a) {
      i = tail + 1;
    }

    lastDirectiveEnd = i;
    pos = i;
  }

  return lastDirectiveEnd;
}

function stripRtWgslComments(wgsl) {
  const len = wgsl.length;
  let out = '';
  let chunkStart = 0;
  let i = 0;
  while (i < len) {
    if (wgsl.charCodeAt(i) === 0x2f && i + 1 < len) {
      const next = wgsl.charCodeAt(i + 1);
      if (next === 0x2f || next === 0x2a) {
        if (i > chunkStart) {
          out += wgsl.slice(chunkStart, i);
        }
        out += ' ';
        i = skipRtWgslCommentAt(wgsl, i);
        chunkStart = i;
        continue;
      }
    }
    i += 1;
  }
  if (chunkStart === 0) {
    return wgsl;
  }
  if (chunkStart < len) {
    out += wgsl.slice(chunkStart);
  }
  return out;
}

function needsRtBvhPreludeInline(wgsl) {
  if (typeof wgsl !== 'string' || wgsl.length === 0) {
    return false;
  }
  if (!RT_BVH_HELPER_CALL_RE.test(wgsl)) {
    return false;
  }
  const code = stripRtWgslComments(wgsl);
  return RT_BVH_HELPER_CALL_RE.test(code) && !RT_BVH_HELPER_DEF_RE.test(code);
}

function applyRtBvhPreludeInline(wgsl) {
  if (!needsRtBvhPreludeInline(wgsl)) {
    return wgsl;
  }
  const source = wgsl.charCodeAt(0) === 0xfeff ? wgsl.slice(1) : wgsl;
  const insertAt = findRtWgslGlobalDirectivesEnd(source);
  if (insertAt === 0) {
    return `${INLINE_RT_BVH_PRELUDE_WGSL}\n${source}`;
  }
  const prefix = source.slice(0, insertAt);
  const rest = source.slice(insertAt);
  const sepBefore = prefix.endsWith('\n') ? '' : '\n';
  const sepAfter = rest.startsWith('\n') || rest.startsWith('\r\n') ? '' : '\n';
  return `${prefix}${sepBefore}${INLINE_RT_BVH_PRELUDE_WGSL}${sepAfter}${rest}`;
}

const INLINE_RT_BVH_PRELUDE_API = Object.freeze({
  RT_BVH_PRELUDE_VERSION: INLINE_RT_BVH_PRELUDE_VERSION,
  RT_BVH_PRELUDE_WGSL: INLINE_RT_BVH_PRELUDE_WGSL,
  needsRtBvhPrelude: needsRtBvhPreludeInline,
  applyRtBvhPrelude: applyRtBvhPreludeInline,
});

const MAX_RT_BVH_PRELUDE_IMPORT_ATTEMPTS = 3;
let cachedRtBvhPreludeApi = null;
let pendingRtBvhPreludePromise = null;
let rtBvhPreludeImportAttempts = 0;

function resolveRtBvhPreludeApiSync() {
  if (cachedRtBvhPreludeApi) {
    return cachedRtBvhPreludeApi;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis.PlayGta6RtBvhPrelude &&
    typeof globalThis.PlayGta6RtBvhPrelude.applyRtBvhPrelude === 'function'
  ) {
    cachedRtBvhPreludeApi = globalThis.PlayGta6RtBvhPrelude;
    return cachedRtBvhPreludeApi;
  }
  if (typeof require === 'function') {
    try {
      const mod = require('./rt_bvh_prelude.js');
      if (mod && typeof mod.applyRtBvhPrelude === 'function') {
        cachedRtBvhPreludeApi = mod;
        return cachedRtBvhPreludeApi;
      }
    } catch (_) {
      // Optional ESM helper unavailable synchronously in this environment.
    }
  }
  return null;
}

function ensureRtBvhPreludeLoaded() {
  const syncApi = resolveRtBvhPreludeApiSync();
  if (syncApi) {
    return Promise.resolve(syncApi);
  }
  if (!pendingRtBvhPreludePromise) {
    if (rtBvhPreludeImportAttempts >= MAX_RT_BVH_PRELUDE_IMPORT_ATTEMPTS) {
      return Promise.resolve(INLINE_RT_BVH_PRELUDE_API);
    }
    rtBvhPreludeImportAttempts += 1;
    pendingRtBvhPreludePromise = Promise.resolve()
      .then(() => import('./rt_bvh_prelude.js'))
      .then((mod) => {
        if (mod && typeof mod.applyRtBvhPrelude === 'function') {
          cachedRtBvhPreludeApi = mod;
          return mod;
        }
        pendingRtBvhPreludePromise = null;
        return INLINE_RT_BVH_PRELUDE_API;
      })
      .catch(() => {
        pendingRtBvhPreludePromise = null;
        return INLINE_RT_BVH_PRELUDE_API;
      });
  }
  return pendingRtBvhPreludePromise;
}

function applyRtBvhPreludeIfNeeded(code) {
  if (typeof code !== 'string' || code.length === 0) {
    return code;
  }
  const api = resolveRtBvhPreludeApiSync();
  if (api && typeof api.applyRtBvhPrelude === 'function') {
    return api.applyRtBvhPrelude(code);
  }
  // Best-effort ESM preload while applying the byte-identical synchronous inline fallback
  // immediately so no shader calling pg6_bvh_intersect/pg6_load32 ever compiles un-prefixed.
  if (!pendingRtBvhPreludePromise && rtBvhPreludeImportAttempts === 0) {
    ensureRtBvhPreludeLoaded();
  }
  return applyRtBvhPreludeInline(code);
}

const WGPK_MAGIC_BYTES = Object.freeze([0x57, 0x47, 0x50, 0x4b]);
const WGPK_VERSION = 1;
const WGPK_HEADER_BYTES = 16;
const WGPK_ENTRY_BYTES = 20;
const WGPK_STAGE_NAMES = Object.freeze({ 1: 'vertex', 2: 'fragment', 3: 'compute' });

function normalizeShaderHashBigInt(hash) {
  if (typeof hash === 'bigint') {
    return hash >= 0n && hash <= U64_MASK ? hash : null;
  }
  if (typeof hash === 'number') {
    return Number.isInteger(hash) && Number.isSafeInteger(hash) && hash >= 0 ? BigInt(hash) : null;
  }
  if (typeof hash === 'string') {
    const trimmed = hash.trim();
    const rawHex =
      trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
    if (/^[0-9a-fA-F]{1,16}$/.test(rawHex)) {
      return BigInt(`0x${rawHex}`);
    }
  }
  return null;
}

function formatShaderHashHex(hashBigInt) {
  return `0x${hashBigInt.toString(16).padStart(16, '0')}`;
}

function parseHashOnlyShaderPayload(trimmed) {
  if (typeof trimmed !== 'string' || trimmed.length === 0) {
    return null;
  }
  const prefixed = /^(?:hash|wgpk)\s*:\s*(?:0[xX])?([0-9a-fA-F]{1,16})$/i.exec(trimmed);
  if (prefixed) {
    return formatShaderHashHex(BigInt(`0x${prefixed[1]}`));
  }
  const hex0x = /^0[xX]([0-9a-fA-F]{1,16})$/.exec(trimmed);
  if (hex0x) {
    return formatShaderHashHex(BigInt(`0x${hex0x[1]}`));
  }
  const bare64 = /^([0-9a-fA-F]{16})$/.exec(trimmed);
  if (bare64) {
    return formatShaderHashHex(BigInt(`0x${bare64[1]}`));
  }
  return null;
}

function unwrapPackLookupResult(raw) {
  if (raw === undefined || raw === null) {
    return null;
  }
  if (typeof raw === 'string') {
    return { wgsl: raw, entryPoint: null };
  }
  if (typeof raw === 'object') {
    if (typeof raw.wgsl === 'string') {
      return {
        wgsl: raw.wgsl,
        entryPoint:
          typeof raw.entryPoint === 'string'
            ? raw.entryPoint
            : typeof raw.entry_point === 'string'
              ? raw.entry_point
              : null,
      };
    }
    if (typeof raw.source === 'string') {
      return { wgsl: raw.source, entryPoint: null };
    }
    if (typeof raw.code === 'string') {
      return { wgsl: raw.code, entryPoint: null };
    }
  }
  return null;
}

function lookupFromPackCache(packCache, hash) {
  if (!packCache) {
    return null;
  }
  const big = normalizeShaderHashBigInt(hash);
  if (big === null) {
    return null;
  }
  const hex = formatShaderHashHex(big);
  const keys = [big, hex];
  const tryCall = (fn, key) => {
    if (typeof fn !== 'function') {
      return null;
    }
    try {
      return unwrapPackLookupResult(fn.call(packCache, key));
    } catch (_) {
      return null;
    }
  };
  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i];
    const hit =
      tryCall(packCache.getWgsl, key) ||
      tryCall(packCache.getSource, key) ||
      tryCall(packCache.getShaderSource, key) ||
      tryCall(packCache.lookup, key) ||
      tryCall(packCache.get, key);
    if (hit) {
      return hit;
    }
  }
  return null;
}

function resolvePackCacheHasKey(packCache, hash) {
  if (!packCache || typeof packCache !== 'object') {
    return { matched: false, key: null };
  }
  const hasFn =
    typeof packCache.has === 'function'
      ? packCache.has
      : typeof packCache.hasShader === 'function'
        ? packCache.hasShader
        : null;
  if (!hasFn) {
    return { matched: false, key: null };
  }
  const big = normalizeShaderHashBigInt(hash);
  if (big === null) {
    return { matched: false, key: null };
  }
  const hex = formatShaderHashHex(big);
  try {
    if (hasFn.call(packCache, hex)) {
      return { matched: true, key: hex };
    }
  } catch (_) {
    // Fall through to BigInt lookup for caches that only accept BigInt keys.
  }
  try {
    if (hasFn.call(packCache, big)) {
      return { matched: true, key: big };
    }
  } catch (_) {
    // Ignore lookup errors on custom pack caches.
  }
  return { matched: false, key: null };
}

const DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES = 64 * 1024 * 1024;
const DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES = 128 * 1024 * 1024;
const SWAPCHAIN_FORMAT_TOKEN_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;

function sanitizeSwapchainFormatToken(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const token = raw.trim().toLowerCase();
  if (token.length === 0 || !SWAPCHAIN_FORMAT_TOKEN_RE.test(token)) {
    return null;
  }
  return token;
}

function readFetchResponseHeader(response, headerName) {
  if (!response || !response.headers) {
    return null;
  }
  const headers = response.headers;
  if (typeof headers.get === 'function') {
    const val = headers.get(headerName);
    return val === undefined ? null : val;
  }
  const target = headerName.toLowerCase();
  for (const [k, v] of Object.entries(headers)) {
    if (k.toLowerCase() === target) {
      return v === undefined || v === null ? null : String(v);
    }
  }
  return null;
}

function parseNonNegativeHeaderInt(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  const trimmed = String(raw).trim();
  if (!/^\d+$/.test(trimmed)) {
    return null;
  }
  const val = Number(trimmed);
  return Number.isSafeInteger(val) && val >= 0 ? val : null;
}

function buildHeadFetchInit(baseInit) {
  return baseInit && typeof baseInit === 'object'
    ? { ...baseInit, method: 'HEAD' }
    : { method: 'HEAD' };
}

function resolveStreamThresholdOption(src) {
  if (!src || typeof src !== 'object') {
    return undefined;
  }
  if (src.streamShaders === false) {
    return 0;
  }
  if (
    typeof src.streamThresholdBytes === 'number' &&
    Number.isInteger(src.streamThresholdBytes) &&
    src.streamThresholdBytes >= 0
  ) {
    return src.streamThresholdBytes;
  }
  if (src.streamShaders === true) {
    return DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES;
  }
  return undefined;
}

function resolveShaderPackStreamThresholdBytes(primary, fallback, tertiary) {
  const fromPrimary = resolveStreamThresholdOption(primary);
  if (fromPrimary !== undefined) {
    return fromPrimary;
  }
  const fromFallback = resolveStreamThresholdOption(fallback);
  if (fromFallback !== undefined) {
    return fromFallback;
  }
  const fromTertiary = resolveStreamThresholdOption(tertiary);
  if (fromTertiary !== undefined) {
    return fromTertiary;
  }
  return DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES;
}

function loadWgslPackIntoCache(packCache, buffer, packName) {
  if (!packCache) {
    return null;
  }
  if (typeof packCache.loadFromBuffer === 'function') {
    return packCache.loadFromBuffer(buffer, packName);
  }
  if (typeof packCache.loadPackBuffer === 'function') {
    return packCache.loadPackBuffer(buffer, packName);
  }
  if (typeof packCache.parse === 'function') {
    return packCache.parse(buffer, packName);
  }
  return null;
}

/**
 * Classic-worker-compatible WGPK v1 (.wgsl.pack) parser & in-memory cache.
 * Binary layout matches shaders/src/pack.rs and web/workers/wgsl_pack_loader.js byte-for-byte.
 */
const INLINE_WGSL_PACK_MAX_DECODED_BYTES = 16 * 1024 * 1024;

class InlineWgslPackCache {
  constructor() {
    this._packs = new Map();
    this._entries = new Map();
    this._decodedCache = new Map();
    this._decodedBytes = 0;
    this._maxDecodedBytes = INLINE_WGSL_PACK_MAX_DECODED_BYTES;
    this._nextPackId = 0;
    this._strictDecoder = new TextDecoder('utf-8', { fatal: true });
  }

  _toBytes(buffer) {
    if (buffer instanceof Uint8Array) {
      return buffer;
    }
    if (ArrayBuffer.isView(buffer)) {
      return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }
    if (
      buffer instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && buffer instanceof SharedArrayBuffer)
    ) {
      return new Uint8Array(buffer);
    }
    throw new TypeError('WGPK buffer must be an ArrayBuffer, SharedArrayBuffer, or ArrayBufferView');
  }

  _decodeEntryWgsl(hash, packBytes, offset, length) {
    const cached = this._decodedCache.get(hash);
    if (cached !== undefined) {
      this._decodedCache.delete(hash);
      this._decodedCache.set(hash, cached);
      return cached;
    }
    const slice = packBytes.subarray(offset, offset + length);
    const isShared =
      typeof SharedArrayBuffer !== 'undefined' && packBytes.buffer instanceof SharedArrayBuffer;
    const wgsl = this._strictDecoder.decode(isShared ? slice.slice() : slice);
    const bytes = wgsl.length * 2;
    if (bytes > 0 && bytes <= this._maxDecodedBytes) {
      while (this._decodedBytes + bytes > this._maxDecodedBytes && this._decodedCache.size > 0) {
        const oldestKey = this._decodedCache.keys().next().value;
        const oldestVal = this._decodedCache.get(oldestKey);
        this._decodedCache.delete(oldestKey);
        if (typeof oldestVal === 'string') {
          this._decodedBytes = Math.max(0, this._decodedBytes - oldestVal.length * 2);
        }
      }
      this._decodedCache.set(hash, wgsl);
      this._decodedBytes += bytes;
    }
    return wgsl;
  }

  loadPackBuffer(buffer, packName) {
    const bytes = this._toBytes(buffer);
    if (bytes.byteLength < WGPK_HEADER_BYTES) {
      throw new Error(
        `WGPK pack truncated: ${bytes.byteLength} bytes (minimum header: ${WGPK_HEADER_BYTES})`
      );
    }
    if (
      bytes[0] !== WGPK_MAGIC_BYTES[0] ||
      bytes[1] !== WGPK_MAGIC_BYTES[1] ||
      bytes[2] !== WGPK_MAGIC_BYTES[2] ||
      bytes[3] !== WGPK_MAGIC_BYTES[3]
    ) {
      throw new Error('Invalid WGPK pack magic (expected "WGPK")');
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = view.getUint32(4, true);
    if (version !== WGPK_VERSION) {
      throw new Error(`Unsupported WGPK pack version ${version} (expected ${WGPK_VERSION})`);
    }
    const entryCount = view.getUint32(8, true);
    const flags = view.getUint32(12, true);
    const dataStart = WGPK_HEADER_BYTES + entryCount * WGPK_ENTRY_BYTES;
    if (bytes.byteLength < dataStart) {
      throw new Error(
        `WGPK pack truncated: ${bytes.byteLength} bytes (entry table requires ${dataStart})`
      );
    }

    const parsedEntries = new Array(entryCount);
    const ranges = [];
    const isShared =
      typeof SharedArrayBuffer !== 'undefined' && bytes.buffer instanceof SharedArrayBuffer;
    const selfCache = this;

    for (let idx = 0; idx < entryCount; idx += 1) {
      const base = WGPK_HEADER_BYTES + idx * WGPK_ENTRY_BYTES;
      const hash = view.getBigUint64(base, true);
      const stageCode = bytes[base + 8];
      const stage = WGPK_STAGE_NAMES[stageCode];
      if (!stage) {
        throw new Error(`Invalid WGPK shader stage value ${stageCode} at entry ${idx}`);
      }
      const offset = view.getUint32(base + 12, true);
      const length = view.getUint32(base + 16, true);
      const end = offset + length;
      if (offset < dataStart || end > bytes.byteLength) {
        throw new Error(`WGPK entry ${idx} payload [${offset}..+${length}] is out of bounds`);
      }
      if (length > 0) {
        ranges.push({ start: offset, end, index: idx });
      }
      const slice = bytes.subarray(offset, end);
      // Validate strict UTF-8 eagerly without pinning decoded UTF-16 strings for every shader (#1077).
      this._strictDecoder.decode(isShared ? slice.slice() : slice);
      const entryObj = {
        hash,
        hashHex: formatShaderHashHex(hash),
        stage,
        stageCode,
        offset,
        length,
      };
      Object.defineProperty(entryObj, 'wgsl', {
        enumerable: true,
        configurable: true,
        get() {
          return selfCache._decodeEntryWgsl(hash, bytes, offset, length);
        },
      });
      parsedEntries[idx] = entryObj;
    }

    if (ranges.length > 1) {
      ranges.sort((a, b) => a.start - b.start || a.end - b.end);
      for (let i = 1; i < ranges.length; i += 1) {
        if (ranges[i].start < ranges[i - 1].end) {
          throw new Error('WGPK overlapping payload ranges detected');
        }
      }
    }

    const name =
      typeof packName === 'string' && packName.length > 0
        ? packName
        : `pack-${this._nextPackId++}`;

    if (this._packs.has(name)) {
      for (const [key, rec] of this._entries.entries()) {
        if (rec.packName === name) {
          this._entries.delete(key);
          const prevDecoded = this._decodedCache.get(key);
          if (prevDecoded !== undefined) {
            this._decodedCache.delete(key);
            this._decodedBytes = Math.max(0, this._decodedBytes - prevDecoded.length * 2);
          }
        }
      }
    }

    const packInfo = {
      name,
      version,
      flags,
      entryCount,
      byteLength: bytes.byteLength,
      entries: parsedEntries,
    };
    this._packs.set(name, packInfo);
    for (let i = 0; i < parsedEntries.length; i += 1) {
      const entry = parsedEntries[i];
      this._entries.set(entry.hash, { entry, packName: name, packBytes: bytes });
    }
    return packInfo;
  }

  loadFromBuffer(buffer, packName) {
    return this.loadPackBuffer(buffer, packName);
  }

  parse(buffer, packName) {
    return this.loadPackBuffer(buffer, packName);
  }

  hasShader(hash) {
    const key = normalizeShaderHashBigInt(hash);
    return key !== null && this._entries.has(key);
  }

  has(hash) {
    return this.hasShader(hash);
  }

  getByHash(hash) {
    const key = normalizeShaderHashBigInt(hash);
    if (key === null) return undefined;
    const rec = this._entries.get(key);
    return rec ? rec.entry : undefined;
  }

  getShaderSource(hash) {
    const key = normalizeShaderHashBigInt(hash);
    if (key === null) return undefined;
    const rec = this._entries.get(key);
    if (!rec) return undefined;
    return this._decodeEntryWgsl(rec.entry.hash, rec.packBytes, rec.entry.offset, rec.entry.length);
  }

  getSource(hash) {
    return this.getShaderSource(hash);
  }

  getWgsl(hash) {
    return this.getShaderSource(hash);
  }

  get(hash) {
    return this.getShaderSource(hash);
  }

  clear() {
    this._packs.clear();
    this._entries.clear();
    this._decodedCache.clear();
    this._decodedBytes = 0;
  }

  getStats() {
    let totalPackBytes = 0;
    for (const pack of this._packs.values()) {
      totalPackBytes += pack.byteLength;
    }
    return {
      packCount: this._packs.size,
      shaderCount: this._entries.size,
      totalPackBytes,
    };
  }
}

const OPTIONAL_ADAPTER_FEATURES = Object.freeze([
  'texture-compression-bc',
  'depth32float-stencil8',
  'float32-filterable',
  'bgra8unorm-storage',
  'rg11b10ufloat-renderable',
  'texture-formats-tier1',
  'float32-blendable',
  'dual-source-blending',
  'depth-clip-control',
]);

function selectSupportedAdapterFeatures(adapter) {
  if (!adapter || !adapter.features || typeof adapter.features.has !== 'function') {
    return [];
  }
  const supported = [];
  for (let i = 0; i < OPTIONAL_ADAPTER_FEATURES.length; i += 1) {
    const feature = OPTIONAL_ADAPTER_FEATURES[i];
    try {
      if (adapter.features.has(feature)) {
        supported.push(feature);
      }
    } catch (_) {
      // Ignore feature-probe errors from non-standard or mock adapter feature sets.
    }
  }
  return supported;
}

const OPTIONAL_ADAPTER_LIMIT_SPECS = Object.freeze([
  Object.freeze({ name: 'maxSampledTexturesPerShaderStage', defaultMin: 16, cap: 256 }),
  Object.freeze({ name: 'maxSamplersPerShaderStage', defaultMin: 16, cap: 64 }),
  Object.freeze({ name: 'maxStorageBuffersPerShaderStage', defaultMin: 8, cap: 32 }),
  Object.freeze({ name: 'maxUniformBuffersPerShaderStage', defaultMin: 12, cap: 32 }),
  Object.freeze({ name: 'maxStorageTexturesPerShaderStage', defaultMin: 4, cap: 16 }),
  Object.freeze({ name: 'maxBindingsPerBindGroup', defaultMin: 1000, cap: 4096 }),
  Object.freeze({ name: 'maxBufferSize', defaultMin: 256 * 1024 * 1024, cap: 1024 * 1024 * 1024 }),
  Object.freeze({
    name: 'maxStorageBufferBindingSize',
    defaultMin: 128 * 1024 * 1024,
    cap: 512 * 1024 * 1024,
  }),
  Object.freeze({ name: 'maxComputeWorkgroupStorageSize', defaultMin: 16384, cap: 65536 }),
]);

function selectSupportedAdapterLimits(adapter) {
  if (!adapter || typeof adapter !== 'object') {
    return {};
  }
  let limits;
  try {
    limits = adapter.limits;
  } catch (_) {
    return {};
  }
  if (!limits || typeof limits !== 'object') {
    return {};
  }
  const requiredLimits = {};
  for (let i = 0; i < OPTIONAL_ADAPTER_LIMIT_SPECS.length; i += 1) {
    const { name, defaultMin, cap } = OPTIONAL_ADAPTER_LIMIT_SPECS[i];
    let supported;
    try {
      supported = limits[name];
    } catch (_) {
      continue;
    }
    if (typeof supported !== 'number' || !Number.isFinite(supported)) {
      continue;
    }
    const clamped = Math.min(Math.floor(supported), cap);
    if (clamped > defaultMin) {
      requiredLimits[name] = clamped;
    }
  }
  return requiredLimits;
}

function deviceSupportsTimestampQuery(device) {
  if (!device || typeof device !== 'object') {
    return false;
  }
  const features = device.features;
  if (!features || typeof features.has !== 'function') {
    return false;
  }
  try {
    if (!features.has('timestamp-query')) {
      return false;
    }
  } catch (_) {
    return false;
  }
  return typeof device.createQuerySet === 'function' && typeof device.createBuffer === 'function';
}

/**
 * Request a GPUDevice from `adapter` using the optional feature / elevated limit negotiation
 * (features + limits, then features only, then a bare device). Shared by initial init and W100
 * device-loss recovery so a recovered device exposes the same capabilities.
 *
 * @param {GPUAdapter} adapter
 * @returns {Promise<{ device: GPUDevice, enabledFeatures: string[] }>}
 */
async function requestNegotiatedDevice(adapter) {
  const requiredFeatures = selectSupportedAdapterFeatures(adapter);
  const requiredLimits = selectSupportedAdapterLimits(adapter);
  const hasRequiredFeatures = requiredFeatures.length > 0;
  const hasRequiredLimits = Object.keys(requiredLimits).length > 0;
  let device;
  let enabledFeatures = [];
  if (hasRequiredFeatures || hasRequiredLimits) {
    const deviceDesc = {};
    if (hasRequiredFeatures) {
      deviceDesc.requiredFeatures = requiredFeatures;
    }
    if (hasRequiredLimits) {
      deviceDesc.requiredLimits = requiredLimits;
    }
    try {
      device = await adapter.requestDevice(deviceDesc);
      enabledFeatures = hasRequiredFeatures ? requiredFeatures.slice() : [];
    } catch (_) {
      if (hasRequiredFeatures && hasRequiredLimits) {
        try {
          device = await adapter.requestDevice({ requiredFeatures });
          enabledFeatures = requiredFeatures.slice();
        } catch (_featureErr) {
          device = await adapter.requestDevice();
          enabledFeatures = [];
        }
      } else {
        device = await adapter.requestDevice();
        enabledFeatures = [];
      }
    }
  } else {
    device = await adapter.requestDevice();
  }
  if (device && device.features && typeof device.features.has === 'function') {
    enabledFeatures = selectSupportedAdapterFeatures(device);
  }
  return { device, enabledFeatures };
}

// W103 / Wave-5: concurrent create{Render,Compute}PipelineAsync compiles allowed in flight (async_pipeline_cache.js).
const DEFAULT_MAX_CONCURRENT_PIPELINE_COMPILES = 2;

// W104: live-resource replay log used to recreate guest GPU resources after device-loss recovery.
// The guest never learns that the browser lost its GPUDevice, so every live resource creation
// packet (and a bounded shadow of small/medium buffer contents) is kept and replayed in
// dependency order on the recovered device.
const REPLAY_SHADOW_BUFFER_MAX_BYTES = 2 * 1024 * 1024;
const REPLAY_SHADOW_TOTAL_MAX_BYTES = 64 * 1024 * 1024;
const REPLAY_LOG_MAX_ENTRIES = 262144;
const REPLAY_LOG_MAX_BYTES = 256 * 1024 * 1024;
const REPLAY_LOG_MAX_BIND_GROUPS = 65536;
const REPLAY_ASYNC_CHUNK_SIZE = 24;
const REPLAY_PHASE_COUNT = 5;

// opcode -> [resource kind, replay phase (1 = leaf resources ... 5 = bind groups)]
const REPLAY_CREATE_OPCODES = new Map([
  [OPCODES.CREATE_BUFFER, ['buffer', 1]],
  [OPCODES.CREATE_TEXTURE, ['texture', 1]],
  [OPCODES.CREATE_TEXTURE2, ['texture', 1]],
  [OPCODES.CREATE_SAMPLER, ['sampler', 1]],
  [OPCODES.CREATE_SAMPLER2, ['sampler', 1]],
  [OPCODES.CREATE_SHADER_MODULE, ['shaderModule', 1]],
  [OPCODES.CREATE_BIND_GROUP_LAYOUT, ['bindGroupLayout', 2]],
  [OPCODES.CREATE_PIPELINE_LAYOUT, ['pipelineLayout', 3]],
  [OPCODES.CREATE_RENDER_PIPELINE, ['renderPipeline', 4]],
  [OPCODES.CREATE_RENDER_PIPELINE2, ['renderPipeline', 4]],
  [OPCODES.CREATE_COMPUTE_PIPELINE, ['computePipeline', 4]],
  [OPCODES.CREATE_COMPUTE_PIPELINE2, ['computePipeline', 4]],
  [OPCODES.CREATE_BIND_GROUP, ['bindGroup', 5]],
  [OPCODES.CREATE_BIND_GROUP2, ['bindGroup', 5]],
]);

const REPLAY_RESOURCE_KINDS = Object.freeze([
  'buffer',
  'texture',
  'sampler',
  'shaderModule',
  'bindGroupLayout',
  'pipelineLayout',
  'renderPipeline',
  'computePipeline',
  'bindGroup',
]);

// ABI v2 opcodes are only meaningful once the packet extension is installed.
const REPLAY_EXTENSION_OPCODES = new Set([
  OPCODES.CREATE_BIND_GROUP_LAYOUT,
  OPCODES.CREATE_PIPELINE_LAYOUT,
  OPCODES.CREATE_RENDER_PIPELINE2,
  OPCODES.CREATE_COMPUTE_PIPELINE2,
  OPCODES.CREATE_BIND_GROUP2,
  OPCODES.CREATE_SAMPLER2,
  OPCODES.CREATE_TEXTURE2,
]);

function resolvePositiveIntOption(value, fallback) {
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

// W100: bounded WebGPU device-loss recovery.
const MAX_DEVICE_RECOVERY_ATTEMPTS = 3;
const DEFAULT_DEVICE_RECOVERY_MIN_INTERVAL_MS = 1000;
const DEVICE_LOST_ERROR_PATTERN =
  /valid external instance reference no longer exists|instance (?:was |has been |is )?(?:lost|destroyed|dropped)|device (?:is |was |has been )?(?:lost|destroyed)/i;

/**
 * True when `err` looks like a lost GPUDevice / torn-down WebGPU instance (`OperationError`,
 * `A valid external Instance reference no longer exists.`, `... device is lost`).
 *
 * @param {unknown} err
 * @returns {boolean}
 */
function isDeviceLostError(err) {
  if (err === null || err === undefined) {
    return false;
  }
  let name = '';
  let message = '';
  try {
    if (typeof err === 'object') {
      name = typeof err.name === 'string' ? err.name : '';
      message = typeof err.message === 'string' ? err.message : '';
    } else if (typeof err === 'string') {
      message = err;
    }
  } catch (_) {
    return false;
  }
  return name === 'OperationError' || DEVICE_LOST_ERROR_PATTERN.test(message);
}

/**
 * Reduce a `GPUDeviceLostInfo` (`{ reason, message }`) to a sanitized single-line string.
 *
 * @param {unknown} info
 * @returns {string}
 */
function sanitizeDeviceLostInfo(info) {
  let reason = '';
  let message = '';
  try {
    if (info && typeof info === 'object') {
      reason = typeof info.reason === 'string' ? info.reason : '';
      message = typeof info.message === 'string' ? info.message : '';
    } else if (typeof info === 'string') {
      message = info;
    }
  } catch (_) {
    // Fall through with empty fields.
  }
  const safeReason = /^[A-Za-z_-]{1,32}$/.test(reason) ? reason : 'unknown';
  return sanitizeGpuErrorDetail({
    name: `DeviceLost(${safeReason})`,
    message: message || 'device lost',
  });
}

function resolveGpuRingBuffer(injectedCtor) {
  if (typeof injectedCtor === 'function') {
    return injectedCtor;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis.PlayGta6GpuRing &&
    typeof globalThis.PlayGta6GpuRing.GpuRingBuffer === 'function'
  ) {
    return globalThis.PlayGta6GpuRing.GpuRingBuffer;
  }
  if (typeof importScripts === 'function') {
    importScripts('./gpu_ring.js');
    if (
      typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6GpuRing &&
      typeof globalThis.PlayGta6GpuRing.GpuRingBuffer === 'function'
    ) {
      return globalThis.PlayGta6GpuRing.GpuRingBuffer;
    }
  }
  if (typeof require === 'function') {
    const mod = require('./gpu_ring.js');
    if (mod && typeof mod.GpuRingBuffer === 'function') {
      return mod.GpuRingBuffer;
    }
  }
  throw new Error('GpuRingBuffer unavailable; load ./gpu_ring.js or inject GpuRingBuffer');
}

function normalizeSplashPixelFormat(format, payload) {
  if (typeof format === 'string' && format.trim().length > 0) {
    const key = format.trim().toLowerCase();
    if (key === 'rgba8' || key === 'rgba' || key === 'rgba8unorm') {
      return 'rgba8';
    }
    if (key === 'yuv420p' || key === 'yuv420') {
      return 'yuv420p';
    }
    if (key === 'yuva420p' || key === 'yuva420') {
      return 'yuva420p';
    }
    if (key === 'nv12') {
      return 'nv12';
    }
    return key;
  }
  const planes = payload && payload.planes && typeof payload.planes === 'object' ? payload.planes : null;
  const hasNv12 =
    (planes && planes.y !== undefined && planes.uv !== undefined && planes.u === undefined && planes.v === undefined) ||
    (payload && payload.y !== undefined && payload.uv !== undefined && payload.u === undefined && payload.v === undefined);
  if (hasNv12) {
    return 'nv12';
  }
  const hasYuv =
    (planes && (planes.y !== undefined || planes.u !== undefined || planes.v !== undefined)) ||
    (payload && (payload.y !== undefined || payload.u !== undefined || payload.v !== undefined));
  if (hasYuv) {
    const hasAlpha =
      (planes && planes.a !== undefined && planes.a !== null) ||
      (payload && payload.a !== undefined && payload.a !== null);
    return hasAlpha ? 'yuva420p' : 'yuv420p';
  }
  return 'rgba8';
}

function normalizeBinkChromaOrderInput(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim().toLowerCase();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeBinkLoadOpInput(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim().toLowerCase();
  return trimmed.length > 0 ? trimmed : null;
}

const BINK_ENCODER_DESC = Object.freeze({
  label: 'playgta6-wgpu-worker-bink-encoder',
});

function pickPositiveIntDim(v) {
  if (typeof v === 'number' && Number.isFinite(v)) {
    const floored = Math.floor(v);
    if (floored >= 1) {
      return floored;
    }
  }
  return 0;
}

function buildMergedBinkPresenterOptions(baseOptions, extraOptions) {
  const merged = {};
  if (baseOptions && typeof baseOptions === 'object') {
    Object.assign(merged, baseOptions);
  }
  if (extraOptions && typeof extraOptions === 'object') {
    Object.assign(merged, extraOptions);
  }
  if (merged.enableTripleBuffering === undefined) {
    merged.enableTripleBuffering = true;
  }
  if (merged.allowUnalignedDirectUpload === undefined) {
    merged.allowUnalignedDirectUpload = true;
  }
  if (merged.chromaOrder !== undefined && merged.chromaOrder !== null) {
    const norm = normalizeBinkChromaOrderInput(merged.chromaOrder);
    if (norm === 'yuv' || norm === 'yvu') {
      merged.chromaOrder = norm;
    }
  }
  if (merged.loadOp !== undefined && merged.loadOp !== null) {
    const norm = normalizeBinkLoadOpInput(merged.loadOp);
    if (norm === 'clear' || norm === 'load') {
      merged.loadOp = norm;
    }
  }
  return Object.keys(merged).length > 0 ? merged : null;
}

function resolveHasVisibleNonBlackPixelsFn() {
  if (
    typeof globalThis !== 'undefined' &&
    globalThis.PlayGta6SplashFrameBridge &&
    typeof globalThis.PlayGta6SplashFrameBridge.hasVisibleNonBlackPixels === 'function'
  ) {
    return globalThis.PlayGta6SplashFrameBridge.hasVisibleNonBlackPixels;
  }
  if (typeof require === 'function') {
    try {
      const mod = require('./splash_frame_bridge.js');
      if (mod && typeof mod.hasVisibleNonBlackPixels === 'function') {
        return mod.hasVisibleNonBlackPixels;
      }
    } catch (_) {
      // Optional helper unavailable.
    }
  }
  if (typeof importScripts === 'function') {
    try {
      importScripts('./splash_frame_bridge.js');
    } catch (_) {
      // Optional helper unavailable in worker environment.
    }
    if (
      typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6SplashFrameBridge &&
      typeof globalThis.PlayGta6SplashFrameBridge.hasVisibleNonBlackPixels === 'function'
    ) {
      return globalThis.PlayGta6SplashFrameBridge.hasVisibleNonBlackPixels;
    }
  }
  return null;
}

const DEFAULT_FALLBACK_SAMPLER_DESCRIPTORS = Object.freeze({
  filtering: Object.freeze({
    label: 'fallback-sampler:filtering',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  }),
  'non-filtering': Object.freeze({
    label: 'fallback-sampler:non-filtering',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
    magFilter: 'nearest',
    minFilter: 'nearest',
    mipmapFilter: 'nearest',
  }),
  comparison: Object.freeze({
    label: 'fallback-sampler:comparison',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'nearest',
    compare: 'less-equal',
  }),
});

function normalizeFallbackSamplerKind(kind) {
  if (typeof kind !== 'string') {
    return 'filtering';
  }
  const key = kind.trim().toLowerCase();
  if (
    key === 'non-filtering' ||
    key === 'non_filtering' ||
    key === 'nonfiltering' ||
    key === 'point' ||
    key === 'nearest' ||
    key === 'nearest-clamp' ||
    key === 'point-clamp'
  ) {
    return 'non-filtering';
  }
  if (
    key === 'comparison' ||
    key === 'compare' ||
    key === 'shadow' ||
    key === 'less-equal' ||
    key === 'less_equal' ||
    key === 'pcf'
  ) {
    return 'comparison';
  }
  return 'filtering';
}

function resolveMaxConcurrentCompilesOption(options) {
  if (options && typeof options === 'object') {
    if (
      options.maxConcurrentCompiles === 0 ||
      (Number.isInteger(options.maxConcurrentCompiles) && options.maxConcurrentCompiles >= 1)
    ) {
      return options.maxConcurrentCompiles;
    }
    if (
      options.maxConcurrentPipelineCompiles === 0 ||
      (Number.isInteger(options.maxConcurrentPipelineCompiles) &&
        options.maxConcurrentPipelineCompiles >= 1)
    ) {
      return options.maxConcurrentPipelineCompiles;
    }
  }
  return DEFAULT_MAX_CONCURRENT_PIPELINE_COMPILES;
}

function isExternalImageLike(obj) {
  if (!obj || typeof obj !== 'object') {
    return false;
  }
  if (typeof ImageBitmap !== 'undefined' && obj instanceof ImageBitmap) {
    return true;
  }
  if (
    typeof obj.close === 'function' &&
    typeof obj.width === 'number' &&
    typeof obj.height === 'number' &&
    obj.planes === undefined &&
    obj.rgba === undefined &&
    obj.y === undefined
  ) {
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// G66 GPU -> CPU read-back (docs/gpu_commands.md "Read-back"): ReadbackBuffer / ReadbackTexture
// copy into a MAP_READ staging buffer in stream order; once mapped, the bytes are written to
// `dstAddr` in the runtime's memory and only then is the u32 at `doneAddr` incremented (also when
// the copy failed), which releases the fence signals the runtime held behind it.
// ---------------------------------------------------------------------------

const READBACK_BUFFER_BYTES = 40;
const READBACK_TEXTURE_BYTES = 72;

function readU64AsNumber(view, offset) {
  return Number(view.getBigUint64(offset, true));
}

/** Decodes a read-back payload; null when it is truncated. */
function decodeReadbackPayload(opcode, view, payloadOffset, payloadBytes) {
  if (opcode === OPCODES.READBACK_BUFFER) {
    if (payloadBytes < READBACK_BUFFER_BYTES) return null;
    return {
      kind: 'buffer',
      seq: view.getUint32(payloadOffset, true),
      bufferId: view.getUint32(payloadOffset + 4, true),
      srcOffset: readU64AsNumber(view, payloadOffset + 8),
      size: readU64AsNumber(view, payloadOffset + 16),
      dstAddr: readU64AsNumber(view, payloadOffset + 24),
      doneAddr: readU64AsNumber(view, payloadOffset + 32),
    };
  }
  if (opcode === OPCODES.READBACK_TEXTURE) {
    if (payloadBytes < READBACK_TEXTURE_BYTES) return null;
    const u32 = (i) => view.getUint32(payloadOffset + 4 * i, true);
    return {
      kind: 'texture',
      seq: u32(0),
      textureId: u32(1),
      mipLevel: u32(2),
      originX: u32(3),
      originY: u32(4),
      originZ: u32(5),
      width: u32(6),
      height: u32(7),
      depthOrLayers: u32(8),
      bytesPerRow: u32(9),
      rowsPerImage: u32(10),
      dstAddr: readU64AsNumber(view, payloadOffset + 48),
      doneAddr: readU64AsNumber(view, payloadOffset + 56),
      size: readU64AsNumber(view, payloadOffset + 64),
    };
  }
  return null;
}

/**
 * Staging layout of a read-back: the bytes the GPU copy writes, and for textures the rows
 * (WebGPU wants 256-byte aligned rows; the guest footprint has its own pitch).
 * `rowBytes` is the width of one row of texel blocks (the dst_size of the last row).
 */
function readbackStagingLayout(req, blockHeight = 1) {
  if (req.kind === 'buffer') {
    const start = req.srcOffset - (req.srcOffset % 4);
    const end = req.srcOffset + req.size;
    const alignedEnd = end + ((4 - (end % 4)) % 4);
    return { copyOffset: start, skip: req.srcOffset - start, stagingSize: Math.max(4, alignedEnd - start) };
  }
  const bh = Math.max(1, blockHeight | 0);
  const rows = Math.max(1, Math.ceil(req.height / bh));
  const depth = Math.max(1, req.depthOrLayers);
  const pitch = req.bytesPerRow;
  const rowBytes = req.size - ((depth - 1) * req.rowsPerImage + (rows - 1)) * pitch;
  const stagingRow = Math.ceil(Math.max(1, rowBytes) / 256) * 256;
  return {
    rows,
    depth,
    rowBytes,
    stagingRow,
    stagingRowsPerImage: rows,
    stagingSize: stagingRow * rows * depth,
  };
}

/** Repacks mapped staging bytes into the guest layout: returns [{ offset, bytes }] pieces. */
function readbackPieces(req, layout, mapped) {
  if (req.kind === 'buffer') {
    return [{ offset: 0, bytes: mapped.subarray(layout.skip, layout.skip + req.size) }];
  }
  const pieces = [];
  if (!(layout.rowBytes > 0) || layout.rowBytes > req.bytesPerRow) return pieces;
  for (let z = 0; z < layout.depth; z += 1) {
    for (let r = 0; r < layout.rows; r += 1) {
      const src = (z * layout.stagingRowsPerImage + r) * layout.stagingRow;
      const dst = (z * req.rowsPerImage + r) * req.bytesPerRow;
      if (dst + layout.rowBytes > req.size) return pieces;
      pieces.push({ offset: dst, bytes: mapped.subarray(src, src + layout.rowBytes) });
    }
  }
  return pieces;
}

/**
 * Writes the pieces into `memory` (the runtime's SharedArrayBuffer / ArrayBuffer) at req.dstAddr,
 * then increments the u32 completion counter at req.doneAddr. `pieces` may be null (failed copy:
 * the counter still advances). Returns true when the bytes were written.
 */
function deliverReadback(memory, req, pieces) {
  if (!memory || typeof memory.byteLength !== 'number') return false;
  let written = false;
  if (pieces && req.dstAddr + req.size <= memory.byteLength) {
    const dst = new Uint8Array(memory, req.dstAddr, req.size);
    for (const piece of pieces) {
      dst.set(piece.bytes, piece.offset);
    }
    written = true;
  }
  if (req.doneAddr % 4 === 0 && req.doneAddr + 4 <= memory.byteLength) {
    const counter = new Uint32Array(memory, req.doneAddr, 1);
    if (typeof SharedArrayBuffer !== 'undefined' && memory instanceof SharedArrayBuffer) {
      Atomics.add(counter, 0, 1);
      Atomics.notify(new Int32Array(memory, req.doneAddr, 1), 0);
    } else {
      counter[0] += 1;
    }
  }
  return written;
}

function textureFormatBlockHeight(format) {
  return typeof format === 'string' && /^(bc|etc2|eac|astc)/.test(format) ? 4 : 1;
}

class WgpuCommandProcessor {
  constructor(device, ctx, options = {}) {
    this.device = device;
    this.ctx = ctx;
    this.format = options.format || 'bgra8unorm';
    this._preferredSdrCanvasFormat = SWAPCHAIN_SDR_BASE_FORMATS.has(this.format)
      ? this.format
      : 'bgra8unorm';
    this._streamSwapchainFormat = null;
    this._swapchainViewFormat = null;
    this._alphaMode = normalizeSwapchainAlphaMode(
      options.alphaMode ?? (options.swapchainOptions && options.swapchainOptions.alphaMode)
    );
    this._swapchainHdrMode = normalizeSwapchainHdrMode(
      options.hdrMode ??
        options.swapchainHdrMode ??
        (options.swapchainOptions && options.swapchainOptions.hdrMode),
      'auto'
    );
    this._directSrgbView = Boolean(
      options.directSrgbView === true ||
        (options.swapchainOptions && options.swapchainOptions.directSrgbView === true)
    );
    this._swapchainToneMapMode = normalizeSwapchainToneMapMode(
      options.toneMapMode ?? (options.swapchainOptions && options.swapchainOptions.toneMapMode)
    );
    this._swapchainExposure = normalizeSwapchainExposure(
      options.exposure ?? (options.swapchainOptions && options.swapchainOptions.exposure),
      1.0
    );
    this._swapchainApplySrgbEncode =
      typeof options.applySrgbEncode === 'boolean'
        ? options.applySrgbEncode
        : options.swapchainOptions && typeof options.swapchainOptions.applySrgbEncode === 'boolean'
          ? options.swapchainOptions.applySrgbEncode
          : null;
    this._swapchainRgb10a2Transfer =
      (options.rgb10a2Transfer ??
        (options.swapchainOptions && options.swapchainOptions.rgb10a2Transfer)) === 'linear'
        ? 'linear'
        : 'sdr';
    this._activeToneMapMode = 'none';
    this._activeApplySrgbEncode = false;
    this.buffers = new Map();
    this.textures = new Map();
    this.textureViews = new Map();
    // G66 read-back: runtime memory (attachGuestMemory), copies recorded this frame, and counters.
    this._guestMemory = options.guestMemory || null;
    this._recordedReadbacks = [];
    this._readbackFlushWanted = false;
    this._earlyFlushedThisFrame = false;
    this._earlyReadbackFlushes = 0;
    this.readbackStats = { requested: 0, delivered: 0, failed: 0, bytes: 0 };
    this.samplers = new Map();
    this.bindGroups = new Map();
    this._bindGroupKinds = new Map();
    this.shaderModules = new Map();
    this.pipelines = new Map();
    this.computePipelines = new Map();
    this._pipelineBindGroupLayouts = new Map();
    this._computePipelineBindGroupLayouts = new Map();
    this._pipelineReflectionGroups = new Map();
    this._computePipelineReflectionGroups = new Map();
    this._defaultFallbackSamplers = new Map();
    this._fallbackSamplersCreated = 0;
    this._fallbackSamplerBindingsSynthesized = 0;
    this._shaderReflectionKeys = new Map();
    this.frameCount = 0;
    this.ring = null;
    this._GpuRingBuffer = typeof options.GpuRingBuffer === 'function' ? options.GpuRingBuffer : null;
    this._decoder = new TextDecoder('utf-8');
    this._destroyed = false;

    // Viewport & splash/first-present tracking
    this._viewportWidth =
      typeof options.width === 'number' && Number.isFinite(options.width) && options.width > 0
        ? Math.floor(options.width)
        : 0;
    this._viewportHeight =
      typeof options.height === 'number' && Number.isFinite(options.height) && options.height > 0
        ? Math.floor(options.height)
        : 0;
    this._onFirstPresent =
      typeof options.onFirstPresent === 'function' ? options.onFirstPresent : null;
    this._firstPresentNotified = false;
    this._firstPresentMs = null;
    this._onFirstVisibleFrame =
      typeof options.onFirstVisibleFrame === 'function' ? options.onFirstVisibleFrame : null;
    this._firstVisibleFrameListeners = new Set();
    this._firstVisibleFrameNotified = false;
    this._firstVisibleFrameMs = null;
    this._blackLoadingMs = null;
    this._requireNonBlackFrameExplicit =
      options.requireNonBlackFrame === false || options.requireNonBlackPixels === false
        ? false
        : options.requireNonBlackFrame === true || options.requireNonBlackPixels === true
          ? true
          : null;
    this._requireNonBlackFrame = this._requireNonBlackFrameExplicit === true;
    this._minVisibleLuma =
      typeof options.minVisibleLuma === 'number' &&
      Number.isFinite(options.minVisibleLuma) &&
      options.minVisibleLuma >= 0 &&
      options.minVisibleLuma <= 255
        ? options.minVisibleLuma
        : SWAPCHAIN_PROBE_DEFAULT_LUMA_THRESHOLD;
    this._minVisiblePixels = resolvePositiveIntOption(options.minVisiblePixels, 4);
    this._nonBlackPixelProbe =
      typeof options.nonBlackPixelProbe === 'function'
        ? options.nonBlackPixelProbe
        : typeof options.pixelProbe === 'function'
          ? options.pixelProbe
          : null;
    this._suppressedBlackFrames = 0;
    this._lastSampledMaxLuma = 0;
    this._pendingFrameLuminance = null;
    this._lumaProbeScratch = {
      visible: false,
      maxLuma: 0,
      visiblePixels: 0,
      sampledPixels: 0,
    };
    // W338: Throttled, non-overlapping async swapchain luminance readback probe state.
    this._swapchainProbeIntervalMs =
      typeof options.swapchainProbeIntervalMs === 'number' &&
      Number.isFinite(options.swapchainProbeIntervalMs) &&
      options.swapchainProbeIntervalMs >= 0
        ? options.swapchainProbeIntervalMs
        : SWAPCHAIN_PROBE_DEFAULT_INTERVAL_MS;
    this._swapchainProbeTimeoutMs =
      options.swapchainProbeTimeoutMs === 0
        ? 0
        : typeof options.swapchainProbeTimeoutMs === 'number' &&
            Number.isFinite(options.swapchainProbeTimeoutMs) &&
            options.swapchainProbeTimeoutMs > 0
          ? Math.floor(options.swapchainProbeTimeoutMs)
          : SWAPCHAIN_PROBE_DEFAULT_TIMEOUT_MS;
    this._swapchainProbeTimeoutTimer = null;
    this._swapchainProbeTimeoutCount = 0;
    this._swapchainProbeBuffer = null;
    this._swapchainProbeBufferDevice = null;
    this._swapchainProbeSupported = null;
    this._swapchainProbeInFlight = false;
    this._swapchainProbeLastStartMs = -Infinity;
    this._swapchainProbeCount = 0;
    this._swapchainProbeErrorCount = 0;
    this._swapchainProbeConsecutiveErrors = 0;
    this._swapchainProbeFallbackWarned = false;
    this._swapchainProbeGeneration = 0;
    this._uniformRingFrameIndex = -1;
    this._stagingSceneTransitionTrimBytes = 0;
    this._frameStartMs = null;
    this._gpuAdapter =
      options.gpuAdapter && typeof options.gpuAdapter === 'object'
        ? Object.freeze(sanitizeGpuAdapterInfo(options.gpuAdapter))
        : null;
    this._ownsDrawExecutor = false;
    this.drawExecutor =
      options.drawExecutor && typeof options.drawExecutor === 'object'
        ? options.drawExecutor
        : null;
    this._reuseDrawResult = options.reuseDrawResult !== false;
    if (this.drawExecutor && this._reuseDrawResult) {
      try {
        if (typeof this.drawExecutor.setReuseDrawResult === 'function') {
          this.drawExecutor.setReuseDrawResult(true);
        } else if ('_reuseDrawResult' in this.drawExecutor) {
          this.drawExecutor._reuseDrawResult = true;
        }
      } catch (_) {
        // Ignore frozen or read-only custom DrawExecutor mocks.
      }
    }
    this._frameDrawSummaryOpts = Object.freeze({ reuse: true });
    this._frameSummaryScratch = {
      hasNonClearDraw: false,
      realPackDrawsThisFrame: 0,
      fallbackDrawsThisFrame: 0,
      drawsThisFrame: 0,
    };
    this.resourceTable =
      options.resourceTable && typeof options.resourceTable === 'object'
        ? options.resourceTable
        : null;
    this._zeroAllocDrains = 0;
    this._zeroAllocDrainOpts = { reuseResult: true, directView: true };
    this._zeroAllocDrainPresents = 0;
    this._zeroAllocDrainSpanBytes = 0;
    this._cachedCmdBuffer = null;
    this._cachedCmdView = null;
    this._drainRingResultScratch = { frameCount: 0, presents: 0, bytesDrained: 0 };
    this._executeResultScratch = { frameCount: 0, presents: 0 };
    // W363: Pre-allocated scratch descriptors for _handleWriteTexture (zero per-packet object alloc).
    this._wtOriginScratch = { x: 0, y: 0, z: 0 };
    this._wtDstScratch = { texture: null, mipLevel: 0, origin: this._wtOriginScratch };
    this._wtSizeScratch = { width: 0, height: 0, depthOrArrayLayers: 1 };
    this._wtLayoutOffsetOnlyScratch = { offset: 0 };
    this._wtLayoutBprScratch = { offset: 0, bytesPerRow: 0 };
    this._wtLayoutRpiScratch = { offset: 0, rowsPerImage: 0 };
    this._wtLayoutFullScratch = { offset: 0, bytesPerRow: 0, rowsPerImage: 0 };
    this._wtStageLayoutScratch = {
      width: 0,
      height: 0,
      depthOrArrayLayers: 1,
      mipLevel: 0,
      origin: this._wtOriginScratch,
      bytesPerRow: undefined,
      rowsPerImage: undefined,
    };
    this._probeCopyOriginScratch = { x: 0, y: 0, z: 0 };
    this._probeCopySrcScratch = {
      texture: null,
      mipLevel: 0,
      origin: this._probeCopyOriginScratch,
    };
    this._probeCopyDstScratch = {
      buffer: null,
      offset: 0,
      bytesPerRow: SWAPCHAIN_PROBE_BYTES_PER_ROW,
      rowsPerImage: 1,
    };
    this._probeCopySizeScratch = {
      width: 1,
      height: 1,
      depthOrArrayLayers: 1,
    };
    this._splashTargetSizeScratch = { width: 1280, height: 720 };
    this._binkDirectPresentOptsScratch = {};
    this._uniformRingEndFrameOptsScratch = { fence: null };
    this._onWorkDoneFenceResolved = () => undefined;
    this._onWorkDoneFenceRejected = (err) => {
      this._noteWorkDoneFenceFailure(err);
      return undefined;
    };
    this._onZeroAllocDrainSpan = (bytesView, byteOffset, byteLength, span) => {
      this._zeroAllocDrainSpanBytes = byteLength;
      const coalesceRing = this._coalescePresentsEnabled !== false;
      if (this.executeCommandBuffer !== WgpuCommandProcessor.prototype.executeCommandBuffer) {
        const viewBytes =
          byteOffset === 0 && byteLength === bytesView.byteLength
            ? bytesView
            : bytesView.subarray(byteOffset, byteOffset + byteLength);
        const execRes = this.executeCommandBuffer(viewBytes, DRAIN_RING_REUSE_OPTS);
        this._zeroAllocDrainPresents += execRes.presents;
        return;
      }
      if (span && span.dataView instanceof DataView) {
        const viewStart = typeof span.viewOffset === 'number' ? span.viewOffset : 0;
        this._zeroAllocDrainPresents += this._executeCommandView(
          span.dataView,
          viewStart,
          viewStart + byteLength,
          coalesceRing,
          false
        );
        return;
      }
      const view = this._getCachedDataView(bytesView.buffer);
      const viewStart = bytesView.byteOffset + byteOffset;
      this._zeroAllocDrainPresents += this._executeCommandView(
        view,
        viewStart,
        viewStart + byteLength,
        coalesceRing,
        false
      );
    };
    // W359: Reusable per-frame pass command pool, binding maps/slots, and present scratch state
    // so steady-state 60 Hz draw & present loops incur zero per-packet/per-frame JS allocations.
    this._clearColor = DEFAULT_CLEAR_COLOR;
    this._lastCustomClearColor = null;
    this._reusablePassCommands = [];
    this._passCommands = this._reusablePassCommands;
    this._passCommandPool = [];
    this._passCommandPoolUsed = 0;
    this._setBindGroup2MetaPool = [];
    this._setBindGroup2MetaPoolUsed = 0;
    this._boundBindGroups = new Map();
    this._boundVertexBuffers = new Map();
    this._boundBindGroupSlots = new Map();
    this._boundVertexBufferSlots = new Map();
    this._fallbackProfileStateScratch = {
      vertexCount: 0,
      instanceCount: 0,
      vertexStride: 0,
      bindGroups: this._boundBindGroups,
      vertexBuffers: this._boundVertexBuffers,
    };
    this._frameHintsScratch = {
      sceneTransition: false,
      resolutionChanged: false,
      isIdleFrame: false,
    };
    this._markFrameContentScratch = {
      hasNonClearDraw: false,
      nonClearDraw: false,
      hasVisibleContent: false,
      hasDraws: false,
      realPackDraws: 0,
      fallbackDraws: 0,
      drawsThisFrame: 0,
      timestampMs: 0,
    };
    this._presentState = {
      encoder: null,
      activeSwapchain: null,
      swapchainTexture: null,
      swapchainView: null,
      activePass: null,
      activePassKind: null,
      renderPassOpened: false,
      renderPassIdx: 0,
      computePassIdx: 0,
      timerSupported: false,
      activePipelineId: null,
      activePipeline: null,
      activeComputePipelineId: null,
      specializedActiveKey: null,
      cachedFrameSummary: null,
    };
    const presentState = this._presentState;
    this._passControl = {
      encoder: null,
      swapchainTexture: null,
      swapchainView: null,
      canvasWidth: undefined,
      canvasHeight: undefined,
      swapchainWidth: undefined,
      swapchainHeight: undefined,
      get pass() {
        return presentState.activePass;
      },
      get passKind() {
        return presentState.activePassKind;
      },
      endPass: () => this._endPresentActivePass(),
      beginRenderPass: (descriptor) => this._beginExtensionRenderPass(descriptor),
    };
    this._reclaimedOversizedBytes = 0;
    this._oversizedRtReclaimIdleFrames = resolvePositiveIntOption(
      options.oversizedRtReclaimIdleFrames,
      DEFAULT_OVERSIZED_RT_RECLAIM_IDLE_FRAMES
    );
    this._oversizedRtReclaimIntervalFrames = resolvePositiveIntOption(
      options.oversizedRtReclaimIntervalFrames,
      DEFAULT_OVERSIZED_RT_RECLAIM_INTERVAL_FRAMES
    );
    this._lastPresentSource = null;
    this._lastPresentWidth = 0;
    this._lastPresentHeight = 0;
    this._pendingSceneTransition = false;
    this._frameDrawsThisFrame = 0;
    this._frameRealPackDrawsThisFrame = 0;
    this._frameFallbackDrawsThisFrame = 0;
    this._frameNonClearDrawsThisFrame = 0;
    this._splashFramesPresented = 0;
    this._lastSplashPresentMs = null;

    // Telemetry counters
    this._packetCount = 0;
    this._drawCallCount = 0;
    this._realPackDrawCount = 0;
    this._fallbackDrawCount = 0;
    this._presentCount = 0;
    this._unboundBindGroupCount = 0;
    this._skippedDrawCount = 0;
    this._computeDispatchCount = 0;
    this._bytesProcessed = 0;
    this._chunkedWriteBufferUploads = 0;
    // W99: torn-down/lost GPU instance hardening counters (fence + present submit failures).
    this._workDoneFenceFailures = 0;
    this._presentSubmitFailures = 0;
    // W100: device-loss detection / recovery state (hooked at the end of the constructor).
    this._deviceGeneration = 0;
    this._deviceLostGeneration = -1;
    this._deviceLostHookedDevice = null;
    this._deviceLostCount = 0;
    this._deviceRecoveredCount = 0;
    this._deviceRecoveryFailedCount = 0;
    this._lastDeviceLostReason = null;
    this._deviceLost = false;
    this._deviceRecovering = false;
    this._droppedWhileRecoveringCount = 0;
    this._deviceLossListener =
      typeof options.onDeviceLost === 'function' ? options.onDeviceLost : null;
    this._logger =
      options.logger !== undefined ? options.logger : typeof console !== 'undefined' ? console : null;
    this._ownedSubsystemFactories = {};
    this._v2BindGroupLayouts = new Map();
    this._v2PipelineLayouts = new Map();

    // W104: live-resource replay log + bounded buffer shadow cache for device-loss recovery.
    this._resourceReplayEnabled = options.enableResourceReplay !== false;
    this._shadowBufferReplayEnabled =
      options.enableShadowBufferReplay !== undefined
        ? options.enableShadowBufferReplay !== false
        : true;
    this._replayingResources = false;
    this._replayAsyncActive = false;
    this._replayAsyncDeltaKeys = new Set();
    this._replayAsyncDestroyedIds = new Set();
    this._replayAsyncShadowIds = new Set();
    this._replayLog = new Map(); // '<kind>:<id>' -> { kind, id, opcode, phase, payload: Uint8Array }
    this._replayLogBytes = 0;
    this._replayLogMaxEntries = resolvePositiveIntOption(
      options.replayLogMaxEntries,
      REPLAY_LOG_MAX_ENTRIES
    );
    this._replayLogMaxBytes = resolvePositiveIntOption(options.replayLogMaxBytes, REPLAY_LOG_MAX_BYTES);
    this._replayLogMaxBindGroups = resolvePositiveIntOption(
      options.replayMaxBindGroups,
      REPLAY_LOG_MAX_BIND_GROUPS
    );
    this._replayBindGroupKeys = new Map(); // bindGroupId -> 'bindGroup:<id>' in LRU order (oldest first)
    this._replayLogOverflowCount = 0;
    this._replayBindGroupEvictions = 0;
    this._replayEvictionWarnNext = 1;
    this._replayOverflowWarnNext = 1;
    this._shadowBuffers = new Map(); // bufferId -> { data: Uint8Array, dirtyEnd } (LRU order)
    this._shadowBufferBytes = 0;
    this._shadowBufferMaxBytes = resolvePositiveIntOption(
      options.shadowBufferMaxBytes,
      REPLAY_SHADOW_BUFFER_MAX_BYTES
    );
    this._shadowBufferTotalMaxBytes = Math.max(
      this._shadowBufferMaxBytes,
      resolvePositiveIntOption(options.shadowBufferTotalMaxBytes, REPLAY_SHADOW_TOTAL_MAX_BYTES)
    );
    this._shadowBufferEvictions = 0;
    this._replayedResourceCount = 0;
    this._replayedBufferWriteCount = 0;
    this._replayFailedCount = 0;
    this._lastResourceReplay = null;

    // W106: lazy GPUShaderModule compilation — defer device.createShaderModule({ code })
    // until a pipeline actually references the shader id so 4,000+ preload shaders never stall
    // Chrome's GPU process or trigger watchdog loss during boot or device recovery.
    // Values: false (eager), 'pack' (defer pack-resolved WGSL), true (defer all shaders).
    this._lazyShaderCompile =
      options.lazyShaderCompile === 'pack' ? 'pack' : Boolean(options.lazyShaderCompile);
    this._lazyRangedShaderCompile = Boolean(options.lazyRangedShaderCompile);
    this._deferredShaderCount = 0;
    this._materializedShaderCount = 0;
    this._cachedBuiltinShaderModules = new Map();

    // Wave-4 (#663, #1077): Hard WGSL single-shader byte-size cap, bounded/time-sliced ranged
    // createShaderModule compilation, deferred ranged fallback pipelines, and Present coalescing.
    this._maxWgslCompileBytes =
      options.maxWgslCompileBytes === 0
        ? 0
        : resolvePositiveIntOption(options.maxWgslCompileBytes, DEFAULT_MAX_WGSL_COMPILE_BYTES);
    this._oversizedWgslFallbacks = 0;
    this._oversizedShaderHashes = new Set();
    this._maxConcurrentShaderCompiles =
      options.maxConcurrentShaderCompiles === 0
        ? 0
        : resolvePositiveIntOption(
            options.maxConcurrentShaderCompiles,
            DEFAULT_MAX_CONCURRENT_SHADER_COMPILES
          );
    this._shaderCompileSliceBudgetMs =
      options.shaderCompileSliceBudgetMs === 0
        ? 0
        : resolvePositiveIntOption(
            options.shaderCompileSliceBudgetMs,
            DEFAULT_SHADER_COMPILE_SLICE_BUDGET_MS
          );
    this._maxPendingRangedPipelines = resolvePositiveIntOption(
      options.maxPendingRangedPipelines,
      DEFAULT_MAX_PENDING_RANGED_PIPELINES
    );
    this._deferRangedFallbackPipelines = options.deferRangedFallbackPipelines !== false;
    this._rangedShaderCompileQueue = [];
    this._rangedShaderCompileSeq = 0;
    this._activeRangedShaderCompiles = 0;
    this._peakConcurrentRangedShaderCompiles = 0;
    this._rangedShaderCompileYields = 0;
    this._rangedShaderCompilePumpTimer = null;
    this._rangedShaderCompileMicrotaskPending = false;
    this._rangedShaderWaitingForPipelineIdle = false;
    this._coalescePresentsEnabled = options.coalescePresents !== false;
    this._coalescedPresents = 0;
    this._presentScanScratch = { count: 0, lastOffset: -1 };
    // Wave-5 (#663, #1077): Rate-limited GPU stall / hang attribution warnings.
    this._gpuStallWarnMs =
      options.gpuStallWarnMs === 0
        ? 0
        : typeof options.gpuStallWarnMs === 'number' &&
            Number.isFinite(options.gpuStallWarnMs) &&
            options.gpuStallWarnMs > 0
          ? options.gpuStallWarnMs
          : DEFAULT_GPU_STALL_WARN_MS;
    this._gpuStallWarnMinIntervalMs =
      options.gpuStallWarnMinIntervalMs === 0
        ? 0
        : typeof options.gpuStallWarnMinIntervalMs === 'number' &&
            Number.isFinite(options.gpuStallWarnMinIntervalMs) &&
            options.gpuStallWarnMinIntervalMs > 0
          ? options.gpuStallWarnMinIntervalMs
          : DEFAULT_GPU_STALL_WARN_MIN_INTERVAL_MS;
    this._gpuStallWarnCount = 0;
    this._gpuStallWarnLastByOp = new Map();
    // Zero-allocation per-opcode timing scratch for executeCommandBuffer stall attribution.
    this._stallProfileActive = false;
    this._stallProfileDetailed = false;
    this._stallExecStartMs = 0;
    this._stallLastProgressMs = 0;
    this._stallOpCount = new Uint32Array(65);
    this._stallOpMs = new Float64Array(65);
    this._stallOpMaxMs = new Float64Array(65);
    this._stallOpBytes = new Float64Array(65);
    this._stallBatchPackets = 0;
    this._stallBatchBytes = 0;
    this._stallReplayMs = 0;
    this._stallSlowPacketLogs = 0;
    this._stallPresentBreakdown = {
      count: 0,
      getTexMs: 0,
      cmdsMs: 0,
      blitMs: 0,
      probeMs: 0,
      finishMs: 0,
      submitMs: 0,
      otherMs: 0,
    };

    // W136 / W159: on-demand HTTP 206 Range shader resolution + deferred/upgradable pipelines.
    this._nextShaderModuleGen = 0;
    this._shaderModuleGenerations = new Map();
    this._inFlightRangedShaders = new Map();
    this._pendingRangedPipelines = new Map();
    this._rangedShaderFetches = 0;
    this._rangedShaderReady = 0;
    this._rangedShaderErrors = 0;

    // Pending staging upload encoder & frame lifecycle tracking
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._frameBegunForIndex = -1;
    this._lastPassTimings = null;
    this._pendingTimerReadbacks = new Set();

    // W342: Mid-frame deferred GPU destroy queue (drained after queue.submit / onSubmittedWorkDone)
    // and 1D -> 2D LUT texture promotion (enabled by default; opt-out via options.promote1DTo2D === false).
    this._deferredGpuDestroys = [];
    this._deferredGpuDestroysSet = new Set();
    this._inFlightDeferredDestroys = new Set();
    this._deferredResourceBytes = typeof WeakMap === 'function' ? new WeakMap() : null;
    this._deferredDestroyCount = 0;
    this._pendingDeferredDestroyBytes = 0;
    this._deferredHighWaterFlushes = 0;
    this._maxDeferredGpuDestroys = resolvePositiveIntOption(
      options.maxDeferredGpuDestroys,
      DEFAULT_MAX_DEFERRED_GPU_DESTROYS
    );
    this._maxDeferredGpuDestroyBytes = resolvePositiveIntOption(
      options.maxDeferredGpuDestroyBytes,
      DEFAULT_MAX_DEFERRED_GPU_DESTROY_BYTES
    );
    // W371 (#1077): Accurate live & peak GPU buffer/texture byte accounting even without resourceRegistry.
    this._liveBufferBytes = 0;
    this._peakBufferBytes = 0;
    this._liveTextureBytes = 0;
    this._peakTextureBytes = 0;
    this._textureMeta = new Map();
    this._textureObjMeta = new WeakMap();
    this._directWriteScratch = null;
    this._clearZeroBuffer = null;
    this._clearZeroBufferBytes = 0;
    this._clearZeroScratch = null;
    this._clearTextureStats = {
      cleared: 0,
      missing: 0,
      invalid: 0,
      renderPassClears: 0,
      copyClears: 0,
      writeTextureClears: 0,
      copies: 0,
    };
    this._promote1DTo2D = options.promote1DTo2D !== false;
    this._promoted1DTo2DCount = 0;

    // Optional WebGpuPipelineCache for reflection layout deduplication & W167 fallback sampler synthesis
    this._ownsPipelineCache = false;
    this.pipelineCache = null;
    if (options.pipelineCache && typeof options.pipelineCache === 'object') {
      this.pipelineCache = options.pipelineCache;
    } else if (options.enablePipelineCache === true && device && typeof device === 'object') {
      const PipelineCacheCtor = resolveSubsystemCtor(
        options.WebGpuPipelineCache,
        'PlayGta6PipelineCache',
        'WebGpuPipelineCache',
        './pipeline_cache.js'
      );
      if (PipelineCacheCtor) {
        this.pipelineCache = new PipelineCacheCtor({
          ...(options.pipelineCacheOptions || {}),
          device,
        });
        this._ownsPipelineCache = true;
      }
    }

    // W103 / W168: priority-scheduled, bounded-concurrency async PSO compilation.
    // render_state_packets.js routes CreateRenderPipeline2 / CreateComputePipeline2 / v1
    // CreateComputePipeline through it when the device exposes create*PipelineAsync, compiling
    // graphics pipelines ahead of background compute during shader bursts.
    this._pendingAsyncPipelineSkips = 0;
    this.asyncPipelineCache = null;
    if (options.asyncPipelineCache && typeof options.asyncPipelineCache === 'object') {
      this.asyncPipelineCache = options.asyncPipelineCache;
    } else if (options.enableAsyncPipelines !== false) {
      const AsyncPipelineCacheCtor = resolveSubsystemCtor(
        options.AsyncPipelineCache,
        'PlayGta6AsyncPipelineCache',
        'AsyncPipelineCache',
        './async_pipeline_cache.js'
      );
      if (AsyncPipelineCacheCtor) {
        const maxConcurrentCompiles = resolveMaxConcurrentCompilesOption(options);
        const prioritizeGraphics = options.prioritizeGraphics !== false;
        const asyncCacheOpts = {
          device,
          maxConcurrentCompiles,
          prioritizeGraphics,
          logger: this._logger ? this._createAsyncPipelineLogger() : null,
          onStall: (ev) => {
            if (!ev || this._gpuStallWarnMs <= 0 || ev.elapsedMs <= this._gpuStallWarnMs) {
              return;
            }
            this._logDiagnostic(
              'warn',
              `[wgpu_worker] GPU stall: op=asyncPipeline(${ev.kind}) durationMs=${Math.round(ev.elapsedMs)} id=${ev.id} failed=${Boolean(ev.failed)} retried=${Boolean(ev.retried)} frame=${this.frameCount}`
            );
          },
        };
        if (maxConcurrentCompiles > 0) {
          asyncCacheOpts.maxConcurrent = maxConcurrentCompiles;
        }
        if (typeof options.now === 'function') {
          asyncCacheOpts.nowFn = options.now;
        }
        this.asyncPipelineCache = new AsyncPipelineCacheCtor(asyncCacheOpts);
      }
    }

    // Saved subsystem options / constructors for lazy construction where supported
    this._reflectionRegistryOptions = options.reflectionRegistryOptions || null;
    this._ReflectionRegistryCtor = options.ReflectionRegistry || null;
    this._rtPoolOptions = options.rtPoolOptions || null;
    this._RenderTargetPoolCtor = options.RenderTargetPool || null;
    this._rtPoolExplicitlyDisabled = options.enableRtPool === false;

    // 1. GpuStagingPool (G14)
    this._ownsStagingPool = false;
    this.stagingPool = null;
    if (options.stagingPool && typeof options.stagingPool === 'object') {
      this.stagingPool = options.stagingPool;
    } else if (
      options.enableStagingPool === true &&
      device &&
      typeof device.createBuffer === 'function'
    ) {
      const StagingCtor = resolveSubsystemCtor(
        options.GpuStagingPool,
        'PlayGta6StagingPool',
        'GpuStagingPool',
        './staging_pool.js'
      );
      if (StagingCtor) {
        this._ownedSubsystemFactories.stagingPool = (dev) =>
          new StagingCtor({ ...(options.stagingPoolOptions || {}), device: dev });
        this.stagingPool = this._ownedSubsystemFactories.stagingPool(device);
        this._ownsStagingPool = true;
      }
    }

    // 2. DynamicUniformRing (G18)
    this._ownsUniformRing = false;
    this.uniformRing = null;
    if (options.uniformRing && typeof options.uniformRing === 'object') {
      this.uniformRing = options.uniformRing;
    } else if (
      options.enableUniformRing === true &&
      device &&
      typeof device.createBuffer === 'function'
    ) {
      const UniformRingCtor = resolveSubsystemCtor(
        options.DynamicUniformRing,
        'PlayGta6UniformRing',
        'DynamicUniformRing',
        './uniform_ring.js'
      );
      if (UniformRingCtor) {
        this._ownedSubsystemFactories.uniformRing = (dev) =>
          new UniformRingCtor({ ...(options.uniformRingOptions || {}), device: dev });
        this.uniformRing = this._ownedSubsystemFactories.uniformRing(device);
        this._ownsUniformRing = true;
      }
    }

    // 3. GpuPassTimer (G16 / W324)
    this._ownsPassTimer = false;
    this.passTimer = null;
    const injectedPassTimer =
      options.passTimer && typeof options.passTimer === 'object'
        ? options.passTimer
        : options.gpuTimer && typeof options.gpuTimer === 'object'
          ? options.gpuTimer
          : null;
    const enablePassTimerOpt =
      options.enablePassTimer !== undefined ? options.enablePassTimer : options.enableGpuTimer;
    const passTimerOpts = options.passTimerOptions || options.gpuTimerOptions || null;
    if (injectedPassTimer) {
      this.passTimer = injectedPassTimer;
    } else if (
      enablePassTimerOpt === true ||
      (enablePassTimerOpt !== false && deviceSupportsTimestampQuery(device))
    ) {
      const PassTimerCtor = resolveSubsystemCtor(
        options.GpuPassTimer,
        'PlayGta6GpuTimer',
        'GpuPassTimer',
        './gpu_timer.js'
      );
      if (PassTimerCtor) {
        this._ownedSubsystemFactories.passTimer = (dev) =>
          new PassTimerCtor({ ...(passTimerOpts || {}), device: dev });
        this.passTimer = this._ownedSubsystemFactories.passTimer(device);
        this._ownsPassTimer = true;
      }
    }

    // 3b. GpuFrameRecorder (W325 30 FPS spike auto-capture)
    this._ownsGpuRecorder = false;
    this.gpuRecorder = null;
    if (options.gpuRecorder && typeof options.gpuRecorder === 'object') {
      this.gpuRecorder = options.gpuRecorder;
    } else if (options.recorder && typeof options.recorder === 'object') {
      this.gpuRecorder = options.recorder;
    } else if (options.enableGpuRecorder === true) {
      const RecorderCtor = resolveSubsystemCtor(
        options.GpuFrameRecorder,
        'PlayGta6GpuRecorder',
        'GpuFrameRecorder',
        './gpu_recorder.js'
      );
      if (RecorderCtor) {
        this.gpuRecorder = new RecorderCtor(options.gpuRecorderOptions || {});
        this._ownsGpuRecorder = true;
      }
    }

    // 4. ReflectionRegistry (G17)
    this._ownsReflectionRegistry = false;
    this.reflectionRegistry = null;
    if (options.reflectionRegistry && typeof options.reflectionRegistry === 'object') {
      this.reflectionRegistry = options.reflectionRegistry;
    } else if (options.enableReflectionRegistry === true) {
      const ReflectionCtor = resolveSubsystemCtor(
        options.ReflectionRegistry,
        'PlayGta6ReflectionLoader',
        'ReflectionRegistry',
        './reflection_loader.js'
      );
      if (ReflectionCtor) {
        this.reflectionRegistry = new ReflectionCtor(options.reflectionRegistryOptions || {});
        this._ownsReflectionRegistry = true;
      }
    }

    // 5. GpuResourceRegistry (G21)
    this._ownsResourceRegistry = false;
    this.resourceRegistry = null;
    if (options.resourceRegistry && typeof options.resourceRegistry === 'object') {
      this.resourceRegistry = options.resourceRegistry;
    } else if (options.enableResourceRegistry === true) {
      const RegistryCtor = resolveSubsystemCtor(
        options.GpuResourceRegistry,
        'PlayGta6ResourceRegistry',
        'GpuResourceRegistry',
        './resource_registry.js'
      );
      if (RegistryCtor) {
        this.resourceRegistry = new RegistryCtor(options.resourceRegistryOptions || {});
        this._ownsResourceRegistry = true;
      }
    }

    // 6. RenderTargetPool (G22)
    this._ownsRtPool = false;
    this.rtPool = null;
    if (options.rtPool && typeof options.rtPool === 'object') {
      this.rtPool = options.rtPool;
    } else if (
      options.enableRtPool === true &&
      device &&
      typeof device.createTexture === 'function'
    ) {
      const RtPoolCtor = resolveSubsystemCtor(
        options.RenderTargetPool,
        'PlayGta6RenderTargetPool',
        'RenderTargetPool',
        './rt_pool.js'
      );
      if (RtPoolCtor) {
        this._ownedSubsystemFactories.rtPool = (dev) =>
          new RtPoolCtor({ ...(options.rtPoolOptions || {}), device: dev });
        this.rtPool = this._ownedSubsystemFactories.rtPool(device);
        this._ownsRtPool = true;
      }
    }

    if (options.reflectionManifest !== undefined && options.reflectionManifest !== null) {
      this.loadReflectionManifest(options.reflectionManifest);
    }

    // 7. WgslPackCache + ShaderFallbackRouter (G28 / S12 / W109 / W121 / W159)
    this._shaderPackStreamThresholdBytes = resolveShaderPackStreamThresholdBytes(options, null);
    this._activeShaderPackStreamThresholdBytes = this._shaderPackStreamThresholdBytes;
    this._ownsWgslPackCache = false;
    this.wgslPackCache = null;
    if (options.wgslPackCache && typeof options.wgslPackCache === 'object') {
      this.wgslPackCache = options.wgslPackCache;
    } else {
      this.wgslPackCache = new InlineWgslPackCache();
      this._ownsWgslPackCache = true;
    }

    this._ownsShaderFallbackRouter = false;
    this.shaderFallbackRouter = null;
    if (options.shaderFallbackRouter && typeof options.shaderFallbackRouter === 'object') {
      this.shaderFallbackRouter = options.shaderFallbackRouter;
      if (
        options.wgslPackCache &&
        this.shaderFallbackRouter._packLoader == null &&
        typeof this.shaderFallbackRouter.setPackLoader === 'function'
      ) {
        this.shaderFallbackRouter.setPackLoader(this.wgslPackCache);
      }
    } else if (options.enableShaderFallback !== false) {
      const FallbackRouterCtor = resolveSubsystemCtor(
        options.ShaderFallbackRouter,
        'PlayGta6ShaderFallbackRouter',
        'ShaderFallbackRouter',
        './shader_fallback_router.js'
      );
      if (FallbackRouterCtor) {
        const routerOpts = {
          ...(options.shaderFallbackRouterOptions || {}),
          packLoader: this.wgslPackCache,
        };
        if (
          options.fallbackShaderHashes &&
          typeof options.fallbackShaderHashes === 'object' &&
          routerOpts.fallbackHashes === undefined
        ) {
          routerOpts.fallbackHashes = options.fallbackShaderHashes;
        }
        this.shaderFallbackRouter = new FallbackRouterCtor(routerOpts);
        this._ownsShaderFallbackRouter = true;
      }
    }

    // W282: Optional owned DrawExecutor configured with reuseDrawResult: true for zero-alloc draws.
    if (
      !this.drawExecutor &&
      options.enableDrawExecutor === true &&
      device &&
      typeof device === 'object'
    ) {
      const DrawExecutorCtor = resolveSubsystemCtor(
        options.DrawExecutor,
        'PlayGta6DrawExecutor',
        'DrawExecutor',
        './draw_executor.js'
      );
      if (DrawExecutorCtor) {
        const execOpts =
          options.drawExecutorOptions && typeof options.drawExecutorOptions === 'object'
            ? { ...options.drawExecutorOptions }
            : {};
        if (execOpts.reuseDrawResult === undefined) {
          execOpts.reuseDrawResult = this._reuseDrawResult;
        }
        execOpts.device = device;
        execOpts.format = execOpts.format || this.format;
        if (execOpts.pipelineCache === undefined && this.pipelineCache) {
          execOpts.pipelineCache = this.pipelineCache;
        }
        if (execOpts.shaderFallbackRouter === undefined && this.shaderFallbackRouter) {
          execOpts.shaderFallbackRouter = this.shaderFallbackRouter;
        }
        this.drawExecutor = new DrawExecutorCtor(execOpts);
        this._ownsDrawExecutor = true;
      }
    }

    if (options.wgslPackBuffer !== undefined && options.wgslPackBuffer !== null) {
      this.loadWgslPack(options.wgslPackBuffer);
    }

    // 8. BinkVideoPresenter (W38 / W165) + SplashFrameBridge (G26)
    this._BinkVideoPresenterCtor =
      options.BinkVideoPresenter !== undefined ? options.BinkVideoPresenter : undefined;
    this._SplashFrameBridgeCtor =
      options.SplashFrameBridge !== undefined ? options.SplashFrameBridge : undefined;
    const topLevelBinkOpts = {};
    const initBinkChromaOrder =
      options.binkChromaOrder !== undefined ? options.binkChromaOrder : options.chromaOrder;
    if (initBinkChromaOrder !== undefined && initBinkChromaOrder !== null) {
      topLevelBinkOpts.chromaOrder = initBinkChromaOrder;
    }
    const initBinkLoadOp =
      options.binkLoadOp !== undefined ? options.binkLoadOp : options.loadOp;
    if (initBinkLoadOp !== undefined && initBinkLoadOp !== null) {
      topLevelBinkOpts.loadOp = initBinkLoadOp;
    }
    if (options.enableTripleBuffering !== undefined) {
      topLevelBinkOpts.enableTripleBuffering = Boolean(options.enableTripleBuffering);
    }
    this._binkPresenterOptions = buildMergedBinkPresenterOptions(
      topLevelBinkOpts,
      options.binkPresenterOptions
    );
    this._splashBridgeOptions =
      options.splashFrameBridgeOptions || options.splashBridgeOptions || null;
    this._splashBridgeExplicitlyDisabled = options.enableSplashBridge === false;

    this._ownsBinkPresenter = false;
    this.binkPresenter = null;
    if (options.binkPresenter && typeof options.binkPresenter === 'object') {
      this.binkPresenter = options.binkPresenter;
      if (this._binkPresenterOptions) {
        if (
          this._binkPresenterOptions.chromaOrder !== undefined &&
          typeof this.binkPresenter.setChromaOrder === 'function'
        ) {
          this.binkPresenter.setChromaOrder(this._binkPresenterOptions.chromaOrder);
        }
        if (
          this._binkPresenterOptions.loadOp !== undefined &&
          typeof this.binkPresenter.setLoadOp === 'function'
        ) {
          this.binkPresenter.setLoadOp(this._binkPresenterOptions.loadOp);
        }
      }
    }

    this._ownsSplashFrameBridge = false;
    this.splashFrameBridge = null;
    if (options.splashFrameBridge && typeof options.splashFrameBridge === 'object') {
      this.splashFrameBridge = options.splashFrameBridge;
      if (!this.binkPresenter && this.splashFrameBridge.presenter) {
        this.binkPresenter = this.splashFrameBridge.presenter;
      }
    } else if (
      options.enableSplashBridge === true &&
      device &&
      typeof device.createTexture === 'function'
    ) {
      this._ensureSplashBridge();
    }

    // Packet ABI v2 handler ({ handlePacket, destroyResource }), see render_state_packets.js.
    this.extension = null;

    // 9. Frame-1 fallback shader profile tracking & cross-buffer specialization (G34 / G38 / G44)
    this._defaultFallbackProfile = normalizeFallbackProfileName(options.defaultFallbackProfile);
    this._explicitFallbackDrawContext = null;
    this._bufferMeta = new Map();
    this._pipelineMeta = new Map();
    this._bindGroupMeta = new Map();
    this._specializedFallbackPipelines = new Map();
    this._specializedFallbackBindGroups = new Map();
    this._specializedFallbackShaderModules = new Map();
    this._pendingCommandStreamShaderProfiles = null;
    this._pendingCommandStreamView = null;
    this._pendingCommandStreamStartOffset = 0;
    this._pendingCommandStreamEndOffset = 0;
    this._pendingCommandStreamScanned = true;
    this._fallbackProfileScanCount = 0;
    this._fallbackProfileCounts = {
      [FALLBACK_PROFILE.DEFAULT]: 0,
      [FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV]: 0,
      [FALLBACK_PROFILE.TEXTURED_QUAD_24B]: 0,
      [FALLBACK_PROFILE.SPLASH_RECT_24B]: 0,
    };
    this._lastFallbackProfile = null;

    // 10. Live WebGPU validation error capture (G35 / G44)
    this._validationNow = typeof options.now === 'function' ? options.now : defaultNowMs;
    const rawMaxErrHistory =
      options.maxValidationErrorHistory !== undefined
        ? options.maxValidationErrorHistory
        : options.maxErrorHistory !== undefined
          ? options.maxErrorHistory
          : DEFAULT_MAX_VALIDATION_ERROR_HISTORY;
    this._maxValidationErrorHistory =
      Number.isInteger(rawMaxErrHistory) && rawMaxErrHistory >= 1
        ? rawMaxErrHistory
        : DEFAULT_MAX_VALIDATION_ERROR_HISTORY;
    const rawMaxErrPosts = options.maxValidationErrorPosts;
    this._maxValidationErrorPosts =
      Number.isInteger(rawMaxErrPosts) && rawMaxErrPosts >= 1
        ? rawMaxErrPosts
        : DEFAULT_MAX_VALIDATION_ERROR_POSTS;
    this._validationErrors = [];
    this._validationErrorCount = 0;
    this._validationErrorSeq = 0;
    this._validationErrorPostedCount = 0;
    this._validationErrorSuppressedCount = 0;
    this._validationErrorDedup = new Map();
    this._validationErrorCallbacks = new Set();
    if (typeof options.onValidationError === 'function') {
      this._validationErrorCallbacks.add(options.onValidationError);
    }
    this._captureValidationErrors = Boolean(
      options.captureValidationErrors === true ||
      options.enableValidationErrors === true ||
      typeof options.onValidationError === 'function' ||
      (options.swapchain && typeof options.swapchain === 'object')
    );
    this._validationErrorHookedDevice = null;
    this._boundDeviceUncapturedErrorHandler = null;
    this._swapchain = null;
    this._ownsSwapchain = false;
    this._swapchainValidationAttached = false;
    this._swapchainOriginalOnValidationError = null;
    this._swapchainWrappedOnValidationError = null;

    if (options.swapchain && typeof options.swapchain === 'object') {
      this.attachSwapchain(options.swapchain, { owned: options.ownsSwapchain === true });
    } else if (this._captureValidationErrors) {
      this._hookDeviceUncapturedError(this.device);
    }

    // W100: observe `device.lost` for the initial device.
    this._hookDeviceLost(this.device);

    this._resetFrameState();
  }

  get destroyed() {
    return this._destroyed;
  }

  get framesPresented() {
    return this.frameCount;
  }

  get presentCount() {
    return this._presentCount;
  }

  get realPackDrawCount() {
    return this._realPackDrawCount;
  }

  get fallbackDrawCount() {
    return this._fallbackDrawCount;
  }

  get unboundBindGroupCount() {
    return this._unboundBindGroupCount;
  }

  get skippedDrawCount() {
    return this._skippedDrawCount;
  }

  get swapchain() {
    return this._swapchain;
  }

  get firstVisibleFrame() {
    return this._firstVisibleFrameNotified;
  }

  get firstVisibleFrameMs() {
    return this._firstVisibleFrameMs;
  }

  get firstPresentMs() {
    return this._firstPresentMs;
  }

  get blackLoadingMs() {
    return this._blackLoadingMs;
  }

  get gpuTimer() {
    return this.passTimer;
  }

  set gpuTimer(timer) {
    this.passTimer = timer;
  }

  get requireNonBlackFrame() {
    return this._requireNonBlackFrame;
  }

  get suppressedBlackFrames() {
    return this._suppressedBlackFrames;
  }

  get lastSampledMaxLuma() {
    return this._lastSampledMaxLuma;
  }

  get swapchainProbeCount() {
    return this._swapchainProbeCount;
  }

  get swapchainProbeInFlight() {
    return this._swapchainProbeInFlight;
  }

  get swapchainProbeSupported() {
    return this._swapchainProbeSupported;
  }

  get swapchainProbeErrorCount() {
    return this._swapchainProbeErrorCount;
  }

  get swapchainProbeConsecutiveErrors() {
    return this._swapchainProbeConsecutiveErrors;
  }

  get swapchainProbeTimeoutCount() {
    return this._swapchainProbeTimeoutCount;
  }

  get gpuStallWarnCount() {
    return this._gpuStallWarnCount;
  }

  setFrameLuminanceMeta(meta) {
    this._pendingFrameLuminance = meta !== undefined ? meta : null;
  }

  get gpuAdapter() {
    return this._gpuAdapter;
  }

  get chunkedWriteBufferUploads() {
    return this._chunkedWriteBufferUploads;
  }

  setGpuAdapter(adapterInfo) {
    this._gpuAdapter =
      adapterInfo && typeof adapterInfo === 'object'
        ? Object.freeze(sanitizeGpuAdapterInfo(adapterInfo))
        : null;
  }

  addFirstVisibleFrameListener(listener) {
    if (typeof listener !== 'function') {
      return () => {};
    }
    this._firstVisibleFrameListeners.add(listener);
    return () => {
      this._firstVisibleFrameListeners.delete(listener);
    };
  }

  get validationErrorCount() {
    return this._validationErrorCount;
  }

  get rangedShaderFetches() {
    return this._rangedShaderFetches;
  }

  get rangedShaderReady() {
    return this._rangedShaderReady;
  }

  get rangedShaderErrors() {
    return this._rangedShaderErrors;
  }

  get rangedPackThresholdBytes() {
    return this._activeShaderPackStreamThresholdBytes;
  }

  get rangedPackActive() {
    const cacheStats =
      this.wgslPackCache && typeof this.wgslPackCache.getStats === 'function'
        ? this.wgslPackCache.getStats()
        : null;
    const mode =
      cacheStats && typeof cacheStats.mode === 'string'
        ? cacheStats.mode.trim().toLowerCase()
        : 'buffer';
    return mode === 'ranged' || mode === 'mixed';
  }

  get asyncShaderPendingCount() {
    return this._inFlightRangedShaders.size;
  }

  _ensureValidationErrorHooked() {
    if (this._destroyed) {
      return false;
    }
    if (this._swapchainValidationAttached) {
      return true;
    }
    return this._hookDeviceUncapturedError(this.device);
  }

  /**
   * Register a callback notified on every captured WebGPU validation/OOM/internal error.
   * Returns an unsubscribe function.
   *
   * @param {(error: { seq: number, type: string, message: string, timestampMs: number }, totalCount: number) => void} callback
   * @returns {() => void}
   */
  addValidationErrorListener(callback) {
    if (typeof callback !== 'function') {
      throw new TypeError('addValidationErrorListener requires a callback function');
    }
    this._validationErrorCallbacks.add(callback);
    this._ensureValidationErrorHooked();
    return () => {
      this._validationErrorCallbacks.delete(callback);
      if (
        this._validationErrorCallbacks.size === 0 &&
        !this._captureValidationErrors &&
        !this._swapchain
      ) {
        this._unhookDeviceUncapturedError();
      }
    };
  }

  /**
   * Return shallow copies of captured sanitized WebGPU validation errors in the bounded history ring.
   *
   * @returns {Array<{ seq: number, type: string, message: string, timestampMs: number }>}
   */
  getValidationErrors() {
    return this._validationErrors.map((entry) => ({ ...entry }));
  }

  /**
   * Clear the bounded validation error history ring and deduplication table
   * (monotonic counter stays intact).
   */
  clearValidationErrors() {
    this._validationErrors.length = 0;
    this._validationErrorDedup.clear();
    this._validationErrorPostedCount = 0;
  }

  _unhookDeviceUncapturedError() {
    const prevDevice = this._validationErrorHookedDevice;
    const prevHandler = this._boundDeviceUncapturedErrorHandler;
    this._validationErrorHookedDevice = null;
    this._boundDeviceUncapturedErrorHandler = null;

    if (
      prevDevice &&
      prevHandler &&
      typeof prevDevice.removeEventListener === 'function'
    ) {
      try {
        prevDevice.removeEventListener('uncapturederror', prevHandler);
      } catch (_) {
        // Ignore removal failures on torn-down or mock devices.
      }
    }
  }

  _hookDeviceUncapturedError(device) {
    if (!device || typeof device !== 'object') {
      this._unhookDeviceUncapturedError();
      return false;
    }
    if (device === this._validationErrorHookedDevice && this._boundDeviceUncapturedErrorHandler) {
      return true;
    }
    this._unhookDeviceUncapturedError();
    if (typeof device.addEventListener !== 'function') {
      return false;
    }
    const handler = (event) => {
      if (this._destroyed || this.device !== device) {
        return;
      }
      this._recordValidationError(event);
    };
    this._validationErrorHookedDevice = device;
    this._boundDeviceUncapturedErrorHandler = handler;
    device.addEventListener('uncapturederror', handler);
    return true;
  }

  _unhookSwapchainValidation() {
    const prevSwapchain = this._swapchain;
    if (prevSwapchain && this._swapchainValidationAttached) {
      if (
        '_onValidationError' in prevSwapchain &&
        prevSwapchain._onValidationError === this._swapchainWrappedOnValidationError
      ) {
        prevSwapchain._onValidationError = this._swapchainOriginalOnValidationError;
      }
    }
    this._swapchainValidationAttached = false;
    this._swapchainOriginalOnValidationError = null;
    this._swapchainWrappedOnValidationError = null;
  }

  /**
   * Bind a WgpuSwapchain to this processor and wire its `onValidationError` hook without
   * double-hooking `device.addEventListener('uncapturederror', ...)`.
   *
   * @param {object | null | undefined} swapchain
   * @param {{ owned?: boolean }} [opts]
   */
  attachSwapchain(swapchain, opts = {}) {
    this._unhookSwapchainValidation();
    this._unhookDeviceUncapturedError();

    if (!swapchain || typeof swapchain !== 'object') {
      this._swapchain = null;
      this._ownsSwapchain = false;
      if (this._captureValidationErrors || this._validationErrorCallbacks.size > 0) {
        this._hookDeviceUncapturedError(this.device);
      }
      return null;
    }

    this._swapchain = swapchain;
    this._ownsSwapchain = Boolean(opts && opts.owned);

    if (!this.device && swapchain.device) {
      this.device = swapchain.device;
    }
    if (!this.ctx && swapchain.context) {
      this.ctx = swapchain.context;
    }

    if ('_onValidationError' in swapchain) {
      const prevCb =
        typeof swapchain._onValidationError === 'function' ? swapchain._onValidationError : null;
      const wrappedCb = (record) => {
        if (prevCb) {
          try {
            prevCb(record);
          } catch (_) {
            // Isolate prior callback failures.
          }
        }
        this._recordValidationError(record);
      };
      swapchain._onValidationError = wrappedCb;
      this._swapchainOriginalOnValidationError = prevCb;
      this._swapchainWrappedOnValidationError = wrappedCb;
      this._swapchainValidationAttached = true;

      // If a custom mock swapchain exposes `_onValidationError` without hooking the device itself,
      // fall back to attaching directly on `device`.
      const swapchainHookedDevice =
        swapchain._errorHookedDevice === this.device ||
        typeof swapchain._hookUncapturedError === 'function';
      if (!swapchainHookedDevice) {
        this._hookDeviceUncapturedError(this.device);
      }
    } else if (typeof swapchain.setOnValidationError === 'function') {
      swapchain.setOnValidationError((record) => {
        this._recordValidationError(record);
      });
      this._swapchainValidationAttached = true;
    } else {
      this._hookDeviceUncapturedError(this.device);
    }

    return this._swapchain;
  }

  /**
   * Bind a newly acquired or recovered GPUDevice and re-attach live validation error capture.
   *
   * @param {GPUDevice} newDevice
   * @param {{ format?: string, recoverSwapchain?: boolean }} [opts]
   */
  bindDevice(newDevice, opts = {}) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    this.device = newDevice;
    if (opts && typeof opts.format === 'string' && opts.format.length > 0) {
      this.format = opts.format;
      this._syncBinkPresentationFormat(this.format);
    }
    if (this.asyncPipelineCache && this.asyncPipelineCache.device !== newDevice) {
      this._resetAsyncPipelineCache(newDevice);
    }
    this._specializedFallbackPipelines.clear();
    this._specializedFallbackBindGroups.clear();

    if (
      this._swapchain &&
      !this._swapchain.isDestroyed &&
      typeof this._swapchain.recover === 'function' &&
      opts.recoverSwapchain !== false
    ) {
      this._swapchain.recover(newDevice, opts.format);
      if (!this._swapchainValidationAttached && (this._captureValidationErrors || this._validationErrorCallbacks.size > 0)) {
        this._hookDeviceUncapturedError(newDevice);
      }
    } else if (this._captureValidationErrors || this._validationErrorCallbacks.size > 0) {
      this._hookDeviceUncapturedError(newDevice);
    } else {
      this._unhookDeviceUncapturedError();
    }
  }

  _recordValidationError(eventOrRecord) {
    if (this._destroyed) {
      return null;
    }
    const rawError =
      eventOrRecord && typeof eventOrRecord === 'object' && 'error' in eventOrRecord
        ? eventOrRecord.error
        : eventOrRecord;
    const type = normalizeValidationErrorType(rawError, eventOrRecord);

    let rawMessage = '';
    if (rawError && typeof rawError === 'object' && typeof rawError.message === 'string') {
      rawMessage = rawError.message;
    } else if (
      eventOrRecord &&
      typeof eventOrRecord === 'object' &&
      typeof eventOrRecord.message === 'string'
    ) {
      rawMessage = eventOrRecord.message;
    } else if (typeof rawError === 'string') {
      rawMessage = rawError;
    } else if (rawError !== null && rawError !== undefined) {
      rawMessage = String(rawError);
    }

    const message = sanitizeValidationErrorMessage(rawMessage);
    let timestampMs;
    if (
      rawError &&
      typeof rawError === 'object' &&
      typeof rawError.timestampMs === 'number' &&
      Number.isFinite(rawError.timestampMs)
    ) {
      timestampMs = rawError.timestampMs;
    } else if (
      eventOrRecord &&
      typeof eventOrRecord === 'object' &&
      typeof eventOrRecord.timestampMs === 'number' &&
      Number.isFinite(eventOrRecord.timestampMs)
    ) {
      timestampMs = eventOrRecord.timestampMs;
    } else {
      const nowVal = this._validationNow();
      timestampMs = typeof nowVal === 'number' && Number.isFinite(nowVal) ? nowVal : Date.now();
    }

    this._validationErrorCount += 1;

    const dedupKey = `${type}|${message}`;
    const existingDedup = this._validationErrorDedup.get(dedupKey);
    if (existingDedup) {
      existingDedup.count += 1;
      this._validationErrorSuppressedCount += 1;
      return null;
    }

    if (this._validationErrorPostedCount >= this._maxValidationErrorPosts) {
      if (this._validationErrorDedup.size < MAX_VALIDATION_ERROR_DEDUP_ENTRIES) {
        this._validationErrorDedup.set(dedupKey, { count: 1, seq: 0, capped: true });
      }
      this._validationErrorSuppressedCount += 1;
      return null;
    }

    this._validationErrorSeq += 1;
    this._validationErrorPostedCount += 1;
    if (this._validationErrorDedup.size >= MAX_VALIDATION_ERROR_DEDUP_ENTRIES) {
      const oldestKey = this._validationErrorDedup.keys().next().value;
      if (oldestKey !== undefined) {
        this._validationErrorDedup.delete(oldestKey);
      }
    }
    this._validationErrorDedup.set(dedupKey, { count: 1, seq: this._validationErrorSeq });

    const record = {
      seq: this._validationErrorSeq,
      type,
      message,
      timestampMs,
    };

    this._validationErrors.push(record);
    if (this._validationErrors.length > this._maxValidationErrorHistory) {
      this._validationErrors.shift();
    }

    const totalCount = this._validationErrorCount;
    for (const cb of Array.from(this._validationErrorCallbacks)) {
      try {
        cb({ ...record }, totalCount);
      } catch (_) {
        // Isolate callback failures so validation error reporting never crashes the worker.
      }
    }

    return { ...record };
  }

  _recordOpValidationError(operation, errOrMessage, errorType = 'validation') {
    if (this._destroyed) {
      return null;
    }
    const opPrefix = typeof operation === 'string' && operation.length > 0 ? operation : 'WebGPU';
    let detail;
    if (errOrMessage && typeof errOrMessage === 'object' && typeof errOrMessage.message === 'string') {
      detail = errOrMessage.message;
    } else if (typeof errOrMessage === 'string') {
      detail = errOrMessage;
    } else if (errOrMessage !== null && errOrMessage !== undefined) {
      detail = String(errOrMessage);
    } else {
      detail = 'operation failed';
    }
    const trimmed = detail.trim();
    const message =
      trimmed.length === 0
        ? `${opPrefix}: operation failed`
        : trimmed.toLowerCase().startsWith(`${opPrefix.toLowerCase()}:`)
          ? trimmed
          : `${opPrefix}: ${trimmed}`;
    return this._recordValidationError({
      error: {
        type: errorType,
        message,
      },
    });
  }

  _getRenderStateExtension() {
    let ext = this.extension;
    while (ext) {
      if (ext.bindGroupLayouts instanceof Map && ext.pipelineLayouts instanceof Map) {
        return ext;
      }
      ext = ext.next || null;
    }
    return null;
  }

  _getTexturePacketsExtension() {
    let ext = this.extension;
    while (ext) {
      if (
        typeof ext.getStats === 'function' &&
        (ext._bc3dFallbackTextures instanceof Set ||
          (ext.stats &&
            typeof ext.stats === 'object' &&
            ext.stats.bc3dFallbacks !== undefined))
      ) {
        return ext;
      }
      ext = ext.next || null;
    }
    return null;
  }

  _getRenderTargetFormat() {
    if (
      this._swapchain &&
      !this._swapchain.isDestroyed &&
      this._swapchain.usesBlitFallback === true &&
      typeof this._swapchain.streamFormat === 'string' &&
      this._swapchain.streamFormat.length > 0
    ) {
      return this._swapchain.streamFormat;
    }
    return this._swapchainViewFormat || this.format;
  }

  /**
   * Select a G34 fallback shader profile for a draw state (`fullscreen_triangle_cbv`,
   * `textured_quad_24b`, or `default`).
   */
  selectFallbackShaderProfile(drawState) {
    return selectFallbackShaderProfile(drawState);
  }

  /**
   * Set or clear an explicit draw-state hint used for subsequent missing-shader resolutions.
   *
   * @param {object | null | undefined} drawState
   */
  setFallbackDrawContext(drawState) {
    this._explicitFallbackDrawContext =
      drawState && typeof drawState === 'object' ? { ...drawState } : null;
  }

  /**
   * Resolve a missing/untranslatable shader hash via `shaderFallbackRouter` using the detected
   * Frame-1 fallback profile, remaining 100% compatible with legacy 2-arg routers.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} stage
   * @param {object | string | null | undefined} [drawStateOrOptions]
   */
  resolveFallbackShader(hash, stage, drawStateOrOptions) {
    if (!this.shaderFallbackRouter || typeof this.shaderFallbackRouter.resolve !== 'function') {
      return null;
    }
    const stageName =
      typeof stage === 'number' ? stageCodeToName(stage) || 'vertex' : String(stage || 'vertex');
    let profile = this._defaultFallbackProfile;
    if (typeof drawStateOrOptions === 'string') {
      profile = normalizeFallbackProfileName(drawStateOrOptions);
    } else if (drawStateOrOptions && typeof drawStateOrOptions === 'object') {
      if (typeof drawStateOrOptions.profile === 'string') {
        profile = normalizeFallbackProfileName(drawStateOrOptions.profile);
      } else {
        profile = selectFallbackShaderProfile(drawStateOrOptions);
      }
    } else if (this._explicitFallbackDrawContext) {
      profile = selectFallbackShaderProfile(this._explicitFallbackDrawContext);
    }

    const big = normalizeShaderHashBigInt(hash);
    const lookupHashHex = big !== null ? formatShaderHashHex(big) : String(hash);
    const resolved = callFallbackRouterResolve(
      this.shaderFallbackRouter,
      lookupHashHex,
      stageName,
      profile
    );
    if (resolved && resolved.source === 'builtin') {
      const effectiveProfile = normalizeFallbackProfileName(resolved.profile || profile);
      this._recordFallbackProfileUsage(effectiveProfile);
      resolved.profile = effectiveProfile;
    }
    return resolved;
  }

  _recordFallbackProfileUsage(profile) {
    const canonical = normalizeFallbackProfileName(profile);
    this._fallbackProfileCounts[canonical] = (this._fallbackProfileCounts[canonical] || 0) + 1;
    this._lastFallbackProfile = canonical;
  }

  _invalidateSpecializedPipelinesFor(pipelineId) {
    if (
      this._specializedFallbackPipelines.size === 0 &&
      this._specializedFallbackBindGroups.size === 0
    ) {
      return;
    }
    const prefix = `${pipelineId}|`;
    if (this._specializedFallbackPipelines.size > 0) {
      for (const key of this._specializedFallbackPipelines.keys()) {
        if (key.startsWith(prefix)) {
          this._specializedFallbackPipelines.delete(key);
        }
      }
    }
    if (this._specializedFallbackBindGroups.size > 0) {
      for (const key of this._specializedFallbackBindGroups.keys()) {
        if (key.startsWith(prefix)) {
          this._specializedFallbackBindGroups.delete(key);
        }
      }
    }
  }

  _invalidateSpecializedBindGroupsFor(bindGroupId) {
    if (this._specializedFallbackBindGroups.size === 0) {
      return;
    }
    const suffix = `|${bindGroupId}`;
    for (const key of this._specializedFallbackBindGroups.keys()) {
      if (key.endsWith(suffix)) {
        this._specializedFallbackBindGroups.delete(key);
      }
    }
  }

  /**
   * W346: Lazily scan the active command buffer for fallback shader profile hints only when
   * a `CreateShaderModule` packet actually misses the WGSL pack and needs fallback routing.
   */
  _ensureCommandStreamFallbackProfiles() {
    if (!this._pendingCommandStreamScanned) {
      this._pendingCommandStreamScanned = true;
      const view = this._pendingCommandStreamView;
      if (view) {
        this._pendingCommandStreamShaderProfiles = this._scanCommandBufferFallbackProfiles(
          view,
          this._pendingCommandStreamStartOffset,
          this._pendingCommandStreamEndOffset
        );
      }
    }
    return this._pendingCommandStreamShaderProfiles;
  }

  _scanCommandBufferFallbackProfiles(view, startOffset = 0, endOffset = view.byteLength) {
    if (!this.shaderFallbackRouter || typeof this.shaderFallbackRouter.resolve !== 'function') {
      return null;
    }
    this._fallbackProfileScanCount += 1;

    const localPipelines = new Map();
    const localBindGroups = new Map();
    const shaderProfileHints = new Map();

    let activePipelineId = null;
    const boundBindGroups = new Map();
    const boundVertexBuffers = new Map();

    const recordHint = (shaderId, profile) => {
      if (!shaderId || profile === FALLBACK_PROFILE.DEFAULT) {
        return;
      }
      const existing = shaderProfileHints.get(shaderId);
      if (!existing || existing === FALLBACK_PROFILE.DEFAULT) {
        shaderProfileHints.set(shaderId, profile);
      }
    };

    const limit = Math.min(view.byteLength, endOffset);
    let offset = startOffset;
    while (offset + 8 <= limit) {
      const opcode = view.getUint16(offset, true);
      const payloadBytes = view.getUint32(offset + 4, true);
      const payloadOffset = offset + 8;
      if (payloadOffset + payloadBytes > limit) break;

      switch (opcode) {
        case OPCODES.BEGIN_FRAME:
        case OPCODES.PRESENT:
          activePipelineId = null;
          boundBindGroups.clear();
          boundVertexBuffers.clear();
          break;
        case OPCODES.CREATE_RENDER_PIPELINE:
          if (payloadBytes >= 24) {
            const pipeId = view.getUint32(payloadOffset, true);
            const vsShaderId = view.getUint32(payloadOffset + 4, true);
            const fsShaderId = view.getUint32(payloadOffset + 8, true);
            const vertexStride = view.getUint32(payloadOffset + 12, true);
            localPipelines.set(pipeId, { vsShaderId, fsShaderId, vertexStride });
          }
          break;
        case OPCODES.CREATE_BIND_GROUP:
          if (payloadBytes >= 16) {
            const bgId = view.getUint32(payloadOffset, true);
            const entryCount = view.getUint32(payloadOffset + 12, true);
            if (16 + entryCount * 32 <= payloadBytes) {
              let hasCbv = false;
              let hasTexture = false;
              let hasSampler = false;
              for (let i = 0; i < entryCount; i += 1) {
                const entryOffset = payloadOffset + 16 + i * 32;
                const resourceType = view.getUint32(entryOffset + 4, true);
                if (resourceType === BIND_RESOURCE_TYPE.BUFFER) {
                  hasCbv = true;
                } else if (resourceType === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
                  hasTexture = true;
                } else if (resourceType === BIND_RESOURCE_TYPE.SAMPLER) {
                  hasSampler = true;
                }
              }
              localBindGroups.set(bgId, { hasCbv, hasTexture, hasSampler });
            }
          }
          break;
        case OPCODES.SET_PIPELINE:
          if (payloadBytes >= 4) {
            activePipelineId = view.getUint32(payloadOffset, true);
          }
          break;
        case OPCODES.SET_BIND_GROUP:
          if (payloadBytes >= 8) {
            const groupIndex = view.getUint32(payloadOffset, true);
            const bgId = view.getUint32(payloadOffset + 4, true);
            const bgMeta =
              localBindGroups.get(bgId) ||
              this._bindGroupMeta.get(bgId) ||
              this.bindGroups.get(bgId) ||
              null;
            if (bgMeta) {
              boundBindGroups.set(groupIndex, bgMeta);
            }
          }
          break;
        case OPCODES.SET_VERTEX_BUFFER:
          if (payloadBytes >= 16) {
            const slot = view.getUint32(payloadOffset, true);
            const bufferId = view.getUint32(payloadOffset + 4, true);
            const existingBuf = this.buffers.get(bufferId);
            boundVertexBuffers.set(slot, {
              bufferId,
              stride:
                existingBuf && typeof existingBuf.stride === 'number'
                  ? existingBuf.stride
                  : existingBuf && typeof existingBuf.arrayStride === 'number'
                    ? existingBuf.arrayStride
                    : undefined,
            });
          }
          break;
        case OPCODES.DRAW:
        case OPCODES.DRAW_INDEXED: {
          const minBytes = opcode === OPCODES.DRAW ? 16 : 20;
          if (payloadBytes >= minBytes && activePipelineId !== null) {
            const vertexCount = view.getUint32(payloadOffset, true);
            const instanceCount = view.getUint32(payloadOffset + 4, true);
            const pipeMeta =
              localPipelines.get(activePipelineId) || this._pipelineMeta.get(activePipelineId);
            if (pipeMeta) {
              const profile = selectFallbackShaderProfile({
                vertexCount,
                instanceCount,
                vertexStride: pipeMeta.vertexStride || 0,
                bindGroups: boundBindGroups,
                vertexBuffers: boundVertexBuffers,
              });
              if (profile !== FALLBACK_PROFILE.DEFAULT) {
                recordHint(pipeMeta.vsShaderId, profile);
                if (pipeMeta.fsShaderId) {
                  recordHint(pipeMeta.fsShaderId, profile);
                }
              }
            }
          }
          break;
        }
        default:
          break;
      }

      offset = payloadOffset + payloadBytes;
    }

    return shaderProfileHints.size > 0 ? shaderProfileHints : null;
  }

  _getOrCreateSpecializedFallbackPipeline(pipeMeta, profile) {
    if (
      !pipeMeta ||
      !this.device ||
      !this.shaderFallbackRouter ||
      typeof this.shaderFallbackRouter.resolve !== 'function'
    ) {
      return null;
    }
    const canonicalProfile = normalizeFallbackProfileName(profile);
    if (
      canonicalProfile === FALLBACK_PROFILE.DEFAULT ||
      canonicalProfile === pipeMeta.fallbackProfile
    ) {
      return null;
    }

    const cacheKey = `${pipeMeta.id}|${canonicalProfile}`;
    if (this._specializedFallbackPipelines.has(cacheKey)) {
      const hit = this._specializedFallbackPipelines.get(cacheKey);
      this._specializedFallbackPipelines.delete(cacheKey);
      this._specializedFallbackPipelines.set(cacheKey, hit);
      return hit;
    }

    const vsOrig = this.ensureShaderModule(pipeMeta.vsShaderId);
    const fsOrig = pipeMeta.fsShaderId ? this.ensureShaderModule(pipeMeta.fsShaderId) : null;
    if (!vsOrig) {
      return null;
    }

    const getOrCreateFallbackStageModule = (stageKey, code) => {
      const modKey = `${stageKey}:${canonicalProfile}`;
      const cachedMod = this._specializedFallbackShaderModules.get(modKey);
      if (cachedMod && cachedMod.code === code) {
        return cachedMod.module;
      }
      const stallStartMs = this._gpuStallWarnMs > 0 ? this._getNowMs() : 0;
      const mod = this.device.createShaderModule({ code });
      if (this._gpuStallWarnMs > 0) {
        this._maybeWarnGpuStall('createShaderModule', stallStartMs, modKey);
      }
      this._specializedFallbackShaderModules.set(modKey, { code, module: mod });
      return mod;
    };

    let vsModule = vsOrig.module;
    let vsEntryPoint = vsOrig.entryPoint || 'vs_main';
    let changed = false;

    if (vsOrig.source === 'builtin') {
      const vsHash = vsOrig.requestedHash || vsOrig.hashHex || fnv1a64Hex(`empty:vertex:${pipeMeta.vsShaderId}`);
      const resolvedVs = callFallbackRouterResolve(
        this.shaderFallbackRouter,
        vsHash,
        'vertex',
        canonicalProfile
      );
      if (resolvedVs && typeof resolvedVs.wgsl === 'string' && resolvedVs.wgsl.length > 0) {
        if (resolvedVs.wgsl !== vsOrig.code) {
          const vsCode = applyRtBvhPreludeIfNeeded(resolvedVs.wgsl);
          vsModule = getOrCreateFallbackStageModule('vertex', vsCode);
          vsEntryPoint = resolvedVs.entryPoint || extractWgslEntryPoint(vsCode, SHADER_STAGE.VERTEX) || 'vs_main';
          changed = true;
        }
        this._recordFallbackProfileUsage(resolvedVs.profile || canonicalProfile);
      }
    }

    let fsModule = fsOrig ? fsOrig.module : null;
    let fsEntryPoint = fsOrig ? fsOrig.entryPoint || 'fs_main' : 'fs_main';

    if (fsOrig && fsOrig.source === 'builtin') {
      const fsHash = fsOrig.requestedHash || fsOrig.hashHex || fnv1a64Hex(`empty:fragment:${pipeMeta.fsShaderId}`);
      const resolvedFs = callFallbackRouterResolve(
        this.shaderFallbackRouter,
        fsHash,
        'fragment',
        canonicalProfile
      );
      if (resolvedFs && typeof resolvedFs.wgsl === 'string' && resolvedFs.wgsl.length > 0) {
        if (resolvedFs.wgsl !== fsOrig.code) {
          const fsCode = applyRtBvhPreludeIfNeeded(resolvedFs.wgsl);
          fsModule = getOrCreateFallbackStageModule('fragment', fsCode);
          fsEntryPoint = resolvedFs.entryPoint || extractWgslEntryPoint(fsCode, SHADER_STAGE.FRAGMENT) || 'fs_main';
          changed = true;
        }
        this._recordFallbackProfileUsage(resolvedFs.profile || canonicalProfile);
      }
    }

    if (!changed) {
      const existing = this.pipelines.get(pipeMeta.id) || null;
      const entry = { pipeline: existing, profile: canonicalProfile, changed: false };
      if (this._specializedFallbackPipelines.size >= MAX_SPECIALIZED_FALLBACK_PIPELINES) {
        const oldestKey = this._specializedFallbackPipelines.keys().next().value;
        if (oldestKey !== undefined) {
          this._specializedFallbackPipelines.delete(oldestKey);
        }
      }
      this._specializedFallbackPipelines.set(cacheKey, entry);
      return entry;
    }

    let vertexBuffers = [];
    if (canonicalProfile === FALLBACK_PROFILE.SPLASH_RECT_24B) {
      vertexBuffers = [
        {
          arrayStride: FALLBACK_VERTEX_LAYOUT_SPLASH_RECT_24B.arrayStride,
          stepMode: FALLBACK_VERTEX_LAYOUT_SPLASH_RECT_24B.stepMode,
          attributes: FALLBACK_VERTEX_LAYOUT_SPLASH_RECT_24B.attributes.map((a) => ({ ...a })),
        },
      ];
    } else if (canonicalProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B) {
      vertexBuffers = [
        {
          arrayStride: FALLBACK_VERTEX_LAYOUT_24B.arrayStride,
          stepMode: FALLBACK_VERTEX_LAYOUT_24B.stepMode,
          attributes: FALLBACK_VERTEX_LAYOUT_24B.attributes.map((a) => ({ ...a })),
        },
      ];
    } else if (canonicalProfile === FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV) {
      vertexBuffers = [];
    }

    const topology = pipeMeta.topology || 'triangle-list';
    const targetFormat = this._getRenderTargetFormat();
    const descriptor = {
      layout: 'auto',
      vertex: {
        module: vsModule,
        entryPoint: vsEntryPoint,
        buffers: vertexBuffers,
      },
      primitive: {
        topology,
      },
    };
    if (fsOrig && fsModule) {
      descriptor.fragment = {
        module: fsModule,
        entryPoint: fsEntryPoint,
        targets: [{ format: targetFormat }],
      };
    }

    const allBuiltinStages =
      vsOrig.source === 'builtin' && (!fsOrig || fsOrig.source === 'builtin');
    const specFastKey = allBuiltinStages
      ? `spec:${canonicalProfile}:${topology}:${fsOrig ? targetFormat : 'none'}`
      : `spec:${pipeMeta.id}:${canonicalProfile}:${topology}:${fsOrig ? targetFormat : 'none'}`;
    let pipeline;
    const pipeStallStartMs = this._gpuStallWarnMs > 0 ? this._getNowMs() : 0;
    if (this.pipelineCache && typeof this.pipelineCache.getFastRenderPipeline === 'function') {
      pipeline = this.pipelineCache.getFastRenderPipeline(specFastKey, descriptor);
    } else if (
      this.pipelineCache &&
      typeof this.pipelineCache.getOrCreateRenderPipeline === 'function'
    ) {
      pipeline = this.pipelineCache.getOrCreateRenderPipeline(descriptor, specFastKey);
    } else {
      pipeline = this.device.createRenderPipeline(descriptor);
    }
    if (this._gpuStallWarnMs > 0) {
      const shaderTag = vsOrig.requestedHash || vsOrig.hashHex || specFastKey;
      this._maybeWarnGpuStall('createRenderPipeline', pipeStallStartMs, shaderTag);
    }
    if (!pipeline) {
      return null;
    }
    const entry = { pipeline, profile: canonicalProfile, changed: true };
    if (this._specializedFallbackPipelines.size >= MAX_SPECIALIZED_FALLBACK_PIPELINES) {
      const oldestKey = this._specializedFallbackPipelines.keys().next().value;
      if (oldestKey !== undefined) {
        this._specializedFallbackPipelines.delete(oldestKey);
      }
    }
    this._specializedFallbackPipelines.set(cacheKey, entry);
    return entry;
  }

  _getOrCreateSpecializedBindGroup(pipeId, profile, specPipeline, groupIndex, bgMeta) {
    if (
      !specPipeline ||
      !bgMeta ||
      !Array.isArray(bgMeta.entries) ||
      typeof specPipeline.getBindGroupLayout !== 'function' ||
      !this.device ||
      typeof this.device.createBindGroup !== 'function'
    ) {
      return null;
    }
    const key = `${pipeId}|${profile}|${bgMeta.id}`;
    if (this._specializedFallbackBindGroups.has(key)) {
      const hit = this._specializedFallbackBindGroups.get(key);
      this._specializedFallbackBindGroups.delete(key);
      this._specializedFallbackBindGroups.set(key, hit);
      return hit;
    }
    try {
      const layout = specPipeline.getBindGroupLayout(groupIndex);
      if (!layout) {
        return null;
      }
      let entries = bgMeta.entries;
      if (
        (profile === FALLBACK_PROFILE.TEXTURED_QUAD_24B ||
          profile === FALLBACK_PROFILE.SPLASH_RECT_24B) &&
        groupIndex === 0 &&
        !bgMeta.hasSampler &&
        typeof this.device.createSampler === 'function'
      ) {
        const samplerBinding = profile === FALLBACK_PROFILE.SPLASH_RECT_24B ? 2 : 1;
        const synth = this.synthesizeFallbackSamplerEntries(
          [{ binding: samplerBinding, sampler: { type: 'filtering' } }],
          entries
        );
        if (synth && Array.isArray(synth.entries)) {
          entries = synth.entries;
        }
      }
      const bg = this.device.createBindGroup({
        layout,
        entries,
      });
      if (this._specializedFallbackBindGroups.size >= MAX_SPECIALIZED_FALLBACK_BIND_GROUPS) {
        const oldestKey = this._specializedFallbackBindGroups.keys().next().value;
        if (oldestKey !== undefined) {
          this._specializedFallbackBindGroups.delete(oldestKey);
        }
      }
      this._specializedFallbackBindGroups.set(key, bg);
      return bg;
    } catch (_) {
      return null;
    }
  }

  _getNowMs() {
    const val = typeof this._validationNow === 'function' ? this._validationNow() : defaultNowMs();
    return typeof val === 'number' && Number.isFinite(val) ? val : Date.now();
  }

  _resetPerFrameDrawCounters() {
    this._frameDrawsThisFrame = 0;
    this._frameRealPackDrawsThisFrame = 0;
    this._frameFallbackDrawsThisFrame = 0;
    this._frameNonClearDrawsThisFrame = 0;
  }

  _notifyFirstPresent(info = {}) {
    if (this._firstPresentNotified) {
      return false;
    }
    this._firstPresentNotified = true;
    const ts =
      typeof info.timestampMs === 'number' && Number.isFinite(info.timestampMs)
        ? info.timestampMs
        : this._getNowMs();
    if (this._firstPresentMs === null) {
      this._firstPresentMs = ts;
    }
    const payload =
      info && typeof info === 'object'
        ? {
            ...info,
            timestampMs: ts,
            gpuAdapter:
              info.gpuAdapter !== undefined ? info.gpuAdapter : this._gpuAdapter,
          }
        : {
            timestampMs: ts,
            gpuAdapter: this._gpuAdapter,
          };
    if (typeof this._onFirstPresent === 'function') {
      try {
        this._onFirstPresent(payload);
      } catch (_) {
        // Telemetry hooks must never break frame presentation.
      }
    }
    return true;
  }

  _notifyFirstVisibleFrame(info = {}) {
    if (this._firstVisibleFrameNotified) {
      return false;
    }
    this._firstVisibleFrameNotified = true;
    // W338: Destroy the pre-allocated swapchain luminance probe buffer immediately
    // once first-visible-frame fires so there is zero steady-state overhead.
    this._destroySwapchainProbeBuffer();
    // Keep swapchain visible-frame state in sync (e.g. when first-visible-frame fires from
    // Bink2 / splash presentation) so WgpuSwapchain can drop COPY_SRC and never re-probe.
    if (
      this._swapchain &&
      !this._swapchain.isDestroyed &&
      this._swapchain.hasVisibleContent !== true &&
      this._swapchain.hasFirstVisibleFrame !== true &&
      typeof this._swapchain.confirmVisibleFrame === 'function'
    ) {
      try {
        const visibleTs =
          info &&
          typeof info.firstVisibleFrameMs === 'number' &&
          Number.isFinite(info.firstVisibleFrameMs)
            ? info.firstVisibleFrameMs
            : this._firstVisibleFrameMs !== null
              ? this._firstVisibleFrameMs
              : this._getNowMs();
        this._swapchain.confirmVisibleFrame({
          timestampMs: visibleTs,
          realPackDraws:
            info && typeof info.realPackDrawsThisFrame === 'number'
              ? info.realPackDrawsThisFrame
              : this._realPackDrawCount,
        });
      } catch (_) {
        // Swapchain telemetry must never break frame presentation.
      }
    }
    if (typeof this._onFirstVisibleFrame === 'function') {
      try {
        this._onFirstVisibleFrame(info);
      } catch (_) {
        // Telemetry hooks must never break frame presentation.
      }
    }
    if (this._firstVisibleFrameListeners && this._firstVisibleFrameListeners.size > 0) {
      for (const listener of this._firstVisibleFrameListeners) {
        try {
          listener(info);
        } catch (_) {
          // Telemetry listeners must never break frame presentation.
        }
      }
    }
    return true;
  }

  _resolveBinkPresenterCtor() {
    if (this._BinkVideoPresenterCtor === null || this._BinkVideoPresenterCtor === false) {
      return null;
    }
    return resolveSubsystemCtor(
      this._BinkVideoPresenterCtor,
      'PlayGta6BinkPresenter',
      'BinkVideoPresenter',
      './bink_presenter.js'
    );
  }

  _resolveSplashFrameBridgeCtor() {
    if (this._SplashFrameBridgeCtor === null || this._SplashFrameBridgeCtor === false) {
      return null;
    }
    return resolveSubsystemCtor(
      this._SplashFrameBridgeCtor,
      'PlayGta6SplashFrameBridge',
      'SplashFrameBridge',
      './splash_frame_bridge.js'
    );
  }

  _syncBinkPresentationFormat(targetFormat = this.format) {
    if (typeof targetFormat !== 'string' || targetFormat.length === 0) {
      return;
    }
    const bridge =
      this.splashFrameBridge && !this.splashFrameBridge.isDestroyed
        ? this.splashFrameBridge
        : null;
    if (!this.binkPresenter && bridge && bridge.presenter) {
      this.binkPresenter = bridge.presenter;
    }
    const presenter =
      this.binkPresenter && !this.binkPresenter.isDestroyed ? this.binkPresenter : null;

    if (bridge && typeof bridge.setPresentationFormat === 'function') {
      try {
        bridge.setPresentationFormat(targetFormat);
      } catch (_) {
        // Fall through to presenter sync / owned rebuild if bridge update fails.
      }
    }
    if (
      presenter &&
      typeof presenter.setPresentationFormat === 'function' &&
      presenter.presentationFormat !== targetFormat
    ) {
      try {
        presenter.setPresentationFormat(targetFormat);
      } catch (_) {
        // Fall through to owned rebuild if presenter update fails.
      }
    }

    // Fallback for custom/legacy presenters without setPresentationFormat():
    // tear down owned bridge + presenter so _ensureSplashBridge() recreates them for targetFormat.
    if (
      this._ownsBinkPresenter &&
      this.binkPresenter &&
      typeof this.binkPresenter.presentationFormat === 'string' &&
      this.binkPresenter.presentationFormat !== targetFormat
    ) {
      if (this.splashFrameBridge && this._ownsSplashFrameBridge) {
        try {
          if (typeof this.splashFrameBridge.destroy === 'function') {
            this.splashFrameBridge.destroy();
          }
        } catch (_) {
          // Ignore teardown errors when reconciling canvas format.
        }
        this.splashFrameBridge = null;
        this._ownsSplashFrameBridge = false;
      }
      try {
        if (typeof this.binkPresenter.destroy === 'function') {
          this.binkPresenter.destroy();
        }
      } catch (_) {
        // Ignore teardown errors when reconciling canvas format.
      }
      this.binkPresenter = null;
      this._ownsBinkPresenter = false;
    }
  }

  _ensureSplashBridge() {
    if (this._destroyed) {
      return null;
    }
    if (this.splashFrameBridge) {
      if (this.splashFrameBridge.isDestroyed) {
        return null;
      }
      if (!this.binkPresenter && this.splashFrameBridge.presenter) {
        this.binkPresenter = this.splashFrameBridge.presenter;
      }
      this._syncBinkPresentationFormat(this.format);
      if (this.splashFrameBridge && !this.splashFrameBridge.isDestroyed) {
        return this.splashFrameBridge;
      }
    }
    if (this._splashBridgeExplicitlyDisabled && !this.binkPresenter) {
      return null;
    }
    if (this.binkPresenter) {
      this._syncBinkPresentationFormat(this.format);
    }
    if (!this.binkPresenter) {
      if (!this.device || typeof this.device.createTexture !== 'function') {
        return null;
      }
      const BinkCtor = this._resolveBinkPresenterCtor();
      if (!BinkCtor) {
        return null;
      }
      this.binkPresenter = new BinkCtor({
        ...(this._binkPresenterOptions || {}),
        presentationFormat: this.format,
        device: this.device,
      });
      this._ownsBinkPresenter = true;
    }
    if (!this._splashBridgeExplicitlyDisabled) {
      const BridgeCtor = this._resolveSplashFrameBridgeCtor();
      if (BridgeCtor) {
        const bridgeOpts = {
          ...(this._splashBridgeOptions || {}),
          presentationFormat: this.format,
          device: this.device,
          presenter: this.binkPresenter,
        };
        if (this._binkPresenterOptions) {
          if (
            bridgeOpts.chromaOrder === undefined &&
            this._binkPresenterOptions.chromaOrder !== undefined
          ) {
            bridgeOpts.chromaOrder = this._binkPresenterOptions.chromaOrder;
          }
          if (
            bridgeOpts.loadOp === undefined &&
            this._binkPresenterOptions.loadOp !== undefined
          ) {
            bridgeOpts.loadOp = this._binkPresenterOptions.loadOp;
          }
        }
        const userOnFirstPresent =
          typeof bridgeOpts.onFirstPresent === 'function' ? bridgeOpts.onFirstPresent : null;
        bridgeOpts.onFirstPresent = (info) => {
          this._notifyFirstPresent({
            ...info,
            framesPresented: this.frameCount + 1,
          });
          if (userOnFirstPresent) {
            try {
              userOnFirstPresent(info);
            } catch (_) {
              // User callback failures must not break presentation.
            }
          }
        };
        if (this._requireNonBlackFrame) {
          if (bridgeOpts.requireNonBlackPixels === undefined) {
            bridgeOpts.requireNonBlackPixels = true;
          }
          if (bridgeOpts.minVisibleLuma === undefined) {
            bridgeOpts.minVisibleLuma = this._minVisibleLuma;
          }
          if (bridgeOpts.minVisiblePixels === undefined) {
            bridgeOpts.minVisiblePixels = this._minVisiblePixels;
          }
          const userOnFirstVisible =
            typeof bridgeOpts.onFirstVisibleFrame === 'function'
              ? bridgeOpts.onFirstVisibleFrame
              : null;
          bridgeOpts.onFirstVisibleFrame = (info) => {
            const visibleTs = this._getNowMs();
            const firstPresentMs = this._firstPresentMs !== null ? this._firstPresentMs : visibleTs;
            const blackLoadingMs = Math.max(0, visibleTs - firstPresentMs);
            const infoLuma =
              info && typeof info.maxLuma === 'number' && Number.isFinite(info.maxLuma)
                ? info.maxLuma
                : info &&
                    typeof info.lastSampledMaxLuma === 'number' &&
                    Number.isFinite(info.lastSampledMaxLuma)
                  ? info.lastSampledMaxLuma
                  : info &&
                      typeof info.sampledMaxLuma === 'number' &&
                      Number.isFinite(info.sampledMaxLuma)
                    ? info.sampledMaxLuma
                    : null;
            if (infoLuma !== null) {
              this._lastSampledMaxLuma = Math.max(0, Math.min(255, Math.floor(infoLuma)));
            }
            if (
              info &&
              typeof info.suppressedBlackFrames === 'number' &&
              Number.isFinite(info.suppressedBlackFrames)
            ) {
              this._suppressedBlackFrames = Math.max(
                this._suppressedBlackFrames,
                Math.floor(info.suppressedBlackFrames)
              );
            }
            if (!this._firstVisibleFrameNotified) {
              this._firstVisibleFrameMs = visibleTs;
              this._blackLoadingMs = blackLoadingMs;
              this._notifyFirstVisibleFrame({
                type: 'first-visible-frame',
                firstVisibleFrame: true,
                firstVisibleFrameMs: visibleTs,
                firstPresentMs,
                blackLoadingMs,
                presentCount: this._presentCount,
                realPackDrawsThisFrame: 0,
                fallbackDrawsThisFrame: 0,
                drawsThisFrame: 0,
                source: (info && info.source) || 'bink2',
                maxLuma: this._lastSampledMaxLuma,
                suppressedBlackFrames: this._suppressedBlackFrames,
                gpuAdapter: this._gpuAdapter,
              });
            }
            if (userOnFirstVisible) {
              try {
                userOnFirstVisible(info);
              } catch (_) {
                // User callback failures must not break presentation.
              }
            }
          };
        }
        this.splashFrameBridge = new BridgeCtor(bridgeOpts);
        this._ownsSplashFrameBridge = true;
      }
    }
    return this.splashFrameBridge;
  }

  _resolveSplashTargetSize(swapTexture, framePayload, extraOptions) {
    const extImage =
      framePayload && (framePayload.image || framePayload.imageBitmap || framePayload.bitmap);
    const canvas = this.ctx && this.ctx.canvas;
    const width =
      pickPositiveIntDim(extraOptions && extraOptions.targetWidth) ||
      pickPositiveIntDim(framePayload && framePayload.targetWidth) ||
      pickPositiveIntDim(swapTexture && swapTexture.width) ||
      pickPositiveIntDim(this._viewportWidth) ||
      pickPositiveIntDim(canvas && canvas.width) ||
      pickPositiveIntDim(framePayload && framePayload.width) ||
      pickPositiveIntDim(extImage && extImage.width) ||
      1280;
    const height =
      pickPositiveIntDim(extraOptions && extraOptions.targetHeight) ||
      pickPositiveIntDim(framePayload && framePayload.targetHeight) ||
      pickPositiveIntDim(swapTexture && swapTexture.height) ||
      pickPositiveIntDim(this._viewportHeight) ||
      pickPositiveIntDim(canvas && canvas.height) ||
      pickPositiveIntDim(framePayload && framePayload.height) ||
      pickPositiveIntDim(extImage && extImage.height) ||
      720;
    const out = this._splashTargetSizeScratch;
    out.width = width;
    out.height = height;
    return out;
  }

  /**
   * Present an RGBA8, YUV420P/YUVA420P, ImageBitmap, or diagnostic splash/Bink2 frame
   * onto the WebGPU swapchain via SplashFrameBridge -> BinkVideoPresenter.
   *
   * @param {object} framePayload
   * @param {{ nowMs?: number, respectPacing?: boolean, targetWidth?: number, targetHeight?: number }} [options]
   */
  presentSplashFrame(framePayload, options = {}) {
    const requestedFrameIndex =
      framePayload && typeof framePayload === 'object' && Number.isInteger(framePayload.frameIndex)
        ? framePayload.frameIndex
        : null;

    if (this._destroyed) {
      return {
        ok: false,
        presented: false,
        reason: 'destroyed',
        error: 'WgpuCommandProcessor has been destroyed',
        frameIndex: requestedFrameIndex,
        framesPresented: this.frameCount,
        splashFramesPresented: this._splashFramesPresented,
        firstPresent: false,
      };
    }

    if (!this.device || !this.ctx || typeof this.ctx.getCurrentTexture !== 'function') {
      return {
        ok: false,
        presented: false,
        reason: 'not-ready',
        error: 'WebGPU device or swapchain context is not ready',
        frameIndex: requestedFrameIndex,
        framesPresented: this.frameCount,
        splashFramesPresented: this._splashFramesPresented,
        firstPresent: false,
      };
    }

    if (!framePayload || typeof framePayload !== 'object') {
      return {
        ok: false,
        presented: false,
        reason: 'invalid-frame',
        error: 'presentSplashFrame requires a frame payload object',
        frameIndex: null,
        framesPresented: this.frameCount,
        splashFramesPresented: this._splashFramesPresented,
        firstPresent: false,
      };
    }

    const opts = options && typeof options === 'object' ? options : {};

    try {
      const bridge = this._ensureSplashBridge();
      const presenter =
        this.binkPresenter || (bridge && bridge.presenter ? bridge.presenter : null);

      if (!bridge && !presenter) {
        return {
          ok: false,
          presented: false,
          reason: 'presenter-unavailable',
          error: 'BinkVideoPresenter / SplashFrameBridge unavailable',
          frameIndex: requestedFrameIndex,
          framesPresented: this.frameCount,
          splashFramesPresented: this._splashFramesPresented,
          firstPresent: false,
        };
      }

      if (
        bridge &&
        !presenter &&
        !bridge._BinkVideoPresenterCtor &&
        !this._resolveBinkPresenterCtor()
      ) {
        return {
          ok: false,
          presented: false,
          reason: 'presenter-unavailable',
          error: 'BinkVideoPresenter unavailable',
          frameIndex: requestedFrameIndex,
          framesPresented: this.frameCount,
          splashFramesPresented: this._splashFramesPresented,
          firstPresent: false,
        };
      }

      const swapTexture = this.ctx.getCurrentTexture();
      const targetView =
        swapTexture && typeof swapTexture.createView === 'function'
          ? swapTexture.createView()
          : swapTexture;
      if (!targetView || typeof targetView !== 'object') {
        return {
          ok: false,
          presented: false,
          reason: 'not-ready',
          error: 'Swapchain texture view unavailable',
          frameIndex: requestedFrameIndex,
          framesPresented: this.frameCount,
          splashFramesPresented: this._splashFramesPresented,
          firstPresent: false,
        };
      }

      const { width: targetWidth, height: targetHeight } = this._resolveSplashTargetSize(
        swapTexture,
        framePayload,
        opts
      );

      const externalImage =
        framePayload.image ||
        framePayload.imageBitmap ||
        framePayload.bitmap ||
        (isExternalImageLike(framePayload) ? framePayload : null);
      const isDiagnostic =
        framePayload.diagnostic === true ||
        (framePayload.source === 'diagnostic' &&
          !framePayload.planes &&
          !framePayload.rgba &&
          !framePayload.data &&
          !externalImage);

      const explicitNowMs =
        typeof opts.nowMs === 'number' && Number.isFinite(opts.nowMs)
          ? opts.nowMs
          : typeof framePayload.nowMs === 'number' && Number.isFinite(framePayload.nowMs)
            ? framePayload.nowMs
            : null;
      const respectPacing =
        explicitNowMs !== null ||
        opts.respectPacing === true ||
        framePayload.respectPacing === true;

      let renderNowMs = explicitNowMs;
      if (!respectPacing) {
        const targetFps =
          presenter && typeof presenter.targetFps === 'number' && presenter.targetFps > 0
            ? presenter.targetFps
            : 30;
        const minStepMs = 1000 / targetFps + 1;
        const baseNow =
          typeof framePayload.timestampMs === 'number' && Number.isFinite(framePayload.timestampMs)
            ? framePayload.timestampMs
            : typeof performance !== 'undefined' && typeof performance.now === 'function'
              ? performance.now()
              : Date.now();
        const lastAdv =
          presenter && typeof presenter._lastAdvanceMs === 'number'
            ? presenter._lastAdvanceMs
            : this._lastSplashPresentMs;
        renderNowMs =
          lastAdv !== null && Number.isFinite(lastAdv)
            ? Math.max(baseNow, lastAdv + minStepMs)
            : baseNow;
      }

      const requestedChromaOrder = normalizeBinkChromaOrderInput(
        opts.chromaOrder !== undefined ? opts.chromaOrder : framePayload.chromaOrder
      );
      const requestedLoadOp = normalizeBinkLoadOpInput(
        opts.loadOp !== undefined ? opts.loadOp : framePayload.loadOp
      );
      const requestedColorRange =
        opts.colorRange !== undefined ? opts.colorRange : framePayload.colorRange;

      if (bridge) {
        const activePresenter =
          presenter ||
          (typeof bridge._ensurePresenter === 'function' ? bridge._ensurePresenter(this.device) : null);
        if (requestedChromaOrder !== null) {
          if (typeof bridge.setChromaOrder === 'function') {
            bridge.setChromaOrder(requestedChromaOrder);
          }
          if (activePresenter && typeof activePresenter.setChromaOrder === 'function') {
            activePresenter.setChromaOrder(requestedChromaOrder);
          }
        }
        if (requestedLoadOp !== null) {
          if (typeof bridge.setLoadOp === 'function') {
            bridge.setLoadOp(requestedLoadOp);
          }
          if (activePresenter && typeof activePresenter.setLoadOp === 'function') {
            activePresenter.setLoadOp(requestedLoadOp);
          }
        }

        let pushed;
        if (externalImage) {
          const extOpts = {};
          if (framePayload.width !== undefined) extOpts.width = framePayload.width;
          if (framePayload.height !== undefined) extOpts.height = framePayload.height;
          if (framePayload.timestampMs !== undefined) extOpts.timestampMs = framePayload.timestampMs;
          if (framePayload.frameIndex !== undefined) extOpts.frameIndex = framePayload.frameIndex;
          if (requestedLoadOp !== null) extOpts.loadOp = requestedLoadOp;
          pushed = bridge.pushExternalImage(externalImage, extOpts);
        } else if (isDiagnostic) {
          const diagW =
            typeof framePayload.width === 'number' && framePayload.width > 0
              ? Math.floor(framePayload.width)
              : targetWidth;
          const diagH =
            typeof framePayload.height === 'number' && framePayload.height > 0
              ? Math.floor(framePayload.height)
              : targetHeight;
          const diagOpts = {
            timestampMs: framePayload.timestampMs,
          };
          if (requestedLoadOp !== null) diagOpts.loadOp = requestedLoadOp;
          pushed = bridge.pushDiagnosticFrame(diagW, diagH, framePayload.frameIndex, diagOpts);
        } else {
          const format = normalizeSplashPixelFormat(framePayload.format, framePayload);
          const rawPlanes =
            framePayload.planes && typeof framePayload.planes === 'object'
              ? framePayload.planes
              : {};
          let planes;
          if (format === 'rgba8') {
            planes = {
              rgba:
                rawPlanes.rgba !== undefined
                  ? rawPlanes.rgba
                  : rawPlanes.data !== undefined
                    ? rawPlanes.data
                    : framePayload.rgba !== undefined
                      ? framePayload.rgba
                      : framePayload.data,
            };
          } else if (format === 'nv12') {
            planes = {
              y:
                rawPlanes.y !== undefined
                  ? rawPlanes.y
                  : rawPlanes.yPlane !== undefined
                    ? rawPlanes.yPlane
                    : framePayload.y !== undefined
                      ? framePayload.y
                      : framePayload.yPlane,
              uv:
                rawPlanes.uv !== undefined
                  ? rawPlanes.uv
                  : rawPlanes.uvPlane !== undefined
                    ? rawPlanes.uvPlane
                    : framePayload.uv !== undefined
                      ? framePayload.uv
                      : framePayload.uvPlane,
            };
          } else {
            planes = {
              y: rawPlanes.y !== undefined ? rawPlanes.y : framePayload.y,
              u: rawPlanes.u !== undefined ? rawPlanes.u : framePayload.u,
              v: rawPlanes.v !== undefined ? rawPlanes.v : framePayload.v,
            };
            if (planes.u === undefined && (rawPlanes.cr !== undefined || framePayload.cr !== undefined)) {
              planes.cr = rawPlanes.cr !== undefined ? rawPlanes.cr : framePayload.cr;
            }
            if (planes.v === undefined && (rawPlanes.cb !== undefined || framePayload.cb !== undefined)) {
              planes.cb = rawPlanes.cb !== undefined ? rawPlanes.cb : framePayload.cb;
            }
            if (format === 'yuva420p') {
              planes.a = rawPlanes.a !== undefined ? rawPlanes.a : framePayload.a;
            }
          }
          const rawFrameDesc = {
            width: framePayload.width,
            height: framePayload.height,
            format,
            planes,
            timestampMs: framePayload.timestampMs,
            colorSpace: framePayload.colorSpace,
            source: framePayload.source,
            frameIndex: framePayload.frameIndex,
          };
          if (requestedColorRange !== undefined) {
            rawFrameDesc.colorRange = requestedColorRange;
          }
          if (requestedChromaOrder !== null) {
            rawFrameDesc.chromaOrder = requestedChromaOrder;
          }
          if (requestedLoadOp !== null) {
            rawFrameDesc.loadOp = requestedLoadOp;
          }
          if (framePayload.yStride !== undefined) {
            rawFrameDesc.yStride = framePayload.yStride;
          }
          if (framePayload.uvStride !== undefined) {
            rawFrameDesc.uvStride = framePayload.uvStride;
          }
          if (framePayload.aStride !== undefined) {
            rawFrameDesc.aStride = framePayload.aStride;
          }
          if (framePayload.strideBytes !== undefined) {
            rawFrameDesc.strideBytes = framePayload.strideBytes;
          }
          if (framePayload.zeroCopy !== undefined) {
            rawFrameDesc.zeroCopy = framePayload.zeroCopy;
          }
          pushed = bridge.pushRawFrame(rawFrameDesc);
        }

        const renderResult = bridge.renderFrame(
          this.device,
          targetView,
          targetWidth,
          targetHeight,
          renderNowMs !== null ? renderNowMs : undefined
        );

        if (!this.binkPresenter && bridge.presenter) {
          this.binkPresenter = bridge.presenter;
        }

        if (renderResult && renderResult.presented) {
          if (renderNowMs !== null) {
            this._lastSplashPresentMs = renderNowMs;
          }
          const splashSceneTransition =
            this._pendingSceneTransition ||
            (this._lastPresentSource !== null && this._lastPresentSource !== 'splash');
          const splashResChanged =
            this._lastPresentWidth > 0 &&
            targetWidth > 0 &&
            (targetWidth !== this._lastPresentWidth || targetHeight !== this._lastPresentHeight);
          this._pendingSceneTransition = false;
          this._lastPresentSource = 'splash';
          if (targetWidth > 0 && targetHeight > 0) {
            this._lastPresentWidth = targetWidth;
            this._lastPresentHeight = targetHeight;
          }
          if (splashSceneTransition || splashResChanged) {
            this.trimTransientMemory({ maxIdleFrames: this._oversizedRtReclaimIdleFrames });
          }
          const prevFrameCount = this.frameCount;
          this.frameCount += 1;
          this._splashFramesPresented += 1;
          const isFirstPresent = prevFrameCount === 0;
          if (isFirstPresent) {
            this._notifyFirstPresent({
              frameIndex: renderResult.frameIndex,
              framesPresented: this.frameCount,
              width: renderResult.width,
              height: renderResult.height,
              source: renderResult.source || 'splash',
              timestampMs: renderResult.timestampMs,
            });
          }
          const hasBinkYuvPlanes = Boolean(
            framePayload &&
              typeof framePayload === 'object' &&
              ((framePayload.planes &&
                typeof framePayload.planes === 'object' &&
                (framePayload.planes.y !== undefined ||
                  framePayload.planes.yPlane !== undefined)) ||
                framePayload.y !== undefined ||
                framePayload.yPlane !== undefined)
          );
          if (
            !this._firstVisibleFrameNotified &&
            (this._requireNonBlackFrame ||
              (this._requireNonBlackFrameExplicit !== false && hasBinkYuvPlanes))
          ) {
            if (
              typeof renderResult.lastSampledMaxLuma === 'number' &&
              Number.isFinite(renderResult.lastSampledMaxLuma)
            ) {
              this._lastSampledMaxLuma = Math.max(
                0,
                Math.min(255, Math.floor(renderResult.lastSampledMaxLuma))
              );
            }
            if (
              typeof renderResult.suppressedBlackFrames === 'number' &&
              Number.isFinite(renderResult.suppressedBlackFrames)
            ) {
              this._suppressedBlackFrames = Math.max(
                this._suppressedBlackFrames,
                Math.floor(renderResult.suppressedBlackFrames)
              );
            }
            if (renderResult.suppressedBlackFrame === true) {
              if (this._suppressedBlackFrames === 0) {
                this._suppressedBlackFrames = 1;
              }
            } else {
              const bridgeAlreadyVerified = Boolean(
                bridge &&
                  bridge.requireNonBlackPixels === true &&
                  (renderResult.firstVisibleFrameEmitted === true ||
                    renderResult.firstVisibleFrame === true)
              );
              const splashCheck = bridgeAlreadyVerified
                ? { evaluated: true, visible: true, maxLuma: this._lastSampledMaxLuma }
                : isDiagnostic && this._requireNonBlackFrame
                  ? { evaluated: true, visible: true, maxLuma: 235 }
                  : this._evaluateNonBlackLuminance(
                      framePayload,
                      renderResult.width || targetWidth,
                      renderResult.height || targetHeight
                    );
              if (splashCheck.evaluated) {
                this._lastSampledMaxLuma = Math.max(
                  0,
                  Math.min(255, Math.floor(splashCheck.maxLuma || 0))
                );
              }
              if (splashCheck.visible) {
                const visibleTs = this._getNowMs();
                const firstPresentMs =
                  this._firstPresentMs !== null ? this._firstPresentMs : visibleTs;
                const blackLoadingMs = Math.max(0, visibleTs - firstPresentMs);
                this._firstVisibleFrameMs = visibleTs;
                this._blackLoadingMs = blackLoadingMs;
                this._notifyFirstVisibleFrame({
                  type: 'first-visible-frame',
                  firstVisibleFrame: true,
                  firstVisibleFrameMs: visibleTs,
                  firstPresentMs,
                  blackLoadingMs,
                  presentCount: this._presentCount,
                  realPackDrawsThisFrame: 0,
                  fallbackDrawsThisFrame: 0,
                  drawsThisFrame: 0,
                  source: renderResult.source || 'bink2',
                  maxLuma: this._lastSampledMaxLuma,
                  suppressedBlackFrames: this._suppressedBlackFrames,
                  gpuAdapter: this._gpuAdapter,
                });
              } else if (this._requireNonBlackFrame || splashCheck.evaluated) {
                this._suppressedBlackFrames += 1;
              }
            }
          }
          const finalPresenter = this.binkPresenter || bridge.presenter || null;
          return {
            ok: true,
            presented: true,
            reason: 'ok',
            frameIndex: renderResult.frameIndex,
            framesPresented: this.frameCount,
            splashFramesPresented: this._splashFramesPresented,
            firstPresent: isFirstPresent,
            width: renderResult.width,
            height: renderResult.height,
            targetWidth,
            targetHeight,
            source: renderResult.source,
            viewport: renderResult.viewport || null,
            timestampMs: renderResult.timestampMs,
            chromaOrder:
              finalPresenter && typeof finalPresenter.chromaOrder === 'string'
                ? finalPresenter.chromaOrder
                : requestedChromaOrder,
            loadOp:
              finalPresenter && typeof finalPresenter.loadOp === 'string'
                ? finalPresenter.loadOp
                : requestedLoadOp,
          };
        }

        return {
          ok: false,
          presented: false,
          reason: (renderResult && renderResult.reason) || 'not-presented',
          frameIndex:
            pushed && pushed.frameIndex !== undefined ? pushed.frameIndex : requestedFrameIndex,
          framesPresented: this.frameCount,
          splashFramesPresented: this._splashFramesPresented,
          firstPresent: false,
          queueDepth: renderResult && renderResult.queueDepth !== undefined ? renderResult.queueDepth : 0,
        };
      }

      // Direct BinkVideoPresenter fallback when SplashFrameBridge is disabled/unavailable
      // but an explicit BinkVideoPresenter instance was injected.
      if (requestedLoadOp !== null && typeof presenter.setLoadOp === 'function') {
        presenter.setLoadOp(requestedLoadOp);
      }

      const format = normalizeSplashPixelFormat(framePayload.format, framePayload);
      const rawPlanes =
        framePayload.planes && typeof framePayload.planes === 'object' ? framePayload.planes : {};
      const width =
        framePayload.width !== undefined
          ? framePayload.width
          : externalImage && externalImage.width;
      const height =
        framePayload.height !== undefined
          ? framePayload.height
          : externalImage && externalImage.height;

      if (format === 'rgba8') {
        if (requestedChromaOrder !== null && typeof presenter.setChromaOrder === 'function') {
          presenter.setChromaOrder(requestedChromaOrder);
        }
        const rgba =
          rawPlanes.rgba !== undefined
            ? rawPlanes.rgba
            : rawPlanes.data !== undefined
              ? rawPlanes.data
              : framePayload.rgba !== undefined
                ? framePayload.rgba
                : framePayload.data !== undefined
                  ? framePayload.data
                  : externalImage && externalImage.data;
        if (framePayload.strideBytes !== undefined) {
          presenter.uploadRgbaFrame(rgba, width, height, { strideBytes: framePayload.strideBytes });
        } else {
          presenter.uploadRgbaFrame(rgba, width, height);
        }
      } else if (format === 'nv12' && typeof presenter.uploadNv12Frame === 'function') {
        const nv12 = {
          y:
            rawPlanes.y !== undefined
              ? rawPlanes.y
              : rawPlanes.yPlane !== undefined
                ? rawPlanes.yPlane
                : framePayload.y !== undefined
                  ? framePayload.y
                  : framePayload.yPlane,
          uv:
            rawPlanes.uv !== undefined
              ? rawPlanes.uv
              : rawPlanes.uvPlane !== undefined
                ? rawPlanes.uvPlane
                : framePayload.uv !== undefined
                  ? framePayload.uv
                  : framePayload.uvPlane,
          width,
          height,
        };
        if (framePayload.yStride !== undefined) nv12.yStride = framePayload.yStride;
        if (framePayload.uvStride !== undefined) nv12.uvStride = framePayload.uvStride;
        if (framePayload.colorSpace) nv12.colorSpace = framePayload.colorSpace;
        if (requestedColorRange !== undefined) nv12.colorRange = requestedColorRange;
        if (requestedChromaOrder !== null) nv12.chromaOrder = requestedChromaOrder;
        presenter.uploadNv12Frame(nv12);
      } else {
        const yuv = {
          y: rawPlanes.y !== undefined ? rawPlanes.y : framePayload.y,
          u: rawPlanes.u !== undefined ? rawPlanes.u : framePayload.u,
          v: rawPlanes.v !== undefined ? rawPlanes.v : framePayload.v,
          width,
          height,
        };
        if (format === 'yuva420p') {
          yuv.a = rawPlanes.a !== undefined ? rawPlanes.a : framePayload.a;
        }
        if (framePayload.yStride !== undefined) yuv.yStride = framePayload.yStride;
        if (framePayload.uvStride !== undefined) yuv.uvStride = framePayload.uvStride;
        if (framePayload.uStride !== undefined) yuv.uStride = framePayload.uStride;
        if (framePayload.vStride !== undefined) yuv.vStride = framePayload.vStride;
        if (framePayload.aStride !== undefined) yuv.aStride = framePayload.aStride;
        if (framePayload.colorSpace) {
          yuv.colorSpace = framePayload.colorSpace;
        }
        if (requestedColorRange !== undefined) {
          yuv.colorRange = requestedColorRange;
        }
        if (requestedChromaOrder !== null) {
          yuv.chromaOrder = requestedChromaOrder;
        }
        presenter.uploadYuv420Frame(yuv);
      }

      const encoder = this.device.createCommandEncoder(BINK_ENCODER_DESC);
      const directPresentOpts = this._binkDirectPresentOptsScratch;
      let hasDirectPresentOpts = false;
      if (requestedLoadOp !== null) {
        directPresentOpts.loadOp = requestedLoadOp;
        hasDirectPresentOpts = true;
      } else if (directPresentOpts.loadOp !== undefined) {
        delete directPresentOpts.loadOp;
      }
      const clearColorOpt =
        opts.clearColor !== undefined ? opts.clearColor : framePayload.clearColor;
      if (clearColorOpt !== undefined) {
        directPresentOpts.clearColor = clearColorOpt;
        hasDirectPresentOpts = true;
      } else if (directPresentOpts.clearColor !== undefined) {
        delete directPresentOpts.clearColor;
      }
      const opacityOpt = opts.opacity !== undefined ? opts.opacity : framePayload.opacity;
      if (opacityOpt !== undefined) {
        directPresentOpts.opacity = opacityOpt;
        hasDirectPresentOpts = true;
      } else if (directPresentOpts.opacity !== undefined) {
        delete directPresentOpts.opacity;
      }
      const presentResult = hasDirectPresentOpts
        ? presenter.presentFrame(encoder, targetView, targetWidth, targetHeight, directPresentOpts)
        : presenter.presentFrame(encoder, targetView, targetWidth, targetHeight);
      if (this.device.queue && typeof this.device.queue.submit === 'function') {
        const cmdBuf = encoder && typeof encoder.finish === 'function' ? encoder.finish() : encoder;
        const submitStartMs = this._gpuStallWarnMs > 0 ? this._getNowMs() : 0;
        this.device.queue.submit([cmdBuf]);
        if (this._gpuStallWarnMs > 0) {
          this._maybeWarnGpuStall('queue.submit', submitStartMs, null, 1, 0);
        }
      }

      if (presentResult && presentResult.presented) {
        const frameIndex =
          requestedFrameIndex !== null ? requestedFrameIndex : this._splashFramesPresented;
        const prevFrameCount = this.frameCount;
        this.frameCount += 1;
        this._splashFramesPresented += 1;
        const isFirstPresent = prevFrameCount === 0;
        const timestampMs =
          typeof framePayload.timestampMs === 'number' && Number.isFinite(framePayload.timestampMs)
            ? framePayload.timestampMs
            : renderNowMs !== null
              ? renderNowMs
              : Date.now();
        if (isFirstPresent) {
          this._notifyFirstPresent({
            frameIndex,
            framesPresented: this.frameCount,
            width,
            height,
            source: framePayload.source || 'raw',
            timestampMs,
          });
        }
        const hasDirectBinkYuvPlanes = Boolean(
          framePayload &&
            typeof framePayload === 'object' &&
            ((framePayload.planes &&
              typeof framePayload.planes === 'object' &&
              (framePayload.planes.y !== undefined ||
                framePayload.planes.yPlane !== undefined)) ||
              framePayload.y !== undefined ||
              framePayload.yPlane !== undefined)
        );
        if (
          !this._firstVisibleFrameNotified &&
          (this._requireNonBlackFrame ||
            (this._requireNonBlackFrameExplicit !== false && hasDirectBinkYuvPlanes))
        ) {
          const directCheck =
            isDiagnostic && this._requireNonBlackFrame
              ? { evaluated: true, visible: true, maxLuma: 235 }
              : this._evaluateNonBlackLuminance(framePayload, width, height);
          if (directCheck.evaluated) {
            this._lastSampledMaxLuma = Math.max(
              0,
              Math.min(255, Math.floor(directCheck.maxLuma || 0))
            );
          }
          if (directCheck.visible) {
            const visibleTs = this._getNowMs();
            const firstPresentMs =
              this._firstPresentMs !== null ? this._firstPresentMs : visibleTs;
            const blackLoadingMs = Math.max(0, visibleTs - firstPresentMs);
            this._firstVisibleFrameMs = visibleTs;
            this._blackLoadingMs = blackLoadingMs;
            this._notifyFirstVisibleFrame({
              type: 'first-visible-frame',
              firstVisibleFrame: true,
              firstVisibleFrameMs: visibleTs,
              firstPresentMs,
              blackLoadingMs,
              presentCount: this._presentCount,
              realPackDrawsThisFrame: 0,
              fallbackDrawsThisFrame: 0,
              drawsThisFrame: 0,
              source: framePayload.source || 'bink2',
              maxLuma: this._lastSampledMaxLuma,
              suppressedBlackFrames: this._suppressedBlackFrames,
              gpuAdapter: this._gpuAdapter,
            });
          } else if (this._requireNonBlackFrame || directCheck.evaluated) {
            this._suppressedBlackFrames += 1;
          }
        }
        return {
          ok: true,
          presented: true,
          reason: 'ok',
          frameIndex,
          framesPresented: this.frameCount,
          splashFramesPresented: this._splashFramesPresented,
          firstPresent: isFirstPresent,
          width,
          height,
          targetWidth,
          targetHeight,
          source: framePayload.source || 'raw',
          viewport: presentResult.viewport || null,
          timestampMs,
          chromaOrder:
            presentResult.chromaOrder !== undefined
              ? presentResult.chromaOrder
              : typeof presenter.chromaOrder === 'string'
                ? presenter.chromaOrder
                : requestedChromaOrder,
          loadOp:
            presentResult.loadOp !== undefined
              ? presentResult.loadOp
              : typeof presenter.loadOp === 'string'
                ? presenter.loadOp
                : requestedLoadOp,
        };
      }

      return {
        ok: false,
        presented: false,
        reason: 'not-presented',
        frameIndex: requestedFrameIndex,
        framesPresented: this.frameCount,
        splashFramesPresented: this._splashFramesPresented,
        firstPresent: false,
      };
    } catch (err) {
      return {
        ok: false,
        presented: false,
        reason: 'error',
        error: String(err),
        frameIndex: requestedFrameIndex,
        framesPresented: this.frameCount,
        splashFramesPresented: this._splashFramesPresented,
        firstPresent: false,
      };
    }
  }

  /** G66: the runtime's memory, where read-back results are written. */
  attachGuestMemory(buffer) {
    this._guestMemory = buffer || null;
  }

  _handleReadback(opcode, view, payloadOffset, payloadBytes) {
    const req = decodeReadbackPayload(opcode, view, payloadOffset, payloadBytes);
    if (!req) {
      this._recordOpValidationError(
        opcode === OPCODES.READBACK_BUFFER ? 'ReadbackBuffer' : 'ReadbackTexture',
        `truncated payload (${payloadBytes}B)`
      );
      return;
    }
    this.readbackStats.requested += 1;
    this._readbackFlushWanted = true;
    // Runs in stream order with the frame's passes: after the work recorded before it.
    this._passCommands.push({
      type: 'readback',
      run: (pc) => this._recordReadbackCopy(pc, req),
    });
  }

  /**
   * G66: the title may block on a fence guarding a read-back, and in the browser a read-back only
   * completes once its copy is submitted and mapped. Submit everything recorded up to the last
   * read-back now (no present) so the maps start without waiting for a Present packet. The
   * remaining commands stay queued for the frame.
   * @returns {boolean} True when a flush ran.
   */
  _flushReadbacksNow() {
    this._readbackFlushWanted = false;
    if (this._destroyed || !this.device || !this.ctx || this._deviceRecovering) return false;
    const cmds = this._passCommands;
    let last = -1;
    for (let i = cmds.length - 1; i >= 0; i -= 1) {
      if (cmds[i].type === 'readback') {
        last = i;
        break;
      }
    }
    if (last < 0) return false;
    const tailLen = cmds.length - (last + 1);
    if (tailLen === 0) {
      // Common path: read-back is the last recorded command, so no tail references remain in
      // `_passCommandPool`. Reset the pool cleanly after submit so multi-readback loading phases
      // without Present never accumulate pooled slots or retain old pipeline/bindGroup refs.
      try {
        this._flushPresent({ readbackOnly: true });
      } finally {
        this._clearPassCommands();
      }
    } else {
      // Preserve tail commands (and keep `_passCommandPoolUsed` intact so any pooled slots in
      // `tail` are never overwritten before the frame's Present calls `_clearPassCommands()`).
      const tail = cmds.slice(last + 1);
      cmds.length = last + 1;
      try {
        this._flushPresent({ readbackOnly: true });
      } finally {
        cmds.length = tail.length;
        for (let i = 0; i < tail.length; i += 1) {
          cmds[i] = tail[i];
        }
        this._passCommands = cmds;
      }
    }
    this._earlyReadbackFlushes = (this._earlyReadbackFlushes || 0) + 1;
    return true;
  }

  _recordReadbackCopy(pc, req) {
    pc.endPass();
    const entry = { req, staging: null, layout: null };
    this._recordedReadbacks.push(entry);
    const encoder = pc.encoder;
    const usage =
      (typeof GPUBufferUsage !== 'undefined' ? GPUBufferUsage.MAP_READ : BUFFER_USAGE.MAP_READ) |
      (typeof GPUBufferUsage !== 'undefined' ? GPUBufferUsage.COPY_DST : BUFFER_USAGE.COPY_DST);
    if (req.kind === 'buffer') {
      const src = this.buffers.get(req.bufferId);
      if (!src || !encoder) return;
      const layout = readbackStagingLayout(req);
      if (typeof src.size === 'number' && layout.copyOffset + layout.stagingSize > src.size) {
        layout.stagingSize = Math.max(0, src.size - layout.copyOffset) & ~3;
      }
      if (layout.stagingSize <= 0 || layout.skip + req.size > layout.stagingSize) return;
      const staging = this.device.createBuffer({ size: layout.stagingSize, usage });
      encoder.copyBufferToBuffer(src, layout.copyOffset, staging, 0, layout.stagingSize);
      entry.staging = staging;
      entry.layout = layout;
      return;
    }
    const tex = this.textures.get(req.textureId);
    if (!tex || !encoder) return;
    const layout = readbackStagingLayout(req, textureFormatBlockHeight(tex.format));
    if (!(layout.rowBytes > 0) || layout.rowBytes > req.bytesPerRow) return;
    const staging = this.device.createBuffer({ size: layout.stagingSize, usage });
    encoder.copyTextureToBuffer(
      { texture: tex, mipLevel: req.mipLevel, origin: { x: req.originX, y: req.originY, z: req.originZ } },
      { buffer: staging, bytesPerRow: layout.stagingRow, rowsPerImage: layout.stagingRowsPerImage },
      { width: req.width, height: req.height, depthOrArrayLayers: Math.max(1, req.depthOrLayers) }
    );
    entry.staging = staging;
    entry.layout = layout;
  }

  /** After the frame's submit: map every staging buffer, then deliver (or fail) each read-back. */
  _startRecordedReadbacks(submitted) {
    const entries = this._recordedReadbacks;
    this._recordedReadbacks = [];
    const mapRead = typeof GPUMapMode !== 'undefined' ? GPUMapMode.READ : 0x0001;
    for (const entry of entries) {
      const finish = (pieces) => {
        const ok = deliverReadback(this._guestMemory, entry.req, pieces);
        if (ok) {
          this.readbackStats.delivered += 1;
          this.readbackStats.bytes += entry.req.size;
        } else {
          this.readbackStats.failed += 1;
        }
      };
      const staging = entry.staging;
      if (!submitted || !staging || typeof staging.mapAsync !== 'function') {
        finish(null);
        if (staging && typeof staging.destroy === 'function') staging.destroy();
        continue;
      }
      staging.mapAsync(mapRead).then(
        () => {
          let pieces = null;
          try {
            const mapped = new Uint8Array(staging.getMappedRange());
            pieces = readbackPieces(entry.req, entry.layout, mapped);
            finish(pieces);
          } catch (err) {
            this._recordOpValidationError('Readback', err);
            if (pieces === null) finish(null);
          }
          try {
            staging.unmap();
            staging.destroy();
          } catch (_) {
            // Ignore teardown errors.
          }
        },
        () => {
          try {
            if (typeof staging.destroy === 'function') {
              staging.destroy();
            }
          } catch (_) {
            // Ignore teardown errors on failed readback staging buffers.
          }
          finish(null);
        }
      );
    }
  }

  _acquirePassCommand(type) {
    const idx = this._passCommandPoolUsed;
    this._passCommandPoolUsed = idx + 1;
    let slot;
    if (idx < this._passCommandPool.length) {
      slot = this._passCommandPool[idx];
    } else {
      slot = createPassCommandSlot();
      this._passCommandPool.push(slot);
    }
    slot.type = type;
    slot._wgpuMeta = null;
    return slot;
  }

  _acquireSetBindGroup2Meta(groupIndex, bindGroupId) {
    const idx = this._setBindGroup2MetaPoolUsed;
    this._setBindGroup2MetaPoolUsed = idx + 1;
    let meta;
    if (idx < this._setBindGroup2MetaPool.length) {
      meta = this._setBindGroup2MetaPool[idx];
      meta.groupIndex = groupIndex;
      meta.bindGroupId = bindGroupId;
    } else {
      meta = { kind: 'setBindGroup2', groupIndex, bindGroupId };
      this._setBindGroup2MetaPool.push(meta);
    }
    return meta;
  }

  _clearPassCommands() {
    const used = this._passCommandPoolUsed;
    const pool = this._passCommandPool;
    for (let i = 0; i < used; i += 1) {
      const slot = pool[i];
      slot.pipeline = null;
      slot.bindGroup = null;
      slot._wgpuMeta = null;
    }
    this._passCommandPoolUsed = 0;
    this._setBindGroup2MetaPoolUsed = 0;
    if (pool.length > MAX_PASS_COMMAND_POOL_SIZE) {
      pool.length = MAX_PASS_COMMAND_POOL_SIZE;
    }
    if (this._setBindGroup2MetaPool.length > MAX_PASS_COMMAND_POOL_SIZE) {
      this._setBindGroup2MetaPool.length = MAX_PASS_COMMAND_POOL_SIZE;
    }
    this._reusablePassCommands.length = 0;
    if (this._passCommands !== this._reusablePassCommands) {
      this._passCommands = this._reusablePassCommands;
    }
    let ext = this.extension;
    while (ext) {
      if (typeof ext._releaseFramePools === 'function') {
        ext._releaseFramePools();
      }
      ext = ext.next || null;
    }
  }

  _getCachedDataView(buffer) {
    if (this._cachedCmdBuffer === buffer && this._cachedCmdView !== null) {
      return this._cachedCmdView;
    }
    const view = new DataView(buffer);
    this._cachedCmdBuffer = buffer;
    this._cachedCmdView = view;
    return view;
  }

  _resetFrameState() {
    this._earlyFlushedThisFrame = false;
    this._readbackFlushWanted = false;
    this._clearColor = DEFAULT_CLEAR_COLOR;
    this._clearPassCommands();
    this._resetPerFrameDrawCounters();
  }

  _ensurePendingEncoder() {
    if (!this._pendingEncoder && this.device && typeof this.device.createCommandEncoder === 'function') {
      this._pendingEncoder = this.device.createCommandEncoder();
      this._hasPendingUploadCommands = false;
    }
    return this._pendingEncoder;
  }

  _flushPendingUploads(retireStaging = true) {
    if (!this._pendingEncoder) {
      return false;
    }
    const encoder = this._pendingEncoder;
    this._pendingEncoder = null;
    const hadCommands = this._hasPendingUploadCommands;
    this._hasPendingUploadCommands = false;

    if (!hadCommands || !this.device || !this.device.queue) {
      return false;
    }

    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.prepareSubmit === 'function'
    ) {
      this.stagingPool.prepareSubmit();
    }
    const uploadSubmitStartMs = this._gpuStallWarnMs > 0 ? this._getNowMs() : 0;
    this.device.queue.submit([encoder.finish()]);
    if (this._gpuStallWarnMs > 0) {
      this._maybeWarnGpuStall('queue.submit', uploadSubmitStartMs, null, 0, 0);
    }
    this._drainDeferredGpuDestroys();
    if (
      retireStaging &&
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.endFrame === 'function'
    ) {
      this.stagingPool.endFrame();
    }
    return true;
  }

  _advanceUniformRingForFrame(frameIndex) {
    if (!this.uniformRing || this.uniformRing._destroyed) {
      return false;
    }
    if (this._uniformRingFrameIndex === frameIndex) {
      return false;
    }
    if (typeof this.uniformRing.advanceFrame === 'function') {
      try {
        this.uniformRing.advanceFrame(frameIndex);
        this._uniformRingFrameIndex = frameIndex;
        return true;
      } catch (_) {
        return false;
      }
    }
    if (typeof this.uniformRing.beginFrame === 'function') {
      try {
        this.uniformRing.beginFrame(frameIndex);
        this._uniformRingFrameIndex = frameIndex;
        return true;
      } catch (_) {
        return false;
      }
    }
    return false;
  }

  _evaluateNonBlackLuminance(input, fallbackWidth, fallbackHeight, extraOpts = null) {
    if (input === null || input === undefined) {
      return { evaluated: false, visible: false, maxLuma: 0 };
    }
    if (typeof input === 'boolean') {
      return { evaluated: true, visible: input, maxLuma: input ? 255 : 0 };
    }

    const opts = extraOpts && typeof extraOpts === 'object' ? extraOpts : null;
    const minVisibleLuma =
      opts && typeof opts.minVisibleLuma === 'number' && Number.isFinite(opts.minVisibleLuma)
        ? opts.minVisibleLuma
        : this._minVisibleLuma;
    const minVisiblePixels =
      opts && Number.isInteger(opts.minVisiblePixels) && opts.minVisiblePixels >= 1
        ? opts.minVisiblePixels
        : this._minVisiblePixels;

    const runPixelCheck = (pixels, width, height, probeOverrides) => {
      const checkFn = resolveHasVisibleNonBlackPixelsFn();
      if (!checkFn || !pixels) {
        return { evaluated: false, visible: false, maxLuma: 0 };
      }
      const scratch = this._lumaProbeScratch;
      scratch.visible = false;
      scratch.maxLuma = 0;
      scratch.visiblePixels = 0;
      scratch.sampledPixels = 0;
      try {
        const probeOpts = {
          minVisibleLuma,
          minVisiblePixels,
          earlyExit: false,
          out: scratch,
          ...(probeOverrides || {}),
        };
        const visible = Boolean(checkFn(pixels, width, height, probeOpts));
        return {
          evaluated: scratch.sampledPixels > 0 || visible,
          visible,
          maxLuma: scratch.maxLuma,
        };
      } catch (_) {
        return { evaluated: false, visible: false, maxLuma: 0 };
      }
    };

    if (
      input instanceof Uint8Array ||
      input instanceof Uint8ClampedArray ||
      ArrayBuffer.isView(input) ||
      input instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer)
    ) {
      return runPixelCheck(input, fallbackWidth, fallbackHeight, opts);
    }

    if (typeof input === 'object') {
      if (
        typeof input.hasVisibleNonBlackPixels === 'boolean' ||
        typeof input.visible === 'boolean' ||
        typeof input.nonBlack === 'boolean'
      ) {
        const visible =
          typeof input.hasVisibleNonBlackPixels === 'boolean'
            ? input.hasVisibleNonBlackPixels
            : typeof input.visible === 'boolean'
              ? input.visible
              : input.nonBlack;
        const maxLuma =
          typeof input.maxLuma === 'number' && Number.isFinite(input.maxLuma)
            ? Math.max(0, Math.min(255, Math.floor(input.maxLuma)))
            : visible
              ? 255
              : 0;
        return { evaluated: true, visible, maxLuma };
      }

      const w =
        typeof input.width === 'number' && Number.isFinite(input.width) && input.width > 0
          ? Math.floor(input.width)
          : fallbackWidth;
      const h =
        typeof input.height === 'number' && Number.isFinite(input.height) && input.height > 0
          ? Math.floor(input.height)
          : fallbackHeight;

      const planes = input.planes && typeof input.planes === 'object' ? input.planes : null;
      const rawRgba =
        (planes && (planes.rgba !== undefined ? planes.rgba : planes.data)) ??
        (input.rgba !== undefined
          ? input.rgba
          : input.pixels !== undefined
            ? input.pixels
            : input.pixelData !== undefined
              ? input.pixelData
              : input.data);
      if (rawRgba !== undefined && rawRgba !== null) {
        const overrides = { bytesPerPixel: 4 };
        if (input.strideBytes !== undefined) {
          overrides.strideBytes = input.strideBytes;
        }
        if (input.bgra === true || input.format === 'bgra8' || input.format === 'bgra8unorm') {
          overrides.bgra = true;
        }
        return runPixelCheck(rawRgba, w, h, overrides);
      }

      const rawY =
        (planes && (planes.y !== undefined ? planes.y : planes.yPlane)) ??
        (input.y !== undefined ? input.y : input.yPlane);
      if (rawY !== undefined && rawY !== null) {
        const overrides = { bytesPerPixel: 1 };
        if (input.yStride !== undefined) {
          overrides.strideBytes = input.yStride;
        }
        if (input.colorRange === 'limited' || input.colorRange === 'full') {
          overrides.colorRange = input.colorRange;
        }
        return runPixelCheck(rawY, w, h, overrides);
      }

      const extImg = input.image || input.imageBitmap || input.bitmap || null;
      if (extImg && typeof extImg === 'object' && extImg.data !== undefined && extImg.data !== null) {
        return runPixelCheck(extImg.data, extImg.width || w, extImg.height || h, {
          bytesPerPixel: 4,
        });
      }

      if (typeof input.maxLuma === 'number' && Number.isFinite(input.maxLuma)) {
        const maxLuma = Math.max(0, Math.min(255, Math.floor(input.maxLuma)));
        const visiblePixels =
          typeof input.visiblePixels === 'number' && Number.isFinite(input.visiblePixels)
            ? Math.floor(input.visiblePixels)
            : null;
        const visible =
          maxLuma >= minVisibleLuma &&
          (visiblePixels === null || visiblePixels >= minVisiblePixels);
        return { evaluated: true, visible, maxLuma };
      }
    }

    return { evaluated: false, visible: false, maxLuma: 0 };
  }

  _evaluateCommandBufferNonBlackGate(markResult, activeSwapchain, width, height, summary) {
    const pendingMeta = this._pendingFrameLuminance;
    this._pendingFrameLuminance = null;

    if (pendingMeta !== null && pendingMeta !== undefined) {
      const res = this._evaluateNonBlackLuminance(pendingMeta, width, height);
      if (res.evaluated) {
        return res;
      }
    }

    if (markResult && typeof markResult === 'object') {
      const res = this._evaluateNonBlackLuminance(markResult, width, height);
      if (res.evaluated) {
        return res;
      }
    }

    if (activeSwapchain && typeof activeSwapchain === 'object') {
      if (typeof activeSwapchain.getLastFrameLuminance === 'function') {
        try {
          const swapMeta = activeSwapchain.getLastFrameLuminance();
          const res = this._evaluateNonBlackLuminance(swapMeta, width, height);
          if (res.evaluated) {
            return res;
          }
        } catch (_) {
          // Ignore throwing swapchain luminance probes.
        }
      }
      if (
        typeof activeSwapchain.hasVisibleNonBlackPixels === 'boolean' ||
        typeof activeSwapchain.lastSampledMaxLuma === 'number'
      ) {
        const res = this._evaluateNonBlackLuminance(
          {
            hasVisibleNonBlackPixels: activeSwapchain.hasVisibleNonBlackPixels,
            maxLuma: activeSwapchain.lastSampledMaxLuma,
          },
          width,
          height
        );
        if (res.evaluated) {
          return res;
        }
      }
    }

    if (typeof this._nonBlackPixelProbe === 'function') {
      try {
        const probeOut = this._nonBlackPixelProbe({
          width,
          height,
          frameCount: this.frameCount,
          presentCount: this._presentCount,
          summary,
        });
        const res = this._evaluateNonBlackLuminance(probeOut, width, height);
        if (res.evaluated) {
          return res;
        }
      } catch (_) {
        // Probe failures must not crash presentation.
      }
    }

    return { evaluated: false, visible: false, maxLuma: 0 };
  }

  _clearSwapchainProbeTimeout() {
    if (this._swapchainProbeTimeoutTimer !== null) {
      try {
        clearTimeout(this._swapchainProbeTimeoutTimer);
      } catch (_) {
        // Ignore timer cleanup errors.
      }
      this._swapchainProbeTimeoutTimer = null;
    }
  }

  /**
   * W338: Destroy the pre-allocated staging GPUBuffer used by the throttled async
   * swapchain non-black luminance probe. Called automatically as soon as
   * `first-visible-frame` fires, on device reset, and on `destroy()`.
   */
  _destroySwapchainProbeBuffer() {
    this._clearSwapchainProbeTimeout();
    this._swapchainProbeGeneration += 1;
    this._swapchainProbeInFlight = false;
    const buf = this._swapchainProbeBuffer;
    this._swapchainProbeBuffer = null;
    this._swapchainProbeBufferDevice = null;
    if (buf && typeof buf.destroy === 'function') {
      try {
        buf.destroy();
      } catch (_) {
        // Ignore teardown errors on already-destroyed or lost GPUBuffers.
      }
    }
  }

  /**
   * W338: Lazily pre-allocate and reuse a single 1024-byte (4 * 256) MAP_READ | COPY_DST
   * GPUBuffer for swapchain luminance probing. Returns null (and caches unsupported) if
   * the device or buffer mock does not expose mapAsync/getMappedRange/unmap.
   */
  _ensureSwapchainProbeBuffer() {
    if (this._destroyed || this._firstVisibleFrameNotified) {
      return null;
    }
    if (
      this._swapchainProbeBuffer &&
      this._swapchainProbeBufferDevice === this.device &&
      !this._swapchainProbeBuffer.destroyed
    ) {
      return this._swapchainProbeBuffer;
    }
    if (this._swapchainProbeBuffer) {
      this._destroySwapchainProbeBuffer();
    }
    if (this._swapchainProbeSupported === false) {
      return null;
    }
    if (!this.device || typeof this.device.createBuffer !== 'function') {
      this._swapchainProbeSupported = false;
      return null;
    }
    const mapReadFlag =
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.MAP_READ === 'number'
        ? GPUBufferUsage.MAP_READ
        : BUFFER_USAGE.MAP_READ;
    const copyDstFlag =
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : BUFFER_USAGE.COPY_DST;
    let buf = null;
    try {
      buf = this.device.createBuffer({
        label: 'playgta6-swapchain-luma-probe',
        size: SWAPCHAIN_PROBE_BUFFER_SIZE,
        usage: mapReadFlag | copyDstFlag,
      });
    } catch (_) {
      this._swapchainProbeSupported = false;
      return null;
    }
    if (
      !buf ||
      typeof buf.mapAsync !== 'function' ||
      typeof buf.getMappedRange !== 'function' ||
      typeof buf.unmap !== 'function'
    ) {
      if (buf && typeof buf.destroy === 'function') {
        try {
          buf.destroy();
        } catch (_) {
          // Ignore cleanup errors on mock buffers.
        }
      }
      this._swapchainProbeSupported = false;
      return null;
    }
    this._swapchainProbeBuffer = buf;
    this._swapchainProbeBufferDevice = this.device;
    this._swapchainProbeSupported = true;
    return buf;
  }

  /**
   * W338: Check whether the active WebGPU device, command encoder, and swapchain texture
   * support non-black GPU readback probing (copyTextureToBuffer + COPY_SRC + MAP_READ).
   * Returns null on mock/headless devices without making any extra createBuffer calls.
   */
  _resolveSwapchainProbeCapability(encoder, texture) {
    if (
      this._destroyed ||
      this._firstVisibleFrameNotified ||
      this._requireNonBlackFrameExplicit === false ||
      this._swapchainProbeSupported === false
    ) {
      return null;
    }
    if (
      this._swapchain &&
      !this._swapchain.isDestroyed &&
      (this._swapchain.hasVisibleContent === true ||
        this._swapchain.hasFirstVisibleFrame === true)
    ) {
      this._destroySwapchainProbeBuffer();
      return null;
    }
    if (
      !encoder ||
      typeof encoder.copyTextureToBuffer !== 'function' ||
      !this.device ||
      typeof this.device.createBuffer !== 'function'
    ) {
      return null;
    }
    const copySrcBit =
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_SRC === 'number'
        ? GPUTextureUsage.COPY_SRC
        : TEXTURE_USAGE.COPY_SRC;
    if (!texture || typeof texture.usage !== 'number' || (texture.usage & copySrcBit) === 0) {
      return null;
    }
    const fmt =
      typeof texture.format === 'string' && texture.format.length > 0
        ? texture.format
        : this.format;
    if (
      fmt !== 'bgra8unorm' &&
      fmt !== 'rgba8unorm' &&
      fmt !== 'bgra8unorm-srgb' &&
      fmt !== 'rgba8unorm-srgb'
    ) {
      return null;
    }
    const texWidth =
      typeof texture.width === 'number' && Number.isFinite(texture.width) && texture.width > 0
        ? Math.floor(texture.width)
        : this._viewportWidth > 0
          ? this._viewportWidth
          : 0;
    const texHeight =
      typeof texture.height === 'number' && Number.isFinite(texture.height) && texture.height > 0
        ? Math.floor(texture.height)
        : this._viewportHeight > 0
          ? this._viewportHeight
          : 0;
    if (texWidth < 1 || texHeight < 1) {
      return null;
    }
    const buffer = this._ensureSwapchainProbeBuffer();
    if (!buffer) {
      return null;
    }
    return { buffer, texWidth, texHeight };
  }

  /**
   * W350: Record a swapchain luminance probe copy/map/read error. After
   * `SWAPCHAIN_PROBE_MAX_CONSECUTIVE_ERRORS` (3) consecutive failures, log a sanitized warning
   * once, mark `_swapchainProbeSupported = false`, destroy the staging buffer, and fall back
   * cleanly so the splash overlay can never hang permanently.
   *
   * @param {string} stage
   * @param {unknown} err
   * @param {object | null} [probeTarget]
   * @returns {boolean} True when the probe was disabled after hitting the consecutive error cap.
   */
  _recordSwapchainProbeError(stage, err, probeTarget = null, opts = null) {
    this._swapchainProbeErrorCount += 1;
    this._swapchainProbeConsecutiveErrors += 1;
    if (this._swapchainProbeConsecutiveErrors < SWAPCHAIN_PROBE_MAX_CONSECUTIVE_ERRORS) {
      return false;
    }
    if (!this._swapchainProbeFallbackWarned) {
      this._swapchainProbeFallbackWarned = true;
      const detail = sanitizeValidationErrorMessage(sanitizeGpuErrorDetail(err));
      this._logDiagnostic(
        'warn',
        `[wgpu_worker] swapchain luminance probe disabled after ${this._swapchainProbeConsecutiveErrors} consecutive ${stage || 'probe'} failures${detail ? `: ${detail}` : ''}`
      );
    }
    this._swapchainProbeSupported = false;
    this._destroySwapchainProbeBuffer();

    const allowImmediateFallback = !(opts && opts.allowImmediateFallback === false);
    const realPackDrawsThisFrame = probeTarget
      ? toNonNegativeInt(probeTarget.realPackDrawsThisFrame, 0)
      : 0;
    const fallbackDrawsThisFrame = probeTarget
      ? toNonNegativeInt(probeTarget.fallbackDrawsThisFrame, 0)
      : 0;
    const drawsThisFrame = probeTarget ? toNonNegativeInt(probeTarget.drawsThisFrame, 0) : 0;
    const hasPresentedDraw =
      this._presentCount > 0 &&
      (this._realPackDrawCount > 0 ||
        this._drawCallCount > 0 ||
        realPackDrawsThisFrame > 0 ||
        drawsThisFrame > 0);
    if (
      allowImmediateFallback &&
      !this._destroyed &&
      !this._firstVisibleFrameNotified &&
      this._requireNonBlackFrameExplicit !== true &&
      hasPresentedDraw
    ) {
      const visibleTs = this._getNowMs();
      const firstPresentMs = this._firstPresentMs !== null ? this._firstPresentMs : visibleTs;
      const blackLoadingMs = Math.max(0, visibleTs - firstPresentMs);
      this._firstVisibleFrameMs = visibleTs;
      this._blackLoadingMs = blackLoadingMs;
      if (this._swapchain && !this._swapchain.isDestroyed) {
        if (typeof this._swapchain.confirmVisibleFrame === 'function') {
          try {
            this._swapchain.confirmVisibleFrame({
              timestampMs: visibleTs,
              realPackDraws: realPackDrawsThisFrame || this._realPackDrawCount,
            });
          } catch (_) {
            // Swapchain telemetry must never break frame presentation.
          }
        } else if (typeof this._swapchain.markFrameContent === 'function') {
          try {
            this._swapchain.markFrameContent({
              hasVisibleContent: true,
              hasDraws: true,
              nonClearDraw: true,
              realPackDraws: realPackDrawsThisFrame || this._realPackDrawCount,
              timestampMs: visibleTs,
            });
          } catch (_) {
            // Ignore custom mock swapchain errors.
          }
        }
      }
      this._notifyFirstVisibleFrame({
        type: 'first-visible-frame',
        firstVisibleFrame: true,
        firstVisibleFrameMs: visibleTs,
        firstPresentMs,
        blackLoadingMs,
        presentCount: this._presentCount,
        realPackDrawsThisFrame,
        fallbackDrawsThisFrame,
        drawsThisFrame,
        source: 'swapchain-probe-fallback',
        maxLuma: this._lastSampledMaxLuma,
        suppressedBlackFrames: this._suppressedBlackFrames,
        swapchainProbeErrorCount: this._swapchainProbeErrorCount,
        gpuAdapter: this._gpuAdapter,
      });
    }
    return true;
  }

  /**
   * W338 / W350: Encode 8 tiny 1-row horizontal strip copies from the presented swapchain texture
   * into the pre-allocated 2048-byte staging buffer covering center/left/right/bottom-right.
   */
  _encodeSwapchainProbeCopies(encoder, texture, probeCap, startMs, summary) {
    const { buffer, texWidth, texHeight } = probeCap;
    const stripWidth = Math.max(1, Math.min(SWAPCHAIN_PROBE_STRIP_MAX_WIDTH, texWidth));
    const maxOriginX = Math.max(0, texWidth - stripWidth);
    const probeSummary = {
      realPackDrawsThisFrame: summary ? summary.realPackDrawsThisFrame : 0,
      fallbackDrawsThisFrame: summary ? summary.fallbackDrawsThisFrame : 0,
      drawsThisFrame: summary ? summary.drawsThisFrame : 0,
    };
    const copyOrigin = this._probeCopyOriginScratch;
    const copySrc = this._probeCopySrcScratch;
    const copyDst = this._probeCopyDstScratch;
    const copySize = this._probeCopySizeScratch;
    copySrc.texture = texture;
    copySrc.mipLevel = 0;
    copyDst.buffer = buffer;
    copyDst.bytesPerRow = SWAPCHAIN_PROBE_BYTES_PER_ROW;
    copyDst.rowsPerImage = 1;
    copySize.width = stripWidth;
    copySize.height = 1;
    copySize.depthOrArrayLayers = 1;
    try {
      for (let s = 0; s < SWAPCHAIN_PROBE_STRIPS.length; s += 1) {
        const strip = SWAPCHAIN_PROBE_STRIPS[s];
        copyOrigin.x = Math.max(0, Math.min(maxOriginX, Math.floor(maxOriginX * strip.xFraction)));
        copyOrigin.y = Math.max(
          0,
          Math.min(texHeight - 1, Math.floor(texHeight * strip.yFraction))
        );
        copyOrigin.z = 0;
        copyDst.offset = s * SWAPCHAIN_PROBE_BYTES_PER_ROW;
        encoder.copyTextureToBuffer(copySrc, copyDst, copySize);
      }
    } catch (err) {
      this._recordSwapchainProbeError('copyTextureToBuffer', err, probeSummary, {
        allowImmediateFallback: false,
      });
      return null;
    } finally {
      copySrc.texture = null;
      copyDst.buffer = null;
    }
    return {
      buffer,
      stripWidth,
      stripCount: SWAPCHAIN_PROBE_STRIPS.length,
      startMs,
      realPackDrawsThisFrame: probeSummary.realPackDrawsThisFrame,
      fallbackDrawsThisFrame: probeSummary.fallbackDrawsThisFrame,
      drawsThisFrame: probeSummary.drawsThisFrame,
    };
  }

  /**
   * W338 / W350: Start the non-overlapping async GPUBuffer.mapAsync readback after queue.submit().
   * Scans RGB channels (i+0, i+1, i+2; ignoring alpha i+3) for both bgra8unorm and rgba8unorm.
   * Immediately emits `first-visible-frame` and destroys the staging buffer on any pixel > threshold.
   */
  _startSwapchainProbeReadback(probeTarget) {
    if (
      !probeTarget ||
      this._destroyed ||
      this._firstVisibleFrameNotified ||
      this._swapchainProbeInFlight
    ) {
      return false;
    }
    const buf = probeTarget.buffer;
    if (!buf || buf !== this._swapchainProbeBuffer || buf.destroyed) {
      return false;
    }
    const gen = this._swapchainProbeGeneration;
    const mapReadMode =
      typeof GPUMapMode !== 'undefined' && typeof GPUMapMode.READ === 'number'
        ? GPUMapMode.READ
        : 0x0001;
    this._swapchainProbeInFlight = true;
    this._swapchainProbeLastStartMs = probeTarget.startMs;
    this._swapchainProbeCount += 1;

    let mapPromise;
    try {
      mapPromise = buf.mapAsync(mapReadMode);
    } catch (err) {
      if (gen === this._swapchainProbeGeneration) {
        this._swapchainProbeInFlight = false;
        this._recordSwapchainProbeError('mapAsync', err, probeTarget);
      }
      return false;
    }
    if (!mapPromise || typeof mapPromise.then !== 'function') {
      if (gen === this._swapchainProbeGeneration) {
        this._swapchainProbeInFlight = false;
        this._recordSwapchainProbeError('mapAsync', 'mapAsync did not return a Promise', probeTarget);
      }
      return false;
    }

    const threshold = this._minVisibleLuma;
    const stripWidth = probeTarget.stripWidth;
    const stripCount = probeTarget.stripCount;
    let settled = false;
    const clearWatchdog = () => {
      if (gen === this._swapchainProbeGeneration) {
        this._clearSwapchainProbeTimeout();
      }
    };
    const timeoutMs = this._swapchainProbeTimeoutMs;
    if (timeoutMs > 0 && typeof setTimeout === 'function') {
      this._clearSwapchainProbeTimeout();
      const timer = setTimeout(() => {
        if (this._swapchainProbeTimeoutTimer === timer) {
          this._swapchainProbeTimeoutTimer = null;
        }
        if (
          settled ||
          this._destroyed ||
          this._firstVisibleFrameNotified ||
          gen !== this._swapchainProbeGeneration
        ) {
          return;
        }
        settled = true;
        this._swapchainProbeTimeoutCount += 1;
        try {
          if (buf && !buf.destroyed && typeof buf.unmap === 'function') {
            buf.unmap();
          }
        } catch (_) {
          // Ignore unmap errors on a stalled/lost GPUBuffer.
        }
        this._destroySwapchainProbeBuffer();
        this._recordSwapchainProbeError(
          'mapAsync-timeout',
          `mapAsync timed out after ${timeoutMs}ms`,
          probeTarget
        );
      }, timeoutMs);
      if (timer && typeof timer.unref === 'function') {
        timer.unref();
      }
      this._swapchainProbeTimeoutTimer = timer;
    }

    mapPromise.then(
      () => {
        if (settled) {
          try {
            if (!buf.destroyed && typeof buf.unmap === 'function') {
              buf.unmap();
            }
          } catch (_) {
            // Ignore late unmap on an already-timed-out/destroyed buffer.
          }
          return;
        }
        settled = true;
        clearWatchdog();
        if (
          this._destroyed ||
          gen !== this._swapchainProbeGeneration ||
          buf !== this._swapchainProbeBuffer
        ) {
          try {
            if (!buf.destroyed && typeof buf.unmap === 'function') {
              buf.unmap();
            }
          } catch (_) {
            // Ignore unmap on already-destroyed buffer.
          }
          if (gen === this._swapchainProbeGeneration) {
            this._swapchainProbeInFlight = false;
          }
          return;
        }

        let maxRgb = 0;
        let foundNonBlack = false;
        let readOk = false;
        let readErr = null;
        try {
          const range = buf.getMappedRange(0, SWAPCHAIN_PROBE_BUFFER_SIZE);
          if (range) {
            const bytes = new Uint8Array(range);
            readOk = true;
            for (let s = 0; s < stripCount; s += 1) {
              const rowBase = s * SWAPCHAIN_PROBE_BYTES_PER_ROW;
              const rowEnd = Math.min(bytes.byteLength, rowBase + stripWidth * 4);
              for (let i = rowBase; i + 2 < rowEnd; i += 4) {
                const c0 = bytes[i];
                const c1 = bytes[i + 1];
                const c2 = bytes[i + 2];
                const pxMax = c0 >= c1 ? (c0 >= c2 ? c0 : c2) : c1 >= c2 ? c1 : c2;
                if (pxMax > maxRgb) {
                  maxRgb = pxMax;
                }
                if (pxMax > threshold) {
                  foundNonBlack = true;
                }
              }
            }
          }
        } catch (err) {
          readErr = err;
        } finally {
          try {
            if (!buf.destroyed && typeof buf.unmap === 'function') {
              buf.unmap();
            }
          } catch (_) {
            // Ignore unmap failures on lost/destroyed buffers.
          }
          if (gen === this._swapchainProbeGeneration) {
            this._swapchainProbeInFlight = false;
          }
        }

        if (this._destroyed || this._firstVisibleFrameNotified) {
          return;
        }
        if (!readOk) {
          this._recordSwapchainProbeError(
            'getMappedRange',
            readErr || 'getMappedRange returned empty range',
            probeTarget
          );
          return;
        }

        this._swapchainProbeConsecutiveErrors = 0;
        this._lastSampledMaxLuma = maxRgb;
        if (foundNonBlack) {
          const visibleTs = this._getNowMs();
          const firstPresentMs =
            this._firstPresentMs !== null ? this._firstPresentMs : visibleTs;
          const blackLoadingMs = Math.max(0, visibleTs - firstPresentMs);
          this._firstVisibleFrameMs = visibleTs;
          this._blackLoadingMs = blackLoadingMs;
          // W348: Confirm visible content on the attached swapchain now that GPU readback
          // verified non-black pixels, keeping swapchain telemetry aligned with the probe.
          if (this._swapchain && !this._swapchain.isDestroyed) {
            if (typeof this._swapchain.confirmVisibleFrame === 'function') {
              try {
                this._swapchain.confirmVisibleFrame({
                  timestampMs: visibleTs,
                  realPackDraws: probeTarget.realPackDrawsThisFrame,
                });
              } catch (_) {
                // Swapchain telemetry must never break frame presentation.
              }
            } else if (typeof this._swapchain.markFrameContent === 'function') {
              try {
                this._swapchain.markFrameContent({
                  hasVisibleContent: true,
                  hasDraws: true,
                  nonClearDraw: true,
                  realPackDraws: probeTarget.realPackDrawsThisFrame,
                  timestampMs: visibleTs,
                });
              } catch (_) {
                // Ignore custom mock swapchain errors.
              }
            }
          }
          this._notifyFirstVisibleFrame({
            type: 'first-visible-frame',
            firstVisibleFrame: true,
            firstVisibleFrameMs: visibleTs,
            firstPresentMs,
            blackLoadingMs,
            presentCount: this._presentCount,
            realPackDrawsThisFrame: probeTarget.realPackDrawsThisFrame,
            fallbackDrawsThisFrame: probeTarget.fallbackDrawsThisFrame,
            drawsThisFrame: probeTarget.drawsThisFrame,
            source: 'swapchain-probe',
            maxLuma: maxRgb,
            suppressedBlackFrames: this._suppressedBlackFrames,
            gpuAdapter: this._gpuAdapter,
          });
        } else {
          this._suppressedBlackFrames += 1;
        }
      },
      (err) => {
        if (settled) {
          return;
        }
        settled = true;
        clearWatchdog();
        if (gen === this._swapchainProbeGeneration) {
          this._swapchainProbeInFlight = false;
          if (!this._destroyed && !this._firstVisibleFrameNotified) {
            this._recordSwapchainProbeError('mapAsync', err || 'mapAsync rejected', probeTarget);
          }
        }
      }
    );

    return true;
  }

  _beginSubsystemsForFrame(frameIndex) {
    this._frameStartMs = this._getNowMs();
    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.beginFrame === 'function'
    ) {
      this.stagingPool.beginFrame(frameIndex);
    }
    this._advanceUniformRingForFrame(frameIndex);
    if (
      this.passTimer &&
      !this.passTimer.isDestroyed &&
      typeof this.passTimer.beginFrame === 'function'
    ) {
      this.passTimer.beginFrame();
    }
    if (
      this.gpuRecorder &&
      typeof this.gpuRecorder.isSpikeTriggerEnabled === 'function' &&
      this.gpuRecorder.isSpikeTriggerEnabled() &&
      typeof this.gpuRecorder.beginFrame === 'function'
    ) {
      try {
        this.gpuRecorder.beginFrame();
      } catch (_) {
        // Recorder telemetry must never break frame begin.
      }
    }
    if (
      this.resourceRegistry &&
      typeof this.resourceRegistry.beginFrame === 'function'
    ) {
      const current =
        typeof this.resourceRegistry.getStats === 'function'
          ? this.resourceRegistry.getStats().currentFrame
          : 0;
      const targetFrame = Math.max(frameIndex, current);
      this.resourceRegistry.beginFrame(targetFrame);
    }
    if (
      this.resourceTable &&
      !this.resourceTable._destroyed &&
      typeof this.resourceTable.beginFrame === 'function'
    ) {
      try {
        this.resourceTable.beginFrame();
      } catch (_) {
        // ResourceTable beginFrame errors must not break frame begin.
      }
    }
    if (
      this.drawExecutor &&
      typeof this.drawExecutor.beginFrame === 'function'
    ) {
      this.drawExecutor.beginFrame();
    }
    this._resetPerFrameDrawCounters();
    this._frameBegunForIndex = frameIndex;
  }

  _ensureFrameBegun() {
    if (this._frameBegunForIndex !== this.frameCount) {
      this._beginSubsystemsForFrame(this.frameCount);
    }
  }

  _endSubsystemsForFrame(workDoneFence = null, frameHints = null) {
    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.endFrame === 'function'
    ) {
      this.stagingPool.endFrame();
    }
    if (
      this.uniformRing &&
      !this.uniformRing._destroyed &&
      typeof this.uniformRing.endFrame === 'function'
    ) {
      try {
        if (workDoneFence && typeof workDoneFence.then === 'function') {
          this._uniformRingEndFrameOptsScratch.fence = workDoneFence;
          this.uniformRing.endFrame(this._uniformRingEndFrameOptsScratch);
          this._uniformRingEndFrameOptsScratch.fence = null;
        } else {
          this.uniformRing.endFrame();
        }
      } catch (_) {
        this._uniformRingEndFrameOptsScratch.fence = null;
        // UniformRing endFrame errors must not break presentation.
      }
    }
    if (
      this.resourceRegistry &&
      typeof this.resourceRegistry.endFrame === 'function'
    ) {
      this.resourceRegistry.endFrame(workDoneFence);
    }
    if (
      this.resourceTable &&
      !this.resourceTable._destroyed &&
      typeof this.resourceTable.endFrame === 'function'
    ) {
      try {
        this.resourceTable.endFrame(this.device ? this.device.queue : undefined);
      } catch (_) {
        // ResourceTable endFrame errors must not break presentation.
      }
    }
    if (
      this.rtPool &&
      !this.rtPool.isDestroyed &&
      typeof this.rtPool.endFrame === 'function'
    ) {
      this.rtPool.endFrame();
    }
    const hints = frameHints && typeof frameHints === 'object' ? frameHints : null;
    if (hints && (hints.sceneTransition || hints.resolutionChanged)) {
      this.trimTransientMemory({ maxIdleFrames: this._oversizedRtReclaimIdleFrames });
    } else if (
      this.rtPool &&
      !this.rtPool.isDestroyed &&
      typeof this.rtPool.reclaimOversizedIdle === 'function'
    ) {
      const shouldReclaim = Boolean(
        (hints && hints.isIdleFrame) ||
          (this.frameCount > 0 &&
            this.frameCount % this._oversizedRtReclaimIntervalFrames === 0)
      );
      if (shouldReclaim) {
        this.reclaimOversizedRenderTargets(this._oversizedRtReclaimIdleFrames);
      }
    }
    this._frameBegunForIndex = -1;
  }

  /**
   * Return a work-done fence for the most recent submission, or null when the queue fence is
   * unavailable. The returned promise never rejects: a torn-down/lost GPU instance
   * (`OperationError: A valid external Instance reference no longer exists.`) either throws
   * synchronously or rejects the fence, and both are recorded as a sanitized validation event
   * instead of escaping as an uncaught worker exception or unhandled rejection.
   *
   * @returns {Promise<void> | null}
   */
  _getWorkDoneFence() {
    let queue = null;
    try {
      queue = this.device ? this.device.queue : null;
    } catch (err) {
      this._noteWorkDoneFenceFailure(err);
      return null;
    }
    if (!queue || typeof queue.onSubmittedWorkDone !== 'function') {
      return null;
    }
    let fence;
    try {
      fence = queue.onSubmittedWorkDone();
    } catch (err) {
      this._noteWorkDoneFenceFailure(err);
      return null;
    }
    if (!fence || typeof fence.then !== 'function') {
      return null;
    }
    try {
      return Promise.resolve(fence).then(
        this._onWorkDoneFenceResolved,
        this._onWorkDoneFenceRejected
      );
    } catch (err) {
      this._noteWorkDoneFenceFailure(err);
      return null;
    }
  }

  /**
   * Record a failed/rejected `onSubmittedWorkDone()` fence once per processor lifetime as a
   * sanitized validation event; subsequent failures only bump the counter.
   *
   * @param {unknown} err
   */
  _noteWorkDoneFenceFailure(err) {
    this._workDoneFenceFailures += 1;
    if (this._destroyed) {
      return;
    }
    // W100: a torn-down instance / lost device on the fence path triggers recovery.
    this._maybeSignalDeviceLostFromError(err, 'work-done-fence');
    if (this._workDoneFenceFailures !== 1) {
      return;
    }
    try {
      this._recordOpValidationError('OnSubmittedWorkDone', sanitizeGpuErrorDetail(err));
    } catch (_) {
      // Never let telemetry recording escape from fence handling.
    }
  }

  /**
   * Record a failed present encode/submit (e.g. torn-down WebGPU instance) as a sanitized
   * `Present:` validation event without throwing.
   *
   * @param {unknown} err
   */
  _notePresentSubmitFailure(err) {
    this._presentSubmitFailures += 1;
    try {
      this._recordOpValidationError('Present', sanitizeGpuErrorDetail(err));
    } catch (_) {
      // Never let telemetry recording escape from the present path.
    }
    // W100: a torn-down instance / lost device on the present path triggers recovery.
    this._maybeSignalDeviceLostFromError(err, 'present');
  }

  /**
   * W100: emit a concise diagnostic line through the configured logger (default `console`).
   *
   * @param {'info' | 'warn' | 'error'} level
   * @param {string} message
   */
  _logDiagnostic(level, message) {
    const logger = this._logger;
    if (!logger) {
      return;
    }
    try {
      if (typeof logger === 'function') {
        logger(level, message);
      } else if (typeof logger[level] === 'function') {
        logger[level](message);
      } else if (typeof logger.log === 'function') {
        logger.log(message);
      }
    } catch (_) {
      // Logging must never break the render loop.
    }
  }

  _resetStallProfileScratch() {
    this._stallOpCount.fill(0);
    this._stallOpMs.fill(0);
    this._stallOpMaxMs.fill(0);
    this._stallOpBytes.fill(0);
    this._stallBatchPackets = 0;
    this._stallBatchBytes = 0;
    this._stallReplayMs = 0;
    this._stallSlowPacketLogs = 0;
    const pb = this._stallPresentBreakdown;
    pb.count = 0;
    pb.getTexMs = 0;
    pb.cmdsMs = 0;
    pb.blitMs = 0;
    pb.probeMs = 0;
    pb.finishMs = 0;
    pb.submitMs = 0;
    pb.otherMs = 0;
  }

  _formatStallProfileSummary() {
    const counts = this._stallOpCount;
    const ms = this._stallOpMs;
    const maxMs = this._stallOpMaxMs;
    const bytes = this._stallOpBytes;
    const entries = [];
    for (let op = 0; op < 65; op += 1) {
      if (counts[op] > 0) {
        entries.push({
          op,
          count: counts[op],
          ms: ms[op],
          maxMs: maxMs[op],
          bytes: bytes[op],
        });
      }
    }
    entries.sort((a, b) => b.ms - a.ms || b.count - a.count);
    const topParts = [];
    const limit = Math.min(5, entries.length);
    for (let i = 0; i < limit; i += 1) {
      const e = entries[i];
      const name = e.op < 64 ? opcodeLabel(e.op) : 'Other';
      const mbPart = e.bytes >= 1024 * 1024 ? `,${(e.bytes / (1024 * 1024)).toFixed(1)}MB` : '';
      topParts.push(`${name}:${e.count}/${Math.round(e.ms)}ms(max=${Math.round(e.maxMs)}ms${mbPart})`);
    }
    const pb = this._stallPresentBreakdown;
    const presentPart =
      pb.count > 0
        ? ` present(${pb.count})=[getTex=${Math.round(pb.getTexMs)}ms,cmds=${Math.round(pb.cmdsMs)}ms,blit=${Math.round(pb.blitMs)}ms,probe=${Math.round(pb.probeMs)}ms,finish=${Math.round(pb.finishMs)}ms,submit=${Math.round(pb.submitMs)}ms,other=${Math.round(pb.otherMs)}ms]`
        : '';
    return `packets=${this._stallBatchPackets} bytes=${this._stallBatchBytes} replayMs=${Math.round(this._stallReplayMs)} topOps=[${topParts.join(',')}]${presentPart}`;
  }

  /**
   * Wave-5 (#663, #1077): Emit a rate-limited sanitized warning whenever a single GPU
   * operation (`executeCommandBuffer`, `queue.submit`, `createShaderModule`,
   * `createRenderPipeline`) exceeds `_gpuStallWarnMs` (default 1000ms).
   */
  _maybeWarnGpuStall(opName, startMs, shaderHash = null, drawsCount = null, dispatchesCount = null, detail = null) {
    if (this._gpuStallWarnMs <= 0 || typeof startMs !== 'number') {
      return false;
    }
    const endMs = this._getNowMs();
    const durationMs = endMs - startMs;
    if (!Number.isFinite(durationMs) || durationMs <= this._gpuStallWarnMs) {
      return false;
    }
    const op = typeof opName === 'string' && opName.length > 0 ? opName : 'gpu';
    const lastWarnMs = this._gpuStallWarnLastByOp.get(op);
    if (
      lastWarnMs !== undefined &&
      this._gpuStallWarnMinIntervalMs > 0 &&
      endMs - lastWarnMs < this._gpuStallWarnMinIntervalMs
    ) {
      return false;
    }
    this._gpuStallWarnLastByOp.set(op, endMs);
    this._gpuStallWarnCount += 1;
    const cleanShader =
      shaderHash !== null && shaderHash !== undefined && String(shaderHash).length > 0
        ? sanitizeValidationErrorMessage(String(shaderHash))
        : null;
    const draws =
      typeof drawsCount === 'number' && Number.isFinite(drawsCount)
        ? Math.max(0, Math.floor(drawsCount))
        : this._frameDrawsThisFrame;
    const dispatches =
      typeof dispatchesCount === 'number' && Number.isFinite(dispatchesCount)
        ? Math.max(0, Math.floor(dispatchesCount))
        : this._computeDispatchCount;
    const shaderPart = cleanShader ? ` shader=${cleanShader}` : '';
    const cleanDetail =
      typeof detail === 'string' && detail.length > 0 ? ` ${detail}` : '';
    this._logDiagnostic(
      'warn',
      `[wgpu_worker] GPU stall: op=${op} durationMs=${Math.round(durationMs)} frame=${this.frameCount}${shaderPart} draws=${draws} dispatches=${dispatches}${cleanDetail}`
    );
    return true;
  }

  /**
   * W100: attach a `device.lost` observer to `device` once. Losses of a device that is no longer
   * current (replaced by recovery) and intentional `device.destroy()` (`reason: 'destroyed'`) or
   * processor teardown are ignored.
   *
   * @param {GPUDevice | null | undefined} device
   * @returns {boolean}
   */
  _hookDeviceLost(device) {
    if (!device || typeof device !== 'object' || device === this._deviceLostHookedDevice) {
      return false;
    }
    let lost;
    try {
      lost = device.lost;
    } catch (_) {
      return false;
    }
    if (!lost || typeof lost.then !== 'function') {
      return false;
    }
    this._deviceLostHookedDevice = device;
    const generation = this._deviceGeneration;
    try {
      Promise.resolve(lost).then(
        (info) => {
          this._handleDeviceLostInfo(info, device, generation);
        },
        () => {
          // A rejected `lost` promise carries no actionable information.
        }
      );
    } catch (_) {
      return false;
    }
    return true;
  }

  _handleDeviceLostInfo(info, device, generation) {
    if (this._destroyed || generation !== this._deviceGeneration || this.device !== device) {
      return;
    }
    let reason = null;
    try {
      reason = info && typeof info === 'object' ? info.reason : null;
    } catch (_) {
      reason = null;
    }
    if (reason === 'destroyed') {
      // `device.destroy()` was called intentionally by the host; do not recover.
      return;
    }
    this._signalDeviceLost(sanitizeDeviceLostInfo(info), 'device.lost');
  }

  /**
   * W100: signal device loss when `err` is a torn-down instance / lost device error.
   *
   * @param {unknown} err
   * @param {string} source
   * @returns {boolean}
   */
  _maybeSignalDeviceLostFromError(err, source) {
    try {
      if (!isDeviceLostError(err)) {
        return false;
      }
      return this._signalDeviceLost(sanitizeGpuErrorDetail(err), source);
    } catch (_) {
      return false;
    }
  }

  /**
   * W100: record a device loss (counted once per device generation) and notify the registered
   * device-loss listener (the worker handler schedules bounded recovery). Never throws.
   *
   * @param {string} reason Sanitized reason string.
   * @param {string} source
   * @returns {boolean}
   */
  _signalDeviceLost(reason, source) {
    if (this._destroyed || this._deviceRecovering) {
      return false;
    }
    const firstForGeneration = this._deviceLostGeneration !== this._deviceGeneration;
    if (firstForGeneration) {
      this._deviceLostGeneration = this._deviceGeneration;
      this._deviceLost = true;
      this._deviceLostCount += 1;
      this._lastDeviceLostReason = typeof reason === 'string' && reason.length > 0 ? reason : null;
      const pipeStats =
        this.asyncPipelineCache && typeof this.asyncPipelineCache.getStats === 'function'
          ? this.asyncPipelineCache.getStats()
          : null;
      const pipeDetail = pipeStats
        ? `, asyncPipeInFlight=${pipeStats.inFlight || 0}, asyncPipeQueued=${pipeStats.queued || 0}, asyncPipeCompiled=${pipeStats.compiled || 0}, asyncPipeMaxMs=${Math.round(pipeStats.maxCompileMs || 0)}`
        : '';
      this._logDiagnostic(
        'warn',
        `[wgpu_worker] WebGPU device lost (${source}, gen=${this._deviceGeneration}, replayEntries=${this._replayLog.size}${pipeDetail}, rangedShaderActive=${this._activeRangedShaderCompiles}, rangedShaderQueued=${this._rangedShaderCompileQueue.length}): ${this._lastDeviceLostReason || 'unknown'}`
      );
    }
    const listener = this._deviceLossListener;
    if (listener) {
      try {
        listener({
          reason: this._lastDeviceLostReason,
          source,
          generation: this._deviceGeneration,
          firstForGeneration,
        });
      } catch (_) {
        // Listener failures must not escape into the render loop.
      }
    }
    return true;
  }

  /**
   * W100: register the device-loss listener (called synchronously on loss; `null` clears it).
   *
   * @param {((event: { reason: string | null, source: string, generation: number, firstForGeneration: boolean }) => void) | null} listener
   */
  setDeviceLossListener(listener) {
    this._deviceLossListener = typeof listener === 'function' ? listener : null;
  }

  get isDeviceLost() {
    return this._deviceLost;
  }

  get isDeviceRecovering() {
    return this._deviceRecovering;
  }

  /** W103: adapt the processor logger to AsyncPipelineCache's { warn, debug }(message, detail). */
  _createAsyncPipelineLogger() {
    const log = (level) => (message, detail) => {
      const suffix = detail && typeof detail.message === 'string' ? `: ${detail.message}` : '';
      this._logDiagnostic(level, `[wgpu_worker] ${message}${suffix}`);
    };
    return { warn: log('warn'), debug: log('debug') };
  }

  /** W103: point the async pipeline cache at `device`, settling pending compiles as stale. */
  _resetAsyncPipelineCache(device) {
    const cache = this.asyncPipelineCache;
    if (!cache || typeof cache.reset !== 'function') {
      return;
    }
    try {
      cache.reset(device || null);
    } catch (err) {
      this._logDiagnostic(
        'warn',
        `[wgpu_worker] async pipeline cache reset failed: ${sanitizeGpuErrorDetail(err)}`
      );
    }
  }

  /**
   * W103: whether pipeline `id` is still compiling asynchronously (draws/dispatches that use it
   * are skipped without a validation error until it is ready).
   *
   * @param {number} id
   * @returns {boolean}
   */
  isAsyncPipelinePending(id) {
    const hasLivePipeline = this.pipelines.has(id) || this.computePipelines.has(id);
    if (!hasLivePipeline) {
      if (this._pendingRangedPipelines && this._pendingRangedPipelines.has(id)) {
        return true;
      }
      const renderStateExt = this._getRenderStateExtension();
      if (
        renderStateExt &&
        ((renderStateExt.pendingAsyncRenderPipelines &&
          renderStateExt.pendingAsyncRenderPipelines.has(id)) ||
          (renderStateExt.pendingAsyncComputePipelines &&
            renderStateExt.pendingAsyncComputePipelines.has(id)))
      ) {
        return true;
      }
    }
    const cache = this.asyncPipelineCache;
    if (!cache || typeof cache.isPending !== 'function') {
      return false;
    }
    try {
      return cache.isPending(id) === true;
    } catch (_) {
      return false;
    }
  }

  /**
   * W103: AsyncPipelineCache counters plus draws/dispatches skipped while a pipeline compiled,
   * or null when async pipeline compilation is disabled.
   */
  getAsyncPipelineStats() {
    const cache = this.asyncPipelineCache;
    if (!cache || typeof cache.getStats !== 'function') {
      return null;
    }
    let stats;
    try {
      stats = cache.getStats();
    } catch (_) {
      return null;
    }
    let renderStateSkips = 0;
    const ext = this.extension;
    if (ext && ext.telemetry && Number.isInteger(ext.telemetry.pendingAsyncPipelineSkips)) {
      renderStateSkips = ext.telemetry.pendingAsyncPipelineSkips;
    }
    return {
      ...(stats && typeof stats === 'object' ? stats : {}),
      prioritizeGraphics:
        stats && typeof stats.prioritizeGraphics === 'boolean'
          ? stats.prioritizeGraphics
          : typeof cache.prioritizeGraphics === 'boolean'
            ? cache.prioritizeGraphics
            : true,
      maxConcurrentCompiles:
        stats && Number.isInteger(stats.maxConcurrentCompiles)
          ? stats.maxConcurrentCompiles
          : Number.isInteger(cache.maxConcurrentCompiles)
            ? cache.maxConcurrentCompiles
            : stats && Number.isInteger(stats.maxConcurrent)
              ? stats.maxConcurrent
              : DEFAULT_MAX_CONCURRENT_PIPELINE_COMPILES,
      pendingAsyncPipelineSkips: this._pendingAsyncPipelineSkips + renderStateSkips,
    };
  }

  /**
   * W100: sanitized device-loss telemetry counters.
   */
  getDeviceLossStats() {
    return {
      deviceLostCount: this._deviceLostCount,
      deviceRecoveredCount: this._deviceRecoveredCount,
      deviceRecoveryFailedCount: this._deviceRecoveryFailedCount,
      lastDeviceLostReason: this._lastDeviceLostReason,
      deviceRecovering: this._deviceRecovering,
      deviceLost: this._deviceLost,
      droppedWhileRecovering: this._droppedWhileRecoveringCount,
      replayedResourceCount: this._replayedResourceCount,
      replayedBufferWriteCount: this._replayedBufferWriteCount,
      replayLogEntries: this._replayLog.size,
      replayBindGroupEvictions: this._replayBindGroupEvictions,
      replayLogOverflowCount: this._replayLogOverflowCount,
    };
  }

  /**
   * W100: enter recovery mode. Submissions are dropped (no throw, no frameCount bump) until
   * `endDeviceRecovery()` is called.
   *
   * @returns {boolean}
   */
  beginDeviceRecovery() {
    if (this._destroyed) {
      return false;
    }
    this._deviceRecovering = true;
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._clearPassCommands();
    return true;
  }

  /**
   * W100: leave recovery mode and update recovered / failed counters.
   *
   * @param {boolean} success
   */
  endDeviceRecovery(success) {
    const wasRecovering = this._deviceRecovering;
    this._deviceRecovering = false;
    if (!wasRecovering) {
      return;
    }
    if (success && !this._destroyed) {
      this._deviceRecoveredCount += 1;
      this._deviceLost = false;
    } else {
      this._deviceRecoveryFailedCount += 1;
    }
  }

  /**
   * W346: toggle live-resource replay recording at runtime (or via worker `init`).
   * Passing `{ clear: true }` while disabling also releases previously recorded packets
   * and shadow buffers immediately.
   *
   * @param {boolean} enabled
   * @param {{ clear?: boolean } | null} [opts]
   * @returns {boolean}
   */
  setEnableResourceReplay(enabled, opts = null) {
    this._resourceReplayEnabled = enabled !== false;
    if (!this._resourceReplayEnabled && opts && opts.clear === true) {
      this._replayLog.clear();
      this._replayBindGroupKeys.clear();
      this._replayLogBytes = 0;
      this._replayAsyncDeltaKeys.clear();
      this._replayAsyncDestroyedIds.clear();
      this._replayAsyncShadowIds.clear();
      this._shadowBuffers.clear();
      this._shadowBufferBytes = 0;
    }
    return this._resourceReplayEnabled;
  }

  get resourceReplayEnabled() {
    return this._resourceReplayEnabled;
  }

  /**
   * Wave-5 (#663, #1077): toggle WriteBuffer shadow-copying independently from lightweight
   * resource creation-packet replay. Passing `{ clear: true }` while disabling releases
   * previously buffered shadow copies immediately.
   *
   * @param {boolean} enabled
   * @param {{ clear?: boolean } | null} [opts]
   * @returns {boolean}
   */
  setEnableShadowBufferReplay(enabled, opts = null) {
    this._shadowBufferReplayEnabled = enabled !== false;
    if (!this._shadowBufferReplayEnabled && opts && opts.clear === true) {
      this._shadowBuffers.clear();
      this._shadowBufferBytes = 0;
    }
    return this._shadowBufferReplayEnabled;
  }

  get shadowBufferReplayEnabled() {
    return this._shadowBufferReplayEnabled;
  }

  _touchReplayBindGroup(id) {
    if (!this._resourceReplayEnabled || this._replayBindGroupKeys.size === 0) {
      return;
    }
    const key = this._replayBindGroupKeys.get(id);
    if (key !== undefined) {
      this._replayBindGroupKeys.delete(id);
      this._replayBindGroupKeys.set(id, key);
    }
  }

  _evictOldestReplayBindGroup() {
    const first = this._replayBindGroupKeys.entries().next();
    if (first.done) {
      return false;
    }
    const [oldestId, oldestKey] = first.value;
    this._replayBindGroupKeys.delete(oldestId);
    const evicted = this._replayLog.get(oldestKey);
    if (evicted) {
      this._replayLog.delete(oldestKey);
      this._replayLogBytes = Math.max(0, this._replayLogBytes - evicted.payload.byteLength);
      this._replayBindGroupEvictions += 1;
      if (this._replayAsyncActive && !this._replayingResources) {
        this._replayAsyncDeltaKeys.delete(oldestKey);
      }
      if (this._replayBindGroupEvictions >= this._replayEvictionWarnNext) {
        this._replayEvictionWarnNext =
          this._replayEvictionWarnNext === 1 ? 64 : this._replayEvictionWarnNext * 8;
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] replay log evicted oldest bindGroup (evictions=${this._replayBindGroupEvictions}, liveBindGroups=${this._replayBindGroupKeys.size}/${this._replayLogMaxBindGroups}, replayEntries=${this._replayLog.size}/${this._replayLogMaxEntries})`
        );
      }
      return true;
    }
    return false;
  }

  /**
   * W104: record one packet into the live-resource replay log. Creation packets are copied
   * (latest creation of an id wins), DestroyResource forgets every record of the id, and
   * WriteBuffer updates the bounded shadow copy of the buffer contents when
   * `_shadowBufferReplayEnabled` is true. Never records while `replayRecordedResources()` runs.
   */
  _recordReplayPacket(opcode, view, payloadOffset, payloadBytes) {
    if (!this._resourceReplayEnabled || this._replayingResources || payloadBytes < 4) {
      return;
    }
    if (opcode === OPCODES.WRITE_BUFFER) {
      if (this._shadowBufferReplayEnabled) {
        this._recordShadowBufferWrite(view, payloadOffset, payloadBytes);
      }
      return;
    }
    if (opcode === OPCODES.DESTROY_RESOURCE) {
      this._forgetReplayResource(view.getUint32(payloadOffset, true));
      return;
    }
    const spec = REPLAY_CREATE_OPCODES.get(opcode);
    if (!spec) {
      return;
    }
    const [kind, phase] = spec;
    const id = view.getUint32(payloadOffset, true);
    const key = `${kind}:${id}`;
    const previous = this._replayLog.get(key);
    if (previous) {
      this._replayLog.delete(key);
      this._replayLogBytes = Math.max(0, this._replayLogBytes - previous.payload.byteLength);
      if (kind === 'bindGroup') {
        this._replayBindGroupKeys.delete(id);
      }
    }
    if (kind === 'buffer') {
      // A re-created buffer starts zero-filled; its old contents must not be replayed.
      this._dropShadowBuffer(id);
    }
    if (kind === 'bindGroup') {
      while (this._replayBindGroupKeys.size >= this._replayLogMaxBindGroups) {
        if (!this._evictOldestReplayBindGroup()) {
          break;
        }
      }
    }
    if (payloadBytes <= this._replayLogMaxBytes) {
      while (
        this._replayBindGroupKeys.size > 0 &&
        (this._replayLog.size >= this._replayLogMaxEntries ||
          this._replayLogBytes + payloadBytes > this._replayLogMaxBytes)
      ) {
        if (!this._evictOldestReplayBindGroup()) {
          break;
        }
      }
    }
    if (
      this._replayLog.size >= this._replayLogMaxEntries ||
      this._replayLogBytes + payloadBytes > this._replayLogMaxBytes
    ) {
      this._replayLogOverflowCount += 1;
      if (this._replayLogOverflowCount >= this._replayOverflowWarnNext) {
        this._replayOverflowWarnNext =
          this._replayOverflowWarnNext === 1 ? 64 : this._replayOverflowWarnNext * 8;
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] replay log overflow dropped ${kind}:${id} (overflows=${this._replayLogOverflowCount}, entries=${this._replayLog.size}/${this._replayLogMaxEntries}, bytes=${this._replayLogBytes}/${this._replayLogMaxBytes})`
        );
      }
      return;
    }
    const payload = new Uint8Array(view.buffer, view.byteOffset + payloadOffset, payloadBytes).slice();
    this._replayLog.set(key, { kind, id, opcode, phase, payload });
    this._replayLogBytes += payloadBytes;
    if (kind === 'bindGroup') {
      this._replayBindGroupKeys.set(id, key);
    }
    if (this._replayAsyncActive && !this._replayingResources) {
      this._replayAsyncDeltaKeys.add(key);
      this._replayAsyncDestroyedIds.delete(id);
    }
  }

  /**
   * W104: record the resource packets of `view` starting at `offset` without executing them
   * (command buffers dropped while the device is lost / recovering).
   */
  _recordReplayPacketsFrom(view, offset, endOffset = view.byteLength) {
    if (!this._resourceReplayEnabled || this._replayingResources) {
      return;
    }
    const limit = Math.min(view.byteLength, endOffset);
    let cursor = offset;
    while (cursor + 8 <= limit) {
      const opcode = view.getUint16(cursor, true);
      const payloadBytes = view.getUint32(cursor + 4, true);
      const payloadOffset = cursor + 8;
      if (payloadOffset + payloadBytes > limit) {
        break;
      }
      this._recordReplayPacket(opcode, view, payloadOffset, payloadBytes);
      cursor = payloadOffset + payloadBytes;
    }
  }

  _forgetReplayResource(id) {
    for (let i = 0; i < REPLAY_RESOURCE_KINDS.length; i += 1) {
      const key = `${REPLAY_RESOURCE_KINDS[i]}:${id}`;
      const entry = this._replayLog.get(key);
      if (entry) {
        this._replayLog.delete(key);
        this._replayLogBytes = Math.max(0, this._replayLogBytes - entry.payload.byteLength);
      }
      if (this._replayAsyncActive && !this._replayingResources) {
        this._replayAsyncDeltaKeys.delete(key);
      }
    }
    this._replayBindGroupKeys.delete(id);
    this._dropShadowBuffer(id);
    if (this._replayAsyncActive && !this._replayingResources) {
      this._replayAsyncDestroyedIds.add(id);
      this._replayAsyncShadowIds.delete(id);
    }
  }

  _dropShadowBuffer(id) {
    const shadow = this._shadowBuffers.get(id);
    if (shadow) {
      this._shadowBuffers.delete(id);
      this._shadowBufferBytes -= shadow.data.byteLength;
    }
  }

  _recordShadowBufferWrite(view, payloadOffset, payloadBytes) {
    if (
      !this._resourceReplayEnabled ||
      !this._shadowBufferReplayEnabled ||
      this._replayingResources ||
      payloadBytes < 16
    ) {
      return;
    }
    const id = view.getUint32(payloadOffset, true);
    const dstOffset = view.getUint32(payloadOffset + 4, true);
    const dataLen = view.getUint32(payloadOffset + 8, true);
    if (16 + dataLen > payloadBytes || dataLen === 0) {
      return;
    }
    const created = this._replayLog.get(`buffer:${id}`);
    if (!created || created.payload.byteLength < 16) {
      return;
    }
    const createView = new DataView(
      created.payload.buffer,
      created.payload.byteOffset,
      created.payload.byteLength
    );
    const usage = createView.getUint32(4, true);
    const byteSize = Number(createView.getBigUint64(8, true));
    if (
      (usage & BUFFER_USAGE.COPY_DST) === 0 ||
      !Number.isSafeInteger(byteSize) ||
      byteSize < 4 ||
      byteSize > this._shadowBufferMaxBytes ||
      dstOffset >= byteSize
    ) {
      return;
    }
    let shadow = this._shadowBuffers.get(id);
    if (shadow) {
      // LRU: move to the most-recently-written end.
      this._shadowBuffers.delete(id);
    } else {
      shadow = { data: new Uint8Array(byteSize), dirtyEnd: 0 };
      this._shadowBufferBytes += byteSize;
    }
    this._shadowBuffers.set(id, shadow);
    const copyLen = Math.min(dataLen, byteSize - dstOffset);
    shadow.data.set(
      new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 16, copyLen),
      dstOffset
    );
    shadow.dirtyEnd = Math.max(shadow.dirtyEnd, dstOffset + copyLen);
    if (this._replayAsyncActive && !this._replayingResources) {
      this._replayAsyncShadowIds.add(id);
    }

    while (this._shadowBufferBytes > this._shadowBufferTotalMaxBytes && this._shadowBuffers.size > 1) {
      const oldestId = this._shadowBuffers.keys().next().value;
      this._dropShadowBuffer(oldestId);
      this._shadowBufferEvictions += 1;
    }
  }

  /** Total replayable bytes of a shadow (WebGPU writeBuffer sizes must be 4-byte multiples). */
  _shadowWriteLength(shadow) {
    const maxLen = shadow.data.byteLength - (shadow.data.byteLength % 4);
    const aligned = shadow.dirtyEnd + ((4 - (shadow.dirtyEnd % 4)) % 4);
    return Math.min(aligned, maxLen);
  }

  /**
   * W104: recreate every recorded live resource on the current (recovered) device in dependency
   * order — leaf resources (buffers, textures, samplers, shader modules), shadowed buffer
   * contents, bind group layouts, pipeline layouts, pipelines, then bind groups. Each packet is
   * isolated so one stale or invalid record never aborts the rest of the replay. Must run after
   * `rebindDevice()` and after the ABI v2 packet extensions were reinstalled on the new device.
   *
   * @returns {{ replayedResources: number, failedResources: number, replayedBufferWrites: number }}
   */
  _collectOrderedReplayEntries() {
    const phases = [];
    for (let p = 0; p < REPLAY_PHASE_COUNT; p += 1) {
      phases.push([]);
    }
    for (const entry of this._replayLog.values()) {
      phases[entry.phase - 1].push(entry);
    }
    const ordered = [];
    for (let p = 0; p < phases.length; p += 1) {
      for (const entry of phases[p]) {
        ordered.push(entry);
      }
    }
    return ordered;
  }

  replayRecordedResources() {
    const summary = { replayedResources: 0, failedResources: 0, replayedBufferWrites: 0 };
    if (
      this._destroyed ||
      this._replayingResources ||
      !this.device ||
      (this._replayLog.size === 0 && this._shadowBuffers.size === 0)
    ) {
      return summary;
    }

    const ordered = this._collectOrderedReplayEntries();
    const prevPendingHints = this._pendingCommandStreamShaderProfiles;
    const prevPendingView = this._pendingCommandStreamView;
    const prevPendingScanned = this._pendingCommandStreamScanned;
    const prevPassCommands = this._passCommands;
    this._replayingResources = true;
    try {
      this._pendingCommandStreamShaderProfiles = null;
      this._pendingCommandStreamView = null;
      this._pendingCommandStreamScanned = true;
      this._passCommands = [];
      let shadowWritesDone = false;
      for (let i = 0; i < ordered.length; i += 1) {
        const entry = ordered[i];
        if (!shadowWritesDone && entry.phase > 1) {
          summary.replayedBufferWrites += this._replayShadowBufferWrites();
          shadowWritesDone = true;
        }
        if (this._replayOneEntry(entry)) {
          summary.replayedResources += 1;
        } else {
          summary.failedResources += 1;
        }
      }
      if (!shadowWritesDone) {
        summary.replayedBufferWrites += this._replayShadowBufferWrites();
      }
    } finally {
      this._passCommands = prevPassCommands;
      this._pendingCommandStreamShaderProfiles = prevPendingHints;
      this._pendingCommandStreamView = prevPendingView;
      this._pendingCommandStreamScanned = prevPendingScanned;
      this._replayingResources = false;
    }

    this._replayedResourceCount += summary.replayedResources;
    this._replayedBufferWriteCount += summary.replayedBufferWrites;
    this._replayFailedCount += summary.failedResources;
    this._lastResourceReplay = { ...summary, deviceGeneration: this._deviceGeneration };
    this._logDiagnostic(
      'info',
      `[wgpu_worker] replayed ${summary.replayedResources} GPU resources and ${summary.replayedBufferWrites} buffer writes after device recovery` +
        (summary.failedResources > 0 ? ` (${summary.failedResources} failed)` : '')
    );
    return summary;
  }

  _collectOrderedReplayEntriesForKeys(keys) {
    const phases = [];
    for (let p = 0; p < REPLAY_PHASE_COUNT; p += 1) {
      phases.push([]);
    }
    for (const key of keys) {
      const entry = this._replayLog.get(key);
      if (entry) {
        phases[entry.phase - 1].push(entry);
      }
    }
    const ordered = [];
    for (let p = 0; p < phases.length; p += 1) {
      for (const entry of phases[p]) {
        ordered.push(entry);
      }
    }
    return ordered;
  }

  _hasAnyLiveReplayEntryForId(id) {
    for (let i = 0; i < REPLAY_RESOURCE_KINDS.length; i += 1) {
      if (this._replayLog.has(`${REPLAY_RESOURCE_KINDS[i]}:${id}`)) {
        return true;
      }
    }
    return false;
  }

  _runSyncReplayChunk(fn) {
    const prevPendingHints = this._pendingCommandStreamShaderProfiles;
    const prevPendingView = this._pendingCommandStreamView;
    const prevPendingScanned = this._pendingCommandStreamScanned;
    const prevPassCommands = this._passCommands;
    this._replayingResources = true;
    try {
      this._pendingCommandStreamShaderProfiles = null;
      this._pendingCommandStreamView = null;
      this._pendingCommandStreamScanned = true;
      this._passCommands = [];
      fn();
    } finally {
      this._passCommands = prevPassCommands;
      this._pendingCommandStreamShaderProfiles = prevPendingHints;
      this._pendingCommandStreamView = prevPendingView;
      this._pendingCommandStreamScanned = prevPendingScanned;
      this._replayingResources = false;
    }
  }

  /**
   * W105: chunked async resource replay for live device recovery. Yields between small batches
   * so Chrome's GPU IPC drains cleanly without stalling the worker or re-triggering TDR.
   *
   * Only holds `_replayingResources = true` during each synchronous chunk (never across `await`)
   * so incoming guest Create/Destroy/WriteBuffer packets that arrive during async yields are
   * still recorded into `_replayLog` and replayed in dependency order before recovery finishes.
   */
  async replayRecordedResourcesAsync(opts = {}) {
    const summary = { replayedResources: 0, failedResources: 0, replayedBufferWrites: 0, aborted: false };
    if (
      this._destroyed ||
      this._replayingResources ||
      this._replayAsyncActive ||
      !this.device ||
      (this._replayLog.size === 0 && this._shadowBuffers.size === 0)
    ) {
      return summary;
    }

    const chunkSize = resolvePositiveIntOption(opts && opts.chunkSize, REPLAY_ASYNC_CHUNK_SIZE);
    const ordered = this._collectOrderedReplayEntries();
    const startGen = this._deviceGeneration;
    const startMs = Date.now();
    const logMiB = (this._replayLogBytes / (1024 * 1024)).toFixed(1);
    this._logDiagnostic(
      'info',
      `[wgpu_worker] starting GPU resource replay (${ordered.length} entries, ${logMiB} MiB, ${this._shadowBuffers.size} shadow buffers` +
        (this._replayBindGroupEvictions > 0 ? `, bindGroupEvictions=${this._replayBindGroupEvictions}` : '') +
        (this._replayLogOverflowCount > 0 ? `, overflows=${this._replayLogOverflowCount}` : '') +
        ')'
    );

    this._replayAsyncActive = true;
    this._replayAsyncDeltaKeys.clear();
    this._replayAsyncDestroyedIds.clear();
    this._replayAsyncShadowIds.clear();
    let shadowWritesDone = false;
    try {
      for (let chunkStart = 0; chunkStart < ordered.length; chunkStart += chunkSize) {
        if (this._destroyed || !this.device || this._deviceGeneration !== startGen) {
          summary.aborted = true;
          break;
        }
        const chunkEnd = Math.min(ordered.length, chunkStart + chunkSize);
        this._runSyncReplayChunk(() => {
          for (let i = chunkStart; i < chunkEnd; i += 1) {
            if (this._destroyed || !this.device || this._deviceGeneration !== startGen) {
              summary.aborted = true;
              break;
            }
            const entry = ordered[i];
            if (this._replayLog.get(`${entry.kind}:${entry.id}`) !== entry) {
              continue;
            }
            if (!shadowWritesDone && entry.phase > 1) {
              summary.replayedBufferWrites += this._replayShadowBufferWrites();
              shadowWritesDone = true;
            }
            if (this._replayOneEntry(entry)) {
              summary.replayedResources += 1;
            } else {
              summary.failedResources += 1;
            }
          }
        });
        if (summary.aborted) {
          break;
        }
        if (chunkEnd < ordered.length) {
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }
      if (!shadowWritesDone && !summary.aborted) {
        this._runSyncReplayChunk(() => {
          summary.replayedBufferWrites += this._replayShadowBufferWrites();
        });
        shadowWritesDone = true;
      }

      // Catch up any resource creations, destroys, or shadow writes that arrived from the guest
      // while the async replay chunks above yielded. The final catch-up pass runs synchronously
      // with zero trailing `await` so no macrotask can slip between catch-up and `endDeviceRecovery`.
      const maxCatchUpPasses = 8;
      for (let pass = 0; pass < maxCatchUpPasses && !summary.aborted; pass += 1) {
        if (this._destroyed || !this.device || this._deviceGeneration !== startGen) {
          summary.aborted = true;
          break;
        }
        if (
          this._replayAsyncDestroyedIds.size === 0 &&
          this._replayAsyncDeltaKeys.size === 0 &&
          this._replayAsyncShadowIds.size === 0
        ) {
          break;
        }
        const destroyedIds = Array.from(this._replayAsyncDestroyedIds);
        const deltaKeys = Array.from(this._replayAsyncDeltaKeys);
        const shadowIds =
          this._replayAsyncShadowIds.size > 0 ? new Set(this._replayAsyncShadowIds) : null;
        this._replayAsyncDestroyedIds.clear();
        this._replayAsyncDeltaKeys.clear();
        this._replayAsyncShadowIds.clear();

        if (destroyedIds.length > 0) {
          this._runSyncReplayChunk(() => {
            for (let i = 0; i < destroyedIds.length; i += 1) {
              const id = destroyedIds[i];
              if (!this._hasAnyLiveReplayEntryForId(id)) {
                this._handleDestroyResource(id);
              }
            }
          });
        }

        const deltaOrdered = this._collectOrderedReplayEntriesForKeys(deltaKeys);
        const allowYield = pass < maxCatchUpPasses - 1 && deltaOrdered.length > chunkSize;
        let deltaShadowDone = !shadowIds || shadowIds.size === 0;
        for (let start = 0; start < deltaOrdered.length; start += allowYield ? chunkSize : deltaOrdered.length) {
          if (this._destroyed || !this.device || this._deviceGeneration !== startGen) {
            summary.aborted = true;
            break;
          }
          const end = allowYield ? Math.min(deltaOrdered.length, start + chunkSize) : deltaOrdered.length;
          this._runSyncReplayChunk(() => {
            for (let i = start; i < end; i += 1) {
              if (this._destroyed || !this.device || this._deviceGeneration !== startGen) {
                summary.aborted = true;
                break;
              }
              const entry = deltaOrdered[i];
              if (this._replayLog.get(`${entry.kind}:${entry.id}`) !== entry) {
                continue;
              }
              if (!deltaShadowDone && entry.phase > 1) {
                summary.replayedBufferWrites += this._replayShadowBufferWrites(shadowIds);
                deltaShadowDone = true;
              }
              if (this._replayOneEntry(entry)) {
                summary.replayedResources += 1;
              } else {
                summary.failedResources += 1;
              }
            }
          });
          if (summary.aborted) {
            break;
          }
          if (allowYield && end < deltaOrdered.length) {
            await new Promise((resolve) => setTimeout(resolve, 0));
          }
        }
        if (!deltaShadowDone && !summary.aborted) {
          this._runSyncReplayChunk(() => {
            summary.replayedBufferWrites += this._replayShadowBufferWrites(shadowIds);
          });
        }
      }
    } finally {
      this._replayAsyncActive = false;
      this._replayAsyncDeltaKeys.clear();
      this._replayAsyncDestroyedIds.clear();
      this._replayAsyncShadowIds.clear();
      this._replayingResources = false;
    }

    this._replayedResourceCount += summary.replayedResources;
    this._replayedBufferWriteCount += summary.replayedBufferWrites;
    this._replayFailedCount += summary.failedResources;
    this._lastResourceReplay = { ...summary, deviceGeneration: this._deviceGeneration };
    const elapsedMs = Math.max(0, Date.now() - startMs);
    this._logDiagnostic(
      'info',
      `[wgpu_worker] replayed ${summary.replayedResources} GPU resources and ${summary.replayedBufferWrites} buffer writes in ${elapsedMs}ms after device recovery` +
        (summary.failedResources > 0 ? ` (${summary.failedResources} failed)` : '') +
        (this._replayBindGroupEvictions > 0 ? ` (bindGroupEvictions=${this._replayBindGroupEvictions})` : '') +
        (this._replayLogOverflowCount > 0 ? ` (overflows=${this._replayLogOverflowCount})` : '') +
        (summary.aborted ? ' (aborted)' : '')
    );
    return summary;
  }

  _replayOneEntry(entry) {
    if (REPLAY_EXTENSION_OPCODES.has(entry.opcode) && !this.extension) {
      return false;
    }
    const entryView = new DataView(
      entry.payload.buffer,
      entry.payload.byteOffset,
      entry.payload.byteLength
    );
    const errorsBefore = this._validationErrorCount;
    try {
      this._dispatchPacket(entry.opcode, entryView, 0, entry.payload.byteLength);
    } catch (_) {
      return false;
    }
    return this._validationErrorCount === errorsBefore;
  }

  _replayShadowBufferWrites(onlyIds = null) {
    const queue = this.device && this.device.queue;
    if (!queue || typeof queue.writeBuffer !== 'function') {
      return 0;
    }
    let written = 0;
    if (onlyIds) {
      for (const id of onlyIds) {
        const shadow = this._shadowBuffers.get(id);
        const buffer = this.buffers.get(id);
        const writeLen = shadow ? this._shadowWriteLength(shadow) : 0;
        if (!buffer || writeLen <= 0) {
          continue;
        }
        try {
          queue.writeBuffer(buffer, 0, shadow.data, 0, writeLen);
          written += 1;
        } catch (_) {
          // A stale shadow never aborts the remaining replay.
        }
      }
      return written;
    }
    for (const [id, shadow] of this._shadowBuffers) {
      const buffer = this.buffers.get(id);
      const writeLen = this._shadowWriteLength(shadow);
      if (!buffer || writeLen <= 0) {
        continue;
      }
      try {
        queue.writeBuffer(buffer, 0, shadow.data, 0, writeLen);
        written += 1;
      } catch (_) {
        // A stale shadow never aborts the remaining replay.
      }
    }
    return written;
  }

  /**
   * W104: replay-log / shadow-cache counters (sanitized integers only).
   */
  getResourceReplayStats() {
    return {
      enabled: this._resourceReplayEnabled,
      recordedResourceCount: this._replayLog.size,
      recordedResourceBytes: this._replayLogBytes,
      replayLogOverflowCount: this._replayLogOverflowCount,
      replayBindGroupEvictions: this._replayBindGroupEvictions,
      shadowBufferCount: this._shadowBuffers.size,
      shadowBufferBytes: this._shadowBufferBytes,
      shadowBufferEvictions: this._shadowBufferEvictions,
      replayedResourceCount: this._replayedResourceCount,
      replayedBufferWriteCount: this._replayedBufferWriteCount,
      replayFailedCount: this._replayFailedCount,
    };
  }

  /**
   * W100: release every GPU resource handle that belongs to the (dead) current device. IDs that
   * the guest references afterwards resolve to `null` and are skipped by the existing lookup
   * paths instead of throwing. Shader WGSL sources in the pack cache are preserved.
   */
  _resetDeviceResources() {
    this._destroySwapchainProbeBuffer();
    this._swapchainProbeSupported = null;
    this._swapchainProbeLastStartMs = -Infinity;
    this._swapchainProbeConsecutiveErrors = 0;
    this._swapchainProbeFallbackWarned = false;
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._clearPassCommands();
    this._clearPresentPassBindings();
    this._frameBegunForIndex = -1;
    this._flushAllDeferredGpuDestroysImmediate();

    const safeDestroy = (resource) => {
      if (resource && typeof resource.destroy === 'function') {
        try {
          resource.destroy();
        } catch (_) {
          // Destroying objects of a lost device may throw; they are released either way.
        }
      }
    };

    if (this.resourceRegistry && typeof this.resourceRegistry.destroyAll === 'function') {
      try {
        this.resourceRegistry.destroyAll();
      } catch (_) {
        // Ignore registry teardown failures on a lost device.
      }
    }
    for (const buffer of this.buffers.values()) {
      safeDestroy(buffer);
    }
    for (const texture of this.textures.values()) {
      safeDestroy(texture);
    }
    if (Array.isArray(this._recordedReadbacks) && this._recordedReadbacks.length > 0) {
      for (let i = 0; i < this._recordedReadbacks.length; i += 1) {
        safeDestroy(this._recordedReadbacks[i] && this._recordedReadbacks[i].staging);
      }
      this._recordedReadbacks = [];
    }

    if (this._clearZeroBuffer) {
      safeDestroy(this._clearZeroBuffer);
      this._clearZeroBuffer = null;
      this._clearZeroBufferBytes = 0;
    }
    this._clearZeroScratch = null;

    this.buffers.clear();
    this._bufferMeta.clear();
    this.textures.clear();
    this._textureMeta.clear();
    this.textureViews.clear();
    this._liveBufferBytes = 0;
    this._liveTextureBytes = 0;
    this._directWriteScratch = null;
    this.samplers.clear();
    this.bindGroups.clear();
    this._bindGroupKinds.clear();
    this._bindGroupMeta.clear();
    this._v2BindGroupLayouts.clear();
    this._v2PipelineLayouts.clear();
    this.shaderModules.clear();
    this._cachedBuiltinShaderModules.clear();
    this._shaderModuleGenerations.clear();
    this._inFlightRangedShaders.clear();
    this._clearRangedShaderCompileQueue();
    this._pendingRangedPipelines.clear();
    this.pipelines.clear();
    this._pipelineMeta.clear();
    this._specializedFallbackPipelines.clear();
    this._specializedFallbackBindGroups.clear();
    this._specializedFallbackShaderModules.clear();
    this._pendingCommandStreamShaderProfiles = null;
    this._pendingCommandStreamView = null;
    this._pendingCommandStreamStartOffset = 0;
    this._pendingCommandStreamEndOffset = 0;
    this._pendingCommandStreamScanned = true;
    this._explicitFallbackDrawContext = null;
    this.computePipelines.clear();
    this._pipelineBindGroupLayouts.clear();
    this._computePipelineBindGroupLayouts.clear();
    this._pipelineReflectionGroups.clear();
    this._computePipelineReflectionGroups.clear();
    this._defaultFallbackSamplers.clear();
    this._shaderReflectionKeys.clear();
    this._pendingTimerReadbacks.clear();
    this._lastPassTimings = null;

    // Packet ABI v2 extensions own device objects too; the worker handler reinstalls them.
    this.extension = null;
    if (Object.prototype.hasOwnProperty.call(this, '__playgta6RsActiveHandler')) {
      this.__playgta6RsActiveHandler = null;
    }

    // Owned splash/Bink presenters are rebuilt lazily by `_ensureSplashBridge()`.
    if (this.splashFrameBridge && this._ownsSplashFrameBridge) {
      safeDestroy(this.splashFrameBridge);
      this.splashFrameBridge = null;
      this._ownsSplashFrameBridge = false;
    }
    if (this.binkPresenter && this._ownsBinkPresenter) {
      safeDestroy(this.binkPresenter);
      this.binkPresenter = null;
      this._ownsBinkPresenter = false;
    }
  }

  /**
   * W100: rebuild owned device-bound subsystems (staging pool, uniform ring, pass timer, RT pool)
   * on `device`. A subsystem that fails to rebuild is disabled instead of throwing.
   *
   * @param {GPUDevice} device
   */
  _recreateOwnedSubsystems(device) {
    const specs = [
      ['stagingPool', '_ownsStagingPool'],
      ['uniformRing', '_ownsUniformRing'],
      ['passTimer', '_ownsPassTimer'],
      ['rtPool', '_ownsRtPool'],
    ];
    for (let i = 0; i < specs.length; i += 1) {
      const [key, ownsKey] = specs[i];
      const factory = this._ownedSubsystemFactories[key];
      if (!this[ownsKey] || typeof factory !== 'function') {
        continue;
      }
      const prev = this[key];
      if (prev && typeof prev.destroy === 'function') {
        try {
          prev.destroy();
        } catch (_) {
          // Ignore teardown failures of subsystems bound to the lost device.
        }
      }
      try {
        this[key] = factory(device);
      } catch (err) {
        this[key] = null;
        this[ownsKey] = false;
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] ${key} unavailable after device recovery: ${sanitizeGpuErrorDetail(err)}`
        );
      }
    }
    this._uniformRingFrameIndex = -1;
  }

  /**
   * #663 / #709 / W112 / W121: Configure the WebGPU canvas context / swapchain to match the
   * swapchain attachment format declared by the command stream with console-accurate color
   * presentation:
   * - `bgra8unorm-srgb` / `rgba8unorm-srgb`: prefers zero-copy direct sRGB canvas views via
   *   `viewFormats` (`usesBlitFallback = false`); if the browser rejects `viewFormats`, falls back
   *   to an offscreen sRGB intermediate with `applySrgbEncode: true` so linear samples are
   *   re-encoded to sRGB before writing to the non-sRGB SDR canvas (preventing double-darkened
   *   gamma).
   * - `rgba16float` / `rg11b10ufloat` / `rgba32float` (linear HDR / scRGB): guards optional
   *   device features (`rg11b10ufloat-renderable`, `float32-filterable`) and by default renders
   *   to an offscreen linear HDR intermediate + ACES/Reinhard tonemapped linear->sRGB blit onto
   *   the SDR canvas (`hdrMode: 'extended'` with `toneMapMode: 'none'` preserves direct
   *   `rgba16float` canvas output).
   * - `rgb10a2unorm`: uses offscreen intermediate blit with `applySrgbEncode: false` by default
   *   (10-bit SDR) or linear tonemapping when explicitly configured via `rgb10a2Transfer: 'linear'`.
   * - `bgra8unorm` / `rgba8unorm`: direct zero-overhead SDR canvas path.
   *
   * @param {string} declaredFormat
   * @returns {boolean}
   */
  setSwapchainFormat(declaredFormat) {
    if (typeof declaredFormat !== 'string' || declaredFormat.length === 0) {
      return false;
    }
    const normalizedFormat = declaredFormat.trim().toLowerCase();
    if (normalizedFormat.length === 0) {
      return false;
    }
    const swapchain =
      this._swapchain && !this._swapchain.isDestroyed ? this._swapchain : null;
    const srgbBaseFormat = SWAPCHAIN_SRGB_TO_BASE[normalizedFormat] || null;
    const isLinearHdr = SWAPCHAIN_LINEAR_HDR_FORMATS.has(normalizedFormat);

    // Track preferred SDR canvas format so returning from extended rgba16float restores SDR.
    if (SWAPCHAIN_SDR_BASE_FORMATS.has(this.format)) {
      this._preferredSdrCanvasFormat = this.format;
    }

    const applyToneMap = (mode, exposure, applySrgbEncode) => {
      this._activeToneMapMode = mode;
      this._activeApplySrgbEncode = Boolean(applySrgbEncode);
      if (swapchain && typeof swapchain.setToneMapOptions === 'function') {
        try {
          swapchain.setToneMapOptions({
            mode,
            exposure,
            applySrgbEncode,
          });
        } catch (_) {
          // Ignore custom swapchain mocks without full tonemap support.
        }
      }
    };

    const deviceHasFeature = (featureName) => {
      if (!this.device || !this.device.features || typeof this.device.features.has !== 'function') {
        return true;
      }
      return this.device.features.has(featureName);
    };

    const finishFormatUpdate = () => {
      this._syncBinkPresentationFormat(this.format);
      return true;
    };

    // Case 1: sRGB backbuffer (`bgra8unorm-srgb`, `rgba8unorm-srgb`)
    if (srgbBaseFormat) {
      if (this._directSrgbView && swapchain && typeof swapchain.setFormat === 'function') {
        try {
          applyToneMap('none', 1.0, null);
          swapchain.setFormat(srgbBaseFormat, [normalizedFormat]);
          const viewFormatsAccepted =
            swapchain.viewFormats === undefined ||
            (Array.isArray(swapchain.viewFormats) &&
              swapchain.viewFormats.includes(normalizedFormat));
          if (viewFormatsAccepted) {
            this.format =
              typeof swapchain.format === 'string' && swapchain.format.length > 0
                ? swapchain.format
                : srgbBaseFormat;
            this._swapchainViewFormat = normalizedFormat;
            this._streamSwapchainFormat = normalizedFormat;
            this._activeToneMapMode = 'none';
            this._activeApplySrgbEncode = false;
            return finishFormatUpdate();
          }
        } catch (err) {
          this._logDiagnostic(
            'warn',
            `[wgpu_worker] direct sRGB swapchain view ${normalizedFormat} failed, falling back to sRGB-encoded blit: ${sanitizeGpuErrorDetail(err)}`
          );
        }
      }
      // Offscreen sRGB intermediate + explicit linear->sRGB encoding on blit into the non-sRGB SDR
      // canvas: sampling an `-srgb` intermediate hardware-decodes sRGB->linear, so `applySrgbEncode: true`
      // is required to re-encode linear->sRGB before writing to `bgra8unorm`/`rgba8unorm` (fixes
      // crushed/double-darkened midtones).
      if (swapchain && typeof swapchain.configureForStreamFormat === 'function') {
        try {
          applyToneMap('none', 1.0, true);
          swapchain.configureForStreamFormat(normalizedFormat, { forceBlit: true });
          this._streamSwapchainFormat = normalizedFormat;
          this._swapchainViewFormat = null;
          if (typeof swapchain.format === 'string' && swapchain.format.length > 0) {
            this.format = swapchain.format;
          }
          return finishFormatUpdate();
        } catch (err) {
          this._logDiagnostic(
            'warn',
            `[wgpu_worker] failed to configure swapchain for stream format ${normalizedFormat}: ${sanitizeGpuErrorDetail(err)}`
          );
          return false;
        }
      }
    }

    // Case 2: Linear HDR backbuffer (`rgba16float`, `rg11b10ufloat`, `rgba32float`)
    if (isLinearHdr && swapchain && typeof swapchain.configureForStreamFormat === 'function') {
      let effectiveFormat = normalizedFormat;
      if (effectiveFormat === 'rg11b10ufloat' && !deviceHasFeature('rg11b10ufloat-renderable')) {
        effectiveFormat = 'rgba16float';
      } else if (effectiveFormat === 'rgba32float' && !deviceHasFeature('float32-filterable')) {
        effectiveFormat = 'rgba16float';
      }
      const wantsSdrToneMapBlit =
        effectiveFormat !== 'rgba16float' ||
        this._swapchainHdrMode === 'sdr-tonemap' ||
        this._swapchainToneMapMode === 'aces' ||
        this._swapchainToneMapMode === 'reinhard' ||
        this._swapchainApplySrgbEncode === true;
      try {
        if (!wantsSdrToneMapBlit) {
          applyToneMap('none', this._swapchainExposure, false);
          swapchain.configureForStreamFormat(effectiveFormat);
        } else {
          // Keep the canvas in SDR (`bgra8unorm` / `rgba8unorm`) so highlights tonemap cleanly
          // via ACES/Reinhard + linear->sRGB encoding and the swapchain luminance probe stays active.
          if (
            !SWAPCHAIN_SDR_BASE_FORMATS.has(swapchain.format) &&
            typeof swapchain.setFormat === 'function'
          ) {
            const sdrBase = SWAPCHAIN_SDR_BASE_FORMATS.has(this._preferredSdrCanvasFormat)
              ? this._preferredSdrCanvasFormat
              : 'bgra8unorm';
            swapchain.setFormat(sdrBase);
          }
          const effectiveToneMap = this._swapchainToneMapMode || 'aces';
          const effectiveSrgbEncode =
            this._swapchainApplySrgbEncode !== null ? this._swapchainApplySrgbEncode : true;
          applyToneMap(effectiveToneMap, this._swapchainExposure, effectiveSrgbEncode);
          swapchain.configureForStreamFormat(effectiveFormat, { forceBlit: true });
        }
      } catch (err) {
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] failed to configure swapchain for HDR stream format ${normalizedFormat}: ${sanitizeGpuErrorDetail(err)}`
        );
        return false;
      }
      this._streamSwapchainFormat = effectiveFormat;
      this._swapchainViewFormat = null;
      if (typeof swapchain.format === 'string' && swapchain.format.length > 0) {
        this.format = swapchain.format;
      }
      return finishFormatUpdate();
    }

    // Case 3: `rgb10a2unorm` (default 10-bit SDR passthrough unless linear HDR transfer requested)
    if (
      normalizedFormat === 'rgb10a2unorm' &&
      swapchain &&
      typeof swapchain.configureForStreamFormat === 'function'
    ) {
      const wantsLinearToneMap =
        this._swapchainRgb10a2Transfer === 'linear' ||
        this._swapchainToneMapMode === 'aces' ||
        this._swapchainToneMapMode === 'reinhard' ||
        this._swapchainApplySrgbEncode === true;
      try {
        if (wantsLinearToneMap) {
          const effectiveToneMap = this._swapchainToneMapMode || 'aces';
          const effectiveSrgbEncode =
            this._swapchainApplySrgbEncode !== null ? this._swapchainApplySrgbEncode : true;
          applyToneMap(effectiveToneMap, this._swapchainExposure, effectiveSrgbEncode);
        } else {
          applyToneMap('none', 1.0, false);
        }
        swapchain.configureForStreamFormat(normalizedFormat);
      } catch (err) {
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] failed to configure swapchain for stream format ${normalizedFormat}: ${sanitizeGpuErrorDetail(err)}`
        );
        return false;
      }
      this._streamSwapchainFormat = normalizedFormat;
      this._swapchainViewFormat = null;
      if (typeof swapchain.format === 'string' && swapchain.format.length > 0) {
        this.format = swapchain.format;
      }
      return finishFormatUpdate();
    }

    // Case 4: Other swapchain-managed formats (`bgra8unorm`, `rgba8unorm`, `rgba16unorm`, etc.)
    if (swapchain && typeof swapchain.configureForStreamFormat === 'function') {
      try {
        applyToneMap('none', 1.0, null);
        swapchain.configureForStreamFormat(normalizedFormat);
      } catch (err) {
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] failed to configure swapchain for stream format ${normalizedFormat}: ${sanitizeGpuErrorDetail(err)}`
        );
        return false;
      }
      this._streamSwapchainFormat = normalizedFormat;
      this._swapchainViewFormat = null;
      if (typeof swapchain.format === 'string' && swapchain.format.length > 0) {
        this.format = swapchain.format;
      }
      return finishFormatUpdate();
    }

    // Raw canvas context fallback (no WgpuSwapchain attached)
    const isSrgb = normalizedFormat.endsWith('-srgb');
    const baseFormat = isSrgb ? normalizedFormat.slice(0, -5) : normalizedFormat;
    if (
      baseFormat !== 'rgba8unorm' &&
      baseFormat !== 'bgra8unorm' &&
      baseFormat !== 'rgba16float'
    ) {
      return false;
    }
    const viewFormat = isSrgb ? normalizedFormat : null;
    this._streamSwapchainFormat = normalizedFormat;
    if (this.format === baseFormat && (this._swapchainViewFormat || null) === viewFormat) {
      return finishFormatUpdate();
    }
    this.format = baseFormat;
    this._swapchainViewFormat = viewFormat;
    const viewFormats = viewFormat ? [viewFormat] : null;
    if (swapchain && typeof swapchain.setFormat === 'function') {
      try {
        swapchain.setFormat(baseFormat, viewFormats);
      } catch (err) {
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] failed to reconfigure swapchain format to ${normalizedFormat}: ${sanitizeGpuErrorDetail(err)}`
        );
        return false;
      }
      return finishFormatUpdate();
    }
    if (this.ctx && typeof this.ctx.configure === 'function' && this.device) {
      try {
        const cfg = {
          device: this.device,
          format: baseFormat,
          alphaMode: this._alphaMode || 'opaque',
        };
        if (viewFormats) {
          cfg.viewFormats = viewFormats;
        }
        configureCanvasContextWithViewFormats(this.ctx, cfg);
      } catch (err) {
        this._logDiagnostic(
          'warn',
          `[wgpu_worker] failed to reconfigure canvas context format to ${normalizedFormat}: ${sanitizeGpuErrorDetail(err)}`
        );
        return false;
      }
      return finishFormatUpdate();
    }
    return finishFormatUpdate();
  }

  /**
   * Configure swapchain HDR/SDR tonemapping and transfer options at runtime.
   * Re-applies the active stream swapchain format immediately when one has been configured.
   *
   * @param {{
   *   mode?: 'auto' | 'none' | 'passthrough' | 'reinhard' | 'aces' | null,
   *   toneMapMode?: 'auto' | 'none' | 'passthrough' | 'reinhard' | 'aces' | null,
   *   exposure?: number,
   *   applySrgbEncode?: boolean | null,
   *   hdrMode?: 'sdr-tonemap' | 'extended' | 'auto',
   *   rgb10a2Transfer?: 'sdr' | 'linear',
   * }} [opts]
   */
  setSwapchainToneMapOptions(opts = {}) {
    const o = opts && typeof opts === 'object' ? opts : {};
    if (o.mode !== undefined || o.toneMapMode !== undefined) {
      this._swapchainToneMapMode = normalizeSwapchainToneMapMode(
        o.mode !== undefined ? o.mode : o.toneMapMode
      );
    }
    if (o.exposure !== undefined) {
      this._swapchainExposure = normalizeSwapchainExposure(o.exposure, this._swapchainExposure);
    }
    if (o.applySrgbEncode !== undefined) {
      this._swapchainApplySrgbEncode =
        typeof o.applySrgbEncode === 'boolean' ? o.applySrgbEncode : null;
    }
    if (o.hdrMode !== undefined) {
      this._swapchainHdrMode = normalizeSwapchainHdrMode(o.hdrMode);
    }
    if (o.rgb10a2Transfer !== undefined) {
      this._swapchainRgb10a2Transfer = o.rgb10a2Transfer === 'linear' ? 'linear' : 'sdr';
    }
    if (this._streamSwapchainFormat) {
      this.setSwapchainFormat(this._streamSwapchainFormat);
    } else if (
      this._swapchain &&
      !this._swapchain.isDestroyed &&
      typeof this._swapchain.setToneMapOptions === 'function'
    ) {
      const mode = this._swapchainToneMapMode || 'none';
      this._activeToneMapMode = mode;
      this._activeApplySrgbEncode = Boolean(this._swapchainApplySrgbEncode);
      try {
        this._swapchain.setToneMapOptions({
          mode,
          exposure: this._swapchainExposure,
          applySrgbEncode: this._swapchainApplySrgbEncode,
        });
      } catch (_) {
        // Ignore mock swapchains without tonemap options.
      }
    }
    return {
      toneMapMode: this._activeToneMapMode,
      configuredToneMapMode: this._swapchainToneMapMode,
      exposure: this._swapchainExposure,
      applySrgbEncode: this._activeApplySrgbEncode,
      hdrMode: this._swapchainHdrMode,
      rgb10a2Transfer: this._swapchainRgb10a2Transfer,
      streamFormat: this._streamSwapchainFormat,
      canvasFormat: this.format,
      viewFormat: this._swapchainViewFormat,
      usesBlitFallback: Boolean(this._swapchain && this._swapchain.usesBlitFallback === true),
    };
  }

  /**
   * W121: Configure the swapchain for a stream-declared back-buffer format
   * (alias for `setSwapchainFormat(declaredFormat)`).
   *
   * @param {string} declaredFormat
   * @returns {boolean}
   */
  configureForStreamFormat(declaredFormat) {
    return this.setSwapchainFormat(declaredFormat);
  }

  /**
   * W100: bind a recovered GPUDevice (and re-acquired canvas context) after device loss. Clears
   * every resource handle owned by the dead device, rebuilds owned subsystems, re-configures the
   * swapchain (via `recover()`) or the raw canvas context, and re-hooks `device.lost`. Telemetry
   * counters, frameCount, the ring and shader pack sources are preserved.
   *
   * @param {GPUDevice} newDevice
   * @param {GPUCanvasContext | null | undefined} newCtx
   * @param {{ format?: string, alphaMode?: string, recoverSwapchain?: boolean }} [opts]
   */
  rebindDevice(newDevice, newCtx, opts = {}) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!newDevice || typeof newDevice !== 'object') {
      throw new TypeError('rebindDevice requires a GPUDevice');
    }
    const options = opts && typeof opts === 'object' ? opts : {};
    this._deviceGeneration += 1;
    this._resetDeviceResources();
    // W167: update pipelineCache device so cached fallback samplers/layouts are invalidated cleanly.
    if (this.pipelineCache && typeof this.pipelineCache.setDevice === 'function') {
      try {
        this.pipelineCache.setDevice(newDevice);
      } catch (_) {
        // Ignore pipelineCache device swap errors on custom mocks.
      }
    }
    // W103: drop in-flight compiles of the dead device; late results are discarded as stale.
    this._resetAsyncPipelineCache(newDevice);
    if (newCtx && typeof newCtx === 'object') {
      this.ctx = newCtx;
    }
    const swapchainRecovers = Boolean(
      this._swapchain &&
        !this._swapchain.isDestroyed &&
        typeof this._swapchain.recover === 'function' &&
        options.recoverSwapchain !== false
    );
    const savedStreamFormat = this._streamSwapchainFormat;
    const savedViewFormat = this._swapchainViewFormat;
    const effectiveFormat = savedStreamFormat
      ? swapchainRecovers
        ? savedViewFormat
          ? this.format
          : savedStreamFormat
        : this.format
      : options.format;
    this.bindDevice(newDevice, {
      format: effectiveFormat,
      recoverSwapchain: options.recoverSwapchain,
    });
    if (
      swapchainRecovers &&
      this._swapchain &&
      typeof this._swapchain.format === 'string' &&
      this._swapchain.format.length > 0
    ) {
      this.format = this._swapchain.format;
    }
    if (swapchainRecovers && savedStreamFormat) {
      this.setSwapchainFormat(savedStreamFormat);
    }
    if (!swapchainRecovers && this.ctx && typeof this.ctx.configure === 'function') {
      const cfg = {
        device: newDevice,
        format: this.format,
        alphaMode:
          typeof options.alphaMode === 'string'
            ? normalizeSwapchainAlphaMode(options.alphaMode)
            : this._alphaMode || 'opaque',
      };
      if (this._swapchainViewFormat) {
        cfg.viewFormats = [this._swapchainViewFormat];
      }
      configureCanvasContextWithCopySrc(this.ctx, cfg);
    }
    this._recreateOwnedSubsystems(newDevice);
    this._hookDeviceLost(newDevice);
    this._deviceLost = false;
    return true;
  }

  /**
   * Begin a frame epoch across enabled subsystems (staging pool, uniform ring,
   * pass timer, and deferred resource GC) and reset transient pass state.
   *
   * @param {number} [frameIndex]
   */
  beginFrame(frameIndex) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    // Submit any out-of-frame staged uploads (e.g. init buffer writes) before rotating epochs.
    if (this._pendingEncoder && this._frameBegunForIndex === -1) {
      this._flushPendingUploads(true);
    }
    this._resetFrameState();
    const targetFrame = frameIndex !== undefined ? frameIndex : this.frameCount;
    this._beginSubsystemsForFrame(targetFrame);
  }

  /**
   * End the current frame epoch cleanly: flushes any uncommitted upload/uniform data,
   * retires staging chunks and deferred resources, and advances the RT pool epoch.
   */
  endFrame() {
    if (this._destroyed) {
      return;
    }
    if (
      this.uniformRing &&
      !this.uniformRing._destroyed &&
      typeof this.uniformRing.flush === 'function'
    ) {
      this.uniformRing.flush();
    }
    this._flushPendingUploads(false);
    const workDoneFence = this._getWorkDoneFence();
    this._drainDeferredGpuDestroys(workDoneFence, { immediate: workDoneFence === null });
    this._endSubsystemsForFrame(workDoneFence);
  }

  /**
   * Append a per-draw constant payload into the active DynamicUniformRing slot.
   * Flushed automatically in a single queue.writeBuffer before command submission.
   *
   * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
   * @returns {{ buffer: GPUBuffer, dynamicOffset: number, size: number, alignedSize: number }}
   */
  pushDrawConstants(data) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.uniformRing || this.uniformRing._destroyed) {
      throw new Error(
        'DynamicUniformRing is not enabled; pass { uniformRing } or { enableUniformRing: true }'
      );
    }
    this._ensureFrameBegun();
    return this.uniformRing.pushConstants(data);
  }

  /**
   * Return the most recently resolved GPU pass timings from GpuPassTimer, or null
   * when timestamp queries are unsupported or no readback has completed yet.
   *
   * @returns {{ totalGpuMs: number, passes: Array<{ name: string, durationMs: number }> } | null}
   */
  getLastPassTimings() {
    if (!this._lastPassTimings) {
      return null;
    }
    return {
      totalGpuMs: this._lastPassTimings.totalGpuMs,
      passes: this._lastPassTimings.passes.map((p) => ({ name: p.name, durationMs: p.durationMs })),
    };
  }

  /**
   * Await any in-flight GPU pass timer readbacks (and collect one pending slot if queued),
   * updating and returning the latest pass timings snapshot.
   */
  async collectPassTimings() {
    if (this._destroyed || !this.passTimer || !this.passTimer.supported) {
      return this.getLastPassTimings();
    }
    if (this._pendingTimerReadbacks.size > 0) {
      await Promise.all(Array.from(this._pendingTimerReadbacks));
    }
    if (
      !this._destroyed &&
      typeof this.passTimer.collectCompletedReadback === 'function'
    ) {
      const timing = await this.passTimer.collectCompletedReadback();
      if (timing && !this._destroyed) {
        this._lastPassTimings = timing;
      }
    }
    return this.getLastPassTimings();
  }

  /**
   * Load and index an S6 shader reflection manifest (JSON string or parsed object).
   * Lazily initializes ReflectionRegistry if one was not injected in the constructor.
   *
   * @param {string | object} jsonOrObj
   * @returns {object} Parsed reflection manifest
   */
  loadReflectionManifest(jsonOrObj) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.reflectionRegistry) {
      const ReflectionCtor = resolveSubsystemCtor(
        this._ReflectionRegistryCtor,
        'PlayGta6ReflectionLoader',
        'ReflectionRegistry',
        './reflection_loader.js'
      );
      if (!ReflectionCtor) {
        throw new Error(
          'ReflectionRegistry unavailable; load ./reflection_loader.js or inject reflectionRegistry'
        );
      }
      this.reflectionRegistry = new ReflectionCtor(this._reflectionRegistryOptions || {});
      this._ownsReflectionRegistry = true;
    }
    return this.reflectionRegistry.loadManifest(jsonOrObj);
  }

  /**
   * Explicitly associate a command-stream shader ID with a reflection hash or name key.
   *
   * @param {number} shaderId
   * @param {bigint | string | number} hashOrName
   */
  registerShaderReflectionKey(shaderId, hashOrName) {
    this._shaderReflectionKeys.set(shaderId, hashOrName);
  }

  /**
   * Parse and load a binary WGPK (`.wgsl.pack`) buffer into `wgslPackCache`.
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
   * @param {string} [packName]
   */
  loadWgslPack(buffer, packName) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.wgslPackCache) {
      this.wgslPackCache = new InlineWgslPackCache();
      this._ownsWgslPackCache = true;
    }
    const result = loadWgslPackIntoCache(this.wgslPackCache, buffer, packName);
    if (
      this.shaderFallbackRouter &&
      this.shaderFallbackRouter._packLoader == null &&
      typeof this.shaderFallbackRouter.setPackLoader === 'function'
    ) {
      this.shaderFallbackRouter.setPackLoader(this.wgslPackCache);
    }
    return result;
  }

  /**
   * W109 / W121 / W159: Fetch and load a `.wgsl.pack` URL into `wgslPackCache`, automatically
   * enabling HTTP `206 Range` index streaming when pack size exceeds `streamThresholdBytes`
   * (defaults to `DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES` = 128 MiB; pass `0` or
   * `streamShaders: false` to force full-buffer loading).
   *
   * @param {string} url
   * @param {{
   *   fetch?: typeof fetch,
   *   packName?: string,
   *   streamThresholdBytes?: number,
   *   streamShaders?: boolean,
   *   initialRangeBytes?: number,
   *   maxCachedWgslBytes?: number,
   *   fetchInit?: RequestInit,
   *   lazyDecode?: boolean,
   * }} [opts]
   */
  async loadShaderPackUrl(url, opts = {}) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (typeof url !== 'string' || url.trim().length === 0) {
      throw new TypeError('loadShaderPackUrl requires a non-empty URL string');
    }
    const options = opts && typeof opts === 'object' ? opts : {};
    const activeFetch =
      typeof options.fetch === 'function'
        ? options.fetch
        : typeof globalThis !== 'undefined' && typeof globalThis.fetch === 'function'
          ? globalThis.fetch
          : null;
    if (!activeFetch) {
      throw new Error('No fetch implementation available for loadShaderPackUrl');
    }
    const resolvedUrl = url.trim();
    const packName =
      typeof options.packName === 'string' && options.packName.length > 0
        ? options.packName
        : resolvedUrl;
    const streamThresholdBytes = resolveShaderPackStreamThresholdBytes(options, {
      streamThresholdBytes: this._shaderPackStreamThresholdBytes,
    });
    this._activeShaderPackStreamThresholdBytes = streamThresholdBytes;

    const buildPackLoadOpts = (includeThreshold) => {
      const loadOpts = {
        fetch: activeFetch,
        packName,
      };
      if (includeThreshold) {
        loadOpts.streamThresholdBytes = streamThresholdBytes;
      }
      if (
        Number.isInteger(options.initialRangeBytes) &&
        options.initialRangeBytes >= WGPK_HEADER_BYTES
      ) {
        loadOpts.initialRangeBytes = options.initialRangeBytes;
      }
      if (Number.isInteger(options.maxCachedWgslBytes) && options.maxCachedWgslBytes >= 0) {
        loadOpts.maxCachedWgslBytes = options.maxCachedWgslBytes;
      }
      if (options.fetchInit !== undefined) {
        loadOpts.fetchInit = options.fetchInit;
      }
      if (options.lazyDecode !== undefined) {
        loadOpts.lazyDecode = Boolean(options.lazyDecode);
      }
      return loadOpts;
    };

    const syncRouterPackLoader = (prevCache) => {
      if (
        this.shaderFallbackRouter &&
        typeof this.shaderFallbackRouter.setPackLoader === 'function' &&
        (this.shaderFallbackRouter._packLoader == null ||
          (prevCache && this.shaderFallbackRouter._packLoader === prevCache))
      ) {
        this.shaderFallbackRouter.setPackLoader(this.wgslPackCache);
      }
    };

    if (this.wgslPackCache && typeof this.wgslPackCache.loadPackUrl === 'function') {
      const result = await this.wgslPackCache.loadPackUrl(resolvedUrl, buildPackLoadOpts(true));
      syncRouterPackLoader(null);
      return result;
    }

    if (streamThresholdBytes > 0 && this._ownsWgslPackCache) {
      let confirmedOverThreshold = false;
      let confirmedContentLength = null;
      let probeNeedsRangedLoader = false;
      let preloadedBuffer = null;

      const headResp = await activeFetch(resolvedUrl, buildHeadFetchInit(options.fetchInit));
      if (!headResp) {
        throw new Error(`Failed to fetch WGSL pack from "${resolvedUrl}": HTTP unknown`);
      }
      if (headResp.ok) {
        const acceptRanges = readFetchResponseHeader(headResp, 'accept-ranges');
        const contentLength = parseNonNegativeHeaderInt(
          readFetchResponseHeader(headResp, 'content-length')
        );
        confirmedContentLength = contentLength;
        const rangeDisabled = Boolean(acceptRanges && acceptRanges.trim().toLowerCase() === 'none');
        const maxFullBufferBytes =
          typeof options.maxFullFallbackBytes === 'number' &&
          Number.isInteger(options.maxFullFallbackBytes) &&
          options.maxFullFallbackBytes >= 0
            ? options.maxFullFallbackBytes
            : 256 * 1024 * 1024;
        if (contentLength !== null && contentLength > streamThresholdBytes) {
          if (rangeDisabled) {
            if (maxFullBufferBytes > 0 && contentLength > maxFullBufferBytes) {
              throw new Error(
                `WGSL pack "${resolvedUrl}" is ${contentLength} bytes (> ${maxFullBufferBytes} byte full-buffer cap) and server returned Accept-Ranges: none; refusing full-buffer download`
              );
            }
            probeNeedsRangedLoader = false;
          } else {
            confirmedOverThreshold = true;
            probeNeedsRangedLoader = true;
          }
        } else if (rangeDisabled) {
          probeNeedsRangedLoader = false;
        } else if (contentLength !== null) {
          probeNeedsRangedLoader = false;
        } else if (typeof headResp.arrayBuffer === 'function') {
          // Simple non-HEAD-aware mock or endpoint already returned the pack body directly.
          const maybeBuf = await headResp.arrayBuffer();
          if (maybeBuf && maybeBuf.byteLength > 0) {
            preloadedBuffer = maybeBuf;
          } else {
            probeNeedsRangedLoader = true;
          }
        } else {
          probeNeedsRangedLoader = true;
        }
      } else if (headResp.status === 405 || headResp.status === 501) {
        probeNeedsRangedLoader = true;
      } else {
        throw new Error(`Failed to fetch WGSL pack from "${resolvedUrl}": HTTP ${headResp.status}`);
      }

      if (preloadedBuffer !== null) {
        return this.loadWgslPack(preloadedBuffer, packName);
      }

      if (probeNeedsRangedLoader) {
        let rangedImportError = null;
        let nextCache = null;
        let prevCache = null;
        try {
          const mod = await import('./wgsl_pack_loader.js');
          if (mod && typeof mod.WgslPackCache === 'function') {
            prevCache = this.wgslPackCache;
            nextCache = new mod.WgslPackCache({
              fetch: activeFetch,
              streamThresholdBytes,
            });
          }
        } catch (importErr) {
          rangedImportError = importErr;
        }
        if (nextCache) {
          this.wgslPackCache = nextCache;
          syncRouterPackLoader(prevCache);
          if (confirmedOverThreshold && typeof nextCache.loadIndexFromUrl === 'function') {
            return await nextCache.loadIndexFromUrl(resolvedUrl, buildPackLoadOpts(false));
          }
          return await nextCache.loadPackUrl(resolvedUrl, buildPackLoadOpts(true));
        }
        if (confirmedOverThreshold) {
          throw new Error(
            `WGSL pack "${resolvedUrl}" is ${confirmedContentLength} bytes (> ${streamThresholdBytes} byte streaming threshold) and ranged wgsl_pack_loader.js is unavailable${rangedImportError ? `: ${sanitizeGpuErrorDetail(rangedImportError)}` : ''}`
          );
        }
      }
    }

    const resp = await activeFetch(resolvedUrl, options.fetchInit);
    if (!resp || !resp.ok || typeof resp.arrayBuffer !== 'function') {
      const status = resp ? resp.status : 'unknown';
      throw new Error(`Failed to fetch WGSL pack from "${resolvedUrl}": HTTP ${status}`);
    }
    const buffer = await resp.arrayBuffer();
    return this.loadWgslPack(buffer, packName);
  }

  /**
   * Alias for `loadShaderPackUrl(url, opts)`.
   */
  loadShaderPack(url, opts = {}) {
    return this.loadShaderPackUrl(url, opts);
  }

  /**
   * Return top missing shaders recorded by `ShaderFallbackRouter` (sorted by hit count desc).
   *
   * @param {{ limit?: number }} [opts]
   */
  getMissingShaders(opts) {
    if (
      this.shaderFallbackRouter &&
      typeof this.shaderFallbackRouter.getMissingReport === 'function'
    ) {
      return this.shaderFallbackRouter.getMissingReport(opts);
    }
    return [];
  }

  /**
   * Return aggregate lookup / fallback telemetry from `ShaderFallbackRouter`, or null if disabled.
   */
  getShaderFallbackStats() {
    if (
      this.shaderFallbackRouter &&
      typeof this.shaderFallbackRouter.getStats === 'function'
    ) {
      return this.shaderFallbackRouter.getStats();
    }
    return null;
  }

  /**
   * W179: Return a bounded, sanitized categorized missing-shader summary from
   * `ShaderFallbackRouter.getMissingShaderSummary()`, or null when unavailable.
   *
   * @param {number | { limit?: number | null, reason?: string | null, stage?: number | string | null } | null} [opts]
   */
  getMissingShaderSummary(opts) {
    if (
      !this.shaderFallbackRouter ||
      typeof this.shaderFallbackRouter.getMissingShaderSummary !== 'function'
    ) {
      return null;
    }
    let requestedLimit = DEFAULT_MISSING_SUMMARY_LIMIT;
    if (typeof opts === 'number' && Number.isInteger(opts) && opts >= 0) {
      requestedLimit = opts;
    } else if (
      opts &&
      typeof opts === 'object' &&
      Number.isInteger(opts.limit) &&
      opts.limit >= 0
    ) {
      requestedLimit = opts.limit;
    }
    try {
      const raw =
        opts !== undefined
          ? this.shaderFallbackRouter.getMissingShaderSummary(opts)
          : this.shaderFallbackRouter.getMissingShaderSummary({ limit: requestedLimit });
      return sanitizeMissingShaderSummary(raw, requestedLimit);
    } catch (_) {
      return null;
    }
  }

  /**
   * W167: Return a cached default fallback `GPUSampler` (`'filtering'`, `'non-filtering'`, or
   * `'comparison'`), delegating to `pipelineCache.getOrCreateDefaultSampler()` when available.
   *
   * @param {'filtering' | 'non-filtering' | 'comparison' | string | null | undefined} [kind='filtering']
   * @returns {GPUSampler | null}
   */
  getOrCreateDefaultSampler(kind = 'filtering') {
    if (this._destroyed || !this.device) {
      return null;
    }
    if (
      this.pipelineCache &&
      typeof this.pipelineCache.getOrCreateDefaultSampler === 'function'
    ) {
      try {
        return this.pipelineCache.getOrCreateDefaultSampler(kind);
      } catch (_) {
        return null;
      }
    }
    if (typeof this.device.createSampler !== 'function') {
      return null;
    }
    const canonical = normalizeFallbackSamplerKind(kind);
    const cached = this._defaultFallbackSamplers.get(canonical);
    if (cached !== undefined) {
      return cached;
    }
    try {
      const desc = { ...DEFAULT_FALLBACK_SAMPLER_DESCRIPTORS[canonical] };
      const sampler = this.device.createSampler(desc);
      if (!sampler || (typeof sampler !== 'object' && typeof sampler !== 'function')) {
        return null;
      }
      this._defaultFallbackSamplers.set(canonical, sampler);
      this._fallbackSamplersCreated += 1;
      return sampler;
    } catch (_) {
      return null;
    }
  }

  /**
   * W167: Synthesize default fallback `GPUSampler` entries for any sampler slot declared in
   * `reflectionGroupOrEntries` that is not already bound in `existingEntries`. Delegates to
   * `pipelineCache.synthesizeFallbackSamplerEntries()` when present.
   *
   * @param {Array<object> | { entries?: Array<object>, bindings?: Array<object> }} reflectionGroupOrEntries
   * @param {Iterable<{ binding: number, resource: unknown }>} [existingEntries=[]]
   */
  synthesizeFallbackSamplerEntries(reflectionGroupOrEntries, existingEntries = []) {
    if (
      this.pipelineCache &&
      typeof this.pipelineCache.synthesizeFallbackSamplerEntries === 'function'
    ) {
      try {
        return this.pipelineCache.synthesizeFallbackSamplerEntries(
          reflectionGroupOrEntries,
          existingEntries
        );
      } catch (_) {
        // Fall through to local safe fallback synthesis.
      }
    }

    let layoutEntries = null;
    if (Array.isArray(reflectionGroupOrEntries)) {
      layoutEntries = reflectionGroupOrEntries;
    } else if (reflectionGroupOrEntries && typeof reflectionGroupOrEntries === 'object') {
      if (Array.isArray(reflectionGroupOrEntries.entries)) {
        layoutEntries = reflectionGroupOrEntries.entries;
      } else if (Array.isArray(reflectionGroupOrEntries.bindings)) {
        layoutEntries = reflectionGroupOrEntries.bindings;
      }
    }
    const byBinding = new Map();
    const boundSet = new Set();
    if (existingEntries !== null && existingEntries !== undefined) {
      for (const item of existingEntries) {
        if (!item || typeof item !== 'object') continue;
        if (!Number.isInteger(item.binding) || item.binding < 0) continue;
        if (item.resource !== undefined && item.resource !== null) {
          byBinding.set(item.binding, { ...item });
          boundSet.add(item.binding);
        }
      }
    }
    const synthesized = [];
    if (layoutEntries) {
      for (let i = 0; i < layoutEntries.length; i += 1) {
        const entry = layoutEntries[i];
        if (
          !entry ||
          typeof entry !== 'object' ||
          !entry.sampler ||
          typeof entry.sampler !== 'object' ||
          !Number.isInteger(entry.binding) ||
          entry.binding < 0 ||
          boundSet.has(entry.binding)
        ) {
          continue;
        }
        const samplerKind = normalizeFallbackSamplerKind(entry.sampler.type);
        const sampler = this.getOrCreateDefaultSampler(samplerKind);
        if (!sampler) {
          continue;
        }
        byBinding.set(entry.binding, { binding: entry.binding, resource: sampler });
        boundSet.add(entry.binding);
        synthesized.push({
          binding: entry.binding,
          kind: samplerKind,
          resource: sampler,
        });
        this._fallbackSamplerBindingsSynthesized += 1;
      }
    }
    const entries = Array.from(byBinding.values()).sort((a, b) => a.binding - b.binding);
    return {
      entries,
      synthesized,
      synthesizedCount: synthesized.length,
    };
  }

  _getBindGroupLayoutEntriesForSynthesis(pipelineId, groupIndex, pipeMeta, renderStateExt) {
    const reflectedRender = this._pipelineReflectionGroups.get(pipelineId);
    if (reflectedRender && reflectedRender[groupIndex]) {
      return reflectedRender[groupIndex];
    }
    const reflectedCompute = this._computePipelineReflectionGroups.get(pipelineId);
    if (reflectedCompute && reflectedCompute[groupIndex]) {
      return reflectedCompute[groupIndex];
    }
    if (renderStateExt && renderStateExt.bindGroupLayouts) {
      const bglRec = renderStateExt.bindGroupLayouts.get(pipelineId);
      if (bglRec && Array.isArray(bglRec.entries) && bglRec.entries.length > 0) {
        return bglRec.entries;
      }
    }
    if (
      pipeMeta &&
      pipeMeta.fallbackProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B &&
      groupIndex === 0
    ) {
      return [{ binding: 1, sampler: { type: 'filtering' } }];
    }
    if (
      pipeMeta &&
      pipeMeta.fallbackProfile === FALLBACK_PROFILE.SPLASH_RECT_24B &&
      groupIndex === 0
    ) {
      return [{ binding: 2, sampler: { type: 'filtering' } }];
    }
    return null;
  }

  /**
   * Resolve the GPUBindGroupLayout for a pipeline handle and group index, preferring
   * S6-reflected layouts built during pipeline creation and falling back to
   * `pipeline.getBindGroupLayout(groupIndex)`.
   *
   * @param {number} pipelineId
   * @param {number} groupIndex
   * @returns {GPUBindGroupLayout | null}
   */
  getBindGroupLayout(pipelineId, groupIndex) {
    const reflectedRender = this._pipelineBindGroupLayouts.get(pipelineId);
    if (reflectedRender && reflectedRender[groupIndex]) {
      return reflectedRender[groupIndex];
    }
    const reflectedCompute = this._computePipelineBindGroupLayouts.get(pipelineId);
    if (reflectedCompute && reflectedCompute[groupIndex]) {
      return reflectedCompute[groupIndex];
    }
    const pipeline = this.pipelines.get(pipelineId) || this.computePipelines.get(pipelineId);
    if (pipeline && typeof pipeline.getBindGroupLayout === 'function') {
      return pipeline.getBindGroupLayout(groupIndex);
    }
    return null;
  }

  /**
   * Acquire a transient render-target texture + view from RenderTargetPool.
   *
   * @param {object} desc
   * @param {string} [label]
   */
  acquireRenderTarget(desc, label) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.rtPool) {
      if (this._rtPoolExplicitlyDisabled) {
        throw new Error('RenderTargetPool is disabled on this WgpuCommandProcessor');
      }
      const RtPoolCtor = resolveSubsystemCtor(
        this._RenderTargetPoolCtor,
        'PlayGta6RenderTargetPool',
        'RenderTargetPool',
        './rt_pool.js'
      );
      if (!RtPoolCtor || !this.device || typeof this.device.createTexture !== 'function') {
        throw new Error(
          'RenderTargetPool unavailable; load ./rt_pool.js or inject rtPool'
        );
      }
      this.rtPool = new RtPoolCtor({
        ...(this._rtPoolOptions || {}),
        device: this.device,
      });
      this._ownsRtPool = true;
    }
    return this.rtPool.acquire(desc, label);
  }

  /**
   * Release a previously acquired transient render-target back into RenderTargetPool
   * for same-frame aliasing or cross-frame reuse.
   *
   * @param {object | number} entry
   * @returns {boolean}
   */
  releaseRenderTarget(entry) {
    if (this._destroyed || !this.rtPool || this.rtPool.isDestroyed) {
      return false;
    }
    return this.rtPool.release(entry);
  }

  /**
   * Notify the processor of a swapchain/viewport resize so idle transient render targets
   * in RenderTargetPool are invalidated cleanly.
   *
   * @param {number} [width]
   * @param {number} [height]
   * @returns {number} Number of invalidated idle render targets
   */
  onResize(width, height) {
    if (typeof width === 'number' && Number.isFinite(width) && width > 0) {
      this._viewportWidth = Math.floor(width);
    }
    if (typeof height === 'number' && Number.isFinite(height) && height > 0) {
      this._viewportHeight = Math.floor(height);
    }
    if (
      !this._destroyed &&
      this.rtPool &&
      !this.rtPool.isDestroyed &&
      typeof this.rtPool.invalidateAll === 'function'
    ) {
      return this.rtPool.invalidateAll();
    }
    return 0;
  }

  handleResize(width, height) {
    return this.onResize(width, height);
  }

  /**
   * W285: Aggressively reclaim idle oversized render targets (e.g. 4K Bink2 targets before
   * Main Menu / In-Game) when `rtPool.reclaimOversizedIdle` is supported.
   *
   * @param {number} [maxIdleFrames=2]
   * @returns {{ reclaimedCount: number, reclaimedBytes: number }}
   */
  reclaimOversizedRenderTargets(maxIdleFrames = this._oversizedRtReclaimIdleFrames) {
    if (
      this._destroyed ||
      !this.rtPool ||
      this.rtPool.isDestroyed ||
      typeof this.rtPool.reclaimOversizedIdle !== 'function'
    ) {
      return { reclaimedCount: 0, reclaimedBytes: 0 };
    }
    try {
      const result = this.rtPool.reclaimOversizedIdle(maxIdleFrames);
      const reclaimedCount = toNonNegativeInt(result && result.reclaimedCount, 0);
      const reclaimedBytes = toNonNegativeInt(result && result.reclaimedBytes, 0);
      if (reclaimedBytes > 0) {
        this._reclaimedOversizedBytes += reclaimedBytes;
      }
      return { reclaimedCount, reclaimedBytes };
    } catch (_) {
      return { reclaimedCount: 0, reclaimedBytes: 0 };
    }
  }

  /**
   * W332 / W319 / W285: Soft memory pressure & scene-transition transient memory trim.
   * Safely trims oversized idle staging slabs via `stagingPool.trimForSceneTransition?.()`
   * and reclaims unused / oversized idle render targets via `rtPool.reclaimUnused?.()` or
   * `rtPool.reclaimOversizedIdle?.()`, returning combined and per-subsystem reclaimed stats.
   *
   * @param {number | { maxIdleFrames?: number, stagingOptions?: object }} [options]
   * @returns {{
   *   reclaimedCount: number,
   *   reclaimedBytes: number,
   *   stagingReclaimedSlabs: number,
   *   stagingReclaimedBytes: number,
   *   rtReclaimedCount: number,
   *   rtReclaimedBytes: number,
   * }}
   */
  trimTransientMemory(options = {}) {
    if (this._destroyed) {
      return {
        reclaimedCount: 0,
        reclaimedBytes: 0,
        stagingReclaimedSlabs: 0,
        stagingReclaimedBytes: 0,
        rtReclaimedCount: 0,
        rtReclaimedBytes: 0,
      };
    }

    const opts =
      options && typeof options === 'object'
        ? options
        : typeof options === 'number' && Number.isFinite(options) && options >= 0
          ? { maxIdleFrames: Math.floor(options) }
          : {};
    const maxIdleFrames =
      opts.maxIdleFrames !== undefined &&
      typeof opts.maxIdleFrames === 'number' &&
      Number.isFinite(opts.maxIdleFrames) &&
      opts.maxIdleFrames >= 0
        ? Math.floor(opts.maxIdleFrames)
        : this._oversizedRtReclaimIdleFrames;

    let stagingReclaimedSlabs = 0;
    let stagingReclaimedBytes = 0;
    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.trimForSceneTransition === 'function'
    ) {
      try {
        const trimRes =
          opts.stagingOptions && typeof opts.stagingOptions === 'object'
            ? this.stagingPool.trimForSceneTransition(opts.stagingOptions)
            : this.stagingPool.trimForSceneTransition();
        stagingReclaimedSlabs = toNonNegativeInt(
          trimRes &&
            (trimRes.reclaimedSlabs !== undefined ? trimRes.reclaimedSlabs : trimRes.reclaimedCount),
          0
        );
        stagingReclaimedBytes = toNonNegativeInt(trimRes && trimRes.reclaimedBytes, 0);
        if (stagingReclaimedBytes > 0) {
          this._stagingSceneTransitionTrimBytes += stagingReclaimedBytes;
        }
      } catch (_) {
        // Staging pool trim errors must never break scene transitions.
      }
    }

    let rtReclaimedCount = 0;
    let rtReclaimedBytes = 0;
    if (this.rtPool && !this.rtPool.isDestroyed) {
      if (typeof this.rtPool.reclaimUnused === 'function') {
        try {
          const rtRes = this.rtPool.reclaimUnused(maxIdleFrames);
          rtReclaimedCount = toNonNegativeInt(
            rtRes && (rtRes.reclaimedCount !== undefined ? rtRes.reclaimedCount : rtRes.evictedCount),
            0
          );
          rtReclaimedBytes = toNonNegativeInt(
            rtRes && (rtRes.reclaimedBytes !== undefined ? rtRes.reclaimedBytes : rtRes.evictedBytes),
            0
          );
          if (rtReclaimedBytes > 0) {
            this._reclaimedOversizedBytes += rtReclaimedBytes;
          }
        } catch (_) {
          // RT pool reclaimUnused errors must never break scene transitions.
        }
      } else if (typeof this.rtPool.reclaimOversizedIdle === 'function') {
        const rtRes = this.reclaimOversizedRenderTargets(maxIdleFrames);
        rtReclaimedCount = rtRes.reclaimedCount;
        rtReclaimedBytes = rtRes.reclaimedBytes;
      }
    }

    let scratchBytesReleased = 0;
    if (this._directWriteScratch) {
      scratchBytesReleased += this._directWriteScratch.byteLength;
      this._directWriteScratch = null;
    }
    if (this._cachedCmdBuffer && this._cachedCmdBuffer.byteLength > 4 * 1024 * 1024) {
      scratchBytesReleased += this._cachedCmdBuffer.byteLength;
      this._cachedCmdBuffer = null;
      this._cachedCmdView = null;
    }
    if (
      this.resourceTable &&
      !this.resourceTable._destroyed &&
      typeof this.resourceTable.trimWriteScratch === 'function'
    ) {
      try {
        scratchBytesReleased += toNonNegativeInt(this.resourceTable.trimWriteScratch(), 0);
      } catch (_) {
        // Ignore scratch trim errors.
      }
    }

    // #1077: Reclaim evicted low-priority (and optionally stale) pipelines on scene transitions.
    if (this.pipelineCache) {
      try {
        if (typeof this.pipelineCache.evictLowPriority === 'function') {
          this.pipelineCache.evictLowPriority();
        }
        if (
          typeof opts.pipelineMaxAgeMs === 'number' &&
          Number.isFinite(opts.pipelineMaxAgeMs) &&
          opts.pipelineMaxAgeMs >= 0 &&
          typeof this.pipelineCache.evictOlderThan === 'function'
        ) {
          this.pipelineCache.evictOlderThan(opts.pipelineMaxAgeMs);
        }
      } catch (_) {
        // Pipeline cache trimming must never disrupt scene transitions.
      }
    }

    // #1077: Trim replay shadow buffer copies down to 16 MiB during scene transitions.
    const shadowTrimTargetBytes = 16 * 1024 * 1024;
    if (this._shadowBuffers && this._shadowBufferBytes > shadowTrimTargetBytes) {
      while (this._shadowBufferBytes > shadowTrimTargetBytes && this._shadowBuffers.size > 0) {
        const oldestKey = this._shadowBuffers.keys().next().value;
        const oldestBuf = this._shadowBuffers.get(oldestKey);
        this._shadowBuffers.delete(oldestKey);
        if (oldestBuf && typeof oldestBuf.byteLength === 'number') {
          this._shadowBufferBytes = Math.max(0, this._shadowBufferBytes - oldestBuf.byteLength);
        }
      }
    }

    let deferredDestroysFlushed = 0;
    if (!this._isFrameOrBatchActive()) {
      deferredDestroysFlushed += this._drainDeferredGpuDestroys(null, DRAIN_DEFERRED_OPTS_IMMEDIATE);
      if (
        this.resourceTable &&
        !this.resourceTable._destroyed &&
        typeof this.resourceTable.flushDeferredDestroys === 'function'
      ) {
        try {
          deferredDestroysFlushed += toNonNegativeInt(this.resourceTable.flushDeferredDestroys(), 0);
        } catch (_) {
          // Ignore resource table deferred destroy flush errors.
        }
      }
    }

    return {
      reclaimedCount: rtReclaimedCount + stagingReclaimedSlabs,
      reclaimedBytes: rtReclaimedBytes + stagingReclaimedBytes,
      stagingReclaimedSlabs,
      stagingReclaimedBytes,
      rtReclaimedCount,
      rtReclaimedBytes,
    };
  }

  /**
   * W285 / W332: Mark an explicit scene transition (e.g. Intro Video -> Main Menu -> In-Game)
   * and immediately trim transient staging slabs & idle oversized render targets.
   *
   * @param {number} [maxIdleFrames=2]
   * @returns {{
   *   reclaimedCount: number,
   *   reclaimedBytes: number,
   *   stagingReclaimedSlabs: number,
   *   stagingReclaimedBytes: number,
   *   rtReclaimedCount: number,
   *   rtReclaimedBytes: number,
   * }}
   */
  notifySceneTransition(maxIdleFrames = this._oversizedRtReclaimIdleFrames) {
    this._pendingSceneTransition = true;
    return this.trimTransientMemory({ maxIdleFrames });
  }

  _resolveShaderReflection(shaderId, shaderEntry, stageName) {
    if (
      !this.reflectionRegistry ||
      typeof this.reflectionRegistry.getShaderReflection !== 'function'
    ) {
      return null;
    }
    if (typeof this.reflectionRegistry.getStats === 'function') {
      const stats = this.reflectionRegistry.getStats();
      if (stats && stats.shaderCount === 0) {
        return null;
      }
    }

    const explicitKey = this._shaderReflectionKeys.get(shaderId);
    if (explicitKey !== undefined) {
      const hit = this.reflectionRegistry.getShaderReflection(explicitKey);
      if (hit) return hit;
    }

    if (shaderEntry && shaderEntry.hashHex) {
      const byHash = this.reflectionRegistry.getShaderReflection(shaderEntry.hashHex);
      if (byHash) return byHash;
    }

    if (shaderEntry && shaderEntry.entryPoint && stageName) {
      const byStageName = this.reflectionRegistry.getShaderReflection(
        `${stageName}:${shaderEntry.entryPoint}`
      );
      if (byStageName) return byStageName;
    }

    return null;
  }

  _buildReflectedPipelineLayout(reflections, label) {
    if (
      !this.reflectionRegistry ||
      !Array.isArray(reflections) ||
      reflections.length === 0 ||
      !this.device ||
      typeof this.device.createPipelineLayout !== 'function'
    ) {
      return null;
    }

    const cache = this.pipelineCache;
    const hasCacheBgl =
      Boolean(cache) &&
      (typeof cache.createBindGroupLayoutFromReflection === 'function' ||
        typeof cache.getOrCreateBindGroupLayout === 'function');
    if (!hasCacheBgl && typeof this.device.createBindGroupLayout !== 'function') {
      return null;
    }

    const capturedBgls = [];
    const capturedGroupDescs = [];
    const collectorCache = {
      createBindGroupLayoutFromReflection: (desc) => {
        let bgl;
        if (cache && typeof cache.createBindGroupLayoutFromReflection === 'function') {
          bgl = cache.createBindGroupLayoutFromReflection(desc);
        } else if (cache && typeof cache.getOrCreateBindGroupLayout === 'function') {
          bgl = cache.getOrCreateBindGroupLayout(desc);
        } else {
          bgl = this.device.createBindGroupLayout(desc);
        }
        capturedBgls.push(bgl);
        capturedGroupDescs.push(desc);
        return bgl;
      },
    };

    const layoutOptions = label ? { label } : {};
    const pipelineLayout = this.reflectionRegistry.buildPipelineLayout(
      this.device,
      collectorCache,
      reflections,
      layoutOptions
    );
    return {
      pipelineLayout,
      bindGroupLayouts: capturedBgls,
      groupDescriptors: capturedGroupDescs,
    };
  }

  _isFrameOrBatchActive() {
    return Boolean(
      this._pendingEncoder !== null ||
        (Array.isArray(this._passCommands) && this._passCommands.length > 0) ||
        this._frameBegunForIndex >= 0
    );
  }

  _retireOrDestroyGpuResource(resource, byteSize = 0) {
    if (!resource || typeof resource.destroy !== 'function') {
      return;
    }
    const recordedBytes =
      byteSize > 0
        ? byteSize
        : this._deferredResourceBytes !== null
          ? this._deferredResourceBytes.get(resource) || 0
          : 0;
    if (recordedBytes > 0 && this._deferredResourceBytes !== null) {
      this._deferredResourceBytes.set(resource, recordedBytes);
    }
    if (this._isFrameOrBatchActive()) {
      if (
        !this._inFlightDeferredDestroys.has(resource) &&
        !this._deferredGpuDestroysSet.has(resource)
      ) {
        this._deferredGpuDestroysSet.add(resource);
        this._deferredGpuDestroys.push(resource);
        if (recordedBytes > 0) {
          this._pendingDeferredDestroyBytes += recordedBytes;
        }
        // W371 (#1077): Prevent unbounded pre-present deferred-destroy accumulation during
        // long loading bursts where many buffers/textures are overwritten before Present.
        if (
          this._deferredGpuDestroys.length >= this._maxDeferredGpuDestroys ||
          this._pendingDeferredDestroyBytes >= this._maxDeferredGpuDestroyBytes
        ) {
          if (this._pendingEncoder !== null && !this._hasPendingUploadCommands) {
            this._pendingEncoder = null;
          }
          if (this._pendingEncoder !== null && this._hasPendingUploadCommands) {
            this._deferredHighWaterFlushes += 1;
            this._flushPendingUploads(false);
          } else if (
            this._pendingEncoder === null &&
            (!Array.isArray(this._passCommands) || this._passCommands.length === 0)
          ) {
            this._deferredHighWaterFlushes += 1;
            this._drainDeferredGpuDestroys();
          }
        }
      }
      return;
    }
    try {
      resource.destroy();
    } catch (_) {
      // Ignore teardown errors on already-destroyed or lost-device resources.
    }
  }

  _decrementDeferredDestroyBytes(res) {
    if (!res || this._deferredResourceBytes === null) {
      return;
    }
    const bytes = this._deferredResourceBytes.get(res) || 0;
    if (bytes > 0) {
      this._pendingDeferredDestroyBytes = Math.max(0, this._pendingDeferredDestroyBytes - bytes);
    }
  }

  _drainDeferredGpuDestroys(workDoneFence = null, { immediate = false } = {}) {
    if (!Array.isArray(this._deferredGpuDestroys) || this._deferredGpuDestroys.length === 0) {
      return 0;
    }
    const batch = this._deferredGpuDestroys;
    this._deferredGpuDestroys = [];
    this._deferredGpuDestroysSet.clear();

    if (immediate) {
      for (let i = 0; i < batch.length; i += 1) {
        const res = batch[i];
        this._inFlightDeferredDestroys.delete(res);
        this._decrementDeferredDestroyBytes(res);
        if (res && typeof res.destroy === 'function') {
          try {
            res.destroy();
          } catch (_) {
            // Destroying objects on a lost or torn-down device is safe to ignore.
          }
        }
        this._deferredDestroyCount += 1;
      }
      if (this._inFlightDeferredDestroys.size === 0 && this._deferredGpuDestroys.length === 0) {
        this._pendingDeferredDestroyBytes = 0;
      }
      return batch.length;
    }

    let fence = workDoneFence;
    if (!fence || typeof fence.then !== 'function') {
      fence = this._getWorkDoneFence();
    }
    if (fence && typeof fence.then === 'function') {
      for (let i = 0; i < batch.length; i += 1) {
        const res = batch[i];
        if (res) {
          this._inFlightDeferredDestroys.add(res);
        }
      }
      const finalize = () => {
        for (let i = 0; i < batch.length; i += 1) {
          const res = batch[i];
          if (!this._inFlightDeferredDestroys.delete(res)) {
            continue;
          }
          this._decrementDeferredDestroyBytes(res);
          if (res && typeof res.destroy === 'function') {
            try {
              res.destroy();
            } catch (_) {
              // Ignore teardown errors after device loss or duplicate destroy.
            }
          }
          this._deferredDestroyCount += 1;
        }
        if (this._inFlightDeferredDestroys.size === 0 && this._deferredGpuDestroys.length === 0) {
          this._pendingDeferredDestroyBytes = 0;
        }
      };
      Promise.resolve(fence)
        .then(finalize, finalize)
        .catch(() => {});
      return batch.length;
    }

    for (let i = 0; i < batch.length; i += 1) {
      const res = batch[i];
      this._inFlightDeferredDestroys.delete(res);
      this._decrementDeferredDestroyBytes(res);
      if (res && typeof res.destroy === 'function') {
        try {
          res.destroy();
        } catch (_) {
          // Ignore teardown errors after device loss or duplicate destroy.
        }
      }
      this._deferredDestroyCount += 1;
    }
    if (this._inFlightDeferredDestroys.size === 0 && this._deferredGpuDestroys.length === 0) {
      this._pendingDeferredDestroyBytes = 0;
    }
    return batch.length;
  }

  /**
   * Publicly flush deferred GPU destroys (fence-gated by default, or immediate when requested).
   */
  flushDeferredDestroys(options = {}) {
    const immediate = Boolean(options && options.immediate === true);
    let count = 0;
    if (immediate) {
      const before = this._deferredDestroyCount;
      this._flushAllDeferredGpuDestroysImmediate();
      count += Math.max(0, this._deferredDestroyCount - before);
      if (
        this.resourceTable &&
        !this.resourceTable._destroyed &&
        typeof this.resourceTable.flushDeferredDestroys === 'function'
      ) {
        count += toNonNegativeInt(this.resourceTable.flushDeferredDestroys(), 0);
      }
      return count;
    }
    count += this._drainDeferredGpuDestroys(null, DRAIN_DEFERRED_OPTS_DEFERRED);
    if (
      this.resourceTable &&
      !this.resourceTable._destroyed &&
      typeof this.resourceTable.endFrame === 'function' &&
      !this._isFrameOrBatchActive()
    ) {
      this.resourceTable.endFrame(this.device ? this.device.queue : undefined);
    }
    return count;
  }

  _flushAllDeferredGpuDestroysImmediate(alreadyDestroyed = null) {
    const seen = alreadyDestroyed instanceof Set ? alreadyDestroyed : new Set();
    const pending = Array.isArray(this._deferredGpuDestroys) ? this._deferredGpuDestroys : [];
    this._deferredGpuDestroys = [];
    this._deferredGpuDestroysSet.clear();
    for (let i = 0; i < pending.length; i += 1) {
      const res = pending[i];
      if (!res) continue;
      this._inFlightDeferredDestroys.delete(res);
      this._decrementDeferredDestroyBytes(res);
      if (seen.has(res)) continue;
      seen.add(res);
      if (typeof res.destroy === 'function') {
        try {
          res.destroy();
        } catch (_) {
          // Destroying objects of a lost or torn-down device may throw; safe to ignore.
        }
      }
      this._deferredDestroyCount += 1;
    }
    if (this._inFlightDeferredDestroys && this._inFlightDeferredDestroys.size > 0) {
      const inFlight = Array.from(this._inFlightDeferredDestroys);
      this._inFlightDeferredDestroys.clear();
      for (let i = 0; i < inFlight.length; i += 1) {
        const res = inFlight[i];
        if (!res || seen.has(res)) continue;
        seen.add(res);
        this._decrementDeferredDestroyBytes(res);
        if (typeof res.destroy === 'function') {
          try {
            res.destroy();
          } catch (_) {
            // Destroying objects of a lost or torn-down device may throw; safe to ignore.
          }
        }
        this._deferredDestroyCount += 1;
      }
    }
    this._pendingDeferredDestroyBytes = 0;
  }

  _destroyBufferIfPresent(id) {
    const existing = this.buffers.get(id);
    if (existing) {
      const meta = this._bufferMeta.get(id);
      const trackedBytes =
        meta && typeof meta.byteSize === 'number' && meta.byteSize > 0 ? meta.byteSize : 0;
      const deferredBytes =
        trackedBytes > 0
          ? trackedBytes
          : typeof existing.size === 'number' && existing.size > 0
            ? existing.size
            : 0;
      if (trackedBytes > 0) {
        this._liveBufferBytes = Math.max(0, this._liveBufferBytes - trackedBytes);
      }
      let retired = false;
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        retired = Boolean(this.resourceRegistry.retire('buffer', id));
      }
      if (!retired) {
        this._retireOrDestroyGpuResource(existing, deferredBytes);
      }
      this.buffers.delete(id);
      this._bufferMeta.delete(id);
    }
  }

  _destroyTextureIfPresent(id) {
    const existing = this.textures.get(id);
    if (existing) {
      const meta = this._textureMeta.get(id);
      const byteSize =
        meta && typeof meta.byteSize === 'number' && meta.byteSize > 0 ? meta.byteSize : 0;
      if (byteSize > 0) {
        this._liveTextureBytes = Math.max(0, this._liveTextureBytes - byteSize);
      }
      let retiredTex = false;
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        retiredTex = Boolean(this.resourceRegistry.retire('texture', id));
        this.resourceRegistry.retire('textureView', id);
      }
      if (!retiredTex) {
        this._retireOrDestroyGpuResource(existing, byteSize);
      }
      this.textures.delete(id);
      this._textureMeta.delete(id);
      if (this._textureObjMeta) {
        this._textureObjMeta.delete(existing);
      }
    }
    this.textureViews.delete(id);
  }

  _acquireDirectWriteScratch(requiredBytes) {
    if (requiredBytes <= 0 || requiredBytes > MAX_DIRECT_WRITE_BUFFER_CHUNK_BYTES) {
      return null;
    }
    if (this._directWriteScratch && this._directWriteScratch.byteLength >= requiredBytes) {
      return this._directWriteScratch;
    }
    const allocBytes = this._directWriteScratch
      ? Math.min(
          MAX_DIRECT_WRITE_BUFFER_CHUNK_BYTES,
          Math.max(requiredBytes, this._directWriteScratch.byteLength * 2)
        )
      : requiredBytes;
    this._directWriteScratch = new Uint8Array(allocBytes);
    return this._directWriteScratch;
  }

  _handleCreateBuffer(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const usage = view.getUint32(payloadOffset + 4, true);
    const byteSize = Number(view.getBigUint64(payloadOffset + 8, true));
    const safeBytes = Number.isFinite(byteSize) && byteSize > 0 ? Math.trunc(byteSize) : 0;
    this._destroyBufferIfPresent(id);
    const buffer = this.device.createBuffer({
      size: byteSize,
      usage,
    });
    this.buffers.set(id, buffer);
    this._bufferMeta.set(id, { usage, byteSize: safeBytes });
    this._liveBufferBytes += safeBytes;
    if (this._liveBufferBytes > this._peakBufferBytes) {
      this._peakBufferBytes = this._liveBufferBytes;
    }
    if (this._deferredResourceBytes !== null && buffer && typeof buffer === 'object') {
      this._deferredResourceBytes.set(buffer, safeBytes);
    }
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('buffer', id, buffer, {
        byteSize: safeBytes,
        label: `buffer:${id}`,
      });
    }
  }

  _handleWriteBuffer(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const dstOffset = view.getUint32(payloadOffset + 4, true);
    const dataLen = view.getUint32(payloadOffset + 8, true);
    if (16 + dataLen > payloadBytes) return;

    const buffer = this.buffers.get(id);
    if (!buffer) return;

    const data = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 16, dataLen);
    if (dataLen === 0) return;

    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.stageBufferUpload === 'function' &&
      dstOffset % 4 === 0
    ) {
      const rem = dataLen % 4;
      const copySize = rem === 0 ? dataLen : dataLen + (4 - rem);
      const fitsDst = typeof buffer.size !== 'number' || dstOffset + copySize <= buffer.size;
      if (fitsDst) {
        const encoder = this._ensurePendingEncoder();
        if (encoder && typeof encoder.copyBufferToBuffer === 'function') {
          try {
            this.stagingPool.stageBufferUpload(encoder, buffer, dstOffset, data);
            this._hasPendingUploadCommands = true;
            return;
          } catch (err) {
            // If stagingPool budget is exhausted mid-frame (e.g. massive loading burst),
            // flush pending uploads and fall back cleanly to <=4 MiB chunked queue.writeBuffer.
            if (err instanceof RangeError) {
              this._flushPendingUploads(true);
            } else {
              throw err;
            }
          }
        } else if (!this._hasPendingUploadCommands) {
          this._pendingEncoder = null;
        }
      }
    }

    const isShared =
      typeof SharedArrayBuffer !== 'undefined' && data.buffer instanceof SharedArrayBuffer;
    const chunkMax = MAX_DIRECT_WRITE_BUFFER_CHUNK_BYTES;
    if (data.byteLength > chunkMax) {
      for (let off = 0; off < data.byteLength; off += chunkMax) {
        const rawLen = Math.min(chunkMax, data.byteLength - off);
        const rem = rawLen % 4;
        const alignedLen = rem === 0 ? rawLen : rawLen + (4 - rem);
        const srcSlice = data.subarray(off, off + rawLen);
        if (!isShared && rem === 0) {
          this.device.queue.writeBuffer(buffer, dstOffset + off, srcSlice);
        } else {
          const scratch = this._acquireDirectWriteScratch(alignedLen);
          const dst = scratch
            ? scratch.byteLength === alignedLen
              ? scratch
              : scratch.subarray(0, alignedLen)
            : new Uint8Array(alignedLen);
          dst.set(srcSlice, 0);
          if (rem !== 0) {
            dst.fill(0, rawLen, alignedLen);
          }
          this.device.queue.writeBuffer(buffer, dstOffset + off, dst);
        }
      }
      this._chunkedWriteBufferUploads += 1;
      return;
    }

    const rem = data.byteLength % 4;
    if (!isShared && rem === 0) {
      this.device.queue.writeBuffer(buffer, dstOffset, data);
      return;
    }
    const alignedLen = rem === 0 ? data.byteLength : data.byteLength + (4 - rem);
    const scratch = this._acquireDirectWriteScratch(alignedLen);
    const dst = scratch
      ? scratch.byteLength === alignedLen
        ? scratch
        : scratch.subarray(0, alignedLen)
      : new Uint8Array(alignedLen);
    dst.set(data, 0);
    if (rem !== 0) {
      dst.fill(0, data.byteLength, alignedLen);
    }
    this.device.queue.writeBuffer(buffer, dstOffset, dst);
  }

  _resolveShaderPayload(id, stage, rawCode, trimmed, hashOnlyHex) {
    const stageName = stageCodeToName(stage) || 'vertex';
    let code = rawCode;
    let hashHex = null;
    let entryPoint = null;
    let source = 'inline';
    let requestedHash = null;
    let resolvedHash = null;
    let fallbackProfile = null;
    let lookupHashHex = null;

    if (trimmed.length === 0 || hashOnlyHex !== null) {
      lookupHashHex = hashOnlyHex;
      if (!lookupHashHex) {
        const explicitKey = this._shaderReflectionKeys.get(id);
        const explicitBigInt =
          explicitKey !== undefined ? normalizeShaderHashBigInt(explicitKey) : null;
        lookupHashHex =
          explicitBigInt !== null
            ? formatShaderHashHex(explicitBigInt)
            : fnv1a64Hex(`empty:${stageName}:${id}`);
      }
      requestedHash = lookupHashHex;

      if (
        this.shaderFallbackRouter &&
        typeof this.shaderFallbackRouter.resolve === 'function'
      ) {
        let hintProfile = this._defaultFallbackProfile;
        // W346: Skip lookahead fallback-profile command-buffer scan when the requested hash
        // is already present in the router's WGSL pack (100% of live pack-backed shaders).
        const routerPackLoader = this.shaderFallbackRouter._packLoader;
        const knownCompileFailure =
          typeof this.shaderFallbackRouter.hasCompileFailure === 'function' &&
          this.shaderFallbackRouter.hasCompileFailure(lookupHashHex);
        const isPackHit =
          !knownCompileFailure &&
          Boolean(routerPackLoader && lookupFromPackCache(routerPackLoader, lookupHashHex));
        if (!isPackHit) {
          const pendingHints = this._ensureCommandStreamFallbackProfiles();
          if (pendingHints && pendingHints.has(id)) {
            hintProfile = pendingHints.get(id);
          } else if (this._explicitFallbackDrawContext) {
            hintProfile = selectFallbackShaderProfile(this._explicitFallbackDrawContext);
          }
        }
        const resolved = callFallbackRouterResolve(
          this.shaderFallbackRouter,
          lookupHashHex,
          stageName,
          hintProfile
        );
        if (resolved && typeof resolved.wgsl === 'string' && resolved.wgsl.length > 0) {
          code = resolved.wgsl;
          source = resolved.source || 'fallback';
          requestedHash = resolved.requestedHash || lookupHashHex;
          resolvedHash = resolved.resolvedHash !== undefined ? resolved.resolvedHash : null;
          hashHex = resolvedHash || fnv1a64Hex(code);
          entryPoint = resolved.entryPoint || extractWgslEntryPoint(code, stage);
          if (source === 'builtin') {
            fallbackProfile = normalizeFallbackProfileName(resolved.profile || hintProfile);
            this._recordFallbackProfileUsage(fallbackProfile);
          }
        }
      } else if (this.wgslPackCache) {
        const packHit = lookupFromPackCache(this.wgslPackCache, lookupHashHex);
        if (packHit && typeof packHit.wgsl === 'string' && packHit.wgsl.length > 0) {
          code = packHit.wgsl;
          source = 'pack';
          resolvedHash = lookupHashHex;
          hashHex = lookupHashHex;
          entryPoint = packHit.entryPoint || extractWgslEntryPoint(code, stage);
        }
      }
    }

    if (hashHex === null) {
      hashHex = fnv1a64Hex(code);
    }
    if (entryPoint === null) {
      entryPoint = extractWgslEntryPoint(code, stage);
    }
    if (requestedHash === null) {
      requestedHash = hashHex;
    }
    if (resolvedHash === null && source === 'inline') {
      resolvedHash = hashHex;
    }
    return { code, hashHex, entryPoint, source, requestedHash, resolvedHash, fallbackProfile, lookupHashHex };
  }

  _recordOversizedShaderHash(hashHex) {
    if (typeof hashHex !== 'string' || hashHex.length === 0) {
      return;
    }
    if (this._oversizedShaderHashes.has(hashHex)) {
      return;
    }
    if (this._oversizedShaderHashes.size >= MAX_OVERSIZED_SHADER_HASHES) {
      const oldest = this._oversizedShaderHashes.values().next().value;
      if (oldest !== undefined) {
        this._oversizedShaderHashes.delete(oldest);
      }
    }
    this._oversizedShaderHashes.add(hashHex);
  }

  _clearRangedShaderCompileQueue() {
    if (this._rangedShaderCompilePumpTimer !== null) {
      clearTimeout(this._rangedShaderCompilePumpTimer);
      this._rangedShaderCompilePumpTimer = null;
    }
    this._rangedShaderCompileMicrotaskPending = false;
    this._rangedShaderWaitingForPipelineIdle = false;
    this._activeRangedShaderCompiles = 0;
    if (Array.isArray(this._rangedShaderCompileQueue) && this._rangedShaderCompileQueue.length > 0) {
      const queued = this._rangedShaderCompileQueue;
      this._rangedShaderCompileQueue = [];
      for (let i = 0; i < queued.length; i += 1) {
        const job = queued[i];
        if (job && typeof job.resolve === 'function') {
          try {
            job.resolve(null);
          } catch (_) {
            // Ignore teardown resolver errors.
          }
        }
      }
    }
  }

  _isAsyncPipelineCacheAtCapacity() {
    const cache = this.asyncPipelineCache;
    if (!cache) {
      return false;
    }
    let inFlight = 0;
    let limit = 0;
    if (typeof cache._inFlight === 'number' && typeof cache.maxConcurrentCompiles === 'number') {
      inFlight = cache._inFlight;
      limit = cache.maxConcurrentCompiles;
    } else if (typeof cache.getStats === 'function') {
      try {
        const st = cache.getStats();
        if (st && typeof st === 'object') {
          inFlight =
            typeof st.activeCompiles === 'number'
              ? st.activeCompiles
              : typeof st.inFlight === 'number'
                ? st.inFlight
                : 0;
          limit =
            typeof st.maxConcurrentCompiles === 'number'
              ? st.maxConcurrentCompiles
              : typeof st.maxConcurrent === 'number'
                ? st.maxConcurrent
                : 0;
        }
      } catch (_) {
        return false;
      }
    }
    return limit > 0 && inFlight >= limit;
  }

  _scheduleRangedShaderCompilePump(useMacrotask, delayMs = 0) {
    if (this._destroyed || this._deviceLost || this._deviceRecovering) {
      return;
    }
    if (useMacrotask) {
      if (this._rangedShaderCompilePumpTimer !== null) {
        return;
      }
      this._rangedShaderCompileYields += 1;
      const waitMs = typeof delayMs === 'number' && delayMs > 0 ? delayMs : 0;
      this._rangedShaderCompilePumpTimer = setTimeout(() => {
        this._rangedShaderCompilePumpTimer = null;
        this._pumpRangedShaderCompileQueue();
      }, waitMs);
      return;
    }
    if (this._rangedShaderCompileMicrotaskPending || this._rangedShaderCompilePumpTimer !== null) {
      return;
    }
    this._rangedShaderCompileMicrotaskPending = true;
    Promise.resolve().then(() => {
      this._rangedShaderCompileMicrotaskPending = false;
      this._pumpRangedShaderCompileQueue();
    });
  }

  _enqueueRangedShaderCompile(runCompile, meta = null) {
    const stage = meta && meta.stage !== undefined ? meta.stage : null;
    const isGraphics =
      stage === SHADER_STAGE.COMPUTE
        ? false
        : typeof stage === 'string'
          ? stage.trim().toLowerCase() !== 'compute'
          : true;
    const rawBytes =
      meta && typeof meta.byteLength === 'number'
        ? meta.byteLength
        : meta && typeof meta.wgslBytes === 'number'
          ? meta.wgslBytes
          : Number.MAX_SAFE_INTEGER;
    const byteLength = Number.isFinite(rawBytes) && rawBytes >= 0 ? rawBytes : Number.MAX_SAFE_INTEGER;
    return new Promise((resolve) => {
      this._rangedShaderCompileQueue.push({
        runCompile,
        resolve,
        stage,
        isGraphics,
        byteLength,
        seq: ++this._rangedShaderCompileSeq,
      });
      this._pumpRangedShaderCompileQueue();
    });
  }

  _dequeueNextRangedShaderCompileJob() {
    const q = this._rangedShaderCompileQueue;
    const len = q.length;
    if (len === 0) {
      return null;
    }
    if (len === 1) {
      return q.pop();
    }
    let bestIdx = 0;
    let best = q[0];
    for (let i = 1; i < len; i += 1) {
      const cand = q[i];
      const candGraphics = cand && cand.isGraphics !== false;
      const bestGraphics = best && best.isGraphics !== false;
      const candBytes =
        cand && typeof cand.byteLength === 'number' ? cand.byteLength : Number.MAX_SAFE_INTEGER;
      const bestBytes =
        best && typeof best.byteLength === 'number' ? best.byteLength : Number.MAX_SAFE_INTEGER;
      const candSeq = cand && typeof cand.seq === 'number' ? cand.seq : i;
      const bestSeq = best && typeof best.seq === 'number' ? best.seq : bestIdx;
      if (
        (candGraphics && !bestGraphics) ||
        (candGraphics === bestGraphics &&
          (candBytes < bestBytes || (candBytes === bestBytes && candSeq < bestSeq)))
      ) {
        bestIdx = i;
        best = cand;
      }
    }
    const last = q.pop();
    if (bestIdx < q.length) {
      q[bestIdx] = last;
    }
    return best;
  }

  _drainRangedShaderCompileQueue() {
    return this._pumpRangedShaderCompileQueue();
  }

  _pumpRangedShaderCompileQueue() {
    if (this._destroyed || this._deviceLost || this._deviceRecovering || !this.device) {
      this._clearRangedShaderCompileQueue();
      return;
    }
    const maxConc = this._maxConcurrentShaderCompiles;
    const budgetMs = this._shaderCompileSliceBudgetMs;
    const sliceStartMs = this._getNowMs();
    let startedInSlice = 0;

    while (
      this._rangedShaderCompileQueue.length > 0 &&
      (maxConc === 0 || this._activeRangedShaderCompiles < maxConc)
    ) {
      // Wave-5 (#663, #1077): Yield shader compilation when AsyncPipelineCache is already at
      // capacity so shader compiles and pipeline compiles never pile up on Dawn simultaneously.
      if (this._isAsyncPipelineCacheAtCapacity()) {
        this._scheduleRangedShaderCompilePump(true, DEFAULT_HEAVY_SHADER_YIELD_MS);
        if (
          !this._rangedShaderWaitingForPipelineIdle &&
          this.asyncPipelineCache &&
          typeof this.asyncPipelineCache.whenIdle === 'function'
        ) {
          this._rangedShaderWaitingForPipelineIdle = true;
          Promise.resolve(this.asyncPipelineCache.whenIdle()).then(
            () => {
              this._rangedShaderWaitingForPipelineIdle = false;
              if (this._destroyed || this._deviceLost || this._deviceRecovering) {
                return;
              }
              if (this._rangedShaderCompileQueue.length > 0) {
                if (this._rangedShaderCompilePumpTimer !== null) {
                  clearTimeout(this._rangedShaderCompilePumpTimer);
                  this._rangedShaderCompilePumpTimer = null;
                }
                this._pumpRangedShaderCompileQueue();
              }
            },
            () => {
              this._rangedShaderWaitingForPipelineIdle = false;
            }
          );
        }
        return;
      }

      if (startedInSlice > 0 && budgetMs > 0) {
        const elapsed = this._getNowMs() - sliceStartMs;
        if (elapsed >= budgetMs) {
          this._scheduleRangedShaderCompilePump(true);
          return;
        }
      }

      const job = this._dequeueNextRangedShaderCompileJob();
      if (!job) {
        continue;
      }
      const isHeavyShader =
        typeof job.byteLength === 'number' &&
        Number.isFinite(job.byteLength) &&
        job.byteLength > DEFAULT_HEAVY_SHADER_YIELD_BYTES;
      startedInSlice += 1;
      this._activeRangedShaderCompiles += 1;
      if (this._activeRangedShaderCompiles > this._peakConcurrentRangedShaderCompiles) {
        this._peakConcurrentRangedShaderCompiles = this._activeRangedShaderCompiles;
      }

      let outcome = null;
      try {
        outcome = job.runCompile();
      } catch (_) {
        outcome = null;
      }

      const compilationPromise =
        outcome && outcome.compilationPromise && typeof outcome.compilationPromise.then === 'function'
          ? outcome.compilationPromise
          : null;
      const finalize = outcome && typeof outcome.finalize === 'function' ? outcome.finalize : null;

      if (compilationPromise) {
        const onSettled = () => {
          this._activeRangedShaderCompiles = Math.max(0, this._activeRangedShaderCompiles - 1);
          let result = null;
          if (finalize) {
            try {
              result = finalize();
            } catch (_) {
              result = null;
            }
          }
          job.resolve(result);
          if (this._rangedShaderCompileQueue.length > 0) {
            const elapsedAfter = this._getNowMs() - sliceStartMs;
            const needMacrotaskYield = isHeavyShader || (budgetMs > 0 && elapsedAfter >= budgetMs);
            this._scheduleRangedShaderCompilePump(
              needMacrotaskYield,
              isHeavyShader ? DEFAULT_HEAVY_SHADER_YIELD_MS : 0
            );
          }
        };
        compilationPromise.then(onSettled, onSettled);
      } else {
        let result = null;
        if (finalize) {
          try {
            result = finalize();
          } catch (_) {
            result = null;
          }
        }
        job.resolve(result);
        const elapsedAfter = this._getNowMs() - sliceStartMs;
        const needMacrotaskYield = isHeavyShader || (budgetMs > 0 && elapsedAfter >= budgetMs);
        const releaseSlot = () => {
          this._activeRangedShaderCompiles = Math.max(0, this._activeRangedShaderCompiles - 1);
          if (
            !this._destroyed &&
            !this._deviceLost &&
            !this._deviceRecovering &&
            this._rangedShaderCompileQueue.length > 0
          ) {
            this._pumpRangedShaderCompileQueue();
          }
        };
        if (needMacrotaskYield) {
          this._rangedShaderCompileYields += 1;
          const yieldDelay = isHeavyShader ? DEFAULT_HEAVY_SHADER_YIELD_MS : 0;
          const timerId = setTimeout(() => {
            if (this._rangedShaderCompilePumpTimer === timerId) {
              this._rangedShaderCompilePumpTimer = null;
            }
            releaseSlot();
          }, yieldDelay);
          if (this._rangedShaderCompilePumpTimer === null) {
            this._rangedShaderCompilePumpTimer = timerId;
          }
          return;
        }
        Promise.resolve().then(releaseSlot);
      }
    }
  }

  _compileShaderModuleEntry(id, entry, allowDirectThrowOnEmpty) {
    if (!this.device) {
      return null;
    }
    if (entry.oversizedRejected === true) {
      return null;
    }
    const stageName = stageCodeToName(entry.stage) || 'vertex';
    const recordCompileFailure = () => {
      if (
        !this.shaderFallbackRouter ||
        typeof this.shaderFallbackRouter.recordMissingShader !== 'function'
      ) {
        return;
      }
      const failHash = entry.requestedHash || entry.hashHex;
      if (!failHash) {
        return;
      }
      try {
        this.shaderFallbackRouter.recordMissingShader(
          failHash,
          stageName,
          'compile_failed'
        );
      } catch (_) {
        // Never allow telemetry tracking errors to interrupt shader compilation.
      }
    };

    const maxBytes = this._maxWgslCompileBytes;
    const rawCodeLen = typeof entry.code === 'string' ? entry.code.length : 0;
    let code =
      maxBytes > 0 && rawCodeLen > maxBytes
        ? entry.code
        : applyRtBvhPreludeIfNeeded(entry.code);
    let codeBytes = typeof code === 'string' ? code.length : 0;

    // Wave-4 (#663, #1077): Hard WGSL single-shader byte-size cap.
    // Never pass multi-MB WGSL (2-15.9 MB) to device.createShaderModule(), which hung Dawn/Tint/DXC.
    if (maxBytes > 0 && codeBytes > maxBytes) {
      this._oversizedWgslFallbacks += 1;
      const failHash = entry.requestedHash || entry.hashHex;
      if (failHash) {
        this._recordOversizedShaderHash(failHash);
      }
      recordCompileFailure();
      if (
        this.shaderFallbackRouter &&
        typeof this.shaderFallbackRouter.resolve === 'function' &&
        entry.source !== 'builtin'
      ) {
        const hintProfile = entry.fallbackProfile || this._defaultFallbackProfile;
        const lookupKey = failHash || fnv1a64Hex(`oversized:${stageName}:${id}`);
        const fb = callFallbackRouterResolve(
          this.shaderFallbackRouter,
          lookupKey,
          stageName,
          hintProfile
        );
        if (
          fb &&
          typeof fb.wgsl === 'string' &&
          fb.wgsl.length > 0 &&
          fb.wgsl.length <= maxBytes &&
          fb.source !== 'pack'
        ) {
          const fbCode = applyRtBvhPreludeIfNeeded(fb.wgsl);
          if (typeof fbCode === 'string' && fbCode.length > 0 && fbCode.length <= maxBytes) {
            entry.code = fb.wgsl;
            entry.codeLen = fb.wgsl.length;
            entry.isInlineCode = false;
            entry.source = fb.source || 'builtin';
            entry.entryPoint = fb.entryPoint || extractWgslEntryPoint(fb.wgsl, entry.stage);
            entry.fallbackProfile = normalizeFallbackProfileName(fb.profile || hintProfile);
            entry.oversizedFallback = true;
            this._recordFallbackProfileUsage(entry.fallbackProfile);
            code = fbCode;
            codeBytes = fbCode.length;
          } else {
            entry.oversizedRejected = true;
            return null;
          }
        } else {
          entry.oversizedRejected = true;
          return null;
        }
      } else {
        entry.oversizedRejected = true;
        return null;
      }
    }

    let module = null;
    let compilationInfoPromise = null;
    const canReuseBuiltinModule =
      entry.source === 'builtin' &&
      !allowDirectThrowOnEmpty &&
      typeof code === 'string' &&
      code.length > 0;
    const builtinCacheKey = canReuseBuiltinModule ? `${entry.stage}:${code}` : null;
    const cachedBuiltin =
      builtinCacheKey !== null ? this._cachedBuiltinShaderModules.get(builtinCacheKey) || null : null;

    if (cachedBuiltin && cachedBuiltin.module) {
      module = cachedBuiltin.module;
      compilationInfoPromise = cachedBuiltin.compilationInfoPromise || null;
    } else {
      const shaderStallStartMs = this._gpuStallWarnMs > 0 ? this._getNowMs() : 0;
      const shaderStallHash = entry.requestedHash || entry.hashHex || null;
      if (allowDirectThrowOnEmpty && !entry.oversizedFallback) {
        module = this.device.createShaderModule({ code });
      } else {
        try {
          module = this.device.createShaderModule({ code });
        } catch (err) {
          if (this._gpuStallWarnMs > 0) {
            this._maybeWarnGpuStall('createShaderModule', shaderStallStartMs, shaderStallHash);
          }
          recordCompileFailure();
          if (!entry.oversizedFallback) {
            this._recordOpValidationError('CreateShaderModule', err);
          }
          return null;
        }
      }
      if (this._gpuStallWarnMs > 0) {
        this._maybeWarnGpuStall('createShaderModule', shaderStallStartMs, shaderStallHash);
      }
      if (module && typeof module.getCompilationInfo === 'function') {
        try {
          compilationInfoPromise = Promise.resolve(module.getCompilationInfo()).then(
            (info) => {
              if (this._gpuStallWarnMs > 0) {
                const infoMs = this._getNowMs() - shaderStallStartMs;
                if (infoMs > this._gpuStallWarnMs) {
                  this._logDiagnostic(
                    'warn',
                    `[wgpu_worker] GPU stall: op=shaderCompilationInfo durationMs=${Math.round(infoMs)} id=${id} stage=${stageName} shader=${shaderStallHash || 'inline'} bytes=${codeBytes} frame=${this.frameCount}`
                  );
                }
              }
              if (this._destroyed || !info || !Array.isArray(info.messages)) {
                return;
              }
              let hadError = false;
              for (let i = 0; i < info.messages.length; i += 1) {
                const m = info.messages[i];
                if (m && m.type === 'error') {
                  hadError = true;
                  const loc =
                    typeof m.lineNum === 'number' && m.lineNum > 0 ? ` (line ${m.lineNum})` : '';
                  this._recordOpValidationError(
                    'CreateShaderModule',
                    `shader ${id}${loc}: ${m.message || 'WGSL compilation error'}`
                  );
                }
              }
              if (hadError) {
                recordCompileFailure();
              }
            },
            () => {}
          );
        } catch (_) {
          compilationInfoPromise = null;
          // Non-fatal: ignore compilation info inspection errors on mock devices.
        }
      }
      if (builtinCacheKey !== null && module) {
        this._cachedBuiltinShaderModules.set(builtinCacheKey, {
          module,
          compilationInfoPromise,
        });
      }
    }
    entry.module = module;
    entry._compilationInfoPromise = compilationInfoPromise;
    this._materializedShaderCount += 1;
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      const byteSize =
        entry.isInlineCode && code === entry.code
          ? Math.max(0, Math.trunc(entry.codeLen || 0))
          : codeBytes;
      this.resourceRegistry.register('shaderModule', id, module, {
        byteSize,
        label: `shaderModule:${id}`,
      });
    }
    // #1077: Release full WGSL text after GPUShaderModule compilation, retaining only a compact
    // metadata stub (@group(0) bindings + stage entry points) for downstream pipeline creation.
    if (
      entry.source !== 'builtin' &&
      typeof entry.code === 'string' &&
      entry.code.length > COMPACT_WGSL_THRESHOLD_CHARS
    ) {
      entry.code = compactWgslForMetadata(entry.code);
    }
    return entry;
  }

  _triggerLazyRangedShaderResolve(id, entry) {
    if (!entry || !entry.lazyRanged) {
      return null;
    }
    const lazy = entry.lazyRanged;
    entry.lazyRanged = null;
    if (this._shaderModuleGenerations.get(id) !== lazy.shaderGen) {
      return null;
    }
    return this._scheduleRangedShaderResolve(
      id,
      lazy.stage,
      lazy.lookupHashHex,
      lazy.shaderGen,
      lazy.packCache,
      lazy.lookupKey
    );
  }

  /**
   * W106: return the shader module entry for `id`, materializing the underlying GPUShaderModule
   * on first pipeline reference when `lazyShaderCompile` is enabled.
   */
  ensureShaderModule(id) {
    const entry = this.shaderModules.get(id);
    if (!entry) {
      return null;
    }
    if (entry.module) {
      return entry;
    }
    if (entry.lazyRanged) {
      this._triggerLazyRangedShaderResolve(id, entry);
      return null;
    }
    if (entry.pendingRanged || entry.oversizedRejected) {
      return null;
    }
    return this._compileShaderModuleEntry(id, entry, false);
  }

  /**
   * W136: await settlement of all currently in-flight ranged shader fetches and queued compiles.
   */
  async flushPendingRangedShaders() {
    while (this._inFlightRangedShaders.size > 0) {
      const pending = Array.from(this._inFlightRangedShaders.values(), (r) => r.promise).filter(Boolean);
      if (pending.length === 0) {
        break;
      }
      await Promise.allSettled(pending);
    }
  }

  _isRangedShaderPending(shaderId) {
    if (!shaderId) {
      return false;
    }
    if (this._inFlightRangedShaders.has(shaderId)) {
      return true;
    }
    const entry = this.shaderModules.get(shaderId);
    return Boolean(entry && entry.pendingRanged === true);
  }

  _copyPacketPayload(view, payloadOffset, payloadBytes) {
    const raw = new Uint8Array(view.buffer, view.byteOffset + payloadOffset, payloadBytes);
    return raw.slice();
  }

  _registerPendingRangedPipeline(id, kind, opcode, shaderIds, view, payloadOffset, payloadBytes) {
    const uniqueShaderIds = [];
    let anyPending = false;
    let deferredOnly = false;
    for (let i = 0; i < shaderIds.length; i += 1) {
      const sid = shaderIds[i];
      if (!sid) {
        continue;
      }
      if (!uniqueShaderIds.includes(sid)) {
        uniqueShaderIds.push(sid);
      }
      const entry = this.shaderModules.get(sid);
      if (entry && entry.lazyRanged) {
        this._triggerLazyRangedShaderResolve(sid, entry);
      }
      const sidPending = this._isRangedShaderPending(sid);
      if (sidPending) {
        anyPending = true;
      }
      if (
        (entry && entry.pendingRanged === true) ||
        (sidPending && (!entry || entry.rangedFailed === true || (!entry.module && !entry.code)))
      ) {
        deferredOnly = true;
      }
    }
    if (!anyPending) {
      this._pendingRangedPipelines.delete(id);
      return { pending: false, deferredOnly: false };
    }
    if (
      !this._pendingRangedPipelines.has(id) &&
      this._pendingRangedPipelines.size >= this._maxPendingRangedPipelines
    ) {
      const oldestId = this._pendingRangedPipelines.keys().next().value;
      if (oldestId !== undefined) {
        this._pendingRangedPipelines.delete(oldestId);
      }
    }
    this._pendingRangedPipelines.set(id, {
      id,
      kind,
      opcode,
      shaderIds: uniqueShaderIds,
      payload: this._copyPacketPayload(view, payloadOffset, payloadBytes),
      deviceGeneration: this._deviceGeneration,
      deferredOnly,
      hadReadyShader: false,
    });
    return { pending: true, deferredOnly };
  }

  _onRangedShaderSettled(shaderId, succeeded) {
    if (this._destroyed || this._deviceLost || this._deviceRecovering || !this.device) {
      return;
    }
    if (this._pendingRangedPipelines.size === 0) {
      return;
    }
    const readyToBuild = [];
    for (const [pipeId, rec] of this._pendingRangedPipelines.entries()) {
      if (!rec || rec.deviceGeneration !== this._deviceGeneration) {
        this._pendingRangedPipelines.delete(pipeId);
        continue;
      }
      if (!rec.shaderIds.includes(shaderId)) {
        continue;
      }
      if (succeeded) {
        rec.hadReadyShader = true;
      }
      let stillPending = false;
      for (let i = 0; i < rec.shaderIds.length; i += 1) {
        if (this._isRangedShaderPending(rec.shaderIds[i])) {
          stillPending = true;
          break;
        }
      }
      if (!stillPending) {
        this._pendingRangedPipelines.delete(pipeId);
        if (rec.hadReadyShader) {
          readyToBuild.push(rec);
        }
      }
    }

    for (let i = 0; i < readyToBuild.length; i += 1) {
      const rec = readyToBuild[i];
      if (this._destroyed || this._deviceLost || this._deviceRecovering || !this.device) {
        break;
      }
      if (rec.deviceGeneration !== this._deviceGeneration) {
        continue;
      }
      let allShadersReady = true;
      for (let j = 0; j < rec.shaderIds.length; j += 1) {
        const entry = this.shaderModules.get(rec.shaderIds[j]);
        if (!entry || entry.pendingRanged || entry.rangedFailed || entry.oversizedRejected) {
          allShadersReady = false;
          break;
        }
      }
      if (!allShadersReady) {
        continue;
      }
      const view = new DataView(rec.payload.buffer, rec.payload.byteOffset, rec.payload.byteLength);
      if (rec.opcode === OPCODES.CREATE_RENDER_PIPELINE) {
        this._handleCreateRenderPipeline(view, 0);
      } else if (rec.opcode === OPCODES.CREATE_COMPUTE_PIPELINE) {
        this._handleCreateComputePipeline(view, 0);
      } else if (
        rec.opcode === OPCODES.CREATE_RENDER_PIPELINE2 ||
        rec.opcode === OPCODES.CREATE_COMPUTE_PIPELINE2
      ) {
        this._dispatchExtensionPacket(rec.opcode, view, 0, rec.payload.byteLength);
      }
    }
  }

  _invalidatePipelinesReferencingShader(shaderId) {
    for (const [pipeId, pipeMeta] of this._pipelineMeta.entries()) {
      if (pipeMeta && (pipeMeta.vsShaderId === shaderId || pipeMeta.fsShaderId === shaderId)) {
        this._invalidateSpecializedPipelinesFor(pipeId);
      }
    }
  }

  _scheduleRangedShaderResolve(id, stage, lookupHashHex, shaderGen, packCache, lookupKey) {
    const existing = this._inFlightRangedShaders.get(id);
    if (
      existing &&
      existing.hashHex === lookupHashHex &&
      existing.stage === stage &&
      existing.deviceGeneration === this._deviceGeneration &&
      existing.packCache === packCache
    ) {
      existing.gen = shaderGen;
      return existing.promise;
    }

    const deviceGeneration = this._deviceGeneration;
    const record = {
      id,
      stage,
      hashHex: lookupHashHex,
      gen: shaderGen,
      deviceGeneration,
      packCache,
      promise: null,
    };
    this._inFlightRangedShaders.set(id, record);
    this._rangedShaderFetches += 1;

    const isStillCurrent = () =>
      !this._destroyed &&
      !this._deviceLost &&
      !this._deviceRecovering &&
      Boolean(this.device) &&
      this._deviceGeneration === deviceGeneration &&
      this._shaderModuleGenerations.get(id) === record.gen;

    const cleanup = () => {
      if (this._inFlightRangedShaders.get(id) === record) {
        this._inFlightRangedShaders.delete(id);
      }
    };

    const failCurrent = (errMsg) => {
      const cur = this.shaderModules.get(id);
      if (cur && cur.pendingRanged) {
        cur.pendingRanged = false;
        cur.rangedFailed = true;
        if (
          !cur.module &&
          cur.source !== 'pack' &&
          cur.source !== 'builtin' &&
          cur.source !== 'fallback'
        ) {
          this.shaderModules.delete(id);
        }
      }
      this._rangedShaderErrors += 1;
      if (errMsg !== undefined && errMsg !== null) {
        this._recordOpValidationError('CreateShaderModule', errMsg);
      }
      this._onRangedShaderSettled(id, false);
    };

    let fetchPromise;
    try {
      fetchPromise = Promise.resolve(packCache.getWgslAsync(lookupKey));
    } catch (err) {
      cleanup();
      if (isStillCurrent()) {
        failCurrent(err);
      }
      return Promise.resolve(null);
    }

    const settledPromise = fetchPromise.then(
      (rawResult) => {
        if (!isStillCurrent()) {
          cleanup();
          return null;
        }
        const hit = unwrapPackLookupResult(rawResult);
        const wgsl = hit && typeof hit.wgsl === 'string' ? hit.wgsl : null;
        if (!wgsl || wgsl.length === 0) {
          cleanup();
          failCurrent(`shader ${id}: ranged WGSL fetch for ${lookupHashHex} returned empty payload`);
          return null;
        }
        let metaEntryPoint = hit && hit.entryPoint ? hit.entryPoint : null;
        if (!metaEntryPoint && typeof packCache.getShaderMetadata === 'function') {
          try {
            const meta = packCache.getShaderMetadata(lookupKey);
            if (meta && typeof meta.entryPoint === 'string' && meta.entryPoint.length > 0) {
              metaEntryPoint = meta.entryPoint;
            }
          } catch (_) {
            // Optional metadata lookup failure is non-fatal.
          }
        }
        const realEntry = {
          module: null,
          stage,
          code: wgsl,
          codeLen: wgsl.length,
          isInlineCode: false,
          hashHex: lookupHashHex,
          entryPoint: metaEntryPoint || extractWgslEntryPoint(wgsl, stage),
          source: 'pack',
          requestedHash: lookupHashHex,
          resolvedHash: lookupHashHex,
          fallbackProfile: null,
        };
        // Wave-5 (#663, #1077): Quarantine oversized ranged WGSL immediately BEFORE ever
        // entering _rangedShaderCompileQueue or calling device.createShaderModule().
        const maxBytes = this._maxWgslCompileBytes;
        if (maxBytes > 0 && wgsl.length > maxBytes) {
          const compiled = this._compileShaderModuleEntry(id, realEntry, false);
          cleanup();
          if (!isStillCurrent()) {
            return null;
          }
          if (!compiled) {
            if (realEntry.oversizedRejected) {
              const cur = this.shaderModules.get(id);
              if (cur && cur.pendingRanged) {
                cur.pendingRanged = false;
                cur.oversizedRejected = true;
              }
              this._onRangedShaderSettled(id, false);
              return null;
            }
            failCurrent(null);
            return null;
          }
          this.shaderModules.set(id, compiled);
          this._shaderModuleGenerations.set(id, ++this._nextShaderModuleGen);
          this._invalidatePipelinesReferencingShader(id);
          this._rangedShaderReady += 1;
          this._onRangedShaderSettled(id, true);
          return compiled;
        }
        return this._enqueueRangedShaderCompile(
          () => {
            if (!isStillCurrent()) {
              cleanup();
              return null;
            }
            const compiled = this._compileShaderModuleEntry(id, realEntry, false);
            return {
              compilationPromise: compiled ? compiled._compilationInfoPromise || null : null,
              finalize: () => {
                cleanup();
                if (!isStillCurrent()) {
                  return null;
                }
                if (!compiled) {
                  if (realEntry.oversizedRejected) {
                    const cur = this.shaderModules.get(id);
                    if (cur && cur.pendingRanged) {
                      cur.pendingRanged = false;
                      cur.oversizedRejected = true;
                    }
                    this._onRangedShaderSettled(id, false);
                    return null;
                  }
                  failCurrent(null);
                  return null;
                }
                this.shaderModules.set(id, compiled);
                this._shaderModuleGenerations.set(id, ++this._nextShaderModuleGen);
                this._invalidatePipelinesReferencingShader(id);
                this._rangedShaderReady += 1;
                this._onRangedShaderSettled(id, true);
                return compiled;
              },
            };
          },
          { stage, byteLength: wgsl.length }
        );
      },
      (err) => {
        cleanup();
        if (
          !this._destroyed &&
          this._deviceGeneration === deviceGeneration &&
          this._shaderModuleGenerations.get(id) === record.gen
        ) {
          failCurrent(err);
        }
        return null;
      }
    );

    record.promise = settledPromise;
    return settledPromise;
  }

  _handleCreateShaderModule(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const stage = view.getUint32(payloadOffset + 4, true);
    const codeLen = view.getUint32(payloadOffset + 8, true);
    if (16 + codeLen > payloadBytes) {
      this._recordOpValidationError(
        'CreateShaderModule',
        `shader ${id}: truncated payload (${payloadBytes}B < ${16 + codeLen}B)`
      );
      return;
    }

    const rawBytes = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 16, codeLen);
    const isShared = typeof SharedArrayBuffer !== 'undefined' && view.buffer instanceof SharedArrayBuffer;
    const rawCode = this._decoder.decode(isShared ? rawBytes.slice() : rawBytes);
    const trimmed = rawCode.trim();
    const hashOnlyHex = parseHashOnlyShaderPayload(trimmed);

    const shaderGen = ++this._nextShaderModuleGen;
    this._shaderModuleGenerations.set(id, shaderGen);

    const resolved = this._resolveShaderPayload(id, stage, rawCode, trimmed, hashOnlyHex);

    // W136: On sync miss for a hash lookup, check whether a ranged pack cache indexes this hash
    // and supports on-demand HTTP 206 Range resolution via `getWgslAsync(hash)`.
    let rangedPackCache = null;
    let rangedLookupKey = null;
    const isQuarantinedOversized =
      resolved.lookupHashHex !== null && this._oversizedShaderHashes.has(resolved.lookupHashHex);
    if (resolved.source !== 'pack' && resolved.lookupHashHex !== null && !isQuarantinedOversized) {
      const candidatePackCache =
        this.wgslPackCache ||
        (this.shaderFallbackRouter && this.shaderFallbackRouter._packLoader) ||
        null;
      if (candidatePackCache && typeof candidatePackCache.getWgslAsync === 'function') {
        const alreadyInMemory =
          this.shaderFallbackRouter &&
          this.shaderFallbackRouter._packLoader !== candidatePackCache
            ? lookupFromPackCache(candidatePackCache, resolved.lookupHashHex)
            : null;
        if (!alreadyInMemory) {
          const hasCheck = resolvePackCacheHasKey(candidatePackCache, resolved.lookupHashHex);
          if (hasCheck.matched) {
            rangedPackCache = candidatePackCache;
            rangedLookupKey = hasCheck.key;
          }
        }
      }
    }

    const wasInFlightRanged = this._inFlightRangedShaders.has(id);
    if (rangedPackCache && this._maxWgslCompileBytes > 0 && typeof rangedPackCache.getShaderMetadata === 'function') {
      let rangedKnownBytes = null;
      try {
        const meta = rangedPackCache.getShaderMetadata(rangedLookupKey);
        if (meta && typeof meta.length === 'number' && Number.isFinite(meta.length) && meta.length > 0) {
          rangedKnownBytes = meta.length;
        }
      } catch (_) {
        // Optional metadata lookup failure is non-fatal.
      }
      if (rangedKnownBytes !== null && rangedKnownBytes > this._maxWgslCompileBytes) {
        // Wave-5 (#663, #1077): Quarantine oversized ranged shaders immediately from index metadata
        // BEFORE HTTP 206 fetch, _rangedShaderCompileQueue, or device.createShaderModule().
        this._inFlightRangedShaders.delete(id);
        this._oversizedWgslFallbacks += 1;
        if (resolved.lookupHashHex) {
          this._recordOversizedShaderHash(resolved.lookupHashHex);
        }
        if (
          this.shaderFallbackRouter &&
          typeof this.shaderFallbackRouter.recordMissingShader === 'function' &&
          resolved.lookupHashHex
        ) {
          try {
            this.shaderFallbackRouter.recordMissingShader(
              resolved.lookupHashHex,
              stageCodeToName(stage) || 'vertex',
              'compile_failed'
            );
          } catch (_) {
            // Never allow telemetry tracking errors to interrupt shader compilation.
          }
        }
        if (
          this.shaderFallbackRouter &&
          resolved.source === 'builtin' &&
          typeof resolved.code === 'string' &&
          resolved.code.length > 0 &&
          resolved.code.length <= this._maxWgslCompileBytes
        ) {
          const fbEntry = {
            module: null,
            stage,
            code: resolved.code,
            codeLen: resolved.code.length,
            isInlineCode: false,
            hashHex: resolved.hashHex,
            entryPoint: resolved.entryPoint,
            source: resolved.source,
            requestedHash: resolved.requestedHash,
            resolvedHash: resolved.resolvedHash,
            fallbackProfile: resolved.fallbackProfile,
            oversizedFallback: true,
          };
          const compiledFb = this._compileShaderModuleEntry(id, fbEntry, false);
          if (compiledFb) {
            this.shaderModules.set(id, compiledFb);
          }
          if (wasInFlightRanged || this._pendingRangedPipelines.size > 0) {
            this._onRangedShaderSettled(id, Boolean(compiledFb));
          }
          return;
        }
        if (wasInFlightRanged) {
          this.shaderModules.delete(id);
        }
        if (wasInFlightRanged || this._pendingRangedPipelines.size > 0) {
          this._onRangedShaderSettled(id, false);
        }
        return;
      }
    }
    if (!rangedPackCache) {
      this._inFlightRangedShaders.delete(id);
    }

    const entry = {
      module: null,
      stage,
      code: resolved.code,
      codeLen,
      isInlineCode: resolved.code === rawCode,
      hashHex: resolved.hashHex,
      entryPoint: resolved.entryPoint,
      source: resolved.source,
      requestedHash: resolved.requestedHash,
      resolvedHash: resolved.resolvedHash,
      fallbackProfile: resolved.fallbackProfile,
    };

    // Wave-4 (#663, #1077): Defer throwaway builtin fallback shader & pipeline compilation when
    // a ranged shader-pack index match exists so we compile 1 real pipeline instead of 2 per PSO.
    if (
      rangedPackCache &&
      (this.shaderFallbackRouter === null || this._deferRangedFallbackPipelines)
    ) {
      entry.pendingRanged = true;
      if (this._lazyRangedShaderCompile) {
        this._inFlightRangedShaders.delete(id);
        entry.lazyRanged = {
          stage,
          lookupHashHex: resolved.lookupHashHex,
          shaderGen,
          packCache: rangedPackCache,
          lookupKey: rangedLookupKey,
        };
        this._deferredShaderCount += 1;
        this.shaderModules.set(id, entry);
        if (this._pendingRangedPipelines.size > 0) {
          for (const rec of this._pendingRangedPipelines.values()) {
            if (rec && Array.isArray(rec.shaderIds) && rec.shaderIds.includes(id)) {
              this._triggerLazyRangedShaderResolve(id, entry);
              break;
            }
          }
        }
        return;
      }
      this.shaderModules.set(id, entry);
      this._scheduleRangedShaderResolve(
        id,
        stage,
        resolved.lookupHashHex,
        shaderGen,
        rangedPackCache,
        rangedLookupKey
      );
      return;
    }

    const isEntryOversized =
      this._maxWgslCompileBytes > 0 &&
      typeof entry.code === 'string' &&
      entry.code.length > this._maxWgslCompileBytes;
    const shouldDefer =
      !isEntryOversized &&
      (this._lazyShaderCompile === true ||
        (this._lazyShaderCompile === 'pack' && entry.source === 'pack'));
    if (shouldDefer) {
      this._deferredShaderCount += 1;
      this.shaderModules.set(id, entry);
      if (rangedPackCache) {
        this._scheduleRangedShaderResolve(
          id,
          stage,
          resolved.lookupHashHex,
          shaderGen,
          rangedPackCache,
          rangedLookupKey
        );
      } else if (wasInFlightRanged || this._pendingRangedPipelines.size > 0) {
        this._onRangedShaderSettled(id, true);
      }
      return;
    }

    const allowDirectThrowOnEmpty =
      this.shaderFallbackRouter === null &&
      !this._captureValidationErrors &&
      this._validationErrorCallbacks.size === 0 &&
      trimmed.length === 0;
    const compiled = this._compileShaderModuleEntry(id, entry, allowDirectThrowOnEmpty);
    if (compiled) {
      this.shaderModules.set(id, compiled);
    }
    if (rangedPackCache) {
      this._scheduleRangedShaderResolve(
        id,
        stage,
        resolved.lookupHashHex,
        shaderGen,
        rangedPackCache,
        rangedLookupKey
      );
    } else if (wasInFlightRanged || this._pendingRangedPipelines.size > 0) {
      this._onRangedShaderSettled(id, Boolean(compiled));
    }
  }

  _handleCreateRenderPipeline(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const vsShaderId = view.getUint32(payloadOffset + 4, true);
    const fsShaderId = view.getUint32(payloadOffset + 8, true);
    const vertexStride = view.getUint32(payloadOffset + 12, true);
    const topologyCode = view.getUint32(payloadOffset + 16, true);
    // Reserved pipeline flags at payloadOffset + 20 for future blend/depth state.

    const pendingRanged = this._registerPendingRangedPipeline(
      id,
      'render',
      OPCODES.CREATE_RENDER_PIPELINE,
      fsShaderId !== 0 ? [vsShaderId, fsShaderId] : [vsShaderId],
      view,
      payloadOffset,
      24
    );
    if (pendingRanged.deferredOnly) {
      return;
    }

    const vsEntry = this.ensureShaderModule(vsShaderId);
    if (!vsEntry) {
      this._recordOpValidationError(
        'CreateRenderPipeline',
        `pipeline ${id}: unknown vertex shader module ${vsShaderId}`
      );
      return;
    }
    const fsEntry = fsShaderId !== 0 ? this.ensureShaderModule(fsShaderId) : null;
    if (fsShaderId !== 0 && !fsEntry) {
      this._recordOpValidationError(
        'CreateRenderPipeline',
        `pipeline ${id}: unknown fragment shader module ${fsShaderId}`
      );
      return;
    }

    const isFallback = Boolean(
      (vsEntry && vsEntry.source === 'builtin') || (fsEntry && fsEntry.source === 'builtin')
    );
    const usesFallbackShader = Boolean(
      (vsEntry && (vsEntry.source === 'builtin' || vsEntry.source === 'fallback')) ||
        (fsEntry && (fsEntry.source === 'builtin' || fsEntry.source === 'fallback'))
    );
    const isRealPack = Boolean(
      vsEntry && vsEntry.source === 'pack' && (fsShaderId === 0 || (fsEntry && fsEntry.source === 'pack'))
    );
    const expectsGroup0 = Boolean(
      wgslDeclaresGroup0(vsEntry && vsEntry.code) || wgslDeclaresGroup0(fsEntry && fsEntry.code)
    );
    const effectiveFallbackProfile =
      vsEntry && vsEntry.fallbackProfile && vsEntry.fallbackProfile !== FALLBACK_PROFILE.DEFAULT
        ? vsEntry.fallbackProfile
        : fsEntry && fsEntry.fallbackProfile && fsEntry.fallbackProfile !== FALLBACK_PROFILE.DEFAULT
          ? fsEntry.fallbackProfile
          : isFallback
            ? FALLBACK_PROFILE.DEFAULT
            : null;

    const vsRefl = this._resolveShaderReflection(vsShaderId, vsEntry, 'vertex');
    const fsRefl = fsEntry ? this._resolveShaderReflection(fsShaderId, fsEntry, 'fragment') : null;
    const matchedRefls = [];
    if (vsRefl) matchedRefls.push(vsRefl);
    if (fsRefl) matchedRefls.push(fsRefl);

    const reflected = this._buildReflectedPipelineLayout(matchedRefls, `render-pipeline-${id}`);
    if (reflected) {
      this._pipelineBindGroupLayouts.set(id, reflected.bindGroupLayouts);
      this._pipelineReflectionGroups.set(id, reflected.groupDescriptors || []);
    } else {
      this._pipelineBindGroupLayouts.delete(id);
      this._pipelineReflectionGroups.delete(id);
    }

    const topology = TOPOLOGY_NAMES[topologyCode] || 'triangle-list';
    let vertexBuffers;
    if (effectiveFallbackProfile === FALLBACK_PROFILE.FULLSCREEN_TRIANGLE_CBV && !vsRefl) {
      vertexBuffers = [];
    } else if (
      effectiveFallbackProfile === FALLBACK_PROFILE.SPLASH_RECT_24B &&
      vertexStride === TEXTURED_QUAD_24B_STRIDE &&
      !vsRefl
    ) {
      vertexBuffers = [
        {
          arrayStride: FALLBACK_VERTEX_LAYOUT_SPLASH_RECT_24B.arrayStride,
          stepMode: FALLBACK_VERTEX_LAYOUT_SPLASH_RECT_24B.stepMode,
          attributes: FALLBACK_VERTEX_LAYOUT_SPLASH_RECT_24B.attributes.map((a) => ({ ...a })),
        },
      ];
    } else if (
      effectiveFallbackProfile === FALLBACK_PROFILE.TEXTURED_QUAD_24B &&
      vertexStride === TEXTURED_QUAD_24B_STRIDE &&
      !vsRefl
    ) {
      vertexBuffers = [
        {
          arrayStride: FALLBACK_VERTEX_LAYOUT_24B.arrayStride,
          stepMode: FALLBACK_VERTEX_LAYOUT_24B.stepMode,
          attributes: FALLBACK_VERTEX_LAYOUT_24B.attributes.map((a) => ({ ...a })),
        },
      ];
    } else {
      vertexBuffers =
        vertexStride > 0
          ? [
              {
                arrayStride: vertexStride,
                attributes: [
                  {
                    shaderLocation: 0,
                    offset: 0,
                    format: vertexStride >= 12 ? 'float32x3' : vertexStride >= 8 ? 'float32x2' : 'float32',
                  },
                ],
              },
            ]
          : [];
    }

    if (
      vsRefl &&
      vertexStride > 0 &&
      Array.isArray(vsRefl.vertexInputs) &&
      vsRefl.vertexInputs.length > 0 &&
      this.reflectionRegistry &&
      typeof this.reflectionRegistry.buildVertexBufferLayouts === 'function'
    ) {
      try {
        const reflectedVbl = this.reflectionRegistry.buildVertexBufferLayouts(vsRefl, vertexStride);
        if (Array.isArray(reflectedVbl) && reflectedVbl.length > 0) {
          vertexBuffers = reflectedVbl;
        }
      } catch (_) {
        // Keep default single-attribute vertex layout when stride/format overrides don't match.
      }
    }

    const descriptor = {
      layout: reflected ? reflected.pipelineLayout : 'auto',
      vertex: {
        module: vsEntry.module,
        entryPoint: vsRefl && vsRefl.entryPoint ? vsRefl.entryPoint : 'vs_main',
        buffers: vertexBuffers,
      },
      primitive: {
        topology,
      },
    };

    if (fsEntry) {
      descriptor.fragment = {
        module: fsEntry.module,
        entryPoint: fsRefl && fsRefl.entryPoint ? fsRefl.entryPoint : 'fs_main',
        targets: [{ format: this._getRenderTargetFormat() }],
      };
    }

    const vsGen = this._shaderModuleGenerations.get(vsShaderId) || 0;
    const fsGen = fsShaderId !== 0 ? this._shaderModuleGenerations.get(fsShaderId) || 0 : 0;
    const rtFormat = fsEntry ? this._getRenderTargetFormat() : 'none';
    const renderFastKey = `v1r:${vsGen}:${fsGen}:${vertexStride}:${topology}:${rtFormat}:${reflected ? 'r' : 'a'}:${effectiveFallbackProfile || ''}`;

    let pipeline;
    const pipeStallStartMs = this._gpuStallWarnMs > 0 ? this._getNowMs() : 0;
    const pipeStallShaderHash =
      (vsEntry && (vsEntry.requestedHash || vsEntry.hashHex)) ||
      (fsEntry && (fsEntry.requestedHash || fsEntry.hashHex)) ||
      null;
    try {
      if (this.pipelineCache && typeof this.pipelineCache.getFastRenderPipeline === 'function') {
        pipeline = this.pipelineCache.getFastRenderPipeline(renderFastKey, descriptor);
      } else if (
        this.pipelineCache &&
        typeof this.pipelineCache.getOrCreateRenderPipeline === 'function'
      ) {
        pipeline = this.pipelineCache.getOrCreateRenderPipeline(descriptor, renderFastKey);
      } else {
        pipeline = this.device.createRenderPipeline(descriptor);
      }
    } catch (err) {
      if (this._gpuStallWarnMs > 0) {
        this._maybeWarnGpuStall('createRenderPipeline', pipeStallStartMs, pipeStallShaderHash);
      }
      if (!pendingRanged.pending) {
        this._recordOpValidationError('CreateRenderPipeline', err);
      }
      return;
    }
    if (this._gpuStallWarnMs > 0) {
      this._maybeWarnGpuStall('createRenderPipeline', pipeStallStartMs, pipeStallShaderHash);
    }
    if (!pipeline) {
      // W268 negative-cached (quarantined) or failed pipeline compile — skip cleanly.
      return;
    }
    this.pipelines.set(id, pipeline);
    this._pipelineMeta.set(id, {
      id,
      vsShaderId,
      fsShaderId,
      vertexStride,
      topology,
      isV2: false,
      hasReflectedLayout: Boolean(reflected),
      isRealPack,
      isFallback,
      usesFallbackShader,
      expectsGroup0,
      fallbackProfile: effectiveFallbackProfile,
    });
    this._invalidateSpecializedPipelinesFor(id);
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('renderPipeline', id, pipeline, {
        byteSize: 0,
        label: `renderPipeline:${id}`,
      });
    }
  }

  _normalizeTextureDescriptor(descriptor) {
    if (!descriptor || typeof descriptor !== 'object') {
      return descriptor;
    }
    const rawDimension = descriptor.dimension || '2d';
    const is1D = rawDimension === '1d';
    const rawUsage = typeof descriptor.usage === 'number' ? descriptor.usage >>> 0 : 0;
    const hasStorageBinding = (rawUsage & TEXTURE_USAGE.STORAGE_BINDING) !== 0;
    const rawFormat = typeof descriptor.format === 'string' ? descriptor.format : '';
    const stripStorageForSrgbFormat = hasStorageBinding && rawFormat.endsWith('-srgb');
    const rawViewFormats = Array.isArray(descriptor.viewFormats) ? descriptor.viewFormats : null;
    let hasIncompatibleSrgbViewFormat = false;
    if (hasStorageBinding && !stripStorageForSrgbFormat && rawViewFormats !== null) {
      for (let i = 0; i < rawViewFormats.length; i += 1) {
        const vf = rawViewFormats[i];
        if (typeof vf === 'string' && vf.endsWith('-srgb')) {
          hasIncompatibleSrgbViewFormat = true;
          break;
        }
      }
    }
    if (!is1D && !stripStorageForSrgbFormat && !hasIncompatibleSrgbViewFormat) {
      return descriptor;
    }
    const out = { ...descriptor };
    if (is1D) {
      const size = descriptor.size;
      let width = 1;
      if (Array.isArray(size)) {
        width = Math.max(1, Math.floor(Number(size[0]) || 1));
      } else if (size && typeof size === 'object') {
        width = Math.max(1, Math.floor(Number(size.width) || 1));
      }
      if (this._promote1DTo2D !== false) {
        this._promoted1DTo2DCount += 1;
        out.dimension = '2d';
      } else {
        out.dimension = '1d';
      }
      out.size = {
        width,
        height: 1,
        depthOrArrayLayers: 1,
      };
      out.mipLevelCount = 1;
    }
    if (stripStorageForSrgbFormat) {
      let sanitizedUsage = rawUsage & ~TEXTURE_USAGE.STORAGE_BINDING;
      if (sanitizedUsage === 0) {
        sanitizedUsage = TEXTURE_USAGE.TEXTURE_BINDING;
      }
      out.usage = sanitizedUsage;
    } else if (hasIncompatibleSrgbViewFormat) {
      const filtered = [];
      for (let i = 0; i < rawViewFormats.length; i += 1) {
        const vf = rawViewFormats[i];
        if (typeof vf !== 'string' || !vf.endsWith('-srgb')) {
          filtered.push(vf);
        }
      }
      if (filtered.length > 0) {
        out.viewFormats = filtered;
      } else {
        delete out.viewFormats;
      }
    }
    return out;
  }

  _createSafeTextureView(texture, viewDescriptor = null, textureId = null) {
    if (!texture || typeof texture.createView !== 'function') {
      return null;
    }
    if (!viewDescriptor || typeof viewDescriptor !== 'object') {
      return texture.createView();
    }
    const texDim =
      typeof texture.dimension === 'string' && texture.dimension
        ? texture.dimension
        : texture.desc && typeof texture.desc.dimension === 'string' && texture.desc.dimension
          ? texture.desc.dimension
          : null;
    const coerce1D = Boolean(
      texDim === '1d' && viewDescriptor.dimension && viewDescriptor.dimension !== '1d'
    );

    let rewrittenFormat = null;
    const reqFormat = viewDescriptor.format;
    if (typeof reqFormat === 'string' && reqFormat.endsWith('-srgb')) {
      const linearSibling = Object.prototype.hasOwnProperty.call(TEXTURE_SRGB_TO_LINEAR, reqFormat)
        ? TEXTURE_SRGB_TO_LINEAR[reqFormat]
        : null;
      if (linearSibling) {
        const meta =
          (textureId !== null && textureId !== undefined ? this._textureMeta.get(textureId) || null : null) ||
          (this._textureObjMeta ? this._textureObjMeta.get(texture) || null : null);
        let rsMeta = null;
        if (textureId !== null && textureId !== undefined) {
          const renderStateExt = this._getRenderStateExtension();
          if (
            renderStateExt &&
            renderStateExt.textureMeta &&
            typeof renderStateExt.textureMeta.get === 'function'
          ) {
            rsMeta = renderStateExt.textureMeta.get(textureId) || null;
          }
        }
        const texDesc = texture.desc && typeof texture.desc === 'object' ? texture.desc : null;
        const hasStorageBinding =
          (typeof texture.usage === 'number' &&
            (texture.usage & TEXTURE_USAGE.STORAGE_BINDING) !== 0) ||
          (meta !== null &&
            typeof meta.usage === 'number' &&
            (meta.usage & TEXTURE_USAGE.STORAGE_BINDING) !== 0) ||
          (rsMeta !== null &&
            typeof rsMeta.usage === 'number' &&
            (rsMeta.usage & TEXTURE_USAGE.STORAGE_BINDING) !== 0) ||
          (texDesc !== null &&
            typeof texDesc.usage === 'number' &&
            (texDesc.usage & TEXTURE_USAGE.STORAGE_BINDING) !== 0);

        let missingSrgbViewFormat = false;
        if (!hasStorageBinding) {
          const texFormat =
            (typeof texture.format === 'string' && texture.format) ||
            (rsMeta && typeof rsMeta.format === 'string' && rsMeta.format) ||
            (meta && typeof meta.format === 'string' && meta.format) ||
            (texDesc && typeof texDesc.format === 'string' && texDesc.format) ||
            null;
          if (texFormat && !texFormat.endsWith('-srgb')) {
            if (Array.isArray(texture.viewFormats)) {
              missingSrgbViewFormat = !texture.viewFormats.includes(reqFormat);
            } else if (texDesc && Array.isArray(texDesc.viewFormats)) {
              missingSrgbViewFormat = !texDesc.viewFormats.includes(reqFormat);
            } else if (rsMeta && 'viewFormats' in rsMeta) {
              missingSrgbViewFormat = !(
                Array.isArray(rsMeta.viewFormats) && rsMeta.viewFormats.includes(reqFormat)
              );
            } else if (meta && 'viewFormats' in meta) {
              missingSrgbViewFormat = !(
                Array.isArray(meta.viewFormats) && meta.viewFormats.includes(reqFormat)
              );
            } else if (texDesc && typeof texDesc.format === 'string' && !Array.isArray(texDesc.viewFormats)) {
              missingSrgbViewFormat = true;
            }
          }
        }
        if (hasStorageBinding || missingSrgbViewFormat) {
          rewrittenFormat = linearSibling;
        }
      }
    }

    if (!coerce1D && rewrittenFormat === null) {
      return texture.createView(viewDescriptor);
    }
    const safeDesc = { ...viewDescriptor };
    if (coerce1D) safeDesc.dimension = '1d';
    if (rewrittenFormat !== null) safeDesc.format = rewrittenFormat;
    return texture.createView(safeDesc);
  }

  _createTrackedTexture(id, descriptor, was1D = false) {
    const effectiveDescriptor = this._normalizeTextureDescriptor(descriptor);
    const promotedFrom1D = Boolean(was1D && effectiveDescriptor.dimension === '2d');

    this._destroyTextureIfPresent(id);
    const texture = this.device.createTexture(effectiveDescriptor);
    this.textures.set(id, texture);
    let textureView = null;
    if (texture && typeof texture.createView === 'function') {
      textureView = promotedFrom1D
        ? this._createSafeTextureView(texture, { dimension: '2d' }, id)
        : this._createSafeTextureView(texture, null, id);
      this.textureViews.set(id, textureView);
    }
    const size = effectiveDescriptor.size || {};
    const sampleCount = Math.max(1, Math.floor(Number(effectiveDescriptor.sampleCount) || 1));
    const byteSize = estimateTextureBytes(
      size.width || 1,
      size.height || 1,
      size.depthOrArrayLayers || 1,
      effectiveDescriptor.mipLevelCount || 1,
      effectiveDescriptor.format,
      effectiveDescriptor.dimension,
      sampleCount
    );
    this._recordTextureByteAccounting(id, texture, {
      byteSize,
      width: size.width || 1,
      height: size.height || 1,
      depthOrArrayLayers: size.depthOrArrayLayers || 1,
      mipLevelCount: effectiveDescriptor.mipLevelCount || 1,
      sampleCount,
      format: effectiveDescriptor.format || 'rgba8unorm',
      dimension: effectiveDescriptor.dimension || '2d',
      usage: typeof effectiveDescriptor.usage === 'number' ? effectiveDescriptor.usage : 0,
      viewFormats: Array.isArray(effectiveDescriptor.viewFormats)
        ? effectiveDescriptor.viewFormats
        : null,
    });
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('texture', id, texture, {
        byteSize,
        label: `texture:${id}`,
      });
      if (textureView) {
        this.resourceRegistry.register('textureView', id, textureView, {
          byteSize: 0,
          label: `textureView:${id}`,
        });
      }
    }
  }

  _recordTextureByteAccounting(id, texture, meta) {
    const safeBytes =
      meta && typeof meta.byteSize === 'number' && Number.isFinite(meta.byteSize) && meta.byteSize > 0
        ? Math.trunc(meta.byteSize)
        : 0;
    const prev = this._textureMeta.get(id);
    if (prev && prev.texture === texture) {
      return;
    }
    if (prev && typeof prev.byteSize === 'number' && prev.byteSize > 0) {
      this._liveTextureBytes = Math.max(0, this._liveTextureBytes - prev.byteSize);
    }
    const entry = {
      texture,
      byteSize: safeBytes,
      width: meta.width || 1,
      height: meta.height || 1,
      depthOrArrayLayers: meta.depthOrArrayLayers || 1,
      mipLevelCount: meta.mipLevelCount || 1,
      sampleCount: meta.sampleCount || 1,
      format: meta.format || 'rgba8unorm',
      dimension: meta.dimension || '2d',
      usage: typeof meta.usage === 'number' ? meta.usage : 0,
    };
    if (meta && 'viewFormats' in meta) {
      entry.viewFormats = Array.isArray(meta.viewFormats) ? meta.viewFormats : null;
    }
    this._textureMeta.set(id, entry);
    if (this._textureObjMeta && texture && typeof texture === 'object') {
      this._textureObjMeta.set(texture, entry);
    }
    this._liveTextureBytes += safeBytes;
    if (this._liveTextureBytes > this._peakTextureBytes) {
      this._peakTextureBytes = this._liveTextureBytes;
    }
    if (this._deferredResourceBytes !== null && texture && typeof texture === 'object') {
      this._deferredResourceBytes.set(texture, safeBytes);
    }
  }

  /**
   * #1077: Reconcile live/peak GPU texture byte accounting for textures created by packet ABI
   * extensions (`TexturePackets` / `RenderStatePackets`), which register directly into
   * `this.textures` without going through `_createTrackedTexture`. O(1) no-op when already
   * accounted for the same `GPUTexture` instance.
   */
  _reconcileTrackedTextureAfterExternalCreate(id, view, payloadOffset, isV2) {
    const texture = this.textures.get(id);
    if (!texture) {
      return;
    }
    const existing = this._textureMeta.get(id);
    if (existing && existing.texture === texture) {
      return;
    }
    const renderStateExt = this._getRenderStateExtension();
    const rsMeta =
      renderStateExt &&
      renderStateExt.textureMeta &&
      typeof renderStateExt.textureMeta.get === 'function'
        ? renderStateExt.textureMeta.get(id)
        : null;
    const texDesc = texture && typeof texture.desc === 'object' ? texture.desc : null;
    const texDescSize = texDesc && typeof texDesc.size === 'object' ? texDesc.size : null;

    let width =
      (rsMeta && rsMeta.width > 0 ? rsMeta.width : 0) ||
      (typeof texture.width === 'number' && texture.width > 0 ? texture.width : 0) ||
      (texDescSize && texDescSize.width > 0 ? texDescSize.width : 0);
    let height =
      (rsMeta && rsMeta.height > 0 ? rsMeta.height : 0) ||
      (typeof texture.height === 'number' && texture.height > 0 ? texture.height : 0) ||
      (texDescSize && texDescSize.height > 0 ? texDescSize.height : 0);
    let depthOrArrayLayers =
      (rsMeta && rsMeta.depthOrArrayLayers > 0 ? rsMeta.depthOrArrayLayers : 0) ||
      (typeof texture.depthOrArrayLayers === 'number' && texture.depthOrArrayLayers > 0
        ? texture.depthOrArrayLayers
        : 0) ||
      (texDescSize && texDescSize.depthOrArrayLayers > 0 ? texDescSize.depthOrArrayLayers : 0);
    let mipLevelCount =
      (typeof texture.mipLevelCount === 'number' && texture.mipLevelCount > 0
        ? texture.mipLevelCount
        : 0) ||
      (texDesc && texDesc.mipLevelCount > 0 ? texDesc.mipLevelCount : 0);
    let sampleCount =
      (rsMeta && rsMeta.sampleCount > 0 ? rsMeta.sampleCount : 0) ||
      (typeof texture.sampleCount === 'number' && texture.sampleCount > 0
        ? texture.sampleCount
        : 0) ||
      (texDesc && texDesc.sampleCount > 0 ? texDesc.sampleCount : 0);
    let format =
      (rsMeta && typeof rsMeta.format === 'string' && rsMeta.format.length > 0
        ? rsMeta.format
        : null) ||
      (typeof texture.format === 'string' && texture.format.length > 0 ? texture.format : null) ||
      (texDesc && typeof texDesc.format === 'string' && texDesc.format.length > 0
        ? texDesc.format
        : null);
    let dimension =
      (rsMeta && typeof rsMeta.dimension === 'string' && rsMeta.dimension.length > 0
        ? rsMeta.dimension
        : null) ||
      (typeof texture.dimension === 'string' && texture.dimension.length > 0
        ? texture.dimension
        : null) ||
      (texDesc && typeof texDesc.dimension === 'string' && texDesc.dimension.length > 0
        ? texDesc.dimension
        : null);
    let usage =
      rsMeta && typeof rsMeta.usage === 'number'
        ? rsMeta.usage
        : typeof texture.usage === 'number'
          ? texture.usage
          : texDesc && typeof texDesc.usage === 'number'
            ? texDesc.usage
            : null;

    if (view && typeof payloadOffset === 'number') {
      if (isV2 && payloadOffset + 32 <= view.byteLength) {
        if (!width) width = Math.max(1, view.getUint32(payloadOffset + 4, true));
        if (!height) height = Math.max(1, view.getUint32(payloadOffset + 8, true));
        if (!depthOrArrayLayers) {
          depthOrArrayLayers = Math.max(1, view.getUint32(payloadOffset + 12, true));
        }
        if (!mipLevelCount) mipLevelCount = Math.max(1, view.getUint32(payloadOffset + 16, true));
        if (!sampleCount) sampleCount = Math.max(1, view.getUint32(payloadOffset + 20, true));
        if (usage === null) usage = view.getUint32(payloadOffset + 24, true);
        if (!dimension) {
          dimension = TEXTURE_DIMENSION_NAMES[view.getUint8(payloadOffset + 28)] || '2d';
        }
        if (!format) {
          const fmtCode = view.getUint8(payloadOffset + 29);
          format = (fmtCode > 0 && TEXTURE_FORMAT_NAMES_V2[fmtCode]) || 'rgba8unorm';
        }
      } else if (!isV2 && payloadOffset + 32 <= view.byteLength) {
        if (!width) width = Math.max(1, view.getUint32(payloadOffset + 4, true));
        if (!height) height = Math.max(1, view.getUint32(payloadOffset + 8, true));
        if (!depthOrArrayLayers) {
          depthOrArrayLayers = Math.max(1, view.getUint32(payloadOffset + 12, true));
        }
        if (!mipLevelCount) mipLevelCount = Math.max(1, view.getUint32(payloadOffset + 16, true));
        if (!format) {
          format = TEXTURE_FORMAT_NAMES[view.getUint32(payloadOffset + 20, true)] || 'rgba8unorm';
        }
        if (usage === null) usage = view.getUint32(payloadOffset + 24, true);
        if (!dimension) {
          dimension = TEXTURE_DIMENSION_NAMES[view.getUint32(payloadOffset + 28, true)] || '2d';
        }
      }
    }

    width = Math.max(1, width || 1);
    height = Math.max(1, height || 1);
    depthOrArrayLayers = Math.max(1, depthOrArrayLayers || 1);
    mipLevelCount = Math.max(1, mipLevelCount || 1);
    sampleCount = Math.max(1, sampleCount || 1);
    format = format || 'rgba8unorm';
    dimension = dimension || '2d';
    usage = typeof usage === 'number' ? usage : 0;

    const byteSize = estimateTextureBytes(
      width,
      height,
      depthOrArrayLayers,
      mipLevelCount,
      format,
      dimension,
      sampleCount
    );
    const accountedMeta = {
      byteSize,
      width,
      height,
      depthOrArrayLayers,
      mipLevelCount,
      sampleCount,
      format,
      dimension,
      usage,
    };
    if (rsMeta && 'viewFormats' in rsMeta) {
      accountedMeta.viewFormats = Array.isArray(rsMeta.viewFormats) ? rsMeta.viewFormats : null;
    } else if (Array.isArray(texture.viewFormats)) {
      accountedMeta.viewFormats = texture.viewFormats;
    } else if (texDesc && Array.isArray(texDesc.viewFormats)) {
      accountedMeta.viewFormats = texDesc.viewFormats;
    }
    this._recordTextureByteAccounting(id, texture, accountedMeta);
  }

  _handleCreateTexture(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const width = view.getUint32(payloadOffset + 4, true);
    const height = view.getUint32(payloadOffset + 8, true);
    const depthOrArrayLayers = Math.max(1, view.getUint32(payloadOffset + 12, true));
    const mipLevelCount = Math.max(1, view.getUint32(payloadOffset + 16, true));
    const formatCode = view.getUint32(payloadOffset + 20, true);
    const usage = view.getUint32(payloadOffset + 24, true);
    const dimensionCode = view.getUint32(payloadOffset + 28, true);

    const format = TEXTURE_FORMAT_NAMES[formatCode] || 'rgba8unorm';
    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode] || '2d';

    this._createTrackedTexture(
      id,
      {
        size: {
          width,
          height,
          depthOrArrayLayers,
        },
        mipLevelCount,
        format,
        usage,
        dimension,
      },
      dimension === '1d'
    );
  }

  _handleCreateTexture2(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const width = Math.max(1, view.getUint32(payloadOffset + 4, true));
    const height = Math.max(1, view.getUint32(payloadOffset + 8, true));
    const depthOrArrayLayers = Math.max(1, view.getUint32(payloadOffset + 12, true));
    const mipLevelCount = Math.max(1, view.getUint32(payloadOffset + 16, true));
    const sampleCount = Math.max(1, view.getUint32(payloadOffset + 20, true));
    const usage = view.getUint32(payloadOffset + 24, true);
    const dimensionCode = view.getUint8(payloadOffset + 28);
    const formatCode = view.getUint8(payloadOffset + 29);
    const viewFormatCount = Math.min(8, view.getUint8(payloadOffset + 30));

    const format = (formatCode > 0 && TEXTURE_FORMAT_NAMES_V2[formatCode]) || 'rgba8unorm';
    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode] || '2d';
    const descriptor = {
      size: {
        width,
        height,
        depthOrArrayLayers,
      },
      mipLevelCount,
      sampleCount,
      format,
      usage,
      dimension,
    };
    if (viewFormatCount > 0) {
      const viewFormats = [];
      for (let i = 0; i < viewFormatCount; i += 1) {
        const code = view.getUint8(payloadOffset + 32 + i);
        const name = code > 0 ? TEXTURE_FORMAT_NAMES_V2[code] : undefined;
        if (name) {
          viewFormats.push(name);
        }
      }
      if (viewFormats.length > 0) {
        descriptor.viewFormats = viewFormats;
      }
    }
    this._createTrackedTexture(id, descriptor, dimension === '1d');
  }

  _handleWriteTexture(view, payloadOffset, payloadBytes) {
    const textureId = view.getUint32(payloadOffset, true);
    const mipLevel = view.getUint32(payloadOffset + 4, true);
    const originX = view.getUint32(payloadOffset + 8, true);
    const originY = view.getUint32(payloadOffset + 12, true);
    const originZ = view.getUint32(payloadOffset + 16, true);
    const width = view.getUint32(payloadOffset + 20, true);
    const height = view.getUint32(payloadOffset + 24, true);
    const depthOrArrayLayers = Math.max(1, view.getUint32(payloadOffset + 28, true));
    const bytesPerRow = view.getUint32(payloadOffset + 32, true);
    const rowsPerImage = view.getUint32(payloadOffset + 36, true);
    const dataLen = view.getUint32(payloadOffset + 40, true);
    if (48 + dataLen > payloadBytes) return;

    const texture = this.textures.get(textureId);
    if (!texture) return;

    const data = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 48, dataLen);

    const origin = this._wtOriginScratch;
    origin.x = originX;
    origin.y = originY;
    origin.z = originZ;

    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.stageTextureUpload === 'function' &&
      width > 0 &&
      height > 0 &&
      dataLen > 0
    ) {
      const effectiveRowsPerImage = rowsPerImage > 0 ? rowsPerImage : height;
      const totalRows = effectiveRowsPerImage * depthOrArrayLayers;
      const canStage =
        bytesPerRow > 0
          ? dataLen >= bytesPerRow * totalRows
          : totalRows > 0 && dataLen % totalRows === 0;
      if (canStage) {
        const encoder = this._ensurePendingEncoder();
        if (encoder && typeof encoder.copyBufferToTexture === 'function') {
          const stageLayout = this._wtStageLayoutScratch;
          stageLayout.width = width;
          stageLayout.height = height;
          stageLayout.depthOrArrayLayers = depthOrArrayLayers;
          stageLayout.mipLevel = mipLevel;
          stageLayout.bytesPerRow = bytesPerRow > 0 ? bytesPerRow : undefined;
          stageLayout.rowsPerImage = rowsPerImage > 0 ? rowsPerImage : undefined;
          try {
            this.stagingPool.stageTextureUpload(encoder, texture, data, stageLayout);
            this._hasPendingUploadCommands = true;
            return;
          } catch (err) {
            if (err instanceof RangeError) {
              this._flushPendingUploads(true);
            } else {
              throw err;
            }
          }
        }
        if (!this._hasPendingUploadCommands) {
          this._pendingEncoder = null;
        }
      }
    }

    let dataLayout;
    if (bytesPerRow > 0) {
      if (rowsPerImage > 0) {
        dataLayout = this._wtLayoutFullScratch;
        dataLayout.bytesPerRow = bytesPerRow;
        dataLayout.rowsPerImage = rowsPerImage;
      } else {
        dataLayout = this._wtLayoutBprScratch;
        dataLayout.bytesPerRow = bytesPerRow;
      }
    } else if (rowsPerImage > 0) {
      dataLayout = this._wtLayoutRpiScratch;
      dataLayout.rowsPerImage = rowsPerImage;
    } else {
      dataLayout = this._wtLayoutOffsetOnlyScratch;
    }

    const dst = this._wtDstScratch;
    dst.texture = texture;
    dst.mipLevel = mipLevel;

    const size = this._wtSizeScratch;
    size.width = width;
    size.height = height;
    size.depthOrArrayLayers = depthOrArrayLayers;

    const isShared =
      typeof SharedArrayBuffer !== 'undefined' && data.buffer instanceof SharedArrayBuffer;
    let uploadData = data;
    if (isShared && data.byteLength > 0) {
      const scratch = this._acquireDirectWriteScratch(data.byteLength);
      const copy = scratch
        ? scratch.byteLength === data.byteLength
          ? scratch
          : scratch.subarray(0, data.byteLength)
        : new Uint8Array(data.byteLength);
      copy.set(data, 0);
      uploadData = copy;
    }

    try {
      let chunked = false;
      if (
        uploadData.byteLength > MAX_DIRECT_WRITE_TEXTURE_CHUNK_BYTES &&
        bytesPerRow > 0 &&
        depthOrArrayLayers === 1 &&
        height > 1
      ) {
        const texMeta = this._textureMeta.get(textureId);
        const fmtKey =
          texMeta && typeof texMeta.format === 'string'
            ? texMeta.format.trim().toLowerCase()
            : '';
        const blockH = TEXTURE_BLOCK_FORMAT_BYTES[fmtKey] > 0 ? 4 : 1;
        if (originY % blockH === 0) {
          const maxBlockRows = Math.max(
            1,
            Math.floor(MAX_DIRECT_WRITE_TEXTURE_CHUNK_BYTES / bytesPerRow)
          );
          const chunkH = maxBlockRows * blockH;
          if (chunkH < height) {
            chunked = true;
            for (let yOff = 0; yOff < height; yOff += chunkH) {
              const sliceH = Math.min(chunkH, height - yOff);
              const sliceBlockRows = Math.ceil(sliceH / blockH);
              const byteStart = (yOff / blockH) * bytesPerRow;
              const byteEnd = Math.min(
                uploadData.byteLength,
                byteStart + sliceBlockRows * bytesPerRow
              );
              if (byteStart >= byteEnd) {
                break;
              }
              origin.y = originY + yOff;
              size.height = sliceH;
              if (rowsPerImage > 0) {
                dataLayout.rowsPerImage = Math.min(rowsPerImage, sliceBlockRows);
              }
              this.device.queue.writeTexture(
                dst,
                uploadData.subarray(byteStart, byteEnd),
                dataLayout,
                size
              );
            }
          }
        }
      }
      if (!chunked) {
        this.device.queue.writeTexture(dst, uploadData, dataLayout, size);
      }
    } finally {
      dst.texture = null;
    }
  }

  _acquireClearZeroBuffer(requiredBytes, minChunkBytes = CLEAR_TEXTURE_ROW_ALIGN) {
    if (!this.device || typeof this.device.createBuffer !== 'function') {
      return null;
    }
    const maxCap = Math.max(CLEAR_TEXTURE_ZERO_BUFFER_BYTES, minChunkBytes);
    const alignedReq = Math.max(
      CLEAR_TEXTURE_ZERO_BUFFER_MIN_BYTES,
      Math.ceil(Math.min(Math.max(CLEAR_TEXTURE_ROW_ALIGN, requiredBytes), maxCap) / CLEAR_TEXTURE_ROW_ALIGN) *
        CLEAR_TEXTURE_ROW_ALIGN
    );
    if (this._clearZeroBuffer && this._clearZeroBufferBytes >= alignedReq) {
      return this._clearZeroBuffer;
    }
    const prev = this._clearZeroBuffer;
    const prevBytes = this._clearZeroBufferBytes;
    const nextSize = prev
      ? Math.min(maxCap, Math.max(alignedReq, prevBytes * 2))
      : Math.min(maxCap, alignedReq);
    const usage =
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_SRC === 'number'
        ? GPUBufferUsage.COPY_SRC
        : BUFFER_USAGE.COPY_SRC;
    const buffer = this.device.createBuffer({
      size: nextSize,
      usage,
    });
    this._clearZeroBuffer = buffer;
    this._clearZeroBufferBytes = nextSize;
    if (prev) {
      this._retireOrDestroyGpuResource(prev, prevBytes);
    }
    return buffer;
  }

  _acquireClearZeroScratch(requiredBytes, minChunkBytes = CLEAR_TEXTURE_ROW_ALIGN) {
    const maxCap = Math.max(CLEAR_TEXTURE_ZERO_SCRATCH_MAX_BYTES, minChunkBytes);
    const targetBytes = Math.min(
      maxCap,
      Math.max(CLEAR_TEXTURE_ROW_ALIGN, Math.ceil(requiredBytes / CLEAR_TEXTURE_ROW_ALIGN) * CLEAR_TEXTURE_ROW_ALIGN)
    );
    if (this._clearZeroScratch && this._clearZeroScratch.byteLength >= targetBytes) {
      return this._clearZeroScratch;
    }
    const nextSize = this._clearZeroScratch
      ? Math.min(maxCap, Math.max(targetBytes, this._clearZeroScratch.byteLength * 2))
      : targetBytes;
    this._clearZeroScratch = new Uint8Array(nextSize);
    return this._clearZeroScratch;
  }

  _resolveClearTextureInfo(textureId, extMeta = null) {
    const texture = this.textures.get(textureId);
    if (!texture || texture.destroyed === true) {
      return null;
    }
    const renderStateExt = this._getRenderStateExtension();
    const rsMeta =
      extMeta ||
      (renderStateExt &&
      renderStateExt.textureMeta &&
      typeof renderStateExt.textureMeta.get === 'function'
        ? renderStateExt.textureMeta.get(textureId) || null
        : null);
    const meta = this._textureMeta.get(textureId) || null;
    const desc = texture && typeof texture.desc === 'object' ? texture.desc : null;
    const descSize = desc && typeof desc.size === 'object' ? desc.size : null;

    const dimension =
      (typeof texture.dimension === 'string' && texture.dimension.length > 0 ? texture.dimension : null) ||
      (rsMeta && typeof rsMeta.dimension === 'string' && rsMeta.dimension.length > 0 ? rsMeta.dimension : null) ||
      (meta && typeof meta.dimension === 'string' && meta.dimension.length > 0 ? meta.dimension : null) ||
      (desc && typeof desc.dimension === 'string' && desc.dimension.length > 0 ? desc.dimension : null) ||
      '2d';
    const is1D = dimension === '1d';

    const width = Math.max(
      1,
      Math.floor(
        Number(
          (typeof texture.width === 'number' && texture.width > 0 ? texture.width : 0) ||
            (rsMeta && rsMeta.width > 0 ? rsMeta.width : 0) ||
            (meta && meta.width > 0 ? meta.width : 0) ||
            (descSize && descSize.width > 0 ? descSize.width : 0) ||
            (Array.isArray(descSize) && descSize[0] > 0 ? descSize[0] : 0) ||
            1
        )
      )
    );
    const height = is1D
      ? 1
      : Math.max(
          1,
          Math.floor(
            Number(
              (typeof texture.height === 'number' && texture.height > 0 ? texture.height : 0) ||
                (rsMeta && rsMeta.height > 0 ? rsMeta.height : 0) ||
                (meta && meta.height > 0 ? meta.height : 0) ||
                (descSize && descSize.height > 0 ? descSize.height : 0) ||
                (Array.isArray(descSize) && descSize[1] > 0 ? descSize[1] : 0) ||
                1
            )
          )
        );
    const depthOrArrayLayers = is1D
      ? 1
      : Math.max(
          1,
          Math.floor(
            Number(
              (typeof texture.depthOrArrayLayers === 'number' && texture.depthOrArrayLayers > 0
                ? texture.depthOrArrayLayers
                : 0) ||
                (rsMeta && rsMeta.depthOrArrayLayers > 0 ? rsMeta.depthOrArrayLayers : 0) ||
                (meta && meta.depthOrArrayLayers > 0 ? meta.depthOrArrayLayers : 0) ||
                (descSize && descSize.depthOrArrayLayers > 0 ? descSize.depthOrArrayLayers : 0) ||
                (Array.isArray(descSize) && descSize[2] > 0 ? descSize[2] : 0) ||
                1
            )
          )
        );
    const mipLevelCount = is1D
      ? 1
      : Math.max(
          1,
          Math.floor(
            Number(
              (typeof texture.mipLevelCount === 'number' && texture.mipLevelCount > 0
                ? texture.mipLevelCount
                : 0) ||
                (rsMeta && rsMeta.mipLevelCount > 0 ? rsMeta.mipLevelCount : 0) ||
                (meta && meta.mipLevelCount > 0 ? meta.mipLevelCount : 0) ||
                (desc && desc.mipLevelCount > 0 ? desc.mipLevelCount : 0) ||
                1
            )
          )
        );
    const sampleCount = Math.max(
      1,
      Math.floor(
        Number(
          (typeof texture.sampleCount === 'number' && texture.sampleCount > 0
            ? texture.sampleCount
            : 0) ||
            (rsMeta && rsMeta.sampleCount > 0 ? rsMeta.sampleCount : 0) ||
            (meta && meta.sampleCount > 0 ? meta.sampleCount : 0) ||
            (desc && desc.sampleCount > 0 ? desc.sampleCount : 0) ||
            1
        )
      )
    );
    const rawFormat =
      (rsMeta && typeof rsMeta.format === 'string' && rsMeta.format.length > 0 ? rsMeta.format : null) ||
      (typeof texture.format === 'string' && texture.format.length > 0 ? texture.format : null) ||
      (meta && typeof meta.format === 'string' && meta.format.length > 0 ? meta.format : null) ||
      (desc && typeof desc.format === 'string' && desc.format.length > 0 ? desc.format : null) ||
      'rgba8unorm';
    const format = rawFormat.trim().toLowerCase();
    const usage =
      typeof texture.usage === 'number'
        ? texture.usage
        : rsMeta && typeof rsMeta.usage === 'number'
          ? rsMeta.usage
          : meta && typeof meta.usage === 'number'
            ? meta.usage
            : desc && typeof desc.usage === 'number'
              ? desc.usage
              : 0;

    return {
      texture,
      width,
      height,
      depthOrArrayLayers,
      mipLevelCount,
      sampleCount,
      format,
      dimension,
      usage,
    };
  }

  /**
   * ClearTexture (opcode 63, kGpuOpcodeClearTexture): zero every mip level and layer/slice
   * of `textureId` so recycled pooled textures read identically to freshly created ones.
   */
  _clearTextureById(textureId, extMeta = null) {
    const info = this._resolveClearTextureInfo(textureId, extMeta);
    if (!info) {
      this._clearTextureStats.missing += 1;
      return { status: 'missing' };
    }
    const {
      texture,
      width,
      height,
      depthOrArrayLayers,
      mipLevelCount,
      sampleCount,
      format,
      dimension,
      usage,
    } = info;

    if (CLEAR_TEXTURE_DEPTH_STENCIL_FORMATS.has(format)) {
      this._clearTextureStats.invalid += 1;
      return {
        status: 'invalid',
        error: `ClearTexture ${textureId}: depth/stencil format '${format}' is not supported`,
      };
    }
    if (sampleCount > 1) {
      this._clearTextureStats.invalid += 1;
      return {
        status: 'invalid',
        error: `ClearTexture ${textureId}: multisampled texture (sampleCount=${sampleCount}) is not supported`,
      };
    }

    const bcBlockBytes = TEXTURE_BLOCK_FORMAT_BYTES[format] || 0;
    const isCompressed = bcBlockBytes > 0;
    const blockW = isCompressed ? 4 : 1;
    const blockH = isCompressed ? 4 : 1;
    const blockBytes = isCompressed ? bcBlockBytes : TEXTURE_FORMAT_BPP[format] || 0;
    if (blockBytes <= 0) {
      this._clearTextureStats.invalid += 1;
      return {
        status: 'invalid',
        error: `ClearTexture ${textureId}: unsupported texture format '${format}'`,
      };
    }

    const hasRenderAttachment = (usage & TEXTURE_USAGE.RENDER_ATTACHMENT) !== 0;
    const hasCopyDst = (usage & TEXTURE_USAGE.COPY_DST) !== 0;
    const canRenderPassClear =
      hasRenderAttachment &&
      !isCompressed &&
      dimension === '2d' &&
      !CLEAR_TEXTURE_NON_RENDERABLE_FORMATS.has(format) &&
      typeof texture.createView === 'function' &&
      this.device &&
      typeof this.device.createCommandEncoder === 'function' &&
      this.device.queue &&
      typeof this.device.queue.submit === 'function';

    if (!canRenderPassClear && !hasCopyDst) {
      this._clearTextureStats.invalid += 1;
      return {
        status: 'invalid',
        error: `ClearTexture ${textureId}: not a single-sample color COPY_DST or RENDER_ATTACHMENT texture`,
      };
    }

    // Flush any earlier staged texture/buffer uploads before clearing so earlier writes are wiped
    // and subsequent writes land cleanly on top of the zeroed texture.
    if (this._pendingEncoder && this._hasPendingUploadCommands) {
      this._flushPendingUploads(false);
    }

    if (canRenderPassClear) {
      try {
        const encoder = this.device.createCommandEncoder();
        if (
          encoder &&
          typeof encoder.beginRenderPass === 'function' &&
          typeof encoder.finish === 'function'
        ) {
          for (let mip = 0; mip < mipLevelCount; mip += 1) {
            for (let layer = 0; layer < depthOrArrayLayers; layer += 1) {
              const view = texture.createView({
                dimension: '2d',
                baseMipLevel: mip,
                mipLevelCount: 1,
                baseArrayLayer: layer,
                arrayLayerCount: 1,
              });
              const pass = encoder.beginRenderPass({
                colorAttachments: [
                  {
                    view,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: CLEAR_TEXTURE_ZERO_COLOR,
                  },
                ],
              });
              pass.end();
            }
          }
          this.device.queue.submit([encoder.finish()]);
          this._clearTextureStats.cleared += 1;
          this._clearTextureStats.renderPassClears += 1;
          this._clearTextureStats.copies += mipLevelCount * depthOrArrayLayers;
          return {
            status: 'cleared',
            mode: 'renderPass',
            subresources: mipLevelCount * depthOrArrayLayers,
          };
        }
      } catch (err) {
        if (!hasCopyDst) {
          this._clearTextureStats.invalid += 1;
          return {
            status: 'invalid',
            error: `ClearTexture ${textureId}: ${String(err && err.message ? err.message : err)}`,
          };
        }
      }
    }

    const plan = planClearTextureCopies(
      {
        width,
        height,
        depthOrArrayLayers,
        mipLevelCount,
        dimension,
        blockW,
        blockH,
        blockBytes,
      },
      CLEAR_TEXTURE_ZERO_BUFFER_BYTES
    );
    if (plan.copies.length === 0) {
      this._clearTextureStats.cleared += 1;
      return { status: 'cleared', mode: 'noop', copies: 0 };
    }

    // Prefer GPU-side copyBufferToTexture from a reusable zero-initialized GPUBuffer (no CPU upload).
    if (
      this.device &&
      typeof this.device.createBuffer === 'function' &&
      typeof this.device.createCommandEncoder === 'function' &&
      this.device.queue &&
      typeof this.device.queue.submit === 'function'
    ) {
      let encoder = null;
      try {
        encoder = this.device.createCommandEncoder();
      } catch (_) {
        encoder = null;
      }
      if (
        encoder &&
        typeof encoder.copyBufferToTexture === 'function' &&
        typeof encoder.finish === 'function'
      ) {
        try {
          const zeroBuffer = this._acquireClearZeroBuffer(plan.maxRequiredBytes, plan.widestRow);
          if (zeroBuffer) {
            for (let i = 0; i < plan.copies.length; i += 1) {
              const c = plan.copies[i];
              encoder.copyBufferToTexture(
                {
                  buffer: zeroBuffer,
                  offset: 0,
                  bytesPerRow: c.bytesPerRow,
                  rowsPerImage: c.rowsPerImage,
                },
                {
                  texture,
                  mipLevel: c.mip,
                  origin: { x: 0, y: c.y, z: c.z },
                  aspect: 'all',
                },
                {
                  width: c.width,
                  height: c.height,
                  depthOrArrayLayers: c.depth,
                }
              );
            }
            this.device.queue.submit([encoder.finish()]);
            this._clearTextureStats.cleared += 1;
            this._clearTextureStats.copyClears += 1;
            this._clearTextureStats.copies += plan.copies.length;
            return { status: 'cleared', mode: 'copyBufferToTexture', copies: plan.copies.length };
          }
        } catch (err) {
          if (!this.device.queue || typeof this.device.queue.writeTexture !== 'function') {
            this._clearTextureStats.invalid += 1;
            return {
              status: 'invalid',
              error: `ClearTexture ${textureId}: ${String(err && err.message ? err.message : err)}`,
            };
          }
        }
      }
    }

    // Fallback when copyBufferToTexture is unavailable: writeTexture with a bounded reusable zero Uint8Array.
    if (this.device && this.device.queue && typeof this.device.queue.writeTexture === 'function') {
      try {
        const zeroScratch = this._acquireClearZeroScratch(plan.maxRequiredBytes, plan.widestRow);
        for (let i = 0; i < plan.copies.length; i += 1) {
          const c = plan.copies[i];
          const data =
            zeroScratch.byteLength === c.copyBytes
              ? zeroScratch
              : zeroScratch.subarray(0, c.copyBytes);
          this.device.queue.writeTexture(
            {
              texture,
              mipLevel: c.mip,
              origin: { x: 0, y: c.y, z: c.z },
              aspect: 'all',
            },
            data,
            {
              offset: 0,
              bytesPerRow: c.bytesPerRow,
              rowsPerImage: c.rowsPerImage,
            },
            {
              width: c.width,
              height: c.height,
              depthOrArrayLayers: c.depth,
            }
          );
        }
        this._clearTextureStats.cleared += 1;
        this._clearTextureStats.writeTextureClears += 1;
        this._clearTextureStats.copies += plan.copies.length;
        return { status: 'cleared', mode: 'writeTexture', copies: plan.copies.length };
      } catch (err) {
        this._clearTextureStats.invalid += 1;
        return {
          status: 'invalid',
          error: `ClearTexture ${textureId}: ${String(err && err.message ? err.message : err)}`,
        };
      }
    }

    this._clearTextureStats.invalid += 1;
    return {
      status: 'invalid',
      error: `ClearTexture ${textureId}: device lacks copyBufferToTexture and queue.writeTexture`,
    };
  }

  _handleClearTexture(view, payloadOffset) {
    const textureId = view.getUint32(payloadOffset, true);
    const result = this._clearTextureById(textureId);
    if (result.status === 'invalid' && result.error) {
      this._recordOpValidationError('ClearTexture', result.error);
    }
  }

  _handleCreateSampler(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const addressModeU = ADDRESS_MODE_NAMES[view.getUint32(payloadOffset + 4, true)] || 'clamp-to-edge';
    const addressModeV = ADDRESS_MODE_NAMES[view.getUint32(payloadOffset + 8, true)] || 'clamp-to-edge';
    const addressModeW = ADDRESS_MODE_NAMES[view.getUint32(payloadOffset + 12, true)] || 'clamp-to-edge';
    const magFilter = FILTER_MODE_NAMES[view.getUint32(payloadOffset + 16, true)] || 'nearest';
    const minFilter = FILTER_MODE_NAMES[view.getUint32(payloadOffset + 20, true)] || 'nearest';
    const mipmapFilter = FILTER_MODE_NAMES[view.getUint32(payloadOffset + 24, true)] || 'nearest';
    const maxAnisotropy = Math.max(1, view.getUint32(payloadOffset + 28, true));

    const sampler = this.device.createSampler({
      addressModeU,
      addressModeV,
      addressModeW,
      magFilter,
      minFilter,
      mipmapFilter,
      maxAnisotropy,
    });
    this.samplers.set(id, sampler);
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('sampler', id, sampler, {
        byteSize: 0,
        label: `sampler:${id}`,
      });
    }
  }

  _handleCreateComputePipeline(view, payloadOffset) {
    const built = this._buildComputePipelineDescriptorV1(view, payloadOffset);
    if (!built) return;
    const csGen = this._shaderModuleGenerations.get(built.csShaderId) || 0;
    const computeFastKey = `v1c:${csGen}:${built.reflected ? 'r' : 'a'}`;
    let pipeline;
    try {
      if (this.pipelineCache && typeof this.pipelineCache.getFastComputePipeline === 'function') {
        pipeline = this.pipelineCache.getFastComputePipeline(computeFastKey, built.descriptor);
      } else if (
        this.pipelineCache &&
        typeof this.pipelineCache.getOrCreateComputePipeline === 'function'
      ) {
        pipeline = this.pipelineCache.getOrCreateComputePipeline(built.descriptor, computeFastKey);
      } else {
        pipeline = this.device.createComputePipeline(built.descriptor);
      }
    } catch (err) {
      if (!this._pendingRangedPipelines.has(built.id)) {
        this._recordOpValidationError('CreateComputePipeline', err);
      }
      return;
    }
    if (!pipeline) {
      // W268 negative-cached (quarantined) or failed compute pipeline compile — skip cleanly.
      return;
    }
    this._registerComputePipelineV1(built.id, pipeline);
  }

  /**
   * Decode a v1 CreateComputePipeline payload into a descriptor (reflected explicit layout when
   * available, else 'auto'). Records the reflected bind group layouts for getBindGroupLayout().
   *
   * @returns {{ id: number, csShaderId: number, reflected: boolean, descriptor: GPUComputePipelineDescriptor } | null}
   */
  _buildComputePipelineDescriptorV1(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const csShaderId = view.getUint32(payloadOffset + 4, true);
    // Reserved flags at payloadOffset + 8 and reserved at payloadOffset + 12.

    const csEntry = this.ensureShaderModule(csShaderId);
    if (!csEntry) return null;

    const csRefl = this._resolveShaderReflection(csShaderId, csEntry, 'compute');
    const reflected = csRefl
      ? this._buildReflectedPipelineLayout([csRefl], `compute-pipeline-${id}`)
      : null;
    if (reflected) {
      this._computePipelineBindGroupLayouts.set(id, reflected.bindGroupLayouts);
      this._computePipelineReflectionGroups.set(id, reflected.groupDescriptors || []);
    } else {
      this._computePipelineBindGroupLayouts.delete(id);
      this._computePipelineReflectionGroups.delete(id);
    }

    return {
      id,
      csShaderId,
      reflected: Boolean(reflected),
      descriptor: {
        layout: reflected ? reflected.pipelineLayout : 'auto',
        compute: {
          module: csEntry.module,
          entryPoint: csRefl && csRefl.entryPoint ? csRefl.entryPoint : 'cs_main',
        },
      },
    };
  }

  _registerComputePipelineV1(id, pipeline) {
    this.computePipelines.set(id, pipeline);
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('computePipeline', id, pipeline, {
        byteSize: 0,
        label: `computePipeline:${id}`,
      });
    }
  }

  _handleCreateBindGroup(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const pipelineId = view.getUint32(payloadOffset + 4, true);
    const groupIndex = view.getUint32(payloadOffset + 8, true);
    const entryCount = view.getUint32(payloadOffset + 12, true);
    if (16 + entryCount * 32 > payloadBytes) {
      this._recordOpValidationError(
        'CreateBindGroup',
        `bind group ${id}: truncated payload (${payloadBytes}B < ${16 + entryCount * 32}B)`
      );
      return;
    }

    const renderPipeline = this.pipelines.get(pipelineId);
    const computePipeline = this.computePipelines.get(pipelineId);
    const pipeline = renderPipeline || computePipeline;
    const pipeMeta = this._pipelineMeta.get(pipelineId) || null;
    const renderStateExt = this._getRenderStateExtension();

    let layout = null;
    if (!pipeline) {
      const bglRec =
        renderStateExt && renderStateExt.bindGroupLayouts
          ? renderStateExt.bindGroupLayouts.get(pipelineId)
          : null;
      const pendingLayout =
        bglRec && bglRec.layout
          ? null
          : this._pendingAsyncPipelineBindGroupLayout(pipelineId, groupIndex);
      if (bglRec && bglRec.layout) {
        layout = bglRec.layout;
      } else if (pendingLayout) {
        layout = pendingLayout;
      } else {
        this._recordOpValidationError(
          'CreateBindGroup',
          `bind group ${id}: unknown pipeline or bind group layout ${pipelineId}`
        );
        return;
      }
    } else if (pipeMeta && pipeMeta.isFallback) {
      try {
        layout = this.getBindGroupLayout(pipelineId, groupIndex);
      } catch (_) {
        layout = null;
      }
    } else {
      try {
        layout = this.getBindGroupLayout(pipelineId, groupIndex);
      } catch (err) {
        this._recordOpValidationError('CreateBindGroup', err);
        return;
      }
      if (!layout) {
        this._recordOpValidationError(
          'CreateBindGroup',
          `bind group ${id}: pipeline ${pipelineId} has no bind group layout for @group(${groupIndex})`
        );
        return;
      }
    }

    const entries = [];
    let hasCbv = false;
    let hasTexture = false;
    let hasSampler = false;
    for (let i = 0; i < entryCount; i++) {
      const entryOffset = payloadOffset + 16 + i * 32;
      const binding = view.getUint32(entryOffset, true);
      const resourceType = view.getUint32(entryOffset + 4, true);
      const resourceId = view.getUint32(entryOffset + 8, true);
      const offset = Number(view.getBigUint64(entryOffset + 16, true));
      const size = Number(view.getBigUint64(entryOffset + 24, true));

      if (resourceType === BIND_RESOURCE_TYPE.BUFFER) {
        const buffer = this.buffers.get(resourceId);
        if (!buffer) {
          this._recordOpValidationError(
            'CreateBindGroup',
            `bind group ${id}: binding ${binding} unknown buffer ${resourceId}`
          );
          return;
        }
        const resource = { buffer, offset };
        if (size > 0) {
          resource.size = size;
        }
        hasCbv = true;
        entries.push({ binding, resource });
      } else if (resourceType === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
        let textureView = this.textureViews.get(resourceId);
        if (!textureView) {
          const texture = this.textures.get(resourceId);
          if (texture && typeof texture.createView === 'function') {
            textureView = this._createSafeTextureView(texture);
            this.textureViews.set(resourceId, textureView);
            if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
              this.resourceRegistry.register('textureView', resourceId, textureView, {
                byteSize: 0,
                label: `textureView:${resourceId}`,
              });
            }
          }
        }
        if (!textureView) {
          this._recordOpValidationError(
            'CreateBindGroup',
            `bind group ${id}: binding ${binding} unknown texture ${resourceId}`
          );
          return;
        }
        hasTexture = true;
        entries.push({ binding, resource: textureView });
      } else if (resourceType === BIND_RESOURCE_TYPE.SAMPLER) {
        let sampler = this.samplers.get(resourceId);
        if (!sampler && resourceId === 0 && typeof this.device.createSampler === 'function') {
          sampler = this.getOrCreateDefaultSampler('filtering');
          if (sampler) {
            this._fallbackSamplerBindingsSynthesized += 1;
          }
        }
        if (!sampler) {
          this._recordOpValidationError(
            'CreateBindGroup',
            `bind group ${id}: binding ${binding} unknown sampler ${resourceId}`
          );
          return;
        }
        hasSampler = true;
        entries.push({ binding, resource: sampler });
      } else {
        this._recordOpValidationError(
          'CreateBindGroup',
          `bind group ${id}: binding ${binding} invalid resourceType ${resourceType}`
        );
        return;
      }
    }

    let finalEntries = entries;
    const layoutGroupForSynth = this._getBindGroupLayoutEntriesForSynthesis(
      pipelineId,
      groupIndex,
      pipeMeta,
      renderStateExt
    );
    if (
      layoutGroupForSynth &&
      this.device &&
      typeof this.device.createSampler === 'function'
    ) {
      const synth = this.synthesizeFallbackSamplerEntries(layoutGroupForSynth, finalEntries);
      if (synth && synth.synthesizedCount > 0 && Array.isArray(synth.entries)) {
        finalEntries = synth.entries;
        hasSampler = true;
      }
    }

    this._bindGroupMeta.set(id, {
      id,
      pipelineId,
      groupIndex,
      hasCbv,
      hasTexture,
      hasSampler,
      entries: finalEntries,
    });
    this._invalidateSpecializedBindGroupsFor(id);

    if (!layout) {
      return;
    }

    let bindGroup;
    try {
      bindGroup = this.device.createBindGroup({
        layout,
        entries: finalEntries,
      });
    } catch (err) {
      this._recordOpValidationError('CreateBindGroup', err);
      return;
    }
    this.bindGroups.set(id, bindGroup);
    this._bindGroupKinds.set(id, computePipeline && !renderPipeline ? 'compute' : 'render');
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('bindGroup', id, bindGroup, {
        byteSize: 0,
        label: `bindGroup:${id}`,
      });
    }
  }

  _dispatchExtensionPacket(opcode, view, payloadOffset, payloadBytes) {
    if (!this.extension || typeof this.extension.handlePacket !== 'function') {
      return;
    }
    let pendingRangedPipeline = false;
    if (opcode === OPCODES.CREATE_RENDER_PIPELINE2 && payloadBytes >= 16 + 376) {
      const pipeId = view.getUint32(payloadOffset, true);
      const vsShaderId = view.getUint32(payloadOffset + 4, true);
      const fsShaderId = view.getUint32(payloadOffset + 8, true);
      const pendingRanged = this._registerPendingRangedPipeline(
        pipeId,
        'render2',
        OPCODES.CREATE_RENDER_PIPELINE2,
        fsShaderId !== 0 ? [vsShaderId, fsShaderId] : [vsShaderId],
        view,
        payloadOffset,
        payloadBytes
      );
      pendingRangedPipeline = pendingRanged.pending;
      if (pendingRanged.deferredOnly) {
        return;
      }
    } else if (opcode === OPCODES.CREATE_COMPUTE_PIPELINE2 && payloadBytes >= 32) {
      const pipeId = view.getUint32(payloadOffset, true);
      const csShaderId = view.getUint32(payloadOffset + 4, true);
      const pendingRanged = this._registerPendingRangedPipeline(
        pipeId,
        'compute2',
        OPCODES.CREATE_COMPUTE_PIPELINE2,
        [csShaderId],
        view,
        payloadOffset,
        payloadBytes
      );
      pendingRangedPipeline = pendingRanged.pending;
      if (pendingRanged.deferredOnly) {
        return;
      }
    }
    const renderStateExt = this._getRenderStateExtension();
    const prevRsErrors =
      renderStateExt && renderStateExt.stats && typeof renderStateExt.stats.errors === 'number'
        ? renderStateExt.stats.errors
        : 0;
    const prevExtErrors =
      this.extension.stats && typeof this.extension.stats.errors === 'number'
        ? this.extension.stats.errors
        : 0;
    const prevCmdLen = this._passCommands.length;

    let handled = false;
    try {
      handled = Boolean(this.extension.handlePacket(opcode, view, payloadOffset, payloadBytes));
    } catch (err) {
      if (!pendingRangedPipeline) {
        this._recordOpValidationError(opcodeLabel(opcode), err);
      }
      return true;
    }

    if (
      renderStateExt &&
      renderStateExt.stats &&
      typeof renderStateExt.stats.errors === 'number' &&
      renderStateExt.stats.errors > prevRsErrors
    ) {
      if (!pendingRangedPipeline) {
        this._recordOpValidationError(
          opcodeLabel(opcode),
          renderStateExt.lastError || 'malformed or invalid ABI v2 packet'
        );
      }
      return true;
    }

    if (
      this.extension !== renderStateExt &&
      this.extension.stats &&
      typeof this.extension.stats.errors === 'number' &&
      this.extension.stats.errors > prevExtErrors
    ) {
      if (!pendingRangedPipeline) {
        this._recordOpValidationError(
          opcodeLabel(opcode),
          this.extension.lastError || 'malformed or invalid extension packet'
        );
      }
      return true;
    }

    if (!handled) {
      return false;
    }

    switch (opcode) {
      case OPCODES.CREATE_TEXTURE2: {
        if (payloadBytes >= 32) {
          const id = view.getUint32(payloadOffset, true);
          this._reconcileTrackedTextureAfterExternalCreate(id, view, payloadOffset, true);
        }
        break;
      }
      case OPCODES.CREATE_BIND_GROUP_LAYOUT: {
        if (payloadBytes >= 24) {
          const id = view.getUint32(payloadOffset, true);
          const count = view.getUint32(payloadOffset + 4, true);
          const flags = view.getUint32(payloadOffset + 8, true);
          this._v2BindGroupLayouts.set(id, {
            id,
            entryCount: count,
            isManifestRef: (flags & 0x1) !== 0,
          });
        }
        break;
      }
      case OPCODES.CREATE_PIPELINE_LAYOUT: {
        if (payloadBytes >= 24) {
          const id = view.getUint32(payloadOffset, true);
          const count = view.getUint32(payloadOffset + 4, true);
          const flags = view.getUint32(payloadOffset + 8, true);
          const isManifestRef = (flags & 0x1) !== 0;
          let groupLayoutIds = null;
          if (!isManifestRef && 24 + count * 4 <= payloadBytes) {
            groupLayoutIds = [];
            for (let g = 0; g < count; g += 1) {
              groupLayoutIds.push(view.getUint32(payloadOffset + 24 + g * 4, true));
            }
          }
          this._v2PipelineLayouts.set(id, { id, groupLayoutIds, isManifestRef });
        }
        break;
      }
      case OPCODES.CREATE_RENDER_PIPELINE2: {
        if (payloadBytes >= 16 + 376) {
          const id = view.getUint32(payloadOffset, true);
          const vsShaderId = view.getUint32(payloadOffset + 4, true);
          const fsShaderId = view.getUint32(payloadOffset + 8, true);
          const layoutId = view.getUint32(payloadOffset + 12, true);
          const vsEntry = this.shaderModules.get(vsShaderId) || null;
          const fsEntry = fsShaderId !== 0 ? this.shaderModules.get(fsShaderId) || null : null;

          const isFallback = Boolean(
            (vsEntry && vsEntry.source === 'builtin') || (fsEntry && fsEntry.source === 'builtin')
          );
          const usesFallbackShader = Boolean(
            (vsEntry && (vsEntry.source === 'builtin' || vsEntry.source === 'fallback')) ||
              (fsEntry && (fsEntry.source === 'builtin' || fsEntry.source === 'fallback'))
          );
          const isRealPack = Boolean(
            vsEntry &&
              vsEntry.source === 'pack' &&
              (fsShaderId === 0 || (fsEntry && fsEntry.source === 'pack'))
          );
          const effectiveFallbackProfile =
            vsEntry && vsEntry.fallbackProfile && vsEntry.fallbackProfile !== FALLBACK_PROFILE.DEFAULT
              ? vsEntry.fallbackProfile
              : fsEntry && fsEntry.fallbackProfile && fsEntry.fallbackProfile !== FALLBACK_PROFILE.DEFAULT
                ? fsEntry.fallbackProfile
                : isFallback
                  ? FALLBACK_PROFILE.DEFAULT
                  : null;

          let expectsGroup0 = Boolean(
            wgslDeclaresGroup0(vsEntry && vsEntry.code) || wgslDeclaresGroup0(fsEntry && fsEntry.code)
          );
          if (layoutId !== 0) {
            const plMeta = this._v2PipelineLayouts.get(layoutId) || null;
            if (plMeta && Array.isArray(plMeta.groupLayoutIds)) {
              const resolvedBgls = plMeta.groupLayoutIds.map((bglId) => {
                if (bglId === 0 || !renderStateExt) return null;
                const rec = renderStateExt.bindGroupLayouts.get(bglId);
                return rec ? rec.layout : null;
              });
              this._pipelineBindGroupLayouts.set(id, resolvedBgls);
              const group0BglId = plMeta.groupLayoutIds.length > 0 ? plMeta.groupLayoutIds[0] : 0;
              if (group0BglId !== 0) {
                const bglMeta = this._v2BindGroupLayouts.get(group0BglId) || null;
                const extBgl = renderStateExt
                  ? renderStateExt.bindGroupLayouts.get(group0BglId)
                  : null;
                if (
                  (bglMeta && (bglMeta.entryCount > 0 || bglMeta.isManifestRef)) ||
                  (extBgl && Array.isArray(extBgl.entries) && extBgl.entries.length > 0)
                ) {
                  expectsGroup0 = true;
                }
              }
            }
          } else {
            this._pipelineBindGroupLayouts.delete(id);
          }

          const keyBase = payloadOffset + 16;
          const bufferCount = Math.min(view.getUint32(keyBase + 172, true), 8);
          const vertexStride = bufferCount > 0 ? view.getUint32(keyBase + 180, true) : 0;

          this._pipelineMeta.set(id, {
            id,
            vsShaderId,
            fsShaderId,
            layoutId,
            vertexStride,
            topology: 'triangle-list',
            isV2: true,
            hasReflectedLayout: layoutId !== 0,
            isRealPack,
            isFallback,
            usesFallbackShader,
            expectsGroup0,
            fallbackProfile: effectiveFallbackProfile,
          });
          this._invalidateSpecializedPipelinesFor(id);
        }
        break;
      }
      case OPCODES.CREATE_BIND_GROUP2: {
        if (payloadBytes >= 16) {
          const id = view.getUint32(payloadOffset, true);
          const layoutId = view.getUint32(payloadOffset + 4, true);
          const count = view.getUint32(payloadOffset + 8, true);
          let hasCbv = false;
          let hasTexture = false;
          let hasSampler = false;
          if (16 + count * 48 <= payloadBytes) {
            for (let i = 0; i < count; i += 1) {
              const entryType = view.getUint32(payloadOffset + 16 + i * 48 + 4, true);
              if (entryType === BIND_RESOURCE_TYPE.BUFFER) hasCbv = true;
              else if (entryType === BIND_RESOURCE_TYPE.TEXTURE_VIEW) hasTexture = true;
              else if (entryType === BIND_RESOURCE_TYPE.SAMPLER) hasSampler = true;
            }
          }
          this._bindGroupMeta.set(id, {
            id,
            pipelineId: 0,
            layoutId,
            isV2: true,
            groupIndex: 0,
            hasCbv,
            hasTexture,
            hasSampler,
            entries: null,
          });
          this._invalidateSpecializedBindGroupsFor(id);
        }
        break;
      }
      case OPCODES.SET_BIND_GROUP2: {
        if (payloadBytes >= 16 && this._passCommands.length > prevCmdLen) {
          this._passCommands[this._passCommands.length - 1]._wgpuMeta =
            this._acquireSetBindGroup2Meta(
              view.getUint32(payloadOffset, true),
              view.getUint32(payloadOffset + 4, true)
            );
        }
        break;
      }
      case OPCODES.BEGIN_RENDER_PASS: {
        if (this._passCommands.length > prevCmdLen) {
          this._passCommands[this._passCommands.length - 1]._wgpuMeta = WGPU_META_BEGIN_RENDER_PASS;
        }
        break;
      }
      case OPCODES.END_RENDER_PASS: {
        if (this._passCommands.length > prevCmdLen) {
          this._passCommands[this._passCommands.length - 1]._wgpuMeta = WGPU_META_END_RENDER_PASS;
        }
        break;
      }
      default:
        break;
    }
    return true;
  }

  _handleDestroyResource(id) {
    this._destroyBufferIfPresent(id);
    this._destroyTextureIfPresent(id);
    if (this.resourceRegistry && typeof this.resourceRegistry.retireHandle === 'function') {
      this.resourceRegistry.retireHandle(id);
    }
    this._bufferMeta.delete(id);
    this.samplers.delete(id);
    this.bindGroups.delete(id);
    this._bindGroupKinds.delete(id);
    this._bindGroupMeta.delete(id);
    this._v2BindGroupLayouts.delete(id);
    this._v2PipelineLayouts.delete(id);
    this._invalidateSpecializedBindGroupsFor(id);
    this.shaderModules.delete(id);
    this._shaderModuleGenerations.delete(id);
    this._inFlightRangedShaders.delete(id);
    if (this._pendingRangedPipelines.size > 0) {
      this._pendingRangedPipelines.delete(id);
      for (const [pipeId, rec] of this._pendingRangedPipelines.entries()) {
        if (rec && Array.isArray(rec.shaderIds) && rec.shaderIds.includes(id)) {
          this._pendingRangedPipelines.delete(pipeId);
        }
      }
    }
    this.pipelines.delete(id);
    this._pipelineMeta.delete(id);
    this._invalidateSpecializedPipelinesFor(id);
    this.computePipelines.delete(id);
    this._pipelineBindGroupLayouts.delete(id);
    this._computePipelineBindGroupLayouts.delete(id);
    this._pipelineReflectionGroups.delete(id);
    this._computePipelineReflectionGroups.delete(id);
    this._shaderReflectionKeys.delete(id);
    if (this.asyncPipelineCache && typeof this.asyncPipelineCache.delete === 'function') {
      this.asyncPipelineCache.delete(id);
    }
    if (this.extension) this.extension.destroyResource(id);
  }

  /** W103: reflected bind group layout of a pipeline that is still compiling asynchronously. */
  _pendingAsyncPipelineBindGroupLayout(pipelineId, groupIndex) {
    if (!this.isAsyncPipelinePending(pipelineId)) {
      return null;
    }
    const reflected =
      this._pipelineBindGroupLayouts.get(pipelineId) ||
      this._computePipelineBindGroupLayouts.get(pipelineId);
    return reflected && reflected[groupIndex] ? reflected[groupIndex] : null;
  }

  _clearPresentPassBindings() {
    const ps = this._presentState;
    ps.activePipelineId = null;
    ps.activePipeline = null;
    ps.activeComputePipelineId = null;
    ps.specializedActiveKey = null;
    if (this._boundBindGroups.size > 0) {
      for (const entry of this._boundBindGroups.values()) {
        if (entry) {
          entry.bindGroup = null;
          entry.meta = null;
          entry.entries = null;
          entry.desc = null;
        }
      }
      this._boundBindGroups.clear();
    }
    if (this._boundVertexBuffers.size > 0) {
      for (const entry of this._boundVertexBuffers.values()) {
        if (entry) {
          entry.buffer = null;
        }
      }
      this._boundVertexBuffers.clear();
    }
  }

  _bindPresentVertexBuffer(slot, bufferId, buffer, offset) {
    let entry = this._boundVertexBufferSlots.get(slot);
    if (!entry) {
      entry = { bufferId: 0, buffer: null, offset: 0, stride: undefined };
      this._boundVertexBufferSlots.set(slot, entry);
    }
    entry.bufferId = bufferId;
    entry.buffer = buffer;
    entry.offset = offset;
    entry.stride =
      typeof buffer.stride === 'number'
        ? buffer.stride
        : typeof buffer.arrayStride === 'number'
          ? buffer.arrayStride
          : undefined;
    this._boundVertexBuffers.set(slot, entry);
  }

  _bindPresentBindGroup(groupIndex, bindGroupId, bindGroup, bgMeta) {
    let entry = this._boundBindGroupSlots.get(groupIndex);
    if (!entry) {
      entry = {
        bindGroupId: 0,
        bindGroup: null,
        meta: null,
        hasCbv: false,
        hasTexture: false,
        hasSampler: false,
        entries: null,
        desc: null,
      };
      this._boundBindGroupSlots.set(groupIndex, entry);
    }
    entry.bindGroupId = bindGroupId;
    entry.bindGroup = bindGroup;
    entry.meta = bgMeta;
    entry.hasCbv = Boolean(bgMeta && bgMeta.hasCbv);
    entry.hasTexture = Boolean(bgMeta && bgMeta.hasTexture);
    entry.hasSampler = Boolean(bgMeta && bgMeta.hasSampler);
    entry.entries = bgMeta ? bgMeta.entries : null;
    entry.desc = bindGroup && bindGroup.desc ? bindGroup.desc : null;
    this._boundBindGroups.set(groupIndex, entry);
  }

  _endPresentActivePass() {
    const ps = this._presentState;
    if (ps.activePass) {
      try {
        ps.activePass.end();
      } catch (err) {
        this._recordOpValidationError('EndRenderPass', err);
      }
      ps.activePass = null;
      ps.activePassKind = null;
      this._clearPresentPassBindings();
    }
  }

  _ensurePresentRenderPass() {
    const ps = this._presentState;
    if (ps.activePassKind === 'render') return ps.activePass;
    this._endPresentActivePass();
    const attachment = {
      view: ps.swapchainView,
      loadOp: ps.renderPassOpened ? 'load' : 'clear',
      storeOp: 'store',
    };
    if (!ps.renderPassOpened) {
      attachment.clearValue = this._clearColor;
    }
    const passDesc = {
      colorAttachments: [attachment],
    };
    if (ps.timerSupported && typeof this.passTimer.getPassTimestampWrites === 'function') {
      const tsWrites = this.passTimer.getPassTimestampWrites(`render_${ps.renderPassIdx}`);
      if (tsWrites) {
        passDesc.timestampWrites = tsWrites;
      }
    }
    ps.renderPassIdx += 1;
    ps.activePass = ps.encoder.beginRenderPass(passDesc);
    ps.activePassKind = 'render';
    ps.renderPassOpened = true;
    return ps.activePass;
  }

  _ensurePresentComputePass() {
    const ps = this._presentState;
    if (ps.activePassKind === 'compute') return ps.activePass;
    this._endPresentActivePass();
    if (ps.timerSupported && typeof this.passTimer.getPassTimestampWrites === 'function') {
      const tsWrites = this.passTimer.getPassTimestampWrites(`compute_${ps.computePassIdx}`);
      ps.computePassIdx += 1;
      ps.activePass = tsWrites
        ? ps.encoder.beginComputePass({ timestampWrites: tsWrites })
        : ps.encoder.beginComputePass();
    } else {
      ps.computePassIdx += 1;
      ps.activePass = ps.encoder.beginComputePass();
    }
    ps.activePassKind = 'compute';
    return ps.activePass;
  }

  _beginExtensionRenderPass(descriptor) {
    const ps = this._presentState;
    this._endPresentActivePass();
    if (ps.timerSupported && typeof this.passTimer.getPassTimestampWrites === 'function') {
      const tsWrites = this.passTimer.getPassTimestampWrites(`render_${ps.renderPassIdx}`);
      if (tsWrites) descriptor = { ...descriptor, timestampWrites: tsWrites };
    }
    ps.renderPassIdx += 1;
    ps.activePass = ps.encoder.beginRenderPass(descriptor);
    ps.activePassKind = 'render';
    ps.renderPassOpened = true;
    return ps.activePass;
  }

  _maybeSpecializeFallbackPipeline(pass, vertexCount, instanceCount) {
    const ps = this._presentState;
    if (ps.activePipelineId === null || !this.shaderFallbackRouter) {
      return;
    }
    const pipeMeta = this._pipelineMeta.get(ps.activePipelineId);
    if (!pipeMeta || !pipeMeta.isFallback) {
      return;
    }
    const scratch = this._fallbackProfileStateScratch;
    scratch.vertexCount = vertexCount;
    scratch.instanceCount = instanceCount;
    scratch.vertexStride = pipeMeta.vertexStride || 0;
    const desiredProfile = selectFallbackShaderProfile(scratch);
    if (
      desiredProfile === FALLBACK_PROFILE.DEFAULT ||
      desiredProfile === pipeMeta.fallbackProfile
    ) {
      return;
    }
    const spec = this._getOrCreateSpecializedFallbackPipeline(pipeMeta, desiredProfile);
    if (!spec || !spec.changed || !spec.pipeline) {
      return;
    }
    const key = `${pipeMeta.id}|${desiredProfile}`;
    if (ps.specializedActiveKey !== key) {
      pass.setPipeline(spec.pipeline);
      for (const [groupIndex, boundBg] of this._boundBindGroups.entries()) {
        if (boundBg && boundBg.meta) {
          const specBg = this._getOrCreateSpecializedBindGroup(
            pipeMeta.id,
            desiredProfile,
            spec.pipeline,
            groupIndex,
            boundBg.meta
          );
          if (specBg) {
            pass.setBindGroup(groupIndex, specBg);
            boundBg.bindGroup = specBg;
          }
        }
      }
      ps.specializedActiveKey = key;
    }
  }

  _executePresentRenderDraw(drawOpName, cmd, isIndexed, indirectBuffer = null) {
    const ps = this._presentState;
    const pass = this._ensurePresentRenderPass();
    const isIndirect = indirectBuffer !== null;
    const count = isIndirect ? 1 : isIndexed ? cmd.indexCount : cmd.vertexCount;
    const instanceCount = isIndirect ? 1 : cmd.instanceCount;
    this._maybeSpecializeFallbackPipeline(pass, count, instanceCount);
    const activePipelineId = ps.activePipelineId;
    const pipeMeta =
      activePipelineId !== null ? this._pipelineMeta.get(activePipelineId) || null : null;
    const activePipeline =
      activePipelineId !== null
        ? this.pipelines.get(activePipelineId) || ps.activePipeline || null
        : null;
    if (activePipelineId !== null && !activePipeline && ps.specializedActiveKey === null) {
      if (this.isAsyncPipelinePending(activePipelineId)) {
        // W103: still compiling; draw normally on a later frame once it is ready.
        this._pendingAsyncPipelineSkips += 1;
        return;
      }
      this._skippedDrawCount += 1;
      this._recordOpValidationError(
        drawOpName,
        `skipped draw: render pipeline ${activePipelineId} is unavailable`
      );
      return;
    }
    const group0Entry = this._boundBindGroups.get(0);
    const hasLiveGroup0 = Boolean(group0Entry && group0Entry.bindGroup);
    if (pipeMeta && pipeMeta.expectsGroup0 && !hasLiveGroup0) {
      this._unboundBindGroupCount += 1;
      this._skippedDrawCount += 1;
      this._recordOpValidationError(
        drawOpName,
        `skipped draw on pipeline ${activePipelineId}: expected @group(0) bind group is not bound`
      );
      return;
    }
    try {
      if (isIndirect) {
        if (isIndexed) {
          pass.drawIndexedIndirect(indirectBuffer, cmd.offset);
        } else {
          pass.drawIndirect(indirectBuffer, cmd.offset);
        }
      } else if (isIndexed) {
        pass.drawIndexed(
          cmd.indexCount,
          cmd.instanceCount,
          cmd.firstIndex,
          cmd.baseVertex,
          cmd.firstInstance
        );
      } else {
        pass.draw(cmd.vertexCount, cmd.instanceCount, cmd.firstVertex, cmd.firstInstance);
      }
    } catch (err) {
      this._skippedDrawCount += 1;
      if (/\b(?:group|bind\s*group)\b/i.test(String(err))) {
        this._unboundBindGroupCount += 1;
      }
      this._recordOpValidationError(drawOpName, err);
      return;
    }
    this._drawCallCount += 1;
    this._frameDrawsThisFrame += 1;
    const isClearPipe = Boolean(pipeMeta && pipeMeta.isClear === true);
    if (pipeMeta && pipeMeta.isRealPack && ps.specializedActiveKey === null) {
      this._realPackDrawCount += 1;
      this._frameRealPackDrawsThisFrame += 1;
    } else if (
      (pipeMeta && (pipeMeta.isFallback || pipeMeta.usesFallbackShader)) ||
      ps.specializedActiveKey !== null
    ) {
      this._fallbackDrawCount += 1;
      this._frameFallbackDrawsThisFrame += 1;
    }
    if (!isClearPipe) {
      this._frameNonClearDrawsThisFrame += 1;
    }
  }

  _ensurePresentFrameDrawSummary() {
    const ps = this._presentState;
    if (ps.cachedFrameSummary !== null) {
      return ps.cachedFrameSummary;
    }
    let rawSummary = null;
    if (this.drawExecutor && typeof this.drawExecutor.getFrameDrawSummary === 'function') {
      try {
        rawSummary = this.drawExecutor.getFrameDrawSummary(this._frameDrawSummaryOpts);
      } catch (_) {
        rawSummary = this.drawExecutor.getFrameDrawSummary();
      }
    }
    const summary = this._frameSummaryScratch;
    if (rawSummary && typeof rawSummary === 'object') {
      summary.hasNonClearDraw =
        Boolean(rawSummary.hasNonClearDraw === true) ||
        this._frameNonClearDrawsThisFrame > 0;
      summary.realPackDrawsThisFrame =
        toNonNegativeInt(rawSummary.realPackDrawsThisFrame, 0) +
        this._frameRealPackDrawsThisFrame;
      summary.fallbackDrawsThisFrame =
        toNonNegativeInt(rawSummary.fallbackDrawsThisFrame, 0) +
        this._frameFallbackDrawsThisFrame;
      summary.drawsThisFrame =
        toNonNegativeInt(rawSummary.drawsThisFrame, 0) + this._frameDrawsThisFrame;
    } else {
      summary.hasNonClearDraw = this._frameNonClearDrawsThisFrame > 0;
      summary.realPackDrawsThisFrame = this._frameRealPackDrawsThisFrame;
      summary.fallbackDrawsThisFrame = this._frameFallbackDrawsThisFrame;
      summary.drawsThisFrame = this._frameDrawsThisFrame;
    }
    ps.cachedFrameSummary = summary;
    return summary;
  }

  /**
   * @param {{ readbackOnly?: boolean }} [opts] G66: `readbackOnly` submits the recorded work up to
   *   the last read-back without presenting (no frame accounting, no frame reset), so the staging
   *   maps can start while the title waits on the fence that guards the read-back.
   */
  _flushPresent(opts) {
    const readbackOnly = Boolean(opts && opts.readbackOnly === true);
    const pendingEncoder = this._pendingEncoder;
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;

    // W99: a torn-down WebGPU instance throws `OperationError` from createCommandEncoder /
    // getCurrentTexture; record it and drop the frame instead of crashing the message loop.
    // W112 / W121: when WgpuSwapchain activates offscreen blit fallback for a non-canvas
    // stream format, bind the intermediate render-target view for SWAPCHAIN_TEXTURE_ID passes
    // and keep the canvas view handy to blit right before encoder.finish() / queue.submit().
    const activeSwapchain =
      this._swapchain && !this._swapchain.isDestroyed ? this._swapchain : null;
    const useSwapchainBlitFallback = Boolean(
      activeSwapchain &&
        activeSwapchain.usesBlitFallback === true &&
        activeSwapchain.isConfigured !== false &&
        typeof activeSwapchain.getRenderTargetView === 'function'
    );

    const profilePresent = this._stallProfileActive === true && this._stallProfileDetailed === true;
    const pStartMs = profilePresent ? this._getNowMs() : 0;
    let encoder;
    let swapchainTexture;
    let swapchainView;
    let canvasTexture = null;
    let canvasView = null;
    try {
      encoder = pendingEncoder || this.device.createCommandEncoder();
      const getTexStartMs = profilePresent ? this._getNowMs() : 0;
      canvasTexture = this.ctx.getCurrentTexture();
      if (profilePresent) {
        this._stallPresentBreakdown.getTexMs += this._getNowMs() - getTexStartMs;
      }
      canvasView =
        canvasTexture && typeof canvasTexture.createView === 'function'
          ? this._swapchainViewFormat && !useSwapchainBlitFallback
            ? canvasTexture.createView({ format: this._swapchainViewFormat })
            : canvasTexture.createView()
          : canvasTexture;
      if (useSwapchainBlitFallback) {
        swapchainView = activeSwapchain.getRenderTargetView();
        swapchainTexture = activeSwapchain.intermediateTexture || canvasTexture;
      } else {
        swapchainTexture = canvasTexture;
        swapchainView = canvasView;
      }
    } catch (err) {
      this._notePresentSubmitFailure(err);
      if (readbackOnly) {
        this._startRecordedReadbacks(false);
        return;
      }
      this._drainDeferredGpuDestroys(null, DRAIN_DEFERRED_OPTS_IMMEDIATE);
      this._endSubsystemsForFrame(null);
      this._resetFrameState();
      return;
    }

    const ps = this._presentState;
    ps.encoder = encoder;
    ps.activeSwapchain = activeSwapchain;
    ps.swapchainTexture = swapchainTexture;
    ps.swapchainView = swapchainView;
    ps.activePass = null;
    ps.activePassKind = null;
    // After an early read-back flush the canvas already holds this frame's work: keep it.
    ps.renderPassOpened = readbackOnly || this._earlyFlushedThisFrame === true;
    ps.renderPassIdx = 0;
    ps.computePassIdx = 0;
    ps.timerSupported = Boolean(this.passTimer && this.passTimer.supported);
    ps.activePipelineId = null;
    ps.activePipeline = null;
    ps.activeComputePipelineId = null;
    ps.specializedActiveKey = null;
    ps.cachedFrameSummary = null;
    this._clearPresentPassBindings();

    const passControl = this._passControl;
    passControl.encoder = encoder;
    passControl.swapchainTexture = swapchainTexture;
    passControl.swapchainView = swapchainView;
    passControl.canvasWidth = this._viewportWidth > 0 ? this._viewportWidth : undefined;
    passControl.canvasHeight = this._viewportHeight > 0 ? this._viewportHeight : undefined;
    passControl.swapchainWidth =
      activeSwapchain && activeSwapchain.width > 0 ? activeSwapchain.width : undefined;
    passControl.swapchainHeight =
      activeSwapchain && activeSwapchain.height > 0 ? activeSwapchain.height : undefined;

    const cmdsStartMs = profilePresent ? this._getNowMs() : 0;
    const cmds = this._passCommands;
    for (let i = 0, cmdLen = cmds.length; i < cmdLen; i += 1) {
      const cmd = cmds[i];
      switch (cmd.type) {
        case 'setPipeline': {
          const pass = this._ensurePresentRenderPass();
          ps.activePipelineId = cmd.pipelineId;
          ps.activePipeline = null;
          ps.specializedActiveKey = null;
          const pipeline = this.pipelines.get(cmd.pipelineId) || cmd.pipeline || null;
          if (pipeline) {
            try {
              pass.setPipeline(pipeline);
              ps.activePipeline = pipeline;
            } catch (err) {
              this._recordOpValidationError('SetPipeline', err);
            }
          } else if (!this.isAsyncPipelinePending(cmd.pipelineId)) {
            this._recordOpValidationError(
              'SetPipeline',
              `render pipeline ${cmd.pipelineId} not found`
            );
          }
          break;
        }
        case 'setVertexBuffer': {
          const pass = this._ensurePresentRenderPass();
          const buffer = this.buffers.get(cmd.bufferId);
          if (buffer) {
            this._bindPresentVertexBuffer(cmd.slot, cmd.bufferId, buffer, cmd.offset);
            try {
              pass.setVertexBuffer(cmd.slot, buffer, cmd.offset);
            } catch (err) {
              this._recordOpValidationError('SetVertexBuffer', err);
            }
          }
          break;
        }
        case 'setIndexBuffer': {
          const pass = this._ensurePresentRenderPass();
          const buffer = this.buffers.get(cmd.bufferId);
          if (buffer) {
            try {
              pass.setIndexBuffer(buffer, cmd.indexFormat, cmd.offset);
            } catch (err) {
              this._recordOpValidationError('SetIndexBuffer', err);
            }
          }
          break;
        }
        case 'setBindGroup': {
          const bindGroup = this.bindGroups.get(cmd.bindGroupId) || cmd.bindGroup || null;
          const bgMeta = this._bindGroupMeta.get(cmd.bindGroupId) || null;
          if (bindGroup || bgMeta) {
            this._bindPresentBindGroup(cmd.groupIndex, cmd.bindGroupId, bindGroup || null, bgMeta);
          } else {
            this._recordOpValidationError(
              'SetBindGroup',
              `bind group ${cmd.bindGroupId} not found for @group(${cmd.groupIndex})`
            );
          }
          if (bindGroup) {
            const pass =
              ps.activePass ||
              (this._bindGroupKinds.get(cmd.bindGroupId) === 'compute'
                ? this._ensurePresentComputePass()
                : this._ensurePresentRenderPass());
            try {
              pass.setBindGroup(cmd.groupIndex, bindGroup);
            } catch (err) {
              this._recordOpValidationError('SetBindGroup', err);
            }
          }
          break;
        }
        case 'draw': {
          this._executePresentRenderDraw('Draw', cmd, false);
          break;
        }
        case 'drawIndexed': {
          this._executePresentRenderDraw('DrawIndexed', cmd, true);
          break;
        }
        case 'setComputePipeline': {
          const pass = this._ensurePresentComputePass();
          ps.activeComputePipelineId = cmd.pipelineId;
          const pipeline = this.computePipelines.get(cmd.pipelineId);
          if (pipeline) {
            try {
              pass.setPipeline(pipeline);
            } catch (err) {
              this._recordOpValidationError('SetComputePipeline', err);
            }
          } else if (!this.isAsyncPipelinePending(cmd.pipelineId)) {
            this._recordOpValidationError(
              'SetComputePipeline',
              `compute pipeline ${cmd.pipelineId} not found`
            );
          }
          break;
        }
        case 'dispatchCompute': {
          const activeComputeId = ps.activeComputePipelineId;
          if (activeComputeId !== null && !this.computePipelines.get(activeComputeId)) {
            if (this.isAsyncPipelinePending(activeComputeId)) {
              this._pendingAsyncPipelineSkips += 1;
              break;
            }
            this._recordOpValidationError(
              'DispatchCompute',
              `skipped dispatch: compute pipeline ${activeComputeId} is unavailable`
            );
            break;
          }
          const pass = this._ensurePresentComputePass();
          try {
            pass.dispatchWorkgroups(cmd.workgroupX, cmd.workgroupY, cmd.workgroupZ);
            this._computeDispatchCount += 1;
          } catch (err) {
            this._recordOpValidationError('DispatchCompute', err);
          }
          break;
        }
        case 'drawIndirect':
        case 'drawIndexedIndirect': {
          const indexed = cmd.type === 'drawIndexedIndirect';
          const opName = indexed ? 'DrawIndexedIndirect' : 'DrawIndirect';
          const buffer = this.buffers.get(cmd.bufferId);
          if (!buffer) {
            this._recordOpValidationError(opName, `indirect buffer ${cmd.bufferId} not found`);
            break;
          }
          this._executePresentRenderDraw(opName, cmd, indexed, buffer);
          break;
        }
        case 'dispatchIndirect': {
          const buffer = this.buffers.get(cmd.bufferId);
          if (!buffer) {
            this._recordOpValidationError('DispatchIndirect', `indirect buffer ${cmd.bufferId} not found`);
            break;
          }
          const activeComputeId = ps.activeComputePipelineId;
          if (activeComputeId !== null && !this.computePipelines.get(activeComputeId)) {
            if (this.isAsyncPipelinePending(activeComputeId)) {
              this._pendingAsyncPipelineSkips += 1;
              break;
            }
            this._recordOpValidationError(
              'DispatchIndirect',
              `skipped dispatch: compute pipeline ${activeComputeId} is unavailable`
            );
            break;
          }
          const pass = this._ensurePresentComputePass();
          try {
            pass.dispatchWorkgroupsIndirect(buffer, cmd.offset);
            this._computeDispatchCount += 1;
          } catch (err) {
            this._recordOpValidationError('DispatchIndirect', err);
          }
          break;
        }
        default:
          if (typeof cmd.run === 'function') {
            const meta = cmd._wgpuMeta || null;
            try {
              cmd.run(passControl);
              if (meta && meta.kind === 'setBindGroup2' && ps.activePass) {
                const bg = this.bindGroups.get(meta.bindGroupId) || null;
                const bgMeta = this._bindGroupMeta.get(meta.bindGroupId) || null;
                this._bindPresentBindGroup(
                  meta.groupIndex,
                  meta.bindGroupId,
                  bg || SENTINEL_BOUND_BIND_GROUP,
                  bgMeta
                );
              }
            } catch (err) {
              this._recordOpValidationError(
                meta && meta.kind ? meta.kind : 'RenderStatePassCommand',
                err
              );
            }
          }
          break;
      }
    }

    this._endPresentActivePass();
    if (!ps.renderPassOpened && !readbackOnly) {
      try {
        this._ensurePresentRenderPass();
      } catch (err) {
        this._recordOpValidationError('BeginRenderPass', sanitizeGpuErrorDetail(err));
      }
      this._endPresentActivePass();
    }
    if (profilePresent) {
      this._stallPresentBreakdown.cmdsMs += this._getNowMs() - cmdsStartMs;
    }

    // W112 / W121: Blit the offscreen intermediate render-target texture onto the canvas
    // right before finishing and submitting when blit fallback is active (zero overhead
    // when usesBlitFallback === false).
    const blitStartMs = profilePresent ? this._getNowMs() : 0;
    if (
      !readbackOnly &&
      useSwapchainBlitFallback &&
      typeof activeSwapchain.blitToCanvasIfNeeded === 'function'
    ) {
      try {
        activeSwapchain.blitToCanvasIfNeeded(encoder, canvasView);
      } catch (err) {
        this._recordOpValidationError('SwapchainBlit', sanitizeGpuErrorDetail(err));
      }
    }
    if (profilePresent) {
      this._stallPresentBreakdown.blitMs += this._getNowMs() - blitStartMs;
    }

    // W338 / W348: While waiting for first-visible-frame on a real WebGPU device with COPY_SRC
    // swapchain readback support, encode 8 small horizontal strip copies before encoder.finish()
    // (throttled to >= 250 ms and strictly non-overlapping; zero work once first-visible-frame fires).
    // Once swapchain probe capability is established on this device (`_swapchainProbeSupported === true`),
    // keep `swapchainProbeActive = true` for EVERY frame (including clear-only frames, throttled frames,
    // frames while a readback is in flight, and transient resize frames) so nothing can fall through to
    // the unverified fallback path until real non-black pixels are read back.
    const swapchainAlreadyVisible = Boolean(
      activeSwapchain &&
        (activeSwapchain.hasVisibleContent === true ||
          activeSwapchain.hasFirstVisibleFrame === true)
    );
    if (swapchainAlreadyVisible && this._swapchainProbeBuffer) {
      this._destroySwapchainProbeBuffer();
    }
    let swapchainProbeActive = Boolean(
      !this._firstVisibleFrameNotified &&
        !swapchainAlreadyVisible &&
        this._requireNonBlackFrameExplicit !== false &&
        this._swapchainProbeSupported === true
    );
    let pendingSwapchainProbe = null;
    if (
      !readbackOnly &&
      !this._firstVisibleFrameNotified &&
      !swapchainAlreadyVisible &&
      this._requireNonBlackFrameExplicit !== false &&
      this._swapchainProbeSupported !== false &&
      encoder &&
      typeof encoder.copyTextureToBuffer === 'function'
    ) {
      const preSummary = this._ensurePresentFrameDrawSummary();
      const hasDrawCandidate =
        this._realPackDrawCount > 0 ||
        preSummary.realPackDrawsThisFrame > 0 ||
        preSummary.hasNonClearDraw;
      if (swapchainProbeActive || hasDrawCandidate) {
        if (
          activeSwapchain &&
          typeof activeSwapchain.ensureCopySrcUsage === 'function' &&
          activeSwapchain.hasCopySrcUsage === false &&
          this._swapchainProbeSupported !== false
        ) {
          try {
            activeSwapchain.ensureCopySrcUsage();
          } catch (_) {
            // Ignore best-effort COPY_SRC restore errors.
          }
        }
        const probeCap = this._resolveSwapchainProbeCapability(encoder, canvasTexture);
        if (probeCap !== null) {
          swapchainProbeActive = true;
          if (this._pendingFrameLuminance === null && !this._swapchainProbeInFlight) {
            const probeNowMs = this._getNowMs();
            const intervalElapsed =
              !Number.isFinite(this._swapchainProbeLastStartMs) ||
              probeNowMs - this._swapchainProbeLastStartMs >= this._swapchainProbeIntervalMs;
            if (intervalElapsed) {
              pendingSwapchainProbe = this._encodeSwapchainProbeCopies(
                encoder,
                canvasTexture,
                probeCap,
                probeNowMs,
                preSummary
              );
              if (this._swapchainProbeSupported === false) {
                swapchainProbeActive = false;
              }
            }
          }
        }
      }
    }

    let timerResolved = false;
    if (!readbackOnly && ps.timerSupported && typeof this.passTimer.resolveFrame === 'function') {
      timerResolved = Boolean(this.passTimer.resolveFrame(encoder));
    }

    // Flush batched per-draw dynamic uniform constants once before queue submission.
    if (
      this.uniformRing &&
      !this.uniformRing._destroyed &&
      typeof this.uniformRing.flush === 'function'
    ) {
      this.uniformRing.flush();
    }

    // Unmap any mapped staging chunks before queue submission.
    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.prepareSubmit === 'function'
    ) {
      this.stagingPool.prepareSubmit();
    }

    // W99: guard encoder.finish() and queue.submit() separately so a torn-down WebGPU instance
    // (`OperationError`) is recorded as a sanitized Present event without crashing the worker.
    let submitted = false;
    let commandBuffer = null;
    const finishStartMs = profilePresent ? this._getNowMs() : 0;
    try {
      commandBuffer = encoder.finish();
    } catch (err) {
      this._notePresentSubmitFailure(err);
    }
    if (profilePresent) {
      this._stallPresentBreakdown.finishMs += this._getNowMs() - finishStartMs;
    }
    if (commandBuffer !== null) {
      const submitStartMs = this._gpuStallWarnMs > 0 || profilePresent ? this._getNowMs() : 0;
      try {
        this.device.queue.submit([commandBuffer]);
        submitted = true;
      } catch (err) {
        this._notePresentSubmitFailure(err);
      }
      if (profilePresent) {
        this._stallPresentBreakdown.submitMs += this._getNowMs() - submitStartMs;
        this._stallPresentBreakdown.count += 1;
        const totalPresentMs = this._getNowMs() - pStartMs;
        const accountedMs =
          this._stallPresentBreakdown.getTexMs +
          this._stallPresentBreakdown.cmdsMs +
          this._stallPresentBreakdown.blitMs +
          this._stallPresentBreakdown.finishMs +
          this._stallPresentBreakdown.submitMs;
        if (totalPresentMs > accountedMs) {
          this._stallPresentBreakdown.otherMs += totalPresentMs - accountedMs;
        }
      }
      if (this._gpuStallWarnMs > 0) {
        this._maybeWarnGpuStall(
          'queue.submit',
          submitStartMs,
          null,
          this._frameDrawsThisFrame,
          this._computeDispatchCount
        );
      }
    }
    this._startRecordedReadbacks(submitted);
    if (readbackOnly) {
      ps.encoder = null;
      ps.activeSwapchain = null;
      ps.swapchainTexture = null;
      ps.swapchainView = null;
      ps.cachedFrameSummary = null;
      passControl.encoder = null;
      passControl.swapchainTexture = null;
      passControl.swapchainView = null;
      if (submitted) this._earlyFlushedThisFrame = true;
      return;
    }

    let presentSceneTransition = false;
    let presentResolutionChanged = false;
    let presentIsIdleFrame = false;

    if (submitted) {
      const presentTs = this._getNowMs();
      this.frameCount += 1;
      this._presentCount += 1;
      const curWidth =
        (swapchainTexture &&
          typeof swapchainTexture.width === 'number' &&
          swapchainTexture.width > 0 &&
          Math.floor(swapchainTexture.width)) ||
        (activeSwapchain &&
          typeof activeSwapchain.width === 'number' &&
          activeSwapchain.width > 0 &&
          Math.floor(activeSwapchain.width)) ||
        this._viewportWidth ||
        0;
      const curHeight =
        (swapchainTexture &&
          typeof swapchainTexture.height === 'number' &&
          swapchainTexture.height > 0 &&
          Math.floor(swapchainTexture.height)) ||
        (activeSwapchain &&
          typeof activeSwapchain.height === 'number' &&
          activeSwapchain.height > 0 &&
          Math.floor(activeSwapchain.height)) ||
        this._viewportHeight ||
        0;
      presentSceneTransition =
        this._pendingSceneTransition ||
        (this._lastPresentSource !== null && this._lastPresentSource !== 'command-buffer');
      presentResolutionChanged =
        this._lastPresentWidth > 0 &&
        curWidth > 0 &&
        (curWidth !== this._lastPresentWidth || curHeight !== this._lastPresentHeight);
      this._pendingSceneTransition = false;
      this._lastPresentSource = 'command-buffer';
      if (curWidth > 0 && curHeight > 0) {
        this._lastPresentWidth = curWidth;
        this._lastPresentHeight = curHeight;
      }

      if (this.frameCount === 1) {
        this._notifyFirstPresent({
          frameIndex: 0,
          framesPresented: 1,
          source: 'command-buffer',
          timestampMs: presentTs,
          gpuAdapter: this._gpuAdapter,
        });
      }

      const summary = this._ensurePresentFrameDrawSummary();
      presentIsIdleFrame = summary.drawsThisFrame === 0;

      // W348: While a non-black pixel gate is active (async swapchain luminance probe OR
      // explicit requireNonBlackFrame), NEVER mark activeSwapchain as having visible content
      // on an unverified / black frame. Doing so would leak hasVisibleContent=true and
      // visiblePresentsTotal>0 into swapchain stats & gpu-telemetry and let
      // swapchainReportedFirstVisible bypass the pixel probe.
      const pixelGateActive =
        !this._firstVisibleFrameNotified &&
        (swapchainProbeActive || this._requireNonBlackFrame);

      let syncGate = null;
      let syncVisibleConfirmed = false;
      if (pixelGateActive) {
        const shouldEvalSyncGate =
          this._pendingFrameLuminance !== null ||
          this._nonBlackPixelProbe !== null ||
          (this._requireNonBlackFrame && summary.hasNonClearDraw);
        if (shouldEvalSyncGate) {
          syncGate = this._evaluateCommandBufferNonBlackGate(
            null,
            activeSwapchain,
            curWidth,
            curHeight,
            summary
          );
          if (syncGate.evaluated) {
            this._lastSampledMaxLuma = Math.max(
              0,
              Math.min(255, Math.floor(syncGate.maxLuma || 0))
            );
            syncVisibleConfirmed = syncGate.visible === true;
          }
        }
      }

      const swapchainFrameVisible = pixelGateActive
        ? syncVisibleConfirmed
        : summary.hasNonClearDraw;

      let swapchainReportedFirstVisible = false;
      let markResult = null;
      if (activeSwapchain && typeof activeSwapchain.markFrameContent === 'function') {
        const prevSwapVisible = Boolean(
          activeSwapchain.hasFirstVisibleFrame === true ||
            activeSwapchain.hasVisibleContent === true
        );
        const markPayload = this._markFrameContentScratch;
        markPayload.hasNonClearDraw = swapchainFrameVisible;
        markPayload.nonClearDraw = swapchainFrameVisible;
        markPayload.hasVisibleContent = swapchainFrameVisible;
        markPayload.hasDraws = swapchainFrameVisible;
        markPayload.realPackDraws = swapchainFrameVisible ? summary.realPackDrawsThisFrame : 0;
        markPayload.fallbackDraws = swapchainFrameVisible ? summary.fallbackDrawsThisFrame : 0;
        markPayload.drawsThisFrame = summary.drawsThisFrame;
        markPayload.timestampMs = presentTs;
        try {
          markResult = activeSwapchain.markFrameContent(markPayload);
        } catch (_) {
          markResult = null;
        }
        if (typeof activeSwapchain.recordFramePresented === 'function') {
          try {
            activeSwapchain.recordFramePresented(presentTs);
          } catch (_) {
            // Swapchain telemetry must never break frame presentation.
          }
        }
        if (!pixelGateActive) {
          const nextSwapVisible = Boolean(
            activeSwapchain.hasFirstVisibleFrame === true ||
              activeSwapchain.hasVisibleContent === true
          );
          if (
            (!prevSwapVisible && nextSwapVisible) ||
            (markResult &&
              typeof markResult === 'object' &&
              (markResult.firstVisible === true ||
                markResult.firstVisibleFrame === true ||
                markResult.isFirstVisibleFrame === true ||
                markResult.hasFirstVisibleFrame === true))
          ) {
            swapchainReportedFirstVisible = true;
          }
        }
      } else if (activeSwapchain && typeof activeSwapchain.recordFramePresented === 'function') {
        try {
          activeSwapchain.recordFramePresented(presentTs);
        } catch (_) {
          // Swapchain telemetry must never break frame presentation.
        }
      }

      if (pixelGateActive && !this._firstVisibleFrameNotified) {
        if (syncVisibleConfirmed) {
          const firstPresentMs =
            this._firstPresentMs !== null ? this._firstPresentMs : presentTs;
          const blackLoadingMs = Math.max(0, presentTs - firstPresentMs);
          this._firstVisibleFrameMs = presentTs;
          this._blackLoadingMs = blackLoadingMs;
          this._notifyFirstVisibleFrame({
            type: 'first-visible-frame',
            firstVisibleFrame: true,
            firstVisibleFrameMs: presentTs,
            firstPresentMs,
            blackLoadingMs,
            presentCount: this._presentCount,
            realPackDrawsThisFrame: summary.realPackDrawsThisFrame,
            fallbackDrawsThisFrame: summary.fallbackDrawsThisFrame,
            drawsThisFrame: summary.drawsThisFrame,
            maxLuma: this._lastSampledMaxLuma,
            suppressedBlackFrames: this._suppressedBlackFrames,
            gpuAdapter: this._gpuAdapter,
          });
        } else {
          if (syncGate && syncGate.evaluated && !syncGate.visible) {
            this._suppressedBlackFrames += 1;
          } else if (
            !swapchainProbeActive &&
            this._requireNonBlackFrame &&
            summary.hasNonClearDraw
          ) {
            this._suppressedBlackFrames += 1;
          }
          if (
            swapchainProbeActive &&
            !this._firstVisibleFrameNotified &&
            pendingSwapchainProbe !== null
          ) {
            this._startSwapchainProbeReadback(pendingSwapchainProbe);
          }
        }
      } else if (
        (summary.hasNonClearDraw || swapchainReportedFirstVisible) &&
        !this._firstVisibleFrameNotified
      ) {
        const firstVisibleFrameMs =
          activeSwapchain &&
          typeof activeSwapchain.firstVisibleFrameMs === 'number' &&
          Number.isFinite(activeSwapchain.firstVisibleFrameMs)
            ? activeSwapchain.firstVisibleFrameMs
            : activeSwapchain &&
                typeof activeSwapchain.firstVisiblePresentMs === 'number' &&
                Number.isFinite(activeSwapchain.firstVisiblePresentMs)
              ? activeSwapchain.firstVisiblePresentMs
              : presentTs;
        const firstPresentMs =
          this._firstPresentMs !== null ? this._firstPresentMs : firstVisibleFrameMs;
        const blackLoadingMs = Math.max(0, firstVisibleFrameMs - firstPresentMs);
        this._firstVisibleFrameMs = firstVisibleFrameMs;
        this._blackLoadingMs = blackLoadingMs;
        this._notifyFirstVisibleFrame({
          type: 'first-visible-frame',
          firstVisibleFrame: true,
          firstVisibleFrameMs,
          firstPresentMs,
          blackLoadingMs,
          presentCount: this._presentCount,
          realPackDrawsThisFrame: summary.realPackDrawsThisFrame,
          fallbackDrawsThisFrame: summary.fallbackDrawsThisFrame,
          drawsThisFrame: summary.drawsThisFrame,
          gpuAdapter: this._gpuAdapter,
        });
      }

      if (
        this.gpuRecorder &&
        typeof this.gpuRecorder.isSpikeTriggerEnabled === 'function' &&
        this.gpuRecorder.isSpikeTriggerEnabled() &&
        typeof this.gpuRecorder.endFrame === 'function'
      ) {
        try {
          const frameDurationMs =
            typeof this._frameStartMs === 'number' && Number.isFinite(this._frameStartMs)
              ? Math.max(0, presentTs - this._frameStartMs)
              : 0;
          this.gpuRecorder.endFrame(frameDurationMs);
        } catch (_) {
          // Recorder spike auto-capture errors must never break presentation.
        }
      }
      this._frameStartMs = null;
    }

    // Release per-frame GPU object references held by presentState / passControl before returning.
    ps.encoder = null;
    ps.activeSwapchain = null;
    ps.swapchainTexture = null;
    ps.swapchainView = null;
    ps.cachedFrameSummary = null;
    passControl.encoder = null;
    passControl.swapchainTexture = null;
    passControl.swapchainView = null;

    const hasDeferredDestroys = this._deferredGpuDestroys.length > 0;
    const needsWorkDoneFence =
      submitted &&
      (hasDeferredDestroys ||
        Boolean(
          this.uniformRing &&
            !this.uniformRing._destroyed &&
            typeof this.uniformRing.endFrame === 'function'
        ) ||
        Boolean(
          this.resourceRegistry &&
            !this.resourceRegistry._destroyed &&
            typeof this.resourceRegistry.endFrame === 'function'
        ));
    const workDoneFence = needsWorkDoneFence ? this._getWorkDoneFence() : null;
    if (hasDeferredDestroys) {
      this._drainDeferredGpuDestroys(
        workDoneFence,
        !submitted || workDoneFence === null
          ? DRAIN_DEFERRED_OPTS_IMMEDIATE
          : DRAIN_DEFERRED_OPTS_DEFERRED
      );
    }
    const frameHints = this._frameHintsScratch;
    frameHints.sceneTransition = presentSceneTransition;
    frameHints.resolutionChanged = presentResolutionChanged;
    frameHints.isIdleFrame = presentIsIdleFrame;
    this._endSubsystemsForFrame(workDoneFence, frameHints);

    if (submitted) {
      this._advanceUniformRingForFrame(this.frameCount);
    }

    if (
      submitted &&
      timerResolved &&
      this.passTimer &&
      typeof this.passTimer.collectCompletedReadback === 'function'
    ) {
      const readbackPromise = Promise.resolve(this.passTimer.collectCompletedReadback()).then(
        (timing) => {
          if (timing && !this._destroyed) {
            this._lastPassTimings = timing;
          }
          return timing;
        },
        () => null
      );
      if (this._pendingTimerReadbacks.size >= 64) {
        const oldestPromise = this._pendingTimerReadbacks.values().next().value;
        if (oldestPromise !== undefined) {
          this._pendingTimerReadbacks.delete(oldestPromise);
        }
      }
      this._pendingTimerReadbacks.add(readbackPromise);
      readbackPromise.finally(() => {
        this._pendingTimerReadbacks.delete(readbackPromise);
      });
    }

    this._resetFrameState();
  }

  attachRing(ringOrSab, options = {}) {
    if (
      ringOrSab &&
      (typeof ringOrSab.drainBatchZeroAlloc === 'function' ||
        typeof ringOrSab.drainTo === 'function')
    ) {
      this.ring = ringOrSab;
      return this.ring;
    }
    const RingCtor = resolveGpuRingBuffer(options.GpuRingBuffer || this._GpuRingBuffer);
    const ctorOptions = {};
    if (options.byteOffset !== undefined) {
      ctorOptions.byteOffset = options.byteOffset;
    }
    if (options.initialize !== undefined) {
      ctorOptions.initialize = options.initialize;
    }
    if (options.capacityBytes !== undefined) {
      ctorOptions.capacityBytes = options.capacityBytes;
    }
    this.ring = new RingCtor(ringOrSab, ctorOptions);
    return this.ring;
  }

  drainRing(options = null) {
    const reuseResult = Boolean(options && options.reuseResult === true);
    if (this._destroyed || !this.ring || !this.device || !this.ctx) {
      if (reuseResult) {
        const out = this._drainRingResultScratch;
        out.frameCount = this.frameCount;
        out.presents = 0;
        out.bytesDrained = 0;
        return out;
      }
      return { frameCount: this.frameCount, presents: 0, bytesDrained: 0 };
    }
    let presents = 0;
    // W284 / W359: Prefer zero-alloc direct-view span draining when supported by the attached ring.
    if (typeof this.ring.drainBatchZeroAlloc === 'function') {
      let bytesDrained = 0;
      for (let batch = 0; batch < MAX_ZERO_ALLOC_RING_DRAIN_BATCHES; batch += 1) {
        this._zeroAllocDrainPresents = 0;
        this._zeroAllocDrainSpanBytes = 0;
        const res = this.ring.drainBatchZeroAlloc(
          this._onZeroAllocDrainSpan,
          this._zeroAllocDrainOpts
        );
        presents += this._zeroAllocDrainPresents;
        const batchBytes =
          res && typeof res === 'object' && typeof res.bytes === 'number'
            ? res.bytes
            : typeof res === 'number' && Number.isFinite(res)
              ? res
              : this._zeroAllocDrainSpanBytes;
        if (batchBytes > 0) {
          bytesDrained += batchBytes;
          this._zeroAllocDrains += 1;
        }
        const remainingBytes =
          res && typeof res === 'object' && typeof res.remainingBytes === 'number'
            ? res.remainingBytes
            : 0;
        if (batchBytes <= 0 || remainingBytes <= 0) {
          break;
        }
      }
      if (this._readbackFlushWanted) {
        this._flushReadbacksNow();
      }
      if (reuseResult) {
        const out = this._drainRingResultScratch;
        out.frameCount = this.frameCount;
        out.presents = presents;
        out.bytesDrained = bytesDrained;
        return out;
      }
      return { frameCount: this.frameCount, presents, bytesDrained };
    }
    const coalesceRing = this._coalescePresentsEnabled !== false;
    const bytesDrained = this.ring.drainTo((bytes) => {
      if (this.executeCommandBuffer !== WgpuCommandProcessor.prototype.executeCommandBuffer) {
        const result = this.executeCommandBuffer(bytes, DRAIN_RING_REUSE_OPTS);
        presents += result.presents;
        return;
      }
      const view = this._getCachedDataView(bytes.buffer);
      presents += this._executeCommandView(
        view,
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
        coalesceRing,
        false
      );
    });
    if (this._readbackFlushWanted) {
      this._flushReadbacksNow();
    }
    if (reuseResult) {
      const out = this._drainRingResultScratch;
      out.frameCount = this.frameCount;
      out.presents = presents;
      out.bytesDrained = bytesDrained;
      return out;
    }
    return { frameCount: this.frameCount, presents, bytesDrained };
  }

  /**
   * Wave-4 (#663): Zero-alloc header walk counting OPCODES.PRESENT packets in [startOffset, endOffset).
   */
  _scanCommandViewPresents(view, startOffset, endOffset) {
    const out = this._presentScanScratch;
    out.count = 0;
    out.lastOffset = -1;
    if (!view) {
      return out;
    }
    let offset = startOffset;
    while (offset + 8 <= endOffset) {
      const opcode = view.getUint16(offset, true);
      const payloadBytes = view.getUint32(offset + 4, true);
      const payloadOffset = offset + 8;
      if (payloadOffset + payloadBytes > endOffset) {
        break;
      }
      if (opcode === OPCODES.PRESENT) {
        out.count += 1;
        out.lastOffset = offset;
      }
      offset = payloadOffset + payloadBytes;
    }
    return out;
  }

  /**
   * Wave-4 (#663): Coalesce a superseded intermediate Present without acquiring the swapchain
   * texture or issuing a redundant swapchain blit. Preserves all staging uploads and compute
   * dispatches recorded before this Present, then resets per-frame pass state.
   */
  _coalesceIntermediatePresent() {
    if (this._pendingEncoder && this._hasPendingUploadCommands) {
      this._flushPendingUploads(false);
    } else if (this._pendingEncoder) {
      this._pendingEncoder = null;
      this._hasPendingUploadCommands = false;
    }

    const cmds = this._passCommands;
    let hasComputeDispatch = false;
    for (let i = 0, n = cmds.length; i < n; i += 1) {
      const t = cmds[i].type;
      if (t === 'dispatchCompute' || t === 'dispatchIndirect') {
        hasComputeDispatch = true;
        break;
      }
    }

    if (hasComputeDispatch && this.device && typeof this.device.createCommandEncoder === 'function') {
      try {
        const encoder = this.device.createCommandEncoder();
        let computePass = null;
        let activeComputePipelineId = null;
        const ensureCompute = () => {
          if (!computePass) {
            computePass = encoder.beginComputePass();
          }
          return computePass;
        };
        for (let i = 0, n = cmds.length; i < n; i += 1) {
          const cmd = cmds[i];
          if (cmd.type === 'setComputePipeline') {
            activeComputePipelineId = cmd.pipelineId;
            const pipeline = this.computePipelines.get(cmd.pipelineId);
            if (pipeline) {
              ensureCompute().setPipeline(pipeline);
            }
          } else if (cmd.type === 'setBindGroup') {
            if (computePass || this._bindGroupKinds.get(cmd.bindGroupId) === 'compute') {
              const bg = this.bindGroups.get(cmd.bindGroupId) || cmd.bindGroup || null;
              if (bg) {
                ensureCompute().setBindGroup(cmd.groupIndex, bg);
              }
            }
          } else if (cmd.type === 'dispatchCompute') {
            if (activeComputePipelineId !== null && this.computePipelines.get(activeComputePipelineId)) {
              ensureCompute().dispatchWorkgroups(cmd.workgroupX, cmd.workgroupY, cmd.workgroupZ);
              this._computeDispatchCount += 1;
            }
          } else if (cmd.type === 'dispatchIndirect') {
            const buf = this.buffers.get(cmd.bufferId);
            if (
              buf &&
              activeComputePipelineId !== null &&
              this.computePipelines.get(activeComputePipelineId)
            ) {
              ensureCompute().dispatchWorkgroupsIndirect(buf, cmd.offset);
              this._computeDispatchCount += 1;
            }
          }
        }
        if (computePass && typeof computePass.end === 'function') {
          computePass.end();
        }
        const cb = encoder.finish();
        if (cb && this.device.queue && typeof this.device.queue.submit === 'function') {
          const submitStartMs = this._gpuStallWarnMs > 0 ? this._getNowMs() : 0;
          this.device.queue.submit([cb]);
          if (this._gpuStallWarnMs > 0) {
            this._maybeWarnGpuStall('queue.submit', submitStartMs, null, 0, this._computeDispatchCount);
          }
        }
      } catch (err) {
        this._notePresentSubmitFailure(err);
      }
    }

    if (this._deferredGpuDestroys.length > 0) {
      this._drainDeferredGpuDestroys(null, DRAIN_DEFERRED_OPTS_DEFERRED);
    }
    this._resetFrameState();
    this._coalescedPresents += 1;
  }

  _executeCommandView(view, startOffset, endOffset, coalescePresents = false, deferFinalPresent = false) {
    if (this._deviceRecovering) {
      // W100: drop submissions for the lost device while recovery is in flight.
      this._droppedWhileRecoveringCount += 1;
      // W104 / W346: resources the guest creates meanwhile are still recorded for replay
      // only when resource replay is enabled.
      if (view && this._resourceReplayEnabled && !this._replayingResources) {
        this._recordReplayPacketsFrom(view, startOffset, endOffset);
      }
      return 0;
    }

    let offset = startOffset;
    let presents = 0;

    let lastPresentOffset = -1;
    let shouldCoalesceIntermediate = false;
    if (deferFinalPresent || coalescePresents) {
      const scan = this._scanCommandViewPresents(view, startOffset, endOffset);
      if (deferFinalPresent) {
        shouldCoalesceIntermediate = scan.count >= 1;
        lastPresentOffset = -1;
      } else if (scan.count > 1) {
        shouldCoalesceIntermediate = true;
        lastPresentOffset = scan.lastOffset;
      }
    }

    const prevPendingHints = this._pendingCommandStreamShaderProfiles;
    const prevPendingView = this._pendingCommandStreamView;
    const prevPendingStart = this._pendingCommandStreamStartOffset;
    const prevPendingEnd = this._pendingCommandStreamEndOffset;
    const prevPendingScanned = this._pendingCommandStreamScanned;
    // W346: Defer fallback profile lookahead scanning until a CreateShaderModule packet
    // actually misses the WGSL pack; steady-state & pack-backed submissions walk once.
    this._pendingCommandStreamShaderProfiles = null;
    this._pendingCommandStreamView = view;
    this._pendingCommandStreamStartOffset = startOffset;
    this._pendingCommandStreamEndOffset = endOffset;
    this._pendingCommandStreamScanned = false;
    const profileOps = this._stallProfileActive === true;
    let detailedTiming = profileOps && this._stallProfileDetailed === true;
    try {
      while (offset + 8 <= endOffset) {
        const opcode = view.getUint16(offset, true);
        const payloadBytes = view.getUint32(offset + 4, true);
        const payloadOffset = offset + 8;
        if (payloadOffset + payloadBytes > endOffset) {
          this._recordOpValidationError(
            'CommandStream',
            `truncated packet ${opcodeLabel(opcode)} (need ${payloadBytes}B, remaining ${endOffset - payloadOffset}B)`
          );
          break;
        }

        const totalPktBytes = 8 + payloadBytes;
        this._packetCount += 1;
        this._bytesProcessed += totalPktBytes;

        const slot = opcode <= 63 ? opcode : 64;
        if (profileOps) {
          this._stallBatchPackets += 1;
          this._stallBatchBytes += totalPktBytes;
          this._stallOpCount[slot] += 1;
          this._stallOpBytes[slot] += totalPktBytes;
          if (!detailedTiming && (this._stallBatchPackets & 127) === 0) {
            if (this._getNowMs() - this._stallExecStartMs >= 250) {
              detailedTiming = true;
              this._stallProfileDetailed = true;
            }
          }
        }

        const tPktStart = detailedTiming ? this._getNowMs() : 0;
        if (this._resourceReplayEnabled && !this._replayingResources) {
          this._recordReplayPacket(opcode, view, payloadOffset, payloadBytes);
        }
        if (
          opcode === OPCODES.PRESENT &&
          shouldCoalesceIntermediate &&
          (deferFinalPresent || offset < lastPresentOffset) &&
          (!Array.isArray(this._recordedReadbacks) || this._recordedReadbacks.length === 0)
        ) {
          this._coalesceIntermediatePresent();
        } else {
          presents += this._dispatchPacket(opcode, view, payloadOffset, payloadBytes);
        }
        if (detailedTiming) {
          const tPktEnd = this._getNowMs();
          const opDur = tPktEnd - tPktStart;
          this._stallOpMs[slot] += opDur;
          if (opDur > this._stallOpMaxMs[slot]) {
            this._stallOpMaxMs[slot] = opDur;
          }
          if (opDur > this._gpuStallWarnMs && this._stallSlowPacketLogs < 10) {
            this._stallSlowPacketLogs += 1;
            const resId = payloadBytes >= 4 ? view.getUint32(payloadOffset, true) : 0;
            let extra = '';
            if (opcode === OPCODES.WRITE_TEXTURE && payloadBytes >= 48) {
              const wtW = view.getUint32(payloadOffset + 20, true);
              const wtH = view.getUint32(payloadOffset + 24, true);
              const wtD = view.getUint32(payloadOffset + 28, true);
              const wtBpr = view.getUint32(payloadOffset + 32, true);
              const wtRpi = view.getUint32(payloadOffset + 36, true);
              const wtLen = view.getUint32(payloadOffset + 40, true);
              const tMeta = this._textureMeta.get(resId);
              const fmt = tMeta ? tMeta.format : '?';
              const aPipe = this.asyncPipelineCache;
              const pIn = aPipe && typeof aPipe._inFlight === 'number' ? aPipe._inFlight : 0;
              const pQ = aPipe && Array.isArray(aPipe._queue) ? aPipe._queue.length : 0;
              extra = ` tex=${wtW}x${wtH}x${wtD} fmt=${fmt} bpr=${wtBpr} rpi=${wtRpi} dataLen=${wtLen} pipeInFlight=${pIn} pipeQ=${pQ} shActive=${this._activeRangedShaderCompiles} shQ=${this._rangedShaderCompileQueue.length}`;
            }
            this._logDiagnostic(
              'warn',
              `[wgpu_worker] slow packet: op=${opcodeLabel(opcode)} durMs=${Math.round(opDur)} payloadBytes=${payloadBytes} id=${resId} pktIdx=${this._stallBatchPackets} frame=${this.frameCount}${extra}`
            );
          }
          if (tPktEnd - this._stallLastProgressMs >= Math.max(5000, this._gpuStallWarnMs * 5)) {
            this._stallLastProgressMs = tPktEnd;
            this._logDiagnostic(
              'warn',
              `[wgpu_worker] GPU stall progress: elapsedMs=${Math.round(tPktEnd - this._stallExecStartMs)} offset=${offset}/${endOffset} curOp=${opcodeLabel(opcode)} ${this._formatStallProfileSummary()}`
            );
          }
        }

        offset = payloadOffset + payloadBytes;
        if (this._deviceRecovering) {
          // W100: device loss detected mid-buffer; skip the remaining packets cleanly.
          // W104 / W346: still record the skipped resource packets when replay is enabled.
          if (this._resourceReplayEnabled && !this._replayingResources) {
            this._recordReplayPacketsFrom(view, offset, endOffset);
          }
          break;
        }
      }
    } finally {
      this._pendingCommandStreamShaderProfiles = prevPendingHints;
      this._pendingCommandStreamView = prevPendingView;
      this._pendingCommandStreamStartOffset = prevPendingStart;
      this._pendingCommandStreamEndOffset = prevPendingEnd;
      this._pendingCommandStreamScanned = prevPendingScanned;
    }
    if (this._readbackFlushWanted) {
      this._flushReadbacksNow();
    }

    return presents;
  }

  _resolveCommandInputSlice(input, byteLengthOpt) {
    if (input instanceof DataView) {
      return { view: input, startOffset: 0, endOffset: input.byteLength };
    }
    if (
      input instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer)
    ) {
      const view = this._getCachedDataView(input);
      const optLen = typeof byteLengthOpt === 'number' ? byteLengthOpt : -1;
      const endOffset =
        optLen >= 0 && optLen <= input.byteLength && Number.isInteger(optLen)
          ? optLen
          : input.byteLength;
      return { view, startOffset: 0, endOffset };
    }
    if (ArrayBuffer.isView(input)) {
      const view = this._getCachedDataView(input.buffer);
      const startOffset = input.byteOffset;
      return { view, startOffset, endOffset: startOffset + input.byteLength };
    }
    return null;
  }

  executeCommandBuffer(input, options = null) {
    const reuseResult = Boolean(options && options.reuseResult === true);
    if (this._destroyed || !this.device || !this.ctx) {
      if (reuseResult) {
        const out = this._executeResultScratch;
        out.frameCount = this.frameCount;
        out.presents = 0;
        return out;
      }
      return { frameCount: this.frameCount, presents: 0 };
    }
    let view;
    let startOffset = 0;
    let endOffset = 0;
    if (input instanceof DataView) {
      view = input;
      startOffset = 0;
      endOffset = input.byteLength;
    } else if (
      input instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer)
    ) {
      view = this._getCachedDataView(input);
      startOffset = 0;
      const optLen =
        options && typeof options.byteLength === 'number' ? options.byteLength : -1;
      endOffset =
        optLen >= 0 && optLen <= input.byteLength && Number.isInteger(optLen)
          ? optLen
          : input.byteLength;
    } else if (ArrayBuffer.isView(input)) {
      view = this._getCachedDataView(input.buffer);
      startOffset = input.byteOffset;
      endOffset = startOffset + input.byteLength;
    } else if (this._deviceRecovering) {
      view = null;
    } else {
      throw new TypeError(
        'WgpuCommandProcessor.executeCommandBuffer expects an ArrayBuffer, SharedArrayBuffer, or ArrayBufferView'
      );
    }

    const deferFinalPresent = Boolean(options && options.deferFinalPresent === true);
    let coalescePresents = false;
    if (options && typeof options.coalescePresents === 'boolean') {
      coalescePresents = options.coalescePresents;
    } else if (options && typeof options.coalescePresent === 'boolean') {
      coalescePresents = options.coalescePresent;
    } else if (!deferFinalPresent && this._coalescePresentsEnabled && view && endOffset - startOffset >= 24) {
      // Coalesce stall backlogs of >= 3 Presents packed into a single buffer while keeping
      // explicit 2-frame command-buffer unit tests intact.
      const scan = this._scanCommandViewPresents(view, startOffset, endOffset);
      if (scan.count >= 3) {
        coalescePresents = true;
      }
    }

    const execStartMs = this._gpuStallWarnMs > 0 ? this._getNowMs() : 0;
    const drawsBefore = this._drawCallCount;
    const dispatchesBefore = this._computeDispatchCount;
    const prevProfileActive = this._stallProfileActive;
    const prevProfileDetailed = this._stallProfileDetailed;
    const shouldProfileStall = this._gpuStallWarnMs > 0 && !prevProfileActive;
    if (shouldProfileStall) {
      this._resetStallProfileScratch();
      this._stallProfileActive = true;
      this._stallProfileDetailed = endOffset - startOffset >= 512 * 1024;
      this._stallExecStartMs = execStartMs;
      this._stallLastProgressMs = execStartMs;
    }
    let presents = 0;
    try {
      presents = this._executeCommandView(
        view,
        startOffset,
        endOffset,
        coalescePresents,
        deferFinalPresent
      );
    } finally {
      this._stallProfileActive = prevProfileActive;
      this._stallProfileDetailed = prevProfileDetailed;
    }
    if (this._gpuStallWarnMs > 0) {
      const execElapsedMs = this._getNowMs() - execStartMs;
      if (execElapsedMs > this._gpuStallWarnMs) {
        const detail = shouldProfileStall ? this._formatStallProfileSummary() : null;
        this._maybeWarnGpuStall(
          'executeCommandBuffer',
          execStartMs,
          null,
          this._drawCallCount - drawsBefore,
          this._computeDispatchCount - dispatchesBefore,
          detail
        );
      }
    }
    if (reuseResult) {
      const out = this._executeResultScratch;
      out.frameCount = this.frameCount;
      out.presents = presents;
      return out;
    }
    return { frameCount: this.frameCount, presents };
  }

  /**
   * Wave-4 (#663): Execute a backlog of queued command batches in order while coalescing all
   * redundant intermediate Presents across the queue into at most one final swapchain Present.
   * All non-Present resource/state commands from every batch execute in exact order.
   */
  executeCommandBatches(batches, options = null) {
    if (this._destroyed || !this.device || !this.ctx || !Array.isArray(batches) || batches.length === 0) {
      return { frameCount: this.frameCount, presents: 0, coalescedPresents: 0 };
    }
    const coalesce =
      options && typeof options.coalescePresents === 'boolean'
        ? options.coalescePresents
        : this._coalescePresentsEnabled !== false;
    const beforeCoalesced = this._coalescedPresents;

    const slices = new Array(batches.length);
    let lastPresentBatchIdx = -1;
    for (let i = 0; i < batches.length; i += 1) {
      const item = batches[i];
      const raw =
        item && typeof item === 'object' && !(item instanceof ArrayBuffer) && !ArrayBuffer.isView(item)
          ? item.cmdBuffer || item.buffer || item.commands || item.data || null
          : item;
      const byteLen =
        item && typeof item === 'object' && typeof item.byteLength === 'number'
          ? item.byteLength
          : -1;
      const resolved = raw ? this._resolveCommandInputSlice(raw, byteLen) : null;
      slices[i] = resolved;
      if (coalesce && resolved && resolved.view) {
        const scan = this._scanCommandViewPresents(
          resolved.view,
          resolved.startOffset,
          resolved.endOffset
        );
        if (scan.count > 0) {
          lastPresentBatchIdx = i;
        }
      }
    }

    let totalPresents = 0;
    for (let i = 0; i < slices.length; i += 1) {
      const s = slices[i];
      if (!s || !s.view) {
        continue;
      }
      const deferFinal = coalesce && lastPresentBatchIdx >= 0 && i < lastPresentBatchIdx;
      const coalesceInternal = coalesce && lastPresentBatchIdx >= 0 && i <= lastPresentBatchIdx;
      totalPresents += this._executeCommandView(
        s.view,
        s.startOffset,
        s.endOffset,
        coalesceInternal,
        deferFinal
      );
    }

    return {
      frameCount: this.frameCount,
      presents: totalPresents,
      coalescedPresents: Math.max(0, this._coalescedPresents - beforeCoalesced),
    };
  }

  /**
   * Dispatch one packet of a command stream. Returns the number of presents (0 or 1).
   */
  _dispatchPacket(opcode, view, payloadOffset, payloadBytes) {
    switch (opcode) {
      case OPCODES.NOP:
        break;
      case OPCODES.BEGIN_FRAME:
        if (this._pendingEncoder && this._frameBegunForIndex === -1) {
          this._flushPendingUploads(true);
        }
        this._resetFrameState();
        if (this._frameBegunForIndex === this.frameCount) {
          if (this.drawExecutor && typeof this.drawExecutor.beginFrame === 'function') {
            this.drawExecutor.beginFrame();
          }
        } else {
          this._ensureFrameBegun();
        }
        break;
      case OPCODES.CLEAR_COLOR:
        if (payloadBytes >= 16) {
          const r = view.getFloat32(payloadOffset, true);
          const g = view.getFloat32(payloadOffset + 4, true);
          const b = view.getFloat32(payloadOffset + 8, true);
          const a = view.getFloat32(payloadOffset + 12, true);
          if (
            r === DEFAULT_CLEAR_COLOR.r &&
            g === DEFAULT_CLEAR_COLOR.g &&
            b === DEFAULT_CLEAR_COLOR.b &&
            a === DEFAULT_CLEAR_COLOR.a
          ) {
            this._clearColor = DEFAULT_CLEAR_COLOR;
          } else if (
            this._lastCustomClearColor !== null &&
            r === this._lastCustomClearColor.r &&
            g === this._lastCustomClearColor.g &&
            b === this._lastCustomClearColor.b &&
            a === this._lastCustomClearColor.a
          ) {
            this._clearColor = this._lastCustomClearColor;
          } else {
            this._lastCustomClearColor = Object.freeze({ r, g, b, a });
            this._clearColor = this._lastCustomClearColor;
          }
        } else {
          this._recordOpValidationError('ClearColor', `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      case OPCODES.CREATE_BUFFER:
        if (payloadBytes >= 16) {
          this._handleCreateBuffer(view, payloadOffset);
        } else {
          this._recordOpValidationError('CreateBuffer', `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      case OPCODES.WRITE_BUFFER:
        if (payloadBytes >= 16) {
          this._handleWriteBuffer(view, payloadOffset, payloadBytes);
        } else {
          this._recordOpValidationError('WriteBuffer', `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      case OPCODES.CREATE_SHADER_MODULE:
        if (payloadBytes >= 16) {
          this._handleCreateShaderModule(view, payloadOffset, payloadBytes);
        } else {
          this._recordOpValidationError(
            'CreateShaderModule',
            `truncated payload (${payloadBytes}B < 16B)`
          );
        }
        break;
      case OPCODES.CREATE_RENDER_PIPELINE:
        if (payloadBytes >= 24) {
          this._handleCreateRenderPipeline(view, payloadOffset);
        } else {
          this._recordOpValidationError(
            'CreateRenderPipeline',
            `truncated payload (${payloadBytes}B < 24B)`
          );
        }
        break;
      case OPCODES.SET_PIPELINE:
        if (payloadBytes >= 4) {
          const pipelineId = view.getUint32(payloadOffset, true);
          const cmd = this._acquirePassCommand('setPipeline');
          cmd.pipelineId = pipelineId;
          cmd.pipeline = this.pipelines.get(pipelineId) || null;
          this._passCommands.push(cmd);
        } else {
          this._recordOpValidationError('SetPipeline', `truncated payload (${payloadBytes}B < 4B)`);
        }
        break;
      case OPCODES.SET_VERTEX_BUFFER:
        if (payloadBytes >= 16) {
          const cmd = this._acquirePassCommand('setVertexBuffer');
          cmd.slot = view.getUint32(payloadOffset, true);
          cmd.bufferId = view.getUint32(payloadOffset + 4, true);
          cmd.offset = readUint64AsNumber(view, payloadOffset + 8);
          this._passCommands.push(cmd);
        } else {
          this._recordOpValidationError('SetVertexBuffer', `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      case OPCODES.DRAW:
        if (payloadBytes >= 16) {
          const cmd = this._acquirePassCommand('draw');
          cmd.vertexCount = view.getUint32(payloadOffset, true);
          cmd.instanceCount = view.getUint32(payloadOffset + 4, true);
          cmd.firstVertex = view.getUint32(payloadOffset + 8, true);
          cmd.firstInstance = view.getUint32(payloadOffset + 12, true);
          this._passCommands.push(cmd);
        } else {
          this._recordOpValidationError('Draw', `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      case OPCODES.PRESENT:
        this._flushPresent();
        return 1;
      case OPCODES.DESTROY_RESOURCE:
        if (payloadBytes >= 4) {
          this._handleDestroyResource(view.getUint32(payloadOffset, true));
        }
        break;
      case OPCODES.CREATE_TEXTURE:
        if (payloadBytes >= 32) {
          this._handleCreateTexture(view, payloadOffset);
          if (this._handleCreateTexture !== WgpuCommandProcessor.prototype._handleCreateTexture) {
            const texId = view.getUint32(payloadOffset, true);
            this._reconcileTrackedTextureAfterExternalCreate(texId, view, payloadOffset, false);
          }
        } else {
          this._recordOpValidationError('CreateTexture', `truncated payload (${payloadBytes}B < 32B)`);
        }
        break;
      case OPCODES.CREATE_TEXTURE2:
        if (this.extension) {
          this._dispatchExtensionPacket(opcode, view, payloadOffset, payloadBytes);
        } else if (payloadBytes >= 40) {
          this._handleCreateTexture2(view, payloadOffset);
        } else {
          this._recordOpValidationError('CreateTexture2', `truncated payload (${payloadBytes}B < 40B)`);
        }
        break;
      case OPCODES.WRITE_TEXTURE:
        if (payloadBytes >= 48) {
          this._handleWriteTexture(view, payloadOffset, payloadBytes);
        } else {
          this._recordOpValidationError('WriteTexture', `truncated payload (${payloadBytes}B < 48B)`);
        }
        break;
      case OPCODE_CLEAR_TEXTURE:
        if (payloadBytes >= CLEAR_TEXTURE_PAYLOAD_BYTES) {
          if (this.extension) {
            const handled = this._dispatchExtensionPacket(opcode, view, payloadOffset, payloadBytes);
            if (!handled) {
              this._handleClearTexture(view, payloadOffset);
            }
          } else {
            this._handleClearTexture(view, payloadOffset);
          }
        } else {
          this._recordOpValidationError(
            'ClearTexture',
            `truncated payload (${payloadBytes}B < ${CLEAR_TEXTURE_PAYLOAD_BYTES}B)`
          );
        }
        break;
      case OPCODES.CREATE_SAMPLER:
        if (payloadBytes >= 32) {
          this._handleCreateSampler(view, payloadOffset);
        } else {
          this._recordOpValidationError('CreateSampler', `truncated payload (${payloadBytes}B < 32B)`);
        }
        break;
      case OPCODES.CREATE_BIND_GROUP:
        if (payloadBytes >= 16) {
          this._handleCreateBindGroup(view, payloadOffset, payloadBytes);
        } else {
          this._recordOpValidationError(
            'CreateBindGroup',
            `truncated payload (${payloadBytes}B < 16B)`
          );
        }
        break;
      case OPCODES.SET_BIND_GROUP:
        if (payloadBytes >= 8) {
          const groupIndex = view.getUint32(payloadOffset, true);
          const bindGroupId = view.getUint32(payloadOffset + 4, true);
          this._touchReplayBindGroup(bindGroupId);
          const cmd = this._acquirePassCommand('setBindGroup');
          cmd.groupIndex = groupIndex;
          cmd.bindGroupId = bindGroupId;
          cmd.bindGroup = this.bindGroups.get(bindGroupId) || null;
          this._passCommands.push(cmd);
        } else {
          this._recordOpValidationError('SetBindGroup', `truncated payload (${payloadBytes}B < 8B)`);
        }
        break;
      case OPCODES.SET_INDEX_BUFFER:
        if (payloadBytes >= 16) {
          const formatCode = view.getUint32(payloadOffset + 4, true);
          const cmd = this._acquirePassCommand('setIndexBuffer');
          cmd.bufferId = view.getUint32(payloadOffset, true);
          cmd.indexFormat = INDEX_FORMAT_NAMES[formatCode] || 'uint16';
          cmd.offset = readUint64AsNumber(view, payloadOffset + 8);
          this._passCommands.push(cmd);
        } else {
          this._recordOpValidationError('SetIndexBuffer', `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      case OPCODES.DRAW_INDEXED:
        if (payloadBytes >= 20) {
          const cmd = this._acquirePassCommand('drawIndexed');
          cmd.indexCount = view.getUint32(payloadOffset, true);
          cmd.instanceCount = view.getUint32(payloadOffset + 4, true);
          cmd.firstIndex = view.getUint32(payloadOffset + 8, true);
          cmd.baseVertex = view.getInt32(payloadOffset + 12, true);
          cmd.firstInstance = view.getUint32(payloadOffset + 16, true);
          this._passCommands.push(cmd);
        } else {
          this._recordOpValidationError('DrawIndexed', `truncated payload (${payloadBytes}B < 20B)`);
        }
        break;
      case OPCODES.CREATE_COMPUTE_PIPELINE:
        if (payloadBytes >= 16) {
          const cpId = view.getUint32(payloadOffset, true);
          const csShaderId = view.getUint32(payloadOffset + 4, true);
          const pendingRanged = this._registerPendingRangedPipeline(
            cpId,
            'compute',
            OPCODES.CREATE_COMPUTE_PIPELINE,
            [csShaderId],
            view,
            payloadOffset,
            16
          );
          if (!pendingRanged.deferredOnly) {
            this._handleCreateComputePipeline(view, payloadOffset);
          }
        } else {
          this._recordOpValidationError(
            'CreateComputePipeline',
            `truncated payload (${payloadBytes}B < 16B)`
          );
        }
        break;
      case OPCODES.SET_COMPUTE_PIPELINE:
        if (payloadBytes >= 4) {
          const cmd = this._acquirePassCommand('setComputePipeline');
          cmd.pipelineId = view.getUint32(payloadOffset, true);
          this._passCommands.push(cmd);
        }
        break;
      case OPCODES.READBACK_BUFFER:
      case OPCODES.READBACK_TEXTURE:
        this._handleReadback(opcode, view, payloadOffset, payloadBytes);
        break;
      case OPCODES.DISPATCH_COMPUTE:
        if (payloadBytes >= 12) {
          const cmd = this._acquirePassCommand('dispatchCompute');
          cmd.workgroupX = view.getUint32(payloadOffset, true);
          cmd.workgroupY = view.getUint32(payloadOffset + 4, true);
          cmd.workgroupZ = view.getUint32(payloadOffset + 8, true);
          this._passCommands.push(cmd);
        }
        break;
      case OPCODES.DRAW_INDIRECT:
      case OPCODES.DRAW_INDEXED_INDIRECT:
      case OPCODES.DISPATCH_INDIRECT:
        // G66 ExecuteIndirect: GpuIndirectPayload { u32 buffer_id, u32 reserved, u64 offset }.
        if (payloadBytes >= 16) {
          const cmd = this._acquirePassCommand(
            opcode === OPCODES.DRAW_INDIRECT
              ? 'drawIndirect'
              : opcode === OPCODES.DRAW_INDEXED_INDIRECT
                ? 'drawIndexedIndirect'
                : 'dispatchIndirect'
          );
          cmd.bufferId = view.getUint32(payloadOffset, true);
          cmd.offset = readUint64AsNumber(view, payloadOffset + 8);
          this._passCommands.push(cmd);
        } else {
          this._recordOpValidationError(OPCODE_NAMES[opcode], `truncated payload (${payloadBytes}B < 16B)`);
        }
        break;
      default:
        if (this.extension) {
          this._dispatchExtensionPacket(opcode, view, payloadOffset, payloadBytes);
        }
        break;
    }
    return 0;
  }

  execute(input) {
    return this.executeCommandBuffer(input);
  }

  /**
   * W371 (#1077): Accurate live GPU buffer, texture, staging-pool, scratch, and
   * deferred-destroy byte accounting across the processor and wired subsystems.
   */
  getGpuMemoryStats(precomputed = null) {
    const stagingStats =
      precomputed && precomputed.stagingPool !== undefined
        ? precomputed.stagingPool
        : this.stagingPool && typeof this.stagingPool.getStats === 'function'
          ? this.stagingPool.getStats()
          : null;
    const rtStats =
      precomputed && precomputed.resourceTable !== undefined
        ? precomputed.resourceTable
        : this.resourceTable && typeof this.resourceTable.getStats === 'function'
          ? this.resourceTable.getStats()
          : null;

    const resourceTableBufferBytes = toNonNegativeInt(rtStats && rtStats.liveBufferBytes, 0);
    const resourceTableTextureBytes = toNonNegativeInt(rtStats && rtStats.liveTextureBytes, 0);
    const resourceTablePeakBufferBytes = toNonNegativeInt(rtStats && rtStats.peakBufferBytes, 0);
    const resourceTablePeakTextureBytes = toNonNegativeInt(rtStats && rtStats.peakTextureBytes, 0);
    const resourceTableScratchBytes = toNonNegativeInt(rtStats && rtStats.writeScratchBytes, 0);
    const resourceTablePendingDestroys = toNonNegativeInt(
      rtStats && (rtStats.pendingDeferredDestroys ?? rtStats.pendingDeferredDestroyCount),
      0
    );
    const resourceTablePendingBytes = toNonNegativeInt(
      rtStats && rtStats.pendingDeferredDestroyBytes,
      0
    );

    const liveBufferBytes = this._liveBufferBytes + resourceTableBufferBytes;
    const peakBufferBytes = Math.max(
      this._peakBufferBytes + resourceTablePeakBufferBytes,
      liveBufferBytes
    );
    const liveTextureBytes = this._liveTextureBytes + resourceTableTextureBytes;
    const peakTextureBytes = Math.max(
      this._peakTextureBytes + resourceTablePeakTextureBytes,
      liveTextureBytes
    );
    const stagingPoolBytes = toNonNegativeInt(
      stagingStats && (stagingStats.stagingPoolBytes ?? stagingStats.allocatedBytes),
      0
    );
    const stagingPoolPeakBytes = toNonNegativeInt(
      stagingStats && (stagingStats.peakAllocatedBytes ?? stagingStats.peakBytes),
      0
    );
    const stagingScratchBytes = toNonNegativeInt(stagingStats && stagingStats.scratchBytes, 0);
    const directWriteScratchBytes =
      this._directWriteScratch && typeof this._directWriteScratch.byteLength === 'number'
        ? this._directWriteScratch.byteLength
        : 0;
    const pendingDeferredDestroys =
      this._deferredGpuDestroys.length +
      this._inFlightDeferredDestroys.size +
      resourceTablePendingDestroys;
    const pendingDeferredDestroyBytes =
      this._pendingDeferredDestroyBytes + resourceTablePendingBytes;
    const totalLiveGpuBytes = liveBufferBytes + liveTextureBytes + stagingPoolBytes;

    return {
      liveBufferBytes,
      peakBufferBytes,
      liveTextureBytes,
      peakTextureBytes,
      stagingPoolBytes,
      stagingPoolPeakBytes,
      stagingScratchBytes,
      directWriteScratchBytes,
      resourceTableBufferBytes,
      resourceTableTextureBytes,
      resourceTableScratchBytes,
      pendingDeferredDestroys,
      pendingDeferredDestroyBytes,
      deferredHighWaterFlushes: this._deferredHighWaterFlushes,
      totalLiveGpuBytes,
    };
  }

  /**
   * Return aggregated processor and subsystem telemetry (packet counts, frame count,
   * staging pool, dynamic uniform ring, GPU pass timings, VRAM resource registry,
   * render-target pool hit rate, and reflection shader count).
   */
  getStats() {
    const stagingPool =
      this.stagingPool && typeof this.stagingPool.getStats === 'function'
        ? this.stagingPool.getStats()
        : null;
    const uniformRing =
      this.uniformRing && typeof this.uniformRing.getStats === 'function'
        ? this.uniformRing.getStats()
        : null;
    const passTimerBase =
      this.passTimer && typeof this.passTimer.getStats === 'function'
        ? this.passTimer.getStats()
        : null;
    const lastPassTimings = this.getLastPassTimings();
    const passTimer = passTimerBase
      ? { ...passTimerBase, lastTimings: lastPassTimings }
      : null;
    const gpuTimer = resolveGpuTimerTelemetry(this, { passTimer });
    const gpuRecorder = resolveGpuRecorderTelemetry(this, null);
    const resourceRegistry =
      this.resourceRegistry && typeof this.resourceRegistry.getStats === 'function'
        ? this.resourceRegistry.getStats()
        : null;
    const rtPool =
      this.rtPool && typeof this.rtPool.getStats === 'function'
        ? this.rtPool.getStats()
        : null;
    const reflectionRegistry =
      this.reflectionRegistry && typeof this.reflectionRegistry.getStats === 'function'
        ? this.reflectionRegistry.getStats()
        : null;
    const shaderFallback = this.getShaderFallbackStats();
    const missingShaderSummary = this.getMissingShaderSummary({
      limit: DEFAULT_MISSING_SUMMARY_LIMIT,
    });
    const asyncPipelineStats = this.getAsyncPipelineStats();
    const asyncPipelineCompileStats = resolveAsyncPipelineCompileTelemetry(this);
    let pipelineCache = null;
    if (this.pipelineCache && typeof this.pipelineCache.getStats === 'function') {
      try {
        pipelineCache = this.pipelineCache.getStats();
      } catch (_) {
        pipelineCache = null;
      }
    }
    const fallbackSamplersCreated =
      toNonNegativeInt(pipelineCache && pipelineCache.fallbackSamplersCreated, 0) +
      this._fallbackSamplersCreated;
    const fallbackSamplerBindingsSynthesized =
      toNonNegativeInt(pipelineCache && pipelineCache.fallbackSamplerBindingsSynthesized, 0) +
      this._fallbackSamplerBindingsSynthesized;
    const wgslPackCache =
      this.wgslPackCache && typeof this.wgslPackCache.getStats === 'function'
        ? this.wgslPackCache.getStats()
        : null;
    const splashBridge =
      this.splashFrameBridge && typeof this.splashFrameBridge.getStats === 'function'
        ? this.splashFrameBridge.getStats()
        : null;
    const activeBinkPresenter =
      this.binkPresenter ||
      (this.splashFrameBridge && this.splashFrameBridge.presenter) ||
      null;
    const binkPresenter =
      activeBinkPresenter && typeof activeBinkPresenter.getStats === 'function'
        ? activeBinkPresenter.getStats()
        : splashBridge && splashBridge.presenterStats
          ? splashBridge.presenterStats
          : null;
    let swapchain = null;
    if (this._swapchain && typeof this._swapchain.getStats === 'function') {
      try {
        swapchain = this._swapchain.getStats();
      } catch (_) {
        swapchain = null;
      }
    }
    if (
      swapchain &&
      typeof swapchain === 'object' &&
      !this._firstVisibleFrameNotified &&
      (this._swapchainProbeSupported === true || this._requireNonBlackFrame)
    ) {
      swapchain = {
        ...swapchain,
        hasVisibleContent: false,
        firstVisiblePresentIndex: null,
        firstVisiblePresentMs: null,
        visiblePresentsTotal: 0,
      };
    }
    const texturePacketsExt = this._getTexturePacketsExtension();
    let texturePackets = null;
    if (texturePacketsExt && typeof texturePacketsExt.getStats === 'function') {
      try {
        texturePackets = texturePacketsExt.getStats();
      } catch (_) {
        texturePackets = null;
      }
    }
    const swapchainFormat = sanitizeSwapchainFormatToken(
      (this._swapchainViewFormat && this._streamSwapchainFormat) ||
        (swapchain && (swapchain.streamFormat || swapchain.format)) ||
        this._streamSwapchainFormat ||
        this.format
    );
    const usesBlitFallback = Boolean(
      (swapchain && swapchain.usesBlitFallback === true) ||
        (this._swapchain && this._swapchain.usesBlitFallback === true)
    );
    const blitCount = toNonNegativeInt(
      swapchain && swapchain.blitCount !== undefined
        ? swapchain.blitCount
        : this._swapchain && this._swapchain.blitCount,
      0
    );
    const bc3dFallbacks = toNonNegativeInt(
      texturePackets && texturePackets.bc3dFallbacks,
      0
    );
    const bc3dFallbackUploadsSkipped = toNonNegativeInt(
      texturePackets && texturePackets.bc3dFallbackUploadsSkipped,
      0
    );
    const wgslPackMode =
      wgslPackCache && typeof wgslPackCache.mode === 'string'
        ? wgslPackCache.mode.trim().toLowerCase()
        : 'buffer';
    const rangedPackActive = wgslPackMode === 'ranged' || wgslPackMode === 'mixed';
    const rangedPackThresholdBytes = toNonNegativeInt(
      this._activeShaderPackStreamThresholdBytes,
      DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES
    );
    const asyncShaderPendingCount = toNonNegativeInt(this._inFlightRangedShaders.size, 0);

    let ringStats = null;
    if (this.ring && typeof this.ring.getStats === 'function') {
      try {
        ringStats = this.ring.getStats();
      } catch (_) {
        ringStats = null;
      }
    }
    let resourceTable = null;
    if (this.resourceTable && typeof this.resourceTable.getStats === 'function') {
      try {
        resourceTable = this.resourceTable.getStats();
      } catch (_) {
        resourceTable = null;
      }
    }
    let drawExecutorStats = null;
    if (this.drawExecutor && typeof this.drawExecutor.getStats === 'function') {
      try {
        drawExecutorStats = this.drawExecutor.getStats();
      } catch (_) {
        drawExecutorStats = null;
      }
    }
    const gpuMemory = this.getGpuMemoryStats({ stagingPool, resourceTable });
    const perf30Fps = resolve30FpsTelemetry(this, {
      swapchain,
      rtPool,
      resourceTable,
      drawExecutor: drawExecutorStats,
      ring: ringStats,
      pipelineCache,
      reclaimedOversizedBytes: this._reclaimedOversizedBytes,
      zeroAllocDrains: this._zeroAllocDrains,
    });

    return {
      frameCount: this.frameCount,
      framesPresented: this.frameCount,
      presentCount: this._presentCount,
      splashFramesPresented: this._splashFramesPresented,
      firstPresent: this._firstPresentNotified || this.frameCount > 0,
      firstPresentMs: this._firstPresentMs,
      firstVisibleFrame: this._firstVisibleFrameNotified,
      firstVisibleFrameMs: this._firstVisibleFrameMs,
      blackLoadingMs: this._blackLoadingMs,
      requireNonBlackFrame: this._requireNonBlackFrame,
      suppressedBlackFrames: this._suppressedBlackFrames,
      lastSampledMaxLuma: this._lastSampledMaxLuma,
      swapchainProbeCount: this._swapchainProbeCount,
      swapchainProbeInFlight: this._swapchainProbeInFlight,
      swapchainProbeSupported: this._swapchainProbeSupported,
      swapchainProbeErrorCount: this._swapchainProbeErrorCount,
      swapchainProbeTimeoutCount: this._swapchainProbeTimeoutCount,
      gpuStallWarnCount: this._gpuStallWarnCount,
      deviceLostCount: this._deviceLostCount,
      deviceRecoveredCount: this._deviceRecoveredCount,
      gpuAdapter: this._gpuAdapter,
      chunkedWriteBufferUploads: this._chunkedWriteBufferUploads,
      packetCount: this._packetCount,
      drawCalls: this._drawCallCount,
      realPackDrawCount: this._realPackDrawCount,
      fallbackDrawCount: this._fallbackDrawCount,
      unboundBindGroupCount: this._unboundBindGroupCount,
      skippedDrawCount: this._skippedDrawCount,
      computeDispatches: this._computeDispatchCount,
      bytesProcessed: this._bytesProcessed,
      destroyed: this._destroyed,
      resources: {
        buffers: this.buffers.size,
        textures: this.textures.size,
        textureViews: this.textureViews.size,
        samplers: this.samplers.size,
        bindGroups: this.bindGroups.size,
        shaderModules: this.shaderModules.size,
        pipelines: this.pipelines.size,
        computePipelines: this.computePipelines.size,
      },
      stagingPool,
      uniformRing,
      passTimer,
      gpuTimer,
      gpuRecorder,
      lastPassTimings,
      resourceRegistry,
      vramBytes: resourceRegistry ? resourceRegistry.totalBytes : gpuMemory.totalLiveGpuBytes,
      gpuMemory,
      liveBufferBytes: gpuMemory.liveBufferBytes,
      liveTextureBytes: gpuMemory.liveTextureBytes,
      stagingPoolBytes: gpuMemory.stagingPoolBytes,
      pendingDeferredDestroys: gpuMemory.pendingDeferredDestroys,
      peakBufferBytes: gpuMemory.peakBufferBytes,
      peakTextureBytes: gpuMemory.peakTextureBytes,
      pendingDeferredDestroyBytes: gpuMemory.pendingDeferredDestroyBytes,
      deferredHighWaterFlushes: gpuMemory.deferredHighWaterFlushes,
      rtPool,
      rtPoolHitRate: rtPool ? rtPool.hitRate : 0,
      reflectionRegistry,
      reflectionShaderCount: reflectionRegistry ? reflectionRegistry.shaderCount : 0,
      shaderFallback,
      missingShaderSummary,
      asyncPipelineStats,
      asyncPipelineCompileStats,
      pipelineCache,
      pipelineCacheStats: perf30Fps.pipelineCacheStats,
      ring: ringStats,
      resourceTable,
      drawExecutor: drawExecutorStats,
      fps30ComplianceRate: perf30Fps.fps30ComplianceRate,
      frameJitterMs: perf30Fps.frameJitterMs,
      longestStallMs: perf30Fps.longestStallMs,
      vramBudgetUtilization: perf30Fps.vramBudgetUtilization,
      bindGroupHitRate: perf30Fps.bindGroupHitRate,
      zeroAllocDrains: perf30Fps.zeroAllocDrains,
      reclaimedOversizedBytes: perf30Fps.reclaimedOversizedBytes,
      fastLookupHits: perf30Fps.pipelineCacheStats.fastLookupHits,
      fastLookupMisses: perf30Fps.pipelineCacheStats.fastLookupMisses,
      fastLookupHitRate: perf30Fps.pipelineCacheStats.fastLookupHitRate,
      negativeCacheHits: perf30Fps.pipelineCacheStats.negativeCacheHits,
      negativeCacheSize: perf30Fps.pipelineCacheStats.negativeCacheSize,
      negativeCacheEvictions: perf30Fps.pipelineCacheStats.negativeCacheEvictions,
      fallbackSamplersCreated,
      fallbackSamplerBindingsSynthesized,
      fallbackProfileCounts: { ...this._fallbackProfileCounts },
      lastFallbackProfile: this._lastFallbackProfile,
      validationErrorCount: this._validationErrorCount,
      validationErrorHistorySize: this._validationErrors.length,
      validationErrorSuppressedCount: this._validationErrorSuppressedCount,
      wgslPackCache,
      rangedPackActive,
      rangedPackThresholdBytes,
      asyncShaderPendingCount,
      rangedShaderFetches: this._rangedShaderFetches,
      rangedShaderReady: this._rangedShaderReady,
      rangedShaderErrors: this._rangedShaderErrors,
      oversizedWgslFallbacks: this._oversizedWgslFallbacks,
      maxWgslCompileBytes: this._maxWgslCompileBytes,
      coalescedPresents: this._coalescedPresents,
      maxConcurrentShaderCompiles: this._maxConcurrentShaderCompiles,
      activeRangedShaderCompiles: this._activeRangedShaderCompiles,
      peakConcurrentRangedShaderCompiles: this._peakConcurrentRangedShaderCompiles,
      rangedShaderCompileYields: this._rangedShaderCompileYields,
      pendingRangedPipelinesCount: this._pendingRangedPipelines.size,
      splashBridge,
      binkPresenter,
      swapchain,
      swapchainFormat,
      swapchainViewFormat: this._swapchainViewFormat,
      toneMapMode: this._activeToneMapMode,
      applySrgbEncode: this._activeApplySrgbEncode,
      usesBlitFallback,
      blitCount,
      texturePackets,
      bc3dFallbacks,
      bc3dFallbackUploadsSkipped,
      deferredGpuDestroysPending: this._deferredGpuDestroys.length,
      deferredGpuDestroysInFlight: this._inFlightDeferredDestroys.size,
      deferredGpuDestroysCompleted: this._deferredDestroyCount,
      promoted1DTo2DCount: this._promoted1DTo2DCount,
      fallbackProfileScanCount: this._fallbackProfileScanCount,
      resourceReplayEnabled: this._resourceReplayEnabled,
      shadowBufferReplayEnabled: this._shadowBufferReplayEnabled,
      replayLogEntries: this._replayLog.size,
      replayBindGroupEvictions: this._replayBindGroupEvictions,
      replayLogOverflowCount: this._replayLogOverflowCount,
      clearTextureStats: { ...this._clearTextureStats },
    };
  }

  /**
   * Idempotently destroy all tracked GPU resources and wired subsystems.
   */
  destroy() {
    if (this._destroyed) {
      return;
    }
    this._destroyed = true;
    // W338: release swapchain luminance probe staging buffer cleanly on worker destroy.
    this._destroySwapchainProbeBuffer();
    // W103: settle queued/in-flight pipeline compiles so nothing writes into a torn-down processor.
    this._resetAsyncPipelineCache(null);
    // W100: intentional teardown — never treat the subsequent `device.lost` as a loss.
    this._deviceLossListener = null;
    this._deviceRecovering = false;
    this._unhookDeviceUncapturedError();
    this._unhookSwapchainValidation();
    if (
      this._swapchain &&
      this._ownsSwapchain &&
      typeof this._swapchain.destroy === 'function'
    ) {
      this._swapchain.destroy();
    }
    this._swapchain = null;
    this._ownsSwapchain = false;
    this._validationErrorCallbacks.clear();
    if (this._firstVisibleFrameListeners) {
      this._firstVisibleFrameListeners.clear();
    }
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._clearPassCommands();
    this._clearPresentPassBindings();
    this._frameBegunForIndex = -1;
    // W104: drop the replay log and shadow buffer cache.
    this._replayLog.clear();
    this._replayBindGroupKeys.clear();
    this._replayLogBytes = 0;
    this._replayAsyncActive = false;
    this._replayAsyncDeltaKeys.clear();
    this._replayAsyncDestroyedIds.clear();
    this._replayAsyncShadowIds.clear();
    this._shadowBuffers.clear();
    this._shadowBufferBytes = 0;

    // 1. Resource registry destroys all registered active & pending-retirement resources.
    const destroyedByRegistry = new Set();
    if (this.resourceRegistry && typeof this.resourceRegistry.destroyAll === 'function') {
      if (typeof this.resourceRegistry.list === 'function') {
        const activeEntries = this.resourceRegistry.list();
        for (let i = 0; i < activeEntries.length; i += 1) {
          destroyedByRegistry.add(activeEntries[i].resource);
        }
      }
      this.resourceRegistry.destroyAll();
    }

    // Destroy any buffers/textures not managed by resourceRegistry.
    for (const buffer of this.buffers.values()) {
      if (buffer && !destroyedByRegistry.has(buffer) && typeof buffer.destroy === 'function') {
        buffer.destroy();
      }
    }
    for (const texture of this.textures.values()) {
      if (texture && !destroyedByRegistry.has(texture) && typeof texture.destroy === 'function') {
        texture.destroy();
      }
    }
    if (Array.isArray(this._recordedReadbacks) && this._recordedReadbacks.length > 0) {
      for (let i = 0; i < this._recordedReadbacks.length; i += 1) {
        const staging = this._recordedReadbacks[i] && this._recordedReadbacks[i].staging;
        if (staging && typeof staging.destroy === 'function') {
          try {
            staging.destroy();
          } catch (_) {
            // Ignore teardown errors on readback staging buffers.
          }
        }
      }
      this._recordedReadbacks = [];
    }
    this._flushAllDeferredGpuDestroysImmediate(destroyedByRegistry);

    if (this._clearZeroBuffer) {
      if (typeof this._clearZeroBuffer.destroy === 'function') {
        try {
          this._clearZeroBuffer.destroy();
        } catch (_) {
          // Ignore teardown errors on clear zero buffer.
        }
      }
      this._clearZeroBuffer = null;
      this._clearZeroBufferBytes = 0;
    }
    this._clearZeroScratch = null;

    this.buffers.clear();
    this._bufferMeta.clear();
    this.textures.clear();
    this._textureMeta.clear();
    this.textureViews.clear();
    this._liveBufferBytes = 0;
    this._liveTextureBytes = 0;
    this._directWriteScratch = null;
    this.samplers.clear();
    this.bindGroups.clear();
    this._bindGroupKinds.clear();
    this._bindGroupMeta.clear();
    this._v2BindGroupLayouts.clear();
    this._v2PipelineLayouts.clear();
    this._validationErrorDedup.clear();
    this.shaderModules.clear();
    this._cachedBuiltinShaderModules.clear();
    this._shaderModuleGenerations.clear();
    this._inFlightRangedShaders.clear();
    this._clearRangedShaderCompileQueue();
    this._pendingRangedPipelines.clear();
    this.pipelines.clear();
    this._pipelineMeta.clear();
    this._specializedFallbackPipelines.clear();
    this._specializedFallbackBindGroups.clear();
    this._specializedFallbackShaderModules.clear();
    this._pendingCommandStreamShaderProfiles = null;
    this._pendingCommandStreamView = null;
    this._pendingCommandStreamStartOffset = 0;
    this._pendingCommandStreamEndOffset = 0;
    this._pendingCommandStreamScanned = true;
    this._explicitFallbackDrawContext = null;
    this.computePipelines.clear();
    this._pipelineBindGroupLayouts.clear();
    this._computePipelineBindGroupLayouts.clear();
    this._pipelineReflectionGroups.clear();
    this._computePipelineReflectionGroups.clear();
    this._defaultFallbackSamplers.clear();
    this._shaderReflectionKeys.clear();

    // 2. Tear down auxiliary subsystems idempotently.
    if (this.stagingPool && typeof this.stagingPool.destroy === 'function') {
      this.stagingPool.destroy();
    }
    if (this.uniformRing && typeof this.uniformRing.destroy === 'function') {
      this.uniformRing.destroy();
    }
    if (this.passTimer && typeof this.passTimer.destroy === 'function') {
      this.passTimer.destroy();
    }
    if (
      this.gpuRecorder &&
      this._ownsGpuRecorder &&
      typeof this.gpuRecorder.stop === 'function'
    ) {
      try {
        this.gpuRecorder.stop();
      } catch (_) {
        // Ignore recorder stop errors during teardown.
      }
    }
    this._uniformRingFrameIndex = -1;
    this._pendingFrameLuminance = null;
    this._frameStartMs = null;
    if (this.rtPool && typeof this.rtPool.destroy === 'function') {
      this.rtPool.destroy();
    }
    if (
      this.pipelineCache &&
      this._ownsPipelineCache &&
      typeof this.pipelineCache.clear === 'function'
    ) {
      this.pipelineCache.clear();
    }
    if (
      this.drawExecutor &&
      this._ownsDrawExecutor &&
      typeof this.drawExecutor.destroy === 'function'
    ) {
      try {
        this.drawExecutor.destroy();
      } catch (_) {
        // Ignore teardown errors on owned drawExecutor.
      }
    }
    if (
      this.reflectionRegistry &&
      this._ownsReflectionRegistry &&
      typeof this.reflectionRegistry.clear === 'function'
    ) {
      this.reflectionRegistry.clear();
    }
    if (
      this.wgslPackCache &&
      this._ownsWgslPackCache &&
      typeof this.wgslPackCache.clear === 'function'
    ) {
      this.wgslPackCache.clear();
    }
    if (this.splashFrameBridge && typeof this.splashFrameBridge.destroy === 'function') {
      this.splashFrameBridge.destroy();
    }
    if (this.binkPresenter && typeof this.binkPresenter.destroy === 'function') {
      this.binkPresenter.destroy();
    }

    this._lastSplashPresentMs = null;
    this._pendingTimerReadbacks.clear();
    this._lastPassTimings = null;
    this._clearPassCommands();
    this._passCommandPool.length = 0;
    this._setBindGroup2MetaPool.length = 0;
    this._clearPresentPassBindings();
    this._boundBindGroupSlots.clear();
    this._boundVertexBufferSlots.clear();
    this._cachedCmdBuffer = null;
    this._cachedCmdView = null;
    this.ring = null;
  }
}

const DEFAULT_TELEMETRY_INTERVAL_MS = 250;
const DEFAULT_TELEMETRY_MISSING_LIMIT = 64;
const DEFAULT_MISSING_SUMMARY_LIMIT = 32;
const VALID_MISSING_SHADER_REASONS = Object.freeze(
  new Set(['not_in_pack', 'reflection_failed', 'compile_failed', 'other'])
);

function sanitizeMissingShaderReason(raw) {
  if (typeof raw !== 'string') {
    return 'other';
  }
  const key = raw
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_');
  return VALID_MISSING_SHADER_REASONS.has(key) ? key : 'other';
}

/**
 * W179: Sanitize and bound a categorized missing-shader summary so telemetry snapshots and
 * `getStats` responses never expose host paths, raw WGSL, raw buffers, or unbounded lists.
 *
 * @param {object | null | undefined} raw
 * @param {number} [limit=DEFAULT_MISSING_SUMMARY_LIMIT]
 */
function sanitizeMissingShaderSummary(raw, limit = DEFAULT_MISSING_SUMMARY_LIMIT) {
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const maxTop =
    Number.isInteger(limit) && limit >= 0 ? limit : DEFAULT_MISSING_SUMMARY_LIMIT;
  const countsByReasonSrc =
    raw.countsByReason && typeof raw.countsByReason === 'object' ? raw.countsByReason : {};
  const countsByStageSrc =
    raw.countsByStage && typeof raw.countsByStage === 'object' ? raw.countsByStage : {};
  const uniqueByReasonSrc =
    raw.uniqueCountsByReason && typeof raw.uniqueCountsByReason === 'object'
      ? raw.uniqueCountsByReason
      : {};
  const uniqueByStageSrc =
    raw.uniqueCountsByStage && typeof raw.uniqueCountsByStage === 'object'
      ? raw.uniqueCountsByStage
      : {};
  const guardSrc =
    raw.realShaderGuard && typeof raw.realShaderGuard === 'object' ? raw.realShaderGuard : {};
  const rawTop = Array.isArray(raw.topMissing) ? raw.topMissing : [];
  const topMissing = [];
  for (let i = 0; i < rawTop.length && topMissing.length < maxTop; i += 1) {
    const row = rawTop[i];
    if (!row || typeof row !== 'object') {
      continue;
    }
    const rawHash =
      row.hashHex !== undefined
        ? row.hashHex
        : row.hash !== undefined
          ? row.hash
          : row.shaderHash;
    const hashBigInt = normalizeShaderHashBigInt(rawHash);
    if (hashBigInt === null) {
      continue;
    }
    const rowReasons = row.reasons && typeof row.reasons === 'object' ? row.reasons : {};
    const reason = sanitizeMissingShaderReason(row.reason);
    const lastReason = sanitizeMissingShaderReason(
      row.lastReason !== undefined ? row.lastReason : reason
    );
    const cleanRow = {
      hashHex: formatShaderHashHex(hashBigInt),
      stage: sanitizeTelemetryStage(row.stage),
      reason,
      lastReason,
      hitCount: toNonNegativeInt(
        row.hitCount !== undefined ? row.hitCount : row.hits !== undefined ? row.hits : row.count,
        0
      ),
      reasons: {
        not_in_pack: toNonNegativeInt(rowReasons.not_in_pack, 0),
        reflection_failed: toNonNegativeInt(rowReasons.reflection_failed, 0),
        compile_failed: toNonNegativeInt(rowReasons.compile_failed, 0),
        other: toNonNegativeInt(rowReasons.other, 0),
      },
    };
    const firstSeenMs = toNonNegativeFinite(row.firstSeenMs);
    const lastSeenMs = toNonNegativeFinite(row.lastSeenMs);
    if (firstSeenMs !== null) {
      cleanRow.firstSeenMs = firstSeenMs;
    }
    if (lastSeenMs !== null) {
      cleanRow.lastSeenMs = lastSeenMs;
    }
    topMissing.push(cleanRow);
  }
  return {
    totalMissingEvents: toNonNegativeInt(raw.totalMissingEvents, 0),
    uniqueMissingHashes: toNonNegativeInt(raw.uniqueMissingHashes, topMissing.length),
    maxTrackedHashes: toNonNegativeInt(raw.maxTrackedHashes, 0),
    overflowCount: toNonNegativeInt(raw.overflowCount, 0),
    countsByReason: {
      not_in_pack: toNonNegativeInt(countsByReasonSrc.not_in_pack, 0),
      reflection_failed: toNonNegativeInt(countsByReasonSrc.reflection_failed, 0),
      compile_failed: toNonNegativeInt(countsByReasonSrc.compile_failed, 0),
      other: toNonNegativeInt(countsByReasonSrc.other, 0),
    },
    countsByStage: {
      vertex: toNonNegativeInt(countsByStageSrc.vertex, 0),
      fragment: toNonNegativeInt(countsByStageSrc.fragment, 0),
      compute: toNonNegativeInt(countsByStageSrc.compute, 0),
    },
    uniqueCountsByReason: {
      not_in_pack: toNonNegativeInt(uniqueByReasonSrc.not_in_pack, 0),
      reflection_failed: toNonNegativeInt(uniqueByReasonSrc.reflection_failed, 0),
      compile_failed: toNonNegativeInt(uniqueByReasonSrc.compile_failed, 0),
      other: toNonNegativeInt(uniqueByReasonSrc.other, 0),
    },
    uniqueCountsByStage: {
      vertex: toNonNegativeInt(uniqueByStageSrc.vertex, 0),
      fragment: toNonNegativeInt(uniqueByStageSrc.fragment, 0),
      compute: toNonNegativeInt(uniqueByStageSrc.compute, 0),
    },
    realShaderGuard: {
      registeredRealShaders: toNonNegativeInt(guardSrc.registeredRealShaders, 0),
      blockedOverrideCount: toNonNegativeInt(guardSrc.blockedOverrideCount, 0),
      overflowBlockedCount: toNonNegativeInt(guardSrc.overflowBlockedCount, 0),
    },
    topMissing,
  };
}

const MAX_GPU_ERROR_DETAIL_CHARS = 256;

/**
 * Reduce a thrown/rejected WebGPU error to a single-line `Name: message` string with URLs
 * stripped and length capped, suitable for validation telemetry.
 *
 * @param {unknown} err
 * @returns {string}
 */
function sanitizeGpuErrorDetail(err) {
  let name = '';
  let message = '';
  try {
    if (err && typeof err === 'object') {
      name = typeof err.name === 'string' ? err.name : '';
      message = typeof err.message === 'string' ? err.message : '';
    } else if (err !== undefined && err !== null) {
      message = String(err);
    }
  } catch (_) {
    message = '';
  }
  message = message
    .replace(/\b(?:https?|file|blob):\/\/\S+/gi, '<url>')
    .replace(/\s+/g, ' ')
    .trim();
  if (message.length === 0) {
    message = 'operation failed';
  }
  const detail = name && name !== 'Error' ? `${name}: ${message}` : message;
  return detail.length > MAX_GPU_ERROR_DETAIL_CHARS
    ? `${detail.slice(0, MAX_GPU_ERROR_DETAIL_CHARS - 3)}...`
    : detail;
}

function toNonNegativeInt(value, fallback = 0) {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.floor(n);
}

function toNonNegativeFinite(value) {
  if (value === undefined || value === null || value === '') {
    return null;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return null;
  }
  return n;
}

function toClampedUnitRate(value) {
  if (value === undefined || value === null || value === '') {
    return 0;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n <= 0) {
    return 0;
  }
  return n >= 1 ? 1 : n;
}

function readOptionalStatsObject(source) {
  if (!source || typeof source !== 'object') {
    return null;
  }
  try {
    if (typeof source.getStats === 'function') {
      const s = source.getStats();
      if (s && typeof s === 'object') {
        return s;
      }
    }
  } catch (_) {
    // Ignore throwing mock getStats() implementations.
  }
  return source;
}

/**
 * W295: Extract and sanitize 30 FPS hot-path telemetry across WgpuSwapchain (W286),
 * RenderTargetPool (W285), ResourceTable (W267), GpuRingBuffer (W284), and
 * WebGpuPipelineCache (W281/W268) with zero host paths or raw buffers.
 */
function resolve30FpsTelemetry(processor, stats) {
  const proc = processor && typeof processor === 'object' ? processor : null;
  const st = stats && typeof stats === 'object' ? stats : null;

  const swapObj =
    (proc && (proc.swapchain || proc._swapchain)) ||
    (st && (st.swapchain || st.swapchainStats)) ||
    null;
  const swapStats = readOptionalStatsObject(swapObj);

  const rtPoolObj = (proc && proc.rtPool) || (st && st.rtPool) || null;
  const rtPoolStats = readOptionalStatsObject(rtPoolObj);

  const resourceTableObj =
    (proc && (proc.resourceTable || proc._resourceTable)) ||
    (st && st.resourceTable) ||
    null;
  const resourceTableStats = readOptionalStatsObject(resourceTableObj);

  const drawExecObj = (proc && proc.drawExecutor) || (st && st.drawExecutor) || null;
  const drawExecStats = readOptionalStatsObject(drawExecObj);

  const ringObj = (proc && proc.ring) || (st && st.ring) || null;
  const ringStats = readOptionalStatsObject(ringObj);

  const pipeCacheObj = (proc && proc.pipelineCache) || (st && st.pipelineCache) || null;
  const pipeCacheStats = readOptionalStatsObject(pipeCacheObj);

  const fps30ComplianceRate = toClampedUnitRate(
    swapStats && swapStats.fps30ComplianceRate !== undefined
      ? swapStats.fps30ComplianceRate
      : swapObj && swapObj.fps30ComplianceRate !== undefined
        ? swapObj.fps30ComplianceRate
        : st && st.fps30ComplianceRate
  );
  const frameJitterRaw = toNonNegativeFinite(
    swapStats && swapStats.frameJitterMs !== undefined
      ? swapStats.frameJitterMs
      : swapObj && swapObj.frameJitterMs !== undefined
        ? swapObj.frameJitterMs
        : st && st.frameJitterMs
  );
  const frameJitterMs = frameJitterRaw !== null ? frameJitterRaw : 0;
  const longestStallRaw = toNonNegativeFinite(
    swapStats && swapStats.longestStallMs !== undefined
      ? swapStats.longestStallMs
      : swapObj && swapObj.longestStallMs !== undefined
        ? swapObj.longestStallMs
        : st && st.longestStallMs
  );
  const longestStallMs = longestStallRaw !== null ? longestStallRaw : 0;

  let rawVramUtil =
    rtPoolStats && rtPoolStats.vramBudgetUtilization !== undefined
      ? rtPoolStats.vramBudgetUtilization
      : st && st.vramBudgetUtilization;
  if (
    (rawVramUtil === undefined || rawVramUtil === null) &&
    rtPoolStats &&
    typeof rtPoolStats.totalBytes === 'number' &&
    typeof rtPoolStats.maxPoolBytes === 'number' &&
    rtPoolStats.maxPoolBytes > 0
  ) {
    rawVramUtil = rtPoolStats.totalBytes / rtPoolStats.maxPoolBytes;
  }
  const vramBudgetUtilization = toClampedUnitRate(rawVramUtil);

  const rtReclaimed = toNonNegativeInt(rtPoolStats && rtPoolStats.reclaimedOversizedBytes, 0);
  const localReclaimed = toNonNegativeInt(
    proc && proc._reclaimedOversizedBytes !== undefined
      ? proc._reclaimedOversizedBytes
      : st && st.reclaimedOversizedBytes,
    0
  );
  const reclaimedOversizedBytes = Math.max(rtReclaimed, localReclaimed);

  let rawBgHitRate =
    resourceTableStats && resourceTableStats.bindGroupHitRate !== undefined
      ? resourceTableStats.bindGroupHitRate
      : st && st.bindGroupHitRate !== undefined
        ? st.bindGroupHitRate
        : drawExecStats && drawExecStats.bindGroupHitRate;
  if (
    (rawBgHitRate === undefined || rawBgHitRate === null) &&
    resourceTableStats &&
    typeof resourceTableStats.bindGroupHits === 'number' &&
    typeof resourceTableStats.bindGroupMisses === 'number'
  ) {
    const totalBg = resourceTableStats.bindGroupHits + resourceTableStats.bindGroupMisses;
    rawBgHitRate = totalBg > 0 ? resourceTableStats.bindGroupHits / totalBg : 0;
  }
  const bindGroupHitRate = toClampedUnitRate(rawBgHitRate);

  const ringZeroAlloc = toNonNegativeInt(
    ringStats && ringStats.zeroAllocDrains !== undefined
      ? ringStats.zeroAllocDrains
      : ringObj && ringObj.zeroAllocDrains,
    0
  );
  const localZeroAlloc = toNonNegativeInt(
    proc && proc._zeroAllocDrains !== undefined
      ? proc._zeroAllocDrains
      : st && st.zeroAllocDrains,
    0
  );
  const zeroAllocDrains = Math.max(ringZeroAlloc, localZeroAlloc);

  const fastLookupHits = toNonNegativeInt(
    pipeCacheStats && pipeCacheStats.fastLookupHits !== undefined
      ? pipeCacheStats.fastLookupHits
      : st && st.fastLookupHits,
    0
  );
  const fastLookupMisses = toNonNegativeInt(
    pipeCacheStats && pipeCacheStats.fastLookupMisses !== undefined
      ? pipeCacheStats.fastLookupMisses
      : st && st.fastLookupMisses,
    0
  );
  const fastLookupHitRate = toClampedUnitRate(
    pipeCacheStats && pipeCacheStats.fastLookupHitRate !== undefined
      ? pipeCacheStats.fastLookupHitRate
      : st && st.fastLookupHitRate !== undefined
        ? st.fastLookupHitRate
        : fastLookupHits + fastLookupMisses > 0
          ? fastLookupHits / (fastLookupHits + fastLookupMisses)
          : 0
  );
  const negativeCacheHits = toNonNegativeInt(
    pipeCacheStats && pipeCacheStats.negativeCacheHits !== undefined
      ? pipeCacheStats.negativeCacheHits
      : pipeCacheStats && pipeCacheStats.quarantineHits !== undefined
        ? pipeCacheStats.quarantineHits
        : pipeCacheStats && pipeCacheStats.negativeHits !== undefined
          ? pipeCacheStats.negativeHits
          : st && st.negativeCacheHits,
    0
  );
  const negativeCacheSize = toNonNegativeInt(
    pipeCacheStats && pipeCacheStats.negativeCacheSize !== undefined
      ? pipeCacheStats.negativeCacheSize
      : pipeCacheStats && pipeCacheStats.quarantinedKeys !== undefined
        ? pipeCacheStats.quarantinedKeys
        : st && st.negativeCacheSize,
    0
  );
  const negativeCacheEvictions = toNonNegativeInt(
    pipeCacheStats && pipeCacheStats.negativeCacheEvictions !== undefined
      ? pipeCacheStats.negativeCacheEvictions
      : st && st.negativeCacheEvictions,
    0
  );

  return {
    fps30ComplianceRate,
    frameJitterMs,
    longestStallMs,
    vramBudgetUtilization,
    bindGroupHitRate,
    zeroAllocDrains,
    reclaimedOversizedBytes,
    pipelineCacheStats: {
      fastLookupHits,
      fastLookupMisses,
      fastLookupHitRate,
      negativeCacheHits,
      negativeCacheSize,
      negativeCacheEvictions,
    },
  };
}

/**
 * W332 / W324: Extract and sanitize GpuPassTimer 30 FPS per-pass histogram summary
 * (zero host paths, bounded pass labels, finite non-negative numbers).
 */
function resolveGpuTimerTelemetry(processor, stats) {
  const proc = processor && typeof processor === 'object' ? processor : null;
  const st = stats && typeof stats === 'object' ? stats : null;
  const timerObj = (proc && (proc.passTimer || proc.gpuTimer)) || null;
  const rawFromObj = timerObj ? readOptionalStatsObject(timerObj) : null;
  const rawFromStats =
    (st && st.gpuTimer && typeof st.gpuTimer === 'object' ? st.gpuTimer : null) ||
    (st && st.passTimer && typeof st.passTimer === 'object' ? st.passTimer : null);
  const raw = rawFromObj || rawFromStats;
  if (!raw || typeof raw !== 'object') {
    return null;
  }

  let rawPassStats = raw.passStats && typeof raw.passStats === 'object' ? raw.passStats : null;
  if (!rawPassStats && timerObj && typeof timerObj.getPassStats === 'function') {
    try {
      const ps = timerObj.getPassStats();
      if (ps && typeof ps === 'object') {
        rawPassStats = ps;
      }
    } catch (_) {
      rawPassStats = null;
    }
  }

  const passStats = {};
  if (rawPassStats) {
    const keys = Object.keys(rawPassStats);
    for (let i = 0; i < keys.length; i += 1) {
      const cleanLabel = sanitizeGpuAdapterField(keys[i]) || `pass_${i}`;
      const entry = rawPassStats[keys[i]];
      if (!entry || typeof entry !== 'object') {
        continue;
      }
      passStats[cleanLabel] = {
        count: toNonNegativeInt(entry.count, 0),
        lastMs: toNonNegativeFinite(entry.lastMs) ?? 0,
        avgMs: toNonNegativeFinite(entry.avgMs) ?? 0,
        p50Ms: toNonNegativeFinite(entry.p50Ms) ?? 0,
        p95Ms: toNonNegativeFinite(entry.p95Ms) ?? 0,
        maxMs: toNonNegativeFinite(entry.maxMs) ?? 0,
        passesOverBudget: toNonNegativeInt(entry.passesOverBudget, 0),
      };
    }
  }

  const passBudgetMs = toNonNegativeFinite(raw.passBudgetMs);
  return {
    supported: Boolean(raw.supported),
    resolvedFrames: toNonNegativeInt(raw.resolvedFrames, 0),
    skippedFrames: toNonNegativeInt(raw.skippedFrames, 0),
    droppedPasses: toNonNegativeInt(raw.droppedPasses, 0),
    droppedQueryFrames: toNonNegativeInt(raw.droppedQueryFrames, 0),
    lastFrameGpuMs: toNonNegativeFinite(raw.lastFrameGpuMs) ?? 0,
    p50PassMs: toNonNegativeFinite(raw.p50PassMs) ?? 0,
    p95PassMs: toNonNegativeFinite(raw.p95PassMs) ?? 0,
    maxPassMs: toNonNegativeFinite(raw.maxPassMs) ?? 0,
    passBudgetMs: passBudgetMs !== null && passBudgetMs > 0 ? passBudgetMs : 33.333,
    passesOverBudget: toNonNegativeInt(raw.passesOverBudget, 0),
    querySlotRecycles: toNonNegativeInt(raw.querySlotRecycles, 0),
    querySlotWraps: toNonNegativeInt(raw.querySlotWraps, 0),
    passStats,
  };
}

/**
 * W332 / W325: Extract and sanitize GpuFrameRecorder 30 FPS spike auto-capture state
 * (zero raw command buffers, zero host paths).
 */
function resolveGpuRecorderTelemetry(processor, stats) {
  const proc = processor && typeof processor === 'object' ? processor : null;
  const st = stats && typeof stats === 'object' ? stats : null;
  const recorderObj = (proc && (proc.gpuRecorder || proc.recorder)) || null;
  const rawFromObj = recorderObj ? readOptionalStatsObject(recorderObj) : null;
  const rawFromStats =
    st && st.gpuRecorder && typeof st.gpuRecorder === 'object' ? st.gpuRecorder : null;
  const raw = rawFromObj || rawFromStats;
  if (!raw || typeof raw !== 'object') {
    return null;
  }

  let spikeTriggerEnabled = Boolean(raw.spikeTriggerEnabled === true);
  if (
    raw.spikeTriggerEnabled === undefined &&
    recorderObj &&
    typeof recorderObj.isSpikeTriggerEnabled === 'function'
  ) {
    try {
      spikeTriggerEnabled = Boolean(recorderObj.isSpikeTriggerEnabled());
    } catch (_) {
      spikeTriggerEnabled = false;
    }
  }

  const spikeBudgetMs = toNonNegativeFinite(raw.spikeBudgetMs);
  return {
    recording: Boolean(raw.recording === true),
    frameCount: toNonNegativeInt(raw.frameCount, 0),
    recordedBytes: toNonNegativeInt(
      raw.recordedBytes !== undefined ? raw.recordedBytes : raw.totalBytes,
      0
    ),
    peakRecordedBytes: toNonNegativeInt(raw.peakRecordedBytes, 0),
    recordedPackets: toNonNegativeInt(
      raw.recordedPackets !== undefined ? raw.recordedPackets : raw.totalPackets,
      0
    ),
    spikeTriggerEnabled,
    spikeBudgetMs: spikeBudgetMs !== null && spikeBudgetMs > 0 ? spikeBudgetMs : 33.333,
    spikeMaxSnapshots: toNonNegativeInt(raw.spikeMaxSnapshots, 0),
    spikeSnapshotsCaptured: toNonNegativeInt(raw.spikeSnapshotsCaptured, 0),
    lastSpikeFrameMs: toNonNegativeFinite(raw.lastSpikeFrameMs) ?? 0,
    bypassedPackets: toNonNegativeInt(raw.bypassedPackets, 0),
  };
}

function sanitizeTelemetryStage(stage) {
  if (typeof stage === 'number') {
    return WGPK_STAGE_NAMES[stage] || 'unknown';
  }
  if (typeof stage === 'string') {
    const key = stage.trim().toLowerCase();
    if (key === 'vertex' || key === 'vs' || key === 'vert' || key === '1') {
      return 'vertex';
    }
    if (
      key === 'fragment' ||
      key === 'fs' ||
      key === 'frag' ||
      key === 'pixel' ||
      key === 'ps' ||
      key === '2'
    ) {
      return 'fragment';
    }
    if (key === 'compute' || key === 'cs' || key === 'comp' || key === '3') {
      return 'compute';
    }
  }
  return 'unknown';
}

function sanitizeMissingShaderRecord(entry) {
  if (!entry || typeof entry !== 'object') {
    return null;
  }
  const rawHash =
    entry.hash !== undefined
      ? entry.hash
      : entry.shaderHash !== undefined
        ? entry.shaderHash
        : entry.hashHex;
  const hashBigInt = normalizeShaderHashBigInt(rawHash);
  if (hashBigInt === null) {
    return null;
  }
  const hash = formatShaderHashHex(hashBigInt);
  const stage = sanitizeTelemetryStage(entry.stage);
  const rawHits =
    entry.hitCount !== undefined
      ? entry.hitCount
      : entry.hits !== undefined
        ? entry.hits
        : entry.drawCount !== undefined
          ? entry.drawCount
          : entry.count;
  const hitCount = rawHits === undefined || rawHits === null ? 1 : toNonNegativeInt(rawHits, 0);
  const firstSeenMs = toNonNegativeFinite(entry.firstSeenMs);
  const lastSeenMs = toNonNegativeFinite(entry.lastSeenMs);

  const sanitized = {
    hash,
    stage,
    hits: hitCount,
    hitCount,
  };
  if (firstSeenMs !== null) {
    sanitized.firstSeenMs = firstSeenMs;
  }
  if (lastSeenMs !== null) {
    sanitized.lastSeenMs = lastSeenMs;
  }
  return sanitized;
}

/**
 * Build a sanitized `{ type: 'gpu-telemetry', frameStats, fallbackStats, missingShaders, timestamp, renderStateStats? }`
 * payload matching `GpuBootPanelController.update(...)` (zero host paths, zero raw bytes).
 *
 * @param {WgpuCommandProcessor | object | null | undefined} processor
 * @param {{ limit?: number, missingShaderLimit?: number, timestamp?: number, now?: () => number }} [options]
 */
function buildGpuTelemetrySnapshot(processor, options = {}) {
  const opts = options && typeof options === 'object' ? options : {};
  const rawLimit =
    opts.limit !== undefined
      ? opts.limit
      : opts.missingShaderLimit !== undefined
        ? opts.missingShaderLimit
        : DEFAULT_TELEMETRY_MISSING_LIMIT;
  const limit =
    Number.isInteger(rawLimit) && rawLimit >= 0 ? rawLimit : DEFAULT_TELEMETRY_MISSING_LIMIT;

  let stats = null;
  if (processor && typeof processor.getStats === 'function') {
    try {
      stats = processor.getStats();
    } catch (_) {
      stats = null;
    }
  }

  let rawFallback = null;
  if (processor && typeof processor.getShaderFallbackStats === 'function') {
    try {
      rawFallback = processor.getShaderFallbackStats();
    } catch (_) {
      rawFallback = null;
    }
  } else if (stats && stats.shaderFallback && typeof stats.shaderFallback === 'object') {
    rawFallback = stats.shaderFallback;
  }

  let rawMissing = [];
  if (processor && typeof processor.getMissingShaders === 'function') {
    try {
      const list = processor.getMissingShaders({ limit });
      if (Array.isArray(list)) {
        rawMissing = list;
      }
    } catch (_) {
      rawMissing = [];
    }
  }

  const missingShaders = [];
  for (let i = 0; i < rawMissing.length && missingShaders.length < limit; i += 1) {
    const clean = sanitizeMissingShaderRecord(rawMissing[i]);
    if (clean) {
      missingShaders.push(clean);
    }
  }

  const framesPresented = toNonNegativeInt(
    stats && stats.frameCount !== undefined
      ? stats.frameCount
      : stats && stats.framesPresented !== undefined
        ? stats.framesPresented
        : processor && processor.frameCount,
    0
  );
  const droppedFrames = toNonNegativeInt(
    stats && stats.droppedFrames !== undefined
      ? stats.droppedFrames
      : stats && stats.splashBridge && stats.splashBridge.droppedFrames,
    0
  );
  const drawCalls = toNonNegativeInt(stats && stats.drawCalls, 0);
  const realPackDrawCount = toNonNegativeInt(
    stats && stats.realPackDrawCount !== undefined
      ? stats.realPackDrawCount
      : processor && processor.realPackDrawCount,
    0
  );
  const fallbackDrawCount = toNonNegativeInt(
    stats && stats.fallbackDrawCount !== undefined
      ? stats.fallbackDrawCount
      : processor && processor.fallbackDrawCount,
    0
  );
  const computeDispatches = toNonNegativeInt(stats && stats.computeDispatches, 0);
  const packetsProcessed = toNonNegativeInt(
    stats && stats.packetCount !== undefined
      ? stats.packetCount
      : stats && stats.packetsProcessed,
    0
  );
  const coalescedPresents = toNonNegativeInt(
    stats && stats.coalescedPresents !== undefined
      ? stats.coalescedPresents
      : processor && processor._coalescedPresents,
    0
  );
  const oversizedWgslFallbacks = toNonNegativeInt(
    stats && stats.oversizedWgslFallbacks !== undefined
      ? stats.oversizedWgslFallbacks
      : processor && processor._oversizedWgslFallbacks,
    0
  );

  const packHits = toNonNegativeInt(rawFallback && rawFallback.packHits, 0);
  const fallbackHits = toNonNegativeInt(rawFallback && rawFallback.fallbackHits, 0);
  const builtinHits = toNonNegativeInt(rawFallback && rawFallback.builtinHits, 0);
  const sumHits = packHits + fallbackHits + builtinHits;
  const totalLookups =
    rawFallback && rawFallback.totalLookups !== undefined && rawFallback.totalLookups !== null
      ? toNonNegativeInt(rawFallback.totalLookups, sumHits)
      : rawFallback && rawFallback.totalResolves !== undefined && rawFallback.totalResolves !== null
        ? toNonNegativeInt(rawFallback.totalResolves, sumHits)
        : sumHits;
  const uniqueMissing =
    rawFallback && rawFallback.uniqueMissing !== undefined && rawFallback.uniqueMissing !== null
      ? toNonNegativeInt(rawFallback.uniqueMissing, missingShaders.length)
      : rawFallback && rawFallback.missingCount !== undefined && rawFallback.missingCount !== null
        ? toNonNegativeInt(rawFallback.missingCount, missingShaders.length)
        : missingShaders.length;
  const overflowMissingCount = toNonNegativeInt(rawFallback && rawFallback.overflowMissingCount, 0);

  let hitRatePct;
  if (
    rawFallback &&
    rawFallback.hitRatePct !== undefined &&
    rawFallback.hitRatePct !== null &&
    Number.isFinite(Number(rawFallback.hitRatePct))
  ) {
    const clamped = Math.min(100, Math.max(0, Number(rawFallback.hitRatePct)));
    hitRatePct = Math.round(clamped * 100) / 100;
  } else {
    hitRatePct = totalLookups > 0 ? Math.round((packHits / totalLookups) * 10000) / 100 : 0;
  }

  const renderStateTelemetry = resolveRenderStateTelemetry(processor, stats);
  const renderStateStats = {
    computeDispatchCount: toNonNegativeInt(
      renderStateTelemetry && renderStateTelemetry.computeDispatchCount,
      0
    ),
    computePipelineCount: toNonNegativeInt(
      renderStateTelemetry && renderStateTelemetry.computePipelineCount,
      0
    ),
    autoLayoutFallbacks: toNonNegativeInt(
      renderStateTelemetry && renderStateTelemetry.autoLayoutFallbacks,
      0
    ),
    bindGroupKindMismatches: toNonNegativeInt(
      renderStateTelemetry && renderStateTelemetry.bindGroupKindMismatches,
      0
    ),
  };

  let timestamp;
  if (typeof opts.timestamp === 'number' && Number.isFinite(opts.timestamp)) {
    timestamp = opts.timestamp;
  } else if (typeof opts.now === 'function') {
    const t = opts.now();
    timestamp = typeof t === 'number' && Number.isFinite(t) ? t : Date.now();
  } else {
    timestamp = Date.now();
  }

  const snapshot = {
    type: 'gpu-telemetry',
    frameStats: {
      framesPresented,
      droppedFrames,
      drawCalls,
      realPackDrawCount,
      fallbackDrawCount,
      computeDispatches,
      packetsProcessed,
    },
    fallbackStats: {
      totalLookups,
      totalResolves: totalLookups,
      packHits,
      fallbackHits,
      builtinHits,
      hitRatePct,
      uniqueMissing,
      missingCount: uniqueMissing,
      overflowMissingCount,
    },
    missingShaders,
    timestamp,
  };
  // W99: live render_state_packets.js compute/auto-layout counters (0 when not installed).
  // Omitted for a null processor so the empty snapshot shape stays unchanged.
  if (processor && typeof processor === 'object') {
    const summaryLimit =
      opts.limit !== undefined || opts.missingShaderLimit !== undefined
        ? limit
        : DEFAULT_MISSING_SUMMARY_LIMIT;
    snapshot.renderStateStats = renderStateStats;
    // W100: WebGPU device-loss / recovery counters (zeros when the processor lacks them).
    snapshot.deviceStats = resolveDeviceLossTelemetry(processor, stats);
    // W103: async pipeline compile queue counters (null when async compilation is disabled).
    snapshot.asyncPipelineStats = resolveAsyncPipelineTelemetry(processor);
    // W168: async pipeline priority & compile-latency telemetry (null when disabled).
    snapshot.asyncPipelineCompileStats = resolveAsyncPipelineCompileTelemetry(processor);
    // W179: bounded, sanitized categorized missing-shader summary (null when unavailable).
    snapshot.missingShaderSummary = resolveMissingShaderSummaryTelemetry(
      processor,
      stats,
      summaryLimit
    );
    // W167: fallback sampler synthesis telemetry counters.
    snapshot.fallbackSamplerStats = resolveFallbackSamplerTelemetry(processor, stats);
    // W121: swapchain format/blit (W112), 3D-BC texture fallback (W108), and WGSL pack mode (W109).
    snapshot.swapchainStats = resolveSwapchainTelemetry(processor, stats);
    snapshot.textureStats = resolveTextureTelemetry(processor, stats);
    snapshot.shaderPackStats = resolveShaderPackTelemetry(processor, stats);
    snapshot.oversizedWgslFallbacks = oversizedWgslFallbacks;
    snapshot.coalescedPresents = coalescedPresents;
    // W257: first-visible-frame readiness & sanitized GPU adapter metadata.
    snapshot.firstVisibleFrame = Boolean(stats && stats.firstVisibleFrame === true);
    snapshot.firstVisibleFrameMs =
      stats &&
      typeof stats.firstVisibleFrameMs === 'number' &&
      Number.isFinite(stats.firstVisibleFrameMs)
        ? stats.firstVisibleFrameMs
        : null;
    snapshot.blackLoadingMs =
      stats &&
      typeof stats.blackLoadingMs === 'number' &&
      Number.isFinite(stats.blackLoadingMs)
        ? stats.blackLoadingMs
        : null;
    snapshot.gpuAdapter =
      stats && stats.gpuAdapter && typeof stats.gpuAdapter === 'object'
        ? sanitizeGpuAdapterInfo(stats.gpuAdapter)
        : null;
    // W350: Surface swapchain luminance probe diagnostics directly in gpu-telemetry.
    snapshot.swapchainProbeCount = toNonNegativeInt(
      stats && stats.swapchainProbeCount !== undefined
        ? stats.swapchainProbeCount
        : processor && processor.swapchainProbeCount,
      0
    );
    snapshot.swapchainProbeInFlight = Boolean(
      stats && stats.swapchainProbeInFlight !== undefined
        ? stats.swapchainProbeInFlight
        : processor && processor.swapchainProbeInFlight
    );
    const rawProbeSupported =
      stats && stats.swapchainProbeSupported !== undefined
        ? stats.swapchainProbeSupported
        : processor && processor.swapchainProbeSupported !== undefined
          ? processor.swapchainProbeSupported
          : null;
    snapshot.swapchainProbeSupported =
      typeof rawProbeSupported === 'boolean' ? rawProbeSupported : null;
    snapshot.swapchainProbeErrorCount = toNonNegativeInt(
      stats && stats.swapchainProbeErrorCount !== undefined
        ? stats.swapchainProbeErrorCount
        : processor && processor.swapchainProbeErrorCount,
      0
    );
    snapshot.swapchainProbeTimeoutCount = toNonNegativeInt(
      stats && stats.swapchainProbeTimeoutCount !== undefined
        ? stats.swapchainProbeTimeoutCount
        : processor && processor.swapchainProbeTimeoutCount,
      0
    );
    snapshot.gpuStallWarnCount = toNonNegativeInt(
      stats && stats.gpuStallWarnCount !== undefined
        ? stats.gpuStallWarnCount
        : processor && processor.gpuStallWarnCount,
      0
    );
    const rawLastMaxLuma =
      stats && stats.lastSampledMaxLuma !== undefined
        ? stats.lastSampledMaxLuma
        : processor && processor.lastSampledMaxLuma;
    snapshot.lastSampledMaxLuma =
      typeof rawLastMaxLuma === 'number' && Number.isFinite(rawLastMaxLuma)
        ? Math.max(0, Math.min(255, Math.floor(rawLastMaxLuma)))
        : 0;
    snapshot.suppressedBlackFrames = toNonNegativeInt(
      stats && stats.suppressedBlackFrames !== undefined
        ? stats.suppressedBlackFrames
        : processor && processor.suppressedBlackFrames,
      0
    );
    // W295: Sanitized 30 FPS hot-path telemetry (W286/W285/W284/W281/W268/W267).
    const perf30Fps = resolve30FpsTelemetry(processor, stats);
    snapshot.fps30ComplianceRate = perf30Fps.fps30ComplianceRate;
    snapshot.frameJitterMs = perf30Fps.frameJitterMs;
    snapshot.longestStallMs = perf30Fps.longestStallMs;
    snapshot.vramBudgetUtilization = perf30Fps.vramBudgetUtilization;
    snapshot.bindGroupHitRate = perf30Fps.bindGroupHitRate;
    snapshot.zeroAllocDrains = perf30Fps.zeroAllocDrains;
    snapshot.reclaimedOversizedBytes = perf30Fps.reclaimedOversizedBytes;
    snapshot.pipelineCacheStats = perf30Fps.pipelineCacheStats;
    // W332: Surface W324 GpuPassTimer 30 FPS histogram & W325 GpuFrameRecorder spike state when active.
    const gpuTimer = resolveGpuTimerTelemetry(processor, stats);
    if (gpuTimer !== null) {
      snapshot.gpuTimer = gpuTimer;
    }
    const gpuRecorder = resolveGpuRecorderTelemetry(processor, stats);
    if (gpuRecorder !== null) {
      snapshot.gpuRecorder = gpuRecorder;
    }
    if (stats && stats.gpuMemory && typeof stats.gpuMemory === 'object') {
      snapshot.gpuMemory = { ...stats.gpuMemory };
    } else if (typeof processor.getGpuMemoryStats === 'function') {
      try {
        snapshot.gpuMemory = processor.getGpuMemoryStats();
      } catch (_) {
        // Ignore memory telemetry errors.
      }
    }
  }
  return snapshot;
}

/**
 * W121: sanitized swapchain format and W112 offscreen blit-fallback counters.
 *
 * @returns {{ swapchainFormat: string | null, canvasFormat: string | null, usesBlitFallback: boolean, blitCount: number }}
 */
function resolveSwapchainTelemetry(processor, stats) {
  let raw = null;
  try {
    const swap =
      (processor && processor.swapchain) ||
      (processor && processor._swapchain) ||
      null;
    if (swap && typeof swap.getStats === 'function') {
      raw = swap.getStats();
    } else if (swap && typeof swap === 'object') {
      raw = swap;
    }
  } catch (_) {
    raw = null;
  }
  if (!raw || typeof raw !== 'object') {
    if (stats && stats.swapchainStats && typeof stats.swapchainStats === 'object') {
      raw = stats.swapchainStats;
    } else if (stats && stats.swapchain && typeof stats.swapchain === 'object') {
      raw = stats.swapchain;
    }
  }
  const src = raw && typeof raw === 'object' ? raw : {};
  const swapchainFormat = sanitizeSwapchainFormatToken(
    (processor && processor._swapchainViewFormat && processor._streamSwapchainFormat) ||
      src.swapchainFormat ||
      src.streamFormat ||
      src.format ||
      (stats && stats.swapchainFormat) ||
      (processor && processor._streamSwapchainFormat) ||
      (processor && processor.format) ||
      null
  );
  const canvasFormat = sanitizeSwapchainFormatToken(
    src.canvasFormat ||
      src.format ||
      (processor && processor.format) ||
      swapchainFormat ||
      null
  );
  const usesBlitFallback = Boolean(
    src.usesBlitFallback === true ||
      (stats && stats.usesBlitFallback === true)
  );
  const blitCount = toNonNegativeInt(
    src.blitCount !== undefined
      ? src.blitCount
      : stats && stats.blitCount,
    0
  );
  return {
    swapchainFormat,
    canvasFormat,
    usesBlitFallback,
    blitCount,
  };
}

/**
 * W121: sanitized W108 3D-BC texture fallback counters from `TexturePackets.getStats()`.
 *
 * @returns {{ bc3dFallbacks: number, bc3dFallbackUploadsSkipped: number }}
 */
function resolveTextureTelemetry(processor, stats) {
  let raw = null;
  const readSource = (source) => {
    if (!source || typeof source !== 'object') {
      return null;
    }
    try {
      if (typeof source.getStats === 'function') {
        const s = source.getStats();
        if (
          s &&
          typeof s === 'object' &&
          (s.bc3dFallbacks !== undefined || s.bc3dFallbackUploadsSkipped !== undefined)
        ) {
          return s;
        }
      }
      if (
        source.stats &&
        typeof source.stats === 'object' &&
        (source.stats.bc3dFallbacks !== undefined ||
          source.stats.bc3dFallbackUploadsSkipped !== undefined)
      ) {
        return source.stats;
      }
      if (
        source.bc3dFallbacks !== undefined ||
        source.bc3dFallbackUploadsSkipped !== undefined
      ) {
        return source;
      }
    } catch (_) {
      return null;
    }
    return null;
  };

  if (stats && typeof stats === 'object') {
    raw = readSource(stats.textureStats) || readSource(stats.texturePackets);
  }
  if (!raw && processor && typeof processor === 'object') {
    const candidates = [processor.texturePackets, processor._texturePackets];
    if (typeof processor._getTexturePacketsExtension === 'function') {
      try {
        candidates.push(processor._getTexturePacketsExtension());
      } catch (_) {
        // ignore
      }
    } else if (processor.extension && typeof processor.extension === 'object') {
      let ext = processor.extension;
      while (ext) {
        candidates.push(ext);
        ext = ext.next || null;
      }
    }
    for (let i = 0; i < candidates.length; i += 1) {
      const found = readSource(candidates[i]);
      if (found) {
        raw = found;
        break;
      }
    }
  }
  if (!raw && stats && typeof stats === 'object') {
    raw = readSource(stats);
  }

  return {
    bc3dFallbacks: toNonNegativeInt(raw && raw.bc3dFallbacks, 0),
    bc3dFallbackUploadsSkipped: toNonNegativeInt(raw && raw.bc3dFallbackUploadsSkipped, 0),
  };
}

/**
 * W121 / W136 / W159: sanitized W109 WGSL pack mode, auto-threshold & ranged-fetch counters, or null when unavailable.
 *
 * @returns {{ mode: 'buffer' | 'ranged' | 'mixed', packCount: number, shaderCount: number, rangeFetches: number, cachedWgslBytes: number, rangedPackActive: boolean, rangedPackThresholdBytes: number, asyncShaderPendingCount: number, rangedShaderFetches: number, rangedShaderReady: number, rangedShaderErrors: number } | null}
 */
function resolveShaderPackTelemetry(processor, stats) {
  let raw = null;
  try {
    if (
      processor &&
      processor.wgslPackCache &&
      typeof processor.wgslPackCache.getStats === 'function'
    ) {
      raw = processor.wgslPackCache.getStats();
    } else if (stats && stats.wgslPackCache && typeof stats.wgslPackCache === 'object') {
      raw = stats.wgslPackCache;
    }
  } catch (_) {
    raw = null;
  }
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const rawMode = typeof raw.mode === 'string' ? raw.mode.trim().toLowerCase() : 'buffer';
  const mode =
    rawMode === 'ranged' || rawMode === 'mixed' || rawMode === 'buffer' ? rawMode : 'buffer';
  const rangedPackActive = mode === 'ranged' || mode === 'mixed';
  const rangedPackThresholdBytes = toNonNegativeInt(
    processor && processor._activeShaderPackStreamThresholdBytes !== undefined
      ? processor._activeShaderPackStreamThresholdBytes
      : processor && processor._shaderPackStreamThresholdBytes !== undefined
        ? processor._shaderPackStreamThresholdBytes
        : stats && stats.rangedPackThresholdBytes !== undefined
          ? stats.rangedPackThresholdBytes
          : raw.rangedPackThresholdBytes,
    DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES
  );
  const asyncShaderPendingCount = toNonNegativeInt(
    processor &&
      processor._inFlightRangedShaders &&
      typeof processor._inFlightRangedShaders.size === 'number'
      ? processor._inFlightRangedShaders.size
      : stats && stats.asyncShaderPendingCount !== undefined
        ? stats.asyncShaderPendingCount
        : raw.asyncShaderPendingCount,
    0
  );
  return {
    mode,
    packCount: toNonNegativeInt(raw.packCount, 0),
    shaderCount: toNonNegativeInt(raw.shaderCount, 0),
    rangeFetches: toNonNegativeInt(raw.rangeFetches, 0),
    cachedWgslBytes: toNonNegativeInt(raw.cachedWgslBytes, 0),
    rangedPackActive,
    rangedPackThresholdBytes,
    asyncShaderPendingCount,
    rangedShaderFetches: toNonNegativeInt(
      processor && processor._rangedShaderFetches !== undefined
        ? processor._rangedShaderFetches
        : stats && stats.rangedShaderFetches !== undefined
          ? stats.rangedShaderFetches
          : raw.rangedShaderFetches,
      0
    ),
    rangedShaderReady: toNonNegativeInt(
      processor && processor._rangedShaderReady !== undefined
        ? processor._rangedShaderReady
        : stats && stats.rangedShaderReady !== undefined
          ? stats.rangedShaderReady
          : raw.rangedShaderReady,
      0
    ),
    rangedShaderErrors: toNonNegativeInt(
      processor && processor._rangedShaderErrors !== undefined
        ? processor._rangedShaderErrors
        : stats && stats.rangedShaderErrors !== undefined
          ? stats.rangedShaderErrors
          : raw.rangedShaderErrors,
      0
    ),
    oversizedWgslFallbacks: toNonNegativeInt(
      processor && processor._oversizedWgslFallbacks !== undefined
        ? processor._oversizedWgslFallbacks
        : stats && stats.oversizedWgslFallbacks !== undefined
          ? stats.oversizedWgslFallbacks
          : raw.oversizedWgslFallbacks,
      0
    ),
  };
}

const ASYNC_PIPELINE_TELEMETRY_KEYS = Object.freeze([
  'generation',
  'maxConcurrent',
  'queued',
  'inFlight',
  'peakQueueDepth',
  'compiled',
  'failed',
  'dedupeHits',
  'syncFallbacks',
  'autoLayoutRetries',
  'discardedStale',
  'pendingAsyncPipelineSkips',
]);

/**
 * W103: sanitized AsyncPipelineCache counters, or null when the processor has no async cache.
 *
 * @returns {object | null}
 */
function resolveAsyncPipelineTelemetry(processor) {
  let raw = null;
  try {
    if (typeof processor.getAsyncPipelineStats === 'function') {
      raw = processor.getAsyncPipelineStats();
    } else if (
      processor.asyncPipelineCache &&
      typeof processor.asyncPipelineCache.getStats === 'function'
    ) {
      raw = processor.asyncPipelineCache.getStats();
    }
  } catch (_) {
    raw = null;
  }
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const out = {};
  for (let i = 0; i < ASYNC_PIPELINE_TELEMETRY_KEYS.length; i += 1) {
    const key = ASYNC_PIPELINE_TELEMETRY_KEYS[i];
    out[key] = toNonNegativeInt(raw[key], 0);
  }
  return out;
}

/**
 * W168: sanitized AsyncPipelineCache priority-queue and compile-latency telemetry,
 * or null when async pipeline compilation is disabled/unavailable.
 */
function resolveAsyncPipelineCompileTelemetry(processor) {
  if (!processor || typeof processor !== 'object') {
    return null;
  }
  let raw = null;
  const cache = processor.asyncPipelineCache || null;
  try {
    if (typeof processor.getAsyncPipelineStats === 'function') {
      raw = processor.getAsyncPipelineStats();
    } else if (cache && typeof cache.getStats === 'function') {
      raw = cache.getStats();
    }
  } catch (_) {
    raw = null;
  }
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const avgMs = toNonNegativeFinite(
    raw.avgCompileMs !== undefined ? raw.avgCompileMs : raw.avgAsyncCompileMs
  );
  const maxMs = toNonNegativeFinite(raw.maxCompileMs);
  return {
    prioritizeGraphics:
      typeof raw.prioritizeGraphics === 'boolean'
        ? raw.prioritizeGraphics
        : cache && typeof cache.prioritizeGraphics === 'boolean'
          ? cache.prioritizeGraphics
          : true,
    maxConcurrentCompiles: toNonNegativeInt(
      raw.maxConcurrentCompiles !== undefined
        ? raw.maxConcurrentCompiles
        : cache && cache.maxConcurrentCompiles !== undefined
          ? cache.maxConcurrentCompiles
          : raw.maxConcurrent,
      0
    ),
    activeCompiles: toNonNegativeInt(
      raw.activeCompiles !== undefined
        ? raw.activeCompiles
        : raw.inFlightCompiles !== undefined
          ? raw.inFlightCompiles
          : raw.inFlight,
      0
    ),
    peakConcurrentCompiles: toNonNegativeInt(
      raw.peakConcurrentCompiles !== undefined
        ? raw.peakConcurrentCompiles
        : raw.peakInFlightCompiles,
      0
    ),
    queuedCompiles: toNonNegativeInt(
      raw.queuedCompiles !== undefined ? raw.queuedCompiles : raw.queued,
      0
    ),
    failedAsyncCompiles: toNonNegativeInt(
      raw.failedAsyncCompiles !== undefined ? raw.failedAsyncCompiles : raw.failed,
      0
    ),
    avgCompileMs: avgMs !== null ? avgMs : 0,
    maxCompileMs: maxMs !== null ? maxMs : 0,
    priorityPromotions: toNonNegativeInt(raw.priorityPromotions, 0),
  };
}

/**
 * W179: sanitized, bounded categorized missing-shader summary from processor or stats.
 */
function resolveMissingShaderSummaryTelemetry(processor, stats, limit = DEFAULT_MISSING_SUMMARY_LIMIT) {
  let raw = null;
  try {
    if (processor && typeof processor.getMissingShaderSummary === 'function') {
      raw = processor.getMissingShaderSummary({ limit });
    } else if (
      processor &&
      processor.shaderFallbackRouter &&
      typeof processor.shaderFallbackRouter.getMissingShaderSummary === 'function'
    ) {
      raw = processor.shaderFallbackRouter.getMissingShaderSummary({ limit });
    } else if (stats && stats.missingShaderSummary && typeof stats.missingShaderSummary === 'object') {
      raw = stats.missingShaderSummary;
    }
  } catch (_) {
    raw = null;
  }
  return sanitizeMissingShaderSummary(raw, limit);
}

/**
 * W167: sanitized default fallback sampler synthesis counters.
 */
function resolveFallbackSamplerTelemetry(processor, stats) {
  let cacheStats = null;
  try {
    if (
      processor &&
      processor.pipelineCache &&
      typeof processor.pipelineCache.getStats === 'function'
    ) {
      cacheStats = processor.pipelineCache.getStats();
    } else if (stats && stats.pipelineCache && typeof stats.pipelineCache === 'object') {
      cacheStats = stats.pipelineCache;
    }
  } catch (_) {
    cacheStats = null;
  }
  const localCreated =
    processor && typeof processor._fallbackSamplersCreated === 'number'
      ? toNonNegativeInt(processor._fallbackSamplersCreated, 0)
      : 0;
  const localSynthesized =
    processor && typeof processor._fallbackSamplerBindingsSynthesized === 'number'
      ? toNonNegativeInt(processor._fallbackSamplerBindingsSynthesized, 0)
      : 0;
  const localCount =
    processor &&
    processor._defaultFallbackSamplers &&
    typeof processor._defaultFallbackSamplers.size === 'number'
      ? toNonNegativeInt(processor._defaultFallbackSamplers.size, 0)
      : 0;
  const cacheCreated = toNonNegativeInt(cacheStats && cacheStats.fallbackSamplersCreated, 0);
  const cacheSynthesized = toNonNegativeInt(
    cacheStats && cacheStats.fallbackSamplerBindingsSynthesized,
    0
  );
  const cacheCount = toNonNegativeInt(cacheStats && cacheStats.fallbackSamplerCount, 0);
  return {
    fallbackSamplersCreated:
      stats && stats.fallbackSamplersCreated !== undefined && cacheCreated + localCreated === 0
        ? toNonNegativeInt(stats.fallbackSamplersCreated, 0)
        : cacheCreated + localCreated,
    fallbackSamplerBindingsSynthesized:
      stats &&
      stats.fallbackSamplerBindingsSynthesized !== undefined &&
      cacheSynthesized + localSynthesized === 0
        ? toNonNegativeInt(stats.fallbackSamplerBindingsSynthesized, 0)
        : cacheSynthesized + localSynthesized,
    fallbackSamplerCount: cacheCount + localCount,
  };
}

/**
 * W100: read sanitized device-loss counters from `processor.getDeviceLossStats()` (or a
 * `stats.deviceLoss` object), defaulting to zeros / null / false.
 *
 * @returns {{ deviceLostCount: number, deviceRecoveredCount: number, lastDeviceLostReason: string | null, deviceRecovering: boolean }}
 */
function resolveDeviceLossTelemetry(processor, stats) {
  let raw = null;
  if (processor && typeof processor.getDeviceLossStats === 'function') {
    try {
      raw = processor.getDeviceLossStats();
    } catch (_) {
      raw = null;
    }
  }
  if (
    (!raw || typeof raw !== 'object') &&
    stats &&
    stats.deviceLoss &&
    typeof stats.deviceLoss === 'object'
  ) {
    raw = stats.deviceLoss;
  }
  const src = raw && typeof raw === 'object' ? raw : {};
  let lastDeviceLostReason = null;
  if (typeof src.lastDeviceLostReason === 'string' && src.lastDeviceLostReason.length > 0) {
    lastDeviceLostReason = sanitizeGpuErrorDetail({ message: src.lastDeviceLostReason });
  }
  const deviceStats = {
    deviceLostCount: toNonNegativeInt(src.deviceLostCount, 0),
    deviceRecoveredCount: toNonNegativeInt(src.deviceRecoveredCount, 0),
    lastDeviceLostReason,
    deviceRecovering: src.deviceRecovering === true,
  };
  // W104: live-resource replay counters (only when the processor tracks replay).
  if (src.replayedResourceCount !== undefined || src.replayedBufferWriteCount !== undefined) {
    deviceStats.replayedResourceCount = toNonNegativeInt(src.replayedResourceCount, 0);
    deviceStats.replayedBufferWriteCount = toNonNegativeInt(src.replayedBufferWriteCount, 0);
  }
  if (src.replayBindGroupEvictions !== undefined || src.replayLogOverflowCount !== undefined) {
    deviceStats.replayBindGroupEvictions = toNonNegativeInt(src.replayBindGroupEvictions, 0);
    deviceStats.replayLogOverflowCount = toNonNegativeInt(src.replayLogOverflowCount, 0);
  }
  return deviceStats;
}

const RENDER_STATE_TELEMETRY_KEYS = Object.freeze([
  'computeDispatchCount',
  'computePipelineCount',
  'autoLayoutFallbacks',
  'bindGroupKindMismatches',
]);

function hasRenderStateTelemetryKeys(obj) {
  if (!obj || typeof obj !== 'object') {
    return false;
  }
  for (let i = 0; i < RENDER_STATE_TELEMETRY_KEYS.length; i += 1) {
    if (obj[RENDER_STATE_TELEMETRY_KEYS[i]] !== undefined) {
      return true;
    }
  }
  return false;
}

function readRenderStateTelemetry(source) {
  if (!source || typeof source !== 'object') {
    return null;
  }
  try {
    if (typeof source.getTelemetry === 'function') {
      const t = source.getTelemetry();
      if (hasRenderStateTelemetryKeys(t)) return t;
    }
    if (hasRenderStateTelemetryKeys(source.telemetry)) {
      return source.telemetry;
    }
    if (typeof source.getStats === 'function') {
      const s = source.getStats();
      if (hasRenderStateTelemetryKeys(s)) return s;
    }
  } catch (_) {
    return null;
  }
  return hasRenderStateTelemetryKeys(source) ? source : null;
}

/**
 * Locate the live render_state_packets.js telemetry (compute dispatch / compute pipeline /
 * auto-layout fallback counters) attached to a processor, or null when absent.
 *
 * @param {object | null | undefined} processor
 * @param {object | null} stats
 * @returns {object | null}
 */
function resolveRenderStateTelemetry(processor, stats) {
  if (stats && typeof stats === 'object') {
    const fromStats = readRenderStateTelemetry(stats.renderState);
    if (fromStats) return fromStats;
  }
  if (processor && typeof processor === 'object') {
    const candidates = [processor.renderStateProcessor, processor._renderStatePackets];
    if (typeof processor._getRenderStateExtension === 'function') {
      try {
        candidates.push(processor._getRenderStateExtension());
      } catch (_) {
        // ignore
      }
    }
    for (let i = 0; i < candidates.length; i += 1) {
      const t = readRenderStateTelemetry(candidates[i]);
      if (t) return t;
    }
  }
  return hasRenderStateTelemetryKeys(stats) ? stats : null;
}

function createWgpuWorkerHandler(options = {}) {
  const postMessage =
    typeof options.postMessage === 'function' ? options.postMessage : () => {};
  const injectedRingCtor =
    typeof options.GpuRingBuffer === 'function' ? options.GpuRingBuffer : null;
  const telemetryIntervalMs =
    typeof options.telemetryIntervalMs === 'number' &&
    Number.isFinite(options.telemetryIntervalMs) &&
    options.telemetryIntervalMs >= 0
      ? options.telemetryIntervalMs
      : DEFAULT_TELEMETRY_INTERVAL_MS;
  const telemetryMissingLimit =
    Number.isInteger(options.telemetryMissingLimit) && options.telemetryMissingLimit >= 0
      ? options.telemetryMissingLimit
      : DEFAULT_TELEMETRY_MISSING_LIMIT;
  const telemetryNow = typeof options.now === 'function' ? options.now : () => Date.now();
  const onFirstPresent =
    typeof options.onFirstPresent === 'function' ? options.onFirstPresent : null;
  const onFirstVisibleFrame =
    typeof options.onFirstVisibleFrame === 'function' ? options.onFirstVisibleFrame : null;
  const onValidationError =
    typeof options.onValidationError === 'function' ? options.onValidationError : null;
  let canvas = options.canvas || null;
  let processor = options.processor || null;
  let pendingGuestMemory = null;  // G66: runtime memory attached before the processor exists
  let swapchain = options.swapchain || (processor && processor.swapchain) || null;
  let attachedRing = processor && processor.ring ? processor.ring : null;
  let currentGpuAdapter =
    options.gpuAdapter && typeof options.gpuAdapter === 'object'
      ? Object.freeze(sanitizeGpuAdapterInfo(options.gpuAdapter))
      : processor && processor.gpuAdapter && typeof processor.gpuAdapter === 'object'
        ? Object.freeze(sanitizeGpuAdapterInfo(processor.gpuAdapter))
        : null;
  let preloadedInDefaultInit = false;
  let lastTelemetryEmitMs = -Infinity;
  let firstPresentNotified = false;
  let firstVisibleFrameNotified = false;
  let unbindValidationListener = null;
  let unbindFirstVisibleFrameListener = null;

  // W100: bounded device-loss recovery state (per worker session).
  const deviceRecoveryEnabled = options.enableDeviceRecovery !== false;
  const maxDeviceRecoveries =
    Number.isInteger(options.maxDeviceRecoveries) && options.maxDeviceRecoveries >= 0
      ? options.maxDeviceRecoveries
      : MAX_DEVICE_RECOVERY_ATTEMPTS;
  const deviceRecoveryMinIntervalMs =
    typeof options.deviceRecoveryMinIntervalMs === 'number' &&
    Number.isFinite(options.deviceRecoveryMinIntervalMs) &&
    options.deviceRecoveryMinIntervalMs >= 0
      ? options.deviceRecoveryMinIntervalMs
      : DEFAULT_DEVICE_RECOVERY_MIN_INTERVAL_MS;
  const recoverGpuDevice =
    typeof options.recoverGpuDevice === 'function'
      ? options.recoverGpuDevice
      : (targetProcessor) => defaultRecoverGpuDevice(targetProcessor);
  let deviceRecoveryAttempts = 0;
  let deviceRecoveryInFlight = null;
  let deviceRecoveryExhaustedLogged = false;
  let lastDeviceRecoveryStartMs = -Infinity;
  let boundDeviceLossProcessor = null;

  function emitValidationError(record, totalCount) {
    const cleanRecord = {
      seq: record.seq,
      type: record.type,
      message: record.message,
      timestampMs: record.timestampMs,
    };
    const count = toNonNegativeInt(totalCount, cleanRecord.seq);
    postMessage({
      type: 'gpu-validation-error',
      error: cleanRecord,
      totalCount: count,
    });
    if (onValidationError) {
      try {
        onValidationError({ ...cleanRecord }, count);
      } catch (_) {
        // Non-fatal validation error callback failure.
      }
    }
  }

  function getActiveGpuAdapter() {
    if (processor && processor._gpuAdapter && typeof processor._gpuAdapter === 'object') {
      return processor._gpuAdapter;
    }
    if (currentGpuAdapter && typeof currentGpuAdapter === 'object') {
      return currentGpuAdapter;
    }
    return null;
  }

  function emitFirstVisibleFrame(info = {}) {
    if (firstVisibleFrameNotified) {
      return false;
    }
    firstVisibleFrameNotified = true;
    const payload = {
      type: 'first-visible-frame',
      firstVisibleFrame: true,
      firstVisibleFrameMs:
        typeof info.firstVisibleFrameMs === 'number' && Number.isFinite(info.firstVisibleFrameMs)
          ? info.firstVisibleFrameMs
          : telemetryNow(),
      firstPresentMs:
        typeof info.firstPresentMs === 'number' && Number.isFinite(info.firstPresentMs)
          ? info.firstPresentMs
          : null,
      blackLoadingMs:
        typeof info.blackLoadingMs === 'number' && Number.isFinite(info.blackLoadingMs)
          ? Math.max(0, info.blackLoadingMs)
          : 0,
      presentCount: toNonNegativeInt(info.presentCount, 0),
      realPackDrawsThisFrame: toNonNegativeInt(info.realPackDrawsThisFrame, 0),
      fallbackDrawsThisFrame: toNonNegativeInt(info.fallbackDrawsThisFrame, 0),
      drawsThisFrame: toNonNegativeInt(info.drawsThisFrame, 0),
      gpuAdapter:
        info.gpuAdapter && typeof info.gpuAdapter === 'object'
          ? sanitizeGpuAdapterInfo(info.gpuAdapter)
          : getActiveGpuAdapter(),
    };
    if (typeof info.source === 'string' && info.source.length > 0) {
      payload.source = info.source;
    }
    if (typeof info.maxLuma === 'number' && Number.isFinite(info.maxLuma)) {
      payload.maxLuma = Math.max(0, Math.min(255, Math.floor(info.maxLuma)));
    }
    if (
      typeof info.suppressedBlackFrames === 'number' &&
      Number.isFinite(info.suppressedBlackFrames)
    ) {
      payload.suppressedBlackFrames = toNonNegativeInt(info.suppressedBlackFrames, 0);
    }
    if (
      typeof info.swapchainProbeErrorCount === 'number' &&
      Number.isFinite(info.swapchainProbeErrorCount)
    ) {
      payload.swapchainProbeErrorCount = toNonNegativeInt(info.swapchainProbeErrorCount, 0);
    }
    postMessage(payload);
    if (onFirstVisibleFrame) {
      try {
        onFirstVisibleFrame(payload);
      } catch (_) {
        // Non-fatal telemetry callback error.
      }
    }
    return true;
  }

  function bindProcessorFirstVisibleFrame(targetProcessor) {
    if (unbindFirstVisibleFrameListener) {
      try {
        unbindFirstVisibleFrameListener();
      } catch (_) {
        // Ignore cleanup errors on prior processor.
      }
      unbindFirstVisibleFrameListener = null;
    }
    if (!targetProcessor) {
      return;
    }
    if (
      currentGpuAdapter &&
      !targetProcessor._gpuAdapter &&
      typeof targetProcessor.setGpuAdapter === 'function'
    ) {
      targetProcessor.setGpuAdapter(currentGpuAdapter);
    } else if (
      !currentGpuAdapter &&
      targetProcessor._gpuAdapter &&
      typeof targetProcessor._gpuAdapter === 'object'
    ) {
      currentGpuAdapter = Object.freeze(sanitizeGpuAdapterInfo(targetProcessor._gpuAdapter));
    }
    if (typeof targetProcessor.addFirstVisibleFrameListener === 'function') {
      unbindFirstVisibleFrameListener = targetProcessor.addFirstVisibleFrameListener((info) => {
        emitFirstVisibleFrame(info);
      });
    }
  }

  function bindProcessorValidationErrors(targetProcessor) {
    if (unbindValidationListener) {
      try {
        unbindValidationListener();
      } catch (_) {
        // Ignore cleanup errors on prior processor/device.
      }
      unbindValidationListener = null;
    }
    bindProcessorDeviceLoss(targetProcessor);
    bindProcessorFirstVisibleFrame(targetProcessor);
    if (!targetProcessor) {
      return;
    }
    if (
      swapchain &&
      !targetProcessor.swapchain &&
      typeof targetProcessor.attachSwapchain === 'function'
    ) {
      targetProcessor.attachSwapchain(swapchain);
    }
    if (typeof targetProcessor.addValidationErrorListener === 'function') {
      unbindValidationListener = targetProcessor.addValidationErrorListener(
        (record, totalCount) => {
          emitValidationError(record, totalCount);
        }
      );
      return;
    }
    const dev = targetProcessor.device;
    if (dev && typeof dev.addEventListener === 'function') {
      let localSeq = 0;
      const fallbackHandler = (event) => {
        const rawError =
          event && typeof event === 'object' && 'error' in event ? event.error : event;
        const type = normalizeValidationErrorType(rawError, event);
        const rawMsg =
          rawError && typeof rawError === 'object' && typeof rawError.message === 'string'
            ? rawError.message
            : event && typeof event === 'object' && typeof event.message === 'string'
              ? event.message
              : typeof rawError === 'string'
                ? rawError
                : rawError !== null && rawError !== undefined
                  ? String(rawError)
                  : '';
        const message = sanitizeValidationErrorMessage(rawMsg);
        const tsVal = telemetryNow();
        const timestampMs =
          typeof tsVal === 'number' && Number.isFinite(tsVal) ? tsVal : Date.now();
        localSeq += 1;
        emitValidationError({ seq: localSeq, type, message, timestampMs }, localSeq);
      };
      dev.addEventListener('uncapturederror', fallbackHandler);
      unbindValidationListener = () => {
        if (typeof dev.removeEventListener === 'function') {
          try {
            dev.removeEventListener('uncapturederror', fallbackHandler);
          } catch (_) {
            // Ignore removal failures.
          }
        }
      };
    }
  }

  if (processor) {
    bindProcessorValidationErrors(processor);
  }

  function notifyHandlerFirstPresent(info) {
    if (firstPresentNotified) {
      return false;
    }
    firstPresentNotified = true;
    if (onFirstPresent) {
      try {
        onFirstPresent(info);
      } catch (_) {
        // Non-fatal telemetry callback error.
      }
    }
    return true;
  }

  function emitTelemetry(overrideOptions = {}) {
    const ts = telemetryNow();
    const snap = buildGpuTelemetrySnapshot(processor, {
      limit:
        overrideOptions.limit !== undefined ? overrideOptions.limit : telemetryMissingLimit,
      timestamp: ts,
    });
    if (overrideOptions.id !== undefined) {
      snap.id = overrideOptions.id;
    }
    postMessage(snap);
    return snap;
  }

  function maybeEmitFrameTelemetry(prevFrameCount, nextFrameCount) {
    if (nextFrameCount <= prevFrameCount) {
      return false;
    }
    const nowMs = telemetryNow();
    const reachedFirstFrame = prevFrameCount === 0 && nextFrameCount >= 1;
    const isFrameOne = nextFrameCount === 1;
    const throttleElapsed =
      lastTelemetryEmitMs === -Infinity || nowMs - lastTelemetryEmitMs >= telemetryIntervalMs;

    if (reachedFirstFrame || isFrameOne || throttleElapsed) {
      lastTelemetryEmitMs = nowMs;
      emitTelemetry();
      return true;
    }
    return false;
  }

  async function preloadBootShaderAssets(targetProcessor, initPayload = {}) {
    const summary = {
      shaderPackLoaded: false,
      shaderPackShaders: 0,
      reflectionLoaded: false,
      reflectionShaders: 0,
    };
    if (!targetProcessor || !initPayload || typeof initPayload !== 'object') {
      return summary;
    }

    const activeFetch =
      typeof initPayload.fetch === 'function'
        ? initPayload.fetch
        : typeof options.fetch === 'function'
          ? options.fetch
          : typeof globalThis !== 'undefined' && typeof globalThis.fetch === 'function'
            ? globalThis.fetch
            : null;

    const shaderPackUrl =
      typeof initPayload.shaderPackUrl === 'string' && initPayload.shaderPackUrl.trim().length > 0
        ? initPayload.shaderPackUrl.trim()
        : null;
    const processorThresholdFallback =
      targetProcessor &&
      typeof targetProcessor._shaderPackStreamThresholdBytes === 'number'
        ? { streamThresholdBytes: targetProcessor._shaderPackStreamThresholdBytes }
        : null;
    const streamThresholdBytes = resolveShaderPackStreamThresholdBytes(
      initPayload,
      options,
      processorThresholdFallback
    );
    const initialRangeBytes =
      initPayload.initialRangeBytes !== undefined
        ? initPayload.initialRangeBytes
        : options.initialRangeBytes;
    const maxCachedWgslBytes =
      initPayload.maxCachedWgslBytes !== undefined
        ? initPayload.maxCachedWgslBytes
        : options.maxCachedWgslBytes;
    const fetchInit =
      initPayload.fetchInit !== undefined ? initPayload.fetchInit : options.fetchInit;
    const reflectionManifestUrl =
      typeof initPayload.reflectionManifestUrl === 'string' &&
      initPayload.reflectionManifestUrl.trim().length > 0
        ? initPayload.reflectionManifestUrl.trim()
        : null;

    const tasks = [];

    if (shaderPackUrl && activeFetch) {
      tasks.push(
        (async () => {
          try {
            if (targetProcessor && typeof targetProcessor === 'object') {
              targetProcessor._activeShaderPackStreamThresholdBytes = streamThresholdBytes;
            }
            const hasLoadPackUrl = Boolean(
              targetProcessor.wgslPackCache &&
                typeof targetProcessor.wgslPackCache.loadPackUrl === 'function'
            );
            if (hasLoadPackUrl) {
              const loadOpts = {
                fetch: activeFetch,
                packName: shaderPackUrl,
                streamThresholdBytes,
              };
              if (
                Number.isInteger(initialRangeBytes) &&
                initialRangeBytes >= WGPK_HEADER_BYTES
              ) {
                loadOpts.initialRangeBytes = initialRangeBytes;
              }
              if (Number.isInteger(maxCachedWgslBytes) && maxCachedWgslBytes >= 0) {
                loadOpts.maxCachedWgslBytes = maxCachedWgslBytes;
              }
              if (fetchInit !== undefined) {
                loadOpts.fetchInit = fetchInit;
              }
              await targetProcessor.wgslPackCache.loadPackUrl(shaderPackUrl, loadOpts);
              if (
                targetProcessor.shaderFallbackRouter &&
                targetProcessor.shaderFallbackRouter._packLoader == null &&
                typeof targetProcessor.shaderFallbackRouter.setPackLoader === 'function'
              ) {
                targetProcessor.shaderFallbackRouter.setPackLoader(targetProcessor.wgslPackCache);
              }
            } else if (typeof targetProcessor.loadShaderPackUrl === 'function') {
              await targetProcessor.loadShaderPackUrl(shaderPackUrl, {
                fetch: activeFetch,
                packName: shaderPackUrl,
                streamThresholdBytes,
                initialRangeBytes,
                maxCachedWgslBytes,
                fetchInit,
              });
            } else {
              const resp = await activeFetch(shaderPackUrl);
              if (!resp || !resp.ok || typeof resp.arrayBuffer !== 'function') {
                return;
              }
              const buffer = await resp.arrayBuffer();
              if (typeof targetProcessor.loadWgslPack === 'function') {
                targetProcessor.loadWgslPack(buffer, shaderPackUrl);
              } else if (targetProcessor.wgslPackCache) {
                loadWgslPackIntoCache(targetProcessor.wgslPackCache, buffer, shaderPackUrl);
              }
            }
            summary.shaderPackLoaded = true;
            if (
              targetProcessor.wgslPackCache &&
              typeof targetProcessor.wgslPackCache.getStats === 'function'
            ) {
              const packStats = targetProcessor.wgslPackCache.getStats();
              summary.shaderPackShaders = packStats ? packStats.shaderCount || 0 : 0;
            }
          } catch (err) {
            // Non-fatal: worker init must never abort if shader pack fetch/parse fails.
            if (typeof targetProcessor._logDiagnostic === 'function') {
              targetProcessor._logDiagnostic(
                'warn',
                `[wgpu_worker] shader pack preload failed: ${sanitizeGpuErrorDetail(err)}`
              );
            }
          }
        })()
      );
    }

    if (reflectionManifestUrl && activeFetch) {
      tasks.push(
        (async () => {
          try {
            const resp = await activeFetch(reflectionManifestUrl);
            if (!resp || !resp.ok) {
              return;
            }
            const manifestData =
              typeof resp.text === 'function'
                ? await resp.text()
                : typeof resp.json === 'function'
                  ? await resp.json()
                  : null;
            if (
              manifestData !== null &&
              typeof targetProcessor.loadReflectionManifest === 'function'
            ) {
              targetProcessor.loadReflectionManifest(manifestData);
              summary.reflectionLoaded = true;
              if (typeof targetProcessor.getStats === 'function') {
                const pStats = targetProcessor.getStats();
                summary.reflectionShaders = pStats ? pStats.reflectionShaderCount || 0 : 0;
              }
            }
          } catch (_) {
            // Non-fatal: worker init must never abort if reflection manifest fetch/parse fails.
          }
        })()
      );
    }

    if (tasks.length > 0) {
      await Promise.all(tasks);
    }
    return summary;
  }

  /**
   * Install the packet ABI v2 render-state + texture packet extensions on `targetProcessor`.
   * Also used after W100 device recovery, since extensions own device-bound objects.
   *
   * @returns {boolean} True when installation did not throw.
   */
  function installPacketExtensions(targetProcessor) {
    try {
      let renderStateApi =
        typeof globalThis !== 'undefined' && globalThis.PlayGta6RenderState
          ? globalThis.PlayGta6RenderState
          : null;
      if (!renderStateApi && typeof importScripts === 'function') {
        importScripts('./render_state_packets.js');
        renderStateApi =
          typeof globalThis !== 'undefined' && globalThis.PlayGta6RenderState
            ? globalThis.PlayGta6RenderState
            : null;
      } else if (!renderStateApi && typeof require === 'function') {
        renderStateApi = require('./render_state_packets.js');
      }
      if (renderStateApi && typeof renderStateApi.installRenderStatePackets === 'function') {
        renderStateApi.installRenderStatePackets(targetProcessor);
      }
      let texturePacketsApi =
        typeof globalThis !== 'undefined' && globalThis.PlayGta6TexturePackets
          ? globalThis.PlayGta6TexturePackets
          : null;
      if (!texturePacketsApi && typeof importScripts === 'function') {
        importScripts('./texture_packets.js');
        texturePacketsApi =
          typeof globalThis !== 'undefined' && globalThis.PlayGta6TexturePackets
            ? globalThis.PlayGta6TexturePackets
            : null;
      } else if (!texturePacketsApi && typeof require === 'function') {
        texturePacketsApi = require('./texture_packets.js');
      }
      if (texturePacketsApi && typeof texturePacketsApi.installTexturePackets === 'function') {
        texturePacketsApi.installTexturePackets(targetProcessor, {
          promote1DTo2D: targetProcessor._promote1DTo2D,
        });
      }
      return true;
    } catch (err) {
      postMessage({ error: `packet ABI v2 unavailable: ${String(err)}` });
      return false;
    }
  }

  /**
   * W100 default device re-acquisition: re-request adapter + device with the same feature /
   * limit negotiation as init and re-acquire the OffscreenCanvas `webgpu` context.
   *
   * @returns {Promise<{ device: GPUDevice, ctx: GPUCanvasContext | null, format: string }>}
   */
  async function defaultRecoverGpuDevice(targetProcessor) {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error('navigator.gpu unavailable in worker');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('no WebGPU adapter available');
    }
    const gpuAdapter = await extractSanitizedGpuAdapter(adapter);
    const { device } = await requestNegotiatedDevice(adapter);
    if (!device || typeof device !== 'object') {
      throw new Error('requestDevice returned no device');
    }
    let format = targetProcessor && targetProcessor.format ? targetProcessor.format : 'bgra8unorm';
    try {
      const preferred = navigator.gpu.getPreferredCanvasFormat();
      if (typeof preferred === 'string' && preferred.length > 0) {
        format = preferred;
      }
    } catch (_) {
      // Keep the previous format.
    }
    let ctx = targetProcessor ? targetProcessor.ctx : null;
    if (canvas && typeof canvas.getContext === 'function') {
      try {
        const nextCtx = canvas.getContext('webgpu');
        if (nextCtx) {
          ctx = nextCtx;
        }
      } catch (_) {
        // Fall back to the existing context object.
      }
    }
    return { device, ctx, format, gpuAdapter };
  }

  /**
   * W100: one async recovery attempt. Never rejects; always leaves recovery mode.
   */
  async function runDeviceRecovery(targetProcessor, attempt) {
    // Synchronous before the first await so submissions are dropped immediately.
    targetProcessor.beginDeviceRecovery();
    let ok = false;
    let acquiredDevice = null;
    try {
      const acquired = await recoverGpuDevice(targetProcessor, { canvas, attempt });
      if (!acquired || !acquired.device) {
        throw new Error('device recovery returned no device');
      }
      acquiredDevice = acquired.device;
      if (targetProcessor !== processor || targetProcessor.destroyed) {
        throw new Error('processor replaced or destroyed during recovery');
      }
      if (acquired.gpuAdapter && typeof acquired.gpuAdapter === 'object') {
        currentGpuAdapter = Object.freeze(sanitizeGpuAdapterInfo(acquired.gpuAdapter));
        if (typeof targetProcessor.setGpuAdapter === 'function') {
          targetProcessor.setGpuAdapter(currentGpuAdapter);
        }
      }
      targetProcessor.rebindDevice(acquired.device, acquired.ctx, {
        format: acquired.format,
        alphaMode: targetProcessor._alphaMode || 'opaque',
      });
      installPacketExtensions(targetProcessor);
      await ensureRtBvhPreludeLoaded();
      // W104/W105: recreate live pre-loss guest resources on the new device in async chunks so
      // Chrome's GPU IPC drains cleanly without stalling the worker.
      let replaySummary = null;
      if (typeof targetProcessor.replayRecordedResourcesAsync === 'function') {
        try {
          replaySummary = await targetProcessor.replayRecordedResourcesAsync();
        } catch (replayErr) {
          targetProcessor._logDiagnostic(
            'warn',
            `[wgpu_worker] GPU resource replay failed: ${sanitizeGpuErrorDetail(replayErr)}`
          );
        }
      } else if (typeof targetProcessor.replayRecordedResources === 'function') {
        try {
          replaySummary = targetProcessor.replayRecordedResources();
        } catch (replayErr) {
          targetProcessor._logDiagnostic(
            'warn',
            `[wgpu_worker] GPU resource replay failed: ${sanitizeGpuErrorDetail(replayErr)}`
          );
        }
      }
      bindProcessorValidationErrors(targetProcessor);
      ok = true;
      const replayedRes = replaySummary ? toNonNegativeInt(replaySummary.replayedResources, 0) : 0;
      const replayedWrites = replaySummary
        ? toNonNegativeInt(replaySummary.replayedBufferWrites, 0)
        : 0;
      const failedRes = replaySummary ? toNonNegativeInt(replaySummary.failedResources, 0) : 0;
      targetProcessor._logDiagnostic(
        'warn',
        `[wgpu_worker] WebGPU device recovered (attempt ${attempt}/${maxDeviceRecoveries}, replayed=${replayedRes}, bufferWrites=${replayedWrites}${failedRes > 0 ? `, failed=${failedRes}` : ''})`
      );
    } catch (err) {
      if (acquiredDevice && acquiredDevice !== targetProcessor.device) {
        try {
          if (typeof acquiredDevice.destroy === 'function') {
            acquiredDevice.destroy();
          }
        } catch (_) {
          // Ignore cleanup failure of an unused device.
        }
      }
      if (typeof targetProcessor._logDiagnostic === 'function') {
        targetProcessor._logDiagnostic(
          'error',
          `[wgpu_worker] WebGPU device recovery failed (attempt ${attempt}/${maxDeviceRecoveries}): ${sanitizeGpuErrorDetail(err)}`
        );
      }
    } finally {
      try {
        targetProcessor.endDeviceRecovery(ok);
      } catch (_) {
        // Never let bookkeeping escape.
      }
      try {
        if (targetProcessor === processor) {
          emitTelemetry();
        }
      } catch (_) {
        // Telemetry is best-effort.
      }
    }
    return ok;
  }

  /**
   * W100: device-loss listener. Re-entrancy guarded, rate limited (min interval between
   * attempts) and bounded (`maxDeviceRecoveries` attempts per worker session).
   */
  function handleProcessorDeviceLost(targetProcessor) {
    if (!deviceRecoveryEnabled || targetProcessor !== processor || deviceRecoveryInFlight) {
      return;
    }
    if (deviceRecoveryAttempts >= maxDeviceRecoveries) {
      if (!deviceRecoveryExhaustedLogged) {
        deviceRecoveryExhaustedLogged = true;
        targetProcessor._logDiagnostic(
          'error',
          `[wgpu_worker] WebGPU device recovery budget exhausted (${maxDeviceRecoveries} attempts); rendering stays disabled`
        );
      }
      return;
    }
    const nowVal = telemetryNow();
    const now = typeof nowVal === 'number' && Number.isFinite(nowVal) ? nowVal : Date.now();
    if (now - lastDeviceRecoveryStartMs < deviceRecoveryMinIntervalMs) {
      return;
    }
    lastDeviceRecoveryStartMs = now;
    deviceRecoveryAttempts += 1;
    const pending = runDeviceRecovery(targetProcessor, deviceRecoveryAttempts);
    deviceRecoveryInFlight = pending;
    pending.then(
      () => {
        if (deviceRecoveryInFlight === pending) {
          deviceRecoveryInFlight = null;
        }
      },
      () => {
        if (deviceRecoveryInFlight === pending) {
          deviceRecoveryInFlight = null;
        }
      }
    );
  }

  function bindProcessorDeviceLoss(targetProcessor) {
    if (boundDeviceLossProcessor && boundDeviceLossProcessor !== targetProcessor) {
      try {
        boundDeviceLossProcessor.setDeviceLossListener(null);
      } catch (_) {
        // Ignore cleanup errors on prior processor.
      }
    }
    boundDeviceLossProcessor = null;
    if (!targetProcessor || typeof targetProcessor.setDeviceLossListener !== 'function') {
      return;
    }
    targetProcessor.setDeviceLossListener(() => {
      handleProcessorDeviceLost(targetProcessor);
    });
    boundDeviceLossProcessor = targetProcessor;
  }

  async function defaultInitGpu(offscreen, width, height, initPayload = {}) {
    const rtPreludePromise = ensureRtBvhPreludeLoaded();
    canvas = offscreen;
    canvas.width = width;
    canvas.height = height;

    if (typeof navigator === 'undefined' || !navigator.gpu) {
      postMessage({ error: 'navigator.gpu unavailable in worker' });
      return null;
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      postMessage({ error: 'no WebGPU adapter available' });
      return null;
    }
    const gpuAdapter = Object.freeze(await extractSanitizedGpuAdapter(adapter));
    currentGpuAdapter = gpuAdapter;
    if (adapter.info && typeof adapter.info === 'object') {
      const desc = [adapter.info.vendor, adapter.info.architecture, adapter.info.description]
        .filter((s) => typeof s === 'string' && s.length > 0)
        .join(' / ');
      const adapterLogger =
        (options && options.logger) || (typeof console !== 'undefined' ? console : null);
      if (desc && adapterLogger && typeof adapterLogger.info === 'function') {
        adapterLogger.info(`[wgpu_worker] WebGPU adapter: ${sanitizeGpuErrorDetail(desc)}`);
      }
    }

    const { device, enabledFeatures } = await requestNegotiatedDevice(adapter);
    const format = navigator.gpu.getPreferredCanvasFormat();
    const ctx = canvas.getContext('webgpu');

    const enableSwapchain =
      initPayload.enableSwapchain !== undefined
        ? initPayload.enableSwapchain !== false
        : options.enableSwapchain !== false;
    const injectedSwapchain =
      initPayload.swapchain && typeof initPayload.swapchain === 'object'
        ? initPayload.swapchain
        : options.swapchain && typeof options.swapchain === 'object'
          ? options.swapchain
          : null;

    const wantCopySrcForProbe =
      initPayload.requireNonBlackFrame !== false &&
      initPayload.requireNonBlackPixels !== false &&
      options.requireNonBlackFrame !== false &&
      options.requireNonBlackPixels !== false;

    let activeSwapchain = null;
    let ownsSwapchain = false;
    if (injectedSwapchain) {
      activeSwapchain = injectedSwapchain;
      if (
        !activeSwapchain.isConfigured &&
        typeof activeSwapchain.configure === 'function'
      ) {
        configureSwapchainWithCopySrc(activeSwapchain, {
          width,
          height,
          devicePixelRatio: initPayload.devicePixelRatio || 1,
          format,
        });
      } else if (
        wantCopySrcForProbe &&
        activeSwapchain.hasCopySrcUsage === false &&
        typeof activeSwapchain.ensureCopySrcUsage === 'function'
      ) {
        try {
          activeSwapchain.ensureCopySrcUsage();
        } catch (_) {
          // Ignore best-effort COPY_SRC enablement on injected swapchain.
        }
      }
    } else if (enableSwapchain) {
      const injectedSwapchainCtor =
        initPayload.WgpuSwapchain !== undefined
          ? initPayload.WgpuSwapchain
          : options.WgpuSwapchain;
      const SwapchainCtor =
        injectedSwapchainCtor === null || injectedSwapchainCtor === false
          ? null
          : resolveSubsystemCtor(
              injectedSwapchainCtor,
              'PlayGta6WgpuSwapchain',
              'WgpuSwapchain',
              './wgpu_swapchain.js'
            );
      if (SwapchainCtor) {
        try {
          const swapOpts = {
            ...(options.swapchainOptions || {}),
            ...(initPayload.swapchainOptions || {}),
            canvas,
            device,
            context: ctx,
            format,
            enableDepth:
              initPayload.enableDepth !== undefined
                ? Boolean(initPayload.enableDepth)
                : options.enableDepth !== undefined
                  ? Boolean(options.enableDepth)
                  : false,
            preserveCopySrcUntilVisible: wantCopySrcForProbe,
            now: telemetryNow,
          };
          activeSwapchain = new SwapchainCtor(swapOpts);
          const swapConfigureOpts = {
            width,
            height,
            devicePixelRatio: initPayload.devicePixelRatio || 1,
          };
          if (
            swapOpts.usage !== undefined ||
            !wantCopySrcForProbe ||
            activeSwapchain._preserveCopySrcUntilVisible === true
          ) {
            activeSwapchain.configure(swapConfigureOpts);
          } else {
            configureSwapchainWithCopySrc(activeSwapchain, swapConfigureOpts);
          }
          ownsSwapchain = true;
        } catch (_) {
          if (activeSwapchain && typeof activeSwapchain.destroy === 'function') {
            try {
              activeSwapchain.destroy();
            } catch (_destroyErr) {
              // Ignore teardown error on fallback.
            }
          }
          activeSwapchain = null;
          ownsSwapchain = false;
        }
      }
    }

    const resolvedAlphaMode = normalizeSwapchainAlphaMode(
      initPayload.alphaMode ??
        options.alphaMode ??
        (initPayload.swapchainOptions && initPayload.swapchainOptions.alphaMode) ??
        (options.swapchainOptions && options.swapchainOptions.alphaMode)
    );
    if (!activeSwapchain) {
      configureCanvasContextWithCopySrc(ctx, {
        device,
        format,
        alphaMode: resolvedAlphaMode,
      });
    }
    swapchain = activeSwapchain;

    await rtPreludePromise;

    const resolvedCanvasFormat =
      activeSwapchain &&
      typeof activeSwapchain.format === 'string' &&
      activeSwapchain.format.length > 0
        ? activeSwapchain.format
        : format;

    processor = new WgpuCommandProcessor(device, ctx, {
      format: resolvedCanvasFormat,
      alphaMode: resolvedAlphaMode,
      hdrMode:
        initPayload.hdrMode ??
        initPayload.swapchainHdrMode ??
        options.hdrMode ??
        options.swapchainHdrMode ??
        'sdr-tonemap',
      directSrgbView:
        initPayload.directSrgbView ??
        options.directSrgbView ??
        (initPayload.swapchainOptions && initPayload.swapchainOptions.directSrgbView) ??
        (options.swapchainOptions && options.swapchainOptions.directSrgbView),
      toneMapMode:
        initPayload.toneMapMode ??
        options.toneMapMode ??
        (initPayload.swapchainOptions && initPayload.swapchainOptions.toneMapMode) ??
        (options.swapchainOptions && options.swapchainOptions.toneMapMode),
      exposure:
        initPayload.exposure ??
        options.exposure ??
        (initPayload.swapchainOptions && initPayload.swapchainOptions.exposure) ??
        (options.swapchainOptions && options.swapchainOptions.exposure),
      applySrgbEncode:
        initPayload.applySrgbEncode ??
        options.applySrgbEncode ??
        (initPayload.swapchainOptions && initPayload.swapchainOptions.applySrgbEncode) ??
        (options.swapchainOptions && options.swapchainOptions.applySrgbEncode),
      rgb10a2Transfer:
        initPayload.rgb10a2Transfer ??
        options.rgb10a2Transfer ??
        (initPayload.swapchainOptions && initPayload.swapchainOptions.rgb10a2Transfer) ??
        (options.swapchainOptions && options.swapchainOptions.rgb10a2Transfer),
      width,
      height,
      now: telemetryNow,
      gpuAdapter,
      drawExecutor: initPayload.drawExecutor ?? options.drawExecutor,
      enableDrawExecutor: initPayload.enableDrawExecutor ?? options.enableDrawExecutor,
      DrawExecutor:
        initPayload.DrawExecutor !== undefined
          ? initPayload.DrawExecutor
          : options.DrawExecutor,
      drawExecutorOptions:
        initPayload.drawExecutorOptions ?? options.drawExecutorOptions,
      reuseDrawResult:
        initPayload.reuseDrawResult !== undefined
          ? initPayload.reuseDrawResult
          : options.reuseDrawResult,
      resourceTable: initPayload.resourceTable ?? options.resourceTable,
      oversizedRtReclaimIdleFrames:
        initPayload.oversizedRtReclaimIdleFrames ?? options.oversizedRtReclaimIdleFrames,
      oversizedRtReclaimIntervalFrames:
        initPayload.oversizedRtReclaimIntervalFrames ??
        options.oversizedRtReclaimIntervalFrames,
      swapchain: activeSwapchain,
      ownsSwapchain,
      logger: options.logger,
      captureValidationErrors: true,
      maxValidationErrorHistory:
        initPayload.maxValidationErrorHistory ?? options.maxValidationErrorHistory,
      maxValidationErrorPosts:
        initPayload.maxValidationErrorPosts ?? options.maxValidationErrorPosts,
      defaultFallbackProfile:
        initPayload.defaultFallbackProfile ?? options.defaultFallbackProfile,
      GpuRingBuffer: injectedRingCtor,
      stagingPool: options.stagingPool,
      uniformRing: options.uniformRing,
      passTimer: options.passTimer ?? options.gpuTimer,
      gpuTimer: options.gpuTimer,
      passTimerOptions: initPayload.passTimerOptions ?? options.passTimerOptions ?? initPayload.gpuTimerOptions ?? options.gpuTimerOptions,
      gpuRecorder: initPayload.gpuRecorder ?? options.gpuRecorder,
      GpuFrameRecorder:
        initPayload.GpuFrameRecorder !== undefined
          ? initPayload.GpuFrameRecorder
          : options.GpuFrameRecorder,
      gpuRecorderOptions: initPayload.gpuRecorderOptions ?? options.gpuRecorderOptions,
      reflectionRegistry: options.reflectionRegistry,
      resourceRegistry: options.resourceRegistry,
      rtPool: options.rtPool,
      requireNonBlackFrame:
        initPayload.requireNonBlackFrame !== undefined
          ? initPayload.requireNonBlackFrame
          : options.requireNonBlackFrame,
      swapchainProbeIntervalMs:
        initPayload.swapchainProbeIntervalMs !== undefined
          ? initPayload.swapchainProbeIntervalMs
          : options.swapchainProbeIntervalMs,
      swapchainProbeTimeoutMs:
        initPayload.swapchainProbeTimeoutMs !== undefined
          ? initPayload.swapchainProbeTimeoutMs
          : options.swapchainProbeTimeoutMs,
      gpuStallWarnMs:
        initPayload.gpuStallWarnMs !== undefined
          ? initPayload.gpuStallWarnMs
          : options.gpuStallWarnMs,
      gpuStallWarnMinIntervalMs:
        initPayload.gpuStallWarnMinIntervalMs !== undefined
          ? initPayload.gpuStallWarnMinIntervalMs
          : options.gpuStallWarnMinIntervalMs,
      minVisibleLuma: initPayload.minVisibleLuma ?? options.minVisibleLuma,
      minVisiblePixels: initPayload.minVisiblePixels ?? options.minVisiblePixels,
      nonBlackPixelProbe:
        initPayload.nonBlackPixelProbe ??
        options.nonBlackPixelProbe ??
        initPayload.pixelProbe ??
        options.pixelProbe,
      pipelineCache: initPayload.pipelineCache ?? options.pipelineCache,
      enablePipelineCache:
        initPayload.enablePipelineCache ?? options.enablePipelineCache,
      WebGpuPipelineCache:
        initPayload.WebGpuPipelineCache !== undefined
          ? initPayload.WebGpuPipelineCache
          : options.WebGpuPipelineCache,
      pipelineCacheOptions:
        initPayload.pipelineCacheOptions ?? options.pipelineCacheOptions,
      asyncPipelineCache:
        initPayload.asyncPipelineCache ?? options.asyncPipelineCache,
      AsyncPipelineCache:
        initPayload.AsyncPipelineCache !== undefined
          ? initPayload.AsyncPipelineCache
          : options.AsyncPipelineCache,
      enableAsyncPipelines:
        initPayload.enableAsyncPipelines !== undefined
          ? initPayload.enableAsyncPipelines
          : options.enableAsyncPipelines,
      maxConcurrentCompiles:
        initPayload.maxConcurrentCompiles !== undefined
          ? initPayload.maxConcurrentCompiles
          : options.maxConcurrentCompiles,
      maxConcurrentPipelineCompiles:
        initPayload.maxConcurrentPipelineCompiles !== undefined
          ? initPayload.maxConcurrentPipelineCompiles
          : options.maxConcurrentPipelineCompiles,
      prioritizeGraphics:
        initPayload.prioritizeGraphics !== undefined
          ? initPayload.prioritizeGraphics
          : options.prioritizeGraphics,
      wgslPackCache: initPayload.wgslPackCache ?? options.wgslPackCache,
      wgslPackBuffer: initPayload.wgslPackBuffer ?? options.wgslPackBuffer,
      streamThresholdBytes:
        initPayload.streamThresholdBytes ?? options.streamThresholdBytes,
      streamShaders: initPayload.streamShaders ?? options.streamShaders,
      shaderFallbackRouter: options.shaderFallbackRouter,
      ShaderFallbackRouter: options.ShaderFallbackRouter,
      shaderFallbackRouterOptions: options.shaderFallbackRouterOptions,
      binkPresenter: initPayload.binkPresenter ?? options.binkPresenter,
      splashFrameBridge: initPayload.splashFrameBridge ?? options.splashFrameBridge,
      BinkVideoPresenter:
        initPayload.BinkVideoPresenter !== undefined
          ? initPayload.BinkVideoPresenter
          : options.BinkVideoPresenter,
      SplashFrameBridge:
        initPayload.SplashFrameBridge !== undefined
          ? initPayload.SplashFrameBridge
          : options.SplashFrameBridge,
      binkPresenterOptions: buildMergedBinkPresenterOptions(
        {
          ...(options.binkPresenterOptions || {}),
          ...(options.enableTripleBuffering !== undefined &&
          (!options.binkPresenterOptions ||
            options.binkPresenterOptions.enableTripleBuffering === undefined)
            ? { enableTripleBuffering: Boolean(options.enableTripleBuffering) }
            : {}),
          ...(options.allowUnalignedDirectUpload !== undefined &&
          (!options.binkPresenterOptions ||
            options.binkPresenterOptions.allowUnalignedDirectUpload === undefined)
            ? { allowUnalignedDirectUpload: Boolean(options.allowUnalignedDirectUpload) }
            : {}),
        },
        {
          ...(initPayload.binkPresenterOptions || {}),
          ...(initPayload.enableTripleBuffering !== undefined &&
          (!initPayload.binkPresenterOptions ||
            initPayload.binkPresenterOptions.enableTripleBuffering === undefined)
            ? { enableTripleBuffering: Boolean(initPayload.enableTripleBuffering) }
            : {}),
          ...(initPayload.allowUnalignedDirectUpload !== undefined &&
          (!initPayload.binkPresenterOptions ||
            initPayload.binkPresenterOptions.allowUnalignedDirectUpload === undefined)
            ? { allowUnalignedDirectUpload: Boolean(initPayload.allowUnalignedDirectUpload) }
            : {}),
        }
      ),
      binkChromaOrder:
        initPayload.binkChromaOrder ??
        initPayload.chromaOrder ??
        options.binkChromaOrder ??
        options.chromaOrder,
      binkLoadOp:
        initPayload.binkLoadOp ??
        initPayload.loadOp ??
        options.binkLoadOp ??
        options.loadOp,
      splashFrameBridgeOptions:
        initPayload.splashFrameBridgeOptions ??
        initPayload.splashBridgeOptions ??
        options.splashFrameBridgeOptions ??
        options.splashBridgeOptions,
      enableStagingPool: initPayload.enableStagingPool ?? options.enableStagingPool,
      enableUniformRing: initPayload.enableUniformRing ?? options.enableUniformRing,
      enablePassTimer:
        initPayload.enablePassTimer ??
        initPayload.enableGpuTimer ??
        options.enablePassTimer ??
        options.enableGpuTimer,
      enableGpuRecorder: initPayload.enableGpuRecorder ?? options.enableGpuRecorder,
      enableReflectionRegistry:
        initPayload.enableReflectionRegistry ?? options.enableReflectionRegistry,
      enableResourceRegistry:
        initPayload.enableResourceRegistry ?? options.enableResourceRegistry,
      enableRtPool: initPayload.enableRtPool ?? options.enableRtPool,
      enableShaderFallback:
        initPayload.enableShaderFallback ?? options.enableShaderFallback,
      enableResourceReplay:
        initPayload.enableResourceReplay !== undefined
          ? initPayload.enableResourceReplay
          : options.enableResourceReplay,
      enableShadowBufferReplay:
        initPayload.enableShadowBufferReplay !== undefined
          ? initPayload.enableShadowBufferReplay
          : options.enableShadowBufferReplay,
      enableSplashBridge:
        initPayload.enableSplashBridge ?? options.enableSplashBridge,
      fallbackShaderHashes:
        initPayload.fallbackShaderHashes ?? options.fallbackShaderHashes,
      reflectionManifest: initPayload.reflectionManifest ?? options.reflectionManifest,
      lazyShaderCompile:
        initPayload.lazyShaderCompile !== undefined
          ? initPayload.lazyShaderCompile
          : options.lazyShaderCompile !== undefined
            ? options.lazyShaderCompile
            : 'pack',
      lazyRangedShaderCompile:
        initPayload.lazyRangedShaderCompile !== undefined
          ? initPayload.lazyRangedShaderCompile
          : options.lazyRangedShaderCompile !== undefined
            ? options.lazyRangedShaderCompile
            : true,
      maxWgslCompileBytes:
        initPayload.maxWgslCompileBytes !== undefined
          ? initPayload.maxWgslCompileBytes
          : options.maxWgslCompileBytes,
      maxConcurrentShaderCompiles:
        initPayload.maxConcurrentShaderCompiles !== undefined
          ? initPayload.maxConcurrentShaderCompiles
          : options.maxConcurrentShaderCompiles,
      shaderCompileSliceBudgetMs:
        initPayload.shaderCompileSliceBudgetMs !== undefined
          ? initPayload.shaderCompileSliceBudgetMs
          : options.shaderCompileSliceBudgetMs,
      maxPendingRangedPipelines:
        initPayload.maxPendingRangedPipelines !== undefined
          ? initPayload.maxPendingRangedPipelines
          : options.maxPendingRangedPipelines,
      deferRangedFallbackPipelines:
        initPayload.deferRangedFallbackPipelines !== undefined
          ? initPayload.deferRangedFallbackPipelines
          : options.deferRangedFallbackPipelines,
      coalescePresents:
        initPayload.coalescePresents !== undefined
          ? initPayload.coalescePresents
          : initPayload.coalescePresent !== undefined
            ? initPayload.coalescePresent
            : options.coalescePresents,
      promote1DTo2D:
        initPayload.promote1DTo2D !== undefined
          ? initPayload.promote1DTo2D
          : options.promote1DTo2D,
    });
    bindProcessorValidationErrors(processor);
    installPacketExtensions(processor);
    if (attachedRing) {
      processor.attachRing(attachedRing);
    }
    if (pendingGuestMemory && typeof processor.attachGuestMemory === 'function') {
      processor.attachGuestMemory(pendingGuestMemory);
    }

    await preloadBootShaderAssets(processor, initPayload);
    preloadedInDefaultInit = true;

    const info = adapter.info
      ? `${adapter.info.vendor || 'gpu'} ${adapter.info.architecture || ''}`.trim()
      : gpuAdapter && (gpuAdapter.vendor || gpuAdapter.architecture)
        ? `${gpuAdapter.vendor || 'gpu'} ${gpuAdapter.architecture || ''}`.trim()
        : 'WebGPU';
    postMessage({ ready: true, adapter: info, gpuAdapter, features: enabledFeatures });
    return processor;
  }

  const initGpu = typeof options.initGpu === 'function' ? options.initGpu : defaultInitGpu;

  function attachRingFromMessage(msg) {
    const ringSource = msg.ring || msg.ringSab;
    if (!ringSource) {
      throw new Error('attachRing requires ringSab or ring');
    }
    const RingCtor =
      ringSource &&
      (typeof ringSource.drainBatchZeroAlloc === 'function' ||
        typeof ringSource.drainTo === 'function')
        ? null
        : resolveGpuRingBuffer(msg.GpuRingBuffer || injectedRingCtor);
    attachedRing = RingCtor
      ? new RingCtor(ringSource, { byteOffset: msg.byteOffset || 0 })
      : ringSource;
    if (processor) {
      processor.attachRing(attachedRing);
    }
    postMessage({
      ringAttached: true,
      capacityBytes: attachedRing.capacityBytes,
    });
    return attachedRing;
  }

  // W363: Reusable options scratch + shared resolved promise for hot synchronous handleMessage paths
  // (avoids per-message Uint8Array, options, and Promise/async-frame allocations).
  const cmdBufferExecOpts = { reuseResult: true, byteLength: 0 };
  const cmdBufferExecCoalesceOpts = { reuseResult: true, byteLength: 0, coalescePresents: true };
  const splashPresentOptsScratch = {
    nowMs: undefined,
    respectPacing: undefined,
    chromaOrder: undefined,
    loadOp: undefined,
    colorRange: undefined,
    targetWidth: undefined,
    targetHeight: undefined,
  };
  const RESOLVED_VOID_PROMISE = Promise.resolve();
  // Wave-4 (#663): Worker-side pending command-batch queue for coalescing back-to-back Presents
  // arriving while async init or a shader-compile stall is active.
  let initInFlight = false;
  const pendingCommandBatches = [];

  function extractRawBatchBuffer(payload) {
    if (!payload || typeof payload !== 'object') {
      return null;
    }
    if (payload.cmdBuffer) {
      return payload.cmdBuffer;
    }
    if (payload.type === 'submit' || payload.type === 'executeBatch') {
      return payload.buffer || payload.commands || payload.data || null;
    }
    return null;
  }

  function flushPendingCommandBatches() {
    if (!processor || pendingCommandBatches.length === 0) {
      return { frameCount: processor ? toNonNegativeInt(processor.frameCount, 0) : 0, presents: 0, coalescedPresents: 0 };
    }
    const queued = pendingCommandBatches.splice(0, pendingCommandBatches.length);
    if (
      pendingGuestMemory &&
      processor._guestMemory !== pendingGuestMemory &&
      typeof processor.attachGuestMemory === 'function'
    ) {
      processor.attachGuestMemory(pendingGuestMemory);
    }
    const prevFrames = toNonNegativeInt(processor.frameCount, 0);
    const recyclable = [];
    for (let i = 0; i < queued.length; i += 1) {
      const p = queued[i];
      if (
        p &&
        (p.frameLuminance !== undefined || p.luminance !== undefined) &&
        typeof processor.setFrameLuminanceMeta === 'function'
      ) {
        processor.setFrameLuminanceMeta(
          p.frameLuminance !== undefined ? p.frameLuminance : p.luminance
        );
      }
      const rawBuf = extractRawBatchBuffer(p);
      if (
        p &&
        p.recycleBuffer === true &&
        p.cmdBuffer === rawBuf &&
        isPlainArrayBuffer(rawBuf) &&
        rawBuf.byteLength >= RECYCLE_CMD_BUFFER_MIN_BYTES &&
        rawBuf.byteLength <= RECYCLE_CMD_BUFFER_MAX_BYTES
      ) {
        recyclable.push(rawBuf);
      }
    }
    let result;
    try {
      if (typeof processor.executeCommandBatches === 'function') {
        result = processor.executeCommandBatches(queued);
      } else {
        let presents = 0;
        for (let i = 0; i < queued.length; i += 1) {
          const rawBuf = extractRawBatchBuffer(queued[i]);
          if (rawBuf) {
            const r = processor.executeCommandBuffer(rawBuf, DRAIN_RING_REUSE_OPTS);
            presents += r && typeof r.presents === 'number' ? r.presents : 0;
          }
        }
        result = { frameCount: toNonNegativeInt(processor.frameCount, prevFrames), presents, coalescedPresents: 0 };
      }
    } finally {
      for (let i = 0; i < recyclable.length; i += 1) {
        const rawBuf = recyclable[i];
        if (processor._cachedCmdBuffer === rawBuf) {
          processor._cachedCmdBuffer = null;
          processor._cachedCmdView = null;
        }
        try {
          postMessage({ type: 'recycle-cmd-buffer', cmdBuffer: rawBuf }, [rawBuf]);
        } catch (_) {
          // Ignore transfer errors if the buffer was detached or the channel closed.
        }
      }
    }
    if (result && result.presents > 0) {
      if (prevFrames === 0 && result.frameCount >= 1) {
        notifyHandlerFirstPresent({
          type: 'first-present',
          firstPresent: true,
          source: 'command-buffer',
          frameIndex: 0,
          framesPresented: result.frameCount,
          timestamp: telemetryNow(),
          gpuAdapter: getActiveGpuAdapter(),
        });
      }
      postMessage({ frame: result.frameCount });
      maybeEmitFrameTelemetry(prevFrames, result.frameCount);
    }
    return result;
  }

  async function handleAsyncInitWithCanvas(payload) {
    initInFlight = true;
    try {
      preloadedInDefaultInit = false;
      await ensureRtBvhPreludeLoaded();
      const created = await initGpu(
        payload.canvas,
        payload.width || 1280,
        payload.height || 720,
        payload
      );
      if (created && created !== processor) {
        processor = created;
        swapchain = created.swapchain || swapchain;
        lastTelemetryEmitMs = -Infinity;
        firstPresentNotified = false;
        firstVisibleFrameNotified = false;
        bindProcessorValidationErrors(processor);
      }
      if (
        processor &&
        pendingGuestMemory &&
        processor._guestMemory !== pendingGuestMemory &&
        typeof processor.attachGuestMemory === 'function'
      ) {
        processor.attachGuestMemory(pendingGuestMemory);
      }
      if (
        processor &&
        payload.enableResourceReplay !== undefined &&
        typeof processor.setEnableResourceReplay === 'function'
      ) {
        processor.setEnableResourceReplay(payload.enableResourceReplay !== false);
      }
      if (
        processor &&
        payload.enableShadowBufferReplay !== undefined &&
        typeof processor.setEnableShadowBufferReplay === 'function'
      ) {
        processor.setEnableShadowBufferReplay(payload.enableShadowBufferReplay !== false);
      }
      if (payload.ringSab || payload.ring) {
        attachRingFromMessage(payload);
      } else if (attachedRing && processor) {
        processor.attachRing(attachedRing);
      }
      if (!preloadedInDefaultInit && processor) {
        await preloadBootShaderAssets(processor, payload);
      }
    } catch (err) {
      postMessage({ error: String(err) });
    } finally {
      initInFlight = false;
      if (processor && pendingCommandBatches.length > 0) {
        flushPendingCommandBatches();
      } else if (!processor && pendingCommandBatches.length > 0) {
        const abandoned = pendingCommandBatches.splice(0, pendingCommandBatches.length);
        for (let i = 0; i < abandoned.length; i += 1) {
          const p = abandoned[i];
          const rawBuf = extractRawBatchBuffer(p);
          if (
            p &&
            p.recycleBuffer === true &&
            p.cmdBuffer === rawBuf &&
            isPlainArrayBuffer(rawBuf) &&
            rawBuf.byteLength >= RECYCLE_CMD_BUFFER_MIN_BYTES &&
            rawBuf.byteLength <= RECYCLE_CMD_BUFFER_MAX_BYTES
          ) {
            try {
              postMessage({ type: 'recycle-cmd-buffer', cmdBuffer: rawBuf }, [rawBuf]);
            } catch (_) {
              // Ignore transfer errors if the buffer was detached or the channel closed.
            }
          }
        }
      }
    }
  }

  async function handleAsyncInitExisting(payload) {
    try {
      await ensureRtBvhPreludeLoaded();
      if (
        payload.enableResourceReplay !== undefined &&
        typeof processor.setEnableResourceReplay === 'function'
      ) {
        processor.setEnableResourceReplay(payload.enableResourceReplay !== false);
      }
      if (
        payload.enableShadowBufferReplay !== undefined &&
        typeof processor.setEnableShadowBufferReplay === 'function'
      ) {
        processor.setEnableShadowBufferReplay(payload.enableShadowBufferReplay !== false);
      }
      await preloadBootShaderAssets(processor, payload);
    } catch (err) {
      postMessage({ error: String(err) });
    }
  }

  async function handleAsyncLoadShaderPack(payload) {
    try {
      await ensureRtBvhPreludeLoaded();
      const packUrl =
        typeof payload.loadShaderPack === 'string'
          ? payload.loadShaderPack
          : typeof payload.shaderPackUrl === 'string'
            ? payload.shaderPackUrl
            : typeof payload.url === 'string'
              ? payload.url
              : null;
      const loadSummary = packUrl
        ? await preloadBootShaderAssets(processor, {
            ...payload,
            shaderPackUrl: packUrl,
          })
        : { shaderPackLoaded: false, shaderPackShaders: 0 };
      const packStats =
        processor.wgslPackCache && typeof processor.wgslPackCache.getStats === 'function'
          ? processor.wgslPackCache.getStats()
          : null;
      const packMode = packStats && typeof packStats.mode === 'string' ? packStats.mode : 'buffer';
      postMessage({
        id: payload.id,
        shaderPackLoaded: loadSummary.shaderPackLoaded,
        shaderCount: loadSummary.shaderPackShaders,
        mode: packMode,
        rangedPackActive: packMode === 'ranged' || packMode === 'mixed',
        rangedPackThresholdBytes:
          processor && processor._activeShaderPackStreamThresholdBytes !== undefined
            ? toNonNegativeInt(
                processor._activeShaderPackStreamThresholdBytes,
                DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES
              )
            : DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES,
      });
    } catch (err) {
      postMessage({ error: String(err) });
    }
  }

  function handleMessage(msg) {
    const payload = msg || {};
    try {
      if (payload.init && payload.canvas) {
        return handleAsyncInitWithCanvas(payload);
      } else if (payload.init && processor) {
        return handleAsyncInitExisting(payload);
      } else if (payload.resize && canvas) {
        const nextW = Math.max(1, payload.width | 0);
        const nextH = Math.max(1, payload.height | 0);
        const activeSwapchain = (processor && processor.swapchain) || swapchain;
        if (
          activeSwapchain &&
          !activeSwapchain.isDestroyed &&
          typeof activeSwapchain.resize === 'function'
        ) {
          activeSwapchain.resize(nextW, nextH, payload.devicePixelRatio || 1);
        } else {
          const dimChanged = canvas.width !== nextW || canvas.height !== nextH;
          canvas.width = nextW;
          canvas.height = nextH;
          if (
            dimChanged &&
            processor &&
            !processor.firstVisibleFrame &&
            processor._swapchainProbeSupported !== false &&
            processor.ctx &&
            processor.device
          ) {
            const resizeCfg = {
              device: processor.device,
              format: processor.format,
              alphaMode: processor._alphaMode || 'opaque',
            };
            if (processor._swapchainViewFormat) {
              resizeCfg.viewFormats = [processor._swapchainViewFormat];
            }
            try {
              configureCanvasContextWithViewFormats(processor.ctx, resizeCfg);
            } catch (_) {
              // Ignore best-effort raw canvas reconfigure errors.
            }
          }
        }
        if (processor && typeof processor.onResize === 'function') {
          processor.onResize(nextW, nextH);
        }
      } else if (payload.attachRing) {
        attachRingFromMessage(payload);
      } else if (payload.loadReflectionManifest && processor) {
        processor.loadReflectionManifest(payload.loadReflectionManifest);
        const stats = processor.getStats();
        postMessage({
          id: payload.id,
          reflectionLoaded: true,
          shaderCount: stats.reflectionShaderCount,
        });
      } else if (
        (payload.loadShaderPack ||
          payload.type === 'load-shader-pack' ||
          payload.type === 'loadShaderPack') &&
        processor
      ) {
        return handleAsyncLoadShaderPack(payload);
      } else if (payload.type === 'request-gpu-telemetry') {
        emitTelemetry({
          id: payload.id,
          limit: payload.limit,
        });
      } else if (payload.missingShaders && processor) {
        const missingOpts =
          payload.limit !== undefined ? { limit: payload.limit } : undefined;
        postMessage({
          id: payload.id,
          missingShaders:
            typeof processor.getMissingShaders === 'function'
              ? processor.getMissingShaders(missingOpts)
              : [],
          shaderFallbackStats:
            typeof processor.getShaderFallbackStats === 'function'
              ? processor.getShaderFallbackStats()
              : null,
          missingShaderSummary:
            typeof processor.getMissingShaderSummary === 'function'
              ? processor.getMissingShaderSummary(missingOpts)
              : null,
        });
      } else if (
        payload.stats ||
        payload.type === 'gpu-stats' ||
        payload.type === 'request-gpu-stats'
      ) {
        const stats =
          processor && typeof processor.getStats === 'function' ? processor.getStats() : null;
        if (payload.type === 'gpu-stats' || payload.type === 'request-gpu-stats') {
          const reply = {
            type: 'gpu-stats',
            stats,
            framesPresented: stats ? toNonNegativeInt(stats.framesPresented, 0) : 0,
            presentCount: stats ? toNonNegativeInt(stats.presentCount, 0) : 0,
            drawCalls: stats ? toNonNegativeInt(stats.drawCalls, 0) : 0,
            realPackDrawCount: stats ? toNonNegativeInt(stats.realPackDrawCount, 0) : 0,
            fallbackDrawCount: stats ? toNonNegativeInt(stats.fallbackDrawCount, 0) : 0,
            validationErrorCount: stats ? toNonNegativeInt(stats.validationErrorCount, 0) : 0,
            rangedPackActive: Boolean(stats && stats.rangedPackActive === true),
            rangedPackThresholdBytes: stats
              ? toNonNegativeInt(
                  stats.rangedPackThresholdBytes,
                  DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES
                )
              : DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES,
            asyncShaderPendingCount: stats
              ? toNonNegativeInt(stats.asyncShaderPendingCount, 0)
              : 0,
            oversizedWgslFallbacks: stats
              ? toNonNegativeInt(stats.oversizedWgslFallbacks, 0)
              : 0,
            coalescedPresents: stats ? toNonNegativeInt(stats.coalescedPresents, 0) : 0,
            deviceLostCount: stats ? toNonNegativeInt(stats.deviceLostCount, 0) : 0,
            deviceRecoveredCount: stats ? toNonNegativeInt(stats.deviceRecoveredCount, 0) : 0,
            deviceStats: processor ? resolveDeviceLossTelemetry(processor, stats) : null,
            shaderPackStats: processor ? resolveShaderPackTelemetry(processor, stats) : null,
            swapchainProbeTimeoutCount: stats
              ? toNonNegativeInt(stats.swapchainProbeTimeoutCount, 0)
              : 0,
            gpuStallWarnCount: stats ? toNonNegativeInt(stats.gpuStallWarnCount, 0) : 0,
            missingShaderSummary:
              stats && stats.missingShaderSummary ? stats.missingShaderSummary : null,
            asyncPipelineStats:
              stats && stats.asyncPipelineStats ? stats.asyncPipelineStats : null,
            asyncPipelineCompileStats:
              stats && stats.asyncPipelineCompileStats ? stats.asyncPipelineCompileStats : null,
            fallbackSamplersCreated: stats
              ? toNonNegativeInt(stats.fallbackSamplersCreated, 0)
              : 0,
            fallbackSamplerBindingsSynthesized: stats
              ? toNonNegativeInt(stats.fallbackSamplerBindingsSynthesized, 0)
              : 0,
            firstVisibleFrame: Boolean(stats && stats.firstVisibleFrame === true),
            firstVisibleFrameMs:
              stats &&
              typeof stats.firstVisibleFrameMs === 'number' &&
              Number.isFinite(stats.firstVisibleFrameMs)
                ? stats.firstVisibleFrameMs
                : null,
            blackLoadingMs:
              stats &&
              typeof stats.blackLoadingMs === 'number' &&
              Number.isFinite(stats.blackLoadingMs)
                ? stats.blackLoadingMs
                : null,
            swapchainProbeCount: stats ? toNonNegativeInt(stats.swapchainProbeCount, 0) : 0,
            swapchainProbeInFlight: Boolean(stats && stats.swapchainProbeInFlight === true),
            swapchainProbeSupported:
              stats && typeof stats.swapchainProbeSupported === 'boolean'
                ? stats.swapchainProbeSupported
                : null,
            swapchainProbeErrorCount: stats
              ? toNonNegativeInt(stats.swapchainProbeErrorCount, 0)
              : 0,
            lastSampledMaxLuma:
              stats &&
              typeof stats.lastSampledMaxLuma === 'number' &&
              Number.isFinite(stats.lastSampledMaxLuma)
                ? Math.max(0, Math.min(255, Math.floor(stats.lastSampledMaxLuma)))
                : 0,
            suppressedBlackFrames: stats ? toNonNegativeInt(stats.suppressedBlackFrames, 0) : 0,
            gpuAdapter: (stats && stats.gpuAdapter) || getActiveGpuAdapter(),
            chunkedWriteBufferUploads: stats
              ? toNonNegativeInt(stats.chunkedWriteBufferUploads, 0)
              : 0,
            gpuMemory:
              stats && stats.gpuMemory && typeof stats.gpuMemory === 'object'
                ? { ...stats.gpuMemory }
                : null,
            liveBufferBytes: stats ? toNonNegativeInt(stats.liveBufferBytes, 0) : 0,
            liveTextureBytes: stats ? toNonNegativeInt(stats.liveTextureBytes, 0) : 0,
            peakBufferBytes: stats ? toNonNegativeInt(stats.peakBufferBytes, 0) : 0,
            peakTextureBytes: stats ? toNonNegativeInt(stats.peakTextureBytes, 0) : 0,
            stagingPoolBytes: stats ? toNonNegativeInt(stats.stagingPoolBytes, 0) : 0,
            pendingDeferredDestroys: stats ? toNonNegativeInt(stats.pendingDeferredDestroys, 0) : 0,
            pendingDeferredDestroyBytes: stats
              ? toNonNegativeInt(stats.pendingDeferredDestroyBytes, 0)
              : 0,
            deferredHighWaterFlushes: stats
              ? toNonNegativeInt(stats.deferredHighWaterFlushes, 0)
              : 0,
            gpuTimer: resolveGpuTimerTelemetry(processor, stats),
            gpuRecorder: resolveGpuRecorderTelemetry(processor, stats),
            ...resolve30FpsTelemetry(processor, stats),
          };
          if (payload.id !== undefined) {
            reply.id = payload.id;
          }
          postMessage(reply);
        } else {
          postMessage({
            id: payload.id,
            stats,
            gpuAdapter: (stats && stats.gpuAdapter) || getActiveGpuAdapter(),
          });
        }
      } else if (
        (payload.type === 'set-swapchain-tonemap' || payload.setSwapchainToneMapOptions) &&
        processor &&
        typeof processor.setSwapchainToneMapOptions === 'function'
      ) {
        const toneMapOpts =
          payload.setSwapchainToneMapOptions &&
          typeof payload.setSwapchainToneMapOptions === 'object'
            ? payload.setSwapchainToneMapOptions
            : payload;
        const result = processor.setSwapchainToneMapOptions(toneMapOpts);
        postMessage({
          id: payload.id,
          type: 'swapchain-tonemap-updated',
          ...result,
        });
      } else if ((payload.type === 'scene-transition' || payload.sceneTransition) && processor) {
        const reclaimed =
          typeof processor.notifySceneTransition === 'function'
            ? processor.notifySceneTransition(payload.maxIdleFrames)
            : typeof processor.trimTransientMemory === 'function'
              ? processor.trimTransientMemory(payload)
              : {
                  reclaimedCount: 0,
                  reclaimedBytes: 0,
                  stagingReclaimedSlabs: 0,
                  stagingReclaimedBytes: 0,
                  rtReclaimedCount: 0,
                  rtReclaimedBytes: 0,
                };
        if (payload.id !== undefined || payload.type === 'scene-transition') {
          const reply = {
            type: 'scene-transition',
            reclaimedCount: toNonNegativeInt(reclaimed && reclaimed.reclaimedCount, 0),
            reclaimedBytes: toNonNegativeInt(reclaimed && reclaimed.reclaimedBytes, 0),
            stagingReclaimedSlabs: toNonNegativeInt(
              reclaimed && reclaimed.stagingReclaimedSlabs,
              0
            ),
            stagingReclaimedBytes: toNonNegativeInt(
              reclaimed && reclaimed.stagingReclaimedBytes,
              0
            ),
            rtReclaimedCount: toNonNegativeInt(reclaimed && reclaimed.rtReclaimedCount, 0),
            rtReclaimedBytes: toNonNegativeInt(reclaimed && reclaimed.rtReclaimedBytes, 0),
          };
          if (payload.id !== undefined) {
            reply.id = payload.id;
          }
          postMessage(reply);
        }
      } else if (
        (payload.type === 'trim-transient-memory' ||
          payload.type === 'memory-pressure' ||
          payload.trimTransientMemory) &&
        processor
      ) {
        const reclaimed =
          typeof processor.trimTransientMemory === 'function'
            ? processor.trimTransientMemory(payload)
            : typeof processor.notifySceneTransition === 'function'
              ? processor.notifySceneTransition(payload.maxIdleFrames)
              : {
                  reclaimedCount: 0,
                  reclaimedBytes: 0,
                  stagingReclaimedSlabs: 0,
                  stagingReclaimedBytes: 0,
                  rtReclaimedCount: 0,
                  rtReclaimedBytes: 0,
                };
        const reply = {
          type: payload.type || 'trim-transient-memory',
          reclaimedCount: toNonNegativeInt(reclaimed && reclaimed.reclaimedCount, 0),
          reclaimedBytes: toNonNegativeInt(reclaimed && reclaimed.reclaimedBytes, 0),
          stagingReclaimedSlabs: toNonNegativeInt(reclaimed && reclaimed.stagingReclaimedSlabs, 0),
          stagingReclaimedBytes: toNonNegativeInt(reclaimed && reclaimed.stagingReclaimedBytes, 0),
          rtReclaimedCount: toNonNegativeInt(reclaimed && reclaimed.rtReclaimedCount, 0),
          rtReclaimedBytes: toNonNegativeInt(reclaimed && reclaimed.rtReclaimedBytes, 0),
        };
        if (payload.id !== undefined) {
          reply.id = payload.id;
        }
        postMessage(reply);
      } else if (payload.type === 'present-splash-frame' || payload.type === 'bink-frame') {
        const framePayload =
          payload.frame && typeof payload.frame === 'object' ? payload.frame : payload;
        const requestedFrameIndex =
          framePayload && Number.isInteger(framePayload.frameIndex)
            ? framePayload.frameIndex
            : Number.isInteger(payload.frameIndex)
              ? payload.frameIndex
              : null;

        if (
          !processor ||
          !processor.device ||
          !processor.ctx ||
          typeof processor.presentSplashFrame !== 'function'
        ) {
          const errMsg = !processor
            ? 'WebGPU processor/device is not initialized yet'
            : !processor.device
              ? 'WebGPU device is not ready yet'
              : !processor.ctx
                ? 'WebGPU canvas context is not ready yet'
                : 'WgpuCommandProcessor does not support presentSplashFrame';
          const errorReply = {
            type: 'splash-frame-presented',
            frameIndex: requestedFrameIndex,
            framesPresented: processor ? toNonNegativeInt(processor.frameCount, 0) : 0,
            ok: false,
            error: errMsg,
          };
          if (payload.id !== undefined) {
            errorReply.id = payload.id;
          }
          postMessage(errorReply);
          return RESOLVED_VOID_PROMISE;
        }

        if (processor.isDeviceRecovering === true) {
          // W100: drop splash/Bink frames cleanly while the lost device is being re-acquired.
          const recoveringReply = {
            type: 'splash-frame-presented',
            frameIndex: requestedFrameIndex,
            framesPresented: toNonNegativeInt(processor.frameCount, 0),
            ok: false,
            reason: 'device-recovering',
            error: 'WebGPU device recovery in progress',
          };
          if (payload.id !== undefined) {
            recoveringReply.id = payload.id;
          }
          postMessage(recoveringReply);
          return RESOLVED_VOID_PROMISE;
        }

        const prevFrames = toNonNegativeInt(processor.frameCount, 0);
        splashPresentOptsScratch.nowMs = payload.nowMs;
        splashPresentOptsScratch.respectPacing = payload.respectPacing;
        splashPresentOptsScratch.chromaOrder =
          framePayload.chromaOrder !== undefined ? framePayload.chromaOrder : payload.chromaOrder;
        splashPresentOptsScratch.loadOp =
          framePayload.loadOp !== undefined ? framePayload.loadOp : payload.loadOp;
        splashPresentOptsScratch.colorRange =
          framePayload.colorRange !== undefined ? framePayload.colorRange : payload.colorRange;
        splashPresentOptsScratch.targetWidth =
          payload.targetWidth !== undefined
            ? payload.targetWidth
            : canvas && Number.isInteger(canvas.width) && canvas.width > 0
              ? canvas.width
              : undefined;
        splashPresentOptsScratch.targetHeight =
          payload.targetHeight !== undefined
            ? payload.targetHeight
            : canvas && Number.isInteger(canvas.height) && canvas.height > 0
              ? canvas.height
              : undefined;

        const result = processor.presentSplashFrame(framePayload, splashPresentOptsScratch);
        const framesPresented = toNonNegativeInt(
          result && result.framesPresented !== undefined
            ? result.framesPresented
            : processor.frameCount,
          prevFrames
        );
        const reply = {
          type: 'splash-frame-presented',
          frameIndex:
            result && result.frameIndex !== undefined ? result.frameIndex : requestedFrameIndex,
          framesPresented,
          ok: Boolean(result && result.ok),
        };
        if (payload.id !== undefined) {
          reply.id = payload.id;
        }
        if (result && result.firstPresent) {
          reply.firstPresent = true;
        }
        if (result && !result.ok) {
          if (result.error) {
            reply.error = result.error;
          }
          if (result.reason) {
            reply.reason = result.reason;
          }
        }
        postMessage(reply);

        if (result && result.ok && result.presented) {
          if (result.firstPresent || (prevFrames === 0 && framesPresented >= 1)) {
            const firstPresentInfo = {
              type: 'first-present',
              firstPresent: true,
              source: result.source || 'splash',
              frameIndex: reply.frameIndex,
              framesPresented,
              width: result.width,
              height: result.height,
              timestamp: telemetryNow(),
              gpuAdapter: getActiveGpuAdapter(),
            };
            if (notifyHandlerFirstPresent(firstPresentInfo)) {
              postMessage(firstPresentInfo);
            }
          }
          maybeEmitFrameTelemetry(prevFrames, framesPresented);
        }
      } else if (payload.type === 'attach-guest-memory') {
        pendingGuestMemory = payload.buffer || null;
        if (processor && typeof processor.attachGuestMemory === 'function') {
          processor.attachGuestMemory(pendingGuestMemory);
        }
      } else if (payload.destroy && processor) {
        if (unbindValidationListener) {
          try {
            unbindValidationListener();
          } catch (_) {
            // Ignore cleanup errors.
          }
          unbindValidationListener = null;
        }
        if (unbindFirstVisibleFrameListener) {
          try {
            unbindFirstVisibleFrameListener();
          } catch (_) {
            // Ignore cleanup errors.
          }
          unbindFirstVisibleFrameListener = null;
        }
        // W100: intentional host destroy must never trigger device-loss recovery.
        bindProcessorDeviceLoss(null);
        if (typeof processor.destroy === 'function') {
          processor.destroy();
        }
        swapchain = null;
        postMessage({ id: payload.id, destroyed: true });
      } else if ((payload.drainRing || payload.type === 'drain-ring') && processor) {
        if (
          pendingGuestMemory &&
          processor._guestMemory !== pendingGuestMemory &&
          typeof processor.attachGuestMemory === 'function'
        ) {
          processor.attachGuestMemory(pendingGuestMemory);
        }
        const prevFrames = toNonNegativeInt(processor.frameCount, 0);
        const result = processor.drainRing(DRAIN_RING_REUSE_OPTS);
        if (result.presents > 0) {
          if (prevFrames === 0 && result.frameCount >= 1) {
            notifyHandlerFirstPresent({
              type: 'first-present',
              firstPresent: true,
              source: 'command-buffer',
              frameIndex: 0,
              framesPresented: result.frameCount,
              timestamp: telemetryNow(),
              gpuAdapter: getActiveGpuAdapter(),
            });
          }
          postMessage({ frame: result.frameCount });
          maybeEmitFrameTelemetry(prevFrames, result.frameCount);
        }
      } else if (
        payload.type === 'execute-batch-queue' &&
        Array.isArray(payload.batches) &&
        processor
      ) {
        for (let i = 0; i < payload.batches.length; i += 1) {
          pendingCommandBatches.push(payload.batches[i]);
        }
        flushPendingCommandBatches();
      } else if (
        initInFlight &&
        !processor &&
        extractRawBatchBuffer(payload) !== null
      ) {
        pendingCommandBatches.push(payload);
      } else if (
        !processor &&
        !initInFlight &&
        payload.recycleBuffer === true &&
        isPlainArrayBuffer(payload.cmdBuffer) &&
        payload.cmdBuffer.byteLength >= RECYCLE_CMD_BUFFER_MIN_BYTES &&
        payload.cmdBuffer.byteLength <= RECYCLE_CMD_BUFFER_MAX_BYTES
      ) {
        try {
          postMessage({ type: 'recycle-cmd-buffer', cmdBuffer: payload.cmdBuffer }, [
            payload.cmdBuffer,
          ]);
        } catch (_) {
          // Ignore transfer errors if the buffer was detached or the channel closed.
        }
      } else if (
        processor &&
        (payload.cmdBuffer ||
          ((payload.type === 'submit' || payload.type === 'executeBatch') &&
            (payload.buffer || payload.commands || payload.data)))
      ) {
        const rawBatchBuffer =
          payload.cmdBuffer || payload.buffer || payload.commands || payload.data;
        const isRawArrayBuffer = isPlainArrayBuffer(rawBatchBuffer);
        const isRawSharedBuffer =
          typeof SharedArrayBuffer !== 'undefined' &&
          rawBatchBuffer &&
          typeof rawBatchBuffer === 'object' &&
          (rawBatchBuffer instanceof SharedArrayBuffer ||
            Object.prototype.toString.call(rawBatchBuffer) === '[object SharedArrayBuffer]');
        const hasValidByteLength =
          typeof payload.byteLength === 'number' &&
          Number.isInteger(payload.byteLength) &&
          payload.byteLength >= 0 &&
          rawBatchBuffer &&
          typeof rawBatchBuffer.byteLength === 'number' &&
          payload.byteLength <= rawBatchBuffer.byteLength;
        const canPassDirectByteLength =
          hasValidByteLength &&
          (isRawArrayBuffer || isRawSharedBuffer) &&
          processor.executeCommandBuffer === WgpuCommandProcessor.prototype.executeCommandBuffer;
        const batchInput =
          hasValidByteLength && (isRawArrayBuffer || isRawSharedBuffer) && !canPassDirectByteLength
            ? new Uint8Array(rawBatchBuffer, 0, payload.byteLength)
            : rawBatchBuffer;
        const shouldRecycle =
          payload.recycleBuffer === true &&
          payload.cmdBuffer === rawBatchBuffer &&
          isRawArrayBuffer &&
          rawBatchBuffer.byteLength >= RECYCLE_CMD_BUFFER_MIN_BYTES &&
          rawBatchBuffer.byteLength <= RECYCLE_CMD_BUFFER_MAX_BYTES;
        if (
          (payload.frameLuminance !== undefined || payload.luminance !== undefined) &&
          typeof processor.setFrameLuminanceMeta === 'function'
        ) {
          processor.setFrameLuminanceMeta(
            payload.frameLuminance !== undefined ? payload.frameLuminance : payload.luminance
          );
        }
        if (
          pendingGuestMemory &&
          processor._guestMemory !== pendingGuestMemory &&
          typeof processor.attachGuestMemory === 'function'
        ) {
          processor.attachGuestMemory(pendingGuestMemory);
        }
        const hasExplicitCoalesce =
          typeof payload.coalescePresents === 'boolean' ||
          typeof payload.coalescePresent === 'boolean';
        const explicitCoalescePresents =
          typeof payload.coalescePresents === 'boolean'
            ? payload.coalescePresents
            : payload.coalescePresent === true;
        const prevFrames = toNonNegativeInt(processor.frameCount, 0);
        let result;
        try {
          if (canPassDirectByteLength) {
            if (hasExplicitCoalesce) {
              cmdBufferExecCoalesceOpts.byteLength = payload.byteLength;
              cmdBufferExecCoalesceOpts.coalescePresents = explicitCoalescePresents;
              result = processor.executeCommandBuffer(batchInput, cmdBufferExecCoalesceOpts);
            } else {
              cmdBufferExecOpts.byteLength = payload.byteLength;
              result = processor.executeCommandBuffer(batchInput, cmdBufferExecOpts);
            }
          } else if (hasExplicitCoalesce) {
            result = processor.executeCommandBuffer(
              batchInput,
              explicitCoalescePresents
                ? DRAIN_RING_REUSE_COALESCE_OPTS
                : DRAIN_RING_REUSE_NO_COALESCE_OPTS
            );
          } else {
            result = processor.executeCommandBuffer(batchInput, DRAIN_RING_REUSE_OPTS);
          }
        } finally {
          if (shouldRecycle) {
            if (processor._cachedCmdBuffer === rawBatchBuffer) {
              processor._cachedCmdBuffer = null;
              processor._cachedCmdView = null;
            }
            try {
              postMessage(
                { type: 'recycle-cmd-buffer', cmdBuffer: rawBatchBuffer },
                [rawBatchBuffer]
              );
            } catch (_) {
              // Ignore transfer errors if the buffer was detached or the channel closed.
            }
          } else if (
            isRawArrayBuffer &&
            rawBatchBuffer.byteLength > 4 * 1024 * 1024 &&
            processor._cachedCmdBuffer === rawBatchBuffer
          ) {
            processor._cachedCmdBuffer = null;
            processor._cachedCmdView = null;
          }
        }
        if (result.presents > 0) {
          if (prevFrames === 0 && result.frameCount >= 1) {
            notifyHandlerFirstPresent({
              type: 'first-present',
              firstPresent: true,
              source: 'command-buffer',
              frameIndex: 0,
              framesPresented: result.frameCount,
              timestamp: telemetryNow(),
              gpuAdapter: getActiveGpuAdapter(),
            });
          }
          postMessage({ frame: result.frameCount });
          maybeEmitFrameTelemetry(prevFrames, result.frameCount);
        }
      }
    } catch (err) {
      postMessage({ error: String(err) });
    }
    return RESOLVED_VOID_PROMISE;
  }

  return {
    handleMessage,
    enqueueCommandBatch(batchPayload) {
      if (extractRawBatchBuffer(batchPayload) !== null || batchPayload instanceof ArrayBuffer || ArrayBuffer.isView(batchPayload)) {
        pendingCommandBatches.push(batchPayload);
      }
      return pendingCommandBatches.length;
    },
    flushCommandQueue: flushPendingCommandBatches,
    get pendingCommandBatchCount() {
      return pendingCommandBatches.length;
    },
    get processor() {
      return processor;
    },
    get swapchain() {
      return (processor && processor.swapchain) || swapchain;
    },
    get ring() {
      return attachedRing;
    },
  };
}

const exportedApi = {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  buildGpuTelemetrySnapshot,
  compactWgslForMetadata,
  DEFAULT_MAX_WGSL_COMPILE_BYTES,
  DEFAULT_MAX_CONCURRENT_SHADER_COMPILES,
  DEFAULT_SHADER_COMPILE_SLICE_BUDGET_MS,
  DEFAULT_MAX_PENDING_RANGED_PIPELINES,
  buildMergedBinkPresenterOptions,
  configureCanvasContextWithCopySrc,
  SWAPCHAIN_PROBE_DEFAULT_INTERVAL_MS,
  SWAPCHAIN_PROBE_DEFAULT_LUMA_THRESHOLD,
  SWAPCHAIN_PROBE_MAX_CONSECUTIVE_ERRORS,
  SWAPCHAIN_PROBE_STRIPS,
  SWAPCHAIN_PROBE_BYTES_PER_ROW,
  SWAPCHAIN_PROBE_BUFFER_SIZE,
  selectSupportedAdapterFeatures,
  OPTIONAL_ADAPTER_FEATURES,
  applyRtBvhPreludeIfNeeded,
  ensureRtBvhPreludeLoaded,
  INLINE_RT_BVH_PRELUDE_VERSION,
  selectSupportedAdapterLimits,
  OPTIONAL_ADAPTER_LIMIT_SPECS,
  extractSanitizedGpuAdapter,
  sanitizeGpuAdapterInfo,
  MAX_GPU_ADAPTER_FIELD_LENGTH,
  MAX_DIRECT_WRITE_BUFFER_CHUNK_BYTES,
  RECYCLE_CMD_BUFFER_MIN_BYTES,
  RECYCLE_CMD_BUFFER_MAX_BYTES,
  selectFallbackShaderProfile,
  sanitizeValidationErrorMessage,
  normalizeValidationErrorType,
  FALLBACK_PROFILE,
  DEFAULT_MAX_VALIDATION_ERROR_HISTORY,
  DEFAULT_MAX_VALIDATION_ERROR_POSTS,
  MAX_VALIDATION_ERROR_MESSAGE_LENGTH,
  DEFAULT_TELEMETRY_INTERVAL_MS,
  DEFAULT_TELEMETRY_MISSING_LIMIT,
  DEFAULT_MISSING_SUMMARY_LIMIT,
  DEFAULT_MAX_CONCURRENT_PIPELINE_COMPILES,
  sanitizeMissingShaderSummary,
  DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES,
  DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES,
  REPLAY_LOG_MAX_ENTRIES,
  REPLAY_LOG_MAX_BYTES,
  REPLAY_LOG_MAX_BIND_GROUPS,
  SWAPCHAIN_PROBE_DEFAULT_TIMEOUT_MS,
  DEFAULT_GPU_STALL_WARN_MS,
  resolveShaderPackStreamThresholdBytes,
  InlineWgslPackCache,
  parseHashOnlyShaderPayload,
  resolveGpuRingBuffer,
  estimateTextureBytes,
  planClearTextureCopies,
  decodeReadbackPayload,
  readbackStagingLayout,
  readbackPieces,
  deliverReadback,
  fnv1a64Hex,
  OPCODES,
  OPCODE_CLEAR_TEXTURE,
  CLEAR_TEXTURE_PAYLOAD_BYTES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  ADDRESS_MODE,
  FILTER_MODE,
  BIND_RESOURCE_TYPE,
  INDEX_FORMAT,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Wgpu = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  ensureRtBvhPreludeLoaded();
  const workerHandler = createWgpuWorkerHandler({
    postMessage: (data, transfer) =>
      transfer && transfer.length > 0
        ? self.postMessage(data, transfer)
        : self.postMessage(data),
  });
  self.onmessage = (ev) => workerHandler.handleMessage(ev ? ev.data : null);
}
