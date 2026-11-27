'use strict';

// WebGPU resource handle registry, frame-deferred garbage collector,
// sRGB/linear sibling viewFormats & cached GPUTextureView manager with
// generation-based stale-view invalidation, and Frame-1 hardened resource
// creation/upload helpers (CreateTexture2 ABI v2 opcode 35, BC1..BC7 fallback
// & 4x4 alignment, r8unorm Bink2 video textures, mip/sampleCount clamping,
// and 4-byte WebGPU buffer size/write alignment).
// Zero external dependencies; UMD/CommonJS/ESM compatible for Web Workers
// (importScripts -> globalThis.PlayGta6ResourceRegistry), browsers, and
// Node.js (require() / import()).

(function (root) {
  const DEFAULT_FRAMES_IN_FLIGHT = 3;
  const DEFAULT_SOFT_BUDGET_BYTES = 512 * 1024 * 1024; // 512 MiB
  const DEFAULT_VRAM_RECLAIM_TRIGGER_RATIO = 0.85;
  const DEFAULT_VRAM_RECLAIM_TARGET_RATIO = 0.72;
  const DEFAULT_OLDEST_LIVE_LIMIT = 16;
  const DEFAULT_MAX_CACHED_VIEWS_PER_TEXTURE = 16;
  const FAST_HANDLE_LIMIT = 65536;
  const DEFAULT_MAX_STALE_HANDLE_HISTORY = 16384;

  const VRAM_PRESSURE_TIER = Object.freeze({
    NORMAL: 'normal',
    SOFT: 'soft',
    HARD: 'hard',
  });

  const FALLBACK_REGISTRY_PERF_PROFILES = Object.freeze({
    low: Object.freeze({
      name: 'low',
      vramBudgetBytes: 1536 * 1024 * 1024,
      vramReclaimTriggerRatio: 0.72,
      vramReclaimTargetRatio: 0.58,
      gcDeferFrames: 2,
    }),
    balanced: Object.freeze({
      name: 'balanced',
      vramBudgetBytes: 3072 * 1024 * 1024,
      vramReclaimTriggerRatio: 0.78,
      vramReclaimTargetRatio: 0.65,
      gcDeferFrames: 2,
    }),
    high: Object.freeze({
      name: 'high',
      vramBudgetBytes: 6144 * 1024 * 1024,
      vramReclaimTriggerRatio: 0.85,
      vramReclaimTargetRatio: 0.72,
      gcDeferFrames: 3,
    }),
  });

  function _resolvePerfProfileModule() {
    if (typeof globalThis !== 'undefined' && globalThis.PlayGta6PerfProfile) {
      return globalThis.PlayGta6PerfProfile;
    }
    if (typeof require === 'function') {
      try {
        return require('./perf_profile.js');
      } catch {
        return null;
      }
    }
    return null;
  }

  function _coercePerfProfile(profileOrName) {
    const mod = _resolvePerfProfileModule();
    if (mod && typeof mod.coercePerfProfile === 'function') {
      return mod.coercePerfProfile(profileOrName);
    }
    if (typeof profileOrName === 'string') {
      const key = profileOrName.trim().toLowerCase();
      if (key === 'auto') {
        return FALLBACK_REGISTRY_PERF_PROFILES.balanced;
      }
      if (Object.prototype.hasOwnProperty.call(FALLBACK_REGISTRY_PERF_PROFILES, key)) {
        return FALLBACK_REGISTRY_PERF_PROFILES[key];
      }
      throw new RangeError(`Unknown performance profile "${profileOrName}"`);
    }
    if (profileOrName && typeof profileOrName === 'object') {
      const name =
        typeof profileOrName.name === 'string'
          ? profileOrName.name.trim().toLowerCase()
          : typeof profileOrName.preset === 'string'
            ? profileOrName.preset.trim().toLowerCase()
            : 'balanced';
      const base = Object.prototype.hasOwnProperty.call(FALLBACK_REGISTRY_PERF_PROFILES, name)
        ? FALLBACK_REGISTRY_PERF_PROFILES[name]
        : FALLBACK_REGISTRY_PERF_PROFILES.balanced;
      return Object.freeze({
        name: base.name,
        vramBudgetBytes:
          profileOrName.vramBudgetBytes !== undefined
            ? profileOrName.vramBudgetBytes
            : profileOrName.softBudgetBytes !== undefined
              ? profileOrName.softBudgetBytes
              : base.vramBudgetBytes,
        vramReclaimTriggerRatio:
          profileOrName.vramReclaimTriggerRatio !== undefined
            ? profileOrName.vramReclaimTriggerRatio
            : base.vramReclaimTriggerRatio,
        vramReclaimTargetRatio:
          profileOrName.vramReclaimTargetRatio !== undefined
            ? profileOrName.vramReclaimTargetRatio
            : base.vramReclaimTargetRatio,
        gcDeferFrames:
          profileOrName.gcDeferFrames !== undefined
            ? profileOrName.gcDeferFrames
            : profileOrName.framesInFlight !== undefined
              ? profileOrName.framesInFlight
              : base.gcDeferFrames,
      });
    }
    throw new TypeError('perfProfile must be a preset name or profile object');
  }

  function assertUnitRatio(value, name) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0 || value > 1) {
      throw new RangeError(`${name} must be a finite number in (0, 1], got ${String(value)}`);
    }
  }

  const CREATE_TEXTURE2_OPCODE = 35;
  const CREATE_TEXTURE2_PAYLOAD_BYTES = 40;
  const MAX_TEXTURE_VIEW_FORMATS = 8;

  const TEXTURE_COMPRESSION_BC_FEATURE = 'texture-compression-bc';
  const BC_FALLBACK_FORMAT = 'rgba8unorm';
  const BC_FALLBACK_SRGB_FORMAT = 'rgba8unorm-srgb';

  const BUFFER_ALIGNMENT_BYTES = 4;
  const UNIFORM_BUFFER_MIN_BYTES = 16;

  // WebGPU-compatible GPUBufferUsage bitmask values (runtime/abi/playgta6_abi.h).
  const GPU_BUFFER_USAGE = Object.freeze({
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

  // WebGPU-compatible GPUTextureUsage bitmask values (runtime/abi/playgta6_abi.h).
  const GPU_TEXTURE_USAGE = Object.freeze({
    COPY_SRC: 0x0001,
    COPY_DST: 0x0002,
    TEXTURE_BINDING: 0x0004,
    STORAGE_BINDING: 0x0008,
    RENDER_ATTACHMENT: 0x0010,
  });

  // v1 GpuTextureFormat enum names (runtime/abi/playgta6_abi.h).
  const V1_TEXTURE_FORMAT_NAMES = Object.freeze([
    'rgba8unorm',
    'bgra8unorm',
    'rgba16float',
    'depth24plus',
    'depth32float',
    'bc7-rgba-unorm',
  ]);

  // v2 texture format codes ("WebGPU enum codes" in docs/gpu_commands.md & texture_packets.js).
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

  // GpuTextureDimension enum names (runtime/abi/playgta6_abi.h).
  const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);

  // Valid WebGPU GPUTextureViewDimension and GPUTextureAspect values.
  const TEXTURE_VIEW_DIMENSIONS = Object.freeze([
    '1d',
    '2d',
    '2d-array',
    'cube',
    'cube-array',
    '3d',
  ]);
  const TEXTURE_VIEW_DIMENSION_SET = new Set(TEXTURE_VIEW_DIMENSIONS);

  const TEXTURE_VIEW_ASPECTS = Object.freeze(['all', 'stencil-only', 'depth-only']);
  const TEXTURE_VIEW_ASPECT_SET = new Set(TEXTURE_VIEW_ASPECTS);

  // WebGPU sRGB <-> linear view-compatible sibling format pairs.
  // Note: single/dual-channel formats such as r8unorm (Bink2 Y/U/V/A planes)
  // and rg8unorm (NV12 UV plane) have no sRGB sibling in WebGPU.
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

  // Formats commonly used by Bink2 planar/semi-planar/RGBA video textures.
  const BINK_VIDEO_TEXTURE_FORMATS = Object.freeze(['r8unorm', 'rg8unorm', 'rgba8unorm']);

  // Depth/stencil aspect-specific view format mapping per WebGPU spec.
  const DEPTH_STENCIL_ASPECT_FORMATS = Object.freeze({
    'depth24plus-stencil8': Object.freeze({
      'depth-only': 'depth24plus',
      'stencil-only': 'stencil8',
    }),
    'depth32float-stencil8': Object.freeze({
      'depth-only': 'depth32float',
      'stencil-only': 'stencil8',
    }),
    depth16unorm: Object.freeze({
      'depth-only': 'depth16unorm',
    }),
    depth24plus: Object.freeze({
      'depth-only': 'depth24plus',
    }),
    depth32float: Object.freeze({
      'depth-only': 'depth32float',
    }),
    stencil8: Object.freeze({
      'stencil-only': 'stencil8',
    }),
  });

  // All WebGPU BC1..BC7 4x4 block-compressed texture formats.
  const BC_TEXTURE_FORMATS = Object.freeze([
    'bc1-rgba-unorm',
    'bc1-rgba-unorm-srgb',
    'bc2-rgba-unorm',
    'bc2-rgba-unorm-srgb',
    'bc3-rgba-unorm',
    'bc3-rgba-unorm-srgb',
    'bc4-r-unorm',
    'bc4-r-snorm',
    'bc5-rg-unorm',
    'bc5-rg-snorm',
    'bc6h-rgb-ufloat',
    'bc6h-rgb-float',
    'bc7-rgba-unorm',
    'bc7-rgba-unorm-srgb',
  ]);

  const BC_FORMAT_SET = new Set(BC_TEXTURE_FORMATS);

  const BC_8BYTE_BLOCK_FORMATS = new Set([
    'bc1-rgba-unorm',
    'bc1-rgba-unorm-srgb',
    'bc4-r-unorm',
    'bc4-r-snorm',
  ]);

  // Core WebGPU formats that support multisampling (sampleCount === 4).
  const MULTISAMPLE_SUPPORTED_FORMATS = new Set([
    'r8unorm',
    'r8uint',
    'r8sint',
    'rg8unorm',
    'rg8uint',
    'rg8sint',
    'rgba8unorm',
    'rgba8unorm-srgb',
    'rgba8uint',
    'rgba8sint',
    'bgra8unorm',
    'bgra8unorm-srgb',
    'r16uint',
    'r16sint',
    'r16float',
    'rg16uint',
    'rg16sint',
    'rg16float',
    'rgba16uint',
    'rgba16sint',
    'rgba16float',
    'rgb10a2uint',
    'rgb10a2unorm',
    'rg11b10ufloat',
    'r32float',
    'stencil8',
    'depth16unorm',
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
  ]);

  const UNCOMPRESSED_FORMAT_BPP = Object.freeze({
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

  const RESOURCE_KIND = Object.freeze({
    BUFFER: 'buffer',
    TEXTURE: 'texture',
    TEXTURE_VIEW: 'textureView',
    SAMPLER: 'sampler',
    BIND_GROUP: 'bindGroup',
    BIND_GROUP_LAYOUT: 'bindGroupLayout',
    PIPELINE_LAYOUT: 'pipelineLayout',
    SHADER_MODULE: 'shaderModule',
    RENDER_PIPELINE: 'renderPipeline',
    COMPUTE_PIPELINE: 'computePipeline',
  });

  const RESOURCE_KINDS = Object.freeze(Object.values(RESOURCE_KIND));
  const VALID_KIND_SET = new Set(RESOURCE_KINDS);

  // Per-category VRAM accounting buckets (W300, #1077/#1207).
  const VRAM_KIND = Object.freeze({
    BUFFER: 'buffer',
    TEXTURE: 'texture',
    RENDER_TARGET: 'renderTarget',
    SAMPLER: 'sampler',
    OTHER: 'other',
  });

  const VRAM_KINDS = Object.freeze([
    VRAM_KIND.BUFFER,
    VRAM_KIND.TEXTURE,
    VRAM_KIND.RENDER_TARGET,
    VRAM_KIND.SAMPLER,
    VRAM_KIND.OTHER,
  ]);
  const VALID_VRAM_KIND_SET = new Set(VRAM_KINDS);

  /**
   * Resolve the VRAM accounting bucket (`buffer`, `texture`, `renderTarget`,
   * `sampler`, or `other`) for a resource registration.
   *
   * @param {string} kind
   * @param {object | null | undefined} resource
   * @param {object | null | undefined} [meta]
   * @returns {'buffer' | 'texture' | 'renderTarget' | 'sampler' | 'other'}
   */
  function resolveVramKind(kind, resource, meta) {
    if (meta && typeof meta === 'object') {
      if (typeof meta.vramKind === 'string') {
        return VALID_VRAM_KIND_SET.has(meta.vramKind) ? meta.vramKind : VRAM_KIND.OTHER;
      }
      if (meta.isRenderTarget === true || meta.renderTarget === true) {
        return VRAM_KIND.RENDER_TARGET;
      }
    }

    if (kind === RESOURCE_KIND.BUFFER) {
      return VRAM_KIND.BUFFER;
    }
    if (kind === RESOURCE_KIND.SAMPLER) {
      return VRAM_KIND.SAMPLER;
    }
    if (kind === RESOURCE_KIND.TEXTURE) {
      const desc =
        meta && meta.descriptor && typeof meta.descriptor === 'object' ? meta.descriptor : null;
      if (desc && (desc.isRenderTarget === true || desc.renderTarget === true)) {
        return VRAM_KIND.RENDER_TARGET;
      }
      let usage = 0;
      if (meta && meta.usage !== undefined) {
        usage = Number(meta.usage) >>> 0;
      } else if (meta && meta.usageFlags !== undefined) {
        usage = Number(meta.usageFlags) >>> 0;
      } else if (meta && meta.usage_flags !== undefined) {
        usage = Number(meta.usage_flags) >>> 0;
      } else if (desc && desc.usage !== undefined) {
        usage = Number(desc.usage) >>> 0;
      } else if (desc && desc.usageFlags !== undefined) {
        usage = Number(desc.usageFlags) >>> 0;
      } else if (resource && typeof resource === 'object' && resource.usage !== undefined) {
        usage = Number(resource.usage) >>> 0;
      }
      if ((usage & GPU_TEXTURE_USAGE.RENDER_ATTACHMENT) !== 0) {
        return VRAM_KIND.RENDER_TARGET;
      }
      return VRAM_KIND.TEXTURE;
    }
    return VRAM_KIND.OTHER;
  }

  function assertValidKind(kind) {
    if (typeof kind !== 'string' || !VALID_KIND_SET.has(kind)) {
      throw new RangeError(
        `Invalid WebGPU resource kind: ${JSON.stringify(kind)} (expected one of ${RESOURCE_KINDS.join(', ')})`
      );
    }
  }

  function assertValidHandle(handle) {
    if (typeof handle !== 'number' || !Number.isSafeInteger(handle) || handle < 0) {
      throw new RangeError(
        `Invalid resource handle: ${String(handle)} (expected a non-negative safe integer)`
      );
    }
  }

  function assertPositiveInteger(value, name) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive safe integer, got ${String(value)}`);
    }
  }

  function assertNonNegativeInteger(value, name) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative safe integer, got ${String(value)}`);
    }
  }

  function sanitizeNonNegativeCounter(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return Math.floor(value);
  }

  function clampUnitRate(numerator, denominator) {
    if (
      typeof numerator !== 'number' ||
      typeof denominator !== 'number' ||
      !Number.isFinite(numerator) ||
      !Number.isFinite(denominator) ||
      denominator <= 0 ||
      numerator <= 0
    ) {
      return 0;
    }
    const rate = numerator / denominator;
    if (!Number.isFinite(rate) || rate <= 0) {
      return 0;
    }
    return rate >= 1 ? 1 : rate;
  }

  function snapshotEntry(entry) {
    return {
      kind: entry.kind,
      handle: entry.handle,
      resource: entry.resource,
      byteSize: entry.byteSize,
      label: entry.label,
      tag: entry.tag,
      registeredAtFrame: entry.registeredAtFrame,
    };
  }

  function isBcFormat(format) {
    return typeof format === 'string' && BC_FORMAT_SET.has(format.toLowerCase());
  }

  /**
   * Return whether a WebGPU texture format name is an `-srgb` format.
   *
   * @param {string} format
   * @returns {boolean}
   */
  function isSrgbFormat(format) {
    return typeof format === 'string' && format.trim().toLowerCase().endsWith('-srgb');
  }

  /**
   * Return the sRGB <-> linear sibling WebGPU format for `format`, or `null`
   * when the format has no sRGB sibling (e.g. `r8unorm` Bink2 video planes,
   * `rg8unorm`, `rgba16float`, depth/stencil formats).
   *
   * @param {string} format
   * @returns {string | null}
   */
  function getSrgbSiblingFormat(format) {
    if (typeof format !== 'string') {
      return null;
    }
    const normalized = format.trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(SRGB_SIBLING_FORMATS, normalized)) {
      return SRGB_SIBLING_FORMATS[normalized];
    }
    return null;
  }

  /**
   * Check whether `viewFormat` is WebGPU view-compatible with `textureFormat`
   * for the given `aspect` (`'all' | 'depth-only' | 'stencil-only'`).
   *
   * @param {string} textureFormat
   * @param {string} viewFormat
   * @param {string} [aspect='all']
   * @returns {boolean}
   */
  function isViewFormatCompatible(textureFormat, viewFormat, aspect = 'all') {
    if (typeof textureFormat !== 'string' || typeof viewFormat !== 'string') {
      return false;
    }
    const texFmt = textureFormat.trim().toLowerCase();
    const vf = viewFormat.trim().toLowerCase();
    if (!texFmt || !vf) {
      return false;
    }
    const normalizedAspect = typeof aspect === 'string' ? aspect.trim().toLowerCase() : 'all';
    if (normalizedAspect === 'depth-only' || normalizedAspect === 'stencil-only') {
      const aspectMap = Object.prototype.hasOwnProperty.call(DEPTH_STENCIL_ASPECT_FORMATS, texFmt)
        ? DEPTH_STENCIL_ASPECT_FORMATS[texFmt]
        : null;
      if (!aspectMap || !Object.prototype.hasOwnProperty.call(aspectMap, normalizedAspect)) {
        return false;
      }
      return vf === aspectMap[normalizedAspect];
    }
    return vf === texFmt || getSrgbSiblingFormat(texFmt) === vf;
  }

  /**
   * Resolve, deduplicate, and filter `viewFormats` for a texture's effective
   * WebGPU format, optionally appending its sRGB/linear sibling when one exists.
   * Incompatible view formats (such as an sRGB view format on an `r8unorm`
   * Bink2 video texture) are stripped and reported in `droppedViewFormats`.
   *
   * @param {string} effectiveFormat
   * @param {Array<string | number> | null | undefined} requestedViewFormats
   * @param {{
   *   bcFormatFallback?: boolean,
   *   includeSrgbSiblingViewFormat?: boolean,
   *   includeSrgbSibling?: boolean,
   * }} [options]
   * @returns {{
   *   viewFormats: string[],
   *   siblingViewFormat: string | null,
   *   droppedViewFormats: string[],
   * }}
   */
  function resolveTextureViewFormats(effectiveFormat, requestedViewFormats, options = {}) {
    const texFmt = resolveTextureFormat(effectiveFormat);
    const siblingViewFormat = getSrgbSiblingFormat(texFmt);
    const bcFormatFallback = Boolean(options && options.bcFormatFallback);
    const includeSibling = Boolean(
      options && (options.includeSrgbSiblingViewFormat || options.includeSrgbSibling)
    );

    const viewFormats = [];
    const droppedViewFormats = [];
    const seen = new Set();

    if (Array.isArray(requestedViewFormats) && requestedViewFormats.length > 0) {
      for (let i = 0; i < requestedViewFormats.length; i += 1) {
        const rawVf = requestedViewFormats[i];
        let vf =
          typeof rawVf === 'string'
            ? rawVf.trim().toLowerCase()
            : typeof rawVf === 'number' && Number.isInteger(rawVf) && rawVf > 0
              ? TEXTURE_FORMAT_NAMES[rawVf]
              : undefined;
        if (!vf || vf === 'undefined') {
          continue;
        }
        if (bcFormatFallback && isBcFormat(vf)) {
          vf = vf.endsWith('-srgb') ? BC_FALLBACK_SRGB_FORMAT : BC_FALLBACK_FORMAT;
        }
        if (!isViewFormatCompatible(texFmt, vf, 'all')) {
          droppedViewFormats.push(vf);
          continue;
        }
        if (!seen.has(vf) && viewFormats.length < MAX_TEXTURE_VIEW_FORMATS) {
          seen.add(vf);
          viewFormats.push(vf);
        }
      }
    }

    if (
      includeSibling &&
      siblingViewFormat &&
      !seen.has(siblingViewFormat) &&
      viewFormats.length < MAX_TEXTURE_VIEW_FORMATS
    ) {
      seen.add(siblingViewFormat);
      viewFormats.push(siblingViewFormat);
    }

    return {
      viewFormats,
      siblingViewFormat,
      droppedViewFormats,
    };
  }

  /**
   * Build the canonical 8-tuple cache key for a cached `GPUTextureView`:
   * `(textureId, format, dimension, baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount, aspect)`.
   *
   * @param {number} textureId
   * @param {{
   *   format: string,
   *   dimension: string,
   *   baseMipLevel: number,
   *   mipLevelCount: number | string,
   *   baseArrayLayer: number,
   *   arrayLayerCount: number | string,
   *   aspect: string,
   * }} normalized
   * @returns {string}
   */
  function buildTextureViewCacheKey(textureId, normalized) {
    return `${textureId}|${normalized.format}|${normalized.dimension}|${normalized.baseMipLevel}|${normalized.mipLevelCount}|${normalized.baseArrayLayer}|${normalized.arrayLayerCount}|${normalized.aspect}`;
  }

  /**
   * Normalize a `GPUTextureViewDescriptor` (with optional texture metadata) into
   * the canonical 8-tuple fields, cache key, and WebGPU `createView()` descriptor.
   *
   * @param {number} textureId
   * @param {object | null | undefined} rawViewDesc
   * @param {object | null | undefined} [textureMeta]
   * @returns {{
   *   textureId: number,
   *   format: string,
   *   dimension: string,
   *   baseMipLevel: number,
   *   mipLevelCount: number | string,
   *   baseArrayLayer: number,
   *   arrayLayerCount: number | string,
   *   aspect: string,
   *   key: string,
   *   gpuViewDescriptor: object,
   * }}
   */
  function normalizeTextureViewDescriptor(textureId, rawViewDesc, textureMeta = null) {
    assertValidHandle(textureId);
    if (rawViewDesc !== undefined && rawViewDesc !== null && typeof rawViewDesc !== 'object') {
      throw new TypeError('texture view descriptor must be an object, null, or undefined');
    }
    const desc = rawViewDesc || {};
    const meta = textureMeta && typeof textureMeta === 'object' ? textureMeta : null;

    // 1. Aspect
    let aspect = 'all';
    if (desc.aspect !== undefined && desc.aspect !== null) {
      if (typeof desc.aspect !== 'string') {
        throw new RangeError(`Invalid texture view aspect: ${String(desc.aspect)}`);
      }
      const lowerAspect = desc.aspect.trim().toLowerCase();
      if (!TEXTURE_VIEW_ASPECT_SET.has(lowerAspect)) {
        throw new RangeError(
          `Invalid texture view aspect: ${JSON.stringify(desc.aspect)} (expected one of ${TEXTURE_VIEW_ASPECTS.join(', ')})`
        );
      }
      aspect = lowerAspect;
    }

    // 2. Format
    let format = '';
    if (desc.format !== undefined && desc.format !== null && desc.format !== 'undefined') {
      if (typeof desc.format === 'string' && desc.format.trim().length > 0) {
        format = desc.format.trim().toLowerCase();
      } else if (
        typeof desc.format === 'number' &&
        Number.isInteger(desc.format) &&
        desc.format > 0 &&
        TEXTURE_FORMAT_NAMES[desc.format]
      ) {
        format = TEXTURE_FORMAT_NAMES[desc.format];
      } else {
        throw new RangeError(`Invalid texture view format: ${String(desc.format)}`);
      }
      if (meta && meta.bcFormatFallback && isBcFormat(format)) {
        format = format.endsWith('-srgb') ? BC_FALLBACK_SRGB_FORMAT : BC_FALLBACK_FORMAT;
      }
    } else if (meta && typeof meta.format === 'string' && meta.format.length > 0) {
      if (aspect === 'depth-only' || aspect === 'stencil-only') {
        const aspectMap = Object.prototype.hasOwnProperty.call(
          DEPTH_STENCIL_ASPECT_FORMATS,
          meta.format
        )
          ? DEPTH_STENCIL_ASPECT_FORMATS[meta.format]
          : null;
        if (!aspectMap || !Object.prototype.hasOwnProperty.call(aspectMap, aspect)) {
          throw new RangeError(
            `Texture ${textureId} (${meta.format}) does not support aspect "${aspect}"`
          );
        }
        format = aspectMap[aspect];
      } else {
        format = meta.format;
      }
    }

    // 3. Mip levels
    let baseMipLevel = 0;
    if (desc.baseMipLevel !== undefined && desc.baseMipLevel !== null) {
      assertNonNegativeInteger(desc.baseMipLevel, 'baseMipLevel');
      baseMipLevel = desc.baseMipLevel;
    }

    let mipLevelCount;
    if (desc.mipLevelCount !== undefined && desc.mipLevelCount !== null) {
      assertPositiveInteger(desc.mipLevelCount, 'mipLevelCount');
      mipLevelCount = desc.mipLevelCount;
    } else if (meta && typeof meta.mipLevelCount === 'number' && meta.mipLevelCount >= 1) {
      mipLevelCount = Math.max(1, meta.mipLevelCount - baseMipLevel);
    } else {
      mipLevelCount = 'all';
    }

    if (meta && typeof meta.mipLevelCount === 'number' && meta.mipLevelCount >= 1) {
      if (baseMipLevel >= meta.mipLevelCount) {
        throw new RangeError(
          `baseMipLevel (${baseMipLevel}) out of bounds for texture ${textureId} (mipLevelCount=${meta.mipLevelCount})`
        );
      }
      if (typeof mipLevelCount === 'number' && baseMipLevel + mipLevelCount > meta.mipLevelCount) {
        throw new RangeError(
          `baseMipLevel (${baseMipLevel}) + mipLevelCount (${mipLevelCount}) exceeds texture ${textureId} mipLevelCount (${meta.mipLevelCount})`
        );
      }
    }

    // 4. Dimension & array layers
    const texDimension = meta && meta.dimension ? meta.dimension : '2d';
    const maxArrayLayers =
      meta && typeof meta.depthOrArrayLayers === 'number' && meta.depthOrArrayLayers >= 1
        ? texDimension === '2d'
          ? meta.depthOrArrayLayers
          : 1
        : null;

    let baseArrayLayer = 0;
    if (desc.baseArrayLayer !== undefined && desc.baseArrayLayer !== null) {
      assertNonNegativeInteger(desc.baseArrayLayer, 'baseArrayLayer');
      baseArrayLayer = desc.baseArrayLayer;
    }

    let explicitArrayLayerCount = null;
    if (desc.arrayLayerCount !== undefined && desc.arrayLayerCount !== null) {
      assertPositiveInteger(desc.arrayLayerCount, 'arrayLayerCount');
      explicitArrayLayerCount = desc.arrayLayerCount;
    }

    let dimension;
    if (desc.dimension !== undefined && desc.dimension !== null) {
      if (typeof desc.dimension !== 'string') {
        throw new RangeError(`Invalid texture view dimension: ${String(desc.dimension)}`);
      }
      const lowerDim = desc.dimension.trim().toLowerCase();
      if (!TEXTURE_VIEW_DIMENSION_SET.has(lowerDim)) {
        throw new RangeError(
          `Invalid texture view dimension: ${JSON.stringify(desc.dimension)} (expected one of ${TEXTURE_VIEW_DIMENSIONS.join(', ')})`
        );
      }
      dimension = lowerDim;
    } else if (texDimension === '1d') {
      dimension = '1d';
    } else if (texDimension === '3d') {
      dimension = '3d';
    } else if (
      (maxArrayLayers !== null && maxArrayLayers > 1 && explicitArrayLayerCount !== 1) ||
      (explicitArrayLayerCount !== null && explicitArrayLayerCount > 1)
    ) {
      dimension = '2d-array';
    } else {
      dimension = '2d';
    }

    let arrayLayerCount;
    if (explicitArrayLayerCount !== null) {
      arrayLayerCount = explicitArrayLayerCount;
    } else if (dimension === 'cube') {
      arrayLayerCount = 6;
    } else if (dimension === '1d' || dimension === '2d' || dimension === '3d') {
      arrayLayerCount = 1;
    } else if (maxArrayLayers !== null) {
      arrayLayerCount = Math.max(1, maxArrayLayers - baseArrayLayer);
    } else {
      arrayLayerCount = 'all';
    }

    if (dimension === 'cube' && typeof arrayLayerCount === 'number' && arrayLayerCount !== 6) {
      throw new RangeError(`cube texture view requires arrayLayerCount === 6, got ${arrayLayerCount}`);
    }
    if (
      dimension === 'cube-array' &&
      typeof arrayLayerCount === 'number' &&
      arrayLayerCount % 6 !== 0
    ) {
      throw new RangeError(
        `cube-array texture view requires arrayLayerCount to be a multiple of 6, got ${arrayLayerCount}`
      );
    }
    if (maxArrayLayers !== null) {
      if (baseArrayLayer >= maxArrayLayers) {
        throw new RangeError(
          `baseArrayLayer (${baseArrayLayer}) out of bounds for texture ${textureId} (arrayLayers=${maxArrayLayers})`
        );
      }
      if (typeof arrayLayerCount === 'number' && baseArrayLayer + arrayLayerCount > maxArrayLayers) {
        throw new RangeError(
          `baseArrayLayer (${baseArrayLayer}) + arrayLayerCount (${arrayLayerCount}) exceeds texture ${textureId} arrayLayers (${maxArrayLayers})`
        );
      }
    }

    const gpuViewDescriptor = {
      dimension,
      baseMipLevel,
      baseArrayLayer,
      aspect,
    };
    if (format) {
      gpuViewDescriptor.format = format;
    }
    if (typeof mipLevelCount === 'number') {
      gpuViewDescriptor.mipLevelCount = mipLevelCount;
    }
    if (typeof arrayLayerCount === 'number') {
      gpuViewDescriptor.arrayLayerCount = arrayLayerCount;
    }
    if (typeof desc.label === 'string' && desc.label.length > 0) {
      gpuViewDescriptor.label = desc.label;
    }

    const normalized = {
      textureId,
      format,
      dimension,
      baseMipLevel,
      mipLevelCount,
      baseArrayLayer,
      arrayLayerCount,
      aspect,
    };
    const key = buildTextureViewCacheKey(textureId, normalized);

    return {
      ...normalized,
      key,
      gpuViewDescriptor,
    };
  }

  /**
   * Check whether a WebGPU device supports `texture-compression-bc`.
   * Returns `false` only when `device.features` exists and does not contain
   * `'texture-compression-bc'`.
   *
   * @param {object | null | undefined} device
   * @returns {boolean}
   */
  function deviceSupportsBc(device) {
    if (!device || !device.features) {
      return true;
    }
    const features = device.features;
    if (typeof features.has === 'function') {
      return Boolean(features.has(TEXTURE_COMPRESSION_BC_FEATURE));
    }
    if (Array.isArray(features)) {
      return features.includes(TEXTURE_COMPRESSION_BC_FEATURE);
    }
    if (typeof features[Symbol.iterator] === 'function') {
      for (const item of features) {
        if (item === TEXTURE_COMPRESSION_BC_FEATURE) {
          return true;
        }
      }
      return false;
    }
    if (typeof features === 'object') {
      return Boolean(features[TEXTURE_COMPRESSION_BC_FEATURE]);
    }
    return false;
  }

  function resolveTextureFormat(format, useV1Table = false) {
    if (typeof format === 'string' && format.length > 0 && format !== 'undefined') {
      return format.toLowerCase();
    }
    if (typeof format === 'number' && Number.isInteger(format)) {
      if (useV1Table) {
        return V1_TEXTURE_FORMAT_NAMES[format] || BC_FALLBACK_FORMAT;
      }
      const name = TEXTURE_FORMAT_NAMES[format];
      if (format > 0 && name && name !== 'undefined') {
        return name;
      }
    }
    return BC_FALLBACK_FORMAT;
  }

  function resolveTextureDimension(dimension) {
    if (typeof dimension === 'string') {
      const lower = dimension.toLowerCase();
      if (lower === '1d' || lower === '2d' || lower === '3d') {
        return lower;
      }
    }
    if (typeof dimension === 'number' && Number.isInteger(dimension)) {
      const name = TEXTURE_DIMENSION_NAMES[dimension];
      if (name) {
        return name;
      }
    }
    return '2d';
  }

  /**
   * Compute the maximum valid WebGPU mip level count for texture dimensions:
   * `floor(log2(max(width, height))) + 1` (or `max(w, h, d)` for `'3d'`).
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {string} [dimension='2d']
   * @returns {number}
   */
  function computeMaxMipLevels(width, height, depthOrArrayLayers = 1, dimension = '2d') {
    const w = Math.max(1, Math.floor(Number(width) || 1));
    const h = dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(height) || 1));
    const d = Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
    const maxDim = dimension === '3d' ? Math.max(w, h, d) : Math.max(w, h);
    return Math.floor(Math.log2(maxDim)) + 1;
  }

  /**
   * Clamp a requested mip level count into `[1, maxMipLevels]`.
   *
   * @param {number | undefined | null} requestedMipLevels
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {string} [dimension='2d']
   * @returns {{ mipLevelCount: number, maxMipLevels: number, clamped: boolean }}
   */
  function clampMipLevelCount(
    requestedMipLevels,
    width,
    height,
    depthOrArrayLayers = 1,
    dimension = '2d'
  ) {
    const maxMipLevels = computeMaxMipLevels(width, height, depthOrArrayLayers, dimension);
    if (requestedMipLevels === undefined || requestedMipLevels === null) {
      return { mipLevelCount: 1, maxMipLevels, clamped: false };
    }
    const numeric = Number(requestedMipLevels);
    if (!Number.isFinite(numeric)) {
      return { mipLevelCount: 1, maxMipLevels, clamped: true };
    }
    const truncated = Math.floor(numeric);
    const clampedValue = Math.min(maxMipLevels, Math.max(1, truncated));
    return {
      mipLevelCount: clampedValue,
      maxMipLevels,
      clamped: clampedValue !== numeric,
    };
  }

  /**
   * Clamp a texture `sampleCount` to WebGPU's valid set `{1, 4}`, falling back
   * to `1` when `mipLevelCount > 1` or multisampling is invalid for the
   * format/dimension/layers/usage.
   *
   * @param {number | undefined | null} requestedSampleCount
   * @param {{
   *   mipLevelCount?: number,
   *   format?: string,
   *   dimension?: string,
   *   depthOrArrayLayers?: number,
   *   usage?: number,
   * }} [context]
   * @returns {{ sampleCount: number, clamped: boolean }}
   */
  function clampSampleCount(requestedSampleCount, context = {}) {
    if (requestedSampleCount === undefined || requestedSampleCount === null) {
      return { sampleCount: 1, clamped: false };
    }
    const numeric = Number(requestedSampleCount);
    if (!Number.isFinite(numeric)) {
      return { sampleCount: 1, clamped: true };
    }

    const mipLevelCount = context.mipLevelCount !== undefined ? context.mipLevelCount : 1;
    const format = (context.format || BC_FALLBACK_FORMAT).toLowerCase();
    const dimension = context.dimension || '2d';
    const depthOrArrayLayers =
      context.depthOrArrayLayers !== undefined ? context.depthOrArrayLayers : 1;
    const usage = context.usage !== undefined ? Number(context.usage) >>> 0 : 0;

    const multisampleSupported =
      mipLevelCount <= 1 &&
      dimension === '2d' &&
      depthOrArrayLayers === 1 &&
      MULTISAMPLE_SUPPORTED_FORMATS.has(format) &&
      (usage & GPU_TEXTURE_USAGE.STORAGE_BINDING) === 0;

    let resolved;
    if (numeric <= 1) {
      resolved = 1;
    } else if (numeric >= 4 && multisampleSupported) {
      resolved = 4;
    } else {
      resolved = 1;
    }

    return {
      sampleCount: resolved,
      clamped: resolved !== numeric,
    };
  }

  /**
   * Round a single texture dimension up to a multiple of 4 (minimum 4) for BCn blocks.
   *
   * @param {number} dim
   * @returns {number}
   */
  function alignBcDimension(dim) {
    const n = Math.max(1, Math.floor(Number(dim) || 1));
    return Math.ceil(n / 4) * 4;
  }

  /**
   * Align a WebGPU `createBuffer` byte size up to a multiple of 4 bytes:
   * `Math.max(4, Math.ceil(size / 4) * 4)`, with a minimum of 16 bytes when
   * `GPUBufferUsage.UNIFORM` is set.
   *
   * @param {number | bigint} requestedSize
   * @param {number} [usage=0]
   * @returns {{ requestedSize: number, alignedSize: number, padded: boolean, isUniform: boolean }}
   */
  function alignBufferSize(requestedSize, usage = 0) {
    const numeric = typeof requestedSize === 'bigint' ? Number(requestedSize) : Number(requestedSize);
    const rawSize = Number.isFinite(numeric) && numeric > 0 ? Math.floor(numeric) : 0;
    const usageFlags = Number(usage) >>> 0;
    const isUniform = (usageFlags & GPU_BUFFER_USAGE.UNIFORM) !== 0;
    const aligned4 = Math.max(BUFFER_ALIGNMENT_BYTES, Math.ceil(rawSize / 4) * 4);
    const alignedSize = isUniform ? Math.max(UNIFORM_BUFFER_MIN_BYTES, aligned4) : aligned4;
    return {
      requestedSize: rawSize,
      alignedSize,
      padded: alignedSize !== rawSize,
      isUniform,
    };
  }

  /**
   * Normalize and 4-byte-align a `queue.writeBuffer` payload, zero-padding
   * non-4-byte-aligned inputs and safely copying `SharedArrayBuffer`-backed views
   * into a non-shared `Uint8Array`.
   *
   * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
   * @param {number} [dataOffset]
   * @param {number} [size]
   * @returns {{
   *   bytes: Uint8Array,
   *   originalByteLength: number,
   *   alignedByteLength: number,
   *   padded: boolean,
   *   copiedFromShared: boolean,
   * }}
   */
  function padBufferUploadData(data, dataOffset, size) {
    let byteView;
    if (data instanceof Uint8Array) {
      if (dataOffset === undefined && size === undefined) {
        byteView = data;
      } else {
        const start = dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset)) : 0;
        const maxLen = Math.max(0, data.byteLength - start);
        const len = size !== undefined ? Math.min(maxLen, Math.max(0, Math.floor(size))) : maxLen;
        byteView = data.subarray(start, start + len);
      }
    } else if (ArrayBuffer.isView(data)) {
      const elemSize = data.BYTES_PER_ELEMENT || 1;
      const startBytes =
        dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset) * elemSize) : 0;
      const maxBytes = Math.max(0, data.byteLength - startBytes);
      const lenBytes =
        size !== undefined
          ? Math.min(maxBytes, Math.max(0, Math.floor(size) * elemSize))
          : maxBytes;
      byteView = new Uint8Array(data.buffer, data.byteOffset + startBytes, lenBytes);
    } else if (
      data instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && data instanceof SharedArrayBuffer)
    ) {
      const startBytes = dataOffset !== undefined ? Math.max(0, Math.floor(dataOffset)) : 0;
      const maxBytes = Math.max(0, data.byteLength - startBytes);
      const lenBytes =
        size !== undefined ? Math.min(maxBytes, Math.max(0, Math.floor(size))) : maxBytes;
      byteView = new Uint8Array(data, startBytes, lenBytes);
    } else {
      throw new TypeError('writeBuffer data must be an ArrayBufferView, ArrayBuffer, or SharedArrayBuffer');
    }

    const originalByteLength = byteView.byteLength;
    const isShared =
      typeof SharedArrayBuffer !== 'undefined' && byteView.buffer instanceof SharedArrayBuffer;

    if (originalByteLength === 0) {
      return {
        bytes: new Uint8Array(0),
        originalByteLength: 0,
        alignedByteLength: 0,
        padded: false,
        copiedFromShared: isShared,
      };
    }

    const remainder = originalByteLength % BUFFER_ALIGNMENT_BYTES;
    if (remainder !== 0) {
      const alignedByteLength = originalByteLength + (BUFFER_ALIGNMENT_BYTES - remainder);
      const paddedCopy = new Uint8Array(alignedByteLength);
      paddedCopy.set(byteView, 0);
      return {
        bytes: paddedCopy,
        originalByteLength,
        alignedByteLength,
        padded: true,
        copiedFromShared: isShared,
      };
    }

    if (isShared) {
      const copy = new Uint8Array(originalByteLength);
      copy.set(byteView, 0);
      return {
        bytes: copy,
        originalByteLength,
        alignedByteLength: originalByteLength,
        padded: false,
        copiedFromShared: true,
      };
    }

    return {
      bytes: byteView,
      originalByteLength,
      alignedByteLength: originalByteLength,
      padded: false,
      copiedFromShared: false,
    };
  }

  /**
   * Estimate VRAM byte size for a WebGPU texture descriptor (including mip chain,
   * BC block sizes, array layers / 3D depth, and MSAA sample count).
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {number} [mipLevelCount=1]
   * @param {string} [format='rgba8unorm']
   * @param {string} [dimension='2d']
   * @param {number} [sampleCount=1]
   * @returns {number}
   */
  function estimateTextureBytes(
    width,
    height,
    depthOrArrayLayers = 1,
    mipLevelCount = 1,
    format = BC_FALLBACK_FORMAT,
    dimension = '2d',
    sampleCount = 1
  ) {
    const w0 = Math.max(1, Math.floor(Number(width) || 1));
    const h0 = Math.max(1, Math.floor(Number(height) || 1));
    const d0 = Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
    const mips = Math.max(1, Math.floor(Number(mipLevelCount) || 1));
    const samples = Math.max(1, Math.floor(Number(sampleCount) || 1));
    const fmt = typeof format === 'string' ? format.toLowerCase() : BC_FALLBACK_FORMAT;
    const is1D = dimension === '1d';
    const is3D = dimension === '3d';
    const compressed = isBcFormat(fmt);
    const bytesPerBlock = compressed ? (BC_8BYTE_BLOCK_FORMATS.has(fmt) ? 8 : 16) : 0;
    const bpp = UNCOMPRESSED_FORMAT_BPP[fmt] || 4;

    let total = 0;
    for (let mip = 0; mip < mips; mip += 1) {
      const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
      const mw = scale === Infinity ? 1 : Math.max(1, Math.floor(w0 / scale));
      const mh = is1D ? 1 : scale === Infinity ? 1 : Math.max(1, Math.floor(h0 / scale));
      const md = is3D ? (scale === Infinity ? 1 : Math.max(1, Math.floor(d0 / scale))) : d0;
      if (compressed) {
        total += Math.ceil(mw / 4) * Math.ceil(mh / 4) * bytesPerBlock * md;
      } else {
        total += mw * mh * md * bpp * samples;
      }
    }
    return total;
  }

  /**
   * Decode a binary ABI v2 `CreateTexture2` payload (opcode 35, 40 bytes,
   * `GpuCreateTexture2Payload` in `runtime/abi/playgta6_abi.h`).
   *
   * Preserves raw `mipLevelCount` and `sampleCount` so `sanitizeTextureDescriptor`
   * can clamp them accurately and track telemetry counters.
   *
   * @param {DataView} view
   * @param {number} [offset=0]
   * @param {number} [payloadBytes]
   * @returns {{ id: number, descriptor?: object, error?: string }}
   */
  function decodeCreateTexture2(view, offset = 0, payloadBytes) {
    if (!(view instanceof DataView)) {
      throw new TypeError('decodeCreateTexture2 expects a DataView');
    }
    const available = payloadBytes !== undefined ? payloadBytes : view.byteLength - offset;
    if (
      available < CREATE_TEXTURE2_PAYLOAD_BYTES ||
      offset < 0 ||
      offset + CREATE_TEXTURE2_PAYLOAD_BYTES > view.byteLength
    ) {
      return {
        id: 0,
        error: `CreateTexture2: payload ${available} < ${CREATE_TEXTURE2_PAYLOAD_BYTES} bytes`,
      };
    }

    const id = view.getUint32(offset, true);
    const width = view.getUint32(offset + 4, true);
    const height = view.getUint32(offset + 8, true);
    const depthOrArrayLayers = Math.max(1, view.getUint32(offset + 12, true));
    const mipLevelCount = view.getUint32(offset + 16, true);
    const sampleCount = view.getUint32(offset + 20, true);
    const usage = view.getUint32(offset + 24, true);
    const dimensionCode = view.getUint8(offset + 28);
    const formatCode = view.getUint8(offset + 29);
    const viewFormatCount = view.getUint8(offset + 30);

    const format = TEXTURE_FORMAT_NAMES[formatCode];
    if (formatCode === 0 || format === undefined) {
      return {
        id,
        error: `CreateTexture2 ${id}: invalid texture format code ${formatCode}`,
      };
    }

    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode];
    if (dimension === undefined) {
      return {
        id,
        error: `CreateTexture2 ${id}: invalid dimension ${dimensionCode}`,
      };
    }

    if (viewFormatCount > MAX_TEXTURE_VIEW_FORMATS) {
      return {
        id,
        error: `CreateTexture2 ${id}: view_format_count ${viewFormatCount} > ${MAX_TEXTURE_VIEW_FORMATS}`,
      };
    }

    const viewFormats = [];
    for (let i = 0; i < viewFormatCount; i += 1) {
      const code = view.getUint8(offset + 32 + i);
      const name = TEXTURE_FORMAT_NAMES[code];
      if (code === 0 || name === undefined) {
        return {
          id,
          error: `CreateTexture2 ${id}: invalid view format code ${code}`,
        };
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
    if (viewFormats.length > 0) {
      descriptor.viewFormats = viewFormats;
    }

    return { id, descriptor };
  }

  function extractTextureSize(descriptor, dimension) {
    let rawWidth = 1;
    let rawHeight = 1;
    let rawDepth = 1;

    if (descriptor.size !== undefined && descriptor.size !== null) {
      if (Array.isArray(descriptor.size)) {
        rawWidth = descriptor.size[0] !== undefined ? descriptor.size[0] : 1;
        rawHeight = descriptor.size[1] !== undefined ? descriptor.size[1] : 1;
        rawDepth = descriptor.size[2] !== undefined ? descriptor.size[2] : 1;
      } else if (typeof descriptor.size === 'object') {
        rawWidth = descriptor.size.width !== undefined ? descriptor.size.width : 1;
        rawHeight = descriptor.size.height !== undefined ? descriptor.size.height : 1;
        rawDepth =
          descriptor.size.depthOrArrayLayers !== undefined
            ? descriptor.size.depthOrArrayLayers
            : descriptor.size.depth !== undefined
              ? descriptor.size.depth
              : descriptor.size.arrayLayers !== undefined
                ? descriptor.size.arrayLayers
                : 1;
      }
    } else {
      rawWidth = descriptor.width !== undefined ? descriptor.width : 1;
      rawHeight = descriptor.height !== undefined ? descriptor.height : 1;
      rawDepth =
        descriptor.depthOrArrayLayers !== undefined
          ? descriptor.depthOrArrayLayers
          : descriptor.depth_or_layers !== undefined
            ? descriptor.depth_or_layers
            : descriptor.depth !== undefined
              ? descriptor.depth
              : descriptor.arrayLayers !== undefined
                ? descriptor.arrayLayers
                : 1;
    }

    const width = Math.max(1, Math.floor(Number(rawWidth) || 1));
    const height = dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(rawHeight) || 1));
    const depthOrArrayLayers =
      dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(rawDepth) || 1));

    return { width, height, depthOrArrayLayers };
  }

  /**
   * Sanitize a WebGPU texture descriptor (including ABI v2 `CreateTexture2`
   * fields `arrayLayers`/`depth`, `mipLevels`, `sampleCount`, `dimension`):
   * - For `bc1..bc7` formats: if `device.features` exists and lacks
   *   `'texture-compression-bc'`, falls back cleanly to `'rgba8unorm'`; when BC
   *   is supported, aligns width/height up to multiples of 4 when needed.
   * - Clamps `mipLevelCount` to `[1, floor(log2(max(width, height))) + 1]`
   *   (or `max(w, h, d)` for `'3d'`).
   * - Clamps `sampleCount` to `{1, 4}`, falling back to `1` when
   *   `mipLevelCount > 1` or multisampling is invalid for the format/dimension.
   * - Normalizes `viewFormats` (remapping BC view formats on BC fallback).
   *
   * @param {object} rawDescriptor
   * @param {object | null | undefined} [device]
   * @param {{ useV1FormatTable?: boolean }} [options]
   * @returns {{
   *   descriptor: object,
   *   originalFormat: string,
   *   effectiveFormat: string,
   *   bcFormatFallback: boolean,
   *   bcDimensionAligned: boolean,
   *   mipClamped: boolean,
   *   sampleClamped: boolean,
   * }}
   */
  function sanitizeTextureDescriptor(rawDescriptor, device, options = {}) {
    if (!rawDescriptor || typeof rawDescriptor !== 'object') {
      throw new TypeError('sanitizeTextureDescriptor expects a descriptor object');
    }

    const dimension = resolveTextureDimension(rawDescriptor.dimension);
    const originalFormat = resolveTextureFormat(
      rawDescriptor.format,
      Boolean(options.useV1FormatTable)
    );
    const usage =
      rawDescriptor.usage !== undefined
        ? Number(rawDescriptor.usage) >>> 0
        : rawDescriptor.usageFlags !== undefined
          ? Number(rawDescriptor.usageFlags) >>> 0
          : rawDescriptor.usage_flags !== undefined
            ? Number(rawDescriptor.usage_flags) >>> 0
            : GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST;

    let { width, height, depthOrArrayLayers } = extractTextureSize(rawDescriptor, dimension);

    let effectiveFormat = originalFormat;
    let bcFormatFallback = false;
    let bcDimensionAligned = false;

    if (isBcFormat(originalFormat)) {
      if (!deviceSupportsBc(device)) {
        effectiveFormat = BC_FALLBACK_FORMAT;
        bcFormatFallback = true;
      } else {
        const alignedWidth = alignBcDimension(width);
        const alignedHeight = dimension === '1d' ? height : alignBcDimension(height);
        if (alignedWidth !== width || alignedHeight !== height) {
          width = alignedWidth;
          height = alignedHeight;
          bcDimensionAligned = true;
        }
      }
    }

    const requestedMipLevels =
      rawDescriptor.mipLevelCount !== undefined
        ? rawDescriptor.mipLevelCount
        : rawDescriptor.mipLevels !== undefined
          ? rawDescriptor.mipLevels
          : rawDescriptor.mip_levels;
    const mipResult = clampMipLevelCount(
      requestedMipLevels,
      width,
      height,
      depthOrArrayLayers,
      dimension
    );

    const requestedSampleCount =
      rawDescriptor.sampleCount !== undefined
        ? rawDescriptor.sampleCount
        : rawDescriptor.sample_count !== undefined
          ? rawDescriptor.sample_count
          : rawDescriptor.samples;
    const sampleResult = clampSampleCount(requestedSampleCount, {
      mipLevelCount: mipResult.mipLevelCount,
      format: effectiveFormat,
      dimension,
      depthOrArrayLayers,
      usage,
    });

    const sanitizedDesc = {
      size: { width, height, depthOrArrayLayers },
      mipLevelCount: mipResult.mipLevelCount,
      sampleCount: sampleResult.sampleCount,
      dimension,
      format: effectiveFormat,
      usage,
    };

    if (typeof rawDescriptor.label === 'string' && rawDescriptor.label.length > 0) {
      sanitizedDesc.label = rawDescriptor.label;
    }

    const rawViewFormats =
      rawDescriptor.viewFormats !== undefined
        ? rawDescriptor.viewFormats
        : rawDescriptor.view_formats;
    const includeSrgbSiblingViewFormat = Boolean(
      (options &&
        (options.includeSrgbSiblingViewFormat || options.includeSrgbSibling)) ||
        rawDescriptor.includeSrgbSiblingViewFormat ||
        rawDescriptor.includeSrgbSibling
    );
    const resolvedViews = resolveTextureViewFormats(effectiveFormat, rawViewFormats, {
      bcFormatFallback,
      includeSrgbSiblingViewFormat,
    });
    if (resolvedViews.viewFormats.length > 0) {
      sanitizedDesc.viewFormats = resolvedViews.viewFormats;
    }

    return {
      descriptor: sanitizedDesc,
      originalFormat,
      effectiveFormat,
      siblingViewFormat: resolvedViews.siblingViewFormat,
      droppedViewFormats: resolvedViews.droppedViewFormats,
      bcFormatFallback,
      bcDimensionAligned,
      mipClamped: mipResult.clamped,
      sampleClamped: sampleResult.clamped,
    };
  }

  /**
   * Tracks WebGPU resources by `(kind, handle)` with VRAM byte accounting,
   * frame-epoch / GPU-fence deferred destruction, and hardened Frame-1
   * texture/buffer creation & upload helpers.
   */
  class GpuResourceRegistry {
    /**
     * @param {{
     *   framesInFlight?: number,
     *   softBudgetBytes?: number | null,
     *   device?: object | null,
     *   onBudgetWarning?: ((info: {
     *     activeBytes: number,
     *     pendingRetireBytes: number,
     *     totalBytes: number,
     *     softBudgetBytes: number,
     *     currentFrame: number,
     *     kind: string,
     *     handle: number,
     *     label: string,
     *   }) => void) | null,
     *   onDestroyError?: ((err: unknown, entry: {
     *     kind: string,
     *     handle: number,
     *     byteSize: number,
     *     label: string,
     *   }) => void) | null,
     *   onSanitizeWarning?: ((event: {
     *     kind: string,
     *     handle: number,
     *     detail: string,
     *   }) => void) | null,
     * }} [options]
     */
    constructor(options = {}) {
      if (options === null || typeof options !== 'object') {
        throw new TypeError('GpuResourceRegistry constructor expects an options object');
      }

      const {
        perfProfile: rawPerfProfile,
        framesInFlight: rawFramesInFlight,
        softBudgetBytes: rawSoftBudgetBytes,
        vramReclaimTriggerRatio: rawTriggerRatio,
        vramReclaimTargetRatio: rawTargetRatio,
        device = null,
        includeSrgbSiblingViewFormat = false,
        maxCachedViewsPerTexture = DEFAULT_MAX_CACHED_VIEWS_PER_TEXTURE,
        maxStaleHandleHistory = DEFAULT_MAX_STALE_HANDLE_HISTORY,
        onBudgetWarning = null,
        onPressureChange = null,
        onDestroyError = null,
        onSanitizeWarning = null,
      } = options;

      const resolvedPerfProfile =
        rawPerfProfile !== undefined && rawPerfProfile !== null
          ? _coercePerfProfile(rawPerfProfile)
          : null;

      const framesInFlight =
        rawFramesInFlight !== undefined
          ? rawFramesInFlight
          : resolvedPerfProfile && resolvedPerfProfile.gcDeferFrames !== undefined
            ? resolvedPerfProfile.gcDeferFrames
            : DEFAULT_FRAMES_IN_FLIGHT;

      const softBudgetBytes =
        rawSoftBudgetBytes !== undefined
          ? rawSoftBudgetBytes
          : resolvedPerfProfile && resolvedPerfProfile.vramBudgetBytes !== undefined
            ? resolvedPerfProfile.vramBudgetBytes
            : DEFAULT_SOFT_BUDGET_BYTES;

      const vramReclaimTriggerRatio =
        rawTriggerRatio !== undefined
          ? rawTriggerRatio
          : resolvedPerfProfile && resolvedPerfProfile.vramReclaimTriggerRatio !== undefined
            ? resolvedPerfProfile.vramReclaimTriggerRatio
            : DEFAULT_VRAM_RECLAIM_TRIGGER_RATIO;

      const vramReclaimTargetRatio =
        rawTargetRatio !== undefined
          ? rawTargetRatio
          : resolvedPerfProfile && resolvedPerfProfile.vramReclaimTargetRatio !== undefined
            ? resolvedPerfProfile.vramReclaimTargetRatio
            : DEFAULT_VRAM_RECLAIM_TARGET_RATIO;

      assertPositiveInteger(framesInFlight, 'framesInFlight');
      assertPositiveInteger(maxCachedViewsPerTexture, 'maxCachedViewsPerTexture');
      assertPositiveInteger(maxStaleHandleHistory, 'maxStaleHandleHistory');
      assertUnitRatio(vramReclaimTriggerRatio, 'vramReclaimTriggerRatio');
      assertUnitRatio(vramReclaimTargetRatio, 'vramReclaimTargetRatio');

      let resolvedBudget = null;
      if (softBudgetBytes !== null && softBudgetBytes !== Infinity) {
        assertPositiveInteger(softBudgetBytes, 'softBudgetBytes');
        resolvedBudget = softBudgetBytes;
      }

      if (device !== null && typeof device !== 'object') {
        throw new TypeError('device must be an object or null when provided');
      }
      if (onBudgetWarning !== null && typeof onBudgetWarning !== 'function') {
        throw new TypeError('onBudgetWarning must be a function when provided');
      }
      if (onPressureChange !== null && typeof onPressureChange !== 'function') {
        throw new TypeError('onPressureChange must be a function when provided');
      }
      if (onDestroyError !== null && typeof onDestroyError !== 'function') {
        throw new TypeError('onDestroyError must be a function when provided');
      }
      if (onSanitizeWarning !== null && typeof onSanitizeWarning !== 'function') {
        throw new TypeError('onSanitizeWarning must be a function when provided');
      }

      this._perfProfile = resolvedPerfProfile;
      this._perfProfileName = resolvedPerfProfile ? resolvedPerfProfile.name || null : null;
      this.framesInFlight = framesInFlight;
      this.softBudgetBytes = resolvedBudget;
      this.vramReclaimTriggerRatio = vramReclaimTriggerRatio;
      this.vramReclaimTargetRatio = Math.min(vramReclaimTargetRatio, vramReclaimTriggerRatio);
      this.device = device;
      this.includeSrgbSiblingViewFormat = Boolean(includeSrgbSiblingViewFormat);
      this.maxCachedViewsPerTexture = maxCachedViewsPerTexture;
      this._maxStaleHandleHistory = maxStaleHandleHistory;
      this._onBudgetWarning = onBudgetWarning;
      this._onPressureChange = onPressureChange;
      this._onDestroyError = onDestroyError;
      this._onSanitizeWarning = onSanitizeWarning;
      this._lastPressureTier = VRAM_PRESSURE_TIER.NORMAL;

      /** @type {Map<string, Map<number, object>>} */
      this._tables = new Map();
      /** @type {Record<string, Map<number, object>>} */
      this._tablesByKind = Object.create(null);
      /** @type {Record<string, Array<object | undefined>>} */
      this._fastSlotsByKind = Object.create(null);
      /** @type {Record<string, Set<number>>} */
      this._retiredHandlesByKind = Object.create(null);
      /** @type {Record<string, number>} */
      this._activeBytesByKind = {};
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const kind = RESOURCE_KINDS[i];
        const kindMap = new Map();
        this._tables.set(kind, kindMap);
        this._tablesByKind[kind] = kindMap;
        this._fastSlotsByKind[kind] = [];
        this._retiredHandlesByKind[kind] = new Set();
        this._activeBytesByKind[kind] = 0;
      }

      // Direct hot-path table & dense slot references (zero Map/string dispatch on typed lookups).
      this._bufferTable = this._tablesByKind[RESOURCE_KIND.BUFFER];
      this._textureTable = this._tablesByKind[RESOURCE_KIND.TEXTURE];
      this._textureViewTable = this._tablesByKind[RESOURCE_KIND.TEXTURE_VIEW];
      this._samplerTable = this._tablesByKind[RESOURCE_KIND.SAMPLER];
      this._bindGroupTable = this._tablesByKind[RESOURCE_KIND.BIND_GROUP];
      this._renderPipelineTable = this._tablesByKind[RESOURCE_KIND.RENDER_PIPELINE];
      this._computePipelineTable = this._tablesByKind[RESOURCE_KIND.COMPUTE_PIPELINE];

      this._fastBuffers = this._fastSlotsByKind[RESOURCE_KIND.BUFFER];
      this._fastTextures = this._fastSlotsByKind[RESOURCE_KIND.TEXTURE];
      this._fastTextureViews = this._fastSlotsByKind[RESOURCE_KIND.TEXTURE_VIEW];
      this._fastSamplers = this._fastSlotsByKind[RESOURCE_KIND.SAMPLER];
      this._fastBindGroups = this._fastSlotsByKind[RESOURCE_KIND.BIND_GROUP];
      this._fastRenderPipelines = this._fastSlotsByKind[RESOURCE_KIND.RENDER_PIPELINE];
      this._fastComputePipelines = this._fastSlotsByKind[RESOURCE_KIND.COMPUTE_PIPELINE];

      this._retiredBuffers = this._retiredHandlesByKind[RESOURCE_KIND.BUFFER];
      this._retiredTextures = this._retiredHandlesByKind[RESOURCE_KIND.TEXTURE];
      this._retiredTextureViews = this._retiredHandlesByKind[RESOURCE_KIND.TEXTURE_VIEW];
      this._retiredSamplers = this._retiredHandlesByKind[RESOURCE_KIND.SAMPLER];
      this._retiredBindGroups = this._retiredHandlesByKind[RESOURCE_KIND.BIND_GROUP];
      this._retiredRenderPipelines = this._retiredHandlesByKind[RESOURCE_KIND.RENDER_PIPELINE];
      this._retiredComputePipelines = this._retiredHandlesByKind[RESOURCE_KIND.COMPUTE_PIPELINE];

      // Per-VRAM-kind live/pending/peak accounting (W300, #1077/#1207).
      this._vramLiveBytesByKind = Object.create(null);
      this._vramPendingBytesByKind = Object.create(null);
      this._vramPeakBytesByKind = Object.create(null);
      this._vramPeakLiveBytesByKind = Object.create(null);
      this._vramLiveCountByKind = Object.create(null);
      for (let i = 0; i < VRAM_KINDS.length; i += 1) {
        const vk = VRAM_KINDS[i];
        this._vramLiveBytesByKind[vk] = 0;
        this._vramPendingBytesByKind[vk] = 0;
        this._vramPeakBytesByKind[vk] = 0;
        this._vramPeakLiveBytesByKind[vk] = 0;
        this._vramLiveCountByKind[vk] = 0;
      }

      /** @type {Map<number, object>} */
      this._bufferMeta = new Map();
      /** @type {Map<number, object>} */
      this._textureMeta = new Map();

      // Per-texture generation counters and cached GPUTextureView tracking (W178).
      /** @type {Map<number, number>} */
      this._textureGenerationByHandle = new Map();
      /** @type {Map<number, number>} */
      this._activeTextureGeneration = new Map();
      /** @type {Map<number, Map<string, object>>} */
      this._textureViewCache = new Map();
      /** @type {Map<number, Set<number>>} */
      this._linkedViewHandlesByTexture = new Map();
      /** @type {WeakMap<object, object>} */
      this._viewMetadata = new WeakMap();

      /** @type {Array<object>} */
      this._pendingRetire = [];
      /** @type {Set<Promise<void>>} */
      this._pendingFences = new Set();
      /** @type {WeakSet<object>} */
      this._destroyedResources = new WeakSet();
      this._hasDestroyedResources = false;

      this._currentFrame = 0;
      this._generation = 0;
      this._nextRegistrationSeq = 0;

      this._activeBytes = 0;
      this._pendingRetireBytes = 0;
      this._peakBytes = 0;

      this._budgetExceeded = false;
      this._budgetWarningCount = 0;

      this._totalRegistered = 0;
      this._totalRetired = 0;
      this._totalDestroyed = 0;
      this._destroyErrors = 0;

      // W300 stale-eviction and O(1) fast-path telemetry counters.
      this._evictedCount = 0;
      this._evictedBytes = 0;
      this._staleLookupMisses = 0;
      this._fastHandleHits = 0;
      this._fastHandleOverflowHits = 0;

      // Sanitized Frame-1 hardening counters exposed on getStats().
      this._bcFormatFallbackCount = 0;
      this._bcDimensionAlignCount = 0;
      this._bufferAlignPadCount = 0;
      this._mipClampCount = 0;
      this._sampleClampCount = 0;

      // Additive W178/W201/W250 view-cache, format, and peak-live telemetry exposed on getStats().
      this._staleViewsInvalidated = 0;
      this._viewCacheHits = 0;
      this._viewCacheMisses = 0;
      this._viewCacheInvalidations = 0;
      this._viewCacheEvictions = 0;
      this._textureRecreations = 0;
      this._r8unormTextures = 0;
      this._rgba8unormTextures = 0;
      this._peakLiveTextures = 0;
      this._peakLiveBuffers = 0;
    }

    /**
     * Attach or replace the WebGPU device used by `createTexture`,
     * `createTexture2`, `createBuffer`, and `writeBuffer`.
     *
     * @param {object | null} device
     */
    setDevice(device) {
      if (device !== null && typeof device !== 'object') {
        throw new TypeError('setDevice expects a WebGPU device object or null');
      }
      this.device = device;
    }

    get perfProfile() {
      return this._perfProfile;
    }

    /**
     * Compute current VRAM budget utilization and pressure tier (`normal` | `soft` | `hard`)
     * against `softBudgetBytes`, `vramReclaimTriggerRatio`, and `vramReclaimTargetRatio`.
     */
    getVramPressure() {
      const activeBytes = sanitizeNonNegativeCounter(this._activeBytes);
      const pendingRetireBytes = sanitizeNonNegativeCounter(this._pendingRetireBytes);
      const totalBytes = activeBytes + pendingRetireBytes;
      const budget = this.softBudgetBytes;

      if (budget === null || budget <= 0) {
        return {
          tier: VRAM_PRESSURE_TIER.NORMAL,
          utilization: 0,
          budgetUtilization: 0,
          totalBytes,
          softBudgetBytes: null,
          triggerRatio: this.vramReclaimTriggerRatio,
          targetRatio: this.vramReclaimTargetRatio,
          triggerBytes: null,
          targetBytes: null,
          shouldReclaim: false,
          bytesOverTarget: 0,
          budgetExceeded: false,
        };
      }

      const utilization = totalBytes > 0 ? totalBytes / budget : 0;
      const budgetUtilization = clampUnitRate(totalBytes, budget);
      const triggerBytes = Math.floor(budget * this.vramReclaimTriggerRatio);
      const targetBytes = Math.floor(budget * this.vramReclaimTargetRatio);

      let tier = VRAM_PRESSURE_TIER.NORMAL;
      if (totalBytes >= budget) {
        tier = VRAM_PRESSURE_TIER.HARD;
      } else if (utilization >= this.vramReclaimTriggerRatio) {
        tier = VRAM_PRESSURE_TIER.SOFT;
      }

      const shouldReclaim = tier !== VRAM_PRESSURE_TIER.NORMAL;
      const bytesOverTarget = shouldReclaim ? Math.max(0, totalBytes - targetBytes) : 0;

      return {
        tier,
        utilization,
        budgetUtilization,
        totalBytes,
        softBudgetBytes: budget,
        triggerRatio: this.vramReclaimTriggerRatio,
        targetRatio: this.vramReclaimTargetRatio,
        triggerBytes,
        targetBytes,
        shouldReclaim,
        bytesOverTarget,
        budgetExceeded: this._budgetExceeded,
      };
    }

    /**
     * Apply a hardware-adaptive performance profile (`'low' | 'balanced' | 'high' | 'auto'`
     * or a resolved profile object from `PlayGta6PerfProfile.resolvePerfProfile`), updating
     * VRAM soft budget, reclaim trigger/target ratios, and deferred-GC frame retention.
     * Immediately collects any retired non-fence resources now eligible under the updated
     * deferred-GC delay.
     *
     * @param {string | object} profileOrName
     */
    applyPerfProfile(profileOrName) {
      const profile = _coercePerfProfile(profileOrName);
      this._perfProfile = profile;
      this._perfProfileName = profile.name || null;

      if (profile.vramBudgetBytes !== undefined && profile.vramBudgetBytes !== null) {
        assertPositiveInteger(profile.vramBudgetBytes, 'vramBudgetBytes');
        this.softBudgetBytes = profile.vramBudgetBytes;
      }
      if (profile.vramReclaimTriggerRatio !== undefined) {
        assertUnitRatio(profile.vramReclaimTriggerRatio, 'vramReclaimTriggerRatio');
        this.vramReclaimTriggerRatio = profile.vramReclaimTriggerRatio;
      }
      if (profile.vramReclaimTargetRatio !== undefined) {
        assertUnitRatio(profile.vramReclaimTargetRatio, 'vramReclaimTargetRatio');
        this.vramReclaimTargetRatio = Math.min(
          profile.vramReclaimTargetRatio,
          this.vramReclaimTriggerRatio
        );
      } else if (this.vramReclaimTargetRatio > this.vramReclaimTriggerRatio) {
        this.vramReclaimTargetRatio = this.vramReclaimTriggerRatio;
      }

      const prevFramesInFlight = this.framesInFlight;
      if (profile.gcDeferFrames !== undefined) {
        assertPositiveInteger(profile.gcDeferFrames, 'gcDeferFrames');
        this.framesInFlight = profile.gcDeferFrames;
      }

      let collected = {
        destroyedCount: 0,
        freedBytes: 0,
        remainingCount: this._pendingRetire.length,
        pendingRetireBytes: this._pendingRetireBytes,
      };
      if (this.framesInFlight < prevFramesInFlight && this._pendingRetire.length > 0) {
        collected = this.collect(false);
      } else {
        this._checkBudget(null);
      }

      return {
        profile: this._perfProfileName,
        softBudgetBytes: this.softBudgetBytes,
        vramBudgetBytes: this.softBudgetBytes,
        vramReclaimTriggerRatio: this.vramReclaimTriggerRatio,
        vramReclaimTargetRatio: this.vramReclaimTargetRatio,
        framesInFlight: this.framesInFlight,
        gcDeferFrames: this.framesInFlight,
        pressure: this.getVramPressure(),
        collected,
      };
    }

    _resolveDevice(overrideDevice, methodName) {
      const target = overrideDevice !== undefined && overrideDevice !== null ? overrideDevice : this.device;
      if (!target || typeof target !== 'object') {
        throw new TypeError(`GpuResourceRegistry.${methodName} requires a WebGPU device`);
      }
      return target;
    }

    _emitSanitizeWarning(kind, handle, detail) {
      if (this._onSanitizeWarning) {
        this._onSanitizeWarning({ kind, handle, detail });
      }
    }

    _updatePeakBytes() {
      const totalBytes =
        sanitizeNonNegativeCounter(this._activeBytes) +
        sanitizeNonNegativeCounter(this._pendingRetireBytes);
      if (totalBytes > this._peakBytes) {
        this._peakBytes = totalBytes;
      }
      return totalBytes;
    }

    _updateVramPeaks(vramKind) {
      const liveBytes = sanitizeNonNegativeCounter(this._vramLiveBytesByKind[vramKind]);
      const pendingBytes = sanitizeNonNegativeCounter(this._vramPendingBytesByKind[vramKind]);
      const totalKindBytes = liveBytes + pendingBytes;
      if (liveBytes > this._vramPeakLiveBytesByKind[vramKind]) {
        this._vramPeakLiveBytesByKind[vramKind] = liveBytes;
      }
      if (totalKindBytes > this._vramPeakBytesByKind[vramKind]) {
        this._vramPeakBytesByKind[vramKind] = totalKindBytes;
      }
    }

    _markHandleStale(kind, handle) {
      const retiredSet = this._retiredHandlesByKind[kind];
      if (!retiredSet) {
        return;
      }
      if (retiredSet.has(handle)) {
        retiredSet.delete(handle);
      } else if (retiredSet.size >= this._maxStaleHandleHistory) {
        const oldest = retiredSet.values().next().value;
        if (oldest !== undefined) {
          retiredSet.delete(oldest);
        }
      }
      retiredSet.add(handle);
    }

    _setFastSlot(kind, handle, resource) {
      if (handle >= 0 && handle < FAST_HANDLE_LIMIT) {
        this._fastSlotsByKind[kind][handle] = resource;
      }
    }

    _clearFastSlot(kind, handle, table) {
      if (handle >= 0 && handle < FAST_HANDLE_LIMIT) {
        const slots = this._fastSlotsByKind[kind];
        if (table.size === 0) {
          slots.length = 0;
        } else if (handle < slots.length) {
          slots[handle] = undefined;
        }
      }
    }

    _cleanupKindMetadata(kind, handle, entry) {
      if (kind === RESOURCE_KIND.BUFFER) {
        this._bufferMeta.delete(handle);
      } else if (kind === RESOURCE_KIND.TEXTURE) {
        this._textureMeta.delete(handle);
        this._activeTextureGeneration.delete(handle);
        this._invalidateTextureViews(handle, { retireLinkedTableViews: true });
      } else if (kind === RESOURCE_KIND.TEXTURE_VIEW) {
        if (entry.textureId !== undefined && entry.textureId !== null) {
          const linkedSet = this._linkedViewHandlesByTexture.get(entry.textureId);
          if (linkedSet) {
            linkedSet.delete(handle);
            if (linkedSet.size === 0) {
              this._linkedViewHandlesByTexture.delete(entry.textureId);
            }
          }
          if (entry.resource && typeof entry.resource === 'object') {
            const viewMeta = this._viewMetadata.get(entry.resource);
            if (viewMeta) {
              viewMeta.invalidated = true;
            }
          }
        }
      }
    }

    _peek(kind, handle) {
      const slots = this._fastSlotsByKind[kind];
      if (!slots) {
        return undefined;
      }
      if (handle >= 0 && handle < FAST_HANDLE_LIMIT && (handle | 0) === handle) {
        return slots[handle];
      }
      const entry = this._tablesByKind[kind].get(handle);
      return entry ? entry.resource : undefined;
    }

    _evictDestroyedEntry(kind, handle) {
      const table = this._tablesByKind[kind];
      if (!table) {
        return 0;
      }
      const entry = table.get(handle);
      if (!entry) {
        return 0;
      }

      table.delete(handle);
      this._clearFastSlot(kind, handle, table);
      this._markHandleStale(kind, handle);
      this._cleanupKindMetadata(kind, handle, entry);

      const byteSize = sanitizeNonNegativeCounter(entry.byteSize);
      const vramKind = entry.vramKind || VRAM_KIND.OTHER;

      this._activeBytes = Math.max(0, this._activeBytes - byteSize);
      this._activeBytesByKind[kind] = Math.max(0, this._activeBytesByKind[kind] - byteSize);
      this._vramLiveBytesByKind[vramKind] = Math.max(
        0,
        this._vramLiveBytesByKind[vramKind] - byteSize
      );
      this._vramLiveCountByKind[vramKind] = Math.max(
        0,
        this._vramLiveCountByKind[vramKind] - 1
      );

      if (entry.resource && typeof entry.resource === 'object') {
        this._destroyedResources.add(entry.resource);
        this._hasDestroyedResources = true;
      }
      entry.destroyed = true;
      entry.resource = null;

      this._totalDestroyed += 1;
      this._evictedCount += 1;
      this._evictedBytes += byteSize;
      this._checkBudget(null);
      return byteSize;
    }

    _buildVramByKindSnapshot() {
      const snapshot = {};
      for (let i = 0; i < VRAM_KINDS.length; i += 1) {
        const vk = VRAM_KINDS[i];
        const count = sanitizeNonNegativeCounter(this._vramLiveCountByKind[vk]);
        const liveBytes = sanitizeNonNegativeCounter(this._vramLiveBytesByKind[vk]);
        const pendingRetireBytes = sanitizeNonNegativeCounter(this._vramPendingBytesByKind[vk]);
        const totalBytes = liveBytes + pendingRetireBytes;
        const peakLiveBytes = Math.max(
          liveBytes,
          sanitizeNonNegativeCounter(this._vramPeakLiveBytesByKind[vk])
        );
        const peakBytes = Math.max(
          totalBytes,
          sanitizeNonNegativeCounter(this._vramPeakBytesByKind[vk])
        );
        snapshot[vk] = {
          count,
          liveBytes,
          pendingRetireBytes,
          totalBytes,
          peakBytes,
          peakLiveBytes,
        };
      }
      return snapshot;
    }

    _updatePeakLiveCounts(kind) {
      if (kind === RESOURCE_KIND.TEXTURE) {
        const liveTextures = this._tables.get(RESOURCE_KIND.TEXTURE).size;
        if (liveTextures > this._peakLiveTextures) {
          this._peakLiveTextures = liveTextures;
        }
      } else if (kind === RESOURCE_KIND.BUFFER) {
        const liveBuffers = this._tables.get(RESOURCE_KIND.BUFFER).size;
        if (liveBuffers > this._peakLiveBuffers) {
          this._peakLiveBuffers = liveBuffers;
        }
      }
    }

    _recordTextureFormatTelemetry(format) {
      if (typeof format !== 'string') {
        return;
      }
      const normalized = format.trim().toLowerCase();
      if (normalized === 'r8unorm') {
        this._r8unormTextures += 1;
      } else if (normalized === 'rgba8unorm') {
        this._rgba8unormTextures += 1;
      }
    }

    _checkBudget(context) {
      const totalBytes = this._activeBytes + this._pendingRetireBytes;
      if (this.softBudgetBytes === null) {
        this._budgetExceeded = false;
        return;
      }

      if (totalBytes > this.softBudgetBytes) {
        if (!this._budgetExceeded) {
          this._budgetExceeded = true;
          this._budgetWarningCount += 1;
          if (this._onBudgetWarning) {
            this._onBudgetWarning({
              activeBytes: this._activeBytes,
              pendingRetireBytes: this._pendingRetireBytes,
              totalBytes,
              softBudgetBytes: this.softBudgetBytes,
              currentFrame: this._currentFrame,
              kind: context ? context.kind : '',
              handle: context ? context.handle : -1,
              label: context ? context.label : '',
            });
          }
        }
      } else {
        this._budgetExceeded = false;
      }

      if (this._onPressureChange) {
        const pressure = this.getVramPressure();
        if (pressure.tier !== this._lastPressureTier) {
          const previousTier = this._lastPressureTier;
          this._lastPressureTier = pressure.tier;
          this._onPressureChange({
            ...pressure,
            previousTier,
            currentFrame: this._currentFrame,
          });
        }
      }
    }

    _sealUnsealedRetirements() {
      const batch = [];
      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        const entry = this._pendingRetire[i];
        if (!entry.sealed) {
          entry.sealed = true;
          batch.push(entry);
        }
      }
      return batch;
    }

    _destroyEntry(entry) {
      if (entry.destroyed) {
        return false;
      }
      entry.destroyed = true;

      const resource = entry.resource;
      entry.resource = null;

      if (resource && typeof resource === 'object' && !this._destroyedResources.has(resource)) {
        this._destroyedResources.add(resource);
        this._hasDestroyedResources = true;
        if (typeof resource.destroy === 'function') {
          try {
            resource.destroy();
          } catch (err) {
            this._destroyErrors += 1;
            if (this._onDestroyError) {
              this._onDestroyError(err, {
                kind: entry.kind,
                handle: entry.handle,
                byteSize: entry.byteSize,
                label: entry.label,
              });
            }
          }
        }
      }

      this._totalDestroyed += 1;
      return true;
    }

    _invalidateTextureViews(textureId, { retireLinkedTableViews = true } = {}) {
      let invalidatedCount = 0;
      const countedViews = new Set();

      const perTexCache = this._textureViewCache.get(textureId);
      if (perTexCache) {
        for (const cached of perTexCache.values()) {
          if (cached && cached.view && typeof cached.view === 'object') {
            const viewMeta = this._viewMetadata.get(cached.view);
            if (viewMeta) {
              viewMeta.invalidated = true;
            }
            if (!countedViews.has(cached.view)) {
              countedViews.add(cached.view);
              invalidatedCount += 1;
            }
          } else {
            invalidatedCount += 1;
          }
        }
        this._textureViewCache.delete(textureId);
      }

      if (retireLinkedTableViews) {
        const linkedHandles = this._linkedViewHandlesByTexture.get(textureId);
        if (linkedHandles && linkedHandles.size > 0) {
          const handlesCopy = Array.from(linkedHandles);
          this._linkedViewHandlesByTexture.delete(textureId);
          const viewTable = this._tables.get(RESOURCE_KIND.TEXTURE_VIEW);
          for (let i = 0; i < handlesCopy.length; i += 1) {
            const viewHandle = handlesCopy[i];
            const viewEntry = viewTable.get(viewHandle);
            if (viewEntry && viewEntry.textureId === textureId) {
              const viewObj = viewEntry.resource;
              if (viewObj && typeof viewObj === 'object') {
                const viewMeta = this._viewMetadata.get(viewObj);
                if (viewMeta) {
                  viewMeta.invalidated = true;
                }
                if (!countedViews.has(viewObj)) {
                  countedViews.add(viewObj);
                  invalidatedCount += 1;
                }
              } else {
                invalidatedCount += 1;
              }
              this.retire(RESOURCE_KIND.TEXTURE_VIEW, viewHandle);
            }
          }
        }
      }

      if (invalidatedCount > 0) {
        this._staleViewsInvalidated += invalidatedCount;
        this._viewCacheInvalidations += invalidatedCount;
      }
      return invalidatedCount;
    }

    _maybeBuildTextureMetaFromRegistration(handle, resource, meta, byteSize, textureGeneration) {
      if (meta && meta._skipAutoTextureMeta) {
        return null;
      }
      const desc =
        meta && meta.descriptor && typeof meta.descriptor === 'object' ? meta.descriptor : null;
      const hasExplicitMeta =
        Boolean(desc) ||
        (meta &&
          (meta.format !== undefined ||
            meta.viewFormats !== undefined ||
            meta.view_formats !== undefined ||
            meta.width !== undefined ||
            meta.height !== undefined ||
            meta.includeSrgbSiblingViewFormat !== undefined));
      const hasResourceFormat =
        resource && typeof resource === 'object' && typeof resource.format === 'string' && resource.format.length > 0;

      if (!hasExplicitMeta && !hasResourceFormat) {
        return null;
      }

      const rawFormat =
        desc && desc.format !== undefined
          ? desc.format
          : meta && meta.format !== undefined
            ? meta.format
            : hasResourceFormat
              ? resource.format
              : undefined;
      const effectiveFormat = resolveTextureFormat(rawFormat);
      const dimension = resolveTextureDimension(
        desc && desc.dimension !== undefined
          ? desc.dimension
          : meta && meta.dimension !== undefined
            ? meta.dimension
            : resource && resource.dimension !== undefined
              ? resource.dimension
              : '2d'
      );

      const metaHasSize =
        meta &&
        (meta.size !== undefined ||
          meta.width !== undefined ||
          meta.height !== undefined ||
          meta.depthOrArrayLayers !== undefined ||
          meta.depth_or_layers !== undefined ||
          meta.depth !== undefined ||
          meta.arrayLayers !== undefined);
      const sizeSource = desc || (metaHasSize ? meta : resource) || {};
      const extractedSize = extractTextureSize(sizeSource, dimension);
      const mipLevelCount = clampMipLevelCount(
        desc && desc.mipLevelCount !== undefined
          ? desc.mipLevelCount
          : meta && meta.mipLevelCount !== undefined
            ? meta.mipLevelCount
            : resource && resource.mipLevelCount !== undefined
              ? resource.mipLevelCount
              : 1,
        extractedSize.width,
        extractedSize.height,
        extractedSize.depthOrArrayLayers,
        dimension
      ).mipLevelCount;
      const sampleCount = clampSampleCount(
        desc && desc.sampleCount !== undefined
          ? desc.sampleCount
          : meta && meta.sampleCount !== undefined
            ? meta.sampleCount
            : resource && resource.sampleCount !== undefined
              ? resource.sampleCount
              : 1,
        {
          mipLevelCount,
          format: effectiveFormat,
          dimension,
          depthOrArrayLayers: extractedSize.depthOrArrayLayers,
          usage:
            desc && desc.usage !== undefined
              ? desc.usage
              : meta && meta.usage !== undefined
                ? meta.usage
                : resource && resource.usage !== undefined
                  ? resource.usage
                  : 0,
        }
      ).sampleCount;
      const usage =
        desc && desc.usage !== undefined
          ? Number(desc.usage) >>> 0
          : meta && meta.usage !== undefined
            ? Number(meta.usage) >>> 0
            : resource && resource.usage !== undefined
              ? Number(resource.usage) >>> 0
              : GPU_TEXTURE_USAGE.TEXTURE_BINDING | GPU_TEXTURE_USAGE.COPY_DST;

      const rawViewFormats =
        desc && desc.viewFormats !== undefined
          ? desc.viewFormats
          : desc && desc.view_formats !== undefined
            ? desc.view_formats
            : meta && meta.viewFormats !== undefined
              ? meta.viewFormats
              : meta && meta.view_formats !== undefined
                ? meta.view_formats
                : resource && Array.isArray(resource.viewFormats)
                  ? resource.viewFormats
                  : undefined;

      const includeSibling = Boolean(
        this.includeSrgbSiblingViewFormat ||
          (meta && (meta.includeSrgbSiblingViewFormat || meta.includeSrgbSibling)) ||
          (desc && (desc.includeSrgbSiblingViewFormat || desc.includeSrgbSibling))
      );

      const resolvedViews = resolveTextureViewFormats(effectiveFormat, rawViewFormats, {
        bcFormatFallback: false,
        includeSrgbSiblingViewFormat: includeSibling,
      });

      return {
        handle,
        width: extractedSize.width,
        height: extractedSize.height,
        depthOrArrayLayers: extractedSize.depthOrArrayLayers,
        mipLevelCount,
        sampleCount,
        dimension,
        format: effectiveFormat,
        originalFormat: effectiveFormat,
        usage,
        viewFormats: resolvedViews.viewFormats,
        siblingViewFormat: resolvedViews.siblingViewFormat,
        textureGeneration,
        bcFormatFallback: false,
        bcDimensionAligned: false,
        mipClamped: false,
        sampleClamped: false,
        isTexture2: Boolean(meta && meta.isTexture2),
        byteSize,
      };
    }

    /**
     * Register a WebGPU resource under `(kind, handle)`.
     * If a different resource already occupies `(kind, handle)`, the prior resource
     * is retired for deferred destruction rather than leaked.
     *
     * For `RESOURCE_KIND.TEXTURE`, replacing an active texture (or re-creating a
     * previously retired handle) bumps its monotonic `textureGeneration`,
     * increments `textureRecreations`, and automatically invalidates all cached
     * and linked `GPUTextureView` handles for that texture ID.
     *
     * @param {string} kind
     * @param {number} handle
     * @param {object} resource
     * @param {{
     *   byteSize?: number,
     *   label?: string,
     *   tag?: unknown,
     *   textureId?: number,
     *   descriptor?: object,
     *   format?: string | number,
     *   viewFormats?: Array<string | number>,
     *   includeSrgbSiblingViewFormat?: boolean,
     * }} [meta]
     * @returns {object} The registered resource
     */
    register(kind, handle, resource, meta = {}) {
      assertValidKind(kind);
      assertValidHandle(handle);

      if (resource === null || typeof resource !== 'object') {
        throw new TypeError('register requires a non-null resource object');
      }
      if (meta === null || typeof meta !== 'object') {
        throw new TypeError('register metadata must be an object when provided');
      }

      const { byteSize = 0, label = '', tag = null } = meta;
      assertNonNegativeInteger(byteSize, 'byteSize');
      if (typeof label !== 'string') {
        throw new TypeError('register metadata.label must be a string');
      }

      let linkedTextureId = null;
      if (
        kind === RESOURCE_KIND.TEXTURE_VIEW &&
        meta.textureId !== undefined &&
        meta.textureId !== null
      ) {
        assertValidHandle(meta.textureId);
        linkedTextureId = meta.textureId;
      }

      const table = this._tablesByKind[kind];
      const existing = table.get(handle);
      const vramKind = resolveVramKind(kind, resource, meta);

      if (existing) {
        if (existing.resource === resource) {
          const prevByteSize = sanitizeNonNegativeCounter(existing.byteSize);
          const prevVramKind = existing.vramKind || VRAM_KIND.OTHER;
          this._activeBytes = Math.max(0, this._activeBytes - prevByteSize) + byteSize;
          this._activeBytesByKind[kind] =
            Math.max(0, this._activeBytesByKind[kind] - prevByteSize) + byteSize;
          this._vramLiveBytesByKind[prevVramKind] = Math.max(
            0,
            this._vramLiveBytesByKind[prevVramKind] - prevByteSize
          );
          if (prevVramKind !== vramKind) {
            this._vramLiveCountByKind[prevVramKind] = Math.max(
              0,
              this._vramLiveCountByKind[prevVramKind] - 1
            );
            this._vramLiveCountByKind[vramKind] += 1;
          }
          this._vramLiveBytesByKind[vramKind] += byteSize;

          existing.byteSize = byteSize;
          existing.vramKind = vramKind;
          existing.label = label;
          existing.tag = tag;

          this._setFastSlot(kind, handle, resource);
          this._retiredHandlesByKind[kind].delete(handle);

          if (kind === RESOURCE_KIND.TEXTURE) {
            const currentGen = this._activeTextureGeneration.get(handle) || 1;
            const autoMeta = this._maybeBuildTextureMetaFromRegistration(
              handle,
              resource,
              meta,
              byteSize,
              currentGen
            );
            if (autoMeta) {
              this._textureMeta.set(handle, autoMeta);
            }
          }

          this._updateVramPeaks(vramKind);
          this._updatePeakBytes();
          this._checkBudget({ kind, handle, label });
          return resource;
        }
        // Cleanly retire the previous resource on handle overwrite so it is never leaked.
        this.retire(kind, handle);
      }

      let textureGeneration = 0;
      if (kind === RESOURCE_KIND.TEXTURE) {
        const prevGen = this._textureGenerationByHandle.get(handle) || 0;
        const isRecreation = prevGen > 0;
        textureGeneration = prevGen + 1;
        this._textureGenerationByHandle.set(handle, textureGeneration);
        this._activeTextureGeneration.set(handle, textureGeneration);
        if (isRecreation) {
          this._textureRecreations += 1;
        }
        // Defense-in-depth: ensure zero stale cached views exist for this handle.
        this._invalidateTextureViews(handle, { retireLinkedTableViews: true });
      } else if (kind === RESOURCE_KIND.TEXTURE_VIEW && linkedTextureId !== null) {
        textureGeneration = this._activeTextureGeneration.get(linkedTextureId) || 0;
      }

      const entry = {
        kind,
        vramKind,
        handle,
        resource,
        byteSize,
        label,
        tag,
        registeredAtFrame: this._currentFrame,
        registrationSeq: this._nextRegistrationSeq,
        retiredAtFrame: -1,
        sealed: false,
        hasFence: false,
        fenceDone: false,
        fenceFailed: false,
        destroyed: false,
        textureId: linkedTextureId,
        textureGeneration,
      };
      this._nextRegistrationSeq += 1;

      table.set(handle, entry);
      this._setFastSlot(kind, handle, resource);
      this._retiredHandlesByKind[kind].delete(handle);
      this._updatePeakLiveCounts(kind);

      if (kind === RESOURCE_KIND.TEXTURE) {
        const autoMeta = this._maybeBuildTextureMetaFromRegistration(
          handle,
          resource,
          meta,
          byteSize,
          textureGeneration
        );
        if (autoMeta) {
          this._textureMeta.set(handle, autoMeta);
          this._recordTextureFormatTelemetry(autoMeta.format);
        }
      } else if (kind === RESOURCE_KIND.TEXTURE_VIEW && linkedTextureId !== null) {
        let linkedSet = this._linkedViewHandlesByTexture.get(linkedTextureId);
        if (!linkedSet) {
          linkedSet = new Set();
          this._linkedViewHandlesByTexture.set(linkedTextureId, linkedSet);
        }
        linkedSet.add(handle);
        this._viewMetadata.set(resource, {
          textureId: linkedTextureId,
          textureGeneration,
          key: null,
          descriptor: null,
          invalidated: false,
          isDefaultView: Boolean(meta._isDefaultView),
        });
      }

      this._activeBytes += byteSize;
      this._activeBytesByKind[kind] += byteSize;
      this._vramLiveBytesByKind[vramKind] += byteSize;
      this._vramLiveCountByKind[vramKind] += 1;
      this._totalRegistered += 1;

      this._updateVramPeaks(vramKind);
      this._updatePeakBytes();
      this._checkBudget({ kind, handle, label });

      return resource;
    }

    /**
     * Create, align, and register a WebGPU buffer under `(RESOURCE_KIND.BUFFER, handle)`.
     * Aligns `size` up to a multiple of 4 bytes (`Math.max(4, Math.ceil(size / 4) * 4)`,
     * minimum 16 bytes for `UNIFORM` usage) and tracks `bufferAlignPadCount`.
     *
     * Accepts either:
     * - `createBuffer(handle, { size, usage, mappedAtCreation, label }, options)`
     * - `createBuffer(handle, byteSize, usage, options)`
     *
     * @param {number} handle
     * @param {object | number | bigint} descOrSize
     * @param {number | object} [usageOrOptions]
     * @param {object} [maybeOptions]
     * @returns {object} GPUBuffer
     */
    createBuffer(handle, descOrSize, usageOrOptions, maybeOptions) {
      assertValidHandle(handle);

      let rawSize;
      let usage;
      let mappedAtCreation = false;
      let label = `buffer:${handle}`;
      let opts = {};

      if (descOrSize !== null && typeof descOrSize === 'object') {
        rawSize = descOrSize.size !== undefined ? descOrSize.size : descOrSize.byteSize;
        usage =
          descOrSize.usage !== undefined
            ? descOrSize.usage
            : descOrSize.usageFlags !== undefined
              ? descOrSize.usageFlags
              : descOrSize.usage_flags;
        mappedAtCreation = Boolean(descOrSize.mappedAtCreation);
        if (typeof descOrSize.label === 'string' && descOrSize.label.length > 0) {
          label = descOrSize.label;
        }
        if (usageOrOptions && typeof usageOrOptions === 'object') {
          opts = usageOrOptions;
        }
      } else {
        rawSize = descOrSize;
        usage = usageOrOptions;
        if (maybeOptions && typeof maybeOptions === 'object') {
          opts = maybeOptions;
        }
      }

      if (typeof opts.label === 'string' && opts.label.length > 0) {
        label = opts.label;
      }

      const device = this._resolveDevice(opts.device, 'createBuffer');
      const usageFlags =
        usage !== undefined
          ? Number(usage) >>> 0
          : GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.VERTEX;

      const aligned = alignBufferSize(rawSize, usageFlags);
      if (aligned.padded) {
        this._bufferAlignPadCount += 1;
        this._emitSanitizeWarning(
          'buffer_size_align',
          handle,
          `aligned buffer size ${aligned.requestedSize} -> ${aligned.alignedSize}`
        );
      }

      const createDesc = {
        size: aligned.alignedSize,
        usage: usageFlags,
      };
      if (mappedAtCreation) {
        createDesc.mappedAtCreation = true;
      }
      if (label) {
        createDesc.label = label;
      }

      const buffer = device.createBuffer(createDesc);
      this.register(RESOURCE_KIND.BUFFER, handle, buffer, {
        byteSize: aligned.alignedSize,
        label,
        tag: opts.tag !== undefined ? opts.tag : null,
      });

      this._bufferMeta.set(handle, {
        handle,
        requestedSize: aligned.requestedSize,
        alignedSize: aligned.alignedSize,
        usage: usageFlags,
        isUniform: aligned.isUniform,
        padded: aligned.padded,
        label,
      });

      return buffer;
    }

    /**
     * Upload bytes to a registered buffer handle (or direct `GPUBuffer`) via
     * `device.queue.writeBuffer`, zero-padding non-4-byte-aligned payloads into
     * a 4-byte-aligned copy and safely copying `SharedArrayBuffer` views.
     *
     * @param {number | object} handleOrBuffer
     * @param {number} bufferOffset
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
     * @param {number} [dataOffset]
     * @param {number} [size]
     * @param {{ device?: object }} [options]
     * @returns {boolean}
     */
    writeBuffer(handleOrBuffer, bufferOffset, data, dataOffset, size, options = {}) {
      let buffer = null;
      let handle = -1;
      if (typeof handleOrBuffer === 'number') {
        assertValidHandle(handleOrBuffer);
        handle = handleOrBuffer;
        buffer = this.get(RESOURCE_KIND.BUFFER, handleOrBuffer) || null;
      } else if (handleOrBuffer && typeof handleOrBuffer === 'object') {
        buffer = handleOrBuffer;
      }
      if (!buffer) {
        return false;
      }

      const device = this._resolveDevice(options && options.device, 'writeBuffer');
      const dstOffset = Math.max(0, Math.floor(Number(bufferOffset) || 0));
      const upload = padBufferUploadData(data, dataOffset, size);
      if (upload.alignedByteLength === 0) {
        return true;
      }

      if (upload.padded) {
        this._bufferAlignPadCount += 1;
        this._emitSanitizeWarning(
          'buffer_write_pad',
          handle,
          `padded writeBuffer ${upload.originalByteLength} -> ${upload.alignedByteLength} bytes`
        );
      }

      device.queue.writeBuffer(buffer, dstOffset, upload.bytes);
      return true;
    }

    /**
     * Create, sanitize, and register a WebGPU texture under
     * `(RESOURCE_KIND.TEXTURE, handle)` (and its default view under
     * `(RESOURCE_KIND.TEXTURE_VIEW, handle)` when `texture.createView` exists).
     *
     * @param {number} handle
     * @param {object} descriptor
     * @param {{
     *   device?: object,
     *   useV1FormatTable?: boolean,
     *   isTexture2?: boolean,
     *   createView?: boolean,
     *   byteSize?: number,
     *   label?: string,
     *   tag?: unknown,
     * }} [options]
     * @returns {object} GPUTexture
     */
    createTexture(handle, descriptor, options = {}) {
      assertValidHandle(handle);
      const device = this._resolveDevice(options && options.device, 'createTexture');
      const includeSrgbSiblingViewFormat = Boolean(
        this.includeSrgbSiblingViewFormat ||
          (options && (options.includeSrgbSiblingViewFormat || options.includeSrgbSibling))
      );
      const sanitized = sanitizeTextureDescriptor(descriptor, device, {
        useV1FormatTable: Boolean(options && options.useV1FormatTable),
        includeSrgbSiblingViewFormat,
      });

      if (sanitized.bcFormatFallback) {
        this._bcFormatFallbackCount += 1;
        this._emitSanitizeWarning(
          'bc_format_fallback',
          handle,
          `${sanitized.originalFormat} -> ${sanitized.effectiveFormat}`
        );
      }
      if (sanitized.bcDimensionAligned) {
        this._bcDimensionAlignCount += 1;
        this._emitSanitizeWarning(
          'bc_dimension_align',
          handle,
          `aligned BC dimensions to ${sanitized.descriptor.size.width}x${sanitized.descriptor.size.height}`
        );
      }
      if (sanitized.mipClamped) {
        this._mipClampCount += 1;
        this._emitSanitizeWarning(
          'mip_clamp',
          handle,
          `clamped mipLevelCount to ${sanitized.descriptor.mipLevelCount}`
        );
      }
      if (sanitized.sampleClamped) {
        this._sampleClampCount += 1;
        this._emitSanitizeWarning(
          'sample_clamp',
          handle,
          `clamped sampleCount to ${sanitized.descriptor.sampleCount}`
        );
      }
      if (sanitized.droppedViewFormats && sanitized.droppedViewFormats.length > 0) {
        this._emitSanitizeWarning(
          'view_format_drop',
          handle,
          `dropped incompatible viewFormats [${sanitized.droppedViewFormats.join(', ')}] for ${sanitized.effectiveFormat}`
        );
      }

      const texture = device.createTexture(sanitized.descriptor);
      const label =
        options && typeof options.label === 'string' && options.label.length > 0
          ? options.label
          : sanitized.descriptor.label || `texture:${handle}`;
      const tag = options && options.tag !== undefined ? options.tag : null;
      const estimatedBytes =
        options && options.byteSize !== undefined
          ? options.byteSize
          : estimateTextureBytes(
              sanitized.descriptor.size.width,
              sanitized.descriptor.size.height,
              sanitized.descriptor.size.depthOrArrayLayers,
              sanitized.descriptor.mipLevelCount,
              sanitized.effectiveFormat,
              sanitized.descriptor.dimension,
              sanitized.descriptor.sampleCount
            );

      const vramKind =
        options && typeof options.vramKind === 'string'
          ? resolveVramKind(RESOURCE_KIND.TEXTURE, texture, options)
          : (options && (options.isRenderTarget === true || options.renderTarget === true)) ||
              (descriptor &&
                typeof descriptor === 'object' &&
                (descriptor.isRenderTarget === true || descriptor.renderTarget === true)) ||
              (sanitized.descriptor.usage & GPU_TEXTURE_USAGE.RENDER_ATTACHMENT) !== 0
            ? VRAM_KIND.RENDER_TARGET
            : VRAM_KIND.TEXTURE;

      // Registering on an existing handle automatically retires the prior texture,
      // increments textureGeneration & textureRecreations, and invalidates all
      // cached/linked views for `handle`.
      this.register(RESOURCE_KIND.TEXTURE, handle, texture, {
        byteSize: estimatedBytes,
        label,
        tag,
        vramKind,
        _skipAutoTextureMeta: true,
      });

      const textureGeneration = this._activeTextureGeneration.get(handle) || 1;

      this._textureMeta.set(handle, {
        handle,
        width: sanitized.descriptor.size.width,
        height: sanitized.descriptor.size.height,
        depthOrArrayLayers: sanitized.descriptor.size.depthOrArrayLayers,
        mipLevelCount: sanitized.descriptor.mipLevelCount,
        sampleCount: sanitized.descriptor.sampleCount,
        dimension: sanitized.descriptor.dimension,
        format: sanitized.effectiveFormat,
        originalFormat: sanitized.originalFormat,
        usage: sanitized.descriptor.usage,
        viewFormats: sanitized.descriptor.viewFormats
          ? sanitized.descriptor.viewFormats.slice()
          : [],
        siblingViewFormat: sanitized.siblingViewFormat,
        textureGeneration,
        bcFormatFallback: sanitized.bcFormatFallback,
        bcDimensionAligned: sanitized.bcDimensionAligned,
        mipClamped: sanitized.mipClamped,
        sampleClamped: sanitized.sampleClamped,
        isTexture2: Boolean(options && options.isTexture2),
        byteSize: estimatedBytes,
      });
      this._recordTextureFormatTelemetry(sanitized.effectiveFormat);

      if (
        (!options || options.createView !== false) &&
        texture &&
        typeof texture.createView === 'function'
      ) {
        const textureView = texture.createView();
        this.register(RESOURCE_KIND.TEXTURE_VIEW, handle, textureView, {
          byteSize: 0,
          label: `textureView:${handle}`,
          tag,
          textureId: handle,
          _isDefaultView: true,
        });
      } else {
        this.retire(RESOURCE_KIND.TEXTURE_VIEW, handle);
      }

      return texture;
    }

    /**
     * Create a texture using ABI v2 `CreateTexture2` metadata (opcode 35).
     * Accepts either:
     * - `createTexture2(handle, descriptor, options)` (structured descriptor with v2 format code/name)
     * - `createTexture2(dataView, payloadOffset, payloadBytes, options)` (raw 40-byte payload)
     *
     * @param {number | DataView} handleOrView
     * @param {object | number} [descriptorOrOffset]
     * @param {object | number} [optionsOrPayloadBytes]
     * @param {object} [maybeOptions]
     * @returns {object}
     */
    createTexture2(handleOrView, descriptorOrOffset, optionsOrPayloadBytes, maybeOptions) {
      if (handleOrView instanceof DataView) {
        const offset = typeof descriptorOrOffset === 'number' ? descriptorOrOffset : 0;
        const payloadBytes =
          typeof optionsOrPayloadBytes === 'number' ? optionsOrPayloadBytes : undefined;
        const opts =
          typeof optionsOrPayloadBytes === 'object' && optionsOrPayloadBytes !== null
            ? optionsOrPayloadBytes
            : maybeOptions || {};
        return this.createTexture2FromPacket(handleOrView, offset, payloadBytes, opts);
      }

      const opts =
        optionsOrPayloadBytes && typeof optionsOrPayloadBytes === 'object'
          ? optionsOrPayloadBytes
          : {};
      return this.createTexture(handleOrView, descriptorOrOffset, {
        ...opts,
        useV1FormatTable: false,
        isTexture2: true,
      });
    }

    /**
     * Decode and create a texture from a binary ABI v2 `CreateTexture2` payload
     * (40 bytes, `GpuCreateTexture2Payload`).
     *
     * @param {DataView} view
     * @param {number} [payloadOffset=0]
     * @param {number} [payloadBytes]
     * @param {object} [options]
     * @returns {{ id: number, texture?: object, view?: object | null, meta?: object | null, error?: string }}
     */
    createTexture2FromPacket(view, payloadOffset = 0, payloadBytes, options = {}) {
      const decoded = decodeCreateTexture2(view, payloadOffset, payloadBytes);
      if (decoded.error) {
        return { id: decoded.id, error: decoded.error };
      }

      try {
        const texture = this.createTexture(decoded.id, decoded.descriptor, {
          ...(options || {}),
          useV1FormatTable: false,
          isTexture2: true,
        });
        return {
          id: decoded.id,
          texture,
          view: this._peek(RESOURCE_KIND.TEXTURE_VIEW, decoded.id) || null,
          meta: this.getTextureMeta(decoded.id) || null,
        };
      } catch (err) {
        return {
          id: decoded.id,
          error: `CreateTexture2 ${decoded.id}: ${String(err && err.message ? err.message : err)}`,
        };
      }
    }

    /**
     * Look up sanitized creation metadata for a buffer created via `createBuffer`.
     *
     * @param {number} handle
     * @returns {object | undefined}
     */
    getBufferMeta(handle) {
      assertValidHandle(handle);
      const meta = this._bufferMeta.get(handle);
      return meta ? { ...meta } : undefined;
    }

    /**
     * Look up sanitized creation metadata for a texture created via `createTexture` / `createTexture2`
     * or registered with descriptor/format metadata via `register(RESOURCE_KIND.TEXTURE, ...)`.
     *
     * @param {number} handle
     * @returns {object | undefined}
     */
    getTextureMeta(handle) {
      assertValidHandle(handle);
      const meta = this._textureMeta.get(handle);
      return meta ? { ...meta, viewFormats: meta.viewFormats.slice() } : undefined;
    }

    /**
     * Return the current active generation counter for a registered texture ID,
     * or `0` if no active texture is currently registered under `textureId`.
     * Generation counters increase monotonically across replacements, retirements,
     * and `destroyAll()` so stale draw-call handles can never alias a new texture.
     *
     * @param {number} textureId
     * @returns {number}
     */
    getTextureGeneration(textureId) {
      assertValidHandle(textureId);
      return this._activeTextureGeneration.get(textureId) || 0;
    }

    /**
     * Return whether `generation` matches the live texture currently registered
     * under `textureId`.
     *
     * @param {number} textureId
     * @param {number} generation
     * @returns {boolean}
     */
    isTextureGenerationCurrent(textureId, generation) {
      assertValidHandle(textureId);
      if (typeof generation !== 'number' || !Number.isSafeInteger(generation) || generation < 1) {
        return false;
      }
      return this._activeTextureGeneration.get(textureId) === generation;
    }

    /**
     * Look up or lazily create a cached `GPUTextureView` for `textureId`, keyed by
     * `(textureId, format, dimension, baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount, aspect)`.
     *
     * Automatically validates sibling sRGB/linear `viewFormats` compatibility,
     * guards against stale generations (e.g. after 1080p <-> 4K recreation or
     * texture destruction), and updates `viewCacheHits` / `viewCacheMisses`.
     *
     * Returns `undefined` when no live texture is registered under `textureId`.
     *
     * @param {number} textureId
     * @param {{
     *   format?: string | number,
     *   dimension?: '1d' | '2d' | '2d-array' | 'cube' | 'cube-array' | '3d',
     *   baseMipLevel?: number,
     *   mipLevelCount?: number,
     *   baseArrayLayer?: number,
     *   arrayLayerCount?: number,
     *   aspect?: 'all' | 'stencil-only' | 'depth-only',
     *   label?: string,
     * }} [viewDesc]
     * @param {{ allowUndeclaredSibling?: boolean }} [options]
     * @returns {object | undefined}
     */
    getTextureView(textureId, viewDesc = {}, options = {}) {
      assertValidHandle(textureId);
      const texture = this.get(RESOURCE_KIND.TEXTURE, textureId);
      const currentGeneration = this._activeTextureGeneration.get(textureId) || 0;
      if (!texture || currentGeneration === 0) {
        if (this._textureViewCache.has(textureId)) {
          this._invalidateTextureViews(textureId, { retireLinkedTableViews: true });
        }
        return undefined;
      }

      const texMeta = this._textureMeta.get(textureId) || null;
      const normalized = normalizeTextureViewDescriptor(textureId, viewDesc, texMeta);

      if (texMeta && texMeta.format && normalized.format) {
        if (normalized.aspect === 'all') {
          if (normalized.format !== texMeta.format) {
            if (!isViewFormatCompatible(texMeta.format, normalized.format, 'all')) {
              throw new RangeError(
                `View format "${normalized.format}" is not compatible with texture ${textureId} format "${texMeta.format}"`
              );
            }
            const declared =
              Array.isArray(texMeta.viewFormats) &&
              texMeta.viewFormats.includes(normalized.format);
            if (!declared && !(options && options.allowUndeclaredSibling)) {
              throw new RangeError(
                `Texture ${textureId} (${texMeta.format}) does not declare viewFormat "${normalized.format}" in viewFormats`
              );
            }
          }
        } else if (
          !isViewFormatCompatible(texMeta.format, normalized.format, normalized.aspect)
        ) {
          throw new RangeError(
            `View format "${normalized.format}" with aspect "${normalized.aspect}" is not compatible with texture ${textureId} format "${texMeta.format}"`
          );
        }
      }

      const cacheKey = normalized.key;
      let perTexCache = this._textureViewCache.get(textureId);
      if (perTexCache && perTexCache.has(cacheKey)) {
        const cached = perTexCache.get(cacheKey);
        if (cached && cached.textureGeneration === currentGeneration) {
          // Maintain LRU order so frequently bound default/subresource views stay resident.
          if (perTexCache.size > 1) {
            perTexCache.delete(cacheKey);
            perTexCache.set(cacheKey, cached);
          }
          this._viewCacheHits += 1;
          return cached.view;
        }
        // Defense-in-depth: evict any generation-mismatched entry.
        if (cached && cached.view && typeof cached.view === 'object') {
          const vm = this._viewMetadata.get(cached.view);
          if (vm) {
            vm.invalidated = true;
          }
        }
        perTexCache.delete(cacheKey);
        this._staleViewsInvalidated += 1;
        this._viewCacheInvalidations += 1;
      }

      this._viewCacheMisses += 1;

      if (typeof texture.createView !== 'function') {
        throw new TypeError(`Texture ${textureId} does not implement createView()`);
      }

      const view = texture.createView(normalized.gpuViewDescriptor);
      if (view === null || typeof view !== 'object') {
        throw new TypeError(`Texture ${textureId} createView() returned a non-object value`);
      }

      if (!perTexCache) {
        perTexCache = new Map();
        this._textureViewCache.set(textureId, perTexCache);
      }

      // Enforce bounded per-texture view cache cap (LRU eviction).
      while (perTexCache.size >= this.maxCachedViewsPerTexture) {
        const oldestKey = perTexCache.keys().next().value;
        if (oldestKey === undefined) {
          break;
        }
        perTexCache.delete(oldestKey);
        this._viewCacheEvictions += 1;
      }

      const isDefaultView =
        viewDesc === undefined ||
        viewDesc === null ||
        (viewDesc.format === undefined &&
          viewDesc.dimension === undefined &&
          viewDesc.baseMipLevel === undefined &&
          viewDesc.mipLevelCount === undefined &&
          viewDesc.baseArrayLayer === undefined &&
          viewDesc.arrayLayerCount === undefined &&
          viewDesc.aspect === undefined);

      const descriptorSnapshot = {
        format: normalized.format,
        dimension: normalized.dimension,
        baseMipLevel: normalized.baseMipLevel,
        mipLevelCount: normalized.mipLevelCount,
        baseArrayLayer: normalized.baseArrayLayer,
        arrayLayerCount: normalized.arrayLayerCount,
        aspect: normalized.aspect,
      };

      perTexCache.set(cacheKey, {
        view,
        key: cacheKey,
        textureId,
        textureGeneration: currentGeneration,
        descriptor: descriptorSnapshot,
        createdAtFrame: this._currentFrame,
        isDefaultView,
      });

      this._viewMetadata.set(view, {
        textureId,
        textureGeneration: currentGeneration,
        key: cacheKey,
        descriptor: descriptorSnapshot,
        invalidated: false,
        isDefaultView,
      });

      return view;
    }

    /**
     * Look up or lazily create and cache the default `GPUTextureView` for
     * `textureId`, sharing the canonical default cache entry with
     * `getTextureView(textureId)`.
     *
     * @param {number} textureId
     * @param {{ allowUndeclaredSibling?: boolean }} [options]
     * @returns {object | undefined}
     */
    getDefaultTextureView(textureId, options = {}) {
      return this.getTextureView(textureId, undefined, options);
    }

    /**
     * Alias for `getTextureView(textureId, viewDesc, options)`.
     *
     * @param {number} textureId
     * @param {object} [viewDesc]
     * @param {object} [options]
     * @returns {object | undefined}
     */
    getOrCreateTextureView(textureId, viewDesc = {}, options = {}) {
      return this.getTextureView(textureId, viewDesc, options);
    }

    /**
     * Explicitly invalidate all cached and linked `GPUTextureView` handles
     * associated with `textureId`, incrementing `staleViewsInvalidated`.
     *
     * @param {number} textureId
     * @returns {number} Number of invalidated views
     */
    invalidateTextureViews(textureId) {
      assertValidHandle(textureId);
      return this._invalidateTextureViews(textureId, { retireLinkedTableViews: true });
    }

    /**
     * Check whether a `GPUTextureView` returned by this registry (via
     * `getTextureView`, `createTexture`, or linked `register(TEXTURE_VIEW, ...)`)
     * is still valid for the currently active texture generation.
     *
     * @param {object | null | undefined} view
     * @returns {boolean}
     */
    isTextureViewValid(view) {
      if (!view || typeof view !== 'object') {
        return false;
      }
      const info = this._viewMetadata.get(view);
      if (!info || info.invalidated || info.textureGeneration < 1) {
        return false;
      }
      return this._activeTextureGeneration.get(info.textureId) === info.textureGeneration;
    }

    /**
     * Alias for `isTextureViewValid(view)`.
     *
     * @param {object | null | undefined} view
     * @returns {boolean}
     */
    isViewValid(view) {
      return this.isTextureViewValid(view);
    }

    /**
     * Return generation & validity metadata for a tracked `GPUTextureView`,
     * or `null` if the view is not tracked by this registry.
     *
     * @param {object | null | undefined} view
     * @returns {{
     *   textureId: number,
     *   textureGeneration: number,
     *   currentGeneration: number,
     *   valid: boolean,
     *   key: string | null,
     *   descriptor: object | null,
     *   isDefaultView: boolean,
     * } | null}
     */
    getTextureViewInfo(view) {
      if (!view || typeof view !== 'object') {
        return null;
      }
      const info = this._viewMetadata.get(view);
      if (!info) {
        return null;
      }
      const currentGeneration = this._activeTextureGeneration.get(info.textureId) || 0;
      const valid =
        !info.invalidated &&
        info.textureGeneration >= 1 &&
        currentGeneration === info.textureGeneration;
      return {
        textureId: info.textureId,
        textureGeneration: info.textureGeneration,
        currentGeneration,
        valid,
        key: info.key,
        descriptor: info.descriptor ? { ...info.descriptor } : null,
        isDefaultView: info.isDefaultView,
      };
    }

    /**
     * Return sanitized snapshots of all currently cached views for `textureId`.
     *
     * @param {number} textureId
     * @returns {Array<{ key: string, textureId: number, textureGeneration: number, descriptor: object, createdAtFrame: number }>}
     */
    getCachedTextureViews(textureId) {
      assertValidHandle(textureId);
      const perTexCache = this._textureViewCache.get(textureId);
      if (!perTexCache) {
        return [];
      }
      const result = [];
      for (const cached of perTexCache.values()) {
        result.push({
          key: cached.key,
          textureId: cached.textureId,
          textureGeneration: cached.textureGeneration,
          descriptor: { ...cached.descriptor },
          createdAtFrame: cached.createdAtFrame,
        });
      }
      return result;
    }

    /**
     * Look up an active resource by `(kind, handle)` using an O(1) zero-allocation
     * direct-indexed fast path for numeric handles `< FAST_HANDLE_LIMIT`.
     * Automatically detects and evicts externally destroyed resources, and records
     * `staleLookupMisses` when querying a previously retired or destroyed handle.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {object | undefined}
     */
    get(kind, handle) {
      const slots = this._fastSlotsByKind[kind];
      if (slots === undefined) {
        assertValidKind(kind);
      }
      if (typeof handle === 'number' && handle >= 0) {
        if (handle < FAST_HANDLE_LIMIT && (handle | 0) === handle) {
          const res = slots[handle];
          if (res !== undefined) {
            if (
              res.destroyed === true ||
              (this._hasDestroyedResources && this._destroyedResources.has(res))
            ) {
              this._evictDestroyedEntry(kind, handle);
              this._staleLookupMisses += 1;
              return undefined;
            }
            this._fastHandleHits += 1;
            return res;
          }
          if (this._retiredHandlesByKind[kind].has(handle)) {
            this._staleLookupMisses += 1;
          }
          return undefined;
        }
        if (Number.isSafeInteger(handle)) {
          const entry = this._tablesByKind[kind].get(handle);
          if (entry !== undefined) {
            const res = entry.resource;
            if (
              res &&
              (res.destroyed === true ||
                (this._hasDestroyedResources && this._destroyedResources.has(res)))
            ) {
              this._evictDestroyedEntry(kind, handle);
              this._staleLookupMisses += 1;
              return undefined;
            }
            this._fastHandleOverflowHits += 1;
            return res;
          }
          if (this._retiredHandlesByKind[kind].has(handle)) {
            this._staleLookupMisses += 1;
          }
          return undefined;
        }
      }
      assertValidHandle(handle);
      return undefined;
    }

    /**
     * O(1) zero-allocation, exception-free handle lookup for hot draw/bind paths.
     * Avoids string formatting, error allocation, and Map hashing for dense
     * numeric handles `< FAST_HANDLE_LIMIT`, falling back to O(1) Map lookup for
     * larger safe-integer handles. Returns `undefined` for missing, invalid, or
     * stale/destroyed handles.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {object | undefined}
     */
    getFast(kind, handle) {
      const slots = this._fastSlotsByKind[kind];
      if (slots === undefined || typeof handle !== 'number' || handle < 0) {
        return undefined;
      }
      if (handle < FAST_HANDLE_LIMIT && (handle | 0) === handle) {
        const res = slots[handle];
        if (res !== undefined) {
          if (
            res.destroyed === true ||
            (this._hasDestroyedResources && this._destroyedResources.has(res))
          ) {
            this._evictDestroyedEntry(kind, handle);
            this._staleLookupMisses += 1;
            return undefined;
          }
          this._fastHandleHits += 1;
          return res;
        }
        if (this._retiredHandlesByKind[kind].has(handle)) {
          this._staleLookupMisses += 1;
        }
        return undefined;
      }
      if (Number.isSafeInteger(handle)) {
        const entry = this._tablesByKind[kind].get(handle);
        if (entry !== undefined) {
          const res = entry.resource;
          if (
            res &&
            (res.destroyed === true ||
              (this._hasDestroyedResources && this._destroyedResources.has(res)))
          ) {
            this._evictDestroyedEntry(kind, handle);
            this._staleLookupMisses += 1;
            return undefined;
          }
          this._fastHandleOverflowHits += 1;
          return res;
        }
        if (this._retiredHandlesByKind[kind].has(handle)) {
          this._staleLookupMisses += 1;
        }
      }
      return undefined;
    }

    _getTypedFast(kind, slots, table, retiredSet, handle) {
      if (typeof handle !== 'number' || handle < 0) {
        return undefined;
      }
      if (handle < FAST_HANDLE_LIMIT && (handle | 0) === handle) {
        const res = slots[handle];
        if (res !== undefined) {
          if (
            res.destroyed === true ||
            (this._hasDestroyedResources && this._destroyedResources.has(res))
          ) {
            this._evictDestroyedEntry(kind, handle);
            this._staleLookupMisses += 1;
            return undefined;
          }
          this._fastHandleHits += 1;
          return res;
        }
        if (retiredSet.has(handle)) {
          this._staleLookupMisses += 1;
        }
        return undefined;
      }
      if (Number.isSafeInteger(handle)) {
        const entry = table.get(handle);
        if (entry !== undefined) {
          const res = entry.resource;
          if (
            res &&
            (res.destroyed === true ||
              (this._hasDestroyedResources && this._destroyedResources.has(res)))
          ) {
            this._evictDestroyedEntry(kind, handle);
            this._staleLookupMisses += 1;
            return undefined;
          }
          this._fastHandleOverflowHits += 1;
          return res;
        }
        if (retiredSet.has(handle)) {
          this._staleLookupMisses += 1;
        }
      }
      return undefined;
    }

    /** O(1) zero-alloc GPUBuffer lookup by numeric handle. */
    getBufferFast(handle) {
      return this._getTypedFast(
        RESOURCE_KIND.BUFFER,
        this._fastBuffers,
        this._bufferTable,
        this._retiredBuffers,
        handle
      );
    }

    /** O(1) zero-alloc GPUTexture lookup by numeric handle. */
    getTextureFast(handle) {
      return this._getTypedFast(
        RESOURCE_KIND.TEXTURE,
        this._fastTextures,
        this._textureTable,
        this._retiredTextures,
        handle
      );
    }

    /** O(1) zero-alloc GPUTextureView lookup by numeric handle. */
    getTextureViewFast(handle) {
      return this._getTypedFast(
        RESOURCE_KIND.TEXTURE_VIEW,
        this._fastTextureViews,
        this._textureViewTable,
        this._retiredTextureViews,
        handle
      );
    }

    /** O(1) zero-alloc GPUSampler lookup by numeric handle. */
    getSamplerFast(handle) {
      return this._getTypedFast(
        RESOURCE_KIND.SAMPLER,
        this._fastSamplers,
        this._samplerTable,
        this._retiredSamplers,
        handle
      );
    }

    /** O(1) zero-alloc GPUBindGroup lookup by numeric handle. */
    getBindGroupFast(handle) {
      return this._getTypedFast(
        RESOURCE_KIND.BIND_GROUP,
        this._fastBindGroups,
        this._bindGroupTable,
        this._retiredBindGroups,
        handle
      );
    }

    /** O(1) zero-alloc GPURenderPipeline lookup by numeric handle. */
    getRenderPipelineFast(handle) {
      return this._getTypedFast(
        RESOURCE_KIND.RENDER_PIPELINE,
        this._fastRenderPipelines,
        this._renderPipelineTable,
        this._retiredRenderPipelines,
        handle
      );
    }

    /** O(1) zero-alloc GPUComputePipeline lookup by numeric handle. */
    getComputePipelineFast(handle) {
      return this._getTypedFast(
        RESOURCE_KIND.COMPUTE_PIPELINE,
        this._fastComputePipelines,
        this._computePipelineTable,
        this._retiredComputePipelines,
        handle
      );
    }

    /**
     * Look up an active resource entry snapshot (including metadata) by `(kind, handle)`.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {{ kind: string, handle: number, resource: object, byteSize: number, label: string, tag: unknown, registeredAtFrame: number } | undefined}
     */
    getEntry(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);
      const entry = this._tablesByKind[kind].get(handle);
      if (!entry) {
        if (this._retiredHandlesByKind[kind].has(handle)) {
          this._staleLookupMisses += 1;
        }
        return undefined;
      }
      const res = entry.resource;
      if (
        res &&
        (res.destroyed === true ||
          (this._hasDestroyedResources && this._destroyedResources.has(res)))
      ) {
        this._evictDestroyedEntry(kind, handle);
        this._staleLookupMisses += 1;
        return undefined;
      }
      return snapshotEntry(entry);
    }

    /**
     * Return whether an active resource exists for `(kind, handle)`.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {boolean}
     */
    has(kind, handle) {
      const slots = this._fastSlotsByKind[kind];
      if (slots === undefined) {
        assertValidKind(kind);
      }
      if (typeof handle === 'number' && handle >= 0) {
        if (handle < FAST_HANDLE_LIMIT && (handle | 0) === handle) {
          const res = slots[handle];
          return (
            res !== undefined &&
            res.destroyed !== true &&
            (!this._hasDestroyedResources || !this._destroyedResources.has(res))
          );
        }
        if (Number.isSafeInteger(handle)) {
          const entry = this._tablesByKind[kind].get(handle);
          if (!entry) {
            return false;
          }
          const res = entry.resource;
          return (
            res !== null &&
            res !== undefined &&
            res.destroyed !== true &&
            (!this._hasDestroyedResources || !this._destroyedResources.has(res))
          );
        }
      }
      assertValidHandle(handle);
      return false;
    }

    /**
     * O(1) zero-allocation, exception-free active resource existence check.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {boolean}
     */
    hasFast(kind, handle) {
      const slots = this._fastSlotsByKind[kind];
      if (slots === undefined || typeof handle !== 'number' || handle < 0) {
        return false;
      }
      if (handle < FAST_HANDLE_LIMIT && (handle | 0) === handle) {
        const res = slots[handle];
        return (
          res !== undefined &&
          res.destroyed !== true &&
          (!this._hasDestroyedResources || !this._destroyedResources.has(res))
        );
      }
      if (Number.isSafeInteger(handle)) {
        const entry = this._tablesByKind[kind].get(handle);
        if (!entry) {
          return false;
        }
        const res = entry.resource;
        return (
          res !== null &&
          res !== undefined &&
          res.destroyed !== true &&
          (!this._hasDestroyedResources || !this._destroyedResources.has(res))
        );
      }
      return false;
    }

    /**
     * List active resource entry snapshots, optionally filtered to a single `kind`.
     *
     * @param {string} [kind]
     * @returns {Array<{ kind: string, handle: number, resource: object, byteSize: number, label: string, tag: unknown, registeredAtFrame: number }>}
     */
    list(kind) {
      if (kind !== undefined) {
        assertValidKind(kind);
        const table = this._tables.get(kind);
        const result = [];
        for (const entry of table.values()) {
          result.push(snapshotEntry(entry));
        }
        return result;
      }

      const result = [];
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const table = this._tables.get(RESOURCE_KINDS[i]);
        for (const entry of table.values()) {
          result.push(snapshotEntry(entry));
        }
      }
      return result;
    }

    /**
     * Remove `(kind, handle)` from the active table immediately and enqueue it
     * for deferred destruction after `framesInFlight` frames or once its frame's
     * GPU fence resolves.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {boolean} True if an active resource was retired, false otherwise
     */
    retire(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);

      const table = this._tablesByKind[kind];
      const entry = table.get(handle);
      if (!entry) {
        return false;
      }

      table.delete(handle);
      this._clearFastSlot(kind, handle, table);
      this._markHandleStale(kind, handle);
      this._cleanupKindMetadata(kind, handle, entry);

      const byteSize = sanitizeNonNegativeCounter(entry.byteSize);
      const vramKind = entry.vramKind || VRAM_KIND.OTHER;

      this._activeBytes = Math.max(0, this._activeBytes - byteSize);
      this._activeBytesByKind[kind] = Math.max(0, this._activeBytesByKind[kind] - byteSize);
      this._pendingRetireBytes += byteSize;

      this._vramLiveBytesByKind[vramKind] = Math.max(
        0,
        this._vramLiveBytesByKind[vramKind] - byteSize
      );
      this._vramLiveCountByKind[vramKind] = Math.max(
        0,
        this._vramLiveCountByKind[vramKind] - 1
      );
      this._vramPendingBytesByKind[vramKind] += byteSize;

      entry.retiredAtFrame = this._currentFrame;
      entry.sealed = false;
      entry.hasFence = false;
      entry.fenceDone = false;
      entry.fenceFailed = false;

      this._pendingRetire.push(entry);
      this._totalRetired += 1;
      return true;
    }

    /**
     * Immediately remove and destroy a single active resource `(kind, handle)`,
     * reclaiming its per-kind VRAM bytes and recording eviction telemetry
     * (`evictedCount`, `evictedBytes`) without waiting for `framesInFlight`.
     *
     * @param {string} kind
     * @param {number} handle
     * @returns {boolean} True if an active resource was destroyed, false otherwise
     */
    destroy(kind, handle) {
      assertValidKind(kind);
      assertValidHandle(handle);

      const table = this._tablesByKind[kind];
      const entry = table.get(handle);
      if (!entry) {
        return false;
      }

      table.delete(handle);
      this._clearFastSlot(kind, handle, table);
      this._markHandleStale(kind, handle);
      this._cleanupKindMetadata(kind, handle, entry);

      const byteSize = sanitizeNonNegativeCounter(entry.byteSize);
      const vramKind = entry.vramKind || VRAM_KIND.OTHER;

      this._activeBytes = Math.max(0, this._activeBytes - byteSize);
      this._activeBytesByKind[kind] = Math.max(0, this._activeBytesByKind[kind] - byteSize);
      this._vramLiveBytesByKind[vramKind] = Math.max(
        0,
        this._vramLiveBytesByKind[vramKind] - byteSize
      );
      this._vramLiveCountByKind[vramKind] = Math.max(
        0,
        this._vramLiveCountByKind[vramKind] - 1
      );

      if (this._destroyEntry(entry)) {
        this._evictedCount += 1;
        this._evictedBytes += byteSize;
      }

      this._checkBudget(null);
      return true;
    }

    /**
     * Proactively sweep all active resource tables and evict any entries whose
     * underlying WebGPU resource has been destroyed externally (`resource.destroyed === true`).
     * Updates per-kind VRAM accounting, `evictedCount`, and `evictedBytes`.
     *
     * @returns {{ evictedCount: number, evictedBytes: number }}
     */
    evictStale() {
      let evictedCount = 0;
      let evictedBytes = 0;

      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const kind = RESOURCE_KINDS[i];
        const table = this._tablesByKind[kind];
        if (table.size === 0) {
          continue;
        }
        const staleHandles = [];
        for (const [handle, entry] of table.entries()) {
          const res = entry.resource;
          if (
            !res ||
            res.destroyed === true ||
            (this._hasDestroyedResources && this._destroyedResources.has(res))
          ) {
            staleHandles.push(handle);
          }
        }
        for (let j = 0; j < staleHandles.length; j += 1) {
          const freed = this._evictDestroyedEntry(kind, staleHandles[j]);
          evictedCount += 1;
          evictedBytes += freed;
        }
      }

      return { evictedCount, evictedBytes };
    }

    /**
     * Convenience helper that retires a numeric `handle` from every resource kind
     * where it is currently registered (mirrors opcode DESTROY_RESOURCE).
     *
     * @param {number} handle
     * @returns {number} Number of active resources retired
     */
    retireHandle(handle) {
      assertValidHandle(handle);
      let count = 0;
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        if (this.retire(RESOURCE_KINDS[i], handle)) {
          count += 1;
        }
      }
      return count;
    }

    /**
     * Advance the frame epoch and collect any deferred resources whose frame
     * latency or GPU fence has completed.
     *
     * @param {number} [frameIndex]
     * @returns {{ frameIndex: number, destroyedCount: number, freedBytes: number, remainingCount: number, pendingRetireBytes: number }}
     */
    beginFrame(frameIndex) {
      // Seal any unsealed retirements from the prior frame as epoch-based so a
      // future endFrame(fence) in this frame does not retroactively attach to them.
      this._sealUnsealedRetirements();

      let nextFrame;
      if (frameIndex === undefined) {
        nextFrame = this._currentFrame + 1;
      } else {
        assertNonNegativeInteger(frameIndex, 'frameIndex');
        if (frameIndex < this._currentFrame) {
          throw new RangeError(
            `frameIndex (${frameIndex}) cannot move backwards from currentFrame (${this._currentFrame})`
          );
        }
        nextFrame = frameIndex;
      }

      this._currentFrame = nextFrame;
      const collected = this.collect(false);

      return {
        frameIndex: this._currentFrame,
        destroyedCount: collected.destroyedCount,
        freedBytes: collected.freedBytes,
        remainingCount: collected.remainingCount,
        pendingRetireBytes: collected.pendingRetireBytes,
      };
    }

    /**
     * Seal all resources retired during the current frame and optionally bind
     * them to a GPU work-done fence promise (e.g. `device.queue.onSubmittedWorkDone()`).
     *
     * When a fence promise is supplied:
     *  - Resolving the fence allows early collection on the next `collect()` /
     *    `beginFrame()`, even before `framesInFlight` frames elapse.
     *  - While the fence remains pending, collection of those entries is gated
     *    until the fence resolves (unless `collect(true)` or `destroyAll()` is called).
     *  - If the fence rejects (e.g. device lost), those entries fall back to
     *    frame-epoch expiration instead of leaking or causing unhandled rejections.
     *
     * @param {Promise<unknown> | null} [fencePromise]
     * @returns {{ frameIndex: number, sealedCount: number, pendingCount: number, pendingRetireBytes: number }}
     */
    endFrame(fencePromise) {
      if (
        fencePromise !== undefined &&
        fencePromise !== null &&
        typeof fencePromise.then !== 'function'
      ) {
        throw new TypeError('endFrame expects fencePromise to be a Promise/thenable, null, or undefined');
      }

      const batch = this._sealUnsealedRetirements();

      if (fencePromise && typeof fencePromise.then === 'function') {
        const generation = this._generation;
        for (let i = 0; i < batch.length; i += 1) {
          batch[i].hasFence = true;
          batch[i].fenceDone = false;
          batch[i].fenceFailed = false;
        }

        const tracked = Promise.resolve(fencePromise).then(
          () => {
            if (generation !== this._generation) {
              return;
            }
            for (let i = 0; i < batch.length; i += 1) {
              if (!batch[i].destroyed) {
                batch[i].fenceDone = true;
              }
            }
          },
          () => {
            if (generation !== this._generation) {
              return;
            }
            for (let i = 0; i < batch.length; i += 1) {
              if (!batch[i].destroyed) {
                batch[i].fenceFailed = true;
              }
            }
          }
        );

        this._pendingFences.add(tracked);
        tracked.finally(() => {
          this._pendingFences.delete(tracked);
        });
      }

      return {
        frameIndex: this._currentFrame,
        sealedCount: batch.length,
        pendingCount: this._pendingRetire.length,
        pendingRetireBytes: this._pendingRetireBytes,
      };
    }

    /**
     * Destroy eligible retired resources and reclaim their pending-retirement VRAM accounting.
     *
     * @param {boolean} [force=false] When true, destroys all pending retired resources immediately regardless of frame epoch or pending fences.
     * @returns {{ destroyedCount: number, freedBytes: number, remainingCount: number, pendingRetireBytes: number }}
     */
    collect(force = false) {
      const forceAll = Boolean(force);
      let destroyedCount = 0;
      let freedBytes = 0;
      const remaining = [];

      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        const entry = this._pendingRetire[i];
        const epochElapsed = this._currentFrame - entry.retiredAtFrame >= this.framesInFlight;

        let eligible;
        if (forceAll) {
          eligible = true;
        } else if (entry.hasFence) {
          eligible = entry.fenceDone || (entry.fenceFailed && epochElapsed);
        } else {
          eligible = epochElapsed;
        }

        if (eligible) {
          const byteSize = sanitizeNonNegativeCounter(entry.byteSize);
          const vramKind = entry.vramKind || VRAM_KIND.OTHER;
          this._pendingRetireBytes = Math.max(0, this._pendingRetireBytes - byteSize);
          this._vramPendingBytesByKind[vramKind] = Math.max(
            0,
            this._vramPendingBytesByKind[vramKind] - byteSize
          );
          freedBytes += byteSize;
          if (this._destroyEntry(entry)) {
            destroyedCount += 1;
            this._evictedCount += 1;
            this._evictedBytes += byteSize;
          }
        } else {
          remaining.push(entry);
        }
      }

      this._pendingRetire = remaining;
      this._checkBudget(null);

      return {
        destroyedCount,
        freedBytes,
        remainingCount: this._pendingRetire.length,
        pendingRetireBytes: this._pendingRetireBytes,
      };
    }

    /**
     * Alias for `collect(force)`.
     *
     * @param {boolean} [force=false]
     */
    flush(force = false) {
      return this.collect(force);
    }

    /**
     * Return up to `limit` currently live resources sorted oldest-first by
     * `createdFrame` (then registration order). Output is strictly sanitized to
     * `{ kind, id, byteSize, createdFrame }` with zero labels, host paths,
     * tags, or raw GPU handles.
     *
     * @param {number} [limit=16]
     * @returns {Array<{ kind: string, id: number, byteSize: number, createdFrame: number }>}
     */
    getOldestLiveResources(limit = DEFAULT_OLDEST_LIVE_LIMIT) {
      assertNonNegativeInteger(limit, 'limit');
      if (limit === 0) {
        return [];
      }

      const live = [];
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const table = this._tables.get(RESOURCE_KINDS[i]);
        for (const entry of table.values()) {
          live.push(entry);
        }
      }

      live.sort((a, b) => {
        if (a.registeredAtFrame !== b.registeredAtFrame) {
          return a.registeredAtFrame - b.registeredAtFrame;
        }
        return a.registrationSeq - b.registrationSeq;
      });

      const count = Math.min(limit, live.length);
      const result = new Array(count);
      for (let i = 0; i < count; i += 1) {
        const entry = live[i];
        result[i] = {
          kind: entry.kind,
          id: entry.handle,
          byteSize: entry.byteSize,
          createdFrame: entry.registeredAtFrame,
        };
      }
      return result;
    }

    /**
     * Return a sanitized per-kind VRAM accounting snapshot broken down into
     * `buffer`, `texture`, `renderTarget`, `sampler`, and `other`, along with
     * total live/pending/peak VRAM and stale-eviction byte counters.
     *
     * @returns {{
     *   buffer: { count: number, liveBytes: number, pendingRetireBytes: number, totalBytes: number, peakBytes: number, peakLiveBytes: number },
     *   texture: { count: number, liveBytes: number, pendingRetireBytes: number, totalBytes: number, peakBytes: number, peakLiveBytes: number },
     *   renderTarget: { count: number, liveBytes: number, pendingRetireBytes: number, totalBytes: number, peakBytes: number, peakLiveBytes: number },
     *   sampler: { count: number, liveBytes: number, pendingRetireBytes: number, totalBytes: number, peakBytes: number, peakLiveBytes: number },
     *   other: { count: number, liveBytes: number, pendingRetireBytes: number, totalBytes: number, peakBytes: number, peakLiveBytes: number },
     *   byKind: Record<string, { count: number, liveBytes: number, pendingRetireBytes: number, totalBytes: number, peakBytes: number, peakLiveBytes: number }>,
     *   totalLiveBytes: number,
     *   totalPendingRetireBytes: number,
     *   totalBytes: number,
     *   peakTotalVramBytes: number,
     *   evictedCount: number,
     *   evictedBytes: number,
     *   softBudgetBytes: number | null,
     *   budgetExceeded: boolean,
     * }}
     */
    getVramBreakdown() {
      const byKind = this._buildVramByKindSnapshot();
      const activeBytes = sanitizeNonNegativeCounter(this._activeBytes);
      const pendingRetireBytes = sanitizeNonNegativeCounter(this._pendingRetireBytes);
      const totalBytes = activeBytes + pendingRetireBytes;
      const peakTotalVramBytes = Math.max(
        totalBytes,
        sanitizeNonNegativeCounter(this._peakBytes)
      );

      const pressure = this.getVramPressure();

      return {
        buffer: byKind[VRAM_KIND.BUFFER],
        texture: byKind[VRAM_KIND.TEXTURE],
        renderTarget: byKind[VRAM_KIND.RENDER_TARGET],
        sampler: byKind[VRAM_KIND.SAMPLER],
        other: byKind[VRAM_KIND.OTHER],
        byKind,
        totalLiveBytes: activeBytes,
        totalPendingRetireBytes: pendingRetireBytes,
        totalBytes,
        peakTotalVramBytes,
        evictedCount: sanitizeNonNegativeCounter(this._evictedCount),
        evictedBytes: sanitizeNonNegativeCounter(this._evictedBytes),
        softBudgetBytes: this.softBudgetBytes,
        budgetExceeded: this._budgetExceeded,
        perfProfile: this._perfProfileName,
        vramReclaimTriggerRatio: this.vramReclaimTriggerRatio,
        vramReclaimTargetRatio: this.vramReclaimTargetRatio,
        vramPressureTier: pressure.tier,
        vramUtilization: pressure.utilization,
        vramShouldReclaim: pressure.shouldReclaim,
      };
    }

    /**
     * Return a sanitized snapshot of per-kind counts, VRAM byte accounting,
     * lifecycle stats, Frame-1 hardening counters, W250 texture-view &
     * r8unorm/rgba8unorm format telemetry, and W300 per-kind VRAM breakdown &
     * stale-eviction telemetry.
     */
    getStats() {
      const countsByKind = {};
      const byKind = {};
      let activeCount = 0;
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const kind = RESOURCE_KINDS[i];
        const count = this._tablesByKind[kind].size;
        countsByKind[kind] = count;
        byKind[kind] = {
          count,
          bytes: sanitizeNonNegativeCounter(this._activeBytesByKind[kind]),
        };
        activeCount += count;
      }

      const pendingCountsByKind = {};
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        pendingCountsByKind[RESOURCE_KINDS[i]] = 0;
      }
      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        pendingCountsByKind[this._pendingRetire[i].kind] += 1;
      }

      let cachedViewCount = 0;
      for (const perTexCache of this._textureViewCache.values()) {
        cachedViewCount += perTexCache.size;
      }

      const activeBytes = sanitizeNonNegativeCounter(this._activeBytes);
      const pendingRetireBytes = sanitizeNonNegativeCounter(this._pendingRetireBytes);
      const totalBytes = activeBytes + pendingRetireBytes;
      const peakBytes = Math.max(totalBytes, sanitizeNonNegativeCounter(this._peakBytes));
      const vramByKind = this._buildVramByKindSnapshot();

      const liveTextures = countsByKind[RESOURCE_KIND.TEXTURE];
      const liveBuffers = countsByKind[RESOURCE_KIND.BUFFER];
      const peakLiveTextures = Math.max(
        liveTextures,
        sanitizeNonNegativeCounter(this._peakLiveTextures)
      );
      const peakLiveBuffers = Math.max(
        liveBuffers,
        sanitizeNonNegativeCounter(this._peakLiveBuffers)
      );

      const viewCacheHits = sanitizeNonNegativeCounter(this._viewCacheHits);
      const viewCacheMisses = sanitizeNonNegativeCounter(this._viewCacheMisses);
      const viewCacheHitRate = clampUnitRate(viewCacheHits, viewCacheHits + viewCacheMisses);
      const viewCacheInvalidations = sanitizeNonNegativeCounter(this._viewCacheInvalidations);
      const pressure = this.getVramPressure();

      return {
        countsByKind,
        byKind,
        vramByKind,
        pendingCountsByKind,
        activeCount,
        pendingRetireCount: this._pendingRetire.length,
        activeBytes,
        pendingRetireBytes,
        totalBytes,
        peakBytes,
        peakTotalBytes: peakBytes,
        peakTotalVramBytes: peakBytes,
        peakLiveTextures,
        peakLiveBuffers,
        softBudgetBytes: this.softBudgetBytes,
        budgetExceeded: this._budgetExceeded,
        budgetWarningCount: sanitizeNonNegativeCounter(this._budgetWarningCount),
        perfProfile: this._perfProfileName,
        vramReclaimTriggerRatio: this.vramReclaimTriggerRatio,
        vramReclaimTargetRatio: this.vramReclaimTargetRatio,
        vramPressureTier: pressure.tier,
        vramUtilization: pressure.utilization,
        vramShouldReclaim: pressure.shouldReclaim,
        totalRegistered: sanitizeNonNegativeCounter(this._totalRegistered),
        totalRetired: sanitizeNonNegativeCounter(this._totalRetired),
        totalDestroyed: sanitizeNonNegativeCounter(this._totalDestroyed),
        destroyErrors: sanitizeNonNegativeCounter(this._destroyErrors),
        evictedCount: sanitizeNonNegativeCounter(this._evictedCount),
        evictedBytes: sanitizeNonNegativeCounter(this._evictedBytes),
        staleLookupMisses: sanitizeNonNegativeCounter(this._staleLookupMisses),
        fastHandleHits: sanitizeNonNegativeCounter(this._fastHandleHits),
        fastHandleOverflowHits: sanitizeNonNegativeCounter(this._fastHandleOverflowHits),
        currentFrame: sanitizeNonNegativeCounter(this._currentFrame),
        framesInFlight: this.framesInFlight,
        bcFormatFallbackCount: sanitizeNonNegativeCounter(this._bcFormatFallbackCount),
        bcDimensionAlignCount: sanitizeNonNegativeCounter(this._bcDimensionAlignCount),
        bufferAlignPadCount: sanitizeNonNegativeCounter(this._bufferAlignPadCount),
        mipClampCount: sanitizeNonNegativeCounter(this._mipClampCount),
        sampleClampCount: sanitizeNonNegativeCounter(this._sampleClampCount),
        staleViewsInvalidated: sanitizeNonNegativeCounter(this._staleViewsInvalidated),
        viewCacheHits,
        viewCacheMisses,
        viewCacheHitRate,
        viewCacheInvalidations,
        viewInvalidations: viewCacheInvalidations,
        viewCacheEvictions: sanitizeNonNegativeCounter(this._viewCacheEvictions),
        maxCachedViewsPerTexture: this.maxCachedViewsPerTexture,
        textureRecreations: sanitizeNonNegativeCounter(this._textureRecreations),
        r8unormTextures: sanitizeNonNegativeCounter(this._r8unormTextures),
        rgba8unormTextures: sanitizeNonNegativeCounter(this._rgba8unormTextures),
        cachedViewCount,
      };
    }

    /**
     * Reset cumulative telemetry counters while preserving all live resources,
     * cached texture views, texture generations, and deferred retirement queues.
     * Re-baselines `peakBytes`, `peakTotalVramBytes`, per-kind `vramByKind` peaks,
     * `peakLiveTextures`, and `peakLiveBuffers` to current live occupancy so
     * peak >= live invariants always hold.
     *
     * @returns {object} Fresh `getStats()` snapshot after reset
     */
    resetStats() {
      this._peakBytes =
        sanitizeNonNegativeCounter(this._activeBytes) +
        sanitizeNonNegativeCounter(this._pendingRetireBytes);
      this._peakLiveTextures = this._tablesByKind[RESOURCE_KIND.TEXTURE].size;
      this._peakLiveBuffers = this._tablesByKind[RESOURCE_KIND.BUFFER].size;

      for (let i = 0; i < VRAM_KINDS.length; i += 1) {
        const vk = VRAM_KINDS[i];
        const liveBytes = sanitizeNonNegativeCounter(this._vramLiveBytesByKind[vk]);
        const pendingBytes = sanitizeNonNegativeCounter(this._vramPendingBytesByKind[vk]);
        this._vramPeakLiveBytesByKind[vk] = liveBytes;
        this._vramPeakBytesByKind[vk] = liveBytes + pendingBytes;
      }

      this._budgetWarningCount = 0;
      this._totalRegistered = 0;
      this._totalRetired = 0;
      this._totalDestroyed = 0;
      this._destroyErrors = 0;

      this._evictedCount = 0;
      this._evictedBytes = 0;
      this._staleLookupMisses = 0;
      this._fastHandleHits = 0;
      this._fastHandleOverflowHits = 0;

      this._bcFormatFallbackCount = 0;
      this._bcDimensionAlignCount = 0;
      this._bufferAlignPadCount = 0;
      this._mipClampCount = 0;
      this._sampleClampCount = 0;

      this._staleViewsInvalidated = 0;
      this._viewCacheHits = 0;
      this._viewCacheMisses = 0;
      this._viewCacheInvalidations = 0;
      this._viewCacheEvictions = 0;
      this._textureRecreations = 0;
      this._r8unormTextures = 0;
      this._rgba8unormTextures = 0;

      return this.getStats();
    }

    /**
     * Idempotently destroy all active and pending-retirement resources
     * (e.g. for `device.lost` recovery or session shutdown).
     *
     * @returns {{ destroyedCount: number, freedBytes: number }}
     */
    destroyAll() {
      this._generation += 1;
      this._pendingFences.clear();

      // Invalidate all cached views and linked TEXTURE_VIEW handles prior to clearing.
      let invalidatedViews = 0;
      const countedViews = new Set();
      for (const perTexCache of this._textureViewCache.values()) {
        for (const cached of perTexCache.values()) {
          if (cached && cached.view && typeof cached.view === 'object') {
            const viewMeta = this._viewMetadata.get(cached.view);
            if (viewMeta) {
              viewMeta.invalidated = true;
            }
            if (!countedViews.has(cached.view)) {
              countedViews.add(cached.view);
              invalidatedViews += 1;
            }
          } else {
            invalidatedViews += 1;
          }
        }
      }
      const activeViewTable = this._tablesByKind[RESOURCE_KIND.TEXTURE_VIEW];
      for (const viewEntry of activeViewTable.values()) {
        if (viewEntry && viewEntry.textureId !== undefined && viewEntry.textureId !== null) {
          const viewObj = viewEntry.resource;
          if (viewObj && typeof viewObj === 'object') {
            const viewMeta = this._viewMetadata.get(viewObj);
            if (viewMeta) {
              viewMeta.invalidated = true;
            }
            if (!countedViews.has(viewObj)) {
              countedViews.add(viewObj);
              invalidatedViews += 1;
            }
          } else {
            invalidatedViews += 1;
          }
        }
      }
      if (invalidatedViews > 0) {
        this._staleViewsInvalidated += invalidatedViews;
        this._viewCacheInvalidations += invalidatedViews;
      }
      this._textureViewCache.clear();
      this._linkedViewHandlesByTexture.clear();
      this._activeTextureGeneration.clear();

      const toDestroy = [];
      for (let i = 0; i < RESOURCE_KINDS.length; i += 1) {
        const kind = RESOURCE_KINDS[i];
        const table = this._tablesByKind[kind];
        for (const entry of table.values()) {
          this._markHandleStale(kind, entry.handle);
          toDestroy.push(entry);
        }
        table.clear();
        this._fastSlotsByKind[kind].length = 0;
        this._activeBytesByKind[kind] = 0;
      }
      for (let i = 0; i < VRAM_KINDS.length; i += 1) {
        const vk = VRAM_KINDS[i];
        this._vramLiveBytesByKind[vk] = 0;
        this._vramPendingBytesByKind[vk] = 0;
        this._vramLiveCountByKind[vk] = 0;
      }
      this._bufferMeta.clear();
      this._textureMeta.clear();

      for (let i = 0; i < this._pendingRetire.length; i += 1) {
        toDestroy.push(this._pendingRetire[i]);
      }
      this._pendingRetire = [];

      this._activeBytes = 0;
      this._pendingRetireBytes = 0;
      this._budgetExceeded = false;

      let destroyedCount = 0;
      let freedBytes = 0;

      for (let i = 0; i < toDestroy.length; i += 1) {
        const entry = toDestroy[i];
        const byteSize = sanitizeNonNegativeCounter(entry.byteSize);
        freedBytes += byteSize;
        if (this._destroyEntry(entry)) {
          destroyedCount += 1;
          this._evictedCount += 1;
          this._evictedBytes += byteSize;
        }
      }

      return {
        destroyedCount,
        freedBytes,
      };
    }

    /**
     * Alias for `destroyAll()` that clears all active and deferred resources,
     * zeroes live/pending VRAM across all kinds, and records eviction stats.
     *
     * @returns {{ destroyedCount: number, freedBytes: number }}
     */
    clear() {
      return this.destroyAll();
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_FRAMES_IN_FLIGHT,
    DEFAULT_SOFT_BUDGET_BYTES,
    DEFAULT_VRAM_RECLAIM_TRIGGER_RATIO,
    DEFAULT_VRAM_RECLAIM_TARGET_RATIO,
    DEFAULT_MAX_CACHED_VIEWS_PER_TEXTURE,
    FAST_HANDLE_LIMIT,
    DEFAULT_MAX_STALE_HANDLE_HISTORY,
    CREATE_TEXTURE2_OPCODE,
    CREATE_TEXTURE2_PAYLOAD_BYTES,
    MAX_TEXTURE_VIEW_FORMATS,
    GPU_BUFFER_USAGE,
    GPU_TEXTURE_USAGE,
    V1_TEXTURE_FORMAT_NAMES,
    TEXTURE_FORMAT_NAMES,
    TEXTURE_DIMENSION_NAMES,
    TEXTURE_VIEW_DIMENSIONS,
    TEXTURE_VIEW_ASPECTS,
    SRGB_SIBLING_FORMATS,
    BINK_VIDEO_TEXTURE_FORMATS,
    BC_TEXTURE_FORMATS,
    BC_FALLBACK_FORMAT,
    TEXTURE_COMPRESSION_BC_FEATURE,
    BUFFER_ALIGNMENT_BYTES,
    UNIFORM_BUFFER_MIN_BYTES,
    RESOURCE_KIND,
    RESOURCE_KINDS,
    VRAM_KIND,
    VRAM_KINDS,
    VRAM_PRESSURE_TIER,
    resolveVramKind,
    isBcFormat,
    isSrgbFormat,
    getSrgbSiblingFormat,
    isViewFormatCompatible,
    resolveTextureViewFormats,
    buildTextureViewCacheKey,
    normalizeTextureViewDescriptor,
    deviceSupportsBc,
    computeMaxMipLevels,
    clampMipLevelCount,
    clampSampleCount,
    alignBcDimension,
    alignBufferSize,
    padBufferUploadData,
    estimateTextureBytes,
    decodeCreateTexture2,
    sanitizeTextureDescriptor,
    GpuResourceRegistry,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6ResourceRegistry = exportsObj;
  } else if (root) {
    root.PlayGta6ResourceRegistry = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      DEFAULT_FRAMES_IN_FLIGHT,
      DEFAULT_SOFT_BUDGET_BYTES,
      DEFAULT_VRAM_RECLAIM_TRIGGER_RATIO,
      DEFAULT_VRAM_RECLAIM_TARGET_RATIO,
      DEFAULT_MAX_CACHED_VIEWS_PER_TEXTURE,
      FAST_HANDLE_LIMIT,
      DEFAULT_MAX_STALE_HANDLE_HISTORY,
      CREATE_TEXTURE2_OPCODE,
      CREATE_TEXTURE2_PAYLOAD_BYTES,
      MAX_TEXTURE_VIEW_FORMATS,
      GPU_BUFFER_USAGE,
      GPU_TEXTURE_USAGE,
      V1_TEXTURE_FORMAT_NAMES,
      TEXTURE_FORMAT_NAMES,
      TEXTURE_DIMENSION_NAMES,
      TEXTURE_VIEW_DIMENSIONS,
      TEXTURE_VIEW_ASPECTS,
      SRGB_SIBLING_FORMATS,
      BINK_VIDEO_TEXTURE_FORMATS,
      BC_TEXTURE_FORMATS,
      BC_FALLBACK_FORMAT,
      TEXTURE_COMPRESSION_BC_FEATURE,
      BUFFER_ALIGNMENT_BYTES,
      UNIFORM_BUFFER_MIN_BYTES,
      RESOURCE_KIND,
      RESOURCE_KINDS,
      VRAM_KIND,
      VRAM_KINDS,
      VRAM_PRESSURE_TIER,
      resolveVramKind,
      isBcFormat,
      isSrgbFormat,
      getSrgbSiblingFormat,
      isViewFormatCompatible,
      resolveTextureViewFormats,
      buildTextureViewCacheKey,
      normalizeTextureViewDescriptor,
      deviceSupportsBc,
      computeMaxMipLevels,
      clampMipLevelCount,
      clampSampleCount,
      alignBcDimension,
      alignBufferSize,
      padBufferUploadData,
      estimateTextureBytes,
      decodeCreateTexture2,
      sanitizeTextureDescriptor,
      GpuResourceRegistry,
    };
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
