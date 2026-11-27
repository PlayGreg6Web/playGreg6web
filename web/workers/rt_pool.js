'use strict';

// WebGPU transient render-target texture pool and per-frame attachment aliasing manager.
// Zero external dependencies; UMD export for Web Workers (importScripts / globalThis.PlayGta6RenderTargetPool),
// browsers, and Node.js (module.exports).
// Designed for RAGE deferred color/depth/shadow/post-FX passes in web/workers/wgpu_worker.js.

(function (root) {
  const DEFAULT_MAX_POOL_BYTES = 512 * 1024 * 1024; // 512 MiB (accommodates simultaneous 4K scene plane 1 + 1080p UI plane 0)
  const DEFAULT_IDLE_FRAME_TTL = 8;
  const DEFAULT_OVERSIZED_TARGET_BYTES = 24 * 1024 * 1024; // 24 MiB (separates 4K targets >= 31.6 MiB from 1080p targets <= 15.8 MiB)
  const DEFAULT_OVERSIZED_RECLAIM_IDLE_FRAMES = 2;
  const DEFAULT_MAX_TEXTURE_DIMENSION_2D = 8192;
  const DEFAULT_MAX_TEXTURE_ARRAY_LAYERS = 256;

  const FALLBACK_RT_PERF_PROFILES = Object.freeze({
    low: Object.freeze({ name: 'low', rtPoolMaxBytes: 128 * 1024 * 1024, rtPoolMaxIdleFrames: 8 }),
    balanced: Object.freeze({
      name: 'balanced',
      rtPoolMaxBytes: 256 * 1024 * 1024,
      rtPoolMaxIdleFrames: 16,
    }),
    high: Object.freeze({
      name: 'high',
      rtPoolMaxBytes: 512 * 1024 * 1024,
      rtPoolMaxIdleFrames: 30,
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
        return FALLBACK_RT_PERF_PROFILES.balanced;
      }
      if (Object.prototype.hasOwnProperty.call(FALLBACK_RT_PERF_PROFILES, key)) {
        return FALLBACK_RT_PERF_PROFILES[key];
      }
      throw new RangeError(`Unknown performance profile "${profileOrName}"`);
    }
    if (profileOrName && typeof profileOrName === 'object') {
      const name =
        typeof profileOrName.name === 'string'
          ? profileOrName.name.trim().toLowerCase()
          : typeof profileOrName.preset === 'string'
            ? profileOrName.preset.trim().toLowerCase()
            : 'custom';
      const base = Object.prototype.hasOwnProperty.call(FALLBACK_RT_PERF_PROFILES, name)
        ? FALLBACK_RT_PERF_PROFILES[name]
        : FALLBACK_RT_PERF_PROFILES.balanced;
      return Object.freeze({
        name: base.name,
        rtPoolMaxBytes:
          profileOrName.rtPoolMaxBytes !== undefined
            ? profileOrName.rtPoolMaxBytes
            : profileOrName.maxPoolBytes !== undefined
              ? profileOrName.maxPoolBytes
              : base.rtPoolMaxBytes,
        rtPoolMaxIdleFrames:
          profileOrName.rtPoolMaxIdleFrames !== undefined
            ? profileOrName.rtPoolMaxIdleFrames
            : profileOrName.idleFrameTtl !== undefined
              ? profileOrName.idleFrameTtl
              : profileOrName.maxIdleFrames !== undefined
                ? profileOrName.maxIdleFrames
                : base.rtPoolMaxIdleFrames,
      });
    }
    throw new TypeError('perfProfile must be a preset name or profile object');
  }

  const VALID_VIEW_DIMENSIONS = Object.freeze([
    '1d',
    '2d',
    '2d-array',
    'cube',
    'cube-array',
    '3d',
  ]);
  const VALID_VIEW_ASPECTS = Object.freeze(['all', 'stencil-only', 'depth-only']);

  // WebGPU GPUTextureUsage bitmask values with headless/Node fallbacks.
  const TEXTURE_USAGE = Object.freeze({
    COPY_SRC:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_SRC === 'number'
        ? GPUTextureUsage.COPY_SRC
        : 0x01,
    COPY_DST:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_DST === 'number'
        ? GPUTextureUsage.COPY_DST
        : 0x02,
    TEXTURE_BINDING:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.TEXTURE_BINDING === 'number'
        ? GPUTextureUsage.TEXTURE_BINDING
        : 0x04,
    STORAGE_BINDING:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.STORAGE_BINDING === 'number'
        ? GPUTextureUsage.STORAGE_BINDING
        : 0x08,
    RENDER_ATTACHMENT:
      typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.RENDER_ATTACHMENT === 'number'
        ? GPUTextureUsage.RENDER_ATTACHMENT
        : 0x10,
  });

  const DEFAULT_TEXTURE_USAGE = TEXTURE_USAGE.RENDER_ATTACHMENT | TEXTURE_USAGE.TEXTURE_BINDING;

  const VALID_DIMENSIONS = Object.freeze(['1d', '2d', '3d']);

  /**
   * Common DXGI / Xbox Series render-target and depth-stencil format aliases mapped
   * to canonical WebGPU `GPUTextureFormat` strings.
   */
  const DXGI_FORMAT_ALIASES = Object.freeze({
    // 8-bit single-channel
    r8_unorm: 'r8unorm',
    r8_snorm: 'r8snorm',
    r8_uint: 'r8uint',
    r8_sint: 'r8sint',

    // 16-bit formats
    r16_float: 'r16float',
    r16_unorm: 'r16unorm',
    r16_snorm: 'r16snorm',
    r16_uint: 'r16uint',
    r16_sint: 'r16sint',
    r8g8_unorm: 'rg8unorm',
    r8g8_snorm: 'rg8snorm',
    r8g8_uint: 'rg8uint',
    r8g8_sint: 'rg8sint',
    d16_unorm: 'depth16unorm',

    // 32-bit color & depth-stencil formats
    r32_float: 'r32float',
    r32_uint: 'r32uint',
    r32_sint: 'r32sint',
    r16g16_float: 'rg16float',
    r16g16_unorm: 'rg16unorm',
    r16g16_snorm: 'rg16snorm',
    r16g16_uint: 'rg16uint',
    r16g16_sint: 'rg16sint',
    r8g8b8a8_unorm: 'rgba8unorm',
    r8g8b8a8_unorm_srgb: 'rgba8unorm-srgb',
    r8g8b8a8_snorm: 'rgba8snorm',
    r8g8b8a8_uint: 'rgba8uint',
    r8g8b8a8_sint: 'rgba8sint',
    b8g8r8a8_unorm: 'bgra8unorm',
    b8g8r8a8_unorm_srgb: 'bgra8unorm-srgb',
    r10g10b10a2_unorm: 'rgb10a2unorm',
    r10g10b10a2_uint: 'rgb10a2uint',
    r11g11b10_float: 'rg11b10ufloat',
    r9g9b9e5_sharedexp: 'rgb9e5ufloat',
    d32_float: 'depth32float',
    d24_unorm_s8_uint: 'depth24plus-stencil8',
    r24g8_typeless: 'depth24plus-stencil8',
    x24_typeless_g8_uint: 'depth24plus-stencil8',

    // 64-bit color & depth-stencil formats
    r32g32_float: 'rg32float',
    r32g32_uint: 'rg32uint',
    r32g32_sint: 'rg32sint',
    r16g16b16a16_float: 'rgba16float',
    r16g16b16a16_unorm: 'rgba16unorm',
    r16g16b16a16_snorm: 'rgba16snorm',
    r16g16b16a16_uint: 'rgba16uint',
    r16g16b16a16_sint: 'rgba16sint',
    d32_float_s8x24_uint: 'depth32float-stencil8',
    r32g8x24_typeless: 'depth32float-stencil8',
    x32_typeless_g8x24_uint: 'depth32float-stencil8',

    // 128-bit color formats
    r32g32b32a32_float: 'rgba32float',
    r32g32b32a32_uint: 'rgba32uint',
    r32g32b32a32_sint: 'rgba32sint',

    // Common shorthand 3D HDR / depth aliases
    rgba16f: 'rgba16float',
    rg16f: 'rg16float',
    r16f: 'r16float',
    rg11b10f: 'rg11b10ufloat',
    rgb10a2: 'rgb10a2unorm',
    r32f: 'r32float',
    rg32f: 'rg32float',
    rgba32f: 'rgba32float',
    d32f: 'depth32float',
    d32fs8: 'depth32float-stencil8',
    d24s8: 'depth24plus-stencil8',
    d16: 'depth16unorm',
  });

  /**
   * Bytes-per-texel lookup table for common WebGPU color, depth, and stencil render-target formats.
   * Note: `depth24plus` occupies 4 bytes/texel in hardware backing storage (D24X8 / D32F);
   * `depth32float-stencil8` occupies 8 bytes/texel (32-bit depth + 8-bit stencil + 24-bit pad).
   */
  const FORMAT_BYTES_PER_PIXEL = Object.freeze({
    // 8-bit / 1-byte formats
    r8unorm: 1,
    r8snorm: 1,
    r8uint: 1,
    r8sint: 1,
    stencil8: 1,

    // 16-bit / 2-byte formats
    r16float: 2,
    r16uint: 2,
    r16sint: 2,
    r16unorm: 2,
    r16snorm: 2,
    rg8unorm: 2,
    rg8snorm: 2,
    rg8uint: 2,
    rg8sint: 2,
    depth16unorm: 2,

    // 32-bit / 4-byte formats
    r32float: 4,
    r32uint: 4,
    r32sint: 4,
    rg16float: 4,
    rg16uint: 4,
    rg16sint: 4,
    rg16unorm: 4,
    rg16snorm: 4,
    rgba8unorm: 4,
    'rgba8unorm-srgb': 4,
    rgba8snorm: 4,
    rgba8uint: 4,
    rgba8sint: 4,
    bgra8unorm: 4,
    'bgra8unorm-srgb': 4,
    rgb10a2unorm: 4,
    rgb10a2uint: 4,
    rg11b10ufloat: 4,
    rgb9e5ufloat: 4,
    depth24plus: 4,
    'depth24plus-stencil8': 4,
    depth32float: 4,

    // 64-bit / 8-byte formats
    rg32float: 8,
    rg32uint: 8,
    rg32sint: 8,
    rgba16float: 8,
    rgba16uint: 8,
    rgba16sint: 8,
    rgba16unorm: 8,
    rgba16snorm: 8,
    'depth32float-stencil8': 8,

    // 128-bit / 16-byte formats
    rgba32float: 16,
    rgba32uint: 16,
    rgba32sint: 16,
  });

  /**
   * WebGPU compatible alternate `viewFormats` for render-target formats (sRGB / linear pairs).
   * A texture can always be viewed in its own base `format` without listing it in `viewFormats`.
   */
  const COMPATIBLE_VIEW_FORMATS = Object.freeze({
    rgba8unorm: Object.freeze(['rgba8unorm-srgb']),
    'rgba8unorm-srgb': Object.freeze(['rgba8unorm']),
    bgra8unorm: Object.freeze(['bgra8unorm-srgb']),
    'bgra8unorm-srgb': Object.freeze(['bgra8unorm']),
  });

  const REQUIRED_3D_RT_FORMATS = Object.freeze([
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
    'rgba16float',
    'rg11b10ufloat',
    'r16float',
    'rgb10a2unorm',
    'rgba8unorm-srgb',
    'rgba8unorm',
    'bgra8unorm',
    'bgra8unorm-srgb',
    'rg16float',
    'r32float',
    'depth16unorm',
  ]);

  const HDR_RENDER_TARGET_FORMATS = new Set([
    'rgba16float',
    'rg11b10ufloat',
    'r16float',
    'rg16float',
    'r32float',
    'rg32float',
    'rgba32float',
    'rgb10a2unorm',
    'rgb9e5ufloat',
  ]);

  const DEPTH_RENDER_TARGET_FORMATS = new Set([
    'depth16unorm',
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
    'stencil8',
  ]);

  const STENCIL_RENDER_TARGET_FORMATS = new Set([
    'depth24plus-stencil8',
    'depth32float-stencil8',
    'stencil8',
  ]);

  const SRGB_RENDER_TARGET_FORMATS = new Set(['rgba8unorm-srgb', 'bgra8unorm-srgb']);

  const EMPTY_VIEW_FORMATS = Object.freeze([]);

  function assertPositiveInteger(value, name) {
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got ${value}`);
    }
  }

  function assertNonNegativeInteger(value, name) {
    if (!Number.isInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative integer, got ${value}`);
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

  /**
   * Resolve the effective 2D texture dimension limit from a WebGPU device,
   * defaulting to `8192` when `device.limits.maxTextureDimension2D` is absent.
   *
   * @param {object | null | undefined} device
   * @returns {number}
   */
  function resolveMaxTextureDimension2D(device) {
    const raw =
      device &&
      device.limits &&
      typeof device.limits.maxTextureDimension2D === 'number' &&
      Number.isFinite(device.limits.maxTextureDimension2D) &&
      device.limits.maxTextureDimension2D >= 1
        ? Math.floor(device.limits.maxTextureDimension2D)
        : DEFAULT_MAX_TEXTURE_DIMENSION_2D;
    return raw;
  }

  function resolveDeviceTextureLimits(device, overrideMaxDim2D) {
    const maxDim2D =
      typeof overrideMaxDim2D === 'number' &&
      Number.isFinite(overrideMaxDim2D) &&
      overrideMaxDim2D >= 1
        ? Math.floor(overrideMaxDim2D)
        : resolveMaxTextureDimension2D(device);

    const maxDim1D =
      device &&
      device.limits &&
      typeof device.limits.maxTextureDimension1D === 'number' &&
      Number.isFinite(device.limits.maxTextureDimension1D) &&
      device.limits.maxTextureDimension1D >= 1
        ? Math.floor(device.limits.maxTextureDimension1D)
        : maxDim2D;

    const maxDim3D =
      device &&
      device.limits &&
      typeof device.limits.maxTextureDimension3D === 'number' &&
      Number.isFinite(device.limits.maxTextureDimension3D) &&
      device.limits.maxTextureDimension3D >= 1
        ? Math.floor(device.limits.maxTextureDimension3D)
        : maxDim2D;

    const maxArrayLayers =
      device &&
      device.limits &&
      typeof device.limits.maxTextureArrayLayers === 'number' &&
      Number.isFinite(device.limits.maxTextureArrayLayers) &&
      device.limits.maxTextureArrayLayers >= 1
        ? Math.floor(device.limits.maxTextureArrayLayers)
        : DEFAULT_MAX_TEXTURE_ARRAY_LAYERS;

    return { maxDim1D, maxDim2D, maxDim3D, maxArrayLayers };
  }

  function clampDimensionValue(raw, maxDim) {
    if (typeof raw !== 'number' || !Number.isFinite(raw)) {
      return { value: 1, clamped: true };
    }
    const floored = Math.floor(raw);
    if (floored < 1) {
      return { value: 1, clamped: true };
    }
    if (floored > maxDim) {
      return { value: maxDim, clamped: true };
    }
    return { value: floored, clamped: floored !== raw };
  }

  function _normalizeFormatWithMeta(format) {
    if (typeof format !== 'string' || format.trim().length === 0) {
      throw new TypeError('desc.format must be a non-empty WebGPU GPUTextureFormat string');
    }
    const lower = format.trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(FORMAT_BYTES_PER_PIXEL, lower)) {
      return { format: lower, normalized: false };
    }
    const aliasKey = lower.startsWith('dxgi_format_')
      ? lower.slice('dxgi_format_'.length)
      : lower;
    if (Object.prototype.hasOwnProperty.call(DXGI_FORMAT_ALIASES, aliasKey)) {
      return { format: DXGI_FORMAT_ALIASES[aliasKey], normalized: true };
    }
    return { format: lower, normalized: false };
  }

  /**
   * Normalize a DXGI / Xbox Series render-target or depth-stencil format string
   * to its canonical WebGPU `GPUTextureFormat`, while preserving valid WebGPU
   * format strings untouched.
   *
   * @param {string} format
   * @returns {string}
   */
  function normalizeRenderTargetFormat(format) {
    return _normalizeFormatWithMeta(format).format;
  }

  function isHdrRenderTargetFormat(format) {
    if (typeof format !== 'string' || format.trim().length === 0) {
      return false;
    }
    return HDR_RENDER_TARGET_FORMATS.has(_normalizeFormatWithMeta(format).format);
  }

  function isDepthRenderTargetFormat(format) {
    if (typeof format !== 'string' || format.trim().length === 0) {
      return false;
    }
    return DEPTH_RENDER_TARGET_FORMATS.has(_normalizeFormatWithMeta(format).format);
  }

  function hasStencilComponent(format) {
    if (typeof format !== 'string' || format.trim().length === 0) {
      return false;
    }
    return STENCIL_RENDER_TARGET_FORMATS.has(_normalizeFormatWithMeta(format).format);
  }

  function getCompatibleSrgbSiblingFormat(format) {
    if (typeof format !== 'string' || format.trim().length === 0) {
      return null;
    }
    const canonical = _normalizeFormatWithMeta(format).format;
    const compatible = COMPATIBLE_VIEW_FORMATS[canonical];
    return Array.isArray(compatible) && compatible.length > 0 ? compatible[0] : null;
  }

  /**
   * Return a copy of `desc` with its compatible sRGB/linear sibling format included in
   * `viewFormats` (when `desc.format` has one, e.g. `rgba8unorm` <-> `rgba8unorm-srgb`
   * or `bgra8unorm` <-> `bgra8unorm-srgb`).
   *
   * @param {object} desc
   * @returns {object}
   */
  function withSrgbSiblingViewFormat(desc) {
    if (!desc || typeof desc !== 'object') {
      throw new TypeError('Render target descriptor must be an object');
    }
    const canonical = _normalizeFormatWithMeta(desc.format).format;
    const sibling = getCompatibleSrgbSiblingFormat(canonical);
    if (!sibling) {
      return { ...desc };
    }
    const existing =
      desc.viewFormats !== undefined && desc.viewFormats !== null
        ? Array.from(desc.viewFormats)
        : [];
    if (!existing.includes(sibling)) {
      existing.push(sibling);
    }
    return {
      ...desc,
      viewFormats: existing,
    };
  }

  function _extractFeatureSource(options) {
    if (!options || typeof options !== 'object') {
      return null;
    }
    if (options.deviceFeatures !== undefined && options.deviceFeatures !== null) {
      return options.deviceFeatures;
    }
    if (
      options.device &&
      typeof options.device === 'object' &&
      options.device.features !== undefined &&
      options.device.features !== null
    ) {
      return options.device.features;
    }
    return null;
  }

  function _hasDeviceFeature(features, featureName) {
    if (!features || !featureName) {
      return false;
    }
    if (typeof features.has === 'function') {
      return Boolean(features.has(featureName));
    }
    if (Array.isArray(features)) {
      return features.includes(featureName);
    }
    if (typeof features[Symbol.iterator] === 'function') {
      for (const item of features) {
        if (item === featureName) {
          return true;
        }
      }
      return false;
    }
    if (typeof features === 'object') {
      return Boolean(features[featureName]);
    }
    return false;
  }

  /**
   * Resolve a canonical WebGPU render-target format against optional device feature support,
   * falling back safely when a device lacks an optional feature (e.g.
   * `depth32float-stencil8` -> `depth24plus-stencil8`,
   * `rg11b10ufloat` with `RENDER_ATTACHMENT` -> `rgba16float`,
   * 16-bit unorm/snorm -> matching float16 format).
   *
   * When neither `options.device.features` nor `options.deviceFeatures` is supplied (as in
   * headless mock devices without feature sets), preserves the requested canonical format
   * unless `options.requireFeatureSupport === true`.
   *
   * @param {string} format
   * @param {{
   *   device?: object | null,
   *   deviceFeatures?: Set<string> | string[] | Iterable<string> | Record<string, boolean> | null,
   *   usage?: number,
   *   requireFeatureSupport?: boolean,
   * }} [options]
   * @returns {{
   *   format: string,
   *   requestedFormat: string,
   *   formatFallback: boolean,
   *   fallbackReason: string | null,
   *   requiredFeature: string | null,
   * }}
   */
  function resolveRenderTargetFormatForDevice(format, options = {}) {
    const requestedFormat = _normalizeFormatWithMeta(format).format;
    const opts = options && typeof options === 'object' ? options : {};
    const features = _extractFeatureSource(opts);
    const enforce = features !== null || Boolean(opts.requireFeatureSupport);
    const usage =
      typeof opts.usage === 'number' && Number.isInteger(opts.usage) && opts.usage > 0
        ? opts.usage
        : DEFAULT_TEXTURE_USAGE;

    if (!enforce) {
      return {
        format: requestedFormat,
        requestedFormat,
        formatFallback: false,
        fallbackReason: null,
        requiredFeature: null,
      };
    }

    if (requestedFormat === 'depth32float-stencil8') {
      if (!_hasDeviceFeature(features, 'depth32float-stencil8')) {
        return {
          format: 'depth24plus-stencil8',
          requestedFormat,
          formatFallback: true,
          fallbackReason: 'missing-depth32float-stencil8',
          requiredFeature: 'depth32float-stencil8',
        };
      }
      return {
        format: requestedFormat,
        requestedFormat,
        formatFallback: false,
        fallbackReason: null,
        requiredFeature: 'depth32float-stencil8',
      };
    }

    if (requestedFormat === 'rg11b10ufloat') {
      const wantsRenderAttachment = (usage & TEXTURE_USAGE.RENDER_ATTACHMENT) !== 0;
      if (wantsRenderAttachment && !_hasDeviceFeature(features, 'rg11b10ufloat-renderable')) {
        return {
          format: 'rgba16float',
          requestedFormat,
          formatFallback: true,
          fallbackReason: 'missing-rg11b10ufloat-renderable',
          requiredFeature: 'rg11b10ufloat-renderable',
        };
      }
      return {
        format: requestedFormat,
        requestedFormat,
        formatFallback: false,
        fallbackReason: null,
        requiredFeature: wantsRenderAttachment ? 'rg11b10ufloat-renderable' : null,
      };
    }

    if (requestedFormat === 'rgb9e5ufloat') {
      if ((usage & TEXTURE_USAGE.RENDER_ATTACHMENT) !== 0) {
        return {
          format: 'rgba16float',
          requestedFormat,
          formatFallback: true,
          fallbackReason: 'rgb9e5ufloat-not-renderable',
          requiredFeature: null,
        };
      }
    }

    if (
      requestedFormat === 'r16unorm' ||
      requestedFormat === 'r16snorm' ||
      requestedFormat === 'rg16unorm' ||
      requestedFormat === 'rg16snorm' ||
      requestedFormat === 'rgba16unorm' ||
      requestedFormat === 'rgba16snorm'
    ) {
      const hasNorm16 =
        _hasDeviceFeature(features, 'texture-formats-tier1') ||
        _hasDeviceFeature(features, 'chromium-experimental-unorm16-texture-formats');
      if (!hasNorm16) {
        const fallback = requestedFormat.startsWith('rgba')
          ? 'rgba16float'
          : requestedFormat.startsWith('rg')
            ? 'rg16float'
            : 'r16float';
        return {
          format: fallback,
          requestedFormat,
          formatFallback: true,
          fallbackReason: 'missing-unorm16',
          requiredFeature: 'texture-formats-tier1',
        };
      }
      return {
        format: requestedFormat,
        requestedFormat,
        formatFallback: false,
        fallbackReason: null,
        requiredFeature: 'texture-formats-tier1',
      };
    }

    return {
      format: requestedFormat,
      requestedFormat,
      formatFallback: false,
      fallbackReason: null,
      requiredFeature: null,
    };
  }

  /**
   * Return true if `viewFormat` is a WebGPU-compatible texture view format for `baseFormat`
   * (either the same canonical format or its compatible sRGB/linear counterpart).
   *
   * @param {string} baseFormat
   * @param {string} viewFormat
   * @returns {boolean}
   */
  function isCompatibleViewFormat(baseFormat, viewFormat) {
    const base = _normalizeFormatWithMeta(baseFormat).format;
    const vf = _normalizeFormatWithMeta(viewFormat).format;
    if (vf === base) {
      return true;
    }
    const compatible = COMPATIBLE_VIEW_FORMATS[base];
    return Array.isArray(compatible) && compatible.includes(vf);
  }

  function _normalizeViewFormatsWithMeta(rawViewFormats, baseFormat, options = {}) {
    if (rawViewFormats === undefined || rawViewFormats === null) {
      return {
        viewFormats: EMPTY_VIEW_FORMATS,
        formatNormalized: false,
        viewFormatsNormalized: false,
      };
    }
    if (
      typeof rawViewFormats === 'string' ||
      typeof rawViewFormats[Symbol.iterator] !== 'function'
    ) {
      throw new TypeError('desc.viewFormats must be an iterable of WebGPU GPUTextureFormat strings');
    }

    const filterIncompatible = Boolean(options.filterIncompatible);
    const compatibleList = COMPATIBLE_VIEW_FORMATS[baseFormat] || EMPTY_VIEW_FORMATS;
    const unique = new Set();
    let formatNormalized = false;
    let viewFormatsNormalized = false;
    let seenCount = 0;

    for (const rawFmt of rawViewFormats) {
      seenCount += 1;
      const fmtRes = _normalizeFormatWithMeta(rawFmt);
      if (fmtRes.normalized) {
        formatNormalized = true;
      }
      const vf = fmtRes.format;
      if (vf === baseFormat) {
        // WebGPU always allows creating a view with the texture's own base format.
        viewFormatsNormalized = true;
        continue;
      }
      if (!compatibleList.includes(vf)) {
        if (filterIncompatible) {
          viewFormatsNormalized = true;
          continue;
        }
        throw new RangeError(
          `Incompatible viewFormat "${vf}" for base texture format "${baseFormat}"`
        );
      }
      if (unique.has(vf)) {
        viewFormatsNormalized = true;
        continue;
      }
      unique.add(vf);
    }

    if (unique.size === 0) {
      return {
        viewFormats: EMPTY_VIEW_FORMATS,
        formatNormalized,
        viewFormatsNormalized: viewFormatsNormalized || seenCount > 0,
      };
    }

    const sorted = Array.from(unique).sort();
    return {
      viewFormats: Object.freeze(sorted),
      formatNormalized,
      viewFormatsNormalized,
    };
  }

  /**
   * Normalize, deduplicate, and canonically sort `viewFormats` for a base texture format.
   *
   * @param {Iterable<string> | null | undefined} viewFormats
   * @param {string} baseFormat
   * @param {{ filterIncompatible?: boolean }} [options]
   * @returns {readonly string[]}
   */
  function normalizeViewFormats(viewFormats, baseFormat, options = {}) {
    const canonicalBase = _normalizeFormatWithMeta(baseFormat).format;
    return _normalizeViewFormatsWithMeta(viewFormats, canonicalBase, options).viewFormats;
  }

  /**
   * Normalize a requested `sampleCount` to WebGPU's valid set `{1, 4}`, falling
   * back to `1` when texture constraints disallow multisampling.
   *
   * @param {number | undefined | null} requestedSampleCount
   * @param {{
   *   mipLevelCount?: number,
   *   dimension?: string,
   *   depthOrArrayLayers?: number,
   *   usage?: number,
   * }} [context]
   * @returns {{ sampleCount: number, normalized: boolean }}
   */
  function normalizeSampleCount(requestedSampleCount, context = {}) {
    if (requestedSampleCount === undefined || requestedSampleCount === null) {
      return { sampleCount: 1, normalized: false };
    }
    const numeric = Number(requestedSampleCount);
    if (!Number.isFinite(numeric)) {
      return { sampleCount: 1, normalized: true };
    }

    const mipLevelCount = context.mipLevelCount !== undefined ? context.mipLevelCount : 1;
    const dimension = context.dimension !== undefined ? context.dimension : '2d';
    const depthOrArrayLayers =
      context.depthOrArrayLayers !== undefined ? context.depthOrArrayLayers : 1;
    const usage = context.usage !== undefined ? context.usage : 0;

    const multisampleAllowed =
      mipLevelCount <= 1 &&
      dimension === '2d' &&
      depthOrArrayLayers === 1 &&
      (usage & TEXTURE_USAGE.STORAGE_BINDING) === 0;

    const resolved = numeric >= 4 && multisampleAllowed ? 4 : 1;
    return {
      sampleCount: resolved,
      normalized: resolved !== requestedSampleCount,
    };
  }

  function extractRawDimensions(desc) {
    let width = desc.width;
    let height = desc.height;
    let depthOrArrayLayers = desc.depthOrArrayLayers;
    if (depthOrArrayLayers === undefined && desc.depthOrLayers !== undefined) {
      depthOrArrayLayers = desc.depthOrLayers;
    }

    if (desc.size !== undefined && desc.size !== null) {
      if (Array.isArray(desc.size)) {
        if (width === undefined) {
          width = desc.size[0];
        }
        if (height === undefined) {
          height = desc.size[1];
        }
        if (depthOrArrayLayers === undefined && desc.size[2] !== undefined) {
          depthOrArrayLayers = desc.size[2];
        }
      } else if (typeof desc.size === 'object') {
        if (width === undefined) {
          width = desc.size.width;
        }
        if (height === undefined) {
          height = desc.size.height;
        }
        if (depthOrArrayLayers === undefined && desc.size.depthOrArrayLayers !== undefined) {
          depthOrArrayLayers = desc.size.depthOrArrayLayers;
        }
      } else {
        throw new TypeError('desc.size must be an object { width, height, depthOrArrayLayers } or array');
      }
    }

    return { width, height, depthOrArrayLayers };
  }

  /**
   * Sanitize a render-target descriptor for WebGPU device safety:
   * - Clamps dimensions to `[1, maxTextureDimension2D]` (default `8192`).
   * - Normalizes DXGI / Xbox render-target & depth-stencil formats to canonical WebGPU strings.
   * - Clamps `mipLevelCount` to the valid chain depth for the clamped dimensions.
   * - Normalizes `sampleCount` to `{1, 4}` respecting WebGPU multisampling constraints.
   *
   * @param {object} desc
   * @param {{ device?: object | null, maxTextureDimension2D?: number }} [options]
   * @returns {{
   *   desc: Readonly<{
   *     width: number,
   *     height: number,
   *     format: string,
   *     usage: number,
   *     sampleCount: number,
   *     mipLevelCount: number,
   *     depthOrArrayLayers: number,
   *     dimension: '1d' | '2d' | '3d',
   *   }>,
   *   dimensionClamped: boolean,
   *   formatNormalized: boolean,
   *   sampleCountNormalized: boolean,
   * }}
   */
  function sanitizeRenderTargetDesc(desc, options = {}) {
    if (!desc || typeof desc !== 'object') {
      throw new TypeError('Render target descriptor must be an object');
    }

    const rawDims = extractRawDimensions(desc);

    const dimension = desc.dimension === undefined ? '2d' : String(desc.dimension).trim().toLowerCase();
    if (!VALID_DIMENSIONS.includes(dimension)) {
      throw new RangeError(
        `Unsupported texture dimension "${desc.dimension}"; expected one of: ${VALID_DIMENSIONS.join(', ')}`
      );
    }

    const limits = resolveDeviceTextureLimits(options.device, options.maxTextureDimension2D);
    const maxWidth =
      dimension === '1d' ? limits.maxDim1D : dimension === '3d' ? limits.maxDim3D : limits.maxDim2D;
    const maxHeight =
      dimension === '1d' ? 1 : dimension === '3d' ? limits.maxDim3D : limits.maxDim2D;
    const maxDepth =
      dimension === '1d' ? 1 : dimension === '3d' ? limits.maxDim3D : limits.maxArrayLayers;

    const widthResult = clampDimensionValue(rawDims.width, maxWidth);

    let heightResult;
    if (dimension === '1d' && rawDims.height === undefined) {
      heightResult = { value: 1, clamped: false };
    } else {
      heightResult = clampDimensionValue(rawDims.height, maxHeight);
    }

    let depthResult;
    if (rawDims.depthOrArrayLayers === undefined) {
      depthResult = { value: 1, clamped: false };
    } else {
      depthResult = clampDimensionValue(rawDims.depthOrArrayLayers, maxDepth);
    }

    const dimensionClamped =
      widthResult.clamped || heightResult.clamped || depthResult.clamped;

    const usage = desc.usage === undefined ? DEFAULT_TEXTURE_USAGE : desc.usage;
    assertPositiveInteger(usage, 'usage');

    const formatResult = _normalizeFormatWithMeta(desc.format);
    const deviceFormatResult = resolveRenderTargetFormatForDevice(formatResult.format, {
      device: options.device,
      deviceFeatures: options.deviceFeatures,
      usage,
      requireFeatureSupport: options.requireFeatureSupport,
    });
    const resolvedFormat = deviceFormatResult.format;

    let rawViewFormats = desc.viewFormats;
    const includeSrgbSibling = Boolean(desc.includeSrgbSibling || options.includeSrgbSibling);
    if (includeSrgbSibling) {
      const sibling = getCompatibleSrgbSiblingFormat(resolvedFormat);
      if (sibling) {
        if (rawViewFormats === undefined || rawViewFormats === null) {
          rawViewFormats = [sibling];
        } else if (
          typeof rawViewFormats !== 'string' &&
          typeof rawViewFormats[Symbol.iterator] === 'function'
        ) {
          const list = Array.from(rawViewFormats);
          if (!list.includes(sibling)) {
            list.push(sibling);
          }
          rawViewFormats = list;
        }
      }
    }

    const viewFormatsResult = _normalizeViewFormatsWithMeta(rawViewFormats, resolvedFormat, {
      filterIncompatible: true,
    });

    const maxExtent =
      dimension === '1d'
        ? widthResult.value
        : dimension === '3d'
          ? Math.max(widthResult.value, heightResult.value, depthResult.value)
          : Math.max(widthResult.value, heightResult.value);
    const maxMips = Math.floor(Math.log2(maxExtent)) + 1;

    let mipLevelCount;
    if (desc.mipLevelCount === undefined) {
      mipLevelCount = 1;
    } else if (
      typeof desc.mipLevelCount !== 'number' ||
      !Number.isFinite(desc.mipLevelCount) ||
      Math.floor(desc.mipLevelCount) < 1
    ) {
      mipLevelCount = 1;
    } else {
      mipLevelCount = Math.min(Math.floor(desc.mipLevelCount), maxMips);
    }

    const sampleResult = normalizeSampleCount(desc.sampleCount, {
      mipLevelCount,
      dimension,
      depthOrArrayLayers: depthResult.value,
      usage,
    });

    return {
      desc: Object.freeze({
        width: widthResult.value,
        height: heightResult.value,
        format: resolvedFormat,
        usage,
        sampleCount: sampleResult.sampleCount,
        mipLevelCount,
        depthOrArrayLayers: depthResult.value,
        dimension,
        viewFormats: viewFormatsResult.viewFormats,
      }),
      dimensionClamped,
      formatNormalized:
        formatResult.normalized ||
        viewFormatsResult.formatNormalized ||
        deviceFormatResult.formatFallback,
      formatFallback: deviceFormatResult.formatFallback,
      requestedFormat: deviceFormatResult.requestedFormat,
      fallbackReason: deviceFormatResult.fallbackReason,
      sampleCountNormalized: sampleResult.normalized,
      viewFormatsNormalized: viewFormatsResult.viewFormatsNormalized,
    };
  }

  /**
   * Normalize a render-target descriptor (accepting flat `{ width, height, ... }` or
   * WebGPU `{ size: { width, height, depthOrArrayLayers } | [w, h, d], ... }` shapes)
   * into a canonical frozen descriptor object.
   *
   * When `options.sanitize` (or `options.clampDimensions` / `options.normalizeSampleCount`)
   * is truthy, delegates to `sanitizeRenderTargetDesc(desc, options)`.
   *
   * @param {object} desc
   * @param {{ sanitize?: boolean, clampDimensions?: boolean, normalizeSampleCount?: boolean, device?: object | null, maxTextureDimension2D?: number }} [options]
   * @returns {{
   *   width: number,
   *   height: number,
   *   format: string,
   *   usage: number,
   *   sampleCount: number,
   *   mipLevelCount: number,
   *   depthOrArrayLayers: number,
   *   dimension: '1d' | '2d' | '3d',
   * }}
   */
  function normalizeRenderTargetDesc(desc, options) {
    if (
      options &&
      typeof options === 'object' &&
      (options.sanitize || options.clampDimensions || options.normalizeSampleCount)
    ) {
      return sanitizeRenderTargetDesc(desc, options).desc;
    }

    if (!desc || typeof desc !== 'object') {
      throw new TypeError('Render target descriptor must be an object');
    }

    let { width, height, depthOrArrayLayers } = extractRawDimensions(desc);

    const dimension = desc.dimension === undefined ? '2d' : String(desc.dimension).trim().toLowerCase();
    if (!VALID_DIMENSIONS.includes(dimension)) {
      throw new RangeError(
        `Unsupported texture dimension "${desc.dimension}"; expected one of: ${VALID_DIMENSIONS.join(', ')}`
      );
    }

    if (dimension === '1d' && height === undefined) {
      height = 1;
    }
    if (depthOrArrayLayers === undefined) {
      depthOrArrayLayers = 1;
    }

    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');
    assertPositiveInteger(depthOrArrayLayers, 'depthOrArrayLayers');

    const format = _normalizeFormatWithMeta(desc.format).format;
    let rawViewFormats = desc.viewFormats;
    if (desc.includeSrgbSibling === true) {
      const sibling = getCompatibleSrgbSiblingFormat(format);
      if (sibling) {
        if (rawViewFormats === undefined || rawViewFormats === null) {
          rawViewFormats = [sibling];
        } else if (
          typeof rawViewFormats !== 'string' &&
          typeof rawViewFormats[Symbol.iterator] === 'function'
        ) {
          const list = Array.from(rawViewFormats);
          if (!list.includes(sibling)) {
            list.push(sibling);
          }
          rawViewFormats = list;
        }
      }
    }
    const viewFormats = _normalizeViewFormatsWithMeta(rawViewFormats, format, {
      filterIncompatible: false,
    }).viewFormats;

    const usage = desc.usage === undefined ? DEFAULT_TEXTURE_USAGE : desc.usage;
    assertPositiveInteger(usage, 'usage');

    const sampleCount = desc.sampleCount === undefined ? 1 : desc.sampleCount;
    if (sampleCount !== 1 && sampleCount !== 4) {
      throw new RangeError(`sampleCount must be 1 or 4, got ${sampleCount}`);
    }

    const mipLevelCount = desc.mipLevelCount === undefined ? 1 : desc.mipLevelCount;
    assertPositiveInteger(mipLevelCount, 'mipLevelCount');

    return Object.freeze({
      width,
      height,
      format,
      usage,
      sampleCount,
      mipLevelCount,
      depthOrArrayLayers,
      dimension,
      viewFormats,
    });
  }

  /**
   * Build a canonical deterministic pool key from a render-target descriptor.
   *
   * @param {object} desc
   * @returns {string}
   */
  function makeRenderTargetKey(desc) {
    const n = normalizeRenderTargetDesc(desc);
    const base =
      `${n.dimension}|${n.width}x${n.height}x${n.depthOrArrayLayers}|` +
      `${n.format}|u${n.usage}|s${n.sampleCount}|m${n.mipLevelCount}`;
    return n.viewFormats.length > 0 ? `${base}|vf${n.viewFormats.join(',')}` : base;
  }

  /**
   * Build a canonical deterministic cache key and WebGPU `GPUTextureViewDescriptor`
   * for a texture view request. Default/empty descriptors collapse to `{ isDefault: true }`
   * so callers reuse the cached default `GPUTextureView` with zero extra `createView()` calls.
   *
   * @param {object | null | undefined} viewDesc
   * @param {string} [baseFormat]
   * @returns {{ isDefault: boolean, key: string, descriptor: object | null }}
   */
  function makeTextureViewKey(viewDesc, baseFormat, textureDesc) {
    if (viewDesc === undefined || viewDesc === null) {
      return { isDefault: true, key: 'default', descriptor: null };
    }
    if (typeof viewDesc !== 'object' || Array.isArray(viewDesc)) {
      throw new TypeError('viewDesc must be an object when provided');
    }

    const canonicalBase =
      typeof baseFormat === 'string' && baseFormat.trim().length > 0
        ? _normalizeFormatWithMeta(baseFormat).format
        : null;

    let format = null;
    if (viewDesc.format !== undefined && viewDesc.format !== null) {
      const normalizedFormat = _normalizeFormatWithMeta(viewDesc.format).format;
      if (canonicalBase === null || normalizedFormat !== canonicalBase) {
        format = normalizedFormat;
      }
    }

    let dimension = null;
    if (viewDesc.dimension !== undefined && viewDesc.dimension !== null) {
      dimension = String(viewDesc.dimension).trim().toLowerCase();
      if (!VALID_VIEW_DIMENSIONS.includes(dimension)) {
        throw new RangeError(
          `Unsupported texture view dimension "${viewDesc.dimension}"; expected one of: ${VALID_VIEW_DIMENSIONS.join(', ')}`
        );
      }
    }

    let aspect = 'all';
    if (viewDesc.aspect !== undefined && viewDesc.aspect !== null) {
      aspect = String(viewDesc.aspect).trim().toLowerCase();
      if (!VALID_VIEW_ASPECTS.includes(aspect)) {
        throw new RangeError(
          `Unsupported texture view aspect "${viewDesc.aspect}"; expected one of: ${VALID_VIEW_ASPECTS.join(', ')}`
        );
      }
    }

    const baseMipLevel = viewDesc.baseMipLevel === undefined ? 0 : viewDesc.baseMipLevel;
    assertNonNegativeInteger(baseMipLevel, 'baseMipLevel');

    let mipLevelCount;
    if (viewDesc.mipLevelCount !== undefined) {
      assertPositiveInteger(viewDesc.mipLevelCount, 'mipLevelCount');
      mipLevelCount = viewDesc.mipLevelCount;
    }

    const baseArrayLayer = viewDesc.baseArrayLayer === undefined ? 0 : viewDesc.baseArrayLayer;
    assertNonNegativeInteger(baseArrayLayer, 'baseArrayLayer');

    let arrayLayerCount;
    if (viewDesc.arrayLayerCount !== undefined) {
      assertPositiveInteger(viewDesc.arrayLayerCount, 'arrayLayerCount');
      arrayLayerCount = viewDesc.arrayLayerCount;
    }

    const usage = viewDesc.usage === undefined ? 0 : viewDesc.usage;
    assertNonNegativeInteger(usage, 'usage');

    // Canonicalize equivalent full-range / default-dimension descriptors against the owning
    // texture so e.g. `{ dimension: 'cube' }` and `{ dimension: 'cube', baseArrayLayer: 0, arrayLayerCount: 6 }`
    // share the exact same cached GPUTextureView without creating duplicate GPU views.
    if (textureDesc && typeof textureDesc === 'object') {
      const texDim = textureDesc.dimension || '2d';
      const texMips = textureDesc.mipLevelCount || 1;
      const texLayers = textureDesc.depthOrArrayLayers || 1;
      const defaultViewDim =
        texDim === '2d' && texLayers > 1 ? '2d-array' : texDim;

      if (dimension === defaultViewDim) {
        dimension = null;
      }
      if (mipLevelCount !== undefined && baseMipLevel === 0 && mipLevelCount === texMips) {
        mipLevelCount = undefined;
      }
      if (
        arrayLayerCount !== undefined &&
        baseArrayLayer === 0 &&
        texDim !== '3d' &&
        arrayLayerCount === texLayers
      ) {
        // Only collapse full layer count when the effective view dimension covers all layers
        // (i.e. not a single-layer '2d' view of a multi-layer texture, which can't happen here
        // anyway since arrayLayerCount === texLayers > 1 wouldn't be '2d').
        if (dimension !== '2d' || texLayers === 1) {
          arrayLayerCount = undefined;
        }
      }
    }

    const isDefault =
      format === null &&
      dimension === null &&
      aspect === 'all' &&
      baseMipLevel === 0 &&
      mipLevelCount === undefined &&
      baseArrayLayer === 0 &&
      arrayLayerCount === undefined &&
      usage === 0;

    if (isDefault) {
      return { isDefault: true, key: 'default', descriptor: null };
    }

    const descriptor = {};
    if (format !== null) {
      descriptor.format = format;
    }
    if (dimension !== null) {
      descriptor.dimension = dimension;
    }
    if (aspect !== 'all') {
      descriptor.aspect = aspect;
    }
    if (baseMipLevel !== 0) {
      descriptor.baseMipLevel = baseMipLevel;
    }
    if (mipLevelCount !== undefined) {
      descriptor.mipLevelCount = mipLevelCount;
    }
    if (baseArrayLayer !== 0) {
      descriptor.baseArrayLayer = baseArrayLayer;
    }
    if (arrayLayerCount !== undefined) {
      descriptor.arrayLayerCount = arrayLayerCount;
    }
    if (usage !== 0) {
      descriptor.usage = usage;
    }
    if (typeof viewDesc.label === 'string' && viewDesc.label.length > 0) {
      descriptor.label = viewDesc.label;
    }

    const key =
      `f=${format || ''}|d=${dimension || ''}|a=${aspect}|` +
      `bm=${baseMipLevel}|mc=${mipLevelCount === undefined ? '' : mipLevelCount}|` +
      `bl=${baseArrayLayer}|lc=${arrayLayerCount === undefined ? '' : arrayLayerCount}|u=${usage}`;

    return { isDefault: false, key, descriptor };
  }

  /**
   * Estimate VRAM footprint in bytes for a WebGPU render target descriptor,
   * accounting for format texel size, full mip chain, array layers / 3D depth,
   * and MSAA `sampleCount`.
   *
   * @param {object} desc
   * @returns {number}
   */
  function estimateRenderTargetBytes(desc) {
    const n = normalizeRenderTargetDesc(desc);
    if (!Object.prototype.hasOwnProperty.call(FORMAT_BYTES_PER_PIXEL, n.format)) {
      throw new RangeError(`Unsupported render target format for byte estimation: "${n.format}"`);
    }

    const bytesPerPixel = FORMAT_BYTES_PER_PIXEL[n.format];
    const is1D = n.dimension === '1d';
    const is3D = n.dimension === '3d';

    let totalBytes = 0;
    for (let mip = 0; mip < n.mipLevelCount; mip += 1) {
      const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
      const mipWidth = scale === Infinity ? 1 : Math.max(1, Math.floor(n.width / scale));
      const mipHeight = is1D
        ? 1
        : scale === Infinity
          ? 1
          : Math.max(1, Math.floor(n.height / scale));
      const mipDepth = is3D
        ? scale === Infinity
          ? 1
          : Math.max(1, Math.floor(n.depthOrArrayLayers / scale))
        : n.depthOrArrayLayers;

      totalBytes += mipWidth * mipHeight * mipDepth * bytesPerPixel * n.sampleCount;
    }

    return totalBytes;
  }

  /**
   * WebGPU transient render-target texture pool & per-frame attachment aliasing manager.
   *
   * - Pools `GPUTexture` + cached default `GPUTextureView` objects keyed by normalized
   *   `(width, height, format, usage, sampleCount, mipLevelCount, depthOrArrayLayers, dimension)`.
   * - Supports immediate same-frame attachment aliasing once `release()` is called, plus
   *   cross-frame reuse without `device.createTexture()` churn.
   * - Enforces `maxPoolBytes` via global LRU eviction of idle targets and expires unused
   *   targets after `idleFrameTtl` frames in `endFrame()`.
   */
  class RenderTargetPool {
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('RenderTargetPool constructor expects an options object');
      }

      const {
        device,
        perfProfile: rawPerfProfile,
        maxPoolBytes: rawMaxPoolBytes,
        maxVramBytes,
        idleFrameTtl: rawIdleFrameTtl,
        maxIdleFrames,
        oversizedTargetBytes = DEFAULT_OVERSIZED_TARGET_BYTES,
        oversizedIdleFrameTtl = null,
        strictBudget = true,
        autoReleaseLeaked = true,
        autoIncludeSrgbSibling = false,
        deviceFeatures = null,
        onLeak = null,
        warn = null,
      } = options;

      const resolvedPerfProfile =
        rawPerfProfile !== undefined && rawPerfProfile !== null
          ? _coercePerfProfile(rawPerfProfile)
          : null;

      const maxPoolBytes =
        rawMaxPoolBytes !== undefined
          ? rawMaxPoolBytes
          : maxVramBytes !== undefined
            ? maxVramBytes
            : resolvedPerfProfile && resolvedPerfProfile.rtPoolMaxBytes !== undefined
              ? resolvedPerfProfile.rtPoolMaxBytes
              : DEFAULT_MAX_POOL_BYTES;
      const idleFrameTtl =
        rawIdleFrameTtl !== undefined
          ? rawIdleFrameTtl
          : maxIdleFrames !== undefined
            ? maxIdleFrames
            : resolvedPerfProfile && resolvedPerfProfile.rtPoolMaxIdleFrames !== undefined
              ? resolvedPerfProfile.rtPoolMaxIdleFrames
              : DEFAULT_IDLE_FRAME_TTL;

      if (!device || typeof device.createTexture !== 'function') {
        throw new TypeError('RenderTargetPool requires a WebGPU device with createTexture()');
      }

      if (maxPoolBytes !== Infinity && (!Number.isInteger(maxPoolBytes) || maxPoolBytes < 1)) {
        throw new RangeError(
          `maxPoolBytes must be a positive integer or Infinity, got ${maxPoolBytes}`
        );
      }
      assertNonNegativeInteger(idleFrameTtl, 'idleFrameTtl');

      if (
        oversizedTargetBytes !== Infinity &&
        (!Number.isInteger(oversizedTargetBytes) || oversizedTargetBytes < 1)
      ) {
        throw new RangeError(
          `oversizedTargetBytes must be a positive integer or Infinity, got ${oversizedTargetBytes}`
        );
      }
      if (oversizedIdleFrameTtl !== null && oversizedIdleFrameTtl !== undefined) {
        assertNonNegativeInteger(oversizedIdleFrameTtl, 'oversizedIdleFrameTtl');
      }

      if (onLeak !== null && typeof onLeak !== 'function') {
        throw new TypeError('onLeak must be a function when provided');
      }
      if (warn !== null && typeof warn !== 'function') {
        throw new TypeError('warn must be a function when provided');
      }

      this._device = device;
      this._perfProfile = resolvedPerfProfile;
      this._perfProfileName = resolvedPerfProfile ? resolvedPerfProfile.name || null : null;
      this._maxPoolBytes = maxPoolBytes;
      this._idleFrameTtl = idleFrameTtl;
      this._oversizedTargetBytes = oversizedTargetBytes;
      this._oversizedIdleFrameTtl =
        oversizedIdleFrameTtl === undefined ? null : oversizedIdleFrameTtl;
      this._strictBudget = Boolean(strictBudget);
      this._autoReleaseLeaked = Boolean(autoReleaseLeaked);
      this._autoIncludeSrgbSibling = Boolean(autoIncludeSrgbSibling);
      this._deviceFeatures = deviceFeatures || null;
      this._onLeak = onLeak;
      this._warn = warn;

      this._destroyed = false;
      this._currentFrame = 0;
      this._nextId = 1;
      this._nextLeaseToken = 1;

      /** @type {Map<number, object>} All live (active + idle) records by id */
      this._allEntries = new Map();
      /** @type {Map<number, object>} Active in-use records by id */
      this._activeEntries = new Map();
      /** @type {Map<string, object[]>} Idle records grouped by normalized key (MRU stack) */
      this._idleBuckets = new Map();
      /** @type {Map<number, object>} Idle records in insertion order (oldest = LRU) */
      this._idleLru = new Map();
      /** @type {Set<object>} Idle records released via releaseTransient() during the current frame */
      this._transientAliases = new Set();
      /** @type {WeakMap<object, { id: number, leaseToken: number }>} Per-acquire handle tokens */
      this._handleTokens = new WeakMap();
      /** @type {WeakSet<object>} Tracks GPUTexture instances already destroyed to guarantee no double-destroy */
      this._destroyedTextures = new WeakSet();

      // Resolved device dimension cap (cached for post-destroy telemetry safety)
      this._maxTextureDimension2D = resolveMaxTextureDimension2D(device);

      // Telemetry counters
      this._hits = 0;
      this._misses = 0;
      this._transientReuseCount = 0;
      this._intraFrameAliasHits = 0;
      this._viewsCached = 0;
      this._viewCacheHits = 0;
      this._viewCacheMisses = 0;
      this._activeBytes = 0;
      this._idleBytes = 0;
      this._peakBytes = 0;
      this._evictions = 0;
      this._budgetEvictions = 0;
      this._idleEvictions = 0;
      this._oversizedEvictions = 0;
      this._reclaimedOversizedBytes = 0;
      this._leakedReleases = 0;
      this._dimensionClampCount = 0;
      this._formatNormalizeCount = 0;
      this._formatFallbackCount = 0;
      this._sampleCountNormalizeCount = 0;
      this._hdrAcquires = 0;
      this._depthAcquires = 0;
      this._srgbAcquires = 0;
    }

    get isDestroyed() {
      return this._destroyed;
    }

    get maxPoolBytes() {
      return this._maxPoolBytes;
    }

    get idleFrameTtl() {
      return this._idleFrameTtl;
    }

    get oversizedTargetBytes() {
      return this._oversizedTargetBytes;
    }

    get oversizedIdleFrameTtl() {
      return this._oversizedIdleFrameTtl;
    }

    get currentFrame() {
      return this._currentFrame;
    }

    get maxTextureDimension2D() {
      return this._effectiveMaxTextureDimension2D();
    }

    get perfProfile() {
      return this._perfProfile;
    }

    _effectiveMaxTextureDimension2D() {
      if (this._device) {
        this._maxTextureDimension2D = resolveMaxTextureDimension2D(this._device);
      }
      return this._maxTextureDimension2D;
    }

    _assertNotDestroyed() {
      if (this._destroyed) {
        throw new Error('RenderTargetPool has been destroyed');
      }
    }

    _removeFromIdleBucket(record) {
      const bucket = this._idleBuckets.get(record.key);
      if (!bucket) {
        return;
      }
      const idx = bucket.indexOf(record);
      if (idx !== -1) {
        bucket.splice(idx, 1);
      }
      if (bucket.length === 0) {
        this._idleBuckets.delete(record.key);
      }
    }

    _destroyTextureSafely(texture) {
      if (!texture || typeof texture.destroy !== 'function') {
        return false;
      }
      if (texture.destroyed === true || this._destroyedTextures.has(texture)) {
        return false;
      }
      this._destroyedTextures.add(texture);
      try {
        texture.destroy();
      } catch (_) {
        // Ignore device-lost / already-destroyed WebGPU errors during teardown.
      }
      return true;
    }

    _isRecordTextureAlive(record) {
      if (!record || record.destroyed || !record.texture) {
        return false;
      }
      if (record.texture.destroyed === true || this._destroyedTextures.has(record.texture)) {
        return false;
      }
      if (record.hasView && !record.view) {
        return false;
      }
      return true;
    }

    _isRecordReusable(record) {
      return this._isRecordTextureAlive(record) && !record.retireOnRelease;
    }

    _destroyRecord(record) {
      if (!record || record.destroyed) {
        return false;
      }
      record.destroyed = true;
      record.active = false;
      record.retireOnRelease = false;
      record.leaseToken = 0;
      record.releasedFrame = -1;
      record.transientAliasFrame = -1;
      this._transientAliases.delete(record);
      this._allEntries.delete(record.id);

      const removedViews =
        (record.view !== null ? 1 : 0) + (record.viewCache ? record.viewCache.size : 0);
      if (removedViews > 0) {
        this._viewsCached = Math.max(0, this._viewsCached - removedViews);
      }

      const tex = record.texture;
      record.texture = null;
      record.view = null;
      if (record.viewCache) {
        record.viewCache.clear();
      }
      this._destroyTextureSafely(tex);
      return true;
    }

    _getOrCreateRecordView(record, viewDesc) {
      if (!this._isRecordTextureAlive(record)) {
        return null;
      }
      const normalized = makeTextureViewKey(viewDesc, record.desc.format, record.desc);
      if (normalized.isDefault) {
        if (record.view !== null) {
          this._viewCacheHits += 1;
        }
        return record.view;
      }
      if (!record.viewCache) {
        record.viewCache = new Map();
      }
      if (record.viewCache.has(normalized.key)) {
        this._viewCacheHits += 1;
        return record.viewCache.get(normalized.key);
      }
      if (typeof record.texture.createView !== 'function') {
        return null;
      }
      const subView = record.texture.createView(normalized.descriptor);
      if (subView === null || subView === undefined) {
        return null;
      }
      record.viewCache.set(normalized.key, subView);
      this._viewsCached += 1;
      this._viewCacheMisses += 1;
      return subView;
    }

    _resolveEntryId(entryOrId, methodName) {
      if (entryOrId === null || entryOrId === undefined) {
        throw new TypeError(`${methodName}() requires an acquired entry object or numeric id`);
      }

      if (typeof entryOrId === 'number') {
        if (!Number.isInteger(entryOrId) || entryOrId < 1) {
          throw new TypeError(`${methodName}() id must be a positive integer, got ${entryOrId}`);
        }
        return { id: entryOrId, expectedLeaseToken: null };
      }
      if (typeof entryOrId === 'object') {
        const tokenInfo = this._handleTokens.get(entryOrId);
        if (tokenInfo) {
          return { id: tokenInfo.id, expectedLeaseToken: tokenInfo.leaseToken };
        }
        if (Number.isInteger(entryOrId.id) && entryOrId.id >= 1) {
          return { id: entryOrId.id, expectedLeaseToken: null };
        }
        throw new TypeError(`${methodName}() entry object must have a valid positive integer id`);
      }
      throw new TypeError(`${methodName}() requires an acquired entry object or numeric id`);
    }

    _isOversizedRecord(record) {
      return Boolean(record && record.byteSize >= this._oversizedTargetBytes);
    }

    _effectiveIdleTtlForRecord(record) {
      if (
        this._oversizedIdleFrameTtl !== null &&
        this._isOversizedRecord(record)
      ) {
        return Math.min(this._idleFrameTtl, this._oversizedIdleFrameTtl);
      }
      return this._idleFrameTtl;
    }

    _evictIdleRecord(record, reason = null) {
      this._idleLru.delete(record.id);
      this._removeFromIdleBucket(record);
      this._idleBytes = Math.max(0, this._idleBytes - record.byteSize);
      if (this._destroyRecord(record)) {
        this._evictions += 1;
        if (this._isOversizedRecord(record)) {
          this._oversizedEvictions += 1;
        }
        if (reason === 'budget') {
          this._budgetEvictions += 1;
        } else if (reason === 'idle') {
          this._idleEvictions += 1;
        }
        return true;
      }
      return false;
    }

    _evictIdleToBudget(targetBytes = this._maxPoolBytes, reserveBytes = 0) {
      let evictedCount = 0;
      let evictedBytes = 0;
      while (
        this._activeBytes + this._idleBytes + reserveBytes > targetBytes &&
        this._idleLru.size > 0
      ) {
        const oldest = this._idleLru.values().next().value;
        const bytes = oldest.byteSize;
        if (this._evictIdleRecord(oldest, 'budget')) {
          evictedCount += 1;
          evictedBytes += bytes;
        }
      }
      return { evictedCount, evictedBytes };
    }

    /**
     * Reconfigure the pool's GPU memory budget (`maxPoolBytes`) at runtime and
     * LRU-evict excess idle targets if current pooled bytes exceed the new budget.
     * Never destroys currently checked-out/active targets.
     *
     * @param {number} maxPoolBytes Positive integer or Infinity.
     * @returns {{ evictedCount: number, evictedBytes: number }}
     */
    setMaxPoolBytes(maxPoolBytes) {
      this._assertNotDestroyed();
      if (maxPoolBytes !== Infinity && (!Number.isInteger(maxPoolBytes) || maxPoolBytes < 1)) {
        throw new RangeError(
          `maxPoolBytes must be a positive integer or Infinity, got ${maxPoolBytes}`
        );
      }
      this._maxPoolBytes = maxPoolBytes;
      return this._evictIdleToBudget(this._maxPoolBytes, 0);
    }

    /**
     * Reconfigure the pool's idle-frame expiration threshold (`idleFrameTtl`) at runtime
     * and immediately evict any idle targets already past the new TTL.
     * Never destroys currently checked-out/active targets.
     *
     * @param {number} idleFrameTtl Non-negative integer frame threshold.
     * @returns {{ evictedCount: number, evictedBytes: number }}
     */
    setIdleFrameTtl(idleFrameTtl) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(idleFrameTtl, 'idleFrameTtl');
      this._idleFrameTtl = idleFrameTtl;

      let evictedCount = 0;
      let evictedBytes = 0;
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        const effectiveTtl = this._effectiveIdleTtlForRecord(record);
        if (this._currentFrame - record.lastUsedFrame > effectiveTtl) {
          const bytes = record.byteSize;
          if (this._evictIdleRecord(record, 'idle')) {
            evictedCount += 1;
            evictedBytes += bytes;
          }
        }
      }
      return { evictedCount, evictedBytes };
    }

    /**
     * Apply a hardware-adaptive performance profile (`'low' | 'balanced' | 'high' | 'auto'`
     * or a resolved profile object from `PlayGta6PerfProfile.resolvePerfProfile`), updating
     * `maxPoolBytes` and `idleFrameTtl` and immediately evicting excess idle targets.
     * Never destroys currently checked-out/active targets.
     *
     * @param {string | object} profileOrName
     * @returns {{
     *   profile: string | null,
     *   maxPoolBytes: number,
     *   idleFrameTtl: number,
     *   evictedCount: number,
     *   evictedBytes: number,
     *   budgetEvictedCount: number,
     *   idleEvictedCount: number,
     * }}
     */
    applyPerfProfile(profileOrName) {
      this._assertNotDestroyed();
      const profile = _coercePerfProfile(profileOrName);
      this._perfProfile = profile;
      this._perfProfileName = profile.name || null;

      let budgetEvict = { evictedCount: 0, evictedBytes: 0 };
      if (profile.rtPoolMaxBytes !== undefined) {
        budgetEvict = this.setMaxPoolBytes(profile.rtPoolMaxBytes);
      }

      let idleEvict = { evictedCount: 0, evictedBytes: 0 };
      if (profile.rtPoolMaxIdleFrames !== undefined) {
        idleEvict = this.setIdleFrameTtl(profile.rtPoolMaxIdleFrames);
      }

      return {
        profile: this._perfProfileName,
        maxPoolBytes: this._maxPoolBytes,
        idleFrameTtl: this._idleFrameTtl,
        evictedCount: budgetEvict.evictedCount + idleEvict.evictedCount,
        evictedBytes: budgetEvict.evictedBytes + idleEvict.evictedBytes,
        budgetEvictedCount: budgetEvict.evictedCount,
        idleEvictedCount: idleEvict.evictedCount,
      };
    }

    /**
     * Configure faster idle-frame eviction for oversized transient render targets
     * (e.g., unused 4K intro/scene targets during intro -> Start Screen -> menu transitions)
     * and immediately evict any idle oversized targets already past the new TTL.
     *
     * @param {{ oversizedTargetBytes?: number, oversizedIdleFrameTtl?: number | null }} [options]
     * @returns {{ evictedCount: number, evictedBytes: number }}
     */
    setOversizedEvictionPolicy(options = {}) {
      this._assertNotDestroyed();
      if (!options || typeof options !== 'object') {
        throw new TypeError('setOversizedEvictionPolicy() expects an options object');
      }
      if (options.oversizedTargetBytes !== undefined) {
        if (
          options.oversizedTargetBytes !== Infinity &&
          (!Number.isInteger(options.oversizedTargetBytes) || options.oversizedTargetBytes < 1)
        ) {
          throw new RangeError(
            `oversizedTargetBytes must be a positive integer or Infinity, got ${options.oversizedTargetBytes}`
          );
        }
        this._oversizedTargetBytes = options.oversizedTargetBytes;
      }
      if (options.oversizedIdleFrameTtl !== undefined) {
        if (options.oversizedIdleFrameTtl !== null) {
          assertNonNegativeInteger(options.oversizedIdleFrameTtl, 'oversizedIdleFrameTtl');
        }
        this._oversizedIdleFrameTtl = options.oversizedIdleFrameTtl;
      }

      let evictedCount = 0;
      let evictedBytes = 0;
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        const effectiveTtl = this._effectiveIdleTtlForRecord(record);
        if (this._currentFrame - record.lastUsedFrame > effectiveTtl) {
          const bytes = record.byteSize;
          if (this._evictIdleRecord(record, 'idle')) {
            evictedCount += 1;
            evictedBytes += bytes;
          }
        }
      }
      return { evictedCount, evictedBytes };
    }

    /**
     * LRU-evict idle targets until total pooled VRAM is at most `targetBytes`
     * without changing `maxPoolBytes` or destroying active targets.
     *
     * @param {number} [targetBytes=this._maxPoolBytes] Non-negative integer or Infinity.
     * @returns {{ evictedCount: number, evictedBytes: number }}
     */
    trimToBudget(targetBytes = this._maxPoolBytes) {
      this._assertNotDestroyed();
      if (targetBytes !== Infinity && (!Number.isInteger(targetBytes) || targetBytes < 0)) {
        throw new RangeError(
          `targetBytes must be a non-negative integer or Infinity, got ${targetBytes}`
        );
      }
      return this._evictIdleToBudget(targetBytes, 0);
    }

    _retireActiveRecord(record) {
      record.active = false;
      record.leaseToken = 0;
      this._activeEntries.delete(record.id);
      this._activeBytes = Math.max(0, this._activeBytes - record.byteSize);
      if (this._destroyRecord(record)) {
        this._evictions += 1;
        if (this._isOversizedRecord(record)) {
          this._oversizedEvictions += 1;
        }
        return true;
      }
      return false;
    }

    _moveActiveToIdle(record, frameIndex) {
      record.active = false;
      record.leaseToken = 0;
      record.lastUsedFrame = frameIndex;
      record.releasedFrame = frameIndex;
      record.transientAliasFrame = -1;
      this._transientAliases.delete(record);
      this._activeEntries.delete(record.id);
      this._activeBytes = Math.max(0, this._activeBytes - record.byteSize);
      this._idleBytes += record.byteSize;

      let bucket = this._idleBuckets.get(record.key);
      if (!bucket) {
        bucket = [];
        this._idleBuckets.set(record.key, bucket);
      }
      bucket.push(record);
      this._idleLru.set(record.id, record);
    }

    _isCompatibleTransientAlias(candidate, reqDesc) {
      if (!candidate || candidate.transientAliasFrame !== this._currentFrame) {
        return false;
      }
      const d = candidate.desc;
      if (
        d.dimension !== reqDesc.dimension ||
        d.width !== reqDesc.width ||
        d.height !== reqDesc.height ||
        d.depthOrArrayLayers !== reqDesc.depthOrArrayLayers ||
        d.format !== reqDesc.format ||
        d.sampleCount !== reqDesc.sampleCount ||
        d.mipLevelCount !== reqDesc.mipLevelCount
      ) {
        return false;
      }
      if ((d.usage & reqDesc.usage) !== reqDesc.usage) {
        return false;
      }
      if (reqDesc.viewFormats.length > 0) {
        for (let i = 0; i < reqDesc.viewFormats.length; i += 1) {
          if (!d.viewFormats.includes(reqDesc.viewFormats[i])) {
            return false;
          }
        }
      }
      return true;
    }

    _recordAcquireSanitizationAndClass(normalizedDesc, sanitized) {
      if (sanitized.dimensionClamped) {
        this._dimensionClampCount += 1;
      }
      if (sanitized.formatNormalized) {
        this._formatNormalizeCount += 1;
      }
      if (sanitized.formatFallback) {
        this._formatFallbackCount += 1;
      }
      if (sanitized.sampleCountNormalized) {
        this._sampleCountNormalizeCount += 1;
      }
      if (HDR_RENDER_TARGET_FORMATS.has(normalizedDesc.format)) {
        this._hdrAcquires += 1;
      }
      if (DEPTH_RENDER_TARGET_FORMATS.has(normalizedDesc.format)) {
        this._depthAcquires += 1;
      }
      if (
        SRGB_RENDER_TARGET_FORMATS.has(normalizedDesc.format) ||
        (normalizedDesc.viewFormats.length > 0 &&
          normalizedDesc.viewFormats.some((vf) => SRGB_RENDER_TARGET_FORMATS.has(vf)))
      ) {
        this._srgbAcquires += 1;
      }
    }

    _claimIdleRecord(record, label, sanitized, isIntraFrame) {
      this._transientAliases.delete(record);
      record.transientAliasFrame = -1;
      record.active = true;
      record.lastUsedFrame = this._currentFrame;
      if (typeof label === 'string' && label.length > 0) {
        record.label = label;
      }

      this._activeEntries.set(record.id, record);
      this._idleBytes = Math.max(0, this._idleBytes - record.byteSize);
      this._activeBytes += record.byteSize;
      this._hits += 1;
      this._transientReuseCount += 1;
      if (isIntraFrame) {
        this._intraFrameAliasHits += 1;
      }
      if (record.view !== null) {
        this._viewCacheHits += 1;
      }

      this._recordAcquireSanitizationAndClass(record.desc, sanitized);

      return this._makeHandle(record, true);
    }

    _makeHandle(record, reused) {
      const leaseToken = this._nextLeaseToken++;
      record.leaseToken = leaseToken;
      const textureRef = record.texture;
      const pool = this;
      const handle = {
        id: record.id,
        key: record.key,
        desc: record.desc,
        texture: textureRef,
        get view() {
          return pool._isRecordTextureAlive(record) ? record.view : null;
        },
        getView(viewDesc) {
          return pool._getOrCreateRecordView(record, viewDesc);
        },
        getDepthView(viewDesc) {
          const opts =
            viewDesc && typeof viewDesc === 'object'
              ? { ...viewDesc, aspect: 'depth-only' }
              : { aspect: 'depth-only' };
          return pool._getOrCreateRecordView(record, opts);
        },
        getStencilView(viewDesc) {
          const opts =
            viewDesc && typeof viewDesc === 'object'
              ? { ...viewDesc, aspect: 'stencil-only' }
              : { aspect: 'stencil-only' };
          return pool._getOrCreateRecordView(record, opts);
        },
        getSrgbView() {
          const sibling = getCompatibleSrgbSiblingFormat(record.desc.format);
          return pool._getOrCreateRecordView(record, sibling ? { format: sibling } : undefined);
        },
        getSliceView(baseArrayLayer = 0, mipLevel = 0, viewDimension) {
          const texDim = record.desc.dimension;
          const resolvedDim =
            viewDimension !== undefined
              ? viewDimension
              : texDim === '3d'
                ? '3d'
                : '2d';
          const desc = {
            dimension: resolvedDim,
            baseMipLevel: mipLevel,
            mipLevelCount: 1,
          };
          if (texDim !== '3d' && resolvedDim !== '3d') {
            desc.baseArrayLayer = baseArrayLayer;
            desc.arrayLayerCount = resolvedDim === 'cube' ? 6 : 1;
          }
          return pool._getOrCreateRecordView(record, desc);
        },
        get destroyed() {
          return !pool._isRecordTextureAlive(record);
        },
        byteSize: record.byteSize,
        reused,
      };
      this._handleTokens.set(handle, { id: record.id, leaseToken });
      return handle;
    }

    /**
     * Acquire a transient render-target texture matching `desc`. Reuses an idle pooled
     * texture (including same-frame targets released earlier in the same frame) when
     * available; otherwise allocates a new `GPUTexture` and cached default `GPUTextureView`,
     * LRU-evicting idle textures as needed to stay within `maxPoolBytes`.
     *
     * Sanitizes requested dimensions into `[1, device.limits.maxTextureDimension2D]`
     * (default `8192`), normalizes common DXGI/Xbox color & depth-stencil formats, and
     * normalizes `sampleCount` to `{1, 4}` to prevent WebGPU validation errors.
     *
     * @param {object} desc
     * @param {string} [label]
     * @returns {{
     *   id: number,
     *   key: string,
     *   desc: Readonly<object>,
     *   texture: GPUTexture | null,
     *   view: GPUTextureView | null,
     *   destroyed: boolean,
     *   byteSize: number,
     *   reused: boolean,
     * }}
     */
    acquire(desc, label) {
      this._assertNotDestroyed();

      const maxDim2D = this._effectiveMaxTextureDimension2D();
      const sanitized = sanitizeRenderTargetDesc(desc, {
        device: this._device,
        deviceFeatures: this._deviceFeatures,
        maxTextureDimension2D: maxDim2D,
        includeSrgbSibling: this._autoIncludeSrgbSibling,
      });
      const normalizedDesc = sanitized.desc;
      const key = makeRenderTargetKey(normalizedDesc);
      const byteSize = estimateRenderTargetBytes(normalizedDesc);

      // 1a. Check idle bucket for an exact descriptor match (MRU pop lets unused duplicates age out).
      //     Skip and clean up any invalidated or externally-destroyed records so stale views are never returned.
      const bucket = this._idleBuckets.get(key);
      if (bucket && bucket.length > 0) {
        while (bucket.length > 0) {
          const record = bucket.pop();
          if (bucket.length === 0) {
            this._idleBuckets.delete(key);
          }
          this._idleLru.delete(record.id);
          this._transientAliases.delete(record);

          if (!this._isRecordReusable(record)) {
            this._idleBytes = Math.max(0, this._idleBytes - record.byteSize);
            if (this._destroyRecord(record)) {
              this._evictions += 1;
              if (this._isOversizedRecord(record)) {
                this._oversizedEvictions += 1;
              }
            }
            continue;
          }

          const isIntraFrame =
            record.releasedFrame === this._currentFrame ||
            record.transientAliasFrame === this._currentFrame;
          return this._claimIdleRecord(record, label, sanitized, isIntraFrame);
        }
      }

      // 1b. Check compatible intra-frame transient targets released via releaseTransient()
      //     (MRU scan: matching geometry/format/sampleCount/mips with superset usage & viewFormats).
      if (this._transientAliases.size > 0) {
        const candidates = Array.from(this._transientAliases);
        for (let i = candidates.length - 1; i >= 0; i -= 1) {
          const candidate = candidates[i];
          if (candidate.transientAliasFrame !== this._currentFrame) {
            this._transientAliases.delete(candidate);
            continue;
          }
          if (!this._isRecordReusable(candidate)) {
            this._evictIdleRecord(candidate);
            continue;
          }
          if (!this._isCompatibleTransientAlias(candidate, normalizedDesc)) {
            continue;
          }
          this._idleLru.delete(candidate.id);
          this._removeFromIdleBucket(candidate);
          return this._claimIdleRecord(candidate, label, sanitized, true);
        }
      }

      // 2. Miss: check strict memory budget before evicting idle targets or allocating.
      if (this._strictBudget && this._activeBytes + byteSize > this._maxPoolBytes) {
        throw new RangeError(
          `RenderTargetPool budget exceeded: cannot allocate ${byteSize} bytes for ` +
            `${normalizedDesc.width}x${normalizedDesc.height} ${normalizedDesc.format} ` +
            `(activeBytes=${this._activeBytes}, maxPoolBytes=${this._maxPoolBytes})`
        );
      }

      // 3. LRU-evict idle targets until totalBytes + byteSize <= maxPoolBytes (or no idle left).
      //    Checked-out / active targets are never destroyed by budget eviction.
      this._evictIdleToBudget(this._maxPoolBytes, byteSize);

      // 4. Allocate new GPUTexture + cached default GPUTextureView.
      const id = this._nextId++;
      const resolvedLabel =
        typeof label === 'string' && label.length > 0
          ? label
          : `playgta6-rt-${id}-${normalizedDesc.width}x${normalizedDesc.height}-${normalizedDesc.format}`;

      const textureDesc = {
        label: resolvedLabel,
        size: {
          width: normalizedDesc.width,
          height: normalizedDesc.height,
          depthOrArrayLayers: normalizedDesc.depthOrArrayLayers,
        },
        mipLevelCount: normalizedDesc.mipLevelCount,
        sampleCount: normalizedDesc.sampleCount,
        dimension: normalizedDesc.dimension,
        format: normalizedDesc.format,
        usage: normalizedDesc.usage,
      };
      if (normalizedDesc.viewFormats.length > 0) {
        textureDesc.viewFormats = normalizedDesc.viewFormats.slice();
      }
      const texture = this._device.createTexture(textureDesc);

      let view = null;
      try {
        view = texture && typeof texture.createView === 'function' ? texture.createView() : null;
      } catch (err) {
        this._destroyTextureSafely(texture);
        throw err;
      }

      const record = {
        id,
        key,
        desc: normalizedDesc,
        label: resolvedLabel,
        texture,
        view,
        hasView: view !== null,
        viewCache: new Map(),
        byteSize,
        active: true,
        destroyed: false,
        retireOnRelease: false,
        lastUsedFrame: this._currentFrame,
        releasedFrame: -1,
        transientAliasFrame: -1,
        leaseToken: 0,
      };

      this._allEntries.set(id, record);
      this._activeEntries.set(id, record);
      this._activeBytes += byteSize;
      this._misses += 1;
      if (view !== null) {
        this._viewsCached += 1;
        this._viewCacheMisses += 1;
      }

      this._recordAcquireSanitizationAndClass(normalizedDesc, sanitized);

      const totalBytes = this._activeBytes + this._idleBytes;
      if (totalBytes > this._peakBytes) {
        this._peakBytes = totalBytes;
      }

      return this._makeHandle(record, false);
    }

    /**
     * Retrieve a cached `GPUTextureView` for a pooled render target.
     * Omitting `viewDesc` (or passing an empty/default descriptor) returns the cached
     * default view with 0 `createView()` calls; custom descriptors (e.g. sRGB format,
     * depth-only/stencil-only aspect, or mip/layer slices) are cached per target and
     * invalidated cleanly upon eviction or destruction.
     *
     * @param {object | number} entryOrId
     * @param {object | null} [viewDesc]
     * @returns {GPUTextureView | null}
     */
    getView(entryOrId, viewDesc) {
      this._assertNotDestroyed();
      const { id, expectedLeaseToken } = this._resolveEntryId(entryOrId, 'getView');
      const record = this._allEntries.get(id);
      if (!record || !this._isRecordTextureAlive(record)) {
        return null;
      }
      if (expectedLeaseToken !== null && record.active && record.leaseToken !== expectedLeaseToken) {
        return null;
      }
      return this._getOrCreateRecordView(record, viewDesc);
    }

    /**
     * Mark an acquired render target available for immediate aliasing within the same frame
     * (or subsequent frames). If the target was invalidated during a resize while in flight
     * (or externally destroyed), it is retired and destroyed cleanly instead of re-entering
     * the idle pool. Safe against duplicate / stale releases.
     *
     * @param {object | number} entryOrId
     * @returns {boolean} True if an active target was released; false if already idle/unknown/stale.
     */
    release(entryOrId) {
      this._assertNotDestroyed();

      const { id, expectedLeaseToken } = this._resolveEntryId(entryOrId, 'release');

      const record = this._allEntries.get(id);
      if (!record || !record.active) {
        return false;
      }

      // Guard against stale handle double-release stealing a newer pass's active lease.
      if (expectedLeaseToken !== null && record.leaseToken !== expectedLeaseToken) {
        return false;
      }

      if (!this._isRecordReusable(record)) {
        this._retireActiveRecord(record);
        return true;
      }

      this._moveActiveToIdle(record, this._currentFrame);
      return true;
    }

    /**
     * Mark an acquired transient render target available for immediate intra-frame aliasing
     * by later non-overlapping passes in the same frame (e.g., G-buffer -> lighting ->
     * post-process -> UI) without allocating duplicate 4K textures. Supports exact descriptor
     * reuse as well as compatible superset `usage` / `viewFormats` aliasing within the current
     * frame. Safe against duplicate or stale handle releases.
     *
     * @param {object | number} entryOrId
     * @returns {boolean} True if an active target was released; false if already idle/unknown/stale.
     */
    releaseTransient(entryOrId) {
      this._assertNotDestroyed();

      const { id, expectedLeaseToken } = this._resolveEntryId(entryOrId, 'releaseTransient');

      const record = this._allEntries.get(id);
      if (!record || !record.active) {
        return false;
      }

      // Guard against stale handle double-release stealing a newer pass's active lease.
      if (expectedLeaseToken !== null && record.leaseToken !== expectedLeaseToken) {
        return false;
      }

      if (!this._isRecordReusable(record)) {
        this._retireActiveRecord(record);
        return true;
      }

      this._moveActiveToIdle(record, this._currentFrame);
      record.transientAliasFrame = this._currentFrame;
      this._transientAliases.add(record);
      return true;
    }

    /**
     * End the current frame epoch:
     * 1. Handle any leaked (still-active) render targets: auto-release back into the idle pool
     *    (or retire/destroy if invalidated during resize) when `autoReleaseLeaked` is enabled,
     *    or warn/notify via `onLeak` / `warn` when disabled.
     * 2. Increment the frame epoch (`currentFrame += 1`).
     * 3. Destroy idle textures unused for more than `idleFrameTtl` frames (or externally destroyed).
     *
     * @returns {{
     *   frame: number,
     *   evictedCount: number,
     *   leakedCount: number,
     *   autoReleasedCount: number,
     *   activeCount: number,
     *   idleCount: number,
     * }}
     */
    endFrame() {
      this._assertNotDestroyed();

      const endedFrame = this._currentFrame;
      this._currentFrame += 1;
      this._transientAliases.clear();

      let leakedCount = 0;
      let autoReleasedCount = 0;
      let evictedCount = 0;

      if (this._activeEntries.size > 0) {
        const leakedRecords = Array.from(this._activeEntries.values());
        leakedCount = leakedRecords.length;

        for (let i = 0; i < leakedRecords.length; i += 1) {
          const record = leakedRecords[i];
          const leakInfo = {
            id: record.id,
            key: record.key,
            label: record.label,
            byteSize: record.byteSize,
            frame: endedFrame,
            autoReleased: this._autoReleaseLeaked,
          };

          if (this._autoReleaseLeaked) {
            if (!this._isRecordReusable(record)) {
              if (this._retireActiveRecord(record)) {
                evictedCount += 1;
              }
            } else {
              this._moveActiveToIdle(record, endedFrame);
            }
            this._leakedReleases += 1;
            autoReleasedCount += 1;
          }

          if (this._onLeak) {
            this._onLeak(leakInfo);
          }

          if (!this._autoReleaseLeaked) {
            const msg =
              `[RenderTargetPool] Leaked active render target #${record.id} ` +
              `(${record.label}, ${record.byteSize} bytes) at end of frame ${endedFrame}`;
            if (this._warn) {
              this._warn(msg, leakInfo);
            } else if (!this._onLeak && typeof console !== 'undefined' && typeof console.warn === 'function') {
              console.warn(msg);
            }
          } else if (this._warn) {
            const msg =
              `[RenderTargetPool] Auto-released leaked render target #${record.id} ` +
              `(${record.label}, ${record.byteSize} bytes) at end of frame ${endedFrame}`;
            this._warn(msg, leakInfo);
          }
        }
      }

      // Evict idle textures unused for more than their effective idleFrameTtl (or oversizedIdleFrameTtl)
      // or no longer reusable.
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        const effectiveTtl = this._effectiveIdleTtlForRecord(record);
        const expired = this._currentFrame - record.lastUsedFrame > effectiveTtl;
        if (expired || !this._isRecordReusable(record)) {
          if (this._evictIdleRecord(record, expired ? 'idle' : null)) {
            evictedCount += 1;
          }
        }
      }

      // Trim any excess idle targets back within maxPoolBytes (e.g. after a strictBudget=false burst)
      // without ever destroying checked-out/active targets.
      if (this._activeBytes + this._idleBytes > this._maxPoolBytes && this._idleLru.size > 0) {
        const trimmed = this._evictIdleToBudget(this._maxPoolBytes, 0);
        evictedCount += trimmed.evictedCount;
      }

      return {
        frame: this._currentFrame,
        evictedCount,
        leakedCount,
        autoReleasedCount,
        activeCount: this._activeEntries.size,
        idleCount: this._idleLru.size,
      };
    }

    /**
     * Explicitly destroy and remove non-acquired (idle) pooled render targets whose
     * idle age satisfies `currentFrame - entry.lastUsedFrame >= maxIdleFrames`
     * (and optionally whose `byteSize >= options.minBytes`).
     * Never evicts currently acquired/in-use targets.
     *
     * @param {number} [maxIdleFrames=this._idleFrameTtl] Non-negative integer frame threshold.
     * @param {{ minBytes?: number } | null} [options]
     * @returns {{ evictedCount: number, evictedBytes: number }}
     */
    evictIdle(maxIdleFrames = this._idleFrameTtl, options = null) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(maxIdleFrames, 'maxIdleFrames');

      let minBytes = 0;
      if (options !== null && options !== undefined) {
        if (typeof options !== 'object') {
          throw new TypeError('evictIdle() options must be an object when provided');
        }
        if (options.minBytes !== undefined) {
          assertNonNegativeInteger(options.minBytes, 'minBytes');
          minBytes = options.minBytes;
        }
      }

      let evictedCount = 0;
      let evictedBytes = 0;
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        if (
          this._currentFrame - record.lastUsedFrame >= maxIdleFrames &&
          record.byteSize >= minBytes
        ) {
          const bytes = record.byteSize;
          if (this._evictIdleRecord(record, 'idle')) {
            evictedCount += 1;
            evictedBytes += bytes;
          }
        }
      }

      return { evictedCount, evictedBytes };
    }

    /**
     * Aggressively destroy and reclaim idle oversized render targets (`byteSize >= oversizedTargetBytes`)
     * whose idle age satisfies `currentFrame - entry.lastUsedFrame >= maxIdleFrames`
     * (e.g., across Start Screen -> 4K Bink2 -> Main Menu -> In-Game scene/resolution transitions).
     * Never destroys currently active/checked-out targets or non-oversized targets.
     *
     * @param {number} [maxIdleFrames=2] Non-negative integer frame threshold.
     * @returns {{ reclaimedCount: number, reclaimedBytes: number }}
     */
    reclaimOversizedIdle(maxIdleFrames = DEFAULT_OVERSIZED_RECLAIM_IDLE_FRAMES) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(maxIdleFrames, 'maxIdleFrames');

      let reclaimedCount = 0;
      let reclaimedBytes = 0;
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        if (
          this._isOversizedRecord(record) &&
          this._currentFrame - record.lastUsedFrame >= maxIdleFrames
        ) {
          const bytes = record.byteSize;
          if (this._evictIdleRecord(record, 'idle')) {
            reclaimedCount += 1;
            reclaimedBytes += bytes;
            this._reclaimedOversizedBytes += bytes;
          }
        }
      }

      return { reclaimedCount, reclaimedBytes };
    }

    /**
     * Destroy all currently idle pooled render targets (e.g., on viewport/swapchain resize)
     * while leaving any currently active in-flight targets intact and flagging them for
     * immediate destruction on release so stale pre-resize targets are never reused.
     *
     * @returns {number} Number of idle textures destroyed immediately.
     */
    invalidateAll() {
      this._assertNotDestroyed();

      let destroyedCount = 0;
      const idleRecords = Array.from(this._idleLru.values());
      for (let i = 0; i < idleRecords.length; i += 1) {
        const record = idleRecords[i];
        if (this._evictIdleRecord(record)) {
          destroyedCount += 1;
        }
      }

      this._idleBuckets.clear();
      this._idleLru.clear();
      this._transientAliases.clear();
      this._idleBytes = 0;

      // Mark in-flight active records so they are retired/destroyed cleanly on release
      // instead of being returned to post-resize passes.
      this._activeEntries.forEach((record) => {
        record.retireOnRelease = true;
      });

      return destroyedCount;
    }

    /**
     * Convenience resize hook: invalidates all idle targets and flags in-flight active
     * targets for retirement on release.
     *
     * @param {number} [_width]
     * @param {number} [_height]
     * @returns {number} Number of idle textures destroyed immediately.
     */
    resize(_width, _height) {
      return this.invalidateAll();
    }

    /**
     * Return pool hit-rate, occupancy, memory, sanitization, and lifecycle telemetry.
     */
    getStats() {
      const poolHits = sanitizeNonNegativeCounter(this._hits);
      const poolMisses = sanitizeNonNegativeCounter(this._misses);
      const totalRequests = poolHits + poolMisses;
      const hitRate = clampUnitRate(poolHits, totalRequests);

      const viewCacheHits = sanitizeNonNegativeCounter(this._viewCacheHits);
      const viewCacheMisses = sanitizeNonNegativeCounter(this._viewCacheMisses);
      const viewCacheHitRate = clampUnitRate(viewCacheHits, viewCacheHits + viewCacheMisses);

      const activeBytes = sanitizeNonNegativeCounter(this._activeBytes);
      const idleBytes = sanitizeNonNegativeCounter(this._idleBytes);
      const totalBytes = activeBytes + idleBytes;
      const peakVramBytes = Math.max(totalBytes, sanitizeNonNegativeCounter(this._peakBytes));

      const evictions = sanitizeNonNegativeCounter(this._evictions);
      const budgetEvictions = sanitizeNonNegativeCounter(this._budgetEvictions);
      const idleEvictions = sanitizeNonNegativeCounter(this._idleEvictions);
      const oversizedEvictions = sanitizeNonNegativeCounter(this._oversizedEvictions);
      const reclaimedOversizedBytes = sanitizeNonNegativeCounter(this._reclaimedOversizedBytes);
      const transientReuseCount = sanitizeNonNegativeCounter(this._transientReuseCount);
      const intraFrameAliasHits = sanitizeNonNegativeCounter(this._intraFrameAliasHits);
      const vramBudgetUtilization = clampUnitRate(totalBytes, this._maxPoolBytes);

      let activeHdrCount = 0;
      let activeDepthCount = 0;
      for (const rec of this._activeEntries.values()) {
        if (HDR_RENDER_TARGET_FORMATS.has(rec.desc.format)) {
          activeHdrCount += 1;
        }
        if (DEPTH_RENDER_TARGET_FORMATS.has(rec.desc.format)) {
          activeDepthCount += 1;
        }
      }

      let idleHdrCount = 0;
      let idleDepthCount = 0;
      for (const rec of this._idleLru.values()) {
        if (HDR_RENDER_TARGET_FORMATS.has(rec.desc.format)) {
          idleHdrCount += 1;
        }
        if (DEPTH_RENDER_TARGET_FORMATS.has(rec.desc.format)) {
          idleDepthCount += 1;
        }
      }

      return {
        hits: poolHits,
        misses: poolMisses,
        poolHits,
        poolMisses,
        transientReuseCount,
        intraFrameAliasHits,
        viewsCached: sanitizeNonNegativeCounter(this._viewsCached),
        viewCacheHits,
        viewCacheMisses,
        viewCacheHitRate,
        hitRate,
        reuseRatio: hitRate,
        activeCount: this._activeEntries.size,
        idleCount: this._idleLru.size,
        activeTargets: this._activeEntries.size,
        idleTargets: this._idleLru.size,
        activeHdrCount,
        activeDepthCount,
        idleHdrCount,
        idleDepthCount,
        activeBytes,
        idleBytes,
        totalBytes,
        estimatedBytes: totalBytes,
        estimatedVramBytes: totalBytes,
        peakBytes: peakVramBytes,
        peakPooledBytes: peakVramBytes,
        peakEstimatedBytes: peakVramBytes,
        peakVramBytes,
        maxPoolBytes: this._maxPoolBytes,
        vramBudgetUtilization,
        idleFrameTtl: this._idleFrameTtl,
        oversizedTargetBytes: this._oversizedTargetBytes,
        oversizedIdleFrameTtl: this._oversizedIdleFrameTtl,
        evictions,
        budgetEvictions,
        idleEvictions,
        oversizedEvictions,
        reclaimedOversizedBytes,
        leakedReleases: sanitizeNonNegativeCounter(this._leakedReleases),
        dimensionClampCount: sanitizeNonNegativeCounter(this._dimensionClampCount),
        formatNormalizeCount: sanitizeNonNegativeCounter(this._formatNormalizeCount),
        formatFallbackCount: sanitizeNonNegativeCounter(this._formatFallbackCount),
        sampleCountNormalizeCount: sanitizeNonNegativeCounter(this._sampleCountNormalizeCount),
        hdrAcquires: sanitizeNonNegativeCounter(this._hdrAcquires),
        depthAcquires: sanitizeNonNegativeCounter(this._depthAcquires),
        srgbAcquires: sanitizeNonNegativeCounter(this._srgbAcquires),
        maxTextureDimension2D: this._effectiveMaxTextureDimension2D(),
        currentFrame: sanitizeNonNegativeCounter(this._currentFrame),
        perfProfile: this._perfProfileName,
      };
    }

    /**
     * Reset cumulative telemetry counters while preserving live pooled GPUTextures,
     * cached GPUTextureViews, active/idle occupancy, and frame/TTL aging state.
     * Re-baselines `peakVramBytes` (and peak aliases) to current live VRAM occupancy
     * so `peakVramBytes >= estimatedVramBytes` always holds.
     */
    resetStats() {
      this._hits = 0;
      this._misses = 0;
      this._transientReuseCount = 0;
      this._intraFrameAliasHits = 0;
      this._viewCacheHits = 0;
      this._viewCacheMisses = 0;
      this._peakBytes =
        sanitizeNonNegativeCounter(this._activeBytes) +
        sanitizeNonNegativeCounter(this._idleBytes);
      this._evictions = 0;
      this._budgetEvictions = 0;
      this._idleEvictions = 0;
      this._oversizedEvictions = 0;
      this._reclaimedOversizedBytes = 0;
      this._leakedReleases = 0;
      this._dimensionClampCount = 0;
      this._formatNormalizeCount = 0;
      this._formatFallbackCount = 0;
      this._sampleCountNormalizeCount = 0;
      this._hdrAcquires = 0;
      this._depthAcquires = 0;
      this._srgbAcquires = 0;
      return this.getStats();
    }

    /**
     * Destroy all active and idle GPU textures owned by the pool and release references.
     * Safe to call multiple times (idempotent).
     */
    destroy() {
      if (this._destroyed) {
        return;
      }

      this._destroyed = true;

      const allRecords = Array.from(this._allEntries.values());
      for (let i = 0; i < allRecords.length; i += 1) {
        this._destroyRecord(allRecords[i]);
      }

      this._allEntries.clear();
      this._activeEntries.clear();
      this._idleBuckets.clear();
      this._idleLru.clear();
      this._transientAliases.clear();
      this._activeBytes = 0;
      this._idleBytes = 0;
      this._viewsCached = 0;
      this._device = null;
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_MAX_POOL_BYTES,
    DEFAULT_IDLE_FRAME_TTL,
    DEFAULT_OVERSIZED_TARGET_BYTES,
    DEFAULT_MAX_TEXTURE_DIMENSION_2D,
    TEXTURE_USAGE,
    DEFAULT_TEXTURE_USAGE,
    FORMAT_BYTES_PER_PIXEL,
    DXGI_FORMAT_ALIASES,
    COMPATIBLE_VIEW_FORMATS,
    REQUIRED_3D_RT_FORMATS,
    normalizeRenderTargetFormat,
    isHdrRenderTargetFormat,
    isDepthRenderTargetFormat,
    hasStencilComponent,
    getCompatibleSrgbSiblingFormat,
    withSrgbSiblingViewFormat,
    resolveRenderTargetFormatForDevice,
    isCompatibleViewFormat,
    normalizeViewFormats,
    normalizeSampleCount,
    sanitizeRenderTargetDesc,
    normalizeRenderTargetDesc,
    makeRenderTargetKey,
    makeTextureViewKey,
    estimateRenderTargetBytes,
    RenderTargetPool,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6RenderTargetPool = exportsObj;
  } else if (root) {
    root.PlayGta6RenderTargetPool = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
