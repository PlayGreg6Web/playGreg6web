'use strict';

(function (root) {
  const DEFAULT_DEPTH_FORMAT = 'depth24plus';

  const SUPPORTED_DEPTH_FORMATS = Object.freeze([
    'depth24plus',
    'depth24plus-stencil8',
    'depth32float',
    'depth32float-stencil8',
    'depth16unorm',
  ]);

  const STENCIL_DEPTH_FORMATS = new Set([
    'depth24plus-stencil8',
    'depth32float-stencil8',
  ]);

  const SUPPORTED_ALPHA_MODES = Object.freeze(['opaque', 'premultiplied']);

  const VALID_CANVAS_FORMATS = Object.freeze(['bgra8unorm', 'rgba8unorm', 'rgba16float']);
  const VALID_CANVAS_FORMAT_SET = new Set(VALID_CANVAS_FORMATS);

  const HDR_STREAM_FORMATS = Object.freeze([
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
  const HDR_STREAM_FORMAT_SET = new Set(HDR_STREAM_FORMATS);
  const SDR_NON_SRGB_CANVAS_FORMAT_SET = new Set(['bgra8unorm', 'rgba8unorm']);

  const TONE_MAP_MODES = Object.freeze(['none', 'passthrough', 'reinhard', 'aces']);
  const TONE_MAP_MODE_CODES = Object.freeze({
    none: 0,
    passthrough: 0,
    reinhard: 1,
    aces: 2,
  });
  const DEFAULT_TONE_MAP_MODE = 'none';
  const DEFAULT_EXPOSURE = 1.0;
  const TONEMAP_UNIFORM_BYTE_SIZE = 16;

  const COMPATIBLE_VIEW_FORMATS = Object.freeze({
    bgra8unorm: Object.freeze(['bgra8unorm-srgb']),
    rgba8unorm: Object.freeze(['rgba8unorm-srgb']),
    'bgra8unorm-srgb': Object.freeze(['bgra8unorm']),
    'rgba8unorm-srgb': Object.freeze(['rgba8unorm']),
  });

  const COMPATIBLE_VIEW_FORMATS_KEY = Object.freeze({
    bgra8unorm: 'bgra8unorm-srgb',
    rgba8unorm: 'rgba8unorm-srgb',
    'bgra8unorm-srgb': 'bgra8unorm',
    'rgba8unorm-srgb': 'rgba8unorm',
  });

  const DEFAULT_CLEAR_COLOR = Object.freeze({ r: 0.05, g: 0.07, b: 0.11, a: 1.0 });
  const DEFAULT_FORMAT = 'bgra8unorm';
  const DEFAULT_ALPHA_MODE = 'opaque';
  const DEFAULT_MAX_DIMENSION = 8192;
  const DEFAULT_FRAME_WINDOW = 60;
  const DEFAULT_MAX_ERROR_HISTORY = 32;
  const DEFAULT_DROPPED_FRAME_THRESHOLD_MS = 1000 / 30;
  const TARGET_30FPS_BUDGET_MS = 1000 / 30;
  const DROPPED_FRAME_CADENCE_MULTIPLIER = 1.5;
  const MAX_VALIDATION_ERROR_MESSAGE_LENGTH = 512;
  const MAX_BLIT_BIND_GROUP_CACHE = 8;

  // WebGPU GPUTextureUsage constants with numeric fallbacks for Node/headless tests.
  const TEXTURE_USAGE_COPY_SRC =
    typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.COPY_SRC === 'number'
      ? GPUTextureUsage.COPY_SRC
      : 0x01;
  const TEXTURE_USAGE_TEXTURE_BINDING =
    typeof GPUTextureUsage !== 'undefined' && typeof GPUTextureUsage.TEXTURE_BINDING === 'number'
      ? GPUTextureUsage.TEXTURE_BINDING
      : 0x04;
  const TEXTURE_USAGE_RENDER_ATTACHMENT =
    typeof GPUTextureUsage !== 'undefined' &&
    typeof GPUTextureUsage.RENDER_ATTACHMENT === 'number'
      ? GPUTextureUsage.RENDER_ATTACHMENT
      : 0x10;
  const DEFAULT_USAGE = TEXTURE_USAGE_RENDER_ATTACHMENT;
  const INTERMEDIATE_TEXTURE_USAGE =
    TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_TEXTURE_BINDING | TEXTURE_USAGE_COPY_SRC;

  const BUFFER_USAGE_COPY_DST =
    typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
      ? GPUBufferUsage.COPY_DST
      : 0x0008;
  const BUFFER_USAGE_UNIFORM =
    typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.UNIFORM === 'number'
      ? GPUBufferUsage.UNIFORM
      : 0x0040;

  const BLIT_FULLSCREEN_WGSL = /* wgsl */ `
struct BlitVsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> BlitVsOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
  );
  var out: BlitVsOut;
  out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
  out.uv = uv[vid];
  return out;
}

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

@fragment
fn fs_main(in: BlitVsOut) -> @location(0) vec4<f32> {
  return textureSample(blit_tex, blit_sampler, in.uv);
}
`.trim();

  const BLIT_SRGB_ENCODE_WGSL = /* wgsl */ `
struct BlitVsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> BlitVsOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
  );
  var out: BlitVsOut;
  out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
  out.uv = uv[vid];
  return out;
}

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

fn linear_to_srgb_channel(v: f32) -> f32 {
  let c = clamp(v, 0.0, 1.0);
  if (c <= 0.0031308) {
    return c * 12.92;
  }
  return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(rgb: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    linear_to_srgb_channel(rgb.r),
    linear_to_srgb_channel(rgb.g),
    linear_to_srgb_channel(rgb.b)
  );
}

@fragment
fn fs_main(in: BlitVsOut) -> @location(0) vec4<f32> {
  let sampled = textureSample(blit_tex, blit_sampler, in.uv);
  return vec4<f32>(linear_to_srgb(sampled.rgb), sampled.a);
}
`.trim();

  const BLIT_TONEMAP_WGSL = /* wgsl */ `
struct BlitVsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

struct ToneMapParams {
  mode: u32,
  exposure: f32,
  apply_srgb: u32,
  _pad: u32,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> BlitVsOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
  );
  var out: BlitVsOut;
  out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
  out.uv = uv[vid];
  return out;
}

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;
@group(0) @binding(2) var<uniform> tonemap_params: ToneMapParams;

fn tonemap_reinhard(c: vec3<f32>) -> vec3<f32> {
  let x = max(c, vec3<f32>(0.0));
  return x / (vec3<f32>(1.0) + x);
}

fn tonemap_aces(c: vec3<f32>) -> vec3<f32> {
  let x = max(c, vec3<f32>(0.0));
  let a = 2.51;
  let b = 0.03;
  let d = 2.43;
  let e = 0.59;
  let f = 0.14;
  return clamp((x * (a * x + vec3<f32>(b))) / (x * (d * x + vec3<f32>(e)) + vec3<f32>(f)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb_channel(v: f32) -> f32 {
  let c = clamp(v, 0.0, 1.0);
  if (c <= 0.0031308) {
    return c * 12.92;
  }
  return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(rgb: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    linear_to_srgb_channel(rgb.r),
    linear_to_srgb_channel(rgb.g),
    linear_to_srgb_channel(rgb.b)
  );
}

@fragment
fn fs_main(in: BlitVsOut) -> @location(0) vec4<f32> {
  let sampled = textureSample(blit_tex, blit_sampler, in.uv);
  var rgb = sampled.rgb * tonemap_params.exposure;
  if (tonemap_params.mode == 1u) {
    rgb = tonemap_reinhard(rgb);
  } else if (tonemap_params.mode == 2u) {
    rgb = tonemap_aces(rgb);
  }
  if (tonemap_params.apply_srgb != 0u) {
    rgb = linear_to_srgb(rgb);
  }
  return vec4<f32>(rgb, sampled.a);
}
`.trim();

  function isValidCanvasFormat(format) {
    return typeof format === 'string' && VALID_CANVAS_FORMAT_SET.has(format);
  }

  function isSrgbFormat(format) {
    if (typeof format !== 'string' || format.length === 0) {
      return false;
    }
    return format.trim().toLowerCase().endsWith('-srgb');
  }

  function isHdrStreamFormat(format) {
    if (typeof format !== 'string' || format.length === 0) {
      return false;
    }
    return HDR_STREAM_FORMAT_SET.has(format.trim().toLowerCase());
  }

  function isHdrToneMapBlitRequested(
    streamFormat,
    canvasFormat,
    toneMapMode,
    exposure,
    applySrgbEncodeOverride
  ) {
    return Boolean(
      streamFormat === 'rgba16float' &&
        SDR_NON_SRGB_CANVAS_FORMAT_SET.has(canvasFormat) &&
        (toneMapMode === 'reinhard' ||
          toneMapMode === 'aces' ||
          exposure !== DEFAULT_EXPOSURE ||
          applySrgbEncodeOverride === true)
    );
  }

  function normalizeToneMapMode(mode) {
    if (mode === undefined || mode === null) {
      return DEFAULT_TONE_MAP_MODE;
    }
    if (typeof mode !== 'string') {
      throw new TypeError(`toneMapMode must be a string, got ${typeof mode}`);
    }
    const normalized = mode.trim().toLowerCase();
    if (!Object.prototype.hasOwnProperty.call(TONE_MAP_MODE_CODES, normalized)) {
      throw new RangeError(
        `Unsupported toneMapMode "${mode}"; expected one of: ${TONE_MAP_MODES.join(', ')}`
      );
    }
    return normalized;
  }

  function normalizeExposure(exposure) {
    if (exposure === undefined || exposure === null) {
      return DEFAULT_EXPOSURE;
    }
    if (typeof exposure !== 'number' || !Number.isFinite(exposure) || exposure <= 0) {
      throw new RangeError(`exposure must be a positive finite number, got ${exposure}`);
    }
    return exposure;
  }

  function normalizeApplySrgbEncode(applySrgbEncode) {
    if (applySrgbEncode === undefined || applySrgbEncode === null) {
      return null;
    }
    if (typeof applySrgbEncode !== 'boolean') {
      throw new TypeError('applySrgbEncode must be a boolean, null, or undefined');
    }
    return applySrgbEncode;
  }

  function getCompatibleViewFormats(format) {
    if (typeof format !== 'string' || format.length === 0) {
      return [];
    }
    const siblings = COMPATIBLE_VIEW_FORMATS[format];
    return siblings ? siblings.slice() : [];
  }

  function normalizeViewFormats(viewFormats) {
    if (viewFormats === null || viewFormats === undefined) {
      return null;
    }
    if (!Array.isArray(viewFormats)) {
      throw new TypeError('viewFormats must be an array of non-empty format strings when provided');
    }
    const clean = [];
    for (let i = 0; i < viewFormats.length; i += 1) {
      const vf = viewFormats[i];
      if (typeof vf !== 'string' || vf.length === 0) {
        throw new TypeError('viewFormats entries must be non-empty strings');
      }
      if (!clean.includes(vf)) {
        clean.push(vf);
      }
    }
    return clean;
  }

  function normalizeFrameIntervalUs(value) {
    if (value === null || value === undefined) {
      return null;
    }
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      throw new RangeError(`frameIntervalUs must be a non-negative finite number, got ${value}`);
    }
    return value;
  }

  function normalizeUsage(usage) {
    if (usage === null || usage === undefined) {
      return null;
    }
    if (typeof usage !== 'number' || !Number.isInteger(usage) || usage < 1) {
      throw new RangeError(
        `usage must be a positive integer GPUTextureUsage bitmask, got ${usage}`
      );
    }
    return usage;
  }

  function normalizeDroppedFrameThresholdMs(value) {
    if (value === null || value === undefined) {
      return null;
    }
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      throw new RangeError(
        `droppedFrameThresholdMs must be a non-negative finite number, got ${value}`
      );
    }
    return value;
  }

  function sanitizeNonNegativeDrawCount(value) {
    if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
      return Math.floor(value);
    }
    if (value === true) {
      return 1;
    }
    return 0;
  }

  function isPositiveContentFlag(value) {
    return (
      value === true ||
      (typeof value === 'number' && Number.isFinite(value) && value > 0)
    );
  }

  function isFrameContentMetadata(obj) {
    if (!obj || typeof obj !== 'object') {
      return false;
    }
    return (
      'hasVisibleContent' in obj ||
      'hasDraws' in obj ||
      'realPackDraws' in obj ||
      'nonClearDraw' in obj ||
      'timestampMs' in obj ||
      'frameIntervalUs' in obj ||
      'dropped' in obj
    );
  }

  function resolvePreferredCanvasFormat() {
    if (
      typeof navigator !== 'undefined' &&
      navigator.gpu &&
      typeof navigator.gpu.getPreferredCanvasFormat === 'function'
    ) {
      const preferred = navigator.gpu.getPreferredCanvasFormat();
      if (isValidCanvasFormat(preferred)) {
        return preferred;
      }
    }
    return DEFAULT_FORMAT;
  }

  function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function isStencilFormat(format) {
    return STENCIL_DEPTH_FORMATS.has(format);
  }

  function validateDepthFormat(depthFormat) {
    if (typeof depthFormat !== 'string' || !SUPPORTED_DEPTH_FORMATS.includes(depthFormat)) {
      throw new RangeError(
        `Unsupported depthFormat "${depthFormat}"; expected one of: ${SUPPORTED_DEPTH_FORMATS.join(', ')}`
      );
    }
  }

  function validateAlphaMode(alphaMode) {
    if (typeof alphaMode !== 'string' || !SUPPORTED_ALPHA_MODES.includes(alphaMode)) {
      throw new RangeError(
        `Unsupported alphaMode "${alphaMode}"; expected one of: ${SUPPORTED_ALPHA_MODES.join(', ')}`
      );
    }
  }

  function clampDimension(cssSize, devicePixelRatio, maxDimension) {
    if (cssSize === Infinity) {
      return maxDimension;
    }
    if (cssSize === -Infinity || typeof cssSize !== 'number' || Number.isNaN(cssSize)) {
      return 1;
    }

    let dpr = 1;
    if (devicePixelRatio !== undefined && devicePixelRatio !== null) {
      if (devicePixelRatio === Infinity) {
        return cssSize > 0 ? maxDimension : 1;
      }
      if (
        typeof devicePixelRatio === 'number' &&
        !Number.isNaN(devicePixelRatio) &&
        devicePixelRatio > 0
      ) {
        dpr = devicePixelRatio;
      }
    }

    const scaled = Math.round(cssSize * dpr);
    if (!Number.isFinite(scaled)) {
      return scaled > 0 ? maxDimension : 1;
    }
    if (scaled < 1) {
      return 1;
    }
    if (scaled > maxDimension) {
      return maxDimension;
    }
    return scaled;
  }

  function validatePositiveDisplayNumber(value, label) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      throw new RangeError(`${label} must be a positive finite number, got ${value}`);
    }
    return value;
  }

  function normalizeDisplayBounds(input, fallbackDpr = 1) {
    if (input === null || input === undefined) {
      return null;
    }
    if (typeof input !== 'object') {
      throw new TypeError('displayBounds must be an object or null');
    }

    if (input.maxDisplayWidth !== undefined || input.maxDisplayHeight !== undefined) {
      const maxW = Math.max(
        1,
        Math.round(validatePositiveDisplayNumber(input.maxDisplayWidth, 'maxDisplayWidth'))
      );
      const maxH = Math.max(
        1,
        Math.round(validatePositiveDisplayNumber(input.maxDisplayHeight, 'maxDisplayHeight'))
      );
      const rawDpr =
        input.devicePixelRatio !== undefined
          ? input.devicePixelRatio
          : input.dpr !== undefined
            ? input.dpr
            : null;
      const dpr =
        typeof rawDpr === 'number' && Number.isFinite(rawDpr) && rawDpr > 0 ? rawDpr : 1;
      const screenWidth =
        typeof input.screenWidth === 'number' &&
        Number.isFinite(input.screenWidth) &&
        input.screenWidth > 0
          ? input.screenWidth
          : maxW;
      const screenHeight =
        typeof input.screenHeight === 'number' &&
        Number.isFinite(input.screenHeight) &&
        input.screenHeight > 0
          ? input.screenHeight
          : maxH;
      return {
        maxDisplayWidth: maxW,
        maxDisplayHeight: maxH,
        screenWidth,
        screenHeight,
        devicePixelRatio: dpr,
      };
    }

    if (input.screenWidth !== undefined || input.screenHeight !== undefined) {
      const screenWidth = validatePositiveDisplayNumber(input.screenWidth, 'screenWidth');
      const screenHeight = validatePositiveDisplayNumber(input.screenHeight, 'screenHeight');
      const rawDpr =
        input.devicePixelRatio !== undefined
          ? input.devicePixelRatio
          : input.dpr !== undefined
            ? input.dpr
            : fallbackDpr;
      const dpr =
        rawDpr === undefined || rawDpr === null
          ? 1
          : validatePositiveDisplayNumber(rawDpr, 'devicePixelRatio');
      return {
        maxDisplayWidth: Math.max(1, Math.round(screenWidth * dpr)),
        maxDisplayHeight: Math.max(1, Math.round(screenHeight * dpr)),
        screenWidth,
        screenHeight,
        devicePixelRatio: dpr,
      };
    }

    if (input.width !== undefined || input.height !== undefined) {
      const maxW = Math.max(
        1,
        Math.round(validatePositiveDisplayNumber(input.width, 'displayBounds.width'))
      );
      const maxH = Math.max(
        1,
        Math.round(validatePositiveDisplayNumber(input.height, 'displayBounds.height'))
      );
      return {
        maxDisplayWidth: maxW,
        maxDisplayHeight: maxH,
        screenWidth: maxW,
        screenHeight: maxH,
        devicePixelRatio: 1,
      };
    }

    throw new RangeError(
      'displayBounds must provide positive maxDisplayWidth/maxDisplayHeight, screenWidth/screenHeight, or width/height'
    );
  }

  function extractDisplayBoundsOption(options, allowDevicePixelRatioFallback = false) {
    if (!options || typeof options !== 'object') {
      return { provided: false, bounds: null };
    }
    if (options.displayBounds !== undefined) {
      return { provided: true, bounds: normalizeDisplayBounds(options.displayBounds) };
    }
    if (options.maxDisplayWidth !== undefined || options.maxDisplayHeight !== undefined) {
      return {
        provided: true,
        bounds: normalizeDisplayBounds({
          maxDisplayWidth: options.maxDisplayWidth,
          maxDisplayHeight: options.maxDisplayHeight,
        }),
      };
    }
    if (options.screenWidth !== undefined || options.screenHeight !== undefined) {
      const rawDpr =
        options.displayDevicePixelRatio !== undefined
          ? options.displayDevicePixelRatio
          : options.screenDpr !== undefined
            ? options.screenDpr
            : options.dpr !== undefined
              ? options.dpr
              : allowDevicePixelRatioFallback && options.devicePixelRatio !== undefined
                ? options.devicePixelRatio
                : 1;
      return {
        provided: true,
        bounds: normalizeDisplayBounds({
          screenWidth: options.screenWidth,
          screenHeight: options.screenHeight,
          devicePixelRatio: rawDpr,
        }),
      };
    }
    return { provided: false, bounds: null };
  }

  /**
   * Proportionally clamp requested backing dimensions to fit within the physical
   * display pixel ceiling (`maxDisplayWidth x maxDisplayHeight`), preserving aspect
   * ratio. Never upscales small canvases and never downscales requests that already
   * fit within the native display resolution.
   *
   * @param {number} requestedWidth
   * @param {number} requestedHeight
   * @param {{ maxDisplayWidth: number, maxDisplayHeight: number } | null | undefined} displayBounds
   * @param {number} [maxDimension]
   * @returns {{ width: number, height: number, clamped: boolean }}
   */
  function clampToDisplayBounds(requestedWidth, requestedHeight, displayBounds, maxDimension) {
    const reqW = Math.max(1, Math.round(requestedWidth));
    const reqH = Math.max(1, Math.round(requestedHeight));
    if (
      !displayBounds ||
      typeof displayBounds !== 'object' ||
      !Number.isFinite(displayBounds.maxDisplayWidth) ||
      !Number.isFinite(displayBounds.maxDisplayHeight) ||
      displayBounds.maxDisplayWidth < 1 ||
      displayBounds.maxDisplayHeight < 1
    ) {
      return { width: reqW, height: reqH, clamped: false };
    }

    let limitW = Math.floor(displayBounds.maxDisplayWidth);
    let limitH = Math.floor(displayBounds.maxDisplayHeight);
    if (typeof maxDimension === 'number' && Number.isFinite(maxDimension) && maxDimension >= 1) {
      const cap = Math.floor(maxDimension);
      limitW = Math.min(limitW, cap);
      limitH = Math.min(limitH, cap);
    }

    if (reqW <= limitW && reqH <= limitH) {
      return { width: reqW, height: reqH, clamped: false };
    }

    const scale = Math.min(limitW / reqW, limitH / reqH);
    const width = Math.max(1, Math.min(limitW, Math.round(reqW * scale)));
    const height = Math.max(1, Math.min(limitH, Math.round(reqH * scale)));
    return {
      width,
      height,
      clamped: width < reqW || height < reqH,
    };
  }

  function normalizeClearColor(clearColor) {
    if (!clearColor) {
      return { ...DEFAULT_CLEAR_COLOR };
    }
    if (Array.isArray(clearColor) && clearColor.length >= 4) {
      return {
        r: clearColor[0],
        g: clearColor[1],
        b: clearColor[2],
        a: clearColor[3],
      };
    }
    return {
      r: typeof clearColor.r === 'number' ? clearColor.r : DEFAULT_CLEAR_COLOR.r,
      g: typeof clearColor.g === 'number' ? clearColor.g : DEFAULT_CLEAR_COLOR.g,
      b: typeof clearColor.b === 'number' ? clearColor.b : DEFAULT_CLEAR_COLOR.b,
      a: typeof clearColor.a === 'number' ? clearColor.a : DEFAULT_CLEAR_COLOR.a,
    };
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

    // Normalize line breaks/tabs to spaces and strip control characters.
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

  /**
   * Manages an OffscreenCanvas WebGPU swapchain:
   * - DPR-aware configuration and dynamic resize with safe clamping
   * - Lazy depth-stencil GPUTexture allocation and leak-free recreation
   * - GPURenderPassDescriptor acquisition compatible with WgpuCommandProcessor
   * - device.lost handling and live GPUDevice recovery
   * - Rolling frame-pacing telemetry (FPS, avg/max frame time, resize count)
   */
  class WgpuSwapchain {
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('WgpuSwapchain requires an options object');
      }

      const {
        canvas,
        device,
        context,
        format,
        streamFormat,
        viewFormats,
        usage,
        depthFormat = DEFAULT_DEPTH_FORMAT,
        alphaMode = DEFAULT_ALPHA_MODE,
        maxDimension = DEFAULT_MAX_DIMENSION,
        enableDepth = true,
        now,
        nowFn,
        frameWindowSize = DEFAULT_FRAME_WINDOW,
        frameIntervalUs,
        droppedFrameThresholdMs,
        onValidationError = null,
        onFirstVisibleFrame = null,
        maxErrorHistory = DEFAULT_MAX_ERROR_HISTORY,
        preserveCopySrcUntilVisible = false,
        toneMapMode = DEFAULT_TONE_MAP_MODE,
        exposure = DEFAULT_EXPOSURE,
        applySrgbEncode = null,
      } = options;

      if (!canvas || typeof canvas !== 'object') {
        throw new TypeError('WgpuSwapchain requires a canvas (OffscreenCanvas or canvas-like object)');
      }
      if (!device || typeof device !== 'object') {
        throw new TypeError('WgpuSwapchain requires a GPUDevice');
      }

      const resolvedContext =
        context || (typeof canvas.getContext === 'function' ? canvas.getContext('webgpu') : null);
      if (!resolvedContext || typeof resolvedContext.configure !== 'function') {
        throw new TypeError('WgpuSwapchain requires a WebGPU GPUCanvasContext');
      }

      validateDepthFormat(depthFormat);
      validateAlphaMode(alphaMode);

      if (!Number.isInteger(maxDimension) || maxDimension < 1) {
        throw new RangeError(`maxDimension must be a positive integer, got ${maxDimension}`);
      }
      const resolvedNow =
        nowFn !== undefined ? nowFn : now !== undefined ? now : defaultNow;
      if (typeof resolvedNow !== 'function') {
        throw new TypeError('now/nowFn must be a function returning a timestamp in milliseconds');
      }
      if (!Number.isInteger(frameWindowSize) || frameWindowSize < 1) {
        throw new RangeError(`frameWindowSize must be a positive integer, got ${frameWindowSize}`);
      }
      const resolvedViewFormats =
        viewFormats !== undefined ? normalizeViewFormats(viewFormats) : null;
      const resolvedExplicitUsage = usage !== undefined ? normalizeUsage(usage) : null;
      const resolvedFrameIntervalUs =
        frameIntervalUs !== undefined ? normalizeFrameIntervalUs(frameIntervalUs) : null;
      const resolvedDroppedFrameThresholdMs =
        droppedFrameThresholdMs !== undefined
          ? normalizeDroppedFrameThresholdMs(droppedFrameThresholdMs)
          : null;
      if (
        onValidationError !== null &&
        onValidationError !== undefined &&
        typeof onValidationError !== 'function'
      ) {
        throw new TypeError('onValidationError must be a function when provided');
      }
      if (
        onFirstVisibleFrame !== null &&
        onFirstVisibleFrame !== undefined &&
        typeof onFirstVisibleFrame !== 'function'
      ) {
        throw new TypeError('onFirstVisibleFrame must be a function when provided');
      }
      if (!Number.isInteger(maxErrorHistory) || maxErrorHistory < 1) {
        throw new RangeError(`maxErrorHistory must be a positive integer, got ${maxErrorHistory}`);
      }
      const resolvedToneMapMode = normalizeToneMapMode(toneMapMode);
      const resolvedExposure = normalizeExposure(exposure);
      const resolvedApplySrgbEncode = normalizeApplySrgbEncode(applySrgbEncode);
      const initialDisplayBounds = extractDisplayBoundsOption(options, true);

      const preferredCanvasFormat = resolvePreferredCanvasFormat();
      let resolvedCanvasFormat = preferredCanvasFormat;
      let resolvedStreamFormat = preferredCanvasFormat;
      let initialUsesBlitFallback = false;

      if (typeof format === 'string' && format.length > 0) {
        if (isValidCanvasFormat(format)) {
          resolvedCanvasFormat = format;
          resolvedStreamFormat = format;
        } else {
          resolvedCanvasFormat = preferredCanvasFormat;
          resolvedStreamFormat = format;
          initialUsesBlitFallback = true;
        }
      }

      if (typeof streamFormat === 'string' && streamFormat.length > 0) {
        const forceHdrToneMapBlit = isHdrToneMapBlitRequested(
          streamFormat,
          resolvedCanvasFormat,
          resolvedToneMapMode,
          resolvedExposure,
          resolvedApplySrgbEncode
        );
        if (isValidCanvasFormat(streamFormat) && !forceHdrToneMapBlit) {
          resolvedCanvasFormat = streamFormat;
          resolvedStreamFormat = streamFormat;
          initialUsesBlitFallback = false;
        } else {
          resolvedStreamFormat = streamFormat;
          initialUsesBlitFallback = true;
        }
      }

      this._canvas = canvas;
      this._device = device;
      this._context = resolvedContext;
      this._preferredCanvasFormat = resolvedCanvasFormat;
      this._format = resolvedCanvasFormat;
      this._streamFormat = resolvedStreamFormat;
      this._usesBlitFallback = initialUsesBlitFallback;
      this._explicitViewFormats = resolvedViewFormats;
      this._viewFormats = resolvedViewFormats;
      this._explicitViewFormatsKey = resolvedViewFormats ? resolvedViewFormats.join(',') : '';
      this._activeViewFormats = [];
      this._viewFormatsFallbackCount = 0;
      this._explicitUsage = resolvedExplicitUsage;
      this._usage = resolvedExplicitUsage !== null ? resolvedExplicitUsage : DEFAULT_USAGE;
      this._preserveCopySrcUntilVisible = Boolean(preserveCopySrcUntilVisible);
      this._copySrcSupported = null;
      this._appliedConfig = {
        device: null,
        format: '',
        usage: 0,
        alphaMode: '',
        width: 0,
        height: 0,
        viewFormatsKey: '',
      };
      this._hasAppliedConfig = false;
      this._depthFormat = depthFormat;
      this._alphaMode = alphaMode;
      this._maxDimension = maxDimension;
      this._enableDepth = Boolean(enableDepth);
      this._now = resolvedNow;
      this._frameWindowSize = frameWindowSize;
      this._frameIntervalUs = resolvedFrameIntervalUs;
      this._droppedFrameThresholdMs = resolvedDroppedFrameThresholdMs;
      this._onValidationError = typeof onValidationError === 'function' ? onValidationError : null;
      this._onFirstVisibleFrame =
        typeof onFirstVisibleFrame === 'function' ? onFirstVisibleFrame : null;
      this._maxErrorHistory = maxErrorHistory;

      this._width = 0;
      this._height = 0;
      this._requestedWidth = 0;
      this._requestedHeight = 0;
      this._displayBounds = initialDisplayBounds.provided ? initialDisplayBounds.bounds : null;
      this._clampedToDisplayCount = 0;
      this._devicePixelRatio = 1;
      this._configured = false;
      this._destroyed = false;

      this._depthTexture = null;
      this._depthTextureView = null;
      this._depthTextureMeta = null;

      this._intermediateTexture = null;
      this._intermediateTextureView = null;
      this._intermediateTextureMeta = null;
      this._blitShaderModule = null;
      this._blitSrgbShaderModule = null;
      this._blitSampler = null;
      this._blitPipeline = null;
      this._blitPipelineMeta = null;
      this._blitBindGroup = null;
      this._blitBindGroupMeta = {
        texture: null,
        view: null,
        sampler: null,
        pipeline: null,
        device: null,
      };
      this._blitBindGroupCache = new Map();
      this._blitBindGroupEntry0 = { binding: 0, resource: null };
      this._blitBindGroupEntry1 = { binding: 1, resource: null };
      this._blitBindGroupDescriptor = {
        label: 'playgta6-swapchain-blit-bg',
        layout: undefined,
        entries: [this._blitBindGroupEntry0, this._blitBindGroupEntry1],
      };
      this._blitClearColor = Object.freeze({ r: 0, g: 0, b: 0, a: 1 });
      this._blitColorAttachment = {
        view: null,
        clearValue: this._blitClearColor,
        loadOp: 'clear',
        storeOp: 'store',
      };
      this._blitPassDescriptor = {
        label: 'playgta6-swapchain-blit-pass',
        colorAttachments: [this._blitColorAttachment],
      };
      this._blitCount = 0;

      // HDR tone-map blit pipeline & preallocated 16-byte uniform state (zero per-frame allocs).
      this._toneMapMode = resolvedToneMapMode;
      this._exposure = resolvedExposure;
      this._applySrgbEncodeOverride = resolvedApplySrgbEncode;
      this._toneMapShaderModule = null;
      this._toneMapPipeline = null;
      this._toneMapPipelineMeta = null;
      this._toneMapUniformBuffer = null;
      this._toneMapUniformBufferDevice = null;
      this._toneMapUniformData = new ArrayBuffer(TONEMAP_UNIFORM_BYTE_SIZE);
      this._toneMapUniformU32 = new Uint32Array(this._toneMapUniformData);
      this._toneMapUniformF32 = new Float32Array(this._toneMapUniformData);
      this._toneMapUniformDirty = true;
      this._lastToneMapPackedMode = -1;
      this._lastToneMapPackedExposure = Number.NaN;
      this._lastToneMapPackedSrgb = -1;
      this._toneMapBindGroup = null;
      this._toneMapBindGroupMeta = {
        texture: null,
        view: null,
        sampler: null,
        buffer: null,
        pipeline: null,
        device: null,
      };
      this._toneMapBindGroupCache = new Map();
      this._toneMapBindGroupEntry0 = { binding: 0, resource: null };
      this._toneMapBindGroupEntry1 = { binding: 1, resource: null };
      this._toneMapBindGroupBufferBinding = {
        buffer: null,
        offset: 0,
        size: TONEMAP_UNIFORM_BYTE_SIZE,
      };
      this._toneMapBindGroupEntry2 = {
        binding: 2,
        resource: this._toneMapBindGroupBufferBinding,
      };
      this._toneMapBindGroupDescriptor = {
        label: 'playgta6-swapchain-tonemap-bg',
        layout: undefined,
        entries: [
          this._toneMapBindGroupEntry0,
          this._toneMapBindGroupEntry1,
          this._toneMapBindGroupEntry2,
        ],
      };
      this._toneMapBlitCount = 0;
      this._toneMapUniformUploads = 0;

      this._isDeviceLost = false;
      this._lastDeviceLostInfo = null;
      this._deviceLostCallbacks = new Set();
      this._deviceLostGeneration = 0;
      this._hookedDevice = null;

      this._errorHookedDevice = null;
      this._boundUncapturedErrorHandler = null;
      this._validationErrors = [];
      this._validationErrorCount = 0;
      this._validationErrorSeq = 0;

      this._totalFrames = 0;
      this._presentCount = 0;
      this._droppedFrameCount = 0;
      this._resizeCount = 0;
      this._reconfigureCount = 0;
      this._skippedRedundantReconfigures = 0;
      this._lastFrameTimestamp = null;
      this._lastPresentIntervalMs = 0;
      this._lastFrameMs = 0;
      this._avgFrameMs = 0;
      this._maxFrameMs = 0;
      this._fps = 0;
      this._fps30ComplianceRate = 0;
      this._frameJitterMs = 0;
      this._longestStallMs = 0;
      this._frameDeltaRing = new Float64Array(frameWindowSize);
      this._frameDeltaHead = 0;
      this._frameDeltaCount = 0;

      // Zero-GPU-readback visible-content readiness tracking (W231).
      this._hasVisibleContent = false;
      this._firstVisiblePresentIndex = null;
      this._firstVisiblePresentMs = null;
      this._consecutiveBlackPresents = 0;
      this._blackPresentsTotal = 0;
      this._visiblePresentsTotal = 0;
      this._firstVisibleFired = false;
      this._pendingFrameVisible = false;
      this._pendingRealPackDraws = 0;
      this._pendingContentTimestampMs = null;
      this._hasPendingFrameContent = false;

      this._liveTelemetry = this._createTelemetryObject();

      this._hookDeviceLost(this._device);
      this._hookUncapturedError(this._device);
    }

    get width() {
      return this._width;
    }

    get height() {
      return this._height;
    }

    get backingWidth() {
      return this._width;
    }

    get backingHeight() {
      return this._height;
    }

    get requestedWidth() {
      return this._requestedWidth;
    }

    get requestedHeight() {
      return this._requestedHeight;
    }

    get displayBounds() {
      return this._displayBounds ? { ...this._displayBounds } : null;
    }

    get clampedToDisplayCount() {
      return this._clampedToDisplayCount;
    }

    get format() {
      return this._format;
    }

    get canvasFormat() {
      return this._format;
    }

    get streamFormat() {
      return this._streamFormat;
    }

    get viewFormats() {
      return this._activeViewFormats.slice();
    }

    get usage() {
      return this._usage;
    }

    get hasCopySrcUsage() {
      return (this._usage & TEXTURE_USAGE_COPY_SRC) !== 0;
    }

    get frameIntervalUs() {
      return this._frameIntervalUs;
    }

    get presentCount() {
      return this._presentCount;
    }

    get reconfigureCount() {
      return this._reconfigureCount;
    }

    get droppedFrameCount() {
      return this._droppedFrameCount;
    }

    get lastFrameMs() {
      return this._lastFrameMs;
    }

    get avgFrameMs() {
      return this._avgFrameMs;
    }

    get maxFrameMs() {
      return this._maxFrameMs;
    }

    get fps30ComplianceRate() {
      return this._fps30ComplianceRate;
    }

    get frameJitterMs() {
      return this._frameJitterMs;
    }

    get longestStallMs() {
      return this._longestStallMs;
    }

    get hasVisibleContent() {
      return this._hasVisibleContent;
    }

    get firstVisiblePresentIndex() {
      return this._firstVisiblePresentIndex;
    }

    get firstVisiblePresentMs() {
      return this._firstVisiblePresentMs;
    }

    get consecutiveBlackPresents() {
      return this._consecutiveBlackPresents;
    }

    get blackPresentsTotal() {
      return this._blackPresentsTotal;
    }

    get visiblePresentsTotal() {
      return this._visiblePresentsTotal;
    }

    get usesBlitFallback() {
      return this._usesBlitFallback;
    }

    get blitCount() {
      return this._blitCount;
    }

    get toneMapMode() {
      return this._toneMapMode;
    }

    get exposure() {
      return this._exposure;
    }

    get applySrgbEncode() {
      return this._resolveEffectiveApplySrgbEncode(this._streamFormat);
    }

    get isHdrStream() {
      return isHdrStreamFormat(this._streamFormat);
    }

    get toneMapActive() {
      return this._shouldUseToneMapPipeline(this._streamFormat);
    }

    get toneMapBlitCount() {
      return this._toneMapBlitCount;
    }

    get toneMapUniformUploads() {
      return this._toneMapUniformUploads;
    }

    get intermediateTexture() {
      return this._intermediateTexture;
    }

    get intermediateTextureView() {
      return this._intermediateTextureView;
    }

    get depthFormat() {
      return this._enableDepth ? this._depthFormat : null;
    }

    get alphaMode() {
      return this._alphaMode;
    }

    get enableDepth() {
      return this._enableDepth;
    }

    get device() {
      return this._device;
    }

    get context() {
      return this._context;
    }

    get canvas() {
      return this._canvas;
    }

    get depthTexture() {
      return this._depthTexture;
    }

    get depthTextureView() {
      return this._depthTextureView;
    }

    get isConfigured() {
      return this._configured;
    }

    get isDeviceLost() {
      return this._isDeviceLost;
    }

    get isDestroyed() {
      return this._destroyed;
    }

    _effectiveMaxDimension() {
      const deviceLimit =
        this._device &&
        this._device.limits &&
        Number.isInteger(this._device.limits.maxTextureDimension2D) &&
        this._device.limits.maxTextureDimension2D > 0
          ? this._device.limits.maxTextureDimension2D
          : this._maxDimension;
      return Math.min(this._maxDimension, deviceLimit);
    }

    _destroyDepthTexture() {
      if (this._depthTexture) {
        if (typeof this._depthTexture.destroy === 'function') {
          this._depthTexture.destroy();
        }
        this._depthTexture = null;
      }
      this._depthTextureView = null;
      this._depthTextureMeta = null;
    }

    _ensureDepthTexture() {
      if (!this._enableDepth) {
        this._destroyDepthTexture();
        return null;
      }
      if (this._width < 1 || this._height < 1 || !this._device) {
        return null;
      }

      const meta = this._depthTextureMeta;
      if (
        this._depthTexture &&
        this._depthTextureView &&
        meta &&
        meta.width === this._width &&
        meta.height === this._height &&
        meta.format === this._depthFormat &&
        meta.device === this._device
      ) {
        return this._depthTexture;
      }

      // Destroy previous depth texture BEFORE allocating the new one to avoid VRAM leaks.
      this._destroyDepthTexture();

      const texture = this._device.createTexture({
        label: 'playgta6-swapchain-depth',
        size: {
          width: this._width,
          height: this._height,
          depthOrArrayLayers: 1,
        },
        format: this._depthFormat,
        usage: TEXTURE_USAGE_RENDER_ATTACHMENT,
      });

      this._depthTexture = texture;
      this._depthTextureView =
        texture && typeof texture.createView === 'function' ? texture.createView() : null;
      this._depthTextureMeta = {
        width: this._width,
        height: this._height,
        format: this._depthFormat,
        device: this._device,
      };
      return texture;
    }

    _invalidateBlitBindGroup() {
      this._blitBindGroup = null;
      const meta = this._blitBindGroupMeta;
      meta.texture = null;
      meta.view = null;
      meta.sampler = null;
      meta.pipeline = null;
      meta.device = null;
      if (this._blitBindGroupCache.size > 0) {
        for (const bySampler of this._blitBindGroupCache.values()) {
          bySampler.clear();
        }
        this._blitBindGroupCache.clear();
      }
      this._blitBindGroupDescriptor.layout = undefined;
      this._blitBindGroupEntry0.resource = null;
      this._blitBindGroupEntry1.resource = null;

      this._toneMapBindGroup = null;
      const tmMeta = this._toneMapBindGroupMeta;
      tmMeta.texture = null;
      tmMeta.view = null;
      tmMeta.sampler = null;
      tmMeta.buffer = null;
      tmMeta.pipeline = null;
      tmMeta.device = null;
      if (this._toneMapBindGroupCache.size > 0) {
        for (const bySampler of this._toneMapBindGroupCache.values()) {
          bySampler.clear();
        }
        this._toneMapBindGroupCache.clear();
      }
      this._toneMapBindGroupDescriptor.layout = undefined;
      this._toneMapBindGroupEntry0.resource = null;
      this._toneMapBindGroupEntry1.resource = null;
      this._toneMapBindGroupBufferBinding.buffer = null;

      this._blitColorAttachment.view = null;
    }

    _destroyIntermediateTexture() {
      if (this._intermediateTexture) {
        if (typeof this._intermediateTexture.destroy === 'function') {
          this._intermediateTexture.destroy();
        }
        this._intermediateTexture = null;
      }
      this._intermediateTextureView = null;
      this._intermediateTextureMeta = null;
      this._invalidateBlitBindGroup();
    }

    _destroyBlitResources() {
      this._destroyIntermediateTexture();
      this._blitPipeline = null;
      this._blitPipelineMeta = null;
      this._blitShaderModule = null;
      this._blitSrgbShaderModule = null;
      this._blitSampler = null;
      this._toneMapPipeline = null;
      this._toneMapPipelineMeta = null;
      this._toneMapShaderModule = null;
      if (this._toneMapUniformBuffer) {
        if (typeof this._toneMapUniformBuffer.destroy === 'function') {
          try {
            this._toneMapUniformBuffer.destroy();
          } catch (_) {
            // Ignore device-lost teardown errors.
          }
        }
        this._toneMapUniformBuffer = null;
      }
      this._toneMapUniformBufferDevice = null;
      this._toneMapUniformDirty = true;
      this._lastToneMapPackedMode = -1;
      this._lastToneMapPackedExposure = Number.NaN;
      this._lastToneMapPackedSrgb = -1;
    }

    _resolveEffectiveApplySrgbEncode(sourceFormat = this._streamFormat) {
      if (this._applySrgbEncodeOverride !== null) {
        return this._applySrgbEncodeOverride;
      }
      if (!SDR_NON_SRGB_CANVAS_FORMAT_SET.has(this._format)) {
        return false;
      }
      // Sampling an `*-srgb` intermediate texture performs hardware sRGB -> linear decode, and
      // linear HDR formats (`rgba16float`, `rgb10a2unorm`, `rg11b10ufloat`, ...) already carry
      // linear values. When presenting onto a non-sRGB SDR canvas (`bgra8unorm` / `rgba8unorm`),
      // exact IEC 61966-2-1 linear -> sRGB encoding MUST run once so midtones are never double-gamma
      // crushed (~x^4.84) and pre-encoded SDR (`bgra8unorm` / `rgba8unorm`) is never double-encoded.
      return isHdrStreamFormat(sourceFormat) || isSrgbFormat(sourceFormat);
    }

    _shouldUseToneMapPipeline(sourceFormat = this._streamFormat) {
      const wantsToneMap =
        this._toneMapMode === 'reinhard' ||
        this._toneMapMode === 'aces' ||
        this._exposure !== DEFAULT_EXPOSURE ||
        this._resolveEffectiveApplySrgbEncode(sourceFormat);
      if (!wantsToneMap) {
        return false;
      }
      if (!this._device || typeof this._device.createBuffer !== 'function') {
        return false;
      }
      return true;
    }

    /**
     * Configure HDR tone-mapping parameters for the intermediate-to-canvas blit pass.
     * Supports `'none'` / `'passthrough'`, `'reinhard'`, and `'aces'` filmic tone-mapping,
     * configurable `exposure`, and explicit or auto (`null`) linear-to-sRGB encoding.
     *
     * @param {{
     *   mode?: 'none' | 'passthrough' | 'reinhard' | 'aces',
     *   toneMapMode?: 'none' | 'passthrough' | 'reinhard' | 'aces',
     *   exposure?: number,
     *   applySrgbEncode?: boolean | null,
     * }} options
     */
    setToneMapOptions(options) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('setToneMapOptions requires an options object');
      }
      const rawMode = options.mode !== undefined ? options.mode : options.toneMapMode;
      const nextMode = rawMode !== undefined ? normalizeToneMapMode(rawMode) : this._toneMapMode;
      const nextExposure =
        options.exposure !== undefined ? normalizeExposure(options.exposure) : this._exposure;
      const nextApplySrgb =
        options.applySrgbEncode !== undefined
          ? normalizeApplySrgbEncode(options.applySrgbEncode)
          : this._applySrgbEncodeOverride;

      const changed =
        nextMode !== this._toneMapMode ||
        nextExposure !== this._exposure ||
        nextApplySrgb !== this._applySrgbEncodeOverride;

      this._toneMapMode = nextMode;
      this._exposure = nextExposure;
      this._applySrgbEncodeOverride = nextApplySrgb;
      if (changed) {
        this._toneMapUniformDirty = true;
      }

      return {
        toneMapMode: this._toneMapMode,
        exposure: this._exposure,
        applySrgbEncode: this._resolveEffectiveApplySrgbEncode(this._streamFormat),
        toneMapActive: this._shouldUseToneMapPipeline(this._streamFormat),
      };
    }

    _ensureBlitSampler() {
      if (!this._blitSampler || (this._blitPipelineMeta && this._blitPipelineMeta.device !== this._device)) {
        if (!this._blitSampler || (this._toneMapPipelineMeta && this._toneMapPipelineMeta.device !== this._device)) {
          this._blitSampler = this._device.createSampler({
            label: 'playgta6-swapchain-blit-sampler',
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
          });
        }
      }
      return this._blitSampler;
    }

    _ensureToneMapUniformBuffer(sourceFormat = this._streamFormat) {
      if (!this._toneMapUniformBuffer || this._toneMapUniformBufferDevice !== this._device) {
        if (this._toneMapUniformBuffer && typeof this._toneMapUniformBuffer.destroy === 'function') {
          try {
            this._toneMapUniformBuffer.destroy();
          } catch (_) {
            // ignore teardown errors
          }
        }
        this._toneMapUniformBuffer = this._device.createBuffer({
          label: 'playgta6-swapchain-tonemap-params',
          size: TONEMAP_UNIFORM_BYTE_SIZE,
          usage: BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST,
        });
        this._toneMapUniformBufferDevice = this._device;
        this._toneMapUniformDirty = true;
        this._lastToneMapPackedMode = -1;
        this._lastToneMapPackedExposure = Number.NaN;
        this._lastToneMapPackedSrgb = -1;
        this._invalidateBlitBindGroup();
      }

      const modeCode = TONE_MAP_MODE_CODES[this._toneMapMode] || 0;
      const exposure = this._exposure;
      const srgbFlag = this._resolveEffectiveApplySrgbEncode(sourceFormat) ? 1 : 0;

      if (
        this._toneMapUniformDirty ||
        modeCode !== this._lastToneMapPackedMode ||
        exposure !== this._lastToneMapPackedExposure ||
        srgbFlag !== this._lastToneMapPackedSrgb
      ) {
        this._toneMapUniformU32[0] = modeCode;
        this._toneMapUniformF32[1] = exposure;
        this._toneMapUniformU32[2] = srgbFlag;
        this._toneMapUniformU32[3] = 0;

        if (this._device && this._device.queue && typeof this._device.queue.writeBuffer === 'function') {
          this._device.queue.writeBuffer(this._toneMapUniformBuffer, 0, this._toneMapUniformData);
          this._toneMapUniformUploads += 1;
        }
        this._lastToneMapPackedMode = modeCode;
        this._lastToneMapPackedExposure = exposure;
        this._lastToneMapPackedSrgb = srgbFlag;
        this._toneMapUniformDirty = false;
      }

      return this._toneMapUniformBuffer;
    }

    _ensureToneMapPipeline() {
      const meta = this._toneMapPipelineMeta;
      if (
        this._toneMapPipeline &&
        this._toneMapShaderModule &&
        this._blitSampler &&
        meta &&
        meta.device === this._device &&
        meta.canvasFormat === this._format
      ) {
        return this._toneMapPipeline;
      }

      if (!meta || meta.device !== this._device) {
        this._toneMapShaderModule = this._device.createShaderModule({
          label: 'playgta6-swapchain-tonemap-wgsl',
          code: BLIT_TONEMAP_WGSL,
        });
      }
      if (!this._blitSampler) {
        this._blitSampler = this._device.createSampler({
          label: 'playgta6-swapchain-blit-sampler',
          magFilter: 'linear',
          minFilter: 'linear',
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
        });
      }

      this._toneMapPipeline = this._device.createRenderPipeline({
        label: 'playgta6-swapchain-tonemap-pipeline',
        layout: 'auto',
        vertex: {
          module: this._toneMapShaderModule,
          entryPoint: 'vs_main',
        },
        fragment: {
          module: this._toneMapShaderModule,
          entryPoint: 'fs_main',
          targets: [{ format: this._format }],
        },
        primitive: {
          topology: 'triangle-list',
          cullMode: 'none',
        },
      });
      this._toneMapPipelineMeta = {
        device: this._device,
        canvasFormat: this._format,
      };
      this._invalidateBlitBindGroup();
      return this._toneMapPipeline;
    }

    _ensureToneMapBindGroup(
      sourceView = this._intermediateTextureView,
      sampler = this._blitSampler,
      uniformBuffer = this._toneMapUniformBuffer
    ) {
      const pipeline = this._toneMapPipeline;
      const device = this._device;
      const texture =
        sourceView === this._intermediateTextureView ? this._intermediateTexture : null;
      const meta = this._toneMapBindGroupMeta;

      if (
        this._toneMapBindGroup &&
        meta.view === sourceView &&
        meta.sampler === sampler &&
        meta.buffer === uniformBuffer &&
        meta.pipeline === pipeline &&
        meta.device === device &&
        meta.texture === texture
      ) {
        return this._toneMapBindGroup;
      }

      let bySampler = this._toneMapBindGroupCache.get(sourceView);
      if (bySampler) {
        const cached = bySampler.get(sampler);
        if (
          cached &&
          cached.pipeline === pipeline &&
          cached.device === device &&
          cached.buffer === uniformBuffer
        ) {
          this._toneMapBindGroup = cached.bindGroup;
          meta.texture = texture;
          meta.view = sourceView;
          meta.sampler = sampler;
          meta.buffer = uniformBuffer;
          meta.pipeline = pipeline;
          meta.device = device;
          return cached.bindGroup;
        }
      }

      const layout =
        pipeline && typeof pipeline.getBindGroupLayout === 'function'
          ? pipeline.getBindGroupLayout(0)
          : undefined;

      this._toneMapBindGroupDescriptor.layout = layout;
      this._toneMapBindGroupEntry0.resource = sourceView;
      this._toneMapBindGroupEntry1.resource = sampler;
      this._toneMapBindGroupBufferBinding.buffer = uniformBuffer;

      const bindGroup = device.createBindGroup(this._toneMapBindGroupDescriptor);

      let totalCached = 0;
      for (const map of this._toneMapBindGroupCache.values()) {
        totalCached += map.size;
      }
      if (totalCached >= MAX_BLIT_BIND_GROUP_CACHE) {
        for (const map of this._toneMapBindGroupCache.values()) {
          map.clear();
        }
        this._toneMapBindGroupCache.clear();
        bySampler = undefined;
      }
      if (!bySampler) {
        bySampler = new Map();
        this._toneMapBindGroupCache.set(sourceView, bySampler);
      }
      bySampler.set(sampler, { pipeline, device, buffer: uniformBuffer, bindGroup });

      this._toneMapBindGroup = bindGroup;
      meta.texture = texture;
      meta.view = sourceView;
      meta.sampler = sampler;
      meta.buffer = uniformBuffer;
      meta.pipeline = pipeline;
      meta.device = device;
      return bindGroup;
    }

    _ensureIntermediateTexture() {
      if (!this._usesBlitFallback) {
        this._destroyIntermediateTexture();
        return null;
      }
      if (this._width < 1 || this._height < 1 || !this._device) {
        return null;
      }

      const meta = this._intermediateTextureMeta;
      if (
        this._intermediateTexture &&
        this._intermediateTextureView &&
        meta &&
        meta.width === this._width &&
        meta.height === this._height &&
        meta.format === this._streamFormat &&
        meta.device === this._device
      ) {
        return this._intermediateTexture;
      }

      // Destroy previous intermediate texture BEFORE allocating the new one to avoid VRAM leaks.
      this._destroyIntermediateTexture();

      const texture = this._device.createTexture({
        label: 'playgta6-swapchain-intermediate',
        size: {
          width: this._width,
          height: this._height,
          depthOrArrayLayers: 1,
        },
        format: this._streamFormat,
        usage: INTERMEDIATE_TEXTURE_USAGE,
      });

      this._intermediateTexture = texture;
      this._intermediateTextureView =
        texture && typeof texture.createView === 'function' ? texture.createView() : null;
      this._intermediateTextureMeta = {
        width: this._width,
        height: this._height,
        format: this._streamFormat,
        device: this._device,
      };
      return texture;
    }

    _ensureBlitPipeline(applySrgb = false) {
      const needsSrgbEncode = Boolean(applySrgb);
      const meta = this._blitPipelineMeta;
      const activeShaderModule = needsSrgbEncode
        ? this._blitSrgbShaderModule
        : this._blitShaderModule;
      if (
        this._blitPipeline &&
        activeShaderModule &&
        this._blitSampler &&
        meta &&
        meta.device === this._device &&
        meta.canvasFormat === this._format &&
        meta.applySrgb === needsSrgbEncode
      ) {
        return this._blitPipeline;
      }

      if (!meta || meta.device !== this._device) {
        this._blitShaderModule = null;
        this._blitSrgbShaderModule = null;
        if (!this._blitSampler || (meta && meta.device !== this._device)) {
          this._blitSampler = this._device.createSampler({
            label: 'playgta6-swapchain-blit-sampler',
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
          });
        }
      }

      let shaderModule;
      if (needsSrgbEncode) {
        if (!this._blitSrgbShaderModule) {
          this._blitSrgbShaderModule = this._device.createShaderModule({
            label: 'playgta6-swapchain-blit-srgb-wgsl',
            code: BLIT_SRGB_ENCODE_WGSL,
          });
        }
        shaderModule = this._blitSrgbShaderModule;
      } else {
        if (!this._blitShaderModule) {
          this._blitShaderModule = this._device.createShaderModule({
            label: 'playgta6-swapchain-blit-wgsl',
            code: BLIT_FULLSCREEN_WGSL,
          });
        }
        shaderModule = this._blitShaderModule;
      }

      this._blitPipeline = this._device.createRenderPipeline({
        label: needsSrgbEncode
          ? 'playgta6-swapchain-blit-srgb-pipeline'
          : 'playgta6-swapchain-blit-pipeline',
        layout: 'auto',
        vertex: {
          module: shaderModule,
          entryPoint: 'vs_main',
        },
        fragment: {
          module: shaderModule,
          entryPoint: 'fs_main',
          targets: [{ format: this._format }],
        },
        primitive: {
          topology: 'triangle-list',
          cullMode: 'none',
        },
      });
      this._blitPipelineMeta = {
        device: this._device,
        canvasFormat: this._format,
        applySrgb: needsSrgbEncode,
      };
      this._invalidateBlitBindGroup();
      return this._blitPipeline;
    }

    _ensureBlitBindGroup(
      sourceView = this._intermediateTextureView,
      sampler = this._blitSampler
    ) {
      const pipeline = this._blitPipeline;
      const device = this._device;
      const texture =
        sourceView === this._intermediateTextureView ? this._intermediateTexture : null;
      const meta = this._blitBindGroupMeta;

      if (
        this._blitBindGroup &&
        meta.view === sourceView &&
        meta.sampler === sampler &&
        meta.pipeline === pipeline &&
        meta.device === device &&
        meta.texture === texture
      ) {
        return this._blitBindGroup;
      }

      let bySampler = this._blitBindGroupCache.get(sourceView);
      if (bySampler) {
        const cached = bySampler.get(sampler);
        if (cached && cached.pipeline === pipeline && cached.device === device) {
          this._blitBindGroup = cached.bindGroup;
          meta.texture = texture;
          meta.view = sourceView;
          meta.sampler = sampler;
          meta.pipeline = pipeline;
          meta.device = device;
          return cached.bindGroup;
        }
      }

      const layout =
        pipeline && typeof pipeline.getBindGroupLayout === 'function'
          ? pipeline.getBindGroupLayout(0)
          : undefined;

      this._blitBindGroupDescriptor.layout = layout;
      this._blitBindGroupEntry0.resource = sourceView;
      this._blitBindGroupEntry1.resource = sampler;

      const bindGroup = device.createBindGroup(this._blitBindGroupDescriptor);

      let totalCached = 0;
      for (const map of this._blitBindGroupCache.values()) {
        totalCached += map.size;
      }
      if (totalCached >= MAX_BLIT_BIND_GROUP_CACHE) {
        for (const map of this._blitBindGroupCache.values()) {
          map.clear();
        }
        this._blitBindGroupCache.clear();
        bySampler = undefined;
      }
      if (!bySampler) {
        bySampler = new Map();
        this._blitBindGroupCache.set(sourceView, bySampler);
      }
      bySampler.set(sampler, { pipeline, device, bindGroup });

      this._blitBindGroup = bindGroup;
      meta.texture = texture;
      meta.view = sourceView;
      meta.sampler = sampler;
      meta.pipeline = pipeline;
      meta.device = device;
      return bindGroup;
    }

    _hookDeviceLost(device) {
      if (!device || device === this._hookedDevice) {
        return;
      }
      if (!device.lost || typeof device.lost.then !== 'function') {
        this._hookedDevice = device;
        return;
      }

      this._hookedDevice = device;
      const generation = this._deviceLostGeneration;

      device.lost.then((info) => {
        if (
          this._destroyed ||
          generation !== this._deviceLostGeneration ||
          this._device !== device
        ) {
          return;
        }
        this._isDeviceLost = true;
        this._lastDeviceLostInfo = info || { reason: 'unknown', message: '' };
        this._destroyDepthTexture();
        this._destroyBlitResources();

        for (const callback of Array.from(this._deviceLostCallbacks)) {
          callback(this._lastDeviceLostInfo, this);
        }
      });
    }

    _unhookUncapturedError() {
      const prevDevice = this._errorHookedDevice;
      const prevHandler = this._boundUncapturedErrorHandler;
      this._errorHookedDevice = null;
      this._boundUncapturedErrorHandler = null;

      if (
        prevDevice &&
        prevHandler &&
        typeof prevDevice.removeEventListener === 'function'
      ) {
        try {
          prevDevice.removeEventListener('uncapturederror', prevHandler);
        } catch {
          // Ignore removal failures on torn-down or mock devices.
        }
      }
    }

    _hookUncapturedError(device) {
      if (!device) {
        this._unhookUncapturedError();
        return;
      }
      if (device === this._errorHookedDevice && this._boundUncapturedErrorHandler) {
        return;
      }

      this._unhookUncapturedError();

      if (typeof device.addEventListener !== 'function') {
        return;
      }

      const handler = (event) => {
        if (this._destroyed || this._device !== device) {
          return;
        }
        this._handleUncapturedError(event);
      };

      this._errorHookedDevice = device;
      this._boundUncapturedErrorHandler = handler;
      device.addEventListener('uncapturederror', handler);
    }

    _handleUncapturedError(event) {
      const rawError =
        event && typeof event === 'object' && 'error' in event ? event.error : event;
      const type = normalizeValidationErrorType(rawError, event);

      let rawMessage = '';
      if (rawError && typeof rawError === 'object' && typeof rawError.message === 'string') {
        rawMessage = rawError.message;
      } else if (event && typeof event === 'object' && typeof event.message === 'string') {
        rawMessage = event.message;
      } else if (typeof rawError === 'string') {
        rawMessage = rawError;
      } else if (rawError !== null && rawError !== undefined) {
        rawMessage = String(rawError);
      }

      const message = sanitizeValidationErrorMessage(rawMessage);
      const nowVal = this._now();
      const timestampMs =
        typeof nowVal === 'number' && Number.isFinite(nowVal) ? nowVal : Date.now();

      this._validationErrorSeq += 1;
      this._validationErrorCount += 1;

      const record = {
        seq: this._validationErrorSeq,
        type,
        message,
        timestampMs,
      };

      this._validationErrors.push(record);
      if (this._validationErrors.length > this._maxErrorHistory) {
        this._validationErrors.shift();
      }

      if (typeof this._onValidationError === 'function') {
        try {
          this._onValidationError({ ...record });
        } catch {
          // Isolate callback exceptions so validation error capture never crashes the worker.
        }
      }

      return record;
    }

    _resolveEffectiveViewFormats(format = this._format, explicitViewFormats = this._explicitViewFormats) {
      if (explicitViewFormats !== null && explicitViewFormats !== undefined) {
        return explicitViewFormats.length > 0 ? explicitViewFormats.slice() : null;
      }
      const compatible = getCompatibleViewFormats(format);
      return compatible.length > 0 ? compatible : null;
    }

    _resolveEffectiveViewFormatsKey(
      format = this._format,
      explicitViewFormats = this._explicitViewFormats
    ) {
      if (explicitViewFormats !== null && explicitViewFormats !== undefined) {
        if (explicitViewFormats === this._explicitViewFormats) {
          return this._explicitViewFormatsKey;
        }
        return explicitViewFormats.length > 0 ? explicitViewFormats.join(',') : '';
      }
      return COMPATIBLE_VIEW_FORMATS_KEY[format] || '';
    }

    _isSameAppliedConfig(width, height, format, usage, alphaMode, viewFormatsKey) {
      const prev = this._appliedConfig;
      return Boolean(
        this._configured &&
        this._hasAppliedConfig &&
        prev.device === this._device &&
        prev.format === format &&
        prev.usage === usage &&
        prev.alphaMode === alphaMode &&
        prev.width === width &&
        prev.height === height &&
        prev.viewFormatsKey === viewFormatsKey
      );
    }

    _replaceDevice(newDevice) {
      this._deviceLostGeneration += 1;
      this._unhookUncapturedError();
      this._destroyDepthTexture();
      this._destroyBlitResources();

      this._device = newDevice;
      this._isDeviceLost = false;
      this._lastDeviceLostInfo = null;
      this._hookedDevice = null;
      this._lastFrameTimestamp = null;
      this._hasAppliedConfig = false;
      this._appliedConfig.device = null;
      this._copySrcSupported = null;

      this._hookDeviceLost(newDevice);
      this._hookUncapturedError(newDevice);
    }

    /**
     * Replace the bound GPUDevice. Elides reconfiguration when the same live device is already
     * configured; otherwise tears down device-bound depth/blit resources, hooks the new device's
     * lost/uncapturederror listeners, and reconfigures the canvas context if already configured.
     *
     * @param {GPUDevice} newDevice
     * @returns {boolean} True when the device changed and was applied, false on no-op.
     */
    setDevice(newDevice) {
      if (this._destroyed) {
        throw new Error('Cannot set device on a destroyed WgpuSwapchain');
      }
      if (!newDevice || typeof newDevice !== 'object') {
        throw new TypeError('setDevice requires a GPUDevice');
      }
      if (
        newDevice === this._device &&
        !this._isDeviceLost &&
        this._configured &&
        this._hasAppliedConfig
      ) {
        this._skippedRedundantReconfigures += 1;
        return false;
      }

      this._replaceDevice(newDevice);

      if (this._configured && this._width > 0 && this._height > 0) {
        const maxDim = this._effectiveMaxDimension();
        const reqW = Math.min(this._requestedWidth > 0 ? this._requestedWidth : this._width, maxDim);
        const reqH = Math.min(this._requestedHeight > 0 ? this._requestedHeight : this._height, maxDim);
        const clamped = clampToDisplayBounds(reqW, reqH, this._displayBounds, maxDim);
        this._requestedWidth = reqW;
        this._requestedHeight = reqH;
        this._width = clamped.width;
        this._height = clamped.height;
        this._applyContextConfiguration();
      }
      return true;
    }

    _resolveEffectiveUsage() {
      const baseUsage = this._explicitUsage !== null ? this._explicitUsage : DEFAULT_USAGE;
      if (
        this._preserveCopySrcUntilVisible &&
        !this._firstVisibleFired &&
        this._copySrcSupported !== false
      ) {
        return baseUsage | TEXTURE_USAGE_COPY_SRC;
      }
      return baseUsage;
    }

    /**
     * W350: Ensure `GPUTextureUsage.COPY_SRC` is enabled on the configured swapchain canvas
     * texture (e.g. for async non-black luminance probing prior to first-visible-frame).
     * Returns `true` when `COPY_SRC` is active, or `false` if unsupported/unconfigured.
     *
     * @returns {boolean}
     */
    ensureCopySrcUsage() {
      if (
        this._destroyed ||
        this._isDeviceLost ||
        !this._configured ||
        !this._device ||
        !this._context
      ) {
        return false;
      }
      if (
        this._preserveCopySrcUntilVisible &&
        (this._firstVisibleFired || this._hasVisibleContent)
      ) {
        return false;
      }
      if ((this._usage & TEXTURE_USAGE_COPY_SRC) !== 0) {
        this._copySrcSupported = true;
        return true;
      }
      if (this._copySrcSupported === false) {
        return false;
      }
      const nextUsage = (this._usage || DEFAULT_USAGE) | TEXTURE_USAGE_COPY_SRC;
      try {
        this.configure({ usage: nextUsage });
        return this.hasCopySrcUsage;
      } catch (_) {
        this._copySrcSupported = false;
        return false;
      }
    }

    _applyContextConfiguration() {
      if (this._canvas.width !== this._width) {
        this._canvas.width = this._width;
      }
      if (this._canvas.height !== this._height) {
        this._canvas.height = this._height;
      }
      const effectiveViewFormats = this._resolveEffectiveViewFormats();
      const effectiveViewFormatsKey = this._resolveEffectiveViewFormatsKey();
      const autoPreservingCopySrc = Boolean(
        this._preserveCopySrcUntilVisible &&
        !this._firstVisibleFired &&
        this._copySrcSupported !== false
      );
      const explicitHasCopySrc = Boolean(
        this._explicitUsage !== null && (this._explicitUsage & TEXTURE_USAGE_COPY_SRC) !== 0
      );
      let requestedUsage =
        this._explicitUsage !== null ? this._explicitUsage : null;
      if (autoPreservingCopySrc) {
        requestedUsage =
          (this._explicitUsage !== null ? this._explicitUsage : DEFAULT_USAGE) |
          TEXTURE_USAGE_COPY_SRC;
      }
      const cfg = {
        device: this._device,
        format: this._format,
        alphaMode: this._alphaMode,
      };
      if (requestedUsage !== null) {
        cfg.usage = requestedUsage;
      }
      let appliedViewFormats = [];
      if (effectiveViewFormats && effectiveViewFormats.length > 0) {
        cfg.viewFormats = effectiveViewFormats.slice();
        appliedViewFormats = effectiveViewFormats.slice();
      }
      const canFallbackDropAutoCopySrc = Boolean(
        autoPreservingCopySrc && !explicitHasCopySrc && requestedUsage !== null
      );
      const configureWithFallback = (configObj, allowViewFormatsRetry) => {
        try {
          this._context.configure(configObj);
          return {
            viewFormats: configObj.viewFormats ? configObj.viewFormats.slice() : [],
            usedViewFallback: false,
          };
        } catch (err) {
          if (allowViewFormatsRetry && configObj.viewFormats) {
            const fallbackCfg = {
              device: this._device,
              format: this._format,
              alphaMode: this._alphaMode,
            };
            if (configObj.usage !== undefined) {
              fallbackCfg.usage = configObj.usage;
            }
            this._context.configure(fallbackCfg);
            return { viewFormats: [], usedViewFallback: true };
          }
          throw err;
        }
      };

      let appliedUsage = requestedUsage !== null ? requestedUsage : DEFAULT_USAGE;
      try {
        const res = configureWithFallback(cfg, true);
        appliedViewFormats = res.viewFormats;
        if (res.usedViewFallback) {
          this._viewFormatsFallbackCount += 1;
        }
        if ((appliedUsage & TEXTURE_USAGE_COPY_SRC) !== 0) {
          this._copySrcSupported = true;
        }
      } catch (err) {
        if (canFallbackDropAutoCopySrc) {
          const fallbackUsage =
            this._explicitUsage !== null
              ? this._explicitUsage
              : requestedUsage & ~TEXTURE_USAGE_COPY_SRC || DEFAULT_USAGE;
          const noCopyCfg = {
            device: this._device,
            format: this._format,
            alphaMode: this._alphaMode,
            usage: fallbackUsage,
          };
          if (effectiveViewFormats && effectiveViewFormats.length > 0) {
            noCopyCfg.viewFormats = effectiveViewFormats.slice();
          }
          const res = configureWithFallback(noCopyCfg, true);
          appliedViewFormats = res.viewFormats;
          if (res.usedViewFallback) {
            this._viewFormatsFallbackCount += 1;
          }
          appliedUsage = fallbackUsage;
          this._copySrcSupported = false;
        } else {
          throw err;
        }
      }
      this._configured = true;
      this._usage = appliedUsage;
      this._activeViewFormats = appliedViewFormats;
      const applied = this._appliedConfig;
      applied.device = this._device;
      applied.format = this._format;
      applied.usage = this._usage;
      applied.alphaMode = this._alphaMode;
      applied.width = this._width;
      applied.height = this._height;
      applied.viewFormatsKey = effectiveViewFormatsKey;
      this._hasAppliedConfig = true;
      this._reconfigureCount += 1;
      this._invalidateBlitBindGroup();
      this._ensureDepthTexture();
      if (this._usesBlitFallback) {
        this._ensureIntermediateTexture();
      }
    }

    /**
     * Configure swapchain rendering for a stream-declared back-buffer format.
     *
     * Valid WebGPU canvas formats ('bgra8unorm', 'rgba8unorm', 'rgba16float') configure the
     * canvas context directly with zero extra textures or blit overhead.
     *
     * Non-canvas formats (e.g. 'rgb10a2unorm', 'rg11b10ufloat', 'rgba8unorm-srgb', 'rgba16unorm')
     * keep the canvas context configured in its valid preferred canvas format and activate an
     * offscreen intermediate render-target texture + fullscreen-triangle blit fallback.
     *
     * @param {string} streamFormat
     * @returns {boolean} True when format/mode changed and was applied, false on no-op or invalid.
     */
    configureForStreamFormat(streamFormat, options) {
      if (this._destroyed || this._isDeviceLost) {
        return false;
      }
      if (typeof streamFormat !== 'string' || streamFormat.length === 0) {
        return false;
      }

      const forceBlit = Boolean(
        (options && typeof options === 'object' && options.forceBlit) ||
          isHdrToneMapBlitRequested(
            streamFormat,
            this._format,
            this._toneMapMode,
            this._exposure,
            this._applySrgbEncodeOverride
          )
      );

      if (isValidCanvasFormat(streamFormat) && !forceBlit) {
        const hadBlit = this._usesBlitFallback;
        const canvasChanged = this._format !== streamFormat;
        const prevEffectiveKey = this._resolveEffectiveViewFormatsKey();
        const nextEffectiveKey = this._resolveEffectiveViewFormatsKey(streamFormat, null);
        const viewFormatsChanged = prevEffectiveKey !== nextEffectiveKey;
        const streamChanged = this._streamFormat !== streamFormat;
        const usageChanged = this._usage !== this._resolveEffectiveUsage();

        if (!hadBlit && !canvasChanged && !viewFormatsChanged && !streamChanged && !usageChanged && this._configured) {
          this._skippedRedundantReconfigures += 1;
          return false;
        }

        if (hadBlit) {
          this._usesBlitFallback = false;
          this._destroyBlitResources();
        }

        this._streamFormat = streamFormat;
        this._format = streamFormat;
        this._explicitViewFormats = null;
        this._viewFormats = null;
        this._explicitViewFormatsKey = '';

        if (
          this._configured &&
          this._device &&
          this._context &&
          (canvasChanged || viewFormatsChanged || usageChanged)
        ) {
          this._applyContextConfiguration();
        }
        return true;
      }

      // Non-canvas stream format: keep canvas configured in a valid canvas format and render
      // into an offscreen intermediate texture in `streamFormat`, blitting to canvas on present.
      const nextCanvasFormat = isValidCanvasFormat(this._format)
        ? this._format
        : this._preferredCanvasFormat;
      const canvasChanged = this._format !== nextCanvasFormat;
      const prevEffectiveKey = this._resolveEffectiveViewFormatsKey();
      const nextEffectiveKey = this._resolveEffectiveViewFormatsKey(nextCanvasFormat, null);
      const viewFormatsChanged = prevEffectiveKey !== nextEffectiveKey;
      const streamChanged = this._streamFormat !== streamFormat;
      const usageChanged = this._usage !== this._resolveEffectiveUsage();
      const needsIntermediateNow =
        this._configured &&
        this._width > 0 &&
        this._height > 0 &&
        (!this._intermediateTexture ||
          !this._intermediateTextureMeta ||
          this._intermediateTextureMeta.format !== streamFormat ||
          this._intermediateTextureMeta.width !== this._width ||
          this._intermediateTextureMeta.height !== this._height ||
          this._intermediateTextureMeta.device !== this._device);

      if (
        this._usesBlitFallback &&
        !streamChanged &&
        !canvasChanged &&
        !viewFormatsChanged &&
        !usageChanged &&
        !needsIntermediateNow
      ) {
        if (this._configured) {
          this._skippedRedundantReconfigures += 1;
        }
        return false;
      }

      this._format = nextCanvasFormat;
      this._explicitViewFormats = null;
      this._viewFormats = null;
      this._explicitViewFormatsKey = '';
      this._streamFormat = streamFormat;
      this._usesBlitFallback = true;

      if (canvasChanged) {
        this._blitPipeline = null;
        this._blitPipelineMeta = null;
        this._toneMapPipeline = null;
        this._toneMapPipelineMeta = null;
        this._toneMapUniformDirty = true;
        this._invalidateBlitBindGroup();
      }
      if (streamChanged) {
        this._toneMapUniformDirty = true;
      }

      if (this._configured && this._device && this._context) {
        if (canvasChanged || viewFormatsChanged || usageChanged) {
          this._applyContextConfiguration();
        } else {
          this._ensureIntermediateTexture();
        }
      }
      return true;
    }

    /**
     * Switch the configured canvas texture format (e.g. matching the stream-declared swapchain
     * attachment format such as 'rgba8unorm') without disturbing canvas dimensions or depth state.
     *
     * @param {string} format
     * @param {string[] | null} [viewFormats]
     * @returns {boolean} True when the format changed and was reapplied.
     */
    setFormat(format, viewFormats = null) {
      if (this._destroyed || this._isDeviceLost) {
        return false;
      }
      if (typeof format !== 'string' || format.length === 0) {
        return false;
      }
      const nextExplicitViewFormats =
        viewFormats !== null && viewFormats !== undefined
          ? normalizeViewFormats(viewFormats)
          : null;
      if (!isValidCanvasFormat(format) && (!nextExplicitViewFormats || nextExplicitViewFormats.length === 0)) {
        return this.configureForStreamFormat(format);
      }
      const prevViewKey = this._resolveEffectiveViewFormatsKey();
      const nextViewKey = this._resolveEffectiveViewFormatsKey(format, nextExplicitViewFormats);
      const nextEffectiveUsage = this._resolveEffectiveUsage();
      const hadBlit = this._usesBlitFallback;
      if (
        !hadBlit &&
        this._format === format &&
        prevViewKey === nextViewKey &&
        this._usage === nextEffectiveUsage &&
        this._configured
      ) {
        this._skippedRedundantReconfigures += 1;
        return false;
      }
      if (hadBlit) {
        this._usesBlitFallback = false;
        this._destroyBlitResources();
      }
      const canvasChanged = this._format !== format;
      this._format = format;
      this._streamFormat = format;
      this._explicitViewFormats = nextExplicitViewFormats;
      this._viewFormats = nextExplicitViewFormats;
      this._explicitViewFormatsKey = nextExplicitViewFormats ? nextExplicitViewFormats.join(',') : '';
      if (canvasChanged) {
        this._blitPipeline = null;
        this._blitPipelineMeta = null;
        this._toneMapPipeline = null;
        this._toneMapPipelineMeta = null;
        this._toneMapUniformDirty = true;
        this._invalidateBlitBindGroup();
      }
      if (
        this._configured &&
        this._device &&
        this._context &&
        !this._isSameAppliedConfig(
          this._width,
          this._height,
          this._format,
          nextEffectiveUsage,
          this._alphaMode,
          nextViewKey
        )
      ) {
        this._applyContextConfiguration();
      }
      return true;
    }

    /**
     * Configure canvas pixel dimensions, WebGPU canvas context, and depth-stencil attachment.
     *
     * @param {{
     *   width?: number,
     *   height?: number,
     *   devicePixelRatio?: number,
     *   device?: GPUDevice,
     *   format?: string,
     *   streamFormat?: string,
     *   viewFormats?: string[] | null,
     *   usage?: number | null,
     *   depthFormat?: string,
     *   alphaMode?: string,
     *   enableDepth?: boolean,
     *   frameIntervalUs?: number | null,
     *   droppedFrameThresholdMs?: number | null,
     * }} config
     */
    configure(config = {}) {
      if (this._destroyed) {
        throw new Error('Cannot configure a destroyed WgpuSwapchain');
      }
      if (!config || typeof config !== 'object') {
        throw new TypeError('configure requires a configuration object');
      }

      // Validate all incoming options before mutating swapchain state.
      if (config.device !== undefined) {
        if (!config.device || typeof config.device !== 'object') {
          throw new TypeError('device must be a GPUDevice');
        }
      }
      const isReplacingDevice = config.device !== undefined && config.device !== this._device;
      if (this._isDeviceLost && !isReplacingDevice) {
        throw new Error('Cannot configure WgpuSwapchain while GPUDevice is lost; call recover() first');
      }
      if (config.format !== undefined) {
        if (typeof config.format !== 'string' || config.format.length === 0) {
          throw new TypeError('format must be a non-empty string');
        }
      }
      if (config.streamFormat !== undefined) {
        if (typeof config.streamFormat !== 'string' || config.streamFormat.length === 0) {
          throw new TypeError('streamFormat must be a non-empty string');
        }
      }
      if (config.depthFormat !== undefined) {
        validateDepthFormat(config.depthFormat);
      }
      if (config.alphaMode !== undefined) {
        validateAlphaMode(config.alphaMode);
      }
      const nextExplicitViewFormats =
        config.viewFormats !== undefined ? normalizeViewFormats(config.viewFormats) : undefined;
      const nextExplicitUsage =
        config.usage !== undefined ? normalizeUsage(config.usage) : undefined;
      const nextFrameIntervalUs =
        config.frameIntervalUs !== undefined
          ? normalizeFrameIntervalUs(config.frameIntervalUs)
          : undefined;
      const nextDroppedFrameThresholdMs =
        config.droppedFrameThresholdMs !== undefined
          ? normalizeDroppedFrameThresholdMs(config.droppedFrameThresholdMs)
          : undefined;
      const displayBoundsUpdate = extractDisplayBoundsOption(config, true);

      const wasConfigured = this._configured;
      const prevWidth = this._width;
      const prevHeight = this._height;
      const prevRequestedWidth = this._requestedWidth;
      const prevRequestedHeight = this._requestedHeight;
      const prevDisplayBounds = this._displayBounds;
      const prevFormat = this._format;
      const prevStreamFormat = this._streamFormat;
      const prevUsesBlit = this._usesBlitFallback;
      const prevDepthFormat = this._depthFormat;
      const prevEnableDepth = this._enableDepth;
      const prevExplicitUsage = this._explicitUsage;
      const prevUsage = this._usage;
      const prevCopySrcSupported = this._copySrcSupported;

      if (isReplacingDevice) {
        this._replaceDevice(config.device);
      }

      if (config.format !== undefined) {
        if (isValidCanvasFormat(config.format)) {
          if (this._format !== config.format) {
            this._blitPipeline = null;
            this._blitPipelineMeta = null;
            this._toneMapPipeline = null;
            this._toneMapPipelineMeta = null;
            this._toneMapUniformDirty = true;
            this._invalidateBlitBindGroup();
            if (nextExplicitViewFormats === undefined) {
              this._explicitViewFormats = null;
              this._viewFormats = null;
              this._explicitViewFormatsKey = '';
            }
          }
          if (this._usesBlitFallback) {
            this._usesBlitFallback = false;
            this._destroyBlitResources();
          }
          this._format = config.format;
          this._streamFormat = config.format;
        } else {
          if (!isValidCanvasFormat(this._format)) {
            this._format = this._preferredCanvasFormat;
          }
          this._streamFormat = config.format;
          this._usesBlitFallback = true;
        }
      }
      if (config.streamFormat !== undefined) {
        const forceHdrToneMapBlit = isHdrToneMapBlitRequested(
          config.streamFormat,
          this._format,
          this._toneMapMode,
          this._exposure,
          this._applySrgbEncodeOverride
        );
        if (isValidCanvasFormat(config.streamFormat) && !forceHdrToneMapBlit) {
          if (this._format !== config.streamFormat) {
            this._blitPipeline = null;
            this._blitPipelineMeta = null;
            this._toneMapPipeline = null;
            this._toneMapPipelineMeta = null;
            this._toneMapUniformDirty = true;
            this._invalidateBlitBindGroup();
            if (nextExplicitViewFormats === undefined) {
              this._explicitViewFormats = null;
              this._viewFormats = null;
              this._explicitViewFormatsKey = '';
            }
          }
          if (this._usesBlitFallback) {
            this._usesBlitFallback = false;
            this._destroyBlitResources();
          }
          this._format = config.streamFormat;
          this._streamFormat = config.streamFormat;
        } else {
          if (!isValidCanvasFormat(this._format)) {
            this._format = this._preferredCanvasFormat;
          }
          this._streamFormat = config.streamFormat;
          this._usesBlitFallback = true;
        }
      }
      if (nextExplicitViewFormats !== undefined) {
        this._explicitViewFormats = nextExplicitViewFormats;
        this._viewFormats = nextExplicitViewFormats;
        this._explicitViewFormatsKey = nextExplicitViewFormats
          ? nextExplicitViewFormats.join(',')
          : '';
      }
      if (nextExplicitUsage !== undefined) {
        this._explicitUsage = nextExplicitUsage;
        this._usage = nextExplicitUsage !== null ? nextExplicitUsage : DEFAULT_USAGE;
      }
      if (config.depthFormat !== undefined) {
        this._depthFormat = config.depthFormat;
      }
      if (config.alphaMode !== undefined) {
        this._alphaMode = config.alphaMode;
      }
      if (config.enableDepth !== undefined) {
        this._enableDepth = Boolean(config.enableDepth);
      }
      if (nextFrameIntervalUs !== undefined) {
        this._frameIntervalUs = nextFrameIntervalUs;
      }
      if (nextDroppedFrameThresholdMs !== undefined) {
        this._droppedFrameThresholdMs = nextDroppedFrameThresholdMs;
      }
      if (displayBoundsUpdate.provided) {
        this._displayBounds = displayBoundsUpdate.bounds;
      }

      const dpr = config.devicePixelRatio !== undefined ? config.devicePixelRatio : 1;
      const maxDim = this._effectiveMaxDimension();
      const nextRequestedWidth =
        config.width === undefined && wasConfigured && prevRequestedWidth > 0
          ? Math.min(prevRequestedWidth, maxDim)
          : config.width === undefined && wasConfigured && prevWidth > 0
            ? Math.min(prevWidth, maxDim)
            : clampDimension(config.width, dpr, maxDim);
      const nextRequestedHeight =
        config.height === undefined && wasConfigured && prevRequestedHeight > 0
          ? Math.min(prevRequestedHeight, maxDim)
          : config.height === undefined && wasConfigured && prevHeight > 0
            ? Math.min(prevHeight, maxDim)
            : clampDimension(config.height, dpr, maxDim);
      const clampedDims = clampToDisplayBounds(
        nextRequestedWidth,
        nextRequestedHeight,
        this._displayBounds,
        maxDim
      );
      const nextWidth = clampedDims.width;
      const nextHeight = clampedDims.height;

      const dimensionsChanged =
        wasConfigured && (nextWidth !== prevWidth || nextHeight !== prevHeight);

      this._requestedWidth = nextRequestedWidth;
      this._requestedHeight = nextRequestedHeight;
      this._width = nextWidth;
      this._height = nextHeight;
      this._devicePixelRatio =
        typeof dpr === 'number' && Number.isFinite(dpr) && dpr > 0 ? dpr : 1;

      const nextEffectiveVfKey = this._resolveEffectiveViewFormatsKey();
      const nextEffectiveUsage = this._resolveEffectiveUsage();
      const sameCanvasConfig = this._isSameAppliedConfig(
        nextWidth,
        nextHeight,
        this._format,
        nextEffectiveUsage,
        this._alphaMode,
        nextEffectiveVfKey
      );
      const depthChanged =
        wasConfigured &&
        (this._depthFormat !== prevDepthFormat || this._enableDepth !== prevEnableDepth);
      const streamChanged =
        wasConfigured &&
        (this._streamFormat !== prevStreamFormat ||
          this._usesBlitFallback !== prevUsesBlit ||
          this._format !== prevFormat);

      if (sameCanvasConfig) {
        if (depthChanged) {
          this._ensureDepthTexture();
        }
        if (streamChanged) {
          if (this._usesBlitFallback) {
            this._ensureIntermediateTexture();
          } else {
            this._destroyBlitResources();
          }
        }
        if (!depthChanged && !streamChanged) {
          this._skippedRedundantReconfigures += 1;
        }
      } else {
        try {
          this._applyContextConfiguration();
        } catch (err) {
          this._explicitUsage = prevExplicitUsage;
          this._usage = prevUsage;
          this._copySrcSupported = prevCopySrcSupported;
          this._displayBounds = prevDisplayBounds;
          this._requestedWidth = prevRequestedWidth;
          this._requestedHeight = prevRequestedHeight;
          this._width = prevWidth;
          this._height = prevHeight;
          throw err;
        }
        if (clampedDims.clamped) {
          this._clampedToDisplayCount += 1;
        }
        if (dimensionsChanged) {
          this._resizeCount += 1;
        }
      }

      return {
        width: this._width,
        height: this._height,
        format: this._format,
        depthFormat: this._enableDepth ? this._depthFormat : null,
        devicePixelRatio: this._devicePixelRatio,
      };
    }

    /**
     * Dynamically resize the swapchain. Safely rejects invalid/zero/negative dimensions and
     * no-ops (returning false) when clamped pixel dimensions match the currently configured
     * dimensions.
     *
     * @param {number} cssWidth
     * @param {number} cssHeight
     * @param {number} [devicePixelRatio=1]
     * @param {object} [displayOptions]
     * @returns {boolean} True if swapchain was reconfigured, false on no-op or invalid input.
     */
    resize(cssWidth, cssHeight, devicePixelRatio = 1, displayOptions) {
      if (this._destroyed) {
        throw new Error('Cannot resize a destroyed WgpuSwapchain');
      }
      if (this._isDeviceLost) {
        throw new Error('Cannot resize WgpuSwapchain while GPUDevice is lost; call recover() first');
      }

      if (
        typeof cssWidth !== 'number' ||
        typeof cssHeight !== 'number' ||
        Number.isNaN(cssWidth) ||
        Number.isNaN(cssHeight) ||
        cssWidth <= 0 ||
        cssHeight <= 0
      ) {
        return false;
      }

      const displayBoundsUpdate =
        displayOptions !== undefined
          ? extractDisplayBoundsOption(displayOptions, true)
          : { provided: false, bounds: null };
      const effectiveDisplayBounds = displayBoundsUpdate.provided
        ? displayBoundsUpdate.bounds
        : this._displayBounds;

      const effectiveDpr =
        typeof devicePixelRatio === 'number' &&
        !Number.isNaN(devicePixelRatio) &&
        devicePixelRatio > 0
          ? devicePixelRatio
          : 1;
      const maxDim = this._effectiveMaxDimension();
      const requestedWidth = clampDimension(cssWidth, effectiveDpr, maxDim);
      const requestedHeight = clampDimension(cssHeight, effectiveDpr, maxDim);
      const clamped = clampToDisplayBounds(
        requestedWidth,
        requestedHeight,
        effectiveDisplayBounds,
        maxDim
      );
      const nextWidth = clamped.width;
      const nextHeight = clamped.height;

      if (
        this._configured &&
        this._hasAppliedConfig &&
        this._appliedConfig.device === this._device &&
        this._usage === this._resolveEffectiveUsage() &&
        nextWidth === this._width &&
        nextHeight === this._height
      ) {
        if (displayBoundsUpdate.provided) {
          this._displayBounds = effectiveDisplayBounds;
        }
        this._requestedWidth = requestedWidth;
        this._requestedHeight = requestedHeight;
        this._skippedRedundantReconfigures += 1;
        return false;
      }

      const nextConfig = {
        width: cssWidth,
        height: cssHeight,
        devicePixelRatio: effectiveDpr,
      };
      if (displayBoundsUpdate.provided) {
        nextConfig.displayBounds = effectiveDisplayBounds;
      }
      this.configure(nextConfig);
      return true;
    }

    /**
     * Update (or clear with `null`) the physical display pixel ceiling used to clamp
     * oversized swapchain requests proportionally to native monitor resolution.
     *
     * Supports:
     * - `setDisplayBounds(screenWidth, screenHeight, dpr = 1)`
     * - `setDisplayBounds({ screenWidth, screenHeight, devicePixelRatio })`
     * - `setDisplayBounds({ maxDisplayWidth, maxDisplayHeight })`
     * - `setDisplayBounds(null)` to disable display-native clamping
     *
     * When already configured, re-clamps the last requested backing dimensions against the
     * new display bounds and reconfigures only if the effective backing dimensions change.
     *
     * @param {number | object | null | undefined} screenWidthOrBounds
     * @param {number} [screenHeight]
     * @param {number} [dpr=1]
     * @returns {boolean} True if active swapchain backing dimensions changed and reconfigured.
     */
    setDisplayBounds(screenWidthOrBounds, screenHeight, dpr = 1) {
      if (this._destroyed) {
        throw new Error('Cannot set display bounds on a destroyed WgpuSwapchain');
      }

      let nextBounds;
      if (screenWidthOrBounds === null || screenWidthOrBounds === undefined) {
        nextBounds = null;
      } else if (typeof screenWidthOrBounds === 'object') {
        nextBounds = normalizeDisplayBounds(screenWidthOrBounds);
      } else {
        nextBounds = normalizeDisplayBounds({
          screenWidth: screenWidthOrBounds,
          screenHeight,
          devicePixelRatio: dpr,
        });
      }

      this._displayBounds = nextBounds;

      if (
        !this._configured ||
        this._isDeviceLost ||
        this._requestedWidth < 1 ||
        this._requestedHeight < 1
      ) {
        return false;
      }

      const maxDim = this._effectiveMaxDimension();
      const reqW = Math.min(this._requestedWidth, maxDim);
      const reqH = Math.min(this._requestedHeight, maxDim);
      const clamped = clampToDisplayBounds(reqW, reqH, this._displayBounds, maxDim);

      if (clamped.width === this._width && clamped.height === this._height) {
        this._skippedRedundantReconfigures += 1;
        return false;
      }

      this._width = clamped.width;
      this._height = clamped.height;
      this._applyContextConfiguration();
      this._resizeCount += 1;
      if (clamped.clamped) {
        this._clampedToDisplayCount += 1;
      }
      return true;
    }

    /**
     * Return the active render-target color GPUTextureView for the current frame.
     *
     * When blit fallback is inactive (valid direct canvas format), this acquires and returns
     * the canvas context texture view directly with zero intermediate allocation overhead.
     * When blit fallback is active, this lazily ensures and returns the offscreen intermediate
     * texture view in `streamFormat`.
     *
     * @param {GPUTextureViewDescriptor} [viewDescriptor]
     * @returns {GPUTextureView}
     */
    getRenderTargetView(viewDescriptor) {
      if (this._destroyed) {
        throw new Error('Cannot get render target view from a destroyed WgpuSwapchain');
      }
      if (this._isDeviceLost) {
        throw new Error('Cannot get render target view while GPUDevice is lost');
      }
      if (!this._configured) {
        throw new Error('WgpuSwapchain must be configured before calling getRenderTargetView()');
      }

      if (this._usesBlitFallback) {
        this._ensureIntermediateTexture();
        return this._intermediateTextureView;
      }

      const currentTexture = this._context.getCurrentTexture();
      return viewDescriptor !== undefined
        ? currentTexture.createView(viewDescriptor)
        : currentTexture.createView();
    }

    _recordBlitPass(commandEncoder, sourceView, canvasTextureView, sourceFormat) {
      const targetView =
        canvasTextureView !== undefined && canvasTextureView !== null
          ? canvasTextureView
          : this._context.getCurrentTexture().createView();

      const useToneMap = this._shouldUseToneMapPipeline(sourceFormat);
      let pipeline;
      let bindGroup;

      if (useToneMap) {
        pipeline = this._ensureToneMapPipeline();
        const uniformBuffer = this._ensureToneMapUniformBuffer(sourceFormat);
        bindGroup = this._ensureToneMapBindGroup(sourceView, this._blitSampler, uniformBuffer);
      } else {
        const applySrgb = this._resolveEffectiveApplySrgbEncode(sourceFormat);
        pipeline = this._ensureBlitPipeline(applySrgb);
        bindGroup = this._ensureBlitBindGroup(sourceView, this._blitSampler);
      }

      this._blitColorAttachment.view = targetView;
      const pass = commandEncoder.beginRenderPass(this._blitPassDescriptor);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();

      this._blitCount += 1;
      if (useToneMap) {
        this._toneMapBlitCount += 1;
      }
      return true;
    }

    /**
     * Record a fullscreen-triangle blit (or HDR tone-map blit when tone-mapping/exposure/sRGB
     * encoding is active) from `sourceView` onto the WebGPU canvas texture view with zero
     * per-frame JS allocations.
     *
     * @param {GPUCommandEncoder} commandEncoder
     * @param {GPUTextureView} [sourceView] Defaults to the swapchain intermediate view when active.
     * @param {GPUTextureView} [canvasTextureView] Optional pre-acquired canvas view.
     * @param {string} [sourceFormat] Optional source format hint for automatic HDR sRGB encoding.
     * @returns {boolean} True if a blit pass was recorded.
     */
    blitToCanvas(commandEncoder, sourceView, canvasTextureView, sourceFormat) {
      if (this._destroyed || this._isDeviceLost || !this._configured || !this._device) {
        return false;
      }
      if (!commandEncoder || typeof commandEncoder.beginRenderPass !== 'function') {
        throw new TypeError('blitToCanvas requires a GPUCommandEncoder');
      }

      let resolvedSourceView = sourceView;
      if (resolvedSourceView === undefined || resolvedSourceView === null) {
        if (!this._usesBlitFallback) {
          return false;
        }
        this._ensureIntermediateTexture();
        resolvedSourceView = this._intermediateTextureView;
      }
      if (!resolvedSourceView) {
        return false;
      }

      const resolvedFormat =
        typeof sourceFormat === 'string' && sourceFormat.length > 0
          ? sourceFormat
          : this._streamFormat;

      return this._recordBlitPass(
        commandEncoder,
        resolvedSourceView,
        canvasTextureView,
        resolvedFormat
      );
    }

    /**
     * Record a fullscreen-triangle blit pass copying the offscreen intermediate render-target
     * texture onto the WebGPU canvas texture view when blit fallback is active.
     *
     * Strict no-op (returning false) when rendering directly to a valid WebGPU canvas format.
     *
     * @param {GPUCommandEncoder} commandEncoder
     * @param {GPUTextureView} [canvasTextureView] Optional pre-acquired canvas view; acquired from
     *   `context.getCurrentTexture().createView()` when omitted.
     * @returns {boolean} True if a blit pass was recorded, false if no-op.
     */
    blitToCanvasIfNeeded(commandEncoder, canvasTextureView) {
      if (!this._usesBlitFallback) {
        return false;
      }
      if (this._destroyed || this._isDeviceLost || !this._configured || !this._device) {
        return false;
      }
      if (!commandEncoder || typeof commandEncoder.beginRenderPass !== 'function') {
        throw new TypeError('blitToCanvasIfNeeded requires a GPUCommandEncoder');
      }

      this._ensureIntermediateTexture();
      if (!this._intermediateTextureView) {
        return false;
      }

      return this._recordBlitPass(
        commandEncoder,
        this._intermediateTextureView,
        canvasTextureView,
        this._streamFormat
      );
    }

    /**
     * Finalize frame presentation on the given command encoder by recording the fullscreen
     * intermediate-to-canvas blit pass if blit fallback is active, and update visible-content
     * readiness / present telemetry when invoked with an encoder or frame metadata.
     *
     * Supported signatures:
     * - `present(commandEncoder, canvasTextureView?, meta?)`
     * - `present(commandEncoder, meta)`
     * - `present(meta)` (metadata-only present without an encoder)
     *
     * @param {GPUCommandEncoder | object} [commandEncoder]
     * @param {GPUTextureView | object} [canvasTextureView]
     * @param {{
     *   hasVisibleContent?: boolean,
     *   hasDraws?: boolean | number,
     *   realPackDraws?: number | boolean,
     *   nonClearDraw?: boolean | number,
     *   timestampMs?: number,
     *   frameIntervalUs?: number | null,
     *   dropped?: boolean,
     * }} [meta]
     * @returns {boolean} True if a blit pass was recorded.
     */
    present(commandEncoder, canvasTextureView, meta) {
      if (!commandEncoder || this._destroyed || this._isDeviceLost) {
        return false;
      }

      // Allow calling present({ hasVisibleContent, realPackDraws, ... }) without an encoder.
      if (typeof commandEncoder.beginRenderPass !== 'function') {
        if (isFrameContentMetadata(commandEncoder)) {
          this.recordFramePresented(commandEncoder);
        }
        return false;
      }

      let resolvedCanvasView = canvasTextureView;
      let resolvedMeta = meta;
      if (resolvedMeta === undefined && isFrameContentMetadata(canvasTextureView)) {
        resolvedMeta = canvasTextureView;
        resolvedCanvasView = undefined;
      }

      const blitted = this.blitToCanvasIfNeeded(commandEncoder, resolvedCanvasView);
      this.recordFramePresented(resolvedMeta);
      return blitted;
    }

    /**
     * Acquire the current swapchain color texture view (and matching depth-stencil view when
     * enabled) packaged as a GPURenderPassDescriptor ready for encoder.beginRenderPass().
     *
     * @param {{
     *   clearColor?: { r: number, g: number, b: number, a: number } | number[],
     *   loadOp?: 'clear' | 'load',
     *   storeOp?: 'store' | 'discard',
     *   depthClearValue?: number,
     *   depthLoadOp?: 'clear' | 'load',
     *   depthStoreOp?: 'store' | 'discard',
     *   stencilClearValue?: number,
     *   stencilLoadOp?: 'clear' | 'load',
     *   stencilStoreOp?: 'store' | 'discard',
     *   label?: string,
     * }} [options]
     * @returns {GPURenderPassDescriptor}
     */
    acquireRenderPassDescriptor(options = {}) {
      if (this._destroyed) {
        throw new Error('Cannot acquire render pass descriptor from a destroyed WgpuSwapchain');
      }
      if (this._isDeviceLost) {
        throw new Error('Cannot acquire render pass descriptor while GPUDevice is lost');
      }
      if (!this._configured) {
        throw new Error('WgpuSwapchain must be configured before calling acquireRenderPassDescriptor()');
      }

      const {
        clearColor,
        loadOp = 'clear',
        storeOp = 'store',
        depthClearValue = 1.0,
        depthLoadOp = 'clear',
        depthStoreOp = 'store',
        stencilClearValue = 0,
        stencilLoadOp = 'clear',
        stencilStoreOp = 'store',
        label,
      } = options || {};

      const colorView = this.getRenderTargetView();

      const colorAttachment = {
        view: colorView,
        loadOp,
        storeOp,
      };
      if (loadOp === 'clear') {
        colorAttachment.clearValue = normalizeClearColor(clearColor);
      }

      const descriptor = {
        colorAttachments: [colorAttachment],
      };

      if (this._enableDepth) {
        this._ensureDepthTexture();
        const depthStencilAttachment = {
          view: this._depthTextureView,
          depthLoadOp,
          depthStoreOp,
        };
        if (depthLoadOp === 'clear') {
          depthStencilAttachment.depthClearValue = depthClearValue;
        }
        if (isStencilFormat(this._depthFormat)) {
          depthStencilAttachment.stencilLoadOp = stencilLoadOp;
          depthStencilAttachment.stencilStoreOp = stencilStoreOp;
          if (stencilLoadOp === 'clear') {
            depthStencilAttachment.stencilClearValue = stencilClearValue;
          }
        }
        descriptor.depthStencilAttachment = depthStencilAttachment;
      }

      if (label !== undefined) {
        descriptor.label = label;
      }

      return descriptor;
    }

    /**
     * Set or clear the declared frame interval target in microseconds (e.g. 16667 for 60Hz,
     * 33333 for 30Hz PresentX cadence).
     *
     * @param {number | null | undefined} frameIntervalUs
     */
    setFrameIntervalUs(frameIntervalUs) {
      this._frameIntervalUs = normalizeFrameIntervalUs(frameIntervalUs);
    }

    _getDroppedFrameThresholdMs() {
      if (this._droppedFrameThresholdMs !== null) {
        return this._droppedFrameThresholdMs;
      }
      if (this._frameIntervalUs !== null && this._frameIntervalUs > 0) {
        return (this._frameIntervalUs / 1000) * DROPPED_FRAME_CADENCE_MULTIPLIER;
      }
      return DEFAULT_DROPPED_FRAME_THRESHOLD_MS;
    }

    _createTelemetryObject() {
      return this._fillTelemetry({
        fps: 0,
        avgFrameTimeMs: 0,
        maxFrameTimeMs: 0,
        totalFrames: 0,
        presentCount: 0,
        droppedFrameCount: 0,
        lastFrameMs: 0,
        avgFrameMs: 0,
        maxFrameMs: 0,
        resizeCount: 0,
        reconfigureCount: 0,
        skippedRedundantReconfigures: 0,
        lastPresentIntervalMs: 0,
        avgPresentIntervalMs: 0,
        width: 0,
        height: 0,
        format: this._format,
        depthFormat: this._enableDepth ? this._depthFormat : null,
        isDeviceLost: false,
        hasVisibleContent: false,
        firstVisiblePresentIndex: null,
        firstVisiblePresentMs: null,
        consecutiveBlackPresents: 0,
        blackPresentsTotal: 0,
        visiblePresentsTotal: 0,
      });
    }

    _fillTelemetry(target) {
      target.fps = this._fps;
      target.avgFrameTimeMs = this._avgFrameMs;
      target.maxFrameTimeMs = this._maxFrameMs;
      target.totalFrames = this._totalFrames;
      target.presentCount = this._presentCount;
      target.droppedFrameCount = this._droppedFrameCount;
      target.lastFrameMs = this._lastFrameMs;
      target.avgFrameMs = this._avgFrameMs;
      target.maxFrameMs = this._maxFrameMs;
      target.resizeCount = this._resizeCount;
      target.reconfigureCount = this._reconfigureCount;
      target.skippedRedundantReconfigures = this._skippedRedundantReconfigures;
      target.lastPresentIntervalMs = this._lastPresentIntervalMs;
      target.avgPresentIntervalMs = this._avgFrameMs;
      target.width = this._width;
      target.height = this._height;
      target.format = this._format;
      target.depthFormat = this._enableDepth ? this._depthFormat : null;
      target.isDeviceLost = this._isDeviceLost;
      target.hasVisibleContent = this._hasVisibleContent;
      target.firstVisiblePresentIndex = this._firstVisiblePresentIndex;
      target.firstVisiblePresentMs = this._firstVisiblePresentMs;
      target.consecutiveBlackPresents = this._consecutiveBlackPresents;
      target.blackPresentsTotal = this._blackPresentsTotal;
      target.visiblePresentsTotal = this._visiblePresentsTotal;
      if (this._frameIntervalUs !== null) {
        target.frameIntervalUs = this._frameIntervalUs;
      } else if ('frameIntervalUs' in target) {
        delete target.frameIntervalUs;
      }
      return target;
    }

    /**
     * Mark whether the pending frame being recorded contains visible (non-black / non-clear)
     * draw content without performing any GPU readback. May be called multiple times per frame;
     * positive draw signals latch until consumed by the next `recordFramePresented()` /
     * `recordPresent()` / `present()` call, unless explicitly overridden by a boolean
     * `hasVisibleContent` flag.
     *
     * @param {{
     *   hasDraws?: boolean | number,
     *   realPackDraws?: number | boolean,
     *   nonClearDraw?: boolean | number,
     *   hasVisibleContent?: boolean,
     *   timestampMs?: number,
     * } | boolean} [content]
     * @returns {boolean} Whether the current pending frame is marked as having visible content.
     */
    markFrameContent(content) {
      if (this._destroyed) {
        return false;
      }
      if (typeof content === 'boolean') {
        this._hasPendingFrameContent = true;
        this._pendingFrameVisible = content;
        return this._pendingFrameVisible;
      }
      if (!content || typeof content !== 'object') {
        return this._pendingFrameVisible;
      }

      this._hasPendingFrameContent = true;

      const realPackCount = sanitizeNonNegativeDrawCount(content.realPackDraws);
      if (realPackCount > 0) {
        this._pendingRealPackDraws += realPackCount;
      }

      if (typeof content.hasVisibleContent === 'boolean') {
        this._pendingFrameVisible = content.hasVisibleContent;
      } else if (
        realPackCount > 0 ||
        isPositiveContentFlag(content.hasDraws) ||
        isPositiveContentFlag(content.nonClearDraw)
      ) {
        this._pendingFrameVisible = true;
      }

      if (typeof content.timestampMs === 'number' && Number.isFinite(content.timestampMs)) {
        this._pendingContentTimestampMs = content.timestampMs;
      }

      return this._pendingFrameVisible;
    }

    /**
     * Confirm that an already-presented frame contained verified non-black visible pixels
     * (e.g. after an async GPU swapchain luminance readback completes) without double-counting
     * `presentCount` / `totalFrames`. Latches `hasVisibleContent`, `firstVisiblePresentIndex`,
     * `firstVisiblePresentMs`, and fires `onFirstVisibleFrame` once.
     *
     * @param {{ timestampMs?: number, realPackDraws?: number | boolean }} [meta]
     * @returns {boolean} True if this call transitioned the swapchain into its first visible frame.
     */
    confirmVisibleFrame(meta = {}) {
      if (this._destroyed) {
        return false;
      }
      const opts = meta && typeof meta === 'object' ? meta : {};
      const nowMs =
        typeof opts.timestampMs === 'number' && Number.isFinite(opts.timestampMs)
          ? opts.timestampMs
          : this._now();
      const realPackDraws = sanitizeNonNegativeDrawCount(opts.realPackDraws);
      const wasFirstVisible = !this._firstVisibleFired;

      this._hasVisibleContent = true;
      this._consecutiveBlackPresents = 0;
      if (this._visiblePresentsTotal === 0) {
        this._visiblePresentsTotal = 1;
        if (this._blackPresentsTotal > 0) {
          this._blackPresentsTotal -= 1;
        }
      }

      if (!this._firstVisibleFired) {
        this._firstVisiblePresentIndex = Math.max(1, this._presentCount);
        this._firstVisiblePresentMs =
          typeof nowMs === 'number' && Number.isFinite(nowMs) ? nowMs : 0;
        this._firstVisibleFired = true;

        if (typeof this._onFirstVisibleFrame === 'function') {
          try {
            this._onFirstVisibleFrame({
              presentIndex: this._firstVisiblePresentIndex,
              firstVisiblePresentIndex: this._firstVisiblePresentIndex,
              timestampMs: this._firstVisiblePresentMs,
              firstVisiblePresentMs: this._firstVisiblePresentMs,
              blackPresentsTotal: this._blackPresentsTotal,
              consecutiveBlackPresents: 0,
              visiblePresentsTotal: this._visiblePresentsTotal,
              realPackDraws,
              hasVisibleContent: true,
              width: this._width,
              height: this._height,
            });
          } catch {
            // Isolate callback exceptions so readiness notifications never break rendering.
          }
        }
      }

      this._fillTelemetry(this._liveTelemetry);
      return wasFirstVisible;
    }

    /**
     * Record one or more dropped frames (e.g. skipped present or backpressure) with zero
     * heap allocations.
     *
     * @param {number} [count=1]
     * @returns {number} Cumulative droppedFrameCount.
     */
    recordFrameDropped(count = 1) {
      if (typeof count !== 'number' || !Number.isInteger(count) || count < 0) {
        throw new RangeError(`dropped frame count must be a non-negative integer, got ${count}`);
      }
      this._droppedFrameCount += count;
      return this._droppedFrameCount;
    }

    /**
     * Record a presented frame timestamp into the pre-allocated rolling frame-pacing ring buffer
     * with zero per-frame allocations, and update visible-content readiness counters.
     *
     * @param {number | {
     *   timestampMs?: number,
     *   frameIntervalUs?: number | null,
     *   dropped?: boolean,
     *   hasVisibleContent?: boolean,
     *   hasDraws?: boolean | number,
     *   realPackDraws?: number | boolean,
     *   nonClearDraw?: boolean | number,
     * }} [timestampOrOptions]
     *   Optional explicit timestamp override in milliseconds (defaults to `nowFn()`/`now()`) or
     *   metadata object.
     * @param {number | {
     *   frameIntervalUs?: number | null,
     *   dropped?: boolean,
     *   hasVisibleContent?: boolean,
     *   hasDraws?: boolean | number,
     *   realPackDraws?: number | boolean,
     *   nonClearDraw?: boolean | number,
     * }} [meta]
     *   Optional present cadence and visible-content metadata, or numeric `frameIntervalUs`.
     */
    recordFramePresented(timestampOrOptions, meta) {
      let explicitTimestampMs;
      let providedFrameIntervalUs;
      let explicitDropped = false;
      let hasExplicitVisible = false;
      let explicitVisible = false;
      let inlinePositiveContent = false;
      let inlineRealPackDraws = 0;

      if (timestampOrOptions && typeof timestampOrOptions === 'object') {
        explicitTimestampMs = timestampOrOptions.timestampMs;
        if ('frameIntervalUs' in timestampOrOptions) {
          providedFrameIntervalUs = timestampOrOptions.frameIntervalUs;
        }
        if (timestampOrOptions.dropped === true) {
          explicitDropped = true;
        }
        if (typeof timestampOrOptions.hasVisibleContent === 'boolean') {
          hasExplicitVisible = true;
          explicitVisible = timestampOrOptions.hasVisibleContent;
        }
        const optRealPack = sanitizeNonNegativeDrawCount(timestampOrOptions.realPackDraws);
        if (optRealPack > 0) {
          inlineRealPackDraws += optRealPack;
          inlinePositiveContent = true;
        }
        if (
          isPositiveContentFlag(timestampOrOptions.hasDraws) ||
          isPositiveContentFlag(timestampOrOptions.nonClearDraw)
        ) {
          inlinePositiveContent = true;
        }
      } else {
        explicitTimestampMs = timestampOrOptions;
      }

      if (meta !== undefined) {
        if (meta && typeof meta === 'object') {
          if ('frameIntervalUs' in meta) {
            providedFrameIntervalUs = meta.frameIntervalUs;
          }
          if (meta.dropped === true) {
            explicitDropped = true;
          }
          if (typeof meta.hasVisibleContent === 'boolean') {
            hasExplicitVisible = true;
            explicitVisible = meta.hasVisibleContent;
          }
          const metaRealPack = sanitizeNonNegativeDrawCount(meta.realPackDraws);
          if (metaRealPack > 0) {
            inlineRealPackDraws += metaRealPack;
            inlinePositiveContent = true;
          }
          if (
            isPositiveContentFlag(meta.hasDraws) ||
            isPositiveContentFlag(meta.nonClearDraw)
          ) {
            inlinePositiveContent = true;
          }
        } else if (typeof meta === 'number' || meta === null) {
          providedFrameIntervalUs = meta;
        }
      }

      if (providedFrameIntervalUs !== undefined) {
        this._frameIntervalUs = normalizeFrameIntervalUs(providedFrameIntervalUs);
      }

      const nowMs =
        typeof explicitTimestampMs === 'number' && Number.isFinite(explicitTimestampMs)
          ? explicitTimestampMs
          : this._now();

      // Consume pending frame-content flags marked prior to present so the next frame starts clean.
      let frameVisible = this._pendingFrameVisible;
      const frameRealPackDraws = this._pendingRealPackDraws + inlineRealPackDraws;
      const pendingContentTimestampMs = this._pendingContentTimestampMs;
      this._pendingFrameVisible = false;
      this._pendingRealPackDraws = 0;
      this._pendingContentTimestampMs = null;
      this._hasPendingFrameContent = false;

      if (hasExplicitVisible) {
        frameVisible = explicitVisible;
      } else if (inlinePositiveContent) {
        frameVisible = true;
      }

      this._totalFrames += 1;
      this._presentCount += 1;

      if (this._lastFrameTimestamp !== null) {
        const deltaMs = nowMs > this._lastFrameTimestamp ? nowMs - this._lastFrameTimestamp : 0;
        this._lastPresentIntervalMs = deltaMs;
        this._lastFrameMs = deltaMs;

        const ring = this._frameDeltaRing;
        ring[this._frameDeltaHead] = deltaMs;
        const nextHead = this._frameDeltaHead + 1;
        this._frameDeltaHead = nextHead === this._frameWindowSize ? 0 : nextHead;
        if (this._frameDeltaCount < this._frameWindowSize) {
          this._frameDeltaCount += 1;
        }

        const count = this._frameDeltaCount;
        let sum = 0;
        let max = 0;
        let compliant30Count = 0;
        for (let i = 0; i < count; i += 1) {
          const dt = ring[i];
          sum += dt;
          if (dt > max) {
            max = dt;
          }
          if (dt <= TARGET_30FPS_BUDGET_MS) {
            compliant30Count += 1;
          }
        }
        const avg = sum / count;
        let sumSqDiff = 0;
        for (let i = 0; i < count; i += 1) {
          const diff = ring[i] - avg;
          sumSqDiff += diff * diff;
        }
        const rawCompliance = compliant30Count / count;
        this._avgFrameMs = avg;
        this._maxFrameMs = max;
        this._fps = avg > 0 ? 1000 / avg : 0;
        this._fps30ComplianceRate =
          rawCompliance < 0 ? 0 : rawCompliance > 1 ? 1 : rawCompliance;
        this._frameJitterMs = Math.sqrt(sumSqDiff / count);
        if (deltaMs > TARGET_30FPS_BUDGET_MS && deltaMs > this._longestStallMs) {
          this._longestStallMs = deltaMs;
        }

        const thresholdMs = this._getDroppedFrameThresholdMs();
        if (explicitDropped || (thresholdMs > 0 && deltaMs > thresholdMs)) {
          this._droppedFrameCount += 1;
        }
      } else if (explicitDropped) {
        this._droppedFrameCount += 1;
      }

      this._lastFrameTimestamp = nowMs;

      if (frameVisible) {
        this._visiblePresentsTotal += 1;
        this._consecutiveBlackPresents = 0;
        this._hasVisibleContent = true;

        if (!this._firstVisibleFired) {
          const visibleTs =
            typeof explicitTimestampMs === 'number' && Number.isFinite(explicitTimestampMs)
              ? explicitTimestampMs
              : pendingContentTimestampMs !== null
                ? pendingContentTimestampMs
                : nowMs;
          this._firstVisiblePresentIndex = this._presentCount;
          this._firstVisiblePresentMs =
            typeof visibleTs === 'number' && Number.isFinite(visibleTs) ? visibleTs : 0;
          this._firstVisibleFired = true;

          if (typeof this._onFirstVisibleFrame === 'function') {
            try {
              this._onFirstVisibleFrame({
                presentIndex: this._firstVisiblePresentIndex,
                firstVisiblePresentIndex: this._firstVisiblePresentIndex,
                timestampMs: this._firstVisiblePresentMs,
                firstVisiblePresentMs: this._firstVisiblePresentMs,
                blackPresentsTotal: this._blackPresentsTotal,
                consecutiveBlackPresents: 0,
                visiblePresentsTotal: this._visiblePresentsTotal,
                realPackDraws: frameRealPackDraws,
                hasVisibleContent: true,
                width: this._width,
                height: this._height,
              });
            } catch {
              // Isolate callback exceptions so readiness notifications never break rendering.
            }
          }
        }
      } else {
        this._blackPresentsTotal += 1;
        this._consecutiveBlackPresents += 1;
      }

      return this._fillTelemetry(this._liveTelemetry);
    }

    /**
     * Alias for `recordFramePresented()`, accepting optional timestamp and visible-content
     * metadata (`hasVisibleContent`, `hasDraws`, `realPackDraws`, `nonClearDraw`).
     *
     * @param {number | object} [timestampOrOptions]
     * @param {number | object} [meta]
     */
    recordPresent(timestampOrOptions, meta) {
      return this.recordFramePresented(timestampOrOptions, meta);
    }

    /**
     * Return rolling frame-pacing and swapchain state telemetry.
     * Pass an optional reusable `out` object for zero-allocation reads.
     *
     * @param {object} [out]
     */
    getTelemetry(out) {
      if (out && typeof out === 'object') {
        return this._fillTelemetry(out);
      }
      return this._fillTelemetry({});
    }

    /**
     * Return compact frame-pacing telemetry (`presentCount`, `reconfigureCount`,
     * `droppedFrameCount`, `lastFrameMs`, `avgFrameMs`, `maxFrameMs`).
     * Pass an optional reusable `out` object for zero-allocation reads.
     *
     * @param {object} [out]
     */
    getFramePacingStats(out) {
      const target = out && typeof out === 'object' ? out : {};
      target.presentCount = this._presentCount;
      target.reconfigureCount = this._reconfigureCount;
      target.droppedFrameCount = this._droppedFrameCount;
      target.lastFrameMs = this._lastFrameMs;
      target.avgFrameMs = this._avgFrameMs;
      target.maxFrameMs = this._maxFrameMs;
      return target;
    }

    /**
     * Return swapchain telemetry plus WebGPU validation-error and blit-fallback counters.
     */
    getStats() {
      return {
        ...this.getTelemetry(),
        usage: this._usage,
        validationErrorCount: this._validationErrorCount,
        validationErrorHistorySize: this._validationErrors.length,
        maxErrorHistory: this._maxErrorHistory,
        isDestroyed: this._destroyed,
        usesBlitFallback: this._usesBlitFallback,
        streamFormat: this._streamFormat,
        canvasFormat: this._format,
        viewFormats: this._activeViewFormats.slice(),
        viewFormatsFallbackCount: this._viewFormatsFallbackCount,
        blitCount: this._blitCount,
        toneMapMode: this._toneMapMode,
        exposure: this._exposure,
        applySrgbEncode: this._resolveEffectiveApplySrgbEncode(this._streamFormat),
        isHdrStream: isHdrStreamFormat(this._streamFormat),
        toneMapActive: this._shouldUseToneMapPipeline(this._streamFormat),
        toneMapBlitCount: this._toneMapBlitCount,
        toneMapUniformUploads: this._toneMapUniformUploads,
        fps30ComplianceRate: this._fps30ComplianceRate,
        frameJitterMs: this._frameJitterMs,
        longestStallMs: this._longestStallMs,
        clampedToDisplayCount: this._clampedToDisplayCount,
        displayBounds: this._displayBounds ? { ...this._displayBounds } : null,
        requestedWidth: this._requestedWidth,
        requestedHeight: this._requestedHeight,
        backingWidth: this._width,
        backingHeight: this._height,
      };
    }

    /**
     * Return shallow copies of captured WebGPU uncaptured validation/OOM error records
     * retained in the bounded history ring.
     *
     * @returns {Array<{ seq: number, type: string, message: string, timestampMs: number }>}
     */
    getValidationErrors() {
      return this._validationErrors.map((entry) => ({ ...entry }));
    }

    /**
     * Clear the bounded WebGPU validation error history ring.
     */
    clearValidationErrors() {
      this._validationErrors.length = 0;
    }

    /**
     * Reset frame-pacing, visible-content readiness, blit, and validation error statistics
     * cleanly for a new session while preserving configured GPU resources and re-arming the
     * one-shot `onFirstVisibleFrame` callback.
     */
    resetStats() {
      this._totalFrames = 0;
      this._presentCount = 0;
      this._droppedFrameCount = 0;
      this._resizeCount = 0;
      this._reconfigureCount = 0;
      this._skippedRedundantReconfigures = 0;
      this._lastFrameTimestamp = null;
      this._lastPresentIntervalMs = 0;
      this._lastFrameMs = 0;
      this._avgFrameMs = 0;
      this._maxFrameMs = 0;
      this._fps = 0;
      this._fps30ComplianceRate = 0;
      this._frameJitterMs = 0;
      this._longestStallMs = 0;
      this._frameDeltaRing.fill(0);
      this._frameDeltaHead = 0;
      this._frameDeltaCount = 0;

      this._blitCount = 0;
      this._toneMapBlitCount = 0;
      this._toneMapUniformUploads = 0;
      this._viewFormatsFallbackCount = 0;
      this._clampedToDisplayCount = 0;

      this._validationErrors.length = 0;
      this._validationErrorCount = 0;
      this._validationErrorSeq = 0;

      this._hasVisibleContent = false;
      this._firstVisiblePresentIndex = null;
      this._firstVisiblePresentMs = null;
      this._consecutiveBlackPresents = 0;
      this._blackPresentsTotal = 0;
      this._visiblePresentsTotal = 0;
      this._firstVisibleFired = false;
      this._pendingFrameVisible = false;
      this._pendingRealPackDraws = 0;
      this._pendingContentTimestampMs = null;
      this._hasPendingFrameContent = false;

      this._fillTelemetry(this._liveTelemetry);
    }

    /**
     * Alias for `resetStats()`. Resets session telemetry and re-arms `onFirstVisibleFrame`.
     */
    reset() {
      this.resetStats();
    }

    /**
     * Register a callback invoked when the active GPUDevice's `device.lost` promise resolves.
     * Returns an unsubscribe function.
     *
     * @param {(info: { reason?: string, message?: string }, swapchain: WgpuSwapchain) => void} onLost
     * @returns {() => void}
     */
    attachDeviceLostHandler(onLost) {
      if (this._destroyed) {
        throw new Error('Cannot attach device lost handler to a destroyed WgpuSwapchain');
      }
      if (typeof onLost !== 'function') {
        throw new TypeError('attachDeviceLostHandler requires a callback function');
      }

      this._deviceLostCallbacks.add(onLost);
      if (this._device && this._device !== this._hookedDevice) {
        this._hookDeviceLost(this._device);
      }

      return () => {
        this._deviceLostCallbacks.delete(onLost);
      };
    }

    /**
     * Recover from a lost GPUDevice by binding a new GPUDevice (and optional new canvas format),
     * destroying any stale depth texture, hooking the new device's `lost` promise, and
     * re-configuring the existing canvas/context bindings.
     *
     * @param {GPUDevice} newDevice
     * @param {string} [newFormat]
     */
    recover(newDevice, newFormat) {
      if (this._destroyed) {
        throw new Error('Cannot recover a destroyed WgpuSwapchain');
      }
      if (!newDevice || typeof newDevice !== 'object') {
        throw new TypeError('recover requires a new GPUDevice');
      }
      if (newFormat !== undefined && newFormat !== null) {
        if (typeof newFormat !== 'string' || newFormat.length === 0) {
          throw new TypeError('newFormat must be a non-empty string when provided');
        }
        if (isValidCanvasFormat(newFormat)) {
          if (this._format !== newFormat) {
            this._explicitViewFormats = null;
            this._viewFormats = null;
            this._explicitViewFormatsKey = '';
          }
          this._format = newFormat;
          this._preferredCanvasFormat = newFormat;
          if (!this._usesBlitFallback) {
            this._streamFormat = newFormat;
          }
        } else {
          this._streamFormat = newFormat;
          this._usesBlitFallback = true;
          if (!isValidCanvasFormat(this._format)) {
            this._format = this._preferredCanvasFormat;
          }
        }
      }

      this._replaceDevice(newDevice);

      if (this._configured && this._width > 0 && this._height > 0) {
        const maxDim = this._effectiveMaxDimension();
        const reqW = Math.min(this._requestedWidth > 0 ? this._requestedWidth : this._width, maxDim);
        const reqH = Math.min(this._requestedHeight > 0 ? this._requestedHeight : this._height, maxDim);
        const clamped = clampToDisplayBounds(reqW, reqH, this._displayBounds, maxDim);
        this._requestedWidth = reqW;
        this._requestedHeight = reqH;
        this._width = clamped.width;
        this._height = clamped.height;
        this._applyContextConfiguration();
      }

      return {
        width: this._width,
        height: this._height,
        format: this._format,
        depthFormat: this._enableDepth ? this._depthFormat : null,
        isDeviceLost: this._isDeviceLost,
      };
    }

    /**
     * Destroy owned GPU resources (depth texture), unconfigure the canvas context if supported,
     * and release references. Safe to call multiple times (idempotent).
     */
    destroy() {
      if (this._destroyed) {
        return;
      }

      this._destroyed = true;
      this._deviceLostGeneration += 1;
      this._unhookUncapturedError();
      this._destroyDepthTexture();
      this._destroyBlitResources();

      if (this._context && typeof this._context.unconfigure === 'function') {
        this._context.unconfigure();
      }

      this._deviceLostCallbacks.clear();
      this._onValidationError = null;
      this._onFirstVisibleFrame = null;
      this._pendingFrameVisible = false;
      this._pendingRealPackDraws = 0;
      this._pendingContentTimestampMs = null;
      this._hasPendingFrameContent = false;
      this._device = null;
      this._context = null;
      this._canvas = null;
      this._hookedDevice = null;
      this._hasAppliedConfig = false;
      this._appliedConfig.device = null;
      this._activeViewFormats = [];
      this._explicitViewFormats = null;
      this._viewFormats = null;
      this._explicitViewFormatsKey = '';
      this._configured = false;
    }
  }

  const exportedApi = {
    DEFAULT_DEPTH_FORMAT,
    SUPPORTED_DEPTH_FORMATS,
    VALID_CANVAS_FORMATS,
    COMPATIBLE_VIEW_FORMATS,
    DEFAULT_CLEAR_COLOR,
    DEFAULT_ALPHA_MODE,
    DEFAULT_MAX_DIMENSION,
    DEFAULT_FRAME_WINDOW,
    DEFAULT_MAX_ERROR_HISTORY,
    DEFAULT_DROPPED_FRAME_THRESHOLD_MS,
    TARGET_30FPS_BUDGET_MS,
    DROPPED_FRAME_CADENCE_MULTIPLIER,
    MAX_VALIDATION_ERROR_MESSAGE_LENGTH,
    TEXTURE_USAGE_COPY_SRC,
    TEXTURE_USAGE_TEXTURE_BINDING,
    TEXTURE_USAGE_RENDER_ATTACHMENT,
    BUFFER_USAGE_UNIFORM,
    BUFFER_USAGE_COPY_DST,
    DEFAULT_USAGE,
    INTERMEDIATE_TEXTURE_USAGE,
    HDR_STREAM_FORMATS,
    TONE_MAP_MODES,
    DEFAULT_TONE_MAP_MODE,
    DEFAULT_EXPOSURE,
    BLIT_FULLSCREEN_WGSL,
    BLIT_SRGB_ENCODE_WGSL,
    BLIT_TONEMAP_WGSL,
    isValidCanvasFormat,
    isSrgbFormat,
    isHdrStreamFormat,
    normalizeToneMapMode,
    normalizeExposure,
    normalizeDisplayBounds,
    clampToDisplayBounds,
    getCompatibleViewFormats,
    WgpuSwapchain,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6WgpuSwapchain = exportedApi;
  } else if (root) {
    root.PlayGta6WgpuSwapchain = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
