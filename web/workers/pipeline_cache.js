'use strict';

// WebGPU RenderPipeline, ComputePipeline, and BindGroupLayout cache manager.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6PipelineCache)
// and Node.js (module.exports). Interfaces cleanly with GPUDevice, GPUShaderModule,
// WGPK 64-bit shader hashes (web/workers/wgsl_pack_loader.js), and S6 reflection metadata.
(function (root) {
  const DEFAULT_MAX_RENDER_PIPELINES = 2048;
  const DEFAULT_MAX_COMPUTE_PIPELINES = 1024;
  const DEFAULT_MAX_BIND_GROUP_LAYOUTS = 512;
  const DEFAULT_PREWARM_CONCURRENCY = 4;
  const DEFAULT_MAX_ERROR_HISTORY = 32;
  const MAX_PIPELINE_ERROR_MESSAGE_LENGTH = 512;
  const MAX_PIPELINE_ERROR_KEY_LENGTH = 64;
  const DEFAULT_MAX_VERTEX_BUFFERS = 8;
  const DEFAULT_MAX_VERTEX_ATTRIBUTES = 16;
  const DEFAULT_MAX_VERTEX_BUFFER_STRIDE = 2048;
  const DEFAULT_COLOR_FORMAT = 'bgra8unorm';
  const FRAME1_VERTEX_STRIDE_24B = 24;
  const FRAME1_CBV_MIN_BINDING_SIZE = 256;

  // WebGPU GPUShaderStage bitmask values with headless/Node fallbacks.
  const SHADER_STAGE_BITS = Object.freeze({
    VERTEX:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.VERTEX === 'number'
        ? GPUShaderStage.VERTEX
        : 0x1,
    FRAGMENT:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.FRAGMENT === 'number'
        ? GPUShaderStage.FRAGMENT
        : 0x2,
    COMPUTE:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.COMPUTE === 'number'
        ? GPUShaderStage.COMPUTE
        : 0x4,
  });

  const STAGE_NAME_TO_BIT = Object.freeze({
    vertex: SHADER_STAGE_BITS.VERTEX,
    vs: SHADER_STAGE_BITS.VERTEX,
    fragment: SHADER_STAGE_BITS.FRAGMENT,
    fs: SHADER_STAGE_BITS.FRAGMENT,
    pixel: SHADER_STAGE_BITS.FRAGMENT,
    ps: SHADER_STAGE_BITS.FRAGMENT,
    compute: SHADER_STAGE_BITS.COMPUTE,
    cs: SHADER_STAGE_BITS.COMPUTE,
    all: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT | SHADER_STAGE_BITS.COMPUTE,
  });

  // WebGPU GPUVertexFormat byte widths (WebGPU specification table).
  const VERTEX_FORMAT_BYTE_SIZES = Object.freeze({
    uint8: 1,
    sint8: 1,
    unorm8: 1,
    snorm8: 1,
    uint8x2: 2,
    sint8x2: 2,
    unorm8x2: 2,
    snorm8x2: 2,
    uint16: 2,
    sint16: 2,
    unorm16: 2,
    snorm16: 2,
    float16: 2,
    uint8x4: 4,
    sint8x4: 4,
    unorm8x4: 4,
    snorm8x4: 4,
    uint16x2: 4,
    sint16x2: 4,
    unorm16x2: 4,
    snorm16x2: 4,
    float16x2: 4,
    float32: 4,
    uint32: 4,
    sint32: 4,
    'unorm10-10-10-2': 4,
    'unorm8x4-bgra': 4,
    uint16x4: 8,
    sint16x4: 8,
    unorm16x4: 8,
    snorm16x4: 8,
    float16x4: 8,
    float32x2: 8,
    uint32x2: 8,
    sint32x2: 8,
    float32x3: 12,
    uint32x3: 12,
    sint32x3: 12,
    float32x4: 16,
    uint32x4: 16,
    sint32x4: 16,
  });

  const VALID_PRIMITIVE_TOPOLOGIES = new Set([
    'point-list',
    'line-list',
    'line-strip',
    'triangle-list',
    'triangle-strip',
  ]);

  // Canonical G34 fallback profile aliases matching web/workers/shader_fallback_router.js.
  const FALLBACK_PROFILE_ALIASES = Object.freeze({
    default: 'default',
    minimal: 'default',
    fullscreen_triangle_cbv: 'fullscreen_triangle_cbv',
    fullscreen_cbv: 'fullscreen_triangle_cbv',
    'fullscreen-triangle-cbv': 'fullscreen_triangle_cbv',
    'fullscreen-cbv': 'fullscreen_triangle_cbv',
    textured_quad_24b: 'textured_quad_24b',
    textured_24b: 'textured_quad_24b',
    'textured-quad-24b': 'textured_quad_24b',
    'textured-24b': 'textured_quad_24b',
  });

  const FALLBACK_LABEL_RE =
    /^fallback:(default|minimal|fullscreen[_-]triangle[_-]cbv|fullscreen[_-]cbv|textured[_-]quad[_-]24b|textured[_-]24b)(?:[:\s]|$)/i;

  // Canonical Frame-1 24-byte-stride vertex layout (DrawInstanced(3, 1) / textured_quad_24b):
  // @location(0) float32x4 @ offset 0 (16 B) + @location(1) float32x2 @ offset 16 (8 B).
  const FRAME1_VERTEX_LAYOUT_24B = Object.freeze({
    arrayStride: FRAME1_VERTEX_STRIDE_24B,
    stepMode: 'vertex',
    attributes: Object.freeze([
      Object.freeze({ shaderLocation: 0, offset: 0, format: 'float32x4' }),
      Object.freeze({ shaderLocation: 1, offset: 16, format: 'float32x2' }),
    ]),
  });

  // Canonical group(0) bind-group layout entries matching G34 fallback WGSL profiles.
  const FALLBACK_BIND_GROUP_ENTRIES = Object.freeze({
    default: Object.freeze([]),
    fullscreen_triangle_cbv: Object.freeze([
      Object.freeze({
        binding: 0,
        visibility: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT,
        buffer: Object.freeze({
          type: 'uniform',
          hasDynamicOffset: false,
          minBindingSize: FRAME1_CBV_MIN_BINDING_SIZE,
        }),
      }),
    ]),
    textured_quad_24b: Object.freeze([
      Object.freeze({
        binding: 0,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        texture: Object.freeze({
          sampleType: 'float',
          viewDimension: '2d',
          multisampled: false,
        }),
      }),
      Object.freeze({
        binding: 1,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: Object.freeze({
          type: 'filtering',
        }),
      }),
    ]),
  });

  // 64-bit FNV-1a constants matching shaders/src/convert.rs & web/workers/wgsl_pack_loader.js.
  const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
  const FNV1A_64_PRIME = 0x00000100000001b3n;
  const U64_MASK = 0xffffffffffffffffn;
  const WGPK_LABEL_RE = /^wgpk:(?:vertex|fragment|compute):(0x[0-9a-f]{1,16})$/i;

  const textEncoder = new TextEncoder();

  // WeakMaps for stable identity of opaque WebGPU objects (GPUShaderModule, GPUBindGroupLayout, etc.)
  // and user-registered WGPK/numeric shader module IDs without leaking GC references.
  const registeredObjectIds = new WeakMap();
  const opaqueObjectIds = new WeakMap();
  const fallbackPipelineLayoutProfiles = new WeakMap();
  let nextOpaqueObjectId = 1;

  function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function formatHashHex64(value) {
    return `0x${(value & U64_MASK).toString(16).padStart(16, '0')}`;
  }

  function normalizeShaderId(id) {
    if (typeof id === 'bigint') {
      return id >= 0n && id <= U64_MASK ? formatHashHex64(id) : `bi:${id.toString(10)}`;
    }
    if (typeof id === 'number') {
      if (!Number.isFinite(id)) {
        throw new RangeError(`Shader module ID must be a finite number, got: ${id}`);
      }
      return `num:${id}`;
    }
    if (typeof id === 'string' && id.length > 0) {
      return `str:${id}`;
    }
    throw new TypeError('Shader module ID must be a BigInt, finite number, or non-empty string');
  }

  /**
   * Associate a WebGPU object (e.g., `GPUShaderModule`, `GPUBindGroupLayout`, `GPUPipelineLayout`)
   * with a deterministic ID or WGPK 64-bit shader hash so `hashDescriptorKey` hashes it stably.
   *
   * @param {object} gpuObject
   * @param {bigint | number | string} id
   * @returns {string} Canonical registered ID token
   */
  function registerShaderModuleId(gpuObject, id) {
    if (!gpuObject || (typeof gpuObject !== 'object' && typeof gpuObject !== 'function')) {
      throw new TypeError('gpuObject must be an object');
    }
    const token = normalizeShaderId(id);
    registeredObjectIds.set(gpuObject, token);
    return token;
  }

  function getOpaqueObjectId(obj) {
    let id = opaqueObjectIds.get(obj);
    if (id === undefined) {
      id = nextOpaqueObjectId++;
      opaqueObjectIds.set(obj, id);
    }
    return id;
  }

  function isNativeWebGpuObject(val) {
    if (typeof GPUShaderModule !== 'undefined' && val instanceof GPUShaderModule) return true;
    if (typeof GPUBindGroupLayout !== 'undefined' && val instanceof GPUBindGroupLayout) return true;
    if (typeof GPUPipelineLayout !== 'undefined' && val instanceof GPUPipelineLayout) return true;
    const tag = Object.prototype.toString.call(val);
    return tag.startsWith('[object GPU');
  }

  function isDuckTypedShaderModule(val) {
    return typeof val.getCompilationInfo === 'function';
  }

  function extractExplicitShaderId(val) {
    if (typeof val.wgpkHash === 'bigint' || typeof val.wgpkHash === 'string' || typeof val.wgpkHash === 'number') {
      return `wgpk:${normalizeShaderId(val.wgpkHash)}`;
    }
    if (typeof val.shaderHash === 'bigint' || typeof val.shaderHash === 'string' || typeof val.shaderHash === 'number') {
      return `sh:${normalizeShaderId(val.shaderHash)}`;
    }
    if (typeof val.hashHex === 'string' && val.hashHex.length > 0) {
      return `hex:${val.hashHex.toLowerCase()}`;
    }
    if (typeof val.hash === 'bigint' || typeof val.hash === 'string' || typeof val.hash === 'number') {
      return `h:${normalizeShaderId(val.hash)}`;
    }
    if (typeof val.shaderId === 'bigint' || typeof val.shaderId === 'string' || typeof val.shaderId === 'number') {
      return `sid:${normalizeShaderId(val.shaderId)}`;
    }
    if (typeof val.moduleId === 'bigint' || typeof val.moduleId === 'string' || typeof val.moduleId === 'number') {
      return `mid:${normalizeShaderId(val.moduleId)}`;
    }
    if (typeof val.label === 'string') {
      const m = WGPK_LABEL_RE.exec(val.label);
      if (m) {
        return `wgpk:${formatHashHex64(BigInt(m[1]))}`;
      }
    }
    return null;
  }

  function hasEnumerableDataKeys(val) {
    const keys = Object.keys(val);
    for (let i = 0; i < keys.length; i += 1) {
      const v = val[keys[i]];
      if (v !== undefined && typeof v !== 'function') {
        return true;
      }
    }
    return false;
  }

  function canonicalizeValue(val, keyName, ancestors) {
    if (val === null) return 'null';
    const t = typeof val;
    if (t === 'undefined') return 'undef';
    if (t === 'boolean') return val ? 'b:1' : 'b:0';
    if (t === 'number') {
      if (Number.isNaN(val)) return 'n:NaN';
      if (val === Infinity) return 'n:Inf';
      if (val === -Infinity) return 'n:-Inf';
      return Object.is(val, -0) ? 'n:0' : `n:${String(val)}`;
    }
    if (t === 'bigint') {
      return `bi:${val.toString(10)}`;
    }
    if (t === 'string') {
      return `s:${val.length}:${val}`;
    }
    if (t === 'symbol') {
      return `sym:${String(val.description || '')}`;
    }
    if (t === 'function') {
      return 'fn';
    }

    // Check registered GPU resource / shader module IDs first.
    const registered = registeredObjectIds.get(val);
    if (registered !== undefined) {
      return `reg:${registered}`;
    }

    // Handle native/opaque WebGPU objects and shader modules before walking keys.
    const isShaderOrLayoutSlot = keyName === 'module' || keyName === 'layout';
    const isGpuHandle = isNativeWebGpuObject(val) || isDuckTypedShaderModule(val);
    if (isGpuHandle || isShaderOrLayoutSlot) {
      const explicitId = extractExplicitShaderId(val);
      if (explicitId !== null) {
        return `gpu:${explicitId}`;
      }
      if (isGpuHandle || !hasEnumerableDataKeys(val)) {
        return `gpu-oid:${getOpaqueObjectId(val)}`;
      }
    }

    if (ancestors.has(val)) {
      throw new TypeError('Circular reference detected in WebGPU descriptor');
    }

    if (Array.isArray(val)) {
      ancestors.add(val);
      const parts = new Array(val.length);
      for (let i = 0; i < val.length; i += 1) {
        parts[i] = canonicalizeValue(val[i], null, ancestors);
      }
      ancestors.delete(val);
      return `a:${val.length}:[${parts.join(',')}]`;
    }

    if (ArrayBuffer.isView(val)) {
      const bytes = new Uint8Array(val.buffer, val.byteOffset, val.byteLength);
      let hex = '';
      for (let i = 0; i < bytes.length; i += 1) {
        hex += bytes[i].toString(16).padStart(2, '0');
      }
      const ctorName = val.constructor ? val.constructor.name : 'View';
      return `bin:${ctorName}:${bytes.byteLength}:${hex}`;
    }

    if (val instanceof ArrayBuffer || (typeof SharedArrayBuffer !== 'undefined' && val instanceof SharedArrayBuffer)) {
      const bytes = new Uint8Array(val);
      let hex = '';
      for (let i = 0; i < bytes.length; i += 1) {
        hex += bytes[i].toString(16).padStart(2, '0');
      }
      return `ab:${bytes.byteLength}:${hex}`;
    }

    if (val instanceof Set) {
      ancestors.add(val);
      const items = [];
      for (const item of val) {
        items.push(canonicalizeValue(item, null, ancestors));
      }
      ancestors.delete(val);
      items.sort();
      return `set:${items.length}:[${items.join(',')}]`;
    }

    if (val instanceof Map) {
      ancestors.add(val);
      const pairs = [];
      for (const [k, v] of val.entries()) {
        pairs.push(`${canonicalizeValue(k, null, ancestors)}=>${canonicalizeValue(v, null, ancestors)}`);
      }
      ancestors.delete(val);
      pairs.sort();
      return `map:${pairs.length}:{${pairs.join(',')}}`;
    }

    // Plain/structured object: sort keys lexicographically for insertion-order invariance.
    ancestors.add(val);
    const keys = Object.keys(val).sort();
    const fields = [];
    for (let i = 0; i < keys.length; i += 1) {
      const k = keys[i];
      const v = val[k];
      if (v === undefined || typeof v === 'function') {
        continue;
      }
      fields.push(`${k.length}:${k}=${canonicalizeValue(v, k, ancestors)}`);
    }
    ancestors.delete(val);
    return `o:${fields.length}:{${fields.join(',')}}`;
  }

  /**
   * Compute a deterministic canonical FNV-1a 64-bit hex hash (`0x` + 16 hex digits)
   * for any WebGPU descriptor or nested JS value, independent of object key insertion order.
   * Supports `BigInt`, nested arrays/objects, TypedArrays, `Map`/`Set`, opaque `GPUShaderModule`s,
   * and WGPK shader hashes.
   *
   * @param {unknown} obj
   * @returns {string}
   */
  function hashDescriptorKey(obj) {
    const canonical = canonicalizeValue(obj, null, new Set());
    const bytes = textEncoder.encode(canonical);
    let hash = FNV1A_64_OFFSET_BASIS;
    for (let i = 0; i < bytes.length; i += 1) {
      hash ^= BigInt(bytes[i]);
      hash = (hash * FNV1A_64_PRIME) & U64_MASK;
    }
    return formatHashHex64(hash);
  }

  function normalizeVisibility(visibility) {
    if (typeof visibility === 'number') {
      if (!Number.isInteger(visibility) || visibility < 0) {
        throw new RangeError(`Invalid shader stage visibility bitmask: ${visibility}`);
      }
      return visibility;
    }

    if (typeof visibility === 'string') {
      const tokens = visibility
        .split(/[\s|,+/]+/)
        .map((s) => s.trim().toLowerCase())
        .filter((s) => s.length > 0);
      if (tokens.length === 0) {
        throw new RangeError('Shader stage visibility string must not be empty');
      }
      let mask = 0;
      for (let i = 0; i < tokens.length; i += 1) {
        const bit = STAGE_NAME_TO_BIT[tokens[i]];
        if (bit === undefined) {
          throw new RangeError(`Unknown shader stage visibility token: "${tokens[i]}"`);
        }
        mask |= bit;
      }
      return mask;
    }

    if (Array.isArray(visibility)) {
      if (visibility.length === 0) {
        throw new RangeError('Shader stage visibility array must not be empty');
      }
      let mask = 0;
      for (let i = 0; i < visibility.length; i += 1) {
        mask |= normalizeVisibility(visibility[i]);
      }
      return mask;
    }

    throw new TypeError('Binding visibility must be a GPUShaderStage bitmask, stage string, or stage array');
  }

  function normalizeReflectionBinding(raw, idx) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      throw new TypeError(`Reflection binding at index ${idx} must be an object`);
    }
    if (!Number.isInteger(raw.binding) || raw.binding < 0) {
      throw new RangeError(`Reflection binding at index ${idx} must have a non-negative integer "binding"`);
    }

    const entry = {
      binding: raw.binding,
      visibility: normalizeVisibility(raw.visibility),
    };

    if (raw.buffer && typeof raw.buffer === 'object') {
      const b = {};
      if (raw.buffer.type !== undefined) b.type = raw.buffer.type;
      const hasDynamicOffset =
        raw.buffer.hasDynamicOffset !== undefined
          ? raw.buffer.hasDynamicOffset
          : raw.buffer.has_dynamic_offset;
      if (hasDynamicOffset !== undefined) b.hasDynamicOffset = Boolean(hasDynamicOffset);
      const minBindingSize =
        raw.buffer.minBindingSize !== undefined
          ? raw.buffer.minBindingSize
          : raw.buffer.min_binding_size;
      if (minBindingSize !== undefined) b.minBindingSize = Number(minBindingSize);
      entry.buffer = b;
    }

    if (raw.sampler && typeof raw.sampler === 'object') {
      const s = {};
      if (raw.sampler.type !== undefined) s.type = raw.sampler.type;
      entry.sampler = s;
    }

    if (raw.texture && typeof raw.texture === 'object') {
      const tx = {};
      const sampleType =
        raw.texture.sampleType !== undefined ? raw.texture.sampleType : raw.texture.sample_type;
      if (sampleType !== undefined) tx.sampleType = sampleType;
      const viewDimension =
        raw.texture.viewDimension !== undefined
          ? raw.texture.viewDimension
          : raw.texture.view_dimension;
      if (viewDimension !== undefined) tx.viewDimension = viewDimension;
      if (raw.texture.multisampled !== undefined) tx.multisampled = Boolean(raw.texture.multisampled);
      entry.texture = tx;
    }

    const rawStorageTex = raw.storageTexture !== undefined ? raw.storageTexture : raw.storage_texture;
    if (rawStorageTex && typeof rawStorageTex === 'object') {
      const st = {};
      if (rawStorageTex.access !== undefined) st.access = rawStorageTex.access;
      if (rawStorageTex.format !== undefined) st.format = rawStorageTex.format;
      const viewDimension =
        rawStorageTex.viewDimension !== undefined
          ? rawStorageTex.viewDimension
          : rawStorageTex.view_dimension;
      if (viewDimension !== undefined) st.viewDimension = viewDimension;
      entry.storageTexture = st;
    }

    const rawExternalTex = raw.externalTexture !== undefined ? raw.externalTexture : raw.external_texture;
    if (rawExternalTex && typeof rawExternalTex === 'object') {
      entry.externalTexture = {};
    }

    return entry;
  }

  function validatePositiveInt(value, name) {
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got: ${value}`);
    }
    return value;
  }

  function alignUp4(value) {
    return Math.ceil(value / 4) * 4;
  }

  function alignDown(value, align) {
    if (align <= 1) return value;
    return value - (value % align);
  }

  /**
   * Normalize a built-in fallback profile name or alias to a canonical G34 profile key.
   * Unknown or omitted values safely resolve to `'default'`.
   *
   * @param {string | null | undefined} profile
   * @returns {'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b'}
   */
  function normalizeFallbackProfile(profile) {
    if (typeof profile !== 'string') {
      return 'default';
    }
    const key = profile.trim().toLowerCase();
    if (key.length === 0) {
      return 'default';
    }
    if (Object.prototype.hasOwnProperty.call(FALLBACK_PROFILE_ALIASES, key)) {
      return FALLBACK_PROFILE_ALIASES[key];
    }
    return 'default';
  }

  function detectModuleFallbackProfile(mod) {
    if (!mod || typeof mod !== 'object') {
      return null;
    }
    if (typeof mod.fallbackProfile === 'string' && mod.fallbackProfile.trim().length > 0) {
      return normalizeFallbackProfile(mod.fallbackProfile);
    }
    if (typeof mod.profile === 'string' && mod.profile.trim().length > 0) {
      return normalizeFallbackProfile(mod.profile);
    }
    if (typeof mod.label === 'string') {
      const m = FALLBACK_LABEL_RE.exec(mod.label.trim());
      if (m) {
        return normalizeFallbackProfile(m[1]);
      }
    }
    if (mod.isFallback === true || mod.source === 'builtin') {
      return 'default';
    }
    return null;
  }

  function detectFallbackProfile(descriptor, options) {
    const opts = options && typeof options === 'object' ? options : null;
    if (opts) {
      if (typeof opts.fallbackProfile === 'string' && opts.fallbackProfile.trim().length > 0) {
        return normalizeFallbackProfile(opts.fallbackProfile);
      }
      if (typeof opts.profile === 'string' && opts.profile.trim().length > 0) {
        return normalizeFallbackProfile(opts.profile);
      }
    }
    if (descriptor && typeof descriptor === 'object') {
      if (
        typeof descriptor.fallbackProfile === 'string' &&
        descriptor.fallbackProfile.trim().length > 0
      ) {
        return normalizeFallbackProfile(descriptor.fallbackProfile);
      }
      const vsProfile =
        descriptor.vertex && typeof descriptor.vertex === 'object'
          ? detectModuleFallbackProfile(descriptor.vertex.module)
          : null;
      const fsProfile =
        descriptor.fragment && typeof descriptor.fragment === 'object'
          ? detectModuleFallbackProfile(descriptor.fragment.module)
          : null;
      if (vsProfile && vsProfile !== 'default') {
        return vsProfile;
      }
      if (fsProfile && fsProfile !== 'default') {
        return fsProfile;
      }
      if (vsProfile || fsProfile) {
        return vsProfile || fsProfile;
      }
    }
    if (opts && opts.isFallback === true) {
      return 'default';
    }
    return null;
  }

  /**
   * Return the byte size of a valid WebGPU `GPUVertexFormat`, or `0` if unrecognized.
   *
   * @param {string} format
   * @returns {number}
   */
  function getVertexFormatByteSize(format) {
    if (
      typeof format === 'string' &&
      Object.prototype.hasOwnProperty.call(VERTEX_FORMAT_BYTE_SIZES, format)
    ) {
      return VERTEX_FORMAT_BYTE_SIZES[format];
    }
    return 0;
  }

  function cloneFrame1VertexLayout24B(stepMode) {
    return {
      arrayStride: FRAME1_VERTEX_STRIDE_24B,
      stepMode: stepMode === 'instance' ? 'instance' : 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    };
  }

  function isCanonical24BLayoutList(buffers) {
    if (!Array.isArray(buffers) || buffers.length !== 1) {
      return false;
    }
    const buf = buffers[0];
    if (!buf || typeof buf !== 'object') {
      return false;
    }
    if (buf.arrayStride !== FRAME1_VERTEX_STRIDE_24B) {
      return false;
    }
    if (buf.stepMode !== undefined && buf.stepMode !== 'vertex') {
      return false;
    }
    if (!Array.isArray(buf.attributes) || buf.attributes.length !== 2) {
      return false;
    }
    const a0 = buf.attributes[0];
    const a1 = buf.attributes[1];
    return (
      Boolean(a0) &&
      Boolean(a1) &&
      a0.shaderLocation === 0 &&
      a0.offset === 0 &&
      a0.format === 'float32x4' &&
      a1.shaderLocation === 1 &&
      a1.offset === 16 &&
      a1.format === 'float32x2'
    );
  }

  function resolveVertexLimits(options) {
    const opts = options && typeof options === 'object' ? options : {};
    const maxVertexBuffers =
      Number.isInteger(opts.maxVertexBuffers) && opts.maxVertexBuffers > 0
        ? opts.maxVertexBuffers
        : DEFAULT_MAX_VERTEX_BUFFERS;
    const maxVertexAttributes =
      Number.isInteger(opts.maxVertexAttributes) && opts.maxVertexAttributes > 0
        ? opts.maxVertexAttributes
        : DEFAULT_MAX_VERTEX_ATTRIBUTES;
    const rawMaxStride =
      Number.isInteger(opts.maxVertexBufferArrayStride) && opts.maxVertexBufferArrayStride >= 4
        ? opts.maxVertexBufferArrayStride
        : DEFAULT_MAX_VERTEX_BUFFER_STRIDE;
    const maxVertexBufferArrayStride = Math.max(4, alignDown(rawMaxStride, 4));
    return { maxVertexBuffers, maxVertexAttributes, maxVertexBufferArrayStride };
  }

  function nextFreeVertexLocation(usedLocations, maxVertexAttributes) {
    for (let loc = 0; loc < maxVertexAttributes; loc += 1) {
      if (!usedLocations.has(loc)) {
        return loc;
      }
    }
    return -1;
  }

  /**
   * Build a WebGPU-valid `GPUVertexBufferLayout[]` array for a given byte stride and/or
   * G34 fallback profile.
   *
   * - `fullscreen_triangle_cbv` / `default` -> `[]` (`@builtin(vertex_index)` shaders)
   * - `textured_quad_24b` or stride 24 -> 24B layout (`float32x4` @ 0 loc 0 + `float32x2` @ 16 loc 1)
   * - Other positive strides -> 4-byte aligned, clamped to `maxVertexBufferArrayStride`.
   *
   * @param {number} vertexStride
   * @param {{
   *   fallbackProfile?: string | null,
   *   profile?: string | null,
   *   stepMode?: 'vertex' | 'instance',
   *   maxVertexBufferArrayStride?: number,
   * }} [options]
   * @returns {Array<GPUVertexBufferLayout>}
   */
  function buildVertexBufferLayoutForStride(vertexStride, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const stepMode = opts.stepMode === 'instance' ? 'instance' : 'vertex';
    const rawProfile =
      typeof opts.fallbackProfile === 'string' && opts.fallbackProfile.trim().length > 0
        ? opts.fallbackProfile
        : typeof opts.profile === 'string' && opts.profile.trim().length > 0
          ? opts.profile
          : null;

    if (rawProfile !== null) {
      const profile = normalizeFallbackProfile(rawProfile);
      if (profile === 'fullscreen_triangle_cbv' || profile === 'default') {
        return [];
      }
      if (profile === 'textured_quad_24b') {
        return [cloneFrame1VertexLayout24B(stepMode)];
      }
    }

    if (typeof vertexStride !== 'number' || !Number.isFinite(vertexStride) || vertexStride <= 0) {
      return [];
    }

    const { maxVertexBufferArrayStride } = resolveVertexLimits(opts);
    const alignedStride = Math.min(maxVertexBufferArrayStride, alignUp4(Math.max(4, vertexStride)));

    if (alignedStride >= FRAME1_VERTEX_STRIDE_24B) {
      return [
        {
          arrayStride: alignedStride,
          stepMode,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x4' },
            { shaderLocation: 1, offset: 16, format: 'float32x2' },
          ],
        },
      ];
    }

    const format =
      alignedStride >= 16
        ? 'float32x4'
        : alignedStride >= 12
          ? 'float32x3'
          : alignedStride >= 8
            ? 'float32x2'
            : 'float32';

    return [
      {
        arrayStride: alignedStride,
        stepMode,
        attributes: [{ shaderLocation: 0, offset: 0, format }],
      },
    ];
  }

  function normalizeVertexBufferLayoutsInternal(buffers, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const { maxVertexBuffers, maxVertexAttributes, maxVertexBufferArrayStride } =
      resolveVertexLimits(opts);
    const reasons = [];
    const addReason = (reason) => {
      if (!reasons.includes(reason)) {
        reasons.push(reason);
      }
    };

    const rawProfile =
      typeof opts.fallbackProfile === 'string' && opts.fallbackProfile.trim().length > 0
        ? opts.fallbackProfile
        : typeof opts.profile === 'string' && opts.profile.trim().length > 0
          ? opts.profile
          : null;

    if (rawProfile !== null) {
      const profile = normalizeFallbackProfile(rawProfile);
      if (profile === 'fullscreen_triangle_cbv' || profile === 'default') {
        const alreadyEmpty = Array.isArray(buffers) && buffers.length === 0;
        if (!alreadyEmpty) {
          addReason('fallback-empty-vertex-buffers');
        }
        return { buffers: [], changed: !alreadyEmpty, reasons };
      }
      if (profile === 'textured_quad_24b') {
        const alreadyCanonical = isCanonical24BLayoutList(buffers);
        if (!alreadyCanonical) {
          addReason('fallback-24b-vertex-layout');
        }
        return {
          buffers: [cloneFrame1VertexLayout24B('vertex')],
          changed: !alreadyCanonical,
          reasons,
        };
      }
    }

    if (typeof buffers === 'number') {
      const built = buildVertexBufferLayoutForStride(buffers, opts);
      addReason('stride-to-vertex-layout');
      return { buffers: built, changed: true, reasons };
    }

    if (buffers === undefined || buffers === null) {
      if (
        typeof opts.vertexStride === 'number' &&
        Number.isFinite(opts.vertexStride) &&
        opts.vertexStride > 0
      ) {
        const built = buildVertexBufferLayoutForStride(opts.vertexStride, opts);
        addReason('vertex-stride-layout-synthesized');
        return { buffers: built, changed: true, reasons };
      }
      const changed = buffers === null;
      if (changed) {
        addReason('null-vertex-buffers-cleared');
      }
      return { buffers: [], changed, reasons };
    }

    if (!Array.isArray(buffers)) {
      addReason('invalid-vertex-buffers-cleared');
      return { buffers: [], changed: true, reasons };
    }

    if (buffers.length === 0) {
      if (
        typeof opts.vertexStride === 'number' &&
        Number.isFinite(opts.vertexStride) &&
        opts.vertexStride > 0
      ) {
        const built = buildVertexBufferLayoutForStride(opts.vertexStride, opts);
        addReason('vertex-stride-layout-synthesized');
        return { buffers: built, changed: true, reasons };
      }
      return { buffers: [], changed: false, reasons };
    }

    let changed = false;
    if (buffers.length > maxVertexBuffers) {
      changed = true;
      addReason('clamped-vertex-buffer-count');
    }

    const usedLocations = new Set();
    const normalizedBuffers = [];
    const slotCount = Math.min(buffers.length, maxVertexBuffers);

    for (let i = 0; i < slotCount; i += 1) {
      const buf = buffers[i];
      if (!buf || typeof buf !== 'object' || Array.isArray(buf)) {
        changed = true;
        addReason('dropped-invalid-vertex-buffer');
        continue;
      }

      let stepMode = 'vertex';
      if (buf.stepMode === 'instance') {
        stepMode = 'instance';
      } else if (buf.stepMode !== undefined && buf.stepMode !== 'vertex') {
        changed = true;
        addReason('normalized-step-mode');
      }

      const rawStride = buf.arrayStride;
      const isExplicitZero = rawStride === 0;
      const isValidPositiveStride =
        typeof rawStride === 'number' && Number.isFinite(rawStride) && rawStride > 0;

      let stride = 0;
      if (isValidPositiveStride) {
        const aligned = Math.min(maxVertexBufferArrayStride, alignUp4(rawStride));
        if (aligned !== rawStride) {
          changed = true;
          addReason('clamped-vertex-stride');
        }
        stride = aligned;
      } else if (isExplicitZero) {
        stride = 0;
      } else {
        changed = true;
        addReason('invalid-vertex-stride');
      }

      const rawAttrs = buf.attributes;
      const normalizedAttrs = [];

      if (Array.isArray(rawAttrs) && rawAttrs.length > 0) {
        for (let j = 0; j < rawAttrs.length; j += 1) {
          const attr = rawAttrs[j];
          if (!attr || typeof attr !== 'object' || Array.isArray(attr)) {
            changed = true;
            addReason('dropped-invalid-vertex-attribute');
            continue;
          }

          let format = attr.format;
          let fmtSize = getVertexFormatByteSize(format);
          if (fmtSize === 0) {
            format =
              stride >= 16 || stride === 0
                ? 'float32x4'
                : stride >= 12
                  ? 'float32x3'
                  : stride >= 8
                    ? 'float32x2'
                    : 'float32';
            fmtSize = getVertexFormatByteSize(format);
            changed = true;
            addReason('fallback-vertex-format');
          }

          const align = Math.min(4, fmtSize);
          let offset;
          if (typeof attr.offset !== 'number' || !Number.isFinite(attr.offset) || attr.offset < 0) {
            offset = 0;
            changed = true;
            addReason('clamped-vertex-offset');
          } else {
            const floored = Math.floor(attr.offset);
            offset = alignDown(floored, align);
            if (offset !== attr.offset) {
              changed = true;
              addReason('aligned-vertex-offset');
            }
          }

          if (isValidPositiveStride) {
            if (stride < fmtSize) {
              stride = Math.min(maxVertexBufferArrayStride, alignUp4(fmtSize));
              offset = 0;
              changed = true;
              addReason('expanded-vertex-stride');
            } else if (offset + fmtSize > stride) {
              offset = alignDown(stride - fmtSize, align);
              changed = true;
              addReason('clamped-vertex-offset');
            }
          } else if (isExplicitZero) {
            if (offset + fmtSize > maxVertexBufferArrayStride) {
              offset = alignDown(maxVertexBufferArrayStride - fmtSize, align);
              changed = true;
              addReason('clamped-vertex-offset');
            }
          } else {
            if (offset + fmtSize > maxVertexBufferArrayStride) {
              offset = alignDown(maxVertexBufferArrayStride - fmtSize, align);
              changed = true;
              addReason('clamped-vertex-offset');
            }
            const required = Math.min(maxVertexBufferArrayStride, alignUp4(offset + fmtSize));
            if (required > stride) {
              stride = required;
            }
          }

          let loc = attr.shaderLocation;
          if (
            !Number.isInteger(loc) ||
            loc < 0 ||
            loc >= maxVertexAttributes ||
            usedLocations.has(loc)
          ) {
            const nextLoc = nextFreeVertexLocation(usedLocations, maxVertexAttributes);
            if (nextLoc < 0) {
              changed = true;
              addReason('clamped-vertex-attribute-count');
              continue;
            }
            loc = nextLoc;
            changed = true;
            addReason('reassigned-vertex-location');
          }

          usedLocations.add(loc);
          normalizedAttrs.push({
            shaderLocation: loc,
            offset,
            format,
          });
        }
      } else {
        if (!Array.isArray(rawAttrs)) {
          changed = true;
        }
        if (stride > 0) {
          const synthesized = buildVertexBufferLayoutForStride(stride, {
            stepMode,
            maxVertexBufferArrayStride,
          });
          const templateAttrs = synthesized.length > 0 ? synthesized[0].attributes : [];
          for (let k = 0; k < templateAttrs.length; k += 1) {
            const tpl = templateAttrs[k];
            const loc = usedLocations.has(tpl.shaderLocation)
              ? nextFreeVertexLocation(usedLocations, maxVertexAttributes)
              : tpl.shaderLocation;
            if (loc < 0) {
              break;
            }
            usedLocations.add(loc);
            normalizedAttrs.push({
              shaderLocation: loc,
              offset: tpl.offset,
              format: tpl.format,
            });
          }
          changed = true;
          addReason(
            stride === FRAME1_VERTEX_STRIDE_24B
              ? 'synthesized-24b-attributes'
              : 'synthesized-vertex-attributes'
          );
        } else if (!isExplicitZero) {
          changed = true;
          addReason('dropped-empty-invalid-vertex-buffer');
          continue;
        }
      }

      normalizedBuffers.push({
        arrayStride: stride,
        stepMode,
        attributes: normalizedAttrs,
      });
    }

    return {
      buffers: normalizedBuffers,
      changed,
      reasons,
    };
  }

  /**
   * Normalize WebGPU vertex buffer layouts for real Frame-1 Xbox draws (including 24-byte-stride
   * `DrawInstanced(3, 1)` buffers) and G34 fallback shader profiles (`fullscreen_triangle_cbv`,
   * `textured_quad_24b`, `default`). Clamps invalid strides/offsets and deduplicates locations.
   *
   * @param {Array<GPUVertexBufferLayout> | number | null | undefined} buffers
   * @param {{
   *   fallbackProfile?: string | null,
   *   profile?: string | null,
   *   vertexStride?: number,
   *   maxVertexBuffers?: number,
   *   maxVertexAttributes?: number,
   *   maxVertexBufferArrayStride?: number,
   * }} [options]
   * @returns {Array<GPUVertexBufferLayout>}
   */
  function normalizeVertexBufferLayouts(buffers, options = {}) {
    return normalizeVertexBufferLayoutsInternal(buffers, options).buffers;
  }

  /**
   * Build a fresh `GPUBindGroupLayoutDescriptor` for group(0) matching a G34 fallback profile:
   * - `fullscreen_triangle_cbv`: `binding(0)` uniform CBV (vertex | fragment, minBindingSize 256)
   * - `textured_quad_24b`: `binding(0)` 2D float texture + `binding(1)` filtering sampler (fragment)
   * - `default`: empty `entries: []`
   *
   * @param {string | null | undefined} profile
   * @returns {GPUBindGroupLayoutDescriptor}
   */
  function buildFallbackBindGroupLayoutDescriptor(profile) {
    const canonical = normalizeFallbackProfile(profile);
    const template = FALLBACK_BIND_GROUP_ENTRIES[canonical] || FALLBACK_BIND_GROUP_ENTRIES.default;
    const entries = template.map((entry) => {
      const copy = {
        binding: entry.binding,
        visibility: entry.visibility,
      };
      if (entry.buffer) copy.buffer = { ...entry.buffer };
      if (entry.texture) copy.texture = { ...entry.texture };
      if (entry.sampler) copy.sampler = { ...entry.sampler };
      return copy;
    });
    return {
      label: `fallback-bgl:${canonical}:group0`,
      entries,
    };
  }

  /**
   * Normalize a `GPURenderPipelineDescriptor` for safe WebGPU validation without mutating the
   * caller's original descriptor. Already-valid non-fallback descriptors pass through untouched.
   *
   * @param {GPURenderPipelineDescriptor} descriptor
   * @param {{
   *   fallbackProfile?: string | null,
   *   profile?: string | null,
   *   isFallback?: boolean,
   *   vertexStride?: number,
   *   fallbackLayout?: 'auto' | GPUPipelineLayout,
   *   defaultColorFormat?: string,
   *   forceFallbackEntryPoints?: boolean,
   *   maxVertexBuffers?: number,
   *   maxVertexAttributes?: number,
   *   maxVertexBufferArrayStride?: number,
   * }} [options]
   * @returns {{
   *   descriptor: GPURenderPipelineDescriptor,
   *   normalized: boolean,
   *   vertexLayoutNormalized: boolean,
   *   layoutNormalized: boolean,
   *   fallbackProfile: 'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | null,
   *   reasons: Array<string>,
   * }}
   */
  function normalizeRenderPipelineDescriptor(descriptor, options = {}) {
    if (!descriptor || typeof descriptor !== 'object' || Array.isArray(descriptor)) {
      throw new TypeError('normalizeRenderPipelineDescriptor requires a descriptor object');
    }

    const opts = options && typeof options === 'object' ? options : {};
    const fallbackProfile = detectFallbackProfile(descriptor, opts);
    const defaultColorFormat =
      typeof opts.defaultColorFormat === 'string' && opts.defaultColorFormat.length > 0
        ? opts.defaultColorFormat
        : DEFAULT_COLOR_FORMAT;

    const reasons = [];
    const addReason = (r) => {
      if (!reasons.includes(r)) {
        reasons.push(r);
      }
    };

    let layoutNormalized = false;
    let vertexLayoutNormalized = false;
    let otherNormalized = false;

    const hasHelperMeta =
      Object.prototype.hasOwnProperty.call(descriptor, 'fallbackProfile') ||
      Object.prototype.hasOwnProperty.call(descriptor, 'vertexStride');

    // 1. Pipeline layout normalization
    const rawLayout = descriptor.layout;
    const hasValidRawLayout =
      rawLayout === 'auto' || (rawLayout !== null && typeof rawLayout === 'object');
    let nextLayout = rawLayout;

    if (fallbackProfile !== null) {
      if (opts.fallbackLayout !== undefined && opts.fallbackLayout !== null) {
        if (rawLayout !== opts.fallbackLayout) {
          nextLayout = opts.fallbackLayout;
          layoutNormalized = true;
          addReason('fallback-layout-supplied');
        }
      } else if (!hasValidRawLayout) {
        nextLayout = 'auto';
        layoutNormalized = true;
        addReason('missing-layout');
      } else if (
        rawLayout !== 'auto' &&
        fallbackPipelineLayoutProfiles.get(rawLayout) !== fallbackProfile
      ) {
        nextLayout = 'auto';
        layoutNormalized = true;
        addReason('fallback-layout-auto');
      }
    } else if (!hasValidRawLayout) {
      nextLayout = 'auto';
      layoutNormalized = true;
      addReason('missing-layout');
    }

    // 2. Vertex stage & vertex-buffer layout normalization
    let nextVertex = descriptor.vertex;
    if (descriptor.vertex && typeof descriptor.vertex === 'object') {
      const effectiveStride =
        typeof opts.vertexStride === 'number'
          ? opts.vertexStride
          : typeof descriptor.vertexStride === 'number'
            ? descriptor.vertexStride
            : undefined;

      const shouldInspectBuffers =
        fallbackProfile !== null ||
        descriptor.vertex.buffers !== undefined ||
        (typeof effectiveStride === 'number' &&
          Number.isFinite(effectiveStride) &&
          effectiveStride > 0);

      let nextBuffers = descriptor.vertex.buffers;
      if (shouldInspectBuffers) {
        const vblResult = normalizeVertexBufferLayoutsInternal(descriptor.vertex.buffers, {
          fallbackProfile,
          vertexStride: effectiveStride,
          maxVertexBuffers: opts.maxVertexBuffers,
          maxVertexAttributes: opts.maxVertexAttributes,
          maxVertexBufferArrayStride: opts.maxVertexBufferArrayStride,
        });
        if (vblResult.changed) {
          nextBuffers = vblResult.buffers;
          vertexLayoutNormalized = true;
          for (let i = 0; i < vblResult.reasons.length; i += 1) {
            addReason(vblResult.reasons[i]);
          }
        }
      }

      let nextVsEntryPoint = descriptor.vertex.entryPoint;
      if (
        fallbackProfile !== null &&
        (typeof nextVsEntryPoint !== 'string' ||
          nextVsEntryPoint.length === 0 ||
          opts.forceFallbackEntryPoints === true) &&
        nextVsEntryPoint !== 'vs_main'
      ) {
        nextVsEntryPoint = 'vs_main';
        otherNormalized = true;
        addReason('fallback-vertex-entrypoint');
      }

      if (vertexLayoutNormalized || nextVsEntryPoint !== descriptor.vertex.entryPoint) {
        nextVertex = {
          ...descriptor.vertex,
          entryPoint: nextVsEntryPoint,
        };
        if (shouldInspectBuffers && vertexLayoutNormalized) {
          nextVertex.buffers = nextBuffers;
        }
      }
    }

    // 3. Fragment stage normalization
    let nextFragment = descriptor.fragment;
    if (descriptor.fragment && typeof descriptor.fragment === 'object') {
      let fragChanged = false;
      let nextFsEntryPoint = descriptor.fragment.entryPoint;
      if (
        fallbackProfile !== null &&
        (typeof nextFsEntryPoint !== 'string' ||
          nextFsEntryPoint.length === 0 ||
          opts.forceFallbackEntryPoints === true) &&
        nextFsEntryPoint !== 'fs_main'
      ) {
        nextFsEntryPoint = 'fs_main';
        fragChanged = true;
        otherNormalized = true;
        addReason('fallback-fragment-entrypoint');
      }

      let nextTargets = descriptor.fragment.targets;
      if (!Array.isArray(nextTargets) || nextTargets.length === 0) {
        nextTargets = [{ format: defaultColorFormat }];
        fragChanged = true;
        otherNormalized = true;
        addReason('default-color-target');
      }

      if (fragChanged) {
        nextFragment = {
          ...descriptor.fragment,
          entryPoint: nextFsEntryPoint,
          targets: nextTargets,
        };
      }
    }

    // 4. Primitive topology normalization
    let nextPrimitive = descriptor.primitive;
    if (
      descriptor.primitive &&
      typeof descriptor.primitive === 'object' &&
      descriptor.primitive.topology !== undefined &&
      !VALID_PRIMITIVE_TOPOLOGIES.has(descriptor.primitive.topology)
    ) {
      nextPrimitive = {
        ...descriptor.primitive,
        topology: 'triangle-list',
      };
      otherNormalized = true;
      addReason('clamped-primitive-topology');
    }

    const normalized = layoutNormalized || vertexLayoutNormalized || otherNormalized;
    if (!normalized && !hasHelperMeta) {
      return {
        descriptor,
        normalized: false,
        vertexLayoutNormalized: false,
        layoutNormalized: false,
        fallbackProfile,
        reasons,
      };
    }

    const nextDescriptor = { ...descriptor };
    if (hasHelperMeta) {
      delete nextDescriptor.fallbackProfile;
      delete nextDescriptor.vertexStride;
    }
    nextDescriptor.layout = nextLayout;
    if (nextVertex !== undefined) {
      nextDescriptor.vertex = nextVertex;
    }
    if (nextFragment !== undefined) {
      nextDescriptor.fragment = nextFragment;
    }
    if (nextPrimitive !== undefined) {
      nextDescriptor.primitive = nextPrimitive;
    }

    return {
      descriptor: nextDescriptor,
      normalized,
      vertexLayoutNormalized,
      layoutNormalized,
      fallbackProfile,
      reasons,
    };
  }

  function normalizeComputePipelineDescriptor(descriptor) {
    if (!descriptor || typeof descriptor !== 'object' || Array.isArray(descriptor)) {
      throw new TypeError('normalizeComputePipelineDescriptor requires a descriptor object');
    }
    const rawLayout = descriptor.layout;
    const hasValidRawLayout =
      rawLayout === 'auto' || (rawLayout !== null && typeof rawLayout === 'object');
    if (hasValidRawLayout) {
      return {
        descriptor,
        normalized: false,
        layoutNormalized: false,
      };
    }
    return {
      descriptor: {
        ...descriptor,
        layout: 'auto',
      },
      normalized: true,
      layoutNormalized: true,
    };
  }

  function classifyPipelineErrorType(err) {
    if (err && typeof err === 'object') {
      if (typeof GPUOutOfMemoryError === 'function' && err instanceof GPUOutOfMemoryError) {
        return 'out-of-memory';
      }
      if (typeof GPUValidationError === 'function' && err instanceof GPUValidationError) {
        return 'validation';
      }
      if (typeof GPUInternalError === 'function' && err instanceof GPUInternalError) {
        return 'internal';
      }
      if (err instanceof TypeError) {
        return 'type';
      }
      if (err instanceof RangeError) {
        return 'range';
      }
      const candidates = [
        err.type,
        err.name,
        err.constructor && err.constructor.name,
      ];
      for (let i = 0; i < candidates.length; i += 1) {
        const raw = candidates[i];
        if (typeof raw !== 'string') continue;
        const lower = raw.trim().toLowerCase();
        if (!lower) continue;
        if (
          lower.includes('outofmemory') ||
          lower.includes('out-of-memory') ||
          lower.includes('out_of_memory') ||
          lower === 'oom'
        ) {
          return 'out-of-memory';
        }
        if (lower.includes('validation')) {
          return 'validation';
        }
        if (lower.includes('internal')) {
          return 'internal';
        }
        if (lower === 'typeerror') {
          return 'type';
        }
        if (lower === 'rangeerror') {
          return 'range';
        }
      }
    }
    return 'unknown';
  }

  /**
   * Sanitize a pipeline creation error message so telemetry never leaks host filesystem paths,
   * embedded binary/hex/base64 dumps, control characters, or unbounded strings.
   *
   * @param {unknown} rawMessage
   * @param {number} [maxLength]
   * @returns {string}
   */
  function sanitizePipelineErrorMessage(rawMessage, maxLength = MAX_PIPELINE_ERROR_MESSAGE_LENGTH) {
    let msg;
    if (typeof rawMessage === 'string') {
      msg = rawMessage;
    } else if (rawMessage === null || rawMessage === undefined) {
      return '';
    } else if (
      typeof rawMessage === 'object' &&
      typeof rawMessage.message === 'string'
    ) {
      msg = rawMessage.message;
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

    const limit =
      Number.isInteger(maxLength) && maxLength >= 4
        ? maxLength
        : MAX_PIPELINE_ERROR_MESSAGE_LENGTH;
    if (msg.length > limit) {
      msg = `${msg.slice(0, limit - 3)}...`;
    }
    return msg;
  }

  function isDescriptorWrapper(item) {
    return (
      Boolean(item) &&
      typeof item === 'object' &&
      !Array.isArray(item) &&
      Object.prototype.hasOwnProperty.call(item, 'descriptor') &&
      !Object.prototype.hasOwnProperty.call(item, 'vertex') &&
      !Object.prototype.hasOwnProperty.call(item, 'compute')
    );
  }

  /**
   * LRU-bounded WebGPU RenderPipeline, ComputePipeline, and BindGroupLayout cache
   * with async compilation in-flight promise deduplication, batch prewarm, fallback-safe
   * Frame-1 vertex/CBV layout normalization, structured error tracking, and telemetry.
   */
  class WebGpuPipelineCache {
    /**
     * @param {{
     *   device: GPUDevice,
     *   maxRenderPipelines?: number,
     *   maxComputePipelines?: number,
     *   maxBindGroupLayouts?: number,
     *   maxErrorHistory?: number,
     *   defaultColorFormat?: string,
     *   onPipelineError?: (record: object) => void,
     *   now?: () => number,
     * }} options
     */
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('WebGpuPipelineCache requires an options object with a WebGPU device');
      }
      if (!options.device || typeof options.device !== 'object') {
        throw new TypeError('WebGpuPipelineCache requires a valid GPUDevice in options.device');
      }

      this._device = options.device;
      this._maxRenderPipelines = validatePositiveInt(
        options.maxRenderPipelines !== undefined
          ? options.maxRenderPipelines
          : DEFAULT_MAX_RENDER_PIPELINES,
        'maxRenderPipelines'
      );
      this._maxComputePipelines = validatePositiveInt(
        options.maxComputePipelines !== undefined
          ? options.maxComputePipelines
          : DEFAULT_MAX_COMPUTE_PIPELINES,
        'maxComputePipelines'
      );
      this._maxBindGroupLayouts = validatePositiveInt(
        options.maxBindGroupLayouts !== undefined
          ? options.maxBindGroupLayouts
          : DEFAULT_MAX_BIND_GROUP_LAYOUTS,
        'maxBindGroupLayouts'
      );
      this._maxErrorHistory = validatePositiveInt(
        options.maxErrorHistory !== undefined
          ? options.maxErrorHistory
          : DEFAULT_MAX_ERROR_HISTORY,
        'maxErrorHistory'
      );

      if (options.now !== undefined && typeof options.now !== 'function') {
        throw new TypeError('options.now must be a function when provided');
      }
      this._now = options.now || defaultNow;

      if (
        options.onPipelineError !== undefined &&
        options.onPipelineError !== null &&
        typeof options.onPipelineError !== 'function'
      ) {
        throw new TypeError('options.onPipelineError must be a function when provided');
      }
      this._onPipelineError =
        typeof options.onPipelineError === 'function' ? options.onPipelineError : null;

      this._defaultColorFormat =
        typeof options.defaultColorFormat === 'string' && options.defaultColorFormat.length > 0
          ? options.defaultColorFormat
          : DEFAULT_COLOR_FORMAT;

      this._renderPipelines = new Map();
      this._computePipelines = new Map();
      this._bindGroupLayouts = new Map();
      this._fallbackPipelineLayouts = new Map();
      this._inFlightRender = new Map();
      this._inFlightCompute = new Map();
      this._generation = 0;

      this._pipelineErrors = [];
      this._pipelineErrorSeq = 0;

      this._stats = {
        renderHits: 0,
        renderMisses: 0,
        computeHits: 0,
        computeMisses: 0,
        layoutHits: 0,
        layoutMisses: 0,
        inFlightDedupes: 0,
        evictions: 0,
        totalCompileMs: 0,
        fallbackNormalizedCount: 0,
        vertexLayoutNormalizedCount: 0,
        layoutNormalizedCount: 0,
        fallbackRecoveryCount: 0,
        pipelineErrorCount: 0,
      };
    }

    /**
     * Associate a `GPUShaderModule` with its WGPK 64-bit hash or numeric ID
     * so descriptors referencing it hash deterministically.
     *
     * @param {object} shaderModule
     * @param {bigint | number | string} hashOrId
     * @returns {string}
     */
    registerShaderModule(shaderModule, hashOrId) {
      return registerShaderModuleId(shaderModule, hashOrId);
    }

    _deviceVertexLimits() {
      const limits = this._device && this._device.limits;
      if (!limits || typeof limits !== 'object') {
        return {};
      }
      const out = {};
      if (Number.isInteger(limits.maxVertexBuffers) && limits.maxVertexBuffers > 0) {
        out.maxVertexBuffers = limits.maxVertexBuffers;
      }
      if (Number.isInteger(limits.maxVertexAttributes) && limits.maxVertexAttributes > 0) {
        out.maxVertexAttributes = limits.maxVertexAttributes;
      }
      if (
        Number.isInteger(limits.maxVertexBufferArrayStride) &&
        limits.maxVertexBufferArrayStride >= 4
      ) {
        out.maxVertexBufferArrayStride = limits.maxVertexBufferArrayStride;
      }
      return out;
    }

    _recordPipelineError(kind, err, details = {}) {
      this._stats.pipelineErrorCount += 1;
      this._pipelineErrorSeq += 1;

      const rawKey = details.key;
      const sanitizedKey =
        rawKey !== undefined && rawKey !== null
          ? sanitizePipelineErrorMessage(String(rawKey), MAX_PIPELINE_ERROR_KEY_LENGTH)
          : null;

      const record = {
        seq: this._pipelineErrorSeq,
        kind,
        errorType: classifyPipelineErrorType(err),
        key: sanitizedKey,
        fallbackProfile: details.fallbackProfile || null,
        recovered: Boolean(details.recovered),
        message: sanitizePipelineErrorMessage(err),
        timestampMs: this._now(),
      };

      while (this._pipelineErrors.length >= this._maxErrorHistory) {
        this._pipelineErrors.shift();
      }
      this._pipelineErrors.push(record);

      if (this._onPipelineError) {
        try {
          this._onPipelineError({ ...record });
        } catch (_callbackErr) {
          // Never allow user error callback throws to disrupt pipeline cache error tracking.
        }
      }

      return record;
    }

    _recordNormalizationStats(norm) {
      if (!norm) return;
      if (norm.normalized) {
        this._stats.fallbackNormalizedCount += 1;
      }
      if (norm.vertexLayoutNormalized) {
        this._stats.vertexLayoutNormalizedCount += 1;
      }
      if (norm.layoutNormalized) {
        this._stats.layoutNormalizedCount += 1;
      }
    }

    /**
     * Return shallow copies of sanitized pipeline creation errors retained in the bounded ring.
     *
     * @returns {Array<{
     *   seq: number,
     *   kind: string,
     *   errorType: string,
     *   key: string | null,
     *   fallbackProfile: string | null,
     *   recovered: boolean,
     *   message: string,
     *   timestampMs: number,
     * }>}
     */
    getPipelineErrors() {
      return this._pipelineErrors.map((entry) => ({ ...entry }));
    }

    /**
     * Clear the bounded pipeline creation error history ring while preserving cumulative error
     * counters and monotonic sequence numbers.
     */
    clearPipelineErrors() {
      this._pipelineErrors.length = 0;
    }

    /**
     * Return a cached group(0) `GPUBindGroupLayout` matching a G34 fallback profile
     * (`fullscreen_triangle_cbv`, `textured_quad_24b`, or `default`).
     *
     * @param {string | null | undefined} profile
     * @returns {GPUBindGroupLayout}
     */
    getOrCreateFallbackBindGroupLayout(profile) {
      const canonical = normalizeFallbackProfile(profile);
      const descriptor = buildFallbackBindGroupLayoutDescriptor(canonical);
      return this.getOrCreateBindGroupLayout(descriptor, `fallback-bgl:${canonical}`);
    }

    /**
     * Return a cached `GPUPipelineLayout` compatible with a G34 fallback profile, or `'auto'`
     * when `device.createPipelineLayout` is unavailable or fails validation.
     *
     * @param {string | null | undefined} profile
     * @returns {GPUPipelineLayout | 'auto'}
     */
    getOrCreateFallbackPipelineLayout(profile) {
      const canonical = normalizeFallbackProfile(profile);
      if (typeof this._device.createPipelineLayout !== 'function') {
        return 'auto';
      }

      const cached = this._fallbackPipelineLayouts.get(canonical);
      if (cached !== undefined) {
        return cached;
      }

      const bindGroupLayouts =
        canonical === 'default' ? [] : [this.getOrCreateFallbackBindGroupLayout(canonical)];

      try {
        const layout = this._device.createPipelineLayout({
          label: `fallback-pl:${canonical}`,
          bindGroupLayouts,
        });
        if (layout && (typeof layout === 'object' || typeof layout === 'function')) {
          registeredObjectIds.set(layout, `fallback-pl:${canonical}`);
          fallbackPipelineLayoutProfiles.set(layout, canonical);
          this._fallbackPipelineLayouts.set(canonical, layout);
          return layout;
        }
        return 'auto';
      } catch (err) {
        this._recordPipelineError('pipeline-layout', err, {
          key: `fallback-pl:${canonical}`,
          fallbackProfile: canonical,
          recovered: true,
        });
        return 'auto';
      }
    }

    /**
     * Normalize a render pipeline descriptor using this cache's device vertex limits,
     * default color format, and optional explicit fallback `GPUPipelineLayout`.
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {object} [options]
     */
    normalizeRenderDescriptor(descriptor, options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const detectedProfile = detectFallbackProfile(descriptor, opts);
      let fallbackLayout = opts.fallbackLayout;
      if (
        detectedProfile !== null &&
        opts.layoutMode === 'explicit' &&
        (fallbackLayout === undefined || fallbackLayout === null)
      ) {
        fallbackLayout = this.getOrCreateFallbackPipelineLayout(detectedProfile);
      }

      const mergedOpts = {
        ...this._deviceVertexLimits(),
        defaultColorFormat: this._defaultColorFormat,
        ...opts,
      };
      if (fallbackLayout !== undefined) {
        mergedOpts.fallbackLayout = fallbackLayout;
      }
      return normalizeRenderPipelineDescriptor(descriptor, mergedOpts);
    }

    _resolveKey(descriptor, keyHint) {
      if (keyHint !== undefined && keyHint !== null) {
        if (typeof keyHint === 'string') {
          if (keyHint.length === 0) {
            throw new RangeError('keyHint string must not be empty');
          }
          return keyHint;
        }
        if (typeof keyHint === 'bigint') {
          return keyHint >= 0n && keyHint <= U64_MASK
            ? formatHashHex64(keyHint)
            : `bi:${keyHint.toString(10)}`;
        }
        if (typeof keyHint === 'number') {
          if (!Number.isFinite(keyHint)) {
            throw new RangeError(`keyHint number must be finite, got: ${keyHint}`);
          }
          return `num:${keyHint}`;
        }
        return hashDescriptorKey(keyHint);
      }
      return hashDescriptorKey(descriptor);
    }

    _getLru(map, key) {
      const entry = map.get(key);
      if (entry === undefined) {
        return undefined;
      }
      entry.lastAccessedAt = this._now();
      // Refresh LRU recency by reinserting at the tail of the Map.
      map.delete(key);
      map.set(key, entry);
      return entry.value;
    }

    _setLru(map, key, value, compileMs, maxSize) {
      const ts = this._now();
      if (map.has(key)) {
        map.delete(key);
      }
      map.set(key, {
        value,
        createdAt: ts,
        lastAccessedAt: ts,
        compileMs,
      });
      while (map.size > maxSize) {
        const oldestKey = map.keys().next().value;
        map.delete(oldestKey);
        this._stats.evictions += 1;
      }
    }

    /**
     * Return a cached `GPUBindGroupLayout` or create and cache it via `device.createBindGroupLayout`.
     *
     * @param {GPUBindGroupLayoutDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @returns {GPUBindGroupLayout}
     */
    getOrCreateBindGroupLayout(descriptor, keyHint) {
      if (!descriptor || typeof descriptor !== 'object') {
        throw new TypeError('getOrCreateBindGroupLayout requires a descriptor object');
      }
      if (typeof this._device.createBindGroupLayout !== 'function') {
        throw new TypeError('device must provide a createBindGroupLayout(descriptor) method');
      }

      const key = this._resolveKey(descriptor, keyHint);
      const cached = this._getLru(this._bindGroupLayouts, key);
      if (cached !== undefined) {
        this._stats.layoutHits += 1;
        return cached;
      }

      this._stats.layoutMisses += 1;
      const t0 = this._now();
      let layout;
      try {
        layout = this._device.createBindGroupLayout(descriptor);
      } catch (err) {
        this._recordPipelineError('bind-group-layout', err, { key });
        throw err;
      }
      const compileMs = Math.max(0, this._now() - t0);
      this._stats.totalCompileMs += compileMs;

      if (layout && (typeof layout === 'object' || typeof layout === 'function')) {
        registeredObjectIds.set(layout, `bgl:${key}`);
      }
      this._setLru(this._bindGroupLayouts, key, layout, compileMs, this._maxBindGroupLayouts);
      return layout;
    }

    /**
     * Build and cache a `GPUBindGroupLayout` from compact S6 shader reflection JSON.
     * Accepts either `{ entries | bindings, label?, group? }` or an array of binding objects
     * `{ binding, visibility, buffer?, sampler?, texture?, storageTexture? }`.
     *
     * @param {object | Array<object>} reflectionGroup
     * @param {string | bigint | number | object} [keyHint]
     * @returns {GPUBindGroupLayout}
     */
    createBindGroupLayoutFromReflection(reflectionGroup, keyHint) {
      let rawEntries;
      let label;

      if (Array.isArray(reflectionGroup)) {
        rawEntries = reflectionGroup;
      } else if (reflectionGroup && typeof reflectionGroup === 'object') {
        if (Array.isArray(reflectionGroup.entries)) {
          rawEntries = reflectionGroup.entries;
        } else if (Array.isArray(reflectionGroup.bindings)) {
          rawEntries = reflectionGroup.bindings;
        } else {
          throw new TypeError(
            'reflectionGroup must be an array or an object with an "entries" or "bindings" array'
          );
        }
        if (typeof reflectionGroup.label === 'string' && reflectionGroup.label.length > 0) {
          label = reflectionGroup.label;
        }
      } else {
        throw new TypeError('reflectionGroup must be an array or object');
      }

      const seenBindings = new Set();
      const entries = new Array(rawEntries.length);
      for (let i = 0; i < rawEntries.length; i += 1) {
        const normalized = normalizeReflectionBinding(rawEntries[i], i);
        if (seenBindings.has(normalized.binding)) {
          throw new Error(`Duplicate reflection binding index ${normalized.binding}`);
        }
        seenBindings.add(normalized.binding);
        entries[i] = normalized;
      }

      // Sort by binding index so semantically identical reflection groups share one cache key.
      entries.sort((a, b) => a.binding - b.binding);

      const descriptor = label !== undefined ? { label, entries } : { entries };
      return this.getOrCreateBindGroupLayout(descriptor, keyHint);
    }

    /**
     * Synchronously return a cached `GPURenderPipeline` or compile via `device.createRenderPipeline`.
     * Automatically normalizes missing layouts, malformed vertex buffers, and G34 fallback profiles
     * while leaving already-valid non-fallback descriptors untouched.
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @param {object} [options]
     * @returns {GPURenderPipeline}
     */
    getOrCreateRenderPipeline(descriptor, keyHint, options) {
      if (!descriptor || typeof descriptor !== 'object') {
        throw new TypeError('getOrCreateRenderPipeline requires a descriptor object');
      }
      if (typeof this._device.createRenderPipeline !== 'function') {
        throw new TypeError('device must provide a createRenderPipeline(descriptor) method');
      }

      const opts = options && typeof options === 'object' ? options : null;
      const norm = this.normalizeRenderDescriptor(descriptor, opts || undefined);
      const effectiveDescriptor = norm.descriptor;

      const key = this._resolveKey(effectiveDescriptor, keyHint);
      const cached = this._getLru(this._renderPipelines, key);
      if (cached !== undefined) {
        this._stats.renderHits += 1;
        return cached;
      }

      this._stats.renderMisses += 1;
      this._recordNormalizationStats(norm);

      const canRetryAutoLayout =
        effectiveDescriptor.layout !== 'auto' &&
        Boolean((opts && opts.retryWithAutoLayout === true) || norm.fallbackProfile !== null);

      const t0 = this._now();
      let pipeline;
      try {
        pipeline = this._device.createRenderPipeline(effectiveDescriptor);
      } catch (err) {
        if (canRetryAutoLayout) {
          const retryDescriptor = {
            ...effectiveDescriptor,
            layout: 'auto',
          };
          try {
            pipeline = this._device.createRenderPipeline(retryDescriptor);
            this._recordPipelineError('render', err, {
              key,
              fallbackProfile: norm.fallbackProfile,
              recovered: true,
            });
            this._stats.fallbackRecoveryCount += 1;
            if (!norm.layoutNormalized) {
              this._stats.layoutNormalizedCount += 1;
            }
            if (!norm.normalized) {
              this._stats.fallbackNormalizedCount += 1;
            }
          } catch (retryErr) {
            this._recordPipelineError('render', err, {
              key,
              fallbackProfile: norm.fallbackProfile,
              recovered: false,
            });
            this._recordPipelineError('render', retryErr, {
              key,
              fallbackProfile: norm.fallbackProfile,
              recovered: false,
            });
            throw retryErr;
          }
        } else {
          this._recordPipelineError('render', err, {
            key,
            fallbackProfile: norm.fallbackProfile,
            recovered: false,
          });
          throw err;
        }
      }

      const compileMs = Math.max(0, this._now() - t0);
      this._stats.totalCompileMs += compileMs;

      this._setLru(this._renderPipelines, key, pipeline, compileMs, this._maxRenderPipelines);
      return pipeline;
    }

    /**
     * Asynchronously return a cached `GPURenderPipeline`, deduplicating concurrent in-flight
     * compilations for the same key. Uses `device.createRenderPipelineAsync` when available,
     * falling back to synchronous `device.createRenderPipeline`.
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @param {object} [options]
     * @returns {Promise<GPURenderPipeline>}
     */
    getOrCreateRenderPipelineAsync(descriptor, keyHint, options) {
      if (!descriptor || typeof descriptor !== 'object') {
        return Promise.reject(
          new TypeError('getOrCreateRenderPipelineAsync requires a descriptor object')
        );
      }

      const opts = options && typeof options === 'object' ? options : null;
      let norm;
      let key;
      try {
        norm = this.normalizeRenderDescriptor(descriptor, opts || undefined);
        key = this._resolveKey(norm.descriptor, keyHint);
      } catch (err) {
        return Promise.reject(err);
      }

      const effectiveDescriptor = norm.descriptor;
      const cached = this._getLru(this._renderPipelines, key);
      if (cached !== undefined) {
        this._stats.renderHits += 1;
        return Promise.resolve(cached);
      }

      const existing = this._inFlightRender.get(key);
      if (existing !== undefined) {
        this._stats.inFlightDedupes += 1;
        return existing;
      }

      const hasAsync = typeof this._device.createRenderPipelineAsync === 'function';
      const hasSync = typeof this._device.createRenderPipeline === 'function';
      if (!hasAsync && !hasSync) {
        return Promise.reject(
          new TypeError('device must provide createRenderPipelineAsync or createRenderPipeline')
        );
      }

      this._stats.renderMisses += 1;
      this._recordNormalizationStats(norm);

      const canRetryAutoLayout =
        effectiveDescriptor.layout !== 'auto' &&
        Boolean((opts && opts.retryWithAutoLayout === true) || norm.fallbackProfile !== null);

      const t0 = this._now();

      if (!hasAsync) {
        try {
          let pipeline;
          try {
            pipeline = this._device.createRenderPipeline(effectiveDescriptor);
          } catch (err) {
            if (!canRetryAutoLayout) {
              this._recordPipelineError('render', err, {
                key,
                fallbackProfile: norm.fallbackProfile,
                recovered: false,
              });
              throw err;
            }
            const retryDescriptor = { ...effectiveDescriptor, layout: 'auto' };
            try {
              pipeline = this._device.createRenderPipeline(retryDescriptor);
              this._recordPipelineError('render', err, {
                key,
                fallbackProfile: norm.fallbackProfile,
                recovered: true,
              });
              this._stats.fallbackRecoveryCount += 1;
              if (!norm.layoutNormalized) {
                this._stats.layoutNormalizedCount += 1;
              }
              if (!norm.normalized) {
                this._stats.fallbackNormalizedCount += 1;
              }
            } catch (retryErr) {
              this._recordPipelineError('render', err, {
                key,
                fallbackProfile: norm.fallbackProfile,
                recovered: false,
              });
              this._recordPipelineError('render', retryErr, {
                key,
                fallbackProfile: norm.fallbackProfile,
                recovered: false,
              });
              throw retryErr;
            }
          }
          const compileMs = Math.max(0, this._now() - t0);
          this._stats.totalCompileMs += compileMs;
          this._setLru(this._renderPipelines, key, pipeline, compileMs, this._maxRenderPipelines);
          return Promise.resolve(pipeline);
        } catch (err) {
          return Promise.reject(err);
        }
      }

      const generation = this._generation;
      const invokeCreateAsync = (desc) => {
        try {
          return Promise.resolve(this._device.createRenderPipelineAsync(desc));
        } catch (syncErr) {
          return Promise.reject(syncErr);
        }
      };

      const finishSuccess = (pipeline) => {
        if (this._inFlightRender.get(key) === trackedPromise) {
          this._inFlightRender.delete(key);
        }
        const compileMs = Math.max(0, this._now() - t0);
        this._stats.totalCompileMs += compileMs;
        if (generation === this._generation) {
          this._setLru(this._renderPipelines, key, pipeline, compileMs, this._maxRenderPipelines);
        }
        return pipeline;
      };

      const trackedPromise = invokeCreateAsync(effectiveDescriptor).then(
        finishSuccess,
        (err) => {
          if (canRetryAutoLayout) {
            const retryDescriptor = { ...effectiveDescriptor, layout: 'auto' };
            return invokeCreateAsync(retryDescriptor).then(
              (recoveredPipeline) => {
                this._recordPipelineError('render', err, {
                  key,
                  fallbackProfile: norm.fallbackProfile,
                  recovered: true,
                });
                this._stats.fallbackRecoveryCount += 1;
                if (!norm.layoutNormalized) {
                  this._stats.layoutNormalizedCount += 1;
                }
                if (!norm.normalized) {
                  this._stats.fallbackNormalizedCount += 1;
                }
                return finishSuccess(recoveredPipeline);
              },
              (retryErr) => {
                if (this._inFlightRender.get(key) === trackedPromise) {
                  this._inFlightRender.delete(key);
                }
                this._recordPipelineError('render', err, {
                  key,
                  fallbackProfile: norm.fallbackProfile,
                  recovered: false,
                });
                this._recordPipelineError('render', retryErr, {
                  key,
                  fallbackProfile: norm.fallbackProfile,
                  recovered: false,
                });
                throw retryErr;
              }
            );
          }

          if (this._inFlightRender.get(key) === trackedPromise) {
            this._inFlightRender.delete(key);
          }
          this._recordPipelineError('render', err, {
            key,
            fallbackProfile: norm.fallbackProfile,
            recovered: false,
          });
          throw err;
        }
      );

      this._inFlightRender.set(key, trackedPromise);
      return trackedPromise;
    }

    /**
     * Convenience wrapper to synchronously create/fetch a fallback-safe render pipeline for a
     * G34 profile (`fullscreen_triangle_cbv`, `textured_quad_24b`, or `default`) with automatic
     * vertex/CBV layout normalization and auto-layout retry recovery.
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {object} [options]
     * @returns {GPURenderPipeline}
     */
    getOrCreateFallbackRenderPipeline(descriptor, options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const detected = detectFallbackProfile(descriptor, opts);
      const fallbackOpts = {
        ...opts,
        fallbackProfile: detected || 'default',
        retryWithAutoLayout: opts.retryWithAutoLayout !== false,
      };
      return this.getOrCreateRenderPipeline(descriptor, opts.keyHint, fallbackOpts);
    }

    /**
     * Convenience wrapper to asynchronously create/fetch a fallback-safe render pipeline for a
     * G34 profile (`fullscreen_triangle_cbv`, `textured_quad_24b`, or `default`).
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {object} [options]
     * @returns {Promise<GPURenderPipeline>}
     */
    getOrCreateFallbackRenderPipelineAsync(descriptor, options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const detected = detectFallbackProfile(descriptor, opts);
      const fallbackOpts = {
        ...opts,
        fallbackProfile: detected || 'default',
        retryWithAutoLayout: opts.retryWithAutoLayout !== false,
      };
      return this.getOrCreateRenderPipelineAsync(descriptor, opts.keyHint, fallbackOpts);
    }

    /**
     * Synchronously return a cached `GPUComputePipeline` or compile via `device.createComputePipeline`.
     *
     * @param {GPUComputePipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @returns {GPUComputePipeline}
     */
    getOrCreateComputePipeline(descriptor, keyHint) {
      if (!descriptor || typeof descriptor !== 'object') {
        throw new TypeError('getOrCreateComputePipeline requires a descriptor object');
      }
      if (typeof this._device.createComputePipeline !== 'function') {
        throw new TypeError('device must provide a createComputePipeline(descriptor) method');
      }

      const norm = normalizeComputePipelineDescriptor(descriptor);
      const effectiveDescriptor = norm.descriptor;

      const key = this._resolveKey(effectiveDescriptor, keyHint);
      const cached = this._getLru(this._computePipelines, key);
      if (cached !== undefined) {
        this._stats.computeHits += 1;
        return cached;
      }

      this._stats.computeMisses += 1;
      this._recordNormalizationStats(norm);

      const t0 = this._now();
      let pipeline;
      try {
        pipeline = this._device.createComputePipeline(effectiveDescriptor);
      } catch (err) {
        this._recordPipelineError('compute', err, { key });
        throw err;
      }
      const compileMs = Math.max(0, this._now() - t0);
      this._stats.totalCompileMs += compileMs;

      this._setLru(this._computePipelines, key, pipeline, compileMs, this._maxComputePipelines);
      return pipeline;
    }

    /**
     * Asynchronously return a cached `GPUComputePipeline`, deduplicating concurrent in-flight
     * compilations for the same key. Uses `device.createComputePipelineAsync` when available,
     * falling back to synchronous `device.createComputePipeline`.
     *
     * @param {GPUComputePipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @returns {Promise<GPUComputePipeline>}
     */
    getOrCreateComputePipelineAsync(descriptor, keyHint) {
      if (!descriptor || typeof descriptor !== 'object') {
        return Promise.reject(
          new TypeError('getOrCreateComputePipelineAsync requires a descriptor object')
        );
      }

      let norm;
      let key;
      try {
        norm = normalizeComputePipelineDescriptor(descriptor);
        key = this._resolveKey(norm.descriptor, keyHint);
      } catch (err) {
        return Promise.reject(err);
      }

      const effectiveDescriptor = norm.descriptor;
      const cached = this._getLru(this._computePipelines, key);
      if (cached !== undefined) {
        this._stats.computeHits += 1;
        return Promise.resolve(cached);
      }

      const existing = this._inFlightCompute.get(key);
      if (existing !== undefined) {
        this._stats.inFlightDedupes += 1;
        return existing;
      }

      const hasAsync = typeof this._device.createComputePipelineAsync === 'function';
      const hasSync = typeof this._device.createComputePipeline === 'function';
      if (!hasAsync && !hasSync) {
        return Promise.reject(
          new TypeError('device must provide createComputePipelineAsync or createComputePipeline')
        );
      }

      this._stats.computeMisses += 1;
      this._recordNormalizationStats(norm);
      const t0 = this._now();

      if (!hasAsync) {
        try {
          const pipeline = this._device.createComputePipeline(effectiveDescriptor);
          const compileMs = Math.max(0, this._now() - t0);
          this._stats.totalCompileMs += compileMs;
          this._setLru(this._computePipelines, key, pipeline, compileMs, this._maxComputePipelines);
          return Promise.resolve(pipeline);
        } catch (err) {
          this._recordPipelineError('compute', err, { key });
          return Promise.reject(err);
        }
      }

      const generation = this._generation;
      let rawPromise;
      try {
        rawPromise = Promise.resolve(this._device.createComputePipelineAsync(effectiveDescriptor));
      } catch (err) {
        this._recordPipelineError('compute', err, { key });
        return Promise.reject(err);
      }

      const trackedPromise = rawPromise.then(
        (pipeline) => {
          if (this._inFlightCompute.get(key) === trackedPromise) {
            this._inFlightCompute.delete(key);
          }
          const compileMs = Math.max(0, this._now() - t0);
          this._stats.totalCompileMs += compileMs;
          if (generation === this._generation) {
            this._setLru(this._computePipelines, key, pipeline, compileMs, this._maxComputePipelines);
          }
          return pipeline;
        },
        (err) => {
          if (this._inFlightCompute.get(key) === trackedPromise) {
            this._inFlightCompute.delete(key);
          }
          this._recordPipelineError('compute', err, { key });
          throw err;
        }
      );

      this._inFlightCompute.set(key, trackedPromise);
      return trackedPromise;
    }

    async _prewarmPool(descriptors, options, cacheMap, inFlightMap, getterAsync, isRender) {
      const opts = options || {};
      const concurrency = validatePositiveInt(
        opts.concurrency !== undefined ? opts.concurrency : DEFAULT_PREWARM_CONCURRENCY,
        'concurrency'
      );

      const list = Array.from(descriptors || []);
      const pipelines = new Array(list.length);
      let cursor = 0;
      let compiled = 0;
      let cached = 0;
      let deduped = 0;

      const worker = async () => {
        while (cursor < list.length) {
          const idx = cursor++;
          const item = list[idx];
          const desc = isDescriptorWrapper(item) ? item.descriptor : item;
          const keyHint = isDescriptorWrapper(item) ? item.keyHint : undefined;
          const itemOpts = isDescriptorWrapper(item) ? item.options : undefined;
          const effectiveDesc = isRender
            ? this.normalizeRenderDescriptor(desc, itemOpts).descriptor
            : normalizeComputePipelineDescriptor(desc).descriptor;
          const key = this._resolveKey(effectiveDesc, keyHint);
          const wasCached = cacheMap.has(key);
          const wasInFlight = !wasCached && inFlightMap.has(key);
          const pipeline = isRender
            ? await getterAsync.call(this, desc, key, itemOpts)
            : await getterAsync.call(this, desc, key);
          pipelines[idx] = pipeline;
          if (wasCached) {
            cached += 1;
          } else if (wasInFlight) {
            deduped += 1;
          } else {
            compiled += 1;
          }
        }
      };

      const workerCount = Math.min(concurrency, list.length);
      const workers = [];
      for (let i = 0; i < workerCount; i += 1) {
        workers.push(worker());
      }
      await Promise.all(workers);

      return {
        total: list.length,
        compiled,
        cached,
        deduped,
        pipelines,
      };
    }

    /**
     * Batch-prewarm an iterable of render pipeline descriptors with bounded concurrency.
     *
     * @param {Iterable<GPURenderPipelineDescriptor | { descriptor: GPURenderPipelineDescriptor, keyHint?: string | bigint | number | object, options?: object }>} descriptors
     * @param {{ concurrency?: number }} [options]
     * @returns {Promise<{ total: number, compiled: number, cached: number, deduped: number, pipelines: Array<GPURenderPipeline> }>}
     */
    prewarmRenderPipelines(descriptors, options = {}) {
      return this._prewarmPool(
        descriptors,
        options,
        this._renderPipelines,
        this._inFlightRender,
        this.getOrCreateRenderPipelineAsync,
        true
      );
    }

    /**
     * Batch-prewarm an iterable of compute pipeline descriptors with bounded concurrency.
     *
     * @param {Iterable<GPUComputePipelineDescriptor | { descriptor: GPUComputePipelineDescriptor, keyHint?: string | bigint | number | object }>} descriptors
     * @param {{ concurrency?: number }} [options]
     * @returns {Promise<{ total: number, compiled: number, cached: number, deduped: number, pipelines: Array<GPUComputePipeline> }>}
     */
    prewarmComputePipelines(descriptors, options = {}) {
      return this._prewarmPool(
        descriptors,
        options,
        this._computePipelines,
        this._inFlightCompute,
        this.getOrCreateComputePipelineAsync,
        false
      );
    }

    /**
     * Evict cached render pipelines, compute pipelines, and bind group layouts
     * whose last access timestamp is strictly older than `now() - ageMs`.
     *
     * @param {number} ageMs
     * @returns {number} Total number of evicted cache entries
     */
    evictOlderThan(ageMs) {
      if (typeof ageMs !== 'number' || !Number.isFinite(ageMs) || ageMs < 0) {
        throw new RangeError(`ageMs must be a finite non-negative number, got: ${ageMs}`);
      }
      const cutoff = this._now() - ageMs;
      let removed = 0;

      const prune = (map) => {
        for (const [key, entry] of map.entries()) {
          if (entry.lastAccessedAt < cutoff) {
            map.delete(key);
            removed += 1;
          }
        }
      };

      prune(this._renderPipelines);
      prune(this._computePipelines);
      prune(this._bindGroupLayouts);

      this._stats.evictions += removed;
      return removed;
    }

    /**
     * Clear all cached render pipelines, compute pipelines, bind group layouts,
     * and detach any in-flight compilations so stale results cannot repopulate the cache.
     *
     * @param {{ resetStats?: boolean }} [options]
     */
    clear(options = {}) {
      this._generation += 1;
      this._renderPipelines.clear();
      this._computePipelines.clear();
      this._bindGroupLayouts.clear();
      this._fallbackPipelineLayouts.clear();
      this._inFlightRender.clear();
      this._inFlightCompute.clear();

      if (options && options.resetStats) {
        this._stats.renderHits = 0;
        this._stats.renderMisses = 0;
        this._stats.computeHits = 0;
        this._stats.computeMisses = 0;
        this._stats.layoutHits = 0;
        this._stats.layoutMisses = 0;
        this._stats.inFlightDedupes = 0;
        this._stats.evictions = 0;
        this._stats.totalCompileMs = 0;
        this._stats.fallbackNormalizedCount = 0;
        this._stats.vertexLayoutNormalizedCount = 0;
        this._stats.layoutNormalizedCount = 0;
        this._stats.fallbackRecoveryCount = 0;
        this._stats.pipelineErrorCount = 0;
        this._pipelineErrors.length = 0;
        this._pipelineErrorSeq = 0;
      }
    }

    /**
     * Return a snapshot of cache hit/miss, deduplication, eviction, count, normalization,
     * error tracking, and compile-time telemetry.
     */
    getStats() {
      return {
        renderHits: this._stats.renderHits,
        renderMisses: this._stats.renderMisses,
        computeHits: this._stats.computeHits,
        computeMisses: this._stats.computeMisses,
        layoutHits: this._stats.layoutHits,
        layoutMisses: this._stats.layoutMisses,
        inFlightDedupes: this._stats.inFlightDedupes,
        evictions: this._stats.evictions,
        renderCount: this._renderPipelines.size,
        computeCount: this._computePipelines.size,
        layoutCount: this._bindGroupLayouts.size,
        totalCompileMs: this._stats.totalCompileMs,
        fallbackNormalizedCount: this._stats.fallbackNormalizedCount,
        vertexLayoutNormalizedCount: this._stats.vertexLayoutNormalizedCount,
        layoutNormalizedCount: this._stats.layoutNormalizedCount,
        fallbackRecoveryCount: this._stats.fallbackRecoveryCount,
        pipelineErrorCount: this._stats.pipelineErrorCount,
        pipelineErrorHistorySize: this._pipelineErrors.length,
        maxErrorHistory: this._maxErrorHistory,
      };
    }
  }

  const exportedApi = {
    DEFAULT_MAX_RENDER_PIPELINES,
    DEFAULT_MAX_COMPUTE_PIPELINES,
    DEFAULT_MAX_BIND_GROUP_LAYOUTS,
    DEFAULT_PREWARM_CONCURRENCY,
    DEFAULT_MAX_ERROR_HISTORY,
    MAX_PIPELINE_ERROR_MESSAGE_LENGTH,
    DEFAULT_MAX_VERTEX_BUFFERS,
    DEFAULT_MAX_VERTEX_ATTRIBUTES,
    DEFAULT_MAX_VERTEX_BUFFER_STRIDE,
    FRAME1_VERTEX_STRIDE_24B,
    FRAME1_CBV_MIN_BINDING_SIZE,
    SHADER_STAGE_BITS,
    VERTEX_FORMAT_BYTE_SIZES,
    FRAME1_VERTEX_LAYOUT_24B,
    FALLBACK_BIND_GROUP_ENTRIES,
    hashDescriptorKey,
    registerShaderModuleId,
    normalizeVisibility,
    normalizeFallbackProfile,
    getVertexFormatByteSize,
    buildVertexBufferLayoutForStride,
    normalizeVertexBufferLayouts,
    buildFallbackBindGroupLayoutDescriptor,
    normalizeRenderPipelineDescriptor,
    sanitizePipelineErrorMessage,
    WebGpuPipelineCache,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6PipelineCache = exportedApi;
  } else if (root) {
    root.PlayGta6PipelineCache = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
