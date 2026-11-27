'use strict';

// S6 shader reflection manifest loader and WebGPU BindGroupLayout / PipelineLayout /
// VertexBufferLayout builder.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6ReflectionLoader)
// and Node.js (module.exports). Matches the exact JSON schema emitted by shaders/src/reflect.rs
// and integrates directly with WebGpuPipelineCache (web/workers/pipeline_cache.js).
(function (root) {
  /** Supported S6 layout-manifest.json schema_version (shaders/src/reflect.rs). */
  const LAYOUT_MANIFEST_SCHEMA_VERSION = 1;

  /** Supported stage-reflection sidecar (`*.reflect.json`) schema & versions (runtime/gpu/stage_reflection.h). */
  const STAGE_REFLECTION_SCHEMA = 'playgta6-stage-reflection';
  const STAGE_REFLECTION_VERSION = 1;
  const STAGE_REFLECTION_MIN_VERSION = 1;
  const STAGE_REFLECTION_VERSION_ROOT_TABLE_OFFSET = 2;
  const STAGE_REFLECTION_VERSION_UNRESOLVED = 3;
  const STAGE_REFLECTION_MAX_VERSION = 3;

  /** WebGPU baseline maxBindGroups (groups 0..3). */
  const MAX_WEBGPU_BIND_GROUPS = 4;

  /** Reflection sanity ceiling matching MAX_REFLECTED_GROUPS in shaders/src/reflect.rs. */
  const MAX_REFLECTED_GROUPS = 64;

  /** Maximum valid bit-shift for S34 dynamic descriptor-heap index calculations (0..31). */
  const MAX_HEAP_INDEX_SHIFT = 31;

  /** Largest lane-indexed heap window a sidecar may request (runtime/gpu/stage_reflection.h). */
  const MAX_LANE_INDEXED_WINDOW = 256;

  /** Mesh output layout bounds matching runtime/gpu/state_bridge.h (G66). */
  const MAX_MESH_VERTICES = 256;
  const MAX_MESH_PRIMITIVES = 256;
  const MAX_MESH_ATTRIBUTES = 15;

  /** Runtime-owned RT arena bind group & binding slots (group 3, bindings 16..21 -> mask 0x3f). */
  const RT_ARENA_BIND_GROUP = 3;
  const RT_ARENA_FIRST_BINDING = 16;
  const RT_ARENA_LAST_BINDING = 21;
  const RT_ARENA_FULL_MASK = 0x3f;

  /** Exact GFX10 S# dword count for #950 / S43 inline sampler sources. */
  const INLINE_SAMPLER_SDWORD_COUNT = 4;

  /** Console ABI user-data SGPR bases & descriptor-table slot size (runtime/gpu/stage_reflection.h). */
  const GRAPHICS_USER_DATA_BASE = 8;
  const COMPUTE_USER_DATA_BASE = 4;
  const MESH_USER_DATA_BASE = 16;
  const TABLE_SLOT_BYTES = 32;
  const UNBOUNDED_DESCRIPTOR_COUNT = 0xffffffff;
  const DESCRIPTOR_RANGE_OFFSET_APPEND = 0xffffffff;

  /** WebGPU / WGSL buffer min_binding_size_bytes alignments (bytes). */
  const UNIFORM_MIN_BINDING_SIZE_ALIGNMENT = 16;
  const STORAGE_MIN_BINDING_SIZE_ALIGNMENT = 4;

  /** Structured mismatch diagnostic codes emitted by diagnoseStageBindings(). */
  const DIAGNOSTIC_CODES = Object.freeze({
    MISSING_SOURCE: 'MISSING_SOURCE',
    INVALID_SOURCE: 'INVALID_SOURCE',
    UNKNOWN_BINDING_KIND: 'UNKNOWN_BINDING_KIND',
    ROOT_PARAMETER_NOT_FOUND: 'ROOT_PARAMETER_NOT_FOUND',
    ROOT_PARAMETER_TYPE_MISMATCH: 'ROOT_PARAMETER_TYPE_MISMATCH',
    TABLE_OFFSET_UNALIGNED: 'TABLE_OFFSET_UNALIGNED',
    TABLE_DESCRIPTOR_MISSING: 'TABLE_DESCRIPTOR_MISSING',
    DESCRIPTOR_TYPE_MISMATCH: 'DESCRIPTOR_TYPE_MISMATCH',
    STATIC_SAMPLER_MISSING: 'STATIC_SAMPLER_MISSING',
    HEAP_INDEX_NOT_ROOT_CONSTANT: 'HEAP_INDEX_NOT_ROOT_CONSTANT',
    HEAP_KIND_MISMATCH: 'HEAP_KIND_MISMATCH',
    MISSING_STORAGE_FORMAT: 'MISSING_STORAGE_FORMAT',
    MIN_BINDING_SIZE_MISALIGNED: 'MIN_BINDING_SIZE_MISALIGNED',
    MIN_BINDING_SIZE_UNEXPECTED: 'MIN_BINDING_SIZE_UNEXPECTED',
    UNRESOLVED_WRITABLE_BINDING: 'UNRESOLVED_WRITABLE_BINDING',
    BINDING_ARRAYS_UNSUPPORTED: 'BINDING_ARRAYS_UNSUPPORTED',
    RT_ARENA_GROUP_CONFLICT: 'RT_ARENA_GROUP_CONFLICT',
  });

  /** WebGPU GPUShaderStage bitmask constants with headless/Node fallbacks. */
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

  const VALID_VISIBILITY_MASK =
    SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT | SHADER_STAGE_BITS.COMPUTE;

  const STAGE_TO_VISIBILITY = Object.freeze({
    vertex: SHADER_STAGE_BITS.VERTEX,
    fragment: SHADER_STAGE_BITS.FRAGMENT,
    compute: SHADER_STAGE_BITS.COMPUTE,
  });

  const STAGE_ORDER = Object.freeze({
    vertex: 0,
    fragment: 1,
    compute: 2,
  });

  const VALID_BUFFER_TYPES = new Set(['uniform', 'storage', 'read-only-storage']);
  const VALID_SAMPLER_TYPES = new Set(['filtering', 'non-filtering', 'comparison']);
  const VALID_TEXTURE_SAMPLE_TYPES = new Set([
    'float',
    'unfilterable-float',
    'depth',
    'sint',
    'uint',
  ]);
  const VALID_TEXTURE_VIEW_DIMENSIONS = new Set([
    '1d',
    '2d',
    '2d-array',
    'cube',
    'cube-array',
    '3d',
  ]);
  const VALID_STORAGE_TEXTURE_ACCESS = new Set(['write-only', 'read-only', 'read-write']);
  const VALID_STORAGE_TEXTURE_VIEW_DIMENSIONS = new Set(['1d', '2d', '2d-array', '3d']);
  const VALID_RESOURCE_KINDS = new Set([
    'buffer',
    'sampler',
    'texture',
    'storageTexture',
    'externalTexture',
    'accelerationStructure',
  ]);
  const VALID_REGISTER_CLASSES = new Set(['b', 't', 'u', 's']);
  const VALID_STEP_MODES = new Set(['vertex', 'instance']);

  /** Shared frozen empty array reused across compacted stage records to avoid per-stage allocations. */
  const EMPTY_ARRAY = Object.freeze([]);

  /** Canonical interned string tables for repeated reflection vocabulary tokens. */
  const CANONICAL_STAGE_TOKENS = Object.freeze({
    vertex: 'vertex',
    fragment: 'fragment',
    pixel: 'fragment',
    compute: 'compute',
  });
  const CANONICAL_VIEW_DIMENSIONS = Object.freeze({
    '1d': '1d',
    '2d': '2d',
    '2d-array': '2d-array',
    cube: 'cube',
    'cube-array': 'cube-array',
    '3d': '3d',
  });
  const CANONICAL_SAMPLE_TYPES = Object.freeze({
    float: 'float',
    'unfilterable-float': 'unfilterable-float',
    depth: 'depth',
    sint: 'sint',
    uint: 'uint',
  });
  const CANONICAL_RESOURCE_KINDS = Object.freeze({
    buffer: 'buffer',
    sampler: 'sampler',
    texture: 'texture',
    storageTexture: 'storageTexture',
    externalTexture: 'externalTexture',
    accelerationStructure: 'accelerationStructure',
  });
  const CANONICAL_VERTEX_INPUT_MODES = Object.freeze({
    vertex_index: 'vertex_index',
    attributes: 'attributes',
  });
  const CANONICAL_ENTRY_POINTS = Object.freeze({
    vs_main: 'vs_main',
    fs_main: 'fs_main',
    cs_main: 'cs_main',
  });

  const COMPACT_STRING_POOL_LIMIT = 4096;

  /** Default bounded negative-miss cache size for unknown shader/stage hashes (W318). */
  const DEFAULT_NEGATIVE_LOOKUP_CACHE_SIZE = 2048;

  /** Maximum length of unknown string queries cached in the bounded negative-miss cache. */
  const MAX_NEGATIVE_STRING_KEY_LENGTH = 128;

  /** Complete WebGPU GPUVertexFormat byte sizes (covers all S6 io_type outputs + packed formats). */
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

  // 64-bit FNV-1a constants matching shaders/src/convert.rs & shaders/src/reflect.rs.
  const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
  const FNV1A_64_PRIME = 0x00000100000001b3n;
  const U64_MAX = 0xffffffffffffffffn;
  const MAX_SAFE_INTEGER_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
  const U32_MAX = 0xffffffff;

  const textEncoder = new TextEncoder();

  // ---------------------------------------------------------------------------
  // Typed Errors
  // ---------------------------------------------------------------------------

  class ReflectionError extends Error {
    constructor(message, code = 'REFLECTION_ERROR', details = undefined) {
      super(message);
      this.name = 'ReflectionError';
      this.code = code;
      if (details !== undefined) {
        this.details = details;
      }
    }
  }

  class ManifestValidationError extends ReflectionError {
    constructor(message, path = undefined) {
      super(path ? `${path}: ${message}` : message, 'MANIFEST_VALIDATION_ERROR', path ? { path } : undefined);
      this.name = 'ManifestValidationError';
      this.path = path || null;
    }
  }

  class BindingConflictError extends ReflectionError {
    constructor(group, binding, first, second, reason) {
      const firstLabel = describeBindingSide(first);
      const secondLabel = describeBindingSide(second);
      const detail = reason ? ` (${reason})` : '';
      super(
        `@group(${group}) @binding(${binding}) has incompatible layouts${detail}: ${firstLabel} vs ${secondLabel}`,
        'BINDING_CONFLICT',
        { group, binding, first, second, reason: reason || null }
      );
      this.name = 'BindingConflictError';
      this.group = group;
      this.binding = binding;
      this.first = first;
      this.second = second;
      this.reason = reason || null;
    }
  }

  class ShaderNotFoundError extends ReflectionError {
    constructor(query) {
      const display = typeof query === 'bigint' ? formatShaderHash(query) : String(query);
      super(`Shader reflection not found: ${display}`, 'SHADER_NOT_FOUND', { query });
      this.name = 'ShaderNotFoundError';
      this.query = query;
    }
  }

  class VertexLayoutError extends ReflectionError {
    constructor(message, details = undefined) {
      super(message, 'VERTEX_LAYOUT_ERROR', details);
      this.name = 'VertexLayoutError';
    }
  }

  function describeBindingSide(side) {
    if (!side) return '<unknown>';
    const stage = side.stage || 'unknown';
    const ep = side.entryPoint || '<unnamed>';
    const name = side.name || '<unnamed>';
    const layout = side.layoutJson || '<unknown>';
    return `${stage} '${ep}' (${name}) {${layout}}`;
  }

  // ---------------------------------------------------------------------------
  // Hash & Canonical Serialization Helpers (matching shaders/src/reflect.rs)
  // ---------------------------------------------------------------------------

  function isPlainObject(val) {
    return Boolean(val) && typeof val === 'object' && !Array.isArray(val);
  }

  function isNonNegativeInt(val) {
    return typeof val === 'number' && Number.isSafeInteger(val) && val >= 0 && val <= U32_MAX;
  }

  function hasOwn(obj, key) {
    return Object.prototype.hasOwnProperty.call(obj, key);
  }

  /**
   * Detach a string from any large parent string slice (such as a 16.3 MB JSON text buffer)
   * so V8 can garbage-collect the raw sidecar payload immediately after indexing.
   *
   * @param {string} str
   * @returns {string}
   */
  function detachString(str) {
    if (typeof str !== 'string' || str.length < 13) {
      return str;
    }
    const len = str.length;
    if (len <= 4096) {
      const codes = new Array(len);
      for (let i = 0; i < len; i += 1) {
        codes[i] = str.charCodeAt(i);
      }
      return String.fromCharCode.apply(null, codes);
    }
    let out = '';
    for (let i = 0; i < len; i += 4096) {
      const end = i + 4096 < len ? i + 4096 : len;
      const chunk = new Array(end - i);
      for (let j = i; j < end; j += 1) {
        chunk[j - i] = str.charCodeAt(j);
      }
      out += String.fromCharCode.apply(null, chunk);
    }
    return out;
  }

  /**
   * Create a parse-local string interner that maps repeated tokens (storage formats, entry points,
   * vertex formats, binding names, etc.) to a single detached flat string instance and is discarded
   * once sidecar/manifest indexing completes.
   *
   * @param {number} [limit]
   * @returns {(str: string | null | undefined) => string | null}
   */
  function createCompactStringInterner(limit = COMPACT_STRING_POOL_LIMIT) {
    const pool = new Map();
    return function internString(value) {
      if (value === undefined || value === null) {
        return null;
      }
      if (hasOwn(CANONICAL_ENTRY_POINTS, value)) return CANONICAL_ENTRY_POINTS[value];
      if (hasOwn(CANONICAL_STAGE_TOKENS, value)) return CANONICAL_STAGE_TOKENS[value];
      if (hasOwn(CANONICAL_VIEW_DIMENSIONS, value)) return CANONICAL_VIEW_DIMENSIONS[value];
      if (hasOwn(CANONICAL_SAMPLE_TYPES, value)) return CANONICAL_SAMPLE_TYPES[value];
      if (hasOwn(CANONICAL_RESOURCE_KINDS, value)) return CANONICAL_RESOURCE_KINDS[value];
      if (hasOwn(CANONICAL_VERTEX_INPUT_MODES, value)) return CANONICAL_VERTEX_INPUT_MODES[value];

      const existing = pool.get(value);
      if (existing !== undefined) {
        return existing;
      }
      const detached = detachString(value);
      if (pool.size < limit) {
        pool.set(detached, detached);
      }
      return detached;
    };
  }

  /**
   * Sanitize a telemetry counter into a finite non-negative integer.
   *
   * @param {unknown} value
   * @returns {number}
   */
  function sanitizeNonNegativeInt(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return Math.floor(value);
  }

  /**
   * Insert a key into a bounded LRU negative-miss Set, evicting the oldest entry at capacity.
   *
   * @param {Set<any>} cache
   * @param {any} key
   * @param {number} maxSize
   */
  function rememberBoundedNegativeKey(cache, key, maxSize) {
    if (maxSize <= 0) {
      return;
    }
    if (cache.size >= maxSize) {
      const oldest = cache.values().next().value;
      cache.delete(oldest);
    }
    cache.add(key);
  }

  /**
   * Zero-allocation stage filter check for `getStageByHashFast(hashBigIntOrNum, stage)`.
   * Accepts `undefined`/`null` (match any stage), canonical stage strings (`'vertex'`,
   * `'fragment'`, `'pixel'`, `'compute'`), non-canonical stage strings, or `SHADER_STAGE_BITS`.
   *
   * @param {object} record
   * @param {string | number | null | undefined} stage
   * @returns {boolean}
   */
  function matchesStageFilter(record, stage) {
    if (stage === undefined || stage === null) {
      return true;
    }
    if (typeof stage === 'string') {
      if (stage === record.stage) {
        return true;
      }
      if (stage === 'pixel') {
        return record.stage === 'fragment';
      }
      if (stage === 'vertex' || stage === 'fragment' || stage === 'compute') {
        return false;
      }
      const normalized = normalizeDiagnosticStage(stage);
      return normalized !== null && record.stage === normalized;
    }
    if (typeof stage === 'number') {
      return (
        Number.isInteger(stage) &&
        stage > 0 &&
        (stage & ~VALID_VISIBILITY_MASK) === 0 &&
        (record.visibility & stage) !== 0
      );
    }
    return false;
  }

  /**
   * Zero-allocation O(1) `BigInt` / safe-integer shader-hash lookup fast path with optional
   * stage filtering and bounded LRU negative-miss caching for unknown hashes (W318).
   *
   * @param {bigint | number} hashBigIntOrNum
   * @param {string | number | null | undefined} stage
   * @param {Map<bigint, any>} byHashBigInt
   * @param {Map<number, any>} byHashSafeInt
   * @param {Set<bigint | number>} negativeHashCache
   * @param {number} maxNegativeCacheSize
   * @param {{ fastHashLookups: number, fastHashHits: number, negativeLookupHits: number, lookupHits: number, lookupMisses: number }} stats
   * @returns {any}
   */
  function lookupFastHash(
    hashBigIntOrNum,
    stage,
    byHashBigInt,
    byHashSafeInt,
    negativeHashCache,
    maxNegativeCacheSize,
    stats
  ) {
    stats.fastHashLookups += 1;

    let found;
    let negativeKey;
    let cacheable = false;

    if (typeof hashBigIntOrNum === 'bigint') {
      if (hashBigIntOrNum < 0n || hashBigIntOrNum > U64_MAX) {
        stats.lookupMisses += 1;
        return undefined;
      }
      found = byHashBigInt.get(hashBigIntOrNum);
      if (found === undefined) {
        negativeKey =
          hashBigIntOrNum <= MAX_SAFE_INTEGER_BIGINT ? Number(hashBigIntOrNum) : hashBigIntOrNum;
        cacheable = true;
      }
    } else if (typeof hashBigIntOrNum === 'number') {
      if (!Number.isSafeInteger(hashBigIntOrNum) || hashBigIntOrNum < 0) {
        stats.lookupMisses += 1;
        return undefined;
      }
      found = byHashSafeInt.get(hashBigIntOrNum);
      if (found === undefined) {
        negativeKey = hashBigIntOrNum;
        cacheable = true;
      }
    } else {
      stats.lookupMisses += 1;
      return undefined;
    }

    if (found !== undefined) {
      if (matchesStageFilter(found, stage)) {
        stats.fastHashHits += 1;
        stats.lookupHits += 1;
        return found;
      }
      stats.lookupMisses += 1;
      return undefined;
    }

    if (cacheable && maxNegativeCacheSize > 0) {
      if (negativeHashCache.has(negativeKey)) {
        negativeHashCache.delete(negativeKey);
        negativeHashCache.add(negativeKey);
        stats.negativeLookupHits += 1;
      } else {
        rememberBoundedNegativeKey(negativeHashCache, negativeKey, maxNegativeCacheSize);
      }
    }

    stats.lookupMisses += 1;
    return undefined;
  }

  /**
   * Compute lookup hit rate strictly clamped to `[0, 1]` (returns `0` when total lookups is 0).
   *
   * @param {number} hits
   * @param {number} misses
   * @returns {number}
   */
  function computeLookupHitRate(hits, misses) {
    const safeHits = Number.isFinite(hits) && hits > 0 ? hits : 0;
    const safeMisses = Number.isFinite(misses) && misses > 0 ? misses : 0;
    const total = safeHits + safeMisses;
    if (total <= 0) {
      return 0;
    }
    const rate = safeHits / total;
    if (rate <= 0) return 0;
    if (rate >= 1) return 1;
    return rate;
  }

  /**
   * Deduplicate and freeze a normalized binding source object across stages in a parse-local pool,
   * eliminating thousands of identical `{ kind, ... }` wrapper objects on 9.5K-stage sidecars.
   *
   * @param {object} source
   * @param {Map<string, object>} pool
   * @returns {object}
   */
  function internNormalizedBindingSource(source, pool) {
    let key;
    switch (source.kind) {
      case 'table':
        key = `t:${source.userSgpr}:${source.tableByteOffset}`;
        break;
      case 'root_constants':
        key = `rc:${source.userSgpr}`;
        break;
      case 'root_descriptor':
        key =
          source.rootTableOffsetGiven === true
            ? `rd:${source.userSgpr}:t:${source.tableByteOffset}`
            : `rd:${source.userSgpr}`;
        break;
      case 'static_sampler':
        key = `ss:${source.index}`;
        break;
      case 'inline_sampler':
        key = `is:${source.sdwords[0]}:${source.sdwords[1]}:${source.sdwords[2]}:${source.sdwords[3]}`;
        break;
      case 'sampler_heap':
        key = `sh:${source.baseUserSgpr}:${source.stride}:${source.indexUserSgpr}:${source.indexShift}`;
        break;
      case 'resource_heap':
        key =
          source.resourceHeap === true
            ? `rhb:${source.userSgpr === undefined ? '' : source.userSgpr}`
            : `rh:${source.baseUserSgpr}:${source.stride}:${source.indexUserSgpr}:${source.indexShift}`;
        break;
      case 'cbv_indexed_sampler_heap':
      case 'cbv_indexed_resource_heap': {
        const tag = source.kind === 'cbv_indexed_sampler_heap' ? 'csh' : 'crh';
        const cbv = source.cbv;
        const tOff = cbv.kind === 'table' ? cbv.tableByteOffset : '';
        key = `${tag}:${cbv.kind}:${cbv.userSgpr}:${tOff}:${source.byteOffset}:${source.bitOffset}:${source.mask}:${source.shift}:${source.stride}`;
        break;
      }
      case 'lane_indexed_sampler_heap':
      case 'lane_indexed_resource_heap': {
        const tag = source.kind === 'lane_indexed_sampler_heap' ? 'lsh' : 'lrh';
        let wbKey = '0';
        if (source.hasWindowBase && source.windowBase) {
          const wb = source.windowBase;
          const cbv = wb.cbv;
          const tOff = cbv.kind === 'table' ? cbv.tableByteOffset : '';
          wbKey = `1:${cbv.kind}:${cbv.userSgpr}:${tOff}:${wb.byteOffset}:${wb.bitOffset}:${wb.mask}`;
        }
        key = `${tag}:${source.maxCount}:${source.shift}:${source.stride}:${wbKey}`;
        break;
      }
      case 'unresolved':
        key = `u:${source.reason === null || source.reason === undefined ? '' : source.reason}`;
        break;
      case 'rt_arena':
        key = 'rt:1';
        break;
      default:
        return source;
    }

    const existing = pool.get(key);
    if (existing !== undefined) {
      return existing;
    }
    if (source.kind === 'inline_sampler' && Array.isArray(source.sdwords)) {
      Object.freeze(source.sdwords);
    } else if (
      (source.kind === 'cbv_indexed_sampler_heap' || source.kind === 'cbv_indexed_resource_heap') &&
      source.cbv
    ) {
      Object.freeze(source.cbv);
    } else if (
      (source.kind === 'lane_indexed_sampler_heap' || source.kind === 'lane_indexed_resource_heap') &&
      source.windowBase
    ) {
      if (source.windowBase.cbv) {
        Object.freeze(source.windowBase.cbv);
      }
      Object.freeze(source.windowBase);
    }
    Object.freeze(source);
    pool.set(key, source);
    return source;
  }

  /**
   * Compute 64-bit FNV-1a hash matching `fnv1a_64` in `shaders/src/convert.rs`.
   *
   * @param {string | Uint8Array | ArrayBuffer | ArrayBufferView} input
   * @returns {bigint}
   */
  function fnv1a64(input) {
    let bytes;
    if (typeof input === 'string') {
      bytes = textEncoder.encode(input);
    } else if (input instanceof Uint8Array) {
      bytes = input;
    } else if (ArrayBuffer.isView(input)) {
      bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    } else if (input instanceof ArrayBuffer) {
      bytes = new Uint8Array(input);
    } else {
      throw new TypeError('fnv1a64 input must be a string, Uint8Array, ArrayBuffer, or ArrayBufferView');
    }

    let hash = FNV1A_64_OFFSET_BASIS;
    for (let i = 0; i < bytes.length; i += 1) {
      hash ^= BigInt(bytes[i]);
      hash = (hash * FNV1A_64_PRIME) & U64_MAX;
    }
    return hash;
  }

  /**
   * Normalize a 64-bit shader/layout hash (`BigInt`, `0x...` hex string, bare hex string,
   * or non-negative safe integer) into an unsigned 64-bit `BigInt`.
   *
   * @param {bigint | string | number} hash
   * @returns {bigint}
   */
  function normalizeShaderHash(hash) {
    if (typeof hash === 'bigint') {
      if (hash < 0n || hash > U64_MAX) {
        throw new RangeError(`Shader hash out of u64 range: ${hash.toString()}`);
      }
      return hash;
    }

    if (typeof hash === 'number') {
      if (!Number.isInteger(hash) || !Number.isSafeInteger(hash) || hash < 0) {
        throw new RangeError(`Numeric shader hash must be a non-negative safe integer, got: ${hash}`);
      }
      return BigInt(hash);
    }

    if (typeof hash === 'string') {
      const trimmed = hash.trim();
      const rawHex =
        trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
      if (!/^[0-9a-fA-F]{1,16}$/.test(rawHex)) {
        throw new RangeError(`Invalid hex shader hash: ${JSON.stringify(hash)}`);
      }
      return BigInt(`0x${rawHex}`);
    }

    throw new TypeError('Shader hash must be a BigInt, hex string, or non-negative safe integer');
  }

  /**
   * Format a 64-bit hash as canonical `"0x"` + 16 lowercase hex digits,
   * matching `format_shader_hash` in `shaders/src/convert.rs`.
   *
   * @param {bigint | string | number} hash
   * @returns {string}
   */
  function formatShaderHash(hash) {
    const value = normalizeShaderHash(hash);
    return `0x${value.toString(16).padStart(16, '0')}`;
  }

  function isHexString(str) {
    for (let i = 0; i < str.length; i += 1) {
      const code = str.charCodeAt(i);
      const isDigit = code >= 48 && code <= 57;
      const isLowerHex = code >= 97 && code <= 102;
      const isUpperHex = code >= 65 && code <= 70;
      if (!isDigit && !isLowerHex && !isUpperHex) {
        return false;
      }
    }
    return str.length > 0;
  }

  /**
   * Fast O(1) lookup by normalized 64-bit hash across `'0x...'` hex strings, bare 16-hex-char
   * keys, short/uppercase hex strings, `BigInt`, safe integers, and optional shader names.
   * Avoids regexes and `BigInt` allocations on canonical `'0x...'` and bare 16-hex-char keys.
   *
   * @param {unknown} query
   * @param {Map<string, any>} byHex
   * @param {Map<string, any>} byBareHex
   * @param {Map<bigint, any>} byHash
   * @param {Map<string, any> | null} [byName]
   * @returns {any}
   */
  function lookupByNormalizedHash(
    query,
    byHex,
    byBareHex,
    byHash,
    byName = null,
    bySafeInt = null
  ) {
    if (typeof query === 'string') {
      // Zero-allocation O(1) fast paths for already-canonical 0x+16 or bare 16-char lowercase keys.
      if (query.length === 18 && query.charCodeAt(0) === 48) {
        const c1 = query.charCodeAt(1);
        if (c1 === 120 || c1 === 88) {
          const direct = byHex.get(query);
          if (direct !== undefined) return direct;
          return byHex.get(query.toLowerCase());
        }
      }
      if (query.length === 16) {
        if (byName) {
          const named = byName.get(query);
          if (named !== undefined) return named;
        }
        const bareDirect = byBareHex.get(query);
        if (bareDirect !== undefined) return bareDirect;
        return byBareHex.get(query.toLowerCase());
      }

      const trimmed = query.trim();
      const len = trimmed.length;
      if (len === 0) {
        return undefined;
      }

      if (len >= 2 && trimmed.charCodeAt(0) === 48 && (trimmed.charCodeAt(1) === 120 || trimmed.charCodeAt(1) === 88)) {
        const bodyLen = len - 2;
        if (bodyLen === 16) {
          return byBareHex.get(trimmed.slice(2).toLowerCase());
        }
        if (bodyLen >= 1 && bodyLen < 16) {
          const body = trimmed.slice(2);
          if (isHexString(body)) {
            return byBareHex.get(body.toLowerCase().padStart(16, '0'));
          }
        }
        return undefined;
      }

      // Check name/entry_point first so names like "fade" are not mistaken for bare hex.
      if (byName) {
        const named = byName.get(trimmed);
        if (named !== undefined) return named;
      }

      if (len === 16) {
        return byBareHex.get(trimmed.toLowerCase());
      }
      if (len < 16 && isHexString(trimmed)) {
        return byBareHex.get(trimmed.toLowerCase().padStart(16, '0'));
      }
      return undefined;
    }

    if (typeof query === 'bigint') {
      if (query >= 0n && query <= U64_MAX) {
        return byHash.get(query);
      }
      return undefined;
    }

    if (typeof query === 'number') {
      if (Number.isSafeInteger(query) && query >= 0) {
        if (bySafeInt !== null) {
          return bySafeInt.get(query);
        }
        return byHash.get(BigInt(query));
      }
      return undefined;
    }

    return undefined;
  }

  function getLayoutMemberKind(entry) {
    let kind = null;
    let count = 0;
    if (entry.buffer !== undefined) {
      kind = 'buffer';
      count += 1;
    }
    if (entry.sampler !== undefined) {
      kind = 'sampler';
      count += 1;
    }
    if (entry.texture !== undefined) {
      kind = 'texture';
      count += 1;
    }
    if (entry.storageTexture !== undefined) {
      kind = 'storageTexture';
      count += 1;
    }
    if (entry.externalTexture !== undefined) {
      kind = 'externalTexture';
      count += 1;
    }
    return { kind, count };
  }

  function serializeLayoutMemberJson(entry) {
    if (entry.buffer) {
      const b = entry.buffer;
      return `"buffer":{"type":${JSON.stringify(b.type)},"hasDynamicOffset":${Boolean(
        b.hasDynamicOffset
      )},"minBindingSize":${b.minBindingSize || 0}}`;
    }
    if (entry.sampler) {
      return `"sampler":{"type":${JSON.stringify(entry.sampler.type)}}`;
    }
    if (entry.texture) {
      const t = entry.texture;
      return `"texture":{"sampleType":${JSON.stringify(t.sampleType)},"viewDimension":${JSON.stringify(
        t.viewDimension
      )},"multisampled":${Boolean(t.multisampled)}}`;
    }
    if (entry.storageTexture) {
      const st = entry.storageTexture;
      return `"storageTexture":{"access":${JSON.stringify(st.access)},"format":${JSON.stringify(
        st.format
      )},"viewDimension":${JSON.stringify(st.viewDimension)}}`;
    }
    if (entry.externalTexture) {
      return '"externalTexture":{}';
    }
    return '"unknown":{}';
  }

  function serializeLayoutEntryJson(entry) {
    return `{"binding":${entry.binding},"visibility":${entry.visibility},${serializeLayoutMemberJson(entry)}}`;
  }

  /**
   * Compute canonical S6 FNV-1a 64-bit content hash for a `GPUBindGroupLayoutDescriptor`,
   * matching `BindGroupLayoutDesc::content_hash()` in `shaders/src/reflect.rs`.
   *
   * @param {{ entries: Array<object> }} descriptor
   * @returns {string} Canonical `0x...` hash hex
   */
  function computeBindGroupLayoutHash(descriptor) {
    if (!descriptor || !Array.isArray(descriptor.entries)) {
      throw new TypeError('computeBindGroupLayoutHash requires a descriptor with an entries array');
    }
    const sorted = descriptor.entries.slice().sort((a, b) => a.binding - b.binding);
    const canonical = `[${sorted.map(serializeLayoutEntryJson).join(',')}]`;
    return formatShaderHash(fnv1a64(`bgl:${canonical}`));
  }

  /**
   * Compute canonical S6 FNV-1a 64-bit content hash for a pipeline layout's ordered BGL IDs,
   * matching `PipelineLayoutDesc::content_hash()` in `shaders/src/reflect.rs`.
   *
   * @param {Array<bigint | string | number>} bglIds
   * @returns {string} Canonical `0x...` hash hex
   */
  function computePipelineLayoutHash(bglIds) {
    if (!Array.isArray(bglIds)) {
      throw new TypeError('computePipelineLayoutHash requires an array of bind group layout IDs');
    }
    const formatted = bglIds.map((id) => formatShaderHash(id));
    return formatShaderHash(fnv1a64(`pl:[${formatted.join(',')}]`));
  }

  function cloneLayoutEntry(entry) {
    const out = {
      binding: entry.binding,
      visibility: entry.visibility,
    };
    if (entry.buffer) {
      out.buffer = {
        type: entry.buffer.type,
        hasDynamicOffset: Boolean(entry.buffer.hasDynamicOffset),
        minBindingSize: entry.buffer.minBindingSize,
      };
    } else if (entry.sampler) {
      out.sampler = { type: entry.sampler.type };
    } else if (entry.texture) {
      out.texture = {
        sampleType: entry.texture.sampleType,
        viewDimension: entry.texture.viewDimension,
        multisampled: Boolean(entry.texture.multisampled),
      };
    } else if (entry.storageTexture) {
      out.storageTexture = {
        access: entry.storageTexture.access,
        format: entry.storageTexture.format,
        viewDimension: entry.storageTexture.viewDimension,
      };
    } else if (entry.externalTexture) {
      out.externalTexture = {};
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // S6 Reflection Manifest Parsing & Validation
  // ---------------------------------------------------------------------------

  function validateHashKey(rawKey, path) {
    try {
      const hash = normalizeShaderHash(rawKey);
      return { hash, hashHex: formatShaderHash(hash) };
    } catch (err) {
      throw new ManifestValidationError(
        `invalid 64-bit hex hash ${JSON.stringify(rawKey)}: ${err.message}`,
        path
      );
    }
  }

  function validateStageName(rawStage, path) {
    if (typeof rawStage !== 'string') {
      throw new ManifestValidationError('stage must be a string ("vertex", "fragment", or "compute")', path);
    }
    const lower = rawStage.trim().toLowerCase();
    if (!Object.prototype.hasOwnProperty.call(STAGE_TO_VISIBILITY, lower)) {
      throw new ManifestValidationError(
        `unsupported shader stage ${JSON.stringify(rawStage)} (expected "vertex", "fragment", or "compute")`,
        path
      );
    }
    return lower;
  }

  function normalizeLayoutEntry(raw, path, fallbackVisibility) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('layout entry must be an object', path);
    }
    if (!isNonNegativeInt(raw.binding)) {
      throw new ManifestValidationError(`invalid binding index ${JSON.stringify(raw.binding)}`, `${path}.binding`);
    }

    let visibility = raw.visibility;
    if (visibility === undefined && fallbackVisibility !== undefined) {
      visibility = fallbackVisibility;
    }
    if (
      typeof visibility !== 'number' ||
      !Number.isInteger(visibility) ||
      visibility < 1 ||
      (visibility & ~VALID_VISIBILITY_MASK) !== 0
    ) {
      throw new ManifestValidationError(
        `invalid visibility bitmask ${JSON.stringify(raw.visibility)} (expected integer bitmask in 1..${VALID_VISIBILITY_MASK})`,
        `${path}.visibility`
      );
    }

    // Accept camelCase (exact S6 JSON) and snake_case aliases if passed from manual objects.
    const rawBuffer = raw.buffer;
    const rawSampler = raw.sampler;
    const rawTexture = raw.texture;
    const rawStorageTexture = raw.storageTexture !== undefined ? raw.storageTexture : raw.storage_texture;
    const rawExternalTexture = raw.externalTexture !== undefined ? raw.externalTexture : raw.external_texture;

    const presentMembers = [
      rawBuffer !== undefined ? 'buffer' : null,
      rawSampler !== undefined ? 'sampler' : null,
      rawTexture !== undefined ? 'texture' : null,
      rawStorageTexture !== undefined ? 'storageTexture' : null,
      rawExternalTexture !== undefined ? 'externalTexture' : null,
    ].filter(Boolean);

    if (presentMembers.length !== 1) {
      throw new ManifestValidationError(
        `layout entry must define exactly one resource member (buffer, sampler, texture, storageTexture, or externalTexture), found [${presentMembers.join(', ')}]`,
        path
      );
    }

    const entry = {
      binding: raw.binding,
      visibility,
    };

    const member = presentMembers[0];
    if (member === 'buffer') {
      if (!isPlainObject(rawBuffer)) {
        throw new ManifestValidationError('buffer layout must be an object', `${path}.buffer`);
      }
      const type = rawBuffer.type !== undefined ? rawBuffer.type : 'uniform';
      if (typeof type !== 'string' || !VALID_BUFFER_TYPES.has(type)) {
        throw new ManifestValidationError(
          `invalid buffer type ${JSON.stringify(rawBuffer.type)}`,
          `${path}.buffer.type`
        );
      }
      const rawDyn =
        rawBuffer.hasDynamicOffset !== undefined
          ? rawBuffer.hasDynamicOffset
          : rawBuffer.has_dynamic_offset;
      if (rawDyn !== undefined && typeof rawDyn !== 'boolean') {
        throw new ManifestValidationError('hasDynamicOffset must be a boolean', `${path}.buffer.hasDynamicOffset`);
      }
      const rawMinSize =
        rawBuffer.minBindingSize !== undefined
          ? rawBuffer.minBindingSize
          : rawBuffer.min_binding_size;
      let minBindingSize = 0;
      if (rawMinSize !== undefined) {
        if (typeof rawMinSize !== 'number' || !Number.isInteger(rawMinSize) || rawMinSize < 0) {
          throw new ManifestValidationError(
            `minBindingSize must be a non-negative integer, got ${JSON.stringify(rawMinSize)}`,
            `${path}.buffer.minBindingSize`
          );
        }
        minBindingSize = rawMinSize;
      }
      entry.buffer = {
        type,
        hasDynamicOffset: Boolean(rawDyn),
        minBindingSize,
      };
    } else if (member === 'sampler') {
      if (!isPlainObject(rawSampler)) {
        throw new ManifestValidationError('sampler layout must be an object', `${path}.sampler`);
      }
      const type = rawSampler.type !== undefined ? rawSampler.type : 'filtering';
      if (typeof type !== 'string' || !VALID_SAMPLER_TYPES.has(type)) {
        throw new ManifestValidationError(
          `invalid sampler type ${JSON.stringify(rawSampler.type)}`,
          `${path}.sampler.type`
        );
      }
      entry.sampler = { type };
    } else if (member === 'texture') {
      if (!isPlainObject(rawTexture)) {
        throw new ManifestValidationError('texture layout must be an object', `${path}.texture`);
      }
      const sampleType =
        rawTexture.sampleType !== undefined
          ? rawTexture.sampleType
          : rawTexture.sample_type !== undefined
            ? rawTexture.sample_type
            : 'float';
      if (typeof sampleType !== 'string' || !VALID_TEXTURE_SAMPLE_TYPES.has(sampleType)) {
        throw new ManifestValidationError(
          `invalid texture sampleType ${JSON.stringify(sampleType)}`,
          `${path}.texture.sampleType`
        );
      }
      const viewDimension =
        rawTexture.viewDimension !== undefined
          ? rawTexture.viewDimension
          : rawTexture.view_dimension !== undefined
            ? rawTexture.view_dimension
            : '2d';
      if (typeof viewDimension !== 'string' || !VALID_TEXTURE_VIEW_DIMENSIONS.has(viewDimension)) {
        throw new ManifestValidationError(
          `invalid texture viewDimension ${JSON.stringify(viewDimension)}`,
          `${path}.texture.viewDimension`
        );
      }
      if (rawTexture.multisampled !== undefined && typeof rawTexture.multisampled !== 'boolean') {
        throw new ManifestValidationError('multisampled must be a boolean', `${path}.texture.multisampled`);
      }
      const multisampled = Boolean(rawTexture.multisampled);
      if (multisampled && viewDimension !== '2d') {
        throw new ManifestValidationError(
          `multisampled texture must have viewDimension "2d", got ${JSON.stringify(viewDimension)}`,
          `${path}.texture`
        );
      }
      entry.texture = { sampleType, viewDimension, multisampled };
    } else if (member === 'storageTexture') {
      if (!isPlainObject(rawStorageTexture)) {
        throw new ManifestValidationError('storageTexture layout must be an object', `${path}.storageTexture`);
      }
      const access = rawStorageTexture.access !== undefined ? rawStorageTexture.access : 'write-only';
      if (typeof access !== 'string' || !VALID_STORAGE_TEXTURE_ACCESS.has(access)) {
        throw new ManifestValidationError(
          `invalid storageTexture access ${JSON.stringify(access)}`,
          `${path}.storageTexture.access`
        );
      }
      const format = rawStorageTexture.format;
      if (typeof format !== 'string' || format.length === 0) {
        throw new ManifestValidationError(
          'storageTexture format must be a non-empty string',
          `${path}.storageTexture.format`
        );
      }
      const viewDimension =
        rawStorageTexture.viewDimension !== undefined
          ? rawStorageTexture.viewDimension
          : rawStorageTexture.view_dimension !== undefined
            ? rawStorageTexture.view_dimension
            : '2d';
      if (
        typeof viewDimension !== 'string' ||
        !VALID_STORAGE_TEXTURE_VIEW_DIMENSIONS.has(viewDimension)
      ) {
        throw new ManifestValidationError(
          `invalid storageTexture viewDimension ${JSON.stringify(viewDimension)}`,
          `${path}.storageTexture.viewDimension`
        );
      }
      entry.storageTexture = { access, format, viewDimension };
    } else if (member === 'externalTexture') {
      if (!isPlainObject(rawExternalTexture)) {
        throw new ManifestValidationError('externalTexture layout must be an object', `${path}.externalTexture`);
      }
      entry.externalTexture = {};
    }

    return entry;
  }

  function parseRegisterConvention(raw) {
    if (raw === undefined || raw === null) {
      return { source: 'dxc-spirv', shifts: { b: 0, t: 0, u: 0, s: 0 } };
    }
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('register_convention must be an object', 'register_convention');
    }
    const source = raw.source !== undefined ? raw.source : 'dxc-spirv';
    if (typeof source !== 'string' || source.length === 0) {
      throw new ManifestValidationError('source must be a non-empty string', 'register_convention.source');
    }
    const shiftsRaw = raw.shifts !== undefined ? raw.shifts : {};
    if (!isPlainObject(shiftsRaw)) {
      throw new ManifestValidationError('shifts must be an object', 'register_convention.shifts');
    }
    const shifts = { b: 0, t: 0, u: 0, s: 0 };
    for (const key of ['b', 't', 'u', 's']) {
      if (shiftsRaw[key] !== undefined) {
        if (!isNonNegativeInt(shiftsRaw[key])) {
          throw new ManifestValidationError(
            `shift must be a non-negative uint32 integer, got ${JSON.stringify(shiftsRaw[key])}`,
            `register_convention.shifts.${key}`
          );
        }
        shifts[key] = shiftsRaw[key];
      }
    }
    return { source, shifts };
  }

  function parseHeapBindingSource(kind, rawHeap, path) {
    if (!isPlainObject(rawHeap)) {
      throw new ManifestValidationError(
        kind === 'resource_heap' ? `${kind} must be an object or true` : `${kind} must be an object`,
        path
      );
    }
    if (!isNonNegativeInt(rawHeap.base_user_sgpr)) {
      throw new ManifestValidationError(
        `${kind}.base_user_sgpr must be a non-negative uint32 safe integer, got ${JSON.stringify(rawHeap.base_user_sgpr)}`,
        `${path}.base_user_sgpr`
      );
    }
    if (!isNonNegativeInt(rawHeap.stride) || rawHeap.stride <= 0) {
      throw new ManifestValidationError(
        `${kind}.stride must be a positive uint32 safe integer (> 0), got ${JSON.stringify(rawHeap.stride)}`,
        `${path}.stride`
      );
    }
    if (!isPlainObject(rawHeap.index)) {
      throw new ManifestValidationError(
        `${kind}.index must be an object with user_sgpr and shift`,
        `${path}.index`
      );
    }
    if (!isNonNegativeInt(rawHeap.index.user_sgpr)) {
      throw new ManifestValidationError(
        `${kind}.index.user_sgpr must be a non-negative uint32 safe integer, got ${JSON.stringify(rawHeap.index.user_sgpr)}`,
        `${path}.index.user_sgpr`
      );
    }
    if (!isNonNegativeInt(rawHeap.index.shift) || rawHeap.index.shift > MAX_HEAP_INDEX_SHIFT) {
      throw new ManifestValidationError(
        `${kind}.index.shift must be an integer in 0..${MAX_HEAP_INDEX_SHIFT}, got ${JSON.stringify(rawHeap.index.shift)}`,
        `${path}.index.shift`
      );
    }
    return {
      kind,
      baseUserSgpr: rawHeap.base_user_sgpr,
      stride: rawHeap.stride,
      indexUserSgpr: rawHeap.index.user_sgpr,
      indexShift: rawHeap.index.shift,
    };
  }

  function parseInlineSamplerBindingSource(rawInlineSampler, path) {
    if (!isPlainObject(rawInlineSampler)) {
      throw new ManifestValidationError('inline_sampler must be an object', path);
    }
    const rawWords = rawInlineSampler.sdwords;
    if (!Array.isArray(rawWords) || rawWords.length !== INLINE_SAMPLER_SDWORD_COUNT) {
      throw new ManifestValidationError(
        `inline_sampler.sdwords must be an array of ${INLINE_SAMPLER_SDWORD_COUNT} unsigned 32-bit integers`,
        `${path}.sdwords`
      );
    }
    const sdwords = new Array(INLINE_SAMPLER_SDWORD_COUNT);
    for (let i = 0; i < INLINE_SAMPLER_SDWORD_COUNT; i += 1) {
      const w = rawWords[i];
      if (!isNonNegativeInt(w)) {
        throw new ManifestValidationError(
          `inline_sampler.sdwords[${i}] must be an unsigned 32-bit integer (0..0xFFFFFFFF), got ${JSON.stringify(w)}`,
          `${path}.sdwords[${i}]`
        );
      }
      sdwords[i] = w;
    }
    return {
      kind: 'inline_sampler',
      sdwords,
    };
  }

  function parseCbvDwordSpec(rawBody, path) {
    if (!isPlainObject(rawBody)) {
      throw new ManifestValidationError('CBV dword descriptor must be an object', path);
    }
    const rawCbv = rawBody.cbv;
    if (!isPlainObject(rawCbv)) {
      throw new ManifestValidationError('cbv must be an object', `${path}.cbv`);
    }
    if (!isNonNegativeInt(rawCbv.user_sgpr)) {
      throw new ManifestValidationError(
        `cbv.user_sgpr must be a non-negative uint32 safe integer, got ${JSON.stringify(rawCbv.user_sgpr)}`,
        `${path}.cbv.user_sgpr`
      );
    }
    if (hasOwn(rawCbv, 'root_descriptor') && typeof rawCbv.root_descriptor !== 'boolean') {
      throw new ManifestValidationError('cbv.root_descriptor must be a boolean', `${path}.cbv.root_descriptor`);
    }
    if (hasOwn(rawCbv, 'root_constants') && typeof rawCbv.root_constants !== 'boolean') {
      throw new ManifestValidationError('cbv.root_constants must be a boolean', `${path}.cbv.root_constants`);
    }
    const isRootDesc = rawCbv.root_descriptor === true;
    const isRootConsts = rawCbv.root_constants === true;
    const hasTableOff = hasOwn(rawCbv, 'table_byte_offset') && rawCbv.table_byte_offset !== undefined;
    const modeCount = (isRootDesc ? 1 : 0) + (isRootConsts ? 1 : 0) + (hasTableOff ? 1 : 0);
    if (modeCount !== 1) {
      throw new ManifestValidationError(
        'cbv must specify exactly one of root_descriptor: true, root_constants: true, or table_byte_offset',
        `${path}.cbv`
      );
    }
    let cbv;
    if (isRootDesc) {
      cbv = { kind: 'root_descriptor', userSgpr: rawCbv.user_sgpr };
    } else if (isRootConsts) {
      cbv = { kind: 'root_constants', userSgpr: rawCbv.user_sgpr };
    } else {
      if (!isNonNegativeInt(rawCbv.table_byte_offset)) {
        throw new ManifestValidationError(
          `cbv.table_byte_offset must be a non-negative uint32 safe integer, got ${JSON.stringify(rawCbv.table_byte_offset)}`,
          `${path}.cbv.table_byte_offset`
        );
      }
      cbv = {
        kind: 'table',
        userSgpr: rawCbv.user_sgpr,
        tableByteOffset: rawCbv.table_byte_offset,
      };
    }

    if (!isNonNegativeInt(rawBody.byte_offset) || rawBody.byte_offset % 4 !== 0) {
      throw new ManifestValidationError(
        `byte_offset must be a dword-aligned non-negative uint32 integer, got ${JSON.stringify(rawBody.byte_offset)}`,
        `${path}.byte_offset`
      );
    }
    let bitOffset = 0;
    if (rawBody.bit_offset !== undefined && rawBody.bit_offset !== null) {
      if (!isNonNegativeInt(rawBody.bit_offset) || rawBody.bit_offset > MAX_HEAP_INDEX_SHIFT) {
        throw new ManifestValidationError(
          `bit_offset must be an integer in 0..${MAX_HEAP_INDEX_SHIFT}, got ${JSON.stringify(rawBody.bit_offset)}`,
          `${path}.bit_offset`
        );
      }
      bitOffset = rawBody.bit_offset;
    }
    let mask = U32_MAX;
    if (rawBody.mask !== undefined && rawBody.mask !== null) {
      if (!isNonNegativeInt(rawBody.mask) || rawBody.mask === 0) {
        throw new ManifestValidationError(
          `mask must be a non-zero uint32 integer, got ${JSON.stringify(rawBody.mask)}`,
          `${path}.mask`
        );
      }
      mask = rawBody.mask;
    }
    return {
      cbv,
      byteOffset: rawBody.byte_offset,
      bitOffset,
      mask,
    };
  }

  function parseCbvIndexedHeapBindingSource(kind, rawBody, path) {
    if (!isPlainObject(rawBody)) {
      throw new ManifestValidationError(`${kind} must be an object`, path);
    }
    const spec = parseCbvDwordSpec(rawBody, path);
    if (!isNonNegativeInt(rawBody.shift) || rawBody.shift > MAX_HEAP_INDEX_SHIFT) {
      throw new ManifestValidationError(
        `${kind}.shift must be an integer in 0..${MAX_HEAP_INDEX_SHIFT}, got ${JSON.stringify(rawBody.shift)}`,
        `${path}.shift`
      );
    }
    let stride = 0;
    if (rawBody.stride !== undefined && rawBody.stride !== null) {
      if (!isNonNegativeInt(rawBody.stride)) {
        throw new ManifestValidationError(
          `${kind}.stride must be a non-negative uint32 integer, got ${JSON.stringify(rawBody.stride)}`,
          `${path}.stride`
        );
      }
      stride = rawBody.stride;
    }
    const out = {
      kind,
      cbv: spec.cbv,
      userSgpr: spec.cbv.userSgpr,
      byteOffset: spec.byteOffset,
      bitOffset: spec.bitOffset,
      mask: spec.mask,
      shift: rawBody.shift,
      stride,
    };
    if (spec.cbv.kind === 'table') {
      out.tableByteOffset = spec.cbv.tableByteOffset;
    }
    return out;
  }

  function parseLaneIndexedHeapBindingSource(kind, rawBody, path) {
    if (!isPlainObject(rawBody)) {
      throw new ManifestValidationError(`${kind} must be an object`, path);
    }
    if (
      !isNonNegativeInt(rawBody.max_count) ||
      rawBody.max_count < 1 ||
      rawBody.max_count > MAX_LANE_INDEXED_WINDOW
    ) {
      throw new ManifestValidationError(
        `${kind}.max_count must be an integer in 1..${MAX_LANE_INDEXED_WINDOW}, got ${JSON.stringify(rawBody.max_count)}`,
        `${path}.max_count`
      );
    }
    if (!isNonNegativeInt(rawBody.shift) || rawBody.shift > MAX_HEAP_INDEX_SHIFT) {
      throw new ManifestValidationError(
        `${kind}.shift must be an integer in 0..${MAX_HEAP_INDEX_SHIFT}, got ${JSON.stringify(rawBody.shift)}`,
        `${path}.shift`
      );
    }
    let stride = 0;
    if (rawBody.stride !== undefined && rawBody.stride !== null) {
      if (!isNonNegativeInt(rawBody.stride)) {
        throw new ManifestValidationError(
          `${kind}.stride must be a non-negative uint32 integer, got ${JSON.stringify(rawBody.stride)}`,
          `${path}.stride`
        );
      }
      stride = rawBody.stride;
    }
    let windowBase = null;
    if (rawBody.window_base !== undefined && rawBody.window_base !== null) {
      windowBase = parseCbvDwordSpec(rawBody.window_base, `${path}.window_base`);
    }
    return {
      kind,
      maxCount: rawBody.max_count,
      shift: rawBody.shift,
      stride,
      hasWindowBase: windowBase !== null,
      windowBase,
    };
  }

  function parseMeshOutputLayout(rawMesh, path) {
    if (!isPlainObject(rawMesh)) {
      throw new ManifestValidationError('mesh_output must be an object', path);
    }
    const maxVertices =
      rawMesh.max_vertices !== undefined ? rawMesh.max_vertices : rawMesh.maxVertices;
    const maxPrimitives =
      rawMesh.max_primitives !== undefined ? rawMesh.max_primitives : rawMesh.maxPrimitives;
    const vertexWords =
      rawMesh.vertex_words !== undefined ? rawMesh.vertex_words : rawMesh.vertexWords;
    const maxVertexWords = 4 + 4 * MAX_MESH_ATTRIBUTES;

    if (!isNonNegativeInt(maxVertices) || maxVertices < 1 || maxVertices > MAX_MESH_VERTICES) {
      throw new ManifestValidationError(
        `mesh_output.max_vertices must be an integer in 1..${MAX_MESH_VERTICES}, got ${JSON.stringify(maxVertices)}`,
        `${path}.max_vertices`
      );
    }
    if (!isNonNegativeInt(maxPrimitives) || maxPrimitives < 1 || maxPrimitives > MAX_MESH_PRIMITIVES) {
      throw new ManifestValidationError(
        `mesh_output.max_primitives must be an integer in 1..${MAX_MESH_PRIMITIVES}, got ${JSON.stringify(maxPrimitives)}`,
        `${path}.max_primitives`
      );
    }
    if (
      !isNonNegativeInt(vertexWords) ||
      vertexWords < 4 ||
      vertexWords % 4 !== 0 ||
      vertexWords > maxVertexWords
    ) {
      throw new ManifestValidationError(
        `mesh_output.vertex_words must be 4 + 4 * n (4..${maxVertexWords}), got ${JSON.stringify(vertexWords)}`,
        `${path}.vertex_words`
      );
    }
    return Object.freeze({
      maxVertices,
      maxPrimitives,
      vertexWords,
      max_vertices: maxVertices,
      max_primitives: maxPrimitives,
      vertex_words: vertexWords,
    });
  }

  /**
   * Parse and normalize a binding `source` entry matching `WgslBindingSource` in
   * `shaders/src/rdna2_to_wgsl.rs`, `runtime/gpu/stage_reflection.cpp` (v1/v2/v3), and
   * `tools/stage_reflection_sidecar.py`:
   * - `{"sampler_heap": {"base_user_sgpr", "stride", "index": {"user_sgpr", "shift"}}}`
   * - `{"resource_heap": {"base_user_sgpr", "stride", "index": {"user_sgpr", "shift"}}}`
   * - `{"resource_heap": true}` (or `{"user_sgpr", "resource_heap": true}`)
   * - `{"cbv_indexed_resource_heap" | "cbv_indexed_sampler_heap": {"cbv", "byte_offset", "shift", "bit_offset"?, "mask"?, "stride"?}}`
   * - `{"lane_indexed_resource_heap" | "lane_indexed_sampler_heap": {"max_count", "shift", "stride"?, "window_base"?}}`
   * - `{"inline_sampler": {"sdwords": [d0, d1, d2, d3]}}`
   * - `{"unresolved": true | "<reason>"}` (version >= 3; rejected for writable `storage` / `storage_texture`)
   * - `{"rt_arena": true}` (group 3, bindings 16..21)
   * - `{"user_sgpr", "table_byte_offset"}`
   * - `{"user_sgpr", "root_constants": true}`
   * - `{"user_sgpr", "root_descriptor": true, "table_byte_offset"? (v2+)}`
   * - `{"static_sampler": <u32>}`
   *
   * Returns `null` when `rawSource` is `undefined` or `null`.
   *
   * @param {unknown} rawSource
   * @param {string} [path]
   * @param {{ version?: number, bindingKind?: string | null }} [options]
   * @returns {object | null}
   */
  function parseBindingSource(rawSource, path = 'source', options = null) {
    if (rawSource === undefined || rawSource === null) {
      return null;
    }
    if (!isPlainObject(rawSource)) {
      throw new ManifestValidationError('binding source must be an object or null', path);
    }

    const version =
      options && typeof options.version === 'number'
        ? options.version
        : STAGE_REFLECTION_MAX_VERSION;
    const bindingKind =
      options && typeof options.bindingKind === 'string' ? options.bindingKind : null;

    const hasSamplerHeap = hasOwn(rawSource, 'sampler_heap');
    const hasResourceHeap = hasOwn(rawSource, 'resource_heap');
    const hasCbvSamplerHeap = hasOwn(rawSource, 'cbv_indexed_sampler_heap');
    const hasCbvResourceHeap = hasOwn(rawSource, 'cbv_indexed_resource_heap');
    const hasLaneSamplerHeap = hasOwn(rawSource, 'lane_indexed_sampler_heap');
    const hasLaneResourceHeap = hasOwn(rawSource, 'lane_indexed_resource_heap');
    const hasInlineSampler = hasOwn(rawSource, 'inline_sampler');
    const hasUnresolved = hasOwn(rawSource, 'unresolved');
    const hasRtArena = hasOwn(rawSource, 'rt_arena');
    const hasStaticSampler = hasOwn(rawSource, 'static_sampler');
    const hasUserSgpr = hasOwn(rawSource, 'user_sgpr');
    const hasTableByteOffset = hasOwn(rawSource, 'table_byte_offset');
    const hasRootConstants = hasOwn(rawSource, 'root_constants');
    const hasRootDescriptor = hasOwn(rawSource, 'root_descriptor');

    if (hasSamplerHeap && hasResourceHeap) {
      throw new ManifestValidationError(
        'binding source cannot define both sampler_heap and resource_heap',
        path
      );
    }

    const hasV2V3Special =
      hasCbvSamplerHeap ||
      hasCbvResourceHeap ||
      hasLaneSamplerHeap ||
      hasLaneResourceHeap ||
      hasUnresolved ||
      hasRtArena;

    if (hasResourceHeap && rawSource.resource_heap === true) {
      if (
        hasV2V3Special ||
        hasInlineSampler ||
        hasStaticSampler ||
        hasTableByteOffset ||
        hasRootConstants ||
        hasRootDescriptor
      ) {
        throw new ManifestValidationError(
          'resource_heap source cannot be combined with other binding source fields',
          path
        );
      }
      if (hasUserSgpr) {
        if (!isNonNegativeInt(rawSource.user_sgpr)) {
          throw new ManifestValidationError(
            `user_sgpr must be a non-negative uint32 safe integer, got ${JSON.stringify(rawSource.user_sgpr)}`,
            `${path}.user_sgpr`
          );
        }
        return {
          kind: 'resource_heap',
          resourceHeap: true,
          userSgpr: rawSource.user_sgpr,
        };
      }
      return {
        kind: 'resource_heap',
        resourceHeap: true,
      };
    }

    if (hasSamplerHeap || hasResourceHeap) {
      const heapKey = hasSamplerHeap ? 'sampler_heap' : 'resource_heap';
      if (
        hasV2V3Special ||
        hasInlineSampler ||
        hasStaticSampler ||
        hasUserSgpr ||
        hasTableByteOffset ||
        hasRootConstants ||
        hasRootDescriptor
      ) {
        throw new ManifestValidationError(
          `${heapKey} source cannot be combined with other binding source fields`,
          path
        );
      }
      return parseHeapBindingSource(heapKey, rawSource[heapKey], `${path}.${heapKey}`);
    }

    if (hasCbvSamplerHeap || hasCbvResourceHeap) {
      if (hasCbvSamplerHeap && hasCbvResourceHeap) {
        throw new ManifestValidationError(
          'binding source cannot define both cbv_indexed_sampler_heap and cbv_indexed_resource_heap',
          path
        );
      }
      const key = hasCbvSamplerHeap ? 'cbv_indexed_sampler_heap' : 'cbv_indexed_resource_heap';
      if (
        hasLaneSamplerHeap ||
        hasLaneResourceHeap ||
        hasInlineSampler ||
        hasUnresolved ||
        hasRtArena ||
        hasStaticSampler ||
        hasUserSgpr ||
        hasTableByteOffset ||
        hasRootConstants ||
        hasRootDescriptor
      ) {
        throw new ManifestValidationError(
          `${key} source cannot be combined with other binding source fields`,
          path
        );
      }
      return parseCbvIndexedHeapBindingSource(key, rawSource[key], `${path}.${key}`);
    }

    if (hasLaneSamplerHeap || hasLaneResourceHeap) {
      if (hasLaneSamplerHeap && hasLaneResourceHeap) {
        throw new ManifestValidationError(
          'binding source cannot define both lane_indexed_sampler_heap and lane_indexed_resource_heap',
          path
        );
      }
      const key = hasLaneSamplerHeap ? 'lane_indexed_sampler_heap' : 'lane_indexed_resource_heap';
      if (
        hasInlineSampler ||
        hasUnresolved ||
        hasRtArena ||
        hasStaticSampler ||
        hasUserSgpr ||
        hasTableByteOffset ||
        hasRootConstants ||
        hasRootDescriptor
      ) {
        throw new ManifestValidationError(
          `${key} source cannot be combined with other binding source fields`,
          path
        );
      }
      return parseLaneIndexedHeapBindingSource(key, rawSource[key], `${path}.${key}`);
    }

    if (hasInlineSampler) {
      if (
        hasUnresolved ||
        hasRtArena ||
        hasStaticSampler ||
        hasUserSgpr ||
        hasTableByteOffset ||
        hasRootConstants ||
        hasRootDescriptor
      ) {
        throw new ManifestValidationError(
          'inline_sampler source cannot be combined with other binding source fields',
          path
        );
      }
      return parseInlineSamplerBindingSource(rawSource.inline_sampler, `${path}.inline_sampler`);
    }

    if (hasRtArena) {
      if (
        hasUnresolved ||
        hasStaticSampler ||
        hasUserSgpr ||
        hasTableByteOffset ||
        hasRootConstants ||
        hasRootDescriptor
      ) {
        throw new ManifestValidationError(
          'rt_arena source cannot be combined with other binding source fields',
          path
        );
      }
      if (rawSource.rt_arena !== true) {
        throw new ManifestValidationError('rt_arena source must be true', `${path}.rt_arena`);
      }
      return {
        kind: 'rt_arena',
        rtArena: true,
      };
    }

    if (hasUnresolved) {
      if (
        hasStaticSampler ||
        hasUserSgpr ||
        hasTableByteOffset ||
        hasRootConstants ||
        hasRootDescriptor
      ) {
        throw new ManifestValidationError(
          'unresolved source cannot be combined with other binding source fields',
          path
        );
      }
      const rawUnresolved = rawSource.unresolved;
      const marked =
        rawUnresolved === true || (typeof rawUnresolved === 'string' && rawUnresolved.length > 0);
      if (version < STAGE_REFLECTION_VERSION_UNRESOLVED || !marked) {
        throw new ManifestValidationError(
          'unresolved source needs "unresolved": true or a non-empty reason string, and version >= 3',
          `${path}.unresolved`
        );
      }
      if (bindingKind === 'storage' || bindingKind === 'storage_texture') {
        throw new ManifestValidationError(
          `a ${bindingKind} binding cannot have an unresolved source`,
          `${path}.unresolved`
        );
      }
      return {
        kind: 'unresolved',
        unresolved: true,
        reason: typeof rawUnresolved === 'string' ? rawUnresolved : null,
      };
    }

    if (hasStaticSampler) {
      if (hasUserSgpr || hasTableByteOffset || hasRootConstants || hasRootDescriptor) {
        throw new ManifestValidationError(
          'static_sampler source cannot be combined with user_sgpr source fields',
          path
        );
      }
      if (!isNonNegativeInt(rawSource.static_sampler)) {
        throw new ManifestValidationError(
          `static_sampler must be a non-negative uint32 safe integer, got ${JSON.stringify(rawSource.static_sampler)}`,
          `${path}.static_sampler`
        );
      }
      return {
        kind: 'static_sampler',
        index: rawSource.static_sampler,
      };
    }

    if (hasUserSgpr) {
      if (!isNonNegativeInt(rawSource.user_sgpr)) {
        throw new ManifestValidationError(
          `user_sgpr must be a non-negative uint32 safe integer, got ${JSON.stringify(rawSource.user_sgpr)}`,
          `${path}.user_sgpr`
        );
      }
      if (hasRootConstants && typeof rawSource.root_constants !== 'boolean') {
        throw new ManifestValidationError(
          'root_constants must be a boolean',
          `${path}.root_constants`
        );
      }
      if (hasRootDescriptor && typeof rawSource.root_descriptor !== 'boolean') {
        throw new ManifestValidationError(
          'root_descriptor must be a boolean',
          `${path}.root_descriptor`
        );
      }

      const isRootConstants = rawSource.root_constants === true;
      const isRootDescriptor = rawSource.root_descriptor === true;
      const isTable = hasTableByteOffset && rawSource.table_byte_offset !== undefined;

      if (isRootConstants) {
        if (isRootDescriptor || isTable) {
          throw new ManifestValidationError(
            'user_sgpr source must specify exactly one of table_byte_offset, root_constants: true, or root_descriptor: true',
            path
          );
        }
        return {
          kind: 'root_constants',
          userSgpr: rawSource.user_sgpr,
        };
      }
      if (isRootDescriptor) {
        if (isTable && version >= STAGE_REFLECTION_VERSION_ROOT_TABLE_OFFSET) {
          if (!isNonNegativeInt(rawSource.table_byte_offset)) {
            throw new ManifestValidationError(
              `bad root_descriptor table_byte_offset ${JSON.stringify(rawSource.table_byte_offset)}`,
              `${path}.table_byte_offset`
            );
          }
          return {
            kind: 'root_descriptor',
            userSgpr: rawSource.user_sgpr,
            tableByteOffset: rawSource.table_byte_offset,
            rootTableOffsetGiven: true,
          };
        }
        return {
          kind: 'root_descriptor',
          userSgpr: rawSource.user_sgpr,
        };
      }
      if (!isTable) {
        throw new ManifestValidationError(
          'user_sgpr source must specify exactly one of table_byte_offset, root_constants: true, or root_descriptor: true',
          path
        );
      }
      if (!isNonNegativeInt(rawSource.table_byte_offset)) {
        throw new ManifestValidationError(
          `table_byte_offset must be a non-negative uint32 safe integer, got ${JSON.stringify(rawSource.table_byte_offset)}`,
          `${path}.table_byte_offset`
        );
      }
      return {
        kind: 'table',
        userSgpr: rawSource.user_sgpr,
        tableByteOffset: rawSource.table_byte_offset,
      };
    }

    if (hasTableByteOffset || hasRootConstants || hasRootDescriptor) {
      throw new ManifestValidationError(
        'binding source requires user_sgpr alongside table_byte_offset, root_constants, or root_descriptor',
        path
      );
    }

    throw new ManifestValidationError(
      'unrecognized binding source (expected sampler_heap, resource_heap, cbv_indexed_*_heap, lane_indexed_*_heap, inline_sampler, unresolved, rt_arena, static_sampler, or user_sgpr)',
      path
    );
  }

  function parseBindingRecord(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('binding record must be an object', path);
    }
    if (!isNonNegativeInt(raw.group)) {
      throw new ManifestValidationError(`invalid group ${JSON.stringify(raw.group)}`, `${path}.group`);
    }
    if (!isNonNegativeInt(raw.binding)) {
      throw new ManifestValidationError(`invalid binding ${JSON.stringify(raw.binding)}`, `${path}.binding`);
    }
    if (raw.name !== undefined && raw.name !== null && typeof raw.name !== 'string') {
      throw new ManifestValidationError('name must be a string or null', `${path}.name`);
    }
    const resource = raw.resource;
    if (typeof resource !== 'string' || !VALID_RESOURCE_KINDS.has(resource)) {
      throw new ManifestValidationError(
        `invalid resource kind ${JSON.stringify(resource)}`,
        `${path}.resource`
      );
    }
    const inLayout = raw.in_layout !== undefined ? raw.in_layout : true;
    if (typeof inLayout !== 'boolean') {
      throw new ManifestValidationError('in_layout must be a boolean', `${path}.in_layout`);
    }

    let register = null;
    if (raw.register !== undefined && raw.register !== null) {
      if (!isPlainObject(raw.register)) {
        throw new ManifestValidationError('register must be an object or null', `${path}.register`);
      }
      if (typeof raw.register.class !== 'string' || !VALID_REGISTER_CLASSES.has(raw.register.class)) {
        throw new ManifestValidationError(
          `invalid register class ${JSON.stringify(raw.register.class)}`,
          `${path}.register.class`
        );
      }
      if (!isNonNegativeInt(raw.register.index) || !isNonNegativeInt(raw.register.space)) {
        throw new ManifestValidationError(
          'register index and space must be non-negative integers',
          `${path}.register`
        );
      }
      register = {
        class: raw.register.class,
        index: raw.register.index,
        space: raw.register.space,
      };
    }

    let spirv = null;
    if (raw.spirv !== undefined && raw.spirv !== null) {
      if (!isPlainObject(raw.spirv)) {
        throw new ManifestValidationError('spirv must be an object or null', `${path}.spirv`);
      }
      if (!isNonNegativeInt(raw.spirv.set) || !isNonNegativeInt(raw.spirv.binding)) {
        throw new ManifestValidationError('spirv set and binding must be non-negative integers', `${path}.spirv`);
      }
      spirv = { set: raw.spirv.set, binding: raw.spirv.binding };
    }

    const rootConstants = raw.root_constants !== undefined ? raw.root_constants : false;
    if (typeof rootConstants !== 'boolean') {
      throw new ManifestValidationError('root_constants must be a boolean', `${path}.root_constants`);
    }

    let bindingArray = null;
    if (raw.binding_array !== undefined && raw.binding_array !== null) {
      if (!isPlainObject(raw.binding_array)) {
        throw new ManifestValidationError('binding_array must be an object or null', `${path}.binding_array`);
      }
      const count = raw.binding_array.count;
      if (count !== null && !isNonNegativeInt(count)) {
        throw new ManifestValidationError(
          'binding_array.count must be a non-negative integer or null',
          `${path}.binding_array.count`
        );
      }
      bindingArray = { count };
    }

    const source = parseBindingSource(raw.source, `${path}.source`);

    let minBindingSizeBytes = null;
    const rawMinBindingSizeBytes =
      raw.min_binding_size_bytes !== undefined
        ? raw.min_binding_size_bytes
        : raw.minBindingSizeBytes;
    if (rawMinBindingSizeBytes !== undefined && rawMinBindingSizeBytes !== null) {
      if (!isNonNegativeInt(rawMinBindingSizeBytes)) {
        throw new ManifestValidationError(
          `min_binding_size_bytes must be a non-negative uint32 safe integer, got ${JSON.stringify(rawMinBindingSizeBytes)}`,
          `${path}.min_binding_size_bytes`
        );
      }
      minBindingSizeBytes = rawMinBindingSizeBytes;
    }

    let storageFormat = null;
    const rawStorageFormat =
      raw.storage_format !== undefined ? raw.storage_format : raw.storageFormat;
    if (rawStorageFormat !== undefined && rawStorageFormat !== null) {
      if (
        (typeof rawStorageFormat !== 'string' || rawStorageFormat.length === 0) &&
        !isNonNegativeInt(rawStorageFormat)
      ) {
        throw new ManifestValidationError(
          `storage_format must be a non-empty string or non-negative integer, got ${JSON.stringify(rawStorageFormat)}`,
          `${path}.storage_format`
        );
      }
      storageFormat = rawStorageFormat;
    }

    return {
      group: raw.group,
      binding: raw.binding,
      name: raw.name !== undefined ? raw.name : null,
      resource,
      in_layout: inLayout,
      inLayout,
      register,
      spirv,
      root_constants: rootConstants,
      rootConstants,
      binding_array: bindingArray,
      bindingArray,
      source,
      min_binding_size_bytes: minBindingSizeBytes,
      minBindingSizeBytes,
      storage_format: storageFormat,
      storageFormat,
      layoutEntry: null,
    };
  }

  function parseVertexInput(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('vertex_input must be an object', path);
    }
    if (!isNonNegativeInt(raw.location)) {
      throw new ManifestValidationError(`invalid location ${JSON.stringify(raw.location)}`, `${path}.location`);
    }
    if (raw.type !== undefined && typeof raw.type !== 'string') {
      throw new ManifestValidationError('type must be a string', `${path}.type`);
    }
    if (raw.format !== undefined && raw.format !== null && typeof raw.format !== 'string') {
      throw new ManifestValidationError('format must be a string or null', `${path}.format`);
    }
    return {
      location: raw.location,
      type: raw.type !== undefined ? raw.type : 'unknown',
      format: raw.format !== undefined ? raw.format : null,
    };
  }

  function parseFragmentOutput(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('fragment_output must be an object', path);
    }
    if (!isNonNegativeInt(raw.location)) {
      throw new ManifestValidationError(`invalid location ${JSON.stringify(raw.location)}`, `${path}.location`);
    }
    if (typeof raw.type !== 'string' || raw.type.length === 0) {
      throw new ManifestValidationError('type must be a non-empty string', `${path}.type`);
    }
    return {
      location: raw.location,
      type: raw.type,
    };
  }

  function parseLoweringNeed(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('requires_lowering item must be an object', path);
    }
    if (typeof raw.kind !== 'string' || raw.kind.length === 0) {
      throw new ManifestValidationError('kind must be a non-empty string', `${path}.kind`);
    }
    if (raw.group !== undefined && raw.group !== null && !isNonNegativeInt(raw.group)) {
      throw new ManifestValidationError('group must be a non-negative integer or null', `${path}.group`);
    }
    if (raw.binding !== undefined && raw.binding !== null && !isNonNegativeInt(raw.binding)) {
      throw new ManifestValidationError('binding must be a non-negative integer or null', `${path}.binding`);
    }
    if (typeof raw.detail !== 'string') {
      throw new ManifestValidationError('detail must be a string', `${path}.detail`);
    }
    return {
      kind: raw.kind,
      group: raw.group !== undefined ? raw.group : null,
      binding: raw.binding !== undefined ? raw.binding : null,
      detail: raw.detail,
    };
  }

  function parseLimitViolation(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('limit_violation item must be an object', path);
    }
    if (typeof raw.limit !== 'string' || raw.limit.length === 0) {
      throw new ManifestValidationError('limit must be a non-empty string', `${path}.limit`);
    }
    let stage = null;
    if (raw.stage !== undefined && raw.stage !== null) {
      stage = validateStageName(raw.stage, `${path}.stage`);
    }
    if (typeof raw.count !== 'number' || !Number.isInteger(raw.count) || raw.count < 0) {
      throw new ManifestValidationError('count must be a non-negative integer', `${path}.count`);
    }
    if (typeof raw.max !== 'number' || !Number.isInteger(raw.max) || raw.max < 0) {
      throw new ManifestValidationError('max must be a non-negative integer', `${path}.max`);
    }
    return {
      limit: raw.limit,
      stage,
      count: raw.count,
      max: raw.max,
    };
  }

  function resolveVertexInputMode(rawStage, stage, vertexInputs, path) {
    let rawMode;
    if (rawStage.vertex_input_mode !== undefined) {
      rawMode = rawStage.vertex_input_mode;
    } else if (rawStage.vertexInputMode !== undefined) {
      rawMode = rawStage.vertexInputMode;
    } else if (
      isPlainObject(rawStage.binding_summary) &&
      rawStage.binding_summary.vertex_input_mode !== undefined
    ) {
      rawMode = rawStage.binding_summary.vertex_input_mode;
    } else if (
      isPlainObject(rawStage.bindingSummary) &&
      rawStage.bindingSummary.vertexInputMode !== undefined
    ) {
      rawMode = rawStage.bindingSummary.vertexInputMode;
    }

    if (rawMode !== undefined && rawMode !== null) {
      if (typeof rawMode !== 'string' || rawMode.trim().length === 0) {
        throw new ManifestValidationError(
          `vertex_input_mode must be a non-empty string, got ${JSON.stringify(rawMode)}`,
          `${path}.vertex_input_mode`
        );
      }
      return rawMode.trim().toLowerCase();
    }

    if (stage === 'vertex') {
      return vertexInputs.length === 0 ? 'vertex_index' : 'attributes';
    }
    return null;
  }

  /**
   * Parse and validate an S6 reflection manifest (`layout-manifest.json` emitted by
   * `shaders/src/reflect.rs`). Normalizes all 64-bit hashes (`0x...` hex and `BigInt`)
   * and resolves per-shader bind group layouts and pipeline references.
   *
   * @param {string | object} jsonOrObject
   * @returns {object} Normalized, indexed manifest object
   */
  function parseReflectionManifest(jsonOrObject) {
    let raw;
    if (typeof jsonOrObject === 'string') {
      try {
        raw = JSON.parse(jsonOrObject);
      } catch (err) {
        throw new ManifestValidationError(
          `malformed JSON: ${err instanceof Error ? err.message : String(err)}`
        );
      }
    } else {
      raw = jsonOrObject;
    }

    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('manifest root must be a JSON object');
    }

    if (raw.schema_version !== LAYOUT_MANIFEST_SCHEMA_VERSION) {
      throw new ManifestValidationError(
        `unsupported schema_version ${JSON.stringify(raw.schema_version)} (expected ${LAYOUT_MANIFEST_SCHEMA_VERSION})`,
        'schema_version'
      );
    }

    const registerConvention = parseRegisterConvention(raw.register_convention);

    // 1. Parse bind_group_layouts table
    const bindGroupLayouts = new Map();
    const bindGroupLayoutsByBigInt = new Map();
    if (raw.bind_group_layouts !== undefined) {
      if (!isPlainObject(raw.bind_group_layouts)) {
        throw new ManifestValidationError('bind_group_layouts must be an object', 'bind_group_layouts');
      }
      for (const [rawId, rawBgl] of Object.entries(raw.bind_group_layouts)) {
        const bglPath = `bind_group_layouts[${JSON.stringify(rawId)}]`;
        const { hash: id, hashHex: idHex } = validateHashKey(rawId, bglPath);
        if (bindGroupLayoutsByBigInt.has(id)) {
          throw new ManifestValidationError(`duplicate bind group layout id ${idHex}`, bglPath);
        }
        if (!isPlainObject(rawBgl) || !Array.isArray(rawBgl.entries)) {
          throw new ManifestValidationError('bind group layout must be an object with an "entries" array', bglPath);
        }

        const seenBindings = new Set();
        const entries = new Array(rawBgl.entries.length);
        for (let i = 0; i < rawBgl.entries.length; i += 1) {
          const entry = normalizeLayoutEntry(rawBgl.entries[i], `${bglPath}.entries[${i}]`);
          if (seenBindings.has(entry.binding)) {
            throw new ManifestValidationError(
              `duplicate binding index ${entry.binding}`,
              `${bglPath}.entries[${i}]`
            );
          }
          seenBindings.add(entry.binding);
          entries[i] = entry;
        }
        entries.sort((a, b) => a.binding - b.binding);

        const normalizedBgl = { id, idHex, entries };
        bindGroupLayouts.set(idHex, normalizedBgl);
        bindGroupLayoutsByBigInt.set(id, normalizedBgl);
      }
    }

    // 2. Parse pipeline_layouts table
    const pipelineLayouts = new Map();
    const pipelineLayoutsByBigInt = new Map();
    if (raw.pipeline_layouts !== undefined) {
      if (!isPlainObject(raw.pipeline_layouts)) {
        throw new ManifestValidationError('pipeline_layouts must be an object', 'pipeline_layouts');
      }
      for (const [rawId, rawPl] of Object.entries(raw.pipeline_layouts)) {
        const plPath = `pipeline_layouts[${JSON.stringify(rawId)}]`;
        const { hash: id, hashHex: idHex } = validateHashKey(rawId, plPath);
        if (pipelineLayoutsByBigInt.has(id)) {
          throw new ManifestValidationError(`duplicate pipeline layout id ${idHex}`, plPath);
        }
        if (!isPlainObject(rawPl) || !Array.isArray(rawPl.bindGroupLayouts)) {
          throw new ManifestValidationError(
            'pipeline layout must be an object with a "bindGroupLayouts" array',
            plPath
          );
        }
        const bglIdHexes = new Array(rawPl.bindGroupLayouts.length);
        for (let g = 0; g < rawPl.bindGroupLayouts.length; g += 1) {
          const { hashHex: bglHex } = validateHashKey(
            rawPl.bindGroupLayouts[g],
            `${plPath}.bindGroupLayouts[${g}]`
          );
          if (!bindGroupLayouts.has(bglHex)) {
            throw new ManifestValidationError(
              `references unknown bind_group_layout ${bglHex}`,
              `${plPath}.bindGroupLayouts[${g}]`
            );
          }
          bglIdHexes[g] = bglHex;
        }
        const normalizedPl = { id, idHex, bindGroupLayouts: bglIdHexes };
        pipelineLayouts.set(idHex, normalizedPl);
        pipelineLayoutsByBigInt.set(id, normalizedPl);
      }
    }

    // 3. Parse shaders table (required)
    if (!isPlainObject(raw.shaders)) {
      throw new ManifestValidationError('shaders must be an object', 'shaders');
    }

    const shaders = new Map();
    const shadersByHash = new Map();
    const shadersByBareHex = new Map();
    const shadersByName = new Map();
    let totalSamplerHeapBindings = 0;
    let totalResourceHeapStages = 0;
    let totalResourceHeapBindings = 0;
    let totalInlineSamplerStages = 0;
    let totalInlineSamplerBindings = 0;
    let totalVertexIndexStages = 0;

    for (const [rawHash, rawShader] of Object.entries(raw.shaders)) {
      const shPath = `shaders[${JSON.stringify(rawHash)}]`;
      const { hash, hashHex } = validateHashKey(rawHash, shPath);
      const bareHashHex = hashHex.slice(2);
      if (shadersByHash.has(hash)) {
        throw new ManifestValidationError(`duplicate shader hash ${hashHex}`, shPath);
      }
      if (!isPlainObject(rawShader)) {
        throw new ManifestValidationError('shader record must be an object', shPath);
      }

      const stage = validateStageName(rawShader.stage, `${shPath}.stage`);
      const visibility = STAGE_TO_VISIBILITY[stage];

      const rawEntryPoint =
        rawShader.entry_point !== undefined ? rawShader.entry_point : rawShader.entryPoint;
      if (typeof rawEntryPoint !== 'string' || rawEntryPoint.length === 0) {
        throw new ManifestValidationError('entry_point must be a non-empty string', `${shPath}.entry_point`);
      }

      if (rawShader.name !== undefined && rawShader.name !== null && typeof rawShader.name !== 'string') {
        throw new ManifestValidationError('name must be a string', `${shPath}.name`);
      }
      const shaderName = typeof rawShader.name === 'string' && rawShader.name.length > 0 ? rawShader.name : null;

      // Validate pipeline_layout reference
      let pipelineLayoutIdHex = null;
      if (rawShader.pipeline_layout !== undefined && rawShader.pipeline_layout !== null) {
        const { hashHex: plHex } = validateHashKey(rawShader.pipeline_layout, `${shPath}.pipeline_layout`);
        if (raw.pipeline_layouts !== undefined && !pipelineLayouts.has(plHex)) {
          throw new ManifestValidationError(
            `references unknown pipeline_layout ${plHex}`,
            `${shPath}.pipeline_layout`
          );
        }
        pipelineLayoutIdHex = plHex;
      }

      // Resolve bind_group_layouts references
      const rawBglIds = rawShader.bind_group_layouts;
      const bindGroupLayoutIds = [];
      const resolvedGroupLayouts = [];

      if (rawBglIds !== undefined) {
        if (!Array.isArray(rawBglIds)) {
          throw new ManifestValidationError(
            'bind_group_layouts must be an array of layout ids',
            `${shPath}.bind_group_layouts`
          );
        }
        if (rawBglIds.length > MAX_REFLECTED_GROUPS) {
          throw new ManifestValidationError(
            `bind_group_layouts length ${rawBglIds.length} exceeds reflection limit ${MAX_REFLECTED_GROUPS}`,
            `${shPath}.bind_group_layouts`
          );
        }
        for (let g = 0; g < rawBglIds.length; g += 1) {
          const { hashHex: bglHex } = validateHashKey(
            rawBglIds[g],
            `${shPath}.bind_group_layouts[${g}]`
          );
          const bgl = bindGroupLayouts.get(bglHex);
          if (!bgl) {
            throw new ManifestValidationError(
              `references unknown bind_group_layout ${bglHex}`,
              `${shPath}.bind_group_layouts[${g}]`
            );
          }
          bindGroupLayoutIds.push(bglHex);
          resolvedGroupLayouts.push({
            group: g,
            id: bgl.id,
            idHex: bgl.idHex,
            entries: bgl.entries.map(cloneLayoutEntry),
          });
        }
      }

      // Parse layout_complete
      let layoutComplete = true;
      if (rawShader.layout_complete !== undefined) {
        if (typeof rawShader.layout_complete !== 'boolean') {
          throw new ManifestValidationError('layout_complete must be a boolean', `${shPath}.layout_complete`);
        }
        layoutComplete = rawShader.layout_complete;
      }

      // Parse workgroup_size
      let workgroupSize = null;
      if (rawShader.workgroup_size !== undefined && rawShader.workgroup_size !== null) {
        if (
          !Array.isArray(rawShader.workgroup_size) ||
          rawShader.workgroup_size.length !== 3 ||
          !rawShader.workgroup_size.every((n) => isNonNegativeInt(n) && n >= 1)
        ) {
          throw new ManifestValidationError(
            'workgroup_size must be null or a 3-element array of positive integers',
            `${shPath}.workgroup_size`
          );
        }
        workgroupSize = [
          rawShader.workgroup_size[0],
          rawShader.workgroup_size[1],
          rawShader.workgroup_size[2],
        ];
      }

      // Parse bindings metadata and cross-link with resolved bind_group_layouts entries
      const bindings = [];
      let shaderSamplerHeapBindings = 0;
      let shaderResourceHeapBindings = 0;
      let shaderInlineSamplerBindings = 0;
      if (rawShader.bindings !== undefined) {
        if (!Array.isArray(rawShader.bindings)) {
          throw new ManifestValidationError('bindings must be an array', `${shPath}.bindings`);
        }
        for (let i = 0; i < rawShader.bindings.length; i += 1) {
          const rec = parseBindingRecord(rawShader.bindings[i], `${shPath}.bindings[${i}]`);
          const groupLayout = resolvedGroupLayouts[rec.group];
          if (groupLayout && rec.inLayout) {
            const matched = groupLayout.entries.find((e) => e.binding === rec.binding);
            if (matched) {
              rec.layoutEntry = cloneLayoutEntry(matched);
            }
          }
          if (rec.source) {
            if (rec.source.kind === 'sampler_heap') {
              shaderSamplerHeapBindings += 1;
            } else if (rec.source.kind === 'resource_heap') {
              shaderResourceHeapBindings += 1;
            } else if (rec.source.kind === 'inline_sampler') {
              shaderInlineSamplerBindings += 1;
            }
          }
          bindings.push(rec);
        }
      }
      totalSamplerHeapBindings += shaderSamplerHeapBindings;
      totalResourceHeapBindings += shaderResourceHeapBindings;
      totalInlineSamplerBindings += shaderInlineSamplerBindings;
      const hasInlineSampler = shaderInlineSamplerBindings > 0;
      const hasResourceHeap = shaderResourceHeapBindings > 0;
      if (hasInlineSampler) {
        totalInlineSamplerStages += 1;
      }
      if (hasResourceHeap) {
        totalResourceHeapStages += 1;
      }

      // Parse vertex_inputs
      const vertexInputs = [];
      const rawVertexInputs =
        rawShader.vertex_inputs !== undefined ? rawShader.vertex_inputs : rawShader.vertexInputs;
      if (rawVertexInputs !== undefined) {
        if (!Array.isArray(rawVertexInputs)) {
          throw new ManifestValidationError('vertex_inputs must be an array', `${shPath}.vertex_inputs`);
        }
        for (let i = 0; i < rawVertexInputs.length; i += 1) {
          vertexInputs.push(parseVertexInput(rawVertexInputs[i], `${shPath}.vertex_inputs[${i}]`));
        }
        vertexInputs.sort((a, b) => a.location - b.location);
      }

      // Parse fragment_outputs
      const fragmentOutputs = [];
      const rawFragmentOutputs =
        rawShader.fragment_outputs !== undefined
          ? rawShader.fragment_outputs
          : rawShader.fragmentOutputs;
      if (rawFragmentOutputs !== undefined) {
        if (!Array.isArray(rawFragmentOutputs)) {
          throw new ManifestValidationError(
            'fragment_outputs must be an array',
            `${shPath}.fragment_outputs`
          );
        }
        for (let i = 0; i < rawFragmentOutputs.length; i += 1) {
          fragmentOutputs.push(
            parseFragmentOutput(rawFragmentOutputs[i], `${shPath}.fragment_outputs[${i}]`)
          );
        }
        fragmentOutputs.sort((a, b) => a.location - b.location);
      }

      // Parse optional FSSM vs_outputs / ps_inputs semantic maps (#2085)
      const vsOutputs = parseStageParamList(
        rawShader.vs_outputs !== undefined ? rawShader.vs_outputs : rawShader.vsOutputs,
        'vs_outputs',
        stage,
        'vertex',
        `${shPath}.vs_outputs`
      );
      const psInputs = parseStageParamList(
        rawShader.ps_inputs !== undefined ? rawShader.ps_inputs : rawShader.psInputs,
        'ps_inputs',
        stage,
        'fragment',
        `${shPath}.ps_inputs`
      );

      // Parse requires_lowering
      const requiresLowering = [];
      if (rawShader.requires_lowering !== undefined) {
        if (!Array.isArray(rawShader.requires_lowering)) {
          throw new ManifestValidationError(
            'requires_lowering must be an array',
            `${shPath}.requires_lowering`
          );
        }
        for (let i = 0; i < rawShader.requires_lowering.length; i += 1) {
          requiresLowering.push(
            parseLoweringNeed(rawShader.requires_lowering[i], `${shPath}.requires_lowering[${i}]`)
          );
        }
      }

      // Parse limit_violations
      const limitViolations = [];
      if (rawShader.limit_violations !== undefined) {
        if (!Array.isArray(rawShader.limit_violations)) {
          throw new ManifestValidationError(
            'limit_violations must be an array',
            `${shPath}.limit_violations`
          );
        }
        for (let i = 0; i < rawShader.limit_violations.length; i += 1) {
          limitViolations.push(
            parseLimitViolation(rawShader.limit_violations[i], `${shPath}.limit_violations[${i}]`)
          );
        }
      }

      const vertexInputMode = resolveVertexInputMode(rawShader, stage, vertexInputs, shPath);
      const isVertexIndexStage = stage === 'vertex' && vertexInputMode === 'vertex_index';
      if (isVertexIndexStage) {
        totalVertexIndexStages += 1;
      }

      const shaderRecord = {
        hash,
        hashHex,
        bareHashHex,
        stage,
        visibility,
        entryPoint: rawEntryPoint,
        entry_point: rawEntryPoint,
        name: shaderName,
        pipelineLayoutId: pipelineLayoutIdHex,
        pipeline_layout: pipelineLayoutIdHex,
        bindGroupLayoutIds,
        bind_group_layouts: bindGroupLayoutIds.slice(),
        bindGroupLayouts: resolvedGroupLayouts,
        layoutComplete,
        layout_complete: layoutComplete,
        workgroupSize,
        workgroup_size: workgroupSize ? workgroupSize.slice() : null,
        bindings,
        vertexInputs,
        vertex_inputs: vertexInputs,
        vertexInputMode,
        vertex_input_mode: vertexInputMode,
        isVertexIndexStage,
        fragmentOutputs,
        fragment_outputs: fragmentOutputs,
        vsOutputs,
        vs_outputs: vsOutputs,
        psInputs,
        ps_inputs: psInputs,
        requiresLowering,
        requires_lowering: requiresLowering,
        limitViolations,
        limit_violations: limitViolations,
        samplerHeapBindings: shaderSamplerHeapBindings,
        resourceHeapBindings: shaderResourceHeapBindings,
        inlineSamplerBindings: shaderInlineSamplerBindings,
        hasInlineSampler,
        hasResourceHeap,
      };

      shaders.set(hashHex, shaderRecord);
      shadersByHash.set(hash, shaderRecord);
      shadersByBareHex.set(bareHashHex, shaderRecord);

      const indexName = (key) => {
        let list = shadersByName.get(key);
        if (!list) {
          list = [];
          shadersByName.set(key, list);
        }
        list.push(shaderRecord);
      };
      indexName(rawEntryPoint);
      indexName(`${stage}:${rawEntryPoint}`);
      if (shaderName && shaderName !== rawEntryPoint) {
        indexName(shaderName);
      }
    }

    // 4. Parse pre-merged pipelines table (optional)
    const pipelines = new Map();
    if (raw.pipelines !== undefined) {
      if (!isPlainObject(raw.pipelines)) {
        throw new ManifestValidationError('pipelines must be an object', 'pipelines');
      }
      for (const [rawKey, rawPipeline] of Object.entries(raw.pipelines)) {
        const pipePath = `pipelines[${JSON.stringify(rawKey)}]`;
        if (!isPlainObject(rawPipeline)) {
          throw new ManifestValidationError('pipeline record must be an object', pipePath);
        }
        if (!isPlainObject(rawPipeline.stages)) {
          throw new ManifestValidationError('stages must be an object', `${pipePath}.stages`);
        }

        const stages = {};
        const stageList = [];
        for (const [rawStage, rawShaderHash] of Object.entries(rawPipeline.stages)) {
          const stage = validateStageName(rawStage, `${pipePath}.stages.${rawStage}`);
          const { hash: shHash, hashHex: shHex } = validateHashKey(
            rawShaderHash,
            `${pipePath}.stages.${rawStage}`
          );
          if (!shadersByHash.has(shHash)) {
            throw new ManifestValidationError(
              `references unknown shader ${shHex}`,
              `${pipePath}.stages.${rawStage}`
            );
          }
          stages[stage] = shHex;
          stageList.push({ stage, hash: shHash, hashHex: shHex });
        }
        stageList.sort((a, b) => STAGE_ORDER[a.stage] - STAGE_ORDER[b.stage] || (a.hash < b.hash ? -1 : a.hash > b.hash ? 1 : 0));
        const canonicalKey = stageList.map((s) => s.hashHex).join('+');

        let pipelineLayoutIdHex = null;
        if (rawPipeline.pipeline_layout !== undefined && rawPipeline.pipeline_layout !== null) {
          const { hashHex: plHex } = validateHashKey(
            rawPipeline.pipeline_layout,
            `${pipePath}.pipeline_layout`
          );
          if (raw.pipeline_layouts !== undefined && !pipelineLayouts.has(plHex)) {
            throw new ManifestValidationError(
              `references unknown pipeline_layout ${plHex}`,
              `${pipePath}.pipeline_layout`
            );
          }
          pipelineLayoutIdHex = plHex;
        }

        const bindGroupLayoutIds = [];
        const resolvedGroupLayouts = [];
        if (rawPipeline.bind_group_layouts !== undefined) {
          if (!Array.isArray(rawPipeline.bind_group_layouts)) {
            throw new ManifestValidationError(
              'bind_group_layouts must be an array',
              `${pipePath}.bind_group_layouts`
            );
          }
          for (let g = 0; g < rawPipeline.bind_group_layouts.length; g += 1) {
            const { hashHex: bglHex } = validateHashKey(
              rawPipeline.bind_group_layouts[g],
              `${pipePath}.bind_group_layouts[${g}]`
            );
            const bgl = bindGroupLayouts.get(bglHex);
            if (!bgl) {
              throw new ManifestValidationError(
                `references unknown bind_group_layout ${bglHex}`,
                `${pipePath}.bind_group_layouts[${g}]`
              );
            }
            bindGroupLayoutIds.push(bglHex);
            resolvedGroupLayouts.push({
              group: g,
              id: bgl.id,
              idHex: bgl.idHex,
              entries: bgl.entries.map(cloneLayoutEntry),
            });
          }
        }

        const layoutComplete =
          rawPipeline.layout_complete !== undefined ? rawPipeline.layout_complete : true;
        if (typeof layoutComplete !== 'boolean') {
          throw new ManifestValidationError(
            'layout_complete must be a boolean',
            `${pipePath}.layout_complete`
          );
        }

        const limitViolations = [];
        if (rawPipeline.limit_violations !== undefined) {
          if (!Array.isArray(rawPipeline.limit_violations)) {
            throw new ManifestValidationError(
              'limit_violations must be an array',
              `${pipePath}.limit_violations`
            );
          }
          for (let i = 0; i < rawPipeline.limit_violations.length; i += 1) {
            limitViolations.push(
              parseLimitViolation(rawPipeline.limit_violations[i], `${pipePath}.limit_violations[${i}]`)
            );
          }
        }

        pipelines.set(canonicalKey, {
          key: canonicalKey,
          rawKey,
          stages,
          stageList,
          pipelineLayoutId: pipelineLayoutIdHex,
          pipeline_layout: pipelineLayoutIdHex,
          bindGroupLayoutIds,
          bind_group_layouts: bindGroupLayoutIds.slice(),
          bindGroupLayouts: resolvedGroupLayouts,
          layoutComplete,
          layout_complete: layoutComplete,
          limitViolations,
          limit_violations: limitViolations,
        });
      }
    }

    const summary = {
      shaderCount: shaders.size,
      stageCount: shaders.size,
      stagesIndexed: shaders.size,
      bindGroupLayoutCount: bindGroupLayouts.size,
      pipelineLayoutCount: pipelineLayouts.size,
      pipelineCount: pipelines.size,
      samplerHeapBindings: totalSamplerHeapBindings,
      resourceHeapStages: totalResourceHeapStages,
      resourceHeapBindings: totalResourceHeapBindings,
      inlineSamplerStages: totalInlineSamplerStages,
      inlineSamplerBindings: totalInlineSamplerBindings,
      vertexIndexStages: totalVertexIndexStages,
    };

    return {
      schemaVersion: LAYOUT_MANIFEST_SCHEMA_VERSION,
      schema_version: LAYOUT_MANIFEST_SCHEMA_VERSION,
      registerConvention,
      register_convention: registerConvention,
      bindGroupLayouts,
      bindGroupLayoutsByBigInt,
      pipelineLayouts,
      pipelineLayoutsByBigInt,
      shaders,
      shadersByHash,
      shadersByBareHex,
      shadersByName,
      pipelines,
      stageCount: shaders.size,
      stagesIndexed: shaders.size,
      samplerHeapBindings: totalSamplerHeapBindings,
      resourceHeapStages: totalResourceHeapStages,
      resourceHeapBindings: totalResourceHeapBindings,
      inlineSamplerStages: totalInlineSamplerStages,
      inlineSamplerBindings: totalInlineSamplerBindings,
      vertexIndexStages: totalVertexIndexStages,
      summary,
    };
  }

  function normalizeSidecarViewDimension(rawDim, isStorage, path) {
    if (rawDim === undefined || rawDim === null) {
      return CANONICAL_VIEW_DIMENSIONS['2d'];
    }
    if (typeof rawDim !== 'string') {
      throw new ManifestValidationError(
        `view_dimension must be a string, got ${JSON.stringify(rawDim)}`,
        path
      );
    }
    const normalized = rawDim.trim().toLowerCase().replace(/_/g, '-');
    const validSet = isStorage ? VALID_STORAGE_TEXTURE_VIEW_DIMENSIONS : VALID_TEXTURE_VIEW_DIMENSIONS;
    if (!validSet.has(normalized)) {
      throw new ManifestValidationError(
        `invalid view_dimension ${JSON.stringify(rawDim)}`,
        path
      );
    }
    return CANONICAL_VIEW_DIMENSIONS[normalized] || normalized;
  }

  function normalizeSidecarSampleType(rawSampleType, path) {
    if (rawSampleType === undefined || rawSampleType === null) {
      return CANONICAL_SAMPLE_TYPES.float;
    }
    if (typeof rawSampleType !== 'string') {
      throw new ManifestValidationError(
        `sample_type must be a string, got ${JSON.stringify(rawSampleType)}`,
        path
      );
    }
    const normalized = rawSampleType.trim().toLowerCase().replace(/_/g, '-');
    if (!VALID_TEXTURE_SAMPLE_TYPES.has(normalized)) {
      throw new ManifestValidationError(
        `invalid sample_type ${JSON.stringify(rawSampleType)}`,
        path
      );
    }
    return CANONICAL_SAMPLE_TYPES[normalized] || normalized;
  }

  function parseSidecarVertexInput(raw, path, internStr = null) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('vertex_input must be an object', path);
    }
    if (!isNonNegativeInt(raw.location)) {
      throw new ManifestValidationError(
        `invalid location ${JSON.stringify(raw.location)}`,
        `${path}.location`
      );
    }
    const rawVbSlot = raw.vb_slot !== undefined ? raw.vb_slot : raw.vbSlot;
    if (rawVbSlot !== undefined && !isNonNegativeInt(rawVbSlot)) {
      throw new ManifestValidationError(
        `vb_slot must be a non-negative uint32 integer, got ${JSON.stringify(rawVbSlot)}`,
        `${path}.vb_slot`
      );
    }
    const rawByteOffset = raw.byte_offset !== undefined ? raw.byte_offset : raw.byteOffset;
    if (rawByteOffset !== undefined && !isNonNegativeInt(rawByteOffset)) {
      throw new ManifestValidationError(
        `byte_offset must be a non-negative uint32 integer, got ${JSON.stringify(rawByteOffset)}`,
        `${path}.byte_offset`
      );
    }
    let components = null;
    if (raw.components !== undefined && raw.components !== null) {
      if (!isNonNegativeInt(raw.components)) {
        throw new ManifestValidationError(
          `components must be a non-negative uint32 integer, got ${JSON.stringify(raw.components)}`,
          `${path}.components`
        );
      }
      components = raw.components;
    }
    const rawComponentType =
      raw.component_type !== undefined ? raw.component_type : raw.componentType;
    if (rawComponentType !== undefined && rawComponentType !== null && typeof rawComponentType !== 'string') {
      throw new ManifestValidationError('component_type must be a string or null', `${path}.component_type`);
    }
    const rawFetchFormat = raw.fetch_format !== undefined ? raw.fetch_format : raw.fetchFormat;
    if (rawFetchFormat !== undefined && rawFetchFormat !== null && typeof rawFetchFormat !== 'string') {
      throw new ManifestValidationError('fetch_format must be a string or null', `${path}.fetch_format`);
    }
    if (raw.format !== undefined && raw.format !== null && typeof raw.format !== 'string') {
      throw new ManifestValidationError('format must be a string or null', `${path}.format`);
    }
    const intern = internStr || ((s) => (s === undefined || s === null ? null : s));
    const fetchFormat = rawFetchFormat !== undefined && rawFetchFormat !== null ? intern(rawFetchFormat) : null;
    const format =
      raw.format !== undefined && raw.format !== null ? intern(raw.format) : fetchFormat;
    const componentType =
      rawComponentType !== undefined && rawComponentType !== null ? intern(rawComponentType) : null;
    const typeStr =
      typeof raw.type === 'string' ? intern(raw.type) : componentType || 'unknown';
    return {
      location: raw.location,
      vb_slot: rawVbSlot !== undefined ? rawVbSlot : null,
      vbSlot: rawVbSlot !== undefined ? rawVbSlot : null,
      byte_offset: rawByteOffset !== undefined ? rawByteOffset : null,
      byteOffset: rawByteOffset !== undefined ? rawByteOffset : null,
      components,
      component_type: componentType,
      componentType,
      fetch_format: fetchFormat,
      fetchFormat,
      type: typeStr,
      format,
    };
  }

  function parseStageParamInfo(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('stage param entry must be an object', path);
    }
    const rawSemanticReg =
      raw.semantic_reg !== undefined ? raw.semantic_reg : raw.semanticReg;
    if (!isNonNegativeInt(rawSemanticReg) || rawSemanticReg > 0xff) {
      throw new ManifestValidationError(
        `semantic_reg must be an integer in 0..255, got ${JSON.stringify(rawSemanticReg)}`,
        `${path}.semantic_reg`
      );
    }
    if (!isNonNegativeInt(raw.location)) {
      throw new ManifestValidationError(
        `location must be a non-negative uint32 integer, got ${JSON.stringify(raw.location)}`,
        `${path}.location`
      );
    }
    let mask = null;
    if (raw.mask !== undefined && raw.mask !== null) {
      if (!isNonNegativeInt(raw.mask) || raw.mask > 0x0f) {
        throw new ManifestValidationError(
          `mask must be an integer in 0..15, got ${JSON.stringify(raw.mask)}`,
          `${path}.mask`
        );
      }
      mask = raw.mask;
    }
    const entry = {
      semantic_reg: rawSemanticReg,
      semanticReg: rawSemanticReg,
      location: raw.location,
    };
    if (mask !== null) {
      entry.mask = mask;
    }
    return Object.freeze(entry);
  }

  function parseStageParamList(rawList, fieldName, stage, expectedStage, path) {
    if (rawList === undefined || rawList === null) {
      return EMPTY_ARRAY;
    }
    if (!Array.isArray(rawList)) {
      throw new ManifestValidationError(`${fieldName} must be an array`, path);
    }
    if (rawList.length === 0) {
      return EMPTY_ARRAY;
    }
    if (stage !== expectedStage) {
      throw new ManifestValidationError(
        `${fieldName} is only valid on ${expectedStage} stages`,
        path
      );
    }
    const out = new Array(rawList.length);
    for (let i = 0; i < rawList.length; i += 1) {
      out[i] = parseStageParamInfo(rawList[i], `${path}[${i}]`);
    }
    return Object.freeze(out);
  }

  function buildSidecarLayoutEntry(rec, visibility) {
    if (rec.kind === 'uniform') {
      return {
        binding: rec.binding,
        visibility,
        buffer: {
          type: 'uniform',
          hasDynamicOffset: false,
          minBindingSize: rec.minBindingSizeBytes || 0,
        },
      };
    }
    if (rec.kind === 'storage') {
      return {
        binding: rec.binding,
        visibility,
        buffer: {
          type: 'storage',
          hasDynamicOffset: false,
          minBindingSize: rec.minBindingSizeBytes || 0,
        },
      };
    }
    if (rec.kind === 'read_only_storage') {
      return {
        binding: rec.binding,
        visibility,
        buffer: {
          type: 'read-only-storage',
          hasDynamicOffset: false,
          minBindingSize: rec.minBindingSizeBytes || 0,
        },
      };
    }
    if (rec.kind === 'texture') {
      return {
        binding: rec.binding,
        visibility,
        texture: {
          sampleType: rec.sampleType || 'float',
          viewDimension: rec.viewDimension || '2d',
          multisampled: Boolean(rec.multisampled),
        },
      };
    }
    if (rec.kind === 'storage_texture') {
      if (typeof rec.storageFormat === 'string' && rec.storageFormat.length > 0) {
        return {
          binding: rec.binding,
          visibility,
          storageTexture: {
            access: 'write-only',
            format: rec.storageFormat,
            viewDimension: rec.viewDimension || '2d',
          },
        };
      }
      return null;
    }
    if (rec.kind === 'sampler') {
      return {
        binding: rec.binding,
        visibility,
        sampler: { type: 'filtering' },
      };
    }
    if (rec.kind === 'comparison_sampler') {
      return {
        binding: rec.binding,
        visibility,
        sampler: { type: 'comparison' },
      };
    }
    return null;
  }

  function getOrCreateCompactSidecarLayoutEntry(rec, visibility, layoutEntryPool) {
    let key;
    if (rec.kind === 'uniform' || rec.kind === 'storage' || rec.kind === 'read_only_storage') {
      key = `b:${visibility}:${rec.binding}:${rec.kind}:${rec.minBindingSizeBytes || 0}`;
    } else if (rec.kind === 'texture') {
      key = `t:${visibility}:${rec.binding}:${rec.sampleType || 'float'}:${rec.viewDimension || '2d'}:${rec.multisampled ? 1 : 0}`;
    } else if (rec.kind === 'storage_texture') {
      if (typeof rec.storageFormat !== 'string' || rec.storageFormat.length === 0) {
        return null;
      }
      key = `st:${visibility}:${rec.binding}:${rec.storageFormat}:${rec.viewDimension || '2d'}`;
    } else if (rec.kind === 'sampler' || rec.kind === 'comparison_sampler') {
      key = `s:${visibility}:${rec.binding}:${rec.kind}`;
    } else {
      return null;
    }

    const existing = layoutEntryPool.get(key);
    if (existing !== undefined) {
      return existing;
    }
    const created = buildSidecarLayoutEntry(rec, visibility);
    if (!created) {
      return null;
    }
    if (created.buffer) Object.freeze(created.buffer);
    else if (created.texture) Object.freeze(created.texture);
    else if (created.storageTexture) Object.freeze(created.storageTexture);
    else if (created.sampler) Object.freeze(created.sampler);
    Object.freeze(created);
    layoutEntryPool.set(key, created);
    return created;
  }

  function sidecarKindToResource(kind) {
    if (kind === 'uniform' || kind === 'storage' || kind === 'read_only_storage') {
      return CANONICAL_RESOURCE_KINDS.buffer;
    }
    if (kind === 'texture') return CANONICAL_RESOURCE_KINDS.texture;
    if (kind === 'storage_texture') return CANONICAL_RESOURCE_KINDS.storageTexture;
    if (kind === 'sampler' || kind === 'comparison_sampler') {
      return CANONICAL_RESOURCE_KINDS.sampler;
    }
    return CANONICAL_RESOURCE_KINDS.buffer;
  }

  /**
   * Construct the public sidecar index API object in an isolated scope so its closures
   * (`getStage`, `getStageByHashFast`, `hasStage`, `getStats`, `resetStats`) cannot retain any
   * raw JSON payload or parse-time interning tables after `parseStageReflectionSidecar` returns.
   */
  function createParsedStageSidecar(
    version,
    stages,
    stagesByHash,
    stagesByBareHex,
    stagesBySafeInt,
    totals,
    maxNegativeCacheSize
  ) {
    const stageCount = stages.size;
    const inlineSamplerStages = totals.inlineSamplerStages;
    const inlineSamplerBindings = totals.inlineSamplerBindings;
    const resourceHeapStages = totals.resourceHeapStages;
    const resourceHeapBindings = totals.resourceHeapBindings;
    const samplerHeapBindings = totals.samplerHeapBindings;
    const vertexIndexStages = totals.vertexIndexStages;
    const unresolvedBindings = totals.unresolvedBindings || 0;
    const cbvIndexedHeapBindings = totals.cbvIndexedHeapBindings || 0;
    const laneIndexedHeapBindings = totals.laneIndexedHeapBindings || 0;
    const meshOutputStages = totals.meshOutputStages || 0;
    const rtArenaStages = totals.rtArenaStages || 0;

    const negativeHashCache = new Set();
    const negativeQueryCache = new Set();

    const lookupStats = {
      lookupHits: 0,
      lookupMisses: 0,
      fastHashLookups: 0,
      fastHashHits: 0,
      negativeLookupHits: 0,
    };

    const summary = {
      stageCount,
      stagesIndexed: stageCount,
      inlineSamplerStages,
      inlineSamplerBindings,
      resourceHeapStages,
      resourceHeapBindings,
      samplerHeapBindings,
      vertexIndexStages,
    };
    Object.defineProperties(summary, {
      unresolvedBindings: { value: unresolvedBindings, enumerable: false, configurable: true },
      cbvIndexedHeapBindings: { value: cbvIndexedHeapBindings, enumerable: false, configurable: true },
      laneIndexedHeapBindings: { value: laneIndexedHeapBindings, enumerable: false, configurable: true },
      meshOutputStages: { value: meshOutputStages, enumerable: false, configurable: true },
      rtArenaStages: { value: rtArenaStages, enumerable: false, configurable: true },
    });

    return {
      schema: STAGE_REFLECTION_SCHEMA,
      version,
      stages,
      stagesByHash,
      _byHashBigInt: stagesByHash,
      stagesByHex: stages,
      stagesByBareHex,
      stageCount,
      stagesIndexed: stageCount,
      inlineSamplerStages,
      inlineSamplerBindings,
      resourceHeapStages,
      resourceHeapBindings,
      samplerHeapBindings,
      vertexIndexStages,
      unresolvedBindings,
      cbvIndexedHeapBindings,
      laneIndexedHeapBindings,
      meshOutputStages,
      rtArenaStages,
      summary,
      getStageByHashFast(hashBigIntOrNum, stage) {
        return lookupFastHash(
          hashBigIntOrNum,
          stage,
          stagesByHash,
          stagesBySafeInt,
          negativeHashCache,
          maxNegativeCacheSize,
          lookupStats
        );
      },
      getStage(hashKey) {
        if (typeof hashKey === 'bigint' || typeof hashKey === 'number') {
          return lookupFastHash(
            hashKey,
            undefined,
            stagesByHash,
            stagesBySafeInt,
            negativeHashCache,
            maxNegativeCacheSize,
            lookupStats
          );
        }

        if (
          typeof hashKey === 'string' &&
          maxNegativeCacheSize > 0 &&
          hashKey.length > 0 &&
          hashKey.length <= MAX_NEGATIVE_STRING_KEY_LENGTH &&
          negativeQueryCache.has(hashKey)
        ) {
          negativeQueryCache.delete(hashKey);
          negativeQueryCache.add(hashKey);
          lookupStats.negativeLookupHits += 1;
          lookupStats.lookupMisses += 1;
          return undefined;
        }

        const found = lookupByNormalizedHash(
          hashKey,
          stages,
          stagesByBareHex,
          stagesByHash,
          null,
          stagesBySafeInt
        );
        if (found !== undefined) {
          lookupStats.lookupHits += 1;
        } else {
          if (
            typeof hashKey === 'string' &&
            maxNegativeCacheSize > 0 &&
            hashKey.length > 0 &&
            hashKey.length <= MAX_NEGATIVE_STRING_KEY_LENGTH
          ) {
            rememberBoundedNegativeKey(negativeQueryCache, hashKey, maxNegativeCacheSize);
          }
          lookupStats.lookupMisses += 1;
        }
        return found;
      },
      hasStage(hashKey) {
        return (
          lookupByNormalizedHash(
            hashKey,
            stages,
            stagesByBareHex,
            stagesByHash,
            null,
            stagesBySafeInt
          ) !== undefined
        );
      },
      resetStats() {
        lookupStats.lookupHits = 0;
        lookupStats.lookupMisses = 0;
        lookupStats.fastHashLookups = 0;
        lookupStats.fastHashHits = 0;
        lookupStats.negativeLookupHits = 0;
      },
      getStats() {
        const lookupHits = sanitizeNonNegativeInt(lookupStats.lookupHits);
        const lookupMisses = sanitizeNonNegativeInt(lookupStats.lookupMisses);
        const fastHashLookups = sanitizeNonNegativeInt(lookupStats.fastHashLookups);
        const fastHashHits = Math.min(
          sanitizeNonNegativeInt(lookupStats.fastHashHits),
          fastHashLookups
        );
        const negativeLookupHits = Math.min(
          sanitizeNonNegativeInt(lookupStats.negativeLookupHits),
          lookupMisses
        );
        const stats = {
          stageCount,
          stagesIndexed: stageCount,
          inlineSamplerStages,
          inlineSamplerBindings,
          resourceHeapStages,
          resourceHeapBindings,
          samplerHeapBindings,
          vertexIndexStages,
          lookupHits,
          lookupMisses,
          lookupHitRate: computeLookupHitRate(lookupHits, lookupMisses),
          fastHashLookups,
          fastHashHits,
          negativeLookupHits,
        };
        Object.defineProperties(stats, {
          unresolvedBindings: { value: unresolvedBindings, enumerable: false, configurable: true },
          cbvIndexedHeapBindings: { value: cbvIndexedHeapBindings, enumerable: false, configurable: true },
          laneIndexedHeapBindings: { value: laneIndexedHeapBindings, enumerable: false, configurable: true },
          meshOutputStages: { value: meshOutputStages, enumerable: false, configurable: true },
          rtArenaStages: { value: rtArenaStages, enumerable: false, configurable: true },
        });
        return stats;
      },
    };
  }

  /**
   * Parse and validate a `playgta6-stage-reflection` v1/v2/v3 sidecar JSON document
   * (e.g., `splash.wgsl.pack.reflect.json` emitted by `tools/stage_reflection_sidecar.py`
   * and consumed by `runtime/gpu/stage_reflection.cpp`).
   *
   * Validates all v1/v2/v3 binding sources (`table`, `root_constants`, `root_descriptor` with
   * optional v2+ `table_byte_offset`, `static_sampler`, `inline_sampler`, `sampler_heap`,
   * `resource_heap`, `cbv_indexed_*_heap`, `lane_indexed_*_heap`, v3 `unresolved` for read-only
   * kinds, and `rt_arena`), stage-level `mesh_output` and `rt_arena`, compacts parsed stage entries
   * with parse-local string/source/layout interning while discarding raw JSON wrappers after
   * indexing, and builds O(1) normalized-hash & zero-alloc `BigInt`/safe-int lookup indexes.
   *
   * @param {string | object} jsonOrObject
   * @param {{ maxNegativeLookupCacheSize?: number }} [options]
   * @returns {object}
   */
  function parseStageReflectionSidecar(jsonOrObject, options = {}) {
    if (options !== undefined && !isPlainObject(options)) {
      throw new TypeError('parseStageReflectionSidecar options must be an object');
    }
    const maxNegativeCacheSize =
      options && options.maxNegativeLookupCacheSize !== undefined
        ? options.maxNegativeLookupCacheSize
        : DEFAULT_NEGATIVE_LOOKUP_CACHE_SIZE;
    if (!Number.isInteger(maxNegativeCacheSize) || maxNegativeCacheSize < 0) {
      throw new RangeError(
        `maxNegativeLookupCacheSize must be a non-negative integer, got ${maxNegativeCacheSize}`
      );
    }
    let raw;
    if (typeof jsonOrObject === 'string') {
      try {
        raw = JSON.parse(jsonOrObject);
      } catch (err) {
        throw new ManifestValidationError(
          `malformed JSON: ${err instanceof Error ? err.message : String(err)}`
        );
      }
    } else {
      raw = jsonOrObject;
    }

    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('stage reflection sidecar root must be a JSON object');
    }
    if (raw.schema !== STAGE_REFLECTION_SCHEMA) {
      throw new ManifestValidationError(
        `unsupported stage reflection schema ${JSON.stringify(raw.schema)} (expected ${JSON.stringify(STAGE_REFLECTION_SCHEMA)})`,
        'schema'
      );
    }
    if (
      !Number.isInteger(raw.version) ||
      raw.version < STAGE_REFLECTION_MIN_VERSION ||
      raw.version > STAGE_REFLECTION_MAX_VERSION
    ) {
      throw new ManifestValidationError(
        `unsupported stage reflection version ${JSON.stringify(raw.version)} (expected ${STAGE_REFLECTION_MIN_VERSION}..${STAGE_REFLECTION_MAX_VERSION})`,
        'version'
      );
    }
    const sidecarVersion = raw.version;
    if (!isPlainObject(raw.stages)) {
      throw new ManifestValidationError('stages must be an object', 'stages');
    }

    const stages = new Map();
    const stagesByHash = new Map();
    const stagesByBareHex = new Map();
    const stagesBySafeInt = new Map();

    // Parse-local interning pools: deduplicate repeated strings, sources, layout entries,
    // and bind group layouts across 9,574+ stages, then become immediately GC-eligible.
    const internStr = createCompactStringInterner();
    const sourcePool = new Map();
    const layoutEntryPool = new Map();
    const bglPool = new Map();
    const bglIdsPool = new Map();

    let totalSamplerHeapBindings = 0;
    let totalResourceHeapStages = 0;
    let totalResourceHeapBindings = 0;
    let totalInlineSamplerStages = 0;
    let totalInlineSamplerBindings = 0;
    let totalVertexIndexStages = 0;
    let totalUnresolvedBindings = 0;
    let totalCbvIndexedHeapBindings = 0;
    let totalLaneIndexedHeapBindings = 0;
    let totalMeshOutputStages = 0;
    let totalRtArenaStages = 0;

    let rawStages = raw.stages;
    const seenSlots = new Set();
    const groupEntriesMap = new Map();

    for (const rawHash in rawStages) {
      if (!hasOwn(rawStages, rawHash)) continue;
      const rawStage = rawStages[rawHash];
      const stPath = `stages[${JSON.stringify(rawHash)}]`;
      const { hash, hashHex } = validateHashKey(rawHash, stPath);
      const bareHashHex = hashHex.slice(2);
      if (stagesByHash.has(hash)) {
        throw new ManifestValidationError(`duplicate stage hash ${hashHex}`, stPath);
      }
      if (!isPlainObject(rawStage)) {
        throw new ManifestValidationError('stage record must be an object', stPath);
      }

      const stage = normalizeDiagnosticStage(rawStage.stage);
      if (!stage) {
        throw new ManifestValidationError(
          `unsupported shader stage ${JSON.stringify(rawStage.stage)} (expected "vertex", "fragment", "pixel", or "compute")`,
          `${stPath}.stage`
        );
      }
      const visibility = STAGE_TO_VISIBILITY[stage];

      const rawMeshOutput =
        rawStage.mesh_output !== undefined ? rawStage.mesh_output : rawStage.meshOutput;
      let meshOutput = null;
      if (rawMeshOutput !== undefined && rawMeshOutput !== null) {
        meshOutput = parseMeshOutputLayout(rawMeshOutput, `${stPath}.mesh_output`);
      }

      let rtArenaMask = 0;
      if (rawStage.rt_arena !== undefined) {
        if (typeof rawStage.rt_arena !== 'boolean') {
          throw new ManifestValidationError('rt_arena must be a boolean', `${stPath}.rt_arena`);
        }
        if (rawStage.rt_arena === true) {
          rtArenaMask = RT_ARENA_FULL_MASK;
        }
      }

      if (!Array.isArray(rawStage.bindings)) {
        throw new ManifestValidationError('bindings must be an array', `${stPath}.bindings`);
      }

      seenSlots.clear();
      groupEntriesMap.clear();
      const rawBindings = rawStage.bindings;
      const bindings = [];
      let maxGroupSeen = -1;
      let stageSamplerHeapBindings = 0;
      let stageResourceHeapBindings = 0;
      let stageInlineSamplerBindings = 0;
      let stageUnresolvedBindings = 0;
      let stageCbvIndexedHeapBindings = 0;
      let stageLaneIndexedHeapBindings = 0;
      let layoutComplete = true;

      for (let i = 0; i < rawBindings.length; i += 1) {
        const b = rawBindings[i];
        const bPath = `${stPath}.bindings[${i}]`;
        if (!isPlainObject(b)) {
          throw new ManifestValidationError('binding record must be an object', bPath);
        }
        if (!isNonNegativeInt(b.group)) {
          throw new ManifestValidationError(`invalid group ${JSON.stringify(b.group)}`, `${bPath}.group`);
        }
        if (!isNonNegativeInt(b.binding)) {
          throw new ManifestValidationError(`invalid binding ${JSON.stringify(b.binding)}`, `${bPath}.binding`);
        }

        // Runtime-owned RT arena bindings (group 3, bindings 16..21): fold into rtArenaMask, not joined.
        if (isPlainObject(b.source) && b.source.rt_arena === true) {
          parseBindingSource(b.source, `${bPath}.source`, { version: sidecarVersion });
          if (
            b.group !== RT_ARENA_BIND_GROUP ||
            b.binding < RT_ARENA_FIRST_BINDING ||
            b.binding > RT_ARENA_LAST_BINDING
          ) {
            throw new ManifestValidationError(
              `an rt_arena binding must be group ${RT_ARENA_BIND_GROUP}, binding ${RT_ARENA_FIRST_BINDING}..${RT_ARENA_LAST_BINDING}`,
              bPath
            );
          }
          rtArenaMask |= 1 << (b.binding - RT_ARENA_FIRST_BINDING);
          continue;
        }

        const slotKey =
          b.group < 65536 && b.binding < 65536
            ? b.group * 65536 + b.binding
            : `${b.group}:${b.binding}`;
        if (seenSlots.has(slotKey)) {
          throw new ManifestValidationError(
            `duplicate binding at @group(${b.group}) @binding(${b.binding})`,
            bPath
          );
        }
        seenSlots.add(slotKey);
        if (b.name !== undefined && b.name !== null && typeof b.name !== 'string') {
          throw new ManifestValidationError('name must be a string or null', `${bPath}.name`);
        }
        if (typeof b.kind !== 'string') {
          throw new ManifestValidationError(
            `kind must be a string, got ${JSON.stringify(b.kind)}`,
            `${bPath}.kind`
          );
        }
        const kind = STAGE_REFLECTION_KIND_MAP[b.kind.trim().toLowerCase()];
        if (!kind) {
          throw new ManifestValidationError(
            `unknown binding kind ${JSON.stringify(b.kind)}`,
            `${bPath}.kind`
          );
        }
        if (b.source === undefined || b.source === null) {
          throw new ManifestValidationError('binding source is required in stage reflection sidecar', `${bPath}.source`);
        }
        const source = internNormalizedBindingSource(
          parseBindingSource(b.source, `${bPath}.source`, {
            version: sidecarVersion,
            bindingKind: kind,
          }),
          sourcePool
        );

        let minBindingSizeBytes = null;
        const rawMinSize =
          b.min_binding_size_bytes !== undefined
            ? b.min_binding_size_bytes
            : b.minBindingSizeBytes;
        if (rawMinSize !== undefined && rawMinSize !== null) {
          if (!isNonNegativeInt(rawMinSize)) {
            throw new ManifestValidationError(
              `min_binding_size_bytes must be a non-negative uint32 safe integer, got ${JSON.stringify(rawMinSize)}`,
              `${bPath}.min_binding_size_bytes`
            );
          }
          minBindingSizeBytes = rawMinSize;
        }

        let storageFormat = null;
        const rawStorageFormat =
          b.storage_format !== undefined ? b.storage_format : b.storageFormat;
        if (rawStorageFormat !== undefined && rawStorageFormat !== null) {
          if (
            (typeof rawStorageFormat !== 'string' || rawStorageFormat.length === 0) &&
            !isNonNegativeInt(rawStorageFormat)
          ) {
            throw new ManifestValidationError(
              `storage_format must be a non-empty string or non-negative integer, got ${JSON.stringify(rawStorageFormat)}`,
              `${bPath}.storage_format`
            );
          }
          storageFormat = typeof rawStorageFormat === 'string' ? internStr(rawStorageFormat) : rawStorageFormat;
        }

        const rawViewDim = b.view_dimension !== undefined ? b.view_dimension : b.viewDimension;
        const viewDimension =
          kind === 'texture' || kind === 'storage_texture'
            ? normalizeSidecarViewDimension(rawViewDim, kind === 'storage_texture', `${bPath}.view_dimension`)
            : null;
        const rawSampleType = b.sample_type !== undefined ? b.sample_type : b.sampleType;
        const sampleType =
          kind === 'texture'
            ? normalizeSidecarSampleType(rawSampleType, `${bPath}.sample_type`)
            : null;
        if (b.multisampled !== undefined && typeof b.multisampled !== 'boolean') {
          throw new ManifestValidationError('multisampled must be a boolean', `${bPath}.multisampled`);
        }
        const multisampled = Boolean(b.multisampled);
        if (kind === 'texture' && multisampled && viewDimension !== '2d') {
          throw new ManifestValidationError(
            `multisampled texture must have view_dimension "2d", got ${JSON.stringify(viewDimension)}`,
            `${bPath}.view_dimension`
          );
        }

        if (source.kind === 'sampler_heap') {
          stageSamplerHeapBindings += 1;
        } else if (source.kind === 'resource_heap') {
          stageResourceHeapBindings += 1;
        } else if (source.kind === 'inline_sampler') {
          stageInlineSamplerBindings += 1;
        } else if (source.kind === 'unresolved') {
          stageUnresolvedBindings += 1;
        } else if (
          source.kind === 'cbv_indexed_resource_heap' ||
          source.kind === 'cbv_indexed_sampler_heap'
        ) {
          stageCbvIndexedHeapBindings += 1;
        } else if (
          source.kind === 'lane_indexed_resource_heap' ||
          source.kind === 'lane_indexed_sampler_heap'
        ) {
          stageLaneIndexedHeapBindings += 1;
        }

        const isRootConsts = source.kind === 'root_constants';
        const bindingArray =
          source.kind === 'lane_indexed_resource_heap' ||
          source.kind === 'lane_indexed_sampler_heap'
            ? Object.freeze({ count: source.maxCount })
            : null;
        const rec = {
          group: b.group,
          binding: b.binding,
          name: b.name !== undefined && b.name !== null ? internStr(b.name) : null,
          kind,
          resource: sidecarKindToResource(kind),
          in_layout: true,
          inLayout: true,
          register: null,
          spirv: null,
          root_constants: isRootConsts,
          rootConstants: isRootConsts,
          binding_array: bindingArray,
          bindingArray,
          source,
          view_dimension: viewDimension,
          viewDimension,
          sample_type: sampleType,
          sampleType,
          multisampled,
          min_binding_size_bytes: minBindingSizeBytes,
          minBindingSizeBytes,
          storage_format: storageFormat,
          storageFormat,
          layoutEntry: null,
        };

        const layoutEntry = getOrCreateCompactSidecarLayoutEntry(rec, visibility, layoutEntryPool);
        rec.layoutEntry = layoutEntry;
        if (layoutEntry) {
          if (rec.group > maxGroupSeen) {
            maxGroupSeen = rec.group;
          }
          let gEntries = groupEntriesMap.get(rec.group);
          if (!gEntries) {
            gEntries = [];
            groupEntriesMap.set(rec.group, gEntries);
          }
          gEntries.push(layoutEntry);
        } else {
          layoutComplete = false;
        }

        bindings.push(rec);
      }

      const finalBindings = bindings.length > 0 ? bindings : EMPTY_ARRAY;

      let resolvedGroupLayouts = EMPTY_ARRAY;
      let bindGroupLayoutIds = EMPTY_ARRAY;
      let pipelineLayoutId = null;

      if (maxGroupSeen >= 0) {
        if (maxGroupSeen >= MAX_REFLECTED_GROUPS) {
          throw new ManifestValidationError(
            `bind group index ${maxGroupSeen} exceeds reflection limit ${MAX_REFLECTED_GROUPS}`,
            `${stPath}.bindings`
          );
        }
        const groupCount = maxGroupSeen + 1;
        resolvedGroupLayouts = new Array(groupCount);
        const rawIds = new Array(groupCount);
        for (let g = 0; g < groupCount; g += 1) {
          const gEntries = groupEntriesMap.get(g);
          let canonicalEntriesJson;
          let sortedEntries;
          if (!gEntries || gEntries.length === 0) {
            sortedEntries = EMPTY_ARRAY;
            canonicalEntriesJson = '[]';
          } else {
            if (gEntries.length > 1) {
              gEntries.sort((a, b) => a.binding - b.binding);
            }
            sortedEntries = gEntries;
            canonicalEntriesJson = `[${gEntries.map(serializeLayoutEntryJson).join(',')}]`;
          }

          const bglKey = `${g}|${canonicalEntriesJson}`;
          let cachedBgl = bglPool.get(bglKey);
          if (cachedBgl === undefined) {
            const id = fnv1a64(`bgl:${canonicalEntriesJson}`);
            const idHex = formatShaderHash(id);
            if (sortedEntries !== EMPTY_ARRAY) {
              Object.freeze(sortedEntries);
            }
            cachedBgl = Object.freeze({ group: g, id, idHex, entries: sortedEntries });
            bglPool.set(bglKey, cachedBgl);
          }
          rawIds[g] = cachedBgl.idHex;
          resolvedGroupLayouts[g] = cachedBgl;
        }

        const idsKey = rawIds.join(',');
        let cachedPl = bglIdsPool.get(idsKey);
        if (cachedPl === undefined) {
          Object.freeze(rawIds);
          const plId = computePipelineLayoutHash(rawIds);
          cachedPl = { ids: rawIds, pipelineLayoutId: plId };
          bglIdsPool.set(idsKey, cachedPl);
        }
        bindGroupLayoutIds = cachedPl.ids;
        pipelineLayoutId = cachedPl.pipelineLayoutId;
      }

      const rawVertexInputs =
        rawStage.vertex_inputs !== undefined ? rawStage.vertex_inputs : rawStage.vertexInputs;
      let vertexInputs = EMPTY_ARRAY;
      if (rawVertexInputs !== undefined && rawVertexInputs !== null) {
        if (!Array.isArray(rawVertexInputs)) {
          throw new ManifestValidationError('vertex_inputs must be an array', `${stPath}.vertex_inputs`);
        }
        if (rawVertexInputs.length > 0) {
          vertexInputs = new Array(rawVertexInputs.length);
          for (let i = 0; i < rawVertexInputs.length; i += 1) {
            vertexInputs[i] = parseSidecarVertexInput(
              rawVertexInputs[i],
              `${stPath}.vertex_inputs[${i}]`,
              internStr
            );
          }
          if (vertexInputs.length > 1) {
            vertexInputs.sort((a, b) => a.location - b.location);
          }
        }
      }

      const vsOutputs = parseStageParamList(
        rawStage.vs_outputs !== undefined ? rawStage.vs_outputs : rawStage.vsOutputs,
        'vs_outputs',
        stage,
        'vertex',
        `${stPath}.vs_outputs`
      );
      const psInputs = parseStageParamList(
        rawStage.ps_inputs !== undefined ? rawStage.ps_inputs : rawStage.psInputs,
        'ps_inputs',
        stage,
        'fragment',
        `${stPath}.ps_inputs`
      );

      const resolvedVim = resolveVertexInputMode(rawStage, stage, vertexInputs, stPath);
      const vertexInputMode = resolvedVim !== null ? internStr(resolvedVim) : null;
      const isVertexIndexStage = stage === 'vertex' && vertexInputMode === 'vertex_index';
      const hasInlineSampler = stageInlineSamplerBindings > 0;
      const hasResourceHeap = stageResourceHeapBindings > 0;
      const hasMeshOutput = meshOutput !== null;
      const hasRtArena = rtArenaMask !== 0;

      totalSamplerHeapBindings += stageSamplerHeapBindings;
      totalResourceHeapBindings += stageResourceHeapBindings;
      totalInlineSamplerBindings += stageInlineSamplerBindings;
      totalUnresolvedBindings += stageUnresolvedBindings;
      totalCbvIndexedHeapBindings += stageCbvIndexedHeapBindings;
      totalLaneIndexedHeapBindings += stageLaneIndexedHeapBindings;
      if (hasInlineSampler) {
        totalInlineSamplerStages += 1;
      }
      if (hasResourceHeap) {
        totalResourceHeapStages += 1;
      }
      if (isVertexIndexStage) {
        totalVertexIndexStages += 1;
      }
      if (hasMeshOutput) {
        totalMeshOutputStages += 1;
      }
      if (hasRtArena) {
        totalRtArenaStages += 1;
      }

      const entryPoint =
        typeof rawStage.entry_point === 'string' && rawStage.entry_point.length > 0
          ? internStr(rawStage.entry_point)
          : typeof rawStage.entryPoint === 'string' && rawStage.entryPoint.length > 0
            ? internStr(rawStage.entryPoint)
            : stage === 'vertex'
              ? CANONICAL_ENTRY_POINTS.vs_main
              : stage === 'fragment'
                ? CANONICAL_ENTRY_POINTS.fs_main
                : CANONICAL_ENTRY_POINTS.cs_main;

      const stageName =
        typeof rawStage.name === 'string' && rawStage.name.length > 0
          ? internStr(rawStage.name)
          : null;

      const stageRecord = {
        hash,
        hashHex,
        bareHashHex,
        stage,
        visibility,
        entryPoint,
        entry_point: entryPoint,
        name: stageName,
        pipelineLayoutId,
        pipeline_layout: pipelineLayoutId,
        bindGroupLayoutIds,
        bind_group_layouts: bindGroupLayoutIds,
        bindGroupLayouts: resolvedGroupLayouts,
        layoutComplete,
        layout_complete: layoutComplete,
        workgroupSize: null,
        workgroup_size: null,
        bindings: finalBindings,
        vertexInputs,
        vertex_inputs: vertexInputs,
        vertexInputMode,
        vertex_input_mode: vertexInputMode,
        isVertexIndexStage,
        fragmentOutputs: EMPTY_ARRAY,
        fragment_outputs: EMPTY_ARRAY,
        vsOutputs,
        vs_outputs: vsOutputs,
        psInputs,
        ps_inputs: psInputs,
        requiresLowering: EMPTY_ARRAY,
        requires_lowering: EMPTY_ARRAY,
        limitViolations: EMPTY_ARRAY,
        limit_violations: EMPTY_ARRAY,
        samplerHeapBindings: stageSamplerHeapBindings,
        resourceHeapBindings: stageResourceHeapBindings,
        inlineSamplerBindings: stageInlineSamplerBindings,
        unresolvedBindings: stageUnresolvedBindings,
        cbvIndexedHeapBindings: stageCbvIndexedHeapBindings,
        laneIndexedHeapBindings: stageLaneIndexedHeapBindings,
        hasInlineSampler,
        hasResourceHeap,
        hasMeshOutput,
        meshOutput,
        mesh_output: meshOutput,
        hasRtArena,
        rtArenaMask,
        rt_arena_mask: rtArenaMask,
      };

      stages.set(hashHex, stageRecord);
      stagesByHash.set(hash, stageRecord);
      stagesByBareHex.set(bareHashHex, stageRecord);
      if (hash <= MAX_SAFE_INTEGER_BIGINT) {
        stagesBySafeInt.set(Number(hash), stageRecord);
      }
    }

    // Drop raw payload references and parse-local scratch collections before building the index view.
    raw = null;
    rawStages = null;
    seenSlots.clear();
    groupEntriesMap.clear();
    sourcePool.clear();
    layoutEntryPool.clear();
    bglPool.clear();
    bglIdsPool.clear();

    return createParsedStageSidecar(
      sidecarVersion,
      stages,
      stagesByHash,
      stagesByBareHex,
      stagesBySafeInt,
      {
        inlineSamplerStages: totalInlineSamplerStages,
        inlineSamplerBindings: totalInlineSamplerBindings,
        resourceHeapStages: totalResourceHeapStages,
        resourceHeapBindings: totalResourceHeapBindings,
        samplerHeapBindings: totalSamplerHeapBindings,
        vertexIndexStages: totalVertexIndexStages,
        unresolvedBindings: totalUnresolvedBindings,
        cbvIndexedHeapBindings: totalCbvIndexedHeapBindings,
        laneIndexedHeapBindings: totalLaneIndexedHeapBindings,
        meshOutputStages: totalMeshOutputStages,
        rtArenaStages: totalRtArenaStages,
      },
      maxNegativeCacheSize
    );
  }

  // ---------------------------------------------------------------------------
  // Multi-Stage BindGroupLayout Merging (matching shaders/src/reflect.rs)
  // ---------------------------------------------------------------------------

  /**
   * Merge two `GPUBindGroupLayoutEntry` layouts at the same `(group, binding)` following
   * the exact rules of `merge_layouts` in `shaders/src/reflect.rs`:
   * - Resource member kinds must match (`buffer`, `sampler`, `texture`, `storageTexture`, `externalTexture`).
   * - Buffers: `type` & `hasDynamicOffset` must match; `minBindingSize` takes `Math.max`.
   * - Samplers: `type` must match.
   * - Textures: `viewDimension` & `multisampled` must match; `"float"` wins over `"unfilterable-float"`.
   * - Storage textures: `access`, `format`, and `viewDimension` must match.
   *
   * Returns `{ merged, reason }` where `merged` is `null` on conflict.
   */
  function mergeLayoutEntries(existing, incoming) {
    const existingKind = getLayoutMemberKind(existing).kind;
    const incomingKind = getLayoutMemberKind(incoming).kind;

    if (existingKind !== incomingKind) {
      return {
        merged: null,
        reason: `resource kind mismatch: ${existingKind} vs ${incomingKind}`,
      };
    }

    const visibility = existing.visibility | incoming.visibility;

    if (existingKind === 'buffer') {
      const a = existing.buffer;
      const b = incoming.buffer;
      if (a.type !== b.type) {
        return {
          merged: null,
          reason: `buffer type mismatch: ${a.type} vs ${b.type}`,
        };
      }
      if (Boolean(a.hasDynamicOffset) !== Boolean(b.hasDynamicOffset)) {
        return {
          merged: null,
          reason: `buffer hasDynamicOffset mismatch: ${Boolean(a.hasDynamicOffset)} vs ${Boolean(b.hasDynamicOffset)}`,
        };
      }
      return {
        merged: {
          binding: existing.binding,
          visibility,
          buffer: {
            type: a.type,
            hasDynamicOffset: Boolean(a.hasDynamicOffset),
            minBindingSize: Math.max(a.minBindingSize || 0, b.minBindingSize || 0),
          },
        },
        reason: null,
      };
    }

    if (existingKind === 'sampler') {
      if (existing.sampler.type !== incoming.sampler.type) {
        return {
          merged: null,
          reason: `sampler type mismatch: ${existing.sampler.type} vs ${incoming.sampler.type}`,
        };
      }
      return {
        merged: {
          binding: existing.binding,
          visibility,
          sampler: { type: existing.sampler.type },
        },
        reason: null,
      };
    }

    if (existingKind === 'texture') {
      const a = existing.texture;
      const b = incoming.texture;
      if (a.viewDimension !== b.viewDimension) {
        return {
          merged: null,
          reason: `texture viewDimension mismatch: ${a.viewDimension} vs ${b.viewDimension}`,
        };
      }
      if (Boolean(a.multisampled) !== Boolean(b.multisampled)) {
        return {
          merged: null,
          reason: `texture multisampled mismatch: ${Boolean(a.multisampled)} vs ${Boolean(b.multisampled)}`,
        };
      }
      let sampleType = a.sampleType;
      if (a.sampleType !== b.sampleType) {
        const floatPair =
          (a.sampleType === 'float' && b.sampleType === 'unfilterable-float') ||
          (a.sampleType === 'unfilterable-float' && b.sampleType === 'float');
        if (!floatPair) {
          return {
            merged: null,
            reason: `texture sampleType mismatch: ${a.sampleType} vs ${b.sampleType}`,
          };
        }
        sampleType = 'float';
      }
      return {
        merged: {
          binding: existing.binding,
          visibility,
          texture: {
            sampleType,
            viewDimension: a.viewDimension,
            multisampled: Boolean(a.multisampled),
          },
        },
        reason: null,
      };
    }

    if (existingKind === 'storageTexture') {
      const a = existing.storageTexture;
      const b = incoming.storageTexture;
      if (a.access !== b.access || a.format !== b.format || a.viewDimension !== b.viewDimension) {
        return {
          merged: null,
          reason: `storageTexture mismatch: (${a.access}, ${a.format}, ${a.viewDimension}) vs (${b.access}, ${b.format}, ${b.viewDimension})`,
        };
      }
      return {
        merged: {
          binding: existing.binding,
          visibility,
          storageTexture: {
            access: a.access,
            format: a.format,
            viewDimension: a.viewDimension,
          },
        },
        reason: null,
      };
    }

    if (existingKind === 'externalTexture') {
      return {
        merged: {
          binding: existing.binding,
          visibility,
          externalTexture: {},
        },
        reason: null,
      };
    }

    return {
      merged: null,
      reason: `unsupported layout entry kind: ${existingKind}`,
    };
  }

  /**
   * Extract normalized `{ group, entry, side }` tuples from a shader reflection object.
   * Supports both parsed S6 `ShaderReflection` records (with `bindGroupLayouts`) and
   * standalone/inline reflection objects (with `bindings` containing WebGPU resource members).
   */
  function extractShaderLayoutSlots(reflection) {
    if (!isPlainObject(reflection)) {
      throw new TypeError('Shader reflection item must be an object');
    }

    const stage =
      typeof reflection.stage === 'string'
        ? validateStageName(reflection.stage, 'reflection.stage')
        : null;
    const defaultVisibility =
      typeof reflection.visibility === 'number'
        ? reflection.visibility
        : stage
          ? STAGE_TO_VISIBILITY[stage]
          : undefined;
    const entryPoint =
      reflection.entryPoint || reflection.entry_point || reflection.name || '<unnamed>';

    // Build optional name lookup from S6 `bindings` metadata table.
    const bindingNameBySlot = new Map();
    if (Array.isArray(reflection.bindings)) {
      for (let i = 0; i < reflection.bindings.length; i += 1) {
        const b = reflection.bindings[i];
        if (b && isNonNegativeInt(b.group) && isNonNegativeInt(b.binding) && typeof b.name === 'string') {
          bindingNameBySlot.set(`${b.group}:${b.binding}`, b.name);
        }
      }
    }

    const slots = [];

    // Primary path: resolved S6 `bindGroupLayouts` array (indexed by group 0..N-1).
    if (Array.isArray(reflection.bindGroupLayouts)) {
      for (let idx = 0; idx < reflection.bindGroupLayouts.length; idx += 1) {
        const bgl = reflection.bindGroupLayouts[idx];
        if (!bgl) continue;
        const group = isNonNegativeInt(bgl.group) ? bgl.group : idx;
        const rawEntries = Array.isArray(bgl.entries) ? bgl.entries : [];
        for (let j = 0; j < rawEntries.length; j += 1) {
          const entry = normalizeLayoutEntry(
            rawEntries[j],
            `reflection(${entryPoint}).bindGroupLayouts[${group}].entries[${j}]`,
            defaultVisibility
          );
          const name = bindingNameBySlot.get(`${group}:${entry.binding}`) || null;
          slots.push({
            group,
            entry,
            side: {
              stage: stage || 'unknown',
              entryPoint,
              name,
              layoutJson: serializeLayoutMemberJson(entry),
            },
          });
        }
      }
      return slots;
    }

    // Fallback path: inline `bindings` array where each element carries a WebGPU resource member.
    if (Array.isArray(reflection.bindings)) {
      for (let i = 0; i < reflection.bindings.length; i += 1) {
        const raw = reflection.bindings[i];
        if (!isPlainObject(raw)) {
          throw new ManifestValidationError('binding item must be an object', `bindings[${i}]`);
        }
        if (raw.in_layout === false || raw.inLayout === false) {
          continue;
        }
        const group = raw.group !== undefined ? raw.group : 0;
        if (!isNonNegativeInt(group)) {
          throw new ManifestValidationError(`invalid group ${JSON.stringify(group)}`, `bindings[${i}].group`);
        }
        const source = isPlainObject(raw.layoutEntry) ? raw.layoutEntry : raw;
        const entry = normalizeLayoutEntry(
          source,
          `reflection(${entryPoint}).bindings[${i}]`,
          defaultVisibility
        );
        slots.push({
          group,
          entry,
          side: {
            stage: stage || 'unknown',
            entryPoint,
            name: typeof raw.name === 'string' ? raw.name : null,
            layoutJson: serializeLayoutMemberJson(entry),
          },
        });
      }
      return slots;
    }

    throw new ManifestValidationError(
      `shader reflection '${entryPoint}' has neither bindGroupLayouts nor bindings`
    );
  }

  /**
   * Merge one or more shader reflection objects into WebGPU `GPUBindGroupLayoutDescriptor[]`,
   * ordered densely by group (`0..maxGroup`, within `0..maxBindGroups-1`) with entries
   * sorted ascending by `binding`.
   *
   * @param {Array<object> | object} reflections
   * @param {{ maxBindGroups?: number, requireLayoutComplete?: boolean }} [options]
   * @returns {Array<{ entries: Array<object> }>}
   */
  function buildBindGroupLayoutDescriptors(reflections, options = {}) {
    const list = Array.isArray(reflections)
      ? reflections
      : reflections && typeof reflections[Symbol.iterator] === 'function' && typeof reflections !== 'string'
        ? Array.from(reflections)
        : [reflections];

    const maxBindGroups =
      options.maxBindGroups !== undefined ? options.maxBindGroups : MAX_WEBGPU_BIND_GROUPS;
    if (!Number.isInteger(maxBindGroups) || maxBindGroups < 1 || maxBindGroups > MAX_REFLECTED_GROUPS) {
      throw new RangeError(
        `maxBindGroups must be an integer in 1..${MAX_REFLECTED_GROUPS}, got ${maxBindGroups}`
      );
    }

    const requireLayoutComplete = Boolean(options.requireLayoutComplete);
    const groups = new Map();
    let maxGroupSeen = -1;

    for (let i = 0; i < list.length; i += 1) {
      const refl = list[i];
      if (!isPlainObject(refl)) {
        throw new TypeError(`Shader reflection at index ${i} must be an object`);
      }
      if (requireLayoutComplete && (refl.layoutComplete === false || refl.layout_complete === false)) {
        const label = refl.hashHex || refl.entryPoint || refl.entry_point || `#${i}`;
        throw new ReflectionError(
          `Shader reflection ${label} has layout_complete=false and requires offline lowering before creating a WebGPU layout`,
          'INCOMPLETE_LAYOUT',
          { shader: label, requiresLowering: refl.requiresLowering || refl.requires_lowering || [] }
        );
      }

      const slots = extractShaderLayoutSlots(refl);
      for (let s = 0; s < slots.length; s += 1) {
        const { group, entry, side } = slots[s];
        if (group >= maxBindGroups) {
          throw new ReflectionError(
            `Bind group index ${group} exceeds WebGPU maxBindGroups limit (${maxBindGroups}, valid range 0..${maxBindGroups - 1})`,
            'GROUP_INDEX_OUT_OF_RANGE',
            { group, maxBindGroups }
          );
        }
        if (group > maxGroupSeen) {
          maxGroupSeen = group;
        }

        let groupMap = groups.get(group);
        if (!groupMap) {
          groupMap = new Map();
          groups.set(group, groupMap);
        }

        const existing = groupMap.get(entry.binding);
        if (!existing) {
          groupMap.set(entry.binding, {
            entry: cloneLayoutEntry(entry),
            side,
          });
        } else {
          const { merged, reason } = mergeLayoutEntries(existing.entry, entry);
          if (!merged) {
            throw new BindingConflictError(group, entry.binding, existing.side, side, reason);
          }
          existing.entry = merged;
        }
      }
    }

    if (maxGroupSeen < 0) {
      return [];
    }

    const descriptors = new Array(maxGroupSeen + 1);
    for (let g = 0; g <= maxGroupSeen; g += 1) {
      const groupMap = groups.get(g);
      if (!groupMap || groupMap.size === 0) {
        descriptors[g] = { entries: [] };
      } else {
        const entries = Array.from(groupMap.values())
          .map((item) => item.entry)
          .sort((a, b) => a.binding - b.binding);
        descriptors[g] = { entries };
      }
    }

    return descriptors;
  }

  // ---------------------------------------------------------------------------
  // Vertex Buffer Layout Builder
  // ---------------------------------------------------------------------------

  /**
   * Convert reflected vertex inputs (`[{ location, format, type? }]` or a vertex shader
   * reflection object) into WebGPU `GPUVertexBufferLayout[]`.
   *
   * @param {object | Array<object>} vertexReflection
   * @param {number | { stride?: number, stepMode?: 'vertex' | 'instance', formatOverrides?: Record<number, string>, attributeOffsets?: Record<number, number> }} [strideOrOptions]
   * @param {{ stepMode?: 'vertex' | 'instance', formatOverrides?: Record<number, string>, attributeOffsets?: Record<number, number> }} [extraOptions]
   * @returns {Array<{ arrayStride: number, stepMode: string, attributes: Array<{ shaderLocation: number, offset: number, format: string }> }>}
   */
  function buildVertexBufferLayouts(vertexReflection, strideOrOptions, extraOptions) {
    let rawInputs;
    if (Array.isArray(vertexReflection)) {
      rawInputs = vertexReflection;
    } else if (isPlainObject(vertexReflection)) {
      if (Array.isArray(vertexReflection.vertexInputs)) {
        rawInputs = vertexReflection.vertexInputs;
      } else if (Array.isArray(vertexReflection.vertex_inputs)) {
        rawInputs = vertexReflection.vertex_inputs;
      } else {
        throw new VertexLayoutError(
          'vertexReflection must be an array of vertex inputs or a shader reflection object with vertex_inputs / vertexInputs'
        );
      }
    } else {
      throw new VertexLayoutError(
        'vertexReflection must be a vertex shader reflection object or an array of vertex inputs'
      );
    }

    let explicitStride;
    let opts = {};
    if (typeof strideOrOptions === 'number') {
      explicitStride = strideOrOptions;
      opts = isPlainObject(extraOptions) ? extraOptions : {};
    } else if (isPlainObject(strideOrOptions)) {
      opts = strideOrOptions;
      explicitStride = opts.stride;
    } else if (strideOrOptions !== undefined && strideOrOptions !== null) {
      throw new VertexLayoutError(
        `stride must be a non-negative integer or options object, got ${typeof strideOrOptions}`
      );
    }

    const stepMode = opts.stepMode !== undefined ? opts.stepMode : 'vertex';
    if (typeof stepMode !== 'string' || !VALID_STEP_MODES.has(stepMode)) {
      throw new VertexLayoutError(
        `Invalid vertex stepMode ${JSON.stringify(stepMode)} (expected "vertex" or "instance")`
      );
    }

    const formatOverrides = isPlainObject(opts.formatOverrides) ? opts.formatOverrides : null;
    const attributeOffsets = isPlainObject(opts.attributeOffsets) ? opts.attributeOffsets : null;

    if (rawInputs.length === 0) {
      if (explicitStride !== undefined) {
        if (!Number.isInteger(explicitStride) || explicitStride < 0) {
          throw new VertexLayoutError(`stride must be a non-negative integer, got ${explicitStride}`);
        }
      }
      return [];
    }

    const normalizedInputs = rawInputs.map((item, idx) => {
      if (!isPlainObject(item)) {
        throw new VertexLayoutError(`Vertex input at index ${idx} must be an object`);
      }
      const location =
        item.location !== undefined ? item.location : item.shaderLocation;
      if (!isNonNegativeInt(location)) {
        throw new VertexLayoutError(
          `Vertex input at index ${idx} has invalid location ${JSON.stringify(location)}`
        );
      }
      const overrideFmt =
        formatOverrides && Object.prototype.hasOwnProperty.call(formatOverrides, location)
          ? formatOverrides[location]
          : undefined;
      const format = overrideFmt !== undefined ? overrideFmt : item.format;
      if (typeof format !== 'string' || !Object.prototype.hasOwnProperty.call(VERTEX_FORMAT_BYTE_SIZES, format)) {
        throw new VertexLayoutError(
          `Vertex input @location(${location}) has unsupported or missing GPUVertexFormat: ${JSON.stringify(format)}`,
          { location, format, type: item.type || null }
        );
      }

      const overrideOffset =
        attributeOffsets && Object.prototype.hasOwnProperty.call(attributeOffsets, location)
          ? attributeOffsets[location]
          : item.offset;
      if (overrideOffset !== undefined && !isNonNegativeInt(overrideOffset)) {
        throw new VertexLayoutError(
          `Vertex input @location(${location}) has invalid byte offset ${JSON.stringify(overrideOffset)}`
        );
      }

      return {
        location,
        format,
        byteSize: VERTEX_FORMAT_BYTE_SIZES[format],
        explicitOffset: overrideOffset,
      };
    });

    normalizedInputs.sort((a, b) => a.location - b.location);

    const seenLocations = new Set();
    const attributes = new Array(normalizedInputs.length);
    let runningOffset = 0;
    let maxEndByte = 0;

    for (let i = 0; i < normalizedInputs.length; i += 1) {
      const inp = normalizedInputs[i];
      if (seenLocations.has(inp.location)) {
        throw new VertexLayoutError(`Duplicate vertex input @location(${inp.location})`, {
          location: inp.location,
        });
      }
      seenLocations.add(inp.location);

      const offset = inp.explicitOffset !== undefined ? inp.explicitOffset : runningOffset;
      const endByte = offset + inp.byteSize;
      if (endByte > maxEndByte) {
        maxEndByte = endByte;
      }
      if (inp.explicitOffset === undefined) {
        runningOffset = endByte;
      } else if (endByte > runningOffset) {
        runningOffset = endByte;
      }

      attributes[i] = {
        shaderLocation: inp.location,
        offset,
        format: inp.format,
      };
    }

    let arrayStride = maxEndByte;
    if (explicitStride !== undefined) {
      if (!Number.isInteger(explicitStride) || explicitStride < 0) {
        throw new VertexLayoutError(`stride must be a non-negative integer, got ${explicitStride}`);
      }
      if (explicitStride < maxEndByte) {
        throw new VertexLayoutError(
          `Explicit vertex stride (${explicitStride}) is smaller than required attribute extent (${maxEndByte} bytes)`,
          { stride: explicitStride, requiredBytes: maxEndByte }
        );
      }
      arrayStride = explicitStride;
    }

    return [
      {
        arrayStride,
        stepMode,
        attributes,
      },
    ];
  }

  // ---------------------------------------------------------------------------
  // Stage Reflection Join Diagnostics & Static-Sampler Fallback Metadata (W171)
  // ---------------------------------------------------------------------------

  const STAGE_REFLECTION_KIND_MAP = Object.freeze({
    uniform: 'uniform',
    storage: 'storage',
    read_only_storage: 'read_only_storage',
    'read-only-storage': 'read_only_storage',
    texture: 'texture',
    texture_2d: 'texture',
    storage_texture: 'storage_texture',
    'storage-texture': 'storage_texture',
    storagetexture: 'storage_texture',
    sampler: 'sampler',
    comparison_sampler: 'comparison_sampler',
    'comparison-sampler': 'comparison_sampler',
    sampler_comparison: 'comparison_sampler',
  });

  const ACCEPTED_DESCRIPTOR_CLASSES = Object.freeze({
    uniform: Object.freeze(['cbv', 'srv']),
    read_only_storage: Object.freeze(['srv', 'uav', 'cbv']),
    storage: Object.freeze(['uav']),
    storage_texture: Object.freeze(['uav']),
    texture: Object.freeze(['srv']),
    sampler: Object.freeze(['sampler']),
    comparison_sampler: Object.freeze(['sampler']),
  });

  const DESCRIPTOR_CLASS_LABELS = Object.freeze({
    cbv: 'CBV',
    srv: 'SRV',
    uav: 'UAV',
    sampler: 'sampler',
  });

  function formatDescriptorClass(cls) {
    return DESCRIPTOR_CLASS_LABELS[cls] || String(cls);
  }

  function normalizeDiagnosticStage(rawStage) {
    if (typeof rawStage !== 'string') {
      return null;
    }
    const lower = rawStage.trim().toLowerCase();
    if (lower === 'pixel') return 'fragment';
    if (lower === 'vertex' || lower === 'fragment' || lower === 'compute') return lower;
    return null;
  }

  function normalizeRootParameterType(rawType) {
    if (typeof rawType === 'number' && Number.isInteger(rawType)) {
      if (rawType === 0) return 'table';
      if (rawType === 1) return 'constants';
      if (rawType === 2) return 'cbv';
      if (rawType === 3) return 'srv';
      if (rawType === 4) return 'uav';
      return null;
    }
    if (typeof rawType !== 'string') return null;
    const lower = rawType.trim().toLowerCase();
    if (lower === 'table' || lower === 'descriptor_table' || lower === 'descriptortable') {
      return 'table';
    }
    if (
      lower === 'constants' ||
      lower === 'root_constants' ||
      lower === 'rootconstants' ||
      lower === 'constants32bit' ||
      lower === '32bit_constants'
    ) {
      return 'constants';
    }
    if (lower === 'cbv' || lower === 'root_cbv' || lower === 'rootcbv') return 'cbv';
    if (lower === 'srv' || lower === 'root_srv' || lower === 'rootsrv') return 'srv';
    if (lower === 'uav' || lower === 'root_uav' || lower === 'rootuav') return 'uav';
    return null;
  }

  function normalizeDescriptorRangeType(rawType) {
    if (typeof rawType === 'number' && Number.isInteger(rawType)) {
      if (rawType === 0) return 'srv';
      if (rawType === 1) return 'uav';
      if (rawType === 2) return 'cbv';
      if (rawType === 3) return 'sampler';
      return null;
    }
    if (typeof rawType !== 'string') return null;
    const lower = rawType.trim().toLowerCase();
    if (lower === 'srv' || lower === 't') return 'srv';
    if (lower === 'uav' || lower === 'u') return 'uav';
    if (lower === 'cbv' || lower === 'b') return 'cbv';
    if (lower === 'sampler' || lower === 's') return 'sampler';
    return null;
  }

  function normalizeDescriptorRange(rawRange, path) {
    if (!isPlainObject(rawRange)) {
      throw new TypeError(`${path} must be an object`);
    }
    const rawType =
      rawRange.type !== undefined
        ? rawRange.type
        : rawRange.range_type !== undefined
          ? rawRange.range_type
          : rawRange.rangeType;
    const type = normalizeDescriptorRangeType(rawType);
    if (!type) {
      throw new TypeError(`${path}.type has unsupported descriptor range type ${JSON.stringify(rawType)}`);
    }

    const rawCount =
      rawRange.count !== undefined
        ? rawRange.count
        : rawRange.num_descriptors !== undefined
          ? rawRange.num_descriptors
          : rawRange.numDescriptors !== undefined
            ? rawRange.numDescriptors
            : 1;
    let count;
    if (
      rawCount === -1 ||
      rawCount === Infinity ||
      rawCount === UNBOUNDED_DESCRIPTOR_COUNT ||
      rawCount === 'unbounded' ||
      rawCount === null
    ) {
      count = UNBOUNDED_DESCRIPTOR_COUNT;
    } else if (isNonNegativeInt(rawCount)) {
      count = rawCount;
    } else {
      throw new TypeError(`${path}.count must be a non-negative integer or unbounded (-1), got ${JSON.stringify(rawCount)}`);
    }

    const rawBaseReg =
      rawRange.base_register !== undefined
        ? rawRange.base_register
        : rawRange.base_shader_register !== undefined
          ? rawRange.base_shader_register
          : rawRange.baseShaderRegister !== undefined
            ? rawRange.baseShaderRegister
            : rawRange.register !== undefined
              ? rawRange.register
              : 0;
    if (!isNonNegativeInt(rawBaseReg)) {
      throw new TypeError(`${path}.base_register must be a non-negative integer, got ${JSON.stringify(rawBaseReg)}`);
    }

    const rawSpace =
      rawRange.space !== undefined
        ? rawRange.space
        : rawRange.register_space !== undefined
          ? rawRange.register_space
          : rawRange.registerSpace !== undefined
            ? rawRange.registerSpace
            : 0;
    if (!isNonNegativeInt(rawSpace)) {
      throw new TypeError(`${path}.space must be a non-negative integer, got ${JSON.stringify(rawSpace)}`);
    }

    const rawOffset =
      rawRange.offset !== undefined
        ? rawRange.offset
        : rawRange.offset_in_descriptors_from_table_start !== undefined
          ? rawRange.offset_in_descriptors_from_table_start
          : rawRange.offsetInDescriptorsFromTableStart;
    let offset;
    if (
      rawOffset === undefined ||
      rawOffset === null ||
      rawOffset === -1 ||
      rawOffset === DESCRIPTOR_RANGE_OFFSET_APPEND ||
      rawOffset === 'append'
    ) {
      offset = DESCRIPTOR_RANGE_OFFSET_APPEND;
    } else if (isNonNegativeInt(rawOffset)) {
      offset = rawOffset;
    } else {
      throw new TypeError(`${path}.offset must be a non-negative integer or append (-1), got ${JSON.stringify(rawOffset)}`);
    }

    return {
      type,
      count,
      baseRegister: rawBaseReg,
      space: rawSpace,
      offset,
    };
  }

  function normalizeRootSignatureSpec(rootSignatureSpec, defaultUserDataBase, options = {}) {
    if (!isPlainObject(rootSignatureSpec)) {
      throw new TypeError('rootSignatureSpec must be an object');
    }

    const rawUserDataBase =
      options.userDataBase !== undefined
        ? options.userDataBase
        : rootSignatureSpec.userDataBase !== undefined
          ? rootSignatureSpec.userDataBase
          : rootSignatureSpec.user_data_base !== undefined
            ? rootSignatureSpec.user_data_base
            : defaultUserDataBase;
    if (!isNonNegativeInt(rawUserDataBase)) {
      throw new RangeError(`userDataBase must be a non-negative integer, got ${JSON.stringify(rawUserDataBase)}`);
    }

    const rawTableSlotBytes =
      options.tableSlotBytes !== undefined
        ? options.tableSlotBytes
        : rootSignatureSpec.tableSlotBytes !== undefined
          ? rootSignatureSpec.tableSlotBytes
          : rootSignatureSpec.table_slot_bytes !== undefined
            ? rootSignatureSpec.table_slot_bytes
            : TABLE_SLOT_BYTES;
    if (!isNonNegativeInt(rawTableSlotBytes) || rawTableSlotBytes <= 0) {
      throw new RangeError(
        `tableSlotBytes must be a positive integer, got ${JSON.stringify(rawTableSlotBytes)}`
      );
    }

    const rawParams =
      rootSignatureSpec.parameters !== undefined
        ? rootSignatureSpec.parameters
        : rootSignatureSpec.root_parameters !== undefined
          ? rootSignatureSpec.root_parameters
          : rootSignatureSpec.rootParameters !== undefined
            ? rootSignatureSpec.rootParameters
            : [];
    if (!Array.isArray(rawParams)) {
      throw new TypeError('rootSignatureSpec.parameters must be an array');
    }

    const parameters = [];
    let currentSgpr = rawUserDataBase;
    for (let i = 0; i < rawParams.length; i += 1) {
      const rawParam = rawParams[i];
      const pPath = `rootSignatureSpec.parameters[${i}]`;
      if (!isPlainObject(rawParam)) {
        throw new TypeError(`${pPath} must be an object`);
      }
      const rawType =
        rawParam.type !== undefined
          ? rawParam.type
          : rawParam.parameter_type !== undefined
            ? rawParam.parameter_type
            : rawParam.parameterType;
      const type = normalizeRootParameterType(rawType);
      if (!type) {
        throw new TypeError(`${pPath}.type has unsupported root parameter type ${JSON.stringify(rawType)}`);
      }

      const rawExplicitSgpr =
        rawParam.user_sgpr !== undefined ? rawParam.user_sgpr : rawParam.userSgpr;
      let startSgpr = currentSgpr;
      if (rawExplicitSgpr !== undefined && rawExplicitSgpr !== null) {
        if (!isNonNegativeInt(rawExplicitSgpr)) {
          throw new TypeError(`${pPath}.user_sgpr must be a non-negative integer`);
        }
        startSgpr = rawExplicitSgpr;
      }

      let dwords;
      let num32BitValues = 0;
      let ranges = [];
      const shaderRegister =
        rawParam.shader_register !== undefined
          ? rawParam.shader_register
          : rawParam.shaderRegister !== undefined
            ? rawParam.shaderRegister
            : rawParam.register !== undefined
              ? rawParam.register
              : 0;
      const registerSpace =
        rawParam.register_space !== undefined
          ? rawParam.register_space
          : rawParam.registerSpace !== undefined
            ? rawParam.registerSpace
            : rawParam.space !== undefined
              ? rawParam.space
              : 0;

      if (type === 'table') {
        dwords = 1;
        const rawRanges =
          rawParam.ranges !== undefined
            ? rawParam.ranges
            : rawParam.descriptor_ranges !== undefined
              ? rawParam.descriptor_ranges
              : rawParam.descriptorRanges !== undefined
                ? rawParam.descriptorRanges
                : [];
        if (!Array.isArray(rawRanges)) {
          throw new TypeError(`${pPath}.ranges must be an array`);
        }
        ranges = rawRanges.map((r, rIdx) => normalizeDescriptorRange(r, `${pPath}.ranges[${rIdx}]`));
      } else if (type === 'constants') {
        const rawNum =
          rawParam.num_32bit_values !== undefined
            ? rawParam.num_32bit_values
            : rawParam.num32BitValues !== undefined
              ? rawParam.num32BitValues
              : rawParam.numValues !== undefined
                ? rawParam.numValues
                : rawParam.dwords !== undefined
                  ? rawParam.dwords
                  : rawParam.count !== undefined
                    ? rawParam.count
                    : 1;
        if (!isNonNegativeInt(rawNum)) {
          throw new TypeError(`${pPath}.num_32bit_values must be a non-negative integer, got ${JSON.stringify(rawNum)}`);
        }
        num32BitValues = rawNum;
        dwords = rawNum;
      } else {
        dwords = 2;
      }

      parameters.push({
        index: i,
        type,
        startSgpr,
        endSgpr: startSgpr + dwords,
        dwords,
        num32BitValues,
        shaderRegister: isNonNegativeInt(shaderRegister) ? shaderRegister : 0,
        registerSpace: isNonNegativeInt(registerSpace) ? registerSpace : 0,
        ranges,
      });
      currentSgpr = startSgpr + dwords;
    }

    const rawStaticSamplers =
      rootSignatureSpec.static_samplers !== undefined
        ? rootSignatureSpec.static_samplers
        : rootSignatureSpec.staticSamplers;
    let staticSamplerCount = 0;
    let staticSamplers = null;
    if (rawStaticSamplers === undefined || rawStaticSamplers === null) {
      staticSamplerCount = 0;
      staticSamplers = [];
    } else if (Array.isArray(rawStaticSamplers)) {
      staticSamplerCount = rawStaticSamplers.length;
      staticSamplers = rawStaticSamplers;
    } else if (isNonNegativeInt(rawStaticSamplers)) {
      staticSamplerCount = rawStaticSamplers;
      staticSamplers = null;
    } else {
      throw new TypeError(
        `rootSignatureSpec.static_samplers must be an array or non-negative integer, got ${JSON.stringify(rawStaticSamplers)}`
      );
    }

    return {
      userDataBase: rawUserDataBase,
      tableSlotBytes: rawTableSlotBytes,
      parameters,
      staticSamplerCount,
      staticSamplers,
    };
  }

  function resolveTableSlot(param, slot) {
    let next = 0;
    for (let i = 0; i < param.ranges.length; i += 1) {
      const r = param.ranges[i];
      const start = r.offset === DESCRIPTOR_RANGE_OFFSET_APPEND ? next : r.offset;
      const unbounded = r.count === UNBOUNDED_DESCRIPTOR_COUNT;
      if (slot >= start && (unbounded || slot < start + r.count)) {
        return {
          cls: r.type,
          shaderRegister: r.baseRegister + (slot - start),
          registerSpace: r.space,
          rangeIndex: i,
        };
      }
      next = unbounded ? Number.MAX_SAFE_INTEGER : start + r.count;
    }
    return null;
  }

  function findRootParameterStartingAt(parameters, userSgpr) {
    for (let i = 0; i < parameters.length; i += 1) {
      if (parameters[i].startSgpr === userSgpr) {
        return parameters[i];
      }
    }
    return null;
  }

  function findRootParameterCovering(parameters, userSgpr) {
    for (let i = 0; i < parameters.length; i += 1) {
      const p = parameters[i];
      if (userSgpr >= p.startSgpr && userSgpr < p.endSgpr) {
        return p;
      }
    }
    return null;
  }

  function isNormalizedCbvSpec(spec) {
    if (!isPlainObject(spec) || !isPlainObject(spec.cbv)) return false;
    const cbv = spec.cbv;
    if (!isNonNegativeInt(cbv.userSgpr)) return false;
    if (cbv.kind === 'table') {
      if (!isNonNegativeInt(cbv.tableByteOffset)) return false;
    } else if (cbv.kind !== 'root_descriptor' && cbv.kind !== 'root_constants') {
      return false;
    }
    return (
      isNonNegativeInt(spec.byteOffset) &&
      spec.byteOffset % 4 === 0 &&
      isNonNegativeInt(spec.bitOffset) &&
      spec.bitOffset <= MAX_HEAP_INDEX_SHIFT &&
      isNonNegativeInt(spec.mask) &&
      spec.mask > 0
    );
  }

  function isNormalizedBindingSource(rawSource) {
    if (!isPlainObject(rawSource) || typeof rawSource.kind !== 'string') {
      return false;
    }
    switch (rawSource.kind) {
      case 'table':
        return isNonNegativeInt(rawSource.userSgpr) && isNonNegativeInt(rawSource.tableByteOffset);
      case 'root_constants':
        return isNonNegativeInt(rawSource.userSgpr);
      case 'root_descriptor':
        if (!isNonNegativeInt(rawSource.userSgpr)) return false;
        if (rawSource.rootTableOffsetGiven === true) {
          return isNonNegativeInt(rawSource.tableByteOffset);
        }
        return true;
      case 'static_sampler':
        return isNonNegativeInt(rawSource.index);
      case 'inline_sampler':
        return (
          Array.isArray(rawSource.sdwords) &&
          rawSource.sdwords.length === INLINE_SAMPLER_SDWORD_COUNT &&
          rawSource.sdwords.every(isNonNegativeInt)
        );
      case 'sampler_heap':
        return (
          isNonNegativeInt(rawSource.baseUserSgpr) &&
          isNonNegativeInt(rawSource.stride) &&
          rawSource.stride > 0 &&
          isNonNegativeInt(rawSource.indexUserSgpr) &&
          isNonNegativeInt(rawSource.indexShift) &&
          rawSource.indexShift <= MAX_HEAP_INDEX_SHIFT
        );
      case 'resource_heap':
        if (rawSource.resourceHeap === true) {
          return rawSource.userSgpr === undefined || isNonNegativeInt(rawSource.userSgpr);
        }
        return (
          isNonNegativeInt(rawSource.baseUserSgpr) &&
          isNonNegativeInt(rawSource.stride) &&
          rawSource.stride > 0 &&
          isNonNegativeInt(rawSource.indexUserSgpr) &&
          isNonNegativeInt(rawSource.indexShift) &&
          rawSource.indexShift <= MAX_HEAP_INDEX_SHIFT
        );
      case 'cbv_indexed_sampler_heap':
      case 'cbv_indexed_resource_heap':
        return (
          isNormalizedCbvSpec(rawSource) &&
          isNonNegativeInt(rawSource.shift) &&
          rawSource.shift <= MAX_HEAP_INDEX_SHIFT &&
          isNonNegativeInt(rawSource.stride)
        );
      case 'lane_indexed_sampler_heap':
      case 'lane_indexed_resource_heap':
        return (
          isNonNegativeInt(rawSource.maxCount) &&
          rawSource.maxCount >= 1 &&
          rawSource.maxCount <= MAX_LANE_INDEXED_WINDOW &&
          isNonNegativeInt(rawSource.shift) &&
          rawSource.shift <= MAX_HEAP_INDEX_SHIFT &&
          isNonNegativeInt(rawSource.stride) &&
          (rawSource.hasWindowBase !== true || isNormalizedCbvSpec(rawSource.windowBase))
        );
      case 'unresolved':
        return rawSource.unresolved === true;
      case 'rt_arena':
        return rawSource.rtArena === true;
      default:
        return false;
    }
  }

  function resolveDiagnosticBindingKind(binding, layoutEntry) {
    if (binding.kind !== undefined && binding.kind !== null) {
      if (typeof binding.kind === 'string') {
        const mapped = STAGE_REFLECTION_KIND_MAP[binding.kind.trim().toLowerCase()];
        if (mapped) {
          return { kind: mapped, rawKind: binding.kind };
        }
      }
      return { kind: null, rawKind: binding.kind };
    }

    const entry = layoutEntry || (isPlainObject(binding.layoutEntry) ? binding.layoutEntry : null);
    if (entry) {
      if (entry.buffer) {
        const mapped = STAGE_REFLECTION_KIND_MAP[String(entry.buffer.type || 'uniform').toLowerCase()];
        return { kind: mapped || 'uniform', rawKind: entry.buffer.type || 'uniform' };
      }
      if (entry.texture) {
        return { kind: 'texture', rawKind: 'texture' };
      }
      if (entry.storageTexture) {
        return { kind: 'storage_texture', rawKind: 'storageTexture' };
      }
      if (entry.sampler) {
        return {
          kind: entry.sampler.type === 'comparison' ? 'comparison_sampler' : 'sampler',
          rawKind: entry.sampler.type || 'sampler',
        };
      }
    }

    if (binding.buffer) {
      const mapped = STAGE_REFLECTION_KIND_MAP[String(binding.buffer.type || 'uniform').toLowerCase()];
      return { kind: mapped || 'uniform', rawKind: binding.buffer.type || 'uniform' };
    }
    if (binding.texture) {
      return { kind: 'texture', rawKind: 'texture' };
    }
    if (binding.storageTexture || binding.storage_texture) {
      return { kind: 'storage_texture', rawKind: 'storageTexture' };
    }
    if (binding.sampler) {
      return {
        kind: binding.sampler.type === 'comparison' ? 'comparison_sampler' : 'sampler',
        rawKind: binding.sampler.type || 'sampler',
      };
    }

    if (typeof binding.resource === 'string') {
      const res = binding.resource.trim();
      if (res === 'texture') return { kind: 'texture', rawKind: res };
      if (res === 'storageTexture') return { kind: 'storage_texture', rawKind: res };
      if (res === 'sampler') return { kind: 'sampler', rawKind: res };
      if (res === 'buffer') {
        if (binding.root_constants === true || binding.rootConstants === true) {
          return { kind: 'uniform', rawKind: res };
        }
        if (binding.register && binding.register.class === 'b') {
          return { kind: 'uniform', rawKind: res };
        }
        if (binding.register && binding.register.class === 't') {
          return { kind: 'read_only_storage', rawKind: res };
        }
        if (binding.register && binding.register.class === 'u') {
          return { kind: 'storage', rawKind: res };
        }
        return { kind: 'uniform', rawKind: res };
      }
    }

    return { kind: null, rawKind: binding.resource !== undefined ? binding.resource : null };
  }

  function formatCbvDesc(cbv) {
    if (cbv.kind === 'root_descriptor') return `root CBV at s${cbv.userSgpr}`;
    if (cbv.kind === 'root_constants') return `root constants at s${cbv.userSgpr}`;
    return `table + ${cbv.tableByteOffset} at s${cbv.userSgpr}`;
  }

  function formatBindingSourceKey(group, binding, source) {
    const base = `@group(${group}) @binding(${binding})`;
    if (!source) return base;
    switch (source.kind) {
      case 'table':
        return `${base} <- table at s${source.userSgpr} + ${source.tableByteOffset}`;
      case 'root_constants':
        return `${base} <- root constants at s${source.userSgpr}`;
      case 'root_descriptor':
        return source.rootTableOffsetGiven === true
          ? `${base} <- root descriptor at s${source.userSgpr} (table + ${source.tableByteOffset})`
          : `${base} <- root descriptor at s${source.userSgpr}`;
      case 'static_sampler':
        return `${base} <- static sampler ${source.index}`;
      case 'inline_sampler': {
        const hexWords = source.sdwords
          .map((w) => (w >>> 0).toString(16).padStart(8, '0'))
          .join(' ');
        return `${base} <- inline sampler ${hexWords}`;
      }
      case 'sampler_heap':
        return `${base} <- sampler heap[s${source.indexUserSgpr} << ${source.indexShift}]`;
      case 'resource_heap':
        if (source.resourceHeap === true) {
          return source.userSgpr !== undefined
            ? `${base} <- resource heap (s${source.userSgpr})`
            : `${base} <- resource heap`;
        }
        return `${base} <- resource heap[s${source.indexUserSgpr} << ${source.indexShift}]`;
      case 'cbv_indexed_sampler_heap':
      case 'cbv_indexed_resource_heap': {
        const heapName = source.kind === 'cbv_indexed_sampler_heap' ? 'sampler' : 'resource';
        return `${base} <- ${heapName} heap[cb(${formatCbvDesc(source.cbv)})[${source.byteOffset}] << ${source.shift}]`;
      }
      case 'lane_indexed_sampler_heap':
      case 'lane_indexed_resource_heap': {
        const heapName = source.kind === 'lane_indexed_sampler_heap' ? 'sampler' : 'resource';
        const baseStr =
          source.hasWindowBase && source.windowBase
            ? `cb(${formatCbvDesc(source.windowBase.cbv)})[${source.windowBase.byteOffset}]`
            : '0';
        const scaleStr =
          source.stride !== 0 && source.stride !== 1 << source.shift
            ? ` * ${source.stride}`
            : ` << ${source.shift}`;
        return `${base} <- ${heapName} heap[(${baseStr} + lane)${scaleStr}] x ${source.maxCount}`;
      }
      case 'unresolved':
        return source.reason ? `${base} <- unresolved (${source.reason})` : `${base} <- unresolved`;
      case 'rt_arena':
        return `${base} <- rt_arena`;
      default:
        return base;
    }
  }

  function extractStaticSamplerComparisonMode(staticSamplerDesc) {
    if (!isPlainObject(staticSamplerDesc)) return null;
    if (typeof staticSamplerDesc.comparison === 'boolean') {
      return staticSamplerDesc.comparison;
    }
    if (typeof staticSamplerDesc.isComparison === 'boolean') {
      return staticSamplerDesc.isComparison;
    }
    if (typeof staticSamplerDesc.type === 'string') {
      const lower = staticSamplerDesc.type.trim().toLowerCase();
      if (lower === 'comparison') return true;
      if (lower === 'filtering' || lower === 'non-filtering') return false;
    }
    if (typeof staticSamplerDesc.compare === 'string' && staticSamplerDesc.compare.length > 0) {
      return true;
    }
    return null;
  }

  function buildStaticSamplerFallbackEntry(bindingInfo, staticSamplerIndex, reason, detailMessage) {
    const isComparison = bindingInfo.kind === 'comparison_sampler';
    const samplerType = isComparison ? 'comparison' : 'filtering';
    const fallbackDescriptor = isComparison
      ? {
          label:
            staticSamplerIndex !== null
              ? `fallback_static_sampler_${staticSamplerIndex}_comparison`
              : `fallback_sampler_g${bindingInfo.group}_b${bindingInfo.binding}_comparison`,
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
          addressModeW: 'clamp-to-edge',
          magFilter: 'linear',
          minFilter: 'linear',
          mipmapFilter: 'nearest',
          compare: 'less-equal',
          maxAnisotropy: 1,
        }
      : {
          label:
            staticSamplerIndex !== null
              ? `fallback_static_sampler_${staticSamplerIndex}_filtering`
              : `fallback_sampler_g${bindingInfo.group}_b${bindingInfo.binding}_filtering`,
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
          addressModeW: 'clamp-to-edge',
          magFilter: 'linear',
          minFilter: 'linear',
          mipmapFilter: 'linear',
          maxAnisotropy: 1,
        };

    return {
      group: bindingInfo.group,
      binding: bindingInfo.binding,
      name: bindingInfo.name,
      kind: bindingInfo.kind,
      samplerType,
      staticSamplerIndex,
      reason,
      message: detailMessage,
      fallbackDescriptor,
    };
  }

  function buildGroupLayoutLookup(stageReflection) {
    const lookup = new Map();
    if (isPlainObject(stageReflection) && Array.isArray(stageReflection.bindGroupLayouts)) {
      for (let i = 0; i < stageReflection.bindGroupLayouts.length; i += 1) {
        const bgl = stageReflection.bindGroupLayouts[i];
        if (!bgl || !Array.isArray(bgl.entries)) continue;
        const group = isNonNegativeInt(bgl.group) ? bgl.group : i;
        for (let j = 0; j < bgl.entries.length; j += 1) {
          const entry = bgl.entries[j];
          if (entry && isNonNegativeInt(entry.binding)) {
            lookup.set(`${group}:${entry.binding}`, entry);
          }
        }
      }
    }
    return lookup;
  }

  function extractMinBindingSizeBytes(rawBinding, layoutEntry) {
    if (rawBinding.min_binding_size_bytes !== undefined && rawBinding.min_binding_size_bytes !== null) {
      return rawBinding.min_binding_size_bytes;
    }
    if (rawBinding.minBindingSizeBytes !== undefined && rawBinding.minBindingSizeBytes !== null) {
      return rawBinding.minBindingSizeBytes;
    }
    if (rawBinding.min_binding_size !== undefined && rawBinding.min_binding_size !== null) {
      return rawBinding.min_binding_size;
    }
    if (rawBinding.minBindingSize !== undefined && rawBinding.minBindingSize !== null) {
      return rawBinding.minBindingSize;
    }
    if (layoutEntry && layoutEntry.buffer && layoutEntry.buffer.minBindingSize !== undefined) {
      return layoutEntry.buffer.minBindingSize;
    }
    if (rawBinding.buffer && rawBinding.buffer.minBindingSize !== undefined) {
      return rawBinding.buffer.minBindingSize;
    }
    return null;
  }

  /**
   * Detect reflected sampler slots whose `static_sampler` index is missing from `rootSignatureSpec`
   * (and optionally missing descriptor-table sampler slots) and synthesize default WebGPU
   * filtering / comparison sampler fallback metadata for callers.
   *
   * @param {object} stageReflection
   * @param {object | null | undefined} [rootSignatureSpec]
   * @param {{ includeMissingTableSamplers?: boolean, compute?: boolean, userDataBase?: number, tableSlotBytes?: number }} [options]
   * @returns {{
   *   needed: boolean,
   *   count: number,
   *   filteringCount: number,
   *   comparisonCount: number,
   *   slots: Array<object>,
   *   bySlot: Map<string, object>,
   *   byStaticSamplerIndex: Map<number, object>
   * }}
   */
  function resolveStageDefaultUserDataBase(stageReflection, compute) {
    if (
      stageReflection &&
      (stageReflection.hasMeshOutput === true ||
        isPlainObject(stageReflection.meshOutput) ||
        isPlainObject(stageReflection.mesh_output))
    ) {
      return MESH_USER_DATA_BASE;
    }
    return compute ? COMPUTE_USER_DATA_BASE : GRAPHICS_USER_DATA_BASE;
  }

  function validateCbvDwordAgainstRootSignature(rs, cbvSpec, key, group, binding, kind, name, source, emit) {
    const cbv = cbvSpec.cbv;
    const param = findRootParameterStartingAt(rs.parameters, cbv.userSgpr);
    if (!param) {
      emit(
        group,
        binding,
        kind,
        DIAGNOSTIC_CODES.ROOT_PARAMETER_NOT_FOUND,
        { userSgpr: cbv.userSgpr, parameterType: cbv.kind },
        { userSgpr: cbv.userSgpr, userDataBase: rs.userDataBase },
        `${key}: no root parameter starts at s${cbv.userSgpr}`,
        name,
        source
      );
      return null;
    }
    const pname = `root parameter ${param.index}`;
    if (cbv.kind === 'root_descriptor') {
      if (param.type !== 'cbv') {
        emit(
          group,
          binding,
          kind,
          DIAGNOSTIC_CODES.ROOT_PARAMETER_TYPE_MISMATCH,
          'cbv',
          param.type,
          `${key}: ${pname} is not a root CBV`,
          name,
          source
        );
        return null;
      }
      return { parameterIndex: param.index, cbvSource: 'root_descriptor', cbTableSlot: 0 };
    }
    if (cbv.kind === 'root_constants') {
      if (param.type !== 'constants' || cbvSpec.byteOffset / 4 >= param.num32BitValues) {
        emit(
          group,
          binding,
          kind,
          DIAGNOSTIC_CODES.ROOT_PARAMETER_TYPE_MISMATCH,
          { type: 'constants', minDwords: cbvSpec.byteOffset / 4 + 1 },
          { type: param.type, num32BitValues: param.num32BitValues },
          `${key}: ${pname} is not root constants covering byte ${cbvSpec.byteOffset}`,
          name,
          source
        );
        return null;
      }
      return { parameterIndex: param.index, cbvSource: 'root_constants', cbTableSlot: 0 };
    }
    // table CBV
    const isSamplerTable = param.ranges.some((r) => r.type === 'sampler');
    if (param.type !== 'table' || isSamplerTable) {
      emit(
        group,
        binding,
        kind,
        DIAGNOSTIC_CODES.ROOT_PARAMETER_TYPE_MISMATCH,
        'resource_table',
        isSamplerTable ? 'sampler_table' : param.type,
        `${key}: ${pname} is not a CBV_SRV_UAV descriptor table`,
        name,
        source
      );
      return null;
    }
    if (cbv.tableByteOffset % rs.tableSlotBytes !== 0) {
      emit(
        group,
        binding,
        kind,
        DIAGNOSTIC_CODES.TABLE_OFFSET_UNALIGNED,
        { tableSlotBytes: rs.tableSlotBytes },
        { tableByteOffset: cbv.tableByteOffset, remainder: cbv.tableByteOffset % rs.tableSlotBytes },
        `${key}: ${pname}: byte offset ${cbv.tableByteOffset} is not a descriptor boundary`,
        name,
        source
      );
      return null;
    }
    return {
      parameterIndex: param.index,
      cbvSource: 'table',
      cbTableSlot: cbv.tableByteOffset / rs.tableSlotBytes,
    };
  }

  function synthesizeMissingStaticSamplers(stageReflection, rootSignatureSpec = null, options = {}) {
    if (!isPlainObject(stageReflection)) {
      throw new TypeError('stageReflection must be an object');
    }
    if (options !== undefined && !isPlainObject(options)) {
      throw new TypeError('options must be an object');
    }

    const stage = normalizeDiagnosticStage(stageReflection.stage) || 'fragment';
    const compute =
      typeof options.compute === 'boolean' ? options.compute : stage === 'compute';
    const defaultBase = resolveStageDefaultUserDataBase(stageReflection, compute);
    const normalizedRs =
      rootSignatureSpec !== null && rootSignatureSpec !== undefined
        ? normalizeRootSignatureSpec(rootSignatureSpec, defaultBase, options)
        : normalizeRootSignatureSpec({ parameters: [], static_samplers: [] }, defaultBase, options);

    const includeMissingTableSamplers = Boolean(options.includeMissingTableSamplers);
    const includeExcludedBindings = Boolean(options.includeExcludedBindings);
    const layoutLookup = buildGroupLayoutLookup(stageReflection);
    const bindings = Array.isArray(stageReflection.bindings) ? stageReflection.bindings : [];

    const slots = [];
    const bySlot = new Map();
    const byStaticSamplerIndex = new Map();
    let filteringCount = 0;
    let comparisonCount = 0;

    for (let i = 0; i < bindings.length; i += 1) {
      const raw = bindings[i];
      if (!isPlainObject(raw)) continue;
      if (!includeExcludedBindings && (raw.in_layout === false || raw.inLayout === false)) {
        continue;
      }

      const group = isNonNegativeInt(raw.group) ? raw.group : 0;
      const binding = isNonNegativeInt(raw.binding) ? raw.binding : i;
      const name = typeof raw.name === 'string' ? raw.name : null;
      const layoutEntry =
        (isPlainObject(raw.layoutEntry) ? raw.layoutEntry : null) ||
        layoutLookup.get(`${group}:${binding}`) ||
        null;
      const { kind } = resolveDiagnosticBindingKind(raw, layoutEntry);
      if (kind !== 'sampler' && kind !== 'comparison_sampler') {
        continue;
      }

      let source = null;
      if (raw.source !== undefined && raw.source !== null) {
        if (isNormalizedBindingSource(raw.source)) {
          source = raw.source;
        } else {
          try {
            source = parseBindingSource(raw.source, `bindings[${i}].source`);
          } catch (_err) {
            continue;
          }
        }
      }
      if (!source) continue;

      let fallbackEntry = null;
      if (source.kind === 'static_sampler') {
        const idx = source.index;
        const missing =
          idx >= normalizedRs.staticSamplerCount ||
          (Array.isArray(normalizedRs.staticSamplers) &&
            (normalizedRs.staticSamplers[idx] === null ||
              normalizedRs.staticSamplers[idx] === undefined));
        if (missing) {
          const key = formatBindingSourceKey(group, binding, source);
          fallbackEntry = buildStaticSamplerFallbackEntry(
            { group, binding, name, kind },
            idx,
            DIAGNOSTIC_CODES.STATIC_SAMPLER_MISSING,
            `${key}: missing static sampler ${idx} (root signature defines ${normalizedRs.staticSamplerCount}); synthesize default ${kind === 'comparison_sampler' ? 'comparison' : 'filtering'} sampler`
          );
          if (!byStaticSamplerIndex.has(idx)) {
            byStaticSamplerIndex.set(idx, fallbackEntry);
          }
        }
      } else if (includeMissingTableSamplers && source.kind === 'table') {
        const param = findRootParameterStartingAt(normalizedRs.parameters, source.userSgpr);
        const misaligned = source.tableByteOffset % normalizedRs.tableSlotBytes !== 0;
        const slotInfo =
          param && param.type === 'table' && !misaligned
            ? resolveTableSlot(param, source.tableByteOffset / normalizedRs.tableSlotBytes)
            : null;
        if (!slotInfo || slotInfo.cls !== 'sampler') {
          const key = formatBindingSourceKey(group, binding, source);
          fallbackEntry = buildStaticSamplerFallbackEntry(
            { group, binding, name, kind },
            null,
            !param
              ? DIAGNOSTIC_CODES.ROOT_PARAMETER_NOT_FOUND
              : param.type !== 'table'
                ? DIAGNOSTIC_CODES.ROOT_PARAMETER_TYPE_MISMATCH
                : misaligned
                  ? DIAGNOSTIC_CODES.TABLE_OFFSET_UNALIGNED
                  : !slotInfo
                    ? DIAGNOSTIC_CODES.TABLE_DESCRIPTOR_MISSING
                    : DIAGNOSTIC_CODES.DESCRIPTOR_TYPE_MISMATCH,
            `${key}: table sampler slot unavailable; synthesize default ${kind === 'comparison_sampler' ? 'comparison' : 'filtering'} sampler`
          );
        }
      }

      if (fallbackEntry) {
        slots.push(fallbackEntry);
        bySlot.set(`${group}:${binding}`, fallbackEntry);
        if (fallbackEntry.samplerType === 'comparison') {
          comparisonCount += 1;
        } else {
          filteringCount += 1;
        }
      }
    }

    slots.sort((a, b) => a.group - b.group || a.binding - b.binding);

    return {
      needed: slots.length > 0,
      count: slots.length,
      filteringCount,
      comparisonCount,
      slots,
      bySlot,
      byStaticSamplerIndex,
    };
  }

  /**
   * Validate each reflected binding's `source` (`table`, `static_sampler`, `root_constants`,
   * `root_descriptor`, `sampler_heap`, `resource_heap`), descriptor class compatibility,
   * storage texture format presence, and `min_binding_size_bytes` alignment against a D3D12
   * root signature layout, returning structured mismatch diagnostics.
   *
   * @param {object} stageReflection
   * @param {object} rootSignatureSpec
   * @param {{
   *   compute?: boolean,
   *   userDataBase?: number,
   *   tableSlotBytes?: number,
   *   uniformMinBindingSizeAlignment?: number,
   *   storageMinBindingSizeAlignment?: number,
   *   synthesizeMissingStaticSamplers?: boolean,
   *   includeExcludedBindings?: boolean,
   * }} [options]
   * @returns {object}
   */
  function diagnoseStageBindings(stageReflection, rootSignatureSpec, options = {}) {
    if (!isPlainObject(stageReflection)) {
      throw new TypeError('stageReflection must be an object');
    }
    if (options !== undefined && !isPlainObject(options)) {
      throw new TypeError('options must be an object');
    }

    const stage = normalizeDiagnosticStage(stageReflection.stage) || 'fragment';
    const compute =
      typeof options.compute === 'boolean' ? options.compute : stage === 'compute';
    const defaultBase = resolveStageDefaultUserDataBase(stageReflection, compute);
    const rs = normalizeRootSignatureSpec(rootSignatureSpec, defaultBase, options);
    const bindingArraysEnabled = Boolean(options.bindingArrays);
    const stageRtArenaMask =
      typeof stageReflection.rtArenaMask === 'number'
        ? stageReflection.rtArenaMask
        : typeof stageReflection.rt_arena_mask === 'number'
          ? stageReflection.rt_arena_mask
          : stageReflection.rt_arena === true
            ? RT_ARENA_FULL_MASK
            : 0;

    const uniformAlign =
      options.uniformMinBindingSizeAlignment !== undefined
        ? options.uniformMinBindingSizeAlignment
        : UNIFORM_MIN_BINDING_SIZE_ALIGNMENT;
    const storageAlign =
      options.storageMinBindingSizeAlignment !== undefined
        ? options.storageMinBindingSizeAlignment
        : STORAGE_MIN_BINDING_SIZE_ALIGNMENT;
    if (!isNonNegativeInt(uniformAlign) || uniformAlign <= 0) {
      throw new RangeError(
        `uniformMinBindingSizeAlignment must be a positive integer, got ${JSON.stringify(uniformAlign)}`
      );
    }
    if (!isNonNegativeInt(storageAlign) || storageAlign <= 0) {
      throw new RangeError(
        `storageMinBindingSizeAlignment must be a positive integer, got ${JSON.stringify(storageAlign)}`
      );
    }

    const allowStaticSamplerFallback = Boolean(options.synthesizeMissingStaticSamplers);
    const includeExcludedBindings = Boolean(options.includeExcludedBindings);
    const layoutLookup = buildGroupLayoutLookup(stageReflection);
    const rawBindings = Array.isArray(stageReflection.bindings) ? stageReflection.bindings : [];

    const diagnostics = [];
    const resolvedBindings = [];
    const staticSamplerFallbacks = [];
    const codeCounts = {};

    const emit = (group, binding, kind, code, expected, actual, message, name, source) => {
      codeCounts[code] = (codeCounts[code] || 0) + 1;
      diagnostics.push({
        group,
        binding,
        kind: kind || 'unknown',
        code,
        expected,
        actual,
        message,
        name: name || null,
        source: source || null,
      });
    };

    let checkedCount = 0;
    let unresolvedCount = 0;
    let laneIndexedBindingsCount = 0;
    let laneIndexedArraysCount = 0;
    let rootDescriptorTablesCount = 0;

    for (let i = 0; i < rawBindings.length; i += 1) {
      const raw = rawBindings[i];
      if (!isPlainObject(raw)) {
        emit(
          0,
          i,
          'unknown',
          DIAGNOSTIC_CODES.INVALID_SOURCE,
          'binding object',
          raw,
          `bindings[${i}]: binding record must be an object`,
          null,
          null
        );
        continue;
      }
      if (!includeExcludedBindings && (raw.in_layout === false || raw.inLayout === false)) {
        continue;
      }
      checkedCount += 1;

      const group = isNonNegativeInt(raw.group) ? raw.group : 0;
      const binding = isNonNegativeInt(raw.binding) ? raw.binding : i;
      const name = typeof raw.name === 'string' ? raw.name : null;
      const layoutEntry =
        (isPlainObject(raw.layoutEntry) ? raw.layoutEntry : null) ||
        layoutLookup.get(`${group}:${binding}`) ||
        null;

      const { kind, rawKind } = resolveDiagnosticBindingKind(raw, layoutEntry);

      // Parse or accept pre-normalized source
      let source = null;
      let sourceOk = true;
      if (raw.source === undefined || raw.source === null) {
        sourceOk = false;
        emit(
          group,
          binding,
          kind || 'unknown',
          DIAGNOSTIC_CODES.MISSING_SOURCE,
          'binding source (table, root_constants, root_descriptor, static_sampler, inline_sampler, sampler_heap, resource_heap, cbv_indexed_*_heap, lane_indexed_*_heap, or unresolved)',
          null,
          `@group(${group}) @binding(${binding}): missing binding source`,
          name,
          null
        );
      } else if (isNormalizedBindingSource(raw.source)) {
        source = raw.source;
      } else {
        try {
          source = parseBindingSource(raw.source, `bindings[${i}].source`);
        } catch (err) {
          sourceOk = false;
          emit(
            group,
            binding,
            kind || 'unknown',
            DIAGNOSTIC_CODES.INVALID_SOURCE,
            'valid WgslBindingSource object',
            raw.source,
            `@group(${group}) @binding(${binding}): invalid binding source (${err.message})`,
            name,
            null
          );
        }
      }

      if (source && source.kind === 'rt_arena') {
        if (
          group !== RT_ARENA_BIND_GROUP ||
          binding < RT_ARENA_FIRST_BINDING ||
          binding > RT_ARENA_LAST_BINDING
        ) {
          emit(
            group,
            binding,
            kind || 'unknown',
            DIAGNOSTIC_CODES.RT_ARENA_GROUP_CONFLICT,
            { group: RT_ARENA_BIND_GROUP, bindingRange: [RT_ARENA_FIRST_BINDING, RT_ARENA_LAST_BINDING] },
            { group, binding },
            `@group(${group}) @binding(${binding}): an rt_arena binding must be group ${RT_ARENA_BIND_GROUP}, binding ${RT_ARENA_FIRST_BINDING}..${RT_ARENA_LAST_BINDING}`,
            name,
            source
          );
        }
        continue;
      }

      if (stageRtArenaMask !== 0 && group >= RT_ARENA_BIND_GROUP) {
        emit(
          group,
          binding,
          kind || 'unknown',
          DIAGNOSTIC_CODES.RT_ARENA_GROUP_CONFLICT,
          { maxGroup: RT_ARENA_BIND_GROUP - 1 },
          { group, rtArenaMask: stageRtArenaMask },
          `@group(${group}) @binding(${binding}): rt_arena stage cannot join bindings in group ${group} (runtime owns group ${RT_ARENA_BIND_GROUP})`,
          name,
          source
        );
      }

      const key = formatBindingSourceKey(group, binding, source);

      let bindingFailed = !sourceOk;
      if (!kind) {
        bindingFailed = true;
        emit(
          group,
          binding,
          'unknown',
          DIAGNOSTIC_CODES.UNKNOWN_BINDING_KIND,
          Object.keys(ACCEPTED_DESCRIPTOR_CLASSES),
          rawKind,
          `${key}: unrecognized binding kind ${JSON.stringify(rawKind)}`,
          name,
          source
        );
      }

      // Check min_binding_size_bytes alignment regardless of source join result
      const minBindingSizeBytes = extractMinBindingSizeBytes(raw, layoutEntry);
      if (minBindingSizeBytes !== null) {
        const isBufferKind =
          kind === 'uniform' || kind === 'storage' || kind === 'read_only_storage';
        if (
          typeof minBindingSizeBytes !== 'number' ||
          !Number.isInteger(minBindingSizeBytes) ||
          minBindingSizeBytes < 0
        ) {
          bindingFailed = true;
          emit(
            group,
            binding,
            kind || 'unknown',
            DIAGNOSTIC_CODES.MIN_BINDING_SIZE_MISALIGNED,
            {
              type: 'non-negative integer',
              alignment: kind === 'uniform' ? uniformAlign : storageAlign,
            },
            { minBindingSizeBytes },
            `${key}: invalid min_binding_size_bytes ${JSON.stringify(minBindingSizeBytes)} (expected non-negative integer)`,
            name,
            source
          );
        } else if (isBufferKind && minBindingSizeBytes > 0) {
          const requiredAlign = kind === 'uniform' ? uniformAlign : storageAlign;
          const remainder = minBindingSizeBytes % requiredAlign;
          if (remainder !== 0) {
            const alignedSizeBytes =
              Math.ceil(minBindingSizeBytes / requiredAlign) * requiredAlign;
            bindingFailed = true;
            emit(
              group,
              binding,
              kind,
              DIAGNOSTIC_CODES.MIN_BINDING_SIZE_MISALIGNED,
              {
                alignment: requiredAlign,
                alignedSizeBytes,
              },
              {
                minBindingSizeBytes,
                remainder,
              },
              `${key}: ${kind} min_binding_size_bytes (${minBindingSizeBytes}) is not aligned to ${requiredAlign} bytes (next aligned size: ${alignedSizeBytes})`,
              name,
              source
            );
          }
        } else if (!isBufferKind && kind && minBindingSizeBytes > 0) {
          bindingFailed = true;
          emit(
            group,
            binding,
            kind,
            DIAGNOSTIC_CODES.MIN_BINDING_SIZE_UNEXPECTED,
            0,
            minBindingSizeBytes,
            `${key}: non-buffer ${kind} binding should not specify non-zero min_binding_size_bytes (${minBindingSizeBytes})`,
            name,
            source
          );
        }
      }

      // Storage texture format check (matching runtime/gpu/stage_reflection.cpp:720)
      if (kind === 'storage_texture') {
        const fmt =
          raw.storage_format !== undefined && raw.storage_format !== null
            ? raw.storage_format
            : raw.storageFormat !== undefined && raw.storageFormat !== null
              ? raw.storageFormat
              : layoutEntry && layoutEntry.storageTexture
                ? layoutEntry.storageTexture.format
                : raw.storageTexture
                  ? raw.storageTexture.format
                  : null;
        if (fmt === null || fmt === 0 || fmt === '') {
          bindingFailed = true;
          emit(
            group,
            binding,
            kind,
            DIAGNOSTIC_CODES.MISSING_STORAGE_FORMAT,
            'non-empty WebGPU storage texture format',
            fmt,
            `${key}: a storage texture needs its format`,
            name,
            source
          );
        }
      }

      if (!sourceOk || !kind) {
        continue;
      }

      // Join source against root signature layout
      let resolvedClass = null;
      let resolvedParamIndex = null;
      let resolvedRegister = 0;
      let resolvedSpace = 0;
      let resolvedRootConstants = false;
      let resolvedHeap = null;
      let resolvedInlineSampler = false;
      let resolvedUnresolved = false;
      let resolvedBindingArraySize = 0;
      let synthesizedFallback = null;
      let joinFailed = false;

      if (source.kind === 'sampler_heap' || source.kind === 'resource_heap') {
        const samplerKind = kind === 'sampler' || kind === 'comparison_sampler';
        if ((source.kind === 'sampler_heap') !== samplerKind) {
          joinFailed = true;
          const heapLabel = source.kind === 'sampler_heap' ? 'sampler' : 'resource';
          emit(
            group,
            binding,
            kind,
            DIAGNOSTIC_CODES.HEAP_KIND_MISMATCH,
            samplerKind ? 'sampler_heap' : 'resource_heap',
            source.kind,
            `${key}: a ${kind} binding cannot come from the ${heapLabel} heap`,
            name,
            source
          );
        }

        if (source.kind === 'resource_heap' && source.resourceHeap === true) {
          const covering =
            source.userSgpr !== undefined
              ? findRootParameterCovering(rs.parameters, source.userSgpr)
              : null;
          if (!joinFailed) {
            resolvedParamIndex = covering ? covering.index : null;
            resolvedClass =
              kind === 'uniform'
                ? 'cbv'
                : kind === 'storage' || kind === 'storage_texture'
                  ? 'uav'
                  : 'srv';
            resolvedHeap = {
              kind: 'resource_heap',
              resourceHeap: true,
              parameterIndex: resolvedParamIndex,
              userSgpr: source.userSgpr !== undefined ? source.userSgpr : null,
            };
          }
        } else {
          const covering = findRootParameterCovering(rs.parameters, source.indexUserSgpr);
          if (!covering || covering.type !== 'constants') {
            joinFailed = true;
            emit(
              group,
              binding,
              kind,
              DIAGNOSTIC_CODES.HEAP_INDEX_NOT_ROOT_CONSTANT,
              { userSgpr: source.indexUserSgpr, parameterType: 'constants' },
              covering
                ? {
                    userSgpr: source.indexUserSgpr,
                    parameterIndex: covering.index,
                    parameterType: covering.type,
                  }
                : { userSgpr: source.indexUserSgpr, parameterIndex: null, parameterType: null },
              `${key}: s${source.indexUserSgpr} is not inside a root-constants parameter`,
              name,
              source
            );
          }
          if (!joinFailed) {
            resolvedParamIndex = covering.index;
            resolvedClass =
              kind === 'uniform'
                ? 'cbv'
                : samplerKind
                  ? 'sampler'
                  : kind === 'storage' || kind === 'storage_texture'
                    ? 'uav'
                    : 'srv';
            resolvedHeap = {
              kind: source.kind,
              parameterIndex: covering.index,
              dwordOffset: source.indexUserSgpr - covering.startSgpr,
              baseUserSgpr: source.baseUserSgpr,
              stride: source.stride,
              shift: source.indexShift,
            };
          }
        }
      } else if (
        source.kind === 'cbv_indexed_sampler_heap' ||
        source.kind === 'cbv_indexed_resource_heap'
      ) {
        const samplerKind = kind === 'sampler' || kind === 'comparison_sampler';
        const isSamplerHeap = source.kind === 'cbv_indexed_sampler_heap';
        if (isSamplerHeap !== samplerKind) {
          joinFailed = true;
          emit(
            group,
            binding,
            kind,
            DIAGNOSTIC_CODES.HEAP_KIND_MISMATCH,
            samplerKind ? 'cbv_indexed_sampler_heap' : 'cbv_indexed_resource_heap',
            source.kind,
            `${key}: a ${kind} binding cannot come from the ${isSamplerHeap ? 'sampler' : 'resource'} heap`,
            name,
            source
          );
        } else {
          const cbvJoin = validateCbvDwordAgainstRootSignature(
            rs,
            source,
            key,
            group,
            binding,
            kind,
            name,
            source,
            emit
          );
          if (!cbvJoin) {
            joinFailed = true;
          } else {
            resolvedParamIndex = cbvJoin.parameterIndex;
            resolvedClass =
              kind === 'uniform'
                ? 'cbv'
                : samplerKind
                  ? 'sampler'
                  : kind === 'storage' || kind === 'storage_texture'
                    ? 'uav'
                    : 'srv';
            resolvedHeap = {
              kind: source.kind,
              parameterIndex: cbvJoin.parameterIndex,
              cbvSource: cbvJoin.cbvSource,
              cbTableSlot: cbvJoin.cbTableSlot,
              byteOffset: source.byteOffset,
              bitOffset: source.bitOffset,
              mask: source.mask,
              shift: source.shift,
              stride: source.stride,
            };
          }
        }
      } else if (
        source.kind === 'lane_indexed_sampler_heap' ||
        source.kind === 'lane_indexed_resource_heap'
      ) {
        const samplerKind = kind === 'sampler' || kind === 'comparison_sampler';
        const isSamplerHeap = source.kind === 'lane_indexed_sampler_heap';
        if (isSamplerHeap !== samplerKind) {
          joinFailed = true;
          emit(
            group,
            binding,
            kind,
            DIAGNOSTIC_CODES.HEAP_KIND_MISMATCH,
            samplerKind ? 'lane_indexed_sampler_heap' : 'lane_indexed_resource_heap',
            source.kind,
            `${key}: a ${kind} binding cannot come from the ${isSamplerHeap ? 'sampler' : 'resource'} heap`,
            name,
            source
          );
        } else {
          let wbJoin = null;
          if (source.hasWindowBase && source.windowBase) {
            wbJoin = validateCbvDwordAgainstRootSignature(
              rs,
              source.windowBase,
              key,
              group,
              binding,
              kind,
              name,
              source,
              emit
            );
            if (!wbJoin) {
              joinFailed = true;
            }
          }
          if (!joinFailed && !bindingArraysEnabled) {
            joinFailed = true;
            laneIndexedBindingsCount += 1;
            emit(
              group,
              binding,
              kind,
              DIAGNOSTIC_CODES.BINDING_ARRAYS_UNSUPPORTED,
              { bindingArrays: true, maxCount: source.maxCount },
              { bindingArrays: false },
              `${key}: lane-indexed heap window of ${source.maxCount} descriptors needs a binding array; consumer does not bind binding arrays`,
              name,
              source
            );
          } else if (!joinFailed) {
            laneIndexedArraysCount += 1;
            resolvedBindingArraySize = source.maxCount;
            resolvedParamIndex = wbJoin ? wbJoin.parameterIndex : null;
            resolvedClass =
              kind === 'uniform'
                ? 'cbv'
                : samplerKind
                  ? 'sampler'
                  : kind === 'storage' || kind === 'storage_texture'
                    ? 'uav'
                    : 'srv';
            resolvedHeap = {
              kind: source.kind,
              maxCount: source.maxCount,
              shift: source.shift,
              stride: source.stride,
              hasWindowBase: source.hasWindowBase,
              parameterIndex: wbJoin ? wbJoin.parameterIndex : null,
              cbvSource: wbJoin ? wbJoin.cbvSource : 'constant',
              cbTableSlot: wbJoin ? wbJoin.cbTableSlot : 0,
            };
          }
        }
      } else if (source.kind === 'unresolved') {
        if (kind === 'storage' || kind === 'storage_texture') {
          joinFailed = true;
          emit(
            group,
            binding,
            kind,
            DIAGNOSTIC_CODES.UNRESOLVED_WRITABLE_BINDING,
            ['uniform', 'read_only_storage', 'texture', 'sampler', 'comparison_sampler'],
            kind,
            `${key}: a ${kind} binding cannot have an unresolved source`,
            name,
            source
          );
        } else {
          const samplerKind = kind === 'sampler' || kind === 'comparison_sampler';
          resolvedClass = kind === 'uniform' ? 'cbv' : samplerKind ? 'sampler' : 'srv';
          resolvedUnresolved = true;
          unresolvedCount += 1;
        }
      } else if (source.kind === 'inline_sampler') {
        const samplerKind = kind === 'sampler' || kind === 'comparison_sampler';
        if (!samplerKind) {
          joinFailed = true;
          emit(
            group,
            binding,
            kind,
            DIAGNOSTIC_CODES.DESCRIPTOR_TYPE_MISMATCH,
            ACCEPTED_DESCRIPTOR_CLASSES[kind].map(formatDescriptorClass),
            formatDescriptorClass('sampler'),
            `${key}: a ${kind} binding cannot take an inline sampler`,
            name,
            source
          );
        } else {
          resolvedClass = 'sampler';
          resolvedInlineSampler = true;
        }
      } else if (source.kind === 'static_sampler') {
        const idx = source.index;
        const samplerKind = kind === 'sampler' || kind === 'comparison_sampler';
        const missing =
          idx >= rs.staticSamplerCount ||
          (Array.isArray(rs.staticSamplers) &&
            (rs.staticSamplers[idx] === null || rs.staticSamplers[idx] === undefined));

        if (!samplerKind) {
          joinFailed = true;
          emit(
            group,
            binding,
            kind,
            DIAGNOSTIC_CODES.DESCRIPTOR_TYPE_MISMATCH,
            ACCEPTED_DESCRIPTOR_CLASSES[kind].map(formatDescriptorClass),
            formatDescriptorClass('sampler'),
            `${key}: a ${kind} binding cannot take a sampler`,
            name,
            source
          );
        } else if (missing) {
          if (allowStaticSamplerFallback) {
            synthesizedFallback = buildStaticSamplerFallbackEntry(
              { group, binding, name, kind },
              idx,
              DIAGNOSTIC_CODES.STATIC_SAMPLER_MISSING,
              `${key}: missing static sampler ${idx} (root signature defines ${rs.staticSamplerCount}); synthesized default ${kind === 'comparison_sampler' ? 'comparison' : 'filtering'} sampler`
            );
            staticSamplerFallbacks.push(synthesizedFallback);
            resolvedClass = 'sampler';
            resolvedRegister = idx;
            resolvedSpace = 0;
          } else {
            joinFailed = true;
            emit(
              group,
              binding,
              kind,
              DIAGNOSTIC_CODES.STATIC_SAMPLER_MISSING,
              {
                staticSamplerCount: rs.staticSamplerCount,
                validRange: rs.staticSamplerCount > 0 ? [0, rs.staticSamplerCount - 1] : null,
              },
              { staticSamplerIndex: idx },
              `${key}: no static sampler ${idx} (root signature defines ${rs.staticSamplerCount})`,
              name,
              source
            );
          }
        } else {
          const desc = Array.isArray(rs.staticSamplers) ? rs.staticSamplers[idx] : null;
          const explicitComparison = extractStaticSamplerComparisonMode(desc);
          if (explicitComparison !== null) {
            const wantsComparison = kind === 'comparison_sampler';
            if (wantsComparison !== explicitComparison) {
              joinFailed = true;
              emit(
                group,
                binding,
                kind,
                DIAGNOSTIC_CODES.DESCRIPTOR_TYPE_MISMATCH,
                wantsComparison ? 'comparison_sampler' : 'filtering_sampler',
                explicitComparison ? 'comparison_sampler' : 'filtering_sampler',
                `${key}: static sampler ${idx} is ${explicitComparison ? 'comparison' : 'filtering'} but binding expects ${kind}`,
                name,
                source
              );
            }
          }
          if (!joinFailed) {
            resolvedClass = 'sampler';
            if (isPlainObject(desc)) {
              const sReg =
                desc.shader_register !== undefined
                  ? desc.shader_register
                  : desc.shaderRegister !== undefined
                    ? desc.shaderRegister
                    : desc.register;
              const sSpace =
                desc.register_space !== undefined
                  ? desc.register_space
                  : desc.registerSpace !== undefined
                    ? desc.registerSpace
                    : desc.space;
              resolvedRegister = isNonNegativeInt(sReg) ? sReg : idx;
              resolvedSpace = isNonNegativeInt(sSpace) ? sSpace : 0;
            } else {
              resolvedRegister = idx;
            }
          }
        }
      } else {
        // table / root_constants / root_descriptor: root parameter must START at user_sgpr
        const param = findRootParameterStartingAt(rs.parameters, source.userSgpr);
        if (!param) {
          joinFailed = true;
          const covering = findRootParameterCovering(rs.parameters, source.userSgpr);
          const insideDetail = covering
            ? ` (inside root parameter ${covering.index} [${covering.type}] spanning s${covering.startSgpr}..s${covering.endSgpr - 1})`
            : '';
          emit(
            group,
            binding,
            kind,
            DIAGNOSTIC_CODES.ROOT_PARAMETER_NOT_FOUND,
            {
              userSgpr: source.userSgpr,
              parameterType:
                source.kind === 'table'
                  ? 'table'
                  : source.kind === 'root_constants'
                    ? 'constants'
                    : ['cbv', 'srv', 'uav'],
            },
            {
              userSgpr: source.userSgpr,
              userDataBase: rs.userDataBase,
              coveringParameterIndex: covering ? covering.index : null,
              coveringParameterType: covering ? covering.type : null,
              availableStartSgprs: rs.parameters.map((p) => p.startSgpr),
            },
            `${key}: no root parameter starts at s${source.userSgpr}${insideDetail}`,
            name,
            source
          );
        } else {
          resolvedParamIndex = param.index;
          const pname = `root parameter ${param.index}`;

          if (source.kind === 'table') {
            if (param.type !== 'table') {
              joinFailed = true;
              emit(
                group,
                binding,
                kind,
                DIAGNOSTIC_CODES.ROOT_PARAMETER_TYPE_MISMATCH,
                'table',
                param.type,
                `${key}: ${pname} is not a descriptor table (got ${param.type})`,
                name,
                source
              );
            } else if (source.tableByteOffset % rs.tableSlotBytes !== 0) {
              joinFailed = true;
              emit(
                group,
                binding,
                kind,
                DIAGNOSTIC_CODES.TABLE_OFFSET_UNALIGNED,
                { tableSlotBytes: rs.tableSlotBytes },
                {
                  tableByteOffset: source.tableByteOffset,
                  remainder: source.tableByteOffset % rs.tableSlotBytes,
                },
                `${key}: ${pname} has no descriptor at byte offset ${source.tableByteOffset} (not a multiple of ${rs.tableSlotBytes} bytes)`,
                name,
                source
              );
            } else {
              const slot = source.tableByteOffset / rs.tableSlotBytes;
              const slotInfo = resolveTableSlot(param, slot);
              if (!slotInfo) {
                joinFailed = true;
                emit(
                  group,
                  binding,
                  kind,
                  DIAGNOSTIC_CODES.TABLE_DESCRIPTOR_MISSING,
                  {
                    parameterIndex: param.index,
                    tableByteOffset: source.tableByteOffset,
                    slot,
                  },
                  {
                    parameterIndex: param.index,
                    rangeCount: param.ranges.length,
                  },
                  `${key}: ${pname} has no descriptor at byte offset ${source.tableByteOffset}`,
                  name,
                  source
                );
              } else {
                resolvedClass = slotInfo.cls;
                resolvedRegister = slotInfo.shaderRegister;
                resolvedSpace = slotInfo.registerSpace;
              }
            }
          } else if (source.kind === 'root_constants') {
            if (param.type !== 'constants') {
              joinFailed = true;
              emit(
                group,
                binding,
                kind,
                DIAGNOSTIC_CODES.ROOT_PARAMETER_TYPE_MISMATCH,
                'constants',
                param.type,
                `${key}: ${pname} is not root constants (got ${param.type})`,
                name,
                source
              );
            } else if (kind !== 'uniform') {
              joinFailed = true;
              emit(
                group,
                binding,
                kind,
                DIAGNOSTIC_CODES.DESCRIPTOR_TYPE_MISMATCH,
                'uniform',
                kind,
                `${key}: root constants bind as a uniform buffer only`,
                name,
                source
              );
            } else {
              resolvedClass = 'cbv';
              resolvedRegister = param.shaderRegister;
              resolvedSpace = param.registerSpace;
              resolvedRootConstants = true;
            }
          } else if (source.kind === 'root_descriptor') {
            if (param.type === 'table' && source.rootTableOffsetGiven === true) {
              const effOffset = source.tableByteOffset;
              if (effOffset % rs.tableSlotBytes !== 0) {
                joinFailed = true;
                emit(
                  group,
                  binding,
                  kind,
                  DIAGNOSTIC_CODES.TABLE_OFFSET_UNALIGNED,
                  { tableSlotBytes: rs.tableSlotBytes },
                  {
                    tableByteOffset: effOffset,
                    remainder: effOffset % rs.tableSlotBytes,
                  },
                  `${key}: ${pname} has no descriptor at byte offset ${effOffset} (not a multiple of ${rs.tableSlotBytes} bytes)`,
                  name,
                  source
                );
              } else {
                const slot = effOffset / rs.tableSlotBytes;
                const slotInfo = resolveTableSlot(param, slot);
                if (!slotInfo) {
                  joinFailed = true;
                  emit(
                    group,
                    binding,
                    kind,
                    DIAGNOSTIC_CODES.TABLE_DESCRIPTOR_MISSING,
                    { parameterIndex: param.index, tableByteOffset: effOffset, slot },
                    { parameterIndex: param.index, rangeCount: param.ranges.length },
                    `${key}: ${pname} has no descriptor at byte offset ${effOffset}`,
                    name,
                    source
                  );
                } else {
                  resolvedClass = slotInfo.cls;
                  resolvedRegister = slotInfo.shaderRegister;
                  resolvedSpace = slotInfo.registerSpace;
                  rootDescriptorTablesCount += 1;
                }
              }
            } else if (param.type !== 'cbv' && param.type !== 'srv' && param.type !== 'uav') {
              joinFailed = true;
              emit(
                group,
                binding,
                kind,
                DIAGNOSTIC_CODES.ROOT_PARAMETER_TYPE_MISMATCH,
                ['cbv', 'srv', 'uav'],
                param.type,
                `${key}: ${pname} is not a root descriptor (got ${param.type})`,
                name,
                source
              );
            } else {
              resolvedClass = param.type;
              resolvedRegister = param.shaderRegister;
              resolvedSpace = param.registerSpace;
            }
          }
        }
      }

      if (!joinFailed && !resolvedRootConstants && resolvedClass !== null) {
        const accepted = ACCEPTED_DESCRIPTOR_CLASSES[kind] || [];
        if (!accepted.includes(resolvedClass)) {
          joinFailed = true;
          emit(
            group,
            binding,
            kind,
            DIAGNOSTIC_CODES.DESCRIPTOR_TYPE_MISMATCH,
            accepted.map(formatDescriptorClass),
            formatDescriptorClass(resolvedClass),
            `${key}: a ${kind} binding cannot take a ${formatDescriptorClass(resolvedClass)}`,
            name,
            source
          );
        }
      }

      if (!joinFailed && !bindingFailed) {
        resolvedBindings.push({
          group,
          binding,
          name,
          kind,
          source,
          parameterIndex: resolvedParamIndex,
          registerClass: resolvedClass,
          shaderRegister: resolvedRegister,
          registerSpace: resolvedSpace,
          rootConstants: resolvedRootConstants,
          heap: resolvedHeap,
          inlineSampler: resolvedInlineSampler,
          inlineSamplerSdwords: resolvedInlineSampler ? source.sdwords.slice() : null,
          unresolved: resolvedUnresolved,
          bindingArraySize: resolvedBindingArraySize,
          minBindingSizeBytes: minBindingSizeBytes !== null ? minBindingSizeBytes : 0,
          synthesizedStaticSampler: Boolean(synthesizedFallback),
          fallbackDescriptor: synthesizedFallback ? synthesizedFallback.fallbackDescriptor : null,
        });
      }
    }

    return {
      ok: diagnostics.length === 0,
      stage,
      compute,
      userDataBase: rs.userDataBase,
      tableSlotBytes: rs.tableSlotBytes,
      rtArenaMask: stageRtArenaMask,
      bindingCount: checkedCount,
      resolvedBindings,
      diagnostics,
      staticSamplerFallbacks,
      summary: {
        totalBindings: checkedCount,
        resolvedCount: resolvedBindings.length,
        mismatchCount: diagnostics.length,
        synthesizedStaticSamplers: staticSamplerFallbacks.length,
        unresolvedBindings: unresolvedCount,
        laneIndexedBindings: laneIndexedBindingsCount,
        laneIndexedArrays: laneIndexedArraysCount,
        rootDescriptorTables: rootDescriptorTablesCount,
        codes: codeCounts,
      },
    };
  }

  // ---------------------------------------------------------------------------
  // ReflectionRegistry
  // ---------------------------------------------------------------------------

  /**
   * Registry for S6 shader reflection manifests with multi-stage bind group layout merging,
   * WebGPU PipelineLayout construction via `WebGpuPipelineCache`, and vertex buffer layout generation.
   */
  class ReflectionRegistry {
    /**
     * @param {{ fetch?: typeof fetch, maxBindGroups?: number, maxNegativeLookupCacheSize?: number }} [options]
     */
    constructor(options = {}) {
      if (options !== undefined && !isPlainObject(options)) {
        throw new TypeError('ReflectionRegistry options must be an object');
      }
      this._fetch = typeof options.fetch === 'function' ? options.fetch : null;
      this._maxBindGroups =
        options.maxBindGroups !== undefined ? options.maxBindGroups : MAX_WEBGPU_BIND_GROUPS;
      if (
        !Number.isInteger(this._maxBindGroups) ||
        this._maxBindGroups < 1 ||
        this._maxBindGroups > MAX_REFLECTED_GROUPS
      ) {
        throw new RangeError(
          `maxBindGroups must be an integer in 1..${MAX_REFLECTED_GROUPS}, got ${this._maxBindGroups}`
        );
      }
      this._maxNegativeLookupCacheSize =
        options.maxNegativeLookupCacheSize !== undefined
          ? options.maxNegativeLookupCacheSize
          : DEFAULT_NEGATIVE_LOOKUP_CACHE_SIZE;
      if (
        !Number.isInteger(this._maxNegativeLookupCacheSize) ||
        this._maxNegativeLookupCacheSize < 0
      ) {
        throw new RangeError(
          `maxNegativeLookupCacheSize must be a non-negative integer, got ${this._maxNegativeLookupCacheSize}`
        );
      }

      this._shadersByHash = new Map();
      this._byHashBigInt = this._shadersByHash;
      this._byHashSafeInt = new Map();
      this._shadersByHex = new Map();
      this._shadersByBareHex = new Map();
      this._shadersByName = new Map();
      this._negativeHashCache = new Set();
      this._negativeQueryCache = new Set();
      this._bindGroupLayouts = new Map();
      this._pipelineLayouts = new Map();
      this._pipelines = new Map();
      this._registerConvention = { source: 'dxc-spirv', shifts: { b: 0, t: 0, u: 0, s: 0 } };
      this._samplerHeapBindings = 0;
      this._resourceHeapStages = 0;
      this._resourceHeapBindings = 0;
      this._inlineSamplerStages = 0;
      this._inlineSamplerBindings = 0;
      this._vertexIndexStages = 0;
      this._unresolvedBindings = 0;
      this._cbvIndexedHeapBindings = 0;
      this._laneIndexedHeapBindings = 0;
      this._meshOutputStages = 0;
      this._rtArenaStages = 0;

      this._stats = {
        manifestsLoaded: 0,
        sidecarsLoaded: 0,
        layoutsBuilt: 0,
        pipelineLayoutsBuilt: 0,
        vertexLayoutsBuilt: 0,
        lookupHits: 0,
        lookupMisses: 0,
        fastHashLookups: 0,
        fastHashHits: 0,
        negativeLookupHits: 0,
        bindingConflicts: 0,
      };
    }

    _registerShaderRecord(shader, indexEntryNames = true) {
      const previous = this._shadersByHash.get(shader.hash);
      if (previous) {
        this._samplerHeapBindings -= previous.samplerHeapBindings || 0;
        this._resourceHeapBindings -= previous.resourceHeapBindings || 0;
        this._inlineSamplerBindings -= previous.inlineSamplerBindings || 0;
        this._unresolvedBindings -= previous.unresolvedBindings || 0;
        this._cbvIndexedHeapBindings -= previous.cbvIndexedHeapBindings || 0;
        this._laneIndexedHeapBindings -= previous.laneIndexedHeapBindings || 0;
        if (previous.hasInlineSampler) {
          this._inlineSamplerStages -= 1;
        }
        if (previous.hasResourceHeap || (previous.resourceHeapBindings || 0) > 0) {
          this._resourceHeapStages -= 1;
        }
        if (previous.isVertexIndexStage) {
          this._vertexIndexStages -= 1;
        }
        if (previous.hasMeshOutput) {
          this._meshOutputStages -= 1;
        }
        if (previous.hasRtArena || (previous.rtArenaMask || 0) !== 0) {
          this._rtArenaStages -= 1;
        }
      }

      this._samplerHeapBindings += shader.samplerHeapBindings || 0;
      this._resourceHeapBindings += shader.resourceHeapBindings || 0;
      this._inlineSamplerBindings += shader.inlineSamplerBindings || 0;
      this._unresolvedBindings += shader.unresolvedBindings || 0;
      this._cbvIndexedHeapBindings += shader.cbvIndexedHeapBindings || 0;
      this._laneIndexedHeapBindings += shader.laneIndexedHeapBindings || 0;
      if (shader.hasInlineSampler) {
        this._inlineSamplerStages += 1;
      }
      if (shader.hasResourceHeap || (shader.resourceHeapBindings || 0) > 0) {
        this._resourceHeapStages += 1;
      }
      if (shader.isVertexIndexStage) {
        this._vertexIndexStages += 1;
      }
      if (shader.hasMeshOutput) {
        this._meshOutputStages += 1;
      }
      if (shader.hasRtArena || (shader.rtArenaMask || 0) !== 0) {
        this._rtArenaStages += 1;
      }

      const bareHex = shader.bareHashHex || shader.hashHex.slice(2);
      this._shadersByHash.set(shader.hash, shader);
      if (shader.hash <= MAX_SAFE_INTEGER_BIGINT) {
        const safeInt = Number(shader.hash);
        this._byHashSafeInt.set(safeInt, shader);
        this._negativeHashCache.delete(safeInt);
      } else {
        this._negativeHashCache.delete(shader.hash);
      }
      this._shadersByHex.set(shader.hashHex, shader);
      this._shadersByBareHex.set(bareHex, shader);

      if (indexEntryNames) {
        this._indexShaderName(shader.entryPoint, shader);
        this._indexShaderName(`${shader.stage}:${shader.entryPoint}`, shader);
        if (shader.name && shader.name !== shader.entryPoint) {
          this._indexShaderName(shader.name, shader);
        }
      } else if (shader.name) {
        this._indexShaderName(shader.name, shader);
      }
    }

    /**
     * Parse and register an S6 reflection manifest (`layout-manifest.json`) or a
     * `playgta6-stage-reflection` v1 sidecar (`*.reflect.json`).
     * Multiple manifests/sidecars may be loaded; later entries overwrite matching hashes.
     *
     * @param {string | object} jsonOrObject
     * @returns {object} Parsed manifest or stage-reflection sidecar
     */
    loadManifest(jsonOrObject) {
      let input = jsonOrObject;
      if (typeof jsonOrObject === 'string') {
        try {
          input = JSON.parse(jsonOrObject);
        } catch (err) {
          throw new ManifestValidationError(
            `malformed JSON: ${err instanceof Error ? err.message : String(err)}`
          );
        }
      }
      if (
        isPlainObject(input) &&
        input.schema === STAGE_REFLECTION_SCHEMA &&
        input.schema_version === undefined
      ) {
        const result = this.loadStageReflection(input);
        input = null;
        return result;
      }

      const parsed = parseReflectionManifest(input);
      input = null;
      this._registerConvention = parsed.registerConvention;

      for (const [idHex, bgl] of parsed.bindGroupLayouts.entries()) {
        this._bindGroupLayouts.set(idHex, bgl);
      }
      for (const [idHex, pl] of parsed.pipelineLayouts.entries()) {
        this._pipelineLayouts.set(idHex, pl);
      }
      for (const shader of parsed.shaders.values()) {
        this._registerShaderRecord(shader, true);
      }
      if (this._negativeQueryCache.size > 0) {
        this._negativeQueryCache.clear();
      }
      for (const [key, pipeline] of parsed.pipelines.entries()) {
        this._pipelines.set(key, pipeline);
      }

      this._stats.manifestsLoaded += 1;
      return parsed;
    }

    /**
     * Parse and register a `playgta6-stage-reflection` v1 sidecar (`*.reflect.json`).
     *
     * @param {string | object} jsonOrObject
     * @returns {object} Parsed stage reflection sidecar
     */
    loadStageReflection(jsonOrObject) {
      const parsed = parseStageReflectionSidecar(jsonOrObject, {
        maxNegativeLookupCacheSize: this._maxNegativeLookupCacheSize,
      });
      for (const stage of parsed.stages.values()) {
        this._registerShaderRecord(stage, false);
      }
      if (this._negativeQueryCache.size > 0) {
        this._negativeQueryCache.clear();
      }
      this._stats.manifestsLoaded += 1;
      this._stats.sidecarsLoaded += 1;
      return parsed;
    }

    _indexShaderName(name, shader) {
      this._shadersByName.set(name, shader);
    }

    /**
     * Fetch an S6 `layout-manifest.json` or `*.reflect.json` sidecar from a URL and register it.
     *
     * @param {string} url
     * @param {typeof fetch} [fetchImpl]
     * @param {RequestInit} [fetchInit]
     * @returns {Promise<object>} Parsed manifest or sidecar
     */
    async loadManifestUrl(url, fetchImpl, fetchInit) {
      if (typeof url !== 'string' || url.length === 0) {
        throw new TypeError('url must be a non-empty string');
      }
      const fetcher = fetchImpl || this._fetch || globalThis.fetch;
      if (typeof fetcher !== 'function') {
        throw new ReflectionError('No fetch implementation available for loadManifestUrl', 'NO_FETCH');
      }

      const response = await fetcher(url, fetchInit);
      if (!response || !response.ok) {
        const status = response ? response.status : 'unknown';
        const statusText = response && response.statusText ? ` ${response.statusText}` : '';
        throw new ReflectionError(
          `Failed to fetch reflection manifest from "${url}": HTTP ${status}${statusText}`,
          'FETCH_FAILED',
          { url, status }
        );
      }

      const payload =
        typeof response.text === 'function' ? await response.text() : await response.json();
      return this.loadManifest(payload);
    }

    /**
     * Fetch a `playgta6-stage-reflection` v1 sidecar (`*.reflect.json`) from a URL and register it.
     *
     * @param {string} url
     * @param {typeof fetch} [fetchImpl]
     * @param {RequestInit} [fetchInit]
     * @returns {Promise<object>} Parsed stage reflection sidecar
     */
    async loadStageReflectionUrl(url, fetchImpl, fetchInit) {
      if (typeof url !== 'string' || url.length === 0) {
        throw new TypeError('url must be a non-empty string');
      }
      const fetcher = fetchImpl || this._fetch || globalThis.fetch;
      if (typeof fetcher !== 'function') {
        throw new ReflectionError(
          'No fetch implementation available for loadStageReflectionUrl',
          'NO_FETCH'
        );
      }

      const response = await fetcher(url, fetchInit);
      if (!response || !response.ok) {
        const status = response ? response.status : 'unknown';
        const statusText = response && response.statusText ? ` ${response.statusText}` : '';
        throw new ReflectionError(
          `Failed to fetch stage reflection sidecar from "${url}": HTTP ${status}${statusText}`,
          'FETCH_FAILED',
          { url, status }
        );
      }

      const payload =
        typeof response.text === 'function' ? await response.text() : await response.json();
      return this.loadStageReflection(payload);
    }

    /**
     * Zero-allocation O(1) lookup of a registered shader/stage reflection by `BigInt` or
     * non-negative safe-integer hash, avoiding `'0x...'` hex string formatting and
     * normalization on hot PSO creation and draw probes (W318).
     *
     * Optionally filters by `stage` (`'vertex'`, `'fragment'`, `'pixel'`, `'compute'`, or
     * `SHADER_STAGE_BITS` bitmask). Unknown hashes are cached in a bounded negative-miss cache.
     *
     * @param {bigint | number} hashBigIntOrNum
     * @param {string | number | null | undefined} [stage]
     * @returns {object | undefined}
     */
    getStageByHashFast(hashBigIntOrNum, stage) {
      return lookupFastHash(
        hashBigIntOrNum,
        stage,
        this._byHashBigInt,
        this._byHashSafeInt,
        this._negativeHashCache,
        this._maxNegativeLookupCacheSize,
        this._stats
      );
    }

    /**
     * Fast O(1) lookup of a registered shader/stage reflection by `BigInt` hash, safe integer,
     * `'0x...'` hex string, bare 16-hex-char string, short hex string, or shader `name` /
     * `entry_point` / `"<stage>:<entry_point>"`.
     *
     * @param {bigint | string | number} hashOrName
     * @returns {object | undefined}
     */
    getShaderReflection(hashOrName) {
      if (typeof hashOrName === 'bigint' || typeof hashOrName === 'number') {
        return this.getStageByHashFast(hashOrName);
      }

      if (
        typeof hashOrName === 'string' &&
        this._maxNegativeLookupCacheSize > 0 &&
        hashOrName.length > 0 &&
        hashOrName.length <= MAX_NEGATIVE_STRING_KEY_LENGTH &&
        this._negativeQueryCache.has(hashOrName)
      ) {
        this._negativeQueryCache.delete(hashOrName);
        this._negativeQueryCache.add(hashOrName);
        this._stats.negativeLookupHits += 1;
        this._stats.lookupMisses += 1;
        return undefined;
      }

      const found = lookupByNormalizedHash(
        hashOrName,
        this._shadersByHex,
        this._shadersByBareHex,
        this._shadersByHash,
        this._shadersByName,
        this._byHashSafeInt
      );

      if (found !== undefined) {
        this._stats.lookupHits += 1;
      } else {
        if (
          typeof hashOrName === 'string' &&
          this._maxNegativeLookupCacheSize > 0 &&
          hashOrName.length > 0 &&
          hashOrName.length <= MAX_NEGATIVE_STRING_KEY_LENGTH
        ) {
          rememberBoundedNegativeKey(
            this._negativeQueryCache,
            hashOrName,
            this._maxNegativeLookupCacheSize
          );
        }
        this._stats.lookupMisses += 1;
      }
      return found;
    }

    /**
     * Fast O(1) lookup of a registered stage reflection by normalized hash or name.
     * Alias for `getShaderReflection(hashOrName)`.
     *
     * @param {bigint | string | number} hashOrName
     * @returns {object | undefined}
     */
    getStageReflection(hashOrName) {
      return this.getShaderReflection(hashOrName);
    }

    /**
     * Look up a pre-merged pipeline reflection from `manifest.pipelines` by `"0x...+0x..."` key
     * or by an array of shader hashes (in any order).
     *
     * @param {string | Array<bigint | string | number>} hashesOrKey
     * @returns {object | undefined}
     */
    getPipelineReflection(hashesOrKey) {
      if (typeof hashesOrKey === 'string') {
        const direct = this._pipelines.get(hashesOrKey.trim());
        if (direct) return direct;
        const parts = hashesOrKey.split('+').map((s) => s.trim()).filter(Boolean);
        if (parts.length > 0) {
          return this.getPipelineReflection(parts);
        }
        return undefined;
      }
      if (!Array.isArray(hashesOrKey) || hashesOrKey.length === 0) {
        return undefined;
      }
      const members = [];
      for (let i = 0; i < hashesOrKey.length; i += 1) {
        const sh = this.getShaderReflection(hashesOrKey[i]);
        if (!sh) return undefined;
        members.push(sh);
      }
      members.sort(
        (a, b) =>
          STAGE_ORDER[a.stage] - STAGE_ORDER[b.stage] ||
          (a.hash < b.hash ? -1 : a.hash > b.hash ? 1 : 0)
      );
      const canonicalKey = members.map((m) => m.hashHex).join('+');
      return this._pipelines.get(canonicalKey);
    }

    /**
     * Resolve a list of shader hashes, names, or reflection objects into normalized
     * `ShaderReflection` objects.
     */
    _resolveReflections(shaderHashesOrReflections) {
      const items = Array.isArray(shaderHashesOrReflections)
        ? shaderHashesOrReflections
        : shaderHashesOrReflections &&
            typeof shaderHashesOrReflections[Symbol.iterator] === 'function' &&
            typeof shaderHashesOrReflections !== 'string'
          ? Array.from(shaderHashesOrReflections)
          : [shaderHashesOrReflections];

      const resolved = new Array(items.length);
      for (let i = 0; i < items.length; i += 1) {
        const item = items[i];
        if (isPlainObject(item)) {
          resolved[i] = item;
        } else if (typeof item === 'bigint' || typeof item === 'string' || typeof item === 'number') {
          const refl = this.getShaderReflection(item);
          if (!refl) {
            throw new ShaderNotFoundError(item);
          }
          resolved[i] = refl;
        } else {
          throw new TypeError(
            `Shader specifier at index ${i} must be a shader hash, name, or reflection object`
          );
        }
      }
      return resolved;
    }

    /**
     * Merge one or more shaders (by hash, name, or reflection object) into WebGPU
     * `GPUBindGroupLayoutDescriptor[]` sorted by group (`0..3`) and ascending `binding`,
     * OR-ing `visibility` bitmasks across stages and rejecting incompatible collisions.
     *
     * @param {Array<bigint | string | number | object> | bigint | string | number | object} shaderHashesOrReflections
     * @param {{ maxBindGroups?: number, requireLayoutComplete?: boolean }} [options]
     * @returns {Array<{ entries: Array<object> }>}
     */
    buildBindGroupLayoutDescriptors(shaderHashesOrReflections, options = {}) {
      const resolved = this._resolveReflections(shaderHashesOrReflections);
      const effectiveOptions = {
        maxBindGroups:
          options.maxBindGroups !== undefined ? options.maxBindGroups : this._maxBindGroups,
        requireLayoutComplete: Boolean(options.requireLayoutComplete),
      };

      try {
        const descriptors = buildBindGroupLayoutDescriptors(resolved, effectiveOptions);
        this._stats.layoutsBuilt += descriptors.length;
        return descriptors;
      } catch (err) {
        if (err instanceof BindingConflictError) {
          this._stats.bindingConflicts += 1;
        }
        throw err;
      }
    }

    /**
     * Build deduplicated `GPUBindGroupLayout`s via `pipelineCache.createBindGroupLayoutFromReflection(...)`
     * (falling back to `device.createBindGroupLayout(...)` when no `pipelineCache` is provided)
     * and construct a WebGPU `GPUPipelineLayout` via `device.createPipelineLayout({ bindGroupLayouts })`.
     *
     * @param {GPUDevice} device
     * @param {object | null | undefined} pipelineCache
     * @param {Array<bigint | string | number | object> | bigint | string | number | object} shaderHashesOrReflections
     * @param {{ label?: string, maxBindGroups?: number, requireLayoutComplete?: boolean }} [options]
     * @returns {GPUPipelineLayout}
     */
    buildPipelineLayout(device, pipelineCache, shaderHashesOrReflections, options = {}) {
      if (!device || typeof device.createPipelineLayout !== 'function') {
        throw new TypeError('device must provide a createPipelineLayout(descriptor) method');
      }

      const descriptors = this.buildBindGroupLayoutDescriptors(shaderHashesOrReflections, options);
      const bindGroupLayouts = new Array(descriptors.length);

      const useReflectionCache =
        Boolean(pipelineCache) && typeof pipelineCache.createBindGroupLayoutFromReflection === 'function';
      const useBasicCache =
        !useReflectionCache &&
        Boolean(pipelineCache) &&
        typeof pipelineCache.getOrCreateBindGroupLayout === 'function';

      if (!useReflectionCache && !useBasicCache && typeof device.createBindGroupLayout !== 'function') {
        throw new TypeError(
          'device must provide createBindGroupLayout(descriptor) when no pipelineCache is supplied'
        );
      }

      for (let g = 0; g < descriptors.length; g += 1) {
        const desc = descriptors[g];
        if (useReflectionCache) {
          bindGroupLayouts[g] = pipelineCache.createBindGroupLayoutFromReflection(desc);
        } else if (useBasicCache) {
          bindGroupLayouts[g] = pipelineCache.getOrCreateBindGroupLayout(desc);
        } else {
          bindGroupLayouts[g] = device.createBindGroupLayout(desc);
        }
      }

      const layoutDesc =
        options && typeof options.label === 'string' && options.label.length > 0
          ? { label: options.label, bindGroupLayouts }
          : { bindGroupLayouts };

      const pipelineLayout = device.createPipelineLayout(layoutDesc);
      this._stats.pipelineLayoutsBuilt += 1;
      return pipelineLayout;
    }

    /**
     * Build WebGPU `GPUVertexBufferLayout[]` from a vertex shader reflection (object, hash,
     * name, or raw `vertex_inputs` array) and optional explicit byte stride.
     *
     * @param {object | Array<object> | bigint | string | number} vertexReflection
     * @param {number | object} [strideOrOptions]
     * @param {object} [extraOptions]
     * @returns {Array<{ arrayStride: number, stepMode: string, attributes: Array<{ shaderLocation: number, offset: number, format: string }> }>}
     */
    buildVertexBufferLayouts(vertexReflection, strideOrOptions, extraOptions) {
      let target = vertexReflection;
      if (
        typeof vertexReflection === 'bigint' ||
        typeof vertexReflection === 'string' ||
        typeof vertexReflection === 'number'
      ) {
        const refl = this.getShaderReflection(vertexReflection);
        if (!refl) {
          throw new ShaderNotFoundError(vertexReflection);
        }
        target = refl;
      }

      const layouts = buildVertexBufferLayouts(target, strideOrOptions, extraOptions);
      this._stats.vertexLayoutsBuilt += 1;
      return layouts;
    }

    /**
     * Validate a registered shader (by hash, name, or reflection object) against a root
     * signature layout using `diagnoseStageBindings`.
     *
     * @param {bigint | string | number | object} shaderHashOrReflection
     * @param {object} rootSignatureSpec
     * @param {object} [options]
     * @returns {object}
     */
    diagnoseStageBindings(shaderHashOrReflection, rootSignatureSpec, options = {}) {
      const [resolved] = this._resolveReflections(shaderHashOrReflection);
      return diagnoseStageBindings(resolved, rootSignatureSpec, options);
    }

    /**
     * Detect missing `static_sampler` slots on a registered shader (by hash, name, or
     * reflection object) and return fallback sampler metadata via `synthesizeMissingStaticSamplers`.
     *
     * @param {bigint | string | number | object} shaderHashOrReflection
     * @param {object | null | undefined} [rootSignatureSpec]
     * @param {object} [options]
     * @returns {object}
     */
    synthesizeMissingStaticSamplers(shaderHashOrReflection, rootSignatureSpec = null, options = {}) {
      const [resolved] = this._resolveReflections(shaderHashOrReflection);
      return synthesizeMissingStaticSamplers(resolved, rootSignatureSpec, options);
    }

    /**
     * Reset cumulative telemetry counters (`lookupHits`, `lookupMisses`, `fastHashLookups`,
     * `fastHashHits`, `negativeLookupHits`, `manifestsLoaded`, `sidecarsLoaded`,
     * `layoutsBuilt`, `pipelineLayoutsBuilt`, `vertexLayoutsBuilt`, `bindingConflicts`)
     * without clearing indexed shaders or stages.
     */
    resetStats() {
      this._stats.manifestsLoaded = 0;
      this._stats.sidecarsLoaded = 0;
      this._stats.layoutsBuilt = 0;
      this._stats.pipelineLayoutsBuilt = 0;
      this._stats.vertexLayoutsBuilt = 0;
      this._stats.lookupHits = 0;
      this._stats.lookupMisses = 0;
      this._stats.fastHashLookups = 0;
      this._stats.fastHashHits = 0;
      this._stats.negativeLookupHits = 0;
      this._stats.bindingConflicts = 0;
    }

    /**
     * Clear all loaded manifests and optionally reset telemetry counters.
     *
     * @param {{ resetStats?: boolean }} [options]
     */
    clear(options = {}) {
      this._shadersByHash.clear();
      this._byHashSafeInt.clear();
      this._shadersByHex.clear();
      this._shadersByBareHex.clear();
      this._shadersByName.clear();
      this._negativeHashCache.clear();
      this._negativeQueryCache.clear();
      this._bindGroupLayouts.clear();
      this._pipelineLayouts.clear();
      this._pipelines.clear();
      this._samplerHeapBindings = 0;
      this._resourceHeapStages = 0;
      this._resourceHeapBindings = 0;
      this._inlineSamplerStages = 0;
      this._inlineSamplerBindings = 0;
      this._vertexIndexStages = 0;
      this._unresolvedBindings = 0;
      this._cbvIndexedHeapBindings = 0;
      this._laneIndexedHeapBindings = 0;
      this._meshOutputStages = 0;
      this._rtArenaStages = 0;

      if (options && options.resetStats) {
        this.resetStats();
      }
    }

    /**
     * Return a telemetry snapshot for the registry.
     */
    getStats() {
      const stagesIndexed = this._shadersByHash.size;
      const lookupHits = sanitizeNonNegativeInt(this._stats.lookupHits);
      const lookupMisses = sanitizeNonNegativeInt(this._stats.lookupMisses);
      const fastHashLookups = sanitizeNonNegativeInt(this._stats.fastHashLookups);
      const fastHashHits = Math.min(
        sanitizeNonNegativeInt(this._stats.fastHashHits),
        fastHashLookups
      );
      const negativeLookupHits = Math.min(
        sanitizeNonNegativeInt(this._stats.negativeLookupHits),
        lookupMisses
      );
      return {
        manifestsLoaded: this._stats.manifestsLoaded,
        sidecarsLoaded: this._stats.sidecarsLoaded,
        shaderCount: stagesIndexed,
        stageCount: stagesIndexed,
        stagesIndexed,
        bindGroupLayoutCount: this._bindGroupLayouts.size,
        pipelineLayoutCount: this._pipelineLayouts.size,
        pipelineCount: this._pipelines.size,
        layoutsBuilt: this._stats.layoutsBuilt,
        pipelineLayoutsBuilt: this._stats.pipelineLayoutsBuilt,
        vertexLayoutsBuilt: this._stats.vertexLayoutsBuilt,
        lookupHits,
        lookupMisses,
        lookupHitRate: computeLookupHitRate(lookupHits, lookupMisses),
        fastHashLookups,
        fastHashHits,
        negativeLookupHits,
        bindingConflicts: this._stats.bindingConflicts,
        samplerHeapBindings: this._samplerHeapBindings,
        resourceHeapStages: this._resourceHeapStages,
        resourceHeapBindings: this._resourceHeapBindings,
        inlineSamplerStages: this._inlineSamplerStages,
        inlineSamplerBindings: this._inlineSamplerBindings,
        vertexIndexStages: this._vertexIndexStages,
        unresolvedBindings: this._unresolvedBindings,
        cbvIndexedHeapBindings: this._cbvIndexedHeapBindings,
        laneIndexedHeapBindings: this._laneIndexedHeapBindings,
        meshOutputStages: this._meshOutputStages,
        rtArenaStages: this._rtArenaStages,
      };
    }
  }

  const exportedApi = {
    LAYOUT_MANIFEST_SCHEMA_VERSION,
    STAGE_REFLECTION_SCHEMA,
    STAGE_REFLECTION_VERSION,
    STAGE_REFLECTION_MIN_VERSION,
    STAGE_REFLECTION_VERSION_ROOT_TABLE_OFFSET,
    STAGE_REFLECTION_VERSION_UNRESOLVED,
    STAGE_REFLECTION_MAX_VERSION,
    MAX_WEBGPU_BIND_GROUPS,
    MAX_HEAP_INDEX_SHIFT,
    MAX_LANE_INDEXED_WINDOW,
    MAX_MESH_VERTICES,
    MAX_MESH_PRIMITIVES,
    MAX_MESH_ATTRIBUTES,
    RT_ARENA_BIND_GROUP,
    RT_ARENA_FIRST_BINDING,
    RT_ARENA_LAST_BINDING,
    RT_ARENA_FULL_MASK,
    INLINE_SAMPLER_SDWORD_COUNT,
    GRAPHICS_USER_DATA_BASE,
    COMPUTE_USER_DATA_BASE,
    MESH_USER_DATA_BASE,
    TABLE_SLOT_BYTES,
    UNIFORM_MIN_BINDING_SIZE_ALIGNMENT,
    STORAGE_MIN_BINDING_SIZE_ALIGNMENT,
    DEFAULT_NEGATIVE_LOOKUP_CACHE_SIZE,
    DIAGNOSTIC_CODES,
    SHADER_STAGE_BITS,
    VERTEX_FORMAT_BYTE_SIZES,
    ReflectionError,
    ManifestValidationError,
    BindingConflictError,
    ShaderNotFoundError,
    VertexLayoutError,
    fnv1a64,
    normalizeShaderHash,
    formatShaderHash,
    computeBindGroupLayoutHash,
    computePipelineLayoutHash,
    parseBindingSource,
    parseMeshOutputLayout,
    parseReflectionManifest,
    parseStageReflectionSidecar,
    buildBindGroupLayoutDescriptors,
    buildVertexBufferLayouts,
    diagnoseStageBindings,
    synthesizeMissingStaticSamplers,
    ReflectionRegistry,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6ReflectionLoader = exportedApi;
  } else if (root) {
    root.PlayGta6ReflectionLoader = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
