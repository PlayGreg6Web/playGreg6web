// Binary WGPK (.wgsl.pack) parser and WebGPU GPUShaderModule cache.
// Pure ES module with zero external dependencies — runs in Web Workers and Node.js.
// Binary layout matches shaders/src/pack.rs and shaders/src/convert.rs byte-for-byte.

import { applyRtBvhPrelude } from './rt_bvh_prelude.js';

/** Binary pack file magic ("WGPK"). */
export const WGPK_MAGIC = 'WGPK';

/** Binary pack magic as raw ASCII bytes. */
export const WGPK_MAGIC_BYTES = Object.freeze([0x57, 0x47, 0x50, 0x4b]);

/** Default WGPK format version (matches PACK_VERSION in shaders/src/pack.rs). */
export const WGPK_VERSION = 1;

/** WGPK v1 format version (20-byte TOC records, 32-bit byte offsets). */
export const WGPK_VERSION_V1 = 1;

/** WGPK v2 format version (24-byte TOC records, 64-bit byte offsets for >4 GiB packs). */
export const WGPK_VERSION_V2 = 2;

/** Fixed header size in bytes: magic(4) + version(4) + entry_count(4) + flags(4). */
export const WGPK_HEADER_BYTES = 16;

/** Fixed WGPK v1 entry table record size in bytes: hash(8) + stage(1) + reserved(3) + offset(4) + length(4). */
export const WGPK_ENTRY_BYTES = 20;

/** Fixed WGPK v1 entry table record size in bytes (explicit alias). */
export const WGPK_ENTRY_BYTES_V1 = 20;

/** Fixed WGPK v2 entry table record size in bytes: hash(8) + stage(1) + reserved(3) + length(4) + offset(8). */
export const WGPK_ENTRY_BYTES_V2 = 24;

/**
 * Return the TOC entry record byte size for a supported WGPK version (`20` for v1, `24` for v2),
 * or `null` for unsupported versions (matches `pack_entry_bytes` in `shaders/src/pack.rs`).
 *
 * @param {number} version
 * @returns {number | null}
 */
export function wgpkEntryBytes(version) {
  if (version === WGPK_VERSION_V1) {
    return WGPK_ENTRY_BYTES_V1;
  }
  if (version === WGPK_VERSION_V2) {
    return WGPK_ENTRY_BYTES_V2;
  }
  return null;
}

/** Default initial byte window for HTTP 206 Range index loading (64 KiB: header + ~3,276 entries in 1 roundtrip). */
export const DEFAULT_INITIAL_RANGE_BYTES = 64 * 1024;

/** Default upper cap for falling back to a full HTTP 200 OK buffer during ranged pack loading (256 MiB). */
export const DEFAULT_MAX_FULL_FALLBACK_BYTES = 256 * 1024 * 1024;

/** Default bounded in-memory LRU cache budget for on-demand WGSL strings in ranged mode (32 MiB). */
export const DEFAULT_MAX_CACHED_WGSL_BYTES = 32 * 1024 * 1024;

/** Default bounded LRU cap for compiled GPUShaderModule entries per GPUDevice (`0` / `null` = unlimited). */
export const DEFAULT_MAX_COMPILED_MODULES = 4096;

/** Default maximum concurrent HTTP 206 Range payload fetches per ranged pack (`0` = unlimited).
 * Kept at 3 so Chrome's 6 HTTP/1.1 connections/origin budget leaves >=3 slots free for io_worker.js .rpf streaming. */
export const DEFAULT_MAX_CONCURRENT_RANGE_FETCHES = 3;

/** Default maximum retries on transient HTTP 206 Range fetch failures (`0` = no retries). */
export const DEFAULT_MAX_RANGE_FETCH_RETRIES = 2;

/** Default base backoff delay in milliseconds before retrying a transient Range fetch failure. */
export const DEFAULT_RANGE_RETRY_DELAY_MS = 25;

/** Default maximum byte gap between nearby shaders to coalesce into a single prefetch HTTP 206 Range request (16 KiB, legacy alias). */
export const DEFAULT_PREFETCH_MAX_GAP_BYTES = 16 * 1024;

/** Default maximum byte span for a single coalesced prefetch HTTP 206 Range request (512 KiB, legacy alias). */
export const DEFAULT_PREFETCH_MAX_SPAN_BYTES = 512 * 1024;

/** Default maximum byte gap between nearby WGPK entries to coalesce into a single HTTP 206 Range fetch (64 KiB). */
export const DEFAULT_MAX_COALESCE_GAP_BYTES = 64 * 1024;

/** Default maximum byte span for a single coalesced WGPK HTTP 206 Range fetch (2 MiB). */
export const DEFAULT_MAX_COALESCE_SPAN_BYTES = 2 * 1024 * 1024;

/** Shader index ABI version matching shaders/src/pack.rs and docs/gpu_commands.md. */
export const SHADER_INDEX_ABI_VERSION = 1;

/**
 * WebGPU shader stage enum matching `ShaderStage` in `shaders/src/convert.rs`.
 * Note: `.wgsl.pack` encodes Vertex=1, Fragment=2, Compute=3.
 */
export const SHADER_STAGE = Object.freeze({
  VERTEX: 1,
  FRAGMENT: 2,
  COMPUTE: 3,
});

/** Reverse lookup from WGPK numeric stage code to canonical stage name. */
export const STAGE_NAMES = Object.freeze({
  [SHADER_STAGE.VERTEX]: 'vertex',
  [SHADER_STAGE.FRAGMENT]: 'fragment',
  [SHADER_STAGE.COMPUTE]: 'compute',
});

const STAGE_NAME_TO_CODE = Object.freeze({
  vertex: SHADER_STAGE.VERTEX,
  fragment: SHADER_STAGE.FRAGMENT,
  compute: SHADER_STAGE.COMPUTE,
});

const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
const FNV1A_64_PRIME = 0x00000100000001b3n;
const U64_MAX = 0xffffffffffffffffn;
const U32_MAX = 0xffffffff;
const U32_RANGE_LIMIT = 0x100000000;
const MAX_SAFE_INTEGER_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);

function normalizeSafeOffset(value, label = 'offset') {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`Invalid WGPK ${label}: ${value}`);
  }
  return value;
}

function normalizeU64Offset(value, label = 'offset') {
  if (typeof value === 'bigint') {
    if (value < 0n || value > U64_MAX) {
      throw new RangeError(`WGPK ${label} out of u64 range: ${value.toString()}`);
    }
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(
        `WGPK ${label} must be a non-negative safe integer or BigInt, got: ${value}`
      );
    }
    return BigInt(value);
  }
  throw new TypeError(`WGPK ${label} must be a number or BigInt`);
}

function resolvePackVersionOption(versionOption, defaultVersion = WGPK_VERSION_V1) {
  if (versionOption === undefined || versionOption === null) {
    return defaultVersion;
  }
  if (versionOption === WGPK_VERSION_V1 || versionOption === '1') {
    return WGPK_VERSION_V1;
  }
  if (versionOption === WGPK_VERSION_V2 || versionOption === '2') {
    return WGPK_VERSION_V2;
  }
  if (typeof versionOption === 'string') {
    const normalized = versionOption.trim().toLowerCase();
    if (normalized === 'v1' || normalized === '1') {
      return WGPK_VERSION_V1;
    }
    if (normalized === 'v2' || normalized === '2') {
      return WGPK_VERSION_V2;
    }
    if (normalized === 'auto') {
      return 'auto';
    }
  }
  throw new RangeError(
    `Unsupported WGPK version option: ${JSON.stringify(versionOption)} (expected 1, 2, 'v1', 'v2', or 'auto')`
  );
}

function sanitizeNonNegativeInt(value, fallback = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return fallback;
  }
  const floored = Math.floor(value);
  return Number.isSafeInteger(floored) ? floored : Number.MAX_SAFE_INTEGER;
}

const textEncoder = new TextEncoder();
const textDecoderStrict = new TextDecoder('utf-8', { fatal: true });

function isSharedArrayBuffer(buf) {
  return typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer;
}

function toUint8Array(input, label = 'buffer') {
  if (typeof input === 'string') {
    return textEncoder.encode(input);
  }
  if (input instanceof Uint8Array) {
    return input;
  }
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  if (input instanceof ArrayBuffer || isSharedArrayBuffer(input)) {
    return new Uint8Array(input);
  }
  throw new TypeError(`Expected ${label} to be an ArrayBuffer, TypedArray, DataView, or string`);
}

function toBinaryView(buffer) {
  if (buffer instanceof Uint8Array) {
    return buffer;
  }
  if (ArrayBuffer.isView(buffer)) {
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  }
  if (buffer instanceof ArrayBuffer || isSharedArrayBuffer(buffer)) {
    return new Uint8Array(buffer);
  }
  throw new TypeError('Expected WGPK buffer to be an ArrayBuffer, SharedArrayBuffer, or ArrayBufferView');
}

function decodeUtf8Strict(slice, entryIndex) {
  const safeSlice = isSharedArrayBuffer(slice.buffer) ? slice.slice() : slice;
  try {
    return textDecoderStrict.decode(safeSlice);
  } catch (err) {
    throw new Error(
      `WGPK entry ${entryIndex} payload is not valid UTF-8: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

function getResponseHeader(response, headerName) {
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

function normalizeHeaderString(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function parseNonNegativeInt(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  const trimmed = String(raw).trim();
  if (!/^\d+$/.test(trimmed)) {
    return null;
  }
  const value = Number(trimmed);
  if (!Number.isSafeInteger(value) || value < 0) {
    return null;
  }
  return value;
}

function parseContentRange(headerValue) {
  if (typeof headerValue !== 'string') {
    return null;
  }
  const match = /^\s*bytes\s+(\d+)-(\d+)\/(\d+|\*)\s*$/i.exec(headerValue);
  if (!match) {
    return null;
  }
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = match[3] === '*' ? null : Number(match[3]);
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    start < 0 ||
    end < start ||
    (total !== null && (!Number.isSafeInteger(total) || total <= end))
  ) {
    return null;
  }
  return { start, end, total };
}

function parseContentRangeTotal(headerValue) {
  if (typeof headerValue !== 'string') {
    return null;
  }
  const satisfied = parseContentRange(headerValue);
  if (satisfied) {
    return satisfied.total;
  }
  const unsatisfied = /^\s*bytes\s+\*\/(\d+)\s*$/i.exec(headerValue);
  if (!unsatisfied) {
    return null;
  }
  const total = Number(unsatisfied[1]);
  return Number.isSafeInteger(total) && total >= 0 ? total : null;
}

function looksLikeMidShaderCorruption(wgsl) {
  if (typeof wgsl !== 'string' || wgsl.length === 0) {
    return false;
  }
  if (wgsl.indexOf('\0') !== -1) {
    return true;
  }
  // Skip leading whitespace and WGSL line/block comments to inspect the first module-scope token.
  let i = 0;
  const len = wgsl.length;
  while (i < len) {
    const ch = wgsl.charCodeAt(i);
    if (ch === 0x20 || ch === 0x09 || ch === 0x0a || ch === 0x0d) {
      i += 1;
      continue;
    }
    if (ch === 0x2f && i + 1 < len) {
      const next = wgsl.charCodeAt(i + 1);
      if (next === 0x2f) {
        i += 2;
        while (i < len && wgsl.charCodeAt(i) !== 0x0a) {
          i += 1;
        }
        continue;
      }
      if (next === 0x2a) {
        i += 2;
        let depth = 1;
        while (i < len && depth > 0) {
          if (wgsl.charCodeAt(i) === 0x2f && i + 1 < len && wgsl.charCodeAt(i + 1) === 0x2a) {
            depth += 1;
            i += 2;
          } else if (wgsl.charCodeAt(i) === 0x2a && i + 1 < len && wgsl.charCodeAt(i + 1) === 0x2f) {
            depth -= 1;
            i += 2;
          } else {
            i += 1;
          }
        }
        continue;
      }
    }
    break;
  }
  if (i >= len) {
    return false;
  }
  const tail = wgsl.slice(i);
  // Closing delimiters or binary operators cannot begin a valid WGSL module.
  if (/^[}\]),.:?+\-*\/%&|^!=<>]/.test(tail)) {
    return true;
  }
  // Function-body-only statements at module scope (e.g. `let` causing WebGPU's
  // `CreateShaderModule: module-scope 'let' is invalid, use 'const'`).
  if (/^(?:let|return|if|else|for|while|loop|switch|case|default|break|continue|discard)\b/.test(tail)) {
    return true;
  }
  return false;
}

function mergeRequestInit(baseInit, overrides = {}) {
  const init = baseInit && typeof baseInit === 'object' ? { ...baseInit } : {};
  const mergedHeaders = {};

  if (init.headers) {
    if (typeof init.headers.forEach === 'function') {
      init.headers.forEach((value, key) => {
        mergedHeaders[key] = value;
      });
    } else if (Array.isArray(init.headers)) {
      for (const pair of init.headers) {
        if (Array.isArray(pair) && pair.length >= 2) {
          mergedHeaders[String(pair[0])] = String(pair[1]);
        }
      }
    } else if (typeof init.headers === 'object') {
      for (const [key, value] of Object.entries(init.headers)) {
        mergedHeaders[key] = value;
      }
    }
  }

  if (overrides.headers) {
    for (const [key, value] of Object.entries(overrides.headers)) {
      // Remove any existing header with the same case-insensitive name before setting.
      const lower = key.toLowerCase();
      for (const existingKey of Object.keys(mergedHeaders)) {
        if (existingKey.toLowerCase() === lower) {
          delete mergedHeaders[existingKey];
        }
      }
      mergedHeaders[key] = value;
    }
  }

  if (overrides.method !== undefined) {
    init.method = overrides.method;
  }
  if (overrides.cache !== undefined && init.cache === undefined) {
    init.cache = overrides.cache;
  }
  if (Object.keys(mergedHeaders).length > 0) {
    init.headers = mergedHeaders;
  }
  return init;
}

function resolveFetchAndOptions(fetchImplOrOptions, maybeOptions, fallbackFetch) {
  let options;
  let fetcher;
  if (typeof fetchImplOrOptions === 'function') {
    fetcher = fetchImplOrOptions;
    options = maybeOptions && typeof maybeOptions === 'object' ? maybeOptions : {};
  } else {
    options = fetchImplOrOptions && typeof fetchImplOrOptions === 'object' ? fetchImplOrOptions : {};
    fetcher = options.fetch ?? fallbackFetch ?? globalThis.fetch;
  }
  return { fetcher, options };
}

function validateMaxCachedWgslBytes(value, defaultValue = DEFAULT_MAX_CACHED_WGSL_BYTES) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`maxCachedWgslBytes must be a non-negative integer, got: ${value}`);
  }
  return value;
}

function resolveMaxCacheBytes(options, defaultValue = DEFAULT_MAX_CACHED_WGSL_BYTES) {
  if (options && options.maxCacheBytes !== undefined) {
    const val = options.maxCacheBytes;
    if (val === null) {
      return defaultValue;
    }
    if (typeof val !== 'number' || !Number.isInteger(val) || val < 0) {
      throw new RangeError(`maxCacheBytes must be a non-negative integer, got: ${val}`);
    }
    return val;
  }
  return validateMaxCachedWgslBytes(
    options ? options.maxCachedWgslBytes : undefined,
    defaultValue
  );
}

function validateMaxCacheEntries(value, defaultValue = null) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`maxCacheEntries must be a non-negative integer, got: ${value}`);
  }
  return value;
}

function validateMaxCompiledModules(value, defaultValue = DEFAULT_MAX_COMPILED_MODULES) {
  if (value === undefined) {
    return defaultValue;
  }
  if (value === null || value === 0) {
    return null;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`maxCompiledModules must be a non-negative integer or null, got: ${value}`);
  }
  return value;
}

function validateReleaseSourceAfterCompile(value, defaultValue = 'auto') {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (value === true || value === false || value === 'auto') {
    return value;
  }
  throw new TypeError(
    `releaseSourceAfterCompile must be true, false, or 'auto', got: ${JSON.stringify(value)}`
  );
}

function validateInitialRangeBytes(value, defaultValue = DEFAULT_INITIAL_RANGE_BYTES) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < WGPK_HEADER_BYTES) {
    throw new RangeError(
      `initialRangeBytes must be an integer >= ${WGPK_HEADER_BYTES}, got: ${value}`
    );
  }
  return value;
}

function validateStreamThresholdBytes(value, defaultValue = 0) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`streamThresholdBytes must be a non-negative integer, got: ${value}`);
  }
  return value;
}

function validateMaxFullFallbackBytes(value, defaultValue = DEFAULT_MAX_FULL_FALLBACK_BYTES) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`maxFullFallbackBytes must be a non-negative integer, got: ${value}`);
  }
  return value;
}

function validateMaxConcurrentRangeFetches(
  value,
  defaultValue = DEFAULT_MAX_CONCURRENT_RANGE_FETCHES
) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(
      `maxConcurrentRangeFetches must be a non-negative integer, got: ${value}`
    );
  }
  return value;
}

function validateMaxRangeFetchRetries(
  value,
  defaultValue = DEFAULT_MAX_RANGE_FETCH_RETRIES
) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`maxRangeFetchRetries must be a non-negative integer, got: ${value}`);
  }
  return value;
}

function validateRangeRetryDelayMs(value, defaultValue = DEFAULT_RANGE_RETRY_DELAY_MS) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new RangeError(
      `rangeRetryDelayMs must be a non-negative finite number, got: ${value}`
    );
  }
  return value;
}

function validateSleepFn(value, defaultValue = null) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'function') {
    throw new TypeError('sleepFn must be a function');
  }
  return value;
}

function validatePrefetchMaxGapBytes(value, defaultValue = DEFAULT_MAX_COALESCE_GAP_BYTES) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new RangeError(`prefetchMaxGapBytes must be a non-negative integer, got: ${value}`);
  }
  return value;
}

function resolvePrefetchMaxGapBytes(options, defaultValue = DEFAULT_MAX_COALESCE_GAP_BYTES) {
  if (!options || typeof options !== 'object') {
    return defaultValue;
  }
  if (options.maxCoalesceGapBytes !== undefined) {
    const val = options.maxCoalesceGapBytes;
    if (val === null) {
      return defaultValue;
    }
    if (typeof val !== 'number' || !Number.isInteger(val) || val < 0) {
      throw new RangeError(`maxCoalesceGapBytes must be a non-negative integer, got: ${val}`);
    }
    return val;
  }
  if (options.maxGapBytes !== undefined) {
    return validatePrefetchMaxGapBytes(options.maxGapBytes, defaultValue);
  }
  return validatePrefetchMaxGapBytes(options.prefetchMaxGapBytes, defaultValue);
}

function validatePrefetchMaxSpanBytes(value, defaultValue = DEFAULT_MAX_COALESCE_SPAN_BYTES) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
    throw new RangeError(`prefetchMaxSpanBytes must be a positive integer, got: ${value}`);
  }
  return value;
}

function resolvePrefetchMaxSpanBytes(options, defaultValue = DEFAULT_MAX_COALESCE_SPAN_BYTES) {
  if (!options || typeof options !== 'object') {
    return defaultValue;
  }
  if (options.maxCoalesceSpanBytes !== undefined) {
    const val = options.maxCoalesceSpanBytes;
    if (val === null) {
      return defaultValue;
    }
    if (typeof val !== 'number' || !Number.isInteger(val) || val < 1) {
      throw new RangeError(`maxCoalesceSpanBytes must be a positive integer, got: ${val}`);
    }
    return val;
  }
  if (options.maxSpanBytes !== undefined) {
    return validatePrefetchMaxSpanBytes(options.maxSpanBytes, defaultValue);
  }
  return validatePrefetchMaxSpanBytes(options.prefetchMaxSpanBytes, defaultValue);
}

/**
 * Pure helper: sort WGPK entry descriptors by ascending byte offset and coalesce
 * adjacent/nearby ranges within `maxCoalesceGapBytes` (default 64 KiB) and
 * `maxCoalesceSpanBytes` (default 2 MiB) into merged HTTP `206 Range` spans.
 *
 * Zero-length entries are skipped (no HTTP Range fetch needed). A single entry whose
 * length exceeds `maxCoalesceSpanBytes` is emitted in its own exact single-entry span
 * so it can still be fetched intact. Does not mutate the caller's `entries` collection.
 *
 * @param {Iterable<{ offset?: number, start?: number, length?: number, end?: number }>} entries
 * @param {{
 *   maxCoalesceGapBytes?: number,
 *   maxGapBytes?: number,
 *   prefetchMaxGapBytes?: number,
 *   maxCoalesceSpanBytes?: number,
 *   maxSpanBytes?: number,
 *   prefetchMaxSpanBytes?: number,
 * }} [options]
 * @returns {Array<{
 *   start: number,
 *   end: number,
 *   rangeEnd: number,
 *   spanBytes: number,
 *   payloadBytes: number,
 *   gapBytes: number,
 *   entries: Array<object>,
 * }>}
 */
export function coalesceEntryRanges(entries, options = {}) {
  const opts = options && typeof options === 'object' ? options : {};
  const maxGapBytes = resolvePrefetchMaxGapBytes(opts, DEFAULT_MAX_COALESCE_GAP_BYTES);
  const maxSpanBytes = resolvePrefetchMaxSpanBytes(opts, DEFAULT_MAX_COALESCE_SPAN_BYTES);

  if (entries === undefined || entries === null) {
    return [];
  }
  if (typeof entries === 'string' || typeof entries[Symbol.iterator] !== 'function') {
    throw new TypeError('entries must be an iterable of WGPK entry ranges');
  }

  const normalized = [];
  let idx = 0;
  for (const entry of entries) {
    if (!entry || typeof entry !== 'object') {
      throw new TypeError(`Entry at index ${idx} must be an object with offset/start and length/end`);
    }
    const rawStart = entry.offset !== undefined ? entry.offset : entry.start;
    if (typeof rawStart !== 'number' || !Number.isSafeInteger(rawStart) || rawStart < 0) {
      throw new RangeError(`Entry at index ${idx} has invalid byte offset: ${rawStart}`);
    }

    let length;
    let end;
    if (entry.length !== undefined) {
      if (
        typeof entry.length !== 'number' ||
        !Number.isSafeInteger(entry.length) ||
        entry.length < 0
      ) {
        throw new RangeError(`Entry at index ${idx} has invalid byte length: ${entry.length}`);
      }
      length = entry.length;
      end = rawStart + length;
    } else if (entry.end !== undefined) {
      if (
        typeof entry.end !== 'number' ||
        !Number.isSafeInteger(entry.end) ||
        entry.end < rawStart
      ) {
        throw new RangeError(`Entry at index ${idx} has invalid byte end: ${entry.end}`);
      }
      end = entry.end;
      length = end - rawStart;
    } else {
      throw new RangeError(`Entry at index ${idx} must specify length or end`);
    }

    if (!Number.isSafeInteger(end)) {
      throw new RangeError(
        `Entry at index ${idx} byte range [${rawStart}..+${length}] exceeds safe integer bounds`
      );
    }

    if (length > 0) {
      normalized.push({ start: rawStart, end, length, entry });
    }
    idx += 1;
  }

  if (normalized.length === 0) {
    return [];
  }

  normalized.sort((a, b) =>
    a.start < b.start ? -1 : a.start > b.start ? 1 : a.end < b.end ? -1 : a.end > b.end ? 1 : 0
  );

  const spans = [];
  let current = {
    start: normalized[0].start,
    end: normalized[0].end,
    rangeEnd: normalized[0].end - 1,
    spanBytes: normalized[0].length,
    payloadBytes: normalized[0].length,
    gapBytes: 0,
    entries: [normalized[0].entry],
  };

  for (let i = 1; i < normalized.length; i += 1) {
    const item = normalized[i];
    const gap = item.start > current.end ? item.start - current.end : 0;
    const proposedEnd = item.end > current.end ? item.end : current.end;
    const proposedSpan = proposedEnd - current.start;
    const isContained = item.start >= current.start && item.end <= current.end;

    if (gap <= maxGapBytes && (proposedSpan <= maxSpanBytes || isContained)) {
      current.end = proposedEnd;
      current.rangeEnd = proposedEnd - 1;
      current.spanBytes = proposedSpan;
      current.payloadBytes += item.length;
      current.gapBytes += gap;
      current.entries.push(item.entry);
    } else {
      spans.push(current);
      current = {
        start: item.start,
        end: item.end,
        rangeEnd: item.end - 1,
        spanBytes: item.length,
        payloadBytes: item.length,
        gapBytes: 0,
        entries: [item.entry],
      };
    }
  }
  spans.push(current);

  return spans;
}

function normalizeHashIterable(hashes) {
  if (hashes === undefined || hashes === null) {
    return [];
  }
  if (typeof hashes === 'string' || typeof hashes[Symbol.iterator] !== 'function') {
    throw new TypeError('hashes must be an iterable of shader hashes');
  }
  const seen = new Set();
  const keys = [];
  for (const rawHash of hashes) {
    const key = normalizeHash(rawHash);
    if (!seen.has(key)) {
      seen.add(key);
      keys.push(key);
    }
  }
  return keys;
}

function createAbortError(signal) {
  if (signal && signal.reason instanceof Error) {
    return signal.reason;
  }
  const err = new Error('The operation was aborted.');
  err.name = 'AbortError';
  err.code = 'ABORT_ERR';
  return err;
}

function defaultSleep(ms, signal = null) {
  if (signal && signal.aborted) {
    return Promise.reject(createAbortError(signal));
  }
  if (ms <= 0) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    let onAbort = null;
    const timer = setTimeout(() => {
      if (signal && onAbort && typeof signal.removeEventListener === 'function') {
        signal.removeEventListener('abort', onAbort);
      }
      resolve();
    }, ms);
    if (signal && typeof signal.addEventListener === 'function') {
      onAbort = () => {
        clearTimeout(timer);
        reject(createAbortError(signal));
      };
      signal.addEventListener('abort', onAbort, { once: true });
    }
  });
}

function computeRangeRetryDelayMs(baseDelayMs, attemptIndex) {
  if (baseDelayMs <= 0) {
    return 0;
  }
  const factor = 1 << Math.min(attemptIndex, 4);
  return baseDelayMs * factor;
}

function isRetryableRangeFetchError(err, fetchInit) {
  if (fetchInit && fetchInit.signal && fetchInit.signal.aborted) {
    return false;
  }
  if (!err) {
    return false;
  }
  if (err.name === 'AbortError' || err.code === 'ABORT_ERR') {
    return false;
  }
  if (err.code === 'WGPK_FULL_FALLBACK_REJECTED') {
    return false;
  }
  if (err.name === 'AssertionError' || err.code === 'ERR_ASSERTION') {
    return false;
  }
  if (err instanceof RangeError) {
    return false;
  }
  if (typeof err.httpStatus === 'number') {
    const status = err.httpStatus;
    return status === 429 || status === 408 || (status >= 500 && status <= 599);
  }
  // Network / transport errors thrown by fetch() or response.arrayBuffer()
  // (e.g. TypeError: Failed to fetch, ECONNRESET, premature connection close).
  return true;
}

function parseWgpkHeader(bytes) {
  if (bytes.byteLength < WGPK_HEADER_BYTES) {
    throw new Error(
      `WGPK pack truncated: ${bytes.byteLength} bytes (minimum required for header: ${WGPK_HEADER_BYTES})`
    );
  }

  if (
    bytes[0] !== WGPK_MAGIC_BYTES[0] ||
    bytes[1] !== WGPK_MAGIC_BYTES[1] ||
    bytes[2] !== WGPK_MAGIC_BYTES[2] ||
    bytes[3] !== WGPK_MAGIC_BYTES[3]
  ) {
    const actualHex = Array.from(bytes.subarray(0, 4))
      .map((b) => `0x${b.toString(16).padStart(2, '0')}`)
      .join(' ');
    throw new Error(`Invalid WGPK pack magic [${actualHex}] (expected "${WGPK_MAGIC}")`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true) >>> 0;
  const entryBytes = wgpkEntryBytes(version);
  if (entryBytes === null) {
    throw new Error(
      `Unsupported WGPK pack version ${version} (expected ${WGPK_VERSION_V1} or ${WGPK_VERSION_V2})`
    );
  }

  const entryCount = view.getUint32(8, true) >>> 0;
  const flags = view.getUint32(12, true) >>> 0;
  const tableBytes = entryCount * entryBytes;
  const dataStart = WGPK_HEADER_BYTES + tableBytes;
  if (!Number.isSafeInteger(tableBytes) || !Number.isSafeInteger(dataStart)) {
    throw new Error(
      `WGPK entry table size overflows safe integer bounds for ${entryCount} entries`
    );
  }

  return { version, entryBytes, entryCount, flags, tableBytes, dataStart };
}

function parseWgpkEntryTable(bytes, header, totalFileBytes) {
  const { version, entryCount, dataStart } = header;
  const entryBytes = header.entryBytes ?? wgpkEntryBytes(version);
  if (entryBytes === null) {
    throw new Error(
      `Unsupported WGPK pack version ${version} (expected ${WGPK_VERSION_V1} or ${WGPK_VERSION_V2})`
    );
  }
  if (bytes.byteLength < dataStart) {
    throw new Error(
      `WGPK pack truncated: ${bytes.byteLength} bytes (header + ${entryCount} entry table records require at least ${dataStart} bytes)`
    );
  }
  if (totalFileBytes !== null && totalFileBytes !== undefined && totalFileBytes < dataStart) {
    throw new Error(
      `WGPK pack truncated: ${totalFileBytes} total bytes (header + ${entryCount} entry table records require at least ${dataStart} bytes)`
    );
  }

  const isV2 = version === WGPK_VERSION_V2;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const rawEntries = new Array(entryCount);
  let ranges = null;
  let prevNonEmptyEnd = dataStart;
  let prevNonEmptyStart = -1;
  let prevNonEmptyIndex = -1;
  const stageCounts = { vertex: 0, fragment: 0, compute: 0 };
  let totalPayloadBytes = 0;
  let maxEnd = dataStart;

  // When totalFileBytes is unknown (e.g. Content-Range: bytes 0-N/*):
  // - WGPK v1 allows unsigned 32-bit byte indices [0..0xFFFFFFFF] up to exact 4 GiB (0x100000000).
  // - WGPK v2 allows 64-bit byte offsets up to Number.MAX_SAFE_INTEGER (~8 PiB).
  const upperBound =
    totalFileBytes !== null && totalFileBytes !== undefined
      ? isV2
        ? totalFileBytes
        : Math.min(totalFileBytes, U32_RANGE_LIMIT)
      : isV2
        ? Number.MAX_SAFE_INTEGER
        : U32_RANGE_LIMIT;

  for (let idx = 0; idx < entryCount; idx += 1) {
    const base = WGPK_HEADER_BYTES + idx * entryBytes;
    const hash = view.getBigUint64(base, true);
    const stageCode = bytes[base + 8];
    const stage = STAGE_NAMES[stageCode];
    if (stage === undefined) {
      throw new Error(`Invalid WGPK shader stage value ${stageCode} at entry ${idx}`);
    }
    // bytes[base + 9 .. base + 12] are reserved for forward compatibility.
    let offset;
    let length;
    let end;
    if (isV2) {
      length = view.getUint32(base + 12, true) >>> 0;
      const offsetBig = view.getBigUint64(base + 16, true);
      const endBig = offsetBig + BigInt(length);
      if (endBig > U64_MAX) {
        throw new Error(
          `WGPK v2 entry ${idx} payload [${offsetBig}..+${length}] overflows u64 bounds`
        );
      }
      if (endBig > MAX_SAFE_INTEGER_BIGINT) {
        throw new Error(
          `WGPK v2 entry ${idx} payload [${offsetBig}..+${length}] exceeds safe integer offset limit (${Number.MAX_SAFE_INTEGER})`
        );
      }
      offset = Number(offsetBig);
      end = Number(endBig);
    } else {
      offset = view.getUint32(base + 12, true) >>> 0;
      length = view.getUint32(base + 16, true) >>> 0;
      end = offset + length;
    }

    if (!Number.isSafeInteger(end) || offset < dataStart || end > upperBound) {
      throw new Error(
        `WGPK entry ${idx} payload [${offset}..+${length}] is out of bounds (valid payload range: [${dataStart}..${upperBound}])`
      );
    }

    if (length > 0) {
      if (ranges !== null) {
        ranges.push({ start: offset, end, index: idx });
      } else if (offset < prevNonEmptyEnd) {
        if (offset >= prevNonEmptyStart) {
          throw new Error(
            `WGPK overlapping payload ranges detected between entry ${prevNonEmptyIndex} [${prevNonEmptyStart}..${prevNonEmptyEnd}) and entry ${idx} [${offset}..${end})`
          );
        }
        // Out-of-order TOC entries: materialize ranges array and fall back to O(N log N) sort check.
        ranges = [];
        for (let k = 0; k < idx; k += 1) {
          const e = rawEntries[k];
          if (e.length > 0) {
            ranges.push({ start: e.offset, end: e.end, index: e.index });
          }
        }
        ranges.push({ start: offset, end, index: idx });
      } else {
        prevNonEmptyStart = offset;
        prevNonEmptyEnd = end;
        prevNonEmptyIndex = idx;
      }
    }

    if (end > maxEnd) {
      maxEnd = end;
    }

    totalPayloadBytes += length;
    stageCounts[stage] += 1;

    rawEntries[idx] = {
      index: idx,
      hash,
      hashHex: formatHashHex(hash),
      stage,
      stageCode,
      offset,
      length,
      end,
    };
  }

  // Reject overlapping non-empty payload ranges across out-of-order entries.
  if (ranges !== null && ranges.length > 1) {
    ranges.sort((a, b) =>
      a.start < b.start ? -1 : a.start > b.start ? 1 : a.end < b.end ? -1 : a.end > b.end ? 1 : 0
    );
    for (let i = 1; i < ranges.length; i += 1) {
      const prev = ranges[i - 1];
      const curr = ranges[i];
      if (curr.start < prev.end) {
        throw new Error(
          `WGPK overlapping payload ranges detected between entry ${prev.index} [${prev.start}..${prev.end}) and entry ${curr.index} [${curr.start}..${curr.end})`
        );
      }
    }
  }

  return { rawEntries, stageCounts, totalPayloadBytes, maxEnd };
}

async function fetchRangeChunk(fetcher, url, start, end, fetchInit, rangeOptions = {}) {
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start) {
    throw new RangeError(`Invalid HTTP byte range: bytes=${start}-${end}`);
  }
  if (fetchInit && fetchInit.signal && fetchInit.signal.aborted) {
    throw createAbortError(fetchInit.signal);
  }
  const rangeHeader = `bytes=${start}-${end}`;
  const response = await fetcher(
    url,
    mergeRequestInit(fetchInit, { cache: 'no-store', headers: { Range: rangeHeader } })
  );
  if (!response || (!response.ok && response.status !== 206)) {
    const status = response ? response.status : 'unknown';
    const statusText = response && response.statusText ? ` ${response.statusText}` : '';
    const err = new Error(
      `Failed to fetch WGSL pack range (${rangeHeader}) from "${url}": HTTP ${status}${statusText}`
    );
    err.httpStatus = response && typeof response.status === 'number' ? response.status : 0;
    err.etag = normalizeHeaderString(getResponseHeader(response, 'etag'));
    err.lastModified = normalizeHeaderString(getResponseHeader(response, 'last-modified'));
    err.contentRangeTotal = parseContentRangeTotal(getResponseHeader(response, 'content-range'));
    throw err;
  }

  const contentRange = parseContentRange(getResponseHeader(response, 'content-range'));
  const contentLength = parseNonNegativeInt(getResponseHeader(response, 'content-length'));
  const etag = normalizeHeaderString(getResponseHeader(response, 'etag'));
  const lastModified = normalizeHeaderString(getResponseHeader(response, 'last-modified'));

  if (response.status !== 206) {
    const requirePartial = Boolean(rangeOptions && rangeOptions.requirePartialContent === true);
    const maxFullFallbackBytes =
      rangeOptions && typeof rangeOptions.maxFullFallbackBytes === 'number'
        ? rangeOptions.maxFullFallbackBytes
        : DEFAULT_MAX_FULL_FALLBACK_BYTES;
    const exceedsCap =
      maxFullFallbackBytes > 0 && contentLength !== null && contentLength > maxFullFallbackBytes;
    if (requirePartial || exceedsCap) {
      if (response.body && typeof response.body.cancel === 'function') {
        try {
          const cancelResult = response.body.cancel();
          if (cancelResult && typeof cancelResult.catch === 'function') {
            cancelResult.catch(() => {});
          }
        } catch (_) {
          // Best-effort cancellation of oversized 200 OK body.
        }
      }
      const sizeDetail =
        contentLength !== null ? `${contentLength} bytes` : 'unknown size';
      const err = new Error(
        `WGSL pack range request (${rangeHeader}) from "${url}" received HTTP ${response.status} (${sizeDetail}) instead of 206 Partial Content; refusing full-pack buffer fallback`
      );
      err.httpStatus = response.status;
      err.code = 'WGPK_FULL_FALLBACK_REJECTED';
      err.contentLength = contentLength;
      err.etag = etag;
      err.lastModified = lastModified;
      throw err;
    }
  }

  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const totalSize =
    contentRange && contentRange.total !== null
      ? contentRange.total
      : response.status === 200
        ? contentLength !== null
          ? contentLength
          : bytes.byteLength
        : null;

  return {
    status: response.status,
    bytes,
    contentRange,
    contentLength,
    etag,
    lastModified,
    totalSize,
  };
}

async function fetchRangeChunkWithRetry(fetcher, url, start, end, fetchInit, retryOptions = {}) {
  const maxRetries = validateMaxRangeFetchRetries(retryOptions.maxRangeFetchRetries);
  const baseDelayMs = validateRangeRetryDelayMs(retryOptions.rangeRetryDelayMs);
  const customSleep = validateSleepFn(retryOptions.sleepFn);
  const onAttempt = typeof retryOptions.onAttempt === 'function' ? retryOptions.onAttempt : null;
  const onRetry = typeof retryOptions.onRetry === 'function' ? retryOptions.onRetry : null;
  const rangeOptions = {
    maxFullFallbackBytes: validateMaxFullFallbackBytes(retryOptions.maxFullFallbackBytes),
    requirePartialContent: Boolean(retryOptions.requirePartialContent === true),
  };
  const signal = fetchInit && fetchInit.signal ? fetchInit.signal : null;

  let attempt = 0;
  for (;;) {
    if (signal && signal.aborted) {
      throw createAbortError(signal);
    }
    if (onAttempt) {
      onAttempt();
    }
    try {
      return await fetchRangeChunk(fetcher, url, start, end, fetchInit, rangeOptions);
    } catch (err) {
      if (attempt >= maxRetries || !isRetryableRangeFetchError(err, fetchInit)) {
        throw err;
      }
      if (onRetry) {
        onRetry(err, attempt);
      }
      const delayMs = computeRangeRetryDelayMs(baseDelayMs, attempt);
      attempt += 1;
      if (customSleep) {
        await customSleep(delayMs);
      } else {
        await defaultSleep(delayMs, signal);
      }
      if (signal && signal.aborted) {
        throw createAbortError(signal);
      }
    }
  }
}

/**
 * Compute deterministic 64-bit FNV-1a hash matching `fnv1a_64` in `shaders/src/convert.rs`.
 *
 * @param {Uint8Array | ArrayBuffer | ArrayBufferView | string} input
 * @returns {bigint} Unsigned 64-bit hash as BigInt
 */
export function fnv1a64(input) {
  const bytes = toUint8Array(input, 'input');
  let hash = FNV1A_64_OFFSET_BASIS;
  for (let i = 0; i < bytes.length; i += 1) {
    hash ^= BigInt(bytes[i]);
    hash = (hash * FNV1A_64_PRIME) & U64_MAX;
  }
  return hash;
}

/**
 * Normalize a shader hash (`BigInt`, hex `string`, or safe non-negative integer `number`)
 * into an unsigned 64-bit `BigInt`.
 *
 * @param {bigint | string | number} hash
 * @returns {bigint}
 */
export function normalizeHash(hash) {
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
    const rawHex = trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
    if (!/^[0-9a-fA-F]{1,16}$/.test(rawHex)) {
      throw new RangeError(`Invalid hex shader hash: ${JSON.stringify(hash)}`);
    }
    return BigInt(`0x${rawHex}`);
  }

  throw new TypeError(`Shader hash must be a BigInt, hex string, or non-negative safe integer`);
}

/**
 * Format a 64-bit shader hash as canonical `0x` + 16 lowercase hex digits,
 * matching `format_shader_hash` in `shaders/src/convert.rs`.
 *
 * @param {bigint | string | number} hash
 * @returns {string}
 */
export function formatHashHex(hash) {
  const value = normalizeHash(hash);
  return `0x${value.toString(16).padStart(16, '0')}`;
}

function resolveStageCode(stage) {
  if (typeof stage === 'number') {
    if (Object.prototype.hasOwnProperty.call(STAGE_NAMES, stage)) {
      return stage;
    }
    throw new RangeError(`Invalid shader stage code: ${stage}`);
  }
  if (typeof stage === 'string') {
    const code = STAGE_NAME_TO_CODE[stage.toLowerCase()];
    if (code !== undefined) {
      return code;
    }
    throw new RangeError(`Invalid shader stage name: ${JSON.stringify(stage)}`);
  }
  throw new TypeError(`Shader stage must be a stage code (1..3) or stage name`);
}

function createDynamicShaderEntry(hash, stage, wgslSource, meta, prevEntry) {
  const key = normalizeHash(hash);
  const hashHex = formatHashHex(key);
  const stageCode = resolveStageCode(stage);
  const stageName = STAGE_NAMES[stageCode];

  if (typeof wgslSource !== 'string') {
    throw new TypeError('wgslSource must be a non-empty string');
  }
  if (wgslSource.trim().length === 0) {
    throw new RangeError('wgslSource must be a non-empty WGSL string');
  }

  let opts = {};
  if (meta !== undefined && meta !== null) {
    if (typeof meta !== 'object' || Array.isArray(meta)) {
      throw new TypeError('meta must be an object');
    }
    opts = meta;
  }

  let label = null;
  if (opts.label !== undefined && opts.label !== null) {
    if (typeof opts.label !== 'string') {
      throw new TypeError('meta.label must be a string');
    }
    label = opts.label;
  }

  let entryPoint = null;
  if (opts.entryPoint !== undefined && opts.entryPoint !== null) {
    if (typeof opts.entryPoint !== 'string') {
      throw new TypeError('meta.entryPoint must be a string');
    }
    entryPoint = opts.entryPoint;
  }

  let source = 'dynamic';
  if (opts.source !== undefined && opts.source !== null) {
    if (typeof opts.source !== 'string' || opts.source.trim().length === 0) {
      throw new TypeError('meta.source must be a non-empty string');
    }
    source = opts.source;
  }

  let tags = [];
  if (opts.tags !== undefined && opts.tags !== null) {
    if (!Array.isArray(opts.tags) || opts.tags.some((t) => typeof t !== 'string')) {
      throw new TypeError('meta.tags must be an array of strings');
    }
    tags = opts.tags.slice();
  }

  const byteLength = textEncoder.encode(wgslSource).byteLength;
  const revision = prevEntry ? prevEntry.revision + 1 : 1;

  return {
    hash: key,
    hashHex,
    stage: stageName,
    stageCode,
    wgsl: wgslSource,
    length: byteLength,
    dynamic: true,
    override: Boolean(opts.override),
    source,
    label,
    entryPoint,
    tags,
    revision,
    registeredAt: Date.now(),
  };
}

function formatDynamicLookupRecord(entry, overridesStatic) {
  return {
    hash: entry.hash,
    hashHex: entry.hashHex,
    stage: entry.stage,
    stageCode: entry.stageCode,
    wgsl: entry.wgsl,
    length: entry.length,
    dynamic: true,
    overridesStatic,
    packName: null,
    source: entry.source,
    label: entry.label,
    entryPoint: entry.entryPoint,
    tags: entry.tags.slice(),
    revision: entry.revision,
    registeredAt: entry.registeredAt,
  };
}

/**
 * Encode the 16-byte WGPK header and version-appropriate TOC records (`20` bytes per entry for
 * v1, `24` bytes per entry for v2) directly from `{ hash, stage, offset, length }` descriptors
 * without materializing payload buffers (matches `encode_pack_toc` in `shaders/src/pack.rs`).
 *
 * @param {Array<{
 *   hash: bigint | string | number,
 *   stage: number | string,
 *   offset: number | bigint,
 *   length: number | bigint,
 * }>} entries
 * @param {{
 *   version?: 1 | 2 | '1' | '2' | 'v1' | 'v2' | 'auto',
 *   flags?: number,
 * }} [options]
 * @returns {ArrayBuffer}
 */
export function buildWgpkTocBuffer(entries, options = {}) {
  if (!Array.isArray(entries)) {
    throw new TypeError('entries must be an array');
  }
  const opts = options && typeof options === 'object' ? options : {};
  const flags = opts.flags === undefined ? 0 : opts.flags;
  if (!Number.isInteger(flags) || flags < 0 || flags > U32_MAX) {
    throw new RangeError(`flags must be a uint32 integer, got: ${flags}`);
  }
  if (entries.length > U32_MAX) {
    throw new RangeError('entry count exceeds u32::MAX');
  }

  const requestedVersion = resolvePackVersionOption(opts.version, 'auto');
  const normalizedEntries = new Array(entries.length);
  const u32MaxBig = BigInt(U32_MAX);
  let requiresV2 = false;

  const v1DataStart = WGPK_HEADER_BYTES + entries.length * WGPK_ENTRY_BYTES_V1;
  if (v1DataStart > U32_MAX) {
    requiresV2 = true;
  }

  for (let i = 0; i < entries.length; i += 1) {
    const item = entries[i];
    if (!item || typeof item !== 'object') {
      throw new TypeError(`Entry ${i} must be an object`);
    }
    const hash = normalizeHash(item.hash);
    const stageCode = resolveStageCode(item.stage);
    const offsetBig = normalizeU64Offset(item.offset, `entry ${i} offset`);

    let length;
    if (typeof item.length === 'bigint') {
      if (item.length < 0n || item.length > u32MaxBig) {
        throw new RangeError(`Entry ${i} length out of u32 range: ${item.length.toString()}`);
      }
      length = Number(item.length);
    } else if (
      typeof item.length === 'number' &&
      Number.isInteger(item.length) &&
      item.length >= 0 &&
      item.length <= U32_MAX
    ) {
      length = item.length;
    } else {
      throw new RangeError(`Entry ${i} length must be a uint32 integer, got: ${item.length}`);
    }

    const endBig = offsetBig + BigInt(length);
    if (endBig > U64_MAX) {
      throw new RangeError(`Entry ${i} payload range [${offsetBig}..+${length}] overflows u64`);
    }
    if (offsetBig > u32MaxBig || endBig > u32MaxBig) {
      requiresV2 = true;
    }
    normalizedEntries[i] = { hash, stageCode, offsetBig, length, endBig };
  }

  const version =
    requestedVersion === 'auto'
      ? requiresV2
        ? WGPK_VERSION_V2
        : WGPK_VERSION_V1
      : requestedVersion;

  if (version === WGPK_VERSION_V1) {
    if (v1DataStart > U32_MAX) {
      throw new RangeError('WGPK v1 header + entry table size exceeds u32::MAX');
    }
    for (let i = 0; i < normalizedEntries.length; i += 1) {
      const e = normalizedEntries[i];
      if (e.offsetBig > u32MaxBig || e.endBig > u32MaxBig) {
        throw new RangeError(
          `Entry ${i} payload range [${e.offsetBig}..+${e.length}] overflows WGPK v1 u32 byte offsets`
        );
      }
    }
  }

  const entryBytes = wgpkEntryBytes(version);
  const tocBytes = WGPK_HEADER_BYTES + normalizedEntries.length * entryBytes;
  if (!Number.isSafeInteger(tocBytes)) {
    throw new RangeError('WGPK TOC byte size exceeds safe integer bounds');
  }

  const buffer = new ArrayBuffer(tocBytes);
  const out = new Uint8Array(buffer);
  const view = new DataView(buffer);

  // Header (16 bytes)
  out[0] = WGPK_MAGIC_BYTES[0];
  out[1] = WGPK_MAGIC_BYTES[1];
  out[2] = WGPK_MAGIC_BYTES[2];
  out[3] = WGPK_MAGIC_BYTES[3];
  view.setUint32(4, version >>> 0, true);
  view.setUint32(8, normalizedEntries.length >>> 0, true);
  view.setUint32(12, flags >>> 0, true);

  // Entry table (20 bytes per record for v1, 24 bytes per record for v2)
  for (let i = 0; i < normalizedEntries.length; i += 1) {
    const { hash, stageCode, offsetBig, length } = normalizedEntries[i];
    const base = WGPK_HEADER_BYTES + i * entryBytes;

    view.setBigUint64(base, hash, true);
    out[base + 8] = stageCode;
    out[base + 9] = 0;
    out[base + 10] = 0;
    out[base + 11] = 0;
    if (version === WGPK_VERSION_V2) {
      view.setUint32(base + 12, length >>> 0, true);
      view.setBigUint64(base + 16, offsetBig, true);
    } else {
      view.setUint32(base + 12, Number(offsetBig) >>> 0, true);
      view.setUint32(base + 16, length >>> 0, true);
    }
  }

  return buffer;
}

/**
 * Serialize shader entries into a binary `.wgsl.pack` `ArrayBuffer` matching
 * `write_pack` / `write_pack_with_layout` in `shaders/src/pack.rs` byte-for-byte.
 *
 * @param {Array<{ hash: bigint | string | number, stage: number | string, wgsl: string | Uint8Array }>} entries
 * @param {{
 *   flags?: number,
 *   version?: 1 | 2 | '1' | '2' | 'v1' | 'v2' | 'auto',
 * }} [options]
 * @returns {ArrayBuffer}
 */
export function buildWgslPackBuffer(entries, options = {}) {
  if (!Array.isArray(entries)) {
    throw new TypeError('entries must be an array');
  }
  const opts = options && typeof options === 'object' ? options : {};
  const flags = opts.flags === undefined ? 0 : opts.flags;
  if (!Number.isInteger(flags) || flags < 0 || flags > U32_MAX) {
    throw new RangeError(`flags must be a uint32 integer, got: ${flags}`);
  }
  if (entries.length > U32_MAX) {
    throw new RangeError('entry count exceeds u32::MAX');
  }

  // Default to WGPK v1 so all existing callers remain 100% byte-identical.
  const requestedVersion = resolvePackVersionOption(opts.version, WGPK_VERSION_V1);

  const encodedEntries = new Array(entries.length);
  let totalPayloadBytes = 0;

  for (let i = 0; i < entries.length; i += 1) {
    const item = entries[i];
    if (!item || typeof item !== 'object') {
      throw new TypeError(`Entry ${i} must be an object`);
    }
    const hash = normalizeHash(item.hash);
    const stageCode = resolveStageCode(item.stage);
    const payload =
      typeof item.wgsl === 'string'
        ? textEncoder.encode(item.wgsl)
        : toBinaryView(item.wgsl);

    if (payload.byteLength > U32_MAX) {
      throw new RangeError(`Entry ${i} payload exceeds u32::MAX bytes`);
    }
    totalPayloadBytes += payload.byteLength;
    encodedEntries[i] = { hash, stageCode, payload };
  }

  const v1DataStart = WGPK_HEADER_BYTES + encodedEntries.length * WGPK_ENTRY_BYTES_V1;
  const v1TotalSize = v1DataStart + totalPayloadBytes;
  const version =
    requestedVersion === 'auto'
      ? v1TotalSize <= U32_MAX
        ? WGPK_VERSION_V1
        : WGPK_VERSION_V2
      : requestedVersion;

  const entryBytes = wgpkEntryBytes(version);
  const dataStart = WGPK_HEADER_BYTES + encodedEntries.length * entryBytes;
  const totalSize = dataStart + totalPayloadBytes;
  if (version === WGPK_VERSION_V1 && totalSize > U32_MAX) {
    throw new RangeError('Total WGPK pack size exceeds u32::MAX byte offsets');
  }
  if (!Number.isSafeInteger(totalSize)) {
    throw new RangeError('Total WGPK pack size exceeds safe integer bounds');
  }

  const buffer = new ArrayBuffer(totalSize);
  const out = new Uint8Array(buffer);
  const view = new DataView(buffer);

  // Header (16 bytes)
  out[0] = WGPK_MAGIC_BYTES[0];
  out[1] = WGPK_MAGIC_BYTES[1];
  out[2] = WGPK_MAGIC_BYTES[2];
  out[3] = WGPK_MAGIC_BYTES[3];
  view.setUint32(4, version >>> 0, true);
  view.setUint32(8, encodedEntries.length >>> 0, true);
  view.setUint32(12, flags >>> 0, true);

  // Entry table (20 bytes each for v1, 24 bytes each for v2) + payloads
  let cursor = dataStart;
  for (let i = 0; i < encodedEntries.length; i += 1) {
    const { hash, stageCode, payload } = encodedEntries[i];
    const base = WGPK_HEADER_BYTES + i * entryBytes;

    view.setBigUint64(base, hash, true);
    out[base + 8] = stageCode;
    out[base + 9] = 0;
    out[base + 10] = 0;
    out[base + 11] = 0;
    if (version === WGPK_VERSION_V2) {
      view.setUint32(base + 12, payload.byteLength >>> 0, true);
      view.setBigUint64(base + 16, BigInt(cursor), true);
    } else {
      view.setUint32(base + 12, cursor >>> 0, true);
      view.setUint32(base + 16, payload.byteLength >>> 0, true);
    }

    out.set(payload, cursor);
    cursor += payload.byteLength;
  }

  return buffer;
}

/**
 * Parsed view over a single `.wgsl.pack` archive with optional live dynamic WGSL overlays.
 */
export class WgslPack {
  /**
   * @param {object} init
   */
  constructor({
    version,
    entryBytes,
    flags,
    byteLength,
    dataStart,
    payloadBytes,
    lazyDecode,
    entryList,
    byHash,
    byHex,
    stageCounts,
    getDecodedCount,
    mode = 'buffer',
    url = null,
    fetchImpl = null,
    fetchInit = undefined,
    maxCacheBytes = undefined,
    maxCachedWgslBytes = DEFAULT_MAX_CACHED_WGSL_BYTES,
    maxCacheEntries = null,
    maxConcurrentRangeFetches = DEFAULT_MAX_CONCURRENT_RANGE_FETCHES,
    maxRangeFetchRetries = DEFAULT_MAX_RANGE_FETCH_RETRIES,
    rangeRetryDelayMs = DEFAULT_RANGE_RETRY_DELAY_MS,
    sleepFn = null,
    maxCoalesceGapBytes = undefined,
    prefetchMaxGapBytes = undefined,
    maxCoalesceSpanBytes = undefined,
    prefetchMaxSpanBytes = undefined,
    initialRangeFetches = 0,
    initialRangeBytesFetched = 0,
    initialRangeFetchRetries = 0,
    initialRangeBytes = DEFAULT_INITIAL_RANGE_BYTES,
    maxFullFallbackBytes = DEFAULT_MAX_FULL_FALLBACK_BYTES,
    etag = null,
    lastModified = null,
    totalSize = null,
    wgslCache = null,
    pinnedWgslCache = null,
    rangedDecodeCounter = null,
    bufferTextState = null,
  }) {
    this.version = version;
    this.entryBytes = entryBytes ?? wgpkEntryBytes(version) ?? WGPK_ENTRY_BYTES;
    this.flags = flags;
    this.byteLength = byteLength;
    this.dataStart = dataStart;
    this.payloadBytes = payloadBytes;
    this.lazyDecode = lazyDecode;
    this.mode = mode;
    this.url = url;
    this.etag = normalizeHeaderString(etag);
    this.lastModified = normalizeHeaderString(lastModified);
    this.totalSize =
      totalSize !== undefined &&
      totalSize !== null &&
      Number.isSafeInteger(totalSize) &&
      totalSize >= 0
        ? totalSize
        : mode === 'buffer'
          ? byteLength
          : null;
    this.entries = byHash;
    this._byHex = byHex;
    this._entryList = entryList;
    this._stageCounts = stageCounts;
    this._getDecodedCount = getDecodedCount;
    this._dynamicEntries = new Map();
    this._dynamicHits = 0;

    // Ranged streaming state (bounded LRU cache with hot-hash pinning, FIFO concurrency
    // semaphore, transient retry, coalesced range prefetch, stale-index reload guard,
    // and concurrent in-flight deduplication).
    this._fetch = fetchImpl;
    this._fetchInit = fetchInit;
    this._initialRangeBytes =
      mode === 'ranged' ? validateInitialRangeBytes(initialRangeBytes) : null;
    this._maxFullFallbackBytes =
      mode === 'ranged' ? validateMaxFullFallbackBytes(maxFullFallbackBytes) : null;
    this._maxCachedWgslBytes =
      mode === 'ranged'
        ? resolveMaxCacheBytes({ maxCacheBytes, maxCachedWgslBytes })
        : null;
    this._maxCacheEntries =
      mode === 'ranged' ? validateMaxCacheEntries(maxCacheEntries, null) : null;
    this._maxConcurrentRangeFetches =
      mode === 'ranged' ? validateMaxConcurrentRangeFetches(maxConcurrentRangeFetches) : null;
    this._maxRangeFetchRetries =
      mode === 'ranged' ? validateMaxRangeFetchRetries(maxRangeFetchRetries) : null;
    this._rangeRetryDelayMs =
      mode === 'ranged' ? validateRangeRetryDelayMs(rangeRetryDelayMs) : null;
    this._sleepFn = mode === 'ranged' ? validateSleepFn(sleepFn) : null;
    this._prefetchMaxGapBytes =
      mode === 'ranged'
        ? resolvePrefetchMaxGapBytes({ maxCoalesceGapBytes, prefetchMaxGapBytes })
        : null;
    this._prefetchMaxSpanBytes =
      mode === 'ranged'
        ? resolvePrefetchMaxSpanBytes({ maxCoalesceSpanBytes, prefetchMaxSpanBytes })
        : null;
    this._pinnedHashes = new Set();
    this._pinnedWgslCache = pinnedWgslCache || new Map();
    this._pinnedWgslBytes = 0;
    this._wgslCache = wgslCache || new Map();
    this._evictableWgslBytes = 0;
    this._cachedWgslBytes = 0;
    this._mruKey = undefined;
    this._inFlightFetches = new Map();
    this._rangeFetchWaiters = [];
    this._activeRangeFetches = 0;
    this._peakConcurrentRangeFetches = 0;
    this._rangeFetches = sanitizeNonNegativeInt(initialRangeFetches, 0);
    this._rangeBytesFetched = sanitizeNonNegativeInt(initialRangeBytesFetched, 0);
    this._rangeFetchRetries = sanitizeNonNegativeInt(initialRangeFetchRetries, 0);
    this._staleIndexReloads = 0;
    this._indexGeneration = 0;
    this._indexReloadPromise = null;
    this._onIndexReloaded = null;
    this._cacheHits = 0;
    this._cacheMisses = 0;
    this._lruHits = 0;
    this._lruMisses = 0;
    this._wgslCacheEvictions = 0;
    this._prefetchBatches = 0;
    this._prefetchShaders = 0;
    this._coalescedRangeFetches = 0;
    this._coalescedEntriesFetched = 0;
    this._httpRequestsSavedByCoalescing = 0;
    this._coalescedGapBytes = 0;
    this._inFlightCoalesced = 0;
    this._rangedDecodeCounter = rangedDecodeCounter;
    // Post-compile WGSL text eviction state & telemetry (#2162, #1077).
    this._bufferTextState = bufferTextState || null;
    this._wgslTextBytesReleased = 0;
    this._wgslTextReleases = 0;
    this._wgslTextLazyRedecodes = 0;
    if (this._bufferTextState) {
      this._bufferTextState.onMutate = () => {
        this._statsVersion += 1;
      };
    }
    // W363: Monotonic mutation version + cached stats snapshot for O(1) steady-state stats().
    this._statsVersion = 0;
    this._cachedStats = null;
    this._cachedStatsVersion = -1;
    this._cachedStatsDecodedCount = -1;
  }

  /**
   * Parse a full in-memory `.wgsl.pack` buffer (alias for `parseWgslPack`).
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
   * @param {{ lazyDecode?: boolean }} [options]
   * @returns {WgslPack}
   */
  static loadFromBuffer(buffer, options = {}) {
    return parseWgslPack(buffer, options);
  }

  /**
   * Load only the WGPK header + entry table via HTTP `206 Range` requests.
   * Seamlessly falls back to full in-memory buffer mode if the server responds `200 OK`.
   *
   * @param {string} url
   * @param {typeof fetch | object} [fetchImplOrOptions]
   * @param {object} [maybeOptions]
   * @returns {Promise<WgslPack>}
   */
  static loadIndexFromUrl(url, fetchImplOrOptions, maybeOptions) {
    return loadIndexFromUrl(url, fetchImplOrOptions, maybeOptions);
  }

  /**
   * Load a `.wgsl.pack` from a URL. When `options.streamThresholdBytes > 0`, automatically
   * switches to HTTP Range index-only mode when the remote pack exceeds the threshold.
   *
   * @param {string} url
   * @param {typeof fetch | object} [fetchImplOrOptions]
   * @param {object} [maybeOptions]
   * @returns {Promise<WgslPack>}
   */
  static loadFromUrl(url, fetchImplOrOptions, maybeOptions) {
    return loadFromUrl(url, fetchImplOrOptions, maybeOptions);
  }

  /** Number of distinct resolvable shader entries (static + non-overriding dynamic). */
  get size() {
    let extraDynamic = 0;
    for (const key of this._dynamicEntries.keys()) {
      if (!this.entries.has(key)) {
        extraDynamic += 1;
      }
    }
    return this._entryList.length + extraDynamic;
  }

  /**
   * Register a live dynamic WGSL shader on this pack instance.
   * Pass `{ override: true }` in `meta` to supersede an existing static pack entry
   * while preserving the static entry for `clearDynamicShaders()`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} stage
   * @param {string} wgslSource
   * @param {{ override?: boolean, label?: string, entryPoint?: string, source?: string, tags?: string[] }} [meta]
   */
  registerDynamicShader(hash, stage, wgslSource, meta) {
    const key = normalizeHash(hash);
    const hasStatic = this.entries.has(key);
    const prevDynamic = this._dynamicEntries.get(key);
    const entry = createDynamicShaderEntry(hash, stage, wgslSource, meta, prevDynamic);

    if (hasStatic && !entry.override) {
      throw new Error(
        `Shader ${entry.hashHex} already exists in static WGSL pack; pass { override: true } to supersede it`
      );
    }

    this._dynamicEntries.set(key, entry);
    this._statsVersion += 1;
    return formatDynamicLookupRecord(entry, hasStatic);
  }

  /**
   * Check whether a dynamic shader is registered for `hash` (and optional `stage`).
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {boolean}
   */
  hasDynamicShader(hash, stage) {
    const key = normalizeHash(hash);
    const entry = this._dynamicEntries.get(key);
    if (!entry) {
      return false;
    }
    if (stage !== undefined && stage !== null) {
      return entry.stageCode === resolveStageCode(stage);
    }
    return true;
  }

  /**
   * Remove a single dynamic shader by hash, restoring any underlying static pack entry.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  unregisterDynamicShader(hash) {
    const key = normalizeHash(hash);
    const removed = this._dynamicEntries.delete(key);
    if (removed) {
      this._statsVersion += 1;
    }
    return removed;
  }

  /**
   * Remove all registered dynamic shaders while preserving static pack entries.
   *
   * @returns {number} Number of dynamic shaders cleared
   */
  clearDynamicShaders() {
    const count = this._dynamicEntries.size;
    if (count > 0) {
      this._dynamicEntries.clear();
      this._statsVersion += 1;
    }
    return count;
  }

  /**
   * Look up a shader entry by hash (and optional stage), checking dynamic overrides first
   * and falling back to static pack entries.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   */
  lookup(hash, stage) {
    const key = normalizeHash(hash);
    const expectedStage =
      stage !== undefined && stage !== null ? resolveStageCode(stage) : null;

    const dynamicEntry = this._dynamicEntries.get(key);
    if (dynamicEntry) {
      if (expectedStage !== null && dynamicEntry.stageCode !== expectedStage) {
        return undefined;
      }
      this._dynamicHits += 1;
      this._statsVersion += 1;
      return formatDynamicLookupRecord(dynamicEntry, this.entries.has(key));
    }

    const staticEntry = this.entries.get(key);
    if (!staticEntry) {
      return undefined;
    }
    if (expectedStage !== null && staticEntry.stageCode !== expectedStage) {
      return undefined;
    }
    return {
      hash: staticEntry.hash,
      hashHex: staticEntry.hashHex,
      stage: staticEntry.stage,
      stageCode: staticEntry.stageCode,
      wgsl: staticEntry.wgsl,
      length: staticEntry.length,
      offset: staticEntry.offset,
      dynamic: false,
      overridesStatic: false,
      packName: null,
      source: 'pack',
      label: null,
      entryPoint: null,
      tags: [],
      revision: 0,
      registeredAt: null,
    };
  }

  /**
   * Alias for `lookup(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   */
  get(hash, stage) {
    return this.lookup(hash, stage);
  }

  /**
   * Look up a shader entry by `BigInt` hash, hex string, or safe integer.
   * Checks dynamic overrides first, then static pack entries.
   *
   * @param {bigint | string | number} hash
   */
  getByHash(hash) {
    if (this._dynamicEntries.size > 0) {
      const key = normalizeHash(hash);
      const dynamicEntry = this._dynamicEntries.get(key);
      if (dynamicEntry) {
        this._dynamicHits += 1;
        this._statsVersion += 1;
        return formatDynamicLookupRecord(dynamicEntry, this.entries.has(key));
      }
      return this.entries.get(key);
    }
    if (typeof hash === 'string') {
      const fast = this._byHex.get(hash);
      if (fast !== undefined) {
        return fast;
      }
    }
    const key = normalizeHash(hash);
    return this.entries.get(key);
  }

  /**
   * Alias for `getByHash(hash)`.
   *
   * @param {bigint | string | number} hash
   */
  getEntry(hash) {
    return this.getByHash(hash);
  }

  /**
   * Check whether a decoded JS WGSL string is currently retained in heap memory for `hash`
   * without triggering lazy UTF-8 decoding or HTTP Range fetches.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  hasCachedWgslText(hash) {
    const key = this._resolveHashKey(hash);
    if (this._dynamicEntries.has(key)) {
      return true;
    }
    if (this.mode === 'ranged') {
      return this._pinnedWgslCache.has(key) || this._wgslCache.has(key);
    }
    if (this._bufferTextState) {
      return this._bufferTextState.isCached(key);
    }
    return false;
  }

  /**
   * Release the decoded JS WGSL string for `hash` from heap memory while preserving its
   * compact binary index (`{ hash, stage, offset, length }`) so subsequent `getWgsl(hash)`
   * or `getWgslAsync(hash)` calls transparently lazily re-decode on demand (#2162).
   *
   * In `ranged` mode, pinned hashes stay resident unless `{ includePinned: true }`.
   *
   * @param {bigint | string | number} hash
   * @param {{ includePinned?: boolean }} [options]
   * @returns {boolean} True if a cached decoded WGSL string was released
   */
  releaseWgslText(hash, options = {}) {
    const key = this._resolveHashKey(hash);
    if (this.mode !== 'ranged') {
      if (!this._bufferTextState) {
        return false;
      }
      const releasedBytes = this._bufferTextState.release(key);
      if (releasedBytes < 0) {
        return false;
      }
      this._statsVersion += 1;
      return true;
    }

    const includePinned = Boolean(options && options.includePinned);
    let releasedBytes = -1;
    const cached = this._wgslCache.get(key);
    if (cached !== undefined) {
      this._wgslCache.delete(key);
      this._evictableWgslBytes -= cached.byteLength;
      this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
      if (this._mruKey === key) {
        this._mruKey = undefined;
      }
      releasedBytes = cached.byteLength;
    } else if (includePinned) {
      const pinned = this._pinnedWgslCache.get(key);
      if (pinned !== undefined) {
        this._pinnedWgslCache.delete(key);
        this._pinnedWgslBytes -= pinned.byteLength;
        this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
        releasedBytes = pinned.byteLength;
      }
    }

    if (releasedBytes < 0) {
      return false;
    }

    const staticEntry = this.entries.get(key);
    if (staticEntry) {
      staticEntry._wasTextReleased = true;
    }
    this._wgslTextBytesReleased += releasedBytes;
    this._wgslTextReleases += 1;
    this._statsVersion += 1;
    return true;
  }

  /**
   * Release all decoded JS WGSL strings currently retained in this pack while keeping
   * the compact binary index intact for transparent lazy re-decode (#2162).
   *
   * @param {{ includePinned?: boolean }} [options]
   * @returns {{ releasedCount: number, bytesReleased: number }}
   */
  releaseAllWgslText(options = {}) {
    if (this.mode !== 'ranged') {
      if (!this._bufferTextState) {
        return { releasedCount: 0, bytesReleased: 0 };
      }
      const res = this._bufferTextState.releaseAll();
      if (res.releasedCount > 0) {
        this._statsVersion += 1;
      }
      return res;
    }

    const includePinned = Boolean(options && options.includePinned);
    let releasedCount = 0;
    let bytesReleased = 0;

    if (this._wgslCache.size > 0) {
      for (const [key, entry] of this._wgslCache.entries()) {
        releasedCount += 1;
        bytesReleased += entry.byteLength;
        const staticEntry = this.entries.get(key);
        if (staticEntry) {
          staticEntry._wasTextReleased = true;
        }
      }
      this._wgslCache.clear();
      this._evictableWgslBytes = 0;
      this._mruKey = undefined;
    }

    if (includePinned && this._pinnedWgslCache.size > 0) {
      for (const [key, entry] of this._pinnedWgslCache.entries()) {
        releasedCount += 1;
        bytesReleased += entry.byteLength;
        const staticEntry = this.entries.get(key);
        if (staticEntry) {
          staticEntry._wasTextReleased = true;
        }
      }
      this._pinnedWgslCache.clear();
      this._pinnedWgslBytes = 0;
    }

    this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
    if (releasedCount > 0) {
      this._wgslTextBytesReleased += bytesReleased;
      this._wgslTextReleases += releasedCount;
      this._statsVersion += 1;
    }
    return { releasedCount, bytesReleased };
  }

  /**
   * Retrieve the UTF-8 WGSL source string for a shader hash.
   * Checks dynamic overrides first; in `lazyDecode` mode, decodes static entries on first access.
   *
   * @param {bigint | string | number} hash
   * @returns {string | undefined}
   */
  getSource(hash) {
    const entry = this.getByHash(hash);
    return entry ? entry.wgsl : undefined;
  }

  /**
   * Alias for `getSource(hash)`.
   *
   * @param {bigint | string | number} hash
   * @returns {string | undefined}
   */
  getWgsl(hash) {
    return this.getSource(hash);
  }

  _resolveHashKey(hash) {
    if (typeof hash === 'bigint') {
      if (hash < 0n || hash > U64_MAX) {
        throw new RangeError(`Shader hash out of u64 range: ${hash.toString()}`);
      }
      return hash;
    }
    if (typeof hash === 'string') {
      const fast = this._byHex.get(hash);
      if (fast !== undefined) {
        return fast.hash;
      }
    }
    return normalizeHash(hash);
  }

  _touchWgslCache(key) {
    const pinned = this._pinnedWgslCache.get(key);
    if (pinned !== undefined) {
      return pinned;
    }
    const cached = this._wgslCache.get(key);
    if (cached === undefined) {
      return undefined;
    }
    if (this._mruKey !== key && this._wgslCache.size > 1) {
      this._wgslCache.delete(key);
      this._wgslCache.set(key, cached);
    }
    this._mruKey = key;
    return cached;
  }

  _evictOldestUnpinned() {
    const oldestKey = this._wgslCache.keys().next().value;
    const oldest = this._wgslCache.get(oldestKey);
    this._wgslCache.delete(oldestKey);
    this._evictableWgslBytes -= oldest.byteLength;
    this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
    this._wgslCacheEvictions += 1;
    this._statsVersion += 1;
    if (this._mruKey === oldestKey) {
      this._mruKey = undefined;
    }
  }

  _enforceCacheLimits(extraBytes = 0, extraEntries = 0) {
    const maxBytes = this._maxCachedWgslBytes;
    const maxEntries = this._maxCacheEntries;
    while (this._wgslCache.size > 0) {
      const overBytes =
        maxBytes !== null && this._cachedWgslBytes + extraBytes > maxBytes;
      const overEntries =
        maxEntries !== null &&
        this._pinnedWgslCache.size + this._wgslCache.size + extraEntries > maxEntries;
      if (!overBytes && !overEntries) {
        break;
      }
      this._evictOldestUnpinned();
    }
  }

  _putWgslCache(key, wgsl, byteLength) {
    const existingPinned = this._pinnedWgslCache.get(key);
    if (existingPinned !== undefined) {
      this._pinnedWgslCache.delete(key);
      this._pinnedWgslBytes -= existingPinned.byteLength;
      this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
      this._statsVersion += 1;
    }
    const existing = this._wgslCache.get(key);
    if (existing !== undefined) {
      this._wgslCache.delete(key);
      this._evictableWgslBytes -= existing.byteLength;
      this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
      this._statsVersion += 1;
      if (this._mruKey === key) {
        this._mruKey = undefined;
      }
    }

    // Pinned hashes stay resident in _pinnedWgslCache and are never evicted by LRU pressure.
    if (this._pinnedHashes.has(key)) {
      this._enforceCacheLimits(byteLength, 1);
      this._pinnedWgslCache.set(key, { wgsl, byteLength });
      this._pinnedWgslBytes += byteLength;
      this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
      this._statsVersion += 1;
      return true;
    }

    const maxBytes = this._maxCachedWgslBytes;
    const maxEntries = this._maxCacheEntries;
    if (maxBytes === null || maxBytes === 0 || this._pinnedWgslBytes + byteLength > maxBytes) {
      return false;
    }
    if (maxEntries !== null && (maxEntries === 0 || this._pinnedWgslCache.size + 1 > maxEntries)) {
      return false;
    }

    this._enforceCacheLimits(byteLength, 1);

    this._wgslCache.set(key, { wgsl, byteLength });
    this._evictableWgslBytes += byteLength;
    this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
    this._mruKey = key;
    this._statsVersion += 1;
    return true;
  }

  /**
   * Pin one or more shader hashes so their decoded WGSL payloads stay resident in cache
   * and are never evicted under LRU byte or entry-count pressure (e.g. hot Bink2 / UI shaders).
   *
   * @param {Iterable<bigint | string | number>} hashes
   * @returns {number} Number of hashes newly pinned
   */
  pinHashes(hashes) {
    const keys = normalizeHashIterable(hashes);
    let added = 0;
    let mutated = false;
    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      if (!this._pinnedHashes.has(key)) {
        this._pinnedHashes.add(key);
        added += 1;
        mutated = true;
      }
      const cached = this._wgslCache.get(key);
      if (cached !== undefined) {
        this._wgslCache.delete(key);
        this._evictableWgslBytes -= cached.byteLength;
        if (this._mruKey === key) {
          this._mruKey = undefined;
        }
        this._pinnedWgslCache.set(key, cached);
        this._pinnedWgslBytes += cached.byteLength;
        this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
        mutated = true;
      }
    }
    if (mutated) {
      this._statsVersion += 1;
    }
    return added;
  }

  /**
   * Unpin one or more shader hashes so they become eligible for standard O(1) LRU eviction.
   * Resident entries are moved back into the evictable LRU cache (or evicted immediately if
   * the configured byte/entry budget cannot accommodate them).
   *
   * @param {Iterable<bigint | string | number>} hashes
   * @returns {number} Number of hashes newly unpinned
   */
  unpinHashes(hashes) {
    const keys = normalizeHashIterable(hashes);
    let removed = 0;
    let mutated = false;
    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      if (this._pinnedHashes.delete(key)) {
        removed += 1;
        mutated = true;
      }
      const pinned = this._pinnedWgslCache.get(key);
      if (pinned !== undefined) {
        this._pinnedWgslCache.delete(key);
        this._pinnedWgslBytes -= pinned.byteLength;
        this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
        mutated = true;

        const maxBytes = this._maxCachedWgslBytes;
        const maxEntries = this._maxCacheEntries;
        const fitsBytes =
          maxBytes !== null &&
          maxBytes > 0 &&
          this._pinnedWgslBytes + pinned.byteLength <= maxBytes;
        const fitsEntries =
          maxEntries === null ||
          (maxEntries > 0 && this._pinnedWgslCache.size + 1 <= maxEntries);
        if (fitsBytes && fitsEntries) {
          this._enforceCacheLimits(pinned.byteLength, 1);
          this._wgslCache.set(key, pinned);
          this._evictableWgslBytes += pinned.byteLength;
          this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
          this._mruKey = key;
        } else {
          this._wgslCacheEvictions += 1;
        }
      }
    }
    if (mutated) {
      this._statsVersion += 1;
    }
    return removed;
  }

  /**
   * Return whether a shader hash is currently pinned.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  isPinned(hash) {
    return this._pinnedHashes.has(this._resolveHashKey(hash));
  }

  /**
   * Return all currently pinned shader hashes as an array of `BigInt` keys.
   *
   * @returns {bigint[]}
   */
  getPinnedHashes() {
    return Array.from(this._pinnedHashes);
  }

  /**
   * Clear decoded WGSL strings cached in the ranged-mode LRU cache.
   * Pass `{ keepPinned: true }` to retain cached pinned payloads.
   * Does not unload the entry table, pinned hash set, or dynamic shaders.
   *
   * @param {{ keepPinned?: boolean }} [options]
   * @returns {number} Number of cached WGSL entries cleared
   */
  clearWgslCache(options = {}) {
    const keepPinned = Boolean(options && options.keepPinned);
    let count = this._wgslCache.size;
    this._wgslCache.clear();
    this._evictableWgslBytes = 0;
    this._mruKey = undefined;
    if (!keepPinned) {
      count += this._pinnedWgslCache.size;
      this._pinnedWgslCache.clear();
      this._pinnedWgslBytes = 0;
    }
    this._cachedWgslBytes = this._pinnedWgslBytes + this._evictableWgslBytes;
    if (count > 0) {
      this._statsVersion += 1;
    }
    return count;
  }

  _acquireRangeFetchSlot(priority = false, signal = null) {
    if (signal && signal.aborted) {
      return Promise.reject(createAbortError(signal));
    }
    const limit = this._maxConcurrentRangeFetches;
    if (limit === null || limit === 0 || this._activeRangeFetches < limit) {
      this._activeRangeFetches += 1;
      if (this._activeRangeFetches > this._peakConcurrentRangeFetches) {
        this._peakConcurrentRangeFetches = this._activeRangeFetches;
      }
      this._statsVersion += 1;
      return null;
    }
    return new Promise((resolve, reject) => {
      const waiter = {
        resolve: () => {
          if (signal && onAbort && typeof signal.removeEventListener === 'function') {
            signal.removeEventListener('abort', onAbort);
          }
          resolve();
        },
      };
      let onAbort = null;
      if (signal && typeof signal.addEventListener === 'function') {
        onAbort = () => {
          const idx = this._rangeFetchWaiters.indexOf(waiter);
          if (idx !== -1) {
            this._rangeFetchWaiters.splice(idx, 1);
            reject(createAbortError(signal));
          }
        };
        signal.addEventListener('abort', onAbort, { once: true });
      }
      if (priority) {
        this._rangeFetchWaiters.unshift(waiter);
      } else {
        this._rangeFetchWaiters.push(waiter);
      }
    });
  }

  _releaseRangeFetchSlot() {
    const limit = this._maxConcurrentRangeFetches;
    if (limit !== null && limit > 0 && this._rangeFetchWaiters.length > 0) {
      // Transfer the slot directly to the next queued waiter so _activeRangeFetches never
      // exceeds the configured limit and new arrivals cannot jump ahead in the queue.
      const next = this._rangeFetchWaiters.shift();
      if (typeof next === 'function') {
        next();
      } else if (next && typeof next.resolve === 'function') {
        next.resolve();
      }
      return;
    }
    if (this._activeRangeFetches > 0) {
      this._activeRangeFetches -= 1;
      this._statsVersion += 1;
    }
  }

  async _withRangeFetchSlot(fn, opts = null) {
    const priority = Boolean(opts && opts.priority === true);
    const signal = opts && opts.signal ? opts.signal : null;
    const waitPromise = this._acquireRangeFetchSlot(priority, signal);
    if (waitPromise !== null) {
      await waitPromise;
    }
    try {
      return await fn();
    } finally {
      this._releaseRangeFetchSlot();
    }
  }

  async _fetchRangeChunkBounded(fetcher, start, end, opts = null) {
    const maxRetries = this._maxRangeFetchRetries ?? 0;
    const baseDelayMs = this._rangeRetryDelayMs ?? DEFAULT_RANGE_RETRY_DELAY_MS;
    const customSleep = this._sleepFn || null;
    const priority = Boolean(opts && opts.priority === true);
    const signal = this._fetchInit && this._fetchInit.signal ? this._fetchInit.signal : null;
    const maxFullFallbackBytes =
      typeof this._maxFullFallbackBytes === 'number'
        ? this._maxFullFallbackBytes
        : DEFAULT_MAX_FULL_FALLBACK_BYTES;
    const requirePartialContent = Boolean(
      this.totalSize !== null &&
        maxFullFallbackBytes > 0 &&
        this.totalSize > maxFullFallbackBytes
    );
    const rangeOptions = { maxFullFallbackBytes, requirePartialContent };

    let attempt = 0;
    for (;;) {
      if (signal && signal.aborted) {
        throw createAbortError(signal);
      }
      try {
        return await this._withRangeFetchSlot(
          () => {
            if (signal && signal.aborted) {
              throw createAbortError(signal);
            }
            this._rangeFetches += 1;
            this._statsVersion += 1;
            return fetchRangeChunk(fetcher, this.url, start, end, this._fetchInit, rangeOptions);
          },
          { priority, signal }
        );
      } catch (err) {
        if (attempt >= maxRetries || !isRetryableRangeFetchError(err, this._fetchInit)) {
          throw err;
        }
        this._rangeFetchRetries += 1;
        this._statsVersion += 1;
        const delayMs = computeRangeRetryDelayMs(baseDelayMs, attempt);
        attempt += 1;
        if (customSleep) {
          await customSleep(delayMs);
        } else {
          await defaultSleep(delayMs, signal);
        }
        if (signal && signal.aborted) {
          throw createAbortError(signal);
        }
      }
    }
  }

  _detectChunkStaleReason(chunk) {
    if (!chunk) {
      return null;
    }
    if (this.etag !== null && chunk.etag !== null && chunk.etag !== this.etag) {
      return `etag mismatch (baseline ${this.etag}, got ${chunk.etag})`;
    }
    const respTotal =
      chunk.totalSize !== undefined && chunk.totalSize !== null
        ? chunk.totalSize
        : chunk.contentRange && chunk.contentRange.total !== null
          ? chunk.contentRange.total
          : null;
    if (this.totalSize !== null && respTotal !== null && respTotal !== this.totalSize) {
      return `total size mismatch (baseline ${this.totalSize}, got ${respTotal})`;
    }
    if (
      this.lastModified !== null &&
      chunk.lastModified !== null &&
      chunk.lastModified !== this.lastModified
    ) {
      return `last-modified mismatch (baseline ${this.lastModified}, got ${chunk.lastModified})`;
    }
    return null;
  }

  _detectErrorStaleReason(err) {
    if (!err || typeof err.httpStatus !== 'number') {
      return null;
    }
    if (err.httpStatus === 412) {
      return 'HTTP 412 Precondition Failed';
    }
    if (err.httpStatus === 416) {
      if (this.etag !== null && err.etag !== null && err.etag !== this.etag) {
        return `HTTP 416 with changed ETag (baseline ${this.etag}, got ${err.etag})`;
      }
      if (
        this.totalSize !== null &&
        err.contentRangeTotal !== null &&
        err.contentRangeTotal !== this.totalSize
      ) {
        return `HTTP 416 with changed total size (baseline ${this.totalSize}, got ${err.contentRangeTotal})`;
      }
      if (
        this.lastModified !== null &&
        err.lastModified !== null &&
        err.lastModified !== this.lastModified
      ) {
        return `HTTP 416 with changed Last-Modified (baseline ${this.lastModified}, got ${err.lastModified})`;
      }
    }
    return null;
  }

  async _reloadStaleIndex(fetcher, observedGeneration) {
    if (this._indexGeneration !== observedGeneration) {
      return;
    }
    if (this._indexReloadPromise !== null) {
      await this._indexReloadPromise;
      return;
    }
    if (!this.url) {
      throw new Error('Cannot reload stale WGPK index: ranged pack has no URL');
    }

    const reloadPromise = (async () => {
      // Immediately drop stale cached WGSL payloads (preserving _pinnedHashes so hot hashes
      // re-pin automatically once re-fetched from the refreshed offsets).
      this.clearWgslCache({ keepPinned: false });

      const initialBytes = Math.max(
        WGPK_HEADER_BYTES,
        this._initialRangeBytes || DEFAULT_INITIAL_RANGE_BYTES
      );
      const initialChunk = await this._fetchRangeChunkBounded(fetcher, 0, initialBytes - 1, {
        priority: true,
      });
      this._rangeBytesFetched += sanitizeNonNegativeInt(initialChunk.bytes.byteLength, 0);
      this._statsVersion += 1;

      let etag = normalizeHeaderString(initialChunk.etag);
      let lastModified = normalizeHeaderString(initialChunk.lastModified);
      let totalBytes =
        initialChunk.totalSize !== undefined && initialChunk.totalSize !== null
          ? initialChunk.totalSize
          : initialChunk.contentRange
            ? initialChunk.contentRange.total
            : null;
      let indexBytes;

      if (initialChunk.status !== 206) {
        totalBytes = initialChunk.bytes.byteLength;
        const fullHeader = parseWgpkHeader(initialChunk.bytes);
        indexBytes = initialChunk.bytes.subarray(0, fullHeader.dataStart);
      } else {
        if (initialChunk.contentRange && initialChunk.contentRange.start !== 0) {
          throw new Error(
            `WGPK stale index reload from "${this.url}" started at byte ${initialChunk.contentRange.start} (expected 0)`
          );
        }
        const header = parseWgpkHeader(initialChunk.bytes);
        const tableEnd = header.dataStart;
        if (totalBytes !== null && totalBytes < tableEnd) {
          throw new Error(
            `WGPK pack truncated on reload: ${totalBytes} total bytes (header + ${header.entryCount} entry table records require at least ${tableEnd} bytes)`
          );
        }
        if (initialChunk.bytes.byteLength >= tableEnd) {
          indexBytes = initialChunk.bytes.slice(0, tableEnd);
        } else {
          indexBytes = new Uint8Array(tableEnd);
          indexBytes.set(initialChunk.bytes, 0);
          let have = initialChunk.bytes.byteLength;

          while (have < tableEnd) {
            const remainder = await this._fetchRangeChunkBounded(fetcher, have, tableEnd - 1, {
              priority: true,
            });
            this._rangeBytesFetched += sanitizeNonNegativeInt(remainder.bytes.byteLength, 0);
            this._statsVersion += 1;

            if (remainder.status !== 206) {
              totalBytes = remainder.bytes.byteLength;
              const fullHeader = parseWgpkHeader(remainder.bytes);
              indexBytes = remainder.bytes.subarray(0, fullHeader.dataStart);
              etag = normalizeHeaderString(remainder.etag) ?? etag;
              lastModified = normalizeHeaderString(remainder.lastModified) ?? lastModified;
              break;
            }

            if (etag !== null && remainder.etag !== null && remainder.etag !== etag) {
              throw new Error(
                `WGPK pack ETag changed mid-reload from "${this.url}": expected ${etag}, got ${remainder.etag}`
              );
            }
            if (etag === null && remainder.etag !== null) {
              etag = remainder.etag;
            }
            if (
              lastModified !== null &&
              remainder.lastModified !== null &&
              remainder.lastModified !== lastModified
            ) {
              throw new Error(
                `WGPK pack Last-Modified changed mid-reload from "${this.url}": expected ${lastModified}, got ${remainder.lastModified}`
              );
            }
            if (lastModified === null && remainder.lastModified !== null) {
              lastModified = remainder.lastModified;
            }

            if (remainder.contentRange) {
              if (remainder.contentRange.start !== have) {
                throw new Error(
                  `WGPK table continuation reload from "${this.url}" started at byte ${remainder.contentRange.start} (expected ${have})`
                );
              }
              if (remainder.contentRange.total !== null) {
                if (totalBytes !== null && remainder.contentRange.total !== totalBytes) {
                  throw new Error(
                    `WGPK pack total size changed mid-reload from "${this.url}": expected ${totalBytes}, got ${remainder.contentRange.total}`
                  );
                }
                totalBytes = remainder.contentRange.total;
              }
            }

            if (remainder.bytes.byteLength === 0) {
              throw new Error(
                `WGPK pack truncated while reloading entry table from "${this.url}": received 0 bytes at offset ${have}/${tableEnd}`
              );
            }

            const copyLen = Math.min(remainder.bytes.byteLength, tableEnd - have);
            indexBytes.set(remainder.bytes.subarray(0, copyLen), have);
            have += copyLen;
          }
        }
      }

      const header = parseWgpkHeader(indexBytes);
      const { rawEntries, stageCounts, totalPayloadBytes, maxEnd } = parseWgpkEntryTable(
        indexBytes,
        header,
        totalBytes
      );

      this.version = header.version;
      this.entryBytes = header.entryBytes;
      this.flags = header.flags;
      this.byteLength = totalBytes !== null ? totalBytes : Math.max(header.dataStart, maxEnd);
      this.dataStart = header.dataStart;
      this.payloadBytes = totalPayloadBytes;
      this._stageCounts = stageCounts;
      this.etag = etag;
      this.lastModified = lastModified;
      this.totalSize = totalBytes;
      this._entryList = populateRangedIndexEntries(
        rawEntries,
        this._wgslCache,
        this._pinnedWgslCache,
        this.entries,
        this._byHex
      );
      this._indexGeneration += 1;
      this._staleIndexReloads += 1;
      this._statsVersion += 1;
      if (typeof this._onIndexReloaded === 'function') {
        this._onIndexReloaded(this);
      }
    })();

    this._indexReloadPromise = reloadPromise;
    try {
      await reloadPromise;
    } finally {
      if (this._indexReloadPromise === reloadPromise) {
        this._indexReloadPromise = null;
      }
    }
  }

  /**
   * Asynchronously resolve the WGSL source string for a shader hash.
   * Works in both full-buffer mode (returns immediately) and ranged index mode
   * (checks bounded LRU cache, deduplicates concurrent in-flight requests, enforces
   * bounded HTTP Range concurrency + transient retry, validates ETag/size against
   * stale index tables with bounded 1x auto-reload, and fetches only the exact
   * `[offset, offset + length - 1]` byte slice via HTTP `206 Range`).
   *
   * @param {bigint | string | number} hash
   * @returns {Promise<string | null>}
   */
  async getWgslAsync(hash) {
    if (this._indexReloadPromise !== null) {
      await this._indexReloadPromise;
    }

    let key;
    let staticEntry;
    if (this._dynamicEntries.size === 0 && typeof hash === 'string') {
      staticEntry = this._byHex.get(hash);
      if (staticEntry !== undefined) {
        key = staticEntry.hash;
      } else {
        key = normalizeHash(hash);
        staticEntry = this.entries.get(key);
      }
    } else {
      key = this._resolveHashKey(hash);
      const dynamicEntry = this._dynamicEntries.get(key);
      if (dynamicEntry) {
        this._dynamicHits += 1;
        this._statsVersion += 1;
        return dynamicEntry.wgsl;
      }
      staticEntry = this.entries.get(key);
    }

    if (!staticEntry) {
      return null;
    }

    if (this.mode !== 'ranged') {
      this._cacheHits += 1;
      this._statsVersion += 1;
      return staticEntry.wgsl;
    }

    // Check bounded in-memory hot-hash LRU cache first.
    const cached = this._touchWgslCache(key);
    if (cached !== undefined) {
      this._cacheHits += 1;
      this._lruHits += 1;
      this._statsVersion += 1;
      return cached.wgsl;
    }

    // Zero-length payload needs no HTTP Range request.
    const length = staticEntry.length >>> 0;
    if (length === 0) {
      this._cacheMisses += 1;
      this._lruMisses += 1;
      this._statsVersion += 1;
      this._putWgslCache(key, '', 0);
      return '';
    }

    // Deduplicate concurrent in-flight fetches (including active coalesced prefetch batches)
    // before acquiring a range fetch slot.
    const existingPromise = this._inFlightFetches.get(key);
    if (existingPromise !== undefined) {
      this._inFlightCoalesced += 1;
      this._statsVersion += 1;
      return existingPromise;
    }

    this._cacheMisses += 1;
    this._lruMisses += 1;
    this._statsVersion += 1;
    const fetcher = this._fetch ?? globalThis.fetch;
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for ranged getWgslAsync');
    }

    // Preserve full non-negative safe integer byte offsets (WGPK v1 u32 and WGPK v2 >4 GiB u64
    // offsets up to Number.MAX_SAFE_INTEGER) without 32-bit wraparound.
    const start = normalizeSafeOffset(staticEntry.offset);
    const end = start + length - 1;
    const generationBeforeFetch = this._indexGeneration;

    let settled = false;
    let fetchPromise;
    fetchPromise = (async () => {
      try {
        let currentEntry = staticEntry;
        let currentStart = start;
        let currentLength = length;
        let currentEnd = end;
        let observedGeneration = generationBeforeFetch;
        let reloaded = false;

        for (;;) {
          let chunk = null;
          let fetchErr = null;
          let staleReason = null;

          try {
            chunk = await this._fetchRangeChunkBounded(fetcher, currentStart, currentEnd);
          } catch (err) {
            fetchErr = err;
            staleReason = this._detectErrorStaleReason(err);
          }

          if (fetchErr !== null) {
            if (staleReason !== null && !reloaded && this.url) {
              reloaded = true;
              await this._reloadStaleIndex(fetcher, observedGeneration);
              observedGeneration = this._indexGeneration;
              const refreshed = this.entries.get(key);
              if (!refreshed) {
                return null;
              }
              currentEntry = refreshed;
              currentLength = refreshed.length >>> 0;
              if (currentLength === 0) {
                this._putWgslCache(key, '', 0);
                return '';
              }
              currentStart = normalizeSafeOffset(refreshed.offset);
              currentEnd = currentStart + currentLength - 1;
              continue;
            }
            throw fetchErr;
          }

          this._rangeBytesFetched += sanitizeNonNegativeInt(chunk.bytes.byteLength, 0);
          this._statsVersion += 1;

          staleReason = this._detectChunkStaleReason(chunk);
          if (staleReason === null && !reloaded && this._indexGeneration !== observedGeneration) {
            staleReason = 'index reloaded while range fetch was in flight';
          }

          if (staleReason === null) {
            let payloadBytes = null;
            if (chunk.status === 206) {
              if (chunk.contentRange && chunk.contentRange.start !== currentStart) {
                throw new Error(
                  `WGPK ranged fetch mismatch for ${currentEntry.hashHex}: expected start byte ${currentStart}, got ${chunk.contentRange.start}`
                );
              }
              if (chunk.bytes.byteLength < currentLength) {
                if (
                  !reloaded &&
                  this.url &&
                  (this.etag !== null || this.totalSize !== null || this.lastModified !== null)
                ) {
                  staleReason = `truncated ranged payload (expected ${currentLength}, got ${chunk.bytes.byteLength})`;
                } else {
                  throw new Error(
                    `WGPK ranged payload truncated for ${currentEntry.hashHex}: expected ${currentLength} bytes, received ${chunk.bytes.byteLength}`
                  );
                }
              } else {
                payloadBytes = chunk.bytes.subarray(0, currentLength);
              }
            } else {
              // Server returned 200 OK (entire pack) instead of 206 Partial Content.
              const sliceEnd = currentStart + currentLength;
              if (chunk.bytes.byteLength < sliceEnd) {
                throw new Error(
                  `WGPK full response truncated for ${currentEntry.hashHex}: need byte range [${currentStart}..${sliceEnd}), received ${chunk.bytes.byteLength} bytes`
                );
              }
              payloadBytes = chunk.bytes.subarray(currentStart, sliceEnd);
            }

            if (staleReason === null) {
              const entryIdx =
                currentEntry._index !== undefined ? currentEntry._index : currentEntry.hashHex;
              let wgsl = null;
              try {
                wgsl = decodeUtf8Strict(payloadBytes, entryIdx);
              } catch (decodeErr) {
                if (!reloaded && this.url) {
                  staleReason = 'invalid UTF-8 payload (possible shifted offsets)';
                } else {
                  throw decodeErr;
                }
              }
              if (staleReason === null) {
                if (!reloaded && this.url && looksLikeMidShaderCorruption(wgsl)) {
                  staleReason = 'mid-shader offset corruption detected';
                } else {
                  if (this._rangedDecodeCounter) {
                    this._rangedDecodeCounter.count += 1;
                  }
                  if (currentEntry._wasTextReleased) {
                    currentEntry._wasTextReleased = false;
                    this._wgslTextLazyRedecodes += 1;
                  }
                  this._putWgslCache(key, wgsl, currentLength);
                  return wgsl;
                }
              }
            }
          }

          if (!reloaded && this.url) {
            reloaded = true;
            await this._reloadStaleIndex(fetcher, observedGeneration);
            observedGeneration = this._indexGeneration;
            const refreshed = this.entries.get(key);
            if (!refreshed) {
              return null;
            }
            currentEntry = refreshed;
            currentLength = refreshed.length >>> 0;
            if (currentLength === 0) {
              this._putWgslCache(key, '', 0);
              return '';
            }
            currentStart = normalizeSafeOffset(refreshed.offset);
            currentEnd = currentStart + currentLength - 1;
            continue;
          }

          throw new Error(
            `WGPK stale pack mismatch (${staleReason}) persisted after index reload for ${currentEntry.hashHex}`
          );
        }
      } finally {
        settled = true;
        if (this._inFlightFetches.get(key) === fetchPromise) {
          this._inFlightFetches.delete(key);
          this._statsVersion += 1;
        }
      }
    })();

    if (!settled) {
      this._inFlightFetches.set(key, fetchPromise);
      this._statsVersion += 1;
    }
    return fetchPromise;
  }

  /**
   * Prefetch a set of shader hashes into the bounded hot-hash WGSL LRU cache by coalescing
   * nearby byte ranges (within `maxGapBytes` and `maxSpanBytes`) into batched HTTP `206 Range`
   * fetches.
   *
   * Already-cached entries (and dynamic overrides) are refreshed/skipped with zero HTTP traffic;
   * hashes already in flight via `getWgslAsync()` or a concurrent `prefetchHashes()` call are
   * deduplicated onto their existing promises.
   *
   * @param {Iterable<bigint | string | number>} hashes
   * @param {{
   *   maxCoalesceGapBytes?: number,
   *   maxGapBytes?: number,
   *   maxCoalesceSpanBytes?: number,
   *   maxSpanBytes?: number,
   *   prefetchMaxGapBytes?: number,
   *   prefetchMaxSpanBytes?: number,
   * }} [options]
   * @returns {Promise<{
   *   requested: number,
   *   prefetched: number,
   *   cached: number,
   *   inFlight: number,
   *   missing: number,
   *   batches: number,
   *   bytesFetched: number,
   * }>}
   */
  async prefetchHashes(hashes, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const maxGapBytes = resolvePrefetchMaxGapBytes(
      opts,
      this._prefetchMaxGapBytes !== null
        ? this._prefetchMaxGapBytes
        : DEFAULT_MAX_COALESCE_GAP_BYTES
    );
    const maxSpanBytes = resolvePrefetchMaxSpanBytes(
      opts,
      this._prefetchMaxSpanBytes !== null
        ? this._prefetchMaxSpanBytes
        : DEFAULT_MAX_COALESCE_SPAN_BYTES
    );

    if (hashes === undefined || hashes === null) {
      return {
        requested: 0,
        prefetched: 0,
        cached: 0,
        inFlight: 0,
        missing: 0,
        batches: 0,
        bytesFetched: 0,
      };
    }
    if (typeof hashes[Symbol.iterator] !== 'function') {
      throw new TypeError('hashes must be an iterable of shader hashes');
    }

    if (this._indexReloadPromise !== null) {
      await this._indexReloadPromise;
    }

    const seen = new Set();
    let requested = 0;
    let cached = 0;
    let inFlight = 0;
    let missing = 0;
    const toFetch = [];
    const inFlightPromises = [];

    for (const rawHash of hashes) {
      const key = this._resolveHashKey(rawHash);
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      requested += 1;

      if (this._dynamicEntries.has(key)) {
        cached += 1;
        continue;
      }

      const staticEntry = this.entries.get(key);
      if (!staticEntry) {
        missing += 1;
        continue;
      }

      if (this.mode !== 'ranged') {
        void staticEntry.wgsl;
        cached += 1;
        continue;
      }

      if (this._touchWgslCache(key) !== undefined) {
        cached += 1;
        continue;
      }

      const length = staticEntry.length >>> 0;
      if (length === 0) {
        this._putWgslCache(key, '', 0);
        cached += 1;
        continue;
      }

      const existingPromise = this._inFlightFetches.get(key);
      if (existingPromise !== undefined) {
        this._inFlightCoalesced += 1;
        this._statsVersion += 1;
        inFlight += 1;
        inFlightPromises.push(existingPromise);
        continue;
      }

      const start = normalizeSafeOffset(staticEntry.offset);
      toFetch.push({
        key,
        entry: staticEntry,
        offset: start,
        start,
        length,
        end: start + length,
      });
    }

    if (toFetch.length === 0) {
      if (inFlightPromises.length > 0) {
        await Promise.all(inFlightPromises);
      }
      return {
        requested,
        prefetched: 0,
        cached,
        inFlight,
        missing,
        batches: 0,
        bytesFetched: 0,
      };
    }

    const fetcher = this._fetch ?? globalThis.fetch;
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for ranged prefetchHashes');
    }

    // Sort by ascending byte offset and coalesce nearby ranges within maxGapBytes & maxSpanBytes.
    const batches = coalesceEntryRanges(toFetch, {
      maxCoalesceGapBytes: maxGapBytes,
      maxCoalesceSpanBytes: maxSpanBytes,
    });

    let prefetched = 0;
    let completedBatches = 0;
    let bytesFetched = 0;

    const generationBeforeFetch = this._indexGeneration;

    const recordCoalescedSpanStats = (span) => {
      if (span.entries.length > 1) {
        this._coalescedRangeFetches += 1;
        this._coalescedEntriesFetched += span.entries.length;
        this._httpRequestsSavedByCoalescing += span.entries.length - 1;
        this._coalescedGapBytes += sanitizeNonNegativeInt(span.gapBytes, 0);
        this._statsVersion += 1;
      }
    };

    // Register per-hash in-flight promises synchronously BEFORE awaiting any batch so concurrent
    // getWgslAsync() or prefetchHashes() calls deduplicate cleanly onto the coalesced batch.
    const batchPromises = batches.map((batch) => {
      const deferreds = batch.entries.map((item) => {
        let resolve;
        let reject;
        let settled = false;
        const promise = new Promise((res, rej) => {
          resolve = (val) => {
            if (!settled) {
              settled = true;
              res(val);
            }
          };
          reject = (err) => {
            if (!settled) {
              settled = true;
              rej(err);
            }
          };
        });
        // Prevent unhandled rejection noise if the batch fails and only prefetchHashes() catches.
        promise.catch(() => {});
        this._inFlightFetches.set(item.key, promise);
        this._statsVersion += 1;
        return {
          item,
          promise,
          resolve,
          reject,
          isSettled: () => settled,
        };
      });

      const clearInFlight = () => {
        let cleared = false;
        for (let i = 0; i < deferreds.length; i += 1) {
          const { item, promise } = deferreds[i];
          if (this._inFlightFetches.get(item.key) === promise) {
            this._inFlightFetches.delete(item.key);
            cleared = true;
          }
        }
        if (cleared) {
          this._statsVersion += 1;
        }
      };

      const retryAfterStaleReload = async () => {
        await this._reloadStaleIndex(fetcher, generationBeforeFetch);
        const decodedResults = new Array(deferreds.length);
        let firstError = null;
        const refreshedItems = [];

        for (let i = 0; i < deferreds.length; i += 1) {
          const { item } = deferreds[i];
          const refreshed = this.entries.get(item.key);
          if (!refreshed) {
            decodedResults[i] = { ok: true, wgsl: null };
            continue;
          }
          const refreshedLen = refreshed.length >>> 0;
          if (refreshedLen === 0) {
            this._putWgslCache(item.key, '', 0);
            prefetched += 1;
            this._prefetchShaders += 1;
            this._statsVersion += 1;
            decodedResults[i] = { ok: true, wgsl: '' };
            continue;
          }
          const rStart = normalizeSafeOffset(refreshed.offset);
          refreshedItems.push({
            deferredIndex: i,
            key: item.key,
            entry: refreshed,
            offset: rStart,
            start: rStart,
            length: refreshedLen,
            end: rStart + refreshedLen,
          });
        }

        const refreshedSpans = coalesceEntryRanges(refreshedItems, {
          maxCoalesceGapBytes: maxGapBytes,
          maxCoalesceSpanBytes: maxSpanBytes,
        });

        await Promise.all(
          refreshedSpans.map(async (rSpan) => {
            let retryChunk;
            try {
              retryChunk = await this._fetchRangeChunkBounded(fetcher, rSpan.start, rSpan.rangeEnd);
            } catch (retryErr) {
              if (firstError === null) {
                firstError = retryErr;
              }
              for (let j = 0; j < rSpan.entries.length; j += 1) {
                decodedResults[rSpan.entries[j].deferredIndex] = { ok: false, err: retryErr };
              }
              return;
            }

            const retryChunkBytes = sanitizeNonNegativeInt(retryChunk.bytes.byteLength, 0);
            this._rangeBytesFetched += retryChunkBytes;
            this._statsVersion += 1;
            bytesFetched += retryChunkBytes;

            const retryStale = this._detectChunkStaleReason(retryChunk);
            if (retryStale !== null) {
              const staleErr = new Error(
                `WGPK stale pack mismatch (${retryStale}) persisted after index reload for ${rSpan.entries[0].entry.hashHex}`
              );
              if (firstError === null) {
                firstError = staleErr;
              }
              for (let j = 0; j < rSpan.entries.length; j += 1) {
                decodedResults[rSpan.entries[j].deferredIndex] = { ok: false, err: staleErr };
              }
              return;
            }

            try {
              if (retryChunk.status === 206) {
                if (retryChunk.contentRange && retryChunk.contentRange.start !== rSpan.start) {
                  throw new Error(
                    `WGPK ranged prefetch mismatch: expected start byte ${rSpan.start}, got ${retryChunk.contentRange.start}`
                  );
                }
                if (retryChunk.bytes.byteLength < rSpan.spanBytes) {
                  throw new Error(
                    `WGPK ranged prefetch truncated: expected ${rSpan.spanBytes} bytes for [${rSpan.start}..${rSpan.end}), received ${retryChunk.bytes.byteLength}`
                  );
                }
              } else if (retryChunk.bytes.byteLength < rSpan.end) {
                throw new Error(
                  `WGPK full prefetch response truncated: need byte range [${rSpan.start}..${rSpan.end}), received ${retryChunk.bytes.byteLength} bytes`
                );
              }
            } catch (boundsErr) {
              if (firstError === null) {
                firstError = boundsErr;
              }
              for (let j = 0; j < rSpan.entries.length; j += 1) {
                decodedResults[rSpan.entries[j].deferredIndex] = { ok: false, err: boundsErr };
              }
              return;
            }

            for (let j = 0; j < rSpan.entries.length; j += 1) {
              const rItem = rSpan.entries[j];
              const relStart = retryChunk.status === 206 ? rItem.start - rSpan.start : rItem.start;
              const payloadBytes = retryChunk.bytes.subarray(relStart, relStart + rItem.length);
              const entryIdx =
                rItem.entry._index !== undefined ? rItem.entry._index : rItem.entry.hashHex;
              try {
                const wgsl = decodeUtf8Strict(payloadBytes, entryIdx);
                if (this._rangedDecodeCounter) {
                  this._rangedDecodeCounter.count += 1;
                }
                if (rItem.entry._wasTextReleased) {
                  rItem.entry._wasTextReleased = false;
                  this._wgslTextLazyRedecodes += 1;
                }
                this._putWgslCache(rItem.key, wgsl, rItem.length);
                prefetched += 1;
                this._prefetchShaders += 1;
                this._statsVersion += 1;
                decodedResults[rItem.deferredIndex] = { ok: true, wgsl };
              } catch (decodeErr) {
                if (firstError === null) {
                  firstError = decodeErr;
                }
                decodedResults[rItem.deferredIndex] = { ok: false, err: decodeErr };
              }
            }

            recordCoalescedSpanStats(rSpan);
            this._prefetchBatches += 1;
            this._statsVersion += 1;
            completedBatches += 1;
          })
        );

        clearInFlight();

        for (let i = 0; i < deferreds.length; i += 1) {
          const res = decodedResults[i];
          if (res && res.ok) {
            deferreds[i].resolve(res.wgsl);
          } else {
            deferreds[i].reject(res ? res.err : firstError);
          }
        }

        if (firstError !== null) {
          throw firstError;
        }
      };

      const runStaleReload = async () => {
        try {
          await retryAfterStaleReload();
        } catch (reloadErr) {
          clearInFlight();
          for (let i = 0; i < deferreds.length; i += 1) {
            deferreds[i].reject(reloadErr);
          }
          throw reloadErr;
        }
      };

      return (async () => {
        try {
          let chunk;
          try {
            chunk = await this._fetchRangeChunkBounded(fetcher, batch.start, batch.rangeEnd);
          } catch (err) {
            if (
              this.url &&
              (this._detectErrorStaleReason(err) !== null ||
                this._indexGeneration !== generationBeforeFetch)
            ) {
              await runStaleReload();
              return;
            }
            clearInFlight();
            for (let i = 0; i < deferreds.length; i += 1) {
              deferreds[i].reject(err);
            }
            throw err;
          }

          const chunkBytes = sanitizeNonNegativeInt(chunk.bytes.byteLength, 0);
          this._rangeBytesFetched += chunkBytes;
          this._statsVersion += 1;
          bytesFetched += chunkBytes;

          if (
            this.url &&
            (this._detectChunkStaleReason(chunk) !== null ||
              this._indexGeneration !== generationBeforeFetch)
          ) {
            await runStaleReload();
            return;
          }

          const expectedSpan = batch.spanBytes;
          let truncatedStale = false;
          try {
            if (chunk.status === 206) {
              if (chunk.contentRange && chunk.contentRange.start !== batch.start) {
                throw new Error(
                  `WGPK ranged prefetch mismatch: expected start byte ${batch.start}, got ${chunk.contentRange.start}`
                );
              }
              if (chunk.bytes.byteLength < expectedSpan) {
                if (
                  this.url &&
                  (this.etag !== null || this.totalSize !== null || this.lastModified !== null)
                ) {
                  truncatedStale = true;
                } else {
                  throw new Error(
                    `WGPK ranged prefetch truncated: expected ${expectedSpan} bytes for [${batch.start}..${batch.end}), received ${chunk.bytes.byteLength}`
                  );
                }
              }
            } else if (chunk.bytes.byteLength < batch.end) {
              throw new Error(
                `WGPK full prefetch response truncated: need byte range [${batch.start}..${batch.end}), received ${chunk.bytes.byteLength} bytes`
              );
            }
          } catch (err) {
            clearInFlight();
            for (let i = 0; i < deferreds.length; i += 1) {
              deferreds[i].reject(err);
            }
            throw err;
          }

          if (truncatedStale) {
            await runStaleReload();
            return;
          }

          const decodedResults = new Array(deferreds.length);
          let firstDecodeError = null;
          let detectedMidShaderCorruption = false;
          for (let i = 0; i < deferreds.length; i += 1) {
            const { item } = deferreds[i];
            const relStart = chunk.status === 206 ? item.start - batch.start : item.start;
            const payloadBytes = chunk.bytes.subarray(relStart, relStart + item.length);
            const entryIdx =
              item.entry._index !== undefined ? item.entry._index : item.entry.hashHex;
            try {
              const wgsl = decodeUtf8Strict(payloadBytes, entryIdx);
              if (this.url && looksLikeMidShaderCorruption(wgsl)) {
                detectedMidShaderCorruption = true;
                break;
              }
              decodedResults[i] = { ok: true, wgsl };
            } catch (decodeErr) {
              if (firstDecodeError === null) {
                firstDecodeError = decodeErr;
              }
              decodedResults[i] = { ok: false, err: decodeErr };
            }
          }

          if (detectedMidShaderCorruption) {
            await runStaleReload();
            return;
          }

          for (let i = 0; i < deferreds.length; i += 1) {
            const res = decodedResults[i];
            if (res.ok) {
              const { item } = deferreds[i];
              if (this._rangedDecodeCounter) {
                this._rangedDecodeCounter.count += 1;
              }
              if (item.entry._wasTextReleased) {
                item.entry._wasTextReleased = false;
                this._wgslTextLazyRedecodes += 1;
              }
              this._putWgslCache(item.key, res.wgsl, item.length);
              prefetched += 1;
              this._prefetchShaders += 1;
              this._statsVersion += 1;
            }
          }

          recordCoalescedSpanStats(batch);
          this._prefetchBatches += 1;
          this._statsVersion += 1;
          completedBatches += 1;
          clearInFlight();

          for (let i = 0; i < deferreds.length; i += 1) {
            const res = decodedResults[i];
            if (res.ok) {
              deferreds[i].resolve(res.wgsl);
            } else {
              deferreds[i].reject(res.err);
            }
          }

          if (firstDecodeError !== null) {
            throw firstDecodeError;
          }
        } catch (unexpectedErr) {
          clearInFlight();
          for (let i = 0; i < deferreds.length; i += 1) {
            if (!deferreds[i].isSettled()) {
              deferreds[i].reject(unexpectedErr);
            }
          }
          throw unexpectedErr;
        } finally {
          clearInFlight();
          for (let i = 0; i < deferreds.length; i += 1) {
            if (!deferreds[i].isSettled()) {
              deferreds[i].reject(new Error('WGPK prefetch batch exited without settling shader promise'));
            }
          }
        }
      })();
    });

    await Promise.all([...batchPromises, ...inFlightPromises]);

    return {
      requested,
      prefetched,
      cached,
      inFlight,
      missing,
      batches: completedBatches,
      bytesFetched,
    };
  }

  /**
   * Check whether a shader hash exists in dynamic overrides or this static pack
   * without triggering lazy UTF-8 decode.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  has(hash) {
    if (this._dynamicEntries.size > 0) {
      const key = normalizeHash(hash);
      if (this._dynamicEntries.has(key)) {
        return true;
      }
      return this.entries.has(key);
    }
    if (typeof hash === 'string') {
      if (this._byHex.has(hash)) {
        return true;
      }
    }
    const key = normalizeHash(hash);
    return this.entries.has(key);
  }

  /**
   * Return all static shader entries in their original pack table order.
   * Does not trigger lazy UTF-8 decoding until `.wgsl` is accessed on an entry.
   */
  listEntries() {
    return this._entryList.slice();
  }

  /**
   * Return metadata summaries for all dynamically registered shaders on this pack.
   *
   * @param {{ stage?: number | string, includeSource?: boolean }} [options]
   */
  listDynamicShaders(options = {}) {
    const expectedStage =
      options && options.stage !== undefined && options.stage !== null
        ? resolveStageCode(options.stage)
        : null;
    const includeSource = Boolean(options && options.includeSource);

    const results = [];
    for (const entry of this._dynamicEntries.values()) {
      if (expectedStage !== null && entry.stageCode !== expectedStage) {
        continue;
      }
      const item = {
        hash: entry.hash,
        hashHex: entry.hashHex,
        stage: entry.stage,
        stageCode: entry.stageCode,
        length: entry.length,
        dynamic: true,
        overridesStatic: this.entries.has(entry.hash),
        source: entry.source,
        label: entry.label,
        entryPoint: entry.entryPoint,
        tags: entry.tags.slice(),
        revision: entry.revision,
        registeredAt: entry.registeredAt,
      };
      if (includeSource) {
        item.wgsl = entry.wgsl;
      }
      results.push(item);
    }
    return results;
  }

  /**
   * Return structural, decode, and dynamic overlay statistics for this pack.
   */
  stats() {
    const decodedCount = this._getDecodedCount();
    if (
      this._cachedStats !== null &&
      this._cachedStatsVersion === this._statsVersion &&
      this._cachedStatsDecodedCount === decodedCount
    ) {
      const c = this._cachedStats;
      return {
        ...c,
        stageCounts: { ...c.stageCounts },
        dynamicStageCounts: { ...c.dynamicStageCounts },
      };
    }

    const entryCount = this._entryList.length;
    let dynamicOverrideCount = 0;
    let dynamicPayloadBytes = 0;
    const dynamicStageCounts = { vertex: 0, fragment: 0, compute: 0 };

    for (const entry of this._dynamicEntries.values()) {
      if (this.entries.has(entry.hash)) {
        dynamicOverrideCount += 1;
      }
      dynamicPayloadBytes += entry.length;
      dynamicStageCounts[entry.stage] += 1;
    }

    const rangeFetchCount = sanitizeNonNegativeInt(this._rangeFetches, 0);
    const rangeBytesFetched = sanitizeNonNegativeInt(this._rangeBytesFetched, 0);
    const rangeFetchRetries = sanitizeNonNegativeInt(this._rangeFetchRetries, 0);
    const staleIndexReloads = sanitizeNonNegativeInt(this._staleIndexReloads, 0);
    const activeRangeFetches = sanitizeNonNegativeInt(this._activeRangeFetches, 0);
    const peakConcurrentRangeFetches = sanitizeNonNegativeInt(this._peakConcurrentRangeFetches, 0);
    const cacheHits = sanitizeNonNegativeInt(this._cacheHits, 0);
    const cacheMisses = sanitizeNonNegativeInt(this._cacheMisses, 0);
    const lruHits = sanitizeNonNegativeInt(this._lruHits, 0);
    const lruMisses = sanitizeNonNegativeInt(this._lruMisses, 0);
    const cachedWgslBytes = sanitizeNonNegativeInt(this._cachedWgslBytes, 0);
    const lruBytes = cachedWgslBytes;
    const pinnedBytes = sanitizeNonNegativeInt(this._pinnedWgslBytes, 0);
    const pinnedCount = sanitizeNonNegativeInt(this._pinnedHashes.size, 0);
    const pinnedCachedEntries = sanitizeNonNegativeInt(this._pinnedWgslCache.size, 0);
    const cachedWgslEntries = sanitizeNonNegativeInt(
      this._pinnedWgslCache.size + this._wgslCache.size,
      0
    );
    const lruEvictions = sanitizeNonNegativeInt(this._wgslCacheEvictions, 0);
    const prefetchBatches = sanitizeNonNegativeInt(this._prefetchBatches, 0);
    const prefetchShaders = sanitizeNonNegativeInt(this._prefetchShaders, 0);
    const coalescedRangeFetches = sanitizeNonNegativeInt(this._coalescedRangeFetches, 0);
    const coalescedEntriesFetched = sanitizeNonNegativeInt(this._coalescedEntriesFetched, 0);
    const httpRequestsSavedByCoalescing = sanitizeNonNegativeInt(
      this._httpRequestsSavedByCoalescing,
      0
    );
    const coalescedGapBytes = sanitizeNonNegativeInt(this._coalescedGapBytes, 0);
    const inFlightCoalesced = sanitizeNonNegativeInt(this._inFlightCoalesced, 0);
    const inFlightCount = sanitizeNonNegativeInt(this._inFlightFetches.size, 0);
    const wgslTextBytesRetained = sanitizeNonNegativeInt(
      this.mode === 'ranged'
        ? cachedWgslBytes
        : this._bufferTextState
          ? this._bufferTextState.retainedBytes
          : 0,
      0
    );
    const wgslTextEntriesRetained = sanitizeNonNegativeInt(
      this.mode === 'ranged'
        ? cachedWgslEntries
        : this._bufferTextState
          ? this._bufferTextState.retainedEntries
          : 0,
      0
    );
    const wgslTextBytesReleased = sanitizeNonNegativeInt(
      this.mode === 'ranged'
        ? this._wgslTextBytesReleased
        : this._bufferTextState
          ? this._bufferTextState.bytesReleased
          : 0,
      0
    );
    const wgslTextReleases = sanitizeNonNegativeInt(
      this.mode === 'ranged'
        ? this._wgslTextReleases
        : this._bufferTextState
          ? this._bufferTextState.releases
          : 0,
      0
    );
    const wgslTextLazyRedecodes = sanitizeNonNegativeInt(
      this.mode === 'ranged'
        ? this._wgslTextLazyRedecodes
        : this._bufferTextState
          ? this._bufferTextState.lazyRedecodes
          : 0,
      0
    );

    const computed = {
      version: this.version,
      entryBytes: this.entryBytes,
      flags: this.flags,
      mode: this.mode,
      etag: this.etag,
      lastModified: this.lastModified,
      totalSize: this.totalSize,
      entryCount,
      byteLength: this.byteLength,
      headerBytes: WGPK_HEADER_BYTES,
      tableBytes: entryCount * this.entryBytes,
      payloadBytes: this.payloadBytes,
      decodedCount,
      lazyDecode: this.lazyDecode,
      wgslTextBytesRetained,
      wgslTextEntriesRetained,
      wgslTextBytesReleased,
      wgslTextReleases,
      wgslTextLazyRedecodes,
      stageCounts: { ...this._stageCounts },
      dynamicCount: this._dynamicEntries.size,
      dynamicOverrideCount,
      dynamicPayloadBytes,
      dynamicHits: this._dynamicHits,
      dynamicStageCounts,
      rangeFetches: rangeFetchCount,
      rangeFetchCount,
      rangeBytesFetched,
      rangeFetchRetries,
      staleIndexReloads,
      activeRangeFetches,
      peakConcurrentRangeFetches,
      maxConcurrentRangeFetches: this._maxConcurrentRangeFetches,
      maxRangeFetchRetries: this._maxRangeFetchRetries,
      cacheHits,
      cacheMisses,
      cacheEvictions: lruEvictions,
      cacheEntries: cachedWgslEntries,
      cacheBytes: cachedWgslBytes,
      pinnedCount,
      pinnedBytes,
      pinnedCachedEntries,
      lruHits,
      lruMisses,
      lruBytes,
      cachedWgslBytes,
      cachedWgslEntries,
      maxCacheBytes: this._maxCachedWgslBytes,
      maxCachedWgslBytes: this._maxCachedWgslBytes,
      maxCacheEntries: this._maxCacheEntries,
      wgslCacheEvictions: lruEvictions,
      lruEvictions,
      prefetchBatches,
      prefetchCoalescedRanges: prefetchBatches,
      prefetchShaders,
      coalescedRangeFetches,
      coalescedEntriesFetched,
      httpRequestsSavedByCoalescing,
      coalescedGapBytes,
      prefetchMaxGapBytes: this._prefetchMaxGapBytes,
      prefetchMaxSpanBytes: this._prefetchMaxSpanBytes,
      maxCoalesceGapBytes: this._prefetchMaxGapBytes,
      maxCoalesceSpanBytes: this._prefetchMaxSpanBytes,
      inFlightDeduplications: inFlightCoalesced,
      inFlightCoalesced,
      inFlightCount,
    };
    this._cachedStats = computed;
    this._cachedStatsVersion = this._statsVersion;
    this._cachedStatsDecodedCount = decodedCount;
    return {
      ...computed,
      stageCounts: { ...computed.stageCounts },
      dynamicStageCounts: { ...computed.dynamicStageCounts },
    };
  }

  /**
   * Alias for `stats()`.
   */
  getStats() {
    return this.stats();
  }
}

/**
 * Parse and validate a binary `.wgsl.pack` (`WGPK`) buffer.
 *
 * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
 * @param {{ lazyDecode?: boolean }} [options]
 * @returns {WgslPack}
 */
export function parseWgslPack(buffer, options = {}) {
  const bytes = toBinaryView(buffer);
  const lazyDecode = Boolean(options && options.lazyDecode);

  const header = parseWgpkHeader(bytes);
  const { rawEntries, stageCounts, totalPayloadBytes } = parseWgpkEntryTable(
    bytes,
    header,
    bytes.byteLength
  );

  const entryList = new Array(header.entryCount);
  const byHash = new Map();
  const byHex = new Map();
  let decodedCount = 0;

  // Compact buffer-mode decoded text tracker: retains zero-copy Uint8Array slices per entry
  // so decoded JS UTF-16 WGSL strings can be released post-compile and lazily re-decoded on demand (#2162).
  const releasers = new Map();
  const cachedCheckers = new Map();
  const bufferTextState = {
    retainedBytes: 0,
    retainedEntries: 0,
    bytesReleased: 0,
    releases: 0,
    lazyRedecodes: 0,
    onMutate: null,
    isCached(hashKey) {
      const check = cachedCheckers.get(hashKey);
      return check ? check() : false;
    },
    release(hashKey) {
      const fn = releasers.get(hashKey);
      return fn ? fn() : -1;
    },
    releaseAll() {
      let releasedCount = 0;
      let bytesReleased = 0;
      for (const fn of releasers.values()) {
        const bytesFreed = fn();
        if (bytesFreed >= 0) {
          releasedCount += 1;
          bytesReleased += bytesFreed;
        }
      }
      return { releasedCount, bytesReleased };
    },
  };

  for (let idx = 0; idx < rawEntries.length; idx += 1) {
    const raw = rawEntries[idx];
    const slice = bytes.subarray(raw.offset, raw.end);
    const entryByteLength = raw.length;

    let cachedWgsl = lazyDecode ? null : decodeUtf8Strict(slice, idx);
    let hasDecoded = !lazyDecode;
    let wasReleased = false;
    if (!lazyDecode) {
      decodedCount += 1;
      bufferTextState.retainedBytes += entryByteLength;
      bufferTextState.retainedEntries += 1;
    }

    const entry = {
      hash: raw.hash,
      hashHex: raw.hashHex,
      stage: raw.stage,
      stageCode: raw.stageCode,
      offset: raw.offset,
      length: entryByteLength,
    };
    Object.defineProperty(entry, 'wgsl', {
      enumerable: true,
      configurable: true,
      get() {
        if (cachedWgsl === null) {
          cachedWgsl = decodeUtf8Strict(slice, idx);
          bufferTextState.retainedBytes += entryByteLength;
          bufferTextState.retainedEntries += 1;
          if (!hasDecoded) {
            hasDecoded = true;
            decodedCount += 1;
          } else if (wasReleased) {
            wasReleased = false;
            bufferTextState.lazyRedecodes += 1;
          }
          if (typeof bufferTextState.onMutate === 'function') {
            bufferTextState.onMutate();
          }
        }
        return cachedWgsl;
      },
    });

    cachedCheckers.set(raw.hash, () => cachedWgsl !== null);
    releasers.set(raw.hash, () => {
      if (cachedWgsl === null) {
        return -1;
      }
      cachedWgsl = null;
      wasReleased = true;
      bufferTextState.retainedBytes = Math.max(0, bufferTextState.retainedBytes - entryByteLength);
      bufferTextState.retainedEntries = Math.max(0, bufferTextState.retainedEntries - 1);
      bufferTextState.bytesReleased += entryByteLength;
      bufferTextState.releases += 1;
      return entryByteLength;
    });

    entryList[idx] = entry;
    byHash.set(raw.hash, entry);
    byHex.set(raw.hashHex, entry);
  }

  return new WgslPack({
    version: header.version,
    entryBytes: header.entryBytes,
    flags: header.flags,
    byteLength: bytes.byteLength,
    dataStart: header.dataStart,
    payloadBytes: totalPayloadBytes,
    lazyDecode,
    entryList,
    byHash,
    byHex,
    stageCounts,
    getDecodedCount: () => decodedCount,
    mode: 'buffer',
    bufferTextState,
  });
}

function populateRangedIndexEntries(rawEntries, wgslCache, pinnedWgslCache, byHash, byHex) {
  const entryList = new Array(rawEntries.length);
  byHash.clear();
  byHex.clear();

  for (let idx = 0; idx < rawEntries.length; idx += 1) {
    const raw = rawEntries[idx];
    const entry = {
      hash: raw.hash,
      hashHex: raw.hashHex,
      stage: raw.stage,
      stageCode: raw.stageCode,
      offset: raw.offset,
      length: raw.length,
    };
    Object.defineProperty(entry, '_index', {
      value: idx,
      enumerable: false,
      configurable: true,
    });
    Object.defineProperty(entry, 'wgsl', {
      enumerable: true,
      configurable: true,
      get() {
        const cached = pinnedWgslCache.get(raw.hash) ?? wgslCache.get(raw.hash);
        if (cached !== undefined) {
          return cached.wgsl;
        }
        if (raw.length === 0) {
          return '';
        }
        throw new Error(
          `WGPK shader ${raw.hashHex} is not loaded locally in ranged mode; call getWgslAsync(hash) first`
        );
      },
    });

    entryList[idx] = entry;
    byHash.set(raw.hash, entry);
    byHex.set(raw.hashHex, entry);
  }

  return entryList;
}

/**
 * Parse only the WGPK header + entry table into a ranged-mode `WgslPack` without
 * holding payload bytes in memory. Payloads are fetched on demand via `getWgslAsync(hash)`.
 *
 * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} indexBuffer
 * @param {{
 *   url?: string | null,
 *   totalBytes?: number | null,
 *   totalSize?: number | null,
 *   etag?: string | null,
 *   lastModified?: string | null,
 *   initialRangeBytes?: number,
 *   fetch?: typeof fetch,
 *   fetchInit?: RequestInit,
 *   maxCachedWgslBytes?: number,
 *   initialRangeFetches?: number,
 * }} [options]
 * @returns {WgslPack}
 */
export function parseWgslPackIndex(indexBuffer, options = {}) {
  const bytes = toBinaryView(indexBuffer);
  const rawTotal =
    options && options.totalSize !== undefined && options.totalSize !== null
      ? options.totalSize
      : options && options.totalBytes !== undefined && options.totalBytes !== null
        ? options.totalBytes
        : null;
  if (rawTotal !== null && (!Number.isSafeInteger(rawTotal) || rawTotal < 0)) {
    throw new RangeError(`totalBytes must be a non-negative safe integer, got: ${rawTotal}`);
  }
  const totalBytes = rawTotal;

  const header = parseWgpkHeader(bytes);
  const { rawEntries, stageCounts, totalPayloadBytes, maxEnd } = parseWgpkEntryTable(
    bytes,
    header,
    totalBytes
  );

  const wgslCache = new Map();
  const pinnedWgslCache = new Map();
  const rangedDecodeCounter = { count: 0 };
  const byHash = new Map();
  const byHex = new Map();
  const entryList = populateRangedIndexEntries(
    rawEntries,
    wgslCache,
    pinnedWgslCache,
    byHash,
    byHex
  );

  const maxCachedWgslBytes = resolveMaxCacheBytes(options);
  const maxCacheEntries = validateMaxCacheEntries(
    options ? options.maxCacheEntries : undefined,
    null
  );
  const maxConcurrentRangeFetches = validateMaxConcurrentRangeFetches(
    options ? options.maxConcurrentRangeFetches : undefined
  );
  const maxRangeFetchRetries = validateMaxRangeFetchRetries(
    options ? options.maxRangeFetchRetries : undefined
  );
  const rangeRetryDelayMs = validateRangeRetryDelayMs(
    options ? options.rangeRetryDelayMs : undefined
  );
  const sleepFn = validateSleepFn(options ? options.sleepFn : undefined);
  const prefetchMaxGapBytes = resolvePrefetchMaxGapBytes(options);
  const prefetchMaxSpanBytes = resolvePrefetchMaxSpanBytes(options);
  const initialRangeBytes = validateInitialRangeBytes(
    options ? options.initialRangeBytes : undefined
  );
  const maxFullFallbackBytes = validateMaxFullFallbackBytes(
    options ? options.maxFullFallbackBytes : undefined
  );
  const initialRangeFetches = sanitizeNonNegativeInt(
    options ? options.initialRangeFetches : 0,
    0
  );
  const initialRangeBytesFetched = sanitizeNonNegativeInt(
    options ? options.initialRangeBytesFetched : 0,
    0
  );
  const initialRangeFetchRetries = sanitizeNonNegativeInt(
    options ? options.initialRangeFetchRetries : 0,
    0
  );
  const etag = normalizeHeaderString(options ? options.etag : null);
  const lastModified = normalizeHeaderString(options ? options.lastModified : null);

  return new WgslPack({
    version: header.version,
    entryBytes: header.entryBytes,
    flags: header.flags,
    byteLength: totalBytes !== null ? totalBytes : Math.max(header.dataStart, maxEnd),
    dataStart: header.dataStart,
    payloadBytes: totalPayloadBytes,
    lazyDecode: true,
    entryList,
    byHash,
    byHex,
    stageCounts,
    getDecodedCount: () => rangedDecodeCounter.count,
    mode: 'ranged',
    url: options && options.url ? String(options.url) : null,
    fetchImpl: options && typeof options.fetch === 'function' ? options.fetch : null,
    fetchInit: options ? options.fetchInit : undefined,
    maxCachedWgslBytes,
    maxCacheEntries,
    maxConcurrentRangeFetches,
    maxRangeFetchRetries,
    rangeRetryDelayMs,
    sleepFn,
    prefetchMaxGapBytes,
    prefetchMaxSpanBytes,
    initialRangeBytes,
    maxFullFallbackBytes,
    initialRangeFetches,
    initialRangeBytesFetched,
    initialRangeFetchRetries,
    etag,
    lastModified,
    totalSize: totalBytes,
    wgslCache,
    pinnedWgslCache,
    rangedDecodeCounter,
  });
}

async function finishPackIndexFromInitialChunk(url, fetcher, initialChunk, options = {}) {
  const lazyDecode = options && options.lazyDecode !== undefined ? Boolean(options.lazyDecode) : true;
  const maxCachedWgslBytes = resolveMaxCacheBytes(options);
  const maxCacheEntries = validateMaxCacheEntries(
    options ? options.maxCacheEntries : undefined,
    null
  );
  const maxConcurrentRangeFetches = validateMaxConcurrentRangeFetches(
    options ? options.maxConcurrentRangeFetches : undefined
  );
  const maxRangeFetchRetries = validateMaxRangeFetchRetries(
    options ? options.maxRangeFetchRetries : undefined
  );
  const rangeRetryDelayMs = validateRangeRetryDelayMs(
    options ? options.rangeRetryDelayMs : undefined
  );
  const sleepFn = validateSleepFn(options ? options.sleepFn : undefined);
  const prefetchMaxGapBytes = resolvePrefetchMaxGapBytes(options);
  const prefetchMaxSpanBytes = resolvePrefetchMaxSpanBytes(options);
  const initialRangeBytes = validateInitialRangeBytes(
    options ? options.initialRangeBytes : undefined
  );
  const maxFullFallbackBytes = validateMaxFullFallbackBytes(
    options ? options.maxFullFallbackBytes : undefined
  );

  // If the server returned 200 OK instead of 206 Partial Content, fall back seamlessly to buffer mode.
  if (initialChunk.status !== 206) {
    return parseWgslPack(initialChunk.bytes, { lazyDecode });
  }

  if (initialChunk.contentRange && initialChunk.contentRange.start !== 0) {
    throw new Error(
      `WGPK initial range response from "${url}" started at byte ${initialChunk.contentRange.start} (expected 0)`
    );
  }

  let totalBytes =
    initialChunk.totalSize !== undefined && initialChunk.totalSize !== null
      ? initialChunk.totalSize
      : initialChunk.contentRange
        ? initialChunk.contentRange.total
        : null;
  let etag = normalizeHeaderString(initialChunk.etag);
  let lastModified = normalizeHeaderString(initialChunk.lastModified);
  let rangeFetches = sanitizeNonNegativeInt(
    options && options.initialRangeFetches !== undefined ? options.initialRangeFetches : 1,
    1
  );
  let rangeFetchRetries = sanitizeNonNegativeInt(
    options && options.initialRangeFetchRetries !== undefined
      ? options.initialRangeFetchRetries
      : 0,
    0
  );
  let rangeBytesFetched = sanitizeNonNegativeInt(initialChunk.bytes.byteLength, 0);

  const header = parseWgpkHeader(initialChunk.bytes);
  const tableEnd = header.dataStart;

  if (totalBytes !== null && totalBytes < tableEnd) {
    throw new Error(
      `WGPK pack truncated: ${totalBytes} total bytes (header + ${header.entryCount} entry table records require at least ${tableEnd} bytes)`
    );
  }

  let indexBytes;
  if (initialChunk.bytes.byteLength >= tableEnd) {
    // Entire header + entry table fit in the initial range chunk (single roundtrip!).
    // Copy only the exact index prefix so any trailing bytes are not retained in memory.
    indexBytes = initialChunk.bytes.slice(0, tableEnd);
  } else {
    // Table extends beyond the initial range chunk: fetch only the remaining table bytes.
    indexBytes = new Uint8Array(tableEnd);
    indexBytes.set(initialChunk.bytes, 0);
    let have = initialChunk.bytes.byteLength;

    while (have < tableEnd) {
      const remainder = await fetchRangeChunkWithRetry(
        fetcher,
        url,
        have,
        tableEnd - 1,
        options.fetchInit,
        {
          maxRangeFetchRetries,
          rangeRetryDelayMs,
          sleepFn,
          maxFullFallbackBytes,
          requirePartialContent: Boolean(
            totalBytes !== null && maxFullFallbackBytes > 0 && totalBytes > maxFullFallbackBytes
          ),
          onAttempt: () => {
            rangeFetches += 1;
          },
          onRetry: () => {
            rangeFetchRetries += 1;
          },
        }
      );

      // Server fell back to 200 OK on the continuation request: switch seamlessly to full buffer mode.
      if (remainder.status !== 206) {
        return parseWgslPack(remainder.bytes, { lazyDecode });
      }

      rangeBytesFetched += sanitizeNonNegativeInt(remainder.bytes.byteLength, 0);

      if (etag !== null && remainder.etag !== null && remainder.etag !== etag) {
        throw new Error(
          `WGPK pack ETag changed while fetching entry table from "${url}": expected ${etag}, got ${remainder.etag}`
        );
      }
      if (etag === null && remainder.etag !== null) {
        etag = remainder.etag;
      }
      if (
        lastModified !== null &&
        remainder.lastModified !== null &&
        remainder.lastModified !== lastModified
      ) {
        throw new Error(
          `WGPK pack Last-Modified changed while fetching entry table from "${url}": expected ${lastModified}, got ${remainder.lastModified}`
        );
      }
      if (lastModified === null && remainder.lastModified !== null) {
        lastModified = remainder.lastModified;
      }

      if (remainder.contentRange) {
        if (remainder.contentRange.start !== have) {
          throw new Error(
            `WGPK table continuation range from "${url}" started at byte ${remainder.contentRange.start} (expected ${have})`
          );
        }
        if (remainder.contentRange.total !== null) {
          if (totalBytes !== null && remainder.contentRange.total !== totalBytes) {
            throw new Error(
              `WGPK pack total size changed while fetching entry table from "${url}": expected ${totalBytes}, got ${remainder.contentRange.total}`
            );
          }
          totalBytes = remainder.contentRange.total;
        }
      }

      if (remainder.bytes.byteLength === 0) {
        throw new Error(
          `WGPK pack truncated while fetching entry table from "${url}": received 0 bytes at offset ${have}/${tableEnd}`
        );
      }

      const copyLen = Math.min(remainder.bytes.byteLength, tableEnd - have);
      indexBytes.set(remainder.bytes.subarray(0, copyLen), have);
      have += copyLen;
    }
  }

  return parseWgslPackIndex(indexBytes, {
    url,
    totalBytes,
    totalSize: totalBytes,
    etag,
    lastModified,
    initialRangeBytes,
    maxFullFallbackBytes,
    fetch: fetcher,
    fetchInit: options.fetchInit,
    maxCachedWgslBytes,
    maxCacheEntries,
    maxConcurrentRangeFetches,
    maxRangeFetchRetries,
    rangeRetryDelayMs,
    sleepFn,
    prefetchMaxGapBytes,
    prefetchMaxSpanBytes,
    initialRangeFetches: rangeFetches,
    initialRangeBytesFetched: rangeBytesFetched,
    initialRangeFetchRetries: rangeFetchRetries,
  });
}

/**
 * Fetch only the WGPK header + entry table from a URL using HTTP `206 Range` requests.
 * Falls back seamlessly to full in-memory buffer mode if the server responds `200 OK`.
 *
 * @param {string} url
 * @param {typeof fetch | {
 *   fetch?: typeof fetch,
 *   fetchInit?: RequestInit,
 *   initialRangeBytes?: number,
 *   maxCacheBytes?: number,
 *   maxCachedWgslBytes?: number,
 *   maxCacheEntries?: number,
 *   maxConcurrentRangeFetches?: number,
 *   maxRangeFetchRetries?: number,
 *   rangeRetryDelayMs?: number,
 *   sleepFn?: (ms: number) => Promise<void>,
 *   maxCoalesceGapBytes?: number,
 *   prefetchMaxGapBytes?: number,
 *   maxCoalesceSpanBytes?: number,
 *   prefetchMaxSpanBytes?: number,
 *   lazyDecode?: boolean,
 * }} [fetchImplOrOptions]
 * @param {object} [maybeOptions]
 * @returns {Promise<WgslPack>}
 */
export async function loadIndexFromUrl(url, fetchImplOrOptions, maybeOptions) {
  if (typeof url !== 'string' || url.length === 0) {
    throw new TypeError('url must be a non-empty string');
  }
  const { fetcher, options } = resolveFetchAndOptions(fetchImplOrOptions, maybeOptions, null);
  if (typeof fetcher !== 'function') {
    throw new Error('No fetch implementation available for loadIndexFromUrl');
  }

  const initialRangeBytes = validateInitialRangeBytes(options.initialRangeBytes);
  const maxFullFallbackBytes = validateMaxFullFallbackBytes(options.maxFullFallbackBytes);
  resolveMaxCacheBytes(options);
  validateMaxCacheEntries(options.maxCacheEntries, null);
  validateMaxConcurrentRangeFetches(options.maxConcurrentRangeFetches);
  const maxRangeFetchRetries = validateMaxRangeFetchRetries(options.maxRangeFetchRetries);
  const rangeRetryDelayMs = validateRangeRetryDelayMs(options.rangeRetryDelayMs);
  const sleepFn = validateSleepFn(options.sleepFn);
  resolvePrefetchMaxGapBytes(options);
  resolvePrefetchMaxSpanBytes(options);

  let initialRangeFetches = 0;
  let initialRangeFetchRetries = 0;
  const initialChunk = await fetchRangeChunkWithRetry(
    fetcher,
    url,
    0,
    initialRangeBytes - 1,
    options.fetchInit,
    {
      maxRangeFetchRetries,
      rangeRetryDelayMs,
      sleepFn,
      maxFullFallbackBytes,
      requirePartialContent: Boolean(options.requirePartialContent === true),
      onAttempt: () => {
        initialRangeFetches += 1;
      },
      onRetry: () => {
        initialRangeFetchRetries += 1;
      },
    }
  );
  return finishPackIndexFromInitialChunk(url, fetcher, initialChunk, {
    ...options,
    initialRangeFetches,
    initialRangeFetchRetries,
  });
}

/**
 * Fetch a `.wgsl.pack` from a URL into a `WgslPack`.
 * When `options.streamThresholdBytes > 0`, probes the remote pack size (via `HEAD` with
 * Range-probe fallback) and automatically switches to ranged index-only mode if
 * `totalSize > options.streamThresholdBytes`. When `streamThresholdBytes` is omitted or `0`,
 * fetches the full buffer via a single `GET` request.
 *
 * @param {string} url
 * @param {typeof fetch | {
 *   fetch?: typeof fetch,
 *   fetchInit?: RequestInit,
 *   lazyDecode?: boolean,
 *   streamThresholdBytes?: number,
 *   initialRangeBytes?: number,
 *   maxCachedWgslBytes?: number,
 * }} [fetchImplOrOptions]
 * @param {object} [maybeOptions]
 * @returns {Promise<WgslPack>}
 */
export async function loadFromUrl(url, fetchImplOrOptions, maybeOptions) {
  if (typeof url !== 'string' || url.length === 0) {
    throw new TypeError('url must be a non-empty string');
  }
  const { fetcher, options } = resolveFetchAndOptions(fetchImplOrOptions, maybeOptions, null);
  if (typeof fetcher !== 'function') {
    throw new Error('No fetch implementation available for loadFromUrl');
  }

  const lazyDecode = options.lazyDecode !== undefined ? Boolean(options.lazyDecode) : true;
  const streamThresholdBytes = validateStreamThresholdBytes(options.streamThresholdBytes, 0);

  if (streamThresholdBytes > 0) {
    const initialRangeBytes = validateInitialRangeBytes(options.initialRangeBytes);
    const maxFullFallbackBytes = validateMaxFullFallbackBytes(options.maxFullFallbackBytes);
    resolveMaxCacheBytes(options);
    validateMaxCacheEntries(options.maxCacheEntries, null);
    validateMaxConcurrentRangeFetches(options.maxConcurrentRangeFetches);
    const maxRangeFetchRetries = validateMaxRangeFetchRetries(options.maxRangeFetchRetries);
    const rangeRetryDelayMs = validateRangeRetryDelayMs(options.rangeRetryDelayMs);
    const sleepFn = validateSleepFn(options.sleepFn);
    resolvePrefetchMaxGapBytes(options);
    resolvePrefetchMaxSpanBytes(options);

    // 1. Prefer lightweight HEAD probe for Content-Length & Accept-Ranges.
    let headResolved = false;
    let useFullGet = false;
    let useRanged = false;

    const headResponse = await fetcher(
      url,
      mergeRequestInit(options.fetchInit, { method: 'HEAD', cache: 'no-store' })
    );
    if (headResponse && headResponse.ok) {
      const acceptRanges = getResponseHeader(headResponse, 'accept-ranges');
      const contentLength = parseNonNegativeInt(getResponseHeader(headResponse, 'content-length'));
      const rangeDisabled = Boolean(acceptRanges && acceptRanges.trim().toLowerCase() === 'none');
      if (rangeDisabled) {
        if (
          contentLength !== null &&
          contentLength > streamThresholdBytes &&
          maxFullFallbackBytes > 0 &&
          contentLength > maxFullFallbackBytes
        ) {
          const err = new Error(
            `WGSL pack "${url}" is ${contentLength} bytes (> ${maxFullFallbackBytes} byte full-buffer cap) and server returned Accept-Ranges: none; refusing full-pack download`
          );
          err.code = 'WGPK_FULL_FALLBACK_REJECTED';
          err.contentLength = contentLength;
          throw err;
        }
        headResolved = true;
        useFullGet = true;
      } else if (contentLength !== null) {
        headResolved = true;
        if (contentLength > streamThresholdBytes) {
          useRanged = true;
        } else {
          useFullGet = true;
        }
      }
    } else if (headResponse && headResponse.status !== 405 && headResponse.status !== 501) {
      const status = headResponse.status;
      const statusText = headResponse.statusText ? ` ${headResponse.statusText}` : '';
      throw new Error(`Failed to fetch WGSL pack from "${url}": HTTP ${status}${statusText}`);
    }

    if (headResolved && useRanged) {
      return loadIndexFromUrl(url, fetcher, options);
    }

    if (!headResolved && !useFullGet) {
      // 2. Fallback probe when HEAD is unsupported or lacks Content-Length: initial byte range.
      let probeFetches = 0;
      let probeRetries = 0;
      const probe = await fetchRangeChunkWithRetry(
        fetcher,
        url,
        0,
        initialRangeBytes - 1,
        options.fetchInit,
        {
          maxRangeFetchRetries,
          rangeRetryDelayMs,
          sleepFn,
          maxFullFallbackBytes,
          onAttempt: () => {
            probeFetches += 1;
          },
          onRetry: () => {
            probeRetries += 1;
          },
        }
      );
      if (probe.status !== 206) {
        return parseWgslPack(probe.bytes, { lazyDecode });
      }
      const probeTotal = probe.contentRange ? probe.contentRange.total : null;
      if (probeTotal !== null && probeTotal <= streamThresholdBytes) {
        if (
          probe.bytes.byteLength >= probeTotal &&
          (!probe.contentRange || probe.contentRange.start === 0)
        ) {
          return parseWgslPack(probe.bytes.subarray(0, probeTotal), { lazyDecode });
        }
        // Small pack larger than initialRangeBytes: fall through to full GET below.
      } else {
        return finishPackIndexFromInitialChunk(url, fetcher, probe, {
          ...options,
          initialRangeFetches: probeFetches,
          initialRangeFetchRetries: probeRetries,
        });
      }
    }
  }

  const response = await fetcher(url, options.fetchInit);
  if (!response || !response.ok) {
    const status = response ? response.status : 'unknown';
    const statusText = response && response.statusText ? ` ${response.statusText}` : '';
    throw new Error(`Failed to fetch WGSL pack from "${url}": HTTP ${status}${statusText}`);
  }

  const buffer = await response.arrayBuffer();
  return parseWgslPack(buffer, { lazyDecode });
}

/**
 * Alias for `parseWgslPack(buffer, options)`.
 *
 * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
 * @param {{ lazyDecode?: boolean }} [options]
 * @returns {WgslPack}
 */
export function loadFromBuffer(buffer, options = {}) {
  return parseWgslPack(buffer, options);
}

/**
 * Multi-pack WGSL loader and compiled WebGPU `GPUShaderModule` cache.
 */
export class WgslPackCache {
  /**
   * @param {{
   *   lazyDecode?: boolean,
   *   labelPrefix?: string,
   *   prewarmConcurrency?: number,
   *   fetch?: typeof fetch,
   * }} [options]
   */
  constructor(options = {}) {
    this._lazyDecode = options.lazyDecode !== undefined ? Boolean(options.lazyDecode) : true;
    this._labelPrefix = options.labelPrefix !== undefined ? String(options.labelPrefix) : 'wgpk';
    this._prewarmConcurrency =
      options.prewarmConcurrency !== undefined ? options.prewarmConcurrency : 8;
    if (!Number.isInteger(this._prewarmConcurrency) || this._prewarmConcurrency < 1) {
      throw new RangeError('prewarmConcurrency must be a positive integer');
    }
    this._fetch = typeof options.fetch === 'function' ? options.fetch : null;
    this._maxCachedWgslBytes = resolveMaxCacheBytes(options);
    this._maxCacheEntries = validateMaxCacheEntries(options.maxCacheEntries, null);
    this._initialRangeBytes = validateInitialRangeBytes(options.initialRangeBytes);
    this._streamThresholdBytes = validateStreamThresholdBytes(options.streamThresholdBytes, 0);
    this._maxConcurrentRangeFetches = validateMaxConcurrentRangeFetches(
      options.maxConcurrentRangeFetches
    );
    this._maxRangeFetchRetries = validateMaxRangeFetchRetries(options.maxRangeFetchRetries);
    this._rangeRetryDelayMs = validateRangeRetryDelayMs(options.rangeRetryDelayMs);
    this._sleepFn = validateSleepFn(options.sleepFn);
    this._prefetchMaxGapBytes = resolvePrefetchMaxGapBytes(options);
    this._prefetchMaxSpanBytes = resolvePrefetchMaxSpanBytes(options);
    this._releaseSourceAfterCompile = validateReleaseSourceAfterCompile(
      options.releaseSourceAfterCompile,
      'auto'
    );
    this._maxCompiledModules = validateMaxCompiledModules(
      options.maxCompiledModules !== undefined
        ? options.maxCompiledModules
        : options.maxCachedModules,
      DEFAULT_MAX_COMPILED_MODULES
    );

    this._packs = new Map();
    this._shaderEntries = new Map();
    this._dynamicEntries = new Map();
    this._deviceModules = new Map();
    this._indexMeta = new Map();
    this._pinnedHashes = new Set();
    this._shaderIndexAbiVersion = null;
    this._shaderIndexPackPath = null;
    this._nextPackId = 0;

    this._stats = {
      hits: 0,
      misses: 0,
      compiles: 0,
      evictions: 0,
      moduleLruEvictions: 0,
      dynamicHits: 0,
    };
    // W363: Monotonic mutation versions + cached stats / entry-stage counts for O(1) steady-state getStats().
    this._statsVersion = 0;
    this._entriesVersion = 0;
    this._cachedEntryCounts = null;
    this._cachedEntryCountsVersion = -1;
    this._cachedStats = null;
    this._cachedStatsVersion = -1;
    this._cachedPackStamps = null;
  }

  _syncReloadedPack(pack, name) {
    if (this._packs.get(name) !== pack) {
      return;
    }
    for (const [hash, record] of this._shaderEntries.entries()) {
      if (record.packName === name) {
        this._shaderEntries.delete(hash);
        if (!this._dynamicEntries.has(hash)) {
          this._evictModuleOnly(hash);
        }
      }
    }
    const toPinOnPack = [];
    for (const entry of pack.listEntries()) {
      if (this._shaderEntries.has(entry.hash) && !this._dynamicEntries.has(entry.hash)) {
        this._evictModuleOnly(entry.hash);
      }
      this._shaderEntries.set(entry.hash, { entry, pack, packName: name });
      if (this._pinnedHashes.has(entry.hash)) {
        toPinOnPack.push(entry.hash);
      }
    }
    if (toPinOnPack.length > 0 && typeof pack.pinHashes === 'function') {
      pack.pinHashes(toPinOnPack);
    }
    this._entriesVersion += 1;
    this._statsVersion += 1;
  }

  _registerPack(pack, packName) {
    const name =
      typeof packName === 'string' && packName.length > 0
        ? packName
        : `pack-${this._nextPackId++}`;

    // If replacing an existing pack with the same name, detach its old entries first.
    const previousPack = this._packs.get(name);
    if (previousPack) {
      previousPack._onIndexReloaded = null;
      for (const [hash, record] of this._shaderEntries.entries()) {
        if (record.packName === name) {
          this._shaderEntries.delete(hash);
          if (!this._dynamicEntries.has(hash)) {
            this._evictModuleOnly(hash);
          }
        }
      }
    }

    this._packs.set(name, pack);
    pack._onIndexReloaded = () => this._syncReloadedPack(pack, name);
    const toPinOnPack = [];
    for (const entry of pack.listEntries()) {
      if (this._shaderEntries.has(entry.hash) && !this._dynamicEntries.has(entry.hash)) {
        this._evictModuleOnly(entry.hash);
      }
      this._shaderEntries.set(entry.hash, { entry, pack, packName: name });
      if (this._pinnedHashes.has(entry.hash)) {
        toPinOnPack.push(entry.hash);
      }
    }
    if (toPinOnPack.length > 0 && typeof pack.pinHashes === 'function') {
      pack.pinHashes(toPinOnPack);
    }

    this._entriesVersion += 1;
    this._statsVersion += 1;
    return pack;
  }

  /**
   * Parse and register a `.wgsl.pack` binary buffer.
   * If a pack with the same `packName` is already loaded, its entries are replaced.
   * Active dynamic shader overrides are preserved across static pack loads.
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
   * @param {string} [packName]
   * @param {{ lazyDecode?: boolean }} [options]
   * @returns {WgslPack}
   */
  loadPackBuffer(buffer, packName, options = {}) {
    const lazyDecode =
      options && options.lazyDecode !== undefined ? Boolean(options.lazyDecode) : this._lazyDecode;
    const pack = parseWgslPack(buffer, { lazyDecode });
    return this._registerPack(pack, packName);
  }

  /**
   * Alias for `loadPackBuffer` accepting either `(buffer, packName, options)` or `(buffer, options)`.
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} buffer
   * @param {string | { packName?: string, lazyDecode?: boolean }} [packNameOrOptions]
   * @param {{ lazyDecode?: boolean }} [maybeOptions]
   * @returns {WgslPack}
   */
  loadFromBuffer(buffer, packNameOrOptions, maybeOptions) {
    if (packNameOrOptions && typeof packNameOrOptions === 'object') {
      return this.loadPackBuffer(buffer, packNameOrOptions.packName, packNameOrOptions);
    }
    return this.loadPackBuffer(buffer, packNameOrOptions, maybeOptions);
  }

  /**
   * Fetch only the WGPK header + entry table from a URL via HTTP `206 Range` and register
   * the resulting ranged pack (or buffer fallback on `200 OK`).
   *
   * @param {string} url
   * @param {typeof fetch | {
   *   packName?: string,
   *   fetch?: typeof fetch,
   *   fetchInit?: RequestInit,
   *   initialRangeBytes?: number,
   *   maxCachedWgslBytes?: number,
   *   lazyDecode?: boolean,
   * }} [fetchImplOrOptions]
   * @param {object} [maybeOptions]
   * @returns {Promise<WgslPack>}
   */
  async loadIndexFromUrl(url, fetchImplOrOptions, maybeOptions) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError('url must be a non-empty string');
    }
    const { fetcher, options } = resolveFetchAndOptions(
      fetchImplOrOptions,
      maybeOptions,
      this._fetch
    );
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for loadIndexFromUrl');
    }

    const pack = await loadIndexFromUrl(url, fetcher, {
      fetchInit: options.fetchInit,
      initialRangeBytes: options.initialRangeBytes ?? this._initialRangeBytes,
      maxCachedWgslBytes: resolveMaxCacheBytes(options, this._maxCachedWgslBytes),
      maxCacheEntries:
        options.maxCacheEntries !== undefined
          ? validateMaxCacheEntries(options.maxCacheEntries, null)
          : this._maxCacheEntries,
      maxConcurrentRangeFetches:
        options.maxConcurrentRangeFetches ?? this._maxConcurrentRangeFetches,
      maxRangeFetchRetries: options.maxRangeFetchRetries ?? this._maxRangeFetchRetries,
      rangeRetryDelayMs: options.rangeRetryDelayMs ?? this._rangeRetryDelayMs,
      sleepFn: options.sleepFn ?? this._sleepFn,
      prefetchMaxGapBytes: resolvePrefetchMaxGapBytes(options, this._prefetchMaxGapBytes),
      prefetchMaxSpanBytes: resolvePrefetchMaxSpanBytes(options, this._prefetchMaxSpanBytes),
      lazyDecode: options.lazyDecode !== undefined ? options.lazyDecode : this._lazyDecode,
    });
    const name = options.packName ?? url;
    return this._registerPack(pack, name);
  }

  /**
   * Alias for `loadIndexFromUrl`.
   */
  loadPackIndexUrl(url, fetchImplOrOptions, maybeOptions) {
    return this.loadIndexFromUrl(url, fetchImplOrOptions, maybeOptions);
  }

  /**
   * Fetch a `.wgsl.pack` bundle from a URL and register it in the cache.
   * If `options.streamThresholdBytes > 0` (or configured on the cache), automatically uses
   * HTTP `206 Range` index-only mode when the remote pack exceeds the threshold.
   *
   * @param {string} url
   * @param {typeof fetch | {
   *   packName?: string,
   *   lazyDecode?: boolean,
   *   fetch?: typeof fetch,
   *   fetchInit?: RequestInit,
   *   streamThresholdBytes?: number,
   *   initialRangeBytes?: number,
   *   maxCachedWgslBytes?: number,
   * }} [fetchImplOrOptions]
   * @param {object} [maybeOptions]
   * @returns {Promise<WgslPack>}
   */
  async loadPackUrl(url, fetchImplOrOptions, maybeOptions) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError('url must be a non-empty string');
    }
    const { fetcher, options } = resolveFetchAndOptions(
      fetchImplOrOptions,
      maybeOptions,
      this._fetch
    );
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for loadPackUrl');
    }

    const streamThresholdBytes = validateStreamThresholdBytes(
      options.streamThresholdBytes !== undefined
        ? options.streamThresholdBytes
        : this._streamThresholdBytes,
      0
    );
    const lazyDecode = options.lazyDecode !== undefined ? Boolean(options.lazyDecode) : this._lazyDecode;
    const name = options.packName ?? url;

    if (streamThresholdBytes > 0) {
      const pack = await loadFromUrl(url, fetcher, {
        fetchInit: options.fetchInit,
        lazyDecode,
        streamThresholdBytes,
        initialRangeBytes: options.initialRangeBytes ?? this._initialRangeBytes,
        maxCachedWgslBytes: resolveMaxCacheBytes(options, this._maxCachedWgslBytes),
        maxCacheEntries:
          options.maxCacheEntries !== undefined
            ? validateMaxCacheEntries(options.maxCacheEntries, null)
            : this._maxCacheEntries,
        maxConcurrentRangeFetches:
          options.maxConcurrentRangeFetches ?? this._maxConcurrentRangeFetches,
        maxRangeFetchRetries: options.maxRangeFetchRetries ?? this._maxRangeFetchRetries,
        rangeRetryDelayMs: options.rangeRetryDelayMs ?? this._rangeRetryDelayMs,
        sleepFn: options.sleepFn ?? this._sleepFn,
        prefetchMaxGapBytes: resolvePrefetchMaxGapBytes(options, this._prefetchMaxGapBytes),
        prefetchMaxSpanBytes: resolvePrefetchMaxSpanBytes(options, this._prefetchMaxSpanBytes),
      });
      return this._registerPack(pack, name);
    }

    const response = await fetcher(url, options.fetchInit);
    if (!response || !response.ok) {
      const status = response ? response.status : 'unknown';
      const statusText = response && response.statusText ? ` ${response.statusText}` : '';
      throw new Error(`Failed to fetch WGSL pack from "${url}": HTTP ${status}${statusText}`);
    }

    const buffer = await response.arrayBuffer();
    return this.loadPackBuffer(buffer, name, { lazyDecode });
  }

  /**
   * Alias for `loadPackUrl`.
   */
  loadFromUrl(url, fetchImplOrOptions, maybeOptions) {
    return this.loadPackUrl(url, fetchImplOrOptions, maybeOptions);
  }

  /**
   * Load optional `shader-index.json` metadata generated by `playgta6-shaders`.
   *
   * @param {object | string} indexData
   * @returns {{ abiVersion: number, pack: string | null, count: number }}
   */
  loadShaderIndex(indexData) {
    const parsed = typeof indexData === 'string' ? JSON.parse(indexData) : indexData;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new TypeError('shader-index.json payload must be a JSON object');
    }

    const abiVersion = parsed.abi_version;
    if (abiVersion !== SHADER_INDEX_ABI_VERSION) {
      throw new Error(
        `Unsupported shader-index.json abi_version ${abiVersion} (expected ${SHADER_INDEX_ABI_VERSION})`
      );
    }

    if (!parsed.shaders || typeof parsed.shaders !== 'object' || Array.isArray(parsed.shaders)) {
      throw new TypeError('shader-index.json must contain a "shaders" object');
    }

    this._shaderIndexAbiVersion = abiVersion;
    this._shaderIndexPackPath = typeof parsed.pack === 'string' ? parsed.pack : null;

    let count = 0;
    for (const [hashKey, raw] of Object.entries(parsed.shaders)) {
      if (!raw || typeof raw !== 'object') {
        throw new TypeError(`Invalid shader-index.json entry for "${hashKey}"`);
      }
      const hash = normalizeHash(hashKey);
      const hashHex = formatHashHex(hash);
      this._indexMeta.set(hash, {
        hash,
        hashHex,
        stage: typeof raw.stage === 'string' ? raw.stage : null,
        wgslPath: typeof raw.wgsl === 'string' ? raw.wgsl : null,
        entryPoint:
          typeof raw.entry_point === 'string'
            ? raw.entry_point
            : typeof raw.entryPoint === 'string'
              ? raw.entryPoint
              : null,
        packOffset:
          typeof raw.pack_offset === 'number'
            ? raw.pack_offset
            : typeof raw.packOffset === 'number'
              ? raw.packOffset
              : null,
        packLength:
          typeof raw.pack_length === 'number'
            ? raw.pack_length
            : typeof raw.packLength === 'number'
              ? raw.packLength
              : null,
      });
      count += 1;
    }

    this._statsVersion += 1;
    return {
      abiVersion,
      pack: this._shaderIndexPackPath,
      count,
    };
  }

  /**
   * Fetch and register `shader-index.json` metadata from a URL.
   *
   * @param {string} url
   * @param {typeof fetch} [fetchImpl]
   */
  async loadShaderIndexUrl(url, fetchImpl) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError('url must be a non-empty string');
    }
    const fetcher = fetchImpl ?? this._fetch ?? globalThis.fetch;
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for loadShaderIndexUrl');
    }
    const response = await fetcher(url);
    if (!response || !response.ok) {
      const status = response ? response.status : 'unknown';
      const statusText = response && response.statusText ? ` ${response.statusText}` : '';
      throw new Error(`Failed to fetch shader-index.json from "${url}": HTTP ${status}${statusText}`);
    }
    const json = await response.json();
    return this.loadShaderIndex(json);
  }

  /**
   * Register a live dynamic WGSL shader at runtime (e.g., freshly translated RDNA2->WGSL
   * or hot-patched shader). Pass `{ override: true }` in `meta` to supersede an existing
   * static pack entry while preserving the static entry for `clearDynamicShaders()`.
   * Automatically evicts any stale compiled `GPUShaderModule` cached for `hash`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} stage
   * @param {string} wgslSource
   * @param {{ override?: boolean, label?: string, entryPoint?: string, source?: string, tags?: string[] }} [meta]
   */
  registerDynamicShader(hash, stage, wgslSource, meta) {
    const key = normalizeHash(hash);
    const staticRecord = this._shaderEntries.get(key);
    const prevDynamic = this._dynamicEntries.get(key);
    const entry = createDynamicShaderEntry(hash, stage, wgslSource, meta, prevDynamic);

    if (staticRecord && !entry.override) {
      throw new Error(
        `Shader ${entry.hashHex} already exists in static pack "${staticRecord.packName}"; pass { override: true } to supersede it`
      );
    }

    this._dynamicEntries.set(key, entry);
    this._evictModuleOnly(key);
    this._entriesVersion += 1;
    this._statsVersion += 1;
    return formatDynamicLookupRecord(entry, Boolean(staticRecord));
  }

  /**
   * Check whether a dynamic shader is registered for `hash` (and optional `stage`).
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {boolean}
   */
  hasDynamicShader(hash, stage) {
    const key = normalizeHash(hash);
    const entry = this._dynamicEntries.get(key);
    if (!entry) {
      return false;
    }
    if (stage !== undefined && stage !== null) {
      return entry.stageCode === resolveStageCode(stage);
    }
    return true;
  }

  /**
   * Remove a single dynamic shader by hash and evict its compiled module(s),
   * restoring any underlying static pack entry.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  unregisterDynamicShader(hash) {
    const key = normalizeHash(hash);
    const removed = this._dynamicEntries.delete(key);
    if (removed) {
      this._evictModuleOnly(key);
      this._entriesVersion += 1;
      this._statsVersion += 1;
    }
    return removed;
  }

  /**
   * Remove all dynamically registered shaders and evict their compiled `GPUShaderModule`s
   * while preserving all static packs, static entries, and shader-index metadata.
   *
   * @returns {number} Number of dynamic shaders cleared
   */
  clearDynamicShaders() {
    const count = this._dynamicEntries.size;
    if (count > 0) {
      for (const key of this._dynamicEntries.keys()) {
        this._evictModuleOnly(key);
      }
      this._dynamicEntries.clear();
      this._entriesVersion += 1;
      this._statsVersion += 1;
    }
    return count;
  }

  /**
   * Look up a shader entry by hash (and optional stage), checking dynamic overrides first
   * and then static pack entries. Increments `dynamicHits` when resolved dynamically.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   */
  lookup(hash, stage) {
    const key = normalizeHash(hash);
    const expectedStage =
      stage !== undefined && stage !== null ? resolveStageCode(stage) : null;

    const dynamicEntry = this._dynamicEntries.get(key);
    if (dynamicEntry) {
      if (expectedStage !== null && dynamicEntry.stageCode !== expectedStage) {
        return undefined;
      }
      this._stats.dynamicHits += 1;
      this._statsVersion += 1;
      return formatDynamicLookupRecord(dynamicEntry, this._shaderEntries.has(key));
    }

    const record = this._shaderEntries.get(key);
    if (!record) {
      return undefined;
    }
    if (expectedStage !== null && record.entry.stageCode !== expectedStage) {
      return undefined;
    }
    const indexMeta = this._indexMeta.get(key);
    return {
      hash: record.entry.hash,
      hashHex: record.entry.hashHex,
      stage: record.entry.stage,
      stageCode: record.entry.stageCode,
      wgsl: record.entry.wgsl,
      length: record.entry.length,
      offset: record.entry.offset,
      dynamic: false,
      overridesStatic: false,
      packName: record.packName,
      source: 'pack',
      label: null,
      entryPoint: indexMeta ? indexMeta.entryPoint : null,
      tags: [],
      revision: 0,
      registeredAt: null,
    };
  }

  /**
   * Alias for `lookup(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   */
  get(hash, stage) {
    return this.lookup(hash, stage);
  }

  /**
   * Alias for `lookup(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   */
  getEntry(hash, stage) {
    return this.lookup(hash, stage);
  }

  /**
   * Check whether a decoded JS WGSL string is currently retained in heap memory for `hash`
   * without triggering lazy UTF-8 decoding or HTTP Range fetches.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  hasCachedWgslText(hash) {
    const key = normalizeHash(hash);
    if (this._dynamicEntries.has(key)) {
      return true;
    }
    const record = this._shaderEntries.get(key);
    if (!record || !record.pack || typeof record.pack.hasCachedWgslText !== 'function') {
      return false;
    }
    return record.pack.hasCachedWgslText(key);
  }

  /**
   * Release the cached decoded JS WGSL string for `hash` from its backing pack while preserving
   * compact binary index metadata so `getWgsl(hash)` / `getWgslAsync(hash)` can lazily re-decode
   * on demand (#2162).
   *
   * @param {bigint | string | number} hash
   * @param {{ includePinned?: boolean }} [options]
   * @returns {boolean}
   */
  releaseWgslText(hash, options = {}) {
    const key = normalizeHash(hash);
    const record = this._shaderEntries.get(key);
    if (!record || !record.pack || typeof record.pack.releaseWgslText !== 'function') {
      return false;
    }
    const released = record.pack.releaseWgslText(key, options);
    if (released) {
      this._statsVersion += 1;
    }
    return released;
  }

  /**
   * Release all cached decoded JS WGSL strings across all loaded packs while preserving compact
   * binary index metadata and compiled `GPUShaderModule` caches (#2162).
   *
   * @param {{ includePinned?: boolean }} [options]
   * @returns {{ releasedCount: number, bytesReleased: number }}
   */
  releaseAllWgslText(options = {}) {
    let releasedCount = 0;
    let bytesReleased = 0;
    for (const pack of this._packs.values()) {
      if (typeof pack.releaseAllWgslText === 'function') {
        const res = pack.releaseAllWgslText(options);
        releasedCount += res.releasedCount;
        bytesReleased += res.bytesReleased;
      }
    }
    if (releasedCount > 0) {
      this._statsVersion += 1;
    }
    return { releasedCount, bytesReleased };
  }

  /**
   * Return whether any dynamic override or loaded static pack contains the given shader hash
   * (optionally matching `stage`).
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {boolean}
   */
  hasShader(hash, stage) {
    const key = normalizeHash(hash);
    const expectedStage =
      stage !== undefined && stage !== null ? resolveStageCode(stage) : null;

    const dynamicEntry = this._dynamicEntries.get(key);
    if (dynamicEntry) {
      return expectedStage === null || dynamicEntry.stageCode === expectedStage;
    }

    const record = this._shaderEntries.get(key);
    if (!record) {
      return false;
    }
    return expectedStage === null || record.entry.stageCode === expectedStage;
  }

  /**
   * Alias for `hasShader(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {boolean}
   */
  has(hash, stage) {
    return this.hasShader(hash, stage);
  }

  /**
   * Get the WGSL source string for a shader hash, checking dynamic overrides first
   * and then loaded static packs.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {string | undefined}
   */
  getShaderSource(hash, stage) {
    const found = this.lookup(hash, stage);
    return found ? found.wgsl : undefined;
  }

  /**
   * Alias for `getShaderSource(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {string | undefined}
   */
  getWgsl(hash, stage) {
    return this.getShaderSource(hash, stage);
  }

  /**
   * Asynchronously retrieve the WGSL source string for a shader hash across both
   * full-buffer packs and HTTP-ranged packs (checking dynamic overrides first).
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {Promise<string | null>}
   */
  async getWgslAsync(hash, stage) {
    const key = normalizeHash(hash);
    const expectedStage =
      stage !== undefined && stage !== null ? resolveStageCode(stage) : null;

    const dynamicEntry = this._dynamicEntries.get(key);
    if (dynamicEntry) {
      if (expectedStage !== null && dynamicEntry.stageCode !== expectedStage) {
        return null;
      }
      this._stats.dynamicHits += 1;
      this._statsVersion += 1;
      return dynamicEntry.wgsl;
    }

    const record = this._shaderEntries.get(key);
    if (!record) {
      return null;
    }
    if (expectedStage !== null && record.entry.stageCode !== expectedStage) {
      return null;
    }
    return record.pack.getWgslAsync(key);
  }

  /**
   * Alias for `getWgslAsync(hash, stage)`.
   *
   * @param {bigint | string | number} hash
   * @param {number | string} [stage]
   * @returns {Promise<string | null>}
   */
  getShaderSourceAsync(hash, stage) {
    return this.getWgslAsync(hash, stage);
  }

  /**
   * Pin one or more shader hashes across loaded (and future) packs so their decoded WGSL
   * payloads stay resident in cache and are never evicted under LRU byte or entry limits.
   *
   * @param {Iterable<bigint | string | number>} hashes
   * @returns {number} Number of hashes newly pinned
   */
  pinHashes(hashes) {
    const keys = normalizeHashIterable(hashes);
    let added = 0;
    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      if (!this._pinnedHashes.has(key)) {
        this._pinnedHashes.add(key);
        added += 1;
      }
      const record = this._shaderEntries.get(key);
      if (record && record.pack && typeof record.pack.pinHashes === 'function') {
        record.pack.pinHashes([key]);
      }
    }
    if (added > 0) {
      this._statsVersion += 1;
    }
    return added;
  }

  /**
   * Unpin one or more shader hashes across all loaded packs so they become eligible for
   * standard O(1) LRU eviction.
   *
   * @param {Iterable<bigint | string | number>} hashes
   * @returns {number} Number of hashes newly unpinned
   */
  unpinHashes(hashes) {
    const keys = normalizeHashIterable(hashes);
    let removed = 0;
    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      let wasPinned = this._pinnedHashes.delete(key);
      for (const pack of this._packs.values()) {
        if (typeof pack.unpinHashes === 'function' && pack.unpinHashes([key]) > 0) {
          wasPinned = true;
        }
      }
      if (wasPinned) {
        removed += 1;
      }
    }
    if (removed > 0) {
      this._statsVersion += 1;
    }
    return removed;
  }

  /**
   * Return whether a shader hash is currently pinned on this loader/cache or any loaded pack.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  isPinned(hash) {
    const key = normalizeHash(hash);
    if (this._pinnedHashes.has(key)) {
      return true;
    }
    const record = this._shaderEntries.get(key);
    if (record && record.pack && typeof record.pack.isPinned === 'function') {
      return record.pack.isPinned(key);
    }
    return false;
  }

  /**
   * Return all currently pinned shader hashes across this loader/cache and its loaded packs.
   *
   * @returns {bigint[]}
   */
  getPinnedHashes() {
    const union = new Set(this._pinnedHashes);
    for (const pack of this._packs.values()) {
      if (pack._pinnedHashes) {
        for (const key of pack._pinnedHashes) {
          union.add(key);
        }
      }
    }
    return Array.from(union);
  }

  /**
   * Prefetch a set of shader hashes across all loaded packs, coalescing nearby byte ranges
   * per ranged pack into batched HTTP `206 Range` fetches and populating each pack's bounded
   * hot-hash WGSL LRU cache.
   *
   * @param {Iterable<bigint | string | number>} hashes
   * @param {{
   *   maxCoalesceGapBytes?: number,
   *   maxGapBytes?: number,
   *   maxCoalesceSpanBytes?: number,
   *   maxSpanBytes?: number,
   *   prefetchMaxGapBytes?: number,
   *   prefetchMaxSpanBytes?: number,
   * }} [options]
   * @returns {Promise<{
   *   requested: number,
   *   prefetched: number,
   *   cached: number,
   *   inFlight: number,
   *   missing: number,
   *   batches: number,
   *   bytesFetched: number,
   * }>}
   */
  async prefetchHashes(hashes, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const hasGapOverride =
      opts.maxCoalesceGapBytes !== undefined ||
      opts.maxGapBytes !== undefined ||
      opts.prefetchMaxGapBytes !== undefined;
    const hasSpanOverride =
      opts.maxCoalesceSpanBytes !== undefined ||
      opts.maxSpanBytes !== undefined ||
      opts.prefetchMaxSpanBytes !== undefined;
    const maxGapBytes = hasGapOverride
      ? resolvePrefetchMaxGapBytes(opts, this._prefetchMaxGapBytes)
      : undefined;
    const maxSpanBytes = hasSpanOverride
      ? resolvePrefetchMaxSpanBytes(opts, this._prefetchMaxSpanBytes)
      : undefined;

    if (hashes === undefined || hashes === null) {
      return {
        requested: 0,
        prefetched: 0,
        cached: 0,
        inFlight: 0,
        missing: 0,
        batches: 0,
        bytesFetched: 0,
      };
    }
    if (typeof hashes[Symbol.iterator] !== 'function') {
      throw new TypeError('hashes must be an iterable of shader hashes');
    }

    const seen = new Set();
    let requested = 0;
    let cached = 0;
    let missing = 0;
    const byPack = new Map();

    for (const rawHash of hashes) {
      const key = normalizeHash(rawHash);
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      requested += 1;

      if (this._dynamicEntries.has(key)) {
        cached += 1;
        continue;
      }

      const record = this._shaderEntries.get(key);
      if (!record) {
        missing += 1;
        continue;
      }

      let list = byPack.get(record.pack);
      if (!list) {
        list = [];
        byPack.set(record.pack, list);
      }
      list.push(key);
    }

    if (byPack.size === 0) {
      return {
        requested,
        prefetched: 0,
        cached,
        inFlight: 0,
        missing,
        batches: 0,
        bytesFetched: 0,
      };
    }

    const forwardOptions = {};
    if (maxGapBytes !== undefined) {
      forwardOptions.maxCoalesceGapBytes = maxGapBytes;
    }
    if (maxSpanBytes !== undefined) {
      forwardOptions.maxCoalesceSpanBytes = maxSpanBytes;
    }

    const results = await Promise.all(
      Array.from(byPack.entries(), ([pack, keys]) => pack.prefetchHashes(keys, forwardOptions))
    );

    let prefetched = 0;
    let inFlight = 0;
    let batches = 0;
    let bytesFetched = 0;
    for (let i = 0; i < results.length; i += 1) {
      const r = results[i];
      prefetched += r.prefetched;
      cached += r.cached;
      inFlight += r.inFlight;
      missing += r.missing;
      batches += r.batches;
      bytesFetched += r.bytesFetched;
    }

    return {
      requested,
      prefetched,
      cached,
      inFlight,
      missing,
      batches,
      bytesFetched,
    };
  }

  /**
   * Get combined dynamic + static pack + optional `shader-index.json` metadata for a shader hash.
   *
   * @param {bigint | string | number} hash
   */
  getShaderMetadata(hash) {
    const key = normalizeHash(hash);
    const dynamicEntry = this._dynamicEntries.get(key);
    const record = this._shaderEntries.get(key);
    const indexMeta = this._indexMeta.get(key);
    if (!dynamicEntry && !record && !indexMeta) {
      return undefined;
    }
    if (dynamicEntry) {
      return {
        hash: key,
        hashHex: formatHashHex(key),
        stage: dynamicEntry.stage,
        stageCode: dynamicEntry.stageCode,
        packName: record ? record.packName : null,
        offset: record ? record.entry.offset : indexMeta ? indexMeta.packOffset : null,
        length: dynamicEntry.length,
        wgslPath: indexMeta ? indexMeta.wgslPath : null,
        entryPoint:
          dynamicEntry.entryPoint !== null
            ? dynamicEntry.entryPoint
            : indexMeta
              ? indexMeta.entryPoint
              : null,
        dynamic: true,
        overridesStatic: Boolean(record),
        source: dynamicEntry.source,
        label: dynamicEntry.label,
        tags: dynamicEntry.tags.slice(),
        revision: dynamicEntry.revision,
        registeredAt: dynamicEntry.registeredAt,
      };
    }
    return {
      hash: key,
      hashHex: formatHashHex(key),
      stage: record ? record.entry.stage : indexMeta.stage,
      stageCode: record ? record.entry.stageCode : null,
      packName: record ? record.packName : null,
      offset: record ? record.entry.offset : indexMeta.packOffset,
      length: record ? record.entry.length : indexMeta.packLength,
      wgslPath: indexMeta ? indexMeta.wgslPath : null,
      entryPoint: indexMeta ? indexMeta.entryPoint : null,
      dynamic: false,
      overridesStatic: false,
    };
  }

  _getDeviceMap(device, createIfMissing) {
    let map = this._deviceModules.get(device);
    if (!map && createIfMissing) {
      map = new Map();
      this._deviceModules.set(device, map);
    }
    return map;
  }

  _hasCachedModule(device, hashBigInt) {
    const map = this._deviceModules.get(device);
    return map ? map.has(hashBigInt) : false;
  }

  _setCachedDeviceModule(deviceMap, key, shaderModule) {
    if (deviceMap.has(key)) {
      deviceMap.delete(key);
    }
    deviceMap.set(key, shaderModule);
    const cap = this._maxCompiledModules;
    if (cap !== null && cap > 0) {
      while (deviceMap.size > cap) {
        const oldestKey = deviceMap.keys().next().value;
        deviceMap.delete(oldestKey);
        this._stats.evictions += 1;
        this._stats.moduleLruEvictions += 1;
      }
    }
  }

  _shouldReleaseSourceAfterCompile(pack) {
    if (!pack || typeof pack.releaseWgslText !== 'function') {
      return false;
    }
    if (this._releaseSourceAfterCompile === true) {
      return true;
    }
    if (this._releaseSourceAfterCompile === 'auto') {
      return pack.mode === 'buffer';
    }
    return false;
  }

  /**
   * Return a cached `GPUShaderModule` for `(device, hash)` or compile and cache it.
   * Resolves dynamic shader overrides ahead of static pack entries.
   *
   * @param {GPUDevice} device
   * @param {bigint | string | number} hash
   * @param {string} [labelPrefix]
   * @returns {GPUShaderModule}
   */
  getOrCreateShaderModule(device, hash, labelPrefix) {
    if (!device || typeof device.createShaderModule !== 'function') {
      throw new TypeError('device must provide a createShaderModule(descriptor) method');
    }

    const key = normalizeHash(hash);
    const deviceMap = this._getDeviceMap(device, true);
    const cached = deviceMap.get(key);
    if (cached !== undefined) {
      if (deviceMap.size > 1) {
        deviceMap.delete(key);
        deviceMap.set(key, cached);
      }
      this._stats.hits += 1;
      if (this._dynamicEntries.has(key)) {
        this._stats.dynamicHits += 1;
      }
      this._statsVersion += 1;
      return cached;
    }

    this._stats.misses += 1;
    this._statsVersion += 1;
    const dynamicEntry = this._dynamicEntries.get(key);
    let entry;
    let staticPack = null;
    if (dynamicEntry) {
      this._stats.dynamicHits += 1;
      entry = dynamicEntry;
    } else {
      const record = this._shaderEntries.get(key);
      if (!record) {
        throw new Error(`Shader ${formatHashHex(key)} not found in any loaded WGSL pack`);
      }
      entry = record.entry;
      staticPack = record.pack;
    }

    const prefix = labelPrefix !== undefined ? String(labelPrefix) : this._labelPrefix;
    const baseLabel = prefix
      ? `${prefix}:${entry.stage}:${entry.hashHex}`
      : `${entry.stage}:${entry.hashHex}`;
    const label =
      dynamicEntry && dynamicEntry.label ? `${baseLabel}:${dynamicEntry.label}` : baseLabel;

    const shaderModule = device.createShaderModule({
      label,
      code: applyRtBvhPrelude(entry.wgsl),
    });

    this._setCachedDeviceModule(deviceMap, key, shaderModule);
    this._stats.compiles += 1;
    if (staticPack && this._shouldReleaseSourceAfterCompile(staticPack)) {
      staticPack.releaseWgslText(key);
    }
    this._statsVersion += 1;
    return shaderModule;
  }

  /**
   * Alias for `getOrCreateShaderModule(device, hash, labelPrefix)`.
   *
   * @param {GPUDevice} device
   * @param {bigint | string | number} hash
   * @param {string} [labelPrefix]
   * @returns {GPUShaderModule}
   */
  compileShaderModule(device, hash, labelPrefix) {
    return this.getOrCreateShaderModule(device, hash, labelPrefix);
  }

  /**
   * Async companion to `getOrCreateShaderModule` supporting both full-buffer and ranged packs.
   *
   * @param {GPUDevice} device
   * @param {bigint | string | number} hash
   * @param {string} [labelPrefix]
   * @returns {Promise<GPUShaderModule>}
   */
  async getOrCreateShaderModuleAsync(device, hash, labelPrefix) {
    if (!device || typeof device.createShaderModule !== 'function') {
      throw new TypeError('device must provide a createShaderModule(descriptor) method');
    }

    const key = normalizeHash(hash);
    const deviceMap = this._getDeviceMap(device, true);
    const cached = deviceMap.get(key);
    if (cached !== undefined) {
      if (deviceMap.size > 1) {
        deviceMap.delete(key);
        deviceMap.set(key, cached);
      }
      this._stats.hits += 1;
      if (this._dynamicEntries.has(key)) {
        this._stats.dynamicHits += 1;
      }
      this._statsVersion += 1;
      return cached;
    }

    const dynamicEntry = this._dynamicEntries.get(key);
    if (dynamicEntry) {
      return this.getOrCreateShaderModule(device, key, labelPrefix);
    }

    const record = this._shaderEntries.get(key);
    if (!record) {
      this._stats.misses += 1;
      this._statsVersion += 1;
      throw new Error(`Shader ${formatHashHex(key)} not found in any loaded WGSL pack`);
    }

    if (record.pack.mode !== 'ranged') {
      return this.getOrCreateShaderModule(device, key, labelPrefix);
    }

    const code = await record.pack.getWgslAsync(key);
    if (code === null) {
      this._stats.misses += 1;
      this._statsVersion += 1;
      throw new Error(`Shader ${formatHashHex(key)} not found in any loaded WGSL pack`);
    }

    // Re-check device cache in case a concurrent caller compiled while we awaited.
    const cachedAfterFetch = deviceMap.get(key);
    if (cachedAfterFetch !== undefined) {
      if (deviceMap.size > 1) {
        deviceMap.delete(key);
        deviceMap.set(key, cachedAfterFetch);
      }
      this._stats.hits += 1;
      this._statsVersion += 1;
      return cachedAfterFetch;
    }

    this._stats.misses += 1;
    this._statsVersion += 1;
    const latestRecord = this._shaderEntries.get(key) || record;
    const prefix = labelPrefix !== undefined ? String(labelPrefix) : this._labelPrefix;
    const label = prefix
      ? `${prefix}:${latestRecord.entry.stage}:${latestRecord.entry.hashHex}`
      : `${latestRecord.entry.stage}:${latestRecord.entry.hashHex}`;

    const shaderModule = device.createShaderModule({
      label,
      code: applyRtBvhPrelude(code),
    });

    this._setCachedDeviceModule(deviceMap, key, shaderModule);
    this._stats.compiles += 1;
    if (latestRecord.pack && this._shouldReleaseSourceAfterCompile(latestRecord.pack)) {
      latestRecord.pack.releaseWgslText(key);
    }
    this._statsVersion += 1;
    return shaderModule;
  }

  /**
   * Batch-compile a set of shader hashes on `device` with bounded concurrency.
   * If `hashes` is omitted, prewarms every shader loaded across all packs.
   *
   * @param {GPUDevice} device
   * @param {Iterable<bigint | string | number>} [hashes]
   * @param {{
   *   concurrency?: number,
   *   labelPrefix?: string,
   *   awaitCompilationInfo?: boolean,
   * }} [options]
   * @returns {Promise<{
   *   total: number,
   *   compiled: number,
   *   cached: number,
   *   modules: Map<bigint, GPUShaderModule>,
   * }>}
   */
  async prewarm(device, hashes, options = {}) {
    if (!device || typeof device.createShaderModule !== 'function') {
      throw new TypeError('device must provide a createShaderModule(descriptor) method');
    }

    const concurrency =
      options && options.concurrency !== undefined
        ? options.concurrency
        : this._prewarmConcurrency;
    if (!Number.isInteger(concurrency) || concurrency < 1) {
      throw new RangeError('prewarm concurrency must be a positive integer');
    }

    const labelPrefix = options ? options.labelPrefix : undefined;
    const awaitCompilationInfo =
      options && options.awaitCompilationInfo !== undefined
        ? Boolean(options.awaitCompilationInfo)
        : true;

    const rawList =
      hashes === undefined || hashes === null
        ? [...this._shaderEntries.keys(), ...this._dynamicEntries.keys()]
        : Array.from(hashes, (h) => normalizeHash(h));

    // Deduplicate while preserving submission order.
    const uniqueHashes = [];
    const seen = new Set();
    for (const h of rawList) {
      if (!seen.has(h)) {
        seen.add(h);
        uniqueHashes.push(h);
      }
    }

    const modules = new Map();
    let compiled = 0;
    let cached = 0;
    let cursor = 0;

    const worker = async () => {
      while (cursor < uniqueHashes.length) {
        const hash = uniqueHashes[cursor++];
        const alreadyCached = this._hasCachedModule(device, hash);
        const record = this._shaderEntries.get(hash);
        const useAsync =
          !alreadyCached &&
          !this._dynamicEntries.has(hash) &&
          record &&
          record.pack.mode === 'ranged';
        const mod = useAsync
          ? await this.getOrCreateShaderModuleAsync(device, hash, labelPrefix)
          : this.getOrCreateShaderModule(device, hash, labelPrefix);
        modules.set(hash, mod);
        if (alreadyCached) {
          cached += 1;
        } else {
          compiled += 1;
          if (awaitCompilationInfo && mod && typeof mod.getCompilationInfo === 'function') {
            await mod.getCompilationInfo();
          }
        }
      }
    };

    const workerCount = Math.min(concurrency, uniqueHashes.length);
    const workers = [];
    for (let i = 0; i < workerCount; i += 1) {
      workers.push(worker());
    }
    await Promise.all(workers);

    return {
      total: uniqueHashes.length,
      compiled,
      cached,
      modules,
    };
  }

  _evictModuleOnly(key) {
    let removed = 0;
    for (const [device, map] of this._deviceModules.entries()) {
      if (map.delete(key)) {
        removed += 1;
        if (map.size === 0) {
          this._deviceModules.delete(device);
        }
      }
    }
    if (removed > 0) {
      this._stats.evictions += removed;
      this._statsVersion += 1;
    }
    return removed > 0;
  }

  /**
   * Evict a compiled `GPUShaderModule` for `hash` from the cache.
   * Pass `{ removeSource: true }` to also drop its loaded pack / dynamic entry.
   *
   * @param {bigint | string | number} hash
   * @param {{ removeSource?: boolean }} [options]
   * @returns {boolean} True if a cached module (or source, when requested) was evicted
   */
  evict(hash, options = {}) {
    const key = normalizeHash(hash);
    const evictedModule = this._evictModuleOnly(key);
    let evictedSource = false;
    if (options && options.removeSource) {
      const removedStatic = this._shaderEntries.delete(key);
      const removedDynamic = this._dynamicEntries.delete(key);
      evictedSource = removedStatic || removedDynamic;
      if (evictedSource) {
        this._entriesVersion += 1;
        this._statsVersion += 1;
      }
    }
    return evictedModule || evictedSource;
  }

  /**
   * Clear all loaded packs, dynamic shaders, cached shader modules, and index metadata.
   *
   * @param {{ resetStats?: boolean }} [options]
   */
  clear(options = {}) {
    for (const pack of this._packs.values()) {
      pack._onIndexReloaded = null;
    }
    this._packs.clear();
    this._shaderEntries.clear();
    this._dynamicEntries.clear();
    this._deviceModules.clear();
    this._indexMeta.clear();
    this._pinnedHashes.clear();
    this._shaderIndexAbiVersion = null;
    this._shaderIndexPackPath = null;
    if (options && options.resetStats) {
      this._stats.hits = 0;
      this._stats.misses = 0;
      this._stats.compiles = 0;
      this._stats.evictions = 0;
      this._stats.moduleLruEvictions = 0;
      this._stats.dynamicHits = 0;
    }
    this._entriesVersion += 1;
    this._statsVersion += 1;
  }

  /**
   * List metadata summaries for all dynamically registered shaders.
   *
   * @param {{ stage?: number | string, includeSource?: boolean }} [options]
   */
  listDynamicShaders(options = {}) {
    const expectedStage =
      options && options.stage !== undefined && options.stage !== null
        ? resolveStageCode(options.stage)
        : null;
    const includeSource = Boolean(options && options.includeSource);

    const results = [];
    for (const entry of this._dynamicEntries.values()) {
      if (expectedStage !== null && entry.stageCode !== expectedStage) {
        continue;
      }
      const item = {
        hash: entry.hash,
        hashHex: entry.hashHex,
        stage: entry.stage,
        stageCode: entry.stageCode,
        length: entry.length,
        dynamic: true,
        overridesStatic: this._shaderEntries.has(entry.hash),
        source: entry.source,
        label: entry.label,
        entryPoint: entry.entryPoint,
        tags: entry.tags.slice(),
        revision: entry.revision,
        registeredAt: entry.registeredAt,
      };
      if (includeSource) {
        item.wgsl = entry.wgsl;
      }
      results.push(item);
    }
    return results;
  }

  /**
   * List active resolvable shaders across static packs and dynamic overrides
   * without triggering lazy UTF-8 decoding unless `includeSource: true`.
   *
   * @param {{
   *   stage?: number | string,
   *   dynamicOnly?: boolean,
   *   staticOnly?: boolean,
   *   includeSource?: boolean,
   * }} [options]
   */
  listShaders(options = {}) {
    const expectedStage =
      options && options.stage !== undefined && options.stage !== null
        ? resolveStageCode(options.stage)
        : null;
    const dynamicOnly = Boolean(options && options.dynamicOnly);
    const staticOnly = Boolean(options && options.staticOnly);
    const includeSource = Boolean(options && options.includeSource);

    const results = [];
    const seen = new Set();

    if (!staticOnly) {
      for (const entry of this._dynamicEntries.values()) {
        seen.add(entry.hash);
        if (expectedStage !== null && entry.stageCode !== expectedStage) {
          continue;
        }
        const item = {
          hash: entry.hash,
          hashHex: entry.hashHex,
          stage: entry.stage,
          stageCode: entry.stageCode,
          length: entry.length,
          dynamic: true,
          overridesStatic: this._shaderEntries.has(entry.hash),
          packName: null,
          source: entry.source,
          label: entry.label,
          entryPoint: entry.entryPoint,
          tags: entry.tags.slice(),
          revision: entry.revision,
          registeredAt: entry.registeredAt,
        };
        if (includeSource) {
          item.wgsl = entry.wgsl;
        }
        results.push(item);
      }
    }

    if (!dynamicOnly) {
      for (const [hash, record] of this._shaderEntries.entries()) {
        if (!staticOnly && seen.has(hash)) {
          continue;
        }
        if (expectedStage !== null && record.entry.stageCode !== expectedStage) {
          continue;
        }
        const indexMeta = this._indexMeta.get(hash);
        const item = {
          hash,
          hashHex: record.entry.hashHex,
          stage: record.entry.stage,
          stageCode: record.entry.stageCode,
          length: record.entry.length,
          offset: record.entry.offset,
          dynamic: false,
          overridesStatic: false,
          overriddenByDynamic: this._dynamicEntries.has(hash),
          packName: record.packName,
          source: 'pack',
          label: null,
          entryPoint: indexMeta ? indexMeta.entryPoint : null,
          tags: [],
          revision: 0,
          registeredAt: null,
        };
        if (includeSource) {
          item.wgsl = record.entry.wgsl;
        }
        results.push(item);
      }
    }

    return results;
  }

  /**
   * Return a JSON-serializable introspection snapshot of loaded packs, index metadata,
   * dynamic overrides, and cache statistics.
   */
  inspect() {
    const stats = this.getStats();
    const dynamicShaders = [];
    for (const entry of this._dynamicEntries.values()) {
      dynamicShaders.push({
        hashHex: entry.hashHex,
        stage: entry.stage,
        stageCode: entry.stageCode,
        length: entry.length,
        overridesStatic: this._shaderEntries.has(entry.hash),
        source: entry.source,
        label: entry.label,
        entryPoint: entry.entryPoint,
        tags: entry.tags.slice(),
        revision: entry.revision,
        registeredAt: entry.registeredAt,
      });
    }

    return {
      stats,
      packNames: Array.from(this._packs.keys()),
      shaderIndex: {
        abiVersion: this._shaderIndexAbiVersion,
        packPath: this._shaderIndexPackPath,
        count: this._indexMeta.size,
      },
      dynamicShaders,
    };
  }

  _packStampsMatch() {
    const stamps = this._cachedPackStamps;
    if (!stamps || stamps.length !== this._packs.size) {
      return false;
    }
    let idx = 0;
    for (const pack of this._packs.values()) {
      const stamp = stamps[idx++];
      if (
        !stamp ||
        stamp.pack !== pack ||
        typeof pack._statsVersion !== 'number' ||
        stamp.version !== pack._statsVersion ||
        (typeof pack._getDecodedCount === 'function' &&
          stamp.decodedCount !== pack._getDecodedCount())
      ) {
        return false;
      }
    }
    return true;
  }

  _cloneCachedStats(cached) {
    const clonedPacks = {};
    const srcPacks = cached.packs;
    for (const name in srcPacks) {
      const ps = srcPacks[name];
      clonedPacks[name] = {
        ...ps,
        stageCounts: { ...ps.stageCounts },
        dynamicStageCounts: { ...ps.dynamicStageCounts },
      };
    }
    return {
      ...cached,
      stageCounts: { ...cached.stageCounts },
      dynamicStageCounts: { ...cached.dynamicStageCounts },
      packs: clonedPacks,
    };
  }

  /**
   * Return cache statistics across all loaded packs, dynamic shaders, and compiled modules.
   */
  getStats() {
    if (
      this._cachedStats !== null &&
      this._cachedStatsVersion === this._statsVersion &&
      this._packStampsMatch()
    ) {
      return this._cloneCachedStats(this._cachedStats);
    }

    let cachedModules = 0;
    for (const map of this._deviceModules.values()) {
      cachedModules += map.size;
    }

    let totalPackBytes = 0;
    let rangeFetches = 0;
    let rangeBytesFetched = 0;
    let rangeFetchRetries = 0;
    let staleIndexReloads = 0;
    let activeRangeFetches = 0;
    let peakConcurrentRangeFetches = 0;
    let cacheHits = 0;
    let cacheMisses = 0;
    let lruHits = 0;
    let lruMisses = 0;
    let lruBytes = 0;
    let cachedWgslBytes = 0;
    let cachedWgslEntries = 0;
    let pinnedBytes = 0;
    let pinnedCachedEntries = 0;
    let wgslCacheEvictions = 0;
    let prefetchBatches = 0;
    let prefetchShaders = 0;
    let coalescedRangeFetches = 0;
    let coalescedEntriesFetched = 0;
    let httpRequestsSavedByCoalescing = 0;
    let coalescedGapBytes = 0;
    let inFlightDeduplications = 0;
    let inFlightCount = 0;
    let wgslTextBytesRetained = 0;
    let wgslTextEntriesRetained = 0;
    let wgslTextBytesReleased = 0;
    let wgslTextReleases = 0;
    let wgslTextLazyRedecodes = 0;
    let hasBufferPack = false;
    let hasRangedPack = false;
    let pinnedUnion = null;
    const packs = {};
    const packStamps = [];
    let allPacksVersioned = true;
    for (const [name, pack] of this._packs.entries()) {
      if (typeof pack._statsVersion === 'number') {
        packStamps.push({
          pack,
          version: pack._statsVersion,
          decodedCount: typeof pack._getDecodedCount === 'function' ? pack._getDecodedCount() : 0,
        });
      } else {
        allPacksVersioned = false;
      }
      const s = pack.stats();
      totalPackBytes += sanitizeNonNegativeInt(s.byteLength, 0);
      rangeFetches += sanitizeNonNegativeInt(s.rangeFetchCount, 0);
      rangeBytesFetched += sanitizeNonNegativeInt(s.rangeBytesFetched, 0);
      rangeFetchRetries += sanitizeNonNegativeInt(s.rangeFetchRetries, 0);
      staleIndexReloads += sanitizeNonNegativeInt(s.staleIndexReloads, 0);
      activeRangeFetches += sanitizeNonNegativeInt(s.activeRangeFetches, 0);
      const packPeak = sanitizeNonNegativeInt(s.peakConcurrentRangeFetches, 0);
      if (packPeak > peakConcurrentRangeFetches) {
        peakConcurrentRangeFetches = packPeak;
      }
      cacheHits += sanitizeNonNegativeInt(s.cacheHits, 0);
      cacheMisses += sanitizeNonNegativeInt(s.cacheMisses, 0);
      lruHits += sanitizeNonNegativeInt(s.lruHits, 0);
      lruMisses += sanitizeNonNegativeInt(s.lruMisses, 0);
      lruBytes += sanitizeNonNegativeInt(s.lruBytes, 0);
      cachedWgslBytes += sanitizeNonNegativeInt(s.cachedWgslBytes, 0);
      cachedWgslEntries += sanitizeNonNegativeInt(s.cachedWgslEntries, 0);
      pinnedBytes += sanitizeNonNegativeInt(s.pinnedBytes, 0);
      pinnedCachedEntries += sanitizeNonNegativeInt(s.pinnedCachedEntries, 0);
      wgslCacheEvictions += sanitizeNonNegativeInt(s.lruEvictions, 0);
      prefetchBatches += sanitizeNonNegativeInt(s.prefetchBatches, 0);
      prefetchShaders += sanitizeNonNegativeInt(s.prefetchShaders, 0);
      coalescedRangeFetches += sanitizeNonNegativeInt(s.coalescedRangeFetches, 0);
      coalescedEntriesFetched += sanitizeNonNegativeInt(s.coalescedEntriesFetched, 0);
      httpRequestsSavedByCoalescing += sanitizeNonNegativeInt(s.httpRequestsSavedByCoalescing, 0);
      coalescedGapBytes += sanitizeNonNegativeInt(s.coalescedGapBytes, 0);
      inFlightDeduplications += sanitizeNonNegativeInt(s.inFlightCoalesced, 0);
      inFlightCount += sanitizeNonNegativeInt(s.inFlightCount, 0);
      wgslTextBytesRetained += sanitizeNonNegativeInt(s.wgslTextBytesRetained, 0);
      wgslTextEntriesRetained += sanitizeNonNegativeInt(s.wgslTextEntriesRetained, 0);
      wgslTextBytesReleased += sanitizeNonNegativeInt(s.wgslTextBytesReleased, 0);
      wgslTextReleases += sanitizeNonNegativeInt(s.wgslTextReleases, 0);
      wgslTextLazyRedecodes += sanitizeNonNegativeInt(s.wgslTextLazyRedecodes, 0);
      if (pack._pinnedHashes && pack._pinnedHashes.size > 0) {
        if (pinnedUnion === null) {
          pinnedUnion = new Set(this._pinnedHashes);
        }
        for (const key of pack._pinnedHashes) {
          pinnedUnion.add(key);
        }
      }
      if (s.mode === 'ranged') {
        hasRangedPack = true;
      } else {
        hasBufferPack = true;
      }
      packs[name] = s;
    }
    const pinnedCount = sanitizeNonNegativeInt(
      pinnedUnion !== null ? pinnedUnion.size : this._pinnedHashes.size,
      0
    );
    if (activeRangeFetches > peakConcurrentRangeFetches) {
      peakConcurrentRangeFetches = activeRangeFetches;
    }

    const mode =
      hasRangedPack && hasBufferPack
        ? 'mixed'
        : hasRangedPack
          ? 'ranged'
          : 'buffer';

    let entryCounts = this._cachedEntryCounts;
    if (entryCounts === null || this._cachedEntryCountsVersion !== this._entriesVersion) {
      const stageCounts = { vertex: 0, fragment: 0, compute: 0 };
      const dynamicStageCounts = { vertex: 0, fragment: 0, compute: 0 };
      let dynamicOverrideCount = 0;
      let dynamicPayloadBytes = 0;

      for (const [hash, entry] of this._dynamicEntries.entries()) {
        if (this._shaderEntries.has(hash)) {
          dynamicOverrideCount += 1;
        }
        dynamicPayloadBytes += entry.length;
        dynamicStageCounts[entry.stage] += 1;
        stageCounts[entry.stage] += 1;
      }

      for (const [hash, record] of this._shaderEntries.entries()) {
        if (!this._dynamicEntries.has(hash)) {
          stageCounts[record.entry.stage] += 1;
        }
      }

      const staticShaderCount = this._shaderEntries.size;
      const dynamicCount = this._dynamicEntries.size;
      const shaderCount = staticShaderCount + (dynamicCount - dynamicOverrideCount);
      entryCounts = {
        stageCounts,
        dynamicStageCounts,
        dynamicOverrideCount,
        dynamicPayloadBytes,
        staticShaderCount,
        dynamicCount,
        shaderCount,
      };
      this._cachedEntryCounts = entryCounts;
      this._cachedEntryCountsVersion = this._entriesVersion;
    }

    const totalLookups = this._stats.hits + this._stats.misses;
    const computed = {
      mode,
      packCount: this._packs.size,
      shaderCount: entryCounts.shaderCount,
      staticShaderCount: entryCounts.staticShaderCount,
      dynamicCount: entryCounts.dynamicCount,
      dynamicOverrideCount: entryCounts.dynamicOverrideCount,
      dynamicPayloadBytes: entryCounts.dynamicPayloadBytes,
      indexedMetadataCount: this._indexMeta.size,
      cachedModules,
      totalPackBytes,
      hits: this._stats.hits,
      misses: this._stats.misses,
      compiles: this._stats.compiles,
      evictions: this._stats.evictions,
      moduleLruEvictions: this._stats.moduleLruEvictions,
      maxCompiledModules: this._maxCompiledModules,
      releaseSourceAfterCompile: this._releaseSourceAfterCompile,
      wgslTextBytesRetained,
      wgslTextEntriesRetained,
      wgslTextBytesReleased,
      wgslTextReleases,
      wgslTextLazyRedecodes,
      dynamicHits: this._stats.dynamicHits,
      hitRate: totalLookups > 0 ? this._stats.hits / totalLookups : 0,
      rangeFetches,
      rangeFetchCount: rangeFetches,
      rangeBytesFetched,
      rangeFetchRetries,
      staleIndexReloads,
      activeRangeFetches,
      peakConcurrentRangeFetches,
      cacheHits,
      cacheMisses,
      cacheEvictions: wgslCacheEvictions,
      cacheEntries: cachedWgslEntries,
      cacheBytes: cachedWgslBytes,
      pinnedCount,
      pinnedBytes,
      pinnedCachedEntries,
      maxCacheBytes: this._maxCachedWgslBytes,
      maxCacheEntries: this._maxCacheEntries,
      lruHits,
      lruMisses,
      lruBytes,
      cachedWgslBytes,
      cachedWgslEntries,
      wgslCacheEvictions,
      lruEvictions: wgslCacheEvictions,
      prefetchBatches,
      prefetchCoalescedRanges: prefetchBatches,
      prefetchShaders,
      coalescedRangeFetches,
      coalescedEntriesFetched,
      httpRequestsSavedByCoalescing,
      coalescedGapBytes,
      inFlightDeduplications,
      inFlightCoalesced: inFlightDeduplications,
      inFlightCount,
      stageCounts: entryCounts.stageCounts,
      dynamicStageCounts: entryCounts.dynamicStageCounts,
      packs,
    };

    if (allPacksVersioned) {
      this._cachedStats = computed;
      this._cachedStatsVersion = this._statsVersion;
      this._cachedPackStamps = packStamps;
      return this._cloneCachedStats(computed);
    }

    this._cachedStats = null;
    this._cachedStatsVersion = -1;
    this._cachedPackStamps = null;
    return {
      ...computed,
      stageCounts: { ...entryCounts.stageCounts },
      dynamicStageCounts: { ...entryCounts.dynamicStageCounts },
    };
  }
}

/**
 * Alias for `WgslPackCache` (multi-pack WGPK v1/v2 loader, bounded LRU cache,
 * hot-hash pinning, coalesced range prefetch, and WebGPU shader module cache).
 */
export const WgslPackLoader = WgslPackCache;
