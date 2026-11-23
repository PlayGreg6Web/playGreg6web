// Binary WGPK (.wgsl.pack) parser and WebGPU GPUShaderModule cache.
// Pure ES module with zero external dependencies — runs in Web Workers and Node.js.
// Binary layout matches shaders/src/pack.rs and shaders/src/convert.rs byte-for-byte.

/** Binary pack file magic ("WGPK"). */
export const WGPK_MAGIC = 'WGPK';

/** Binary pack magic as raw ASCII bytes. */
export const WGPK_MAGIC_BYTES = Object.freeze([0x57, 0x47, 0x50, 0x4b]);

/** Supported WGPK format version (matches PACK_VERSION in shaders/src/pack.rs). */
export const WGPK_VERSION = 1;

/** Fixed header size in bytes: magic(4) + version(4) + entry_count(4) + flags(4). */
export const WGPK_HEADER_BYTES = 16;

/** Fixed entry table record size in bytes: hash(8) + stage(1) + reserved(3) + offset(4) + length(4). */
export const WGPK_ENTRY_BYTES = 20;

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

/**
 * Serialize shader entries into a binary `.wgsl.pack` `ArrayBuffer` matching
 * `write_pack` / `write_pack_with_layout` in `shaders/src/pack.rs` byte-for-byte.
 *
 * @param {Array<{ hash: bigint | string | number, stage: number | string, wgsl: string | Uint8Array }>} entries
 * @param {{ flags?: number }} [options]
 * @returns {ArrayBuffer}
 */
export function buildWgslPackBuffer(entries, options = {}) {
  if (!Array.isArray(entries)) {
    throw new TypeError('entries must be an array');
  }
  const flags = options.flags === undefined ? 0 : options.flags;
  if (!Number.isInteger(flags) || flags < 0 || flags > U32_MAX) {
    throw new RangeError(`flags must be a uint32 integer, got: ${flags}`);
  }
  if (entries.length > U32_MAX) {
    throw new RangeError('entry count exceeds u32::MAX');
  }

  const encodedEntries = new Array(entries.length);
  const dataStart = WGPK_HEADER_BYTES + entries.length * WGPK_ENTRY_BYTES;
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

  const totalSize = dataStart + totalPayloadBytes;
  if (totalSize > U32_MAX) {
    throw new RangeError('Total WGPK pack size exceeds u32::MAX byte offsets');
  }

  const buffer = new ArrayBuffer(totalSize);
  const out = new Uint8Array(buffer);
  const view = new DataView(buffer);

  // Header (16 bytes)
  out[0] = WGPK_MAGIC_BYTES[0];
  out[1] = WGPK_MAGIC_BYTES[1];
  out[2] = WGPK_MAGIC_BYTES[2];
  out[3] = WGPK_MAGIC_BYTES[3];
  view.setUint32(4, WGPK_VERSION, true);
  view.setUint32(8, encodedEntries.length, true);
  view.setUint32(12, flags >>> 0, true);

  // Entry table (20 bytes each) + payloads
  let cursor = dataStart;
  for (let i = 0; i < encodedEntries.length; i += 1) {
    const { hash, stageCode, payload } = encodedEntries[i];
    const base = WGPK_HEADER_BYTES + i * WGPK_ENTRY_BYTES;

    view.setBigUint64(base, hash, true);
    out[base + 8] = stageCode;
    out[base + 9] = 0;
    out[base + 10] = 0;
    out[base + 11] = 0;
    view.setUint32(base + 12, cursor, true);
    view.setUint32(base + 16, payload.byteLength, true);

    out.set(payload, cursor);
    cursor += payload.byteLength;
  }

  return buffer;
}

/**
 * Parsed read-only view over a single `.wgsl.pack` archive.
 */
export class WgslPack {
  /**
   * @param {object} init
   */
  constructor({
    version,
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
  }) {
    this.version = version;
    this.flags = flags;
    this.byteLength = byteLength;
    this.dataStart = dataStart;
    this.payloadBytes = payloadBytes;
    this.lazyDecode = lazyDecode;
    this.entries = byHash;
    this._byHex = byHex;
    this._entryList = entryList;
    this._stageCounts = stageCounts;
    this._getDecodedCount = getDecodedCount;
  }

  /** Number of shader entries in this pack. */
  get size() {
    return this._entryList.length;
  }

  /**
   * Look up a shader entry by `BigInt` hash, hex string, or safe integer.
   *
   * @param {bigint | string | number} hash
   */
  getByHash(hash) {
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
   * Retrieve the UTF-8 WGSL source string for a shader hash.
   * In `lazyDecode` mode, decodes and caches the string on first access.
   *
   * @param {bigint | string | number} hash
   * @returns {string | undefined}
   */
  getSource(hash) {
    const entry = this.getByHash(hash);
    return entry ? entry.wgsl : undefined;
  }

  /**
   * Check whether a shader hash exists in this pack without triggering lazy UTF-8 decode.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  has(hash) {
    if (typeof hash === 'string') {
      if (this._byHex.has(hash)) {
        return true;
      }
    }
    const key = normalizeHash(hash);
    return this.entries.has(key);
  }

  /**
   * Return all shader entries in their original pack table order.
   * Does not trigger lazy UTF-8 decoding until `.wgsl` is accessed on an entry.
   */
  listEntries() {
    return this._entryList.slice();
  }

  /**
   * Return structural and decode statistics for this pack.
   */
  stats() {
    const entryCount = this._entryList.length;
    return {
      version: this.version,
      flags: this.flags,
      entryCount,
      byteLength: this.byteLength,
      headerBytes: WGPK_HEADER_BYTES,
      tableBytes: entryCount * WGPK_ENTRY_BYTES,
      payloadBytes: this.payloadBytes,
      decodedCount: this._getDecodedCount(),
      lazyDecode: this.lazyDecode,
      stageCounts: { ...this._stageCounts },
    };
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

  if (bytes.byteLength < WGPK_HEADER_BYTES) {
    throw new Error(
      `WGPK pack truncated: ${bytes.byteLength} bytes (minimum required for header: ${WGPK_HEADER_BYTES})`
    );
  }

  // Validate 4-byte magic ("WGPK")
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
  const version = view.getUint32(4, true);
  if (version !== WGPK_VERSION) {
    throw new Error(`Unsupported WGPK pack version ${version} (expected ${WGPK_VERSION})`);
  }

  const entryCount = view.getUint32(8, true);
  const flags = view.getUint32(12, true);

  const tableBytes = entryCount * WGPK_ENTRY_BYTES;
  const dataStart = WGPK_HEADER_BYTES + tableBytes;
  if (bytes.byteLength < dataStart) {
    throw new Error(
      `WGPK pack truncated: ${bytes.byteLength} bytes (header + ${entryCount} entry table records require at least ${dataStart} bytes)`
    );
  }

  const entryList = new Array(entryCount);
  const byHash = new Map();
  const byHex = new Map();
  const ranges = [];
  const stageCounts = { vertex: 0, fragment: 0, compute: 0 };
  let totalPayloadBytes = 0;
  let decodedCount = 0;

  for (let idx = 0; idx < entryCount; idx += 1) {
    const base = WGPK_HEADER_BYTES + idx * WGPK_ENTRY_BYTES;
    const hash = view.getBigUint64(base, true);
    const stageCode = bytes[base + 8];
    const stage = STAGE_NAMES[stageCode];
    if (stage === undefined) {
      throw new Error(`Invalid WGPK shader stage value ${stageCode} at entry ${idx}`);
    }
    // bytes[base + 9 .. base + 12] are reserved for forward compatibility.
    const offset = view.getUint32(base + 12, true);
    const length = view.getUint32(base + 16, true);
    const end = offset + length;

    if (offset < dataStart || end > bytes.byteLength) {
      throw new Error(
        `WGPK entry ${idx} payload [${offset}..+${length}] is out of bounds (valid payload range: [${dataStart}..${bytes.byteLength}])`
      );
    }

    if (length > 0) {
      ranges.push({ start: offset, end, index: idx });
    }

    totalPayloadBytes += length;
    stageCounts[stage] += 1;

    const hashHex = formatHashHex(hash);
    const slice = bytes.subarray(offset, end);

    let entry;
    if (lazyDecode) {
      let cachedWgsl = null;
      entry = {
        hash,
        hashHex,
        stage,
        stageCode,
        offset,
        length,
      };
      Object.defineProperty(entry, 'wgsl', {
        enumerable: true,
        configurable: true,
        get() {
          if (cachedWgsl === null) {
            cachedWgsl = decodeUtf8Strict(slice, idx);
            decodedCount += 1;
          }
          return cachedWgsl;
        },
      });
    } else {
      const wgsl = decodeUtf8Strict(slice, idx);
      decodedCount += 1;
      entry = {
        hash,
        hashHex,
        stage,
        stageCode,
        offset,
        length,
        wgsl,
      };
    }

    entryList[idx] = entry;
    byHash.set(hash, entry);
    byHex.set(hashHex, entry);
  }

  // Reject overlapping non-empty payload ranges across entries.
  if (ranges.length > 1) {
    ranges.sort((a, b) => a.start - b.start || a.end - b.end);
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

  return new WgslPack({
    version,
    flags,
    byteLength: bytes.byteLength,
    dataStart,
    payloadBytes: totalPayloadBytes,
    lazyDecode,
    entryList,
    byHash,
    byHex,
    stageCounts,
    getDecodedCount: () => decodedCount,
  });
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

    this._packs = new Map();
    this._shaderEntries = new Map();
    this._deviceModules = new Map();
    this._indexMeta = new Map();
    this._shaderIndexAbiVersion = null;
    this._shaderIndexPackPath = null;
    this._nextPackId = 0;

    this._stats = {
      hits: 0,
      misses: 0,
      compiles: 0,
      evictions: 0,
    };
  }

  /**
   * Parse and register a `.wgsl.pack` binary buffer.
   * If a pack with the same `packName` is already loaded, its entries are replaced.
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
    const name =
      typeof packName === 'string' && packName.length > 0
        ? packName
        : `pack-${this._nextPackId++}`;

    // If replacing an existing pack with the same name, detach its old entries first.
    if (this._packs.has(name)) {
      for (const [hash, record] of this._shaderEntries.entries()) {
        if (record.packName === name) {
          this._shaderEntries.delete(hash);
          this._evictModuleOnly(hash);
        }
      }
    }

    this._packs.set(name, pack);
    for (const entry of pack.listEntries()) {
      if (this._shaderEntries.has(entry.hash)) {
        this._evictModuleOnly(entry.hash);
      }
      this._shaderEntries.set(entry.hash, { entry, pack, packName: name });
    }

    return pack;
  }

  /**
   * Fetch a `.wgsl.pack` bundle from a URL and register it in the cache.
   *
   * @param {string} url
   * @param {typeof fetch} [fetchImpl]
   * @param {{ packName?: string, lazyDecode?: boolean, fetchInit?: RequestInit }} [options]
   * @returns {Promise<WgslPack>}
   */
  async loadPackUrl(url, fetchImpl, options = {}) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError('url must be a non-empty string');
    }
    const fetcher = fetchImpl ?? this._fetch ?? globalThis.fetch;
    if (typeof fetcher !== 'function') {
      throw new Error('No fetch implementation available for loadPackUrl');
    }

    const response = await fetcher(url, options.fetchInit);
    if (!response || !response.ok) {
      const status = response ? response.status : 'unknown';
      const statusText = response && response.statusText ? ` ${response.statusText}` : '';
      throw new Error(`Failed to fetch WGSL pack from "${url}": HTTP ${status}${statusText}`);
    }

    const buffer = await response.arrayBuffer();
    const name = options.packName ?? url;
    return this.loadPackBuffer(buffer, name, { lazyDecode: options.lazyDecode });
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
   * Return whether any loaded pack contains the given shader hash.
   *
   * @param {bigint | string | number} hash
   * @returns {boolean}
   */
  hasShader(hash) {
    return this._shaderEntries.has(normalizeHash(hash));
  }

  /**
   * Get the WGSL source string for a shader hash from loaded packs.
   *
   * @param {bigint | string | number} hash
   * @returns {string | undefined}
   */
  getShaderSource(hash) {
    const record = this._shaderEntries.get(normalizeHash(hash));
    return record ? record.entry.wgsl : undefined;
  }

  /**
   * Get combined pack + optional `shader-index.json` metadata for a shader hash.
   *
   * @param {bigint | string | number} hash
   */
  getShaderMetadata(hash) {
    const key = normalizeHash(hash);
    const record = this._shaderEntries.get(key);
    const indexMeta = this._indexMeta.get(key);
    if (!record && !indexMeta) {
      return undefined;
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

  /**
   * Return a cached `GPUShaderModule` for `(device, hash)` or compile and cache it.
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
      this._stats.hits += 1;
      return cached;
    }

    this._stats.misses += 1;
    const record = this._shaderEntries.get(key);
    if (!record) {
      throw new Error(`Shader ${formatHashHex(key)} not found in any loaded WGSL pack`);
    }

    const { entry } = record;
    const prefix = labelPrefix !== undefined ? String(labelPrefix) : this._labelPrefix;
    const label = prefix
      ? `${prefix}:${entry.stage}:${entry.hashHex}`
      : `${entry.stage}:${entry.hashHex}`;

    const shaderModule = device.createShaderModule({
      label,
      code: entry.wgsl,
    });

    deviceMap.set(key, shaderModule);
    this._stats.compiles += 1;
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
        ? Array.from(this._shaderEntries.keys())
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
        const mod = this.getOrCreateShaderModule(device, hash, labelPrefix);
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
    }
    return removed > 0;
  }

  /**
   * Evict a compiled `GPUShaderModule` for `hash` from the cache.
   * Pass `{ removeSource: true }` to also drop its loaded pack entry.
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
      evictedSource = this._shaderEntries.delete(key);
    }
    return evictedModule || evictedSource;
  }

  /**
   * Clear all loaded packs, cached shader modules, and index metadata.
   *
   * @param {{ resetStats?: boolean }} [options]
   */
  clear(options = {}) {
    this._packs.clear();
    this._shaderEntries.clear();
    this._deviceModules.clear();
    this._indexMeta.clear();
    this._shaderIndexAbiVersion = null;
    this._shaderIndexPackPath = null;
    if (options && options.resetStats) {
      this._stats.hits = 0;
      this._stats.misses = 0;
      this._stats.compiles = 0;
      this._stats.evictions = 0;
    }
  }

  /**
   * Return cache statistics across all loaded packs and compiled shader modules.
   */
  getStats() {
    let cachedModules = 0;
    for (const map of this._deviceModules.values()) {
      cachedModules += map.size;
    }

    let totalPackBytes = 0;
    const packs = {};
    for (const [name, pack] of this._packs.entries()) {
      const s = pack.stats();
      totalPackBytes += s.byteLength;
      packs[name] = s;
    }

    const totalLookups = this._stats.hits + this._stats.misses;
    return {
      packCount: this._packs.size,
      shaderCount: this._shaderEntries.size,
      indexedMetadataCount: this._indexMeta.size,
      cachedModules,
      totalPackBytes,
      hits: this._stats.hits,
      misses: this._stats.misses,
      compiles: this._stats.compiles,
      evictions: this._stats.evictions,
      hitRate: totalLookups > 0 ? this._stats.hits / totalLookups : 0,
      packs,
    };
  }
}
