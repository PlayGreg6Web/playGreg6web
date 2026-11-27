'use strict';

// Streaming PGIM v1 guest image + boot shader bundle preloader.
// Validates the PGIM header from the first HTTP 206 Range chunk, then streams the rest of the
// container in parallel Range chunks (or a single GET when Range is unsupported) into one
// contiguous ArrayBuffer / SharedArrayBuffer before guest entry.
// The binary layout matches recomp/src/image.rs (authoritative) and runtime/os/image_loader.h.
// Zero external dependencies; UMD export for Browser/Worker (globalThis.PlayGta6PgimPreloader)
// and Node.js (module.exports).

(function (root) {

  /** Container magic ("PGIM"). */
  const PGIM_MAGIC = 'PGIM';
  /** Container magic as raw ASCII bytes. */
  const PGIM_MAGIC_BYTES = Object.freeze([0x50, 0x47, 0x49, 0x4d]);
  /** Container format version (IMAGE_VERSION / pgim::kVersion). */
  const PGIM_VERSION = 1;
  /** Header size in bytes (IMAGE_HEADER_BYTES / pgim::kHeaderBytes). */
  const PGIM_HEADER_BYTES = 64;
  /** Section table entry size in bytes (IMAGE_SECTION_ENTRY_BYTES / pgim::kSectionEntryBytes). */
  const PGIM_SECTION_ENTRY_BYTES = 64;
  /** Payload alignment / guest page size (IMAGE_PAGE_SIZE / pgim::kPayloadAlignment). */
  const PGIM_PAGE_SIZE = 4096;

  /** Header flags (FLAG_* in image.rs, kFlag* in image_loader.h). */
  const PGIM_FLAGS = Object.freeze({
    HEADERS_MAPPED: 1 << 0,
    EXECUTABLE_BYTES_OMITTED: 1 << 1,
    RELOCS_STRIPPED: 1 << 2,
  });

  /** Section flags (SECTION_FLAG_* in image.rs, kSectionFlag* in image_loader.h). */
  const PGIM_SECTION_FLAGS = Object.freeze({
    HEADERS: 1 << 0,
    BYTES_OMITTED: 1 << 1,
  });

  /** Section protection bits (kProt* in image_loader.h). */
  const PGIM_PROT = Object.freeze({ READ: 1 << 0, WRITE: 1 << 1, EXECUTE: 1 << 2 });

  /** S6 reflection manifest schema version accepted by web/workers/reflection_loader.js. */
  const REFLECTION_SCHEMA_VERSION = 1;
  /** Multi-module program manifest schema produced by `playgta6-recomp build-program`. */
  const PROGRAM_MANIFEST_SCHEMA = 'playgta6-recomp-program';
  const PROGRAM_MANIFEST_SCHEMA_VERSION = 1;
  const DEFAULT_BOOT_MANIFEST_FILE = 'boot-manifest.json';
  const WGPK_MAGIC_BYTES = [0x57, 0x47, 0x50, 0x4b];
  const WGPK_VERSION = 1;
  const WGPK_HEADER_BYTES = 16;
  const WGPK_ENTRY_BYTES = 20;

  const SECTION_NAME_BYTES = 16;
  const KNOWN_HEADER_FLAGS = PGIM_FLAGS.HEADERS_MAPPED | PGIM_FLAGS.EXECUTABLE_BYTES_OMITTED | PGIM_FLAGS.RELOCS_STRIPPED;
  const KNOWN_SECTION_FLAGS = PGIM_SECTION_FLAGS.HEADERS | PGIM_SECTION_FLAGS.BYTES_OMITTED;
  const U64_MAX = 0xffffffffffffffffn;
  const BYTES_PER_MIB = 1024 * 1024;
  const DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024;
  const DEFAULT_CHUNK_BYTES = DEFAULT_CHUNK_SIZE;
  const DEFAULT_CONCURRENCY = 4;
  const DEFAULT_MAX_RETRIES = 3;
  const DEFAULT_MAX_VERSION_RETRIES = 2;
  const DEFAULT_BASE_RETRY_DELAY_MS = 50;
  const DEFAULT_MAX_RETRY_DELAY_MS = 30000;
  const PGIM_OPFS_CACHE_VERSION = 1;
  const PGIM_OPFS_PROVENANCE_SCHEMA = 1;
  const DEFAULT_OPFS_CACHE_DIR = 'playgta6-pgim-cache';
  const DEFAULT_OPFS_CACHE_MODE = 'auto';
  /** Explicit `?streamShaders=1` default threshold (matches web/workers/wgpu_worker.js). */
  const DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES = 64 * 1024 * 1024;
  /** Default auto-stream threshold for plain URLs without `?streamShaders` (matches web/workers/wgpu_worker.js). */
  const DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES = 128 * 1024 * 1024;

  function resolveShaderPackPreloadPolicy({ streamShaders, streamThresholdBytes } = {}) {
    const validThreshold =
      typeof streamThresholdBytes === 'number' &&
      Number.isSafeInteger(streamThresholdBytes) &&
      streamThresholdBytes >= 0
        ? streamThresholdBytes
        : undefined;
    if (streamShaders === true) {
      return {
        skipFullPreload: true,
        maxPreloadBytes:
          validThreshold !== undefined && validThreshold > 0
            ? validThreshold
            : DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES,
      };
    }
    if (streamShaders === false) {
      return { skipFullPreload: false, maxPreloadBytes: 0 };
    }
    if (validThreshold !== undefined) {
      return { skipFullPreload: false, maxPreloadBytes: validThreshold };
    }
    return {
      skipFullPreload: false,
      maxPreloadBytes: DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES,
    };
  }

  function defaultSleep(ms, signal) {
    if (signal && signal.aborted) {
      return Promise.reject(abortReason(signal));
    }
    const delayMs = Number.isFinite(ms) && ms > 0 ? ms : 0;
    return new Promise((resolve, reject) => {
      let onAbort = null;
      const timer = setTimeout(() => {
        if (signal && onAbort) signal.removeEventListener('abort', onAbort);
        resolve();
      }, delayMs);
      if (signal) {
        onAbort = () => {
          clearTimeout(timer);
          reject(abortReason(signal));
        };
        signal.addEventListener('abort', onAbort, { once: true });
      }
    });
  }

  function isAbortError(err) {
    if (!err || typeof err !== 'object') return false;
    return err.name === 'AbortError' || err.code === 'ABORT_ERR';
  }

  function isRetryableHttpStatus(status) {
    return (
      status === 408 ||
      status === 425 ||
      status === 429 ||
      (Number.isInteger(status) && status >= 500 && status <= 599)
    );
  }

  function isRetryableNetworkError(err) {
    if (!err || typeof err !== 'object') return false;
    if (isAbortError(err) || err.name === 'TimeoutError') return false;
    if (err instanceof PgimPreloadError) {
      if (err.code === 'NETWORK_ERROR') {
        if (err.cause && (isAbortError(err.cause) || err.cause.name === 'TimeoutError')) return false;
        return true;
      }
      return err.code === 'HTTP_ERROR' && isRetryableHttpStatus(err.status);
    }
    // Fetch network failures typically surface as TypeError or socket error codes.
    return (
      err.name === 'TypeError' ||
      err.code === 'ECONNRESET' ||
      err.code === 'ETIMEDOUT' ||
      err.code === 'ECONNREFUSED' ||
      err.code === 'EPIPE' ||
      err.code === 'EAI_AGAIN'
    );
  }

  function sanitizeNonNegativeFinite(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0;
  }

  function sanitizeNonNegativeInt(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
  }

  function computeBytesPerSec(bytes, elapsedMs) {
    if (!(typeof bytes === 'number' && Number.isFinite(bytes) && bytes > 0)) return 0;
    if (!(typeof elapsedMs === 'number' && Number.isFinite(elapsedMs) && elapsedMs > 0)) return 0;
    const rate = (bytes * 1000) / elapsedMs;
    return Number.isFinite(rate) && rate > 0 ? rate : 0;
  }

  function computeThroughputMBps(bytesPerSec) {
    if (!(typeof bytesPerSec === 'number' && Number.isFinite(bytesPerSec) && bytesPerSec > 0)) return 0;
    const mbps = bytesPerSec / BYTES_PER_MIB;
    return Number.isFinite(mbps) && mbps > 0 ? mbps : 0;
  }

  function computeOpfsHitRatePct(hits, misses) {
    const h = sanitizeNonNegativeInt(hits);
    const m = sanitizeNonNegativeInt(misses);
    const total = h + m;
    if (total <= 0) return 0;
    const pct = (h / total) * 100;
    return Number.isFinite(pct) && pct > 0 ? Math.min(100, pct) : 0;
  }

  function emptyTelemetrySnapshot() {
    return {
      state: 'idle',
      bytesLoaded: 0,
      totalBytes: 0,
      chunksFetched: 0,
      requests: 0,
      durationMs: 0,
      elapsedMs: 0,
      bytesPerSec: 0,
      peakBytesPerSec: 0,
      throughputMBps: 0,
      parallelMode: false,
      retryCount: 0,
      abortedCount: 0,
      opfsEnabled: false,
      opfsHits: 0,
      opfsMisses: 0,
      opfsWrites: 0,
      opfsInvalidations: 0,
      bytesFromOpfs: 0,
      bytesFromNetwork: 0,
      opfsHitRatePct: 0,
      wireBytesReleased: 0,
      droppedWasmBuffers: 0,
    };
  }

  class TelemetryRun {
    constructor(nowFn) {
      this._nowFn = nowFn;
      this.state = 'loading';
      this.startedAt = this._readClock();
      this.lastSampleAt = this.startedAt;
      this.bytesLoaded = 0;
      this.totalBytes = 0;
      this.chunksFetched = 0;
      this.requests = 0;
      this.durationMs = 0;
      this.bytesPerSec = 0;
      this.peakBytesPerSec = 0;
      this.parallelMode = false;
      this.retryCount = 0;
      this.abortedCount = 0;
      this.opfsHits = 0;
      this.opfsMisses = 0;
      this.opfsWrites = 0;
      this.opfsInvalidations = 0;
      this.bytesFromOpfs = 0;
      this.bytesFromNetwork = 0;
      this.wireBytesReleased = 0;
      this.droppedWasmBuffers = 0;
      this._abortRecorded = false;
    }

    _readClock() {
      let ts;
      try {
        ts = this._nowFn();
      } catch {
        return 0;
      }
      return typeof ts === 'number' && Number.isFinite(ts) ? ts : 0;
    }

    _elapsedMs(from, to) {
      const delta = to - from;
      return Number.isFinite(delta) && delta > 0 ? delta : 0;
    }

    recordRequest() {
      this.requests += 1;
    }

    recordChunk(count = 1) {
      const n = sanitizeNonNegativeInt(count);
      if (n > 0) this.chunksFetched += n;
    }

    setParallelMode(enabled) {
      if (enabled) this.parallelMode = true;
    }

    recordRetry() {
      this.retryCount += 1;
    }

    recordOpfsHit(bytes = 0) {
      this.opfsHits += 1;
      const n = sanitizeNonNegativeInt(bytes);
      if (n > 0) this.bytesFromOpfs += n;
    }

    recordOpfsMiss() {
      this.opfsMisses += 1;
    }

    recordOpfsWrite() {
      this.opfsWrites += 1;
    }

    recordOpfsInvalidation() {
      this.opfsInvalidations += 1;
    }

    recordNetworkBytes(deltaBytes) {
      const delta = sanitizeNonNegativeInt(deltaBytes);
      if (delta > 0) this.bytesFromNetwork += delta;
    }

    recordWireBytesReleased(deltaBytes, droppedBuffers = 1) {
      const bytes = sanitizeNonNegativeInt(deltaBytes);
      const count = sanitizeNonNegativeInt(droppedBuffers);
      if (bytes > 0) this.wireBytesReleased += bytes;
      if (count > 0) this.droppedWasmBuffers += count;
    }

    recordAbort() {
      if (!this._abortRecorded) {
        this._abortRecorded = true;
        this.abortedCount += 1;
      }
      this.state = 'aborted';
      const ts = this._readClock();
      this.durationMs = this._elapsedMs(this.startedAt, ts);
      const avg = computeBytesPerSec(this.bytesLoaded, this.durationMs);
      this.bytesPerSec = avg;
      if (avg > this.peakBytesPerSec) this.peakBytesPerSec = avg;
    }

    setTotalBytes(totalBytes) {
      const sanitized = sanitizeNonNegativeInt(totalBytes);
      if (sanitized > 0) this.totalBytes = sanitized;
    }

    recordBytes(deltaBytes) {
      const delta = sanitizeNonNegativeInt(deltaBytes);
      if (delta <= 0) return;
      const ts = this._readClock();
      this.bytesLoaded += delta;
      const totalElapsed = this._elapsedMs(this.startedAt, ts);
      const sampleElapsed = this._elapsedMs(this.lastSampleAt, ts);
      this.durationMs = totalElapsed;
      const avgRate = computeBytesPerSec(this.bytesLoaded, totalElapsed);
      const sampleRate = computeBytesPerSec(delta, sampleElapsed);
      this.bytesPerSec = avgRate;
      const peakCandidate = Math.max(avgRate, sampleRate);
      if (peakCandidate > this.peakBytesPerSec) {
        this.peakBytesPerSec = peakCandidate;
      }
      if (ts >= this.lastSampleAt) {
        this.lastSampleAt = ts;
      }
    }

    complete(totalBytes = null) {
      if (totalBytes !== null && totalBytes !== undefined) {
        const sanitized = sanitizeNonNegativeInt(totalBytes);
        if (sanitized > this.totalBytes) this.totalBytes = sanitized;
      }
      const ts = this._readClock();
      this.durationMs = this._elapsedMs(this.startedAt, ts);
      const avgRate = computeBytesPerSec(this.bytesLoaded, this.durationMs);
      this.bytesPerSec = avgRate;
      if (avgRate > this.peakBytesPerSec) {
        this.peakBytesPerSec = avgRate;
      }
      this.state = 'complete';
    }

    fail() {
      const ts = this._readClock();
      this.durationMs = this._elapsedMs(this.startedAt, ts);
      const avgRate = computeBytesPerSec(this.bytesLoaded, this.durationMs);
      this.bytesPerSec = avgRate;
      if (avgRate > this.peakBytesPerSec) {
        this.peakBytesPerSec = avgRate;
      }
      if (this.state !== 'aborted') {
        this.state = 'error';
      }
    }

    snapshot() {
      const durationMs = sanitizeNonNegativeFinite(this.durationMs);
      const bytesPerSec = sanitizeNonNegativeFinite(this.bytesPerSec);
      const peakBytesPerSec = sanitizeNonNegativeFinite(Math.max(this.peakBytesPerSec, bytesPerSec));
      return {
        state: this.state,
        bytesLoaded: sanitizeNonNegativeInt(this.bytesLoaded),
        totalBytes: sanitizeNonNegativeInt(this.totalBytes),
        chunksFetched: sanitizeNonNegativeInt(this.chunksFetched),
        requests: sanitizeNonNegativeInt(this.requests),
        durationMs,
        elapsedMs: durationMs,
        bytesPerSec,
        peakBytesPerSec,
        throughputMBps: computeThroughputMBps(bytesPerSec),
        parallelMode: Boolean(this.parallelMode),
        retryCount: sanitizeNonNegativeInt(this.retryCount),
        abortedCount: sanitizeNonNegativeInt(this.abortedCount),
        opfsEnabled: false,
        opfsHits: sanitizeNonNegativeInt(this.opfsHits),
        opfsMisses: sanitizeNonNegativeInt(this.opfsMisses),
        opfsWrites: sanitizeNonNegativeInt(this.opfsWrites),
        opfsInvalidations: sanitizeNonNegativeInt(this.opfsInvalidations),
        bytesFromOpfs: sanitizeNonNegativeInt(this.bytesFromOpfs),
        bytesFromNetwork: sanitizeNonNegativeInt(this.bytesFromNetwork),
        opfsHitRatePct: computeOpfsHitRatePct(this.opfsHits, this.opfsMisses),
        wireBytesReleased: sanitizeNonNegativeInt(this.wireBytesReleased),
        droppedWasmBuffers: sanitizeNonNegativeInt(this.droppedWasmBuffers),
      };
    }
  }

  /** Error raised for corrupt/truncated containers and failed transfers. `code` mirrors ImageError. */
  class PgimPreloadError extends Error {
    constructor(code, message, details = {}) {
      super(message, details.cause !== undefined ? { cause: details.cause } : undefined);
      this.name = 'PgimPreloadError';
      this.code = code;
      for (const [key, value] of Object.entries(details)) {
        if (key !== 'cause') this[key] = value;
      }
    }
  }

  function isSharedArrayBuffer(buf) {
    return typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer;
  }

  function isCompiledWasmModule(value) {
    return Boolean(
      value &&
        typeof value === 'object' &&
        typeof WebAssembly !== 'undefined' &&
        typeof WebAssembly.Module === 'function' &&
        value instanceof WebAssembly.Module
    );
  }

  function isTransferableArrayBuffer(value) {
    if (!value || typeof value !== 'object' || isSharedArrayBuffer(value)) return false;
    return (
      value instanceof ArrayBuffer ||
      Object.prototype.toString.call(value) === '[object ArrayBuffer]'
    );
  }

  function getBinaryBufferInfo(value) {
    if (!value || typeof value !== 'object' || isCompiledWasmModule(value)) {
      return null;
    }
    if (value instanceof ArrayBuffer) {
      return { backingBuffer: value, byteLength: value.byteLength };
    }
    if (isSharedArrayBuffer(value)) {
      return { backingBuffer: value, byteLength: value.byteLength };
    }
    if (ArrayBuffer.isView(value)) {
      return { backingBuffer: value.buffer, byteLength: value.byteLength };
    }
    return null;
  }

  /**
   * Collect unique standalone transferable non-SAB `ArrayBuffer`s reachable inside `payload`
   * so Window -> Worker `postMessage(msg, transferList)` transfers large `.wasm` / `.pgim` /
   * `.wgsl.pack` wire buffers zero-copy rather than cloning them.
   *
   * @param {unknown} payload
   * @returns {ArrayBuffer[]}
   */
  function collectTransferableBuffers(payload) {
    const unique = new Set();
    const visited = new Set();

    const visit = (value) => {
      if (!value || typeof value !== 'object' || isCompiledWasmModule(value)) return;
      if (isTransferableArrayBuffer(value)) {
        if (!value.detached && value.byteLength > 0) unique.add(value);
        return;
      }
      if (isSharedArrayBuffer(value)) return;
      if (ArrayBuffer.isView(value)) {
        const buf = value.buffer;
        if (
          isTransferableArrayBuffer(buf) &&
          !buf.detached &&
          value.byteOffset === 0 &&
          value.byteLength === buf.byteLength &&
          buf.byteLength > 0
        ) {
          unique.add(buf);
        }
        return;
      }
      if (visited.has(value)) return;
      visited.add(value);
      if (Array.isArray(value)) {
        for (let i = 0; i < value.length; i += 1) visit(value[i]);
        return;
      }
      if (value instanceof Map || value instanceof Set) {
        for (const entryVal of value.values()) visit(entryVal);
        return;
      }
      const keys = Object.keys(value);
      for (let i = 0; i < keys.length; i += 1) {
        let child;
        try {
          child = value[keys[i]];
        } catch {
          continue;
        }
        visit(child);
      }
    };

    visit(payload);
    return Array.from(unique);
  }

  function isWritableOwnProperty(obj, key) {
    if (!obj || typeof obj !== 'object') return false;
    if (!Object.prototype.hasOwnProperty.call(obj, key)) return false;
    const desc = Object.getOwnPropertyDescriptor(obj, key);
    if (!desc) return false;
    if ('get' in desc || 'set' in desc) {
      return typeof desc.set === 'function';
    }
    return desc.writable === true;
  }

  const BINARY_PAYLOAD_PROP_KEYS = Object.freeze([
    'bytes',
    'buffer',
    'wasmBinary',
    'wireBytes',
    'rawBytes',
    'arrayBuffer',
    'data',
    'wasmBytes',
    'wasmBuffer',
  ]);

  /**
   * Release large binary payload references (`ArrayBuffer`, `SharedArrayBuffer`, or TypedArray views)
   * held on a preloaded boot bundle or asset preload result while preserving lightweight metadata
   * (`bytesLoaded`, `totalBytes`, `entryCount`, `manifest`, `header`, `timingMs`, `durationMs`,
   * `transport`) and compiled `WebAssembly.Module` instances.
   * Defensive and idempotent: never throws on null, primitive, or frozen inputs.
   *
   * @param {unknown} bundle
   * @returns {{ releasedFields: string[], releasedBytes: number }}
   */
  function releaseBootBundleBytes(bundle) {
    const releasedFields = [];
    let releasedBytes = 0;
    if (!bundle || typeof bundle !== 'object') {
      return { releasedFields, releasedBytes };
    }

    const countedByBuffer = new Map();
    const visitedObjects = new Set();

    const recordBufferBytes = (info) => {
      if (!info || !info.backingBuffer) return;
      const len =
        typeof info.byteLength === 'number' && Number.isFinite(info.byteLength) && info.byteLength > 0
          ? info.byteLength
          : 0;
      const prev = countedByBuffer.get(info.backingBuffer) || 0;
      if (len > prev) {
        releasedBytes += len - prev;
        countedByBuffer.set(info.backingBuffer, len);
      }
    };

    const nullBinaryProp = (obj, key, path) => {
      if (!isWritableOwnProperty(obj, key)) return false;
      let val;
      try {
        val = obj[key];
      } catch {
        return false;
      }
      const info = getBinaryBufferInfo(val);
      if (!info) return false;
      try {
        obj[key] = null;
      } catch {
        return false;
      }
      releasedFields.push(path);
      recordBufferBytes(info);
      return true;
    };

    const releasePayloadContainer = (obj, prefix) => {
      if (!obj || typeof obj !== 'object' || isCompiledWasmModule(obj) || visitedObjects.has(obj)) return;
      visitedObjects.add(obj);
      for (let i = 0; i < BINARY_PAYLOAD_PROP_KEYS.length; i += 1) {
        const propKey = BINARY_PAYLOAD_PROP_KEYS[i];
        nullBinaryProp(obj, propKey, `${prefix}.${propKey}`);
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'sideModules')) {
        releaseCollection(obj.sideModules, `${prefix}.sideModules`);
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'modules')) {
        releaseCollection(obj.modules, `${prefix}.modules`);
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'modulesByName')) {
        releaseCollection(obj.modulesByName, `${prefix}.modulesByName`);
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'guestModuleBytes')) {
        releaseCollection(obj.guestModuleBytes, `${prefix}.guestModuleBytes`);
      }
      if (Object.prototype.hasOwnProperty.call(obj, 'preloadedBuffers')) {
        releaseCollection(obj.preloadedBuffers, `${prefix}.preloadedBuffers`);
      }
    };

    const releaseAssetSlot = (parent, key, prefix) => {
      if (!parent || typeof parent !== 'object' || !Object.prototype.hasOwnProperty.call(parent, key)) {
        return;
      }
      let val;
      try {
        val = parent[key];
      } catch {
        return;
      }
      if (!val || typeof val !== 'object' || isCompiledWasmModule(val)) return;
      if (getBinaryBufferInfo(val)) {
        nullBinaryProp(parent, key, prefix);
        return;
      }
      if (val instanceof Map || Array.isArray(val)) {
        releaseCollection(val, prefix);
        return;
      }
      releasePayloadContainer(val, prefix);
    };

    const releaseModuleList = (list, prefix) => {
      if (!Array.isArray(list) || visitedObjects.has(list)) return;
      visitedObjects.add(list);
      let allDirectBinary = list.length > 0;
      for (let i = 0; i < list.length; i += 1) {
        const item = list[i];
        if (!item || typeof item !== 'object' || isCompiledWasmModule(item)) {
          allDirectBinary = false;
          continue;
        }
        if (getBinaryBufferInfo(item)) {
          nullBinaryProp(list, String(i), `${prefix}[${i}]`);
        } else {
          allDirectBinary = false;
          releasePayloadContainer(item, `${prefix}[${i}]`);
        }
      }
      if (allDirectBinary && !Object.isFrozen(list)) {
        try {
          list.length = 0;
        } catch {
          // Ignore non-resizable arrays.
        }
      }
    };

    const releaseBinaryMap = (map, prefix) => {
      if (!(map instanceof Map) || visitedObjects.has(map)) return;
      visitedObjects.add(map);
      for (const [k, val] of Array.from(map.entries())) {
        if (!val || typeof val !== 'object' || isCompiledWasmModule(val)) continue;
        const info = getBinaryBufferInfo(val);
        const keyLabel = typeof k === 'string' || typeof k === 'number' ? String(k) : 'entry';
        if (info) {
          map.delete(k);
          releasedFields.push(`${prefix}.${keyLabel}`);
          recordBufferBytes(info);
        } else {
          releasePayloadContainer(val, `${prefix}.${keyLabel}`);
        }
      }
    };

    const releaseModuleMap = (map, prefix) => {
      if (!map || typeof map !== 'object' || Array.isArray(map) || visitedObjects.has(map)) return;
      if (map instanceof Map) {
        releaseBinaryMap(map, prefix);
        return;
      }
      visitedObjects.add(map);
      for (const key of Object.keys(map)) {
        releaseAssetSlot(map, key, `${prefix}.${key}`);
      }
    };

    const releaseCollection = (col, prefix) => {
      if (!col || typeof col !== 'object') return;
      if (Array.isArray(col)) {
        releaseModuleList(col, prefix);
      } else if (col instanceof Map) {
        releaseBinaryMap(col, prefix);
      } else {
        releaseModuleMap(col, prefix);
      }
    };

    visitedObjects.add(bundle);

    nullBinaryProp(bundle, 'buffer', 'buffer');
    nullBinaryProp(bundle, 'bytes', 'bytes');
    nullBinaryProp(bundle, 'pgimBytes', 'pgimBytes');
    nullBinaryProp(bundle, 'pgimBuffer', 'pgimBuffer');
    nullBinaryProp(bundle, 'shaderPackBuffer', 'shaderPackBuffer');
    nullBinaryProp(bundle, 'wasmBinary', 'wasmBinary');
    nullBinaryProp(bundle, 'wasmBytes', 'wasmBytes');
    nullBinaryProp(bundle, 'wasmBuffer', 'wasmBuffer');
    nullBinaryProp(bundle, 'wireBytes', 'wireBytes');
    nullBinaryProp(bundle, 'rawBytes', 'rawBytes');
    nullBinaryProp(bundle, 'arrayBuffer', 'arrayBuffer');

    releaseAssetSlot(bundle, 'shaderPackBytes', 'shaderPackBytes');
    releaseAssetSlot(bundle, 'pgim', 'pgim');
    releaseAssetSlot(bundle, 'shaderPackIndex', 'shaderPackIndex');
    releaseAssetSlot(bundle, 'reflectionManifest', 'reflectionManifest');
    releaseAssetSlot(bundle, 'reflectionSidecar', 'reflectionSidecar');
    releaseAssetSlot(bundle, 'runtimeWasm', 'runtimeWasm');
    releaseAssetSlot(bundle, 'opfsWarmBoot', 'opfsWarmBoot');

    if (Object.prototype.hasOwnProperty.call(bundle, 'modules')) {
      releaseCollection(bundle.modules, 'modules');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'modulesByName')) {
      releaseCollection(bundle.modulesByName, 'modulesByName');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'sideModules')) {
      releaseCollection(bundle.sideModules, 'sideModules');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'sideModuleBytes')) {
      releaseCollection(bundle.sideModuleBytes, 'sideModuleBytes');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'guestModules')) {
      releaseCollection(bundle.guestModules, 'guestModules');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'guestModuleBytes')) {
      releaseCollection(bundle.guestModuleBytes, 'guestModuleBytes');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'wasmModules')) {
      releaseCollection(bundle.wasmModules, 'wasmModules');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'preloadedBuffers')) {
      releaseCollection(bundle.preloadedBuffers, 'preloadedBuffers');
    }
    if (Object.prototype.hasOwnProperty.call(bundle, 'mainModule')) {
      releaseAssetSlot(bundle, 'mainModule', 'mainModule');
    }
    if (
      Object.prototype.hasOwnProperty.call(bundle, 'program') &&
      bundle.program &&
      typeof bundle.program === 'object' &&
      !visitedObjects.has(bundle.program)
    ) {
      const program = bundle.program;
      visitedObjects.add(program);
      for (let i = 0; i < BINARY_PAYLOAD_PROP_KEYS.length; i += 1) {
        const propKey = BINARY_PAYLOAD_PROP_KEYS[i];
        nullBinaryProp(program, propKey, `program.${propKey}`);
      }
      if (Object.prototype.hasOwnProperty.call(program, 'mainModule')) {
        releaseAssetSlot(program, 'mainModule', 'program.mainModule');
      }
      if (Object.prototype.hasOwnProperty.call(program, 'modules')) {
        releaseCollection(program.modules, 'program.modules');
      }
      if (Object.prototype.hasOwnProperty.call(program, 'modulesByName')) {
        releaseCollection(program.modulesByName, 'program.modulesByName');
      }
    }

    if (
      Object.prototype.hasOwnProperty.call(bundle, 'sidecars') &&
      bundle.sidecars &&
      typeof bundle.sidecars === 'object' &&
      !Array.isArray(bundle.sidecars) &&
      !visitedObjects.has(bundle.sidecars)
    ) {
      const sidecars = bundle.sidecars;
      visitedObjects.add(sidecars);
      for (const key of Object.keys(sidecars)) {
        releaseAssetSlot(sidecars, key, `sidecars.${key}`);
      }
    }

    return { releasedFields, releasedBytes };
  }

  function toBytes(input, label = 'buffer') {
    if (input instanceof Uint8Array) return input;
    if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    if (input instanceof ArrayBuffer || isSharedArrayBuffer(input)) return new Uint8Array(input);
    throw new TypeError(`Expected ${label} to be an ArrayBuffer, SharedArrayBuffer, or ArrayBufferView`);
  }

  function hexBytes(bytes) {
    return Array.from(bytes, (b) => `0x${b.toString(16).padStart(2, '0')}`).join(' ');
  }

  function now() {
    return typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now();
  }

  /**
   * FNV-1a 64 of `input` (matches `fnv1a_64` in recomp/src/image.rs). Uses 16-bit limbs so large
   * payloads hash without per-byte BigInt arithmetic.
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} input
   * @returns {bigint}
   */
  function pgimFnv1a64(input) {
    const bytes = toBytes(input, 'input');
    let h0 = 0x2325;
    let h1 = 0x8422;
    let h2 = 0x9ce4;
    let h3 = 0xcbf2;
    for (let i = 0; i < bytes.length; i += 1) {
      h0 ^= bytes[i];
      // prime = 2^40 + 0x1b3: limb i contributes 0x1b3 to limb i and 0x100 to limb i + 2.
      const t0 = h0 * 0x1b3;
      let t1 = h1 * 0x1b3 + (t0 >>> 16);
      let t2 = h2 * 0x1b3 + h0 * 0x100;
      let t3 = h3 * 0x1b3 + h1 * 0x100;
      h0 = t0 & 0xffff;
      h1 = t1 & 0xffff;
      t2 += t1 >>> 16;
      h2 = t2 & 0xffff;
      t3 += t2 >>> 16;
      h3 = t3 & 0xffff;
    }
    return (BigInt(h3) << 48n) | (BigInt(h2) << 32n) | (BigInt(h1) << 16n) | BigInt(h0);
  }

  /**
   * Parse and strictly validate the 64-byte PGIM v1 header (same checks and order as `read_image`).
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} input At least the first 64 bytes.
   * @param {{ totalBytes?: number | null }} [options] Transport size (Content-Length / Content-Range
   *   total / buffer length); when given, the header `file_size` must equal it.
   */
  function parsePgimHeader(input, options = {}) {
    const bytes = toBytes(input, 'PGIM header');
    if (bytes.byteLength < PGIM_HEADER_BYTES) {
      throw new PgimPreloadError(
        'TRUNCATED',
        `truncated PGIM image header: needed ${PGIM_HEADER_BYTES} bytes, only ${bytes.byteLength} available`,
        { needed: PGIM_HEADER_BYTES, available: bytes.byteLength }
      );
    }
    for (let i = 0; i < 4; i += 1) {
      if (bytes[i] !== PGIM_MAGIC_BYTES[i]) {
        throw new PgimPreloadError(
          'BAD_MAGIC',
          `invalid PGIM image magic [${hexBytes(bytes.subarray(0, 4))}] (expected "${PGIM_MAGIC}")`
        );
      }
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, PGIM_HEADER_BYTES);
    const version = view.getUint32(4, true);
    if (version !== PGIM_VERSION) {
      throw new PgimPreloadError('UNSUPPORTED_VERSION', `unsupported PGIM image version ${version} (expected ${PGIM_VERSION})`, {
        version,
      });
    }
    const badField = (field, value, extra = '') =>
      new PgimPreloadError('BAD_HEADER_FIELD', `invalid PGIM header field ${field} = 0x${value.toString(16)}${extra}`, {
        field,
        value,
      });
    const expectU32 = (at, field, expected) => {
      const value = view.getUint32(at, true);
      if (value !== expected) throw badField(field, value);
    };
    expectU32(8, 'header_bytes', PGIM_HEADER_BYTES);
    expectU32(12, 'section_entry_bytes', PGIM_SECTION_ENTRY_BYTES);
    expectU32(20, 'page_size', PGIM_PAGE_SIZE);
    const flags = view.getUint32(16, true);
    if ((flags & ~KNOWN_HEADER_FLAGS) !== 0) throw badField('flags', flags);

    const imageBase = view.getBigUint64(24, true);
    const sizeOfImage = view.getUint32(32, true);
    const sectionAlignment = view.getUint32(36, true);
    const entryPointRva = view.getUint32(40, true);
    const sectionCount = view.getUint32(44, true);
    const dir64Count = view.getUint32(48, true);
    const highlowCount = view.getUint32(52, true);
    const fileSizeBig = view.getBigUint64(56, true);

    if (imageBase + BigInt(sizeOfImage) > U64_MAX) throw badField('image_base + size_of_image', imageBase);
    if (fileSizeBig > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw badField('file_size', fileSizeBig, ' (exceeds the JavaScript addressable range)');
    }
    const fileSize = Number(fileSizeBig);
    const { totalBytes = null } = options;
    if (totalBytes !== null && totalBytes !== undefined) {
      if (!Number.isSafeInteger(totalBytes) || totalBytes < 0) {
        throw new TypeError(`totalBytes must be a non-negative safe integer, got: ${totalBytes}`);
      }
      if (fileSize !== totalBytes) {
        throw badField('file_size', fileSizeBig, ` (transport reports ${totalBytes} bytes)`);
      }
    }

    // All counts are u32, so these stay well inside Number.MAX_SAFE_INTEGER.
    const relocTableOffset = PGIM_HEADER_BYTES + sectionCount * PGIM_SECTION_ENTRY_BYTES;
    const tablesEnd = relocTableOffset + (dir64Count + highlowCount) * 4;
    if (tablesEnd > fileSize) {
      throw new PgimPreloadError(
        'TRUNCATED',
        `truncated PGIM section and relocation tables: needed ${tablesEnd} bytes, only ${fileSize} available`,
        { needed: tablesEnd, available: fileSize }
      );
    }

    return {
      magic: PGIM_MAGIC,
      version,
      flags,
      headersMapped: (flags & PGIM_FLAGS.HEADERS_MAPPED) !== 0,
      executableBytesOmitted: (flags & PGIM_FLAGS.EXECUTABLE_BYTES_OMITTED) !== 0,
      relocsStripped: (flags & PGIM_FLAGS.RELOCS_STRIPPED) !== 0,
      imageBase,
      sizeOfImage,
      sectionAlignment,
      entryPointRva,
      entryPoint: entryPointRva === 0 ? 0n : imageBase + BigInt(entryPointRva),
      sectionCount,
      dir64Count,
      highlowCount,
      relocCount: dir64Count + highlowCount,
      fileSize,
      sectionTableOffset: PGIM_HEADER_BYTES,
      relocTableOffset,
      tablesEnd,
    };
  }

  const nameDecoder = new TextDecoder('utf-8', { fatal: true });


  /**
   * Parse and validate the header, section table, and relocation table of a complete container
   * (every structural check of `read_image` + `GuestImage::validate`). Payload checksums are only
   * verified with `verifyChecksums` (the C++ loader verifies them while mapping anyway).
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} input Whole container.
   * @param {{ verifyChecksums?: boolean }} [options]
   */
  function parsePgimLayout(input, options = {}) {
    const bytes = toBytes(input, 'PGIM image');
    const header = parsePgimHeader(bytes, { totalBytes: bytes.byteLength });
    const verifyChecksums = Boolean(options.verifyChecksums);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const fileLen = bytes.byteLength;
    const invalid = (index, reason) =>
      new PgimPreloadError('INVALID_SECTION', `PGIM section ${index}: ${reason}`, { index, reason });

    const sections = new Array(header.sectionCount);
    const payloads = [];
    for (let index = 0; index < header.sectionCount; index += 1) {
      const at = PGIM_HEADER_BYTES + index * PGIM_SECTION_ENTRY_BYTES;
      const rawName = bytes.subarray(at, at + SECTION_NAME_BYTES);
      let nameLen = rawName.indexOf(0);
      if (nameLen < 0) nameLen = SECTION_NAME_BYTES;
      for (let i = nameLen; i < SECTION_NAME_BYTES; i += 1) {
        if (rawName[i] !== 0) throw invalid(index, 'name padding is not zero');
      }
      let name;
      try {
        name = nameDecoder.decode(rawName.slice(0, nameLen));
      } catch {
        throw invalid(index, 'name is not UTF-8');
      }
      const protection = view.getUint32(at + 24, true);
      if ((protection & ~0b111) !== 0) throw invalid(index, 'unknown protection bits');
      const sectionFlags = view.getUint32(at + 28, true);
      if ((sectionFlags & ~KNOWN_SECTION_FLAGS) !== 0) throw invalid(index, 'unknown section flags');
      if (view.getUint32(at + 60, true) !== 0) throw invalid(index, 'reserved field is not zero');
      const offsetBig = view.getBigUint64(at + 32, true);
      const lengthBig = view.getBigUint64(at + 40, true);
      const checksum = view.getBigUint64(at + 48, true);

      let payloadOffset = 0;
      let payloadLength = 0;
      if (lengthBig === 0n) {
        if (offsetBig !== 0n) throw invalid(index, 'empty payload with non-zero offset');
      } else {
        if (offsetBig % BigInt(PGIM_PAGE_SIZE) !== 0n) {
          throw new PgimPreloadError(
            'PAYLOAD_MISALIGNED',
            `PGIM section ${index} payload offset 0x${offsetBig.toString(16)} is not ${PGIM_PAGE_SIZE}-byte aligned`,
            { index, offset: offsetBig }
          );
        }
        const end = offsetBig + lengthBig;
        if (end > BigInt(fileLen) || offsetBig < BigInt(header.tablesEnd)) {
          throw new PgimPreloadError(
            'PAYLOAD_OUT_OF_RANGE',
            `PGIM section ${index} payload [0x${offsetBig.toString(16)}..+0x${lengthBig.toString(16)}] is outside the container (0x${fileLen.toString(16)} bytes) or its tables`,
            { index, offset: offsetBig, length: lengthBig }
          );
        }
        payloadOffset = Number(offsetBig);
        payloadLength = Number(lengthBig);
        payloads.push([payloadOffset, payloadOffset + payloadLength, index]);
      }
      if (verifyChecksums) {
        const actual = pgimFnv1a64(bytes.subarray(payloadOffset, payloadOffset + payloadLength));
        if (actual !== checksum) {
          throw new PgimPreloadError(
            'CHECKSUM_MISMATCH',
            `PGIM section ${index} checksum mismatch: stored 0x${checksum.toString(16).padStart(16, '0')}, computed 0x${actual.toString(16).padStart(16, '0')}`,
            { index, expected: checksum, actual }
          );
        }
      }
      sections[index] = {
        name,
        rva: view.getUint32(at + 16, true),
        virtualSize: view.getUint32(at + 20, true),
        protection,
        flags: sectionFlags,
        isHeaders: (sectionFlags & PGIM_SECTION_FLAGS.HEADERS) !== 0,
        bytesOmitted: (sectionFlags & PGIM_SECTION_FLAGS.BYTES_OMITTED) !== 0,
        payloadOffset,
        payloadLength,
        checksum,
        characteristics: view.getUint32(at + 56, true),
      };
    }

    payloads.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    for (let i = 1; i < payloads.length; i += 1) {
      if (payloads[i - 1][1] > payloads[i][0]) {
        throw new PgimPreloadError('PAYLOAD_OVERLAP', `PGIM section ${payloads[i][2]} payload overlaps another payload`, {
          index: payloads[i][2],
        });
      }
    }

    const readRvas = (start, count) => {
      const out = new Uint32Array(count);
      for (let i = 0; i < count; i += 1) {
        const rva = view.getUint32(start + i * 4, true);
        if (i > 0 && out[i - 1] >= rva) {
          throw new PgimPreloadError('RELOCATIONS_UNSORTED', `PGIM relocation table is not strictly ascending at 0x${rva.toString(16)}`, {
            rva,
          });
        }
        out[i] = rva;
      }
      return out;
    };
    const dir64 = readRvas(header.relocTableOffset, header.dir64Count);
    const highlow = readRvas(header.relocTableOffset + header.dir64Count * 4, header.highlowCount);

    // GuestImage::validate
    const firstIsHeaders = sections.length > 0 && sections[0].isHeaders;
    if (header.headersMapped !== firstIsHeaders) throw invalid(0, 'headers flag does not match the first section');
    const omit = header.executableBytesOmitted;
    for (let index = 0; index < sections.length; index += 1) {
      const s = sections[index];
      const execute = (s.protection & PGIM_PROT.EXECUTE) !== 0;
      if (s.virtualSize === 0) throw invalid(index, 'zero in-memory size');
      if (s.payloadLength > s.virtualSize) throw invalid(index, 'initialised bytes exceed in-memory size');
      if (s.isHeaders && (index !== 0 || s.rva !== 0)) throw invalid(index, 'headers pseudo-section must be first and at RVA 0');
      if (s.bytesOmitted && !(omit && execute && s.payloadLength === 0)) {
        throw invalid(index, 'omitted bytes only allowed for empty executable sections');
      }
      if (omit && execute && !s.isHeaders && !s.bytesOmitted) throw invalid(index, 'executable section bytes must be omitted');
      if (s.rva + s.virtualSize > header.sizeOfImage) {
        throw new PgimPreloadError(
          'SECTION_OUT_OF_IMAGE',
          `PGIM section ${s.name} [0x${s.rva.toString(16)}..+0x${s.virtualSize.toString(16)}] exceeds SizeOfImage 0x${header.sizeOfImage.toString(16)}`,
          { index }
        );
      }
      if (index > 0) {
        const prev = sections[index - 1];
        if (prev.rva + prev.virtualSize > s.rva) {
          throw new PgimPreloadError('OVERLAPPING_SECTIONS', `PGIM sections ${prev.name} and ${s.name} overlap`, { index });
        }
      }
    }

    // Merge both lists in (rva, kind) order, as validate() walks the combined sorted table.
    let i = 0;
    let j = 0;
    let prevRva = -1;
    let prevEnd = 0;
    let sectionIndex = 0;
    while (i < dir64.length || j < highlow.length) {
      let rva;
      let width;
      if (j >= highlow.length || (i < dir64.length && dir64[i] < highlow[j])) {
        rva = dir64[i++];
        width = 8;
      } else {
        rva = highlow[j++];
        width = 4;
      }
      if (prevRva >= rva) {
        throw new PgimPreloadError('RELOCATIONS_UNSORTED', `PGIM relocation table is not strictly ascending at 0x${rva.toString(16)}`, {
          rva,
        });
      }
      if (prevEnd > rva) {
        throw new PgimPreloadError(
          'OVERLAPPING_RELOCATIONS',
          `PGIM relocation sites 0x${prevRva.toString(16)} and 0x${rva.toString(16)} overlap`,
          { first: prevRva, second: rva }
        );
      }
      const end = rva + width;
      if (end > header.sizeOfImage) {
        throw new PgimPreloadError(
          'RELOC_SITE_OUT_OF_RANGE',
          `PGIM relocation site 0x${rva.toString(16)} (+${width}) is outside 0x${header.sizeOfImage.toString(16)}`,
          { rva }
        );
      }
      // Relocations arrive in increasing order and sections cannot overlap. Keep
      // the containing-section cursor rather than searching the same table twice
      // for every relocation (large images can have millions of sites).
      while (sectionIndex < sections.length &&
             sections[sectionIndex].rva + sections[sectionIndex].virtualSize <= rva) {
        sectionIndex += 1;
      }
      const candidate = sections[sectionIndex];
      const first = candidate && candidate.rva <= rva ? candidate : null;
      let lastIndex = sectionIndex;
      while (lastIndex + 1 < sections.length && sections[lastIndex + 1].rva <= end - 1) {
        lastIndex += 1;
      }
      const lastCandidate = sections[lastIndex];
      const last = lastCandidate && lastCandidate.rva <= end - 1 &&
        end - 1 < lastCandidate.rva + lastCandidate.virtualSize ? lastCandidate : null;
      const mapped =
        first !== null &&
        last !== null &&
        !first.bytesOmitted &&
        !last.bytesOmitted &&
        (first === last || first.rva + first.virtualSize === last.rva);
      if (!mapped) {
        throw new PgimPreloadError(
          'RELOC_SITE_UNMAPPED',
          `PGIM relocation site 0x${rva.toString(16)} is not inside a mapped section with stored bytes`,
          { rva }
        );
      }
      prevRva = rva;
      prevEnd = end;
    }

    return { header, sections, dir64, highlow };
  }

  function abortReason(signal) {
    if (signal && signal.reason !== undefined) return signal.reason;
    return new DOMException('The operation was aborted.', 'AbortError');
  }

  function throwIfAborted(signal) {
    if (signal && signal.aborted) throw abortReason(signal);
  }

  function raceAbort(promise, signal) {
    if (!signal) return promise;
    if (signal.aborted) {
      Promise.resolve(promise).catch(() => {});
      return Promise.reject(abortReason(signal));
    }
    return new Promise((resolve, reject) => {
      const onAbort = () => reject(abortReason(signal));
      signal.addEventListener('abort', onAbort, { once: true });
      Promise.resolve(promise).then(
        (value) => {
          signal.removeEventListener('abort', onAbort);
          resolve(value);
        },
        (err) => {
          signal.removeEventListener('abort', onAbort);
          reject(err);
        }
      );
    });
  }

  /** Child controller that aborts when `outer` does; returns `{ controller, unlink }`. */
  function linkSignal(outer) {
    const controller = new AbortController();
    if (!outer) return { controller, unlink: () => {} };
    if (outer.aborted) {
      controller.abort(abortReason(outer));
      return { controller, unlink: () => {} };
    }
    const onAbort = () => controller.abort(abortReason(outer));
    outer.addEventListener('abort', onAbort, { once: true });
    return { controller, unlink: () => outer.removeEventListener('abort', onAbort) };
  }

  function headerValue(response, name) {
    const headers = response && response.headers;
    if (!headers || typeof headers.get !== 'function') return null;
    const value = headers.get(name);
    return value === undefined ? null : value;
  }

  function contentLength(response) {
    const encoding = headerValue(response, 'content-encoding');
    if (encoding && encoding.trim().toLowerCase() !== 'identity') return null;
    const raw = headerValue(response, 'content-length');
    if (raw === null || !/^\s*\d+\s*$/.test(raw)) return null;
    const value = Number(raw);
    return Number.isSafeInteger(value) ? value : null;
  }

  function parseContentRange(raw) {
    if (typeof raw !== 'string') return null;
    const match = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i.exec(raw.trim());
    if (!match) return null;
    const start = Number(match[1]);
    const end = Number(match[2]);
    if (end < start) return null;
    return { start, end, total: match[3] === '*' ? null : Number(match[3]) };
  }

  function makeEmitter(onProgress, extra = null) {
    if (typeof onProgress !== 'function') return () => {};
    let maxLoaded = 0;
    let maxRatio = 0;
    return (phase, loadedBytes, totalBytes, eventExtra = null) => {
      const safeLoaded = Math.max(maxLoaded, sanitizeNonNegativeInt(loadedBytes));
      maxLoaded = safeLoaded;
      const safeTotal =
        totalBytes !== null && totalBytes !== undefined && Number.isFinite(totalBytes) && totalBytes >= 0
          ? Math.floor(totalBytes)
          : null;
      const rawRatio = safeTotal && safeTotal > 0 ? Math.min(1, Math.max(0, safeLoaded / safeTotal)) : 0;
      const ratio = safeTotal && safeTotal > 0 ? Math.max(maxRatio, rawRatio) : maxRatio;
      if (safeTotal && safeTotal > 0) maxRatio = ratio;
      const pct = Math.min(100, ratio * 100);
      const event = { phase, loadedBytes: safeLoaded, totalBytes: safeTotal, ratio, pct };
      if (extra && eventExtra) {
        onProgress({ ...event, ...extra, ...eventExtra });
      } else if (extra) {
        onProgress({ ...event, ...extra });
      } else if (eventExtra) {
        onProgress({ ...event, ...eventExtra });
      } else {
        onProgress(event);
      }
    };
  }

  function isQuotaExceededError(err) {
    if (!err || typeof err !== 'object') return false;
    return (
      err.name === 'QuotaExceededError' ||
      err.code === 22 ||
      err.code === 1014 ||
      /quota/i.test(String(err.message || ''))
    );
  }

  function isNotFoundError(err) {
    if (!err || typeof err !== 'object') return false;
    return (
      err.name === 'NotFoundError' ||
      err.code === 'ENOENT' ||
      err.code === 8 ||
      /not\s*found/i.test(String(err.message || ''))
    );
  }

  function normalizeEtag(rawEtag) {
    if (typeof rawEtag !== 'string') return null;
    const cleaned = rawEtag.replace(/[\x00-\x1f\x7f]/g, '').trim();
    return cleaned.length > 0 ? cleaned : null;
  }

  /**
   * Return true when `rawEtag` is a weak HTTP entity-tag (`W/"..."` or `w/"..."`).
   *
   * @param {unknown} rawEtag
   * @returns {boolean}
   */
  function isWeakEtag(rawEtag) {
    const normalized = normalizeEtag(rawEtag);
    if (!normalized) return false;
    return /^(?:W|w)\//.test(normalized);
  }

  /**
   * Return true when `rawEtag` is a non-empty quoted strong HTTP entity-tag (`"..."`)
   * without a weak `W/` prefix (RFC 9110 §8.8.3).
   *
   * @param {unknown} rawEtag
   * @returns {boolean}
   */
  function isStrongEtag(rawEtag) {
    const normalized = normalizeEtag(rawEtag);
    if (!normalized) return false;
    if (/^(?:W|w)\//.test(normalized)) return false;
    return (
      normalized.length >= 3 &&
      normalized.startsWith('"') &&
      normalized.endsWith('"') &&
      normalized.indexOf('"', 1) === normalized.length - 1
    );
  }

  function validatorStrengthForEtag(rawEtag) {
    if (isStrongEtag(rawEtag)) return 'strong';
    if (isWeakEtag(rawEtag)) return 'weak';
    return 'none';
  }

  function hasVerifiedOpfsProvenance(meta) {
    return Boolean(
      meta &&
        isPlainObject(meta.provenance) &&
        meta.provenance.schema === PGIM_OPFS_PROVENANCE_SCHEMA &&
        meta.provenance.verified === true &&
        typeof meta.provenance.source === 'string' &&
        meta.provenance.source.length > 0
    );
  }

  function normalizeLastModified(rawLastModified) {
    if (typeof rawLastModified !== 'string') return null;
    const cleaned = rawLastModified.replace(/[\x00-\x1f\x7f]/g, '').trim();
    return cleaned.length > 0 ? cleaned : null;
  }

  function fnv1a64HexString(str) {
    const bytes = new TextEncoder().encode(String(str));
    return pgimFnv1a64(bytes).toString(16).padStart(16, '0');
  }

  /**
   * Sanitize a PGIM URL or path into a canonical relative `urlPath` with origin, credentials,
   * query strings, fragments, backslashes, and traversal segments stripped.
   *
   * @param {string} rawUrl
   * @returns {string}
   */
  function sanitizePgimCacheUrlPath(rawUrl) {
    if (typeof rawUrl !== 'string' || rawUrl.trim().length === 0) {
      throw new TypeError('PGIM cache url must be a non-empty string');
    }
    let candidate = rawUrl.trim().replace(/[\x00-\x1f\x7f]/g, '');
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(candidate)) {
      try {
        candidate = new URL(candidate).pathname;
      } catch {
        candidate = candidate.split(/[?#]/, 1)[0];
      }
    } else {
      candidate = candidate.split(/[?#]/, 1)[0];
    }
    candidate = candidate.replace(/\\/g, '/');
    const rawSegments = candidate.split('/');
    const safeSegments = [];
    for (let i = 0; i < rawSegments.length; i += 1) {
      const seg = rawSegments[i];
      if (!seg || seg === '.') continue;
      if (seg === '..') {
        if (safeSegments.length > 0) safeSegments.pop();
        continue;
      }
      safeSegments.push(seg);
    }
    return safeSegments.length > 0 ? safeSegments.join('/') : 'guest.pgim';
  }

  function normalizeOpfsCacheVersion(version) {
    if (version === undefined || version === null) return PGIM_OPFS_CACHE_VERSION;
    if (typeof version === 'number') {
      if (!Number.isSafeInteger(version) || version < 1) {
        throw new RangeError(`opfsCacheVersion must be a positive integer or non-empty token, got: ${version}`);
      }
      return version;
    }
    if (typeof version === 'string') {
      const trimmed = version.trim();
      if (!/^[A-Za-z0-9._-]{1,32}$/.test(trimmed)) {
        throw new RangeError(`opfsCacheVersion contains unsupported characters: ${version}`);
      }
      return trimmed;
    }
    throw new TypeError('opfsCacheVersion must be a positive integer or non-empty string');
  }

  /**
   * Build a deterministic, OPFS-filename-safe versioned fingerprint and cache keys from
   * `(urlPath, etag, byteLength)`.
   *
   * @param {string} url
   * @param {string | null} [etag=null]
   * @param {number | null} [byteLength=null]
   * @param {number | string} [version=PGIM_OPFS_CACHE_VERSION]
   */
  function buildPgimOpfsFingerprint(url, etag = null, byteLength = null, version = PGIM_OPFS_CACHE_VERSION) {
    const ver = normalizeOpfsCacheVersion(version);
    const urlPath = sanitizePgimCacheUrlPath(url);
    const normEtag = normalizeEtag(etag);
    const validSize = Number.isSafeInteger(byteLength) && byteLength >= 0 ? byteLength : null;
    const pathHash = fnv1a64HexString(urlPath);
    const etagHash = normEtag !== null ? fnv1a64HexString(normEtag) : 'noetag';
    const sizeToken = validSize !== null ? String(validSize) : 'nosize';
    const leafRaw = urlPath.includes('/') ? urlPath.slice(urlPath.lastIndexOf('/') + 1) : urlPath;
    const stemRaw = leafRaw.replace(/\.pgim$/i, '');
    const safeStem = (stemRaw.replace(/[^A-Za-z0-9_-]/g, '_').replace(/^_+|_+$/g, '') || 'pgim').slice(0, 28);
    const versionTag = `v${ver}`;
    const fingerprint = `${versionTag}:${pathHash}:${sizeToken}:${etagHash}`;
    const dataKey = `pgim-${versionTag}-${pathHash}-${safeStem}-${sizeToken}-${etagHash}.pgim`;
    const metaKey = `pgim-${versionTag}-${pathHash}-${safeStem}.meta.json`;
    return {
      version: ver,
      urlPath,
      etag: normEtag,
      byteLength: validSize,
      pathHash,
      etagHash,
      fingerprint,
      dataKey,
      metaKey,
    };
  }

  function buildPgimOpfsCacheKey(url, etag = null, byteLength = null, version = PGIM_OPFS_CACHE_VERSION) {
    return buildPgimOpfsFingerprint(url, etag, byteLength, version).dataKey;
  }

  /**
   * Build a deterministic, OPFS-filename-safe versioned fingerprint and cache keys for JSON
   * manifests (`program-manifest.json`, `boot-manifest.json`).
   *
   * @param {string} url
   * @param {string | null} [etag=null]
   * @param {number | null} [byteLength=null]
   * @param {number | string} [version=PGIM_OPFS_CACHE_VERSION]
   */
  function buildManifestOpfsFingerprint(url, etag = null, byteLength = null, version = PGIM_OPFS_CACHE_VERSION) {
    const ver = normalizeOpfsCacheVersion(version);
    const urlPath = sanitizePgimCacheUrlPath(url);
    const normEtag = normalizeEtag(etag);
    const validSize = Number.isSafeInteger(byteLength) && byteLength >= 0 ? byteLength : null;
    const pathHash = fnv1a64HexString(urlPath);
    const etagHash = normEtag !== null ? fnv1a64HexString(normEtag) : 'noetag';
    const sizeToken = validSize !== null ? String(validSize) : 'nosize';
    const leafRaw = urlPath.includes('/') ? urlPath.slice(urlPath.lastIndexOf('/') + 1) : urlPath;
    const stemRaw = leafRaw.replace(/\.json$/i, '');
    const safeStem = (stemRaw.replace(/[^A-Za-z0-9_-]/g, '_').replace(/^_+|_+$/g, '') || 'manifest').slice(0, 28);
    const versionTag = `v${ver}`;
    const fingerprint = `manifest-${versionTag}:${pathHash}:${sizeToken}:${etagHash}`;
    const dataKey = `manifest-${versionTag}-${pathHash}-${safeStem}-${sizeToken}-${etagHash}.json`;
    const metaKey = `manifest-${versionTag}-${pathHash}-${safeStem}.meta.json`;
    return {
      version: ver,
      urlPath,
      etag: normEtag,
      byteLength: validSize,
      pathHash,
      etagHash,
      fingerprint,
      dataKey,
      metaKey,
    };
  }

  function resolveNativeOpfsStorage(storageObj) {
    if (storageObj !== undefined) {
      return storageObj && typeof storageObj.getDirectory === 'function' ? storageObj : null;
    }
    try {
      const nav =
        typeof globalThis !== 'undefined' && globalThis && globalThis.navigator
          ? globalThis.navigator
          : typeof navigator !== 'undefined'
            ? navigator
            : null;
      const storage = nav && nav.storage ? nav.storage : null;
      return storage && typeof storage.getDirectory === 'function' ? storage : null;
    } catch {
      return null;
    }
  }

  function extractCachedBytes(raw) {
    if (raw === null || raw === undefined) return null;
    if (raw instanceof Uint8Array) return raw;
    if (ArrayBuffer.isView(raw)) return new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
    if (raw instanceof ArrayBuffer || isSharedArrayBuffer(raw)) return new Uint8Array(raw);
    if (typeof raw === 'object') {
      if (raw.hit === false) return null;
      if (raw.bytes !== undefined && raw.bytes !== null) return extractCachedBytes(raw.bytes);
      if (raw.data !== undefined && raw.data !== null) return extractCachedBytes(raw.data);
      if (raw.buffer !== undefined && raw.buffer !== null) return extractCachedBytes(raw.buffer);
    }
    return null;
  }

  function isFileSystemDirectoryHandle(value) {
    return Boolean(
      value &&
        typeof value === 'object' &&
        typeof value.getFileHandle === 'function' &&
        typeof value.getDirectoryHandle === 'function'
    );
  }

  function isOpfsDisabledByQuery() {
    try {
      const search =
        typeof globalThis !== 'undefined' &&
        globalThis.location &&
        typeof globalThis.location.search === 'string'
          ? globalThis.location.search
          : '';
      if (!search) return false;
      const query = search.startsWith('?') ? search.slice(1) : search;
      const params = new URLSearchParams(query);
      for (const [k, v] of params.entries()) {
        if (k.toLowerCase() === 'opfsl2') {
          const val = String(v).trim().toLowerCase();
          if (val === '0' || val === 'false' || val === 'no' || val === 'off') {
            return true;
          }
        }
      }
    } catch {
      // Ignore malformed location or URLSearchParams.
    }
    return false;
  }

  function hasNativeOpfsStorage(storageObj = undefined) {
    return resolveNativeOpfsStorage(storageObj) !== null;
  }

  function createNativeOpfsStore({ dirHandle = null, dirName = DEFAULT_OPFS_CACHE_DIR, storageObj = undefined } = {}) {
    let cachedDir = dirHandle || null;
    const resolveDir = async (create) => {
      if (cachedDir) return cachedDir;
      const storage = resolveNativeOpfsStorage(storageObj);
      if (!storage) return null;
      let rootDir;
      try {
        rootDir = await storage.getDirectory();
      } catch {
        return null;
      }
      if (!rootDir || typeof rootDir.getDirectoryHandle !== 'function') return null;
      try {
        const dir = await rootDir.getDirectoryHandle(dirName, { create });
        if (dir) cachedDir = dir;
        return dir;
      } catch (err) {
        if (!create && isNotFoundError(err)) return null;
        throw err;
      }
    };

    return {
      async read(key) {
        const dir = await resolveDir(false);
        if (!dir) return null;
        let handle;
        try {
          handle = await dir.getFileHandle(key, { create: false });
        } catch (err) {
          if (isNotFoundError(err)) return null;
          throw err;
        }
        if (!handle || typeof handle.getFile !== 'function') return null;
        const file = await handle.getFile();
        if (!file || typeof file.arrayBuffer !== 'function') return null;
        return new Uint8Array(await file.arrayBuffer());
      },
      async write(key, bytes) {
        if (!bytes || (bytes.buffer && bytes.buffer.detached)) return false;
        const dir = await resolveDir(true);
        if (!dir) return false;
        const handle = await dir.getFileHandle(key, { create: true });
        if (!handle) return false;
        let payload = isSharedArrayBuffer(bytes.buffer) ? bytes.slice() : bytes;
        if (typeof handle.createWritable === 'function') {
          const writable = await handle.createWritable();
          try {
            await writable.write(payload);
            payload = null;
            await writable.close();
            return true;
          } catch (err) {
            payload = null;
            if (typeof writable.abort === 'function') {
              try {
                await writable.abort();
              } catch {
                // Ignore abort cleanup error.
              }
            }
            throw err;
          }
        }
        if (typeof handle.createSyncAccessHandle === 'function') {
          const syncHandle = await handle.createSyncAccessHandle();
          try {
            if (typeof syncHandle.truncate === 'function') {
              syncHandle.truncate(0);
            }
            let pos = 0;
            while (pos < payload.byteLength) {
              const step = syncHandle.write(pos === 0 ? payload : payload.subarray(pos), { at: pos });
              if (step === undefined) {
                pos = payload.byteLength;
                break;
              }
              if (typeof step !== 'number' || step <= 0) {
                throw new Error(`Short OPFS sync write (${pos}/${payload.byteLength})`);
              }
              pos += step;
            }
            if (typeof syncHandle.flush === 'function') {
              syncHandle.flush();
            }
            return true;
          } finally {
            payload = null;
            if (typeof syncHandle.close === 'function') {
              syncHandle.close();
            }
          }
        }
        return false;
      },
      async delete(key) {
        const dir = await resolveDir(false);
        if (!dir || typeof dir.removeEntry !== 'function') return false;
        try {
          await dir.removeEntry(key);
          return true;
        } catch (err) {
          if (isNotFoundError(err)) return false;
          throw err;
        }
      },
    };
  }

  function normalizeOpfsCacheStore(opfsCache, { dirName = DEFAULT_OPFS_CACHE_DIR, storageObj = undefined } = {}) {
    if (opfsCache === null || opfsCache === false) {
      return { store: null, enabled: false };
    }
    if (opfsCache === undefined || opfsCache === DEFAULT_OPFS_CACHE_MODE) {
      if (isOpfsDisabledByQuery()) {
        return { store: null, enabled: false };
      }
      const nativeStorage = resolveNativeOpfsStorage(storageObj);
      if (!nativeStorage) {
        return { store: null, enabled: false };
      }
      return { store: createNativeOpfsStore({ dirName, storageObj: nativeStorage }), enabled: true };
    }
    if (opfsCache === true) {
      const nativeStorage = resolveNativeOpfsStorage(storageObj);
      return { store: createNativeOpfsStore({ dirName, storageObj }), enabled: nativeStorage !== null };
    }
    if (isFileSystemDirectoryHandle(opfsCache)) {
      return { store: createNativeOpfsStore({ dirHandle: opfsCache, dirName }), enabled: true };
    }
    if (typeof opfsCache === 'object') {
      const hasReadWrite = typeof opfsCache.read === 'function' && typeof opfsCache.write === 'function';
      const hasGetPut = typeof opfsCache.get === 'function' && typeof opfsCache.put === 'function';
      if (hasReadWrite || hasGetPut) {
        const deleteFn =
          typeof opfsCache.delete === 'function'
            ? (key) => opfsCache.delete(key)
            : typeof opfsCache.remove === 'function'
              ? (key) => opfsCache.remove(key)
              : null;
        return {
          store: {
            read: hasReadWrite
              ? (key, meta) => opfsCache.read(key, meta)
              : (key, meta) => opfsCache.get(key, meta),
            write: hasReadWrite
              ? (key, bytes, meta) => opfsCache.write(key, bytes, meta)
              : (key, bytes, meta) => opfsCache.put(key, bytes, meta),
            delete: deleteFn,
          },
          enabled: true,
        };
      }
    }
    throw new TypeError(
      "opfsCache must be 'auto', null, boolean, a FileSystemDirectoryHandle, or an object exposing { read, write, delete } (or { get, put, delete })"
    );
  }

  function httpError(url, status, what = 'PGIM image') {
    if (status === 404 || status === 410) {
      return new PgimPreloadError('NOT_FOUND', `${what} not found: ${url} (HTTP ${status})`, { status, url });
    }
    return new PgimPreloadError('HTTP_ERROR', `${what} request failed: ${url} (HTTP ${status})`, { status, url });
  }

  function isPlainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  function validateWgslPackBytes(bytes, url) {
    const fail = (reason) => new PgimPreloadError('BAD_SHADER_PACK', `invalid WGPK shader pack ${url}: ${reason}`, { url });
    if (bytes.byteLength < WGPK_HEADER_BYTES) throw fail(`truncated header (${bytes.byteLength} bytes)`);
    for (let i = 0; i < 4; i += 1) {
      if (bytes[i] !== WGPK_MAGIC_BYTES[i]) throw fail(`bad magic [${hexBytes(bytes.subarray(0, 4))}] (expected "WGPK")`);
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = view.getUint32(4, true);
    if (version !== WGPK_VERSION) throw fail(`unsupported version ${version} (expected ${WGPK_VERSION})`);
    const entryCount = view.getUint32(8, true);
    const tableEnd = WGPK_HEADER_BYTES + entryCount * WGPK_ENTRY_BYTES;
    if (tableEnd > bytes.byteLength) throw fail(`entry table needs ${tableEnd} bytes, only ${bytes.byteLength} available`);
    return bytes;
  }

  function parseReflectionManifestJson(bytes, url) {
    const fail = (reason, cause) =>
      new PgimPreloadError('BAD_REFLECTION_MANIFEST', `invalid reflection manifest ${url}: ${reason}`, { url, cause });
    let raw;
    try {
      raw = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    } catch (err) {
      throw fail(`malformed JSON (${err instanceof Error ? err.message : String(err)})`, err);
    }
    if (!isPlainObject(raw)) throw fail('manifest root must be a JSON object');
    if (raw.schema_version !== REFLECTION_SCHEMA_VERSION) {
      throw fail(`unsupported schema_version ${JSON.stringify(raw.schema_version)} (expected ${REFLECTION_SCHEMA_VERSION})`);
    }
    return raw;
  }

  // ---------------------------------------------------------------------------------------------
  // Multi-module program-manifest.json & single-module boot-manifest.json parsing and URL safety.
  // ---------------------------------------------------------------------------------------------

  const CONTROL_CHAR_RE = /[\x00-\x1f\x7f]/;
  const SCHEME_RE = /^[a-zA-Z][a-zA-Z0-9+.-]*:/;
  const WINDOWS_DRIVE_RE = /^[a-zA-Z]:/;
  const SAFE_MODULE_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

  function unsafeUrlError(rawUrl, reason, extra = {}) {
    return new PgimPreloadError('UNSAFE_URL', `unsafe URL "${rawUrl}": ${reason}`, {
      url: rawUrl,
      reason,
      ...extra,
    });
  }

  function decodeUrlPasses(raw, label) {
    const passes = [raw];
    let current = raw;
    for (let i = 0; i < 4; i += 1) {
      let decoded;
      try {
        decoded = decodeURIComponent(current);
      } catch {
        throw unsafeUrlError(raw, `${label} contains malformed percent-encoding`);
      }
      if (decoded === current) break;
      passes.push(decoded);
      current = decoded;
    }
    return passes;
  }

  function normalizeSafePathSegments(pathname, rawForError, label) {
    const leadingSlash = pathname.startsWith('/');
    const trimmed = leadingSlash ? pathname.slice(1) : pathname;
    if (trimmed.length === 0) {
      throw unsafeUrlError(rawForError, `${label} must not be empty`);
    }
    const rawSegments = trimmed.split('/');
    const out = [];
    for (const seg of rawSegments) {
      if (seg === '') {
        throw unsafeUrlError(rawForError, `${label} contains empty path segment`);
      }
      if (seg === '..') {
        throw unsafeUrlError(rawForError, `${label} contains path traversal segment ".."`);
      }
      if (seg === '.') continue;
      out.push(seg);
    }
    if (out.length === 0) {
      throw unsafeUrlError(rawForError, `${label} resolves to an empty path`);
    }
    return { leadingSlash, segments: out, normalized: (leadingSlash ? '/' : '') + out.join('/') };
  }

  function parseManifestBase(baseUrl) {
    if (baseUrl === null || baseUrl === undefined || baseUrl === '') {
      return { hasOrigin: false, origin: '', dir: '', isRootRelative: false };
    }
    if (typeof baseUrl !== 'string') {
      throw new TypeError('baseUrl must be a string or null');
    }
    if (baseUrl.trim() !== baseUrl || CONTROL_CHAR_RE.test(baseUrl) || baseUrl.includes('\\')) {
      throw unsafeUrlError(baseUrl, 'manifest base URL contains forbidden characters');
    }
    const noHashOrQuery = baseUrl.split(/[?#]/, 1)[0];
    const passes = decodeUrlPasses(noHashOrQuery, 'manifest base URL');
    for (const pass of passes) {
      if (CONTROL_CHAR_RE.test(pass) || pass.includes('\\')) {
        throw unsafeUrlError(baseUrl, 'manifest base URL contains forbidden characters');
      }
      if (WINDOWS_DRIVE_RE.test(pass)) {
        throw unsafeUrlError(baseUrl, 'manifest base URL must not be a filesystem drive path');
      }
    }

    if (SCHEME_RE.test(noHashOrQuery)) {
      let parsed;
      try {
        parsed = new URL(noHashOrQuery);
      } catch {
        throw unsafeUrlError(baseUrl, 'manifest base URL is not a valid URL');
      }
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        throw unsafeUrlError(baseUrl, `unsupported manifest URL scheme "${parsed.protocol}"`);
      }
      if (parsed.username || parsed.password) {
        throw unsafeUrlError(baseUrl, 'manifest base URL must not contain credentials');
      }
      for (const pass of passes) {
        if (pass.split('/').includes('..')) {
          throw unsafeUrlError(baseUrl, 'manifest base URL contains path traversal');
        }
      }
      const pathname = parsed.pathname;
      const decodedPathnamePasses = decodeUrlPasses(pathname, 'manifest base URL');
      for (const pass of decodedPathnamePasses) {
        if (pass.split('/').includes('..')) {
          throw unsafeUrlError(baseUrl, 'manifest base URL contains path traversal');
        }
      }
      const lastSlash = pathname.lastIndexOf('/');
      const dir = lastSlash >= 0 ? pathname.slice(0, lastSlash + 1) : '/';
      return { hasOrigin: true, origin: parsed.origin, dir, isRootRelative: true };
    }

    if (noHashOrQuery.startsWith('//')) {
      throw unsafeUrlError(baseUrl, 'manifest base URL must not be protocol-relative');
    }
    for (const pass of passes) {
      if (pass.startsWith('//') || pass.split('/').includes('..')) {
        throw unsafeUrlError(baseUrl, 'manifest base URL contains path traversal');
      }
    }
    const isRootRelative = noHashOrQuery.startsWith('/');
    const lastSlash = noHashOrQuery.lastIndexOf('/');
    const dir = lastSlash >= 0 ? noHashOrQuery.slice(0, lastSlash + 1) : '';
    return { hasOrigin: false, origin: '', dir, isRootRelative };
  }

  function validateManifestEndpointUrl(url, label = 'manifest URL') {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError(`${label} must be a non-empty string`);
    }
    if (url.trim() !== url || CONTROL_CHAR_RE.test(url) || url.includes('\\')) {
      throw unsafeUrlError(url, `${label} contains forbidden characters`);
    }
    const noHashOrQuery = url.split(/[?#]/, 1)[0];
    if (noHashOrQuery.length === 0) {
      throw unsafeUrlError(url, `${label} must not be empty`);
    }
    const passes = decodeUrlPasses(noHashOrQuery, label);
    for (const pass of passes) {
      if (CONTROL_CHAR_RE.test(pass) || pass.includes('\\')) {
        throw unsafeUrlError(url, `${label} contains forbidden characters`);
      }
      if (WINDOWS_DRIVE_RE.test(pass)) {
        throw unsafeUrlError(url, `${label} must not be a filesystem drive path`);
      }
    }
    if (SCHEME_RE.test(noHashOrQuery)) {
      let parsed;
      try {
        parsed = new URL(url);
      } catch {
        throw unsafeUrlError(url, `${label} is not a valid URL`);
      }
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        throw unsafeUrlError(url, `${label} has unsupported scheme "${parsed.protocol}"`);
      }
      if (parsed.username || parsed.password) {
        throw unsafeUrlError(url, `${label} must not contain credentials`);
      }
      for (const pass of passes) {
        if (pass.split('/').includes('..')) {
          throw unsafeUrlError(url, `${label} contains path traversal`);
        }
      }
      for (const pass of decodeUrlPasses(parsed.pathname, label)) {
        if (pass.split('/').includes('..')) {
          throw unsafeUrlError(url, `${label} contains path traversal`);
        }
      }
      return url;
    }
    if (noHashOrQuery.startsWith('//')) {
      throw unsafeUrlError(url, `${label} must not be protocol-relative`);
    }
    for (const pass of passes) {
      if (pass.startsWith('//') || pass.split('/').includes('..') || SCHEME_RE.test(pass)) {
        throw unsafeUrlError(url, `${label} contains unsafe traversal or scheme`);
      }
    }
    return url;
  }

  function deriveBootManifestUrl(manifestUrl, bootManifestFile = DEFAULT_BOOT_MANIFEST_FILE) {
    const base = parseManifestBase(manifestUrl);
    return (base.hasOrigin ? base.origin : '') + base.dir + bootManifestFile;
  }

  function validateModuleName(rawName, manifestUrl) {
    if (typeof rawName !== 'string' || rawName.length === 0) {
      throw new PgimPreloadError(
        'BAD_PROGRAM_MANIFEST',
        `invalid program manifest ${manifestUrl}: module name must be a non-empty string`,
        { url: manifestUrl }
      );
    }
    if (
      rawName.trim() !== rawName ||
      CONTROL_CHAR_RE.test(rawName) ||
      rawName.includes('/') ||
      rawName.includes('\\') ||
      rawName.includes('..') ||
      rawName.includes('%') ||
      rawName.includes(':') ||
      rawName.includes('?') ||
      rawName.includes('#') ||
      !SAFE_MODULE_NAME_RE.test(rawName)
    ) {
      throw unsafeUrlError(rawName, 'module name contains unsafe path or URL characters');
    }
    return rawName;
  }

  /**
   * Resolve a manifest-declared asset path against `baseUrl` while strictly rejecting path
   * traversal, encoded traversal, backslashes, unsafe schemes, cross-origin URLs, and paths
   * escaping the manifest's base directory.
   *
   * @param {string} rawPath
   * @param {string | null} [baseUrl]
   * @param {{ requirePgimExtension?: boolean, label?: string }} [options]
   * @returns {{ url: string, path: string }}
   */
  function resolveSafeManifestUrl(rawPath, baseUrl = null, options = {}) {
    const { requirePgimExtension = true, label = 'module PGIM URL' } = options;
    if (typeof rawPath !== 'string' || rawPath.length === 0) {
      throw unsafeUrlError(String(rawPath), `${label} must be a non-empty string`);
    }
    if (rawPath.trim() !== rawPath) {
      throw unsafeUrlError(rawPath, `${label} must not contain leading or trailing whitespace`);
    }
    if (CONTROL_CHAR_RE.test(rawPath)) {
      throw unsafeUrlError(rawPath, `${label} contains control characters`);
    }
    if (rawPath.includes('\\')) {
      throw unsafeUrlError(rawPath, `${label} must not contain backslashes`);
    }
    if (rawPath.includes('?') || rawPath.includes('#')) {
      throw unsafeUrlError(rawPath, `${label} must not contain query strings or fragments`);
    }

    const base = parseManifestBase(baseUrl);
    const passes = decodeUrlPasses(rawPath, label);
    for (const pass of passes) {
      if (CONTROL_CHAR_RE.test(pass)) {
        throw unsafeUrlError(rawPath, `${label} contains encoded control characters`);
      }
      if (pass.includes('\\')) {
        throw unsafeUrlError(rawPath, `${label} contains encoded backslashes`);
      }
      if (pass.includes('?') || pass.includes('#')) {
        throw unsafeUrlError(rawPath, `${label} contains encoded query or fragment delimiters`);
      }
      if (WINDOWS_DRIVE_RE.test(pass)) {
        throw unsafeUrlError(rawPath, `${label} must not be a filesystem drive path`);
      }
    }

    let resolvedUrl;
    let relativePath;
    let lastSegment;

    const hasScheme = passes.some((pass) => SCHEME_RE.test(pass));
    if (hasScheme) {
      if (!SCHEME_RE.test(rawPath)) {
        throw unsafeUrlError(rawPath, `${label} contains an encoded URL scheme`);
      }
      let parsed;
      try {
        parsed = new URL(rawPath);
      } catch {
        throw unsafeUrlError(rawPath, `${label} is not a valid URL`);
      }
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        throw unsafeUrlError(rawPath, `${label} uses forbidden scheme "${parsed.protocol}"`);
      }
      if (parsed.username || parsed.password) {
        throw unsafeUrlError(rawPath, `${label} must not contain embedded credentials`);
      }
      if (!base.hasOrigin || parsed.origin !== base.origin) {
        throw unsafeUrlError(rawPath, `${label} must not reference a different or external origin`);
      }
      for (const pass of passes) {
        if (pass.split('/').includes('..')) {
          throw unsafeUrlError(rawPath, `${label} contains path traversal`);
        }
      }
      for (const pass of decodeUrlPasses(parsed.pathname, label)) {
        if (pass.includes('\\') || pass.split('/').includes('..')) {
          throw unsafeUrlError(rawPath, `${label} contains path traversal`);
        }
      }
      const { normalized, segments } = normalizeSafePathSegments(parsed.pathname, rawPath, label);
      if (base.dir && !normalized.startsWith(base.dir)) {
        throw unsafeUrlError(rawPath, `${label} escapes manifest base directory "${base.dir}"`);
      }
      resolvedUrl = parsed.origin + normalized;
      relativePath = base.dir ? normalized.slice(base.dir.length) : normalized.replace(/^\/+/, '');
      lastSegment = segments[segments.length - 1];
    } else {
      if (passes.some((pass) => pass.startsWith('//'))) {
        throw unsafeUrlError(rawPath, `${label} must not be protocol-relative`);
      }
      for (const pass of passes) {
        normalizeSafePathSegments(pass, rawPath, label);
      }
      const { leadingSlash, normalized, segments } = normalizeSafePathSegments(rawPath, rawPath, label);
      lastSegment = segments[segments.length - 1];
      if (leadingSlash) {
        if (base.dir) {
          if (!base.isRootRelative || !normalized.startsWith(base.dir)) {
            throw unsafeUrlError(rawPath, `${label} escapes manifest base directory "${base.dir}"`);
          }
          resolvedUrl = base.hasOrigin ? base.origin + normalized : normalized;
          relativePath = normalized.slice(base.dir.length);
        } else {
          resolvedUrl = base.hasOrigin ? base.origin + normalized : normalized;
          relativePath = segments.join('/');
        }
      } else {
        relativePath = normalized;
        resolvedUrl = (base.hasOrigin ? base.origin : '') + base.dir + relativePath;
      }
    }

    if (requirePgimExtension) {
      const lowerLast = lastSegment.toLowerCase();
      if (!lowerLast.endsWith('.pgim') || lowerLast.length <= '.pgim'.length) {
        throw unsafeUrlError(rawPath, `${label} must reference a .pgim file`);
      }
    }

    return { url: resolvedUrl, path: relativePath };
  }

  function decodeManifestJson(input, errorCode, label, url) {
    const fail = (reason, cause) =>
      new PgimPreloadError(errorCode, `invalid ${label} ${url}: ${reason}`, { url, cause });
    let text;
    if (typeof input === 'string') {
      text = input;
    } else if (
      input instanceof Uint8Array ||
      ArrayBuffer.isView(input) ||
      input instanceof ArrayBuffer ||
      isSharedArrayBuffer(input)
    ) {
      try {
        text = new TextDecoder('utf-8', { fatal: true }).decode(toBytes(input, label));
      } catch (err) {
        throw fail(`invalid UTF-8 (${err instanceof Error ? err.message : String(err)})`, err);
      }
    } else if (isPlainObject(input)) {
      return input;
    } else {
      throw fail('manifest input must be a JSON object, string, or byte buffer');
    }
    let raw;
    try {
      raw = JSON.parse(text);
    } catch (err) {
      throw fail(`malformed JSON (${err instanceof Error ? err.message : String(err)})`, err);
    }
    if (!isPlainObject(raw)) {
      throw fail('manifest root must be a JSON object');
    }
    return raw;
  }

  function extractExplicitPgimField(obj) {
    if (!isPlainObject(obj)) return null;
    const directKeys = ['pgimUrl', 'pgim_url', 'pgimPath', 'pgim_path'];
    for (const key of directKeys) {
      if (obj[key] !== undefined && obj[key] !== null) {
        return obj[key];
      }
    }
    if (obj.pgim !== undefined && obj.pgim !== null) {
      if (typeof obj.pgim === 'string') return obj.pgim;
      if (isPlainObject(obj.pgim)) {
        return obj.pgim.url ?? obj.pgim.path ?? null;
      }
      return obj.pgim;
    }
    if (obj.image !== undefined && obj.image !== null) {
      if (typeof obj.image === 'string') return obj.image;
      if (isPlainObject(obj.image)) {
        const candidate = obj.image.url ?? obj.image.path ?? obj.image.pgim ?? null;
        if (candidate !== null) return candidate;
      }
    }
    return null;
  }

  function extractExpectedBytes(entry, matchedFile = null) {
    const candidates = [
      entry && entry.pgimSize,
      entry && entry.pgim_size,
      entry && entry.containerBytes,
      entry && entry.container_bytes,
      entry && isPlainObject(entry.image) ? entry.image.container_bytes : null,
      entry && isPlainObject(entry.image) ? entry.image.size : null,
      matchedFile ? matchedFile.size : null,
    ];
    for (const value of candidates) {
      if (Number.isSafeInteger(value) && value >= 0) return value;
    }
    return null;
  }

  function extractManifestPgimFiles(rawFiles, effectiveBase) {
    if (!Array.isArray(rawFiles)) return [];
    const results = [];
    const seen = new Set();
    for (const item of rawFiles) {
      let rawPath = null;
      let size = null;
      if (typeof item === 'string') {
        rawPath = item;
      } else if (isPlainObject(item) && typeof item.path === 'string') {
        rawPath = item.path;
        if (Number.isSafeInteger(item.size) && item.size >= 0) size = item.size;
      }
      if (!rawPath) continue;
      const lower = rawPath.toLowerCase();
      if (!lower.endsWith('.pgim')) continue;
      const resolved = resolveSafeManifestUrl(rawPath, effectiveBase, {
        requirePgimExtension: true,
        label: 'manifest file PGIM path',
      });
      if (seen.has(resolved.url)) continue;
      seen.add(resolved.url);
      results.push({ url: resolved.url, path: resolved.path, size });
    }
    return results;
  }

  function matchModuleFileEntry(moduleName, fileEntries) {
    if (fileEntries.length === 0) return null;
    const target = moduleName.toLowerCase();
    const canonical = `${target}/image/guest.pgim`;
    const canonicalSuffix = `/${canonical}`;
    for (const entry of fileEntries) {
      const norm = entry.path.toLowerCase();
      if (norm === canonical || norm.endsWith(canonicalSuffix)) return entry;
    }
    const prefix = `${target}/`;
    for (const entry of fileEntries) {
      if (entry.path.toLowerCase().startsWith(prefix)) return entry;
    }
    const segment = `/${target}/`;
    for (const entry of fileEntries) {
      if (entry.path.toLowerCase().includes(segment)) return entry;
    }
    return null;
  }

  /**
   * Parse and validate a multi-module `program-manifest.json` document (produced by
   * `playgta6-recomp build-program`). Discovers every guest module in initialisation order,
   * resolves its `.pgim` URL relative to `baseUrl`, and strictly rejects unsafe/traversal URLs.
   *
   * @param {object | string | ArrayBuffer | SharedArrayBuffer | ArrayBufferView} input
   * @param {{ baseUrl?: string | null, manifestUrl?: string | null, url?: string | null }} [options]
   */
  function parseProgramManifest(input, options = {}) {
    const effectiveBase = options.baseUrl ?? options.manifestUrl ?? null;
    const labelUrl = options.url ?? options.manifestUrl ?? options.baseUrl ?? 'program-manifest.json';
    const fail = (reason) =>
      new PgimPreloadError('BAD_PROGRAM_MANIFEST', `invalid program manifest ${labelUrl}: ${reason}`, { url: labelUrl });

    const raw = decodeManifestJson(input, 'BAD_PROGRAM_MANIFEST', 'program manifest', labelUrl);
    if (raw.schema !== undefined && raw.schema !== PROGRAM_MANIFEST_SCHEMA) {
      throw fail(`unsupported schema ${JSON.stringify(raw.schema)} (expected "${PROGRAM_MANIFEST_SCHEMA}")`);
    }
    if (raw.schema_version !== undefined && raw.schema_version !== PROGRAM_MANIFEST_SCHEMA_VERSION) {
      throw fail(
        `unsupported schema_version ${JSON.stringify(raw.schema_version)} (expected ${PROGRAM_MANIFEST_SCHEMA_VERSION})`
      );
    }
    if (!Array.isArray(raw.modules) || raw.modules.length === 0) {
      throw fail('manifest "modules" must be a non-empty array');
    }

    const base = parseManifestBase(effectiveBase);
    const fileEntries = extractManifestPgimFiles(raw.files, effectiveBase);
    const modules = [];
    const seenNames = new Set();

    for (let i = 0; i < raw.modules.length; i += 1) {
      const entry = raw.modules[i];
      let rawName;
      let explicitPgim = null;
      let objEntry = null;
      if (typeof entry === 'string') {
        rawName = entry;
      } else if (isPlainObject(entry)) {
        objEntry = entry;
        rawName = entry.module ?? entry.name ?? entry.id;
        explicitPgim = extractExplicitPgimField(entry);
      } else {
        throw fail(`modules[${i}] must be a string or object`);
      }

      const name = validateModuleName(rawName, labelUrl);
      const key = name.toLowerCase();
      if (seenNames.has(key)) {
        throw fail(`duplicate module "${name}"`);
      }
      seenNames.add(key);

      const matchedFile = matchModuleFileEntry(name, fileEntries);
      let resolved;
      if (explicitPgim !== null) {
        resolved = resolveSafeManifestUrl(explicitPgim, effectiveBase, {
          requirePgimExtension: true,
          label: `module "${name}" PGIM URL`,
        });
      } else if (matchedFile) {
        resolved = { url: matchedFile.url, path: matchedFile.path };
      } else {
        resolved = resolveSafeManifestUrl(`${name}/image/guest.pgim`, effectiveBase, {
          requirePgimExtension: true,
          label: `module "${name}" PGIM URL`,
        });
      }

      let buildManifestUrl = null;
      if (objEntry && isPlainObject(objEntry.build_manifest) && objEntry.build_manifest.path !== undefined) {
        buildManifestUrl = resolveSafeManifestUrl(objEntry.build_manifest.path, effectiveBase, {
          requirePgimExtension: false,
          label: `module "${name}" build_manifest path`,
        }).url;
      }

      modules.push({
        name,
        module: name,
        url: resolved.url,
        path: resolved.path,
        role: objEntry && typeof objEntry.role === 'string' ? objEntry.role : null,
        dll: objEntry && typeof objEntry.dll === 'string' ? objEntry.dll : null,
        file: objEntry && typeof objEntry.file === 'string' ? objEntry.file : null,
        base: objEntry && objEntry.base !== undefined ? objEntry.base : null,
        sizeOfImage:
          objEntry && (objEntry.size_of_image !== undefined || objEntry.sizeOfImage !== undefined)
            ? (objEntry.size_of_image ?? objEntry.sizeOfImage)
            : null,
        entryPoint:
          objEntry && (objEntry.entry_point !== undefined || objEntry.entryPoint !== undefined)
            ? (objEntry.entry_point ?? objEntry.entryPoint)
            : null,
        expectedBytes: extractExpectedBytes(objEntry, matchedFile),
        buildManifestUrl,
      });
    }

    return {
      schema: raw.schema || PROGRAM_MANIFEST_SCHEMA,
      schemaVersion: raw.schema_version ?? PROGRAM_MANIFEST_SCHEMA_VERSION,
      baseUrl: (base.hasOrigin ? base.origin : '') + base.dir,
      modules,
      initOrder: modules.map((m) => m.name),
      raw,
    };
  }

  /**
   * Parse and validate a legacy single-module `boot-manifest.json` (or `build-manifest.json`)
   * fallback manifest when `program-manifest.json` is absent (HTTP 404).
   *
   * @param {object | string | ArrayBuffer | SharedArrayBuffer | ArrayBufferView} input
   * @param {{ baseUrl?: string | null, manifestUrl?: string | null, url?: string | null }} [options]
   */
  function parseBootManifest(input, options = {}) {
    const effectiveBase = options.baseUrl ?? options.manifestUrl ?? null;
    const labelUrl = options.url ?? options.manifestUrl ?? options.baseUrl ?? DEFAULT_BOOT_MANIFEST_FILE;
    const fail = (reason) =>
      new PgimPreloadError('BAD_BOOT_MANIFEST', `invalid boot manifest ${labelUrl}: ${reason}`, { url: labelUrl });

    const raw = decodeManifestJson(input, 'BAD_BOOT_MANIFEST', 'boot manifest', labelUrl);
    if (raw.schema_version !== undefined && raw.schema_version !== 1) {
      throw fail(`unsupported schema_version ${JSON.stringify(raw.schema_version)} (expected 1)`);
    }

    const base = parseManifestBase(effectiveBase);
    const fileEntries = extractManifestPgimFiles(raw.files, effectiveBase);

    let name = 'main';
    if (raw.module !== undefined && raw.module !== null) {
      name = validateModuleName(raw.module, labelUrl);
    } else if (raw.name !== undefined && raw.name !== null) {
      name = validateModuleName(raw.name, labelUrl);
    }

    let explicitPgim = extractExplicitPgimField(raw);
    let matchedFile = null;
    let role = typeof raw.role === 'string' ? raw.role : 'exe';
    let dll = typeof raw.dll === 'string' ? raw.dll : null;
    let file = typeof raw.file === 'string' ? raw.file : null;

    if (explicitPgim === null && Array.isArray(raw.modules) && raw.modules.length > 0) {
      const parsed = parseProgramManifest(raw, options);
      const first = parsed.modules[0];
      return {
        ...parsed,
        pgimUrl: first.url,
        shaderPackUrl: null,
        reflectionManifestUrl: null,
      };
    }

    let resolved;
    if (explicitPgim !== null) {
      resolved = resolveSafeManifestUrl(explicitPgim, effectiveBase, {
        requirePgimExtension: true,
        label: 'boot manifest PGIM URL',
      });
    } else if (fileEntries.length > 0) {
      matchedFile = matchModuleFileEntry(name, fileEntries) || fileEntries[0];
      resolved = { url: matchedFile.url, path: matchedFile.path };
    } else if (typeof raw.module === 'string') {
      const defaultPath = raw.schema === 'playgta6-recomp-build' ? 'image/guest.pgim' : `${name}/image/guest.pgim`;
      resolved = resolveSafeManifestUrl(defaultPath, effectiveBase, {
        requirePgimExtension: true,
        label: 'boot manifest PGIM URL',
      });
    } else {
      throw fail('boot manifest does not declare a .pgim image path');
    }

    let shaderPackUrl = null;
    const rawShader = raw.shaderPackUrl ?? raw.shader_pack_url ?? raw.shaderPack ?? null;
    if (rawShader !== null) {
      shaderPackUrl = resolveSafeManifestUrl(rawShader, effectiveBase, {
        requirePgimExtension: false,
        label: 'boot manifest shaderPackUrl',
      }).url;
    }

    let reflectionManifestUrl = null;
    const rawReflection = raw.reflectionManifestUrl ?? raw.reflection_manifest_url ?? raw.reflectionUrl ?? null;
    if (rawReflection !== null) {
      reflectionManifestUrl = resolveSafeManifestUrl(rawReflection, effectiveBase, {
        requirePgimExtension: false,
        label: 'boot manifest reflectionManifestUrl',
      }).url;
    }

    const mod = {
      name,
      module: name,
      url: resolved.url,
      path: resolved.path,
      role,
      dll,
      file,
      base: raw.base !== undefined ? raw.base : null,
      sizeOfImage: raw.size_of_image ?? raw.sizeOfImage ?? null,
      entryPoint: raw.entry_point ?? raw.entryPoint ?? null,
      expectedBytes: extractExpectedBytes(raw, matchedFile),
      buildManifestUrl: null,
    };

    return {
      schema: typeof raw.schema === 'string' ? raw.schema : 'boot-manifest',
      schemaVersion: raw.schema_version ?? 1,
      baseUrl: (base.hasOrigin ? base.origin : '') + base.dir,
      modules: [mod],
      pgimUrl: mod.url,
      shaderPackUrl,
      reflectionManifestUrl,
      initOrder: [mod.name],
      raw,
    };
  }

  /**
   * Streams a PGIM guest image (and optional boot shader assets) into memory before guest entry.
   */
  class PgimPreloader {
  /**
   * @param {{
   *   fetchImpl?: typeof fetch,
   *   rangeCache?: { read(path: string, offset: number, length: number): Promise<Uint8Array> } | null,
   *   opfsCache?: boolean | FileSystemDirectoryHandle | {
   *     read?: (key: string, meta?: object) => Promise<any>,
   *     write?: (key: string, bytes: Uint8Array, meta?: object) => Promise<any>,
   *     delete?: (key: string) => Promise<any>,
   *     get?: (key: string, meta?: object) => Promise<any>,
   *     put?: (key: string, bytes: Uint8Array, meta?: object) => Promise<any>,
   *   } | null,
   *   opfsCacheVersion?: number | string,
   *   opfsCacheDirName?: string,
   *   opfsValidateWithHead?: boolean,
   *   opfsStorageObj?: { getDirectory?: () => Promise<any> } | null,
   *   chunkSize?: number,
   *   chunkBytes?: number,
   *   concurrency?: number,
   *   maxRetries?: number,
   *   retryDelayMs?: number,
   *   baseRetryDelayMs?: number,
   *   maxRetryDelayMs?: number,
   *   sleepFn?: (ms: number, signal?: AbortSignal) => Promise<void>,
   *   sleep?: (ms: number, signal?: AbortSignal) => Promise<void>,
   *   nowFn?: () => number,
   *   useSharedArrayBuffer?: boolean,
   *   targetBuffer?: ArrayBuffer | SharedArrayBuffer | null,
   * }} [options] With `rangeCache`, image chunks are read through `rangeCache.read(url, ...)`
   *   (e.g. web/workers/range_cache.js) and `url` must be a path that cache accepts.
   */
  constructor({
    fetchImpl,
    rangeCache = null,
    opfsCache = DEFAULT_OPFS_CACHE_MODE,
    opfsCacheVersion = PGIM_OPFS_CACHE_VERSION,
    opfsCacheDirName = DEFAULT_OPFS_CACHE_DIR,
    opfsValidateWithHead = true,
    opfsStorageObj = undefined,
    chunkSize,
    chunkBytes,
    concurrency = DEFAULT_CONCURRENCY,
    maxRetries = DEFAULT_MAX_RETRIES,
    maxVersionRetries = DEFAULT_MAX_VERSION_RETRIES,
    retryDelayMs,
    baseRetryDelayMs,
    maxRetryDelayMs,
    sleepFn,
    sleep,
    nowFn = now,
    useSharedArrayBuffer = false,
    targetBuffer = null,
    streamShaders = undefined,
    streamThresholdBytes = undefined,
  } = {}) {
    const impl =
      fetchImpl !== undefined && fetchImpl !== null
        ? fetchImpl
        : typeof globalThis.fetch === 'function'
          ? globalThis.fetch.bind(globalThis)
          : null;
    if (typeof impl !== 'function') {
      throw new TypeError('PgimPreloader requires a fetchImpl function (no global fetch available)');
    }
    if (rangeCache !== null && (typeof rangeCache !== 'object' || typeof rangeCache.read !== 'function')) {
      throw new TypeError('rangeCache must expose read(path, offset, length)');
    }
    const normalizedOpfsVersion = normalizeOpfsCacheVersion(opfsCacheVersion);
    const normalizedOpfsDir =
      typeof opfsCacheDirName === 'string' && opfsCacheDirName.trim().length > 0
        ? opfsCacheDirName.trim()
        : DEFAULT_OPFS_CACHE_DIR;
    const normalizedOpfs = normalizeOpfsCacheStore(opfsCache, {
      dirName: normalizedOpfsDir,
      storageObj: opfsStorageObj,
    });
    const normalizedOpfsStore = normalizedOpfs.store;
    const opfsEnabled = normalizedOpfs.enabled;
    const effectiveChunkSize =
      chunkSize !== undefined
        ? chunkSize
        : chunkBytes !== undefined
          ? chunkBytes
          : DEFAULT_CHUNK_SIZE;
    if (!Number.isSafeInteger(effectiveChunkSize) || effectiveChunkSize < PGIM_HEADER_BYTES) {
      throw new RangeError(`chunkSize must be an integer >= ${PGIM_HEADER_BYTES}, got: ${effectiveChunkSize}`);
    }
    if (!Number.isSafeInteger(concurrency) || concurrency < 1) {
      throw new RangeError(`concurrency must be a positive integer, got: ${concurrency}`);
    }
    if (!Number.isSafeInteger(maxRetries) || maxRetries < 0) {
      throw new RangeError(`maxRetries must be a non-negative integer, got: ${maxRetries}`);
    }
    if (!Number.isSafeInteger(maxVersionRetries) || maxVersionRetries < 0) {
      throw new RangeError(`maxVersionRetries must be a non-negative integer, got: ${maxVersionRetries}`);
    }
    const effectiveRetryDelayMs =
      retryDelayMs !== undefined
        ? retryDelayMs
        : baseRetryDelayMs !== undefined
          ? baseRetryDelayMs
          : DEFAULT_BASE_RETRY_DELAY_MS;
    if (!Number.isFinite(effectiveRetryDelayMs) || effectiveRetryDelayMs < 0) {
      throw new RangeError(`retryDelayMs must be a non-negative number, got: ${effectiveRetryDelayMs}`);
    }
    const effectiveMaxRetryDelayMs =
      maxRetryDelayMs !== undefined
        ? maxRetryDelayMs
        : Math.max(DEFAULT_MAX_RETRY_DELAY_MS, effectiveRetryDelayMs);
    if (!Number.isFinite(effectiveMaxRetryDelayMs) || effectiveMaxRetryDelayMs < 0) {
      throw new RangeError(`maxRetryDelayMs must be a non-negative number, got: ${effectiveMaxRetryDelayMs}`);
    }
    const effectiveSleepFn =
      sleepFn !== undefined ? sleepFn : sleep !== undefined ? sleep : defaultSleep;
    if (typeof effectiveSleepFn !== 'function') {
      throw new TypeError('sleepFn must be a function');
    }
    if (typeof nowFn !== 'function') {
      throw new TypeError('nowFn must be a function');
    }
    if (
      targetBuffer !== null &&
      !(targetBuffer instanceof ArrayBuffer || isSharedArrayBuffer(targetBuffer))
    ) {
      throw new TypeError('targetBuffer must be an ArrayBuffer, SharedArrayBuffer, or null');
    }
    this.fetchImpl = impl;
    this.rangeCache = rangeCache;
    this.opfsCache = opfsCache;
    this._opfsStore = normalizedOpfsStore;
    this._opfsEnabled = opfsEnabled;
    this.opfsEnabled = opfsEnabled;
    this.opfsCacheVersion = normalizedOpfsVersion;
    this.opfsCacheDirName = normalizedOpfsDir;
    this.opfsValidateWithHead = Boolean(opfsValidateWithHead);
    this._pendingOpfsWrites = new Set();
    this.chunkSize = effectiveChunkSize;
    this.chunkBytes = effectiveChunkSize;
    this.concurrency = concurrency;
    this.maxRetries = maxRetries;
    this.maxVersionRetries = maxVersionRetries;
    this.retryDelayMs = effectiveRetryDelayMs;
    this.baseRetryDelayMs = effectiveRetryDelayMs;
    this.maxRetryDelayMs = effectiveMaxRetryDelayMs;
    this.sleepFn = effectiveSleepFn;
    this.sleep = effectiveSleepFn;
    this.nowFn = nowFn;
    this.useSharedArrayBuffer = Boolean(useSharedArrayBuffer);
    this.targetBuffer = targetBuffer;
    this.streamShaders = typeof streamShaders === 'boolean' ? streamShaders : undefined;
    this.streamThresholdBytes =
      typeof streamThresholdBytes === 'number' &&
      Number.isSafeInteger(streamThresholdBytes) &&
      streamThresholdBytes >= 0
        ? streamThresholdBytes
        : undefined;
    this._telemetryRun = null;
    this._wireBytesReleased = 0;
    this._droppedWasmBuffers = 0;
    this._streamingCompiles = 0;
    this._fallbackCompiles = 0;
  }

  /**
   * Wait for any background non-blocking OPFS write-through tasks to finish settling.
   * Never rejects even if a background OPFS write hit `QuotaExceededError`.
   */
  async flushOpfsWrites() {
    while (this._pendingOpfsWrites.size > 0) {
      const batch = Array.from(this._pendingOpfsWrites);
      await Promise.allSettled(batch);
    }
  }

  _resolveTransferConfig({ chunkSize, chunkBytes, concurrency } = {}) {
    const effectiveChunkSize =
      chunkSize !== undefined
        ? chunkSize
        : chunkBytes !== undefined
          ? chunkBytes
          : this.chunkSize;
    if (!Number.isSafeInteger(effectiveChunkSize) || effectiveChunkSize < PGIM_HEADER_BYTES) {
      throw new RangeError(`chunkSize must be an integer >= ${PGIM_HEADER_BYTES}, got: ${effectiveChunkSize}`);
    }
    const effectiveConcurrency = concurrency !== undefined ? concurrency : this.concurrency;
    if (!Number.isSafeInteger(effectiveConcurrency) || effectiveConcurrency < 1) {
      throw new RangeError(`concurrency must be a positive integer, got: ${effectiveConcurrency}`);
    }
    return { chunkSize: effectiveChunkSize, concurrency: effectiveConcurrency };
  }

  /**
   * Return sanitized finite throughput and retry/abort telemetry for the latest preload run.
   */
  getStats() {
    const snap = this._telemetryRun ? this._telemetryRun.snapshot() : emptyTelemetrySnapshot();
    snap.wireBytesReleased = Math.max(snap.wireBytesReleased, sanitizeNonNegativeInt(this._wireBytesReleased));
    snap.droppedWasmBuffers = Math.max(snap.droppedWasmBuffers, sanitizeNonNegativeInt(this._droppedWasmBuffers));
    snap.opfsEnabled = Boolean(this._opfsEnabled);
    snap.opfsHitRatePct = computeOpfsHitRatePct(snap.opfsHits, snap.opfsMisses);
    return snap;
  }

  /**
   * Return sanitized numeric telemetry for wire-byte drops and streaming Wasm compilation.
   */
  getMemoryDropTelemetry() {
    const snap = this.getStats();
    return {
      wireBytesReleased: snap.wireBytesReleased,
      droppedWasmBuffers: snap.droppedWasmBuffers,
      streamingCompiles: sanitizeNonNegativeInt(this._streamingCompiles),
      fallbackCompiles: sanitizeNonNegativeInt(this._fallbackCompiles),
    };
  }

  /**
   * Alias for `getStats()`, returning a sanitized status snapshot.
   */
  getSnapshot() {
    return this.getStats();
  }

  /**
   * Reset all throughput and retry/abort telemetry counters back to idle zeros.
   */
  resetStats() {
    this._telemetryRun = null;
    this._wireBytesReleased = 0;
    this._droppedWasmBuffers = 0;
    this._streamingCompiles = 0;
    this._fallbackCompiles = 0;
    return this.getStats();
  }

  /**
   * Fetch and compile a `.wasm` binary (e.g. `runtime.wasm` or `guest_part_*.wasm`) into a
   * `WebAssembly.Module` without retaining raw wire bytes on the Window heap.
   *
   * Prefers `WebAssembly.compileStreaming` whenever supported and the HTTP response carries
   * `application/wasm` (so wire bytes never materialize as a JS `ArrayBuffer`). Falls back
   * cleanly to `response.arrayBuffer()` -> `WebAssembly.compile(bytes)` with immediate wire-byte
   * release and increments `wireBytesReleased` / `droppedWasmBuffers`.
   *
   * @param {string} url
   * @param {{ signal?: AbortSignal, wasmApi?: typeof WebAssembly, allowStreamingWithoutContentType?: boolean }} [options]
   */
  async compileWasmStreaming(url, options = {}) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError('Wasm module url must be a non-empty string');
    }
    const safeOpts = options && typeof options === 'object' ? options : {};
    const signal = safeOpts.signal || null;
    const wasmApi =
      safeOpts.wasmApi && typeof safeOpts.wasmApi === 'object'
        ? safeOpts.wasmApi
        : typeof WebAssembly !== 'undefined'
          ? WebAssembly
          : null;
    if (!wasmApi || typeof wasmApi.compile !== 'function') {
      throw new Error('WebAssembly.compile is unavailable');
    }
    throwIfAborted(signal);
    const started = this._now();
    const run = this._telemetryRun || this._startTelemetryRun();
    const ctx = {
      url,
      signal,
      emit: () => {},
      requests: 0,
      run,
      ownsTotalBytes: false,
      what: 'Wasm module',
    };
    const response = await this._request(ctx, { method: 'GET', credentials: 'same-origin' });
    if (!response || !response.ok) {
      throw httpError(url, response ? response.status : 0, 'Wasm module');
    }
    const declaredBytes = contentLength(response);
    const contentType = String(headerValue(response, 'content-type') || '').toLowerCase();
    const isRealResponse = typeof Response === 'undefined' || response instanceof Response;
    const canStream =
      typeof wasmApi.compileStreaming === 'function' &&
      (isRealResponse || safeOpts.allowStreamingWithoutContentType === true) &&
      (contentType.includes('application/wasm') || safeOpts.allowStreamingWithoutContentType === true);

    if (canStream) {
      try {
        const mod = await raceAbort(wasmApi.compileStreaming(Promise.resolve(response)), signal);
        this._streamingCompiles += 1;
        const bytesLoaded = declaredBytes !== null ? declaredBytes : 0;
        return {
          module: mod,
          url,
          streaming: true,
          bytesLoaded,
          wireBytesReleased: 0,
          durationMs: this._elapsedSince(started),
        };
      } catch (streamErr) {
        if ((signal && signal.aborted) || isAbortError(streamErr)) {
          throw streamErr;
        }
        // If body was already consumed by compileStreaming, re-throw; otherwise fall back to arrayBuffer().
        if (response.bodyUsed) {
          throw streamErr;
        }
      }
    }

    let wireBytes = await raceAbort(response.arrayBuffer(), signal);
    throwIfAborted(signal);
    const byteLength = wireBytes && typeof wireBytes.byteLength === 'number' ? wireBytes.byteLength : 0;
    let mod;
    try {
      const compilePromise = wasmApi.compile(wireBytes);
      wireBytes = null;
      mod = await raceAbort(compilePromise, signal);
    } finally {
      wireBytes = null;
      if (byteLength > 0) {
        this._wireBytesReleased += byteLength;
        this._droppedWasmBuffers += 1;
        if (run) run.recordWireBytesReleased(byteLength, 1);
      }
    }
    this._fallbackCompiles += 1;
    return {
      module: mod,
      url,
      streaming: false,
      bytesLoaded: byteLength,
      wireBytesReleased: byteLength,
      durationMs: this._elapsedSince(started),
    };
  }

  /**
   * Alias for `resetStats()`.
   */
  reset() {
    return this.resetStats();
  }

  _now() {
    let ts;
    try {
      ts = this.nowFn();
    } catch {
      return 0;
    }
    return typeof ts === 'number' && Number.isFinite(ts) ? ts : 0;
  }

  _elapsedSince(started) {
    const delta = this._now() - started;
    return Number.isFinite(delta) && delta > 0 ? delta : 0;
  }

  _startTelemetryRun() {
    const run = new TelemetryRun(this.nowFn);
    this._telemetryRun = run;
    return run;
  }

  _finishTelemetryError(run, err, signal) {
    if (!run) return;
    if ((signal && signal.aborted) || isAbortError(err)) {
      run.recordAbort();
    } else {
      run.fail();
    }
  }

  _computeRetryDelayMs(attempt) {
    const base = this.retryDelayMs;
    if (!(base > 0)) return 0;
    const exp = Math.min(Math.max(0, attempt), 30);
    const raw = base * Math.pow(2, exp);
    const cap = this.maxRetryDelayMs;
    if (!Number.isFinite(raw)) return cap;
    return Math.min(cap, raw);
  }

  async _sleepWithAbort(delayMs, signal) {
    throwIfAborted(signal);
    await raceAbort(Promise.resolve().then(() => this.sleepFn(delayMs, signal)), signal);
    throwIfAborted(signal);
  }

  /**
   * Validate and stream a PGIM container into one contiguous Uint8Array.
   *
   * @param {string} url
   * @param {{ onProgress?: (e: { phase: string, loadedBytes: number, totalBytes: number | null, pct: number }) => void,
   *           signal?: AbortSignal, verifyChecksums?: boolean,
   *           chunkSize?: number, chunkBytes?: number, concurrency?: number }} [options]
   */
  async preloadGuestImage(
    url,
    {
      onProgress,
      signal,
      verifyChecksums = false,
      chunkSize,
      chunkBytes,
      concurrency,
      opfsValidateWithHead,
      validateWithHead,
      expectedEtag = null,
      expectedBytes = null,
    } = {}
  ) {
    if (typeof url !== 'string' || url.length === 0) throw new TypeError('PGIM url must be a non-empty string');
    const transfer = this._resolveTransferConfig({ chunkSize, chunkBytes, concurrency });
    const run = this._startTelemetryRun();
    const effectiveValidateWithHead =
      opfsValidateWithHead !== undefined
        ? Boolean(opfsValidateWithHead)
        : validateWithHead !== undefined
          ? Boolean(validateWithHead)
          : this.opfsValidateWithHead;
    try {
      const result = await this._preloadGuestImageCore(url, {
        onProgress,
        signal,
        verifyChecksums,
        chunkSize: transfer.chunkSize,
        concurrency: transfer.concurrency,
        precomputedProbe: null,
        emitProbe: true,
        emitComplete: true,
        telemetryRun: run,
        ownsTotalBytes: true,
        opfsValidateWithHead: effectiveValidateWithHead,
        expectedEtag,
        expectedBytes,
      });
      run.complete(result.totalBytes);
      const stats = run.snapshot();
      return {
        ...result,
        chunksFetched: stats.chunksFetched,
        elapsedMs: stats.elapsedMs,
        bytesPerSec: stats.bytesPerSec,
        peakBytesPerSec: stats.peakBytesPerSec,
        throughputMBps: stats.throughputMBps,
        parallelMode: stats.parallelMode,
        retryCount: stats.retryCount,
        abortedCount: stats.abortedCount,
      };
    } catch (err) {
      this._finishTelemetryError(run, err, signal);
      throw err;
    }
  }

  async _deleteOpfsKeys(keys) {
    const store = this._opfsStore;
    if (!store || typeof store.delete !== 'function') return;
    const unique = new Set();
    for (let i = 0; i < keys.length; i += 1) {
      const k = keys[i];
      if (typeof k === 'string' && k.length > 0 && !unique.has(k)) {
        unique.add(k);
      }
    }
    for (const key of unique) {
      try {
        await store.delete(key);
      } catch {
        // Best-effort stale entry cleanup; never fail preload on delete error.
      }
    }
  }

  async _readOpfsMeta(metaKey, signal, kind = 'pgim-meta') {
    const store = this._opfsStore;
    if (!store) return null;
    let raw;
    try {
      raw = await raceAbort(Promise.resolve().then(() => store.read(metaKey, { kind })), signal);
    } catch (err) {
      if ((signal && signal.aborted) || isAbortError(err)) throw err;
      return null;
    }
    throwIfAborted(signal);
    const bytes = extractCachedBytes(raw);
    if (!bytes || bytes.byteLength === 0) return null;
    try {
      const parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
      if (!isPlainObject(parsed)) return null;
      return parsed;
    } catch {
      return { _corrupt: true };
    }
  }

  async _tryLoadFromOpfsCache(
    ctx,
    probe,
    {
      verifyChecksums = false,
      expectedEtag = null,
      expectedBytes = null,
      expectedLastModified = null,
      countMiss = true,
    } = {}
  ) {
    const store = this._opfsStore;
    if (!store) return null;
    throwIfAborted(ctx.signal);

    const probeEtag = probe && probe.etag !== undefined ? normalizeEtag(probe.etag) : normalizeEtag(expectedEtag);
    const probeLastModified =
      probe && probe.lastModified !== undefined
        ? normalizeLastModified(probe.lastModified)
        : normalizeLastModified(expectedLastModified);
    const probeSize =
      probe && typeof probe.totalBytes === 'number' && Number.isSafeInteger(probe.totalBytes) && probe.totalBytes >= 0
        ? probe.totalBytes
        : Number.isSafeInteger(expectedBytes) && expectedBytes >= 0
          ? expectedBytes
          : null;
    const headValidated = Boolean(probe && probe.headOk);

    const baseFp = buildPgimOpfsFingerprint(ctx.url, probeEtag, probeSize, this.opfsCacheVersion);
    const meta = await this._readOpfsMeta(baseFp.metaKey, ctx.signal, 'pgim-meta');

    let candidateDataKey = null;
    let candidateEtag = probeEtag;
    let candidateLastModified = probeLastModified;
    let candidateFingerprint = baseFp.fingerprint;
    let invalidated = false;

    if (meta !== null) {
      const metaEtag = normalizeEtag(meta.etag);
      const validMetaShape =
        !meta._corrupt &&
        meta.version === this.opfsCacheVersion &&
        meta.urlPath === baseFp.urlPath &&
        typeof meta.dataKey === 'string' &&
        meta.dataKey.length > 0 &&
        Number.isSafeInteger(meta.byteLength) &&
        meta.byteLength >= PGIM_HEADER_BYTES &&
        hasVerifiedOpfsProvenance(meta) &&
        !isWeakEtag(metaEtag);

      if (!validMetaShape) {
        invalidated = true;
      } else {
        const metaLastModified = normalizeLastModified(meta.lastModified);
        let mismatch = false;
        if (probeSize !== null && meta.byteLength !== probeSize) {
          mismatch = true;
        }
        if (headValidated) {
          // Weak validators do not guarantee byte-for-byte identity for binary PGIM caching.
          if (isWeakEtag(probeEtag)) {
            mismatch = true;
          } else if (metaEtag !== probeEtag) {
            // Strict ETag equality when HEAD succeeded (detects ETag rotation, addition, or removal).
            mismatch = true;
          } else if (probeEtag === null && metaLastModified !== probeLastModified) {
            // Secondary validator when server omits ETag: Last-Modified change/addition/removal invalidates.
            mismatch = true;
          }
        } else {
          if (probeEtag !== null && (isWeakEtag(probeEtag) || metaEtag !== probeEtag)) {
            mismatch = true;
          } else if (probeEtag === null && probeLastModified !== null && metaLastModified !== probeLastModified) {
            mismatch = true;
          }
        }
        if (mismatch) {
          invalidated = true;
        } else {
          candidateDataKey = meta.dataKey;
          candidateEtag = metaEtag;
          candidateLastModified = metaLastModified;
          candidateFingerprint =
            typeof meta.fingerprint === 'string' && meta.fingerprint.length > 0
              ? meta.fingerprint
              : buildPgimOpfsFingerprint(ctx.url, metaEtag, meta.byteLength, this.opfsCacheVersion).fingerprint;
        }
      }

      if (invalidated) {
        const keysToDelete = [baseFp.metaKey];
        if (typeof meta.dataKey === 'string' && meta.dataKey.length > 0) {
          keysToDelete.push(meta.dataKey);
        }
        await this._deleteOpfsKeys(keysToDelete);
        if (ctx.run) {
          ctx.run.recordOpfsInvalidation();
          if (countMiss) ctx.run.recordOpfsMiss();
        }
        return null;
      }
    }

    if (!candidateDataKey) {
      if (countMiss && ctx.run) ctx.run.recordOpfsMiss();
      return null;
    }

    let rawData;
    try {
      rawData = await raceAbort(
        Promise.resolve().then(() =>
          store.read(candidateDataKey, {
            kind: 'pgim',
            version: this.opfsCacheVersion,
            urlPath: baseFp.urlPath,
            etag: candidateEtag,
            byteLength: probeSize,
            fingerprint: candidateFingerprint,
          })
        ),
        ctx.signal
      );
    } catch (err) {
      if ((ctx.signal && ctx.signal.aborted) || isAbortError(err)) throw err;
      if (countMiss && ctx.run) ctx.run.recordOpfsMiss();
      return null;
    }
    throwIfAborted(ctx.signal);

    const cachedRawBytes = extractCachedBytes(rawData);
    if (!cachedRawBytes || cachedRawBytes.byteLength < PGIM_HEADER_BYTES) {
      if (meta !== null) {
        await this._deleteOpfsKeys([candidateDataKey, baseFp.metaKey]);
        if (ctx.run) ctx.run.recordOpfsInvalidation();
      }
      if (countMiss && ctx.run) ctx.run.recordOpfsMiss();
      return null;
    }

    if (probeSize !== null && cachedRawBytes.byteLength !== probeSize) {
      await this._deleteOpfsKeys([candidateDataKey, baseFp.metaKey]);
      if (ctx.run) {
        ctx.run.recordOpfsInvalidation();
        if (countMiss) ctx.run.recordOpfsMiss();
      }
      return null;
    }

    let targetBytes;
    let shared = false;
    if (this.targetBuffer !== null || this.useSharedArrayBuffer) {
      const dest = this._allocate(cachedRawBytes.byteLength);
      dest.bytes.set(cachedRawBytes, 0);
      targetBytes = dest.bytes;
      shared = dest.shared;
    } else {
      const isTightArrayBuffer =
        cachedRawBytes.buffer instanceof ArrayBuffer &&
        cachedRawBytes.byteOffset === 0 &&
        cachedRawBytes.byteLength === cachedRawBytes.buffer.byteLength;
      targetBytes = isTightArrayBuffer ? cachedRawBytes : cachedRawBytes.slice();
      shared = false;
    }

    let layout;
    try {
      layout = parsePgimLayout(targetBytes, { verifyChecksums });
    } catch {
      await this._deleteOpfsKeys([candidateDataKey, baseFp.metaKey]);
      if (ctx.run) {
        ctx.run.recordOpfsInvalidation();
        if (countMiss) ctx.run.recordOpfsMiss();
      }
      return null;
    }

    const total = targetBytes.byteLength;
    if (ctx.run) {
      if (ctx.ownsTotalBytes) ctx.run.setTotalBytes(total);
      ctx.run.recordOpfsHit(total);
      ctx.run.recordBytes(total);
    }
    ctx.emit('cache-hit', total, total, { opfsCacheHit: true });
    return {
      bytes: targetBytes,
      shared,
      layout,
      transport: 'opfs-cache',
      chunksFetched: 0,
      parallelMode: false,
      opfsCacheHit: true,
      etag: candidateEtag,
      hasOwnEtag: true,
      lastModified: candidateLastModified,
      cacheKey: candidateDataKey,
      cacheFingerprint: candidateFingerprint,
      versionVerified: true,
      validatorStrength: validatorStrengthForEtag(candidateEtag),
      versionProvenance: 'opfs-cache',
    };
  }

  _scheduleOpfsWrite(run, { url, etag, lastModified = null, bytes, totalBytes, provenance = null }) {
    const store = this._opfsStore;
    if (!store || !bytes || !(totalBytes > 0)) return;
    if (bytes.byteLength !== totalBytes || (bytes.buffer && bytes.buffer.detached)) return;
    if (!provenance || provenance.verified !== true) return;
    if (isWeakEtag(etag)) return;
    const fp = buildPgimOpfsFingerprint(url, etag, totalBytes, this.opfsCacheVersion);
    const normLastModified = normalizeLastModified(lastModified);
    const taggedProvenance = {
      schema: PGIM_OPFS_PROVENANCE_SCHEMA,
      verified: true,
      source: typeof provenance.source === 'string' && provenance.source.length > 0 ? provenance.source : 'verified',
      validatorStrength: validatorStrengthForEtag(fp.etag),
    };
    // Synchronously clone a detached-safe non-SAB copy so immediate Worker zero-copy transfer
    // (`transferPreloadPayloadToWorker`) or `releaseBootBundleBytes()` cannot detach/corrupt the
    // background OPFS write. Dropped immediately in `finally` to preserve W258 zero-retention.
    let payloadToWrite;
    try {
      payloadToWrite = bytes.slice();
    } catch {
      return;
    }
    let metaPayload = new TextEncoder().encode(
      JSON.stringify({
        version: this.opfsCacheVersion,
        urlPath: fp.urlPath,
        etag: fp.etag,
        lastModified: normLastModified,
        byteLength: totalBytes,
        fingerprint: fp.fingerprint,
        dataKey: fp.dataKey,
        provenance: taggedProvenance,
      })
    );
    const dataMeta = {
      kind: 'pgim',
      version: this.opfsCacheVersion,
      urlPath: fp.urlPath,
      etag: fp.etag,
      lastModified: normLastModified,
      byteLength: totalBytes,
      fingerprint: fp.fingerprint,
      provenance: taggedProvenance,
    };
    const metaInfo = {
      kind: 'pgim-meta',
      version: this.opfsCacheVersion,
      urlPath: fp.urlPath,
      etag: fp.etag,
      lastModified: normLastModified,
      byteLength: totalBytes,
      fingerprint: fp.fingerprint,
      provenance: taggedProvenance,
    };

    const cleanupKeys = async (keys) => {
      if (typeof store.delete !== 'function') return;
      for (let i = 0; i < keys.length; i += 1) {
        try {
          await store.delete(keys[i]);
        } catch {
          // Ignore cleanup failure.
        }
      }
    };

    const runWrite = async () => {
      let dataToWrite = payloadToWrite;
      payloadToWrite = null;
      let metaToWrite = metaPayload;
      metaPayload = null;
      try {
        let writeRes;
        try {
          writeRes = await store.write(fp.dataKey, dataToWrite, dataMeta);
        } catch (dataErr) {
          dataToWrite = null;
          await cleanupKeys([fp.dataKey]);
          throw dataErr;
        }
        dataToWrite = null;
        const stored =
          writeRes !== false &&
          !(writeRes && typeof writeRes === 'object' && writeRes.stored === false);
        if (!stored) {
          await cleanupKeys([fp.dataKey]);
          return;
        }
        try {
          const metaWriteRes = await store.write(fp.metaKey, metaToWrite, metaInfo);
          metaToWrite = null;
          const metaStored =
            metaWriteRes !== false &&
            !(metaWriteRes && typeof metaWriteRes === 'object' && metaWriteRes.stored === false);
          if (!metaStored) {
            await cleanupKeys([fp.dataKey, fp.metaKey]);
            return;
          }
        } catch (metaErr) {
          metaToWrite = null;
          await cleanupKeys([fp.dataKey, fp.metaKey]);
          throw metaErr;
        }
        if (run) run.recordOpfsWrite();
      } finally {
        dataToWrite = null;
        metaToWrite = null;
      }
    };

    const task = runWrite()
      .catch((err) => {
        // Non-blocking persistence: swallow QuotaExceededError and transient storage errors.
        void isQuotaExceededError(err);
      })
      .finally(() => {
        this._pendingOpfsWrites.delete(task);
      });
    this._pendingOpfsWrites.add(task);
  }

  _scheduleManifestOpfsWrite(run, { url, etag, lastModified = null, bytes }) {
    const store = this._opfsStore;
    if (!store || !bytes || !(bytes.byteLength > 0)) return;
    if (bytes.buffer && bytes.buffer.detached) return;
    const totalBytes = bytes.byteLength;
    const fp = buildManifestOpfsFingerprint(url, etag, totalBytes, this.opfsCacheVersion);
    const normLastModified = normalizeLastModified(lastModified);
    let payloadToWrite;
    try {
      payloadToWrite = bytes.slice();
    } catch {
      return;
    }
    let metaPayload = new TextEncoder().encode(
      JSON.stringify({
        kind: 'manifest',
        version: this.opfsCacheVersion,
        urlPath: fp.urlPath,
        etag: fp.etag,
        lastModified: normLastModified,
        byteLength: totalBytes,
        fingerprint: fp.fingerprint,
        dataKey: fp.dataKey,
      })
    );
    const dataMeta = {
      kind: 'manifest',
      version: this.opfsCacheVersion,
      urlPath: fp.urlPath,
      etag: fp.etag,
      lastModified: normLastModified,
      byteLength: totalBytes,
      fingerprint: fp.fingerprint,
    };
    const metaInfo = {
      kind: 'manifest-meta',
      version: this.opfsCacheVersion,
      urlPath: fp.urlPath,
      etag: fp.etag,
      lastModified: normLastModified,
      byteLength: totalBytes,
      fingerprint: fp.fingerprint,
    };

    const runWrite = async () => {
      let dataToWrite = payloadToWrite;
      payloadToWrite = null;
      let metaToWrite = metaPayload;
      metaPayload = null;
      try {
        const writeRes = await store.write(fp.dataKey, dataToWrite, dataMeta);
        dataToWrite = null;
        const stored =
          writeRes !== false &&
          !(writeRes && typeof writeRes === 'object' && writeRes.stored === false);
        if (!stored) return;
        try {
          await store.write(fp.metaKey, metaToWrite, metaInfo);
        } catch (metaErr) {
          if (typeof store.delete === 'function') {
            try {
              await store.delete(fp.dataKey);
            } catch {
              // Ignore cleanup failure.
            }
          }
          throw metaErr;
        }
        if (run) run.recordOpfsWrite();
      } finally {
        dataToWrite = null;
        metaToWrite = null;
      }
    };

    const task = runWrite()
      .catch((err) => {
        void isQuotaExceededError(err);
      })
      .finally(() => {
        this._pendingOpfsWrites.delete(task);
      });
    this._pendingOpfsWrites.add(task);
  }

  async _fetchManifestWithOpfs(url, label, parseFn, { telemetryRun = null, signal = null, opfsValidateWithHead = this.opfsValidateWithHead } = {}) {
    throwIfAborted(signal);
    const store = this._opfsStore;
    if (!store) {
      const ctx = { url, signal, emit: () => {}, requests: 0, run: telemetryRun, ownsTotalBytes: false };
      const response = await this._request(ctx, { method: 'GET' });
      if (!response.ok) throw httpError(url, response.status, label);
      const bytes = new Uint8Array(await raceAbort(response.arrayBuffer(), signal));
      throwIfAborted(signal);
      if (telemetryRun && bytes.byteLength > 0) {
        telemetryRun.recordNetworkBytes(bytes.byteLength);
      }
      const parsed = parseFn(bytes, { baseUrl: url, manifestUrl: url, url });
      return {
        parsed,
        requests: ctx.requests,
        opfsCacheHit: false,
        etag: normalizeEtag(headerValue(response, 'etag')),
        lastModified: normalizeLastModified(headerValue(response, 'last-modified')),
      };
    }

    let requests = 0;
    const baseFp = buildManifestOpfsFingerprint(url, null, null, this.opfsCacheVersion);
    let meta = await this._readOpfsMeta(baseFp.metaKey, signal, 'manifest-meta');

    if (meta !== null) {
      const validMetaShape =
        !meta._corrupt &&
        meta.version === this.opfsCacheVersion &&
        meta.urlPath === baseFp.urlPath &&
        typeof meta.dataKey === 'string' &&
        meta.dataKey.length > 0 &&
        Number.isSafeInteger(meta.byteLength) &&
        meta.byteLength > 0;
      if (!validMetaShape) {
        const keysToDelete = [baseFp.metaKey];
        if (typeof meta.dataKey === 'string' && meta.dataKey.length > 0) {
          keysToDelete.push(meta.dataKey);
        }
        await this._deleteOpfsKeys(keysToDelete);
        if (telemetryRun) telemetryRun.recordOpfsInvalidation();
        meta = null;
      }
    }

    const tryReadAndParseCached = async (cachedMeta, expectedSize = null) => {
      let rawData;
      try {
        rawData = await raceAbort(
          Promise.resolve().then(() =>
            store.read(cachedMeta.dataKey, {
              kind: 'manifest',
              version: this.opfsCacheVersion,
              urlPath: baseFp.urlPath,
              etag: normalizeEtag(cachedMeta.etag),
              byteLength: expectedSize ?? cachedMeta.byteLength,
              fingerprint: cachedMeta.fingerprint || null,
            })
          ),
          signal
        );
      } catch (err) {
        if ((signal && signal.aborted) || isAbortError(err)) throw err;
        return null;
      }
      throwIfAborted(signal);
      const cachedBytes = extractCachedBytes(rawData);
      if (!cachedBytes || cachedBytes.byteLength === 0) {
        await this._deleteOpfsKeys([cachedMeta.dataKey, baseFp.metaKey]);
        if (telemetryRun) telemetryRun.recordOpfsInvalidation();
        return null;
      }
      const checkSize = expectedSize !== null ? expectedSize : cachedMeta.byteLength;
      if (checkSize !== null && cachedBytes.byteLength !== checkSize) {
        await this._deleteOpfsKeys([cachedMeta.dataKey, baseFp.metaKey]);
        if (telemetryRun) telemetryRun.recordOpfsInvalidation();
        return null;
      }
      let parsed;
      try {
        parsed = parseFn(cachedBytes, { baseUrl: url, manifestUrl: url, url });
      } catch {
        await this._deleteOpfsKeys([cachedMeta.dataKey, baseFp.metaKey]);
        if (telemetryRun) telemetryRun.recordOpfsInvalidation();
        return null;
      }
      if (telemetryRun) {
        telemetryRun.recordOpfsHit(cachedBytes.byteLength);
      }
      return {
        parsed,
        requests,
        opfsCacheHit: true,
        etag: normalizeEtag(cachedMeta.etag),
        lastModified: normalizeLastModified(cachedMeta.lastModified),
      };
    };

    if (meta !== null) {
      if (!opfsValidateWithHead) {
        const fastHit = await tryReadAndParseCached(meta, meta.byteLength);
        if (fastHit !== null) return fastHit;
        meta = null;
      } else {
        const metaEtag = normalizeEtag(meta.etag);
        const metaLastModified = normalizeLastModified(meta.lastModified);
        const headCtx = { url, signal, emit: () => {}, requests: 0, run: telemetryRun, ownsTotalBytes: false };
        let headResp = null;
        try {
          headResp = await this._request(headCtx, { method: 'HEAD' });
          requests += headCtx.requests;
        } catch (err) {
          requests += headCtx.requests;
          if ((signal && signal.aborted) || isAbortError(err)) throw err;
          headResp = null;
        }

        let needsConditionalGet = false;
        if (headResp && (headResp.status === 404 || headResp.status === 410)) {
          await this._deleteOpfsKeys([meta.dataKey, baseFp.metaKey]);
          if (telemetryRun) telemetryRun.recordOpfsInvalidation();
          throw httpError(url, headResp.status, label);
        } else if (headResp && headResp.ok) {
          const headEtag = normalizeEtag(headerValue(headResp, 'etag'));
          const headSize = contentLength(headResp);
          const headLastMod = normalizeLastModified(headerValue(headResp, 'last-modified'));
          const hasHeadValidators = headEtag !== null || headSize !== null || headLastMod !== null;
          let mismatch = false;
          if (headSize !== null && meta.byteLength !== headSize) {
            mismatch = true;
          }
          if (headEtag !== null && metaEtag !== headEtag) {
            mismatch = true;
          } else if (headEtag === null && metaEtag === null && metaLastModified !== headLastMod) {
            mismatch = true;
          }
          if (mismatch) {
            await this._deleteOpfsKeys([meta.dataKey, baseFp.metaKey]);
            if (telemetryRun) telemetryRun.recordOpfsInvalidation();
            meta = null;
          } else if (hasHeadValidators && !(headEtag === null && metaEtag !== null)) {
            const headHit = await tryReadAndParseCached(meta, headSize !== null ? headSize : meta.byteLength);
            if (headHit !== null) return headHit;
            meta = null;
          } else {
            needsConditionalGet = true;
          }
        } else {
          // HEAD unsupported (e.g. 405) or transient error: validate via conditional GET.
          needsConditionalGet = true;
        }

        if (meta !== null && needsConditionalGet) {
          const condHeaders = {};
          if (metaEtag !== null) condHeaders['If-None-Match'] = metaEtag;
          if (metaLastModified !== null) condHeaders['If-Modified-Since'] = metaLastModified;
          const getCtx = { url, signal, emit: () => {}, requests: 0, run: telemetryRun, ownsTotalBytes: false };
          const init =
            Object.keys(condHeaders).length > 0 ? { method: 'GET', headers: condHeaders } : { method: 'GET' };
          const condResp = await this._request(getCtx, init);
          requests += getCtx.requests;
          if (condResp.status === 304) {
            const condHit = await tryReadAndParseCached(meta, meta.byteLength);
            if (condHit !== null) return condHit;
            meta = null;
          } else if (!condResp.ok) {
            if (condResp.status === 404 || condResp.status === 410) {
              await this._deleteOpfsKeys([meta.dataKey, baseFp.metaKey]);
              if (telemetryRun) telemetryRun.recordOpfsInvalidation();
            }
            throw httpError(url, condResp.status, label);
          } else {
            const bytes = new Uint8Array(await raceAbort(condResp.arrayBuffer(), signal));
            throwIfAborted(signal);
            const respEtag = normalizeEtag(headerValue(condResp, 'etag'));
            const respLastMod = normalizeLastModified(headerValue(condResp, 'last-modified'));
            const changed =
              bytes.byteLength !== meta.byteLength ||
              (respEtag !== null && respEtag !== metaEtag) ||
              (respEtag === null && respLastMod !== null && respLastMod !== metaLastModified);
            if (changed) {
              await this._deleteOpfsKeys([meta.dataKey, baseFp.metaKey]);
              if (telemetryRun) telemetryRun.recordOpfsInvalidation();
            }
            if (telemetryRun) {
              telemetryRun.recordOpfsMiss();
              telemetryRun.recordNetworkBytes(bytes.byteLength);
            }
            const parsed = parseFn(bytes, { baseUrl: url, manifestUrl: url, url });
            if (!(signal && signal.aborted)) {
              this._scheduleManifestOpfsWrite(telemetryRun, {
                url,
                etag: respEtag,
                lastModified: respLastMod,
                bytes,
              });
            }
            return {
              parsed,
              requests,
              opfsCacheHit: false,
              etag: respEtag,
              lastModified: respLastMod,
            };
          }
        }
      }
    }

    const getCtx = { url, signal, emit: () => {}, requests: 0, run: telemetryRun, ownsTotalBytes: false };
    const response = await this._request(getCtx, { method: 'GET' });
    requests += getCtx.requests;
    if (!response.ok) throw httpError(url, response.status, label);
    const bytes = new Uint8Array(await raceAbort(response.arrayBuffer(), signal));
    throwIfAborted(signal);
    if (telemetryRun) {
      telemetryRun.recordOpfsMiss();
      telemetryRun.recordNetworkBytes(bytes.byteLength);
    }
    const parsed = parseFn(bytes, { baseUrl: url, manifestUrl: url, url });
    const respEtag = normalizeEtag(headerValue(response, 'etag'));
    const respLastMod = normalizeLastModified(headerValue(response, 'last-modified'));
    if (!(signal && signal.aborted)) {
      this._scheduleManifestOpfsWrite(telemetryRun, {
        url,
        etag: respEtag,
        lastModified: respLastMod,
        bytes,
      });
    }
    return {
      parsed,
      requests,
      opfsCacheHit: false,
      etag: respEtag,
      lastModified: respLastMod,
    };
  }

  async _preloadGuestImageCore(
    url,
    {
      onProgress,
      signal,
      verifyChecksums = false,
      chunkSize = this.chunkSize,
      concurrency = this.concurrency,
      precomputedProbe = null,
      emitProbe = true,
      emitComplete = true,
      telemetryRun = null,
      ownsTotalBytes = true,
      opfsValidateWithHead = this.opfsValidateWithHead,
      expectedEtag = null,
      expectedBytes = null,
    } = {}
  ) {
    if (typeof url !== 'string' || url.length === 0) throw new TypeError('PGIM url must be a non-empty string');
    throwIfAborted(signal);
    const started = this._now();
    const emit = makeEmitter(onProgress);
    const ctx = {
      url,
      signal,
      emit,
      requests: 0,
      run: telemetryRun,
      ownsTotalBytes,
      chunkSize,
      concurrency,
      what: 'PGIM image',
    };

    let probe = precomputedProbe;
    let loaded = null;

    // Optional cached-metadata fast path when HEAD validation is explicitly disabled.
    if (this._opfsStore && probe === null && !opfsValidateWithHead) {
      loaded = await this._tryLoadFromOpfsCache(ctx, null, {
        verifyChecksums,
        expectedEtag,
        expectedBytes,
        countMiss: false,
      });
    }

    if (loaded === null) {
      if (probe === null) {
        if (emitProbe) emit('probe', 0, null);
        probe = await this._probe(ctx);
      }
      if (this._opfsStore) {
        loaded = await this._tryLoadFromOpfsCache(ctx, probe, {
          verifyChecksums,
          expectedEtag,
          expectedBytes,
          countMiss: true,
        });
      }
    }

    if (loaded === null) {
      let versionAttempt = 0;
      for (;;) {
        throwIfAborted(signal);
        if (versionAttempt > 0) {
          probe = await this._probe(ctx);
          if (this._opfsStore) {
            loaded = await this._tryLoadFromOpfsCache(ctx, probe, {
              verifyChecksums,
              expectedEtag,
              expectedBytes,
              countMiss: false,
            });
            if (loaded !== null) break;
          }
        }
        const preAttemptBytesLoaded = ctx.run ? ctx.run.bytesLoaded : 0;
        const preAttemptChunksFetched = ctx.run ? ctx.run.chunksFetched : 0;
        try {
          if (this.rangeCache) {
            loaded = await this._loadViaRangeCache(ctx, probe);
          } else if (probe.acceptRanges === false) {
            loaded = await this._fetchFullCoherent(ctx, probe);
          } else {
            loaded = await this._loadRanged(ctx, probe);
          }
          break;
        } catch (err) {
          if ((signal && signal.aborted) || isAbortError(err)) throw err;
          const isRetryableVersion =
            err instanceof PgimPreloadError && err.retryableVersionChange === true;
          if (!isRetryableVersion) throw err;
          if (ctx.run) {
            ctx.run.bytesLoaded = preAttemptBytesLoaded;
            ctx.run.chunksFetched = preAttemptChunksFetched;
          }
          if (versionAttempt < this.maxVersionRetries) {
            versionAttempt += 1;
            if (ctx.run) ctx.run.recordRetry();
            continue;
          }
          throw new PgimPreloadError(
            err.code,
            `${err.message} (exhausted ${this.maxVersionRetries} version retry attempts)`,
            {
              url,
              cause: err,
              expectedEtag: err.expectedEtag ?? null,
              observedEtag: err.observedEtag ?? null,
              status: err.status,
              versionRetries: versionAttempt,
            }
          );
        }
      }
    }
    throwIfAborted(signal);

    const layout = loaded.layout || parsePgimLayout(loaded.bytes, { verifyChecksums });
    const totalBytes = loaded.bytes.byteLength;
    const effectiveEtag = loaded.hasOwnEtag
      ? (loaded.etag ?? null)
      : loaded.etag !== undefined && loaded.etag !== null
        ? loaded.etag
        : probe && probe.etag !== undefined
          ? probe.etag
          : null;
    const effectiveLastModified = loaded.hasOwnEtag
      ? (loaded.lastModified ?? null)
      : loaded.lastModified !== undefined && loaded.lastModified !== null
        ? loaded.lastModified
        : probe && probe.lastModified !== undefined
          ? probe.lastModified
          : null;
    const opfsCacheHit = Boolean(loaded.opfsCacheHit);
    let versionVerified = Boolean(loaded.versionVerified);
    let versionProvenance = loaded.versionProvenance || null;
    if (!versionVerified && loaded.transport === 'range-cache' && verifyChecksums) {
      versionVerified = true;
      versionProvenance = 'checksum-verified-range-cache';
    }
    const validatorStrength = loaded.validatorStrength || validatorStrengthForEtag(effectiveEtag);

    if (
      !opfsCacheHit &&
      this._opfsStore &&
      !(signal && signal.aborted) &&
      versionVerified &&
      !isWeakEtag(effectiveEtag)
    ) {
      this._scheduleOpfsWrite(ctx.run, {
        url,
        etag: effectiveEtag,
        lastModified: effectiveLastModified,
        bytes: loaded.bytes,
        totalBytes,
        provenance: {
          schema: PGIM_OPFS_PROVENANCE_SCHEMA,
          verified: true,
          source: versionProvenance || 'verified',
          validatorStrength,
        },
      });
    }

    const durationMs = this._elapsedSince(started);
    if (emitComplete) {
      emit('complete', totalBytes, totalBytes, opfsCacheHit ? { opfsCacheHit: true } : null);
    }
    return {
      bytes: loaded.bytes,
      buffer: loaded.bytes.buffer,
      header: layout.header,
      layout,
      shared: loaded.shared,
      transport: loaded.transport,
      opfsCacheHit,
      etag: effectiveEtag,
      versionVerified,
      validatorStrength,
      versionProvenance,
      cacheKey: loaded.cacheKey || null,
      cacheFingerprint: loaded.cacheFingerprint || null,
      requests: ctx.requests,
      chunksFetched: loaded.chunksFetched,
      parallelMode: loaded.parallelMode,
      totalBytes,
      bytesLoaded: totalBytes,
      timingMs: durationMs,
      durationMs,
      elapsedMs: durationMs,
    };
  }

  /**
   * Discover and preload all guest `.pgim` modules listed in a multi-module `program-manifest.json`
   * with unified aggregate byte-level progress (`loadedBytes`, `totalBytes`, `moduleIndex`,
   * `moduleCount`, `currentModule`). Gracefully falls back to single-module `boot-manifest.json`
   * (or an optional `fallbackPgimUrl`) when `program-manifest.json` returns HTTP 404/410.
   *
   * @param {string | { programManifestUrl?: string, manifestUrl?: string, bootManifestUrl?: string | null,
   *                   fallbackPgimUrl?: string | null, onProgress?: Function, signal?: AbortSignal,
   *                   verifyChecksums?: boolean, chunkSize?: number, chunkBytes?: number, concurrency?: number }} manifestUrl
   * @param {{ bootManifestUrl?: string | null, fallbackPgimUrl?: string | null,
   *           onProgress?: (e: object) => void, signal?: AbortSignal, verifyChecksums?: boolean,
   *           chunkSize?: number, chunkBytes?: number, concurrency?: number }} [options]
   */
  async preloadProgramModules(manifestUrl, options = {}) {
    let requestedUrl = manifestUrl;
    let opts = options;
    if (isPlainObject(manifestUrl)) {
      opts = manifestUrl;
      requestedUrl = opts.programManifestUrl ?? opts.manifestUrl;
    }
    validateManifestEndpointUrl(requestedUrl, 'program manifest URL');
    const transfer = this._resolveTransferConfig(opts || {});
    const run = this._startTelemetryRun();
    try {
      const result = await this._preloadProgramModulesCore(
        requestedUrl,
        { ...(opts || {}), chunkSize: transfer.chunkSize, concurrency: transfer.concurrency },
        run
      );
      run.complete(result.totalBytes);
      const stats = run.snapshot();
      return {
        ...result,
        chunksFetched: stats.chunksFetched,
        elapsedMs: stats.elapsedMs,
        bytesPerSec: stats.bytesPerSec,
        peakBytesPerSec: stats.peakBytesPerSec,
        throughputMBps: stats.throughputMBps,
        parallelMode: stats.parallelMode,
        retryCount: stats.retryCount,
        abortedCount: stats.abortedCount,
      };
    } catch (err) {
      this._finishTelemetryError(run, err, opts && opts.signal);
      throw err;
    }
  }

  async _preloadProgramModulesCore(requestedUrl, opts = {}, telemetryRun = null) {
    const {
      bootManifestUrl = undefined,
      fallbackPgimUrl = null,
      onProgress,
      signal,
      verifyChecksums = false,
      chunkSize = this.chunkSize,
      concurrency = this.concurrency,
      opfsValidateWithHead,
      validateWithHead,
    } = opts || {};

    const effectiveValidateWithHead =
      opfsValidateWithHead !== undefined
        ? Boolean(opfsValidateWithHead)
        : validateWithHead !== undefined
          ? Boolean(validateWithHead)
          : this.opfsValidateWithHead;

    validateManifestEndpointUrl(requestedUrl, 'program manifest URL');
    throwIfAborted(signal);

    const started = this._now();
    let totalRequests = 0;
    const emitEvent = (payload) => {
      if (typeof onProgress === 'function') onProgress(payload);
    };

    emitEvent({
      phase: 'manifest',
      loadedBytes: 0,
      totalBytes: null,
      pct: 0,
      moduleIndex: 0,
      moduleCount: 0,
      currentModule: null,
      moduleUrl: null,
      moduleLoadedBytes: 0,
      moduleTotalBytes: null,
    });

    let parsedManifest = null;
    let source = 'program-manifest';
    let resolvedManifestUrl = requestedUrl;
    let usedFallback = false;
    let manifestOpfsCacheHit = false;

    try {
      const fetched = await this._fetchManifestWithOpfs(
        requestedUrl,
        'program manifest',
        parseProgramManifest,
        {
          telemetryRun,
          signal,
          opfsValidateWithHead: effectiveValidateWithHead,
        }
      );
      totalRequests += fetched.requests;
      parsedManifest = fetched.parsed;
      manifestOpfsCacheHit = Boolean(fetched.opfsCacheHit);
    } catch (err) {
      if (!(err instanceof PgimPreloadError && err.code === 'NOT_FOUND')) {
        throw err;
      }

      let resolvedBootUrl = null;
      if (bootManifestUrl !== null && bootManifestUrl !== false) {
        resolvedBootUrl =
          typeof bootManifestUrl === 'string'
            ? validateManifestEndpointUrl(bootManifestUrl, 'boot manifest URL')
            : deriveBootManifestUrl(requestedUrl);
      }

      if (resolvedBootUrl !== null) {
        emitEvent({
          phase: 'fallback',
          loadedBytes: 0,
          totalBytes: null,
          pct: 0,
          moduleIndex: 0,
          moduleCount: 0,
          currentModule: null,
          moduleUrl: null,
          moduleLoadedBytes: 0,
          moduleTotalBytes: null,
          fallbackFrom: requestedUrl,
          fallbackTo: resolvedBootUrl,
        });
        try {
          const bootFetched = await this._fetchManifestWithOpfs(
            resolvedBootUrl,
            'boot manifest',
            parseBootManifest,
            {
              telemetryRun,
              signal,
              opfsValidateWithHead: effectiveValidateWithHead,
            }
          );
          totalRequests += bootFetched.requests;
          parsedManifest = bootFetched.parsed;
          manifestOpfsCacheHit = Boolean(bootFetched.opfsCacheHit);
          source = 'boot-manifest';
          resolvedManifestUrl = resolvedBootUrl;
          usedFallback = true;
        } catch (bootErr) {
          if (
            bootErr instanceof PgimPreloadError &&
            bootErr.code === 'NOT_FOUND' &&
            typeof fallbackPgimUrl === 'string' &&
            fallbackPgimUrl.length > 0
          ) {
            const safeUrl = validateManifestEndpointUrl(fallbackPgimUrl, 'fallback PGIM URL');
            const fallbackPathPart = safeUrl.split(/[?#]/, 1)[0];
            if (!fallbackPathPart.toLowerCase().endsWith('.pgim')) {
              throw unsafeUrlError(fallbackPgimUrl, 'fallback PGIM URL must reference a .pgim file');
            }
            parsedManifest = {
              schema: 'pgim-fallback',
              schemaVersion: 1,
              baseUrl: '',
              modules: [
                {
                  name: 'main',
                  module: 'main',
                  url: safeUrl,
                  path: fallbackPathPart.replace(/^\/+/, ''),
                  role: 'exe',
                  dll: null,
                  file: null,
                  base: null,
                  sizeOfImage: null,
                  entryPoint: null,
                  expectedBytes: null,
                  buildManifestUrl: null,
                },
              ],
              initOrder: ['main'],
              raw: null,
            };
            source = 'pgim-fallback';
            resolvedManifestUrl = null;
            usedFallback = true;
          } else {
            throw bootErr;
          }
        }
      } else if (typeof fallbackPgimUrl === 'string' && fallbackPgimUrl.length > 0) {
        const safeUrl = validateManifestEndpointUrl(fallbackPgimUrl, 'fallback PGIM URL');
        const fallbackPathPart = safeUrl.split(/[?#]/, 1)[0];
        if (!fallbackPathPart.toLowerCase().endsWith('.pgim')) {
          throw unsafeUrlError(fallbackPgimUrl, 'fallback PGIM URL must reference a .pgim file');
        }
        emitEvent({
          phase: 'fallback',
          loadedBytes: 0,
          totalBytes: null,
          pct: 0,
          moduleIndex: 0,
          moduleCount: 0,
          currentModule: null,
          moduleUrl: null,
          moduleLoadedBytes: 0,
          moduleTotalBytes: null,
          fallbackFrom: requestedUrl,
          fallbackTo: safeUrl,
        });
        parsedManifest = {
          schema: 'pgim-fallback',
          schemaVersion: 1,
          baseUrl: '',
          modules: [
            {
              name: 'main',
              module: 'main',
              url: safeUrl,
              path: fallbackPathPart.replace(/^\/+/, ''),
              role: 'exe',
              dll: null,
              file: null,
              base: null,
              sizeOfImage: null,
              entryPoint: null,
              expectedBytes: null,
              buildManifestUrl: null,
            },
          ],
          initOrder: ['main'],
          raw: null,
        };
        source = 'pgim-fallback';
        resolvedManifestUrl = null;
        usedFallback = true;
      } else {
        throw err;
      }
    }

    const descriptors = parsedManifest.modules;
    const moduleCount = descriptors.length;
    if (this.targetBuffer !== null && moduleCount > 1) {
      throw new PgimPreloadError(
        'ALLOCATION_FAILED',
        `targetBuffer cannot be shared across ${moduleCount} PGIM modules`
      );
    }

    // Pre-probe every module with HEAD (bounded by concurrency) to learn aggregate totalBytes upfront.
    // When OPFS no-HEAD fast path is enabled, read cached byteLength from OPFS metadata when available.
    const probes = new Array(moduleCount).fill(null);
    const cachedSizes = new Array(moduleCount).fill(null);
    const { controller: probeController, unlink: unlinkProbe } = linkSignal(signal);
    let nextProbe = 0;
    let probeFailure = null;
    const probeWorker = async () => {
      while (probeFailure === null && nextProbe < moduleCount) {
        const idx = nextProbe;
        nextProbe += 1;
        if (this._opfsStore && !effectiveValidateWithHead) {
          const baseFp = buildPgimOpfsFingerprint(descriptors[idx].url, null, null, this.opfsCacheVersion);
          const meta = await this._readOpfsMeta(baseFp.metaKey, probeController.signal, 'pgim-meta');
          if (
            meta &&
            !meta._corrupt &&
            meta.version === this.opfsCacheVersion &&
            meta.urlPath === baseFp.urlPath &&
            Number.isSafeInteger(meta.byteLength) &&
            meta.byteLength >= PGIM_HEADER_BYTES &&
            hasVerifiedOpfsProvenance(meta) &&
            !isWeakEtag(meta.etag)
          ) {
            cachedSizes[idx] = meta.byteLength;
            probes[idx] = null;
            continue;
          }
        }
        const pCtx = {
          url: descriptors[idx].url,
          signal: probeController.signal,
          emit: () => {},
          requests: 0,
          run: telemetryRun,
          ownsTotalBytes: false,
        };
        try {
          probes[idx] = await this._probe(pCtx);
          totalRequests += pCtx.requests;
        } catch (err) {
          totalRequests += pCtx.requests;
          if (probeFailure === null) {
            probeFailure = err;
            probeController.abort(err);
          }
          return;
        }
      }
    };
    try {
      await Promise.all(Array.from({ length: Math.min(concurrency, moduleCount) }, probeWorker));
    } finally {
      unlinkProbe();
    }
    throwIfAborted(signal);
    if (probeFailure !== null) throw probeFailure;

    const moduleLoaded = new Array(moduleCount).fill(0);
    const moduleTotals = probes.map((p, idx) =>
      p && typeof p.totalBytes === 'number' && p.totalBytes > 0
        ? p.totalBytes
        : cachedSizes[idx] !== null
          ? cachedSizes[idx]
          : null
    );

    const computeAggregate = () => {
      let loadedBytes = 0;
      let totalBytes = 0;
      let allTotalsKnown = true;
      for (let i = 0; i < moduleCount; i += 1) {
        loadedBytes += moduleLoaded[i];
        if (moduleTotals[i] === null) {
          allTotalsKnown = false;
        } else {
          totalBytes += moduleTotals[i];
        }
      }
      const aggTotal = allTotalsKnown ? totalBytes : null;
      const ratio = aggTotal ? Math.min(1, Math.max(0, loadedBytes / aggTotal)) : 0;
      const pct = Math.min(100, ratio * 100);
      return { loadedBytes, totalBytes: aggTotal, ratio, pct };
    };

    let maxEmittedLoaded = 0;
    let maxEmittedRatio = 0;
    let maxEmittedPct = 0;
    const emitAggregate = (phase, moduleIndex, descriptor, extra = null) => {
      const agg = computeAggregate();
      if (telemetryRun && agg.totalBytes !== null) {
        telemetryRun.setTotalBytes(agg.totalBytes);
      }
      const loadedBytes = Math.max(maxEmittedLoaded, agg.loadedBytes);
      const ratio = agg.totalBytes !== null ? Math.max(maxEmittedRatio, agg.ratio) : maxEmittedRatio;
      const pct = agg.totalBytes !== null ? Math.max(maxEmittedPct, agg.pct) : maxEmittedPct;
      maxEmittedLoaded = loadedBytes;
      maxEmittedRatio = ratio;
      maxEmittedPct = pct;
      const payload = {
        phase,
        loadedBytes,
        totalBytes: agg.totalBytes,
        ratio,
        pct,
        moduleIndex,
        moduleCount,
        currentModule: descriptor.name,
        moduleUrl: descriptor.url,
        moduleLoadedBytes: moduleLoaded[moduleIndex],
        moduleTotalBytes: moduleTotals[moduleIndex],
      };
      emitEvent(extra ? { ...payload, ...extra } : payload);
    };

    emitAggregate('probe', 0, descriptors[0]);

    const loadedModules = [];
    const modulesByName = {};

    for (let i = 0; i < moduleCount; i += 1) {
      throwIfAborted(signal);
      const descriptor = descriptors[i];
      const modResult = await this._preloadGuestImageCore(descriptor.url, {
        signal,
        verifyChecksums,
        chunkSize,
        concurrency,
        precomputedProbe: probes[i],
        emitProbe: false,
        emitComplete: false,
        telemetryRun,
        ownsTotalBytes: false,
        opfsValidateWithHead: effectiveValidateWithHead,
        onProgress: (evt) => {
          if (typeof evt.loadedBytes === 'number' && evt.loadedBytes > moduleLoaded[i]) {
            moduleLoaded[i] = evt.loadedBytes;
          }
          if (typeof evt.totalBytes === 'number' && evt.totalBytes > 0) {
            moduleTotals[i] = evt.totalBytes;
          }
          emitAggregate(evt.phase, i, descriptor, evt.opfsCacheHit ? { opfsCacheHit: true } : null);
        },
      });
      totalRequests += modResult.requests;
      moduleLoaded[i] = modResult.totalBytes;
      moduleTotals[i] = modResult.totalBytes;

      const entry = {
        name: descriptor.name,
        module: descriptor.name,
        url: descriptor.url,
        path: descriptor.path,
        role: descriptor.role,
        dll: descriptor.dll,
        file: descriptor.file,
        index: i,
        header: modResult.header,
        layout: modResult.layout,
        bytes: modResult.bytes,
        buffer: modResult.buffer,
        shared: modResult.shared,
        transport: modResult.transport,
        opfsCacheHit: modResult.opfsCacheHit,
        etag: modResult.etag,
        requests: modResult.requests,
        chunksFetched: modResult.chunksFetched,
        parallelMode: modResult.parallelMode,
        totalBytes: modResult.totalBytes,
        bytesLoaded: modResult.bytesLoaded,
        timingMs: modResult.timingMs,
        durationMs: modResult.durationMs,
        elapsedMs: modResult.elapsedMs,
      };
      loadedModules.push(entry);
      Object.defineProperty(modulesByName, descriptor.name, {
        value: entry,
        enumerable: true,
        configurable: true,
        writable: true,
      });
      emitAggregate('module-complete', i, descriptor, modResult.opfsCacheHit ? { opfsCacheHit: true } : null);
    }

    const totalBytes = loadedModules.reduce((acc, m) => acc + m.totalBytes, 0);
    const durationMs = this._elapsedSince(started);
    const lastIdx = moduleCount - 1;
    const lastDescriptor = descriptors[lastIdx];
    const allOpfsHits = loadedModules.length > 0 && loadedModules.every((m) => m.opfsCacheHit);
    emitEvent({
      phase: 'complete',
      loadedBytes: totalBytes,
      totalBytes,
      ratio: 1,
      pct: 100,
      moduleIndex: lastIdx,
      moduleCount,
      currentModule: lastDescriptor.name,
      moduleUrl: lastDescriptor.url,
      moduleLoadedBytes: loadedModules[lastIdx].totalBytes,
      moduleTotalBytes: loadedModules[lastIdx].totalBytes,
      ...(allOpfsHits ? { opfsCacheHit: true } : {}),
    });

    const mainModule =
      loadedModules.find((m) => m.role === 'exe') ||
      loadedModules.find((m) => m.name.toLowerCase() === 'main') ||
      loadedModules[lastIdx];

    return {
      source,
      manifestUrl: resolvedManifestUrl,
      requestedManifestUrl: requestedUrl,
      usedFallback,
      manifestOpfsCacheHit,
      manifest: parsedManifest.raw !== null ? parsedManifest : null,
      modules: loadedModules,
      modulesByName,
      moduleCount,
      mainModule,
      totalBytes,
      bytesLoaded: totalBytes,
      requests: totalRequests,
      timingMs: durationMs,
      durationMs,
    };
  }

  /**
   * Stream and validate a PGIM v1 guest image. Returns `{ header, buffer, bytesLoaded, durationMs }`.
   *
   * @param {string} url
   * @param {{ onProgress?: Function, signal?: AbortSignal, verifyChecksums?: boolean,
   *           chunkSize?: number, chunkBytes?: number, concurrency?: number }} [options]
   */
  async preloadPgim(url, options = {}) {
    const result = await this.preloadGuestImage(url, options);
    return {
      header: result.header,
      buffer: result.buffer,
      bytesLoaded: result.bytesLoaded,
      totalBytes: result.totalBytes,
      durationMs: result.durationMs,
      elapsedMs: result.elapsedMs,
      layout: result.layout,
      bytes: result.bytes,
      shared: result.shared,
      transport: result.transport,
      opfsCacheHit: result.opfsCacheHit,
      etag: result.etag,
      versionVerified: result.versionVerified,
      validatorStrength: result.validatorStrength,
      versionProvenance: result.versionProvenance,
      cacheKey: result.cacheKey,
      cacheFingerprint: result.cacheFingerprint,
      requests: result.requests,
      chunksFetched: result.chunksFetched,
      parallelMode: result.parallelMode,
      bytesPerSec: result.bytesPerSec,
      peakBytesPerSec: result.peakBytesPerSec,
      throughputMBps: result.throughputMBps,
      retryCount: result.retryCount,
      abortedCount: result.abortedCount,
    };
  }

  /**
   * Stream a `.wgsl.pack` bundle and validate the WGPK magic/version/entry table header.
   *
   * @param {string} url
   * @param {{ onProgress?: Function, signal?: AbortSignal,
   *           chunkSize?: number, chunkBytes?: number, concurrency?: number }} [options]
   */
  async preloadShaderPack(
    url,
    { onProgress, signal, chunkSize, chunkBytes, concurrency, streamShaders, streamThresholdBytes } = {}
  ) {
    if (typeof url !== 'string' || url.length === 0) throw new TypeError('shader pack url must be a non-empty string');
    const transfer = this._resolveTransferConfig({ chunkSize, chunkBytes, concurrency });
    const policy = resolveShaderPackPreloadPolicy({
      streamShaders: streamShaders !== undefined ? streamShaders : this.streamShaders,
      streamThresholdBytes:
        streamThresholdBytes !== undefined ? streamThresholdBytes : this.streamThresholdBytes,
    });
    const run = this._startTelemetryRun();
    try {
      throwIfAborted(signal);
      const started = this._now();
      if (policy.skipFullPreload) {
        if (typeof onProgress === 'function') {
          onProgress({
            phase: 'skipped',
            reason: 'stream-shaders',
            loadedBytes: 0,
            totalBytes: 0,
            pct: 100,
          });
        }
        run.complete(0);
        const stats = run.snapshot();
        return {
          buffer: null,
          bytes: null,
          bytesLoaded: 0,
          totalBytes: 0,
          skipped: true,
          skipReason: 'stream-shaders',
          transport: 'stream-deferred',
          chunksFetched: 0,
          durationMs: this._elapsedSince(started),
          elapsedMs: stats.elapsedMs,
          bytesPerSec: stats.bytesPerSec,
          peakBytesPerSec: stats.peakBytesPerSec,
          throughputMBps: stats.throughputMBps,
          parallelMode: false,
          retryCount: stats.retryCount,
          abortedCount: stats.abortedCount,
        };
      }
      const emit = makeEmitter(onProgress);
      const fetched = await this._fetchAsset(url, 'WGPK shader pack', signal, emit, run, true, {
        allowParallelRange: true,
        chunkSize: transfer.chunkSize,
        concurrency: transfer.concurrency,
        maxPreloadBytes: policy.maxPreloadBytes,
      });
      if (fetched.skipped) {
        const skipReason = fetched.skipReason || 'exceeds-preload-threshold';
        const knownTotal = fetched.totalBytes || 0;
        if (typeof onProgress === 'function') {
          onProgress({
            phase: 'skipped',
            reason: skipReason,
            loadedBytes: 0,
            totalBytes: knownTotal,
            pct: 100,
          });
        }
        run.complete(0);
        const stats = run.snapshot();
        return {
          buffer: null,
          bytes: null,
          bytesLoaded: 0,
          totalBytes: knownTotal,
          skipped: true,
          skipReason,
          transport: 'stream-deferred',
          chunksFetched: 0,
          durationMs: this._elapsedSince(started),
          elapsedMs: stats.elapsedMs,
          bytesPerSec: stats.bytesPerSec,
          peakBytesPerSec: stats.peakBytesPerSec,
          throughputMBps: stats.throughputMBps,
          parallelMode: false,
          retryCount: stats.retryCount,
          abortedCount: stats.abortedCount,
        };
      }
      const validated = validateWgslPackBytes(fetched.bytes, url);
      const isExactBuffer =
        validated.byteOffset === 0 && validated.byteLength === validated.buffer.byteLength;
      const tightBuffer = isExactBuffer ? validated.buffer : validated.slice().buffer;
      const tightBytes = isExactBuffer ? validated : new Uint8Array(tightBuffer);
      run.complete(tightBytes.byteLength);
      const stats = run.snapshot();
      return {
        buffer: tightBuffer,
        bytes: tightBytes,
        bytesLoaded: tightBytes.byteLength,
        totalBytes: tightBytes.byteLength,
        transport: fetched.transport,
        chunksFetched: stats.chunksFetched,
        durationMs: this._elapsedSince(started),
        elapsedMs: stats.elapsedMs,
        bytesPerSec: stats.bytesPerSec,
        peakBytesPerSec: stats.peakBytesPerSec,
        throughputMBps: stats.throughputMBps,
        parallelMode: stats.parallelMode,
        retryCount: stats.retryCount,
        abortedCount: stats.abortedCount,
      };
    } catch (err) {
      this._finishTelemetryError(run, err, signal);
      throw err;
    }
  }

  /**
   * Fetch and validate an S6 reflection / layout manifest JSON document.
   *
   * @param {string} url
   * @param {{ onProgress?: Function, signal?: AbortSignal }} [options]
   */
  async preloadReflectionManifest(url, { onProgress, signal } = {}) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError('reflection manifest url must be a non-empty string');
    }
    const run = this._startTelemetryRun();
    try {
      throwIfAborted(signal);
      const started = this._now();
      const emit = makeEmitter(onProgress);
      const fetched = await this._fetchAsset(url, 'reflection manifest', signal, emit, run, true, {
        allowParallelRange: false,
      });
      const bytes = fetched.bytes;
      const manifest = parseReflectionManifestJson(bytes, url);
      run.complete(bytes.byteLength);
      const stats = run.snapshot();
      return {
        manifest,
        bytes,
        bytesLoaded: bytes.byteLength,
        totalBytes: bytes.byteLength,
        chunksFetched: stats.chunksFetched,
        durationMs: this._elapsedSince(started),
        elapsedMs: stats.elapsedMs,
        bytesPerSec: stats.bytesPerSec,
        peakBytesPerSec: stats.peakBytesPerSec,
        throughputMBps: stats.throughputMBps,
        parallelMode: stats.parallelMode,
        retryCount: stats.retryCount,
        abortedCount: stats.abortedCount,
      };
    } catch (err) {
      this._finishTelemetryError(run, err, signal);
      throw err;
    }
  }

  /**
   * Preload the PGIM image (or multi-module program via `programManifestUrl`) plus optional
   * `.wgsl.pack` and S6 reflection manifest in parallel.
   * Omitted URLs and HTTP 404/410 for optional shader/reflection assets resolve to `null`.
   *
   * @param {{ pgimUrl?: string | null, programManifestUrl?: string | null, bootManifestUrl?: string | null,
   *           shaderPackUrl?: string | null, reflectionUrl?: string | null,
   *           onProgress?: (e: object) => void, signal?: AbortSignal, verifyChecksums?: boolean,
   *           chunkSize?: number, chunkBytes?: number, concurrency?: number }} options
   *   Progress events carry an extra `asset` field: `pgim`, `shaderPack`, `reflection`, or `bundle`.
   */
  async preloadBootBundle({
    pgimUrl = null,
    programManifestUrl = null,
    bootManifestUrl = undefined,
    shaderPackUrl = null,
    reflectionUrl = null,
    onProgress,
    signal,
    verifyChecksums = false,
    chunkSize,
    chunkBytes,
    concurrency,
    opfsValidateWithHead,
    validateWithHead,
    streamShaders,
    streamThresholdBytes,
  } = {}) {
    const hasProgramManifest = typeof programManifestUrl === 'string' && programManifestUrl.length > 0;
    const hasPgimUrl = typeof pgimUrl === 'string' && pgimUrl.length > 0;
    if (!hasProgramManifest && !hasPgimUrl) {
      throw new TypeError('pgimUrl or programManifestUrl must be a non-empty string');
    }
    const transfer = this._resolveTransferConfig({ chunkSize, chunkBytes, concurrency });
    const shaderPolicy = resolveShaderPackPreloadPolicy({
      streamShaders: streamShaders !== undefined ? streamShaders : this.streamShaders,
      streamThresholdBytes:
        streamThresholdBytes !== undefined ? streamThresholdBytes : this.streamThresholdBytes,
    });
    const effectiveValidateWithHead =
      opfsValidateWithHead !== undefined
        ? Boolean(opfsValidateWithHead)
        : validateWithHead !== undefined
          ? Boolean(validateWithHead)
          : this.opfsValidateWithHead;
    const run = this._startTelemetryRun();
    try {
      throwIfAborted(signal);
      const started = this._now();
      const timingMs = { pgim: null, shaderPack: null, reflection: null, total: 0 };
      let shaderPackSkipInfo = null;
      const { controller, unlink } = linkSignal(signal);
      const inner = controller.signal;
      const timed = async (key, fn) => {
        const t0 = this._now();
        try {
          return await fn();
        } finally {
          timingMs[key] = this._elapsedSince(t0);
        }
      };
      const forward = (asset) =>
        typeof onProgress === 'function' ? (event) => onProgress({ ...event, asset }) : undefined;

      const optionalAsset = async (key, url, label, validate, allowParallelRange, maxPreloadBytes = 0) => {
        if (!url) return null;
        try {
          return await timed(key, async () => {
            const fetched = await this._fetchAsset(
              url,
              label,
              inner,
              makeEmitter(onProgress, { asset: key }),
              run,
              false,
              {
                allowParallelRange,
                chunkSize: transfer.chunkSize,
                concurrency: transfer.concurrency,
                maxPreloadBytes,
              }
            );
            if (fetched.skipped) {
              const reason = fetched.skipReason || 'exceeds-preload-threshold';
              const totalBytes = fetched.totalBytes ?? null;
              if (key === 'shaderPack') {
                shaderPackSkipInfo = { reason, totalBytes };
              }
              if (typeof onProgress === 'function') {
                onProgress({
                  phase: 'skipped',
                  asset: key,
                  reason,
                  loadedBytes: 0,
                  totalBytes: totalBytes || 0,
                  pct: 100,
                });
              }
              return null;
            }
            return validate(fetched.bytes, url);
          });
        } catch (err) {
          if (err instanceof PgimPreloadError && err.code === 'NOT_FOUND') {
            timingMs[key] = timingMs[key] ?? this._elapsedSince(started);
            if (typeof onProgress === 'function') {
              onProgress({ phase: 'optional-missing', asset: key, loadedBytes: 0, totalBytes: 0, pct: 0, status: err.status });
            }
            return null;
          }
          throw err;
        }
      };

      const shaderPackTask = !shaderPackUrl
        ? Promise.resolve(null)
        : shaderPolicy.skipFullPreload
          ? Promise.resolve().then(() => {
              timingMs.shaderPack = 0;
              shaderPackSkipInfo = { reason: 'stream-shaders', totalBytes: null };
              if (typeof onProgress === 'function') {
                onProgress({
                  phase: 'skipped',
                  asset: 'shaderPack',
                  reason: 'stream-shaders',
                  loadedBytes: 0,
                  totalBytes: 0,
                  pct: 100,
                });
              }
              return null;
            })
          : optionalAsset(
              'shaderPack',
              shaderPackUrl,
              'WGPK shader pack',
              validateWgslPackBytes,
              true,
              shaderPolicy.maxPreloadBytes
            );

      const tasks = [
        timed('pgim', () =>
          hasProgramManifest
            ? this._preloadProgramModulesCore(
                programManifestUrl,
                {
                  bootManifestUrl,
                  fallbackPgimUrl: hasPgimUrl ? pgimUrl : null,
                  onProgress: forward('pgim'),
                  signal: inner,
                  verifyChecksums,
                  chunkSize: transfer.chunkSize,
                  concurrency: transfer.concurrency,
                  opfsValidateWithHead: effectiveValidateWithHead,
                },
                run
              )
            : this._preloadGuestImageCore(pgimUrl, {
                onProgress: forward('pgim'),
                signal: inner,
                verifyChecksums,
                chunkSize: transfer.chunkSize,
                concurrency: transfer.concurrency,
                precomputedProbe: null,
                emitProbe: true,
                emitComplete: true,
                telemetryRun: run,
                ownsTotalBytes: false,
                opfsValidateWithHead: effectiveValidateWithHead,
              })
        ),
        shaderPackTask,
        optionalAsset('reflection', reflectionUrl, 'reflection manifest', parseReflectionManifestJson, false),
      ];

      let results;
      try {
        results = await Promise.all(
          tasks.map((task) =>
            task.catch((err) => {
              controller.abort(err);
              throw err;
            })
          )
        );
      } catch (err) {
        throwIfAborted(signal);
        throw err;
      } finally {
        unlink();
      }
      const [pgimResult, shaderPackBytes, reflectionManifest] = results;
      timingMs.total = this._elapsedSince(started);
      run.complete(run.bytesLoaded);
      const stats = run.snapshot();

      if (hasProgramManifest) {
        const primary = pgimResult.mainModule;
        const allOpfsHits = pgimResult.modules.length > 0 && pgimResult.modules.every((m) => m.opfsCacheHit);
        if (typeof onProgress === 'function') {
          onProgress({
            phase: 'complete',
            asset: 'bundle',
            loadedBytes: pgimResult.bytesLoaded,
            totalBytes: pgimResult.bytesLoaded,
            pct: 100,
            moduleIndex: pgimResult.moduleCount - 1,
            moduleCount: pgimResult.moduleCount,
            currentModule: pgimResult.modules[pgimResult.moduleCount - 1].name,
            ...(allOpfsHits ? { opfsCacheHit: true } : {}),
          });
        }
        return {
          pgim: primary,
          pgimBytes: primary.bytes,
          pgimHeader: primary.header,
          pgimLayout: primary.layout,
          buffer: primary.buffer,
          header: primary.header,
          shared: primary.shared,
          opfsCacheHit: allOpfsHits,
          manifestOpfsCacheHit: Boolean(pgimResult.manifestOpfsCacheHit),
          program: pgimResult,
          modules: pgimResult.modules,
          modulesByName: pgimResult.modulesByName,
          moduleCount: pgimResult.moduleCount,
          shaderPackBytes,
          ...(shaderPackSkipInfo
            ? {
                shaderPackSkipped: true,
                shaderPackSkipReason: shaderPackSkipInfo.reason,
                shaderPackStreamedBytes: shaderPackSkipInfo.totalBytes,
              }
            : {}),
          reflectionManifest,
          timingMs,
          durationMs: timingMs.total,
          elapsedMs: stats.elapsedMs,
          bytesLoaded: pgimResult.bytesLoaded,
          totalBytes: stats.totalBytes,
          chunksFetched: stats.chunksFetched,
          bytesPerSec: stats.bytesPerSec,
          peakBytesPerSec: stats.peakBytesPerSec,
          throughputMBps: stats.throughputMBps,
          parallelMode: stats.parallelMode,
          retryCount: stats.retryCount,
          abortedCount: stats.abortedCount,
        };
      }

      const image = pgimResult;
      if (typeof onProgress === 'function') {
        const bundleCompleteEvent = {
          phase: 'complete',
          asset: 'bundle',
          loadedBytes: image.bytesLoaded,
          totalBytes: image.bytesLoaded,
          pct: 100,
        };
        if (image.opfsCacheHit) {
          bundleCompleteEvent.opfsCacheHit = true;
        }
        onProgress(bundleCompleteEvent);
      }
      return {
        pgim: image,
        pgimBytes: image.bytes,
        pgimHeader: image.header,
        pgimLayout: image.layout,
        buffer: image.buffer,
        header: image.header,
        shared: image.shared,
        opfsCacheHit: Boolean(image.opfsCacheHit),
        etag: image.etag ?? null,
        shaderPackBytes,
        ...(shaderPackSkipInfo
          ? {
              shaderPackSkipped: true,
              shaderPackSkipReason: shaderPackSkipInfo.reason,
              shaderPackStreamedBytes: shaderPackSkipInfo.totalBytes,
            }
          : {}),
        reflectionManifest,
        timingMs,
        durationMs: timingMs.total,
        elapsedMs: stats.elapsedMs,
        bytesLoaded: image.bytesLoaded,
        totalBytes: stats.totalBytes,
        chunksFetched: stats.chunksFetched,
        bytesPerSec: stats.bytesPerSec,
        peakBytesPerSec: stats.peakBytesPerSec,
        throughputMBps: stats.throughputMBps,
        parallelMode: stats.parallelMode,
        retryCount: stats.retryCount,
        abortedCount: stats.abortedCount,
      };
    } catch (err) {
      this._finishTelemetryError(run, err, signal);
      throw err;
    }
  }

  async _request(ctx, init, signal = ctx.signal) {
    let attempt = 0;
    for (;;) {
      throwIfAborted(signal);
      ctx.requests += 1;
      if (ctx.run) ctx.run.recordRequest();
      try {
        const response = await raceAbort(
          Promise.resolve().then(() => this.fetchImpl(ctx.url, { ...init, signal })),
          signal
        );
        if (isRetryableHttpStatus(response.status) && attempt < this.maxRetries) {
          throwIfAborted(signal);
          if (response.body && typeof response.body.cancel === 'function') {
            response.body.cancel().catch(() => {});
          }
          const delayMs = this._computeRetryDelayMs(attempt);
          attempt += 1;
          if (ctx.run) ctx.run.recordRetry();
          await this._sleepWithAbort(delayMs, signal);
          continue;
        }
        return response;
      } catch (err) {
        if (signal && signal.aborted) throw abortReason(signal);
        if (isAbortError(err)) throw err;
        if (err instanceof PgimPreloadError && err.code !== 'NETWORK_ERROR') throw err;
        const wrapped =
          err instanceof PgimPreloadError
            ? err
            : new PgimPreloadError(
                'NETWORK_ERROR',
                `PGIM image request failed: ${ctx.url} (${err && err.message ? err.message : err})`,
                { url: ctx.url, cause: err }
              );
        if (attempt < this.maxRetries && isRetryableNetworkError(wrapped)) {
          throwIfAborted(signal);
          const delayMs = this._computeRetryDelayMs(attempt);
          attempt += 1;
          if (ctx.run) ctx.run.recordRetry();
          await this._sleepWithAbort(delayMs, signal);
          continue;
        }
        throw wrapped;
      }
    }
  }

    async _probe(ctx) {
      let response;
      try {
        response = await this._request(ctx, { method: 'HEAD' });
      } catch (err) {
        if (ctx.signal && ctx.signal.aborted) throw err;
        return {
          totalBytes: null,
          acceptRanges: null,
          etag: null,
          strongEtag: null,
          weakEtag: false,
          lastModified: null,
          headOk: false,
        }; // HEAD unsupported: probe with a Range GET.
      }
      if (response.status === 404 || response.status === 410) throw httpError(ctx.url, response.status);
      if (!response.ok) {
        return {
          totalBytes: null,
          acceptRanges: null,
          etag: null,
          strongEtag: null,
          weakEtag: false,
          lastModified: null,
          headOk: false,
        };
      }
      const totalBytes = contentLength(response);
      if (totalBytes !== null && totalBytes < PGIM_HEADER_BYTES) {
        throw new PgimPreloadError(
          'TRUNCATED',
          `truncated PGIM image header: needed ${PGIM_HEADER_BYTES} bytes, server reports ${totalBytes}`,
          { needed: PGIM_HEADER_BYTES, available: totalBytes }
        );
      }
      const ranges = headerValue(response, 'accept-ranges');
      let acceptRanges = null;
      if (ranges !== null) {
        if (/\bbytes\b/i.test(ranges)) acceptRanges = true;
        else if (ranges.trim().toLowerCase() === 'none') acceptRanges = false;
      }
      const etag = normalizeEtag(headerValue(response, 'etag'));
      const lastModified = normalizeLastModified(headerValue(response, 'last-modified'));
      return {
        totalBytes,
        acceptRanges,
        etag,
        strongEtag: isStrongEtag(etag) ? etag : null,
        weakEtag: isWeakEtag(etag),
        lastModified,
        headOk: true,
      };
    }

    async _probeAsset(ctx, what) {
      let response;
      try {
        response = await this._request(ctx, { method: 'HEAD' });
      } catch (err) {
        if (ctx.signal && ctx.signal.aborted) throw err;
        return { totalBytes: null, acceptRanges: null };
      }
      if (response.status === 404 || response.status === 410) throw httpError(ctx.url, response.status, what);
      if (!response.ok) return { totalBytes: null, acceptRanges: null };
      const totalBytes = contentLength(response);
      const ranges = headerValue(response, 'accept-ranges');
      let acceptRanges = null;
      if (ranges !== null) {
        if (/\bbytes\b/i.test(ranges)) acceptRanges = true;
        else if (ranges.trim().toLowerCase() === 'none') acceptRanges = false;
      }
      return { totalBytes, acceptRanges };
    }

    /**
     * Lightweight HEAD probe returning the shader pack byte length (or null if unavailable).
     * Never throws on 404/405/network errors; propagates AbortSignal cancellation.
     */
    async probeShaderPackSize(url, { signal } = {}) {
      if (typeof url !== 'string' || url.length === 0) return null;
      throwIfAborted(signal);
      const ctx = {
        url,
        signal: signal || null,
        emit: () => {},
        requests: 0,
        run: null,
        ownsTotalBytes: false,
        chunkSize: this.chunkSize,
        concurrency: this.concurrency,
        what: 'WGPK shader pack',
      };
      try {
        const probe = await this._probeAsset(ctx, 'WGPK shader pack');
        return probe && typeof probe.totalBytes === 'number' && Number.isSafeInteger(probe.totalBytes)
          ? probe.totalBytes
          : null;
      } catch (err) {
        if ((signal && signal.aborted) || isAbortError(err)) throw err;
        return null;
      }
    }

  _allocate(totalBytes) {
    if (this.targetBuffer) {
      if (this.targetBuffer.byteLength < totalBytes) {
        throw new PgimPreloadError(
          'ALLOCATION_FAILED',
          `targetBuffer is ${this.targetBuffer.byteLength} bytes but PGIM needs ${totalBytes}`
        );
      }
      return {
        bytes: new Uint8Array(this.targetBuffer, 0, totalBytes),
        shared: isSharedArrayBuffer(this.targetBuffer),
      };
    }
    if (this.useSharedArrayBuffer && typeof SharedArrayBuffer === 'function') {
      try {
        return { bytes: new Uint8Array(new SharedArrayBuffer(totalBytes)), shared: true };
      } catch {
        // Fall back to a regular ArrayBuffer below.
      }
    }
    try {
      return { bytes: new Uint8Array(new ArrayBuffer(totalBytes)), shared: false };
    } catch (err) {
      throw new PgimPreloadError('ALLOCATION_FAILED', `cannot allocate ${totalBytes} bytes for the PGIM image`, { cause: err });
    }
  }

    async _readRangeBodyInto(response, dest, start, expectedLen, signal) {
      const body = response.body;
      if (!body || typeof body.getReader !== 'function') {
        const data = new Uint8Array(await raceAbort(response.arrayBuffer(), signal));
        throwIfAborted(signal);
        if (data.byteLength !== expectedLen) {
          throw new PgimPreloadError(
            'RANGE_MISMATCH',
            `PGIM chunk bytes=${start}-${start + expectedLen - 1} returned ${data.byteLength} bytes, expected ${expectedLen}`
          );
        }
        dest.set(data, start);
        return expectedLen;
      }

      const reader = body.getReader();
      let written = 0;
      try {
        for (;;) {
          const { done, value } = await raceAbort(reader.read(), signal);
          if (done) break;
          const part = toBytes(value, 'range chunk');
          if (written + part.byteLength > expectedLen) {
            throw new PgimPreloadError(
              'RANGE_MISMATCH',
              `PGIM chunk bytes=${start}-${start + expectedLen - 1} exceeded expected ${expectedLen} bytes`
            );
          }
          dest.set(part, start + written);
          written += part.byteLength;
        }
      } catch (err) {
        reader.cancel(err).catch(() => {});
        throwIfAborted(signal);
        throw err;
      }
      throwIfAborted(signal);
      if (written !== expectedLen) {
        throw new PgimPreloadError(
          'RANGE_MISMATCH',
          `PGIM chunk bytes=${start}-${start + expectedLen - 1} returned ${written} bytes, expected ${expectedLen}`
        );
      }
      return written;
    }

    async _fetchRangeChunkWithRetry(
      ctx,
      start,
      end,
      dest,
      signal = ctx.signal,
      {
        allowFullFallback = false,
        validateTotal = null,
        pinnedEtag = null,
        requireContentRangeTotal = false,
      } = {}
    ) {
      const expectedLen = end - start;
      const rangeHeader = `bytes=${start}-${end - 1}`;
      const strongPinnedEtag = isStrongEtag(pinnedEtag) ? pinnedEtag : null;
      const what = ctx.what || 'PGIM image';
      let attempt = 0;
      for (;;) {
        throwIfAborted(signal);
        ctx.requests += 1;
        if (ctx.run) ctx.run.recordRequest();
        const reqHeaders = { Range: rangeHeader };
        if (strongPinnedEtag !== null) {
          reqHeaders['If-Range'] = strongPinnedEtag;
        }
        try {
          const res = await raceAbort(
            Promise.resolve().then(() =>
              this.fetchImpl(ctx.url, { method: 'GET', headers: reqHeaders, signal })
            ),
            signal
          );
          if (isRetryableHttpStatus(res.status) && attempt < this.maxRetries) {
            throwIfAborted(signal);
            if (res.body && typeof res.body.cancel === 'function') {
              res.body.cancel().catch(() => {});
            }
            const delayMs = this._computeRetryDelayMs(attempt);
            attempt += 1;
            if (ctx.run) ctx.run.recordRetry();
            await this._sleepWithAbort(delayMs, signal);
            continue;
          }
          if (res.status === 412) {
            const observedEtag = normalizeEtag(headerValue(res, 'etag'));
            if (res.body && typeof res.body.cancel === 'function') {
              res.body.cancel().catch(() => {});
            }
            throw new PgimPreloadError(
              'VERSION_MISMATCH',
              `PGIM chunk ${rangeHeader} precondition failed (HTTP 412; expected ${strongPinnedEtag ?? 'current version'}, got ${observedEtag ?? 'none'})`,
              {
                url: ctx.url,
                status: 412,
                range: rangeHeader,
                expectedEtag: strongPinnedEtag,
                observedEtag,
                retryableVersionChange: true,
              }
            );
          }
          if (allowFullFallback && res.status === 200) {
            return { fullResponse: res };
          }
          if (res.status === 416 && start === 0) {
            if (res.body && typeof res.body.cancel === 'function') {
              res.body.cancel().catch(() => {});
            }
            throw new PgimPreloadError('TRUNCATED', `truncated PGIM image header: ${ctx.url} is empty (HTTP 416)`, {
              needed: PGIM_HEADER_BYTES,
              available: 0,
            });
          }
          if (res.status === 416 && start > 0 && strongPinnedEtag !== null) {
            const observedEtag = normalizeEtag(headerValue(res, 'etag'));
            if (res.body && typeof res.body.cancel === 'function') {
              res.body.cancel().catch(() => {});
            }
            throw new PgimPreloadError(
              'VERSION_MISMATCH',
              `PGIM chunk ${rangeHeader} became unsatisfiable mid-download (HTTP 416; file likely rotated or shrank)`,
              {
                url: ctx.url,
                status: 416,
                range: rangeHeader,
                expectedEtag: strongPinnedEtag,
                observedEtag,
                retryableVersionChange: true,
              }
            );
          }
          if (res.status !== 206) {
            if (res.body && typeof res.body.cancel === 'function') {
              res.body.cancel().catch(() => {});
            }
            throw res.status === 200
              ? new PgimPreloadError('RANGE_NOT_HONORED', `server ignored Range ${rangeHeader} for ${ctx.url}`)
              : httpError(ctx.url, res.status, what);
          }

          const cr = parseContentRange(headerValue(res, 'content-range'));
          const chunkEtag = normalizeEtag(headerValue(res, 'etag'));
          const chunkLastModified = normalizeLastModified(headerValue(res, 'last-modified'));

          if (strongPinnedEtag !== null && (!isStrongEtag(chunkEtag) || chunkEtag !== strongPinnedEtag)) {
            if (res.body && typeof res.body.cancel === 'function') {
              res.body.cancel().catch(() => {});
            }
            throw new PgimPreloadError(
              'VERSION_MISMATCH',
              `PGIM chunk ${rangeHeader} ETag mismatch (expected ${strongPinnedEtag}, got ${chunkEtag ?? 'none'})`,
              {
                url: ctx.url,
                range: rangeHeader,
                expectedEtag: strongPinnedEtag,
                observedEtag: chunkEtag,
                retryableVersionChange: true,
              }
            );
          }

          if (dest !== null) {
            if (requireContentRangeTotal) {
              if (!cr) {
                if (res.body && typeof res.body.cancel === 'function') {
                  res.body.cancel().catch(() => {});
                }
                throw new PgimPreloadError('RANGE_MISMATCH', `missing Content-Range header for ${rangeHeader}`, {
                  url: ctx.url,
                  range: rangeHeader,
                });
              }
              if (cr.start !== start || cr.end !== end - 1) {
                if (res.body && typeof res.body.cancel === 'function') {
                  res.body.cancel().catch(() => {});
                }
                throw new PgimPreloadError('RANGE_MISMATCH', `unexpected Content-Range for ${rangeHeader}`, {
                  url: ctx.url,
                  range: rangeHeader,
                });
              }
              if (cr.total === null || (validateTotal !== null && cr.total !== validateTotal)) {
                if (res.body && typeof res.body.cancel === 'function') {
                  res.body.cancel().catch(() => {});
                }
                const totalChanged = cr.total !== null && validateTotal !== null && cr.total !== validateTotal;
                throw new PgimPreloadError(
                  'RANGE_MISMATCH',
                  `PGIM chunk ${rangeHeader} Content-Range total mismatch (expected ${validateTotal}, got ${cr.total ?? '*'})`,
                  {
                    url: ctx.url,
                    range: rangeHeader,
                    expectedTotal: validateTotal,
                    observedTotal: cr.total,
                    retryableVersionChange: totalChanged,
                  }
                );
              }
            } else if (
              cr &&
              (cr.start !== start ||
                cr.end !== end - 1 ||
                (validateTotal !== null && cr.total !== null && cr.total !== validateTotal))
            ) {
              if (res.body && typeof res.body.cancel === 'function') {
                res.body.cancel().catch(() => {});
              }
              throw new PgimPreloadError('RANGE_MISMATCH', `unexpected Content-Range for ${rangeHeader}`);
            }
            const declaredLen = contentLength(res);
            if (declaredLen !== null && declaredLen !== expectedLen) {
              if (res.body && typeof res.body.cancel === 'function') {
                res.body.cancel().catch(() => {});
              }
              throw new PgimPreloadError(
                'RANGE_MISMATCH',
                `PGIM chunk ${rangeHeader} declared ${declaredLen} bytes, expected ${expectedLen}`
              );
            }
            const bytesWritten = await this._readRangeBodyInto(res, dest, start, expectedLen, signal);
            return { bytesWritten, range: cr, etag: chunkEtag, lastModified: chunkLastModified };
          }

          if (cr && cr.start !== start) {
            if (res.body && typeof res.body.cancel === 'function') {
              res.body.cancel().catch(() => {});
            }
            throw new PgimPreloadError('RANGE_MISMATCH', `unexpected Content-Range start for ${rangeHeader}`);
          }
          if (cr && cr.total !== null && validateTotal !== null && cr.total !== validateTotal) {
            if (res.body && typeof res.body.cancel === 'function') {
              res.body.cancel().catch(() => {});
            }
            throw new PgimPreloadError(
              'SIZE_MISMATCH',
              `PGIM size changed between HEAD (${validateTotal}) and Range response (${cr.total})`,
              {
                url: ctx.url,
                expectedTotal: validateTotal,
                observedTotal: cr.total,
                retryableVersionChange: true,
              }
            );
          }

          if (strongPinnedEtag === null && allowFullFallback) {
            const knownTotal = cr && cr.total !== null ? cr.total : validateTotal;
            if (knownTotal !== null && knownTotal > end && !isStrongEtag(chunkEtag)) {
              if (res.body && typeof res.body.cancel === 'function') {
                res.body.cancel().catch(() => {});
              }
              return { needsFullFallback: true, etag: chunkEtag, lastModified: chunkLastModified };
            }
          }

          const declaredLen = contentLength(res);
          if (cr !== null && declaredLen !== null && declaredLen !== cr.end - cr.start + 1) {
            if (res.body && typeof res.body.cancel === 'function') {
              res.body.cancel().catch(() => {});
            }
            throw new PgimPreloadError(
              'RANGE_MISMATCH',
              `PGIM chunk ${rangeHeader} Content-Length (${declaredLen}) disagrees with Content-Range`
            );
          }
          const bytes = new Uint8Array(await raceAbort(res.arrayBuffer(), signal));
          throwIfAborted(signal);
          if (declaredLen !== null && bytes.byteLength !== declaredLen) {
            throw new PgimPreloadError(
              'RANGE_MISMATCH',
              `PGIM chunk ${rangeHeader} returned ${bytes.byteLength} bytes, declared ${declaredLen}`
            );
          }
          return {
            bytes,
            range: cr,
            etag: chunkEtag,
            lastModified: chunkLastModified,
          };
        } catch (err) {
          if (signal && signal.aborted) throw abortReason(signal);
          if (isAbortError(err)) throw err;
          if (err instanceof PgimPreloadError && err.code !== 'NETWORK_ERROR') throw err;
          const wrapped =
            err instanceof PgimPreloadError
              ? err
              : new PgimPreloadError(
                  'NETWORK_ERROR',
                  `${what} request failed: ${ctx.url} (${err && err.message ? err.message : err})`,
                  { url: ctx.url, cause: err }
                );
          if (attempt < this.maxRetries && isRetryableNetworkError(wrapped)) {
            throwIfAborted(signal);
            const delayMs = this._computeRetryDelayMs(attempt);
            attempt += 1;
            if (ctx.run) ctx.run.recordRetry();
            await this._sleepWithAbort(delayMs, signal);
            continue;
          }
          throw wrapped;
        }
      }
    }

    async _fetchFullCoherent(ctx, probe) {
      const response = await this._request(ctx, { method: 'GET' });
      if (!response.ok) throw httpError(ctx.url, response.status, ctx.what || 'PGIM image');
      const responseEtag = normalizeEtag(headerValue(response, 'etag'));
      const responseLastModified = normalizeLastModified(headerValue(response, 'last-modified'));
      const rotatedSinceHead =
        probe !== null &&
        ((probe.etag !== null && responseEtag !== null && responseEtag !== probe.etag) ||
          (probe.etag === null &&
            probe.lastModified !== null &&
            responseLastModified !== null &&
            responseLastModified !== probe.lastModified));
      // Full GET is a single coherent representation: if the file rotated since HEAD (or HEAD had no strong
      // validator), trust the response's own Content-Length and PGIM header file_size.
      const expectedTotal =
        rotatedSinceHead || (probe && !isStrongEtag(probe.etag)) ? null : probe ? probe.totalBytes : null;
      return this._consumeFullBody(ctx, response, expectedTotal);
    }

    async _loadRanged(ctx, probe) {
      const chunkSize = ctx.chunkSize ?? this.chunkSize;
      const concurrency = ctx.concurrency ?? this.concurrency;
      const headStrongEtag = isStrongEtag(probe.etag) ? probe.etag : null;

      // Weak or non-strong ETag on HEAD cannot safely validate multi-chunk byte ranges (RFC 9110 §13.1.5):
      // use a single coherent full GET instead of stitching unverified range chunks.
      if (probe.headOk && probe.etag !== null && headStrongEtag === null) {
        return this._fetchFullCoherent(ctx, probe);
      }

      const firstLen = probe.totalBytes !== null ? Math.min(chunkSize, probe.totalBytes) : chunkSize;
      const firstRes = await this._fetchRangeChunkWithRetry(ctx, 0, firstLen, null, ctx.signal, {
        allowFullFallback: true,
        validateTotal: probe.totalBytes,
        pinnedEtag: headStrongEtag,
      });
      if (firstRes.fullResponse) {
        // Server either ignored Range or returned 200 OK because If-Range detected version rotation:
        // consume the coherent full body, trusting its own Content-Length on rotation or unpinned HEAD.
        const respEtag = normalizeEtag(headerValue(firstRes.fullResponse, 'etag'));
        const rotated = headStrongEtag !== null && respEtag !== headStrongEtag;
        const expectedFullBytes = rotated || headStrongEtag === null ? null : probe.totalBytes;
        return this._consumeFullBody(ctx, firstRes.fullResponse, expectedFullBytes);
      }
      if (firstRes.needsFullFallback) {
        // Multi-chunk download has no strong validator: fall back to a coherent single-response full GET.
        return this._fetchFullCoherent(ctx, probe);
      }

      let { bytes: first, range, etag: chunkEtag, lastModified: chunkLastModified } = firstRes;
      firstRes.bytes = null;
      if (range && (range.start !== 0 || range.end - range.start + 1 !== first.byteLength)) {
        first = null;
        throw new PgimPreloadError(
          'RANGE_MISMATCH',
          `PGIM first chunk Content-Range does not match the ${first.byteLength} received bytes`
        );
      }
      if (range && range.total !== null && probe.totalBytes !== null && range.total !== probe.totalBytes) {
        first = null;
        throw new PgimPreloadError(
          'SIZE_MISMATCH',
          `PGIM size changed between HEAD (${probe.totalBytes}) and Range response (${range.total})`,
          {
            url: ctx.url,
            expectedTotal: probe.totalBytes,
            observedTotal: range.total,
            retryableVersionChange: true,
          }
        );
      }
      const transportTotal = range && range.total !== null ? range.total : probe.totalBytes;
      const header = parsePgimHeader(first, { totalBytes: transportTotal });
      const total = header.fileSize;
      const firstByteLen = first.byteLength;
      if (firstByteLen !== Math.min(firstLen, total)) {
        first = null;
        throw new PgimPreloadError(
          'RANGE_MISMATCH',
          `PGIM first chunk returned ${firstByteLen} bytes, expected ${Math.min(firstLen, total)}`
        );
      }

      // Single-chunk download (entire file came from one coherent 206 response): no range stitching needed.
      if (firstByteLen === total) {
        const dest = this._allocate(total);
        dest.bytes.set(first, 0);
        first = null;
        if (ctx.run) {
          if (ctx.ownsTotalBytes) ctx.run.setTotalBytes(total);
          ctx.run.recordBytes(firstByteLen);
          ctx.run.recordNetworkBytes(firstByteLen);
          ctx.run.recordChunk(1);
        }
        ctx.emit('header', firstByteLen, total);
        const resolvedEtag = chunkEtag;
        return {
          ...dest,
          transport: 'range',
          chunksFetched: 1,
          parallelMode: false,
          opfsCacheHit: false,
          etag: resolvedEtag,
          hasOwnEtag: true,
          lastModified: chunkLastModified ?? (resolvedEtag === probe.etag ? probe.lastModified : null),
          versionVerified: true,
          validatorStrength: validatorStrengthForEtag(resolvedEtag),
          versionProvenance: isStrongEtag(resolvedEtag) ? 'strong-etag-single-range' : 'single-response-range',
        };
      }

      // Multi-chunk download requires one strong version validator pinned across every range chunk.
      const pinnedEtag = headStrongEtag ?? (isStrongEtag(chunkEtag) ? chunkEtag : null);
      if (pinnedEtag === null) {
        first = null;
        return this._fetchFullCoherent(ctx, probe);
      }
      if (!isStrongEtag(chunkEtag) || chunkEtag !== pinnedEtag) {
        first = null;
        throw new PgimPreloadError(
          'VERSION_MISMATCH',
          `PGIM first chunk ETag mismatch (expected ${pinnedEtag}, got ${chunkEtag ?? 'none'})`,
          {
            url: ctx.url,
            expectedEtag: pinnedEtag,
            observedEtag: chunkEtag,
            retryableVersionChange: true,
          }
        );
      }

      const preChunkBytesLoaded = ctx.run ? ctx.run.bytesLoaded : 0;
      const preChunkChunksFetched = ctx.run ? ctx.run.chunksFetched : 0;
      let dest = this._allocate(total);
      dest.bytes.set(first, 0);
      first = null;
      if (ctx.run) {
        if (ctx.ownsTotalBytes) ctx.run.setTotalBytes(total);
        ctx.run.recordBytes(firstByteLen);
        ctx.run.recordNetworkBytes(firstByteLen);
        ctx.run.recordChunk(1);
      }
      ctx.emit('header', firstByteLen, total);

      let fallbackFullResponse = null;
      let parallel;
      try {
        parallel = await this._parallelChunks(
          ctx,
          dest.bytes,
          firstByteLen,
          async (start, end, signal) => {
            const chunkRes = await this._fetchRangeChunkWithRetry(ctx, start, end, dest.bytes, signal, {
              allowFullFallback: true,
              validateTotal: total,
              pinnedEtag,
              requireContentRangeTotal: true,
            });
            if (chunkRes.fullResponse) {
              const isWinner = fallbackFullResponse === null;
              if (isWinner) {
                fallbackFullResponse = chunkRes.fullResponse;
              } else if (
                chunkRes.fullResponse.body &&
                typeof chunkRes.fullResponse.body.cancel === 'function'
              ) {
                chunkRes.fullResponse.body.cancel().catch(() => {});
              }
              throw new PgimPreloadError(
                'VERSION_MISMATCH',
                `PGIM resource rotated mid-download (server returned HTTP 200 to If-Range ${pinnedEtag})`,
                {
                  url: ctx.url,
                  status: 200,
                  expectedEtag: pinnedEtag,
                  retryableVersionChange: true,
                  _preserveWorkerSignal: isWinner,
                }
              );
            }
            return chunkRes.bytesWritten;
          },
          { directWrite: true, countNetworkBytes: true }
        );
      } catch (err) {
        dest = null;
        const unlinkWinner = err && typeof err._unlinkWinner === 'function' ? err._unlinkWinner : null;
        if (fallbackFullResponse !== null && !(ctx.signal && ctx.signal.aborted)) {
          const fullRes = fallbackFullResponse;
          fallbackFullResponse = null;
          if (ctx.run) {
            ctx.run.bytesLoaded = preChunkBytesLoaded;
            ctx.run.chunksFetched = preChunkChunksFetched;
          }
          try {
            return await this._consumeFullBody(ctx, fullRes, null);
          } finally {
            if (unlinkWinner) unlinkWinner();
          }
        }
        if (unlinkWinner) unlinkWinner();
        if (
          fallbackFullResponse !== null &&
          fallbackFullResponse.body &&
          typeof fallbackFullResponse.body.cancel === 'function'
        ) {
          fallbackFullResponse.body.cancel().catch(() => {});
        }
        throw err;
      }

      const chunksFetched = 1 + parallel.chunksCompleted;
      const parallelMode = concurrency > 1 && chunksFetched > 1;
      if (parallelMode && ctx.run) ctx.run.setParallelMode(true);
      return {
        ...dest,
        transport: 'range',
        chunksFetched,
        parallelMode,
        opfsCacheHit: false,
        etag: pinnedEtag,
        hasOwnEtag: true,
        lastModified: chunkLastModified ?? (pinnedEtag === probe.etag ? probe.lastModified : null),
        versionVerified: true,
        validatorStrength: 'strong',
        versionProvenance: 'strong-etag-range',
      };
    }

    async _loadViaRangeCache(ctx, probe) {
      const chunkSize = ctx.chunkSize ?? this.chunkSize;
      const concurrency = ctx.concurrency ?? this.concurrency;
      const cache = this.rangeCache;
      const read = async (offset, length, signal) => {
        throwIfAborted(signal);
        ctx.requests += 1;
        if (ctx.run) ctx.run.recordRequest();
        return toBytes(await raceAbort(Promise.resolve().then(() => cache.read(ctx.url, offset, length)), signal), 'rangeCache chunk');
      };
      const firstLen = probe.totalBytes !== null ? Math.min(chunkSize, probe.totalBytes) : chunkSize;
      let first = await read(0, firstLen, ctx.signal);
      const header = parsePgimHeader(first, { totalBytes: probe.totalBytes });
      const total = header.fileSize;
      const firstByteLen = first.byteLength;
      if (firstByteLen !== Math.min(firstLen, total)) {
        throw new PgimPreloadError('RANGE_MISMATCH', `rangeCache returned ${firstByteLen} bytes, expected ${Math.min(firstLen, total)}`);
      }
      const dest = this._allocate(total);
      dest.bytes.set(first, 0);
      first = null;
      if (ctx.run) {
        if (ctx.ownsTotalBytes) ctx.run.setTotalBytes(total);
        ctx.run.recordBytes(firstByteLen);
        ctx.run.recordChunk(1);
      }
      ctx.emit('header', firstByteLen, total);
      const parallel = await this._parallelChunks(ctx, dest.bytes, firstByteLen, (start, end, signal) =>
        read(start, end - start, signal)
      );
      const chunksFetched = 1 + parallel.chunksCompleted;
      const parallelMode = concurrency > 1 && chunksFetched > 1;
      if (parallelMode && ctx.run) ctx.run.setParallelMode(true);
      return {
        ...dest,
        transport: 'range-cache',
        chunksFetched,
        parallelMode,
        versionVerified: false,
        validatorStrength: 'none',
        versionProvenance: chunksFetched === 1 ? 'range-cache-single' : 'range-cache-unverified',
      };
    }

    async _parallelChunks(ctx, dest, startOffset, fetchChunk, { directWrite = false, countNetworkBytes = false } = {}) {
      const chunkSize = ctx.chunkSize ?? this.chunkSize;
      const concurrency = ctx.concurrency ?? this.concurrency;
      const total = dest.byteLength;
      const chunks = [];
      for (let start = startOffset; start < total; start += chunkSize) {
        chunks.push([start, Math.min(total, start + chunkSize)]);
      }
      if (chunks.length === 0) return { chunksCompleted: 0 };
      const workerCount = Math.min(concurrency, chunks.length);
      const workerLinks = Array.from({ length: workerCount }, () => linkSignal(ctx.signal));
      let next = 0;
      let loaded = startOffset;
      let chunksCompleted = 0;
      let failure = null;
      let preserveWorkerIdx = -1;
      const abortWorkers = (err, skipIdx = -1) => {
        for (let k = 0; k < workerLinks.length; k += 1) {
          if (k !== skipIdx) workerLinks[k].controller.abort(err);
        }
      };
      const worker = async (workerIdx) => {
        const workerSignal = workerLinks[workerIdx].controller.signal;
        while (failure === null && next < chunks.length) {
          const [start, end] = chunks[next];
          next += 1;
          try {
            const expectedLen = end - start;
            let written;
            if (directWrite) {
              written = await fetchChunk(start, end, workerSignal);
              if (failure !== null) return;
              if (written !== expectedLen) {
                throw new PgimPreloadError(
                  'RANGE_MISMATCH',
                  `PGIM chunk bytes=${start}-${end - 1} returned ${written} bytes, expected ${expectedLen}`
                );
              }
            } else {
              const data = await fetchChunk(start, end, workerSignal);
              if (failure !== null) return;
              if (data.byteLength !== expectedLen) {
                throw new PgimPreloadError(
                  'RANGE_MISMATCH',
                  `PGIM chunk bytes=${start}-${end - 1} returned ${data.byteLength} bytes, expected ${expectedLen}`
                );
              }
              dest.set(data, start);
              written = data.byteLength;
            }
            loaded += written;
            chunksCompleted += 1;
            if (ctx.run) {
              ctx.run.recordBytes(written);
              if (countNetworkBytes) ctx.run.recordNetworkBytes(written);
              ctx.run.recordChunk(1);
            }
            ctx.emit('download', loaded, total);
          } catch (err) {
            if (failure === null) {
              failure = err;
              if (err && err._preserveWorkerSignal === true) {
                preserveWorkerIdx = workerIdx;
                abortWorkers(err, workerIdx);
              } else {
                abortWorkers(err);
              }
            }
            return;
          }
        }
      };
      try {
        await Promise.all(Array.from({ length: workerCount }, (_, idx) => worker(idx)));
      } finally {
        for (let k = 0; k < workerLinks.length; k += 1) {
          if (k === preserveWorkerIdx && failure !== null) {
            failure._unlinkWinner = workerLinks[k].unlink;
          } else {
            workerLinks[k].unlink();
          }
        }
      }
      if (ctx.signal && ctx.signal.aborted) {
        if (failure && typeof failure._unlinkWinner === 'function') {
          failure._unlinkWinner();
          failure._unlinkWinner = null;
        }
        throw abortReason(ctx.signal);
      }
      if (failure !== null) throw failure;
      return { chunksCompleted };
    }

    async _consumeFullBody(ctx, response, knownTotal) {
      const responseEtag = normalizeEtag(headerValue(response, 'etag'));
      const responseLastModified = normalizeLastModified(headerValue(response, 'last-modified'));
      const declared = contentLength(response);
      if (declared !== null && knownTotal !== null && declared !== knownTotal) {
        throw new PgimPreloadError('SIZE_MISMATCH', `PGIM size changed between HEAD (${knownTotal}) and GET (${declared})`);
      }
      const expected = declared !== null ? declared : knownTotal;
      const body = response.body;
      if (!body || typeof body.getReader !== 'function') {
        let whole = new Uint8Array(await raceAbort(response.arrayBuffer(), ctx.signal));
        throwIfAborted(ctx.signal);
        const header = parsePgimHeader(whole, { totalBytes: expected });
        const wholeByteLen = whole.byteLength;
        if (wholeByteLen !== header.fileSize) {
          throw new PgimPreloadError('TRUNCATED', `truncated PGIM image: needed ${header.fileSize} bytes, only ${wholeByteLen} available`, {
            needed: header.fileSize,
            available: wholeByteLen,
          });
        }
        let dest;
        if (
          this.targetBuffer === null &&
          !this.useSharedArrayBuffer &&
          whole.buffer instanceof ArrayBuffer &&
          whole.byteOffset === 0 &&
          whole.byteLength === whole.buffer.byteLength
        ) {
          dest = { bytes: whole, shared: false };
          whole = null;
        } else {
          dest = this._allocate(header.fileSize);
          dest.bytes.set(whole, 0);
          whole = null;
        }
        if (ctx.run) {
          if (ctx.ownsTotalBytes) ctx.run.setTotalBytes(header.fileSize);
          ctx.run.recordBytes(wholeByteLen);
          ctx.run.recordNetworkBytes(wholeByteLen);
          ctx.run.recordChunk(1);
        }
        ctx.emit('download', wholeByteLen, header.fileSize);
        return {
          ...dest,
          transport: 'full',
          chunksFetched: 1,
          parallelMode: false,
          opfsCacheHit: false,
          etag: responseEtag,
          hasOwnEtag: true,
          lastModified: responseLastModified,
          versionVerified: true,
          validatorStrength: validatorStrengthForEtag(responseEtag),
          versionProvenance: 'single-response',
        };
      }

      const reader = body.getReader();
      let pending = [];
      let pendingBytes = 0;
      let dest = null;
      let total = 0;
      let loaded = 0;
      const write = (chunk) => {
        if (loaded + chunk.byteLength > total) {
          throw new PgimPreloadError('SIZE_MISMATCH', `PGIM body is larger than its header file_size (${total} bytes)`);
        }
        dest.bytes.set(chunk, loaded);
        loaded += chunk.byteLength;
      };
      try {
        for (;;) {
          const { done, value } = await raceAbort(reader.read(), ctx.signal);
          if (done) break;
          const chunk = toBytes(value, 'response chunk');
          if (dest === null) {
            pending.push(chunk);
            pendingBytes += chunk.byteLength;
            if (pendingBytes < PGIM_HEADER_BYTES) continue;
            const head = new Uint8Array(PGIM_HEADER_BYTES);
            let cursor = 0;
            for (const part of pending) {
              const take = Math.min(part.byteLength, PGIM_HEADER_BYTES - cursor);
              head.set(part.subarray(0, take), cursor);
              cursor += take;
              if (cursor === PGIM_HEADER_BYTES) break;
            }
            const header = parsePgimHeader(head, { totalBytes: expected });
            total = header.fileSize;
            dest = this._allocate(total);
            for (const part of pending) write(part);
            pending = null;
            if (ctx.run) {
              if (ctx.ownsTotalBytes) ctx.run.setTotalBytes(total);
              ctx.run.recordBytes(loaded);
              ctx.run.recordNetworkBytes(loaded);
            }
            ctx.emit('header', loaded, total);
          } else {
            write(chunk);
            if (ctx.run) {
              ctx.run.recordBytes(chunk.byteLength);
              ctx.run.recordNetworkBytes(chunk.byteLength);
            }
            ctx.emit('download', loaded, total);
          }
        }
      } catch (err) {
        reader.cancel(err).catch(() => {});
        throwIfAborted(ctx.signal);
        throw err;
      }
      throwIfAborted(ctx.signal);
      if (dest === null) {
        throw new PgimPreloadError(
          'TRUNCATED',
          `truncated PGIM image header: needed ${PGIM_HEADER_BYTES} bytes, only ${pendingBytes} available`,
          { needed: PGIM_HEADER_BYTES, available: pendingBytes }
        );
      }
      if (loaded !== total) {
        throw new PgimPreloadError('TRUNCATED', `truncated PGIM image: needed ${total} bytes, only ${loaded} available`, {
          needed: total,
          available: loaded,
        });
      }
      if (ctx.run) ctx.run.recordChunk(1);
      return {
        ...dest,
        transport: 'full',
        chunksFetched: 1,
        parallelMode: false,
        opfsCacheHit: false,
        etag: responseEtag,
        hasOwnEtag: true,
        lastModified: responseLastModified,
        versionVerified: true,
        validatorStrength: validatorStrengthForEtag(responseEtag),
        versionProvenance: 'single-response',
      };
    }

    async _fetchAsset(
      url,
      what,
      signal,
      emit,
      telemetryRun = null,
      ownsTotalBytes = false,
      { allowParallelRange = false, chunkSize = this.chunkSize, concurrency = this.concurrency, maxPreloadBytes = 0 } = {}
    ) {
      const ctx = {
        url,
        signal,
        emit,
        requests: 0,
        run: telemetryRun,
        ownsTotalBytes,
        chunkSize,
        concurrency,
        what,
      };
      const hasMaxPreloadBytes = Number.isSafeInteger(maxPreloadBytes) && maxPreloadBytes > 0;
      const shouldProbe = (allowParallelRange && concurrency > 1) || hasMaxPreloadBytes;
      if (shouldProbe) {
        const probe = await this._probeAsset(ctx, what);
        if (hasMaxPreloadBytes && probe.totalBytes !== null && probe.totalBytes >= maxPreloadBytes) {
          return {
            bytes: null,
            skipped: true,
            skipReason: 'exceeds-preload-threshold',
            totalBytes: probe.totalBytes,
            transport: 'stream-deferred',
            chunksFetched: 0,
            parallelMode: false,
          };
        }
        if (
          allowParallelRange &&
          concurrency > 1 &&
          probe.acceptRanges === true &&
          probe.totalBytes !== null &&
          probe.totalBytes > chunkSize
        ) {
          const total = probe.totalBytes;
          const dest = new Uint8Array(total);
          if (telemetryRun && ownsTotalBytes) telemetryRun.setTotalBytes(total);
          emit('download', 0, total);
          const parallel = await this._parallelChunks(
            ctx,
            dest,
            0,
            async (start, end, chunkSignal) => {
              const chunkRes = await this._fetchRangeChunkWithRetry(ctx, start, end, dest, chunkSignal, {
                allowFullFallback: false,
                validateTotal: total,
              });
              return chunkRes.bytesWritten;
            },
            { directWrite: true, countNetworkBytes: true }
          );
          const parallelMode = parallel.chunksCompleted > 1;
          if (parallelMode && telemetryRun) telemetryRun.setParallelMode(true);
          emit('complete', total, total);
          return { bytes: dest, transport: 'range', chunksFetched: parallel.chunksCompleted, parallelMode };
        }
      }

      const response = await this._request(ctx, { method: 'GET' });
      if (!response.ok) throw httpError(url, response.status, what);
      const declared = contentLength(response);
      if (hasMaxPreloadBytes && declared !== null && declared >= maxPreloadBytes) {
        if (response.body && typeof response.body.cancel === 'function') {
          response.body.cancel().catch(() => {});
        }
        return {
          bytes: null,
          skipped: true,
          skipReason: 'exceeds-preload-threshold',
          totalBytes: declared,
          transport: 'stream-deferred',
          chunksFetched: 0,
          parallelMode: false,
        };
      }
      emit('download', 0, declared);
      let bytes;
      const body = response.body;
      if (body && typeof body.getReader === 'function') {
        const reader = body.getReader();
        let prealloc = declared !== null && declared > 0 ? new Uint8Array(declared) : null;
        let parts = prealloc === null ? [] : null;
        let loaded = 0;
        if (telemetryRun && ownsTotalBytes && declared !== null) {
          telemetryRun.setTotalBytes(declared);
        }
        try {
          for (;;) {
            const { done, value } = await raceAbort(reader.read(), signal);
            if (done) break;
            const part = toBytes(value, what);
            if (hasMaxPreloadBytes && loaded + part.byteLength >= maxPreloadBytes) {
              reader.cancel().catch(() => {});
              prealloc = null;
              parts = null;
              return {
                bytes: null,
                skipped: true,
                skipReason: 'exceeds-preload-threshold',
                totalBytes: declared,
                transport: 'stream-deferred',
                chunksFetched: 0,
                parallelMode: false,
              };
            }
            if (prealloc !== null) {
              if (loaded + part.byteLength <= prealloc.byteLength) {
                prealloc.set(part, loaded);
              } else {
                parts = loaded > 0 ? [prealloc.slice(0, loaded), part] : [part];
                prealloc = null;
              }
            } else {
              parts.push(part);
            }
            loaded += part.byteLength;
            if (telemetryRun) {
              telemetryRun.recordBytes(part.byteLength);
              telemetryRun.recordNetworkBytes(part.byteLength);
            }
            emit('download', loaded, declared);
          }
        } catch (err) {
          reader.cancel(err).catch(() => {});
          throwIfAborted(signal);
          throw err;
        }
        throwIfAborted(signal);
        if (prealloc !== null) {
          bytes = loaded === prealloc.byteLength ? prealloc : prealloc.slice(0, loaded);
          prealloc = null;
        } else if (parts.length === 1) {
          bytes = parts[0];
          parts.length = 0;
        } else {
          bytes = new Uint8Array(loaded);
          let offset = 0;
          for (const part of parts) {
            bytes.set(part, offset);
            offset += part.byteLength;
          }
          parts.length = 0;
        }
        if (telemetryRun) {
          if (ownsTotalBytes) telemetryRun.setTotalBytes(bytes.byteLength);
          telemetryRun.recordChunk(1);
        }
      } else {
        bytes = new Uint8Array(await raceAbort(response.arrayBuffer(), signal));
        throwIfAborted(signal);
        if (telemetryRun) {
          if (ownsTotalBytes) telemetryRun.setTotalBytes(bytes.byteLength);
          telemetryRun.recordBytes(bytes.byteLength);
          telemetryRun.recordNetworkBytes(bytes.byteLength);
          telemetryRun.recordChunk(1);
        }
      }
      emit('complete', bytes.byteLength, bytes.byteLength);
      return { bytes, transport: 'full', chunksFetched: 1, parallelMode: false };
    }
  }


  /**
   * Convenience wrapper around `PgimPreloader.prototype.preloadProgramModules`.
   *
   * @param {string | object} manifestUrl
   * @param {object} [options]
   */
  async function preloadProgramModules(manifestUrl, options = {}) {
    let url = manifestUrl;
    let opts = options;
    if (isPlainObject(manifestUrl)) {
      opts = manifestUrl;
      url = opts.programManifestUrl ?? opts.manifestUrl;
    }
    const {
      preloader = null,
      fetchImpl,
      rangeCache = null,
      opfsCache = DEFAULT_OPFS_CACHE_MODE,
      opfsCacheVersion,
      opfsCacheDirName,
      opfsValidateWithHead,
      opfsStorageObj,
      chunkSize,
      chunkBytes,
      concurrency,
      maxRetries,
      maxVersionRetries,
      retryDelayMs,
      baseRetryDelayMs,
      maxRetryDelayMs,
      sleepFn,
      sleep,
      nowFn,
      useSharedArrayBuffer,
      targetBuffer,
      ...callOptions
    } = opts || {};
    const instance =
      preloader instanceof PgimPreloader
        ? preloader
        : new PgimPreloader({
            fetchImpl,
            rangeCache,
            ...(opfsCache !== undefined ? { opfsCache } : {}),
            ...(opfsCacheVersion !== undefined ? { opfsCacheVersion } : {}),
            ...(opfsCacheDirName !== undefined ? { opfsCacheDirName } : {}),
            ...(opfsValidateWithHead !== undefined ? { opfsValidateWithHead } : {}),
            ...(opfsStorageObj !== undefined ? { opfsStorageObj } : {}),
            ...(chunkSize !== undefined ? { chunkSize } : {}),
            ...(chunkBytes !== undefined ? { chunkBytes } : {}),
            ...(concurrency !== undefined ? { concurrency } : {}),
            ...(maxRetries !== undefined ? { maxRetries } : {}),
            ...(maxVersionRetries !== undefined ? { maxVersionRetries } : {}),
            ...(retryDelayMs !== undefined ? { retryDelayMs } : {}),
            ...(baseRetryDelayMs !== undefined ? { baseRetryDelayMs } : {}),
            ...(maxRetryDelayMs !== undefined ? { maxRetryDelayMs } : {}),
            ...(sleepFn !== undefined ? { sleepFn } : {}),
            ...(sleep !== undefined ? { sleep } : {}),
            ...(nowFn !== undefined ? { nowFn } : {}),
            ...(useSharedArrayBuffer !== undefined ? { useSharedArrayBuffer } : {}),
            ...(targetBuffer !== undefined ? { targetBuffer } : {}),
          });
    return instance.preloadProgramModules(url, callOptions);
  }

  const exportedApi = Object.freeze({
    PGIM_MAGIC,
    PGIM_MAGIC_BYTES,
    PGIM_VERSION,
    PGIM_HEADER_BYTES,
    PGIM_SECTION_ENTRY_BYTES,
    PGIM_PAGE_SIZE,
    PGIM_FLAGS,
    PGIM_SECTION_FLAGS,
    PGIM_PROT,
    DEFAULT_CHUNK_BYTES,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_VERSION_RETRIES,
    PGIM_OPFS_CACHE_VERSION,
    PGIM_OPFS_PROVENANCE_SCHEMA,
    DEFAULT_OPFS_CACHE_DIR,
    DEFAULT_OPFS_CACHE_MODE,
    DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES,
    DEFAULT_AUTO_STREAM_SHADER_THRESHOLD_BYTES,
    PROGRAM_MANIFEST_SCHEMA,
    PROGRAM_MANIFEST_SCHEMA_VERSION,
    PgimPreloadError,
    pgimFnv1a64,
    isStrongEtag,
    isWeakEtag,
    parsePgimHeader,
    parsePgimLayout,
    resolveSafeManifestUrl,
    sanitizePgimCacheUrlPath,
    buildPgimOpfsFingerprint,
    buildPgimOpfsCacheKey,
    buildManifestOpfsFingerprint,
    parseProgramManifest,
    parseBootManifest,
    preloadProgramModules,
    collectTransferableBuffers,
    releaseBootBundleBytes,
    PgimPreloader,
    default: PgimPreloader,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6PgimPreloader = exportedApi;
  } else if (root) {
    root.PlayGta6PgimPreloader = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
