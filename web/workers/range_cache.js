'use strict';

// Fixed-size HTTP 206 Range chunk cache with memory LRU, optional OPFS L2 persistence,
// concurrent in-flight deduplication, coalesced adjacent range merging, adaptive
// sequential read-ahead prefetching, and bootset.json prefetching.
// Compatible with classic Web Workers (importScripts) and Node CommonJS/ESM tests.

const DEFAULT_CHUNK_SIZE = 256 * 1024; // 256 KiB
const DEFAULT_MAX_BYTES = 64 * 1024 * 1024; // 64 MiB
const DEFAULT_PREFETCH_CONCURRENCY = 4;
const DEFAULT_COALESCE_MAX_GAP_BYTES = 64 * 1024; // 64 KiB
const DEFAULT_COALESCE_MAX_MERGED_BYTES = 2 * 1024 * 1024; // 2 MiB
const DEFAULT_MAX_READ_AHEAD_BYTES = 2 * 1024 * 1024; // 2 MiB (legacy W181 constant)
const DEFAULT_MIN_READAHEAD_BYTES = 256 * 1024; // 256 KiB (W312 adaptive floor)
const DEFAULT_MAX_READAHEAD_BYTES = 4 * 1024 * 1024; // 4 MiB (W312 adaptive ceiling)
const DEFAULT_SEQUENTIAL_STREAK_THRESHOLD = 2;
const DEFAULT_MAX_PENDING_OPFS_WRITES = 64;
const DEFAULT_MAX_PENDING_OPFS_WRITE_BYTES = 16 * 1024 * 1024; // 16 MiB byte cap on queued OPFS writes
const DEFAULT_MAX_OPFS_WRITE_CONCURRENCY = 4;
const ESTIMATED_HTTP_RANGE_OVERHEAD_BYTES = 256;
const MIN_AUTO_OPTIMIZE_CHUNK_SIZE = 16 * 1024; // 16 KiB
const MIN_AUTO_OPTIMIZE_CACHE_BYTES = 1024 * 1024; // 1 MiB
const MAX_TRACKED_CHUNK_LENGTHS = 65536;
const MAX_TRACKED_FILE_SIZES = 4096;
const MAX_TRACKED_SEQUENTIAL_KEYS = 1024;
const MAX_TRACKED_OPFS_ENTRIES = 16384;
const MAX_NORMALIZED_PATH_CACHE = 2048;
const SLRU_MIN_CHUNKS = 8;
const SHARED_TEXT_ENCODER = typeof TextEncoder !== 'undefined' ? new TextEncoder() : null;
const EMPTY_UINT8 = new Uint8Array(0);

function isQuotaExceededError(err) {
  return Boolean(
    err &&
      (err.name === 'QuotaExceededError' ||
        err.code === 22 ||
        err.code === 1014 ||
        /quota/i.test(String(err.message || ''))),
  );
}

function isNotFoundError(err) {
  return Boolean(
    err &&
      (err.name === 'NotFoundError' ||
        err.code === 'ENOENT' ||
        err.code === 8 ||
        /not\s*found/i.test(String(err.message || ''))),
  );
}

function normalizeDataBase(base) {
  const raw = typeof base === 'string' && base.length > 0 ? base : '/data/';
  return raw.endsWith('/') ? raw : `${raw}/`;
}

function normalizeRelPath(relPath) {
  if (typeof relPath !== 'string' || relPath.length === 0) {
    throw new Error('RangeCache path must be a non-empty string');
  }
  if (relPath.includes('\\') || relPath.includes('\0')) {
    throw new Error(`Invalid RangeCache path: ${relPath}`);
  }
  const stripped = relPath.replace(/^\/+/, '');
  if (!stripped) {
    throw new Error(`Invalid RangeCache path: ${relPath}`);
  }
  const parts = stripped.split('/');
  for (const part of parts) {
    if (!part || part === '.' || part === '..') {
      throw new Error(`Invalid RangeCache path segment in: ${relPath}`);
    }
  }
  return stripped;
}

function utf8ToHex(str) {
  const bytes = SHARED_TEXT_ENCODER ? SHARED_TEXT_ENCODER.encode(str) : new TextEncoder().encode(str);
  let hex = '';
  for (let i = 0; i < bytes.length; i += 1) {
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  return hex;
}

function fnv1a64Hex(str) {
  // Deterministic two-lane 32-bit FNV-1a hash (128-bit hex) for long path keys.
  const bytes = SHARED_TEXT_ENCODER ? SHARED_TEXT_ENCODER.encode(str) : new TextEncoder().encode(str);
  let h1 = 0x811c9dc5;
  let h2 = 0x811c9dc5 ^ 0x9e3779b9;
  let h3 = 0x811c9dc5 ^ 0x85ebca6b;
  let h4 = 0x811c9dc5 ^ 0xc2b2ae35;
  for (let i = 0; i < bytes.length; i += 1) {
    const b = bytes[i];
    h1 = Math.imul(h1 ^ b, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ ((b + i) & 0xff), 0x01000193) >>> 0;
    h3 = Math.imul(h3 ^ (b ^ 0xa5), 0x01000193) >>> 0;
    h4 = Math.imul(h4 ^ ((b + (i * 31)) & 0xff), 0x01000193) >>> 0;
  }
  return (
    h1.toString(16).padStart(8, '0') +
    h2.toString(16).padStart(8, '0') +
    h3.toString(16).padStart(8, '0') +
    h4.toString(16).padStart(8, '0')
  );
}

function buildFileFingerprint(sizeOrMeta, etag = null) {
  if (sizeOrMeta === null || sizeOrMeta === undefined) {
    if (typeof etag === 'string' && etag.trim().length > 0) {
      return `e_${fnv1a64Hex(etag.trim()).slice(0, 16)}`;
    }
    return null;
  }
  if (typeof sizeOrMeta === 'string') {
    const trimmed = sizeOrMeta.trim();
    if (!trimmed) return null;
    if (/^[0-9a-zA-Z_-]{1,64}$/.test(trimmed)) {
      return trimmed;
    }
    return `h_${fnv1a64Hex(trimmed).slice(0, 16)}`;
  }

  let size = null;
  let tag = etag;
  if (typeof sizeOrMeta === 'object') {
    if (typeof sizeOrMeta.fileFingerprint === 'string' && sizeOrMeta.fileFingerprint.trim()) {
      return buildFileFingerprint(sizeOrMeta.fileFingerprint);
    }
    if (sizeOrMeta.fileSize !== undefined) {
      size = sizeOrMeta.fileSize;
    } else if (sizeOrMeta.size !== undefined) {
      size = sizeOrMeta.size;
    }
    if (tag === null && sizeOrMeta.etag !== undefined) {
      tag = sizeOrMeta.etag;
    }
  } else {
    size = sizeOrMeta;
  }

  const validSize = Number.isSafeInteger(size) && size >= 0 ? size : null;
  const validTag = typeof tag === 'string' && tag.trim().length > 0 ? tag.trim() : null;

  if (validSize !== null && validTag !== null) {
    return `${validSize}_${fnv1a64Hex(validTag).slice(0, 16)}`;
  }
  if (validSize !== null) {
    return `${validSize}`;
  }
  if (validTag !== null) {
    return `e_${fnv1a64Hex(validTag).slice(0, 16)}`;
  }
  return null;
}

function readResponseHeader(headers, name) {
  if (!headers) return null;
  if (typeof headers.get === 'function') {
    return headers.get(name) || headers.get(name.toLowerCase()) || null;
  }
  if (typeof headers === 'object') {
    return headers[name] || headers[name.toLowerCase()] || null;
  }
  return null;
}

function opfsChunkFilePrefix(relPath, chunkIndex, chunkSize) {
  const encoded = utf8ToHex(relPath);
  const pathToken = encoded.length <= 128 ? encoded : `${encoded.slice(0, 64)}_${fnv1a64Hex(relPath)}`;
  return `c_${chunkSize}_${chunkIndex}_${pathToken}`;
}

function opfsChunkFileName(relPath, chunkIndex, chunkSize, fileFingerprint = null) {
  const prefix = opfsChunkFilePrefix(relPath, chunkIndex, chunkSize);
  const fp = buildFileFingerprint(fileFingerprint);
  if (fp) {
    return `${prefix}_fp_${fp}.bin`;
  }
  return `${prefix}.bin`;
}

function parseContentRange(header) {
  if (typeof header !== 'string') return null;
  const match = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i.exec(header.trim());
  if (!match) return null;
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = match[3] === '*' ? null : Number(match[3]);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {
    return null;
  }
  return { start, end, total };
}

function parseUnsatisfiedContentRangeTotal(header) {
  if (typeof header !== 'string') return null;
  const match = /^bytes\s+\*\/(\d+)$/i.exec(header.trim());
  if (!match) return null;
  const total = Number(match[1]);
  return Number.isSafeInteger(total) && total >= 0 ? total : null;
}

function normalizeInputRange(rawRange) {
  if (!rawRange || typeof rawRange !== 'object') return null;
  let offset;
  let length;

  if (Number.isFinite(rawRange.offset)) {
    offset = Math.floor(rawRange.offset);
  } else if (Number.isFinite(rawRange.start)) {
    offset = Math.floor(rawRange.start);
  } else {
    return null;
  }

  if (Number.isFinite(rawRange.length)) {
    length = Math.floor(rawRange.length);
  } else if (Number.isFinite(rawRange.end) && rawRange.end > offset) {
    length = Math.floor(rawRange.end) - offset;
  } else {
    return null;
  }

  if (offset < 0 || length <= 0) {
    return null;
  }

  const hasPath = typeof rawRange.path === 'string' && rawRange.path.length > 0;
  return {
    hasPath,
    path: hasPath ? rawRange.path : null,
    offset,
    length,
  };
}

function coalesceAdjacentRanges(
  ranges,
  maxGapBytes = DEFAULT_COALESCE_MAX_GAP_BYTES,
  maxMergedBytes = DEFAULT_COALESCE_MAX_MERGED_BYTES,
) {
  if (!Array.isArray(ranges) || ranges.length === 0) {
    return [];
  }

  const maxGap =
    Number.isFinite(maxGapBytes) && maxGapBytes >= 0
      ? Math.floor(maxGapBytes)
      : DEFAULT_COALESCE_MAX_GAP_BYTES;
  const maxMerged =
    Number.isFinite(maxMergedBytes) && maxMergedBytes > 0
      ? Math.floor(maxMergedBytes)
      : DEFAULT_COALESCE_MAX_MERGED_BYTES;

  const groups = new Map();
  for (let i = 0; i < ranges.length; i += 1) {
    const normalized = normalizeInputRange(ranges[i]);
    if (!normalized) continue;
    const groupKey = normalized.hasPath ? `p:${normalized.path}` : '__pathless__';
    let list = groups.get(groupKey);
    if (!list) {
      list = [];
      groups.set(groupKey, list);
    }
    list.push(normalized);
  }

  const merged = [];
  for (const list of groups.values()) {
    list.sort((a, b) => (a.offset !== b.offset ? a.offset - b.offset : a.length - b.length));

    let current = {
      hasPath: list[0].hasPath,
      path: list[0].path,
      offset: list[0].offset,
      length: list[0].length,
      mergedCount: 1,
      gapBytes: 0,
      originalBytes: list[0].length,
    };

    const pushCurrent = () => {
      const unionBytes = current.length - current.gapBytes;
      const overlapBytes = Math.max(0, current.originalBytes - unionBytes);
      const out = {
        offset: current.offset,
        length: current.length,
        end: current.offset + current.length,
        mergedCount: current.mergedCount,
        gapBytes: current.gapBytes,
        originalBytes: current.originalBytes,
        overlapBytes,
      };
      if (current.hasPath) {
        out.path = current.path;
      }
      merged.push(out);
    };

    for (let i = 1; i < list.length; i += 1) {
      const item = list[i];
      const curEnd = current.offset + current.length;
      const itemEnd = item.offset + item.length;
      const gap = item.offset > curEnd ? item.offset - curEnd : 0;
      const candidateEnd = Math.max(curEnd, itemEnd);
      const candidateLength = candidateEnd - current.offset;
      const isContained = item.offset >= current.offset && itemEnd <= curEnd;

      if (gap <= maxGap && (candidateLength <= maxMerged || isContained)) {
        current.length = candidateLength;
        current.mergedCount += 1;
        current.gapBytes += gap;
        current.originalBytes += item.length;
      } else {
        pushCurrent();
        current = {
          hasPath: item.hasPath,
          path: item.path,
          offset: item.offset,
          length: item.length,
          mergedCount: 1,
          gapBytes: 0,
          originalBytes: item.length,
        };
      }
    }

    pushCurrent();
  }

  return merged;
}

function detectSequentialRead(prevState, offset, length, options = {}) {
  const chunkSize =
    Number.isInteger(options.chunkSize) && options.chunkSize > 0
      ? options.chunkSize
      : DEFAULT_CHUNK_SIZE;
  const rawMaxReadAhead =
    options.maxReadaheadBytes !== undefined
      ? options.maxReadaheadBytes
      : options.maxReadAheadBytes;
  const maxReadAheadBytes =
    Number.isFinite(rawMaxReadAhead) && rawMaxReadAhead >= 0
      ? Math.floor(rawMaxReadAhead)
      : DEFAULT_MAX_READ_AHEAD_BYTES;
  const rawMinReadAhead =
    options.minReadaheadBytes !== undefined
      ? options.minReadaheadBytes
      : options.minReadAheadBytes !== undefined
        ? options.minReadAheadBytes
        : options.initialReadAheadBytes;
  const minReadaheadBytes =
    maxReadAheadBytes > 0
      ? Math.min(
          maxReadAheadBytes,
          Number.isFinite(rawMinReadAhead) && rawMinReadAhead > 0
            ? Math.floor(rawMinReadAhead)
            : Math.max(chunkSize, length),
        )
      : 0;
  const maxGapBytes =
    Number.isFinite(options.maxGapBytes) && options.maxGapBytes >= 0
      ? Math.floor(options.maxGapBytes)
      : Math.min(DEFAULT_COALESCE_MAX_GAP_BYTES, chunkSize);
  const maxOverlapBytes =
    Number.isFinite(options.maxOverlapBytes) && options.maxOverlapBytes >= 0
      ? Math.floor(options.maxOverlapBytes)
      : Math.min(DEFAULT_COALESCE_MAX_GAP_BYTES, Math.floor(chunkSize / 2));
  const streakThreshold =
    Number.isInteger(options.streakThreshold) && options.streakThreshold >= 1
      ? options.streakThreshold
      : DEFAULT_SEQUENTIAL_STREAK_THRESHOLD;
  const growthFactor =
    Number.isFinite(options.growthFactor) && options.growthFactor >= 1
      ? options.growthFactor
      : 2;
  const shrinkFactor =
    Number.isFinite(options.shrinkFactor) && options.shrinkFactor > 0 && options.shrinkFactor < 1
      ? options.shrinkFactor
      : 0.5;

  const end = offset + length;
  let isSequential = false;
  let isRandomSeek = false;
  let gapBytes = 0;
  let stride = length;
  let streak = 0;
  let readAheadBytes = 0;
  let adaptiveWindowBytes = minReadaheadBytes;

  if (
    prevState &&
    Number.isFinite(prevState.lastOffset) &&
    Number.isFinite(prevState.lastEnd)
  ) {
    const prevOffset = prevState.lastOffset;
    const prevEnd = prevState.lastEnd;
    const prevStreak = Number.isInteger(prevState.streak) && prevState.streak >= 0 ? prevState.streak : 0;
    const prevAdaptiveWindow =
      maxReadAheadBytes > 0
        ? Math.min(
            maxReadAheadBytes,
            Math.max(
              minReadaheadBytes,
              Number.isFinite(prevState.adaptiveWindowBytes) && prevState.adaptiveWindowBytes > 0
                ? Math.floor(prevState.adaptiveWindowBytes)
                : Number.isFinite(prevState.readAheadBytes) && prevState.readAheadBytes > 0
                  ? Math.floor(prevState.readAheadBytes)
                  : minReadaheadBytes,
            ),
          )
        : 0;

    if (offset === prevEnd) {
      isSequential = true;
      gapBytes = 0;
      stride = offset - prevOffset;
      streak = prevStreak + 1;
    } else if (offset > prevEnd && offset - prevEnd <= maxGapBytes) {
      isSequential = true;
      gapBytes = offset - prevEnd;
      stride = offset - prevOffset;
      streak = prevStreak + 1;
    } else if (
      offset > prevOffset &&
      offset < prevEnd &&
      end > prevEnd &&
      prevEnd - offset <= maxOverlapBytes
    ) {
      isSequential = true;
      gapBytes = 0;
      stride = offset - prevOffset;
      streak = prevStreak + 1;
    } else {
      isSequential = false;
      isRandomSeek = true;
      gapBytes = 0;
      stride = length;
      streak = 0;
    }

    if (isRandomSeek) {
      adaptiveWindowBytes =
        maxReadAheadBytes > 0
          ? Math.max(minReadaheadBytes, Math.floor(prevAdaptiveWindow * shrinkFactor))
          : 0;
    } else if (isSequential && streak >= streakThreshold && maxReadAheadBytes > 0) {
      const prevWindow =
        Number.isFinite(prevState.readAheadBytes) && prevState.readAheadBytes > 0
          ? prevState.readAheadBytes
          : 0;
      if (prevWindow > 0) {
        readAheadBytes = Math.min(
          maxReadAheadBytes,
          Math.max(minReadaheadBytes, Math.floor(prevWindow * growthFactor)),
        );
      } else {
        const baseWindow = Math.min(maxReadAheadBytes, Math.max(minReadaheadBytes, prevAdaptiveWindow));
        const stepsAboveThreshold = streak - streakThreshold;
        const scaled = Math.floor(baseWindow * Math.pow(growthFactor, stepsAboveThreshold));
        readAheadBytes = Math.min(maxReadAheadBytes, Math.max(baseWindow, scaled));
      }
      adaptiveWindowBytes = readAheadBytes;
    } else {
      adaptiveWindowBytes = prevAdaptiveWindow;
    }
  }

  return {
    isSequential,
    isRandomSeek,
    streak,
    gapBytes,
    stride,
    readAheadBytes,
    adaptiveWindowBytes,
    nextState: {
      lastOffset: offset,
      lastLength: length,
      lastEnd: end,
      stride,
      streak,
      readAheadBytes,
      adaptiveWindowBytes,
    },
  };
}

async function openOpfsCacheDir(subdir = 'playgta6-cache', storageObj = null) {
  const storage =
    storageObj ||
    (typeof navigator !== 'undefined' && navigator && navigator.storage ? navigator.storage : null);
  if (!storage || typeof storage.getDirectory !== 'function') {
    return null;
  }
  try {
    const root = await storage.getDirectory();
    if (!root || typeof root.getDirectoryHandle !== 'function') {
      return null;
    }
    return await root.getDirectoryHandle(subdir, { create: true });
  } catch {
    return null;
  }
}

class RangeCache {
  constructor(options = {}) {
    this.dataBase = normalizeDataBase(options.dataBase);
    this.chunkSize =
      Number.isInteger(options.chunkSize) && options.chunkSize > 0
        ? options.chunkSize
        : DEFAULT_CHUNK_SIZE;
    this.maxBytes =
      Number.isFinite(options.maxBytes) && options.maxBytes >= 0
        ? Math.floor(options.maxBytes)
        : DEFAULT_MAX_BYTES;
    this.maxPrefetchConcurrency =
      Number.isInteger(options.maxPrefetchConcurrency) && options.maxPrefetchConcurrency > 0
        ? options.maxPrefetchConcurrency
        : DEFAULT_PREFETCH_CONCURRENCY;
    this.fetchImpl = options.fetchImpl || (typeof fetch === 'function' ? fetch.bind(globalThis) : null);
    this.opfsDir = options.opfsDir || null;
    this.opfsCache = options.opfsCache || null;
    this.backgroundOpfsWrites = options.backgroundOpfsWrites !== false;
    this.maxPendingOpfsWrites =
      Number.isInteger(options.maxPendingOpfsWrites) && options.maxPendingOpfsWrites > 0
        ? options.maxPendingOpfsWrites
        : DEFAULT_MAX_PENDING_OPFS_WRITES;
    this.maxPendingOpfsWriteBytes =
      Number.isFinite(options.maxPendingOpfsWriteBytes) && options.maxPendingOpfsWriteBytes > 0
        ? Math.floor(options.maxPendingOpfsWriteBytes)
        : DEFAULT_MAX_PENDING_OPFS_WRITE_BYTES;
    this.maxOpfsWriteConcurrency =
      Number.isInteger(options.maxOpfsWriteConcurrency) && options.maxOpfsWriteConcurrency > 0
        ? options.maxOpfsWriteConcurrency
        : DEFAULT_MAX_OPFS_WRITE_CONCURRENCY;

    this._explicitCoalesceMaxMergedBytes = options.coalesceMaxMergedBytes !== undefined;
    this.coalesceMaxGapBytes =
      Number.isFinite(options.coalesceMaxGapBytes) && options.coalesceMaxGapBytes >= 0
        ? Math.floor(options.coalesceMaxGapBytes)
        : DEFAULT_COALESCE_MAX_GAP_BYTES;
    this.coalesceMaxMergedBytes =
      Number.isFinite(options.coalesceMaxMergedBytes) && options.coalesceMaxMergedBytes > 0
        ? Math.floor(options.coalesceMaxMergedBytes)
        : DEFAULT_COALESCE_MAX_MERGED_BYTES;
    const rawMaxReadahead =
      options.maxReadaheadBytes !== undefined
        ? options.maxReadaheadBytes
        : options.maxReadAheadBytes;
    this.maxReadaheadBytes =
      Number.isFinite(rawMaxReadahead) && rawMaxReadahead >= 0
        ? Math.floor(rawMaxReadahead)
        : DEFAULT_MAX_READAHEAD_BYTES;
    this.maxReadAheadBytes = this.maxReadaheadBytes;
    const rawMinReadahead =
      options.minReadaheadBytes !== undefined
        ? options.minReadaheadBytes
        : options.minReadAheadBytes;
    const requestedMinReadahead =
      Number.isFinite(rawMinReadahead) && rawMinReadahead > 0
        ? Math.floor(rawMinReadahead)
        : DEFAULT_MIN_READAHEAD_BYTES;
    this.minReadaheadBytes =
      this.maxReadaheadBytes > 0 ? Math.min(requestedMinReadahead, this.maxReadaheadBytes) : 0;
    this.sequentialStreakThreshold =
      Number.isInteger(options.sequentialStreakThreshold) && options.sequentialStreakThreshold >= 1
        ? options.sequentialStreakThreshold
        : DEFAULT_SEQUENTIAL_STREAK_THRESHOLD;
    this.sequentialMaxGapBytes =
      Number.isFinite(options.sequentialMaxGapBytes) && options.sequentialMaxGapBytes >= 0
        ? Math.floor(options.sequentialMaxGapBytes)
        : Math.min(this.coalesceMaxGapBytes, this.chunkSize);

    const autoOptimize =
      this.chunkSize >= MIN_AUTO_OPTIMIZE_CHUNK_SIZE &&
      this.maxBytes >= MIN_AUTO_OPTIMIZE_CACHE_BYTES;

    if (typeof options.coalesceAdjacent === 'boolean') {
      this.coalesceAdjacent = options.coalesceAdjacent;
    } else if (
      options.coalesceMaxGapBytes !== undefined ||
      options.coalesceMaxMergedBytes !== undefined
    ) {
      this.coalesceAdjacent = true;
    } else {
      this.coalesceAdjacent = autoOptimize;
    }

    const explicitReadAhead =
      typeof options.enableReadAhead === 'boolean'
        ? options.enableReadAhead
        : typeof options.readAhead === 'boolean'
          ? options.readAhead
          : null;
    if (explicitReadAhead !== null) {
      this.enableReadAhead = explicitReadAhead && this.maxReadaheadBytes > 0;
    } else if (
      options.maxReadaheadBytes !== undefined ||
      options.maxReadAheadBytes !== undefined ||
      options.minReadaheadBytes !== undefined ||
      options.minReadAheadBytes !== undefined
    ) {
      this.enableReadAhead = this.maxReadaheadBytes > 0;
    } else {
      this.enableReadAhead = autoOptimize && this.maxReadaheadBytes > 0;
    }

    this._memCache = new Map();
    this._probationChunks = new Map(); // key -> byteLength (SLRU probation LRU order)
    this._protectedChunks = new Map(); // key -> byteLength (SLRU protected LRU order)
    this._probationPrefetched = new Set(); // O(1) probation prefetched/readahead eviction index
    this._memMruKey = null;
    this._probationMruKey = null;
    this._protectedMruKey = null;
    this._opfsMruFileName = null;
    this._seqTrackerMruKey = null;
    this._protectedBytesCached = 0;
    this._chunkSwitchCount = 0;
    this._lastAccessedChunkKey = null;
    this._chunkLastSwitch = new Map();
    this._memChunkFingerprints = new Map();
    this._pathChunkKeys = new Map(); // cleanPath -> Map<chunkIndex, key> (resident chunk key intern cache)
    this._relPathCache = new Map();
    this._normalizedPathCache = new Map();
    this._normalizedPathCacheBase = this.dataBase;
    this._readIntoKeys = [];
    this._readIntoChunks = [];
    this._seqDetectOpts = {
      chunkSize: this.chunkSize,
      maxGapBytes: this.sequentialMaxGapBytes,
      minReadaheadBytes: this.enableReadAhead ? this.minReadaheadBytes : 0,
      maxReadaheadBytes: this.enableReadAhead ? this.maxReadaheadBytes : 0,
      streakThreshold: this.sequentialStreakThreshold,
    };
    this._inFlight = new Map();
    this._inFlightOpfsLookups = new Map();
    this._opfsEntries = new Map();
    this._keyToOpfsFile = new Map();
    this._fileFingerprints = new Map();
    this._observedHttpMeta = new Map();
    this._opfsWriteQueue = new Map();
    this._opfsQueuedBytes = 0;
    this._activeOpfsWrites = new Set();
    this._opfsWriteWaiters = [];
    this._opfsDirKnownFiles = null;
    this._opfsDirIndexPromise = null;
    this._opfsDirIndexedStore = null;
    this._quotaEvictChain = Promise.resolve();
    this._expectedChunkLengths = new Map();
    this._fileSizes = new Map();
    this._seqTrackers = new Map();
    this._prefetchedKeys = new Set();
    this._readaheadKeys = new Map();
    this._readaheadByPath = new Map();
    this._inFlightPrefetchKeys = new Set();
    this._inFlightReadaheadKeys = new Set();
    this._activePrefetches = new Set();
    this._activeReadAheadBatches = 0;
    this._bytesCached = 0;
    this._stats = this._createEmptyStats();

    if (options.fileFingerprints instanceof Map) {
      for (const [p, fp] of options.fileFingerprints.entries()) {
        this.setFileFingerprint(p, fp);
      }
    } else if (options.fileFingerprints && typeof options.fileFingerprints === 'object') {
      for (const [p, fp] of Object.entries(options.fileFingerprints)) {
        this.setFileFingerprint(p, fp);
      }
    }
  }

  _createEmptyStats() {
    return {
      hits: 0,
      misses: 0,
      opfsHits: 0,
      opfsMisses: 0,
      opfsWrites: 0,
      opfsWriteErrors: 0,
      opfsStaleInvalidations: 0,
      evictions: 0,
      quotaEvictions: 0,
      quotaRetrySuccesses: 0,
      corruptEvictions: 0,
      coalescedRangeCount: 0,
      sequentialHits: 0,
      prefetchHits: 0,
      prefetchBytes: 0,
      bytesSavedByCoalescing: 0,
      adaptiveReadaheadWindowBytes: 0,
      sequentialStreakPeak: 0,
      readaheadHits: 0,
      readaheadWastedBytes: 0,
      zeroAllocCopyHits: 0,
      mruFastPathHits: 0,
    };
  }

  _normalizeRelPathCached(relPath) {
    if (typeof relPath === 'string') {
      const cached = this._relPathCache.get(relPath);
      if (cached !== undefined) {
        return cached;
      }
    }
    const clean = normalizeRelPath(relPath);
    if (this._relPathCache.size >= MAX_NORMALIZED_PATH_CACHE) {
      const oldest = this._relPathCache.keys().next().value;
      if (oldest !== undefined) {
        this._relPathCache.delete(oldest);
      }
    }
    this._relPathCache.set(relPath, clean);
    return clean;
  }

  _normalizePathOrUrl(urlOrPath) {
    if (typeof urlOrPath !== 'string' || urlOrPath.length === 0) {
      throw new Error('RangeCache path must be a non-empty string');
    }
    if (this._normalizedPathCacheBase !== this.dataBase) {
      this._normalizedPathCache.clear();
      this._normalizedPathCacheBase = this.dataBase;
    } else {
      const cached = this._normalizedPathCache.get(urlOrPath);
      if (cached !== undefined) {
        return cached;
      }
    }
    let candidate = urlOrPath;
    if (candidate.startsWith(this.dataBase)) {
      candidate = candidate.slice(this.dataBase.length);
    } else if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(candidate)) {
      candidate = candidate.replace(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^/]*/, '');
      candidate = candidate.replace(/[?#].*$/, '');
      const basePath = this.dataBase.replace(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^/]*/, '');
      if (basePath && candidate.startsWith(basePath)) {
        candidate = candidate.slice(basePath.length);
      }
    }
    const clean = normalizeRelPath(candidate);
    if (this._normalizedPathCache.size >= MAX_NORMALIZED_PATH_CACHE) {
      const oldest = this._normalizedPathCache.keys().next().value;
      if (oldest !== undefined) {
        this._normalizedPathCache.delete(oldest);
      }
    }
    this._normalizedPathCache.set(urlOrPath, clean);
    return clean;
  }

  _trackPrefetchedKey(key, relPath, chunkIndex, isReadahead, byteLength) {
    this._prefetchedKeys.add(key);
    if (isReadahead) {
      this._readaheadKeys.set(key, byteLength);
      let byPath = this._readaheadByPath.get(relPath);
      if (byPath === undefined) {
        byPath = new Map();
        this._readaheadByPath.set(relPath, byPath);
      }
      byPath.set(chunkIndex, key);
    }
    if (this._probationChunks.has(key)) {
      this._probationPrefetched.add(key);
    }
  }

  _deleteReadaheadKeyInternal(key) {
    const byteLength = this._readaheadKeys.get(key);
    if (byteLength === undefined) {
      return undefined;
    }
    this._readaheadKeys.delete(key);
    const sep = key.lastIndexOf('::');
    if (sep > 0) {
      const relPath = key.slice(0, sep);
      const byPath = this._readaheadByPath.get(relPath);
      if (byPath !== undefined) {
        byPath.delete(Number(key.slice(sep + 2)));
        if (byPath.size === 0) {
          this._readaheadByPath.delete(relPath);
        }
      }
    }
    if (!this._prefetchedKeys.has(key)) {
      this._probationPrefetched.delete(key);
    }
    return byteLength;
  }

  _recordReadaheadWastedKey(key) {
    const wastedBytes = this._deleteReadaheadKeyInternal(key);
    if (wastedBytes !== undefined && wastedBytes > 0) {
      this._stats.readaheadWastedBytes += wastedBytes;
    }
  }

  _accountAbandonedReadaheadOnSeek(relPath, keepFirstChunk, keepLastChunk) {
    if (this._readaheadKeys.size === 0) {
      return;
    }
    const byPath = this._readaheadByPath.get(relPath);
    if (byPath === undefined || byPath.size === 0) {
      return;
    }
    for (const [chunkIndex, key] of byPath) {
      if (chunkIndex >= keepFirstChunk && chunkIndex <= keepLastChunk) {
        continue;
      }
      byPath.delete(chunkIndex);
      const byteLength = this._readaheadKeys.get(key);
      if (byteLength !== undefined) {
        this._readaheadKeys.delete(key);
        if (!this._prefetchedKeys.has(key)) {
          this._probationPrefetched.delete(key);
        }
        if (byteLength > 0) {
          this._stats.readaheadWastedBytes += byteLength;
        }
      }
    }
    if (byPath.size === 0) {
      this._readaheadByPath.delete(relPath);
    }
  }

  _resetOpfsDirIndex() {
    this._opfsDirKnownFiles = null;
    this._opfsDirIndexPromise = null;
    this._opfsDirIndexedStore = null;
  }

  _supportsOpfsDirIndex(dir) {
    return Boolean(
      dir &&
        typeof dir.getFileHandle === 'function' &&
        (typeof dir.keys === 'function' || typeof dir.entries === 'function'),
    );
  }

  _ensureOpfsDirIndex(dir) {
    if (!this._supportsOpfsDirIndex(dir)) {
      return Promise.resolve(null);
    }
    if (this._opfsDirIndexedStore !== dir) {
      this._opfsDirKnownFiles = null;
      this._opfsDirIndexPromise = null;
      this._opfsDirIndexedStore = dir;
    }
    if (this._opfsDirKnownFiles !== null) {
      return Promise.resolve(this._opfsDirKnownFiles);
    }
    if (this._opfsDirIndexPromise !== null) {
      return this._opfsDirIndexPromise;
    }
    const targetDir = dir;
    const promise = (async () => {
      const known = new Set();
      try {
        if (typeof targetDir.keys === 'function') {
          for await (const name of targetDir.keys()) {
            if (typeof name === 'string') {
              known.add(name);
            }
          }
        } else if (typeof targetDir.entries === 'function') {
          for await (const pair of targetDir.entries()) {
            if (Array.isArray(pair) && typeof pair[0] === 'string') {
              known.add(pair[0]);
            }
          }
        }
      } catch {
        if (this._opfsDirIndexPromise === promise) {
          this._opfsDirIndexPromise = null;
        }
        return null;
      }
      if (this._opfsDirIndexedStore === targetDir && this._opfsDirIndexPromise === promise) {
        for (const fileName of this._opfsEntries.keys()) {
          known.add(fileName);
        }
        this._opfsDirKnownFiles = known;
        this._opfsDirIndexPromise = null;
        return known;
      }
      return this._opfsDirKnownFiles;
    })();
    this._opfsDirIndexPromise = promise;
    return promise;
  }

  setOpfsDir(opfsDir) {
    const nextDir = opfsDir || null;
    if (nextDir !== this.opfsDir) {
      this._opfsEntries.clear();
      this._opfsMruFileName = null;
      this._keyToOpfsFile.clear();
      this._opfsWriteQueue.clear();
      this._opfsQueuedBytes = 0;
      this._resetOpfsDirIndex();
    }
    this.opfsDir = nextDir;
  }

  setOpfsCache(opfsCache) {
    const nextCache = opfsCache || null;
    if (nextCache !== this.opfsCache) {
      this._opfsWriteQueue.clear();
      this._opfsQueuedBytes = 0;
    }
    this.opfsCache = nextCache;
  }

  setFileFingerprint(relPath, fingerprintOrMeta) {
    const cleanPath = normalizeRelPath(relPath);
    const nextFp = buildFileFingerprint(fingerprintOrMeta);
    const prevFp = this._fileFingerprints.get(cleanPath) || null;
    if (nextFp === null) {
      this._fileFingerprints.delete(cleanPath);
    } else {
      this._fileFingerprints.set(cleanPath, nextFp);
    }
    if (prevFp !== null && prevFp !== nextFp) {
      this._evictPathFromMemory(cleanPath);
    }
    return nextFp;
  }

  getFileFingerprint(relPath) {
    let cleanPath;
    try {
      cleanPath = normalizeRelPath(relPath);
    } catch {
      return null;
    }
    return this._fileFingerprints.get(cleanPath) || null;
  }

  clearFileFingerprint(relPath) {
    let cleanPath;
    try {
      cleanPath = normalizeRelPath(relPath);
    } catch {
      return false;
    }
    return this._fileFingerprints.delete(cleanPath);
  }

  _resolveEffectiveFingerprint(relPath, overrideFingerprint = undefined) {
    if (overrideFingerprint !== undefined && overrideFingerprint !== null) {
      return buildFileFingerprint(overrideFingerprint);
    }
    return this._fileFingerprints.get(relPath) || null;
  }

  _deleteMemEntry(key) {
    const bytes = this._memCache.get(key);
    if (bytes === undefined) {
      return 0;
    }
    this._memCache.delete(key);
    if (this._memMruKey === key) {
      this._memMruKey = null;
    }
    this._memChunkFingerprints.delete(key);
    this._chunkLastSwitch.delete(key);
    if (this._probationChunks.delete(key)) {
      if (this._probationMruKey === key) {
        this._probationMruKey = null;
      }
      this._probationPrefetched.delete(key);
    }
    const protBytes = this._protectedChunks.get(key);
    if (protBytes !== undefined) {
      this._protectedChunks.delete(key);
      if (this._protectedMruKey === key) {
        this._protectedMruKey = null;
      }
      this._protectedBytesCached = Math.max(0, this._protectedBytesCached - protBytes);
    }
    const sep = key.lastIndexOf('::');
    if (sep > 0) {
      const relPath = key.slice(0, sep);
      const pathKeys = this._pathChunkKeys.get(relPath);
      if (pathKeys !== undefined) {
        const idx = Number(key.slice(sep + 2));
        if (Number.isInteger(idx)) {
          pathKeys.delete(idx);
          if (pathKeys.size === 0) {
            this._pathChunkKeys.delete(relPath);
          }
        }
      }
    }
    const len = bytes && Number.isFinite(bytes.byteLength) ? bytes.byteLength : 0;
    if (len > 0) {
      this._bytesCached = Math.max(0, this._bytesCached - len);
    }
    return len;
  }

  _isSlruEnabled() {
    return this.chunkSize > 0 && Math.floor(this.maxBytes / this.chunkSize) >= SLRU_MIN_CHUNKS;
  }

  _evictPathFromMemory(relPath) {
    const prefix = `${relPath}::`;
    const pathKeys = this._pathChunkKeys.get(relPath);
    if (pathKeys !== undefined && pathKeys.size > 0) {
      for (const key of pathKeys.values()) {
        this._deleteMemEntry(key);
        this._prefetchedKeys.delete(key);
        this._recordReadaheadWastedKey(key);
        const queued = this._opfsWriteQueue.get(key);
        if (queued) {
          this._opfsWriteQueue.delete(key);
          if (queued.bytes) {
            this._opfsQueuedBytes = Math.max(0, this._opfsQueuedBytes - queued.bytes.byteLength);
          }
        }
      }
    }
    // Sweep any synthetic/direct _memCache keys for this path without allocating Array.from().
    for (const key of this._memCache.keys()) {
      if (key.startsWith(prefix)) {
        this._deleteMemEntry(key);
        this._prefetchedKeys.delete(key);
        this._recordReadaheadWastedKey(key);
        const queued = this._opfsWriteQueue.get(key);
        if (queued) {
          this._opfsWriteQueue.delete(key);
          if (queued.bytes) {
            this._opfsQueuedBytes = Math.max(0, this._opfsQueuedBytes - queued.bytes.byteLength);
          }
        }
      }
    }
    for (const key of this._expectedChunkLengths.keys()) {
      if (key.startsWith(prefix)) {
        this._expectedChunkLengths.delete(key);
      }
    }
    this._fileSizes.delete(relPath);
  }

  _recordObservedHttpMeta(relPath, totalSize, etag) {
    const validSize = Number.isSafeInteger(totalSize) && totalSize >= 0 ? totalSize : null;
    const validTag = typeof etag === 'string' && etag.trim().length > 0 ? etag.trim() : null;
    if (validSize === null && validTag === null) {
      return;
    }
    const prev = this._observedHttpMeta.get(relPath) || null;
    if (prev) {
      const sizeChanged = validSize !== null && prev.totalSize !== null && validSize !== prev.totalSize;
      const etagChanged = validTag !== null && prev.etag !== null && validTag !== prev.etag;
      if (sizeChanged || etagChanged) {
        this._evictPathFromMemory(relPath);
        if (this._fileFingerprints.has(relPath)) {
          const updatedFp = buildFileFingerprint(
            validSize !== null ? validSize : prev.totalSize,
            validTag !== null ? validTag : prev.etag,
          );
          if (updatedFp) {
            this._fileFingerprints.set(relPath, updatedFp);
          }
        }
      }
      this._observedHttpMeta.delete(relPath);
    } else if (this._observedHttpMeta.size >= MAX_TRACKED_FILE_SIZES) {
      const oldest = this._observedHttpMeta.keys().next().value;
      if (oldest !== undefined) {
        this._observedHttpMeta.delete(oldest);
      }
    }
    this._observedHttpMeta.set(relPath, {
      totalSize: validSize !== null ? validSize : prev ? prev.totalSize : null,
      etag: validTag !== null ? validTag : prev ? prev.etag : null,
    });
  }

  _chunkKey(relPath, chunkIndex) {
    const pathKeys = this._pathChunkKeys.get(relPath);
    if (pathKeys !== undefined) {
      const cached = pathKeys.get(chunkIndex);
      if (cached !== undefined) {
        return cached;
      }
    }
    return `${relPath}::${chunkIndex}`;
  }

  _recordOpfsEntry(fileName, key, relPath, chunkIndex, byteLength) {
    if (!fileName) return;
    if (fileName === this._opfsMruFileName && this._opfsEntries.has(fileName)) {
      this._opfsEntries.set(fileName, { key, relPath, chunkIndex, byteLength });
      if (key) {
        this._keyToOpfsFile.set(key, fileName);
      }
      return;
    }
    if (this._opfsEntries.has(fileName)) {
      this._opfsEntries.delete(fileName);
    } else if (this._opfsEntries.size >= MAX_TRACKED_OPFS_ENTRIES) {
      const oldest = this._opfsEntries.keys().next().value;
      if (oldest !== undefined) {
        const oldMeta = this._opfsEntries.get(oldest);
        this._opfsEntries.delete(oldest);
        if (this._opfsMruFileName === oldest) {
          this._opfsMruFileName = null;
        }
        if (oldMeta && oldMeta.key && this._keyToOpfsFile.get(oldMeta.key) === oldest) {
          this._keyToOpfsFile.delete(oldMeta.key);
        }
      }
    }
    this._opfsEntries.set(fileName, { key, relPath, chunkIndex, byteLength });
    this._opfsMruFileName = fileName;
    if (key) {
      this._keyToOpfsFile.set(key, fileName);
    }
  }

  _touchOpfsEntryByKey(key) {
    if (this._keyToOpfsFile.size === 0) return;
    const fileName = this._keyToOpfsFile.get(key);
    if (fileName === undefined || fileName === this._opfsMruFileName) return;
    const meta = this._opfsEntries.get(fileName);
    if (meta === undefined) return;
    this._opfsEntries.delete(fileName);
    this._opfsEntries.set(fileName, meta);
    this._opfsMruFileName = fileName;
  }

  _recordExpectedChunkLength(key, byteLength) {
    if (!Number.isInteger(byteLength) || byteLength <= 0) return;
    if (this._expectedChunkLengths.has(key)) {
      this._expectedChunkLengths.delete(key);
    } else if (this._expectedChunkLengths.size >= MAX_TRACKED_CHUNK_LENGTHS) {
      const oldest = this._expectedChunkLengths.keys().next().value;
      if (oldest !== undefined) {
        this._expectedChunkLengths.delete(oldest);
      }
    }
    this._expectedChunkLengths.set(key, byteLength);
  }

  _recordFileSize(relPath, totalSize) {
    if (!Number.isSafeInteger(totalSize) || totalSize < 0) return;
    if (this._fileSizes.has(relPath)) {
      this._fileSizes.delete(relPath);
    } else if (this._fileSizes.size >= MAX_TRACKED_FILE_SIZES) {
      const oldest = this._fileSizes.keys().next().value;
      if (oldest !== undefined) {
        this._fileSizes.delete(oldest);
      }
    }
    this._fileSizes.set(relPath, totalSize);
  }

  _getExpectedChunkLength(key, relPath, chunkIndex) {
    const expected = this._expectedChunkLengths.get(key);
    if (expected !== undefined) {
      return expected;
    }
    const total = this._fileSizes.get(relPath);
    if (total !== undefined) {
      const chunkStart = chunkIndex * this.chunkSize;
      return Math.max(0, Math.min(this.chunkSize, total - chunkStart));
    }
    return null;
  }

  _validateCachedChunk(key, relPath, chunkIndex, bytes, requiredEndInChunk) {
    if (!(bytes instanceof Uint8Array)) {
      return 'corrupt';
    }
    if (bytes.byteLength === 0 || bytes.byteLength > this.chunkSize) {
      return 'corrupt';
    }
    const expected = this._getExpectedChunkLength(key, relPath, chunkIndex);
    if (expected !== null) {
      return bytes.byteLength === expected ? 'valid' : 'corrupt';
    }
    if (
      Number.isInteger(requiredEndInChunk) &&
      requiredEndInChunk > 0 &&
      bytes.byteLength < requiredEndInChunk
    ) {
      return 'suspect';
    }
    return 'valid';
  }

  _touchMemoryHit(key, bytes) {
    if (key === this._memMruKey) {
      this._stats.mruFastPathHits += 1;
    } else {
      this._memCache.delete(key);
      this._memCache.set(key, bytes);
      this._memMruKey = key;
    }
    this._touchOpfsEntryByKey(key);

    if (this._isSlruEnabled()) {
      if (key !== this._lastAccessedChunkKey) {
        this._chunkSwitchCount += 1;
        this._lastAccessedChunkKey = key;
      }
      const protLen = this._protectedChunks.get(key);
      if (protLen !== undefined) {
        if (key !== this._protectedMruKey) {
          this._protectedChunks.delete(key);
          this._protectedChunks.set(key, protLen);
          this._protectedMruKey = key;
        }
      } else {
        const probLen = this._probationChunks.get(key);
        if (probLen !== undefined) {
          const prevSwitch = this._chunkLastSwitch.get(key);
          const reuseDistance =
            prevSwitch !== undefined ? this._chunkSwitchCount - prevSwitch : 0;
          if (reuseDistance >= 2 && !this._readaheadKeys.has(key)) {
            this._probationChunks.delete(key);
            if (this._probationMruKey === key) {
              this._probationMruKey = null;
            }
            this._probationPrefetched.delete(key);
            this._protectedChunks.set(key, probLen);
            this._protectedMruKey = key;
            this._protectedBytesCached += probLen;
            const protectedBudget = Math.min(
              Math.floor(this.maxBytes * 0.75),
              Math.max(0, this.maxBytes - this.chunkSize),
            );
            while (this._protectedBytesCached > protectedBudget && this._protectedChunks.size > 0) {
              const demoteKey = this._protectedChunks.keys().next().value;
              const demoteLen = this._protectedChunks.get(demoteKey);
              this._protectedChunks.delete(demoteKey);
              if (this._protectedMruKey === demoteKey) {
                this._protectedMruKey = null;
              }
              this._protectedBytesCached = Math.max(0, this._protectedBytesCached - demoteLen);
              this._probationChunks.set(demoteKey, demoteLen);
              this._probationMruKey = demoteKey;
              if (this._readaheadKeys.has(demoteKey) || this._prefetchedKeys.has(demoteKey)) {
                this._probationPrefetched.add(demoteKey);
              }
              this._chunkLastSwitch.set(demoteKey, this._chunkSwitchCount);
            }
          } else if (key !== this._probationMruKey) {
            this._probationChunks.delete(key);
            this._probationChunks.set(key, probLen);
            this._probationMruKey = key;
            if (this._probationPrefetched.delete(key)) {
              this._probationPrefetched.add(key);
            }
          }
        }
      }
      this._chunkLastSwitch.set(key, this._chunkSwitchCount);
    }
    return bytes;
  }

  _touchMemory(key) {
    const bytes = this._memCache.get(key);
    if (bytes === undefined) return null;
    return this._touchMemoryHit(key, bytes);
  }

  _evictOneMemVictim(preferPrefetched = false) {
    let victimKey;
    if (this._isSlruEnabled()) {
      if (preferPrefetched && this._readaheadKeys.size > 0 && this._probationPrefetched.size > 0) {
        victimKey = this._probationPrefetched.values().next().value;
      }
      if (victimKey === undefined && this._probationChunks.size > 0) {
        victimKey = this._probationChunks.keys().next().value;
      }
      if (victimKey === undefined && this._protectedChunks.size > 0) {
        victimKey = this._protectedChunks.keys().next().value;
      }
    }
    if (victimKey === undefined && this._memCache.size > 0) {
      victimKey = this._memCache.keys().next().value;
    }
    if (victimKey === undefined) {
      return 0;
    }
    const freed = this._deleteMemEntry(victimKey);
    this._prefetchedKeys.delete(victimKey);
    this._recordReadaheadWastedKey(victimKey);
    this._stats.evictions += 1;
    return freed;
  }

  _putMemory(key, bytes, fileFingerprint = null) {
    if (!(bytes instanceof Uint8Array)) return;
    // Always store a tightly-sized owned Uint8Array so evicting one chunk never
    // leaves a larger backing ArrayBuffer pinned in memory (#1077).
    const owned =
      bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes
        : bytes.slice();

    if (this._memCache.has(key)) {
      this._deleteMemEntry(key);
    }

    if (this.maxBytes <= 0 || owned.byteLength > this.maxBytes) {
      return;
    }

    while (this._bytesCached + owned.byteLength > this.maxBytes && this._memCache.size > 0) {
      if (this._evictOneMemVictim(false) === 0 && this._memCache.size === 0) {
        break;
      }
    }

    this._memCache.set(key, owned);
    this._memMruKey = key;
    const sep = key.lastIndexOf('::');
    if (sep > 0) {
      const idx = Number(key.slice(sep + 2));
      if (Number.isInteger(idx) && idx >= 0) {
        const relPath = key.slice(0, sep);
        let pathKeys = this._pathChunkKeys.get(relPath);
        if (pathKeys === undefined) {
          pathKeys = new Map();
          this._pathChunkKeys.set(relPath, pathKeys);
        }
        pathKeys.set(idx, key);
      }
    }
    if (this._isSlruEnabled()) {
      if (key !== this._lastAccessedChunkKey) {
        this._chunkSwitchCount += 1;
        this._lastAccessedChunkKey = key;
      }
      this._probationChunks.set(key, owned.byteLength);
      this._probationMruKey = key;
      this._chunkLastSwitch.set(key, this._chunkSwitchCount);
    }
    if (fileFingerprint !== null && fileFingerprint !== undefined) {
      this._memChunkFingerprints.set(key, fileFingerprint);
    } else {
      this._memChunkFingerprints.delete(key);
    }
    this._bytesCached += owned.byteLength;
  }

  trimToBytes(targetBytes = 0, options = {}) {
    const budget =
      Number.isFinite(targetBytes) && targetBytes >= 0 ? Math.floor(targetBytes) : 0;
    const dropQueuedOpfsWrites = options.dropQueuedOpfsWrites !== false;
    let opfsQueueDroppedBytes = 0;
    if (dropQueuedOpfsWrites && this._opfsWriteQueue.size > 0) {
      opfsQueueDroppedBytes = this._opfsQueuedBytes;
      this._opfsWriteQueue.clear();
      this._opfsQueuedBytes = 0;
    }
    const beforeBytes = this._bytesCached;
    while (this._bytesCached > budget && this._memCache.size > 0) {
      if (this._evictOneMemVictim(true) === 0 && this._memCache.size === 0) {
        break;
      }
    }
    const memReclaimed = Math.max(0, beforeBytes - this._bytesCached);
    const totalReclaimed = memReclaimed + opfsQueueDroppedBytes;
    return {
      bytesFreed: totalReclaimed,
      reclaimedBytes: totalReclaimed,
      memReclaimedBytes: memReclaimed,
      opfsQueueDroppedBytes,
      remainingBytes: this._bytesCached,
    };
  }

  reclaimMemory(options = {}) {
    const ratio =
      Number.isFinite(options.targetRatio) && options.targetRatio >= 0 && options.targetRatio <= 1
        ? options.targetRatio
        : 0.5;
    const targetBytes =
      Number.isFinite(options.targetBytes) && options.targetBytes >= 0
        ? Math.floor(options.targetBytes)
        : Math.floor(this._bytesCached * ratio);
    return this.trimToBytes(targetBytes, options);
  }

  async _deleteOpfsFile(fileName) {
    if (!this.opfsDir || !fileName) return false;
    if (this._opfsDirKnownFiles !== null && this._opfsDirIndexedStore === this.opfsDir) {
      this._opfsDirKnownFiles.delete(fileName);
    }
    try {
      if (typeof this.opfsDir.removeEntry === 'function') {
        await this.opfsDir.removeEntry(fileName);
        return true;
      }
      if (typeof this.opfsDir.delete === 'function') {
        return Boolean(await this.opfsDir.delete(fileName));
      }
      if (typeof this.opfsDir.remove === 'function') {
        await this.opfsDir.remove(fileName);
        return true;
      }
    } catch {
      // Ignore NotFoundError or unsupported delete failures.
    }
    return false;
  }

  async _listOpfsEntries() {
    if (!this.opfsDir) return [];
    const results = [];
    try {
      if (typeof this.opfsDir.entries === 'function') {
        for await (const pair of this.opfsDir.entries()) {
          if (!Array.isArray(pair) || typeof pair[0] !== 'string') continue;
          const name = pair[0];
          const val = pair[1];
          let byteLength = 0;
          if (val instanceof Uint8Array || val instanceof ArrayBuffer) {
            byteLength = val.byteLength;
          } else if (val && typeof val.getFile === 'function') {
            try {
              const file = await val.getFile();
              if (file && Number.isFinite(file.size)) {
                byteLength = file.size;
              }
            } catch {
              // Fall back to chunkSize estimate.
            }
          }
          results.push({ name, byteLength });
        }
        return results;
      }
      if (typeof this.opfsDir.keys === 'function') {
        for await (const name of this.opfsDir.keys()) {
          if (typeof name === 'string') {
            results.push({ name, byteLength: 0 });
          }
        }
      }
    } catch {
      // Ignore directory enumeration failures.
    }
    return results;
  }

  async _listOpfsEntryNames() {
    if (!this.opfsDir) return [];
    const names = [];
    try {
      if (typeof this.opfsDir.keys === 'function') {
        for await (const name of this.opfsDir.keys()) {
          if (typeof name === 'string') {
            names.push(name);
          }
        }
        return names;
      }
      if (typeof this.opfsDir.entries === 'function') {
        for await (const pair of this.opfsDir.entries()) {
          if (Array.isArray(pair) && typeof pair[0] === 'string') {
            names.push(pair[0]);
          }
        }
        return names;
      }
    } catch {
      // Ignore directory enumeration failures.
    }
    return names;
  }

  async _invalidateStaleOpfsChunkEntries(relPath, chunkIndex, activeFileName) {
    if (!this.opfsDir) return 0;
    const prefix = opfsChunkFilePrefix(relPath, chunkIndex, this.chunkSize);
    const legacyName = `${prefix}.bin`;
    const fpPrefix = `${prefix}_fp_`;
    const isStaleMatch = (name) =>
      typeof name === 'string' &&
      name !== activeFileName &&
      (name === legacyName || (name.startsWith(fpPrefix) && name.endsWith('.bin')));

    const staleNames = new Set();
    for (const fileName of this._opfsEntries.keys()) {
      if (isStaleMatch(fileName)) {
        staleNames.add(fileName);
      }
    }

    let usedIndex = false;
    if (this._supportsOpfsDirIndex(this.opfsDir)) {
      const knownFiles = await this._ensureOpfsDirIndex(this.opfsDir);
      if (knownFiles !== null) {
        usedIndex = true;
        for (const fileName of knownFiles) {
          if (isStaleMatch(fileName)) {
            staleNames.add(fileName);
          }
        }
      }
    }

    if (!usedIndex) {
      const dirNames = await this._listOpfsEntryNames();
      for (let i = 0; i < dirNames.length; i += 1) {
        if (isStaleMatch(dirNames[i])) {
          staleNames.add(dirNames[i]);
        }
      }

      // Fallback for minimal KV stores lacking keys()/entries(): probe legacy un-fingerprinted key.
      if (
        staleNames.size === 0 &&
        dirNames.length === 0 &&
        legacyName !== activeFileName &&
        typeof this.opfsDir.get === 'function'
      ) {
        try {
          const legacyVal = await this.opfsDir.get(legacyName);
          if (legacyVal !== undefined && legacyVal !== null) {
            staleNames.add(legacyName);
          }
        } catch {
          // Ignore probe failures.
        }
      }
    }

    let invalidated = 0;
    const key = this._chunkKey(relPath, chunkIndex);
    for (const staleName of staleNames) {
      this._opfsEntries.delete(staleName);
      if (this._keyToOpfsFile.get(key) === staleName) {
        this._keyToOpfsFile.delete(key);
      }
      const deleted = await this._deleteOpfsFile(staleName);
      if (deleted) {
        invalidated += 1;
      }
    }

    if (invalidated > 0) {
      this._stats.opfsStaleInvalidations += invalidated;
    }
    return invalidated;
  }

  async _evictCorruptEntry(key, relPath, chunkIndex, countEviction = true, fileFingerprint = null) {
    this._deleteMemEntry(key);
    this._prefetchedKeys.delete(key);
    this._recordReadaheadWastedKey(key);
    const queued = this._opfsWriteQueue.get(key);
    if (queued) {
      this._opfsWriteQueue.delete(key);
      if (queued.bytes) {
        this._opfsQueuedBytes = Math.max(0, this._opfsQueuedBytes - queued.bytes.byteLength);
      }
    }
    const fp = this._resolveEffectiveFingerprint(relPath, fileFingerprint);
    if (this.opfsCache && typeof this.opfsCache.delete === 'function') {
      try {
        await this.opfsCache.delete(opfsChunkFileName(relPath, chunkIndex, this.chunkSize));
      } catch {
        // Ignore backing store delete errors.
      }
    }
    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize, fp);
    this._opfsEntries.delete(fileName);
    this._keyToOpfsFile.delete(key);
    await this._deleteOpfsFile(fileName);
    if (countEviction) {
      this._stats.corruptEvictions += 1;
    }
  }

  async _evictForQuotaSerialized(excludeKey, excludeFileName, targetBytes) {
    const run = async () => this._evictForQuota(excludeKey, excludeFileName, targetBytes);
    const next = this._quotaEvictChain.then(run, run);
    this._quotaEvictChain = next.then(
      () => {},
      () => {},
    );
    return next;
  }

  async _evictForQuota(excludeKey, excludeFileName, targetBytes) {
    let freedBytes = 0;
    let evictedEntries = 0;
    const evictedFileNames = new Set();
    if (excludeFileName) {
      evictedFileNames.add(excludeFileName);
    }

    // 1. Evict oldest tracked OPFS/backing-store entries first (LRU order).
    if (this._opfsEntries.size > 0) {
      for (const [fileName, meta] of this._opfsEntries) {
        if (freedBytes >= targetBytes) {
          break;
        }
        if (fileName === excludeFileName || meta.key === excludeKey) {
          continue;
        }
        this._opfsEntries.delete(fileName);
        if (this._opfsMruFileName === fileName) {
          this._opfsMruFileName = null;
        }
        if (meta.key && this._keyToOpfsFile.get(meta.key) === fileName) {
          this._keyToOpfsFile.delete(meta.key);
        }
        evictedFileNames.add(fileName);

        const entryBytes = Number.isFinite(meta.byteLength) ? meta.byteLength : 0;
        await this._deleteOpfsFile(fileName);
        freedBytes += entryBytes > 0 ? entryBytes : this.chunkSize;
        evictedEntries += 1;
        this._stats.quotaEvictions += 1;
      }
    }

    // 2. If target still not reached, enumerate untracked OPFS cache files (e.g. from prior sessions).
    if (freedBytes < targetBytes && this.opfsDir) {
      const existingEntries = await this._listOpfsEntries();
      for (let i = 0; i < existingEntries.length && freedBytes < targetBytes; i += 1) {
        const entry = existingEntries[i];
        if (evictedFileNames.has(entry.name)) {
          continue;
        }
        if (!/^c_\d+_\d+_.*\.bin$/.test(entry.name)) {
          continue;
        }
        evictedFileNames.add(entry.name);
        this._opfsEntries.delete(entry.name);
        await this._deleteOpfsFile(entry.name);

        freedBytes += entry.byteLength > 0 ? entry.byteLength : this.chunkSize;
        evictedEntries += 1;
        this._stats.quotaEvictions += 1;
      }
    }

    return { freedBytes, evictedEntries };
  }

  async _readOpfsDetailed(relPath, chunkIndex, fileFingerprint = null) {
    if (!this.opfsDir && !this.opfsCache) {
      return { bytes: null, corrupt: false, staleInvalidated: false, checkedOpfs: false, fileName: null };
    }
    const fp = this._resolveEffectiveFingerprint(relPath, fileFingerprint);

    if (this.opfsCache && typeof this.opfsCache.getEntry === 'function') {
      const baseKey = opfsChunkFileName(relPath, chunkIndex, this.chunkSize);
      try {
        const entry = await this.opfsCache.getEntry(baseKey, fp ? { fileFingerprint: fp } : {});
        if (!entry || entry.supported === false) {
          return { bytes: null, corrupt: false, staleInvalidated: false, checkedOpfs: false, fileName: baseKey };
        }
        if (entry.hit && entry.data instanceof Uint8Array) {
          return { bytes: entry.data, corrupt: false, staleInvalidated: false, checkedOpfs: true, fileName: baseKey };
        }
        if (entry.invalidated) {
          this._stats.opfsStaleInvalidations += 1;
          return { bytes: null, corrupt: false, staleInvalidated: true, checkedOpfs: true, fileName: baseKey };
        }
        return { bytes: null, corrupt: false, staleInvalidated: false, checkedOpfs: true, fileName: baseKey };
      } catch {
        return { bytes: null, corrupt: true, staleInvalidated: false, checkedOpfs: true, fileName: baseKey };
      }
    }

    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize, fp);
    if (typeof this.opfsDir.getFileHandle === 'function') {
      const knownFiles = this._supportsOpfsDirIndex(this.opfsDir)
        ? await this._ensureOpfsDirIndex(this.opfsDir)
        : null;
      if (knownFiles !== null && !knownFiles.has(fileName)) {
        let staleInvalidated = false;
        if (fp !== null) {
          staleInvalidated = (await this._invalidateStaleOpfsChunkEntries(relPath, chunkIndex, fileName)) > 0;
        }
        return { bytes: null, corrupt: false, staleInvalidated, checkedOpfs: true, fileName };
      }
      let handle;
      try {
        handle = await this.opfsDir.getFileHandle(fileName, { create: false });
      } catch (err) {
        if (isNotFoundError(err)) {
          if (knownFiles !== null) {
            knownFiles.delete(fileName);
          }
          let staleInvalidated = false;
          if (fp !== null) {
            staleInvalidated = (await this._invalidateStaleOpfsChunkEntries(relPath, chunkIndex, fileName)) > 0;
          }
          return { bytes: null, corrupt: false, staleInvalidated, checkedOpfs: true, fileName };
        }
        return { bytes: null, corrupt: true, staleInvalidated: false, checkedOpfs: true, fileName };
      }
      try {
        let bytes = null;
        if (typeof handle.createSyncAccessHandle === 'function') {
          try {
            const syncHandle = await handle.createSyncAccessHandle();
            try {
              const sz = typeof syncHandle.getSize === 'function' ? syncHandle.getSize() : 0;
              if (Number.isFinite(sz) && sz > 0) {
                const out = new Uint8Array(Math.floor(sz));
                let pos = 0;
                while (pos < out.byteLength) {
                  const view = pos === 0 ? out : out.subarray(pos);
                  const step = syncHandle.read(view, { at: pos });
                  if (typeof step !== 'number' || !Number.isFinite(step) || step <= 0) {
                    break;
                  }
                  pos += step;
                }
                bytes = pos === out.byteLength ? out : out.slice(0, pos);
              } else {
                bytes = new Uint8Array(0);
              }
            } finally {
              if (typeof syncHandle.close === 'function') {
                syncHandle.close();
              }
            }
          } catch {
            // Fall back cleanly to async handle.getFile() when SyncAccessHandle is locked/unsupported.
            bytes = null;
          }
        }
        if (bytes === null) {
          const file = await handle.getFile();
          const buf = await file.arrayBuffer();
          bytes = new Uint8Array(buf);
        }
        if (knownFiles !== null) {
          knownFiles.add(fileName);
        }
        return { bytes, corrupt: false, staleInvalidated: false, checkedOpfs: true, fileName };
      } catch {
        return { bytes: null, corrupt: true, staleInvalidated: false, checkedOpfs: true, fileName };
      }
    }
    if (typeof this.opfsDir.get === 'function') {
      try {
        const val = await this.opfsDir.get(fileName);
        if (val === undefined || val === null) {
          let staleInvalidated = false;
          if (fp !== null) {
            staleInvalidated = (await this._invalidateStaleOpfsChunkEntries(relPath, chunkIndex, fileName)) > 0;
          }
          return { bytes: null, corrupt: false, staleInvalidated, checkedOpfs: true, fileName };
        }
        if (val instanceof Uint8Array) {
          return { bytes: val.slice(), corrupt: false, staleInvalidated: false, checkedOpfs: true, fileName };
        }
        if (val instanceof ArrayBuffer) {
          return { bytes: new Uint8Array(val.slice(0)), corrupt: false, staleInvalidated: false, checkedOpfs: true, fileName };
        }
        return { bytes: null, corrupt: true, staleInvalidated: false, checkedOpfs: true, fileName };
      } catch (err) {
        if (isNotFoundError(err)) {
          let staleInvalidated = false;
          if (fp !== null) {
            staleInvalidated = (await this._invalidateStaleOpfsChunkEntries(relPath, chunkIndex, fileName)) > 0;
          }
          return { bytes: null, corrupt: false, staleInvalidated, checkedOpfs: true, fileName };
        }
        return { bytes: null, corrupt: true, staleInvalidated: false, checkedOpfs: true, fileName };
      }
    }
    return { bytes: null, corrupt: false, staleInvalidated: false, checkedOpfs: false, fileName };
  }

  async _readOpfs(relPath, chunkIndex, fileFingerprint = null) {
    const res = await this._readOpfsDetailed(relPath, chunkIndex, fileFingerprint);
    return res.bytes;
  }

  async _writeOpfsOnce(fileName, bytes) {
    if (typeof this.opfsDir.getFileHandle === 'function') {
      const handle = await this.opfsDir.getFileHandle(fileName, { create: true });
      if (typeof handle.createSyncAccessHandle === 'function') {
        try {
          const syncHandle = await handle.createSyncAccessHandle();
          try {
            if (typeof syncHandle.truncate === 'function') {
              syncHandle.truncate(0);
            }
            const written = syncHandle.write(bytes, { at: 0 });
            if (typeof written === 'number' && written < bytes.byteLength) {
              let pos = written;
              while (pos < bytes.byteLength) {
                const step = syncHandle.write(bytes.subarray(pos), { at: pos });
                if (typeof step !== 'number' || step <= 0) {
                  throw new Error(`Short OPFS sync write (${pos}/${bytes.byteLength})`);
                }
                pos += step;
              }
            }
            if (typeof syncHandle.flush === 'function') {
              syncHandle.flush();
            }
          } finally {
            if (typeof syncHandle.close === 'function') {
              syncHandle.close();
            }
          }
          if (this._opfsDirKnownFiles !== null && this._opfsDirIndexedStore === this.opfsDir) {
            this._opfsDirKnownFiles.add(fileName);
          }
          return true;
        } catch (err) {
          if (isQuotaExceededError(err) || typeof handle.createWritable !== 'function') {
            throw err;
          }
          // Fall back cleanly to async FileSystemWritableFileStream if SyncAccessHandle fails.
        }
      }
      if (typeof handle.createWritable === 'function') {
        const writable = await handle.createWritable();
        try {
          await writable.write(bytes);
          await writable.close();
        } catch (err) {
          if (typeof writable.abort === 'function') {
            try {
              await writable.abort();
            } catch {
              // Ignore abort failures during error cleanup.
            }
          }
          throw err;
        }
        if (this._opfsDirKnownFiles !== null && this._opfsDirIndexedStore === this.opfsDir) {
          this._opfsDirKnownFiles.add(fileName);
        }
        return true;
      }
      return false;
    }
    if (typeof this.opfsDir.set === 'function') {
      await this.opfsDir.set(fileName, bytes.slice());
      return true;
    }
    return false;
  }

  async _writeOpfs(relPath, chunkIndex, bytes, fileFingerprint = null) {
    if ((!this.opfsDir && !this.opfsCache) || !(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
      return false;
    }
    const key = this._chunkKey(relPath, chunkIndex);
    const fp = this._resolveEffectiveFingerprint(relPath, fileFingerprint);

    if (this.opfsCache && typeof this.opfsCache.put === 'function') {
      const baseKey = opfsChunkFileName(relPath, chunkIndex, this.chunkSize);
      try {
        const res = await this.opfsCache.put(baseKey, bytes, fp ? { fileFingerprint: fp } : {});
        if (res && res.stored) {
          this._stats.opfsWrites += 1;
          return true;
        }
        this._stats.opfsWriteErrors += 1;
        return false;
      } catch {
        this._stats.opfsWriteErrors += 1;
        return false;
      }
    }

    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize, fp);

    try {
      const wrote = await this._writeOpfsOnce(fileName, bytes);
      if (wrote) {
        this._recordOpfsEntry(fileName, key, relPath, chunkIndex, bytes.byteLength);
        this._stats.opfsWrites += 1;
      } else {
        this._stats.opfsWriteErrors += 1;
      }
      return wrote;
    } catch (err) {
      // Remove any partial/0-byte file left behind by the failed write.
      this._opfsEntries.delete(fileName);
      if (this._keyToOpfsFile.get(key) === fileName) {
        this._keyToOpfsFile.delete(key);
      }
      await this._deleteOpfsFile(fileName);

      if (!isQuotaExceededError(err)) {
        // Ignore non-quota OPFS write failures and continue memory-only.
        this._stats.opfsWriteErrors += 1;
        return false;
      }

      // QuotaExceededError: evict oldest entries until freeing at least
      // max(entryBytes, ceil(maxBytes * 0.25)) (or all evictable entries), then retry ONCE.
      const targetBytes = Math.max(bytes.byteLength, Math.ceil(this.maxBytes * 0.25));
      await this._evictForQuotaSerialized(key, fileName, targetBytes);

      try {
        const retryWrote = await this._writeOpfsOnce(fileName, bytes);
        if (retryWrote) {
          this._recordOpfsEntry(fileName, key, relPath, chunkIndex, bytes.byteLength);
          this._stats.quotaRetrySuccesses += 1;
          this._stats.opfsWrites += 1;
        } else {
          this._stats.opfsWriteErrors += 1;
        }
        return retryWrote;
      } catch {
        // Quota remains exhausted: clean up any partial file and degrade gracefully.
        this._opfsEntries.delete(fileName);
        if (this._keyToOpfsFile.get(key) === fileName) {
          this._keyToOpfsFile.delete(key);
        }
        await this._deleteOpfsFile(fileName);
        this._stats.opfsWriteErrors += 1;
        return false;
      }
    }
  }

  _enqueueOpfsWrite(relPath, chunkIndex, bytes, fileFingerprint = null) {
    if ((!this.opfsDir && !this.opfsCache) || !(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
      return;
    }
    if (bytes.byteLength > this.maxPendingOpfsWriteBytes) {
      return;
    }
    const key = this._chunkKey(relPath, chunkIndex);
    const fp = this._resolveEffectiveFingerprint(relPath, fileFingerprint);
    if (this._opfsWriteQueue.has(key)) {
      const prev = this._opfsWriteQueue.get(key);
      if (prev && prev.bytes) {
        this._opfsQueuedBytes = Math.max(0, this._opfsQueuedBytes - prev.bytes.byteLength);
      }
      this._opfsWriteQueue.delete(key);
    }
    while (
      this._opfsWriteQueue.size > 0 &&
      (this._opfsWriteQueue.size >= this.maxPendingOpfsWrites ||
        this._opfsQueuedBytes + bytes.byteLength > this.maxPendingOpfsWriteBytes)
    ) {
      const oldestKey = this._opfsWriteQueue.keys().next().value;
      if (oldestKey === undefined) break;
      const oldestTask = this._opfsWriteQueue.get(oldestKey);
      this._opfsWriteQueue.delete(oldestKey);
      if (oldestTask && oldestTask.bytes) {
        this._opfsQueuedBytes = Math.max(0, this._opfsQueuedBytes - oldestTask.bytes.byteLength);
      }
    }
    this._opfsWriteQueue.set(key, { relPath, chunkIndex, bytes, fileFingerprint: fp });
    this._opfsQueuedBytes += bytes.byteLength;
    this._pumpOpfsWriteQueue();
  }

  _pumpOpfsWriteQueue() {
    while (
      this._activeOpfsWrites.size < this.maxOpfsWriteConcurrency &&
      this._opfsWriteQueue.size > 0
    ) {
      const nextEntry = this._opfsWriteQueue.entries().next().value;
      if (!nextEntry) break;
      const [key, task] = nextEntry;
      this._opfsWriteQueue.delete(key);
      if (task && task.bytes) {
        this._opfsQueuedBytes = Math.max(0, this._opfsQueuedBytes - task.bytes.byteLength);
      }

      const writePromise = (async () => {
        try {
          await this._writeOpfs(task.relPath, task.chunkIndex, task.bytes, task.fileFingerprint);
        } catch {
          this._stats.opfsWriteErrors += 1;
        }
      })();

      this._activeOpfsWrites.add(writePromise);
      writePromise.finally(() => {
        this._activeOpfsWrites.delete(writePromise);
        this._pumpOpfsWriteQueue();
        if (
          this._activeOpfsWrites.size === 0 &&
          this._opfsWriteQueue.size === 0 &&
          this._opfsWriteWaiters.length > 0
        ) {
          const waiters = this._opfsWriteWaiters.splice(0, this._opfsWriteWaiters.length);
          for (let i = 0; i < waiters.length; i += 1) {
            waiters[i]();
          }
        }
      });
    }
  }

  async flushOpfsWrites() {
    while (this._activeOpfsWrites.size > 0 || this._opfsWriteQueue.size > 0) {
      this._pumpOpfsWriteQueue();
      await new Promise((resolve) => {
        if (this._activeOpfsWrites.size === 0 && this._opfsWriteQueue.size === 0) {
          resolve();
          return;
        }
        this._opfsWriteWaiters.push(resolve);
      });
    }
    if (this.opfsCache && typeof this.opfsCache.flushBackgroundWrites === 'function') {
      await this.opfsCache.flushBackgroundWrites();
    }
  }

  async _fetchChunkFromNetwork(relPath, chunkIndex) {
    if (typeof this.fetchImpl !== 'function') {
      throw new Error('RangeCache has no fetch implementation available');
    }
    const chunkStart = chunkIndex * this.chunkSize;
    const chunkEnd = chunkStart + this.chunkSize - 1;
    const url = `${this.dataBase}${relPath}`;

    const res = await this.fetchImpl(url, {
      headers: { Range: `bytes=${chunkStart}-${chunkEnd}` },
    });

    if (res.status === 416) {
      const contentRangeHeader =
        res.headers && typeof res.headers.get === 'function'
          ? res.headers.get('Content-Range') || res.headers.get('content-range')
          : null;
      const total = parseUnsatisfiedContentRangeTotal(contentRangeHeader);
      if (total !== null) {
        this._recordFileSize(relPath, total);
      }
      return new Uint8Array(0);
    }

    if (!res.ok && res.status !== 206) {
      throw new Error(`HTTP ${res.status} reading ${relPath}[${chunkStart}..${chunkEnd}]`);
    }

    const raw = new Uint8Array(await res.arrayBuffer());
    const etagHeader = readResponseHeader(res.headers, 'ETag');

    if (res.status === 206) {
      const contentRangeHeader = readResponseHeader(res.headers, 'Content-Range');
      const parsed = parseContentRange(contentRangeHeader);
      if (parsed) {
        if (Number.isSafeInteger(parsed.total) && parsed.total >= 0) {
          this._recordObservedHttpMeta(relPath, parsed.total, etagHeader);
          this._recordFileSize(relPath, parsed.total);
        } else if (etagHeader) {
          this._recordObservedHttpMeta(relPath, null, etagHeader);
        }
        const expectedLen = parsed.end - parsed.start + 1;
        const relOffset = chunkStart >= parsed.start ? chunkStart - parsed.start : 0;
        const available = Math.min(raw.byteLength, expectedLen);
        if (relOffset >= available) {
          return new Uint8Array(0);
        }
        const sliceLen = Math.min(this.chunkSize, available - relOffset);
        return raw.slice(relOffset, relOffset + sliceLen);
      }
      if (etagHeader) {
        this._recordObservedHttpMeta(relPath, null, etagHeader);
      }
      if (raw.byteLength <= this.chunkSize) {
        return raw;
      }
      return raw.slice(0, this.chunkSize);
    }

    // 200 OK fallback (server returned full resource instead of 206 Partial Content).
    this._recordObservedHttpMeta(relPath, raw.byteLength, etagHeader);
    this._recordFileSize(relPath, raw.byteLength);
    if (chunkStart >= raw.byteLength) {
      return new Uint8Array(0);
    }
    return raw.slice(chunkStart, Math.min(raw.byteLength, chunkStart + this.chunkSize));
  }

  async _fetchChunkRunFromNetwork(relPath, startChunk, endChunk) {
    if (typeof this.fetchImpl !== 'function') {
      throw new Error('RangeCache has no fetch implementation available');
    }

    const resultMap = new Map();
    const uncovered = [];
    const runStart = startChunk * this.chunkSize;
    const knownTotalBefore = this._fileSizes.has(relPath) ? this._fileSizes.get(relPath) : null;

    if (knownTotalBefore !== null && runStart >= knownTotalBefore) {
      for (let idx = startChunk; idx <= endChunk; idx += 1) {
        resultMap.set(idx, new Uint8Array(0));
      }
      return { chunks: resultMap, uncovered };
    }

    const naturalEnd = (endChunk + 1) * this.chunkSize - 1;
    const reqEnd =
      knownTotalBefore !== null && knownTotalBefore > 0
        ? Math.min(naturalEnd, knownTotalBefore - 1)
        : naturalEnd;
    const url = `${this.dataBase}${relPath}`;

    const res = await this.fetchImpl(url, {
      headers: { Range: `bytes=${runStart}-${reqEnd}` },
    });

    if (res.status === 416) {
      const contentRangeHeader =
        res.headers && typeof res.headers.get === 'function'
          ? res.headers.get('Content-Range') || res.headers.get('content-range')
          : null;
      const total = parseUnsatisfiedContentRangeTotal(contentRangeHeader);
      if (total !== null) {
        this._recordFileSize(relPath, total);
      }
      for (let idx = startChunk; idx <= endChunk; idx += 1) {
        resultMap.set(idx, new Uint8Array(0));
      }
      return { chunks: resultMap, uncovered };
    }

    if (!res.ok && res.status !== 206) {
      throw new Error(`HTTP ${res.status} reading ${relPath}[${runStart}..${reqEnd}]`);
    }

    const raw = new Uint8Array(await res.arrayBuffer());
    const etagHeader = readResponseHeader(res.headers, 'ETag');

    let actualStart = runStart;
    let actualEnd = raw.byteLength > 0 ? runStart + raw.byteLength - 1 : runStart - 1;
    let knownTotal = knownTotalBefore;
    let reachedEof = false;

    if (res.status === 206) {
      const contentRangeHeader = readResponseHeader(res.headers, 'Content-Range');
      const parsed = parseContentRange(contentRangeHeader);
      if (parsed) {
        actualStart = parsed.start;
        const headerSpan = parsed.end - parsed.start + 1;
        const usableLen = Math.min(raw.byteLength, headerSpan);
        actualEnd = usableLen > 0 ? actualStart + usableLen - 1 : actualStart - 1;
        if (Number.isSafeInteger(parsed.total) && parsed.total >= 0) {
          this._recordObservedHttpMeta(relPath, parsed.total, etagHeader);
          this._recordFileSize(relPath, parsed.total);
          knownTotal = parsed.total;
          if (actualEnd + 1 >= parsed.total) {
            reachedEof = true;
          }
        } else if (etagHeader) {
          this._recordObservedHttpMeta(relPath, null, etagHeader);
        }
      } else if (etagHeader) {
        this._recordObservedHttpMeta(relPath, null, etagHeader);
      }
    } else {
      // 200 OK fallback (server returned full resource instead of 206 Partial Content).
      this._recordObservedHttpMeta(relPath, raw.byteLength, etagHeader);
      this._recordFileSize(relPath, raw.byteLength);
      knownTotal = raw.byteLength;
      actualStart = 0;
      actualEnd = raw.byteLength - 1;
      reachedEof = true;
    }

    for (let idx = startChunk; idx <= endChunk; idx += 1) {
      const cStart = idx * this.chunkSize;
      const cMaxEnd = cStart + this.chunkSize - 1;

      if (knownTotal !== null && cStart >= knownTotal) {
        resultMap.set(idx, new Uint8Array(0));
        continue;
      }

      const expectedEnd = knownTotal !== null ? Math.min(cMaxEnd, knownTotal - 1) : cMaxEnd;
      const startsInResponse =
        raw.byteLength > 0 && cStart >= actualStart && cStart <= actualEnd;

      if (startsInResponse) {
        const relStart = cStart - actualStart;
        const availFromChunkStart = actualEnd - cStart + 1;
        const neededLen = expectedEnd - cStart + 1;
        const coversCompleteChunk =
          actualEnd >= expectedEnd || (reachedEof && availFromChunkStart > 0);
        if (coversCompleteChunk) {
          const takeLen = Math.min(this.chunkSize, neededLen, availFromChunkStart);
          resultMap.set(idx, raw.slice(relStart, relStart + takeLen));
          continue;
        }
      } else if (reachedEof && cStart > actualEnd) {
        resultMap.set(idx, new Uint8Array(0));
        continue;
      }

      uncovered.push(idx);
    }

    return { chunks: resultMap, uncovered };
  }

  _consumePrefetchHit(key) {
    const wasPrefetch = this._prefetchedKeys.delete(key);
    if (wasPrefetch) {
      this._stats.prefetchHits += 1;
    }
    if (this._readaheadKeys.size > 0 && this._deleteReadaheadKeyInternal(key) !== undefined) {
      this._stats.readaheadHits += 1;
    }
    this._probationPrefetched.delete(key);
    return wasPrefetch;
  }

  _updateSequentialTracker(relPath, offset, length) {
    const prevState = this._seqTrackers.get(relPath) || null;
    const opts = this._seqDetectOpts;
    opts.chunkSize = this.chunkSize;
    opts.maxGapBytes = this.sequentialMaxGapBytes;
    opts.minReadaheadBytes = this.enableReadAhead ? this.minReadaheadBytes : 0;
    opts.maxReadaheadBytes = this.enableReadAhead ? this.maxReadaheadBytes : 0;
    opts.streakThreshold = this.sequentialStreakThreshold;
    const detection = detectSequentialRead(prevState, offset, length, opts);

    if (relPath === this._seqTrackerMruKey && this._seqTrackers.has(relPath)) {
      this._seqTrackers.set(relPath, detection.nextState);
    } else {
      if (this._seqTrackers.has(relPath)) {
        this._seqTrackers.delete(relPath);
      } else if (this._seqTrackers.size >= MAX_TRACKED_SEQUENTIAL_KEYS) {
        const oldest = this._seqTrackers.keys().next().value;
        if (oldest !== undefined) {
          this._seqTrackers.delete(oldest);
          if (this._seqTrackerMruKey === oldest) {
            this._seqTrackerMruKey = null;
          }
        }
      }
      this._seqTrackers.set(relPath, detection.nextState);
      this._seqTrackerMruKey = relPath;
    }

    if (detection.isSequential) {
      this._stats.sequentialHits += 1;
      if (detection.streak > this._stats.sequentialStreakPeak) {
        this._stats.sequentialStreakPeak = detection.streak;
      }
    } else if (detection.isRandomSeek && length > 0) {
      const firstChunk = Math.floor(offset / this.chunkSize);
      const lastChunk = Math.floor((offset + length - 1) / this.chunkSize);
      this._accountAbandonedReadaheadOnSeek(relPath, firstChunk, lastChunk);
    }

    if (this.enableReadAhead) {
      this._stats.adaptiveReadaheadWindowBytes = detection.adaptiveWindowBytes;
    }

    return detection;
  }

  getSequentialState(relPath) {
    let cleanPath;
    try {
      cleanPath = this._normalizePathOrUrl(relPath);
    } catch {
      return null;
    }
    const state = this._seqTrackers.get(cleanPath);
    return state ? { ...state } : null;
  }

  async _getChunk(relPath, chunkIndex, requiredEndInChunk = 0, options = {}) {
    const isPrefetch = Boolean(options.isPrefetch);
    const isReadahead = Boolean(options.isReadahead);
    const countMiss = options.countMiss !== false;
    const fp = this._resolveEffectiveFingerprint(relPath, options.fileFingerprint);
    const key = this._chunkKey(relPath, chunkIndex);
    let suspectByteLength = null;

    const memHit = this._touchMemory(key);
    if (memHit !== null) {
      const memFp = this._memChunkFingerprints.get(key) || null;
      if (fp !== null && memFp !== fp) {
        this._deleteMemEntry(key);
        this._prefetchedKeys.delete(key);
        this._recordReadaheadWastedKey(key);
      } else {
        const verdict = this._validateCachedChunk(key, relPath, chunkIndex, memHit, requiredEndInChunk);
        if (verdict === 'valid') {
          if (!isPrefetch) {
            this._stats.hits += 1;
            this._consumePrefetchHit(key);
          }
          return memHit;
        }
        if (verdict === 'corrupt') {
          await this._evictCorruptEntry(key, relPath, chunkIndex, true, fp);
        } else {
          suspectByteLength = memHit.byteLength;
          await this._evictCorruptEntry(key, relPath, chunkIndex, false, fp);
        }
      }
    }

    const existingFlight = this._inFlight.get(key);
    if (existingFlight) {
      const joinedPrefetch = !isPrefetch && this._inFlightPrefetchKeys.delete(key);
      const joinedReadahead = !isPrefetch && this._inFlightReadaheadKeys.delete(key);
      const inflightBytes = await existingFlight;
      if (inflightBytes instanceof Uint8Array && inflightBytes.byteLength === 0) {
        return inflightBytes;
      }
      const verdict = this._validateCachedChunk(
        key,
        relPath,
        chunkIndex,
        inflightBytes,
        requiredEndInChunk,
      );
      if (verdict === 'valid') {
        if (joinedPrefetch) {
          this._prefetchedKeys.delete(key);
          this._probationPrefetched.delete(key);
          this._stats.prefetchHits += 1;
        }
        if (joinedReadahead) {
          this._deleteReadaheadKeyInternal(key);
          this._stats.readaheadHits += 1;
        }
        return inflightBytes;
      }
      if (verdict === 'corrupt') {
        await this._evictCorruptEntry(key, relPath, chunkIndex, true, fp);
      } else {
        suspectByteLength = inflightBytes.byteLength;
        await this._evictCorruptEntry(key, relPath, chunkIndex, false, fp);
      }
    }

    if (isPrefetch) {
      this._inFlightPrefetchKeys.add(key);
      if (isReadahead) {
        this._inFlightReadaheadKeys.add(key);
      }
    }

    const loadPromise = (async () => {
      if (suspectByteLength === null) {
        const opfsResult = await this._readOpfsDetailed(relPath, chunkIndex, fp);
        if (opfsResult.corrupt) {
          if (!isPrefetch && opfsResult.checkedOpfs) {
            this._stats.opfsMisses += 1;
          }
          await this._evictCorruptEntry(key, relPath, chunkIndex, true, fp);
        } else if (opfsResult.bytes !== null) {
          const verdict = this._validateCachedChunk(
            key,
            relPath,
            chunkIndex,
            opfsResult.bytes,
            requiredEndInChunk,
          );
          if (verdict === 'valid') {
            const fileName = opfsResult.fileName || opfsChunkFileName(relPath, chunkIndex, this.chunkSize, fp);
            this._recordOpfsEntry(fileName, key, relPath, chunkIndex, opfsResult.bytes.byteLength);
            if (!isPrefetch) {
              this._stats.opfsHits += 1;
              this._stats.hits += 1;
            }
            this._putMemory(key, opfsResult.bytes, fp);
            if (isPrefetch && this._inFlightPrefetchKeys.has(key) && this._memCache.has(key)) {
              this._trackPrefetchedKey(
                key,
                relPath,
                chunkIndex,
                isReadahead && this._inFlightReadaheadKeys.has(key),
                opfsResult.bytes.byteLength,
              );
            }
            return opfsResult.bytes;
          }
          if (!isPrefetch && opfsResult.checkedOpfs) {
            this._stats.opfsMisses += 1;
          }
          if (verdict === 'corrupt') {
            await this._evictCorruptEntry(key, relPath, chunkIndex, true, fp);
          } else {
            suspectByteLength = opfsResult.bytes.byteLength;
            await this._evictCorruptEntry(key, relPath, chunkIndex, false, fp);
          }
        } else if (!isPrefetch && opfsResult.checkedOpfs) {
          this._stats.opfsMisses += 1;
        }
      }

      if (countMiss) {
        this._stats.misses += 1;
      }
      const netBytes = await this._fetchChunkFromNetwork(relPath, chunkIndex);
      if (suspectByteLength !== null && netBytes.byteLength !== suspectByteLength) {
        this._stats.corruptEvictions += 1;
      }
      if (netBytes.byteLength > 0) {
        const effectiveFp = this._resolveEffectiveFingerprint(relPath, options.fileFingerprint);
        this._recordExpectedChunkLength(key, netBytes.byteLength);
        this._putMemory(key, netBytes, effectiveFp);
        if (isPrefetch) {
          this._stats.prefetchBytes += netBytes.byteLength;
          if (this._inFlightPrefetchKeys.has(key) && this._memCache.has(key)) {
            this._trackPrefetchedKey(
              key,
              relPath,
              chunkIndex,
              isReadahead && this._inFlightReadaheadKeys.has(key),
              netBytes.byteLength,
            );
          }
        }
        if (this.backgroundOpfsWrites) {
          this._enqueueOpfsWrite(relPath, chunkIndex, netBytes, effectiveFp);
        } else {
          await this._writeOpfs(relPath, chunkIndex, netBytes, effectiveFp);
        }
      }
      return netBytes;
    })();

    this._inFlight.set(key, loadPromise);
    try {
      return await loadPromise;
    } finally {
      this._inFlightPrefetchKeys.delete(key);
      this._inFlightReadaheadKeys.delete(key);
      if (this._inFlight.get(key) === loadPromise) {
        this._inFlight.delete(key);
      }
    }
  }

  async _getChunksCoalesced(relPath, chunkSpecs, options = {}) {
    const isPrefetch = Boolean(options.isPrefetch);
    const isReadahead = Boolean(options.isReadahead);
    const countMiss = options.countMiss !== false;
    const countCoalesce = options.countCoalesce !== false;
    const fp = this._resolveEffectiveFingerprint(relPath, options.fileFingerprint);
    const maxMergedBytes =
      Number.isFinite(options.maxMergedBytes) && options.maxMergedBytes > 0
        ? Math.floor(options.maxMergedBytes)
        : this.coalesceMaxMergedBytes;

    const total = chunkSpecs.length;
    if (total === 0) {
      return [];
    }

    const results = new Array(total);
    const unresolved = [];

    // 1. Resolve memory hits and concurrent in-flight promises.
    for (let i = 0; i < total; i += 1) {
      const spec = chunkSpecs[i];
      const key = this._chunkKey(relPath, spec.chunkIndex);
      let suspectByteLength = null;

      const memHit = this._touchMemory(key);
      if (memHit !== null) {
        const memFp = this._memChunkFingerprints.get(key) || null;
        if (fp !== null && memFp !== fp) {
          this._deleteMemEntry(key);
          this._prefetchedKeys.delete(key);
          this._recordReadaheadWastedKey(key);
        } else {
          const verdict = this._validateCachedChunk(
            key,
            relPath,
            spec.chunkIndex,
            memHit,
            spec.requiredEndInChunk,
          );
          if (verdict === 'valid') {
            if (!isPrefetch) {
              this._stats.hits += 1;
              this._consumePrefetchHit(key);
            }
            results[i] = memHit;
            continue;
          }
          if (verdict === 'corrupt') {
            await this._evictCorruptEntry(key, relPath, spec.chunkIndex, true, fp);
          } else {
            suspectByteLength = memHit.byteLength;
            await this._evictCorruptEntry(key, relPath, spec.chunkIndex, false, fp);
          }
        }
      }

      const existingFlight = this._inFlight.get(key);
      if (existingFlight) {
        unresolved.push({
          pos: i,
          spec,
          key,
          suspectByteLength,
          existingFlight,
          joinedPrefetch: !isPrefetch && this._inFlightPrefetchKeys.delete(key),
          joinedReadahead: !isPrefetch && this._inFlightReadaheadKeys.delete(key),
        });
        continue;
      }

      unresolved.push({
        pos: i,
        spec,
        key,
        suspectByteLength,
        existingFlight: null,
        joinedPrefetch: false,
        joinedReadahead: false,
      });
    }

    // 2. Await in-flight joins and check OPFS for non-in-flight entries.
    const missing = [];
    await Promise.all(
      unresolved.map(async (item) => {
        if (item.existingFlight) {
          const inflightBytes = await item.existingFlight;
          if (inflightBytes instanceof Uint8Array && inflightBytes.byteLength === 0) {
            results[item.pos] = inflightBytes;
            return;
          }
          const verdict = this._validateCachedChunk(
            item.key,
            relPath,
            item.spec.chunkIndex,
            inflightBytes,
            item.spec.requiredEndInChunk,
          );
          if (verdict === 'valid') {
            if (item.joinedPrefetch) {
              this._prefetchedKeys.delete(item.key);
              this._probationPrefetched.delete(item.key);
              this._stats.prefetchHits += 1;
            }
            if (item.joinedReadahead) {
              this._deleteReadaheadKeyInternal(item.key);
              this._stats.readaheadHits += 1;
            }
            results[item.pos] = inflightBytes;
            return;
          }
          if (verdict === 'corrupt') {
            await this._evictCorruptEntry(item.key, relPath, item.spec.chunkIndex, true, fp);
          } else {
            item.suspectByteLength = inflightBytes.byteLength;
            await this._evictCorruptEntry(item.key, relPath, item.spec.chunkIndex, false, fp);
          }
        }

        if (item.suspectByteLength === null && (this.opfsDir || this.opfsCache)) {
          let lookupPromise = this._inFlightOpfsLookups.get(item.key);
          const isLeader = !lookupPromise;
          if (isLeader) {
            lookupPromise = this._readOpfsDetailed(relPath, item.spec.chunkIndex, fp);
            this._inFlightOpfsLookups.set(item.key, lookupPromise);
          }
          let opfsResult;
          try {
            opfsResult = await lookupPromise;
          } finally {
            if (isLeader && this._inFlightOpfsLookups.get(item.key) === lookupPromise) {
              this._inFlightOpfsLookups.delete(item.key);
            }
          }
          if (opfsResult.corrupt) {
            if (isLeader) {
              if (!isPrefetch && opfsResult.checkedOpfs) {
                this._stats.opfsMisses += 1;
              }
              await this._evictCorruptEntry(item.key, relPath, item.spec.chunkIndex, true, fp);
            }
          } else if (opfsResult.bytes !== null) {
            const verdict = this._validateCachedChunk(
              item.key,
              relPath,
              item.spec.chunkIndex,
              opfsResult.bytes,
              item.spec.requiredEndInChunk,
            );
            if (verdict === 'valid') {
              if (isLeader) {
                const fileName =
                  opfsResult.fileName || opfsChunkFileName(relPath, item.spec.chunkIndex, this.chunkSize, fp);
                this._recordOpfsEntry(
                  fileName,
                  item.key,
                  relPath,
                  item.spec.chunkIndex,
                  opfsResult.bytes.byteLength,
                );
                if (!isPrefetch) {
                  this._stats.opfsHits += 1;
                  this._stats.hits += 1;
                }
                this._putMemory(item.key, opfsResult.bytes, fp);
                if (isPrefetch && this._memCache.has(item.key)) {
                  this._trackPrefetchedKey(
                    item.key,
                    relPath,
                    item.spec.chunkIndex,
                    isReadahead,
                    opfsResult.bytes.byteLength,
                  );
                }
              } else if (!isPrefetch) {
                this._stats.hits += 1;
              }
              results[item.pos] = opfsResult.bytes;
              return;
            }
            if (isLeader) {
              if (!isPrefetch && opfsResult.checkedOpfs) {
                this._stats.opfsMisses += 1;
              }
              if (verdict === 'corrupt') {
                await this._evictCorruptEntry(item.key, relPath, item.spec.chunkIndex, true, fp);
              } else {
                item.suspectByteLength = opfsResult.bytes.byteLength;
                await this._evictCorruptEntry(item.key, relPath, item.spec.chunkIndex, false, fp);
              }
            } else if (verdict !== 'corrupt') {
              item.suspectByteLength = opfsResult.bytes.byteLength;
            }
          } else if (isLeader && !isPrefetch && opfsResult.checkedOpfs) {
            this._stats.opfsMisses += 1;
          }
        }

        missing.push(item);
      }),
    );

    if (missing.length === 0) {
      return results;
    }

    // Re-check in-flight / memory after async OPFS phase and sort missing ascending by chunkIndex.
    missing.sort((a, b) => a.spec.chunkIndex - b.spec.chunkIndex);
    const toFetch = [];
    const lateJoins = [];

    for (let i = 0; i < missing.length; i += 1) {
      const item = missing[i];
      const memHit = this._touchMemory(item.key);
      if (memHit !== null && item.suspectByteLength === null) {
        const memFp = this._memChunkFingerprints.get(item.key) || null;
        if (fp === null || memFp === fp) {
          const verdict = this._validateCachedChunk(
            item.key,
            relPath,
            item.spec.chunkIndex,
            memHit,
            item.spec.requiredEndInChunk,
          );
          if (verdict === 'valid') {
            if (!isPrefetch) {
              this._stats.hits += 1;
              this._consumePrefetchHit(item.key);
            }
            results[item.pos] = memHit;
            continue;
          }
        }
      }
      const lateFlight = this._inFlight.get(item.key);
      if (lateFlight) {
        const joinedPrefetch = !isPrefetch && this._inFlightPrefetchKeys.delete(item.key);
        const joinedReadahead = !isPrefetch && this._inFlightReadaheadKeys.delete(item.key);
        lateJoins.push(
          lateFlight.then((bytes) => {
            if (bytes instanceof Uint8Array && bytes.byteLength > 0) {
              if (joinedPrefetch) {
                this._prefetchedKeys.delete(item.key);
                this._probationPrefetched.delete(item.key);
                this._stats.prefetchHits += 1;
              }
              if (joinedReadahead) {
                this._deleteReadaheadKeyInternal(item.key);
                this._stats.readaheadHits += 1;
              }
            }
            results[item.pos] = bytes;
          }),
        );
        continue;
      }
      toFetch.push(item);
    }

    // 3. Group contiguous missing chunks into coalesced runs bounded by maxMergedBytes.
    const maxRunChunks = Math.max(1, Math.floor(maxMergedBytes / this.chunkSize));
    const runs = [];
    let currentRun = [];

    for (let i = 0; i < toFetch.length; i += 1) {
      const item = toFetch[i];
      if (
        currentRun.length > 0 &&
        item.spec.chunkIndex === currentRun[currentRun.length - 1].spec.chunkIndex + 1 &&
        currentRun.length < maxRunChunks
      ) {
        currentRun.push(item);
      } else {
        if (currentRun.length > 0) {
          runs.push(currentRun);
        }
        currentRun = [item];
      }
    }
    if (currentRun.length > 0) {
      runs.push(currentRun);
    }

    const storeFetchedChunk = async (item, bytes) => {
      if (item.suspectByteLength !== null && bytes.byteLength !== item.suspectByteLength) {
        this._stats.corruptEvictions += 1;
      }
      if (bytes.byteLength > 0) {
        const effectiveFp = this._resolveEffectiveFingerprint(relPath, options.fileFingerprint);
        this._recordExpectedChunkLength(item.key, bytes.byteLength);
        this._putMemory(item.key, bytes, effectiveFp);
        if (isPrefetch) {
          this._stats.prefetchBytes += bytes.byteLength;
          if (this._inFlightPrefetchKeys.has(item.key) && this._memCache.has(item.key)) {
            this._trackPrefetchedKey(
              item.key,
              relPath,
              item.spec.chunkIndex,
              isReadahead && this._inFlightReadaheadKeys.has(item.key),
              bytes.byteLength,
            );
          }
        }
        if (this.backgroundOpfsWrites) {
          this._enqueueOpfsWrite(relPath, item.spec.chunkIndex, bytes, effectiveFp);
        } else {
          await this._writeOpfs(relPath, item.spec.chunkIndex, bytes, effectiveFp);
        }
      }
    };

    const runPromises = runs.map((run) => {
      const deferreds = run.map((item) => {
        let resolve;
        let reject;
        const promise = new Promise((res, rej) => {
          resolve = res;
          reject = rej;
        });
        this._inFlight.set(item.key, promise);
        if (isPrefetch) {
          this._inFlightPrefetchKeys.add(item.key);
          if (isReadahead) {
            this._inFlightReadaheadKeys.add(item.key);
          }
        }
        return { item, promise, resolve, reject };
      });

      const executeRun = async () => {
        try {
          if (run.length === 1) {
            const d = deferreds[0];
            if (countMiss) {
              this._stats.misses += 1;
            }
            const netBytes = await this._fetchChunkFromNetwork(relPath, d.item.spec.chunkIndex);
            await storeFetchedChunk(d.item, netBytes);
            results[d.item.pos] = netBytes;
            d.resolve(netBytes);
            return;
          }

          if (countCoalesce) {
            this._stats.coalescedRangeCount += 1;
            this._stats.bytesSavedByCoalescing +=
              (run.length - 1) * ESTIMATED_HTTP_RANGE_OVERHEAD_BYTES;
          }
          if (countMiss) {
            this._stats.misses += run.length;
          }

          const startChunk = run[0].spec.chunkIndex;
          const endChunk = run[run.length - 1].spec.chunkIndex;
          const { chunks: fetchedMap, uncovered } = await this._fetchChunkRunFromNetwork(
            relPath,
            startChunk,
            endChunk,
          );
          const uncoveredSet = new Set(uncovered);

          for (let k = 0; k < deferreds.length; k += 1) {
            const d = deferreds[k];
            const idx = d.item.spec.chunkIndex;
            if (uncoveredSet.has(idx)) {
              try {
                const fallbackBytes = await this._fetchChunkFromNetwork(relPath, idx);
                await storeFetchedChunk(d.item, fallbackBytes);
                results[d.item.pos] = fallbackBytes;
                d.resolve(fallbackBytes);
              } catch (err) {
                d.reject(err);
              }
              continue;
            }
            const bytes = fetchedMap.get(idx) || new Uint8Array(0);
            await storeFetchedChunk(d.item, bytes);
            results[d.item.pos] = bytes;
            d.resolve(bytes);
          }
        } catch (err) {
          for (let k = 0; k < deferreds.length; k += 1) {
            deferreds[k].reject(err);
          }
        } finally {
          for (let k = 0; k < deferreds.length; k += 1) {
            const d = deferreds[k];
            this._inFlightPrefetchKeys.delete(d.item.key);
            this._inFlightReadaheadKeys.delete(d.item.key);
            if (this._inFlight.get(d.item.key) === d.promise) {
              this._inFlight.delete(d.item.key);
            }
          }
        }
      };

      executeRun();
      return Promise.all(deferreds.map((d) => d.promise));
    });

    await Promise.all([...lateJoins, ...runPromises]);
    return results;
  }

  _scheduleReadAhead(relPath, lastChunk, detection) {
    if (
      !this.enableReadAhead ||
      !detection ||
      detection.readAheadBytes <= 0 ||
      this.maxBytes < this.chunkSize * 2 ||
      this._activeReadAheadBatches >= this.maxPrefetchConcurrency ||
      (typeof this.fetchImpl !== 'function' && !this.opfsDir && !this.opfsCache)
    ) {
      return;
    }

    const safeBudget = Math.min(
      detection.readAheadBytes,
      this.maxReadaheadBytes,
      Math.max(0, this.maxBytes - this.chunkSize),
    );
    if (safeBudget < this.chunkSize) {
      return;
    }

    const aheadCount = Math.max(1, Math.floor(safeBudget / this.chunkSize));
    const aheadFirstChunk = lastChunk + 1;
    let aheadLastChunk = aheadFirstChunk + aheadCount - 1;

    if (this._fileSizes.has(relPath)) {
      const knownTotal = this._fileSizes.get(relPath);
      if (aheadFirstChunk * this.chunkSize >= knownTotal) {
        return;
      }
      const maxChunk = Math.floor((knownTotal - 1) / this.chunkSize);
      aheadLastChunk = Math.min(aheadLastChunk, maxChunk);
      if (aheadLastChunk < aheadFirstChunk) {
        return;
      }
    }

    // Watermark hysteresis: avoid issuing tiny 1-chunk sliding-edge fetches while at least
    // half of the adaptive read-ahead window is already cached or in-flight immediately ahead.
    let contiguousReady = 0;
    for (let idx = aheadFirstChunk; idx <= aheadLastChunk; idx += 1) {
      const key = this._chunkKey(relPath, idx);
      if (this._memCache.has(key) || this._inFlight.has(key)) {
        contiguousReady += 1;
      } else {
        break;
      }
    }

    const windowChunks = aheadLastChunk - aheadFirstChunk + 1;
    if (contiguousReady >= windowChunks) {
      return;
    }
    const refillWatermark = Math.ceil(aheadCount / 2);
    if (contiguousReady >= refillWatermark) {
      return;
    }

    const missingSpecs = [];
    for (let idx = aheadFirstChunk + contiguousReady; idx <= aheadLastChunk; idx += 1) {
      const key = this._chunkKey(relPath, idx);
      if (this._memCache.has(key) || this._inFlight.has(key)) {
        continue;
      }
      missingSpecs.push({ chunkIndex: idx, requiredEndInChunk: 0 });
    }

    if (missingSpecs.length === 0) {
      return;
    }

    const readAheadMaxMergedBytes = this._explicitCoalesceMaxMergedBytes
      ? this.coalesceMaxMergedBytes
      : Math.max(this.coalesceMaxMergedBytes, Math.min(safeBudget, this.maxReadaheadBytes));

    this._activeReadAheadBatches += 1;
    const prefetchPromise = (async () => {
      try {
        if (this.coalesceAdjacent || missingSpecs.length > 1) {
          await this._getChunksCoalesced(relPath, missingSpecs, {
            isPrefetch: true,
            isReadahead: true,
            countMiss: false,
            countCoalesce: true,
            maxMergedBytes: readAheadMaxMergedBytes,
          });
        } else {
          await this._getChunk(relPath, missingSpecs[0].chunkIndex, 0, {
            isPrefetch: true,
            isReadahead: true,
            countMiss: false,
          });
        }
      } catch {
        // Speculative read-ahead errors must never disrupt foreground reads.
      } finally {
        this._activeReadAheadBatches = Math.max(0, this._activeReadAheadBatches - 1);
      }
    })();

    this._activePrefetches.add(prefetchPromise);
    prefetchPromise.finally(() => {
      this._activePrefetches.delete(prefetchPromise);
    });
  }

  async flushPrefetches() {
    while (this._activePrefetches.size > 0) {
      const pending = Array.from(this._activePrefetches);
      await Promise.allSettled(pending);
    }
  }

  /**
   * Zero-alloc synchronous L1 cache-hit fast path: copies cached bytes for [offset, offset + length)
   * directly into caller-provided `targetBytes` without intermediate buffer/array allocations.
   * Returns exact bytes written (>= 0) on full L1 memory hit, or -1 on cache miss (leaving
   * `targetBytes` unmodified so the caller can fall back to async `read()`/`readInto()`).
   */
  readCachedRangeInto(url, offset, length, targetBytes, options = {}) {
    const cleanPath = this._normalizePathOrUrl(url);
    if (!Number.isInteger(offset) || offset < 0) {
      throw new Error(`Invalid read offset: ${offset}`);
    }
    if (!Number.isInteger(length) || length < 0) {
      throw new Error(`Invalid read length: ${length}`);
    }
    if (!(targetBytes instanceof Uint8Array)) {
      throw new TypeError('readCachedRangeInto requires a Uint8Array targetBytes');
    }

    const readOpts = options && typeof options === 'object' ? options : null;
    const targetOffset =
      readOpts && Number.isInteger(readOpts.targetOffset) && readOpts.targetOffset >= 0
        ? readOpts.targetOffset
        : 0;
    if (targetOffset > targetBytes.byteLength) {
      throw new RangeError(
        `targetOffset (${targetOffset}) exceeds targetBytes length (${targetBytes.byteLength})`,
      );
    }

    if (length === 0) {
      return 0;
    }

    let effectiveLength = length;
    const knownTotal = this._fileSizes.get(cleanPath);
    if (knownTotal !== undefined) {
      if (offset >= knownTotal) {
        return 0;
      }
      const rem = knownTotal - offset;
      if (rem < effectiveLength) {
        effectiveLength = rem;
      }
    }

    const fp = this._resolveEffectiveFingerprint(
      cleanPath,
      readOpts ? readOpts.fileFingerprint : undefined,
    );
    const chunkSize = this.chunkSize;
    const firstChunk = Math.floor(offset / chunkSize);
    const lastChunk = Math.floor((offset + effectiveLength - 1) / chunkSize);

    // Single-chunk fast path: 1 key lookup, 1 Map lookup, 1 validation, O(1) MRU touch, direct copy.
    if (firstChunk === lastChunk) {
      const key = this._chunkKey(cleanPath, firstChunk);
      const chunk = this._memCache.get(key);
      if (chunk === undefined) {
        return -1;
      }
      if (fp !== null && (this._memChunkFingerprints.get(key) || null) !== fp) {
        return -1;
      }
      const chunkStart = firstChunk * chunkSize;
      const maxEnd = offset + effectiveLength - chunkStart;
      const requiredEndInChunk = maxEnd < chunkSize ? maxEnd : chunkSize;
      if (this._validateCachedChunk(key, cleanPath, firstChunk, chunk, requiredEndInChunk) !== 'valid') {
        return -1;
      }
      const sliceStart = offset - chunkStart;
      const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
      const copyLen = sliceEnd > sliceStart ? sliceEnd - sliceStart : 0;
      if (targetBytes.byteLength - targetOffset < copyLen) {
        throw new RangeError(
          `targetBytes capacity (${targetBytes.byteLength - targetOffset}) is smaller than required bytes (${copyLen})`,
        );
      }

      this._touchMemoryHit(key, chunk);
      this._stats.hits += 1;
      this._consumePrefetchHit(key);

      if (copyLen > 0) {
        if (sliceStart === 0 && sliceEnd === chunk.byteLength) {
          targetBytes.set(chunk, targetOffset);
        } else if (copyLen <= 64) {
          for (let j = 0; j < copyLen; j += 1) {
            targetBytes[targetOffset + j] = chunk[sliceStart + j];
          }
        } else {
          targetBytes.set(chunk.subarray(sliceStart, sliceEnd), targetOffset);
        }
        this._stats.zeroAllocCopyHits += 1;
        const seqDetection = this._updateSequentialTracker(cleanPath, offset, copyLen);
        if (chunk.byteLength >= chunkSize && seqDetection.readAheadBytes > 0) {
          this._scheduleReadAhead(cleanPath, firstChunk, seqDetection);
        }
      }
      return copyLen;
    }

    // Multi-chunk path: Pass 1 validates all required chunks and stashes (key, chunk) in
    // reusable scratch arrays so Pass 2 never recomputes keys or re-queries _memCache.
    const keysBuf = this._readIntoKeys;
    const chunksBuf = this._readIntoChunks;
    keysBuf.length = 0;
    chunksBuf.length = 0;

    let totalBytes = 0;
    let actualLastChunk = firstChunk;
    let reachedEof = false;

    try {
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const key = this._chunkKey(cleanPath, idx);
        const chunk = this._memCache.get(key);
        if (chunk === undefined) {
          return -1;
        }
        if (fp !== null && (this._memChunkFingerprints.get(key) || null) !== fp) {
          return -1;
        }
        const chunkStart = idx * chunkSize;
        const maxEnd = offset + effectiveLength - chunkStart;
        const requiredEndInChunk = maxEnd < chunkSize ? maxEnd : chunkSize;
        if (this._validateCachedChunk(key, cleanPath, idx, chunk, requiredEndInChunk) !== 'valid') {
          return -1;
        }
        const sliceStart = offset > chunkStart ? offset - chunkStart : 0;
        const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
        if (sliceEnd > sliceStart) {
          totalBytes += sliceEnd - sliceStart;
        }
        keysBuf.push(key);
        chunksBuf.push(chunk);
        actualLastChunk = idx;
        if (chunk.byteLength < chunkSize) {
          reachedEof = true;
          break;
        }
      }

      if (targetBytes.byteLength - targetOffset < totalBytes) {
        throw new RangeError(
          `targetBytes capacity (${targetBytes.byteLength - targetOffset}) is smaller than required bytes (${totalBytes})`,
        );
      }

      // Pass 2: touch LRU via cached (key, chunk), consume prefetch/readahead hits, and copy directly.
      let cursor = targetOffset;
      const chunkCount = chunksBuf.length;
      for (let i = 0; i < chunkCount; i += 1) {
        const idx = firstChunk + i;
        const key = keysBuf[i];
        const chunk = chunksBuf[i];
        this._touchMemoryHit(key, chunk);
        this._stats.hits += 1;
        this._consumePrefetchHit(key);

        const chunkStart = idx * chunkSize;
        const sliceStart = offset > chunkStart ? offset - chunkStart : 0;
        const maxEnd = offset + effectiveLength - chunkStart;
        const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
        const copyLen = sliceEnd - sliceStart;
        if (copyLen > 0) {
          if (sliceStart === 0 && sliceEnd === chunk.byteLength) {
            targetBytes.set(chunk, cursor);
          } else if (copyLen <= 64) {
            for (let j = 0; j < copyLen; j += 1) {
              targetBytes[cursor + j] = chunk[sliceStart + j];
            }
          } else {
            targetBytes.set(chunk.subarray(sliceStart, sliceEnd), cursor);
          }
          cursor += copyLen;
        }
      }
    } finally {
      keysBuf.length = 0;
      chunksBuf.length = 0;
    }

    if (totalBytes > 0) {
      this._stats.zeroAllocCopyHits += 1;
      const seqDetection = this._updateSequentialTracker(cleanPath, offset, totalBytes);
      if (!reachedEof && seqDetection.readAheadBytes > 0) {
        this._scheduleReadAhead(cleanPath, actualLastChunk, seqDetection);
      }
    }

    return totalBytes;
  }

  /**
   * Zero-copy synchronous L1 fast path for single-chunk ranges: returns a direct read-only
   * `Uint8Array.subarray()` view into the resident cached chunk without allocating a new
   * backing ArrayBuffer or copying bytes. Returns `null` on cache miss or if the requested
   * range spans multiple chunks (callers should fall back to `readCachedRangeInto()`/`readInto()`).
   */
  getCachedSlice(url, offset, length, options = {}) {
    const cleanPath = this._normalizePathOrUrl(url);
    if (!Number.isInteger(offset) || offset < 0) {
      throw new Error(`Invalid read offset: ${offset}`);
    }
    if (!Number.isInteger(length) || length < 0) {
      throw new Error(`Invalid read length: ${length}`);
    }
    if (length === 0) {
      return EMPTY_UINT8;
    }

    let effectiveLength = length;
    const knownTotal = this._fileSizes.get(cleanPath);
    if (knownTotal !== undefined) {
      if (offset >= knownTotal) {
        return EMPTY_UINT8;
      }
      const rem = knownTotal - offset;
      if (rem < effectiveLength) {
        effectiveLength = rem;
      }
    }

    const chunkSize = this.chunkSize;
    const firstChunk = Math.floor(offset / chunkSize);
    const lastChunk = Math.floor((offset + effectiveLength - 1) / chunkSize);
    if (firstChunk !== lastChunk) {
      return null;
    }

    const readOpts = options && typeof options === 'object' ? options : null;
    const fp = this._resolveEffectiveFingerprint(
      cleanPath,
      readOpts ? readOpts.fileFingerprint : undefined,
    );
    const key = this._chunkKey(cleanPath, firstChunk);
    const chunk = this._memCache.get(key);
    if (chunk === undefined) {
      return null;
    }
    if (fp !== null && (this._memChunkFingerprints.get(key) || null) !== fp) {
      return null;
    }
    const chunkStart = firstChunk * chunkSize;
    const maxEnd = offset + effectiveLength - chunkStart;
    const requiredEndInChunk = maxEnd < chunkSize ? maxEnd : chunkSize;
    if (this._validateCachedChunk(key, cleanPath, firstChunk, chunk, requiredEndInChunk) !== 'valid') {
      return null;
    }

    this._touchMemoryHit(key, chunk);
    this._stats.hits += 1;
    this._consumePrefetchHit(key);

    const sliceStart = offset - chunkStart;
    const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
    if (sliceEnd <= sliceStart) {
      return EMPTY_UINT8;
    }

    const view =
      sliceStart === 0 && sliceEnd === chunk.byteLength
        ? chunk
        : chunk.subarray(sliceStart, sliceEnd);
    this._stats.zeroAllocCopyHits += 1;
    const seqDetection = this._updateSequentialTracker(cleanPath, offset, view.byteLength);
    if (chunk.byteLength >= chunkSize && seqDetection.readAheadBytes > 0) {
      this._scheduleReadAhead(cleanPath, firstChunk, seqDetection);
    }
    return view;
  }

  /**
   * Direct-to-destination range read without intermediate output buffer allocation.
   * Supports both ergonomic call signatures:
   *   - `readInto(file, offset, dst, options?)`
   *   - `readInto(url, offset, length, dst, dstOffset?, options?)`
   *
   * Attempts synchronous zero-alloc L1 `readCachedRangeInto` first, and on miss fetches
   * missing chunks from OPFS/HTTP and copies directly into `dst`.
   *
   * @returns {Promise<number>} Exact number of bytes written into `dst`.
   */
  async readInto(url, offset, dstOrLength, dstOrOptions, maybeOffsetOrOptions, maybeOptions) {
    let dst;
    let length;
    let targetOffset = 0;
    let readOpts = null;

    if (dstOrLength instanceof Uint8Array) {
      dst = dstOrLength;
      readOpts = dstOrOptions && typeof dstOrOptions === 'object' ? dstOrOptions : null;
      if (readOpts && Number.isInteger(readOpts.targetOffset) && readOpts.targetOffset >= 0) {
        targetOffset = readOpts.targetOffset;
      }
      length =
        readOpts && Number.isInteger(readOpts.length) && readOpts.length >= 0
          ? readOpts.length
          : dst.byteLength - targetOffset;
    } else if (typeof dstOrLength === 'number') {
      length = dstOrLength;
      dst = dstOrOptions;
      if (!(dst instanceof Uint8Array)) {
        throw new TypeError('RangeCache.readInto requires a Uint8Array destination buffer');
      }
      if (typeof maybeOffsetOrOptions === 'number') {
        targetOffset = maybeOffsetOrOptions;
        readOpts = maybeOptions && typeof maybeOptions === 'object' ? maybeOptions : null;
      } else {
        readOpts =
          maybeOffsetOrOptions && typeof maybeOffsetOrOptions === 'object'
            ? maybeOffsetOrOptions
            : null;
        if (readOpts && Number.isInteger(readOpts.targetOffset) && readOpts.targetOffset >= 0) {
          targetOffset = readOpts.targetOffset;
        }
      }
    } else {
      throw new TypeError('RangeCache.readInto requires a Uint8Array destination buffer');
    }

    if (!Number.isInteger(offset) || offset < 0) {
      throw new Error(`Invalid read offset: ${offset}`);
    }
    if (!Number.isInteger(length) || length < 0) {
      throw new Error(`Invalid read length: ${length}`);
    }
    if (!Number.isInteger(targetOffset) || targetOffset < 0 || targetOffset > dst.byteLength) {
      throw new RangeError(
        `targetOffset (${targetOffset}) exceeds destination length (${dst.byteLength})`,
      );
    }
    if (length === 0) {
      return 0;
    }

    const fileFingerprint =
      readOpts && readOpts.fileFingerprint !== undefined ? readOpts.fileFingerprint : undefined;

    // 1. Synchronous zero-alloc L1 fast path.
    const syncWritten = this.readCachedRangeInto(url, offset, length, dst, {
      targetOffset,
      fileFingerprint,
    });
    if (syncWritten >= 0) {
      return syncWritten;
    }

    // 2. Miss path: fetch chunks and copy directly into `dst` (zero intermediate output buffer).
    const cleanPath = this._normalizePathOrUrl(url);
    let effectiveLength = length;
    const knownTotal = this._fileSizes.get(cleanPath);
    if (knownTotal !== undefined) {
      if (offset >= knownTotal) {
        return 0;
      }
      const rem = knownTotal - offset;
      if (rem < effectiveLength) {
        effectiveLength = rem;
      }
    }
    if (dst.byteLength - targetOffset < effectiveLength && knownTotal !== undefined) {
      throw new RangeError(
        `destination capacity (${dst.byteLength - targetOffset}) is smaller than required bytes (${effectiveLength})`,
      );
    }

    const seqDetection =
      readOpts && readOpts._skipSequentialTracking
        ? null
        : this._updateSequentialTracker(cleanPath, offset, effectiveLength);

    const chunkSize = this.chunkSize;
    const firstChunk = Math.floor(offset / chunkSize);
    const lastChunk = Math.floor((offset + effectiveLength - 1) / chunkSize);

    if (firstChunk === lastChunk) {
      const chunkStart = firstChunk * chunkSize;
      const maxEnd = offset + effectiveLength - chunkStart;
      const requiredEndInChunk = maxEnd < chunkSize ? maxEnd : chunkSize;
      const chunk = await this._getChunk(cleanPath, firstChunk, requiredEndInChunk, {
        fileFingerprint,
      });
      if (chunk.byteLength === 0) {
        return 0;
      }
      const sliceStart = offset - chunkStart;
      const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
      const copyLen = sliceEnd > sliceStart ? sliceEnd - sliceStart : 0;
      if (dst.byteLength - targetOffset < copyLen) {
        throw new RangeError(
          `destination capacity (${dst.byteLength - targetOffset}) is smaller than required bytes (${copyLen})`,
        );
      }
      if (copyLen > 0) {
        if (sliceStart === 0 && sliceEnd === chunk.byteLength) {
          dst.set(chunk, targetOffset);
        } else if (copyLen <= 64) {
          for (let j = 0; j < copyLen; j += 1) {
            dst[targetOffset + j] = chunk[sliceStart + j];
          }
        } else {
          dst.set(chunk.subarray(sliceStart, sliceEnd), targetOffset);
        }
      }
      if (chunk.byteLength >= chunkSize && seqDetection && seqDetection.readAheadBytes > 0) {
        this._scheduleReadAhead(cleanPath, firstChunk, seqDetection);
      }
      return copyLen;
    }

    let chunks;
    if (this.coalesceAdjacent) {
      const specs = new Array(lastChunk - firstChunk + 1);
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * chunkSize;
        const requiredEndInChunk = Math.min(chunkSize, offset + effectiveLength - chunkStart);
        specs[idx - firstChunk] = { chunkIndex: idx, requiredEndInChunk };
      }
      chunks = await this._getChunksCoalesced(cleanPath, specs, {
        isPrefetch: false,
        countMiss: true,
        countCoalesce: true,
        fileFingerprint,
      });
    } else {
      const chunkPromises = new Array(lastChunk - firstChunk + 1);
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * chunkSize;
        const requiredEndInChunk = Math.min(chunkSize, offset + effectiveLength - chunkStart);
        chunkPromises[idx - firstChunk] = this._getChunk(cleanPath, idx, requiredEndInChunk, {
          fileFingerprint,
        });
      }
      chunks = await Promise.all(chunkPromises);
    }

    let totalBytes = 0;
    let reachedEof = false;
    let usedChunks = 0;
    for (let i = 0; i < chunks.length; i += 1) {
      const chunk = chunks[i];
      if (chunk.byteLength === 0) {
        reachedEof = true;
        break;
      }
      const chunkStart = (firstChunk + i) * chunkSize;
      const sliceStart = offset > chunkStart ? offset - chunkStart : 0;
      const maxEnd = offset + effectiveLength - chunkStart;
      const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
      if (sliceEnd > sliceStart) {
        totalBytes += sliceEnd - sliceStart;
      }
      usedChunks = i + 1;
      if (chunk.byteLength < chunkSize) {
        reachedEof = true;
        break;
      }
    }

    if (dst.byteLength - targetOffset < totalBytes) {
      throw new RangeError(
        `destination capacity (${dst.byteLength - targetOffset}) is smaller than required bytes (${totalBytes})`,
      );
    }

    let cursor = targetOffset;
    for (let i = 0; i < usedChunks; i += 1) {
      const chunk = chunks[i];
      const chunkStart = (firstChunk + i) * chunkSize;
      const sliceStart = offset > chunkStart ? offset - chunkStart : 0;
      const maxEnd = offset + effectiveLength - chunkStart;
      const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
      const copyLen = sliceEnd - sliceStart;
      if (copyLen > 0) {
        if (sliceStart === 0 && sliceEnd === chunk.byteLength) {
          dst.set(chunk, cursor);
        } else if (copyLen <= 64) {
          for (let j = 0; j < copyLen; j += 1) {
            dst[cursor + j] = chunk[sliceStart + j];
          }
        } else {
          dst.set(chunk.subarray(sliceStart, sliceEnd), cursor);
        }
        cursor += copyLen;
      }
    }

    if (!reachedEof && seqDetection && seqDetection.readAheadBytes > 0) {
      this._scheduleReadAhead(cleanPath, lastChunk, seqDetection);
    }

    return totalBytes;
  }

  async read(relPath, offset, length, options = {}) {
    const cleanPath = this._normalizeRelPathCached(relPath);
    if (!Number.isInteger(offset) || offset < 0) {
      throw new Error(`Invalid read offset: ${offset}`);
    }
    if (!Number.isInteger(length) || length < 0) {
      throw new Error(`Invalid read length: ${length}`);
    }
    if (length === 0) {
      return new Uint8Array(0);
    }

    const readOpts = options && typeof options === 'object' ? options : null;
    const fileFingerprint =
      readOpts && readOpts.fileFingerprint !== undefined ? readOpts.fileFingerprint : undefined;
    const zeroCopy = Boolean(readOpts && readOpts.zeroCopy);

    let effectiveLength = length;
    const knownTotal = this._fileSizes.get(cleanPath);
    if (knownTotal !== undefined) {
      if (offset >= knownTotal) {
        return new Uint8Array(0);
      }
      const rem = knownTotal - offset;
      if (rem < effectiveLength) {
        effectiveLength = rem;
      }
    }

    const seqDetection =
      readOpts && readOpts._skipSequentialTracking
        ? null
        : this._updateSequentialTracker(cleanPath, offset, effectiveLength);

    const chunkSize = this.chunkSize;
    const firstChunk = Math.floor(offset / chunkSize);
    const lastChunk = Math.floor((offset + effectiveLength - 1) / chunkSize);

    // Single-chunk fast path: avoids chunkPromises array, Promise.all, and slices array.
    if (firstChunk === lastChunk) {
      const chunkStart = firstChunk * chunkSize;
      const maxEnd = offset + effectiveLength - chunkStart;
      const requiredEndInChunk = maxEnd < chunkSize ? maxEnd : chunkSize;
      const chunk = await this._getChunk(cleanPath, firstChunk, requiredEndInChunk, {
        fileFingerprint,
      });
      if (chunk.byteLength === 0) {
        return new Uint8Array(0);
      }
      const sliceStart = offset - chunkStart;
      const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
      if (chunk.byteLength >= chunkSize && seqDetection && seqDetection.readAheadBytes > 0) {
        this._scheduleReadAhead(cleanPath, firstChunk, seqDetection);
      }
      if (sliceEnd <= sliceStart) {
        return new Uint8Array(0);
      }
      if (zeroCopy) {
        return sliceStart === 0 && sliceEnd === chunk.byteLength
          ? chunk
          : chunk.subarray(sliceStart, sliceEnd);
      }
      return chunk.slice(sliceStart, sliceEnd);
    }

    let chunks;
    if (this.coalesceAdjacent) {
      const specs = new Array(lastChunk - firstChunk + 1);
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * chunkSize;
        const requiredEndInChunk = Math.min(chunkSize, offset + effectiveLength - chunkStart);
        specs[idx - firstChunk] = { chunkIndex: idx, requiredEndInChunk };
      }
      chunks = await this._getChunksCoalesced(cleanPath, specs, {
        isPrefetch: false,
        countMiss: true,
        countCoalesce: true,
        fileFingerprint,
      });
    } else {
      const chunkPromises = new Array(lastChunk - firstChunk + 1);
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * chunkSize;
        const requiredEndInChunk = Math.min(chunkSize, offset + effectiveLength - chunkStart);
        chunkPromises[idx - firstChunk] = this._getChunk(cleanPath, idx, requiredEndInChunk, {
          fileFingerprint,
        });
      }
      chunks = await Promise.all(chunkPromises);
    }

    let totalBytes = 0;
    let reachedEof = false;
    let usedChunks = 0;

    for (let i = 0; i < chunks.length; i += 1) {
      const chunk = chunks[i];
      if (chunk.byteLength === 0) {
        reachedEof = true;
        break;
      }
      const chunkStart = (firstChunk + i) * chunkSize;
      const sliceStart = offset > chunkStart ? offset - chunkStart : 0;
      const maxEnd = offset + length - chunkStart;
      const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
      if (sliceEnd > sliceStart) {
        totalBytes += sliceEnd - sliceStart;
      }
      usedChunks = i + 1;
      if (chunk.byteLength < chunkSize) {
        // Reached tail chunk (EOF).
        reachedEof = true;
        break;
      }
    }

    if (!reachedEof && seqDetection && seqDetection.readAheadBytes > 0) {
      this._scheduleReadAhead(cleanPath, lastChunk, seqDetection);
    }

    // Always return a fresh owned buffer so callers (e.g. postMessage transferables)
    // can transfer bytes.buffer without detaching cached Uint8Array instances.
    const out = new Uint8Array(totalBytes);
    let cursor = 0;
    for (let i = 0; i < usedChunks; i += 1) {
      const chunk = chunks[i];
      const chunkStart = (firstChunk + i) * chunkSize;
      const sliceStart = offset > chunkStart ? offset - chunkStart : 0;
      const maxEnd = offset + length - chunkStart;
      const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
      const copyLen = sliceEnd - sliceStart;
      if (copyLen > 0) {
        if (sliceStart === 0 && sliceEnd === chunk.byteLength) {
          out.set(chunk, cursor);
        } else if (copyLen <= 64) {
          for (let j = 0; j < copyLen; j += 1) {
            out[cursor + j] = chunk[sliceStart + j];
          }
        } else {
          out.set(chunk.subarray(sliceStart, sliceEnd), cursor);
        }
        cursor += copyLen;
      }
    }
    return out;
  }

  _chunkAlignedSpanBytes(offset, length) {
    if (length <= 0) return 0;
    const firstChunk = Math.floor(offset / this.chunkSize);
    const lastChunk = Math.floor((offset + length - 1) / this.chunkSize);
    return (lastChunk - firstChunk + 1) * this.chunkSize;
  }

  _planCoalescedSpans(rawEntries, maxGapBytes, maxMergedBytes) {
    const normalizedEntries = [];
    for (let i = 0; i < rawEntries.length; i += 1) {
      const entry = rawEntries[i];
      if (!entry || typeof entry.path !== 'string') continue;
      const offset = Number.isInteger(entry.offset) && entry.offset >= 0 ? entry.offset : 0;
      const length = Number.isInteger(entry.length) && entry.length > 0 ? entry.length : 0;
      if (length === 0) continue;

      let cleanPath;
      try {
        cleanPath = normalizeRelPath(entry.path);
      } catch {
        continue;
      }
      normalizedEntries.push({ path: cleanPath, offset, length });
    }

    const coalesced = coalesceAdjacentRanges(normalizedEntries, maxGapBytes, maxMergedBytes);

    const multiEntrySpans = coalesced.reduce((count, span) => count + (span.mergedCount > 1 ? 1 : 0), 0);
    if (multiEntrySpans === 0) return coalesced.map((span) => ({ ...span, savedBytes: 0 }));
    // For small manifests or a few dense spans, the original direct scan costs less
    // than building an index. Index dispersed manifests with many merged spans only.
    const useIndex = normalizedEntries.length > 128 && multiEntrySpans > 4;

    // Track per-span individual chunk-aligned bytes so bytesSavedByCoalescing captures
    // both overlap reduction and shared-chunk deduplication accurately.
    const byPath = new Map();
    for (let i = 0; i < normalizedEntries.length; i += 1) {
      const e = normalizedEntries[i];
      let list = byPath.get(e.path);
      if (!list) {
        list = [];
        byPath.set(e.path, list);
      }
      list.push(useIndex ? { ...e, order: i } : e);
    }
    if (useIndex) {
      for (const list of byPath.values()) {
        list.sort((a, b) => a.offset - b.offset);
      }
    }

    return coalesced.map((span) => {
      if (span.mergedCount <= 1) return { ...span, savedBytes: 0 };
      const pathEntries = byPath.get(span.path) || [];
      let sumIndividualAligned = 0;
      if (!useIndex) {
        for (const e of pathEntries) {
          if (e.offset >= span.offset && e.offset + e.length <= span.end) {
            sumIndividualAligned += this._chunkAlignedSpanBytes(e.offset, e.length);
          }
        }
      } else {
        // Only entries starting inside this span can contribute. Keep the end inclusive:
        // at very large integer offsets, adding a small positive length can leave the offset unchanged.
        let lo = 0;
        let hi = pathEntries.length;
        while (lo < hi) {
          const mid = lo + Math.floor((hi - lo) / 2);
          if (pathEntries[mid].offset < span.offset) lo = mid + 1;
          else hi = mid;
        }
        const contained = [];
        for (let i = lo; i < pathEntries.length && pathEntries[i].offset <= span.end; i += 1) {
          const e = pathEntries[i];
          if (e.offset + e.length <= span.end) contained.push(e);
        }
        // Retain input summation order even for counts above Number's exact integer range.
        contained.sort((a, b) => a.order - b.order);
        for (const e of contained) {
          sumIndividualAligned += this._chunkAlignedSpanBytes(e.offset, e.length);
        }
      }
      const mergedAligned = this._chunkAlignedSpanBytes(span.offset, span.length);
      const chunkRoundingSavings = Math.max(0, sumIndividualAligned - mergedAligned);
      const headerSavings =
        span.mergedCount > 1 ? (span.mergedCount - 1) * ESTIMATED_HTTP_RANGE_OVERHEAD_BYTES : 0;
      const savedBytes =
        span.mergedCount > 1
          ? Math.max(
              0,
              Math.max(span.overlapBytes, chunkRoundingSavings) + headerSavings - span.gapBytes,
            )
          : 0;
      return { ...span, savedBytes };
    });
  }

  async prefetchRanges(ranges, options = {}) {
    const rawList = Array.isArray(ranges) ? ranges : [];
    const concurrency =
      Number.isInteger(options.concurrency) && options.concurrency > 0
        ? options.concurrency
        : this.maxPrefetchConcurrency;
    const maxGapBytes =
      Number.isFinite(options.maxGapBytes) && options.maxGapBytes >= 0
        ? Math.floor(options.maxGapBytes)
        : this.coalesceMaxGapBytes;
    const maxMergedBytes =
      Number.isFinite(options.maxMergedBytes) && options.maxMergedBytes > 0
        ? Math.floor(options.maxMergedBytes)
        : this.coalesceMaxMergedBytes;

    const spans = this._planCoalescedSpans(rawList, maxGapBytes, maxMergedBytes);
    const uniqueChunkKeys = new Set();
    const spanSpecs = spans.map((span) => {
      const firstChunk = Math.floor(span.offset / this.chunkSize);
      const lastChunk = Math.floor((span.offset + span.length - 1) / this.chunkSize);
      const specs = [];
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * this.chunkSize;
        const requiredEndInChunk = Math.min(this.chunkSize, span.offset + span.length - chunkStart);
        uniqueChunkKeys.add(this._chunkKey(span.path, idx));
        specs.push({ chunkIndex: idx, requiredEndInChunk });
      }
      return { span, specs };
    });

    let nextIndex = 0;
    let prefetched = 0;
    let failed = 0;
    let batchBytesSaved = 0;
    const bytesBefore = this._stats.prefetchBytes;

    const worker = async () => {
      while (true) {
        const current = nextIndex;
        if (current >= spanSpecs.length) return;
        nextIndex += 1;
        const { span, specs } = spanSpecs[current];
        if (span.mergedCount > 1) {
          this._stats.coalescedRangeCount += 1;
          this._stats.bytesSavedByCoalescing += span.savedBytes;
          batchBytesSaved += span.savedBytes;
        }
        try {
          await this._getChunksCoalesced(span.path, specs, {
            isPrefetch: true,
            countMiss: false,
            countCoalesce: span.mergedCount <= 1,
            maxMergedBytes,
          });
          prefetched += specs.length;
        } catch {
          failed += specs.length;
        }
      }
    };

    const workerCount = Math.min(concurrency, spanSpecs.length);
    const workers = [];
    for (let i = 0; i < workerCount; i += 1) {
      workers.push(worker());
    }
    await Promise.all(workers);

    return {
      requestedRanges: rawList.length,
      coalescedRanges: spans.length,
      uniqueChunks: uniqueChunkKeys.size,
      prefetched,
      failed,
      bytesPrefetched: this._stats.prefetchBytes - bytesBefore,
      bytesSavedByCoalescing: batchBytesSaved,
    };
  }

  async readRanges(ranges, options = {}) {
    if (!Array.isArray(ranges) || ranges.length === 0) {
      return [];
    }
    const maxGapBytes =
      Number.isFinite(options.maxGapBytes) && options.maxGapBytes >= 0
        ? Math.floor(options.maxGapBytes)
        : this.coalesceMaxGapBytes;
    const maxMergedBytes =
      Number.isFinite(options.maxMergedBytes) && options.maxMergedBytes > 0
        ? Math.floor(options.maxMergedBytes)
        : this.coalesceMaxMergedBytes;
    const concurrency =
      Number.isInteger(options.concurrency) && options.concurrency > 0
        ? options.concurrency
        : this.maxPrefetchConcurrency;

    // Validate all requested ranges up front.
    const validated = ranges.map((entry) => {
      if (!entry || typeof entry !== 'object') {
        throw new Error('RangeCache.readRanges entry must be an object');
      }
      const cleanPath = normalizeRelPath(entry.path);
      const offset = entry.offset;
      const length = entry.length;
      if (!Number.isInteger(offset) || offset < 0) {
        throw new Error(`Invalid read offset: ${offset}`);
      }
      if (!Number.isInteger(length) || length < 0) {
        throw new Error(`Invalid read length: ${length}`);
      }
      return {
        path: cleanPath,
        offset,
        length,
        fileFingerprint:
          entry.fileFingerprint !== undefined ? entry.fileFingerprint : options.fileFingerprint,
      };
    });

    const nonZero = validated.filter((r) => r.length > 0);
    if (nonZero.length > 0) {
      const spans = this._planCoalescedSpans(nonZero, maxGapBytes, maxMergedBytes);
      let nextIndex = 0;

      const worker = async () => {
        while (true) {
          const current = nextIndex;
          if (current >= spans.length) return;
          nextIndex += 1;
          const span = spans[current];
          if (span.mergedCount > 1) {
            this._stats.coalescedRangeCount += 1;
            this._stats.bytesSavedByCoalescing += span.savedBytes;
          }
          const firstChunk = Math.floor(span.offset / this.chunkSize);
          const lastChunk = Math.floor((span.offset + span.length - 1) / this.chunkSize);
          const specs = [];
          for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
            const chunkStart = idx * this.chunkSize;
            const requiredEndInChunk = Math.min(
              this.chunkSize,
              span.offset + span.length - chunkStart,
            );
            specs.push({ chunkIndex: idx, requiredEndInChunk });
          }
          await this._getChunksCoalesced(span.path, specs, {
            isPrefetch: false,
            countMiss: true,
            countCoalesce: span.mergedCount <= 1,
            maxMergedBytes,
            fileFingerprint: options.fileFingerprint,
          });
        }
      };

      const workerCount = Math.min(concurrency, spans.length);
      const workers = [];
      for (let i = 0; i < workerCount; i += 1) {
        workers.push(worker());
      }
      await Promise.all(workers);
    }

    // Slice exact outputs directly from Phase-1 warmed chunks without double-counting
    // hits or feeding random batch ranges into the sequential readahead tracker.
    const outputs = new Array(validated.length);
    for (let i = 0; i < validated.length; i += 1) {
      const req = validated[i];
      if (req.length === 0) {
        outputs[i] = new Uint8Array(0);
        continue;
      }
      let effectiveLen = req.length;
      if (this._fileSizes.has(req.path)) {
        const knownTotal = this._fileSizes.get(req.path);
        if (req.offset >= knownTotal) {
          outputs[i] = new Uint8Array(0);
          continue;
        }
        effectiveLen = Math.min(req.length, knownTotal - req.offset);
      }
      const chunkSize = this.chunkSize;
      const firstChunk = Math.floor(req.offset / chunkSize);
      const lastChunk = Math.floor((req.offset + effectiveLen - 1) / chunkSize);
      const fp = this._resolveEffectiveFingerprint(req.path, req.fileFingerprint);
      if (firstChunk === lastChunk) {
        const key = this._chunkKey(req.path, firstChunk);
        const chunk = this._memCache.get(key);
        if (
          chunk !== undefined &&
          (fp === null || (this._memChunkFingerprints.get(key) || null) === fp)
        ) {
          const chunkStart = firstChunk * chunkSize;
          const sliceStart = req.offset - chunkStart;
          const maxEnd = req.offset + effectiveLen - chunkStart;
          const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
          outputs[i] = sliceEnd > sliceStart ? chunk.slice(sliceStart, sliceEnd) : new Uint8Array(0);
          continue;
        }
        outputs[i] = await this.read(req.path, req.offset, req.length, {
          fileFingerprint: req.fileFingerprint,
          _skipSequentialTracking: true,
        });
        continue;
      }
      let allResident = true;
      const chunksBuf = this._readIntoChunks;
      chunksBuf.length = 0;
      let totalBytes = 0;
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const key = this._chunkKey(req.path, idx);
        const chunk = this._memCache.get(key);
        if (chunk === undefined) {
          allResident = false;
          break;
        }
        if (fp !== null && (this._memChunkFingerprints.get(key) || null) !== fp) {
          allResident = false;
          break;
        }
        const chunkStart = idx * chunkSize;
        const sliceStart = req.offset > chunkStart ? req.offset - chunkStart : 0;
        const maxEnd = req.offset + effectiveLen - chunkStart;
        const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
        if (sliceEnd > sliceStart) {
          totalBytes += sliceEnd - sliceStart;
        }
        chunksBuf.push(chunk);
        if (chunk.byteLength < chunkSize) {
          break;
        }
      }
      if (allResident) {
        const out = new Uint8Array(totalBytes);
        let cursor = 0;
        for (let c = 0; c < chunksBuf.length; c += 1) {
          const chunk = chunksBuf[c];
          const chunkStart = (firstChunk + c) * chunkSize;
          const sliceStart = req.offset > chunkStart ? req.offset - chunkStart : 0;
          const maxEnd = req.offset + effectiveLen - chunkStart;
          const sliceEnd = chunk.byteLength < maxEnd ? chunk.byteLength : maxEnd;
          const copyLen = sliceEnd - sliceStart;
          if (copyLen > 0) {
            if (sliceStart === 0 && sliceEnd === chunk.byteLength) {
              out.set(chunk, cursor);
            } else {
              out.set(chunk.subarray(sliceStart, sliceEnd), cursor);
            }
            cursor += copyLen;
          }
        }
        chunksBuf.length = 0;
        outputs[i] = out;
      } else {
        chunksBuf.length = 0;
        outputs[i] = await this.read(req.path, req.offset, req.length, {
          fileFingerprint: req.fileFingerprint,
          _skipSequentialTracking: true,
        });
      }
    }
    return outputs;
  }

  async prefetchBootset(entries, options = {}) {
    const rawList = Array.isArray(entries)
      ? entries
      : entries && Array.isArray(entries.entries)
        ? entries.entries
        : [];

    const useCoalesce =
      options.coalesce === true || (this.coalesceAdjacent && options.coalesce !== false);

    if (useCoalesce) {
      const summary = await this.prefetchRanges(rawList, options);
      return {
        requestedEntries: rawList.length,
        uniqueChunks: summary.uniqueChunks,
        prefetched: summary.prefetched,
        failed: summary.failed,
        coalescedRanges: summary.coalescedRanges,
      };
    }

    const concurrency =
      Number.isInteger(options.concurrency) && options.concurrency > 0
        ? options.concurrency
        : this.maxPrefetchConcurrency;

    const tasks = [];
    const taskIndexByKey = new Map();

    for (const entry of rawList) {
      if (!entry || typeof entry.path !== 'string') continue;
      const offset = Number.isInteger(entry.offset) && entry.offset >= 0 ? entry.offset : 0;
      const length = Number.isInteger(entry.length) && entry.length > 0 ? entry.length : 0;
      if (length === 0) continue;

      let cleanPath;
      try {
        cleanPath = normalizeRelPath(entry.path);
      } catch {
        continue;
      }

      const firstChunk = Math.floor(offset / this.chunkSize);
      const lastChunk = Math.floor((offset + length - 1) / this.chunkSize);
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * this.chunkSize;
        const requiredEndInChunk = Math.min(this.chunkSize, offset + length - chunkStart);
        const key = this._chunkKey(cleanPath, idx);
        if (!taskIndexByKey.has(key)) {
          taskIndexByKey.set(key, tasks.length);
          tasks.push({ path: cleanPath, chunkIndex: idx, requiredEndInChunk });
        } else {
          const existingTask = tasks[taskIndexByKey.get(key)];
          if (requiredEndInChunk > existingTask.requiredEndInChunk) {
            existingTask.requiredEndInChunk = requiredEndInChunk;
          }
        }
      }
    }

    let nextIndex = 0;
    let prefetched = 0;
    let failed = 0;

    const worker = async () => {
      while (true) {
        const current = nextIndex;
        if (current >= tasks.length) return;
        nextIndex += 1;
        const task = tasks[current];
        try {
          await this._getChunk(task.path, task.chunkIndex, task.requiredEndInChunk, {
            isPrefetch: true,
            countMiss: true,
          });
          prefetched += 1;
        } catch {
          failed += 1;
        }
      }
    };

    const workerCount = Math.min(concurrency, tasks.length);
    const workers = [];
    for (let i = 0; i < workerCount; i += 1) {
      workers.push(worker());
    }
    await Promise.all(workers);

    return {
      requestedEntries: rawList.length,
      uniqueChunks: tasks.length,
      prefetched,
      failed,
    };
  }

  clear() {
    this._memCache.clear();
    this._probationChunks.clear();
    this._protectedChunks.clear();
    this._probationPrefetched.clear();
    this._memMruKey = null;
    this._probationMruKey = null;
    this._protectedMruKey = null;
    this._seqTrackerMruKey = null;
    this._pathChunkKeys.clear();
    this._protectedBytesCached = 0;
    this._chunkLastSwitch.clear();
    this._lastAccessedChunkKey = null;
    this._memChunkFingerprints.clear();
    this._opfsWriteQueue.clear();
    this._opfsQueuedBytes = 0;
    this._inFlightOpfsLookups.clear();
    this._seqTrackers.clear();
    this._prefetchedKeys.clear();
    this._readaheadKeys.clear();
    this._readaheadByPath.clear();
    this._resetOpfsDirIndex();
    this._bytesCached = 0;
    this._stats.adaptiveReadaheadWindowBytes = 0;
  }

  resetStats() {
    this._stats = this._createEmptyStats();
  }

  getStats() {
    const opfsLookups = this._stats.opfsHits + this._stats.opfsMisses;
    const opfsHitRate =
      opfsLookups > 0 ? Math.max(0, Math.min(1, this._stats.opfsHits / opfsLookups)) : 0;
    return {
      hits: this._stats.hits,
      misses: this._stats.misses,
      opfsHits: this._stats.opfsHits,
      opfsMisses: this._stats.opfsMisses,
      opfsWrites: this._stats.opfsWrites,
      opfsWriteErrors: this._stats.opfsWriteErrors,
      opfsStaleInvalidations: this._stats.opfsStaleInvalidations,
      opfsHitRate,
      evictions: this._stats.evictions,
      quotaEvictions: this._stats.quotaEvictions,
      quotaRetrySuccesses: this._stats.quotaRetrySuccesses,
      corruptEvictions: this._stats.corruptEvictions,
      coalescedRangeCount: this._stats.coalescedRangeCount,
      sequentialHits: this._stats.sequentialHits,
      prefetchHits: this._stats.prefetchHits,
      prefetchBytes: this._stats.prefetchBytes,
      bytesSavedByCoalescing: this._stats.bytesSavedByCoalescing,
      adaptiveReadaheadWindowBytes: this._stats.adaptiveReadaheadWindowBytes,
      sequentialStreakPeak: this._stats.sequentialStreakPeak,
      readaheadHits: this._stats.readaheadHits,
      readaheadWastedBytes: this._stats.readaheadWastedBytes,
      zeroAllocCopyHits: this._stats.zeroAllocCopyHits,
      mruFastPathHits: this._stats.mruFastPathHits,
      bytesCached: this._bytesCached,
      entries: this._memCache.size,
    };
  }
}

const exportedApi = {
  RangeCache,
  DEFAULT_CHUNK_SIZE,
  DEFAULT_MAX_BYTES,
  DEFAULT_PREFETCH_CONCURRENCY,
  DEFAULT_COALESCE_MAX_GAP_BYTES,
  DEFAULT_COALESCE_MAX_MERGED_BYTES,
  DEFAULT_MAX_READ_AHEAD_BYTES,
  DEFAULT_MIN_READAHEAD_BYTES,
  DEFAULT_MAX_READAHEAD_BYTES,
  DEFAULT_SEQUENTIAL_STREAK_THRESHOLD,
  DEFAULT_MAX_PENDING_OPFS_WRITES,
  DEFAULT_MAX_PENDING_OPFS_WRITE_BYTES,
  DEFAULT_MAX_OPFS_WRITE_CONCURRENCY,
  normalizeRelPath,
  buildFileFingerprint,
  opfsChunkFileName,
  parseContentRange,
  openOpfsCacheDir,
  isQuotaExceededError,
  coalesceAdjacentRanges,
  detectSequentialRead,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6RangeCache = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
