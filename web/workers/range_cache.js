'use strict';

// Fixed-size HTTP 206 Range chunk cache with memory LRU, optional OPFS L2 persistence,
// concurrent in-flight deduplication, and bootset.json prefetching.
// Compatible with classic Web Workers (importScripts) and Node CommonJS/ESM tests.

const DEFAULT_CHUNK_SIZE = 256 * 1024; // 256 KiB
const DEFAULT_MAX_BYTES = 64 * 1024 * 1024; // 64 MiB
const DEFAULT_PREFETCH_CONCURRENCY = 4;

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
  const bytes = new TextEncoder().encode(str);
  let hex = '';
  for (let i = 0; i < bytes.length; i += 1) {
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  return hex;
}

function fnv1a64Hex(str) {
  // Deterministic two-lane 32-bit FNV-1a hash (128-bit hex) for long path keys.
  const bytes = new TextEncoder().encode(str);
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

function opfsChunkFileName(relPath, chunkIndex, chunkSize) {
  const encoded = utf8ToHex(relPath);
  const pathToken = encoded.length <= 128 ? encoded : `${encoded.slice(0, 64)}_${fnv1a64Hex(relPath)}`;
  return `c_${chunkSize}_${chunkIndex}_${pathToken}.bin`;
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

    this._memCache = new Map();
    this._inFlight = new Map();
    this._bytesCached = 0;
    this._stats = {
      hits: 0,
      misses: 0,
      opfsHits: 0,
      evictions: 0,
    };
  }

  setOpfsDir(opfsDir) {
    this.opfsDir = opfsDir || null;
  }

  _chunkKey(relPath, chunkIndex) {
    return `${relPath}::${chunkIndex}`;
  }

  _touchMemory(key) {
    if (!this._memCache.has(key)) return null;
    const bytes = this._memCache.get(key);
    this._memCache.delete(key);
    this._memCache.set(key, bytes);
    return bytes;
  }

  _putMemory(key, bytes) {
    if (!(bytes instanceof Uint8Array)) return;
    if (this._memCache.has(key)) {
      const existing = this._memCache.get(key);
      this._bytesCached -= existing.byteLength;
      this._memCache.delete(key);
    }

    if (this.maxBytes <= 0 || bytes.byteLength > this.maxBytes) {
      return;
    }

    while (this._bytesCached + bytes.byteLength > this.maxBytes && this._memCache.size > 0) {
      const oldestKey = this._memCache.keys().next().value;
      const oldestBytes = this._memCache.get(oldestKey);
      this._memCache.delete(oldestKey);
      this._bytesCached -= oldestBytes.byteLength;
      this._stats.evictions += 1;
    }

    this._memCache.set(key, bytes);
    this._bytesCached += bytes.byteLength;
  }

  async _readOpfs(relPath, chunkIndex) {
    if (!this.opfsDir) return null;
    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize);
    try {
      if (typeof this.opfsDir.getFileHandle === 'function') {
        const handle = await this.opfsDir.getFileHandle(fileName, { create: false });
        const file = await handle.getFile();
        const buf = await file.arrayBuffer();
        return new Uint8Array(buf);
      }
      if (typeof this.opfsDir.get === 'function') {
        const val = await this.opfsDir.get(fileName);
        if (val instanceof Uint8Array) return val.slice();
        if (val instanceof ArrayBuffer) return new Uint8Array(val.slice(0));
      }
    } catch {
      // Cache miss or OPFS unavailable/corrupt -> gracefully fall back to network.
    }
    return null;
  }

  async _writeOpfs(relPath, chunkIndex, bytes) {
    if (!this.opfsDir || !(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
      return;
    }
    const fileName = opfsChunkFileName(relPath, chunkIndex, this.chunkSize);
    try {
      if (typeof this.opfsDir.getFileHandle === 'function') {
        const handle = await this.opfsDir.getFileHandle(fileName, { create: true });
        if (typeof handle.createWritable === 'function') {
          const writable = await handle.createWritable();
          await writable.write(bytes);
          await writable.close();
        } else if (typeof handle.createSyncAccessHandle === 'function') {
          const syncHandle = await handle.createSyncAccessHandle();
          try {
            syncHandle.truncate(0);
            syncHandle.write(bytes, { at: 0 });
            syncHandle.flush();
          } finally {
            syncHandle.close();
          }
        }
      } else if (typeof this.opfsDir.set === 'function') {
        await this.opfsDir.set(fileName, bytes.slice());
      }
    } catch {
      // Ignore OPFS write failures (e.g. quota exceeded) and continue memory-only.
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
      return new Uint8Array(0);
    }

    if (!res.ok && res.status !== 206) {
      throw new Error(`HTTP ${res.status} reading ${relPath}[${chunkStart}..${chunkEnd}]`);
    }

    const raw = new Uint8Array(await res.arrayBuffer());

    if (res.status === 206) {
      const contentRangeHeader =
        res.headers && typeof res.headers.get === 'function'
          ? res.headers.get('Content-Range') || res.headers.get('content-range')
          : null;
      const parsed = parseContentRange(contentRangeHeader);
      if (parsed) {
        const expectedLen = parsed.end - parsed.start + 1;
        const relOffset = chunkStart >= parsed.start ? chunkStart - parsed.start : 0;
        const available = Math.min(raw.byteLength, expectedLen);
        if (relOffset >= available) {
          return new Uint8Array(0);
        }
        const sliceLen = Math.min(this.chunkSize, available - relOffset);
        return raw.slice(relOffset, relOffset + sliceLen);
      }
      if (raw.byteLength <= this.chunkSize) {
        return raw;
      }
      return raw.slice(0, this.chunkSize);
    }

    // 200 OK fallback (server returned full resource instead of 206 Partial Content).
    if (chunkStart >= raw.byteLength) {
      return new Uint8Array(0);
    }
    return raw.slice(chunkStart, Math.min(raw.byteLength, chunkStart + this.chunkSize));
  }

  async _getChunk(relPath, chunkIndex) {
    const key = this._chunkKey(relPath, chunkIndex);

    const memHit = this._touchMemory(key);
    if (memHit !== null) {
      this._stats.hits += 1;
      return memHit;
    }

    const existingFlight = this._inFlight.get(key);
    if (existingFlight) {
      return existingFlight;
    }

    const loadPromise = (async () => {
      const opfsBytes = await this._readOpfs(relPath, chunkIndex);
      if (opfsBytes !== null) {
        this._stats.opfsHits += 1;
        this._stats.hits += 1;
        this._putMemory(key, opfsBytes);
        return opfsBytes;
      }

      this._stats.misses += 1;
      const netBytes = await this._fetchChunkFromNetwork(relPath, chunkIndex);
      if (netBytes.byteLength > 0) {
        this._putMemory(key, netBytes);
        await this._writeOpfs(relPath, chunkIndex, netBytes);
      }
      return netBytes;
    })();

    this._inFlight.set(key, loadPromise);
    try {
      return await loadPromise;
    } finally {
      if (this._inFlight.get(key) === loadPromise) {
        this._inFlight.delete(key);
      }
    }
  }

  async read(relPath, offset, length) {
    const cleanPath = normalizeRelPath(relPath);
    if (!Number.isInteger(offset) || offset < 0) {
      throw new Error(`Invalid read offset: ${offset}`);
    }
    if (!Number.isInteger(length) || length < 0) {
      throw new Error(`Invalid read length: ${length}`);
    }
    if (length === 0) {
      return new Uint8Array(0);
    }

    const firstChunk = Math.floor(offset / this.chunkSize);
    const lastChunk = Math.floor((offset + length - 1) / this.chunkSize);

    const chunkPromises = [];
    for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
      chunkPromises.push(this._getChunk(cleanPath, idx));
    }
    const chunks = await Promise.all(chunkPromises);

    const slices = [];
    let totalBytes = 0;

    for (let i = 0; i < chunks.length; i += 1) {
      const chunkIndex = firstChunk + i;
      const chunk = chunks[i];
      if (chunk.byteLength === 0) {
        break;
      }
      const chunkStart = chunkIndex * this.chunkSize;
      const sliceStart = Math.max(0, offset - chunkStart);
      const sliceEnd = Math.min(chunk.byteLength, offset + length - chunkStart);
      if (sliceEnd > sliceStart) {
        const part = chunk.subarray(sliceStart, sliceEnd);
        slices.push(part);
        totalBytes += part.byteLength;
      }
      if (chunk.byteLength < this.chunkSize) {
        // Reached tail chunk (EOF).
        break;
      }
    }

    // Always return a fresh owned buffer so callers (e.g. postMessage transferables)
    // can transfer bytes.buffer without detaching cached Uint8Array instances.
    const out = new Uint8Array(totalBytes);
    let cursor = 0;
    for (const part of slices) {
      out.set(part, cursor);
      cursor += part.byteLength;
    }
    return out;
  }

  async prefetchBootset(entries, options = {}) {
    const rawList = Array.isArray(entries)
      ? entries
      : entries && Array.isArray(entries.entries)
        ? entries.entries
        : [];

    const concurrency =
      Number.isInteger(options.concurrency) && options.concurrency > 0
        ? options.concurrency
        : this.maxPrefetchConcurrency;

    const tasks = [];
    const seen = new Set();

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
        const key = this._chunkKey(cleanPath, idx);
        if (!seen.has(key)) {
          seen.add(key);
          tasks.push({ path: cleanPath, chunkIndex: idx });
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
          await this._getChunk(task.path, task.chunkIndex);
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
    this._bytesCached = 0;
  }

  getStats() {
    return {
      hits: this._stats.hits,
      misses: this._stats.misses,
      opfsHits: this._stats.opfsHits,
      evictions: this._stats.evictions,
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
  normalizeRelPath,
  opfsChunkFileName,
  parseContentRange,
  openOpfsCacheDir,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6RangeCache = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
