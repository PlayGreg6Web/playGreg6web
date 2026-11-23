'use strict';

// HTTP Range (206 Partial Content) VFS worker + OPFS/LRU RangeCache + bootset prefetch.
if (typeof importScripts === 'function') {
  importScripts('./range_cache.js');
}

const rangeCacheModule =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6RangeCache) ||
  (typeof require === 'function' ? require('./range_cache.js') : null);

let dataBase = '/data/';
let manifest = null;
let rangeCache = null;

function ensureRangeCache() {
  if (!rangeCache) {
    if (!rangeCacheModule || typeof rangeCacheModule.RangeCache !== 'function') {
      throw new Error('PlayGta6RangeCache unavailable in io_worker');
    }
    rangeCache = new rangeCacheModule.RangeCache({ dataBase });
  }
  return rangeCache;
}

function emptyStats() {
  return {
    hits: 0,
    misses: 0,
    opfsHits: 0,
    evictions: 0,
    bytesCached: 0,
    entries: 0,
  };
}

async function init(options = {}) {
  const base = typeof options === 'string' ? options : options.dataBase || '/data/';
  dataBase = base.endsWith('/') ? base : `${base}/`;
  manifest = null;

  const opfsDir =
    rangeCacheModule && typeof rangeCacheModule.openOpfsCacheDir === 'function'
      ? await rangeCacheModule.openOpfsCacheDir('playgta6-cache')
      : null;

  rangeCache = new rangeCacheModule.RangeCache({
    dataBase,
    chunkSize: options.chunkSize,
    maxBytes: options.maxBytes,
    maxPrefetchConcurrency: options.maxPrefetchConcurrency,
    opfsDir,
  });

  let manifestPart = 'no data-manifest.json yet';
  try {
    const res = await fetch(`${dataBase}data-manifest.json`);
    if (res.ok) {
      manifest = await res.json();
      manifestPart = `${manifest.file_count || 0} files indexed`;
    }
  } catch {
    // Manifest is optional until the user generates local-data/data-manifest.json.
  }

  let bootsetCount = 0;
  try {
    const bootRes = await fetch(`${dataBase}bootset.json`);
    if (bootRes.ok) {
      const bootJson = await bootRes.json();
      const entries = Array.isArray(bootJson)
        ? bootJson
        : bootJson && Array.isArray(bootJson.entries)
          ? bootJson.entries
          : [];
      bootsetCount = entries.length;
      if (bootsetCount > 0) {
        // Prefetch hot initial ranges in the background without blocking worker readiness.
        rangeCache.prefetchBootset(entries).catch(() => {});
      }
    }
  } catch {
    // bootset.json is optional; absence must never fail boot.
  }

  const cacheTier = opfsDir ? 'OPFS+LRU' : 'memory LRU';
  const bootPart = bootsetCount > 0 ? `, prefetching ${bootsetCount} bootset ranges` : '';
  const summary = `${manifestPart} (${cacheTier}${bootPart})`;

  self.postMessage({
    ready: true,
    summary,
    opfs: Boolean(opfsDir),
    bootsetEntries: bootsetCount,
  });
}

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = async (ev) => {
    const msg = ev.data || {};
    try {
      if (msg.init) {
        await init(msg);
      } else if (msg.read) {
        const cache = ensureRangeCache();
        const bytes = await cache.read(msg.path, msg.offset, msg.length);
        self.postMessage({ id: msg.id, bytes }, [bytes.buffer]);
      } else if (msg.stats) {
        const stats = rangeCache ? rangeCache.getStats() : emptyStats();
        self.postMessage({ id: msg.id, stats });
      }
    } catch (err) {
      self.postMessage({ id: msg.id, error: String(err) });
    }
  };
}

