'use strict';

// HTTP Range (206 Partial Content) VFS worker + OPFS/LRU RangeCache + IoPriorityScheduler
// + lazy Rpf9Vfs / RpfVfsMountTable (rpf://<archivePath>/<innerPath>) + SAB HttpfsMailbox.
(function initPlayGta6IoWorker() {
  const RPF_URI_PREFIX = 'rpf://';

  const FALLBACK_IO_PRIORITY = Object.freeze({
    CRITICAL: 0,
    HIGH: 1,
    NORMAL: 2,
    PREFETCH: 3,
  });

  function loadScriptIsolated(url) {
    // Classic Web Worker fallback when two sibling UMD scripts share a top-level lexical binding
    // (e.g. `const exportedApi` in range_cache.js and rpf9_vfs.js). Evaluating in a Function scope
    // isolates top-level const bindings while still attaching the global PlayGta6* namespace.
    if (typeof XMLHttpRequest !== 'function') {
      return false;
    }
    try {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', url, false);
      xhr.send(null);
      if (xhr.status >= 200 && xhr.status < 300 && typeof xhr.responseText === 'string') {
        const exec = new Function(xhr.responseText);
        exec();
        return true;
      }
    } catch {
      // Ignore; module remains optional or can be injected programmatically.
    }
    return false;
  }

  function importWorkerScript(url, globalKey, required) {
    if (typeof globalThis !== 'undefined' && globalThis[globalKey]) {
      return;
    }
    if (typeof importScripts !== 'function') {
      return;
    }
    try {
      importScripts(url);
    } catch (err) {
      if (!loadScriptIsolated(url) && required) {
        throw err;
      }
    }
  }

  if (typeof importScripts === 'function') {
    importWorkerScript('./range_cache.js', 'PlayGta6RangeCache', true);
    importWorkerScript('./httpfs_mailbox.js', 'PlayGta6HttpfsMailbox', true);
    importWorkerScript('./io_scheduler.js', 'PlayGta6IoScheduler', false);
    importWorkerScript('./rpf9_vfs.js', 'PlayGta6Rpf9Vfs', false);
    importWorkerScript('./perf_profile.js', 'PlayGta6PerfProfile', false);
  }

  function tryRequire(modPath) {
    if (typeof require !== 'function') {
      return null;
    }
    try {
      return require(modPath);
    } catch {
      return null;
    }
  }

  function resolveRangeCacheModule(override) {
    return (
      override ||
      (typeof globalThis !== 'undefined' && globalThis.PlayGta6RangeCache) ||
      tryRequire('./range_cache.js')
    );
  }

  function resolveHttpfsMailboxModule(override) {
    return (
      override ||
      (typeof globalThis !== 'undefined' && globalThis.PlayGta6HttpfsMailbox) ||
      tryRequire('./httpfs_mailbox.js')
    );
  }

  function resolveIoSchedulerModule(override) {
    return (
      override ||
      (typeof globalThis !== 'undefined' && globalThis.PlayGta6IoScheduler) ||
      tryRequire('./io_scheduler.js')
    );
  }

  function resolveRpf9VfsModule(override) {
    return (
      override ||
      (typeof globalThis !== 'undefined' && globalThis.PlayGta6Rpf9Vfs) ||
      tryRequire('./rpf9_vfs.js')
    );
  }

  const FALLBACK_IO_PERF_PROFILES = Object.freeze({
    low: Object.freeze({
      name: 'low',
      ioSlabCacheBytes: 64 * 1024 * 1024,
      ioMaxConcurrentLanes: 4,
    }),
    balanced: Object.freeze({
      name: 'balanced',
      ioSlabCacheBytes: 128 * 1024 * 1024,
      ioMaxConcurrentLanes: 6,
    }),
    high: Object.freeze({
      name: 'high',
      ioSlabCacheBytes: 192 * 1024 * 1024,
      ioMaxConcurrentLanes: 8,
    }),
  });

  function resolvePerfProfileModule(override) {
    return (
      override ||
      (typeof globalThis !== 'undefined' && globalThis.PlayGta6PerfProfile) ||
      tryRequire('./perf_profile.js')
    );
  }

  function coerceIoPerfProfile(profileOrName, overrideMod) {
    const mod = resolvePerfProfileModule(overrideMod);
    if (mod && typeof mod.coercePerfProfile === 'function') {
      return mod.coercePerfProfile(profileOrName);
    }
    if (typeof profileOrName === 'string') {
      const key = profileOrName.trim().toLowerCase();
      if (key === 'auto') {
        return FALLBACK_IO_PERF_PROFILES.balanced;
      }
      if (Object.prototype.hasOwnProperty.call(FALLBACK_IO_PERF_PROFILES, key)) {
        return FALLBACK_IO_PERF_PROFILES[key];
      }
      throw new RangeError(`Unknown performance profile "${profileOrName}"`);
    }
    if (profileOrName && typeof profileOrName === 'object') {
      const name =
        typeof profileOrName.name === 'string'
          ? profileOrName.name.trim().toLowerCase()
          : typeof profileOrName.preset === 'string'
            ? profileOrName.preset.trim().toLowerCase()
            : 'balanced';
      const base = Object.prototype.hasOwnProperty.call(FALLBACK_IO_PERF_PROFILES, name)
        ? FALLBACK_IO_PERF_PROFILES[name]
        : FALLBACK_IO_PERF_PROFILES.balanced;
      return Object.freeze({
        name: base.name,
        ioSlabCacheBytes:
          profileOrName.ioSlabCacheBytes !== undefined
            ? profileOrName.ioSlabCacheBytes
            : profileOrName.slabCacheBytes !== undefined
              ? profileOrName.slabCacheBytes
              : profileOrName.slabMaxBytes !== undefined
                ? profileOrName.slabMaxBytes
                : base.ioSlabCacheBytes,
        ioMaxConcurrentLanes:
          profileOrName.ioMaxConcurrentLanes !== undefined
            ? profileOrName.ioMaxConcurrentLanes
            : profileOrName.maxConcurrentLanes !== undefined
              ? profileOrName.maxConcurrentLanes
              : profileOrName.maxConcurrentSlabFetches !== undefined
                ? profileOrName.maxConcurrentSlabFetches
                : base.ioMaxConcurrentLanes,
      });
    }
    throw new TypeError('perfProfile must be a preset name or profile object');
  }

  const defaultIoSchedulerMod = resolveIoSchedulerModule();
  const IO_PRIORITY =
    (defaultIoSchedulerMod && defaultIoSchedulerMod.IO_PRIORITY) || FALLBACK_IO_PRIORITY;

  const utf8Decoder = new TextDecoder('utf-8', { fatal: false });
  const utf8Encoder = new TextEncoder();

  const MOUNT_CACHE_PREFIX = '__pg6_mount__/';
  const DUMP_VIRTUAL_PREFIX = '/pg6_dump/';
  const DEFAULT_DUMP_ENDPOINT = '/__pg6_dump/';
  const DEFAULT_DUMP_MAX_BUFFER_BYTES = 16 * 1024 * 1024; // 16 MiB bounded per-handle dump buffer cap
  const DEFAULT_DUMP_FLUSH_CHUNK_BYTES = 1024 * 1024; // 1 MiB chunk size for sequential dump POSTs
  const HTTPFS_OP_WRITE = 4;
  const HTTPFS_OPEN_FLAG_WRITE = 1n;
  const HTTPFS_OPEN_FLAG_APPEND = 2n;

  const GAME_PACKAGE_EXT_RE = /\.(?:rpf|ttpf|dll|png|xvd|xvc|bin|pri|bk2|bik)$/i;
  const IMMUTABLE_HTTPFS_EXT_RE = /\.(?:rpf|pgim|dll|ttpf|pack|reflect\.json|bk2|bik)$/i;
  const VIDEO_STREAM_EXT_RE = /\.(?:bk2|bik)$/i;

  const DEFAULT_IMMUTABLE_CHUNK_SIZE = 64 * 1024; // 64 KiB aligned chunks
  const DEFAULT_IMMUTABLE_MAX_BYTES = 128 * 1024 * 1024; // 128 MiB RAM cap
  const DEFAULT_WHOLE_FILE_MAX_BYTES = 256 * 1024 * 1024; // 256 MiB safe cap for eager .pgim / manifest buffering
  const DEFAULT_WHOLE_FILE_TOTAL_MAX_BYTES = 256 * 1024 * 1024; // 256 MiB aggregate cap across all active whole-file buffers
  const DEFAULT_SLAB_SIZE = 2 * 1024 * 1024; // 2 MiB aligned read-ahead slabs
  const DEFAULT_SLAB_MAX_BYTES = 192 * 1024 * 1024; // 192 MiB bounded LRU cap (96 x 2 MiB slabs; holds ~115 MB shaders.rpf working set without thrashing)
  const MAX_SLAB_SIZE = 16 * 1024 * 1024; // 16 MiB upper safety cap per slab
  const MAX_SLAB_CACHE_BYTES = 256 * 1024 * 1024; // 256 MiB upper safety cap for worker slab RAM
  const DEFAULT_SLAB_PREFETCH_AHEAD = 1; // Lightweight +1 sequential read-ahead slab (opt-out via prefetchAheadSlabs: 0)
  const DEFAULT_AUTO_PREFETCH_MIN_SLAB_SIZE = 512 * 1024; // Enable default +1 sequential prefetch for production >=512 KiB aligned slabs
  const MAX_SLAB_PREFETCH_AHEAD = 8;
  const DEFAULT_MAX_CONCURRENT_SLAB_FETCHES = 8;
  const MAX_CONCURRENT_SLAB_FETCHES = 16;
  const DEFAULT_SEQUENTIAL_THRESHOLD_READS = 3;
  const MAX_STREAM_TRACKER_ENTRIES = 64;
  const SLRU_MIN_SLABS = 4; // Enable 2-segment scan-resistant SLRU when budget fits >= 4 slabs
  const DEFAULT_NEGATIVE_CACHE_TTL_MS = 30 * 1000; // 30s bounded TTL for missing optional files
  const DEFAULT_NEGATIVE_CACHE_MAX_ENTRIES = 512;
  const DEFAULT_METADATA_MAX_ENTRIES = 4096;
  const MAX_FILE_FINGERPRINTS = 4096;
  const MAX_COALESCED_FETCH_BYTES = 4 * 1024 * 1024; // 4 MiB cap per single HTTP 206 fetch
  const DEFAULT_BATCH_COALESCE_MAX_GAP_BYTES = 64 * 1024; // 64 KiB max gap merged between same-asset batch ranges
  const DEFAULT_BATCH_COALESCE_MAX_BYTES = 2 * 1024 * 1024; // 2 MiB default cap per coalesced batch span
  const DEFAULT_HOT_OPFS_RPF_ALLOWLIST = Object.freeze(['shaders.rpf', 'core_bvh4.rpf']);
  const DEFAULT_OPFS_L2_MAX_PENDING_WRITES = 64;
  const DEFAULT_OPFS_L2_MAX_PENDING_WRITE_BYTES = 32 * 1024 * 1024; // 32 MiB byte cap on queued OPFS L2 write-through payloads
  const DEFAULT_OPFS_L2_MAX_WRITE_CONCURRENCY = 4;
  const DEFAULT_OPFS_L2_MAX_BYTES = 512 * 1024 * 1024; // 512 MiB bounded OPFS L2 cap
  const MAX_OPFS_L2_TRACKED_ENTRIES = 16384;
  const OPFS_L2_CHUNK_READ_CONCURRENCY = 4;

  const SLAB_TUNING_PROFILES = Object.freeze({
    default: Object.freeze({
      slabSize: DEFAULT_SLAB_SIZE,
      slabMaxBytes: DEFAULT_SLAB_MAX_BYTES,
      prefetchAheadSlabs: DEFAULT_SLAB_PREFETCH_AHEAD,
      maxConcurrentSlabFetches: DEFAULT_MAX_CONCURRENT_SLAB_FETCHES,
      sequentialThresholdReads: DEFAULT_SEQUENTIAL_THRESHOLD_READS,
    }),
    'fast-boot': Object.freeze({
      slabSize: DEFAULT_SLAB_SIZE,
      slabMaxBytes: MAX_SLAB_CACHE_BYTES, // 256 MiB bounded RAM LRU cap for heavy .rpf asset-loading phases
      prefetchAheadSlabs: 1,
      maxConcurrentSlabFetches: DEFAULT_MAX_CONCURRENT_SLAB_FETCHES,
      sequentialThresholdReads: 2,
    }),
    video4k: Object.freeze({
      slabSize: 4 * 1024 * 1024, // 4 MiB slabs for smooth 4K Bink2 sequential reads
      slabMaxBytes: 192 * 1024 * 1024, // 192 MiB bounded LRU cap (48 slabs; keeps shader/asset working set warm during 4K video playback)
      prefetchAheadSlabs: 2, // Bounded 2-slab sequential read-ahead
      maxConcurrentSlabFetches: 3, // Leaves HTTP/1.1 connection headroom for live dump POSTs
      sequentialThresholdReads: 2,
    }),
    'low-memory': Object.freeze({
      slabSize: 1024 * 1024, // 1 MiB slabs
      slabMaxBytes: 16 * 1024 * 1024, // 16 MiB cap
      prefetchAheadSlabs: 1,
      maxConcurrentSlabFetches: 2,
      sequentialThresholdReads: 2,
    }),
  });

  const VIDEO_STREAM_SLAB_PRESET = SLAB_TUNING_PROFILES.video4k;

  function isVideoStreamPath(cleanPath) {
    if (typeof cleanPath !== 'string' || cleanPath.length === 0) {
      return false;
    }
    return VIDEO_STREAM_EXT_RE.test(cleanPath);
  }

  function isWholeFileBufferedPath(cleanPath) {
    if (typeof cleanPath !== 'string' || cleanPath.length === 0) {
      return false;
    }
    const lower = cleanPath.toLowerCase();
    const base = basenameOfPath(lower);
    return (
      base === 'program-manifest.json' ||
      base === 'boot-options.json' ||
      lower.endsWith('.pgim') ||
      lower.endsWith('.reflect.json')
    );
  }

  function isImmutableHttpfsPath(cleanPath) {
    if (typeof cleanPath !== 'string' || cleanPath.length === 0) {
      return false;
    }
    const base = basenameOfPath(cleanPath).toLowerCase();
    if (base === 'microsoftgame.config') {
      return true;
    }
    return IMMUTABLE_HTTPFS_EXT_RE.test(cleanPath);
  }

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

  function utf8ToHex(str) {
    const bytes = utf8Encoder.encode(str);
    let hex = '';
    for (let i = 0; i < bytes.length; i += 1) {
      hex += bytes[i].toString(16).padStart(2, '0');
    }
    return hex;
  }

  function fnv1a128Hex(str) {
    // Deterministic 128-bit four-lane FNV-1a hash matching web/workers/range_cache.js & web/storage/opfs_cache.js.
    const bytes = utf8Encoder.encode(str);
    let h1 = 0x811c9dc5;
    let h2 = 0x811c9dc5 ^ 0x9e3779b9;
    let h3 = 0x811c9dc5 ^ 0x85ebca6b;
    let h4 = 0x811c9dc5 ^ 0xc2b2ae35;
    for (let i = 0; i < bytes.length; i += 1) {
      const b = bytes[i];
      h1 = Math.imul(h1 ^ b, 0x01000193) >>> 0;
      h2 = Math.imul(h2 ^ ((b + i) & 0xff), 0x01000193) >>> 0;
      h3 = Math.imul(h3 ^ (b ^ 0xa5), 0x01000193) >>> 0;
      h4 = Math.imul(h4 ^ ((b + i * 31) & 0xff), 0x01000193) >>> 0;
    }
    return (
      h1.toString(16).padStart(8, '0') +
      h2.toString(16).padStart(8, '0') +
      h3.toString(16).padStart(8, '0') +
      h4.toString(16).padStart(8, '0')
    );
  }

  /**
   * Builds a deterministic, OPFS-filename-safe version fingerprint (`<size>_<etagHash16>`,
   * `<size>`, `e_<etagHash16>`, or sanitized token) matching W262 (`range_cache.js` / `opfs_cache.js`).
   */
  function buildFileFingerprint(sizeOrMeta, etag = null) {
    if (sizeOrMeta === null || sizeOrMeta === undefined) {
      if (typeof etag === 'string' && etag.trim().length > 0) {
        return `e_${fnv1a128Hex(etag.trim()).slice(0, 16)}`;
      }
      return null;
    }
    if (typeof sizeOrMeta === 'string') {
      const trimmed = sizeOrMeta.trim();
      if (!trimmed) return null;
      if (/^[0-9a-zA-Z_-]{1,64}$/.test(trimmed)) {
        return trimmed;
      }
      return `h_${fnv1a128Hex(trimmed).slice(0, 16)}`;
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
      return `${validSize}_${fnv1a128Hex(validTag).slice(0, 16)}`;
    }
    if (validSize !== null) {
      return `${validSize}`;
    }
    if (validTag !== null) {
      return `e_${fnv1a128Hex(validTag).slice(0, 16)}`;
    }
    return null;
  }

  function parseSizeFromFingerprint(fileFingerprint) {
    if (typeof fileFingerprint !== 'string') {
      return null;
    }
    const match = /^(\d+)(?:_[0-9a-f]{16})?$/.exec(fileFingerprint);
    if (!match) {
      return null;
    }
    const parsed = Number(match[1]);
    return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : null;
  }

  /**
   * Normalizes a URL or Xbox/Win32/POSIX path into a clean relative POSIX path for OPFS L2 keying
   * (e.g. `http://127.0.0.1:8080/game/shaders.rpf` -> `game/shaders.rpf`).
   */
  function normalizeOpfsL2Path(urlOrPath) {
    if (typeof urlOrPath !== 'string' || urlOrPath.length === 0) {
      return null;
    }
    if (urlOrPath.includes('\0')) {
      return null;
    }
    let s = urlOrPath.trim();
    if (!s) {
      return null;
    }
    s = s.replace(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^/]*/, '');
    s = s.replace(/[?#].*$/, '');
    s = s.replace(/\\/g, '/');
    if (/^\/\/[?.]\//.test(s)) {
      s = s.slice(4);
    }
    if (/^[A-Za-z]:/.test(s)) {
      s = s.slice(2);
    }
    while (s.startsWith('/') || s.startsWith('./')) {
      s = s.startsWith('/') ? s.slice(1) : s.slice(2);
    }
    if (!s || hasTraversalOrNulSegment(s)) {
      return null;
    }
    const parts = s.split('/');
    const clean = [];
    for (let i = 0; i < parts.length; i += 1) {
      const seg = parts[i];
      if (!seg || seg === '.') {
        continue;
      }
      if (seg === '..') {
        return null;
      }
      clean.push(seg);
    }
    return clean.length > 0 ? clean.join('/') : null;
  }

  function isHotOpfsL2Path(pathOrUrl, allowlist = DEFAULT_HOT_OPFS_RPF_ALLOWLIST) {
    if (allowlist === false || allowlist === null) {
      return false;
    }
    const normPath = normalizeOpfsL2Path(pathOrUrl);
    if (!normPath) {
      return false;
    }
    if (allowlist === true || allowlist === '*') {
      return true;
    }
    if (typeof allowlist === 'function') {
      try {
        return Boolean(allowlist(normPath, pathOrUrl));
      } catch {
        return false;
      }
    }
    const items =
      allowlist instanceof Set
        ? Array.from(allowlist)
        : Array.isArray(allowlist)
          ? allowlist
          : null;
    if (!items || items.length === 0) {
      return false;
    }
    const pathLower = normPath.toLowerCase();
    const baseLower = basenameOfPath(pathLower);
    for (let i = 0; i < items.length; i += 1) {
      const entry = items[i];
      if (entry === '*') {
        return true;
      }
      if (entry instanceof RegExp) {
        if (entry.test(normPath)) {
          return true;
        }
        continue;
      }
      if (typeof entry !== 'string') {
        continue;
      }
      const cleanEntry = entry.trim().replace(/\\/g, '/').replace(/^\/+/, '').toLowerCase();
      if (!cleanEntry) {
        continue;
      }
      if (cleanEntry.includes('/')) {
        if (pathLower === cleanEntry || pathLower.endsWith(`/${cleanEntry}`)) {
          return true;
        }
      } else if (baseLower === cleanEntry) {
        return true;
      }
    }
    return false;
  }

  function opfsL2PathToken(normalizedPath) {
    const canonical =
      typeof normalizedPath === 'string' ? normalizedPath.toLowerCase() : String(normalizedPath || '');
    const encoded = utf8ToHex(canonical);
    return encoded.length <= 128
      ? encoded
      : `${encoded.slice(0, 64)}_${fnv1a128Hex(canonical)}`;
  }

  function opfsL2RangeFilePrefix(normalizedPath, offset, length) {
    return `r_${offset}_${length}_${opfsL2PathToken(normalizedPath)}`;
  }

  function opfsL2RangeFileName(normalizedPath, offset, length, fileFingerprint = null) {
    const prefix = opfsL2RangeFilePrefix(normalizedPath, offset, length);
    const fp = buildFileFingerprint(fileFingerprint);
    return fp ? `${prefix}_fp_${fp}.bin` : `${prefix}.bin`;
  }

  function parseOpfsL2RangeFileMeta(fileName) {
    if (typeof fileName !== 'string' || !fileName.startsWith('r_') || !fileName.endsWith('.bin')) {
      return null;
    }
    const match = /^r_(\d+)_(\d+)_(.+)_fp_([0-9a-zA-Z_-]{1,64})\.bin$/.exec(fileName);
    if (!match) {
      return null;
    }
    return {
      offset: Number(match[1]),
      unitLength: Number(match[2]),
      pathToken: match[3],
      fp: match[4],
    };
  }

  function normalizeDataBase(base) {
    const raw = typeof base === 'string' && base.length > 0 ? base : '/data/';
    return raw.endsWith('/') ? raw : `${raw}/`;
  }

  function normalizeOptionalBase(base) {
    if (typeof base !== 'string') {
      return null;
    }
    const trimmed = base.trim();
    if (!trimmed) {
      return null;
    }
    return trimmed.endsWith('/') ? trimmed : `${trimmed}/`;
  }

  function basenameOfPath(posixPath) {
    const slash = posixPath.lastIndexOf('/');
    return slash < 0 ? posixPath : posixPath.slice(slash + 1);
  }

  function hasTraversalOrNulSegment(pathStr) {
    if (pathStr.includes('\0')) {
      return true;
    }
    const parts = pathStr.replace(/\\/g, '/').split('/');
    for (let i = 0; i < parts.length; i += 1) {
      const seg = parts[i];
      if (seg === '..') {
        return true;
      }
      if (seg.includes('%')) {
        try {
          const decoded = decodeURIComponent(seg);
          if (
            decoded === '..' ||
            decoded.includes('\0') ||
            decoded.includes('/') ||
            decoded.includes('\\')
          ) {
            return true;
          }
        } catch {
          // Ignore malformed percent-escape sequences.
        }
      }
    }
    return false;
  }

  function isDataPinnedPath(cleanPath) {
    const base = basenameOfPath(cleanPath).toLowerCase();
    return base === 'data-manifest.json' || base === 'bootset.json';
  }

  function isImplicitBootPath(cleanPath) {
    const lower = cleanPath.toLowerCase();
    return basenameOfPath(lower) === 'program-manifest.json' || lower.endsWith('.pgim');
  }

  function isGamePackagePath(cleanPath) {
    const base = basenameOfPath(cleanPath).toLowerCase();
    if (base === 'microsoftgame.config') {
      return true;
    }
    return GAME_PACKAGE_EXT_RE.test(cleanPath);
  }

  function hasMalformedPercentEscape(str) {
    if (typeof str !== 'string' || !str.includes('%')) {
      return false;
    }
    for (let i = 0; i < str.length; i += 1) {
      if (str[i] === '%') {
        const hex = str.slice(i + 1, i + 3);
        if (hex.length < 2 || !/^[0-9A-Fa-f]{2}$/.test(hex)) {
          return true;
        }
        i += 2;
      }
    }
    return false;
  }

  function normalizeDumpEndpoint(endpoint) {
    if (typeof endpoint !== 'string') {
      return null;
    }
    const trimmed = endpoint.trim();
    if (!trimmed || trimmed.includes('\0') || /[\x00-\x1f\x7f]/.test(trimmed)) {
      return null;
    }
    // Strip scheme + authority when validating path segments so URLs like
    // `http://127.0.0.1:8080/__pg6_dump/` pass cleanly while `..` segments are rejected.
    const pathPart = trimmed.replace(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^/]*/, '');
    if (hasTraversalOrNulSegment(pathPart)) {
      return null;
    }
    return trimmed.endsWith('/') ? trimmed : `${trimmed}/`;
  }

  function resolveDumpConfig(options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    if (opts.allowDumpWrites === false) {
      return { allowDumpWrites: false, dumpEndpoint: null };
    }
    if (opts.dumpEndpoint !== undefined && opts.dumpEndpoint !== null) {
      const normalized = normalizeDumpEndpoint(opts.dumpEndpoint);
      if (!normalized) {
        return { allowDumpWrites: false, dumpEndpoint: null };
      }
      return { allowDumpWrites: true, dumpEndpoint: normalized };
    }
    if (opts.allowDumpWrites === true) {
      return { allowDumpWrites: true, dumpEndpoint: DEFAULT_DUMP_ENDPOINT };
    }
    return { allowDumpWrites: false, dumpEndpoint: null };
  }

  function parseDumpVirtualPath(rawPath) {
    if (typeof rawPath !== 'string' || rawPath.length === 0) {
      return {
        ok: false,
        errno: -22,
        code: 'INVALID_PATH',
        error: 'Dump path must be a non-empty string',
      };
    }
    if (
      rawPath.includes('\0') ||
      hasTraversalOrNulSegment(rawPath) ||
      hasMalformedPercentEscape(rawPath)
    ) {
      return {
        ok: false,
        errno: -22,
        code: 'PATH_TRAVERSAL',
        error: 'path traversal blocked',
      };
    }

    // Reject URI schemes (e.g. rpf://, http://), Win32 device prefixes (\\?\), and drive prefixes (G:\).
    const slashNormalized = rawPath.replace(/\\/g, '/');
    if (
      /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(slashNormalized) ||
      /^\/\/[?.]\//.test(slashNormalized) ||
      /^[A-Za-z]:/.test(slashNormalized)
    ) {
      return {
        ok: false,
        errno: -30,
        code: 'READ_ONLY',
        error: 'path is read-only (outside /pg6_dump/)',
      };
    }

    let s = slashNormalized;
    while (s.startsWith('/') || s.startsWith('./')) {
      if (s.startsWith('/')) {
        s = s.slice(1);
      } else {
        s = s.slice(2);
      }
    }

    const rawSegments = s.split('/');
    const cleanSegments = [];
    for (let i = 0; i < rawSegments.length; i += 1) {
      const seg = rawSegments[i];
      if (!seg || seg === '.') {
        continue;
      }
      if (seg === '..') {
        return {
          ok: false,
          errno: -22,
          code: 'PATH_TRAVERSAL',
          error: 'path traversal blocked',
        };
      }
      cleanSegments.push(seg);
    }

    if (cleanSegments.length === 0) {
      return {
        ok: false,
        errno: -22,
        code: 'INVALID_PATH',
        error: 'invalid dump path',
      };
    }

    if (cleanSegments[0].toLowerCase() !== 'pg6_dump') {
      return {
        ok: false,
        errno: -30,
        code: 'READ_ONLY',
        error: 'path is read-only (outside /pg6_dump/)',
      };
    }

    const relSegments = cleanSegments.slice(1);
    if (relSegments.length === 0) {
      return {
        ok: false,
        errno: -22,
        code: 'INVALID_PATH',
        error: 'Dump path must include a relative filename under /pg6_dump/',
      };
    }

    for (let i = 0; i < relSegments.length; i += 1) {
      const seg = relSegments[i];
      if (/[\x00-\x1f\x7f:*?"<>|]/.test(seg)) {
        return {
          ok: false,
          errno: -22,
          code: 'INVALID_PATH',
          error: `Invalid character in dump path segment: ${seg}`,
        };
      }
    }

    const relPath = relSegments.join('/');
    const encodedRelPath = relSegments.map((seg) => encodeURIComponent(seg)).join('/');
    return {
      ok: true,
      relPath,
      encodedRelPath,
    };
  }

  function isDumpVirtualPath(rawPath) {
    return parseDumpVirtualPath(rawPath).ok;
  }

  /**
   * Strictly validate a virtual `/pg6_dump/<relpath>` write target against opt-in dump config.
   * Rejects path traversal (`-22`), paths outside `/pg6_dump/` (`-30` EROFS), and disabled writes (`-30`).
   */
  function resolveDumpWriteRoute(rawPath, options = {}) {
    const parsed = parseDumpVirtualPath(rawPath);
    if (!parsed.ok) {
      return parsed;
    }

    const cfg = resolveDumpConfig(options);
    if (!cfg.allowDumpWrites || !cfg.dumpEndpoint) {
      return {
        ok: false,
        errno: -30,
        code: 'DUMP_WRITES_DISABLED',
        error: 'dump writes are disabled (opt-in required)',
      };
    }

    const url = `${cfg.dumpEndpoint}${parsed.encodedRelPath}`;
    return {
      ok: true,
      mount: 'dump',
      base: cfg.dumpEndpoint,
      relPath: parsed.relPath,
      encodedRelPath: parsed.encodedRelPath,
      url,
      appendUrl: `${url}?append=1`,
    };
  }

  function toUint8ArrayPayload(data) {
    if (data instanceof Uint8Array) {
      return data.byteOffset === 0 && data.byteLength === data.buffer.byteLength
        ? data
        : data.slice();
    }
    if (data instanceof ArrayBuffer) {
      return new Uint8Array(data).slice();
    }
    if (ArrayBuffer.isView(data)) {
      return new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
    }
    if (typeof data === 'string') {
      return utf8Encoder.encode(data);
    }
    return null;
  }

  function createSlabCacheTuningProfile(profileOrOptions = 'video4k') {
    const opts =
      typeof profileOrOptions === 'string'
        ? { profile: profileOrOptions }
        : profileOrOptions && typeof profileOrOptions === 'object'
          ? profileOrOptions
          : {};

    const rawProfile =
      typeof opts.profile === 'string'
        ? opts.profile.trim().toLowerCase()
        : typeof opts.slabProfile === 'string'
          ? opts.slabProfile.trim().toLowerCase()
          : null;

    let base = SLAB_TUNING_PROFILES.default;
    if (
      rawProfile === 'fast-boot' ||
      rawProfile === 'fast_boot' ||
      rawProfile === 'high-throughput'
    ) {
      base = SLAB_TUNING_PROFILES['fast-boot'];
    } else if (
      rawProfile === 'video4k' ||
      rawProfile === 'bink2_4k' ||
      rawProfile === 'bink2' ||
      rawProfile === 'video-streaming'
    ) {
      base = SLAB_TUNING_PROFILES.video4k;
    } else if (rawProfile === 'low-memory' || rawProfile === 'low_memory') {
      base = SLAB_TUNING_PROFILES['low-memory'];
    }

    const rawSlabSize =
      opts.slabSize !== undefined
        ? opts.slabSize
        : opts.blockSize !== undefined
          ? opts.blockSize
          : base.slabSize;
    const slabSize =
      Number.isInteger(rawSlabSize) && rawSlabSize >= 0
        ? Math.min(rawSlabSize, MAX_SLAB_SIZE)
        : base.slabSize;

    const rawSlabMaxBytes =
      opts.slabMaxBytes !== undefined
        ? opts.slabMaxBytes
        : opts.maxRamCacheBytes !== undefined
          ? opts.maxRamCacheBytes
          : base.slabMaxBytes;
    const slabMaxBytes =
      Number.isFinite(rawSlabMaxBytes) && rawSlabMaxBytes >= 0
        ? Math.min(Math.floor(rawSlabMaxBytes), MAX_SLAB_CACHE_BYTES)
        : base.slabMaxBytes;

    const rawPrefetch =
      opts.prefetchAheadSlabs !== undefined
        ? opts.prefetchAheadSlabs
        : base.prefetchAheadSlabs;
    const maxSafePrefetch =
      slabSize > 0 && slabMaxBytes >= slabSize * 2
        ? Math.min(MAX_SLAB_PREFETCH_AHEAD, Math.floor(slabMaxBytes / slabSize) - 1)
        : 0;
    const prefetchAheadSlabs =
      Number.isInteger(rawPrefetch) && rawPrefetch >= 0
        ? Math.min(rawPrefetch, maxSafePrefetch)
        : Math.min(base.prefetchAheadSlabs, maxSafePrefetch);

    const rawConc =
      opts.maxConcurrentSlabFetches !== undefined
        ? opts.maxConcurrentSlabFetches
        : base.maxConcurrentSlabFetches;
    const maxConcurrentSlabFetches =
      Number.isInteger(rawConc) && rawConc >= 1
        ? Math.min(rawConc, MAX_CONCURRENT_SLAB_FETCHES)
        : base.maxConcurrentSlabFetches;

    const rawSeqThresh =
      opts.sequentialThresholdReads !== undefined
        ? opts.sequentialThresholdReads
        : base.sequentialThresholdReads;
    const sequentialThresholdReads =
      Number.isInteger(rawSeqThresh) && rawSeqThresh >= 1
        ? Math.min(rawSeqThresh, 32)
        : base.sequentialThresholdReads;

    return {
      slabSize,
      slabMaxBytes,
      prefetchAheadSlabs,
      maxConcurrentSlabFetches,
      sequentialThresholdReads,
    };
  }

  /**
   * Normalize Xbox/Win32/POSIX HTTPFS paths and route them against `bootBase`, `gameBase`,
   * `shadersBase`, or `dataBase` while preserving `rpf://...` virtual URIs and blocking
   * `..` / NUL traversal.
   */
  function resolveHttpfsRoute(rawPath, options = {}) {
    if (typeof rawPath !== 'string' || rawPath.length === 0) {
      return { ok: false, error: 'HTTPFS path must be a non-empty string' };
    }
    if (rawPath.includes('\0')) {
      return { ok: false, error: 'path traversal blocked' };
    }

    const opts = options && typeof options === 'object' ? options : {};
    const dataBase = normalizeDataBase(opts.dataBase);
    const bootBase = normalizeOptionalBase(opts.bootBase);
    const gameBase = normalizeOptionalBase(opts.gameBase);
    const shadersBase = normalizeOptionalBase(opts.shadersBase);

    if (isRpfVirtualPath(rawPath)) {
      const rest = rawPath.slice(RPF_URI_PREFIX.length);
      if (hasTraversalOrNulSegment(rest)) {
        return { ok: false, error: 'path traversal blocked' };
      }
      return {
        ok: true,
        isRpf: true,
        mount: 'rpf',
        base: null,
        relPath: rawPath,
        url: rawPath,
      };
    }

    let s = rawPath.replace(/\\/g, '/');

    // Strip Win32 extended-length / device prefixes (e.g. `\\?\G:\...` or `\\.\G:\...`).
    if (/^\/\/[?.]\//.test(s)) {
      s = s.slice(4);
    }

    // Strip Xbox/Win32 single-letter drive prefixes (e.g. `G:/`, `T:/`, `D:/`, `C:/`, `G:`).
    let hadDrivePrefix = false;
    if (/^[A-Za-z]:/.test(s)) {
      hadDrivePrefix = true;
      s = s.slice(2);
    }

    // Strip leading `/` and `./` prefixes.
    while (s.startsWith('/') || s.startsWith('./')) {
      if (s.startsWith('/')) {
        s = s.slice(1);
      } else {
        s = s.slice(2);
      }
    }

    if (hasTraversalOrNulSegment(s)) {
      return { ok: false, error: 'path traversal blocked' };
    }

    const rawSegments = s.split('/');
    const cleanSegments = [];
    for (let i = 0; i < rawSegments.length; i += 1) {
      const seg = rawSegments[i];
      if (!seg || seg === '.') {
        continue;
      }
      cleanSegments.push(seg);
    }

    if (cleanSegments.length === 0) {
      return { ok: false, error: 'invalid HTTPFS path' };
    }

    const cleanPath = cleanSegments.join('/');
    const lower = cleanPath.toLowerCase();

    // 1. Data-pinned manifests always resolve against dataBase.
    if (isDataPinnedPath(cleanPath)) {
      return {
        ok: true,
        isRpf: false,
        mount: 'data',
        base: dataBase,
        relPath: cleanPath,
        url: `${dataBase}${cleanPath}`,
      };
    }

    // 2. Explicit `boot/...` prefix routes against bootBase (stripping `boot/`) when configured.
    if (bootBase && lower.startsWith('boot/')) {
      const stripped = cleanPath.slice('boot/'.length);
      if (!stripped) {
        return { ok: false, error: 'invalid HTTPFS path' };
      }
      return {
        ok: true,
        isRpf: false,
        mount: 'boot',
        base: bootBase,
        relPath: stripped,
        url: `${bootBase}${stripped}`,
      };
    }

    // 3. Explicit `game/...` mount prefix (without an Xbox/Win32 drive prefix) routes against
    //    gameBase (stripping leading `game/`) when configured.
    if (gameBase && !hadDrivePrefix && lower.startsWith('game/')) {
      const stripped = cleanPath.slice('game/'.length);
      if (!stripped) {
        return { ok: false, error: 'invalid HTTPFS path' };
      }
      return {
        ok: true,
        isRpf: false,
        mount: 'game',
        base: gameBase,
        relPath: stripped,
        url: `${gameBase}${stripped}`,
      };
    }

    // 3b. Explicit `shaders/...` mount prefix (without an Xbox/Win32 drive prefix) routes against
    //     shadersBase (stripping leading `shaders/`) when configured.
    if (shadersBase && !hadDrivePrefix && lower.startsWith('shaders/')) {
      const stripped = cleanPath.slice('shaders/'.length);
      if (!stripped) {
        return { ok: false, error: 'invalid HTTPFS path' };
      }
      return {
        ok: true,
        isRpf: false,
        mount: 'shaders',
        base: shadersBase,
        relPath: stripped,
        url: `${shadersBase}${stripped}`,
      };
    }

    // 4. Implicit boot files (`program-manifest.json`, `*.pgim`) route against bootBase when configured.
    if (bootBase && isImplicitBootPath(cleanPath)) {
      return {
        ok: true,
        isRpf: false,
        mount: 'boot',
        base: bootBase,
        relPath: cleanPath,
        url: `${bootBase}${cleanPath}`,
      };
    }

    // 5. Game package files route against gameBase when configured (falling back to dataBase otherwise).
    if (gameBase && isGamePackagePath(cleanPath)) {
      return {
        ok: true,
        isRpf: false,
        mount: 'game',
        base: gameBase,
        relPath: cleanPath,
        url: `${gameBase}${cleanPath}`,
      };
    }

    // 6. Default: resolve against dataBase.
    return {
      ok: true,
      isRpf: false,
      mount: 'data',
      base: dataBase,
      relPath: cleanPath,
      url: `${dataBase}${cleanPath}`,
    };
  }

  function normalizeRelPath(relPath) {
    if (typeof relPath !== 'string' || relPath.length === 0) {
      throw new Error('HTTPFS path must be a non-empty string');
    }
    if (relPath.includes('\\') || relPath.includes('\0')) {
      throw new Error(`Invalid HTTPFS path: ${relPath}`);
    }
    const stripped = relPath.replace(/^\/+/, '');
    if (!stripped) {
      throw new Error(`Invalid HTTPFS path: ${relPath}`);
    }
    const parts = stripped.split('/');
    for (let i = 0; i < parts.length; i += 1) {
      const part = parts[i];
      if (!part || part === '.' || part === '..') {
        throw new Error(`Invalid HTTPFS path segment in: ${relPath}`);
      }
    }
    return stripped;
  }

  function isRpfVirtualPath(pathStr) {
    return (
      typeof pathStr === 'string' &&
      pathStr.length >= RPF_URI_PREFIX.length &&
      pathStr.slice(0, RPF_URI_PREFIX.length).toLowerCase() === RPF_URI_PREFIX
    );
  }

  function createErrnoError(errno, message) {
    const err = new Error(message);
    err.errno = errno;
    return err;
  }

  function parseRpfVirtualPath(rawPath, options = {}) {
    if (!isRpfVirtualPath(rawPath)) {
      return null;
    }
    if (rawPath.includes('\0')) {
      throw createErrnoError(-22, `Invalid RPF virtual path (NUL byte): ${rawPath}`);
    }

    const rest = rawPath.slice(RPF_URI_PREFIX.length);
    if (!rest || rest.startsWith('/')) {
      throw createErrnoError(-22, `Invalid RPF virtual path: ${rawPath}`);
    }

    const rawSegments = rest.split('/');
    const allowEmptyInner = Boolean(options.allowEmptyInner);

    // Prefer the first segment ending in ".rpf" (case-insensitive) so both
    // `rpf://appdata0.rpf/config/settings.xml` and `rpf://update/update.rpf/common/data.xml`
    // split cleanly on the archive boundary.
    let archiveEndIdx = -1;
    for (let i = 0; i < rawSegments.length; i += 1) {
      if (/\.rpf$/i.test(rawSegments[i])) {
        archiveEndIdx = i;
        break;
      }
    }

    // Fallback: match against manifest paths if provided and no ".rpf" segment was found.
    const manifest = options.manifest || null;
    if (archiveEndIdx === -1 && manifest && Array.isArray(manifest.files)) {
      const manifestSet = new Set();
      for (let i = 0; i < manifest.files.length; i += 1) {
        const entry = manifest.files[i];
        if (entry && typeof entry.path === 'string') {
          try {
            manifestSet.add(normalizeRelPath(entry.path));
          } catch {
            // Ignore malformed manifest entries.
          }
        }
      }
      for (let i = rawSegments.length - 1; i >= 0; i -= 1) {
        const candidate = rawSegments.slice(0, i + 1).join('/');
        if (manifestSet.has(candidate)) {
          archiveEndIdx = i;
          break;
        }
      }
    }

    // Final fallback: first segment is archive path, remaining segments are inner path.
    if (archiveEndIdx === -1) {
      if (rawSegments.length >= 2) {
        archiveEndIdx = 0;
      } else if (allowEmptyInner && rawSegments.length === 1 && rawSegments[0]) {
        archiveEndIdx = 0;
      } else {
        throw createErrnoError(-22, `RPF virtual path missing inner path: ${rawPath}`);
      }
    }

    const archiveRaw = rawSegments.slice(0, archiveEndIdx + 1).join('/');
    let archivePath;
    try {
      archivePath = normalizeRelPath(archiveRaw);
    } catch {
      throw createErrnoError(-22, `Invalid RPF archive path in URI: ${rawPath}`);
    }

    const innerRaw = rawSegments.slice(archiveEndIdx + 1).join('/');
    const rpfMod = resolveRpf9VfsModule(options.rpf9VfsModule);
    let innerPath = '';
    if (innerRaw.length > 0) {
      if (rpfMod && typeof rpfMod.normalizeVirtualPath === 'function') {
        try {
          innerPath = rpfMod.normalizeVirtualPath(innerRaw).path;
        } catch {
          throw createErrnoError(-22, `Invalid RPF inner path in URI: ${rawPath}`);
        }
      } else {
        const parts = innerRaw.replace(/\\/g, '/').split('/');
        const clean = [];
        for (let i = 0; i < parts.length; i += 1) {
          const seg = parts[i];
          if (!seg || seg === '.') continue;
          if (seg === '..') {
            throw createErrnoError(-22, `Path traversal in RPF inner path: ${rawPath}`);
          }
          clean.push(seg);
        }
        innerPath = clean.join('/');
      }
    }

    if (!innerPath && !allowEmptyInner) {
      throw createErrnoError(-22, `RPF virtual path missing inner file path: ${rawPath}`);
    }

    const mountPath = innerPath ? `${archivePath}/${innerPath}` : archivePath;
    return {
      archivePath,
      innerPath,
      mountPath,
    };
  }

  function mapErrorToErrno(err, defaultErrno = -2) {
    if (!err || typeof err !== 'object') {
      return defaultErrno;
    }
    if (typeof err.errno === 'number' && Number.isInteger(err.errno) && err.errno < 0) {
      return err.errno;
    }
    if (err.code === 'ENCRYPTED_ARCHIVE' || err.name === 'EncryptedArchiveError') {
      return -13; // EACCES
    }
    if (err.code === 'PATH_NOT_FOUND') {
      return -2; // ENOENT
    }
    if (err.code === 'NOT_A_FILE') {
      return -21; // EISDIR
    }
    if (err.code === 'NOT_A_DIRECTORY') {
      return -20; // ENOTDIR
    }
    if (err.code === 'INVALID_PATH') {
      return -22; // EINVAL
    }
    if (err.status === 403 || err.statusCode === 403) {
      return -13; // EACCES
    }
    if (err.status === 404 || err.statusCode === 404) {
      return -2; // ENOENT
    }
    const msg = typeof err.message === 'string' ? err.message : '';
    if (/\bHTTP\s+403\b/i.test(msg)) {
      return -13; // EACCES
    }
    if (/\bHTTP\s+404\b/i.test(msg) || /\bnot found\b/i.test(msg)) {
      return -2; // ENOENT
    }
    if (err.name === 'RpfFormatError') {
      return -5; // EIO
    }
    return defaultErrno;
  }

  function getResponseHeader(headers, name) {
    if (!headers) {
      return null;
    }
    const lower = name.toLowerCase();
    if (typeof headers.get === 'function') {
      return headers.get(name) || headers.get(lower) || null;
    }
    if (typeof headers === 'object') {
      if (typeof headers[name] === 'string') return headers[name];
      if (typeof headers[lower] === 'string') return headers[lower];
    }
    return null;
  }

  function parseContentRangeHeader(header) {
    if (typeof header !== 'string') {
      return null;
    }
    const trimmed = header.trim();
    const unsatisfied = /^bytes\s+\*\/(\d+)$/i.exec(trimmed);
    if (unsatisfied) {
      const total = Number(unsatisfied[1]);
      if (!Number.isSafeInteger(total) || total < 0) {
        return null;
      }
      return { start: null, end: null, total, unsatisfied: true };
    }
    const match = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i.exec(trimmed);
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
      (total !== null && (!Number.isSafeInteger(total) || total < 0))
    ) {
      return null;
    }
    return { start, end, total, unsatisfied: false };
  }

  function resolveSlabConfig(options = {}) {
    const hasProfile = typeof options.slabProfile === 'string' && options.slabProfile.trim().length > 0;
    const profileDefaults = hasProfile
      ? createSlabCacheTuningProfile({ profile: options.slabProfile })
      : null;

    const hasValidRecommendedMax =
      Number.isFinite(options.recommendedSlabMaxBytes) && options.recommendedSlabMaxBytes > 0;
    const explicitChunkOrMax =
      options.chunkSize !== undefined || options.maxBytes !== undefined;
    const slabExplicitlyConfigured =
      hasProfile ||
      options.slabSize !== undefined ||
      options.blockSize !== undefined ||
      options.slabMaxBytes !== undefined ||
      options.maxRamCacheBytes !== undefined ||
      hasValidRecommendedMax ||
      options.slabCacheEnabled === true ||
      options.prefetchAheadSlabs !== undefined;

    const slabEnabled =
      options.slabCacheEnabled === false
        ? false
        : slabExplicitlyConfigured
          ? true
          : !explicitChunkOrMax;

    const rawSlabSize =
      options.slabSize !== undefined
        ? options.slabSize
        : options.blockSize !== undefined
          ? options.blockSize
          : profileDefaults
            ? profileDefaults.slabSize
            : DEFAULT_SLAB_SIZE;
    const slabSize = !slabEnabled
      ? 0
      : Number.isInteger(rawSlabSize) && rawSlabSize >= 0
        ? Math.min(rawSlabSize, MAX_SLAB_SIZE)
        : DEFAULT_SLAB_SIZE;

    const rawSlabMaxBytes =
      options.slabMaxBytes !== undefined
        ? options.slabMaxBytes
        : options.maxRamCacheBytes !== undefined
          ? options.maxRamCacheBytes
          : hasValidRecommendedMax
            ? options.recommendedSlabMaxBytes
            : profileDefaults
              ? profileDefaults.slabMaxBytes
              : DEFAULT_SLAB_MAX_BYTES;
    const slabMaxBytes = !slabEnabled
      ? 0
      : Number.isFinite(rawSlabMaxBytes) && rawSlabMaxBytes >= 0
        ? Math.min(Math.floor(rawSlabMaxBytes), MAX_SLAB_CACHE_BYTES)
        : DEFAULT_SLAB_MAX_BYTES;

    const maxSafePrefetch =
      slabEnabled && slabSize > 0 && slabMaxBytes >= slabSize * 2
        ? Math.min(MAX_SLAB_PREFETCH_AHEAD, Math.floor(slabMaxBytes / slabSize) - 1)
        : 0;
    const rawPrefetch =
      options.prefetchAheadSlabs !== undefined
        ? options.prefetchAheadSlabs
        : profileDefaults
          ? profileDefaults.prefetchAheadSlabs
          : slabSize >= DEFAULT_AUTO_PREFETCH_MIN_SLAB_SIZE
            ? DEFAULT_SLAB_PREFETCH_AHEAD
            : 0;
    const prefetchAheadSlabs = !slabEnabled
      ? 0
      : Number.isInteger(rawPrefetch) && rawPrefetch >= 0
        ? Math.min(rawPrefetch, maxSafePrefetch)
        : 0;

    const rawConc =
      options.maxConcurrentSlabFetches !== undefined
        ? options.maxConcurrentSlabFetches
        : profileDefaults
          ? profileDefaults.maxConcurrentSlabFetches
          : DEFAULT_MAX_CONCURRENT_SLAB_FETCHES;
    const maxConcurrentSlabFetches =
      Number.isInteger(rawConc) && rawConc >= 1
        ? Math.min(rawConc, MAX_CONCURRENT_SLAB_FETCHES)
        : DEFAULT_MAX_CONCURRENT_SLAB_FETCHES;

    const rawSeqThresh =
      options.sequentialThresholdReads !== undefined
        ? options.sequentialThresholdReads
        : profileDefaults
          ? profileDefaults.sequentialThresholdReads
          : DEFAULT_SEQUENTIAL_THRESHOLD_READS;
    const sequentialThresholdReads =
      Number.isInteger(rawSeqThresh) && rawSeqThresh >= 1
        ? Math.min(rawSeqThresh, 32)
        : DEFAULT_SEQUENTIAL_THRESHOLD_READS;

    return {
      slabSize,
      slabMaxBytes,
      prefetchAheadSlabs,
      maxConcurrentSlabFetches,
      sequentialThresholdReads,
    };
  }

  function emptyImmutableCacheStats(options = {}) {
    const {
      slabSize,
      slabMaxBytes,
      prefetchAheadSlabs,
      maxConcurrentSlabFetches,
    } = resolveSlabConfig(options);
    return {
      hits: 0,
      misses: 0,
      cacheHits: 0,
      cacheMisses: 0,
      slabCacheHits: 0,
      slabCacheMisses: 0,
      slabEvictions: 0,
      slabNetworkFetches: 0,
      slabBytesCached: 0,
      slabMaxBytes,
      slabSize,
      slabEntries: 0,
      slabPrefetchesScheduled: 0,
      slabPrefetchesCompleted: 0,
      slabPrefetchErrors: 0,
      slabPrefetchWasted: 0,
      slabPrefetchCancelled: 0,
      slabProtectedEntries: 0,
      slabProbationEntries: 0,
      slabProtectedPromotions: 0,
      prefetchHits: 0,
      prefetchAheadSlabs,
      maxConcurrentSlabFetches,
      activeSlabFetches: 0,
      peakConcurrentSlabFetches: 0,
      metaHits: 0,
      metaMisses: 0,
      negativeHits: 0,
      negativeEntries: 0,
      inFlightCoalesced: 0,
      inflightCoalesced: 0,
      networkFetches: 0,
      bytesFetched: 0,
      bytesServedFromCache: 0,
      evictions: 0,
      opfsL2Hits: 0,
      opfsL2Misses: 0,
      opfsL2Writes: 0,
      opfsL2WriteErrors: 0,
      opfsL2WriteDrops: 0,
      opfsL2ProactiveEvictions: 0,
      opfsL2Invalidations: 0,
      opfsL2QueuedBytes: 0,
      memoryReclaims: 0,
      bytesReclaimed: 0,
      bytesCached: 0,
      maxBytes:
        Number.isFinite(options.maxBytes) && options.maxBytes >= 0
          ? Math.floor(options.maxBytes)
          : DEFAULT_IMMUTABLE_MAX_BYTES,
      chunkSize:
        Number.isInteger(options.chunkSize) && options.chunkSize > 0
          ? options.chunkSize
          : DEFAULT_IMMUTABLE_CHUNK_SIZE,
      entries: 0,
      filesIndexed: 0,
    };
  }

  /**
   * Bounded in-memory chunked LRU + sequential read-ahead slab LRU + file-length metadata
   * + short negative cache for immutable HTTPFS files (`*.rpf`, `*.pgim`, `*.dll`, `*.ttpf`,
   * `*.pack`, `*.reflect.json`, `*.bk2`, `*.bik`, `MicrosoftGame.Config`). Eliminates redundant
   * HTTP 206 round-trips across guest worker threads while keeping RAM strictly bounded and
   * returning 100% exact byte slices.
   */
  class ImmutableHttpfsCache {
    constructor(options = {}) {
      const resolvedPerfProfile =
        options.perfProfile !== undefined && options.perfProfile !== null
          ? coerceIoPerfProfile(options.perfProfile)
          : null;
      this._perfProfile = resolvedPerfProfile;
      this._perfProfileName = resolvedPerfProfile ? resolvedPerfProfile.name || null : null;

      this.chunkSize =
        Number.isInteger(options.chunkSize) && options.chunkSize > 0
          ? options.chunkSize
          : DEFAULT_IMMUTABLE_CHUNK_SIZE;
      this.maxBytes =
        Number.isFinite(options.maxBytes) && options.maxBytes >= 0
          ? Math.floor(options.maxBytes)
          : DEFAULT_IMMUTABLE_MAX_BYTES;
      const slabResolveOptions = resolvedPerfProfile
        ? Object.assign({}, options, {
            slabMaxBytes:
              options.slabMaxBytes !== undefined
                ? options.slabMaxBytes
                : options.maxRamCacheBytes !== undefined
                  ? options.maxRamCacheBytes
                  : resolvedPerfProfile.ioSlabCacheBytes,
            maxConcurrentSlabFetches:
              options.maxConcurrentSlabFetches !== undefined
                ? options.maxConcurrentSlabFetches
                : resolvedPerfProfile.ioMaxConcurrentLanes,
          })
        : options;
      const slabCfg = resolveSlabConfig(slabResolveOptions);
      this.slabSize = slabCfg.slabSize;
      this.slabMaxBytes = slabCfg.slabMaxBytes;
      this.prefetchAheadSlabs = slabCfg.prefetchAheadSlabs;
      this.maxConcurrentSlabFetches = slabCfg.maxConcurrentSlabFetches;
      this.sequentialThresholdReads = slabCfg.sequentialThresholdReads;
      this.negativeCacheTtlMs =
        Number.isFinite(options.negativeCacheTtlMs) && options.negativeCacheTtlMs >= 0
          ? Math.floor(options.negativeCacheTtlMs)
          : DEFAULT_NEGATIVE_CACHE_TTL_MS;
      this.negativeCacheMaxEntries =
        Number.isInteger(options.negativeCacheMaxEntries) && options.negativeCacheMaxEntries >= 0
          ? options.negativeCacheMaxEntries
          : DEFAULT_NEGATIVE_CACHE_MAX_ENTRIES;
      this.metadataMaxEntries =
        Number.isInteger(options.metadataMaxEntries) && options.metadataMaxEntries > 0
          ? options.metadataMaxEntries
          : DEFAULT_METADATA_MAX_ENTRIES;
      this.fetchImpl =
        typeof options.fetchImpl === 'function'
          ? options.fetchImpl
          : typeof fetch === 'function'
            ? fetch.bind(globalThis)
            : null;
      this._now = typeof options.now === 'function' ? options.now : () => Date.now();

      // Optional versioned OPFS L2 write-through persistence (W298 / W262).
      this.opfsL2 = options.opfsL2 || null;
      this.opfsCache = options.opfsCache || null;
      this.opfsDir = options.opfsDir || null;
      this.opfsL2Enabled = options.opfsL2Enabled !== false;
      this.opfsL2Allowlist =
        options.opfsL2Allowlist !== undefined
          ? options.opfsL2Allowlist
          : options.hotOpfsAllowlist !== undefined
            ? options.hotOpfsAllowlist
            : options.opfsHotRpfAllowlist !== undefined
              ? options.opfsHotRpfAllowlist
              : DEFAULT_HOT_OPFS_RPF_ALLOWLIST;
      this.opfsL2RequireEtag = Boolean(options.opfsL2RequireEtag);
      this.maxPendingOpfsL2Writes =
        Number.isInteger(options.maxPendingOpfsL2Writes) && options.maxPendingOpfsL2Writes > 0
          ? options.maxPendingOpfsL2Writes
          : DEFAULT_OPFS_L2_MAX_PENDING_WRITES;
      this.maxPendingOpfsL2WriteBytes =
        Number.isFinite(options.maxPendingOpfsL2WriteBytes) &&
        options.maxPendingOpfsL2WriteBytes > 0
          ? Math.floor(options.maxPendingOpfsL2WriteBytes)
          : DEFAULT_OPFS_L2_MAX_PENDING_WRITE_BYTES;
      this.maxOpfsL2WriteConcurrency =
        Number.isInteger(options.maxOpfsL2WriteConcurrency) && options.maxOpfsL2WriteConcurrency > 0
          ? options.maxOpfsL2WriteConcurrency
          : DEFAULT_OPFS_L2_MAX_WRITE_CONCURRENCY;
      this.opfsL2MaxBytes =
        Number.isFinite(options.opfsL2MaxBytes) && options.opfsL2MaxBytes >= 0
          ? Math.floor(options.opfsL2MaxBytes)
          : DEFAULT_OPFS_L2_MAX_BYTES;

      this._chunks = new Map(); // key (`${url}::${chunkIndex}`) -> owned Uint8Array (LRU order)
      this._slabs = new Map(); // key (`${url}::slab::${slabIndex}`) -> owned Uint8Array (master map)
      this._probationSlabs = new Map(); // SLRU probation segment (~25%)
      this._protectedSlabs = new Map(); // SLRU protected segment (~75%)
      this._probationSlabBytes = 0;
      this._protectedSlabBytes = 0;
      this._slabLastSwitch = new Map(); // slabKey -> slabSwitchCount at last access (reuse-distance guard)
      this._lastAccessedSlabKey = null;
      this._slabSwitchCount = 0;
      this._fileMeta = new Map(); // normalized url -> { size: number | null, etag: string | null, fileFingerprint: string | null } (LRU order)
      this._fileFingerprints = new Map(); // normalizedPath -> { fp: string, explicit: boolean, size: number | null, etag: string | null }
      this._negativeCache = new Map(); // normalized url -> { errno, status, expiresAt } (LRU order)
      this._inFlightChunks = new Map(); // chunkKey -> Promise<Uint8Array>
      this._inFlightSlabs = new Map(); // slabKey -> Promise<Uint8Array>
      this._inFlightSlabMeta = new Map(); // slabKey -> { isPrefetch, prefetchHitCounted, waiter, url }
      this._prefetchedSlabKeys = new Set(); // slabKey set for unconsumed prefetched slabs in RAM LRU
      this._urlPrefetchBackoff = new Map(); // url -> { consecutiveWastes, suppressReads }
      this._opfsL2HitBuffers = new WeakSet(); // WeakSet<Uint8Array> tagging OPFS L2 hit buffers for bytesServedFromCache
      this._inFlightOpen = new Map(); // url -> Promise<{ ok, size, errno, status }>
      this._streamTrackers = new Map(); // url -> { lastOffset, lastEndOffset, lastSlabIndex, frontierEndOffset, frontierSlabIndex, sequentialCount }
      this._activeSlabFetches = 0;
      this._activeDemandSlabFetches = 0;
      this._slabWaitQueue = []; // { resolve, priority, url, key, cancelled } (demand=0 before prefetch=1)
      this._pendingPrefetches = new Set();
      this._opfsL2WriteQueue = new Map(); // writeKey -> { normalizedPath, offset, unitLength, bytes, fileFingerprint }
      this._opfsL2QueuedBytes = 0;
      this._activeOpfsL2Writes = new Set(); // Set<Promise>
      this._inFlightOpfsL2WriteKeys = new Set(); // Set<writeKey> preventing concurrent SyncAccessHandle lock collisions
      this._inFlightOpfsL2WritePayloads = new Map(); // writeKey -> Uint8Array (serves hot RAM-evicted reads before disk flush)
      this._opfsL2WriteWaiters = [];
      this._opfsL2Entries = new Map(); // fileName -> { normalizedPath, offset, unitLength, byteLength }
      this._opfsL2TotalBytes = 0;
      this._opfsL2ValidatedFingerprints = new Map(); // normalizedPath.toLowerCase() -> last validated fp
      this._opfsDirKnownFiles = null; // lazily-populated Set<string> of filenames in current opfsDir
      this._opfsDirPathFingerprints = new Map(); // pathToken -> Set<fp> discovered in current opfsDir
      this._opfsDirIndexPromise = null; // coalesced in-flight directory enumeration promise
      this._opfsDirIndexedStore = null; // opfsDir store reference owning _opfsDirKnownFiles
      this._inFlightFingerprintProbes = new Map(); // url -> Promise<string|null> coalesced warm-boot HEAD validation
      this._opfsL2CheckedFingerprints = new Map(); // normalizedPath.toLowerCase() -> last fp checked for stale OPFS dir entries
      this._opfsL2QuotaEvictChain = Promise.resolve();
      this._bytesCached = 0;
      this._slabBytesCached = 0;

      this._stats = {
        chunkHits: 0,
        chunkMisses: 0,
        slabHits: 0,
        slabMisses: 0,
        slabEvictions: 0,
        slabNetworkFetches: 0,
        slabPrefetchesScheduled: 0,
        slabPrefetchesCompleted: 0,
        slabPrefetchErrors: 0,
        slabPrefetchWasted: 0,
        slabPrefetchCancelled: 0,
        slabProtectedPromotions: 0,
        prefetchHits: 0,
        peakConcurrentSlabFetches: 0,
        metaHits: 0,
        metaMisses: 0,
        negativeHits: 0,
        inFlightCoalesced: 0,
        networkFetches: 0,
        bytesFetched: 0,
        bytesServedFromCache: 0,
        evictions: 0,
        opfsL2Hits: 0,
        opfsL2Misses: 0,
        opfsL2Writes: 0,
        opfsL2WriteErrors: 0,
        opfsL2WriteDrops: 0,
        opfsL2ProactiveEvictions: 0,
        opfsL2Invalidations: 0,
        memoryReclaims: 0,
        bytesReclaimed: 0,
      };

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

    _chunkKey(url, chunkIndex) {
      return `${url}::${chunkIndex}`;
    }

    _slabKey(url, slabIndex) {
      return `${url}::slab::${slabIndex}`;
    }

    _isSlabModeEnabled() {
      return this.slabSize > 0 && this.slabMaxBytes >= this.slabSize;
    }

    _isSlruEnabled() {
      const minSlabs = Math.max(SLRU_MIN_SLABS, (this.prefetchAheadSlabs || 0) + 2);
      return this.slabSize > 0 && Math.floor(this.slabMaxBytes / this.slabSize) >= minSlabs;
    }

    _protectedSlabBudgetBytes() {
      const probationReserve = Math.max(1, (this.prefetchAheadSlabs || 0) + 1) * this.slabSize;
      const maxProtected = Math.max(this.slabSize, this.slabMaxBytes - probationReserve);
      return Math.min(maxProtected, Math.max(this.slabSize, Math.floor(this.slabMaxBytes * 0.75)));
    }

    _clearOpfsL2WriteQueue() {
      this._opfsL2WriteQueue.clear();
      this._opfsL2QueuedBytes = 0;
    }

    _trackOpfsL2Entry(fileName, meta) {
      const prev = this._opfsL2Entries.get(fileName);
      if (prev) {
        const prevLen = Number.isFinite(prev.byteLength) && prev.byteLength > 0 ? prev.byteLength : 0;
        this._opfsL2TotalBytes = Math.max(0, this._opfsL2TotalBytes - prevLen);
        this._opfsL2Entries.delete(fileName);
      } else {
        while (this._opfsL2Entries.size >= MAX_OPFS_L2_TRACKED_ENTRIES) {
          const oldestKey = this._opfsL2Entries.keys().next().value;
          if (oldestKey === undefined) break;
          const oldestMeta = this._opfsL2Entries.get(oldestKey);
          this._opfsL2Entries.delete(oldestKey);
          if (oldestMeta && Number.isFinite(oldestMeta.byteLength) && oldestMeta.byteLength > 0) {
            this._opfsL2TotalBytes = Math.max(0, this._opfsL2TotalBytes - oldestMeta.byteLength);
          }
        }
      }
      this._opfsL2Entries.set(fileName, meta);
      const nextLen = meta && Number.isFinite(meta.byteLength) && meta.byteLength > 0 ? meta.byteLength : 0;
      this._opfsL2TotalBytes += nextLen;
    }

    _untrackOpfsL2Entry(fileName) {
      const prev = this._opfsL2Entries.get(fileName);
      if (!prev) return false;
      this._opfsL2Entries.delete(fileName);
      const prevLen = Number.isFinite(prev.byteLength) && prev.byteLength > 0 ? prev.byteLength : 0;
      this._opfsL2TotalBytes = Math.max(0, this._opfsL2TotalBytes - prevLen);
      return true;
    }

    _setFileFingerprintBounded(pathKey, record) {
      if (this._fileFingerprints.has(pathKey)) {
        this._fileFingerprints.delete(pathKey);
      } else {
        while (this._fileFingerprints.size >= MAX_FILE_FINGERPRINTS) {
          const oldest = this._fileFingerprints.keys().next().value;
          if (oldest === undefined) break;
          this._fileFingerprints.delete(oldest);
        }
      }
      this._fileFingerprints.set(pathKey, record);
    }

    _resetOpfsDirIndex() {
      this._opfsDirKnownFiles = null;
      this._opfsDirPathFingerprints.clear();
      this._opfsDirIndexPromise = null;
      this._opfsDirIndexedStore = null;
      this._inFlightFingerprintProbes.clear();
      this._opfsL2CheckedFingerprints.clear();
      this._opfsL2TotalBytes = 0;
    }

    _recordOpfsDirFileName(fileName) {
      if (typeof fileName !== 'string' || fileName.length === 0) {
        return;
      }
      if (this._opfsDirKnownFiles !== null) {
        this._opfsDirKnownFiles.add(fileName);
      }
      const parsed = parseOpfsL2RangeFileMeta(fileName);
      if (parsed && parsed.pathToken && parsed.fp) {
        let fps = this._opfsDirPathFingerprints.get(parsed.pathToken);
        if (!fps) {
          fps = new Set();
          this._opfsDirPathFingerprints.set(parsed.pathToken, fps);
        }
        fps.add(parsed.fp);
      }
    }

    _removeOpfsDirFileName(fileName) {
      if (typeof fileName !== 'string' || fileName.length === 0) {
        return;
      }
      if (this._opfsDirKnownFiles !== null) {
        this._opfsDirKnownFiles.delete(fileName);
      }
      const parsed = parseOpfsL2RangeFileMeta(fileName);
      if (parsed && parsed.pathToken && parsed.fp) {
        const fps = this._opfsDirPathFingerprints.get(parsed.pathToken);
        if (fps && this._opfsDirKnownFiles !== null) {
          let stillHasFp = false;
          const needle = `_${parsed.pathToken}_fp_${parsed.fp}.bin`;
          for (const existing of this._opfsDirKnownFiles) {
            if (existing.endsWith(needle)) {
              stillHasFp = true;
              break;
            }
          }
          if (!stillHasFp) {
            fps.delete(parsed.fp);
            if (fps.size === 0) {
              this._opfsDirPathFingerprints.delete(parsed.pathToken);
            }
          }
        }
      }
    }

    warmOpfsIndex() {
      const backend = this._getOpfsL2Backend();
      if (!backend || backend.kind !== 'opfsDir' || !this._supportsOpfsDirIndex(backend.store)) {
        return Promise.resolve(null);
      }
      return this._ensureOpfsDirIndex(backend.store);
    }

    setOpfsL2(store) {
      const next = store || null;
      if (next !== this.opfsL2) {
        this._clearOpfsL2WriteQueue();
        this._opfsL2Entries.clear();
        this._opfsL2TotalBytes = 0;
        this._opfsL2ValidatedFingerprints.clear();
        this._resetOpfsDirIndex();
      }
      this.opfsL2 = next;
    }

    setOpfsCache(opfsCache) {
      const next = opfsCache || null;
      if (next !== this.opfsCache) {
        this._clearOpfsL2WriteQueue();
        this._opfsL2ValidatedFingerprints.clear();
        this._resetOpfsDirIndex();
      }
      this.opfsCache = next;
    }

    setOpfsDir(opfsDir) {
      const next = opfsDir || null;
      if (next !== this.opfsDir) {
        this._clearOpfsL2WriteQueue();
        this._opfsL2Entries.clear();
        this._opfsL2TotalBytes = 0;
        this._opfsL2ValidatedFingerprints.clear();
        this._resetOpfsDirIndex();
      }
      this.opfsDir = next;
    }

    _getOpfsL2Backend() {
      if (!this.opfsL2Enabled) {
        return null;
      }
      if (this.opfsL2) {
        if (
          typeof this.opfsL2.readRange === 'function' ||
          typeof this.opfsL2.writeRange === 'function'
        ) {
          return { kind: 'range', store: this.opfsL2 };
        }
        if (
          typeof this.opfsL2.getEntry === 'function' &&
          typeof this.opfsL2.put === 'function'
        ) {
          return { kind: 'opfsCache', store: this.opfsL2 };
        }
        if (
          typeof this.opfsL2.getFileHandle === 'function' ||
          typeof this.opfsL2.get === 'function'
        ) {
          return { kind: 'opfsDir', store: this.opfsL2 };
        }
      }
      if (
        this.opfsCache &&
        typeof this.opfsCache.getEntry === 'function' &&
        typeof this.opfsCache.put === 'function'
      ) {
        return { kind: 'opfsCache', store: this.opfsCache };
      }
      if (
        this.opfsDir &&
        (typeof this.opfsDir.getFileHandle === 'function' ||
          typeof this.opfsDir.get === 'function')
      ) {
        return { kind: 'opfsDir', store: this.opfsDir };
      }
      return null;
    }

    _hasOpfsL2ForUrl(url) {
      return Boolean(this._getOpfsL2Backend() && isHotOpfsL2Path(url, this.opfsL2Allowlist));
    }

    _evictUrlFromMemory(url) {
      const prefix = `${url}::`;
      for (const key of Array.from(this._slabs.keys())) {
        if (key.startsWith(prefix)) {
          this._deleteSlabInternal(key, false);
        }
      }
      for (const [key, bytes] of Array.from(this._chunks.entries())) {
        if (key.startsWith(prefix)) {
          this._chunks.delete(key);
          this._bytesCached = Math.max(0, this._bytesCached - bytes.byteLength);
        }
      }
    }

    _evictNormalizedPathFromMemory(normalizedPath) {
      const targetLower = typeof normalizedPath === 'string' ? normalizedPath.toLowerCase() : '';
      for (const key of Array.from(this._slabs.keys())) {
        const sep = key.indexOf('::');
        const keyUrl = sep >= 0 ? key.slice(0, sep) : key;
        const keyNorm = normalizeOpfsL2Path(keyUrl);
        const keyLower = keyNorm ? keyNorm.toLowerCase() : '';
        if (
          keyLower === targetLower ||
          (keyLower && basenameOfPath(keyLower) === targetLower)
        ) {
          this._deleteSlabInternal(key, false);
        }
      }
      for (const [key, bytes] of Array.from(this._chunks.entries())) {
        const sep = key.indexOf('::');
        const keyUrl = sep >= 0 ? key.slice(0, sep) : key;
        const keyNorm = normalizeOpfsL2Path(keyUrl);
        const keyLower = keyNorm ? keyNorm.toLowerCase() : '';
        if (
          keyLower === targetLower ||
          (keyLower && basenameOfPath(keyLower) === targetLower)
        ) {
          this._chunks.delete(key);
          this._bytesCached = Math.max(0, this._bytesCached - bytes.byteLength);
        }
      }
      for (const [wKey, task] of Array.from(this._opfsL2WriteQueue.entries())) {
        const taskLower = task && task.normalizedPath ? task.normalizedPath.toLowerCase() : '';
        if (
          taskLower === targetLower ||
          (taskLower && basenameOfPath(taskLower) === targetLower)
        ) {
          this._opfsL2WriteQueue.delete(wKey);
          if (task && task.bytes instanceof Uint8Array) {
            this._opfsL2QueuedBytes = Math.max(0, this._opfsL2QueuedBytes - task.bytes.byteLength);
          }
        }
      }
    }

    setFileFingerprint(urlOrPath, fingerprintOrMeta) {
      const normPath = normalizeOpfsL2Path(urlOrPath);
      if (!normPath) {
        return null;
      }
      const pathKey = normPath.toLowerCase();
      const nextFp = buildFileFingerprint(fingerprintOrMeta);
      const prevRec = this._fileFingerprints.get(pathKey) || null;
      let metaSize = null;
      let metaEtag = null;
      if (fingerprintOrMeta && typeof fingerprintOrMeta === 'object') {
        const rawSize =
          fingerprintOrMeta.fileSize !== undefined
            ? fingerprintOrMeta.fileSize
            : fingerprintOrMeta.size;
        if (Number.isSafeInteger(rawSize) && rawSize >= 0) {
          metaSize = rawSize;
        }
        if (typeof fingerprintOrMeta.etag === 'string' && fingerprintOrMeta.etag.trim()) {
          metaEtag = fingerprintOrMeta.etag.trim();
        }
      }
      if (metaSize === null && nextFp !== null) {
        metaSize = parseSizeFromFingerprint(nextFp);
      }

      if (nextFp === null) {
        this._fileFingerprints.delete(pathKey);
      } else {
        this._setFileFingerprintBounded(pathKey, {
          fp: nextFp,
          explicit: true,
          size: metaSize,
          etag: metaEtag,
        });
      }

      if (prevRec && prevRec.fp && prevRec.fp !== nextFp) {
        this._evictNormalizedPathFromMemory(normPath);
        for (const [metaUrl, meta] of Array.from(this._fileMeta.entries())) {
          const metaNorm = normalizeOpfsL2Path(metaUrl);
          if (metaNorm && metaNorm.toLowerCase() === pathKey) {
            this._fileMeta.set(metaUrl, {
              size: metaSize !== null ? metaSize : meta.size,
              etag: metaEtag !== null ? metaEtag : null,
              fileFingerprint: nextFp,
            });
          }
        }
        if (nextFp !== null && this._getOpfsL2Backend()) {
          this._enqueueOpfsL2Invalidation(normPath, nextFp, { countIfNoReport: true });
        }
      }
      return nextFp;
    }

    getFileFingerprint(urlOrPath) {
      if (typeof urlOrPath !== 'string' || urlOrPath.length === 0) {
        return null;
      }
      const meta = this._fileMeta.get(urlOrPath);
      if (meta && meta.fileFingerprint) {
        return meta.fileFingerprint;
      }
      const normPath = normalizeOpfsL2Path(urlOrPath);
      if (!normPath) {
        return null;
      }
      const pathKey = normPath.toLowerCase();
      const direct = this._fileFingerprints.get(pathKey);
      if (direct && direct.fp) {
        return direct.fp;
      }
      const base = basenameOfPath(pathKey);
      if (base !== pathKey) {
        const byBase = this._fileFingerprints.get(base);
        if (byBase && byBase.fp) {
          return byBase.fp;
        }
      }
      return null;
    }

    _lookupSeededFingerprintRecord(url) {
      const normPath = normalizeOpfsL2Path(url);
      if (!normPath) {
        return null;
      }
      const pathKey = normPath.toLowerCase();
      const direct = this._fileFingerprints.get(pathKey);
      if (direct) {
        return { normPath, pathKey, record: direct };
      }
      const base = basenameOfPath(pathKey);
      if (base !== pathKey) {
        const byBase = this._fileFingerprints.get(base);
        if (byBase) {
          return { normPath, pathKey, record: byBase };
        }
      }
      return { normPath, pathKey, record: null };
    }

    _touchMeta(url) {
      if (!this._fileMeta.has(url)) {
        return null;
      }
      const meta = this._fileMeta.get(url);
      this._fileMeta.delete(url);
      this._fileMeta.set(url, meta);
      return meta;
    }

    setFileMetadata(url, size, etag = undefined) {
      if (typeof url !== 'string' || url.length === 0) {
        return;
      }
      const cleanSize = Number.isSafeInteger(size) && size >= 0 ? size : null;
      const cleanEtag =
        typeof etag === 'string' && etag.trim().length > 0 ? etag.trim() : null;
      const existing = this._fileMeta.get(url);
      const nextSize = cleanSize !== null ? cleanSize : existing ? existing.size : null;
      const nextEtag =
        etag !== undefined ? cleanEtag : existing ? existing.etag || null : null;
      const seeded = this._lookupSeededFingerprintRecord(url);
      const seededFp = seeded && seeded.record ? seeded.record.fp : null;
      const derivedFp =
        nextEtag !== null
          ? buildFileFingerprint(nextSize, nextEtag)
          : existing && existing.fileFingerprint
            ? existing.fileFingerprint
            : seededFp;

      if (existing) {
        this._fileMeta.delete(url);
      } else if (this._fileMeta.size >= this.metadataMaxEntries) {
        const oldestUrl = this._fileMeta.keys().next().value;
        if (oldestUrl !== undefined) {
          this._fileMeta.delete(oldestUrl);
        }
      }
      this._fileMeta.set(url, {
        size: nextSize,
        etag: nextEtag,
        fileFingerprint: derivedFp || null,
      });
      // A confirmed existing file clears any stale negative entry.
      this._negativeCache.delete(url);
    }

    _recordObservedHttpMeta(url, totalSize, etag) {
      if (typeof url !== 'string' || url.length === 0) {
        return null;
      }
      const validSize = Number.isSafeInteger(totalSize) && totalSize >= 0 ? totalSize : null;
      const validTag = typeof etag === 'string' && etag.trim().length > 0 ? etag.trim() : null;
      const existing = this._fileMeta.get(url) || null;
      const seededLookup = this._lookupSeededFingerprintRecord(url);
      const normPath = seededLookup ? seededLookup.normPath : null;
      const pathKey = seededLookup ? seededLookup.pathKey : null;
      const seededRec = seededLookup ? seededLookup.record : null;

      const prevSize = existing ? existing.size : seededRec ? seededRec.size : null;
      const prevEtag = existing ? existing.etag : seededRec ? seededRec.etag : null;
      const prevFp = existing
        ? existing.fileFingerprint
        : seededRec
          ? seededRec.fp
          : null;
      const hadExplicitFp = Boolean(seededRec && seededRec.explicit);

      const nextSize = validSize !== null ? validSize : prevSize;
      const nextEtag = validTag !== null ? validTag : prevEtag;

      let nextFp = null;
      if (this.opfsL2RequireEtag) {
        nextFp = nextEtag !== null ? buildFileFingerprint(nextSize, nextEtag) : null;
      } else {
        nextFp = buildFileFingerprint(nextSize, nextEtag);
      }

      const sizeChanged = prevSize !== null && validSize !== null && prevSize !== validSize;
      const etagChanged = prevEtag !== null && validTag !== null && prevEtag !== validTag;
      const explicitFpMismatch =
        hadExplicitFp && prevFp !== null && nextFp !== null && prevFp !== nextFp;

      if (existing) {
        this._fileMeta.delete(url);
      } else if (this._fileMeta.size >= this.metadataMaxEntries) {
        const oldestUrl = this._fileMeta.keys().next().value;
        if (oldestUrl !== undefined) {
          this._fileMeta.delete(oldestUrl);
        }
      }
      this._fileMeta.set(url, {
        size: nextSize,
        etag: nextEtag,
        fileFingerprint: nextFp,
      });
      this._negativeCache.delete(url);

      if (pathKey && nextFp !== null) {
        this._setFileFingerprintBounded(pathKey, {
          fp: nextFp,
          explicit: hadExplicitFp && !explicitFpMismatch,
          size: nextSize,
          etag: nextEtag,
        });
      }

      if (sizeChanged || etagChanged || explicitFpMismatch) {
        this._evictUrlFromMemory(url);
        if (normPath && pathKey && nextFp !== null && this._hasOpfsL2ForUrl(url)) {
          this._opfsL2ValidatedFingerprints.set(pathKey, nextFp);
          this._enqueueOpfsL2Invalidation(normPath, nextFp, { countIfNoReport: true });
        }
      } else if (
        normPath &&
        pathKey &&
        nextFp !== null &&
        this._hasOpfsL2ForUrl(url) &&
        this._opfsL2ValidatedFingerprints.get(pathKey) !== nextFp
      ) {
        this._opfsL2ValidatedFingerprints.set(pathKey, nextFp);
        this._enqueueOpfsL2Invalidation(normPath, nextFp, { countIfNoReport: false });
      }

      return nextFp;
    }

    getFileMetadata(url) {
      const meta = this._touchMeta(url);
      return meta
        ? {
            size: meta.size,
            etag: meta.etag || null,
            fileFingerprint: meta.fileFingerprint || null,
          }
        : null;
    }

    getFileSize(url) {
      const meta = this._fileMeta.get(url);
      return meta && Number.isSafeInteger(meta.size) ? meta.size : null;
    }

    // --- Versioned OPFS L2 Persistence (W298 / W262 / W331) ---

    _supportsOpfsDirIndex(store) {
      return Boolean(
        store &&
          typeof store.getFileHandle === 'function' &&
          (typeof store.entries === 'function' || typeof store.keys === 'function'),
      );
    }

    _ensureOpfsDirIndex(store) {
      if (!this._supportsOpfsDirIndex(store)) {
        return Promise.resolve(null);
      }
      if (this._opfsDirIndexedStore !== store) {
        this._opfsDirKnownFiles = null;
        this._opfsDirPathFingerprints.clear();
        this._opfsDirIndexPromise = null;
        this._opfsDirIndexedStore = store;
        this._opfsL2CheckedFingerprints.clear();
      }
      if (this._opfsDirKnownFiles !== null) {
        return Promise.resolve(this._opfsDirKnownFiles);
      }
      if (this._opfsDirIndexPromise !== null) {
        return this._opfsDirIndexPromise;
      }
      const targetStore = store;
      const promise = (async () => {
        const known = new Set();
        const pathFps = new Map();
        const indexName = (name) => {
          known.add(name);
          const parsed = parseOpfsL2RangeFileMeta(name);
          if (parsed && parsed.pathToken && parsed.fp) {
            let fps = pathFps.get(parsed.pathToken);
            if (!fps) {
              fps = new Set();
              pathFps.set(parsed.pathToken, fps);
            }
            fps.add(parsed.fp);
          }
        };
        try {
          if (typeof targetStore.keys === 'function') {
            for await (const name of targetStore.keys()) {
              if (typeof name === 'string') {
                indexName(name);
              }
            }
          } else if (typeof targetStore.entries === 'function') {
            for await (const pair of targetStore.entries()) {
              if (Array.isArray(pair) && typeof pair[0] === 'string') {
                indexName(pair[0]);
              }
            }
          }
        } catch {
          if (this._opfsDirIndexPromise === promise) {
            this._opfsDirIndexPromise = null;
          }
          return null;
        }
        if (this._opfsDirIndexedStore === targetStore && this._opfsDirIndexPromise === promise) {
          for (const fileName of this._opfsL2Entries.keys()) {
            indexName(fileName);
          }
          // Seed tracked OPFS L2 byte accounting from parsed `r_<off>_<unitLength>_...` filenames
          // without touching `handle.getFile()` on warm boot.
          let indexedBytes = 0;
          for (const fileName of known) {
            const existing = this._opfsL2Entries.get(fileName);
            if (existing && Number.isFinite(existing.byteLength) && existing.byteLength > 0) {
              indexedBytes += existing.byteLength;
            } else {
              const parsed = parseOpfsL2RangeFileMeta(fileName);
              if (parsed && Number.isFinite(parsed.unitLength) && parsed.unitLength > 0) {
                indexedBytes += parsed.unitLength;
              }
            }
          }
          this._opfsL2TotalBytes = indexedBytes;
          this._opfsDirKnownFiles = known;
          this._opfsDirPathFingerprints = pathFps;
          this._opfsDirIndexPromise = null;
          return known;
        }
        return this._opfsDirKnownFiles;
      })();
      this._opfsDirIndexPromise = promise;
      return promise;
    }

    async _deleteOpfsDirEntry(store, fileName) {
      if (!store || !fileName) return false;
      if (this._opfsDirKnownFiles !== null && this._opfsDirIndexedStore === store) {
        this._removeOpfsDirFileName(fileName);
      }
      try {
        if (typeof store.removeEntry === 'function') {
          await store.removeEntry(fileName);
          return true;
        }
        if (typeof store.delete === 'function') {
          return Boolean(await store.delete(fileName));
        }
        if (typeof store.remove === 'function') {
          await store.remove(fileName);
          return true;
        }
      } catch {
        // Ignore NotFoundError or unsupported delete failures.
      }
      return false;
    }

    async _listOpfsDirEntryNames(store) {
      if (!store) return [];
      const names = [];
      try {
        if (typeof store.keys === 'function') {
          for await (const name of store.keys()) {
            if (typeof name === 'string') {
              names.push(name);
            }
          }
          return names;
        }
        if (typeof store.entries === 'function') {
          for await (const pair of store.entries()) {
            if (Array.isArray(pair) && typeof pair[0] === 'string') {
              names.push(pair[0]);
            }
          }
          return names;
        }
      } catch {
        // Ignore directory enumeration errors.
      }
      return names;
    }

    async _invalidateStaleOpfsDirPath(store, normalizedPath, expectedFp, specificPrefix = null) {
      if (!store || !normalizedPath) return 0;
      const pathToken = opfsL2PathToken(normalizedPath);
      const expectedSuffix = expectedFp ? `_fp_${expectedFp}.bin` : '.bin';
      const isStaleMatch = (name) => {
        if (typeof name !== 'string' || !name.endsWith('.bin')) {
          return false;
        }
        if (specificPrefix !== null) {
          const activeName = expectedFp
            ? `${specificPrefix}_fp_${expectedFp}.bin`
            : `${specificPrefix}.bin`;
          if (name === activeName) return false;
          return (
            name === `${specificPrefix}.bin` ||
            name.startsWith(`${specificPrefix}_fp_`)
          );
        }
        const belongsToPath =
          name.includes(`_${pathToken}.bin`) || name.includes(`_${pathToken}_fp_`);
        if (!belongsToPath) return false;
        if (expectedFp && name.endsWith(expectedSuffix)) {
          return false;
        }
        return true;
      };

      const staleNames = new Set();
      if (this._supportsOpfsDirIndex(store)) {
        const knownFiles = await this._ensureOpfsDirIndex(store);
        if (knownFiles !== null) {
          if (specificPrefix === null && expectedFp) {
            this._opfsL2CheckedFingerprints.set(normalizedPath.toLowerCase(), expectedFp);
          }
          for (const fileName of this._opfsL2Entries.keys()) {
            if (isStaleMatch(fileName)) {
              staleNames.add(fileName);
            }
          }
          for (const fileName of knownFiles) {
            if (isStaleMatch(fileName)) {
              staleNames.add(fileName);
            }
          }
          for (const staleName of staleNames) {
            this._removeOpfsDirFileName(staleName);
            this._untrackOpfsL2Entry(staleName);
          }
          let invalidated = 0;
          for (const staleName of staleNames) {
            const deleted = await this._deleteOpfsDirEntry(store, staleName);
            if (deleted) {
              invalidated += 1;
            }
          }
          return invalidated;
        }
      }

      for (const fileName of this._opfsL2Entries.keys()) {
        if (isStaleMatch(fileName)) {
          staleNames.add(fileName);
        }
      }
      const dirNames = await this._listOpfsDirEntryNames(store);
      for (let i = 0; i < dirNames.length; i += 1) {
        if (isStaleMatch(dirNames[i])) {
          staleNames.add(dirNames[i]);
        }
      }

      let invalidated = 0;
      for (const staleName of staleNames) {
        this._untrackOpfsL2Entry(staleName);
        const deleted = await this._deleteOpfsDirEntry(store, staleName);
        if (deleted) {
          invalidated += 1;
        }
      }
      return invalidated;
    }

    async _evictOpfsDirForQuota(store, excludeFileName, targetBytes) {
      let freedBytes = 0;
      const evicted = new Set();
      if (excludeFileName) {
        evicted.add(excludeFileName);
      }
      const snapshot = Array.from(this._opfsL2Entries.entries());
      for (let i = 0; i < snapshot.length && freedBytes < targetBytes; i += 1) {
        const [fileName, meta] = snapshot[i];
        if (evicted.has(fileName)) continue;
        this._untrackOpfsL2Entry(fileName);
        evicted.add(fileName);
        await this._deleteOpfsDirEntry(store, fileName);
        freedBytes += meta && Number.isFinite(meta.byteLength) && meta.byteLength > 0
          ? meta.byteLength
          : this.slabSize || this.chunkSize;
      }
      if (freedBytes < targetBytes) {
        const knownFiles = this._supportsOpfsDirIndex(store)
          ? await this._ensureOpfsDirIndex(store)
          : null;
        const dirNames =
          knownFiles !== null
            ? Array.from(knownFiles)
            : await this._listOpfsDirEntryNames(store);
        for (let i = 0; i < dirNames.length && freedBytes < targetBytes; i += 1) {
          const name = dirNames[i];
          if (evicted.has(name) || !/^r_\d+_\d+_.*\.bin$/.test(name)) continue;
          evicted.add(name);
          const wasTracked = this._untrackOpfsL2Entry(name);
          await this._deleteOpfsDirEntry(store, name);
          const parsed = parseOpfsL2RangeFileMeta(name);
          const estBytes =
            parsed && Number.isFinite(parsed.unitLength) && parsed.unitLength > 0
              ? parsed.unitLength
              : this.slabSize || this.chunkSize;
          if (!wasTracked) {
            this._opfsL2TotalBytes = Math.max(0, this._opfsL2TotalBytes - estBytes);
          }
          freedBytes += estBytes;
        }
      }
      return freedBytes;
    }

    _evictOpfsDirForQuotaSerialized(store, excludeFileName, targetBytes) {
      const run = () => this._evictOpfsDirForQuota(store, excludeFileName, targetBytes);
      const next = this._opfsL2QuotaEvictChain.then(run, run);
      this._opfsL2QuotaEvictChain = next.then(
        () => {},
        () => {},
      );
      return next;
    }

    async _writeOpfsDirOnce(store, fileName, bytes) {
      if (typeof store.getFileHandle === 'function') {
        const handle = await store.getFileHandle(fileName, { create: true });
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
            if (this._opfsDirKnownFiles !== null && this._opfsDirIndexedStore === store) {
              this._recordOpfsDirFileName(fileName);
            }
            return true;
          } catch (err) {
            if (isQuotaExceededError(err) || typeof handle.createWritable !== 'function') {
              throw err;
            }
            // Fall back cleanly to async FileSystemWritableFileStream when SyncAccessHandle fails or is locked.
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
                // Ignore abort failures during cleanup.
              }
            }
            throw err;
          }
          if (this._opfsDirKnownFiles !== null && this._opfsDirIndexedStore === store) {
            this._recordOpfsDirFileName(fileName);
          }
          return true;
        }
        return false;
      }
      if (typeof store.set === 'function') {
        await store.set(fileName, bytes.slice());
        return true;
      }
      return false;
    }

    _extractInvalidatedCount(res) {
      if (typeof res === 'number' && Number.isFinite(res) && res > 0) {
        return Math.floor(res);
      }
      if (res === true) {
        return 1;
      }
      if (res && typeof res === 'object') {
        const raw =
          res.invalidatedCount !== undefined
            ? res.invalidatedCount
            : res.invalidated !== undefined
              ? res.invalidated
              : res.removedCount !== undefined
                ? res.removedCount
                : 0;
        if (typeof raw === 'number' && Number.isFinite(raw) && raw > 0) {
          return Math.floor(raw);
        }
        if (raw === true) {
          return 1;
        }
      }
      return 0;
    }

    async _runOpfsL2Invalidation(normalizedPath, expectedFp, countIfNoReport) {
      const backend = this._getOpfsL2Backend();
      if (!backend || !normalizedPath) {
        return 0;
      }
      let count = 0;
      try {
        if (backend.kind === 'range') {
          if (typeof backend.store.invalidateFingerprint === 'function') {
            const res = await backend.store.invalidateFingerprint(normalizedPath, expectedFp);
            count = this._extractInvalidatedCount(res);
          }
        } else if (backend.kind === 'opfsCache') {
          if (typeof backend.store.invalidateStaleFingerprint === 'function') {
            const res = await backend.store.invalidateStaleFingerprint(
              { path: normalizedPath },
              expectedFp,
            );
            count = this._extractInvalidatedCount(res);
          }
        } else if (backend.kind === 'opfsDir') {
          count = await this._invalidateStaleOpfsDirPath(
            backend.store,
            normalizedPath,
            expectedFp,
            null,
          );
        }
      } catch {
        // Degrade gracefully on OPFS invalidation failures.
      }
      if (count > 0) {
        this._stats.opfsL2Invalidations += count;
      } else if (countIfNoReport) {
        this._stats.opfsL2Invalidations += 1;
      }
      return count;
    }

    _enqueueOpfsL2Invalidation(normalizedPath, expectedFp, options = {}) {
      const countIfNoReport = Boolean(options && options.countIfNoReport);
      const taskPromise = this._runOpfsL2Invalidation(normalizedPath, expectedFp, countIfNoReport);
      this._activeOpfsL2Writes.add(taskPromise);
      taskPromise.finally(() => {
        this._activeOpfsL2Writes.delete(taskPromise);
        if (this._activeOpfsL2Writes.size === 0 && this._opfsL2WriteQueue.size === 0) {
          const waiters = this._opfsL2WriteWaiters.splice(0, this._opfsL2WriteWaiters.length);
          for (let i = 0; i < waiters.length; i += 1) {
            waiters[i]();
          }
        }
      });
    }

    _probeOpfsFingerprint(url, normPath) {
      const alreadyKnown = this.getFileFingerprint(url);
      if (alreadyKnown) {
        return Promise.resolve(alreadyKnown);
      }
      if (typeof this.fetchImpl !== 'function') {
        return Promise.resolve(null);
      }
      const existingProbe = this._inFlightFingerprintProbes.get(url);
      if (existingProbe) {
        return existingProbe;
      }
      const probePromise = (async () => {
        try {
          const res = await this.fetchImpl(url, { method: 'HEAD' });
          if (res && res.body && typeof res.body.cancel === 'function') {
            try {
              await res.body.cancel();
            } catch {
              // Ignore body cancel errors on HEAD responses.
            }
          }
          if (!res || !res.ok) {
            return null;
          }
          let etag = getResponseHeader(res.headers, 'ETag');
          const clRaw = getResponseHeader(res.headers, 'Content-Length');
          let headSize = null;
          if (typeof clRaw === 'string' && /^\d+$/.test(clRaw.trim())) {
            const parsed = Number(clRaw.trim());
            if (Number.isSafeInteger(parsed) && parsed >= 0) {
              headSize = parsed;
            }
          }
          // If HEAD returned Content-Length without an ETag while OPFS already holds
          // `<size>_<etagHash16>` entries for this path, do NOT emit a size-only fingerprint
          // that would self-invalidate warm OPFS slabs; probe `GET Range: bytes=0-0` to retrieve ETag.
          if (etag === null && headSize !== null && normPath) {
            const cachedFps = this._opfsDirPathFingerprints.get(opfsL2PathToken(normPath));
            let hasTaggedForSize = false;
            if (cachedFps) {
              const sizePrefix = `${headSize}_`;
              for (const existingFp of cachedFps) {
                if (typeof existingFp === 'string' && existingFp.startsWith(sizePrefix)) {
                  hasTaggedForSize = true;
                  break;
                }
              }
            }
            if (hasTaggedForSize) {
              try {
                const probeRes = await this.fetchImpl(url, {
                  headers: { Range: 'bytes=0-0' },
                });
                if (probeRes && probeRes.body && typeof probeRes.body.cancel === 'function') {
                  try {
                    await probeRes.body.cancel();
                  } catch {
                    // Ignore body cancel errors on Range probe.
                  }
                }
                if (probeRes && (probeRes.ok || probeRes.status === 206)) {
                  const rangeEtag = getResponseHeader(probeRes.headers, 'ETag');
                  if (typeof rangeEtag === 'string' && rangeEtag.trim().length > 0) {
                    etag = rangeEtag.trim();
                  }
                }
              } catch {
                // Fall back to returning null so the first real Range GET supplies ETag without
                // prematurely invalidating warm OPFS files.
              }
              if (etag === null) {
                return null;
              }
            }
          }
          if (etag === null && headSize === null) {
            return null;
          }
          const resolvedFp = this._recordObservedHttpMeta(url, headSize, etag);
          return resolvedFp || this.getFileFingerprint(url) || null;
        } catch {
          // Degrade gracefully when HEAD is unsupported or fails; Range GET will populate metadata.
          return null;
        }
      })();
      this._inFlightFingerprintProbes.set(url, probePromise);
      probePromise.finally(() => {
        if (this._inFlightFingerprintProbes.get(url) === probePromise) {
          this._inFlightFingerprintProbes.delete(url);
        }
      });
      return probePromise;
    }

    async _tryReadOpfsL2(url, offset, unitLength, knownSize) {
      const backend = this._getOpfsL2Backend();
      if (!backend || !isHotOpfsL2Path(url, this.opfsL2Allowlist)) {
        return null;
      }
      const normPath = normalizeOpfsL2Path(url);
      if (!normPath) {
        return null;
      }
      const pathKey = normPath.toLowerCase();
      let fp = this.getFileFingerprint(url);
      if (!fp) {
        if (backend.kind === 'opfsDir' && this._supportsOpfsDirIndex(backend.store)) {
          await this._ensureOpfsDirIndex(backend.store);
          const cachedFps = this._opfsDirPathFingerprints.get(opfsL2PathToken(normPath));
          if (cachedFps && cachedFps.size > 0) {
            fp = await this._probeOpfsFingerprint(url, normPath);
          }
        }
        if (!fp) {
          this._stats.opfsL2Misses += 1;
          return null;
        }
      }

      let rawBytes = null;
      let reportedSize = null;
      let reportedEtag = null;
      let staleInvalidated = false;

      const writeKey = `${pathKey}::${offset}::${unitLength}::${fp}`;
      const queuedTask = this._opfsL2WriteQueue.get(writeKey);
      const pendingBytes =
        queuedTask && queuedTask.bytes instanceof Uint8Array
          ? queuedTask.bytes
          : this._inFlightOpfsL2WritePayloads.get(writeKey) || null;
      if (pendingBytes instanceof Uint8Array && pendingBytes.byteLength > 0) {
        rawBytes = pendingBytes.slice();
      }

      try {
        if (rawBytes !== null) {
          // Served directly from pending/in-flight OPFS write-through buffer.
        } else if (backend.kind === 'range') {
          if (typeof backend.store.readRange !== 'function') {
            this._stats.opfsL2Misses += 1;
            return null;
          }
          const res = await backend.store.readRange(normPath, offset, unitLength, fp);
          if (res instanceof Uint8Array) {
            rawBytes = res.slice();
          } else if (res instanceof ArrayBuffer) {
            rawBytes = new Uint8Array(res.slice(0));
          } else if (res && typeof res === 'object') {
            if (res.invalidated === true || res.stale === true) {
              staleInvalidated = true;
            } else if (res.hit !== false) {
              const payload = res.bytes !== undefined ? res.bytes : res.data;
              if (payload instanceof Uint8Array) {
                rawBytes = payload.slice();
              } else if (payload instanceof ArrayBuffer) {
                rawBytes = new Uint8Array(payload.slice(0));
              }
              if (Number.isSafeInteger(res.size) && res.size >= 0) {
                reportedSize = res.size;
              }
              if (typeof res.etag === 'string' && res.etag.trim()) {
                reportedEtag = res.etag.trim();
              }
            }
          }
        } else if (backend.kind === 'opfsCache') {
          const entry = await backend.store.getEntry(
            { path: normPath, offset, length: unitLength },
            { fileFingerprint: fp },
          );
          if (entry && entry.invalidated) {
            staleInvalidated = true;
          } else if (entry && entry.hit && entry.data instanceof Uint8Array) {
            rawBytes = entry.data.slice();
          }
        } else if (backend.kind === 'opfsDir') {
          const fileName = opfsL2RangeFileName(normPath, offset, unitLength, fp);
          const prefix = opfsL2RangeFilePrefix(normPath, offset, unitLength);
          if (typeof backend.store.getFileHandle === 'function') {
            const knownFiles = this._supportsOpfsDirIndex(backend.store)
              ? await this._ensureOpfsDirIndex(backend.store)
              : null;
            if (knownFiles !== null && !knownFiles.has(fileName)) {
              if (this._opfsL2CheckedFingerprints.get(pathKey) !== fp) {
                const inv = await this._invalidateStaleOpfsDirPath(
                  backend.store,
                  normPath,
                  fp,
                  null,
                );
                if (inv > 0) {
                  this._stats.opfsL2Invalidations += inv;
                }
              }
              this._stats.opfsL2Misses += 1;
              return null;
            }
            let handle = null;
            try {
              handle = await backend.store.getFileHandle(fileName, { create: false });
            } catch (err) {
              if (isNotFoundError(err)) {
                if (knownFiles !== null) {
                  this._removeOpfsDirFileName(fileName);
                  this._untrackOpfsL2Entry(fileName);
                  if (this._opfsL2CheckedFingerprints.get(pathKey) !== fp) {
                    const inv = await this._invalidateStaleOpfsDirPath(
                      backend.store,
                      normPath,
                      fp,
                      null,
                    );
                    if (inv > 0) {
                      this._stats.opfsL2Invalidations += inv;
                    }
                  }
                } else {
                  const inv = await this._invalidateStaleOpfsDirPath(
                    backend.store,
                    normPath,
                    fp,
                    prefix,
                  );
                  if (inv > 0) {
                    this._stats.opfsL2Invalidations += inv;
                  }
                }
                this._stats.opfsL2Misses += 1;
                return null;
              }
              throw err;
            }
            // Prefer FileSystemSyncAccessHandle in DedicatedWorker contexts before getFile()
            // to avoid Blob/File allocation and IPC overhead on warm 2 MiB slab reads.
            if (typeof handle.createSyncAccessHandle === 'function') {
              try {
                const syncHandle = await handle.createSyncAccessHandle();
                try {
                  const sz = typeof syncHandle.getSize === 'function' ? syncHandle.getSize() : 0;
                  if (sz > 0) {
                    const out = new Uint8Array(sz);
                    const readLen = syncHandle.read(out, { at: 0 });
                    rawBytes = readLen === sz ? out : out.slice(0, Math.max(0, readLen));
                  }
                } finally {
                  if (typeof syncHandle.close === 'function') {
                    syncHandle.close();
                  }
                }
              } catch {
                // Fall back to handle.getFile() when SyncAccessHandle is locked or unsupported.
              }
            }
            if (rawBytes === null && typeof handle.getFile === 'function') {
              const file = await handle.getFile();
              const buf = await file.arrayBuffer();
              rawBytes = new Uint8Array(buf);
            }
            if (knownFiles !== null) {
              this._recordOpfsDirFileName(fileName);
            }
            if (this._opfsL2Entries.has(fileName)) {
              const meta = this._opfsL2Entries.get(fileName);
              this._opfsL2Entries.delete(fileName);
              this._opfsL2Entries.set(fileName, meta);
            } else if (rawBytes instanceof Uint8Array) {
              this._trackOpfsL2Entry(fileName, {
                normalizedPath: normPath,
                offset,
                unitLength,
                byteLength: rawBytes.byteLength,
              });
            }
          } else if (typeof backend.store.get === 'function') {
            const val = await backend.store.get(fileName);
            if (val === undefined || val === null) {
              const inv = await this._invalidateStaleOpfsDirPath(
                backend.store,
                normPath,
                fp,
                prefix,
              );
              if (inv > 0) {
                this._stats.opfsL2Invalidations += inv;
              }
              this._stats.opfsL2Misses += 1;
              return null;
            }
            if (val instanceof Uint8Array) {
              rawBytes = val.slice();
            } else if (val instanceof ArrayBuffer) {
              rawBytes = new Uint8Array(val.slice(0));
            }
          }
        }
      } catch {
        // Degrade gracefully on OPFS L2 read errors and fall back to HTTP.
        this._stats.opfsL2Misses += 1;
        return null;
      }

      if (staleInvalidated) {
        this._stats.opfsL2Invalidations += 1;
        this._stats.opfsL2Misses += 1;
        return null;
      }

      if (!(rawBytes instanceof Uint8Array) || rawBytes.byteLength === 0 || rawBytes.byteLength > unitLength) {
        this._stats.opfsL2Misses += 1;
        return null;
      }

      const expectedSize =
        knownSize !== null
          ? knownSize
          : reportedSize !== null
            ? reportedSize
            : parseSizeFromFingerprint(fp);
      if (expectedSize !== null) {
        const expectedLen = Math.max(0, Math.min(unitLength, expectedSize - offset));
        if (rawBytes.byteLength !== expectedLen) {
          this._stats.opfsL2Misses += 1;
          this._enqueueOpfsL2Invalidation(normPath, fp, { countIfNoReport: true });
          return null;
        }
      }

      this._stats.opfsL2Hits += 1;
      this._opfsL2ValidatedFingerprints.set(pathKey, fp);
      this._opfsL2HitBuffers.add(rawBytes);

      if (this.getFileSize(url) === null) {
        if (expectedSize !== null) {
          this.setFileMetadata(url, expectedSize, reportedEtag !== null ? reportedEtag : undefined);
        } else if (offset === 0 && rawBytes.byteLength < unitLength) {
          this.setFileMetadata(url, rawBytes.byteLength, reportedEtag !== null ? reportedEtag : undefined);
        }
      }

      return rawBytes;
    }

    async _writeOpfsL2Task(task) {
      const backend = this._getOpfsL2Backend();
      if (!backend) {
        return false;
      }
      const { normalizedPath, offset, unitLength, bytes, fileFingerprint } = task;
      try {
        if (backend.kind === 'range') {
          if (typeof backend.store.writeRange !== 'function') {
            return false;
          }
          const res = await backend.store.writeRange(
            normalizedPath,
            offset,
            unitLength,
            bytes,
            fileFingerprint,
          );
          if (
            res === false ||
            (res &&
              typeof res === 'object' &&
              (res.stored === false || res.ok === false || res.written === false))
          ) {
            this._stats.opfsL2WriteErrors += 1;
            return false;
          }
          this._stats.opfsL2Writes += 1;
          return true;
        }

        if (backend.kind === 'opfsCache') {
          const res = await backend.store.put(
            { path: normalizedPath, offset, length: unitLength },
            bytes,
            { fileFingerprint },
          );
          if (res && res.stored) {
            this._stats.opfsL2Writes += 1;
            return true;
          }
          this._stats.opfsL2WriteErrors += 1;
          return false;
        }

        if (backend.kind === 'opfsDir') {
          const fileName = opfsL2RangeFileName(normalizedPath, offset, unitLength, fileFingerprint);
          const enforceProactiveDiskBudget = async () => {
            if (this.opfsL2MaxBytes > 0 && this._opfsL2TotalBytes > this.opfsL2MaxBytes) {
              const excess = this._opfsL2TotalBytes - this.opfsL2MaxBytes;
              const freed = await this._evictOpfsDirForQuotaSerialized(
                backend.store,
                fileName,
                excess,
              );
              if (freed > 0) {
                this._stats.opfsL2ProactiveEvictions += 1;
              }
            }
          };
          try {
            const wrote = await this._writeOpfsDirOnce(backend.store, fileName, bytes);
            if (wrote) {
              this._trackOpfsL2Entry(fileName, {
                normalizedPath,
                offset,
                unitLength,
                byteLength: bytes.byteLength,
              });
              this._stats.opfsL2Writes += 1;
              await enforceProactiveDiskBudget();
              return true;
            }
            this._stats.opfsL2WriteErrors += 1;
            return false;
          } catch (err) {
            this._untrackOpfsL2Entry(fileName);
            await this._deleteOpfsDirEntry(backend.store, fileName);
            if (!isQuotaExceededError(err)) {
              this._stats.opfsL2WriteErrors += 1;
              return false;
            }
            const targetBytes = Math.max(
              bytes.byteLength,
              Math.ceil(Math.max(this.slabMaxBytes || this.maxBytes || bytes.byteLength, 1) * 0.25),
            );
            await this._evictOpfsDirForQuotaSerialized(backend.store, fileName, targetBytes);
            try {
              const retryWrote = await this._writeOpfsDirOnce(backend.store, fileName, bytes);
              if (retryWrote) {
                this._trackOpfsL2Entry(fileName, {
                  normalizedPath,
                  offset,
                  unitLength,
                  byteLength: bytes.byteLength,
                });
                this._stats.opfsL2Writes += 1;
                await enforceProactiveDiskBudget();
                return true;
              }
              this._stats.opfsL2WriteErrors += 1;
              return false;
            } catch {
              this._untrackOpfsL2Entry(fileName);
              await this._deleteOpfsDirEntry(backend.store, fileName);
              this._stats.opfsL2WriteErrors += 1;
              return false;
            }
          }
        }
      } catch {
        this._stats.opfsL2WriteErrors += 1;
        return false;
      }
      return false;
    }

    _pumpOpfsL2WriteQueue(options = {}) {
      const force = Boolean(options && options.force);
      // Read-priority gating: defer background OPFS L2 write-through while demand slab fetches
      // are active so disk flushes do not compete with hot HTTP/OPFS demand reads.
      if (!force && this._activeDemandSlabFetches > 0) {
        return;
      }
      while (
        this._activeOpfsL2Writes.size < this.maxOpfsL2WriteConcurrency &&
        this._opfsL2WriteQueue.size > 0
      ) {
        const nextEntry = this._opfsL2WriteQueue.entries().next().value;
        if (!nextEntry) break;
        const [writeKey, task] = nextEntry;
        this._opfsL2WriteQueue.delete(writeKey);
        if (task && task.bytes instanceof Uint8Array) {
          this._opfsL2QueuedBytes = Math.max(0, this._opfsL2QueuedBytes - task.bytes.byteLength);
        }
        if (this._inFlightOpfsL2WriteKeys.has(writeKey)) {
          continue;
        }
        this._inFlightOpfsL2WriteKeys.add(writeKey);
        if (task && task.bytes instanceof Uint8Array) {
          this._inFlightOpfsL2WritePayloads.set(writeKey, task.bytes);
        }

        const writePromise = (async () => {
          try {
            await this._writeOpfsL2Task(task);
          } catch {
            this._stats.opfsL2WriteErrors += 1;
          }
        })();

        this._activeOpfsL2Writes.add(writePromise);
        writePromise.finally(() => {
          this._inFlightOpfsL2WriteKeys.delete(writeKey);
          this._inFlightOpfsL2WritePayloads.delete(writeKey);
          this._activeOpfsL2Writes.delete(writePromise);
          this._pumpOpfsL2WriteQueue({ force });
          if (this._activeOpfsL2Writes.size === 0 && this._opfsL2WriteQueue.size === 0) {
            const waiters = this._opfsL2WriteWaiters.splice(0, this._opfsL2WriteWaiters.length);
            for (let i = 0; i < waiters.length; i += 1) {
              waiters[i]();
            }
          }
        });
      }
    }

    _maybeWriteThroughOpfsL2(url, offset, unitLength, bytes) {
      const backend = this._getOpfsL2Backend();
      if (
        !backend ||
        !isHotOpfsL2Path(url, this.opfsL2Allowlist) ||
        !(bytes instanceof Uint8Array) ||
        bytes.byteLength === 0
      ) {
        return false;
      }
      const normPath = normalizeOpfsL2Path(url);
      if (!normPath) {
        return false;
      }
      const fp = this.getFileFingerprint(url);
      if (!fp) {
        return false;
      }
      const pathKey = normPath.toLowerCase();
      const writeKey = `${pathKey}::${offset}::${unitLength}::${fp}`;
      if (this._inFlightOpfsL2WriteKeys.has(writeKey)) {
        return false;
      }
      if (
        backend.kind === 'opfsDir' &&
        this._opfsDirKnownFiles !== null &&
        this._opfsDirIndexedStore === backend.store
      ) {
        const fileName = opfsL2RangeFileName(normPath, offset, unitLength, fp);
        if (this._opfsDirKnownFiles.has(fileName)) {
          return false;
        }
      }
      if (
        this.maxPendingOpfsL2WriteBytes > 0 &&
        bytes.byteLength > this.maxPendingOpfsL2WriteBytes
      ) {
        this._stats.opfsL2WriteDrops += 1;
        return false;
      }
      const existingTask = this._opfsL2WriteQueue.get(writeKey);
      if (existingTask) {
        this._opfsL2WriteQueue.delete(writeKey);
        if (existingTask.bytes instanceof Uint8Array) {
          this._opfsL2QueuedBytes = Math.max(
            0,
            this._opfsL2QueuedBytes - existingTask.bytes.byteLength,
          );
        }
      } else {
        while (
          this._opfsL2WriteQueue.size >= this.maxPendingOpfsL2Writes ||
          (this.maxPendingOpfsL2WriteBytes > 0 &&
            this._opfsL2QueuedBytes + bytes.byteLength > this.maxPendingOpfsL2WriteBytes &&
            this._opfsL2WriteQueue.size > 0)
        ) {
          const oldestEntry = this._opfsL2WriteQueue.entries().next().value;
          if (!oldestEntry) break;
          const [oldestKey, oldestTask] = oldestEntry;
          this._opfsL2WriteQueue.delete(oldestKey);
          if (oldestTask && oldestTask.bytes instanceof Uint8Array) {
            this._opfsL2QueuedBytes = Math.max(
              0,
              this._opfsL2QueuedBytes - oldestTask.bytes.byteLength,
            );
          }
          this._stats.opfsL2WriteDrops += 1;
        }
      }
      // `bytes` is an immutable owned Uint8Array stored in RAM LRU; avoid a redundant 2 MiB slice copy.
      this._opfsL2WriteQueue.set(writeKey, {
        normalizedPath: normPath,
        offset,
        unitLength,
        bytes,
        fileFingerprint: fp,
      });
      this._opfsL2QueuedBytes += bytes.byteLength;
      this._pumpOpfsL2WriteQueue();
      return true;
    }

    async flushOpfsL2Writes() {
      while (this._activeOpfsL2Writes.size > 0 || this._opfsL2WriteQueue.size > 0) {
        this._pumpOpfsL2WriteQueue({ force: true });
        await new Promise((resolve) => {
          if (this._activeOpfsL2Writes.size === 0 && this._opfsL2WriteQueue.size === 0) {
            resolve();
            return;
          }
          this._opfsL2WriteWaiters.push(resolve);
        });
      }
      const backend = this._getOpfsL2Backend();
      if (backend && backend.store) {
        if (typeof backend.store.flushBackgroundWrites === 'function') {
          await backend.store.flushBackgroundWrites();
        } else if (typeof backend.store.flushWrites === 'function') {
          await backend.store.flushWrites();
        } else if (typeof backend.store.flush === 'function') {
          await backend.store.flush();
        }
      }
    }

    _getNegative(url) {
      const entry = this._negativeCache.get(url);
      if (!entry) {
        return null;
      }
      if (this.negativeCacheTtlMs <= 0 || this._now() >= entry.expiresAt) {
        this._negativeCache.delete(url);
        return null;
      }
      // Refresh LRU position on active negative hit.
      this._negativeCache.delete(url);
      this._negativeCache.set(url, entry);
      return entry;
    }

    _putNegative(url, errno, status) {
      if (this.negativeCacheTtlMs <= 0 || this.negativeCacheMaxEntries <= 0) {
        return;
      }
      if (this._negativeCache.has(url)) {
        this._negativeCache.delete(url);
      } else if (this._negativeCache.size >= this.negativeCacheMaxEntries) {
        const oldestUrl = this._negativeCache.keys().next().value;
        if (oldestUrl !== undefined) {
          this._negativeCache.delete(oldestUrl);
        }
      }
      this._negativeCache.set(url, {
        errno,
        status,
        expiresAt: this._now() + this.negativeCacheTtlMs,
      });
    }

    _touchChunk(key) {
      if (!this._chunks.has(key)) {
        return null;
      }
      const bytes = this._chunks.get(key);
      this._chunks.delete(key);
      this._chunks.set(key, bytes);
      return bytes;
    }

    _putChunk(key, bytes) {
      if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
        return;
      }
      // Always store a tightly-sized owned Uint8Array so evicting one chunk
      // never leaves a larger coalesced ArrayBuffer pinned in memory.
      const owned =
        bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
          ? bytes
          : bytes.slice();

      if (this._chunks.has(key)) {
        const existing = this._chunks.get(key);
        this._bytesCached -= existing.byteLength;
        this._chunks.delete(key);
      }

      if (this.maxBytes <= 0 || owned.byteLength > this.maxBytes) {
        return;
      }

      while (this._bytesCached + owned.byteLength > this.maxBytes && this._chunks.size > 0) {
        const oldestKey = this._chunks.keys().next().value;
        const oldestBytes = this._chunks.get(oldestKey);
        this._chunks.delete(oldestKey);
        this._bytesCached -= oldestBytes.byteLength;
        this._stats.evictions += 1;
      }

      this._chunks.set(key, owned);
      this._bytesCached += owned.byteLength;
    }

    _recordPrefetchWaste(slabKey) {
      const sep = slabKey.indexOf('::slab::');
      if (sep <= 0) return;
      const url = slabKey.slice(0, sep);
      const rec = this._urlPrefetchBackoff.get(url) || { consecutiveWastes: 0, suppressReads: 0 };
      rec.consecutiveWastes = Math.min(rec.consecutiveWastes + 1, 16);
      // After 2 consecutive wasted prefetches on the same URL, back off speculative prefetch briefly.
      if (rec.consecutiveWastes >= 2) {
        rec.suppressReads = Math.min(64, 1 << Math.min(rec.consecutiveWastes, 6));
      }
      this._urlPrefetchBackoff.delete(url);
      if (this._urlPrefetchBackoff.size >= MAX_STREAM_TRACKER_ENTRIES) {
        const oldest = this._urlPrefetchBackoff.keys().next().value;
        if (oldest !== undefined) {
          this._urlPrefetchBackoff.delete(oldest);
        }
      }
      this._urlPrefetchBackoff.set(url, rec);
    }

    _recordPrefetchConsumed(slabKey) {
      const sep = slabKey.indexOf('::slab::');
      if (sep <= 0) return;
      const url = slabKey.slice(0, sep);
      if (this._urlPrefetchBackoff.has(url)) {
        this._urlPrefetchBackoff.delete(url);
      }
    }

    _deleteSlabInternal(key, countEviction = false) {
      const bytes = this._slabs.get(key);
      if (!bytes) {
        return 0;
      }
      const len = bytes.byteLength;
      this._slabs.delete(key);
      this._slabBytesCached = Math.max(0, this._slabBytesCached - len);
      if (this._probationSlabs.delete(key)) {
        this._probationSlabBytes = Math.max(0, this._probationSlabBytes - len);
      }
      if (this._protectedSlabs.delete(key)) {
        this._protectedSlabBytes = Math.max(0, this._protectedSlabBytes - len);
      }
      this._slabLastSwitch.delete(key);
      if (this._prefetchedSlabKeys.delete(key) && countEviction) {
        this._stats.slabPrefetchWasted += 1;
        this._recordPrefetchWaste(key);
      }
      if (countEviction) {
        this._stats.slabEvictions += 1;
      }
      return len;
    }

    _evictOneSlab() {
      let victimKey;
      if (this._isSlruEnabled()) {
        if (this._probationSlabs.size > 0) {
          victimKey = this._probationSlabs.keys().next().value;
        } else if (this._protectedSlabs.size > 0) {
          victimKey = this._protectedSlabs.keys().next().value;
        } else {
          victimKey = this._slabs.keys().next().value;
        }
      } else {
        victimKey = this._slabs.keys().next().value;
      }
      if (victimKey === undefined) {
        return 0;
      }
      return this._deleteSlabInternal(victimKey, true);
    }

    _touchSlab(key) {
      if (!this._slabs.has(key)) {
        return null;
      }
      const bytes = this._slabs.get(key);
      this._slabs.delete(key);
      this._slabs.set(key, bytes);

      const switched = key !== this._lastAccessedSlabKey;
      if (switched) {
        this._slabSwitchCount += 1;
        this._lastAccessedSlabKey = key;
      }

      if (!this._isSlruEnabled()) {
        this._slabLastSwitch.set(key, this._slabSwitchCount);
        return bytes;
      }

      if (this._protectedSlabs.has(key)) {
        this._protectedSlabs.delete(key);
        this._protectedSlabs.set(key, bytes);
        this._slabLastSwitch.set(key, this._slabSwitchCount);
        return bytes;
      }

      if (this._probationSlabs.has(key)) {
        const prevSwitch = this._slabLastSwitch.get(key);
        const isFirstDemandConsume = this._prefetchedSlabKeys.has(key) || prevSwitch === undefined;
        // Require a genuine second demand use with reuse distance >= 2 slab switches before
        // promoting Probation -> Protected. Unconsumed prefetched slabs (first demand hit) and
        // consecutive 64 KiB reads inside the same 2 MiB slab never promote, preventing
        // sequential level/video scans from polluting Protected hot shader slabs.
        if (!isFirstDemandConsume && switched && this._slabSwitchCount - prevSwitch >= 2) {
          this._probationSlabs.delete(key);
          this._probationSlabBytes = Math.max(0, this._probationSlabBytes - bytes.byteLength);
          this._protectedSlabs.set(key, bytes);
          this._protectedSlabBytes += bytes.byteLength;
          this._slabLastSwitch.set(key, this._slabSwitchCount);
          this._stats.slabProtectedPromotions += 1;

          const protectedBudget = this._protectedSlabBudgetBytes();
          while (this._protectedSlabBytes > protectedBudget && this._protectedSlabs.size > 1) {
            const demotedKey = this._protectedSlabs.keys().next().value;
            if (demotedKey === undefined) break;
            const demotedBytes = this._protectedSlabs.get(demotedKey);
            this._protectedSlabs.delete(demotedKey);
            this._protectedSlabBytes = Math.max(
              0,
              this._protectedSlabBytes - demotedBytes.byteLength,
            );
            this._probationSlabs.set(demotedKey, demotedBytes);
            this._probationSlabBytes += demotedBytes.byteLength;
            this._slabLastSwitch.set(demotedKey, this._slabSwitchCount);
          }
        } else {
          this._probationSlabs.delete(key);
          this._probationSlabs.set(key, bytes);
          if (switched || isFirstDemandConsume) {
            this._slabLastSwitch.set(key, this._slabSwitchCount);
          }
        }
      } else {
        // Entry was inserted before SLRU became active; place in Probation.
        this._probationSlabs.set(key, bytes);
        this._probationSlabBytes += bytes.byteLength;
        this._slabLastSwitch.set(key, this._slabSwitchCount);
      }

      return bytes;
    }

    _putSlab(key, bytes, options = {}) {
      if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
        return;
      }
      const isPrefetch = Boolean(options && options.isPrefetch);
      const owned =
        bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
          ? bytes
          : bytes.slice();

      if (this._slabs.has(key)) {
        this._deleteSlabInternal(key, false);
      }

      if (this.slabMaxBytes <= 0 || owned.byteLength > this.slabMaxBytes) {
        return;
      }

      while (
        this._slabBytesCached + owned.byteLength > this.slabMaxBytes &&
        this._slabs.size > 0
      ) {
        const freed = this._evictOneSlab();
        if (freed <= 0) break;
      }

      if (!isPrefetch && key !== this._lastAccessedSlabKey) {
        this._slabSwitchCount += 1;
        this._lastAccessedSlabKey = key;
      }

      this._slabs.set(key, owned);
      this._slabBytesCached += owned.byteLength;
      if (this._isSlruEnabled()) {
        this._probationSlabs.set(key, owned);
        this._probationSlabBytes += owned.byteLength;
      }
      // Only demand-inserted slabs record an initial demand switch epoch; prefetched slabs wait
      // until their first demand read before becoming eligible for Protected promotion.
      if (isPrefetch) {
        this._slabLastSwitch.delete(key);
      } else {
        this._slabLastSwitch.set(key, this._slabSwitchCount);
      }
    }

    trimToBytes(targetSlabBytes, options = {}) {
      const target =
        Number.isFinite(targetSlabBytes) && targetSlabBytes >= 0
          ? Math.floor(targetSlabBytes)
          : 0;
      const dropQueuedOpfsWrites = Boolean(options && options.dropQueuedOpfsWrites);
      const trimChunks = options && options.trimChunks !== false;
      let bytesFreed = 0;
      let slabsEvicted = 0;
      let chunksEvicted = 0;
      let queuedWritesDropped = 0;

      // 1. Evict unconsumed prefetched slabs first (zero cost to active working set).
      if (this._slabBytesCached > target && this._prefetchedSlabKeys.size > 0) {
        for (const key of Array.from(this._prefetchedSlabKeys)) {
          if (this._slabBytesCached <= target) break;
          if (!this._slabs.has(key)) {
            this._prefetchedSlabKeys.delete(key);
            continue;
          }
          const freed = this._deleteSlabInternal(key, true);
          if (freed > 0) {
            bytesFreed += freed;
            slabsEvicted += 1;
          }
        }
      }

      // 2. Evict Probation slabs before Protected slabs (or oldest LRU when SLRU is inactive).
      while (this._slabBytesCached > target && this._slabs.size > 0) {
        const freed = this._evictOneSlab();
        if (freed <= 0) break;
        bytesFreed += freed;
        slabsEvicted += 1;
      }

      // 3. Optionally trim legacy chunk cache when present.
      if (trimChunks && this._chunks.size > 0) {
        const chunkTarget = target === 0 ? 0 : Math.min(this._bytesCached, target);
        while (this._bytesCached > chunkTarget && this._chunks.size > 0) {
          const oldestKey = this._chunks.keys().next().value;
          const oldestBytes = this._chunks.get(oldestKey);
          this._chunks.delete(oldestKey);
          const len = oldestBytes ? oldestBytes.byteLength : 0;
          this._bytesCached = Math.max(0, this._bytesCached - len);
          this._stats.evictions += 1;
          bytesFreed += len;
          chunksEvicted += 1;
        }
      }

      // 4. Optionally drop queued (not yet active) OPFS L2 write payloads under severe pressure.
      if (dropQueuedOpfsWrites && this._opfsL2WriteQueue.size > 0) {
        queuedWritesDropped = this._opfsL2WriteQueue.size;
        bytesFreed += this._opfsL2QueuedBytes;
        this._stats.opfsL2WriteDrops += queuedWritesDropped;
        this._clearOpfsL2WriteQueue();
      }

      if (bytesFreed > 0 || slabsEvicted > 0 || chunksEvicted > 0 || queuedWritesDropped > 0) {
        this._stats.memoryReclaims += 1;
        this._stats.bytesReclaimed += bytesFreed;
      }

      return {
        bytesFreed,
        slabsEvicted,
        chunksEvicted,
        queuedWritesDropped,
        slabBytesCached: this._slabBytesCached,
        bytesCached: this._bytesCached,
        slabSize: this.slabSize,
      };
    }

    reclaimMemory(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const fraction =
        Number.isFinite(opts.fraction) && opts.fraction >= 0 && opts.fraction <= 1
          ? opts.fraction
          : 0.5;
      const targetSlabBytes =
        Number.isFinite(opts.targetSlabBytes) && opts.targetSlabBytes >= 0
          ? Math.floor(opts.targetSlabBytes)
          : Number.isFinite(opts.targetBytes) && opts.targetBytes >= 0
            ? Math.floor(opts.targetBytes)
            : Math.floor(this._slabBytesCached * (1 - fraction));
      return this.trimToBytes(targetSlabBytes, {
        dropQueuedOpfsWrites: Boolean(opts.dropQueuedOpfsWrites || targetSlabBytes === 0),
        trimChunks: opts.trimChunks !== false,
      });
    }

    _makeHttpError(status, url, rangeLabel) {
      const errno = status === 403 ? -13 : status === 404 ? -2 : -5;
      const err = createErrnoError(errno, `HTTP ${status} reading ${url}[${rangeLabel}]`);
      err.status = status;
      return err;
    }

    async _fetchRangeBytes(url, reqStart, reqEnd) {
      if (typeof this.fetchImpl !== 'function') {
        throw createErrnoError(-19, 'ImmutableHttpfsCache has no fetch implementation available');
      }

      this._stats.networkFetches += 1;
      const res = await this.fetchImpl(url, {
        headers: { Range: `bytes=${reqStart}-${reqEnd}` },
      });

      const status = res && typeof res.status === 'number' ? res.status : 0;
      const etag = res ? getResponseHeader(res.headers, 'ETag') : null;

      if (status === 416) {
        const cr = parseContentRangeHeader(getResponseHeader(res.headers, 'Content-Range'));
        if (cr && Number.isSafeInteger(cr.total)) {
          this._recordObservedHttpMeta(url, cr.total, etag);
        } else if (etag) {
          this._recordObservedHttpMeta(url, null, etag);
        }
        return {
          bytes: new Uint8Array(0),
          actualStart: reqStart,
          actualEnd: reqStart - 1,
          knownTotal: cr && Number.isSafeInteger(cr.total) ? cr.total : this.getFileSize(url),
          etag,
          reachedEof: true,
        };
      }

      if (status === 404 || status === 403) {
        const errno = status === 403 ? -13 : -2;
        this._putNegative(url, errno, status);
        throw this._makeHttpError(status, url, `${reqStart}..${reqEnd}`);
      }

      if (!res || (!res.ok && status !== 206)) {
        throw this._makeHttpError(status || 500, url, `${reqStart}..${reqEnd}`);
      }

      const raw = new Uint8Array(await res.arrayBuffer());
      this._stats.bytesFetched += raw.byteLength;

      if (status === 206) {
        const cr = parseContentRangeHeader(getResponseHeader(res.headers, 'Content-Range'));
        if (cr && !cr.unsatisfied) {
          if (cr.total !== null) {
            this._recordObservedHttpMeta(url, cr.total, etag);
          } else {
            this._recordObservedHttpMeta(url, null, etag);
          }
          const expectedLen = cr.end - cr.start + 1;
          const availLen = Math.min(raw.byteLength, expectedLen);
          const actualEnd = cr.start + availLen - 1;
          const reachedEof =
            (cr.total !== null && actualEnd + 1 >= cr.total) || availLen === 0;
          return {
            bytes: availLen === raw.byteLength ? raw : raw.subarray(0, availLen),
            actualStart: cr.start,
            actualEnd,
            knownTotal: cr.total !== null ? cr.total : this.getFileSize(url),
            etag,
            reachedEof,
          };
        }

        // 206 without parseable Content-Range header.
        const requestedLen = reqEnd - reqStart + 1;
        const reachedEof = raw.byteLength < requestedLen;
        if (reachedEof) {
          this._recordObservedHttpMeta(url, reqStart + raw.byteLength, etag);
        } else {
          this._recordObservedHttpMeta(url, null, etag);
        }
        return {
          bytes: raw,
          actualStart: reqStart,
          actualEnd: reqStart + raw.byteLength - 1,
          knownTotal: reachedEof ? reqStart + raw.byteLength : this.getFileSize(url),
          etag,
          reachedEof,
        };
      }

      // 200 OK fallback (server returned full resource instead of 206 Partial Content).
      this._recordObservedHttpMeta(url, raw.byteLength, etag);
      return {
        bytes: raw,
        actualStart: 0,
        actualEnd: raw.byteLength - 1,
        knownTotal: raw.byteLength,
        etag,
        reachedEof: true,
      };
    }

    _fetchChunkRun(url, runStartIdx, runEndIdx) {
      const deferreds = [];
      for (let idx = runStartIdx; idx <= runEndIdx; idx += 1) {
        let resolve;
        let reject;
        const promise = new Promise((res, rej) => {
          resolve = res;
          reject = rej;
        });
        const key = this._chunkKey(url, idx);
        this._inFlightChunks.set(key, promise);
        deferreds.push({ idx, key, promise, resolve, reject });
      }

      const executeRun = async () => {
        try {
          let pending = deferreds;
          if (this._hasOpfsL2ForUrl(url)) {
            const afterL2 = new Array(deferreds.length).fill(null);
            let nextIdx = 0;
            const workerCount = Math.min(OPFS_L2_CHUNK_READ_CONCURRENCY, deferreds.length);
            const runL2Worker = async () => {
              while (nextIdx < deferreds.length) {
                const pos = nextIdx;
                nextIdx += 1;
                const item = deferreds[pos];
                const knownNow = this.getFileSize(url);
                const cStart = item.idx * this.chunkSize;
                if (knownNow !== null && cStart >= knownNow) {
                  item.resolve(new Uint8Array(0));
                  continue;
                }
                const l2Bytes = await this._tryReadOpfsL2(url, cStart, this.chunkSize, knownNow);
                if (l2Bytes !== null) {
                  if (l2Bytes.byteLength > 0) {
                    this._putChunk(item.key, l2Bytes);
                  }
                  item.resolve(l2Bytes);
                } else {
                  afterL2[pos] = item;
                }
              }
            };
            const l2Workers = [];
            for (let w = 0; w < workerCount; w += 1) {
              l2Workers.push(runL2Worker());
            }
            await Promise.all(l2Workers);
            pending = afterL2.filter(Boolean);
            if (pending.length === 0) {
              return;
            }
          }

          // Group remaining missing chunks into contiguous runs for coalesced HTTP 206 fetches.
          const subRuns = [];
          let curSub = [pending[0]];
          for (let i = 1; i < pending.length; i += 1) {
            if (pending[i].idx === pending[i - 1].idx + 1) {
              curSub.push(pending[i]);
            } else {
              subRuns.push(curSub);
              curSub = [pending[i]];
            }
          }
          subRuns.push(curSub);

          for (let r = 0; r < subRuns.length; r += 1) {
            const runItems = subRuns[r];
            const subStartIdx = runItems[0].idx;
            const subEndIdx = runItems[runItems.length - 1].idx;
            const knownSize = this.getFileSize(url);
            const runStart = subStartIdx * this.chunkSize;
            if (knownSize !== null && runStart >= knownSize) {
              for (let i = 0; i < runItems.length; i += 1) {
                runItems[i].resolve(new Uint8Array(0));
              }
              continue;
            }

            const naturalEnd = (subEndIdx + 1) * this.chunkSize - 1;
            const reqEnd =
              knownSize !== null && knownSize > 0
                ? Math.min(naturalEnd, knownSize - 1)
                : naturalEnd;

            const fetched = await this._fetchRangeBytes(url, runStart, reqEnd);
            const totalAfter =
              fetched.knownTotal !== null ? fetched.knownTotal : this.getFileSize(url);
            const uncovered = [];

            for (let i = 0; i < runItems.length; i += 1) {
              const item = runItems[i];
              const cStart = item.idx * this.chunkSize;
              const cMaxEnd = (item.idx + 1) * this.chunkSize - 1;

              if (totalAfter !== null && cStart >= totalAfter) {
                item.resolve(new Uint8Array(0));
                continue;
              }

              const expectedEnd =
                totalAfter !== null ? Math.min(cMaxEnd, totalAfter - 1) : cMaxEnd;

              const startsInResponse =
                fetched.bytes.byteLength > 0 &&
                cStart >= fetched.actualStart &&
                cStart <= fetched.actualEnd;

              if (startsInResponse) {
                const relStart = cStart - fetched.actualStart;
                const availFromChunkStart = fetched.bytes.byteLength - relStart;
                const neededLen = expectedEnd - cStart + 1;
                const coversCompleteChunk =
                  fetched.actualEnd >= expectedEnd ||
                  (fetched.reachedEof && availFromChunkStart > 0);

                if (coversCompleteChunk) {
                  const takeLen = Math.min(this.chunkSize, neededLen, availFromChunkStart);
                  const chunkBytes = fetched.bytes.slice(relStart, relStart + takeLen);
                  if (chunkBytes.byteLength > 0) {
                    this._putChunk(item.key, chunkBytes);
                    this._maybeWriteThroughOpfsL2(url, cStart, this.chunkSize, chunkBytes);
                  }
                  item.resolve(chunkBytes);
                  continue;
                }
              } else if (fetched.reachedEof && cStart > fetched.actualEnd) {
                item.resolve(new Uint8Array(0));
                continue;
              }

              uncovered.push(item);
            }

            // If the server returned a truncated 206 covering only a prefix of the
            // coalesced run, fetch each remaining uncovered chunk individually.
            for (let i = 0; i < uncovered.length; i += 1) {
              const item = uncovered[i];
              const latestSize = this.getFileSize(url);
              const cStart = item.idx * this.chunkSize;
              if (latestSize !== null && cStart >= latestSize) {
                item.resolve(new Uint8Array(0));
                continue;
              }
              const cNaturalEnd = (item.idx + 1) * this.chunkSize - 1;
              const cReqEnd =
                latestSize !== null && latestSize > 0
                  ? Math.min(cNaturalEnd, latestSize - 1)
                  : cNaturalEnd;
              try {
                const single = await this._fetchRangeBytes(url, cStart, cReqEnd);
                if (
                  single.bytes.byteLength === 0 ||
                  cStart < single.actualStart ||
                  cStart > single.actualEnd
                ) {
                  item.resolve(new Uint8Array(0));
                } else {
                  const relStart = cStart - single.actualStart;
                  const takeLen = Math.min(this.chunkSize, single.bytes.byteLength - relStart);
                  const chunkBytes = single.bytes.slice(relStart, relStart + takeLen);
                  if (chunkBytes.byteLength > 0) {
                    this._putChunk(item.key, chunkBytes);
                    this._maybeWriteThroughOpfsL2(url, cStart, this.chunkSize, chunkBytes);
                  }
                  item.resolve(chunkBytes);
                }
              } catch (err) {
                item.reject(err);
              }
            }
          }
        } catch (err) {
          for (let i = 0; i < deferreds.length; i += 1) {
            deferreds[i].reject(err);
          }
        } finally {
          for (let i = 0; i < deferreds.length; i += 1) {
            const item = deferreds[i];
            if (this._inFlightChunks.get(item.key) === item.promise) {
              this._inFlightChunks.delete(item.key);
            }
          }
        }
      };

      executeRun();
      return deferreds.map((d) => d.promise);
    }

    _getChunkRange(url, firstChunk, lastChunk) {
      const totalChunks = lastChunk - firstChunk + 1;
      const chunkPromises = new Array(totalChunks);
      const maxRunChunks = Math.max(1, Math.floor(MAX_COALESCED_FETCH_BYTES / this.chunkSize));

      let runStartPos = -1;
      const flushMissingRun = (endPosExclusive) => {
        if (runStartPos < 0) {
          return;
        }
        let curPos = runStartPos;
        while (curPos < endPosExclusive) {
          const batchEndExclusive = Math.min(endPosExclusive, curPos + maxRunChunks);
          const runStartIdx = firstChunk + curPos;
          const runEndIdx = firstChunk + batchEndExclusive - 1;
          const promises = this._fetchChunkRun(url, runStartIdx, runEndIdx);
          for (let k = 0; k < promises.length; k += 1) {
            chunkPromises[curPos + k] = promises[k];
          }
          curPos = batchEndExclusive;
        }
        runStartPos = -1;
      };

      for (let i = 0; i < totalChunks; i += 1) {
        const chunkIndex = firstChunk + i;
        const key = this._chunkKey(url, chunkIndex);

        const memHit = this._touchChunk(key);
        if (memHit !== null) {
          flushMissingRun(i);
          this._stats.chunkHits += 1;
          chunkPromises[i] = Promise.resolve(memHit);
          continue;
        }

        const inFlight = this._inFlightChunks.get(key);
        if (inFlight) {
          flushMissingRun(i);
          this._stats.inFlightCoalesced += 1;
          chunkPromises[i] = inFlight;
          continue;
        }

        this._stats.chunkMisses += 1;
        if (runStartPos < 0) {
          runStartPos = i;
        }
      }
      flushMissingRun(totalChunks);

      return Promise.all(chunkPromises);
    }

    async open(url) {
      const existingMeta = this._touchMeta(url);
      if (existingMeta !== null) {
        this._stats.metaHits += 1;
        return { ok: true, size: existingMeta.size };
      }

      const neg = this._getNegative(url);
      if (neg !== null) {
        this._stats.negativeHits += 1;
        return { ok: false, errno: neg.errno, status: neg.status };
      }

      const existingOpen = this._inFlightOpen.get(url);
      if (existingOpen) {
        this._stats.inFlightCoalesced += 1;
        return existingOpen;
      }

      this._stats.metaMisses += 1;
      const openPromise = (async () => {
        try {
          if (this._isSlabModeEnabled()) {
            await this._getSlabRange(url, 0, 0);
          } else {
            await this._getChunkRange(url, 0, 0);
          }
          const meta = this._fileMeta.get(url);
          if (!meta) {
            this.setFileMetadata(url, null);
          }
          const finalMeta = this._fileMeta.get(url);
          return { ok: true, size: finalMeta ? finalMeta.size : null };
        } catch (err) {
          return {
            ok: false,
            errno: mapErrorToErrno(err, -2),
            status: err && typeof err.status === 'number' ? err.status : null,
          };
        }
      })();

      this._inFlightOpen.set(url, openPromise);
      try {
        return await openPromise;
      } finally {
        if (this._inFlightOpen.get(url) === openPromise) {
          this._inFlightOpen.delete(url);
        }
      }
    }

    _acquireSlabSlot(isPrefetch = false, meta = null) {
      const limit = Math.max(1, this.maxConcurrentSlabFetches || DEFAULT_MAX_CONCURRENT_SLAB_FETCHES);
      if (this._activeSlabFetches < limit) {
        this._activeSlabFetches += 1;
        if (this._activeSlabFetches > this._stats.peakConcurrentSlabFetches) {
          this._stats.peakConcurrentSlabFetches = this._activeSlabFetches;
        }
        return Promise.resolve(true);
      }
      return new Promise((resolve) => {
        const waiter = {
          resolve,
          priority: isPrefetch ? 1 : 0,
          url: meta ? meta.url : null,
          key: meta ? meta.key : null,
          cancelled: false,
        };
        if (meta) {
          meta.waiter = waiter;
        }
        if (waiter.priority === 0) {
          // Insert demand reads ahead of queued background prefetch waiters.
          let insertAt = this._slabWaitQueue.length;
          for (let i = 0; i < this._slabWaitQueue.length; i += 1) {
            if (this._slabWaitQueue[i].priority > 0) {
              insertAt = i;
              break;
            }
          }
          this._slabWaitQueue.splice(insertAt, 0, waiter);
        } else {
          this._slabWaitQueue.push(waiter);
        }
      });
    }

    _countQueuedPrefetchWaiters() {
      let count = 0;
      for (let i = 0; i < this._slabWaitQueue.length; i += 1) {
        if (this._slabWaitQueue[i].priority > 0) {
          count += 1;
        }
      }
      return count;
    }

    _cancelQueuedPrefetchesForUrl(url) {
      if (!url || this._slabWaitQueue.length === 0) {
        return 0;
      }
      let cancelled = 0;
      for (let i = this._slabWaitQueue.length - 1; i >= 0; i -= 1) {
        const w = this._slabWaitQueue[i];
        if (w.priority > 0 && w.url === url) {
          this._slabWaitQueue.splice(i, 1);
          w.cancelled = true;
          cancelled += 1;
          this._stats.slabPrefetchCancelled += 1;
          w.resolve(false);
        }
      }
      return cancelled;
    }

    _promoteSlabWaiter(waiter) {
      if (!waiter || waiter.priority === 0 || waiter.cancelled) {
        return;
      }
      const idx = this._slabWaitQueue.indexOf(waiter);
      if (idx < 0) {
        return;
      }
      waiter.priority = 0;
      this._slabWaitQueue.splice(idx, 1);
      let insertAt = this._slabWaitQueue.length;
      for (let i = 0; i < this._slabWaitQueue.length; i += 1) {
        if (this._slabWaitQueue[i].priority > 0) {
          insertAt = i;
          break;
        }
      }
      this._slabWaitQueue.splice(insertAt, 0, waiter);
    }

    _releaseSlabSlot() {
      const limit = Math.max(1, this.maxConcurrentSlabFetches || DEFAULT_MAX_CONCURRENT_SLAB_FETCHES);
      if (this._slabWaitQueue.length > 0 && this._activeSlabFetches <= limit) {
        const next = this._slabWaitQueue.shift();
        next.resolve(true);
        return;
      }
      if (this._activeSlabFetches > 0) {
        this._activeSlabFetches -= 1;
      }
      while (this._slabWaitQueue.length > 0 && this._activeSlabFetches < limit) {
        this._activeSlabFetches += 1;
        if (this._activeSlabFetches > this._stats.peakConcurrentSlabFetches) {
          this._stats.peakConcurrentSlabFetches = this._activeSlabFetches;
        }
        const next = this._slabWaitQueue.shift();
        next.resolve(true);
      }
    }

    _fetchSlab(url, slabIndex, options = {}) {
      const isPrefetch = Boolean(options && options.isPrefetch);
      const key = this._slabKey(url, slabIndex);
      const existing = this._inFlightSlabs.get(key);
      if (existing) {
        this._stats.inFlightCoalesced += 1;
        if (!isPrefetch) {
          const existingMeta = this._inFlightSlabMeta.get(key);
          if (existingMeta && existingMeta.isPrefetch) {
            if (!existingMeta.prefetchHitCounted) {
              existingMeta.prefetchHitCounted = true;
              this._stats.prefetchHits += 1;
              this._recordPrefetchConsumed(key);
            }
            if (existingMeta.waiter) {
              existingMeta.promoted = true;
              this._promoteSlabWaiter(existingMeta.waiter);
            }
          }
        }
        return existing;
      }

      const meta = { isPrefetch, prefetchHitCounted: false, promoted: false, waiter: null, url, key };

      const promise = (async () => {
        const slabStart = slabIndex * this.slabSize;
        const knownSizeBefore = this.getFileSize(url);
        if (knownSizeBefore !== null && slabStart >= knownSizeBefore) {
          return new Uint8Array(0);
        }

        if (this._hasOpfsL2ForUrl(url)) {
          const l2Hit = await this._tryReadOpfsL2(url, slabStart, this.slabSize, knownSizeBefore);
          if (l2Hit !== null) {
            this._putSlab(key, l2Hit, { isPrefetch: meta.isPrefetch && !meta.prefetchHitCounted });
            if (meta.isPrefetch && !meta.prefetchHitCounted && this._slabs.has(key)) {
              this._prefetchedSlabKeys.add(key);
            }
            return l2Hit;
          }
        }

        const acquired = await this._acquireSlabSlot(isPrefetch, meta);
        meta.waiter = null;
        if (!acquired) {
          // Prefetch waiter was cancelled before obtaining a network slot (e.g. distant random seek).
          return new Uint8Array(0);
        }
        const countedDemand = !meta.isPrefetch || meta.promoted;
        if (countedDemand) {
          this._activeDemandSlabFetches += 1;
        }
        try {
          const knownSize = this.getFileSize(url);
          if (knownSize !== null && slabStart >= knownSize) {
            return new Uint8Array(0);
          }

          const naturalEnd = (slabIndex + 1) * this.slabSize - 1;
          const reqEnd =
            knownSize !== null && knownSize > 0
              ? Math.min(naturalEnd, knownSize - 1)
              : naturalEnd;

          this._stats.slabNetworkFetches += 1;
          const first = await this._fetchRangeBytes(url, slabStart, reqEnd);
          const totalAfter =
            first.knownTotal !== null ? first.knownTotal : this.getFileSize(url);

          if (totalAfter !== null && slabStart >= totalAfter) {
            return new Uint8Array(0);
          }

          const expectedEnd =
            totalAfter !== null ? Math.min(naturalEnd, totalAfter - 1) : naturalEnd;
          const expectedLen = expectedEnd - slabStart + 1;

          if (
            first.bytes.byteLength > 0 &&
            slabStart >= first.actualStart &&
            slabStart <= first.actualEnd
          ) {
            const relStart = slabStart - first.actualStart;
            const avail = first.bytes.byteLength - relStart;
            const coversComplete =
              first.actualEnd >= expectedEnd || (first.reachedEof && avail > 0);
            if (coversComplete) {
              const takeLen = Math.min(this.slabSize, expectedLen, avail);
              const slabBytes =
                relStart === 0 &&
                takeLen === first.bytes.byteLength &&
                first.bytes.byteOffset === 0 &&
                first.bytes.byteLength === first.bytes.buffer.byteLength
                  ? first.bytes
                  : first.bytes.slice(relStart, relStart + takeLen);
              if (slabBytes.byteLength > 0) {
                this._putSlab(key, slabBytes, {
                  isPrefetch: meta.isPrefetch && !meta.prefetchHitCounted,
                });
                if (meta.isPrefetch && !meta.prefetchHitCounted && this._slabs.has(key)) {
                  this._prefetchedSlabKeys.add(key);
                }
                this._maybeWriteThroughOpfsL2(url, slabStart, this.slabSize, slabBytes);
              }
              return slabBytes;
            }

            // Server returned a truncated 206 prefix of the slab; continue fetching remaining tail.
            const parts = [first.bytes.slice(relStart, relStart + avail)];
            let filled = avail;
            let cursor = slabStart + avail;
            const maxContinuations = 64;
            for (let step = 0; step < maxContinuations && cursor <= expectedEnd; step += 1) {
              const latestSize = this.getFileSize(url);
              const curTargetEnd =
                latestSize !== null ? Math.min(naturalEnd, latestSize - 1) : expectedEnd;
              if (cursor > curTargetEnd) {
                break;
              }
              this._stats.slabNetworkFetches += 1;
              const next = await this._fetchRangeBytes(url, cursor, curTargetEnd);
              if (
                next.bytes.byteLength === 0 ||
                cursor < next.actualStart ||
                cursor > next.actualEnd
              ) {
                break;
              }
              const nRel = cursor - next.actualStart;
              const nAvail = Math.min(curTargetEnd - cursor + 1, next.bytes.byteLength - nRel);
              if (nAvail <= 0) {
                break;
              }
              parts.push(next.bytes.slice(nRel, nRel + nAvail));
              filled += nAvail;
              cursor += nAvail;
              if (next.reachedEof) {
                break;
              }
            }

            const combined = new Uint8Array(filled);
            let pos = 0;
            for (let i = 0; i < parts.length; i += 1) {
              combined.set(parts[i], pos);
              pos += parts[i].byteLength;
            }
            if (combined.byteLength > 0) {
              this._putSlab(key, combined, {
                isPrefetch: meta.isPrefetch && !meta.prefetchHitCounted,
              });
              if (meta.isPrefetch && !meta.prefetchHitCounted && this._slabs.has(key)) {
                this._prefetchedSlabKeys.add(key);
              }
              this._maybeWriteThroughOpfsL2(url, slabStart, this.slabSize, combined);
            }
            return combined;
          }

          return new Uint8Array(0);
        } finally {
          meta.waiter = null;
          if (countedDemand) {
            this._activeDemandSlabFetches = Math.max(0, this._activeDemandSlabFetches - 1);
          }
          this._releaseSlabSlot();
          if (this._activeDemandSlabFetches === 0 && this._opfsL2WriteQueue.size > 0) {
            this._pumpOpfsL2WriteQueue();
          }
        }
      })();

      this._inFlightSlabs.set(key, promise);
      this._inFlightSlabMeta.set(key, meta);
      const cleanup = () => {
        if (this._inFlightSlabs.get(key) === promise) {
          this._inFlightSlabs.delete(key);
        }
        if (this._inFlightSlabMeta.get(key) === meta) {
          this._inFlightSlabMeta.delete(key);
        }
      };
      promise.then(cleanup, cleanup);
      return promise;
    }

    _recordSequentialAccess(url, offset, length, lastSlab) {
      if (this.prefetchAheadSlabs <= 0 || this.slabSize <= 0 || this.slabMaxBytes < this.slabSize * 2) {
        return 0;
      }
      const endOffset = offset + length;
      const maxBackwardJitter = Math.min(8 * 64 * 1024, this.slabSize);
      const maxForwardGap = Math.min(256 * 1024, this.slabSize);
      let tracker = this._streamTrackers.get(url);
      if (tracker) {
        this._streamTrackers.delete(url);
        const frontierEnd = tracker.frontierEndOffset;
        const withinFrontierWindow =
          offset >= Math.max(0, frontierEnd - maxBackwardJitter) &&
          offset <= frontierEnd + maxForwardGap;
        const advancesFrontier = endOffset > frontierEnd || lastSlab > tracker.frontierSlabIndex;
        if (withinFrontierWindow) {
          if (advancesFrontier || (offset !== tracker.lastOffset && offset >= frontierEnd - maxBackwardJitter)) {
            tracker.sequentialCount = Math.min(tracker.sequentialCount + 1, 1024);
          }
          if (endOffset > tracker.frontierEndOffset) {
            tracker.frontierEndOffset = endOffset;
          }
          if (lastSlab > tracker.frontierSlabIndex) {
            tracker.frontierSlabIndex = lastSlab;
          }
        } else {
          // Distant random seek: reset sequential streak and cancel still-queued prefetch waiters
          // for this URL so they do not consume network slots or evict hot slabs.
          tracker.sequentialCount = 1;
          tracker.frontierEndOffset = endOffset;
          tracker.frontierSlabIndex = lastSlab;
          this._cancelQueuedPrefetchesForUrl(url);
        }
        tracker.lastOffset = offset;
        tracker.lastEndOffset = endOffset;
        tracker.lastSlabIndex = lastSlab;
      } else {
        if (this._streamTrackers.size >= MAX_STREAM_TRACKER_ENTRIES) {
          const oldest = this._streamTrackers.keys().next().value;
          if (oldest !== undefined) {
            this._streamTrackers.delete(oldest);
          }
        }
        tracker = {
          lastOffset: offset,
          lastEndOffset: endOffset,
          lastSlabIndex: lastSlab,
          frontierEndOffset: endOffset,
          frontierSlabIndex: lastSlab,
          sequentialCount: 1,
        };
      }
      this._streamTrackers.set(url, tracker);

      if (tracker.sequentialCount >= this.sequentialThresholdReads) {
        const backoff = this._urlPrefetchBackoff.get(url);
        if (backoff && backoff.suppressReads > 0) {
          backoff.suppressReads -= 1;
          return 0;
        }
        const targetNextSlab = Math.max(lastSlab, tracker.frontierSlabIndex) + 1;
        return this._scheduleSlabPrefetch(url, targetNextSlab, this.prefetchAheadSlabs);
      }
      return 0;
    }

    _scheduleSlabPrefetch(url, startSlabIndex, requestedCount) {
      if (
        this.slabSize <= 0 ||
        this.slabMaxBytes < this.slabSize * 2 ||
        !Number.isInteger(startSlabIndex) ||
        startSlabIndex < 0 ||
        !Number.isInteger(requestedCount) ||
        requestedCount <= 0
      ) {
        return 0;
      }
      if (this._getNegative(url) !== null) {
        return 0;
      }

      const maxFitSlabs = Math.floor(this.slabMaxBytes / this.slabSize);
      const boundedCount = Math.min(requestedCount, MAX_SLAB_PREFETCH_AHEAD, Math.max(0, maxFitSlabs - 1));
      if (boundedCount <= 0) {
        return 0;
      }

      const maxQueuedPrefetches = Math.max(4, this.maxConcurrentSlabFetches * 2);
      const knownSize = this.getFileSize(url);
      let scheduled = 0;

      for (let i = 0; i < boundedCount; i += 1) {
        if (this._countQueuedPrefetchWaiters() >= maxQueuedPrefetches) {
          break;
        }
        const slabIdx = startSlabIndex + i;
        const slabStart = slabIdx * this.slabSize;
        if (knownSize !== null && slabStart >= knownSize) {
          break;
        }
        const key = this._slabKey(url, slabIdx);
        if (this._slabs.has(key) || this._inFlightSlabs.has(key)) {
          continue;
        }

        this._stats.slabPrefetchesScheduled += 1;
        scheduled += 1;
        const p = this._fetchSlab(url, slabIdx, { isPrefetch: true }).then(
          (bytes) => {
            if (bytes && bytes.byteLength > 0) {
              this._stats.slabPrefetchesCompleted += 1;
            }
          },
          () => {
            this._stats.slabPrefetchErrors += 1;
          },
        );
        this._pendingPrefetches.add(p);
        p.finally(() => {
          this._pendingPrefetches.delete(p);
        });
      }

      return scheduled;
    }

    async prefetchSlabs(url, startOffset = 0, count) {
      if (typeof url !== 'string' || url.length === 0) {
        return { scheduled: 0, alreadyCached: 0, skipped: 0, completed: 0, failed: 0 };
      }
      if (this.slabSize <= 0 || this.slabMaxBytes < this.slabSize) {
        return { scheduled: 0, alreadyCached: 0, skipped: 0, completed: 0, failed: 0 };
      }
      if (!Number.isInteger(startOffset) || startOffset < 0) {
        throw createErrnoError(-22, `Invalid prefetch startOffset: ${startOffset}`);
      }
      const reqCount =
        Number.isInteger(count) && count >= 0
          ? count
          : Math.max(1, this.prefetchAheadSlabs || 1);
      const maxFitSlabs = Math.floor(this.slabMaxBytes / this.slabSize);
      const boundedCount = Math.min(reqCount, MAX_SLAB_PREFETCH_AHEAD, Math.max(1, maxFitSlabs));
      const firstSlab = Math.floor(startOffset / this.slabSize);
      const knownSize = this.getFileSize(url);

      let scheduled = 0;
      let alreadyCached = 0;
      let skipped = 0;
      let completed = 0;
      let failed = 0;
      const tasks = [];

      for (let i = 0; i < boundedCount; i += 1) {
        const slabIdx = firstSlab + i;
        if (knownSize !== null && slabIdx * this.slabSize >= knownSize) {
          skipped += 1;
          break;
        }
        const key = this._slabKey(url, slabIdx);
        if (this._slabs.has(key)) {
          this._touchSlab(key);
          alreadyCached += 1;
          continue;
        }
        this._stats.slabPrefetchesScheduled += 1;
        scheduled += 1;
        const p = this._fetchSlab(url, slabIdx, { isPrefetch: true }).then(
          (bytes) => {
            if (bytes && bytes.byteLength > 0) {
              this._stats.slabPrefetchesCompleted += 1;
              completed += 1;
            }
          },
          () => {
            this._stats.slabPrefetchErrors += 1;
            failed += 1;
          },
        );
        this._pendingPrefetches.add(p);
        p.finally(() => {
          this._pendingPrefetches.delete(p);
        });
        tasks.push(p);
      }

      if (tasks.length > 0) {
        await Promise.all(tasks);
      }
      return { scheduled, alreadyCached, skipped, completed, failed };
    }

    async waitForPrefetches() {
      while (this._pendingPrefetches.size > 0) {
        await Promise.all(Array.from(this._pendingPrefetches));
      }
    }

    tuneSlabCache(profileOrOptions = 'video4k') {
      const merged =
        typeof profileOrOptions === 'string'
          ? createSlabCacheTuningProfile({ profile: profileOrOptions })
          : createSlabCacheTuningProfile(
              Object.assign(
                {
                  slabSize: this.slabSize || DEFAULT_SLAB_SIZE,
                  slabMaxBytes: this.slabMaxBytes || DEFAULT_SLAB_MAX_BYTES,
                  prefetchAheadSlabs: this.prefetchAheadSlabs,
                  maxConcurrentSlabFetches: this.maxConcurrentSlabFetches,
                  sequentialThresholdReads: this.sequentialThresholdReads,
                },
                profileOrOptions && typeof profileOrOptions === 'object' ? profileOrOptions : {},
              ),
            );

      const prevSlabSize = this.slabSize;
      this.slabSize = merged.slabSize;
      this.slabMaxBytes = merged.slabMaxBytes;
      this.prefetchAheadSlabs = merged.prefetchAheadSlabs;
      this.maxConcurrentSlabFetches = merged.maxConcurrentSlabFetches;
      this.sequentialThresholdReads = merged.sequentialThresholdReads;

      if (prevSlabSize !== this.slabSize || !this._isSlabModeEnabled()) {
        this._slabs.clear();
        this._probationSlabs.clear();
        this._protectedSlabs.clear();
        this._probationSlabBytes = 0;
        this._protectedSlabBytes = 0;
        this._slabLastSwitch.clear();
        this._lastAccessedSlabKey = null;
        this._prefetchedSlabKeys.clear();
        this._slabBytesCached = 0;
        this._streamTrackers.clear();
        this._urlPrefetchBackoff.clear();
      } else {
        while (this._slabBytesCached > this.slabMaxBytes && this._slabs.size > 0) {
          const freed = this._evictOneSlab();
          if (freed <= 0) break;
        }
      }
      // Clear dormant legacy chunk pool when slab mode is active to prevent dual-pool RAM bloat.
      if (this._isSlabModeEnabled() && this._chunks.size > 0) {
        this._chunks.clear();
        this._bytesCached = 0;
      }

      return {
        slabSize: this.slabSize,
        slabMaxBytes: this.slabMaxBytes,
        prefetchAheadSlabs: this.prefetchAheadSlabs,
        maxConcurrentSlabFetches: this.maxConcurrentSlabFetches,
        sequentialThresholdReads: this.sequentialThresholdReads,
      };
    }

    applyPerfProfile(profileOrName) {
      const profile = coerceIoPerfProfile(profileOrName);
      this._perfProfile = profile;
      this._perfProfileName = profile.name || null;
      const tuned = this.tuneSlabCache({
        slabMaxBytes: profile.ioSlabCacheBytes,
        maxConcurrentSlabFetches: profile.ioMaxConcurrentLanes,
      });
      return Object.assign({ profile: this._perfProfileName }, tuned);
    }

    _getSlabRange(url, firstSlab, lastSlab, cacheFlagsOut = null) {
      const resolveSingleSlab = (slabIndex, slotIdx) => {
        const key = this._slabKey(url, slabIndex);

        const memHit = this._touchSlab(key);
        if (memHit !== null) {
          this._stats.slabHits += 1;
          if (this._prefetchedSlabKeys.delete(key)) {
            this._stats.prefetchHits += 1;
            this._recordPrefetchConsumed(key);
          }
          if (cacheFlagsOut) {
            cacheFlagsOut[slotIdx] = true;
          }
          return Promise.resolve(memHit);
        }

        const inFlight = this._inFlightSlabs.get(key);
        if (inFlight) {
          this._stats.inFlightCoalesced += 1;
          const meta = this._inFlightSlabMeta.get(key);
          if (meta && meta.isPrefetch) {
            if (!meta.prefetchHitCounted) {
              meta.prefetchHitCounted = true;
              this._stats.prefetchHits += 1;
            }
            if (meta.waiter) {
              this._promoteSlabWaiter(meta.waiter);
            }
          }
          return inFlight;
        }

        this._stats.slabMisses += 1;
        return this._fetchSlab(url, slabIndex);
      };

      const totalSlabs = lastSlab - firstSlab + 1;
      if (totalSlabs > 1 && this.getFileSize(url) === null) {
        return (async () => {
          const firstBytes = await resolveSingleSlab(firstSlab, 0);
          const results = [firstBytes];
          if (!firstBytes || firstBytes.byteLength < this.slabSize) {
            return results;
          }
          const knownSize = this.getFileSize(url);
          const clampedLast =
            knownSize !== null
              ? knownSize === 0
                ? firstSlab
                : Math.min(lastSlab, Math.floor((knownSize - 1) / this.slabSize))
              : lastSlab;
          if (clampedLast > firstSlab) {
            const rest = [];
            for (let s = firstSlab + 1; s <= clampedLast; s += 1) {
              rest.push(resolveSingleSlab(s, s - firstSlab));
            }
            const restBytes = await Promise.all(rest);
            for (let i = 0; i < restBytes.length; i += 1) {
              results.push(restBytes[i]);
            }
          }
          return results;
        })();
      }

      const slabPromises = new Array(totalSlabs);
      for (let i = 0; i < totalSlabs; i += 1) {
        slabPromises[i] = resolveSingleSlab(firstSlab + i, i);
      }
      return Promise.all(slabPromises);
    }

    hasInFlightOverlap(url, offset, length) {
      if (
        typeof url !== 'string' ||
        url.length === 0 ||
        !Number.isInteger(offset) ||
        offset < 0 ||
        !Number.isInteger(length) ||
        length <= 0
      ) {
        return false;
      }
      if (this._inFlightOpen.has(url)) {
        return true;
      }
      if (this._isSlabModeEnabled() && this._inFlightSlabs.size > 0) {
        const firstSlab = Math.floor(offset / this.slabSize);
        const lastSlab = Math.floor((offset + length - 1) / this.slabSize);
        for (let s = firstSlab; s <= lastSlab; s += 1) {
          if (this._inFlightSlabs.has(this._slabKey(url, s))) {
            return true;
          }
        }
      }
      if (this._inFlightChunks.size > 0) {
        const firstChunk = Math.floor(offset / this.chunkSize);
        const lastChunk = Math.floor((offset + length - 1) / this.chunkSize);
        for (let c = firstChunk; c <= lastChunk; c += 1) {
          if (this._inFlightChunks.has(this._chunkKey(url, c))) {
            return true;
          }
        }
      }
      return false;
    }

    tryReadIntoSync(url, offset, length, dst, dstOffset = 0) {
      if (
        !Number.isInteger(offset) ||
        offset < 0 ||
        !Number.isInteger(length) ||
        length < 0 ||
        !(dst instanceof Uint8Array) ||
        !Number.isInteger(dstOffset) ||
        dstOffset < 0 ||
        dstOffset > dst.byteLength
      ) {
        return -1;
      }
      if (length === 0) {
        return 0;
      }
      if (this._getNegative(url) !== null) {
        return -1;
      }

      let effectiveLen = Math.min(length, dst.byteLength - dstOffset);
      if (effectiveLen <= 0) {
        return 0;
      }
      const meta = this._touchMeta(url);
      if (meta && Number.isSafeInteger(meta.size)) {
        if (offset >= meta.size) {
          return 0;
        }
        effectiveLen = Math.min(effectiveLen, meta.size - offset);
        if (effectiveLen <= 0) {
          return 0;
        }
      }

      if (this._isSlabModeEnabled() && this._slabs.size > 0) {
        const firstSlab = Math.floor(offset / this.slabSize);
        const lastSlab = Math.floor((offset + effectiveLen - 1) / this.slabSize);
        if (firstSlab === lastSlab) {
          const key = this._slabKey(url, firstSlab);
          const slab = this._touchSlab(key);
          if (slab === null) {
            return -1;
          }
          this._stats.slabHits += 1;
          if (this._prefetchedSlabKeys.delete(key)) {
            this._stats.prefetchHits += 1;
            this._recordPrefetchConsumed(key);
          }
          const slabStart = firstSlab * this.slabSize;
          const sliceStart = offset - slabStart;
          const sliceEnd = Math.min(slab.byteLength, sliceStart + effectiveLen);
          const written = sliceEnd > sliceStart ? sliceEnd - sliceStart : 0;
          if (written > 0) {
            dst.set(slab.subarray(sliceStart, sliceEnd), dstOffset);
            this._stats.bytesServedFromCache += written;
          }
          if (this.prefetchAheadSlabs > 0) {
            this._recordSequentialAccess(url, offset, effectiveLen, lastSlab);
          }
          return written;
        }

        for (let s = firstSlab; s <= lastSlab; s += 1) {
          if (!this._slabs.has(this._slabKey(url, s))) {
            return -1;
          }
        }
        let totalWritten = 0;
        for (let s = firstSlab; s <= lastSlab; s += 1) {
          const key = this._slabKey(url, s);
          const slab = this._touchSlab(key);
          this._stats.slabHits += 1;
          if (this._prefetchedSlabKeys.delete(key)) {
            this._stats.prefetchHits += 1;
            this._recordPrefetchConsumed(key);
          }
          if (!slab || slab.byteLength === 0) {
            break;
          }
          const slabStart = s * this.slabSize;
          const sliceStart = Math.max(0, offset - slabStart);
          const sliceEnd = Math.min(slab.byteLength, offset + effectiveLen - slabStart);
          if (sliceEnd > sliceStart) {
            dst.set(slab.subarray(sliceStart, sliceEnd), dstOffset + totalWritten);
            totalWritten += sliceEnd - sliceStart;
          }
          if (slab.byteLength < this.slabSize) {
            break;
          }
        }
        if (totalWritten > 0) {
          this._stats.bytesServedFromCache += totalWritten;
        }
        if (this.prefetchAheadSlabs > 0) {
          this._recordSequentialAccess(url, offset, effectiveLen, lastSlab);
        }
        return totalWritten;
      }

      if (this._chunks.size > 0) {
        const firstChunk = Math.floor(offset / this.chunkSize);
        const lastChunk = Math.floor((offset + effectiveLen - 1) / this.chunkSize);
        for (let c = firstChunk; c <= lastChunk; c += 1) {
          if (!this._chunks.has(this._chunkKey(url, c))) {
            return -1;
          }
        }
        let totalWritten = 0;
        for (let c = firstChunk; c <= lastChunk; c += 1) {
          const chunk = this._touchChunk(this._chunkKey(url, c));
          this._stats.chunkHits += 1;
          if (!chunk || chunk.byteLength === 0) {
            break;
          }
          const chunkStart = c * this.chunkSize;
          const sliceStart = Math.max(0, offset - chunkStart);
          const sliceEnd = Math.min(chunk.byteLength, offset + effectiveLen - chunkStart);
          if (sliceEnd > sliceStart) {
            dst.set(chunk.subarray(sliceStart, sliceEnd), dstOffset + totalWritten);
            totalWritten += sliceEnd - sliceStart;
          }
          if (chunk.byteLength < this.chunkSize) {
            break;
          }
        }
        if (totalWritten > 0) {
          this._stats.bytesServedFromCache += totalWritten;
        }
        return totalWritten;
      }

      return -1;
    }

    async readInto(url, offset, length, dst, dstOffset = 0) {
      if (!Number.isInteger(offset) || offset < 0) {
        throw createErrnoError(-22, `Invalid read offset: ${offset}`);
      }
      if (!Number.isInteger(length) || length < 0) {
        throw createErrnoError(-22, `Invalid read length: ${length}`);
      }
      if (!(dst instanceof Uint8Array) || !Number.isInteger(dstOffset) || dstOffset < 0 || dstOffset > dst.byteLength) {
        throw createErrnoError(-22, 'Invalid destination buffer for readInto');
      }
      if (length === 0) {
        return 0;
      }

      const neg = this._getNegative(url);
      if (neg !== null) {
        this._stats.negativeHits += 1;
        throw this._makeHttpError(neg.status || 404, url, `${offset}..${offset + length - 1}`);
      }

      let effectiveLen = Math.min(length, dst.byteLength - dstOffset);
      if (effectiveLen <= 0) {
        return 0;
      }
      const meta = this._touchMeta(url);
      if (meta && Number.isSafeInteger(meta.size)) {
        if (offset >= meta.size) {
          return 0;
        }
        effectiveLen = Math.min(effectiveLen, meta.size - offset);
        if (effectiveLen <= 0) {
          return 0;
        }
      }

      const syncWritten = this.tryReadIntoSync(url, offset, effectiveLen, dst, dstOffset);
      if (syncWritten >= 0) {
        return syncWritten;
      }

      if (this._isSlabModeEnabled()) {
        const firstSlab = Math.floor(offset / this.slabSize);
        const lastSlab = Math.floor((offset + effectiveLen - 1) / this.slabSize);
        const cacheFlags = new Array(lastSlab - firstSlab + 1).fill(false);
        const slabs = await this._getSlabRange(url, firstSlab, lastSlab, cacheFlags);

        let totalWritten = 0;
        for (let i = 0; i < slabs.length; i += 1) {
          const slabIndex = firstSlab + i;
          const slab = slabs[i];
          if (!slab || slab.byteLength === 0) {
            break;
          }
          const slabStart = slabIndex * this.slabSize;
          const sliceStart = Math.max(0, offset - slabStart);
          const sliceEnd = Math.min(slab.byteLength, offset + effectiveLen - slabStart);
          if (sliceEnd > sliceStart) {
            const len = sliceEnd - sliceStart;
            dst.set(slab.subarray(sliceStart, sliceEnd), dstOffset + totalWritten);
            totalWritten += len;
            if (cacheFlags[i] || this._opfsL2HitBuffers.has(slab)) {
              this._stats.bytesServedFromCache += len;
            }
          }
          if (slab.byteLength < this.slabSize) {
            break;
          }
        }
        if (this.prefetchAheadSlabs > 0) {
          this._recordSequentialAccess(url, offset, effectiveLen, lastSlab);
        }
        return totalWritten;
      }

      const firstChunk = Math.floor(offset / this.chunkSize);
      const lastChunk = Math.floor((offset + effectiveLen - 1) / this.chunkSize);
      const chunks = await this._getChunkRange(url, firstChunk, lastChunk);

      let totalWritten = 0;
      for (let i = 0; i < chunks.length; i += 1) {
        const chunkIndex = firstChunk + i;
        const chunk = chunks[i];
        if (!chunk || chunk.byteLength === 0) {
          break;
        }
        const chunkStart = chunkIndex * this.chunkSize;
        const sliceStart = Math.max(0, offset - chunkStart);
        const sliceEnd = Math.min(chunk.byteLength, offset + effectiveLen - chunkStart);
        if (sliceEnd > sliceStart) {
          const len = sliceEnd - sliceStart;
          dst.set(chunk.subarray(sliceStart, sliceEnd), dstOffset + totalWritten);
          totalWritten += len;
          if (this._opfsL2HitBuffers.has(chunk)) {
            this._stats.bytesServedFromCache += len;
          }
        }
        if (chunk.byteLength < this.chunkSize) {
          break;
        }
      }
      return totalWritten;
    }

    tryReadSync(url, offset, length, options = {}) {
      if (!Number.isInteger(offset) || offset < 0) {
        return null;
      }
      if (!Number.isInteger(length) || length < 0) {
        return null;
      }
      if (length === 0) {
        return new Uint8Array(0);
      }
      if (this._getNegative(url) !== null) {
        return null;
      }

      let effectiveLen = length;
      const meta = this._touchMeta(url);
      if (meta && Number.isSafeInteger(meta.size)) {
        if (offset >= meta.size) {
          return new Uint8Array(0);
        }
        effectiveLen = Math.min(length, meta.size - offset);
        if (effectiveLen <= 0) {
          return new Uint8Array(0);
        }
      }

      const copy = options.copy !== false;

      if (this._isSlabModeEnabled() && this._slabs.size > 0) {
        const firstSlab = Math.floor(offset / this.slabSize);
        const lastSlab = Math.floor((offset + effectiveLen - 1) / this.slabSize);
        if (firstSlab === lastSlab) {
          const key = this._slabKey(url, firstSlab);
          const slab = this._touchSlab(key);
          if (slab === null) {
            return null;
          }
          this._stats.slabHits += 1;
          if (this._prefetchedSlabKeys.delete(key)) {
            this._stats.prefetchHits += 1;
            this._recordPrefetchConsumed(key);
          }
          const slabStart = firstSlab * this.slabSize;
          const sliceStart = offset - slabStart;
          const sliceEnd = Math.min(slab.byteLength, sliceStart + effectiveLen);
          const part =
            sliceEnd > sliceStart ? slab.subarray(sliceStart, sliceEnd) : new Uint8Array(0);
          if (part.byteLength > 0) {
            this._stats.bytesServedFromCache += part.byteLength;
          }
          if (this.prefetchAheadSlabs > 0) {
            this._recordSequentialAccess(url, offset, effectiveLen, lastSlab);
          }
          return copy ? part.slice() : part;
        }

        let allSlabsPresent = true;
        for (let s = firstSlab; s <= lastSlab; s += 1) {
          if (!this._slabs.has(this._slabKey(url, s))) {
            allSlabsPresent = false;
            break;
          }
        }
        if (allSlabsPresent) {
          const slices = [];
          let totalBytes = 0;
          for (let s = firstSlab; s <= lastSlab; s += 1) {
            const key = this._slabKey(url, s);
            const slab = this._touchSlab(key);
            this._stats.slabHits += 1;
            if (this._prefetchedSlabKeys.delete(key)) {
              this._stats.prefetchHits += 1;
            }
            if (!slab || slab.byteLength === 0) {
              break;
            }
            const slabStart = s * this.slabSize;
            const sliceStart = Math.max(0, offset - slabStart);
            const sliceEnd = Math.min(slab.byteLength, offset + effectiveLen - slabStart);
            if (sliceEnd > sliceStart) {
              const part = slab.subarray(sliceStart, sliceEnd);
              slices.push(part);
              totalBytes += part.byteLength;
            }
            if (slab.byteLength < this.slabSize) {
              break;
            }
          }
          if (totalBytes > 0) {
            this._stats.bytesServedFromCache += totalBytes;
          }
          if (this.prefetchAheadSlabs > 0) {
            this._recordSequentialAccess(url, offset, effectiveLen, lastSlab);
          }
          if (!copy && slices.length === 1) {
            return slices[0];
          }
          const out = new Uint8Array(totalBytes);
          let cursor = 0;
          for (let i = 0; i < slices.length; i += 1) {
            out.set(slices[i], cursor);
            cursor += slices[i].byteLength;
          }
          return out;
        }
      }

      if (this._chunks.size > 0) {
        const firstChunk = Math.floor(offset / this.chunkSize);
        const lastChunk = Math.floor((offset + effectiveLen - 1) / this.chunkSize);
        let allChunksPresent = true;
        for (let c = firstChunk; c <= lastChunk; c += 1) {
          if (!this._chunks.has(this._chunkKey(url, c))) {
            allChunksPresent = false;
            break;
          }
        }
        if (allChunksPresent) {
          const slices = [];
          let totalBytes = 0;
          for (let c = firstChunk; c <= lastChunk; c += 1) {
            const chunk = this._touchChunk(this._chunkKey(url, c));
            this._stats.chunkHits += 1;
            if (!chunk || chunk.byteLength === 0) {
              break;
            }
            const chunkStart = c * this.chunkSize;
            const sliceStart = Math.max(0, offset - chunkStart);
            const sliceEnd = Math.min(chunk.byteLength, offset + effectiveLen - chunkStart);
            if (sliceEnd > sliceStart) {
              const part = chunk.subarray(sliceStart, sliceEnd);
              slices.push(part);
              totalBytes += part.byteLength;
            }
            if (chunk.byteLength < this.chunkSize) {
              break;
            }
          }
          if (totalBytes > 0) {
            this._stats.bytesServedFromCache += totalBytes;
          }
          if (!copy && slices.length === 1) {
            return slices[0];
          }
          const out = new Uint8Array(totalBytes);
          let cursor = 0;
          for (let i = 0; i < slices.length; i += 1) {
            out.set(slices[i], cursor);
            cursor += slices[i].byteLength;
          }
          return out;
        }
      }

      return null;
    }

    async read(url, offset, length) {
      if (!Number.isInteger(offset) || offset < 0) {
        throw createErrnoError(-22, `Invalid read offset: ${offset}`);
      }
      if (!Number.isInteger(length) || length < 0) {
        throw createErrnoError(-22, `Invalid read length: ${length}`);
      }
      if (length === 0) {
        return new Uint8Array(0);
      }

      const neg = this._getNegative(url);
      if (neg !== null) {
        this._stats.negativeHits += 1;
        throw this._makeHttpError(neg.status || 404, url, `${offset}..${offset + length - 1}`);
      }

      let effectiveLen = length;
      const meta = this._touchMeta(url);
      if (meta && Number.isSafeInteger(meta.size)) {
        if (offset >= meta.size) {
          return new Uint8Array(0);
        }
        effectiveLen = Math.min(length, meta.size - offset);
        if (effectiveLen <= 0) {
          return new Uint8Array(0);
        }
      }

      const syncHit = this.tryReadSync(url, offset, effectiveLen, { copy: true });
      if (syncHit !== null) {
        return syncHit;
      }

      if (this._isSlabModeEnabled()) {
        const firstSlab = Math.floor(offset / this.slabSize);
        const lastSlab = Math.floor((offset + effectiveLen - 1) / this.slabSize);
        const cacheFlags = new Array(lastSlab - firstSlab + 1).fill(false);
        const slabs = await this._getSlabRange(url, firstSlab, lastSlab, cacheFlags);

        const slices = [];
        let totalBytes = 0;
        for (let i = 0; i < slabs.length; i += 1) {
          const slabIndex = firstSlab + i;
          const slab = slabs[i];
          if (!slab || slab.byteLength === 0) {
            break;
          }
          const slabStart = slabIndex * this.slabSize;
          const sliceStart = Math.max(0, offset - slabStart);
          const sliceEnd = Math.min(slab.byteLength, offset + effectiveLen - slabStart);
          if (sliceEnd > sliceStart) {
            const part = slab.subarray(sliceStart, sliceEnd);
            slices.push(part);
            totalBytes += part.byteLength;
            if (cacheFlags[i] || this._opfsL2HitBuffers.has(slab)) {
              this._stats.bytesServedFromCache += part.byteLength;
            }
          }
          if (slab.byteLength < this.slabSize) {
            break;
          }
        }

        const out = new Uint8Array(totalBytes);
        let cursor = 0;
        for (let i = 0; i < slices.length; i += 1) {
          const part = slices[i];
          out.set(part, cursor);
          cursor += part.byteLength;
        }
        if (this.prefetchAheadSlabs > 0) {
          this._recordSequentialAccess(url, offset, effectiveLen, lastSlab);
        }
        return out;
      }

      const firstChunk = Math.floor(offset / this.chunkSize);
      const lastChunk = Math.floor((offset + effectiveLen - 1) / this.chunkSize);
      const chunks = await this._getChunkRange(url, firstChunk, lastChunk);

      const slices = [];
      let totalBytes = 0;

      for (let i = 0; i < chunks.length; i += 1) {
        const chunkIndex = firstChunk + i;
        const chunk = chunks[i];
        if (!chunk || chunk.byteLength === 0) {
          break;
        }
        const chunkStart = chunkIndex * this.chunkSize;
        const sliceStart = Math.max(0, offset - chunkStart);
        const sliceEnd = Math.min(chunk.byteLength, offset + effectiveLen - chunkStart);
        if (sliceEnd > sliceStart) {
          const part = chunk.subarray(sliceStart, sliceEnd);
          slices.push(part);
          totalBytes += part.byteLength;
          if (this._opfsL2HitBuffers.has(chunk)) {
            this._stats.bytesServedFromCache += part.byteLength;
          }
        }
        if (chunk.byteLength < this.chunkSize) {
          break;
        }
      }

      // Always return a fresh owned buffer so callers cannot mutate cached chunks.
      const out = new Uint8Array(totalBytes);
      let cursor = 0;
      for (let i = 0; i < slices.length; i += 1) {
        const part = slices[i];
        out.set(part, cursor);
        cursor += part.byteLength;
      }
      return out;
    }

    clear() {
      this._chunks.clear();
      this._slabs.clear();
      this._probationSlabs.clear();
      this._protectedSlabs.clear();
      this._probationSlabBytes = 0;
      this._protectedSlabBytes = 0;
      this._slabLastSwitch.clear();
      this._lastAccessedSlabKey = null;
      this._prefetchedSlabKeys.clear();
      this._urlPrefetchBackoff.clear();
      this._fileMeta.clear();
      this._fileFingerprints.clear();
      this._negativeCache.clear();
      this._streamTrackers.clear();
      this._clearOpfsL2WriteQueue();
      // Preserve _inFlightOpfsL2WriteKeys so active SyncAccessHandle writes cannot be collided
      // with if a concurrent write for the same key arrives immediately after clear().
      this._inFlightOpfsL2WritePayloads.clear();
      this._opfsL2Entries.clear();
      this._opfsL2TotalBytes = 0;
      this._opfsL2ValidatedFingerprints.clear();
      this._resetOpfsDirIndex();
      this._bytesCached = 0;
      this._slabBytesCached = 0;
    }

    getStats() {
      const cacheHits =
        this._stats.chunkHits + this._stats.slabHits + this._stats.opfsL2Hits;
      const cacheMisses = this._stats.chunkMisses + this._stats.slabMisses;
      return {
        hits: this._stats.chunkHits,
        misses: this._stats.chunkMisses,
        cacheHits,
        cacheMisses,
        slabCacheHits: this._stats.slabHits,
        slabCacheMisses: this._stats.slabMisses,
        slabEvictions: this._stats.slabEvictions,
        slabNetworkFetches: this._stats.slabNetworkFetches,
        slabBytesCached: this._slabBytesCached,
        slabMaxBytes: this.slabMaxBytes,
        slabSize: this.slabSize,
        slabEntries: this._slabs.size,
        slabPrefetchesScheduled: this._stats.slabPrefetchesScheduled,
        slabPrefetchesCompleted: this._stats.slabPrefetchesCompleted,
        slabPrefetchErrors: this._stats.slabPrefetchErrors,
        slabPrefetchWasted: this._stats.slabPrefetchWasted,
        slabPrefetchCancelled: this._stats.slabPrefetchCancelled,
        slabProtectedEntries: this._protectedSlabs.size,
        slabProbationEntries: this._probationSlabs.size,
        slabProtectedPromotions: this._stats.slabProtectedPromotions,
        prefetchHits: this._stats.prefetchHits,
        prefetchAheadSlabs: this.prefetchAheadSlabs,
        maxConcurrentSlabFetches: this.maxConcurrentSlabFetches,
        activeSlabFetches: this._activeSlabFetches,
        peakConcurrentSlabFetches: this._stats.peakConcurrentSlabFetches,
        metaHits: this._stats.metaHits,
        metaMisses: this._stats.metaMisses,
        negativeHits: this._stats.negativeHits,
        negativeEntries: this._negativeCache.size,
        inFlightCoalesced: this._stats.inFlightCoalesced,
        inflightCoalesced: this._stats.inFlightCoalesced,
        networkFetches: this._stats.networkFetches,
        bytesFetched: this._stats.bytesFetched,
        bytesServedFromCache: this._stats.bytesServedFromCache,
        evictions: this._stats.evictions,
        opfsL2Hits: this._stats.opfsL2Hits,
        opfsL2Misses: this._stats.opfsL2Misses,
        opfsL2Writes: this._stats.opfsL2Writes,
        opfsL2WriteErrors: this._stats.opfsL2WriteErrors,
        opfsL2WriteDrops: this._stats.opfsL2WriteDrops,
        opfsL2ProactiveEvictions: this._stats.opfsL2ProactiveEvictions,
        opfsL2Invalidations: this._stats.opfsL2Invalidations,
        opfsL2QueuedBytes: this._opfsL2QueuedBytes,
        memoryReclaims: this._stats.memoryReclaims,
        bytesReclaimed: this._stats.bytesReclaimed,
        bytesCached: this._bytesCached,
        maxBytes: this.maxBytes,
        chunkSize: this.chunkSize,
        entries: this._chunks.size,
        filesIndexed: this._fileMeta.size,
        perfProfile: this._perfProfileName,
      };
    }
  }

  function emptyRangeCacheStats() {
    return {
      hits: 0,
      misses: 0,
      opfsHits: 0,
      evictions: 0,
      bytesCached: 0,
      entries: 0,
    };
  }

  function emptySchedulerStats() {
    const mk = (priority) => ({
      priority,
      queued: 0,
      inFlight: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      retries: 0,
    });
    return {
      queued: 0,
      inFlight: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
      retries: 0,
      byPriority: {
        CRITICAL: mk(IO_PRIORITY.CRITICAL),
        HIGH: mk(IO_PRIORITY.HIGH),
        NORMAL: mk(IO_PRIORITY.NORMAL),
        PREFETCH: mk(IO_PRIORITY.PREFETCH),
      },
    };
  }

  function emptyWholeFileCacheStats(maxBytes = DEFAULT_WHOLE_FILE_MAX_BYTES) {
    return {
      pgimFullFetches: 0,
      wholeFileFetches: 0,
      hits: 0,
      fallbacks: 0,
      releases: 0,
      bytesFetched: 0,
      bytesServedFromCache: 0,
      activeEntries: 0,
      activeBytes: 0,
      maxBytes:
        Number.isFinite(maxBytes) && maxBytes >= 0
          ? Math.floor(maxBytes)
          : DEFAULT_WHOLE_FILE_MAX_BYTES,
    };
  }

  function emptyMailboxStats() {
    return {
      requests: 0,
      syncFastPathHits: 0,
      asyncPathHits: 0,
    };
  }

  function emptyDumpWriteStats(enabled = false) {
    return {
      enabled: Boolean(enabled),
      dumpWritesCompleted: 0,
      dumpBytesWritten: 0,
      dumpWriteErrors: 0,
      dumpHandlesOpened: 0,
      dumpFlushesCompleted: 0,
      openDumpHandles: 0,
      bufferedBytes: 0,
    };
  }

  function defaultNowMs() {
    if (typeof performance !== 'undefined' && performance && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function isTransferableArrayBuffer(buf) {
    if (!buf || !(buf instanceof ArrayBuffer)) {
      return false;
    }
    if (typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer) {
      return false;
    }
    if (buf.detached === true) {
      return false;
    }
    return true;
  }

  function emptyStreamingStats() {
    return {
      readsCompleted: 0,
      bytesTransferred: 0,
      batchedRequests: 0,
      coalescedFetches: 0,
      rangesCoalesced: 0,
      cacheHits: 0,
      errorCount: 0,
      avgReadMs: 0,
      totalReadMs: 0,
      maxReadMs: 0,
    };
  }

  function emptyStats() {
    const cacheStats = emptyRangeCacheStats();
    const immStats = emptyImmutableCacheStats();
    const wholeStats = emptyWholeFileCacheStats();
    const mailboxStats = emptyMailboxStats();
    const dumpWriteStats = emptyDumpWriteStats(false);
    const streamingStats = emptyStreamingStats();
    return {
      hits: cacheStats.hits,
      misses: cacheStats.misses,
      opfsHits: cacheStats.opfsHits,
      opfsMisses: 0,
      opfsWrites: 0,
      opfsErrors: 0,
      evictions: cacheStats.evictions,
      bytesCached: cacheStats.bytesCached,
      entries: cacheStats.entries,
      pgimFullFetches: wholeStats.pgimFullFetches,
      slabCacheHits: immStats.slabCacheHits,
      slabCacheMisses: immStats.slabCacheMisses,
      opfsL2Hits: immStats.opfsL2Hits,
      opfsL2Misses: immStats.opfsL2Misses,
      opfsL2Writes: immStats.opfsL2Writes,
      opfsL2WriteErrors: immStats.opfsL2WriteErrors,
      opfsL2Invalidations: immStats.opfsL2Invalidations,
      syncFastPathHits: mailboxStats.syncFastPathHits,
      dumpWritesCompleted: dumpWriteStats.dumpWritesCompleted,
      dumpBytesWritten: dumpWriteStats.dumpBytesWritten,
      dumpWriteErrors: dumpWriteStats.dumpWriteErrors,
      readsCompleted: streamingStats.readsCompleted,
      bytesTransferred: streamingStats.bytesTransferred,
      batchedRequests: streamingStats.batchedRequests,
      coalescedFetches: streamingStats.coalescedFetches,
      cacheHits: streamingStats.cacheHits,
      cacheMisses: 0,
      inflightCoalesced: immStats.inflightCoalesced,
      prefetchHits: immStats.prefetchHits,
      bytesFetched: 0,
      bytesServedFromCache: 0,
      errorCount: streamingStats.errorCount,
      avgReadMs: streamingStats.avgReadMs,
      rangeCache: cacheStats,
      immutableCache: immStats,
      wholeFileCache: wholeStats,
      mailbox: mailboxStats,
      dumpWrites: dumpWriteStats,
      streaming: streamingStats,
      scheduler: emptySchedulerStats(),
      rpf: {
        mountedCount: 0,
        archives: [],
        mountErrors: 0,
      },
      mountedRpfCount: 0,
      openHandles: 0,
      mailboxLanes: 0,
    };
  }

  function ensureTransferableBytes(bytes) {
    if (!(bytes instanceof Uint8Array)) {
      return new Uint8Array(0);
    }
    if (
      bytes.byteOffset === 0 &&
      bytes.byteLength === bytes.buffer.byteLength &&
      isTransferableArrayBuffer(bytes.buffer)
    ) {
      return bytes;
    }
    return bytes.slice();
  }

  function collectTransferableBuffers(results) {
    if (!Array.isArray(results) || results.length === 0) {
      return [];
    }
    const seen = new Set();
    const transfer = [];
    for (let i = 0; i < results.length; i += 1) {
      const item = results[i];
      if (!item || !item.ok || !(item.bytes instanceof Uint8Array)) {
        continue;
      }
      const buf = item.bytes.buffer;
      if (isTransferableArrayBuffer(buf) && buf.byteLength > 0 && !seen.has(buf)) {
        seen.add(buf);
        transfer.push(buf);
      }
    }
    return transfer;
  }

  function isValidIoPriorityValue(priority) {
    return (
      priority === IO_PRIORITY.CRITICAL ||
      priority === IO_PRIORITY.HIGH ||
      priority === IO_PRIORITY.NORMAL ||
      priority === IO_PRIORITY.PREFETCH
    );
  }

  /**
   * Validate, route, group by canonical asset, and coalesce adjacent/overlapping/near-adjacent
   * ranges for a batched multi-range read while preserving each caller's original index and id.
   */
  function coalesceBatchReadRequests(requests, options = {}) {
    if (!Array.isArray(requests)) {
      throw createErrnoError(-22, 'readBatch requests must be an array');
    }

    const opts = options && typeof options === 'object' ? options : {};
    const routeBases = {
      dataBase: opts.dataBase,
      bootBase: opts.bootBase,
      gameBase: opts.gameBase,
      shadersBase: opts.shadersBase,
    };
    const defaultPath = typeof opts.path === 'string' && opts.path.length > 0 ? opts.path : null;
    const defaultPriority =
      opts.priority !== undefined ? opts.priority : IO_PRIORITY.NORMAL;
    const maxGapBytes =
      Number.isFinite(opts.maxGapBytes) && opts.maxGapBytes >= 0
        ? Math.floor(opts.maxGapBytes)
        : DEFAULT_BATCH_COALESCE_MAX_GAP_BYTES;
    const rawMaxCoalesced =
      Number.isFinite(opts.maxCoalescedBytes) && opts.maxCoalescedBytes > 0
        ? Math.floor(opts.maxCoalescedBytes)
        : DEFAULT_BATCH_COALESCE_MAX_BYTES;
    const maxCoalescedBytes = Math.max(1, Math.min(rawMaxCoalesced, MAX_COALESCED_FETCH_BYTES));

    const byAsset = new Map();
    const zeroItems = [];
    const invalidItems = [];

    for (let i = 0; i < requests.length; i += 1) {
      const req = requests[i];
      const itemId = req && typeof req === 'object' ? req.id : undefined;

      if (!req || typeof req !== 'object') {
        invalidItems.push({
          index: i,
          id: itemId,
          errno: -22,
          error: `Batch read entry at index ${i} must be an object`,
        });
        continue;
      }

      const rawPath =
        typeof req.path === 'string' && req.path.length > 0 ? req.path : defaultPath;
      const route = resolveHttpfsRoute(rawPath, routeBases);
      if (!route.ok) {
        invalidItems.push({
          index: i,
          id: itemId,
          errno: -22,
          error: route.error || `Invalid path in batch read entry at index ${i}`,
        });
        continue;
      }

      const offset = req.offset !== undefined ? req.offset : req.start;
      if (!Number.isInteger(offset) || offset < 0) {
        invalidItems.push({
          index: i,
          id: itemId,
          errno: -22,
          error: `Invalid read offset at index ${i}: ${offset}`,
        });
        continue;
      }

      const length =
        req.length !== undefined
          ? req.length
          : Number.isInteger(req.end)
            ? req.end - offset
            : undefined;
      if (!Number.isInteger(length) || length < 0) {
        invalidItems.push({
          index: i,
          id: itemId,
          errno: -22,
          error: `Invalid read length at index ${i}: ${length}`,
        });
        continue;
      }

      const priority = req.priority !== undefined ? req.priority : defaultPriority;
      if (!isValidIoPriorityValue(priority)) {
        invalidItems.push({
          index: i,
          id: itemId,
          errno: -22,
          error: `Invalid IO priority at index ${i}: ${priority}`,
        });
        continue;
      }

      const normalizedItem = {
        index: i,
        id: itemId,
        rawPath,
        route,
        offset,
        length,
        priority,
      };

      if (length === 0) {
        zeroItems.push(normalizedItem);
        continue;
      }

      const assetKey = route.isRpf ? `rpf:${route.url.toLowerCase()}` : `http:${route.url}`;
      let list = byAsset.get(assetKey);
      if (!list) {
        list = [];
        byAsset.set(assetKey, list);
      }
      list.push(normalizedItem);
    }

    const groups = [];
    for (const [assetKey, list] of byAsset.entries()) {
      list.sort((a, b) =>
        a.offset !== b.offset
          ? a.offset - b.offset
          : a.length !== b.length
            ? a.length - b.length
            : a.index - b.index,
      );

      let current = {
        assetKey,
        route: list[0].route,
        rawPath: list[0].rawPath,
        offset: list[0].offset,
        end: list[0].offset + list[0].length,
        length: list[0].length,
        priority: list[0].priority,
        gapBytes: 0,
        items: [list[0]],
      };

      const finalize = () => {
        groups.push({
          assetKey: current.assetKey,
          route: current.route,
          rawPath: current.rawPath,
          offset: current.offset,
          length: current.end - current.offset,
          end: current.end,
          priority: current.priority,
          gapBytes: current.gapBytes,
          coalesced: current.items.length > 1,
          items: current.items,
        });
      };

      for (let i = 1; i < list.length; i += 1) {
        const item = list[i];
        const itemEnd = item.offset + item.length;
        const gap = item.offset > current.end ? item.offset - current.end : 0;
        const candidateEnd = Math.max(current.end, itemEnd);
        const candidateSpan = candidateEnd - current.offset;
        const isContained = item.offset >= current.offset && itemEnd <= current.end;

        if (gap <= maxGapBytes && (candidateSpan <= maxCoalescedBytes || isContained)) {
          current.end = candidateEnd;
          current.length = candidateSpan;
          current.gapBytes += gap;
          if (item.priority < current.priority) {
            current.priority = item.priority;
          }
          current.items.push(item);
        } else {
          finalize();
          current = {
            assetKey,
            route: item.route,
            rawPath: item.rawPath,
            offset: item.offset,
            end: itemEnd,
            length: item.length,
            priority: item.priority,
            gapBytes: 0,
            items: [item],
          };
        }
      }

      finalize();
    }

    return {
      groups,
      zeroItems,
      invalidItems,
    };
  }

  class IoWorkerController {
    constructor(options = {}) {
      this._rangeCacheModule = options.rangeCacheModule || null;
      this._httpfsMailboxModule = options.httpfsMailboxModule || null;
      this._ioSchedulerModule = options.ioSchedulerModule || null;
      this._rpf9VfsModule = options.rpf9VfsModule || null;
      this._perfProfileModule = options.perfProfileModule || null;

      const resolvedPerfProfile =
        options.perfProfile !== undefined && options.perfProfile !== null
          ? coerceIoPerfProfile(options.perfProfile, this._perfProfileModule)
          : null;
      this._perfProfile = resolvedPerfProfile;
      this._perfProfileName = resolvedPerfProfile ? resolvedPerfProfile.name || null : null;
      this._maxMailboxLanes =
        resolvedPerfProfile &&
        Number.isInteger(resolvedPerfProfile.ioMaxConcurrentLanes) &&
        resolvedPerfProfile.ioMaxConcurrentLanes > 0
          ? resolvedPerfProfile.ioMaxConcurrentLanes
          : null;

      this.dataBase = normalizeDataBase(options.dataBase);
      this.bootBase = normalizeOptionalBase(options.bootBase);
      this.gameBase = normalizeOptionalBase(options.gameBase);
      this.shadersBase =
        options.shadersBase !== undefined
          ? normalizeOptionalBase(options.shadersBase)
          : '/shaders/';
      const dumpCfg = resolveDumpConfig({
        allowDumpWrites: options.allowDumpWrites,
        dumpEndpoint: options.dumpEndpoint,
      });
      this.allowDumpWrites = dumpCfg.allowDumpWrites;
      this.dumpEndpoint = dumpCfg.dumpEndpoint;
      this._dumpMaxBufferBytes =
        Number.isFinite(options.dumpMaxBufferBytes) && options.dumpMaxBufferBytes > 0
          ? Math.floor(options.dumpMaxBufferBytes)
          : DEFAULT_DUMP_MAX_BUFFER_BYTES;
      this._dumpFlushChunkBytes =
        Number.isFinite(options.dumpFlushChunkBytes) && options.dumpFlushChunkBytes > 0
          ? Math.min(Math.floor(options.dumpFlushChunkBytes), this._dumpMaxBufferBytes)
          : Math.min(DEFAULT_DUMP_FLUSH_CHUNK_BYTES, this._dumpMaxBufferBytes);
      this.manifest = options.manifest || null;
      this._manifestIndexRef = null;
      this._manifestIndexExact = null;
      this._manifestIndexLower = null;
      this.rangeCache = options.rangeCache || null;
      this._customRangeCacheProvided = Boolean(options.rangeCache);
      this._hasRoutedRangeCache = !options.rangeCache;
      this.immutableCache = options.immutableCache || null;
      this._immutableCacheEnabled = options.immutableCacheEnabled !== false;
      this.scheduler = options.scheduler || null;
      this.rpfMountTable = options.rpfMountTable || null;
      this.fetchImpl = typeof options.fetchImpl === 'function' ? options.fetchImpl : null;
      this.postMessage = typeof options.postMessage === 'function' ? options.postMessage : null;
      this.decryptBlock = typeof options.decryptBlock === 'function' ? options.decryptBlock : null;
      this.decompressEntry =
        typeof options.decompressEntry === 'function' ? options.decompressEntry : null;

      this._cacheOptions = {
        chunkSize: options.chunkSize,
        maxBytes: options.maxBytes,
        maxPrefetchConcurrency:
          options.maxPrefetchConcurrency !== undefined
            ? options.maxPrefetchConcurrency
            : resolvedPerfProfile
              ? resolvedPerfProfile.ioMaxConcurrentLanes
              : undefined,
        opfsDir: options.opfsDir !== undefined ? options.opfsDir : undefined,
        coalesceAdjacent: options.coalesceAdjacent,
        coalesceMaxGapBytes: options.coalesceMaxGapBytes,
        coalesceMaxMergedBytes: options.coalesceMaxMergedBytes,
      };
      this._now =
        typeof options.now === 'function'
          ? options.now
          : typeof options.nowFn === 'function'
            ? options.nowFn
            : defaultNowMs;
      this._batchCoalesceMaxGapBytes =
        Number.isFinite(options.batchCoalesceMaxGapBytes) && options.batchCoalesceMaxGapBytes >= 0
          ? Math.floor(options.batchCoalesceMaxGapBytes)
          : DEFAULT_BATCH_COALESCE_MAX_GAP_BYTES;
      this._batchCoalesceMaxBytes =
        Number.isFinite(options.batchCoalesceMaxBytes) && options.batchCoalesceMaxBytes > 0
          ? Math.min(Math.floor(options.batchCoalesceMaxBytes), MAX_COALESCED_FETCH_BYTES)
          : DEFAULT_BATCH_COALESCE_MAX_BYTES;
      this._immutableCacheOptions = {
        chunkSize: options.immutableCacheChunkSize,
        maxBytes: options.immutableCacheMaxBytes,
        slabSize:
          options.slabSize !== undefined
            ? options.slabSize
            : options.immutableSlabSize !== undefined
              ? options.immutableSlabSize
              : options.blockSize,
        slabMaxBytes:
          options.slabMaxBytes !== undefined
            ? options.slabMaxBytes
            : options.immutableSlabMaxBytes !== undefined
              ? options.immutableSlabMaxBytes
              : options.maxRamCacheBytes !== undefined
                ? options.maxRamCacheBytes
                : Number.isFinite(options.recommendedSlabMaxBytes) &&
                    options.recommendedSlabMaxBytes > 0
                  ? options.recommendedSlabMaxBytes
                  : resolvedPerfProfile
                    ? resolvedPerfProfile.ioSlabCacheBytes
                    : undefined,
        slabCacheEnabled: options.slabCacheEnabled,
        slabProfile: options.slabProfile,
        prefetchAheadSlabs: options.prefetchAheadSlabs,
        maxConcurrentSlabFetches:
          options.maxConcurrentSlabFetches !== undefined
            ? options.maxConcurrentSlabFetches
            : resolvedPerfProfile
              ? resolvedPerfProfile.ioMaxConcurrentLanes
              : undefined,
        sequentialThresholdReads: options.sequentialThresholdReads,
        negativeCacheTtlMs: options.negativeCacheTtlMs,
        negativeCacheMaxEntries: options.negativeCacheMaxEntries,
        metadataMaxEntries: options.metadataMaxEntries,
        opfsL2: options.opfsL2,
        opfsCache: options.opfsCache,
        opfsDir: options.opfsDir,
        opfsL2Enabled: options.opfsL2Enabled,
        opfsL2Allowlist:
          options.opfsL2Allowlist !== undefined
            ? options.opfsL2Allowlist
            : options.hotOpfsAllowlist !== undefined
              ? options.hotOpfsAllowlist
              : options.opfsHotRpfAllowlist,
        opfsL2RequireEtag: options.opfsL2RequireEtag,
        fileFingerprints: options.fileFingerprints,
        maxPendingOpfsL2Writes: options.maxPendingOpfsL2Writes,
        maxPendingOpfsL2WriteBytes: options.maxPendingOpfsL2WriteBytes,
        maxOpfsL2WriteConcurrency: options.maxOpfsL2WriteConcurrency,
        opfsL2MaxBytes: options.opfsL2MaxBytes,
        now: typeof options.now === 'function' ? options.now : undefined,
      };
      this._wholeFileBufferEnabled = options.wholeFileBufferEnabled !== false;
      const rawWholeFileMax =
        options.wholeFileMaxBytes !== undefined
          ? options.wholeFileMaxBytes
          : options.pgimMaxBytes;
      this._wholeFileMaxBytes =
        Number.isFinite(rawWholeFileMax) && rawWholeFileMax >= 0
          ? Math.floor(rawWholeFileMax)
          : DEFAULT_WHOLE_FILE_MAX_BYTES;
      const rawWholeFileTotalMax = options.wholeFileTotalMaxBytes;
      this._wholeFileTotalMaxBytes =
        Number.isFinite(rawWholeFileTotalMax) && rawWholeFileTotalMax >= 0
          ? Math.floor(rawWholeFileTotalMax)
          : Math.max(this._wholeFileMaxBytes, DEFAULT_WHOLE_FILE_TOTAL_MAX_BYTES);
      this._schedulerOptions = {
        maxConcurrency:
          options.maxConcurrency !== undefined
            ? options.maxConcurrency
            : resolvedPerfProfile
              ? resolvedPerfProfile.ioMaxConcurrentLanes
              : undefined,
        maxRetries: options.maxRetries,
        baseRetryDelayMs: options.baseRetryDelayMs,
        sleep: options.sleep,
      };

      this._mailboxHost = null;
      this._mailboxLanes = 0;
      this._handles = new Map();
      this._nextHandle = 1;
      this._activeWholeFiles = new Map(); // url -> { bytes, size, refCount }
      this._inFlightWholeFiles = new Map(); // url -> Promise<...>
      this._wholeFileStats = {
        pgimFullFetches: 0,
        wholeFileFetches: 0,
        hits: 0,
        fallbacks: 0,
        releases: 0,
        bytesFetched: 0,
        bytesServedFromCache: 0,
      };
      this._dumpWriteStats = {
        dumpWritesCompleted: 0,
        dumpBytesWritten: 0,
        dumpWriteErrors: 0,
        dumpHandlesOpened: 0,
        dumpFlushesCompleted: 0,
      };
      this._streamingStats = emptyStreamingStats();
      this._countedErrors = new WeakSet();
      this._mountedArchives = new Map();
      this._inFlightMounts = new Map();
      this._rpfMountErrors = 0;
      this._prefetchPromise = null;
      this._prefetchAbortController = null;
      this._initPromise = null;
    }

    get prefetchPromise() {
      return this._prefetchPromise;
    }

    get mailboxHost() {
      return this._mailboxHost;
    }

    get mailboxLanes() {
      return this._mailboxLanes;
    }

    get perfProfile() {
      return this._perfProfile;
    }

    get maxConcurrentLanes() {
      if (this._maxMailboxLanes !== null) {
        return this._maxMailboxLanes;
      }
      if (
        this.immutableCache &&
        Number.isInteger(this.immutableCache.maxConcurrentSlabFetches) &&
        this.immutableCache.maxConcurrentSlabFetches > 0
      ) {
        return this.immutableCache.maxConcurrentSlabFetches;
      }
      if (
        Number.isInteger(this._immutableCacheOptions.maxConcurrentSlabFetches) &&
        this._immutableCacheOptions.maxConcurrentSlabFetches > 0
      ) {
        return this._immutableCacheOptions.maxConcurrentSlabFetches;
      }
      return DEFAULT_MAX_CONCURRENT_SLAB_FETCHES;
    }

    _getRangeCacheModule() {
      return resolveRangeCacheModule(this._rangeCacheModule);
    }

    _getHttpfsMailboxModule() {
      return resolveHttpfsMailboxModule(this._httpfsMailboxModule);
    }

    _getIoSchedulerModule() {
      return resolveIoSchedulerModule(this._ioSchedulerModule);
    }

    _getRpf9VfsModule() {
      return resolveRpf9VfsModule(this._rpf9VfsModule);
    }

    _getFetchImpl(override) {
      if (typeof override === 'function') return override;
      if (typeof this.fetchImpl === 'function') return this.fetchImpl;
      if (typeof fetch === 'function') return fetch.bind(globalThis);
      return null;
    }

    _getRouteBases() {
      return {
        dataBase: this.dataBase,
        bootBase: this.bootBase,
        gameBase: this.gameBase,
        shadersBase: this.shadersBase,
      };
    }

    _resolveRoute(rawPath) {
      return resolveHttpfsRoute(rawPath, this._getRouteBases());
    }

    _routeToCachePath(route) {
      if (
        this._hasRoutedRangeCache &&
        (route.mount === 'boot' || route.mount === 'game' || route.mount === 'shaders')
      ) {
        return `${MOUNT_CACHE_PREFIX}${route.mount}/${route.relPath}`;
      }
      return route.relPath;
    }

    _createRoutedFetch(baseFetch) {
      if (typeof baseFetch !== 'function') {
        return null;
      }
      const controller = this;
      return function routedFetch(url, init) {
        if (typeof url === 'string') {
          const dataPrefix = controller.dataBase;
          const bootMarker = `${dataPrefix}${MOUNT_CACHE_PREFIX}boot/`;
          if (controller.bootBase && url.startsWith(bootMarker)) {
            const rel = url.slice(bootMarker.length);
            return baseFetch(`${controller.bootBase}${rel}`, init);
          }
          const gameMarker = `${dataPrefix}${MOUNT_CACHE_PREFIX}game/`;
          if (controller.gameBase && url.startsWith(gameMarker)) {
            const rel = url.slice(gameMarker.length);
            return baseFetch(`${controller.gameBase}${rel}`, init);
          }
          const shadersMarker = `${dataPrefix}${MOUNT_CACHE_PREFIX}shaders/`;
          if (controller.shadersBase && url.startsWith(shadersMarker)) {
            const rel = url.slice(shadersMarker.length);
            return baseFetch(`${controller.shadersBase}${rel}`, init);
          }
        }
        return baseFetch(url, init);
      };
    }

    ensureRangeCache() {
      if (!this.rangeCache) {
        const mod = this._getRangeCacheModule();
        if (!mod || typeof mod.RangeCache !== 'function') {
          throw new Error('PlayGta6RangeCache unavailable in io_worker');
        }
        this._hasRoutedRangeCache = true;
        const rangeCacheOpts = {
          dataBase: this.dataBase,
          chunkSize: this._cacheOptions.chunkSize,
          maxBytes: this._cacheOptions.maxBytes,
          maxPrefetchConcurrency: this._cacheOptions.maxPrefetchConcurrency,
          opfsDir: this._cacheOptions.opfsDir || null,
          fetchImpl: this._createRoutedFetch(this._getFetchImpl()),
        };
        if (typeof this._cacheOptions.coalesceAdjacent === 'boolean') {
          rangeCacheOpts.coalesceAdjacent = this._cacheOptions.coalesceAdjacent;
        }
        if (this._cacheOptions.coalesceMaxGapBytes !== undefined) {
          rangeCacheOpts.coalesceMaxGapBytes = this._cacheOptions.coalesceMaxGapBytes;
        }
        if (this._cacheOptions.coalesceMaxMergedBytes !== undefined) {
          rangeCacheOpts.coalesceMaxMergedBytes = this._cacheOptions.coalesceMaxMergedBytes;
        }
        this.rangeCache = new mod.RangeCache(rangeCacheOpts);
      }
      return this.rangeCache;
    }

    ensureImmutableCache() {
      if (!this.immutableCache) {
        this.immutableCache = new ImmutableHttpfsCache({
          chunkSize: this._immutableCacheOptions.chunkSize,
          maxBytes: this._immutableCacheOptions.maxBytes,
          slabSize: this._immutableCacheOptions.slabSize,
          slabMaxBytes: this._immutableCacheOptions.slabMaxBytes,
          slabCacheEnabled: this._immutableCacheOptions.slabCacheEnabled,
          slabProfile: this._immutableCacheOptions.slabProfile,
          prefetchAheadSlabs: this._immutableCacheOptions.prefetchAheadSlabs,
          maxConcurrentSlabFetches: this._immutableCacheOptions.maxConcurrentSlabFetches,
          sequentialThresholdReads: this._immutableCacheOptions.sequentialThresholdReads,
          negativeCacheTtlMs: this._immutableCacheOptions.negativeCacheTtlMs,
          negativeCacheMaxEntries: this._immutableCacheOptions.negativeCacheMaxEntries,
          metadataMaxEntries: this._immutableCacheOptions.metadataMaxEntries,
          opfsL2: this._immutableCacheOptions.opfsL2,
          opfsCache: this._immutableCacheOptions.opfsCache,
          opfsDir: this._immutableCacheOptions.opfsDir,
          opfsL2Enabled: this._immutableCacheOptions.opfsL2Enabled,
          opfsL2Allowlist: this._immutableCacheOptions.opfsL2Allowlist,
          opfsL2RequireEtag: this._immutableCacheOptions.opfsL2RequireEtag,
          fileFingerprints: this._immutableCacheOptions.fileFingerprints,
          maxPendingOpfsL2Writes: this._immutableCacheOptions.maxPendingOpfsL2Writes,
          maxPendingOpfsL2WriteBytes: this._immutableCacheOptions.maxPendingOpfsL2WriteBytes,
          maxOpfsL2WriteConcurrency: this._immutableCacheOptions.maxOpfsL2WriteConcurrency,
          opfsL2MaxBytes: this._immutableCacheOptions.opfsL2MaxBytes,
          perfProfile: this._perfProfile,
          now: this._immutableCacheOptions.now,
          fetchImpl: this._getFetchImpl(),
        });
      }
      return this.immutableCache;
    }

    async flushOpfsL2Writes() {
      if (this.immutableCache && typeof this.immutableCache.flushOpfsL2Writes === 'function') {
        await this.immutableCache.flushOpfsL2Writes();
      }
      if (this.rangeCache && typeof this.rangeCache.flushOpfsWrites === 'function') {
        await this.rangeCache.flushOpfsWrites();
      }
    }

    reclaimMemory(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      let bytesFreed = 0;
      let immutableResult = null;
      let rangeResult = null;

      if (this.immutableCache && typeof this.immutableCache.reclaimMemory === 'function') {
        immutableResult = this.immutableCache.reclaimMemory(opts);
        bytesFreed += immutableResult.bytesFreed || 0;
      }
      if (this.rangeCache && typeof this.rangeCache.reclaimMemory === 'function') {
        rangeResult = this.rangeCache.reclaimMemory(opts);
        bytesFreed += rangeResult.bytesFreed || 0;
      }

      return {
        ok: true,
        bytesFreed,
        immutableCache: immutableResult,
        rangeCache: rangeResult,
      };
    }

    trimCache(options = {}) {
      return this.reclaimMemory(options);
    }

    tuneSlabCache(profileOrOptions = 'video4k') {
      const immCache = this.ensureImmutableCache();
      const tuned = immCache.tuneSlabCache(profileOrOptions);
      this._immutableCacheOptions.slabSize = tuned.slabSize;
      this._immutableCacheOptions.slabMaxBytes = tuned.slabMaxBytes;
      this._immutableCacheOptions.slabCacheEnabled =
        tuned.slabSize > 0 && tuned.slabMaxBytes >= tuned.slabSize;
      this._immutableCacheOptions.prefetchAheadSlabs = tuned.prefetchAheadSlabs;
      this._immutableCacheOptions.maxConcurrentSlabFetches = tuned.maxConcurrentSlabFetches;
      this._immutableCacheOptions.sequentialThresholdReads = tuned.sequentialThresholdReads;
      return tuned;
    }

    applyPerfProfile(profileOrName) {
      const profile = coerceIoPerfProfile(profileOrName, this._perfProfileModule);
      this._perfProfile = profile;
      this._perfProfileName = profile.name || null;

      const slabMaxBytes = profile.ioSlabCacheBytes;
      const maxLanes = profile.ioMaxConcurrentLanes;

      this._maxMailboxLanes =
        Number.isInteger(maxLanes) && maxLanes > 0 ? maxLanes : this._maxMailboxLanes;
      this._immutableCacheOptions.slabMaxBytes = slabMaxBytes;
      this._immutableCacheOptions.maxConcurrentSlabFetches = maxLanes;
      this._cacheOptions.maxPrefetchConcurrency = maxLanes;
      this._schedulerOptions.maxConcurrency = maxLanes;

      let slabConfig = {
        slabSize: this._immutableCacheOptions.slabSize || DEFAULT_SLAB_SIZE,
        slabMaxBytes,
        prefetchAheadSlabs:
          this._immutableCacheOptions.prefetchAheadSlabs !== undefined
            ? this._immutableCacheOptions.prefetchAheadSlabs
            : DEFAULT_SLAB_PREFETCH_AHEAD,
        maxConcurrentSlabFetches: maxLanes,
        sequentialThresholdReads:
          this._immutableCacheOptions.sequentialThresholdReads !== undefined
            ? this._immutableCacheOptions.sequentialThresholdReads
            : DEFAULT_SEQUENTIAL_THRESHOLD_READS,
      };
      if (this.immutableCache && typeof this.immutableCache.applyPerfProfile === 'function') {
        slabConfig = this.immutableCache.applyPerfProfile(profile);
        this._immutableCacheOptions.slabSize = slabConfig.slabSize;
        this._immutableCacheOptions.slabMaxBytes = slabConfig.slabMaxBytes;
        this._immutableCacheOptions.prefetchAheadSlabs = slabConfig.prefetchAheadSlabs;
        this._immutableCacheOptions.maxConcurrentSlabFetches = slabConfig.maxConcurrentSlabFetches;
        this._immutableCacheOptions.sequentialThresholdReads = slabConfig.sequentialThresholdReads;
      }
      if (this.rangeCache && typeof this.rangeCache === 'object') {
        this.rangeCache.maxPrefetchConcurrency = maxLanes;
      }
      if (this.scheduler && typeof this.scheduler === 'object') {
        this.scheduler.maxConcurrency = maxLanes;
      }

      return {
        profile: this._perfProfileName,
        slabMaxBytes: slabConfig.slabMaxBytes,
        slabSize: slabConfig.slabSize,
        prefetchAheadSlabs: slabConfig.prefetchAheadSlabs,
        maxConcurrentSlabFetches: slabConfig.maxConcurrentSlabFetches,
        maxConcurrentLanes: this._maxMailboxLanes,
        maxPrefetchConcurrency: maxLanes,
      };
    }

    async prefetchVideoSlabs(rawPath, startOffset = 0, count) {
      const route = this._resolveRoute(rawPath);
      if (!route.ok || route.isRpf) {
        throw createErrnoError(-22, route.error || 'Invalid prefetch path');
      }
      const immCache = this.ensureImmutableCache();
      return immCache.prefetchSlabs(route.url, startOffset, count);
    }

    _shouldUseWholeFileBuffer(route) {
      if (
        !this._wholeFileBufferEnabled ||
        this._wholeFileMaxBytes <= 0 ||
        !route ||
        !route.ok ||
        route.isRpf
      ) {
        return false;
      }
      if (!isWholeFileBufferedPath(route.relPath)) {
        return false;
      }
      if (this._customRangeCacheProvided && !this.fetchImpl) {
        return false;
      }
      return Boolean(this._getFetchImpl());
    }

    _getActiveWholeFileBytes() {
      let total = 0;
      for (const rec of this._activeWholeFiles.values()) {
        if (rec && rec.bytes) {
          total += rec.bytes.byteLength;
        }
      }
      return total;
    }

    _retainWholeFile(url, bytes, size) {
      let rec = this._activeWholeFiles.get(url);
      if (!rec) {
        rec = { bytes, size, refCount: 0 };
        this._activeWholeFiles.set(url, rec);
      }
      rec.refCount += 1;
      return rec;
    }

    _releaseWholeFile(url) {
      const rec = this._activeWholeFiles.get(url);
      if (!rec) {
        return false;
      }
      rec.refCount -= 1;
      if (rec.refCount <= 0) {
        this._activeWholeFiles.delete(url);
        this._wholeFileStats.releases += 1;
        return true;
      }
      return false;
    }

    async _fetchWholeFile(url, cleanPath, knownSize) {
      const existing = this._activeWholeFiles.get(url);
      if (existing) {
        return {
          ok: true,
          buffered: true,
          bytes: existing.bytes,
          size: existing.size,
          reused: true,
        };
      }

      const activeTotal = this._getActiveWholeFileBytes();
      const remainingTotal = Math.max(0, this._wholeFileTotalMaxBytes - activeTotal);
      const cap = Math.min(this._wholeFileMaxBytes, remainingTotal);
      if (!this._wholeFileBufferEnabled || cap <= 0) {
        if (this._wholeFileBufferEnabled && this._wholeFileMaxBytes > 0 && remainingTotal <= 0) {
          this._wholeFileStats.fallbacks += 1;
        }
        return { ok: true, buffered: false, size: knownSize, probed: false };
      }

      const immCache =
        this._immutableCacheEnabled && isImmutableHttpfsPath(cleanPath)
          ? this.ensureImmutableCache()
          : null;

      if (immCache) {
        const neg = immCache._getNegative(url);
        if (neg !== null) {
          immCache._stats.negativeHits += 1;
          return { ok: false, buffered: false, errno: neg.errno, status: neg.status };
        }
        if (!Number.isSafeInteger(knownSize)) {
          const cachedSize = immCache.getFileSize(url);
          if (Number.isSafeInteger(cachedSize)) {
            knownSize = cachedSize;
          }
        }
      }

      if (Number.isSafeInteger(knownSize) && knownSize > cap) {
        this._wholeFileStats.fallbacks += 1;
        return { ok: true, buffered: false, size: knownSize, probed: false };
      }

      const inFlight = this._inFlightWholeFiles.get(url);
      if (inFlight) {
        return inFlight;
      }

      const fetchFn = this._getFetchImpl();
      if (typeof fetchFn !== 'function') {
        return { ok: true, buffered: false, size: knownSize, probed: false };
      }

      const isPgim = cleanPath.toLowerCase().endsWith('.pgim');
      const rangeEnd =
        Number.isSafeInteger(knownSize) && knownSize > 0
          ? Math.min(knownSize - 1, cap - 1)
          : cap - 1;

      const promise = (async () => {
        try {
          const res = await fetchFn(url, {
            headers: { Range: `bytes=0-${rangeEnd}` },
          });
          const status = res && typeof res.status === 'number' ? res.status : 0;

          if (status === 416) {
            const cr = parseContentRangeHeader(getResponseHeader(res.headers, 'Content-Range'));
            if (cr && cr.total === 0) {
              if (immCache) {
                immCache.setFileMetadata(url, 0);
              }
              this._wholeFileStats.wholeFileFetches += 1;
              if (isPgim) {
                this._wholeFileStats.pgimFullFetches += 1;
              }
              return {
                ok: true,
                buffered: true,
                bytes: new Uint8Array(0),
                size: 0,
                reused: false,
                probed: true,
              };
            }
            return { ok: false, buffered: false, errno: -5, status: 416 };
          }

          if (status === 404 || status === 403) {
            const errno = status === 403 ? -13 : -2;
            if (immCache) {
              immCache._putNegative(url, errno, status);
            }
            return { ok: false, buffered: false, errno, status };
          }

          if (!res || (!res.ok && status !== 206)) {
            return { ok: false, buffered: false, errno: -5, status: status || 500 };
          }

          if (status === 206) {
            const cr = parseContentRangeHeader(getResponseHeader(res.headers, 'Content-Range'));
            if (cr && !cr.unsatisfied) {
              if (cr.total !== null && immCache) {
                immCache.setFileMetadata(url, cr.total);
              }
              if (cr.total !== null && cr.total > cap) {
                if (res.body && typeof res.body.cancel === 'function') {
                  try {
                    await res.body.cancel();
                  } catch {
                    // Ignore stream cancel errors.
                  }
                }
                this._wholeFileStats.fallbacks += 1;
                return { ok: true, buffered: false, size: cr.total, probed: true };
              }
            }

            const raw = new Uint8Array(await res.arrayBuffer());
            if (cr && !cr.unsatisfied) {
              const total = cr.total;
              const gotComplete =
                cr.start === 0 &&
                (total !== null
                  ? cr.end + 1 >= total && raw.byteLength >= total
                  : cr.end < rangeEnd && raw.byteLength <= cap);
              if (gotComplete) {
                const finalBytes =
                  total !== null && raw.byteLength > total ? raw.slice(0, total) : raw;
                const finalSize = total !== null ? total : finalBytes.byteLength;
                if (immCache) {
                  immCache.setFileMetadata(url, finalSize);
                }
                this._wholeFileStats.wholeFileFetches += 1;
                if (isPgim) {
                  this._wholeFileStats.pgimFullFetches += 1;
                }
                this._wholeFileStats.bytesFetched += finalBytes.byteLength;
                return {
                  ok: true,
                  buffered: true,
                  bytes: finalBytes,
                  size: finalSize,
                  reused: false,
                  probed: true,
                };
              }
              this._wholeFileStats.fallbacks += 1;
              return { ok: true, buffered: false, size: total, probed: true };
            }

            if (raw.byteLength < rangeEnd + 1 && raw.byteLength <= cap) {
              if (immCache) {
                immCache.setFileMetadata(url, raw.byteLength);
              }
              this._wholeFileStats.wholeFileFetches += 1;
              if (isPgim) {
                this._wholeFileStats.pgimFullFetches += 1;
              }
              this._wholeFileStats.bytesFetched += raw.byteLength;
              return {
                ok: true,
                buffered: true,
                bytes: raw,
                size: raw.byteLength,
                reused: false,
                probed: true,
              };
            }
            this._wholeFileStats.fallbacks += 1;
            return { ok: true, buffered: false, size: null, probed: true };
          }

          // 200 OK fallback (server ignored Range and sent full resource).
          const contentLenHeader = getResponseHeader(res.headers, 'Content-Length');
          const declaredLen = contentLenHeader !== null ? Number(contentLenHeader) : null;
          if (Number.isSafeInteger(declaredLen) && declaredLen > cap) {
            if (res.body && typeof res.body.cancel === 'function') {
              try {
                await res.body.cancel();
              } catch {
                // Ignore stream cancel errors.
              }
            }
            if (immCache) {
              immCache.setFileMetadata(url, declaredLen);
            }
            this._wholeFileStats.fallbacks += 1;
            return { ok: true, buffered: false, size: declaredLen, probed: true };
          }

          const raw = new Uint8Array(await res.arrayBuffer());
          if (immCache) {
            immCache.setFileMetadata(url, raw.byteLength);
          }
          if (raw.byteLength <= cap) {
            this._wholeFileStats.wholeFileFetches += 1;
            if (isPgim) {
              this._wholeFileStats.pgimFullFetches += 1;
            }
            this._wholeFileStats.bytesFetched += raw.byteLength;
            return {
              ok: true,
              buffered: true,
              bytes: raw,
              size: raw.byteLength,
              reused: false,
              probed: true,
            };
          }
          this._wholeFileStats.fallbacks += 1;
          return { ok: true, buffered: false, size: raw.byteLength, probed: true };
        } catch (err) {
          return {
            ok: false,
            buffered: false,
            errno: mapErrorToErrno(err, -2),
            status: err && typeof err.status === 'number' ? err.status : null,
          };
        }
      })();

      this._inFlightWholeFiles.set(url, promise);
      try {
        return await promise;
      } finally {
        if (this._inFlightWholeFiles.get(url) === promise) {
          this._inFlightWholeFiles.delete(url);
        }
      }
    }

    _shouldUseImmutableCache(route) {
      if (!this._immutableCacheEnabled || !route || !route.ok || route.isRpf) {
        return false;
      }
      if (!isImmutableHttpfsPath(route.relPath)) {
        return false;
      }
      if (this._customRangeCacheProvided && !this.immutableCache && !this.fetchImpl) {
        return false;
      }
      return Boolean(this.immutableCache || this._getFetchImpl());
    }

    ensureScheduler() {
      const cache = this.ensureRangeCache();
      if (!this.scheduler) {
        const mod = this._getIoSchedulerModule();
        if (mod && typeof mod.IoPriorityScheduler === 'function') {
          const sleepFn =
            typeof this._schedulerOptions.sleep === 'function'
              ? this._schedulerOptions.sleep
              : typeof setTimeout === 'function'
                ? undefined
                : () => Promise.resolve();
          this.scheduler = new mod.IoPriorityScheduler({
            rangeCache: cache,
            maxConcurrency: this._schedulerOptions.maxConcurrency,
            maxRetries: this._schedulerOptions.maxRetries,
            baseRetryDelayMs: this._schedulerOptions.baseRetryDelayMs,
            sleep: sleepFn,
          });
        }
      }
      return this.scheduler;
    }

    ensureRpfMountTable() {
      if (!this.rpfMountTable) {
        const mod = this._getRpf9VfsModule();
        if (mod && typeof mod.RpfVfsMountTable === 'function') {
          this.rpfMountTable = new mod.RpfVfsMountTable();
        }
      }
      return this.rpfMountTable;
    }

    _ensureManifestIndex() {
      if (!this.manifest || !Array.isArray(this.manifest.files)) {
        this._manifestIndexRef = null;
        this._manifestIndexExact = null;
        this._manifestIndexLower = null;
        return null;
      }
      if (this._manifestIndexRef === this.manifest.files && this._manifestIndexExact) {
        return {
          exact: this._manifestIndexExact,
          lower: this._manifestIndexLower,
        };
      }
      const exact = new Map();
      const lower = new Map();
      const files = this.manifest.files;
      for (let i = 0; i < files.length; i += 1) {
        const entry = files[i];
        if (!entry || typeof entry.path !== 'string') continue;
        try {
          const norm = normalizeRelPath(entry.path);
          if (!exact.has(norm)) {
            exact.set(norm, entry);
          }
          const lowerKey = norm.toLowerCase();
          if (!lower.has(lowerKey)) {
            lower.set(lowerKey, entry);
          }
        } catch {
          // Ignore malformed manifest entries.
        }
      }
      this._manifestIndexRef = files;
      this._manifestIndexExact = exact;
      this._manifestIndexLower = lower;
      return { exact, lower };
    }

    _findManifestEntry(cleanPath) {
      const idx = this._ensureManifestIndex();
      if (!idx) {
        return { known: null, entry: null };
      }
      const exactHit = idx.exact.get(cleanPath);
      if (exactHit) {
        return { known: true, entry: exactHit };
      }
      const lowerKey = cleanPath.toLowerCase();
      const lowerHit = idx.lower.get(lowerKey);
      if (lowerHit) {
        return { known: true, entry: lowerHit };
      }
      return { known: false, entry: null };
    }

    _scheduledRead(cleanPath, offset, length, options = {}) {
      const scheduler = this.ensureScheduler();
      const priority = options.priority === undefined ? IO_PRIORITY.NORMAL : options.priority;
      if (scheduler) {
        return scheduler.scheduleRead(cleanPath, offset, length, {
          priority,
          signal: options.signal || null,
        });
      }
      const cache = this.ensureRangeCache();
      return cache.read(cleanPath, offset, length);
    }

    prefetchBootset(entries, options = {}) {
      const rawList = Array.isArray(entries)
        ? entries
        : entries && Array.isArray(entries.entries)
          ? entries.entries
          : [];

      if (rawList.length === 0) {
        return Promise.resolve({
          requested: 0,
          scheduled: 0,
          completed: 0,
          cancelled: 0,
          failed: 0,
        });
      }

      const routedEntries = [];
      for (let i = 0; i < rawList.length; i += 1) {
        const entry = rawList[i];
        if (!entry || typeof entry.path !== 'string') {
          continue;
        }
        const route = this._resolveRoute(entry.path);
        if (!route.ok || route.isRpf) {
          continue;
        }
        routedEntries.push(
          Object.assign({}, entry, { path: this._routeToCachePath(route) }),
        );
      }

      if (routedEntries.length === 0) {
        return Promise.resolve({
          requested: rawList.length,
          scheduled: 0,
          completed: 0,
          cancelled: 0,
          failed: 0,
        });
      }

      const scheduler = this.ensureScheduler();
      if (scheduler && typeof scheduler.schedulePrefetchList === 'function') {
        let signal = options.signal || null;
        if (!signal && typeof AbortController === 'function') {
          this._prefetchAbortController = new AbortController();
          signal = this._prefetchAbortController.signal;
        }
        const p = scheduler.schedulePrefetchList(routedEntries, { signal });
        this._prefetchPromise = p;
        return p;
      }

      const cache = this.ensureRangeCache();
      const p = cache.prefetchBootset(routedEntries, options);
      this._prefetchPromise = p;
      return p;
    }

    cancelPrefetch(options = {}) {
      let cancelled = 0;
      if (this.scheduler && typeof this.scheduler.cancelPrefetch === 'function') {
        cancelled = this.scheduler.cancelPrefetch();
      }
      if (options.abortInFlight && this._prefetchAbortController) {
        try {
          this._prefetchAbortController.abort(options.reason || 'Prefetch cancelled');
        } catch {
          // Ignore already-aborted controller errors.
        }
      }
      return cancelled;
    }

    async ensureRpfMounted(archivePath, options = {}) {
      const route = this._resolveRoute(archivePath);
      if (!route.ok || route.isRpf) {
        throw createErrnoError(-22, route.error || `Invalid RPF archive path: ${archivePath}`);
      }
      const cleanArchive = route.relPath;
      const cacheArchivePath = this._routeToCachePath(route);
      const archiveKey = cleanArchive.toLowerCase();

      const existing = this._mountedArchives.get(archiveKey);
      if (existing) {
        return existing;
      }

      const inFlight = this._inFlightMounts.get(archiveKey);
      if (inFlight) {
        return inFlight;
      }

      const rpfMod = this._getRpf9VfsModule();
      if (!rpfMod || typeof rpfMod.RpfArchiveVfs !== 'function') {
        throw createErrnoError(-38, 'PlayGta6Rpf9Vfs unavailable in io_worker');
      }

      const manifestLookup =
        route.mount === 'data'
          ? this._findManifestEntry(cleanArchive)
          : { known: null, entry: null };
      if (manifestLookup.known === false) {
        throw createErrnoError(-2, `RPF archive not found in manifest: ${cleanArchive}`);
      }

      const totalSize =
        options.totalSize !== undefined && options.totalSize !== null
          ? options.totalSize
          : manifestLookup.entry && Number.isSafeInteger(manifestLookup.entry.size)
            ? manifestLookup.entry.size
            : null;

      const mountPriority =
        options.priority === undefined ? IO_PRIORITY.NORMAL : options.priority;

      const mountPromise = (async () => {
        const record = {
          archivePath: cleanArchive,
          archiveKey,
          vfs: null,
          mountInfo: null,
          mounting: true,
          openPriority: mountPriority,
          pendingPriority: null,
        };

        const rangeReader = (offset, length) => {
          const prio = record.mounting
            ? record.openPriority
            : record.pendingPriority !== null
              ? record.pendingPriority
              : IO_PRIORITY.NORMAL;
          return this._scheduledRead(cacheArchivePath, offset, length, { priority: prio });
        };

        try {
          const vfs = await rpfMod.RpfArchiveVfs.open(rangeReader, totalSize, {
            archivePath: cleanArchive,
            decryptBlock: options.decryptBlock || this.decryptBlock || undefined,
            decompressEntry: options.decompressEntry || this.decompressEntry || undefined,
          });
          record.mounting = false;
          record.vfs = vfs;

          const mountTable = this.ensureRpfMountTable();
          if (mountTable) {
            const mountPrefix =
              typeof options.mountPrefix === 'string' ? options.mountPrefix : cleanArchive;
            record.mountInfo = mountTable.mount(mountPrefix, vfs, {
              priority: options.mountPriority,
            });
          }

          this._mountedArchives.set(archiveKey, record);
          return record;
        } catch (err) {
          this._rpfMountErrors += 1;
          throw err;
        } finally {
          if (this._inFlightMounts.get(archiveKey) === mountPromise) {
            this._inFlightMounts.delete(archiveKey);
          }
        }
      })();

      this._inFlightMounts.set(archiveKey, mountPromise);
      return mountPromise;
    }

    _readFromMountedRpf(record, innerPath, offset, length, priority) {
      // RpfArchiveVfs.readFileRange invokes its _readRange callback synchronously before
      // awaiting, so scoping `pendingPriority` around the call attaches the exact caller
      // priority to the scheduled IoPriorityScheduler task without cross-request races.
      record.pendingPriority = priority === undefined ? IO_PRIORITY.NORMAL : priority;
      try {
        return record.vfs.readFileRange(innerPath, offset, length);
      } finally {
        record.pendingPriority = null;
      }
    }

    _recordReadSuccess(byteLength, elapsedMs) {
      const bytes = Number.isFinite(byteLength) && byteLength > 0 ? Math.floor(byteLength) : 0;
      const dur = Number.isFinite(elapsedMs) && elapsedMs > 0 ? elapsedMs : 0;
      this._streamingStats.readsCompleted += 1;
      this._streamingStats.bytesTransferred += bytes;
      this._streamingStats.totalReadMs += dur;
      if (dur > this._streamingStats.maxReadMs) {
        this._streamingStats.maxReadMs = dur;
      }
    }

    _recordReadError(err, forceCount = false) {
      if (err && typeof err === 'object') {
        if (!forceCount && this._countedErrors.has(err)) {
          return;
        }
        this._countedErrors.add(err);
      }
      this._streamingStats.errorCount += 1;
    }

    async _readRpfRange(pathStr, offset, length, priority) {
      const parsed = parseRpfVirtualPath(pathStr, {
        manifest: this.manifest,
        rpf9VfsModule: this._getRpf9VfsModule(),
      });
      const record = await this.ensureRpfMounted(parsed.archivePath, { priority });
      const mountTable = this.ensureRpfMountTable();
      const resolved = mountTable ? mountTable.resolve(parsed.mountPath) : null;
      if (resolved) {
        if (!resolved.stat.isFile) {
          throw createErrnoError(-21, `Path is a directory, not a file: ${pathStr}`);
        }
        const targetRec =
          (resolved.vfs.archivePath &&
            this._mountedArchives.get(resolved.vfs.archivePath.toLowerCase())) ||
          record;
        return this._readFromMountedRpf(
          targetRec,
          resolved.archivePath,
          offset,
          length,
          priority,
        );
      }
      return this._readFromMountedRpf(record, parsed.innerPath, offset, length, priority);
    }

    async _readCoalescedRangeCacheSpan(cache, relPath, offset, length, maxMergedBytes, signal) {
      if (signal && signal.aborted) {
        const abortErr = createErrnoError(-4, 'The operation was aborted');
        abortErr.name = 'AbortError';
        abortErr.code = 'ABORT_ERR';
        throw abortErr;
      }
      if (length === 0) {
        return new Uint8Array(0);
      }
      const cleanPath = normalizeRelPath(relPath);
      let effectiveLength = length;
      if (cache._fileSizes && cache._fileSizes.has(cleanPath)) {
        const knownTotal = cache._fileSizes.get(cleanPath);
        if (offset >= knownTotal) {
          return new Uint8Array(0);
        }
        effectiveLength = Math.min(length, knownTotal - offset);
      }
      const firstChunk = Math.floor(offset / cache.chunkSize);
      const lastChunk = Math.floor((offset + effectiveLength - 1) / cache.chunkSize);
      const specs = [];
      for (let idx = firstChunk; idx <= lastChunk; idx += 1) {
        const chunkStart = idx * cache.chunkSize;
        const requiredEndInChunk = Math.min(
          cache.chunkSize,
          offset + effectiveLength - chunkStart,
        );
        specs.push({ chunkIndex: idx, requiredEndInChunk });
      }
      const chunks = await cache._getChunksCoalesced(cleanPath, specs, {
        isPrefetch: false,
        countMiss: true,
        countCoalesce: true,
        maxMergedBytes,
      });
      if (signal && signal.aborted) {
        const abortErr = createErrnoError(-4, 'The operation was aborted');
        abortErr.name = 'AbortError';
        abortErr.code = 'ABORT_ERR';
        throw abortErr;
      }
      const slices = [];
      let totalBytes = 0;
      for (let i = 0; i < chunks.length; i += 1) {
        const chunkIndex = firstChunk + i;
        const chunk = chunks[i];
        if (!chunk || chunk.byteLength === 0) {
          break;
        }
        const chunkStart = chunkIndex * cache.chunkSize;
        const sliceStart = Math.max(0, offset - chunkStart);
        const sliceEnd = Math.min(chunk.byteLength, offset + effectiveLength - chunkStart);
        if (sliceEnd > sliceStart) {
          const part = chunk.subarray(sliceStart, sliceEnd);
          slices.push(part);
          totalBytes += part.byteLength;
        }
        if (chunk.byteLength < cache.chunkSize) {
          break;
        }
      }
      const out = new Uint8Array(totalBytes);
      let cursor = 0;
      for (let i = 0; i < slices.length; i += 1) {
        out.set(slices[i], cursor);
        cursor += slices[i].byteLength;
      }
      return out;
    }

    async _fetchBatchGroupSpan(group, options, maxCoalescedBytes) {
      const signal = options && options.signal ? options.signal : null;
      if (signal && signal.aborted) {
        const abortErr = createErrnoError(-4, 'The operation was aborted');
        abortErr.name = 'AbortError';
        abortErr.code = 'ABORT_ERR';
        throw abortErr;
      }
      if (group.route.isRpf) {
        return this._readRpfRange(group.route.url, group.offset, group.length, group.priority);
      }
      if (this.immutableCache && this._shouldUseImmutableCache(group.route)) {
        const hotImmutableBytes = this.immutableCache.tryReadSync(
          group.route.url,
          group.offset,
          group.length,
          { copy: true },
        );
        if (hotImmutableBytes !== null) {
          return hotImmutableBytes;
        }
        if (this.immutableCache.hasInFlightOverlap(group.route.url, group.offset, group.length)) {
          return this.immutableCache.read(group.route.url, group.offset, group.length);
        }
      }
      const cachePath = this._routeToCachePath(group.route);
      const cache = this.ensureRangeCache();
      if (
        group.coalesced &&
        !this._customRangeCacheProvided &&
        cache &&
        !cache.coalesceAdjacent &&
        typeof cache._getChunksCoalesced === 'function' &&
        Number.isInteger(cache.chunkSize) &&
        cache.chunkSize > 0
      ) {
        const firstChunk = Math.floor(group.offset / cache.chunkSize);
        const lastChunk = Math.floor((group.offset + group.length - 1) / cache.chunkSize);
        if (lastChunk > firstChunk) {
          return this._readCoalescedRangeCacheSpan(
            cache,
            cachePath,
            group.offset,
            group.length,
            maxCoalescedBytes,
            signal,
          );
        }
      }
      return this._scheduledRead(cachePath, group.offset, group.length, {
        priority: group.priority,
        signal,
      });
    }

    async read(pathStr, offset, length, options = {}) {
      const priority = options.priority === undefined ? IO_PRIORITY.NORMAL : options.priority;
      const startMs = this._now();

      try {
        let bytes;
        if (isRpfVirtualPath(pathStr)) {
          bytes = await this._readRpfRange(pathStr, offset, length, priority);
        } else {
          const route = this._resolveRoute(pathStr);
          if (!route.ok) {
            throw createErrnoError(-22, route.error);
          }
          if (options.signal && options.signal.aborted) {
            const abortErr = createErrnoError(-4, 'The operation was aborted');
            abortErr.name = 'AbortError';
            abortErr.code = 'ABORT_ERR';
            throw abortErr;
          }
          const hotImmutableBytes =
            this.immutableCache && this._shouldUseImmutableCache(route)
              ? this.immutableCache.tryReadSync(route.url, offset, length, { copy: true })
              : null;
          if (hotImmutableBytes !== null) {
            bytes = hotImmutableBytes;
          } else if (
            this.immutableCache &&
            this._shouldUseImmutableCache(route) &&
            this.immutableCache.hasInFlightOverlap(route.url, offset, length)
          ) {
            bytes = await this.immutableCache.read(route.url, offset, length);
          } else {
            bytes = await this._scheduledRead(this._routeToCachePath(route), offset, length, {
              priority,
              signal: options.signal || null,
            });
          }
        }
        const elapsedMs = Math.max(0, this._now() - startMs);
        this._recordReadSuccess(bytes ? bytes.byteLength : 0, elapsedMs);
        return bytes;
      } catch (err) {
        this._recordReadError(err);
        throw err;
      }
    }

    async readBatch(requests, options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      if (!Array.isArray(requests)) {
        const err = createErrnoError(-22, 'readBatch requests must be an array');
        this._recordReadError(err);
        throw err;
      }

      this._streamingStats.batchedRequests += 1;

      if (requests.length === 0) {
        return {
          ok: true,
          results: [],
          requested: 0,
          succeeded: 0,
          failed: 0,
          fetches: 0,
          coalescedFetches: 0,
          bytesTransferred: 0,
          transferList: [],
        };
      }

      const failFast = Boolean(opts.failFast);
      const isolateBuffers = Boolean(opts.isolateBuffers);
      const maxGapBytes =
        Number.isFinite(opts.maxGapBytes) && opts.maxGapBytes >= 0
          ? Math.floor(opts.maxGapBytes)
          : this._batchCoalesceMaxGapBytes;
      const rawMaxCoalesced =
        Number.isFinite(opts.maxCoalescedBytes) && opts.maxCoalescedBytes > 0
          ? Math.floor(opts.maxCoalescedBytes)
          : this._batchCoalesceMaxBytes;
      const maxCoalescedBytes = Math.max(1, Math.min(rawMaxCoalesced, MAX_COALESCED_FETCH_BYTES));

      const plan = coalesceBatchReadRequests(
        requests,
        Object.assign({}, this._getRouteBases(), {
          path: opts.path,
          priority: opts.priority,
          maxGapBytes,
          maxCoalescedBytes,
        }),
      );

      if (failFast && plan.invalidItems.length > 0) {
        const firstInvalid = plan.invalidItems[0];
        const err = createErrnoError(firstInvalid.errno, firstInvalid.error);
        this._recordReadError(err);
        throw err;
      }

      const results = new Array(requests.length);
      let succeeded = 0;
      let failed = 0;
      let batchCoalescedFetches = 0;
      let batchBytesTransferred = 0;

      const makeSuccessResult = (index, id, bytes, coalesced) => {
        const out = {
          index,
          ok: true,
          bytes,
          byteOffset: bytes.byteOffset,
          byteLength: bytes.byteLength,
          coalesced: Boolean(coalesced),
        };
        if (id !== undefined) {
          out.id = id;
        }
        return out;
      };

      const makeErrorResult = (index, id, errorMsg, errno) => {
        const out = {
          index,
          ok: false,
          bytes: null,
          error: errorMsg,
          errno,
        };
        if (id !== undefined) {
          out.id = id;
        }
        return out;
      };

      for (let i = 0; i < plan.invalidItems.length; i += 1) {
        const inv = plan.invalidItems[i];
        this._streamingStats.errorCount += 1;
        failed += 1;
        results[inv.index] = makeErrorResult(inv.index, inv.id, inv.error, inv.errno);
      }

      for (let i = 0; i < plan.zeroItems.length; i += 1) {
        const z = plan.zeroItems[i];
        const empty = new Uint8Array(0);
        this._recordReadSuccess(0, 0);
        succeeded += 1;
        results[z.index] = makeSuccessResult(z.index, z.id, empty, false);
      }

      const groupTasks = plan.groups.map(async (group) => {
        if (group.coalesced) {
          batchCoalescedFetches += 1;
          this._streamingStats.coalescedFetches += 1;
          this._streamingStats.rangesCoalesced += group.items.length;
        }
        const groupStartMs = this._now();
        try {
          const rawSpan = await this._fetchBatchGroupSpan(group, opts, maxCoalescedBytes);
          const groupElapsedMs = Math.max(0, this._now() - groupStartMs);
          const ownedSpan = ensureTransferableBytes(rawSpan);

          let sharedHighWaterRelEnd = 0;
          for (let i = 0; i < group.items.length; i += 1) {
            const item = group.items[i];
            const relStart = item.offset - group.offset;
            let itemBytes;
            if (relStart >= ownedSpan.byteLength) {
              itemBytes = new Uint8Array(0);
            } else {
              const relEnd = Math.min(ownedSpan.byteLength, relStart + item.length);
              if (
                group.items.length === 1 &&
                relStart === 0 &&
                relEnd === ownedSpan.byteLength
              ) {
                itemBytes = ownedSpan;
                sharedHighWaterRelEnd = relEnd;
              } else if (!isolateBuffers && relStart >= sharedHighWaterRelEnd) {
                // Zero-copy subarray view sharing the coalesced transferable ArrayBuffer.
                itemBytes = ownedSpan.subarray(relStart, relEnd);
                sharedHighWaterRelEnd = relEnd;
              } else {
                // Isolated copy for overlapping ranges or explicit isolateBuffers mode.
                itemBytes = ownedSpan.slice(relStart, relEnd);
              }
            }

            this._recordReadSuccess(itemBytes.byteLength, groupElapsedMs);
            batchBytesTransferred += itemBytes.byteLength;
            succeeded += 1;
            results[item.index] = makeSuccessResult(
              item.index,
              item.id,
              itemBytes,
              group.coalesced,
            );
          }
        } catch (err) {
          if (failFast) {
            this._recordReadError(err);
            throw err;
          }
          const errno = mapErrorToErrno(err, -5);
          const errStr = err && err.message ? err.message : String(err);
          if (err && typeof err === 'object') {
            this._countedErrors.add(err);
          }
          for (let i = 0; i < group.items.length; i += 1) {
            const item = group.items[i];
            this._streamingStats.errorCount += 1;
            failed += 1;
            results[item.index] = makeErrorResult(item.index, item.id, errStr, errno);
          }
        }
      });

      await Promise.all(groupTasks);

      const transferList = collectTransferableBuffers(results);
      return {
        ok: failed === 0,
        results,
        requested: requests.length,
        succeeded,
        failed,
        fetches: plan.groups.length,
        coalescedFetches: batchCoalescedFetches,
        bytesTransferred: batchBytesTransferred,
        transferList,
      };
    }

    async stat(pathStr, options = {}) {
      const priority = options.priority === undefined ? IO_PRIORITY.NORMAL : options.priority;
      if (isRpfVirtualPath(pathStr)) {
        const parsed = parseRpfVirtualPath(pathStr, {
          manifest: this.manifest,
          rpf9VfsModule: this._getRpf9VfsModule(),
          allowEmptyInner: true,
        });
        const record = await this.ensureRpfMounted(parsed.archivePath, { priority });
        const mountTable = this.ensureRpfMountTable();
        const st = mountTable
          ? mountTable.stat(parsed.mountPath)
          : record.vfs.stat(parsed.innerPath);
        if (!st) {
          return null;
        }
        return Object.assign({}, st, {
          archivePath: parsed.archivePath,
          innerPath: parsed.innerPath,
        });
      }

      const route = this._resolveRoute(pathStr);
      if (!route.ok) {
        throw createErrnoError(-22, route.error);
      }
      const cleanPath = route.relPath;
      let manifestLookup =
        route.mount === 'data' || route.mount === 'game'
          ? this._findManifestEntry(cleanPath)
          : { known: null, entry: null };
      if (route.mount === 'game' && manifestLookup.known === false) {
        const prefixed = this._findManifestEntry(`game/${cleanPath}`);
        manifestLookup = prefixed.known === true ? prefixed : { known: null, entry: null };
      }
      if (manifestLookup.known === true) {
        return {
          path: cleanPath,
          kind: 'file',
          isFile: true,
          isDirectory: false,
          size: manifestLookup.entry.size ?? null,
        };
      }
      if (this.immutableCache && this._shouldUseImmutableCache(route)) {
        const cachedMeta = this.immutableCache.getFileMetadata(route.url);
        if (cachedMeta) {
          return {
            path: cleanPath,
            kind: 'file',
            isFile: true,
            isDirectory: false,
            size: cachedMeta.size,
          };
        }
      }
      return null;
    }

    /**
     * Fast 1-roundtrip HTTPFS Stat resolver for `HttpfsOp.Stat` (opcode 5).
     * Returns an exact non-negative integer byte size synchronously (zero network fetches)
     * whenever the file size is cached in `manifest.files`, `_activeWholeFiles`,
     * `immutableCache`, or an open handle; returns `-2` (ENOENT) synchronously when the
     * manifest proves absence on the `/data/` mount; otherwise falls back to a single
     * `HEAD` / `Range: bytes=0-0` probe (never 40-60 binary-search probes).
     */
    statHandle(rawPath) {
      if (typeof rawPath !== 'string' || rawPath.length === 0) {
        return -22; // EINVAL
      }
      if (isRpfVirtualPath(rawPath)) {
        return this.stat(rawPath, { priority: IO_PRIORITY.CRITICAL }).then(
          (info) => {
            if (!info) return -2;
            if (!info.isFile) return -21;
            return Number.isSafeInteger(info.size) && info.size >= 0 ? info.size : -2;
          },
          (err) => mapErrorToErrno(err, -2),
        );
      }

      const route = this._resolveRoute(rawPath);
      if (!route.ok) {
        return -22; // EINVAL
      }

      const cleanPath = route.relPath;
      // 1. In-memory manifest lookup (0 network fetches).
      if (route.mount === 'data' || route.mount === 'game') {
        let manifestLookup = this._findManifestEntry(cleanPath);
        if (route.mount === 'game' && manifestLookup.known === false) {
          const prefixed = this._findManifestEntry(`game/${cleanPath}`);
          manifestLookup = prefixed.known === true ? prefixed : { known: null, entry: null };
        }
        if (
          manifestLookup.known === true &&
          manifestLookup.entry &&
          Number.isSafeInteger(manifestLookup.entry.size) &&
          manifestLookup.entry.size >= 0
        ) {
          if (this._streamingStats) {
            this._streamingStats.statHits = (this._streamingStats.statHits || 0) + 1;
          }
          return manifestLookup.entry.size;
        }
        if (route.mount === 'data' && manifestLookup.known === false) {
          if (this._streamingStats) {
            this._streamingStats.statHits = (this._streamingStats.statHits || 0) + 1;
          }
          return -2; // ENOENT
        }
      }

      // 2. Active whole-file buffer hit (0 network fetches).
      const existingWhole = this._activeWholeFiles.get(route.url);
      if (existingWhole) {
        const size = Number.isSafeInteger(existingWhole.size)
          ? existingWhole.size
          : existingWhole.bytes
            ? existingWhole.bytes.byteLength
            : null;
        if (Number.isSafeInteger(size) && size >= 0) {
          if (this._streamingStats) {
            this._streamingStats.statHits = (this._streamingStats.statHits || 0) + 1;
          }
          return size;
        }
      }

      // 3. ImmutableHttpfsCache metadata / negative cache hit (0 network fetches).
      if (this.immutableCache) {
        const cachedMeta = this.immutableCache.getFileMetadata(route.url);
        if (cachedMeta && Number.isSafeInteger(cachedMeta.size) && cachedMeta.size >= 0) {
          if (this._streamingStats) {
            this._streamingStats.statHits = (this._streamingStats.statHits || 0) + 1;
          }
          return cachedMeta.size;
        }
      }

      // 4. Already-open handle size hit (0 network fetches).
      for (const entry of this._handles.values()) {
        if (
          entry &&
          entry.kind === 'direct' &&
          entry.url === route.url &&
          Number.isSafeInteger(entry.size) &&
          entry.size >= 0
        ) {
          if (this._streamingStats) {
            this._streamingStats.statHits = (this._streamingStats.statHits || 0) + 1;
          }
          return entry.size;
        }
      }

      // 5. Fallback: 1-request probe (ImmutableHttpfsCache.open or single HEAD/Range probe).
      if (this._streamingStats) {
        this._streamingStats.statMisses = (this._streamingStats.statMisses || 0) + 1;
      }
      if (this._shouldUseImmutableCache(route)) {
        const immCache = this.ensureImmutableCache();
        return immCache.open(route.url).then(
          (openRes) => {
            if (!openRes || !openRes.ok) {
              return openRes && typeof openRes.errno === 'number' && openRes.errno < 0
                ? openRes.errno
                : -2;
            }
            return Number.isSafeInteger(openRes.size) && openRes.size >= 0 ? openRes.size : -38;
          },
          (err) => mapErrorToErrno(err, -2),
        );
      }

      const fetchFn = this._getFetchImpl();
      if (typeof fetchFn !== 'function') {
        return -38; // ENOSYS
      }
      return (async () => {
        try {
          const headRes = await fetchFn(route.url, { method: 'HEAD' });
          if (headRes.status === 404) return -2;
          if (headRes.status === 403) return -13;
          if (headRes.ok) {
            const cl = getResponseHeader(headRes.headers, 'Content-Length');
            if (cl !== null && /^\d+$/.test(cl.trim())) {
              const parsed = Number(cl.trim());
              if (Number.isSafeInteger(parsed) && parsed >= 0) {
                return parsed;
              }
            }
          }
          // Fallback to single 1-byte Range probe to read Content-Range total size.
          const rangeRes = await fetchFn(route.url, {
            method: 'GET',
            headers: { Range: 'bytes=0-0' },
          });
          if (rangeRes.status === 404) return -2;
          if (rangeRes.status === 403) return -13;
          if (rangeRes.status === 416 || rangeRes.status === 206) {
            const cr = parseContentRangeHeader(getResponseHeader(rangeRes.headers, 'Content-Range'));
            if (cr && Number.isSafeInteger(cr.total) && cr.total >= 0) {
              return cr.total;
            }
          }
          if (rangeRes.status === 200) {
            const cl = getResponseHeader(rangeRes.headers, 'Content-Length');
            if (cl !== null && /^\d+$/.test(cl.trim())) {
              const parsed = Number(cl.trim());
              if (Number.isSafeInteger(parsed) && parsed >= 0) {
                return parsed;
              }
            }
            const buf = await rangeRes.arrayBuffer();
            return buf.byteLength;
          }
          return -38;
        } catch (err) {
          return mapErrorToErrno(err, -5);
        }
      })();
    }

    async listDir(pathStr, options = {}) {
      const priority = options.priority === undefined ? IO_PRIORITY.NORMAL : options.priority;
      if (!isRpfVirtualPath(pathStr)) {
        throw createErrnoError(-22, `listDir requires an rpf:// virtual path: ${pathStr}`);
      }
      const parsed = parseRpfVirtualPath(pathStr, {
        manifest: this.manifest,
        rpf9VfsModule: this._getRpf9VfsModule(),
        allowEmptyInner: true,
      });
      const record = await this.ensureRpfMounted(parsed.archivePath, { priority });
      const mountTable = this.ensureRpfMountTable();
      if (mountTable) {
        return mountTable.listDir(parsed.mountPath);
      }
      return record.vfs.listDir(parsed.innerPath);
    }

    async openHandle(rawPath, options = {}) {
      const probeOnOpen = options.probeOnOpen !== false;
      const modeStr = typeof options.mode === 'string' ? options.mode.trim().toLowerCase() : '';
      const wantsWrite = Boolean(
        options.write ||
        options.create ||
        options.append ||
        modeStr === 'w' ||
        modeStr === 'wb' ||
        modeStr === 'a' ||
        modeStr === 'ab' ||
        modeStr === 'write' ||
        modeStr === 'create' ||
        modeStr === 'append' ||
        modeStr === 'w+' ||
        modeStr === 'r+' ||
        modeStr === 'a+',
      );

      if (wantsWrite) {
        const dumpRoute = resolveDumpWriteRoute(rawPath, {
          allowDumpWrites: this.allowDumpWrites,
          dumpEndpoint: this.dumpEndpoint,
        });
        if (!dumpRoute.ok) {
          this._dumpWriteStats.dumpWriteErrors += 1;
          return dumpRoute.errno;
        }
        const handle = this._nextHandle;
        this._nextHandle += 1;
        const appendMode = Boolean(
          options.append ||
          modeStr === 'a' ||
          modeStr === 'ab' ||
          modeStr === 'a+' ||
          modeStr === 'append',
        );
        this._handles.set(handle, {
          kind: 'dump_write',
          path: dumpRoute.relPath,
          mount: 'dump',
          url: dumpRoute.url,
          appendUrl: dumpRoute.appendUrl,
          writable: true,
          buffered: Boolean(options.buffered),
          appendOnFirstFlush: appendMode,
          hasFlushed: false,
          bytesWritten: 0,
          pendingChunks: [],
          pendingBytes: 0,
          writeQueue: Promise.resolve(),
        });
        this._dumpWriteStats.dumpHandlesOpened += 1;
        return handle;
      }

      if (isRpfVirtualPath(rawPath)) {
        let parsed;
        try {
          parsed = parseRpfVirtualPath(rawPath, {
            manifest: this.manifest,
            rpf9VfsModule: this._getRpf9VfsModule(),
          });
        } catch (err) {
          return mapErrorToErrno(err, -22);
        }

        let record;
        try {
          record = await this.ensureRpfMounted(parsed.archivePath, {
            priority: IO_PRIORITY.CRITICAL,
          });
        } catch (err) {
          return mapErrorToErrno(err, -2);
        }

        const mountTable = this.ensureRpfMountTable();
        const resolved = mountTable ? mountTable.resolve(parsed.mountPath) : null;
        const stat = resolved ? resolved.stat : record.vfs.stat(parsed.innerPath);

        if (!stat) {
          return -2; // ENOENT
        }
        if (!stat.isFile) {
          return -21; // EISDIR
        }

        const vfs = resolved ? resolved.vfs : record.vfs;
        const innerPath = resolved ? resolved.archivePath : stat.path;
        const mountRecord =
          (vfs.archivePath && this._mountedArchives.get(vfs.archivePath.toLowerCase())) || record;

        const handle = this._nextHandle;
        this._nextHandle += 1;
        this._handles.set(handle, {
          kind: 'rpf',
          rawPath,
          archivePath: parsed.archivePath,
          innerPath,
          mountPath: parsed.mountPath,
          vfs,
          mountRecord,
          size: stat.size,
        });
        return handle;
      }

      const route = this._resolveRoute(rawPath);
      if (!route.ok) {
        return -22; // EINVAL
      }

      const cleanPath = route.relPath;
      const cachePath = this._routeToCachePath(route);
      const manifestLookup =
        route.mount === 'data'
          ? this._findManifestEntry(cleanPath)
          : { known: null, entry: null };
      if (manifestLookup.known === false) {
        return -2; // ENOENT
      }

      const useImmutableCache = this._shouldUseImmutableCache(route);
      const useWholeFileBuffer = this._shouldUseWholeFileBuffer(route);
      let resolvedSize =
        manifestLookup.entry && Number.isSafeInteger(manifestLookup.entry.size)
          ? manifestLookup.entry.size
          : null;
      let wholeFileBytes = null;
      let wholeFilePending = false;
      let alreadyProbed = false;

      if (useWholeFileBuffer) {
        const existingWhole = this._activeWholeFiles.get(route.url);
        if (existingWhole) {
          const rec = this._retainWholeFile(route.url, existingWhole.bytes, existingWhole.size);
          wholeFileBytes = rec.bytes;
          resolvedSize = rec.size;
          alreadyProbed = true;
        } else if (probeOnOpen) {
          const wfRes = await this._fetchWholeFile(route.url, cleanPath, resolvedSize);
          if (!wfRes.ok) {
            return typeof wfRes.errno === 'number' && wfRes.errno < 0 ? wfRes.errno : -2;
          }
          if (wfRes.buffered) {
            const rec = this._retainWholeFile(route.url, wfRes.bytes, wfRes.size);
            wholeFileBytes = rec.bytes;
            resolvedSize = rec.size;
            alreadyProbed = true;
          } else {
            if (Number.isSafeInteger(wfRes.size)) {
              resolvedSize = wfRes.size;
            }
            alreadyProbed = Boolean(wfRes.probed);
          }
        } else {
          wholeFilePending = true;
        }
      }

      if (wholeFileBytes === null && !wholeFilePending) {
        if (useImmutableCache) {
          const immCache = this.ensureImmutableCache();
          if (manifestLookup.known === true) {
            immCache.setFileMetadata(route.url, resolvedSize);
          } else if (probeOnOpen && !alreadyProbed) {
            const openRes = await immCache.open(route.url);
            if (!openRes.ok) {
              return typeof openRes.errno === 'number' && openRes.errno < 0 ? openRes.errno : -2;
            }
            if (Number.isSafeInteger(openRes.size)) {
              resolvedSize = openRes.size;
            }
          } else if (resolvedSize === null) {
            const cachedSize = immCache.getFileSize(route.url);
            if (Number.isSafeInteger(cachedSize)) {
              resolvedSize = cachedSize;
            }
          }
        } else {
          try {
            this.ensureRangeCache();
          } catch {
            return -19; // ENODEV
          }

          if (manifestLookup.known === null && probeOnOpen && !alreadyProbed) {
            try {
              await this._scheduledRead(cachePath, 0, 1, { priority: IO_PRIORITY.CRITICAL });
            } catch (err) {
              return mapErrorToErrno(err, -2);
            }
          }
        }
      }

      const handle = this._nextHandle;
      this._nextHandle += 1;
      this._handles.set(handle, {
        kind: 'direct',
        path: cleanPath,
        cachePath,
        mount: route.mount,
        url: route.url,
        useImmutableCache,
        wholeFileBytes,
        wholeFilePending,
        size: resolvedSize,
      });
      return handle;
    }

    _readHandleFast(handle, offset, reqLen) {
      const entry = this._handles.get(handle);
      if (!entry) {
        this._streamingStats.errorCount += 1;
        return null;
      }
      if (entry.kind === 'dump_write') {
        this._streamingStats.errorCount += 1;
        return null;
      }
      if (reqLen === 0) {
        this._recordReadSuccess(0, 0);
        return new Uint8Array(0);
      }
      const startMs = this._now();
      if (entry.wholeFileBytes !== null && entry.wholeFileBytes !== undefined) {
        this._wholeFileStats.hits += 1;
        const buf = entry.wholeFileBytes;
        const out =
          offset >= buf.byteLength
            ? new Uint8Array(0)
            : buf.subarray(offset, Math.min(buf.byteLength, offset + reqLen));
        this._wholeFileStats.bytesServedFromCache += out.byteLength;
        this._recordReadSuccess(out.byteLength, Math.max(0, this._now() - startMs));
        return out;
      }
      if (
        entry.kind === 'direct' &&
        entry.useImmutableCache &&
        !entry.wholeFilePending &&
        this.immutableCache
      ) {
        const syncBytes = this.immutableCache.tryReadSync(entry.url, offset, reqLen, {
          copy: false,
        });
        if (syncBytes !== null) {
          if (entry.size === null) {
            const learnedSize = this.immutableCache.getFileSize(entry.url);
            if (Number.isSafeInteger(learnedSize)) {
              entry.size = learnedSize;
            }
          }
          this._recordReadSuccess(syncBytes.byteLength, Math.max(0, this._now() - startMs));
          return syncBytes;
        }
      }
      return this.readHandle(handle, offset, reqLen);
    }

    async readHandle(handle, offset, reqLen) {
      const entry = this._handles.get(handle);
      if (!entry) {
        this._streamingStats.errorCount += 1;
        return null;
      }
      if (entry.kind === 'dump_write') {
        const err = createErrnoError(-9, 'Cannot read from write-only /pg6_dump/ handle');
        this._recordReadError(err);
        throw err;
      }
      if (reqLen === 0) {
        this._recordReadSuccess(0, 0);
        return new Uint8Array(0);
      }
      const startMs = this._now();
      try {
        let bytes;
        if (entry.kind === 'rpf') {
          bytes = await this._readFromMountedRpf(
            entry.mountRecord,
            entry.innerPath,
            offset,
            reqLen,
            IO_PRIORITY.CRITICAL,
          );
        } else {
          if (entry.wholeFilePending) {
            entry.wholeFilePending = false;
            const wfRes = await this._fetchWholeFile(entry.url, entry.path, entry.size);
            if (!wfRes.ok) {
              throw createErrnoError(
                typeof wfRes.errno === 'number' && wfRes.errno < 0 ? wfRes.errno : -5,
                `Failed to fetch ${entry.url}`,
              );
            }
            if (wfRes.buffered) {
              const rec = this._retainWholeFile(entry.url, wfRes.bytes, wfRes.size);
              entry.wholeFileBytes = rec.bytes;
              entry.size = rec.size;
            } else if (Number.isSafeInteger(wfRes.size)) {
              entry.size = wfRes.size;
            }
          }
          if (entry.wholeFileBytes !== null && entry.wholeFileBytes !== undefined) {
            this._wholeFileStats.hits += 1;
            const buf = entry.wholeFileBytes;
            if (offset >= buf.byteLength) {
              bytes = new Uint8Array(0);
            } else {
              const end = Math.min(buf.byteLength, offset + reqLen);
              bytes = buf.slice(offset, end);
            }
            this._wholeFileStats.bytesServedFromCache += bytes.byteLength;
          } else if (entry.useImmutableCache) {
            const immCache = this.ensureImmutableCache();
            bytes = await immCache.read(entry.url, offset, reqLen);
            if (entry.size === null) {
              const learnedSize = immCache.getFileSize(entry.url);
              if (Number.isSafeInteger(learnedSize)) {
                entry.size = learnedSize;
              }
            }
          } else {
            bytes = await this._scheduledRead(entry.cachePath || entry.path, offset, reqLen, {
              priority: IO_PRIORITY.CRITICAL,
            });
          }
        }
        this._recordReadSuccess(bytes ? bytes.byteLength : 0, Math.max(0, this._now() - startMs));
        return bytes;
      } catch (err) {
        this._recordReadError(err);
        throw err;
      }
    }

    openWriteHandle(rawPath, options = {}) {
      return this.openHandle(rawPath, Object.assign({}, options, { write: true }));
    }

    async _postDumpBytes(url, appendUrl, bytes, useAppend) {
      const fetchFn = this._getFetchImpl();
      if (typeof fetchFn !== 'function') {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-19, 'No fetch implementation available for dump write');
      }
      const targetUrl = useAppend ? appendUrl : url;
      const body =
        bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
          ? bytes
          : bytes.slice();
      let res;
      try {
        res = await fetchFn(targetUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/octet-stream',
          },
          body,
        });
      } catch (err) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(mapErrorToErrno(err, -5), `Dump POST failed: ${err && err.message ? err.message : err}`);
      }

      const status = res && typeof res.status === 'number' ? res.status : 0;
      if (!res || (!res.ok && (status < 200 || status >= 300))) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        const errno =
          status === 400
            ? -22
            : status === 403 || status === 405
              ? -13
              : status === 404
                ? -2
                : status === 413
                  ? -27
                  : -5;
        const err = createErrnoError(errno, `HTTP ${status || 500} posting dump chunk`);
        err.status = status || 500;
        throw err;
      }

      this._dumpWriteStats.dumpWritesCompleted += 1;
      this._dumpWriteStats.dumpBytesWritten += body.byteLength;
      return {
        ok: true,
        status: status || 200,
        bytesWritten: body.byteLength,
        append: Boolean(useAppend),
        url: targetUrl,
      };
    }

    async _flushDumpBytesChunked(entry, payload, forceAppend = false) {
      const chunkLimit = Math.max(1, this._dumpFlushChunkBytes || DEFAULT_DUMP_FLUSH_CHUNK_BYTES);
      if (payload.byteLength === 0) {
        const useAppend = Boolean(forceAppend || entry.hasFlushed || entry.appendOnFirstFlush);
        const res = await this._postDumpBytes(entry.url, entry.appendUrl, payload, useAppend);
        entry.hasFlushed = true;
        return {
          ok: true,
          bytesWritten: 0,
          totalBytesWritten: entry.bytesWritten,
          chunks: 1,
          append: useAppend,
          url: res.url,
          status: res.status,
        };
      }

      let offset = 0;
      let chunks = 0;
      let firstAppend = false;
      let lastUrl = entry.url;
      let lastStatus = 200;

      while (offset < payload.byteLength) {
        const end = Math.min(payload.byteLength, offset + chunkLimit);
        const slice = payload.subarray(offset, end);
        const useAppend = Boolean(
          forceAppend || chunks > 0 || entry.hasFlushed || entry.appendOnFirstFlush,
        );
        if (chunks === 0) {
          firstAppend = useAppend;
        }
        const res = await this._postDumpBytes(entry.url, entry.appendUrl, slice, useAppend);
        entry.hasFlushed = true;
        entry.bytesWritten += slice.byteLength;
        lastUrl = res.url;
        lastStatus = res.status;
        offset = end;
        chunks += 1;
      }

      return {
        ok: true,
        bytesWritten: payload.byteLength,
        totalBytesWritten: entry.bytesWritten,
        chunks,
        append: firstAppend,
        url: lastUrl,
        status: lastStatus,
      };
    }

    _drainPendingChunks(entry) {
      if (!entry.pendingChunks || entry.pendingChunks.length === 0) {
        entry.pendingBytes = 0;
        return new Uint8Array(0);
      }
      if (entry.pendingChunks.length === 1) {
        const single = entry.pendingChunks[0];
        entry.pendingChunks = [];
        entry.pendingBytes = 0;
        return single;
      }
      const combined = new Uint8Array(entry.pendingBytes);
      let cursor = 0;
      for (let i = 0; i < entry.pendingChunks.length; i += 1) {
        combined.set(entry.pendingChunks[i], cursor);
        cursor += entry.pendingChunks[i].byteLength;
      }
      entry.pendingChunks = [];
      entry.pendingBytes = 0;
      return combined;
    }

    _enqueueDumpOp(entry, fn) {
      const next = entry.writeQueue.then(fn, fn);
      entry.writeQueue = next.then(
        () => {},
        () => {},
      );
      return next;
    }

    async writeHandle(handle, data, options = {}) {
      const entry = this._handles.get(handle);
      if (!entry) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-9, `Invalid file handle: ${handle}`);
      }
      if (entry.kind !== 'dump_write' || !entry.writable) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-30, 'Read-only handle: writes are only permitted under /pg6_dump/');
      }
      if (!this.allowDumpWrites || !this.dumpEndpoint) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-30, 'dump writes are disabled (opt-in required)');
      }

      const payload = toUint8ArrayPayload(data);
      if (payload === null) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-22, 'Dump write data must be a Uint8Array, ArrayBuffer, typed array, or string');
      }

      const shouldBuffer =
        options.flush === false || (entry.buffered && options.flush !== true);
      const forceAppend = Boolean(options.append);

      return this._enqueueDumpOp(entry, async () => {
        if (shouldBuffer) {
          if (payload.byteLength > 0) {
            entry.pendingChunks.push(payload);
            entry.pendingBytes += payload.byteLength;
          }
          if (
            entry.pendingBytes >= this._dumpFlushChunkBytes ||
            entry.pendingBytes >= this._dumpMaxBufferBytes
          ) {
            const drained = this._drainPendingChunks(entry);
            const flushed = await this._flushDumpBytesChunked(entry, drained, forceAppend);
            this._dumpWriteStats.dumpFlushesCompleted += 1;
            return Object.assign({ buffered: false, autoFlushed: true }, flushed);
          }
          return {
            ok: true,
            buffered: true,
            bytesQueued: payload.byteLength,
            pendingBytes: entry.pendingBytes,
            bytesWritten: 0,
            totalBytesWritten: entry.bytesWritten,
          };
        }

        let toSend = payload;
        if (entry.pendingBytes > 0) {
          const prior = this._drainPendingChunks(entry);
          toSend = new Uint8Array(prior.byteLength + payload.byteLength);
          toSend.set(prior, 0);
          toSend.set(payload, prior.byteLength);
        }
        return this._flushDumpBytesChunked(entry, toSend, forceAppend);
      });
    }

    async flushHandle(handle) {
      const entry = this._handles.get(handle);
      if (!entry) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-9, `Invalid file handle: ${handle}`);
      }
      if (entry.kind !== 'dump_write' || !entry.writable) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-30, 'Read-only handle: flush is only permitted on /pg6_dump/ handles');
      }
      return this._enqueueDumpOp(entry, async () => {
        if (entry.pendingBytes === 0 && entry.pendingChunks.length === 0) {
          return {
            ok: true,
            bytesWritten: 0,
            totalBytesWritten: entry.bytesWritten,
            chunks: 0,
          };
        }
        const drained = this._drainPendingChunks(entry);
        const flushed = await this._flushDumpBytesChunked(entry, drained, false);
        this._dumpWriteStats.dumpFlushesCompleted += 1;
        return flushed;
      });
    }

    async closeDumpHandle(handle) {
      const entry = this._handles.get(handle);
      if (!entry) {
        return false;
      }
      if (entry.kind !== 'dump_write') {
        return this.closeHandle(handle);
      }
      try {
        await this.flushHandle(handle);
      } finally {
        this._handles.delete(handle);
      }
      return true;
    }

    async writeDump(rawPath, data, options = {}) {
      const dumpRoute = resolveDumpWriteRoute(rawPath, {
        allowDumpWrites: this.allowDumpWrites,
        dumpEndpoint: this.dumpEndpoint,
      });
      if (!dumpRoute.ok) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(dumpRoute.errno, dumpRoute.error);
      }
      const payload = toUint8ArrayPayload(data);
      if (payload === null) {
        this._dumpWriteStats.dumpWriteErrors += 1;
        throw createErrnoError(-22, 'Dump write data must be a Uint8Array, ArrayBuffer, typed array, or string');
      }
      const tempEntry = {
        url: dumpRoute.url,
        appendUrl: dumpRoute.appendUrl,
        appendOnFirstFlush: Boolean(options.append),
        hasFlushed: false,
        bytesWritten: 0,
      };
      const res = await this._flushDumpBytesChunked(tempEntry, payload, Boolean(options.append));
      return Object.assign({ relPath: dumpRoute.relPath }, res);
    }

    closeHandle(handle) {
      const entry = this._handles.get(handle);
      if (!entry) {
        return false;
      }
      if (entry.kind === 'dump_write' && entry.pendingBytes > 0) {
        const drained = this._drainPendingChunks(entry);
        this._handles.delete(handle);
        this._enqueueDumpOp(entry, () => this._flushDumpBytesChunked(entry, drained, false)).catch(
          () => {},
        );
        return true;
      }
      this._handles.delete(handle);
      if (entry.wholeFileBytes !== null && entry.wholeFileBytes !== undefined) {
        entry.wholeFileBytes = null;
        if (entry.url) {
          this._releaseWholeFile(entry.url);
        }
      }
      return true;
    }

    getHandleInfo(handle) {
      const entry = this._handles.get(handle);
      if (!entry) return null;
      if (entry.kind === 'rpf') {
        return {
          kind: 'rpf',
          archivePath: entry.archivePath,
          innerPath: entry.innerPath,
          size: entry.size,
        };
      }
      if (entry.kind === 'dump_write') {
        return {
          kind: 'dump_write',
          path: entry.path,
          mount: 'dump',
          url: entry.url,
          writable: true,
          bytesWritten: entry.bytesWritten,
          pendingBytes: entry.pendingBytes,
          hasFlushed: entry.hasFlushed,
        };
      }
      return {
        kind: 'direct',
        path: entry.path,
        mount: entry.mount || 'data',
        url: entry.url || null,
        size: entry.size,
        wholeFileBuffered: Boolean(entry.wholeFileBytes),
      };
    }

    async detachMailbox() {
      if (this._mailboxHost) {
        const prev = this._mailboxHost;
        this._mailboxHost = null;
        this._mailboxLanes = 0;
        await prev.stop();
      } else {
        this._mailboxLanes = 0;
      }
      for (const entry of this._handles.values()) {
        if (entry && entry.wholeFileBytes) {
          entry.wholeFileBytes = null;
        }
      }
      this._handles.clear();
      this._activeWholeFiles.clear();
      this._inFlightWholeFiles.clear();
    }

    _createMailboxHostBinding(mailboxMod, options = {}) {
      const probeOnOpen = options.probeOnOpen !== false;
      const BaseHost = mailboxMod.HttpfsMailboxHost;
      const HttpfsStatus = mailboxMod.HttpfsStatus;
      const controller = this;

      class IoWorkerMailboxHost extends BaseHost {
        _handleOpen(lengthBig, cap) {
          if (lengthBig === 0n || lengthBig > BigInt(cap)) {
            this.completeResponse(-22n, HttpfsStatus.Error);
            return;
          }

          const pathLen = Number(lengthBig);
          const copy = new Uint8Array(pathLen);
          copy.set(this._payloadView.subarray(0, pathLen));
          const rawPath = utf8Decoder.decode(copy);
          const openFlags = this._headerView.getBigUint64(mailboxMod.HttpfsOffsets.OFFSET, true);
          const writeFlag = (openFlags & HTTPFS_OPEN_FLAG_WRITE) !== 0n;
          const appendFlag = (openFlags & HTTPFS_OPEN_FLAG_APPEND) !== 0n;

          const pending = controller.openHandle(rawPath, {
            probeOnOpen,
            write: writeFlag,
            append: appendFlag,
          });
          if (pending && typeof pending.then === 'function') {
            return pending.then((handle) => {
              this._finishOpenResult(handle);
            });
          }
          this._finishOpenResult(pending);
        }

        _handleStat(lengthBig, cap) {
          if (lengthBig === 0n || lengthBig > BigInt(cap)) {
            this.completeResponse(-22n, HttpfsStatus.Error);
            return;
          }
          const pathLen = Number(lengthBig);
          const copy = new Uint8Array(pathLen);
          copy.set(this._payloadView.subarray(0, pathLen));
          const rawPath = utf8Decoder.decode(copy);

          const pending = controller.statHandle(rawPath);
          if (pending && typeof pending.then === 'function') {
            return pending.then((sizeOrErrno) => {
              this._finishStatResult(sizeOrErrno);
            });
          }
          this._finishStatResult(pending);
        }

        _handleRead(handle, offsetBig, lengthBig, cap) {
          // If host-side read-ahead was explicitly opted in, preserve base class path verbatim.
          if (this._shared && this._shared.readAhead && this._shared.readAhead.enabled) {
            return super._handleRead(handle, offsetBig, lengthBig, cap);
          }
          if (offsetBig > BigInt(Number.MAX_SAFE_INTEGER)) {
            this.completeResponse(-22n, HttpfsStatus.Error);
            return;
          }
          const offset = Number(offsetBig);
          const reqLen = lengthBig > BigInt(cap) ? cap : Number(lengthBig);
          const entry = controller._handles.get(handle);
          if (!entry || entry.kind === 'dump_write') {
            controller._streamingStats.errorCount += 1;
            this.completeResponse(-9n, HttpfsStatus.Error);
            return;
          }
          if (typeof this._getOrCreateHandleRecord === 'function') {
            const rec = this._getOrCreateHandleRecord(handle, false);
            if (rec) {
              rec.reads += 1;
              rec.lastOffset = offset;
            }
          }
          if (reqLen === 0) {
            controller._recordReadSuccess(0, 0);
            this.completeResponse(0n, HttpfsStatus.ResponseReady);
            return;
          }

          const startMs = controller._now();

          // Direct zero-copy write into SAB payload view for whole-file buffered assets (.pgim / manifests).
          if (entry.wholeFileBytes !== null && entry.wholeFileBytes !== undefined) {
            controller._wholeFileStats.hits += 1;
            const buf = entry.wholeFileBytes;
            let copied = 0;
            if (offset < buf.byteLength) {
              const end = Math.min(buf.byteLength, offset + reqLen);
              copied = end - offset;
              if (copied > 0) {
                this._payloadView.set(buf.subarray(offset, end), 0);
              }
            }
            controller._wholeFileStats.bytesServedFromCache += copied;
            controller._recordReadSuccess(copied, Math.max(0, controller._now() - startMs));
            this.completeResponse(BigInt(copied), HttpfsStatus.ResponseReady);
            return;
          }

          // Direct zero-copy write into SAB payload view for ImmutableHttpfsCache handles (zero slice/Uint8Array allocs).
          if (entry.kind === 'direct' && entry.useImmutableCache && !entry.wholeFilePending) {
            const immCache = controller.immutableCache;
            if (immCache) {
              const syncWritten = immCache.tryReadIntoSync(
                entry.url,
                offset,
                reqLen,
                this._payloadView,
                0,
              );
              if (syncWritten >= 0) {
                if (entry.size === null) {
                  const learnedSize = immCache.getFileSize(entry.url);
                  if (Number.isSafeInteger(learnedSize)) {
                    entry.size = learnedSize;
                  }
                }
                controller._recordReadSuccess(syncWritten, Math.max(0, controller._now() - startMs));
                this.completeResponse(BigInt(syncWritten), HttpfsStatus.ResponseReady);
                return;
              }
            }

            const ensuredImm = controller.ensureImmutableCache();
            return ensuredImm.readInto(entry.url, offset, reqLen, this._payloadView, 0).then(
              (written) => {
                if (entry.size === null) {
                  const learnedSize = ensuredImm.getFileSize(entry.url);
                  if (Number.isSafeInteger(learnedSize)) {
                    entry.size = learnedSize;
                  }
                }
                controller._recordReadSuccess(written, Math.max(0, controller._now() - startMs));
                this.completeResponse(BigInt(written), HttpfsStatus.ResponseReady);
              },
              (err) => {
                controller._recordReadError(err);
                this.completeResponse(BigInt(mapErrorToErrno(err, -5)), HttpfsStatus.Error);
              },
            );
          }

          return super._handleRead(handle, offsetBig, lengthBig, cap);
        }

        _dispatchOnce() {
          const status = Atomics.load(this._statusView, 0);
          if (status !== HttpfsStatus.RequestReady) {
            return false;
          }
          const op = this._headerView.getUint32(mailboxMod.HttpfsOffsets.OP, true);
          if (op === HTTPFS_OP_WRITE) {
            this._stats.requests += 1;
            if (typeof this._slotRequests === 'number') {
              this._slotRequests += 1;
            }
            const handle = this._headerView.getInt32(mailboxMod.HttpfsOffsets.HANDLE, true);
            const offsetBig = this._headerView.getBigUint64(mailboxMod.HttpfsOffsets.OFFSET, true);
            const lengthBig = this._headerView.getBigUint64(mailboxMod.HttpfsOffsets.LENGTH, true);
            const cap = this.payloadCapacity;
            if (lengthBig > BigInt(cap)) {
              controller._dumpWriteStats.dumpWriteErrors += 1;
              this.completeResponse(-22n, HttpfsStatus.Error);
              this._stats.syncFastPathHits += 1;
              if (typeof this._slotSyncFastPathHits === 'number') {
                this._slotSyncFastPathHits += 1;
              }
              return true;
            }
            const writeLen = Number(lengthBig);
            const copy = new Uint8Array(writeLen);
            if (writeLen > 0) {
              copy.set(this._payloadView.subarray(0, writeLen));
            }
            const appendFlag = (offsetBig & HTTPFS_OPEN_FLAG_APPEND) !== 0n;
            this._stats.asyncPathHits += 1;
            if (typeof this._slotAsyncPathHits === 'number') {
              this._slotAsyncPathHits += 1;
            }
            return controller.writeHandle(handle, copy, { append: appendFlag }).then(
              (res) => {
                const written = res && typeof res.bytesWritten === 'number' ? res.bytesWritten : writeLen;
                this.completeResponse(BigInt(written), HttpfsStatus.ResponseReady);
                return true;
              },
              (err) => {
                this.completeResponse(BigInt(mapErrorToErrno(err, -5)), HttpfsStatus.Error);
                return true;
              },
            );
          }
          return super._dispatchOnce();
        }
      }

      const hostOptions = {
        getRangeCache: () => this.ensureRangeCache(),
        getManifest: () => this.manifest,
        probeOnOpen,
        forcePoll: options.forcePoll,
        fallbackPollMs: options.fallbackPollMs,
        open: (relPath) => this.openHandle(relPath, { probeOnOpen }),
        read: (handle, offset, reqLen) => this._readHandleFast(handle, offset, reqLen),
        close: (handle) => {
          const entry = this._handles.get(handle);
          if (entry && entry.kind === 'dump_write') {
            return this.closeDumpHandle(handle);
          }
          return this.closeHandle(handle);
        },
        stat: (relPath) => this.statHandle(relPath),
      };
      if (options.readAhead !== undefined) {
        hostOptions.readAhead = options.readAhead;
      } else if (options.enableReadAhead !== undefined) {
        hostOptions.enableReadAhead = options.enableReadAhead;
      } else {
        // IoWorkerController performs slab/range caching and coalescing directly; disable
        // redundant host-side read-ahead amplification unless explicitly opted in.
        hostOptions.readAhead = false;
      }

      return { IoWorkerMailboxHost, hostOptions };
    }

    async attachMailboxPool(sabs, options = {}) {
      if (!Array.isArray(sabs) || sabs.length < 1) {
        throw new RangeError('attachMailboxPool requires a non-empty array of SharedArrayBuffers');
      }
      const mailboxMod = this._getHttpfsMailboxModule();
      if (
        !mailboxMod ||
        typeof mailboxMod.HttpfsMailboxHost !== 'function' ||
        typeof mailboxMod.HttpfsMailboxPoolHost !== 'function'
      ) {
        throw new Error('PlayGta6HttpfsMailbox.HttpfsMailboxPoolHost unavailable in io_worker');
      }
      await this.detachMailbox();

      const { IoWorkerMailboxHost, hostOptions } = this._createMailboxHostBinding(
        mailboxMod,
        options,
      );
      const PoolHost = mailboxMod.HttpfsMailboxPoolHost;

      class IoWorkerMailboxPoolHost extends PoolHost {
        constructor(slotBuffers, poolOptions) {
          super(slotBuffers, poolOptions);
          this._slots = slotBuffers.map(
            (slotSab) =>
              new IoWorkerMailboxHost(
                slotSab,
                Object.assign({}, poolOptions, { _sharedState: this._shared }),
              ),
          );
        }
      }

      this._mailboxHost = new IoWorkerMailboxPoolHost(sabs, hostOptions);
      this._mailboxLanes = sabs.length;

      this._mailboxHost.start().catch((err) => {
        if (this.postMessage) {
          this.postMessage({ error: `httpfs_mailbox: ${err}` });
        }
      });

      return this._mailboxHost;
    }

    async attachMailbox(sabOrSabs, options = {}) {
      if (Array.isArray(sabOrSabs)) {
        if (sabOrSabs.length === 0) {
          throw new RangeError('attachMailbox requires at least one SharedArrayBuffer');
        }
        if (sabOrSabs.length > 1) {
          return this.attachMailboxPool(sabOrSabs, options);
        }
        sabOrSabs = sabOrSabs[0];
      }

      const mailboxMod = this._getHttpfsMailboxModule();
      if (!mailboxMod || typeof mailboxMod.HttpfsMailboxHost !== 'function') {
        throw new Error('PlayGta6HttpfsMailbox unavailable in io_worker');
      }
      await this.detachMailbox();

      const { IoWorkerMailboxHost, hostOptions } = this._createMailboxHostBinding(
        mailboxMod,
        options,
      );

      this._mailboxHost = new IoWorkerMailboxHost(sabOrSabs, hostOptions);
      this._mailboxLanes = 1;

      this._mailboxHost.start().catch((err) => {
        if (this.postMessage) {
          this.postMessage({ error: `httpfs_mailbox: ${err}` });
        }
      });

      return this._mailboxHost;
    }

    getStats() {
      const cacheStats = this.rangeCache ? this.rangeCache.getStats() : emptyRangeCacheStats();
      const immutableCacheStats = this.immutableCache
        ? this.immutableCache.getStats()
        : emptyImmutableCacheStats({
            chunkSize: this._immutableCacheOptions.chunkSize,
            maxBytes: this._immutableCacheOptions.maxBytes,
            slabSize: this._immutableCacheOptions.slabSize,
            slabMaxBytes: this._immutableCacheOptions.slabMaxBytes,
            slabCacheEnabled: this._immutableCacheOptions.slabCacheEnabled,
            slabProfile: this._immutableCacheOptions.slabProfile,
            prefetchAheadSlabs: this._immutableCacheOptions.prefetchAheadSlabs,
            maxConcurrentSlabFetches: this._immutableCacheOptions.maxConcurrentSlabFetches,
            sequentialThresholdReads: this._immutableCacheOptions.sequentialThresholdReads,
          });
      const mailboxStats =
        this._mailboxHost && typeof this._mailboxHost.getStats === 'function'
          ? this._mailboxHost.getStats()
          : emptyMailboxStats();
      let openDumpHandles = 0;
      let dumpBufferedBytes = 0;
      for (const entry of this._handles.values()) {
        if (entry && entry.kind === 'dump_write') {
          openDumpHandles += 1;
          dumpBufferedBytes += entry.pendingBytes || 0;
        }
      }
      const dumpWriteStats = {
        enabled: Boolean(this.allowDumpWrites && this.dumpEndpoint),
        dumpWritesCompleted: this._dumpWriteStats.dumpWritesCompleted,
        dumpBytesWritten: this._dumpWriteStats.dumpBytesWritten,
        dumpWriteErrors: this._dumpWriteStats.dumpWriteErrors,
        dumpHandlesOpened: this._dumpWriteStats.dumpHandlesOpened,
        dumpFlushesCompleted: this._dumpWriteStats.dumpFlushesCompleted,
        openDumpHandles,
        bufferedBytes: dumpBufferedBytes,
      };
      let activeWholeFileBytes = 0;
      for (const rec of this._activeWholeFiles.values()) {
        if (rec && rec.bytes) {
          activeWholeFileBytes += rec.bytes.byteLength;
        }
      }
      const wholeFileStats = {
        pgimFullFetches: this._wholeFileStats.pgimFullFetches,
        wholeFileFetches: this._wholeFileStats.wholeFileFetches,
        hits: this._wholeFileStats.hits,
        fallbacks: this._wholeFileStats.fallbacks,
        releases: this._wholeFileStats.releases,
        bytesFetched: this._wholeFileStats.bytesFetched,
        bytesServedFromCache: this._wholeFileStats.bytesServedFromCache || 0,
        activeEntries: this._activeWholeFiles.size,
        activeBytes: activeWholeFileBytes,
        maxBytes: this._wholeFileMaxBytes,
      };
      const schedulerStats = this.scheduler ? this.scheduler.getStats() : emptySchedulerStats();
      const mounts = this.rpfMountTable ? this.rpfMountTable.listMounts() : [];
      const mountedArchives = mounts.map((m) => {
        const rec = m.archivePath ? this._mountedArchives.get(m.archivePath.toLowerCase()) : null;
        const header = rec && rec.vfs && rec.vfs.header ? rec.vfs.header : null;
        return {
          id: m.id,
          prefix: m.prefix,
          archivePath: m.archivePath,
          priority: m.priority,
          order: m.order,
          magic: header ? header.magic : null,
          rawMagic: header ? header.rawMagic : null,
          magicEndian: header ? header.magicEndian : null,
          entryCount: header ? header.entryCount : null,
        };
      });

      const totalCacheHits =
        (Number.isFinite(cacheStats.hits) ? cacheStats.hits : 0) +
        (Number.isFinite(immutableCacheStats.hits) ? immutableCacheStats.hits : 0) +
        (Number.isFinite(immutableCacheStats.slabCacheHits) ? immutableCacheStats.slabCacheHits : 0) +
        (Number.isFinite(immutableCacheStats.opfsL2Hits) ? immutableCacheStats.opfsL2Hits : 0) +
        (Number.isFinite(wholeFileStats.hits) ? wholeFileStats.hits : 0);
      const totalCacheMisses =
        (Number.isFinite(cacheStats.misses) ? cacheStats.misses : 0) +
        (Number.isFinite(immutableCacheStats.misses) ? immutableCacheStats.misses : 0) +
        (Number.isFinite(immutableCacheStats.slabCacheMisses)
          ? immutableCacheStats.slabCacheMisses
          : 0);
      const totalInflightCoalesced = Number.isFinite(immutableCacheStats.inFlightCoalesced)
        ? immutableCacheStats.inFlightCoalesced
        : 0;
      const totalPrefetchHits = Number.isFinite(immutableCacheStats.prefetchHits)
        ? immutableCacheStats.prefetchHits
        : 0;
      const totalBytesFetched =
        (Number.isFinite(immutableCacheStats.bytesFetched) ? immutableCacheStats.bytesFetched : 0) +
        (Number.isFinite(wholeFileStats.bytesFetched) ? wholeFileStats.bytesFetched : 0);
      const totalBytesServedFromCache =
        (Number.isFinite(immutableCacheStats.bytesServedFromCache)
          ? immutableCacheStats.bytesServedFromCache
          : 0) +
        (Number.isFinite(wholeFileStats.bytesServedFromCache)
          ? wholeFileStats.bytesServedFromCache
          : 0);
      const readsCompleted = this._streamingStats.readsCompleted;
      const avgReadMs =
        readsCompleted > 0 ? this._streamingStats.totalReadMs / readsCompleted : 0;
      const streamingStats = {
        readsCompleted,
        bytesTransferred: this._streamingStats.bytesTransferred,
        batchedRequests: this._streamingStats.batchedRequests,
        coalescedFetches: this._streamingStats.coalescedFetches,
        rangesCoalesced: this._streamingStats.rangesCoalesced,
        cacheHits: totalCacheHits,
        errorCount: this._streamingStats.errorCount,
        avgReadMs,
        totalReadMs: this._streamingStats.totalReadMs,
        maxReadMs: this._streamingStats.maxReadMs,
      };

      const totalOpfsHits =
        (Number.isFinite(cacheStats.opfsHits) ? cacheStats.opfsHits : 0) +
        (Number.isFinite(immutableCacheStats.opfsL2Hits) ? immutableCacheStats.opfsL2Hits : 0);
      const totalOpfsMisses =
        (Number.isFinite(cacheStats.opfsMisses) ? cacheStats.opfsMisses : 0) +
        (Number.isFinite(immutableCacheStats.opfsL2Misses) ? immutableCacheStats.opfsL2Misses : 0);
      const totalOpfsWrites =
        (Number.isFinite(cacheStats.opfsWrites) ? cacheStats.opfsWrites : 0) +
        (Number.isFinite(immutableCacheStats.opfsL2Writes) ? immutableCacheStats.opfsL2Writes : 0);
      const totalOpfsErrors =
        (Number.isFinite(cacheStats.opfsWriteErrors) ? cacheStats.opfsWriteErrors : 0) +
        (Number.isFinite(immutableCacheStats.opfsL2WriteErrors)
          ? immutableCacheStats.opfsL2WriteErrors
          : 0);

      return {
        // Backwards-compatible top-level RangeCache + aggregate OPFS fields
        hits: cacheStats.hits,
        misses: cacheStats.misses,
        opfsHits: totalOpfsHits,
        opfsMisses: totalOpfsMisses,
        opfsWrites: totalOpfsWrites,
        opfsErrors: totalOpfsErrors,
        evictions: cacheStats.evictions,
        bytesCached: cacheStats.bytesCached,
        entries: cacheStats.entries,
        // Top-level HTTPFS acceleration counters
        pgimFullFetches: wholeFileStats.pgimFullFetches,
        slabCacheHits: immutableCacheStats.slabCacheHits,
        slabCacheMisses: immutableCacheStats.slabCacheMisses,
        opfsL2Hits: immutableCacheStats.opfsL2Hits,
        opfsL2Misses: immutableCacheStats.opfsL2Misses,
        opfsL2Writes: immutableCacheStats.opfsL2Writes,
        opfsL2WriteErrors: immutableCacheStats.opfsL2WriteErrors,
        opfsL2Invalidations: immutableCacheStats.opfsL2Invalidations,
        syncFastPathHits: mailboxStats.syncFastPathHits,
        // Top-level opt-in /pg6_dump/* write forwarding telemetry
        dumpWritesCompleted: dumpWriteStats.dumpWritesCompleted,
        dumpBytesWritten: dumpWriteStats.dumpBytesWritten,
        dumpWriteErrors: dumpWriteStats.dumpWriteErrors,
        // Top-level streaming & batched range dispatch telemetry (W219 / W356)
        readsCompleted: streamingStats.readsCompleted,
        bytesTransferred: streamingStats.bytesTransferred,
        batchedRequests: streamingStats.batchedRequests,
        coalescedFetches: streamingStats.coalescedFetches,
        cacheHits: streamingStats.cacheHits,
        cacheMisses: totalCacheMisses,
        inflightCoalesced: totalInflightCoalesced,
        prefetchHits: totalPrefetchHits,
        bytesFetched: totalBytesFetched,
        bytesServedFromCache: totalBytesServedFromCache,
        errorCount: streamingStats.errorCount,
        avgReadMs: streamingStats.avgReadMs,
        // Detailed subsystem telemetry
        rangeCache: cacheStats,
        immutableCache: immutableCacheStats,
        wholeFileCache: wholeFileStats,
        mailbox: mailboxStats,
        dumpWrites: dumpWriteStats,
        streaming: streamingStats,
        scheduler: schedulerStats,
        rpf: {
          mountedCount: mountedArchives.length,
          archives: mountedArchives,
          mountErrors: this._rpfMountErrors,
        },
        mountedRpfCount: mountedArchives.length,
        openHandles: this._handles.size,
        mailboxAttached: Boolean(this._mailboxHost),
        mailboxLanes: this._mailboxLanes,
        maxConcurrentLanes: this.maxConcurrentLanes,
        perfProfile: this._perfProfileName,
        statHits: this._streamingStats.statHits || 0,
        statMisses: this._streamingStats.statMisses || 0,
      };
    }

    init(options = {}) {
      const initPromise = this._initInternal(options);
      this._initPromise = initPromise;
      initPromise.finally(() => {
        if (this._initPromise === initPromise) {
          this._initPromise = null;
        }
      });
      return initPromise;
    }

    async _initInternal(options = {}) {
      const opts = typeof options === 'string' ? { dataBase: options } : options || {};
      this.cancelPrefetch({ abortInFlight: true });
      for (const entry of this._handles.values()) {
        if (entry && entry.wholeFileBytes) {
          entry.wholeFileBytes = null;
        }
      }
      this._handles.clear();
      this._activeWholeFiles.clear();
      this._inFlightWholeFiles.clear();
      this._wholeFileStats = {
        pgimFullFetches: 0,
        wholeFileFetches: 0,
        hits: 0,
        fallbacks: 0,
        releases: 0,
        bytesFetched: 0,
        bytesServedFromCache: 0,
      };
      this._dumpWriteStats = {
        dumpWritesCompleted: 0,
        dumpBytesWritten: 0,
        dumpWriteErrors: 0,
        dumpHandlesOpened: 0,
        dumpFlushesCompleted: 0,
      };
      this._streamingStats = emptyStreamingStats();
      this._countedErrors = new WeakSet();
      if (typeof opts.now === 'function') {
        this._now = opts.now;
      } else if (typeof opts.nowFn === 'function') {
        this._now = opts.nowFn;
      }
      if (Number.isFinite(opts.batchCoalesceMaxGapBytes) && opts.batchCoalesceMaxGapBytes >= 0) {
        this._batchCoalesceMaxGapBytes = Math.floor(opts.batchCoalesceMaxGapBytes);
      }
      if (Number.isFinite(opts.batchCoalesceMaxBytes) && opts.batchCoalesceMaxBytes > 0) {
        this._batchCoalesceMaxBytes = Math.min(
          Math.floor(opts.batchCoalesceMaxBytes),
          MAX_COALESCED_FETCH_BYTES,
        );
      }
      this._mountedArchives.clear();
      this._inFlightMounts.clear();
      this._rpfMountErrors = 0;
      this._prefetchPromise = null;
      this._prefetchAbortController = null;

      this.dataBase = normalizeDataBase(opts.dataBase !== undefined ? opts.dataBase : this.dataBase);
      if (opts.bootBase !== undefined) {
        this.bootBase = normalizeOptionalBase(opts.bootBase);
      }
      if (opts.gameBase !== undefined) {
        this.gameBase = normalizeOptionalBase(opts.gameBase);
      }
      if (opts.shadersBase !== undefined) {
        this.shadersBase = normalizeOptionalBase(opts.shadersBase);
      }
      if (opts.allowDumpWrites !== undefined || opts.dumpEndpoint !== undefined) {
        const dumpCfg = resolveDumpConfig({
          allowDumpWrites: opts.allowDumpWrites,
          dumpEndpoint: opts.dumpEndpoint,
        });
        this.allowDumpWrites = dumpCfg.allowDumpWrites;
        this.dumpEndpoint = dumpCfg.dumpEndpoint;
      }
      if (Number.isFinite(opts.dumpMaxBufferBytes) && opts.dumpMaxBufferBytes > 0) {
        this._dumpMaxBufferBytes = Math.floor(opts.dumpMaxBufferBytes);
      }
      if (Number.isFinite(opts.dumpFlushChunkBytes) && opts.dumpFlushChunkBytes > 0) {
        this._dumpFlushChunkBytes = Math.min(
          Math.floor(opts.dumpFlushChunkBytes),
          this._dumpMaxBufferBytes,
        );
      }
      if (opts.wholeFileBufferEnabled !== undefined) {
        this._wholeFileBufferEnabled = opts.wholeFileBufferEnabled !== false;
      }
      const nextWholeMax =
        opts.wholeFileMaxBytes !== undefined ? opts.wholeFileMaxBytes : opts.pgimMaxBytes;
      if (Number.isFinite(nextWholeMax) && nextWholeMax >= 0) {
        this._wholeFileMaxBytes = Math.floor(nextWholeMax);
      }
      if (Number.isFinite(opts.wholeFileTotalMaxBytes) && opts.wholeFileTotalMaxBytes >= 0) {
        this._wholeFileTotalMaxBytes = Math.floor(opts.wholeFileTotalMaxBytes);
      } else if (Number.isFinite(nextWholeMax) && nextWholeMax >= 0) {
        this._wholeFileTotalMaxBytes = Math.max(
          this._wholeFileTotalMaxBytes,
          this._wholeFileMaxBytes,
        );
      }
      if (typeof opts.fetchImpl === 'function') {
        this.fetchImpl = opts.fetchImpl;
      }
      if (typeof opts.decryptBlock === 'function') {
        this.decryptBlock = opts.decryptBlock;
      }
      if (typeof opts.decompressEntry === 'function') {
        this.decompressEntry = opts.decompressEntry;
      }

      this.manifest = opts.manifest || null;

      const rangeCacheMod = this._getRangeCacheModule();
      if (!rangeCacheMod || typeof rangeCacheMod.RangeCache !== 'function') {
        throw new Error('PlayGta6RangeCache unavailable in io_worker');
      }

      const resolvedOpfsL2Enabled =
        opts.opfsL2Enabled !== undefined
          ? opts.opfsL2Enabled
          : this._immutableCacheOptions.opfsL2Enabled;

      let opfsDir = null;
      if (resolvedOpfsL2Enabled === false) {
        opfsDir = null;
      } else if (opts.opfsDir !== undefined) {
        opfsDir = opts.opfsDir;
      } else if (this._cacheOptions.opfsDir !== undefined) {
        opfsDir = this._cacheOptions.opfsDir;
      } else if (typeof rangeCacheMod.openOpfsCacheDir === 'function') {
        opfsDir = await rangeCacheMod.openOpfsCacheDir('playgta6-cache');
      }

      const fetchFn = this._getFetchImpl(opts.fetchImpl);
      const routedFetchFn = this._createRoutedFetch(fetchFn);

      if (opts.immutableCacheEnabled !== undefined) {
        this._immutableCacheEnabled = opts.immutableCacheEnabled !== false;
      }
      this._customRangeCacheProvided = Boolean(opts.rangeCache);
      this._hasRoutedRangeCache = !opts.rangeCache;
      const initPerfProfile =
        opts.perfProfile !== undefined && opts.perfProfile !== null
          ? coerceIoPerfProfile(opts.perfProfile, this._perfProfileModule)
          : null;
      if (initPerfProfile) {
        this._perfProfile = initPerfProfile;
        this._perfProfileName = initPerfProfile.name || null;
        if (
          Number.isInteger(initPerfProfile.ioMaxConcurrentLanes) &&
          initPerfProfile.ioMaxConcurrentLanes > 0
        ) {
          this._maxMailboxLanes = initPerfProfile.ioMaxConcurrentLanes;
        }
      }

      const resolvedSlabSize =
        opts.slabSize !== undefined
          ? opts.slabSize
          : opts.immutableSlabSize !== undefined
            ? opts.immutableSlabSize
            : opts.blockSize !== undefined
              ? opts.blockSize
              : this._immutableCacheOptions.slabSize;
      const resolvedSlabMaxBytes =
        opts.slabMaxBytes !== undefined
          ? opts.slabMaxBytes
          : opts.immutableSlabMaxBytes !== undefined
            ? opts.immutableSlabMaxBytes
            : opts.maxRamCacheBytes !== undefined
              ? opts.maxRamCacheBytes
              : Number.isFinite(opts.recommendedSlabMaxBytes) &&
                  opts.recommendedSlabMaxBytes > 0
                ? opts.recommendedSlabMaxBytes
                : initPerfProfile
                  ? initPerfProfile.ioSlabCacheBytes
                  : this._immutableCacheOptions.slabMaxBytes;
      const resolvedSlabEnabled =
        opts.slabCacheEnabled !== undefined
          ? opts.slabCacheEnabled
          : this._immutableCacheOptions.slabCacheEnabled;
      const resolvedSlabProfile =
        opts.slabProfile !== undefined
          ? opts.slabProfile
          : this._immutableCacheOptions.slabProfile;
      const resolvedPrefetchAhead =
        opts.prefetchAheadSlabs !== undefined
          ? opts.prefetchAheadSlabs
          : this._immutableCacheOptions.prefetchAheadSlabs;
      const resolvedMaxConcurrentSlabFetches =
        opts.maxConcurrentSlabFetches !== undefined
          ? opts.maxConcurrentSlabFetches
          : initPerfProfile
            ? initPerfProfile.ioMaxConcurrentLanes
            : this._immutableCacheOptions.maxConcurrentSlabFetches;
      const resolvedSequentialThreshold =
        opts.sequentialThresholdReads !== undefined
          ? opts.sequentialThresholdReads
          : this._immutableCacheOptions.sequentialThresholdReads;

      if (initPerfProfile) {
        this._immutableCacheOptions.slabMaxBytes = resolvedSlabMaxBytes;
        this._immutableCacheOptions.maxConcurrentSlabFetches = resolvedMaxConcurrentSlabFetches;
        if (opts.maxPrefetchConcurrency === undefined) {
          this._cacheOptions.maxPrefetchConcurrency = initPerfProfile.ioMaxConcurrentLanes;
        }
        if (opts.maxConcurrency === undefined) {
          this._schedulerOptions.maxConcurrency = initPerfProfile.ioMaxConcurrentLanes;
        }
      }

      const resolvedOpfsL2 =
        opts.opfsL2 !== undefined ? opts.opfsL2 : this._immutableCacheOptions.opfsL2;
      const resolvedOpfsCache =
        opts.opfsCache !== undefined ? opts.opfsCache : this._immutableCacheOptions.opfsCache;
      const resolvedOpfsL2Allowlist =
        opts.opfsL2Allowlist !== undefined
          ? opts.opfsL2Allowlist
          : opts.hotOpfsAllowlist !== undefined
            ? opts.hotOpfsAllowlist
            : opts.opfsHotRpfAllowlist !== undefined
              ? opts.opfsHotRpfAllowlist
              : this._immutableCacheOptions.opfsL2Allowlist;
      const resolvedOpfsL2RequireEtag =
        opts.opfsL2RequireEtag !== undefined
          ? opts.opfsL2RequireEtag
          : this._immutableCacheOptions.opfsL2RequireEtag;
      const resolvedFileFingerprints =
        opts.fileFingerprints !== undefined
          ? opts.fileFingerprints
          : this._immutableCacheOptions.fileFingerprints;
      const resolvedMaxPendingOpfsL2Writes =
        opts.maxPendingOpfsL2Writes !== undefined
          ? opts.maxPendingOpfsL2Writes
          : this._immutableCacheOptions.maxPendingOpfsL2Writes;
      const resolvedMaxPendingOpfsL2WriteBytes =
        opts.maxPendingOpfsL2WriteBytes !== undefined
          ? opts.maxPendingOpfsL2WriteBytes
          : this._immutableCacheOptions.maxPendingOpfsL2WriteBytes;
      const resolvedMaxOpfsL2WriteConcurrency =
        opts.maxOpfsL2WriteConcurrency !== undefined
          ? opts.maxOpfsL2WriteConcurrency
          : this._immutableCacheOptions.maxOpfsL2WriteConcurrency;
      const resolvedOpfsL2MaxBytes =
        opts.opfsL2MaxBytes !== undefined
          ? opts.opfsL2MaxBytes
          : this._immutableCacheOptions.opfsL2MaxBytes;

      this._cacheOptions.opfsDir = opfsDir;
      this._immutableCacheOptions.opfsDir = opfsDir;
      this._immutableCacheOptions.opfsL2 = resolvedOpfsL2;
      this._immutableCacheOptions.opfsCache = resolvedOpfsCache;
      this._immutableCacheOptions.opfsL2Enabled = resolvedOpfsL2Enabled;
      this._immutableCacheOptions.opfsL2Allowlist = resolvedOpfsL2Allowlist;
      this._immutableCacheOptions.opfsL2RequireEtag = resolvedOpfsL2RequireEtag;
      this._immutableCacheOptions.fileFingerprints = resolvedFileFingerprints;
      this._immutableCacheOptions.maxPendingOpfsL2Writes = resolvedMaxPendingOpfsL2Writes;
      this._immutableCacheOptions.maxPendingOpfsL2WriteBytes = resolvedMaxPendingOpfsL2WriteBytes;
      this._immutableCacheOptions.maxOpfsL2WriteConcurrency = resolvedMaxOpfsL2WriteConcurrency;
      this._immutableCacheOptions.opfsL2MaxBytes = resolvedOpfsL2MaxBytes;

      this.immutableCache =
        opts.immutableCache ||
        new ImmutableHttpfsCache({
          chunkSize:
            opts.immutableCacheChunkSize !== undefined
              ? opts.immutableCacheChunkSize
              : this._immutableCacheOptions.chunkSize,
          maxBytes:
            opts.immutableCacheMaxBytes !== undefined
              ? opts.immutableCacheMaxBytes
              : this._immutableCacheOptions.maxBytes,
          slabSize: resolvedSlabSize,
          slabMaxBytes: resolvedSlabMaxBytes,
          slabCacheEnabled: resolvedSlabEnabled,
          slabProfile: resolvedSlabProfile,
          prefetchAheadSlabs: resolvedPrefetchAhead,
          maxConcurrentSlabFetches: resolvedMaxConcurrentSlabFetches,
          sequentialThresholdReads: resolvedSequentialThreshold,
          negativeCacheTtlMs:
            opts.negativeCacheTtlMs !== undefined
              ? opts.negativeCacheTtlMs
              : this._immutableCacheOptions.negativeCacheTtlMs,
          negativeCacheMaxEntries:
            opts.negativeCacheMaxEntries !== undefined
              ? opts.negativeCacheMaxEntries
              : this._immutableCacheOptions.negativeCacheMaxEntries,
          metadataMaxEntries:
            opts.metadataMaxEntries !== undefined
              ? opts.metadataMaxEntries
              : this._immutableCacheOptions.metadataMaxEntries,
          opfsL2: resolvedOpfsL2,
          opfsCache: resolvedOpfsCache,
          opfsDir,
          opfsL2Enabled: resolvedOpfsL2Enabled,
          opfsL2Allowlist: resolvedOpfsL2Allowlist,
          opfsL2RequireEtag: resolvedOpfsL2RequireEtag,
          fileFingerprints: resolvedFileFingerprints,
          maxPendingOpfsL2Writes: resolvedMaxPendingOpfsL2Writes,
          maxPendingOpfsL2WriteBytes: resolvedMaxPendingOpfsL2WriteBytes,
          maxOpfsL2WriteConcurrency: resolvedMaxOpfsL2WriteConcurrency,
          opfsL2MaxBytes: resolvedOpfsL2MaxBytes,
          perfProfile: this._perfProfile,
          now:
            typeof opts.now === 'function' ? opts.now : this._immutableCacheOptions.now,
          fetchImpl: fetchFn,
        });
      if (
        resolvedOpfsL2Enabled !== false &&
        opfsDir &&
        this.immutableCache &&
        typeof this.immutableCache.warmOpfsIndex === 'function'
      ) {
        // Hydrate the O(1) OPFS directory index in the background so warm-boot reads
        // immediately recognize persisted slabs without adding cold-boot startup latency.
        this.immutableCache.warmOpfsIndex().catch(() => {});
      }
      const initRangeCacheOpts = {
        dataBase: this.dataBase,
        chunkSize:
          opts.chunkSize !== undefined ? opts.chunkSize : this._cacheOptions.chunkSize,
        maxBytes: opts.maxBytes !== undefined ? opts.maxBytes : this._cacheOptions.maxBytes,
        maxPrefetchConcurrency:
          opts.maxPrefetchConcurrency !== undefined
            ? opts.maxPrefetchConcurrency
            : this._cacheOptions.maxPrefetchConcurrency,
        opfsDir,
        fetchImpl: routedFetchFn,
      };
      const resolvedCoalesceAdjacent =
        typeof opts.coalesceAdjacent === 'boolean'
          ? opts.coalesceAdjacent
          : this._cacheOptions.coalesceAdjacent;
      if (typeof resolvedCoalesceAdjacent === 'boolean') {
        initRangeCacheOpts.coalesceAdjacent = resolvedCoalesceAdjacent;
      }
      const resolvedCoalesceMaxGap =
        opts.coalesceMaxGapBytes !== undefined
          ? opts.coalesceMaxGapBytes
          : this._cacheOptions.coalesceMaxGapBytes;
      if (resolvedCoalesceMaxGap !== undefined) {
        initRangeCacheOpts.coalesceMaxGapBytes = resolvedCoalesceMaxGap;
      }
      const resolvedCoalesceMaxMerged =
        opts.coalesceMaxMergedBytes !== undefined
          ? opts.coalesceMaxMergedBytes
          : this._cacheOptions.coalesceMaxMergedBytes;
      if (resolvedCoalesceMaxMerged !== undefined) {
        initRangeCacheOpts.coalesceMaxMergedBytes = resolvedCoalesceMaxMerged;
      }
      this.rangeCache = opts.rangeCache || new rangeCacheMod.RangeCache(initRangeCacheOpts);

      const schedulerMod = this._getIoSchedulerModule();
      if (opts.scheduler) {
        this.scheduler = opts.scheduler;
      } else if (schedulerMod && typeof schedulerMod.IoPriorityScheduler === 'function') {
        const sleepFn =
          typeof opts.sleep === 'function'
            ? opts.sleep
            : typeof this._schedulerOptions.sleep === 'function'
              ? this._schedulerOptions.sleep
              : typeof setTimeout === 'function'
                ? undefined
                : () => Promise.resolve();
        this.scheduler = new schedulerMod.IoPriorityScheduler({
          rangeCache: this.rangeCache,
          maxConcurrency:
            opts.maxConcurrency !== undefined
              ? opts.maxConcurrency
              : this._schedulerOptions.maxConcurrency,
          maxRetries:
            opts.maxRetries !== undefined ? opts.maxRetries : this._schedulerOptions.maxRetries,
          baseRetryDelayMs:
            opts.baseRetryDelayMs !== undefined
              ? opts.baseRetryDelayMs
              : this._schedulerOptions.baseRetryDelayMs,
          sleep: sleepFn,
        });
      } else {
        this.scheduler = null;
      }

      const rpfMod = this._getRpf9VfsModule();
      if (opts.rpfMountTable) {
        this.rpfMountTable = opts.rpfMountTable;
      } else if (rpfMod && typeof rpfMod.RpfVfsMountTable === 'function') {
        this.rpfMountTable = new rpfMod.RpfVfsMountTable();
      } else {
        this.rpfMountTable = null;
      }

      let manifestPart = 'no data-manifest.json yet';
      if (this.manifest) {
        manifestPart = `${this.manifest.file_count || (Array.isArray(this.manifest.files) ? this.manifest.files.length : 0)} files indexed`;
      } else if (fetchFn) {
        try {
          const res = await fetchFn(`${this.dataBase}data-manifest.json`);
          if (res.ok) {
            this.manifest = await res.json();
            manifestPart = `${this.manifest.file_count || (Array.isArray(this.manifest.files) ? this.manifest.files.length : 0)} files indexed`;
          }
        } catch {
          // Manifest is optional until the user generates local-data/data-manifest.json.
        }
      }

      let bootsetCount = 0;
      let bootEntries = null;
      if (opts.bootset !== undefined) {
        bootEntries = Array.isArray(opts.bootset)
          ? opts.bootset
          : opts.bootset && Array.isArray(opts.bootset.entries)
            ? opts.bootset.entries
            : [];
      } else if (fetchFn) {
        try {
          const bootRes = await fetchFn(`${this.dataBase}bootset.json`);
          if (bootRes.ok) {
            const bootJson = await bootRes.json();
            bootEntries = Array.isArray(bootJson)
              ? bootJson
              : bootJson && Array.isArray(bootJson.entries)
                ? bootJson.entries
                : [];
          }
        } catch {
          // bootset.json is optional; absence must never fail boot.
        }
      }

      if (Array.isArray(bootEntries) && bootEntries.length > 0) {
        bootsetCount = bootEntries.length;
        // Prefetch hot initial ranges at lowest (PREFETCH) priority without blocking worker readiness.
        this.prefetchBootset(bootEntries).catch(() => {});
      }

      const cacheTier = opfsDir ? 'OPFS+LRU' : 'memory LRU';
      const bootPart = bootsetCount > 0 ? `, prefetching ${bootsetCount} bootset ranges` : '';
      const summary = `${manifestPart} (${cacheTier}${bootPart})`;

      const readyPayload = {
        ready: true,
        summary,
        opfs: Boolean(opfsDir),
        bootsetEntries: bootsetCount,
        scheduler: Boolean(this.scheduler),
        rpfVfs: Boolean(this.rpfMountTable),
        dataBase: this.dataBase,
        bootBase: this.bootBase,
        gameBase: this.gameBase,
        shadersBase: this.shadersBase,
        allowDumpWrites: this.allowDumpWrites,
        dumpEndpoint: this.dumpEndpoint,
        perfProfile: this._perfProfileName,
        maxConcurrentLanes: this.maxConcurrentLanes,
      };

      if (this.postMessage && !opts._deferReady) {
        this.postMessage(readyPayload);
      }

      return readyPayload;
    }

    async handleMessage(msg) {
      const m = msg || {};
      try {
        if (m.init) {
          const hasInlineMailbox =
            (Array.isArray(m.mailboxSabs) && m.mailboxSabs.length > 0) || Boolean(m.mailboxSab);
          if (hasInlineMailbox) {
            const readyPayload = await this.init(Object.assign({}, m, { _deferReady: true }));
            if (Array.isArray(m.mailboxSabs) && m.mailboxSabs.length > 0) {
              await this.attachMailbox(m.mailboxSabs, m);
            } else if (m.mailboxSab) {
              await this.attachMailbox(m.mailboxSab, m);
            }
            if (this.postMessage) {
              this.postMessage(readyPayload);
            }
          } else {
            await this.init(m);
          }
        } else if (m.attachMailboxPool) {
          if (this._initPromise) {
            try {
              await this._initPromise;
            } catch {
              // Proceed so mailbox attachment still reports an explicit result if init failed.
            }
          }
          const poolSabs = m.sabs || m.mailboxSabs || m.sab || m.mailboxSab;
          await this.attachMailboxPool(poolSabs, m);
          const reply = { mailboxAttached: true, mailboxLanes: this._mailboxLanes };
          if (m.id !== undefined) reply.id = m.id;
          if (this.postMessage) this.postMessage(reply);
        } else if (m.attachMailbox) {
          if (this._initPromise) {
            try {
              await this._initPromise;
            } catch {
              // Proceed so mailbox attachment still reports an explicit result if init failed.
            }
          }
          const target =
            Array.isArray(m.sabs) && m.sabs.length > 0
              ? m.sabs
              : Array.isArray(m.mailboxSabs) && m.mailboxSabs.length > 0
                ? m.mailboxSabs
                : m.sab || m.mailboxSab;
          await this.attachMailbox(target, m);
          const reply = { mailboxAttached: true, mailboxLanes: this._mailboxLanes };
          if (m.id !== undefined) reply.id = m.id;
          if (this.postMessage) this.postMessage(reply);
        } else if (m.detachMailbox) {
          await this.detachMailbox();
          const reply = { mailboxDetached: true };
          if (m.id !== undefined) reply.id = m.id;
          if (this.postMessage) this.postMessage(reply);
        } else if (m.read) {
          const bytes = await this.read(m.path, m.offset, m.length, {
            priority: m.priority,
          });
          const transferable = ensureTransferableBytes(bytes);
          if (this.postMessage) {
            this.postMessage({ id: m.id, bytes: transferable }, [transferable.buffer]);
          }
        } else if (m.readBatch || m.batchRead) {
          const rawRequests = Array.isArray(m.requests)
            ? m.requests
            : Array.isArray(m.ranges)
              ? m.ranges
              : Array.isArray(m.reads)
                ? m.reads
                : Array.isArray(m.readBatch)
                  ? m.readBatch
                  : Array.isArray(m.batchRead)
                    ? m.batchRead
                    : null;
          const batch = await this.readBatch(rawRequests, {
            path: m.path,
            priority: m.priority,
            maxGapBytes: m.maxGapBytes,
            maxCoalescedBytes: m.maxCoalescedBytes,
            failFast: Boolean(m.failFast),
            isolateBuffers: Boolean(m.isolateBuffers),
          });
          if (this.postMessage) {
            const payload = {
              id: m.id,
              ok: batch.ok,
              results: batch.results,
              requested: batch.requested,
              succeeded: batch.succeeded,
              failed: batch.failed,
              fetches: batch.fetches,
              coalescedFetches: batch.coalescedFetches,
              bytesTransferred: batch.bytesTransferred,
            };
            if (batch.transferList.length > 0) {
              this.postMessage(payload, batch.transferList);
            } else {
              this.postMessage(payload);
            }
          }
        } else if (m.openWrite || m.openDump) {
          const handle = await this.openWriteHandle(m.path, {
            append: Boolean(m.append),
            buffered: Boolean(m.buffered),
            mode: m.mode,
          });
          if (this.postMessage) {
            if (handle < 0) {
              this.postMessage({
                id: m.id,
                ok: false,
                handle,
                errno: handle,
                error: handle === -22 ? 'path traversal or invalid dump path' : 'dump write refused (read-only or disabled)',
              });
            } else {
              this.postMessage({
                id: m.id,
                ok: true,
                handle,
                info: this.getHandleInfo(handle),
              });
            }
          }
        } else if (m.writeDump || m.write) {
          const payload = m.bytes !== undefined ? m.bytes : m.data;
          const result =
            m.handle !== undefined
              ? await this.writeHandle(m.handle, payload, {
                  append: m.append,
                  flush: m.flush,
                })
              : await this.writeDump(m.path, payload, { append: m.append });
          if (this.postMessage) {
            this.postMessage(Object.assign({ id: m.id }, result));
          }
        } else if (m.flushDump) {
          const result = await this.flushHandle(m.handle);
          if (this.postMessage) {
            this.postMessage(Object.assign({ id: m.id }, result));
          }
        } else if (m.closeDump) {
          const closed = await this.closeDumpHandle(m.handle);
          if (this.postMessage) {
            this.postMessage({ id: m.id, ok: closed, closed });
          }
        } else if (
          m.reclaimMemory ||
          m.trimCache ||
          m.type === 'memory-pressure' ||
          m.type === 'trim-transient-memory' ||
          m.type === 'memory-reclaim'
        ) {
          const reclaimOpts =
            m.reclaimMemory && typeof m.reclaimMemory === 'object'
              ? m.reclaimMemory
              : m.trimCache && typeof m.trimCache === 'object'
                ? m.trimCache
                : m;
          const result = this.reclaimMemory(reclaimOpts);
          if (this.postMessage) {
            this.postMessage(Object.assign({ id: m.id }, result));
          }
        } else if (m.tuneSlabCache) {
          const target =
            typeof m.tuneSlabCache === 'string'
              ? m.tuneSlabCache
              : m.profile || m.config || m;
          const slabConfig = this.tuneSlabCache(target);
          if (this.postMessage) {
            this.postMessage({ id: m.id, ok: true, slabConfig });
          }
        } else if (m.applyPerfProfile || m.type === 'apply-perf-profile') {
          const target =
            typeof m.applyPerfProfile === 'string' ||
            (m.applyPerfProfile && typeof m.applyPerfProfile === 'object')
              ? m.applyPerfProfile
              : m.perfProfile || m.profile || m.preset || m;
          const perfProfile = this.applyPerfProfile(target);
          if (this.postMessage) {
            this.postMessage({ id: m.id, ok: true, perfProfile });
          }
        } else if (m.prefetchSlabs) {
          const prefetch = await this.prefetchVideoSlabs(m.path, m.offset || 0, m.count);
          if (this.postMessage) {
            this.postMessage({ id: m.id, ok: true, prefetch });
          }
        } else if (m.stat) {
          const info = await this.stat(m.path, { priority: m.priority });
          if (this.postMessage) {
            this.postMessage({ id: m.id, stat: info });
          }
        } else if (m.listDir) {
          const entries = await this.listDir(m.path, { priority: m.priority });
          if (this.postMessage) {
            this.postMessage({ id: m.id, entries });
          }
        } else if (m.mountRpf) {
          const rec = await this.ensureRpfMounted(m.archivePath || m.path, {
            mountPrefix: m.mountPrefix,
            totalSize: m.totalSize,
            priority: m.priority,
            mountPriority: m.mountPriority,
          });
          if (this.postMessage) {
            this.postMessage({
              id: m.id,
              mounted: {
                archivePath: rec.archivePath,
                mountInfo: rec.mountInfo,
                magic: rec.vfs && rec.vfs.header ? rec.vfs.header.magic : null,
                rawMagic: rec.vfs && rec.vfs.header ? rec.vfs.header.rawMagic : null,
                entryCount: rec.vfs && rec.vfs.header ? rec.vfs.header.entryCount : 0,
              },
            });
          }
        } else if (m.cancelPrefetch) {
          const cancelled = this.cancelPrefetch({ abortInFlight: Boolean(m.abortInFlight) });
          if (this.postMessage) {
            this.postMessage({ id: m.id, cancelled });
          }
        } else if (m.stats) {
          const stats = this.getStats();
          if (this.postMessage) {
            this.postMessage({ id: m.id, stats });
          }
        }
      } catch (err) {
        this._recordReadError(err);
        if (this.postMessage) {
          const errPayload = { id: m.id, error: String(err) };
          if (err && typeof err.errno === 'number') {
            errPayload.errno = err.errno;
          }
          this.postMessage(errPayload);
        }
      }
    }
  }

  function createIoWorkerHandler(options = {}) {
    const controller = new IoWorkerController(options);
    const handler = (evOrMsg) => {
      const data =
        evOrMsg &&
        typeof evOrMsg === 'object' &&
        Object.prototype.hasOwnProperty.call(evOrMsg, 'data') &&
        evOrMsg.data &&
        typeof evOrMsg.data === 'object'
          ? evOrMsg.data
          : evOrMsg;
      return controller.handleMessage(data);
    };
    handler.controller = controller;
    return handler;
  }

  const defaultController = new IoWorkerController({
    postMessage:
      typeof self !== 'undefined' && typeof self.postMessage === 'function'
        ? (msg, transfer) =>
            transfer ? self.postMessage(msg, transfer) : self.postMessage(msg)
        : null,
  });

  if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
    self.onmessage = (ev) => defaultController.handleMessage(ev && ev.data);
  }

  const exportedApi = {
    IO_PRIORITY,
    RPF_URI_PREFIX,
    DUMP_VIRTUAL_PREFIX,
    DEFAULT_DUMP_ENDPOINT,
    DEFAULT_DUMP_MAX_BUFFER_BYTES,
    DEFAULT_DUMP_FLUSH_CHUNK_BYTES,
    HTTPFS_OP_WRITE,
    HTTPFS_OPEN_FLAG_WRITE,
    HTTPFS_OPEN_FLAG_APPEND,
    DEFAULT_IMMUTABLE_CHUNK_SIZE,
    DEFAULT_IMMUTABLE_MAX_BYTES,
    DEFAULT_WHOLE_FILE_MAX_BYTES,
    DEFAULT_WHOLE_FILE_TOTAL_MAX_BYTES,
    DEFAULT_SLAB_SIZE,
    DEFAULT_SLAB_MAX_BYTES,
    MAX_SLAB_SIZE,
    MAX_SLAB_CACHE_BYTES,
    DEFAULT_SLAB_PREFETCH_AHEAD,
    DEFAULT_AUTO_PREFETCH_MIN_SLAB_SIZE,
    DEFAULT_MAX_CONCURRENT_SLAB_FETCHES,
    DEFAULT_SEQUENTIAL_THRESHOLD_READS,
    SLAB_TUNING_PROFILES,
    VIDEO_STREAM_SLAB_PRESET,
    DEFAULT_NEGATIVE_CACHE_TTL_MS,
    DEFAULT_BATCH_COALESCE_MAX_GAP_BYTES,
    DEFAULT_BATCH_COALESCE_MAX_BYTES,
    DEFAULT_HOT_OPFS_RPF_ALLOWLIST,
    DEFAULT_OPFS_L2_MAX_PENDING_WRITES,
    DEFAULT_OPFS_L2_MAX_PENDING_WRITE_BYTES,
    DEFAULT_OPFS_L2_MAX_WRITE_CONCURRENCY,
    DEFAULT_OPFS_L2_MAX_BYTES,
    ImmutableHttpfsCache,
    IoWorkerController,
    createIoWorkerHandler,
    coalesceBatchReadRequests,
    collectTransferableBuffers,
    resolveHttpfsRoute,
    normalizeDumpEndpoint,
    resolveDumpConfig,
    isDumpVirtualPath,
    resolveDumpWriteRoute,
    createSlabCacheTuningProfile,
    isImmutableHttpfsPath,
    isWholeFileBufferedPath,
    isVideoStreamPath,
    buildFileFingerprint,
    normalizeOpfsL2Path,
    isHotOpfsL2Path,
    opfsL2RangeFileName,
    parseContentRangeHeader,
    isRpfVirtualPath,
    parseRpfVirtualPath,
    emptyStats,
    emptyRangeCacheStats,
    emptyImmutableCacheStats,
    emptyWholeFileCacheStats,
    emptyMailboxStats,
    emptyDumpWriteStats,
    emptyStreamingStats,
    emptySchedulerStats,
    getDefaultController: () => defaultController,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6IoWorker = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})();
