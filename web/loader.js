'use strict';

// Main runtime coordinator worker (keeps the browser DOM thread completely free).
(function initPlayGta6Loader() {
if (typeof importScripts === 'function') {
  importScripts('./workers/userdata.js');
  importScripts('./workers/httpfs_mailbox.js');
}

const progress =
  typeof self !== 'undefined' && typeof BroadcastChannel === 'function'
    ? new BroadcastChannel('game-progress')
    : null;

const userdataApi =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6Userdata) ||
  (typeof require === 'function' ? require('./workers/userdata.js') : null);
const httpfsMailboxApi =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6HttpfsMailbox) ||
  (typeof require === 'function' ? require('./workers/httpfs_mailbox.js') : null);

const userdataStore = userdataApi ? new userdataApi.UserdataStore() : null;

const utf8Decoder = new TextDecoder('utf-8', { fatal: false });

// GpuPacketHeader / GpuOpcode from runtime/abi/playgta6_abi.h and docs/gpu_commands.md.
// Present is opcode 7 (ClearColor is 2). Detecting opcode 2 as Present would dismiss splash on clear.
const GPU_PACKET_HEADER_BYTES = 8;
const GPU_OPCODE_PRESENT = 7;
const BOOT_INIT_FORWARD_KEYS = Object.freeze([
  'shaderPackUrl',
  'reflectionManifestUrl',
  'pgimUrl',
  'bootBase',
  'gameBase',
  'programManifestUrl',
]);
const GAME_PACKAGE_HTTPFS_EXT_RE = /\.(?:rpf|ttpf|dll|png|xvd|xvc|bin|pri)$/i;
const WIN32_DRIVE_PATH_RE = /^(?:\\\\[?.]\\|\/\/[?.]\/)?([A-Za-z]):(.*)$/;
const GPU_TELEMETRY_MESSAGE_TYPES = new Set([
  'gpu-stats',
  'gpu-telemetry',
  'missing-shaders',
  'frame-presented',
]);
const GPU_SHADER_STAGE_NAMES = Object.freeze({
  1: 'vertex',
  2: 'fragment',
  3: 'compute',
});
const GPU_SHADER_STAGE_ALIASES = Object.freeze({
  vertex: 'vertex',
  vs: 'vertex',
  vert: 'vertex',
  1: 'vertex',
  fragment: 'fragment',
  fs: 'fragment',
  frag: 'fragment',
  pixel: 'fragment',
  ps: 'fragment',
  2: 'fragment',
  compute: 'compute',
  cs: 'compute',
  comp: 'compute',
  3: 'compute',
});
const U64_MAX = 0xffffffffffffffffn;

let gpuWorker = null;
let ioWorker = null;
let audioWorker = null;
let inputSab = null;
let httpfsMailboxSab = null;
let httpfsClient = null;

function report(label) {
  if (progress) {
    progress.postMessage({ label });
  }
}

function ptrToNumber(value, label = 'pointer') {
  if (typeof value === 'bigint') {
    if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new RangeError(`${label} out of addressable range: ${value}`);
    }
    return Number(value);
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
      throw new RangeError(`${label} must be a non-negative integer: ${value}`);
    }
    return value;
  }
  throw new TypeError(`${label} must be a BigInt or Number`);
}

function resolveBuffer(memoryOrBuffer) {
  if (!memoryOrBuffer) return null;
  if (memoryOrBuffer.buffer) return memoryOrBuffer.buffer;
  if (memoryOrBuffer instanceof ArrayBuffer) return memoryOrBuffer;
  if (typeof SharedArrayBuffer !== 'undefined' && memoryOrBuffer instanceof SharedArrayBuffer) {
    return memoryOrBuffer;
  }
  return null;
}

function copyMemoryBytes(buf, ptr, len) {
  const byteOffset = ptrToNumber(ptr, 'ptr');
  const byteLength = ptrToNumber(len, 'len');
  if (byteOffset + byteLength > buf.byteLength) {
    throw new RangeError(
      `Wasm memory read out of bounds: [${byteOffset}..${byteOffset + byteLength}) > ${buf.byteLength}`,
    );
  }
  const copy = new Uint8Array(byteLength);
  if (byteLength > 0) {
    copy.set(new Uint8Array(buf, byteOffset, byteLength));
  }
  return copy;
}

function asDataView(bytes) {
  if (
    bytes instanceof ArrayBuffer ||
    (typeof SharedArrayBuffer !== 'undefined' && bytes instanceof SharedArrayBuffer)
  ) {
    return new DataView(bytes);
  }
  if (ArrayBuffer.isView(bytes)) {
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  return null;
}

/**
 * Walk a little-endian GpuPacketHeader stream (u16 opcode, u16 flags, u32 payloadSize)
 * and report whether it contains a Present packet. Never throws on truncated or malformed input.
 */
function inspectGpuPacketBatchForPresent(bytes) {
  const empty = {
    packetCount: 0,
    hasPresent: false,
    firstPresentOffset: -1,
    truncated: false,
  };
  let view;
  try {
    view = asDataView(bytes);
  } catch {
    return empty;
  }
  if (!view) {
    return empty;
  }

  const result = {
    packetCount: 0,
    hasPresent: false,
    firstPresentOffset: -1,
    truncated: false,
  };
  const total = view.byteLength;
  let offset = 0;

  while (offset < total) {
    if (offset + GPU_PACKET_HEADER_BYTES > total) {
      result.truncated = true;
      break;
    }
    let opcode;
    let payloadBytes;
    try {
      opcode = view.getUint16(offset, true);
      payloadBytes = view.getUint32(offset + 4, true);
    } catch {
      result.truncated = true;
      break;
    }
    const payloadOffset = offset + GPU_PACKET_HEADER_BYTES;
    const remaining = total - payloadOffset;
    if (payloadBytes > remaining) {
      result.truncated = true;
      break;
    }
    if (opcode === GPU_OPCODE_PRESENT && !result.hasPresent) {
      result.hasPresent = true;
      result.firstPresentOffset = offset;
    }
    result.packetCount += 1;
    offset = payloadOffset + payloadBytes;
  }

  return result;
}

function ensureTrailingSlash(base) {
  if (typeof base !== 'string' || base.length === 0) {
    return '';
  }
  return base.endsWith('/') ? base : `${base}/`;
}

function sanitizeHttpfsRelPath(relPath) {
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

function basenameOfPath(posixPath) {
  const slash = posixPath.lastIndexOf('/');
  return slash < 0 ? posixPath : posixPath.slice(slash + 1);
}

function isDataPinnedHttpfsPath(cleanPath) {
  const base = basenameOfPath(cleanPath).toLowerCase();
  if (base === 'data-manifest.json' || base === 'bootset.json') {
    return true;
  }
  return /\.rpf$/i.test(cleanPath);
}

function isManifestPinnedHttpfsPath(cleanPath) {
  const base = basenameOfPath(cleanPath).toLowerCase();
  return base === 'data-manifest.json' || base === 'bootset.json';
}

function isExplicitBootHttpfsPath(cleanPath) {
  const lower = cleanPath.toLowerCase();
  return lower === 'boot' || lower.startsWith('boot/');
}

function isBootRoutedHttpfsPath(cleanPath) {
  const lower = cleanPath.toLowerCase();
  if (isExplicitBootHttpfsPath(cleanPath)) {
    return true;
  }
  if (basenameOfPath(lower) === 'program-manifest.json') {
    return true;
  }
  return /\.pgim$/i.test(lower);
}

function isGamePrefixedHttpfsPath(cleanPath) {
  const lower = cleanPath.toLowerCase();
  return lower === 'game' || lower.startsWith('game/');
}

function isGamePackageHttpfsPath(cleanPath) {
  const base = basenameOfPath(cleanPath).toLowerCase();
  if (base === 'microsoftgame.config') {
    return true;
  }
  return GAME_PACKAGE_HTTPFS_EXT_RE.test(cleanPath);
}

function stripBootPrefix(cleanPath) {
  const lower = cleanPath.toLowerCase();
  if (lower === 'boot') {
    return '';
  }
  if (lower.startsWith('boot/')) {
    return cleanPath.slice('boot/'.length);
  }
  return cleanPath;
}

function stripGamePrefix(cleanPath) {
  const lower = cleanPath.toLowerCase();
  if (lower === 'game') {
    return '';
  }
  if (lower.startsWith('game/')) {
    return cleanPath.slice('game/'.length);
  }
  return cleanPath;
}

function joinHttpfsBase(base, rel) {
  const prefix = ensureTrailingSlash(base) || '/data/';
  const tail = String(rel || '').replace(/^\/+/, '');
  return tail ? `${prefix}${tail}` : prefix;
}

/**
 * Route guest HTTPFS paths: `boot/...`, `program-manifest.json`, and `*.pgim` go to
 * `bootBase` when configured; Xbox game package/drive paths (`G:\...`, `game/...`,
 * `*.rpf`, `*.dll`, `MicrosoftGame.Config`, etc.) go to `gameBase` when configured;
 * `data-manifest.json`, `bootset.json`, and fallback paths resolve on `dataBase`.
 */
function normalizeBootHttpfsPath(relPath, options = {}) {
  if (typeof relPath !== 'string' || relPath.length === 0) {
    throw new Error('HTTPFS path must be a non-empty string');
  }
  if (relPath.includes('\0')) {
    throw new Error(`Invalid HTTPFS path: ${relPath}`);
  }

  const dataBase = options && options.dataBase !== undefined ? options.dataBase : '/data/';
  const bootBase =
    options && typeof options.bootBase === 'string' ? options.bootBase.trim() : '';
  const gameBase =
    options && typeof options.gameBase === 'string' ? options.gameBase.trim() : '';

  let hadDrivePrefix = false;
  let pathToSanitize = relPath;
  if (gameBase) {
    const driveMatch = WIN32_DRIVE_PATH_RE.exec(relPath);
    if (driveMatch) {
      hadDrivePrefix = true;
      pathToSanitize = driveMatch[2].replace(/\\/g, '/');
    }
  }

  const clean = sanitizeHttpfsRelPath(pathToSanitize);

  if (isManifestPinnedHttpfsPath(clean)) {
    return joinHttpfsBase(dataBase, clean);
  }
  if (bootBase && isExplicitBootHttpfsPath(clean) && !isDataPinnedHttpfsPath(clean)) {
    return joinHttpfsBase(bootBase, stripBootPrefix(clean));
  }
  if (gameBase && !hadDrivePrefix && isGamePrefixedHttpfsPath(clean)) {
    return joinHttpfsBase(gameBase, stripGamePrefix(clean));
  }
  if (bootBase && isBootRoutedHttpfsPath(clean) && !isDataPinnedHttpfsPath(clean)) {
    return joinHttpfsBase(bootBase, stripBootPrefix(clean));
  }
  if (gameBase && (hadDrivePrefix || isGamePackageHttpfsPath(clean))) {
    return joinHttpfsBase(gameBase, clean);
  }
  return joinHttpfsBase(dataBase, clean);
}

function toNonNegativeInt(value, fallback = 0) {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.floor(n);
}

function toNonNegativeFinite(value) {
  if (value === undefined || value === null || value === '') {
    return null;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return null;
  }
  return n;
}

function clampPct(value) {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n <= 0) return 0;
  if (n >= 100) return 100;
  return n;
}

function normalizeShaderHashHex(raw) {
  if (typeof raw === 'bigint') {
    if (raw < 0n || raw > U64_MAX) return null;
    return `0x${raw.toString(16).padStart(16, '0')}`;
  }
  if (typeof raw === 'number') {
    if (!Number.isSafeInteger(raw) || raw < 0) return null;
    return `0x${BigInt(raw).toString(16).padStart(16, '0')}`;
  }
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    const hex =
      trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
    if (!/^[0-9a-fA-F]{1,16}$/.test(hex)) {
      return null;
    }
    return `0x${BigInt(`0x${hex}`).toString(16).padStart(16, '0')}`;
  }
  return null;
}

function normalizeShaderStageName(stage) {
  if (typeof stage === 'number') {
    return GPU_SHADER_STAGE_NAMES[stage] || 'unknown';
  }
  if (typeof stage === 'string') {
    const key = stage.trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(GPU_SHADER_STAGE_ALIASES, key)) {
      return GPU_SHADER_STAGE_ALIASES[key];
    }
  }
  return 'unknown';
}

function sanitizeMissingShadersList(rawList) {
  if (!rawList || typeof rawList === 'string') {
    return [];
  }
  let items;
  if (Array.isArray(rawList)) {
    items = rawList;
  } else if (typeof rawList[Symbol.iterator] === 'function') {
    try {
      items = Array.from(rawList);
    } catch {
      return [];
    }
  } else {
    return [];
  }

  const out = [];
  for (let i = 0; i < items.length; i += 1) {
    const entry = items[i];
    if (!entry || typeof entry !== 'object') {
      continue;
    }
    const rawHash =
      entry.hash !== undefined
        ? entry.hash
        : entry.shaderHash !== undefined
          ? entry.shaderHash
          : entry.hashHex;
    const hash = normalizeShaderHashHex(rawHash);
    if (hash === null) {
      continue;
    }
    const stage = normalizeShaderStageName(entry.stage);
    const rawHits =
      entry.hits !== undefined
        ? entry.hits
        : entry.hitCount !== undefined
          ? entry.hitCount
          : entry.drawCount !== undefined
            ? entry.drawCount
            : entry.count;
    const hits = rawHits === undefined || rawHits === null ? 1 : toNonNegativeInt(rawHits, 0);
    const sanitized = { hash, stage, hits };
    const firstSeenMs = toNonNegativeFinite(entry.firstSeenMs);
    const lastSeenMs = toNonNegativeFinite(entry.lastSeenMs);
    if (firstSeenMs !== null) {
      sanitized.firstSeenMs = firstSeenMs;
    }
    if (lastSeenMs !== null) {
      sanitized.lastSeenMs = lastSeenMs;
    }
    out.push(sanitized);
  }
  return out;
}

function sanitizeFallbackStats(raw) {
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const packHits = toNonNegativeInt(raw.packHits, 0);
  const fallbackHits = toNonNegativeInt(raw.fallbackHits, 0);
  const builtinHits = toNonNegativeInt(raw.builtinHits, 0);
  const sumHits = packHits + fallbackHits + builtinHits;
  const totalLookups =
    raw.totalLookups !== undefined && raw.totalLookups !== null
      ? toNonNegativeInt(raw.totalLookups, sumHits)
      : sumHits;
  const overflowMissingCount = toNonNegativeInt(raw.overflowMissingCount, 0);
  let hitRatePct;
  if (
    raw.hitRatePct !== undefined &&
    raw.hitRatePct !== null &&
    Number.isFinite(Number(raw.hitRatePct))
  ) {
    hitRatePct = Math.round(clampPct(Number(raw.hitRatePct)) * 100) / 100;
  } else {
    hitRatePct = totalLookups > 0 ? Math.round((packHits / totalLookups) * 10000) / 100 : 0;
  }
  return {
    totalLookups,
    packHits,
    fallbackHits,
    builtinHits,
    hitRatePct,
    overflowMissingCount,
  };
}

function sanitizeFrameStats(eventData) {
  const rawFrameStats =
    eventData.frameStats && typeof eventData.frameStats === 'object' ? eventData.frameStats : null;
  const rawStats =
    eventData.stats && typeof eventData.stats === 'object' ? eventData.stats : null;

  const hasFrameField =
    Boolean(rawFrameStats) ||
    Boolean(rawStats) ||
    typeof eventData.framesPresented === 'number' ||
    typeof eventData.droppedFrames === 'number' ||
    typeof eventData.frame === 'number' ||
    typeof eventData.frameCount === 'number' ||
    eventData.type === 'frame-presented' ||
    eventData.type === 'gpu-stats';

  if (!hasFrameField) {
    return null;
  }

  const rawPresented =
    rawFrameStats && rawFrameStats.framesPresented !== undefined
      ? rawFrameStats.framesPresented
      : rawFrameStats && rawFrameStats.frameCount !== undefined
        ? rawFrameStats.frameCount
        : eventData.framesPresented !== undefined
          ? eventData.framesPresented
          : rawStats && rawStats.framesPresented !== undefined
            ? rawStats.framesPresented
            : rawStats && rawStats.frameCount !== undefined
              ? rawStats.frameCount
              : eventData.frameCount !== undefined
                ? eventData.frameCount
                : eventData.frame;

  const rawDropped =
    rawFrameStats && rawFrameStats.droppedFrames !== undefined
      ? rawFrameStats.droppedFrames
      : eventData.droppedFrames !== undefined
        ? eventData.droppedFrames
        : rawStats && rawStats.droppedFrames !== undefined
          ? rawStats.droppedFrames
          : 0;

  const frameStats = {
    framesPresented: toNonNegativeInt(rawPresented, 0),
    droppedFrames: toNonNegativeInt(rawDropped, 0),
  };

  const drawCallsSrc =
    rawFrameStats && rawFrameStats.drawCalls !== undefined
      ? rawFrameStats.drawCalls
      : rawStats && rawStats.drawCalls !== undefined
        ? rawStats.drawCalls
        : eventData.drawCalls;
  if (drawCallsSrc !== undefined && drawCallsSrc !== null) {
    frameStats.drawCalls = toNonNegativeInt(drawCallsSrc, 0);
  }

  const computeSrc =
    rawFrameStats && rawFrameStats.computeDispatches !== undefined
      ? rawFrameStats.computeDispatches
      : rawStats && rawStats.computeDispatches !== undefined
        ? rawStats.computeDispatches
        : eventData.computeDispatches;
  if (computeSrc !== undefined && computeSrc !== null) {
    frameStats.computeDispatches = toNonNegativeInt(computeSrc, 0);
  }

  const packetSrc =
    rawFrameStats && rawFrameStats.packetCount !== undefined
      ? rawFrameStats.packetCount
      : rawStats && rawStats.packetCount !== undefined
        ? rawStats.packetCount
        : eventData.packetCount;
  if (packetSrc !== undefined && packetSrc !== null) {
    frameStats.packetCount = toNonNegativeInt(packetSrc, 0);
  }

  const vramSrc =
    rawFrameStats && rawFrameStats.vramBytes !== undefined
      ? rawFrameStats.vramBytes
      : rawStats && rawStats.vramBytes !== undefined
        ? rawStats.vramBytes
        : eventData.vramBytes;
  if (vramSrc !== undefined && vramSrc !== null) {
    frameStats.vramBytes = toNonNegativeInt(vramSrc, 0);
  }

  return frameStats;
}

function isGpuTelemetryMessage(eventData) {
  if (!eventData || typeof eventData !== 'object' || Array.isArray(eventData)) {
    return false;
  }
  if (typeof eventData.type === 'string' && GPU_TELEMETRY_MESSAGE_TYPES.has(eventData.type)) {
    return true;
  }
  if (Array.isArray(eventData.missingShaders)) {
    return true;
  }
  if (
    (eventData.shaderFallback && typeof eventData.shaderFallback === 'object') ||
    (eventData.shaderFallbackStats && typeof eventData.shaderFallbackStats === 'object') ||
    (eventData.fallbackStats && typeof eventData.fallbackStats === 'object') ||
    (eventData.frameStats && typeof eventData.frameStats === 'object') ||
    (eventData.stats && typeof eventData.stats === 'object')
  ) {
    return true;
  }
  if (
    (typeof eventData.framesPresented === 'number' && Number.isFinite(eventData.framesPresented)) ||
    (typeof eventData.frame === 'number' && Number.isFinite(eventData.frame))
  ) {
    return true;
  }
  return false;
}

/**
 * Inspect a message emitted by `wgpu_worker.js`, sanitize/normalize GPU frame,
 * shader-fallback, and missing-shader telemetry, and forward `{ type: 'gpu-telemetry', ... }`
 * to the main thread via `postMessageFn`. Returns the forwarded payload or `null` when ignored.
 */
function forwardGpuWorkerMessage(eventData, postMessageFn) {
  if (!isGpuTelemetryMessage(eventData)) {
    return null;
  }

  const rawStats =
    eventData.stats && typeof eventData.stats === 'object' ? eventData.stats : null;
  const rawFallback =
    eventData.fallbackStats ||
    eventData.shaderFallbackStats ||
    eventData.shaderFallback ||
    (rawStats && (rawStats.shaderFallback || rawStats.fallbackStats || rawStats.shaderFallbackStats)) ||
    null;
  const rawMissing =
    eventData.missingShaders !== undefined
      ? eventData.missingShaders
      : rawStats && rawStats.missingShaders !== undefined
        ? rawStats.missingShaders
        : null;

  const timestamp =
    typeof eventData.timestamp === 'number' && Number.isFinite(eventData.timestamp)
      ? eventData.timestamp
      : Date.now();

  const telemetry = {
    type: 'gpu-telemetry',
    frameStats: sanitizeFrameStats(eventData),
    fallbackStats: sanitizeFallbackStats(rawFallback),
    missingShaders: sanitizeMissingShadersList(rawMissing),
    timestamp,
  };

  if (typeof postMessageFn === 'function') {
    try {
      postMessageFn(telemetry);
    } catch {
      // Telemetry forwarding must never disrupt the GPU worker loop.
    }
  }

  return telemetry;
}

function collectBootBridgeInitFields(src) {
  const extra = {};
  if (!src || typeof src !== 'object') {
    return extra;
  }
  for (let i = 0; i < BOOT_INIT_FORWARD_KEYS.length; i += 1) {
    const key = BOOT_INIT_FORWARD_KEYS[i];
    if (src[key] !== undefined) {
      extra[key] = src[key];
    }
  }
  return extra;
}

function createWorkerBootInitPayloads(msg = {}) {
  const bootFields = collectBootBridgeInitFields(msg);
  return {
    ioInit: Object.assign({ init: true, dataBase: msg.dataBase }, bootFields),
    gpuInit: Object.assign(
      { init: true, canvas: msg.canvas, width: msg.width, height: msg.height },
      bootFields,
    ),
  };
}

// Builds Wasm64/Wasm32 `env` host import functions matching docs/wasm_abi.md and runtime/os/os_shim.cpp.
function createHostImports(options = {}) {
  let currentMemory = options.memory || null;
  const getMemory = typeof options.getMemory === 'function' ? options.getMemory : null;
  const logFn = typeof options.log === 'function' ? options.log : (msg) => report(`[wasm] ${msg}`);
  const wgpuSubmitFn = typeof options.wgpuSubmit === 'function' ? options.wgpuSubmit : null;
  const onFirstPresent =
    typeof options.onFirstPresent === 'function' ? options.onFirstPresent : null;
  const reportFn = typeof options.report === 'function' ? options.report : report;
  const emitFirstPresentLog = options.emitFirstPresentLog === true;
  const fsClient = options.httpfsClient || null;
  const store = options.userdataStore || null;
  let firstPresentEmitted = false;

  function activeBuffer() {
    const raw = getMemory ? getMemory() || currentMemory : currentMemory;
    return resolveBuffer(raw);
  }

  function host_log(ptr, len) {
    const buf = activeBuffer();
    if (!buf) return;
    const bytes = copyMemoryBytes(buf, ptr, len);
    logFn(utf8Decoder.decode(bytes));
  }

  function emitFirstPresent(bytes, info) {
    if (firstPresentEmitted || !info || !info.hasPresent) {
      return;
    }
    firstPresentEmitted = true;
    const detail = {
      packetCount: info.packetCount,
      byteLength: bytes && typeof bytes.byteLength === 'number' ? bytes.byteLength : 0,
    };
    try {
      if (onFirstPresent) {
        onFirstPresent(detail);
      }
    } catch {
      // First-present hooks must never break the GPU submit path.
    }
    try {
      reportFn('[boot] FIRST_PRESENT');
    } catch {
      // Progress channel failures are non-fatal.
    }
    // Do not call host_log's logFn unless asked: existing Wasm64 tests treat `log` as host_log only.
    if (emitFirstPresentLog) {
      try {
        logFn('[boot] FIRST_PRESENT');
      } catch {
        // Boot log failures are non-fatal.
      }
    }
  }

  function host_wgpu_submit(ptr, len) {
    const buf = activeBuffer();
    if (!buf) return;
    const bytes = copyMemoryBytes(buf, ptr, len);
    if (!firstPresentEmitted) {
      emitFirstPresent(bytes, inspectGpuPacketBatchForPresent(bytes));
    }
    if (wgpuSubmitFn) {
      wgpuSubmitFn(bytes);
    }
  }

  function host_httpfs_open(ptr, len) {
    if (!fsClient || typeof fsClient.open !== 'function') {
      return -1;
    }
    const buf = activeBuffer();
    if (!buf) return -1;
    try {
      const bytes = copyMemoryBytes(buf, ptr, len);
      const relPath = utf8Decoder.decode(bytes);
      const handle = fsClient.open(relPath);
      return Number.isInteger(handle) ? handle | 0 : -1;
    } catch {
      return -1;
    }
  }

  function host_httpfs_read(handle, offset, dstPtr, len) {
    if (!fsClient || typeof fsClient.read !== 'function') {
      return -1n;
    }
    const buf = activeBuffer();
    if (!buf) return -1n;
    try {
      const h = Number(handle) | 0;
      if (h < 0) return -1n;
      const dstOffset = ptrToNumber(dstPtr, 'dstPtr');
      const byteLength = ptrToNumber(len, 'len');
      if (dstOffset + byteLength > buf.byteLength) {
        return -1n;
      }
      if (byteLength === 0) {
        return 0n;
      }
      const dstView = new Uint8Array(buf, dstOffset, byteLength);
      const n = fsClient.read(h, offset, dstView, byteLength);
      if (typeof n === 'bigint') {
        return n;
      }
      return Number.isFinite(n) ? BigInt(Math.trunc(n)) : -1n;
    } catch {
      return -1n;
    }
  }

  function host_httpfs_close(handle) {
    if (!fsClient || typeof fsClient.close !== 'function') {
      return;
    }
    try {
      fsClient.close(Number(handle) | 0);
    } catch {
      // Close errors are non-fatal for void ABI.
    }
  }

  const env = {
    host_log,
    host_wgpu_submit,
    host_httpfs_open,
    host_httpfs_read,
    host_httpfs_close,
  };

  const userdata = {
    restore: (fsAdapter, opts) => {
      if (!store) {
        return Promise.resolve({
          restored: 0,
          skipped: 0,
          timedOut: false,
          error: 'userdata store unavailable',
        });
      }
      return store.restoreAll(fsAdapter, opts);
    },
    save: (path, bytes, mtime) => {
      if (!store) return Promise.resolve(null);
      return store.saveFile(path, bytes, mtime).catch((err) => {
        report(`userdata save warning (${path}): ${err}`);
        return null;
      });
    },
    delete: (path) => {
      if (!store) return Promise.resolve(null);
      return store.deleteFile(path).catch((err) => {
        report(`userdata delete warning (${path}): ${err}`);
        return null;
      });
    },
  };

  return {
    env,
    userdata,
    setMemory(mem) {
      currentMemory = mem || null;
    },
  };
}

function encodeTestFramePacket(r, g, b, a) {
  // Matches runtime/abi/playgta6_abi.h:
  // BeginFrame (8B) + ClearColor (8B + 16B) + Present (8B) = 40 bytes
  const buf = new ArrayBuffer(40);
  const view = new DataView(buf);
  // BeginFrame (opcode 1, payload 0)
  view.setUint16(0, 1, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, 0, true);
  // ClearColor (opcode 2, payload 16)
  view.setUint16(8, 2, true);
  view.setUint16(10, 0, true);
  view.setUint32(12, 16, true);
  view.setFloat32(16, r, true);
  view.setFloat32(20, g, true);
  view.setFloat32(24, b, true);
  view.setFloat32(28, a, true);
  // Present (opcode 7, payload 0)
  view.setUint16(32, 7, true);
  view.setUint16(34, 0, true);
  view.setUint32(36, 0, true);
  return buf;
}

const ABSOLUTE_URL_RE = /^[a-zA-Z][a-zA-Z0-9+.-]*:/;
const DEFAULT_RUNTIME_SCRIPT_RELATIVE = './dist/runtime.js';
const DEFAULT_RUNTIME_SCRIPT_FALLBACK = 'dist/runtime.js';

function isAbsoluteUrl(value) {
  return typeof value === 'string' && ABSOLUTE_URL_RE.test(value);
}

function dirnameWithSlash(relPath) {
  if (typeof relPath !== 'string' || relPath.length === 0) {
    return 'dist/';
  }
  const clean = relPath.split(/[?#]/, 1)[0];
  const slash = clean.lastIndexOf('/');
  if (slash < 0) {
    return 'dist/';
  }
  return clean.slice(0, slash + 1);
}

/**
 * Resolve the Emscripten main script URL (`dist/runtime.js`) relative to the
 * coordinator worker location so `-pthread` workers spawn `dist/runtime.js`
 * instead of recursively spawning `loader.js` via `self.location.href`.
 */
function resolveRuntimeScriptUrl(runtimeScriptUrl, locationHref) {
  const rawScript =
    typeof runtimeScriptUrl === 'string' && runtimeScriptUrl.trim().length > 0
      ? runtimeScriptUrl.trim()
      : '';
  if (rawScript && isAbsoluteUrl(rawScript)) {
    return rawScript;
  }

  const baseHref =
    typeof locationHref === 'string' && locationHref.trim().length > 0
      ? locationHref.trim()
      : typeof self !== 'undefined' &&
          self &&
          self.location &&
          typeof self.location.href === 'string'
        ? self.location.href
        : '';

  const relativeTarget = rawScript || DEFAULT_RUNTIME_SCRIPT_RELATIVE;
  if (baseHref && typeof URL === 'function') {
    try {
      return new URL(relativeTarget, baseHref).href;
    } catch {
      // Fall through to relative fallback when baseHref is not a valid URL.
    }
  }

  return rawScript || DEFAULT_RUNTIME_SCRIPT_FALLBACK;
}

function createRuntimeLocateFile(runtimeScriptUrl, locateFileBase) {
  const explicitBase =
    typeof locateFileBase === 'string' && locateFileBase.trim().length > 0
      ? ensureTrailingSlash(locateFileBase.trim())
      : '';

  return function locateFile(file) {
    if (typeof file !== 'string' || file.length === 0) {
      return '';
    }
    if (isAbsoluteUrl(file)) {
      return file;
    }
    if (explicitBase) {
      if (isAbsoluteUrl(explicitBase) && typeof URL === 'function') {
        try {
          return new URL(file, explicitBase).href;
        } catch {
          // Fall through to string join.
        }
      }
      return `${explicitBase}${file.replace(/^\.\//, '')}`;
    }
    if (isAbsoluteUrl(runtimeScriptUrl) && typeof URL === 'function') {
      try {
        return new URL(file, runtimeScriptUrl).href;
      } catch {
        // Fall through to relative dist/ join.
      }
    }
    return `${dirnameWithSlash(runtimeScriptUrl)}${file.replace(/^\.\//, '')}`;
  };
}

function extractMemoryFromImports(imports) {
  if (!imports || typeof imports !== 'object') {
    return null;
  }
  if (imports.env && imports.env.memory) {
    return imports.env.memory;
  }
  if (imports.a && imports.a.memory) {
    return imports.a.memory;
  }
  return null;
}

function mergeHostImportsIntoWasmImports(imports, hostEnv) {
  const merged = Object.assign({}, imports && typeof imports === 'object' ? imports : {});
  const baseEnv = merged.env && typeof merged.env === 'object' ? merged.env : {};
  merged.env = Object.assign({}, baseEnv, hostEnv || {});
  return merged;
}

function createInstantiateWasmHandler(options = {}) {
  const locateFile =
    typeof options.locateFile === 'function' ? options.locateFile : (f) => `dist/${f}`;
  const hostImports = options.hostImports || null;
  const reportFn = typeof options.reportFn === 'function' ? options.reportFn : report;
  const fetchFn =
    typeof options.fetchFn === 'function'
      ? options.fetchFn
      : typeof fetch === 'function'
        ? fetch
        : null;
  const wasmBinary = options.wasmBinary || null;
  const onError = typeof options.onError === 'function' ? options.onError : null;

  return function instantiateWasm(imports, receiveInstance) {
    const importedMemory = extractMemoryFromImports(imports);
    if (importedMemory && hostImports && typeof hostImports.setMemory === 'function') {
      hostImports.setMemory(importedMemory);
    }

    const mergedImports = mergeHostImportsIntoWasmImports(
      imports,
      hostImports && hostImports.env ? hostImports.env : null,
    );

    const finish = (result) => {
      const instance = result && result.instance ? result.instance : result;
      const mod =
        result && result.module
          ? result.module
          : typeof WebAssembly !== 'undefined' && wasmBinary instanceof WebAssembly.Module
            ? wasmBinary
            : undefined;
      if (
        instance &&
        instance.exports &&
        instance.exports.memory &&
        hostImports &&
        typeof hostImports.setMemory === 'function'
      ) {
        hostImports.setMemory(instance.exports.memory);
      }
      if (typeof receiveInstance === 'function') {
        receiveInstance(instance, mod);
      }
    };

    const handleFailure = (err) => {
      try {
        reportFn(`runtime wasm instantiation failed: ${err}`);
      } catch {
        // Progress channel failures must not mask instantiation errors.
      }
      if (onError) {
        try {
          onError(err);
        } catch {
          // Ignore secondary error hook exceptions.
        }
      }
    };

    const wasmUrl = locateFile('runtime.wasm');
    let instantiatePromise;
    if (wasmBinary) {
      instantiatePromise = WebAssembly.instantiate(wasmBinary, mergedImports);
    } else if (!fetchFn) {
      handleFailure(new Error(`fetch unavailable while loading ${wasmUrl}`));
      return {};
    } else {
      instantiatePromise = Promise.resolve()
        .then(() => fetchFn(wasmUrl, { credentials: 'same-origin' }))
        .then((resp) => {
          if (!resp || resp.ok === false) {
            const status = resp && resp.status !== undefined ? resp.status : 'unknown';
            throw new Error(`Failed to fetch ${wasmUrl} (HTTP ${status})`);
          }
          const contentType =
            resp.headers && typeof resp.headers.get === 'function'
              ? String(resp.headers.get('content-type') || '')
              : '';
          const canStream =
            typeof WebAssembly.instantiateStreaming === 'function' &&
            (typeof Response === 'undefined' || resp instanceof Response) &&
            contentType.toLowerCase().includes('application/wasm');
          if (canStream) {
            return WebAssembly.instantiateStreaming(Promise.resolve(resp), mergedImports);
          }
          return resp.arrayBuffer().then((bytes) => WebAssembly.instantiate(bytes, mergedImports));
        });
    }

    instantiatePromise.then(finish, handleFailure);
    return {};
  };
}

/**
 * Build the Emscripten `createRuntimeModule(...)` configuration object with:
 * - `mainScriptUrlOrBlob` pinned to `dist/runtime.js` (never `loader.js`),
 * - `locateFile` resolving `runtime.wasm` / `runtime.worker.js` cleanly under `dist/`,
 * - `env` + `instantiateWasm` merging `host_log`, `host_httpfs_*`, and `host_wgpu_submit`.
 */
function buildRuntimeModuleConfig(options = {}) {
  let boundModule = null;
  const activeReportFn =
    typeof options.reportFn === 'function'
      ? options.reportFn
      : typeof options.report === 'function'
        ? options.report
        : report;
  const activeHttpfsClient =
    options.httpfsClient !== undefined ? options.httpfsClient : httpfsClient;
  const activeUserdataStore =
    options.userdataStore !== undefined ? options.userdataStore : userdataStore;
  const activeGpuWorker = options.gpuWorker !== undefined ? options.gpuWorker : gpuWorker;

  const wgpuSubmit =
    typeof options.wgpuSubmit === 'function'
      ? options.wgpuSubmit
      : (bytes) => {
          if (!activeGpuWorker) return;
          const copy = bytes.slice().buffer;
          activeGpuWorker.postMessage({ cmdBuffer: copy }, [copy]);
        };

  const logFn =
    typeof options.log === 'function' ? options.log : (msg) => activeReportFn(`[wasm] ${msg}`);

  const activeHostImports =
    options.hostImports && typeof options.hostImports === 'object'
      ? options.hostImports
      : createHostImports({
          memory: options.memory || null,
          getMemory: () =>
            options.memory ||
            (boundModule &&
              (boundModule.wasmMemory || (boundModule.HEAPU8 && boundModule.HEAPU8.buffer))) ||
            null,
          log: logFn,
          report: activeReportFn,
          wgpuSubmit,
          httpfsClient: activeHttpfsClient,
          userdataStore: activeUserdataStore,
          onFirstPresent:
            typeof options.onFirstPresent === 'function' ? options.onFirstPresent : null,
          emitFirstPresentLog: options.emitFirstPresentLog === true,
        });

  const resolvedRuntimeScriptUrl = resolveRuntimeScriptUrl(
    options.runtimeScriptUrl,
    options.locationHref,
  );
  const locateFile = createRuntimeLocateFile(resolvedRuntimeScriptUrl, options.locateFileBase);

  const preRun = Array.isArray(options.preRun)
    ? options.preRun
    : [
        (mod) => {
          boundModule = mod || null;
          if (
            mod &&
            (mod.wasmMemory || (mod.HEAPU8 && mod.HEAPU8.buffer)) &&
            typeof activeHostImports.setMemory === 'function'
          ) {
            activeHostImports.setMemory(mod.wasmMemory || mod.HEAPU8.buffer);
          }
          if (!activeUserdataStore || !mod || !mod.FS) return;
          const useRunDep =
            typeof mod.addRunDependency === 'function' &&
            typeof mod.removeRunDependency === 'function';
          if (useRunDep) {
            mod.addRunDependency('userdata-restore');
          }
          activeUserdataStore
            .restoreAll(mod.FS, { timeoutMs: 8000 })
            .then((res) => {
              if (res.timedOut) {
                activeReportFn(`userdata restore timed out (continuing boot): ${res.error}`);
              } else if (res.error) {
                activeReportFn(`userdata restore skipped: ${res.error}`);
              } else {
                activeReportFn(`userdata restored (${res.restored} files)`);
              }
            })
            .catch((err) => {
              activeReportFn(`userdata restore warning: ${err}`);
            })
            .finally(() => {
              if (useRunDep) {
                mod.removeRunDependency('userdata-restore');
              }
            });
        },
      ];

  const userdataApiObj = activeHostImports.userdata || {};
  const config = {
    mainScriptUrlOrBlob: resolvedRuntimeScriptUrl,
    locateFile,
    preRun,
    env: activeHostImports.env,
    hostImports: activeHostImports,
    httpfsMailboxSab:
      options.httpfsMailboxSab !== undefined ? options.httpfsMailboxSab : httpfsMailboxSab,
    inputSab: options.inputSab !== undefined ? options.inputSab : inputSab,
    hostCallbacks: options.hostCallbacks || {
      log: logFn,
      wgpuSubmit,
      httpfsOpen: (relPath) => (activeHttpfsClient ? activeHttpfsClient.open(relPath) : -1),
      httpfsRead: (handle, offset, dst, len) =>
        activeHttpfsClient ? activeHttpfsClient.read(handle, offset, dst, len) : -1,
      httpfsClose: (handle) => (activeHttpfsClient ? activeHttpfsClient.close(handle) : false),
      userdataRestore: userdataApiObj.restore,
      userdataSave: userdataApiObj.save,
      userdataDelete: userdataApiObj.delete,
    },
  };

  if (typeof options.instantiateWasm === 'function') {
    config.instantiateWasm = options.instantiateWasm;
  } else if (options.instantiateWasm !== null && options.instantiateWasm !== false) {
    config.instantiateWasm = createInstantiateWasmHandler({
      locateFile,
      hostImports: activeHostImports,
      reportFn: activeReportFn,
      fetchFn: options.fetchFn,
      wasmBinary: options.wasmBinary,
      onError: options.onWasmError,
    });
  }

  return config;
}

function createRuntimeModuleOptions(overrides = {}) {
  return buildRuntimeModuleConfig(overrides);
}

async function tryBootWasmRuntime() {
  try {
    const head = await fetch('./dist/runtime.js', { method: 'HEAD' });
    if (!head.ok) {
      report('No web/dist/runtime.js built yet (running WebGPU command-ring self-test).');
      const pkt = encodeTestFramePacket(0.08, 0.14, 0.24, 1.0);
      gpuWorker.postMessage({ cmdBuffer: pkt }, [pkt]);
      return;
    }
    report('Loading Emscripten Wasm64 runtime (dist/runtime.js)...');
    importScripts('./dist/runtime.js');
    if (typeof self.createRuntimeModule === 'function') {
      await self.createRuntimeModule(
        createRuntimeModuleOptions({
          onWasmError: (err) => {
            if (progress) {
              progress.postMessage({ error: `runtime wasm: ${err}` });
            }
          },
        }),
      );
    }
  } catch (err) {
    if (progress) {
      progress.postMessage({ error: `runtime boot: ${err}` });
    }
  }
}

if (typeof self !== 'undefined') {
  const postToMain =
    typeof self.postMessage === 'function' ? (payload) => self.postMessage(payload) : null;

  self.onmessage = (ev) => {
    const msg = ev.data || {};
    if (msg.type === 'request-gpu-telemetry') {
      if (gpuWorker && typeof gpuWorker.postMessage === 'function') {
        const statsReq = { stats: true };
        if (msg.id !== undefined) {
          statsReq.id = msg.id;
        }
        gpuWorker.postMessage(statsReq);
        if (msg.includeMissingShaders !== false) {
          const missingReq = { missingShaders: true };
          if (msg.id !== undefined) {
            missingReq.id = msg.id;
          }
          if (msg.limit !== undefined) {
            missingReq.limit = msg.limit;
          }
          gpuWorker.postMessage(missingReq);
        }
      }
      return;
    }
    if (msg.resize && gpuWorker) {
      gpuWorker.postMessage({ resize: true, width: msg.width, height: msg.height });
      return;
    }

    if (!msg.canvas) return;
    inputSab = msg.inputSab || null;

    if (
      typeof SharedArrayBuffer !== 'undefined' &&
      httpfsMailboxApi &&
      typeof httpfsMailboxApi.createHttpfsMailboxBuffer === 'function'
    ) {
      httpfsMailboxSab = httpfsMailboxApi.createHttpfsMailboxBuffer();
      httpfsClient = new httpfsMailboxApi.HttpfsMailboxClient(httpfsMailboxSab);
    } else {
      httpfsMailboxSab = null;
      httpfsClient = null;
    }

    ioWorker = new Worker('./workers/io_worker.js');
    gpuWorker = new Worker('./workers/wgpu_worker.js');
    audioWorker = new Worker('./workers/audio_worker.js');

    let readyCount = 0;
    const onWorkerReady = (name, detail) => {
      report(`${name} ready${detail ? ` (${detail})` : ''}`);
      if (++readyCount === 3) {
        tryBootWasmRuntime();
      }
    };

    ioWorker.onmessage = (e) => {
      if (e.data.ready) onWorkerReady('io_worker', e.data.summary);
      else if (e.data.error && progress) progress.postMessage({ error: `io_worker: ${e.data.error}` });
    };
    gpuWorker.onmessage = (e) => {
      const data = e && e.data;
      if (!data || typeof data !== 'object') return;
      if (data.ready) {
        onWorkerReady('wgpu_worker', data.adapter);
        return;
      }
      if (data.frame) {
        report(`WebGPU frame #${data.frame} presented`);
      } else if (data.error && progress) {
        progress.postMessage({ error: `wgpu_worker: ${data.error}` });
      }
      forwardGpuWorkerMessage(data, postToMain);
    };
    audioWorker.onmessage = (e) => {
      if (e.data.ready) onWorkerReady('audio_worker');
    };

    const workerInits = createWorkerBootInitPayloads(msg);
    ioWorker.postMessage(workerInits.ioInit);
    if (httpfsMailboxSab) {
      ioWorker.postMessage({ attachMailbox: true, sab: httpfsMailboxSab });
    }
    gpuWorker.postMessage(workerInits.gpuInit, [msg.canvas]);
    audioWorker.postMessage({ init: true });
  };
}

const exportedApi = {
  GPU_OPCODE_PRESENT,
  GPU_PACKET_HEADER_BYTES,
  BOOT_INIT_FORWARD_KEYS,
  inspectGpuPacketBatchForPresent,
  normalizeBootHttpfsPath,
  collectBootBridgeInitFields,
  createWorkerBootInitPayloads,
  forwardGpuWorkerMessage,
  createHostImports,
  resolveRuntimeScriptUrl,
  buildRuntimeModuleConfig,
  createRuntimeModuleOptions,
  encodeTestFramePacket,
  ptrToNumber,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Loader = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
})();
