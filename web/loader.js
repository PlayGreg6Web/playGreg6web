'use strict';

// Main runtime coordinator worker (keeps the browser DOM thread completely free).
(function initPlayGta6Loader() {
const EMSCRIPTEN_PTHREAD_WORKER_NAME = 'em-pthread';
const PROGRESS_CHANNEL_NAME = 'game-progress';
const PTHREAD_WGPU_CHANNEL_NAME = 'playgta6-wgpu-pthread';
const PTHREAD_HTTPFS_SAB_CHANNEL_NAME = 'playgta6-httpfs-sab';
const PTHREAD_TX_MSG_WGPU_SUBMIT = 'wgpu-submit';
const PTHREAD_TX_MSG_PROGRESS = 'progress';
const PTHREAD_TX_MSG_RECYCLE_CMD_BUFFER = 'recycle-cmd-buffer';
const PTHREAD_TX_MSG_FAULT_RECORD = 'fault-record';
const FAULT_RECORD_MESSAGE_TYPE = 'fault-record';
const FAULT_RECORD_PREFIX = '[fault-record]';
const FAULT_RECORD_LINE_RE = /\[fault-record\][ \t]*([^\r\n]*)/i;
const FAULT_RECORD_FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);
const FAULT_RECORD_NUMERIC_KEYS = new Set([
  'thread',
  'elapsed_ms',
  'slot',
  'com_slot',
  'unimplemented_com_slot',
]);
const FLUSH_PIPELINE_DUMPS_MESSAGE_TYPE = 'flush-pipeline-dumps';
const FLUSH_PIPELINE_DUMP_MESSAGE_TYPE = 'flush-pipeline-dump';
const PIPELINE_DUMPS_FLUSHED_MESSAGE_TYPE = 'pipeline-dumps-flushed';
const WGPU_CMD_POOL_MIN_BYTES = 256 * 1024;
const WGPU_CMD_POOL_MAX_BYTES = 32 * 1024 * 1024;
const WGPU_CMD_POOL_MAX_BUFFERS = 3;
const WGPU_CMD_POOL_MAX_SLACK_BYTES = 2 * 1024 * 1024;
const PTHREAD_CMD_LOAD = 'load';
const PTHREAD_CMD_LOADED = 'loaded';
const PTHREAD_CMD_RUN = 'run';
const PTHREAD_CMD_OBJECT_TRANSFER = 'objectTransfer';
const DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS = 2;
const DEFAULT_HTTPFS_MAILBOX_LANES = 8;
const DEFAULT_GUEST_MANIFEST_RELATIVE = './dist/guest_manifest.json';
const INDIRECT_DISPATCH_MODES = new Set(['shard', 'table']);
const V8_SIDE_MODULE_CAP_BYTES = 1024 * 1024 * 1024; // 1024 MiB V8 per-module Wasm cap (#663)
const V8_SIDE_MODULE_WARN_RATIO = 0.85; // Emit headroom warning above 85% of 1 GiB
const FAST_BOOT_EXPORT_NAMES = Object.freeze([
  'playgta6_get_input_state_ptr',
  'playgta6_audio_ring',
  'playgta6_prewarm_load_list',
  'playgta6_prewarm_take',
  'playgta6_prewarm_complete',
  'playgta6_prewarm_progress',
  'playgta6_prewarm_save_list',
  'playgta6_relaunch_requested',
  'playgta6_relaunch_args_ptr',
  'playgta6_storage_generation',
  'playgta6_storage_take_changed',
]);
const HOST_ABI_IMPORT_NAMES = Object.freeze([
  'host_log',
  'host_httpfs_open',
  'host_httpfs_read',
  'host_httpfs_close',
  'host_wgpu_submit',
]);
const OPTIONAL_HOST_ABI_IMPORT_NAMES = Object.freeze([
  'host_httpfs_stat',
]);
const ALL_HOST_ABI_IMPORT_NAMES = Object.freeze([
  ...HOST_ABI_IMPORT_NAMES,
  ...OPTIONAL_HOST_ABI_IMPORT_NAMES,
]);
const HOST_ABI_IMPORT_SET = new Set(HOST_ABI_IMPORT_NAMES);
const ALL_HOST_ABI_IMPORT_SET = new Set(ALL_HOST_ABI_IMPORT_NAMES);
const GUEST_SIDE_MODULE_BASENAME_RE = /^guest_part_\d+\.wasm$/i;
const DEFAULT_RUNTIME_SCRIPT_RELATIVE = './dist/runtime.js';
const DEFAULT_RUNTIME_SCRIPT_FALLBACK = 'dist/runtime.js';
const ABSOLUTE_URL_RE = /^[a-zA-Z][a-zA-Z0-9+.-]*:/;

// GpuPacketHeader / GpuOpcode from runtime/abi/playgta6_abi.h and docs/gpu_commands.md.
// Present is opcode 7 (ClearColor is 2). Detecting opcode 2 as Present would dismiss splash on clear.
const GPU_PACKET_HEADER_BYTES = 8;
const GPU_OPCODE_PRESENT = 7;

// playgta6::InputState (40 bytes = 10 x 32-bit words, runtime/abi/playgta6_abi.h & docs/wasm_abi.md).
const INPUT_STATE_BYTES = 40;
const INPUT_STATE_WORDS = 10;
const DEFAULT_INPUT_BRIDGE_POLL_MS = 16;
const DEFAULT_INPUT_BRIDGE_WAIT_ASYNC_TIMEOUT_MS = 250;
const DEFAULT_INPUT_BRIDGE_WAIT_ASYNC_FALLBACK_POLL_MS = 50;
const INPUT_SNAPSHOT_MAX_RETRIES = 64;
// Playgta6AudioRing (runtime/os/audio_downmix.h): 40 bytes on Wasm64 (8-byte uintptr_t + 4-byte
// padding at +20), 24 bytes on Wasm32 (4-byte uintptr_t).
const AUDIO_RING_STRUCT_BYTES_WASM64 = 40;
const AUDIO_RING_STRUCT_BYTES_WASM32 = 24;
const AUDIO_RING_CHANNELS = 2;
const AUDIO_RING_MIN_CAPACITY_SAMPLES = 2;
const AUDIO_RING_MAX_CAPACITY_SAMPLES = 1 << 24;
const AUDIO_RING_MIN_SAMPLE_RATE = 8000;
const AUDIO_RING_MAX_SAMPLE_RATE = 384000;
const DEFAULT_AUDIO_RING_RETRY_INTERVAL_MS = 250;
const DEFAULT_AUDIO_RING_MAX_RETRIES = 120; // 120 * 250ms = 30s bounded retry window
const AUDIO_RING_REASON = Object.freeze({
  OK: 'ok',
  NULL_PTR: 'null_ptr',
  NO_EXPORT: 'no_export',
  WORKER_NOT_READY: 'worker_not_ready',
  INVALID_STRUCT: 'invalid_struct',
});
const AUDIO_RING_REASON_SET = new Set(Object.values(AUDIO_RING_REASON));
const GUEST_AUDIO_MILESTONE_LOG_RE =
  /(?:xaudio2|playgta6_audio_ring|audio[\s_-]*ring|spatial[\s_-]*audio|audio[\s_-]*downmix|wasapi|iaudiorenderclient)/i;
const DEFAULT_MEMFS_PIPELINE_DUMP_DIR = '/pg6_dump';
const DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT = '/__pg6_dump/';
const DEFAULT_MEMFS_PIPELINE_DUMP_INTERVAL_MS = 3000;
const DEFAULT_MEMFS_PIPELINE_DUMP_MAX_CONCURRENT_UPLOADS = 4;
const MAX_MEMFS_DUMP_WALK_DEPTH = 16;
const utf8Decoder = new TextDecoder('utf-8', { fatal: false });
const utf8Encoder = new TextEncoder();
// Must stay above the em-pthread early-return bootstrap below so pthread workers never hit TDZ.
const SHARED_WASM_STUB_CUSTOM_SECTION_NAME = 'pg6.shared.ref';
const SHARED_WASM_STUB_MAX_BYTES = 16384;

// Emscripten 7.0.1 `-pthread` spawns `new Worker(pthreadMainJs, { name: "em-pthread-" + id })`
// (or legacy `{ name: "em-pthread" }`), which our coordinator Worker Proxy redirects to `loader.js`.
// Detect pthread workers immediately so we bridge `env.host_*` imports and delegate straight
// to `./dist/runtime.js` without overwriting Emscripten's `self.onmessage = handleMessage`.
const initialWorkerScope = typeof self !== 'undefined' ? self : globalThis;
if (isEmscriptenPthreadWorker(initialWorkerScope) && typeof importScripts === 'function') {
  try {
    importScripts('./workers/httpfs_mailbox.js');
  } catch {
    // Degrade safely if httpfs_mailbox.js cannot be loaded inside a pthread worker.
  }
  const pthreadHttpfsMailboxApi =
    (initialWorkerScope && initialWorkerScope.PlayGta6HttpfsMailbox) ||
    (typeof globalThis !== 'undefined' && globalThis.PlayGta6HttpfsMailbox) ||
    null;
  bootstrapEmscriptenPthreadWorker({
    scope: initialWorkerScope,
    importScriptsFn: importScripts,
    runtimeScriptUrl: DEFAULT_RUNTIME_SCRIPT_RELATIVE,
    httpfsMailboxApi: pthreadHttpfsMailboxApi,
  });
  return;
}

if (typeof importScripts === 'function') {
  importScripts('./workers/userdata.js');
  importScripts('./workers/httpfs_mailbox.js');
  try {
    importScripts('./workers/prewarm_bridge.js');
  } catch {
    // Degrade safely if prewarm_bridge.js cannot be loaded in a custom harness.
  }
}

// kRelaunchSelfExitCode (runtime/os/win32_compat.h, ASCII "RELN"): the title called XLaunchNewGame
// on itself. The runtime exits with it and the page relaunches the boot with the same URL/arguments.
const RELAUNCH_SELF_EXIT_CODE = 0x52454c4e;
const GUEST_RELAUNCH_MESSAGE_TYPE = 'guest-relaunch';

/** Progress payload asking the page to relaunch the guest, or null for any other exit status. */
function guestRelaunchPayload(status) {
  // Emscripten may report the int32 status sign-extended; compare as uint32.
  if (typeof status !== 'number' || !Number.isFinite(status)) {
    return null;
  }
  if (status >>> 0 !== RELAUNCH_SELF_EXIT_CODE) {
    return null;
  }
  return { type: GUEST_RELAUNCH_MESSAGE_TYPE, exitCode: RELAUNCH_SELF_EXIT_CODE };
}

function postProgress(payload) {
  if (typeof self === 'undefined') {
    return false;
  }
  const BroadcastChannelCtor =
    typeof self.BroadcastChannel === 'function'
      ? self.BroadcastChannel
      : typeof BroadcastChannel === 'function'
        ? BroadcastChannel
        : null;
  return postBroadcastOnce(BroadcastChannelCtor, PROGRESS_CHANNEL_NAME, payload);
}

const userdataApi =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6Userdata) ||
  (typeof require === 'function' ? require('./workers/userdata.js') : null);
const httpfsMailboxApi =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6HttpfsMailbox) ||
  (typeof require === 'function' ? require('./workers/httpfs_mailbox.js') : null);
const prewarmBridgeApi =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6PrewarmBridge) ||
  (typeof require === 'function' ? require('./workers/prewarm_bridge.js') : null);

const userdataStore = userdataApi ? new userdataApi.UserdataStore() : null;
const BOOT_INIT_FORWARD_KEYS = Object.freeze([
  'shaderPackUrl',
  'reflectionManifestUrl',
  'pgimUrl',
  'bootBase',
  'gameBase',
  'programManifestUrl',
  // #1077: opt-in `?opfsL2=0` bypass of the io_worker OPFS L2 hot-file cache.
  'opfsL2Enabled',
]);
const DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES = 64 * 1024 * 1024;
const GAME_PACKAGE_HTTPFS_EXT_RE = /\.(?:rpf|ttpf|dll|png|xvd|xvc|bin|pri)$/i;
const WIN32_DRIVE_PATH_RE = /^(?:\\\\[?.]\\|\/\/[?.]\/)?([A-Za-z]):(.*)$/;
// W101: render_state_packets.js counters surfaced from wgpu_worker.js `renderStateStats` (W99).
// The first four are always present (0 default); the rest are forwarded only when reported.
const RENDER_STATE_STATS_CORE_KEYS = Object.freeze([
  'computeDispatchCount',
  'computePipelineCount',
  'autoLayoutFallbacks',
  'bindGroupKindMismatches',
]);
const RENDER_STATE_STATS_OPTIONAL_KEYS = Object.freeze([
  'skippedComputeDispatchCount',
  'clampedComputeDispatchCount',
  'rootSigGroup0Layouts',
  'dynamicOffsetRemaps',
  'dynamicOffsetClamps',
  'remappedBindGroupEvictions',
]);
let lastRenderStateStats = null;
const GPU_TELEMETRY_MESSAGE_TYPES = new Set([
  'gpu-stats',
  'gpu-telemetry',
  'missing-shaders',
  'frame-presented',
]);
const DEFAULT_MAX_IN_FLIGHT_GPU_PRESENTS = 4;
const DEFAULT_GPU_PRESENT_BACKPRESSURE_DECAY_MS = 5000;
// Pure/keep-alive frame lifecycle opcodes (NOP=0, BeginFrame=1, ClearColor=2, Present=7).
// Any batch containing other opcodes (resource creation, uploads, pipelines, draws, compute, state)
// is NEVER dropped by Present backpressure.
const GPU_PURE_PRESENT_OPCODES = new Set([0, 1, 2, GPU_OPCODE_PRESENT]);
const GPU_MEMORY_CANONICAL_KEYS = Object.freeze([
  'liveBufferBytes',
  'liveTextureBytes',
  'peakBufferBytes',
  'peakTextureBytes',
  'stagingPoolBytes',
  'totalLiveGpuBytes',
  'pendingDeferredDestroys',
  'pendingDeferredDestroyBytes',
  'deferredHighWaterFlushes',
  'vramBytes',
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
let audioWorkerReady = false;
let inputSab = null;
let activeCoordinatorInputBridge = null;
let activeCoordinatorRuntimeModule = null;
let activeCoordinatorModuleConfig = null;
let coordinatorPostToMain = null;
let lastCoordinatorFaultRecord = null;
let lastCoordinatorFaultRecordLine = null;
let lastCoordinatorFaultRecordConsoleLine = null;
let httpfsMailboxSab = null;
let httpfsMailboxSabs = [];
let httpfsClient = null;
let nextHttpfsBroadcastLane = 0;
let ioWorkerMailboxAttached = false;
let ioWorkerMailboxLanes = 0;
let lastIoWorkerStats = null;
let lastIoStatsPollMs = 0;
const IO_STATS_POLL_INTERVAL_MS = 1000;

function isFaultRecordLine(line) {
  return typeof line === 'string' && FAULT_RECORD_LINE_RE.test(line);
}

function normalizeFaultRecordHex(value) {
  try {
    if (typeof value === 'bigint') {
      return value >= 0n ? `0x${value.toString(16)}` : null;
    }
    if (typeof value === 'number') {
      return Number.isSafeInteger(value) && value >= 0 ? `0x${value.toString(16)}` : null;
    }
    if (typeof value === 'string') {
      const m = /^\s*0x([0-9a-f]{1,16})\s*$/i.exec(value);
      return m ? `0x${BigInt(`0x${m[1]}`).toString(16)}` : null;
    }
  } catch {
    return null;
  }
  return null;
}

function parseFaultRecordKvValue(key, rawValue) {
  if (rawValue === 'null') {
    return null;
  }
  if (rawValue === 'true') {
    return true;
  }
  if (rawValue === 'false') {
    return false;
  }
  if (
    (rawValue.startsWith('{') && rawValue.endsWith('}')) ||
    (rawValue.startsWith('[') && rawValue.endsWith(']'))
  ) {
    try {
      return JSON.parse(rawValue);
    } catch {
      return rawValue;
    }
  }
  if (FAULT_RECORD_NUMERIC_KEYS.has(key) && /^-?\d+$/.test(rawValue)) {
    const parsedInt = Number.parseInt(rawValue, 10);
    if (Number.isSafeInteger(parsedInt)) {
      return parsedInt;
    }
  }
  return rawValue;
}

function parseFaultRecordKv(rawPayload) {
  const out = {};
  let matchedAny = false;
  let i = 0;
  const len = rawPayload.length;

  while (i < len) {
    while (i < len && /\s/.test(rawPayload[i])) {
      i += 1;
    }
    if (i >= len) {
      break;
    }
    const keyStart = i;
    while (i < len && /[A-Za-z0-9_.-]/.test(rawPayload[i])) {
      i += 1;
    }
    const key = rawPayload.slice(keyStart, i);
    if (!key || i >= len || rawPayload[i] !== '=') {
      while (i < len && !/\s/.test(rawPayload[i])) {
        i += 1;
      }
      continue;
    }
    const allowedKey = !FAULT_RECORD_FORBIDDEN_KEYS.has(key);
    i += 1;
    if (i >= len) {
      if (allowedKey) {
        out[key] = '';
        matchedAny = true;
      }
      break;
    }

    const ch = rawPayload[i];
    if (ch === '"' || ch === "'") {
      const quote = ch;
      const quotedStart = i;
      i += 1;
      let buf = '';
      while (i < len && rawPayload[i] !== quote) {
        if (rawPayload[i] === '\\' && i + 1 < len) {
          i += 1;
        }
        buf += rawPayload[i];
        i += 1;
      }
      const hasClosingQuote = i < len && rawPayload[i] === quote;
      if (hasClosingQuote) {
        i += 1;
      }
      let decoded = buf;
      if (quote === '"' && hasClosingQuote) {
        try {
          decoded = JSON.parse(rawPayload.slice(quotedStart, i));
        } catch {
          decoded = buf;
        }
      }
      if (allowedKey) {
        out[key] = decoded;
        matchedAny = true;
      }
      continue;
    }

    if (ch === '{' || ch === '[') {
      const openCh = ch;
      const closeCh = ch === '{' ? '}' : ']';
      let depth = 0;
      let inStr = false;
      let escaped = false;
      const valStart = i;
      while (i < len) {
        const c = rawPayload[i];
        if (inStr) {
          if (escaped) {
            escaped = false;
          } else if (c === '\\') {
            escaped = true;
          } else if (c === '"') {
            inStr = false;
          }
        } else if (c === '"') {
          inStr = true;
        } else if (c === openCh) {
          depth += 1;
        } else if (c === closeCh) {
          depth -= 1;
          if (depth === 0) {
            i += 1;
            break;
          }
        }
        i += 1;
      }
      if (allowedKey) {
        out[key] = parseFaultRecordKvValue(key, rawPayload.slice(valStart, i));
        matchedAny = true;
      }
      continue;
    }

    const valStart = i;
    while (i < len && !/\s/.test(rawPayload[i])) {
      i += 1;
    }
    if (allowedKey) {
      out[key] = parseFaultRecordKvValue(key, rawPayload.slice(valStart, i));
      matchedAny = true;
    }
  }

  return matchedAny ? out : null;
}

function cloneFaultRecord(record) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    return null;
  }
  try {
    return JSON.parse(JSON.stringify(record));
  } catch {
    return Object.assign({}, record);
  }
}

function normalizeFaultRecordPayload(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null;
  }
  const kind = typeof raw.kind === 'string' ? raw.kind.trim() : '';
  if (!kind) {
    return null;
  }
  const cloned = cloneFaultRecord(raw);
  if (!cloned) {
    return null;
  }
  cloned.kind = kind;
  const normVa = normalizeFaultRecordHex(raw.va);
  if (normVa !== null) {
    cloned.va = normVa;
  }
  const normRip = normalizeFaultRecordHex(raw.rip);
  if (normRip !== null) {
    cloned.rip = normRip;
  }
  const normTarget = normalizeFaultRecordHex(raw.target);
  if (normTarget !== null) {
    cloned.target = normTarget;
  } else if (cloned.target === undefined && typeof cloned.va === 'string' && cloned.va.length > 0) {
    cloned.target = cloned.va;
  }
  return cloned;
}

function parseFaultRecordLine(line) {
  if (typeof line !== 'string' || line.length === 0) {
    return null;
  }
  const match = FAULT_RECORD_LINE_RE.exec(line);
  if (!match) {
    return null;
  }
  const payload = match[1].trim();
  if (!payload) {
    return null;
  }
  let parsed = null;
  if (payload.startsWith('{')) {
    try {
      parsed = JSON.parse(payload);
    } catch {
      return null;
    }
  } else {
    parsed = parseFaultRecordKv(payload);
  }
  return normalizeFaultRecordPayload(parsed);
}

function formatFaultRecordConsoleLine(rawLine, faultRecord) {
  if (typeof rawLine === 'string' && isFaultRecordLine(rawLine)) {
    return rawLine;
  }
  if (faultRecord && typeof faultRecord === 'object') {
    try {
      return `[wasm] ${FAULT_RECORD_PREFIX} ${JSON.stringify(faultRecord)}`;
    } catch {
      return `[wasm] ${FAULT_RECORD_PREFIX} {"kind":"unknown"}`;
    }
  }
  return `[wasm] ${FAULT_RECORD_PREFIX}`;
}

function emitFaultRecordConsoleError(line, consoleObj) {
  const targetConsole =
    consoleObj && typeof consoleObj.error === 'function'
      ? consoleObj
      : typeof console !== 'undefined' && console && typeof console.error === 'function'
        ? console
        : null;
  if (!targetConsole) {
    return false;
  }
  try {
    targetConsole.error(line);
    return true;
  } catch {
    return false;
  }
}

function getLastFaultRecord() {
  return lastCoordinatorFaultRecord ? cloneFaultRecord(lastCoordinatorFaultRecord) : null;
}

function resetLastFaultRecord() {
  lastCoordinatorFaultRecord = null;
  lastCoordinatorFaultRecordLine = null;
  lastCoordinatorFaultRecordConsoleLine = null;
  if (typeof globalThis !== 'undefined') {
    try {
      globalThis.__playgta6LastFaultRecord = null;
    } catch {
      // Ignore non-writable globalThis property.
    }
  }
}

function recordCoordinatorFaultRecord(rawRecord, meta = {}) {
  const normalized =
    normalizeFaultRecordPayload(rawRecord) ||
    parseFaultRecordLine(meta.rawLine || meta.label || '');
  if (!normalized) {
    return null;
  }
  const consoleLine = formatFaultRecordConsoleLine(meta.rawLine || meta.label, normalized);
  lastCoordinatorFaultRecord = cloneFaultRecord(normalized);
  lastCoordinatorFaultRecordLine = consoleLine;
  if (typeof globalThis !== 'undefined') {
    try {
      globalThis.__playgta6LastFaultRecord = cloneFaultRecord(normalized);
    } catch {
      // Ignore non-writable globalThis property.
    }
  }

  const shouldEmitConsole =
    meta.consoleEmitted !== true &&
    (meta.forceConsole === true || lastCoordinatorFaultRecordConsoleLine !== consoleLine);
  if (shouldEmitConsole) {
    lastCoordinatorFaultRecordConsoleLine = consoleLine;
    emitFaultRecordConsoleError(consoleLine, meta.console);
  }

  const postFn =
    typeof meta.postToMain === 'function'
      ? meta.postToMain
      : typeof coordinatorPostToMain === 'function'
        ? coordinatorPostToMain
        : null;
  if (postFn) {
    try {
      postFn({
        type: FAULT_RECORD_MESSAGE_TYPE,
        faultRecord: cloneFaultRecord(normalized),
        rawLine: consoleLine,
        label: typeof meta.label === 'string' && meta.label.length > 0 ? meta.label : consoleLine,
        source: typeof meta.source === 'string' ? meta.source : 'wasm',
        workerName: typeof meta.workerName === 'string' ? meta.workerName : null,
        onAbort: meta.onAbort === true,
        timestamp: Date.now(),
      });
    } catch {
      // Non-fatal postMessage failure on fault-record forwarding.
    }
  }

  if (
    meta.flushDumps !== false &&
    activeCoordinatorPipelineDumpSyncer &&
    activeCoordinatorPipelineDumpSyncer.enabled &&
    typeof activeCoordinatorPipelineDumpSyncer.flush === 'function'
  ) {
    try {
      const p = activeCoordinatorPipelineDumpSyncer.flush({ snapshotFiles: true });
      if (p && typeof p.catch === 'function') {
        p.catch(() => {});
      }
    } catch {
      // Pipeline dump flush must never block fault-record capture.
    }
  }

  return cloneFaultRecord(normalized);
}

function flushActiveCoordinatorPipelineDumps(options = {}) {
  const flushOpts =
    options && typeof options === 'object' ? options : { snapshotFiles: true };
  if (
    activeCoordinatorPipelineDumpSyncer &&
    typeof activeCoordinatorPipelineDumpSyncer.flush === 'function'
  ) {
    try {
      return Promise.resolve(activeCoordinatorPipelineDumpSyncer.flush(flushOpts));
    } catch (err) {
      return Promise.reject(err);
    }
  }
  if (
    activeCoordinatorModuleConfig &&
    typeof activeCoordinatorModuleConfig.flushPipelineDumps === 'function'
  ) {
    try {
      return Promise.resolve(activeCoordinatorModuleConfig.flushPipelineDumps(flushOpts));
    } catch (err) {
      return Promise.reject(err);
    }
  }
  return Promise.resolve({
    enabled: false,
    filesUploaded: 0,
    bytesUploaded: 0,
    uploadErrors: 0,
    lastFlushAtMs: null,
    pendingCount: 0,
  });
}

function normalizeHttpfsMailboxLanes(value, fallback = DEFAULT_HTTPFS_MAILBOX_LANES) {
  const defaultLanes =
    Number.isInteger(fallback) && fallback >= 1 ? fallback : DEFAULT_HTTPFS_MAILBOX_LANES;
  if (value === undefined || value === null || value === '') {
    return defaultLanes;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return defaultLanes;
  }
  return Math.floor(parsed);
}

function allocateHttpfsMailboxPool(
  lanes = DEFAULT_HTTPFS_MAILBOX_LANES,
  mailboxApi = httpfsMailboxApi,
) {
  if (typeof SharedArrayBuffer === 'undefined' || !mailboxApi) {
    return { sab: null, sabs: [], lanes: 0 };
  }
  const requestedLanes = normalizeHttpfsMailboxLanes(lanes, DEFAULT_HTTPFS_MAILBOX_LANES);
  const maxSlots =
    Number.isInteger(mailboxApi.HTTPFS_MAX_PIPELINE_SLOTS) &&
    mailboxApi.HTTPFS_MAX_PIPELINE_SLOTS >= 1
      ? mailboxApi.HTTPFS_MAX_PIPELINE_SLOTS
      : 16;
  const targetLanes = Math.min(requestedLanes, maxSlots);

  if (targetLanes > 1 && typeof mailboxApi.createHttpfsMailboxPool === 'function') {
    const sabs = mailboxApi.createHttpfsMailboxPool(targetLanes);
    if (Array.isArray(sabs) && sabs.length > 0) {
      return {
        sab: sabs[0] || null,
        sabs: sabs.slice(),
        lanes: sabs.length,
      };
    }
  }
  if (typeof mailboxApi.createHttpfsMailboxBuffer === 'function') {
    const sab = mailboxApi.createHttpfsMailboxBuffer();
    return {
      sab,
      sabs: sab ? [sab] : [],
      lanes: sab ? 1 : 0,
    };
  }
  return { sab: null, sabs: [], lanes: 0 };
}

function report(label) {
  postProgress({ label });
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

function isSharedArrayBuffer(value) {
  if (!value || typeof value !== 'object') {
    return false;
  }
  if (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer) {
    return true;
  }
  return Object.prototype.toString.call(value) === '[object SharedArrayBuffer]';
}

function isTransferableArrayBuffer(value) {
  if (!value || typeof value !== 'object' || isSharedArrayBuffer(value)) {
    return false;
  }
  return (
    value instanceof ArrayBuffer ||
    Object.prototype.toString.call(value) === '[object ArrayBuffer]'
  );
}

/**
 * Return a standalone transferable `ArrayBuffer` for a GPU command batch.
 *
 * When `bytes` is already a standalone non-shared `ArrayBuffer` or a view covering
 * the full length of a non-shared `ArrayBuffer` (such as the fresh `Uint8Array`
 * returned by `copyMemoryBytes()`), its backing buffer is returned directly with
 * zero extra allocation or memcpy. Subviews and `SharedArrayBuffer`-backed views
 * fall back to copying into a fresh `ArrayBuffer`.
 */
function toTransferableArrayBuffer(bytes) {
  if (!bytes) {
    return null;
  }
  if (isTransferableArrayBuffer(bytes)) {
    return bytes;
  }
  if (ArrayBuffer.isView(bytes)) {
    const buf = bytes.buffer;
    if (
      isTransferableArrayBuffer(buf) &&
      bytes.byteOffset === 0 &&
      bytes.byteLength === buf.byteLength
    ) {
      return buf;
    }
    const copy = new Uint8Array(bytes.byteLength);
    if (bytes.byteLength > 0) {
      copy.set(new Uint8Array(buf, bytes.byteOffset, bytes.byteLength));
    }
    return copy.buffer;
  }
  if (isSharedArrayBuffer(bytes)) {
    const copy = new Uint8Array(bytes.byteLength);
    if (bytes.byteLength > 0) {
      copy.set(new Uint8Array(bytes));
    }
    return copy.buffer;
  }
  return null;
}

/**
 * Bounded pool of warm transferable `ArrayBuffer`s for large GPU command batches (G75, #663).
 * Reusing already-faulted buffers returned from `wgpu_worker` avoids per-frame OS zero-page
 * faults when streaming 4K Bink2 texture uploads (~12 MiB/frame).
 */
function createWarmCmdBufferPool(options = {}) {
  const minBytes =
    Number.isInteger(options.minBytes) && options.minBytes > 0
      ? options.minBytes
      : WGPU_CMD_POOL_MIN_BYTES;
  const maxBytes =
    Number.isInteger(options.maxBytes) && options.maxBytes >= minBytes
      ? options.maxBytes
      : WGPU_CMD_POOL_MAX_BYTES;
  const maxBuffers =
    Number.isInteger(options.maxBuffers) && options.maxBuffers > 0
      ? options.maxBuffers
      : WGPU_CMD_POOL_MAX_BUFFERS;
  const maxSlackBytes =
    Number.isInteger(options.maxSlackBytes) && options.maxSlackBytes >= 0
      ? options.maxSlackBytes
      : WGPU_CMD_POOL_MAX_SLACK_BYTES;
  const buffers = [];

  return {
    get size() {
      return buffers.length;
    },
    isEligibleSize(byteLength) {
      return (
        typeof byteLength === 'number' &&
        Number.isInteger(byteLength) &&
        byteLength >= minBytes &&
        byteLength <= maxBytes
      );
    },
    acquire(byteLength) {
      if (!this.isEligibleSize(byteLength) || buffers.length === 0) {
        return null;
      }
      const upperBound = Math.max(byteLength * 2, byteLength + maxSlackBytes);
      let bestIndex = -1;
      let bestSize = Infinity;
      for (let i = buffers.length - 1; i >= 0; i -= 1) {
        const buf = buffers[i];
        if (!isTransferableArrayBuffer(buf) || buf.byteLength === 0) {
          buffers.splice(i, 1);
          continue;
        }
      }
      for (let i = 0; i < buffers.length; i += 1) {
        const buf = buffers[i];
        const len = buf.byteLength;
        if (len >= byteLength && len <= upperBound && len < bestSize) {
          bestIndex = i;
          bestSize = len;
        }
      }
      if (bestIndex < 0) {
        return null;
      }
      const [picked] = buffers.splice(bestIndex, 1);
      return picked;
    },
    recycle(buf) {
      if (!isTransferableArrayBuffer(buf)) {
        return false;
      }
      const len = buf.byteLength;
      if (len < minBytes || len > maxBytes) {
        return false;
      }
      if (buffers.length >= maxBuffers) {
        buffers.shift();
      }
      buffers.push(buf);
      return true;
    },
    clear() {
      buffers.length = 0;
    },
  };
}

function asDataView(bytes) {
  if (isTransferableArrayBuffer(bytes) || isSharedArrayBuffer(bytes)) {
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

/**
 * Inspect a GpuPacketHeader stream (bounded to optional `explicitByteLength`) for
 * Present backpressure classification: identifies whether the batch contains a
 * Present packet and whether it is a pure/keep-alive Present batch (only NOP,
 * BeginFrame, ClearColor, Present — zero resource/state/draw/compute packets).
 * Never throws on truncated or malformed input.
 */
function inspectGpuPacketBatchForBackpressure(bytes, explicitByteLength) {
  const empty = {
    packetCount: 0,
    hasPresent: false,
    isPurePresent: false,
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
  const total =
    typeof explicitByteLength === 'number' &&
    Number.isInteger(explicitByteLength) &&
    explicitByteLength >= 0 &&
    explicitByteLength <= view.byteLength
      ? explicitByteLength
      : view.byteLength;
  if (total === 0) {
    return empty;
  }

  let packetCount = 0;
  let hasPresent = false;
  let hasNonPurePacket = false;
  let truncated = false;
  let offset = 0;

  while (offset < total) {
    if (offset + GPU_PACKET_HEADER_BYTES > total) {
      truncated = true;
      break;
    }
    let opcode;
    let payloadBytes;
    try {
      opcode = view.getUint16(offset, true);
      payloadBytes = view.getUint32(offset + 4, true);
    } catch {
      truncated = true;
      break;
    }
    const payloadOffset = offset + GPU_PACKET_HEADER_BYTES;
    const remaining = total - payloadOffset;
    if (payloadBytes > remaining) {
      truncated = true;
      break;
    }
    if (opcode === GPU_OPCODE_PRESENT) {
      hasPresent = true;
    }
    if (!GPU_PURE_PRESENT_OPCODES.has(opcode)) {
      hasNonPurePacket = true;
    }
    packetCount += 1;
    offset = payloadOffset + payloadBytes;
  }

  return {
    packetCount,
    hasPresent,
    isPurePresent: hasPresent && !truncated && !hasNonPurePacket,
    truncated,
  };
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

// Returns a numeric-only copy of a renderStateStats object, or null when absent/invalid.
function sanitizeRenderStateStats(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null;
  }
  const out = {};
  for (const key of RENDER_STATE_STATS_CORE_KEYS) {
    out[key] = toNonNegativeInt(raw[key], 0);
  }
  for (const key of RENDER_STATE_STATS_OPTIONAL_KEYS) {
    if (raw[key] !== undefined && raw[key] !== null) {
      out[key] = toNonNegativeInt(raw[key], 0);
    }
  }
  return out;
}

function sanitizeGpuMemoryStats(rawObj, fallbackSources = null) {
  const out = {};
  const copyNumericFields = (src, allowAllOwnKeys) => {
    if (!src || typeof src !== 'object' || Array.isArray(src)) {
      return;
    }
    const keys = allowAllOwnKeys ? Object.keys(src) : GPU_MEMORY_CANONICAL_KEYS;
    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      if (out[key] !== undefined) {
        continue;
      }
      const val = src[key];
      if (typeof val === 'number' && Number.isFinite(val) && val >= 0) {
        out[key] = Number.isInteger(val) ? val : val;
      }
    }
  };

  if (rawObj && typeof rawObj === 'object' && !Array.isArray(rawObj)) {
    copyNumericFields(rawObj, true);
  }
  if (Array.isArray(fallbackSources)) {
    for (let i = 0; i < fallbackSources.length; i += 1) {
      const src = fallbackSources[i];
      if (!src || typeof src !== 'object' || Array.isArray(src)) {
        continue;
      }
      if (src.gpuMemory && typeof src.gpuMemory === 'object' && !Array.isArray(src.gpuMemory)) {
        copyNumericFields(src.gpuMemory, true);
      }
      copyNumericFields(src, false);
    }
  }

  const keys = Object.keys(out);
  if (keys.length === 0) {
    return null;
  }
  if (
    out.totalLiveGpuBytes === undefined &&
    (out.liveBufferBytes !== undefined ||
      out.liveTextureBytes !== undefined ||
      out.stagingPoolBytes !== undefined)
  ) {
    out.totalLiveGpuBytes =
      (out.liveBufferBytes || 0) + (out.liveTextureBytes || 0) + (out.stagingPoolBytes || 0);
  }
  return out;
}

function sanitizeFrameStats(eventData) {
  const rawFrameStats =
    eventData.frameStats && typeof eventData.frameStats === 'object' ? eventData.frameStats : null;
  const rawStats =
    eventData.stats && typeof eventData.stats === 'object' ? eventData.stats : null;
  const rawGpuMemory =
    (rawFrameStats && rawFrameStats.gpuMemory && typeof rawFrameStats.gpuMemory === 'object'
      ? rawFrameStats.gpuMemory
      : null) ||
    (rawStats && rawStats.gpuMemory && typeof rawStats.gpuMemory === 'object'
      ? rawStats.gpuMemory
      : null) ||
    (eventData.gpuMemory && typeof eventData.gpuMemory === 'object' ? eventData.gpuMemory : null);

  const hasFrameField =
    Boolean(rawFrameStats) ||
    Boolean(rawStats) ||
    Boolean(rawGpuMemory) ||
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

  const realPackDrawCountSrc =
    rawFrameStats &&
    rawFrameStats.realPackDrawCount !== undefined &&
    rawFrameStats.realPackDrawCount !== null
      ? rawFrameStats.realPackDrawCount
      : rawStats &&
          rawStats.realPackDrawCount !== undefined &&
          rawStats.realPackDrawCount !== null
        ? rawStats.realPackDrawCount
        : eventData.realPackDrawCount;
  if (realPackDrawCountSrc !== undefined && realPackDrawCountSrc !== null) {
    frameStats.realPackDrawCount = toNonNegativeInt(realPackDrawCountSrc, 0);
  }

  const fallbackDrawCountSrc =
    rawFrameStats &&
    rawFrameStats.fallbackDrawCount !== undefined &&
    rawFrameStats.fallbackDrawCount !== null
      ? rawFrameStats.fallbackDrawCount
      : rawStats &&
          rawStats.fallbackDrawCount !== undefined &&
          rawStats.fallbackDrawCount !== null
        ? rawStats.fallbackDrawCount
        : eventData.fallbackDrawCount;
  if (fallbackDrawCountSrc !== undefined && fallbackDrawCountSrc !== null) {
    frameStats.fallbackDrawCount = toNonNegativeInt(fallbackDrawCountSrc, 0);
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

  const packetCountSrc =
    rawFrameStats && rawFrameStats.packetCount !== undefined && rawFrameStats.packetCount !== null
      ? rawFrameStats.packetCount
      : rawStats && rawStats.packetCount !== undefined && rawStats.packetCount !== null
        ? rawStats.packetCount
        : eventData.packetCount !== undefined && eventData.packetCount !== null
          ? eventData.packetCount
          : undefined;
  const packetsProcessedSrc =
    rawFrameStats &&
    rawFrameStats.packetsProcessed !== undefined &&
    rawFrameStats.packetsProcessed !== null
      ? rawFrameStats.packetsProcessed
      : rawStats && rawStats.packetsProcessed !== undefined && rawStats.packetsProcessed !== null
        ? rawStats.packetsProcessed
        : eventData.packetsProcessed !== undefined && eventData.packetsProcessed !== null
          ? eventData.packetsProcessed
          : undefined;
  const packetSrc = packetCountSrc !== undefined ? packetCountSrc : packetsProcessedSrc;
  if (packetSrc !== undefined && packetSrc !== null) {
    frameStats.packetCount = toNonNegativeInt(packetSrc, 0);
  }
  if (packetsProcessedSrc !== undefined) {
    frameStats.packetsProcessed = toNonNegativeInt(packetsProcessedSrc, 0);
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

  // #1077: preserve structured WebGPU VRAM breakdown (`gpuMemory`) when reported by wgpu_worker.
  const gpuMemoryFallbackSources = [rawFrameStats, rawStats, eventData].filter(Boolean);
  const hasFlatGpuMemField = gpuMemoryFallbackSources.some(
    (src) =>
      typeof src.liveBufferBytes === 'number' ||
      typeof src.liveTextureBytes === 'number' ||
      typeof src.peakBufferBytes === 'number' ||
      typeof src.peakTextureBytes === 'number' ||
      typeof src.stagingPoolBytes === 'number' ||
      typeof src.totalLiveGpuBytes === 'number',
  );
  if (rawGpuMemory || hasFlatGpuMemField) {
    const sanitizedGpuMem = sanitizeGpuMemoryStats(rawGpuMemory, gpuMemoryFallbackSources);
    if (sanitizedGpuMem) {
      frameStats.gpuMemory = sanitizedGpuMem;
      if (frameStats.vramBytes === undefined && sanitizedGpuMem.totalLiveGpuBytes !== undefined) {
        frameStats.vramBytes = toNonNegativeInt(sanitizedGpuMem.totalLiveGpuBytes, 0);
      }
    }
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
    (eventData.stats && typeof eventData.stats === 'object') ||
    (eventData.gpuMemory && typeof eventData.gpuMemory === 'object' && !Array.isArray(eventData.gpuMemory))
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

function formatGpuWorkerError(err) {
  if (err && typeof err === 'object') {
    const prefix = typeof err.type === 'string' && err.type.length > 0 ? `[${err.type}] ` : '';
    if (typeof err.message === 'string' && err.message.length > 0) {
      return `${prefix}${err.message}`;
    }
    try {
      return `${prefix}${JSON.stringify(err)}`;
    } catch {
      return `${prefix}${String(err)}`;
    }
  }
  return String(err);
}

function sanitizeGpuAdapterForTelemetry(raw) {
  if (raw === undefined) {
    return undefined;
  }
  if (raw === null) {
    return null;
  }
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  if (typeof raw === 'object' && !Array.isArray(raw)) {
    const out = {};
    let hasField = false;
    for (const key of ['vendor', 'architecture', 'device', 'description', 'label']) {
      if (typeof raw[key] === 'string') {
        out[key] = raw[key];
        hasField = true;
      }
    }
    if (typeof raw.isFallbackAdapter === 'boolean') {
      out.isFallbackAdapter = raw.isFallbackAdapter;
      hasField = true;
    }
    return hasField ? out : null;
  }
  return undefined;
}

function pickTelemetryBoolean(sources, keys) {
  for (let i = 0; i < sources.length; i += 1) {
    const src = sources[i];
    if (!src || typeof src !== 'object') continue;
    for (let j = 0; j < keys.length; j += 1) {
      const val = src[keys[j]];
      if (typeof val === 'boolean') {
        return { found: true, value: val };
      }
    }
  }
  return { found: false, value: undefined };
}

function pickTelemetryBooleanOrNull(sources, keys) {
  for (let i = 0; i < sources.length; i += 1) {
    const src = sources[i];
    if (!src || typeof src !== 'object') continue;
    for (let j = 0; j < keys.length; j += 1) {
      const val = src[keys[j]];
      if (val === null || typeof val === 'boolean') {
        return { found: true, value: val };
      }
    }
  }
  return { found: false, value: undefined };
}

function pickTelemetryNumberOrNull(sources, keys) {
  for (let i = 0; i < sources.length; i += 1) {
    const src = sources[i];
    if (!src || typeof src !== 'object') continue;
    for (let j = 0; j < keys.length; j += 1) {
      const val = src[keys[j]];
      if (val === null) {
        return { found: true, value: null };
      }
      if (typeof val === 'number' && Number.isFinite(val) && val >= 0) {
        return { found: true, value: val };
      }
    }
  }
  return { found: false, value: undefined };
}

function pickTelemetryFiniteNumber(sources, keys) {
  for (let i = 0; i < sources.length; i += 1) {
    const src = sources[i];
    if (!src || typeof src !== 'object') continue;
    for (let j = 0; j < keys.length; j += 1) {
      const val = src[keys[j]];
      if (typeof val === 'number' && Number.isFinite(val)) {
        return { found: true, value: val };
      }
    }
  }
  return { found: false, value: undefined };
}

// W358: forward swapchain luminance probe counters and capability flags across
// first-visible-frame and gpu-telemetry payloads (camelCase, snake_case, nested stats/frameStats).
function applySwapchainProbeTelemetryFields(target, sources) {
  const probeCountField = pickTelemetryFiniteNumber(sources, [
    'swapchainProbeCount',
    'swapchain_probe_count',
  ]);
  if (probeCountField.found) {
    target.swapchainProbeCount = toNonNegativeInt(probeCountField.value, 0);
  }
  const probeInFlightField = pickTelemetryBoolean(sources, [
    'swapchainProbeInFlight',
    'swapchain_probe_in_flight',
  ]);
  if (probeInFlightField.found) {
    target.swapchainProbeInFlight = probeInFlightField.value;
  }
  const probeSupportedField = pickTelemetryBooleanOrNull(sources, [
    'swapchainProbeSupported',
    'swapchain_probe_supported',
  ]);
  if (probeSupportedField.found) {
    target.swapchainProbeSupported = probeSupportedField.value;
  }
  const probeErrorCountField = pickTelemetryFiniteNumber(sources, [
    'swapchainProbeErrorCount',
    'swapchain_probe_error_count',
  ]);
  if (probeErrorCountField.found) {
    target.swapchainProbeErrorCount = toNonNegativeInt(probeErrorCountField.value, 0);
  }
  const probeTimeoutCountField = pickTelemetryFiniteNumber(sources, [
    'swapchainProbeTimeoutCount',
    'swapchain_probe_timeout_count',
  ]);
  if (probeTimeoutCountField.found) {
    target.swapchainProbeTimeoutCount = toNonNegativeInt(probeTimeoutCountField.value, 0);
  }
}

/**
 * Inspect a message emitted by `wgpu_worker.js`, sanitize/normalize GPU frame,
 * shader-fallback, missing-shader, first-visible-frame, and validation-error telemetry,
 * and forward to the main thread via `postMessageFn`. Returns the forwarded payload or `null` when ignored.
 */
function forwardGpuWorkerMessage(eventData, postMessageFn) {
  if (
    eventData &&
    typeof eventData === 'object' &&
    !Array.isArray(eventData) &&
    eventData.type === 'gpu-validation-error'
  ) {
    const rawErr =
      eventData.error && typeof eventData.error === 'object' ? eventData.error : null;
    const rawTs = rawErr
      ? rawErr.timestampMs
      : eventData.timestampMs !== undefined
        ? eventData.timestampMs
        : eventData.timestamp;
    const timestampMs =
      typeof rawTs === 'number' && Number.isFinite(rawTs) ? rawTs : Date.now();
    const cleanErr = rawErr
      ? {
          seq: toNonNegativeInt(rawErr.seq, 0),
          type:
            typeof rawErr.type === 'string' && rawErr.type.length > 0
              ? rawErr.type
              : typeof rawErr.errorType === 'string' && rawErr.errorType.length > 0
                ? rawErr.errorType
                : 'validation',
          message:
            rawErr.message !== undefined && rawErr.message !== null
              ? String(rawErr.message)
              : '',
          timestampMs,
        }
      : {
          seq: toNonNegativeInt(eventData.seq, 0),
          type:
            typeof eventData.errorType === 'string' && eventData.errorType.length > 0
              ? eventData.errorType
              : 'validation',
          message:
            eventData.message !== undefined && eventData.message !== null
              ? String(eventData.message)
              : typeof eventData.error === 'string'
                ? eventData.error
                : '',
          timestampMs,
        };
    const payload = {
      type: 'gpu-validation-error',
      error: cleanErr,
      totalCount: toNonNegativeInt(eventData.totalCount, cleanErr.seq || 1),
    };
    if (typeof eventData.source === 'string' && eventData.source.length > 0) {
      payload.source = eventData.source;
    }
    if (typeof postMessageFn === 'function') {
      try {
        postMessageFn(payload);
      } catch {
        // Telemetry forwarding must never disrupt the GPU worker loop.
      }
    }
    return payload;
  }

  // W349: forward explicit first-visible-frame / visible-frame events straight through to main thread.
  if (
    eventData &&
    typeof eventData === 'object' &&
    !Array.isArray(eventData) &&
    (eventData.type === 'first-visible-frame' || eventData.type === 'visible-frame')
  ) {
    const payload = {
      type: eventData.type,
      firstVisibleFrame:
        typeof eventData.firstVisibleFrame === 'boolean' ? eventData.firstVisibleFrame : true,
    };
    if (
      eventData.firstVisibleFrameMs === null ||
      (typeof eventData.firstVisibleFrameMs === 'number' &&
        Number.isFinite(eventData.firstVisibleFrameMs) &&
        eventData.firstVisibleFrameMs >= 0)
    ) {
      payload.firstVisibleFrameMs = eventData.firstVisibleFrameMs;
    }
    if (
      eventData.firstPresentMs === null ||
      (typeof eventData.firstPresentMs === 'number' &&
        Number.isFinite(eventData.firstPresentMs) &&
        eventData.firstPresentMs >= 0)
    ) {
      payload.firstPresentMs = eventData.firstPresentMs;
    }
    if (
      eventData.blackLoadingMs === null ||
      (typeof eventData.blackLoadingMs === 'number' &&
        Number.isFinite(eventData.blackLoadingMs) &&
        eventData.blackLoadingMs >= 0)
    ) {
      payload.blackLoadingMs = eventData.blackLoadingMs;
    }
    if (typeof eventData.presentCount === 'number' && Number.isFinite(eventData.presentCount)) {
      payload.presentCount = toNonNegativeInt(eventData.presentCount, 0);
    }
    if (
      typeof eventData.framesPresented === 'number' &&
      Number.isFinite(eventData.framesPresented)
    ) {
      payload.framesPresented = toNonNegativeInt(eventData.framesPresented, 0);
    }
    if (
      typeof eventData.realPackDrawsThisFrame === 'number' &&
      Number.isFinite(eventData.realPackDrawsThisFrame)
    ) {
      payload.realPackDrawsThisFrame = toNonNegativeInt(eventData.realPackDrawsThisFrame, 0);
    }
    if (
      typeof eventData.fallbackDrawsThisFrame === 'number' &&
      Number.isFinite(eventData.fallbackDrawsThisFrame)
    ) {
      payload.fallbackDrawsThisFrame = toNonNegativeInt(eventData.fallbackDrawsThisFrame, 0);
    }
    if (
      typeof eventData.drawsThisFrame === 'number' &&
      Number.isFinite(eventData.drawsThisFrame)
    ) {
      payload.drawsThisFrame = toNonNegativeInt(eventData.drawsThisFrame, 0);
    }
    if (
      typeof eventData.realPackDrawCount === 'number' &&
      Number.isFinite(eventData.realPackDrawCount)
    ) {
      payload.realPackDrawCount = toNonNegativeInt(eventData.realPackDrawCount, 0);
    }
    if (typeof eventData.source === 'string' && eventData.source.length > 0) {
      payload.source = eventData.source;
    }
    const rawMaxLuma =
      eventData.maxLuma !== undefined ? eventData.maxLuma : eventData.lastSampledMaxLuma;
    if (typeof rawMaxLuma === 'number' && Number.isFinite(rawMaxLuma)) {
      payload.maxLuma = Math.max(0, Math.min(255, Math.floor(rawMaxLuma)));
    }
    if (
      typeof eventData.suppressedBlackFrames === 'number' &&
      Number.isFinite(eventData.suppressedBlackFrames)
    ) {
      payload.suppressedBlackFrames = toNonNegativeInt(eventData.suppressedBlackFrames, 0);
    }
    const visibleRawStats =
      eventData.stats && typeof eventData.stats === 'object' ? eventData.stats : null;
    const visibleRawFrameStats =
      eventData.frameStats && typeof eventData.frameStats === 'object'
        ? eventData.frameStats
        : null;
    const visibleSources = [eventData, visibleRawStats, visibleRawFrameStats].filter(Boolean);
    applySwapchainProbeTelemetryFields(payload, visibleSources);
    if (eventData.gpuAdapter !== undefined) {
      const sanitizedAdapter = sanitizeGpuAdapterForTelemetry(eventData.gpuAdapter);
      if (sanitizedAdapter !== undefined) {
        payload.gpuAdapter = sanitizedAdapter;
      }
    }
    if (
      typeof eventData.elapsedMs === 'number' &&
      Number.isFinite(eventData.elapsedMs) &&
      eventData.elapsedMs >= 0
    ) {
      payload.elapsedMs = eventData.elapsedMs;
    }
    if (typeof eventData.fadeOutMs === 'number' && Number.isFinite(eventData.fadeOutMs)) {
      payload.fadeOutMs = eventData.fadeOutMs;
    }
    if (
      typeof eventData.fadeDurationMs === 'number' &&
      Number.isFinite(eventData.fadeDurationMs)
    ) {
      payload.fadeDurationMs = eventData.fadeDurationMs;
    }
    if (typeof eventData.fadeMs === 'number' && Number.isFinite(eventData.fadeMs)) {
      payload.fadeMs = eventData.fadeMs;
    }
    if (typeof eventData.timestamp === 'number' && Number.isFinite(eventData.timestamp)) {
      payload.timestamp = eventData.timestamp;
    }
    if (typeof postMessageFn === 'function') {
      try {
        postMessageFn(payload);
      } catch {
        // Telemetry forwarding must never disrupt the GPU worker loop.
      }
    }
    return payload;
  }

  if (!isGpuTelemetryMessage(eventData)) {
    return null;
  }

  const rawStats =
    eventData.stats && typeof eventData.stats === 'object' ? eventData.stats : null;
  const rawFrameStats =
    eventData.frameStats && typeof eventData.frameStats === 'object' ? eventData.frameStats : null;
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
  const renderStateStats = sanitizeRenderStateStats(
    eventData.renderStateStats ||
      (rawStats && (rawStats.renderStateStats || rawStats.renderState)) ||
      null
  );
  if (renderStateStats) {
    telemetry.renderStateStats = renderStateStats;
    lastRenderStateStats = renderStateStats;
  }
  if (telemetry.frameStats && telemetry.frameStats.gpuMemory) {
    telemetry.gpuMemory = Object.assign({}, telemetry.frameStats.gpuMemory);
  }

  // W349: preserve explicit non-black probe & adapter fields on gpu-telemetry payloads.
  const telemetrySources = [eventData, rawStats, rawFrameStats].filter(Boolean);
  const firstVisibleField = pickTelemetryBoolean(telemetrySources, [
    'firstVisibleFrame',
    'first_visible_frame',
  ]);
  if (firstVisibleField.found) {
    telemetry.firstVisibleFrame = firstVisibleField.value;
  }
  const firstVisibleMsField = pickTelemetryNumberOrNull(telemetrySources, [
    'firstVisibleFrameMs',
    'first_visible_frame_ms',
  ]);
  if (firstVisibleMsField.found) {
    telemetry.firstVisibleFrameMs = firstVisibleMsField.value;
  }
  const blackLoadingMsField = pickTelemetryNumberOrNull(telemetrySources, [
    'blackLoadingMs',
    'black_loading_ms',
  ]);
  if (blackLoadingMsField.found) {
    telemetry.blackLoadingMs = blackLoadingMsField.value;
  }
  const maxLumaField = pickTelemetryFiniteNumber(telemetrySources, [
    'maxLuma',
    'lastSampledMaxLuma',
  ]);
  if (maxLumaField.found) {
    telemetry.maxLuma = Math.max(0, Math.min(255, Math.floor(maxLumaField.value)));
  }
  const suppressedBlackField = pickTelemetryFiniteNumber(telemetrySources, [
    'suppressedBlackFrames',
  ]);
  if (suppressedBlackField.found) {
    telemetry.suppressedBlackFrames = toNonNegativeInt(suppressedBlackField.value, 0);
  }
  applySwapchainProbeTelemetryFields(telemetry, telemetrySources);

  // Wave-5 (#663, #1077): forward Wave-4/Wave-5 hang guards, device-loss recovery, and ranged
  // shader-pack streaming counters so app_bootstrap/HUD/panel diagnostics never lose them.
  const rawDeviceStats =
    (eventData.deviceStats && typeof eventData.deviceStats === 'object' && !Array.isArray(eventData.deviceStats)
      ? eventData.deviceStats
      : null) ||
    (rawStats && rawStats.deviceStats && typeof rawStats.deviceStats === 'object' && !Array.isArray(rawStats.deviceStats)
      ? rawStats.deviceStats
      : null);
  const rawShaderPackStats =
    (eventData.shaderPackStats && typeof eventData.shaderPackStats === 'object' && !Array.isArray(eventData.shaderPackStats)
      ? eventData.shaderPackStats
      : null) ||
    (rawStats && rawStats.shaderPackStats && typeof rawStats.shaderPackStats === 'object' && !Array.isArray(rawStats.shaderPackStats)
      ? rawStats.shaderPackStats
      : null);
  const extendedSources = [
    ...telemetrySources,
    ...(rawDeviceStats ? [rawDeviceStats] : []),
    ...(rawShaderPackStats ? [rawShaderPackStats] : []),
  ];

  const coalescedPresentsField = pickTelemetryFiniteNumber(telemetrySources, [
    'coalescedPresents',
    'coalesced_presents',
  ]);
  if (coalescedPresentsField.found) {
    telemetry.coalescedPresents = toNonNegativeInt(coalescedPresentsField.value, 0);
  }
  const oversizedFallbacksField = pickTelemetryFiniteNumber(extendedSources, [
    'oversizedWgslFallbacks',
    'oversized_wgsl_fallbacks',
  ]);
  if (oversizedFallbacksField.found) {
    telemetry.oversizedWgslFallbacks = toNonNegativeInt(oversizedFallbacksField.value, 0);
  }
  const deviceLostField = pickTelemetryFiniteNumber(extendedSources, [
    'deviceLostCount',
    'device_lost_count',
  ]);
  if (deviceLostField.found) {
    telemetry.deviceLostCount = toNonNegativeInt(deviceLostField.value, 0);
  }
  const deviceRecoveredField = pickTelemetryFiniteNumber(extendedSources, [
    'deviceRecoveredCount',
    'device_recovered_count',
  ]);
  if (deviceRecoveredField.found) {
    telemetry.deviceRecoveredCount = toNonNegativeInt(deviceRecoveredField.value, 0);
  }
  const gpuStallWarnField = pickTelemetryFiniteNumber(telemetrySources, [
    'gpuStallWarnCount',
    'gpu_stall_warn_count',
  ]);
  if (gpuStallWarnField.found) {
    telemetry.gpuStallWarnCount = toNonNegativeInt(gpuStallWarnField.value, 0);
  }
  if (rawDeviceStats) {
    const cleanDeviceStats = {
      deviceLostCount: toNonNegativeInt(rawDeviceStats.deviceLostCount, 0),
      deviceRecoveredCount: toNonNegativeInt(rawDeviceStats.deviceRecoveredCount, 0),
      lastDeviceLostReason:
        typeof rawDeviceStats.lastDeviceLostReason === 'string' &&
        rawDeviceStats.lastDeviceLostReason.length > 0
          ? rawDeviceStats.lastDeviceLostReason
          : null,
      deviceRecovering: Boolean(rawDeviceStats.deviceRecovering),
      replayedResourceCount: toNonNegativeInt(rawDeviceStats.replayedResourceCount, 0),
      replayedBufferWriteCount: toNonNegativeInt(rawDeviceStats.replayedBufferWriteCount, 0),
    };
    if (
      rawDeviceStats.replayBindGroupEvictions !== undefined ||
      rawDeviceStats.replayLogOverflowCount !== undefined
    ) {
      cleanDeviceStats.replayBindGroupEvictions = toNonNegativeInt(
        rawDeviceStats.replayBindGroupEvictions,
        0,
      );
      cleanDeviceStats.replayLogOverflowCount = toNonNegativeInt(
        rawDeviceStats.replayLogOverflowCount,
        0,
      );
    }
    telemetry.deviceStats = cleanDeviceStats;
  }
  const rangedPackActiveField = pickTelemetryBoolean(extendedSources, [
    'rangedPackActive',
    'ranged_pack_active',
  ]);
  if (rangedPackActiveField.found) {
    telemetry.rangedPackActive = rangedPackActiveField.value;
  }
  if (rawShaderPackStats) {
    const cleanPackStats = {
      mode:
        typeof rawShaderPackStats.mode === 'string' && rawShaderPackStats.mode.length > 0
          ? rawShaderPackStats.mode
          : 'buffer',
      packCount: toNonNegativeInt(rawShaderPackStats.packCount, 0),
      shaderCount: toNonNegativeInt(rawShaderPackStats.shaderCount, 0),
      rangeFetches: toNonNegativeInt(rawShaderPackStats.rangeFetches, 0),
      cachedWgslBytes: toNonNegativeInt(rawShaderPackStats.cachedWgslBytes, 0),
      rangedPackActive: Boolean(rawShaderPackStats.rangedPackActive),
      rangedPackThresholdBytes: toNonNegativeInt(
        rawShaderPackStats.rangedPackThresholdBytes,
        DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES,
      ),
      asyncShaderPendingCount: toNonNegativeInt(rawShaderPackStats.asyncShaderPendingCount, 0),
      rangedShaderFetches: toNonNegativeInt(rawShaderPackStats.rangedShaderFetches, 0),
      rangedShaderReady: toNonNegativeInt(rawShaderPackStats.rangedShaderReady, 0),
      rangedShaderErrors: toNonNegativeInt(rawShaderPackStats.rangedShaderErrors, 0),
      oversizedWgslFallbacks: toNonNegativeInt(rawShaderPackStats.oversizedWgslFallbacks, 0),
    };
    telemetry.shaderPackStats = cleanPackStats;
    if (telemetry.rangedPackActive === undefined) {
      telemetry.rangedPackActive = cleanPackStats.rangedPackActive;
    }
  }

  const rawGpuAdapter =
    eventData.gpuAdapter !== undefined
      ? eventData.gpuAdapter
      : rawStats && rawStats.gpuAdapter !== undefined
        ? rawStats.gpuAdapter
        : rawFrameStats && rawFrameStats.gpuAdapter !== undefined
          ? rawFrameStats.gpuAdapter
          : undefined;
  if (rawGpuAdapter !== undefined) {
    const sanitizedAdapter = sanitizeGpuAdapterForTelemetry(rawGpuAdapter);
    if (sanitizedAdapter !== undefined) {
      telemetry.gpuAdapter = sanitizedAdapter;
    }
  }

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
  // Wave-5 (#663, #1077): Enable lightweight resource creation-packet replay by default so
  // W100/W104 device-lost recovery restores buffers/textures/layouts/pipelines/bindGroups,
  // while keeping W345's 64 MiB WriteBuffer shadow-copy memcpy disabled by default in live runs.
  const enableResourceReplay =
    msg && typeof msg === 'object' && typeof msg.enableResourceReplay === 'boolean'
      ? msg.enableResourceReplay
      : true;
  const enableShadowBufferReplay =
    msg && typeof msg === 'object' && typeof msg.enableShadowBufferReplay === 'boolean'
      ? msg.enableShadowBufferReplay
      : false;
  const gpuExtra = {};
  if (
    msg &&
    typeof msg === 'object' &&
    typeof msg.devicePixelRatio === 'number' &&
    Number.isFinite(msg.devicePixelRatio) &&
    msg.devicePixelRatio > 0
  ) {
    gpuExtra.devicePixelRatio = msg.devicePixelRatio;
  }
  if (
    msg &&
    typeof msg === 'object' &&
    msg.swapchainOptions &&
    typeof msg.swapchainOptions === 'object' &&
    !Array.isArray(msg.swapchainOptions)
  ) {
    gpuExtra.swapchainOptions = Object.assign({}, msg.swapchainOptions);
  }
  if (msg && typeof msg === 'object' && typeof msg.requireNonBlackFrame === 'boolean') {
    gpuExtra.requireNonBlackFrame = msg.requireNonBlackFrame;
  }
  if (
    msg &&
    typeof msg === 'object' &&
    typeof msg.swapchainProbeTimeoutMs === 'number' &&
    Number.isFinite(msg.swapchainProbeTimeoutMs) &&
    msg.swapchainProbeTimeoutMs >= 0
  ) {
    gpuExtra.swapchainProbeTimeoutMs = Math.floor(msg.swapchainProbeTimeoutMs);
  }
  if (
    msg &&
    typeof msg === 'object' &&
    typeof msg.gpuStallWarnMs === 'number' &&
    Number.isFinite(msg.gpuStallWarnMs) &&
    msg.gpuStallWarnMs >= 0
  ) {
    gpuExtra.gpuStallWarnMs = msg.gpuStallWarnMs;
  }

  // Wave-5 (#663, #1077): Forward explicit streamShaders / streamThresholdBytes and auto-enable
  // ranged shader streaming (64 MiB threshold) whenever shaderPackUrl is present unless explicitly
  // disabled (`streamShaders: false` or `streamThresholdBytes: 0`).
  const hasShaderPackUrl =
    typeof bootFields.shaderPackUrl === 'string' && bootFields.shaderPackUrl.trim().length > 0;
  const explicitStreamDisabled =
    msg && typeof msg === 'object' && (msg.streamShaders === false || msg.streamThresholdBytes === 0);
  const explicitThresholdBytes =
    msg &&
    typeof msg === 'object' &&
    typeof msg.streamThresholdBytes === 'number' &&
    Number.isInteger(msg.streamThresholdBytes) &&
    msg.streamThresholdBytes > 0
      ? msg.streamThresholdBytes
      : null;
  if (explicitStreamDisabled) {
    if (msg.streamShaders === false) {
      gpuExtra.streamShaders = false;
    }
    if (msg.streamThresholdBytes === 0) {
      gpuExtra.streamThresholdBytes = 0;
    }
  } else if (
    (msg && typeof msg === 'object' && msg.streamShaders === true) ||
    explicitThresholdBytes !== null ||
    hasShaderPackUrl
  ) {
    gpuExtra.streamShaders = true;
    gpuExtra.streamThresholdBytes =
      explicitThresholdBytes !== null
        ? explicitThresholdBytes
        : DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES;
  }

  return {
    ioInit: Object.assign({ init: true, dataBase: msg.dataBase }, bootFields),
    gpuInit: Object.assign(
      {
        init: true,
        canvas: msg.canvas,
        width: msg.width,
        height: msg.height,
        enableResourceReplay,
        enableShadowBufferReplay,
      },
      gpuExtra,
      bootFields,
    ),
  };
}

/**
 * Build worker init payloads plus zero-copy transfer lists for any preloaded non-SAB
 * `ArrayBuffer`s (`pgimBuffer`/`pgimBytes` -> `ioWorker`, `shaderPackBuffer`/`shaderPackBytes`
 * -> `gpuWorker`) alongside `msg.canvas`, avoiding structured-clone duplication.
 */
function createWorkerBootInitPayloadsWithTransfers(msg = {}) {
  const base = createWorkerBootInitPayloads(msg);
  const ioTransferList = [];
  const gpuTransferList = [];
  const seenTransfers = new Set();

  const tryQueueTransfer = (buf, list) => {
    if (!isTransferableArrayBuffer(buf) || buf.detached || buf.byteLength <= 0 || seenTransfers.has(buf)) {
      return false;
    }
    seenTransfers.add(buf);
    list.push(buf);
    return true;
  };

  if (msg && typeof msg === 'object') {
    const rawPgim = msg.pgimBuffer !== undefined && msg.pgimBuffer !== null
      ? msg.pgimBuffer
      : msg.pgimBytes !== undefined && msg.pgimBytes !== null
        ? msg.pgimBytes
        : msg.bundle && msg.bundle.pgim
          ? msg.bundle.pgim.buffer || msg.bundle.pgim.bytes || null
          : null;
    if (rawPgim) {
      if (isSharedArrayBuffer(rawPgim) || (ArrayBuffer.isView(rawPgim) && isSharedArrayBuffer(rawPgim.buffer))) {
        base.ioInit.pgimBuffer = ArrayBuffer.isView(rawPgim) ? rawPgim.buffer : rawPgim;
      } else {
        const pgimBuf = toTransferableArrayBuffer(rawPgim);
        if (pgimBuf && pgimBuf.byteLength > 0) {
          base.ioInit.pgimBuffer = pgimBuf;
          tryQueueTransfer(pgimBuf, ioTransferList);
        }
      }
    }

    const rawShaderPack = msg.shaderPackBuffer !== undefined && msg.shaderPackBuffer !== null
      ? msg.shaderPackBuffer
      : msg.shaderPackBytes !== undefined && msg.shaderPackBytes !== null
        ? msg.shaderPackBytes
        : msg.bundle && msg.bundle.shaderPackBytes
          ? msg.bundle.shaderPackBytes.buffer || msg.bundle.shaderPackBytes.bytes || msg.bundle.shaderPackBytes
          : null;
    if (rawShaderPack) {
      if (
        isSharedArrayBuffer(rawShaderPack) ||
        (ArrayBuffer.isView(rawShaderPack) && isSharedArrayBuffer(rawShaderPack.buffer))
      ) {
        base.gpuInit.shaderPackBuffer = ArrayBuffer.isView(rawShaderPack)
          ? rawShaderPack.buffer
          : rawShaderPack;
      } else {
        const shaderBuf = toTransferableArrayBuffer(rawShaderPack);
        if (shaderBuf && shaderBuf.byteLength > 0) {
          base.gpuInit.shaderPackBuffer = shaderBuf;
          tryQueueTransfer(shaderBuf, gpuTransferList);
        }
      }
    }

    if (msg.canvas) {
      gpuTransferList.unshift(msg.canvas);
    }
  }

  return {
    ioInit: base.ioInit,
    gpuInit: base.gpuInit,
    ioTransferList,
    gpuTransferList,
    transferredArrayBufferCount:
      ioTransferList.length + (msg && msg.canvas ? Math.max(0, gpuTransferList.length - 1) : gpuTransferList.length),
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
  let currentHttpfsClient = options.httpfsClient || null;
  const getHttpfsClient =
    typeof options.getHttpfsClient === 'function' ? options.getHttpfsClient : null;
  const store = options.userdataStore || null;
  const cmdBufferPool =
    options.cmdBufferPool && typeof options.cmdBufferPool.acquire === 'function'
      ? options.cmdBufferPool
      : null;
  let firstPresentEmitted = false;

  function activeBuffer() {
    const raw = getMemory ? getMemory() || currentMemory : currentMemory;
    return resolveBuffer(raw);
  }

  function activeHttpfsClient() {
    return getHttpfsClient ? getHttpfsClient() || currentHttpfsClient : currentHttpfsClient;
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
    if (cmdBufferPool) {
      const byteOffset = ptrToNumber(ptr, 'ptr');
      const byteLength = ptrToNumber(len, 'len');
      if (byteOffset + byteLength > buf.byteLength) {
        throw new RangeError(
          `Wasm memory read out of bounds: [${byteOffset}..${byteOffset + byteLength}) > ${buf.byteLength}`,
        );
      }
      if (byteLength >= WGPU_CMD_POOL_MIN_BYTES && byteLength <= WGPU_CMD_POOL_MAX_BYTES) {
        const pooled = cmdBufferPool.acquire(byteLength);
        const targetBuf = pooled || new ArrayBuffer(byteLength);
        const targetView = new Uint8Array(targetBuf, 0, byteLength);
        if (byteLength > 0) {
          targetView.set(new Uint8Array(buf, byteOffset, byteLength));
        }
        if (!firstPresentEmitted) {
          emitFirstPresent(targetView, inspectGpuPacketBatchForPresent(targetView));
        }
        if (wgpuSubmitFn) {
          wgpuSubmitFn(targetView, {
            cmdBuffer: targetBuf,
            byteLength,
            recycleBuffer: true,
          });
        }
        return;
      }
    }
    const bytes = copyMemoryBytes(buf, ptr, len);
    if (!firstPresentEmitted) {
      emitFirstPresent(bytes, inspectGpuPacketBatchForPresent(bytes));
    }
    if (wgpuSubmitFn) {
      wgpuSubmitFn(bytes);
    }
  }

  function host_httpfs_open(ptr, len) {
    const fsClient = activeHttpfsClient();
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
    const fsClient = activeHttpfsClient();
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
    const fsClient = activeHttpfsClient();
    if (!fsClient || typeof fsClient.close !== 'function') {
      return;
    }
    try {
      fsClient.close(Number(handle) | 0);
    } catch {
      // Close errors are non-fatal for void ABI.
    }
  }

  function host_httpfs_stat(ptr, len) {
    const fsClient = activeHttpfsClient();
    if (!fsClient || typeof fsClient.stat !== 'function') {
      return -38n;
    }
    const buf = activeBuffer();
    if (!buf) return -38n;
    try {
      const bytes = copyMemoryBytes(buf, ptr, len);
      const relPath = utf8Decoder.decode(bytes);
      const sizeOrErr = fsClient.stat(relPath);
      if (typeof sizeOrErr === 'bigint') {
        return sizeOrErr;
      }
      return Number.isFinite(sizeOrErr) ? BigInt(Math.trunc(sizeOrErr)) : -2n;
    } catch {
      return -2n;
    }
  }

  const env = {
    host_log,
    host_wgpu_submit,
    host_httpfs_open,
    host_httpfs_read,
    host_httpfs_close,
    host_httpfs_stat,
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
    setHttpfsClient(client) {
      currentHttpfsClient = client || null;
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

function tryOpenBroadcastChannel(BroadcastChannelCtor, channelName) {
  if (typeof BroadcastChannelCtor !== 'function') {
    return null;
  }
  try {
    return new BroadcastChannelCtor(channelName);
  } catch {
    return null;
  }
}

function closeBroadcastChannel(ch) {
  if (!ch || typeof ch.close !== 'function') {
    return;
  }
  try {
    ch.close();
  } catch {
    // Ignore BroadcastChannel close errors.
  }
}

function resolveMessageChannelCtor(explicitCtor, scope) {
  if (explicitCtor !== undefined) {
    return typeof explicitCtor === 'function' ? explicitCtor : null;
  }
  if (scope && typeof scope.MessageChannel === 'function') {
    return scope.MessageChannel;
  }
  if (typeof self !== 'undefined' && self && typeof self.MessageChannel === 'function') {
    return self.MessageChannel;
  }
  if (typeof MessageChannel === 'function') {
    return MessageChannel;
  }
  return null;
}

function closeMessagePort(port) {
  if (!port || typeof port !== 'object') {
    return;
  }
  try {
    port.onmessage = null;
  } catch {
    // Ignore errors clearing MessagePort handler.
  }
  if (typeof port.close === 'function') {
    try {
      port.close();
    } catch {
      // Ignore MessagePort close errors.
    }
  }
}

/**
 * Send one payload on a short-lived BroadcastChannel and close it immediately in `finally`.
 * Keeps send-only threads (especially Emscripten pthreads running Wasm or `Atomics.wait`) from
 * holding open BroadcastChannel receive queues that never drain.
 */
function postBroadcastOnce(BroadcastChannelCtor, channelName, payload) {
  const ch = tryOpenBroadcastChannel(BroadcastChannelCtor, channelName);
  if (!ch) {
    return false;
  }
  try {
    if (typeof ch.postMessage !== 'function') {
      return false;
    }
    ch.postMessage(payload);
    return true;
  } catch {
    return false;
  } finally {
    closeBroadcastChannel(ch);
  }
}

function isEmscriptenPthreadWorkerName(name) {
  if (typeof name !== 'string') {
    return false;
  }
  return (
    name === EMSCRIPTEN_PTHREAD_WORKER_NAME ||
    name.startsWith(`${EMSCRIPTEN_PTHREAD_WORKER_NAME}-`)
  );
}

function isEmscriptenPthreadWorker(scope) {
  return Boolean(scope && isEmscriptenPthreadWorkerName(scope.name));
}

function inspectRequiredHostAbiImports(wasmModule, wasmNs) {
  const ModuleCtor =
    (wasmNs && typeof wasmNs.Module === 'function' ? wasmNs.Module : null) ||
    (typeof WebAssembly !== 'undefined' && typeof WebAssembly.Module === 'function'
      ? WebAssembly.Module
      : null);
  if (!ModuleCtor || typeof ModuleCtor.imports !== 'function') {
    return null;
  }
  if (!(wasmModule instanceof ModuleCtor)) {
    return null;
  }
  try {
    const entries = ModuleCtor.imports(wasmModule);
    if (!Array.isArray(entries)) {
      return null;
    }
    const needed = new Set();
    for (let i = 0; i < entries.length; i += 1) {
      const imp = entries[i];
      if (imp && imp.module === 'env' && ALL_HOST_ABI_IMPORT_SET.has(imp.name)) {
        needed.add(imp.name);
      }
    }
    return needed;
  } catch {
    return null;
  }
}

function createPatchedPthreadEnv(baseEnv, hostEnv, requiredHostNames) {
  if (!baseEnv || typeof baseEnv !== 'object' || !hostEnv) {
    return baseEnv;
  }
  const targetNames = requiredHostNames
    ? Array.from(requiredHostNames)
    : ALL_HOST_ABI_IMPORT_NAMES;
  if (targetNames.length === 0) {
    return baseEnv;
  }

  // If baseEnv is frozen or has a non-configurable non-writable host_* stub (or write-only
  // non-configurable accessor), a Proxy get-trap cannot return a replacement without throwing.
  let needsClone = false;
  try {
    needsClone = Object.isFrozen(baseEnv);
  } catch {
    needsClone = false;
  }
  if (!needsClone) {
    for (let i = 0; i < targetNames.length; i += 1) {
      let desc = null;
      try {
        desc = Reflect.getOwnPropertyDescriptor(baseEnv, targetNames[i]);
      } catch {
        // If baseEnv's getOwnPropertyDescriptor trap throws, a Proxy get-trap invariant check
        // will also fail when substituting a host ABI function; clone instead.
        needsClone = true;
        break;
      }
      if (
        desc &&
        !desc.configurable &&
        (('value' in desc && !desc.writable) || (!('value' in desc) && desc.get === undefined))
      ) {
        needsClone = true;
        break;
      }
    }
  }

  if (needsClone) {
    const cloned = {};
    try {
      const ownKeys = Reflect.ownKeys(baseEnv);
      for (let i = 0; i < ownKeys.length; i += 1) {
        const key = ownKeys[i];
        try {
          const desc = Reflect.getOwnPropertyDescriptor(baseEnv, key);
          if (desc) {
            if ('value' in desc) {
              Object.defineProperty(cloned, key, {
                value: desc.value,
                writable: true,
                configurable: true,
                enumerable: desc.enumerable !== false,
              });
            } else {
              Object.defineProperty(cloned, key, {
                get: desc.get,
                set: desc.set,
                configurable: true,
                enumerable: desc.enumerable !== false,
              });
            }
          }
        } catch {
          try {
            cloned[key] = baseEnv[key];
          } catch {
            // Ignore exotic keys that fail both descriptor inspection and property read.
          }
        }
      }
    } catch {
      try {
        Object.assign(cloned, baseEnv);
      } catch {
        // Ignore if baseEnv cannot be enumerated via Object.assign.
      }
    }
    for (let i = 0; i < targetNames.length; i += 1) {
      const name = targetNames[i];
      if (typeof hostEnv[name] !== 'function') {
        continue;
      }
      let existing;
      try {
        existing = cloned[name];
      } catch {
        existing = undefined;
      }
      if (typeof existing !== 'function' || existing.stub === true) {
        Object.defineProperty(cloned, name, {
          value: hostEnv[name],
          writable: true,
          configurable: true,
          enumerable: true,
        });
      }
    }
    return cloned;
  }

  // Emscripten MAIN_MODULE=1 generates JS wrappers (`function _host_log(...args) {
  //   if (!wasmImports["host_log"] || wasmImports["host_log"].stub) abort(...);
  //   return wasmImports["host_log"](...args);
  // }`) that read the unwrapped `wasmImports` object directly rather than our Proxy.
  // Overwrite writable (or configurable non-writable) own `.stub === true` properties in-place
  // so those wrappers resolve.
  for (let i = 0; i < targetNames.length; i += 1) {
    const name = targetNames[i];
    if (typeof hostEnv[name] !== 'function') {
      continue;
    }
    try {
      const desc = Reflect.getOwnPropertyDescriptor(baseEnv, name);
      if (desc && 'value' in desc && typeof desc.value === 'function' && desc.value.stub === true) {
        if (desc.writable) {
          baseEnv[name] = hostEnv[name];
        } else if (desc.configurable) {
          Object.defineProperty(baseEnv, name, {
            value: hostEnv[name],
            writable: true,
            configurable: true,
            enumerable: desc.enumerable !== false,
          });
        }
      }
    } catch {
      // Ignore non-writable or Proxy-protected targets; the returned Proxy still intercepts reads.
    }
  }

  // Wrap baseEnv in a transparent Proxy so Emscripten MAIN_MODULE=2 / SIDE_MODULE=1
  // dynamic-linking Proxies, __memory_base, __table_base, __indirect_function_table,
  // and all non-stub lifted/C functions remain completely untouched.
  return new Proxy(baseEnv, {
    get(target, prop, receiver) {
      const isTargetHost =
        typeof prop === 'string' &&
        ALL_HOST_ABI_IMPORT_SET.has(prop) &&
        (!requiredHostNames || requiredHostNames.has(prop));
      let existing;
      try {
        existing = Reflect.get(target, prop, receiver);
      } catch (err) {
        if (isTargetHost && typeof hostEnv[prop] === 'function') {
          return hostEnv[prop];
        }
        throw err;
      }
      if (isTargetHost && typeof hostEnv[prop] === 'function') {
        if (typeof existing !== 'function' || existing.stub === true) {
          return hostEnv[prop];
        }
      }
      return existing;
    },
  });
}

function patchPthreadWasmImports(wasmModuleOrBytes, rawImports, wasmNs, hostImports, hostEnv) {
  if (!rawImports || typeof rawImports !== 'object') {
    return rawImports;
  }

  const importedMemory = extractMemoryFromImports(rawImports);
  if (importedMemory && hostImports && typeof hostImports.setMemory === 'function') {
    hostImports.setMemory(importedMemory);
  }

  if (!hostEnv) {
    return rawImports;
  }

  const requiredHostNames = inspectRequiredHostAbiImports(wasmModuleOrBytes, wasmNs);
  if (requiredHostNames && requiredHostNames.size === 0) {
    // Side module imports none of the 5 host_* ABI symbols; leave imports & imports.env untouched.
    return rawImports;
  }

  const hasEnvObj = Boolean(rawImports.env && typeof rawImports.env === 'object');
  if (!hasEnvObj) {
    if (!requiredHostNames || requiredHostNames.size === 0) {
      return rawImports;
    }
    const synthesizedEnv = {};
    requiredHostNames.forEach((name) => {
      if (typeof hostEnv[name] === 'function') {
        synthesizedEnv[name] = hostEnv[name];
      }
    });
    const patchedImports = Object.assign({}, rawImports);
    patchedImports.env = synthesizedEnv;
    return patchedImports;
  }

  const patchedEnv = createPatchedPthreadEnv(rawImports.env, hostEnv, requiredHostNames);
  if (patchedEnv === rawImports.env) {
    return rawImports;
  }

  // Shallow-copy top-level namespace map so GOT.mem, GOT.func, wasi_snapshot_preview1, etc.
  // keep their exact object/Proxy references untouched.
  const patchedImports = Object.assign({}, rawImports);
  patchedImports.env = patchedEnv;
  return patchedImports;
}

function encodeWasmUleb128(value) {
  let n = value >>> 0;
  const out = [];
  do {
    let byte = n & 0x7f;
    n >>>= 7;
    if (n !== 0) {
      byte |= 0x80;
    }
    out.push(byte);
  } while (n !== 0);
  return out;
}

function encodeWasmCustomSection(name, payloadBytes) {
  const nameBytes = utf8Encoder.encode(name);
  const nameLenLeb = encodeWasmUleb128(nameBytes.byteLength);
  const payload =
    payloadBytes instanceof Uint8Array
      ? payloadBytes
      : payloadBytes instanceof ArrayBuffer
        ? new Uint8Array(payloadBytes)
        : new Uint8Array(0);
  const bodyLen = nameLenLeb.length + nameBytes.byteLength + payload.byteLength;
  const bodyLenLeb = encodeWasmUleb128(bodyLen);
  const section = new Uint8Array(1 + bodyLenLeb.length + bodyLen);
  let offset = 0;
  section[offset] = 0x00;
  offset += 1;
  section.set(bodyLenLeb, offset);
  offset += bodyLenLeb.length;
  section.set(nameLenLeb, offset);
  offset += nameLenLeb.length;
  section.set(nameBytes, offset);
  offset += nameBytes.byteLength;
  if (payload.byteLength > 0) {
    section.set(payload, offset);
  }
  return section;
}

/**
 * Build a minimal (~100-byte) valid WebAssembly binary carrying the compiled module's exact
 * `dylink.0` / `dylink` custom section(s) plus a tiny `pg6.shared.ref` marker.
 *
 * Emscripten's `loadDynamicLibrary` fetches side modules via `fetch(url).then(r => r.arrayBuffer())`,
 * parses `dylink.0`, and then calls `WebAssembly.compile` / `new WebAssembly.Module` /
 * `WebAssembly.instantiate`. Returning this ~100-byte stub instead of re-fetching a ~908 MB
 * `guest_part_0.wasm` prevents `RangeError: Array buffer allocation failed` when pthreads spawn
 * late in a long session (#663, #1077).
 */
function buildSharedWasmModuleDylinkStub(wasmModule, moduleKey, wasmNs) {
  const sections = [];
  const ModuleCtor =
    (wasmNs && typeof wasmNs.Module === 'function' ? wasmNs.Module : null) ||
    (typeof WebAssembly !== 'undefined' && typeof WebAssembly.Module === 'function'
      ? WebAssembly.Module
      : null);
  if (ModuleCtor && typeof ModuleCtor.customSections === 'function' && wasmModule) {
    const dylinkNames = ['dylink.0', 'dylink'];
    for (let i = 0; i < dylinkNames.length; i += 1) {
      const secName = dylinkNames[i];
      try {
        const found = ModuleCtor.customSections(wasmModule, secName);
        if (Array.isArray(found)) {
          for (let j = 0; j < found.length; j += 1) {
            if (found[j] && typeof found[j].byteLength === 'number') {
              sections.push(encodeWasmCustomSection(secName, new Uint8Array(found[j])));
            }
          }
        }
      } catch {
        // Mock WebAssembly.Module instances may not support customSections(); continue with stub.
      }
    }
  }
  const cleanKey = typeof moduleKey === 'string' ? moduleKey.trim().toLowerCase() : '';
  if (cleanKey) {
    sections.push(
      encodeWasmCustomSection(SHARED_WASM_STUB_CUSTOM_SECTION_NAME, utf8Encoder.encode(cleanKey)),
    );
  }
  let totalBytes = 8;
  for (let i = 0; i < sections.length; i += 1) {
    totalBytes += sections[i].byteLength;
  }
  const stub = new Uint8Array(totalBytes);
  // WebAssembly magic (`\0asm`) + version 1
  stub[0] = 0x00;
  stub[1] = 0x61;
  stub[2] = 0x73;
  stub[3] = 0x6d;
  stub[4] = 0x01;
  stub[5] = 0x00;
  stub[6] = 0x00;
  stub[7] = 0x00;
  let cursor = 8;
  for (let i = 0; i < sections.length; i += 1) {
    stub.set(sections[i], cursor);
    cursor += sections[i].byteLength;
  }
  return stub;
}

function extractSharedWasmStubKey(bytesOrBuf) {
  let view = null;
  if (bytesOrBuf instanceof Uint8Array) {
    view = bytesOrBuf;
  } else if (isTransferableArrayBuffer(bytesOrBuf) || isSharedArrayBuffer(bytesOrBuf)) {
    view = new Uint8Array(bytesOrBuf);
  } else if (ArrayBuffer.isView(bytesOrBuf)) {
    view = new Uint8Array(bytesOrBuf.buffer, bytesOrBuf.byteOffset, bytesOrBuf.byteLength);
  }
  if (!view || view.byteLength < 10 || view.byteLength > SHARED_WASM_STUB_MAX_BYTES) {
    return null;
  }
  if (
    view[0] !== 0x00 ||
    view[1] !== 0x61 ||
    view[2] !== 0x73 ||
    view[3] !== 0x6d ||
    view[4] !== 0x01 ||
    view[5] !== 0x00 ||
    view[6] !== 0x00 ||
    view[7] !== 0x00
  ) {
    return null;
  }
  const len = view.byteLength;
  let offset = 8;
  const readUleb128 = () => {
    let result = 0;
    let shift = 0;
    while (offset < len && shift <= 28) {
      const b = view[offset];
      offset += 1;
      result |= (b & 0x7f) << shift;
      if ((b & 0x80) === 0) {
        return result >>> 0;
      }
      shift += 7;
    }
    return -1;
  };
  while (offset < len) {
    const secId = view[offset];
    offset += 1;
    const secSize = readUleb128();
    if (secSize < 0 || offset + secSize > len) {
      return null;
    }
    const secEnd = offset + secSize;
    if (secId === 0x00) {
      const nameLen = readUleb128();
      if (nameLen >= 0 && offset + nameLen <= secEnd) {
        const name = utf8Decoder.decode(view.subarray(offset, offset + nameLen));
        offset += nameLen;
        if (name === SHARED_WASM_STUB_CUSTOM_SECTION_NAME) {
          return utf8Decoder.decode(view.subarray(offset, secEnd)) || null;
        }
      }
    }
    offset = secEnd;
  }
  return null;
}

function extractFetchRequestUrl(input) {
  if (typeof input === 'string') {
    return input;
  }
  if (input && typeof input === 'object') {
    if (typeof input.url === 'string') {
      return input.url;
    }
    if (typeof input.href === 'string') {
      return input.href;
    }
  }
  return '';
}

function extractFetchRequestMethod(input, init) {
  if (init && typeof init === 'object' && typeof init.method === 'string') {
    return init.method.trim().toUpperCase();
  }
  if (input && typeof input === 'object' && typeof input.method === 'string') {
    return input.method.trim().toUpperCase();
  }
  return 'GET';
}

function createSharedWasmStubResponse(url, stubBytes, cachedMod, stubBufferToModule, ResponseCtor) {
  const makeTaggedCopy = () => {
    const copy = stubBytes.slice();
    stubBufferToModule.set(copy.buffer, cachedMod);
    return copy;
  };
  const contentLength = String(stubBytes.byteLength);
  if (typeof ResponseCtor === 'function') {
    try {
      const resp = new ResponseCtor(stubBytes, {
        status: 200,
        headers: {
          'content-type': 'application/wasm',
          'content-length': contentLength,
        },
      });
      try {
        Object.defineProperty(resp, 'url', { value: url, configurable: true });
      } catch {
        // Ignore if Response.url is non-configurable in a custom harness.
      }
      resp.arrayBuffer = () => Promise.resolve(makeTaggedCopy().buffer);
      resp.bytes = () => Promise.resolve(makeTaggedCopy());
      return resp;
    } catch {
      // Fall back to lightweight Response-like stub below.
    }
  }
  const headerMap = new Map([
    ['content-type', 'application/wasm'],
    ['content-length', contentLength],
  ]);
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    url,
    bodyUsed: false,
    headers: {
      get(name) {
        return headerMap.get(String(name || '').toLowerCase()) || null;
      },
      has(name) {
        return headerMap.has(String(name || '').toLowerCase());
      },
    },
    arrayBuffer() {
      this.bodyUsed = true;
      return Promise.resolve(makeTaggedCopy().buffer);
    },
    bytes() {
      this.bodyUsed = true;
      return Promise.resolve(makeTaggedCopy());
    },
    body: {
      cancel() {
        return Promise.resolve();
      },
    },
  };
}

/**
 * Install a lightweight `fetch` + `WebAssembly.{compile,Module,instantiate,compileStreaming,
 * instantiateStreaming}` shim on `globalScope` so precompiled side modules (`guest_part_*.wasm`)
 * never re-fetch or allocate 908 MB `ArrayBuffer`s in the coordinator or dynamically spawned
 * Emscripten pthread workers (#663, #1077).
 */
function installSharedWasmModuleShim(globalScope, options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  const resolveSharedModule =
    typeof safeOpts.resolveSharedModule === 'function' ? safeOpts.resolveSharedModule : null;
  const onModuleCompiled =
    typeof safeOpts.onModuleCompiled === 'function' ? safeOpts.onModuleCompiled : null;
  if (!globalScope || (!resolveSharedModule && !onModuleCompiled)) {
    return function noopRestore() {};
  }

  const fetchScope = safeOpts.fetchScope || globalScope;
  const wasmNs =
    safeOpts.wasmNs ||
    (globalScope.WebAssembly && typeof globalScope.WebAssembly === 'object'
      ? globalScope.WebAssembly
      : null);
  const ResponseCtor =
    (fetchScope && typeof fetchScope.Response === 'function' ? fetchScope.Response : null) ||
    (typeof Response === 'function' ? Response : null);
  const OriginalFetch =
    fetchScope && typeof fetchScope.fetch === 'function' ? fetchScope.fetch : null;
  const OriginalModule =
    wasmNs && typeof wasmNs.Module === 'function' ? wasmNs.Module : null;
  const OriginalCompile =
    wasmNs && typeof wasmNs.compile === 'function' ? wasmNs.compile : null;
  const OriginalInstantiate =
    wasmNs && typeof wasmNs.instantiate === 'function' ? wasmNs.instantiate : null;
  const OriginalCompileStreaming =
    wasmNs && typeof wasmNs.compileStreaming === 'function' ? wasmNs.compileStreaming : null;
  const OriginalInstantiateStreaming =
    wasmNs && typeof wasmNs.instantiateStreaming === 'function'
      ? wasmNs.instantiateStreaming
      : null;
  const OriginalInstance =
    wasmNs && typeof wasmNs.Instance === 'function' ? wasmNs.Instance : null;

  const stubBufferToModule = new WeakMap();
  const stubBytesByModule = new WeakMap();

  const cancelResponseBodyQuietly = (resp) => {
    try {
      if (resp && resp.body && typeof resp.body.cancel === 'function' && !resp.bodyUsed) {
        const p = resp.body.cancel();
        if (p && typeof p.catch === 'function') {
          p.catch(() => {});
        }
      }
    } catch {
      // Ignore body cancel errors on mock or already-consumed Response objects.
    }
  };

  const getOrBuildStubBytes = (cachedMod, rawUrl) => {
    let stub = stubBytesByModule.get(cachedMod);
    if (!stub) {
      const key = extractWasmBasename(rawUrl) || rawUrl;
      stub = buildSharedWasmModuleDylinkStub(cachedMod, key, { Module: OriginalModule });
      stubBytesByModule.set(cachedMod, stub);
    }
    return stub;
  };

  const resolveCachedModuleFromBytes = (source) => {
    if (!source || typeof source !== 'object') {
      return null;
    }
    if (isCompiledWasmModule(source, { Module: OriginalModule })) {
      return null;
    }
    const backing = ArrayBuffer.isView(source) ? source.buffer : source;
    if (backing && typeof backing === 'object') {
      const direct = stubBufferToModule.get(backing);
      if (direct && isCompiledWasmModule(direct, { Module: OriginalModule })) {
        return direct;
      }
    }
    if (resolveSharedModule) {
      const stubKey = extractSharedWasmStubKey(source);
      if (stubKey) {
        const matched = resolveSharedModule(stubKey);
        if (isCompiledWasmModule(matched, { Module: OriginalModule })) {
          return matched;
        }
      }
    }
    return null;
  };

  let WrappedFetch = null;
  if (OriginalFetch && resolveSharedModule) {
    WrappedFetch = function wrappedSharedWasmFetch(input, init) {
      const method = extractFetchRequestMethod(input, init);
      if (method === 'GET') {
        const rawUrl = extractFetchRequestUrl(input);
        if (rawUrl) {
          const cachedMod = resolveSharedModule(rawUrl);
          if (isCompiledWasmModule(cachedMod, { Module: OriginalModule })) {
            const stubBytes = getOrBuildStubBytes(cachedMod, rawUrl);
            return Promise.resolve(
              createSharedWasmStubResponse(
                rawUrl,
                stubBytes,
                cachedMod,
                stubBufferToModule,
                ResponseCtor,
              ),
            );
          }
        }
      }
      return OriginalFetch.apply(this || fetchScope, arguments);
    };
    try {
      fetchScope.fetch = WrappedFetch;
    } catch {
      WrappedFetch = null;
    }
  }

  let WrappedModule = null;
  if (OriginalModule && resolveSharedModule) {
    WrappedModule = new Proxy(OriginalModule, {
      construct(target, args, newTarget) {
        if (args.length >= 1) {
          const cachedMod = resolveCachedModuleFromBytes(args[0]);
          if (cachedMod) {
            return cachedMod;
          }
        }
        const targetNew = newTarget === WrappedModule ? target : newTarget;
        return Reflect.construct(target, args, targetNew);
      },
    });
    try {
      wasmNs.Module = WrappedModule;
    } catch {
      WrappedModule = null;
    }
  }

  let WrappedCompile = null;
  if (OriginalCompile && resolveSharedModule) {
    WrappedCompile = function wrappedSharedWasmCompile(bytes) {
      const cachedMod = resolveCachedModuleFromBytes(bytes);
      if (cachedMod) {
        return Promise.resolve(cachedMod);
      }
      return OriginalCompile.apply(wasmNs, arguments);
    };
    try {
      wasmNs.compile = WrappedCompile;
    } catch {
      WrappedCompile = null;
    }
  }

  let WrappedInstantiate = null;
  if (OriginalInstantiate && resolveSharedModule && safeOpts.wrapInstantiate !== false) {
    WrappedInstantiate = function wrappedSharedWasmInstantiate(moduleOrBytes, imports) {
      const cachedMod = resolveCachedModuleFromBytes(moduleOrBytes);
      if (cachedMod) {
        const instPromise =
          arguments.length >= 2
            ? OriginalInstantiate.call(wasmNs, cachedMod, imports)
            : OriginalInstantiate.call(wasmNs, cachedMod);
        return Promise.resolve(instPromise).then((instance) => ({ instance, module: cachedMod }));
      }
      return OriginalInstantiate.apply(wasmNs, arguments);
    };
    try {
      wasmNs.instantiate = WrappedInstantiate;
    } catch {
      WrappedInstantiate = null;
    }
  }

  let WrappedCompileStreaming = null;
  if (OriginalCompileStreaming && (resolveSharedModule || onModuleCompiled)) {
    WrappedCompileStreaming = function wrappedSharedWasmCompileStreaming(source) {
      return Promise.resolve(source).then((resp) => {
        const respUrl = resp && typeof resp.url === 'string' ? resp.url : '';
        if (respUrl && resolveSharedModule) {
          const cachedMod = resolveSharedModule(respUrl);
          if (isCompiledWasmModule(cachedMod, { Module: OriginalModule })) {
            cancelResponseBodyQuietly(resp);
            return cachedMod;
          }
        }
        return OriginalCompileStreaming.call(wasmNs, Promise.resolve(resp)).then((compiledMod) => {
          if (respUrl && onModuleCompiled && isCompiledWasmModule(compiledMod, { Module: OriginalModule })) {
            try {
              onModuleCompiled(respUrl, compiledMod);
            } catch {
              // Capture hook failures must never break compilation.
            }
          }
          return compiledMod;
        });
      });
    };
    try {
      wasmNs.compileStreaming = WrappedCompileStreaming;
    } catch {
      WrappedCompileStreaming = null;
    }
  }

  let WrappedInstantiateStreaming = null;
  if (
    OriginalInstantiateStreaming &&
    (resolveSharedModule || onModuleCompiled) &&
    safeOpts.wrapInstantiateStreaming !== false
  ) {
    WrappedInstantiateStreaming = function wrappedSharedWasmInstantiateStreaming(source, imports) {
      return Promise.resolve(source).then((resp) => {
        const respUrl = resp && typeof resp.url === 'string' ? resp.url : '';
        if (respUrl && resolveSharedModule) {
          const cachedMod = resolveSharedModule(respUrl);
          if (isCompiledWasmModule(cachedMod, { Module: OriginalModule })) {
            cancelResponseBodyQuietly(resp);
            if (OriginalInstantiate) {
              return OriginalInstantiate.call(wasmNs, cachedMod, imports).then((instance) => ({
                instance,
                module: cachedMod,
              }));
            }
            if (OriginalInstance) {
              return { instance: new OriginalInstance(cachedMod, imports), module: cachedMod };
            }
          }
        }
        return OriginalInstantiateStreaming.call(wasmNs, Promise.resolve(resp), imports).then(
          (instantiated) => {
            const compiledMod = instantiated && instantiated.module ? instantiated.module : null;
            if (
              respUrl &&
              onModuleCompiled &&
              isCompiledWasmModule(compiledMod, { Module: OriginalModule })
            ) {
              try {
                onModuleCompiled(respUrl, compiledMod);
              } catch {
                // Capture hook failures must never break instantiation.
              }
            }
            return instantiated;
          },
        );
      });
    };
    try {
      wasmNs.instantiateStreaming = WrappedInstantiateStreaming;
    } catch {
      WrappedInstantiateStreaming = null;
    }
  }

  let restored = false;
  const restore = function restoreSharedWasmModuleShim() {
    if (restored) {
      return;
    }
    restored = true;
    if (WrappedFetch && fetchScope && fetchScope.fetch === WrappedFetch) {
      try {
        fetchScope.fetch = OriginalFetch;
      } catch {
        // Ignore restore failures on frozen scopes.
      }
    }
    if (wasmNs) {
      if (WrappedModule && wasmNs.Module === WrappedModule) {
        wasmNs.Module = OriginalModule;
      }
      if (WrappedCompile && wasmNs.compile === WrappedCompile) {
        wasmNs.compile = OriginalCompile;
      }
      if (WrappedInstantiate && wasmNs.instantiate === WrappedInstantiate) {
        wasmNs.instantiate = OriginalInstantiate;
      }
      if (WrappedCompileStreaming && wasmNs.compileStreaming === WrappedCompileStreaming) {
        wasmNs.compileStreaming = OriginalCompileStreaming;
      }
      if (
        WrappedInstantiateStreaming &&
        wasmNs.instantiateStreaming === WrappedInstantiateStreaming
      ) {
        wasmNs.instantiateStreaming = OriginalInstantiateStreaming;
      }
    }
  };
  restore.resolveCachedModuleFromBytes = resolveCachedModuleFromBytes;
  return restore;
}

/**
 * Wrap `globalScope.WebAssembly.Instance` (and `WebAssembly.instantiate` when present) so that
 * inside an Emscripten `em-pthread` worker, both the main module and `SIDE_MODULE=1` side modules
 * bind `imports.env.memory` (when present) onto `hostImports` and replace missing or `.stub === true`
 * entries for the 5 host ABI imports (`host_log`, `host_httpfs_*`, `host_wgpu_submit`) while
 * preserving `GOT.mem`, `GOT.func`, `env.__memory_base`, `env.__table_base`,
 * `env.__indirect_function_table`, dynamic-linking Proxies, non-stub functions, and
 * `WebAssembly.Instance.prototype` / `instanceof`.
 * Also intercepts `fetch`, `WebAssembly.compile`, `WebAssembly.Module`, `WebAssembly.instantiate`,
 * `WebAssembly.compileStreaming`, and `WebAssembly.instantiateStreaming` for precompiled shared
 * side modules so dynamically spawned pthreads never re-fetch or allocate 908 MB `ArrayBuffer`s.
 * Returns an idempotent restore function.
 */
function installPthreadHostImportShim(globalScope, hostImports, shimOptions = {}) {
  if (
    !globalScope ||
    !globalScope.WebAssembly ||
    typeof globalScope.WebAssembly.Instance !== 'function'
  ) {
    return function noopRestore() {};
  }

  const wasmNs = globalScope.WebAssembly;
  const OriginalInstance = wasmNs.Instance;
  const OriginalInstantiate =
    typeof wasmNs.instantiate === 'function' ? wasmNs.instantiate : null;
  const OriginalInstantiateStreaming =
    typeof wasmNs.instantiateStreaming === 'function' ? wasmNs.instantiateStreaming : null;
  const resolveSharedModule =
    shimOptions && typeof shimOptions.resolveSharedModule === 'function'
      ? shimOptions.resolveSharedModule
      : null;
  const fetchScope =
    shimOptions && shimOptions.fetchScope ? shimOptions.fetchScope : globalScope;
  const hostEnv =
    hostImports && typeof hostImports === 'object'
      ? hostImports.env && typeof hostImports.env === 'object'
        ? hostImports.env
        : hostImports
      : null;

  // Install shared side-module fetch + compile + Module + compileStreaming interceptor first.
  const restoreSharedShim = resolveSharedModule
    ? installSharedWasmModuleShim(globalScope, {
        fetchScope,
        wasmNs,
        resolveSharedModule,
        wrapInstantiate: false,
        wrapInstantiateStreaming: false,
      })
    : null;
  const resolveCachedModuleFromBytes =
    restoreSharedShim && typeof restoreSharedShim.resolveCachedModuleFromBytes === 'function'
      ? restoreSharedShim.resolveCachedModuleFromBytes
      : () => null;

  const cancelResponseBodyQuietly = (resp) => {
    try {
      if (resp && resp.body && typeof resp.body.cancel === 'function' && !resp.bodyUsed) {
        const p = resp.body.cancel();
        if (p && typeof p.catch === 'function') {
          p.catch(() => {});
        }
      }
    } catch {
      // Ignore body cancel errors on mock or already-consumed Response objects.
    }
  };

  const bindExportedMemoryIfNeeded = (rawImports, instance) => {
    if (
      (!rawImports || !extractMemoryFromImports(rawImports)) &&
      instance &&
      instance.exports &&
      instance.exports.memory &&
      hostImports &&
      typeof hostImports.setMemory === 'function'
    ) {
      hostImports.setMemory(instance.exports.memory);
    }
  };

  const WrappedInstance = new Proxy(OriginalInstance, {
    construct(target, args, newTarget) {
      const nextArgs = Array.prototype.slice.call(args);
      const rawImports = nextArgs.length >= 2 ? nextArgs[1] : undefined;

      if (rawImports && typeof rawImports === 'object') {
        nextArgs[1] = patchPthreadWasmImports(
          nextArgs[0],
          rawImports,
          wasmNs,
          hostImports,
          hostEnv,
        );
      }

      const targetNew = newTarget === WrappedInstance ? target : newTarget;
      const instance = Reflect.construct(target, nextArgs, targetNew);
      bindExportedMemoryIfNeeded(rawImports, instance);
      return instance;
    },
  });

  wasmNs.Instance = WrappedInstance;

  let WrappedInstantiate = null;
  if (OriginalInstantiate) {
    WrappedInstantiate = function wrappedWasmInstantiate(moduleOrBytes, imports) {
      const cachedMod = resolveCachedModuleFromBytes(moduleOrBytes);
      if (cachedMod) {
        const patchedImports =
          imports && typeof imports === 'object'
            ? patchPthreadWasmImports(cachedMod, imports, wasmNs, hostImports, hostEnv)
            : imports;
        const res =
          arguments.length >= 2
            ? OriginalInstantiate.call(wasmNs, cachedMod, patchedImports)
            : OriginalInstantiate.call(wasmNs, cachedMod);
        return Promise.resolve(res).then((instance) => {
          bindExportedMemoryIfNeeded(imports, instance);
          return { instance, module: cachedMod };
        });
      }
      const patchedImports =
        imports && typeof imports === 'object'
          ? patchPthreadWasmImports(moduleOrBytes, imports, wasmNs, hostImports, hostEnv)
          : imports;
      const res =
        arguments.length >= 2
          ? OriginalInstantiate.call(wasmNs, moduleOrBytes, patchedImports)
          : OriginalInstantiate.call(wasmNs, moduleOrBytes);
      if (res && typeof res.then === 'function') {
        return res.then((instantiated) => {
          const instance =
            instantiated && instantiated.instance ? instantiated.instance : instantiated;
          bindExportedMemoryIfNeeded(imports, instance);
          return instantiated;
        });
      }
      return res;
    };
    wasmNs.instantiate = WrappedInstantiate;
  }

  let WrappedInstantiateStreaming = null;
  if (OriginalInstantiateStreaming && resolveSharedModule) {
    WrappedInstantiateStreaming = function wrappedWasmInstantiateStreaming(source, imports) {
      return Promise.resolve(source).then((resp) => {
        const respUrl = resp && typeof resp.url === 'string' ? resp.url : '';
        if (respUrl) {
          const cachedMod = resolveSharedModule(respUrl);
          if (isCompiledWasmModule(cachedMod, wasmNs)) {
            cancelResponseBodyQuietly(resp);
            const patchedImports =
              imports && typeof imports === 'object'
                ? patchPthreadWasmImports(cachedMod, imports, wasmNs, hostImports, hostEnv)
                : imports;
            if (OriginalInstantiate) {
              return OriginalInstantiate.call(wasmNs, cachedMod, patchedImports).then((instance) => {
                bindExportedMemoryIfNeeded(imports, instance);
                return { instance, module: cachedMod };
              });
            }
            const instance = new OriginalInstance(cachedMod, patchedImports);
            bindExportedMemoryIfNeeded(imports, instance);
            return { instance, module: cachedMod };
          }
        }
        const patchedImports =
          imports && typeof imports === 'object'
            ? patchPthreadWasmImports(null, imports, wasmNs, hostImports, hostEnv)
            : imports;
        return OriginalInstantiateStreaming.call(wasmNs, Promise.resolve(resp), patchedImports).then(
          (instantiated) => {
            const instance =
              instantiated && instantiated.instance ? instantiated.instance : instantiated;
            bindExportedMemoryIfNeeded(imports, instance);
            return instantiated;
          },
        );
      });
    };
    wasmNs.instantiateStreaming = WrappedInstantiateStreaming;
  }

  let restored = false;
  return function restorePthreadHostImportShim() {
    if (restored) {
      return;
    }
    restored = true;
    if (wasmNs.Instance === WrappedInstance) {
      wasmNs.Instance = OriginalInstance;
    }
    if (OriginalInstantiate && wasmNs.instantiate === WrappedInstantiate) {
      wasmNs.instantiate = OriginalInstantiate;
    }
    if (OriginalInstantiateStreaming && wasmNs.instantiateStreaming === WrappedInstantiateStreaming) {
      wasmNs.instantiateStreaming = OriginalInstantiateStreaming;
    }
    if (restoreSharedShim) {
      restoreSharedShim();
    }
  };
}

/**
 * Bootstrap an Emscripten `em-pthread` worker spawned via `new Worker(self.location.href, { name: "em-pthread" })`:
 * - Sends `host_log`, side-module progress, and `host_wgpu_submit` batches over a dedicated
 *   send-only `MessagePort` transferred from the coordinator at spawn (`__playgta6_tx_port`),
 *   eliminating per-message `BroadcastChannel` construction/close overhead and cross-pthread fan-out.
 * - Falls back gracefully to short-lived send-only `BroadcastChannel` instances when no `MessagePort`
 *   is available or if the port is detached/closed.
 * - Acquires `httpfsMailboxSab` via `Worker.postMessage` or temporary `BroadcastChannel('playgta6-httpfs-sab')` (closed immediately once SAB arrives) so pthreads can call `host_httpfs_*` without keeping any open BroadcastChannels.
 * - Configures side-module `locateFile` resolution against `dist/` for `guest_part_*.wasm`.
 * - Installs the `WebAssembly.Instance` host-import shim BEFORE `importScripts(runtimeScriptUrl)`.
 */
function bootstrapEmscriptenPthreadWorker(options = {}) {
  const scope =
    options.scope ||
    (typeof self !== 'undefined' ? self : typeof globalThis !== 'undefined' ? globalThis : null);
  const importScriptsFn =
    typeof options.importScriptsFn === 'function'
      ? options.importScriptsFn
      : typeof importScripts === 'function'
        ? importScripts
        : null;
  const runtimeScriptUrl =
    typeof options.runtimeScriptUrl === 'string' && options.runtimeScriptUrl.trim().length > 0
      ? options.runtimeScriptUrl.trim()
      : DEFAULT_RUNTIME_SCRIPT_RELATIVE;
  const scopeLocationHref =
    typeof options.locationHref === 'string' && options.locationHref.trim().length > 0
      ? options.locationHref.trim()
      : scope && scope.location && typeof scope.location.href === 'string'
        ? scope.location.href
        : '';
  const resolvedRuntimeScriptUrl = resolveRuntimeScriptUrl(
    runtimeScriptUrl === DEFAULT_RUNTIME_SCRIPT_RELATIVE ? undefined : runtimeScriptUrl,
    scopeLocationHref,
  );
  const BroadcastChannelCtor =
    options.BroadcastChannelCtor !== undefined
      ? options.BroadcastChannelCtor
      : scope && typeof scope.BroadcastChannel === 'function'
        ? scope.BroadcastChannel
        : typeof BroadcastChannel === 'function'
          ? BroadcastChannel
          : null;
  const mailboxApi =
    options.httpfsMailboxApi !== undefined
      ? options.httpfsMailboxApi
      : (scope && scope.PlayGta6HttpfsMailbox) ||
        (typeof globalThis !== 'undefined' && globalThis.PlayGta6HttpfsMailbox) ||
        null;

  let txPort = null;
  let txPortClosed = false;
  let txPortRecycleArmed = false;
  const cmdBufferPool = createWarmCmdBufferPool();

  const onTxPortMessage = (ev) => {
    const msg = ev && ev.data !== undefined ? ev.data : ev;
    if (
      msg &&
      typeof msg === 'object' &&
      msg.type === PTHREAD_TX_MSG_RECYCLE_CMD_BUFFER &&
      msg.cmdBuffer
    ) {
      cmdBufferPool.recycle(msg.cmdBuffer);
    }
  };

  const armTxPortRecycle = (port) => {
    if (!port || txPortRecycleArmed) {
      return;
    }
    txPortRecycleArmed = true;
    try {
      port.onmessage = onTxPortMessage;
    } catch {
      // Non-fatal if a mock port exposes read-only onmessage.
    }
    if (typeof port.unref === 'function') {
      try {
        port.unref();
      } catch {
        // Ignore unref failures on non-Node MessagePort implementations.
      }
    }
  };

  const closeTxPort = () => {
    const port = txPort;
    txPort = null;
    txPortClosed = true;
    if (port && txPortRecycleArmed) {
      try {
        port.onmessage = null;
      } catch {
        // Ignore errors clearing MessagePort handler.
      }
    }
    txPortRecycleArmed = false;
    closeMessagePort(port);
  };

  const attachTxPort = (port) => {
    if (!port || typeof port.postMessage !== 'function') {
      return false;
    }
    if (txPort === port) {
      txPortClosed = false;
      return true;
    }
    if (txPort && txPort !== port) {
      if (txPortRecycleArmed) {
        try {
          txPort.onmessage = null;
        } catch {
          // Ignore errors clearing previous MessagePort handler.
        }
      }
      closeMessagePort(txPort);
    }
    txPort = port;
    txPortClosed = false;
    txPortRecycleArmed = false;
    return true;
  };

  if (options.txPort) {
    attachTxPort(options.txPort);
  }

  let lastPthreadFaultRecord = null;
  const pthreadConsoleObj =
    options.console && typeof options.console.error === 'function'
      ? options.console
      : scope && scope.console && typeof scope.console.error === 'function'
        ? scope.console
        : typeof console !== 'undefined'
          ? console
          : null;
  const pthreadWorkerName =
    scope && typeof scope.name === 'string' && scope.name.length > 0
      ? scope.name
      : EMSCRIPTEN_PTHREAD_WORKER_NAME;

  const pthreadReport = (label) => {
    const payload = { label };
    if (isFaultRecordLine(label)) {
      const faultRecord = parseFaultRecordLine(label);
      if (faultRecord) {
        lastPthreadFaultRecord = faultRecord;
        if (scope && typeof scope === 'object') {
          try {
            scope.__playgta6LastFaultRecord = cloneFaultRecord(faultRecord);
          } catch {
            // Ignore non-writable scope property.
          }
        }
      }
      // Emit synchronously to console.error BEFORE std::abort() can tear down the pthread worker
      // so Chrome CDP captures the complete [fault-record] line even if IPC ports close immediately.
      emitFaultRecordConsoleError(label, pthreadConsoleObj);
      // Deliver directly over BroadcastChannel('game-progress') so main-thread BootTracer receives
      // [fault-record] immediately even if coordinator txPort is closed on worker error/abort.
      const broadcastSent = postBroadcastOnce(
        BroadcastChannelCtor,
        PROGRESS_CHANNEL_NAME,
        payload,
      );
      let txSent = false;
      if (txPort && !txPortClosed) {
        try {
          txPort.postMessage({
            type: PTHREAD_TX_MSG_FAULT_RECORD,
            faultRecord: faultRecord ? cloneFaultRecord(faultRecord) : null,
            label,
            rawLine: label,
            workerName: pthreadWorkerName,
            broadcastSent,
            consoleEmitted: true,
          });
          txSent = true;
        } catch {
          closeTxPort();
        }
      }
      return broadcastSent || txSent;
    }
    if (txPort && !txPortClosed) {
      try {
        txPort.postMessage({ type: PTHREAD_TX_MSG_PROGRESS, payload });
        return true;
      } catch {
        closeTxPort();
      }
    }
    return postBroadcastOnce(BroadcastChannelCtor, PROGRESS_CHANNEL_NAME, payload);
  };

  const pthreadLocateFile = createRuntimeLocateFile(
    resolvedRuntimeScriptUrl,
    options.locateFileBase,
    {
      reportFn: pthreadReport,
      sideModuleNames: sanitizeSideModuleUrls(options.sideModuleUrls),
    },
  );

  // Ensure Emscripten's pthread bootstrap resolves side modules (`guest_part_*.wasm`)
  // relative to `dist/` even if the worker's `self.location.href` is `loader.js`.
  let installedScopeModule = false;
  let installedLocateFileOnModule = false;
  let installedMainScriptOnModule = false;
  if (scope && typeof scope === 'object') {
    if (!scope.Module || typeof scope.Module !== 'object') {
      scope.Module = {
        locateFile: pthreadLocateFile,
        mainScriptUrlOrBlob: resolvedRuntimeScriptUrl,
      };
      installedScopeModule = true;
    } else {
      if (typeof scope.Module.locateFile !== 'function') {
        scope.Module.locateFile = pthreadLocateFile;
        installedLocateFileOnModule = true;
      }
      if (!scope.Module.mainScriptUrlOrBlob) {
        scope.Module.mainScriptUrlOrBlob = resolvedRuntimeScriptUrl;
        installedMainScriptOnModule = true;
      }
    }
  }

  let pthreadHttpfsClient = null;
  let attachedHttpfsSab = null;
  let httpfsChannel = null;

  // W371 (#1077): Track precompiled WebAssembly.Module side modules delivered via CMD_LOAD
  // or options so pthreads reuse compiled modules without ever re-fetching or re-compiling.
  const pthreadSharedModulesByName = new Map();
  const pthreadSharedModulesList = [];
  let pthreadSharedModuleHits = 0;

  const normalizeSharedModuleName = (rawName) => {
    if (typeof rawName !== 'string' || rawName.length === 0) {
      return '';
    }
    const base = extractWasmBasename(rawName) || rawName.trim();
    return base.toLowerCase();
  };

  const registerPthreadSharedModules = (modules, names) => {
    const targetWasmNs =
      scope && scope.WebAssembly
        ? scope.WebAssembly
        : typeof globalThis !== 'undefined' && globalThis.WebAssembly
          ? globalThis.WebAssembly
          : null;
    if (modules && typeof modules === 'object' && !Array.isArray(modules)) {
      if (modules instanceof Map) {
        modules.forEach((mod, key) => {
          if (isCompiledWasmModule(mod, targetWasmNs)) {
            const norm = normalizeSharedModuleName(String(key));
            if (norm) {
              pthreadSharedModulesByName.set(norm, mod);
            }
            if (!pthreadSharedModulesList.includes(mod)) {
              pthreadSharedModulesList.push(mod);
            }
          }
        });
      } else {
        const keys = Object.keys(modules);
        for (let i = 0; i < keys.length; i += 1) {
          const mod = modules[keys[i]];
          if (isCompiledWasmModule(mod, targetWasmNs)) {
            const norm = normalizeSharedModuleName(keys[i]);
            if (norm) {
              pthreadSharedModulesByName.set(norm, mod);
            }
            if (!pthreadSharedModulesList.includes(mod)) {
              pthreadSharedModulesList.push(mod);
            }
          }
        }
      }
    } else if (Array.isArray(modules)) {
      stripWireBytesFromSharedModules(modules, { wasmApi: targetWasmNs, recordTelemetry: false });
      for (let i = 0; i < modules.length; i += 1) {
        const mod = modules[i];
        if (!isCompiledWasmModule(mod, targetWasmNs)) {
          continue;
        }
        if (!pthreadSharedModulesList.includes(mod)) {
          pthreadSharedModulesList.push(mod);
        }
        const rawName = Array.isArray(names) && typeof names[i] === 'string' ? names[i] : '';
        const norm = normalizeSharedModuleName(rawName);
        if (norm) {
          pthreadSharedModulesByName.set(norm, mod);
        }
      }
    }
    if (
      scope &&
      scope.Module &&
      typeof scope.Module === 'object' &&
      pthreadSharedModulesList.length > 0 &&
      (!Array.isArray(scope.Module.sharedModules) || scope.Module.sharedModules.length === 0)
    ) {
      scope.Module.sharedModules = pthreadSharedModulesList.slice();
    }
  };

  if (options.sharedModulesByName) {
    registerPthreadSharedModules(options.sharedModulesByName);
  }
  if (Array.isArray(options.sharedModules)) {
    registerPthreadSharedModules(options.sharedModules, options.sharedModuleNames);
  }

  const closeHttpfsChannel = () => {
    const ch = httpfsChannel;
    httpfsChannel = null;
    if (!ch) {
      return;
    }
    try {
      ch.onmessage = null;
    } catch {
      // Ignore errors clearing BroadcastChannel message handler.
    }
    closeBroadcastChannel(ch);
  };

  const hostImports = createHostImports({
    log: (msg) => pthreadReport(`[wasm] ${msg}`),
    report: () => {},
    cmdBufferPool,
    wgpuSubmit: (bytes, meta) => {
      const hasRecyclableMeta = Boolean(
        meta &&
          meta.recycleBuffer === true &&
          isTransferableArrayBuffer(meta.cmdBuffer) &&
          typeof meta.byteLength === 'number',
      );
      const cmdBuffer = hasRecyclableMeta ? meta.cmdBuffer : toTransferableArrayBuffer(bytes);
      if (!cmdBuffer) {
        return;
      }
      if (txPort && !txPortClosed) {
        if (hasRecyclableMeta) {
          armTxPortRecycle(txPort);
        }
        const msg = hasRecyclableMeta
          ? {
              type: PTHREAD_TX_MSG_WGPU_SUBMIT,
              cmdBuffer,
              byteLength: meta.byteLength,
              recycleBuffer: true,
            }
          : { type: PTHREAD_TX_MSG_WGPU_SUBMIT, cmdBuffer };
        try {
          txPort.postMessage(msg, [cmdBuffer]);
          return;
        } catch {
          closeTxPort();
        }
      }
      if (typeof BroadcastChannelCtor !== 'function') {
        return;
      }
      const bcMsg = hasRecyclableMeta
        ? {
            type: PTHREAD_TX_MSG_WGPU_SUBMIT,
            cmdBuffer,
            byteLength: meta.byteLength,
          }
        : {
            type: PTHREAD_TX_MSG_WGPU_SUBMIT,
            cmdBuffer,
          };
      postBroadcastOnce(BroadcastChannelCtor, PTHREAD_WGPU_CHANNEL_NAME, bcMsg);
    },
    getHttpfsClient: () => pthreadHttpfsClient,
  });

  const attachHttpfsSab = (sab) => {
    if (!sab || !mailboxApi || typeof mailboxApi.HttpfsMailboxClient !== 'function') {
      return false;
    }
    if (attachedHttpfsSab === sab && pthreadHttpfsClient) {
      closeHttpfsChannel();
      return true;
    }
    try {
      pthreadHttpfsClient = new mailboxApi.HttpfsMailboxClient(sab);
      attachedHttpfsSab = sab;
      hostImports.setHttpfsClient(pthreadHttpfsClient);
      closeHttpfsChannel();
      return true;
    } catch {
      return false;
    }
  };

  if (options.httpfsMailboxSab) {
    attachHttpfsSab(options.httpfsMailboxSab);
  }

  const onPthreadBootstrapMessage = (ev) => {
    const msg = ev && ev.data;
    if (!msg || typeof msg !== 'object') {
      return;
    }
    let handled = false;
    const incomingHttpfsSab = msg.__playgta6_httpfs_lane_sab || msg.__playgta6_httpfs_sab;
    if (incomingHttpfsSab) {
      attachHttpfsSab(incomingHttpfsSab);
      handled = true;
    }
    if (msg.__playgta6_tx_port) {
      attachTxPort(msg.__playgta6_tx_port);
      handled = true;
    }
    if (msg.__playgta6_shared_modules) {
      registerPthreadSharedModules(
        msg.__playgta6_shared_modules,
        msg.__playgta6_shared_module_names,
      );
      handled = true;
    }
    // Passively observe CMD_LOAD sharedModules WITHOUT stopping propagation so Emscripten's
    // own pthread message handler still receives CMD_LOAD intact.
    if (isPthreadLoadMessage(msg) && msg.sharedModules) {
      registerPthreadSharedModules(msg.sharedModules, msg.__playgta6_shared_module_names);
    }
    if (handled && typeof ev.stopImmediatePropagation === 'function') {
      ev.stopImmediatePropagation();
    }
  };

  let installedBootstrapMessageListener = false;
  if (scope && typeof scope.addEventListener === 'function') {
    scope.addEventListener('message', onPthreadBootstrapMessage);
    installedBootstrapMessageListener = true;
  }

  const canAttachHttpfs = Boolean(
    mailboxApi && typeof mailboxApi.HttpfsMailboxClient === 'function',
  );
  if (!attachedHttpfsSab && canAttachHttpfs) {
    httpfsChannel = tryOpenBroadcastChannel(
      BroadcastChannelCtor,
      PTHREAD_HTTPFS_SAB_CHANNEL_NAME,
    );
    if (httpfsChannel) {
      httpfsChannel.onmessage = (ev) => {
        const msg = ev && ev.data;
        if (msg && msg.type === 'httpfs-sab') {
          const candidateSab =
            msg.sab || (Array.isArray(msg.sabs) && msg.sabs.length > 0 ? msg.sabs[0] : null);
          if (candidateSab) {
            attachHttpfsSab(candidateSab);
          }
        }
      };
      try {
        if (typeof httpfsChannel.postMessage === 'function') {
          httpfsChannel.postMessage({ type: 'request-httpfs-sab' });
        }
      } catch {
        // Ignore runtimes where BroadcastChannel messaging is restricted.
      }
    }
  }

  const wasmTargetScope =
    scope && scope.WebAssembly
      ? scope
      : typeof globalThis !== 'undefined' && globalThis.WebAssembly
        ? globalThis
        : scope;
  const resolvePthreadSharedModule = (urlOrName) => {
    const key = normalizeSharedModuleName(urlOrName);
    if (!key) {
      return null;
    }
    const hit = pthreadSharedModulesByName.get(key) || null;
    if (hit) {
      pthreadSharedModuleHits += 1;
    }
    return hit;
  };
  const restoreWasmInstance = installPthreadHostImportShim(wasmTargetScope, hostImports, {
    fetchScope: scope,
    resolveSharedModule: resolvePthreadSharedModule,
  });

  if (importScriptsFn) {
    importScriptsFn(runtimeScriptUrl);
  }

  return {
    hostImports,
    locateFile: pthreadLocateFile,
    runtimeScriptUrl: resolvedRuntimeScriptUrl,
    restoreWasmInstance,
    attachHttpfsSab,
    hasHttpfsSab: () => Boolean(pthreadHttpfsClient),
    getHttpfsSab: () => attachedHttpfsSab,
    attachTxPort,
    hasTxPort: () => Boolean(txPort && !txPortClosed),
    getLastFaultRecord: () => (lastPthreadFaultRecord ? cloneFaultRecord(lastPthreadFaultRecord) : null),
    getSharedModules: () => pthreadSharedModulesList.slice(),
    getSharedModuleByName: (name) =>
      pthreadSharedModulesByName.get(normalizeSharedModuleName(name)) || null,
    getSharedModuleHits: () => pthreadSharedModuleHits,
    cmdBufferPool,
    channels: {
      progress: null,
      wgpu: null,
      get httpfs() {
        return httpfsChannel;
      },
      get txPort() {
        return txPort;
      },
    },
    cleanup() {
      restoreWasmInstance();
      closeTxPort();
      cmdBufferPool.clear();
      pthreadSharedModulesByName.clear();
      pthreadSharedModulesList.length = 0;
      attachedHttpfsSab = null;
      pthreadHttpfsClient = null;
      if (hostImports && typeof hostImports.setHttpfsClient === 'function') {
        hostImports.setHttpfsClient(null);
      }
      if (
        installedBootstrapMessageListener &&
        scope &&
        typeof scope.removeEventListener === 'function'
      ) {
        try {
          scope.removeEventListener('message', onPthreadBootstrapMessage);
        } catch {
          // Ignore removeEventListener errors during cleanup.
        }
        installedBootstrapMessageListener = false;
      }
      if (scope && typeof scope === 'object') {
        if (installedScopeModule) {
          delete scope.Module;
        } else if (scope.Module && typeof scope.Module === 'object') {
          if (installedLocateFileOnModule && scope.Module.locateFile === pthreadLocateFile) {
            delete scope.Module.locateFile;
          }
          if (
            installedMainScriptOnModule &&
            scope.Module.mainScriptUrlOrBlob === resolvedRuntimeScriptUrl
          ) {
            delete scope.Module.mainScriptUrlOrBlob;
          }
        }
      }
      closeHttpfsChannel();
    },
  };
}

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

function extractWasmBasename(value) {
  if (typeof value !== 'string' || value.length === 0) {
    return '';
  }
  const clean = value.split(/[?#]/, 1)[0];
  return basenameOfPath(clean);
}

function sanitizeSideModuleUrl(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const trimmed = raw.trim();
  if (!trimmed || trimmed.includes('\0') || trimmed.includes('\\')) {
    return null;
  }
  const withoutQuery = trimmed.split(/[?#]/, 1)[0];
  if (!/\.wasm$/i.test(withoutQuery)) {
    return null;
  }
  if (isAbsoluteUrl(trimmed)) {
    if (!/^https?:/i.test(trimmed)) {
      return null;
    }
    if (typeof URL === 'function') {
      try {
        const parsed = new URL(trimmed);
        if (parsed.pathname.split('/').includes('..')) {
          return null;
        }
      } catch {
        return null;
      }
    }
    return trimmed;
  }
  const segments = withoutQuery.replace(/^\/+/, '').split('/');
  for (let i = 0; i < segments.length; i += 1) {
    if (segments[i] === '..') {
      return null;
    }
  }
  return trimmed;
}

function sanitizeSideModuleUrls(rawList) {
  if (!Array.isArray(rawList)) {
    return [];
  }
  const out = [];
  const seen = new Set();
  for (let i = 0; i < rawList.length; i += 1) {
    const clean = sanitizeSideModuleUrl(rawList[i]);
    if (clean && !seen.has(clean)) {
      seen.add(clean);
      out.push(clean);
    }
  }
  return out;
}

/**
 * Resolve a runtime asset or Emscripten `SIDE_MODULE=1` (`guest_part_*.wasm`)
 * filename/URL relative to `locateFileBase` or the `dist/` directory of
 * `runtimeScriptUrl`, avoiding duplicate `dist/dist/` prefixes.
 */
function resolveSideModuleUrl(file, runtimeScriptUrl, locateFileBase) {
  if (typeof file !== 'string') {
    return '';
  }
  const trimmed = file.trim();
  if (trimmed.length === 0) {
    return '';
  }
  if (isAbsoluteUrl(trimmed)) {
    return trimmed;
  }

  const explicitBase =
    typeof locateFileBase === 'string' && locateFileBase.trim().length > 0
      ? ensureTrailingSlash(locateFileBase.trim())
      : '';
  const effectiveScriptUrl =
    typeof runtimeScriptUrl === 'string' && runtimeScriptUrl.trim().length > 0
      ? runtimeScriptUrl.trim()
      : DEFAULT_RUNTIME_SCRIPT_FALLBACK;

  if (trimmed.startsWith('/')) {
    const absBase = isAbsoluteUrl(explicitBase)
      ? explicitBase
      : isAbsoluteUrl(effectiveScriptUrl)
        ? effectiveScriptUrl
        : '';
    if (absBase && typeof URL === 'function') {
      try {
        return new URL(trimmed, absBase).href;
      } catch {
        // Fall through to returning root-relative path.
      }
    }
    return trimmed;
  }

  const relNoDot = trimmed.replace(/^\.\//, '');

  if (explicitBase) {
    if (isAbsoluteUrl(explicitBase) && typeof URL === 'function') {
      try {
        return new URL(relNoDot, explicitBase).href;
      } catch {
        // Fall through to string join.
      }
    }
    return `${explicitBase}${relNoDot}`;
  }

  const scriptDir = dirnameWithSlash(effectiveScriptUrl);
  const effectiveRel =
    (scriptDir === 'dist/' || scriptDir.endsWith('/dist/')) && relNoDot.startsWith('dist/')
      ? relNoDot.slice('dist/'.length)
      : relNoDot;

  if (isAbsoluteUrl(effectiveScriptUrl) && typeof URL === 'function') {
    try {
      return new URL(effectiveRel, effectiveScriptUrl).href;
    } catch {
      // Fall through to relative dist/ join.
    }
  }
  return `${scriptDir}${effectiveRel}`;
}

function normalizeIndirectDispatchMode(raw) {
  if (typeof raw === 'string') {
    const lower = raw.trim().toLowerCase();
    if (INDIRECT_DISPATCH_MODES.has(lower)) {
      return lower;
    }
  }
  return 'table';
}

function sanitizeManifestScriptUrl(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const trimmed = raw.trim();
  if (!trimmed || trimmed.includes('\0') || trimmed.includes('\\')) {
    return null;
  }
  const withoutQuery = trimmed.split(/[?#]/, 1)[0];
  if (!/\.m?js$/i.test(withoutQuery)) {
    return null;
  }
  if (isAbsoluteUrl(trimmed)) {
    if (!/^https?:/i.test(trimmed)) {
      return null;
    }
    if (typeof URL === 'function') {
      try {
        const parsed = new URL(trimmed);
        if (parsed.pathname.split('/').includes('..')) {
          return null;
        }
      } catch {
        return null;
      }
    }
    return trimmed;
  }
  const segments = withoutQuery.replace(/^\/+/, '').split('/');
  for (let i = 0; i < segments.length; i += 1) {
    if (segments[i] === '..') {
      return null;
    }
  }
  return trimmed;
}

function extractSideModuleEntryUrl(entry) {
  if (typeof entry === 'string') {
    return sanitizeSideModuleUrl(entry);
  }
  if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
    const candidate =
      entry.url !== undefined
        ? entry.url
        : entry.path !== undefined
          ? entry.path
          : entry.name !== undefined
            ? entry.name
            : entry.file;
    return sanitizeSideModuleUrl(candidate);
  }
  return null;
}

function sanitizeManifestSideModules(rawList) {
  if (!Array.isArray(rawList)) {
    return [];
  }
  const out = [];
  const seen = new Set();
  for (let i = 0; i < rawList.length; i += 1) {
    const clean = extractSideModuleEntryUrl(rawList[i]);
    if (clean && !seen.has(clean)) {
      seen.add(clean);
      out.push(clean);
    }
  }
  return out;
}

function pickFirstDefined(obj, keys) {
  if (!obj || typeof obj !== 'object') {
    return undefined;
  }
  for (let i = 0; i < keys.length; i += 1) {
    const val = obj[keys[i]];
    if (val !== undefined && val !== null && val !== '') {
      return val;
    }
  }
  return undefined;
}

function toOptionalNonNegativeInt(value) {
  if (value === undefined || value === null || value === '' || typeof value === 'boolean') {
    return null;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return null;
  }
  return Math.floor(n);
}

function sanitizeWasmMetrics(rawMetrics, fallbackSideModuleCount = 0, topLevelManifest = null) {
  const src =
    rawMetrics && typeof rawMetrics === 'object' && !Array.isArray(rawMetrics) ? rawMetrics : null;
  const top =
    topLevelManifest && typeof topLevelManifest === 'object' && !Array.isArray(topLevelManifest)
      ? topLevelManifest
      : null;

  const totalTableSize = toOptionalNonNegativeInt(
    pickFirstDefined(src, [
      'totalTableSize',
      'total_table_size',
      'tableSize',
      'table_size',
      'wasmTableSize',
      'wasm_table_size',
    ]) !== undefined
      ? pickFirstDefined(src, [
          'totalTableSize',
          'total_table_size',
          'tableSize',
          'table_size',
          'wasmTableSize',
          'wasm_table_size',
        ])
      : pickFirstDefined(top, [
          'totalTableSize',
          'total_table_size',
          'wasmTableSize',
          'wasm_table_size',
        ]),
  );
  const totalGotFuncImports = toOptionalNonNegativeInt(
    pickFirstDefined(src, [
      'totalGotFuncImports',
      'total_got_func_imports',
      'gotFuncImports',
      'got_func_imports',
    ]) !== undefined
      ? pickFirstDefined(src, [
          'totalGotFuncImports',
          'total_got_func_imports',
          'gotFuncImports',
          'got_func_imports',
        ])
      : pickFirstDefined(top, [
          'totalGotFuncImports',
          'total_got_func_imports',
          'gotFuncImports',
          'got_func_imports',
        ]),
  );
  const explicitSideModuleCount = toOptionalNonNegativeInt(
    pickFirstDefined(src, ['sideModuleCount', 'side_module_count']),
  );

  const optionalSpecs = [
    [
      'totalGotMemImports',
      ['totalGotMemImports', 'total_got_mem_imports', 'gotMemImports', 'got_mem_imports'],
    ],
    ['totalWasmBytes', ['totalWasmBytes', 'total_wasm_bytes', 'wasmBytes', 'wasm_bytes']],
    [
      'mainWasmBytes',
      ['mainWasmBytes', 'main_wasm_bytes', 'runtimeWasmBytes', 'runtime_wasm_bytes'],
    ],
    ['sideModulesTotalBytes', ['sideModulesTotalBytes', 'side_modules_total_bytes']],
    ['shardCount', ['shardCount', 'shard_count']],
    ['dispatchShardCount', ['dispatchShardCount', 'dispatch_shard_count']],
    ['functionCount', ['functionCount', 'function_count']],
    ['exportedFunctionCount', ['exportedFunctionCount', 'exported_function_count']],
  ];

  const extras = {};
  let hasAnyOptional = false;
  if (src) {
    for (let i = 0; i < optionalSpecs.length; i += 1) {
      const [outKey, aliases] = optionalSpecs[i];
      const val = toOptionalNonNegativeInt(pickFirstDefined(src, aliases));
      if (val !== null) {
        extras[outKey] = val;
        hasAnyOptional = true;
      }
    }
  }

  const hasMetricsSource =
    Boolean(src) || totalTableSize !== null || totalGotFuncImports !== null || hasAnyOptional;
  if (!hasMetricsSource) {
    return null;
  }

  const baseCount = toNonNegativeInt(fallbackSideModuleCount, 0);
  const sideModuleCount =
    explicitSideModuleCount !== null ? Math.max(explicitSideModuleCount, baseCount) : baseCount;

  const sanitized = {
    sideModuleCount,
  };
  if (totalTableSize !== null) {
    sanitized.totalTableSize = totalTableSize;
  }
  if (totalGotFuncImports !== null) {
    sanitized.totalGotFuncImports = totalGotFuncImports;
  }
  const extraKeys = Object.keys(extras);
  for (let i = 0; i < extraKeys.length; i += 1) {
    sanitized[extraKeys[i]] = extras[extraKeys[i]];
  }
  return sanitized;
}

/**
 * Parse and sanitize `guest_manifest.json` while preserving backwards compatibility with
 * legacy `table`-dispatch manifests and surfacing optional `indirectDispatch` (`"shard"` | `"table"`)
 * and numeric-only `wasmMetrics` (`totalTableSize`, `totalGotFuncImports`, `sideModuleCount`, ...).
 */
function parseGuestManifest(rawManifest) {
  let manifest = rawManifest;
  if (typeof manifest === 'string') {
    try {
      manifest = JSON.parse(manifest);
    } catch {
      return null;
    }
  }
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) {
    return null;
  }

  const rawSideList =
    manifest.sideModules !== undefined
      ? manifest.sideModules
      : manifest.side_modules !== undefined
        ? manifest.side_modules
        : manifest.sideModuleUrls !== undefined
          ? manifest.sideModuleUrls
          : manifest.side_module_urls !== undefined
            ? manifest.side_module_urls
            : manifest.dynamicLibraries !== undefined
              ? manifest.dynamicLibraries
              : manifest.dynamic_libraries;
  const sideModules = sanitizeManifestSideModules(rawSideList);

  const explicitSideCount = toOptionalNonNegativeInt(
    pickFirstDefined(manifest, ['sideModuleCount', 'side_module_count']),
  );
  const sideModuleCount =
    explicitSideCount !== null ? Math.max(explicitSideCount, sideModules.length) : sideModules.length;

  const rawMetrics =
    manifest.wasmMetrics !== undefined ? manifest.wasmMetrics : manifest.wasm_metrics;
  const rawDispatch =
    pickFirstDefined(manifest, ['indirectDispatch', 'indirect_dispatch']) !== undefined
      ? pickFirstDefined(manifest, ['indirectDispatch', 'indirect_dispatch'])
      : pickFirstDefined(rawMetrics, ['indirectDispatch', 'indirect_dispatch']);
  const indirectDispatch = normalizeIndirectDispatchMode(rawDispatch);

  const wasmMetrics = sanitizeWasmMetrics(rawMetrics, sideModuleCount, manifest);
  const maxConcurrentPthreadLoads = toOptionalNonNegativeInt(
    pickFirstDefined(manifest, ['maxConcurrentPthreadLoads', 'max_concurrent_pthread_loads']),
  );
  const runtimeScriptUrl = sanitizeManifestScriptUrl(
    pickFirstDefined(manifest, ['runtimeScriptUrl', 'runtimeJs', 'runtime_js']),
  );

  return {
    indirectDispatch,
    sideModules,
    sideModuleCount,
    wasmMetrics,
    maxConcurrentPthreadLoads,
    runtimeScriptUrl,
  };
}

function extractSideModuleEntries(rawManifest) {
  let manifest = rawManifest;
  if (typeof manifest === 'string') {
    try {
      manifest = JSON.parse(manifest);
    } catch {
      return [];
    }
  }
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) {
    return [];
  }
  const rawSideList =
    manifest.sideModules !== undefined
      ? manifest.sideModules
      : manifest.side_modules !== undefined
        ? manifest.side_modules
        : manifest.sideModuleUrls !== undefined
          ? manifest.sideModuleUrls
          : manifest.side_module_urls !== undefined
            ? manifest.side_module_urls
            : manifest.dynamicLibraries !== undefined
              ? manifest.dynamicLibraries
              : manifest.dynamic_libraries;
  if (!Array.isArray(rawSideList)) {
    return [];
  }
  const entries = [];
  const seen = new Set();
  for (let i = 0; i < rawSideList.length; i += 1) {
    const raw = rawSideList[i];
    const url = extractSideModuleEntryUrl(raw);
    if (!url || seen.has(url)) {
      continue;
    }
    seen.add(url);
    const name = extractWasmBasename(url) || url;
    const obj = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : null;
    const byteLength = obj
      ? toOptionalNonNegativeInt(
          pickFirstDefined(obj, [
            'wasmBytes',
            'wasm_bytes',
            'byteLength',
            'byte_length',
            'sizeBytes',
            'size_bytes',
            'size',
            'bytes',
          ]),
        )
      : null;
    const shardCount = obj
      ? toOptionalNonNegativeInt(pickFirstDefined(obj, ['shardCount', 'shard_count', 'shards']))
      : null;
    entries.push({
      name,
      url,
      byteLength,
      shardCount,
    });
  }
  return entries;
}

/**
 * Evaluate V8 1024 MiB per-side-module size cap headroom and asymmetric/collapsed split shape
 * (#663). Accepts a raw manifest object/JSON, an array of `{ name, byteLength }` descriptors,
 * a `Map<name, byteLength>`, or a single numeric byte size. Never leaks host paths.
 */
function evaluateV8SideModuleBudget(input, options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  const capBytes =
    typeof safeOpts.capBytes === 'number' && Number.isFinite(safeOpts.capBytes) && safeOpts.capBytes > 0
      ? Math.floor(safeOpts.capBytes)
      : V8_SIDE_MODULE_CAP_BYTES;
  const warnRatio =
    typeof safeOpts.warnRatio === 'number' &&
    Number.isFinite(safeOpts.warnRatio) &&
    safeOpts.warnRatio > 0 &&
    safeOpts.warnRatio <= 1
      ? safeOpts.warnRatio
      : V8_SIDE_MODULE_WARN_RATIO;
  const warnThresholdBytes = Math.floor(capBytes * warnRatio);

  const normalized = [];
  if (typeof input === 'number' && Number.isFinite(input) && input >= 0) {
    normalized.push({ name: 'side_module.wasm', byteLength: Math.floor(input) });
  } else if (input instanceof Map) {
    input.forEach((val, key) => {
      const len = toOptionalNonNegativeInt(val);
      const base = extractWasmBasename(String(key || '')) || 'side_module.wasm';
      normalized.push({ name: base, byteLength: len });
    });
  } else if (Array.isArray(input)) {
    for (let i = 0; i < input.length; i += 1) {
      const item = input[i];
      if (typeof item === 'number' && Number.isFinite(item) && item >= 0) {
        normalized.push({ name: `guest_part_${i}.wasm`, byteLength: Math.floor(item) });
      } else if (item && typeof item === 'object') {
        const rawName = item.name || item.url || item.path || item.file || `guest_part_${i}.wasm`;
        const base = extractWasmBasename(String(rawName)) || `guest_part_${i}.wasm`;
        const len = toOptionalNonNegativeInt(
          pickFirstDefined(item, [
            'byteLength',
            'byte_length',
            'wasmBytes',
            'wasm_bytes',
            'sizeBytes',
            'size_bytes',
            'size',
            'bytes',
          ]),
        );
        normalized.push({ name: base, byteLength: len });
      }
    }
  } else if (input && typeof input === 'object') {
    const extracted = extractSideModuleEntries(input);
    for (let i = 0; i < extracted.length; i += 1) {
      normalized.push({ name: extracted[i].name, byteLength: extracted[i].byteLength });
    }
    const rawMetrics = input.wasmMetrics || input.wasm_metrics || null;
    const explicitLargest = toOptionalNonNegativeInt(
      pickFirstDefined(rawMetrics, ['largestSideModuleBytes', 'largest_side_module_bytes']) ??
        pickFirstDefined(input, ['largestSideModuleBytes', 'largest_side_module_bytes']),
    );
    if (explicitLargest !== null && normalized.every((e) => e.byteLength === null)) {
      if (normalized.length > 0) {
        normalized[0].byteLength = explicitLargest;
      } else {
        normalized.push({ name: 'guest_part_0.wasm', byteLength: explicitLargest });
      }
    }
  }

  let totalSideModuleBytes = 0;
  let largestSideModuleBytes = 0;
  let largestSideModuleName = null;
  let smallestSideModuleBytes = null;
  let sizedCount = 0;
  const warnings = [];

  for (let i = 0; i < normalized.length; i += 1) {
    const entry = normalized[i];
    const len = entry.byteLength;
    if (len === null || len === undefined) {
      continue;
    }
    sizedCount += 1;
    totalSideModuleBytes += len;
    if (len >= largestSideModuleBytes) {
      largestSideModuleBytes = len;
      largestSideModuleName = entry.name;
    }
    if (smallestSideModuleBytes === null || len < smallestSideModuleBytes) {
      smallestSideModuleBytes = len;
    }
    const mib = (len / (1024 * 1024)).toFixed(1);
    const capMib = (capBytes / (1024 * 1024)).toFixed(0);
    if (len > capBytes) {
      warnings.push(
        `Wasm side module ${entry.name} (${mib} MiB) exceeds V8 ${capMib} MiB per-module cap`,
      );
    } else if (len >= warnThresholdBytes) {
      const headroomMib = ((capBytes - len) / (1024 * 1024)).toFixed(1);
      warnings.push(
        `Wasm side module ${entry.name} (${mib} MiB) is within ${headroomMib} MiB of V8 ${capMib} MiB cap`,
      );
    }
  }

  const sideModuleCount = normalized.length;
  const sideModuleHeadroomBytes = Math.max(0, capBytes - largestSideModuleBytes);
  const utilizationRatio =
    capBytes > 0 ? Math.round((largestSideModuleBytes / capBytes) * 10000) / 10000 : 0;
  const exceedsCap = largestSideModuleBytes > capBytes;
  const nearCapWarning = largestSideModuleBytes >= warnThresholdBytes;

  let splitShape = 'none';
  if (sideModuleCount === 1) {
    splitShape = 'single-collapsed';
  } else if (sideModuleCount >= 2) {
    if (
      sizedCount >= 2 &&
      smallestSideModuleBytes !== null &&
      (smallestSideModuleBytes === 0 || largestSideModuleBytes >= smallestSideModuleBytes * 10)
    ) {
      splitShape = 'asymmetric';
    } else {
      splitShape = 'balanced';
    }
  }

  return {
    v8SideModuleCapBytes: capBytes,
    warnThresholdBytes,
    warnRatio,
    sideModuleCount,
    totalSideModuleBytes,
    largestSideModuleBytes,
    largestSideModuleName,
    smallestSideModuleBytes: smallestSideModuleBytes !== null ? smallestSideModuleBytes : 0,
    sideModuleHeadroomBytes,
    utilizationRatio,
    nearCapWarning,
    exceedsCap,
    splitShape,
    warnings,
  };
}

/**
 * Inspect a Wasm runtime module's exports cleanly without expecting or invoking `_main` (#663).
 * Reports whether `_main` is exported and which fast-boot exports are present.
 */
function inspectRuntimeExports(mod) {
  if (!mod || typeof mod !== 'object') {
    return {
      hasMain: false,
      hasCallMain: false,
      fastBootExports: [],
      mode: 'fast-boot',
    };
  }
  const hasFn = (name) => {
    const prefixed = `_${name}`;
    return Boolean(
      typeof mod[prefixed] === 'function' ||
        typeof mod[name] === 'function' ||
        (mod.wasmExports &&
          (typeof mod.wasmExports[name] === 'function' ||
            typeof mod.wasmExports[prefixed] === 'function')) ||
        (mod.asm &&
          (typeof mod.asm[name] === 'function' || typeof mod.asm[prefixed] === 'function')),
    );
  };
  const hasMain = hasFn('main');
  const hasCallMain = typeof mod.callMain === 'function';
  const fastBootExports = [];
  for (let i = 0; i < FAST_BOOT_EXPORT_NAMES.length; i += 1) {
    const exportName = FAST_BOOT_EXPORT_NAMES[i];
    if (hasFn(exportName)) {
      fastBootExports.push(exportName);
    }
  }
  return {
    hasMain,
    hasCallMain,
    fastBootExports,
    mode: hasMain ? 'main' : 'fast-boot',
  };
}

function readWasmGlobalOrInt(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  let inner = raw;
  let isGlobal = false;
  if (typeof inner === 'object' && 'value' in inner) {
    isGlobal = true;
    try {
      inner = inner.value;
    } catch {
      return null;
    }
  }
  if (typeof inner === 'bigint') {
    if (inner < 0n || inner > BigInt(Number.MAX_SAFE_INTEGER)) {
      return null;
    }
    return { value: Number(inner), isBigInt: true, isGlobal };
  }
  if (typeof inner === 'number' && Number.isFinite(inner) && inner >= 0) {
    return { value: Math.floor(inner), isBigInt: false, isGlobal };
  }
  return null;
}

/**
 * Audit `__indirect_function_table`, `__table_base`, and `GOT.func` imports cleanly across
 * asymmetric (384+ `GOT.func` dispatchers, table size >= 5103) and single-side-module collapsed
 * (0 `GOT.func`) builds, without mutating Emscripten lazy-stub `GOT.func` Proxies (#663).
 */
function auditIndirectTableAndGotFunc(imports, manifestMetrics = null) {
  const env = imports && typeof imports === 'object' && imports.env && typeof imports.env === 'object'
    ? imports.env
    : null;
  const table = env
    ? env.__indirect_function_table || env.table || null
    : imports && typeof imports === 'object' && typeof imports.grow === 'function'
      ? imports
      : null;
  const rawTableLen = table && table.length !== undefined ? table.length : null;
  const tableLenInfo = readWasmGlobalOrInt(rawTableLen);
  const tableBaseInfo = env ? readWasmGlobalOrInt(env.__table_base) : null;

  let gotFuncOwnCount = 0;
  const gotNs =
    imports && typeof imports === 'object' && imports['GOT.func'] && typeof imports['GOT.func'] === 'object'
      ? imports['GOT.func']
      : imports &&
          typeof imports === 'object' &&
          imports.GOT &&
          typeof imports.GOT === 'object' &&
          imports.GOT.func &&
          typeof imports.GOT.func === 'object'
        ? imports.GOT.func
        : null;
  if (gotNs) {
    try {
      gotFuncOwnCount = Object.keys(gotNs).length;
    } catch {
      gotFuncOwnCount = 0;
    }
  }

  const metrics =
    manifestMetrics && typeof manifestMetrics === 'object' ? manifestMetrics : null;
  const manifestTableSize = metrics
    ? toOptionalNonNegativeInt(
        pickFirstDefined(metrics, ['totalTableSize', 'total_table_size', 'tableSize', 'table_size']),
      )
    : null;
  const manifestGotFuncCount = metrics
    ? toOptionalNonNegativeInt(
        pickFirstDefined(metrics, [
          'totalGotFuncImports',
          'total_got_func_imports',
          'gotFuncImports',
          'got_func_imports',
        ]),
      )
    : null;

  const gotFuncCount =
    manifestGotFuncCount !== null ? Math.max(manifestGotFuncCount, gotFuncOwnCount) : gotFuncOwnCount;
  const currentTableLength = tableLenInfo ? tableLenInfo.value : 0;
  const tableBase = tableBaseInfo ? tableBaseInfo.value : 0;
  const isTable64 = Boolean(
    (tableLenInfo && tableLenInfo.isBigInt) || (tableBaseInfo && tableBaseInfo.isBigInt),
  );
  const requiredTableSize = Math.max(
    manifestTableSize !== null ? manifestTableSize : 0,
    tableBase + gotFuncCount,
  );

  return {
    hasTable: Boolean(table && typeof table.grow === 'function'),
    currentTableLength,
    tableBase,
    gotFuncCount,
    manifestTableSize,
    manifestGotFuncCount,
    requiredTableSize,
    needsGrow: Boolean(table && typeof table.grow === 'function' && currentTableLength < requiredTableSize),
    isTable64,
  };
}

/**
 * Ensure `env.__indirect_function_table` has at least `requiredSlots` (with NO hardcoded cap)
 * so large tables (>= 5103+) and high `GOT.func` dispatcher counts (384+) or collapsed single
 * modules (0 `GOT.func`) never fail dynamic linking due to table exhaustion (#663).
 * Safely handles both Wasm32 (`number`) and Wasm64 (`BigInt` / `WebAssembly.Global`) indices.
 */
function ensureIndirectFunctionTableCapacity(importsOrTable, requiredSlots = 0, options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  const isDirectTable = Boolean(
    importsOrTable &&
      typeof importsOrTable === 'object' &&
      typeof importsOrTable.grow === 'function' &&
      !importsOrTable.env,
  );
  const env =
    !isDirectTable &&
    importsOrTable &&
    typeof importsOrTable === 'object' &&
    importsOrTable.env &&
    typeof importsOrTable.env === 'object'
      ? importsOrTable.env
      : null;
  const table = isDirectTable
    ? importsOrTable
    : env
      ? env.__indirect_function_table || env.table || null
      : null;

  const audit = auditIndirectTableAndGotFunc(
    isDirectTable ? { env: { __indirect_function_table: table } } : importsOrTable,
    safeOpts.manifestMetrics || safeOpts.wasmMetrics || null,
  );

  const explicitSlots = toNonNegativeInt(requiredSlots, 0);
  const tableBaseOverride =
    safeOpts.tableBase !== undefined ? readWasmGlobalOrInt(safeOpts.tableBase) : null;
  const effectiveTableBase = tableBaseOverride ? tableBaseOverride.value : audit.tableBase;
  const gotFuncExtra = toNonNegativeInt(
    safeOpts.gotFuncCount !== undefined ? safeOpts.gotFuncCount : audit.gotFuncCount,
    0,
  );
  const headroom = toNonNegativeInt(safeOpts.headroom, 0);

  const targetLength = Math.max(
    audit.currentTableLength,
    audit.requiredTableSize,
    explicitSlots,
    effectiveTableBase + gotFuncExtra + headroom,
  );

  if (!table || typeof table.grow !== 'function') {
    return {
      grew: false,
      previousLength: audit.currentTableLength,
      currentLength: audit.currentTableLength,
      targetLength,
      tableBase: effectiveTableBase,
      gotFuncCount: gotFuncExtra,
      isTable64: audit.isTable64,
      error: null,
    };
  }

  const previousLength = audit.currentTableLength;
  if (previousLength >= targetLength) {
    return {
      grew: false,
      previousLength,
      currentLength: previousLength,
      targetLength,
      tableBase: effectiveTableBase,
      gotFuncCount: gotFuncExtra,
      isTable64: audit.isTable64,
      error: null,
    };
  }

  const delta = targetLength - previousLength;
  const growArg = typeof table.length === 'bigint' ? BigInt(delta) : delta;
  try {
    table.grow(growArg);
  } catch (err) {
    return {
      grew: false,
      previousLength,
      currentLength: readWasmGlobalOrInt(table.length)?.value ?? previousLength,
      targetLength,
      tableBase: effectiveTableBase,
      gotFuncCount: gotFuncExtra,
      isTable64: audit.isTable64,
      error: err instanceof Error ? err.message : String(err),
    };
  }

  const afterLen = readWasmGlobalOrInt(table.length)?.value ?? targetLength;
  return {
    grew: afterLen > previousLength,
    previousLength,
    currentLength: afterLen,
    targetLength,
    tableBase: effectiveTableBase,
    gotFuncCount: gotFuncExtra,
    isTable64: audit.isTable64,
    error: null,
  };
}

async function loadGuestManifest(options = {}) {
  const url =
    typeof options.url === 'string' && options.url.trim().length > 0
      ? options.url.trim()
      : DEFAULT_GUEST_MANIFEST_RELATIVE;
  const fetchFn =
    typeof options.fetchFn === 'function'
      ? options.fetchFn
      : typeof fetch === 'function'
        ? fetch
        : null;
  const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
  const reportFn = typeof options.reportFn === 'function' ? options.reportFn : null;

  if (!fetchFn) {
    return null;
  }

  let parsed = null;
  try {
    const resp = await fetchFn(url, { credentials: 'same-origin' });
    if (!resp || resp.ok === false) {
      return null;
    }
    const raw =
      typeof resp.json === 'function'
        ? await resp.json()
        : typeof resp.text === 'function'
          ? await resp.text()
          : null;
    parsed = parseGuestManifest(raw);
    if (parsed && raw) {
      const budget = evaluateV8SideModuleBudget(raw);
      if (budget.largestSideModuleBytes > 0) {
        recordSideModuleByteSize(
          budget.largestSideModuleName || 'guest_part_0.wasm',
          budget.largestSideModuleBytes,
          reportFn,
        );
      }
    }
  } catch {
    return null;
  }

  if (!parsed) {
    return null;
  }

  if (onProgress) {
    try {
      onProgress({
        phase: 'manifest',
        indirectDispatch: parsed.indirectDispatch,
        sideModules: parsed.sideModules.slice(),
        sideModuleCount: parsed.sideModuleCount,
        wasmMetrics: parsed.wasmMetrics ? Object.assign({}, parsed.wasmMetrics) : null,
      });
    } catch {
      // Progress hooks must never fail manifest loading.
    }
  }
  if (reportFn) {
    try {
      const tableSuffix =
        parsed.wasmMetrics && parsed.wasmMetrics.totalTableSize !== undefined
          ? `, table=${parsed.wasmMetrics.totalTableSize}`
          : '';
      reportFn(
        `Guest manifest loaded (dispatch=${parsed.indirectDispatch}, sideModules=${parsed.sideModuleCount}${tableSuffix})`,
      );
    } catch {
      // Progress reporting failures are non-fatal.
    }
  }

  return parsed;
}

function createRuntimeLocateFile(runtimeScriptUrl, locateFileBase, options = {}) {
  const reportFn = typeof options.reportFn === 'function' ? options.reportFn : null;
  const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
  const indirectDispatch =
    options.indirectDispatch !== undefined
      ? normalizeIndirectDispatchMode(options.indirectDispatch)
      : null;
  const wasmMetrics =
    options.wasmMetrics && typeof options.wasmMetrics === 'object'
      ? Object.assign({}, options.wasmMetrics)
      : null;
  const knownSideModules = new Set();
  const expectedModuleNames = [];
  if (Array.isArray(options.sideModuleNames)) {
    for (let i = 0; i < options.sideModuleNames.length; i += 1) {
      const base = extractWasmBasename(options.sideModuleNames[i]);
      if (base) {
        const lower = base.toLowerCase();
        if (!knownSideModules.has(lower)) {
          knownSideModules.add(lower);
          expectedModuleNames.push(base);
        }
      }
    }
  }
  const explicitExpectedCount = toOptionalNonNegativeInt(options.expectedSideModuleCount);
  const reportedSideModules = new Set();
  const loadedModuleNames = [];

  function getProgress() {
    const loadedCount = reportedSideModules.size;
    const baseExpected = Math.max(
      knownSideModules.size,
      explicitExpectedCount !== null ? explicitExpectedCount : 0,
    );
    const totalCount = Math.max(baseExpected, loadedCount);
    const remainingCount = Math.max(0, totalCount - loadedCount);
    const complete = totalCount === 0 || loadedCount >= totalCount;
    const pct = totalCount > 0 ? Math.round((loadedCount / totalCount) * 10000) / 100 : 100;
    return {
      loadedCount,
      totalCount,
      remainingCount,
      complete,
      pct,
      loadedModules: loadedModuleNames.slice(),
      expectedModules: expectedModuleNames.slice(),
    };
  }

  function locateFile(file) {
    const resolved = resolveSideModuleUrl(file, runtimeScriptUrl, locateFileBase);
    if (!resolved) {
      return '';
    }
    const base = extractWasmBasename(resolved) || extractWasmBasename(file);
    const baseLower = base ? base.toLowerCase() : '';
    if (
      baseLower &&
      (GUEST_SIDE_MODULE_BASENAME_RE.test(baseLower) || knownSideModules.has(baseLower)) &&
      !reportedSideModules.has(baseLower)
    ) {
      reportedSideModules.add(baseLower);
      loadedModuleNames.push(base);
      const snapshot = getProgress();
      if (reportFn) {
        try {
          reportFn(`Loading Wasm side module (${base})...`);
        } catch {
          // Progress channel errors must never break locateFile.
        }
      }
      if (onProgress) {
        try {
          const payload = {
            phase: 'side-module',
            module: base,
            url: resolved,
            loadedCount: snapshot.loadedCount,
            totalCount: snapshot.totalCount,
            remainingCount: snapshot.remainingCount,
            complete: snapshot.complete,
            pct: snapshot.pct,
          };
          if (indirectDispatch) {
            payload.indirectDispatch = indirectDispatch;
          }
          if (wasmMetrics) {
            payload.wasmMetrics = Object.assign({}, wasmMetrics);
          }
          onProgress(payload);
        } catch {
          // Progress callback errors must never break locateFile.
        }
      }
    }
    return resolved;
  }

  locateFile.getProgress = getProgress;
  return locateFile;
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

const WASM_WIRE_BINARY_KEYS = Object.freeze([
  'wasmBinary',
  'wireBytes',
  'rawBytes',
  'bytes',
  'buffer',
  'arrayBuffer',
  'data',
  'wasmBytes',
  'wasmBuffer',
  'pgimBytes',
  'pgimBuffer',
  'shaderPackBytes',
  'shaderPackBuffer',
]);
const WASM_WIRE_COLLECTION_KEYS = Object.freeze([
  'preloadedBuffers',
  'guestModuleBytes',
  'sideModuleBytes',
  'sideModules',
  'guestModules',
  'wasmModules',
  'modules',
  'modulesByName',
  'sidecars',
  'bundle',
  'program',
  'runtimeWasm',
  'pgim',
]);
const WASM_KEY_HINT_RE = /(?:wasm|guest_?part|guestmodule|sidemodule|wirebytes)/i;

const wasmMemoryTelemetry = {
  wireBytesReleased: 0,
  droppedWasmBuffers: 0,
  droppedPreloadBuffers: 0,
  transferredArrayBuffers: 0,
  streamingCompiles: 0,
  fallbackCompiles: 0,
  largestSideModuleBytes: 0,
  largestSideModuleName: null,
  sideModuleCapWarnings: 0,
  peakWireBytesInFlight: 0,
  currentWireBytesInFlight: 0,
  sharedModulesDeliveredToPthreads: 0,
};

function recordSideModuleByteSize(moduleName, byteLength, reportFn) {
  const bytes = toNonNegativeInt(byteLength, 0);
  if (bytes <= 0) {
    return null;
  }
  const cleanName = extractWasmBasename(String(moduleName || '')) || 'side_module.wasm';
  if (bytes > wasmMemoryTelemetry.largestSideModuleBytes) {
    wasmMemoryTelemetry.largestSideModuleBytes = bytes;
    wasmMemoryTelemetry.largestSideModuleName = cleanName;
  }
  const warnThreshold = Math.floor(V8_SIDE_MODULE_CAP_BYTES * V8_SIDE_MODULE_WARN_RATIO);
  if (bytes >= warnThreshold) {
    wasmMemoryTelemetry.sideModuleCapWarnings += 1;
    const mib = (bytes / (1024 * 1024)).toFixed(1);
    const headroomMib = (Math.max(0, V8_SIDE_MODULE_CAP_BYTES - bytes) / (1024 * 1024)).toFixed(1);
    const msg =
      bytes > V8_SIDE_MODULE_CAP_BYTES
        ? `[wasm] WARNING: side module ${cleanName} (${mib} MiB) exceeds V8 1024 MiB cap`
        : `[wasm] WARNING: side module ${cleanName} (${mib} MiB) has only ${headroomMib} MiB headroom below V8 1024 MiB cap`;
    if (typeof reportFn === 'function') {
      try {
        reportFn(msg);
      } catch {
        // Non-fatal warning hook failure.
      }
    }
    return msg;
  }
  return null;
}

function trackWireBytesInFlightDelta(delta) {
  wasmMemoryTelemetry.currentWireBytesInFlight = Math.max(
    0,
    wasmMemoryTelemetry.currentWireBytesInFlight + delta,
  );
  if (wasmMemoryTelemetry.currentWireBytesInFlight > wasmMemoryTelemetry.peakWireBytesInFlight) {
    wasmMemoryTelemetry.peakWireBytesInFlight = wasmMemoryTelemetry.currentWireBytesInFlight;
  }
}

let activeSharedWasmModuleCount = 0;

function isCompiledWasmModule(value, wasmNs) {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const ModuleCtor =
    (wasmNs && typeof wasmNs.Module === 'function' ? wasmNs.Module : null) ||
    (typeof WebAssembly !== 'undefined' && typeof WebAssembly.Module === 'function'
      ? WebAssembly.Module
      : null);
  return Boolean(ModuleCtor && value instanceof ModuleCtor);
}

function getBinaryWireInfo(value, wasmNs) {
  if (!value || typeof value !== 'object' || isCompiledWasmModule(value, wasmNs)) {
    return null;
  }
  if (isTransferableArrayBuffer(value)) {
    return { backingBuffer: value, byteLength: value.byteLength, isShared: false };
  }
  if (isSharedArrayBuffer(value)) {
    return { backingBuffer: value, byteLength: value.byteLength, isShared: true };
  }
  if (ArrayBuffer.isView(value)) {
    return {
      backingBuffer: value.buffer,
      byteLength: value.byteLength,
      isShared: isSharedArrayBuffer(value.buffer),
    };
  }
  return null;
}

function recordDroppedWasmWireBytes(byteLength, count = 1) {
  const bytes = toNonNegativeInt(byteLength, 0);
  const buffers = toNonNegativeInt(count, 0);
  if (bytes > 0) {
    wasmMemoryTelemetry.wireBytesReleased += bytes;
  }
  if (buffers > 0) {
    wasmMemoryTelemetry.droppedWasmBuffers += buffers;
  }
}

function recordDroppedPreloadBytes(byteLength, count = 1) {
  const bytes = toNonNegativeInt(byteLength, 0);
  const buffers = toNonNegativeInt(count, 0);
  if (bytes > 0) {
    wasmMemoryTelemetry.wireBytesReleased += bytes;
  }
  if (buffers > 0) {
    wasmMemoryTelemetry.droppedPreloadBuffers += buffers;
  }
}

function recordTransferredArrayBuffers(count = 1) {
  const n = toNonNegativeInt(count, 0);
  if (n > 0) {
    wasmMemoryTelemetry.transferredArrayBuffers += n;
  }
}

function getWasmMemoryTelemetry() {
  const largest = wasmMemoryTelemetry.largestSideModuleBytes;
  return {
    wireBytesReleased: wasmMemoryTelemetry.wireBytesReleased,
    droppedWasmBuffers: wasmMemoryTelemetry.droppedWasmBuffers,
    droppedPreloadBuffers: wasmMemoryTelemetry.droppedPreloadBuffers,
    transferredArrayBuffers: wasmMemoryTelemetry.transferredArrayBuffers,
    streamingCompiles: wasmMemoryTelemetry.streamingCompiles,
    fallbackCompiles: wasmMemoryTelemetry.fallbackCompiles,
    sharedModuleCount: activeSharedWasmModuleCount,
    v8SideModuleCapBytes: V8_SIDE_MODULE_CAP_BYTES,
    largestSideModuleBytes: largest,
    largestSideModuleName: wasmMemoryTelemetry.largestSideModuleName,
    sideModuleHeadroomBytes: Math.max(0, V8_SIDE_MODULE_CAP_BYTES - largest),
    sideModuleCapWarnings: wasmMemoryTelemetry.sideModuleCapWarnings,
    peakWireBytesInFlight: wasmMemoryTelemetry.peakWireBytesInFlight,
    sharedModulesDeliveredToPthreads: wasmMemoryTelemetry.sharedModulesDeliveredToPthreads,
  };
}

function resetWasmMemoryTelemetry() {
  wasmMemoryTelemetry.wireBytesReleased = 0;
  wasmMemoryTelemetry.droppedWasmBuffers = 0;
  wasmMemoryTelemetry.droppedPreloadBuffers = 0;
  wasmMemoryTelemetry.transferredArrayBuffers = 0;
  wasmMemoryTelemetry.streamingCompiles = 0;
  wasmMemoryTelemetry.fallbackCompiles = 0;
  wasmMemoryTelemetry.largestSideModuleBytes = 0;
  wasmMemoryTelemetry.largestSideModuleName = null;
  wasmMemoryTelemetry.sideModuleCapWarnings = 0;
  wasmMemoryTelemetry.peakWireBytesInFlight = 0;
  wasmMemoryTelemetry.currentWireBytesInFlight = 0;
  wasmMemoryTelemetry.sharedModulesDeliveredToPthreads = 0;
  return getWasmMemoryTelemetry();
}

function isWritableOwnProp(obj, key) {
  if (!obj || typeof obj !== 'object') {
    return false;
  }
  if (!Object.prototype.hasOwnProperty.call(obj, key)) {
    return false;
  }
  const desc = Object.getOwnPropertyDescriptor(obj, key);
  if (!desc) {
    return false;
  }
  if ('get' in desc || 'set' in desc) {
    return typeof desc.set === 'function';
  }
  return desc.writable === true;
}

/**
 * Collect unique standalone non-SAB `ArrayBuffer`s reachable inside `payload` for
 * zero-copy `postMessage(msg, transferList)`. Skips `SharedArrayBuffer`, `WebAssembly.Module`,
 * detached buffers, and subviews that do not cover their entire backing buffer.
 */
function collectTransferableArrayBuffers(payload, wasmNs) {
  const unique = new Set();
  const visited = new Set();

  const visit = (value) => {
    if (!value || typeof value !== 'object' || isCompiledWasmModule(value, wasmNs)) {
      return;
    }
    if (isTransferableArrayBuffer(value)) {
      if (!value.detached && value.byteLength > 0) {
        unique.add(value);
      }
      return;
    }
    if (isSharedArrayBuffer(value)) {
      return;
    }
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
    if (visited.has(value)) {
      return;
    }
    visited.add(value);
    if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i += 1) {
        visit(value[i]);
      }
      return;
    }
    if (value instanceof Map || value instanceof Set) {
      value.forEach((v) => visit(v));
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

/**
 * Immediately drop raw `.wasm` / preload `ArrayBuffer` & `Uint8Array` references from a
 * message, map, array, or bundle (`= null` / `Map.delete` / `array.length = 0`) while
 * preserving compiled `WebAssembly.Module` objects and lightweight metadata.
 */
function releaseWireBytesFromObject(target, options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  const recordTelemetry = safeOpts.recordTelemetry !== false;
  const assumeWasm = safeOpts.assumeWasm === true;
  const wasmNs = safeOpts.wasmApi || (typeof WebAssembly !== 'undefined' ? WebAssembly : null);
  const releasedFields = [];
  let releasedBytes = 0;
  let droppedWasmCount = 0;
  let droppedPreloadCount = 0;

  if (!target || typeof target !== 'object' || isCompiledWasmModule(target, wasmNs)) {
    return { releasedFields, releasedBytes, droppedWasmBuffers: 0, droppedPreloadBuffers: 0 };
  }

  const countedByBuffer = new Map();
  const visited = new Set();

  const countBuffer = (info, path) => {
    if (!info || !info.backingBuffer) {
      return;
    }
    const isWasm = assumeWasm || WASM_KEY_HINT_RE.test(path);
    const len = info.byteLength > 0 ? info.byteLength : 0;
    const prev = countedByBuffer.get(info.backingBuffer);
    if (prev === undefined) {
      countedByBuffer.set(info.backingBuffer, len);
      releasedBytes += len;
      if (isWasm) {
        droppedWasmCount += 1;
      } else {
        droppedPreloadCount += 1;
      }
    } else if (len > prev) {
      releasedBytes += len - prev;
      countedByBuffer.set(info.backingBuffer, len);
    }
  };

  const nullBinaryProp = (obj, key, path) => {
    if (!isWritableOwnProp(obj, key)) {
      return false;
    }
    let val;
    try {
      val = obj[key];
    } catch {
      return false;
    }
    const info = getBinaryWireInfo(val, wasmNs);
    if (!info) {
      return false;
    }
    try {
      obj[key] = null;
    } catch {
      return false;
    }
    releasedFields.push(path);
    countBuffer(info, path);
    return true;
  };

  const walk = (node, prefix) => {
    if (!node || typeof node !== 'object' || isCompiledWasmModule(node, wasmNs) || visited.has(node)) {
      return;
    }
    visited.add(node);

    if (node instanceof Map) {
      const entries = Array.from(node.entries());
      for (let i = 0; i < entries.length; i += 1) {
        const [k, v] = entries[i];
        const label = typeof k === 'string' || typeof k === 'number' ? String(k) : 'entry';
        const path = prefix ? `${prefix}.${label}` : label;
        const info = getBinaryWireInfo(v, wasmNs);
        if (info) {
          node.delete(k);
          releasedFields.push(path);
          countBuffer(info, path);
        } else if (v && typeof v === 'object' && !isCompiledWasmModule(v, wasmNs)) {
          walk(v, path);
        }
      }
      return;
    }

    if (Array.isArray(node)) {
      let allDirectBinary = node.length > 0;
      for (let i = 0; i < node.length; i += 1) {
        const item = node[i];
        const path = prefix ? `${prefix}[${i}]` : `[${i}]`;
        const info = getBinaryWireInfo(item, wasmNs);
        if (info) {
          if (isWritableOwnProp(node, String(i))) {
            node[i] = null;
            releasedFields.push(path);
            countBuffer(info, path);
          }
        } else {
          allDirectBinary = false;
          if (item && typeof item === 'object' && !isCompiledWasmModule(item, wasmNs)) {
            walk(item, path);
          }
        }
      }
      if (allDirectBinary && !Object.isFrozen(node)) {
        try {
          node.length = 0;
        } catch {
          // Ignore non-resizable arrays.
        }
      }
      return;
    }

    for (let i = 0; i < WASM_WIRE_BINARY_KEYS.length; i += 1) {
      const k = WASM_WIRE_BINARY_KEYS[i];
      if (Object.prototype.hasOwnProperty.call(node, k)) {
        const path = prefix ? `${prefix}.${k}` : k;
        if (!nullBinaryProp(node, k, path)) {
          const child = node[k];
          if (child && typeof child === 'object' && !isCompiledWasmModule(child, wasmNs)) {
            walk(child, path);
          }
        }
      }
    }

    for (let i = 0; i < WASM_WIRE_COLLECTION_KEYS.length; i += 1) {
      const k = WASM_WIRE_COLLECTION_KEYS[i];
      if (!Object.prototype.hasOwnProperty.call(node, k)) {
        continue;
      }
      let child;
      try {
        child = node[k];
      } catch {
        continue;
      }
      if (!child || typeof child !== 'object' || isCompiledWasmModule(child, wasmNs)) {
        continue;
      }
      const path = prefix ? `${prefix}.${k}` : k;
      if (getBinaryWireInfo(child, wasmNs)) {
        nullBinaryProp(node, k, path);
      } else {
        walk(child, path);
      }
    }

    // For plain maps (e.g. `preloadedBuffers` / `guestModuleBytes`), also drop direct binary keys.
    if (
      prefix &&
      /(?:preloadedBuffers|guestModuleBytes|sideModuleBytes|sidecars|modulesByName)$/i.test(prefix)
    ) {
      const keys = Object.keys(node);
      for (let i = 0; i < keys.length; i += 1) {
        const k = keys[i];
        const path = `${prefix}.${k}`;
        if (!nullBinaryProp(node, k, path)) {
          const child = node[k];
          if (child && typeof child === 'object' && !isCompiledWasmModule(child, wasmNs)) {
            walk(child, path);
          }
        }
      }
    }
  };

  walk(target, '');

  if (recordTelemetry && (releasedBytes > 0 || droppedWasmCount > 0 || droppedPreloadCount > 0)) {
    if (droppedWasmCount > 0) {
      recordDroppedWasmWireBytes(releasedBytes, droppedWasmCount);
      if (droppedPreloadCount > 0) {
        wasmMemoryTelemetry.droppedPreloadBuffers += droppedPreloadCount;
      }
    } else if (droppedPreloadCount > 0) {
      recordDroppedPreloadBytes(releasedBytes, droppedPreloadCount);
    } else if (releasedBytes > 0) {
      wasmMemoryTelemetry.wireBytesReleased += releasedBytes;
    }
  }

  return {
    releasedFields,
    releasedBytes,
    droppedWasmBuffers: droppedWasmCount,
    droppedPreloadBuffers: droppedPreloadCount,
  };
}

/**
 * Strip any raw wire-byte `ArrayBuffer` / `Uint8Array` payloads from an Emscripten
 * `sharedModules` list, map, or object so compiled `WebAssembly.Module` instances are
 * shared across pthreads WITHOUT ever keeping or cloning raw `.wasm` wire bytes.
 */
function stripWireBytesFromSharedModules(sharedModules, options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  const wasmNs = safeOpts.wasmApi || (typeof WebAssembly !== 'undefined' ? WebAssembly : null);
  if (!sharedModules || typeof sharedModules !== 'object') {
    return { releasedFields: [], releasedBytes: 0, droppedWasmBuffers: 0, droppedPreloadBuffers: 0 };
  }
  const result = releaseWireBytesFromObject(sharedModules, {
    wasmApi: wasmNs,
    assumeWasm: true,
    recordTelemetry: safeOpts.recordTelemetry !== false,
  });
  return result;
}

/**
 * Compile a WebAssembly module preferring `WebAssembly.compileStreaming` so large
 * `guest_part_*.wasm` binaries never materialize as 1 GB JS `ArrayBuffer`s.
 *
 * Falls back cleanly to `response.arrayBuffer()` + `WebAssembly.compile(bytes)` and
 * immediately drops all raw wire-byte references once compilation settles.
 */
async function compileWasmModuleStreaming(source, options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  const wasmNs =
    safeOpts.wasmApi && typeof safeOpts.wasmApi === 'object'
      ? safeOpts.wasmApi
      : typeof WebAssembly !== 'undefined'
        ? WebAssembly
        : null;
  if (!wasmNs || typeof wasmNs.compile !== 'function') {
    throw new Error('WebAssembly.compile is unavailable');
  }
  const fetchFn =
    typeof safeOpts.fetchFn === 'function'
      ? safeOpts.fetchFn
      : typeof fetch === 'function'
        ? fetch
        : null;

  if (isCompiledWasmModule(source, wasmNs)) {
    return { module: source, streaming: false, wireBytesReleased: 0 };
  }

  const releaseOwnerSlot = () => {
    if (!safeOpts.owner || safeOpts.ownerKey === undefined) {
      return;
    }
    try {
      if (safeOpts.owner instanceof Map) {
        safeOpts.owner.delete(safeOpts.ownerKey);
      } else if (isWritableOwnProp(safeOpts.owner, safeOpts.ownerKey)) {
        safeOpts.owner[safeOpts.ownerKey] = null;
      }
    } catch {
      // Ignore release errors on frozen/exotic containers.
    }
  };

  const directBinary = getBinaryWireInfo(source, wasmNs);
  if (directBinary) {
    let localBytes = source;
    const byteLength = directBinary.byteLength;
    const moduleNameHint =
      typeof safeOpts.moduleName === 'string' && safeOpts.moduleName
        ? safeOpts.moduleName
        : typeof safeOpts.ownerKey === 'string'
          ? safeOpts.ownerKey
          : '';
    releaseOwnerSlot();
    if (byteLength > 0) {
      trackWireBytesInFlightDelta(byteLength);
      if (moduleNameHint) {
        recordSideModuleByteSize(moduleNameHint, byteLength, safeOpts.reportFn);
      }
    }
    let mod;
    try {
      const compilePromise = wasmNs.compile(localBytes);
      localBytes = null;
      mod = await compilePromise;
    } finally {
      localBytes = null;
      releaseOwnerSlot();
      if (byteLength > 0) {
        trackWireBytesInFlightDelta(-byteLength);
        recordDroppedWasmWireBytes(byteLength, 1);
      }
    }
    wasmMemoryTelemetry.fallbackCompiles += 1;
    return { module: mod, streaming: false, wireBytesReleased: byteLength, moduleByteLength: byteLength };
  }

  let resp = source;
  const sourceUrlHint =
    typeof source === 'string'
      ? source
      : typeof safeOpts.moduleName === 'string'
        ? safeOpts.moduleName
        : '';
  if (typeof source === 'string') {
    if (!fetchFn) {
      throw new Error(`fetch unavailable while compiling ${source}`);
    }
    const fetchInit = { credentials: 'same-origin' };
    if (safeOpts.signal) {
      fetchInit.signal = safeOpts.signal;
    }
    resp = await fetchFn(source, fetchInit);
  } else if (resp && typeof resp.then === 'function') {
    resp = await resp;
  }

  if (!resp || resp.ok === false) {
    const status = resp && resp.status !== undefined ? resp.status : 'unknown';
    throw new Error(`Failed to fetch Wasm module (HTTP ${status})`);
  }

  const contentType =
    resp.headers && typeof resp.headers.get === 'function'
      ? String(resp.headers.get('content-type') || '')
      : '';
  const contentLengthHeader =
    resp.headers && typeof resp.headers.get === 'function'
      ? resp.headers.get('content-length')
      : null;
  const declaredByteLength = toOptionalNonNegativeInt(contentLengthHeader);
  const effectiveModuleName =
    (typeof safeOpts.moduleName === 'string' && safeOpts.moduleName) ||
    (resp && typeof resp.url === 'string' && resp.url) ||
    sourceUrlHint;
  const isRealResponse = typeof Response === 'undefined' || resp instanceof Response;
  const canStream =
    typeof wasmNs.compileStreaming === 'function' &&
    (isRealResponse || safeOpts.allowStreamingWithoutContentType === true) &&
    (contentType.toLowerCase().includes('application/wasm') ||
      safeOpts.allowStreamingWithoutContentType === true);

  if (canStream) {
    if (declaredByteLength !== null && declaredByteLength > 0 && effectiveModuleName) {
      recordSideModuleByteSize(effectiveModuleName, declaredByteLength, safeOpts.reportFn);
    }
    try {
      const mod = await wasmNs.compileStreaming(Promise.resolve(resp));
      wasmMemoryTelemetry.streamingCompiles += 1;
      return {
        module: mod,
        streaming: true,
        wireBytesReleased: 0,
        moduleByteLength: declaredByteLength !== null ? declaredByteLength : 0,
      };
    } catch (streamErr) {
      if (resp.bodyUsed) {
        throw streamErr;
      }
      // Fall back to arrayBuffer() + compile() when streaming was rejected before consuming body.
    }
  }

  let wireBytes = await resp.arrayBuffer();
  const byteLength = wireBytes && typeof wireBytes.byteLength === 'number' ? wireBytes.byteLength : 0;
  if (byteLength > 0) {
    trackWireBytesInFlightDelta(byteLength);
    if (effectiveModuleName) {
      recordSideModuleByteSize(effectiveModuleName, byteLength, safeOpts.reportFn);
    }
  }
  let mod;
  try {
    const compilePromise = wasmNs.compile(wireBytes);
    wireBytes = null;
    mod = await compilePromise;
  } finally {
    wireBytes = null;
    if (byteLength > 0) {
      trackWireBytesInFlightDelta(-byteLength);
      recordDroppedWasmWireBytes(byteLength, 1);
    }
  }
  wasmMemoryTelemetry.fallbackCompiles += 1;
  return { module: mod, streaming: false, wireBytesReleased: byteLength, moduleByteLength: byteLength };
}

/**
 * Shared compiled `WebAssembly.Module` cache for Emscripten side modules (`guest_part_*.wasm`).
 * Compiles each module at most once, deduplicates concurrent requests, shares the compiled
 * `WebAssembly.Module` across all `em-pthread` workers, and NEVER retains raw wire bytes.
 */
function createSharedWasmModuleCache(options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  const modulesByKey = new Map();
  const inFlightByKey = new Map();
  const moduleSizesByKey = new Map();
  const compileCountsByKey = new Map();
  const trackGlobalCount = safeOpts.trackGlobalCount === true;
  let totalCompileCount = 0;
  let cacheHits = 0;

  const normalizeKey = (rawKey) => {
    if (typeof rawKey !== 'string') {
      return '';
    }
    const base = extractWasmBasename(rawKey) || rawKey.trim();
    return base.toLowerCase();
  };

  const syncGlobalCount = () => {
    if (trackGlobalCount) {
      activeSharedWasmModuleCount = modulesByKey.size;
    }
  };

  const releaseDuplicateCallerWireBytes = (source, callOpts) => {
    if (!callOpts || !callOpts.owner || callOpts.ownerKey === undefined) {
      return;
    }
    const info = getBinaryWireInfo(source, callOpts.wasmApi || safeOpts.wasmApi);
    if (callOpts.owner instanceof Map) {
      callOpts.owner.delete(callOpts.ownerKey);
    } else if (isWritableOwnProp(callOpts.owner, callOpts.ownerKey)) {
      callOpts.owner[callOpts.ownerKey] = null;
    }
    if (info && info.byteLength > 0) {
      recordDroppedWasmWireBytes(info.byteLength, 1);
    }
  };

  function setModule(nameOrUrl, mod, byteLength = 0) {
    const key = normalizeKey(nameOrUrl);
    if (!key || !isCompiledWasmModule(mod, safeOpts.wasmApi)) {
      return false;
    }
    modulesByKey.set(key, mod);
    const bytes = toNonNegativeInt(byteLength, 0);
    if (bytes > 0) {
      moduleSizesByKey.set(key, bytes);
    }
    syncGlobalCount();
    return true;
  }

  function getModule(nameOrUrl) {
    const key = normalizeKey(nameOrUrl);
    return key ? modulesByKey.get(key) || null : null;
  }

  function hasModule(nameOrUrl) {
    const key = normalizeKey(nameOrUrl);
    return key ? modulesByKey.has(key) : false;
  }

  async function compileAndShare(nameOrUrl, source, callOpts = {}) {
    const key = normalizeKey(nameOrUrl);
    if (!key) {
      throw new TypeError('Shared Wasm module key must be a non-empty string');
    }
    const existing = modulesByKey.get(key);
    if (existing) {
      cacheHits += 1;
      // Even on cache hit, immediately release any raw wire bytes passed by caller.
      releaseDuplicateCallerWireBytes(source, callOpts);
      return existing;
    }
    const pending = inFlightByKey.get(key);
    if (pending) {
      cacheHits += 1;
      releaseDuplicateCallerWireBytes(source, callOpts);
      return pending;
    }
    totalCompileCount += 1;
    compileCountsByKey.set(key, (compileCountsByKey.get(key) || 0) + 1);
    const promise = compileWasmModuleStreaming(source, {
      wasmApi: callOpts.wasmApi || safeOpts.wasmApi,
      fetchFn: callOpts.fetchFn || safeOpts.fetchFn,
      signal: callOpts.signal,
      allowStreamingWithoutContentType: callOpts.allowStreamingWithoutContentType,
      reportFn: callOpts.reportFn || safeOpts.reportFn,
      moduleName: key,
      owner: callOpts.owner,
      ownerKey: callOpts.ownerKey,
    })
      .then((res) => {
        if (res && res.module) {
          modulesByKey.set(key, res.module);
          if (typeof res.moduleByteLength === 'number' && res.moduleByteLength > 0) {
            moduleSizesByKey.set(key, res.moduleByteLength);
          }
          syncGlobalCount();
          return res.module;
        }
        return null;
      })
      .finally(() => {
        inFlightByKey.delete(key);
      });
    inFlightByKey.set(key, promise);
    return promise;
  }

  async function precompileSideModules(sideModuleUrls, callOpts = {}) {
    if (!Array.isArray(sideModuleUrls) || sideModuleUrls.length === 0) {
      return [];
    }
    const results = new Array(sideModuleUrls.length);
    const orderedKeys = new Array(sideModuleUrls.length);
    await Promise.all(
      sideModuleUrls.map(async (entry, idx) => {
        const url = extractSideModuleEntryUrl(entry);
        if (!url) {
          results[idx] = null;
          orderedKeys[idx] = '';
          return;
        }
        orderedKeys[idx] = normalizeKey(url);
        const resolvedUrl =
          typeof callOpts.resolveUrl === 'function' ? callOpts.resolveUrl(url) : url;
        results[idx] = await compileAndShare(url, resolvedUrl, callOpts);
      }),
    );
    // Re-order modulesByKey entries to match deterministic manifest order even when tiny
    // asymmetric parts (e.g. 0.14 MiB guest_part_1) finish compiling before 890 MiB guest_part_0.
    for (let i = 0; i < orderedKeys.length; i += 1) {
      const k = orderedKeys[i];
      if (k && modulesByKey.has(k)) {
        const mod = modulesByKey.get(k);
        modulesByKey.delete(k);
        modulesByKey.set(k, mod);
      }
    }
    return results.filter((m) => isCompiledWasmModule(m, callOpts.wasmApi || safeOpts.wasmApi));
  }

  async function ingestPreloadedBuffers(container, callOpts = {}) {
    if (!container || typeof container !== 'object') {
      return [];
    }
    const compiled = [];
    const entries =
      container instanceof Map
        ? Array.from(container.entries())
        : Object.keys(container).map((k) => [k, container[k]]);
    for (let i = 0; i < entries.length; i += 1) {
      const [key, raw] = entries[i];
      if (!raw) {
        continue;
      }
      const wasmNs = callOpts.wasmApi || safeOpts.wasmApi;
      if (isCompiledWasmModule(raw, wasmNs)) {
        setModule(key, raw);
        compiled.push(raw);
        continue;
      }
      let wireCandidate = raw;
      if (!getBinaryWireInfo(raw, wasmNs) && typeof raw === 'object') {
        if (isCompiledWasmModule(raw.module, wasmNs)) {
          setModule(key, raw.module);
          compiled.push(raw.module);
          releaseWireBytesFromObject(raw, { wasmApi: wasmNs, assumeWasm: true });
          continue;
        }
        wireCandidate =
          raw.wasmBinary || raw.wireBytes || raw.bytes || raw.buffer || raw.arrayBuffer || null;
      }
      if (getBinaryWireInfo(wireCandidate, wasmNs)) {
        const mod = await compileAndShare(key, wireCandidate, {
          ...callOpts,
          owner: container,
          ownerKey: key,
        });
        if (mod) {
          compiled.push(mod);
        }
        if (raw && typeof raw === 'object' && raw !== wireCandidate) {
          releaseWireBytesFromObject(raw, { wasmApi: wasmNs, assumeWasm: true });
        }
      }
    }
    return compiled;
  }

  function getSharedModulesArray() {
    return Array.from(modulesByKey.values());
  }

  function getSharedModuleNames() {
    return Array.from(modulesByKey.keys());
  }

  function getSharedModulesEntries() {
    const out = [];
    modulesByKey.forEach((mod, name) => {
      out.push({
        name,
        module: mod,
        byteLength: moduleSizesByKey.get(name) || 0,
      });
    });
    return out;
  }

  function getSharedModulesMap() {
    return new Map(modulesByKey);
  }

  function getBudgetSummary(budgetOpts) {
    return evaluateV8SideModuleBudget(moduleSizesByKey, budgetOpts);
  }

  function clear() {
    modulesByKey.clear();
    inFlightByKey.clear();
    moduleSizesByKey.clear();
    compileCountsByKey.clear();
    syncGlobalCount();
  }

  return {
    setModule,
    getModule,
    hasModule,
    compileAndShare,
    precompileSideModules,
    ingestPreloadedBuffers,
    getSharedModulesArray,
    getSharedModuleNames,
    getSharedModulesEntries,
    getSharedModulesMap,
    getBudgetSummary,
    clear,
    get size() {
      return modulesByKey.size;
    },
    getStats() {
      const budget = evaluateV8SideModuleBudget(moduleSizesByKey);
      return {
        sharedModuleCount: modulesByKey.size,
        inFlightCompiles: inFlightByKey.size,
        compileCount: totalCompileCount,
        cacheHits,
        compileCountsByKey: Object.fromEntries(compileCountsByKey),
        v8SideModuleCapBytes: budget.v8SideModuleCapBytes,
        largestSideModuleBytes: budget.largestSideModuleBytes,
        sideModuleHeadroomBytes: budget.sideModuleHeadroomBytes,
      };
    },
  };
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
  const wasmNs =
    options.wasmApi && typeof options.wasmApi === 'object'
      ? options.wasmApi
      : typeof WebAssembly !== 'undefined'
        ? WebAssembly
        : null;
  const precompiledModule = isCompiledWasmModule(options.wasmBinary, wasmNs)
    ? options.wasmBinary
    : null;
  // Hold raw wire bytes in a mutable slot (never captured by `finish`) and clear immediately
  // upon handing to `WebAssembly.instantiate` so the closure retains 0 wire bytes.
  let pendingWireBytes = precompiledModule ? null : options.wasmBinary || null;
  const onError = typeof options.onError === 'function' ? options.onError : null;
  const onMemory = typeof options.onMemory === 'function' ? options.onMemory : null;
  const manifestWasmMetrics =
    options.wasmMetrics && typeof options.wasmMetrics === 'object' ? options.wasmMetrics : null;

  function releasePendingWireBytes() {
    if (!pendingWireBytes) {
      return 0;
    }
    const info = getBinaryWireInfo(pendingWireBytes, wasmNs);
    const len = info ? info.byteLength : 0;
    pendingWireBytes = null;
    if (len > 0) {
      recordDroppedWasmWireBytes(len, 1);
    }
    return len;
  }

  function instantiateWasm(imports, receiveInstance) {
    const importedMemory = extractMemoryFromImports(imports);
    if (importedMemory) {
      if (hostImports && typeof hostImports.setMemory === 'function') {
        hostImports.setMemory(importedMemory);
      }
      if (onMemory) {
        onMemory(importedMemory);
      }
    }

    // W371 (#663): When manifest metrics declare a large __indirect_function_table (e.g. >= 5103
    // with 384+ GOT.func dispatchers), grow an undersized imported table safely before linking.
    if (
      manifestWasmMetrics &&
      typeof manifestWasmMetrics.totalTableSize === 'number' &&
      manifestWasmMetrics.totalTableSize > 0 &&
      imports &&
      imports.env &&
      imports.env.__indirect_function_table
    ) {
      ensureIndirectFunctionTableCapacity(imports, manifestWasmMetrics.totalTableSize, {
        wasmMetrics: manifestWasmMetrics,
      });
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
          : precompiledModule || undefined;
      if (instance && instance.exports && instance.exports.memory) {
        if (hostImports && typeof hostImports.setMemory === 'function') {
          hostImports.setMemory(instance.exports.memory);
        }
        if (onMemory) {
          onMemory(instance.exports.memory);
        }
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
    if (precompiledModule) {
      instantiatePromise = wasmNs.instantiate(precompiledModule, mergedImports);
    } else if (pendingWireBytes) {
      let localWireBytes = pendingWireBytes;
      pendingWireBytes = null;
      const wireInfo = getBinaryWireInfo(localWireBytes, wasmNs);
      const wireLen = wireInfo ? wireInfo.byteLength : 0;
      const rawPromise = wasmNs.instantiate(localWireBytes, mergedImports);
      localWireBytes = null;
      instantiatePromise = rawPromise.finally(() => {
        if (wireLen > 0) {
          recordDroppedWasmWireBytes(wireLen, 1);
        }
        wasmMemoryTelemetry.fallbackCompiles += 1;
      });
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
          const isRealResponse = typeof Response === 'undefined' || resp instanceof Response;
          const isWasmMime = contentType.toLowerCase().includes('application/wasm');
          const canInstantiateStreaming =
            typeof wasmNs.instantiateStreaming === 'function' && isRealResponse && isWasmMime;
          if (canInstantiateStreaming) {
            wasmMemoryTelemetry.streamingCompiles += 1;
            return wasmNs.instantiateStreaming(Promise.resolve(resp), mergedImports);
          }
          const canCompileStreaming =
            typeof wasmNs.compileStreaming === 'function' && isRealResponse && isWasmMime;
          if (canCompileStreaming) {
            wasmMemoryTelemetry.streamingCompiles += 1;
            return wasmNs.compileStreaming(Promise.resolve(resp)).then((mod) =>
              wasmNs.instantiate(mod, mergedImports).then((instance) => ({ instance, module: mod })),
            );
          }
          return resp.arrayBuffer().then((bytes) => {
            let localBytes = bytes;
            const wireLen = localBytes && typeof localBytes.byteLength === 'number' ? localBytes.byteLength : 0;
            const p = wasmNs.instantiate(localBytes, mergedImports);
            localBytes = null;
            return p.finally(() => {
              if (wireLen > 0) {
                recordDroppedWasmWireBytes(wireLen, 1);
              }
              wasmMemoryTelemetry.fallbackCompiles += 1;
            });
          });
        });
    }

    instantiatePromise.then(finish, handleFailure);
    return {};
  }

  instantiateWasm.releasePendingWireBytes = releasePendingWireBytes;
  return instantiateWasm;
}

function isSharedArrayBufferValue(buf) {
  return typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer;
}

function isSupportedInputBuffer(buf) {
  if (!buf) {
    return false;
  }
  const validType = buf instanceof ArrayBuffer || isSharedArrayBufferValue(buf);
  return validType && buf.byteLength >= INPUT_STATE_BYTES;
}

function canUseAtomicsOnInt32View(atomics, view, isShared) {
  if (
    !atomics ||
    typeof atomics.load !== 'function' ||
    typeof atomics.store !== 'function' ||
    typeof atomics.add !== 'function'
  ) {
    return false;
  }
  if (isShared) {
    return true;
  }
  try {
    atomics.load(view, 0);
    return true;
  } catch {
    return false;
  }
}

const attachedAudioRingByWorker = new WeakMap();

function resolveAudioRingPointerGetter(mod, explicitGetter) {
  if (typeof explicitGetter === 'function') {
    return explicitGetter;
  }
  if (!mod || typeof mod !== 'object') {
    return null;
  }
  if (typeof mod._playgta6_audio_ring === 'function') {
    return () => mod._playgta6_audio_ring();
  }
  if (typeof mod.playgta6_audio_ring === 'function') {
    return () => mod.playgta6_audio_ring();
  }
  if (mod.asm && typeof mod.asm._playgta6_audio_ring === 'function') {
    return () => mod.asm._playgta6_audio_ring();
  }
  if (mod.asm && typeof mod.asm.playgta6_audio_ring === 'function') {
    return () => mod.asm.playgta6_audio_ring();
  }
  if (mod.wasmExports && typeof mod.wasmExports._playgta6_audio_ring === 'function') {
    return () => mod.wasmExports._playgta6_audio_ring();
  }
  if (mod.wasmExports && typeof mod.wasmExports.playgta6_audio_ring === 'function') {
    return () => mod.wasmExports.playgta6_audio_ring();
  }
  return null;
}

function resolveRuntimeMemoryBuffer(mod, explicitMemoryOrBuffer) {
  const explicitBuf = resolveBuffer(explicitMemoryOrBuffer);
  if (explicitBuf && !explicitBuf.detached && explicitBuf.byteLength > 0) {
    return explicitBuf;
  }
  if (!mod || typeof mod !== 'object') {
    return null;
  }
  const candidates = [
    mod.wasmMemory,
    mod.memory,
    mod.HEAPU8 && mod.HEAPU8.buffer,
    mod.HEAP32 && mod.HEAP32.buffer,
    mod.asm && mod.asm.memory,
    mod.wasmExports && mod.wasmExports.memory,
  ];
  for (let i = 0; i < candidates.length; i += 1) {
    const buf = resolveBuffer(candidates[i]);
    if (buf && !buf.detached && buf.byteLength > 0) {
      return buf;
    }
  }
  return null;
}

function sanitizeAudioRingReason(reason) {
  if (typeof reason === 'string') {
    const normalized = reason.trim().toLowerCase();
    if (AUDIO_RING_REASON_SET.has(normalized)) {
      return normalized;
    }
    if (
      normalized === 'no-audio-worker' ||
      normalized === 'audio-worker-not-ready' ||
      normalized === 'post-message-error'
    ) {
      return AUDIO_RING_REASON.WORKER_NOT_READY;
    }
    if (normalized === 'non-shared-buffer' || normalized === 'no-audio-ring') {
      return AUDIO_RING_REASON.INVALID_STRUCT;
    }
  }
  return AUDIO_RING_REASON.INVALID_STRUCT;
}

function normalizeAudioRingRetryIntervalMs(value, fallback = DEFAULT_AUDIO_RING_RETRY_INTERVAL_MS) {
  if (value === undefined || value === null || value === '' || typeof value === 'boolean') {
    return fallback;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.floor(n);
}

function normalizeAudioRingMaxRetries(value, fallback = DEFAULT_AUDIO_RING_MAX_RETRIES) {
  if (value === undefined || value === null || value === '' || typeof value === 'boolean') {
    return fallback;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.floor(n);
}

/**
 * Probe `playgta6_audio_ring()` on a Wasm module or exports object and decode the
 * `Playgta6AudioRing` struct (`runtime/os/audio_downmix.h`) from linear memory.
 *
 * Returns `{ descriptor, reason }` where `reason` is one of the sanitized enum values:
 * `'ok' | 'no_export' | 'null_ptr' | 'invalid_struct'`.
 */
function probePlaygta6AudioRingDescriptor(moduleOrExports, memoryBuffer, options) {
  let explicitMem = memoryBuffer;
  let opts = options;
  if (
    opts === undefined &&
    memoryBuffer &&
    typeof memoryBuffer === 'object' &&
    resolveBuffer(memoryBuffer) === null
  ) {
    opts = memoryBuffer;
    explicitMem = null;
  }
  const safeOpts = opts && typeof opts === 'object' ? opts : {};

  const ptrGetter = resolveAudioRingPointerGetter(moduleOrExports, safeOpts.getAudioRingPtr);
  if (!ptrGetter) {
    return { descriptor: null, reason: AUDIO_RING_REASON.NO_EXPORT };
  }

  let rawPtr;
  try {
    rawPtr = ptrGetter();
  } catch {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (rawPtr === null || rawPtr === undefined || rawPtr === 0 || rawPtr === 0n) {
    return { descriptor: null, reason: AUDIO_RING_REASON.NULL_PTR };
  }

  const pointerSize =
    safeOpts.pointerSize === 8 || safeOpts.pointerSize === 4
      ? safeOpts.pointerSize
      : typeof safeOpts.wasm64 === 'boolean'
        ? safeOpts.wasm64
          ? 8
          : 4
        : typeof rawPtr === 'bigint'
          ? 8
          : 4;
  const structSize =
    pointerSize === 8 ? AUDIO_RING_STRUCT_BYTES_WASM64 : AUDIO_RING_STRUCT_BYTES_WASM32;

  let structOffset;
  try {
    structOffset = ptrToNumber(rawPtr, 'audioRingPtr');
  } catch {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (structOffset <= 0 || structOffset % pointerSize !== 0) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }

  const memCandidate =
    explicitMem ||
    (typeof safeOpts.getMemory === 'function'
      ? safeOpts.getMemory()
      : safeOpts.memory || safeOpts.buffer || null);
  const buf = resolveRuntimeMemoryBuffer(moduleOrExports, memCandidate);
  if (!buf || structOffset + structSize > buf.byteLength) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }

  let view;
  try {
    view = new DataView(buf);
  } catch {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }

  let rawBase;
  let capacity;
  let channels;
  let sampleRate;
  let rawReadIndexPtr;
  let rawWriteIndexPtr;
  try {
    if (pointerSize === 8) {
      rawBase = view.getBigUint64(structOffset + 0, true);
      capacity = view.getUint32(structOffset + 8, true);
      channels = view.getUint32(structOffset + 12, true);
      sampleRate = view.getUint32(structOffset + 16, true);
      // Bytes +20..+23 are 4-byte alignment padding before 8-byte read_index_ptr.
      rawReadIndexPtr = view.getBigUint64(structOffset + 24, true);
      rawWriteIndexPtr = view.getBigUint64(structOffset + 32, true);
    } else {
      rawBase = view.getUint32(structOffset + 0, true);
      capacity = view.getUint32(structOffset + 4, true);
      channels = view.getUint32(structOffset + 8, true);
      sampleRate = view.getUint32(structOffset + 12, true);
      rawReadIndexPtr = view.getUint32(structOffset + 16, true);
      rawWriteIndexPtr = view.getUint32(structOffset + 20, true);
    }
  } catch {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }

  let baseOffset;
  let readIndexOffset;
  let writeIndexOffset;
  try {
    baseOffset = ptrToNumber(rawBase, 'audioRing.base');
    readIndexOffset = ptrToNumber(rawReadIndexPtr, 'audioRing.read_index_ptr');
    writeIndexOffset = ptrToNumber(rawWriteIndexPtr, 'audioRing.write_index_ptr');
  } catch {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }

  if (
    !Number.isInteger(capacity) ||
    capacity < AUDIO_RING_MIN_CAPACITY_SAMPLES ||
    capacity > AUDIO_RING_MAX_CAPACITY_SAMPLES ||
    (capacity & (capacity - 1)) !== 0
  ) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (!Number.isInteger(channels) || channels !== AUDIO_RING_CHANNELS) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (
    !Number.isInteger(sampleRate) ||
    sampleRate < AUDIO_RING_MIN_SAMPLE_RATE ||
    sampleRate > AUDIO_RING_MAX_SAMPLE_RATE
  ) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }

  if (baseOffset <= 0 || baseOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (readIndexOffset <= 0 || readIndexOffset % Uint32Array.BYTES_PER_ELEMENT !== 0) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (writeIndexOffset <= 0 || writeIndexOffset % Uint32Array.BYTES_PER_ELEMENT !== 0) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }

  const dataByteLength = capacity * Float32Array.BYTES_PER_ELEMENT;
  const dataEnd = baseOffset + dataByteLength;
  if (dataEnd > buf.byteLength) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (readIndexOffset + Uint32Array.BYTES_PER_ELEMENT > buf.byteLength) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (writeIndexOffset + Uint32Array.BYTES_PER_ELEMENT > buf.byteLength) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (readIndexOffset === writeIndexOffset) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (readIndexOffset >= baseOffset && readIndexOffset < dataEnd) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }
  if (writeIndexOffset >= baseOffset && writeIndexOffset < dataEnd) {
    return { descriptor: null, reason: AUDIO_RING_REASON.INVALID_STRUCT };
  }

  return {
    descriptor: {
      sab: buf,
      buffer: buf,
      base: baseOffset,
      dataByteOffset: baseOffset,
      capacity,
      capacitySamples: capacity,
      capacityFrames: capacity >>> 1,
      channels,
      sampleRate,
      readIndexPtr: readIndexOffset,
      readIndexByteOffset: readIndexOffset,
      writeIndexPtr: writeIndexOffset,
      writeIndexByteOffset: writeIndexOffset,
      structOffset,
      pointerSize,
      isShared: isSharedArrayBufferValue(buf),
    },
    reason: AUDIO_RING_REASON.OK,
  };
}

/**
 * Probe `playgta6_audio_ring()` on a Wasm module or exports object and decode the
 * `Playgta6AudioRing` struct (`runtime/os/audio_downmix.h`) from linear memory.
 *
 * Handles both Wasm64 (8-byte `uintptr_t`, 40-byte struct with 4-byte padding at +20,
 * `BigInt` return pointer) and Wasm32 (4-byte `uintptr_t`, 24-byte struct, `number`
 * return pointer). Returns a validated descriptor object compatible with
 * `audio_worker.js` (`GuestPcmRingReader`), or `null` on missing/invalid/out-of-bounds state.
 */
function readPlaygta6AudioRingDescriptor(moduleOrExports, memoryBuffer, options) {
  return probePlaygta6AudioRingDescriptor(moduleOrExports, memoryBuffer, options).descriptor;
}

function resetAttachedRuntimeAudioRing(targetAudioWorker) {
  if (targetAudioWorker && typeof targetAudioWorker === 'object') {
    attachedAudioRingByWorker.delete(targetAudioWorker);
  }
}

/**
 * Probe `playgta6_audio_ring()` from the Wasm runtime and post an `attach-guest-pcm-ring`
 * message to `targetAudioWorker` matching `web/workers/audio_worker.js`'s exact schema.
 *
 * - Deduplicates repeated attachments for the same `(buffer, baseOffset)` on the same worker
 *   unless `options.force === true` or `resetAttachedRuntimeAudioRing(worker)` was called.
 * - Never throws when `_playgta6_audio_ring` is absent, returns `0`/`0n`, or `audioWorker`
 *   is missing or unavailable.
 */
function attachRuntimeAudioRing(moduleOrExports, targetAudioWorker, options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  if (!targetAudioWorker || typeof targetAudioWorker.postMessage !== 'function') {
    return {
      attached: false,
      alreadyAttached: false,
      descriptor: null,
      reason: 'no-audio-worker',
      statusReason: AUDIO_RING_REASON.WORKER_NOT_READY,
    };
  }

  const probe = probePlaygta6AudioRingDescriptor(
    moduleOrExports,
    safeOpts.memory || safeOpts.buffer || null,
    safeOpts,
  );
  const descriptor = probe.descriptor;
  if (!descriptor) {
    return {
      attached: false,
      alreadyAttached: false,
      descriptor: null,
      reason: 'no-audio-ring',
      statusReason: probe.reason,
    };
  }

  if (safeOpts.requireShared === true && !descriptor.isShared) {
    return {
      attached: false,
      alreadyAttached: false,
      descriptor,
      reason: 'non-shared-buffer',
      statusReason: AUDIO_RING_REASON.INVALID_STRUCT,
    };
  }

  const previous = attachedAudioRingByWorker.get(targetAudioWorker);
  if (
    !safeOpts.force &&
    previous &&
    previous.buffer === descriptor.buffer &&
    previous.baseOffset === descriptor.dataByteOffset
  ) {
    return {
      attached: true,
      alreadyAttached: true,
      descriptor: previous.descriptor,
      statusReason: AUDIO_RING_REASON.OK,
    };
  }

  const message = {
    type: 'attach-guest-pcm-ring',
    sab: descriptor.sab,
    buffer: descriptor.buffer,
    base: descriptor.base,
    dataByteOffset: descriptor.dataByteOffset,
    capacity: descriptor.capacity,
    capacitySamples: descriptor.capacitySamples,
    capacityFrames: descriptor.capacityFrames,
    channels: descriptor.channels,
    sampleRate: descriptor.sampleRate,
    readIndexPtr: descriptor.readIndexPtr,
    readIndexByteOffset: descriptor.readIndexByteOffset,
    writeIndexPtr: descriptor.writeIndexPtr,
    writeIndexByteOffset: descriptor.writeIndexByteOffset,
    autoPump: safeOpts.autoPump !== undefined ? Boolean(safeOpts.autoPump) : true,
  };
  if (safeOpts.id !== undefined) {
    message.id = safeOpts.id;
  }
  if (safeOpts.quantumFrames !== undefined) {
    message.quantumFrames = safeOpts.quantumFrames;
  }
  if (safeOpts.softClip !== undefined) {
    message.softClip = Boolean(safeOpts.softClip);
  }
  if (safeOpts.limiterKnee !== undefined) {
    message.limiterKnee = safeOpts.limiterKnee;
  }
  if (safeOpts.resetStats !== undefined) {
    message.resetStats = Boolean(safeOpts.resetStats);
  }

  try {
    targetAudioWorker.postMessage(message);
  } catch {
    return {
      attached: false,
      alreadyAttached: false,
      descriptor,
      reason: 'post-message-error',
      statusReason: AUDIO_RING_REASON.WORKER_NOT_READY,
    };
  }

  attachedAudioRingByWorker.set(targetAudioWorker, {
    buffer: descriptor.buffer,
    baseOffset: descriptor.dataByteOffset,
    descriptor,
  });

  return {
    attached: true,
    alreadyAttached: false,
    descriptor,
    statusReason: AUDIO_RING_REASON.OK,
  };
}

/**
 * Bounded auto-retry controller for lazy `_playgta6_audio_ring()` discovery and delayed
 * `audioWorker` readiness (W308 / #1259).
 *
 * - Polls at `retryIntervalMs` (default 250 ms) up to `maxRetries` (default 120 = 30 s).
 * - Stops immediately upon successful attachment, `reset()`, `stop()`, or `destroy()`/`dispose()`.
 * - Supports immediate probe triggers via `onAudioWorkerReady`, `setAudioWorker`, and
 *   `notifyGuestMilestone` / `onFirstPresent`.
 */
function createAudioRingAttachmentController(options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  let retryIntervalMs = normalizeAudioRingRetryIntervalMs(
    safeOpts.retryIntervalMs !== undefined ? safeOpts.retryIntervalMs : safeOpts.intervalMs,
    DEFAULT_AUDIO_RING_RETRY_INTERVAL_MS,
  );
  let maxRetries = normalizeAudioRingMaxRetries(
    safeOpts.maxRetries,
    DEFAULT_AUDIO_RING_MAX_RETRIES,
  );
  const autoRetry = safeOpts.autoRetry !== false;
  const retryOnNoExport = safeOpts.retryOnNoExport === true;
  const setTimeoutFn =
    typeof safeOpts.setTimeoutFn === 'function'
      ? safeOpts.setTimeoutFn
      : typeof setTimeout === 'function'
        ? (fn, ms) => setTimeout(fn, ms)
        : null;
  const clearTimeoutFn =
    typeof safeOpts.clearTimeoutFn === 'function'
      ? safeOpts.clearTimeoutFn
      : typeof clearTimeout === 'function'
        ? (id) => clearTimeout(id)
        : null;

  let boundModule =
    safeOpts.module && typeof safeOpts.module === 'object' ? safeOpts.module : null;
  let pendingModule = null;
  let dynamicWorker = undefined;
  let dynamicWorkerReady = undefined;
  let activeDescriptor = null;
  let activeWorker = null;
  let retryTimerId = null;
  let retryCount = 0;
  let scheduledRetries = 0;
  let lastReason = null;
  let hasAttempted = false;
  let retryExhausted = false;
  let destroyed = false;

  function clearRetryTimer() {
    if (retryTimerId !== null) {
      const id = retryTimerId;
      retryTimerId = null;
      if (clearTimeoutFn) {
        try {
          clearTimeoutFn(id);
        } catch {
          // Ignore timer teardown errors.
        }
      }
    }
  }

  function resolveModule(modOverride) {
    if (modOverride && typeof modOverride === 'object') {
      boundModule = modOverride;
      return modOverride;
    }
    if (typeof safeOpts.getModule === 'function') {
      try {
        const dynamicMod = safeOpts.getModule();
        if (dynamicMod && typeof dynamicMod === 'object') {
          boundModule = dynamicMod;
          return dynamicMod;
        }
      } catch {
        // Ignore dynamic module getter errors.
      }
    }
    return boundModule || pendingModule || null;
  }

  function resolveWorker(workerOverride) {
    if (workerOverride && typeof workerOverride.postMessage === 'function') {
      return workerOverride;
    }
    if (dynamicWorker !== undefined) {
      return dynamicWorker;
    }
    if (typeof safeOpts.getAudioWorker === 'function') {
      try {
        return safeOpts.getAudioWorker();
      } catch {
        return null;
      }
    }
    if (safeOpts.audioWorker !== undefined) {
      return safeOpts.audioWorker;
    }
    return null;
  }

  function isWorkerReady(workerOverride) {
    if (workerOverride && typeof workerOverride.postMessage === 'function') {
      return true;
    }
    if (dynamicWorkerReady !== undefined) {
      return dynamicWorkerReady;
    }
    if (typeof safeOpts.isAudioWorkerReady === 'function') {
      try {
        return Boolean(safeOpts.isAudioWorkerReady());
      } catch {
        return false;
      }
    }
    if (safeOpts.audioWorker !== undefined || typeof safeOpts.getAudioWorker === 'function') {
      return true;
    }
    return false;
  }

  function buildAttachOptions(targetMod, attachOpts) {
    const dynamicOpts =
      typeof safeOpts.getAttachOptions === 'function' ? safeOpts.getAttachOptions(targetMod) : null;
    const baseOpts =
      safeOpts.attachOptions && typeof safeOpts.attachOptions === 'object'
        ? safeOpts.attachOptions
        : null;
    const inlineOpts = {};
    if (safeOpts.memory !== undefined) inlineOpts.memory = safeOpts.memory;
    if (safeOpts.buffer !== undefined) inlineOpts.buffer = safeOpts.buffer;
    if (typeof safeOpts.getMemory === 'function') inlineOpts.getMemory = safeOpts.getMemory;
    if (typeof safeOpts.getAudioRingPtr === 'function') {
      inlineOpts.getAudioRingPtr = safeOpts.getAudioRingPtr;
    }
    if (safeOpts.pointerSize !== undefined) inlineOpts.pointerSize = safeOpts.pointerSize;
    if (safeOpts.wasm64 !== undefined) inlineOpts.wasm64 = safeOpts.wasm64;
    if (safeOpts.autoPump !== undefined) inlineOpts.autoPump = safeOpts.autoPump;
    if (safeOpts.requireShared !== undefined) inlineOpts.requireShared = safeOpts.requireShared;
    return Object.assign(
      inlineOpts,
      dynamicOpts && typeof dynamicOpts === 'object' ? dynamicOpts : null,
      baseOpts,
      attachOpts && typeof attachOpts === 'object' ? attachOpts : null,
    );
  }

  function getTelemetry() {
    return {
      audioRingAttached: activeDescriptor !== null,
      audioRingRetryCount: retryCount,
      audioRingLastReason: lastReason,
      audioRingRetryPending: retryTimerId !== null,
      audioRingRetryExhausted: retryExhausted,
      audioRingMaxRetries: maxRetries,
      audioRingRetryIntervalMs: retryIntervalMs,
      destroyed,
    };
  }

  function notifyAttempt(res) {
    if (typeof safeOpts.onAttempt === 'function') {
      try {
        safeOpts.onAttempt(res, getTelemetry());
      } catch {
        // Ignore telemetry observer errors.
      }
    }
  }

  function markExhausted() {
    if (retryExhausted) {
      return;
    }
    retryExhausted = true;
    if (typeof safeOpts.onExhausted === 'function') {
      try {
        safeOpts.onExhausted(getTelemetry());
      } catch {
        // Ignore observer errors.
      }
    }
  }

  function shouldRetryReason(statusReason, targetMod, mergedAttachOpts) {
    if (statusReason === AUDIO_RING_REASON.OK) {
      return false;
    }
    if (statusReason === AUDIO_RING_REASON.NO_EXPORT) {
      return retryOnNoExport || (typeof safeOpts.getModule === 'function' && !targetMod);
    }
    if (statusReason === AUDIO_RING_REASON.WORKER_NOT_READY) {
      if (!targetMod) {
        return typeof safeOpts.getModule === 'function';
      }
      const hasExport = Boolean(
        resolveAudioRingPointerGetter(
          targetMod,
          mergedAttachOpts && mergedAttachOpts.getAudioRingPtr,
        ),
      );
      return hasExport || retryOnNoExport;
    }
    return true;
  }

  function scheduleNextTimer() {
    if (destroyed || activeDescriptor !== null || retryTimerId !== null) {
      return false;
    }
    if (!setTimeoutFn || maxRetries <= 0) {
      markExhausted();
      return false;
    }
    if (scheduledRetries >= maxRetries) {
      markExhausted();
      return false;
    }
    retryTimerId = setTimeoutFn(() => {
      retryTimerId = null;
      if (destroyed || activeDescriptor !== null) {
        return;
      }
      scheduledRetries += 1;
      const res = attemptAttach(null, null, null, { isRetry: true, allowSchedule: true });
      if ((!res || !res.attached) && !destroyed && retryTimerId === null && scheduledRetries >= maxRetries) {
        markExhausted();
      }
    }, retryIntervalMs);
    return true;
  }

  function attemptAttach(modOverride, workerOverride, attachOpts, internalOpts = {}) {
    if (destroyed) {
      return {
        attached: false,
        alreadyAttached: false,
        descriptor: null,
        reason: 'destroyed',
        statusReason: lastReason || AUDIO_RING_REASON.WORKER_NOT_READY,
      };
    }

    const targetMod = resolveModule(modOverride);
    if (!targetMod) {
      if (internalOpts.isRetry) {
        retryCount += 1;
      }
      hasAttempted = true;
      lastReason = AUDIO_RING_REASON.NO_EXPORT;
      if (
        internalOpts.allowSchedule !== false &&
        autoRetry &&
        shouldRetryReason(lastReason, null, null)
      ) {
        scheduleNextTimer();
      }
      return null;
    }

    if (internalOpts.isRetry) {
      retryCount += 1;
    }
    hasAttempted = true;

    const mergedAttachOpts = buildAttachOptions(targetMod, attachOpts);
    const targetWorker = resolveWorker(workerOverride);
    if (!targetWorker || typeof targetWorker.postMessage !== 'function') {
      pendingModule = targetMod;
      lastReason = AUDIO_RING_REASON.WORKER_NOT_READY;
      const res = {
        attached: false,
        alreadyAttached: false,
        pending: true,
        descriptor: null,
        reason: 'no-audio-worker',
        statusReason: AUDIO_RING_REASON.WORKER_NOT_READY,
      };
      if (
        internalOpts.allowSchedule !== false &&
        autoRetry &&
        shouldRetryReason(lastReason, targetMod, mergedAttachOpts)
      ) {
        scheduleNextTimer();
      }
      notifyAttempt(res);
      return res;
    }

    if (!workerOverride && !isWorkerReady(null)) {
      pendingModule = targetMod;
      lastReason = AUDIO_RING_REASON.WORKER_NOT_READY;
      const res = {
        attached: false,
        alreadyAttached: false,
        pending: true,
        descriptor: null,
        reason: 'audio-worker-not-ready',
        statusReason: AUDIO_RING_REASON.WORKER_NOT_READY,
      };
      if (
        internalOpts.allowSchedule !== false &&
        autoRetry &&
        shouldRetryReason(lastReason, targetMod, mergedAttachOpts)
      ) {
        scheduleNextTimer();
      }
      notifyAttempt(res);
      return res;
    }

    const res = attachRuntimeAudioRing(targetMod, targetWorker, mergedAttachOpts);
    const statusReason = sanitizeAudioRingReason(res && (res.statusReason || res.reason));
    lastReason = res && res.attached ? AUDIO_RING_REASON.OK : statusReason;

    if (res && res.attached && res.descriptor) {
      clearRetryTimer();
      pendingModule = null;
      activeDescriptor = res.descriptor;
      activeWorker = targetWorker;
      retryExhausted = false;
      notifyAttempt(res);
      if (!res.alreadyAttached && typeof safeOpts.onAttached === 'function') {
        try {
          safeOpts.onAttached(res, getTelemetry());
        } catch {
          // Ignore callback errors.
        }
      }
      return res;
    }

    pendingModule = targetMod;
    if (
      internalOpts.allowSchedule !== false &&
      autoRetry &&
      shouldRetryReason(lastReason, targetMod, mergedAttachOpts)
    ) {
      scheduleNextTimer();
    } else if (lastReason === AUDIO_RING_REASON.NO_EXPORT && !retryOnNoExport) {
      clearRetryTimer();
    }
    notifyAttempt(res);
    return res;
  }

  function schedule(retryOpts) {
    if (destroyed || activeDescriptor !== null) {
      return getTelemetry();
    }
    if (retryOpts && typeof retryOpts === 'object') {
      if (retryOpts.retryIntervalMs !== undefined || retryOpts.intervalMs !== undefined) {
        retryIntervalMs = normalizeAudioRingRetryIntervalMs(
          retryOpts.retryIntervalMs !== undefined
            ? retryOpts.retryIntervalMs
            : retryOpts.intervalMs,
          retryIntervalMs,
        );
      }
      if (retryOpts.maxRetries !== undefined) {
        maxRetries = normalizeAudioRingMaxRetries(retryOpts.maxRetries, maxRetries);
        if (scheduledRetries < maxRetries) {
          retryExhausted = false;
        }
      }
      if (retryOpts.immediate === true) {
        clearRetryTimer();
        attemptAttach(
          retryOpts.module || null,
          retryOpts.audioWorker || null,
          retryOpts.attachOptions || null,
          { isRetry: hasAttempted, allowSchedule: true },
        );
        return getTelemetry();
      }
    }
    scheduleNextTimer();
    return getTelemetry();
  }

  function onAudioWorkerReady(workerOverride) {
    if (destroyed) {
      return null;
    }
    if (workerOverride && typeof workerOverride.postMessage === 'function') {
      dynamicWorker = workerOverride;
    }
    dynamicWorkerReady = true;
    const targetWorker = resolveWorker(workerOverride);
    if (targetWorker && typeof targetWorker.postMessage === 'function') {
      resetAttachedRuntimeAudioRing(targetWorker);
    }
    if (activeWorker && targetWorker && activeWorker !== targetWorker) {
      resetAttachedRuntimeAudioRing(activeWorker);
    }
    activeWorker = null;
    activeDescriptor = null;
    const targetMod = resolveModule(null);
    if (!targetMod) {
      return null;
    }
    clearRetryTimer();
    return attemptAttach(targetMod, targetWorker, { force: true }, {
      isRetry: hasAttempted,
      allowSchedule: true,
    });
  }

  function setAudioWorker(nextWorker, readyOverride) {
    if (destroyed) {
      return null;
    }
    if (activeWorker && activeWorker !== nextWorker) {
      resetAttachedRuntimeAudioRing(activeWorker);
      activeWorker = null;
      activeDescriptor = null;
    }
    const hasPostMessage = Boolean(nextWorker && typeof nextWorker.postMessage === 'function');
    dynamicWorker = hasPostMessage ? nextWorker : null;
    dynamicWorkerReady =
      readyOverride !== undefined ? Boolean(readyOverride) : hasPostMessage;

    const targetMod = resolveModule(null);
    if (!hasPostMessage || !dynamicWorkerReady) {
      if (targetMod) {
        pendingModule = targetMod;
        lastReason = AUDIO_RING_REASON.WORKER_NOT_READY;
        const mergedAttachOpts = buildAttachOptions(targetMod, null);
        if (autoRetry && shouldRetryReason(lastReason, targetMod, mergedAttachOpts)) {
          scheduleNextTimer();
        }
      }
      return null;
    }
    resetAttachedRuntimeAudioRing(nextWorker);
    if (!targetMod) {
      return null;
    }
    clearRetryTimer();
    return attemptAttach(targetMod, nextWorker, { force: true }, {
      isRetry: hasAttempted,
      allowSchedule: true,
    });
  }

  function notifyGuestMilestone() {
    if (destroyed || activeDescriptor !== null) {
      return null;
    }
    const targetMod = resolveModule(null);
    if (!targetMod) {
      return null;
    }
    clearRetryTimer();
    return attemptAttach(targetMod, null, null, {
      isRetry: hasAttempted,
      allowSchedule: true,
    });
  }

  function stop() {
    clearRetryTimer();
  }

  function reset() {
    clearRetryTimer();
    if (activeWorker) {
      resetAttachedRuntimeAudioRing(activeWorker);
    }
    activeWorker = null;
    activeDescriptor = null;
    pendingModule = null;
    boundModule =
      typeof safeOpts.getModule === 'function'
        ? null
        : safeOpts.module && typeof safeOpts.module === 'object'
          ? safeOpts.module
          : null;
    dynamicWorker = undefined;
    dynamicWorkerReady = undefined;
    retryCount = 0;
    scheduledRetries = 0;
    lastReason = null;
    hasAttempted = false;
    retryExhausted = false;
    destroyed = false;
  }

  function destroy() {
    clearRetryTimer();
    if (activeWorker) {
      resetAttachedRuntimeAudioRing(activeWorker);
    }
    activeWorker = null;
    activeDescriptor = null;
    pendingModule = null;
    boundModule = null;
    dynamicWorker = undefined;
    dynamicWorkerReady = undefined;
    destroyed = true;
  }

  return {
    attach: (mod, workerOverride, attachOpts) =>
      attemptAttach(mod, workerOverride, attachOpts, { isRetry: false, allowSchedule: true }),
    probeNow: (attachOpts) => {
      if (destroyed) {
        return null;
      }
      clearRetryTimer();
      return attemptAttach(null, null, attachOpts, {
        isRetry: hasAttempted,
        allowSchedule: true,
      });
    },
    schedule,
    stop,
    reset,
    destroy,
    dispose: destroy,
    onAudioWorkerReady,
    notifyAudioWorkerReady: onAudioWorkerReady,
    setAudioWorker,
    notifyGuestMilestone,
    onFirstPresent: () => notifyGuestMilestone('first_present'),
    isAttached: () => activeDescriptor !== null,
    getDescriptor: () => activeDescriptor,
    getTelemetry,
    getStats: getTelemetry,
  };
}

/**
 * Start a bounded auto-retry poller for `_playgta6_audio_ring()` attachment (W308 / #1259).
 *
 * Accepts either `(options)` or `(moduleOrExports, targetAudioWorker, options)`.
 * Returns a controller handle with `probeNow()`, `onAudioWorkerReady()`, `setAudioWorker()`,
 * `notifyGuestMilestone()`, `getTelemetry()`, `reset()`, `stop()`, and `destroy()`/`dispose()`.
 */
function scheduleAudioRingRetry(moduleOrOptions, targetAudioWorker, maybeOptions) {
  let opts;
  const looksLikeWasmMod = Boolean(
    moduleOrOptions &&
      typeof moduleOrOptions === 'object' &&
      (typeof moduleOrOptions._playgta6_audio_ring === 'function' ||
        typeof moduleOrOptions.playgta6_audio_ring === 'function' ||
        (moduleOrOptions.asm &&
          (typeof moduleOrOptions.asm._playgta6_audio_ring === 'function' ||
            typeof moduleOrOptions.asm.playgta6_audio_ring === 'function')) ||
        (moduleOrOptions.wasmExports &&
          (typeof moduleOrOptions.wasmExports._playgta6_audio_ring === 'function' ||
            typeof moduleOrOptions.wasmExports.playgta6_audio_ring === 'function'))),
  );
  if (
    arguments.length <= 1 &&
    moduleOrOptions &&
    typeof moduleOrOptions === 'object' &&
    !looksLikeWasmMod
  ) {
    opts = Object.assign({}, moduleOrOptions);
  } else {
    opts = Object.assign(
      {},
      maybeOptions && typeof maybeOptions === 'object' ? maybeOptions : null,
    );
    if (moduleOrOptions !== undefined) {
      opts.module = moduleOrOptions;
    }
    if (targetAudioWorker !== undefined) {
      opts.audioWorker = targetAudioWorker;
    }
  }
  const controller = createAudioRingAttachmentController(opts);
  if (opts.immediate !== false) {
    controller.attach(opts.module || null, opts.audioWorker || null, opts.attachOptions || null);
  } else {
    controller.schedule();
  }
  return controller;
}

// W377 (#1028): Headed AudioWorklet SharedArrayBuffer & AudioContext state bridge between
// the main thread (`web/app_bootstrap.js` / `web/audio/audio_engine.js`) and `loader.js`'s
// child `audioWorker` (`web/workers/audio_worker.js`).
const AUDIO_CONTEXT_STATE_MAP = Object.freeze({
  active: 'active',
  running: 'running',
  worklet: 'worklet',
  live: 'live',
  playing: 'playing',
  suspended: 'suspended',
  paused: 'suspended',
  interrupted: 'interrupted',
  closed: 'closed',
  stopped: 'closed',
  'headless-drain': 'headless-drain',
  headless: 'headless',
  unavailable: 'unavailable',
  unsupported: 'unsupported',
  error: 'error',
});

function normalizeAudioContextStateValue(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const key = value.trim().toLowerCase();
  if (!key) {
    return null;
  }
  return Object.prototype.hasOwnProperty.call(AUDIO_CONTEXT_STATE_MAP, key)
    ? AUDIO_CONTEXT_STATE_MAP[key]
    : null;
}

function sanitizeAudioBridgeSampleRate(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  return Math.floor(value);
}

function sanitizeAudioBridgeChannels(value) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > 8) {
    return null;
  }
  return value;
}

function extractAudioRingSab(msg, options = {}) {
  if (!msg || typeof msg !== 'object') {
    return null;
  }
  const allowGenericSab = Boolean(options && options.allowGenericSab === true);
  if (isSharedArrayBuffer(msg.ringSab)) {
    return msg.ringSab;
  }
  if (isSharedArrayBuffer(msg.audioSab)) {
    return msg.audioSab;
  }
  if (allowGenericSab && isSharedArrayBuffer(msg.sab)) {
    return msg.sab;
  }
  return null;
}

function isGpuRingControlMessage(msg) {
  return Boolean(
    msg &&
      typeof msg === 'object' &&
      (msg.attachRing ||
        msg.drainRing ||
        msg.type === 'attach-ring' ||
        msg.type === 'drain-ring'),
  );
}

function isAudioContextStateMessage(msg) {
  if (!msg || typeof msg !== 'object') {
    return false;
  }
  return (
    msg.setOutputState === true ||
    msg.type === 'audio-context-state' ||
    msg.type === 'output-state'
  );
}

function isAudioRingSabMessage(msg) {
  if (!msg || typeof msg !== 'object' || isGpuRingControlMessage(msg)) {
    return false;
  }
  if (
    msg.attachRingSab === true ||
    msg.type === 'attach-ring-sab' ||
    msg.type === 'attachRingSab' ||
    msg.type === 'attach-sab' ||
    msg.type === 'audio-sab' ||
    msg.type === 'set-audio-sab'
  ) {
    return true;
  }
  return isSharedArrayBuffer(msg.audioSab) || isSharedArrayBuffer(msg.ringSab);
}

function createAudioWorkerInitPayload(msg = {}, pendingState = null) {
  const safeMsg = msg && typeof msg === 'object' ? msg : {};
  const pending = pendingState && typeof pendingState === 'object' ? pendingState : null;
  const pendingAttach =
    pending && pending.pendingRingAttach && typeof pending.pendingRingAttach === 'object'
      ? pending.pendingRingAttach
      : null;
  const pendingCtx =
    pending && pending.pendingContextState && typeof pending.pendingContextState === 'object'
      ? pending.pendingContextState
      : null;

  const payload = { init: true };

  const bootSab = extractAudioRingSab(safeMsg, { allowGenericSab: false });
  const pendingSab =
    (pendingAttach && isSharedArrayBuffer(pendingAttach.ringSab) ? pendingAttach.ringSab : null) ||
    (pendingCtx && isSharedArrayBuffer(pendingCtx.ringSab) ? pendingCtx.ringSab : null);
  const ringSab = bootSab || pendingSab;
  if (ringSab) {
    payload.ringSab = ringSab;
  }

  const bootSampleRate = sanitizeAudioBridgeSampleRate(
    safeMsg.audioSampleRate !== undefined ? safeMsg.audioSampleRate : safeMsg.sampleRate,
  );
  const pendingSampleRate = sanitizeAudioBridgeSampleRate(
    pendingCtx && pendingCtx.sampleRate !== undefined
      ? pendingCtx.sampleRate
      : pendingAttach && pendingAttach.sampleRate !== undefined
        ? pendingAttach.sampleRate
        : null,
  );
  const sampleRate = bootSampleRate !== null ? bootSampleRate : pendingSampleRate;
  if (sampleRate !== null) {
    payload.sampleRate = sampleRate;
  }

  const bootChannels = sanitizeAudioBridgeChannels(
    safeMsg.audioChannels !== undefined ? safeMsg.audioChannels : safeMsg.channels,
  );
  const pendingChannels = sanitizeAudioBridgeChannels(
    pendingCtx && pendingCtx.channels !== undefined
      ? pendingCtx.channels
      : pendingAttach && pendingAttach.channels !== undefined
        ? pendingAttach.channels
        : null,
  );
  const channels = bootChannels !== null ? bootChannels : pendingChannels;
  if (channels !== null) {
    payload.channels = channels;
  }

  const bootState = normalizeAudioContextStateValue(
    safeMsg.audioOutputState !== undefined
      ? safeMsg.audioOutputState
      : safeMsg.audioContextState !== undefined
        ? safeMsg.audioContextState
        : safeMsg.outputState,
  );
  const pendingCtxState =
    pendingCtx && typeof pendingCtx.outputState === 'string'
      ? normalizeAudioContextStateValue(pendingCtx.outputState)
      : null;
  const pendingAttachState =
    pendingAttach && typeof pendingAttach.outputState === 'string'
      ? normalizeAudioContextStateValue(pendingAttach.outputState)
      : null;
  // Do not let a stale pre-boot `'closed'` state suppress a brand-new boot `audioSab`.
  const usablePendingState =
    pendingCtxState === 'closed' && bootSab !== null
      ? null
      : pendingCtxState || pendingAttachState;
  const outputState = bootState || usablePendingState;
  if (outputState !== null) {
    payload.outputState = outputState;
  }

  if (typeof safeMsg.headless === 'boolean') {
    payload.headless = safeMsg.headless;
  } else if (pendingAttach && typeof pendingAttach.headless === 'boolean') {
    payload.headless = pendingAttach.headless;
  }

  return payload;
}

/**
 * Coordinates headed AudioWorklet SAB (`ringSab` / `attach-ring-sab`) and main-thread
 * `audio-context-state` forwarding from `web/loader.js` to its child `audioWorker`,
 * handling pre-boot/late-arrival ordering and teardown cleanup without steady-state
 * per-frame allocations.
 */
function createAudioWorkerBridge(options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  const onResetWorker =
    typeof safeOpts.onResetWorker === 'function' ? safeOpts.onResetWorker : null;

  let worker = null;
  let workerReady = false;
  let initPosted = false;
  let destroyed = false;
  let attachedRingSab = null;
  let pendingRingAttach = null;
  let pendingContextState = null;
  let lastOutputState = null;
  let lastSampleRate = null;

  function canPostNow() {
    return Boolean(!destroyed && worker && typeof worker.postMessage === 'function' && initPosted);
  }

  function safePost(targetWorker, payload) {
    if (!targetWorker || typeof targetWorker.postMessage !== 'function') {
      return false;
    }
    try {
      targetWorker.postMessage(payload);
      return true;
    } catch {
      return false;
    }
  }

  function buildAttachRingSabPayload(msg) {
    const explicitAttach = Boolean(
      msg.attachRingSab === true ||
        msg.type === 'attach-ring-sab' ||
        msg.type === 'attachRingSab' ||
        msg.type === 'attach-sab' ||
        msg.type === 'audio-sab' ||
        msg.type === 'set-audio-sab',
    );
    const ringSab = extractAudioRingSab(msg, { allowGenericSab: explicitAttach });
    if (!ringSab) {
      return null;
    }
    const payload = {
      type: 'attach-ring-sab',
      attachRingSab: true,
      ringSab,
    };
    const sampleRate = sanitizeAudioBridgeSampleRate(msg.sampleRate);
    if (sampleRate !== null) {
      payload.sampleRate = sampleRate;
    }
    const channels = sanitizeAudioBridgeChannels(msg.channels);
    if (channels !== null) {
      payload.channels = channels;
    }
    const state = normalizeAudioContextStateValue(
      msg.outputState !== undefined
        ? msg.outputState
        : msg.state !== undefined
          ? msg.state
          : msg.audioContextState,
    );
    if (state !== null) {
      payload.outputState = state;
    }
    if (typeof msg.headless === 'boolean') {
      payload.headless = msg.headless;
    }
    if (msg.id !== undefined) {
      payload.id = msg.id;
    }
    return payload;
  }

  function buildAudioContextStatePayload(msg) {
    const rawState =
      msg.outputState !== undefined
        ? msg.outputState
        : msg.state !== undefined
          ? msg.state
          : msg.audioContextState;
    const outputState = normalizeAudioContextStateValue(rawState);
    if (outputState === null) {
      return null;
    }
    const payload = {
      type: 'audio-context-state',
      setOutputState: true,
      outputState,
    };
    const sampleRate = sanitizeAudioBridgeSampleRate(msg.sampleRate);
    if (sampleRate !== null) {
      payload.sampleRate = sampleRate;
    }
    const channels = sanitizeAudioBridgeChannels(msg.channels);
    if (channels !== null) {
      payload.channels = channels;
    }
    if (outputState !== 'closed') {
      const ringSab = extractAudioRingSab(msg, { allowGenericSab: false });
      if (ringSab) {
        payload.ringSab = ringSab;
      }
    }
    if (typeof msg.unlockedOnGesture === 'boolean') {
      payload.unlockedOnGesture = msg.unlockedOnGesture;
    }
    if (typeof msg.interrupted === 'boolean') {
      payload.interrupted = msg.interrupted;
    } else if (outputState === 'interrupted') {
      payload.interrupted = true;
    }
    if (typeof msg.recoveredFromInterruption === 'boolean') {
      payload.recoveredFromInterruption = msg.recoveredFromInterruption;
    }
    if (msg.id !== undefined) {
      payload.id = msg.id;
    }
    return payload;
  }

  function applyAttachTracking(payload) {
    attachedRingSab = payload.ringSab;
    if (payload.sampleRate !== undefined) {
      lastSampleRate = payload.sampleRate;
    }
    if (payload.outputState !== undefined) {
      lastOutputState = payload.outputState;
    } else if (payload.headless === true) {
      lastOutputState = 'headless-drain';
    } else if (!lastOutputState || lastOutputState === 'headless-drain') {
      lastOutputState = 'active';
    }
  }

  function applyContextStateTracking(payload) {
    lastOutputState = payload.outputState;
    if (payload.sampleRate !== undefined) {
      lastSampleRate = payload.sampleRate;
    }
    if (payload.outputState === 'closed') {
      attachedRingSab = null;
      pendingRingAttach = null;
    } else if (payload.ringSab) {
      attachedRingSab = payload.ringSab;
    }
  }

  function flushPendingToActiveWorker() {
    if (!canPostNow()) {
      return { flushedRingSab: false, flushedContextState: false };
    }
    let flushedRingSab = false;
    let flushedContextState = false;
    const queuedAttach = pendingRingAttach;
    const queuedCtx = pendingContextState;
    pendingRingAttach = null;
    pendingContextState = null;

    if (queuedAttach) {
      const needsSend =
        queuedAttach.ringSab !== attachedRingSab ||
        queuedAttach.id !== undefined ||
        queuedAttach.outputState !== undefined ||
        queuedAttach.sampleRate !== undefined ||
        queuedAttach.channels !== undefined ||
        queuedAttach.headless !== undefined;
      if (needsSend) {
        if (safePost(worker, queuedAttach)) {
          applyAttachTracking(queuedAttach);
          flushedRingSab = true;
        } else {
          pendingRingAttach = queuedAttach;
        }
      }
    }

    if (queuedCtx) {
      if (safePost(worker, queuedCtx)) {
        applyContextStateTracking(queuedCtx);
        flushedContextState = true;
      } else {
        pendingContextState = queuedCtx;
      }
    }

    return { flushedRingSab, flushedContextState };
  }

  function handleAttachRingSab(msg) {
    if (destroyed) {
      return { forwarded: false, queued: false, reason: 'destroyed', payload: null };
    }
    if (!msg || typeof msg !== 'object') {
      return { forwarded: false, queued: false, reason: 'invalid-message', payload: null };
    }
    const payload = buildAttachRingSabPayload(msg);
    if (!payload) {
      return { forwarded: false, queued: false, reason: 'invalid-ring-sab', payload: null };
    }

    if (!canPostNow()) {
      pendingRingAttach = payload;
      return { forwarded: false, queued: true, reason: 'worker-not-ready', payload };
    }

    const force = msg.force === true;
    const isDuplicate =
      !force &&
      payload.ringSab === attachedRingSab &&
      payload.id === undefined &&
      payload.outputState === undefined &&
      payload.sampleRate === undefined &&
      payload.channels === undefined &&
      payload.headless === undefined;
    if (isDuplicate) {
      return { forwarded: false, queued: false, deduplicated: true, payload };
    }

    if (!safePost(worker, payload)) {
      pendingRingAttach = payload;
      return { forwarded: false, queued: true, reason: 'post-message-error', payload };
    }
    pendingRingAttach = null;
    applyAttachTracking(payload);
    return { forwarded: true, queued: false, payload };
  }

  function handleAudioContextState(msg) {
    if (destroyed) {
      return { forwarded: false, queued: false, reason: 'destroyed', payload: null };
    }
    if (!msg || typeof msg !== 'object') {
      return { forwarded: false, queued: false, reason: 'invalid-message', payload: null };
    }
    const payload = buildAudioContextStatePayload(msg);
    if (!payload) {
      const fallbackSab = extractAudioRingSab(msg, { allowGenericSab: false });
      if (fallbackSab) {
        return handleAttachRingSab(msg);
      }
      return { forwarded: false, queued: false, reason: 'invalid-state', payload: null };
    }

    if (!canPostNow()) {
      pendingContextState = payload;
      if (payload.outputState === 'closed') {
        pendingRingAttach = null;
      } else if (payload.ringSab) {
        pendingRingAttach = {
          type: 'attach-ring-sab',
          attachRingSab: true,
          ringSab: payload.ringSab,
        };
      }
      lastOutputState = payload.outputState;
      if (payload.sampleRate !== undefined) {
        lastSampleRate = payload.sampleRate;
      }
      return { forwarded: false, queued: true, reason: 'worker-not-ready', payload };
    }

    if (!safePost(worker, payload)) {
      pendingContextState = payload;
      return { forwarded: false, queued: true, reason: 'post-message-error', payload };
    }
    pendingContextState = null;
    applyContextStateTracking(payload);
    return { forwarded: true, queued: false, payload };
  }

  function initWorker(nextWorker, bootMsg = {}) {
    if (!nextWorker || typeof nextWorker.postMessage !== 'function') {
      return null;
    }
    destroyed = false;
    worker = nextWorker;
    workerReady = false;
    initPosted = false;
    attachedRingSab = null;

    const queuedAttach = pendingRingAttach;
    const queuedCtx = pendingContextState;
    const safeBootMsg = bootMsg && typeof bootMsg === 'object' ? bootMsg : {};
    const bootSab = extractAudioRingSab(safeBootMsg, { allowGenericSab: false });
    const bootExplicitState = normalizeAudioContextStateValue(
      safeBootMsg.audioOutputState !== undefined
        ? safeBootMsg.audioOutputState
        : safeBootMsg.audioContextState !== undefined
          ? safeBootMsg.audioContextState
          : safeBootMsg.outputState,
    );

    const initPayload = createAudioWorkerInitPayload(safeBootMsg, {
      pendingRingAttach: queuedAttach,
      pendingContextState: queuedCtx,
    });

    // Clear pending slots and commit init state BEFORE calling postMessage so a
    // synchronous `onmessage({ data: { ready: true } })` re-entry cannot double-flush.
    initPosted = true;
    pendingRingAttach = null;
    pendingContextState = null;

    if (initPayload.ringSab) {
      attachedRingSab = initPayload.ringSab;
    }
    if (initPayload.outputState !== undefined) {
      lastOutputState = initPayload.outputState;
    } else if (initPayload.headless === true) {
      lastOutputState = 'headless-drain';
    } else if (initPayload.ringSab) {
      lastOutputState = 'active';
    } else {
      lastOutputState = null;
    }
    if (initPayload.sampleRate !== undefined) {
      lastSampleRate = initPayload.sampleRate;
    }

    const postedInit = safePost(nextWorker, initPayload);
    if (!postedInit) {
      initPosted = false;
      attachedRingSab = null;
      pendingRingAttach = queuedAttach;
      pendingContextState = queuedCtx;
      return null;
    }

    if (queuedAttach && queuedAttach.id !== undefined && queuedAttach.ringSab === initPayload.ringSab) {
      safePost(nextWorker, queuedAttach);
    }

    if (
      queuedCtx &&
      bootExplicitState === null &&
      !(queuedCtx.outputState === 'closed' && bootSab !== null)
    ) {
      const ctxToPost =
        bootSab !== null && queuedCtx.ringSab && queuedCtx.ringSab !== bootSab
          ? Object.assign({}, queuedCtx, { ringSab: bootSab })
          : queuedCtx;
      if (safePost(nextWorker, ctxToPost)) {
        applyContextStateTracking(ctxToPost);
      }
    }

    return initPayload;
  }

  function setWorker(nextWorker, readyOverride) {
    if (destroyed) {
      return false;
    }
    const prevWorker = worker;
    if (prevWorker && prevWorker !== nextWorker && onResetWorker) {
      try {
        onResetWorker(prevWorker);
      } catch {
        // Ignore worker reset callback errors.
      }
    }
    if (!nextWorker || typeof nextWorker.postMessage !== 'function') {
      worker = null;
      workerReady = false;
      initPosted = false;
      attachedRingSab = null;
      return false;
    }
    worker = nextWorker;
    attachedRingSab = null;
    const isReady = readyOverride !== undefined ? Boolean(readyOverride) : true;
    workerReady = isReady;
    initPosted = isReady;
    if (isReady) {
      flushPendingToActiveWorker();
    }
    return true;
  }

  function onWorkerReady(readyWorker) {
    if (destroyed) {
      return { flushedRingSab: false, flushedContextState: false };
    }
    if (readyWorker && typeof readyWorker.postMessage === 'function') {
      if (!worker) {
        worker = readyWorker;
      } else if (worker !== readyWorker) {
        return { flushedRingSab: false, flushedContextState: false };
      }
    }
    if (!worker) {
      return { flushedRingSab: false, flushedContextState: false };
    }
    workerReady = true;
    initPosted = true;
    return flushPendingToActiveWorker();
  }

  function clearPending() {
    pendingRingAttach = null;
    pendingContextState = null;
  }

  function reset() {
    const prevWorker = worker;
    worker = null;
    workerReady = false;
    initPosted = false;
    destroyed = false;
    attachedRingSab = null;
    pendingRingAttach = null;
    pendingContextState = null;
    lastOutputState = null;
    lastSampleRate = null;
    if (prevWorker && onResetWorker) {
      try {
        onResetWorker(prevWorker);
      } catch {
        // Ignore worker reset callback errors.
      }
    }
  }

  function destroy(destroyOpts = {}) {
    const opts = destroyOpts && typeof destroyOpts === 'object' ? destroyOpts : {};
    const prevWorker = worker;
    if (opts.sendStop === true && prevWorker && typeof prevWorker.postMessage === 'function') {
      safePost(prevWorker, { type: 'stop', stop: true });
    }
    reset();
    destroyed = true;
  }

  function getState() {
    return {
      hasWorker: worker !== null,
      workerReady,
      initPosted,
      ringSabAttached: attachedRingSab !== null,
      outputState: lastOutputState,
      sampleRate: lastSampleRate,
      hasPendingRingSab: pendingRingAttach !== null,
      hasPendingContextState: pendingContextState !== null,
      destroyed,
    };
  }

  return {
    initWorker,
    setWorker,
    onWorkerReady,
    handleAttachRingSab,
    handleAudioContextState,
    flushPending: flushPendingToActiveWorker,
    clearPending,
    reset,
    destroy,
    dispose: destroy,
    getState,
  };
}

function resolveInputStatePointerGetter(mod, explicitGetter) {
  if (typeof explicitGetter === 'function') {
    return explicitGetter;
  }
  if (!mod || typeof mod !== 'object') {
    return null;
  }
  if (typeof mod._playgta6_get_input_state_ptr === 'function') {
    return () => mod._playgta6_get_input_state_ptr();
  }
  if (mod.asm && typeof mod.asm.playgta6_get_input_state_ptr === 'function') {
    return () => mod.asm.playgta6_get_input_state_ptr();
  }
  if (mod.wasmExports && typeof mod.wasmExports.playgta6_get_input_state_ptr === 'function') {
    return () => mod.wasmExports.playgta6_get_input_state_ptr();
  }
  return null;
}

function makeInactiveInputBridgeHandle() {
  return {
    active: false,
    inputStatePtr: 0,
    syncNow() {
      return false;
    },
    stop() {},
    getStats() {
      return {
        active: false,
        stopped: true,
        inputStatePtr: 0,
        syncCount: 0,
        lastSrcSeq: 0,
        lastDstSeq: 0,
        bufferReacquires: 0,
        waitAsyncWakeups: 0,
        pollTicks: 0,
      };
    },
  };
}

/**
 * Bridge browser `InputBridge` (`inputSab`, 40-byte `playgta6::InputState` seqlock) into the
 * Wasm64 runtime's static linear-memory `InputState` exported via `_playgta6_get_input_state_ptr`.
 *
 * - Degrades cleanly (returns an inactive no-op handle) when `inputSab` is absent or the `.wasm`
 *   build does not export `_playgta6_get_input_state_ptr` (or returns 0 / 0n).
 * - Re-acquires `Int32Array(wasmMemory.buffer, ptr, 10)` automatically whenever linear memory
 *   grows or its backing buffer is replaced.
 * - Uses `Atomics.waitAsync` (default 250 ms timeout, woken instantaneously via `Atomics.notify`)
 *   when available, paired with a lightweight periodic fallback interval (50 ms default when
 *   `waitAsync` is active, or `pollIntervalMs` when explicitly specified / `waitAsync` unavailable)
 *   so writers that omit `Atomics.notify` never miss an edge without burning loader CPU when idle.
 */
function startInputSabBridge(options = {}) {
  const srcBuf = options && options.inputSab ? options.inputSab : null;
  if (!isSupportedInputBuffer(srcBuf)) {
    return makeInactiveInputBridgeHandle();
  }

  const mod = options && options.module && typeof options.module === 'object' ? options.module : null;
  const ptrGetter = resolveInputStatePointerGetter(mod, options.getInputStatePtr);
  if (!ptrGetter) {
    return makeInactiveInputBridgeHandle();
  }

  let rawPtr;
  try {
    rawPtr = ptrGetter();
  } catch {
    return makeInactiveInputBridgeHandle();
  }
  if (rawPtr === null || rawPtr === undefined || rawPtr === 0 || rawPtr === 0n) {
    return makeInactiveInputBridgeHandle();
  }

  let inputStatePtr = 0;
  try {
    inputStatePtr = ptrToNumber(rawPtr, 'inputStatePtr');
  } catch {
    return makeInactiveInputBridgeHandle();
  }
  if (inputStatePtr <= 0 || (inputStatePtr & 3) !== 0) {
    return makeInactiveInputBridgeHandle();
  }

  const hasExplicitPollInterval =
    typeof options.pollIntervalMs === 'number' &&
    Number.isFinite(options.pollIntervalMs) &&
    options.pollIntervalMs > 0;
  const pollIntervalMs = hasExplicitPollInterval
    ? Math.max(1, Math.floor(options.pollIntervalMs))
    : DEFAULT_INPUT_BRIDGE_POLL_MS;
  const waitAsyncTimeoutMs =
    typeof options.waitAsyncTimeoutMs === 'number' &&
    Number.isFinite(options.waitAsyncTimeoutMs) &&
    options.waitAsyncTimeoutMs > 0
      ? Math.max(1, Math.floor(options.waitAsyncTimeoutMs))
      : Math.max(pollIntervalMs, DEFAULT_INPUT_BRIDGE_WAIT_ASYNC_TIMEOUT_MS);

  const atomics =
    options.atomicsObj !== undefined
      ? options.atomicsObj
      : typeof Atomics === 'object' && Atomics !== null
        ? Atomics
        : null;
  const setIntervalFn =
    options.setIntervalFn !== undefined
      ? options.setIntervalFn
      : typeof setInterval === 'function'
        ? setInterval
        : null;
  const clearIntervalFn =
    options.clearIntervalFn !== undefined
      ? options.clearIntervalFn
      : typeof clearInterval === 'function'
        ? clearInterval
        : null;

  const isSharedSrc = isSharedArrayBufferValue(srcBuf);
  const srcI32 = new Int32Array(srcBuf, 0, INPUT_STATE_WORDS);
  const useSrcAtomics = canUseAtomicsOnInt32View(atomics, srcI32, isSharedSrc);

  const loadSrc = (idx) => (useSrcAtomics ? atomics.load(srcI32, idx) : srcI32[idx]);
  const exchangeSrc = (idx, val) => {
    if (useSrcAtomics && typeof atomics.exchange === 'function') {
      return atomics.exchange(srcI32, idx, val);
    }
    const prev = srcI32[idx];
    srcI32[idx] = val | 0;
    return prev;
  };

  function resolveWasmBuffer() {
    const raw =
      typeof options.getMemory === 'function'
        ? options.getMemory()
        : options.memory ||
          (mod &&
            (mod.wasmMemory ||
              (mod.HEAPU8 && mod.HEAPU8.buffer) ||
              (mod.HEAP32 && mod.HEAP32.buffer) ||
              (mod.asm && mod.asm.memory) ||
              (mod.wasmExports && mod.wasmExports.memory))) ||
          null;
    const buf = resolveBuffer(raw);
    if (!buf || buf.detached || buf.byteLength < inputStatePtr + INPUT_STATE_BYTES) {
      return null;
    }
    return buf;
  }

  let cachedDstBuf = null;
  let cachedDstI32 = null;
  let cachedDstUseAtomics = false;
  let bufferReacquires = 0;

  function acquireDstView() {
    const buf = resolveWasmBuffer();
    if (!buf) {
      cachedDstBuf = null;
      cachedDstI32 = null;
      cachedDstUseAtomics = false;
      return null;
    }
    if (buf !== cachedDstBuf || !cachedDstI32) {
      if (cachedDstBuf !== null) {
        bufferReacquires += 1;
      }
      cachedDstBuf = buf;
      cachedDstI32 = new Int32Array(buf, inputStatePtr, INPUT_STATE_WORDS);
      cachedDstUseAtomics = canUseAtomicsOnInt32View(
        atomics,
        cachedDstI32,
        isSharedArrayBufferValue(buf),
      );
    }
    return cachedDstI32;
  }

  const loadDst = (dst, idx) => (cachedDstUseAtomics ? atomics.load(dst, idx) : dst[idx]);
  const storeDst = (dst, idx, val) => {
    if (cachedDstUseAtomics) {
      atomics.store(dst, idx, val | 0);
    } else {
      dst[idx] = val | 0;
    }
  };
  const addDst = (dst, idx, delta) => {
    if (cachedDstUseAtomics) {
      atomics.add(dst, idx, delta | 0);
    } else {
      dst[idx] = (dst[idx] + delta) | 0;
    }
  };

  // Pre-allocated 9-word payload scratch (buttons, mouse_dx, mouse_dy, 6 x f32 axes/triggers).
  const payloadWords = new Int32Array(INPUT_STATE_WORDS - 1);

  let stopped = false;
  let hasSynced = false;
  let lastSrcSeq = 0;
  let dstSeq = 0;
  let syncCount = 0;
  let waitAsyncWakeups = 0;
  let pollTicks = 0;
  let waitAsyncEnabled =
    options.useWaitAsync !== false &&
    isSharedSrc &&
    Boolean(atomics && typeof atomics.waitAsync === 'function');
  let waitAsyncInFlight = false;
  let intervalId = null;

  function readConsistentSnapshot() {
    for (let attempt = 0; attempt < INPUT_SNAPSHOT_MAX_RETRIES; attempt += 1) {
      const seqBefore = loadSrc(0) >>> 0;
      if ((seqBefore & 1) !== 0) {
        continue;
      }
      for (let w = 1; w < INPUT_STATE_WORDS; w += 1) {
        payloadWords[w - 1] = loadSrc(w);
      }
      const seqAfter = loadSrc(0) >>> 0;
      if (seqBefore === seqAfter) {
        return seqBefore;
      }
    }
    return -1;
  }

  function hasNonZeroPayload() {
    for (let i = 0; i < payloadWords.length; i += 1) {
      if (payloadWords[i] !== 0) {
        return true;
      }
    }
    return false;
  }

  function writeSnapshotToWasm() {
    const dst = acquireDstView();
    if (!dst) {
      return false;
    }

    const existingDstSeq = loadDst(dst, 0) >>> 0;
    const baseSeq = existingDstSeq > dstSeq ? existingDstSeq : dstSeq;
    const oddSeq = ((baseSeq & 1) !== 0 ? baseSeq : baseSeq + 1) >>> 0;
    const nextEvenSeq = (oddSeq + 1) >>> 0;

    // Drain accumulated mouse deltas atomically from inputSab so we neither drop nor double-count.
    const dx = payloadWords[1] !== 0 ? exchangeSrc(2, 0) : 0;
    const dy = payloadWords[2] !== 0 ? exchangeSrc(3, 0) : 0;

    // Seqlock write matching MonotonicSeq & EvenOddSeqlock in runtime/os/game_input.cpp:
    // odd seq -> write 36-byte payload -> even seq.
    storeDst(dst, 0, oddSeq | 0);
    storeDst(dst, 1, payloadWords[0]);
    if (dx !== 0) {
      addDst(dst, 2, dx);
    }
    if (dy !== 0) {
      addDst(dst, 3, dy);
    }
    for (let w = 4; w < INPUT_STATE_WORDS; w += 1) {
      storeDst(dst, w, payloadWords[w - 1]);
    }
    storeDst(dst, 0, nextEvenSeq | 0);

    dstSeq = nextEvenSeq;
    return true;
  }

  function syncNow(force = false) {
    if (stopped) {
      return false;
    }
    const curSeq = loadSrc(0) >>> 0;
    if (!force && hasSynced && (curSeq & 1) === 0 && curSeq === lastSrcSeq) {
      return false;
    }

    const snapSeq = readConsistentSnapshot();
    if (snapSeq < 0) {
      return false;
    }

    if (!hasSynced) {
      if (!force && snapSeq === 0 && !hasNonZeroPayload()) {
        hasSynced = true;
        lastSrcSeq = 0;
        return false;
      }
      if (!writeSnapshotToWasm()) {
        return false;
      }
      hasSynced = true;
      lastSrcSeq = snapSeq;
      syncCount += 1;
      return true;
    }

    if (!force && snapSeq === lastSrcSeq) {
      return false;
    }

    if (!writeSnapshotToWasm()) {
      return false;
    }
    lastSrcSeq = snapSeq;
    syncCount += 1;
    return true;
  }

  function onWaitAsyncFulfilled(outcome) {
    waitAsyncInFlight = false;
    if (stopped) {
      return;
    }
    if (outcome === 'ok') {
      waitAsyncWakeups += 1;
    }
    const curSeq = (useSrcAtomics ? atomics.load(srcI32, 0) : srcI32[0]) >>> 0;
    if (!hasSynced || curSeq !== lastSrcSeq || (curSeq & 1) !== 0) {
      syncNow(false);
    }
    armWaitAsync(0);
  }

  function onWaitAsyncRejected() {
    waitAsyncInFlight = false;
    waitAsyncEnabled = false;
  }

  function armWaitAsync(syncDepth = 0) {
    if (stopped || !waitAsyncEnabled || waitAsyncInFlight) {
      return;
    }
    const expectedSeq = (hasSynced ? lastSrcSeq : 0) | 0;
    let res;
    try {
      res = atomics.waitAsync(srcI32, 0, expectedSeq, waitAsyncTimeoutMs);
    } catch {
      waitAsyncEnabled = false;
      return;
    }
    if (!res) {
      return;
    }
    if (res.async === false) {
      if (res.value === 'not-equal') {
        const synced = syncNow(false);
        if (synced && !stopped && syncDepth < 4) {
          armWaitAsync(syncDepth + 1);
        }
      }
      return;
    }
    if (res.value && typeof res.value.then === 'function') {
      waitAsyncInFlight = true;
      res.value.then(onWaitAsyncFulfilled, onWaitAsyncRejected);
    }
  }

  // Immediate initial sync on attach, then reactive waitAsync + periodic fallback timer.
  syncNow(false);
  armWaitAsync(0);

  const fallbackIntervalMs =
    !hasExplicitPollInterval && waitAsyncEnabled
      ? Math.max(pollIntervalMs, DEFAULT_INPUT_BRIDGE_WAIT_ASYNC_FALLBACK_POLL_MS)
      : pollIntervalMs;

  if (options.disableFallbackInterval !== true && typeof setIntervalFn === 'function') {
    intervalId = setIntervalFn(() => {
      if (stopped) {
        return;
      }
      pollTicks += 1;
      const curSeq = (useSrcAtomics ? atomics.load(srcI32, 0) : srcI32[0]) >>> 0;
      if (hasSynced && curSeq === lastSrcSeq && (curSeq & 1) === 0) {
        if (!waitAsyncInFlight && waitAsyncEnabled) {
          armWaitAsync(0);
        }
        return;
      }
      syncNow(false);
      if (!waitAsyncInFlight && waitAsyncEnabled) {
        armWaitAsync(0);
      }
    }, fallbackIntervalMs);
  }

  function stop() {
    if (stopped) {
      return;
    }
    stopped = true;
    if (intervalId !== null && typeof clearIntervalFn === 'function') {
      try {
        clearIntervalFn(intervalId);
      } catch {
        // Ignore timer cleanup errors.
      }
      intervalId = null;
    }
    if (waitAsyncInFlight && isSharedSrc && atomics && typeof atomics.notify === 'function') {
      try {
        atomics.notify(srcI32, 0, 1);
      } catch {
        // Ignore notify failures during teardown.
      }
    }
    cachedDstBuf = null;
    cachedDstI32 = null;
    cachedDstUseAtomics = false;
  }

  return {
    active: true,
    inputStatePtr,
    syncNow,
    stop,
    getStats() {
      return {
        active: !stopped,
        stopped,
        inputStatePtr,
        syncCount,
        lastSrcSeq,
        lastDstSeq: dstSeq,
        bufferReacquires,
        waitAsyncWakeups,
        pollTicks,
      };
    },
  };
}

function parseDumpPipelinesFromSearch(rawSearch) {
  if (typeof rawSearch !== 'string' || rawSearch.length === 0) {
    return false;
  }
  const withoutHash = rawSearch.split('#', 1)[0];
  const query = withoutHash.startsWith('?') ? withoutHash.slice(1) : withoutHash;
  if (!query) {
    return false;
  }
  try {
    if (typeof URLSearchParams === 'function') {
      const params = new URLSearchParams(query);
      const val = params.get('dumpPipelines');
      if (val === null) {
        return false;
      }
      const norm = val.trim().toLowerCase();
      return norm === '1' || norm === 'true';
    }
  } catch {
    // Fall through to simple query parser.
  }
  const pairs = query.split('&');
  for (let i = 0; i < pairs.length; i += 1) {
    const pair = pairs[i];
    if (!pair) continue;
    const eq = pair.indexOf('=');
    const rawKey = eq >= 0 ? pair.slice(0, eq) : pair;
    let key;
    try {
      key = decodeURIComponent(rawKey);
    } catch {
      key = rawKey;
    }
    if (key !== 'dumpPipelines') continue;
    const rawVal = eq >= 0 ? pair.slice(eq + 1) : '';
    let val;
    try {
      val = decodeURIComponent(rawVal);
    } catch {
      val = rawVal;
    }
    const norm = val.trim().toLowerCase();
    return norm === '1' || norm === 'true';
  }
  return false;
}

function extractSearchFromHref(rawHref) {
  if (typeof rawHref !== 'string' || rawHref.length === 0) {
    return '';
  }
  try {
    if (typeof URL === 'function' && isAbsoluteUrl(rawHref)) {
      return new URL(rawHref).search || '';
    }
  } catch {
    // Fall through to string extraction.
  }
  const qIdx = rawHref.indexOf('?');
  if (qIdx < 0) {
    return '';
  }
  const hashIdx = rawHref.indexOf('#', qIdx);
  return hashIdx >= 0 ? rawHref.slice(qIdx, hashIdx) : rawHref.slice(qIdx);
}

function resolveDumpPipelinesEnabled(options = {}) {
  if (options && options.dumpPipelines === true) {
    return true;
  }
  if (options && options.dumpPipelines === false) {
    return false;
  }
  if (options && typeof options.locationSearch === 'string') {
    return parseDumpPipelinesFromSearch(options.locationSearch);
  }
  if (options && typeof options.locationHref === 'string') {
    return parseDumpPipelinesFromSearch(extractSearchFromHref(options.locationHref));
  }
  try {
    const loc =
      typeof self !== 'undefined' && self && self.location
        ? self.location
        : typeof globalThis !== 'undefined' && globalThis && globalThis.location
          ? globalThis.location
          : null;
    if (!loc || typeof loc !== 'object') {
      return false;
    }
    if (typeof loc.search === 'string' && loc.search.length > 0) {
      return parseDumpPipelinesFromSearch(loc.search);
    }
    if (typeof loc.href === 'string' && loc.href.length > 0) {
      return parseDumpPipelinesFromSearch(extractSearchFromHref(loc.href));
    }
  } catch {
    // Safe fallback in restricted/headless environments without location.
  }
  return false;
}

function hasControlChar(str) {
  for (let i = 0; i < str.length; i += 1) {
    const code = str.charCodeAt(i);
    if (code < 32 || code === 127) {
      return true;
    }
  }
  return false;
}

function normalizeMemfsDumpDir(rawDir) {
  if (typeof rawDir !== 'string') {
    return DEFAULT_MEMFS_PIPELINE_DUMP_DIR;
  }
  const trimmed = rawDir.trim();
  if (!trimmed || trimmed.includes('\0') || trimmed.includes('\\') || hasControlChar(trimmed)) {
    return DEFAULT_MEMFS_PIPELINE_DUMP_DIR;
  }
  const stripped = trimmed.replace(/^\/+/, '').replace(/\/+$/, '');
  if (!stripped) {
    return DEFAULT_MEMFS_PIPELINE_DUMP_DIR;
  }
  const segments = stripped.split('/');
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i];
    if (!seg || seg === '.' || seg === '..' || seg.includes(':')) {
      return DEFAULT_MEMFS_PIPELINE_DUMP_DIR;
    }
  }
  return `/${segments.join('/')}`;
}

function normalizeDumpEndpoint(rawEndpoint) {
  if (typeof rawEndpoint !== 'string') {
    return DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT;
  }
  const trimmed = rawEndpoint.trim();
  if (!trimmed || trimmed.includes('\0') || trimmed.includes('\\') || hasControlChar(trimmed)) {
    return DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT;
  }
  const withoutQuery = trimmed.split(/[?#]/, 1)[0];
  if (isAbsoluteUrl(trimmed)) {
    if (!/^https?:/i.test(trimmed)) {
      return DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT;
    }
    try {
      if (typeof URL === 'function') {
        const parsed = new URL(trimmed);
        if (parsed.pathname.split('/').includes('..')) {
          return DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT;
        }
      }
    } catch {
      return DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT;
    }
    return ensureTrailingSlash(withoutQuery);
  }
  if (withoutQuery.split('/').includes('..')) {
    return DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT;
  }
  return ensureTrailingSlash(withoutQuery);
}

function sanitizeDumpRelPath(relPath) {
  if (typeof relPath !== 'string' || relPath.length === 0) {
    return null;
  }
  if (
    relPath.includes('\0') ||
    relPath.includes('\\') ||
    relPath.includes(':') ||
    relPath.startsWith('/') ||
    hasControlChar(relPath)
  ) {
    return null;
  }
  const segments = relPath.split('/');
  if (segments.length === 0) {
    return null;
  }
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i];
    if (!seg || seg === '.' || seg === '..') {
      return null;
    }
  }
  return segments.join('/');
}

function isFsExistsError(err) {
  if (!err) return false;
  if (err.code === 'EEXIST' || err.errno === 20) {
    return true;
  }
  const msg = typeof err.message === 'string' ? err.message : String(err);
  return /EEXIST|File exists/i.test(msg);
}

function isFsNotFoundError(err) {
  if (!err) return false;
  if (err.code === 'ENOENT' || err.errno === 44 || err.errno === 2) {
    return true;
  }
  const msg = typeof err.message === 'string' ? err.message : String(err);
  return /ENOENT|No such file/i.test(msg);
}

function toUint8ArrayCopy(raw) {
  if (raw instanceof Uint8Array) {
    const copy = new Uint8Array(raw.byteLength);
    copy.set(raw);
    return copy;
  }
  if (ArrayBuffer.isView(raw)) {
    const copy = new Uint8Array(raw.byteLength);
    copy.set(new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength));
    return copy;
  }
  if (raw instanceof ArrayBuffer) {
    return new Uint8Array(raw.slice(0));
  }
  if (typeof raw === 'string') {
    return utf8Encoder.encode(raw);
  }
  throw new TypeError('Unsupported FS.readFile return type');
}

function compareDumpCandidates(a, b) {
  const aIsManifest = a.relpath === 'manifest.json';
  const bIsManifest = b.relpath === 'manifest.json';
  if (aIsManifest !== bIsManifest) {
    return aIsManifest ? 1 : -1;
  }
  if (a.relpath < b.relpath) return -1;
  if (a.relpath > b.relpath) return 1;
  return 0;
}

/**
 * Opt-in Emscripten `Module.FS` `/pg6_dump` -> `POST /__pg6_dump/<relpath>` pipeline-dump syncer.
 *
 * - Disabled by default (zero overhead: no FS access, no timer, no fetch) unless
 *   `options.dumpPipelines === true` or URL search param `dumpPipelines=1|true` is present.
 * - Ensures `dumpDir` exists during `preRun` via `FS.mkdirTree` / `FS.mkdir` (ignoring `EEXIST`).
 * - Incrementally uploads new files and re-uploads updated files whose byte size changed
 *   (e.g. `manifest.json`), with bounded concurrency (`maxConcurrentUploads`).
 * - Never throws on HTTP 403/500 or network errors; increments `uploadErrors` and retries next flush.
 * - Exposes sanitized `getStats()` with zero host paths or raw shader bytes.
 */
function createMemfsPipelineDumpSyncer(options = {}) {
  const enabled = resolveDumpPipelinesEnabled(options);
  const dumpDir = normalizeMemfsDumpDir(options.dumpDir);
  const dumpEndpoint = normalizeDumpEndpoint(options.dumpEndpoint);
  const intervalMs =
    typeof options.intervalMs === 'number' &&
    Number.isFinite(options.intervalMs) &&
    options.intervalMs > 0
      ? Math.max(10, Math.floor(options.intervalMs))
      : DEFAULT_MEMFS_PIPELINE_DUMP_INTERVAL_MS;
  const maxConcurrentUploads =
    typeof options.maxConcurrentUploads === 'number' &&
    Number.isFinite(options.maxConcurrentUploads) &&
    options.maxConcurrentUploads >= 1
      ? Math.floor(options.maxConcurrentUploads)
      : DEFAULT_MEMFS_PIPELINE_DUMP_MAX_CONCURRENT_UPLOADS;

  const fetchFn =
    typeof options.fetchFn === 'function'
      ? options.fetchFn
      : typeof globalThis !== 'undefined' && typeof globalThis.fetch === 'function'
        ? (url, init) => globalThis.fetch(url, init)
        : typeof fetch === 'function'
          ? (url, init) => fetch(url, init)
          : null;
  const setIntervalFn =
    options.setIntervalFn !== undefined
      ? options.setIntervalFn
      : typeof setInterval === 'function'
        ? (fn, ms) => setInterval(fn, ms)
        : null;
  const clearIntervalFn =
    options.clearIntervalFn !== undefined
      ? options.clearIntervalFn
      : typeof clearInterval === 'function'
        ? (id) => clearInterval(id)
        : null;
  const nowFn = typeof options.nowFn === 'function' ? options.nowFn : () => Date.now();

  let explicitFs = options.fs || options.FS || null;
  const getFsFn = typeof options.getFS === 'function' ? options.getFS : null;
  let attachedModule = options.module && typeof options.module === 'object' ? options.module : null;
  const getModuleFn = typeof options.getModule === 'function' ? options.getModule : null;

  const uploadedSizes = new Map();
  let filesUploaded = 0;
  let bytesUploaded = 0;
  let uploadErrors = 0;
  let lastFlushAtMs = null;
  let pendingCount = 0;

  let timerId = null;
  let activeFlush = null;
  let queuedFlush = null;

  function resolveFs(fsOverride) {
    if (fsOverride && typeof fsOverride === 'object') {
      return fsOverride;
    }
    if (explicitFs && typeof explicitFs === 'object') {
      return explicitFs;
    }
    if (getFsFn) {
      try {
        const res = getFsFn();
        if (res && typeof res === 'object') {
          return res;
        }
      } catch {
        // Ignore getter errors.
      }
    }
    if (attachedModule && attachedModule.FS && typeof attachedModule.FS === 'object') {
      return attachedModule.FS;
    }
    if (getModuleFn) {
      try {
        const mod = getModuleFn();
        if (mod && mod.FS && typeof mod.FS === 'object') {
          return mod.FS;
        }
      } catch {
        // Ignore getter errors.
      }
    }
    return null;
  }

  function getStats() {
    return {
      enabled,
      filesUploaded,
      bytesUploaded,
      uploadErrors,
      lastFlushAtMs,
      pendingCount,
    };
  }

  function setFS(fsObj) {
    explicitFs = fsObj && typeof fsObj === 'object' ? fsObj : null;
  }

  function attachModule(mod) {
    if (mod && typeof mod === 'object') {
      attachedModule = mod;
    }
  }

  function ensureDumpDir(fsOverride) {
    if (!enabled) {
      return false;
    }
    const fsObj = resolveFs(fsOverride);
    if (!fsObj) {
      return false;
    }
    try {
      if (typeof fsObj.mkdirTree === 'function') {
        fsObj.mkdirTree(dumpDir);
        return true;
      }
      if (typeof fsObj.mkdir === 'function') {
        const parts = dumpDir.split('/').filter(Boolean);
        let cur = '';
        for (let i = 0; i < parts.length; i += 1) {
          cur += `/${parts[i]}`;
          try {
            fsObj.mkdir(cur);
          } catch (err) {
            if (!isFsExistsError(err)) {
              throw err;
            }
          }
        }
        return true;
      }
    } catch (err) {
      if (isFsExistsError(err)) {
        return true;
      }
      // Directory creation failure must never crash runtime boot.
    }
    return false;
  }

  function isDirMode(fsObj, mode) {
    if (typeof fsObj.isDir === 'function') {
      return Boolean(fsObj.isDir(mode));
    }
    return typeof mode === 'number' && (mode & 0o170000) === 0o040000;
  }

  function isFileMode(fsObj, mode) {
    if (typeof fsObj.isFile === 'function') {
      return Boolean(fsObj.isFile(mode));
    }
    return typeof mode === 'number' && (mode & 0o170000) === 0o100000;
  }

  function collectDumpFiles(fsObj) {
    if (
      !fsObj ||
      typeof fsObj.readdir !== 'function' ||
      typeof fsObj.stat !== 'function' ||
      typeof fsObj.readFile !== 'function'
    ) {
      return [];
    }

    const found = [];
    const visitedDirs = new Set();

    function walk(dirFullPath, relPrefix, depth) {
      if (depth > MAX_MEMFS_DUMP_WALK_DEPTH) {
        return;
      }
      if (visitedDirs.has(dirFullPath)) {
        return;
      }
      visitedDirs.add(dirFullPath);

      let entries;
      try {
        entries = fsObj.readdir(dirFullPath);
      } catch (err) {
        if (isFsNotFoundError(err)) {
          return;
        }
        return;
      }
      if (!Array.isArray(entries)) {
        return;
      }

      for (let i = 0; i < entries.length; i += 1) {
        const name = entries[i];
        if (typeof name !== 'string' || name === '.' || name === '..' || name.length === 0) {
          continue;
        }
        if (
          name.includes('/') ||
          name.includes('\\') ||
          name.includes('\0') ||
          name.includes(':') ||
          hasControlChar(name)
        ) {
          continue;
        }
        const candidateRel = relPrefix ? `${relPrefix}/${name}` : name;
        const cleanRel = sanitizeDumpRelPath(candidateRel);
        if (!cleanRel) {
          continue;
        }
        const fullPath = `${dirFullPath}/${name}`;
        let st;
        try {
          st = fsObj.stat(fullPath);
        } catch {
          continue;
        }
        if (!st || typeof st !== 'object') {
          continue;
        }
        if (isDirMode(fsObj, st.mode)) {
          walk(fullPath, cleanRel, depth + 1);
        } else if (isFileMode(fsObj, st.mode)) {
          const sizeBytes =
            typeof st.size === 'number' && Number.isFinite(st.size) && st.size >= 0
              ? Math.floor(st.size)
              : null;
          found.push({
            relpath: cleanRel,
            fullPath,
            sizeBytes,
          });
        }
      }
    }

    walk(dumpDir, '', 0);
    found.sort(compareDumpCandidates);
    return found;
  }

  function snapshotPendingFiles(fsObj) {
    if (!fsObj) {
      return [];
    }
    const allFiles = collectDumpFiles(fsObj);
    const snapshotted = [];
    for (let i = 0; i < allFiles.length; i += 1) {
      const item = allFiles[i];
      const prevSize = uploadedSizes.get(item.relpath);
      if (prevSize !== undefined && item.sizeBytes !== null && prevSize === item.sizeBytes) {
        continue;
      }
      const copy = {
        relpath: item.relpath,
        fullPath: item.fullPath,
        sizeBytes: item.sizeBytes,
      };
      try {
        const raw = fsObj.readFile(item.fullPath);
        copy.prefetchedBytes = toUint8ArrayCopy(raw);
      } catch {
        copy.prefetchFailed = true;
      }
      snapshotted.push(copy);
    }
    return snapshotted;
  }

  async function runFlushPass(passOptions = {}) {
    if (!enabled) {
      return getStats();
    }
    const preSnapshotted =
      passOptions && Array.isArray(passOptions.preSnapshottedItems)
        ? passOptions.preSnapshottedItems
        : null;
    const fsObj = resolveFs();
    if ((!fsObj && (!preSnapshotted || preSnapshotted.length === 0)) || !fetchFn) {
      lastFlushAtMs = nowFn();
      pendingCount = 0;
      return getStats();
    }

    const toUpload = [];
    const seenRelpaths = new Set();
    if (fsObj) {
      const allFiles = collectDumpFiles(fsObj);
      for (let i = 0; i < allFiles.length; i += 1) {
        const item = allFiles[i];
        const prevSize = uploadedSizes.get(item.relpath);
        if (prevSize !== undefined && item.sizeBytes !== null && prevSize === item.sizeBytes) {
          continue;
        }
        seenRelpaths.add(item.relpath);
        toUpload.push(item);
      }
    }
    if (preSnapshotted) {
      for (let i = 0; i < preSnapshotted.length; i += 1) {
        const snap = preSnapshotted[i];
        const snapSize =
          snap.prefetchedBytes instanceof Uint8Array
            ? snap.prefetchedBytes.byteLength
            : snap.sizeBytes;
        const prevSize = uploadedSizes.get(snap.relpath);
        if (prevSize !== undefined && snapSize !== null && prevSize === snapSize) {
          continue;
        }
        if (!seenRelpaths.has(snap.relpath)) {
          seenRelpaths.add(snap.relpath);
          toUpload.push(snap);
        }
      }
      toUpload.sort(compareDumpCandidates);
    }

    if (toUpload.length === 0) {
      pendingCount = 0;
      lastFlushAtMs = nowFn();
      return getStats();
    }

    const snapshotFiles = Boolean(passOptions && passOptions.snapshotFiles === true);
    if (snapshotFiles && fsObj) {
      // Synchronously read all pending MEMFS files before the first await so teardown/onAbort
      // cannot lose queued files beyond maxConcurrentUploads.
      for (let i = 0; i < toUpload.length; i += 1) {
        const item = toUpload[i];
        if (item.prefetchedBytes instanceof Uint8Array || item.prefetchFailed) {
          continue;
        }
        try {
          const raw = fsObj.readFile(item.fullPath);
          item.prefetchedBytes = toUint8ArrayCopy(raw);
        } catch {
          item.prefetchFailed = true;
        }
      }
    }

    let remainingOrInFlight = toUpload.length;
    let failedInPass = 0;
    pendingCount = remainingOrInFlight;
    let cursor = 0;

    async function uploadOne(item) {
      let bytes;
      if (item.prefetchedBytes instanceof Uint8Array) {
        bytes = item.prefetchedBytes;
      } else if (item.prefetchFailed) {
        uploadErrors += 1;
        failedInPass += 1;
        remainingOrInFlight = Math.max(0, remainingOrInFlight - 1);
        pendingCount = remainingOrInFlight + failedInPass;
        return;
      } else {
        try {
          const raw = fsObj.readFile(item.fullPath);
          bytes = toUint8ArrayCopy(raw);
        } catch {
          uploadErrors += 1;
          failedInPass += 1;
          remainingOrInFlight = Math.max(0, remainingOrInFlight - 1);
          pendingCount = remainingOrInFlight + failedInPass;
          return;
        }
      }

      const actualSize = bytes.byteLength;
      if (item.sizeBytes === null && uploadedSizes.get(item.relpath) === actualSize) {
        remainingOrInFlight = Math.max(0, remainingOrInFlight - 1);
        pendingCount = remainingOrInFlight + failedInPass;
        return;
      }

      const encodedRel = item.relpath
        .split('/')
        .map((seg) => encodeURIComponent(seg))
        .join('/');
      const url = `${dumpEndpoint}${encodedRel}`;

      try {
        const resp = await fetchFn(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: bytes,
        });
        const ok = Boolean(
          resp &&
            resp.ok !== false &&
            (typeof resp.status !== 'number' || (resp.status >= 200 && resp.status < 300)),
        );
        // Drain the small JSON response body to EOF so HTTP/1.1 keep-alive returns the TCP
        // socket to Chrome's idle pool instead of tearing it down on GC and exhausting
        // Windows's 16,384 ephemeral TCP ports during large pipeline dumps (#663, #1077).
        if (resp && !resp.bodyUsed) {
          try {
            if (typeof resp.arrayBuffer === 'function') {
              await resp.arrayBuffer();
            } else if (typeof resp.text === 'function') {
              await resp.text();
            } else if (resp.body && typeof resp.body.cancel === 'function') {
              await resp.body.cancel();
            }
          } catch {
            try {
              if (resp.body && typeof resp.body.cancel === 'function') {
                await resp.body.cancel();
              }
            } catch {
              // Ignore response body cleanup errors on aborted or mocked streams.
            }
          }
        }
        if (!ok) {
          uploadErrors += 1;
          failedInPass += 1;
        } else {
          uploadedSizes.set(item.relpath, actualSize);
          filesUploaded += 1;
          bytesUploaded += actualSize;
        }
      } catch {
        uploadErrors += 1;
        failedInPass += 1;
      } finally {
        remainingOrInFlight = Math.max(0, remainingOrInFlight - 1);
        pendingCount = remainingOrInFlight + failedInPass;
      }
    }

    const workerCount = Math.min(maxConcurrentUploads, toUpload.length);
    const workers = [];
    for (let w = 0; w < workerCount; w += 1) {
      workers.push(
        (async () => {
          while (cursor < toUpload.length) {
            const idx = cursor;
            cursor += 1;
            await uploadOne(toUpload[idx]);
          }
        })(),
      );
    }

    await Promise.all(workers);
    lastFlushAtMs = nowFn();
    return getStats();
  }

  function startFlushPass(passOptions) {
    const p = runFlushPass(passOptions).finally(() => {
      if (activeFlush === p) {
        activeFlush = null;
      }
    });
    activeFlush = p;
    return p;
  }

  let queuedSnapshotFiles = false;
  let queuedPreSnapshottedByRelpath = null;

  function mergePreSnapshottedItems(items) {
    if (!Array.isArray(items) || items.length === 0) {
      return;
    }
    if (!queuedPreSnapshottedByRelpath) {
      queuedPreSnapshottedByRelpath = new Map();
    }
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i];
      if (!item || !item.relpath) {
        continue;
      }
      const existing = queuedPreSnapshottedByRelpath.get(item.relpath);
      if (
        !existing ||
        item.prefetchedBytes instanceof Uint8Array ||
        !(existing.prefetchedBytes instanceof Uint8Array)
      ) {
        queuedPreSnapshottedByRelpath.set(item.relpath, item);
      }
    }
  }

  function flush(flushOptions) {
    if (!enabled) {
      return Promise.resolve(getStats());
    }
    const wantsSnapshot = Boolean(flushOptions && flushOptions.snapshotFiles === true);
    if (!activeFlush) {
      return startFlushPass(wantsSnapshot ? { snapshotFiles: true } : undefined);
    }
    if (wantsSnapshot) {
      queuedSnapshotFiles = true;
      const fsObj = resolveFs();
      if (fsObj) {
        mergePreSnapshottedItems(snapshotPendingFiles(fsObj));
      }
    }
    if (!queuedFlush) {
      queuedFlush = activeFlush.then(() => {
        const nextSnapshot = queuedSnapshotFiles;
        const nextPreSnapshotted = queuedPreSnapshottedByRelpath
          ? Array.from(queuedPreSnapshottedByRelpath.values())
          : null;
        queuedSnapshotFiles = false;
        queuedPreSnapshottedByRelpath = null;
        queuedFlush = null;
        return startFlushPass(
          nextSnapshot || (nextPreSnapshotted && nextPreSnapshotted.length > 0)
            ? {
                snapshotFiles: nextSnapshot,
                preSnapshottedItems: nextPreSnapshotted,
              }
            : undefined,
        );
      });
    }
    return queuedFlush;
  }

  function clearTimer() {
    if (timerId !== null && typeof clearIntervalFn === 'function') {
      try {
        clearIntervalFn(timerId);
      } catch {
        // Ignore timer cleanup errors.
      }
      timerId = null;
    }
  }

  function start(modOrFs) {
    if (!enabled) {
      return false;
    }
    if (modOrFs && typeof modOrFs === 'object') {
      if (modOrFs.FS && typeof modOrFs.FS === 'object') {
        attachModule(modOrFs);
      } else if (typeof modOrFs.readdir === 'function' || typeof modOrFs.mkdir === 'function') {
        setFS(modOrFs);
      }
    }
    ensureDumpDir();
    if (timerId === null && typeof setIntervalFn === 'function') {
      timerId = setIntervalFn(() => {
        flush().catch(() => {});
      }, intervalMs);
    }
    return true;
  }

  function stop() {
    clearTimer();
    if (!enabled) {
      return Promise.resolve(getStats());
    }
    return flush({ snapshotFiles: true });
  }

  return {
    enabled,
    start,
    stop,
    flush,
    ensureDumpDir,
    attachModule,
    setFS,
    getStats,
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
  const resolveTargetGpuWorker = () =>
    options.gpuWorker !== undefined ? options.gpuWorker : gpuWorker;

  // G66 / W360 read-back: the GPU worker writes ReadbackBuffer / ReadbackTexture results straight
  // into the runtime's shared memory, so it must receive the current SharedArrayBuffer on ALL live
  // paths (including SPSC gpu_ring, instantiateWasm, preRun, onRuntimeInitialized, and after any
  // WebAssembly.Memory.grow() — O(1) reference equality check when unchanged).
  let instantiatedMemory = null;
  let attachedGuestMemory = null;
  let attachedGuestMemoryWorker = null;
  const resolveRuntimeMemory = (modOverride, memoryOverride) => {
    const targetMod =
      modOverride && typeof modOverride === 'object' ? modOverride : boundModule;
    return (
      memoryOverride ||
      options.memory ||
      (targetMod &&
        (targetMod.wasmMemory ||
          (targetMod.HEAPU8 && targetMod.HEAPU8.buffer) ||
          (targetMod.HEAP32 && targetMod.HEAP32.buffer) ||
          (targetMod.asm && targetMod.asm.memory) ||
          (targetMod.wasmExports && targetMod.wasmExports.memory))) ||
      instantiatedMemory ||
      null
    );
  };
  const attachGuestMemoryToGpuWorker = (modOverride, memoryOverride) => {
    const targetWorker = resolveTargetGpuWorker();
    if (!targetWorker || typeof targetWorker.postMessage !== 'function') {
      return false;
    }
    let buffer = null;
    try {
      buffer = resolveBuffer(resolveRuntimeMemory(modOverride, memoryOverride));
    } catch {
      buffer = null;
    }
    if (
      !buffer ||
      (buffer === attachedGuestMemory && targetWorker === attachedGuestMemoryWorker) ||
      typeof SharedArrayBuffer === 'undefined' ||
      !(buffer instanceof SharedArrayBuffer)
    ) {
      return false;
    }
    attachedGuestMemory = buffer;
    attachedGuestMemoryWorker = targetWorker;
    targetWorker.postMessage({ type: 'attach-guest-memory', buffer });
    return true;
  };
  // Attach immediately when options.memory is already a SharedArrayBuffer / shared Wasm memory.
  attachGuestMemoryToGpuWorker();

  const wgpuSubmit =
    typeof options.wgpuSubmit === 'function'
      ? options.wgpuSubmit
      : (bytes) => {
          const targetWorker = resolveTargetGpuWorker();
          if (!targetWorker || typeof targetWorker.postMessage !== 'function') return;
          const cmdBuffer = toTransferableArrayBuffer(bytes);
          if (!cmdBuffer) return;
          attachGuestMemoryToGpuWorker();
          targetWorker.postMessage({ cmdBuffer }, [cmdBuffer]);
        };

  const userLogFn =
    typeof options.log === 'function' ? options.log : (msg) => activeReportFn(`[wasm] ${msg}`);
  const userOnFirstPresent =
    typeof options.onFirstPresent === 'function' ? options.onFirstPresent : null;
  const userOnFaultRecord =
    typeof options.onFaultRecord === 'function' ? options.onFaultRecord : null;
  const userPrint = typeof options.print === 'function' ? options.print : null;
  const userPrintErr = typeof options.printErr === 'function' ? options.printErr : null;
  const configConsoleObj =
    options.console && typeof options.console.error === 'function'
      ? options.console
      : typeof console !== 'undefined'
        ? console
        : null;
  const configPostToMain =
    typeof options.postToMain === 'function'
      ? options.postToMain
      : typeof coordinatorPostToMain === 'function'
        ? coordinatorPostToMain
        : null;
  let audioRingController = null;
  let lastModuleFaultRecord = null;
  let lastModuleFaultRecordLine = null;
  let lastModuleFaultRecordConsoleLine = null;
  let flushPipelineDumpInternal = () => Promise.resolve(null);

  const captureModuleFaultRecord = (rawText, meta = {}) => {
    if (typeof rawText !== 'string' || !isFaultRecordLine(rawText)) {
      return null;
    }
    const parsed = parseFaultRecordLine(rawText);
    if (!parsed) {
      return null;
    }
    const formattedLine = rawText.includes('[wasm]') ? rawText : `[wasm] ${rawText}`;
    lastModuleFaultRecord = cloneFaultRecord(parsed);
    lastModuleFaultRecordLine = formattedLine;

    const alreadyEmittedConsole = lastModuleFaultRecordConsoleLine === formattedLine;
    if (!alreadyEmittedConsole) {
      lastModuleFaultRecordConsoleLine = formattedLine;
      emitFaultRecordConsoleError(formattedLine, configConsoleObj);
    }

    if (userOnFaultRecord) {
      try {
        userOnFaultRecord(cloneFaultRecord(parsed), {
          rawLine: formattedLine,
          source: meta.source || 'wasm',
          onAbort: meta.onAbort === true,
        });
      } catch {
        // Fault-record observer errors must never interfere with capture.
      }
    }

    recordCoordinatorFaultRecord(parsed, {
      rawLine: formattedLine,
      label: formattedLine,
      source: meta.source || 'wasm',
      onAbort: meta.onAbort === true,
      console: configConsoleObj,
      consoleEmitted: true,
      postToMain: configPostToMain,
      flushDumps: false,
    });

    try {
      const p = flushPipelineDumpInternal({ snapshotFiles: true });
      if (p && typeof p.catch === 'function') {
        p.catch(() => {});
      }
    } catch {
      // Dump flush must never disrupt fault-record delivery.
    }

    return cloneFaultRecord(parsed);
  };

  const logFn = (msg) => {
    if (typeof msg === 'string' && isFaultRecordLine(msg)) {
      captureModuleFaultRecord(msg, { source: 'host_log' });
    }
    if (
      audioRingController &&
      !audioRingController.isAttached() &&
      typeof msg === 'string' &&
      GUEST_AUDIO_MILESTONE_LOG_RE.test(msg)
    ) {
      try {
        audioRingController.notifyGuestMilestone('guest_audio_log');
      } catch {
        // Audio ring milestone probe must never interfere with host_log.
      }
    }
    return userLogFn(msg);
  };

  const handleFirstPresent = (detail) => {
    if (audioRingController && !audioRingController.isAttached()) {
      try {
        audioRingController.notifyGuestMilestone('first_present');
      } catch {
        // Audio ring milestone probe must never interfere with FIRST_PRESENT.
      }
    }
    if (userOnFirstPresent) {
      return userOnFirstPresent(detail);
    }
    return undefined;
  };

  const activeHostImports =
    options.hostImports && typeof options.hostImports === 'object'
      ? options.hostImports
      : createHostImports({
          memory: options.memory || null,
          getMemory: () => {
            const mem = resolveRuntimeMemory();
            attachGuestMemoryToGpuWorker(null, mem);
            return mem;
          },
          log: logFn,
          report: activeReportFn,
          wgpuSubmit,
          httpfsClient: activeHttpfsClient,
          userdataStore: activeUserdataStore,
          onFirstPresent: handleFirstPresent,
          emitFirstPresentLog: options.emitFirstPresentLog === true,
        });

  const rawManifestInput =
    options.guestManifest !== undefined ? options.guestManifest : options.manifest;
  const manifestSummary =
    rawManifestInput !== undefined && rawManifestInput !== null
      ? parseGuestManifest(rawManifestInput)
      : null;

  const manifestRuntimeScriptUrl =
    manifestSummary && manifestSummary.runtimeScriptUrl
      ? resolveSideModuleUrl(
          manifestSummary.runtimeScriptUrl,
          DEFAULT_RUNTIME_SCRIPT_RELATIVE,
          options.locateFileBase,
        )
      : undefined;
  const resolvedRuntimeScriptUrl = resolveRuntimeScriptUrl(
    options.runtimeScriptUrl || manifestRuntimeScriptUrl || undefined,
    options.locationHref,
  );
  const rawDynamicLibraries = sanitizeSideModuleUrls(options.dynamicLibraries);
  const rawSideModuleUrls = sanitizeSideModuleUrls(options.sideModuleUrls);
  const manifestSideModules = manifestSummary ? manifestSummary.sideModules : [];
  const combinedDynamicLibraries = [];
  const seenSideModules = new Set();
  const addSideModuleEntry = (entry) => {
    const key = entry.toLowerCase();
    const baseKey = extractWasmBasename(entry).toLowerCase();
    if (seenSideModules.has(key) || (baseKey && seenSideModules.has(baseKey))) {
      return;
    }
    seenSideModules.add(key);
    if (baseKey) {
      seenSideModules.add(baseKey);
    }
    combinedDynamicLibraries.push(entry);
  };
  for (let i = 0; i < rawDynamicLibraries.length; i += 1) {
    addSideModuleEntry(rawDynamicLibraries[i]);
  }
  for (let i = 0; i < rawSideModuleUrls.length; i += 1) {
    addSideModuleEntry(rawSideModuleUrls[i]);
  }
  if (combinedDynamicLibraries.length === 0 && manifestSideModules.length > 0) {
    for (let i = 0; i < manifestSideModules.length; i += 1) {
      addSideModuleEntry(manifestSideModules[i]);
    }
  }
  const resolvedSideModuleUrls = combinedDynamicLibraries.map((entry) =>
    resolveSideModuleUrl(entry, resolvedRuntimeScriptUrl, options.locateFileBase),
  );

  const locateFile = createRuntimeLocateFile(resolvedRuntimeScriptUrl, options.locateFileBase, {
    reportFn: activeReportFn,
    onProgress: typeof options.onProgress === 'function' ? options.onProgress : null,
    sideModuleNames: combinedDynamicLibraries,
    expectedSideModuleCount:
      manifestSummary && combinedDynamicLibraries.length === 0
        ? manifestSummary.sideModuleCount
        : combinedDynamicLibraries.length,
    indirectDispatch: manifestSummary ? manifestSummary.indirectDispatch : undefined,
    wasmMetrics: manifestSummary ? manifestSummary.wasmMetrics : null,
  });

  const pipelineDumpSyncer =
    options.pipelineDumpSyncer && typeof options.pipelineDumpSyncer === 'object'
      ? options.pipelineDumpSyncer
      : createMemfsPipelineDumpSyncer({
          dumpPipelines: options.dumpPipelines,
          dumpDir: options.dumpDir,
          dumpEndpoint: options.dumpEndpoint,
          intervalMs:
            options.dumpIntervalMs !== undefined ? options.dumpIntervalMs : options.intervalMs,
          maxConcurrentUploads:
            options.dumpMaxConcurrentUploads !== undefined
              ? options.dumpMaxConcurrentUploads
              : options.maxConcurrentUploads,
          fetchFn:
            typeof options.dumpFetchFn === 'function'
              ? options.dumpFetchFn
              : typeof options.fetchFn === 'function'
                ? options.fetchFn
                : undefined,
          locationHref: options.locationHref,
          locationSearch: options.locationSearch,
          setIntervalFn: options.dumpSetIntervalFn,
          clearIntervalFn: options.dumpClearIntervalFn,
          nowFn: options.dumpNowFn,
          getModule: () => boundModule,
        });

  let pipelineDumpSyncerStarted = false;
  const attachPipelineDumpSyncer = (mod) => {
    if (!pipelineDumpSyncer || !pipelineDumpSyncer.enabled || pipelineDumpSyncerStarted) {
      return;
    }
    const targetMod = mod && typeof mod === 'object' ? mod : boundModule;
    if (!targetMod || !targetMod.FS) {
      return;
    }
    pipelineDumpSyncerStarted = true;
    try {
      pipelineDumpSyncer.start(targetMod);
    } catch {
      // Pipeline dump sync setup must never block or crash boot.
    }
  };

  const stopPipelineDumpSyncer = () => {
    if (!pipelineDumpSyncer || !pipelineDumpSyncer.enabled) {
      return Promise.resolve(pipelineDumpSyncer ? pipelineDumpSyncer.getStats() : null);
    }
    if (boundModule && boundModule.FS && typeof pipelineDumpSyncer.attachModule === 'function') {
      try {
        pipelineDumpSyncer.attachModule(boundModule);
      } catch {
        // Ignore attach failures during teardown flush.
      }
    }
    try {
      return pipelineDumpSyncer.stop();
    } catch {
      return Promise.resolve(pipelineDumpSyncer.getStats());
    }
  };

  flushPipelineDumpInternal = (flushOpts) => {
    if (!pipelineDumpSyncer || !pipelineDumpSyncer.enabled) {
      return Promise.resolve(pipelineDumpSyncer ? pipelineDumpSyncer.getStats() : null);
    }
    if (boundModule && boundModule.FS && typeof pipelineDumpSyncer.attachModule === 'function') {
      try {
        pipelineDumpSyncer.attachModule(boundModule);
      } catch {
        // Ignore attach failures during explicit flush.
      }
    }
    try {
      return pipelineDumpSyncer.flush(flushOpts);
    } catch {
      return Promise.resolve(pipelineDumpSyncer.getStats());
    }
  };

  const preRun = Array.isArray(options.preRun)
    ? [
        (mod) => {
          if (mod && !boundModule) {
            boundModule = mod;
          }
          attachGuestMemoryToGpuWorker(mod);
          attachPipelineDumpSyncer(mod);
        },
        ...options.preRun,
      ]
    : [
        (mod) => {
          boundModule = mod || null;
          if (typeof globalThis !== 'undefined') {
            globalThis.__playgta6RuntimeModule = boundModule;
          }
          if (
            mod &&
            (mod.wasmMemory || (mod.HEAPU8 && mod.HEAPU8.buffer)) &&
            typeof activeHostImports.setMemory === 'function'
          ) {
            activeHostImports.setMemory(mod.wasmMemory || mod.HEAPU8.buffer);
          }
          attachGuestMemoryToGpuWorker(mod);
          attachPipelineDumpSyncer(mod);
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

  const activeInputSab = options.inputSab !== undefined ? options.inputSab : inputSab;
  let activeInputBridge = null;

  const stopInputBridge = () => {
    if (activeInputBridge) {
      try {
        activeInputBridge.stop();
      } catch {
        // Ignore input bridge teardown errors.
      }
      activeInputBridge = null;
    }
  };

  const attachInputBridge = (mod) => {
    if (activeInputBridge && activeInputBridge.active) {
      return activeInputBridge;
    }
    const targetMod = mod && typeof mod === 'object' ? mod : boundModule;
    if (!activeInputSab || !targetMod) {
      return null;
    }
    const bridge = startInputSabBridge({
      inputSab: activeInputSab,
      module: targetMod,
      getMemory: () => {
        const mem = resolveRuntimeMemory(targetMod);
        attachGuestMemoryToGpuWorker(targetMod, mem);
        return mem;
      },
      pollIntervalMs: options.inputPollIntervalMs,
    });
    if (bridge && bridge.active) {
      activeInputBridge = bridge;
      return bridge;
    }
    return null;
  };

  const audioRingRetryOpts =
    options.audioRingRetry && typeof options.audioRingRetry === 'object'
      ? options.audioRingRetry
      : {};
  const resolveConfiguredAudioWorker = () => {
    if (typeof options.getAudioWorker === 'function') {
      return options.getAudioWorker();
    }
    if (options.audioWorker !== undefined) {
      return options.audioWorker;
    }
    return audioWorker;
  };
  const checkConfiguredAudioWorkerReady = () => {
    if (typeof options.isAudioWorkerReady === 'function') {
      try {
        return Boolean(options.isAudioWorkerReady());
      } catch {
        return false;
      }
    }
    if (options.audioWorker !== undefined || typeof options.getAudioWorker === 'function') {
      return true;
    }
    return audioWorkerReady;
  };

  audioRingController = createAudioRingAttachmentController({
    getModule: () => boundModule,
    getAudioWorker: resolveConfiguredAudioWorker,
    isAudioWorkerReady: checkConfiguredAudioWorkerReady,
    getAttachOptions: (targetMod) => ({
      getMemory: () =>
        resolveRuntimeMemory() || resolveRuntimeMemoryBuffer(targetMod, null) || null,
      pointerSize: options.audioRingPointerSize,
      autoPump: options.audioRingAutoPump,
    }),
    retryIntervalMs:
      options.audioRingRetryIntervalMs !== undefined
        ? options.audioRingRetryIntervalMs
        : audioRingRetryOpts.retryIntervalMs !== undefined
          ? audioRingRetryOpts.retryIntervalMs
          : options.retryIntervalMs,
    maxRetries:
      options.audioRingMaxRetries !== undefined
        ? options.audioRingMaxRetries
        : audioRingRetryOpts.maxRetries !== undefined
          ? audioRingRetryOpts.maxRetries
          : options.maxRetries,
    autoRetry: options.audioRingAutoRetry !== false && audioRingRetryOpts.autoRetry !== false,
    retryOnNoExport:
      options.audioRingRetryOnNoExport === true || audioRingRetryOpts.retryOnNoExport === true,
    setTimeoutFn:
      typeof options.audioRingSetTimeoutFn === 'function'
        ? options.audioRingSetTimeoutFn
        : typeof audioRingRetryOpts.setTimeoutFn === 'function'
          ? audioRingRetryOpts.setTimeoutFn
          : undefined,
    clearTimeoutFn:
      typeof options.audioRingClearTimeoutFn === 'function'
        ? options.audioRingClearTimeoutFn
        : typeof audioRingRetryOpts.clearTimeoutFn === 'function'
          ? audioRingRetryOpts.clearTimeoutFn
          : undefined,
    onAttached: (res, telemetry) => {
      try {
        if (res && res.descriptor) {
          activeReportFn(
            `[audio] audio_ring_attached=true sampleRate=${res.descriptor.sampleRate} channels=${res.descriptor.channels} capacity=${res.descriptor.capacity} retries=${telemetry.audioRingRetryCount}`,
          );
        }
      } catch {
        // Audio attach telemetry report must never break runtime execution.
      }
    },
  });

  const resetAudioRingAttachment = () => {
    audioRingController.reset();
  };

  const attachAudioRing = (mod, workerOverride, attachOpts) => {
    if (mod && typeof mod === 'object' && !boundModule) {
      boundModule = mod;
    }
    return audioRingController.attach(mod, workerOverride, attachOpts);
  };

  const notifyAudioWorkerReady = (workerOverride) =>
    audioRingController.onAudioWorkerReady(workerOverride);

  const onAudioWorkerReady = notifyAudioWorkerReady;

  const setAudioWorker = (nextWorker, readyOverride) =>
    audioRingController.setAudioWorker(nextWorker, readyOverride);

  const notifyGuestMilestone = (milestoneName) =>
    audioRingController.notifyGuestMilestone(milestoneName);

  const scheduleConfigAudioRingRetry = (retryOpts) =>
    audioRingController.schedule(retryOpts);

  const stopAudioRingRetry = () => {
    audioRingController.stop();
  };

  const getAudioRingTelemetry = () => audioRingController.getTelemetry();

  let activePrewarmBridge = null;

  const stopPrewarmBridge = (stopOpts = {}) => {
    const bridge = activePrewarmBridge;
    if (!bridge) {
      return Promise.resolve(null);
    }
    const shouldSave = Boolean(stopOpts && stopOpts.save);
    if (shouldSave && typeof bridge.stop === 'function') {
      try {
        const p = bridge.stop({ save: true });
        activePrewarmBridge = null;
        return Promise.resolve(p).finally(() => {
          try {
            bridge.destroy();
          } catch {
            // Ignore secondary destroy errors.
          }
        });
      } catch {
        activePrewarmBridge = null;
      }
    }
    activePrewarmBridge = null;
    try {
      bridge.destroy();
    } catch {
      // Ignore prewarm bridge teardown errors.
    }
    return Promise.resolve(null);
  };

  const attachPrewarmBridge = (mod, attachOpts = {}) => {
    if (options.prewarmEnabled === false || !prewarmBridgeApi || !prewarmBridgeApi.PrewarmBridge) {
      return null;
    }
    const targetMod = mod && typeof mod === 'object' ? mod : boundModule;
    if (!targetMod) {
      return null;
    }
    if (activePrewarmBridge && activePrewarmBridge.available) {
      return activePrewarmBridge;
    }
    const mergedOpts = Object.assign(
      {
        module: targetMod,
        getMemory: () =>
          resolveRuntimeMemory(targetMod) || resolveRuntimeMemoryBuffer(targetMod, null) || null,
        gpuWorker:
          typeof options.getGpuWorker === 'function'
            ? options.getGpuWorker()
            : options.gpuWorker !== undefined
              ? options.gpuWorker
              : gpuWorker,
        buildTask: options.prewarmBuildTask,
        maxInFlight: options.prewarmMaxInFlight,
        pollIntervalMs: options.prewarmPollIntervalMs,
        store: options.prewarmStore,
        storeOptions: options.prewarmStoreOptions,
        storeKey: options.prewarmStoreKey,
        isIdle:
          typeof options.prewarmIsIdle === 'function'
            ? options.prewarmIsIdle
            : () =>
                _inFlightGpuPresents === 0 &&
                _pendingCoalescedPresent === null &&
                pendingRecycleTxPorts.length === 0,
        wasm64: options.prewarmWasm64,
        pointerSize: options.prewarmPointerSize,
        scratchOffset: options.prewarmScratchOffset,
        scratchSize: options.prewarmScratchSize,
        alloc: options.prewarmAlloc,
        free: options.prewarmFree,
        now: options.prewarmNow,
        onProgress: options.onPrewarmProgress,
      },
      attachOpts && typeof attachOpts === 'object' ? attachOpts : {},
    );
    const bridge = createRuntimePrewarmBridge(targetMod, mergedOpts);
    if (!bridge || !bridge.available) {
      return null;
    }
    activePrewarmBridge = bridge;
    if (options.prewarmAutoLoad !== false) {
      Promise.resolve(bridge.loadPersistedList())
        .then(() => {
          if (activePrewarmBridge === bridge && options.prewarmAutoStart !== false) {
            bridge.start();
          }
        })
        .catch(() => {
          if (activePrewarmBridge === bridge && options.prewarmAutoStart !== false) {
            try {
              bridge.start();
            } catch {
              // Non-fatal prewarm start failure.
            }
          }
        });
    } else if (options.prewarmAutoStart !== false) {
      bridge.start();
    }
    return bridge;
  };

  const getPrewarmProgress = () => {
    if (activePrewarmBridge) {
      return activePrewarmBridge.readProgress();
    }
    return readPrewarmProgress(boundModule, {
      getMemory: () =>
        resolveRuntimeMemory(boundModule) || resolveRuntimeMemoryBuffer(boundModule, null) || null,
      wasm64: options.prewarmWasm64,
      pointerSize: options.prewarmPointerSize,
      scratchOffset: options.prewarmScratchOffset,
      scratchSize: options.prewarmScratchSize,
      alloc: options.prewarmAlloc,
      free: options.prewarmFree,
    });
  };

  const getPrewarmDiagnostics = () =>
    activePrewarmBridge ? activePrewarmBridge.getDiagnostics() : null;

  const savePrewarmList = (saveOpts) => {
    if (activePrewarmBridge) {
      return activePrewarmBridge.saveWarmList(saveOpts);
    }
    const bytes = savePrewarmWarmList(boundModule, {
      getMemory: () =>
        resolveRuntimeMemory(boundModule) || resolveRuntimeMemoryBuffer(boundModule, null) || null,
      wasm64: options.prewarmWasm64,
      pointerSize: options.prewarmPointerSize,
      scratchOffset: options.prewarmScratchOffset,
      scratchSize: options.prewarmScratchSize,
      alloc: options.prewarmAlloc,
      free: options.prewarmFree,
    });
    return Promise.resolve({
      ok: bytes !== null,
      bytes,
      byteLength: bytes ? bytes.byteLength : 0,
      persisted: false,
    });
  };

  const handlePrewarmMessage = (msg) =>
    activePrewarmBridge ? activePrewarmBridge.handleWorkerMessage(msg) : false;

  let runtimeExportsSummary = null;
  let instantiatedWasmHandler = null;

  const destroyRuntimeConfig = () => {
    const drainPromise = stopPipelineDumpSyncer();
    if (drainPromise && typeof drainPromise.catch === 'function') {
      drainPromise.catch(() => {});
    }
    stopPrewarmBridge({ save: false });
    stopInputBridge();
    audioRingController.destroy();
    if (
      instantiatedWasmHandler &&
      typeof instantiatedWasmHandler.releasePendingWireBytes === 'function'
    ) {
      instantiatedWasmHandler.releasePendingWireBytes();
    }
    boundModule = null;
    instantiatedMemory = null;
    attachedGuestMemory = null;
    attachedGuestMemoryWorker = null;
    runtimeExportsSummary = null;
  };

  const resetRuntimeConfig = () => {
    stopPrewarmBridge({ save: false });
    stopInputBridge();
    audioRingController.reset();
    attachedGuestMemory = null;
    attachedGuestMemoryWorker = null;
  };

  const userOnRuntimeInitialized =
    typeof options.onRuntimeInitialized === 'function' ? options.onRuntimeInitialized : null;
  const userOnExit = typeof options.onExit === 'function' ? options.onExit : null;
  const userOnAbort = typeof options.onAbort === 'function' ? options.onAbort : null;

  const print = function onPlayGta6Print(text) {
    const isFault = typeof text === 'string' && isFaultRecordLine(text);
    if (isFault) {
      const captured = captureModuleFaultRecord(text, { source: 'print' });
      if (captured && !userPrint) {
        try {
          activeReportFn(text.includes('[wasm]') ? text : `[wasm] ${text}`);
        } catch {
          // Ignore progress report failures during print capture.
        }
      }
    }
    if (userPrint) {
      return userPrint.apply(this, arguments);
    }
    if (!isFault && configConsoleObj && typeof configConsoleObj.log === 'function') {
      try {
        configConsoleObj.log(text);
      } catch {
        // Ignore console.log failures.
      }
    }
    return undefined;
  };

  const printErr = function onPlayGta6PrintErr(text) {
    const isFault = typeof text === 'string' && isFaultRecordLine(text);
    if (isFault) {
      captureModuleFaultRecord(text, { source: 'printErr' });
      try {
        activeReportFn(text.includes('[wasm]') ? text : `[wasm] ${text}`);
      } catch {
        // Ignore progress report failures during printErr capture.
      }
    }
    if (userPrintErr) {
      return userPrintErr.apply(this, arguments);
    }
    if (!isFault && configConsoleObj && typeof configConsoleObj.error === 'function') {
      try {
        configConsoleObj.error(text);
      } catch {
        // Ignore console.error failures.
      }
    }
    return undefined;
  };

  const onRuntimeInitialized = function onPlayGta6RuntimeInitialized() {
    const mod = this && typeof this === 'object' ? this : boundModule;
    if (mod && !boundModule) {
      boundModule = mod;
    }
    // W371 (#663): Inspect fast-boot exports cleanly without expecting or calling `_main`.
    if (mod && typeof mod === 'object') {
      runtimeExportsSummary = inspectRuntimeExports(mod);
    }
    // W351 (#1077): Drop raw Emscripten `Module.wasmBinary` wire bytes once compiled/initialized.
    if (mod && typeof mod === 'object' && Object.prototype.hasOwnProperty.call(mod, 'wasmBinary')) {
      const binInfo = getBinaryWireInfo(mod.wasmBinary);
      if (binInfo && isWritableOwnProp(mod, 'wasmBinary')) {
        mod.wasmBinary = null;
        if (binInfo.byteLength > 0) {
          recordDroppedWasmWireBytes(binInfo.byteLength, 1);
        }
      }
    }
    attachGuestMemoryToGpuWorker(mod);
    attachPipelineDumpSyncer(mod);
    attachInputBridge(mod);
    attachAudioRing(mod);
    try {
      attachPrewarmBridge(mod);
    } catch {
      // Prewarm bridge initialization must never block runtime boot.
    }
    if (userOnRuntimeInitialized) {
      return userOnRuntimeInitialized.apply(this, arguments);
    }
    return undefined;
  };

  const onExit = function onPlayGta6RuntimeExit() {
    // Snapshot and flush any pending MEMFS /pg6_dump files and warm list BEFORE clearing boundModule.
    const drainPromise = stopPipelineDumpSyncer();
    if (drainPromise && typeof drainPromise.catch === 'function') {
      drainPromise.catch(() => {});
    }
    const prewarmSavePromise = stopPrewarmBridge({ save: true });
    if (prewarmSavePromise && typeof prewarmSavePromise.catch === 'function') {
      prewarmSavePromise.catch(() => {});
    }
    stopInputBridge();
    resetAudioRingAttachment();
    if (
      instantiatedWasmHandler &&
      typeof instantiatedWasmHandler.releasePendingWireBytes === 'function'
    ) {
      instantiatedWasmHandler.releasePendingWireBytes();
    }
    boundModule = null;
    instantiatedMemory = null;
    attachedGuestMemory = null;
    attachedGuestMemoryWorker = null;
    runtimeExportsSummary = null;
    if (userOnExit) {
      return userOnExit.apply(this, arguments);
    }
    return undefined;
  };

  const onAbort = function onPlayGta6RuntimeAbort(what) {
    // Immediately flush MEMFS /pg6_dump BEFORE teardown clears boundModule.
    const drainPromise = stopPipelineDumpSyncer();
    if (drainPromise && typeof drainPromise.catch === 'function') {
      drainPromise.catch(() => {});
    }
    const whatText = what !== undefined && what !== null ? String(what) : '';
    if (whatText && isFaultRecordLine(whatText)) {
      captureModuleFaultRecord(whatText, { source: 'onAbort', onAbort: true });
      try {
        activeReportFn(whatText.includes('[wasm]') ? whatText : `[wasm] ${whatText}`);
      } catch {
        // Ignore progress report failures during onAbort capture.
      }
    } else {
      const latched = lastModuleFaultRecord || lastCoordinatorFaultRecord;
      if (latched) {
        const latchedLine =
          lastModuleFaultRecordLine ||
          lastCoordinatorFaultRecordLine ||
          formatFaultRecordConsoleLine(null, latched);
        recordCoordinatorFaultRecord(latched, {
          rawLine: latchedLine,
          label: latchedLine,
          source: 'onAbort',
          onAbort: true,
          console: configConsoleObj,
          postToMain: configPostToMain,
          flushDumps: false,
        });
      }
    }
    stopPrewarmBridge({ save: false });
    stopInputBridge();
    resetAudioRingAttachment();
    if (
      instantiatedWasmHandler &&
      typeof instantiatedWasmHandler.releasePendingWireBytes === 'function'
    ) {
      instantiatedWasmHandler.releasePendingWireBytes();
    }
    boundModule = null;
    instantiatedMemory = null;
    attachedGuestMemory = null;
    attachedGuestMemoryWorker = null;
    runtimeExportsSummary = null;
    if (userOnAbort) {
      return userOnAbort.apply(this, arguments);
    }
    return drainPromise;
  };

  const userdataApiObj = activeHostImports.userdata || {};
  const config = {
    mainScriptUrlOrBlob: resolvedRuntimeScriptUrl,
    locateFile,
    preRun,
    print,
    printErr,
    onRuntimeInitialized,
    onExit,
    onAbort,
    env: activeHostImports.env,
    hostImports: activeHostImports,
    httpfsMailboxSab:
      options.httpfsMailboxSab !== undefined ? options.httpfsMailboxSab : httpfsMailboxSab,
    httpfsMailboxSabs: Array.isArray(options.httpfsMailboxSabs)
      ? options.httpfsMailboxSabs.slice()
      : httpfsMailboxSabs.slice(),
    inputSab: activeInputSab,
    attachGuestMemory: attachGuestMemoryToGpuWorker,
    attachInputBridge,
    stopInputBridge,
    getInputBridge: () => activeInputBridge,
    attachAudioRing,
    notifyAudioWorkerReady,
    onAudioWorkerReady,
    setAudioWorker,
    notifyGuestMilestone,
    scheduleAudioRingRetry: scheduleConfigAudioRingRetry,
    stopAudioRingRetry,
    resetAudioRingAttachment,
    getAudioRingDescriptor: () => audioRingController.getDescriptor(),
    getAudioRingTelemetry,
    attachPrewarmBridge,
    stopPrewarmBridge,
    getPrewarmBridge: () => activePrewarmBridge,
    getPrewarmProgress,
    getPrewarmDiagnostics,
    savePrewarmList,
    handlePrewarmMessage,
    getRuntimeExportsSummary: () =>
      runtimeExportsSummary
        ? {
            hasMain: runtimeExportsSummary.hasMain,
            hasCallMain: runtimeExportsSummary.hasCallMain,
            fastBootExports: runtimeExportsSummary.fastBootExports.slice(),
            mode: runtimeExportsSummary.mode,
          }
        : null,
    reset: resetRuntimeConfig,
    destroy: destroyRuntimeConfig,
    dispose: destroyRuntimeConfig,
    pipelineDumpSyncer,
    getPipelineDumpSyncer: () => pipelineDumpSyncer,
    flushPipelineDumps: (flushOpts) => flushPipelineDumpInternal(flushOpts),
    flushPipelineDump: (flushOpts) => flushPipelineDumpInternal(flushOpts),
    stopPipelineDumpSyncer,
    getLastFaultRecord: () =>
      lastModuleFaultRecord ? cloneFaultRecord(lastModuleFaultRecord) : null,
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

  if (combinedDynamicLibraries.length > 0) {
    config.dynamicLibraries = combinedDynamicLibraries.slice();
    config.sideModuleUrls = resolvedSideModuleUrls.slice();
  }
  if (Array.isArray(options.sharedModules) && options.sharedModules.length > 0) {
    stripWireBytesFromSharedModules(options.sharedModules);
    const compiledOnly = options.sharedModules.filter((entry) => isCompiledWasmModule(entry));
    if (compiledOnly.length > 0) {
      config.sharedModules = compiledOnly;
    }
  }
  if (manifestSummary) {
    config.indirectDispatch = manifestSummary.indirectDispatch;
    config.wasmMetrics = manifestSummary.wasmMetrics
      ? Object.assign({}, manifestSummary.wasmMetrics)
      : null;
    config.manifestSummary = {
      indirectDispatch: manifestSummary.indirectDispatch,
      sideModules: manifestSummary.sideModules.slice(),
      sideModuleCount: manifestSummary.sideModuleCount,
      wasmMetrics: manifestSummary.wasmMetrics
        ? Object.assign({}, manifestSummary.wasmMetrics)
        : null,
    };
  }
  if (typeof locateFile.getProgress === 'function') {
    config.getSideModuleProgress = () => locateFile.getProgress();
  }

  if (typeof options.instantiateWasm === 'function') {
    config.instantiateWasm = options.instantiateWasm;
  } else if (options.instantiateWasm !== null && options.instantiateWasm !== false) {
    instantiatedWasmHandler = createInstantiateWasmHandler({
      locateFile,
      hostImports: activeHostImports,
      reportFn: activeReportFn,
      fetchFn: options.fetchFn,
      wasmBinary: options.wasmBinary,
      wasmMetrics: manifestSummary ? manifestSummary.wasmMetrics : null,
      onError: options.onWasmError,
      onMemory: (mem) => {
        instantiatedMemory = mem || null;
        attachGuestMemoryToGpuWorker(null, mem);
      },
    });
    config.instantiateWasm = instantiatedWasmHandler;
  }

  return config;
}

function createRuntimeModuleOptions(overrides = {}) {
  return buildRuntimeModuleConfig(overrides);
}

function normalizeMaxConcurrentPthreadLoads(value, fallback = DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS) {
  if (value === undefined || value === null || value === '' || typeof value === 'boolean') {
    return fallback;
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.floor(n);
}

function isPthreadLoadMessage(msg) {
  return Boolean(
    msg &&
      typeof msg === 'object' &&
      (msg.cmd === PTHREAD_CMD_LOAD || msg.cmd === 'CMD_LOAD'),
  );
}

function isPthreadLoadedMessage(msg) {
  return Boolean(
    msg &&
      typeof msg === 'object' &&
      (msg.cmd === PTHREAD_CMD_LOADED || msg.cmd === 'CMD_LOADED'),
  );
}

function isPthreadPostLoadMessage(msg) {
  return Boolean(
    msg &&
      typeof msg === 'object' &&
      (msg.cmd === PTHREAD_CMD_RUN ||
        msg.cmd === 'CMD_RUN' ||
        msg.cmd === PTHREAD_CMD_OBJECT_TRANSFER ||
        msg.cmd === 'CMD_OBJECT_TRANSFER'),
  );
}

function isPthreadBootstrapControlMessage(msg) {
  return Boolean(
    msg &&
      typeof msg === 'object' &&
      (msg.__playgta6_httpfs_sab || msg.__playgta6_tx_port),
  );
}

function hookWorkerEventProperty(worker, propName, observer) {
  let protoDesc = null;
  let proto = Object.getPrototypeOf(worker);
  while (proto && !protoDesc) {
    try {
      protoDesc = Object.getOwnPropertyDescriptor(proto, propName) || null;
    } catch {
      protoDesc = null;
    }
    proto = Object.getPrototypeOf(proto);
  }
  let ownDesc = null;
  try {
    ownDesc = Object.getOwnPropertyDescriptor(worker, propName) || null;
  } catch {
    return null;
  }
  if (ownDesc && !ownDesc.configurable) {
    return null;
  }

  const hasProtoAccessor = Boolean(protoDesc && typeof protoDesc.set === 'function');
  let userHandler =
    !hasProtoAccessor && ownDesc && typeof ownDesc.value === 'function' ? ownDesc.value : null;
  let wrappedHandler = null;

  const makeWrapped = (fn) => {
    if (typeof fn !== 'function') {
      return null;
    }
    return function wrappedWorkerEventHandler(ev) {
      observer(ev);
      return fn.call(this, ev);
    };
  };

  try {
    if (hasProtoAccessor) {
      const existing = protoDesc.get ? protoDesc.get.call(worker) : null;
      if (typeof existing === 'function') {
        userHandler = existing;
        wrappedHandler = makeWrapped(existing);
        protoDesc.set.call(worker, wrappedHandler);
      }
      Object.defineProperty(worker, propName, {
        configurable: true,
        enumerable: protoDesc.enumerable !== false,
        get() {
          return userHandler;
        },
        set(fn) {
          userHandler = typeof fn === 'function' ? fn : null;
          wrappedHandler = makeWrapped(userHandler);
          protoDesc.set.call(worker, wrappedHandler);
        },
      });
    } else {
      wrappedHandler = makeWrapped(userHandler);
      Object.defineProperty(worker, propName, {
        configurable: true,
        enumerable: ownDesc ? ownDesc.enumerable !== false : true,
        get() {
          return wrappedHandler;
        },
        set(fn) {
          userHandler = typeof fn === 'function' ? fn : null;
          wrappedHandler = makeWrapped(userHandler);
        },
      });
    }
  } catch {
    return null;
  }

  let unhooked = false;
  return function unhookWorkerEventProperty() {
    if (unhooked) {
      return;
    }
    unhooked = true;
    const current = userHandler;
    userHandler = null;
    wrappedHandler = null;
    try {
      delete worker[propName];
      if (hasProtoAccessor) {
        if (typeof current === 'function') {
          protoDesc.set.call(worker, current);
        }
      } else {
        worker[propName] = current;
      }
    } catch {
      // Ignore unhook errors on frozen or exotic worker mocks.
    }
  };
}

/**
 * Coordinates 52+ concurrent Emscripten `em-pthread` / `em-pthread-<N>` workers:
 * - Delivers `{ __playgta6_httpfs_sab }` immediately before `CMD_LOAD` when available.
 * - Throttles concurrent in-flight `CMD_LOAD` instantiations to `maxConcurrentPthreadLoads`
 *   (0 = unthrottled) so burst spawns do not starve `io_worker.js` / `wgpu_worker.js` or blow V8 memory.
 * - Buffers any `CMD_RUN`, `CMD_OBJECT_TRANSFER`, or interleaved post-load messages (even if posted
 *   before `CMD_LOAD` is dispatched or while `CMD_LOAD` is in flight) and flushes them in strict FIFO
 *   order only AFTER `CMD_LOADED` completes.
 * - Cleans up all worker tracking references on `terminate()` or `'error'` and frees load slots
 *   immediately so queued workers never deadlock or leak memory.
 * - Tracks clean numeric-only telemetry (`pthreadSpawnedCount`, `pthreadActiveCount`,
 *   `pthreadPeakCount`, `pthreadLoadQueueDepth`, `pthreadFailedCount`).
 */
function createPthreadPoolCoordinator(options = {}) {
  let maxConcurrentLoads = normalizeMaxConcurrentPthreadLoads(
    options.maxConcurrentPthreadLoads,
    DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS,
  );
  const hasExplicitSingleSabOption =
    typeof options.getHttpfsMailboxSab === 'function' || options.httpfsMailboxSab !== undefined;
  const getHttpfsMailboxSab =
    typeof options.getHttpfsMailboxSab === 'function'
      ? options.getHttpfsMailboxSab
      : () => options.httpfsMailboxSab || null;
  const getHttpfsMailboxSabs =
    typeof options.getHttpfsMailboxSabs === 'function'
      ? options.getHttpfsMailboxSabs
      : Array.isArray(options.httpfsMailboxSabs)
        ? () => options.httpfsMailboxSabs
        : null;
  const onPthreadTxMessage =
    typeof options.onPthreadTxMessage === 'function' ? options.onPthreadTxMessage : null;
  const getSharedModules =
    typeof options.getSharedModules === 'function'
      ? options.getSharedModules
      : Array.isArray(options.sharedModules)
        ? () => options.sharedModules
        : null;
  const getSharedModuleNames =
    typeof options.getSharedModuleNames === 'function'
      ? options.getSharedModuleNames
      : Array.isArray(options.sharedModuleNames)
        ? () => options.sharedModuleNames
        : null;
  const explicitMessageChannelCtor = options.MessageChannelCtor;
  const onTelemetryChange =
    typeof options.onTelemetryChange === 'function' ? options.onTelemetryChange : null;

  const trackedWorkers = new Map();
  const loadQueue = [];
  let queueHead = 0;
  let queuedCount = 0;
  let activeLoads = 0;
  let draining = false;
  let nextHttpfsMailboxLane = 0;

  let pthreadSpawnedCount = 0;
  let pthreadActiveCount = 0;
  let pthreadPeakCount = 0;
  let pthreadFailedCount = 0;
  let pthreadLoadedCount = 0;
  let peakConcurrentLoads = 0;
  let pthreadActiveTxPorts = 0;
  let pthreadSharedModulesDelivered = 0;
  const activeTxPorts = new Set();

  function resolveHttpfsMailboxPool() {
    const rawPool = getHttpfsMailboxSabs ? getHttpfsMailboxSabs() : null;
    if (Array.isArray(rawPool) && rawPool.length > 0) {
      return rawPool;
    }
    const single = getHttpfsMailboxSab();
    return single ? [single] : [];
  }

  function assignNextHttpfsMailboxLane(meta) {
    if (meta && meta.httpfsMailboxSab !== undefined) {
      const explicitSab = meta.httpfsMailboxSab;
      return {
        primarySab: explicitSab,
        laneSab: explicitSab,
        laneIndex: explicitSab ? 0 : -1,
        laneCount: explicitSab ? 1 : 0,
      };
    }
    const pool = resolveHttpfsMailboxPool();
    if (pool.length === 0) {
      return { primarySab: null, laneSab: null, laneIndex: -1, laneCount: 0 };
    }
    const laneIndex = nextHttpfsMailboxLane % pool.length;
    nextHttpfsMailboxLane += 1;
    const laneSab = pool[laneIndex] || null;
    // Preserve __playgta6_httpfs_sab pointing to lane 0 when a legacy single-SAB getter is
    // supplied alongside the multi-lane pool; dedicated per-pthread lanes arrive via
    // __playgta6_httpfs_lane_sab (preferred by bootstrapEmscriptenPthreadWorker).
    const primarySab =
      hasExplicitSingleSabOption && pool.length > 1
        ? getHttpfsMailboxSab() || pool[0] || laneSab
        : laneSab;
    return {
      primarySab,
      laneSab,
      laneIndex,
      laneCount: pool.length,
    };
  }

  function getTelemetry() {
    return {
      pthreadSpawnedCount,
      pthreadActiveCount,
      pthreadPeakCount,
      pthreadLoadQueueDepth: queuedCount,
      pthreadFailedCount,
      pthreadLoadedCount,
      pthreadActiveLoads: activeLoads,
      pthreadPeakConcurrentLoads: peakConcurrentLoads,
      maxConcurrentPthreadLoads: maxConcurrentLoads,
      pthreadActiveTxPorts,
      pthreadSharedModulesDelivered,
      httpfsMailboxLanes: resolveHttpfsMailboxPool().length,
    };
  }

  function emitTelemetry() {
    if (!onTelemetryChange) {
      return;
    }
    try {
      onTelemetryChange(getTelemetry());
    } catch {
      // Telemetry callbacks must never interfere with pthread scheduling.
    }
  }

  function compactQueueIfNeeded() {
    if (queuedCount === 0 || queueHead >= loadQueue.length) {
      loadQueue.length = 0;
      queueHead = 0;
      return;
    }
    if (queueHead >= 32 && queueHead * 2 >= loadQueue.length) {
      let writeIdx = 0;
      for (let i = queueHead; i < loadQueue.length; i += 1) {
        const item = loadQueue[i];
        if (item && !item.cleanedUp && item.state === 'queued') {
          loadQueue[writeIdx] = item;
          item.queueIndex = writeIdx;
          writeIdx += 1;
        }
      }
      loadQueue.length = writeIdx;
      queueHead = 0;
    }
  }

  function enqueueLoad(entry) {
    entry.state = 'queued';
    entry.queueIndex = loadQueue.length;
    loadQueue.push(entry);
    queuedCount += 1;
  }

  function removeFromLoadQueue(entry) {
    if (entry.queueIndex >= 0 && entry.queueIndex < loadQueue.length) {
      if (loadQueue[entry.queueIndex] === entry) {
        loadQueue[entry.queueIndex] = null;
        queuedCount = Math.max(0, queuedCount - 1);
      }
    }
    entry.queueIndex = -1;
    compactQueueIfNeeded();
  }

  function detachMessageObservers(entry) {
    if (!entry.messageCleanupHooks || entry.messageCleanupHooks.length === 0) {
      return;
    }
    const hooks = entry.messageCleanupHooks;
    entry.messageCleanupHooks = [];
    for (let i = 0; i < hooks.length; i += 1) {
      try {
        hooks[i]();
      } catch {
        // Ignore listener detach errors.
      }
    }
  }

  function detachEntryTxPort(entry) {
    if (!entry || !entry.txPort) {
      return;
    }
    const port = entry.txPort;
    entry.txPort = null;
    activeTxPorts.delete(port);
    pthreadActiveTxPorts = Math.max(0, pthreadActiveTxPorts - 1);
    closeMessagePort(port);
  }

  function cleanupEntry(entry, finalState) {
    if (!entry || entry.cleanedUp) {
      return;
    }
    entry.cleanedUp = true;

    const prevState = entry.state;
    if (prevState === 'queued') {
      removeFromLoadQueue(entry);
    } else if (prevState === 'loading') {
      activeLoads = Math.max(0, activeLoads - 1);
    }

    entry.state = finalState;
    if (finalState === 'failed') {
      pthreadFailedCount += 1;
    }
    pthreadActiveCount = Math.max(0, pthreadActiveCount - 1);

    entry.pendingMessages.length = 0;
    entry.loadArgs = null;

    detachEntryTxPort(entry);
    detachMessageObservers(entry);
    const hooks = entry.cleanupHooks;
    entry.cleanupHooks = [];
    for (let i = 0; i < hooks.length; i += 1) {
      try {
        hooks[i]();
      } catch {
        // Ignore worker hook restoration errors.
      }
    }

    const worker = entry.worker;
    entry.worker = null;
    if (worker) {
      trackedWorkers.delete(worker);
    }

    emitTelemetry();
    drainLoadQueue();
  }

  function handleWorkerLoaded(entry) {
    if (!entry || entry.cleanedUp) {
      return;
    }
    if (entry.state !== 'loading' && entry.state !== 'queued' && entry.state !== 'spawned') {
      return;
    }
    if (entry.state === 'queued') {
      removeFromLoadQueue(entry);
    } else if (entry.state === 'loading') {
      activeLoads = Math.max(0, activeLoads - 1);
    }

    entry.state = 'loaded';
    entry.loadArgs = null;
    pthreadLoadedCount += 1;
    detachMessageObservers(entry);

    const worker = entry.worker;
    const origPost = entry.origPostMessage;
    const queued = entry.pendingMessages;
    entry.pendingMessages = [];

    if (worker && origPost) {
      for (let i = 0; i < queued.length; i += 1) {
        if (entry.cleanedUp || entry.state !== 'loaded') {
          break;
        }
        try {
          origPost.apply(worker, queued[i]);
        } catch {
          cleanupEntry(entry, 'failed');
          return;
        }
      }
    }

    emitTelemetry();
    drainLoadQueue();
  }

  function drainLoadQueue() {
    if (draining) {
      return;
    }
    draining = true;
    try {
      while (queuedCount > 0 && (maxConcurrentLoads === 0 || activeLoads < maxConcurrentLoads)) {
        let nextEntry = null;
        while (queueHead < loadQueue.length) {
          const candidate = loadQueue[queueHead];
          loadQueue[queueHead] = null;
          queueHead += 1;
          if (candidate && !candidate.cleanedUp && candidate.state === 'queued') {
            nextEntry = candidate;
            break;
          }
        }
        if (!nextEntry) {
          queuedCount = 0;
          compactQueueIfNeeded();
          break;
        }

        nextEntry.queueIndex = -1;
        queuedCount = Math.max(0, queuedCount - 1);
        compactQueueIfNeeded();

        const loadArgs = nextEntry.loadArgs;
        nextEntry.loadArgs = null;
        if (!loadArgs || !nextEntry.worker || !nextEntry.origPostMessage) {
          cleanupEntry(nextEntry, 'failed');
          continue;
        }

        nextEntry.state = 'loading';
        activeLoads += 1;
        if (activeLoads > peakConcurrentLoads) {
          peakConcurrentLoads = activeLoads;
        }
        emitTelemetry();

        try {
          nextEntry.origPostMessage.apply(nextEntry.worker, loadArgs);
        } catch {
          cleanupEntry(nextEntry, 'failed');
        }
      }
    } finally {
      draining = false;
    }
  }

  function registerWorker(worker, meta = {}) {
    if (!worker || typeof worker !== 'object') {
      return worker;
    }
    if (trackedWorkers.has(worker)) {
      return worker;
    }

    const origPostMessage = typeof worker.postMessage === 'function' ? worker.postMessage : null;
    const hasOwnPostMessage = Object.prototype.hasOwnProperty.call(worker, 'postMessage');
    const origTerminate = typeof worker.terminate === 'function' ? worker.terminate : null;
    const hasOwnTerminate = Object.prototype.hasOwnProperty.call(worker, 'terminate');

    const workerName =
      meta && typeof meta.name === 'string' ? meta.name : EMSCRIPTEN_PTHREAD_WORKER_NAME;
    const laneAssignment = assignNextHttpfsMailboxLane(meta);
    const sab = laneAssignment.primarySab;
    const laneSab = laneAssignment.laneSab;

    let coordPort = null;
    let pthreadPort = null;
    if (onPthreadTxMessage && origPostMessage) {
      const mcCtor = resolveMessageChannelCtor(explicitMessageChannelCtor);
      if (mcCtor) {
        try {
          const mc = new mcCtor();
          if (mc && mc.port1 && mc.port2 && typeof mc.port2.postMessage === 'function') {
            coordPort = mc.port1;
            pthreadPort = mc.port2;
            coordPort.onmessage = (ev) => {
              try {
                onPthreadTxMessage(ev && ev.data !== undefined ? ev.data : ev, workerName, coordPort);
              } catch {
                // Pthread TX handler errors must never disrupt the coordinator worker loop.
              }
            };
            if (typeof coordPort.unref === 'function') {
              try {
                coordPort.unref();
              } catch {
                // Ignore unref failures on non-Node MessagePort implementations.
              }
            }
          } else {
            closeMessagePort(mc && mc.port1);
            closeMessagePort(mc && mc.port2);
          }
        } catch {
          closeMessagePort(coordPort);
          closeMessagePort(pthreadPort);
          coordPort = null;
          pthreadPort = null;
        }
      }
    }

    if (origPostMessage && (sab || laneSab || pthreadPort)) {
      const bootstrapMsg = {};
      const transferList = [];
      if (sab) {
        bootstrapMsg.__playgta6_httpfs_sab = sab;
      }
      if (laneSab) {
        bootstrapMsg.__playgta6_httpfs_lane_sab = laneSab;
        if (laneAssignment.laneIndex >= 0) {
          bootstrapMsg.__playgta6_httpfs_lane = laneAssignment.laneIndex;
        }
      }
      if (pthreadPort) {
        bootstrapMsg.__playgta6_tx_port = pthreadPort;
        transferList.push(pthreadPort);
      }
      try {
        if (transferList.length > 0) {
          origPostMessage.call(worker, bootstrapMsg, transferList);
        } else {
          origPostMessage.call(worker, bootstrapMsg);
        }
      } catch {
        if (coordPort || pthreadPort) {
          closeMessagePort(coordPort);
          closeMessagePort(pthreadPort);
          coordPort = null;
          pthreadPort = null;
          if (sab || laneSab) {
            const fallbackMsg = {};
            if (sab) {
              fallbackMsg.__playgta6_httpfs_sab = sab;
            }
            if (laneSab) {
              fallbackMsg.__playgta6_httpfs_lane_sab = laneSab;
              if (laneAssignment.laneIndex >= 0) {
                fallbackMsg.__playgta6_httpfs_lane = laneAssignment.laneIndex;
              }
            }
            try {
              origPostMessage.call(worker, fallbackMsg);
            } catch {
              // Non-fatal: BroadcastChannel handshake remains as fallback.
            }
          }
        }
      }
    }

    pthreadSpawnedCount += 1;
    pthreadActiveCount += 1;
    if (pthreadActiveCount > pthreadPeakCount) {
      pthreadPeakCount = pthreadActiveCount;
    }
    if (coordPort) {
      activeTxPorts.add(coordPort);
      pthreadActiveTxPorts += 1;
    }

    const entry = {
      worker,
      name: workerName,
      state: 'spawned',
      cleanedUp: false,
      queueIndex: -1,
      loadArgs: null,
      pendingMessages: [],
      origPostMessage,
      origTerminate,
      txPort: coordPort,
      httpfsLaneIndex: laneAssignment.laneIndex,
      httpfsLaneSab: laneSab,
      messageCleanupHooks: [],
      cleanupHooks: [],
    };
    trackedWorkers.set(worker, entry);

    const onWorkerMessage = (ev) => {
      const data = ev && ev.data !== undefined ? ev.data : ev;
      if (isPthreadLoadedMessage(data)) {
        handleWorkerLoaded(entry);
      }
    };
    const onWorkerError = () => {
      cleanupEntry(entry, 'failed');
    };

    if (typeof worker.addEventListener === 'function') {
      try {
        worker.addEventListener('message', onWorkerMessage);
        entry.messageCleanupHooks.push(() => {
          if (typeof worker.removeEventListener === 'function') {
            worker.removeEventListener('message', onWorkerMessage);
          }
        });
      } catch {
        // Ignore non-standard addEventListener failures.
      }
      try {
        worker.addEventListener('error', onWorkerError);
        entry.cleanupHooks.push(() => {
          if (typeof worker.removeEventListener === 'function') {
            worker.removeEventListener('error', onWorkerError);
          }
        });
      } catch {
        // Ignore non-standard addEventListener failures.
      }
    }

    const unhookOnMessage = hookWorkerEventProperty(worker, 'onmessage', onWorkerMessage);
    if (unhookOnMessage) {
      entry.messageCleanupHooks.push(unhookOnMessage);
    }
    const unhookOnError = hookWorkerEventProperty(worker, 'onerror', onWorkerError);
    if (unhookOnError) {
      entry.cleanupHooks.push(unhookOnError);
    }

    if (origPostMessage) {
      const wrappedPostMessage = function wrappedPthreadPostMessage(msg) {
        const callArgs = Array.prototype.slice.call(arguments);
        if (entry.cleanedUp || entry.state === 'terminated' || entry.state === 'failed') {
          return undefined;
        }
        if (isPthreadBootstrapControlMessage(msg)) {
          return origPostMessage.apply(worker, callArgs);
        }
        if (entry.state === 'loaded') {
          return origPostMessage.apply(worker, callArgs);
        }
        if (isPthreadLoadMessage(msg)) {
          if (entry.state === 'spawned') {
            // W351 / W371 (#1077): Share compiled WebAssembly.Module objects across ALL 48+
            // pthreads (even if Emscripten passes an empty `sharedModules: []`) and never carry
            // or clone raw wire-byte ArrayBuffers alongside compiled modules in CMD_LOAD.
            if (msg && typeof msg === 'object') {
              const mergedModules = [];
              const mergedNames = [];
              const seenModules = new Set();
              if (msg.sharedModules) {
                stripWireBytesFromSharedModules(msg.sharedModules);
                if (Array.isArray(msg.sharedModules)) {
                  for (let i = 0; i < msg.sharedModules.length; i += 1) {
                    const m = msg.sharedModules[i];
                    if (isCompiledWasmModule(m) && !seenModules.has(m)) {
                      seenModules.add(m);
                      mergedModules.push(m);
                      const existingName =
                        Array.isArray(msg.__playgta6_shared_module_names) &&
                        typeof msg.__playgta6_shared_module_names[i] === 'string'
                          ? msg.__playgta6_shared_module_names[i]
                          : '';
                      mergedNames.push(existingName);
                    }
                  }
                }
              }
              if (getSharedModules) {
                const shared = getSharedModules();
                const sharedNames = getSharedModuleNames ? getSharedModuleNames() : null;
                if (Array.isArray(shared) && shared.length > 0) {
                  for (let i = 0; i < shared.length; i += 1) {
                    const m = shared[i];
                    if (isCompiledWasmModule(m) && !seenModules.has(m)) {
                      seenModules.add(m);
                      mergedModules.push(m);
                      const name =
                        Array.isArray(sharedNames) && typeof sharedNames[i] === 'string'
                          ? sharedNames[i]
                          : '';
                      mergedNames.push(name);
                    }
                  }
                }
              }
              if (mergedModules.length > 0) {
                msg.sharedModules = mergedModules;
                if (mergedNames.some((n) => n.length > 0)) {
                  msg.__playgta6_shared_module_names = mergedNames;
                }
                pthreadSharedModulesDelivered += mergedModules.length;
                wasmMemoryTelemetry.sharedModulesDeliveredToPthreads += mergedModules.length;
              }
              if (isCompiledWasmModule(msg.wasmModule)) {
                if (
                  Object.prototype.hasOwnProperty.call(msg, 'wasmBinary') &&
                  getBinaryWireInfo(msg.wasmBinary)
                ) {
                  const info = getBinaryWireInfo(msg.wasmBinary);
                  if (isWritableOwnProp(msg, 'wasmBinary')) {
                    msg.wasmBinary = null;
                    if (info && info.byteLength > 0) {
                      recordDroppedWasmWireBytes(info.byteLength, 1);
                    }
                  }
                }
                if (msg.guestModuleBytes) {
                  releaseWireBytesFromObject(msg.guestModuleBytes, { assumeWasm: true });
                }
                if (msg.preloadedBuffers) {
                  releaseWireBytesFromObject(msg.preloadedBuffers, { assumeWasm: true });
                }
              }
            }
            entry.loadArgs = callArgs;
            enqueueLoad(entry);
            emitTelemetry();
            drainLoadQueue();
          }
          return undefined;
        }
        if (
          entry.state === 'queued' ||
          entry.state === 'loading' ||
          isPthreadPostLoadMessage(msg)
        ) {
          entry.pendingMessages.push(callArgs);
          return undefined;
        }
        return origPostMessage.apply(worker, callArgs);
      };
      try {
        worker.postMessage = wrappedPostMessage;
        entry.cleanupHooks.push(() => {
          if (worker.postMessage === wrappedPostMessage) {
            if (hasOwnPostMessage) {
              worker.postMessage = origPostMessage;
            } else {
              delete worker.postMessage;
            }
          }
        });
      } catch {
        // Ignore if worker.postMessage is non-writable.
      }
    }

    if (origTerminate) {
      const wrappedTerminate = function wrappedPthreadTerminate() {
        cleanupEntry(entry, 'terminated');
        return origTerminate.apply(worker, arguments);
      };
      try {
        worker.terminate = wrappedTerminate;
        entry.cleanupHooks.push(() => {
          if (worker.terminate === wrappedTerminate) {
            if (hasOwnTerminate) {
              worker.terminate = origTerminate;
            } else {
              delete worker.terminate;
            }
          }
        });
      } catch {
        // Ignore if worker.terminate is non-writable.
      }
    }

    emitTelemetry();
    return worker;
  }

  return {
    registerWorker,
    getTelemetry,
    getStats: getTelemetry,
    getTrackedCount: () => trackedWorkers.size,
    getMaxConcurrentPthreadLoads: () => maxConcurrentLoads,
    setMaxConcurrentPthreadLoads(nextLimit) {
      maxConcurrentLoads = normalizeMaxConcurrentPthreadLoads(nextLimit, maxConcurrentLoads);
      emitTelemetry();
      drainLoadQueue();
      return maxConcurrentLoads;
    },
    notifyWorkerLoaded(worker) {
      const entry = trackedWorkers.get(worker);
      if (entry) {
        handleWorkerLoaded(entry);
      }
    },
    notifyWorkerError(worker) {
      const entry = trackedWorkers.get(worker);
      if (entry) {
        cleanupEntry(entry, 'failed');
      }
    },
    notifyWorkerTerminated(worker) {
      const entry = trackedWorkers.get(worker);
      if (entry) {
        cleanupEntry(entry, 'terminated');
      }
    },
    closeTxPorts() {
      trackedWorkers.forEach((entry) => {
        detachEntryTxPort(entry);
      });
      activeTxPorts.clear();
      emitTelemetry();
    },
    isActiveTxPort(port) {
      return Boolean(port && activeTxPorts.has(port));
    },
    getWorkerHttpfsLane(worker) {
      const entry = trackedWorkers.get(worker);
      if (!entry) {
        return null;
      }
      return {
        laneIndex: entry.httpfsLaneIndex,
        sab: entry.httpfsLaneSab,
      };
    },
    resetHttpfsLaneCursor() {
      nextHttpfsMailboxLane = 0;
    },
  };
}

/**
 * Wrap a `Worker` constructor so `em-pthread` / `em-pthread-<N>` workers redirect to
 * `getRedirectHref()` and register with `coordinator` for `httpfsMailboxSab` delivery,
 * bounded `CMD_LOAD` throttling, strict `CMD_LOAD`-before-`CMD_RUN`/`CMD_OBJECT_TRANSFER`
 * ordering, cleanup on error/terminate, and numeric pthread telemetry.
 */
function installPthreadWorkerProxy(options = {}) {
  const OriginalWorker = options.WorkerCtor || null;
  if (typeof OriginalWorker !== 'function') {
    return null;
  }
  const coordinator = options.coordinator || createPthreadPoolCoordinator(options);
  const getRedirectHref =
    typeof options.getRedirectHref === 'function'
      ? options.getRedirectHref
      : () => (typeof options.redirectHref === 'string' ? options.redirectHref : '');

  const WrappedWorker = new Proxy(OriginalWorker, {
    construct(target, args, newTarget) {
      const nextArgs = Array.prototype.slice.call(args);
      const workerOpts = nextArgs.length >= 2 ? nextArgs[1] : undefined;
      const isEmPthread = Boolean(
        workerOpts &&
          typeof workerOpts === 'object' &&
          isEmscriptenPthreadWorkerName(workerOpts.name),
      );
      if (isEmPthread) {
        const redirectHref = getRedirectHref();
        if (typeof redirectHref === 'string' && redirectHref.length > 0) {
          nextArgs[0] = redirectHref;
        }
      }
      const targetNew = newTarget === WrappedWorker ? target : newTarget;
      const instance = Reflect.construct(target, nextArgs, targetNew);
      if (isEmPthread && instance) {
        coordinator.registerWorker(instance, { name: workerOpts.name });
      }
      return instance;
    },
  });

  return { WrappedWorker, coordinator };
}

let coordinatorFirstPresentEmitted = false;
let configuredSideModuleUrls = null;
let configuredDynamicLibraries = null;
let configuredGuestManifest = null;
let configuredGuestManifestUrl = null;
let configuredMaxConcurrentPthreadLoads = null;
let configuredDumpPipelines = undefined;
let configuredDumpDir = undefined;
let configuredDumpEndpoint = undefined;
let configuredDumpIntervalMs = undefined;
let configuredDumpMaxConcurrentUploads = undefined;
let activeCoordinatorPipelineDumpSyncer = null;
let activeGuestManifestSummary = null;
let activeSideModuleProgressGetter = null;
let pendingPreloadedWasmBinary = null;
let pendingPreloadedGuestBuffers = null;
const coordinatorSharedWasmModules = createSharedWasmModuleCache({ trackGlobalCount: true });
let activeCoordinatorSharedModuleShimRestore = null;

function restoreCoordinatorSharedModuleShim() {
  const restore = activeCoordinatorSharedModuleShimRestore;
  activeCoordinatorSharedModuleShimRestore = null;
  if (typeof restore === 'function') {
    try {
      restore();
    } catch {
      // Ignore shim teardown errors.
    }
  }
}

function attachActiveCoordinatorGuestMemory(modOverride) {
  if (
    activeCoordinatorModuleConfig &&
    typeof activeCoordinatorModuleConfig.attachGuestMemory === 'function'
  ) {
    try {
      return activeCoordinatorModuleConfig.attachGuestMemory(
        modOverride !== undefined ? modOverride : activeCoordinatorRuntimeModule,
      );
    } catch {
      return false;
    }
  }
  return false;
}

// G75 (#663): bounded FIFO mapping in-flight recycled command buffers back to the
// originating pthread coordinator `txPort` so warm buffers return to the submitting thread.
const pendingRecycleTxPorts = [];
const MAX_PENDING_RECYCLE_TX_PORTS = 16;

function enqueueRecycleTxPort(port) {
  if (!port || typeof port.postMessage !== 'function') {
    return;
  }
  if (pendingRecycleTxPorts.length >= MAX_PENDING_RECYCLE_TX_PORTS) {
    pendingRecycleTxPorts.shift();
  }
  pendingRecycleTxPorts.push(port);
}

function handleRecycledCmdBuffer(cmdBuffer) {
  if (
    !isTransferableArrayBuffer(cmdBuffer) ||
    cmdBuffer.byteLength < WGPU_CMD_POOL_MIN_BYTES ||
    cmdBuffer.byteLength > WGPU_CMD_POOL_MAX_BYTES
  ) {
    return false;
  }
  while (pendingRecycleTxPorts.length > 0) {
    const port = pendingRecycleTxPorts.shift();
    if (!port || typeof port.postMessage !== 'function') {
      continue;
    }
    if (
      coordinatorPthreadPool &&
      typeof coordinatorPthreadPool.isActiveTxPort === 'function' &&
      !coordinatorPthreadPool.isActiveTxPort(port)
    ) {
      continue;
    }
    try {
      port.postMessage(
        { type: PTHREAD_TX_MSG_RECYCLE_CMD_BUFFER, cmdBuffer },
        [cmdBuffer],
      );
      return true;
    } catch {
      // Port closed while the batch was in flight; drop the buffer cleanly.
    }
  }
  return false;
}

// Orch-32 P0 #2 (#663): Bounded in-flight GPU Present backpressure.
// Prevents hundreds of guest Present command buffers from queuing unboundedly onto
// `gpuWorker` during shader-compile stalls (which otherwise causes multi-hundred-frame
// compositor bursts once compilation completes), while NEVER dropping resource/state
// creation packets and never deadlocking on worker reset/error or dropped acks.
let _inFlightGpuPresents = 0;
let _maxInFlightGpuPresents = DEFAULT_MAX_IN_FLIGHT_GPU_PRESENTS;
let _coalescedPresentSubmissions = 0;
let _pendingCoalescedPresent = null;
let _lastGpuPresentProgressAtMs = 0;

function normalizeMaxInFlightGpuPresents(value, fallback = DEFAULT_MAX_IN_FLIGHT_GPU_PRESENTS) {
  if (typeof value === 'number' && Number.isInteger(value) && value >= 1) {
    return value;
  }
  return fallback;
}

function recycleAbandonedCmdBufferToPort(entry) {
  if (
    !entry ||
    !entry.wantRecycle ||
    !entry.sourcePort ||
    typeof entry.sourcePort.postMessage !== 'function' ||
    !isTransferableArrayBuffer(entry.transferBuf)
  ) {
    return false;
  }
  if (
    coordinatorPthreadPool &&
    typeof coordinatorPthreadPool.isActiveTxPort === 'function' &&
    !coordinatorPthreadPool.isActiveTxPort(entry.sourcePort)
  ) {
    return false;
  }
  try {
    entry.sourcePort.postMessage(
      { type: PTHREAD_TX_MSG_RECYCLE_CMD_BUFFER, cmdBuffer: entry.transferBuf },
      [entry.transferBuf],
    );
    return true;
  } catch {
    return false;
  }
}

function resetGpuPresentBackpressure(options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  _inFlightGpuPresents = 0;
  _lastGpuPresentProgressAtMs = 0;
  pendingRecycleTxPorts.length = 0;
  if (_pendingCoalescedPresent) {
    recycleAbandonedCmdBufferToPort(_pendingCoalescedPresent);
    _pendingCoalescedPresent = null;
  }
  if (safeOpts.resetCounters === true) {
    _coalescedPresentSubmissions = 0;
  }
  if (safeOpts.maxInFlightPresents !== undefined) {
    _maxInFlightGpuPresents = normalizeMaxInFlightGpuPresents(
      safeOpts.maxInFlightPresents,
      DEFAULT_MAX_IN_FLIGHT_GPU_PRESENTS,
    );
  }
}

function maybeDecayInFlightGpuPresents(nowMs = Date.now()) {
  if (
    _inFlightGpuPresents > 0 &&
    _lastGpuPresentProgressAtMs > 0 &&
    nowMs - _lastGpuPresentProgressAtMs >= DEFAULT_GPU_PRESENT_BACKPRESSURE_DECAY_MS
  ) {
    _inFlightGpuPresents = 0;
    _lastGpuPresentProgressAtMs = nowMs;
  }
}

function flushPendingCoalescedPresent() {
  if (
    !_pendingCoalescedPresent ||
    _inFlightGpuPresents >= _maxInFlightGpuPresents ||
    !gpuWorker ||
    typeof gpuWorker.postMessage !== 'function'
  ) {
    return false;
  }
  const entry = _pendingCoalescedPresent;
  _pendingCoalescedPresent = null;
  attachActiveCoordinatorGuestMemory();
  const gpuMsg = { cmdBuffer: entry.transferBuf, coalescePresent: true };
  if (entry.hasExplicitByteLength) {
    gpuMsg.byteLength = entry.effectiveByteLength;
  }
  if (entry.wantRecycle) {
    gpuMsg.recycleBuffer = true;
    enqueueRecycleTxPort(entry.sourcePort);
  }
  if (_inFlightGpuPresents === 0) {
    _lastGpuPresentProgressAtMs = Date.now();
  }
  _inFlightGpuPresents += 1;
  try {
    gpuWorker.postMessage(gpuMsg, [entry.transferBuf]);
  } catch (err) {
    _inFlightGpuPresents = Math.max(0, _inFlightGpuPresents - 1);
    if (entry.wantRecycle && pendingRecycleTxPorts.length > 0) {
      pendingRecycleTxPorts.pop();
    }
    throw err;
  }
  return true;
}

function ackGpuPresentFromWorkerMessage(data) {
  if (!data || typeof data !== 'object') {
    return false;
  }
  if (data.error) {
    resetGpuPresentBackpressure();
    return false;
  }
  const isAck =
    data.type === 'gpu-present-ack' ||
    data.type === 'frame-stats' ||
    data.type === 'first-present' ||
    data.type === 'first-visible-frame' ||
    data.type === 'visible-frame' ||
    data.type === 'gpu-telemetry' ||
    data.type === 'frame-presented' ||
    (typeof data.frame === 'number' && Number.isFinite(data.frame) && data.frame > 0);
  if (!isAck) {
    return false;
  }
  if (_inFlightGpuPresents > 0) {
    _inFlightGpuPresents = Math.max(0, _inFlightGpuPresents - 1);
  }
  _lastGpuPresentProgressAtMs = Date.now();
  flushPendingCoalescedPresent();
  return true;
}

// W345: Keep pthread GPU submits flowing through the coordinator's multiplexed `txPort`
// so `FIRST_PRESENT` detection (`GPU_OPCODE_PRESENT = 7`), audio-ring milestone delivery,
// and `pthreadActiveTxPorts` telemetry stay intact, while transferring the received
// `ArrayBuffer` straight through to `gpuWorker` with zero extra copy.
function handlePthreadWgpuSubmit(data, options = {}) {
  const isEnvelope = Boolean(
    data &&
      typeof data === 'object' &&
      !isTransferableArrayBuffer(data) &&
      !isSharedArrayBuffer(data) &&
      !ArrayBuffer.isView(data),
  );
  const raw = isEnvelope ? data.cmdBuffer || data.bytes || null : data;
  let transferBuf = null;
  if (options.ownedArrayBuffer !== false && isTransferableArrayBuffer(raw)) {
    if (raw.byteLength === 0) {
      return false;
    }
    transferBuf = raw;
  } else {
    const view = asDataView(raw);
    if (!view || view.byteLength === 0) {
      return false;
    }
    if (options.ownedArrayBuffer !== false) {
      transferBuf = toTransferableArrayBuffer(raw);
    } else {
      const copyBytes = new Uint8Array(view.byteLength);
      copyBytes.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
      transferBuf = copyBytes.buffer;
    }
  }
  if (!transferBuf || transferBuf.byteLength === 0) {
    return false;
  }
  const hasExplicitByteLength = Boolean(
    isEnvelope &&
      typeof data.byteLength === 'number' &&
      Number.isInteger(data.byteLength) &&
      data.byteLength >= 0 &&
      data.byteLength <= transferBuf.byteLength,
  );
  const effectiveByteLength = hasExplicitByteLength ? data.byteLength : transferBuf.byteLength;
  const batchInfo =
    effectiveByteLength > 0
      ? inspectGpuPacketBatchForBackpressure(transferBuf, effectiveByteLength)
      : { packetCount: 0, hasPresent: false, isPurePresent: false, truncated: false };
  if (!coordinatorFirstPresentEmitted && batchInfo.hasPresent) {
    coordinatorFirstPresentEmitted = true;
    if (
      activeCoordinatorModuleConfig &&
      typeof activeCoordinatorModuleConfig.notifyGuestMilestone === 'function'
    ) {
      try {
        activeCoordinatorModuleConfig.notifyGuestMilestone('first_present');
      } catch {
        // Non-fatal audio ring milestone retry failure.
      }
    }
    report('[boot] FIRST_PRESENT');
  }
  if (gpuWorker && typeof gpuWorker.postMessage === 'function') {
    maybeDecayInFlightGpuPresents();
    const maxInFlight =
      options && options.maxInFlightPresents !== undefined
        ? normalizeMaxInFlightGpuPresents(options.maxInFlightPresents, _maxInFlightGpuPresents)
        : _maxInFlightGpuPresents;
    const sourcePort = options && options.sourcePort ? options.sourcePort : null;
    const wantRecycle = Boolean(
      isEnvelope &&
        data.recycleBuffer === true &&
        sourcePort &&
        typeof sourcePort.postMessage === 'function' &&
        transferBuf.byteLength >= WGPU_CMD_POOL_MIN_BYTES &&
        transferBuf.byteLength <= WGPU_CMD_POOL_MAX_BYTES,
    );
    const isBackpressuredPresent = batchInfo.hasPresent && _inFlightGpuPresents >= maxInFlight;
    if (isBackpressuredPresent && batchInfo.isPurePresent) {
      // Pure/keep-alive Present while gpuWorker already has maxInFlight presents queued:
      // coalesce into a single latest pending slot so shader-compile stalls never build
      // an unbounded multi-hundred-frame compositor burst.
      if (_pendingCoalescedPresent) {
        recycleAbandonedCmdBufferToPort(_pendingCoalescedPresent);
      }
      _pendingCoalescedPresent = {
        transferBuf,
        hasExplicitByteLength,
        effectiveByteLength,
        sourcePort,
        wantRecycle,
      };
      _coalescedPresentSubmissions += 1;
      return true;
    }
    attachActiveCoordinatorGuestMemory();
    const gpuMsg = { cmdBuffer: transferBuf };
    if (hasExplicitByteLength) {
      gpuMsg.byteLength = effectiveByteLength;
    }
    if (wantRecycle) {
      gpuMsg.recycleBuffer = true;
      enqueueRecycleTxPort(sourcePort);
    }
    if (isBackpressuredPresent) {
      // Batch contains resource/state/draw packets alongside a Present: NEVER drop it,
      // but signal gpuWorker to coalesce the trailing Present if still saturated.
      gpuMsg.coalescePresent = true;
      _coalescedPresentSubmissions += 1;
    }
    if (batchInfo.hasPresent) {
      if (_inFlightGpuPresents === 0) {
        _lastGpuPresentProgressAtMs = Date.now();
      }
      _inFlightGpuPresents += 1;
    }
    try {
      gpuWorker.postMessage(gpuMsg, [transferBuf]);
    } catch (err) {
      if (batchInfo.hasPresent) {
        _inFlightGpuPresents = Math.max(0, _inFlightGpuPresents - 1);
      }
      if (wantRecycle && pendingRecycleTxPorts.length > 0) {
        pendingRecycleTxPorts.pop();
      }
      throw err;
    }
    return true;
  }
  return false;
}

function handlePthreadTxMessage(data, _workerName, sourcePort) {
  if (!data || typeof data !== 'object') {
    return false;
  }
  if (data.type === PTHREAD_TX_MSG_WGPU_SUBMIT) {
    return handlePthreadWgpuSubmit(data, { ownedArrayBuffer: true, sourcePort });
  }
  if (data.type === PTHREAD_TX_MSG_FAULT_RECORD) {
    const rawLine =
      typeof data.rawLine === 'string'
        ? data.rawLine
        : typeof data.label === 'string'
          ? data.label
          : '';
    recordCoordinatorFaultRecord(data.faultRecord, {
      rawLine,
      label: typeof data.label === 'string' ? data.label : rawLine,
      source: 'pthread',
      workerName: typeof data.workerName === 'string' ? data.workerName : _workerName,
      consoleEmitted: data.consoleEmitted === true,
    });
    if (data.broadcastSent !== true && rawLine) {
      postProgress({ label: rawLine });
    }
    return true;
  }
  if (data.type === PTHREAD_TX_MSG_PROGRESS) {
    if (data.payload && typeof data.payload === 'object') {
      if (typeof data.payload.label === 'string' && isFaultRecordLine(data.payload.label)) {
        recordCoordinatorFaultRecord(null, {
          rawLine: data.payload.label,
          label: data.payload.label,
          source: 'pthread',
          workerName: _workerName,
        });
      }
      return postProgress(data.payload);
    }
    if (typeof data.label === 'string') {
      if (isFaultRecordLine(data.label)) {
        recordCoordinatorFaultRecord(null, {
          rawLine: data.label,
          label: data.label,
          source: 'pthread',
          workerName: _workerName,
        });
      }
      return postProgress({ label: data.label });
    }
    return false;
  }
  return false;
}

const coordinatorPthreadPool = createPthreadPoolCoordinator({
  getHttpfsMailboxSab: () => httpfsMailboxSab,
  getHttpfsMailboxSabs: () => httpfsMailboxSabs,
  getSharedModules: () => coordinatorSharedWasmModules.getSharedModulesArray(),
  getSharedModuleNames: () => coordinatorSharedWasmModules.getSharedModuleNames(),
  onPthreadTxMessage: handlePthreadTxMessage,
});

const coordinatorAudioBridge = createAudioWorkerBridge({
  onResetWorker: (prevWorker) => {
    resetAttachedRuntimeAudioRing(prevWorker);
  },
});

function teardownCoordinatorAudioWorker(options = {}) {
  const safeOpts = options && typeof options === 'object' ? options : {};
  const prevAudioWorker = audioWorker;
  audioWorker = null;
  audioWorkerReady = false;
  if (prevAudioWorker) {
    try {
      prevAudioWorker.onmessage = null;
    } catch {
      // Ignore non-writable onmessage on mock workers.
    }
    try {
      prevAudioWorker.onerror = null;
    } catch {
      // Ignore non-writable onerror on mock workers.
    }
    resetAttachedRuntimeAudioRing(prevAudioWorker);
    if (safeOpts.sendStop !== false && typeof prevAudioWorker.postMessage === 'function') {
      try {
        prevAudioWorker.postMessage({ type: 'stop', stop: true });
      } catch {
        // Ignore stop postMessage errors during teardown.
      }
    }
    if (safeOpts.terminate === true && typeof prevAudioWorker.terminate === 'function') {
      try {
        prevAudioWorker.terminate();
      } catch {
        // Ignore terminate errors during teardown.
      }
    }
  }
  if (safeOpts.destroyBridge === true) {
    coordinatorAudioBridge.destroy({ sendStop: false });
  } else {
    coordinatorAudioBridge.reset();
  }
}

function getPthreadTelemetry() {
  return coordinatorPthreadPool.getTelemetry();
}

function toSafeIoNonNegInt(val, fallback = 0) {
  const n = typeof val === 'bigint' ? Number(val) : Number(val);
  if (!Number.isFinite(n) || n < 0) return fallback;
  return Math.floor(n);
}

function toSafeIoNonNegNum(val, fallback = 0) {
  const n = typeof val === 'bigint' ? Number(val) : Number(val);
  if (!Number.isFinite(n) || n < 0) return fallback;
  return n;
}

function sanitizeIoWorkerStats(rawStats, mailboxState = null) {
  const src = rawStats && typeof rawStats === 'object' ? rawStats : {};
  const mailbox = src.mailbox && typeof src.mailbox === 'object' ? src.mailbox : {};
  const streaming = src.streaming && typeof src.streaming === 'object' ? src.streaming : {};
  const imm = src.immutableCache && typeof src.immutableCache === 'object' ? src.immutableCache : {};
  const mState = mailboxState && typeof mailboxState === 'object' ? mailboxState : null;

  const attached =
    typeof src.mailboxAttached === 'boolean'
      ? src.mailboxAttached
      : mState && typeof mState.mailboxAttached === 'boolean'
        ? mState.mailboxAttached
        : false;
  const rawLanes =
    src.mailboxLanes !== undefined
      ? src.mailboxLanes
      : mState && mState.mailboxLanes !== undefined
        ? mState.mailboxLanes
        : 0;
  const inFlightRequests = toSafeIoNonNegInt(
    src.inFlightRequests !== undefined
      ? src.inFlightRequests
      : mailbox.activeLanes !== undefined
        ? mailbox.activeLanes
        : imm.inFlightRequests !== undefined
          ? imm.inFlightRequests
          : mailbox.activePendingSlots,
    0,
  );

  return {
    mailboxAttached: Boolean(attached),
    mailboxLanes: toSafeIoNonNegInt(rawLanes, 0),
    openHandles: toSafeIoNonNegInt(src.openHandles, 0),
    readsCompleted: toSafeIoNonNegInt(
      src.readsCompleted !== undefined ? src.readsCompleted : streaming.readsCompleted,
      0,
    ),
    bytesTransferred: toSafeIoNonNegInt(
      src.bytesTransferred !== undefined ? src.bytesTransferred : streaming.bytesTransferred,
      0,
    ),
    inFlightRequests,
    activeLanes: toSafeIoNonNegInt(
      mailbox.activeLanes !== undefined ? mailbox.activeLanes : src.activeLanes,
      0,
    ),
    peakActiveLanes: toSafeIoNonNegInt(
      mailbox.peakActiveLanes !== undefined ? mailbox.peakActiveLanes : src.peakActiveLanes,
      0,
    ),
    requests: toSafeIoNonNegInt(
      mailbox.totalRequests !== undefined
        ? mailbox.totalRequests
        : mailbox.requests !== undefined
          ? mailbox.requests
          : src.requests,
      0,
    ),
    syncFastPathHits: toSafeIoNonNegInt(
      src.syncFastPathHits !== undefined ? src.syncFastPathHits : mailbox.syncFastPathHits,
      0,
    ),
    asyncPathHits: toSafeIoNonNegInt(
      mailbox.asyncPathHits !== undefined ? mailbox.asyncPathHits : src.asyncPathHits,
      0,
    ),
    errorCount: toSafeIoNonNegInt(
      src.errorCount !== undefined ? src.errorCount : streaming.errorCount,
      0,
    ),
    cacheHits: toSafeIoNonNegInt(
      src.cacheHits !== undefined ? src.cacheHits : streaming.cacheHits,
      0,
    ),
    cacheMisses: toSafeIoNonNegInt(src.cacheMisses, 0),
    bytesServedFromCache: toSafeIoNonNegInt(src.bytesServedFromCache, 0),
    bytesFetched: toSafeIoNonNegInt(src.bytesFetched, 0),
    slabCacheHits: toSafeIoNonNegInt(
      src.slabCacheHits !== undefined ? src.slabCacheHits : imm.slabCacheHits,
      0,
    ),
    slabCacheMisses: toSafeIoNonNegInt(
      src.slabCacheMisses !== undefined ? src.slabCacheMisses : imm.slabCacheMisses,
      0,
    ),
    opfsHits: toSafeIoNonNegInt(src.opfsHits, 0),
    opfsMisses: toSafeIoNonNegInt(src.opfsMisses, 0),
    avgReadMs: toSafeIoNonNegNum(
      src.avgReadMs !== undefined ? src.avgReadMs : streaming.avgReadMs,
      0,
    ),
    statHits: toSafeIoNonNegInt(src.statHits, 0),
    statMisses: toSafeIoNonNegInt(src.statMisses, 0),
  };
}

function buildCoordinatorIoWorkerStats() {
  const lanes =
    ioWorkerMailboxLanes || httpfsMailboxSabs.length || (httpfsMailboxSab ? 1 : 0);
  if (lastIoWorkerStats) {
    return Object.assign({}, lastIoWorkerStats, {
      mailboxAttached: Boolean(ioWorkerMailboxAttached || lastIoWorkerStats.mailboxAttached),
      mailboxLanes: ioWorkerMailboxLanes || lastIoWorkerStats.mailboxLanes || lanes,
    });
  }
  return sanitizeIoWorkerStats(null, {
    mailboxAttached: ioWorkerMailboxAttached,
    mailboxLanes: lanes,
  });
}

function createRuntimePrewarmBridge(moduleOrExports, options = {}) {
  if (!prewarmBridgeApi || typeof prewarmBridgeApi.PrewarmBridge !== 'function') {
    return null;
  }
  const opts = options && typeof options === 'object' ? Object.assign({}, options) : {};
  if (moduleOrExports && typeof moduleOrExports === 'object') {
    opts.module = moduleOrExports;
  }
  if (typeof opts.getMemory !== 'function') {
    opts.getMemory = () => resolveRuntimeMemoryBuffer(opts.module, opts.memory || null);
  }
  return new prewarmBridgeApi.PrewarmBridge(opts);
}

function loadPrewarmWarmList(moduleOrExports, bytes, options = {}) {
  if (!prewarmBridgeApi || typeof prewarmBridgeApi.resolvePrewarmWasmApi !== 'function') {
    return -1;
  }
  const opts = options && typeof options === 'object' ? Object.assign({}, options) : {};
  if (typeof opts.getMemory !== 'function') {
    opts.getMemory = () => resolveRuntimeMemoryBuffer(moduleOrExports, opts.memory || null);
  }
  const api = prewarmBridgeApi.resolvePrewarmWasmApi(moduleOrExports, opts);
  if (!api.hasLoadList) {
    return -1;
  }
  const parsed = prewarmBridgeApi.parseWarmList(bytes);
  if (!parsed.ok) {
    return -1;
  }
  try {
    return api.loadList(bytes);
  } catch {
    return -1;
  }
}

function readPrewarmProgress(moduleOrExports, options = {}) {
  if (!prewarmBridgeApi || typeof prewarmBridgeApi.resolvePrewarmWasmApi !== 'function') {
    return null;
  }
  if (!moduleOrExports || typeof moduleOrExports !== 'object') {
    return null;
  }
  const opts = options && typeof options === 'object' ? Object.assign({}, options) : {};
  if (typeof opts.getMemory !== 'function') {
    opts.getMemory = () => resolveRuntimeMemoryBuffer(moduleOrExports, opts.memory || null);
  }
  const api = prewarmBridgeApi.resolvePrewarmWasmApi(moduleOrExports, opts);
  if (!api.hasProgress) {
    return null;
  }
  try {
    return api.progress();
  } catch {
    return null;
  }
}

function savePrewarmWarmList(moduleOrExports, options = {}) {
  if (!prewarmBridgeApi || typeof prewarmBridgeApi.resolvePrewarmWasmApi !== 'function') {
    return null;
  }
  if (!moduleOrExports || typeof moduleOrExports !== 'object') {
    return null;
  }
  const opts = options && typeof options === 'object' ? Object.assign({}, options) : {};
  if (typeof opts.getMemory !== 'function') {
    opts.getMemory = () => resolveRuntimeMemoryBuffer(moduleOrExports, opts.memory || null);
  }
  const api = prewarmBridgeApi.resolvePrewarmWasmApi(moduleOrExports, opts);
  if (!api.hasSaveList) {
    return null;
  }
  try {
    return api.saveList();
  } catch {
    return null;
  }
}

function readRuntimeDiagnostics(moduleOrExports) {
  if (!moduleOrExports || typeof moduleOrExports !== 'object') {
    return null;
  }
  const fn =
    typeof moduleOrExports._playgta6_runtime_diagnostics_json === 'function'
      ? moduleOrExports._playgta6_runtime_diagnostics_json
      : typeof moduleOrExports.playgta6_runtime_diagnostics_json === 'function'
        ? moduleOrExports.playgta6_runtime_diagnostics_json
        : moduleOrExports.asm &&
            typeof moduleOrExports.asm.playgta6_runtime_diagnostics_json === 'function'
          ? moduleOrExports.asm.playgta6_runtime_diagnostics_json
          : null;
  if (!fn) {
    return null;
  }
  try {
    const rawPtr = fn();
    const ptr = ptrToNumber(rawPtr, 'runtimeDiagnosticsPtr');
    if (!Number.isSafeInteger(ptr) || ptr <= 0) {
      return null;
    }
    let heapU8 = null;
    if (moduleOrExports.HEAPU8 instanceof Uint8Array) {
      heapU8 = moduleOrExports.HEAPU8;
    } else if (moduleOrExports.wasmMemory && moduleOrExports.wasmMemory.buffer) {
      heapU8 = new Uint8Array(moduleOrExports.wasmMemory.buffer);
    } else if (moduleOrExports.memory && moduleOrExports.memory.buffer) {
      heapU8 = new Uint8Array(moduleOrExports.memory.buffer);
    }
    if (!heapU8 || ptr >= heapU8.byteLength) {
      return null;
    }
    const maxLen = Math.min(heapU8.byteLength - ptr, 256 * 1024);
    let end = ptr;
    const limit = ptr + maxLen;
    while (end < limit && heapU8[end] !== 0) {
      end += 1;
    }
    if (end <= ptr) {
      return null;
    }
    const copy = new Uint8Array(end - ptr);
    copy.set(heapU8.subarray(ptr, end));
    const parsed = JSON.parse(utf8Decoder.decode(copy));
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

function handleIoWorkerMessage(data, hooks = {}) {
  if (!data || typeof data !== 'object') {
    return false;
  }
  const state =
    hooks.state && typeof hooks.state === 'object'
      ? hooks.state
      : {
          get mailboxAttached() {
            return ioWorkerMailboxAttached;
          },
          set mailboxAttached(v) {
            ioWorkerMailboxAttached = Boolean(v);
          },
          get mailboxLanes() {
            return ioWorkerMailboxLanes;
          },
          set mailboxLanes(v) {
            ioWorkerMailboxLanes = toSafeIoNonNegInt(v, 0);
          },
          get lastStats() {
            return lastIoWorkerStats;
          },
          set lastStats(v) {
            lastIoWorkerStats = v;
          },
        };
  const emitToMain =
    typeof hooks.postToMain === 'function' ? hooks.postToMain : coordinatorPostToMain;
  const emitProgress = typeof hooks.postProgress === 'function' ? hooks.postProgress : postProgress;
  const onReady = typeof hooks.onWorkerReady === 'function' ? hooks.onWorkerReady : null;
  const requestStats = typeof hooks.requestStats === 'function' ? hooks.requestStats : null;

  if (data.ready) {
    if (onReady) onReady('io_worker', data.summary);
    return true;
  }
  if (data.mailboxAttached === true) {
    state.mailboxAttached = true;
    if (data.mailboxLanes !== undefined) {
      state.mailboxLanes = toSafeIoNonNegInt(data.mailboxLanes, state.mailboxLanes);
    }
    const snapshot = sanitizeIoWorkerStats(state.lastStats, {
      mailboxAttached: true,
      mailboxLanes: state.mailboxLanes,
    });
    state.lastStats = snapshot;
    if (emitToMain) {
      try {
        emitToMain({
          type: 'io-stats',
          mailboxAttached: true,
          mailboxLanes: snapshot.mailboxLanes,
          ioWorkerStats: snapshot,
          timestamp: Date.now(),
        });
      } catch {
        // Non-fatal IO telemetry forward error.
      }
    }
    if (requestStats) {
      try {
        requestStats();
      } catch {
        // Non-fatal stats poll error.
      }
    }
    return true;
  }
  if (data.mailboxDetached === true) {
    state.mailboxAttached = false;
    state.mailboxLanes = 0;
    const snapshot = sanitizeIoWorkerStats(state.lastStats, {
      mailboxAttached: false,
      mailboxLanes: 0,
    });
    state.lastStats = snapshot;
    if (emitToMain) {
      try {
        emitToMain({
          type: 'io-stats',
          mailboxAttached: false,
          mailboxLanes: 0,
          ioWorkerStats: snapshot,
          timestamp: Date.now(),
        });
      } catch {
        // Non-fatal IO telemetry forward error.
      }
    }
    return true;
  }
  if (data.stats && typeof data.stats === 'object') {
    const snapshot = sanitizeIoWorkerStats(data.stats, {
      mailboxAttached:
        typeof data.stats.mailboxAttached === 'boolean'
          ? data.stats.mailboxAttached
          : state.mailboxAttached,
      mailboxLanes:
        data.stats.mailboxLanes !== undefined ? data.stats.mailboxLanes : state.mailboxLanes,
    });
    state.mailboxAttached = snapshot.mailboxAttached;
    if (snapshot.mailboxLanes > 0) {
      state.mailboxLanes = snapshot.mailboxLanes;
    }
    state.lastStats = snapshot;
    if (emitToMain) {
      const payload = {
        type: 'io-stats',
        mailboxAttached: snapshot.mailboxAttached,
        mailboxLanes: snapshot.mailboxLanes,
        ioWorkerStats: snapshot,
        stats: snapshot,
        timestamp: Date.now(),
      };
      if (data.id !== undefined) {
        payload.id = data.id;
      }
      try {
        emitToMain(payload);
      } catch {
        // Non-fatal IO stats forward error.
      }
    }
    return true;
  }
  if (data.error) {
    emitProgress({ error: `io_worker: ${data.error}` });
    return true;
  }
  return false;
}

function getLoaderStatus() {
  const pthread = coordinatorPthreadPool.getTelemetry();
  const audioBridgeState = coordinatorAudioBridge.getState();
  const sideModules =
    typeof activeSideModuleProgressGetter === 'function'
      ? activeSideModuleProgressGetter()
      : {
          loadedCount: 0,
          totalCount: activeGuestManifestSummary ? activeGuestManifestSummary.sideModuleCount : 0,
          remainingCount: activeGuestManifestSummary
            ? activeGuestManifestSummary.sideModuleCount
            : 0,
          complete: !activeGuestManifestSummary || activeGuestManifestSummary.sideModuleCount === 0,
          pct:
            !activeGuestManifestSummary || activeGuestManifestSummary.sideModuleCount === 0
              ? 100
              : 0,
          loadedModules: [],
          expectedModules: activeGuestManifestSummary
            ? activeGuestManifestSummary.sideModules.slice()
            : [],
        };
  const audioRingTelemetry =
    activeCoordinatorModuleConfig &&
    typeof activeCoordinatorModuleConfig.getAudioRingTelemetry === 'function'
      ? activeCoordinatorModuleConfig.getAudioRingTelemetry()
      : null;
  return {
    pthreadSpawnedCount: pthread.pthreadSpawnedCount,
    pthreadActiveCount: pthread.pthreadActiveCount,
    pthreadPeakCount: pthread.pthreadPeakCount,
    pthreadLoadQueueDepth: pthread.pthreadLoadQueueDepth,
    pthreadFailedCount: pthread.pthreadFailedCount,
    pthreadLoadedCount: pthread.pthreadLoadedCount,
    pthreadActiveLoads: pthread.pthreadActiveLoads,
    pthreadPeakConcurrentLoads: pthread.pthreadPeakConcurrentLoads,
    maxConcurrentPthreadLoads: pthread.maxConcurrentPthreadLoads,
    httpfsMailboxLanes: httpfsMailboxSabs.length || (httpfsMailboxSab ? 1 : 0),
    mailboxAttached: Boolean(ioWorkerMailboxAttached),
    ioWorkerStats: buildCoordinatorIoWorkerStats(),
    runtimeDiagnostics: readRuntimeDiagnostics(activeCoordinatorRuntimeModule),
    indirectDispatch: activeGuestManifestSummary
      ? activeGuestManifestSummary.indirectDispatch
      : 'table',
    wasmMetrics:
      activeGuestManifestSummary && activeGuestManifestSummary.wasmMetrics
        ? Object.assign({}, activeGuestManifestSummary.wasmMetrics)
        : null,
    manifest: activeGuestManifestSummary
      ? {
          indirectDispatch: activeGuestManifestSummary.indirectDispatch,
          sideModules: activeGuestManifestSummary.sideModules.slice(),
          sideModuleCount: activeGuestManifestSummary.sideModuleCount,
          wasmMetrics: activeGuestManifestSummary.wasmMetrics
            ? Object.assign({}, activeGuestManifestSummary.wasmMetrics)
            : null,
        }
      : null,
    sideModules,
    firstPresentEmitted: coordinatorFirstPresentEmitted,
    inFlightGpuPresents: _inFlightGpuPresents,
    maxInFlightPresents: _maxInFlightGpuPresents,
    coalescedPresentSubmissions: _coalescedPresentSubmissions,
    audioRingAttached: audioRingTelemetry ? Boolean(audioRingTelemetry.audioRingAttached) : false,
    audioRingRetryCount: audioRingTelemetry ? audioRingTelemetry.audioRingRetryCount : 0,
    audioRingLastReason: audioRingTelemetry ? audioRingTelemetry.audioRingLastReason : null,
    audioRingSabAttached: Boolean(audioBridgeState.ringSabAttached),
    audioContextState: audioBridgeState.outputState,
    renderStateStats: lastRenderStateStats ? Object.assign({}, lastRenderStateStats) : null,
    wireBytesReleased: wasmMemoryTelemetry.wireBytesReleased,
    droppedWasmBuffers: wasmMemoryTelemetry.droppedWasmBuffers,
    droppedPreloadBuffers: wasmMemoryTelemetry.droppedPreloadBuffers,
    transferredArrayBuffers: wasmMemoryTelemetry.transferredArrayBuffers,
    streamingCompiles: wasmMemoryTelemetry.streamingCompiles,
    fallbackCompiles: wasmMemoryTelemetry.fallbackCompiles,
    sharedModuleCount: coordinatorSharedWasmModules.size,
    v8SideModuleCapBytes: V8_SIDE_MODULE_CAP_BYTES,
    largestSideModuleBytes: wasmMemoryTelemetry.largestSideModuleBytes,
    largestSideModuleName: wasmMemoryTelemetry.largestSideModuleName,
    sideModuleHeadroomBytes: Math.max(
      0,
      V8_SIDE_MODULE_CAP_BYTES - wasmMemoryTelemetry.largestSideModuleBytes,
    ),
    sideModuleCapWarnings: wasmMemoryTelemetry.sideModuleCapWarnings,
    peakWireBytesInFlight: wasmMemoryTelemetry.peakWireBytesInFlight,
    sharedModulesDeliveredToPthreads: wasmMemoryTelemetry.sharedModulesDeliveredToPthreads,
    lastFaultRecord: getLastFaultRecord(),
    pipelineDumpStats:
      activeCoordinatorPipelineDumpSyncer &&
      typeof activeCoordinatorPipelineDumpSyncer.getStats === 'function'
        ? activeCoordinatorPipelineDumpSyncer.getStats()
        : null,
    runtimeExports:
      activeCoordinatorModuleConfig &&
      typeof activeCoordinatorModuleConfig.getRuntimeExportsSummary === 'function'
        ? activeCoordinatorModuleConfig.getRuntimeExportsSummary()
        : activeCoordinatorRuntimeModule
          ? inspectRuntimeExports(activeCoordinatorRuntimeModule)
          : null,
    prewarmProgress:
      activeCoordinatorModuleConfig &&
      typeof activeCoordinatorModuleConfig.getPrewarmProgress === 'function'
        ? activeCoordinatorModuleConfig.getPrewarmProgress()
        : readPrewarmProgress(activeCoordinatorRuntimeModule),
    prewarmDiagnostics:
      activeCoordinatorModuleConfig &&
      typeof activeCoordinatorModuleConfig.getPrewarmDiagnostics === 'function'
        ? activeCoordinatorModuleConfig.getPrewarmDiagnostics()
        : null,
  };
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

    let manifestSummary = null;
    if (configuredGuestManifest !== null) {
      manifestSummary = parseGuestManifest(configuredGuestManifest);
      const cfgBudget = evaluateV8SideModuleBudget(configuredGuestManifest);
      if (cfgBudget.largestSideModuleBytes > 0) {
        recordSideModuleByteSize(
          cfgBudget.largestSideModuleName || 'guest_part_0.wasm',
          cfgBudget.largestSideModuleBytes,
          report,
        );
      }
    } else if (typeof fetch === 'function') {
      manifestSummary = await loadGuestManifest({
        url: configuredGuestManifestUrl || DEFAULT_GUEST_MANIFEST_RELATIVE,
        fetchFn: fetch,
        reportFn: report,
      });
    }
    if (manifestSummary) {
      activeGuestManifestSummary = manifestSummary;
      if (
        configuredMaxConcurrentPthreadLoads === null &&
        manifestSummary.maxConcurrentPthreadLoads !== null
      ) {
        coordinatorPthreadPool.setMaxConcurrentPthreadLoads(
          manifestSummary.maxConcurrentPthreadLoads,
        );
      }
      postProgress({
        metrics: {
          indirectDispatch: manifestSummary.indirectDispatch,
          sideModuleCount: manifestSummary.sideModuleCount,
          sideModuleUrls: manifestSummary.sideModules.slice(),
          wasmMetrics: manifestSummary.wasmMetrics
            ? Object.assign({}, manifestSummary.wasmMetrics)
            : null,
        },
      });
    }

    report('Loading Emscripten Wasm64 runtime (dist/runtime.js)...');
    importScripts('./dist/runtime.js');
    if (typeof self.createRuntimeModule === 'function') {
      if (pendingPreloadedGuestBuffers) {
        const toIngest = pendingPreloadedGuestBuffers;
        pendingPreloadedGuestBuffers = null;
        try {
          await coordinatorSharedWasmModules.ingestPreloadedBuffers(toIngest, { fetchFn: fetch });
        } finally {
          releaseWireBytesFromObject(toIngest, { assumeWasm: true });
        }
      }

      // Precompile split-module side modules (`guest_part_*.wasm`) ONCE via WebAssembly.compileStreaming
      // so neither the coordinator nor any dynamically spawned pthread worker ever allocates a
      // ~908 MB `ArrayBuffer` during `loadDynamicLibrary` (#663, #1077).
      const candidateSideModules = [];
      const seenSideKeys = new Set();
      const collectSideModule = (rawEntry) => {
        const entryUrl = extractSideModuleEntryUrl(rawEntry);
        if (!entryUrl) {
          return;
        }
        const key = (extractWasmBasename(entryUrl) || entryUrl).toLowerCase();
        if (!key || seenSideKeys.has(key) || coordinatorSharedWasmModules.hasModule(key)) {
          return;
        }
        seenSideKeys.add(key);
        candidateSideModules.push(entryUrl);
      };
      if (Array.isArray(configuredDynamicLibraries)) {
        for (let i = 0; i < configuredDynamicLibraries.length; i += 1) {
          collectSideModule(configuredDynamicLibraries[i]);
        }
      }
      if (Array.isArray(configuredSideModuleUrls)) {
        for (let i = 0; i < configuredSideModuleUrls.length; i += 1) {
          collectSideModule(configuredSideModuleUrls[i]);
        }
      }
      if (manifestSummary && Array.isArray(manifestSummary.sideModules)) {
        for (let i = 0; i < manifestSummary.sideModules.length; i += 1) {
          collectSideModule(manifestSummary.sideModules[i]);
        }
      }

      const resolvedRuntimeUrlForSide = resolveRuntimeScriptUrl(
        manifestSummary && manifestSummary.runtimeScriptUrl
          ? resolveSideModuleUrl(manifestSummary.runtimeScriptUrl, DEFAULT_RUNTIME_SCRIPT_RELATIVE)
          : undefined,
        self.location && typeof self.location.href === 'string' ? self.location.href : '',
      );
      if (
        candidateSideModules.length > 0 &&
        typeof fetch === 'function' &&
        typeof WebAssembly !== 'undefined' &&
        typeof WebAssembly.compile === 'function'
      ) {
        try {
          await coordinatorSharedWasmModules.precompileSideModules(candidateSideModules, {
            fetchFn: fetch,
            reportFn: report,
            resolveUrl: (u) => resolveSideModuleUrl(u, resolvedRuntimeUrlForSide),
          });
        } catch (precompileErr) {
          report(
            `side-module streaming precompile skipped (falling back to Emscripten loader): ${precompileErr}`,
          );
        }
      }

      restoreCoordinatorSharedModuleShim();
      activeCoordinatorSharedModuleShimRestore = installSharedWasmModuleShim(self, {
        resolveSharedModule: (urlOrName) => coordinatorSharedWasmModules.getModule(urlOrName),
        onModuleCompiled: (urlOrName, compiledMod) => {
          const base = extractWasmBasename(urlOrName);
          if (base && GUEST_SIDE_MODULE_BASENAME_RE.test(base)) {
            coordinatorSharedWasmModules.setModule(base, compiledMod);
          }
        },
      });

      const runtimeOptions = {
        onFirstPresent: () => {
          coordinatorFirstPresentEmitted = true;
        },
        onExit: (status) => {
          const relaunch = guestRelaunchPayload(status);
          if (relaunch) {
            postProgress(relaunch);
          }
          restoreCoordinatorSharedModuleShim();
          coordinatorPthreadPool.closeTxPorts();
          coordinatorSharedWasmModules.clear();
          coordinatorAudioBridge.clearPending();
          if (audioWorker) {
            resetAttachedRuntimeAudioRing(audioWorker);
          }
          pendingPreloadedWasmBinary = null;
          pendingPreloadedGuestBuffers = null;
          activeCoordinatorRuntimeModule = null;
          activeCoordinatorModuleConfig = null;
        },
        onAbort: (what) => {
          if (
            activeCoordinatorPipelineDumpSyncer &&
            typeof activeCoordinatorPipelineDumpSyncer.stop === 'function'
          ) {
            try {
              const p = activeCoordinatorPipelineDumpSyncer.stop();
              if (p && typeof p.catch === 'function') {
                p.catch(() => {});
              }
            } catch {
              // Ignore dump flush errors during abort teardown.
            }
          }
          restoreCoordinatorSharedModuleShim();
          coordinatorPthreadPool.closeTxPorts();
          coordinatorSharedWasmModules.clear();
          coordinatorAudioBridge.clearPending();
          if (audioWorker) {
            resetAttachedRuntimeAudioRing(audioWorker);
          }
          pendingPreloadedWasmBinary = null;
          pendingPreloadedGuestBuffers = null;
          activeCoordinatorRuntimeModule = null;
          activeCoordinatorModuleConfig = null;
          postProgress({ error: `runtime abort: ${what}` });
        },
        getAudioWorker: () => audioWorker,
        isAudioWorkerReady: () => audioWorkerReady,
        onWasmError: (err) => {
          postProgress({ error: `runtime wasm: ${err}` });
        },
      };
      if (pendingPreloadedWasmBinary) {
        runtimeOptions.wasmBinary = pendingPreloadedWasmBinary;
        pendingPreloadedWasmBinary = null;
      }
      const precompiledSharedModules = coordinatorSharedWasmModules.getSharedModulesArray();
      if (precompiledSharedModules.length > 0) {
        runtimeOptions.sharedModules = precompiledSharedModules;
      }
      if (manifestSummary) {
        runtimeOptions.guestManifest = manifestSummary;
      }
      if (configuredSideModuleUrls) {
        runtimeOptions.sideModuleUrls = configuredSideModuleUrls;
      }
      if (configuredDynamicLibraries) {
        runtimeOptions.dynamicLibraries = configuredDynamicLibraries;
      }
      if (configuredDumpPipelines !== undefined) {
        runtimeOptions.dumpPipelines = configuredDumpPipelines;
      }
      if (configuredDumpDir !== undefined) {
        runtimeOptions.dumpDir = configuredDumpDir;
      }
      if (configuredDumpEndpoint !== undefined) {
        runtimeOptions.dumpEndpoint = configuredDumpEndpoint;
      }
      if (configuredDumpIntervalMs !== undefined) {
        runtimeOptions.dumpIntervalMs = configuredDumpIntervalMs;
      }
      if (configuredDumpMaxConcurrentUploads !== undefined) {
        runtimeOptions.dumpMaxConcurrentUploads = configuredDumpMaxConcurrentUploads;
      }
      if (activeCoordinatorPipelineDumpSyncer) {
        try {
          activeCoordinatorPipelineDumpSyncer.stop().catch(() => {});
        } catch {
          // Ignore teardown errors from a previous runtime boot.
        }
        activeCoordinatorPipelineDumpSyncer = null;
      }
      if (activeCoordinatorModuleConfig) {
        try {
          if (typeof activeCoordinatorModuleConfig.destroy === 'function') {
            activeCoordinatorModuleConfig.destroy();
          } else if (typeof activeCoordinatorModuleConfig.resetAudioRingAttachment === 'function') {
            activeCoordinatorModuleConfig.resetAudioRingAttachment();
          }
        } catch {
          // Ignore teardown errors from a previous runtime boot.
        }
        activeCoordinatorModuleConfig = null;
      }
      const moduleConfig = createRuntimeModuleOptions(runtimeOptions);
      activeCoordinatorModuleConfig = moduleConfig;
      if (typeof moduleConfig.getPipelineDumpSyncer === 'function') {
        activeCoordinatorPipelineDumpSyncer = moduleConfig.getPipelineDumpSyncer();
      }
      if (typeof moduleConfig.getSideModuleProgress === 'function') {
        activeSideModuleProgressGetter = moduleConfig.getSideModuleProgress;
      }
      if (activeCoordinatorInputBridge) {
        try {
          activeCoordinatorInputBridge.stop();
        } catch {
          // Ignore teardown errors from a previous runtime boot.
        }
        activeCoordinatorInputBridge = null;
      }
      const runtimeMod = await self.createRuntimeModule(moduleConfig);
      activeCoordinatorRuntimeModule = runtimeMod;
      if (typeof moduleConfig.attachGuestMemory === 'function') {
        moduleConfig.attachGuestMemory(runtimeMod);
      }
      if (typeof moduleConfig.attachInputBridge === 'function') {
        const attached = moduleConfig.attachInputBridge(runtimeMod);
        if (attached && attached.active) {
          activeCoordinatorInputBridge = attached;
        }
      } else if (typeof moduleConfig.getInputBridge === 'function') {
        const existing = moduleConfig.getInputBridge();
        if (existing && existing.active) {
          activeCoordinatorInputBridge = existing;
        }
      }
      if (typeof moduleConfig.attachAudioRing === 'function') {
        moduleConfig.attachAudioRing(runtimeMod);
      }
    }
  } catch (err) {
    restoreCoordinatorSharedModuleShim();
    // W371 (#1077): Release any unconsumed preloaded wire bytes on boot failure so nothing leaks.
    if (pendingPreloadedWasmBinary) {
      const binInfo = getBinaryWireInfo(pendingPreloadedWasmBinary);
      pendingPreloadedWasmBinary = null;
      if (binInfo && binInfo.byteLength > 0) {
        recordDroppedWasmWireBytes(binInfo.byteLength, 1);
      }
    }
    if (pendingPreloadedGuestBuffers) {
      const leftover = pendingPreloadedGuestBuffers;
      pendingPreloadedGuestBuffers = null;
      releaseWireBytesFromObject(leftover, { assumeWasm: true });
    }
    postProgress({ error: `runtime boot: ${err}` });
  }
}

if (typeof self !== 'undefined') {
  const OriginalWorker =
    typeof self.Worker === 'function'
      ? self.Worker
      : typeof Worker === 'function'
        ? Worker
        : null;
  if (OriginalWorker) {
    const proxyBundle = installPthreadWorkerProxy({
      WorkerCtor: OriginalWorker,
      coordinator: coordinatorPthreadPool,
      getRedirectHref: () =>
        self.location && typeof self.location.href === 'string' ? self.location.href : '',
    });
    const WrappedWorker = proxyBundle ? proxyBundle.WrappedWorker : null;
    if (WrappedWorker) {
      try {
        self.Worker = WrappedWorker;
      } catch {
        // Ignore if self.Worker is non-writable in a custom harness.
      }
      if (
        typeof globalThis !== 'undefined' &&
        globalThis !== self &&
        globalThis.Worker === OriginalWorker
      ) {
        try {
          globalThis.Worker = WrappedWorker;
        } catch {
          // Ignore if globalThis.Worker is non-writable.
        }
      }
    }
  }

  const postToMain =
    typeof self.postMessage === 'function' ? (payload) => self.postMessage(payload) : null;
  coordinatorPostToMain = postToMain;

  const wgpuPthreadChannel =
    typeof BroadcastChannel === 'function'
      ? tryOpenBroadcastChannel(BroadcastChannel, PTHREAD_WGPU_CHANNEL_NAME)
      : null;
  const httpfsSabChannel =
    typeof BroadcastChannel === 'function'
      ? tryOpenBroadcastChannel(BroadcastChannel, PTHREAD_HTTPFS_SAB_CHANNEL_NAME)
      : null;

  const pickBroadcastHttpfsLaneSab = () => {
    if (Array.isArray(httpfsMailboxSabs) && httpfsMailboxSabs.length > 0) {
      const idx = nextHttpfsBroadcastLane % httpfsMailboxSabs.length;
      nextHttpfsBroadcastLane += 1;
      return httpfsMailboxSabs[idx] || httpfsMailboxSab;
    }
    return httpfsMailboxSab;
  };

  const broadcastHttpfsSab = (assignedLaneSab) => {
    const laneSab = assignedLaneSab || httpfsMailboxSab;
    if (
      !httpfsSabChannel ||
      !laneSab ||
      typeof httpfsSabChannel.postMessage !== 'function'
    ) {
      return;
    }
    try {
      const payload = { type: 'httpfs-sab', sab: laneSab };
      if (Array.isArray(httpfsMailboxSabs) && httpfsMailboxSabs.length > 0) {
        payload.sabs = httpfsMailboxSabs.slice();
      }
      httpfsSabChannel.postMessage(payload);
    } catch {
      // Some runtimes reject SharedArrayBuffer over BroadcastChannel; degrade safely.
    }
  };

  if (wgpuPthreadChannel) {
    wgpuPthreadChannel.onmessage = (ev) => {
      handlePthreadWgpuSubmit(ev && ev.data);
    };
  }

  if (httpfsSabChannel) {
    httpfsSabChannel.onmessage = (ev) => {
      const msg = ev && ev.data;
      if (msg && msg.type === 'request-httpfs-sab') {
        broadcastHttpfsSab(pickBroadcastHttpfsLaneSab());
      }
    };
  }

  self.onmessage = (ev) => {
    const msg = ev.data || {};
    if (msg.type === 'request-loader-status' || msg.type === 'request-pthread-telemetry') {
      if (ioWorker && typeof ioWorker.postMessage === 'function') {
        try {
          ioWorker.postMessage({ stats: true });
        } catch {
          // Non-fatal io_worker stats refresh failure.
        }
      }
      if (postToMain) {
        const payload = Object.assign({ type: 'loader-status' }, getLoaderStatus(), {
          timestamp: Date.now(),
        });
        if (msg.id !== undefined) {
          payload.id = msg.id;
        }
        try {
          postToMain(payload);
        } catch {
          // Non-fatal status reply failure.
        }
      }
      return;
    }
    if (msg.type === 'request-io-stats') {
      if (ioWorker && typeof ioWorker.postMessage === 'function') {
        const ioReq = { stats: true };
        if (msg.id !== undefined) {
          ioReq.id = msg.id;
        }
        try {
          ioWorker.postMessage(ioReq);
        } catch {
          // Non-fatal io_worker stats query failure.
        }
      } else if (postToMain) {
        const snapshot = buildCoordinatorIoWorkerStats();
        const payload = {
          type: 'io-stats',
          mailboxAttached: snapshot.mailboxAttached,
          mailboxLanes: snapshot.mailboxLanes,
          ioWorkerStats: snapshot,
          stats: snapshot,
          timestamp: Date.now(),
        };
        if (msg.id !== undefined) {
          payload.id = msg.id;
        }
        try {
          postToMain(payload);
        } catch {
          // Non-fatal io-stats reply failure.
        }
      }
      return;
    }
    if (
      msg.type === FLUSH_PIPELINE_DUMPS_MESSAGE_TYPE ||
      msg.type === FLUSH_PIPELINE_DUMP_MESSAGE_TYPE ||
      msg.flushPipelineDumps === true ||
      msg.flushPipelineDump === true
    ) {
      Promise.resolve(flushActiveCoordinatorPipelineDumps({ snapshotFiles: true }))
        .then((stats) => {
          if (postToMain) {
            const payload = {
              type: PIPELINE_DUMPS_FLUSHED_MESSAGE_TYPE,
              ok: true,
              stats,
              timestamp: Date.now(),
            };
            if (msg.id !== undefined) {
              payload.id = msg.id;
            }
            try {
              postToMain(payload);
            } catch {
              // Non-fatal dump flush reply failure.
            }
          }
        })
        .catch((err) => {
          if (postToMain) {
            const payload = {
              type: PIPELINE_DUMPS_FLUSHED_MESSAGE_TYPE,
              ok: false,
              error: String(err),
              timestamp: Date.now(),
            };
            if (msg.id !== undefined) {
              payload.id = msg.id;
            }
            try {
              postToMain(payload);
            } catch {
              // Non-fatal dump flush error reply failure.
            }
          }
        });
      return;
    }
    if (msg.type === 'request-last-fault-record') {
      if (postToMain) {
        const payload = {
          type: FAULT_RECORD_MESSAGE_TYPE,
          faultRecord: getLastFaultRecord(),
          rawLine: lastCoordinatorFaultRecordLine,
          timestamp: Date.now(),
        };
        if (msg.id !== undefined) {
          payload.id = msg.id;
        }
        try {
          postToMain(payload);
        } catch {
          // Non-fatal fault-record query reply failure.
        }
      }
      return;
    }
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
      if (ioWorker && typeof ioWorker.postMessage === 'function') {
        try {
          ioWorker.postMessage({ stats: true });
        } catch {
          // Non-fatal io_worker stats poll failure.
        }
      }
      return;
    }
    if (msg.resize && gpuWorker) {
      const resizePayload = { resize: true, width: msg.width, height: msg.height };
      if (
        typeof msg.devicePixelRatio === 'number' &&
        Number.isFinite(msg.devicePixelRatio) &&
        msg.devicePixelRatio > 0
      ) {
        resizePayload.devicePixelRatio = msg.devicePixelRatio;
      }
      gpuWorker.postMessage(resizePayload);
      return;
    }
    if (
      (msg.attachRing ||
        msg.drainRing ||
        msg.type === 'attach-ring' ||
        msg.type === 'drain-ring') &&
      gpuWorker &&
      typeof gpuWorker.postMessage === 'function'
    ) {
      attachActiveCoordinatorGuestMemory();
      gpuWorker.postMessage(msg);
      return;
    }

    if (!msg.canvas) {
      if (msg.type === 'audio-teardown') {
        teardownCoordinatorAudioWorker({ sendStop: true, terminate: true, destroyBridge: false });
        return;
      }
      if (msg.type === 'teardown' || msg.type === 'destroy') {
        restoreCoordinatorSharedModuleShim();
        resetGpuPresentBackpressure({ resetCounters: true });
        if (activeCoordinatorPipelineDumpSyncer) {
          try {
            activeCoordinatorPipelineDumpSyncer.stop().catch(() => {});
          } catch {
            // Ignore pipeline dump syncer teardown errors.
          }
          activeCoordinatorPipelineDumpSyncer = null;
        }
        coordinatorPthreadPool.closeTxPorts();
        coordinatorSharedWasmModules.clear();
        if (activeCoordinatorInputBridge) {
          try {
            activeCoordinatorInputBridge.stop();
          } catch {
            // Ignore input bridge teardown errors.
          }
          activeCoordinatorInputBridge = null;
        }
        if (activeCoordinatorModuleConfig) {
          try {
            if (typeof activeCoordinatorModuleConfig.destroy === 'function') {
              activeCoordinatorModuleConfig.destroy();
            } else if (typeof activeCoordinatorModuleConfig.resetAudioRingAttachment === 'function') {
              activeCoordinatorModuleConfig.resetAudioRingAttachment();
            }
          } catch {
            // Ignore module config teardown errors.
          }
        }
        activeCoordinatorRuntimeModule = null;
        activeCoordinatorModuleConfig = null;
        ioWorkerMailboxAttached = false;
        ioWorkerMailboxLanes = 0;
        lastIoWorkerStats = null;
        lastIoStatsPollMs = 0;
        teardownCoordinatorAudioWorker({
          sendStop: true,
          terminate: true,
          destroyBridge: msg.type === 'destroy',
        });
        return;
      }
      if (isAudioContextStateMessage(msg)) {
        coordinatorAudioBridge.handleAudioContextState(msg);
        return;
      }
      if (isAudioRingSabMessage(msg)) {
        coordinatorAudioBridge.handleAttachRingSab(msg);
        return;
      }
      return;
    }
    resetLastFaultRecord();
    restoreCoordinatorSharedModuleShim();
    resetGpuPresentBackpressure({
      resetCounters: true,
      maxInFlightPresents:
        msg.maxInFlightPresents !== undefined
          ? msg.maxInFlightPresents
          : DEFAULT_MAX_IN_FLIGHT_GPU_PRESENTS,
    });
    coordinatorPthreadPool.closeTxPorts();
    if (activeCoordinatorInputBridge) {
      try {
        activeCoordinatorInputBridge.stop();
      } catch {
        // Ignore teardown errors when re-initializing workers.
      }
      activeCoordinatorInputBridge = null;
    }
    if (activeCoordinatorModuleConfig) {
      try {
        if (typeof activeCoordinatorModuleConfig.destroy === 'function') {
          activeCoordinatorModuleConfig.destroy();
        } else if (typeof activeCoordinatorModuleConfig.resetAudioRingAttachment === 'function') {
          activeCoordinatorModuleConfig.resetAudioRingAttachment();
        }
      } catch {
        // Ignore audio ring teardown errors when re-initializing workers.
      }
    }
    activeCoordinatorRuntimeModule = null;
    activeCoordinatorModuleConfig = null;
    if (audioWorker) {
      teardownCoordinatorAudioWorker({ sendStop: true, terminate: true, destroyBridge: false });
    } else {
      audioWorkerReady = false;
    }
    if (activeCoordinatorPipelineDumpSyncer) {
      try {
        activeCoordinatorPipelineDumpSyncer.stop().catch(() => {});
      } catch {
        // Ignore pipeline dump syncer teardown errors when re-initializing workers.
      }
      activeCoordinatorPipelineDumpSyncer = null;
    }
    inputSab = msg.inputSab || null;
    configuredDumpPipelines =
      typeof msg.dumpPipelines === 'boolean' ? msg.dumpPipelines : undefined;
    configuredDumpDir = typeof msg.dumpDir === 'string' ? msg.dumpDir : undefined;
    configuredDumpEndpoint =
      typeof msg.dumpEndpoint === 'string' ? msg.dumpEndpoint : undefined;
    configuredDumpIntervalMs =
      typeof msg.dumpIntervalMs === 'number' ? msg.dumpIntervalMs : undefined;
    configuredDumpMaxConcurrentUploads =
      typeof msg.dumpMaxConcurrentUploads === 'number' ? msg.dumpMaxConcurrentUploads : undefined;
    configuredSideModuleUrls = Array.isArray(msg.sideModuleUrls) ? msg.sideModuleUrls : null;
    configuredDynamicLibraries = Array.isArray(msg.dynamicLibraries)
      ? msg.dynamicLibraries
      : null;
    configuredGuestManifest =
      msg.guestManifest !== undefined ? msg.guestManifest : msg.manifest !== undefined ? msg.manifest : null;
    configuredGuestManifestUrl =
      typeof msg.guestManifestUrl === 'string' && msg.guestManifestUrl.trim().length > 0
        ? msg.guestManifestUrl.trim()
        : null;
    configuredMaxConcurrentPthreadLoads = toOptionalNonNegativeInt(msg.maxConcurrentPthreadLoads);
    if (configuredMaxConcurrentPthreadLoads !== null) {
      coordinatorPthreadPool.setMaxConcurrentPthreadLoads(configuredMaxConcurrentPthreadLoads);
    }

    const requestedHttpfsMailboxLanes =
      msg.httpfsMailboxLanes !== undefined
        ? msg.httpfsMailboxLanes
        : DEFAULT_HTTPFS_MAILBOX_LANES;
    const allocatedMailboxPool = allocateHttpfsMailboxPool(
      requestedHttpfsMailboxLanes,
      httpfsMailboxApi,
    );
    httpfsMailboxSabs = allocatedMailboxPool.sabs;
    httpfsMailboxSab = allocatedMailboxPool.sab;
    nextHttpfsBroadcastLane = 0;
    ioWorkerMailboxAttached = false;
    ioWorkerMailboxLanes = 0;
    lastIoWorkerStats = null;
    lastIoStatsPollMs = 0;
    if (typeof coordinatorPthreadPool.resetHttpfsLaneCursor === 'function') {
      coordinatorPthreadPool.resetHttpfsLaneCursor();
    }
    if (
      httpfsMailboxSab &&
      httpfsMailboxApi &&
      typeof httpfsMailboxApi.HttpfsMailboxClient === 'function'
    ) {
      httpfsClient = new httpfsMailboxApi.HttpfsMailboxClient(httpfsMailboxSab);
      broadcastHttpfsSab(httpfsMailboxSab);
    } else {
      httpfsMailboxSabs = [];
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
      handleIoWorkerMessage(e && e.data, {
        onWorkerReady,
        postToMain,
        postProgress,
        requestStats: () => {
          if (ioWorker && typeof ioWorker.postMessage === 'function') {
            ioWorker.postMessage({ stats: true });
          }
        },
      });
    };
    gpuWorker.onerror = () => {
      resetGpuPresentBackpressure();
    };
    gpuWorker.onmessage = (e) => {
      const data = e && e.data;
      if (!data || typeof data !== 'object') return;
      if (data.type === PTHREAD_TX_MSG_RECYCLE_CMD_BUFFER) {
        handleRecycledCmdBuffer(data.cmdBuffer);
        return;
      }
      if (
        prewarmBridgeApi &&
        data.type === prewarmBridgeApi.PREWARM_MSG_RESULT &&
        activeCoordinatorModuleConfig &&
        typeof activeCoordinatorModuleConfig.handlePrewarmMessage === 'function'
      ) {
        activeCoordinatorModuleConfig.handlePrewarmMessage(data);
        return;
      }
      if (data.ready) {
        resetGpuPresentBackpressure();
        attachActiveCoordinatorGuestMemory();
        onWorkerReady('wgpu_worker', data.adapter);
        return;
      }
      ackGpuPresentFromWorkerMessage(data);
      if (data.type === 'gpu-present-ack') {
        return;
      }
      if (data.ringAttached) {
        attachActiveCoordinatorGuestMemory();
      }
      if (data.type === 'first-visible-frame' || data.type === 'visible-frame') {
        attachActiveCoordinatorGuestMemory();
        forwardGpuWorkerMessage(data, postToMain);
        return;
      }
      if (data.type === 'gpu-validation-error') {
        postProgress({
          error: `wgpu_worker: ${formatGpuWorkerError(data.error)}`,
          gpuValidationForwarded: true,
        });
        forwardGpuWorkerMessage(data, postToMain);
        return;
      }
      if (data.frame) {
        attachActiveCoordinatorGuestMemory();
        if (
          activeCoordinatorModuleConfig &&
          typeof activeCoordinatorModuleConfig.notifyGuestMilestone === 'function'
        ) {
          try {
            activeCoordinatorModuleConfig.notifyGuestMilestone('frame_presented');
          } catch {
            // Non-fatal audio ring milestone retry failure.
          }
        }
        report(`WebGPU frame #${data.frame} presented`);
        const nowPollMs = Date.now();
        if (
          ioWorker &&
          typeof ioWorker.postMessage === 'function' &&
          nowPollMs - lastIoStatsPollMs >= IO_STATS_POLL_INTERVAL_MS
        ) {
          lastIoStatsPollMs = nowPollMs;
          try {
            ioWorker.postMessage({ stats: true });
          } catch {
            // Non-fatal periodic io_worker stats poll error.
          }
          const rtDiag = readRuntimeDiagnostics(activeCoordinatorRuntimeModule);
          if (rtDiag && postToMain) {
            try {
              postToMain({
                type: 'loader-status',
                mailboxAttached: Boolean(ioWorkerMailboxAttached),
                ioWorkerStats: buildCoordinatorIoWorkerStats(),
                runtimeDiagnostics: rtDiag,
                timestamp: nowPollMs,
              });
            } catch {
              // Non-fatal runtime diagnostics forward error.
            }
          }
        }
      } else if (data.error) {
        postProgress({ error: `wgpu_worker: ${formatGpuWorkerError(data.error)}` });
      }
      forwardGpuWorkerMessage(data, postToMain);
    };
    audioWorker.onmessage = (e) => {
      const data = e && e.data;
      if (!data || typeof data !== 'object') return;
      if (data.ready) {
        const wasReady = audioWorkerReady;
        audioWorkerReady = true;
        coordinatorAudioBridge.onWorkerReady(audioWorker);
        if (
          activeCoordinatorModuleConfig &&
          typeof activeCoordinatorModuleConfig.onAudioWorkerReady === 'function'
        ) {
          activeCoordinatorModuleConfig.onAudioWorkerReady(audioWorker);
        } else if (
          activeCoordinatorModuleConfig &&
          typeof activeCoordinatorModuleConfig.notifyAudioWorkerReady === 'function'
        ) {
          activeCoordinatorModuleConfig.notifyAudioWorkerReady(audioWorker);
        } else if (activeCoordinatorRuntimeModule) {
          resetAttachedRuntimeAudioRing(audioWorker);
          attachRuntimeAudioRing(activeCoordinatorRuntimeModule, audioWorker);
        }
        if (!wasReady) {
          onWorkerReady('audio_worker');
        }
      }
    };

    const workerInits = createWorkerBootInitPayloadsWithTransfers(msg);
    if (workerInits.transferredArrayBufferCount > 0) {
      recordTransferredArrayBuffers(workerInits.transferredArrayBufferCount);
    }

    // Consume any preloaded Wasm binary / side-module wire buffers from `msg` and immediately
    // null out all raw binary properties on `msg` so `self.onmessage`'s closure retains 0 bytes.
    coordinatorSharedWasmModules.clear();
    pendingPreloadedWasmBinary =
      msg.wasmBinary !== undefined && msg.wasmBinary !== null ? msg.wasmBinary : null;
    if (msg.wasmBinary !== undefined && isWritableOwnProp(msg, 'wasmBinary')) {
      msg.wasmBinary = null;
    }
    if (Array.isArray(msg.sharedModules)) {
      stripWireBytesFromSharedModules(msg.sharedModules);
      for (let i = 0; i < msg.sharedModules.length; i += 1) {
        if (isCompiledWasmModule(msg.sharedModules[i])) {
          coordinatorSharedWasmModules.setModule(`shared_${i}.wasm`, msg.sharedModules[i]);
        }
      }
    }
    pendingPreloadedGuestBuffers = null;
    if (msg.guestModuleBytes && typeof msg.guestModuleBytes === 'object') {
      pendingPreloadedGuestBuffers = msg.guestModuleBytes;
      if (isWritableOwnProp(msg, 'guestModuleBytes')) {
        msg.guestModuleBytes = null;
      }
    } else if (msg.preloadedBuffers && typeof msg.preloadedBuffers === 'object') {
      pendingPreloadedGuestBuffers = msg.preloadedBuffers;
      if (isWritableOwnProp(msg, 'preloadedBuffers')) {
        msg.preloadedBuffers = null;
      }
    }

    if (workerInits.ioTransferList.length > 0) {
      ioWorker.postMessage(workerInits.ioInit, workerInits.ioTransferList);
    } else {
      ioWorker.postMessage(workerInits.ioInit);
    }
    if (httpfsMailboxSab) {
      const attachMailboxMsg = { attachMailbox: true, sab: httpfsMailboxSab };
      if (httpfsMailboxSabs.length > 0) {
        attachMailboxMsg.sabs = httpfsMailboxSabs.slice();
      }
      ioWorker.postMessage(attachMailboxMsg);
    }
    if (workerInits.gpuTransferList.length > 0) {
      gpuWorker.postMessage(workerInits.gpuInit, workerInits.gpuTransferList);
    } else {
      gpuWorker.postMessage(workerInits.gpuInit, [msg.canvas]);
    }
    coordinatorAudioBridge.initWorker(audioWorker, msg);

    // Clear all binary buffer references from `msg` and `workerInits` immediately after handoff.
    releaseWireBytesFromObject(msg, { recordTelemetry: false });
    workerInits.ioInit = null;
    workerInits.gpuInit = null;
    workerInits.ioTransferList = null;
    workerInits.gpuTransferList = null;
  };
}

const exportedApi = {
  EMSCRIPTEN_PTHREAD_WORKER_NAME,
  PROGRESS_CHANNEL_NAME,
  PTHREAD_WGPU_CHANNEL_NAME,
  PTHREAD_HTTPFS_SAB_CHANNEL_NAME,
  PTHREAD_TX_MSG_WGPU_SUBMIT,
  PTHREAD_TX_MSG_PROGRESS,
  PTHREAD_TX_MSG_RECYCLE_CMD_BUFFER,
  PTHREAD_TX_MSG_FAULT_RECORD,
  FAULT_RECORD_MESSAGE_TYPE,
  FAULT_RECORD_PREFIX,
  FLUSH_PIPELINE_DUMPS_MESSAGE_TYPE,
  FLUSH_PIPELINE_DUMP_MESSAGE_TYPE,
  PIPELINE_DUMPS_FLUSHED_MESSAGE_TYPE,
  isFaultRecordLine,
  parseFaultRecordLine,
  getLastFaultRecord,
  resetLastFaultRecord,
  flushPipelineDump: flushActiveCoordinatorPipelineDumps,
  flushPipelineDumps: flushActiveCoordinatorPipelineDumps,
  WGPU_CMD_POOL_MIN_BYTES,
  WGPU_CMD_POOL_MAX_BYTES,
  WGPU_CMD_POOL_MAX_BUFFERS,
  createWarmCmdBufferPool,
  PTHREAD_CMD_LOAD,
  PTHREAD_CMD_LOADED,
  PTHREAD_CMD_RUN,
  PTHREAD_CMD_OBJECT_TRANSFER,
  DEFAULT_MAX_CONCURRENT_PTHREAD_LOADS,
  DEFAULT_HTTPFS_MAILBOX_LANES,
  V8_SIDE_MODULE_CAP_BYTES,
  V8_SIDE_MODULE_WARN_RATIO,
  FAST_BOOT_EXPORT_NAMES,
  HOST_ABI_IMPORT_NAMES,
  GPU_OPCODE_PRESENT,
  GPU_PACKET_HEADER_BYTES,
  DEFAULT_MAX_IN_FLIGHT_GPU_PRESENTS,
  INPUT_STATE_BYTES,
  INPUT_STATE_WORDS,
  DEFAULT_INPUT_BRIDGE_POLL_MS,
  DEFAULT_INPUT_BRIDGE_WAIT_ASYNC_TIMEOUT_MS,
  DEFAULT_INPUT_BRIDGE_WAIT_ASYNC_FALLBACK_POLL_MS,
  AUDIO_RING_STRUCT_BYTES_WASM64,
  AUDIO_RING_STRUCT_BYTES_WASM32,
  AUDIO_RING_CHANNELS,
  AUDIO_RING_MIN_CAPACITY_SAMPLES,
  AUDIO_RING_MAX_CAPACITY_SAMPLES,
  AUDIO_RING_MIN_SAMPLE_RATE,
  AUDIO_RING_MAX_SAMPLE_RATE,
  DEFAULT_AUDIO_RING_RETRY_INTERVAL_MS,
  DEFAULT_AUDIO_RING_MAX_RETRIES,
  AUDIO_RING_REASON,
  DEFAULT_MEMFS_PIPELINE_DUMP_DIR,
  DEFAULT_MEMFS_PIPELINE_DUMP_ENDPOINT,
  DEFAULT_MEMFS_PIPELINE_DUMP_INTERVAL_MS,
  DEFAULT_MEMFS_PIPELINE_DUMP_MAX_CONCURRENT_UPLOADS,
  BOOT_INIT_FORWARD_KEYS,
  isEmscriptenPthreadWorkerName,
  isEmscriptenPthreadWorker,
  normalizeHttpfsMailboxLanes,
  allocateHttpfsMailboxPool,
  createPatchedPthreadEnv,
  buildSharedWasmModuleDylinkStub,
  extractSharedWasmStubKey,
  installSharedWasmModuleShim,
  installPthreadHostImportShim,
  bootstrapEmscriptenPthreadWorker,
  createPthreadPoolCoordinator,
  installPthreadWorkerProxy,
  getPthreadTelemetry,
  getLoaderStatus,
  sanitizeIoWorkerStats,
  createRuntimePrewarmBridge,
  loadPrewarmWarmList,
  readPrewarmProgress,
  savePrewarmWarmList,
  readRuntimeDiagnostics,
  handleIoWorkerMessage,
  getWasmMemoryTelemetry,
  resetWasmMemoryTelemetry,
  collectTransferableArrayBuffers,
  releaseWireBytesFromObject,
  stripWireBytesFromSharedModules,
  compileWasmModuleStreaming,
  createSharedWasmModuleCache,
  parseGuestManifest,
  extractSideModuleEntries,
  evaluateV8SideModuleBudget,
  inspectRuntimeExports,
  auditIndirectTableAndGotFunc,
  ensureIndirectFunctionTableCapacity,
  loadGuestManifest,
  inspectGpuPacketBatchForPresent,
  normalizeBootHttpfsPath,
  collectBootBridgeInitFields,
  createWorkerBootInitPayloads,
  createWorkerBootInitPayloadsWithTransfers,
  forwardGpuWorkerMessage,
  sanitizeFrameStats,
  sanitizeGpuMemoryStats,
  sanitizeRenderStateStats,
  createHostImports,
  sanitizeAudioRingReason,
  probePlaygta6AudioRingDescriptor,
  readPlaygta6AudioRingDescriptor,
  attachRuntimeAudioRing,
  resetAttachedRuntimeAudioRing,
  scheduleAudioRingRetry,
  normalizeAudioContextStateValue,
  extractAudioRingSab,
  isAudioContextStateMessage,
  isAudioRingSabMessage,
  createAudioWorkerInitPayload,
  createAudioWorkerBridge,
  startInputSabBridge,
  resolveDumpPipelinesEnabled,
  sanitizeDumpRelPath,
  createMemfsPipelineDumpSyncer,
  resolveRuntimeScriptUrl,
  resolveSideModuleUrl,
  sanitizeSideModuleUrls,
  buildRuntimeModuleConfig,
  createRuntimeModuleOptions,
  encodeTestFramePacket,
  postBroadcastOnce,
  toTransferableArrayBuffer,
  ptrToNumber,
  RELAUNCH_SELF_EXIT_CODE,
  GUEST_RELAUNCH_MESSAGE_TYPE,
  guestRelaunchPayload,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Loader = exportedApi;
  if (typeof globalThis.__pg6FlushDumps !== 'function') {
    try {
      globalThis.__pg6FlushDumps = flushActiveCoordinatorPipelineDumps;
    } catch {
      // Ignore non-writable globalThis property.
    }
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
})();
