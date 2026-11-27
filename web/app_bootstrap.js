// Unified browser application bootstrapper for playgta6web.
// Instantiates and connects CrashReporter, PerfMonitor, InputProfileManager,
// InputBridge, BrowserStorageManager, AudioEngine, RuntimeSessionController,
// SmokeDemoRunner (WebGPU G5 fallback), and Wasm64 loader worker lifecycle.
// Zero external dependencies; ES module compatible with Browser and Node.js.

import {
  DEFAULT_STATUS_URL,
  RuntimeSessionController,
  SESSION_MODE,
  SESSION_STATE,
} from './runtime/session_controller.js';
import { SmokeDemoRunner } from './demo/run_smoke_demo.js';
import {
  FIRST_VISIBLE_FRAME_ANCHOR_PRESETS,
  INPUT_SCRIPT_PRESETS,
  InputBridge,
} from './input/input_bridge.js';
import { InputProfileManager } from './input/input_profiles.js';
import { AudioEngine } from './audio/audio_engine.js';
import { BrowserStorageManager } from './storage/storage_manager.js';
import {
  DEFAULT_FRAME_BUDGET_MS,
  DEFAULT_TARGET_FPS,
  FRAME_BOTTLENECKS,
  PERF_PHASES,
  PerfMonitor,
} from './telemetry/perf_monitor.js';
import { CrashReporter, sanitizeString } from './telemetry/crash_report.js';
import { BootTracer, FAST_BOOT_MILESTONES, sanitizeGpuAdapter } from './telemetry/boot_tracer.js';
import './workers/perf_profile.js';

export const GUEST_RELAUNCH_MESSAGE_TYPE = 'guest-relaunch';
export const MAX_GUEST_RELAUNCHES = 3;
const GUEST_RELAUNCH_COUNT_KEY = 'pg6.guestRelaunchCount';

/** Clears the self-relaunch counter (called once a boot presents its first frame). */
export function resetGuestRelaunchCount(windowObj) {
  try {
    if (windowObj && windowObj.sessionStorage) {
      windowObj.sessionStorage.removeItem(GUEST_RELAUNCH_COUNT_KEY);
    }
  } catch {
    // Ignore storage failures.
  }
}

/**
 * Counts a guest self-relaunch in sessionStorage and says whether another one is allowed
 * (at most MAX_GUEST_RELAUNCHES per tab session, so a relaunch loop cannot spin forever).
 * Without storage the relaunch is allowed.
 */
export function shouldRelaunchGuest(windowObj, max = MAX_GUEST_RELAUNCHES) {
  let storage = null;
  try {
    storage = windowObj && windowObj.sessionStorage ? windowObj.sessionStorage : null;
  } catch {
    storage = null;
  }
  if (!storage) {
    return true;
  }
  try {
    const count = Number.parseInt(storage.getItem(GUEST_RELAUNCH_COUNT_KEY) || '0', 10) || 0;
    if (count >= max) {
      return false;
    }
    storage.setItem(GUEST_RELAUNCH_COUNT_KEY, String(count + 1));
  } catch {
    // Storage failures never block the relaunch.
  }
  return true;
}

export const DEFAULT_LOADER_URL = new URL('./loader.js', import.meta.url).href;
export const DEFAULT_SHADER_PACK_URL = '/shaders/fallback.wgsl.pack';
export const DEFAULT_DUMP_ENDPOINT = '/__pg6_dump/';
export const DEFAULT_VIEWPORT_WIDTH = 640;
export const DEFAULT_VIEWPORT_HEIGHT = 360;
export const DEFAULT_CANVAS_FORMAT = 'bgra8unorm';
export const DEFAULT_MAX_VALIDATION_ERRORS = 32;
export const DEFAULT_SPLASH_FADE_OUT_MS = 220;
export const DEFAULT_STREAM_SHADERS_THRESHOLD_MB = 64;
export const DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES =
  DEFAULT_STREAM_SHADERS_THRESHOLD_MB * 1024 * 1024;
export const MAX_STREAM_THRESHOLD_MB = 4096;
export const MAX_STREAM_THRESHOLD_BYTES = MAX_STREAM_THRESHOLD_MB * 1024 * 1024;
export const MIN_STREAM_THRESHOLD_BYTES = 1;

const BYTES_PER_MIB = 1024 * 1024;
const SAFE_INPUT_SCRIPT_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;
const POSITIVE_DECIMAL_RE = /^(?:\d+(?:\.\d+)?|\.\d+)$/;

const BUILTIN_INPUT_SCRIPT_PRESETS = Object.freeze([
  'start-screen',
  'menu-flow',
  'calibration-skip',
  'intro-watch',
]);

const ALLOWED_INPUT_SCRIPT_PRESETS = (() => {
  const map = new Map();
  if (INPUT_SCRIPT_PRESETS && typeof INPUT_SCRIPT_PRESETS === 'object') {
    const keys = Object.keys(INPUT_SCRIPT_PRESETS);
    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      const lower = key.toLowerCase();
      if (SAFE_INPUT_SCRIPT_RE.test(lower)) {
        map.set(lower, key);
      }
    }
  }
  for (let i = 0; i < BUILTIN_INPUT_SCRIPT_PRESETS.length; i += 1) {
    const preset = BUILTIN_INPUT_SCRIPT_PRESETS[i];
    if (!map.has(preset)) {
      map.set(preset, preset);
    }
  }
  return map;
})();

const FIRST_VISIBLE_FRAME_ANCHOR_PRESET_SET = new Set(
  Array.isArray(FIRST_VISIBLE_FRAME_ANCHOR_PRESETS) ? FIRST_VISIBLE_FRAME_ANCHOR_PRESETS : []
);

function resolveInputScriptStartOptions(preset, callerOptions) {
  const hasCallerOptions = Boolean(callerOptions && typeof callerOptions === 'object');
  if (!FIRST_VISIBLE_FRAME_ANCHOR_PRESET_SET.has(preset)) {
    return hasCallerOptions ? callerOptions : null;
  }
  if (
    hasCallerOptions &&
    (callerOptions.startOn !== undefined || callerOptions.waitForFirstVisibleFrame !== undefined)
  ) {
    return callerOptions;
  }
  return hasCallerOptions
    ? { ...callerOptions, startOn: 'FIRST_VISIBLE_FRAME' }
    : { startOn: 'FIRST_VISIBLE_FRAME' };
}

function buildPreloaderStreamOptions(streamConfig, explicitDisabled) {
  if (streamConfig && streamConfig.streamShaders === true) {
    return {
      streamShaders: true,
      ...(isPositiveFinite(streamConfig.streamThresholdBytes)
        ? { streamThresholdBytes: streamConfig.streamThresholdBytes }
        : {}),
    };
  }
  if (explicitDisabled === true) {
    return { streamShaders: false };
  }
  return {};
}

const MAX_VALIDATION_ERROR_MESSAGE_LENGTH = 512;
const MAX_VALIDATION_ERROR_CODE_LENGTH = 64;
const MAX_TELEMETRY_URL_LENGTH = 512;
const MAX_SIDE_MODULE_URLS = 256;
const ABSOLUTE_URL_RE = /^[a-zA-Z][a-zA-Z0-9+.-]*:/;
const WIN_DRIVE_PREFIX_RE = /^[a-zA-Z]:[\\/]/;
const POSIX_HOST_ROOT_RE =
  /^\/(?:home|Users|root|tmp|var|etc|opt|usr|mnt|media|Volumes|private|srv|proc|sys|dev|run|Library|Applications)(?:\/|$)/i;

const INDIRECT_DISPATCH_ALIASES = Object.freeze({
  shard: 'shard',
  sharded: 'shard',
  'shard-dispatch': 'shard',
  shard_dispatch: 'shard',
  'split-shard': 'shard',
  split_shard: 'shard',
  table: 'table',
  'indirect-table': 'table',
  indirect_table: 'table',
  'wasm-table': 'table',
  wasm_table: 'table',
  monolithic: 'table',
});

const AUDIO_OUTPUT_STATE_ALIASES = Object.freeze({
  running: 'running',
  active: 'running',
  playing: 'running',
  suspended: 'suspended',
  paused: 'suspended',
  closed: 'closed',
  stopped: 'closed',
  interrupted: 'interrupted',
  idle: 'idle',
  ready: 'idle',
  uninitialized: 'idle',
  degraded: 'degraded',
  fallback: 'degraded',
  error: 'error',
  failed: 'error',
  unknown: 'unknown',
});

const LOADER_TELEMETRY_MESSAGE_TYPES = new Set([
  'pthread-stats',
  'pthread-status',
  'loader-stats',
  'loader-status',
  'loader-telemetry',
  'runtime-stats',
  'runtime-telemetry',
  'manifest-stats',
  'shard-dispatch',
  'boot-telemetry',
]);

const AUDIO_RING_MESSAGE_TYPES = new Set([
  'audio-ring',
  'audio-ring-status',
  'audio-ring-attached',
  'audio-ring-telemetry',
  'audio-ring-attempt',
]);

const SESSION_PHASE_MESSAGE_TYPES = new Set(['session-phase', 'phase-transition']);

const MEMORY_PRESSURE_MESSAGE_TYPES = new Set([
  'memory-pressure',
  'soft-memory-pressure',
  'vram-pressure',
  'memory-reclaim',
]);

const AUDIO_RING_ATTACHED_LOG_RE =
  /\b(?:audio_ring_attached|audioRingAttached|guest_ring_attached|guestRingAttached|pcm_ring_attached|pcmRingAttached)\s*[=:]\s*(true|false|yes|no|1|0)\b/i;
const AUDIO_RING_RETRIES_LOG_RE = /\bretries\s*[=:]\s*(\d+)\b/i;
const AUDIO_RING_ATTEMPTS_LOG_RE =
  /\b(?:audio_ring_attach_attempts|audioRingAttachAttempts|attach_attempts|attachAttempts|attempts)\s*[=:]\s*(\d+)\b/i;

const DEFAULT_AUDIO_RING_RETRY_INTERVAL_MS = 250;
const DEFAULT_AUDIO_RING_MAX_RETRIES = 120;

const FAST_BOOT_MILESTONE_TO_SESSION_PHASE = Object.freeze({
  START_SCREEN: 'start_screen',
  INTRO_VIDEO: 'intro_video',
  MAIN_MENU: 'main_menu',
  IN_GAME: 'in_game',
});

const CANONICAL_SESSION_PHASES = Object.freeze([
  'boot',
  'start_screen',
  'intro_video',
  'main_menu',
  'in_game',
]);
const CANONICAL_SESSION_PHASE_SET = new Set(CANONICAL_SESSION_PHASES);

const PERF_TELEMETRY_MESSAGE_TYPES = new Set([
  'perf-stats',
  'perf-telemetry',
  'frame-stats',
  'frame-telemetry',
]);

const IO_TELEMETRY_MESSAGE_TYPES = new Set([
  'io-stats',
  'io-telemetry',
  'httpfs-stats',
  'mailbox-stats',
]);

const ALLOWED_FRAME_BOTTLENECKS = new Set(Object.values(FRAME_BOTTLENECKS));
const FAST_BOOT_PRESSURE_LEVEL_SET = new Set(['low', 'medium', 'high', 'critical']);
const FAST_BOOT_MILESTONE_SET = new Set(FAST_BOOT_MILESTONES);

const FAST_BOOT_TRACER_METHODS = Object.freeze({
  START_SCREEN: 'recordStartScreen',
  INTRO_VIDEO: 'recordIntroVideo',
  MAIN_MENU: 'recordMainMenu',
  IN_GAME: 'recordInGame',
});

const FAST_BOOT_MESSAGE_TYPE_TO_MILESTONE = Object.freeze({
  'start-screen': 'START_SCREEN',
  start_screen: 'START_SCREEN',
  'intro-video': 'INTRO_VIDEO',
  intro_video: 'INTRO_VIDEO',
  'main-menu': 'MAIN_MENU',
  main_menu: 'MAIN_MENU',
  'in-game': 'IN_GAME',
  in_game: 'IN_GAME',
});

const FAST_BOOT_MILESTONE_LOG_PATTERNS = Object.freeze([
  [
    'START_SCREEN',
    /\bSTART_SCREEN\b|\bstart[\s_-]*screen\b|\bpress[\s_-]*(?:any[\s_-]*button|start)\b/i,
  ],
  [
    'INTRO_VIDEO',
    /\bINTRO_VIDEO\b|\bintro[\s_-]*(?:video|movie|cinematic|bink)\b|\bopening[\s_-]*(?:video|cinematic)\b/i,
  ],
  ['MAIN_MENU', /\bMAIN_MENU\b|\bmain[\s_-]*menu\b|\btitle[\s_-]*menu\b/i],
  [
    'IN_GAME',
    /\bIN_GAME\b|\bin[\s_-]*game\b|\bgameplay[\s_-]*(?:started|active|ready|entered)\b/i,
  ],
]);

const PANEL_PERF_TELEMETRY_KEYS = Object.freeze([
  'fps30ComplianceRate',
  'frameJitterMs',
  'jsHeapUsedMb',
  'vramBudgetUtilization',
  'bindGroupHitRate',
  'mailboxLanes',
]);

function defaultNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function defaultEpochNow() {
  return Date.now();
}

function isPositiveFinite(value) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

function isEventTargetLike(obj) {
  return Boolean(
    obj &&
      typeof obj === 'object' &&
      typeof obj.addEventListener === 'function' &&
      typeof obj.removeEventListener === 'function'
  );
}

export function detectWasm64(wasmApi) {
  if (!wasmApi || typeof wasmApi.Memory !== 'function') {
    return false;
  }
  // Probe spec-standard `{ address: 'i64' }` (Chrome 133+, Firefox 134+, Node 22+)
  // first, then fall back to legacy proposal `{ index: 'i64' }`. Passing BigInt
  // `initial`/`maximum` ensures non-Wasm64 engines (or mocks) that silently ignore
  // unknown descriptor keys still throw a TypeError on BigInt-to-Number coercion.
  for (const key of ['address', 'index']) {
    try {
      const mem = new wasmApi.Memory({ initial: 1n, maximum: 2n, [key]: 'i64' });
      if (!mem || typeof mem !== 'object') {
        continue;
      }
      if (typeof mem.type === 'function') {
        try {
          const info = mem.type();
          if (info && typeof info === 'object') {
            const addr = info.address ?? info.index;
            if (addr !== undefined && addr !== 'i64') {
              continue;
            }
          }
        } catch {
          // Ignore optional type-reflection errors.
        }
      }
      return true;
    } catch {
      // Try next descriptor key or report unsupported below.
    }
  }
  return false;
}

function resolveDefaultWindow(options) {
  if (options.windowObj !== undefined) {
    return options.windowObj;
  }
  return typeof window !== 'undefined' ? window : null;
}

function resolveDefaultDocument(options, windowObj) {
  if (options.documentObj !== undefined) {
    return options.documentObj;
  }
  if (windowObj && windowObj.document) {
    return windowObj.document;
  }
  return typeof document !== 'undefined' ? document : null;
}

function resolveDefaultNavigator(options, windowObj) {
  if (options.navigatorObj !== undefined) {
    return options.navigatorObj;
  }
  if (windowObj && windowObj.navigator) {
    return windowObj.navigator;
  }
  return typeof navigator !== 'undefined' ? navigator : null;
}

function resolveDefaultStorage(options, windowObj, hasExplicitWindow) {
  if (options.inputProfileOptions && options.inputProfileOptions.storage !== undefined) {
    return options.inputProfileOptions.storage;
  }
  if (options.storage !== undefined) {
    return options.storage;
  }
  if (hasExplicitWindow) {
    try {
      const ls = windowObj && windowObj.localStorage;
      if (ls && typeof ls.getItem === 'function' && typeof ls.setItem === 'function') {
        return ls;
      }
    } catch {
      return null;
    }
    return null;
  }
  return undefined;
}

function resolveDefaultFetch(options, windowObj) {
  if (options.fetchImpl !== undefined) {
    return options.fetchImpl;
  }
  if (windowObj && typeof windowObj.fetch === 'function') {
    return windowObj.fetch.bind(windowObj);
  }
  if (typeof globalThis !== 'undefined' && typeof globalThis.fetch === 'function') {
    return globalThis.fetch.bind(globalThis);
  }
  return null;
}

function resolveDefaultDataBase(options, windowObj) {
  if (typeof options.dataBase === 'string' && options.dataBase.length > 0) {
    return options.dataBase;
  }
  const origin = windowObj && windowObj.location && windowObj.location.origin;
  if (typeof origin === 'string' && origin.length > 0 && origin !== 'null') {
    return `${origin}/data/`;
  }
  return '/data/';
}

function resolveDefaultBootBase(options, windowObj) {
  if (typeof options.bootBase === 'string' && options.bootBase.length > 0) {
    return options.bootBase;
  }
  const origin = windowObj && windowObj.location && windowObj.location.origin;
  if (typeof origin === 'string' && origin.length > 0 && origin !== 'null') {
    return `${origin}/boot/`;
  }
  return '/boot/';
}

export function resolveDefaultGameBase(options = {}, windowObj = null) {
  if (options && typeof options.gameBase === 'string' && options.gameBase.length > 0) {
    return options.gameBase;
  }
  const origin = windowObj && windowObj.location && windowObj.location.origin;
  if (typeof origin === 'string' && origin.length > 0 && origin !== 'null') {
    return `${origin}/game/`;
  }
  return '/game/';
}

export function sanitizeInputScriptPreset(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const trimmed = raw.trim();
  if (!trimmed || trimmed.length > 64 || /[\x00-\x1F\x7F]/.test(trimmed)) {
    return null;
  }
  const lower = trimmed.toLowerCase();
  if (!SAFE_INPUT_SCRIPT_RE.test(lower)) {
    return null;
  }
  return ALLOWED_INPUT_SCRIPT_PRESETS.get(lower) ?? null;
}

function parseStreamThresholdMb(raw) {
  let parsed = null;
  if (typeof raw === 'number') {
    parsed = raw;
  } else if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!trimmed || trimmed.length > 32 || !POSITIVE_DECIMAL_RE.test(trimmed)) {
      return null;
    }
    parsed = Number(trimmed);
  } else {
    return null;
  }
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  const mb = Math.min(parsed, MAX_STREAM_THRESHOLD_MB);
  const bytes = Math.max(
    MIN_STREAM_THRESHOLD_BYTES,
    Math.min(MAX_STREAM_THRESHOLD_BYTES, Math.floor(mb * BYTES_PER_MIB))
  );
  return { mb, bytes };
}

function parsePositiveStreamBytes(raw) {
  if (typeof raw !== 'number' || !Number.isFinite(raw) || raw <= 0) {
    return null;
  }
  return Math.max(
    MIN_STREAM_THRESHOLD_BYTES,
    Math.min(MAX_STREAM_THRESHOLD_BYTES, Math.floor(raw))
  );
}

function parseBooleanQueryFlag(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim().toLowerCase();
  if (normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on') {
    return true;
  }
  if (normalized === '0' || normalized === 'false' || normalized === 'no' || normalized === 'off') {
    return false;
  }
  return null;
}

function getQueryParamInsensitive(params, name) {
  if (params.has(name)) {
    return params.get(name);
  }
  const targetLower = name.toLowerCase();
  for (const [key, value] of params.entries()) {
    if (key.toLowerCase() === targetLower) {
      return value;
    }
  }
  return null;
}

function getLastQueryParamInsensitive(params, name) {
  const targetLower = name.toLowerCase();
  let lastValue = null;
  for (const [key, value] of params.entries()) {
    if (key.toLowerCase() === targetLower) {
      lastValue = value;
    }
  }
  return lastValue;
}

const GPU_ADAPTER_LOG_RE = /(?:\[wgpu_worker\]\s*)?WebGPU\s+adapter\s*:\s*(.+)$/i;
const WGPU_WORKER_READY_ADAPTER_RE =
  /^(?:\[loader\]\s*)?wgpu_worker\s+ready\s*\(([^)]+)\)\s*$/i;
const WEBGPU_FRAME_PRESENTED_LABEL_RE = /^WebGPU\s+frame\s+#(\d+)\s+presented$/i;
const FRAME_PRESENTED_LOG_MILESTONES = Object.freeze(new Set([1, 10, 60]));
const FRAME_PRESENTED_LOG_INTERVAL_FRAMES = 300;
const FRAME_PRESENTED_LOG_INTERVAL_MS = 10000;
const FRAME_PRESENTED_HUD_THROTTLE_MS = 100;
const HIDDEN_HUD_FRAME_METRICS_THROTTLE_MS = 500;
const JS_HEAP_SAMPLE_THROTTLE_MS = 1000;
const MAX_SYNTHETIC_FRAME_DELTA_STEPS = 2048;
const FIRST_VISIBLE_FRAME_LOG_RE =
  /\bFIRST_VISIBLE_FRAME\b|\bfirst[\s_-]*(?:visible|non[\s_-]*black)[\s_-]*frame\b/i;
const VISIBLE_FRAME_LOG_FAILURE_RE =
  /\b(?:fail(?:ed|s|ure)?|error|timeout|timed[\s_-]*out|abort(?:ed)?)\b/i;

/**
 * Parses a loader per-frame presentation log label (e.g. `"WebGPU frame #42 presented"`),
 * returning the positive integer frame number or `null` for non-matching strings.
 *
 * @param {unknown} label
 * @returns {number|null}
 */
export function parseWebGpuFramePresentedLabel(label) {
  if (typeof label !== 'string' || label.length === 0) {
    return null;
  }
  const m = WEBGPU_FRAME_PRESENTED_LABEL_RE.exec(label.trim());
  if (!m) {
    return null;
  }
  const parsed = Number.parseInt(m[1], 10);
  return Number.isSafeInteger(parsed) && parsed >= 1 ? parsed : null;
}

/**
 * Determines whether a `"WebGPU frame #N presented"` log should be recorded into
 * `CrashReporter.recentLogs` / HUD log DOM, preserving frame #1 and milestone frames
 * (`#10`, `#60`) while throttling steady-state logging to at most once every 300 frames
 * or 10 seconds so the 200-entry diagnostic ring is never flooded.
 *
 * @param {number} frameNum
 * @param {{ lastLoggedFrame?: number, lastLoggedAtMs?: number, nowMs?: number }} [state]
 * @returns {boolean}
 */
export function shouldLogWebGpuPresentedFrame(frameNum, state = {}) {
  if (!Number.isSafeInteger(frameNum) || frameNum < 1) {
    return false;
  }
  if (FRAME_PRESENTED_LOG_MILESTONES.has(frameNum)) {
    return true;
  }
  if (frameNum <= 60) {
    return false;
  }
  const lastLoggedFrame =
    state && Number.isSafeInteger(state.lastLoggedFrame) && state.lastLoggedFrame >= 0
      ? state.lastLoggedFrame
      : 0;
  if (frameNum - lastLoggedFrame >= FRAME_PRESENTED_LOG_INTERVAL_FRAMES) {
    return true;
  }
  const lastLoggedAtMs =
    state && typeof state.lastLoggedAtMs === 'number' && Number.isFinite(state.lastLoggedAtMs)
      ? state.lastLoggedAtMs
      : -Infinity;
  const nowMs =
    state && typeof state.nowMs === 'number' && Number.isFinite(state.nowMs) ? state.nowMs : null;
  if (nowMs !== null && nowMs - lastLoggedAtMs >= FRAME_PRESENTED_LOG_INTERVAL_MS) {
    return true;
  }
  return false;
}

function stripHtmlAndUrlsFromAdapterText(text) {
  return text
    .replace(/<script\b[\s\S]*?<\/script>/gi, ' ')
    .replace(/<[^>]*>/g, ' ')
    .replace(/\b(?:https?|file|data|javascript|ftp|wss?):\/\/[^\s"'`<>)\]},;]*/gi, ' ')
    .replace(/\b(?:javascript|data):[^\s"'`<>)\]},;]*/gi, ' ');
}

function extractSanitizedGpuAdapter(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (typeof raw === 'string') {
    return sanitizeGpuAdapter(stripHtmlAndUrlsFromAdapterText(raw));
  }
  if (typeof raw === 'object' && !Array.isArray(raw)) {
    const src =
      raw.info && typeof raw.info === 'object' && !Array.isArray(raw.info) ? raw.info : raw;
    const direct = sanitizeGpuAdapter(src);
    if (direct) {
      return direct;
    }
    const keys = [
      'gpuAdapter',
      'gpu_adapter',
      'adapterLabel',
      'adapter_label',
      'label',
      'description',
      'device',
      'name',
    ];
    for (const source of [src, raw]) {
      for (const key of keys) {
        if (
          Object.prototype.hasOwnProperty.call(source, key) &&
          typeof source[key] === 'string' &&
          source[key].trim().length > 0
        ) {
          const sanitized = sanitizeGpuAdapter(stripHtmlAndUrlsFromAdapterText(source[key]));
          if (sanitized) {
            return sanitized;
          }
        }
      }
    }
  }
  return null;
}

/**
 * Extracts and sanitizes a WebGPU adapter label from a worker/runtime log line
 * (e.g. `[wgpu_worker] WebGPU adapter: NVIDIA / Turing`), or returns `null`.
 *
 * @param {unknown} line
 * @returns {string|null}
 */
export function extractGpuAdapterFromLogLine(line) {
  if (typeof line !== 'string' || line.length === 0) {
    return null;
  }
  const m = GPU_ADAPTER_LOG_RE.exec(line);
  if (m) {
    return extractSanitizedGpuAdapter(m[1]);
  }
  const readyMatch = WGPU_WORKER_READY_ADAPTER_RE.exec(line.trim());
  if (readyMatch) {
    const candidate = readyMatch[1].trim();
    const lower = candidate.toLowerCase();
    if (!candidate || lower === 'webgpu' || lower === 'unknown' || lower === 'gpu') {
      return null;
    }
    return extractSanitizedGpuAdapter(candidate);
  }
  return null;
}

/**
 * Parses the optional `?splashHold=` / `?holdUntilVisible=` query parameter
 * (`1`/`true`/`on`/`yes` -> `true`, `0`/`false`/`no`/`off` -> `false`;
 * case-insensitive; last occurrence across either alias wins).
 * Returns `true`, `false`, or `null` when absent or invalid.
 *
 * @param {string|URLSearchParams|URL|Location|object|null|undefined} search
 * @returns {boolean|null}
 */
export function parseSplashHoldQuery(search) {
  const params = toUrlSearchParams(search);
  if (!params) {
    return null;
  }
  let raw = null;
  for (const [key, value] of params.entries()) {
    const lower = String(key).trim().toLowerCase();
    if (lower === 'splashhold' || lower === 'holduntilvisible') {
      raw = value;
    }
  }
  if (raw === null) {
    return null;
  }
  return parseBooleanQueryFlag(raw);
}

function toUrlSearchParams(search) {
  if (search === null || search === undefined || search === '') {
    return null;
  }
  if (typeof URLSearchParams === 'function' && search instanceof URLSearchParams) {
    return search;
  }
  let raw = null;
  if (typeof search === 'string') {
    raw = search;
  } else if (typeof search === 'object') {
    if (typeof search.search === 'string') {
      raw = search.search;
    } else if (typeof search.href === 'string') {
      raw = search.href;
    }
  }
  if (typeof raw !== 'string' || raw.length === 0) {
    return null;
  }
  const withoutHash = raw.split('#', 1)[0];
  if (!withoutHash) {
    return null;
  }
  let query = '';
  if (ABSOLUTE_URL_RE.test(withoutHash)) {
    const qIdx = withoutHash.indexOf('?');
    query = qIdx >= 0 ? withoutHash.slice(qIdx + 1) : '';
  } else if (withoutHash.startsWith('?')) {
    query = withoutHash.slice(1);
  } else if (withoutHash.includes('?')) {
    query = withoutHash.slice(withoutHash.indexOf('?') + 1);
  } else {
    query = withoutHash;
  }
  if (!query) {
    return null;
  }
  try {
    return new URLSearchParams(query);
  } catch {
    return null;
  }
}

/**
 * Pure URL query-parameter parser and sanitizer for bootstrap options:
 * - `?inputScript=<preset>` (case-insensitive, strictly allowlisted to known preset IDs:
 *   `start-screen`, `menu-flow`, `calibration-skip`, `intro-watch`, and valid presets in `INPUT_SCRIPT_PRESETS`)
 * - `?streamShaders=1|true` (enables ranged shader streaming with default 64 MiB threshold)
 * - `?streamThresholdMb=<N>` (finite positive number clamped to safe bounds, converted to bytes)
 * - `?dumpPipelines=1|true` (enables opt-in MEMFS `/pg6_dump` pipeline dump sync in `loader.js`, default `false`)
 * - `?allowDumpWrites=1|true` (enables opt-in `/pg6_dump/*` write forwarding in `io_worker.js` to `/__pg6_dump/`, default `false`)
 *
 * @param {string|URLSearchParams|URL|Location|object|null|undefined} search
 * @returns {Readonly<{
 *   inputScript: string | null,
 *   streamShaders: boolean,
 *   streamThresholdMb: number | null,
 *   streamThresholdBytes: number | null,
 *   dumpPipelines: boolean,
 *   allowDumpWrites: boolean,
 *   dumpEndpoint: string | null,
 * }>}
 */
export function parseBootUrlOptions(search) {
  const params = toUrlSearchParams(search);
  if (!params) {
    return Object.freeze({
      inputScript: null,
      streamShaders: false,
      streamThresholdMb: null,
      streamThresholdBytes: null,
      dumpPipelines: false,
      allowDumpWrites: false,
      dumpEndpoint: null,
    });
  }

  const inputScript = sanitizeInputScriptPreset(getQueryParamInsensitive(params, 'inputScript'));
  const streamFlag = parseBooleanQueryFlag(getQueryParamInsensitive(params, 'streamShaders'));
  const parsedThreshold = parseStreamThresholdMb(
    getQueryParamInsensitive(params, 'streamThresholdMb')
  );
  const dumpPipelinesFlag = parseBooleanQueryFlag(
    getQueryParamInsensitive(params, 'dumpPipelines')
  );
  const allowDumpWritesFlag = parseBooleanQueryFlag(
    getQueryParamInsensitive(params, 'allowDumpWrites')
  );

  let streamShaders = false;
  let streamThresholdMb = null;
  let streamThresholdBytes = null;

  if (streamFlag === false) {
    streamShaders = false;
  } else if (parsedThreshold !== null) {
    streamShaders = true;
    streamThresholdMb = parsedThreshold.mb;
    streamThresholdBytes = parsedThreshold.bytes;
  } else if (streamFlag === true) {
    streamShaders = true;
    streamThresholdMb = DEFAULT_STREAM_SHADERS_THRESHOLD_MB;
    streamThresholdBytes = DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES;
  }

  const dumpPipelines = dumpPipelinesFlag === true;
  const allowDumpWrites = allowDumpWritesFlag === true;
  const dumpEndpoint = allowDumpWrites ? DEFAULT_DUMP_ENDPOINT : null;

  return Object.freeze({
    inputScript,
    streamShaders,
    streamThresholdMb,
    streamThresholdBytes,
    dumpPipelines,
    allowDumpWrites,
    dumpEndpoint,
  });
}

function isStreamShadersExplicitlyDisabledInSearch(search) {
  const params = toUrlSearchParams(search);
  if (!params) {
    return false;
  }
  return parseBooleanQueryFlag(getQueryParamInsensitive(params, 'streamShaders')) === false;
}

function sanitizeDumpEndpoint(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const clean = sanitizeTelemetryUrl(raw);
  if (!clean) {
    return null;
  }
  return clean.endsWith('/') ? clean : `${clean}/`;
}

function resolveBooleanOverride(value, fallback) {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = parseBooleanQueryFlag(value);
    if (parsed !== null) {
      return parsed;
    }
  }
  return fallback;
}

function resolveDumpConfigOverride(source, base) {
  const baseDumpPipelines = Boolean(base && base.dumpPipelines === true);
  const baseAllowDumpWrites = Boolean(base && base.allowDumpWrites === true);
  const baseDumpEndpoint = baseAllowDumpWrites
    ? (base && typeof base.dumpEndpoint === 'string' && base.dumpEndpoint) || DEFAULT_DUMP_ENDPOINT
    : null;

  if (!source || typeof source !== 'object') {
    return Object.freeze({
      dumpPipelines: baseDumpPipelines,
      allowDumpWrites: baseAllowDumpWrites,
      dumpEndpoint: baseDumpEndpoint,
    });
  }

  const hasDumpPipelines =
    Object.prototype.hasOwnProperty.call(source, 'dumpPipelines') &&
    source.dumpPipelines !== undefined;
  const hasAllowDumpWrites =
    Object.prototype.hasOwnProperty.call(source, 'allowDumpWrites') &&
    source.allowDumpWrites !== undefined;
  const hasDumpEndpoint =
    Object.prototype.hasOwnProperty.call(source, 'dumpEndpoint') &&
    source.dumpEndpoint !== undefined;

  if (!hasDumpPipelines && !hasAllowDumpWrites && !hasDumpEndpoint) {
    return Object.freeze({
      dumpPipelines: baseDumpPipelines,
      allowDumpWrites: baseAllowDumpWrites,
      dumpEndpoint: baseDumpEndpoint,
    });
  }

  const dumpPipelines = hasDumpPipelines
    ? resolveBooleanOverride(source.dumpPipelines, baseDumpPipelines)
    : baseDumpPipelines;
  const allowDumpWrites = hasAllowDumpWrites
    ? resolveBooleanOverride(source.allowDumpWrites, baseAllowDumpWrites)
    : baseAllowDumpWrites;

  let dumpEndpoint = null;
  if (allowDumpWrites) {
    if (hasDumpEndpoint) {
      dumpEndpoint = sanitizeDumpEndpoint(source.dumpEndpoint) || DEFAULT_DUMP_ENDPOINT;
    } else {
      dumpEndpoint = baseDumpEndpoint || DEFAULT_DUMP_ENDPOINT;
    }
  }

  return Object.freeze({
    dumpPipelines,
    allowDumpWrites,
    dumpEndpoint,
  });
}

function resolveInputScriptOverride(source, fallback) {
  if (!source || typeof source !== 'object') {
    return fallback;
  }
  if (
    !Object.prototype.hasOwnProperty.call(source, 'inputScript') ||
    source.inputScript === undefined
  ) {
    return fallback;
  }
  if (source.inputScript === null || source.inputScript === false || source.inputScript === '') {
    return null;
  }
  return sanitizeInputScriptPreset(source.inputScript);
}

function resolveStreamConfigOverride(source, base) {
  if (!source || typeof source !== 'object') {
    return base;
  }
  const hasStreamFlag =
    Object.prototype.hasOwnProperty.call(source, 'streamShaders') &&
    source.streamShaders !== undefined;
  const hasThresholdBytes =
    Object.prototype.hasOwnProperty.call(source, 'streamThresholdBytes') &&
    source.streamThresholdBytes !== undefined;
  const hasThresholdMb =
    Object.prototype.hasOwnProperty.call(source, 'streamThresholdMb') &&
    source.streamThresholdMb !== undefined;

  if (!hasStreamFlag && !hasThresholdBytes && !hasThresholdMb) {
    return base;
  }

  if (hasStreamFlag && source.streamShaders === false) {
    return Object.freeze({
      streamShaders: false,
      streamThresholdMb: null,
      streamThresholdBytes: null,
    });
  }

  let bytes = null;
  let mb = null;
  if (hasThresholdBytes) {
    const parsedBytes = parsePositiveStreamBytes(source.streamThresholdBytes);
    if (parsedBytes !== null) {
      bytes = parsedBytes;
      mb = parsedBytes / BYTES_PER_MIB;
    }
  }
  if (bytes === null && hasThresholdMb) {
    const parsedMb = parseStreamThresholdMb(source.streamThresholdMb);
    if (parsedMb !== null) {
      bytes = parsedMb.bytes;
      mb = parsedMb.mb;
    }
  }

  if (bytes !== null) {
    return Object.freeze({
      streamShaders: true,
      streamThresholdMb: mb,
      streamThresholdBytes: bytes,
    });
  }

  if (hasStreamFlag && source.streamShaders === true) {
    if (base && base.streamShaders && isPositiveFinite(base.streamThresholdBytes)) {
      return base;
    }
    return Object.freeze({
      streamShaders: true,
      streamThresholdMb: DEFAULT_STREAM_SHADERS_THRESHOLD_MB,
      streamThresholdBytes: DEFAULT_STREAM_SHADERS_THRESHOLD_BYTES,
    });
  }

  return base;
}

function pickBootUrl(...candidates) {
  for (const candidate of candidates) {
    if (typeof candidate === 'string') {
      const trimmed = candidate.trim();
      if (trimmed.length > 0) {
        return trimmed;
      }
    } else if (candidate && typeof candidate === 'object') {
      for (const prop of ['url', 'path', 'href']) {
        if (typeof candidate[prop] === 'string' && candidate[prop].trim().length > 0) {
          return candidate[prop].trim();
        }
      }
    }
  }
  return null;
}

function toNonNegativeInt(value, fallback = null) {
  if (
    value === undefined ||
    value === null ||
    value === '' ||
    typeof value === 'boolean' ||
    typeof value === 'symbol' ||
    typeof value === 'object'
  ) {
    return fallback;
  }
  if (typeof value === 'bigint') {
    if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
      return fallback;
    }
    return Number(value);
  }
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n) || n < 0) {
    return fallback;
  }
  return Math.floor(n);
}

function toSafeNonNegativeInt(value, fallback = 0) {
  const parsed = toNonNegativeInt(value, null);
  return parsed === null ? fallback : parsed;
}

function sanitizeNonNegativeFiniteNumber(raw) {
  if (
    raw === undefined ||
    raw === null ||
    raw === '' ||
    typeof raw === 'boolean' ||
    typeof raw === 'symbol' ||
    typeof raw === 'object' ||
    typeof raw === 'bigint'
  ) {
    return null;
  }
  const n = typeof raw === 'number' ? raw : Number(raw);
  if (!Number.isFinite(n) || n < 0) {
    return null;
  }
  return n === 0 ? 0 : n;
}

function sanitizeUnitInterval(raw) {
  if (
    raw === undefined ||
    raw === null ||
    raw === '' ||
    typeof raw === 'boolean' ||
    typeof raw === 'symbol' ||
    typeof raw === 'object' ||
    typeof raw === 'bigint'
  ) {
    return null;
  }
  const n = typeof raw === 'number' ? raw : Number(raw);
  if (!Number.isFinite(n)) {
    return null;
  }
  const clamped = Math.min(1, Math.max(0, n));
  return clamped === 0 ? 0 : clamped;
}

function sanitizePositiveInt(raw) {
  const parsed = toNonNegativeInt(raw, null);
  return parsed !== null && parsed >= 1 ? parsed : null;
}

function normalizeDominantBottleneck(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim().toLowerCase().replace(/[\s-]+/g, '_');
  return normalized && ALLOWED_FRAME_BOTTLENECKS.has(normalized) ? normalized : null;
}

function normalizeFastBootMilestone(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const normalized = raw.trim().toUpperCase().replace(/[\s-]+/g, '_');
  return normalized && FAST_BOOT_MILESTONE_SET.has(normalized) ? normalized : null;
}

function sanitizeFastBootPressureLevel(raw) {
  if (typeof raw !== 'string') {
    return 'unknown';
  }
  const normalized = raw.trim().toLowerCase();
  return FAST_BOOT_PRESSURE_LEVEL_SET.has(normalized) ? normalized : 'unknown';
}

function detectFastBootMilestonesFromLog(text) {
  if (typeof text !== 'string' || text.length === 0) {
    return [];
  }
  if (VISIBLE_FRAME_LOG_FAILURE_RE.test(text)) {
    return [];
  }
  const matched = [];
  for (let i = 0; i < FAST_BOOT_MILESTONE_LOG_PATTERNS.length; i += 1) {
    const [milestone, pattern] = FAST_BOOT_MILESTONE_LOG_PATTERNS[i];
    if (pattern.test(text)) {
      matched.push(milestone);
    }
  }
  return matched;
}

function normalizeIndirectDispatch(raw) {
  if (typeof raw !== 'string') {
    return 'unknown';
  }
  const normalized = raw.trim().toLowerCase();
  if (!normalized) {
    return 'unknown';
  }
  return Object.prototype.hasOwnProperty.call(INDIRECT_DISPATCH_ALIASES, normalized)
    ? INDIRECT_DISPATCH_ALIASES[normalized]
    : 'unknown';
}

function normalizeAudioOutputState(raw) {
  if (typeof raw !== 'string') {
    return 'unknown';
  }
  const normalized = raw.trim().toLowerCase();
  if (!normalized) {
    return 'unknown';
  }
  return Object.prototype.hasOwnProperty.call(AUDIO_OUTPUT_STATE_ALIASES, normalized)
    ? AUDIO_OUTPUT_STATE_ALIASES[normalized]
    : 'unknown';
}

function sanitizeTelemetryUrl(raw) {
  if (typeof raw !== 'string') {
    return null;
  }
  const trimmed = raw.trim();
  if (
    !trimmed ||
    trimmed.length > MAX_TELEMETRY_URL_LENGTH ||
    /[\x00-\x1F\x7F]/.test(trimmed) ||
    trimmed.includes('\\') ||
    WIN_DRIVE_PREFIX_RE.test(trimmed)
  ) {
    return null;
  }
  const withoutQuery = trimmed.split(/[?#]/, 1)[0];
  if (!withoutQuery) {
    return null;
  }
  if (ABSOLUTE_URL_RE.test(trimmed)) {
    if (!/^https?:/i.test(trimmed)) {
      return null;
    }
    if (typeof URL === 'function') {
      try {
        const parsed = new URL(trimmed);
        if (parsed.username || parsed.password) {
          return null;
        }
        const pathSegs = parsed.pathname.split('/');
        if (pathSegs.includes('..') || POSIX_HOST_ROOT_RE.test(parsed.pathname)) {
          return null;
        }
        return `${parsed.origin}${parsed.pathname}`;
      } catch {
        return null;
      }
    }
    return withoutQuery;
  }
  if (POSIX_HOST_ROOT_RE.test(withoutQuery)) {
    return null;
  }
  const segments = withoutQuery.replace(/^\/+/, '').split('/');
  for (let i = 0; i < segments.length; i += 1) {
    if (segments[i] === '..') {
      return null;
    }
  }
  return withoutQuery;
}

function sanitizeDiagnosticText(raw, maxLength = MAX_VALIDATION_ERROR_MESSAGE_LENGTH) {
  if (raw === null || raw === undefined) {
    return '';
  }
  let text = typeof raw === 'string' ? raw : String(raw);
  // Strip multi-line and inline JS stack frames so raw exception stacks never leak.
  text = text.replace(/\r?\n[ \t]*at[ \t]+[\s\S]*$/, '');
  text = text.replace(
    /\s+at\s+(?:[A-Za-z0-9_.$<>]+\s+\([^)]*:\d+:\d+\)|(?:file:\/\/|https?:\/\/|[A-Za-z]:[\\/]|\/)[^\s)]+:\d+:\d+)/g,
    ''
  );
  text = text.replace(/[\r\n\t]+/g, ' ').replace(/[\x00-\x1F\x7F]+/g, '');
  text = sanitizeString(text);
  // Strip embedded data URIs, byte arrays, long hex dumps, and base64 blobs so
  // diagnostic messages never carry raw memory or asset dumps.
  text = text.replace(/\bdata:[^\s;,'"`<>]+;base64,[A-Za-z0-9+/=]{16,}/gi, '<buffer>');
  text = text.replace(/(?:\b(?:0x)?[0-9a-fA-F]{1,4}\s*,\s*){16,}(?:0x)?[0-9a-fA-F]{1,4}\b/g, '<buffer>');
  text = text.replace(/(?:\b[0-9a-fA-F]{2}\s+){16,}[0-9a-fA-F]{2}\b/g, '<buffer>');
  text = text.replace(/\b(?:0x)?[0-9a-fA-F]{48,}\b/g, '<buffer>');
  text = text.replace(/(^|[\s"'`(\[=,:])[A-Za-z0-9+/]{64,}={0,2}(?=$|[\s"'`)\]=,.;:])/g, '$1<buffer>');
  text = text.replace(/\s{2,}/g, ' ').trim();
  if (text.length > maxLength) {
    text = `${text.slice(0, Math.max(0, maxLength - 3))}...`;
  }
  return text;
}

function sanitizeValidationErrorCode(raw, fallback = 'validation') {
  if (raw === null || raw === undefined || raw === '') {
    return fallback;
  }
  if (typeof raw === 'number') {
    return Number.isFinite(raw) ? String(Math.trunc(raw)) : fallback;
  }
  if (typeof raw !== 'string') {
    return fallback;
  }
  const cleaned = sanitizeDiagnosticText(raw, MAX_VALIDATION_ERROR_CODE_LENGTH);
  if (!cleaned || cleaned.includes('[REDACTED')) {
    return fallback;
  }
  return cleaned;
}

function classifyGpuValidationErrorType(raw, fallback = 'validation') {
  if (typeof raw !== 'string') {
    return fallback;
  }
  const normalized = raw.trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if (
    normalized === 'out-of-memory' ||
    normalized === 'out_of_memory' ||
    normalized === 'outofmemory' ||
    normalized === 'oom' ||
    normalized.includes('outofmemory') ||
    normalized.includes('out-of-memory') ||
    normalized.includes('out_of_memory')
  ) {
    return 'out-of-memory';
  }
  if (normalized === 'validation' || normalized.includes('validation')) {
    return 'validation';
  }
  if (normalized === 'internal' || normalized.includes('internal')) {
    return 'internal';
  }
  if (normalized === 'unknown') {
    return 'unknown';
  }
  return fallback;
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
  if (ABSOLUTE_URL_RE.test(trimmed)) {
    if (!/^https?:/i.test(trimmed)) {
      return null;
    }
    if (typeof URL === 'function') {
      try {
        const parsed = new URL(trimmed);
        if (parsed.username || parsed.password) {
          return null;
        }
        const pathSegs = parsed.pathname.split('/');
        if (pathSegs.includes('..') || POSIX_HOST_ROOT_RE.test(parsed.pathname)) {
          return null;
        }
        return `${parsed.origin}${parsed.pathname}`;
      } catch {
        return null;
      }
    }
    return withoutQuery;
  }
  if (POSIX_HOST_ROOT_RE.test(withoutQuery)) {
    return null;
  }
  const segments = withoutQuery.replace(/^\/+/, '').split('/');
  for (let i = 0; i < segments.length; i += 1) {
    if (segments[i] === '..') {
      return null;
    }
  }
  return withoutQuery;
}

export function sanitizeSideModuleUrls(rawList) {
  if (!Array.isArray(rawList)) {
    return [];
  }
  const out = [];
  const seen = new Set();
  for (let i = 0; i < rawList.length && out.length < MAX_SIDE_MODULE_URLS; i += 1) {
    const clean = sanitizeSideModuleUrl(rawList[i]);
    if (clean && !seen.has(clean)) {
      seen.add(clean);
      out.push(clean);
    }
  }
  return out;
}

export function extractBootAssetUrls(source) {
  if (!source || typeof source !== 'object') {
    return {};
  }
  const out = {};
  const bootBase = pickBootUrl(source.bootBase);
  const gameBase = pickBootUrl(source.gameBase);
  const shaderPackUrl = pickBootUrl(
    source.preferredShaderPackUrl,
    source.shaderPackUrl,
    source.shaderPack,
    source.wgslPack
  );
  const reflectionManifestUrl = pickBootUrl(
    source.reflectionManifestUrl,
    source.reflectionUrl,
    source.reflectionManifest
  );
  const pgimUrl = pickBootUrl(source.pgimUrl, source.pgim, source.pgimImage);
  const programManifestUrl = pickBootUrl(source.programManifestUrl, source.programManifest);
  if (bootBase) out.bootBase = bootBase;
  if (gameBase) out.gameBase = gameBase;
  if (shaderPackUrl) out.shaderPackUrl = shaderPackUrl;
  if (reflectionManifestUrl) out.reflectionManifestUrl = reflectionManifestUrl;
  if (pgimUrl) out.pgimUrl = pgimUrl;
  if (programManifestUrl) out.programManifestUrl = programManifestUrl;

  if (Array.isArray(source.sideModuleUrls)) {
    out.sideModuleUrls = sanitizeSideModuleUrls(source.sideModuleUrls);
  }
  const parsedSideModuleCount = toNonNegativeInt(source.sideModuleCount, null);
  if (parsedSideModuleCount !== null) {
    out.sideModuleCount =
      Array.isArray(out.sideModuleUrls) && out.sideModuleUrls.length > parsedSideModuleCount
        ? out.sideModuleUrls.length
        : parsedSideModuleCount;
  } else if (Array.isArray(out.sideModuleUrls)) {
    out.sideModuleCount = out.sideModuleUrls.length;
  }
  return out;
}

const DEFAULT_MAX_DEVICE_PIXEL_RATIO = 2;

function resolveViewportMetrics(canvas, windowObj, overrides = {}) {
  const maxDpr = isPositiveFinite(overrides.maxDevicePixelRatio)
    ? overrides.maxDevicePixelRatio
    : DEFAULT_MAX_DEVICE_PIXEL_RATIO;
  const dpr = isPositiveFinite(overrides.devicePixelRatio)
    ? overrides.devicePixelRatio
    : isPositiveFinite(windowObj && windowObj.devicePixelRatio)
      ? Math.min(windowObj.devicePixelRatio, maxDpr)
      : 1;

  const isHtmlDefaultCanvasSize = Boolean(
    canvas &&
      !isPositiveFinite(canvas.clientWidth) &&
      !isPositiveFinite(canvas.clientHeight) &&
      canvas.width === 300 &&
      canvas.height === 150
  );

  const width = isPositiveFinite(overrides.width)
    ? overrides.width
    : isPositiveFinite(canvas && canvas.clientWidth)
      ? canvas.clientWidth
      : !isHtmlDefaultCanvasSize && isPositiveFinite(canvas && canvas.width)
        ? canvas.width
        : isPositiveFinite(windowObj && windowObj.innerWidth)
          ? windowObj.innerWidth
          : isPositiveFinite(canvas && canvas.width)
            ? canvas.width
            : DEFAULT_VIEWPORT_WIDTH;

  const height = isPositiveFinite(overrides.height)
    ? overrides.height
    : isPositiveFinite(canvas && canvas.clientHeight)
      ? canvas.clientHeight
      : !isHtmlDefaultCanvasSize && isPositiveFinite(canvas && canvas.height)
        ? canvas.height
        : isPositiveFinite(windowObj && windowObj.innerHeight)
          ? windowObj.innerHeight
          : isPositiveFinite(canvas && canvas.height)
            ? canvas.height
            : DEFAULT_VIEWPORT_HEIGHT;

  return { width, height, dpr };
}

function resolvePerfProfileApi(options = null, windowObj = null) {
  if (
    options &&
    typeof options === 'object' &&
    options.perfProfileApi &&
    typeof options.perfProfileApi.resolvePerfProfile === 'function'
  ) {
    return options.perfProfileApi;
  }
  if (
    windowObj &&
    typeof windowObj === 'object' &&
    windowObj.PlayGta6PerfProfile &&
    typeof windowObj.PlayGta6PerfProfile.resolvePerfProfile === 'function'
  ) {
    return windowObj.PlayGta6PerfProfile;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis.PlayGta6PerfProfile &&
    typeof globalThis.PlayGta6PerfProfile.resolvePerfProfile === 'function'
  ) {
    return globalThis.PlayGta6PerfProfile;
  }
  return null;
}

function cloneDisplayBounds(bounds) {
  if (!bounds || typeof bounds !== 'object') {
    return null;
  }
  return Object.freeze({
    width: bounds.width,
    height: bounds.height,
    maxDisplayWidth: bounds.maxDisplayWidth,
    maxDisplayHeight: bounds.maxDisplayHeight,
    screenWidth: bounds.screenWidth,
    screenHeight: bounds.screenHeight,
    devicePixelRatio: bounds.devicePixelRatio,
    maxTextureDimension2D:
      bounds.maxTextureDimension2D !== undefined ? bounds.maxTextureDimension2D : null,
    clampedToMaxTextureDimension: Boolean(bounds.clampedToMaxTextureDimension),
  });
}

function displayBoundsKey(bounds) {
  if (!bounds || typeof bounds !== 'object') {
    return '';
  }
  return `${bounds.maxDisplayWidth}x${bounds.maxDisplayHeight}@${bounds.devicePixelRatio}:${bounds.screenWidth}x${bounds.screenHeight}`;
}

/**
 * Safely resolves display-native physical pixel bounds from `window.screen` and
 * `window.devicePixelRatio` (with safe guards when `window`/`screen` are absent in Node/worker tests).
 *
 * @param {object} [params]
 * @returns {Readonly<{
 *   width: number,
 *   height: number,
 *   maxDisplayWidth: number,
 *   maxDisplayHeight: number,
 *   screenWidth: number,
 *   screenHeight: number,
 *   devicePixelRatio: number,
 *   maxTextureDimension2D: number | null,
 *   clampedToMaxTextureDimension: boolean,
 * }> | null}
 */
export function resolveDisplayBounds(params = {}) {
  if (!params || typeof params !== 'object') {
    return null;
  }
  const windowObj =
    params.windowObj && typeof params.windowObj === 'object' ? params.windowObj : null;
  const screenObj =
    params.screenObj && typeof params.screenObj === 'object'
      ? params.screenObj
      : windowObj && windowObj.screen && typeof windowObj.screen === 'object'
        ? windowObj.screen
        : null;
  const api = resolvePerfProfileApi(params, windowObj);

  const explicit = params.displayBounds;
  if (explicit === null) {
    return null;
  }

  let screenWidth = null;
  let screenHeight = null;
  let dpr = null;
  let maxTextureDimension2D = isPositiveFinite(params.maxTextureDimension2D)
    ? Math.floor(params.maxTextureDimension2D)
    : null;

  if (explicit && typeof explicit === 'object') {
    if (isPositiveFinite(explicit.maxTextureDimension2D)) {
      maxTextureDimension2D = Math.floor(explicit.maxTextureDimension2D);
    }
    if (isPositiveFinite(explicit.maxDisplayWidth) && isPositiveFinite(explicit.maxDisplayHeight)) {
      const explicitDpr = isPositiveFinite(explicit.devicePixelRatio)
        ? explicit.devicePixelRatio
        : isPositiveFinite(explicit.dpr)
          ? explicit.dpr
          : 1;
      const maxW = Math.max(1, Math.round(explicit.maxDisplayWidth));
      const maxH = Math.max(1, Math.round(explicit.maxDisplayHeight));
      const sw = isPositiveFinite(explicit.screenWidth)
        ? explicit.screenWidth
        : Math.max(1, Math.round(maxW / explicitDpr));
      const sh = isPositiveFinite(explicit.screenHeight)
        ? explicit.screenHeight
        : Math.max(1, Math.round(maxH / explicitDpr));
      let width = maxW;
      let height = maxH;
      let clampedToMaxTextureDimension = Boolean(explicit.clampedToMaxTextureDimension);
      if (maxTextureDimension2D !== null) {
        if (width > maxTextureDimension2D) {
          width = maxTextureDimension2D;
          clampedToMaxTextureDimension = true;
        }
        if (height > maxTextureDimension2D) {
          height = maxTextureDimension2D;
          clampedToMaxTextureDimension = true;
        }
      }
      return Object.freeze({
        width,
        height,
        maxDisplayWidth: width,
        maxDisplayHeight: height,
        screenWidth: sw,
        screenHeight: sh,
        devicePixelRatio: explicitDpr,
        maxTextureDimension2D,
        clampedToMaxTextureDimension,
      });
    }
    if (isPositiveFinite(explicit.screenWidth) && isPositiveFinite(explicit.screenHeight)) {
      screenWidth = explicit.screenWidth;
      screenHeight = explicit.screenHeight;
      dpr = isPositiveFinite(explicit.devicePixelRatio)
        ? explicit.devicePixelRatio
        : isPositiveFinite(explicit.dpr)
          ? explicit.dpr
          : null;
    } else if (isPositiveFinite(explicit.width) && isPositiveFinite(explicit.height)) {
      screenWidth = explicit.width;
      screenHeight = explicit.height;
      dpr = isPositiveFinite(explicit.devicePixelRatio)
        ? explicit.devicePixelRatio
        : isPositiveFinite(explicit.dpr)
          ? explicit.dpr
          : 1;
    }
  }

  if (screenWidth === null || screenHeight === null) {
    const rawW =
      params.screenWidth !== undefined
        ? params.screenWidth
        : screenObj && screenObj.width !== undefined
          ? screenObj.width
          : null;
    const rawH =
      params.screenHeight !== undefined
        ? params.screenHeight
        : screenObj && screenObj.height !== undefined
          ? screenObj.height
          : null;
    if (isPositiveFinite(rawW) && isPositiveFinite(rawH)) {
      screenWidth = rawW;
      screenHeight = rawH;
    }
  }

  if (screenWidth === null || screenHeight === null) {
    return null;
  }

  if (dpr === null) {
    const rawDpr =
      params.devicePixelRatio !== undefined
        ? params.devicePixelRatio
        : windowObj && isPositiveFinite(windowObj.devicePixelRatio)
          ? windowObj.devicePixelRatio
          : 1;
    dpr = isPositiveFinite(rawDpr) ? rawDpr : 1;
  }

  if (api && typeof api.computeDisplayNativeBounds === 'function') {
    const computed = api.computeDisplayNativeBounds({
      screenWidth,
      screenHeight,
      devicePixelRatio: dpr,
      maxTextureDimension2D,
    });
    return cloneDisplayBounds(computed);
  }

  let width = Math.max(1, Math.round(screenWidth * dpr));
  let height = Math.max(1, Math.round(screenHeight * dpr));
  let clampedToMaxTextureDimension = false;
  if (maxTextureDimension2D !== null) {
    if (width > maxTextureDimension2D) {
      width = maxTextureDimension2D;
      clampedToMaxTextureDimension = true;
    }
    if (height > maxTextureDimension2D) {
      height = maxTextureDimension2D;
      clampedToMaxTextureDimension = true;
    }
  }
  return Object.freeze({
    width,
    height,
    maxDisplayWidth: width,
    maxDisplayHeight: height,
    screenWidth,
    screenHeight,
    devicePixelRatio: dpr,
    maxTextureDimension2D,
    clampedToMaxTextureDimension,
  });
}

function toCloneablePerfProfile(profile, displayBounds = null) {
  if (!profile || typeof profile !== 'object') {
    return null;
  }
  const cleanDisplayBounds = cloneDisplayBounds(
    displayBounds !== undefined ? displayBounds : profile.displayNativeBounds || null
  );
  const cleanHardware =
    profile.hardware && typeof profile.hardware === 'object'
      ? Object.freeze({
          deviceMemoryGb: profile.hardware.deviceMemoryGb ?? null,
          hardwareConcurrency: profile.hardware.hardwareConcurrency ?? null,
          maxBufferSize: profile.hardware.maxBufferSize ?? null,
          maxStorageBufferBindingSize: profile.hardware.maxStorageBufferBindingSize ?? null,
          maxTextureDimension2D: profile.hardware.maxTextureDimension2D ?? null,
          isSoftwareAdapter: Boolean(profile.hardware.isSoftwareAdapter),
          isLowPowerAdapter: Boolean(profile.hardware.isLowPowerAdapter),
        })
      : null;

  return Object.freeze({
    name: typeof profile.name === 'string' ? profile.name : 'balanced',
    preset: typeof profile.preset === 'string' ? profile.preset : profile.name || 'balanced',
    requestedPreset:
      typeof profile.requestedPreset === 'string' ? profile.requestedPreset : 'auto',
    source: typeof profile.source === 'string' ? profile.source : 'auto',
    rtPoolMaxBytes: profile.rtPoolMaxBytes,
    rtPoolMaxIdleFrames: profile.rtPoolMaxIdleFrames,
    vramBudgetBytes: profile.vramBudgetBytes,
    vramReclaimTriggerRatio: profile.vramReclaimTriggerRatio,
    vramReclaimTargetRatio: profile.vramReclaimTargetRatio,
    gcDeferFrames: profile.gcDeferFrames,
    ioSlabCacheBytes: profile.ioSlabCacheBytes,
    ioMaxConcurrentLanes: profile.ioMaxConcurrentLanes,
    // Always preserve full visual quality (lossless textureMipCap: 0).
    textureMipCap: 0,
    dprCap: profile.dprCap ?? null,
    preserveFullQuality: true,
    clampToDisplayResolution: profile.clampToDisplayResolution !== false,
    clampToDisplayNative: profile.clampToDisplayNative !== false,
    prewarmMaxInFlight: profile.prewarmMaxInFlight,
    shaderCompileConcurrency: profile.shaderCompileConcurrency,
    displayNativeBounds: cleanDisplayBounds,
    reasons: Object.freeze(Array.isArray(profile.reasons) ? profile.reasons.slice() : []),
    hardware: cleanHardware,
    overridesApplied: Object.freeze(
      Array.isArray(profile.overridesApplied) ? profile.overridesApplied.slice() : []
    ),
  });
}

/**
 * Resolves the hardware-adaptive, full-visual-quality (`textureMipCap: 0`) performance profile
 * and display-native physical pixel bounds safely across browsers and headless Node environments.
 *
 * @param {object} [params]
 * @returns {Readonly<{ perfProfile: Readonly<object>, displayBounds: Readonly<object> | null }>}
 */
export function resolveBootstrapPerfProfile(params = {}) {
  const opts = params && typeof params === 'object' ? params : {};
  const bootstrapOpts =
    opts.options && typeof opts.options === 'object' ? opts.options : opts;
  const windowObj =
    opts.windowObj !== undefined
      ? opts.windowObj
      : typeof window !== 'undefined'
        ? window
        : null;
  const navigatorObj =
    opts.navigatorObj !== undefined
      ? opts.navigatorObj
      : windowObj && windowObj.navigator
        ? windowObj.navigator
        : null;
  const urlSearch =
    opts.urlSearch !== undefined
      ? opts.urlSearch
      : windowObj && windowObj.location
        ? typeof windowObj.location.search === 'string'
          ? windowObj.location.search
          : typeof windowObj.location.href === 'string'
            ? windowObj.location.href
            : ''
        : '';

  const api = resolvePerfProfileApi(opts.perfProfileApi ? opts : bootstrapOpts, windowObj);
  const displayBounds = resolveDisplayBounds({
    windowObj,
    screenObj: opts.screenObj ?? bootstrapOpts.screenObj,
    screenWidth: opts.screenWidth ?? bootstrapOpts.screenWidth,
    screenHeight: opts.screenHeight ?? bootstrapOpts.screenHeight,
    devicePixelRatio: opts.devicePixelRatio ?? bootstrapOpts.devicePixelRatio,
    maxTextureDimension2D: opts.maxTextureDimension2D ?? bootstrapOpts.maxTextureDimension2D,
    displayBounds:
      opts.displayBounds !== undefined ? opts.displayBounds : bootstrapOpts.displayBounds,
    perfProfileApi: api,
  });

  const explicitProfile =
    opts.perfProfile !== undefined ? opts.perfProfile : bootstrapOpts.perfProfile;
  const resolveInput = {
    queryString: urlSearch,
    useNavigator: Boolean(navigatorObj && typeof navigatorObj === 'object'),
    navigator: navigatorObj && typeof navigatorObj === 'object' ? navigatorObj : null,
  };

  if (navigatorObj && typeof navigatorObj === 'object') {
    if (isPositiveFinite(navigatorObj.deviceMemory)) {
      resolveInput.deviceMemoryGb = navigatorObj.deviceMemory;
    }
    if (isPositiveFinite(navigatorObj.hardwareConcurrency)) {
      resolveInput.hardwareConcurrency = navigatorObj.hardwareConcurrency;
    }
  }
  if (isPositiveFinite(bootstrapOpts.deviceMemoryGb)) {
    resolveInput.deviceMemoryGb = bootstrapOpts.deviceMemoryGb;
  }
  if (isPositiveFinite(bootstrapOpts.hardwareConcurrency)) {
    resolveInput.hardwareConcurrency = bootstrapOpts.hardwareConcurrency;
  }
  if (bootstrapOpts.adapterInfo !== undefined) {
    resolveInput.adapterInfo = bootstrapOpts.adapterInfo;
  }
  if (bootstrapOpts.adapterLimits !== undefined) {
    resolveInput.adapterLimits = bootstrapOpts.adapterLimits;
  }
  if (displayBounds) {
    resolveInput.screenWidth = displayBounds.screenWidth;
    resolveInput.screenHeight = displayBounds.screenHeight;
    resolveInput.devicePixelRatio = displayBounds.devicePixelRatio;
    if (displayBounds.maxTextureDimension2D !== null) {
      resolveInput.maxTextureDimension2D = displayBounds.maxTextureDimension2D;
    }
  }

  if (typeof explicitProfile === 'string') {
    resolveInput.preset = explicitProfile;
  } else if (explicitProfile && typeof explicitProfile === 'object') {
    if (typeof explicitProfile.preset === 'string') {
      resolveInput.preset = explicitProfile.preset;
    } else if (typeof explicitProfile.name === 'string') {
      resolveInput.preset = explicitProfile.name;
    }
    resolveInput.overrides = {
      ...explicitProfile,
      textureMipCap: 0,
      preserveFullQuality: true,
    };
  } else if (typeof bootstrapOpts.quality === 'string') {
    resolveInput.preset = bootstrapOpts.quality;
  }

  let resolvedRaw = null;
  if (api && typeof api.resolvePerfProfile === 'function') {
    resolvedRaw = api.resolvePerfProfile(resolveInput);
  } else {
    const MiB = 1024 * 1024;
    resolvedRaw = {
      name: 'balanced',
      preset: 'balanced',
      requestedPreset: 'auto',
      source: 'auto',
      rtPoolMaxBytes: 256 * MiB,
      rtPoolMaxIdleFrames: 16,
      vramBudgetBytes: 3072 * MiB,
      vramReclaimTriggerRatio: 0.78,
      vramReclaimTargetRatio: 0.65,
      gcDeferFrames: 2,
      ioSlabCacheBytes: 128 * MiB,
      ioMaxConcurrentLanes: 6,
      textureMipCap: 0,
      dprCap: null,
      preserveFullQuality: true,
      clampToDisplayResolution: true,
      clampToDisplayNative: true,
      prewarmMaxInFlight: 2,
      shaderCompileConcurrency: 2,
      displayNativeBounds: displayBounds,
      reasons: ['fallback:balanced'],
      hardware: null,
      overridesApplied: [],
    };
  }

  const finalBounds = displayBounds || cloneDisplayBounds(resolvedRaw.displayNativeBounds || null);
  const perfProfile = toCloneablePerfProfile(resolvedRaw, finalBounds);
  return Object.freeze({ perfProfile, displayBounds: finalBounds });
}

/**
 * Builds structured-clone-safe worker init fields for `ioWorker` and `gpuWorker` from a
 * resolved `perfProfile` and optional `displayBounds`.
 */
export function buildWorkerPerfInitPayloads(perfProfile, displayBounds = null) {
  const cleanBounds = cloneDisplayBounds(
    displayBounds !== undefined && displayBounds !== null
      ? displayBounds
      : perfProfile && perfProfile.displayNativeBounds
        ? perfProfile.displayNativeBounds
        : null
  );
  const cleanProfile = toCloneablePerfProfile(perfProfile, cleanBounds);
  return Object.freeze({
    ioWorker: Object.freeze({
      ...(cleanProfile ? { perfProfile: cleanProfile } : {}),
    }),
    gpuWorker: Object.freeze({
      ...(cleanProfile ? { perfProfile: cleanProfile } : {}),
      ...(cleanBounds ? { displayBounds: cleanBounds } : {}),
    }),
  });
}

function createCapabilityProbe({ windowObj, navigatorObj, webgpuAdapterFactory }) {
  return () => {
    const g = typeof globalThis !== 'undefined' ? globalThis : {};
    const host = windowObj && typeof windowObj === 'object' ? windowObj : g;
    const nav = navigatorObj && typeof navigatorObj === 'object' ? navigatorObj : null;

    const crossOriginIsolated =
      windowObj && typeof windowObj === 'object' && 'crossOriginIsolated' in windowObj
        ? windowObj.crossOriginIsolated === true
        : g.crossOriginIsolated === true;

    const sharedArrayBuffer =
      windowObj && typeof windowObj === 'object' && 'SharedArrayBuffer' in windowObj
        ? typeof windowObj.SharedArrayBuffer === 'function'
        : typeof SharedArrayBuffer === 'function';

    const webgpu = Boolean(
      typeof webgpuAdapterFactory === 'function' || (nav && 'gpu' in nav && nav.gpu != null)
    );

    const wasmApi =
      windowObj && typeof windowObj === 'object' && 'WebAssembly' in windowObj
        ? windowObj.WebAssembly
        : typeof WebAssembly !== 'undefined'
          ? WebAssembly
          : null;
    const wasm64 = detectWasm64(wasmApi);

    const opfs = Boolean(nav && nav.storage && typeof nav.storage.getDirectory === 'function');
    const gamepad = Boolean(nav && typeof nav.getGamepads === 'function');

    const hasAudioContext =
      typeof host.AudioContext === 'function' ||
      typeof host.webkitAudioContext === 'function' ||
      typeof g.AudioContext === 'function' ||
      typeof g.webkitAudioContext === 'function';
    const hasAudioWorkletNode =
      typeof host.AudioWorkletNode === 'function' || typeof g.AudioWorkletNode === 'function';

    return {
      crossOriginIsolated,
      sharedArrayBuffer,
      webgpu,
      wasm64,
      opfs,
      gamepad,
      audioWorklet: Boolean(hasAudioContext && hasAudioWorkletNode),
    };
  };
}

function computePhaseProgressRatio(progress) {
  if (!progress || typeof progress !== 'object') {
    return null;
  }
  if (typeof progress.pct === 'number' && Number.isFinite(progress.pct)) {
    return Math.min(1, Math.max(0, progress.pct / 100));
  }
  if (typeof progress.progressPct === 'number' && Number.isFinite(progress.progressPct)) {
    return Math.min(1, Math.max(0, progress.progressPct / 100));
  }
  const loaded = Number(progress.loadedBytes);
  const total = Number(progress.totalBytes);
  if (Number.isFinite(loaded) && Number.isFinite(total) && total > 0) {
    return Math.min(1, Math.max(0, loaded / total));
  }
  return null;
}

const PRIMARY_PRELOAD_PROGRESS_ASSETS = new Set(['pgim', 'bundle', 'shaderpack', 'wasm']);

function isPrimaryPreloadProgress(progress) {
  if (!progress || typeof progress !== 'object') {
    return false;
  }
  if (typeof progress.asset !== 'string' || progress.asset.trim().length === 0) {
    return true;
  }
  return PRIMARY_PRELOAD_PROGRESS_ASSETS.has(progress.asset.trim().toLowerCase());
}

function createSplashOverlayAdapter(splashOverlay) {
  if (!splashOverlay || typeof splashOverlay !== 'object') {
    return null;
  }
  const hasBootState = typeof splashOverlay.updateBootState === 'function';
  if (!hasBootState) {
    return splashOverlay;
  }
  return {
    init(payload = {}) {
      if (typeof splashOverlay.show === 'function') {
        splashOverlay.show();
      }
      if (payload.splashImage && typeof splashOverlay.setSplashImageUrl === 'function') {
        splashOverlay.setSplashImageUrl(payload.splashImage);
      }
      if (payload.tracer) {
        splashOverlay.updateBootState(payload.tracer);
      }
    },
    show(opts) {
      if (typeof splashOverlay.show === 'function') {
        splashOverlay.show(opts);
      }
    },
    setSplashImage(url) {
      if (typeof splashOverlay.setSplashImageUrl === 'function') {
        splashOverlay.setSplashImageUrl(url);
      } else if (typeof splashOverlay.setSplashImage === 'function') {
        splashOverlay.setSplashImage(url);
      }
    },
    update(payload = {}) {
      const base = payload.tracer && typeof payload.tracer === 'object' ? { ...payload.tracer } : {};
      const hasStage = typeof base.stage === 'string' && base.stage.length > 0 && base.stage !== 'NOT_STARTED';
      if (payload.progress && typeof payload.progress === 'object') {
        base.streamingProgress = payload.progress;
        if (isPrimaryPreloadProgress(payload.progress)) {
          const ratio = computePhaseProgressRatio(payload.progress);
          if (ratio !== null && base.phaseProgress === undefined) {
            base.phaseProgress = ratio;
          }
        }
      }
      if (payload.phase === 'DISCOVERING') {
        if (!hasStage) {
          base.stage = 'PGIM_LOAD';
          if (base.phaseProgress === undefined) {
            base.phaseProgress = 0.05;
          }
        }
        if (!base.statusText) {
          base.statusText = 'Discovering local boot assets...';
        }
      } else if (payload.phase === 'PRELOADING') {
        if (!hasStage) {
          base.stage = 'PGIM_LOAD';
        }
        if (!base.statusText) {
          base.statusText = 'Streaming guest image & shaders...';
        }
      } else if (payload.phase === 'LAUNCHING_RUNTIME') {
        if (!hasStage || base.stage === 'PGIM_LOAD') {
          base.stage = 'WASM_INIT';
        }
        if (!base.statusText) {
          base.statusText = 'Launching Wasm64 runtime...';
        }
      } else if (payload.phase === 'FALLBACK_SMOKE_DEMO') {
        if (!hasStage || base.stage === 'PGIM_LOAD') {
          base.stage = 'D3D12_INIT';
        }
        if (!base.statusText) {
          base.statusText = 'Starting WebGPU smoke demo...';
        }
      }
      splashOverlay.updateBootState(base);
    },
    updateProgress(progress) {
      const patch = { streamingProgress: progress };
      if (isPrimaryPreloadProgress(progress)) {
        const ratio = computePhaseProgressRatio(progress);
        if (ratio !== null) {
          patch.stage = 'PGIM_LOAD';
          patch.phaseProgress = ratio;
        }
      }
      splashOverlay.updateBootState(patch);
    },
    updateTracer(tracer) {
      splashOverlay.updateBootState(tracer);
    },
    dismiss(meta) {
      if (typeof splashOverlay.onFirstPresent === 'function') {
        splashOverlay.onFirstPresent(meta);
      } else if (typeof splashOverlay.dismiss === 'function') {
        splashOverlay.dismiss(meta);
      }
    },
    setLoadingAssetsProgress(progress) {
      if (typeof splashOverlay.setLoadingAssetsProgress === 'function') {
        splashOverlay.setLoadingAssetsProgress(progress);
      }
    },
    notifyFirstVisibleFrame(meta) {
      if (typeof splashOverlay.notifyFirstVisibleFrame === 'function') {
        splashOverlay.notifyFirstVisibleFrame(meta);
      } else if (typeof splashOverlay.dismiss === 'function') {
        splashOverlay.dismiss(meta);
      }
    },
    hide(hideOpts) {
      // Allow smooth dismiss fade or visible-frame hold to remain active until finished.
      const state = typeof splashOverlay.getState === 'function' ? splashOverlay.getState() : null;
      if (state && (state.fading || state.holdingForVisibleFrame)) {
        return;
      }
      if (typeof splashOverlay.hide === 'function') {
        if (hideOpts !== undefined) {
          splashOverlay.hide(hideOpts);
        } else {
          splashOverlay.hide();
        }
      }
    },
    destroy() {
      if (typeof splashOverlay.destroy === 'function') {
        splashOverlay.destroy();
      }
    },
  };
}

/**
 * Unified lifecycle handle returned by `bootstrapPlayGta6App(options)`.
 */
export class PlayGta6AppHandle {
  constructor(components) {
    this.session = components.session;
    this.perfMonitor = components.perfMonitor;
    this.crashReporter = components.crashReporter;
    this.inputProfiles = components.inputProfiles;
    this.inputBridge = components.inputBridge;
    this.audioEngine = components.audioEngine;
    this.storageManager = components.storageManager;
    this.bootTracer = components.bootTracer || null;
    this.splashOverlay = components.splashOverlay || null;
    this.gpuBootPanel = components.gpuBootPanel || null;
    this.pgimPreloader = components.pgimPreloader || null;
    this.bootOrchestrator = components.bootOrchestrator || null;
    this.bootUrlOptions = components.bootUrlOptions || null;

    this._perfProfileFallback = components.perfProfile || null;
    this._displayBoundsFallback = components.displayBounds || null;
    this._getPerfProfile = components.getPerfProfile || (() => this._perfProfileFallback);
    this._getDisplayBounds = components.getDisplayBounds || (() => this._displayBoundsFallback);
    this._setPerfProfile = components.setPerfProfile || (() => this._perfProfileFallback);
    this._setDisplayBounds = components.setDisplayBounds || (() => this._displayBoundsFallback);
    this._getSmokeRunner = components.getSmokeRunner;
    this._getWasmWorker = components.getWasmWorker;
    this._getDiagnosticsSnapshot = components.getDiagnosticsSnapshot || (() => null);
    this._getBootTelemetry = components.getBootTelemetry || (() => null);
    this._getFastBootPlan = components.getFastBootPlan || (() => null);
    this._getPhaseTelemetry = components.getPhaseTelemetry || (() => null);
    this._reclaimSoftMemory = components.reclaimSoftMemory || (async () => ({ ok: false }));
    this._registerReclaimHook = components.registerReclaimHook || (() => () => {});
    this._attachAudioRing = components.attachAudioRing || (() => null);
    this._unlockAudio = components.unlockAudio || (async () => ({ ok: false, state: 'idle', unlocked: false }));
    this._getAudioStatus = components.getAudioStatus || (() => null);
    this._requestPointerLock = components.requestPointerLock || (async () => ({ ok: false, locked: false }));
    this._exitPointerLock = components.exitPointerLock || (() => ({ ok: false, locked: false }));
    this._togglePointerLock = components.togglePointerLock || (async () => ({ ok: false, locked: false }));
    this._isPointerLocked = components.isPointerLocked || (() => false);
    this._notifyFirstPresent = components.notifyFirstPresent || (() => ({ ok: true }));
    this._notifyFirstVisibleFrame = components.notifyFirstVisibleFrame || (() => ({ ok: true }));
    this._recordFastBootMilestone = components.recordFastBootMilestone || (() => null);
    this._start = components.start;
    this._stop = components.stop;
    this._resize = components.resize;
    this._destroy = components.destroy;
    this._isDestroyed = components.isDestroyed;
  }

  get destroyed() {
    return this._isDestroyed();
  }

  get perfProfile() {
    return this._getPerfProfile();
  }

  get displayBounds() {
    return this._getDisplayBounds();
  }

  getSmokeRunner() {
    return this._getSmokeRunner();
  }

  getPerfProfile() {
    return this._getPerfProfile();
  }

  getDisplayBounds() {
    return this._getDisplayBounds();
  }

  setPerfProfile(profileOrName) {
    const next = this._setPerfProfile(profileOrName);
    this._perfProfileFallback = next;
    return next;
  }

  setDisplayBounds(boundsOrWidth, screenHeight, dpr) {
    const next = this._setDisplayBounds(boundsOrWidth, screenHeight, dpr);
    this._displayBoundsFallback = next;
    return next;
  }

  getWasmWorker() {
    return this._getWasmWorker();
  }

  getDiagnosticsSnapshot() {
    return this._getDiagnosticsSnapshot();
  }

  getBootTelemetry() {
    return this._getBootTelemetry();
  }

  getFastBootPlan() {
    return this._getFastBootPlan();
  }

  getPhaseTelemetry() {
    return this._getPhaseTelemetry();
  }

  reclaimSoftMemory(reason) {
    return this._reclaimSoftMemory(reason);
  }

  registerReclaimHook(hook) {
    return this._registerReclaimHook(hook);
  }

  attachAudioRing(mod, opts) {
    return this._attachAudioRing(mod, opts);
  }

  unlockAudio(opts) {
    return this._unlockAudio(opts);
  }

  getAudioStatus() {
    return this._getAudioStatus();
  }

  requestPointerLock(opts) {
    return this._requestPointerLock(opts);
  }

  exitPointerLock() {
    return this._exitPointerLock();
  }

  togglePointerLock(opts) {
    return this._togglePointerLock(opts);
  }

  isPointerLocked() {
    return this._isPointerLocked();
  }

  notifyFirstPresent(meta) {
    return this._notifyFirstPresent(meta);
  }

  notifyFirstVisibleFrame(meta) {
    return this._notifyFirstVisibleFrame(meta);
  }

  recordFastBootMilestone(milestone, meta) {
    return this._recordFastBootMilestone(milestone, meta);
  }

  start(opts) {
    return this._start(opts);
  }

  stop() {
    return this._stop();
  }

  resize(width, height, dpr) {
    return this._resize(width, height, dpr);
  }

  destroy() {
    return this._destroy();
  }
}

/**
 * Bootstraps all core browser client subsystems for playgta6web and wires them into
 * a single testable `PlayGta6AppHandle`.
 *
 * @param {object} [options]
 * @param {Window|object|null} [options.windowObj]
 * @param {Document|object|null} [options.documentObj]
 * @param {Navigator|object|null} [options.navigatorObj]
 * @param {typeof fetch|null} [options.fetchImpl]
 * @param {HTMLCanvasElement|OffscreenCanvas|object|null} [options.canvas]
 * @param {object|null} [options.hud]
 * @param {Function|null} [options.webgpuAdapterFactory]
 * @param {Function|null} [options.createWasmWorker]
 * @param {string} [options.statusUrl='/api/status']
 * @param {string} [options.loaderUrl]
 * @param {string} [options.dataBase]
 * @param {string} [options.bootBase]
 * @param {string} [options.gameBase]
 * @param {string} [options.shaderPackUrl]
 * @param {string} [options.programManifestUrl]
 * @param {object|null} [options.gpuBootPanel]
 * @param {Element|object|null} [options.gpuBootPanelRoot]
 * @param {Function|null} [options.GpuBootPanelCtor]
 * @param {object} [options.gpuBootPanelOptions]
 * @param {boolean} [options.enableGpuBootPanel=true]
 * @param {string} [options.appVersion]
 * @param {string} [options.activeProfileId]
 * @param {boolean} [options.attachGlobalErrorHandlers=true]
 * @param {boolean} [options.autoResize=true]
 * @param {number} [options.hudIntervalMs]
 * @param {() => number} [options.now]
 * @param {() => number} [options.epochNow]
 * @param {(cb: (ts: number) => void) => any} [options.raf]
 * @param {(handle: any) => void} [options.cancelRaf]
 * @param {(fn: () => void, ms: number) => any} [options.setInterval]
 * @param {(id: any) => void} [options.clearInterval]
 * @param {Function|null} [options.capabilityProbe]
 * @param {object} [options.crashReporterOptions]
 * @param {object} [options.perfMonitorOptions]
 * @param {object} [options.inputProfileOptions]
 * @param {object} [options.inputBridgeOptions]
 * @param {object} [options.storageOptions]
 * @param {object} [options.audioOptions]
 * @param {object|null} [options.audioEngine]
 * @param {boolean} [options.holdUntilVisible]
 * @param {boolean} [options.defaultHoldUntilVisible=false]
 * @param {object} [options.smokeDemoOptions]
 * @returns {PlayGta6AppHandle}
 */
export function bootstrapPlayGta6App(options = {}) {
  if (options !== undefined && (options === null || typeof options !== 'object')) {
    throw new TypeError('bootstrapPlayGta6App options must be an object');
  }

  const hasExplicitWindow = options.windowObj !== undefined;
  const windowObj = resolveDefaultWindow(options);
  const documentObj = resolveDefaultDocument(options, windowObj);
  const navigatorObj = resolveDefaultNavigator(options, windowObj);
  const fetchImpl = resolveDefaultFetch(options, windowObj);

  let canvas =
    options.canvas !== undefined
      ? options.canvas
      : documentObj && typeof documentObj.getElementById === 'function'
        ? documentObj.getElementById('viewport')
        : null;

  const hud =
    options.hud !== undefined
      ? options.hud
      : windowObj && windowObj.playgta6Hud
        ? windowObj.playgta6Hud
        : null;

  const webgpuAdapterFactory =
    options.webgpuAdapterFactory !== undefined ? options.webgpuAdapterFactory : null;
  const createWasmWorker =
    options.createWasmWorker !== undefined ? options.createWasmWorker : null;

  if (webgpuAdapterFactory !== null && typeof webgpuAdapterFactory !== 'function') {
    throw new TypeError('webgpuAdapterFactory must be a function or null');
  }
  if (createWasmWorker !== null && typeof createWasmWorker !== 'function') {
    throw new TypeError('createWasmWorker must be a function or null');
  }

  const now = typeof options.now === 'function' ? options.now : defaultNow;
  const epochNow =
    typeof options.epochNow === 'function'
      ? options.epochNow
      : typeof options.now === 'function'
        ? options.now
        : defaultEpochNow;

  const loaderUrl =
    typeof options.loaderUrl === 'string' && options.loaderUrl.length > 0
      ? options.loaderUrl
      : DEFAULT_LOADER_URL;
  const statusUrl =
    typeof options.statusUrl === 'string' && options.statusUrl.length > 0
      ? options.statusUrl
      : DEFAULT_STATUS_URL;
  const dataBase = resolveDefaultDataBase(options, windowObj);
  const explicitBootBase =
    typeof options.bootBase === 'string' && options.bootBase.length > 0 ? options.bootBase : null;
  const defaultBootBase = resolveDefaultBootBase(options, windowObj);
  const explicitGameBase =
    typeof options.gameBase === 'string' && options.gameBase.length > 0 ? options.gameBase : null;
  const defaultGameBase = resolveDefaultGameBase(options, windowObj);
  const explicitShaderPackUrl = pickBootUrl(
    options.shaderPackUrl,
    options.preferredShaderPackUrl,
    options.shaderPack,
    options.wgslPack
  );
  let startOverrideShaderPackUrl = null;
  const initialBootAssets = extractBootAssetUrls(options);
  let latestDiscoveredBootAssets = { ...initialBootAssets };
  let lastNotifiedSideModuleKey = null;
  let session = null;

  const rawUrlSearch =
    options.urlSearch !== undefined
      ? options.urlSearch
      : windowObj && windowObj.location
        ? typeof windowObj.location.search === 'string'
          ? windowObj.location.search
          : typeof windowObj.location.href === 'string'
            ? windowObj.location.href
            : ''
        : '';
  const parsedUrlOptions = parseBootUrlOptions(rawUrlSearch);
  const parsedSplashHold = parseSplashHoldQuery(rawUrlSearch);
  const initialPerfResolution = resolveBootstrapPerfProfile({
    options,
    windowObj,
    navigatorObj,
    urlSearch: rawUrlSearch,
  });
  let activePerfProfile = initialPerfResolution.perfProfile;
  let activeDisplayBounds = initialPerfResolution.displayBounds;
  let lastSentDisplayBoundsKey = displayBoundsKey(activeDisplayBounds);
  const externalIoWorker =
    options.ioWorker && typeof options.ioWorker === 'object' ? options.ioWorker : null;
  const externalGpuWorker =
    options.gpuWorker && typeof options.gpuWorker === 'object' ? options.gpuWorker : null;
  // #1077: `?opfsL2=0` bypasses the io_worker OPFS L2 hot-file cache (#1238). Off by default.
  const opfsL2Disabled = (() => {
    try {
      const qs = String(rawUrlSearch || '');
      const q = qs.indexOf('?');
      const params = new URLSearchParams(q >= 0 ? qs.slice(q + 1) : qs);
      return parseBooleanQueryFlag(getQueryParamInsensitive(params, 'opfsL2')) === false;
    } catch {
      return false;
    }
  })();
  const splashOverlayOptions =
    options.splashOverlayOptions && typeof options.splashOverlayOptions === 'object'
      ? options.splashOverlayOptions
      : {};
  let holdUntilVisibleExplicit = false;
  const resolvedHoldUntilVisible = (() => {
    if (
      Object.prototype.hasOwnProperty.call(options, 'holdUntilVisible') &&
      options.holdUntilVisible !== undefined &&
      options.holdUntilVisible !== null
    ) {
      holdUntilVisibleExplicit = true;
      return resolveBooleanOverride(options.holdUntilVisible, false);
    }
    if (
      Object.prototype.hasOwnProperty.call(splashOverlayOptions, 'holdUntilVisible') &&
      splashOverlayOptions.holdUntilVisible !== undefined &&
      splashOverlayOptions.holdUntilVisible !== null
    ) {
      holdUntilVisibleExplicit = true;
      return resolveBooleanOverride(splashOverlayOptions.holdUntilVisible, false);
    }
    if (parsedSplashHold !== null) {
      holdUntilVisibleExplicit = true;
      return parsedSplashHold;
    }
    if (
      Object.prototype.hasOwnProperty.call(options, 'defaultHoldUntilVisible') &&
      options.defaultHoldUntilVisible !== undefined &&
      options.defaultHoldUntilVisible !== null
    ) {
      return resolveBooleanOverride(options.defaultHoldUntilVisible, false);
    }
    return false;
  })();
  const baseInputScript = resolveInputScriptOverride(options, parsedUrlOptions.inputScript);
  const baseStreamConfig = resolveStreamConfigOverride(options, parsedUrlOptions);
  const baseDumpConfig = resolveDumpConfigOverride(options, parsedUrlOptions);
  const resolvedBootUrlOptions = Object.freeze({
    inputScript: baseInputScript,
    streamShaders: baseStreamConfig.streamShaders,
    streamThresholdMb: baseStreamConfig.streamThresholdMb,
    streamThresholdBytes: baseStreamConfig.streamThresholdBytes,
    dumpPipelines: baseDumpConfig.dumpPipelines,
    allowDumpWrites: baseDumpConfig.allowDumpWrites,
    dumpEndpoint: baseDumpConfig.dumpEndpoint,
    splashHold: parsedSplashHold,
    holdUntilVisible: resolvedHoldUntilVisible,
  });
  let activeStreamConfig = baseStreamConfig;
  const urlStreamExplicitDisabled = isStreamShadersExplicitlyDisabledInSearch(rawUrlSearch);
  const baseStreamExplicitDisabled =
    options.streamShaders === false || (options.streamShaders !== true && urlStreamExplicitDisabled);
  let activeStreamExplicitDisabled = baseStreamExplicitDisabled;
  let activeDumpConfig = baseDumpConfig;
  let configuredInputScript = baseInputScript;
  let activeInputScript = null;

  const gpuTelemetry = {
    realPackDrawCount: 0,
    fallbackDrawCount: 0,
    presentCount: 0,
    unboundBindGroupCount: 0,
    skippedDrawCount: 0,
    lastValidationErrorCode: null,
    lastValidationErrorMessage: null,
    gpuAdapter: null,
    hasVisibleContent: false,
    firstVisibleFrameMs: null,
    blackLoadingMs: null,
    loadingAssets: false,
    swapchainProbeCount: 0,
    swapchainProbeInFlight: false,
    swapchainProbeSupported: null,
    swapchainProbeErrorCount: 0,
    swapchainProbeTimeoutCount: 0,
    lastSampledMaxLuma: 0,
    suppressedBlackFrames: 0,
    coalescedPresents: 0,
    oversizedWgslFallbacks: 0,
    deviceLostCount: 0,
    deviceRecoveredCount: 0,
    gpuStallWarnCount: 0,
    rangedPackActive: null,
    rangedShaderReady: 0,
    rangedShaderFetches: 0,
    rangedShaderErrors: 0,
    asyncShaderPendingCount: 0,
    gpuMemory: null,
  };

  const audioTelemetry = {
    hasReceivedStats: false,
    outputState: 'unknown',
    framesDrained: 0,
    underruns: 0,
    overruns: 0,
  };

  const audioRingTelemetry = {
    audioRingAttached: false,
    audioRingAttachAttempts: 0,
    audioRingLastReason: null,
  };

  const loaderTelemetry = {
    indirectDispatch: normalizeIndirectDispatch(
      options.indirectDispatch ?? options.dispatchMode ?? null
    ),
    pthreadSpawnedCount: toSafeNonNegativeInt(options.pthreadSpawnedCount, 0),
    pthreadActiveCount: toSafeNonNegativeInt(options.pthreadActiveCount, 0),
    pthreadPeakCount: toSafeNonNegativeInt(options.pthreadPeakCount, 0),
    pthreadLoadQueueDepth: toSafeNonNegativeInt(options.pthreadLoadQueueDepth, 0),
    preferredShaderPackUrl: sanitizeTelemetryUrl(
      pickBootUrl(options.preferredShaderPackUrl, options.shaderPackUrl)
    ),
  };

  const perfTelemetry = {
    fps30ComplianceRate: null,
    dominantBottleneck: null,
    frameJitterMs: null,
    vramBudgetUtilization: null,
    bindGroupHitRate: null,
    jsHeapUsedMb: null,
    jsHeapTotalMb: null,
    mailboxLanes: null,
  };

  const fastBootMilestones = {
    START_SCREEN: null,
    INTRO_VIDEO: null,
    MAIN_MENU: null,
    IN_GAME: null,
  };

  let latestFastBootPlan = null;
  let latestQuotaEviction = null;
  let latestIoWorkerStats = null;
  let latestRuntimeDiagnostics = null;
  let lastJsHeapSampleAtMs = -Infinity;
  let jsHeapNotifiedToHud = false;
  let lastFramePresentedForwardAtMs = -Infinity;
  let lastFramePresentedForwardedDraws = -1;
  let lastFramePresentedForwardedErrors = -1;
  let lastFramePresentedForwardedVisible = false;
  const bootStartedAtMs = now();

  function isHudHidden() {
    const ctrl =
      options.hudController ||
      (windowObj && typeof windowObj === 'object' ? windowObj.playgta6HudController : null);
    if (ctrl && typeof ctrl === 'object') {
      if (typeof ctrl.isVisible === 'function') {
        try {
          return ctrl.isVisible() === false;
        } catch {
          // Ignore optional visibility checks.
        }
      }
      if (typeof ctrl.visible === 'boolean') {
        return ctrl.visible === false;
      }
    }
    if (hud && typeof hud === 'object') {
      if (typeof hud.isVisible === 'function') {
        try {
          return hud.isVisible() === false;
        } catch {
          // Ignore optional visibility checks.
        }
      }
      if (typeof hud.visible === 'boolean') {
        return hud.visible === false;
      }
    }
    return false;
  }

  function readRawPerformanceMemory() {
    const perfCandidates = [
      options.performanceObj,
      windowObj && typeof windowObj === 'object' ? windowObj.performance : null,
      typeof performance !== 'undefined' ? performance : null,
    ];
    for (let i = 0; i < perfCandidates.length; i += 1) {
      const perf = perfCandidates[i];
      if (!perf || typeof perf !== 'object') {
        continue;
      }
      try {
        const mem = perf.memory;
        if (mem && typeof mem === 'object') {
          const used = sanitizeNonNegativeFiniteNumber(mem.usedJSHeapSize);
          const total = sanitizeNonNegativeFiniteNumber(mem.totalJSHeapSize);
          const limit = sanitizeNonNegativeFiniteNumber(mem.jsHeapSizeLimit);
          if (used !== null || total !== null || limit !== null) {
            return { usedJSHeapSize: used, totalJSHeapSize: total, jsHeapSizeLimit: limit };
          }
        }
      } catch {
        // Ignore restricted performance.memory getters.
      }
    }
    return null;
  }

  function sampleJsHeapMemory() {
    const raw = readRawPerformanceMemory();
    if (!raw) {
      return null;
    }
    const jsHeapUsedMb =
      raw.usedJSHeapSize !== null
        ? Math.round((raw.usedJSHeapSize / BYTES_PER_MIB) * 100) / 100
        : null;
    const jsHeapTotalMb =
      raw.totalJSHeapSize !== null
        ? Math.round((raw.totalJSHeapSize / BYTES_PER_MIB) * 100) / 100
        : null;
    if (jsHeapUsedMb === null && jsHeapTotalMb === null) {
      return null;
    }
    return { jsHeapUsedMb, jsHeapTotalMb };
  }

  function collectPlainSources(candidates) {
    const out = [];
    for (let i = 0; i < candidates.length; i += 1) {
      const item = candidates[i];
      if (item && typeof item === 'object' && !Array.isArray(item)) {
        out.push(item);
      }
    }
    return out;
  }

  function pickFirstDefinedField(sources, keys) {
    for (let i = 0; i < sources.length; i += 1) {
      const src = sources[i];
      for (let j = 0; j < keys.length; j += 1) {
        const key = keys[j];
        if (Object.prototype.hasOwnProperty.call(src, key) && src[key] !== undefined) {
          return { found: true, value: src[key] };
        }
      }
    }
    return { found: false, value: undefined };
  }

  function pickFirstBooleanField(sources, keys) {
    for (let i = 0; i < sources.length; i += 1) {
      const src = sources[i];
      for (let j = 0; j < keys.length; j += 1) {
        const key = keys[j];
        if (Object.prototype.hasOwnProperty.call(src, key) && typeof src[key] === 'boolean') {
          return { found: true, value: src[key] };
        }
      }
    }
    return { found: false, value: undefined };
  }

  function resolveSanitizedPreferredShaderPackUrl() {
    const candidates = [
      startOverrideShaderPackUrl,
      explicitShaderPackUrl,
      loaderTelemetry.preferredShaderPackUrl,
      latestDiscoveredBootAssets.shaderPackUrl,
      DEFAULT_SHADER_PACK_URL,
    ];
    for (let i = 0; i < candidates.length; i += 1) {
      const clean = sanitizeTelemetryUrl(candidates[i]);
      if (clean) {
        return clean;
      }
    }
    return DEFAULT_SHADER_PACK_URL;
  }

  function invokeFirstHook(target, methodNames, ...args) {
    if (!target || typeof target !== 'object') {
      return false;
    }
    for (let i = 0; i < methodNames.length; i += 1) {
      const fn = target[methodNames[i]];
      if (typeof fn === 'function') {
        try {
          fn.apply(target, args);
        } catch {
          // Optional observer hooks must never disrupt bootstrap or runtime execution.
        }
        return true;
      }
    }
    return false;
  }

  function resolveHudTargets() {
    const targets = [];
    const pushUnique = (candidate) => {
      if (candidate && typeof candidate === 'object' && !targets.includes(candidate)) {
        targets.push(candidate);
      }
    };
    pushUnique(hud);
    pushUnique(options.hudController);
    if (windowObj && typeof windowObj === 'object') {
      pushUnique(windowObj.playgta6HudController);
    }
    return targets;
  }

  function notifySideModulePreflight(assets) {
    if (destroyed || !assets || typeof assets !== 'object') {
      return;
    }
    const hasUrls = Array.isArray(assets.sideModuleUrls);
    const hasCount =
      typeof assets.sideModuleCount === 'number' && Number.isFinite(assets.sideModuleCount);
    if (!hasUrls && !hasCount) {
      return;
    }
    const urls = hasUrls ? assets.sideModuleUrls.slice() : [];
    const count = hasCount ? Math.max(0, Math.floor(assets.sideModuleCount)) : urls.length;
    const signature = `${count}|${urls.join(',')}`;
    if (signature === lastNotifiedSideModuleKey) {
      return;
    }
    lastNotifiedSideModuleKey = signature;

    const payload = Object.freeze({
      sideModuleCount: count,
      sideModuleUrls: Object.freeze(urls.slice()),
    });

    if (session) {
      invokeFirstHook(
        session,
        [
          'setSideModules',
          'updateSideModules',
          'onSideModulesDiscovered',
          'recordSideModules',
          'setSplitModuleInfo',
          'updateSplitModuleStatus',
        ],
        payload
      );
    }

    const hudTargets = resolveHudTargets();
    for (let i = 0; i < hudTargets.length; i += 1) {
      const target = hudTargets[i];
      const handled = invokeFirstHook(
        target,
        [
          'setSideModules',
          'updateSideModules',
          'onSideModulesDiscovered',
          'recordSideModules',
          'setSplitModuleInfo',
          'updateSplitModuleStatus',
        ],
        payload
      );
      if (!handled && target === hud && typeof target.updateMetrics === 'function') {
        try {
          target.updateMetrics({ sideModuleCount: count, sideModuleUrls: urls.slice() });
        } catch {
          // Ignore HUD metrics errors.
        }
      }
    }
  }

  function recordDiscoveredBootAssets(payload) {
    if (!payload || typeof payload !== 'object') {
      return;
    }
    const fromDirect = extractBootAssetUrls(payload);
    const fromAssets = extractBootAssetUrls(payload.assets);
    const fromStatus = extractBootAssetUrls(payload.status || payload.serverStatus);
    const merged = { ...fromDirect, ...fromAssets, ...fromStatus };
    if (Object.keys(merged).length > 0) {
      latestDiscoveredBootAssets = { ...latestDiscoveredBootAssets, ...merged };
      if (merged.sideModuleUrls !== undefined || merged.sideModuleCount !== undefined) {
        notifySideModulePreflight(latestDiscoveredBootAssets);
      }
    }
    const preferredFromPayload = sanitizeTelemetryUrl(
      pickBootUrl(
        payload.preferredShaderPackUrl,
        payload.status && payload.status.preferredShaderPackUrl,
        payload.serverStatus && payload.serverStatus.preferredShaderPackUrl,
        payload.assets && payload.assets.preferredShaderPackUrl,
        merged.shaderPackUrl
      )
    );
    if (preferredFromPayload) {
      loaderTelemetry.preferredShaderPackUrl = preferredFromPayload;
    }
    ingestLoaderTelemetry(payload, { notify: false, parseLabel: false });
  }

  function resolveBootForwardFields(launchContext) {
    let fromOrchestrator = {};
    if (bootOrchestrator && typeof bootOrchestrator.getState === 'function') {
      try {
        const state = bootOrchestrator.getState();
        if (state && typeof state === 'object') {
          fromOrchestrator = {
            ...extractBootAssetUrls(state.assets),
            ...extractBootAssetUrls(state.status),
          };
        }
      } catch {
        // Ignore orchestrator state errors.
      }
    }
    const fromLaunchStatus = extractBootAssetUrls(launchContext && launchContext.serverStatus);
    const merged = {
      ...initialBootAssets,
      ...latestDiscoveredBootAssets,
      ...fromOrchestrator,
      ...fromLaunchStatus,
    };
    const fields = {
      bootBase: explicitBootBase || merged.bootBase || defaultBootBase,
      gameBase: explicitGameBase || merged.gameBase || defaultGameBase,
      shaderPackUrl:
        startOverrideShaderPackUrl ||
        explicitShaderPackUrl ||
        merged.shaderPackUrl ||
        DEFAULT_SHADER_PACK_URL,
    };
    if (merged.reflectionManifestUrl) {
      fields.reflectionManifestUrl = merged.reflectionManifestUrl;
    }
    if (merged.pgimUrl) {
      fields.pgimUrl = merged.pgimUrl;
    }
    if (merged.programManifestUrl) {
      fields.programManifestUrl = merged.programManifestUrl;
    }
    if (Array.isArray(merged.sideModuleUrls)) {
      fields.sideModuleUrls = merged.sideModuleUrls.slice();
    }
    if (typeof merged.sideModuleCount === 'number' && Number.isFinite(merged.sideModuleCount)) {
      fields.sideModuleCount = Math.max(0, Math.floor(merged.sideModuleCount));
    } else if (Array.isArray(fields.sideModuleUrls)) {
      fields.sideModuleCount = fields.sideModuleUrls.length;
    }
    if (
      activeStreamConfig &&
      activeStreamConfig.streamShaders === true &&
      isPositiveFinite(activeStreamConfig.streamThresholdBytes)
    ) {
      fields.streamShaders = true;
      fields.streamThresholdBytes = activeStreamConfig.streamThresholdBytes;
    } else if (
      activeStreamExplicitDisabled &&
      urlStreamExplicitDisabled &&
      options.streamShaders !== false
    ) {
      fields.streamShaders = false;
    }
    if (activeDumpConfig && activeDumpConfig.dumpPipelines === true) {
      fields.dumpPipelines = true;
    }
    if (activeDumpConfig && activeDumpConfig.allowDumpWrites === true) {
      fields.allowDumpWrites = true;
      fields.dumpEndpoint = activeDumpConfig.dumpEndpoint || DEFAULT_DUMP_ENDPOINT;
    }
    if (latestFastBootPlan && latestFastBootPlan.recommendedSlabMaxBytes > 0) {
      fields.recommendedSlabMaxBytes = latestFastBootPlan.recommendedSlabMaxBytes;
    }
    const audioRingRetryOpts =
      options.audioRingRetry && typeof options.audioRingRetry === 'object'
        ? options.audioRingRetry
        : null;
    const resolvedAudioRingIntervalMs = sanitizePositiveInt(
      options.audioRingRetryIntervalMs !== undefined
        ? options.audioRingRetryIntervalMs
        : audioRingRetryOpts && audioRingRetryOpts.retryIntervalMs !== undefined
          ? audioRingRetryOpts.retryIntervalMs
          : undefined
    );
    const resolvedAudioRingMaxRetries = toNonNegativeInt(
      options.audioRingMaxRetries !== undefined
        ? options.audioRingMaxRetries
        : audioRingRetryOpts && audioRingRetryOpts.maxRetries !== undefined
          ? audioRingRetryOpts.maxRetries
          : undefined,
      null
    );
    const hasExplicitAudioRingAutoRetry =
      typeof options.audioRingAutoRetry === 'boolean' ||
      Boolean(audioRingRetryOpts && typeof audioRingRetryOpts.autoRetry === 'boolean');
    const resolvedAudioRingAutoRetry =
      typeof options.audioRingAutoRetry === 'boolean'
        ? options.audioRingAutoRetry
        : audioRingRetryOpts && typeof audioRingRetryOpts.autoRetry === 'boolean'
          ? audioRingRetryOpts.autoRetry
          : undefined;
    const hasExplicitAudioRingRetryOnNoExport =
      typeof options.audioRingRetryOnNoExport === 'boolean' ||
      Boolean(audioRingRetryOpts && typeof audioRingRetryOpts.retryOnNoExport === 'boolean');
    const resolvedAudioRingRetryOnNoExport =
      typeof options.audioRingRetryOnNoExport === 'boolean'
        ? options.audioRingRetryOnNoExport
        : audioRingRetryOpts && typeof audioRingRetryOpts.retryOnNoExport === 'boolean'
          ? audioRingRetryOpts.retryOnNoExport
          : undefined;
    if (resolvedAudioRingIntervalMs !== null) {
      fields.audioRingRetryIntervalMs = resolvedAudioRingIntervalMs;
    }
    if (resolvedAudioRingMaxRetries !== null) {
      fields.audioRingMaxRetries = resolvedAudioRingMaxRetries;
    }
    if (hasExplicitAudioRingAutoRetry) {
      fields.audioRingAutoRetry = resolvedAudioRingAutoRetry;
    }
    if (hasExplicitAudioRingRetryOnNoExport) {
      fields.audioRingRetryOnNoExport = resolvedAudioRingRetryOnNoExport;
    }
    if (typeof options.audioRingAutoPump === 'boolean') {
      fields.audioRingAutoPump = options.audioRingAutoPump;
    }
    if (opfsL2Disabled) {
      fields.opfsL2Enabled = false;
    }
    if (typeof options.requireNonBlackFrame === 'boolean') {
      fields.requireNonBlackFrame = options.requireNonBlackFrame;
    } else if (holdUntilVisibleExplicit && resolvedHoldUntilVisible === false) {
      fields.requireNonBlackFrame = false;
    }
    if (activePerfProfile) {
      fields.perfProfile = activePerfProfile;
    }
    if (activeDisplayBounds) {
      fields.displayBounds = activeDisplayBounds;
    }
    return fields;
  }

  // 1. CrashReporter (sanitized diagnostics + global window error handlers).
  const crashReporterOptions =
    options.crashReporterOptions && typeof options.crashReporterOptions === 'object'
      ? options.crashReporterOptions
      : {};
  const crashReporterGlobalObj =
    crashReporterOptions.globalObj !== undefined
      ? crashReporterOptions.globalObj
      : windowObj || (typeof globalThis !== 'undefined' ? globalThis : undefined);
  const crashReporter = new CrashReporter({
    ...crashReporterOptions,
    ...(options.appVersion !== undefined ? { appVersion: options.appVersion } : {}),
    now: crashReporterOptions.now || epochNow,
    navigatorObj:
      crashReporterOptions.navigatorObj !== undefined
        ? crashReporterOptions.navigatorObj
        : navigatorObj || undefined,
    globalObj: crashReporterGlobalObj,
  });

  const crashReporterCleanupFns = [];
  if (typeof crashReporter.collectCapabilities === 'function') {
    const hasOwnCollect = Object.prototype.hasOwnProperty.call(
      crashReporter,
      'collectCapabilities'
    );
    const originalCollectCapabilities = crashReporter.collectCapabilities;
    const wrappedCollectCapabilities = function collectCapabilitiesWithWasm64() {
      const caps = originalCollectCapabilities.apply(this, arguments);
      if (!caps || typeof caps !== 'object') {
        return caps;
      }
      const wasmApi =
        crashReporterGlobalObj &&
        typeof crashReporterGlobalObj === 'object' &&
        'WebAssembly' in crashReporterGlobalObj
          ? crashReporterGlobalObj.WebAssembly
          : typeof WebAssembly !== 'undefined'
            ? WebAssembly
            : null;
      return {
        ...caps,
        wasm64: detectWasm64(wasmApi),
      };
    };
    crashReporter.collectCapabilities = wrappedCollectCapabilities;
    crashReporterCleanupFns.push(() => {
      if (crashReporter.collectCapabilities === wrappedCollectCapabilities) {
        if (hasOwnCollect) {
          crashReporter.collectCapabilities = originalCollectCapabilities;
        } else {
          delete crashReporter.collectCapabilities;
        }
      }
    });
  }

  let syncCrashReporterDiagnosticsRef = null;
  let syncingCrashReport = false;
  if (typeof crashReporter.buildReport === 'function') {
    const hasOwnBuildReport = Object.prototype.hasOwnProperty.call(crashReporter, 'buildReport');
    const originalBuildReport = crashReporter.buildReport;
    const wrappedBuildReport = function buildReportWithLiveTelemetry(...args) {
      if (!syncingCrashReport && typeof syncCrashReporterDiagnosticsRef === 'function') {
        syncingCrashReport = true;
        try {
          syncCrashReporterDiagnosticsRef();
        } catch {
          // Live diagnostic sync must never prevent crash report export.
        } finally {
          syncingCrashReport = false;
        }
      }
      return originalBuildReport.apply(this, args);
    };
    crashReporter.buildReport = wrappedBuildReport;
    crashReporterCleanupFns.push(() => {
      if (crashReporter.buildReport === wrappedBuildReport) {
        if (hasOwnBuildReport) {
          crashReporter.buildReport = originalBuildReport;
        } else {
          delete crashReporter.buildReport;
        }
      }
    });
  }

  if (options.attachGlobalErrorHandlers !== false && isEventTargetLike(windowObj)) {
    crashReporter.attachGlobalHandlers(windowObj);
  }

  // Adapter matching RuntimeSessionController._reportError(err, context) -> CrashReporter.recordError(category, err, context).
  const sessionCrashAdapter = (err, context) => {
    const preferredCategory =
      context && typeof context === 'object' && typeof context.category === 'string'
        ? context.category
        : null;
    crashReporter.recordError(preferredCategory, err, context);
  };

  // 2. PerfMonitor (rolling frame/GPU/IO/audio telemetry).
  const perfMonitorOptions =
    options.perfMonitorOptions && typeof options.perfMonitorOptions === 'object'
      ? options.perfMonitorOptions
      : {};
  const resolvedSetInterval =
    options.setInterval ||
    perfMonitorOptions.setInterval ||
    (windowObj && typeof windowObj.setInterval === 'function'
      ? windowObj.setInterval.bind(windowObj)
      : undefined);
  const resolvedClearInterval =
    options.clearInterval ||
    perfMonitorOptions.clearInterval ||
    (windowObj && typeof windowObj.clearInterval === 'function'
      ? windowObj.clearInterval.bind(windowObj)
      : undefined);

  const perfMonitor =
    options.perfMonitor && typeof options.perfMonitor === 'object'
      ? options.perfMonitor
      : new PerfMonitor({
          ...perfMonitorOptions,
          now: perfMonitorOptions.now || now,
          ...(resolvedSetInterval ? { setInterval: resolvedSetInterval } : {}),
          ...(resolvedClearInterval ? { clearInterval: resolvedClearInterval } : {}),
        });

  // 3. InputProfileManager (load persisted profile safely) + InputBridge.
  const inputProfileOptions =
    options.inputProfileOptions && typeof options.inputProfileOptions === 'object'
      ? options.inputProfileOptions
      : {};
  const profileStorage = resolveDefaultStorage(options, windowObj, hasExplicitWindow);
  const inputProfiles = new InputProfileManager({
    ...inputProfileOptions,
    ...(profileStorage !== undefined ? { storage: profileStorage } : {}),
    autoLoad: false,
  });

  try {
    inputProfiles.load();
  } catch (err) {
    crashReporter.recordError('io_error', err, {
      source: 'inputProfiles',
      phase: 'load',
    });
    if (hud && typeof hud.appendLog === 'function') {
      try {
        hud.appendLog(
          `Failed to load saved input profile; using defaults: ${err?.message || err}`,
          'warn'
        );
      } catch {
        // Ignore HUD errors.
      }
    }
  }

  if (typeof options.activeProfileId === 'string' && options.activeProfileId.length > 0) {
    try {
      inputProfiles.setActiveProfile(options.activeProfileId);
    } catch (err) {
      crashReporter.recordError('runtime', err, {
        source: 'inputProfiles',
        phase: 'setActiveProfile',
        profileId: options.activeProfileId,
      });
    }
  }

  const inputBridgeOptions =
    options.inputBridgeOptions && typeof options.inputBridgeOptions === 'object'
      ? options.inputBridgeOptions
      : {};
  const userPointerLockCb =
    typeof inputBridgeOptions.onPointerLockChange === 'function'
      ? inputBridgeOptions.onPointerLockChange
      : null;

  const mergedInputBridgeOptions = inputProfiles.toInputBridgeOptions(undefined, {
    ...inputBridgeOptions,
    windowObj,
    documentObj,
    navigatorObj,
    onPointerLockChange: (locked) => {
      if (hud && typeof hud.updateMetrics === 'function') {
        try {
          hud.updateMetrics({ pointerLock: locked });
        } catch {
          // Ignore HUD errors.
        }
      }
      if (userPointerLockCb) {
        userPointerLockCb(locked);
      }
    },
  });
  const inputBridge = new InputBridge(canvas, mergedInputBridgeOptions);

  // 4. BrowserStorageManager (quota, OPFS cache, IndexedDB /userdata).
  const storageOptions =
    options.storageOptions && typeof options.storageOptions === 'object'
      ? options.storageOptions
      : {};
  const storageObj =
    storageOptions.storageObj !== undefined
      ? storageOptions.storageObj
      : navigatorObj && navigatorObj.storage
        ? navigatorObj.storage
        : null;
  const indexedDBFactory =
    storageOptions.indexedDBFactory !== undefined
      ? storageOptions.indexedDBFactory
      : storageOptions.indexedDB !== undefined
        ? storageOptions.indexedDB
        : windowObj && 'indexedDB' in windowObj
          ? windowObj.indexedDB
          : undefined;

  const storageManager =
    options.storageManager && typeof options.storageManager === 'object'
      ? options.storageManager
      : new BrowserStorageManager({
          ...storageOptions,
          storageObj,
          ...(indexedDBFactory !== undefined ? { indexedDBFactory } : {}),
          now: storageOptions.now || epochNow,
        });

  // 5. AudioEngine (WebAudio + AudioWorklet + SAB ring).
  const audioOptions =
    options.audioOptions && typeof options.audioOptions === 'object' ? options.audioOptions : {};
  const userAudioOnError =
    typeof audioOptions.onError === 'function' ? audioOptions.onError : null;
  const resolvedAudioContextCtor =
    audioOptions.AudioContextCtor !== undefined
      ? audioOptions.AudioContextCtor
      : hasExplicitWindow
        ? (windowObj && (windowObj.AudioContext || windowObj.webkitAudioContext)) || null
        : undefined;
  const resolvedAudioWorkletNodeCtor =
    audioOptions.AudioWorkletNodeCtor !== undefined
      ? audioOptions.AudioWorkletNodeCtor
      : hasExplicitWindow
        ? (windowObj && windowObj.AudioWorkletNode) || null
        : undefined;
  const effectiveAudioContextCtor =
    resolvedAudioContextCtor !== undefined
      ? resolvedAudioContextCtor
      : (windowObj && (windowObj.AudioContext || windowObj.webkitAudioContext)) ||
        (typeof globalThis !== 'undefined' &&
          (globalThis.AudioContext || globalThis.webkitAudioContext)) ||
        null;
  const effectiveAudioWorkletNodeCtor =
    resolvedAudioWorkletNodeCtor !== undefined
      ? resolvedAudioWorkletNodeCtor
      : (windowObj && windowObj.AudioWorkletNode) ||
        (typeof globalThis !== 'undefined' && globalThis.AudioWorkletNode) ||
        null;
  const hasWebAudioSupport =
    typeof effectiveAudioContextCtor === 'function' &&
    typeof effectiveAudioWorkletNodeCtor === 'function';

  const userAudioOnStateChange =
    typeof audioOptions.onStateChange === 'function' ? audioOptions.onStateChange : null;
  let notifyRuntimeAudioContextState = () => {};

  const hasExplicitAudioEngine =
    options.audioEngine !== undefined && options.audioEngine !== null;
  const audioEngine = hasExplicitAudioEngine
    ? options.audioEngine
    : new AudioEngine({
        ...audioOptions,
        ...(audioOptions.audioWorker === undefined &&
        audioOptions.worker === undefined &&
        audioOptions.bridge === undefined &&
        options.audioWorker
          ? { audioWorker: options.audioWorker }
          : {}),
        ...(resolvedAudioContextCtor !== undefined
          ? { AudioContextCtor: resolvedAudioContextCtor }
          : {}),
        ...(resolvedAudioWorkletNodeCtor !== undefined
          ? { AudioWorkletNodeCtor: resolvedAudioWorkletNodeCtor }
          : {}),
        onStateChange: (nextState) => {
          if (nextState === 'running') {
            notifyRuntimeAudioContextState('active');
          } else if (nextState === 'suspended' || nextState === 'interrupted') {
            notifyRuntimeAudioContextState('suspended');
          }
          if (userAudioOnStateChange) {
            userAudioOnStateChange(nextState);
          }
        },
        onError: (err) => {
          crashReporter.recordError('runtime', err, { source: 'audioEngine' });
          if (userAudioOnError) {
            userAudioOnError(err);
          }
        },
      });

  // 5b. BootTracer (W40), optional SplashOverlay (W42), PgimPreloader (W41) & BootOrchestrator (W44).
  const bootTracerOptions =
    options.bootTracerOptions && typeof options.bootTracerOptions === 'object'
      ? options.bootTracerOptions
      : {};
  const bootTracer =
    options.bootTracer !== undefined
      ? options.bootTracer
      : new BootTracer({
          ...bootTracerOptions,
          crashReporter,
          hud,
          now: bootTracerOptions.now || now,
        });

  const globalObj =
    windowObj && typeof windowObj === 'object'
      ? windowObj
      : typeof globalThis !== 'undefined'
        ? globalThis
        : {};

  const splashContainer =
    options.splashContainer !== undefined ? options.splashContainer : null;
  const SplashOverlayCtor =
    options.SplashOverlayCtor ||
    (globalObj.PlayGta6SplashOverlay && globalObj.PlayGta6SplashOverlay.SplashOverlayController) ||
    (typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6SplashOverlay &&
      globalThis.PlayGta6SplashOverlay.SplashOverlayController) ||
    null;

  let hasExplicitSplashFadeOutMs = false;
  const resolvedSplashFadeOutMs = (() => {
    const candidates = [
      options.splashFadeOutMs,
      options.fadeOutMs,
      splashOverlayOptions.fadeOutMs,
    ];
    for (let i = 0; i < candidates.length; i += 1) {
      const value = candidates[i];
      if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
        hasExplicitSplashFadeOutMs = true;
        return value;
      }
    }
    return DEFAULT_SPLASH_FADE_OUT_MS;
  })();
  const hasExplicitVisibleFrameFadeMs =
    typeof splashOverlayOptions.visibleFrameFadeMs === 'number' &&
    Number.isFinite(splashOverlayOptions.visibleFrameFadeMs) &&
    splashOverlayOptions.visibleFrameFadeMs >= 0;

  let splashOverlay = null;
  let ownsSplashOverlay = false;
  if (options.splashOverlay !== undefined) {
    splashOverlay = options.splashOverlay;
  } else if (
    options.enableSplashOverlay !== false &&
    splashContainer &&
    typeof SplashOverlayCtor === 'function'
  ) {
    splashOverlay = new SplashOverlayCtor({
      ...splashOverlayOptions,
      fadeOutMs: resolvedSplashFadeOutMs,
      ...(!hasExplicitVisibleFrameFadeMs && !hasExplicitSplashFadeOutMs
        ? { visibleFrameFadeMs: 450 }
        : {}),
      nowFn: splashOverlayOptions.nowFn || now,
      holdUntilVisible: resolvedHoldUntilVisible,
      ...(splashOverlayOptions.locationSearch === undefined &&
      splashOverlayOptions.search === undefined &&
      rawUrlSearch
        ? { locationSearch: rawUrlSearch }
        : {}),
      parent: splashContainer,
      documentObj: documentObj || undefined,
      windowObj: windowObj || undefined,
    });
    ownsSplashOverlay = true;
  }

  const gpuBootPanelRoot =
    options.gpuBootPanelRoot !== undefined ? options.gpuBootPanelRoot : null;
  const GpuBootPanelCtor =
    options.GpuBootPanelCtor ||
    (globalObj.PlayGta6GpuBootPanel && globalObj.PlayGta6GpuBootPanel.GpuBootPanelController) ||
    (typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6GpuBootPanel &&
      globalThis.PlayGta6GpuBootPanel.GpuBootPanelController) ||
    null;

  let gpuBootPanel = null;
  let ownsGpuBootPanel = false;
  if (options.gpuBootPanel !== undefined) {
    gpuBootPanel = options.gpuBootPanel;
  } else if (
    options.enableGpuBootPanel !== false &&
    gpuBootPanelRoot &&
    typeof GpuBootPanelCtor === 'function'
  ) {
    const gpuBootPanelOptions =
      options.gpuBootPanelOptions && typeof options.gpuBootPanelOptions === 'object'
        ? options.gpuBootPanelOptions
        : {};
    gpuBootPanel = new GpuBootPanelCtor({
      ...gpuBootPanelOptions,
      container: gpuBootPanelRoot,
      documentObj: documentObj || undefined,
      now: gpuBootPanelOptions.now || epochNow,
    });
    ownsGpuBootPanel = true;
  }

  let latestPanelBootSummary = null;
  let latestPanelFrameStats = undefined;
  let latestPanelFallbackStats = undefined;
  let latestPanelMissingShaders = undefined;
  const latestPanelPerfFields = Object.create(null);
  const bootTracerCleanupFns = [];

  function safeUpdateGpuBootPanel(patch = {}) {
    if (destroyed) {
      return;
    }
    if (patch.bootSummary !== undefined) {
      latestPanelBootSummary = patch.bootSummary;
    }
    if (patch.frameStats !== undefined) {
      latestPanelFrameStats = patch.frameStats;
    }
    if (patch.fallbackStats !== undefined) {
      latestPanelFallbackStats = patch.fallbackStats;
    }
    if (patch.missingShaders !== undefined) {
      latestPanelMissingShaders = patch.missingShaders;
    }
    for (let i = 0; i < PANEL_PERF_TELEMETRY_KEYS.length; i += 1) {
      const key = PANEL_PERF_TELEMETRY_KEYS[i];
      if (patch[key] !== undefined) {
        latestPanelPerfFields[key] = patch[key];
      } else if (perfTelemetry[key] !== null) {
        latestPanelPerfFields[key] = perfTelemetry[key];
      }
    }

    if (!gpuBootPanel || typeof gpuBootPanel.update !== 'function') {
      return;
    }

    const payload = {};
    if (latestPanelBootSummary !== null && latestPanelBootSummary !== undefined) {
      payload.bootSummary = latestPanelBootSummary;
    } else if (bootTracer && typeof bootTracer.getSummary === 'function') {
      try {
        payload.bootSummary = bootTracer.getSummary();
      } catch {
        // Ignore tracer summary errors.
      }
    }
    if (latestPanelFrameStats !== undefined) {
      payload.frameStats = latestPanelFrameStats;
    }
    if (latestPanelFallbackStats !== undefined) {
      payload.fallbackStats = latestPanelFallbackStats;
    }
    if (latestPanelMissingShaders !== undefined) {
      payload.missingShaders = latestPanelMissingShaders;
    }
    for (let i = 0; i < PANEL_PERF_TELEMETRY_KEYS.length; i += 1) {
      const key = PANEL_PERF_TELEMETRY_KEYS[i];
      if (latestPanelPerfFields[key] !== undefined && latestPanelPerfFields[key] !== null) {
        payload[key] = latestPanelPerfFields[key];
      }
    }

    try {
      gpuBootPanel.update(payload);
    } catch {
      // Diagnostics panel errors must never interrupt boot or runtime.
    }
  }

  const maxValidationErrors =
    Number.isInteger(options.maxValidationErrors) && options.maxValidationErrors >= 1
      ? options.maxValidationErrors
      : DEFAULT_MAX_VALIDATION_ERRORS;
  let validationErrorSeq = 0;
  let validationErrorCount = 0;
  const recentValidationErrors = [];
  const recentValidationEventKeys = new Set();
  const MAX_VALIDATION_EVENT_DEDUPE_KEYS = 64;
  let wgpuFramesPresented = 0;
  let splashFramesPresented = 0;
  let wasmPerfLastFrames = 0;
  let wasmPerfLastDrawCalls = 0;
  let wasmPerfLastPackets = 0;
  let wasmPerfLastComputeDispatches = 0;
  let wasmPerfLastFrameNowMs = null;

  function resetWasmPerfTracking() {
    wasmPerfLastFrames = 0;
    wasmPerfLastDrawCalls = 0;
    wasmPerfLastPackets = 0;
    wasmPerfLastComputeDispatches = 0;
    wasmPerfLastFrameNowMs = null;
    wgpuFramesPresented = 0;
    gpuTelemetry.presentCount = 0;
    gpuTelemetry.gpuMemory = null;
    if (perfMonitor && typeof perfMonitor.reset === 'function') {
      try {
        perfMonitor.reset();
      } catch {
        // PerfMonitor reset errors must never block worker startup.
      }
    }
  }

  function recordRealWasmPerfUpdate(
    sourceData,
    { framesPresented = null, detailedFrameRecorded = false } = {}
  ) {
    if (destroyed || !perfMonitor) {
      return false;
    }
    const sources =
      sourceData && typeof sourceData === 'object'
        ? collectPlainSources([
            sourceData,
            sourceData.frameStats,
            sourceData.stats,
            sourceData.gpuStats,
            sourceData.drawStats,
            sourceData.renderStateStats,
            sourceData.swapchainStats,
            sourceData.swapchain,
          ])
        : [];

    const nowMs = now();

    // 1. Record monotonic GPU batch deltas before committing frames so drawCallsPerFrame stays accurate.
    const drawCallsField = pickFirstDefinedField(sources, [
      'drawCalls',
      'totalDrawCalls',
      'draws',
    ]);
    let cumulativeDrawCalls = drawCallsField.found
      ? toNonNegativeInt(drawCallsField.value, null)
      : null;
    if (cumulativeDrawCalls === null) {
      const realPackField = pickFirstDefinedField(sources, [
        'realPackDrawCount',
        'real_pack_draw_count',
        'realPackDraws',
      ]);
      const fallbackField = pickFirstDefinedField(sources, [
        'fallbackDrawCount',
        'fallbackDraws',
      ]);
      if (realPackField.found || fallbackField.found) {
        cumulativeDrawCalls =
          (realPackField.found ? toSafeNonNegativeInt(realPackField.value, 0) : 0) +
          (fallbackField.found ? toSafeNonNegativeInt(fallbackField.value, 0) : 0);
      }
    }

    const packetsField = pickFirstDefinedField(sources, [
      'packetsProcessed',
      'packetCount',
      'totalGpuPackets',
      'gpuPackets',
      'packets',
    ]);
    const cumulativePackets = packetsField.found
      ? toNonNegativeInt(packetsField.value, null)
      : null;

    const computeField = pickFirstDefinedField(sources, [
      'computeDispatches',
      'totalComputeDispatches',
    ]);
    const cumulativeCompute = computeField.found
      ? toNonNegativeInt(computeField.value, null)
      : null;

    let deltaDraws = 0;
    if (cumulativeDrawCalls !== null && cumulativeDrawCalls > wasmPerfLastDrawCalls) {
      deltaDraws = cumulativeDrawCalls - wasmPerfLastDrawCalls;
      wasmPerfLastDrawCalls = cumulativeDrawCalls;
    }
    let deltaPackets = 0;
    if (cumulativePackets !== null && cumulativePackets > wasmPerfLastPackets) {
      deltaPackets = cumulativePackets - wasmPerfLastPackets;
      wasmPerfLastPackets = cumulativePackets;
    }
    let deltaCompute = 0;
    if (cumulativeCompute !== null && cumulativeCompute > wasmPerfLastComputeDispatches) {
      deltaCompute = cumulativeCompute - wasmPerfLastComputeDispatches;
      wasmPerfLastComputeDispatches = cumulativeCompute;
    }

    let recorded = false;
    if (
      (deltaDraws > 0 || deltaPackets > 0 || deltaCompute > 0) &&
      typeof perfMonitor.recordGpuBatch === 'function'
    ) {
      try {
        perfMonitor.recordGpuBatch({
          packets: deltaPackets,
          drawCalls: deltaDraws,
          computeDispatches: deltaCompute,
          timestampMs: nowMs,
        });
        recorded = true;
      } catch {
        // PerfMonitor errors must never interrupt worker telemetry.
      }
    }

    // 2. Record monotonic presented frame deltas (skipping if recordDetailedFrame already logged this frame).
    let resolvedFrames = toNonNegativeInt(framesPresented, null);
    if (resolvedFrames === null && sources.length > 0) {
      const framesField = pickFirstDefinedField(sources, [
        'framesPresented',
        'presentCount',
        'frameCount',
        'frame',
      ]);
      if (framesField.found) {
        resolvedFrames = toNonNegativeInt(framesField.value, null);
      }
    }

    if (resolvedFrames !== null && resolvedFrames > wasmPerfLastFrames) {
      const deltaFrames = resolvedFrames - wasmPerfLastFrames;
      wasmPerfLastFrames = resolvedFrames;
      if (!detailedFrameRecorded && typeof perfMonitor.recordFrame === 'function') {
        try {
          const steps = Math.min(deltaFrames, MAX_SYNTHETIC_FRAME_DELTA_STEPS);
          if (steps === 1) {
            perfMonitor.recordFrame(nowMs);
          } else if (
            wasmPerfLastFrameNowMs !== null &&
            Number.isFinite(nowMs) &&
            nowMs > wasmPerfLastFrameNowMs
          ) {
            const perFrameMs = (nowMs - wasmPerfLastFrameNowMs) / deltaFrames;
            if (typeof perfMonitor.recordFrameTimeMs === 'function') {
              for (let i = 0; i < steps; i += 1) {
                perfMonitor.recordFrameTimeMs(perFrameMs);
              }
            } else {
              for (let i = 1; i <= steps; i += 1) {
                perfMonitor.recordFrame(wasmPerfLastFrameNowMs + perFrameMs * i);
              }
            }
          } else {
            for (let i = 0; i < steps; i += 1) {
              perfMonitor.recordFrame(nowMs);
            }
          }
          recorded = true;
        } catch {
          // PerfMonitor errors must never interrupt worker telemetry.
        }
      }
      if (Number.isFinite(nowMs)) {
        wasmPerfLastFrameNowMs = nowMs;
      }
    }

    return recorded;
  }

  function buildCrashReporterGpuTelemetryPayload() {
    const effectivePresentCount = Math.max(gpuTelemetry.presentCount, resolveFramesPresented());
    const { gpuAdapter, firstVisibleFrameMs, blackLoadingMs } = resolveTracerVisibleTiming();
    let perfSnap = null;
    if (perfMonitor && typeof perfMonitor.getSnapshot === 'function') {
      try {
        perfSnap = perfMonitor.getSnapshot();
      } catch {
        perfSnap = null;
      }
    }

    const totalDrawCalls =
      (latestPanelFrameStats && toNonNegativeInt(latestPanelFrameStats.drawCalls, null)) ??
      (perfSnap && perfSnap.gpu ? toNonNegativeInt(perfSnap.gpu.totalDrawCalls, null) : null) ??
      gpuTelemetry.realPackDrawCount + gpuTelemetry.fallbackDrawCount;
    const totalPackets =
      (latestPanelFrameStats &&
        toNonNegativeInt(
          latestPanelFrameStats.packetsProcessed ?? latestPanelFrameStats.packetCount,
          null
        )) ??
      (perfSnap && perfSnap.gpu ? toNonNegativeInt(perfSnap.gpu.totalGpuPackets, null) : null);
    const totalComputeDispatches =
      (latestPanelFrameStats && toNonNegativeInt(latestPanelFrameStats.computeDispatches, null)) ??
      (perfSnap && perfSnap.gpu
        ? toNonNegativeInt(perfSnap.gpu.totalComputeDispatches, null)
        : null);

    const payload = {
      backend: 'webgpu',
      framesPresented: effectivePresentCount,
      presentCount: effectivePresentCount,
      splashFramesPresented,
      realPackDrawCount: toSafeNonNegativeInt(gpuTelemetry.realPackDrawCount, 0),
      fallbackDrawCount: toSafeNonNegativeInt(gpuTelemetry.fallbackDrawCount, 0),
      drawCalls: toSafeNonNegativeInt(totalDrawCalls, 0),
      unboundBindGroupCount: toSafeNonNegativeInt(gpuTelemetry.unboundBindGroupCount, 0),
      skippedDrawCount: toSafeNonNegativeInt(gpuTelemetry.skippedDrawCount, 0),
      validationErrorCount: toSafeNonNegativeInt(validationErrorCount, 0),
      lastValidationErrorCode: gpuTelemetry.lastValidationErrorCode,
      lastValidationErrorMessage: gpuTelemetry.lastValidationErrorMessage,
      gpuAdapter,
      firstVisibleFrameMs,
      blackLoadingMs,
    };

    if (totalPackets !== null) {
      payload.packetsProcessed = totalPackets;
      payload.packetCount = totalPackets;
    }
    if (totalComputeDispatches !== null) {
      payload.computeDispatches = totalComputeDispatches;
    }
    if (perfSnap && perfSnap.frames && perfSnap.frames.totalFrames > 0) {
      if (typeof perfSnap.frames.fps === 'number' && Number.isFinite(perfSnap.frames.fps)) {
        payload.fps = perfSnap.frames.fps;
        payload.presentFpsAvg = perfSnap.frames.fps;
      }
      if (
        typeof perfSnap.frames.onePercentLowFps === 'number' &&
        Number.isFinite(perfSnap.frames.onePercentLowFps)
      ) {
        payload.onePercentLowFps = perfSnap.frames.onePercentLowFps;
      }
      if (
        typeof perfSnap.frames.p95FrameTimeMs === 'number' &&
        Number.isFinite(perfSnap.frames.p95FrameTimeMs)
      ) {
        payload.p95FrameMs = perfSnap.frames.p95FrameTimeMs;
      }
      if (
        typeof perfSnap.frames.p99FrameTimeMs === 'number' &&
        Number.isFinite(perfSnap.frames.p99FrameTimeMs)
      ) {
        payload.p99FrameMs = perfSnap.frames.p99FrameTimeMs;
      }
      if (
        typeof perfSnap.frames.maxFrameTimeMs === 'number' &&
        Number.isFinite(perfSnap.frames.maxFrameTimeMs)
      ) {
        payload.maxFrameMs = perfSnap.frames.maxFrameTimeMs;
      }
    }
    if (latestPanelFrameStats && typeof latestPanelFrameStats === 'object') {
      payload.frameStats = {
        ...latestPanelFrameStats,
        framesPresented: Math.max(
          effectivePresentCount,
          toSafeNonNegativeInt(latestPanelFrameStats.framesPresented, 0)
        ),
      };
    } else if (effectivePresentCount > 0) {
      payload.frameStats = {
        framesPresented: effectivePresentCount,
        droppedFrames: 0,
        drawCalls: toSafeNonNegativeInt(totalDrawCalls, 0),
        realPackDrawCount: toSafeNonNegativeInt(gpuTelemetry.realPackDrawCount, 0),
        fallbackDrawCount: toSafeNonNegativeInt(gpuTelemetry.fallbackDrawCount, 0),
      };
    }
    if (latestPanelFallbackStats && typeof latestPanelFallbackStats === 'object') {
      payload.fallbackStats = latestPanelFallbackStats;
    }
    if (Array.isArray(latestPanelMissingShaders)) {
      payload.missingShaders = latestPanelMissingShaders;
    }
    return payload;
  }

  function syncCrashReporterDiagnostics() {
    if (destroyed || !crashReporter || typeof crashReporter !== 'object') {
      return;
    }
    const { gpuAdapter, firstVisibleFrameMs, blackLoadingMs } = resolveTracerVisibleTiming();
    if (gpuAdapter !== null && typeof crashReporter.setGpuAdapter === 'function') {
      try {
        crashReporter.setGpuAdapter(gpuAdapter);
      } catch {
        // Telemetry errors must never disrupt boot or export.
      }
    }

    const currentBootMode = resolveCurrentBootMode();
    if (
      bootTracer &&
      typeof bootTracer.getSummary === 'function' &&
      typeof crashReporter.setBootSnapshot === 'function'
    ) {
      try {
        const summary = bootTracer.getSummary();
        if (
          summary &&
          typeof summary === 'object' &&
          (summary.milestonesReached > 0 ||
            summary.trapCount > 0 ||
            summary.gpuAdapter !== null ||
            summary.firstVisibleFrameMs !== null ||
            firstVisibleFrameMs !== null ||
            currentBootMode !== 'idle')
        ) {
          crashReporter.setBootSnapshot({
            ...summary,
            bootMode: currentBootMode,
            ...(gpuAdapter !== null && !summary.gpuAdapter ? { gpuAdapter } : {}),
            ...(firstVisibleFrameMs !== null && summary.firstVisibleFrameMs === null
              ? { firstVisibleFrameMs }
              : {}),
            ...(blackLoadingMs !== null && summary.blackLoadingMs === null
              ? { blackLoadingMs }
              : {}),
          });
        }
      } catch {
        // Ignore BootTracer summary errors.
      }
    }

    const effectivePresentCount = Math.max(gpuTelemetry.presentCount, resolveFramesPresented());
    const hasLiveGpuActivity =
      effectivePresentCount > 0 ||
      gpuAdapter !== null ||
      gpuTelemetry.hasVisibleContent ||
      firstVisibleFrameMs !== null ||
      blackLoadingMs !== null ||
      validationErrorCount > 0 ||
      latestPanelFrameStats !== undefined ||
      latestPanelFallbackStats !== undefined ||
      latestPanelMissingShaders !== undefined;
    if (hasLiveGpuActivity && typeof crashReporter.setGpuTelemetry === 'function') {
      try {
        crashReporter.setGpuTelemetry(buildCrashReporterGpuTelemetryPayload());
      } catch {
        // Ignore CrashReporter GPU telemetry errors.
      }
    }

    if (
      perfMonitor &&
      typeof perfMonitor.getSnapshot === 'function' &&
      typeof crashReporter.setFrameBudget === 'function'
    ) {
      try {
        const perfSnap = perfMonitor.getSnapshot();
        if (
          perfSnap &&
          ((perfSnap.frames && perfSnap.frames.totalFrames > 0) ||
            perfTelemetry.fps30ComplianceRate !== null)
        ) {
          crashReporter.setFrameBudget(perfSnap);
        }
      } catch {
        // Ignore PerfMonitor snapshot errors.
      }
    }

    if (typeof crashReporter.setBootPhase === 'function') {
      try {
        const phaseTel = getPhaseTelemetry();
        if (phaseTel && phaseTel.currentPhase) {
          crashReporter.setBootPhase(phaseTel.currentPhase);
        }
      } catch {
        // Ignore session phase errors.
      }
    }

    if (
      (audioTelemetry.hasReceivedStats ||
        audioRingTelemetry.audioRingAttached ||
        audioRingTelemetry.audioRingAttachAttempts > 0) &&
      typeof crashReporter.setAudioTelemetry === 'function'
    ) {
      try {
        crashReporter.setAudioTelemetry({
          outputState: audioTelemetry.hasReceivedStats
            ? audioTelemetry.outputState
            : normalizeAudioOutputState(audioEngine && audioEngine.state),
          guestRingAttached: Boolean(audioRingTelemetry.audioRingAttached),
          guestFramesDrained: toSafeNonNegativeInt(audioTelemetry.framesDrained, 0),
          guestUnderruns: toSafeNonNegativeInt(audioTelemetry.underruns, 0),
          guestOverruns: toSafeNonNegativeInt(audioTelemetry.overruns, 0),
        });
      } catch {
        // Ignore audio telemetry sync errors.
      }
    }

    if (
      latestIoWorkerStats &&
      typeof crashReporter.setIoWorkerStats === 'function'
    ) {
      try {
        crashReporter.setIoWorkerStats(latestIoWorkerStats);
      } catch {
        // Ignore IO worker stats sync errors.
      }
    }

    if (
      latestRuntimeDiagnostics &&
      typeof crashReporter.setRuntimeDiagnostics === 'function'
    ) {
      try {
        crashReporter.setRuntimeDiagnostics(latestRuntimeDiagnostics);
      } catch {
        // Ignore runtime diagnostics sync errors.
      }
    }
  }
  syncCrashReporterDiagnosticsRef = syncCrashReporterDiagnostics;

  function recordValidationError(rawData, opts = {}) {
    if (destroyed) {
      return null;
    }
    const data = rawData && typeof rawData === 'object' ? rawData : { message: rawData };
    const inner =
      (data.record && typeof data.record === 'object' ? data.record : null) ||
      (data.error && typeof data.error === 'object' ? data.error : null);

    const rawType =
      data.errorType ??
      data.kind ??
      (inner && (inner.errorType ?? inner.type ?? inner.name)) ??
      (typeof data.name === 'string' ? data.name : null);
    const errorType = classifyGpuValidationErrorType(rawType, 'validation');

    const rawMessage =
      data.message !== undefined && data.message !== null
        ? data.message
        : inner && inner.message !== undefined && inner.message !== null
          ? inner.message
          : typeof data.error === 'string'
            ? data.error
            : '';
    const message = sanitizeDiagnosticText(rawMessage, MAX_VALIDATION_ERROR_MESSAGE_LENGTH);

    const rawCode =
      data.code ??
      data.errorCode ??
      (inner && (inner.code ?? inner.errorCode)) ??
      null;
    const errorCode = sanitizeValidationErrorCode(rawCode, errorType);

    const rawSeq = data.workerSeq ?? data.seq ?? (inner ? inner.workerSeq ?? inner.seq : null);
    const parsedSeq = toNonNegativeInt(rawSeq, null);
    const rawTs = data.timestampMs ?? data.timestamp ?? (inner ? inner.timestampMs : null);
    const hasExplicitTs = typeof rawTs === 'number' && Number.isFinite(rawTs);
    const timestampMs = hasExplicitTs ? rawTs : epochNow();

    const rawSource =
      opts.source ?? data.source ?? (inner ? inner.source : null) ?? 'wgpu_worker';
    const source = sanitizeDiagnosticText(rawSource, 64) || 'wgpu_worker';

    // Deduplicate identical worker+loader / dual-channel validation error deliveries.
    const dedupeKey =
      parsedSeq !== null && parsedSeq > 0
        ? `seq:${source}:${parsedSeq}:${hasExplicitTs ? timestampMs : ''}:${errorType}:${message}`
        : hasExplicitTs && message.length > 0
          ? `msg:${source}:${timestampMs}:${errorType}:${message}`
          : null;
    if (dedupeKey !== null && recentValidationEventKeys.has(dedupeKey)) {
      return recentValidationErrors.length > 0
        ? recentValidationErrors[recentValidationErrors.length - 1]
        : null;
    }
    if (dedupeKey !== null) {
      recentValidationEventKeys.add(dedupeKey);
      if (recentValidationEventKeys.size > MAX_VALIDATION_EVENT_DEDUPE_KEYS) {
        const oldest = recentValidationEventKeys.values().next().value;
        if (oldest !== undefined) {
          recentValidationEventKeys.delete(oldest);
        }
      }
    }

    validationErrorSeq =
      parsedSeq !== null && parsedSeq > validationErrorSeq ? parsedSeq : validationErrorSeq + 1;
    const rawTotalCount = data.totalCount ?? (inner ? inner.totalCount : null);
    const parsedTotalCount = toNonNegativeInt(rawTotalCount, null);
    validationErrorCount =
      parsedTotalCount !== null
        ? Math.max(validationErrorCount, parsedTotalCount)
        : validationErrorCount + 1;
    gpuTelemetry.lastValidationErrorCode = errorCode;
    gpuTelemetry.lastValidationErrorMessage = message || null;

    const record = Object.freeze({
      seq: validationErrorSeq,
      ...(parsedSeq !== null && parsedSeq > 0 ? { workerSeq: parsedSeq } : {}),
      errorType,
      message,
      timestampMs,
      source,
    });

    recentValidationErrors.push(record);
    while (recentValidationErrors.length > maxValidationErrors) {
      recentValidationErrors.shift();
    }

    const eventPayload = Object.freeze({
      type: 'gpu-validation-error',
      seq: record.seq,
      ...(record.workerSeq !== undefined ? { workerSeq: record.workerSeq } : {}),
      errorType: record.errorType,
      message: record.message,
      timestampMs: record.timestampMs,
      source: record.source,
      validationErrorCount,
    });

    let sessionHandledViaRecordMethod = false;
    if (session) {
      const sessionMethodNames = [
        'onGpuValidationError',
        'recordGpuValidationError',
        'handleGpuValidationError',
        'handleWorkerMessage',
      ];
      for (let i = 0; i < sessionMethodNames.length; i += 1) {
        const name = sessionMethodNames[i];
        const fn = session[name];
        if (typeof fn === 'function') {
          if (name === 'recordGpuValidationError') {
            sessionHandledViaRecordMethod = true;
          }
          try {
            fn.call(session, eventPayload);
          } catch {
            // Optional observer hooks must never disrupt bootstrap or runtime execution.
          }
          break;
        }
      }
    }

    const sessionRecordedCrash = Boolean(
      sessionHandledViaRecordMethod &&
        session &&
        (session._crashReporter === crashReporter || session._crashReporter === sessionCrashAdapter)
    );
    if (opts.skipCrashReporter !== true && !sessionRecordedCrash) {
      const errorName =
        errorType === 'out-of-memory'
          ? 'GPUOutOfMemoryError'
          : errorType === 'internal'
            ? 'GPUInternalError'
            : 'GPUValidationError';
      crashReporter.recordError(
        'webgpu_validation',
        { name: errorName, message },
        { source, errorType, seq: record.seq, timestampMs }
      );
    }

    const sessionNotifiedHud = Boolean(
      sessionHandledViaRecordMethod &&
        session &&
        hud &&
        session._hud === hud &&
        (typeof hud.recordValidationError === 'function' ||
          typeof hud.onValidationError === 'function' ||
          typeof hud.updateGpuValidation === 'function')
    );

    const hudTargets = resolveHudTargets();
    for (let i = 0; i < hudTargets.length; i += 1) {
      const target = hudTargets[i];
      if (sessionNotifiedHud && target === hud) {
        continue;
      }
      const handled = invokeFirstHook(
        target,
        [
          'onGpuValidationError',
          'recordGpuValidationError',
          'handleGpuValidationError',
          'reportValidationError',
        ],
        eventPayload
      );
      if (
        !handled &&
        target === hud &&
        opts.skipHudLog !== true &&
        typeof target.appendLog === 'function'
      ) {
        try {
          target.appendLog(
            `[WebGPU ${record.errorType}] ${record.message || 'Validation error'}`,
            'error'
          );
        } catch {
          // Ignore HUD errors.
        }
      }
    }

    if (gpuBootPanel) {
      invokeFirstHook(
        gpuBootPanel,
        ['onGpuValidationError', 'recordGpuValidationError', 'handleGpuValidationError'],
        eventPayload
      );
    }

    return record;
  }

  function handleGpuValidationErrorMessage(data) {
    if (!data || typeof data !== 'object' || data.type !== 'gpu-validation-error') {
      return false;
    }
    recordValidationError(data, { source: data.source || 'wgpu_worker' });
    return true;
  }

  function isSplashOverlayHolding() {
    if (!splashOverlay || typeof splashOverlay !== 'object') {
      return false;
    }
    if (typeof splashOverlay.getState === 'function') {
      try {
        const state = splashOverlay.getState();
        if (state && typeof state === 'object') {
          return Boolean(state.holdingForVisibleFrame);
        }
      } catch {
        // Ignore overlay state errors.
      }
    }
    return Boolean(resolvedHoldUntilVisible && firstPresentNotified && !firstVisibleFrameNotified);
  }

  function recordGpuAdapter(rawAdapter) {
    if (destroyed) {
      return null;
    }
    const sanitized = extractSanitizedGpuAdapter(rawAdapter);
    if (!sanitized) {
      return null;
    }
    const changed = gpuTelemetry.gpuAdapter !== sanitized;
    gpuTelemetry.gpuAdapter = sanitized;

    if (changed) {
      if (crashReporter && typeof crashReporter.setGpuAdapter === 'function') {
        try {
          crashReporter.setGpuAdapter(sanitized);
        } catch {
          // Telemetry errors must never disrupt boot.
        }
      }
      if (bootTracer && typeof bootTracer.recordGpuAdapter === 'function') {
        try {
          bootTracer.recordGpuAdapter(sanitized);
        } catch {
          // Telemetry errors must never disrupt boot.
        }
      }

      const hudTargets = resolveHudTargets();
      for (let i = 0; i < hudTargets.length; i += 1) {
        const target = hudTargets[i];
        const handled = invokeFirstHook(
          target,
          ['setGpuAdapter', 'updateGpuAdapter', 'recordGpuAdapter', 'onGpuAdapter'],
          sanitized
        );
        if (!handled && target === hud && typeof target.updateMetrics === 'function') {
          try {
            target.updateMetrics({ gpuAdapter: sanitized });
          } catch {
            // Ignore HUD metrics errors.
          }
        }
      }

      if (gpuBootPanel) {
        invokeFirstHook(
          gpuBootPanel,
          ['setGpuAdapter', 'updateGpuAdapter', 'recordGpuAdapter'],
          sanitized
        );
        safeUpdateGpuBootPanel();
      }
    }

    return sanitized;
  }

  function ingestGpuAdapterFromPayload(data) {
    if (!data || typeof data !== 'object') {
      return null;
    }
    const sources = collectPlainSources([
      data,
      data.stats,
      data.gpuStats,
      data.drawStats,
      data.frameStats,
      data.swapchainStats,
      data.swapchain,
      data.metrics,
    ]);
    const adapterField = pickFirstDefinedField(sources, [
      'gpuAdapter',
      'gpu_adapter',
      'adapterLabel',
      'adapter_label',
      'adapterInfo',
      'adapter_info',
      'adapter',
    ]);
    if (!adapterField.found || adapterField.value === null) {
      return null;
    }
    return recordGpuAdapter(adapterField.value);
  }

  function detectVisibleFrameSignal(data) {
    const sources = collectPlainSources([
      data,
      data && data.stats,
      data && data.gpuStats,
      data && data.drawStats,
      data && data.frameStats,
      data && data.swapchainStats,
      data && data.swapchain,
    ]);
    // W349: check explicit boolean non-black pixel probe flags first (ignoring null/non-boolean
    // placeholders). When the worker explicitly reports `firstVisibleFrame === false` (or
    // `hasVisibleContent === false`), NEVER fall back to firing FIRST_VISIBLE_FRAME on
    // `realPackDrawCount > 0` or `hasNonClearDraw`.
    const explicitVisible = pickFirstBooleanField(sources, [
      'firstVisibleFrame',
      'first_visible_frame',
      'hasVisibleContent',
      'has_visible_content',
    ]);
    if (explicitVisible.found) {
      if (explicitVisible.value === false) {
        sawExplicitVisibleFrameGate = true;
        return {
          hasExplicitFlag: true,
          isVisible: false,
        };
      }
      return {
        hasExplicitFlag: true,
        isVisible: true,
      };
    }
    const visiblePresents = pickFirstDefinedField(sources, [
      'visiblePresentsTotal',
      'visible_presents_total',
    ]);
    if (visiblePresents.found) {
      const visibleCount = toSafeNonNegativeInt(visiblePresents.value, null);
      if (visibleCount !== null) {
        if (visibleCount > 0) {
          return {
            hasExplicitFlag: true,
            isVisible: true,
          };
        }
        sawExplicitVisibleFrameGate = true;
        return {
          hasExplicitFlag: true,
          isVisible: false,
        };
      }
    }
    // W349: once the worker has signaled active non-black pixel probing (`firstVisibleFrame: false`),
    // do not let subsequent partial telemetry payloads (without an explicit boolean flag) fall back
    // to `realPackDrawCount > 0` before an explicit `first-visible-frame` / `firstVisibleFrame: true`.
    if (sawExplicitVisibleFrameGate) {
      return {
        hasExplicitFlag: true,
        isVisible: false,
      };
    }
    const realPackField = pickFirstDefinedField(sources, [
      'realPackDrawCount',
      'real_pack_draw_count',
      'realPackDraws',
      'packDrawCount',
      'packDraws',
    ]);
    if (realPackField.found && toSafeNonNegativeInt(realPackField.value, 0) > 0) {
      return {
        hasExplicitFlag: false,
        isVisible: true,
      };
    }
    return {
      hasExplicitFlag: false,
      isVisible: false,
    };
  }

  function updateSplashLoadingAssetsProgress(extra = {}) {
    if (destroyed || !splashOverlay || typeof splashOverlay.setLoadingAssetsProgress !== 'function') {
      return false;
    }
    if (!isSplashOverlayHolding()) {
      return false;
    }
    const effectivePresentCount = Math.max(gpuTelemetry.presentCount, resolveFramesPresented());
    const payload = {
      presents:
        typeof extra.presents === 'number' && Number.isFinite(extra.presents) && extra.presents >= 0
          ? Math.floor(extra.presents)
          : effectivePresentCount,
      realPackDraws:
        typeof extra.realPackDraws === 'number' &&
        Number.isFinite(extra.realPackDraws) &&
        extra.realPackDraws >= 0
          ? Math.floor(extra.realPackDraws)
          : gpuTelemetry.realPackDrawCount,
    };
    if (typeof extra.message === 'string' && extra.message.trim().length > 0) {
      payload.message = extra.message.trim();
    }
    try {
      splashOverlay.setLoadingAssetsProgress(payload);
      return true;
    } catch {
      // Overlay errors must never interrupt telemetry.
      return false;
    }
  }

  function syncVisibleFrameFromTelemetry(data, { source = 'gpu-stats', deferObserverForward = true } = {}) {
    if (destroyed) {
      return false;
    }
    const signal = detectVisibleFrameSignal(data);
    if (signal.isVisible) {
      const effectivePresentCount = Math.max(gpuTelemetry.presentCount, resolveFramesPresented());
      const sources = collectPlainSources([
        data,
        data && data.stats,
        data && data.gpuStats,
        data && data.frameStats,
        data && data.swapchainStats,
        data && data.swapchain,
      ]);
      const blackField = pickFirstDefinedField(sources, ['blackLoadingMs', 'black_loading_ms']);
      const visibleMsField = pickFirstDefinedField(sources, [
        'firstVisibleFrameMs',
        'first_visible_frame_ms',
      ]);
      notifyFirstVisibleFrame({
        source,
        presents: effectivePresentCount,
        realPackDraws: gpuTelemetry.realPackDrawCount,
        ...(blackField.found &&
        typeof blackField.value === 'number' &&
        Number.isFinite(blackField.value) &&
        blackField.value >= 0
          ? { blackLoadingMs: blackField.value }
          : {}),
        ...(visibleMsField.found &&
        typeof visibleMsField.value === 'number' &&
        Number.isFinite(visibleMsField.value) &&
        visibleMsField.value >= 0
          ? { firstVisibleFrameMs: visibleMsField.value }
          : {}),
        _deferObserverForward: deferObserverForward,
      });
      return true;
    }
    if (isSplashOverlayHolding()) {
      gpuTelemetry.loadingAssets = true;
      updateSplashLoadingAssetsProgress();
    }
    return false;
  }

  function handleSplashFramePresentedMessage(data) {
    if (!data || typeof data !== 'object' || data.type !== 'splash-frame-presented') {
      return false;
    }
    if (destroyed) {
      return false;
    }

    ingestGpuAdapterFromPayload(data);

    const ok = data.ok !== false;
    const frameIndex =
      Number.isInteger(data.frameIndex) && data.frameIndex >= 0 ? data.frameIndex : null;
    const reportedFrames = toNonNegativeInt(data.framesPresented, null);
    const firstPresent = Boolean(data.firstPresent);
    const error = data.error ? sanitizeDiagnosticText(data.error) : null;
    const reason = data.reason ? sanitizeDiagnosticText(data.reason, 128) : null;
    const rawTs = data.timestampMs ?? data.timestamp;
    const timestampMs =
      typeof rawTs === 'number' && Number.isFinite(rawTs) ? rawTs : epochNow();

    const realPackField = pickFirstDefinedField([data], [
      'realPackDrawCount',
      'real_pack_draw_count',
      'realPackDraws',
    ]);
    if (realPackField.found) {
      gpuTelemetry.realPackDrawCount = Math.max(
        gpuTelemetry.realPackDrawCount,
        toSafeNonNegativeInt(realPackField.value, 0)
      );
    }

    if (reportedFrames !== null) {
      wgpuFramesPresented = Math.max(wgpuFramesPresented, reportedFrames);
    } else if (ok) {
      wgpuFramesPresented += 1;
    }
    if (ok) {
      splashFramesPresented += 1;
    }
    const effectiveFramesPresented = Math.max(wgpuFramesPresented, reportedFrames ?? 0);
    gpuTelemetry.presentCount = Math.max(gpuTelemetry.presentCount, effectiveFramesPresented);
    if (ok) {
      recordRealWasmPerfUpdate(data, { framesPresented: effectiveFramesPresented });
    }

    if (ok && (firstPresent || effectiveFramesPresented >= 1)) {
      notifyFirstPresent({
        source: 'splash-frame-presented',
        frameIndex,
        framesPresented: effectiveFramesPresented,
      });
    } else if (!ok && error) {
      crashReporter.recordLog('warn', 'wgpu_worker', `splash-frame-presented failed: ${error}`);
    }

    if (ok) {
      syncVisibleFrameFromTelemetry(data, {
        source: 'splash-frame-presented',
        deferObserverForward: true,
      });
    }

    if (ok && gpuBootPanel) {
      safeUpdateGpuBootPanel({
        frameStats: {
          ...(latestPanelFrameStats && typeof latestPanelFrameStats === 'object'
            ? latestPanelFrameStats
            : {}),
          framesPresented: effectiveFramesPresented,
          droppedFrames:
            latestPanelFrameStats && Number.isFinite(latestPanelFrameStats.droppedFrames)
              ? latestPanelFrameStats.droppedFrames
              : 0,
        },
      });
    }

    const loadingAssets = isSplashOverlayHolding();
    const eventPayload = Object.freeze({
      type: 'splash-frame-presented',
      ok,
      frameIndex,
      framesPresented: effectiveFramesPresented,
      splashFramesPresented,
      firstPresent,
      loadingAssets,
      ...(gpuTelemetry.gpuAdapter !== null ? { gpuAdapter: gpuTelemetry.gpuAdapter } : {}),
      ...(gpuTelemetry.firstVisibleFrameMs !== null
        ? { firstVisibleFrameMs: gpuTelemetry.firstVisibleFrameMs }
        : {}),
      ...(gpuTelemetry.blackLoadingMs !== null
        ? { blackLoadingMs: gpuTelemetry.blackLoadingMs }
        : {}),
      ...(error ? { error } : {}),
      ...(reason ? { reason } : {}),
      timestampMs,
    });

    if (session) {
      invokeFirstHook(
        session,
        [
          'onSplashFramePresented',
          'recordSplashFramePresented',
          'handleSplashFramePresented',
          'handleWorkerMessage',
        ],
        eventPayload
      );
    }

    const hudTargets = resolveHudTargets();
    for (let i = 0; i < hudTargets.length; i += 1) {
      invokeFirstHook(
        hudTargets[i],
        [
          'onSplashFramePresented',
          'recordSplashFramePresented',
          'handleSplashFramePresented',
        ],
        eventPayload
      );
    }

    if (gpuBootPanel) {
      invokeFirstHook(
        gpuBootPanel,
        ['onSplashFramePresented', 'handleSplashFramePresented'],
        eventPayload
      );
    }

    return true;
  }

  function buildGpuStatsPayload() {
    const effectivePresentCount = Math.max(gpuTelemetry.presentCount, resolveFramesPresented());
    const loadingAssets = isSplashOverlayHolding();
    const payload = {
      type: 'gpu-stats',
      realPackDrawCount: gpuTelemetry.realPackDrawCount,
      fallbackDrawCount: gpuTelemetry.fallbackDrawCount,
      presentCount: effectivePresentCount,
      unboundBindGroupCount: gpuTelemetry.unboundBindGroupCount,
      skippedDrawCount: gpuTelemetry.skippedDrawCount,
      validationErrorCount,
      lastValidationErrorCode: gpuTelemetry.lastValidationErrorCode,
      lastValidationErrorMessage: gpuTelemetry.lastValidationErrorMessage,
      hasVisibleContent: gpuTelemetry.hasVisibleContent,
      loadingAssets,
      swapchainProbeCount: gpuTelemetry.swapchainProbeCount,
      swapchainProbeInFlight: gpuTelemetry.swapchainProbeInFlight,
      swapchainProbeSupported: gpuTelemetry.swapchainProbeSupported,
      swapchainProbeErrorCount: gpuTelemetry.swapchainProbeErrorCount,
      swapchainProbeTimeoutCount: gpuTelemetry.swapchainProbeTimeoutCount,
      lastSampledMaxLuma: gpuTelemetry.lastSampledMaxLuma,
      suppressedBlackFrames: gpuTelemetry.suppressedBlackFrames,
      coalescedPresents: gpuTelemetry.coalescedPresents,
      oversizedWgslFallbacks: gpuTelemetry.oversizedWgslFallbacks,
      deviceLostCount: gpuTelemetry.deviceLostCount,
      deviceRecoveredCount: gpuTelemetry.deviceRecoveredCount,
      gpuStallWarnCount: gpuTelemetry.gpuStallWarnCount,
      rangedPackActive: gpuTelemetry.rangedPackActive,
      rangedShaderReady: gpuTelemetry.rangedShaderReady,
      rangedShaderFetches: gpuTelemetry.rangedShaderFetches,
      rangedShaderErrors: gpuTelemetry.rangedShaderErrors,
      asyncShaderPendingCount: gpuTelemetry.asyncShaderPendingCount,
    };
    if (gpuTelemetry.gpuAdapter !== null) {
      payload.gpuAdapter = gpuTelemetry.gpuAdapter;
    }
    if (gpuTelemetry.firstVisibleFrameMs !== null) {
      payload.firstVisibleFrameMs = gpuTelemetry.firstVisibleFrameMs;
    }
    if (gpuTelemetry.blackLoadingMs !== null) {
      payload.blackLoadingMs = gpuTelemetry.blackLoadingMs;
    }
    if (gpuTelemetry.gpuMemory !== null) {
      payload.gpuMemory = Object.freeze({ ...gpuTelemetry.gpuMemory });
    }
    return Object.freeze(payload);
  }

  function forwardGpuStatsToObservers(gpuStatsPayload, { allowHudMetricsFallback = true } = {}) {
    if (destroyed) {
      return;
    }
    if (session) {
      invokeFirstHook(
        session,
        ['updateGpuStats', 'setGpuStats', 'onGpuStats', 'recordGpuStats'],
        gpuStatsPayload
      );
    }

    const hudTargets = resolveHudTargets();
    for (let i = 0; i < hudTargets.length; i += 1) {
      const target = hudTargets[i];
      const handled = invokeFirstHook(
        target,
        ['updateGpuStats', 'setGpuStats', 'onGpuStats', 'recordGpuStats', 'updateStats'],
        gpuStatsPayload
      );
      if (
        !handled &&
        allowHudMetricsFallback &&
        target === hud &&
        typeof target.updateMetrics === 'function'
      ) {
        try {
          target.updateMetrics({
            gpuStats: gpuStatsPayload,
            realPackDrawCount: gpuStatsPayload.realPackDrawCount,
            fallbackDrawCount: gpuStatsPayload.fallbackDrawCount,
            presentCount: gpuStatsPayload.presentCount,
            unboundBindGroupCount: gpuStatsPayload.unboundBindGroupCount,
            skippedDrawCount: gpuStatsPayload.skippedDrawCount,
            validationErrorCount: gpuStatsPayload.validationErrorCount,
            ...(gpuStatsPayload.gpuAdapter !== undefined
              ? { gpuAdapter: gpuStatsPayload.gpuAdapter }
              : {}),
            ...(gpuStatsPayload.firstVisibleFrameMs !== undefined
              ? { firstVisibleFrameMs: gpuStatsPayload.firstVisibleFrameMs }
              : {}),
            ...(gpuStatsPayload.blackLoadingMs !== undefined
              ? { blackLoadingMs: gpuStatsPayload.blackLoadingMs }
              : {}),
            ...(perfTelemetry.fps30ComplianceRate !== null
              ? { fps30ComplianceRate: perfTelemetry.fps30ComplianceRate }
              : {}),
            ...(perfTelemetry.dominantBottleneck !== null
              ? { dominantBottleneck: perfTelemetry.dominantBottleneck }
              : {}),
            ...(perfTelemetry.frameJitterMs !== null
              ? { frameJitterMs: perfTelemetry.frameJitterMs }
              : {}),
            ...(perfTelemetry.jsHeapUsedMb !== null
              ? { jsHeapUsedMb: perfTelemetry.jsHeapUsedMb }
              : {}),
            ...(perfTelemetry.vramBudgetUtilization !== null
              ? { vramBudgetUtilization: perfTelemetry.vramBudgetUtilization }
              : {}),
            ...(perfTelemetry.bindGroupHitRate !== null
              ? { bindGroupHitRate: perfTelemetry.bindGroupHitRate }
              : {}),
            ...(perfTelemetry.mailboxLanes !== null
              ? { mailboxLanes: perfTelemetry.mailboxLanes }
              : {}),
          });
        } catch {
          // Ignore HUD metrics errors.
        }
      }
    }

    if (gpuBootPanel) {
      invokeFirstHook(
        gpuBootPanel,
        ['updateGpuStats', 'setGpuStats', 'onGpuStats'],
        gpuStatsPayload
      );
    }
  }

  function ingestPerfAndIoTelemetry(
    data,
    { notifyHud = true, updatePanel = true, allowLanesAlias = false } = {}
  ) {
    if (!data || typeof data !== 'object' || destroyed) {
      return { updated: false, panelFieldUpdated: false, panelPatch: {} };
    }

    const sources = collectPlainSources([
      data,
      data.stats,
      data.perfStats,
      data.perf,
      data.pacingStats,
      data.pacing,
      data.frameBudget,
      data.budget,
      data.frameStats,
      data.frames,
      data.gpuStats,
      data.gpu,
      data.drawStats,
      data.swapchainStats,
      data.swapchain,
      data.memoryStats,
      data.memory,
      data.vramStats,
      data.vram,
      data.resourceTableStats,
      data.resourceTable,
      data.bindGroupStats,
      data.renderStateStats,
      data.ioStats,
      data.io,
      data.httpfsStats,
      data.mailboxStats,
      data.mailbox,
      data.metrics,
      data.frameBreakdown,
      data.timing,
      data.frameTiming,
    ]);

    let updated = false;
    let frameBudgetUpdated = false;
    let memoryUpdated = false;
    let mailboxUpdated = false;
    let panelFieldUpdated = false;
    const panelPatch = {};

    // 1. Feed detailed per-stage frame timing into PerfMonitor.recordDetailedFrame() (W287).
    const frameMsField = pickFirstDefinedField(sources, ['frameMs', 'frameTimeMs', 'frame_ms']);
    const ioWaitMsField = pickFirstDefinedField(sources, ['ioWaitMs', 'io_wait_ms']);
    const shaderCompileMsField = pickFirstDefinedField(sources, [
      'shaderCompileMs',
      'shader_compile_ms',
    ]);
    const gpuMsField = pickFirstDefinedField(sources, ['gpuMs', 'gpu_ms', 'gpuTimeMs']);
    const cpuMsField = pickFirstDefinedField(sources, ['cpuMs', 'cpu_ms', 'cpuTimeMs']);

    const parsedFrameMs = frameMsField.found
      ? sanitizeNonNegativeFiniteNumber(frameMsField.value)
      : null;
    const parsedIoWaitMs = ioWaitMsField.found
      ? sanitizeNonNegativeFiniteNumber(ioWaitMsField.value)
      : null;
    const parsedShaderCompileMs = shaderCompileMsField.found
      ? sanitizeNonNegativeFiniteNumber(shaderCompileMsField.value)
      : null;
    const parsedGpuMs = gpuMsField.found ? sanitizeNonNegativeFiniteNumber(gpuMsField.value) : null;
    const parsedCpuMs = cpuMsField.found ? sanitizeNonNegativeFiniteNumber(cpuMsField.value) : null;

    let perfSnap = null;
    let recordedDetailedFrame = false;
    const hasDetailedFrameTiming =
      parsedFrameMs !== null ||
      parsedIoWaitMs !== null ||
      parsedShaderCompileMs !== null ||
      parsedGpuMs !== null ||
      parsedCpuMs !== null;

    if (hasDetailedFrameTiming && perfMonitor && typeof perfMonitor.recordDetailedFrame === 'function') {
      try {
        const sample = {};
        if (parsedFrameMs !== null) sample.frameMs = parsedFrameMs;
        if (parsedIoWaitMs !== null) sample.ioWaitMs = parsedIoWaitMs;
        if (parsedShaderCompileMs !== null) sample.shaderCompileMs = parsedShaderCompileMs;
        if (parsedGpuMs !== null) sample.gpuMs = parsedGpuMs;
        if (parsedCpuMs !== null) sample.cpuMs = parsedCpuMs;
        perfMonitor.recordDetailedFrame(sample);
        recordedDetailedFrame = true;
        if (typeof perfMonitor.getSnapshot === 'function') {
          perfSnap = perfMonitor.getSnapshot();
        }
      } catch {
        // PerfMonitor errors must never interrupt worker telemetry.
      }
    }

    // 2. 30 FPS compliance rate & dominant bottleneck (explicit or derived from PerfMonitor).
    const complianceField = pickFirstDefinedField(sources, [
      'fps30ComplianceRate',
      'fps30_compliance_rate',
      'complianceRate30Fps',
      'compliance_rate_30fps',
      'fps30Compliance',
      'fps30_compliance',
    ]);
    if (complianceField.found) {
      const rate = sanitizeUnitInterval(complianceField.value);
      if (rate !== null) {
        perfTelemetry.fps30ComplianceRate = rate;
        panelPatch.fps30ComplianceRate = rate;
        frameBudgetUpdated = true;
        panelFieldUpdated = true;
        updated = true;
      }
    } else if (perfSnap && typeof perfSnap.fps30ComplianceRate === 'number') {
      const rate = sanitizeUnitInterval(perfSnap.fps30ComplianceRate);
      if (rate !== null) {
        perfTelemetry.fps30ComplianceRate = rate;
        panelPatch.fps30ComplianceRate = rate;
        frameBudgetUpdated = true;
        panelFieldUpdated = true;
        updated = true;
      }
    }

    const bottleneckField = pickFirstDefinedField(sources, [
      'dominantBottleneck',
      'dominant_bottleneck',
      'bottleneck',
    ]);
    if (bottleneckField.found) {
      const bottleneck = normalizeDominantBottleneck(bottleneckField.value);
      if (bottleneck !== null) {
        perfTelemetry.dominantBottleneck = bottleneck;
        frameBudgetUpdated = true;
        updated = true;
      }
    } else if (perfSnap && typeof perfSnap.dominantBottleneck === 'string') {
      const bottleneck = normalizeDominantBottleneck(perfSnap.dominantBottleneck);
      if (bottleneck !== null) {
        perfTelemetry.dominantBottleneck = bottleneck;
        frameBudgetUpdated = true;
        updated = true;
      }
    }

    // 3. Frame jitter (ms).
    const jitterField = pickFirstDefinedField(sources, [
      'frameJitterMs',
      'frame_jitter_ms',
      'jitterMs',
    ]);
    if (jitterField.found) {
      const jitter = sanitizeNonNegativeFiniteNumber(jitterField.value);
      if (jitter !== null) {
        perfTelemetry.frameJitterMs = jitter;
        panelPatch.frameJitterMs = jitter;
        panelFieldUpdated = true;
        updated = true;
      }
    }

    // 4. VRAM budget utilization ([0, 1]).
    const vramField = pickFirstDefinedField(sources, [
      'vramBudgetUtilization',
      'vram_budget_utilization',
      'vramUtilization',
      'vram_utilization',
    ]);
    if (vramField.found) {
      const util = sanitizeUnitInterval(vramField.value);
      if (util !== null) {
        perfTelemetry.vramBudgetUtilization = util;
        panelPatch.vramBudgetUtilization = util;
        panelFieldUpdated = true;
        updated = true;
      }
    } else {
      const vramUsedField = pickFirstDefinedField(sources, ['vramUsedBytes', 'vram_used_bytes']);
      const vramBudgetField = pickFirstDefinedField(sources, [
        'vramBudgetBytes',
        'vram_budget_bytes',
      ]);
      if (vramUsedField.found && vramBudgetField.found) {
        const used = sanitizeNonNegativeFiniteNumber(vramUsedField.value);
        const budget = sanitizeNonNegativeFiniteNumber(vramBudgetField.value);
        if (used !== null && budget !== null && budget > 0) {
          const util = sanitizeUnitInterval(used / budget);
          if (util !== null) {
            perfTelemetry.vramBudgetUtilization = util;
            panelPatch.vramBudgetUtilization = util;
            panelFieldUpdated = true;
            updated = true;
          }
        }
      }
    }

    // 5. Bind-group cache hit rate ([0, 1]).
    const bgHitField = pickFirstDefinedField(sources, [
      'bindGroupHitRate',
      'bind_group_hit_rate',
      'bindGroupCacheHitRate',
      'bind_group_cache_hit_rate',
    ]);
    if (bgHitField.found) {
      const hitRate = sanitizeUnitInterval(bgHitField.value);
      if (hitRate !== null) {
        perfTelemetry.bindGroupHitRate = hitRate;
        panelPatch.bindGroupHitRate = hitRate;
        panelFieldUpdated = true;
        updated = true;
      }
    } else {
      const bgHitsField = pickFirstDefinedField(sources, ['bindGroupCacheHits']);
      const bgMissesField = pickFirstDefinedField(sources, ['bindGroupCacheMisses']);
      if (bgHitsField.found && bgMissesField.found) {
        const hits = sanitizeNonNegativeFiniteNumber(bgHitsField.value);
        const misses = sanitizeNonNegativeFiniteNumber(bgMissesField.value);
        if (hits !== null && misses !== null && hits + misses > 0) {
          const hitRate = sanitizeUnitInterval(hits / (hits + misses));
          if (hitRate !== null) {
            perfTelemetry.bindGroupHitRate = hitRate;
            panelPatch.bindGroupHitRate = hitRate;
            panelFieldUpdated = true;
            updated = true;
          }
        }
      }
    }

    // 6. JS heap memory stats (explicit payload or performance.memory fallback).
    const heapUsedMbField = pickFirstDefinedField(sources, [
      'jsHeapUsedMb',
      'js_heap_used_mb',
      'heapUsedMb',
      'heap_used_mb',
    ]);
    const heapTotalMbField = pickFirstDefinedField(sources, [
      'jsHeapTotalMb',
      'js_heap_total_mb',
      'heapTotalMb',
      'heap_total_mb',
    ]);
    const heapUsedBytesField = pickFirstDefinedField(sources, [
      'usedJSHeapSize',
      'jsHeapUsedBytes',
      'js_heap_used_bytes',
    ]);
    const heapTotalBytesField = pickFirstDefinedField(sources, [
      'totalJSHeapSize',
      'jsHeapTotalBytes',
      'js_heap_total_bytes',
    ]);

    let sawExplicitMemory = false;
    if (heapUsedMbField.found) {
      sawExplicitMemory = true;
      const usedMb = sanitizeNonNegativeFiniteNumber(heapUsedMbField.value);
      if (usedMb !== null) {
        perfTelemetry.jsHeapUsedMb = usedMb;
        panelPatch.jsHeapUsedMb = usedMb;
        memoryUpdated = true;
        panelFieldUpdated = true;
        updated = true;
      }
    } else if (heapUsedBytesField.found) {
      sawExplicitMemory = true;
      const usedBytes = sanitizeNonNegativeFiniteNumber(heapUsedBytesField.value);
      if (usedBytes !== null) {
        const usedMb = Math.round((usedBytes / BYTES_PER_MIB) * 100) / 100;
        perfTelemetry.jsHeapUsedMb = usedMb;
        panelPatch.jsHeapUsedMb = usedMb;
        memoryUpdated = true;
        panelFieldUpdated = true;
        updated = true;
      }
    }

    if (heapTotalMbField.found) {
      sawExplicitMemory = true;
      const totalMb = sanitizeNonNegativeFiniteNumber(heapTotalMbField.value);
      if (totalMb !== null) {
        perfTelemetry.jsHeapTotalMb = totalMb;
        memoryUpdated = true;
        updated = true;
      }
    } else if (heapTotalBytesField.found) {
      sawExplicitMemory = true;
      const totalBytes = sanitizeNonNegativeFiniteNumber(heapTotalBytesField.value);
      if (totalBytes !== null) {
        perfTelemetry.jsHeapTotalMb = Math.round((totalBytes / BYTES_PER_MIB) * 100) / 100;
        memoryUpdated = true;
        updated = true;
      }
    }

    if (!sawExplicitMemory) {
      const sampleNowMs = now();
      const needsFirstHudNotify = notifyHud && !jsHeapNotifiedToHud;
      const shouldSampleHeap =
        perfTelemetry.jsHeapUsedMb === null ||
        needsFirstHudNotify ||
        sampleNowMs - lastJsHeapSampleAtMs >= JS_HEAP_SAMPLE_THROTTLE_MS ||
        sampleNowMs < lastJsHeapSampleAtMs;
      if (shouldSampleHeap) {
        lastJsHeapSampleAtMs = sampleNowMs;
        const sampledMem = sampleJsHeapMemory();
        if (sampledMem) {
          if (
            sampledMem.jsHeapUsedMb !== null &&
            (sampledMem.jsHeapUsedMb !== perfTelemetry.jsHeapUsedMb || needsFirstHudNotify)
          ) {
            perfTelemetry.jsHeapUsedMb = sampledMem.jsHeapUsedMb;
            panelPatch.jsHeapUsedMb = sampledMem.jsHeapUsedMb;
            memoryUpdated = true;
            panelFieldUpdated = true;
            updated = true;
          }
          if (
            sampledMem.jsHeapTotalMb !== null &&
            (sampledMem.jsHeapTotalMb !== perfTelemetry.jsHeapTotalMb || needsFirstHudNotify)
          ) {
            perfTelemetry.jsHeapTotalMb = sampledMem.jsHeapTotalMb;
            memoryUpdated = true;
            updated = true;
          }
          if (notifyHud && memoryUpdated) {
            jsHeapNotifiedToHud = true;
          }
        }
      }
    }

    // 7. HTTPFS I/O mailbox lane count (W260/W294).
    const mailboxKeys = allowLanesAlias
      ? [
          'mailboxLanes',
          'mailbox_lanes',
          'ioMailboxLanes',
          'io_mailbox_lanes',
          'httpfsMailboxLanes',
          'httpfs_mailbox_lanes',
          'lanes',
        ]
      : [
          'mailboxLanes',
          'mailbox_lanes',
          'ioMailboxLanes',
          'io_mailbox_lanes',
          'httpfsMailboxLanes',
          'httpfs_mailbox_lanes',
        ];
    let mailboxField = pickFirstDefinedField(sources, mailboxKeys);
    if (!mailboxField.found && !allowLanesAlias) {
      const ioOnlySources = collectPlainSources([
        data.ioStats,
        data.httpfsStats,
        data.mailboxStats,
        data.mailbox,
      ]);
      if (ioOnlySources.length > 0) {
        mailboxField = pickFirstDefinedField(ioOnlySources, ['lanes']);
      }
    }
    if (mailboxField.found) {
      const lanes = sanitizePositiveInt(mailboxField.value);
      if (lanes !== null) {
        perfTelemetry.mailboxLanes = lanes;
        panelPatch.mailboxLanes = lanes;
        mailboxUpdated = true;
        panelFieldUpdated = true;
        updated = true;
      }
    }

    // 8. Forward cleanly to HudController (W289) with defensive feature checks.
    if (updated && notifyHud) {
      const hudTargets = resolveHudTargets();
      const frameBudgetPayload = {};
      if (perfTelemetry.fps30ComplianceRate !== null) {
        frameBudgetPayload.fps30ComplianceRate = perfTelemetry.fps30ComplianceRate;
      }
      if (perfTelemetry.dominantBottleneck !== null) {
        frameBudgetPayload.dominantBottleneck = perfTelemetry.dominantBottleneck;
      }

      const memoryPayload = {};
      if (perfTelemetry.jsHeapUsedMb !== null) {
        memoryPayload.jsHeapUsedMb = perfTelemetry.jsHeapUsedMb;
      }
      if (perfTelemetry.jsHeapTotalMb !== null) {
        memoryPayload.jsHeapTotalMb = perfTelemetry.jsHeapTotalMb;
      }

      const perfPayload = {};
      if (perfTelemetry.fps30ComplianceRate !== null) {
        perfPayload.fps30ComplianceRate = perfTelemetry.fps30ComplianceRate;
      }
      if (perfTelemetry.dominantBottleneck !== null) {
        perfPayload.dominantBottleneck = perfTelemetry.dominantBottleneck;
      }
      if (perfTelemetry.frameJitterMs !== null) {
        perfPayload.frameJitterMs = perfTelemetry.frameJitterMs;
      }
      if (perfTelemetry.jsHeapUsedMb !== null) {
        perfPayload.jsHeapUsedMb = perfTelemetry.jsHeapUsedMb;
      }
      if (perfTelemetry.jsHeapTotalMb !== null) {
        perfPayload.jsHeapTotalMb = perfTelemetry.jsHeapTotalMb;
      }
      if (perfTelemetry.vramBudgetUtilization !== null) {
        perfPayload.vramBudgetUtilization = perfTelemetry.vramBudgetUtilization;
      }
      if (perfTelemetry.bindGroupHitRate !== null) {
        perfPayload.bindGroupHitRate = perfTelemetry.bindGroupHitRate;
      }
      if (perfTelemetry.mailboxLanes !== null) {
        perfPayload.mailboxLanes = perfTelemetry.mailboxLanes;
      }
      if (perfSnap && perfSnap.frames && typeof perfSnap.frames === 'object') {
        if (typeof perfSnap.frames.fps === 'number' && Number.isFinite(perfSnap.frames.fps) && perfSnap.frames.fps > 0) {
          perfPayload.fps = perfSnap.frames.fps;
        }
        if (
          typeof perfSnap.frames.onePercentLowFps === 'number' &&
          Number.isFinite(perfSnap.frames.onePercentLowFps) &&
          perfSnap.frames.onePercentLowFps > 0
        ) {
          perfPayload.onePercentLowFps = perfSnap.frames.onePercentLowFps;
        }
        if (
          typeof perfSnap.frames.p99FrameMs === 'number' &&
          Number.isFinite(perfSnap.frames.p99FrameMs) &&
          perfSnap.frames.p99FrameMs > 0
        ) {
          perfPayload.p99FrameMs = perfSnap.frames.p99FrameMs;
        }
      }

      for (let i = 0; i < hudTargets.length; i += 1) {
        const target = hudTargets[i];
        if (!target || typeof target !== 'object') {
          continue;
        }
        if (frameBudgetUpdated && typeof target.setFrameBudgetStats === 'function') {
          try {
            target.setFrameBudgetStats(frameBudgetPayload);
          } catch {
            // Optional HUD hooks must never interrupt telemetry.
          }
        }
        if (memoryUpdated && typeof target.setMemoryStats === 'function') {
          try {
            target.setMemoryStats(memoryPayload);
          } catch {
            // Optional HUD hooks must never interrupt telemetry.
          }
        }
        if (
          mailboxUpdated &&
          perfTelemetry.mailboxLanes !== null &&
          typeof target.setIoMailboxLanes === 'function'
        ) {
          try {
            target.setIoMailboxLanes(perfTelemetry.mailboxLanes);
          } catch {
            // Optional HUD hooks must never interrupt telemetry.
          }
        }
        if (typeof target.updatePerfStats === 'function') {
          try {
            target.updatePerfStats(perfPayload);
          } catch {
            // Optional HUD hooks must never interrupt telemetry.
          }
        }
      }
    }

    // 9. Forward cleanly to GpuBootPanel (W292) when requested.
    if (updatePanel && panelFieldUpdated && gpuBootPanel) {
      safeUpdateGpuBootPanel(panelPatch);
    }

    return { updated, panelFieldUpdated, panelPatch, recordedDetailedFrame };
  }

  function applyGpuStatsFields(data) {
    const sources = collectPlainSources([
      data,
      data.stats,
      data.gpuStats,
      data.drawStats,
      data.frameStats,
      data.swapchainStats,
      data.swapchain,
      data.deviceStats,
      data.stats && data.stats.deviceStats,
      data.shaderPackStats,
      data.stats && data.stats.shaderPackStats,
      data.shaderStreaming,
    ]);
    let matchedW83Field = false;

    const adapterLabel = ingestGpuAdapterFromPayload(data);
    if (adapterLabel !== null) {
      matchedW83Field = true;
    }

    const realPackField = pickFirstDefinedField(sources, [
      'realPackDrawCount',
      'real_pack_draw_count',
      'realPackDraws',
      'packDrawCount',
      'packDraws',
    ]);
    if (realPackField.found) {
      gpuTelemetry.realPackDrawCount = toSafeNonNegativeInt(realPackField.value, 0);
      matchedW83Field = true;
    }

    const fallbackField = pickFirstDefinedField(sources, [
      'fallbackDrawCount',
      'fallbackDraws',
    ]);
    if (fallbackField.found) {
      gpuTelemetry.fallbackDrawCount = toSafeNonNegativeInt(fallbackField.value, 0);
      matchedW83Field = true;
    }

    const presentField = pickFirstDefinedField(sources, [
      'presentCount',
      'presents',
      'framesPresented',
      'frameCount',
    ]);
    if (presentField.found) {
      const parsedPresent = toSafeNonNegativeInt(presentField.value, 0);
      gpuTelemetry.presentCount = parsedPresent;
      if (parsedPresent > 0) {
        wgpuFramesPresented = Math.max(wgpuFramesPresented, parsedPresent);
      }
      matchedW83Field = true;
    }

    const unboundField = pickFirstDefinedField(sources, [
      'unboundBindGroupCount',
      'unboundBindGroups',
      'missingBindGroupCount',
    ]);
    if (unboundField.found) {
      gpuTelemetry.unboundBindGroupCount = toSafeNonNegativeInt(unboundField.value, 0);
      matchedW83Field = true;
    }

    const skippedField = pickFirstDefinedField(sources, ['skippedDrawCount', 'skippedDraws']);
    if (skippedField.found) {
      gpuTelemetry.skippedDrawCount = toSafeNonNegativeInt(skippedField.value, 0);
      matchedW83Field = true;
    }

    const validationCountField = pickFirstDefinedField(sources, [
      'validationErrorCount',
      'validationErrors',
    ]);
    if (validationCountField.found) {
      const parsedCount = toSafeNonNegativeInt(validationCountField.value, 0);
      validationErrorCount = Math.max(validationErrorCount, parsedCount);
      matchedW83Field = true;
    }

    const validationCodeField = pickFirstDefinedField(sources, [
      'lastValidationErrorCode',
      'validationErrorCode',
    ]);
    if (validationCodeField.found && validationCodeField.value !== null) {
      gpuTelemetry.lastValidationErrorCode = sanitizeValidationErrorCode(
        validationCodeField.value,
        'validation'
      );
      matchedW83Field = true;
    }

    const validationMsgField = pickFirstDefinedField(sources, [
      'lastValidationErrorMessage',
      'validationErrorMessage',
    ]);
    if (validationMsgField.found && validationMsgField.value !== null) {
      const sanitizedMsg = sanitizeDiagnosticText(
        validationMsgField.value,
        MAX_VALIDATION_ERROR_MESSAGE_LENGTH
      );
      gpuTelemetry.lastValidationErrorMessage = sanitizedMsg || null;
      matchedW83Field = true;
    }

    const probeCountField = pickFirstDefinedField(sources, [
      'swapchainProbeCount',
      'swapchain_probe_count',
    ]);
    if (probeCountField.found && probeCountField.value !== null) {
      gpuTelemetry.swapchainProbeCount = toSafeNonNegativeInt(probeCountField.value, 0);
      matchedW83Field = true;
    }

    const probeInFlightField = pickFirstDefinedField(sources, [
      'swapchainProbeInFlight',
      'swapchain_probe_in_flight',
    ]);
    if (probeInFlightField.found && typeof probeInFlightField.value === 'boolean') {
      gpuTelemetry.swapchainProbeInFlight = probeInFlightField.value;
      matchedW83Field = true;
    }

    const probeSupportedField = pickFirstDefinedField(sources, [
      'swapchainProbeSupported',
      'swapchain_probe_supported',
    ]);
    if (probeSupportedField.found) {
      if (typeof probeSupportedField.value === 'boolean') {
        gpuTelemetry.swapchainProbeSupported = probeSupportedField.value;
        matchedW83Field = true;
      } else if (probeSupportedField.value === null) {
        gpuTelemetry.swapchainProbeSupported = null;
      }
      if (inputBridge && typeof inputBridge.setSwapchainProbeSupported === 'function') {
        try {
          inputBridge.setSwapchainProbeSupported(gpuTelemetry.swapchainProbeSupported);
        } catch {
          // Non-fatal input bridge probe sync failure.
        }
      }
    }

    const probeErrorCountField = pickFirstDefinedField(sources, [
      'swapchainProbeErrorCount',
      'swapchain_probe_error_count',
    ]);
    if (probeErrorCountField.found && probeErrorCountField.value !== null) {
      gpuTelemetry.swapchainProbeErrorCount = toSafeNonNegativeInt(probeErrorCountField.value, 0);
      matchedW83Field = true;
    }

    const maxLumaField = pickFirstDefinedField(sources, [
      'lastSampledMaxLuma',
      'last_sampled_max_luma',
      'maxLuma',
      'max_luma',
    ]);
    if (
      maxLumaField.found &&
      typeof maxLumaField.value === 'number' &&
      Number.isFinite(maxLumaField.value) &&
      maxLumaField.value >= 0
    ) {
      gpuTelemetry.lastSampledMaxLuma = Math.min(255, Math.floor(maxLumaField.value));
      matchedW83Field = true;
    }

    const suppressedBlackField = pickFirstDefinedField(sources, [
      'suppressedBlackFrames',
      'suppressed_black_frames',
    ]);
    if (suppressedBlackField.found && suppressedBlackField.value !== null) {
      gpuTelemetry.suppressedBlackFrames = toSafeNonNegativeInt(suppressedBlackField.value, 0);
      matchedW83Field = true;
    }

    const probeTimeoutCountField = pickFirstDefinedField(sources, [
      'swapchainProbeTimeoutCount',
      'swapchain_probe_timeout_count',
    ]);
    if (probeTimeoutCountField.found && probeTimeoutCountField.value !== null) {
      gpuTelemetry.swapchainProbeTimeoutCount = toSafeNonNegativeInt(probeTimeoutCountField.value, 0);
      matchedW83Field = true;
    }

    const coalescedPresentsField = pickFirstDefinedField(sources, [
      'coalescedPresents',
      'coalesced_presents',
    ]);
    if (coalescedPresentsField.found && coalescedPresentsField.value !== null) {
      gpuTelemetry.coalescedPresents = toSafeNonNegativeInt(coalescedPresentsField.value, 0);
      matchedW83Field = true;
    }

    const oversizedWgslField = pickFirstDefinedField(sources, [
      'oversizedWgslFallbacks',
      'oversized_wgsl_fallbacks',
    ]);
    if (oversizedWgslField.found && oversizedWgslField.value !== null) {
      gpuTelemetry.oversizedWgslFallbacks = toSafeNonNegativeInt(oversizedWgslField.value, 0);
      matchedW83Field = true;
    }

    const deviceLostField = pickFirstDefinedField(sources, [
      'deviceLostCount',
      'device_lost_count',
    ]);
    if (deviceLostField.found && deviceLostField.value !== null) {
      gpuTelemetry.deviceLostCount = toSafeNonNegativeInt(deviceLostField.value, 0);
      matchedW83Field = true;
    }

    const deviceRecoveredField = pickFirstDefinedField(sources, [
      'deviceRecoveredCount',
      'device_recovered_count',
    ]);
    if (deviceRecoveredField.found && deviceRecoveredField.value !== null) {
      gpuTelemetry.deviceRecoveredCount = toSafeNonNegativeInt(deviceRecoveredField.value, 0);
      matchedW83Field = true;
    }

    const gpuStallWarnField = pickFirstDefinedField(sources, [
      'gpuStallWarnCount',
      'gpu_stall_warn_count',
    ]);
    if (gpuStallWarnField.found && gpuStallWarnField.value !== null) {
      gpuTelemetry.gpuStallWarnCount = toSafeNonNegativeInt(gpuStallWarnField.value, 0);
      matchedW83Field = true;
    }

    const rangedPackActiveField = pickFirstDefinedField(sources, [
      'rangedPackActive',
      'ranged_pack_active',
    ]);
    if (rangedPackActiveField.found && typeof rangedPackActiveField.value === 'boolean') {
      gpuTelemetry.rangedPackActive = rangedPackActiveField.value;
      matchedW83Field = true;
    }

    const rangedShaderReadyField = pickFirstDefinedField(sources, [
      'rangedShaderReady',
      'ranged_shader_ready',
    ]);
    if (rangedShaderReadyField.found && rangedShaderReadyField.value !== null) {
      gpuTelemetry.rangedShaderReady = toSafeNonNegativeInt(rangedShaderReadyField.value, 0);
      matchedW83Field = true;
    }

    const rangedShaderFetchesField = pickFirstDefinedField(sources, [
      'rangedShaderFetches',
      'ranged_shader_fetches',
    ]);
    if (rangedShaderFetchesField.found && rangedShaderFetchesField.value !== null) {
      gpuTelemetry.rangedShaderFetches = toSafeNonNegativeInt(rangedShaderFetchesField.value, 0);
      matchedW83Field = true;
    }

    const rangedShaderErrorsField = pickFirstDefinedField(sources, [
      'rangedShaderErrors',
      'ranged_shader_errors',
    ]);
    if (rangedShaderErrorsField.found && rangedShaderErrorsField.value !== null) {
      gpuTelemetry.rangedShaderErrors = toSafeNonNegativeInt(rangedShaderErrorsField.value, 0);
      matchedW83Field = true;
    }

    const asyncShaderPendingField = pickFirstDefinedField(sources, [
      'asyncShaderPendingCount',
      'async_shader_pending_count',
    ]);
    if (asyncShaderPendingField.found && asyncShaderPendingField.value !== null) {
      gpuTelemetry.asyncShaderPendingCount = toSafeNonNegativeInt(asyncShaderPendingField.value, 0);
      matchedW83Field = true;
    }

    // #1077: ingest structured WebGPU VRAM counters (`gpuMemory`) from wgpu_worker / loader.
    const rawGpuMemoryObj = (() => {
      for (let i = 0; i < sources.length; i += 1) {
        const candidate = sources[i] && sources[i].gpuMemory;
        if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
          return candidate;
        }
      }
      return null;
    })();
    const GPU_MEM_FLAT_KEYS = [
      'liveBufferBytes',
      'liveTextureBytes',
      'peakBufferBytes',
      'peakTextureBytes',
      'stagingPoolBytes',
      'totalLiveGpuBytes',
      'pendingDeferredDestroys',
      'pendingDeferredDestroyBytes',
      'deferredHighWaterFlushes',
    ];
    const extractedGpuMem = {};
    let sawGpuMemField = false;
    if (rawGpuMemoryObj) {
      for (const key of Object.keys(rawGpuMemoryObj)) {
        const val = rawGpuMemoryObj[key];
        if (typeof val === 'number' && Number.isFinite(val) && val >= 0) {
          extractedGpuMem[key] = val;
          sawGpuMemField = true;
        }
      }
    }
    for (let i = 0; i < sources.length; i += 1) {
      const src = sources[i];
      if (!src) continue;
      for (let j = 0; j < GPU_MEM_FLAT_KEYS.length; j += 1) {
        const key = GPU_MEM_FLAT_KEYS[j];
        if (extractedGpuMem[key] !== undefined) continue;
        const val = src[key];
        if (typeof val === 'number' && Number.isFinite(val) && val >= 0) {
          extractedGpuMem[key] = val;
          sawGpuMemField = true;
        }
      }
    }
    if (sawGpuMemField) {
      const prev = gpuTelemetry.gpuMemory && typeof gpuTelemetry.gpuMemory === 'object'
        ? gpuTelemetry.gpuMemory
        : {};
      const merged = { ...prev, ...extractedGpuMem };
      if (typeof prev.peakBufferBytes === 'number' && typeof merged.peakBufferBytes === 'number') {
        merged.peakBufferBytes = Math.max(prev.peakBufferBytes, merged.peakBufferBytes);
      }
      if (typeof merged.liveBufferBytes === 'number') {
        merged.peakBufferBytes = Math.max(
          typeof merged.peakBufferBytes === 'number' ? merged.peakBufferBytes : 0,
          merged.liveBufferBytes
        );
      }
      if (typeof prev.peakTextureBytes === 'number' && typeof merged.peakTextureBytes === 'number') {
        merged.peakTextureBytes = Math.max(prev.peakTextureBytes, merged.peakTextureBytes);
      }
      if (typeof merged.liveTextureBytes === 'number') {
        merged.peakTextureBytes = Math.max(
          typeof merged.peakTextureBytes === 'number' ? merged.peakTextureBytes : 0,
          merged.liveTextureBytes
        );
      }
      if (
        typeof prev.deferredHighWaterFlushes === 'number' &&
        typeof merged.deferredHighWaterFlushes === 'number'
      ) {
        merged.deferredHighWaterFlushes = Math.max(
          prev.deferredHighWaterFlushes,
          merged.deferredHighWaterFlushes
        );
      }
      if (
        merged.totalLiveGpuBytes === undefined &&
        (merged.liveBufferBytes !== undefined ||
          merged.liveTextureBytes !== undefined ||
          merged.stagingPoolBytes !== undefined)
      ) {
        merged.totalLiveGpuBytes =
          (merged.liveBufferBytes || 0) +
          (merged.liveTextureBytes || 0) +
          (merged.stagingPoolBytes || 0);
      }
      gpuTelemetry.gpuMemory = merged;
      matchedW83Field = true;
    }

    return matchedW83Field;
  }

  function handleGpuStatsMessage(data) {
    if (!data || typeof data !== 'object' || destroyed) {
      return false;
    }
    applyGpuStatsFields(data);
    const perfRes = ingestPerfAndIoTelemetry(data, { notifyHud: true, updatePanel: false });
    recordRealWasmPerfUpdate(data, {
      framesPresented: gpuTelemetry.presentCount > 0 ? gpuTelemetry.presentCount : null,
      detailedFrameRecorded: Boolean(perfRes && perfRes.recordedDetailedFrame),
    });
    syncVisibleFrameFromTelemetry(data, { source: 'gpu-stats', deferObserverForward: true });
    const gpuStatsPayload = buildGpuStatsPayload();
    forwardGpuStatsToObservers(gpuStatsPayload, { allowHudMetricsFallback: true });
    if (gpuStatsPayload.presentCount > 0 || perfRes.panelFieldUpdated) {
      const panelPatch = { ...perfRes.panelPatch };
      if (gpuStatsPayload.presentCount > 0) {
        panelPatch.frameStats = {
          ...(latestPanelFrameStats && typeof latestPanelFrameStats === 'object'
            ? latestPanelFrameStats
            : {}),
          framesPresented: gpuStatsPayload.presentCount,
          droppedFrames:
            latestPanelFrameStats && Number.isFinite(latestPanelFrameStats.droppedFrames)
              ? latestPanelFrameStats.droppedFrames
              : 0,
        };
      }
      safeUpdateGpuBootPanel(panelPatch);
    }
    return true;
  }

  function ingestAudioRingTelemetry(data, { allowGenericAttached = false } = {}) {
    if (!data || typeof data !== 'object' || destroyed) {
      return false;
    }
    const sources = collectPlainSources([
      data,
      data.stats,
      data.audioStats,
      data.audio,
      data.audioRing,
      data.audioRingTelemetry,
      data.metrics,
    ]);
    let updated = false;

    const attachedKeys = allowGenericAttached
      ? [
          'audioRingAttached',
          'audio_ring_attached',
          'guestRingAttached',
          'guest_ring_attached',
          'pcmRingAttached',
          'pcm_ring_attached',
          'ringAttached',
          'attached',
        ]
      : [
          'audioRingAttached',
          'audio_ring_attached',
          'guestRingAttached',
          'guest_ring_attached',
          'pcmRingAttached',
          'pcm_ring_attached',
          'ringAttached',
        ];
    const attachedField = pickFirstDefinedField(sources, attachedKeys);
    let sawAttached = null;
    if (attachedField.found) {
      if (typeof attachedField.value === 'boolean') {
        sawAttached = attachedField.value;
      } else if (typeof attachedField.value === 'string') {
        sawAttached = parseBooleanQueryFlag(attachedField.value);
      }
      if (sawAttached !== null) {
        audioRingTelemetry.audioRingAttached = sawAttached;
        updated = true;
      }
    }

    const reasonField = pickFirstDefinedField(sources, [
      'audioRingLastReason',
      'audio_ring_last_reason',
      'statusReason',
    ]);
    let sawReason = null;
    if (reasonField.found && typeof reasonField.value === 'string') {
      const cleanReason = sanitizeDiagnosticText(reasonField.value, 64);
      if (cleanReason) {
        sawReason = cleanReason;
        audioRingTelemetry.audioRingLastReason = cleanReason;
        updated = true;
      }
    }

    const attemptsField = pickFirstDefinedField(sources, [
      'audioRingAttachAttempts',
      'audio_ring_attach_attempts',
      'attachAttempts',
      'attach_attempts',
    ]);
    const retryField = pickFirstDefinedField(sources, [
      'audioRingRetryCount',
      'audio_ring_retry_count',
    ]);

    if (attemptsField.found) {
      const parsedAttempts = toNonNegativeInt(attemptsField.value, null);
      if (parsedAttempts !== null) {
        audioRingTelemetry.audioRingAttachAttempts = Math.max(
          audioRingTelemetry.audioRingAttachAttempts,
          parsedAttempts
        );
        updated = true;
      }
    } else if (retryField.found) {
      const parsedRetries = toNonNegativeInt(retryField.value, null);
      if (parsedRetries !== null) {
        const minAttempts =
          sawAttached === true || sawReason !== null || parsedRetries > 0
            ? parsedRetries + (sawAttached === true || sawReason !== null ? 1 : 0)
            : 0;
        audioRingTelemetry.audioRingAttachAttempts = Math.max(
          audioRingTelemetry.audioRingAttachAttempts,
          minAttempts
        );
        updated = true;
      }
    } else if (sawAttached === true && audioRingTelemetry.audioRingAttachAttempts === 0) {
      audioRingTelemetry.audioRingAttachAttempts = 1;
    }

    return updated;
  }

  function handleAudioStatsMessage(data) {
    if (!data || typeof data !== 'object' || destroyed) {
      return false;
    }
    const sources = collectPlainSources([data, data.stats, data.audioStats, data.audio]);
    audioTelemetry.hasReceivedStats = true;
    ingestAudioRingTelemetry(data, { allowGenericAttached: false });

    const outputStateField = pickFirstDefinedField(sources, [
      'outputState',
      'state',
      'audioState',
      'contextState',
    ]);
    if (outputStateField.found) {
      audioTelemetry.outputState = normalizeAudioOutputState(outputStateField.value);
    }

    const framesDrainedField = pickFirstDefinedField(sources, [
      'framesDrained',
      'drainedFrames',
      'totalPulledFrames',
      'pulledFrames',
    ]);
    if (framesDrainedField.found) {
      audioTelemetry.framesDrained = toSafeNonNegativeInt(framesDrainedField.value, 0);
    }

    const underrunsField = pickFirstDefinedField(sources, ['underruns', 'underrunCount']);
    if (underrunsField.found) {
      audioTelemetry.underruns = toSafeNonNegativeInt(underrunsField.value, 0);
    }

    const overrunsField = pickFirstDefinedField(sources, [
      'overruns',
      'overrunCount',
      'overflows',
      'overflowCount',
    ]);
    if (overrunsField.found) {
      audioTelemetry.overruns = toSafeNonNegativeInt(overrunsField.value, 0);
    }

    const audioStatsPayload = Object.freeze({
      type: 'audio-stats',
      outputState: audioTelemetry.outputState,
      framesDrained: audioTelemetry.framesDrained,
      underruns: audioTelemetry.underruns,
      overruns: audioTelemetry.overruns,
    });

    if (session) {
      invokeFirstHook(
        session,
        ['updateAudioStats', 'setAudioStats', 'onAudioStats', 'recordAudioStats'],
        audioStatsPayload
      );
    }

    const hudTargets = resolveHudTargets();
    for (let i = 0; i < hudTargets.length; i += 1) {
      const target = hudTargets[i];
      const handled = invokeFirstHook(
        target,
        ['updateAudioStats', 'setAudioStats', 'onAudioStats', 'recordAudioStats', 'updateStats'],
        audioStatsPayload
      );
      if (!handled && target === hud && typeof target.updateMetrics === 'function') {
        try {
          target.updateMetrics({
            audioStats: audioStatsPayload,
            audioOutputState: audioStatsPayload.outputState,
            framesDrained: audioStatsPayload.framesDrained,
            audioUnderruns: audioStatsPayload.underruns,
            audioOverruns: audioStatsPayload.overruns,
          });
        } catch {
          // Ignore HUD metrics errors.
        }
      }
    }

    return true;
  }

  function ingestLoaderTelemetry(data, { notify = true, parseLabel = true } = {}) {
    if (!data || typeof data !== 'object' || destroyed) {
      return false;
    }
    const sources = collectPlainSources([
      data,
      data.metrics,
      data.stats,
      data.pthread,
      data.pthreads,
      data.pthreadStats,
      data.manifest,
      data.programManifest,
      data.bootManifest,
      data.status,
      data.serverStatus,
    ]);
    let updated = false;

    const dispatchField = pickFirstDefinedField(sources, [
      'indirectDispatch',
      'indirect_dispatch',
      'dispatchMode',
      'indirectDispatchMode',
    ]);
    if (dispatchField.found) {
      loaderTelemetry.indirectDispatch = normalizeIndirectDispatch(dispatchField.value);
      updated = true;
    } else if (
      parseLabel &&
      typeof data.label === 'string' &&
      loaderTelemetry.indirectDispatch === 'unknown'
    ) {
      const m = /\bindirect[_ -]?dispatch\s*[:=]\s*(shard|table)\b/i.exec(data.label);
      if (m) {
        loaderTelemetry.indirectDispatch = normalizeIndirectDispatch(m[1]);
        updated = true;
      } else if (/\bshard[_ -]?dispatch\b/i.test(data.label)) {
        loaderTelemetry.indirectDispatch = 'shard';
        updated = true;
      }
    }

    const spawnedField = pickFirstDefinedField(sources, [
      'pthreadSpawnedCount',
      'pthreadSpawned',
      'spawnedCount',
      'spawned',
      'totalSpawned',
    ]);
    if (spawnedField.found) {
      loaderTelemetry.pthreadSpawnedCount = toSafeNonNegativeInt(spawnedField.value, 0);
      updated = true;
    }

    const activeField = pickFirstDefinedField(sources, [
      'pthreadActiveCount',
      'pthreadActive',
      'activeCount',
      'active',
      'liveCount',
    ]);
    if (activeField.found) {
      loaderTelemetry.pthreadActiveCount = toSafeNonNegativeInt(activeField.value, 0);
      updated = true;
    }

    const peakField = pickFirstDefinedField(sources, [
      'pthreadPeakCount',
      'pthreadPeak',
      'peakCount',
      'peak',
      'maxActiveCount',
    ]);
    if (peakField.found) {
      loaderTelemetry.pthreadPeakCount = toSafeNonNegativeInt(peakField.value, 0);
      if (loaderTelemetry.pthreadActiveCount > loaderTelemetry.pthreadPeakCount) {
        loaderTelemetry.pthreadPeakCount = loaderTelemetry.pthreadActiveCount;
      }
      updated = true;
    } else if (activeField.found) {
      loaderTelemetry.pthreadPeakCount = Math.max(
        loaderTelemetry.pthreadPeakCount,
        loaderTelemetry.pthreadActiveCount
      );
    }

    const queueDepthField = pickFirstDefinedField(sources, [
      'pthreadLoadQueueDepth',
      'loadQueueDepth',
      'pthreadQueueDepth',
      'queueDepth',
      'pendingLoadCount',
    ]);
    if (queueDepthField.found) {
      loaderTelemetry.pthreadLoadQueueDepth = toSafeNonNegativeInt(queueDepthField.value, 0);
      updated = true;
    }

    const preferredPackField = pickFirstDefinedField(sources, ['preferredShaderPackUrl']);
    if (preferredPackField.found) {
      const cleanUrl = sanitizeTelemetryUrl(preferredPackField.value);
      if (cleanUrl) {
        loaderTelemetry.preferredShaderPackUrl = cleanUrl;
        updated = true;
      }
    }

    if (updated && notify) {
      const loaderStatsPayload = Object.freeze({
        type: 'loader-stats',
        indirectDispatch: loaderTelemetry.indirectDispatch,
        pthreadSpawnedCount: loaderTelemetry.pthreadSpawnedCount,
        pthreadActiveCount: loaderTelemetry.pthreadActiveCount,
        pthreadPeakCount: loaderTelemetry.pthreadPeakCount,
        pthreadLoadQueueDepth: loaderTelemetry.pthreadLoadQueueDepth,
        preferredShaderPackUrl: resolveSanitizedPreferredShaderPackUrl(),
      });

      if (session) {
        invokeFirstHook(
          session,
          ['updateLoaderStats', 'updatePthreadStats', 'setPthreadStats', 'onPthreadStats'],
          loaderStatsPayload
        );
      }

      const hudTargets = resolveHudTargets();
      for (let i = 0; i < hudTargets.length; i += 1) {
        invokeFirstHook(
          hudTargets[i],
          [
            'updateLoaderStats',
            'updatePthreadStats',
            'setPthreadStats',
            'onPthreadStats',
            'updateRuntimeStats',
            'updateBootTelemetry',
            'updateStats',
          ],
          loaderStatsPayload
        );
      }
    }

    const perfRes = ingestPerfAndIoTelemetry(data, { notifyHud: notify, updatePanel: notify });
    if (perfRes.updated) {
      updated = true;
    }
    if (ingestAudioRingTelemetry(data, { allowGenericAttached: false })) {
      updated = true;
    }
    if (ingestIoWorkerTelemetry(data)) {
      updated = true;
    }

    return updated;
  }

  function ingestIoWorkerTelemetry(data) {
    if (!data || typeof data !== 'object') {
      return false;
    }
    let updated = false;
    const rawIo =
      (data.ioWorkerStats && typeof data.ioWorkerStats === 'object' ? data.ioWorkerStats : null) ||
      (data.ioStats && typeof data.ioStats === 'object' ? data.ioStats : null) ||
      (typeof data.type === 'string' &&
      IO_TELEMETRY_MESSAGE_TYPES.has(data.type) &&
      data.stats &&
      typeof data.stats === 'object'
        ? data.stats
        : null);
    if (rawIo || typeof data.mailboxAttached === 'boolean') {
      const prev = latestIoWorkerStats && typeof latestIoWorkerStats === 'object' ? latestIoWorkerStats : {};
      const next = Object.assign({}, prev, rawIo || {});
      if (typeof data.mailboxAttached === 'boolean') {
        next.mailboxAttached = data.mailboxAttached;
      } else if (typeof next.mailboxAttached !== 'boolean') {
        next.mailboxAttached = false;
      }
      if (typeof data.mailboxLanes === 'number' && Number.isFinite(data.mailboxLanes) && data.mailboxLanes >= 0) {
        next.mailboxLanes = Math.floor(data.mailboxLanes);
      }
      latestIoWorkerStats = Object.freeze(next);
      updated = true;
    }
    if (data.runtimeDiagnostics && typeof data.runtimeDiagnostics === 'object') {
      latestRuntimeDiagnostics = Object.freeze(Object.assign({}, data.runtimeDiagnostics));
      updated = true;
    }
    return updated;
  }

  function handleGpuTelemetryMessage(data) {
    if (!data || typeof data !== 'object' || data.type !== 'gpu-telemetry') {
      return false;
    }
    const patch = {};
    if (data.bootSummary !== undefined) {
      patch.bootSummary = data.bootSummary;
    }
    if (data.frameStats !== undefined) {
      patch.frameStats = data.frameStats;
      const reportedFrames = toNonNegativeInt(
        data.frameStats && data.frameStats.framesPresented,
        null
      );
      if (reportedFrames !== null) {
        wgpuFramesPresented = Math.max(wgpuFramesPresented, reportedFrames);
        gpuTelemetry.presentCount = Math.max(gpuTelemetry.presentCount, reportedFrames);
      }
    }
    if (data.fallbackStats !== undefined) {
      patch.fallbackStats = data.fallbackStats;
    }
    if (data.missingShaders !== undefined) {
      patch.missingShaders = data.missingShaders;
    }
    if (data.shaderPackStats !== undefined) {
      patch.shaderPackStats = data.shaderPackStats;
    }
    if (data.shaderStreaming !== undefined) {
      patch.shaderStreaming = data.shaderStreaming;
    }
    if (typeof data.rangedPackActive === 'boolean') {
      patch.rangedPackActive = data.rangedPackActive;
    }
    if (data.deviceStats !== undefined) {
      patch.deviceStats = data.deviceStats;
    }
    const hadDrawOrBindFields = applyGpuStatsFields(data);
    const perfRes = ingestPerfAndIoTelemetry(data, { notifyHud: true, updatePanel: false });
    recordRealWasmPerfUpdate(data, {
      framesPresented: wgpuFramesPresented > 0 ? wgpuFramesPresented : null,
      detailedFrameRecorded: Boolean(perfRes && perfRes.recordedDetailedFrame),
    });
    Object.assign(patch, perfRes.panelPatch);
    if (data.fastBootMilestone !== undefined) {
      recordFastBootMilestone(data.fastBootMilestone, { source: 'gpu-telemetry' });
    }
    const triggeredVisible = syncVisibleFrameFromTelemetry(data, {
      source: 'gpu-telemetry',
      deferObserverForward: true,
    });
    safeUpdateGpuBootPanel(patch);
    if (hadDrawOrBindFields || triggeredVisible) {
      forwardGpuStatsToObservers(buildGpuStatsPayload(), { allowHudMetricsFallback: false });
    }
    return true;
  }

  function looksLikeAudioStatsPayload(data) {
    const src = data && typeof data.stats === 'object' && data.stats ? data.stats : data;
    if (!src || typeof src !== 'object') {
      return false;
    }
    return (
      'outputState' in src ||
      'framesDrained' in src ||
      'drainedFrames' in src ||
      'underruns' in src ||
      'overruns' in src ||
      'overflows' in src
    );
  }

  function handleWorkerMessage(data) {
    if (!data || typeof data !== 'object') {
      return false;
    }
    if (data.type === GUEST_RELAUNCH_MESSAGE_TYPE) {
      // XLaunchNewGame of the running title: reload the page, i.e. boot again with the same URL.
      if (shouldRelaunchGuest(windowObj)) {
        feedBootLog('The title relaunched itself (XLaunchNewGame); restarting the boot.');
        try {
          windowObj.location.reload();
        } catch {
          // A page without a reloadable location keeps the stopped boot.
        }
      } else {
        feedBootLog('ERROR: the title kept relaunching itself; not restarting again.');
      }
      return true;
    }
    if (data.type === 'gpu-telemetry') {
      return handleGpuTelemetryMessage(data);
    }
    if (data.type === 'gpu-stats') {
      return handleGpuStatsMessage(data);
    }
    if (typeof data.type === 'string' && PERF_TELEMETRY_MESSAGE_TYPES.has(data.type)) {
      ingestPerfAndIoTelemetry(data, { notifyHud: true, updatePanel: true });
      return true;
    }
    if (typeof data.type === 'string' && IO_TELEMETRY_MESSAGE_TYPES.has(data.type)) {
      ingestIoWorkerTelemetry(data);
      ingestPerfAndIoTelemetry(data, {
        notifyHud: true,
        updatePanel: true,
        allowLanesAlias: true,
      });
      return true;
    }
    if (
      typeof data.type === 'string' &&
      Object.prototype.hasOwnProperty.call(FAST_BOOT_MESSAGE_TYPE_TO_MILESTONE, data.type)
    ) {
      recordFastBootMilestone(FAST_BOOT_MESSAGE_TYPE_TO_MILESTONE[data.type], {
        ...data,
        source: data.source || data.type,
      });
      return true;
    }
    if (
      data.type === 'boot-milestone' ||
      data.type === 'fast-boot-milestone' ||
      data.type === 'milestone'
    ) {
      const targetMilestone = normalizeFastBootMilestone(
        data.milestone ?? data.stage ?? data.name
      );
      if (targetMilestone) {
        recordFastBootMilestone(targetMilestone, {
          ...data,
          source: data.source || data.type,
        });
        return true;
      }
    }
    if (data.type === 'stats') {
      if (
        looksLikeAudioStatsPayload(data) &&
        !('realPackDrawCount' in data) &&
        !('fallbackDrawCount' in data) &&
        !('presentCount' in data) &&
        !(
          data.stats &&
          typeof data.stats === 'object' &&
          ('realPackDrawCount' in data.stats ||
            'fallbackDrawCount' in data.stats ||
            'presentCount' in data.stats)
        )
      ) {
        return handleAudioStatsMessage(data);
      }
      return handleGpuStatsMessage(data);
    }
    if (data.type === 'audio-stats' || data.type === 'audio-telemetry') {
      return handleAudioStatsMessage(data);
    }
    if (typeof data.type === 'string' && AUDIO_RING_MESSAGE_TYPES.has(data.type)) {
      const updated = ingestAudioRingTelemetry(data, { allowGenericAttached: true });
      if (data.type === 'audio-ring-attempt' && !updated) {
        audioRingTelemetry.audioRingAttachAttempts += 1;
      }
      if (data.type === 'audio-ring-attached' && data.attached !== false && data.audioRingAttached !== false) {
        audioRingTelemetry.audioRingAttached = true;
        audioRingTelemetry.audioRingAttachAttempts = Math.max(
          audioRingTelemetry.audioRingAttachAttempts,
          1
        );
      }
      return true;
    }
    if (typeof data.type === 'string' && SESSION_PHASE_MESSAGE_TYPES.has(data.type)) {
      if (session && typeof session.recordPhase === 'function') {
        try {
          session.recordPhase(data.phase, data.timestampMs);
        } catch {
          // Session phase transitions must never throw out of worker message handling.
        }
      }
      return true;
    }
    if (typeof data.type === 'string' && MEMORY_PRESSURE_MESSAGE_TYPES.has(data.type)) {
      const pressureReason =
        typeof data.reason === 'string' && data.reason.trim().length > 0
          ? data.reason.trim()
          : data.type;
      Promise.resolve(reclaimSoftMemory(pressureReason)).catch(() => {
        // Soft memory reclaim rejections must never surface as unhandled rejections.
      });
      return true;
    }
    if (data.type === 'gpu-validation-error') {
      return handleGpuValidationErrorMessage(data);
    }
    if (data.type === 'splash-frame-presented') {
      return handleSplashFramePresentedMessage(data);
    }
    if (data.type === 'frame-presented') {
      ingestGpuAdapterFromPayload(data);
      const perfRes = ingestPerfAndIoTelemetry(data, { notifyHud: true, updatePanel: false });
      const reportedFrames = toNonNegativeInt(
        data.presentCount ?? data.framesPresented ?? data.frame ?? data.frameCount,
        null
      );
      if (reportedFrames !== null) {
        wgpuFramesPresented = Math.max(wgpuFramesPresented, reportedFrames);
        gpuTelemetry.presentCount = Math.max(gpuTelemetry.presentCount, reportedFrames);
      } else {
        wgpuFramesPresented += 1;
        gpuTelemetry.presentCount = Math.max(gpuTelemetry.presentCount, wgpuFramesPresented);
      }
      recordRealWasmPerfUpdate(data, {
        framesPresented: wgpuFramesPresented,
        detailedFrameRecorded: Boolean(perfRes && perfRes.recordedDetailedFrame),
      });
      const realPackField = pickFirstDefinedField([data], [
        'realPackDrawCount',
        'real_pack_draw_count',
        'realPackDraws',
      ]);
      if (realPackField.found) {
        gpuTelemetry.realPackDrawCount = Math.max(
          gpuTelemetry.realPackDrawCount,
          toSafeNonNegativeInt(realPackField.value, 0)
        );
      }
      syncVisibleFrameFromTelemetry(data, { source: 'frame-presented', deferObserverForward: true });
      const frameFwdNowMs = now();
      const currentVisible = Boolean(gpuTelemetry.hasVisibleContent);
      const shouldForwardFrameObservers =
        wgpuFramesPresented <= 1 ||
        !isHudHidden() ||
        gpuTelemetry.realPackDrawCount !== lastFramePresentedForwardedDraws ||
        validationErrorCount !== lastFramePresentedForwardedErrors ||
        currentVisible !== lastFramePresentedForwardedVisible ||
        frameFwdNowMs - lastFramePresentedForwardAtMs >= HIDDEN_HUD_FRAME_METRICS_THROTTLE_MS ||
        frameFwdNowMs < lastFramePresentedForwardAtMs;
      if (shouldForwardFrameObservers) {
        lastFramePresentedForwardAtMs = frameFwdNowMs;
        lastFramePresentedForwardedDraws = gpuTelemetry.realPackDrawCount;
        lastFramePresentedForwardedErrors = validationErrorCount;
        lastFramePresentedForwardedVisible = currentVisible;
        forwardGpuStatsToObservers(buildGpuStatsPayload(), { allowHudMetricsFallback: false });
      }
      if (gpuBootPanel && perfRes.panelFieldUpdated) {
        safeUpdateGpuBootPanel(perfRes.panelPatch);
      }
      return true;
    }
    if (data.type === 'first-present') {
      ingestGpuAdapterFromPayload(data);
      const reportedFrames = toNonNegativeInt(data.framesPresented ?? data.presentCount, null);
      if (reportedFrames !== null) {
        wgpuFramesPresented = Math.max(wgpuFramesPresented, reportedFrames);
      } else {
        wgpuFramesPresented = Math.max(wgpuFramesPresented, 1);
      }
      gpuTelemetry.presentCount = Math.max(gpuTelemetry.presentCount, wgpuFramesPresented);
      recordRealWasmPerfUpdate(data, { framesPresented: wgpuFramesPresented });
      const realPackField = pickFirstDefinedField([data], [
        'realPackDrawCount',
        'real_pack_draw_count',
        'realPackDraws',
      ]);
      if (realPackField.found) {
        gpuTelemetry.realPackDrawCount = Math.max(
          gpuTelemetry.realPackDrawCount,
          toSafeNonNegativeInt(realPackField.value, 0)
        );
      }
      notifyFirstPresent({
        source: sanitizeDiagnosticText(data.source || 'wgpu_worker', 64) || 'wgpu_worker',
        frameIndex:
          Number.isInteger(data.frameIndex) && data.frameIndex >= 0 ? data.frameIndex : 0,
        framesPresented: wgpuFramesPresented,
      });
      syncVisibleFrameFromTelemetry(data, { source: 'first-present', deferObserverForward: false });
      return true;
    }
    if (data.type === 'first-visible-frame' || data.type === 'visible-frame') {
      ingestGpuAdapterFromPayload(data);
      const reportedFrames = toNonNegativeInt(data.framesPresented ?? data.presentCount, null);
      if (reportedFrames !== null) {
        wgpuFramesPresented = Math.max(wgpuFramesPresented, reportedFrames);
        gpuTelemetry.presentCount = Math.max(gpuTelemetry.presentCount, reportedFrames);
      }
      recordRealWasmPerfUpdate(data, {
        framesPresented: wgpuFramesPresented > 0 ? wgpuFramesPresented : null,
      });
      const realPackField = pickFirstDefinedField([data], [
        'realPackDrawCount',
        'real_pack_draw_count',
        'realPackDraws',
      ]);
      if (realPackField.found) {
        gpuTelemetry.realPackDrawCount = Math.max(
          gpuTelemetry.realPackDrawCount,
          toSafeNonNegativeInt(realPackField.value, 0)
        );
      }
      if (
        data.firstVisibleFrame === false ||
        data.first_visible_frame === false ||
        data.hasVisibleContent === false ||
        data.has_visible_content === false
      ) {
        sawExplicitVisibleFrameGate = true;
        return true;
      }
      const rawBlackLoadingMs =
        data.blackLoadingMs !== undefined ? data.blackLoadingMs : data.black_loading_ms;
      const rawFirstVisibleFrameMs =
        data.firstVisibleFrameMs !== undefined
          ? data.firstVisibleFrameMs
          : data.first_visible_frame_ms;
      const rawMaxLuma =
        data.maxLuma !== undefined
          ? data.maxLuma
          : data.max_luma !== undefined
            ? data.max_luma
            : data.lastSampledMaxLuma;
      const rawSuppressedBlackFrames =
        data.suppressedBlackFrames !== undefined
          ? data.suppressedBlackFrames
          : data.suppressed_black_frames;
      const rawProbeErrorCount =
        data.swapchainProbeErrorCount !== undefined
          ? data.swapchainProbeErrorCount
          : data.swapchain_probe_error_count;
      if (typeof rawMaxLuma === 'number' && Number.isFinite(rawMaxLuma) && rawMaxLuma >= 0) {
        gpuTelemetry.lastSampledMaxLuma = Math.min(255, Math.floor(rawMaxLuma));
      }
      if (
        typeof rawSuppressedBlackFrames === 'number' &&
        Number.isFinite(rawSuppressedBlackFrames) &&
        rawSuppressedBlackFrames >= 0
      ) {
        gpuTelemetry.suppressedBlackFrames = Math.floor(rawSuppressedBlackFrames);
      }
      if (
        typeof rawProbeErrorCount === 'number' &&
        Number.isFinite(rawProbeErrorCount) &&
        rawProbeErrorCount >= 0
      ) {
        gpuTelemetry.swapchainProbeErrorCount = Math.floor(rawProbeErrorCount);
      }
      notifyFirstVisibleFrame({
        source: sanitizeDiagnosticText(data.source || data.type, 64) || data.type,
        ...(typeof data.elapsedMs === 'number' && Number.isFinite(data.elapsedMs) && data.elapsedMs >= 0
          ? { elapsedMs: data.elapsedMs }
          : {}),
        ...(typeof rawFirstVisibleFrameMs === 'number' &&
        Number.isFinite(rawFirstVisibleFrameMs) &&
        rawFirstVisibleFrameMs >= 0
          ? { firstVisibleFrameMs: rawFirstVisibleFrameMs }
          : {}),
        ...(typeof rawBlackLoadingMs === 'number' &&
        Number.isFinite(rawBlackLoadingMs) &&
        rawBlackLoadingMs >= 0
          ? { blackLoadingMs: rawBlackLoadingMs }
          : {}),
        ...(typeof rawMaxLuma === 'number' && Number.isFinite(rawMaxLuma) && rawMaxLuma >= 0
          ? { maxLuma: Math.min(255, Math.floor(rawMaxLuma)) }
          : {}),
        ...(typeof rawSuppressedBlackFrames === 'number' &&
        Number.isFinite(rawSuppressedBlackFrames) &&
        rawSuppressedBlackFrames >= 0
          ? { suppressedBlackFrames: Math.floor(rawSuppressedBlackFrames) }
          : {}),
        ...(typeof data.fadeOutMs === 'number' && Number.isFinite(data.fadeOutMs)
          ? { fadeOutMs: data.fadeOutMs }
          : {}),
        ...(typeof data.fadeDurationMs === 'number' && Number.isFinite(data.fadeDurationMs)
          ? { fadeDurationMs: data.fadeDurationMs }
          : {}),
        ...(typeof data.fadeMs === 'number' && Number.isFinite(data.fadeMs)
          ? { fadeMs: data.fadeMs }
          : {}),
      });
      return true;
    }
    if (data.type === 'ready' || data.type === 'wgpu-ready') {
      ingestGpuAdapterFromPayload(data);
      return data.label === undefined && data.error === undefined && data.metrics === undefined;
    }
    if (
      data.type === 'gpu-adapter' ||
      data.type === 'webgpu-adapter' ||
      data.type === 'adapter-info'
    ) {
      ingestGpuAdapterFromPayload(data);
      return true;
    }
    if (data.type === 'log' || data.type === 'worker-log') {
      const text =
        typeof data.message === 'string'
          ? data.message
          : typeof data.line === 'string'
            ? data.line
            : typeof data.text === 'string'
              ? data.text
              : typeof data.label === 'string'
                ? data.label
                : null;
      if (text) {
        feedBootLog(text);
      }
      return true;
    }
    if (typeof data.type === 'string' && LOADER_TELEMETRY_MESSAGE_TYPES.has(data.type)) {
      ingestLoaderTelemetry(data, { notify: true, parseLabel: true });
      if (data.fastBootMilestone !== undefined) {
        recordFastBootMilestone(data.fastBootMilestone, { source: data.type });
      }
      return true;
    }
    if (data.type === undefined && data.label === undefined && data.error === undefined) {
      let handledBare = false;
      if (ingestAudioRingTelemetry(data, { allowGenericAttached: false })) {
        handledBare = true;
      }
      if (ingestLoaderTelemetry(data, { notify: true, parseLabel: false })) {
        handledBare = true;
      }
      if (handledBare) {
        return true;
      }
    }
    return false;
  }

  function hookBootTracerForPanel() {
    if (!bootTracer || typeof bootTracer !== 'object' || !gpuBootPanel) {
      return;
    }
    let hookDepth = 0;
    let pendingEmit = false;

    const emitFromTracer = (explicitSummary) => {
      if (destroyed) {
        return;
      }
      let summary = explicitSummary;
      if (summary === undefined && typeof bootTracer.getSummary === 'function') {
        try {
          summary = bootTracer.getSummary();
        } catch {
          summary = undefined;
        }
      }
      if (summary !== undefined) {
        safeUpdateGpuBootPanel({ bootSummary: summary });
      }
    };

    const wrapMethod = (methodName, shouldEmitOnResult) => {
      if (typeof bootTracer[methodName] !== 'function') {
        return;
      }
      const hasOwn = Object.prototype.hasOwnProperty.call(bootTracer, methodName);
      const original = bootTracer[methodName];
      const wrapped = function (...args) {
        hookDepth += 1;
        try {
          const result = original.apply(this, args);
          if (shouldEmitOnResult(result)) {
            pendingEmit = true;
          }
          return result;
        } finally {
          hookDepth -= 1;
          if (hookDepth === 0 && pendingEmit) {
            pendingEmit = false;
            emitFromTracer();
          }
        }
      };
      try {
        bootTracer[methodName] = wrapped;
        bootTracerCleanupFns.push(() => {
          if (bootTracer[methodName] === wrapped) {
            if (hasOwn) {
              bootTracer[methodName] = original;
            } else {
              delete bootTracer[methodName];
            }
          }
        });
      } catch {
        // Ignore frozen or non-writable tracer mocks.
      }
    };

    wrapMethod('markMilestone', () => true);
    wrapMethod('recordFirstVisibleFrame', () => true);
    wrapMethod('recordStartScreen', () => true);
    wrapMethod('recordIntroVideo', () => true);
    wrapMethod('recordMainMenu', () => true);
    wrapMethod('recordInGame', () => true);
    wrapMethod('recordGpuAdapter', () => true);
    wrapMethod('recordTrap', () => true);
    wrapMethod('reset', () => true);
    wrapMethod('ingestLogLine', (res) => Boolean(res));
    wrapMethod('ingestLog', (res) => Boolean(res));

    if (isEventTargetLike(bootTracer)) {
      const onTracerEvent = (ev) => {
        const detail =
          ev && typeof ev === 'object' ? ev.detail || ev.summary || ev.bootSummary : undefined;
        const summary =
          detail &&
          typeof detail === 'object' &&
          ('stage' in detail || 'milestonesReached' in detail)
            ? detail
            : undefined;
        emitFromTracer(summary);
      };
      bootTracer.addEventListener('milestone', onTracerEvent);
      bootTracer.addEventListener('update', onTracerEvent);
      bootTracer.addEventListener('summary', onTracerEvent);
      bootTracerCleanupFns.push(() => {
        bootTracer.removeEventListener('milestone', onTracerEvent);
        bootTracer.removeEventListener('update', onTracerEvent);
        bootTracer.removeEventListener('summary', onTracerEvent);
      });
    }
  }

  const PgimPreloaderCtor =
    options.PgimPreloaderCtor ||
    (globalObj.PlayGta6PgimPreloader && globalObj.PlayGta6PgimPreloader.PgimPreloader) ||
    (typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6PgimPreloader &&
      globalThis.PlayGta6PgimPreloader.PgimPreloader) ||
    null;
  const pgimPreloaderOptions =
    options.pgimPreloaderOptions && typeof options.pgimPreloaderOptions === 'object'
      ? options.pgimPreloaderOptions
      : {};
  const resolvedPgimOpfsCache =
    options.opfsCache !== undefined ? options.opfsCache : pgimPreloaderOptions.opfsCache;
  const resolvedPgimOpfsCacheVersion =
    options.opfsCacheVersion !== undefined
      ? options.opfsCacheVersion
      : pgimPreloaderOptions.opfsCacheVersion;
  const resolvedPgimOpfsCacheDirName =
    options.opfsCacheDirName !== undefined
      ? options.opfsCacheDirName
      : pgimPreloaderOptions.opfsCacheDirName;
  const resolvedPgimOpfsValidateWithHead =
    options.opfsValidateWithHead !== undefined
      ? options.opfsValidateWithHead
      : pgimPreloaderOptions.opfsValidateWithHead;
  const resolvedPgimOpfsStorageObj =
    options.opfsStorageObj !== undefined
      ? options.opfsStorageObj
      : pgimPreloaderOptions.opfsStorageObj !== undefined
        ? pgimPreloaderOptions.opfsStorageObj
        : resolvedPgimOpfsCache === true && storageObj
          ? storageObj
          : undefined;
  const pgimPreloader =
    options.pgimPreloader !== undefined
      ? options.pgimPreloader
      : typeof PgimPreloaderCtor === 'function'
        ? new PgimPreloaderCtor({
            ...buildPreloaderStreamOptions(baseStreamConfig, baseStreamExplicitDisabled),
            ...pgimPreloaderOptions,
            ...(resolvedPgimOpfsCache !== undefined ? { opfsCache: resolvedPgimOpfsCache } : {}),
            ...(resolvedPgimOpfsCacheVersion !== undefined
              ? { opfsCacheVersion: resolvedPgimOpfsCacheVersion }
              : {}),
            ...(resolvedPgimOpfsCacheDirName !== undefined
              ? { opfsCacheDirName: resolvedPgimOpfsCacheDirName }
              : {}),
            ...(resolvedPgimOpfsValidateWithHead !== undefined
              ? { opfsValidateWithHead: resolvedPgimOpfsValidateWithHead }
              : {}),
            ...(resolvedPgimOpfsStorageObj !== undefined
              ? { opfsStorageObj: resolvedPgimOpfsStorageObj }
              : {}),
            nowFn: pgimPreloaderOptions.nowFn || now,
            fetchImpl: fetchImpl || undefined,
          })
        : null;

  const BootOrchestratorCtor =
    options.BootOrchestratorCtor ||
    (globalObj.PlayGta6BootOrchestrator && globalObj.PlayGta6BootOrchestrator.BootOrchestrator) ||
    (typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6BootOrchestrator &&
      globalThis.PlayGta6BootOrchestrator.BootOrchestrator) ||
    null;

  let bootOrchestrator = null;
  let ownsBootOrchestrator = false;
  let firstPresentNotified = false;
  let firstVisibleFrameNotified = false;
  let sawExplicitVisibleFrameGate = false;

  function setCanvasBootState(state) {
    if (canvas && typeof canvas === 'object' && canvas.dataset && typeof canvas.dataset === 'object') {
      try {
        canvas.dataset.bootState = state;
      } catch {
        // Ignore read-only mock dataset objects.
      }
    }
  }
  setCanvasBootState('loading');

  function notifyFirstPresent(meta = {}) {
    if (destroyed) {
      return { ok: false, destroyed: true };
    }
    if (firstPresentNotified) {
      return { ok: true, alreadyPresented: true };
    }
    // The boot made it to a frame: a later self-relaunch starts a fresh relaunch budget.
    resetGuestRelaunchCount(windowObj);
    firstPresentNotified = true;
    setCanvasBootState('presenting');
    notifyAudioRingGuestMilestone('first_present');
    if (inputBridge && typeof inputBridge.notifyFirstPresent === 'function') {
      try {
        if (typeof gpuTelemetry.swapchainProbeSupported === 'boolean') {
          inputBridge.notifyFirstPresent({
            swapchainProbeSupported: gpuTelemetry.swapchainProbeSupported,
          });
        } else {
          inputBridge.notifyFirstPresent();
        }
      } catch {
        // InputBridge milestone sync must never interrupt presentation.
      }
    }
    if (bootOrchestrator && typeof bootOrchestrator.notifyFirstPresent === 'function') {
      try {
        const result = bootOrchestrator.notifyFirstPresent(meta);
        gpuTelemetry.loadingAssets = isSplashOverlayHolding();
        syncCrashReporterDiagnostics();
        return result;
      } catch {
        // Fall through to direct overlay dismissal.
      }
    }
    if (bootTracer && typeof bootTracer.markMilestone === 'function') {
      try {
        bootTracer.markMilestone('FIRST_PRESENT', meta);
      } catch {
        // Ignore duplicate/validation errors.
      }
    }
    if (splashOverlay) {
      try {
        if (typeof splashOverlay.onFirstPresent === 'function') {
          splashOverlay.onFirstPresent(meta);
        } else if (typeof splashOverlay.dismiss === 'function') {
          splashOverlay.dismiss(meta);
        }
      } catch {
        // Overlay errors must never interrupt rendering.
      }
    }
    gpuTelemetry.loadingAssets = isSplashOverlayHolding();
    syncCrashReporterDiagnostics();
    return { ok: true, alreadyPresented: false };
  }

  function notifyFirstVisibleFrame(meta = {}) {
    if (destroyed) {
      return { ok: false, destroyed: true };
    }
    if (firstVisibleFrameNotified) {
      return {
        ok: true,
        alreadyVisible: true,
        firstVisibleFrameMs: gpuTelemetry.firstVisibleFrameMs,
        blackLoadingMs: gpuTelemetry.blackLoadingMs,
      };
    }
    firstVisibleFrameNotified = true;
    setCanvasBootState('visible');
    gpuTelemetry.hasVisibleContent = true;
    gpuTelemetry.loadingAssets = false;

    const safeMeta = meta && typeof meta === 'object' && !Array.isArray(meta) ? meta : {};
    if (safeMeta.gpuAdapter !== undefined || safeMeta.adapterInfo !== undefined) {
      recordGpuAdapter(safeMeta.gpuAdapter ?? safeMeta.adapterInfo);
    }
    const providedBlackLoadingMs =
      typeof safeMeta.blackLoadingMs === 'number' &&
      Number.isFinite(safeMeta.blackLoadingMs) &&
      safeMeta.blackLoadingMs >= 0
        ? safeMeta.blackLoadingMs
        : null;
    const providedFirstVisibleFrameMs =
      typeof safeMeta.firstVisibleFrameMs === 'number' &&
      Number.isFinite(safeMeta.firstVisibleFrameMs) &&
      safeMeta.firstVisibleFrameMs >= 0
        ? safeMeta.firstVisibleFrameMs
        : null;

    if (bootTracer && typeof bootTracer.recordFirstVisibleFrame === 'function') {
      try {
        const tracerOpts = {};
        if (
          typeof safeMeta.elapsedMs === 'number' &&
          Number.isFinite(safeMeta.elapsedMs) &&
          safeMeta.elapsedMs >= 0
        ) {
          tracerOpts.elapsedMs = safeMeta.elapsedMs;
        } else if (providedBlackLoadingMs !== null && typeof bootTracer.getSummary === 'function') {
          try {
            const preSummary = bootTracer.getSummary();
            if (
              preSummary &&
              typeof preSummary.firstPresentMs === 'number' &&
              Number.isFinite(preSummary.firstPresentMs) &&
              preSummary.firstPresentMs >= 0
            ) {
              tracerOpts.elapsedMs = preSummary.firstPresentMs + providedBlackLoadingMs;
            } else if (providedFirstVisibleFrameMs !== null) {
              tracerOpts.elapsedMs = providedFirstVisibleFrameMs;
            }
          } catch {
            // Fall through to tracer default clock.
          }
        } else if (providedFirstVisibleFrameMs !== null) {
          tracerOpts.elapsedMs = providedFirstVisibleFrameMs;
        }
        if (typeof safeMeta.source === 'string' && safeMeta.source.trim().length > 0) {
          tracerOpts.source = sanitizeDiagnosticText(safeMeta.source, 64);
        }
        if (safeMeta.detail !== undefined) {
          tracerOpts.detail = safeMeta.detail;
        }
        const rec = bootTracer.recordFirstVisibleFrame(tracerOpts);
        if (rec && typeof rec === 'object') {
          if (typeof rec.firstVisibleFrameMs === 'number' && Number.isFinite(rec.firstVisibleFrameMs)) {
            gpuTelemetry.firstVisibleFrameMs = rec.firstVisibleFrameMs;
          }
          if (typeof rec.blackLoadingMs === 'number' && Number.isFinite(rec.blackLoadingMs)) {
            gpuTelemetry.blackLoadingMs = rec.blackLoadingMs;
          }
        }
      } catch {
        // Tracer errors must never interrupt rendering.
      }
    } else if (bootTracer && typeof bootTracer.markMilestone === 'function') {
      try {
        bootTracer.markMilestone('FIRST_VISIBLE_FRAME', safeMeta);
      } catch {
        // Ignore stub tracers without optional milestones.
      }
    }

    if (
      (gpuTelemetry.firstVisibleFrameMs === null || gpuTelemetry.blackLoadingMs === null) &&
      bootTracer &&
      typeof bootTracer.getSummary === 'function'
    ) {
      try {
        const summary = bootTracer.getSummary();
        if (summary && typeof summary === 'object') {
          if (
            gpuTelemetry.firstVisibleFrameMs === null &&
            typeof summary.firstVisibleFrameMs === 'number' &&
            Number.isFinite(summary.firstVisibleFrameMs)
          ) {
            gpuTelemetry.firstVisibleFrameMs = summary.firstVisibleFrameMs;
          }
          if (
            gpuTelemetry.blackLoadingMs === null &&
            typeof summary.blackLoadingMs === 'number' &&
            Number.isFinite(summary.blackLoadingMs)
          ) {
            gpuTelemetry.blackLoadingMs = summary.blackLoadingMs;
          }
        }
      } catch {
        // Ignore tracer errors.
      }
    }

    if (gpuTelemetry.firstVisibleFrameMs === null && providedFirstVisibleFrameMs !== null) {
      gpuTelemetry.firstVisibleFrameMs = providedFirstVisibleFrameMs;
    }
    if (gpuTelemetry.blackLoadingMs === null && providedBlackLoadingMs !== null) {
      gpuTelemetry.blackLoadingMs = providedBlackLoadingMs;
    }

    if (splashOverlay) {
      try {
        const splashMeta = { ...safeMeta };
        delete splashMeta._deferObserverForward;
        if (gpuTelemetry.firstVisibleFrameMs !== null) {
          splashMeta.firstVisibleFrameMs = gpuTelemetry.firstVisibleFrameMs;
        }
        if (gpuTelemetry.blackLoadingMs !== null) {
          splashMeta.blackLoadingMs = gpuTelemetry.blackLoadingMs;
        }
        if (gpuTelemetry.gpuAdapter !== null) {
          splashMeta.gpuAdapter = gpuTelemetry.gpuAdapter;
        }
        if (typeof splashOverlay.notifyFirstVisibleFrame === 'function') {
          splashOverlay.notifyFirstVisibleFrame(splashMeta);
        } else if (typeof splashOverlay.dismiss === 'function') {
          splashOverlay.dismiss(splashMeta);
        }
      } catch {
        // Overlay errors must never interrupt rendering.
      }
    }

    if (inputBridge && typeof inputBridge.notifyFirstVisibleFrame === 'function') {
      try {
        inputBridge.notifyFirstVisibleFrame();
      } catch {
        // InputBridge milestone sync must never interrupt rendering.
      }
    }

    if (safeMeta._deferObserverForward !== true) {
      forwardGpuStatsToObservers(buildGpuStatsPayload(), { allowHudMetricsFallback: false });
    }
    syncCrashReporterDiagnostics();

    return {
      ok: true,
      alreadyVisible: false,
      firstVisibleFrameMs: gpuTelemetry.firstVisibleFrameMs,
      blackLoadingMs: gpuTelemetry.blackLoadingMs,
    };
  }

  function syncPerfPhaseForFastBootMilestone(milestone) {
    if (!perfMonitor || typeof perfMonitor.setPhase !== 'function') {
      return;
    }
    let nextPhase = null;
    if (milestone === 'START_SCREEN' || milestone === 'MAIN_MENU') {
      nextPhase = PERF_PHASES.MENU;
    } else if (milestone === 'IN_GAME') {
      nextPhase = PERF_PHASES.GAMEPLAY;
    }
    if (!nextPhase) {
      return;
    }
    try {
      perfMonitor.setPhase(nextPhase);
    } catch {
      // Telemetry phase updates must never disrupt boot.
    }
  }

  function syncFastBootMilestonesFromTracer() {
    if (!bootTracer || typeof bootTracer.getSummary !== 'function') {
      return;
    }
    try {
      const summary = bootTracer.getSummary();
      if (!summary || typeof summary !== 'object') {
        return;
      }
      const pairs = [
        ['START_SCREEN', summary.bootToStartScreenMs],
        ['INTRO_VIDEO', summary.bootToIntroVideoMs],
        ['MAIN_MENU', summary.bootToMainMenuMs],
        ['IN_GAME', summary.bootToInGameMs],
      ];
      for (let i = 0; i < pairs.length; i += 1) {
        const [name, ms] = pairs[i];
        if (typeof ms === 'number' && Number.isFinite(ms) && ms >= 0) {
          const wasUnset = fastBootMilestones[name] === null;
          fastBootMilestones[name] = ms;
          if (wasUnset) {
            syncPerfPhaseForFastBootMilestone(name);
          }
        }
      }
    } catch {
      // Ignore tracer summary errors.
    }
  }

  function recordFastBootMilestone(rawMilestone, meta = {}) {
    if (destroyed) {
      return { ok: false, destroyed: true };
    }
    const milestone = normalizeFastBootMilestone(rawMilestone);
    if (!milestone) {
      return null;
    }
    const safeMeta = meta && typeof meta === 'object' && !Array.isArray(meta) ? meta : {};
    const overwrite = safeMeta.overwrite === true;
    if (fastBootMilestones[milestone] !== null && !overwrite) {
      return Object.freeze({
        ok: true,
        milestone,
        elapsedMs: fastBootMilestones[milestone],
        alreadyRecorded: true,
      });
    }

    const explicitElapsedMs = sanitizeNonNegativeFiniteNumber(safeMeta.elapsedMs);
    const tracerOpts = {};
    if (explicitElapsedMs !== null) {
      tracerOpts.elapsedMs = explicitElapsedMs;
    }
    if (overwrite) {
      tracerOpts.overwrite = true;
    }
    if (typeof safeMeta.source === 'string' && safeMeta.source.trim().length > 0) {
      tracerOpts.source = sanitizeDiagnosticText(safeMeta.source, 64);
    }
    if (safeMeta.detail !== undefined) {
      tracerOpts.detail = safeMeta.detail;
    }

    let recordedMs = null;
    const methodName = FAST_BOOT_TRACER_METHODS[milestone];
    if (bootTracer && methodName && typeof bootTracer[methodName] === 'function') {
      try {
        const rec = bootTracer[methodName](tracerOpts);
        if (rec && typeof rec === 'object') {
          const candidate =
            rec.elapsedMs ??
            rec.bootToStartScreenMs ??
            rec.bootToIntroVideoMs ??
            rec.bootToMainMenuMs ??
            rec.bootToInGameMs;
          recordedMs = sanitizeNonNegativeFiniteNumber(candidate);
        }
      } catch {
        // Tracer errors must never interrupt boot.
      }
    } else if (bootTracer && typeof bootTracer.markMilestone === 'function') {
      try {
        const rec = bootTracer.markMilestone(milestone, safeMeta);
        if (rec && typeof rec === 'object') {
          recordedMs = sanitizeNonNegativeFiniteNumber(rec.elapsedMs);
        }
      } catch {
        // Ignore stub tracers without fast-boot milestones.
      }
    }

    if (recordedMs === null) {
      syncFastBootMilestonesFromTracer();
      recordedMs = fastBootMilestones[milestone];
    }
    if (recordedMs === null) {
      if (explicitElapsedMs !== null) {
        recordedMs = explicitElapsedMs;
      } else {
        const fallbackElapsed = now() - bootStartedAtMs;
        recordedMs = Number.isFinite(fallbackElapsed) && fallbackElapsed >= 0 ? fallbackElapsed : 0;
      }
    }

    fastBootMilestones[milestone] = recordedMs;
    syncPerfPhaseForFastBootMilestone(milestone);
    const mappedSessionPhase = FAST_BOOT_MILESTONE_TO_SESSION_PHASE[milestone];
    if (mappedSessionPhase && session && typeof session.recordPhase === 'function') {
      try {
        session.recordPhase(mappedSessionPhase);
      } catch {
        // Session phase transitions must never disrupt fast-boot milestone recording.
      }
    }
    notifyAudioRingGuestMilestone(milestone.toLowerCase());

    return Object.freeze({
      ok: true,
      milestone,
      elapsedMs: recordedMs,
      alreadyRecorded: false,
    });
  }

  function ingestAudioRingLogLine(text) {
    if (typeof text !== 'string' || text.length === 0) {
      return false;
    }
    const match = AUDIO_RING_ATTACHED_LOG_RE.exec(text);
    if (!match) {
      return false;
    }
    const parsedFlag = parseBooleanQueryFlag(match[1]);
    if (parsedFlag === null) {
      return false;
    }
    audioRingTelemetry.audioRingAttached = parsedFlag;
    const attemptsMatch = AUDIO_RING_ATTEMPTS_LOG_RE.exec(text);
    const retriesMatch = AUDIO_RING_RETRIES_LOG_RE.exec(text);
    if (attemptsMatch) {
      const parsedAttempts = toNonNegativeInt(attemptsMatch[1], null);
      if (parsedAttempts !== null) {
        audioRingTelemetry.audioRingAttachAttempts = Math.max(
          audioRingTelemetry.audioRingAttachAttempts,
          parsedAttempts
        );
      }
    } else if (retriesMatch) {
      const parsedRetries = toNonNegativeInt(retriesMatch[1], null);
      if (parsedRetries !== null) {
        audioRingTelemetry.audioRingAttachAttempts = Math.max(
          audioRingTelemetry.audioRingAttachAttempts,
          parsedRetries + 1
        );
      }
    } else if (parsedFlag && audioRingTelemetry.audioRingAttachAttempts === 0) {
      audioRingTelemetry.audioRingAttachAttempts = 1;
    }
    if (parsedFlag) {
      audioRingTelemetry.audioRingLastReason = 'ok';
    }
    return true;
  }

  function feedBootLog(line) {
    if (destroyed) {
      return;
    }
    const text = typeof line === 'string' ? line : '';
    if (text) {
      ingestAudioRingLogLine(text);
    }
    const extractedAdapter = text ? extractGpuAdapterFromLogLine(text) : null;
    if (extractedAdapter) {
      recordGpuAdapter(extractedAdapter);
    }
    const logIndicatesVisibleFrame =
      Boolean(text) &&
      FIRST_VISIBLE_FRAME_LOG_RE.test(text) &&
      !VISIBLE_FRAME_LOG_FAILURE_RE.test(text);
    const detectedFastBootMilestones = text ? detectFastBootMilestonesFromLog(text) : [];

    const recordDetectedFastBootMilestones = () => {
      syncFastBootMilestonesFromTracer();
      for (let i = 0; i < detectedFastBootMilestones.length; i += 1) {
        const milestone = detectedFastBootMilestones[i];
        if (fastBootMilestones[milestone] === null) {
          recordFastBootMilestone(milestone, { source: 'boot_log' });
        }
      }
    };

    if (bootOrchestrator && typeof bootOrchestrator.feedRuntimeLog === 'function') {
      try {
        bootOrchestrator.feedRuntimeLog(line);
        if (logIndicatesVisibleFrame) {
          notifyFirstVisibleFrame({ source: 'boot_log' });
        }
        recordDetectedFastBootMilestones();
        return;
      } catch {
        // Fall back to direct tracer update.
      }
    }
    if (
      bootTracer &&
      (typeof bootTracer.ingestLogLine === 'function' || typeof bootTracer.ingestLog === 'function')
    ) {
      try {
        if (typeof bootTracer.ingestLogLine === 'function') {
          bootTracer.ingestLogLine(line);
        } else {
          bootTracer.ingestLog(line);
        }
        const summary = bootTracer.getSummary();
        if (splashOverlay && typeof splashOverlay.updateBootState === 'function') {
          splashOverlay.updateBootState(summary);
        }
        if (summary && summary.readyForSplash) {
          notifyFirstPresent({ source: 'boot_log' });
        }
        if (
          logIndicatesVisibleFrame ||
          (summary &&
            typeof summary.firstVisibleFrameMs === 'number' &&
            Number.isFinite(summary.firstVisibleFrameMs))
        ) {
          notifyFirstVisibleFrame({ source: 'boot_log' });
        }
        recordDetectedFastBootMilestones();
      } catch {
        // Tracer errors must never interrupt runtime logs.
      }
    } else {
      if (logIndicatesVisibleFrame) {
        notifyFirstVisibleFrame({ source: 'boot_log' });
      }
      recordDetectedFastBootMilestones();
    }
  }

  // Shared mutable lifecycle state managed by the app handle.
  const smokeDemoOptions =
    options.smokeDemoOptions && typeof options.smokeDemoOptions === 'object'
      ? options.smokeDemoOptions
      : {};
  const maxDevicePixelRatio = isPositiveFinite(options.maxDevicePixelRatio)
    ? options.maxDevicePixelRatio
    : isPositiveFinite(smokeDemoOptions.maxDevicePixelRatio)
      ? smokeDemoOptions.maxDevicePixelRatio
      : DEFAULT_MAX_DEVICE_PIXEL_RATIO;
  const swapchainOptions =
    options.swapchainOptions &&
    typeof options.swapchainOptions === 'object' &&
    !Array.isArray(options.swapchainOptions)
      ? { ...options.swapchainOptions }
      : null;
  const initialViewport = resolveViewportMetrics(canvas, windowObj, {
    ...smokeDemoOptions,
    maxDevicePixelRatio,
  });
  let viewportWidth = initialViewport.width;
  let viewportHeight = initialViewport.height;
  let viewportDpr = initialViewport.dpr;
  let canvasTransferredToOffscreen = false;
  let transferredOffscreenCanvas = null;

  let activeSmokeRunner = null;
  let activeWasmHandle = null;
  let activeWasmWorker = null;
  let destroyed = false;
  let destroyPromise = null;
  const externalWorkerCleanups = [];

  // W308: Lazy _playgta6_audio_ring() auto-retry wiring when audioWorker is present.
  const externalAudioWorker = options.audioWorker || null;
  let externalAudioWorkerReady =
    externalAudioWorker !== null ? options.audioWorkerReady !== false : false;
  const audioRingRetryConfig =
    options.audioRingRetry && typeof options.audioRingRetry === 'object'
      ? options.audioRingRetry
      : {};
  const resolvedSetTimeout =
    options.setTimeout ||
    audioRingRetryConfig.setTimeoutFn ||
    (windowObj && typeof windowObj.setTimeout === 'function'
      ? windowObj.setTimeout.bind(windowObj)
      : typeof setTimeout === 'function'
        ? (fn, ms) => setTimeout(fn, ms)
        : null);
  const resolvedClearTimeout =
    options.clearTimeout ||
    audioRingRetryConfig.clearTimeoutFn ||
    (windowObj && typeof windowObj.clearTimeout === 'function'
      ? windowObj.clearTimeout.bind(windowObj)
      : typeof clearTimeout === 'function'
        ? (id) => clearTimeout(id)
        : null);
  const scheduleAudioRingRetryFn =
    options.scheduleAudioRingRetry === null
      ? null
      : typeof options.scheduleAudioRingRetry === 'function'
        ? options.scheduleAudioRingRetry
        : (globalObj.PlayGta6Loader &&
            typeof globalObj.PlayGta6Loader.scheduleAudioRingRetry === 'function' &&
            globalObj.PlayGta6Loader.scheduleAudioRingRetry) ||
          (typeof globalThis !== 'undefined' &&
            globalThis.PlayGta6Loader &&
            typeof globalThis.PlayGta6Loader.scheduleAudioRingRetry === 'function' &&
            globalThis.PlayGta6Loader.scheduleAudioRingRetry) ||
          null;
  const attachRuntimeAudioRingFn =
    options.attachRuntimeAudioRing === null
      ? null
      : typeof options.attachRuntimeAudioRing === 'function'
        ? options.attachRuntimeAudioRing
        : (globalObj.PlayGta6Loader &&
            typeof globalObj.PlayGta6Loader.attachRuntimeAudioRing === 'function' &&
            globalObj.PlayGta6Loader.attachRuntimeAudioRing) ||
          (typeof globalThis !== 'undefined' &&
            globalThis.PlayGta6Loader &&
            typeof globalThis.PlayGta6Loader.attachRuntimeAudioRing === 'function' &&
            globalThis.PlayGta6Loader.attachRuntimeAudioRing) ||
          null;

  let boundAudioRingModule =
    options.wasmModule && typeof options.wasmModule === 'object' ? options.wasmModule : null;
  let audioRingController =
    options.audioRingController && typeof options.audioRingController === 'object'
      ? options.audioRingController
      : null;
  let ownsAudioRingController = false;

  function resolveExternalWasmModule(modOverride) {
    if (modOverride && typeof modOverride === 'object') {
      boundAudioRingModule = modOverride;
      return modOverride;
    }
    if (typeof options.getWasmModule === 'function') {
      try {
        const dynamicMod = options.getWasmModule();
        if (dynamicMod && typeof dynamicMod === 'object') {
          boundAudioRingModule = dynamicMod;
          return dynamicMod;
        }
      } catch {
        // Ignore dynamic module getter errors.
      }
    }
    return boundAudioRingModule;
  }

  function resolveAudioRingPtrFromModule(mod) {
    if (!mod || typeof mod !== 'object') {
      return { hasExport: false, ptr: null };
    }
    const candidates = [
      mod._playgta6_audio_ring,
      mod.playgta6_audio_ring,
      mod.asm && mod.asm._playgta6_audio_ring,
      mod.asm && mod.asm.playgta6_audio_ring,
      mod.wasmExports && mod.wasmExports._playgta6_audio_ring,
      mod.wasmExports && mod.wasmExports.playgta6_audio_ring,
    ];
    for (let i = 0; i < candidates.length; i += 1) {
      if (typeof candidates[i] === 'function') {
        try {
          return { hasExport: true, ptr: candidates[i].call(mod) };
        } catch {
          return { hasExport: true, ptr: null };
        }
      }
    }
    return { hasExport: false, ptr: null };
  }

  function createFallbackAudioRingController() {
    const retryIntervalMs =
      sanitizePositiveInt(
        options.audioRingRetryIntervalMs !== undefined
          ? options.audioRingRetryIntervalMs
          : audioRingRetryConfig.retryIntervalMs
      ) ?? DEFAULT_AUDIO_RING_RETRY_INTERVAL_MS;
    const maxRetries =
      toNonNegativeInt(
        options.audioRingMaxRetries !== undefined
          ? options.audioRingMaxRetries
          : audioRingRetryConfig.maxRetries,
        null
      ) ?? DEFAULT_AUDIO_RING_MAX_RETRIES;
    const autoRetry =
      options.audioRingAutoRetry !== false && audioRingRetryConfig.autoRetry !== false;
    const retryOnNoExport =
      options.audioRingRetryOnNoExport === true || audioRingRetryConfig.retryOnNoExport === true;

    let timerId = null;
    let retries = 0;
    let attempts = 0;
    let attached = false;
    let lastReason = null;
    let ctrlDestroyed = false;

    const clearTimer = () => {
      if (timerId !== null) {
        const id = timerId;
        timerId = null;
        if (typeof resolvedClearTimeout === 'function') {
          try {
            resolvedClearTimeout(id);
          } catch {
            // Ignore timer clear errors.
          }
        }
      }
    };

    const scheduleNext = () => {
      if (
        ctrlDestroyed ||
        destroyed ||
        attached ||
        !autoRetry ||
        timerId !== null ||
        retries >= maxRetries ||
        typeof resolvedSetTimeout !== 'function'
      ) {
        return false;
      }
      timerId = resolvedSetTimeout(() => {
        timerId = null;
        if (ctrlDestroyed || destroyed || attached) {
          return;
        }
        retries += 1;
        attempt(null, null);
      }, retryIntervalMs);
      return true;
    };

    const attempt = (modOverride, attachOpts) => {
      if (ctrlDestroyed || destroyed) {
        return { attached: false, reason: 'destroyed' };
      }
      if (attached) {
        return { attached: true, alreadyAttached: true, reason: 'ok' };
      }
      attempts += 1;
      audioRingTelemetry.audioRingAttachAttempts = Math.max(
        audioRingTelemetry.audioRingAttachAttempts + 1,
        attempts
      );
      const targetMod = resolveExternalWasmModule(modOverride);
      if (!targetMod) {
        lastReason = 'no_export';
        audioRingTelemetry.audioRingLastReason = lastReason;
        if (typeof options.getWasmModule === 'function' || retryOnNoExport) {
          scheduleNext();
        }
        return { attached: false, reason: lastReason };
      }
      if (!externalAudioWorker || typeof externalAudioWorker.postMessage !== 'function' || !externalAudioWorkerReady) {
        lastReason = 'worker_not_ready';
        audioRingTelemetry.audioRingLastReason = lastReason;
        scheduleNext();
        return { attached: false, reason: lastReason };
      }
      if (attachRuntimeAudioRingFn) {
        try {
          const res = attachRuntimeAudioRingFn(targetMod, externalAudioWorker, {
            ...(options.audioRingMemory !== undefined ? { memory: options.audioRingMemory } : {}),
            ...(options.audioRingPointerSize !== undefined
              ? { pointerSize: options.audioRingPointerSize }
              : {}),
            ...(typeof options.audioRingAutoPump === 'boolean'
              ? { autoPump: options.audioRingAutoPump }
              : {}),
            ...(attachOpts && typeof attachOpts === 'object' ? attachOpts : {}),
          });
          if (res && res.attached) {
            attached = true;
            lastReason = 'ok';
            clearTimer();
            audioRingTelemetry.audioRingAttached = true;
            audioRingTelemetry.audioRingLastReason = 'ok';
            return res;
          }
          lastReason = sanitizeDiagnosticText((res && (res.statusReason || res.reason)) || 'null_ptr', 64);
          audioRingTelemetry.audioRingLastReason = lastReason;
          if (lastReason !== 'no_export' || retryOnNoExport) {
            scheduleNext();
          }
          return res || { attached: false, reason: lastReason };
        } catch {
          lastReason = 'invalid_struct';
          audioRingTelemetry.audioRingLastReason = lastReason;
          scheduleNext();
          return { attached: false, reason: lastReason };
        }
      }
      const probed = resolveAudioRingPtrFromModule(targetMod);
      if (!probed.hasExport) {
        lastReason = 'no_export';
        audioRingTelemetry.audioRingLastReason = lastReason;
        if (retryOnNoExport) {
          scheduleNext();
        }
        return { attached: false, reason: lastReason };
      }
      if (probed.ptr === null || probed.ptr === undefined || probed.ptr === 0 || probed.ptr === 0n) {
        lastReason = 'null_ptr';
        audioRingTelemetry.audioRingLastReason = lastReason;
        scheduleNext();
        return { attached: false, reason: lastReason };
      }
      try {
        externalAudioWorker.postMessage({
          type: 'attach-guest-pcm-ring',
          attachGuestPcmRing: true,
          audioRingPtr: probed.ptr,
          ...(options.audioRingMemory !== undefined ? { buffer: options.audioRingMemory } : {}),
        });
        attached = true;
        lastReason = 'ok';
        clearTimer();
        audioRingTelemetry.audioRingAttached = true;
        audioRingTelemetry.audioRingLastReason = 'ok';
        return { attached: true, reason: 'ok', audioRingPtr: probed.ptr };
      } catch {
        lastReason = 'worker_not_ready';
        audioRingTelemetry.audioRingLastReason = lastReason;
        scheduleNext();
        return { attached: false, reason: lastReason };
      }
    };

    return {
      attach: (modOverride, attachOpts) => attempt(modOverride, attachOpts),
      probeNow: () => attempt(null, null),
      onAudioWorkerReady() {
        externalAudioWorkerReady = true;
        if (!attached) {
          return attempt(null, null);
        }
        return { attached: true, alreadyAttached: true, reason: 'ok' };
      },
      notifyGuestMilestone() {
        if (!attached) {
          return attempt(null, null);
        }
        return { attached: true, alreadyAttached: true, reason: 'ok' };
      },
      getTelemetry() {
        return {
          audioRingAttached: attached,
          audioRingAttachAttempts: attempts,
          audioRingRetryCount: retries,
          audioRingLastReason: lastReason,
          audioRingRetryPending: timerId !== null,
          audioRingRetryExhausted: !attached && retries >= maxRetries,
          audioRingMaxRetries: maxRetries,
          audioRingRetryIntervalMs: retryIntervalMs,
          destroyed: ctrlDestroyed,
        };
      },
      stop: clearTimer,
      reset() {
        clearTimer();
        attached = false;
        retries = 0;
        lastReason = null;
      },
      destroy() {
        ctrlDestroyed = true;
        clearTimer();
      },
    };
  }

  function ensureAudioRingController(modOverride = null, attachOpts = undefined) {
    if (destroyed) {
      return { controller: null, probedOnCreate: false };
    }
    if (modOverride && typeof modOverride === 'object') {
      boundAudioRingModule = modOverride;
    }
    if (audioRingController) {
      return { controller: audioRingController, probedOnCreate: false };
    }
    if (!externalAudioWorker) {
      return { controller: null, probedOnCreate: false };
    }
    const hasModuleSource = Boolean(boundAudioRingModule || typeof options.getWasmModule === 'function');
    if (!hasModuleSource) {
      return { controller: null, probedOnCreate: false };
    }
    if (scheduleAudioRingRetryFn) {
      const retryIntervalMs =
        sanitizePositiveInt(
          options.audioRingRetryIntervalMs !== undefined
            ? options.audioRingRetryIntervalMs
            : audioRingRetryConfig.retryIntervalMs
        ) ?? DEFAULT_AUDIO_RING_RETRY_INTERVAL_MS;
      const maxRetries =
        toNonNegativeInt(
          options.audioRingMaxRetries !== undefined
            ? options.audioRingMaxRetries
            : audioRingRetryConfig.maxRetries,
          null
        ) ?? DEFAULT_AUDIO_RING_MAX_RETRIES;
      const autoRetry =
        options.audioRingAutoRetry !== false && audioRingRetryConfig.autoRetry !== false;
      const retryOnNoExport =
        options.audioRingRetryOnNoExport === true || audioRingRetryConfig.retryOnNoExport === true;
      audioRingController = scheduleAudioRingRetryFn({
        module: boundAudioRingModule,
        getModule: () => resolveExternalWasmModule(null),
        audioWorker: externalAudioWorker,
        isAudioWorkerReady: () => externalAudioWorkerReady,
        ...(options.audioRingMemory !== undefined ? { memory: options.audioRingMemory } : {}),
        ...(options.audioRingPointerSize !== undefined
          ? { pointerSize: options.audioRingPointerSize }
          : {}),
        ...(typeof options.audioRingAutoPump === 'boolean'
          ? { autoPump: options.audioRingAutoPump }
          : {}),
        ...(attachOpts && typeof attachOpts === 'object' ? { attachOptions: attachOpts } : {}),
        retryIntervalMs,
        maxRetries,
        autoRetry,
        retryOnNoExport,
        ...(typeof resolvedSetTimeout === 'function' ? { setTimeoutFn: resolvedSetTimeout } : {}),
        ...(typeof resolvedClearTimeout === 'function'
          ? { clearTimeoutFn: resolvedClearTimeout }
          : {}),
        onAttempt(res, telemetry) {
          audioRingTelemetry.audioRingAttachAttempts += 1;
          if (res && res.attached) {
            audioRingTelemetry.audioRingAttached = true;
          } else if (telemetry && typeof telemetry.audioRingAttached === 'boolean') {
            audioRingTelemetry.audioRingAttached = telemetry.audioRingAttached;
          }
          const reason =
            (res && (res.statusReason || res.reason)) ||
            (telemetry && telemetry.audioRingLastReason) ||
            null;
          if (typeof reason === 'string' && reason.length > 0) {
            audioRingTelemetry.audioRingLastReason = sanitizeDiagnosticText(reason, 64);
          }
        },
        onAttached(res, telemetry) {
          audioRingTelemetry.audioRingAttached = true;
          audioRingTelemetry.audioRingLastReason = 'ok';
          const retryCount =
            telemetry && typeof telemetry.audioRingRetryCount === 'number'
              ? toSafeNonNegativeInt(telemetry.audioRingRetryCount, 0)
              : 0;
          audioRingTelemetry.audioRingAttachAttempts = Math.max(
            audioRingTelemetry.audioRingAttachAttempts,
            retryCount + 1
          );
        },
      });
      ownsAudioRingController = true;
      return { controller: audioRingController, probedOnCreate: true };
    }
    audioRingController = createFallbackAudioRingController();
    ownsAudioRingController = true;
    return { controller: audioRingController, probedOnCreate: false };
  }

  function attachAudioRing(modOverride = null, attachOpts = undefined) {
    if (destroyed) {
      return { attached: false, destroyed: true };
    }
    const { controller: ctrl, probedOnCreate } = ensureAudioRingController(modOverride, attachOpts);
    if (!ctrl) {
      return null;
    }
    if (probedOnCreate) {
      return typeof ctrl.getTelemetry === 'function' ? ctrl.getTelemetry() : { attached: audioRingTelemetry.audioRingAttached };
    }
    try {
      if (typeof ctrl.attach === 'function') {
        return ctrl.attach(modOverride, attachOpts);
      }
      if (typeof ctrl.probeNow === 'function') {
        return ctrl.probeNow();
      }
    } catch {
      // Audio-ring attachment probes must never throw into callers.
    }
    return null;
  }

  function notifyAudioRingGuestMilestone(milestoneName) {
    if (destroyed || audioRingTelemetry.audioRingAttached) {
      return;
    }
    const ctrl = audioRingController || ensureAudioRingController(null).controller;
    if (!ctrl) {
      return;
    }
    try {
      if (typeof ctrl.notifyGuestMilestone === 'function') {
        ctrl.notifyGuestMilestone(milestoneName);
      } else if (typeof ctrl.probeNow === 'function') {
        ctrl.probeNow();
      }
    } catch {
      // Milestone audio-ring retry triggers must never disrupt boot.
    }
  }

  for (const workerCandidate of [options.gpuWorker, options.audioWorker, options.ioWorker]) {
    if (isEventTargetLike(workerCandidate)) {
      const isAudioCandidate = workerCandidate === externalAudioWorker;
      const onExternalWorkerMessage = (ev) => {
        if (destroyed) {
          return;
        }
        const data = ev && typeof ev === 'object' && ev.data !== undefined ? ev.data : ev;
        if (isAudioCandidate && data && typeof data === 'object') {
          if (data.ready === true) {
            externalAudioWorkerReady = true;
            const ensured = audioRingController
              ? { controller: audioRingController, probedOnCreate: false }
              : ensureAudioRingController(null);
            const ctrl = ensured.controller;
            if (ctrl && !ensured.probedOnCreate) {
              try {
                if (typeof ctrl.onAudioWorkerReady === 'function') {
                  ctrl.onAudioWorkerReady(externalAudioWorker);
                } else if (typeof ctrl.probeNow === 'function') {
                  ctrl.probeNow();
                }
              } catch {
                // Non-fatal audio worker ready probe failure.
              }
            }
          }
          if (
            data.guestRingAttached !== undefined ||
            data.audioRingAttached !== undefined ||
            data.attached !== undefined
          ) {
            ingestAudioRingTelemetry(data, { allowGenericAttached: true });
          }
        }
        handleWorkerMessage(data);
      };
      try {
        workerCandidate.addEventListener('message', onExternalWorkerMessage);
        externalWorkerCleanups.push(() => {
          workerCandidate.removeEventListener('message', onExternalWorkerMessage);
        });
      } catch {
        // Ignore worker mocks that do not support message listeners.
      }
    }
  }

  hookBootTracerForPanel();

  const resolvedRaf =
    options.raf ||
    smokeDemoOptions.raf ||
    (windowObj && typeof windowObj.requestAnimationFrame === 'function'
      ? windowObj.requestAnimationFrame.bind(windowObj)
      : undefined);
  const resolvedCancelRaf =
    options.cancelRaf ||
    smokeDemoOptions.cancelRaf ||
    (windowObj && typeof windowObj.cancelAnimationFrame === 'function'
      ? windowObj.cancelAnimationFrame.bind(windowObj)
      : undefined);

  // 6a. Wasm64 runtime launcher (spawns web/loader.js worker or injected createWasmWorker).
  async function startWasmRuntime(launchContext) {
    if (destroyed) {
      throw new Error('Cannot start Wasm64 runtime on a destroyed PlayGta6AppHandle');
    }

    resetWasmPerfTracking();

    // Re-measure live layout dimensions right before launching the GPU worker so late
    // CSS layout / async preload resizes never lock in a stale initial backing store.
    if (isPositiveFinite(canvas && canvas.clientWidth) && isPositiveFinite(canvas && canvas.clientHeight)) {
      viewportWidth = canvas.clientWidth;
      viewportHeight = canvas.clientHeight;
      if (isPositiveFinite(windowObj && windowObj.devicePixelRatio)) {
        viewportDpr = Math.min(windowObj.devicePixelRatio, maxDevicePixelRatio);
      }
    }

    const pixelWidth = Math.max(1, Math.round(viewportWidth * viewportDpr));
    const pixelHeight = Math.max(1, Math.round(viewportHeight * viewportDpr));
    if (launchContext && launchContext.serverStatus) {
      recordDiscoveredBootAssets(launchContext.serverStatus);
    }
    const liveBoundsBeforeLaunch = resolveDisplayBounds({
      windowObj,
      screenObj: options.screenObj,
      maxTextureDimension2D: options.maxTextureDimension2D,
      displayBounds: options.displayBounds,
      perfProfileApi: options.perfProfileApi,
    });
    if (liveBoundsBeforeLaunch) {
      activeDisplayBounds = liveBoundsBeforeLaunch;
      activePerfProfile = toCloneablePerfProfile(activePerfProfile, activeDisplayBounds);
    }
    lastSentDisplayBoundsKey = displayBoundsKey(activeDisplayBounds);
    const bootForwardFields = resolveBootForwardFields(launchContext);
    const workerPerfInitPayloads = buildWorkerPerfInitPayloads(
      activePerfProfile,
      activeDisplayBounds
    );

    let worker;
    if (createWasmWorker) {
      worker = await createWasmWorker({
        url: loaderUrl,
        canvas,
        width: pixelWidth,
        height: pixelHeight,
        cssWidth: viewportWidth,
        cssHeight: viewportHeight,
        dpr: viewportDpr,
        inputSab: inputBridge.sab,
        audioSab: audioEngine.sab,
        dataBase,
        ...bootForwardFields,
        launchContext,
      });
    } else {
      const WorkerCtor =
        (windowObj && typeof windowObj.Worker === 'function' && windowObj.Worker) ||
        (typeof Worker === 'function' ? Worker : null);
      if (!WorkerCtor) {
        throw new Error('Web Worker API is unavailable; cannot start Wasm64 loader worker');
      }
      worker = new WorkerCtor(loaderUrl);
    }

    if (!worker || typeof worker !== 'object') {
      throw new Error('Wasm64 worker factory did not return a valid Worker instance');
    }

    if (destroyed) {
      if (typeof worker.terminate === 'function') {
        worker.terminate();
      } else if (typeof worker.stop === 'function') {
        await worker.stop();
      }
      throw new Error('PlayGta6AppHandle was destroyed while starting Wasm64 worker');
    }

    let stopped = false;

    const onWorkerError = (ev) => {
      const errObj =
        ev && ev.error !== undefined && ev.error !== null
          ? ev.error
          : ev instanceof Error
            ? ev
            : new Error(
                ev && typeof ev.message === 'string' && ev.message.length > 0
                  ? ev.message
                  : 'Wasm64 loader worker error'
              );
      crashReporter.recordError('worker_error', errObj, {
        source: 'wasmWorker',
        filename: ev && typeof ev.filename === 'string' ? ev.filename : undefined,
      });
      if (hud && typeof hud.setStatus === 'function') {
        try {
          hud.setStatus('Worker error', 'error');
        } catch {
          // Ignore HUD errors.
        }
      }
      if (hud && typeof hud.appendLog === 'function') {
        try {
          hud.appendLog(`loader worker error: ${errObj.message || errObj}`, 'error');
        } catch {
          // Ignore HUD errors.
        }
      }
    };

    const onMessageError = (ev) => {
      crashReporter.recordError(
        'worker_error',
        new Error('Wasm64 loader worker message deserialization error'),
        { source: 'wasmWorker', eventType: ev?.type || 'messageerror' }
      );
    };

    const onWorkerMessage = (ev) => {
      if (stopped || destroyed) {
        return;
      }
      const data = ev && typeof ev === 'object' && ev.data !== undefined ? ev.data : ev;
      handleWorkerMessage(data);
    };

    let usedAddEventListener = false;
    let prevOnError = null;
    if (isEventTargetLike(worker)) {
      usedAddEventListener = true;
      worker.addEventListener('error', onWorkerError);
      worker.addEventListener('messageerror', onMessageError);
    } else {
      prevOnError = worker.onerror ?? null;
      worker.onerror = onWorkerError;
    }

    let attachedWorkerMessageListener = false;
    if (typeof worker.addEventListener === 'function') {
      try {
        worker.addEventListener('message', onWorkerMessage);
        attachedWorkerMessageListener = true;
      } catch {
        attachedWorkerMessageListener = false;
      }
    }

    let progressChannel = null;
    let lastLoggedFrameNumber = 0;
    let lastLoggedFrameAtMs = -Infinity;
    let lastHudFrameMetricsAtMs = -Infinity;
    const BroadcastChannelCtor =
      windowObj && typeof windowObj.BroadcastChannel === 'function'
        ? windowObj.BroadcastChannel
        : null;
    if (BroadcastChannelCtor) {
      try {
        progressChannel = new BroadcastChannelCtor('game-progress');
        progressChannel.onmessage = (ev) => {
          const data = ev && ev.data && typeof ev.data === 'object' ? ev.data : null;
          if (!data) return;

          // W366 fast-path: pure per-frame progress labels (`WebGPU frame #N presented`)
          // arrive at 30-60 Hz; bypass 32-source loader/perf scanning and performance.memory
          // sampling when no extra telemetry keys are attached.
          let fastPresentedFrameNum = null;
          if (
            data.type === undefined &&
            data.error === undefined &&
            data.metrics === undefined &&
            typeof data.label === 'string'
          ) {
            const parsedNum = parseWebGpuFramePresentedLabel(data.label);
            if (parsedNum !== null) {
              let hasExtraKeys = false;
              for (const key in data) {
                if (
                  key !== 'label' &&
                  key !== 'framesPresented' &&
                  key !== 'presentCount' &&
                  key !== 'frame' &&
                  key !== 'frameCount' &&
                  key !== 'timestampMs' &&
                  key !== 'drawCalls' &&
                  key !== 'packetsProcessed' &&
                  key !== 'packetCount' &&
                  key !== 'computeDispatches'
                ) {
                  hasExtraKeys = true;
                  break;
                }
              }
              if (!hasExtraKeys) {
                fastPresentedFrameNum = parsedNum;
              }
            }
          }

          if (fastPresentedFrameNum === null) {
            if (handleWorkerMessage(data)) {
              return;
            }
            ingestGpuAdapterFromPayload(data);
            ingestLoaderTelemetry(data, { notify: true, parseLabel: true });
            if (data.metrics && hud && typeof hud.updateMetrics === 'function') {
              try {
                hud.updateMetrics(data.metrics);
              } catch {
                // Ignore HUD errors.
              }
            }
          }
          if (data.error) {
            // Skip duplicate worker_error crash/HUD logging when loader.js already
            // forwards the structured `gpu-validation-error` message over postMessage.
            if (data.gpuValidationForwarded === true) {
              return;
            }
            crashReporter.recordError('worker_error', String(data.error), {
              source: 'loader',
            });
            feedBootLog(`ERROR: ${data.error}`);
            if (hud && typeof hud.appendLog === 'function') {
              try {
                hud.appendLog(`ERROR: ${data.error}`, 'error');
              } catch {
                // Ignore HUD errors.
              }
            }
          } else if (data.label) {
            const labelStr = String(data.label);
            const presentedFrameNum =
              fastPresentedFrameNum !== null
                ? fastPresentedFrameNum
                : parseWebGpuFramePresentedLabel(labelStr);
            if (presentedFrameNum !== null) {
              const frameNowMs = now();
              wgpuFramesPresented = Math.max(wgpuFramesPresented, presentedFrameNum);
              gpuTelemetry.presentCount = Math.max(gpuTelemetry.presentCount, presentedFrameNum);
              recordRealWasmPerfUpdate(data, { framesPresented: presentedFrameNum });
              if (!firstPresentNotified && presentedFrameNum >= 1) {
                notifyFirstPresent({
                  source: 'wgpu_worker',
                  frameIndex: 0,
                  framesPresented: presentedFrameNum,
                });
              }
              if (isSplashOverlayHolding()) {
                updateSplashLoadingAssetsProgress();
              }
              const hudFrameThrottleMs = isHudHidden()
                ? HIDDEN_HUD_FRAME_METRICS_THROTTLE_MS
                : FRAME_PRESENTED_HUD_THROTTLE_MS;
              if (
                hud &&
                typeof hud.updateMetrics === 'function' &&
                (presentedFrameNum === 1 ||
                  frameNowMs - lastHudFrameMetricsAtMs >= hudFrameThrottleMs)
              ) {
                lastHudFrameMetricsAtMs = frameNowMs;
                try {
                  const snap =
                    perfMonitor && typeof perfMonitor.getSnapshot === 'function'
                      ? perfMonitor.getSnapshot()
                      : null;
                  const metricsPatch = {
                    framesPresented: wgpuFramesPresented,
                    totalFrames: wgpuFramesPresented,
                  };
                  if (
                    snap &&
                    snap.frames &&
                    typeof snap.frames.fps === 'number' &&
                    Number.isFinite(snap.frames.fps) &&
                    snap.frames.fps > 0
                  ) {
                    metricsPatch.fps = snap.frames.fps;
                  }
                  hud.updateMetrics(metricsPatch);
                } catch {
                  // Ignore HUD metrics errors.
                }
              }
              if (
                shouldLogWebGpuPresentedFrame(presentedFrameNum, {
                  lastLoggedFrame: lastLoggedFrameNumber,
                  lastLoggedAtMs: lastLoggedFrameAtMs,
                  nowMs: frameNowMs,
                })
              ) {
                lastLoggedFrameNumber = presentedFrameNum;
                lastLoggedFrameAtMs = frameNowMs;
                crashReporter.recordLog('info', 'loader', labelStr);
                feedBootLog(labelStr);
                if (hud && typeof hud.appendLog === 'function') {
                  try {
                    hud.appendLog(labelStr, 'info');
                  } catch {
                    // Ignore HUD errors.
                  }
                }
              }
              return;
            }
            crashReporter.recordLog('info', 'loader', labelStr);
            feedBootLog(labelStr);
            if (hud && typeof hud.appendLog === 'function') {
              try {
                hud.appendLog(labelStr, labelStr.includes('ready') ? 'ok' : 'info');
              } catch {
                // Ignore HUD errors.
              }
            }
          }
        };
      } catch {
        progressChannel = null;
      }
    }

    if (typeof worker.postMessage === 'function') {
      let transferCanvas = canvas;
      let transferList = null;
      if (canvas && typeof canvas.transferControlToOffscreen === 'function') {
        if (!canvasTransferredToOffscreen) {
          const offscreen = canvas.transferControlToOffscreen();
          canvasTransferredToOffscreen = true;
          transferredOffscreenCanvas = offscreen;
          transferCanvas = offscreen;
          transferList = [offscreen];
        } else if (
          canvas.parentNode &&
          typeof canvas.cloneNode === 'function' &&
          typeof canvas.parentNode.replaceChild === 'function'
        ) {
          const freshCanvas = canvas.cloneNode(false);
          canvas.parentNode.replaceChild(freshCanvas, canvas);
          canvas = freshCanvas;
          const offscreen = canvas.transferControlToOffscreen();
          transferredOffscreenCanvas = offscreen;
          transferCanvas = offscreen;
          transferList = [offscreen];
        } else if (transferredOffscreenCanvas) {
          transferCanvas = transferredOffscreenCanvas;
        }
      }

      const initPayload = {
        canvas: transferCanvas,
        width: pixelWidth,
        height: pixelHeight,
        devicePixelRatio: 1,
        ...(swapchainOptions ? { swapchainOptions } : {}),
        inputSab: inputBridge.sab,
        audioSab: audioEngine.sab,
        dataBase,
        ...bootForwardFields,
      };
      if (transferList) {
        worker.postMessage(initPayload, transferList);
      } else {
        worker.postMessage(initPayload);
      }
    }

    if (externalIoWorker && typeof externalIoWorker.postMessage === 'function' && activePerfProfile) {
      try {
        externalIoWorker.postMessage({
          type: 'apply-perf-profile',
          perfProfile: workerPerfInitPayloads.ioWorker.perfProfile,
        });
      } catch {
        // Non-fatal external ioWorker perf profile forwarding error.
      }
    }
    if (externalGpuWorker && typeof externalGpuWorker.postMessage === 'function') {
      try {
        if (activePerfProfile) {
          externalGpuWorker.postMessage({
            type: 'apply-perf-profile',
            perfProfile: workerPerfInitPayloads.gpuWorker.perfProfile,
          });
        }
        if (activeDisplayBounds) {
          externalGpuWorker.postMessage({
            type: 'set-display-bounds',
            displayBounds: workerPerfInitPayloads.gpuWorker.displayBounds,
          });
        }
      } catch {
        // Non-fatal external gpuWorker perf profile / display bounds forwarding error.
      }
    }

    let lastSentPixelWidth = pixelWidth;
    let lastSentPixelHeight = pixelHeight;

    const wasmHandle = {
      mode: SESSION_MODE.WASM,
      worker,
      resize(nextWidth, nextHeight, nextDpr = viewportDpr) {
        if (stopped) {
          return false;
        }
        const pw = Math.max(1, Math.round(nextWidth * nextDpr));
        const ph = Math.max(1, Math.round(nextHeight * nextDpr));
        if (pw === lastSentPixelWidth && ph === lastSentPixelHeight) {
          return false;
        }
        if (typeof worker.postMessage === 'function') {
          lastSentPixelWidth = pw;
          lastSentPixelHeight = ph;
          worker.postMessage({ resize: true, width: pw, height: ph });
          return true;
        }
        return false;
      },
      async stop() {
        if (stopped) {
          return;
        }
        stopped = true;

        if (usedAddEventListener) {
          worker.removeEventListener('error', onWorkerError);
          worker.removeEventListener('messageerror', onMessageError);
        } else if (worker.onerror === onWorkerError) {
          worker.onerror = prevOnError;
        }

        if (attachedWorkerMessageListener) {
          attachedWorkerMessageListener = false;
          if (typeof worker.removeEventListener === 'function') {
            try {
              worker.removeEventListener('message', onWorkerMessage);
            } catch {
              // Ignore listener removal errors.
            }
          }
        }

        if (progressChannel) {
          try {
            progressChannel.onmessage = null;
            if (typeof progressChannel.close === 'function') {
              progressChannel.close();
            }
          } catch {
            // Ignore channel close errors.
          }
          progressChannel = null;
        }

        try {
          if (typeof worker.terminate === 'function') {
            worker.terminate();
          } else if (typeof worker.stop === 'function') {
            await worker.stop();
          } else if (typeof worker.destroy === 'function') {
            await worker.destroy();
          }
        } finally {
          if (activeWasmHandle === wasmHandle) {
            activeWasmHandle = null;
          }
          if (activeWasmWorker === worker) {
            activeWasmWorker = null;
          }
        }
      },
      destroy() {
        return this.stop();
      },
    };

    activeWasmHandle = wasmHandle;
    activeWasmWorker = worker;
    return wasmHandle;
  }

  // 6b. WebGPU G5 smoke demo launcher (used when dist/runtime.wasm is absent or preferSmoke: true).
  async function startSmokeDemo(launchContext) {
    if (destroyed) {
      throw new Error('Cannot start SmokeDemoRunner on a destroyed PlayGta6AppHandle');
    }
    if (launchContext && launchContext.serverStatus) {
      recordDiscoveredBootAssets(launchContext.serverStatus);
    }
    if (!canvas || typeof canvas !== 'object') {
      throw new Error('Cannot start SmokeDemoRunner: canvas is not configured');
    }

    let adapter = null;
    let device = null;

    if (webgpuAdapterFactory) {
      const factoryResult = await webgpuAdapterFactory({
        navigatorObj,
        canvas,
        launchContext,
      });
      if (factoryResult && typeof factoryResult.requestDevice === 'function') {
        adapter = factoryResult;
        device = await adapter.requestDevice(smokeDemoOptions.deviceOptions);
      } else if (factoryResult && factoryResult.device && typeof factoryResult.device === 'object') {
        adapter = factoryResult.adapter || null;
        device = factoryResult.device;
      } else {
        throw new Error('WebGPU adapter factory did not return a valid GPUAdapter');
      }
    } else {
      const gpu = navigatorObj && navigatorObj.gpu;
      if (!gpu || typeof gpu.requestAdapter !== 'function') {
        throw new Error('WebGPU navigator.gpu.requestAdapter is unavailable');
      }
      adapter = await gpu.requestAdapter(smokeDemoOptions.adapterOptions);
      if (!adapter || typeof adapter.requestDevice !== 'function') {
        throw new Error('Failed to acquire WebGPU GPUAdapter');
      }
      device = await adapter.requestDevice(smokeDemoOptions.deviceOptions);
    }

    if (!device || typeof device !== 'object') {
      throw new Error('Failed to acquire WebGPU GPUDevice');
    }

    if (adapter && typeof adapter === 'object' && adapter.info) {
      recordGpuAdapter(adapter.info);
    }

    if (destroyed) {
      if (typeof device.destroy === 'function') {
        try {
          device.destroy();
        } catch {
          // Ignore.
        }
      }
      throw new Error('PlayGta6AppHandle was destroyed while acquiring WebGPU device');
    }

    const gpu = navigatorObj && navigatorObj.gpu;
    const format =
      typeof smokeDemoOptions.format === 'string' && smokeDemoOptions.format.length > 0
        ? smokeDemoOptions.format
        : gpu && typeof gpu.getPreferredCanvasFormat === 'function'
          ? gpu.getPreferredCanvasFormat()
          : DEFAULT_CANVAS_FORMAT;

    let deviceStopped = false;
    const onUncapturedError = (ev) => {
      if (deviceStopped || destroyed) return;
      const errObj = ev && ev.error ? ev.error : ev;
      const sessionRecordsCrash =
        Boolean(session) && typeof session.recordGpuValidationError === 'function';
      if (!sessionRecordsCrash) {
        crashReporter.recordError('webgpu_validation', errObj, { source: 'smokeDemo' });
      }
      recordValidationError(
        { error: errObj, source: 'smokeDemo' },
        { source: 'smokeDemo', skipCrashReporter: true, skipHudLog: true }
      );
    };

    const hasDeviceEvents = isEventTargetLike(device);
    if (hasDeviceEvents) {
      device.addEventListener('uncapturederror', onUncapturedError);
    }

    if (device.lost && typeof device.lost.then === 'function') {
      device.lost
        .then((info) => {
          if (deviceStopped || destroyed) return;
          if (info && info.reason === 'destroyed') return;
          crashReporter.recordError('webgpu_device_lost', info || 'WebGPU device lost', {
            source: 'smokeDemo',
          });
        })
        .catch(() => {
          // Never allow device.lost rejections to surface as unhandled rejections.
        });
    }

    const runner = new SmokeDemoRunner({
      ...smokeDemoOptions,
      device,
      canvas,
      format,
      width: viewportWidth,
      height: viewportHeight,
      devicePixelRatio: viewportDpr,
      inputSab: inputBridge.sab,
      perfMonitor,
      hud,
      ...(resolvedRaf ? { raf: resolvedRaf } : {}),
      ...(resolvedCancelRaf ? { cancelRaf: resolvedCancelRaf } : {}),
      now: smokeDemoOptions.now || now,
    });

    activeSmokeRunner = runner;
    if (smokeDemoOptions.autoStartLoop !== false) {
      runner.start();
    } else {
      runner.init();
    }
    notifyFirstPresent({ source: 'smoke_demo' });
    if (isSplashOverlayHolding()) {
      notifyFirstVisibleFrame({ source: 'smoke_demo' });
    }

    const cleanupDeviceListeners = () => {
      deviceStopped = true;
      if (hasDeviceEvents) {
        device.removeEventListener('uncapturederror', onUncapturedError);
      }
    };

    return {
      mode: SESSION_MODE.SMOKE,
      runner,
      adapter,
      device,
      stop() {
        cleanupDeviceListeners();
        if (!runner.destroyed) {
          runner.destroy();
        }
        if (activeSmokeRunner === runner) {
          activeSmokeRunner = null;
        }
      },
      destroy() {
        this.stop();
      },
    };
  }

  // 7. RuntimeSessionController wiring all subsystems together.
  const capabilityProbe =
    typeof options.capabilityProbe === 'function'
      ? options.capabilityProbe
      : createCapabilityProbe({ windowObj, navigatorObj, webgpuAdapterFactory });

  const fallbackReclaimHooks = new Set();
  const reclaimUnregisters = [];
  let fallbackReclaimRuns = 0;
  let fallbackReclaimFailures = 0;
  let fallbackLastReclaimReason = null;
  let fallbackLastReclaimAtMs = null;

  const userMemoryReclaimHooks = [];
  if (typeof options.onMemoryReclaim === 'function') {
    userMemoryReclaimHooks.push(options.onMemoryReclaim);
  }
  if (typeof options.onMemoryPressure === 'function') {
    userMemoryReclaimHooks.push(options.onMemoryPressure);
  }
  if (Array.isArray(options.memoryPressureCallbacks)) {
    for (let i = 0; i < options.memoryPressureCallbacks.length; i += 1) {
      if (typeof options.memoryPressureCallbacks[i] === 'function') {
        userMemoryReclaimHooks.push(options.memoryPressureCallbacks[i]);
      }
    }
  }
  if (Array.isArray(options.reclaimHooks)) {
    for (let i = 0; i < options.reclaimHooks.length; i += 1) {
      if (typeof options.reclaimHooks[i] === 'function') {
        userMemoryReclaimHooks.push(options.reclaimHooks[i]);
      }
    }
  }

  async function runBuiltinSoftMemoryReclaim(context) {
    if (destroyed) {
      return;
    }
    if (bootOrchestrator && typeof bootOrchestrator.releasePreloadBuffers === 'function') {
      try {
        bootOrchestrator.releasePreloadBuffers();
      } catch {
        // Preload buffer release must never interrupt soft memory reclaim.
      }
    }
    const reason = context && typeof context.reason === 'string' ? context.reason : 'memory-pressure';
    const shouldEvictOnReclaim =
      options.reclaimEvictsOpfs !== false &&
      (reason !== 'restart' || Boolean(latestFastBootPlan && latestFastBootPlan.shouldEvictColdChunks));
    if (
      shouldEvictOnReclaim &&
      storageManager &&
      typeof storageManager.runQuotaPressureEviction === 'function'
    ) {
      try {
        const rawEviction = await storageManager.runQuotaPressureEviction();
        if (!destroyed && rawEviction && typeof rawEviction === 'object') {
          latestQuotaEviction = Object.freeze({
            triggered: true,
            ok: rawEviction.ok !== false,
            evictedBytes: toSafeNonNegativeInt(rawEviction.evictedBytes, 0),
            evictedEntries: toSafeNonNegativeInt(rawEviction.evictedEntries, 0),
          });
        }
      } catch {
        // Storage quota eviction errors must never interrupt soft memory reclaim.
      }
    }
  }

  session = new RuntimeSessionController({
    fetchImpl,
    statusUrl,
    hud,
    inputBridge,
    audioEngine,
    storageManager,
    perfMonitor,
    crashReporter: sessionCrashAdapter,
    startWasmRuntime,
    startSmokeDemo,
    capabilityProbe,
    now,
    ...(options.hudIntervalMs !== undefined ? { hudIntervalMs: options.hudIntervalMs } : {}),
    audioGestureTarget:
      options.audioGestureTarget !== undefined ? options.audioGestureTarget : windowObj || undefined,
    onMemoryReclaim: runBuiltinSoftMemoryReclaim,
  });

  function registerReclaimHook(hook) {
    if (typeof hook !== 'function') {
      throw new TypeError('registerReclaimHook requires a hook function');
    }
    if (destroyed) {
      return () => {};
    }
    if (session && typeof session.registerReclaimHook === 'function') {
      const unregister = session.registerReclaimHook(hook);
      reclaimUnregisters.push(unregister);
      return unregister;
    }
    fallbackReclaimHooks.add(hook);
    let active = true;
    const unregister = () => {
      if (!active) return;
      active = false;
      fallbackReclaimHooks.delete(hook);
    };
    reclaimUnregisters.push(unregister);
    return unregister;
  }

  for (let i = 0; i < userMemoryReclaimHooks.length; i += 1) {
    registerReclaimHook(userMemoryReclaimHooks[i]);
  }

  async function reclaimSoftMemory(reason = 'memory-pressure') {
    const rawReason =
      typeof reason === 'string' && reason.trim().length > 0
        ? reason.trim()
        : reason && typeof reason === 'object' && typeof reason.reason === 'string' && reason.reason.trim().length > 0
          ? reason.reason.trim()
          : 'memory-pressure';
    const sanitizedReason = sanitizeDiagnosticText(rawReason, 128) || 'memory-pressure';
    if (destroyed) {
      return Object.freeze({
        ok: false,
        destroyed: true,
        reason: sanitizedReason,
        hooksInvoked: 0,
        hooksSucceeded: 0,
        hooksFailed: 0,
        reclaimRuns: fallbackReclaimRuns,
        reclaimFailures: fallbackReclaimFailures,
      });
    }
    if (session && typeof session.reclaimSoftMemory === 'function') {
      try {
        return await session.reclaimSoftMemory(sanitizedReason);
      } catch (err) {
        crashReporter.recordError('runtime', err, {
          source: 'session',
          phase: 'reclaimSoftMemory',
          reason: sanitizedReason,
        });
        return Object.freeze({
          ok: false,
          reason: sanitizedReason,
          error: sanitizeDiagnosticText(err && err.message ? err.message : String(err)),
        });
      }
    }
    if (session && typeof session.requestMemoryReclaim === 'function') {
      try {
        return await session.requestMemoryReclaim(sanitizedReason);
      } catch (err) {
        crashReporter.recordError('runtime', err, {
          source: 'session',
          phase: 'requestMemoryReclaim',
          reason: sanitizedReason,
        });
        return Object.freeze({
          ok: false,
          reason: sanitizedReason,
          error: sanitizeDiagnosticText(err && err.message ? err.message : String(err)),
        });
      }
    }
    const timestampMs = now();
    fallbackReclaimRuns += 1;
    fallbackLastReclaimReason = sanitizedReason;
    fallbackLastReclaimAtMs = timestampMs;
    const hooks = [runBuiltinSoftMemoryReclaim, ...fallbackReclaimHooks];
    let hooksInvoked = 0;
    let hooksSucceeded = 0;
    let hooksFailed = 0;
    for (let i = 0; i < hooks.length; i += 1) {
      hooksInvoked += 1;
      try {
        await hooks[i]({
          reason: sanitizedReason,
          state: resolveSessionStateText(),
          bootMode: resolveCurrentBootMode(),
          timestampMs,
        });
        hooksSucceeded += 1;
      } catch (err) {
        hooksFailed += 1;
        fallbackReclaimFailures += 1;
        crashReporter.recordError('runtime', err, {
          source: 'memoryReclaim',
          reason: sanitizedReason,
        });
      }
    }
    return Object.freeze({
      ok: hooksFailed === 0,
      reason: sanitizedReason,
      hooksInvoked,
      hooksSucceeded,
      hooksFailed,
      reclaimRuns: fallbackReclaimRuns,
      reclaimFailures: fallbackReclaimFailures,
      timestampMs,
    });
  }

  if (
    initialBootAssets.sideModuleUrls !== undefined ||
    initialBootAssets.sideModuleCount !== undefined
  ) {
    notifySideModulePreflight(initialBootAssets);
  }

  // 7b. Optional BootOrchestrator (W44/W309) wired when splash overlay or explicit flag is enabled.
  const useBootOrchestrator =
    options.bootOrchestrator !== undefined ||
    options.enableBootOrchestrator === true ||
    Boolean(splashOverlay && options.enableBootOrchestrator !== false);

  let lastBootSessionResult = null;
  let pendingStartOpts = undefined;

  if (options.bootOrchestrator !== undefined) {
    bootOrchestrator = options.bootOrchestrator;
  } else if (useBootOrchestrator && typeof BootOrchestratorCtor === 'function') {
    const splashAdapter = createSplashOverlayAdapter(splashOverlay);
    const bootOrchestratorOptions =
      options.bootOrchestratorOptions && typeof options.bootOrchestratorOptions === 'object'
        ? options.bootOrchestratorOptions
        : {};
    const orchestratorConfig = {
      ...bootOrchestratorOptions,
      ...(options.parallelPreload !== undefined ? { parallelPreload: options.parallelPreload } : {}),
      ...(options.preloadConcurrency !== undefined
        ? { preloadConcurrency: options.preloadConcurrency }
        : {}),
      ...(options.releasePreloadBuffers !== undefined
        ? { releasePreloadBuffers: options.releasePreloadBuffers }
        : {}),
      ...(options.stageTimeoutMs !== undefined ? { stageTimeoutMs: options.stageTimeoutMs } : {}),
      ...(options.preloadRuntimeWasm !== undefined
        ? { preloadRuntimeWasm: options.preloadRuntimeWasm }
        : {}),
      ...(options.opfsProbe !== undefined ? { opfsProbe: options.opfsProbe } : {}),
      ...(options.preloadShaderPackHeader !== undefined
        ? { preloadShaderPackHeader: options.preloadShaderPackHeader }
        : {}),
      ...(options.preloadReflectionSidecar !== undefined
        ? { preloadReflectionSidecar: options.preloadReflectionSidecar }
        : {}),
      ...(options.preloadTasks !== undefined ? { preloadTasks: options.preloadTasks } : {}),
      ...(options.onPhaseChange !== undefined ? { onPhaseChange: options.onPhaseChange } : {}),
      ...(options.onFirstPresent !== undefined ? { onFirstPresent: options.onFirstPresent } : {}),
      ...(options.onStageStall !== undefined ? { onStageStall: options.onStageStall } : {}),
      fetchImpl,
      statusUrl,
      splashOverlay: splashAdapter,
      preloader: pgimPreloader,
      bootTracer,
      nowFn: bootOrchestratorOptions.nowFn || now,
      ...(bootOrchestratorOptions.getStreamConfig === undefined
        ? {
            getStreamConfig: () =>
              buildPreloaderStreamOptions(activeStreamConfig, activeStreamExplicitDisabled),
          }
        : {}),
      onLaunchRuntime: async (bootPayload) => {
        recordDiscoveredBootAssets(bootPayload);
        lastBootSessionResult = await session.start(pendingStartOpts);
      },
      onFallbackSmokeDemo: async (bootPayload) => {
        recordDiscoveredBootAssets(bootPayload);
        const fallbackOpts =
          pendingStartOpts && typeof pendingStartOpts === 'object'
            ? { preferSmoke: true, ...pendingStartOpts }
            : { preferSmoke: true };
        lastBootSessionResult = await session.start(fallbackOpts);
      },
    };
    bootOrchestrator = new BootOrchestratorCtor(orchestratorConfig);
    ownsBootOrchestrator = true;
  }

  // 8. Viewport resize handler (manual API + optional window resize & ResizeObserver listeners).
  function notifyWorkersDisplayBounds(nextBounds) {
    const key = displayBoundsKey(nextBounds);
    if (key === lastSentDisplayBoundsKey) {
      return false;
    }
    lastSentDisplayBoundsKey = key;
    const msg = {
      type: 'set-display-bounds',
      displayBounds: nextBounds,
    };
    let posted = false;
    const loaderWorker =
      activeWasmHandle && activeWasmHandle.worker && typeof activeWasmHandle.worker.postMessage === 'function'
        ? activeWasmHandle.worker
        : null;
    if (loaderWorker) {
      try {
        loaderWorker.postMessage(msg);
        posted = true;
      } catch {
        // Non-fatal display bounds forwarding error.
      }
    }
    if (externalGpuWorker && typeof externalGpuWorker.postMessage === 'function') {
      try {
        externalGpuWorker.postMessage(msg);
        posted = true;
      } catch {
        // Non-fatal external gpuWorker display bounds forwarding error.
      }
    }
    return posted;
  }

  function syncDisplayBoundsFromWindow(explicitDpr) {
    if (destroyed) {
      return false;
    }
    const nextBounds = resolveDisplayBounds({
      windowObj,
      screenObj: options.screenObj,
      devicePixelRatio: isPositiveFinite(windowObj && windowObj.devicePixelRatio)
        ? windowObj.devicePixelRatio
        : isPositiveFinite(explicitDpr)
          ? explicitDpr
          : undefined,
      maxTextureDimension2D: options.maxTextureDimension2D,
      displayBounds: options.displayBounds,
      perfProfileApi: options.perfProfileApi,
    });
    if (!nextBounds) {
      return false;
    }
    const prevKey = displayBoundsKey(activeDisplayBounds);
    const nextKey = displayBoundsKey(nextBounds);
    activeDisplayBounds = nextBounds;
    if (activePerfProfile) {
      activePerfProfile = toCloneablePerfProfile(activePerfProfile, activeDisplayBounds);
    }
    if (nextKey !== prevKey || nextKey !== lastSentDisplayBoundsKey) {
      return notifyWorkersDisplayBounds(activeDisplayBounds);
    }
    return false;
  }

  function setPerfProfile(profileOrName) {
    if (destroyed) {
      return activePerfProfile;
    }
    const resolved = resolveBootstrapPerfProfile({
      options: {
        ...options,
        perfProfile: profileOrName,
      },
      windowObj,
      navigatorObj,
      urlSearch: '',
      displayBounds: activeDisplayBounds,
    });
    activePerfProfile = resolved.perfProfile;
    if (resolved.displayBounds) {
      activeDisplayBounds = resolved.displayBounds;
    }
    const msg = {
      type: 'apply-perf-profile',
      perfProfile: activePerfProfile,
    };
    const loaderWorker =
      activeWasmHandle && activeWasmHandle.worker && typeof activeWasmHandle.worker.postMessage === 'function'
        ? activeWasmHandle.worker
        : null;
    for (const target of [loaderWorker, externalIoWorker, externalGpuWorker]) {
      if (target && typeof target.postMessage === 'function') {
        try {
          target.postMessage(msg);
        } catch {
          // Non-fatal perf profile forwarding error.
        }
      }
    }
    return activePerfProfile;
  }

  function setDisplayBounds(boundsOrWidth, screenHeight, dpr) {
    if (destroyed) {
      return activeDisplayBounds;
    }
    let nextBounds = null;
    if (boundsOrWidth === null || boundsOrWidth === undefined) {
      nextBounds = null;
    } else if (typeof boundsOrWidth === 'object') {
      nextBounds = resolveDisplayBounds({
        windowObj,
        displayBounds: boundsOrWidth,
        maxTextureDimension2D: options.maxTextureDimension2D,
        perfProfileApi: options.perfProfileApi,
      });
    } else {
      nextBounds = resolveDisplayBounds({
        windowObj,
        screenWidth: boundsOrWidth,
        screenHeight,
        devicePixelRatio: dpr,
        maxTextureDimension2D: options.maxTextureDimension2D,
        perfProfileApi: options.perfProfileApi,
      });
    }
    activeDisplayBounds = nextBounds;
    if (activePerfProfile) {
      activePerfProfile = toCloneablePerfProfile(activePerfProfile, activeDisplayBounds);
    }
    notifyWorkersDisplayBounds(activeDisplayBounds);
    return activeDisplayBounds;
  }

  function resize(nextWidth, nextHeight, nextDpr) {
    if (destroyed) {
      return false;
    }
    if (!isPositiveFinite(nextWidth) || !isPositiveFinite(nextHeight)) {
      return false;
    }

    const resolvedDpr = isPositiveFinite(nextDpr)
      ? nextDpr
      : isPositiveFinite(windowObj && windowObj.devicePixelRatio)
        ? Math.min(windowObj.devicePixelRatio, maxDevicePixelRatio)
        : viewportDpr;

    viewportWidth = nextWidth;
    viewportHeight = nextHeight;
    viewportDpr = resolvedDpr;

    let handled = false;
    if (activeSmokeRunner && !activeSmokeRunner.destroyed) {
      handled = activeSmokeRunner.resize(nextWidth, nextHeight, resolvedDpr);
    } else if (activeWasmHandle) {
      handled = activeWasmHandle.resize(nextWidth, nextHeight, resolvedDpr);
    }
    syncDisplayBoundsFromWindow(isPositiveFinite(nextDpr) ? nextDpr : undefined);
    return handled;
  }

  let resizeListenerAttached = false;
  let canvasResizeObserver = null;

  const measureAndResizeViewport = () => {
    if (destroyed) return;
    const w =
      isPositiveFinite(canvas && canvas.clientWidth)
        ? canvas.clientWidth
        : windowObj && windowObj.innerWidth;
    const h =
      isPositiveFinite(canvas && canvas.clientHeight)
        ? canvas.clientHeight
        : windowObj && windowObj.innerHeight;
    if (isPositiveFinite(w) && isPositiveFinite(h)) {
      resize(w, h);
    } else {
      syncDisplayBoundsFromWindow();
    }
  };

  const onWindowResize = () => {
    measureAndResizeViewport();
  };

  if (options.autoResize !== false && isEventTargetLike(windowObj)) {
    windowObj.addEventListener('resize', onWindowResize);
    resizeListenerAttached = true;
  }

  const ResizeObserverCtor =
    (windowObj && typeof windowObj.ResizeObserver === 'function' && windowObj.ResizeObserver) ||
    (typeof globalThis !== 'undefined' && typeof globalThis.ResizeObserver === 'function'
      ? globalThis.ResizeObserver
      : null);
  if (options.autoResize !== false && ResizeObserverCtor && canvas && typeof canvas === 'object') {
    try {
      canvasResizeObserver = new ResizeObserverCtor(() => {
        onWindowResize();
      });
      if (typeof canvasResizeObserver.observe === 'function') {
        canvasResizeObserver.observe(canvas);
      }
    } catch {
      canvasResizeObserver = null;
    }
  }

  // 8b. First-user-gesture Web Audio unlock (Click + Key + Gamepad), visibilitychange recovery,
  //     and Pointer Lock UX helpers (W261 / W370 / W382 / #1028 / #1759).
  notifyRuntimeAudioContextState = (outputState) => {
    if (destroyed || !audioEngine || typeof audioEngine !== 'object') {
      return;
    }
    const loaderWorker =
      activeWasmHandle && activeWasmHandle.worker && typeof activeWasmHandle.worker.postMessage === 'function'
        ? activeWasmHandle.worker
        : null;
    if (!loaderWorker) {
      return;
    }
    try {
      const payload = {
        type: 'audio-context-state',
        setOutputState: true,
        outputState,
        unlockedOnGesture: Boolean(audioEngine.unlockedOnGesture),
      };
      if (
        typeof audioEngine.sampleRate === 'number' &&
        Number.isFinite(audioEngine.sampleRate) &&
        audioEngine.sampleRate > 0
      ) {
        payload.sampleRate = audioEngine.sampleRate;
      }
      if (audioEngine.sab) {
        payload.ringSab = audioEngine.sab;
      }
      loaderWorker.postMessage(payload);
    } catch {
      // Non-fatal audio-context-state forwarding error.
    }
  };

  const audioGestureTarget =
    options.audioGestureTarget !== undefined
      ? options.audioGestureTarget
      : isEventTargetLike(windowObj)
        ? windowObj
        : isEventTargetLike(documentObj)
          ? documentObj
          : null;
  let audioGestureUnlockAttempted = false;
  let audioGestureUnlockInFlight = false;
  let audioGestureRetriesRemaining = 5;
  let audioGestureCleanup = null;
  let audioVisibilityRecoveries = 0;
  let audioVisibilityInFlight = false;
  let gamepadUnlockTimerId = null;
  let visibilityLifecycleCleanup = null;
  let pointerLockUiCleanup = null;
  const audioGestureEvents = ['pointerdown', 'keydown', 'click', 'touchstart', 'gamepadconnected'];
  const GAMEPAD_AUDIO_UNLOCK_POLL_MS = 120;
  const GAMEPAD_AXIS_ACTIVITY_THRESHOLD = 0.25;

  function isAudioCurrentlyRunning() {
    return Boolean(audioEngine && typeof audioEngine === 'object' && audioEngine.state === 'running');
  }

  function isAudioNeedsUnlock() {
    if (destroyed || !audioEngine || typeof audioEngine !== 'object') {
      return false;
    }
    const st = audioEngine.state;
    return st === 'idle' || st === 'suspended' || st === 'interrupted';
  }

  function readConnectedGamepads() {
    if (!navigatorObj || typeof navigatorObj.getGamepads !== 'function') {
      return [];
    }
    try {
      const raw = navigatorObj.getGamepads();
      if (!raw || typeof raw.length !== 'number') {
        return [];
      }
      const pads = [];
      for (let i = 0; i < raw.length; i += 1) {
        const pad = raw[i];
        if (pad && pad.connected !== false) {
          pads.push(pad);
        }
      }
      return pads;
    } catch {
      return [];
    }
  }

  function hasActiveGamepadInput() {
    const pads = readConnectedGamepads();
    for (let p = 0; p < pads.length; p += 1) {
      const pad = pads[p];
      const buttons = pad.buttons;
      if (buttons && typeof buttons.length === 'number') {
        for (let b = 0; b < buttons.length; b += 1) {
          const btn = buttons[b];
          if (typeof btn === 'number') {
            if (btn > 0.5) return true;
          } else if (
            btn &&
            (btn.pressed === true || (typeof btn.value === 'number' && btn.value > 0.5))
          ) {
            return true;
          }
        }
      }
      const axes = pad.axes;
      if (axes && typeof axes.length === 'number') {
        for (let a = 0; a < axes.length; a += 1) {
          const val = axes[a];
          if (
            typeof val === 'number' &&
            Number.isFinite(val) &&
            Math.abs(val) > GAMEPAD_AXIS_ACTIVITY_THRESHOLD
          ) {
            return true;
          }
        }
      }
    }
    return false;
  }

  function stopGamepadAudioUnlockPoll() {
    if (gamepadUnlockTimerId === null) {
      return;
    }
    const timerId = gamepadUnlockTimerId;
    gamepadUnlockTimerId = null;
    try {
      const clearFn =
        resolvedClearInterval ||
        (typeof clearInterval === 'function' ? clearInterval : null);
      if (clearFn) {
        clearFn(timerId);
      }
    } catch {
      // Ignore timer teardown errors.
    }
  }

  function startGamepadAudioUnlockPoll() {
    if (
      destroyed ||
      options.enableGamepadAudioUnlock === false ||
      gamepadUnlockTimerId !== null ||
      !isAudioNeedsUnlock() ||
      readConnectedGamepads().length === 0
    ) {
      return;
    }
    const setFn =
      resolvedSetInterval ||
      (typeof setInterval === 'function' ? setInterval : null);
    if (!setFn) {
      return;
    }
    try {
      gamepadUnlockTimerId = setFn(() => {
        if (destroyed || !isAudioNeedsUnlock()) {
          stopGamepadAudioUnlockPoll();
          return;
        }
        if (readConnectedGamepads().length === 0) {
          stopGamepadAudioUnlockPoll();
          return;
        }
        if (hasActiveGamepadInput()) {
          audioGestureUnlockAttempted = false;
          void unlockAudio({ source: 'gamepad' });
        }
      }, GAMEPAD_AUDIO_UNLOCK_POLL_MS);
    } catch {
      gamepadUnlockTimerId = null;
    }
  }

  function detachAudioGestureUnlock() {
    if (!audioGestureCleanup) {
      return;
    }
    const cleanup = audioGestureCleanup;
    audioGestureCleanup = null;
    try {
      cleanup();
    } catch {
      // Ignore audio gesture listener removal errors.
    }
  }

  function armAudioGestureUnlock() {
    if (destroyed || audioGestureUnlockAttempted || !isEventTargetLike(audioGestureTarget) || !audioEngine) {
      return;
    }
    detachAudioGestureUnlock();
    const registered = [];
    for (let i = 0; i < audioGestureEvents.length; i += 1) {
      const evt = audioGestureEvents[i];
      try {
        audioGestureTarget.addEventListener(evt, invokeAudioGestureUnlock, {
          passive: true,
          once: true,
        });
        registered.push([audioGestureTarget, evt]);
      } catch {
        // Ignore listener registration errors in restricted hosts.
      }
    }
    if (audioGestureTarget !== windowObj && isEventTargetLike(windowObj)) {
      try {
        windowObj.addEventListener('gamepadconnected', invokeAudioGestureUnlock, {
          passive: true,
          once: true,
        });
        registered.push([windowObj, 'gamepadconnected']);
      } catch {
        // Ignore listener registration errors in restricted hosts.
      }
    }
    if (registered.length > 0) {
      audioGestureCleanup = () => {
        for (let i = 0; i < registered.length; i += 1) {
          const [target, evt] = registered[i];
          try {
            target.removeEventListener(evt, invokeAudioGestureUnlock);
          } catch {
            // Ignore listener removal errors.
          }
        }
      };
    }
  }

  function runAudioUnlockAttempt(opts = {}) {
    if (destroyed || !audioEngine || typeof audioEngine !== 'object') {
      return Promise.resolve({
        ok: false,
        state: 'closed',
        unlocked: false,
        unlockedOnGesture: false,
        reason: destroyed ? 'destroyed' : 'no-audio-engine',
      });
    }
    if (audioGestureUnlockInFlight) {
      return Promise.resolve({
        ok: isAudioCurrentlyRunning(),
        state: normalizeAudioOutputState(audioEngine.state),
        unlocked: isAudioCurrentlyRunning(),
        unlockedOnGesture: Boolean(audioEngine.unlockedOnGesture),
        reason: 'in-flight',
      });
    }

    audioGestureUnlockInFlight = true;
    detachAudioGestureUnlock();

    const source = typeof opts.source === 'string' ? opts.source : 'gesture';
    const isRecovery = source === 'visibilitychange' || source === 'focus' || source === 'pageshow';

    const finalize = (failedExplicitly) => {
      audioGestureUnlockInFlight = false;
      if (destroyed || !audioEngine || typeof audioEngine !== 'object') {
        audioGestureUnlockAttempted = true;
        stopGamepadAudioUnlockPoll();
        return {
          ok: false,
          state: 'closed',
          unlocked: false,
          unlockedOnGesture: false,
          reason: 'destroyed',
        };
      }
      const rawState = audioEngine.state;
      const stillSuspended = Boolean(
        (audioEngine.context || rawState === 'suspended' || rawState === 'interrupted') &&
          (rawState === 'suspended' || rawState === 'interrupted')
      );
      if (failedExplicitly || stillSuspended) {
        if (audioGestureRetriesRemaining > 0) {
          audioGestureRetriesRemaining -= 1;
          audioGestureUnlockAttempted = false;
          armAudioGestureUnlock();
        } else {
          audioGestureUnlockAttempted = true;
        }
        if (readConnectedGamepads().length > 0) {
          startGamepadAudioUnlockPoll();
        }
        return {
          ok: false,
          state: normalizeAudioOutputState(rawState),
          unlocked: false,
          unlockedOnGesture: Boolean(audioEngine.unlockedOnGesture),
          reason: failedExplicitly ? 'resume-rejected' : 'still-suspended',
        };
      }

      audioGestureUnlockAttempted = true;
      stopGamepadAudioUnlockPoll();
      if (rawState === 'running') {
        notifyRuntimeAudioContextState('active');
      }
      return {
        ok: true,
        state: normalizeAudioOutputState(rawState),
        unlocked: true,
        unlockedOnGesture: Boolean(audioEngine.unlockedOnGesture || rawState === 'running'),
        sampleRate:
          typeof audioEngine.sampleRate === 'number' && Number.isFinite(audioEngine.sampleRate)
            ? audioEngine.sampleRate
            : null,
        reason: null,
      };
    };

    const invokeMethod = () => {
      if (destroyed || !audioEngine || typeof audioEngine !== 'object') {
        return Promise.resolve(finalize(false));
      }
      try {
        let result;
        if (isRecovery && typeof audioEngine.recoverFromInterruption === 'function') {
          result = audioEngine.recoverFromInterruption({ armGestureFallback: false });
        } else if (typeof audioEngine.unlockOnUserGesture === 'function') {
          result = audioEngine.unlockOnUserGesture();
        } else if (typeof audioEngine.unlockOnGesture === 'function') {
          result = audioEngine.unlockOnGesture();
        } else if (typeof audioEngine.resume === 'function') {
          result = audioEngine.resume();
        }
        if (result && typeof result.then === 'function') {
          return result.then(
            (val) => {
              const failed =
                val === false ||
                (val && typeof val === 'object' && (val.ok === false || val.unlocked === false));
              return finalize(Boolean(failed));
            },
            () => finalize(true)
          );
        }
        const failed =
          result === false ||
          (result &&
            typeof result === 'object' &&
            (result.ok === false || result.unlocked === false));
        return Promise.resolve(finalize(Boolean(failed)));
      } catch {
        // Audio unlock errors must never interrupt input or rendering.
        return Promise.resolve(finalize(true));
      }
    };

    const needsLazyStart =
      !isRecovery &&
      !hasExplicitAudioEngine &&
      hasWebAudioSupport &&
      !audioEngine.context &&
      audioEngine.state === 'idle' &&
      typeof audioEngine.start === 'function';

    if (needsLazyStart) {
      try {
        return Promise.resolve(audioEngine.start())
          .then((startRes) => {
            if (startRes && typeof startRes === 'object' && startRes.ok === false) {
              return finalize(true);
            }
            return invokeMethod();
          })
          .catch(() => finalize(true));
      } catch {
        return Promise.resolve(finalize(true));
      }
    }

    return invokeMethod();
  }

  function invokeAudioGestureUnlock(ev) {
    if (
      destroyed ||
      audioGestureUnlockAttempted ||
      audioGestureUnlockInFlight ||
      !audioEngine ||
      typeof audioEngine !== 'object'
    ) {
      return;
    }
    const source = ev && typeof ev.type === 'string' ? ev.type : 'gesture';
    void runAudioUnlockAttempt({ source });
  }

  function unlockAudio(opts = {}) {
    if (destroyed || !audioEngine || typeof audioEngine !== 'object') {
      return Promise.resolve({
        ok: false,
        state: 'closed',
        unlocked: false,
        unlockedOnGesture: false,
        reason: destroyed ? 'destroyed' : 'no-audio-engine',
      });
    }
    if (audioEngine.state === 'running') {
      audioGestureUnlockAttempted = true;
      detachAudioGestureUnlock();
      stopGamepadAudioUnlockPoll();
      notifyRuntimeAudioContextState('active');
      return Promise.resolve({
        ok: true,
        state: 'running',
        unlocked: true,
        alreadyRunning: true,
        unlockedOnGesture: Boolean(audioEngine.unlockedOnGesture || true),
        sampleRate:
          typeof audioEngine.sampleRate === 'number' && Number.isFinite(audioEngine.sampleRate)
            ? audioEngine.sampleRate
            : null,
        reason: null,
      });
    }
    audioGestureUnlockAttempted = false;
    audioGestureRetriesRemaining = Math.max(audioGestureRetriesRemaining, 5);
    return runAudioUnlockAttempt({
      source: opts && typeof opts.source === 'string' ? opts.source : 'manual',
    });
  }

  function getAudioStatus() {
    const rawState = audioEngine && typeof audioEngine === 'object' ? audioEngine.state ?? 'idle' : 'closed';
    const visState =
      documentObj && typeof documentObj.visibilityState === 'string'
        ? documentObj.visibilityState
        : documentObj && documentObj.hidden === true
          ? 'hidden'
          : 'visible';
    return {
      state: normalizeAudioOutputState(rawState),
      rawState: typeof rawState === 'string' ? rawState : 'unknown',
      unlocked: Boolean(
        audioEngine && (audioEngine.unlockedOnGesture || audioEngine.state === 'running')
      ),
      unlockedOnGesture: Boolean(audioEngine && audioEngine.unlockedOnGesture),
      gestureArmed: Boolean(audioGestureCleanup),
      gamepadPollActive: gamepadUnlockTimerId !== null,
      sampleRate:
        audioEngine &&
        typeof audioEngine.sampleRate === 'number' &&
        Number.isFinite(audioEngine.sampleRate)
          ? audioEngine.sampleRate
          : null,
      muted: Boolean(audioEngine && audioEngine.muted),
      volume:
        audioEngine && typeof audioEngine.volume === 'number' && Number.isFinite(audioEngine.volume)
          ? audioEngine.volume
          : 1.0,
      visibilityRecoveries: audioVisibilityRecoveries,
      visibilityState: visState,
    };
  }

  function isPointerLocked() {
    if (canvas && documentObj && 'pointerLockElement' in documentObj) {
      return documentObj.pointerLockElement === canvas;
    }
    if (inputBridge && typeof inputBridge.isPointerLocked === 'function') {
      try {
        return Boolean(inputBridge.isPointerLocked());
      } catch {
        return false;
      }
    }
    return false;
  }

  function exitPointerLock() {
    if (destroyed) {
      return { ok: false, locked: false, reason: 'destroyed' };
    }
    if (!isPointerLocked()) {
      return { ok: true, locked: false, alreadyUnlocked: true, reason: null };
    }
    if (!documentObj || typeof documentObj.exitPointerLock !== 'function') {
      return { ok: false, locked: isPointerLocked(), reason: 'unsupported' };
    }
    try {
      documentObj.exitPointerLock();
      const stillLocked = isPointerLocked();
      if (hud && typeof hud.updateMetrics === 'function') {
        try {
          hud.updateMetrics({ pointerLock: stillLocked });
        } catch {
          // Ignore HUD update errors.
        }
      }
      return { ok: true, locked: stillLocked, reason: null };
    } catch (err) {
      const msg = sanitizeString(err && err.message ? err.message : String(err));
      return { ok: false, locked: isPointerLocked(), reason: 'exit-failed', error: msg };
    }
  }

  function requestPointerLock(opts = {}) {
    if (destroyed) {
      return Promise.resolve({ ok: false, locked: false, requested: false, reason: 'destroyed' });
    }
    // Same user gesture seamlessly unlocks Web Audio if suspended/idle.
    if (opts.unlockAudio !== false && isAudioNeedsUnlock() && !audioGestureUnlockInFlight) {
      audioGestureUnlockAttempted = false;
      void runAudioUnlockAttempt({ source: 'pointer-lock' });
    }
    if (isPointerLocked()) {
      return Promise.resolve({
        ok: true,
        locked: true,
        requested: false,
        alreadyLocked: true,
        reason: null,
      });
    }
    if (!canvas || typeof canvas.requestPointerLock !== 'function') {
      return Promise.resolve({ ok: false, locked: false, requested: false, reason: 'unsupported' });
    }
    let result;
    try {
      result =
        opts && opts.unadjustedMovement !== undefined
          ? canvas.requestPointerLock({ unadjustedMovement: Boolean(opts.unadjustedMovement) })
          : canvas.requestPointerLock();
    } catch (err) {
      const msg = sanitizeString(err && err.message ? err.message : String(err));
      if (hud && typeof hud.appendLog === 'function') {
        try {
          hud.appendLog(`Pointer lock request failed: ${msg}`, 'warn');
        } catch {
          // Ignore HUD errors.
        }
      }
      return Promise.resolve({
        ok: false,
        locked: isPointerLocked(),
        requested: false,
        reason: 'request-failed',
        error: msg,
      });
    }

    if (result && typeof result.then === 'function') {
      return result.then(
        () => {
          const lockedNow = isPointerLocked();
          if (hud && typeof hud.updateMetrics === 'function') {
            try {
              hud.updateMetrics({ pointerLock: lockedNow });
            } catch {
              // Ignore HUD errors.
            }
          }
          return { ok: true, locked: lockedNow, requested: true, reason: null };
        },
        (err) => {
          const msg = sanitizeString(err && err.message ? err.message : String(err));
          if (hud && typeof hud.appendLog === 'function') {
            try {
              hud.appendLog(`Pointer lock request rejected: ${msg}`, 'warn');
            } catch {
              // Ignore HUD errors.
            }
          }
          return {
            ok: false,
            locked: isPointerLocked(),
            requested: false,
            reason: 'request-rejected',
            error: msg,
          };
        }
      );
    }

    const lockedNow = isPointerLocked();
    if (hud && typeof hud.updateMetrics === 'function') {
      try {
        hud.updateMetrics({ pointerLock: lockedNow });
      } catch {
        // Ignore HUD errors.
      }
    }
    return Promise.resolve({ ok: true, locked: lockedNow, requested: true, reason: null });
  }

  function togglePointerLock(opts = {}) {
    if (isPointerLocked()) {
      return Promise.resolve(exitPointerLock());
    }
    return requestPointerLock(opts);
  }

  function isPageHidden() {
    if (!documentObj || typeof documentObj !== 'object') {
      return false;
    }
    if (typeof documentObj.visibilityState === 'string') {
      return documentObj.visibilityState === 'hidden';
    }
    return documentObj.hidden === true;
  }

  function recoverAudioOnVisibility(triggerSource) {
    if (destroyed || !audioEngine || typeof audioEngine !== 'object' || audioVisibilityInFlight) {
      return;
    }
    if (isPageHidden()) {
      return;
    }
    const rawState = audioEngine.state;
    if (rawState === 'running') {
      notifyRuntimeAudioContextState('active');
      return;
    }
    const hasRecoverableContext = Boolean(
      audioEngine.context || rawState === 'suspended' || rawState === 'interrupted'
    );
    if (!hasRecoverableContext || (rawState !== 'suspended' && rawState !== 'interrupted')) {
      return;
    }

    audioVisibilityInFlight = true;
    audioGestureUnlockAttempted = false;
    audioGestureRetriesRemaining = Math.max(audioGestureRetriesRemaining, 5);
    runAudioUnlockAttempt({ source: triggerSource })
      .then((res) => {
        audioVisibilityInFlight = false;
        if (destroyed) {
          return;
        }
        if (res && res.ok) {
          audioVisibilityRecoveries += 1;
          notifyRuntimeAudioContextState('active');
        } else {
          notifyRuntimeAudioContextState('suspended');
          armAudioGestureUnlock();
        }
      })
      .catch(() => {
        audioVisibilityInFlight = false;
        if (!destroyed) {
          armAudioGestureUnlock();
        }
      });
  }

  function handleVisibilityChange() {
    if (destroyed) {
      return;
    }
    if (isPageHidden()) {
      if (isPointerLocked()) {
        exitPointerLock();
      }
      return;
    }
    recoverAudioOnVisibility('visibilitychange');
  }

  function handleWindowFocusOrShow(ev) {
    if (destroyed || isPageHidden()) {
      return;
    }
    const src = ev && typeof ev.type === 'string' ? ev.type : 'focus';
    recoverAudioOnVisibility(src);
  }

  function attachVisibilityAndPointerLockLifecycle() {
    const cleanups = [];
    if (isEventTargetLike(documentObj)) {
      try {
        documentObj.addEventListener('visibilitychange', handleVisibilityChange);
        cleanups.push(() => {
          documentObj.removeEventListener('visibilitychange', handleVisibilityChange);
        });
      } catch {
        // Ignore listener registration errors.
      }
      const onPointerLockError = () => {
        if (destroyed) return;
        if (hud && typeof hud.updateMetrics === 'function') {
          try {
            hud.updateMetrics({ pointerLock: isPointerLocked() });
          } catch {
            // Ignore HUD update errors.
          }
        }
      };
      try {
        documentObj.addEventListener('pointerlockerror', onPointerLockError);
        cleanups.push(() => {
          documentObj.removeEventListener('pointerlockerror', onPointerLockError);
        });
      } catch {
        // Ignore listener registration errors.
      }
    }
    if (isEventTargetLike(windowObj)) {
      try {
        windowObj.addEventListener('focus', handleWindowFocusOrShow);
        cleanups.push(() => {
          windowObj.removeEventListener('focus', handleWindowFocusOrShow);
        });
      } catch {
        // Ignore listener registration errors.
      }
      try {
        windowObj.addEventListener('pageshow', handleWindowFocusOrShow);
        cleanups.push(() => {
          windowObj.removeEventListener('pageshow', handleWindowFocusOrShow);
        });
      } catch {
        // Ignore listener registration errors.
      }
    }
    if (isEventTargetLike(canvas)) {
      const onCanvasClick = () => {
        if (destroyed) return;
        // Clicking the viewport unlocks audio on the same user gesture even if
        // the event does not bubble to window/document in embedded hosts.
        if (isAudioNeedsUnlock() && !audioGestureUnlockInFlight) {
          invokeAudioGestureUnlock({ type: 'click' });
        }
        // InputBridge already handles canvas click -> requestPointerLock when active;
        // only fall back here if InputBridge is disposed while session is running.
        if (
          options.autoPointerLockOnCanvasClick !== false &&
          inputBridge &&
          inputBridge.disposed &&
          !isPointerLocked() &&
          (session.getState() === SESSION_STATE.RUNNING_WASM ||
            session.getState() === SESSION_STATE.RUNNING_SMOKE)
        ) {
          void requestPointerLock({ unlockAudio: false });
        }
      };
      try {
        canvas.addEventListener('click', onCanvasClick, { passive: true });
        cleanups.push(() => {
          canvas.removeEventListener('click', onCanvasClick);
        });
      } catch {
        // Ignore listener registration errors.
      }
    }

    if (cleanups.length > 0) {
      visibilityLifecycleCleanup = () => {
        for (let i = 0; i < cleanups.length; i += 1) {
          try {
            cleanups[i]();
          } catch {
            // Ignore cleanup errors.
          }
        }
      };
    }
  }

  function detachVisibilityAndPointerLockLifecycle() {
    if (visibilityLifecycleCleanup) {
      const fn = visibilityLifecycleCleanup;
      visibilityLifecycleCleanup = null;
      fn();
    }
    if (pointerLockUiCleanup) {
      const fn = pointerLockUiCleanup;
      pointerLockUiCleanup = null;
      fn();
    }
  }

  armAudioGestureUnlock();
  attachVisibilityAndPointerLockLifecycle();

  function resolveCurrentBootMode() {
    if (session && typeof session.getBootMode === 'function') {
      try {
        const explicit = session.getBootMode();
        if (typeof explicit === 'string' && explicit.trim().length > 0) {
          return sanitizeDiagnosticText(explicit.trim(), 64);
        }
      } catch {
        // Fall through to mode inference.
      }
    }
    const sessionMode =
      session && typeof session.getMode === 'function' ? session.getMode() : SESSION_MODE.NONE;
    if (sessionMode === SESSION_MODE.WASM) {
      return 'wasm-guest';
    }
    if (sessionMode === SESSION_MODE.SMOKE) {
      return 'smoke-demo';
    }
    if (bootOrchestrator && typeof bootOrchestrator.getState === 'function') {
      try {
        const bState = bootOrchestrator.getState();
        if (bState && typeof bState.bootMode === 'string' && bState.bootMode !== 'pending') {
          return sanitizeDiagnosticText(bState.bootMode, 64);
        }
      } catch {
        // Ignore orchestrator state errors.
      }
    }
    return 'idle';
  }

  function resolveCurrentMilestone() {
    if (bootTracer && typeof bootTracer.getSummary === 'function') {
      try {
        const summary = bootTracer.getSummary();
        if (summary && typeof summary.stage === 'string' && summary.stage.length > 0) {
          return sanitizeDiagnosticText(summary.stage, 64);
        }
      } catch {
        // Ignore tracer errors.
      }
    }
    return 'NOT_STARTED';
  }

  function resolveSideModuleCount() {
    if (
      typeof latestDiscoveredBootAssets.sideModuleCount === 'number' &&
      Number.isFinite(latestDiscoveredBootAssets.sideModuleCount) &&
      latestDiscoveredBootAssets.sideModuleCount >= 0
    ) {
      return Math.floor(latestDiscoveredBootAssets.sideModuleCount);
    }
    if (Array.isArray(latestDiscoveredBootAssets.sideModuleUrls)) {
      return latestDiscoveredBootAssets.sideModuleUrls.length;
    }
    return 0;
  }

  function resolveFramesPresented() {
    let frames = wgpuFramesPresented;
    if (activeSmokeRunner && Number.isFinite(activeSmokeRunner.frameIndex)) {
      frames = Math.max(frames, Math.max(0, Math.floor(activeSmokeRunner.frameIndex)));
    }
    if (perfMonitor && typeof perfMonitor.getTotalFrames === 'function') {
      try {
        const total = perfMonitor.getTotalFrames();
        if (typeof total === 'number' && Number.isFinite(total) && total > 0) {
          frames = Math.max(frames, Math.floor(total));
        }
      } catch {
        // Ignore PerfMonitor errors.
      }
    } else if (perfMonitor && typeof perfMonitor.getSnapshot === 'function') {
      try {
        const snap = perfMonitor.getSnapshot();
        const total = snap && snap.frames && snap.frames.totalFrames;
        if (typeof total === 'number' && Number.isFinite(total) && total > 0) {
          frames = Math.max(frames, Math.floor(total));
        }
      } catch {
        // Ignore PerfMonitor errors.
      }
    }
    return frames;
  }

  function resolveSessionStateText() {
    let sessionState = SESSION_STATE.IDLE;
    if (session && typeof session.getState === 'function') {
      try {
        const rawState = session.getState();
        if (typeof rawState === 'string' && rawState.length > 0) {
          sessionState = sanitizeDiagnosticText(rawState, 64);
        }
      } catch {
        sessionState = SESSION_STATE.ERROR;
      }
    }
    return sessionState;
  }

  function resolveDiagnosticInputScript() {
    if (activeInputScript) {
      return sanitizeInputScriptPreset(activeInputScript);
    }
    if (inputBridge && typeof inputBridge.getScriptState === 'function') {
      try {
        const scriptState = inputBridge.getScriptState();
        if (scriptState && scriptState.active && typeof scriptState.preset === 'string') {
          const sanitized = sanitizeInputScriptPreset(scriptState.preset);
          if (sanitized) {
            return sanitized;
          }
        }
      } catch {
        // Ignore inputBridge state errors.
      }
    }
    return configuredInputScript ? sanitizeInputScriptPreset(configuredInputScript) : null;
  }

  function resolveTracerVisibleTiming() {
    let gpuAdapter = gpuTelemetry.gpuAdapter;
    let firstVisibleFrameMs = gpuTelemetry.firstVisibleFrameMs;
    let blackLoadingMs = gpuTelemetry.blackLoadingMs;
    if (
      (gpuAdapter === null || firstVisibleFrameMs === null || blackLoadingMs === null) &&
      bootTracer &&
      typeof bootTracer.getSummary === 'function'
    ) {
      try {
        const summary = bootTracer.getSummary();
        if (summary && typeof summary === 'object') {
          if (gpuAdapter === null && typeof summary.gpuAdapter === 'string' && summary.gpuAdapter) {
            gpuAdapter = summary.gpuAdapter;
          }
          if (
            firstVisibleFrameMs === null &&
            typeof summary.firstVisibleFrameMs === 'number' &&
            Number.isFinite(summary.firstVisibleFrameMs)
          ) {
            firstVisibleFrameMs = summary.firstVisibleFrameMs;
          }
          if (
            blackLoadingMs === null &&
            typeof summary.blackLoadingMs === 'number' &&
            Number.isFinite(summary.blackLoadingMs)
          ) {
            blackLoadingMs = summary.blackLoadingMs;
          }
        }
      } catch {
        // Ignore tracer errors.
      }
    }
    return { gpuAdapter, firstVisibleFrameMs, blackLoadingMs };
  }

  function resolveFastBootTimings() {
    syncFastBootMilestonesFromTracer();
    return {
      bootToStartScreenMs: fastBootMilestones.START_SCREEN,
      bootToIntroVideoMs: fastBootMilestones.INTRO_VIDEO,
      bootToMainMenuMs: fastBootMilestones.MAIN_MENU,
      bootToInGameMs: fastBootMilestones.IN_GAME,
    };
  }

  function resolvePerfTelemetrySnapshot() {
    let fps30ComplianceRate = perfTelemetry.fps30ComplianceRate;
    let dominantBottleneck = perfTelemetry.dominantBottleneck;
    let targetFps = DEFAULT_TARGET_FPS;
    let budgetMs = DEFAULT_FRAME_BUDGET_MS;

    if (perfMonitor && typeof perfMonitor.getSnapshot === 'function') {
      try {
        const snap = perfMonitor.getSnapshot();
        if (snap && typeof snap === 'object') {
          if (typeof snap.targetFps === 'number' && Number.isFinite(snap.targetFps) && snap.targetFps > 0) {
            targetFps = snap.targetFps;
          }
          if (typeof snap.budgetMs === 'number' && Number.isFinite(snap.budgetMs) && snap.budgetMs > 0) {
            budgetMs = snap.budgetMs;
          }
          const within = sanitizeNonNegativeFiniteNumber(snap.framesWithin30FpsBudget) ?? 0;
          const over = sanitizeNonNegativeFiniteNumber(snap.framesOver30FpsBudget) ?? 0;
          if (within + over > 0) {
            if (fps30ComplianceRate === null) {
              fps30ComplianceRate = sanitizeUnitInterval(snap.fps30ComplianceRate);
            }
            if (dominantBottleneck === null) {
              dominantBottleneck = normalizeDominantBottleneck(snap.dominantBottleneck);
            }
          }
        }
      } catch {
        // Ignore PerfMonitor snapshot errors.
      }
    }

    let jsHeapUsedMb = perfTelemetry.jsHeapUsedMb;
    let jsHeapTotalMb = perfTelemetry.jsHeapTotalMb;
    if (jsHeapUsedMb === null || jsHeapTotalMb === null) {
      const sampled = sampleJsHeapMemory();
      if (sampled) {
        if (jsHeapUsedMb === null && sampled.jsHeapUsedMb !== null) {
          jsHeapUsedMb = sampled.jsHeapUsedMb;
        }
        if (jsHeapTotalMb === null && sampled.jsHeapTotalMb !== null) {
          jsHeapTotalMb = sampled.jsHeapTotalMb;
        }
      }
    }

    return Object.freeze({
      targetFps,
      budgetMs,
      fps30ComplianceRate,
      dominantBottleneck,
      frameJitterMs: perfTelemetry.frameJitterMs,
      jsHeapUsedMb,
      jsHeapTotalMb,
      vramBudgetUtilization: perfTelemetry.vramBudgetUtilization,
      bindGroupHitRate: perfTelemetry.bindGroupHitRate,
      mailboxLanes: perfTelemetry.mailboxLanes,
    });
  }

  function resolveAudioRingSnapshot() {
    let attached = Boolean(audioRingTelemetry.audioRingAttached);
    let attempts = toSafeNonNegativeInt(audioRingTelemetry.audioRingAttachAttempts, 0);
    let lastReason = audioRingTelemetry.audioRingLastReason;
    if (audioRingController && typeof audioRingController.getTelemetry === 'function') {
      try {
        const ctrlTel = audioRingController.getTelemetry();
        if (ctrlTel && typeof ctrlTel === 'object') {
          if (ctrlTel.audioRingAttached === true) {
            attached = true;
          }
          const explicitAttempts = toNonNegativeInt(ctrlTel.audioRingAttachAttempts, null);
          if (explicitAttempts !== null) {
            attempts = Math.max(attempts, explicitAttempts);
          } else {
            const retries = toNonNegativeInt(ctrlTel.audioRingRetryCount, null);
            if (retries !== null) {
              const inferred =
                ctrlTel.audioRingAttached === true || ctrlTel.audioRingLastReason !== null
                  ? retries + 1
                  : retries;
              attempts = Math.max(attempts, inferred);
            }
          }
          if (lastReason === null && typeof ctrlTel.audioRingLastReason === 'string') {
            lastReason = sanitizeDiagnosticText(ctrlTel.audioRingLastReason, 64) || null;
          }
        }
      } catch {
        // Ignore audioRingController telemetry errors.
      }
    }
    return Object.freeze({
      audioRingAttached: attached,
      audioRingAttachAttempts: attempts,
      audioRingLastReason: lastReason,
    });
  }

  function getPhaseTelemetry() {
    let raw = null;
    if (session && typeof session.getPhaseTelemetry === 'function') {
      try {
        raw = session.getPhaseTelemetry();
      } catch {
        raw = null;
      }
    }
    let sessionStats = null;
    if (!raw && session && typeof session.getStats === 'function') {
      try {
        sessionStats = session.getStats();
      } catch {
        sessionStats = null;
      }
    }
    let rawDurations =
      (raw && raw.phaseDurationsMs && typeof raw.phaseDurationsMs === 'object'
        ? raw.phaseDurationsMs
        : null) ||
      (sessionStats && sessionStats.phaseDurationsMs && typeof sessionStats.phaseDurationsMs === 'object'
        ? sessionStats.phaseDurationsMs
        : null);
    if (!rawDurations && session && typeof session.getPhaseDurations === 'function') {
      try {
        rawDurations = session.getPhaseDurations();
      } catch {
        rawDurations = null;
      }
    }

    const sanitizedDurations = {};
    for (let i = 0; i < CANONICAL_SESSION_PHASES.length; i += 1) {
      const phaseKey = CANONICAL_SESSION_PHASES[i];
      const val = rawDurations ? sanitizeNonNegativeFiniteNumber(rawDurations[phaseKey]) : null;
      sanitizedDurations[phaseKey] = val !== null ? val : 0;
    }

    let currentPhase = null;
    const rawCurrentPhase =
      (raw && typeof raw.currentPhase === 'string' ? raw.currentPhase : null) ??
      (sessionStats && typeof sessionStats.currentPhase === 'string'
        ? sessionStats.currentPhase
        : null) ??
      (session && typeof session.getCurrentPhase === 'function'
        ? (() => {
            try {
              return session.getCurrentPhase();
            } catch {
              return null;
            }
          })()
        : null);
    if (typeof rawCurrentPhase === 'string' && rawCurrentPhase.length > 0) {
      if (CANONICAL_SESSION_PHASE_SET.has(rawCurrentPhase)) {
        currentPhase = rawCurrentPhase;
      } else {
        const trimmed = rawCurrentPhase.trim();
        if (trimmed.length > 0) {
          currentPhase = sanitizeDiagnosticText(trimmed, 64) || null;
        }
      }
    }

    const phaseTransitionCount = toSafeNonNegativeInt(
      (raw && raw.phaseTransitionCount) ?? (sessionStats && sessionStats.phaseTransitionCount),
      0
    );
    const reclaimRuns = toSafeNonNegativeInt(
      (raw && raw.reclaimRuns) ?? (sessionStats && sessionStats.reclaimRuns) ?? fallbackReclaimRuns,
      0
    );
    const reclaimFailures = toSafeNonNegativeInt(
      (raw && raw.reclaimFailures) ??
        (sessionStats && sessionStats.reclaimFailures) ??
        fallbackReclaimFailures,
      0
    );
    const rawLastReason =
      (raw && raw.lastReclaimReason) ??
      (sessionStats && sessionStats.lastReclaimReason) ??
      fallbackLastReclaimReason;
    const lastReclaimReason =
      typeof rawLastReason === 'string' && rawLastReason.trim().length > 0
        ? sanitizeDiagnosticText(rawLastReason.trim(), 128)
        : null;
    const lastReclaimAtMs = sanitizeNonNegativeFiniteNumber(
      (raw && raw.lastReclaimAtMs) ??
        (sessionStats && sessionStats.lastReclaimAtMs) ??
        fallbackLastReclaimAtMs
    );

    return Object.freeze({
      currentPhase,
      phaseDurationsMs: Object.freeze(sanitizedDurations),
      phaseTransitionCount,
      reclaimRuns,
      reclaimFailures,
      lastReclaimReason,
      lastReclaimAtMs,
    });
  }

  function resolvePreloadTelemetrySnapshot() {
    let orchStats = null;
    if (bootOrchestrator) {
      try {
        if (typeof bootOrchestrator.getStats === 'function') {
          orchStats = bootOrchestrator.getStats();
        } else if (typeof bootOrchestrator.getTimingStats === 'function') {
          orchStats = bootOrchestrator.getTimingStats();
        } else if (typeof bootOrchestrator.getPreloadStats === 'function') {
          orchStats = { preloadStats: bootOrchestrator.getPreloadStats() };
        }
      } catch {
        orchStats = null;
      }
    }
    let preloaderStats = null;
    if (pgimPreloader && typeof pgimPreloader.getStats === 'function') {
      try {
        preloaderStats = pgimPreloader.getStats();
      } catch {
        preloaderStats = null;
      }
    }
    if (!orchStats && !preloaderStats) {
      return null;
    }
    const pStats =
      orchStats && orchStats.preloadStats && typeof orchStats.preloadStats === 'object'
        ? orchStats.preloadStats
        : {};
    const stageDurations = {};
    if (orchStats && orchStats.stageDurationsMs && typeof orchStats.stageDurationsMs === 'object') {
      const keys = Object.keys(orchStats.stageDurationsMs);
      for (let i = 0; i < keys.length; i += 1) {
        const key = sanitizeDiagnosticText(keys[i], 64);
        const val = sanitizeNonNegativeFiniteNumber(orchStats.stageDurationsMs[keys[i]]);
        if (key && val !== null) {
          stageDurations[key] = val;
        }
      }
    }
    return Object.freeze({
      parallelPreload:
        orchStats && typeof orchStats.parallelPreload === 'boolean'
          ? orchStats.parallelPreload
          : orchStats && typeof orchStats.parallelPreloadEnabled === 'boolean'
            ? orchStats.parallelPreloadEnabled
            : Boolean(pStats.parallelEnabled),
      preloadConcurrency: toSafeNonNegativeInt(
        (orchStats && orchStats.preloadConcurrency) ?? pStats.concurrencyLimit,
        0
      ),
      serialEstimateMs:
        sanitizeNonNegativeFiniteNumber(
          (orchStats && orchStats.serialEstimateMs) ?? pStats.serialEstimateMs
        ) ?? 0,
      parallelWallMs:
        sanitizeNonNegativeFiniteNumber(
          (orchStats && orchStats.parallelWallMs) ?? pStats.parallelWallMs
        ) ?? 0,
      overlapSavedMs:
        sanitizeNonNegativeFiniteNumber(
          (orchStats && orchStats.overlapSavedMs) ??
            (orchStats && orchStats.parallelPreloadSavedMs) ??
            pStats.overlapSavedMs
        ) ?? 0,
      parallelStagesCount: toSafeNonNegativeInt(
        (orchStats && orchStats.parallelStagesCount) ?? pStats.parallelStagesCount,
        0
      ),
      totalBootMs: sanitizeNonNegativeFiniteNumber(orchStats && orchStats.totalBootMs) ?? 0,
      stageDurationsMs: Object.freeze(stageDurations),
      opfsHits: toSafeNonNegativeInt(preloaderStats && preloaderStats.opfsHits, 0),
      opfsMisses: toSafeNonNegativeInt(preloaderStats && preloaderStats.opfsMisses, 0),
      opfsWrites: toSafeNonNegativeInt(preloaderStats && preloaderStats.opfsWrites, 0),
      opfsInvalidations: toSafeNonNegativeInt(preloaderStats && preloaderStats.opfsInvalidations, 0),
      bytesFromOpfs: toSafeNonNegativeInt(preloaderStats && preloaderStats.bytesFromOpfs, 0),
      bytesFromNetwork: toSafeNonNegativeInt(preloaderStats && preloaderStats.bytesFromNetwork, 0),
      bytesLoaded: toSafeNonNegativeInt(preloaderStats && preloaderStats.bytesLoaded, 0),
      totalBytes: toSafeNonNegativeInt(preloaderStats && preloaderStats.totalBytes, 0),
      throughputMBps: sanitizeNonNegativeFiniteNumber(preloaderStats && preloaderStats.throughputMBps) ?? 0,
    });
  }

  function resolveSplashTelemetrySnapshot() {
    if (!splashOverlay || typeof splashOverlay !== 'object') {
      return null;
    }
    let raw = null;
    try {
      if (typeof splashOverlay.getStats === 'function') {
        raw = splashOverlay.getStats();
      } else if (typeof splashOverlay.getState === 'function') {
        raw = splashOverlay.getState();
      }
    } catch {
      raw = null;
    }
    if (!raw || typeof raw !== 'object') {
      return null;
    }
    const stageDurations = {};
    if (raw.stageDurationsMs && typeof raw.stageDurationsMs === 'object') {
      const keys = Object.keys(raw.stageDurationsMs);
      for (let i = 0; i < keys.length; i += 1) {
        const key = sanitizeDiagnosticText(keys[i], 64);
        const val = sanitizeNonNegativeFiniteNumber(raw.stageDurationsMs[keys[i]]);
        if (key && val !== null) {
          stageDurations[key] = val;
        }
      }
    }
    return Object.freeze({
      phase:
        typeof raw.phase === 'string' && raw.phase.length > 0
          ? sanitizeDiagnosticText(raw.phase, 64)
          : null,
      stage:
        typeof raw.stage === 'string' && raw.stage.length > 0
          ? sanitizeDiagnosticText(raw.stage, 64)
          : null,
      progressPct: toSafeNonNegativeInt(raw.progressPct, 0),
      fadeOutMs:
        sanitizeNonNegativeFiniteNumber(raw.fadeOutMs) ?? resolvedSplashFadeOutMs,
      fadeOutCompleted: Boolean(raw.fadeOutCompleted),
      visible: Boolean(raw.visible),
      dismissed: Boolean(raw.dismissed),
      totalSplashMs: sanitizeNonNegativeFiniteNumber(raw.totalSplashMs) ?? 0,
      stageDurationsMs: Object.freeze(stageDurations),
    });
  }

  function getDiagnosticsSnapshot() {
    const sessionState = resolveSessionStateText();
    const recentCopy = Object.freeze(
      recentValidationErrors.map((entry) => Object.freeze({ ...entry }))
    );
    const { gpuAdapter, firstVisibleFrameMs, blackLoadingMs } = resolveTracerVisibleTiming();
    const {
      bootToStartScreenMs,
      bootToIntroVideoMs,
      bootToMainMenuMs,
      bootToInGameMs,
    } = resolveFastBootTimings();
    const perf = resolvePerfTelemetrySnapshot();
    const audioRing = resolveAudioRingSnapshot();
    const phaseTelemetry = getPhaseTelemetry();
    const preloadTelemetry = resolvePreloadTelemetrySnapshot();
    const splashTelemetry = resolveSplashTelemetrySnapshot();
    const loadingAssets = isSplashOverlayHolding();
    return Object.freeze({
      bootMode: resolveCurrentBootMode(),
      sessionState,
      sessionPhase: phaseTelemetry.currentPhase,
      phaseDurationsMs: phaseTelemetry.phaseDurationsMs,
      phaseTelemetry,
      reclaimRuns: phaseTelemetry.reclaimRuns,
      reclaimFailures: phaseTelemetry.reclaimFailures,
      milestone: resolveCurrentMilestone(),
      sideModuleCount: resolveSideModuleCount(),
      validationErrorCount,
      recentValidationErrors: recentCopy,
      framesPresented: resolveFramesPresented(),
      splashFramesPresented,
      gpuAdapter,
      hasVisibleContent: Boolean(gpuTelemetry.hasVisibleContent),
      firstVisibleFrameMs,
      blackLoadingMs,
      swapchainProbeCount: toSafeNonNegativeInt(gpuTelemetry.swapchainProbeCount, 0),
      swapchainProbeInFlight: Boolean(gpuTelemetry.swapchainProbeInFlight),
      swapchainProbeSupported:
        typeof gpuTelemetry.swapchainProbeSupported === 'boolean'
          ? gpuTelemetry.swapchainProbeSupported
          : null,
      swapchainProbeErrorCount: toSafeNonNegativeInt(gpuTelemetry.swapchainProbeErrorCount, 0),
      swapchainProbeTimeoutCount: toSafeNonNegativeInt(gpuTelemetry.swapchainProbeTimeoutCount, 0),
      lastSampledMaxLuma: Math.max(
        0,
        Math.min(255, toSafeNonNegativeInt(gpuTelemetry.lastSampledMaxLuma, 0))
      ),
      suppressedBlackFrames: toSafeNonNegativeInt(gpuTelemetry.suppressedBlackFrames, 0),
      coalescedPresents: toSafeNonNegativeInt(gpuTelemetry.coalescedPresents, 0),
      oversizedWgslFallbacks: toSafeNonNegativeInt(gpuTelemetry.oversizedWgslFallbacks, 0),
      deviceLostCount: toSafeNonNegativeInt(gpuTelemetry.deviceLostCount, 0),
      deviceRecoveredCount: toSafeNonNegativeInt(gpuTelemetry.deviceRecoveredCount, 0),
      gpuStallWarnCount: toSafeNonNegativeInt(gpuTelemetry.gpuStallWarnCount, 0),
      rangedPackActive:
        typeof gpuTelemetry.rangedPackActive === 'boolean' ? gpuTelemetry.rangedPackActive : null,
      gpuMemory: gpuTelemetry.gpuMemory ? Object.freeze({ ...gpuTelemetry.gpuMemory }) : null,
      bootToStartScreenMs,
      bootToIntroVideoMs,
      bootToMainMenuMs,
      bootToInGameMs,
      audioRingAttached: audioRing.audioRingAttached,
      audioRingAttachAttempts: audioRing.audioRingAttachAttempts,
      fps30ComplianceRate: perf.fps30ComplianceRate,
      dominantBottleneck: perf.dominantBottleneck,
      frameJitterMs: perf.frameJitterMs,
      jsHeapUsedMb: perf.jsHeapUsedMb,
      vramBudgetUtilization: perf.vramBudgetUtilization,
      bindGroupHitRate: perf.bindGroupHitRate,
      mailboxLanes: perf.mailboxLanes,
      mailboxAttached: latestIoWorkerStats ? Boolean(latestIoWorkerStats.mailboxAttached) : null,
      ioWorkerStats: latestIoWorkerStats,
      runtimeDiagnostics: latestRuntimeDiagnostics,
      fastBootPlan: latestFastBootPlan,
      preloadTelemetry,
      splashTelemetry,
      loadingAssets,
      holdUntilVisible: Boolean(resolvedHoldUntilVisible),
      dumpPipelines: Boolean(activeDumpConfig && activeDumpConfig.dumpPipelines),
      allowDumpWrites: Boolean(activeDumpConfig && activeDumpConfig.allowDumpWrites),
      inputScript: resolveDiagnosticInputScript(),
      destroyed,
    });
  }

  function getBootTelemetry() {
    const sessionState = resolveSessionStateText();
    const effectivePresentCount = Math.max(gpuTelemetry.presentCount, resolveFramesPresented());
    const effectiveAudioOutputState = audioTelemetry.hasReceivedStats
      ? audioTelemetry.outputState
      : normalizeAudioOutputState(audioEngine && audioEngine.state);
    const { gpuAdapter, firstVisibleFrameMs, blackLoadingMs } = resolveTracerVisibleTiming();
    const {
      bootToStartScreenMs,
      bootToIntroVideoMs,
      bootToMainMenuMs,
      bootToInGameMs,
    } = resolveFastBootTimings();
    const perf = resolvePerfTelemetrySnapshot();
    const audioRing = resolveAudioRingSnapshot();
    const phaseTelemetry = getPhaseTelemetry();
    const preloadTelemetry = resolvePreloadTelemetrySnapshot();
    const splashTelemetry = resolveSplashTelemetrySnapshot();
    const loadingAssets = isSplashOverlayHolding();

    const gpu = Object.freeze({
      realPackDrawCount: toSafeNonNegativeInt(gpuTelemetry.realPackDrawCount, 0),
      fallbackDrawCount: toSafeNonNegativeInt(gpuTelemetry.fallbackDrawCount, 0),
      presentCount: toSafeNonNegativeInt(effectivePresentCount, 0),
      unboundBindGroupCount: toSafeNonNegativeInt(gpuTelemetry.unboundBindGroupCount, 0),
      skippedDrawCount: toSafeNonNegativeInt(gpuTelemetry.skippedDrawCount, 0),
      validationErrorCount: toSafeNonNegativeInt(validationErrorCount, 0),
      lastValidationErrorCode: gpuTelemetry.lastValidationErrorCode,
      lastValidationErrorMessage: gpuTelemetry.lastValidationErrorMessage,
      gpuAdapter,
      hasVisibleContent: Boolean(gpuTelemetry.hasVisibleContent),
      firstVisibleFrameMs,
      blackLoadingMs,
      loadingAssets,
      swapchainProbeCount: toSafeNonNegativeInt(gpuTelemetry.swapchainProbeCount, 0),
      swapchainProbeInFlight: Boolean(gpuTelemetry.swapchainProbeInFlight),
      swapchainProbeSupported:
        typeof gpuTelemetry.swapchainProbeSupported === 'boolean'
          ? gpuTelemetry.swapchainProbeSupported
          : null,
      swapchainProbeErrorCount: toSafeNonNegativeInt(gpuTelemetry.swapchainProbeErrorCount, 0),
      swapchainProbeTimeoutCount: toSafeNonNegativeInt(gpuTelemetry.swapchainProbeTimeoutCount, 0),
      lastSampledMaxLuma: Math.max(
        0,
        Math.min(255, toSafeNonNegativeInt(gpuTelemetry.lastSampledMaxLuma, 0))
      ),
      suppressedBlackFrames: toSafeNonNegativeInt(gpuTelemetry.suppressedBlackFrames, 0),
      coalescedPresents: toSafeNonNegativeInt(gpuTelemetry.coalescedPresents, 0),
      oversizedWgslFallbacks: toSafeNonNegativeInt(gpuTelemetry.oversizedWgslFallbacks, 0),
      deviceLostCount: toSafeNonNegativeInt(gpuTelemetry.deviceLostCount, 0),
      deviceRecoveredCount: toSafeNonNegativeInt(gpuTelemetry.deviceRecoveredCount, 0),
      gpuStallWarnCount: toSafeNonNegativeInt(gpuTelemetry.gpuStallWarnCount, 0),
      rangedPackActive:
        typeof gpuTelemetry.rangedPackActive === 'boolean' ? gpuTelemetry.rangedPackActive : null,
      gpuMemory: gpuTelemetry.gpuMemory ? Object.freeze({ ...gpuTelemetry.gpuMemory }) : null,
    });

    const audio = Object.freeze({
      outputState: effectiveAudioOutputState,
      framesDrained: toSafeNonNegativeInt(audioTelemetry.framesDrained, 0),
      underruns: toSafeNonNegativeInt(audioTelemetry.underruns, 0),
      overruns: toSafeNonNegativeInt(audioTelemetry.overruns, 0),
    });

    const loader = Object.freeze({
      indirectDispatch: loaderTelemetry.indirectDispatch,
      pthreadSpawnedCount: toSafeNonNegativeInt(loaderTelemetry.pthreadSpawnedCount, 0),
      pthreadActiveCount: toSafeNonNegativeInt(loaderTelemetry.pthreadActiveCount, 0),
      pthreadPeakCount: toSafeNonNegativeInt(loaderTelemetry.pthreadPeakCount, 0),
      pthreadLoadQueueDepth: toSafeNonNegativeInt(loaderTelemetry.pthreadLoadQueueDepth, 0),
      sideModuleCount: toSafeNonNegativeInt(resolveSideModuleCount(), 0),
      preferredShaderPackUrl: resolveSanitizedPreferredShaderPackUrl(),
    });

    const fastBootMilestonesSnap = Object.freeze({
      START_SCREEN: bootToStartScreenMs,
      INTRO_VIDEO: bootToIntroVideoMs,
      MAIN_MENU: bootToMainMenuMs,
      IN_GAME: bootToInGameMs,
    });

    const storage = Object.freeze({
      fastBootPlan: latestFastBootPlan,
      quotaEviction: latestQuotaEviction,
    });

    return Object.freeze({
      bootMode: resolveCurrentBootMode(),
      sessionState,
      sessionPhase: phaseTelemetry.currentPhase,
      phaseDurationsMs: phaseTelemetry.phaseDurationsMs,
      phaseTelemetry,
      reclaimRuns: phaseTelemetry.reclaimRuns,
      reclaimFailures: phaseTelemetry.reclaimFailures,
      milestone: resolveCurrentMilestone(),
      destroyed: Boolean(destroyed),
      dumpPipelines: Boolean(activeDumpConfig && activeDumpConfig.dumpPipelines),
      allowDumpWrites: Boolean(activeDumpConfig && activeDumpConfig.allowDumpWrites),
      inputScript: resolveDiagnosticInputScript(),
      holdUntilVisible: Boolean(resolvedHoldUntilVisible),
      gpuAdapter: gpu.gpuAdapter,
      hasVisibleContent: gpu.hasVisibleContent,
      firstVisibleFrameMs: gpu.firstVisibleFrameMs,
      blackLoadingMs: gpu.blackLoadingMs,
      swapchainProbeCount: gpu.swapchainProbeCount,
      swapchainProbeInFlight: gpu.swapchainProbeInFlight,
      swapchainProbeSupported: gpu.swapchainProbeSupported,
      swapchainProbeErrorCount: gpu.swapchainProbeErrorCount,
      swapchainProbeTimeoutCount: gpu.swapchainProbeTimeoutCount,
      lastSampledMaxLuma: gpu.lastSampledMaxLuma,
      suppressedBlackFrames: gpu.suppressedBlackFrames,
      coalescedPresents: gpu.coalescedPresents,
      oversizedWgslFallbacks: gpu.oversizedWgslFallbacks,
      deviceLostCount: gpu.deviceLostCount,
      deviceRecoveredCount: gpu.deviceRecoveredCount,
      gpuStallWarnCount: gpu.gpuStallWarnCount,
      rangedPackActive: gpu.rangedPackActive,
      gpuMemory: gpu.gpuMemory,
      bootToStartScreenMs,
      bootToIntroVideoMs,
      bootToMainMenuMs,
      bootToInGameMs,
      fastBootMilestones: fastBootMilestonesSnap,
      audioRingAttached: audioRing.audioRingAttached,
      audioRingAttachAttempts: audioRing.audioRingAttachAttempts,
      fps30ComplianceRate: perf.fps30ComplianceRate,
      dominantBottleneck: perf.dominantBottleneck,
      frameJitterMs: perf.frameJitterMs,
      jsHeapUsedMb: perf.jsHeapUsedMb,
      jsHeapTotalMb: perf.jsHeapTotalMb,
      vramBudgetUtilization: perf.vramBudgetUtilization,
      bindGroupHitRate: perf.bindGroupHitRate,
      mailboxLanes: perf.mailboxLanes,
      mailboxAttached: latestIoWorkerStats ? Boolean(latestIoWorkerStats.mailboxAttached) : null,
      ioWorkerStats: latestIoWorkerStats,
      runtimeDiagnostics: latestRuntimeDiagnostics,
      fastBootPlan: latestFastBootPlan,
      quotaEviction: latestQuotaEviction,
      preloadTelemetry,
      splashTelemetry,
      loadingAssets: gpu.loadingAssets,
      realPackDrawCount: gpu.realPackDrawCount,
      fallbackDrawCount: gpu.fallbackDrawCount,
      presentCount: gpu.presentCount,
      unboundBindGroupCount: gpu.unboundBindGroupCount,
      skippedDrawCount: gpu.skippedDrawCount,
      validationErrorCount: gpu.validationErrorCount,
      lastValidationErrorCode: gpu.lastValidationErrorCode,
      lastValidationErrorMessage: gpu.lastValidationErrorMessage,
      outputState: audio.outputState,
      framesDrained: audio.framesDrained,
      underruns: audio.underruns,
      overruns: audio.overruns,
      indirectDispatch: loader.indirectDispatch,
      pthreadSpawnedCount: loader.pthreadSpawnedCount,
      pthreadActiveCount: loader.pthreadActiveCount,
      pthreadPeakCount: loader.pthreadPeakCount,
      pthreadLoadQueueDepth: loader.pthreadLoadQueueDepth,
      sideModuleCount: loader.sideModuleCount,
      preferredShaderPackUrl: loader.preferredShaderPackUrl,
      gpu,
      audio,
      loader,
      perf,
      storage,
    });
  }

  const globalScope =
    options.globalScope !== undefined ? options.globalScope : windowObj;
  const diagnosticHook = Object.freeze({
    getDiagnosticsSnapshot,
  });
  const bootStateGetter = () => getBootTelemetry();
  const telemetryHook = Object.freeze(
    Object.assign(
      function getPlayGta6BootTelemetry() {
        return getBootTelemetry();
      },
      {
        getBootTelemetry,
        getSnapshot: getBootTelemetry,
      }
    )
  );
  const runtimeGlobalThis =
    typeof globalThis !== 'undefined' && globalThis && typeof globalThis === 'object'
      ? globalThis
      : null;
  let attachedGlobalHook = false;
  let attachedBootStateHook = false;
  let attachedGlobalScopeTelemetryHook = false;
  let attachedGlobalThisTelemetryHook = false;
  if (options.attachGlobalHook !== false) {
    if (globalScope && typeof globalScope === 'object') {
      try {
        globalScope.__playgta6 = diagnosticHook;
        attachedGlobalHook = true;
      } catch {
        // Ignore frozen or non-extensible global objects.
      }
      try {
        Object.defineProperty(globalScope, '__playgta6BootState', {
          configurable: true,
          enumerable: true,
          get: bootStateGetter,
        });
        attachedBootStateHook = true;
      } catch {
        // Ignore frozen or non-extensible global objects.
      }
      if (globalScope !== runtimeGlobalThis) {
        try {
          globalScope.__playgta6Telemetry = telemetryHook;
          attachedGlobalScopeTelemetryHook = true;
        } catch {
          // Ignore frozen or non-extensible global objects.
        }
      }
    }
    if (runtimeGlobalThis) {
      try {
        runtimeGlobalThis.__playgta6Telemetry = telemetryHook;
        attachedGlobalThisTelemetryHook = true;
      } catch {
        // Ignore frozen or non-extensible globalThis objects.
      }
    }
  }

  function stopActiveInputScript() {
    const hasRunningScript =
      activeInputScript !== null ||
      (inputBridge &&
        ((typeof inputBridge.getScriptState === 'function' &&
          Boolean(inputBridge.getScriptState()?.active)) ||
          (typeof inputBridge.getStats === 'function' &&
            Boolean(inputBridge.getStats()?.scriptPending))));
    activeInputScript = null;
    if (!hasRunningScript || !inputBridge || typeof inputBridge.stopScript !== 'function') {
      return;
    }
    try {
      inputBridge.stopScript();
    } catch (err) {
      crashReporter.recordError('runtime', err, {
        source: 'inputBridge',
        phase: 'stopScript',
      });
    }
  }

  function startActiveInputScript(preset) {
    if (destroyed || !preset) {
      return false;
    }
    if (!inputBridge || typeof inputBridge.startScript !== 'function') {
      return false;
    }
    try {
      if (
        typeof gpuTelemetry.swapchainProbeSupported === 'boolean' &&
        typeof inputBridge.setSwapchainProbeSupported === 'function'
      ) {
        inputBridge.setSwapchainProbeSupported(gpuTelemetry.swapchainProbeSupported);
      }
      const scriptOptions = resolveInputScriptStartOptions(preset, options.inputScriptOptions);
      if (scriptOptions) {
        inputBridge.startScript(preset, scriptOptions);
      } else {
        inputBridge.startScript(preset);
      }
      activeInputScript = preset;
      if (hud && typeof hud.appendLog === 'function') {
        try {
          hud.appendLog(`Input script active: ${preset}`, 'info');
        } catch {
          // Ignore HUD errors.
        }
      }
      return true;
    } catch (err) {
      activeInputScript = null;
      crashReporter.recordError('runtime', err, {
        source: 'inputBridge',
        phase: 'startScript',
        preset,
      });
      return false;
    }
  }

  async function initializeFastBootStoragePlan() {
    if (destroyed || !storageManager || typeof storageManager.getFastBootPlan !== 'function') {
      return null;
    }
    latestFastBootPlan = null;
    latestQuotaEviction = null;

    const planOpts = {};
    const rawMemory = readRawPerformanceMemory();
    if (rawMemory) {
      planOpts.memory = rawMemory;
    }
    if (
      navigatorObj &&
      typeof navigatorObj === 'object' &&
      typeof navigatorObj.deviceMemory === 'number' &&
      Number.isFinite(navigatorObj.deviceMemory) &&
      navigatorObj.deviceMemory > 0
    ) {
      planOpts.deviceMemoryGb = navigatorObj.deviceMemory;
    }

    let rawPlan = null;
    try {
      rawPlan =
        Object.keys(planOpts).length > 0
          ? await storageManager.getFastBootPlan(planOpts)
          : await storageManager.getFastBootPlan();
    } catch {
      return null;
    }

    if (destroyed || !rawPlan || typeof rawPlan !== 'object') {
      return null;
    }

    latestFastBootPlan = Object.freeze({
      warmCacheReady: Boolean(rawPlan.warmCacheReady),
      recommendedSlabMaxBytes: toSafeNonNegativeInt(rawPlan.recommendedSlabMaxBytes, 0),
      recommendedOpfsBudgetBytes: toSafeNonNegativeInt(rawPlan.recommendedOpfsBudgetBytes, 0),
      quotaHeadroomBytes: toSafeNonNegativeInt(rawPlan.quotaHeadroomBytes, 0),
      pressureLevel: sanitizeFastBootPressureLevel(rawPlan.pressureLevel),
      shouldEvictColdChunks: Boolean(rawPlan.shouldEvictColdChunks),
    });

    if (
      latestFastBootPlan.shouldEvictColdChunks &&
      typeof storageManager.runQuotaPressureEviction === 'function'
    ) {
      try {
        const rawEviction = await storageManager.runQuotaPressureEviction();
        if (!destroyed && rawEviction && typeof rawEviction === 'object') {
          latestQuotaEviction = Object.freeze({
            triggered: true,
            ok: rawEviction.ok !== false,
            evictedBytes: toSafeNonNegativeInt(rawEviction.evictedBytes, 0),
            evictedEntries: toSafeNonNegativeInt(rawEviction.evictedEntries, 0),
          });
        }
      } catch {
        if (!destroyed) {
          latestQuotaEviction = Object.freeze({
            triggered: true,
            ok: false,
            evictedBytes: 0,
            evictedEntries: 0,
          });
        }
      }
    }

    if (!destroyed) {
      const hudTargets = resolveHudTargets();
      for (let i = 0; i < hudTargets.length; i += 1) {
        invokeFirstHook(
          hudTargets[i],
          ['setFastBootPlan', 'updateFastBootPlan'],
          latestFastBootPlan
        );
      }
    }

    return latestFastBootPlan;
  }

  async function start(startOpts) {
    if (destroyed) {
      return {
        ok: false,
        destroyed: true,
        state: SESSION_STATE.STOPPED,
        mode: SESSION_MODE.NONE,
        error: new Error('PlayGta6AppHandle has been destroyed'),
      };
    }
    stopActiveInputScript();
    activeStreamConfig = resolveStreamConfigOverride(startOpts, baseStreamConfig);
    activeStreamExplicitDisabled =
      startOpts &&
      typeof startOpts === 'object' &&
      Object.prototype.hasOwnProperty.call(startOpts, 'streamShaders') &&
      startOpts.streamShaders !== undefined
        ? startOpts.streamShaders === false
        : baseStreamExplicitDisabled;
    activeDumpConfig = resolveDumpConfigOverride(startOpts, baseDumpConfig);
    const desiredInputScript = resolveInputScriptOverride(startOpts, baseInputScript);
    configuredInputScript = desiredInputScript;
    lastNotifiedSideModuleKey = null;
    startOverrideShaderPackUrl =
      startOpts && typeof startOpts === 'object'
        ? pickBootUrl(
            startOpts.shaderPackUrl,
            startOpts.preferredShaderPackUrl,
            startOpts.shaderPack,
            startOpts.wgslPack
          )
        : null;
    await initializeFastBootStoragePlan();
    if (destroyed) {
      return {
        ok: false,
        destroyed: true,
        state: SESSION_STATE.STOPPED,
        mode: SESSION_MODE.NONE,
        error: new Error('PlayGta6AppHandle has been destroyed'),
      };
    }
    let result;
    if (bootOrchestrator && typeof bootOrchestrator.start === 'function') {
      pendingStartOpts = startOpts;
      lastBootSessionResult = null;
      latestDiscoveredBootAssets = { ...initialBootAssets };
      loaderTelemetry.preferredShaderPackUrl = sanitizeTelemetryUrl(
        pickBootUrl(options.preferredShaderPackUrl, options.shaderPackUrl)
      );
      const bootResult = await bootOrchestrator.start();
      recordDiscoveredBootAssets(bootResult);
      if (lastBootSessionResult) {
        if (lastBootSessionResult.serverStatus) {
          recordDiscoveredBootAssets(lastBootSessionResult.serverStatus);
        }
        result = lastBootSessionResult;
      } else {
        const fallbackResult = await session.start(startOpts);
        if (fallbackResult && fallbackResult.serverStatus) {
          recordDiscoveredBootAssets(fallbackResult.serverStatus);
        }
        result = fallbackResult;
      }
    } else {
      latestDiscoveredBootAssets = { ...initialBootAssets };
      loaderTelemetry.preferredShaderPackUrl = sanitizeTelemetryUrl(
        pickBootUrl(options.preferredShaderPackUrl, options.shaderPackUrl)
      );
      const sessionResult = await session.start(startOpts);
      if (sessionResult && sessionResult.serverStatus) {
        recordDiscoveredBootAssets(sessionResult.serverStatus);
      }
      result = sessionResult;
    }
    if (!destroyed && result && result.ok) {
      if (desiredInputScript) {
        startActiveInputScript(desiredInputScript);
      }
      if (
        (externalAudioWorker && (boundAudioRingModule || typeof options.getWasmModule === 'function')) ||
        audioRingController
      ) {
        attachAudioRing(null);
      }
      if (isAudioNeedsUnlock() && readConnectedGamepads().length > 0) {
        startGamepadAudioUnlockPoll();
      }
    }
    return result;
  }

  async function stop() {
    stopActiveInputScript();
    stopGamepadAudioUnlockPoll();
    firstPresentNotified = false;
    firstVisibleFrameNotified = false;
    sawExplicitVisibleFrameGate = false;
    setCanvasBootState('loading');
    if (audioRingController && typeof audioRingController.stop === 'function') {
      try {
        audioRingController.stop();
      } catch {
        // Ignore audio ring retry stop errors.
      }
    }
    if (destroyed) {
      return {
        ok: true,
        alreadyStopped: true,
        destroyed: true,
        state: session.getState(),
        mode: session.getMode(),
      };
    }
    return session.stop();
  }

  async function destroy() {
    if (destroyPromise) {
      return destroyPromise;
    }
    if (destroyed) {
      return;
    }

    destroyed = true;
    destroyPromise = (async () => {
      if (audioRingController) {
        const ctrl = audioRingController;
        audioRingController = null;
        try {
          if (ownsAudioRingController && typeof ctrl.destroy === 'function') {
            ctrl.destroy();
          } else if (typeof ctrl.stop === 'function') {
            ctrl.stop();
          }
        } catch {
          // Ignore audio ring retry controller teardown errors.
        }
      }

      while (reclaimUnregisters.length > 0) {
        const unregister = reclaimUnregisters.pop();
        try {
          unregister();
        } catch {
          // Ignore reclaim hook unregister errors.
        }
      }
      fallbackReclaimHooks.clear();
      if (attachedGlobalHook && globalScope && typeof globalScope === 'object') {
        attachedGlobalHook = false;
        try {
          if (globalScope.__playgta6 === diagnosticHook) {
            delete globalScope.__playgta6;
          }
        } catch {
          // Ignore non-configurable global properties.
        }
      }

      if (attachedBootStateHook && globalScope && typeof globalScope === 'object') {
        attachedBootStateHook = false;
        try {
          const desc = Object.getOwnPropertyDescriptor(globalScope, '__playgta6BootState');
          if (desc && desc.get === bootStateGetter) {
            delete globalScope.__playgta6BootState;
          }
        } catch {
          // Ignore non-configurable global properties.
        }
      }

      if (attachedGlobalScopeTelemetryHook && globalScope && typeof globalScope === 'object') {
        attachedGlobalScopeTelemetryHook = false;
        try {
          if (globalScope.__playgta6Telemetry === telemetryHook) {
            delete globalScope.__playgta6Telemetry;
          }
        } catch {
          // Ignore non-configurable global properties.
        }
      }

      if (attachedGlobalThisTelemetryHook && runtimeGlobalThis) {
        attachedGlobalThisTelemetryHook = false;
        try {
          if (runtimeGlobalThis.__playgta6Telemetry === telemetryHook) {
            delete runtimeGlobalThis.__playgta6Telemetry;
          }
        } catch {
          // Ignore non-configurable globalThis properties.
        }
      }

      while (externalWorkerCleanups.length > 0) {
        const cleanup = externalWorkerCleanups.pop();
        try {
          cleanup();
        } catch {
          // Ignore external worker teardown errors.
        }
      }

      if (canvasResizeObserver) {
        const observer = canvasResizeObserver;
        canvasResizeObserver = null;
        try {
          if (typeof observer.disconnect === 'function') {
            observer.disconnect();
          }
        } catch {
          // Ignore ResizeObserver teardown errors.
        }
      }

      if (resizeListenerAttached && isEventTargetLike(windowObj)) {
        resizeListenerAttached = false;
        windowObj.removeEventListener('resize', onWindowResize);
      }

      detachAudioGestureUnlock();
      stopGamepadAudioUnlockPoll();
      detachVisibilityAndPointerLockLifecycle();

      stopActiveInputScript();

      try {
        await session.stop();
      } catch {
        // Ensure remaining subsystems still tear down cleanly.
      }

      if (activeSmokeRunner) {
        const runner = activeSmokeRunner;
        activeSmokeRunner = null;
        try {
          if (!runner.destroyed) {
            runner.destroy();
          }
        } catch {
          // Ignore teardown errors.
        }
      }

      if (activeWasmHandle) {
        const wasmHandle = activeWasmHandle;
        activeWasmHandle = null;
        activeWasmWorker = null;
        try {
          await wasmHandle.stop();
        } catch {
          // Ignore teardown errors.
        }
      }

      try {
        if (perfMonitor && typeof perfMonitor.detachHud === 'function') {
          perfMonitor.detachHud();
        }
      } catch {
        // Ignore teardown errors.
      }

      try {
        inputBridge.destroy();
      } catch {
        // Ignore teardown errors.
      }

      try {
        if (audioEngine && typeof audioEngine.destroy === 'function') {
          await audioEngine.destroy();
        }
      } catch {
        // Ignore teardown errors.
      }

      while (bootTracerCleanupFns.length > 0) {
        const cleanup = bootTracerCleanupFns.pop();
        try {
          cleanup();
        } catch {
          // Ignore teardown errors.
        }
      }

      try {
        if (ownsGpuBootPanel && gpuBootPanel && typeof gpuBootPanel.destroy === 'function') {
          gpuBootPanel.destroy();
        }
      } catch {
        // Ignore teardown errors.
      }

      try {
        if (ownsBootOrchestrator && bootOrchestrator && typeof bootOrchestrator.destroy === 'function') {
          bootOrchestrator.destroy();
        }
      } catch {
        // Ignore teardown errors.
      }

      try {
        if (ownsSplashOverlay && splashOverlay && typeof splashOverlay.destroy === 'function') {
          splashOverlay.destroy();
        }
      } catch {
        // Ignore teardown errors.
      }

      while (crashReporterCleanupFns.length > 0) {
        const cleanup = crashReporterCleanupFns.pop();
        try {
          cleanup();
        } catch {
          // Ignore CrashReporter hook teardown errors.
        }
      }

      try {
        crashReporter.detachGlobalHandlers();
      } catch {
        // Ignore teardown errors.
      }
    })();

    return destroyPromise;
  }

  return new PlayGta6AppHandle({
    session,
    perfMonitor,
    crashReporter,
    inputProfiles,
    inputBridge,
    audioEngine,
    storageManager,
    bootTracer,
    splashOverlay,
    gpuBootPanel,
    pgimPreloader,
    bootOrchestrator,
    bootUrlOptions: resolvedBootUrlOptions,
    perfProfile: activePerfProfile,
    displayBounds: activeDisplayBounds,
    getPerfProfile: () => activePerfProfile,
    getDisplayBounds: () => activeDisplayBounds,
    setPerfProfile,
    setDisplayBounds,
    getSmokeRunner: () => activeSmokeRunner,
    getWasmWorker: () => activeWasmWorker,
    getDiagnosticsSnapshot,
    getBootTelemetry,
    getFastBootPlan: () => latestFastBootPlan,
    getPhaseTelemetry,
    reclaimSoftMemory,
    registerReclaimHook,
    attachAudioRing,
    unlockAudio,
    getAudioStatus,
    requestPointerLock,
    exitPointerLock,
    togglePointerLock,
    isPointerLocked,
    notifyFirstPresent,
    notifyFirstVisibleFrame,
    recordFastBootMilestone,
    start,
    stop,
    resize,
    destroy,
    isDestroyed: () => destroyed,
  });
}

export default bootstrapPlayGta6App;
