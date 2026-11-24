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
import { InputBridge } from './input/input_bridge.js';
import { InputProfileManager } from './input/input_profiles.js';
import { AudioEngine } from './audio/audio_engine.js';
import { BrowserStorageManager } from './storage/storage_manager.js';
import { PerfMonitor } from './telemetry/perf_monitor.js';
import { CrashReporter } from './telemetry/crash_report.js';
import { BootTracer } from './telemetry/boot_tracer.js';

export const DEFAULT_LOADER_URL = new URL('./loader.js', import.meta.url).href;
export const DEFAULT_VIEWPORT_WIDTH = 640;
export const DEFAULT_VIEWPORT_HEIGHT = 360;
export const DEFAULT_CANVAS_FORMAT = 'bgra8unorm';

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

export function extractBootAssetUrls(source) {
  if (!source || typeof source !== 'object') {
    return {};
  }
  const out = {};
  const bootBase = pickBootUrl(source.bootBase);
  const gameBase = pickBootUrl(source.gameBase);
  const shaderPackUrl = pickBootUrl(source.shaderPackUrl, source.shaderPack, source.wgslPack);
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
  return out;
}

function resolveViewportMetrics(canvas, windowObj, overrides = {}) {
  const dpr = isPositiveFinite(overrides.devicePixelRatio)
    ? overrides.devicePixelRatio
    : isPositiveFinite(windowObj && windowObj.devicePixelRatio)
      ? windowObj.devicePixelRatio
      : 1;

  const width = isPositiveFinite(overrides.width)
    ? overrides.width
    : isPositiveFinite(canvas && canvas.clientWidth)
      ? canvas.clientWidth
      : isPositiveFinite(canvas && canvas.width)
        ? canvas.width
        : isPositiveFinite(windowObj && windowObj.innerWidth)
          ? windowObj.innerWidth
          : DEFAULT_VIEWPORT_WIDTH;

  const height = isPositiveFinite(overrides.height)
    ? overrides.height
    : isPositiveFinite(canvas && canvas.clientHeight)
      ? canvas.clientHeight
      : isPositiveFinite(canvas && canvas.height)
        ? canvas.height
        : isPositiveFinite(windowObj && windowObj.innerHeight)
          ? windowObj.innerHeight
          : DEFAULT_VIEWPORT_HEIGHT;

  return { width, height, dpr };
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
    show() {
      if (typeof splashOverlay.show === 'function') {
        splashOverlay.show();
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
      if (payload.progress && typeof payload.progress === 'object') {
        base.streamingProgress = payload.progress;
      }
      if (payload.phase === 'DISCOVERING' && !base.statusText) {
        base.statusText = 'Discovering local boot assets...';
      } else if (payload.phase === 'PRELOADING' && !base.statusText) {
        base.statusText = 'Streaming guest image & shaders...';
      } else if (payload.phase === 'LAUNCHING_RUNTIME' && !base.statusText) {
        base.statusText = 'Launching Wasm64 runtime...';
      } else if (payload.phase === 'FALLBACK_SMOKE_DEMO' && !base.statusText) {
        base.statusText = 'Starting WebGPU smoke demo...';
      }
      splashOverlay.updateBootState(base);
    },
    updateProgress(progress) {
      splashOverlay.updateBootState({ streamingProgress: progress });
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
    hide() {
      // Allow smooth dismiss fade to complete when dismiss() already started fading.
      const state = typeof splashOverlay.getState === 'function' ? splashOverlay.getState() : null;
      if (state && state.fading) {
        return;
      }
      if (typeof splashOverlay.hide === 'function') {
        splashOverlay.hide();
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

    this._getSmokeRunner = components.getSmokeRunner;
    this._getWasmWorker = components.getWasmWorker;
    this._notifyFirstPresent = components.notifyFirstPresent || (() => ({ ok: true }));
    this._start = components.start;
    this._stop = components.stop;
    this._resize = components.resize;
    this._destroy = components.destroy;
    this._isDestroyed = components.isDestroyed;
  }

  get destroyed() {
    return this._isDestroyed();
  }

  getSmokeRunner() {
    return this._getSmokeRunner();
  }

  getWasmWorker() {
    return this._getWasmWorker();
  }

  notifyFirstPresent(meta) {
    return this._notifyFirstPresent(meta);
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

  const canvas =
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
  const initialBootAssets = extractBootAssetUrls(options);
  let latestDiscoveredBootAssets = { ...initialBootAssets };

  function recordDiscoveredBootAssets(payload) {
    if (!payload || typeof payload !== 'object') {
      return;
    }
    const fromDirect = extractBootAssetUrls(payload);
    const fromStatus = extractBootAssetUrls(payload.status || payload.serverStatus);
    const fromAssets = extractBootAssetUrls(payload.assets);
    const merged = { ...fromDirect, ...fromStatus, ...fromAssets };
    if (Object.keys(merged).length > 0) {
      latestDiscoveredBootAssets = { ...latestDiscoveredBootAssets, ...merged };
    }
  }

  function resolveBootForwardFields(launchContext) {
    let fromOrchestrator = {};
    if (bootOrchestrator && typeof bootOrchestrator.getState === 'function') {
      try {
        const state = bootOrchestrator.getState();
        if (state && typeof state === 'object') {
          fromOrchestrator = {
            ...extractBootAssetUrls(state.status),
            ...extractBootAssetUrls(state.assets),
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
    };
    if (merged.shaderPackUrl) {
      fields.shaderPackUrl = merged.shaderPackUrl;
    }
    if (merged.reflectionManifestUrl) {
      fields.reflectionManifestUrl = merged.reflectionManifestUrl;
    }
    if (merged.pgimUrl) {
      fields.pgimUrl = merged.pgimUrl;
    }
    if (merged.programManifestUrl) {
      fields.programManifestUrl = merged.programManifestUrl;
    }
    return fields;
  }

  // 1. CrashReporter (sanitized diagnostics + global window error handlers).
  const crashReporterOptions =
    options.crashReporterOptions && typeof options.crashReporterOptions === 'object'
      ? options.crashReporterOptions
      : {};
  const crashReporter = new CrashReporter({
    ...crashReporterOptions,
    ...(options.appVersion !== undefined ? { appVersion: options.appVersion } : {}),
    now: crashReporterOptions.now || epochNow,
    navigatorObj:
      crashReporterOptions.navigatorObj !== undefined
        ? crashReporterOptions.navigatorObj
        : navigatorObj || undefined,
    globalObj:
      crashReporterOptions.globalObj !== undefined
        ? crashReporterOptions.globalObj
        : windowObj || (typeof globalThis !== 'undefined' ? globalThis : undefined),
  });

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

  const perfMonitor = new PerfMonitor({
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

  const storageManager = new BrowserStorageManager({
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

  const audioEngine = new AudioEngine({
    ...audioOptions,
    ...(resolvedAudioContextCtor !== undefined
      ? { AudioContextCtor: resolvedAudioContextCtor }
      : {}),
    ...(resolvedAudioWorkletNodeCtor !== undefined
      ? { AudioWorkletNodeCtor: resolvedAudioWorkletNodeCtor }
      : {}),
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

  let splashOverlay = null;
  let ownsSplashOverlay = false;
  if (options.splashOverlay !== undefined) {
    splashOverlay = options.splashOverlay;
  } else if (
    options.enableSplashOverlay !== false &&
    splashContainer &&
    typeof SplashOverlayCtor === 'function'
  ) {
    const splashOverlayOptions =
      options.splashOverlayOptions && typeof options.splashOverlayOptions === 'object'
        ? options.splashOverlayOptions
        : {};
    splashOverlay = new SplashOverlayCtor({
      ...splashOverlayOptions,
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
  const bootTracerCleanupFns = [];

  function safeUpdateGpuBootPanel(patch = {}) {
    if (destroyed || !gpuBootPanel || typeof gpuBootPanel.update !== 'function') {
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

    try {
      gpuBootPanel.update(payload);
    } catch {
      // Diagnostics panel errors must never interrupt boot or runtime.
    }
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
    }
    if (data.fallbackStats !== undefined) {
      patch.fallbackStats = data.fallbackStats;
    }
    if (data.missingShaders !== undefined) {
      patch.missingShaders = data.missingShaders;
    }
    safeUpdateGpuBootPanel(patch);
    return true;
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
  const pgimPreloader =
    options.pgimPreloader !== undefined
      ? options.pgimPreloader
      : typeof PgimPreloaderCtor === 'function'
        ? new PgimPreloaderCtor({
            ...(options.pgimPreloaderOptions && typeof options.pgimPreloaderOptions === 'object'
              ? options.pgimPreloaderOptions
              : {}),
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

  function notifyFirstPresent(meta = {}) {
    if (destroyed) {
      return { ok: false, destroyed: true };
    }
    if (firstPresentNotified) {
      return { ok: true, alreadyPresented: true };
    }
    firstPresentNotified = true;
    if (bootOrchestrator && typeof bootOrchestrator.notifyFirstPresent === 'function') {
      try {
        return bootOrchestrator.notifyFirstPresent(meta);
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
    return { ok: true, alreadyPresented: false };
  }

  function feedBootLog(line) {
    if (bootOrchestrator && typeof bootOrchestrator.feedRuntimeLog === 'function') {
      try {
        bootOrchestrator.feedRuntimeLog(line);
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
      } catch {
        // Tracer errors must never interrupt runtime logs.
      }
    }
  }

  // Shared mutable lifecycle state managed by the app handle.
  const smokeDemoOptions =
    options.smokeDemoOptions && typeof options.smokeDemoOptions === 'object'
      ? options.smokeDemoOptions
      : {};
  const initialViewport = resolveViewportMetrics(canvas, windowObj, smokeDemoOptions);
  let viewportWidth = initialViewport.width;
  let viewportHeight = initialViewport.height;
  let viewportDpr = initialViewport.dpr;

  let activeSmokeRunner = null;
  let activeWasmHandle = null;
  let activeWasmWorker = null;
  let destroyed = false;
  let destroyPromise = null;

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

    const pixelWidth = Math.max(1, Math.round(viewportWidth * viewportDpr));
    const pixelHeight = Math.max(1, Math.round(viewportHeight * viewportDpr));
    if (launchContext && launchContext.serverStatus) {
      recordDiscoveredBootAssets(launchContext.serverStatus);
    }
    const bootForwardFields = resolveBootForwardFields(launchContext);

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
      handleGpuTelemetryMessage(data);
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
          if (data.type === 'gpu-telemetry') {
            handleGpuTelemetryMessage(data);
            return;
          }
          if (data.metrics && hud && typeof hud.updateMetrics === 'function') {
            try {
              hud.updateMetrics(data.metrics);
            } catch {
              // Ignore HUD errors.
            }
          }
          if (data.error) {
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
            crashReporter.recordLog('info', 'loader', String(data.label));
            feedBootLog(String(data.label));
            if (hud && typeof hud.appendLog === 'function') {
              try {
                hud.appendLog(String(data.label), String(data.label).includes('ready') ? 'ok' : 'info');
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
        const offscreen = canvas.transferControlToOffscreen();
        transferCanvas = offscreen;
        transferList = [offscreen];
      }

      const initPayload = {
        canvas: transferCanvas,
        width: pixelWidth,
        height: pixelHeight,
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

    const wasmHandle = {
      mode: SESSION_MODE.WASM,
      worker,
      resize(nextWidth, nextHeight, nextDpr = viewportDpr) {
        if (stopped) {
          return false;
        }
        const pw = Math.max(1, Math.round(nextWidth * nextDpr));
        const ph = Math.max(1, Math.round(nextHeight * nextDpr));
        if (typeof worker.postMessage === 'function') {
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
      crashReporter.recordError('webgpu_validation', errObj, { source: 'smokeDemo' });
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

  const session = new RuntimeSessionController({
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
  });

  // 7b. Optional BootOrchestrator (W44) wired when splash overlay or explicit flag is enabled.
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
    bootOrchestrator = new BootOrchestratorCtor({
      fetchImpl,
      statusUrl,
      splashOverlay: splashAdapter,
      preloader: pgimPreloader,
      bootTracer,
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
    });
    ownsBootOrchestrator = true;
  }

  // 8. Viewport resize handler (manual API + optional window resize listener).
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
        ? windowObj.devicePixelRatio
        : viewportDpr;

    viewportWidth = nextWidth;
    viewportHeight = nextHeight;
    viewportDpr = resolvedDpr;

    if (activeSmokeRunner && !activeSmokeRunner.destroyed) {
      return activeSmokeRunner.resize(nextWidth, nextHeight, resolvedDpr);
    }
    if (activeWasmHandle) {
      return activeWasmHandle.resize(nextWidth, nextHeight, resolvedDpr);
    }
    return false;
  }

  let resizeListenerAttached = false;
  const onWindowResize = () => {
    if (destroyed || !windowObj) return;
    const w =
      isPositiveFinite(canvas && canvas.clientWidth)
        ? canvas.clientWidth
        : windowObj.innerWidth;
    const h =
      isPositiveFinite(canvas && canvas.clientHeight)
        ? canvas.clientHeight
        : windowObj.innerHeight;
    if (isPositiveFinite(w) && isPositiveFinite(h)) {
      resize(w, h, windowObj.devicePixelRatio);
    }
  };

  if (options.autoResize !== false && isEventTargetLike(windowObj)) {
    windowObj.addEventListener('resize', onWindowResize);
    resizeListenerAttached = true;
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
    if (bootOrchestrator && typeof bootOrchestrator.start === 'function') {
      pendingStartOpts = startOpts;
      lastBootSessionResult = null;
      latestDiscoveredBootAssets = { ...initialBootAssets };
      const bootResult = await bootOrchestrator.start();
      recordDiscoveredBootAssets(bootResult);
      if (lastBootSessionResult) {
        return lastBootSessionResult;
      }
      if (bootResult && bootResult.ok === false) {
        return session.start(startOpts);
      }
      return session.start(startOpts);
    }
    latestDiscoveredBootAssets = { ...initialBootAssets };
    return session.start(startOpts);
  }

  async function stop() {
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
      if (resizeListenerAttached && isEventTargetLike(windowObj)) {
        resizeListenerAttached = false;
        windowObj.removeEventListener('resize', onWindowResize);
      }

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
        perfMonitor.detachHud();
      } catch {
        // Ignore teardown errors.
      }

      try {
        inputBridge.destroy();
      } catch {
        // Ignore teardown errors.
      }

      try {
        await audioEngine.destroy();
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
    getSmokeRunner: () => activeSmokeRunner,
    getWasmWorker: () => activeWasmWorker,
    notifyFirstPresent,
    start,
    stop,
    resize,
    destroy,
    isDestroyed: () => destroyed,
  });
}

export default bootstrapPlayGta6App;
