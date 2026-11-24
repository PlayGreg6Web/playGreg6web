'use strict';

// Browser boot pipeline orchestrator for playgta6web.
// Discovers local boot artifacts via `/api/status`, drives SplashOverlay + PgimPreloader,
// feeds BootTracer milestones, launches Wasm64 runtime or falls back to the smoke demo,
// and dismisses splash on FIRST_PRESENT.
// Zero external dependencies; UMD export for Browser (globalThis.PlayGta6BootOrchestrator)
// and Node.js (module.exports).

(function (root) {
  const DEFAULT_STATUS_URL = '/api/status';

  /** Ordered high-level boot pipeline phases. */
  const BOOT_PHASE = Object.freeze({
    IDLE: 'IDLE',
    DISCOVERING: 'DISCOVERING',
    PRELOADING: 'PRELOADING',
    LAUNCHING_RUNTIME: 'LAUNCHING_RUNTIME',
    FALLBACK_SMOKE_DEMO: 'FALLBACK_SMOKE_DEMO',
    PRESENTING: 'PRESENTING',
    ERROR: 'ERROR',
    CANCELLED: 'CANCELLED',
    DESTROYED: 'DESTROYED',
  });

  /** Explicit boot mode distinguishing real Wasm64 guest boot vs smoke-demo fallback. */
  const BOOT_MODE = Object.freeze({
    PENDING: 'pending',
    WASM_GUEST: 'wasm-guest',
    SMOKE_DEMO: 'smoke-demo',
  });

  function defaultFetchImpl() {
    if (typeof globalThis !== 'undefined' && typeof globalThis.fetch === 'function') {
      return globalThis.fetch.bind(globalThis);
    }
    return null;
  }

  function abortReason(signal) {
    if (signal && signal.reason !== undefined) return signal.reason;
    const Err = typeof DOMException === 'function' ? DOMException : Error;
    try {
      return new Err('The operation was aborted.', 'AbortError');
    } catch {
      const err = new Error('The operation was aborted.');
      err.name = 'AbortError';
      return err;
    }
  }

  function isAbortError(err) {
    if (!err || typeof err !== 'object') return false;
    if (err.name === 'AbortError') return true;
    if (typeof DOMException === 'function' && err instanceof DOMException && err.name === 'AbortError') {
      return true;
    }
    return false;
  }

  function throwIfAborted(signal) {
    if (signal && signal.aborted) {
      throw abortReason(signal);
    }
  }

  function toError(err, fallbackMessage = 'Unknown boot orchestrator error') {
    if (err instanceof Error) return err;
    if (typeof err === 'string' && err.length > 0) return new Error(err);
    if (err && typeof err === 'object' && typeof err.message === 'string') return new Error(err.message);
    return new Error(fallbackMessage);
  }

  function nonEmptyString(value) {
    return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;
  }

  function optionalFn(value, label) {
    if (value == null) return null;
    if (typeof value !== 'function') {
      throw new TypeError(`${label} must be a function or null`);
    }
    return value;
  }

  function optionalObject(value, label) {
    if (value == null) return null;
    if (typeof value !== 'object') {
      throw new TypeError(`${label} must be an object or null`);
    }
    return value;
  }

  function callQuiet(target, method, ...args) {
    if (!target || typeof target[method] !== 'function') return undefined;
    try {
      return target[method](...args);
    } catch {
      return undefined;
    }
  }

  function invokeQuiet(fn, ...args) {
    if (typeof fn !== 'function') return undefined;
    try {
      return fn(...args);
    } catch {
      return undefined;
    }
  }

  async function invokeQuietAsync(fn, ...args) {
    if (typeof fn !== 'function') return undefined;
    try {
      return await fn(...args);
    } catch {
      return undefined;
    }
  }

  function pickUrl(...candidates) {
    for (const candidate of candidates) {
      const url = nonEmptyString(candidate);
      if (url) return url;
    }
    return null;
  }

  function presentFlag(value) {
    if (value === true) return true;
    if (value && typeof value === 'object') {
      if (value.present === true || value.available === true || value.ok === true || value.nonEmpty === true) {
        return true;
      }
      if (nonEmptyString(value.url) || nonEmptyString(value.path) || nonEmptyString(value.href)) {
        return true;
      }
    }
    return typeof value === 'string' && value.trim().length > 0;
  }

  function urlFromArtifact(value) {
    if (typeof value === 'string') return nonEmptyString(value);
    if (value && typeof value === 'object') {
      return pickUrl(value.url, value.path, value.href);
    }
    return null;
  }

  function normalizeFirstPresentSource(value) {
    if (typeof value !== 'string') return null;
    const normalized = value.trim().toLowerCase();
    if (normalized === 'wasm-guest' || normalized === 'wasm_guest') {
      return BOOT_MODE.WASM_GUEST;
    }
    if (normalized === 'smoke-demo' || normalized === 'smoke_demo') {
      return BOOT_MODE.SMOKE_DEMO;
    }
    return null;
  }

  function resolveFirstPresentSource(meta, bootMode) {
    const explicit =
      meta && typeof meta === 'object'
        ? normalizeFirstPresentSource(meta.firstPresentSource) || normalizeFirstPresentSource(meta.source)
        : null;
    if (explicit) return explicit;
    if (bootMode === BOOT_MODE.WASM_GUEST) return BOOT_MODE.WASM_GUEST;
    if (bootMode === BOOT_MODE.SMOKE_DEMO) return BOOT_MODE.SMOKE_DEMO;
    return null;
  }

  /**
   * Returns true ONLY when the supplied orchestrator, state snapshot, assets object, or
   * mode string represents a real Wasm64 guest boot (`bootMode === 'wasm-guest'`).
   *
   * @param {unknown} target
   * @returns {boolean}
   */
  function isRealGuestBoot(target) {
    if (typeof target === 'string') {
      return target === BOOT_MODE.WASM_GUEST;
    }
    if (target && typeof target === 'object') {
      if (typeof target.isRealGuestBoot === 'function') {
        return target.isRealGuestBoot() === true;
      }
      if (typeof target.getState === 'function') {
        const state = target.getState();
        return Boolean(state && state.bootMode === BOOT_MODE.WASM_GUEST);
      }
      return target.bootMode === BOOT_MODE.WASM_GUEST;
    }
    return false;
  }

  /**
   * Normalize `/api/status` JSON into the boot-discovery fields used by this orchestrator.
   * Accepts both the current server payload (`hasWasmBinary`, `shaderPackCount`) and the
   * richer artifact URLs (`splashImage`, `pgimUrl`, `programManifestUrl`, `gameBase`,
   * `shaderPackUrl`, `reflectionManifestUrl`, `runtimeWasm`) that tests and later server
   * revisions expose.
   */
  function discoverBootAssets(status) {
    const src = status && typeof status === 'object' ? status : {};
    const splashImage = pickUrl(
      src.splashImage,
      src.splashUrl,
      src.splashScreenImage,
      urlFromArtifact(src.splash)
    );
    const pgimUrl = pickUrl(src.pgimUrl, src.pgim, urlFromArtifact(src.pgimImage));
    const programManifestUrl = pickUrl(
      src.programManifestUrl,
      urlFromArtifact(src.programManifest)
    );
    const gameBase = pickUrl(src.gameBase, src.gameBaseUrl);
    const shaderPackUrl = pickUrl(src.shaderPackUrl, src.shaderPack, urlFromArtifact(src.wgslPack));
    const reflectionManifestUrl = pickUrl(
      src.reflectionManifestUrl,
      src.reflectionUrl,
      urlFromArtifact(src.reflectionManifest)
    );

    const runtimeArtifact = src.runtimeWasm !== undefined ? src.runtimeWasm : src.hasWasmBinary;
    const runtimeWasmUrl = urlFromArtifact(src.runtimeWasm) || (src.hasWasmBinary ? '/dist/runtime.wasm' : null);
    const runtimeWasmAvailable = presentFlag(runtimeArtifact) || src.hasWasmBinary === true;
    const canLaunchGuest = runtimeWasmAvailable && Boolean(pgimUrl || programManifestUrl);
    const bootMode = canLaunchGuest ? BOOT_MODE.WASM_GUEST : BOOT_MODE.SMOKE_DEMO;

    return {
      splashImage,
      pgimUrl,
      programManifestUrl,
      gameBase,
      shaderPackUrl,
      reflectionManifestUrl,
      hasWasmBinary: runtimeWasmAvailable,
      runtimeWasm: runtimeWasmAvailable ? runtimeWasmUrl || true : null,
      runtimeWasmUrl: runtimeWasmAvailable ? runtimeWasmUrl : null,
      runtimeWasmAvailable,
      bootMode,
    };
  }

  function tracerSnapshot(bootTracer) {
    if (!bootTracer || typeof bootTracer.getSummary !== 'function') return null;
    try {
      return bootTracer.getSummary();
    } catch {
      return null;
    }
  }

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

  /**
   * Coordinates `/api/status` discovery, splash overlay, PGIM preload, runtime launch
   * or smoke-demo fallback, and FIRST_PRESENT splash dismissal.
   */
  class BootOrchestrator {
    /**
     * @param {object} [options]
     * @param {typeof fetch | null} [options.fetchImpl]
     * @param {string} [options.statusUrl='/api/status']
     * @param {object|null} [options.splashOverlay]
     * @param {object|null} [options.preloader]
     * @param {object|null} [options.bootTracer]
     * @param {Function|null} [options.onLaunchRuntime]
     * @param {Function|null} [options.onFallbackSmokeDemo]
     * @param {Function|null} [options.onPhaseChange]
     * @param {Function|null} [options.onFirstPresent]
     */
    constructor(options = {}) {
      if (options !== undefined && (options === null || typeof options !== 'object')) {
        throw new TypeError('BootOrchestrator options must be an object');
      }

      const {
        fetchImpl = defaultFetchImpl(),
        statusUrl = DEFAULT_STATUS_URL,
        splashOverlay = null,
        preloader = null,
        bootTracer = null,
        onLaunchRuntime = null,
        onFallbackSmokeDemo = null,
        onPhaseChange = null,
        onFirstPresent = null,
      } = options;

      if (fetchImpl !== null && fetchImpl !== undefined && typeof fetchImpl !== 'function') {
        throw new TypeError('fetchImpl must be a function or null');
      }
      if (typeof statusUrl !== 'string' || statusUrl.length === 0) {
        throw new TypeError('statusUrl must be a non-empty string');
      }

      this._fetchImpl = fetchImpl || null;
      this._statusUrl = statusUrl;
      this._splashOverlay = optionalObject(splashOverlay, 'splashOverlay');
      this._preloader = optionalObject(preloader, 'preloader');
      this._bootTracer = optionalObject(bootTracer, 'bootTracer');
      this._onLaunchRuntime = optionalFn(onLaunchRuntime, 'onLaunchRuntime');
      this._onFallbackSmokeDemo = optionalFn(onFallbackSmokeDemo, 'onFallbackSmokeDemo');
      this._onPhaseChange = optionalFn(onPhaseChange, 'onPhaseChange');
      this._onFirstPresent = optionalFn(onFirstPresent, 'onFirstPresent');

      this._phase = BOOT_PHASE.IDLE;
      this._bootMode = BOOT_MODE.PENDING;
      this._status = null;
      this._assets = null;
      this._bundle = null;
      this._progress = null;
      this._fallbackReason = null;
      this._error = null;
      this._firstPresent = false;
      this._firstPresentSource = null;
      this._destroyed = false;
      this._aborted = false;
      this._pgimLoadMarked = false;
      this._startPromise = null;
      this._sessionToken = 0;
      this._abortUnlink = null;
      this._controller = new AbortController();
    }

    /**
     * Returns true ONLY when the orchestrator is running a real Wasm64 guest boot
     * (`bootMode === 'wasm-guest'`).
     */
    isRealGuestBoot() {
      return this._bootMode === BOOT_MODE.WASM_GUEST;
    }

    getState() {
      return {
        phase: this._phase,
        bootMode: this._bootMode,
        programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
        gameBase: this._assets ? this._assets.gameBase : null,
        status: this._status ? { ...this._status } : null,
        assets: this._assets ? { ...this._assets } : null,
        bundle: this._bundle,
        progress: this._progress ? { ...this._progress } : null,
        tracer: tracerSnapshot(this._bootTracer),
        fallbackReason: this._fallbackReason,
        error: this._error ? this._error.message : null,
        firstPresent: this._firstPresent,
        firstPresentSource: this._firstPresentSource,
        destroyed: this._destroyed,
        aborted: this._aborted,
      };
    }

    _setPhase(nextPhase, detail = null) {
      if (this._phase === nextPhase) return;
      const previousPhase = this._phase;
      this._phase = nextPhase;
      const event = {
        phase: nextPhase,
        previousPhase,
        bootMode: this._bootMode,
        programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
        gameBase: this._assets ? this._assets.gameBase : null,
        firstPresentSource: this._firstPresentSource,
        status: this._status,
        assets: this._assets,
        bundle: this._bundle,
        progress: this._progress,
        tracer: tracerSnapshot(this._bootTracer),
        fallbackReason: this._fallbackReason,
        error: this._error,
        ...(detail && typeof detail === 'object' ? detail : {}),
      };
      invokeQuiet(this._onPhaseChange, event);
      this._pushOverlay({ phase: nextPhase, ...event });
    }

    _pushOverlay(payload) {
      const overlay = this._splashOverlay;
      if (!overlay) return;
      callQuiet(overlay, 'update', payload);
      if (payload && payload.progress) {
        callQuiet(overlay, 'updateProgress', payload.progress);
      }
      if (payload && payload.tracer) {
        callQuiet(overlay, 'updateTracer', payload.tracer);
        callQuiet(overlay, 'setTracerSummary', payload.tracer);
      }
    }

    _initOverlay(splashImage) {
      const overlay = this._splashOverlay;
      if (!overlay) return;
      const tracer = tracerSnapshot(this._bootTracer);
      const payload = {
        splashImage,
        bootMode: this._bootMode,
        programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
        gameBase: this._assets ? this._assets.gameBase : null,
        status: this._status,
        assets: this._assets,
        tracer,
        phase: this._phase,
      };
      if (typeof overlay.init === 'function') {
        callQuiet(overlay, 'init', payload);
      } else {
        callQuiet(overlay, 'show', payload);
      }
      if (splashImage) {
        callQuiet(overlay, 'setSplashImage', splashImage);
        callQuiet(overlay, 'setImage', splashImage);
      }
      this._pushOverlay(payload);
    }

    _forwardProgress(event) {
      const progress = event && typeof event === 'object' ? event : { loadedBytes: 0, totalBytes: null, pct: 0 };
      this._progress = {
        phase: progress.phase || 'download',
        asset: progress.asset || 'bundle',
        loadedBytes: Number.isFinite(progress.loadedBytes) ? progress.loadedBytes : 0,
        totalBytes: Number.isFinite(progress.totalBytes) ? progress.totalBytes : null,
        pct: Number.isFinite(progress.pct) ? progress.pct : 0,
      };
      callQuiet(this._bootTracer, 'recordProgress', this._progress);
      callQuiet(this._bootTracer, 'updateProgress', this._progress);

      const completedPgim =
        (this._progress.asset === 'pgim' || this._progress.asset === 'bundle') &&
        this._progress.phase === 'complete';
      if (completedPgim && !this._pgimLoadMarked && this._bootTracer && typeof this._bootTracer.markMilestone === 'function') {
        this._pgimLoadMarked = true;
        try {
          this._bootTracer.markMilestone('PGIM_LOAD', {
            loadedBytes: this._progress.loadedBytes,
            totalBytes: this._progress.totalBytes,
          });
        } catch {
          // Milestone names are fixed; ignore tracer validation errors.
        }
      }

      this._pushOverlay({
        phase: this._phase,
        bootMode: this._bootMode,
        progress: this._progress,
        tracer: tracerSnapshot(this._bootTracer),
        status: this._status,
      });
    }

    async _fetchStatus(signal) {
      if (!this._fetchImpl) {
        throw new Error(`Unable to fetch ${this._statusUrl}: fetch API is unavailable`);
      }
      throwIfAborted(signal);
      const response = await this._fetchImpl(this._statusUrl, { cache: 'no-store', signal });
      throwIfAborted(signal);
      if (!response || typeof response.ok !== 'boolean') {
        throw new Error(`Invalid response from ${this._statusUrl}`);
      }
      if (!response.ok) {
        const err = new Error(`Status probe ${this._statusUrl} failed with HTTP ${response.status ?? 'unknown'}`);
        err.status = response.status ?? null;
        throw err;
      }
      const payload = await response.json();
      throwIfAborted(signal);
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        throw new Error(`Status probe ${this._statusUrl} returned non-object JSON`);
      }
      return payload;
    }

    async _preloadBundle(assets, signal) {
      const preloader = this._preloader;
      if (!preloader) {
        throw new Error('Boot artifacts are present but no preloader is configured');
      }
      const onProgress = (event) => this._forwardProgress(event);

      if (assets.pgimUrl) {
        if (typeof preloader.preloadBootBundle !== 'function') {
          throw new TypeError('preloader.preloadBootBundle must be a function');
        }
        return preloader.preloadBootBundle({
          pgimUrl: assets.pgimUrl,
          shaderPackUrl: assets.shaderPackUrl,
          reflectionUrl: assets.reflectionManifestUrl,
          onProgress,
          signal,
        });
      }

      const bundle = {
        pgim: null,
        shaderPackBytes: null,
        reflectionManifest: null,
        bytesLoaded: 0,
      };
      if (assets.shaderPackUrl && typeof preloader.preloadShaderPack === 'function') {
        bundle.shaderPackBytes = await preloader.preloadShaderPack(assets.shaderPackUrl, { onProgress, signal });
      }
      if (assets.reflectionManifestUrl && typeof preloader.preloadReflectionManifest === 'function') {
        bundle.reflectionManifest = await preloader.preloadReflectionManifest(assets.reflectionManifestUrl, {
          onProgress,
          signal,
        });
      }
      return bundle;
    }

    _fallbackReasonFor(assets) {
      if (!assets.runtimeWasmAvailable) {
        return 'No Wasm64 runtime binary available (runtimeWasm/hasWasmBinary); falling back to WebGPU G5 smoke demo.';
      }
      if (!assets.pgimUrl && !assets.programManifestUrl) {
        return 'No PGIM guest image URL (pgimUrl) or programManifestUrl discovered; falling back to WebGPU G5 smoke demo.';
      }
      return 'Runtime launch requirements were not met; falling back to WebGPU G5 smoke demo.';
    }

    /**
     * Run discovery → optional preload → runtime launch or smoke-demo fallback.
     *
     * @param {{ signal?: AbortSignal }} [options]
     */
    async start(options = {}) {
      if (this._destroyed) {
        return {
          ok: false,
          destroyed: true,
          phase: BOOT_PHASE.DESTROYED,
          bootMode: this._bootMode,
          error: new Error('BootOrchestrator has been destroyed'),
        };
      }
      if (this._startPromise) {
        return this._startPromise;
      }
      if (
        this._phase === BOOT_PHASE.LAUNCHING_RUNTIME ||
        this._phase === BOOT_PHASE.FALLBACK_SMOKE_DEMO ||
        this._phase === BOOT_PHASE.PRESENTING
      ) {
        return {
          ok: true,
          alreadyStarted: true,
          phase: this._phase,
          bootMode: this._bootMode,
          programManifestUrl: this._assets ? this._assets.programManifestUrl : null,
          gameBase: this._assets ? this._assets.gameBase : null,
          status: this._status,
          assets: this._assets,
          bundle: this._bundle,
          fallbackReason: this._fallbackReason,
        };
      }

      const outerSignal = options && options.signal ? options.signal : null;
      if (outerSignal && typeof outerSignal.aborted !== 'boolean') {
        throw new TypeError('start({ signal }) requires an AbortSignal');
      }

      const token = ++this._sessionToken;
      const linked = linkSignal(outerSignal);
      this._controller = linked.controller;
      this._abortUnlink = linked.unlink;
      const signal = this._controller.signal;

      const promise = this._startInternal(signal, token).finally(() => {
        if (this._startPromise === promise) {
          this._startPromise = null;
        }
        if (typeof this._abortUnlink === 'function') {
          this._abortUnlink();
          this._abortUnlink = null;
        }
      });
      this._startPromise = promise;
      return promise;
    }

    async _startInternal(signal, token) {
      this._bootMode = BOOT_MODE.PENDING;
      this._error = null;
      this._fallbackReason = null;
      this._bundle = null;
      this._progress = null;
      this._aborted = false;
      this._pgimLoadMarked = false;

      try {
        throwIfAborted(signal);
        this._setPhase(BOOT_PHASE.DISCOVERING);
        const status = await this._fetchStatus(signal);
        if (token !== this._sessionToken) {
          return { ok: false, aborted: true, phase: this._phase, bootMode: this._bootMode };
        }
        this._assets = discoverBootAssets(status);
        this._status = {
          ...status,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          bootMode: this._assets.bootMode,
        };
        this._initOverlay(this._assets.splashImage);
        this._pushOverlay({
          phase: this._phase,
          bootMode: this._bootMode,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          status: this._status,
          assets: this._assets,
          tracer: tracerSnapshot(this._bootTracer),
          splashImage: this._assets.splashImage,
        });

        throwIfAborted(signal);
        const shouldPreload = Boolean(this._assets.pgimUrl || this._assets.shaderPackUrl);
        if (shouldPreload) {
          this._setPhase(BOOT_PHASE.PRELOADING);
          this._bundle = (await this._preloadBundle(this._assets, signal)) || null;
        }

        throwIfAborted(signal);
        if (token !== this._sessionToken) {
          return { ok: false, aborted: true, phase: this._phase, bootMode: this._bootMode };
        }

        const canLaunch =
          this._assets.runtimeWasmAvailable &&
          Boolean(this._assets.pgimUrl || this._assets.programManifestUrl);

        if (canLaunch) {
          this._bootMode = BOOT_MODE.WASM_GUEST;
          this._status = { ...this._status, bootMode: this._bootMode };
          const launchPayload = {
            bootMode: this._bootMode,
            programManifestUrl: this._assets.programManifestUrl,
            gameBase: this._assets.gameBase,
            status: this._status,
            assets: this._assets,
            bundle: this._bundle,
            bootTracer: this._bootTracer,
          };
          this._setPhase(BOOT_PHASE.LAUNCHING_RUNTIME);
          await invokeQuietAsync(this._onLaunchRuntime, launchPayload);
          return {
            ok: true,
            phase: this._phase,
            ...launchPayload,
          };
        }

        this._bootMode = BOOT_MODE.SMOKE_DEMO;
        this._status = { ...this._status, bootMode: this._bootMode };
        this._fallbackReason = this._fallbackReasonFor(this._assets);
        this._setPhase(BOOT_PHASE.FALLBACK_SMOKE_DEMO, { reason: this._fallbackReason });
        await invokeQuietAsync(this._onFallbackSmokeDemo, {
          bootMode: this._bootMode,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          status: this._status,
          bundle: this._bundle,
          reason: this._fallbackReason,
          assets: this._assets,
          bootTracer: this._bootTracer,
        });
        return {
          ok: true,
          fallback: true,
          phase: this._phase,
          bootMode: this._bootMode,
          programManifestUrl: this._assets.programManifestUrl,
          gameBase: this._assets.gameBase,
          status: this._status,
          bundle: this._bundle,
          assets: this._assets,
          reason: this._fallbackReason,
          fallbackReason: this._fallbackReason,
        };
      } catch (err) {
        if (this._destroyed) {
          return { ok: false, destroyed: true, phase: BOOT_PHASE.DESTROYED, bootMode: this._bootMode };
        }
        if (signal.aborted || isAbortError(err) || token !== this._sessionToken) {
          this._aborted = true;
          this._setPhase(BOOT_PHASE.CANCELLED);
          return {
            ok: false,
            aborted: true,
            phase: this._phase,
            bootMode: this._bootMode,
            error: toError(err, 'Boot pipeline aborted'),
          };
        }
        const normalized = toError(err, 'Boot pipeline failed');
        this._error = normalized;
        this._setPhase(BOOT_PHASE.ERROR, { error: normalized });
        return {
          ok: false,
          phase: this._phase,
          bootMode: this._bootMode,
          error: normalized,
        };
      }
    }

    /**
     * Forward a runtime log line into BootTracer, refresh the splash overlay, and
     * auto-dismiss on FIRST_PRESENT (`readyForSplash === true`).
     *
     * @param {unknown} line
     */
    feedRuntimeLog(line) {
      if (this._destroyed) return null;
      let ingested = null;
      if (this._bootTracer && typeof this._bootTracer.ingestLogLine === 'function') {
        try {
          ingested = this._bootTracer.ingestLogLine(line);
        } catch {
          ingested = null;
        }
      }
      const tracer = tracerSnapshot(this._bootTracer);
      this._pushOverlay({
        phase: this._phase,
        bootMode: this._bootMode,
        firstPresentSource: this._firstPresentSource,
        tracer,
        status: this._status,
        logLine: typeof line === 'string' ? line : null,
        ingested,
      });
      if (tracer && tracer.readyForSplash === true) {
        this.notifyFirstPresent({ source: 'bootTracer', tracer, ingested });
      }
      return ingested;
    }

    /**
     * Transition to PRESENTING, dismiss splash, record `firstPresentSource`
     * (`'wasm-guest'` vs `'smoke-demo'`), and fire `onFirstPresent` once.
     *
     * @param {object} [meta]
     */
    notifyFirstPresent(meta = {}) {
      if (this._destroyed || this._firstPresent) {
        return {
          ok: true,
          alreadyPresented: true,
          phase: this._phase,
          bootMode: this._bootMode,
          firstPresentSource: this._firstPresentSource,
        };
      }
      const safeMeta = meta && typeof meta === 'object' ? meta : {};
      const firstPresentSource = resolveFirstPresentSource(safeMeta, this._bootMode);
      const enrichedMeta = {
        ...safeMeta,
        bootMode: this._bootMode,
        firstPresentSource,
        isRealGuestBoot: this.isRealGuestBoot(),
      };

      this._firstPresent = true;
      this._firstPresentSource = firstPresentSource;
      if (this._bootTracer && typeof this._bootTracer.markMilestone === 'function') {
        const tracer = tracerSnapshot(this._bootTracer);
        if (!tracer || tracer.readyForSplash !== true) {
          try {
            this._bootTracer.markMilestone('FIRST_PRESENT', enrichedMeta);
          } catch {
            // Ignore unknown-milestone errors from stub tracers.
          }
        }
      }
      this._setPhase(BOOT_PHASE.PRESENTING, {
        meta: enrichedMeta,
        firstPresentSource: this._firstPresentSource,
      });
      callQuiet(this._splashOverlay, 'dismiss', enrichedMeta);
      callQuiet(this._splashOverlay, 'hide', enrichedMeta);
      invokeQuiet(this._onFirstPresent, {
        ...enrichedMeta,
        status: this._status,
        assets: this._assets,
        bundle: this._bundle,
        tracer: tracerSnapshot(this._bootTracer),
      });
      return {
        ok: true,
        alreadyPresented: false,
        phase: this._phase,
        bootMode: this._bootMode,
        firstPresentSource: this._firstPresentSource,
      };
    }

    /**
     * Alias for `notifyFirstPresent(meta)` to mark FIRST_PRESENT cleanly for guest or smoke frames.
     *
     * @param {object} [meta]
     */
    markFirstPresent(meta = {}) {
      return this.notifyFirstPresent(meta);
    }

    /**
     * Abort in-flight work, destroy the overlay, and freeze the orchestrator.
     * Safe to call more than once.
     */
    destroy() {
      if (this._destroyed) {
        return { ok: true, alreadyDestroyed: true, phase: this._phase };
      }
      this._destroyed = true;
      this._sessionToken += 1;
      this._startPromise = null;
      try {
        if (this._controller && typeof this._controller.abort === 'function' && !this._controller.signal.aborted) {
          this._controller.abort(abortReason(this._controller.signal));
        }
      } catch {
        // Ignore abort errors during teardown.
      }
      if (typeof this._abortUnlink === 'function') {
        try {
          this._abortUnlink();
        } catch {
          // Ignore.
        }
        this._abortUnlink = null;
      }
      callQuiet(this._splashOverlay, 'destroy');
      this._setPhase(BOOT_PHASE.DESTROYED);
      return { ok: true, alreadyDestroyed: false, phase: this._phase };
    }
  }

  const exportedApi = Object.freeze({
    BOOT_PHASE,
    BOOT_MODE,
    DEFAULT_STATUS_URL,
    BootOrchestrator,
    discoverBootAssets,
    isRealGuestBoot,
    default: BootOrchestrator,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6BootOrchestrator = exportedApi;
  } else if (root) {
    root.PlayGta6BootOrchestrator = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
