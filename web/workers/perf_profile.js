'use strict';

// Hardware-adaptive Low / Balanced / High / Auto RAM & VRAM performance profiles.
// Zero external dependencies; UMD/CommonJS/worker-safe module exposing
// globalThis.PlayGta6PerfProfile (importScripts / browsers) and module.exports (Node.js).
// Designed to protect 8-16 GB RAM / 4-6 GB VRAM PCs, integrated GPUs, and laptops
// from VRAM paging and browser tab OOM crashes.

(function initPlayGta6PerfProfile(root) {
  const MiB = 1024 * 1024;
  const GiB = 1024 * 1024 * 1024;

  const PERF_PROFILE_NAMES = Object.freeze(['low', 'balanced', 'high']);
  const VALID_PRESET_INPUTS = Object.freeze(['low', 'balanced', 'high', 'auto']);

  const AUTO_THRESHOLDS = Object.freeze({
    lowMaxDeviceMemoryGb: 8,
    lowMaxHardwareConcurrency: 4,
    lowMaxBufferSizeBytes: 256 * MiB,
    lowMaxTextureDimension2D: 4096,
    balancedMaxDeviceMemoryGb: 16,
    balancedMaxHardwareConcurrency: 12,
    balancedMaxBufferSizeBytes: 1 * GiB,
  });

  const PERF_PROFILES = Object.freeze({
    low: Object.freeze({
      name: 'low',
      rtPoolMaxBytes: 128 * MiB,
      rtPoolMaxIdleFrames: 8,
      vramBudgetBytes: 1536 * MiB,
      vramReclaimTriggerRatio: 0.72,
      vramReclaimTargetRatio: 0.58,
      gcDeferFrames: 2,
      ioSlabCacheBytes: 64 * MiB,
      ioMaxConcurrentLanes: 4,
      textureMipCap: 512,
      dprCap: 1.0,
      prewarmMaxInFlight: 1,
      shaderCompileConcurrency: 1,
    }),
    balanced: Object.freeze({
      name: 'balanced',
      rtPoolMaxBytes: 256 * MiB,
      rtPoolMaxIdleFrames: 16,
      vramBudgetBytes: 3072 * MiB,
      vramReclaimTriggerRatio: 0.78,
      vramReclaimTargetRatio: 0.65,
      gcDeferFrames: 2,
      ioSlabCacheBytes: 128 * MiB,
      ioMaxConcurrentLanes: 6,
      textureMipCap: 1024,
      dprCap: 1.0,
      prewarmMaxInFlight: 2,
      shaderCompileConcurrency: 2,
    }),
    high: Object.freeze({
      name: 'high',
      rtPoolMaxBytes: 512 * MiB,
      rtPoolMaxIdleFrames: 30,
      vramBudgetBytes: 6144 * MiB,
      vramReclaimTriggerRatio: 0.85,
      vramReclaimTargetRatio: 0.72,
      gcDeferFrames: 3,
      ioSlabCacheBytes: 192 * MiB,
      ioMaxConcurrentLanes: 8,
      textureMipCap: 0,
      dprCap: 2.0,
      prewarmMaxInFlight: 4,
      shaderCompileConcurrency: 4,
    }),
  });

  // Generic software/CPU renderer hints (no proprietary vendor strings).
  const SOFTWARE_ADAPTER_RE =
    /\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software|basic\s+render|warp|cpu|emulated|fallback)\b/i;

  // Generic low-power/integrated GPU hints (no proprietary vendor strings).
  const LOW_POWER_ADAPTER_RE =
    /\b(?:integrated|igpu|low[\s_-]?power|mobile|shared[\s_-]?memory)\b/i;

  const QUERY_PARAM_KEYS = Object.freeze(['quality', 'perfProfile', 'perf_profile', 'profile']);

  const OVERRIDE_FIELD_ALIASES = Object.freeze([
    { field: 'rtPoolMaxBytes', keys: ['rtPoolMaxBytes', 'maxPoolBytes'], type: 'positiveInt' },
    {
      field: 'rtPoolMaxIdleFrames',
      keys: ['rtPoolMaxIdleFrames', 'idleFrameTtl', 'maxIdleFrames'],
      type: 'nonNegativeInt',
    },
    { field: 'vramBudgetBytes', keys: ['vramBudgetBytes', 'softBudgetBytes'], type: 'positiveInt' },
    {
      field: 'vramReclaimTriggerRatio',
      keys: ['vramReclaimTriggerRatio'],
      type: 'unitRatio',
    },
    {
      field: 'vramReclaimTargetRatio',
      keys: ['vramReclaimTargetRatio'],
      type: 'unitRatio',
    },
    { field: 'gcDeferFrames', keys: ['gcDeferFrames', 'framesInFlight'], type: 'positiveInt' },
    {
      field: 'ioSlabCacheBytes',
      keys: ['ioSlabCacheBytes', 'slabCacheBytes', 'slabMaxBytes'],
      type: 'positiveInt',
    },
    {
      field: 'ioMaxConcurrentLanes',
      keys: ['ioMaxConcurrentLanes', 'maxConcurrentLanes', 'maxConcurrentSlabFetches'],
      type: 'positiveInt',
    },
    { field: 'textureMipCap', keys: ['textureMipCap'], type: 'nonNegativeInt' },
    { field: 'dprCap', keys: ['dprCap'], type: 'positiveFloat' },
    { field: 'prewarmMaxInFlight', keys: ['prewarmMaxInFlight'], type: 'positiveInt' },
    {
      field: 'shaderCompileConcurrency',
      keys: ['shaderCompileConcurrency'],
      type: 'positiveInt',
    },
  ]);

  function normalizePresetName(raw) {
    if (typeof raw !== 'string') {
      return null;
    }
    const lower = raw.trim().toLowerCase();
    if (lower === 'low' || lower === 'balanced' || lower === 'high' || lower === 'auto') {
      return lower;
    }
    return null;
  }

  function isConcretePreset(name) {
    return name === 'low' || name === 'balanced' || name === 'high';
  }

  /**
   * Parse a performance preset (`'low' | 'balanced' | 'high' | 'auto'`) from a URL,
   * query string (`'?quality=low'`), or `URLSearchParams`. Returns `null` when absent
   * or unrecognized so invalid inputs fall back safely.
   *
   * @param {string | URLSearchParams | null | undefined} queryInput
   * @returns {'low' | 'balanced' | 'high' | 'auto' | null}
   */
  function parsePerfProfileFromQuery(queryInput) {
    if (queryInput === null || queryInput === undefined) {
      return null;
    }

    let params = null;
    if (typeof URLSearchParams !== 'undefined' && queryInput instanceof URLSearchParams) {
      params = queryInput;
    } else if (typeof queryInput === 'string') {
      const trimmed = queryInput.trim();
      if (!trimmed) {
        return null;
      }
      let search = trimmed;
      const qIdx = search.indexOf('?');
      if (qIdx >= 0) {
        search = search.slice(qIdx + 1);
      } else if (!search.includes('=')) {
        return null;
      }
      const hashIdx = search.indexOf('#');
      if (hashIdx >= 0) {
        search = search.slice(0, hashIdx);
      }
      if (!search) {
        return null;
      }
      try {
        params = new URLSearchParams(search);
      } catch {
        return null;
      }
    } else {
      return null;
    }

    for (let i = 0; i < QUERY_PARAM_KEYS.length; i += 1) {
      const val = params.get(QUERY_PARAM_KEYS[i]);
      if (val !== null) {
        const normalized = normalizePresetName(val);
        if (normalized !== null) {
          return normalized;
        }
      }
    }
    return null;
  }

  function collectAdapterText(adapterInfo) {
    if (!adapterInfo) {
      return '';
    }
    if (typeof adapterInfo === 'string') {
      return adapterInfo;
    }
    if (typeof adapterInfo !== 'object') {
      return '';
    }
    const parts = [];
    const fields = ['description', 'architecture', 'device', 'vendor', 'name', 'type'];
    for (let i = 0; i < fields.length; i += 1) {
      const val = adapterInfo[fields[i]];
      if (typeof val === 'string' && val.trim().length > 0) {
        parts.push(val.trim());
      }
    }
    return parts.join(' ');
  }

  /**
   * Inspect optional WebGPU `adapterInfo` safely for software/fallback or
   * low-power/integrated renderer hints without hardcoding proprietary names.
   *
   * @param {object | string | null | undefined} adapterInfo
   * @returns {{ isSoftwareAdapter: boolean, isLowPowerAdapter: boolean }}
   */
  function classifyAdapterInfo(adapterInfo) {
    if (!adapterInfo) {
      return { isSoftwareAdapter: false, isLowPowerAdapter: false };
    }

    const text = collectAdapterText(adapterInfo);
    const obj = typeof adapterInfo === 'object' ? adapterInfo : null;

    const isSoftwareAdapter = Boolean(
      (obj &&
        (obj.isFallbackAdapter === true ||
          obj.isSoftware === true ||
          obj.software === true ||
          (typeof obj.type === 'string' &&
            (obj.type.toLowerCase() === 'cpu' || obj.type.toLowerCase() === 'software')))) ||
        (text.length > 0 && SOFTWARE_ADAPTER_RE.test(text))
    );

    const isLowPowerAdapter = Boolean(
      !isSoftwareAdapter &&
        ((obj &&
          (obj.isIntegrated === true ||
            obj.integrated === true ||
            obj.lowPower === true ||
            obj.powerPreference === 'low-power' ||
            (typeof obj.type === 'string' &&
              (obj.type.toLowerCase() === 'integrated' ||
                obj.type.toLowerCase() === 'integrated-gpu' ||
                obj.type.toLowerCase() === 'low-power')))) ||
          (text.length > 0 && LOW_POWER_ADAPTER_RE.test(text)))
    );

    return { isSoftwareAdapter, isLowPowerAdapter };
  }

  function sanitizePositiveNumber(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return null;
    }
    return value;
  }

  function sanitizeOverrideValue(raw, type) {
    if (raw === undefined || raw === null) {
      return null;
    }
    const num = Number(raw);
    if (!Number.isFinite(num)) {
      return null;
    }
    if (type === 'positiveInt') {
      const floored = Math.floor(num);
      return floored >= 1 ? floored : null;
    }
    if (type === 'nonNegativeInt') {
      const floored = Math.floor(num);
      return floored >= 0 ? floored : null;
    }
    if (type === 'unitRatio') {
      if (num <= 0 || num > 1) {
        return null;
      }
      return num;
    }
    if (type === 'positiveFloat') {
      return num > 0 ? num : null;
    }
    return null;
  }

  function resolveHardwareSignals(opts) {
    const isNodeRuntime =
      typeof process !== 'undefined' &&
      Boolean(process.versions && process.versions.node) &&
      typeof window === 'undefined';
    const hasExplicitNavigator = opts.navigator !== undefined;
    const nav = hasExplicitNavigator
      ? opts.navigator
      : typeof navigator !== 'undefined' && navigator !== null
        ? navigator
        : null;

    const hasExplicitDeviceMemory = opts.deviceMemoryGb !== undefined;
    const hasExplicitConcurrency = opts.hardwareConcurrency !== undefined;
    const hasExplicitLimits = opts.adapterLimits !== undefined && opts.adapterLimits !== null;
    const hasExplicitAdapterInfo = opts.adapterInfo !== undefined && opts.adapterInfo !== null;
    const hasAnyExplicitHardware =
      hasExplicitDeviceMemory ||
      hasExplicitConcurrency ||
      hasExplicitLimits ||
      hasExplicitAdapterInfo;

    const allowNavigatorRead =
      nav !== null &&
      opts.useNavigator !== false &&
      (hasExplicitNavigator || !isNodeRuntime || !hasAnyExplicitHardware);

    const rawDeviceMemory = hasExplicitDeviceMemory
      ? opts.deviceMemoryGb
      : allowNavigatorRead
        ? nav.deviceMemory
        : null;

    const rawConcurrency = hasExplicitConcurrency
      ? opts.hardwareConcurrency
      : allowNavigatorRead
        ? nav.hardwareConcurrency
        : null;

    const deviceMemoryGb = sanitizePositiveNumber(
      rawDeviceMemory !== null && rawDeviceMemory !== undefined ? Number(rawDeviceMemory) : null
    );
    const hardwareConcurrency = sanitizePositiveNumber(
      rawConcurrency !== null && rawConcurrency !== undefined ? Number(rawConcurrency) : null
    );

    const limits =
      opts.adapterLimits && typeof opts.adapterLimits === 'object' ? opts.adapterLimits : null;
    const maxBufferSize = limits
      ? sanitizePositiveNumber(
          limits.maxBufferSize !== undefined ? Number(limits.maxBufferSize) : null
        )
      : null;
    const maxStorageBufferBindingSize = limits
      ? sanitizePositiveNumber(
          limits.maxStorageBufferBindingSize !== undefined
            ? Number(limits.maxStorageBufferBindingSize)
            : null
        )
      : null;
    const maxTextureDimension2D = limits
      ? sanitizePositiveNumber(
          limits.maxTextureDimension2D !== undefined ? Number(limits.maxTextureDimension2D) : null
        )
      : null;

    const adapterClass = classifyAdapterInfo(opts.adapterInfo);

    return Object.freeze({
      deviceMemoryGb,
      hardwareConcurrency,
      maxBufferSize,
      maxStorageBufferBindingSize,
      maxTextureDimension2D,
      isSoftwareAdapter: adapterClass.isSoftwareAdapter,
      isLowPowerAdapter: adapterClass.isLowPowerAdapter,
    });
  }

  function detectAutoPreset(hw) {
    const lowReasons = [];
    if (hw.deviceMemoryGb !== null && hw.deviceMemoryGb <= AUTO_THRESHOLDS.lowMaxDeviceMemoryGb) {
      lowReasons.push(`deviceMemoryGb<=${AUTO_THRESHOLDS.lowMaxDeviceMemoryGb}`);
    }
    if (
      hw.hardwareConcurrency !== null &&
      hw.hardwareConcurrency <= AUTO_THRESHOLDS.lowMaxHardwareConcurrency
    ) {
      lowReasons.push(`hardwareConcurrency<=${AUTO_THRESHOLDS.lowMaxHardwareConcurrency}`);
    }
    if (hw.isSoftwareAdapter) {
      lowReasons.push('softwareAdapter');
    }
    if (hw.maxBufferSize !== null && hw.maxBufferSize <= AUTO_THRESHOLDS.lowMaxBufferSizeBytes) {
      lowReasons.push('maxBufferSize<=256MiB');
    }
    if (
      hw.maxTextureDimension2D !== null &&
      hw.maxTextureDimension2D <= AUTO_THRESHOLDS.lowMaxTextureDimension2D
    ) {
      lowReasons.push(`maxTextureDimension2D<=${AUTO_THRESHOLDS.lowMaxTextureDimension2D}`);
    }
    if (lowReasons.length > 0) {
      return { preset: 'low', reasons: lowReasons };
    }

    const balancedReasons = [];
    if (
      hw.deviceMemoryGb !== null &&
      hw.deviceMemoryGb <= AUTO_THRESHOLDS.balancedMaxDeviceMemoryGb
    ) {
      balancedReasons.push(`deviceMemoryGb<=${AUTO_THRESHOLDS.balancedMaxDeviceMemoryGb}`);
    }
    if (
      hw.hardwareConcurrency !== null &&
      hw.hardwareConcurrency <= AUTO_THRESHOLDS.balancedMaxHardwareConcurrency
    ) {
      balancedReasons.push(`hardwareConcurrency<=${AUTO_THRESHOLDS.balancedMaxHardwareConcurrency}`);
    }
    if (
      hw.maxBufferSize !== null &&
      hw.maxBufferSize <= AUTO_THRESHOLDS.balancedMaxBufferSizeBytes
    ) {
      balancedReasons.push('maxBufferSize<=1GiB');
    }
    if (hw.isLowPowerAdapter) {
      balancedReasons.push('lowPowerAdapter');
    }
    if (balancedReasons.length > 0) {
      return { preset: 'balanced', reasons: balancedReasons };
    }

    const hasAnySignal =
      hw.deviceMemoryGb !== null ||
      hw.hardwareConcurrency !== null ||
      hw.maxBufferSize !== null ||
      hw.maxStorageBufferBindingSize !== null ||
      hw.maxTextureDimension2D !== null;

    if (!hasAnySignal) {
      return { preset: 'balanced', reasons: ['noHardwareSignals:safeDefault'] };
    }

    return { preset: 'high', reasons: ['highSpecHardware'] };
  }

  /**
   * Resolve a hardware-adaptive performance profile (`low`, `balanced`, or `high`)
   * from an explicit preset, URL query string, hardware/WebGPU adapter signals,
   * and optional per-field numeric overrides.
   *
   * @param {string | object | null | undefined} [options]
   * @returns {Readonly<object>}
   */
  function resolvePerfProfile(options = {}) {
    let opts = {};
    if (typeof options === 'string') {
      const trimmed = options.trim();
      if (trimmed.includes('=') || trimmed.startsWith('?')) {
        opts = { queryString: trimmed };
      } else {
        opts = { preset: trimmed };
      }
    } else if (options && typeof options === 'object') {
      opts = options;
    }

    const rawPreset =
      opts.preset !== undefined
        ? opts.preset
        : opts.profile !== undefined
          ? opts.profile
          : opts.name !== undefined
            ? opts.name
            : undefined;
    const normalizedExplicitPreset = normalizePresetName(rawPreset);

    const queryInput =
      opts.queryString !== undefined
        ? opts.queryString
        : opts.query !== undefined
          ? opts.query
          : opts.search;
    const queryPreset = parsePerfProfileFromQuery(queryInput);

    const hardware = resolveHardwareSignals(opts);

    let resolvedPreset;
    let source;
    let reasons;
    let requestedPreset = normalizedExplicitPreset || queryPreset || 'auto';

    if (isConcretePreset(normalizedExplicitPreset)) {
      resolvedPreset = normalizedExplicitPreset;
      source = 'explicit';
      reasons = [`explicit:${resolvedPreset}`];
      requestedPreset = resolvedPreset;
    } else if (isConcretePreset(queryPreset)) {
      resolvedPreset = queryPreset;
      source = 'query';
      reasons = [`query:${resolvedPreset}`];
      requestedPreset = queryPreset;
    } else {
      const detected = detectAutoPreset(hardware);
      resolvedPreset = detected.preset;
      source = 'auto';
      reasons = detected.reasons;
      requestedPreset = 'auto';
    }

    const base = PERF_PROFILES[resolvedPreset];
    const resolved = {
      name: resolvedPreset,
      preset: resolvedPreset,
      requestedPreset,
      source,
      rtPoolMaxBytes: base.rtPoolMaxBytes,
      rtPoolMaxIdleFrames: base.rtPoolMaxIdleFrames,
      vramBudgetBytes: base.vramBudgetBytes,
      vramReclaimTriggerRatio: base.vramReclaimTriggerRatio,
      vramReclaimTargetRatio: base.vramReclaimTargetRatio,
      gcDeferFrames: base.gcDeferFrames,
      ioSlabCacheBytes: base.ioSlabCacheBytes,
      ioMaxConcurrentLanes: base.ioMaxConcurrentLanes,
      textureMipCap: base.textureMipCap,
      dprCap: base.dprCap,
      prewarmMaxInFlight: base.prewarmMaxInFlight,
      shaderCompileConcurrency: base.shaderCompileConcurrency,
    };

    const nestedOverrides =
      opts.overrides && typeof opts.overrides === 'object' ? opts.overrides : null;
    const overridesApplied = [];

    for (let i = 0; i < OVERRIDE_FIELD_ALIASES.length; i += 1) {
      const spec = OVERRIDE_FIELD_ALIASES[i];
      let foundRaw = undefined;
      for (let j = 0; j < spec.keys.length; j += 1) {
        const key = spec.keys[j];
        if (nestedOverrides && nestedOverrides[key] !== undefined) {
          foundRaw = nestedOverrides[key];
          break;
        }
        if (opts[key] !== undefined) {
          foundRaw = opts[key];
          break;
        }
      }
      if (foundRaw !== undefined) {
        const sanitized = sanitizeOverrideValue(foundRaw, spec.type);
        if (sanitized !== null && sanitized !== resolved[spec.field]) {
          resolved[spec.field] = sanitized;
          overridesApplied.push(spec.field);
        }
      }
    }

    // Keep reclaim target ratio <= trigger ratio so hysteresis stays well-ordered.
    if (resolved.vramReclaimTargetRatio > resolved.vramReclaimTriggerRatio) {
      resolved.vramReclaimTargetRatio = resolved.vramReclaimTriggerRatio;
    }

    resolved.reasons = Object.freeze(reasons.slice());
    resolved.hardware = hardware;
    resolved.overridesApplied = Object.freeze(overridesApplied);

    return Object.freeze(resolved);
  }

  /**
   * Resolve a preset name (`'low' | 'balanced' | 'high' | 'auto'`), query string,
   * options object, or already-resolved profile object for worker `applyPerfProfile()`.
   * Throws `TypeError` for null/undefined or non-string/non-object inputs.
   *
   * @param {string | object} profileOrName
   * @returns {Readonly<object>}
   */
  function coercePerfProfile(profileOrName) {
    if (typeof profileOrName === 'string') {
      const normalized = normalizePresetName(profileOrName);
      if (normalized !== null) {
        return resolvePerfProfile({ preset: normalized });
      }
      const fromQuery = parsePerfProfileFromQuery(profileOrName);
      if (fromQuery !== null) {
        return resolvePerfProfile({ preset: fromQuery });
      }
      throw new RangeError(
        `Unknown performance profile "${profileOrName}" (expected one of ${VALID_PRESET_INPUTS.join(', ')})`
      );
    }
    if (profileOrName && typeof profileOrName === 'object') {
      return resolvePerfProfile(profileOrName);
    }
    throw new TypeError(
      'Performance profile must be a preset name ("low" | "balanced" | "high" | "auto") or profile object'
    );
  }

  const exportedApi = Object.freeze({
    MiB,
    GiB,
    PERF_PROFILE_NAMES,
    VALID_PRESET_INPUTS,
    AUTO_THRESHOLDS,
    PERF_PROFILES,
    normalizePresetName,
    parsePerfProfileFromQuery,
    classifyAdapterInfo,
    resolvePerfProfile,
    coercePerfProfile,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6PerfProfile = exportedApi;
  } else if (root) {
    root.PlayGta6PerfProfile = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
