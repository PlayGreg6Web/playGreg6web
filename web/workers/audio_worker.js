'use strict';

(function (root) {
  const DEFAULT_SAMPLE_RATE = 48000;
  const MIN_SAMPLE_RATE = 8000;
  const MAX_SAMPLE_RATE = 384000;
  const DEFAULT_CHANNELS = 2;
  const MAX_CHANNELS = 8;
  const DEFAULT_CAPACITY_FRAMES = 16384; // ~341 ms @ 48 kHz for jitter/hitch absorption
  const MAX_CAPACITY_FRAMES = 1048576;

  // Bounded headless/suspended drain defaults (48 kHz real-time paced)
  // and active-mode guest PCM pump cadence (well above Chrome's 4 ms timer clamp).
  const DEFAULT_DRAIN_INTERVAL_MS = 20;
  const DEFAULT_ACTIVE_PUMP_INTERVAL_MS = 8;
  const MIN_DRAIN_INTERVAL_MS = 4;
  const MAX_DRAIN_INTERVAL_MS = 250;
  const MAX_DRAIN_ELAPSED_MS = 100;

  // W293/W299: Target jitter/latency buffer tuning bounds (clamped to [10, 120] ms, default 30 ms).
  const DEFAULT_TARGET_LATENCY_MS = 30;
  const MIN_TARGET_LATENCY_MS = 10;
  const MAX_TARGET_LATENCY_MS = 120;

  // Single-packet PCM safety bound (4 MiB = 524,288 stereo float32 frames ~= 10.9s @ 48 kHz).
  const DEFAULT_MAX_PCM_BYTES = 4 * 1024 * 1024;
  const MAX_ID_STRING_LEN = 128;

  // Underrun de-click micro-fade ramp defaults (32 frames ~= 0.67 ms @ 48 kHz).
  const DEFAULT_DECLICK_FRAMES = 32;
  const MAX_DECLICK_FRAMES = 4096;
  const DEFAULT_FADE_CURVE = 'linear';
  const EMPTY_FLOAT32 = new Float32Array(0);
  const COS_RECURRENCE_RESYNC_MASK = 4095; // Re-sync cosine recurrence every 4096 frames
  const MAX_SCRATCH_VIEW_CACHE = 16;

  // Reusable internal extract options/result for downmixToStereo/resampleLinear when
  // callers pass a scratch pool (always cleared before returning to avoid retaining buffers).
  const _INTERNAL_EXTRACT_RESULT = {
    ok: false,
    pcm: null,
    copiedFromSab: false,
    sanitizedSamples: 0,
    error: '',
  };
  const _INTERNAL_EXTRACT_OPTS = {
    maxPcmBytes: 64 * 1024 * 1024,
    scratch: null,
    outResult: _INTERNAL_EXTRACT_RESULT,
  };

  // Multi-channel ITU-R BS.775 downmix, soft-clip limiter, and resampling safeguards.
  const DEFAULT_SOFT_CLIP_KNEE = 0.95;
  const ITU_BS775_CENTER_GAIN = Math.SQRT1_2;
  const ITU_BS775_SURROUND_GAIN = Math.SQRT1_2;
  const ITU_BS775_LFE_GAIN = 0.0;
  const MAX_RESAMPLE_RATIO = 16;
  const DEFAULT_MAX_RESAMPLE_FRAMES = 524288;
  const SUPPORTED_DOWNMIX_CHANNELS = Object.freeze([1, 2, 4, 6, 8]);

  const OUTPUT_STATES = Object.freeze({
    ACTIVE: 'active',
    SUSPENDED: 'suspended',
    HEADLESS_DRAIN: 'headless-drain',
  });

  // W222: Bounded ring-overflow drop policies and underrun recovery defaults.
  const OVERFLOW_POLICIES = Object.freeze({
    DROP_TAIL: 'drop-tail',
    DROP_NEWEST: 'drop-newest',
    DROP_OLDEST: 'drop-oldest',
  });
  const DEFAULT_OVERFLOW_POLICY = OVERFLOW_POLICIES.DROP_TAIL;
  const DEFAULT_MAX_FILL_RATIO = 1.0;
  const DEFAULT_UNDERRUN_RECOVERY_FRAMES = 0;

  // W235: Host stereo PCM ring defaults mirroring runtime/os/audio_downmix.h (Playgta6AudioRing).
  const GUEST_PCM_RING_DEFAULT_SAMPLE_RATE = 48000;
  const GUEST_PCM_RING_CHANNELS = 2;
  const GUEST_PCM_RING_DEFAULT_CAPACITY_SAMPLES = 1 << 17; // 131072 floats = 65536 stereo frames
  const GUEST_PCM_RING_MIN_CAPACITY_SAMPLES = 2;
  const GUEST_PCM_RING_MAX_CAPACITY_SAMPLES = 1 << 24;
  const GUEST_PCM_RING_DEFAULT_QUANTUM_FRAMES = 480; // 10 ms @ 48 kHz
  const GUEST_PCM_RING_ACTIVE_BURST_CHUNK_FRAMES = 4096; // ~85 ms zero-alloc burst chunk
  const GUEST_PCM_RING_MAX_QUANTUM_FRAMES = 65536;
  const GUEST_TRIM_CROSSFADE_FRAMES = 64; // ~1.33 ms @ 48 kHz click-free backlog trim splice
  const GUEST_HIGH_WATERMARK_MS = 250; // Trim active guest-ring backlog only above 250 ms (>180 ms burst)
  const GUEST_TARGET_WATERMARK_MS = 96; // Post-trim steady cushion (~4608 frames @ 48 kHz)
  const GUEST_RESUME_HIGH_WATERMARK_MS = 120; // Trim stale backlog on resume/unlock above 120 ms
  const GUEST_RESUME_TARGET_WATERMARK_MS = 80; // Post-resume fresh cushion (~3840 frames @ 48 kHz)
  const GUEST_THROTTLED_DRAIN_HIGH_WATERMARK_MS = 200; // Trim during 1 Hz throttled background drain
  const ACTIVE_STALL_EVICT_MS = 60; // Dead-consumer wall-clock threshold (>21 ms OS hardware buffer / GPU jitter) before SPSC fallback
  const ACTIVE_STALL_EVICT_MAX_TICKS = 16; // Fallback tick bound if monotonic clock is frozen in tests

  // O43 (#1028/#663): Live audio-stats BroadcastChannel telemetry defaults matching
  // web/loader.js & web/app_bootstrap.js ('game-progress').
  const PROGRESS_CHANNEL_NAME = 'game-progress';
  const DEFAULT_TELEMETRY_BROADCAST_INTERVAL_MS = 1000;
  const MIN_TELEMETRY_BROADCAST_INTERVAL_MS = 50;
  const MAX_TELEMETRY_BROADCAST_INTERVAL_MS = 60000;

  // Atomic Int32 header layout at byte offset 0..32 of the SharedArrayBuffer.
  const HDR_WRITE_CURSOR = 0;
  const HDR_READ_CURSOR = 1;
  const HDR_CAPACITY_FRAMES = 2;
  const HDR_CHANNELS = 3;
  const HDR_UNDERRUNS = 4;
  const HDR_UNDERRUN_FRAMES = 5;
  const HDR_OVERFLOWS = 6;
  const HDR_DROPPED_FRAMES = 7;

  const RING_HEADER_INT32_COUNT = 8;
  const RING_HEADER_BYTES = RING_HEADER_INT32_COUNT * Int32Array.BYTES_PER_ELEMENT;

  function clampVolume(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return 1.0;
    }
    if (value < 0) return 0.0;
    if (value > 1) return 1.0;
    return value;
  }

  function sanitizeNonNegInt(value, fallback = 0) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      return fallback;
    }
    return Math.floor(value);
  }

  function sanitizePositiveInt(value, fallback, minVal, maxVal) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return fallback;
    }
    const n = Math.floor(value);
    if (n < minVal) return fallback;
    if (maxVal !== undefined && n > maxVal) return maxVal;
    return n;
  }

  function sanitizeDeclickFrames(value, fallback) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return fallback;
    }
    const n = Math.floor(value);
    if (n < 0) return fallback;
    if (n > MAX_DECLICK_FRAMES) return MAX_DECLICK_FRAMES;
    return n;
  }

  function sanitizeFadeCurve(value, fallback = DEFAULT_FADE_CURVE) {
    if (value === 'cosine' || value === 'linear') {
      return value;
    }
    return fallback;
  }

  function clampTargetLatencyMs(value, fallback = DEFAULT_TARGET_LATENCY_MS) {
    const safeFallback =
      typeof fallback === 'number' && Number.isFinite(fallback)
        ? fallback <= MIN_TARGET_LATENCY_MS
          ? MIN_TARGET_LATENCY_MS
          : fallback >= MAX_TARGET_LATENCY_MS
            ? MAX_TARGET_LATENCY_MS
            : fallback
        : DEFAULT_TARGET_LATENCY_MS;
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      return safeFallback;
    }
    if (value <= MIN_TARGET_LATENCY_MS) return MIN_TARGET_LATENCY_MS;
    if (value >= MAX_TARGET_LATENCY_MS) return MAX_TARGET_LATENCY_MS;
    return value;
  }

  function sanitizeSampleRate(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return DEFAULT_SAMPLE_RATE;
    }
    if (value < MIN_SAMPLE_RATE) return MIN_SAMPLE_RATE;
    if (value > MAX_SAMPLE_RATE) return MAX_SAMPLE_RATE;
    return Math.floor(value);
  }

  function sanitizeMessageId(id) {
    if (typeof id === 'number' && Number.isFinite(id)) {
      return id;
    }
    if (typeof id === 'string') {
      return id.length > MAX_ID_STRING_LEN ? id.slice(0, MAX_ID_STRING_LEN) : id;
    }
    return undefined;
  }

  function isSharedArrayBuffer(buf) {
    return typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer;
  }

  function isDetachedArrayBuffer(buf) {
    if (!(buf instanceof ArrayBuffer)) {
      return false;
    }
    if (buf.detached === true) {
      return true;
    }
    try {
      // Accessing byteLength never throws in ES2024, but guard older engines.
      void buf.byteLength;
      return false;
    } catch (_err) {
      return true;
    }
  }

  function normalizeOutputState(value, fallback = OUTPUT_STATES.HEADLESS_DRAIN) {
    if (typeof value !== 'string') {
      return fallback;
    }
    const normalized = value.trim().toLowerCase();
    if (
      normalized === OUTPUT_STATES.ACTIVE ||
      normalized === 'running' ||
      normalized === 'worklet' ||
      normalized === 'live' ||
      normalized === 'playing'
    ) {
      return OUTPUT_STATES.ACTIVE;
    }
    if (normalized === OUTPUT_STATES.SUSPENDED || normalized === 'interrupted') {
      return OUTPUT_STATES.SUSPENDED;
    }
    if (
      normalized === OUTPUT_STATES.HEADLESS_DRAIN ||
      normalized === 'headless' ||
      normalized === 'unavailable' ||
      normalized === 'unsupported' ||
      normalized === 'closed' ||
      normalized === 'error'
    ) {
      return OUTPUT_STATES.HEADLESS_DRAIN;
    }
    return fallback;
  }

  function detectDefaultOutputState() {
    const hasAudioCtx =
      typeof globalThis !== 'undefined' &&
      (typeof globalThis.AudioContext === 'function' ||
        typeof globalThis.webkitAudioContext === 'function');
    const hasWorklet =
      typeof globalThis !== 'undefined' &&
      typeof globalThis.AudioWorkletNode === 'function';
    if (hasAudioCtx && hasWorklet) {
      return OUTPUT_STATES.ACTIVE;
    }
    return OUTPUT_STATES.HEADLESS_DRAIN;
  }

  function normalizeOverflowPolicy(value, fallback = DEFAULT_OVERFLOW_POLICY) {
    if (typeof value !== 'string') {
      return fallback;
    }
    const normalized = value.trim().toLowerCase();
    if (
      normalized === OVERFLOW_POLICIES.DROP_TAIL ||
      normalized === 'tail' ||
      normalized === 'partial'
    ) {
      return OVERFLOW_POLICIES.DROP_TAIL;
    }
    if (
      normalized === OVERFLOW_POLICIES.DROP_NEWEST ||
      normalized === 'newest' ||
      normalized === 'drop-packet'
    ) {
      return OVERFLOW_POLICIES.DROP_NEWEST;
    }
    if (
      normalized === OVERFLOW_POLICIES.DROP_OLDEST ||
      normalized === 'oldest' ||
      normalized === 'evict-oldest'
    ) {
      return OVERFLOW_POLICIES.DROP_OLDEST;
    }
    return fallback;
  }

  function sanitizeFillRatio(value, fallback = DEFAULT_MAX_FILL_RATIO) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return fallback;
    }
    if (value > 1.0) return 1.0;
    return value;
  }

  function sanitizeErrorMessage(err) {
    const raw = err && typeof err.message === 'string' ? err.message : String(err || 'Unknown audio worker error');
    // Strip control characters and clamp length to prevent noisy/leaky error strings.
    const clean = raw.replace(/[\r\n\t]+/g, ' ').trim();
    return clean.length > 200 ? clean.slice(0, 200) : clean;
  }

  /**
   * Filters a postMessage transfer list so SharedArrayBuffers, detached
   * ArrayBuffers, TypedArray views, and duplicates never trigger DataCloneError.
   *
   * @param {any[]} transfer
   * @returns {any[]}
   */
  function sanitizeTransferList(transfer) {
    if (!Array.isArray(transfer) || transfer.length === 0) {
      return [];
    }
    const out = [];
    const seen = new Set();
    for (let i = 0; i < transfer.length; i++) {
      const item = transfer[i];
      if (!item || typeof item !== 'object') {
        continue;
      }
      if (isSharedArrayBuffer(item)) {
        continue;
      }
      if (item instanceof ArrayBuffer && isDetachedArrayBuffer(item)) {
        continue;
      }
      if (ArrayBuffer.isView(item)) {
        continue;
      }
      if (!seen.has(item)) {
        seen.add(item);
        out.push(item);
      }
    }
    return out;
  }

  /**
   * De-interleaves interleaved PCM samples (`[L0, R0, L1, R1, ...]`) into
   * planar per-channel Float32Arrays (`[[L0, L1, ...], [R0, R1, ...]]`).
   * Writes into caller-provided `planarTargets` with zero TypedArray allocation
   * when supplied. Non-finite samples are sanitized to 0.0.
   *
   * @param {Float32Array} interleaved
   * @param {number} channels Channel count in [1, MAX_CHANNELS].
   * @param {number} [frameCount] Frames to convert (defaults to floor(interleaved.length / channels)).
   * @param {Float32Array[]} [planarTargets] Optional pre-allocated planar channel buffers.
   * @returns {Float32Array[]}
   */
  function deinterleavePcm(interleaved, channels, frameCount, planarTargets) {
    if (!Number.isInteger(channels) || channels < 1 || channels > MAX_CHANNELS) {
      throw new RangeError(
        `deinterleavePcm channels must be an integer in [1, ${MAX_CHANNELS}], got ${channels}`
      );
    }
    if (!ArrayBuffer.isView(interleaved) && !Array.isArray(interleaved)) {
      throw new TypeError('deinterleavePcm interleaved source must be a TypedArray or Array');
    }
    const maxFrames = Math.floor(interleaved.length / channels);
    const frames =
      frameCount !== undefined
        ? Math.min(sanitizeNonNegInt(frameCount, 0), maxFrames)
        : maxFrames;

    let targets;
    if (Array.isArray(planarTargets) && planarTargets.length >= channels) {
      for (let c = 0; c < channels; c++) {
        if (!(planarTargets[c] instanceof Float32Array) || planarTargets[c].length < frames) {
          throw new RangeError(
            `deinterleavePcm planar target channel ${c} must be a Float32Array of length >= ${frames}`
          );
        }
      }
      targets = planarTargets;
    } else {
      targets = new Array(channels);
      for (let c = 0; c < channels; c++) {
        targets[c] = new Float32Array(frames);
      }
    }

    if (frames <= 0) {
      return targets;
    }

    if (channels === 2) {
      const ch0 = targets[0];
      const ch1 = targets[1];
      for (let f = 0; f < frames; f++) {
        const idx = f * 2;
        const l = interleaved[idx];
        const r = interleaved[idx + 1];
        ch0[f] = Number.isFinite(l) ? l : 0.0;
        ch1[f] = Number.isFinite(r) ? r : 0.0;
      }
    } else if (channels === 1) {
      const ch0 = targets[0];
      for (let f = 0; f < frames; f++) {
        const v = interleaved[f];
        ch0[f] = Number.isFinite(v) ? v : 0.0;
      }
    } else {
      for (let f = 0; f < frames; f++) {
        const base = f * channels;
        for (let c = 0; c < channels; c++) {
          const v = interleaved[base + c];
          targets[c][f] = Number.isFinite(v) ? v : 0.0;
        }
      }
    }

    return targets;
  }

  /**
   * Interleaves planar per-channel Float32Arrays into a single interleaved
   * Float32Array. Writes into caller-provided `interleavedTarget` with zero
   * TypedArray allocation when supplied. Non-finite samples are sanitized to 0.0.
   *
   * @param {Float32Array[]} planarChannels
   * @param {number} [frameCount]
   * @param {Float32Array} [interleavedTarget]
   * @returns {Float32Array}
   */
  function interleavePcm(planarChannels, frameCount, interleavedTarget) {
    if (!Array.isArray(planarChannels) || planarChannels.length < 1 || planarChannels.length > MAX_CHANNELS) {
      throw new RangeError(
        `interleavePcm planarChannels must be an array of 1..${MAX_CHANNELS} channel buffers`
      );
    }
    const channels = planarChannels.length;
    let minLen = Infinity;
    for (let c = 0; c < channels; c++) {
      const ch = planarChannels[c];
      if (!(ch instanceof Float32Array)) {
        throw new TypeError(`interleavePcm channel ${c} must be a Float32Array`);
      }
      if (!isSharedArrayBuffer(ch.buffer) && isDetachedArrayBuffer(ch.buffer)) {
        throw new TypeError(`interleavePcm channel ${c} buffer is detached`);
      }
      if (ch.length < minLen) {
        minLen = ch.length;
      }
    }

    const frames =
      frameCount !== undefined
        ? Math.min(sanitizeNonNegInt(frameCount, 0), minLen)
        : minLen;
    const totalSamples = frames * channels;

    let out;
    if (interleavedTarget instanceof Float32Array && interleavedTarget.length >= totalSamples) {
      out = interleavedTarget;
    } else {
      out = new Float32Array(totalSamples);
    }

    if (frames <= 0) {
      return out;
    }

    if (channels === 2) {
      const ch0 = planarChannels[0];
      const ch1 = planarChannels[1];
      for (let f = 0; f < frames; f++) {
        const idx = f * 2;
        const l = ch0[f];
        const r = ch1[f];
        out[idx] = Number.isFinite(l) ? l : 0.0;
        out[idx + 1] = Number.isFinite(r) ? r : 0.0;
      }
    } else if (channels === 1) {
      const ch0 = planarChannels[0];
      for (let f = 0; f < frames; f++) {
        const v = ch0[f];
        out[f] = Number.isFinite(v) ? v : 0.0;
      }
    } else {
      for (let f = 0; f < frames; f++) {
        const base = f * channels;
        for (let c = 0; c < channels; c++) {
          const v = planarChannels[c][f];
          out[base + c] = Number.isFinite(v) ? v : 0.0;
        }
      }
    }

    return out;
  }

  /**
   * Reusable PCM scratch buffer pool supporting named interleaved slots and
   * per-channel planar scratch buffers so steady-state packet forwarding,
   * downmixing, resampling, and interleaved<->planar conversion perform zero
   * per-packet TypedArray allocations.
   */
  class PcmScratchPool {
    constructor(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      this._maxSamples = sanitizePositiveInt(
        opts.maxSamples,
        Math.floor(DEFAULT_MAX_PCM_BYTES / Float32Array.BYTES_PER_ELEMENT),
        1,
        16 * 1024 * 1024
      );
      this._slots = Object.create(null);
      this._slotCount = 0;

      this._planarBacking = [];
      this._planarViews = [];
      this._planarViewCache = Object.create(null);
      this._planarViewCacheSize = 0;
      this._planarChannels = 0;
      this._planarCapacityFrames = 0;
      this._planarViewFrames = 0;

      this._lastRawSab = null;
      this._lastRawSabSamples = 0;
      this._lastRawSabView = null;

      this._lastRawByteBuf = null;
      this._lastRawByteOffset = 0;
      this._lastRawByteLen = 0;
      this._lastRawByteView = null;

      this._lastRawF32Buf = null;
      this._lastRawF32Offset = 0;
      this._lastRawF32Len = 0;
      this._lastRawF32View = null;

      this._allocations = 0;
      this._reuses = 0;
    }

    get allocations() {
      return this._allocations;
    }

    get reuses() {
      return this._reuses;
    }

    get planarChannels() {
      return this._planarChannels;
    }

    get planarCapacityFrames() {
      return this._planarCapacityFrames;
    }

    _getOrCreateSlot(slotName, sampleCount, needBytes) {
      const count = sanitizeNonNegInt(sampleCount, 0);
      if (count > this._maxSamples) {
        throw new RangeError(
          `PCM scratch request (${count} samples) exceeds maxSamples (${this._maxSamples})`
        );
      }

      let slot = this._slots[slotName];
      if (!slot) {
        const cap = count > 0 ? count : 1;
        const backing = new Float32Array(cap);
        const byteBacking = needBytes ? new Uint8Array(backing.buffer) : null;
        const view = count === cap ? backing : backing.subarray(0, count);
        const byteLen = count * Float32Array.BYTES_PER_ELEMENT;
        const byteView = needBytes
          ? count === cap
            ? byteBacking
            : byteBacking.subarray(0, byteLen)
          : null;
        const viewCache = Object.create(null);
        viewCache[count] = view;
        const byteViewCache = Object.create(null);
        let byteViewCacheSize = 0;
        if (needBytes) {
          byteViewCache[byteLen] = byteView;
          byteViewCacheSize = 1;
        }
        slot = {
          backing,
          capacity: cap,
          view,
          viewLen: count,
          viewCache,
          viewCacheSize: 1,
          byteBacking,
          byteView,
          byteViewLen: needBytes ? byteLen : -1,
          byteViewCache,
          byteViewCacheSize,
          pair: { floats: null, bytes: null },
        };
        slot.pair.floats = slot.view;
        slot.pair.bytes = slot.byteView;
        this._slots[slotName] = slot;
        this._slotCount += 1;
        this._allocations += 1;
        return slot;
      }

      if (count > slot.capacity) {
        const backing = new Float32Array(count);
        slot.backing = backing;
        slot.capacity = count;
        slot.view = backing;
        slot.viewLen = count;
        slot.viewCache = Object.create(null);
        slot.viewCache[count] = backing;
        slot.viewCacheSize = 1;
        if (needBytes || slot.byteBacking !== null) {
          const byteLen = count * Float32Array.BYTES_PER_ELEMENT;
          slot.byteBacking = new Uint8Array(backing.buffer);
          slot.byteView = slot.byteBacking;
          slot.byteViewLen = byteLen;
          slot.byteViewCache = Object.create(null);
          slot.byteViewCache[byteLen] = slot.byteBacking;
          slot.byteViewCacheSize = 1;
        }
        slot.pair.floats = slot.view;
        slot.pair.bytes = slot.byteView;
        this._allocations += 1;
        return slot;
      }

      if (slot.viewLen !== count) {
        let cachedView = slot.viewCache[count];
        if (!cachedView) {
          cachedView = count === slot.capacity ? slot.backing : slot.backing.subarray(0, count);
          if (slot.viewCacheSize < MAX_SCRATCH_VIEW_CACHE) {
            slot.viewCache[count] = cachedView;
            slot.viewCacheSize += 1;
          }
        }
        slot.view = cachedView;
        slot.viewLen = count;
        slot.pair.floats = cachedView;
      }

      if (needBytes) {
        const byteLen = count * Float32Array.BYTES_PER_ELEMENT;
        if (!slot.byteBacking) {
          slot.byteBacking = new Uint8Array(slot.backing.buffer);
          slot.byteViewLen = -1;
          slot.byteViewCache = Object.create(null);
          slot.byteViewCacheSize = 0;
        }
        if (slot.byteViewLen !== byteLen) {
          let cachedByteView = slot.byteViewCache[byteLen];
          if (!cachedByteView) {
            cachedByteView =
              byteLen === slot.byteBacking.byteLength
                ? slot.byteBacking
                : slot.byteBacking.subarray(0, byteLen);
            if (slot.byteViewCacheSize < MAX_SCRATCH_VIEW_CACHE) {
              slot.byteViewCache[byteLen] = cachedByteView;
              slot.byteViewCacheSize += 1;
            }
          }
          slot.byteView = cachedByteView;
          slot.byteViewLen = byteLen;
          slot.pair.bytes = cachedByteView;
        }
      }

      this._reuses += 1;
      return slot;
    }

    acquireFloat32(slotName, sampleCount) {
      return this._getOrCreateSlot(slotName, sampleCount, false).view;
    }

    /**
     * Returns the pre-allocated backing `Float32Array` for `slotName` with length
     * `>= minSampleCount`. Unlike `acquireFloat32`, varying `minSampleCount` below
     * capacity never allocates a `.subarray()` wrapper, guaranteeing zero allocations
     * across variable-size stereo ring drains.
     *
     * @param {string} slotName
     * @param {number} minSampleCount
     * @returns {Float32Array}
     */
    acquireFloat32Backing(slotName, minSampleCount) {
      const count = sanitizeNonNegInt(minSampleCount, 0);
      if (count > this._maxSamples) {
        throw new RangeError(
          `PCM scratch request (${count} samples) exceeds maxSamples (${this._maxSamples})`
        );
      }

      let slot = this._slots[slotName];
      if (!slot) {
        const cap = count > 0 ? count : 1;
        const backing = new Float32Array(cap);
        const view = count === cap ? backing : backing.subarray(0, count);
        const viewCache = Object.create(null);
        viewCache[count] = view;
        slot = {
          backing,
          capacity: cap,
          view,
          viewLen: count,
          viewCache,
          viewCacheSize: 1,
          byteBacking: null,
          byteView: null,
          byteViewLen: -1,
          byteViewCache: Object.create(null),
          byteViewCacheSize: 0,
          pair: { floats: null, bytes: null },
        };
        slot.pair.floats = slot.view;
        slot.pair.bytes = null;
        this._slots[slotName] = slot;
        this._slotCount += 1;
        this._allocations += 1;
        return slot.backing;
      }

      if (count > slot.capacity) {
        const backing = new Float32Array(count);
        slot.backing = backing;
        slot.capacity = count;
        slot.view = backing;
        slot.viewLen = count;
        slot.viewCache = Object.create(null);
        slot.viewCache[count] = backing;
        slot.viewCacheSize = 1;
        if (slot.byteBacking !== null) {
          const byteLen = count * Float32Array.BYTES_PER_ELEMENT;
          slot.byteBacking = new Uint8Array(backing.buffer);
          slot.byteView = slot.byteBacking;
          slot.byteViewLen = byteLen;
          slot.byteViewCache = Object.create(null);
          slot.byteViewCache[byteLen] = slot.byteBacking;
          slot.byteViewCacheSize = 1;
        }
        slot.pair.floats = slot.view;
        slot.pair.bytes = slot.byteView;
        this._allocations += 1;
        return slot.backing;
      }

      this._reuses += 1;
      return slot.backing;
    }

    acquireFloat32WithBytes(slotName, sampleCount) {
      return this._getOrCreateSlot(slotName, sampleCount, true).pair;
    }

    acquireRawSabView(sab, sampleCount) {
      if (this._lastRawSab === sab && this._lastRawSabSamples === sampleCount && this._lastRawSabView) {
        return this._lastRawSabView;
      }
      const view = new Float32Array(sab, 0, sampleCount);
      this._lastRawSab = sab;
      this._lastRawSabSamples = sampleCount;
      this._lastRawSabView = view;
      return view;
    }

    acquireRawByteView(buf, byteOffset, byteLength) {
      if (
        this._lastRawByteBuf === buf &&
        this._lastRawByteOffset === byteOffset &&
        this._lastRawByteLen === byteLength &&
        this._lastRawByteView
      ) {
        return this._lastRawByteView;
      }
      const view = new Uint8Array(buf, byteOffset, byteLength);
      this._lastRawByteBuf = buf;
      this._lastRawByteOffset = byteOffset;
      this._lastRawByteLen = byteLength;
      this._lastRawByteView = view;
      return view;
    }

    acquireRawFloat32View(buf, byteOffset, sampleCount) {
      if (
        this._lastRawF32Buf === buf &&
        this._lastRawF32Offset === byteOffset &&
        this._lastRawF32Len === sampleCount &&
        this._lastRawF32View
      ) {
        return this._lastRawF32View;
      }
      const view = new Float32Array(buf, byteOffset, sampleCount);
      this._lastRawF32Buf = buf;
      this._lastRawF32Offset = byteOffset;
      this._lastRawF32Len = sampleCount;
      this._lastRawF32View = view;
      return view;
    }

    /**
     * Returns a reusable array of `channels` Float32Arrays each of exact length
     * `frameCount`. Steady-state calls with the same `(channels, frameCount)`
     * return the exact same Array and Float32Array instances with zero allocations.
     *
     * @param {number} channels
     * @param {number} frameCount
     * @returns {Float32Array[]}
     */
    acquirePlanar(channels, frameCount) {
      if (!Number.isInteger(channels) || channels < 1 || channels > MAX_CHANNELS) {
        throw new RangeError(
          `acquirePlanar channels must be an integer in [1, ${MAX_CHANNELS}], got ${channels}`
        );
      }
      const frames = sanitizeNonNegInt(frameCount, 0);
      if (frames * channels > this._maxSamples) {
        throw new RangeError(
          `Planar PCM scratch request (${frames}x${channels} samples) exceeds maxSamples (${this._maxSamples})`
        );
      }

      const needMoreChannels = channels > this._planarBacking.length;
      const needMoreFrames = frames > this._planarCapacityFrames;

      if (needMoreChannels || needMoreFrames) {
        const nextCap = needMoreFrames ? Math.max(frames, 1) : Math.max(this._planarCapacityFrames, 1);
        const targetChannelCount = Math.max(channels, this._planarBacking.length);
        if (needMoreFrames) {
          this._planarBacking = new Array(targetChannelCount);
          for (let c = 0; c < targetChannelCount; c++) {
            this._planarBacking[c] = new Float32Array(nextCap);
          }
          this._allocations += targetChannelCount;
          this._planarViewCache = Object.create(null);
          this._planarViewCacheSize = 0;
        } else {
          for (let c = this._planarBacking.length; c < targetChannelCount; c++) {
            this._planarBacking.push(new Float32Array(nextCap));
            this._allocations += 1;
          }
        }
        this._planarCapacityFrames = nextCap;
        const views = new Array(channels);
        for (let c = 0; c < channels; c++) {
          views[c] =
            frames === nextCap ? this._planarBacking[c] : this._planarBacking[c].subarray(0, frames);
        }
        const cacheKey = channels * 0x10000000 + frames;
        if (this._planarViewCacheSize < MAX_SCRATCH_VIEW_CACHE * 2) {
          this._planarViewCache[cacheKey] = views;
          this._planarViewCacheSize += 1;
        }
        this._planarViews = views;
        this._planarChannels = channels;
        this._planarViewFrames = frames;
        return this._planarViews;
      }

      if (channels !== this._planarChannels || frames !== this._planarViewFrames) {
        const cacheKey = channels * 0x10000000 + frames;
        let cached = this._planarViewCache[cacheKey];
        if (!cached) {
          cached = new Array(channels);
          for (let c = 0; c < channels; c++) {
            cached[c] =
              frames === this._planarCapacityFrames
                ? this._planarBacking[c]
                : this._planarBacking[c].subarray(0, frames);
          }
          if (this._planarViewCacheSize < MAX_SCRATCH_VIEW_CACHE * 2) {
            this._planarViewCache[cacheKey] = cached;
            this._planarViewCacheSize += 1;
          }
        }
        this._planarViews = cached;
        this._planarChannels = channels;
        this._planarViewFrames = frames;
      }

      this._reuses += 1;
      return this._planarViews;
    }

    deinterleave(interleaved, channels, frameCount) {
      if (!Number.isInteger(channels) || channels < 1 || channels > MAX_CHANNELS) {
        throw new RangeError(
          `deinterleave channels must be an integer in [1, ${MAX_CHANNELS}], got ${channels}`
        );
      }
      if (!ArrayBuffer.isView(interleaved) && !Array.isArray(interleaved)) {
        throw new TypeError('deinterleave source must be a TypedArray or Array');
      }
      const maxFrames = Math.floor(interleaved.length / channels);
      const frames =
        frameCount !== undefined
          ? Math.min(sanitizeNonNegInt(frameCount, 0), maxFrames)
          : maxFrames;
      const targets = this.acquirePlanar(channels, frames);
      return deinterleavePcm(interleaved, channels, frames, targets);
    }

    interleave(planarChannels, frameCount) {
      if (!Array.isArray(planarChannels) || planarChannels.length < 1 || planarChannels.length > MAX_CHANNELS) {
        throw new RangeError(
          `interleave planarChannels must be an array of 1..${MAX_CHANNELS} channel buffers`
        );
      }
      const channels = planarChannels.length;
      let minLen = Infinity;
      for (let c = 0; c < channels; c++) {
        const ch = planarChannels[c];
        if (!(ch instanceof Float32Array)) {
          throw new TypeError(`interleave channel ${c} must be a Float32Array`);
        }
        if (ch.length < minLen) {
          minLen = ch.length;
        }
      }
      const frames =
        frameCount !== undefined
          ? Math.min(sanitizeNonNegInt(frameCount, 0), minLen)
          : minLen;
      const target = this.acquireFloat32('interleave', frames * channels);
      return interleavePcm(planarChannels, frames, target);
    }

    resetStats() {
      this._allocations = 0;
      this._reuses = 0;
    }

    clear() {
      this._slots = Object.create(null);
      this._slotCount = 0;
      this._planarBacking = [];
      this._planarViews = [];
      this._planarViewCache = Object.create(null);
      this._planarViewCacheSize = 0;
      this._planarChannels = 0;
      this._planarCapacityFrames = 0;
      this._planarViewFrames = 0;
      this._lastRawSab = null;
      this._lastRawSabSamples = 0;
      this._lastRawSabView = null;
      this._lastRawByteBuf = null;
      this._lastRawByteOffset = 0;
      this._lastRawByteLen = 0;
      this._lastRawByteView = null;
      this._lastRawF32Buf = null;
      this._lastRawF32Offset = 0;
      this._lastRawF32Len = 0;
      this._lastRawF32View = null;
      this._allocations = 0;
      this._reuses = 0;
    }

    getStats() {
      return {
        allocations: this._allocations,
        reuses: this._reuses,
        planarChannels: this._planarChannels,
        planarCapacityFrames: this._planarCapacityFrames,
        slotCount: this._slotCount,
      };
    }
  }

  /**
   * Validates and extracts PCM Float32 samples safely from any supported payload.
   * - Rejects null, malformed types, detached ArrayBuffers, and oversized payloads.
   * - Copies SharedArrayBuffer-backed views (and unaligned TypedArray views) into
   *   a fresh or reusable scratch ArrayBuffer-backed Float32Array to prevent
   *   concurrent mutation races and DataCloneError on transfer.
   * - Sanitizes non-finite float samples (NaN / +-Infinity) to 0.0.
   *
   * @param {any} samples
   * @param {{ maxPcmBytes?: number, scratch?: PcmScratchPool }} [options]
   * @returns {{ ok: true, pcm: Float32Array, copiedFromSab: boolean, sanitizedSamples: number } | { ok: false, error: string }}
   */
  function extractSafePcmSamples(samples, options = {}) {
    const maxPcmBytes = sanitizePositiveInt(
      options && options.maxPcmBytes,
      DEFAULT_MAX_PCM_BYTES,
      Float32Array.BYTES_PER_ELEMENT,
      64 * 1024 * 1024
    );
    const maxSamples = Math.floor(maxPcmBytes / Float32Array.BYTES_PER_ELEMENT);
    const scratch =
      options &&
      options.scratch &&
      typeof options.scratch.acquireFloat32 === 'function'
        ? options.scratch
        : null;
    const outRes =
      options && options.outResult && typeof options.outResult === 'object'
        ? options.outResult
        : null;

    const finishOk = (pcm, copiedFromSab, sanitizedSamples) => {
      if (outRes) {
        outRes.ok = true;
        outRes.pcm = pcm;
        outRes.copiedFromSab = copiedFromSab;
        outRes.sanitizedSamples = sanitizedSamples;
        outRes.error = '';
        return outRes;
      }
      return { ok: true, pcm, copiedFromSab, sanitizedSamples };
    };
    const finishErr = (error) => {
      if (outRes) {
        outRes.ok = false;
        outRes.pcm = null;
        outRes.copiedFromSab = false;
        outRes.sanitizedSamples = 0;
        outRes.error = error;
        return outRes;
      }
      return { ok: false, error };
    };

    if (samples === null || samples === undefined) {
      return finishErr('PCM samples missing or null');
    }

    // 1. Raw SharedArrayBuffer
    if (isSharedArrayBuffer(samples)) {
      const byteLen = samples.byteLength;
      if (byteLen > maxPcmBytes) {
        return finishErr(
          `PCM payload (${byteLen} bytes) exceeds max allowed (${maxPcmBytes} bytes)`
        );
      }
      const sampleCount = Math.floor(byteLen / Float32Array.BYTES_PER_ELEMENT);
      const view =
        scratch && typeof scratch.acquireRawSabView === 'function'
          ? scratch.acquireRawSabView(samples, sampleCount)
          : new Float32Array(samples, 0, sampleCount);
      const copy = scratch
        ? scratch.acquireFloat32('extract', sampleCount)
        : new Float32Array(sampleCount);
      let sanitizedSamples = 0;
      for (let i = 0; i < sampleCount; i++) {
        const v = view[i];
        if (Number.isFinite(v)) {
          copy[i] = v;
        } else {
          copy[i] = 0.0;
          sanitizedSamples += 1;
        }
      }
      return finishOk(copy, true, sanitizedSamples);
    }

    // 2. Raw ArrayBuffer
    if (samples instanceof ArrayBuffer) {
      if (isDetachedArrayBuffer(samples)) {
        return finishErr('PCM ArrayBuffer is detached');
      }
      const byteLen = samples.byteLength;
      if (byteLen > maxPcmBytes) {
        return finishErr(
          `PCM payload (${byteLen} bytes) exceeds max allowed (${maxPcmBytes} bytes)`
        );
      }
      const sampleCount = Math.floor(byteLen / Float32Array.BYTES_PER_ELEMENT);
      const view =
        scratch && typeof scratch.acquireRawFloat32View === 'function'
          ? scratch.acquireRawFloat32View(samples, 0, sampleCount)
          : new Float32Array(samples, 0, sampleCount);
      let sanitizedSamples = 0;
      let copy = null;
      for (let i = 0; i < sampleCount; i++) {
        if (!Number.isFinite(view[i])) {
          if (!copy) {
            copy = scratch
              ? scratch.acquireFloat32('extract', sampleCount)
              : new Float32Array(sampleCount);
            copy.set(view);
          }
          copy[i] = 0.0;
          sanitizedSamples += 1;
        }
      }
      return finishOk(copy || view, false, sanitizedSamples);
    }

    // 3. TypedArray or DataView (Float32Array, Uint8Array, etc.)
    if (ArrayBuffer.isView(samples)) {
      let backing;
      let byteOffset;
      let byteLength;
      try {
        backing = samples.buffer;
        byteOffset = samples.byteOffset;
        byteLength = samples.byteLength;
      } catch (_err) {
        return finishErr('PCM TypedArray view is unreadable or detached');
      }

      if (!backing) {
        return finishErr('PCM view has no backing buffer');
      }
      const fromSab = isSharedArrayBuffer(backing);
      if (!fromSab && isDetachedArrayBuffer(backing)) {
        return finishErr('PCM TypedArray backing ArrayBuffer is detached');
      }
      if (byteLength > maxPcmBytes) {
        return finishErr(
          `PCM payload (${byteLength} bytes) exceeds max allowed (${maxPcmBytes} bytes)`
        );
      }

      const alignedByteLength =
        Math.floor(byteLength / Float32Array.BYTES_PER_ELEMENT) * Float32Array.BYTES_PER_ELEMENT;
      const sampleCount = alignedByteLength / Float32Array.BYTES_PER_ELEMENT;

      if (fromSab || byteOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
        // Always copy SAB-backed or unaligned views into a standalone or reusable scratch ArrayBuffer.
        let copy;
        let sanitizedSamples = 0;
        if (samples instanceof Float32Array) {
          copy = scratch
            ? scratch.acquireFloat32('extract', sampleCount)
            : new Float32Array(sampleCount);
          for (let i = 0; i < sampleCount; i++) {
            const v = samples[i];
            if (Number.isFinite(v)) {
              copy[i] = v;
            } else {
              copy[i] = 0.0;
              sanitizedSamples += 1;
            }
          }
          return finishOk(copy, fromSab, sanitizedSamples);
        }

        if (scratch && typeof scratch.acquireFloat32WithBytes === 'function') {
          const pair = scratch.acquireFloat32WithBytes('extract', sampleCount);
          if (samples instanceof Uint8Array) {
            for (let i = 0; i < alignedByteLength; i++) {
              pair.bytes[i] = samples[i];
            }
          } else {
            const srcBytes =
              typeof scratch.acquireRawByteView === 'function'
                ? scratch.acquireRawByteView(backing, byteOffset, alignedByteLength)
                : new Uint8Array(backing, byteOffset, alignedByteLength);
            pair.bytes.set(srcBytes);
          }
          copy = pair.floats;
        } else {
          const byteCopy = new Uint8Array(alignedByteLength);
          byteCopy.set(new Uint8Array(backing, byteOffset, alignedByteLength));
          copy = new Float32Array(byteCopy.buffer);
        }

        for (let i = 0; i < sampleCount; i++) {
          if (!Number.isFinite(copy[i])) {
            copy[i] = 0.0;
            sanitizedSamples += 1;
          }
        }
        return finishOk(copy, fromSab, sanitizedSamples);
      }

      const view =
        samples instanceof Float32Array
          ? samples
          : scratch && typeof scratch.acquireRawFloat32View === 'function'
            ? scratch.acquireRawFloat32View(backing, byteOffset, sampleCount)
            : new Float32Array(backing, byteOffset, sampleCount);
      let sanitizedSamples = 0;
      let copy = null;
      for (let i = 0; i < view.length; i++) {
        if (!Number.isFinite(view[i])) {
          if (!copy) {
            copy = scratch
              ? scratch.acquireFloat32('extract', view.length)
              : new Float32Array(view.length);
            copy.set(view);
          }
          copy[i] = 0.0;
          sanitizedSamples += 1;
        }
      }
      return finishOk(copy || view, false, sanitizedSamples);
    }

    // 4. Plain JS number[]
    if (Array.isArray(samples)) {
      if (samples.length > maxSamples) {
        return finishErr(
          `PCM array (${samples.length} samples) exceeds max allowed (${maxSamples} samples)`
        );
      }
      const out = scratch
        ? scratch.acquireFloat32('extract', samples.length)
        : new Float32Array(samples.length);
      let sanitizedSamples = 0;
      for (let i = 0; i < samples.length; i++) {
        const v = samples[i];
        if (typeof v !== 'number') {
          return finishErr(`PCM array contains non-numeric element at index ${i}`);
        }
        if (Number.isFinite(v)) {
          out[i] = v;
        } else {
          out[i] = 0.0;
          sanitizedSamples += 1;
        }
      }
      return finishOk(out, false, sanitizedSamples);
    }

    return finishErr(
      'PCM samples must be a Float32Array, ArrayBuffer, SharedArrayBuffer, TypedArray, or number[]'
    );
  }

  /**
   * Produces a standalone Float32Array + safe transfer list `[pcm.buffer]`
   * suitable for `postMessage(..., transfer)`. Always copies SAB-backed or
   * sliced views so transfer never throws DataCloneError or detaches guest memory.
   *
   * @param {any} samples
   * @param {{ maxPcmBytes?: number }} [options]
   */
  function clonePcmForTransfer(samples, options = {}) {
    const res = extractSafePcmSamples(samples, options);
    if (!res.ok) {
      return res;
    }
    const pcm = res.pcm;
    const needsOwnedCopy =
      isSharedArrayBuffer(pcm.buffer) ||
      pcm.byteOffset !== 0 ||
      pcm.byteLength !== pcm.buffer.byteLength;
    if (needsOwnedCopy) {
      const owned = new Float32Array(pcm.length);
      owned.set(pcm);
      return {
        ok: true,
        pcm: owned,
        transfer: [owned.buffer],
        copiedFromSab: res.copiedFromSab,
        sanitizedSamples: res.sanitizedSamples,
      };
    }
    return {
      ok: true,
      pcm,
      transfer: [pcm.buffer],
      copiedFromSab: res.copiedFromSab,
      sanitizedSamples: res.sanitizedSamples,
    };
  }

  function toFloat32Array(samples) {
    const extracted = extractSafePcmSamples(samples, { maxPcmBytes: 64 * 1024 * 1024 });
    if (!extracted.ok) {
      throw new TypeError(extracted.error);
    }
    return extracted.pcm;
  }

  function clampSample(value, min, max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
  }

  /**
   * Smooth C1-continuous soft-clip peak limiter backed by Math.tanh.
   * Non-finite inputs (NaN / +-Infinity) are sanitized to 0.0.
   *
   * @param {number} x
   * @param {number} [knee=0.95]
   * @returns {number}
   */
  function softClipSample(x, knee = DEFAULT_SOFT_CLIP_KNEE) {
    if (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0) {
      throw new RangeError(`knee must be in [0, 1), got ${knee}`);
    }
    if (!Number.isFinite(x)) {
      return 0.0;
    }
    const absX = Math.abs(x);
    if (absX <= knee) {
      return x;
    }
    const headroom = 1.0 - knee;
    const compressed = knee + headroom * Math.tanh((absX - knee) / headroom);
    const bounded = compressed > 1.0 ? 1.0 : compressed;
    return x < 0 ? -bounded : bounded;
  }

  /**
   * W272-aligned zero-allocation in-place half-cosine gain ramp across `frameCount`
   * interleaved frames from `startGain` to `endGain`. Non-finite samples are sanitized to 0.0.
   *
   * @param {Float32Array|number[]} samples
   * @param {number} [channels=1]
   * @param {number} [frameCount]
   * @param {number} [startGain=1.0]
   * @param {number} [endGain=0.0]
   * @returns {Float32Array|number[]}
   */
  function applyFadeRamp(samples, channels = 1, frameCount, startGain = 1.0, endGain = 0.0) {
    if (!samples || typeof samples.length !== 'number') {
      throw new TypeError('applyFadeRamp requires a Float32Array or array-like sample buffer');
    }
    if (!Number.isInteger(channels) || channels <= 0) {
      throw new RangeError(`channels must be a positive integer, got ${channels}`);
    }
    const maxFrames = Math.floor(samples.length / channels);
    let n;
    if (frameCount === undefined || frameCount === null) {
      n = maxFrames;
    } else {
      if (typeof frameCount !== 'number' || !Number.isFinite(frameCount) || frameCount < 0) {
        throw new RangeError(`frameCount must be a non-negative finite number, got ${frameCount}`);
      }
      n = Math.floor(frameCount);
      if (n > maxFrames) n = maxFrames;
    }
    if (n <= 0) {
      return samples;
    }

    const g0 = typeof startGain === 'number' && Number.isFinite(startGain) ? startGain : 0.0;
    const g1 = typeof endGain === 'number' && Number.isFinite(endGain) ? endGain : 0.0;

    if (n === 1) {
      for (let c = 0; c < channels; c++) {
        const s = samples[c];
        samples[c] = g1 === 0.0 || s - s !== 0 ? 0.0 : s * g1;
      }
      return samples;
    }

    const gDiff = g1 - g0;
    const halfGDiff = 0.5 * gDiff;
    const lastIdx = n - 1;
    const step = Math.PI / lastIdx;
    const halfStep = 0.5 * step;
    const s = Math.sin(halfStep);
    const k = 4.0 * s * s;

    if (channels === 2) {
      // Exact start endpoint (f === 0 -> g0)
      if (g0 === 0.0) {
        samples[0] = 0.0;
        samples[1] = 0.0;
      } else {
        const s0 = samples[0];
        const s1 = samples[1];
        samples[0] = s0 - s0 !== 0 ? 0.0 : s0 * g0;
        samples[1] = s1 - s1 !== 0 ? 0.0 : s1 * g0;
      }
      // Interior frames (1 .. lastIdx - 1) using second-order cosine recurrence
      // c_{f+1} - 2*cos(step)*c_f + c_{f-1} = 0  <=>  c += delta; delta -= k * c
      // with periodic resync every 4096 frames for arbitrarily long ramps (< 1e-6 error).
      if (lastIdx > 1) {
        let c = 1.0 - 0.5 * k;
        let delta = -0.5 * k - k * c;
        for (let f = 1; f < lastIdx; f++) {
          const gain = g0 + halfGDiff * (1.0 - c);
          const base = f * 2;
          if (gain === 0.0) {
            samples[base] = 0.0;
            samples[base + 1] = 0.0;
          } else {
            const s0 = samples[base];
            const s1 = samples[base + 1];
            samples[base] = s0 - s0 !== 0 ? 0.0 : s0 * gain;
            samples[base + 1] = s1 - s1 !== 0 ? 0.0 : s1 * gain;
          }
          if ((f & COS_RECURRENCE_RESYNC_MASK) === 0 && f + 1 < lastIdx) {
            const nextTheta = (f + 1) * step;
            c = Math.cos(nextTheta);
            delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
          } else {
            c += delta;
            delta -= k * c;
          }
        }
      }
      // Exact end endpoint (f === lastIdx -> g1)
      const endBase = lastIdx * 2;
      if (g1 === 0.0) {
        samples[endBase] = 0.0;
        samples[endBase + 1] = 0.0;
      } else {
        const s0 = samples[endBase];
        const s1 = samples[endBase + 1];
        samples[endBase] = s0 - s0 !== 0 ? 0.0 : s0 * g1;
        samples[endBase + 1] = s1 - s1 !== 0 ? 0.0 : s1 * g1;
      }
    } else if (channels === 1) {
      const startSample = samples[0];
      samples[0] = g0 === 0.0 || startSample - startSample !== 0 ? 0.0 : startSample * g0;
      if (lastIdx > 1) {
        let c = 1.0 - 0.5 * k;
        let delta = -0.5 * k - k * c;
        for (let f = 1; f < lastIdx; f++) {
          const gain = g0 + halfGDiff * (1.0 - c);
          const sv = samples[f];
          samples[f] = gain === 0.0 || sv - sv !== 0 ? 0.0 : sv * gain;
          if ((f & COS_RECURRENCE_RESYNC_MASK) === 0 && f + 1 < lastIdx) {
            const nextTheta = (f + 1) * step;
            c = Math.cos(nextTheta);
            delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
          } else {
            c += delta;
            delta -= k * c;
          }
        }
      }
      const endSample = samples[lastIdx];
      samples[lastIdx] = g1 === 0.0 || endSample - endSample !== 0 ? 0.0 : endSample * g1;
    } else {
      if (g0 === 0.0) {
        for (let ch = 0; ch < channels; ch++) {
          samples[ch] = 0.0;
        }
      } else {
        for (let ch = 0; ch < channels; ch++) {
          const sv = samples[ch];
          samples[ch] = sv - sv !== 0 ? 0.0 : sv * g0;
        }
      }
      if (lastIdx > 1) {
        let c = 1.0 - 0.5 * k;
        let delta = -0.5 * k - k * c;
        for (let f = 1; f < lastIdx; f++) {
          const gain = g0 + halfGDiff * (1.0 - c);
          const base = f * channels;
          if (gain === 0.0) {
            for (let ch = 0; ch < channels; ch++) {
              samples[base + ch] = 0.0;
            }
          } else {
            for (let ch = 0; ch < channels; ch++) {
              const sv = samples[base + ch];
              samples[base + ch] = sv - sv !== 0 ? 0.0 : sv * gain;
            }
          }
          if ((f & COS_RECURRENCE_RESYNC_MASK) === 0 && f + 1 < lastIdx) {
            const nextTheta = (f + 1) * step;
            c = Math.cos(nextTheta);
            delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
          } else {
            c += delta;
            delta -= k * c;
          }
        }
      }
      const endBase = lastIdx * channels;
      if (g1 === 0.0) {
        for (let ch = 0; ch < channels; ch++) {
          samples[endBase + ch] = 0.0;
        }
      } else {
        for (let ch = 0; ch < channels; ch++) {
          const sv = samples[endBase + ch];
          samples[endBase + ch] = sv - sv !== 0 ? 0.0 : sv * g1;
        }
      }
    }

    return samples;
  }

  /**
   * W272-aligned zero-allocation in-place non-finite sanitizer & smooth `tanh` soft-limiter.
   * Pass a reusable `outStats` object to avoid heap allocations on hot paths.
   *
   * @param {Float32Array|number[]} samples
   * @param {number} [count]
   * @param {number|object} [kneeOrOptions=DEFAULT_SOFT_CLIP_KNEE]
   * @param {{ softClippedSamples: number, nonFiniteSanitized: number }} [outStats]
   * @returns {{ softClippedSamples: number, nonFiniteSanitized: number }}
   */
  function sanitizeAndSoftLimitBuffer(
    samples,
    count,
    kneeOrOptions = DEFAULT_SOFT_CLIP_KNEE,
    outStats = null
  ) {
    if (!samples || typeof samples.length !== 'number') {
      throw new TypeError('sanitizeAndSoftLimitBuffer requires a Float32Array or array-like buffer');
    }

    let knee = DEFAULT_SOFT_CLIP_KNEE;
    let statsTarget = outStats;
    if (typeof kneeOrOptions === 'object' && kneeOrOptions !== null) {
      if (kneeOrOptions.knee !== undefined) {
        knee = kneeOrOptions.knee;
      } else if (kneeOrOptions.softLimitKnee !== undefined) {
        knee = kneeOrOptions.softLimitKnee;
      } else if (kneeOrOptions.limiterKnee !== undefined) {
        knee = kneeOrOptions.limiterKnee;
      }
      if (!statsTarget && kneeOrOptions.stats && typeof kneeOrOptions.stats === 'object') {
        statsTarget = kneeOrOptions.stats;
      } else if (
        !statsTarget &&
        kneeOrOptions.knee === undefined &&
        kneeOrOptions.softLimitKnee === undefined &&
        kneeOrOptions.limiterKnee === undefined
      ) {
        statsTarget = kneeOrOptions;
      }
    } else if (kneeOrOptions !== undefined) {
      knee = kneeOrOptions;
    }

    if (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0) {
      throw new RangeError(`knee must be in [0, 1), got ${knee}`);
    }

    let n;
    if (count === undefined || count === null) {
      n = samples.length;
    } else {
      if (typeof count !== 'number' || !Number.isFinite(count) || count < 0) {
        throw new RangeError(`count must be a non-negative finite number, got ${count}`);
      }
      n = Math.floor(count);
      if (n > samples.length) n = samples.length;
    }

    const negKnee = -knee;
    const headroom = 1.0 - knee;
    let softClippedSamples = 0;
    let nonFiniteSanitized = 0;

    for (let i = 0; i < n; i++) {
      const s = samples[i];
      if (s - s !== 0) {
        samples[i] = 0.0;
        nonFiniteSanitized += 1;
      } else if (s > knee) {
        const compressed = knee + headroom * Math.tanh((s - knee) / headroom);
        samples[i] = compressed > 1.0 ? 1.0 : compressed;
        softClippedSamples += 1;
      } else if (s < negKnee) {
        const compressed = knee + headroom * Math.tanh((-s - knee) / headroom);
        samples[i] = compressed > 1.0 ? -1.0 : -compressed;
        softClippedSamples += 1;
      }
    }

    if (statsTarget) {
      statsTarget.softClippedSamples = (statsTarget.softClippedSamples || 0) + softClippedSamples;
      statsTarget.nonFiniteSanitized = (statsTarget.nonFiniteSanitized || 0) + nonFiniteSanitized;
      return statsTarget;
    }

    return { softClippedSamples, nonFiniteSanitized };
  }

  function validateDownmixGain(value, fallback, name) {
    if (value === undefined) {
      return fallback;
    }
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative finite number, got ${value}`);
    }
    return value;
  }

  /**
   * ITU-R BS.775 multi-channel (1, 2, 4, 5.1 [6ch], 7.1 [8ch]) to stereo downmix
   * with per-channel non-finite sanitization and smooth soft-clip peak limiting.
   *
   * @param {Float32Array|ArrayBuffer|SharedArrayBuffer|ArrayBufferView|number[]} samples
   * @param {number} channels Source channel count (1, 2, 4, 6, or 8).
   * @param {object} [options]
   * @returns {Float32Array} Interleaved stereo Float32Array.
   */
  function downmixToStereo(samples, channels, options = {}) {
    if (!Number.isInteger(channels) || !SUPPORTED_DOWNMIX_CHANNELS.includes(channels)) {
      throw new RangeError(
        `downmixToStereo channels must be one of ${SUPPORTED_DOWNMIX_CHANNELS.join(', ')}, got ${channels}`
      );
    }

    const opts = options && typeof options === 'object' ? options : {};
    const centerGain = validateDownmixGain(
      opts.centerGain !== undefined ? opts.centerGain : opts.centerWeight,
      ITU_BS775_CENTER_GAIN,
      'centerGain'
    );
    const surroundGain = validateDownmixGain(
      opts.surroundGain !== undefined ? opts.surroundGain : opts.surroundWeight,
      ITU_BS775_SURROUND_GAIN,
      'surroundGain'
    );
    const lfeGain = validateDownmixGain(
      opts.lfeGain !== undefined ? opts.lfeGain : opts.lfeWeight,
      ITU_BS775_LFE_GAIN,
      'lfeGain'
    );
    const softClip = opts.softClip !== undefined ? Boolean(opts.softClip) : true;
    const knee =
      opts.limiterKnee !== undefined
        ? opts.limiterKnee
        : opts.knee !== undefined
          ? opts.knee
          : DEFAULT_SOFT_CLIP_KNEE;
    if (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0) {
      throw new RangeError(`limiterKnee must be in [0, 1), got ${knee}`);
    }

    let src;
    let nanSanitized = 0;
    if (samples instanceof Float32Array && !isSharedArrayBuffer(samples.buffer)) {
      if (isDetachedArrayBuffer(samples.buffer)) {
        throw new TypeError('PCM TypedArray backing ArrayBuffer is detached');
      }
      src = samples;
    } else {
      _INTERNAL_EXTRACT_OPTS.scratch = opts.scratch || null;
      const extracted = extractSafePcmSamples(samples, _INTERNAL_EXTRACT_OPTS);
      _INTERNAL_EXTRACT_OPTS.scratch = null;
      if (!extracted.ok) {
        const errMsg = extracted.error;
        _INTERNAL_EXTRACT_RESULT.pcm = null;
        throw new TypeError(errMsg);
      }
      src = extracted.pcm;
      nanSanitized += extracted.sanitizedSamples;
      _INTERNAL_EXTRACT_RESULT.pcm = null;
    }

    const frameCount = Math.floor(src.length / channels);
    const outSampleCount = frameCount * 2;
    let out;
    if (opts.output instanceof Float32Array && opts.output.length >= outSampleCount) {
      out =
        opts.output.length === outSampleCount
          ? opts.output
          : opts.output.subarray(0, outSampleCount);
    } else if (opts.scratch && typeof opts.scratch.acquireFloat32 === 'function') {
      out = opts.scratch.acquireFloat32('downmix', outSampleCount);
    } else {
      out = new Float32Array(outSampleCount);
    }
    let clipped = 0;
    let peak = 0.0;

    for (let f = 0; f < frameCount; f++) {
      const base = f * channels;
      let rawL;
      let rawR;

      if (channels === 6) {
        // 5.1: FL, FR, FC, LFE, BL, BR
        let fl = src[base];
        let fr = src[base + 1];
        let fc = src[base + 2];
        let lfe = src[base + 3];
        let bl = src[base + 4];
        let br = src[base + 5];

        if (!Number.isFinite(fl)) { fl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fr)) { fr = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fc)) { fc = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(lfe)) { lfe = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(bl)) { bl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(br)) { br = 0.0; nanSanitized += 1; }

        rawL = fl + fc * centerGain + bl * surroundGain + lfe * lfeGain;
        rawR = fr + fc * centerGain + br * surroundGain + lfe * lfeGain;
      } else if (channels === 8) {
        // 7.1: FL, FR, FC, LFE, BL, BR, SL, SR
        let fl = src[base];
        let fr = src[base + 1];
        let fc = src[base + 2];
        let lfe = src[base + 3];
        let bl = src[base + 4];
        let br = src[base + 5];
        let sl = src[base + 6];
        let sr = src[base + 7];

        if (!Number.isFinite(fl)) { fl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fr)) { fr = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fc)) { fc = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(lfe)) { lfe = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(bl)) { bl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(br)) { br = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(sl)) { sl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(sr)) { sr = 0.0; nanSanitized += 1; }

        rawL = fl + fc * centerGain + (bl + sl) * surroundGain + lfe * lfeGain;
        rawR = fr + fc * centerGain + (br + sr) * surroundGain + lfe * lfeGain;
      } else if (channels === 4) {
        // Quad: FL, FR, BL, BR
        let fl = src[base];
        let fr = src[base + 1];
        let bl = src[base + 2];
        let br = src[base + 3];

        if (!Number.isFinite(fl)) { fl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fr)) { fr = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(bl)) { bl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(br)) { br = 0.0; nanSanitized += 1; }

        rawL = fl + bl * surroundGain;
        rawR = fr + br * surroundGain;
      } else if (channels === 2) {
        let fl = src[base];
        let fr = src[base + 1];
        if (!Number.isFinite(fl)) { fl = 0.0; nanSanitized += 1; }
        if (!Number.isFinite(fr)) { fr = 0.0; nanSanitized += 1; }
        rawL = fl;
        rawR = fr;
      } else {
        let m = src[base];
        if (!Number.isFinite(m)) { m = 0.0; nanSanitized += 1; }
        rawL = m;
        rawR = m;
      }

      if (!Number.isFinite(rawL)) { rawL = 0.0; nanSanitized += 1; }
      if (!Number.isFinite(rawR)) { rawR = 0.0; nanSanitized += 1; }

      let outL;
      let outR;
      if (softClip) {
        if (rawL > knee || rawL < -knee) clipped += 1;
        if (rawR > knee || rawR < -knee) clipped += 1;
        outL = softClipSample(rawL, knee);
        outR = softClipSample(rawR, knee);
      } else {
        if (rawL > 1.0 || rawL < -1.0) clipped += 1;
        if (rawR > 1.0 || rawR < -1.0) clipped += 1;
        outL = clampSample(rawL, -1.0, 1.0);
        outR = clampSample(rawR, -1.0, 1.0);
      }

      const absL = Math.abs(outL);
      const absR = Math.abs(outR);
      if (absL > peak) peak = absL;
      if (absR > peak) peak = absR;

      const outIdx = f * 2;
      out[outIdx] = outL;
      out[outIdx + 1] = outR;
    }

    if (opts.stats && typeof opts.stats === 'object') {
      const dmFrames = channels > 2 ? frameCount : 0;
      opts.stats.downmixedFrames = (opts.stats.downmixedFrames || 0) + dmFrames;
      opts.stats.clippedSamples = (opts.stats.clippedSamples || 0) + clipped;
      opts.stats.nanSanitizedSamples = (opts.stats.nanSanitizedSamples || 0) + nanSanitized;
      opts.stats.peakLevel = Math.max(opts.stats.peakLevel || 0.0, peak);
    }

    return out;
  }

  /**
   * Smooth linear PCM resampler with safeguards for 44.1 kHz <-> 48 kHz and
   * general Bink2 / XAudio2 sample-rate conversions.
   *
   * @param {Float32Array|ArrayBuffer|SharedArrayBuffer|ArrayBufferView|number[]} samples
   * @param {number} channels Interleaved channel count (1..8).
   * @param {number} srcSampleRate Source sample rate in Hz.
   * @param {number} dstSampleRate Destination sample rate in Hz.
   * @param {object} [options]
   * @returns {Float32Array}
   */
  function resampleLinear(samples, channels, srcSampleRate, dstSampleRate, options = {}) {
    if (!Number.isInteger(channels) || channels < 1 || channels > MAX_CHANNELS) {
      throw new RangeError(
        `resampleLinear channels must be an integer in [1, ${MAX_CHANNELS}], got ${channels}`
      );
    }
    if (
      typeof srcSampleRate !== 'number' ||
      !Number.isFinite(srcSampleRate) ||
      srcSampleRate < MIN_SAMPLE_RATE ||
      srcSampleRate > MAX_SAMPLE_RATE
    ) {
      throw new RangeError(
        `srcSampleRate must be in [${MIN_SAMPLE_RATE}, ${MAX_SAMPLE_RATE}], got ${srcSampleRate}`
      );
    }
    if (
      typeof dstSampleRate !== 'number' ||
      !Number.isFinite(dstSampleRate) ||
      dstSampleRate < MIN_SAMPLE_RATE ||
      dstSampleRate > MAX_SAMPLE_RATE
    ) {
      throw new RangeError(
        `dstSampleRate must be in [${MIN_SAMPLE_RATE}, ${MAX_SAMPLE_RATE}], got ${dstSampleRate}`
      );
    }

    const ratio = dstSampleRate / srcSampleRate;
    if (
      !Number.isFinite(ratio) ||
      ratio <= 0 ||
      ratio > MAX_RESAMPLE_RATIO ||
      1.0 / ratio > MAX_RESAMPLE_RATIO
    ) {
      throw new RangeError(
        `Resample ratio (${srcSampleRate} -> ${dstSampleRate}) exceeds max guard (${MAX_RESAMPLE_RATIO}x)`
      );
    }

    const opts = options && typeof options === 'object' ? options : {};
    const maxOutputFrames =
      typeof opts.maxOutputFrames === 'number' &&
      Number.isFinite(opts.maxOutputFrames) &&
      opts.maxOutputFrames > 0
        ? Math.floor(opts.maxOutputFrames)
        : DEFAULT_MAX_RESAMPLE_FRAMES;
    const softClip = Boolean(opts.softClip);
    const knee =
      opts.limiterKnee !== undefined
        ? opts.limiterKnee
        : opts.knee !== undefined
          ? opts.knee
          : DEFAULT_SOFT_CLIP_KNEE;
    if (softClip && (typeof knee !== 'number' || !Number.isFinite(knee) || knee < 0 || knee >= 1.0)) {
      throw new RangeError(`limiterKnee must be in [0, 1), got ${knee}`);
    }

    let src;
    let nanSanitized = 0;
    if (samples instanceof Float32Array && !isSharedArrayBuffer(samples.buffer)) {
      if (isDetachedArrayBuffer(samples.buffer)) {
        throw new TypeError('PCM TypedArray backing ArrayBuffer is detached');
      }
      src = samples;
    } else {
      _INTERNAL_EXTRACT_OPTS.scratch = opts.scratch || null;
      const extracted = extractSafePcmSamples(samples, _INTERNAL_EXTRACT_OPTS);
      _INTERNAL_EXTRACT_OPTS.scratch = null;
      if (!extracted.ok) {
        const errMsg = extracted.error;
        _INTERNAL_EXTRACT_RESULT.pcm = null;
        throw new TypeError(errMsg);
      }
      src = extracted.pcm;
      nanSanitized += extracted.sanitizedSamples;
      _INTERNAL_EXTRACT_RESULT.pcm = null;
    }

    const maxSrcFrames = Math.floor(src.length / channels);
    const srcFrames =
      opts.srcFrames !== undefined
        ? Math.min(sanitizeNonNegInt(opts.srcFrames, 0), maxSrcFrames)
        : maxSrcFrames;
    if (srcFrames <= 0) {
      return EMPTY_FLOAT32;
    }

    const state = opts.state && typeof opts.state === 'object' ? opts.state : null;
    const EPS = 1e-12;
    let clipped = 0;

    const startPhase =
      state && typeof state.phase === 'number' && Number.isFinite(state.phase) && state.phase >= 0
        ? state.phase
        : 0.0;

    if (srcSampleRate === dstSampleRate && startPhase < EPS) {
      if (srcFrames > maxOutputFrames) {
        throw new RangeError(
          `Resampled output (${srcFrames} frames) exceeds maxOutputFrames (${maxOutputFrames})`
        );
      }
      const totalSamples = srcFrames * channels;
      let out;
      if (opts.output instanceof Float32Array && opts.output.length >= totalSamples) {
        out =
          opts.output.length === totalSamples
            ? opts.output
            : opts.output.subarray(0, totalSamples);
      } else if (opts.scratch && typeof opts.scratch.acquireFloat32 === 'function') {
        out = opts.scratch.acquireFloat32('resample', totalSamples);
      } else {
        out = new Float32Array(totalSamples);
      }
      for (let i = 0; i < totalSamples; i++) {
        let v = src[i];
        if (!Number.isFinite(v)) {
          v = 0.0;
          nanSanitized += 1;
        }
        if (softClip) {
          if (v > knee || v < -knee) clipped += 1;
          v = softClipSample(v, knee);
        }
        out[i] = v;
      }
      if (state) {
        state.phase = 0.0;
      }
      if (opts.stats && typeof opts.stats === 'object') {
        opts.stats.nanSanitizedSamples = (opts.stats.nanSanitizedSamples || 0) + nanSanitized;
        opts.stats.clippedSamples = (opts.stats.clippedSamples || 0) + clipped;
      }
      return out;
    }

    const step = srcSampleRate / dstSampleRate;
    if (srcFrames - startPhase <= EPS) {
      if (state) {
        const rem = startPhase - srcFrames;
        state.phase = rem > EPS ? rem : 0.0;
      }
      return EMPTY_FLOAT32;
    }

    const outFrames = Math.floor((srcFrames - startPhase - EPS) / step) + 1;
    if (outFrames > maxOutputFrames) {
      throw new RangeError(
        `Resampled output (${outFrames} frames) exceeds maxOutputFrames (${maxOutputFrames})`
      );
    }

    const outSampleCount = outFrames * channels;
    let out;
    if (opts.output instanceof Float32Array && opts.output.length >= outSampleCount) {
      out =
        opts.output.length === outSampleCount
          ? opts.output
          : opts.output.subarray(0, outSampleCount);
    } else if (opts.scratch && typeof opts.scratch.acquireFloat32 === 'function') {
      out = opts.scratch.acquireFloat32('resample', outSampleCount);
    } else {
      out = new Float32Array(outSampleCount);
    }
    const lastFrame = srcFrames - 1;

    for (let f = 0; f < outFrames; f++) {
      const pos = startPhase + f * step;
      let i0 = Math.floor(pos);
      if (i0 < 0) i0 = 0;
      else if (i0 > lastFrame) i0 = lastFrame;
      const frac = pos - i0;
      const i1 = i0 < lastFrame ? i0 + 1 : lastFrame;
      const base0 = i0 * channels;
      const base1 = i1 * channels;
      const outBase = f * channels;

      for (let c = 0; c < channels; c++) {
        let s0 = src[base0 + c];
        if (!Number.isFinite(s0)) {
          s0 = 0.0;
          nanSanitized += 1;
        }
        let val;
        if (frac <= 0 || i1 === i0) {
          val = s0;
        } else {
          let s1 = src[base1 + c];
          if (!Number.isFinite(s1)) {
            s1 = 0.0;
            nanSanitized += 1;
          }
          val = s0 + (s1 - s0) * frac;
          if (!Number.isFinite(val)) {
            val = 0.0;
            nanSanitized += 1;
          }
        }
        if (softClip) {
          if (val > knee || val < -knee) clipped += 1;
          val = softClipSample(val, knee);
        }
        out[outBase + c] = val;
      }
    }

    if (state) {
      let nextPhase = startPhase + outFrames * step - srcFrames;
      if (!Number.isFinite(nextPhase) || nextPhase < EPS) {
        nextPhase = 0.0;
      }
      state.phase = nextPhase;
    }

    if (opts.stats && typeof opts.stats === 'object') {
      opts.stats.resampledFrames = (opts.stats.resampledFrames || 0) + outFrames;
      opts.stats.nanSanitizedSamples = (opts.stats.nanSanitizedSamples || 0) + nanSanitized;
      opts.stats.clippedSamples = (opts.stats.clippedSamples || 0) + clipped;
    }

    return out;
  }

  /**
   * Builds a strictly sanitized `audio-stats` telemetry snapshot containing only
   * finite numeric counters, booleans, and whitelisted enum strings.
   * Strips raw buffers, SharedArrayBuffers, host paths, and unknown fields.
   *
   * @param {AudioMixerWorker|object|null} source
   * @param {{ now?: () => number, timestamp?: number }} [options]
   */
  function buildAudioTelemetrySnapshot(source, options = {}) {
    let raw = null;
    if (source && typeof source === 'object') {
      if (typeof source.getStats === 'function') {
        try {
          raw = source.getStats();
        } catch (_err) {
          raw = null;
        }
      } else {
        raw = source;
      }
    }

    const s = raw && typeof raw === 'object' ? raw : {};
    const nowFn = typeof options.now === 'function' ? options.now : () => Date.now();
    const rawTs = options.timestamp !== undefined ? options.timestamp : nowFn();
    const timestamp =
      typeof rawTs === 'number' && Number.isFinite(rawTs) && rawTs >= 0
        ? Math.floor(rawTs)
        : 0;

    const initialized = Boolean(s.initialized);
    const sampleRate = sanitizeSampleRate(s.sampleRate);
    const channels = sanitizePositiveInt(s.channels, DEFAULT_CHANNELS, 1, MAX_CHANNELS);
    const capacityFrames = sanitizeNonNegInt(s.capacityFrames, 0);
    const availableFrames = Math.min(
      sanitizeNonNegInt(s.availableFrames, 0),
      capacityFrames > 0 ? capacityFrames : sanitizeNonNegInt(s.availableFrames, 0)
    );
    const freeFrames =
      capacityFrames > 0
        ? Math.max(0, capacityFrames - availableFrames)
        : sanitizeNonNegInt(s.freeFrames, 0);
    const volume = clampVolume(s.volume !== undefined ? s.volume : 1.0);
    const muted = Boolean(s.muted);
    const effectiveGain = muted ? 0.0 : volume;
    const underruns = sanitizeNonNegInt(
      s.underruns !== undefined ? s.underruns : s.underrunCount,
      0
    );
    const underrunCount = sanitizeNonNegInt(
      s.underrunCount !== undefined ? s.underrunCount : underruns,
      underruns
    );
    const underrunFrames = sanitizeNonNegInt(s.underrunFrames, 0);
    const fadeOutCount = sanitizeNonNegInt(s.fadeOutCount, 0);
    const fadeInCount = sanitizeNonNegInt(s.fadeInCount, 0);
    const underrunFadeCount = sanitizeNonNegInt(
      s.underrunFadeCount !== undefined ? s.underrunFadeCount : fadeOutCount,
      fadeOutCount
    );
    const recoveryFadeCount = sanitizeNonNegInt(
      s.recoveryFadeCount !== undefined ? s.recoveryFadeCount : fadeInCount,
      fadeInCount
    );
    const rawPeakFill = sanitizeNonNegInt(s.peakFillFrames, 0);
    const peakFillFrames =
      capacityFrames > 0 ? Math.min(rawPeakFill, capacityFrames) : rawPeakFill;
    const rawMinFill = sanitizeNonNegInt(s.minFillFrames, 0);
    const minFillFrames =
      capacityFrames > 0 ? Math.min(rawMinFill, capacityFrames) : rawMinFill;
    const rawAvgFillRatio =
      typeof s.avgFillRatio === 'number' && Number.isFinite(s.avgFillRatio)
        ? s.avgFillRatio
        : 0.0;
    const avgFillRatio =
      rawAvgFillRatio < 0 ? 0.0 : rawAvgFillRatio > 1 ? 1.0 : rawAvgFillRatio;
    const overflows = sanitizeNonNegInt(s.overflows, 0);
    const overruns = sanitizeNonNegInt(
      s.overruns !== undefined ? s.overruns : overflows,
      overflows
    );
    const droppedFrames = sanitizeNonNegInt(s.droppedFrames, 0);
    const totalPushedFrames = sanitizeNonNegInt(s.totalPushedFrames, 0);
    const totalPulledFrames = sanitizeNonNegInt(s.totalPulledFrames, 0);
    const framesDrained = sanitizeNonNegInt(s.framesDrained, 0);
    const drainTicks = sanitizeNonNegInt(s.drainTicks, 0);
    const rejectedPackets = sanitizeNonNegInt(s.rejectedPackets, 0);
    const sabCopiedPackets = sanitizeNonNegInt(s.sabCopiedPackets, 0);
    const sanitizedSamples = sanitizeNonNegInt(
      s.sanitizedSamples !== undefined
        ? s.sanitizedSamples
        : s.nanSanitizedSamples !== undefined
          ? s.nanSanitizedSamples
          : s.sanitizedNonFiniteSamples,
      0
    );
    const nanSanitizedSamples = sanitizeNonNegInt(
      s.nanSanitizedSamples !== undefined ? s.nanSanitizedSamples : sanitizedSamples,
      sanitizedSamples
    );
    const sanitizedNonFiniteSamples = sanitizeNonNegInt(
      s.sanitizedNonFiniteSamples !== undefined
        ? s.sanitizedNonFiniteSamples
        : sanitizedSamples,
      sanitizedSamples
    );
    const downmixedFrames = sanitizeNonNegInt(s.downmixedFrames, 0);
    const resampledFrames = sanitizeNonNegInt(s.resampledFrames, 0);
    const clippedSamples = sanitizeNonNegInt(s.clippedSamples, 0);
    const softClippedSamples = sanitizeNonNegInt(s.softClippedSamples, 0);
    const targetLatencyMs = clampTargetLatencyMs(s.targetLatencyMs, DEFAULT_TARGET_LATENCY_MS);
    const zeroAllocDrains = sanitizeNonNegInt(s.zeroAllocDrains, 0);
    const activeVoicePeak = sanitizeNonNegInt(
      s.activeVoicePeak !== undefined ? s.activeVoicePeak : s.activeVoices,
      0
    );
    const packetsReceived = sanitizeNonNegInt(s.packetsReceived, 0);
    const framesDecoded = sanitizeNonNegInt(
      s.framesDecoded !== undefined ? s.framesDecoded : totalPushedFrames,
      0
    );
    const droppedPackets = sanitizeNonNegInt(s.droppedPackets, 0);
    const underrunRecoveries = sanitizeNonNegInt(s.underrunRecoveries, 0);
    const rawBufferFillPct =
      typeof s.bufferFillPct === 'number' && Number.isFinite(s.bufferFillPct)
        ? s.bufferFillPct
        : capacityFrames > 0
          ? (availableFrames / capacityFrames) * 100
          : 0.0;
    const bufferFillPct =
      rawBufferFillPct < 0 ? 0.0 : rawBufferFillPct > 100 ? 100.0 : rawBufferFillPct;
    const scratchAllocations = sanitizeNonNegInt(s.scratchAllocations, 0);
    const scratchReuses = sanitizeNonNegInt(s.scratchReuses, 0);
    const scratchBufferReuses = sanitizeNonNegInt(
      s.scratchBufferReuses !== undefined ? s.scratchBufferReuses : scratchReuses,
      scratchReuses
    );
    const fastCosineRamps = sanitizeNonNegInt(s.fastCosineRamps, 0);
    const overflowPolicy = normalizeOverflowPolicy(s.overflowPolicy, DEFAULT_OVERFLOW_POLICY);
    const maxFillRatio = sanitizeFillRatio(s.maxFillRatio, DEFAULT_MAX_FILL_RATIO);
    const outputState = normalizeOutputState(s.outputState, OUTPUT_STATES.HEADLESS_DRAIN);
    const guestRingAttached = Boolean(s.guestRingAttached);
    const guestFramesDrained = sanitizeNonNegInt(s.guestFramesDrained, 0);
    const guestUnderruns = sanitizeNonNegInt(s.guestUnderruns, 0);
    const guestOverruns = sanitizeNonNegInt(s.guestOverruns, 0);
    const rawGuestPeakLevel =
      typeof s.guestPeakLevel === 'number' && Number.isFinite(s.guestPeakLevel)
        ? s.guestPeakLevel
        : 0.0;
    const guestPeakLevel =
      rawGuestPeakLevel < 0 ? 0.0 : rawGuestPeakLevel > 1 ? 1.0 : rawGuestPeakLevel;

    return {
      type: 'audio-stats',
      timestamp,
      stats: {
        initialized,
        outputState,
        sampleRate,
        channels,
        capacityFrames,
        availableFrames,
        freeFrames,
        bufferFillPct,
        volume,
        muted,
        effectiveGain,
        framesDrained,
        packetsReceived,
        framesDecoded,
        droppedPackets,
        underruns,
        underrunCount,
        underrunFrames,
        underrunRecoveries,
        fadeOutCount,
        fadeInCount,
        underrunFadeCount,
        recoveryFadeCount,
        peakFillFrames,
        minFillFrames,
        avgFillRatio,
        overruns,
        overflows,
        overflowPolicy,
        maxFillRatio,
        targetLatencyMs,
        droppedFrames,
        totalPushedFrames,
        totalPulledFrames,
        drainTicks,
        rejectedPackets,
        sabCopiedPackets,
        sanitizedSamples,
        nanSanitizedSamples,
        sanitizedNonFiniteSamples,
        downmixedFrames,
        resampledFrames,
        clippedSamples,
        softClippedSamples,
        activeVoicePeak,
        scratchAllocations,
        scratchReuses,
        scratchBufferReuses,
        fastCosineRamps,
        zeroAllocDrains,
        guestRingAttached,
        guestFramesDrained,
        guestUnderruns,
        guestOverruns,
        guestPeakLevel,
      },
    };
  }

  function isPowerOfTwo(value) {
    return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
  }

  function sanitizeAlignedByteOffset(value, name) {
    let num = value;
    if (typeof num === 'bigint') {
      if (num < 0n || num > 9007199254740991n) {
        throw new RangeError(`${name} byte offset is out of safe range`);
      }
      num = Number(num);
    }
    if (typeof num !== 'number' || !Number.isFinite(num) || !Number.isInteger(num) || num < 0) {
      throw new RangeError(`${name} must be a non-negative integer byte offset, got ${value}`);
    }
    if (num % 4 !== 0) {
      throw new RangeError(`${name} (${num}) must be 4-byte aligned`);
    }
    return num;
  }

  function resolveGuestRingSab(descriptor) {
    if (!descriptor || typeof descriptor !== 'object' || Array.isArray(descriptor)) {
      throw new TypeError('Guest PCM ring descriptor must be an object');
    }
    const candidate =
      descriptor.sab !== undefined
        ? descriptor.sab
        : descriptor.buffer !== undefined
          ? descriptor.buffer
          : descriptor.memory && typeof descriptor.memory === 'object'
            ? descriptor.memory.buffer
            : undefined;
    if (!isSharedArrayBuffer(candidate)) {
      throw new TypeError('Guest PCM ring requires a SharedArrayBuffer');
    }
    return candidate;
  }

  /**
   * Lock-free SPSC reader for the runtime's stereo interleaved float32 PCM ring
   * (`Playgta6AudioRing` in `runtime/os/audio_downmix.h`).
   *
   * - Free-running uint32 sample cursors wrapping at 2^32.
   * - Power-of-two float32 sample capacity (slot = index & (capacity - 1)).
   * - Zero per-quantum heap allocation during `readInto`.
   * - Non-finite sanitization, smooth soft-clipping into [-1, 1], underrun
   *   silence fill, and overrun drop accounting.
   */
  class GuestPcmRingReader {
    constructor(descriptor) {
      const sab = resolveGuestRingSab(descriptor);

      const channels =
        descriptor.channels !== undefined ? descriptor.channels : GUEST_PCM_RING_CHANNELS;
      if (!Number.isInteger(channels) || channels !== GUEST_PCM_RING_CHANNELS) {
        throw new RangeError(
          `Guest PCM ring requires stereo (${GUEST_PCM_RING_CHANNELS}) channels, got ${channels}`
        );
      }

      const rawSampleRate =
        descriptor.sampleRate !== undefined
          ? descriptor.sampleRate
          : descriptor.sample_rate !== undefined
            ? descriptor.sample_rate
            : GUEST_PCM_RING_DEFAULT_SAMPLE_RATE;
      if (
        typeof rawSampleRate !== 'number' ||
        !Number.isFinite(rawSampleRate) ||
        rawSampleRate < MIN_SAMPLE_RATE ||
        rawSampleRate > MAX_SAMPLE_RATE
      ) {
        throw new RangeError(
          `Guest PCM ring sampleRate must be in [${MIN_SAMPLE_RATE}, ${MAX_SAMPLE_RATE}], got ${rawSampleRate}`
        );
      }
      const sampleRate = Math.floor(rawSampleRate);

      const rawBase =
        descriptor.dataByteOffset !== undefined
          ? descriptor.dataByteOffset
          : descriptor.base !== undefined
            ? descriptor.base
            : descriptor.dataOffset;
      const dataByteOffset = sanitizeAlignedByteOffset(rawBase, 'dataByteOffset');

      let capacitySamples;
      if (descriptor.capacitySamples !== undefined) {
        capacitySamples = descriptor.capacitySamples;
      } else if (descriptor.capacity !== undefined) {
        capacitySamples = descriptor.capacity;
      } else if (descriptor.capacityFrames !== undefined) {
        if (!Number.isInteger(descriptor.capacityFrames) || descriptor.capacityFrames <= 0) {
          throw new RangeError(
            `Guest PCM ring capacityFrames must be a positive integer, got ${descriptor.capacityFrames}`
          );
        }
        capacitySamples = descriptor.capacityFrames * channels;
      } else {
        throw new RangeError(
          'Guest PCM ring requires capacity, capacitySamples, or capacityFrames'
        );
      }

      if (
        !isPowerOfTwo(capacitySamples) ||
        capacitySamples < GUEST_PCM_RING_MIN_CAPACITY_SAMPLES ||
        capacitySamples > GUEST_PCM_RING_MAX_CAPACITY_SAMPLES
      ) {
        throw new RangeError(
          `Guest PCM ring capacitySamples must be a power of two in [${GUEST_PCM_RING_MIN_CAPACITY_SAMPLES}, ${GUEST_PCM_RING_MAX_CAPACITY_SAMPLES}], got ${capacitySamples}`
        );
      }

      const dataByteLength = capacitySamples * Float32Array.BYTES_PER_ELEMENT;
      if (dataByteOffset + dataByteLength > sab.byteLength) {
        throw new RangeError(
          `Guest PCM ring data region [${dataByteOffset}, ${dataByteOffset + dataByteLength}) exceeds SharedArrayBuffer byteLength (${sab.byteLength})`
        );
      }

      const rawReadOffset =
        descriptor.readIndexByteOffset !== undefined
          ? descriptor.readIndexByteOffset
          : descriptor.read_index_ptr !== undefined
            ? descriptor.read_index_ptr
            : descriptor.readIndexPtr !== undefined
              ? descriptor.readIndexPtr
              : descriptor.readIndexOffset;
      const rawWriteOffset =
        descriptor.writeIndexByteOffset !== undefined
          ? descriptor.writeIndexByteOffset
          : descriptor.write_index_ptr !== undefined
            ? descriptor.write_index_ptr
            : descriptor.writeIndexPtr !== undefined
              ? descriptor.writeIndexPtr
              : descriptor.writeIndexOffset;

      const readIndexByteOffset = sanitizeAlignedByteOffset(rawReadOffset, 'readIndexByteOffset');
      const writeIndexByteOffset = sanitizeAlignedByteOffset(rawWriteOffset, 'writeIndexByteOffset');

      if (readIndexByteOffset + Int32Array.BYTES_PER_ELEMENT > sab.byteLength) {
        throw new RangeError(
          `Guest PCM ring readIndexByteOffset (${readIndexByteOffset}) exceeds SharedArrayBuffer byteLength (${sab.byteLength})`
        );
      }
      if (writeIndexByteOffset + Int32Array.BYTES_PER_ELEMENT > sab.byteLength) {
        throw new RangeError(
          `Guest PCM ring writeIndexByteOffset (${writeIndexByteOffset}) exceeds SharedArrayBuffer byteLength (${sab.byteLength})`
        );
      }
      if (readIndexByteOffset === writeIndexByteOffset) {
        throw new RangeError(
          'Guest PCM ring readIndexByteOffset and writeIndexByteOffset must not alias'
        );
      }
      const dataEnd = dataByteOffset + dataByteLength;
      if (readIndexByteOffset >= dataByteOffset && readIndexByteOffset < dataEnd) {
        throw new RangeError('Guest PCM ring readIndexByteOffset overlaps PCM sample data');
      }
      if (writeIndexByteOffset >= dataByteOffset && writeIndexByteOffset < dataEnd) {
        throw new RangeError('Guest PCM ring writeIndexByteOffset overlaps PCM sample data');
      }

      this._sab = sab;
      this._dataByteOffset = dataByteOffset;
      this._readIndexByteOffset = readIndexByteOffset;
      this._writeIndexByteOffset = writeIndexByteOffset;
      this._capacitySamples = capacitySamples;
      this._capacityFrames = capacitySamples >>> 1;
      this._mask = capacitySamples - 1;
      this._channels = channels;
      this._sampleRate = sampleRate;

      // Pre-allocated views: zero per-quantum allocations during steady-state reads.
      this._data = new Float32Array(sab, dataByteOffset, capacitySamples);
      this._readCtrl = new Int32Array(sab, readIndexByteOffset, 1);
      this._writeCtrl = new Int32Array(sab, writeIndexByteOffset, 1);
      this._lastStereoFrame = new Float32Array(GUEST_PCM_RING_CHANNELS);
      this._trimXfadeSamples = new Float32Array(
        GUEST_TRIM_CROSSFADE_FRAMES * GUEST_PCM_RING_CHANNELS
      );
      this._pendingTrimXfadeFrames = 0;
      this._hasActiveAudio = false;
      this._inUnderrun = false;
      this._fadeInPos = 0;
      this._fadeInTotal = 0;
      this._underrunFadeCount = 0;
      this._recoveryFadeCount = 0;
      this._fastCosineRamps = 0;
      this._reusableResult = {
        framesRead: 0,
        underrun: false,
        underrunFrames: 0,
        overrun: false,
        overrunDroppedFrames: 0,
        clippedSamples: 0,
        softClippedSamples: 0,
        sanitizedSamples: 0,
        underrunFade: false,
        recoveryFade: false,
        peakLevel: 0.0,
      };
    }

    get underrunFadeCount() {
      return this._underrunFadeCount;
    }

    get recoveryFadeCount() {
      return this._recoveryFadeCount;
    }

    get fastCosineRamps() {
      return this._fastCosineRamps;
    }

    resetFadeState() {
      this._lastStereoFrame[0] = 0.0;
      this._lastStereoFrame[1] = 0.0;
      this._trimXfadeSamples.fill(0);
      this._pendingTrimXfadeFrames = 0;
      this._hasActiveAudio = false;
      this._inUnderrun = false;
      this._fadeInPos = 0;
      this._fadeInTotal = 0;
      this._underrunFadeCount = 0;
      this._recoveryFadeCount = 0;
      this._fastCosineRamps = 0;
    }

    get sab() {
      return this._sab;
    }

    get dataByteOffset() {
      return this._dataByteOffset;
    }

    get readIndexByteOffset() {
      return this._readIndexByteOffset;
    }

    get writeIndexByteOffset() {
      return this._writeIndexByteOffset;
    }

    get capacitySamples() {
      return this._capacitySamples;
    }

    get capacityFrames() {
      return this._capacityFrames;
    }

    get channels() {
      return this._channels;
    }

    get sampleRate() {
      return this._sampleRate;
    }

    readCursor() {
      return Atomics.load(this._readCtrl, 0) >>> 0;
    }

    writeCursor() {
      return Atomics.load(this._writeCtrl, 0) >>> 0;
    }

    readableSamples() {
      const write = Atomics.load(this._writeCtrl, 0) >>> 0;
      const read = Atomics.load(this._readCtrl, 0) >>> 0;
      return (write - read) >>> 0;
    }

    availableFrames() {
      const readable = this.readableSamples();
      if (readable >= 0x80000000) {
        return 0;
      }
      const clamped = readable > this._capacitySamples ? this._capacitySamples : readable;
      return clamped >>> 1;
    }

    /**
     * Lock-free, zero-allocation SPSC consumer-side backlog trim.
     * Advances `_readCtrl` to leave at most `targetFrames` queued in the guest
     * ring. When `crossfadeFrames > 0` and audio is active, captures the old
     * head frames into `_trimXfadeSamples` so the next `readInto()` splices
     * across the jump with a click-free half-cosine crossfade.
     *
     * @param {number} targetFrames
     * @param {number} [crossfadeFrames=GUEST_TRIM_CROSSFADE_FRAMES]
     * @returns {number} Number of stale frames trimmed.
     */
    trimToFrames(targetFrames, crossfadeFrames = GUEST_TRIM_CROSSFADE_FRAMES) {
      const target = sanitizeNonNegInt(targetFrames, 0);
      let read = Atomics.load(this._readCtrl, 0) >>> 0;
      const write = Atomics.load(this._writeCtrl, 0) >>> 0;
      let readableSamples = (write - read) >>> 0;
      if (readableSamples >= 0x80000000) {
        Atomics.store(this._readCtrl, 0, write | 0);
        this._pendingTrimXfadeFrames = 0;
        return 0;
      }
      if (readableSamples > this._capacitySamples) {
        const excessSamples = readableSamples - this._capacitySamples;
        const overrunFrames = Math.ceil(excessSamples / GUEST_PCM_RING_CHANNELS);
        read = (read + overrunFrames * GUEST_PCM_RING_CHANNELS) >>> 0;
        readableSamples = (write - read) >>> 0;
      }
      const availFrames = readableSamples >>> 1;
      if (availFrames <= target) {
        return 0;
      }
      const skipFrames = availFrames - target;
      const requestedXfade = sanitizeNonNegInt(crossfadeFrames, 0);
      if (
        requestedXfade > 0 &&
        this._hasActiveAudio &&
        this._pendingTrimXfadeFrames === 0
      ) {
        const xf = Math.min(requestedXfade, GUEST_TRIM_CROSSFADE_FRAMES, availFrames);
        const xfSamples = xf * GUEST_PCM_RING_CHANNELS;
        const startSlot = read & this._mask;
        const tailRoom = this._capacitySamples - startSlot;
        const firstSamples = xfSamples <= tailRoom ? xfSamples : tailRoom;
        const secondSamples = xfSamples - firstSamples;
        const ringData = this._data;
        const dst = this._trimXfadeSamples;
        for (let i = 0; i < firstSamples; i++) {
          const v = ringData[startSlot + i];
          dst[i] = v - v === 0 ? v : 0.0;
        }
        for (let i = 0; i < secondSamples; i++) {
          const v = ringData[i];
          dst[firstSamples + i] = v - v === 0 ? v : 0.0;
        }
        this._pendingTrimXfadeFrames = xf;
      } else if (requestedXfade === 0) {
        this._pendingTrimXfadeFrames = 0;
      }
      const nextRead = (read + skipFrames * GUEST_PCM_RING_CHANNELS) >>> 0;
      Atomics.store(this._readCtrl, 0, nextRead | 0);
      return skipFrames;
    }

    /**
     * Reads up to `requestedFrames` interleaved stereo float32 frames into `output`.
     * Performs zero TypedArray/heap allocations and updates the atomic read cursor.
     *
     * @param {Float32Array} output Pre-allocated destination buffer (>= requestedFrames * 2).
     * @param {number} requestedFrames Stereo frames to read.
     * @param {{ softClip?: boolean, limiterKnee?: number, fillSilence?: boolean }} [options]
     * @param {object} [outResult] Optional reusable result object to populate in place.
     */
    readInto(output, requestedFrames, options, outResult) {
      if (!(output instanceof Float32Array)) {
        throw new TypeError('GuestPcmRingReader.readInto target must be a Float32Array');
      }
      if (!isSharedArrayBuffer(output.buffer) && isDetachedArrayBuffer(output.buffer)) {
        throw new TypeError('GuestPcmRingReader.readInto target buffer is detached');
      }
      const reqFrames = sanitizeNonNegInt(requestedFrames, 0);
      if (output.length < reqFrames * GUEST_PCM_RING_CHANNELS) {
        throw new RangeError(
          `GuestPcmRingReader.readInto output length (${output.length}) < requested stereo samples (${reqFrames * GUEST_PCM_RING_CHANNELS})`
        );
      }

      const res = outResult || this._reusableResult;
      res.framesRead = 0;
      res.underrun = false;
      res.underrunFrames = 0;
      res.overrun = false;
      res.overrunDroppedFrames = 0;
      res.clippedSamples = 0;
      res.softClippedSamples = 0;
      res.sanitizedSamples = 0;
      res.underrunFade = false;
      res.recoveryFade = false;
      res.peakLevel = 0.0;

      if (reqFrames <= 0) {
        return res;
      }

      const opts = options && typeof options === 'object' ? options : null;
      const softClip = opts && opts.softClip !== undefined ? Boolean(opts.softClip) : true;
      const rawKnee = opts && opts.limiterKnee !== undefined ? opts.limiterKnee : DEFAULT_SOFT_CLIP_KNEE;
      const knee =
        typeof rawKnee === 'number' && Number.isFinite(rawKnee) && rawKnee >= 0 && rawKnee < 1.0
          ? rawKnee
          : DEFAULT_SOFT_CLIP_KNEE;
      const negKnee = -knee;
      const headroom = 1.0 - knee;
      const fillSilence = opts && opts.fillSilence !== undefined ? Boolean(opts.fillSilence) : true;
      const declickFrames =
        opts && opts.declickFrames !== undefined
          ? sanitizeDeclickFrames(opts.declickFrames, 0)
          : 0;
      const useCosine =
        !opts || opts.fadeCurve === undefined || opts.fadeCurve === 'cosine';

      let read = Atomics.load(this._readCtrl, 0) >>> 0;
      const write = Atomics.load(this._writeCtrl, 0) >>> 0;
      let readableSamples = (write - read) >>> 0;
      let overrun = false;
      let overrunDroppedFrames = 0;

      if (readableSamples > this._capacitySamples) {
        overrun = true;
        if (readableSamples >= 0x80000000) {
          // Read cursor got ahead of write cursor (e.g. out-of-band reset anomaly): resync to write.
          read = write;
          readableSamples = 0;
        } else {
          const excessSamples = readableSamples - this._capacitySamples;
          overrunDroppedFrames = Math.ceil(excessSamples / GUEST_PCM_RING_CHANNELS);
          const skipSamples = overrunDroppedFrames * GUEST_PCM_RING_CHANNELS;
          read = (read + skipSamples) >>> 0;
          readableSamples = (write - read) >>> 0;
        }
      }

      const availFrames = readableSamples >>> 1;
      const framesToRead = reqFrames <= availFrames ? reqFrames : availFrames;
      const samplesToRead = framesToRead * GUEST_PCM_RING_CHANNELS;

      let clippedSamples = 0;
      let softClippedSamples = 0;
      let sanitizedSamples = 0;
      let peakLevel = 0.0;

      if (samplesToRead > 0) {
        const startSlot = read & this._mask;
        const tailRoom = this._capacitySamples - startSlot;
        const firstSamples = samplesToRead <= tailRoom ? samplesToRead : tailRoom;
        const secondSamples = samplesToRead - firstSamples;
        const ringData = this._data;

        for (let i = 0; i < firstSamples; i++) {
          let v = ringData[startSlot + i];
          if (v - v !== 0) {
            v = 0.0;
            sanitizedSamples += 1;
          } else if (softClip) {
            if (v > knee) {
              clippedSamples += 1;
              softClippedSamples += 1;
              const compressed = knee + headroom * Math.tanh((v - knee) / headroom);
              v = compressed > 1.0 ? 1.0 : compressed;
            } else if (v < negKnee) {
              clippedSamples += 1;
              softClippedSamples += 1;
              const compressed = knee + headroom * Math.tanh((-v - knee) / headroom);
              v = compressed > 1.0 ? -1.0 : -compressed;
            }
          } else if (v > 1.0) {
            v = 1.0;
            clippedSamples += 1;
          } else if (v < -1.0) {
            v = -1.0;
            clippedSamples += 1;
          }
          const absV = v < 0 ? -v : v;
          if (absV > peakLevel) {
            peakLevel = absV;
          }
          output[i] = v;
        }

        for (let i = 0; i < secondSamples; i++) {
          let v = ringData[i];
          if (v - v !== 0) {
            v = 0.0;
            sanitizedSamples += 1;
          } else if (softClip) {
            if (v > knee) {
              clippedSamples += 1;
              softClippedSamples += 1;
              const compressed = knee + headroom * Math.tanh((v - knee) / headroom);
              v = compressed > 1.0 ? 1.0 : compressed;
            } else if (v < negKnee) {
              clippedSamples += 1;
              softClippedSamples += 1;
              const compressed = knee + headroom * Math.tanh((-v - knee) / headroom);
              v = compressed > 1.0 ? -1.0 : -compressed;
            }
          } else if (v > 1.0) {
            v = 1.0;
            clippedSamples += 1;
          } else if (v < -1.0) {
            v = -1.0;
            clippedSamples += 1;
          }
          const absV = v < 0 ? -v : v;
          if (absV > peakLevel) {
            peakLevel = absV;
          }
          output[firstSamples + i] = v;
        }

        read = (read + samplesToRead) >>> 0;
      }

      if (samplesToRead > 0 || overrun) {
        Atomics.store(this._readCtrl, 0, read | 0);
      }

      // Click-free half-cosine crossfade across a consumer-side backlog trim boundary.
      if (framesToRead > 0 && this._pendingTrimXfadeFrames > 0) {
        const xfFrames =
          this._pendingTrimXfadeFrames < framesToRead
            ? this._pendingTrimXfadeFrames
            : framesToRead;
        const oldBuf = this._trimXfadeSamples;
        const denom = xfFrames + 1;
        const step = Math.PI / denom;
        const halfStep = 0.5 * step;
        const s = Math.sin(halfStep);
        const k = 4.0 * s * s;
        let c = 1.0 - 0.5 * k;
        let delta = -0.5 * k - k * c;
        for (let f = 0; f < xfFrames; f++) {
          const wNew = 0.5 * (1.0 - c);
          const wOld = 1.0 - wNew;
          const base = f * GUEST_PCM_RING_CHANNELS;
          const l = oldBuf[base] * wOld + output[base] * wNew;
          const r = oldBuf[base + 1] * wOld + output[base + 1] * wNew;
          output[base] = l;
          output[base + 1] = r;
          const absL = l < 0 ? -l : l;
          const absR = r < 0 ? -r : r;
          if (absL > peakLevel) peakLevel = absL;
          if (absR > peakLevel) peakLevel = absR;
          if ((f & COS_RECURRENCE_RESYNC_MASK) === COS_RECURRENCE_RESYNC_MASK && f + 1 < xfFrames) {
            const nextTheta = (f + 2) * step;
            c = Math.cos(nextTheta);
            delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
          } else {
            c += delta;
            delta -= k * c;
          }
        }
        this._fastCosineRamps += 1;
        this._pendingTrimXfadeFrames = 0;
      }

      // W272/W299 click-free half-cosine (or linear) recovery fade-in when audio resumes after underrun.
      if (framesToRead > 0 && declickFrames > 0 && fillSilence) {
        if (this._inUnderrun) {
          this._inUnderrun = false;
          this._fadeInPos = 0;
          this._fadeInTotal = declickFrames;
          this._recoveryFadeCount += 1;
          res.recoveryFade = true;
        }
        if (this._fadeInTotal > 0 && this._fadeInPos < this._fadeInTotal) {
          const remaining = this._fadeInTotal - this._fadeInPos;
          const rampCount = framesToRead < remaining ? framesToRead : remaining;
          const total = this._fadeInTotal;
          const startPos = this._fadeInPos;
          if (useCosine) {
            const step = Math.PI / total;
            const halfStep = 0.5 * step;
            const s = Math.sin(halfStep);
            const k = 4.0 * s * s;
            const theta0 = startPos * step;
            let c = startPos === 0 ? 1.0 : Math.cos(theta0);
            let delta =
              startPos === 0 ? -0.5 * k : -2.0 * Math.sin(theta0 + halfStep) * s;
            for (let f = 0; f < rampCount; f++) {
              const rampGain = 0.5 * (1.0 - c);
              const base = f * GUEST_PCM_RING_CHANNELS;
              output[base] *= rampGain;
              output[base + 1] *= rampGain;
              if (
                (f & COS_RECURRENCE_RESYNC_MASK) === COS_RECURRENCE_RESYNC_MASK &&
                f + 1 < rampCount
              ) {
                const nextTheta = (startPos + f + 1) * step;
                c = Math.cos(nextTheta);
                delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
              } else {
                c += delta;
                delta -= k * c;
              }
            }
            this._fastCosineRamps += 1;
          } else {
            const invTotal = 1.0 / total;
            for (let f = 0; f < rampCount; f++) {
              const rampGain = (startPos + f) * invTotal;
              const base = f * GUEST_PCM_RING_CHANNELS;
              output[base] *= rampGain;
              output[base + 1] *= rampGain;
            }
          }
          this._fadeInPos += rampCount;
          if (this._fadeInPos >= this._fadeInTotal) {
            this._fadeInPos = 0;
            this._fadeInTotal = 0;
          }
        }
      }

      if (framesToRead < reqFrames) {
        const missingFrames = reqFrames - framesToRead;
        if (declickFrames > 0 && fillSilence) {
          this._fadeInPos = 0;
          this._fadeInTotal = 0;
          if (framesToRead > 0 || this._hasActiveAudio) {
            if (
              framesToRead > 0 &&
              missingFrames < declickFrames &&
              framesToRead > missingFrames
            ) {
              const fadeFrames = declickFrames < framesToRead ? declickFrames : framesToRead;
              const fadeStart = framesToRead - fadeFrames;
              const lastFade = fadeFrames - 1;
              if (useCosine) {
                if (lastFade > 0) {
                  const step = Math.PI / fadeFrames;
                  const halfStep = 0.5 * step;
                  const s = Math.sin(halfStep);
                  const k = 4.0 * s * s;
                  let c = 1.0 - 0.5 * k;
                  let delta = -0.5 * k - k * c;
                  for (let i = 0; i < lastFade; i++) {
                    const rampGain = 0.5 * (1.0 + c);
                    const base = (fadeStart + i) * GUEST_PCM_RING_CHANNELS;
                    output[base] *= rampGain;
                    output[base + 1] *= rampGain;
                    if (
                      (i & COS_RECURRENCE_RESYNC_MASK) === COS_RECURRENCE_RESYNC_MASK &&
                      i + 1 < lastFade
                    ) {
                      const nextTheta = (i + 2) * step;
                      c = Math.cos(nextTheta);
                      delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
                    } else {
                      c += delta;
                      delta -= k * c;
                    }
                  }
                }
                const endBase = (fadeStart + lastFade) * GUEST_PCM_RING_CHANNELS;
                output[endBase] = 0.0;
                output[endBase + 1] = 0.0;
                this._fastCosineRamps += 1;
              } else {
                const invFade = 1.0 / fadeFrames;
                for (let i = 0; i < fadeFrames; i++) {
                  const rampGain = i === lastFade ? 0.0 : (lastFade - i) * invFade;
                  const base = (fadeStart + i) * GUEST_PCM_RING_CHANNELS;
                  output[base] *= rampGain;
                  output[base + 1] *= rampGain;
                }
              }
              output.fill(0, samplesToRead, reqFrames * GUEST_PCM_RING_CHANNELS);
            } else {
              if (framesToRead > 0) {
                const lastBase = (framesToRead - 1) * GUEST_PCM_RING_CHANNELS;
                this._lastStereoFrame[0] = output[lastBase];
                this._lastStereoFrame[1] = output[lastBase + 1];
              }
              const fadeFrames = declickFrames < missingFrames ? declickFrames : missingFrames;
              const lastL = this._lastStereoFrame[0];
              const lastR = this._lastStereoFrame[1];
              const lastFade = fadeFrames - 1;
              if (useCosine) {
                if (lastFade > 0) {
                  const step = Math.PI / fadeFrames;
                  const halfStep = 0.5 * step;
                  const s = Math.sin(halfStep);
                  const k = 4.0 * s * s;
                  let c = 1.0 - 0.5 * k;
                  let delta = -0.5 * k - k * c;
                  for (let i = 0; i < lastFade; i++) {
                    const rampGain = 0.5 * (1.0 + c);
                    const base = (framesToRead + i) * GUEST_PCM_RING_CHANNELS;
                    output[base] = lastL * rampGain;
                    output[base + 1] = lastR * rampGain;
                    if (
                      (i & COS_RECURRENCE_RESYNC_MASK) === COS_RECURRENCE_RESYNC_MASK &&
                      i + 1 < lastFade
                    ) {
                      const nextTheta = (i + 2) * step;
                      c = Math.cos(nextTheta);
                      delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
                    } else {
                      c += delta;
                      delta -= k * c;
                    }
                  }
                }
                const endBase = (framesToRead + lastFade) * GUEST_PCM_RING_CHANNELS;
                output[endBase] = 0.0;
                output[endBase + 1] = 0.0;
                this._fastCosineRamps += 1;
              } else {
                const invFade = 1.0 / fadeFrames;
                for (let i = 0; i < fadeFrames; i++) {
                  const rampGain = i === lastFade ? 0.0 : (lastFade - i) * invFade;
                  const base = (framesToRead + i) * GUEST_PCM_RING_CHANNELS;
                  output[base] = lastL * rampGain;
                  output[base + 1] = lastR * rampGain;
                }
              }
              if (framesToRead + fadeFrames < reqFrames) {
                output.fill(
                  0,
                  (framesToRead + fadeFrames) * GUEST_PCM_RING_CHANNELS,
                  reqFrames * GUEST_PCM_RING_CHANNELS
                );
              }
            }
            this._underrunFadeCount += 1;
            res.underrunFade = true;
            this._hasActiveAudio = false;
            this._lastStereoFrame[0] = 0.0;
            this._lastStereoFrame[1] = 0.0;
          } else {
            output.fill(0, samplesToRead, reqFrames * GUEST_PCM_RING_CHANNELS);
          }
          this._inUnderrun = true;
        } else if (fillSilence) {
          output.fill(0, samplesToRead, reqFrames * GUEST_PCM_RING_CHANNELS);
        }
        res.underrun = true;
        res.underrunFrames = missingFrames;
      } else if (framesToRead > 0) {
        const lastBase = (framesToRead - 1) * GUEST_PCM_RING_CHANNELS;
        this._lastStereoFrame[0] = output[lastBase];
        this._lastStereoFrame[1] = output[lastBase + 1];
        this._hasActiveAudio = true;
      }

      res.framesRead = framesToRead;
      res.overrun = overrun;
      res.overrunDroppedFrames = overrunDroppedFrames;
      res.clippedSamples = clippedSamples;
      res.softClippedSamples = softClippedSamples;
      res.sanitizedSamples = sanitizedSamples;
      res.peakLevel = peakLevel;
      return res;
    }
  }

  /**
   * Lock-free Single-Producer / Single-Consumer (SPSC) PCM audio ring buffer
   * backed by a SharedArrayBuffer.
   *
   * Uses double-modulo cursors in [0, 2 * capacityFrames) so 100% of
   * capacityFrames are usable without a wasted sentinel slot and for any
   * positive capacityFrames (power-of-two or arbitrary).
   */
  class AudioRingBuffer {
    static HEADER_INT32_COUNT = RING_HEADER_INT32_COUNT;
    static HEADER_BYTES = RING_HEADER_BYTES;

    static byteLength(capacityFrames, channels = DEFAULT_CHANNELS) {
      if (!Number.isInteger(capacityFrames) || capacityFrames <= 0) {
        throw new RangeError(`capacityFrames must be a positive integer, got ${capacityFrames}`);
      }
      if (!Number.isInteger(channels) || channels <= 0) {
        throw new RangeError(`channels must be a positive integer, got ${channels}`);
      }
      return (
        RING_HEADER_BYTES +
        capacityFrames * channels * Float32Array.BYTES_PER_ELEMENT
      );
    }

    static create(
      capacityFrames = DEFAULT_CAPACITY_FRAMES,
      channels = DEFAULT_CHANNELS,
      options = {}
    ) {
      const byteLen = AudioRingBuffer.byteLength(capacityFrames, channels);
      const sab = new SharedArrayBuffer(byteLen);
      const opts = options && typeof options === 'object' ? options : {};
      return new AudioRingBuffer(sab, {
        ...opts,
        capacityFrames,
        channels,
        initialize: true,
      });
    }

    constructor(sab, options = {}) {
      if (typeof SharedArrayBuffer === 'undefined' || !(sab instanceof SharedArrayBuffer)) {
        throw new TypeError('AudioRingBuffer requires a SharedArrayBuffer');
      }
      if (sab.byteLength < RING_HEADER_BYTES) {
        throw new RangeError(
          `SharedArrayBuffer too small (${sab.byteLength} bytes; need > ${RING_HEADER_BYTES})`
        );
      }

      this._sab = sab;
      this._ctrl = new Int32Array(sab, 0, RING_HEADER_INT32_COUNT);

      const existingCap = Atomics.load(this._ctrl, HDR_CAPACITY_FRAMES);
      const existingCh = Atomics.load(this._ctrl, HDR_CHANNELS);
      const shouldInit = Boolean(options.initialize) || existingCap === 0;

      if (shouldInit) {
        const channels = options.channels !== undefined ? options.channels : DEFAULT_CHANNELS;
        if (!Number.isInteger(channels) || channels <= 0) {
          throw new RangeError(`channels must be a positive integer, got ${channels}`);
        }
        const maxFrames = Math.floor(
          (sab.byteLength - RING_HEADER_BYTES) /
            (channels * Float32Array.BYTES_PER_ELEMENT)
        );
        const capacityFrames =
          options.capacityFrames !== undefined ? options.capacityFrames : maxFrames;
        if (
          !Number.isInteger(capacityFrames) ||
          capacityFrames <= 0 ||
          capacityFrames > maxFrames
        ) {
          throw new RangeError(
            `capacityFrames (${capacityFrames}) exceeds SharedArrayBuffer capacity (${maxFrames})`
          );
        }

        Atomics.store(this._ctrl, HDR_WRITE_CURSOR, 0);
        Atomics.store(this._ctrl, HDR_READ_CURSOR, 0);
        Atomics.store(this._ctrl, HDR_CAPACITY_FRAMES, capacityFrames);
        Atomics.store(this._ctrl, HDR_CHANNELS, channels);
        Atomics.store(this._ctrl, HDR_UNDERRUNS, 0);
        Atomics.store(this._ctrl, HDR_UNDERRUN_FRAMES, 0);
        Atomics.store(this._ctrl, HDR_OVERFLOWS, 0);
        Atomics.store(this._ctrl, HDR_DROPPED_FRAMES, 0);

        this._capacityFrames = capacityFrames;
        this._channels = channels;
      } else {
        const channels = existingCh > 0 ? existingCh : (options.channels || DEFAULT_CHANNELS);
        const requiredBytes = AudioRingBuffer.byteLength(existingCap, channels);
        if (sab.byteLength < requiredBytes) {
          throw new RangeError(
            `SharedArrayBuffer (${sab.byteLength} bytes) is smaller than header requires (${requiredBytes} bytes)`
          );
        }
        this._capacityFrames = existingCap;
        this._channels = channels;
      }

      this._modulus = this._capacityFrames * 2;
      this._samples = new Float32Array(
        sab,
        RING_HEADER_BYTES,
        this._capacityFrames * this._channels
      );

      this._volume = clampVolume(options.volume !== undefined ? options.volume : 1.0);
      this._muted = Boolean(options.muted);
      this._declickFrames = sanitizeDeclickFrames(options.declickFrames, 0);
      this._fadeCurve = sanitizeFadeCurve(
        options.fadeCurve,
        options.cosineRamp === true ? 'cosine' : DEFAULT_FADE_CURVE
      );
      this._overflowPolicy = normalizeOverflowPolicy(
        options.overflowPolicy,
        DEFAULT_OVERFLOW_POLICY
      );
      this._maxFillRatio = sanitizeFillRatio(options.maxFillRatio, DEFAULT_MAX_FILL_RATIO);
      this._underrunRecoveryFrames = sanitizeNonNegInt(
        options.underrunRecoveryFrames,
        DEFAULT_UNDERRUN_RECOVERY_FRAMES
      );
      this._lastOutputFrame = new Float32Array(this._channels);
      this._hasActiveAudio = false;
      this._inUnderrun = false;
      this._fadeInPos = 0;
      this._fadeInTotal = 0;
      this._fadeOutCount = 0;
      this._fadeInCount = 0;
      this._underrunRecoveries = 0;
      this._consecutiveUnderruns = 0;
      this._maxConsecutiveUnderruns = 0;
      this._peakFillFrames = 0;
      this._minFillFrames = Infinity;
      this._fillFramesSum = 0;
      this._fillSamples = 0;
      this._lastObservedFill = null;
      this._packetsReceived = 0;
      this._framesDecoded = 0;
      this._droppedPackets = 0;
      this._lastPushDroppedFrames = 0;
      this._lastPushOverflowed = false;
      this._fastCosineRamps = 0;
      this._pushScratch = null;
      this._pushExtractResult = null;
      this._pushExtractOpts = null;
      this._pullScratch = null;
      this._pullScratchView = null;
      this._pullScratchViewLen = 0;
      this._pullScratchAllocations = 0;
      this._pullScratchReuses = 0;
    }

    get sab() {
      return this._sab;
    }

    get capacityFrames() {
      return this._capacityFrames;
    }

    get channels() {
      return this._channels;
    }

    get volume() {
      return this._volume;
    }

    get muted() {
      return this._muted;
    }

    get declickFrames() {
      return this._declickFrames;
    }

    get fadeCurve() {
      return this._fadeCurve;
    }

    get overflowPolicy() {
      return this._overflowPolicy;
    }

    get maxFillRatio() {
      return this._maxFillRatio;
    }

    get underrunRecoveryFrames() {
      return this._underrunRecoveryFrames;
    }

    get underruns() {
      return Atomics.load(this._ctrl, HDR_UNDERRUNS) >>> 0;
    }

    get underrunCount() {
      return this.underruns;
    }

    get underrunFrames() {
      return Atomics.load(this._ctrl, HDR_UNDERRUN_FRAMES) >>> 0;
    }

    get underrunRecoveries() {
      return this._underrunRecoveries;
    }

    get consecutiveUnderruns() {
      return this._consecutiveUnderruns;
    }

    get maxConsecutiveUnderruns() {
      return this._maxConsecutiveUnderruns;
    }

    get fadeOutCount() {
      return this._fadeOutCount;
    }

    get fadeInCount() {
      return this._fadeInCount;
    }

    get fastCosineRamps() {
      return this._fastCosineRamps;
    }

    get peakFillFrames() {
      return this._fillSamples > 0 ? this._peakFillFrames : 0;
    }

    get minFillFrames() {
      return this._fillSamples > 0 ? this._minFillFrames : 0;
    }

    get avgFillRatio() {
      if (this._fillSamples <= 0 || this._capacityFrames <= 0) {
        return 0.0;
      }
      const ratio = this._fillFramesSum / this._fillSamples / this._capacityFrames;
      if (ratio < 0) return 0.0;
      if (ratio > 1) return 1.0;
      return ratio;
    }

    get bufferFillPct() {
      if (this._capacityFrames <= 0) {
        return 0.0;
      }
      const pct = (this.availableFrames() / this._capacityFrames) * 100;
      if (pct < 0) return 0.0;
      if (pct > 100) return 100.0;
      return pct;
    }

    get overflows() {
      return Atomics.load(this._ctrl, HDR_OVERFLOWS) >>> 0;
    }

    get overruns() {
      return this.overflows;
    }

    get droppedFrames() {
      return Atomics.load(this._ctrl, HDR_DROPPED_FRAMES) >>> 0;
    }

    get packetsReceived() {
      return this._packetsReceived;
    }

    get framesDecoded() {
      return this._framesDecoded;
    }

    get droppedPackets() {
      return this._droppedPackets;
    }

    setVolume(volume) {
      this._volume = clampVolume(volume);
      return this._volume;
    }

    setMuted(muted) {
      this._muted = Boolean(muted);
      return this._muted;
    }

    setDeclickFrames(frames) {
      this._declickFrames = sanitizeDeclickFrames(frames, this._declickFrames);
      return this._declickFrames;
    }

    setFadeCurve(curve) {
      this._fadeCurve = sanitizeFadeCurve(curve, this._fadeCurve);
      return this._fadeCurve;
    }

    setOverflowPolicy(policy) {
      this._overflowPolicy = normalizeOverflowPolicy(policy, this._overflowPolicy);
      return this._overflowPolicy;
    }

    setMaxFillRatio(ratio) {
      this._maxFillRatio = sanitizeFillRatio(ratio, this._maxFillRatio);
      return this._maxFillRatio;
    }

    setUnderrunRecoveryFrames(frames) {
      this._underrunRecoveryFrames = sanitizeNonNegInt(frames, this._underrunRecoveryFrames);
      return this._underrunRecoveryFrames;
    }

    _normalizeCursor(cursor) {
      const mod = this._modulus;
      if (cursor >= 0 && cursor < mod) {
        return cursor;
      }
      const norm = cursor % mod;
      return norm < 0 ? norm + mod : norm;
    }

    _effectiveMaxFillFrames(override) {
      const cap = this._capacityFrames;
      if (typeof override === 'number' && Number.isFinite(override) && override > 0) {
        const n = Math.floor(override);
        return n < cap ? n : cap;
      }
      if (this._maxFillRatio >= 1.0) {
        return cap;
      }
      const scaled = Math.floor(cap * this._maxFillRatio);
      return scaled < 1 ? 1 : scaled > cap ? cap : scaled;
    }

    _recordFillSample(fillFrames) {
      const clamped =
        fillFrames < 0
          ? 0
          : fillFrames > this._capacityFrames
            ? this._capacityFrames
            : fillFrames;
      if (this._fillSamples === 0) {
        this._peakFillFrames = clamped;
        this._minFillFrames = clamped;
      } else {
        if (clamped > this._peakFillFrames) this._peakFillFrames = clamped;
        if (clamped < this._minFillFrames) this._minFillFrames = clamped;
      }
      this._fillFramesSum += clamped;
      this._fillSamples += 1;
      this._lastObservedFill = clamped;
    }

    _recordFillTransition(beforeFill, afterFill) {
      if (beforeFill !== afterFill && this._lastObservedFill !== beforeFill) {
        this._recordFillSample(beforeFill);
      }
      this._recordFillSample(afterFill);
    }

    effectiveGain() {
      return this._muted ? 0.0 : this._volume;
    }

    availableFrames() {
      const write = this._normalizeCursor(Atomics.load(this._ctrl, HDR_WRITE_CURSOR));
      const read = this._normalizeCursor(Atomics.load(this._ctrl, HDR_READ_CURSOR));
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }
      if (used < 0) return 0;
      if (used > this._capacityFrames) return this._capacityFrames;
      return used;
    }

    freeFrames() {
      return this._capacityFrames - this.availableFrames();
    }

    /**
     * Pushes interleaved PCM float samples into the ring buffer.
     * Applies bounded overflow drop guards (`drop-tail`, `drop-newest`, or
     * `drop-oldest`) respecting `maxFillFrames` / `maxFillRatio`, performs zero
     * per-packet TypedArray allocation on steady-state Float32Array inputs, and
     * updates atomic overflow counters on drop.
     *
     * @param {Float32Array|ArrayBuffer|SharedArrayBuffer|number[]} samples Interleaved PCM samples.
     * @param {number|{ frameCount?: number, overflowPolicy?: string, maxFillFrames?: number }} [optionsOrFrameCount]
     * @returns {number} Number of complete frames written.
     */
    pushSamples(samples, optionsOrFrameCount) {
      let pcm;
      if (samples instanceof Float32Array && !isSharedArrayBuffer(samples.buffer)) {
        if (isDetachedArrayBuffer(samples.buffer)) {
          throw new TypeError('PCM TypedArray backing ArrayBuffer is detached');
        }
        pcm = samples;
      } else {
        if (!this._pushScratch) {
          this._pushScratch = new PcmScratchPool({ maxSamples: 16 * 1024 * 1024 });
          this._pushExtractResult = {
            ok: false,
            pcm: null,
            copiedFromSab: false,
            sanitizedSamples: 0,
            error: '',
          };
          this._pushExtractOpts = {
            maxPcmBytes: 64 * 1024 * 1024,
            scratch: this._pushScratch,
            outResult: this._pushExtractResult,
          };
        }
        const extracted = extractSafePcmSamples(samples, this._pushExtractOpts);
        if (!extracted.ok) {
          const errMsg = extracted.error;
          this._pushExtractResult.pcm = null;
          throw new TypeError(errMsg);
        }
        pcm = extracted.pcm;
        this._pushExtractResult.pcm = null;
      }

      const channels = this._channels;
      const maxAvailFrames = Math.floor(pcm.length / channels);
      let frameCountOpt;
      let policyOpt = this._overflowPolicy;
      let maxFillOpt;

      if (typeof optionsOrFrameCount === 'number') {
        frameCountOpt = optionsOrFrameCount;
      } else if (optionsOrFrameCount && typeof optionsOrFrameCount === 'object') {
        if (optionsOrFrameCount.frameCount !== undefined) {
          frameCountOpt = optionsOrFrameCount.frameCount;
        }
        if (optionsOrFrameCount.overflowPolicy !== undefined) {
          policyOpt = normalizeOverflowPolicy(
            optionsOrFrameCount.overflowPolicy,
            this._overflowPolicy
          );
        }
        if (optionsOrFrameCount.maxFillFrames !== undefined) {
          maxFillOpt = optionsOrFrameCount.maxFillFrames;
        }
      }

      const inputFrames =
        frameCountOpt !== undefined
          ? Math.min(sanitizeNonNegInt(frameCountOpt, 0), maxAvailFrames)
          : maxAvailFrames;

      this._lastPushDroppedFrames = 0;
      this._lastPushOverflowed = false;
      this._packetsReceived += 1;
      this._framesDecoded += inputFrames;

      if (inputFrames <= 0) {
        return 0;
      }

      const cap = this._capacityFrames;
      const maxFillFrames = this._effectiveMaxFillFrames(maxFillOpt);
      const write = this._normalizeCursor(Atomics.load(this._ctrl, HDR_WRITE_CURSOR));
      let read = this._normalizeCursor(Atomics.load(this._ctrl, HDR_READ_CURSOR));
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }
      if (used > cap) {
        used = cap;
      }

      const effectiveFree = maxFillFrames > used ? maxFillFrames - used : 0;
      let framesToWrite = 0;
      let srcFrameOffset = 0;
      let droppedFrames = 0;
      let overflowed = false;

      if (inputFrames <= effectiveFree) {
        framesToWrite = inputFrames;
      } else if (policyOpt === OVERFLOW_POLICIES.DROP_NEWEST) {
        framesToWrite = 0;
        droppedFrames = inputFrames;
        overflowed = true;
      } else if (policyOpt === OVERFLOW_POLICIES.DROP_OLDEST) {
        overflowed = true;
        let toEvict = 0;
        let trimmedInput = 0;
        if (inputFrames >= maxFillFrames) {
          toEvict = used;
          trimmedInput = inputFrames - maxFillFrames;
          srcFrameOffset = trimmedInput;
          framesToWrite = maxFillFrames;
        } else {
          toEvict = used + inputFrames - maxFillFrames;
          if (toEvict > used) {
            toEvict = used;
          }
          framesToWrite = inputFrames;
        }
        if (toEvict > 0) {
          let nextRead = read + toEvict;
          if (nextRead >= this._modulus) {
            nextRead -= this._modulus;
          }
          Atomics.store(this._ctrl, HDR_READ_CURSOR, nextRead);
          read = nextRead;
          used -= toEvict;
        }
        droppedFrames = toEvict + trimmedInput;
      } else {
        // Default: 'drop-tail' — write as many frames as fit without clobbering unread audio.
        framesToWrite = effectiveFree;
        droppedFrames = inputFrames - framesToWrite;
        overflowed = droppedFrames > 0;
      }

      if (framesToWrite > 0) {
        const writeSlot = write >= cap ? write - cap : write;
        const firstFrames = Math.min(framesToWrite, cap - writeSlot);
        const secondFrames = framesToWrite - firstFrames;
        const firstSampleCount = firstFrames * channels;
        const secondSampleCount = secondFrames * channels;
        const srcBase = srcFrameOffset * channels;
        const dstBase = writeSlot * channels;
        const ringSamples = this._samples;

        for (let i = 0; i < firstSampleCount; i++) {
          const v = pcm[srcBase + i];
          ringSamples[dstBase + i] = Number.isFinite(v) ? v : 0.0;
        }

        if (secondSampleCount > 0) {
          const srcSecondBase = srcBase + firstSampleCount;
          for (let i = 0; i < secondSampleCount; i++) {
            const v = pcm[srcSecondBase + i];
            ringSamples[i] = Number.isFinite(v) ? v : 0.0;
          }
        }

        let nextWrite = write + framesToWrite;
        if (nextWrite >= this._modulus) {
          nextWrite -= this._modulus;
        }
        Atomics.store(this._ctrl, HDR_WRITE_CURSOR, nextWrite);
      }

      this._recordFillSample(used + framesToWrite);

      if (overflowed || droppedFrames > 0) {
        Atomics.add(this._ctrl, HDR_OVERFLOWS, 1);
        Atomics.add(this._ctrl, HDR_DROPPED_FRAMES, droppedFrames);
        this._droppedPackets += 1;
        this._lastPushDroppedFrames = droppedFrames;
        this._lastPushOverflowed = true;
      }

      return framesToWrite;
    }

    /**
     * Pulls interleaved PCM float samples out of the ring buffer into `output`.
     * Applies master volume / mute gain. On underrun, applies a smooth linear
     * de-click fade-out when `declickFrames > 0` (or zero-fills when disabled)
     * and increments atomic underrun counters. On recovery after underrun,
     * optionally waits for `underrunRecoveryFrames` to accumulate before
     * resuming with a matching linear fade-in. Steady-state pulls with no
     * underrun remain bit-exact unchanged with zero TypedArray allocations.
     *
     * @param {Float32Array} output Destination buffer (length = frames * channels).
     * @param {number} [gainOverride] Optional explicit gain override in [0, 1].
     * @returns {number} Number of real frames pulled before any underrun fill.
     */
    pullSamples(output, gainOverride, frameCount) {
      if (!(output instanceof Float32Array)) {
        throw new TypeError('pullSamples target must be a Float32Array');
      }
      if (!isSharedArrayBuffer(output.buffer) && isDetachedArrayBuffer(output.buffer)) {
        throw new TypeError('pullSamples target buffer is detached');
      }

      const channels = this._channels;
      const maxOutputFrames = Math.floor(output.length / channels);
      const requestedFrames =
        frameCount !== undefined
          ? Math.min(sanitizeNonNegInt(frameCount, 0), maxOutputFrames)
          : maxOutputFrames;
      if (requestedFrames <= 0) {
        return 0;
      }

      const cap = this._capacityFrames;
      const read = this._normalizeCursor(Atomics.load(this._ctrl, HDR_READ_CURSOR));
      const write = this._normalizeCursor(Atomics.load(this._ctrl, HDR_WRITE_CURSOR));
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }
      if (used > cap) {
        used = cap;
      }

      // Optional underrun re-priming watermark: hold silence until enough frames
      // have accumulated to resume cleanly without immediate re-underrun buzz.
      if (this._inUnderrun && this._underrunRecoveryFrames > 0) {
        const recoveryTarget =
          this._underrunRecoveryFrames < cap ? this._underrunRecoveryFrames : cap;
        if (used < recoveryTarget) {
          output.fill(0, 0, requestedFrames * channels);
          this._recordFillTransition(used, used);
          this._consecutiveUnderruns += 1;
          if (this._consecutiveUnderruns > this._maxConsecutiveUnderruns) {
            this._maxConsecutiveUnderruns = this._consecutiveUnderruns;
          }
          Atomics.add(this._ctrl, HDR_UNDERRUNS, 1);
          Atomics.add(this._ctrl, HDR_UNDERRUN_FRAMES, requestedFrames);
          return 0;
        }
      }

      const framesToRead = requestedFrames <= used ? requestedFrames : used;
      const gain =
        gainOverride !== undefined ? clampVolume(gainOverride) : this.effectiveGain();

      if (framesToRead > 0) {
        const readSlot = read >= cap ? read - cap : read;
        const firstFrames = Math.min(framesToRead, cap - readSlot);
        const secondFrames = framesToRead - firstFrames;
        const firstSamples = firstFrames * channels;
        const secondSamples = secondFrames * channels;
        const ringSamples = this._samples;
        const srcBase = readSlot * channels;

        if (gain === 0.0) {
          output.fill(0, 0, framesToRead * channels);
        } else if (gain === 1.0) {
          for (let i = 0; i < firstSamples; i++) {
            output[i] = ringSamples[srcBase + i];
          }
          for (let i = 0; i < secondSamples; i++) {
            output[firstSamples + i] = ringSamples[i];
          }
        } else {
          for (let i = 0; i < firstSamples; i++) {
            output[i] = ringSamples[srcBase + i] * gain;
          }
          for (let i = 0; i < secondSamples; i++) {
            output[firstSamples + i] = ringSamples[i] * gain;
          }
        }

        let nextRead = read + framesToRead;
        if (nextRead >= this._modulus) {
          nextRead -= this._modulus;
        }
        Atomics.store(this._ctrl, HDR_READ_CURSOR, nextRead);
      }

      const afterFill = Math.max(0, used - framesToRead);
      this._recordFillTransition(used, afterFill);

      const declickFrames = this._declickFrames;
      const useCosine = this._fadeCurve === 'cosine';

      // Track recovery from underrun and apply smooth linear or half-cosine fade-in when enabled.
      if (framesToRead > 0) {
        if (this._inUnderrun) {
          this._inUnderrun = false;
          this._consecutiveUnderruns = 0;
          this._underrunRecoveries += 1;
          if (declickFrames > 0) {
            this._fadeInPos = 0;
            this._fadeInTotal = declickFrames;
            this._fadeInCount += 1;
          }
        }
        if (declickFrames > 0 && this._fadeInTotal > 0 && this._fadeInPos < this._fadeInTotal) {
          const remaining = this._fadeInTotal - this._fadeInPos;
          const rampCount = framesToRead < remaining ? framesToRead : remaining;
          const total = this._fadeInTotal;
          const startPos = this._fadeInPos;
          if (useCosine) {
            const step = Math.PI / total;
            const halfStep = 0.5 * step;
            const s = Math.sin(halfStep);
            const k = 4.0 * s * s;
            const theta0 = startPos * step;
            let c = startPos === 0 ? 1.0 : Math.cos(theta0);
            let delta =
              startPos === 0 ? -0.5 * k : -2.0 * Math.sin(theta0 + halfStep) * s;
            for (let f = 0; f < rampCount; f++) {
              const rampGain = 0.5 * (1.0 - c);
              const base = f * channels;
              for (let ch = 0; ch < channels; ch++) {
                output[base + ch] *= rampGain;
              }
              if (
                (f & COS_RECURRENCE_RESYNC_MASK) === COS_RECURRENCE_RESYNC_MASK &&
                f + 1 < rampCount
              ) {
                const nextTheta = (startPos + f + 1) * step;
                c = Math.cos(nextTheta);
                delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
              } else {
                c += delta;
                delta -= k * c;
              }
            }
            this._fastCosineRamps += 1;
          } else {
            const invTotal = 1.0 / total;
            for (let f = 0; f < rampCount; f++) {
              const rampGain = (startPos + f) * invTotal;
              const base = f * channels;
              for (let ch = 0; ch < channels; ch++) {
                output[base + ch] *= rampGain;
              }
            }
          }
          this._fadeInPos += rampCount;
          if (this._fadeInPos >= this._fadeInTotal) {
            this._fadeInPos = 0;
            this._fadeInTotal = 0;
          }
        }
      }

      if (framesToRead < requestedFrames) {
        const missingFrames = requestedFrames - framesToRead;

        if (declickFrames > 0) {
          // Entering underrun cancels any partial recovery fade-in.
          this._fadeInPos = 0;
          this._fadeInTotal = 0;

          if (framesToRead > 0 || this._hasActiveAudio) {
            if (
              framesToRead > 0 &&
              missingFrames < declickFrames &&
              framesToRead > missingFrames
            ) {
              // Fade out the last available frames toward 0.0, then zero-fill missing tail.
              const fadeFrames =
                declickFrames < framesToRead ? declickFrames : framesToRead;
              const fadeStart = framesToRead - fadeFrames;
              const lastFade = fadeFrames - 1;
              if (useCosine) {
                if (lastFade > 0) {
                  const step = Math.PI / fadeFrames;
                  const halfStep = 0.5 * step;
                  const s = Math.sin(halfStep);
                  const k = 4.0 * s * s;
                  let c = 1.0 - 0.5 * k;
                  let delta = -0.5 * k - k * c;
                  for (let i = 0; i < lastFade; i++) {
                    const rampGain = 0.5 * (1.0 + c);
                    const base = (fadeStart + i) * channels;
                    for (let ch = 0; ch < channels; ch++) {
                      output[base + ch] *= rampGain;
                    }
                    if (
                      (i & COS_RECURRENCE_RESYNC_MASK) === COS_RECURRENCE_RESYNC_MASK &&
                      i + 1 < lastFade
                    ) {
                      const nextTheta = (i + 2) * step;
                      c = Math.cos(nextTheta);
                      delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
                    } else {
                      c += delta;
                      delta -= k * c;
                    }
                  }
                }
                const endBase = (fadeStart + lastFade) * channels;
                for (let ch = 0; ch < channels; ch++) {
                  output[endBase + ch] = 0.0;
                }
                this._fastCosineRamps += 1;
              } else {
                const invFade = 1.0 / fadeFrames;
                for (let i = 0; i < fadeFrames; i++) {
                  const rampGain = (lastFade - i) * invFade;
                  const base = (fadeStart + i) * channels;
                  for (let ch = 0; ch < channels; ch++) {
                    output[base + ch] *= rampGain;
                  }
                }
              }
              output.fill(0, framesToRead * channels, requestedFrames * channels);
            } else {
              // Ramp smoothly from the last output sample toward 0.0 over the missing tail.
              if (framesToRead > 0) {
                const lastBase = (framesToRead - 1) * channels;
                for (let c = 0; c < channels; c++) {
                  this._lastOutputFrame[c] = output[lastBase + c];
                }
              }
              const fadeFrames =
                declickFrames < missingFrames ? declickFrames : missingFrames;
              const lastFade = fadeFrames - 1;
              if (fadeFrames === 1) {
                const base = framesToRead * channels;
                for (let ch = 0; ch < channels; ch++) {
                  output[base + ch] = this._lastOutputFrame[ch] * 0.5;
                }
              } else if (useCosine) {
                const step = Math.PI / fadeFrames;
                const halfStep = 0.5 * step;
                const s = Math.sin(halfStep);
                const k = 4.0 * s * s;
                let c = 1.0 - 0.5 * k;
                let delta = -0.5 * k - k * c;
                for (let i = 0; i < lastFade; i++) {
                  const rampGain = 0.5 * (1.0 + c);
                  const base = (framesToRead + i) * channels;
                  for (let ch = 0; ch < channels; ch++) {
                    output[base + ch] = this._lastOutputFrame[ch] * rampGain;
                  }
                  if (
                    (i & COS_RECURRENCE_RESYNC_MASK) === COS_RECURRENCE_RESYNC_MASK &&
                    i + 1 < lastFade
                  ) {
                    const nextTheta = (i + 2) * step;
                    c = Math.cos(nextTheta);
                    delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
                  } else {
                    c += delta;
                    delta -= k * c;
                  }
                }
                const endBase = (framesToRead + lastFade) * channels;
                for (let ch = 0; ch < channels; ch++) {
                  output[endBase + ch] = 0.0;
                }
                this._fastCosineRamps += 1;
              } else {
                const invFade = 1.0 / fadeFrames;
                for (let i = 0; i < fadeFrames; i++) {
                  const rampGain = (lastFade - i) * invFade;
                  const base = (framesToRead + i) * channels;
                  for (let ch = 0; ch < channels; ch++) {
                    output[base + ch] = this._lastOutputFrame[ch] * rampGain;
                  }
                }
              }
              if (framesToRead + fadeFrames < requestedFrames) {
                output.fill(
                  0,
                  (framesToRead + fadeFrames) * channels,
                  requestedFrames * channels
                );
              }
            }

            this._fadeOutCount += 1;
            this._hasActiveAudio = false;
            this._lastOutputFrame.fill(0);
          } else {
            output.fill(0, framesToRead * channels, requestedFrames * channels);
          }
        } else {
          output.fill(0, framesToRead * channels, requestedFrames * channels);
        }

        this._inUnderrun = true;
        this._consecutiveUnderruns += 1;
        if (this._consecutiveUnderruns > this._maxConsecutiveUnderruns) {
          this._maxConsecutiveUnderruns = this._consecutiveUnderruns;
        }
        Atomics.add(this._ctrl, HDR_UNDERRUNS, 1);
        Atomics.add(this._ctrl, HDR_UNDERRUN_FRAMES, missingFrames);
      } else {
        // Steady-state full pull: record trailing frame without mutating output.
        this._consecutiveUnderruns = 0;
        const lastBase = (requestedFrames - 1) * channels;
        for (let c = 0; c < channels; c++) {
          this._lastOutputFrame[c] = output[lastBase + c];
        }
        this._hasActiveAudio = true;
      }

      return framesToRead;
    }

    /**
     * Pulls interleaved ring audio through an internal reusable scratch buffer
     * and de-interleaves directly into caller-provided planar channel buffers
     * (`Float32Array[]`) with zero per-call TypedArray allocations in steady state.
     *
     * @param {Float32Array[]} planarOutputs One Float32Array per channel.
     * @param {number} [frameCount]
     * @param {number} [gainOverride]
     * @returns {number} Number of real frames pulled before any underrun fill.
     */
    pullPlanarSamples(planarOutputs, frameCount, gainOverride) {
      const channels = this._channels;
      if (!Array.isArray(planarOutputs) || planarOutputs.length < channels) {
        throw new TypeError(
          `pullPlanarSamples requires an array of at least ${channels} Float32Array channel buffers`
        );
      }
      let minLen = Infinity;
      for (let c = 0; c < channels; c++) {
        const ch = planarOutputs[c];
        if (!(ch instanceof Float32Array)) {
          throw new TypeError(`pullPlanarSamples channel ${c} must be a Float32Array`);
        }
        if (ch.length < minLen) {
          minLen = ch.length;
        }
      }
      const requestedFrames =
        frameCount !== undefined
          ? Math.min(sanitizeNonNegInt(frameCount, 0), minLen)
          : minLen;
      if (requestedFrames <= 0) {
        return 0;
      }

      const neededSamples = requestedFrames * channels;
      if (!this._pullScratch || this._pullScratch.length < neededSamples) {
        this._pullScratch = new Float32Array(neededSamples);
        this._pullScratchView = this._pullScratch;
        this._pullScratchViewLen = neededSamples;
        this._pullScratchAllocations += 1;
      } else {
        this._pullScratchReuses += 1;
      }

      const framesRead = this.pullSamples(this._pullScratch, gainOverride, requestedFrames);
      deinterleavePcm(this._pullScratch, channels, requestedFrames, planarOutputs);
      return framesRead;
    }

    /**
     * Atomically advances the read cursor by up to `maxFrames` available frames
     * without allocating or copying samples and without recording underruns.
     * Keeps guest audio producers unblocked when Web Audio is suspended or absent.
     *
     * @param {number} [maxFrames] Maximum frames to discard (defaults to all available).
     * @returns {number} Actual number of frames discarded.
     */
    discardFrames(maxFrames) {
      const cap = this._capacityFrames;
      if (cap <= 0) {
        return 0;
      }

      const read = this._normalizeCursor(Atomics.load(this._ctrl, HDR_READ_CURSOR));
      const write = this._normalizeCursor(Atomics.load(this._ctrl, HDR_WRITE_CURSOR));
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }
      if (used <= 0) {
        return 0;
      }
      if (used > cap) {
        used = cap;
      }

      let toDiscard = used;
      if (maxFrames !== undefined && maxFrames !== null) {
        if (typeof maxFrames !== 'number' || !Number.isFinite(maxFrames) || maxFrames <= 0) {
          return 0;
        }
        const floored = Math.floor(maxFrames);
        if (floored <= 0) {
          return 0;
        }
        toDiscard = floored < used ? floored : used;
      }

      let nextRead = read + toDiscard;
      if (nextRead >= this._modulus) {
        nextRead -= this._modulus;
      }
      Atomics.store(this._ctrl, HDR_READ_CURSOR, nextRead);
      this._recordFillTransition(used, Math.max(0, used - toDiscard));
      return toDiscard;
    }

    reset() {
      Atomics.store(this._ctrl, HDR_WRITE_CURSOR, 0);
      Atomics.store(this._ctrl, HDR_READ_CURSOR, 0);
      Atomics.store(this._ctrl, HDR_UNDERRUNS, 0);
      Atomics.store(this._ctrl, HDR_UNDERRUN_FRAMES, 0);
      Atomics.store(this._ctrl, HDR_OVERFLOWS, 0);
      Atomics.store(this._ctrl, HDR_DROPPED_FRAMES, 0);
      this._lastOutputFrame.fill(0);
      this._hasActiveAudio = false;
      this._inUnderrun = false;
      this._fadeInPos = 0;
      this._fadeInTotal = 0;
      this._fadeOutCount = 0;
      this._fadeInCount = 0;
      this._underrunRecoveries = 0;
      this._consecutiveUnderruns = 0;
      this._maxConsecutiveUnderruns = 0;
      this._peakFillFrames = 0;
      this._minFillFrames = Infinity;
      this._fillFramesSum = 0;
      this._fillSamples = 0;
      this._lastObservedFill = null;
      this._packetsReceived = 0;
      this._framesDecoded = 0;
      this._droppedPackets = 0;
      this._lastPushDroppedFrames = 0;
      this._lastPushOverflowed = false;
      this._fastCosineRamps = 0;
      this._pullScratchReuses = 0;
      if (this._pushScratch) {
        this._pushScratch.resetStats();
      }
    }

    getStats() {
      const overflows = this.overflows;
      const underruns = this.underruns;
      const availableFrames = this.availableFrames();
      const bufferFillPct =
        this._capacityFrames > 0
          ? Math.min(100, Math.max(0, (availableFrames / this._capacityFrames) * 100))
          : 0.0;
      return {
        capacityFrames: this._capacityFrames,
        channels: this._channels,
        availableFrames,
        freeFrames: this._capacityFrames - availableFrames,
        bufferFillPct,
        volume: this._volume,
        muted: this._muted,
        effectiveGain: this.effectiveGain(),
        declickFrames: this._declickFrames,
        fadeCurve: this._fadeCurve,
        overflowPolicy: this._overflowPolicy,
        maxFillRatio: this._maxFillRatio,
        underrunRecoveryFrames: this._underrunRecoveryFrames,
        packetsReceived: this._packetsReceived,
        framesDecoded: this._framesDecoded,
        droppedPackets: this._droppedPackets,
        underruns,
        underrunCount: underruns,
        underrunFrames: this.underrunFrames,
        underrunRecoveries: this._underrunRecoveries,
        consecutiveUnderruns: this._consecutiveUnderruns,
        maxConsecutiveUnderruns: this._maxConsecutiveUnderruns,
        fadeOutCount: this._fadeOutCount,
        fadeInCount: this._fadeInCount,
        fastCosineRamps: this._fastCosineRamps,
        scratchBufferReuses:
          this._pullScratchReuses + (this._pushScratch ? this._pushScratch.reuses : 0),
        peakFillFrames: this.peakFillFrames,
        minFillFrames: this.minFillFrames,
        avgFillRatio: this.avgFillRatio,
        overflows,
        overruns: overflows,
        droppedFrames: this.droppedFrames,
      };
    }
  }

  /**
   * Audio mixer worker controller managing initialization, SAB ring buffer I/O,
   * bounded headless/autoplay-suspended draining, payload hardening, master
   * volume/mute state, and sanitized `audio-stats` telemetry.
   */
  class AudioMixerWorker {
    constructor(postMessageFn = null, options = {}) {
      this._postMessage =
        typeof postMessageFn === 'function' ? postMessageFn : () => {};
      const opts = options && typeof options === 'object' ? options : {};

      this._now = typeof opts.now === 'function' ? opts.now : () => Date.now();
      this._setInterval =
        typeof opts.setIntervalFn === 'function'
          ? opts.setIntervalFn
          : typeof setInterval === 'function'
            ? (fn, ms) => setInterval(fn, ms)
            : null;
      this._clearInterval =
        typeof opts.clearIntervalFn === 'function'
          ? opts.clearIntervalFn
          : typeof clearInterval === 'function'
            ? (id) => clearInterval(id)
            : null;

      this._maxPcmBytes = sanitizePositiveInt(
        opts.maxPcmBytes,
        DEFAULT_MAX_PCM_BYTES,
        Float32Array.BYTES_PER_ELEMENT,
        64 * 1024 * 1024
      );
      this._declickFrames = sanitizeDeclickFrames(
        opts.declickFrames,
        DEFAULT_DECLICK_FRAMES
      );
      this._fadeCurve = sanitizeFadeCurve(
        opts.fadeCurve,
        opts.cosineRamp === true ? 'cosine' : DEFAULT_FADE_CURVE
      );
      this._guestFadeCurve = sanitizeFadeCurve(
        opts.guestFadeCurve !== undefined ? opts.guestFadeCurve : opts.fadeCurve,
        'cosine'
      );
      this._targetLatencyMs = clampTargetLatencyMs(
        opts.targetLatencyMs,
        DEFAULT_TARGET_LATENCY_MS
      );
      this._targetLatencyConfigured = opts.targetLatencyMs !== undefined;
      this._autoDrain = opts.autoDrain !== undefined ? Boolean(opts.autoDrain) : true;
      const defaultDrainInterval = this._targetLatencyConfigured
        ? sanitizePositiveInt(
            Math.floor(this._targetLatencyMs / 2),
            DEFAULT_DRAIN_INTERVAL_MS,
            MIN_DRAIN_INTERVAL_MS,
            MAX_DRAIN_INTERVAL_MS
          )
        : DEFAULT_DRAIN_INTERVAL_MS;
      this._drainIntervalMs = sanitizePositiveInt(
        opts.drainIntervalMs,
        defaultDrainInterval,
        MIN_DRAIN_INTERVAL_MS,
        MAX_DRAIN_INTERVAL_MS
      );
      this._explicitDrainInterval = opts.drainIntervalMs !== undefined;
      this._activeTimerIntervalMs = null;
      this._explicitMaxDrainFramesPerTick =
        typeof opts.maxDrainFramesPerTick === 'number' &&
        Number.isFinite(opts.maxDrainFramesPerTick) &&
        opts.maxDrainFramesPerTick > 0
          ? Math.floor(opts.maxDrainFramesPerTick)
          : null;

      this._hasExplicitOutputState =
        opts.outputState !== undefined || opts.headless !== undefined;
      if (opts.outputState !== undefined) {
        this.outputState = normalizeOutputState(opts.outputState, OUTPUT_STATES.HEADLESS_DRAIN);
      } else if (opts.headless === true) {
        this.outputState = OUTPUT_STATES.HEADLESS_DRAIN;
      } else {
        this.outputState = detectDefaultOutputState();
      }

      this._softClip = opts.softClip !== undefined ? Boolean(opts.softClip) : false;
      const rawKnee =
        opts.limiterKnee !== undefined
          ? opts.limiterKnee
          : opts.softClipKnee !== undefined
            ? opts.softClipKnee
            : DEFAULT_SOFT_CLIP_KNEE;
      this._limiterKnee =
        typeof rawKnee === 'number' && Number.isFinite(rawKnee) && rawKnee >= 0 && rawKnee < 1.0
          ? rawKnee
          : DEFAULT_SOFT_CLIP_KNEE;

      this._overflowPolicy = normalizeOverflowPolicy(
        opts.overflowPolicy,
        DEFAULT_OVERFLOW_POLICY
      );
      this._maxFillRatio = sanitizeFillRatio(opts.maxFillRatio, DEFAULT_MAX_FILL_RATIO);
      this._underrunRecoveryFrames = sanitizeNonNegInt(
        opts.underrunRecoveryFrames,
        DEFAULT_UNDERRUN_RECOVERY_FRAMES
      );

      this._scratchPool = new PcmScratchPool({
        maxSamples: Math.max(1, Math.floor(this._maxPcmBytes / Float32Array.BYTES_PER_ELEMENT)),
      });
      this._extractResult = {
        ok: false,
        pcm: null,
        copiedFromSab: false,
        sanitizedSamples: 0,
        error: '',
      };
      this._extractOpts = {
        maxPcmBytes: this._maxPcmBytes,
        scratch: this._scratchPool,
        outResult: this._extractResult,
      };
      this._dmStats = {
        downmixedFrames: 0,
        clippedSamples: 0,
        nanSanitizedSamples: 0,
        peakLevel: 0.0,
      };
      this._downmixOpts = {
        centerGain: undefined,
        surroundGain: undefined,
        lfeGain: undefined,
        softClip: true,
        limiterKnee: this._limiterKnee,
        scratch: this._scratchPool,
        stats: this._dmStats,
      };
      this._rsStats = {
        resampledFrames: 0,
        nanSanitizedSamples: 0,
        clippedSamples: 0,
      };
      this._pushOpts = {
        frameCount: 0,
        overflowPolicy: this._overflowPolicy,
        maxFillFrames: undefined,
      };
      this._pullPlanarResult = {
        framesRead: 0,
        channels: [],
        frameCount: 0,
      };

      this.initialized = false;
      this.sampleRate = DEFAULT_SAMPLE_RATE;
      this.channels = DEFAULT_CHANNELS;
      this.volume = 1.0;
      this.muted = false;
      this.ring = null;
      this._hasExternalRingSab = false;
      this._stalledFullTicks = 0;
      this._stalledFullSinceMs = null;
      this._lastObservedReadCursor = null;
      this.totalPushedFrames = 0;
      this.totalPulledFrames = 0;
      this.framesDrained = 0;
      this._packetsReceived = 0;
      this._framesDecoded = 0;
      this._droppedPackets = 0;
      this._drainTicks = 0;
      this._overruns = 0;
      this._rejectedPackets = 0;
      this._sabCopiedPackets = 0;
      this._sanitizedSamples = 0;
      this._downmixedFrames = 0;
      this._resampledFrames = 0;
      this._clippedSamples = 0;
      this._softClippedSamples = 0;
      this._zeroAllocDrains = 0;
      this._fastCosineRamps = 0;
      this._activeVoicePeak = 0;
      this._resamplerState = { phase: 0.0 };
      this._lastResampleSrcRate = null;
      this._drainTimer = null;
      this._lastDrainMs = null;

      // W235: Attached R40 host stereo PCM ring reader state & telemetry.
      this._guestRing = null;
      this._guestRingSoftClip = true;
      this._guestRingLimiterKnee = this._limiterKnee;
      this._guestRingQuantumFrames = GUEST_PCM_RING_DEFAULT_QUANTUM_FRAMES;
      this._guestRingExplicitQuantum = false;
      this._guestRingAutoPump = false;
      this._guestResamplerState = { phase: 0.0 };
      this._guestFramesDrained = 0;
      this._guestUnderruns = 0;
      this._guestUnderrunFrames = 0;
      this._guestOverruns = 0;
      this._guestOverrunFrames = 0;
      this._guestPeakLevel = 0.0;
      this._guestUnderrunFadeCount = 0;
      this._guestRecoveryFadeCount = 0;
      this._guestTrimmedFrames = 0;
      this._guestTrimCount = 0;
      this._directResampleLastFrame = new Float32Array(GUEST_PCM_RING_CHANNELS);
      this._directResampleHasActive = false;
      this._directResampleInUnderrun = false;
      this._directResampleFadeInPos = 0;
      this._directResampleFadeInTotal = 0;
      this._inDrainTick = false;

      // O43 (#1028/#663): Live BroadcastChannel audio-stats telemetry state.
      this._telemetryBroadcast =
        opts.telemetryBroadcast !== undefined
          ? Boolean(opts.telemetryBroadcast)
          : opts.BroadcastChannelCtor !== undefined || opts.broadcastChannelName !== undefined;
      this._broadcastChannelName =
        typeof opts.broadcastChannelName === 'string' && opts.broadcastChannelName.length > 0
          ? opts.broadcastChannelName
          : PROGRESS_CHANNEL_NAME;
      this._BroadcastChannelCtor =
        typeof opts.BroadcastChannelCtor === 'function' ? opts.BroadcastChannelCtor : null;
      this._telemetryBroadcastIntervalMs = sanitizePositiveInt(
        opts.telemetryBroadcastIntervalMs,
        DEFAULT_TELEMETRY_BROADCAST_INTERVAL_MS,
        MIN_TELEMETRY_BROADCAST_INTERVAL_MS,
        MAX_TELEMETRY_BROADCAST_INTERVAL_MS
      );
      this._lastTelemetryBroadcastMs = null;
      this._lastBroadcastFramesDrained = -1;
      this._lastBroadcastGuestFramesDrained = -1;
      this._firstGuestDrainBroadcasted = false;
      this._telemetryBroadcastCount = 0;
      // W365 (#663): Reuse one persistent BroadcastChannel('game-progress') instead of
      // constructing/closing a new Mojo IPC channel on every audio-stats broadcast.
      this._telemetryChannel = null;
      this._telemetryChannelCtor = null;
      this._telemetryChannelName = null;
      this._snapshotOpts = { timestamp: 0 };
      this._drainGuestOutResult = {
        framesDrained: 0,
        requestedFrames: 0,
        pushedFrames: 0,
        droppedFrames: 0,
        underrun: false,
        underrunFrames: 0,
        overrun: false,
        overrunDroppedFrames: 0,
        peakLevel: 0.0,
        zeroAlloc: true,
        sanitizedNonFiniteSamples: 0,
        softClippedSamples: 0,
      };
      this._resampleOpts = {
        srcFrames: 0,
        state: null,
        maxOutputFrames: 0,
        scratch: this._scratchPool,
        stats: this._rsStats,
        softClip: true,
        limiterKnee: this._limiterKnee,
      };
      this._onDrainTimerTick = () => {
        try {
          if (
            this.outputState === OUTPUT_STATES.HEADLESS_DRAIN ||
            this.outputState === OUTPUT_STATES.SUSPENDED
          ) {
            this.drainTick();
          } else if (this._guestRing !== null && this._guestRingAutoPump) {
            this._pumpActiveGuestTick();
          }
        } catch (_err) {
          // Never throw from background drain timer.
        }
      };
      this._guestReadOpts = {
        softClip: true,
        limiterKnee: this._limiterKnee,
        fillSilence: true,
        declickFrames: this._declickFrames,
        fadeCurve: this._guestFadeCurve,
      };
      this._guestReadScratch = {
        framesRead: 0,
        underrun: false,
        underrunFrames: 0,
        overrun: false,
        overrunDroppedFrames: 0,
        clippedSamples: 0,
        softClippedSamples: 0,
        sanitizedSamples: 0,
        underrunFade: false,
        recoveryFade: false,
        peakLevel: 0.0,
      };

      if (typeof opts.initialScratchFrames === 'number' && opts.initialScratchFrames > 0) {
        const initFrames = sanitizeNonNegInt(opts.initialScratchFrames, 0);
        if (initFrames > 0) {
          this._scratchPool.acquirePlanar(this.channels, initFrames);
        }
      }
    }

    _safePostMessage(payload, transfer) {
      const safeTransfer = sanitizeTransferList(transfer);
      try {
        if (safeTransfer.length > 0) {
          this._postMessage(payload, safeTransfer);
        } else {
          this._postMessage(payload);
        }
      } catch (_err) {
        if (safeTransfer.length > 0) {
          try {
            this._postMessage(payload);
          } catch (_fallbackErr) {
            // Never allow postMessage errors to crash the audio worker.
          }
        }
      }
    }

    _defaultMaxDrainFramesPerTick() {
      return Math.max(128, Math.ceil((this.sampleRate * MAX_DRAIN_ELAPSED_MS) / 1000));
    }

    _effectiveMaxDrainFramesPerTick() {
      if (this._explicitMaxDrainFramesPerTick !== null) {
        return this._explicitMaxDrainFramesPerTick;
      }
      return this._defaultMaxDrainFramesPerTick();
    }

    _shouldDrain() {
      return (
        this.outputState === OUTPUT_STATES.HEADLESS_DRAIN ||
        this.outputState === OUTPUT_STATES.SUSPENDED
      );
    }

    _shouldRunTimer() {
      return this._shouldDrain() || (this._guestRing !== null && this._guestRingAutoPump);
    }

    _resolveBroadcastChannelCtor() {
      if (typeof this._BroadcastChannelCtor === 'function') {
        return this._BroadcastChannelCtor;
      }
      if (typeof self !== 'undefined' && typeof self.BroadcastChannel === 'function') {
        return self.BroadcastChannel;
      }
      if (typeof BroadcastChannel === 'function') {
        return BroadcastChannel;
      }
      return null;
    }

    _getTelemetryChannel() {
      const ctor = this._resolveBroadcastChannelCtor();
      if (
        !ctor ||
        typeof this._broadcastChannelName !== 'string' ||
        this._broadcastChannelName.length === 0
      ) {
        return null;
      }
      if (
        this._telemetryChannel !== null &&
        this._telemetryChannelCtor === ctor &&
        this._telemetryChannelName === this._broadcastChannelName
      ) {
        return this._telemetryChannel;
      }
      this._closeTelemetryChannel();
      try {
        const ch = new ctor(this._broadcastChannelName);
        if (!ch || typeof ch.postMessage !== 'function') {
          if (ch && typeof ch.close === 'function') {
            try {
              ch.close();
            } catch (_closeErr) {
              // Ignore BroadcastChannel close errors.
            }
          }
          return null;
        }
        this._telemetryChannel = ch;
        this._telemetryChannelCtor = ctor;
        this._telemetryChannelName = this._broadcastChannelName;
        return ch;
      } catch (_err) {
        return null;
      }
    }

    _closeTelemetryChannel() {
      const ch = this._telemetryChannel;
      this._telemetryChannel = null;
      this._telemetryChannelCtor = null;
      this._telemetryChannelName = null;
      if (ch && typeof ch.close === 'function') {
        try {
          ch.close();
        } catch (_err) {
          // Ignore BroadcastChannel close errors during teardown/reset.
        }
      }
    }

    _postTelemetryMessage(payload) {
      const ch = this._getTelemetryChannel();
      if (!ch) {
        return false;
      }
      try {
        ch.postMessage(payload);
        return true;
      } catch (_err) {
        this._closeTelemetryChannel();
        return false;
      }
    }

    _publishAudioStatsInternal(force, milestone, nowMs) {
      if (!this._telemetryBroadcast) {
        return false;
      }
      const framesChanged =
        this.framesDrained !== this._lastBroadcastFramesDrained ||
        this._guestFramesDrained !== this._lastBroadcastGuestFramesDrained;
      if (!force && !milestone) {
        if (!framesChanged) {
          return false;
        }
        if (
          this._lastTelemetryBroadcastMs !== null &&
          nowMs - this._lastTelemetryBroadcastMs < this._telemetryBroadcastIntervalMs
        ) {
          return false;
        }
      }
      if (!this._getTelemetryChannel()) {
        return false;
      }
      this._snapshotOpts.timestamp = nowMs;
      const snapshot = buildAudioTelemetrySnapshot(this, this._snapshotOpts);
      const sent = this._postTelemetryMessage(snapshot);
      if (!sent) {
        return false;
      }
      if (milestone && this._guestFramesDrained > 0) {
        const rate = this._guestRing ? this._guestRing.sampleRate : this.sampleRate;
        const peak = Number.isFinite(this._guestPeakLevel) ? this._guestPeakLevel.toFixed(4) : '0.0000';
        this._postTelemetryMessage({
          label: `[audio] first_guest_pcm_drain guest_frames_drained=${this._guestFramesDrained} sampleRate=${rate} peak=${peak}`,
        });
      }
      this._lastTelemetryBroadcastMs = nowMs;
      this._lastBroadcastFramesDrained = this.framesDrained;
      this._lastBroadcastGuestFramesDrained = this._guestFramesDrained;
      this._telemetryBroadcastCount += 1;
      return true;
    }

    _publishAudioStats(options = {}) {
      if (!this._telemetryBroadcast) {
        return false;
      }
      const force = Boolean(options && options.force);
      const milestone = Boolean(options && options.milestone);
      const nowMs =
        options && typeof options.nowMs === 'number' && Number.isFinite(options.nowMs)
          ? options.nowMs
          : this._now();
      return this._publishAudioStatsInternal(force, milestone, nowMs);
    }

    _maybeBroadcastAudioStats(nowMs) {
      if (!this._telemetryBroadcast) {
        return false;
      }
      if (this._guestFramesDrained > 0 && !this._firstGuestDrainBroadcasted) {
        this._firstGuestDrainBroadcasted = true;
        return this._publishAudioStatsInternal(true, true, nowMs);
      }
      if (
        this.framesDrained === this._lastBroadcastFramesDrained &&
        this._guestFramesDrained === this._lastBroadcastGuestFramesDrained
      ) {
        return false;
      }
      if (
        this._lastTelemetryBroadcastMs !== null &&
        nowMs - this._lastTelemetryBroadcastMs < this._telemetryBroadcastIntervalMs
      ) {
        return false;
      }
      return this._publishAudioStatsInternal(false, false, nowMs);
    }

    _effectiveTimerIntervalMs() {
      if (this._explicitDrainInterval || this._targetLatencyConfigured) {
        return this._drainIntervalMs;
      }
      if (
        this.outputState === OUTPUT_STATES.ACTIVE &&
        this._guestRing !== null &&
        this._guestRingAutoPump
      ) {
        return DEFAULT_ACTIVE_PUMP_INTERVAL_MS;
      }
      return this._drainIntervalMs;
    }

    _startDrainTimer() {
      this._stopDrainTimer();
      if (!this.initialized || !this.ring || !this._autoDrain || !this._shouldRunTimer()) {
        return;
      }
      if (!this._setInterval) {
        return;
      }
      const intervalMs = this._effectiveTimerIntervalMs();
      this._lastDrainMs = this._now();
      const handle = this._setInterval(this._onDrainTimerTick, intervalMs);
      if (handle && typeof handle.unref === 'function') {
        handle.unref();
      }
      this._drainTimer = handle;
      this._activeTimerIntervalMs = intervalMs;
    }

    _stopDrainTimer() {
      if (this._drainTimer !== null && this._clearInterval) {
        try {
          this._clearInterval(this._drainTimer);
        } catch (_err) {
          // Ignore clearInterval failures during teardown.
        }
      }
      this._drainTimer = null;
      this._activeTimerIntervalMs = null;
    }

    setSampleRate(sampleRate) {
      if (
        typeof sampleRate !== 'number' ||
        !Number.isFinite(sampleRate) ||
        sampleRate < MIN_SAMPLE_RATE ||
        sampleRate > MAX_SAMPLE_RATE
      ) {
        return this.sampleRate;
      }
      const nextRate = Math.floor(sampleRate);
      if (this._guestRing) {
        const rateRatio = this._guestRing.sampleRate / nextRate;
        if (rateRatio > MAX_RESAMPLE_RATIO || 1.0 / rateRatio > MAX_RESAMPLE_RATIO) {
          throw new RangeError(
            `Guest PCM ring sampleRate ratio (${this._guestRing.sampleRate} -> ${nextRate}) exceeds ${MAX_RESAMPLE_RATIO}x guard`
          );
        }
      }
      if (nextRate !== this.sampleRate) {
        this.sampleRate = nextRate;
        this._guestResamplerState.phase = 0.0;
        this._resamplerState.phase = 0.0;
        this._lastResampleSrcRate = null;
      }
      return this.sampleRate;
    }

    /**
     * Attaches (or swaps) the shared AudioRingBuffer SAB shared with the main-thread
     * AudioEngine / AudioWorkletProcessor WITHOUT losing an already-attached guest
     * PCM ring (`playgta6_audio_ring`) or wiping guest drain telemetry.
     *
     * Cleanly transitions `outputState` from `'headless-drain'` to `'active'`
     * unless an explicit `outputState` / `audioContextState` / `headless: true`
     * option is supplied.
     *
     * @param {SharedArrayBuffer} ringSab
     * @param {object} [options]
     */
    attachRingSab(ringSab, options = {}) {
      if (typeof SharedArrayBuffer === 'undefined' || !isSharedArrayBuffer(ringSab)) {
        throw new TypeError('Invalid ringSab: expected SharedArrayBuffer');
      }
      const opts = options && typeof options === 'object' ? options : {};
      const nextChannels = sanitizePositiveInt(
        opts.channels,
        this.channels || DEFAULT_CHANNELS,
        1,
        MAX_CHANNELS
      );
      const nextRing = new AudioRingBuffer(ringSab, {
        channels: nextChannels,
        volume: this.volume,
        muted: this.muted,
        declickFrames: this._declickFrames,
        fadeCurve: this._fadeCurve,
        overflowPolicy: this._overflowPolicy,
        maxFillRatio: this._maxFillRatio,
        underrunRecoveryFrames: this._underrunRecoveryFrames,
      });

      if (opts.sampleRate !== undefined) {
        this.setSampleRate(opts.sampleRate);
      }

      const prevState = this.outputState;
      this.ring = nextRing;
      this.channels = nextRing.channels;
      this.initialized = true;
      this._hasExternalRingSab = true;
      this._stalledFullTicks = 0;
      this._stalledFullSinceMs = null;
      this._lastObservedReadCursor = null;

      if (typeof opts.outputState === 'string') {
        this.outputState = normalizeOutputState(opts.outputState, OUTPUT_STATES.ACTIVE);
      } else if (typeof opts.state === 'string') {
        this.outputState = normalizeOutputState(opts.state, OUTPUT_STATES.ACTIVE);
      } else if (typeof opts.audioContextState === 'string') {
        this.outputState = normalizeOutputState(opts.audioContextState, OUTPUT_STATES.ACTIVE);
      } else if (opts.headless === true) {
        this.outputState = OUTPUT_STATES.HEADLESS_DRAIN;
      } else if (this.outputState === OUTPUT_STATES.HEADLESS_DRAIN) {
        this.outputState = OUTPUT_STATES.ACTIVE;
      }

      if (
        this.outputState === OUTPUT_STATES.ACTIVE &&
        (prevState !== OUTPUT_STATES.ACTIVE || opts.resyncBacklog === true) &&
        this._guestRing
      ) {
        this._guestResamplerState.phase = 0.0;
        const guestRate = this._guestRing.sampleRate;
        this._trimGuestBacklog(
          Math.round((guestRate * GUEST_RESUME_HIGH_WATERMARK_MS) / 1000),
          Math.round((guestRate * GUEST_RESUME_TARGET_WATERMARK_MS) / 1000),
          false
        );
      }

      if (this._shouldRunTimer()) {
        const desiredInterval = this._effectiveTimerIntervalMs();
        if (this._drainTimer === null || this._activeTimerIntervalMs !== desiredInterval) {
          this._startDrainTimer();
        }
      } else {
        this._stopDrainTimer();
      }

      this._publishAudioStatsInternal(true, false, this._now());

      return {
        attached: true,
        ringSabAttached: true,
        outputState: this.outputState,
        sampleRate: this.sampleRate,
        channels: this.channels,
        capacityFrames: this.ring.capacityFrames,
        guestRingAttached: this._guestRing !== null,
      };
    }

    _trimGuestBacklog(highWatermarkFrames, targetFrames, withCrossfade = true) {
      if (!this._guestRing) {
        return 0;
      }
      const avail = this._guestRing.availableFrames();
      if (avail <= highWatermarkFrames) {
        return 0;
      }
      const safeTarget = targetFrames < highWatermarkFrames ? targetFrames : highWatermarkFrames;
      const trimmed = this._guestRing.trimToFrames(
        safeTarget,
        withCrossfade ? GUEST_TRIM_CROSSFADE_FRAMES : 0
      );
      if (trimmed > 0) {
        this._guestTrimmedFrames += trimmed;
        this._guestTrimCount += 1;
      }
      return trimmed;
    }

    setOutputState(state, options = null) {
      if (options && typeof options === 'object' && options.sampleRate !== undefined) {
        this.setSampleRate(options.sampleRate);
      }
      const prevState = this.outputState;
      this.outputState = normalizeOutputState(state, OUTPUT_STATES.HEADLESS_DRAIN);
      const resyncRequested = Boolean(options && typeof options === 'object' && options.resyncBacklog);
      if (this.outputState !== prevState) {
        this._stalledFullTicks = 0;
        this._stalledFullSinceMs = null;
        this._lastObservedReadCursor = null;
      }
      if (
        this.outputState === OUTPUT_STATES.ACTIVE &&
        (prevState !== OUTPUT_STATES.ACTIVE || resyncRequested) &&
        this._guestRing
      ) {
        this._guestResamplerState.phase = 0.0;
        const guestRate = this._guestRing.sampleRate;
        this._trimGuestBacklog(
          Math.round((guestRate * GUEST_RESUME_HIGH_WATERMARK_MS) / 1000),
          Math.round((guestRate * GUEST_RESUME_TARGET_WATERMARK_MS) / 1000),
          false
        );
      }
      if (this._shouldRunTimer()) {
        const desiredInterval = this._effectiveTimerIntervalMs();
        if (this._drainTimer === null || this._activeTimerIntervalMs !== desiredInterval) {
          this._startDrainTimer();
        }
      } else {
        this._stopDrainTimer();
      }
      return this.outputState;
    }

    /**
     * Executes one time-paced bounded drain step in headless or suspended mode.
     * Advances the SAB ring read cursor by at most the real-time frame budget
     * for the elapsed interval (capped by maxDrainFramesPerTick).
     *
     * @param {number} [nowMs] Optional monotonic timestamp in ms.
     * @returns {number} Number of frames drained.
     */
    drainTick(nowMs) {
      if (!this.ring || !this._shouldDrain()) {
        return 0;
      }
      const now =
        typeof nowMs === 'number' && Number.isFinite(nowMs) ? nowMs : this._now();
      let rawElapsedMs;
      let elapsedMs;
      if (this._lastDrainMs === null) {
        rawElapsedMs = this._drainIntervalMs;
        elapsedMs = this._drainIntervalMs;
      } else {
        rawElapsedMs = now - this._lastDrainMs;
        elapsedMs = rawElapsedMs;
        if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
          rawElapsedMs = this._drainIntervalMs;
          elapsedMs = this._drainIntervalMs;
        } else if (elapsedMs > MAX_DRAIN_ELAPSED_MS) {
          elapsedMs = MAX_DRAIN_ELAPSED_MS;
        }
      }
      this._lastDrainMs = now;

      let timeBudgetFrames = Math.floor((this.sampleRate * elapsedMs) / 1000);
      if (elapsedMs > 0 && timeBudgetFrames <= 0) {
        timeBudgetFrames = 1;
      }
      const maxFrames = Math.min(timeBudgetFrames, this._effectiveMaxDrainFramesPerTick());
      let drained = 0;
      const prevInDrainTick = this._inDrainTick;
      this._inDrainTick = true;
      try {
        if (this._guestRing && this._guestRingAutoPump && maxFrames > 0) {
          const guestRate = this._guestRing.sampleRate;
          // Under background-tab timer throttling (e.g. Chrome 1 Hz -> rawElapsedMs > 100 ms),
          // 100 ms real-time drain per 1000 ms cannot keep up with the 65536-frame guest ring.
          // Trim stale guest-ring backlog directly so tab-return never plays >1 s behind video.
          if (rawElapsedMs > MAX_DRAIN_ELAPSED_MS) {
            const trimmed = this._trimGuestBacklog(
              Math.round((guestRate * GUEST_THROTTLED_DRAIN_HIGH_WATERMARK_MS) / 1000),
              Math.round((guestRate * GUEST_RESUME_TARGET_WATERMARK_MS) / 1000),
              false
            );
            drained += trimmed;
          }
          // Chunk by this.ring.capacityFrames (scaled by guestRate / targetRate if resampling)
          // so slow/throttled browser timer ticks never trigger spurious overruns.
          const targetRate = this.sampleRate;
          const guestBudget =
            guestRate !== targetRate
              ? Math.min(
                  Math.max(elapsedMs > 0 ? 1 : 0, Math.floor((guestRate * elapsedMs) / 1000)),
                  this._effectiveMaxDrainFramesPerTick()
                )
              : maxFrames;
          const chunkCap =
            guestRate !== targetRate
              ? Math.max(
                  1,
                  Math.floor(this.ring.capacityFrames * Math.min(1, guestRate / targetRate))
                )
              : Math.max(1, this.ring.capacityFrames);
          let remaining = guestBudget;
          while (remaining > 0) {
            const step = remaining <= chunkCap ? remaining : chunkCap;
            const pumpRes = this._drainGuestPcmRingInto(step, null, this._drainGuestOutResult);
            const discardLimit =
              pumpRes.pushedFrames > 0 ? pumpRes.pushedFrames : step;
            const stepDrained = this.ring.discardFrames(discardLimit);
            drained += stepDrained;
            remaining -= step;
            if (pumpRes.framesDrained < step && this.ring.availableFrames() === 0) {
              break;
            }
          }
        } else if (maxFrames > 0) {
          drained = this.ring.discardFrames(maxFrames);
        }
      } finally {
        this._inDrainTick = prevInDrainTick;
      }
      this._drainTicks += 1;
      this.framesDrained += drained;
      this._maybeBroadcastAudioStats(now);
      return drained;
    }

    /**
     * Active-mode guest PCM ring pump: drains available stereo frames (including
     * post-hitch bursts) from `playgta6_audio_ring` into `this.ring` up to free
     * space without discarding unread samples or throttling to 1x elapsed wall-clock
     * time, so AudioWorkletProcessor maintains a healthy jitter cushion.
     *
     * Bounds stale backlog via click-free consumer-side guest-ring trimming above
     * 250 ms and preserves strict SPSC safety on the output SAB ring by waiting for
     * a real dead-consumer stall (>= 250 ms stationary read cursor) before ever
     * falling back to producer-side eviction.
     *
     * @param {number} [nowMs] Optional monotonic timestamp in ms.
     * @returns {number} Number of guest stereo frames pumped.
     */
    _pumpActiveGuestTick(nowMs) {
      if (!this.ring || !this._guestRing || !this._guestRingAutoPump || this._shouldDrain()) {
        return 0;
      }
      const now =
        typeof nowMs === 'number' && Number.isFinite(nowMs) ? nowMs : this._now();
      const timerIntervalMs = this._effectiveTimerIntervalMs();
      let elapsedMs;
      if (this._lastDrainMs === null) {
        elapsedMs = timerIntervalMs;
      } else {
        elapsedMs = now - this._lastDrainMs;
        if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
          elapsedMs = timerIntervalMs;
        } else if (elapsedMs > MAX_DRAIN_ELAPSED_MS) {
          elapsedMs = MAX_DRAIN_ELAPSED_MS;
        }
      }
      this._lastDrainMs = now;

      const guestRate = this._guestRing.sampleRate;
      const guestHighWatermark = Math.max(
        1,
        Math.round((guestRate * GUEST_HIGH_WATERMARK_MS) / 1000)
      );
      const guestTargetWatermark = Math.max(
        1,
        Math.round((guestRate * GUEST_TARGET_WATERMARK_MS) / 1000)
      );
      // Trim extreme stale guest-ring backlog (>250 ms, e.g. after tab-switch or 1 s stall)
      // down to ~96 ms with a click-free 64-frame half-cosine crossfade before pumping.
      this._trimGuestBacklog(guestHighWatermark, guestTargetWatermark, true);

      const availGuestFrames = this._guestRing.availableFrames();
      if (availGuestFrames <= 0) {
        return 0;
      }

      const targetRate = this.sampleRate;
      const needsResample = guestRate !== targetRate;
      const tickCap = this._effectiveMaxDrainFramesPerTick();
      const maxScratchFrames = Math.max(
        1,
        Math.floor(this._scratchPool._maxSamples / GUEST_PCM_RING_CHANNELS)
      );
      const burstCap =
        this._explicitMaxDrainFramesPerTick !== null
          ? tickCap
          : Math.min(
              Math.max(tickCap, this.ring.capacityFrames),
              GUEST_PCM_RING_MAX_QUANTUM_FRAMES,
              maxScratchFrames
            );

      const maxFillFrames = this._effectiveMaxFillFrames();
      const currentUsed = this.ring.availableFrames();
      let free =
        maxFillFrames !== undefined
          ? Math.max(0, maxFillFrames - currentUsed)
          : this.ring.freeFrames();
      const curRead = Atomics.load(this.ring._ctrl, HDR_READ_CURSOR);
      if (free <= 0) {
        let shouldEvictOldest = false;
        if (this._overflowPolicy === OVERFLOW_POLICIES.DROP_OLDEST) {
          shouldEvictOldest = true;
        } else {
          if (this._lastObservedReadCursor === curRead) {
            this._stalledFullTicks += 1;
            if (this._stalledFullSinceMs === null) {
              this._stalledFullSinceMs = now;
            }
          } else {
            this._stalledFullTicks = 1;
            this._stalledFullSinceMs = now;
            this._lastObservedReadCursor = curRead;
          }
          const stalledMs =
            this._stalledFullSinceMs !== null ? now - this._stalledFullSinceMs : 0;
          if (
            stalledMs >= ACTIVE_STALL_EVICT_MS ||
            this._stalledFullTicks >= ACTIVE_STALL_EVICT_MAX_TICKS
          ) {
            shouldEvictOldest = true;
          }
        }
        if (!shouldEvictOldest) {
          return 0;
        }
        const outputStep = Math.min(
          this.ring.capacityFrames,
          Math.max(1, Math.floor((targetRate * Math.max(elapsedMs, 1)) / 1000)),
          tickCap
        );
        const evicted = this.ring.discardFrames(outputStep);
        if (evicted > 0) {
          this.framesDrained += evicted;
          this._lastObservedReadCursor = Atomics.load(this.ring._ctrl, HDR_READ_CURSOR);
        }
        const nextUsed = this.ring.availableFrames();
        free =
          maxFillFrames !== undefined
            ? Math.max(0, maxFillFrames - nextUsed)
            : this.ring.freeFrames();
        if (free <= 0) {
          return 0;
        }
      } else {
        this._stalledFullTicks = 0;
        this._stalledFullSinceMs = null;
        this._lastObservedReadCursor = curRead;
      }

      const maxGuestForFree = needsResample
        ? free > 2
          ? Math.floor((free - 2) * (guestRate / targetRate))
          : 0
        : free;
      if (maxGuestForFree <= 0) {
        return 0;
      }
      const totalTarget = Math.min(availGuestFrames, maxGuestForFree, burstCap);
      if (totalTarget <= 0) {
        return 0;
      }

      const chunkCap = Math.min(
        Math.max(this._guestRingQuantumFrames, GUEST_PCM_RING_ACTIVE_BURST_CHUNK_FRAMES),
        GUEST_PCM_RING_MAX_QUANTUM_FRAMES,
        maxScratchFrames
      );
      const prevInDrainTick = this._inDrainTick;
      this._inDrainTick = true;
      let pumped = 0;
      let remaining = totalTarget;
      try {
        while (remaining > 0) {
          const step = remaining <= chunkCap ? remaining : chunkCap;
          const pumpRes = this._drainGuestPcmRingInto(step, null, this._drainGuestOutResult);
          pumped += pumpRes.framesDrained;
          remaining -= step;
          if (pumpRes.framesDrained < step || pumpRes.droppedFrames > 0) {
            break;
          }
        }
      } finally {
        this._inDrainTick = prevInDrainTick;
      }
      if (pumped > 0) {
        this._maybeBroadcastAudioStats(now);
      }
      return pumped;
    }

    /**
     * Drains up to `maxFrames` available frames from the SAB ring while in
     * headless-drain or suspended mode.
     *
     * @param {number} [maxFrames]
     * @param {{ force?: boolean }} [options]
     * @returns {number} Number of frames drained.
     */
    drain(maxFrames, options = {}) {
      if (!this.ring) {
        return 0;
      }
      const force = Boolean(options && options.force);
      if (!force && !this._shouldDrain()) {
        return 0;
      }

      const capBound = this._effectiveMaxDrainFramesPerTick();
      let limit;
      if (typeof maxFrames === 'number' && Number.isFinite(maxFrames) && maxFrames > 0) {
        const requested = Math.floor(maxFrames);
        const upperBound =
          this._explicitMaxDrainFramesPerTick !== null
            ? capBound
            : Math.max(capBound, this.ring.capacityFrames);
        limit = Math.min(requested, upperBound);
      } else {
        limit = capBound;
      }

      const drained = this.ring.discardFrames(limit);
      this._drainTicks += 1;
      this.framesDrained += drained;
      return drained;
    }

    get declickFrames() {
      return this._declickFrames;
    }

    get fadeCurve() {
      return this._fadeCurve;
    }

    setDeclickFrames(frames) {
      this._declickFrames = sanitizeDeclickFrames(frames, this._declickFrames);
      if (this.ring) {
        this.ring.setDeclickFrames(this._declickFrames);
      }
      return this._declickFrames;
    }

    setFadeCurve(curve) {
      const next = sanitizeFadeCurve(curve, this._fadeCurve);
      this._fadeCurve = next;
      this._guestFadeCurve = next;
      if (this.ring) {
        this.ring.setFadeCurve(next);
      }
      return this._fadeCurve;
    }

    get targetLatencyMs() {
      return this._targetLatencyMs;
    }

    get targetLatencyFrames() {
      return Math.max(1, Math.round((this._targetLatencyMs / 1000) * this.sampleRate));
    }

    get drainWatermarkFrames() {
      const targetFrames = this.targetLatencyFrames;
      if (this.ring && this.ring.capacityFrames > 0) {
        return targetFrames < this.ring.capacityFrames ? targetFrames : this.ring.capacityFrames;
      }
      return targetFrames;
    }

    get drainIntervalMs() {
      return this._drainIntervalMs;
    }

    /**
     * Tunes the jitter/latency target in milliseconds (clamped to [10, 120] ms,
     * default 30 ms), updating the ring drain watermark, refill cadence, and
     * pre-allocated guest stereo drain scratch capacity cleanly.
     *
     * @param {number} ms
     * @returns {number} Active clamped target latency in ms.
     */
    setTargetLatencyMs(ms) {
      if (typeof ms !== 'number' || !Number.isFinite(ms)) {
        return this._targetLatencyMs;
      }
      const clamped = clampTargetLatencyMs(ms, this._targetLatencyMs);
      this._targetLatencyMs = clamped;
      this._targetLatencyConfigured = true;

      const nextInterval = sanitizePositiveInt(
        Math.floor(clamped / 2),
        DEFAULT_DRAIN_INTERVAL_MS,
        MIN_DRAIN_INTERVAL_MS,
        MAX_DRAIN_INTERVAL_MS
      );
      this._drainIntervalMs = nextInterval;

      const targetFrames = this.targetLatencyFrames;
      const maxScratchFrames = Math.max(
        1,
        Math.floor(this._scratchPool._maxSamples / GUEST_PCM_RING_CHANNELS)
      );
      const quantumCap = Math.min(GUEST_PCM_RING_MAX_QUANTUM_FRAMES, maxScratchFrames);

      if (!this._guestRingExplicitQuantum) {
        this._guestRingQuantumFrames = Math.min(
          Math.max(128, Math.floor(targetFrames / 2)),
          quantumCap
        );
      }

      if (this._guestRing) {
        const prewarmFrames = Math.min(
          Math.max(this._guestRingQuantumFrames, this.drainWatermarkFrames),
          quantumCap
        );
        this._scratchPool.acquireFloat32Backing(
          'guestDrain',
          prewarmFrames * GUEST_PCM_RING_CHANNELS
        );
      }

      if (this._drainTimer !== null) {
        this._startDrainTimer();
      }

      return this._targetLatencyMs;
    }

    get overflowPolicy() {
      return this._overflowPolicy;
    }

    setOverflowPolicy(policy) {
      this._overflowPolicy = normalizeOverflowPolicy(policy, this._overflowPolicy);
      if (this.ring) {
        this.ring.setOverflowPolicy(this._overflowPolicy);
      }
      return this._overflowPolicy;
    }

    get maxFillRatio() {
      return this._maxFillRatio;
    }

    setMaxFillRatio(ratio) {
      this._maxFillRatio = sanitizeFillRatio(ratio, this._maxFillRatio);
      if (this.ring) {
        this.ring.setMaxFillRatio(this._maxFillRatio);
      }
      return this._maxFillRatio;
    }

    get underrunRecoveryFrames() {
      return this._underrunRecoveryFrames;
    }

    setUnderrunRecoveryFrames(frames) {
      this._underrunRecoveryFrames = sanitizeNonNegInt(frames, this._underrunRecoveryFrames);
      if (this.ring) {
        this.ring.setUnderrunRecoveryFrames(this._underrunRecoveryFrames);
      }
      return this._underrunRecoveryFrames;
    }

    _effectiveMaxFillFrames() {
      if (!this.ring) {
        return undefined;
      }
      if (this._maxFillRatio >= 1.0) {
        return this.ring.capacityFrames;
      }
      const scaled = Math.floor(this.ring.capacityFrames * this._maxFillRatio);
      return scaled < 1 ? 1 : scaled > this.ring.capacityFrames ? this.ring.capacityFrames : scaled;
    }

    /**
     * Converts interleaved PCM into the worker's reusable planar scratch buffers
     * (`Float32Array[]`) with zero per-packet TypedArray allocation in steady state.
     *
     * @param {Float32Array|number[]} samples
     * @param {number} [channels]
     * @param {number} [frameCount]
     * @returns {Float32Array[]}
     */
    interleavedToPlanar(samples, channels = this.channels, frameCount) {
      return this._scratchPool.deinterleave(samples, channels, frameCount);
    }

    /**
     * Converts planar channel buffers into the worker's reusable interleaved
     * scratch buffer (`Float32Array`) with zero per-packet TypedArray allocation
     * in steady state.
     *
     * @param {Float32Array[]} planarChannels
     * @param {number} [frameCount]
     * @returns {Float32Array}
     */
    planarToInterleaved(planarChannels, frameCount) {
      return this._scratchPool.interleave(planarChannels, frameCount);
    }

    /**
     * Pulls `frameCount` frames from the ring into reusable planar scratch buffers
     * (`Float32Array[]`) with zero per-call TypedArray allocation in steady state.
     *
     * @param {number} frameCount
     * @param {number} [gainOverride]
     * @returns {{ framesRead: number, channels: Float32Array[], frameCount: number }}
     */
    pullPlanar(frameCount, gainOverride) {
      const frames = sanitizeNonNegInt(frameCount, 0);
      const channels = this.channels;
      const planar = this._scratchPool.acquirePlanar(channels, frames);
      if (frames <= 0) {
        this._pullPlanarResult.framesRead = 0;
        this._pullPlanarResult.channels = planar;
        this._pullPlanarResult.frameCount = 0;
        return this._pullPlanarResult;
      }
      if (!this.ring) {
        for (let c = 0; c < channels; c++) {
          planar[c].fill(0);
        }
        this._pullPlanarResult.framesRead = 0;
        this._pullPlanarResult.channels = planar;
        this._pullPlanarResult.frameCount = frames;
        return this._pullPlanarResult;
      }
      const scratchInterleaved = this._scratchPool.acquireFloat32Backing('pull', frames * channels);
      const framesRead = this.ring.pullSamples(scratchInterleaved, gainOverride, frames);
      this.totalPulledFrames += framesRead;
      deinterleavePcm(scratchInterleaved, channels, frames, planar);
      this._pullPlanarResult.framesRead = framesRead;
      this._pullPlanarResult.channels = planar;
      this._pullPlanarResult.frameCount = frames;
      return this._pullPlanarResult;
    }

    getScratchStats() {
      return this._scratchPool.getStats();
    }

    recordActiveVoices(count) {
      const sanitized = sanitizeNonNegInt(count, -1);
      if (sanitized > this._activeVoicePeak) {
        this._activeVoicePeak = sanitized;
      }
      return this._activeVoicePeak;
    }

    /**
     * Attaches the runtime's stereo interleaved float32 PCM ring (`Playgta6AudioRing`)
     * backed by a SharedArrayBuffer. Pre-warms reusable scratch memory so steady-state
     * `drainGuestPcmRing` calls perform zero per-quantum heap allocations.
     *
     * @param {object} descriptor
     */
    attachGuestPcmRing(descriptor) {
      const reader = new GuestPcmRingReader(descriptor);

      const rateRatio = reader.sampleRate / this.sampleRate;
      if (rateRatio > MAX_RESAMPLE_RATIO || 1.0 / rateRatio > MAX_RESAMPLE_RATIO) {
        throw new RangeError(
          `Guest PCM ring sampleRate ratio (${reader.sampleRate} -> ${this.sampleRate}) exceeds ${MAX_RESAMPLE_RATIO}x guard`
        );
      }

      const softClip =
        descriptor.softClip !== undefined ? Boolean(descriptor.softClip) : true;
      const rawKnee =
        descriptor.limiterKnee !== undefined
          ? descriptor.limiterKnee
          : descriptor.softClipKnee !== undefined
            ? descriptor.softClipKnee
            : this._limiterKnee;
      if (typeof rawKnee !== 'number' || !Number.isFinite(rawKnee) || rawKnee < 0 || rawKnee >= 1.0) {
        throw new RangeError(`Guest PCM ring limiterKnee must be in [0, 1), got ${rawKnee}`);
      }

      const maxScratchFrames = Math.max(
        1,
        Math.floor(this._scratchPool._maxSamples / GUEST_PCM_RING_CHANNELS)
      );
      const quantumCap = Math.min(GUEST_PCM_RING_MAX_QUANTUM_FRAMES, maxScratchFrames);
      const hasExplicitQuantum =
        descriptor.quantumFrames !== undefined || descriptor.defaultQuantumFrames !== undefined;
      const defaultQuantum = this._targetLatencyConfigured
        ? Math.min(Math.max(128, Math.floor(this.targetLatencyFrames / 2)), quantumCap)
        : Math.min(GUEST_PCM_RING_DEFAULT_QUANTUM_FRAMES, quantumCap);
      const quantumFrames = sanitizePositiveInt(
        descriptor.quantumFrames !== undefined
          ? descriptor.quantumFrames
          : descriptor.defaultQuantumFrames,
        defaultQuantum,
        1,
        quantumCap
      );

      this._guestRing = reader;
      this._guestRingSoftClip = softClip;
      this._guestRingLimiterKnee = rawKnee;
      this._guestRingQuantumFrames = quantumFrames;
      this._guestRingExplicitQuantum = hasExplicitQuantum;
      this._guestRingAutoPump = Boolean(descriptor.autoPump);
      if (descriptor.fadeCurve !== undefined) {
        this._guestFadeCurve = sanitizeFadeCurve(descriptor.fadeCurve, this._guestFadeCurve);
      }
      this._guestResamplerState.phase = 0.0;

      if (descriptor.resetStats === true) {
        this._guestFramesDrained = 0;
        this._guestUnderruns = 0;
        this._guestUnderrunFrames = 0;
        this._guestOverruns = 0;
        this._guestOverrunFrames = 0;
        this._guestPeakLevel = 0.0;
        this._guestUnderrunFadeCount = 0;
        this._guestRecoveryFadeCount = 0;
      }

      // Pre-warm the guest drain stereo scratch buffer once on attach for zero-alloc quanta & bursts.
      const basePrewarmFrames = Math.min(
        Math.max(quantumFrames, GUEST_PCM_RING_ACTIVE_BURST_CHUNK_FRAMES),
        quantumCap
      );
      const prewarmFrames = this._targetLatencyConfigured
        ? Math.min(Math.max(basePrewarmFrames, this.drainWatermarkFrames), quantumCap)
        : basePrewarmFrames;
      this._scratchPool.acquireFloat32Backing(
        'guestDrain',
        prewarmFrames * GUEST_PCM_RING_CHANNELS
      );
      if (reader.sampleRate !== this.sampleRate) {
        const resampledPrewarmFrames = Math.min(
          Math.max(
            prewarmFrames,
            Math.ceil((prewarmFrames * this.sampleRate) / reader.sampleRate) + 4
          ),
          quantumCap
        );
        this._scratchPool.acquireFloat32Backing(
          'resample',
          resampledPrewarmFrames * GUEST_PCM_RING_CHANNELS
        );
      }

      if (this.initialized && this.ring && this._autoDrain && this._shouldRunTimer()) {
        const desiredInterval = this._effectiveTimerIntervalMs();
        if (this._drainTimer === null || this._activeTimerIntervalMs !== desiredInterval) {
          this._startDrainTimer();
        }
      }

      this._publishAudioStatsInternal(true, false, this._now());

      return {
        attached: true,
        guestRingAttached: true,
        sampleRate: reader.sampleRate,
        channels: reader.channels,
        capacitySamples: reader.capacitySamples,
        capacityFrames: reader.capacityFrames,
        quantumFrames,
      };
    }

    /**
     * Detaches the currently attached guest PCM ring, releasing SAB views cleanly.
     */
    detachGuestPcmRing() {
      const wasAttached = this._guestRing !== null;
      this._guestRing = null;
      this._guestRingExplicitQuantum = false;
      this._guestRingAutoPump = false;
      this._guestResamplerState.phase = 0.0;
      if (!this._shouldRunTimer()) {
        this._stopDrainTimer();
      }
      if (wasAttached) {
        this._publishAudioStatsInternal(true, false, this._now());
      }
      return {
        detached: true,
        wasAttached,
        guestRingAttached: false,
      };
    }

    /**
     * Reads stereo interleaved float32 PCM frames from the attached guest PCM ring
     * with atomic cursor handling, soft-clipping into [-1, 1], optional resampling,
     * graceful underrun silence fill, and overrun drop accounting.
     *
     * Supports two zero-allocation modes:
     * 1. Direct output mode: pass a pre-allocated `Float32Array` (or `{ output }`)
     *    to fill directly (always zero-filling any underrun tail).
     * 2. Pump mode: pass `maxFrames` or `{ maxFrames, fillSilence }` to pump frames
     *    into `this.ring` (respecting headless/suspended pre-drain & overflow drops).
     *
     * @param {Float32Array|number|object} [targetOrOptions]
     * @param {object} [maybeOptions]
     */
    drainGuestPcmRing(targetOrOptions, maybeOptions) {
      return this._drainGuestPcmRingInto(targetOrOptions, maybeOptions, null);
    }

    _drainGuestPcmRingInto(targetOrOptions, maybeOptions, outResult) {
      let output = null;
      let opts = null;
      let maxFramesOpt;

      if (targetOrOptions instanceof Float32Array) {
        output = targetOrOptions;
        opts = maybeOptions && typeof maybeOptions === 'object' ? maybeOptions : null;
        if (opts && opts.maxFrames !== undefined) {
          maxFramesOpt = opts.maxFrames;
        }
      } else if (typeof targetOrOptions === 'number') {
        maxFramesOpt = targetOrOptions;
        opts = maybeOptions && typeof maybeOptions === 'object' ? maybeOptions : null;
      } else if (targetOrOptions && typeof targetOrOptions === 'object') {
        opts = targetOrOptions;
        if (opts.output instanceof Float32Array) {
          output = opts.output;
        }
        if (opts.maxFrames !== undefined) {
          maxFramesOpt = opts.maxFrames;
        } else if (opts.frameCount !== undefined) {
          maxFramesOpt = opts.frameCount;
        }
      }

      if (output !== null && !isSharedArrayBuffer(output.buffer) && isDetachedArrayBuffer(output.buffer)) {
        throw new TypeError('drainGuestPcmRing output buffer is detached');
      }

      const maxScratchFrames = Math.max(
        1,
        Math.floor(this._scratchPool._maxSamples / GUEST_PCM_RING_CHANNELS)
      );
      let requestedFrames;
      if (output !== null) {
        const outputMaxFrames = Math.floor(output.length / GUEST_PCM_RING_CHANNELS);
        requestedFrames =
          maxFramesOpt !== undefined
            ? Math.min(sanitizeNonNegInt(maxFramesOpt, outputMaxFrames), outputMaxFrames)
            : outputMaxFrames;
      } else {
        const upper = Math.min(GUEST_PCM_RING_MAX_QUANTUM_FRAMES, maxScratchFrames);
        if (maxFramesOpt !== undefined) {
          requestedFrames = sanitizePositiveInt(maxFramesOpt, this._guestRingQuantumFrames, 1, upper);
        } else if (this._targetLatencyConfigured && this.ring) {
          const watermark = this.drainWatermarkFrames;
          const buffered = this.ring.availableFrames();
          const deficit = watermark > buffered ? watermark - buffered : 0;
          requestedFrames = deficit > 0 ? Math.min(deficit, upper) : 0;
        } else {
          requestedFrames = Math.min(this._guestRingQuantumFrames, upper);
        }
      }

      if (!this._guestRing || requestedFrames <= 0) {
        if (output !== null && requestedFrames > 0) {
          const fillSil = opts && opts.fillSilence !== undefined ? Boolean(opts.fillSilence) : true;
          if (fillSil) {
            output.fill(0, 0, requestedFrames * GUEST_PCM_RING_CHANNELS);
          }
        }
        if (outResult) {
          outResult.framesDrained = 0;
          outResult.requestedFrames = requestedFrames;
          outResult.pushedFrames = 0;
          outResult.droppedFrames = 0;
          outResult.underrun = false;
          outResult.underrunFrames = 0;
          outResult.overrun = false;
          outResult.overrunDroppedFrames = 0;
          outResult.peakLevel = 0.0;
          outResult.zeroAlloc = true;
          outResult.sanitizedNonFiniteSamples = 0;
          outResult.softClippedSamples = 0;
          return outResult;
        }
        return {
          framesDrained: 0,
          requestedFrames,
          pushedFrames: 0,
          droppedFrames: 0,
          underrun: false,
          underrunFrames: 0,
          overrun: false,
          overrunDroppedFrames: 0,
          peakLevel: 0.0,
          zeroAlloc: true,
          sanitizedNonFiniteSamples: 0,
          softClippedSamples: 0,
        };
      }

      const reader = this._guestRing;
      const softClip =
        opts && opts.softClip !== undefined ? Boolean(opts.softClip) : this._guestRingSoftClip;
      const rawKnee =
        opts && opts.limiterKnee !== undefined
          ? opts.limiterKnee
          : opts && opts.softClipKnee !== undefined
            ? opts.softClipKnee
            : this._guestRingLimiterKnee;
      const limiterKnee =
        typeof rawKnee === 'number' && Number.isFinite(rawKnee) && rawKnee >= 0 && rawKnee < 1.0
          ? rawKnee
          : this._guestRingLimiterKnee;
      const fillSilence =
        opts && opts.fillSilence !== undefined
          ? Boolean(opts.fillSilence)
          : output !== null;
      const drainDeclickFrames =
        opts && opts.declickFrames !== undefined
          ? sanitizeDeclickFrames(opts.declickFrames, this._declickFrames)
          : this._declickFrames;
      const drainFadeCurve =
        opts && opts.fadeCurve !== undefined
          ? sanitizeFadeCurve(opts.fadeCurve, this._guestFadeCurve)
          : this._guestFadeCurve;

      const guestRate = reader.sampleRate;
      const targetRate = this.sampleRate;
      const needsResample = guestRate !== targetRate;
      const readOpts = this._guestReadOpts;
      readOpts.softClip = softClip;
      readOpts.limiterKnee = limiterKnee;
      readOpts.declickFrames = drainDeclickFrames;
      readOpts.fadeCurve = drainFadeCurve;

      const allocsBefore = this._scratchPool.allocations;

      if (output !== null) {
        // Direct output buffer mode (zero per-quantum allocation).
        if (!needsResample) {
          readOpts.fillSilence = fillSilence;
          const readRes = reader.readInto(output, requestedFrames, readOpts, this._guestReadScratch);
          this._guestFramesDrained += readRes.framesRead;
          if (readRes.underrun) {
            this._guestUnderruns += 1;
            this._guestUnderrunFrames += readRes.underrunFrames;
          }
          if (readRes.overrun) {
            this._guestOverruns += 1;
            this._guestOverrunFrames += readRes.overrunDroppedFrames;
          }
          if (readRes.peakLevel > this._guestPeakLevel) {
            this._guestPeakLevel = readRes.peakLevel;
          }
          if (readRes.underrunFade) {
            this._guestUnderrunFadeCount += 1;
          }
          if (readRes.recoveryFade) {
            this._guestRecoveryFadeCount += 1;
          }
          this._clippedSamples += readRes.clippedSamples;
          this._softClippedSamples += readRes.softClippedSamples;
          this._sanitizedSamples += readRes.sanitizedSamples;
          this._zeroAllocDrains += 1;
          if (readRes.framesRead > 0) {
            this.recordActiveVoices(1);
          }
          if (outResult) {
            outResult.framesDrained = readRes.framesRead;
            outResult.requestedFrames = requestedFrames;
            outResult.pushedFrames = 0;
            outResult.droppedFrames = 0;
            outResult.underrun = readRes.underrun;
            outResult.underrunFrames = readRes.underrunFrames;
            outResult.overrun = readRes.overrun;
            outResult.overrunDroppedFrames = readRes.overrunDroppedFrames;
            outResult.peakLevel = readRes.peakLevel;
            outResult.zeroAlloc = true;
            outResult.sanitizedNonFiniteSamples = readRes.sanitizedSamples;
            outResult.softClippedSamples = readRes.softClippedSamples;
            return outResult;
          }
          return {
            framesDrained: readRes.framesRead,
            requestedFrames,
            pushedFrames: 0,
            droppedFrames: 0,
            underrun: readRes.underrun,
            underrunFrames: readRes.underrunFrames,
            overrun: readRes.overrun,
            overrunDroppedFrames: readRes.overrunDroppedFrames,
            peakLevel: readRes.peakLevel,
            zeroAlloc: true,
            sanitizedNonFiniteSamples: readRes.sanitizedSamples,
            softClippedSamples: readRes.softClippedSamples,
          };
        }

        // Direct output mode with sample-rate conversion (reuses pre-allocated stereo scratch).
        const step = guestRate / targetRate;
        const startPhase = this._guestResamplerState.phase || 0.0;
        const neededSrcFrames = Math.min(
          maxScratchFrames,
          Math.max(1, Math.ceil(startPhase + (requestedFrames - 1) * step + 1e-9))
        );
        const srcScratch = this._scratchPool.acquireFloat32Backing(
          'guestDrain',
          neededSrcFrames * GUEST_PCM_RING_CHANNELS
        );
        readOpts.fillSilence = false;
        readOpts.declickFrames = 0;
        const readRes = reader.readInto(srcScratch, neededSrcFrames, readOpts, this._guestReadScratch);

        this._guestFramesDrained += readRes.framesRead;
        if (readRes.overrun) {
          this._guestOverruns += 1;
          this._guestOverrunFrames += readRes.overrunDroppedFrames;
        }
        this._clippedSamples += readRes.clippedSamples;
        this._softClippedSamples += readRes.softClippedSamples;
        this._sanitizedSamples += readRes.sanitizedSamples;

        let outFramesWritten = 0;
        let peakLevel = readRes.peakLevel;
        let totalSanitized = readRes.sanitizedSamples;
        let totalSoftClipped = readRes.softClippedSamples;
        if (readRes.framesRead > 0) {
          const rsStats = this._rsStats;
          rsStats.resampledFrames = 0;
          rsStats.nanSanitizedSamples = 0;
          rsStats.clippedSamples = 0;
          const rsOpts = this._resampleOpts;
          rsOpts.srcFrames = readRes.framesRead;
          rsOpts.state = this._guestResamplerState;
          rsOpts.maxOutputFrames = maxScratchFrames;
          rsOpts.scratch = this._scratchPool;
          rsOpts.stats = rsStats;
          rsOpts.softClip = softClip;
          rsOpts.limiterKnee = limiterKnee;
          const resampled = resampleLinear(
            srcScratch,
            GUEST_PCM_RING_CHANNELS,
            guestRate,
            targetRate,
            rsOpts
          );
          this._resampledFrames += rsStats.resampledFrames;
          this._sanitizedSamples += rsStats.nanSanitizedSamples;
          this._clippedSamples += rsStats.clippedSamples;
          if (softClip) {
            this._softClippedSamples += rsStats.clippedSamples;
            totalSoftClipped += rsStats.clippedSamples;
          }
          totalSanitized += rsStats.nanSanitizedSamples;

          const resampledFrames = Math.floor(resampled.length / GUEST_PCM_RING_CHANNELS);
          outFramesWritten =
            resampledFrames <= requestedFrames ? resampledFrames : requestedFrames;
          const samplesToCopy = outFramesWritten * GUEST_PCM_RING_CHANNELS;
          for (let i = 0; i < samplesToCopy; i++) {
            const v = resampled[i];
            output[i] = v;
            const absV = v < 0 ? -v : v;
            if (absV > peakLevel) {
              peakLevel = absV;
            }
          }
          if (outFramesWritten > 0) {
            this.recordActiveVoices(1);
          }
        }

        if (peakLevel > this._guestPeakLevel) {
          this._guestPeakLevel = peakLevel;
        }

        const useCosine = drainFadeCurve !== 'linear';
        if (outFramesWritten > 0 && drainDeclickFrames > 0 && fillSilence) {
          if (this._directResampleInUnderrun) {
            this._directResampleInUnderrun = false;
            this._directResampleFadeInPos = 0;
            this._directResampleFadeInTotal = drainDeclickFrames;
            this._guestRecoveryFadeCount += 1;
          }
          if (
            this._directResampleFadeInTotal > 0 &&
            this._directResampleFadeInPos < this._directResampleFadeInTotal
          ) {
            const rem = this._directResampleFadeInTotal - this._directResampleFadeInPos;
            const rampCount = outFramesWritten < rem ? outFramesWritten : rem;
            const total = this._directResampleFadeInTotal;
            const startPos = this._directResampleFadeInPos;
            if (useCosine) {
              const step = Math.PI / total;
              const halfStep = 0.5 * step;
              const s = Math.sin(halfStep);
              const k = 4.0 * s * s;
              const theta0 = startPos * step;
              let c = startPos === 0 ? 1.0 : Math.cos(theta0);
              let delta =
                startPos === 0 ? -0.5 * k : -2.0 * Math.sin(theta0 + halfStep) * s;
              for (let f = 0; f < rampCount; f++) {
                const rampGain = 0.5 * (1.0 - c);
                const base = f * GUEST_PCM_RING_CHANNELS;
                output[base] *= rampGain;
                output[base + 1] *= rampGain;
                if (
                  (f & COS_RECURRENCE_RESYNC_MASK) === COS_RECURRENCE_RESYNC_MASK &&
                  f + 1 < rampCount
                ) {
                  const nextTheta = (startPos + f + 1) * step;
                  c = Math.cos(nextTheta);
                  delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
                } else {
                  c += delta;
                  delta -= k * c;
                }
              }
              this._fastCosineRamps += 1;
            } else {
              const invTotal = 1.0 / total;
              for (let f = 0; f < rampCount; f++) {
                const rampGain = (startPos + f) * invTotal;
                const base = f * GUEST_PCM_RING_CHANNELS;
                output[base] *= rampGain;
                output[base + 1] *= rampGain;
              }
            }
            this._directResampleFadeInPos += rampCount;
            if (this._directResampleFadeInPos >= this._directResampleFadeInTotal) {
              this._directResampleFadeInPos = 0;
              this._directResampleFadeInTotal = 0;
            }
          }
        }

        let underrun = false;
        let underrunFrames = 0;
        if (outFramesWritten < requestedFrames) {
          underrun = true;
          underrunFrames = requestedFrames - outFramesWritten;
          if (drainDeclickFrames > 0 && fillSilence) {
            this._directResampleFadeInPos = 0;
            this._directResampleFadeInTotal = 0;
            if (outFramesWritten > 0 || this._directResampleHasActive) {
              if (outFramesWritten > 0) {
                const lastBase = (outFramesWritten - 1) * GUEST_PCM_RING_CHANNELS;
                this._directResampleLastFrame[0] = output[lastBase];
                this._directResampleLastFrame[1] = output[lastBase + 1];
              }
              const fadeFrames =
                drainDeclickFrames < underrunFrames ? drainDeclickFrames : underrunFrames;
              const lastL = this._directResampleLastFrame[0];
              const lastR = this._directResampleLastFrame[1];
              const lastFade = fadeFrames - 1;
              if (fadeFrames === 1) {
                const base = outFramesWritten * GUEST_PCM_RING_CHANNELS;
                output[base] = lastL * 0.5;
                output[base + 1] = lastR * 0.5;
              } else if (useCosine) {
                const step = Math.PI / fadeFrames;
                const halfStep = 0.5 * step;
                const s = Math.sin(halfStep);
                const k = 4.0 * s * s;
                let c = 1.0 - 0.5 * k;
                let delta = -0.5 * k - k * c;
                for (let i = 0; i < lastFade; i++) {
                  const rampGain = 0.5 * (1.0 + c);
                  const base = (outFramesWritten + i) * GUEST_PCM_RING_CHANNELS;
                  output[base] = lastL * rampGain;
                  output[base + 1] = lastR * rampGain;
                  if (
                    (i & COS_RECURRENCE_RESYNC_MASK) === COS_RECURRENCE_RESYNC_MASK &&
                    i + 1 < lastFade
                  ) {
                    const nextTheta = (i + 2) * step;
                    c = Math.cos(nextTheta);
                    delta = -2.0 * Math.sin(nextTheta + halfStep) * s;
                  } else {
                    c += delta;
                    delta -= k * c;
                  }
                }
                const endBase = (outFramesWritten + lastFade) * GUEST_PCM_RING_CHANNELS;
                output[endBase] = 0.0;
                output[endBase + 1] = 0.0;
                this._fastCosineRamps += 1;
              } else {
                const invFade = 1.0 / fadeFrames;
                for (let i = 0; i < fadeFrames; i++) {
                  const rampGain = (lastFade - i) * invFade;
                  const base = (outFramesWritten + i) * GUEST_PCM_RING_CHANNELS;
                  output[base] = lastL * rampGain;
                  output[base + 1] = lastR * rampGain;
                }
              }
              if (outFramesWritten + fadeFrames < requestedFrames) {
                output.fill(
                  0,
                  (outFramesWritten + fadeFrames) * GUEST_PCM_RING_CHANNELS,
                  requestedFrames * GUEST_PCM_RING_CHANNELS
                );
              }
              this._guestUnderrunFadeCount += 1;
              this._directResampleHasActive = false;
              this._directResampleLastFrame[0] = 0.0;
              this._directResampleLastFrame[1] = 0.0;
            } else {
              output.fill(
                0,
                outFramesWritten * GUEST_PCM_RING_CHANNELS,
                requestedFrames * GUEST_PCM_RING_CHANNELS
              );
            }
            this._directResampleInUnderrun = true;
          } else if (fillSilence) {
            output.fill(
              0,
              outFramesWritten * GUEST_PCM_RING_CHANNELS,
              requestedFrames * GUEST_PCM_RING_CHANNELS
            );
          }
          this._guestUnderruns += 1;
          this._guestUnderrunFrames += underrunFrames;
        } else if (outFramesWritten > 0) {
          const lastBase = (outFramesWritten - 1) * GUEST_PCM_RING_CHANNELS;
          this._directResampleLastFrame[0] = output[lastBase];
          this._directResampleLastFrame[1] = output[lastBase + 1];
          this._directResampleHasActive = true;
        }

        const zeroAlloc = this._scratchPool.allocations === allocsBefore;
        if (zeroAlloc) {
          this._zeroAllocDrains += 1;
        }

        if (outResult) {
          outResult.framesDrained = readRes.framesRead;
          outResult.requestedFrames = requestedFrames;
          outResult.outputFrames = outFramesWritten;
          outResult.pushedFrames = 0;
          outResult.droppedFrames = 0;
          outResult.underrun = underrun;
          outResult.underrunFrames = underrunFrames;
          outResult.overrun = readRes.overrun;
          outResult.overrunDroppedFrames = readRes.overrunDroppedFrames;
          outResult.peakLevel = peakLevel;
          outResult.zeroAlloc = zeroAlloc;
          outResult.sanitizedNonFiniteSamples = totalSanitized;
          outResult.softClippedSamples = totalSoftClipped;
          return outResult;
        }

        return {
          framesDrained: readRes.framesRead,
          requestedFrames,
          outputFrames: outFramesWritten,
          pushedFrames: 0,
          droppedFrames: 0,
          underrun,
          underrunFrames,
          overrun: readRes.overrun,
          overrunDroppedFrames: readRes.overrunDroppedFrames,
          peakLevel,
          zeroAlloc,
          sanitizedNonFiniteSamples: totalSanitized,
          softClippedSamples: totalSoftClipped,
        };
      }

      // Pump mode: drain from guest PCM ring into worker output ring (`this.ring`)
      // using the pre-allocated stereo scratch buffer with zero per-drain allocations.
      if (this.ring && this.ring.channels !== GUEST_PCM_RING_CHANNELS) {
        throw new RangeError(
          `Cannot pump stereo guest PCM ring into ${this.ring.channels}-channel worker ring`
        );
      }

      const scratchBuf = this._scratchPool.acquireFloat32Backing(
        'guestDrain',
        requestedFrames * GUEST_PCM_RING_CHANNELS
      );
      readOpts.fillSilence = fillSilence;
      const readRes = reader.readInto(scratchBuf, requestedFrames, readOpts, this._guestReadScratch);

      this._guestFramesDrained += readRes.framesRead;
      if (readRes.underrun) {
        this._guestUnderruns += 1;
        this._guestUnderrunFrames += readRes.underrunFrames;
      }
      if (readRes.overrun) {
        this._guestOverruns += 1;
        this._guestOverrunFrames += readRes.overrunDroppedFrames;
      }
      if (readRes.peakLevel > this._guestPeakLevel) {
        this._guestPeakLevel = readRes.peakLevel;
      }
      if (readRes.underrunFade) {
        this._guestUnderrunFadeCount += 1;
      }
      if (readRes.recoveryFade) {
        this._guestRecoveryFadeCount += 1;
      }
      this._clippedSamples += readRes.clippedSamples;
      this._softClippedSamples += readRes.softClippedSamples;
      this._sanitizedSamples += readRes.sanitizedSamples;

      let totalSanitized = readRes.sanitizedSamples;
      let totalSoftClipped = readRes.softClippedSamples;
      const sourceFramesToPump = fillSilence ? requestedFrames : readRes.framesRead;
      let pcmToPush = scratchBuf;
      let framesToPush = sourceFramesToPump;

      if (needsResample && sourceFramesToPump > 0) {
        const rsStats = this._rsStats;
        rsStats.resampledFrames = 0;
        rsStats.nanSanitizedSamples = 0;
        rsStats.clippedSamples = 0;
        const rsOpts = this._resampleOpts;
        rsOpts.srcFrames = sourceFramesToPump;
        rsOpts.state = this._guestResamplerState;
        rsOpts.maxOutputFrames = maxScratchFrames;
        rsOpts.scratch = this._scratchPool;
        rsOpts.stats = rsStats;
        rsOpts.softClip = softClip;
        rsOpts.limiterKnee = limiterKnee;
        pcmToPush = resampleLinear(
          scratchBuf,
          GUEST_PCM_RING_CHANNELS,
          guestRate,
          targetRate,
          rsOpts
        );
        framesToPush = Math.floor(pcmToPush.length / GUEST_PCM_RING_CHANNELS);
        this._resampledFrames += rsStats.resampledFrames;
        this._sanitizedSamples += rsStats.nanSanitizedSamples;
        this._clippedSamples += rsStats.clippedSamples;
        if (softClip) {
          this._softClippedSamples += rsStats.clippedSamples;
          totalSoftClipped += rsStats.clippedSamples;
        }
        totalSanitized += rsStats.nanSanitizedSamples;
      }

      let pushedFrames = 0;
      let droppedFrames = 0;
      let overrun = readRes.overrun;
      let overrunDroppedFrames = readRes.overrunDroppedFrames;

      if (this.ring && framesToPush > 0) {
        this._packetsReceived += 1;
        this._framesDecoded += framesToPush;
        this.recordActiveVoices(1);

        if (this._shouldDrain()) {
          const free = this.ring.freeFrames();
          if (framesToPush > free) {
            const deficit = framesToPush - free;
            const drainCap = this._effectiveMaxDrainFramesPerTick();
            const drainBudget =
              this._explicitMaxDrainFramesPerTick !== null
                ? Math.min(deficit, drainCap)
                : Math.min(deficit, Math.max(drainCap, this.ring.capacityFrames));
            const preDrained = this.ring.discardFrames(drainBudget);
            if (preDrained > 0) {
              this.framesDrained += preDrained;
            }
          }
        }

        this._pushOpts.frameCount = framesToPush;
        this._pushOpts.overflowPolicy = this._overflowPolicy;
        this._pushOpts.maxFillFrames = this._effectiveMaxFillFrames();

        pushedFrames = this.ring.pushSamples(pcmToPush, this._pushOpts);
        droppedFrames = this.ring._lastPushDroppedFrames;
        const pushOverflowed = this.ring._lastPushOverflowed;
        this.totalPushedFrames += pushedFrames;

        if (pushOverflowed || droppedFrames > 0) {
          this._droppedPackets += 1;
          this._guestOverruns += 1;
          this._guestOverrunFrames += droppedFrames;
          overrun = true;
          overrunDroppedFrames += droppedFrames;
        }
      }

      const zeroAlloc = this._scratchPool.allocations === allocsBefore;
      if (zeroAlloc) {
        this._zeroAllocDrains += 1;
      }

      if (!this._inDrainTick && readRes.framesRead > 0) {
        this._maybeBroadcastAudioStats(this._now());
      }

      if (outResult) {
        outResult.framesDrained = readRes.framesRead;
        outResult.requestedFrames = requestedFrames;
        outResult.pushedFrames = pushedFrames;
        outResult.droppedFrames = droppedFrames;
        outResult.underrun = readRes.underrun;
        outResult.underrunFrames = readRes.underrunFrames;
        outResult.overrun = overrun;
        outResult.overrunDroppedFrames = overrunDroppedFrames;
        outResult.peakLevel = readRes.peakLevel;
        outResult.zeroAlloc = zeroAlloc;
        outResult.sanitizedNonFiniteSamples = totalSanitized;
        outResult.softClippedSamples = totalSoftClipped;
        return outResult;
      }

      return {
        framesDrained: readRes.framesRead,
        requestedFrames,
        pushedFrames,
        droppedFrames,
        underrun: readRes.underrun,
        underrunFrames: readRes.underrunFrames,
        overrun,
        overrunDroppedFrames,
        peakLevel: readRes.peakLevel,
        zeroAlloc,
        sanitizedNonFiniteSamples: totalSanitized,
        softClippedSamples: totalSoftClipped,
      };
    }

    _resetTelemetryAndCounters() {
      this.totalPushedFrames = 0;
      this.totalPulledFrames = 0;
      this.framesDrained = 0;
      this._packetsReceived = 0;
      this._framesDecoded = 0;
      this._droppedPackets = 0;
      this._drainTicks = 0;
      this._overruns = 0;
      this._rejectedPackets = 0;
      this._sabCopiedPackets = 0;
      this._sanitizedSamples = 0;
      this._downmixedFrames = 0;
      this._resampledFrames = 0;
      this._clippedSamples = 0;
      this._softClippedSamples = 0;
      this._zeroAllocDrains = 0;
      this._fastCosineRamps = 0;
      this._activeVoicePeak = 0;
      this._resamplerState.phase = 0.0;
      this._lastResampleSrcRate = null;
      this._lastDrainMs = null;
      this._stalledFullTicks = 0;
      this._stalledFullSinceMs = null;
      this._lastObservedReadCursor = null;
      this._guestResamplerState.phase = 0.0;
      this._guestFramesDrained = 0;
      this._guestUnderruns = 0;
      this._guestUnderrunFrames = 0;
      this._guestOverruns = 0;
      this._guestOverrunFrames = 0;
      this._guestPeakLevel = 0.0;
      this._guestUnderrunFadeCount = 0;
      this._guestRecoveryFadeCount = 0;
      this._guestTrimmedFrames = 0;
      this._guestTrimCount = 0;
      this._directResampleLastFrame[0] = 0.0;
      this._directResampleLastFrame[1] = 0.0;
      this._directResampleHasActive = false;
      this._directResampleInUnderrun = false;
      this._directResampleFadeInPos = 0;
      this._directResampleFadeInTotal = 0;
      this._lastTelemetryBroadcastMs = null;
      this._lastBroadcastFramesDrained = -1;
      this._lastBroadcastGuestFramesDrained = -1;
      this._firstGuestDrainBroadcasted = false;
      this._telemetryBroadcastCount = 0;
      if (this._guestRing) {
        this._guestRing.resetFadeState();
      }
      this._scratchPool.resetStats();
    }

    reset() {
      if (this.ring) {
        this.ring.reset();
      }
      this._closeTelemetryChannel();
      this._resetTelemetryAndCounters();
    }

    stop() {
      this._stopDrainTimer();
      if (
        this.framesDrained !== this._lastBroadcastFramesDrained ||
        this._guestFramesDrained !== this._lastBroadcastGuestFramesDrained
      ) {
        this._publishAudioStatsInternal(true, false, this._now());
      }
      this.reset();
    }

    destroy() {
      this._stopDrainTimer();
      if (
        this.framesDrained !== this._lastBroadcastFramesDrained ||
        this._guestFramesDrained !== this._lastBroadcastGuestFramesDrained
      ) {
        this._publishAudioStatsInternal(true, false, this._now());
      }
      this.detachGuestPcmRing();
      this._closeTelemetryChannel();
    }

    effectiveGain() {
      return this.muted ? 0.0 : this.volume;
    }

    pullSamples(output, gainOverride) {
      if (
        !(output instanceof Float32Array) ||
        (!isSharedArrayBuffer(output.buffer) && isDetachedArrayBuffer(output.buffer))
      ) {
        return 0;
      }
      if (!this.ring) {
        output.fill(0);
        return 0;
      }
      const framesRead = this.ring.pullSamples(output, gainOverride);
      this.totalPulledFrames += framesRead;
      return framesRead;
    }

    getStats() {
      const ringStats = this.ring
        ? this.ring.getStats()
        : {
            capacityFrames: 0,
            channels: this.channels,
            availableFrames: 0,
            freeFrames: 0,
            bufferFillPct: 0.0,
            volume: this.volume,
            muted: this.muted,
            effectiveGain: this.effectiveGain(),
            declickFrames: this._declickFrames,
            overflowPolicy: this._overflowPolicy,
            maxFillRatio: this._maxFillRatio,
            underrunRecoveryFrames: this._underrunRecoveryFrames,
            underruns: 0,
            underrunCount: 0,
            underrunFrames: 0,
            underrunRecoveries: 0,
            fadeOutCount: 0,
            fadeInCount: 0,
            fastCosineRamps: 0,
            scratchBufferReuses: 0,
            peakFillFrames: 0,
            minFillFrames: 0,
            avgFillRatio: 0.0,
            overflows: 0,
            overruns: 0,
            droppedFrames: 0,
          };

      const overruns = ringStats.overflows + this._overruns;

      return {
        initialized: this.initialized,
        outputState: this.outputState,
        sampleRate: this.sampleRate,
        channels: this.channels,
        capacityFrames: ringStats.capacityFrames,
        availableFrames: ringStats.availableFrames,
        freeFrames: ringStats.freeFrames,
        bufferFillPct: ringStats.bufferFillPct,
        volume: this.volume,
        muted: this.muted,
        effectiveGain: this.effectiveGain(),
        declickFrames: this._declickFrames,
        fadeCurve: this._fadeCurve,
        overflowPolicy: this._overflowPolicy,
        maxFillRatio: this._maxFillRatio,
        targetLatencyMs: this._targetLatencyMs,
        underrunRecoveryFrames: this._underrunRecoveryFrames,
        framesDrained: this.framesDrained,
        packetsReceived: this._packetsReceived,
        framesDecoded: this._framesDecoded,
        droppedPackets: this._droppedPackets,
        underruns: ringStats.underruns,
        underrunCount: ringStats.underrunCount,
        underrunFrames: ringStats.underrunFrames,
        underrunRecoveries: ringStats.underrunRecoveries,
        fadeOutCount: ringStats.fadeOutCount,
        fadeInCount: ringStats.fadeInCount,
        underrunFadeCount: ringStats.fadeOutCount + this._guestUnderrunFadeCount,
        recoveryFadeCount: ringStats.fadeInCount + this._guestRecoveryFadeCount,
        peakFillFrames: ringStats.peakFillFrames,
        minFillFrames: ringStats.minFillFrames,
        avgFillRatio: ringStats.avgFillRatio,
        overruns,
        overflows: ringStats.overflows,
        droppedFrames: ringStats.droppedFrames,
        totalPushedFrames: this.totalPushedFrames,
        totalPulledFrames: this.totalPulledFrames,
        drainTicks: this._drainTicks,
        rejectedPackets: this._rejectedPackets,
        sabCopiedPackets: this._sabCopiedPackets,
        sanitizedSamples: this._sanitizedSamples,
        nanSanitizedSamples: this._sanitizedSamples,
        sanitizedNonFiniteSamples: this._sanitizedSamples,
        downmixedFrames: this._downmixedFrames,
        resampledFrames: this._resampledFrames,
        clippedSamples: this._clippedSamples,
        softClippedSamples: this._softClippedSamples,
        activeVoicePeak: this._activeVoicePeak,
        scratchAllocations: this._scratchPool.allocations,
        scratchReuses: this._scratchPool.reuses,
        scratchBufferReuses: this._scratchPool.reuses + (ringStats.scratchBufferReuses || 0),
        fastCosineRamps:
          (ringStats.fastCosineRamps || 0) +
          (this._guestRing ? this._guestRing.fastCosineRamps : 0) +
          this._fastCosineRamps,
        zeroAllocDrains: this._zeroAllocDrains,
        guestRingAttached: this._guestRing !== null,
        guestFramesDrained: this._guestFramesDrained,
        guestUnderruns: this._guestUnderruns,
        guestOverruns: this._guestOverruns,
        guestPeakLevel: this._guestPeakLevel,
        guestTrimmedFrames: this._guestTrimmedFrames,
        guestTrimCount: this._guestTrimCount,
      };
    }

    getSanitizedStats() {
      return buildAudioTelemetrySnapshot(this, { now: this._now }).stats;
    }

    _resolveInitOutputState(msg) {
      if (typeof msg.outputState === 'string') {
        return normalizeOutputState(msg.outputState, OUTPUT_STATES.HEADLESS_DRAIN);
      }
      if (msg.headless === true || msg.audioAvailable === false || msg.workletAvailable === false) {
        return OUTPUT_STATES.HEADLESS_DRAIN;
      }
      if (typeof msg.audioContextState === 'string') {
        return normalizeOutputState(msg.audioContextState, OUTPUT_STATES.HEADLESS_DRAIN);
      }
      if (this._hasExplicitOutputState) {
        return this.outputState;
      }
      const providedRingSab =
        msg.ringSab !== undefined && msg.ringSab !== null
          ? msg.ringSab
          : msg.audioSab !== undefined && msg.audioSab !== null
            ? msg.audioSab
            : msg.sab;
      if (isSharedArrayBuffer(providedRingSab)) {
        return OUTPUT_STATES.ACTIVE;
      }
      return detectDefaultOutputState();
    }

    handleMessage(msg) {
      if (!msg || typeof msg !== 'object' || Array.isArray(msg)) {
        this._rejectedPackets += 1;
        return;
      }

      let safeId;
      try {
        safeId = sanitizeMessageId(msg.id);
      } catch (_err) {
        this._rejectedPackets += 1;
        return;
      }

      try {
        if (msg.init || msg.type === 'init') {
          this._stopDrainTimer();

          if (msg.autoDrain !== undefined) {
            this._autoDrain = Boolean(msg.autoDrain);
          }
          if (msg.drainIntervalMs !== undefined) {
            this._drainIntervalMs = sanitizePositiveInt(
              msg.drainIntervalMs,
              this._drainIntervalMs,
              MIN_DRAIN_INTERVAL_MS,
              MAX_DRAIN_INTERVAL_MS
            );
            this._explicitDrainInterval = true;
          }
          if (
            typeof msg.maxDrainFramesPerTick === 'number' &&
            Number.isFinite(msg.maxDrainFramesPerTick) &&
            msg.maxDrainFramesPerTick > 0
          ) {
            this._explicitMaxDrainFramesPerTick = Math.floor(msg.maxDrainFramesPerTick);
          }
          if (msg.maxPcmBytes !== undefined) {
            this._maxPcmBytes = sanitizePositiveInt(
              msg.maxPcmBytes,
              this._maxPcmBytes,
              Float32Array.BYTES_PER_ELEMENT,
              64 * 1024 * 1024
            );
          }
          if (msg.declickFrames !== undefined) {
            this._declickFrames = sanitizeDeclickFrames(
              msg.declickFrames,
              this._declickFrames
            );
          }
          if (msg.fadeCurve !== undefined || msg.cosineRamp !== undefined) {
            const nextCurve = sanitizeFadeCurve(
              msg.fadeCurve,
              msg.cosineRamp === true ? 'cosine' : this._fadeCurve
            );
            this._fadeCurve = nextCurve;
            this._guestFadeCurve = nextCurve;
          }
          if (msg.targetLatencyMs !== undefined) {
            this._targetLatencyMs = clampTargetLatencyMs(
              msg.targetLatencyMs,
              this._targetLatencyMs
            );
            this._targetLatencyConfigured = true;
            if (msg.drainIntervalMs === undefined) {
              this._drainIntervalMs = sanitizePositiveInt(
                Math.floor(this._targetLatencyMs / 2),
                DEFAULT_DRAIN_INTERVAL_MS,
                MIN_DRAIN_INTERVAL_MS,
                MAX_DRAIN_INTERVAL_MS
              );
            }
          }
          if (msg.overflowPolicy !== undefined) {
            this._overflowPolicy = normalizeOverflowPolicy(
              msg.overflowPolicy,
              this._overflowPolicy
            );
          }
          if (msg.maxFillRatio !== undefined) {
            this._maxFillRatio = sanitizeFillRatio(msg.maxFillRatio, this._maxFillRatio);
          }
          if (msg.underrunRecoveryFrames !== undefined) {
            this._underrunRecoveryFrames = sanitizeNonNegInt(
              msg.underrunRecoveryFrames,
              this._underrunRecoveryFrames
            );
          }
          if (msg.softClip !== undefined) {
            this._softClip = Boolean(msg.softClip);
          }
          if (msg.limiterKnee !== undefined || msg.softClipKnee !== undefined) {
            const rawKnee =
              msg.limiterKnee !== undefined ? msg.limiterKnee : msg.softClipKnee;
            if (typeof rawKnee === 'number' && Number.isFinite(rawKnee) && rawKnee >= 0 && rawKnee < 1.0) {
              this._limiterKnee = rawKnee;
            }
          }

          const nextSampleRate = sanitizeSampleRate(msg.sampleRate);
          const nextChannels = sanitizePositiveInt(
            msg.channels,
            DEFAULT_CHANNELS,
            1,
            MAX_CHANNELS
          );

          let nextRing;
          if (typeof SharedArrayBuffer === 'undefined') {
            this._rejectedPackets += 1;
            const errReply = {
              ready: false,
              outputState: OUTPUT_STATES.HEADLESS_DRAIN,
              error: 'SharedArrayBuffer is unavailable',
            };
            if (safeId !== undefined) errReply.id = safeId;
            this._safePostMessage(errReply);
            return;
          }
          const rawInitSab =
            msg.ringSab !== undefined && msg.ringSab !== null
              ? msg.ringSab
              : msg.audioSab !== undefined && msg.audioSab !== null
                ? msg.audioSab
                : msg.sab !== undefined && msg.sab !== null
                  ? msg.sab
                  : null;
          if (rawInitSab !== null) {
            if (!isSharedArrayBuffer(rawInitSab)) {
              this._rejectedPackets += 1;
              const errReply = {
                ready: false,
                error: 'Invalid ringSab: expected SharedArrayBuffer',
              };
              if (safeId !== undefined) errReply.id = safeId;
              this._safePostMessage(errReply);
              return;
            }
            const reconfiguringSameSab = Boolean(
              this.ring && this.ring.sab === rawInitSab
            );
            nextRing = new AudioRingBuffer(rawInitSab, {
              channels: nextChannels,
              volume: this.volume,
              muted: this.muted,
              declickFrames: this._declickFrames,
              fadeCurve: this._fadeCurve,
              overflowPolicy: this._overflowPolicy,
              maxFillRatio: this._maxFillRatio,
              underrunRecoveryFrames: this._underrunRecoveryFrames,
            });
            if (reconfiguringSameSab) {
              nextRing.reset();
            }
            this._hasExternalRingSab = true;
          } else {
            const cap = sanitizePositiveInt(
              msg.capacityFrames,
              DEFAULT_CAPACITY_FRAMES,
              1,
              MAX_CAPACITY_FRAMES
            );
            nextRing = AudioRingBuffer.create(cap, nextChannels, {
              declickFrames: this._declickFrames,
              fadeCurve: this._fadeCurve,
              overflowPolicy: this._overflowPolicy,
              maxFillRatio: this._maxFillRatio,
              underrunRecoveryFrames: this._underrunRecoveryFrames,
            });
            nextRing.setVolume(this.volume);
            nextRing.setMuted(this.muted);
            this._hasExternalRingSab = false;
          }

          this.sampleRate = nextSampleRate;
          this.ring = nextRing;
          this.channels = nextRing.channels;
          this.outputState = this._resolveInitOutputState(msg);
          this.initialized = true;
          if (msg.preserveGuestStats === true) {
            this._stalledFullTicks = 0;
            this._stalledFullSinceMs = null;
            this._lastObservedReadCursor = null;
          } else {
            this._resetTelemetryAndCounters();
          }

          if (this._shouldRunTimer()) {
            this._startDrainTimer();
          }

          this._safePostMessage({
            ready: true,
            sampleRate: this.sampleRate,
            channels: this.channels,
            capacityFrames: this.ring.capacityFrames,
          });
          return;
        }

        if (
          msg.attachRingSab ||
          msg.type === 'attach-ring-sab' ||
          msg.type === 'attachRingSab' ||
          msg.type === 'attach-sab'
        ) {
          const sab =
            msg.ringSab !== undefined && msg.ringSab !== null
              ? msg.ringSab
              : msg.audioSab !== undefined && msg.audioSab !== null
                ? msg.audioSab
                : msg.sab;
          try {
            const info = this.attachRingSab(sab, msg);
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                attached: true,
                ringSabAttached: true,
                outputState: info.outputState,
                sampleRate: info.sampleRate,
                channels: info.channels,
                capacityFrames: info.capacityFrames,
                guestRingAttached: info.guestRingAttached,
              });
            }
          } catch (attachErr) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                attached: false,
                ringSabAttached: this._hasExternalRingSab,
                error: sanitizeErrorMessage(attachErr),
              });
            }
          }
          return;
        }

        if (msg.pushPcm) {
          if (!this.ring) {
            this._rejectedPackets += 1;
            this._safePostMessage({
              id: safeId,
              error: 'AudioMixerWorker not initialized',
            });
            return;
          }

          let extracted;
          let inferredChannels = this.channels;
          const isPlanarPayload =
            msg.planar === true || Array.isArray(msg.channelData);
          if (isPlanarPayload) {
            const planarArr = Array.isArray(msg.channelData) ? msg.channelData : msg.samples;
            if (
              !Array.isArray(planarArr) ||
              planarArr.length < 1 ||
              planarArr.length > MAX_CHANNELS
            ) {
              this._rejectedPackets += 1;
              if (safeId !== undefined) {
                this._safePostMessage({
                  id: safeId,
                  error: 'Planar PCM payload must be an array of 1..8 Float32Array channel buffers',
                });
              }
              return;
            }
            let minFrames = Infinity;
            let copiedFromSab = false;
            for (let c = 0; c < planarArr.length; c++) {
              const ch = planarArr[c];
              if (!(ch instanceof Float32Array)) {
                this._rejectedPackets += 1;
                if (safeId !== undefined) {
                  this._safePostMessage({
                    id: safeId,
                    error: `Planar PCM channel ${c} must be a Float32Array`,
                  });
                }
                return;
              }
              if (isSharedArrayBuffer(ch.buffer)) {
                copiedFromSab = true;
              } else if (isDetachedArrayBuffer(ch.buffer)) {
                this._rejectedPackets += 1;
                if (safeId !== undefined) {
                  this._safePostMessage({
                    id: safeId,
                    error: `Planar PCM channel ${c} backing ArrayBuffer is detached`,
                  });
                }
                return;
              }
              if (ch.length < minFrames) {
                minFrames = ch.length;
              }
            }
            const totalBytes =
              minFrames * planarArr.length * Float32Array.BYTES_PER_ELEMENT;
            if (totalBytes > this._maxPcmBytes) {
              this._rejectedPackets += 1;
              if (safeId !== undefined) {
                this._safePostMessage({
                  id: safeId,
                  error: `PCM payload (${totalBytes} bytes) exceeds max allowed (${this._maxPcmBytes} bytes)`,
                });
              }
              return;
            }
            let sanitizedSamples = 0;
            const interleaved = this._scratchPool.acquireFloat32(
              'interleave',
              minFrames * planarArr.length
            );
            const chCount = planarArr.length;
            for (let f = 0; f < minFrames; f++) {
              const base = f * chCount;
              for (let c = 0; c < chCount; c++) {
                const v = planarArr[c][f];
                if (Number.isFinite(v)) {
                  interleaved[base + c] = v;
                } else {
                  interleaved[base + c] = 0.0;
                  sanitizedSamples += 1;
                }
              }
            }
            inferredChannels = chCount;
            extracted = this._extractResult;
            extracted.ok = true;
            extracted.pcm = interleaved;
            extracted.copiedFromSab = copiedFromSab;
            extracted.sanitizedSamples = sanitizedSamples;
            extracted.error = '';
          } else {
            this._extractOpts.maxPcmBytes = this._maxPcmBytes;
            extracted = extractSafePcmSamples(msg.samples, this._extractOpts);
          }

          if (!extracted.ok) {
            const extractErr = extracted.error;
            this._extractResult.pcm = null;
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: extractErr,
              });
            }
            return;
          }

          const extractedPcm = extracted.pcm;
          const extractedCopiedFromSab = extracted.copiedFromSab;
          const extractedSanitizedSamples = extracted.sanitizedSamples;
          this._extractResult.pcm = null;

          const srcChannels = msg.channels !== undefined ? msg.channels : inferredChannels;
          if (!Number.isInteger(srcChannels) || srcChannels < 1 || srcChannels > MAX_CHANNELS) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: `Unsupported PCM channel count: ${srcChannels}`,
              });
            }
            return;
          }
          if (
            srcChannels !== this.channels &&
            !(this.channels === 2 && SUPPORTED_DOWNMIX_CHANNELS.includes(srcChannels))
          ) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: `Cannot convert ${srcChannels}-channel PCM into ${this.channels}-channel ring`,
              });
            }
            return;
          }

          const srcSampleRate = msg.sampleRate !== undefined ? msg.sampleRate : this.sampleRate;
          if (
            typeof srcSampleRate !== 'number' ||
            !Number.isFinite(srcSampleRate) ||
            srcSampleRate < MIN_SAMPLE_RATE ||
            srcSampleRate > MAX_SAMPLE_RATE
          ) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: `Invalid PCM sampleRate: ${srcSampleRate}`,
              });
            }
            return;
          }
          const rateRatio = srcSampleRate / this.sampleRate;
          if (rateRatio > MAX_RESAMPLE_RATIO || 1.0 / rateRatio > MAX_RESAMPLE_RATIO) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: `PCM sampleRate ratio (${srcSampleRate} -> ${this.sampleRate}) exceeds ${MAX_RESAMPLE_RATIO}x guard`,
              });
            }
            return;
          }

          this._packetsReceived += 1;

          if (extractedCopiedFromSab) {
            this._sabCopiedPackets += 1;
          }
          if (extractedSanitizedSamples > 0) {
            this._sanitizedSamples += extractedSanitizedSamples;
          }

          let pcm = extractedPcm;

          // Step 1: ITU-R BS.775 multi-channel (5.1 / 7.1 / 4ch) or mono -> stereo downmix/adapt.
          if (srcChannels !== this.channels) {
            const dmStats = this._dmStats;
            dmStats.downmixedFrames = 0;
            dmStats.clippedSamples = 0;
            dmStats.nanSanitizedSamples = 0;
            dmStats.peakLevel = 0.0;
            const dmSoftClip = msg.softClip !== undefined ? Boolean(msg.softClip) : true;
            const dmKnee =
              msg.limiterKnee !== undefined
                ? msg.limiterKnee
                : msg.softClipKnee !== undefined
                  ? msg.softClipKnee
                  : this._limiterKnee;
            const dmOpts = this._downmixOpts;
            dmOpts.centerGain = msg.centerGain;
            dmOpts.surroundGain = msg.surroundGain;
            dmOpts.lfeGain = msg.lfeGain;
            dmOpts.softClip = dmSoftClip;
            dmOpts.limiterKnee = dmKnee;
            pcm = downmixToStereo(pcm, srcChannels, dmOpts);
            this._downmixedFrames += dmStats.downmixedFrames;
            this._clippedSamples += dmStats.clippedSamples;
            if (dmSoftClip) {
              this._softClippedSamples += dmStats.clippedSamples;
            }
            this._sanitizedSamples += dmStats.nanSanitizedSamples;
          } else if (msg.softClip === true || (msg.softClip === undefined && this._softClip)) {
            const knee =
              msg.limiterKnee !== undefined
                ? msg.limiterKnee
                : msg.softClipKnee !== undefined
                  ? msg.softClipKnee
                  : this._limiterKnee;
            const limited = this._scratchPool.acquireFloat32('limit', pcm.length);
            let clipped = 0;
            for (let i = 0; i < pcm.length; i++) {
              const v = pcm[i];
              if (v > knee || v < -knee) {
                clipped += 1;
              }
              limited[i] = softClipSample(v, knee);
            }
            this._clippedSamples += clipped;
            this._softClippedSamples += clipped;
            pcm = limited;
          }

          // Step 2: Smooth 44.1 kHz <-> 48 kHz (and general) linear resampling with phase continuity.
          if (srcSampleRate !== this.sampleRate) {
            if (this._lastResampleSrcRate !== srcSampleRate) {
              this._resamplerState.phase = 0.0;
              this._lastResampleSrcRate = srcSampleRate;
            }
            const maxOutputFrames = Math.max(
              1,
              Math.floor(this._maxPcmBytes / (this.channels * Float32Array.BYTES_PER_ELEMENT))
            );
            const rsStats = this._rsStats;
            rsStats.resampledFrames = 0;
            rsStats.nanSanitizedSamples = 0;
            rsStats.clippedSamples = 0;
            const rsOpts = this._resampleOpts;
            rsOpts.srcFrames = undefined;
            rsOpts.state = this._resamplerState;
            rsOpts.maxOutputFrames = maxOutputFrames;
            rsOpts.scratch = this._scratchPool;
            rsOpts.stats = rsStats;
            rsOpts.softClip = false;
            rsOpts.limiterKnee = this._limiterKnee;
            pcm = resampleLinear(pcm, this.channels, srcSampleRate, this.sampleRate, rsOpts);
            this._resampledFrames += rsStats.resampledFrames;
            this._sanitizedSamples += rsStats.nanSanitizedSamples;
            this._clippedSamples += rsStats.clippedSamples;
          } else if (this._lastResampleSrcRate !== null) {
            this._resamplerState.phase = 0.0;
            this._lastResampleSrcRate = null;
          }

          const inputFrames = Math.floor(pcm.length / this.channels);
          this._framesDecoded += inputFrames;

          if (msg.convertToPlanar === true && inputFrames > 0) {
            this._scratchPool.deinterleave(pcm, this.channels, inputFrames);
          }

          if (
            typeof msg.activeVoices === 'number' &&
            Number.isFinite(msg.activeVoices) &&
            msg.activeVoices >= 0
          ) {
            this.recordActiveVoices(msg.activeVoices);
          } else if (inputFrames > 0) {
            this.recordActiveVoices(1);
          }

          // In headless-drain or autoplay-suspended mode, drain oldest frames at
          // bounded rate when ring pressure builds so guest audio producers never
          // deadlock waiting for free ring space.
          if (inputFrames > 0 && this._shouldDrain()) {
            const free = this.ring.freeFrames();
            if (inputFrames > free) {
              const deficit = inputFrames - free;
              const drainCap = this._effectiveMaxDrainFramesPerTick();
              const drainBudget =
                this._explicitMaxDrainFramesPerTick !== null
                  ? Math.min(deficit, drainCap)
                  : Math.min(deficit, Math.max(drainCap, this.ring.capacityFrames));
              const preDrained = this.ring.discardFrames(drainBudget);
              if (preDrained > 0) {
                this.framesDrained += preDrained;
              }
            }
          }

          const pktPolicy =
            msg.overflowPolicy !== undefined
              ? normalizeOverflowPolicy(msg.overflowPolicy, this._overflowPolicy)
              : this._overflowPolicy;
          this._pushOpts.frameCount = inputFrames;
          this._pushOpts.overflowPolicy = pktPolicy;
          this._pushOpts.maxFillFrames = this._effectiveMaxFillFrames();

          const pushedFrames =
            inputFrames > 0 ? this.ring.pushSamples(pcm, this._pushOpts) : 0;
          const droppedFrames = inputFrames > 0 ? this.ring._lastPushDroppedFrames : 0;
          const overflowed = inputFrames > 0 ? this.ring._lastPushOverflowed : false;
          this.totalPushedFrames += pushedFrames;
          if (overflowed || droppedFrames > 0) {
            this._droppedPackets += 1;
          }

          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              pushedFrames,
              droppedFrames,
              availableFrames: this.ring.availableFrames(),
            });
          }
          return;
        }

        if (msg.pullPlanar) {
          if (!this.ring) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                error: 'AudioMixerWorker not initialized',
              });
            }
            return;
          }
          const reqFrames = sanitizeNonNegInt(msg.frameCount, 128);
          const pulled = this.pullPlanar(reqFrames, msg.gainOverride);
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              framesRead: pulled.framesRead,
              frameCount: pulled.frameCount,
              channels: this.channels,
              availableFrames: this.ring.availableFrames(),
              bufferFillPct: this.ring.bufferFillPct,
            });
          }
          return;
        }

        if (msg.attachGuestPcmRing || msg.type === 'attach-guest-pcm-ring') {
          const desc =
            msg.ring && typeof msg.ring === 'object'
              ? msg.ring
              : msg.descriptor && typeof msg.descriptor === 'object'
                ? msg.descriptor
                : typeof msg.attachGuestPcmRing === 'object' && msg.attachGuestPcmRing !== null
                  ? msg.attachGuestPcmRing
                  : msg;
          try {
            const info = this.attachGuestPcmRing(desc);
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                attached: true,
                guestRingAttached: true,
                sampleRate: info.sampleRate,
                channels: info.channels,
                capacitySamples: info.capacitySamples,
                capacityFrames: info.capacityFrames,
                quantumFrames: info.quantumFrames,
              });
            }
          } catch (attachErr) {
            this._rejectedPackets += 1;
            if (safeId !== undefined) {
              this._safePostMessage({
                id: safeId,
                attached: false,
                guestRingAttached: this._guestRing !== null,
                error: sanitizeErrorMessage(attachErr),
              });
            }
          }
          return;
        }

        if (msg.drainGuestPcmRing || msg.type === 'drain-guest-pcm-ring') {
          const drainArg =
            typeof msg.drainGuestPcmRing === 'object' && msg.drainGuestPcmRing !== null
              ? msg.drainGuestPcmRing
              : msg;
          const res = this.drainGuestPcmRing(drainArg);
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              guestRingAttached: this._guestRing !== null,
              framesDrained: res.framesDrained,
              requestedFrames: res.requestedFrames,
              pushedFrames: res.pushedFrames,
              droppedFrames: res.droppedFrames,
              underrun: res.underrun,
              underrunFrames: res.underrunFrames,
              overrun: res.overrun,
              overrunDroppedFrames: res.overrunDroppedFrames,
              peakLevel: res.peakLevel,
              guestFramesDrained: this._guestFramesDrained,
              guestUnderruns: this._guestUnderruns,
              guestOverruns: this._guestOverruns,
              guestPeakLevel: this._guestPeakLevel,
              zeroAlloc: res.zeroAlloc,
              zeroAllocDrains: this._zeroAllocDrains,
              sanitizedNonFiniteSamples: this._sanitizedSamples,
              softClippedSamples: this._softClippedSamples,
              availableFrames: this.ring ? this.ring.availableFrames() : 0,
            });
          }
          return;
        }

        if (msg.detachGuestPcmRing || msg.type === 'detach-guest-pcm-ring') {
          const res = this.detachGuestPcmRing();
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              detached: true,
              wasAttached: res.wasAttached,
              guestRingAttached: false,
            });
          }
          return;
        }

        if (msg.drain || msg.drainRing) {
          const drainedFrames = this.drain(msg.maxFrames, {
            force: Boolean(msg.force),
          });
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              drainedFrames,
              framesDrained: this.framesDrained,
              outputState: this.outputState,
              availableFrames: this.ring ? this.ring.availableFrames() : 0,
            });
          }
          return;
        }

        if (
          msg.type === 'set-target-latency' ||
          msg.type === 'setTargetLatencyMs' ||
          msg.type === 'setTargetLatency' ||
          msg.setTargetLatency ||
          msg.setTargetLatencyMs
        ) {
          const rawMs =
            msg.targetLatencyMs !== undefined
              ? msg.targetLatencyMs
              : msg.latencyMs !== undefined
                ? msg.latencyMs
                : msg.ms;
          const appliedMs = this.setTargetLatencyMs(rawMs);
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              type: 'target-latency',
              targetLatencyMs: appliedMs,
              targetLatencyFrames: this.targetLatencyFrames,
              drainWatermarkFrames: this.drainWatermarkFrames,
              drainIntervalMs: this._drainIntervalMs,
            });
          }
          return;
        }

        if (
          msg.type === 'setFadeCurve' ||
          msg.type === 'set-fade-curve' ||
          msg.setFadeCurve
        ) {
          const appliedCurve = this.setFadeCurve(
            msg.fadeCurve !== undefined ? msg.fadeCurve : msg.curve
          );
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              type: 'fade-curve',
              fadeCurve: appliedCurve,
            });
          }
          return;
        }

        if (
          msg.setSampleRate ||
          msg.type === 'set-sample-rate' ||
          msg.type === 'setSampleRate'
        ) {
          const appliedRate = this.setSampleRate(msg.sampleRate);
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              type: 'sample-rate',
              sampleRate: appliedRate,
            });
          }
          return;
        }

        if (msg.setOutputState || msg.type === 'output-state' || msg.type === 'audio-context-state') {
          const attachedSab =
            msg.ringSab !== undefined && msg.ringSab !== null
              ? msg.ringSab
              : msg.audioSab !== undefined && msg.audioSab !== null
                ? msg.audioSab
                : null;
          if (isSharedArrayBuffer(attachedSab) && (!this.ring || this.ring.sab !== attachedSab)) {
            this.attachRingSab(attachedSab, msg);
          }
          const stateOpts = {};
          let hasStateOpts = false;
          if (msg.sampleRate !== undefined) {
            stateOpts.sampleRate = msg.sampleRate;
            hasStateOpts = true;
          }
          if (msg.resyncBacklog !== undefined) {
            stateOpts.resyncBacklog = Boolean(msg.resyncBacklog);
            hasStateOpts = true;
          }
          const nextState = this.setOutputState(
            msg.outputState !== undefined
              ? msg.outputState
              : msg.state !== undefined
                ? msg.state
                : msg.audioContextState,
            hasStateOpts ? stateOpts : null
          );
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              outputState: nextState,
              sampleRate: this.sampleRate,
            });
          }
          return;
        }

        if (msg.setVolume) {
          if (typeof msg.volume === 'number' && Number.isFinite(msg.volume)) {
            this.volume = clampVolume(msg.volume);
          }
          if (typeof msg.muted === 'boolean') {
            this.muted = msg.muted;
          }
          if (this.ring) {
            this.ring.setVolume(this.volume);
            this.ring.setMuted(this.muted);
          }
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              volume: this.volume,
              muted: this.muted,
              effectiveGain: this.effectiveGain(),
            });
          }
          return;
        }

        if (msg.setMute) {
          this.muted = Boolean(msg.muted);
          if (this.ring) {
            this.ring.setMuted(this.muted);
          }
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              volume: this.volume,
              muted: this.muted,
              effectiveGain: this.effectiveGain(),
            });
          }
          return;
        }

        if (msg.reset || msg.type === 'reset') {
          this.reset();
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              reset: true,
              outputState: this.outputState,
            });
          }
          return;
        }

        if (msg.stop || msg.destroy || msg.type === 'stop') {
          this.stop();
          if (safeId !== undefined) {
            this._safePostMessage({
              id: safeId,
              stopped: true,
              outputState: this.outputState,
            });
          }
          return;
        }

        if (
          msg.stats ||
          msg.type === 'audio-stats' ||
          msg.type === 'request-audio-stats'
        ) {
          const snapshot = buildAudioTelemetrySnapshot(this, { now: this._now });
          const reply = {
            type: 'audio-stats',
            id: safeId,
            stats: snapshot.stats,
            timestamp: snapshot.timestamp,
          };
          this._safePostMessage(reply);
          return;
        }
      } catch (err) {
        this._rejectedPackets += 1;
        if (safeId !== undefined) {
          this._safePostMessage({
            id: safeId,
            error: sanitizeErrorMessage(err),
          });
        }
      }
    }
  }

  const exportsObj = {
    AudioRingBuffer,
    AudioMixerWorker,
    GuestPcmRingReader,
    PcmScratchPool,
    OUTPUT_STATES,
    OVERFLOW_POLICIES,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    DEFAULT_CAPACITY_FRAMES,
    MAX_CAPACITY_FRAMES,
    MAX_CHANNELS,
    DEFAULT_DRAIN_INTERVAL_MS,
    DEFAULT_ACTIVE_PUMP_INTERVAL_MS,
    DEFAULT_TARGET_LATENCY_MS,
    MIN_TARGET_LATENCY_MS,
    MAX_TARGET_LATENCY_MS,
    DEFAULT_MAX_PCM_BYTES,
    DEFAULT_DECLICK_FRAMES,
    MAX_DECLICK_FRAMES,
    DEFAULT_FADE_CURVE,
    DEFAULT_SOFT_CLIP_KNEE,
    DEFAULT_OVERFLOW_POLICY,
    DEFAULT_MAX_FILL_RATIO,
    DEFAULT_UNDERRUN_RECOVERY_FRAMES,
    GUEST_PCM_RING_DEFAULT_SAMPLE_RATE,
    GUEST_PCM_RING_CHANNELS,
    GUEST_PCM_RING_DEFAULT_CAPACITY_SAMPLES,
    GUEST_PCM_RING_DEFAULT_QUANTUM_FRAMES,
    GUEST_PCM_RING_ACTIVE_BURST_CHUNK_FRAMES,
    PROGRESS_CHANNEL_NAME,
    DEFAULT_TELEMETRY_BROADCAST_INTERVAL_MS,
    ITU_BS775_CENTER_GAIN,
    ITU_BS775_SURROUND_GAIN,
    ITU_BS775_LFE_GAIN,
    MAX_RESAMPLE_RATIO,
    DEFAULT_MAX_RESAMPLE_FRAMES,
    SUPPORTED_DOWNMIX_CHANNELS,
    RING_HEADER_BYTES,
    clampTargetLatencyMs,
    deinterleavePcm,
    interleavePcm,
    extractSafePcmSamples,
    clonePcmForTransfer,
    sanitizeTransferList,
    softClipSample,
    applyFadeRamp,
    sanitizeAndSoftLimitBuffer,
    downmixToStereo,
    resampleLinear,
    buildAudioTelemetrySnapshot,
  };

  // UMD-style exports for browser globals and Node.js tests.
  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6Audio = exportsObj;
  } else if (root) {
    root.PlayGta6Audio = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }

  // Classic Dedicated Worker execution path.
  const isDedicatedWorker =
    typeof self !== 'undefined' &&
    typeof self.postMessage === 'function' &&
    typeof window === 'undefined';

  if (isDedicatedWorker) {
    const worker = new AudioMixerWorker(
      (payload, transfer) => {
        const safeTransfer = sanitizeTransferList(transfer);
        if (safeTransfer.length > 0) {
          self.postMessage(payload, safeTransfer);
        } else {
          self.postMessage(payload);
        }
      },
      { telemetryBroadcast: true }
    );
    self.playGta6AudioWorker = worker;
    self.onmessage = (ev) => {
      try {
        worker.handleMessage(ev ? ev.data : null);
      } catch (_err) {
        // Never allow an uncaught worker error to disrupt game boot.
      }
    };
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
