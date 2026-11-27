'use strict';

(function (root) {
  const DEFAULT_MAX_PASSES_PER_FRAME = 32;
  const DEFAULT_READBACK_SLOTS = 3;
  const DEFAULT_QUERY_RING_SLOTS = 1;
  const DEFAULT_HISTORY_SIZE = 120;
  const DEFAULT_FRAME_BUDGET_MS = 16.67;
  const DEFAULT_PASS_BUDGET_MS = 33.333;
  const DEFAULT_30FPS_BUDGET_MS = DEFAULT_PASS_BUDGET_MS;
  const TIMESTAMP_BYTE_SIZE = 8;
  const NS_PER_MS = 1e6;
  const PASS_KIND_UNKNOWN = 0;
  const PASS_KIND_RENDER = 1;
  const PASS_KIND_COMPUTE = 2;
  const PASS_KIND_COPY = 3;
  const PASS_KINDS = Object.freeze(['unknown', 'render', 'compute']);
  const GPU_TIME_MAX_KEYS = 1024;
  const GPU_TIME_SUPPORT = Object.freeze({
    UNKNOWN: 'unknown',
    UNAVAILABLE: 'unavailable',
    AVAILABLE: 'available',
  });
  const U64_MASK = 0xffffffffffffffffn;

  function normalizePassKindCode(kind) {
    if (kind === 'render' || kind === PASS_KIND_RENDER) {
      return PASS_KIND_RENDER;
    }
    if (kind === 'compute' || kind === PASS_KIND_COMPUTE) {
      return PASS_KIND_COMPUTE;
    }
    if (kind === 'copy' || kind === PASS_KIND_COPY) {
      return PASS_KIND_COPY;
    }
    if (typeof kind === 'string') {
      const lower = kind.trim().toLowerCase();
      if (lower === 'render') {
        return PASS_KIND_RENDER;
      }
      if (lower === 'compute') {
        return PASS_KIND_COMPUTE;
      }
      if (lower === 'copy') {
        return PASS_KIND_COPY;
      }
    }
    return PASS_KIND_UNKNOWN;
  }

  function passKindCodeToString(code) {
    if (code === PASS_KIND_RENDER) {
      return 'render';
    }
    if (code === PASS_KIND_COMPUTE) {
      return 'compute';
    }
    if (code === PASS_KIND_COPY) {
      return 'copy';
    }
    return 'unknown';
  }

  function normalizeShaderHash(value) {
    if (typeof value === 'bigint') {
      return value >= 0n ? value & U64_MASK : 0n;
    }
    if (typeof value === 'number') {
      if (Number.isSafeInteger(value) && value >= 0) {
        return BigInt(value);
      }
      return 0n;
    }
    if (typeof value === 'string') {
      const trimmed = value.trim();
      if (/^0x[0-9a-f]{1,16}$/i.test(trimmed)) {
        try {
          return BigInt(trimmed) & U64_MASK;
        } catch (_) {
          return 0n;
        }
      }
      if (/^[0-9a-f]{16}$/i.test(trimmed)) {
        try {
          return BigInt(`0x${trimmed}`) & U64_MASK;
        } catch (_) {
          return 0n;
        }
      }
    }
    return 0n;
  }

  function formatHashHex16(hash) {
    const normalized = normalizeShaderHash(hash);
    return `0x${normalized.toString(16).padStart(16, '0')}`;
  }

  /**
   * Bounded per-(shaderHash, passKind) GPU nanosecond accumulator matching
   * `playgta6::gpu::d3d12_guest::GpuTimeTable` in `runtime/gpu/gpu_time.h` (#1608).
   * Numbers only ever come from measured GPU timestamps; when unsupported nothing is invented.
   */
  class GpuTimeTable {
    constructor(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      this._maxKeys = normalizePositiveInt(opts.maxKeys, GPU_TIME_MAX_KEYS, 'maxKeys');
      this._support = GPU_TIME_SUPPORT.UNKNOWN;
      this._reason = '';
      this._map = new Map();
      this._other = { passes: 0, total_ns: 0, max_ns: 0 };
      this._framesMeasured = 0;
      this._framesSkipped = 0;
      this._passesSkipped = 0;
      this._frameGpuNs = 0;
    }

    setSupport(support, reason = '') {
      if (
        support === GPU_TIME_SUPPORT.AVAILABLE ||
        support === GPU_TIME_SUPPORT.UNAVAILABLE ||
        support === GPU_TIME_SUPPORT.UNKNOWN
      ) {
        this._support = support;
      }
      this._reason = typeof reason === 'string' ? reason : '';
    }

    get support() {
      return this._support;
    }

    get available() {
      return this._support === GPU_TIME_SUPPORT.AVAILABLE;
    }

    get tracked() {
      return this._map.size;
    }

    record(hash, kind, durationNs) {
      const ns =
        typeof durationNs === 'bigint'
          ? Number(durationNs > 0n ? durationNs : 0n)
          : typeof durationNs === 'number' && Number.isFinite(durationNs) && durationNs >= 0
            ? Math.round(durationNs)
            : 0;
      this._support = GPU_TIME_SUPPORT.AVAILABLE;
      const hashBig = normalizeShaderHash(hash);
      const kindStr = typeof kind === 'number' ? passKindCodeToString(kind) : passKindCodeToString(normalizePassKindCode(kind));
      const key = `${hashBig.toString(16)}:${kindStr}`;

      let entry = this._map.get(key);
      if (!entry) {
        if (this._map.size < this._maxKeys) {
          entry = {
            hashBig,
            hash: formatHashHex16(hashBig),
            kind: kindStr,
            passes: 0,
            total_ns: 0,
            max_ns: 0,
          };
          this._map.set(key, entry);
        } else {
          entry = this._other;
        }
      }
      entry.passes += 1;
      entry.total_ns += ns;
      if (ns > entry.max_ns) {
        entry.max_ns = ns;
      }
    }

    noteSkippedFrame() {
      this._framesSkipped += 1;
    }

    noteSkippedPasses(count = 1) {
      if (typeof count === 'number' && Number.isFinite(count) && count > 0) {
        this._passesSkipped += Math.floor(count);
      }
    }

    noteMeasuredFrame(gpuNs) {
      const ns =
        typeof gpuNs === 'bigint'
          ? Number(gpuNs > 0n ? gpuNs : 0n)
          : typeof gpuNs === 'number' && Number.isFinite(gpuNs) && gpuNs >= 0
            ? Math.round(gpuNs)
            : 0;
      this._framesMeasured += 1;
      this._frameGpuNs += ns;
    }

    measuredTotals() {
      return {
        frames: this._framesMeasured,
        gpu_ns: this._frameGpuNs,
      };
    }

    top(n = 32) {
      const limit = typeof n === 'number' && Number.isFinite(n) && n >= 0 ? Math.floor(n) : 32;
      if (limit === 0 || this._map.size === 0) {
        return [];
      }
      const list = Array.from(this._map.values());
      list.sort((a, b) => {
        if (a.total_ns !== b.total_ns) return b.total_ns - a.total_ns;
        if (a.hashBig !== b.hashBig) return a.hashBig < b.hashBig ? -1 : 1;
        if (a.kind !== b.kind) return a.kind < b.kind ? -1 : 1;
        return 0;
      });
      const out = [];
      const count = Math.min(limit, list.length);
      for (let i = 0; i < count; i += 1) {
        const e = list[i];
        out.push({
          hash: e.hash,
          kind: e.kind,
          passes: e.passes,
          total_ns: e.total_ns,
          max_ns: e.max_ns,
          avg_ns: e.passes > 0 ? Math.floor(e.total_ns / e.passes) : 0,
        });
      }
      return out;
    }

    getSnapshot(topN = 32) {
      let totalNs = this._other.total_ns;
      for (const e of this._map.values()) {
        totalNs += e.total_ns;
      }
      const available = this._support === GPU_TIME_SUPPORT.AVAILABLE;
      const snapshot = {
        available,
        support: this._support,
      };
      if (!available) {
        snapshot.reason =
          this._reason ||
          'no timestamp-query results were fed (no GPU executor attached, or the adapter lacks timestamp-query)';
      }
      snapshot.frames_measured = this._framesMeasured;
      snapshot.frames_skipped = this._framesSkipped;
      snapshot.passes_skipped = this._passesSkipped;
      snapshot.frame_gpu_ns = this._frameGpuNs;
      snapshot.total_ns = totalNs;
      snapshot.tracked = this._map.size;
      snapshot.other = {
        passes: this._other.passes,
        total_ns: this._other.total_ns,
      };
      snapshot.top = this.top(topN);
      return snapshot;
    }

    toJson(topN = 32) {
      return JSON.stringify(this.getSnapshot(topN));
    }

    reset({ preserveSupport = false } = {}) {
      if (!preserveSupport) {
        this._support = GPU_TIME_SUPPORT.UNKNOWN;
        this._reason = '';
      }
      this._map.clear();
      this._other = { passes: 0, total_ns: 0, max_ns: 0 };
      this._framesMeasured = 0;
      this._framesSkipped = 0;
      this._passesSkipped = 0;
      this._frameGpuNs = 0;
    }
  }

  // WebGPU GPUBufferUsage & GPUMapMode constants with headless/Node fallbacks.
  const BUFFER_USAGE = Object.freeze({
    MAP_READ:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.MAP_READ === 'number'
        ? GPUBufferUsage.MAP_READ
        : 0x0001,
    COPY_SRC:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_SRC === 'number'
        ? GPUBufferUsage.COPY_SRC
        : 0x0004,
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
    QUERY_RESOLVE:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.QUERY_RESOLVE === 'number'
        ? GPUBufferUsage.QUERY_RESOLVE
        : 0x0200,
  });

  const MAP_MODE = Object.freeze({
    READ:
      typeof GPUMapMode !== 'undefined' && typeof GPUMapMode.READ === 'number'
        ? GPUMapMode.READ
        : 0x0001,
  });

  function normalizePositiveInt(value, fallback, name) {
    if (value === undefined) {
      return fallback;
    }
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got ${value}`);
    }
    return value;
  }

  function normalizePositiveNumber(value, fallback, name) {
    if (value === undefined) {
      return fallback;
    }
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      throw new RangeError(`${name} must be a positive finite number, got ${value}`);
    }
    return value;
  }

  function percentileFromSorted(sorted, count, fraction) {
    if (count <= 0) {
      return 0;
    }
    const idx = Math.min(count - 1, Math.max(0, Math.ceil(fraction * count) - 1));
    return sorted[idx];
  }

  /**
   * Pure helper computing deterministic nearest-rank percentile statistics over
   * numeric duration samples (ms) without mutating the input collection.
   *
   * @param {ArrayLike<number> | Iterable<number> | null | undefined} samples
   * @returns {{ count: number, avgMs: number, minMs: number, p50Ms: number, p95Ms: number, p99Ms: number, maxMs: number }}
   */
  function computePercentiles(samples) {
    if (!samples || typeof samples !== 'object') {
      return {
        count: 0,
        avgMs: 0,
        minMs: 0,
        p50Ms: 0,
        p95Ms: 0,
        p99Ms: 0,
        maxMs: 0,
      };
    }

    const values = [];
    if (typeof samples.length === 'number') {
      for (let i = 0; i < samples.length; i += 1) {
        const v = samples[i];
        if (typeof v === 'number' && Number.isFinite(v)) {
          values.push(v);
        }
      }
    } else if (typeof samples[Symbol.iterator] === 'function') {
      for (const v of samples) {
        if (typeof v === 'number' && Number.isFinite(v)) {
          values.push(v);
        }
      }
    }

    const count = values.length;
    if (count === 0) {
      return {
        count: 0,
        avgMs: 0,
        minMs: 0,
        p50Ms: 0,
        p95Ms: 0,
        p99Ms: 0,
        maxMs: 0,
      };
    }

    values.sort((a, b) => a - b);

    let sum = 0;
    for (let i = 0; i < count; i += 1) {
      sum += values[i];
    }

    return {
      count,
      avgMs: sum / count,
      minMs: values[0],
      p50Ms: percentileFromSorted(values, count, 0.5),
      p95Ms: percentileFromSorted(values, count, 0.95),
      p99Ms: percentileFromSorted(values, count, 0.99),
      maxMs: values[count - 1],
    };
  }

  function hasTimestampQueryFeature(device) {
    if (!device || typeof device !== 'object') {
      return false;
    }
    const features = device.features;
    if (!features || typeof features.has !== 'function') {
      return false;
    }
    try {
      if (!features.has('timestamp-query')) {
        return false;
      }
    } catch (_) {
      return false;
    }
    return typeof device.createQuerySet === 'function' && typeof device.createBuffer === 'function';
  }

  function isSlotAvailable(slot) {
    if (!slot || slot.state !== 'idle' || !slot.buffer || slot.buffer.destroyed) {
      return false;
    }
    const mapState = slot.buffer.mapState;
    return mapState === undefined || mapState === 'unmapped';
  }

  function readTimestampsView(mappedRange, expectedCount) {
    if (mappedRange instanceof BigUint64Array) {
      return mappedRange;
    }
    if (ArrayBuffer.isView(mappedRange)) {
      const count = Math.min(
        expectedCount,
        Math.floor(mappedRange.byteLength / TIMESTAMP_BYTE_SIZE)
      );
      return new BigUint64Array(mappedRange.buffer, mappedRange.byteOffset, count);
    }
    if (mappedRange && typeof mappedRange.byteLength === 'number') {
      const count = Math.min(
        expectedCount,
        Math.floor(mappedRange.byteLength / TIMESTAMP_BYTE_SIZE)
      );
      return new BigUint64Array(mappedRange, 0, count);
    }
    return new BigUint64Array(0);
  }

  function computeDurationMs(startNs, endNs) {
    if (typeof startNs !== 'bigint' || typeof endNs !== 'bigint') {
      return 0;
    }
    if (endNs <= startNs) {
      return 0;
    }
    const durationMs = Number(endNs - startNs) / NS_PER_MS;
    if (!Number.isFinite(durationMs) || durationMs < 0) {
      return 0;
    }
    return durationMs;
  }

  function defaultNowMs() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  /**
   * Non-blocking WebGPU per-pass GPU profiler backed by optional `'timestamp-query'`
   * `GPUQuerySet` resolution and multi-slot staging buffer readback.
   *
   * Supports named pass profiling via `beginPass(label)` / `endPass()` and
   * `getTopPasses(limit)` bottleneck summaries. When WebGPU `'timestamp-query'`
   * is unavailable, `beginPass(label)` / `endPass()` gracefully degrade to CPU
   * `performance.now()` timing while keeping GPU-only methods (`getPassTimestampWrites`,
   * `resolveFrame`, `collectCompletedReadback`) as safe no-ops.
   */
  class GpuPassTimer {
    /**
     * @param {object} [options]
     * @param {GPUDevice} [options.device]
     * @param {number} [options.maxPassesPerFrame=32]
     * @param {number} [options.readbackSlots=3]
     * @param {number} [options.queryRingSlots=1]
     * @param {number} [options.historySize=120]
     * @param {number} [options.frameBudgetMs=16.67]
     * @param {number} [options.passBudgetMs=33.333]
     * @param {() => number} [options.now] Optional clock returning milliseconds (defaults to `performance.now()`).
     */
    constructor(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const device = opts.device || null;

      this._maxPassesPerFrame = normalizePositiveInt(
        opts.maxPassesPerFrame,
        DEFAULT_MAX_PASSES_PER_FRAME,
        'maxPassesPerFrame'
      );
      this._readbackSlotCount = normalizePositiveInt(
        opts.readbackSlots,
        DEFAULT_READBACK_SLOTS,
        'readbackSlots'
      );
      this._queryRingSlots = normalizePositiveInt(
        opts.queryRingSlots,
        DEFAULT_QUERY_RING_SLOTS,
        'queryRingSlots'
      );
      this._historySize = normalizePositiveInt(
        opts.historySize,
        DEFAULT_HISTORY_SIZE,
        'historySize'
      );
      this._frameBudgetMs = normalizePositiveNumber(
        opts.frameBudgetMs,
        DEFAULT_FRAME_BUDGET_MS,
        'frameBudgetMs'
      );
      this._passBudgetMs = normalizePositiveNumber(
        opts.passBudgetMs,
        DEFAULT_PASS_BUDGET_MS,
        'passBudgetMs'
      );
      if (opts.now !== undefined && typeof opts.now !== 'function') {
        throw new TypeError('now must be a function');
      }
      if (opts.onComplete !== undefined && opts.onComplete !== null && typeof opts.onComplete !== 'function') {
        throw new TypeError('onComplete must be a function');
      }
      this._timestampPeriodNs = normalizePositiveNumber(
        opts.timestampPeriodNs,
        1.0,
        'timestampPeriodNs'
      );
      this._now = opts.now || defaultNowMs;
      this._completeListeners = new Set();
      if (typeof opts.onComplete === 'function') {
        this._completeListeners.add(opts.onComplete);
      }
      this._gpuTimeTable = new GpuTimeTable({ maxKeys: opts.maxGpuTimeKeys });

      this._device = device;
      this._maxQueries = this._maxPassesPerFrame * 2;
      this._totalQueries = this._maxQueries * this._queryRingSlots;
      this._bufferByteSize = this._maxQueries * TIMESTAMP_BYTE_SIZE;

      this._supported = false;
      this._destroyed = false;
      this._deviceLost = false;

      this._querySet = null;
      this._resolveBuffer = null;
      this._readbackSlots = [];
      this._timestampWriteDescriptors = [];
      this._pendingRing = new Array(this._readbackSlotCount).fill(null);
      this._pendingHead = 0;
      this._pendingTail = 0;
      this._pendingCount = 0;

      // Pre-allocated query-slot ring recycling state (zero hot-path allocations).
      this._activeQueryRingSlice = 0;
      this._queryRingSliceUsed = new Uint32Array(this._queryRingSlots);
      this._querySlotRecycles = 0;
      this._querySlotWraps = 0;

      // Pre-allocated default pass names so unlabeled passes never allocate template strings.
      this._defaultPassNames = new Array(this._maxPassesPerFrame);
      for (let i = 0; i < this._maxPassesPerFrame; i += 1) {
        this._defaultPassNames[i] = `pass_${i}`;
      }

      // Pre-allocated per-frame pass recording storage (zero hot-path allocations).
      this._framePassNames = new Array(this._maxPassesPerFrame);
      this._framePassBeginIdx = new Uint32Array(this._maxPassesPerFrame);
      this._framePassEndIdx = new Uint32Array(this._maxPassesPerFrame);
      this._framePassKinds = new Uint8Array(this._maxPassesPerFrame);
      this._framePassHashes = new BigUint64Array(this._maxPassesPerFrame);
      this._framePassCount = 0;
      this._queryCursor = 0;

      // Pre-allocated active pass stack for beginPass(label, kind) / endPass().
      this._activePassLabels = new Array(this._maxPassesPerFrame);
      this._activePassStarts = new Float64Array(this._maxPassesPerFrame);
      this._activePassModes = new Uint8Array(this._maxPassesPerFrame); // 0 = GPU, 1 = CPU
      this._activePassKinds = new Uint8Array(this._maxPassesPerFrame);
      this._activePassDepth = 0;

      // Pre-allocated rolling frame total ring for share-of-frame % calculations.
      this._frameTotals = new Float64Array(this._historySize);
      this._frameTotalSeqs = new Float64Array(this._historySize);
      this._frameTotalWriteIdx = 0;
      this._frameTotalWindowCount = 0;
      this._frameSeq = 0;

      // Pre-allocated per-category (render / compute / other) rolling frame totals.
      this._renderFrameTotals = new Float64Array(this._historySize);
      this._computeFrameTotals = new Float64Array(this._historySize);
      this._otherFrameTotals = new Float64Array(this._historySize);
      this._renderPassCount = 0;
      this._computePassCount = 0;
      this._otherPassCount = 0;
      this._lastFrameRenderMs = 0;
      this._lastFrameComputeMs = 0;
      this._lastFrameOtherMs = 0;
      this._lastFramePassCount = 0;

      // Pre-allocated rolling pass duration ring for global p50PassMs / p95PassMs / maxPassMs stats.
      this._passDurations = new Float64Array(this._historySize);
      this._passDurationWriteIdx = 0;
      this._passDurationWindowCount = 0;

      // Reusable typed scratch buffer & stats record for zero-alloc percentile sorting.
      this._sortScratch = new Float64Array(this._historySize);
      this._windowStatsScratch = {
        count: 0,
        avgMs: 0,
        minMs: 0,
        p50Ms: 0,
        p95Ms: 0,
        p99Ms: 0,
        maxMs: 0,
      };

      // Open CPU fallback frame accumulators (active only when !supported and beginPass/endPass are used).
      this._cpuFrameOpen = false;
      this._cpuFrameSeq = 0;
      this._cpuFrameTotalMs = 0;
      this._cpuFrameRenderMs = 0;
      this._cpuFrameComputeMs = 0;
      this._cpuFrameOtherMs = 0;
      this._cpuFramePassCount = 0;

      this._resolvedFrames = 0;
      this._skippedFrames = 0;
      this._droppedPasses = 0;
      this._droppedQueryFrames = 0;
      this._mapErrors = 0;
      this._passesOverBudget = 0;
      this._lastFrameGpuMs = 0;
      this._resetGeneration = 0;
      this._passHistories = new Map();

      if (hasTimestampQueryFeature(device)) {
        this._initGpuResources(device);
      }
      if (this._supported) {
        this._gpuTimeTable.setSupport(GPU_TIME_SUPPORT.AVAILABLE);
      } else {
        this._gpuTimeTable.setSupport(
          GPU_TIME_SUPPORT.UNAVAILABLE,
          device ? 'adapter lacks timestamp-query' : 'no GPU device'
        );
      }
    }

    _initGpuResources(device) {
      let querySet = null;
      let resolveBuffer = null;
      const readbackSlots = [];

      try {
        querySet = device.createQuerySet({
          label: 'playgta6-gpu-timer-queries',
          type: 'timestamp',
          count: this._totalQueries,
        });
        if (!querySet) {
          throw new Error('createQuerySet returned empty query set');
        }

        resolveBuffer = device.createBuffer({
          label: 'playgta6-gpu-timer-resolve',
          size: this._bufferByteSize,
          usage: BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC,
        });
        if (!resolveBuffer) {
          throw new Error('createBuffer returned empty resolve buffer');
        }

        for (let i = 0; i < this._readbackSlotCount; i += 1) {
          const buffer = device.createBuffer({
            label: `playgta6-gpu-timer-readback-${i}`,
            size: this._bufferByteSize,
            usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
          });
          if (!buffer) {
            throw new Error(`createBuffer returned empty readback buffer ${i}`);
          }
          readbackSlots.push({
            index: i,
            buffer,
            state: 'idle',
            passNames: new Array(this._maxPassesPerFrame),
            passBeginIdx: new Uint32Array(this._maxPassesPerFrame),
            passEndIdx: new Uint32Array(this._maxPassesPerFrame),
            passKinds: new Uint8Array(this._maxPassesPerFrame),
            passHashes: new BigUint64Array(this._maxPassesPerFrame),
            passCount: 0,
            queryCount: 0,
            copyBytes: 0,
            generation: 0,
          });
        }

        const totalDescriptorSlots = this._queryRingSlots * this._maxPassesPerFrame;
        const timestampWriteDescriptors = new Array(totalDescriptorSlots);
        for (let slice = 0; slice < this._queryRingSlots; slice += 1) {
          const queryBase = slice * this._maxQueries;
          const descBase = slice * this._maxPassesPerFrame;
          for (let i = 0; i < this._maxPassesPerFrame; i += 1) {
            timestampWriteDescriptors[descBase + i] = Object.freeze({
              querySet,
              beginningOfPassWriteIndex: queryBase + i * 2,
              endOfPassWriteIndex: queryBase + i * 2 + 1,
            });
          }
        }

        this._querySet = querySet;
        this._resolveBuffer = resolveBuffer;
        this._readbackSlots = readbackSlots;
        this._timestampWriteDescriptors = timestampWriteDescriptors;
        this._supported = true;

        if (device && device.lost && typeof device.lost.then === 'function') {
          const handleDeviceLost = () => {
            this._deviceLost = true;
            if (this._destroyed) {
              return;
            }
            // Release any queued (not yet mapping) pending readback slots cleanly so stale
            // device buffers are never mapped after device loss and zero slots leak.
            while (this._pendingCount > 0) {
              const slot = this._pendingRing[this._pendingHead];
              this._pendingRing[this._pendingHead] = null;
              this._pendingHead = (this._pendingHead + 1) % this._readbackSlotCount;
              this._pendingCount -= 1;
              if (slot && slot.state === 'pending') {
                slot.state = 'idle';
                slot.passCount = 0;
                slot.passNames.fill(undefined);
                slot.passKinds.fill(PASS_KIND_UNKNOWN);
                slot.queryCount = 0;
                slot.copyBytes = 0;
                slot.generation = this._resetGeneration;
                this._droppedQueryFrames += 1;
              }
            }
          };
          device.lost.then(handleDeviceLost, handleDeviceLost);
        }
      } catch (_) {
        if (querySet && typeof querySet.destroy === 'function') {
          try {
            querySet.destroy();
          } catch (__) {
            // ignore cleanup errors
          }
        }
        if (resolveBuffer && typeof resolveBuffer.destroy === 'function') {
          try {
            resolveBuffer.destroy();
          } catch (__) {
            // ignore cleanup errors
          }
        }
        for (const slot of readbackSlots) {
          if (slot.buffer && typeof slot.buffer.destroy === 'function') {
            try {
              slot.buffer.destroy();
            } catch (__) {
              // ignore cleanup errors
            }
          }
        }
        this._querySet = null;
        this._resolveBuffer = null;
        this._readbackSlots = [];
        this._timestampWriteDescriptors = [];
        this._supported = false;
      }
    }

    get supported() {
      return this._supported;
    }

    get mode() {
      return this._supported ? 'gpu' : 'cpu';
    }

    get maxPassesPerFrame() {
      return this._maxPassesPerFrame;
    }

    get readbackSlots() {
      return this._readbackSlotCount;
    }

    get queryRingSlots() {
      return this._queryRingSlots;
    }

    get historySize() {
      return this._historySize;
    }

    get frameBudgetMs() {
      return this._frameBudgetMs;
    }

    get passBudgetMs() {
      return this._passBudgetMs;
    }

    get isDestroyed() {
      return this._destroyed;
    }

    get isDeviceLost() {
      return this._deviceLost;
    }

    get gpuTimeTable() {
      return this._gpuTimeTable;
    }

    /**
     * Register a completion callback invoked asynchronously whenever GPU timestamp readbacks
     * resolve (or when `complete()` is called directly). Returns an unsubscribe function.
     *
     * @param {(result: object) => void} callback
     * @returns {() => void}
     */
    onComplete(callback) {
      if (typeof callback !== 'function') {
        throw new TypeError('onComplete callback must be a function');
      }
      this._completeListeners.add(callback);
      return () => {
        this._completeListeners.delete(callback);
      };
    }

    setOnComplete(callback) {
      this._completeListeners.clear();
      if (typeof callback === 'function') {
        this._completeListeners.add(callback);
      }
    }

    _emitComplete(payload) {
      if (this._completeListeners.size === 0) {
        return;
      }
      for (const listener of this._completeListeners) {
        try {
          listener(payload);
        } catch (_) {
          // Telemetry callbacks must never interrupt rendering or readback cleanup.
        }
      }
    }

    _fillWindowStats(ring, count, target) {
      if (count <= 0) {
        target.count = 0;
        target.avgMs = 0;
        target.minMs = 0;
        target.p50Ms = 0;
        target.p95Ms = 0;
        target.p99Ms = 0;
        target.maxMs = 0;
        return target;
      }

      const scratch = this._sortScratch;
      for (let i = 0; i < count; i += 1) {
        scratch[i] = ring[i];
      }

      const view = count === scratch.length ? scratch : scratch.subarray(0, count);
      view.sort();

      let sum = 0;
      for (let i = 0; i < count; i += 1) {
        sum += view[i];
      }

      target.count = count;
      target.avgMs = sum / count;
      target.minMs = view[0];
      target.p50Ms = percentileFromSorted(view, count, 0.5);
      target.p95Ms = percentileFromSorted(view, count, 0.95);
      target.p99Ms = percentileFromSorted(view, count, 0.99);
      target.maxMs = view[count - 1];
      return target;
    }

    _computeWindowStats(ring, count) {
      return this._fillWindowStats(ring, count, {
        count: 0,
        avgMs: 0,
        minMs: 0,
        p50Ms: 0,
        p95Ms: 0,
        p99Ms: 0,
        maxMs: 0,
      });
    }

    _recordFrameTotal(totalMs, frameSeq, renderMs = 0, computeMs = 0, otherMs = 0) {
      const safeTotalMs =
        typeof totalMs === 'number' && Number.isFinite(totalMs) && totalMs >= 0 ? totalMs : 0;
      const idx = this._frameTotalWriteIdx;
      this._frameTotals[idx] = safeTotalMs;
      this._frameTotalSeqs[idx] = frameSeq;
      this._renderFrameTotals[idx] =
        typeof renderMs === 'number' && Number.isFinite(renderMs) && renderMs >= 0 ? renderMs : 0;
      this._computeFrameTotals[idx] =
        typeof computeMs === 'number' && Number.isFinite(computeMs) && computeMs >= 0
          ? computeMs
          : 0;
      this._otherFrameTotals[idx] =
        typeof otherMs === 'number' && Number.isFinite(otherMs) && otherMs >= 0 ? otherMs : 0;
      this._frameTotalWriteIdx = (idx + 1) % this._historySize;
      if (this._frameTotalWindowCount < this._historySize) {
        this._frameTotalWindowCount += 1;
      }
    }

    _finalizeCpuFrame() {
      if (this._cpuFramePassCount === 0) {
        this._cpuFrameOpen = false;
        this._cpuFrameSeq = 0;
        this._cpuFrameTotalMs = 0;
        this._cpuFrameRenderMs = 0;
        this._cpuFrameComputeMs = 0;
        this._cpuFrameOtherMs = 0;
        return false;
      }
      const frameSeq = this._cpuFrameSeq;
      const totalMs = this._cpuFrameTotalMs;
      const renderMs = this._cpuFrameRenderMs;
      const computeMs = this._cpuFrameComputeMs;
      const otherMs = this._cpuFrameOtherMs;
      const passCount = this._cpuFramePassCount;

      this._frameSeq = frameSeq;
      this._recordFrameTotal(totalMs, frameSeq, renderMs, computeMs, otherMs);
      this._resolvedFrames += 1;
      this._lastFrameGpuMs = totalMs;
      this._lastFrameRenderMs = renderMs;
      this._lastFrameComputeMs = computeMs;
      this._lastFrameOtherMs = otherMs;
      this._lastFramePassCount = passCount;

      this._cpuFrameOpen = false;
      this._cpuFrameSeq = 0;
      this._cpuFrameTotalMs = 0;
      this._cpuFrameRenderMs = 0;
      this._cpuFrameComputeMs = 0;
      this._cpuFrameOtherMs = 0;
      this._cpuFramePassCount = 0;
      return true;
    }

    /**
     * Resets per-frame pass metadata and query cursor for a new frame.
     * In CPU fallback mode, commits any pending passes recorded during the previous frame.
     */
    beginFrame() {
      if (this._destroyed) {
        return;
      }
      if (this._supported) {
        this._framePassCount = 0;
        this._queryCursor = 0;
        this._activePassDepth = 0;
        return;
      }
      if (this._cpuFramePassCount > 0) {
        this._finalizeCpuFrame();
      }
      this._activePassDepth = 0;
      this._cpuFrameOpen = true;
      this._cpuFrameSeq = 0;
      this._cpuFrameTotalMs = 0;
      this._cpuFrameRenderMs = 0;
      this._cpuFrameComputeMs = 0;
      this._cpuFrameOtherMs = 0;
      this._cpuFramePassCount = 0;
    }

    /**
     * Explicitly finalizes the current frame in CPU fallback mode.
     * In GPU mode this is a safe no-op (frames finalize when `collectCompletedReadback()` resolves).
     *
     * @returns {boolean} True if a CPU fallback frame with recorded passes was committed.
     */
    endFrame() {
      if (this._destroyed) {
        return false;
      }
      this._activePassDepth = 0;
      if (this._supported) {
        return false;
      }
      return this._finalizeCpuFrame();
    }

    /**
     * Begins a named pass scope without hot-path heap allocations.
     *
     * - When WebGPU `'timestamp-query'` is supported: allocates a query pair and returns
     *   the pre-allocated `timestampWrites` descriptor for `beginRenderPass` / `beginComputePass`.
     * - When `'timestamp-query'` is unavailable: captures a CPU `performance.now()` start
     *   timestamp and returns `undefined`.
     *
     * @param {string} [label]
     * @param {'render' | 'compute' | 'unknown'} [kind] Optional pass category/kind.
     * @returns {{ querySet: GPUQuerySet, beginningOfPassWriteIndex: number, endOfPassWriteIndex: number } | undefined}
     */
    beginPass(label, kind, shaderHash) {
      if (this._destroyed) {
        return undefined;
      }

      let passLabel = label;
      let passKind = kind;
      let passHash = shaderHash;
      if (label && typeof label === 'object') {
        passLabel = label.label !== undefined ? label.label : label.name;
        passKind = label.kind !== undefined ? label.kind : kind;
        passHash = label.hash !== undefined ? label.hash : label.shaderHash;
      }

      const kindCode = normalizePassKindCode(passKind);

      if (this._supported) {
        const descriptor = this.getPassTimestampWrites(passLabel, kindCode, passHash);
        if (descriptor !== undefined && this._activePassDepth < this._maxPassesPerFrame) {
          const depth = this._activePassDepth;
          this._activePassLabels[depth] = this._framePassNames[this._framePassCount - 1];
          this._activePassStarts[depth] = 0;
          this._activePassModes[depth] = 0;
          this._activePassKinds[depth] = kindCode;
          this._activePassDepth = depth + 1;
        }
        return descriptor;
      }

      if (
        this._activePassDepth >= this._maxPassesPerFrame ||
        this._cpuFramePassCount + this._activePassDepth >= this._maxPassesPerFrame
      ) {
        this._droppedPasses += 1;
        return undefined;
      }

      const depth = this._activePassDepth;
      const passSlotIdx = this._cpuFramePassCount + depth;
      const name =
        typeof passLabel === 'string' && passLabel.length > 0
          ? passLabel
          : this._defaultPassNames[passSlotIdx];
      this._activePassLabels[depth] = name;
      this._activePassStarts[depth] = this._now();
      this._activePassModes[depth] = 1;
      this._activePassKinds[depth] = kindCode;
      this._activePassDepth = depth + 1;
      return undefined;
    }

    /**
     * Ends the most recently started pass from `beginPass(label, kind)`.
     *
     * - In CPU fallback mode: records the elapsed `performance.now()` duration into the
     *   pass's rolling history and returns the duration in milliseconds.
     * - In GPU timestamp-query mode: closes the active pass scope and returns `null`
     *   (GPU timings are recorded asynchronously when `collectCompletedReadback()` resolves).
     *
     * @returns {number | null}
     */
    endPass() {
      if (this._destroyed || this._activePassDepth === 0) {
        return null;
      }

      const depth = this._activePassDepth - 1;
      this._activePassDepth = depth;
      const mode = this._activePassModes[depth];
      const label = this._activePassLabels[depth];
      const kindCode = this._activePassKinds[depth];
      this._activePassLabels[depth] = undefined;
      this._activePassKinds[depth] = PASS_KIND_UNKNOWN;

      if (mode === 0) {
        return null;
      }

      const endMs = this._now();
      const rawDelta = endMs - this._activePassStarts[depth];
      const durationMs =
        typeof rawDelta === 'number' && Number.isFinite(rawDelta) && rawDelta > 0 ? rawDelta : 0;

      if (this._cpuFramePassCount === 0) {
        this._cpuFrameSeq = this._frameSeq + 1;
        this._cpuFrameOpen = true;
      }
      this._cpuFrameTotalMs += durationMs;
      if (kindCode === PASS_KIND_RENDER) {
        this._cpuFrameRenderMs += durationMs;
      } else if (kindCode === PASS_KIND_COMPUTE) {
        this._cpuFrameComputeMs += durationMs;
      } else {
        this._cpuFrameOtherMs += durationMs;
      }
      this._cpuFramePassCount += 1;
      this._recordPassSample(label, durationMs, this._cpuFrameSeq, kindCode);
      return durationMs;
    }

    /**
     * Allocates a `(beginningOfPassWriteIndex, endOfPassWriteIndex)` pair for a render
     * or compute pass descriptor's `timestampWrites` field.
     *
     * Returns `undefined` when `'timestamp-query'` is unsupported, the timer is destroyed,
     * or the per-frame query budget is exhausted.
     *
     * @param {string} [passName]
     * @param {'render' | 'compute' | 'unknown' | number} [kind] Optional pass category/kind.
     * @returns {{ querySet: GPUQuerySet, beginningOfPassWriteIndex: number, endOfPassWriteIndex: number } | undefined}
     */
    getPassTimestampWrites(passName, kind, shaderHash) {
      if (!this._supported || this._destroyed || !this._querySet) {
        return undefined;
      }
      if (this._queryCursor + 2 > this._maxQueries) {
        this._droppedPasses += 1;
        this._gpuTimeTable.noteSkippedPasses(1);
        return undefined;
      }

      let rawName = passName;
      let rawKind = kind;
      let rawHash = shaderHash;
      if (passName && typeof passName === 'object') {
        rawName = passName.label !== undefined ? passName.label : passName.name;
        rawKind = passName.kind !== undefined ? passName.kind : kind;
        rawHash = passName.hash !== undefined ? passName.hash : passName.shaderHash;
      }

      const passIdx = this._framePassCount;
      const beginningOfPassWriteIndex = this._queryCursor;
      const endOfPassWriteIndex = this._queryCursor + 1;
      this._queryCursor += 2;
      this._framePassCount = passIdx + 1;

      const name =
        typeof rawName === 'string' && rawName.length > 0
          ? rawName
          : this._defaultPassNames[passIdx];

      this._framePassNames[passIdx] = name;
      this._framePassBeginIdx[passIdx] = beginningOfPassWriteIndex;
      this._framePassEndIdx[passIdx] = endOfPassWriteIndex;
      this._framePassKinds[passIdx] = normalizePassKindCode(rawKind);
      this._framePassHashes[passIdx] = normalizeShaderHash(rawHash);

      const descriptorIdx = this._activeQueryRingSlice * this._maxPassesPerFrame + passIdx;
      return this._timestampWriteDescriptors[descriptorIdx];
    }

    /**
     * Encodes `resolveQuerySet` + `copyBufferToBuffer` into an available readback slot.
     * Skips cleanly (incrementing `skippedFrames`) if all readback slots are currently
     * in-flight or mapped so rendering never stalls.
     *
     * In CPU fallback mode, commits any passes recorded via `beginPass()`/`endPass()`
     * for the current frame and returns `false` (no GPU resolve commands encoded).
     *
     * @param {GPUCommandEncoder} encoder
     * @returns {boolean} True if resolve commands were encoded, false if skipped or no-op.
     */
    resolveFrame(encoder) {
      if (this._destroyed) {
        return false;
      }
      if (!this._supported) {
        if (this._cpuFramePassCount > 0) {
          this._finalizeCpuFrame();
        }
        this._activePassDepth = 0;
        return false;
      }
      if (!this._querySet || !this._resolveBuffer) {
        return false;
      }
      const passCount = this._framePassCount;
      if (passCount === 0) {
        return false;
      }
      if (
        !encoder ||
        typeof encoder.resolveQuerySet !== 'function' ||
        typeof encoder.copyBufferToBuffer !== 'function'
      ) {
        return false;
      }

      if (this._deviceLost) {
        this._droppedQueryFrames += 1;
        this._gpuTimeTable.noteSkippedFrame();
        this._gpuTimeTable.noteSkippedPasses(passCount);
        this._framePassCount = 0;
        this._queryCursor = 0;
        this._activePassDepth = 0;
        return false;
      }

      let freeSlot = null;
      for (let i = 0; i < this._readbackSlots.length; i += 1) {
        if (isSlotAvailable(this._readbackSlots[i])) {
          freeSlot = this._readbackSlots[i];
          break;
        }
      }

      if (!freeSlot) {
        this._skippedFrames += 1;
        this._droppedQueryFrames += 1;
        this._gpuTimeTable.noteSkippedFrame();
        this._gpuTimeTable.noteSkippedPasses(passCount);
        this._framePassCount = 0;
        this._queryCursor = 0;
        this._activePassDepth = 0;
        return false;
      }

      const sliceIdx = this._activeQueryRingSlice;
      const firstQuery = sliceIdx * this._maxQueries;
      const queryCount = this._queryCursor;
      const copyBytes = queryCount * TIMESTAMP_BYTE_SIZE;

      try {
        encoder.resolveQuerySet(this._querySet, firstQuery, queryCount, this._resolveBuffer, 0);
        encoder.copyBufferToBuffer(this._resolveBuffer, 0, freeSlot.buffer, 0, copyBytes);
      } catch (_) {
        this._droppedQueryFrames += 1;
        this._gpuTimeTable.noteSkippedFrame();
        this._gpuTimeTable.noteSkippedPasses(passCount);
        this._framePassCount = 0;
        this._queryCursor = 0;
        this._activePassDepth = 0;
        return false;
      }

      freeSlot.state = 'pending';
      for (let i = 0; i < passCount; i += 1) {
        freeSlot.passNames[i] = this._framePassNames[i];
        freeSlot.passBeginIdx[i] = this._framePassBeginIdx[i];
        freeSlot.passEndIdx[i] = this._framePassEndIdx[i];
        freeSlot.passKinds[i] = this._framePassKinds[i];
        freeSlot.passHashes[i] = this._framePassHashes[i];
      }
      freeSlot.passCount = passCount;
      freeSlot.queryCount = queryCount;
      freeSlot.copyBytes = copyBytes;
      freeSlot.generation = this._resetGeneration;

      this._pendingRing[this._pendingTail] = freeSlot;
      this._pendingTail = (this._pendingTail + 1) % this._readbackSlotCount;
      this._pendingCount += 1;

      if (this._queryRingSliceUsed[sliceIdx] > 0) {
        this._querySlotRecycles += 1;
      }
      this._queryRingSliceUsed[sliceIdx] += 1;
      const nextSlice = (sliceIdx + 1) % this._queryRingSlots;
      if (nextSlice === 0) {
        this._querySlotWraps += 1;
      }
      this._activeQueryRingSlice = nextSlice;

      this._framePassCount = 0;
      this._queryCursor = 0;
      this._activePassDepth = 0;
      return true;
    }

    _recordPassSample(name, durationMs, frameSeq, kindCode = PASS_KIND_UNKNOWN) {
      const safeDurationMs =
        typeof durationMs === 'number' && Number.isFinite(durationMs) && durationMs >= 0
          ? durationMs
          : 0;

      this._passDurations[this._passDurationWriteIdx] = safeDurationMs;
      this._passDurationWriteIdx = (this._passDurationWriteIdx + 1) % this._historySize;
      if (this._passDurationWindowCount < this._historySize) {
        this._passDurationWindowCount += 1;
      }

      if (kindCode === PASS_KIND_RENDER) {
        this._renderPassCount += 1;
      } else if (kindCode === PASS_KIND_COMPUTE) {
        this._computePassCount += 1;
      } else {
        this._otherPassCount += 1;
      }

      let entry = this._passHistories.get(name);
      if (!entry) {
        entry = {
          samples: new Float64Array(this._historySize),
          writeIdx: 0,
          windowCount: 0,
          count: 0,
          lastMs: 0,
          kind: kindCode,
          budgetExceedCount: 0,
          passesOverBudget: 0,
          frameSamples: new Float64Array(this._historySize),
          frameSeqs: new Float64Array(this._historySize),
          frameWriteIdx: 0,
          frameWindowCount: 0,
          lastFrameSeq: 0,
          lastFrameSlotIdx: -1,
        };
        this._passHistories.set(name, entry);
      } else if (kindCode !== PASS_KIND_UNKNOWN) {
        entry.kind = kindCode;
      }

      entry.samples[entry.writeIdx] = safeDurationMs;
      entry.writeIdx = (entry.writeIdx + 1) % this._historySize;
      if (entry.windowCount < this._historySize) {
        entry.windowCount += 1;
      }
      entry.count += 1;
      entry.lastMs = safeDurationMs;
      if (safeDurationMs > this._frameBudgetMs) {
        entry.budgetExceedCount += 1;
      }
      if (safeDurationMs > this._passBudgetMs) {
        entry.passesOverBudget += 1;
        this._passesOverBudget += 1;
      }

      if (typeof frameSeq === 'number' && frameSeq > 0) {
        if (entry.lastFrameSeq === frameSeq && entry.lastFrameSlotIdx >= 0) {
          entry.frameSamples[entry.lastFrameSlotIdx] += safeDurationMs;
        } else {
          const slotIdx = entry.frameWriteIdx;
          entry.frameSamples[slotIdx] = safeDurationMs;
          entry.frameSeqs[slotIdx] = frameSeq;
          entry.lastFrameSeq = frameSeq;
          entry.lastFrameSlotIdx = slotIdx;
          entry.frameWriteIdx = (slotIdx + 1) % this._historySize;
          if (entry.frameWindowCount < this._historySize) {
            entry.frameWindowCount += 1;
          }
        }
      }
    }

    /**
     * Asynchronously maps the oldest submitted readback buffer, converts `BigUint64Array`
     * nanosecond timestamps to milliseconds, updates per-pass rolling stats, unmaps the
     * buffer, and returns the frame's pass timings.
     *
     * @returns {Promise<{ totalGpuMs: number, passes: Array<{ name: string, durationMs: number }> } | null>}
     */
    async collectCompletedReadback() {
      if (!this._supported || this._destroyed) {
        return null;
      }
      if (this._pendingCount === 0) {
        return null;
      }

      const slot = this._pendingRing[this._pendingHead];
      this._pendingRing[this._pendingHead] = null;
      this._pendingHead = (this._pendingHead + 1) % this._readbackSlotCount;
      this._pendingCount -= 1;

      if (!slot || !slot.buffer || slot.state !== 'pending') {
        if (!this._destroyed) {
          this._droppedQueryFrames += 1;
        }
        return null;
      }

      const buffer = slot.buffer;
      const passCount = slot.passCount;
      const queryCount = slot.queryCount;
      const copyBytes = slot.copyBytes;
      const slotGeneration = slot.generation;

      slot.state = 'mapping';

      let mapped = false;
      let unmapAttempted = false;
      try {
        if (buffer.destroyed) {
          throw new Error('Readback buffer was destroyed before mapAsync');
        }
        if (typeof buffer.mapAsync !== 'function') {
          throw new TypeError('Readback buffer does not implement mapAsync');
        }

        await buffer.mapAsync(MAP_MODE.READ);
        mapped = true;

        if (
          this._destroyed ||
          slot.buffer !== buffer ||
          slotGeneration !== this._resetGeneration
        ) {
          return null;
        }

        if (typeof buffer.getMappedRange !== 'function') {
          throw new TypeError('Readback buffer does not implement getMappedRange');
        }

        const range = buffer.getMappedRange(0, copyBytes);
        if (!range) {
          throw new Error('Readback buffer getMappedRange returned empty range');
        }
        const timestamps = readTimestampsView(range, queryCount);

        let totalGpuMs = 0;
        let totalGpuNs = 0;
        let renderMs = 0;
        let computeMs = 0;
        let otherMs = 0;
        let skippedPasses = 0;
        const passes = new Array(passCount);
        const detailedPasses = new Array(passCount);
        const periodNs = this._timestampPeriodNs;

        for (let i = 0; i < passCount; i += 1) {
          const passName = slot.passNames[i];
          const beginIdx = slot.passBeginIdx[i];
          const endIdx = slot.passEndIdx[i];
          const kindCode = slot.passKinds[i];
          const shaderHash = slot.passHashes[i];
          const startNs = beginIdx < timestamps.length ? timestamps[beginIdx] : 0n;
          const endNs = endIdx < timestamps.length ? timestamps[endIdx] : 0n;
          const validDelta = typeof startNs === 'bigint' && typeof endNs === 'bigint' && endNs > startNs;
          const rawDeltaNs = validDelta ? Math.round(Number(endNs - startNs) * periodNs) : 0;
          const durationNs = Number.isFinite(rawDeltaNs) && rawDeltaNs > 0 ? rawDeltaNs : 0;
          const durationMs =
            periodNs === 1.0
              ? computeDurationMs(startNs, endNs)
              : durationNs / NS_PER_MS;
          totalGpuMs += durationMs;
          totalGpuNs += durationNs;
          if (kindCode === PASS_KIND_RENDER) {
            renderMs += durationMs;
          } else if (kindCode === PASS_KIND_COMPUTE) {
            computeMs += durationMs;
          } else {
            otherMs += durationMs;
          }
          if (validDelta && durationNs > 0) {
            this._gpuTimeTable.record(shaderHash, kindCode, durationNs);
          } else {
            skippedPasses += 1;
          }
          passes[i] = { name: passName, durationMs };
          detailedPasses[i] = {
            name: passName,
            kind: passKindCodeToString(kindCode),
            hash: formatHashHex16(shaderHash),
            durationNs,
            durationMs,
          };
        }

        if (typeof buffer.unmap === 'function') {
          unmapAttempted = true;
          mapped = false;
          buffer.unmap();
        } else {
          mapped = false;
        }

        if (skippedPasses > 0) {
          this._gpuTimeTable.noteSkippedPasses(skippedPasses);
        }
        this._gpuTimeTable.noteMeasuredFrame(totalGpuNs);

        const frameSeq = this._frameSeq + 1;
        this._frameSeq = frameSeq;

        for (let i = 0; i < passCount; i += 1) {
          this._recordPassSample(passes[i].name, passes[i].durationMs, frameSeq, slot.passKinds[i]);
        }

        this._recordFrameTotal(totalGpuMs, frameSeq, renderMs, computeMs, otherMs);
        this._resolvedFrames += 1;
        this._lastFrameGpuMs = totalGpuMs;
        this._lastFrameRenderMs = renderMs;
        this._lastFrameComputeMs = computeMs;
        this._lastFrameOtherMs = otherMs;
        this._lastFramePassCount = passCount;

        this._emitComplete({
          mode: 'gpu',
          supported: true,
          frameSeq,
          totalGpuMs,
          totalGpuNs,
          skippedPasses,
          passes: detailedPasses,
        });

        return { totalGpuMs, passes };
      } catch (_) {
        if (!this._destroyed && slotGeneration === this._resetGeneration) {
          this._mapErrors += 1;
          this._droppedQueryFrames += 1;
          this._gpuTimeTable.noteSkippedPasses(passCount);
          this._gpuTimeTable.noteSkippedFrame();
        }
        return null;
      } finally {
        const stillMapped =
          !unmapAttempted &&
          buffer &&
          !buffer.destroyed &&
          (mapped || buffer.mapState === 'mapped');
        if (stillMapped && typeof buffer.unmap === 'function') {
          try {
            buffer.unmap();
          } catch (_) {
            // ignore secondary unmap errors if buffer was destroyed or device was lost
          }
        }
        if (!this._destroyed && slot.buffer === buffer) {
          slot.state = 'idle';
          slot.passCount = 0;
          slot.passNames.fill(undefined);
          slot.passKinds.fill(PASS_KIND_UNKNOWN);
          slot.passHashes.fill(0n);
          slot.queryCount = 0;
          slot.copyBytes = 0;
          slot.generation = this._resetGeneration;
        }
      }
    }

    /**
     * Direct timestamp completion hook mirroring `GpuTimeRing::complete()` in `runtime/gpu/gpu_time.h` (#1608).
     * Accepts resolved timestamp ticks (2 per pass: `[begin, end, ...]`), converts valid deltas to
     * nanoseconds/milliseconds without inventing numbers for zero or non-increasing pairs, updates
     * `GpuTimeTable` and rolling stats, and fires registered `onComplete` listeners.
     *
     * @param {BigUint64Array | Array<bigint | number>} ticks
     * @param {{
     *   passes?: Array<{ name?: string, label?: string, kind?: string | number, hash?: bigint | number | string, shaderHash?: bigint | number | string }>,
     *   periodNs?: number,
     *   requireNonZeroStart?: boolean
     * }} [options]
     */
    complete(ticks, options = {}) {
      if (this._destroyed || !ticks || typeof ticks.length !== 'number' || ticks.length < 2) {
        return null;
      }
      const opts = options && typeof options === 'object' ? options : {};
      const periodNs =
        opts.periodNs !== undefined
          ? normalizePositiveNumber(opts.periodNs, this._timestampPeriodNs, 'periodNs')
          : this._timestampPeriodNs;
      const requireNonZeroStart = opts.requireNonZeroStart !== false;
      const passMeta = Array.isArray(opts.passes) ? opts.passes : null;
      const passCount = passMeta ? passMeta.length : Math.floor(ticks.length / 2);
      if (passCount <= 0) {
        return null;
      }

      let totalGpuNs = 0;
      let totalGpuMs = 0;
      let renderMs = 0;
      let computeMs = 0;
      let otherMs = 0;
      let measuredPasses = 0;
      let skippedPasses = 0;
      const passes = new Array(passCount);

      const frameSeq = this._frameSeq + 1;
      this._frameSeq = frameSeq;

      for (let i = 0; i < passCount; i += 1) {
        const meta = passMeta && passMeta[i] && typeof passMeta[i] === 'object' ? passMeta[i] : null;
        const name =
          meta && typeof (meta.name || meta.label) === 'string' && (meta.name || meta.label).length > 0
            ? meta.name || meta.label
            : i < this._defaultPassNames.length
              ? this._defaultPassNames[i]
              : `pass_${i}`;
        const kindCode = normalizePassKindCode(meta ? meta.kind : PASS_KIND_UNKNOWN);
        const hashBig = normalizeShaderHash(meta ? (meta.hash !== undefined ? meta.hash : meta.shaderHash) : 0n);

        const idxBegin = i * 2;
        const idxEnd = idxBegin + 1;
        let startTick = 0n;
        let endTick = 0n;
        if (idxEnd < ticks.length) {
          const rawStart = ticks[idxBegin];
          const rawEnd = ticks[idxEnd];
          startTick = typeof rawStart === 'bigint' ? rawStart : BigInt(Math.max(0, Math.trunc(Number(rawStart) || 0)));
          endTick = typeof rawEnd === 'bigint' ? rawEnd : BigInt(Math.max(0, Math.trunc(Number(rawEnd) || 0)));
        }

        const valid = (requireNonZeroStart ? startTick !== 0n : true) && endTick > startTick;
        let durationNs = 0;
        let durationMs = 0;
        if (valid) {
          const scaled = Math.floor(Number(endTick - startTick) * periodNs);
          if (Number.isFinite(scaled) && scaled > 0) {
            durationNs = scaled;
            durationMs = durationNs / NS_PER_MS;
          }
        }

        if (valid && durationNs > 0) {
          measuredPasses += 1;
          totalGpuNs += durationNs;
          totalGpuMs += durationMs;
          if (kindCode === PASS_KIND_RENDER) {
            renderMs += durationMs;
          } else if (kindCode === PASS_KIND_COMPUTE) {
            computeMs += durationMs;
          } else {
            otherMs += durationMs;
          }
          this._gpuTimeTable.record(hashBig, kindCode, durationNs);
          this._recordPassSample(name, durationMs, frameSeq, kindCode);
        } else {
          skippedPasses += 1;
        }

        passes[i] = {
          name,
          kind: passKindCodeToString(kindCode),
          hash: formatHashHex16(hashBig),
          durationNs,
          durationMs,
        };
      }

      if (skippedPasses > 0) {
        this._gpuTimeTable.noteSkippedPasses(skippedPasses);
      }
      this._gpuTimeTable.noteMeasuredFrame(totalGpuNs);
      this._recordFrameTotal(totalGpuMs, frameSeq, renderMs, computeMs, otherMs);
      this._resolvedFrames += 1;
      this._lastFrameGpuMs = totalGpuMs;
      this._lastFrameRenderMs = renderMs;
      this._lastFrameComputeMs = computeMs;
      this._lastFrameOtherMs = otherMs;
      this._lastFramePassCount = passCount;

      const result = {
        mode: 'gpu',
        supported: this._supported || this._gpuTimeTable.available,
        frameSeq,
        totalGpuNs,
        totalGpuMs,
        measuredPasses,
        skippedPasses,
        passes,
      };
      this._emitComplete(result);
      return result;
    }

    /**
     * Asynchronously drains all currently queued pending readback slots without blocking the render loop.
     *
     * @returns {Promise<Array<{ totalGpuMs: number, passes: Array<{ name: string, durationMs: number }> }>>}
     */
    async flushReadbacks() {
      const results = [];
      if (!this._supported || this._destroyed) {
        return results;
      }
      const maxToDrain = this._pendingCount;
      for (let i = 0; i < maxToDrain; i += 1) {
        const res = await this.collectCompletedReadback();
        if (res) {
          results.push(res);
        }
      }
      return results;
    }

    /**
     * Return a per-pipeline GPU time telemetry snapshot matching `GpuTimeTable::json()`
     * in `runtime/gpu/gpu_time.h` (#1608).
     */
    getGpuTimeStats(topN = 32) {
      return this._gpuTimeTable.getSnapshot(topN);
    }

    getGpuTimeJson(topN = 32) {
      return this._gpuTimeTable.toJson(topN);
    }

    /**
     * Returns rolling percentile statistics and budget-exceedance spike count for
     * a single pass label, or `null` if no samples have been recorded for it.
     *
     * @param {string} label
     * @returns {{
     *   count: number,
     *   totalCount: number,
     *   lastMs: number,
     *   avgMs: number,
     *   minMs: number,
     *   p50Ms: number,
     *   p95Ms: number,
     *   p99Ms: number,
     *   maxMs: number,
     *   budgetExceedCount: number
     * } | null}
     */
    getPassPercentiles(label) {
      if (typeof label !== 'string' || label.length === 0) {
        return null;
      }
      const entry = this._passHistories.get(label);
      if (!entry || entry.windowCount === 0) {
        return null;
      }
      const percentiles = this._computeWindowStats(entry.samples, entry.windowCount);
      return {
        count: percentiles.count,
        totalCount: entry.count,
        lastMs: entry.lastMs,
        avgMs: percentiles.avgMs,
        minMs: percentiles.minMs,
        p50Ms: percentiles.p50Ms,
        p95Ms: percentiles.p95Ms,
        p99Ms: percentiles.p99Ms,
        maxMs: percentiles.maxMs,
        budgetExceedCount: entry.budgetExceedCount,
      };
    }

    /**
     * Returns a sanitized numeric telemetry summary including budget configuration,
     * rolling window capacity, per-pass percentile breakdowns, and total spike count.
     *
     * @returns {{
     *   supported: boolean,
     *   frameBudgetMs: number,
     *   historySize: number,
     *   resolvedFrames: number,
     *   skippedFrames: number,
     *   droppedPasses: number,
     *   lastFrameGpuMs: number,
     *   budgetExceedCount: number,
     *   passes: Record<string, {
     *     count: number,
     *     totalCount: number,
     *     lastMs: number,
     *     avgMs: number,
     *     minMs: number,
     *     p50Ms: number,
     *     p95Ms: number,
     *     p99Ms: number,
     *     maxMs: number,
     *     budgetExceedCount: number
     *   }>
     * }}
     */
    getSummary() {
      const passes = {};
      let budgetExceedCount = 0;

      for (const name of this._passHistories.keys()) {
        const passStats = this.getPassPercentiles(name);
        if (passStats) {
          passes[name] = passStats;
          budgetExceedCount += passStats.budgetExceedCount;
        }
      }

      return {
        supported: this._supported,
        frameBudgetMs: this._frameBudgetMs,
        historySize: this._historySize,
        resolvedFrames: this._resolvedFrames,
        skippedFrames: this._skippedFrames,
        droppedPasses: this._droppedPasses,
        lastFrameGpuMs: this._lastFrameGpuMs,
        budgetExceedCount,
        passes,
      };
    }

    /**
     * Returns the slowest passes ranked by rolling window frame contribution (highest
     * share-of-frame % first), computing rolling per-pass average ms, max ms, and
     * share-of-frame % over the active `historySize` rolling window.
     *
     * @param {number} [limit] Maximum number of passes to return (defaults to all active passes).
     * @returns {Array<{
     *   label: string,
     *   name: string,
     *   count: number,
     *   totalCount: number,
     *   avgMs: number,
     *   maxMs: number,
     *   lastMs: number,
     *   totalMs: number,
     *   shareOfFramePct: number,
     *   mode: 'gpu' | 'cpu'
     * }>}
     */
    getTopPasses(limit) {
      let maxResults = Infinity;
      if (limit !== undefined && limit !== null) {
        if (typeof limit !== 'number' || Number.isNaN(limit) || limit <= 0) {
          return [];
        }
        maxResults = Number.isFinite(limit) ? Math.floor(limit) : Infinity;
        if (maxResults <= 0) {
          return [];
        }
      }

      const hasOpenCpuFrame = !this._supported && this._cpuFramePassCount > 0;
      const finalizedCount = this._frameTotalWindowCount;
      if (finalizedCount === 0 && !hasOpenCpuFrame) {
        return [];
      }

      let maxSeq = 0;
      let minSeq = 0;
      let rollingFrameTotalMs = 0;

      if (hasOpenCpuFrame) {
        const activeFrameCount = Math.min(this._historySize, finalizedCount + 1);
        maxSeq = this._cpuFrameSeq;
        minSeq = maxSeq - activeFrameCount + 1;
        rollingFrameTotalMs = this._cpuFrameTotalMs;
        for (let i = 0; i < finalizedCount; i += 1) {
          const seq = this._frameTotalSeqs[i];
          if (seq >= minSeq && seq <= maxSeq) {
            rollingFrameTotalMs += this._frameTotals[i];
          }
        }
      } else {
        maxSeq = this._frameSeq;
        minSeq = maxSeq - finalizedCount + 1;
        for (let i = 0; i < finalizedCount; i += 1) {
          rollingFrameTotalMs += this._frameTotals[i];
        }
      }

      const mode = this._supported ? 'gpu' : 'cpu';
      const ranked = [];

      for (const [label, entry] of this._passHistories.entries()) {
        let activeCount = 0;
        let sum = 0;
        let maxMs = 0;
        let lastMs = 0;
        let latestSeq = -1;

        for (let i = 0; i < entry.frameWindowCount; i += 1) {
          const seq = entry.frameSeqs[i];
          if (seq >= minSeq && seq <= maxSeq) {
            const v = entry.frameSamples[i];
            activeCount += 1;
            sum += v;
            if (v > maxMs) {
              maxMs = v;
            }
            if (seq >= latestSeq) {
              latestSeq = seq;
              lastMs = v;
            }
          }
        }

        if (activeCount === 0) {
          continue;
        }

        const avgMs = sum / activeCount;
        const shareOfFramePct =
          rollingFrameTotalMs > 0 ? (sum / rollingFrameTotalMs) * 100 : 0;

        ranked.push({
          label,
          name: label,
          count: activeCount,
          totalCount: entry.count,
          avgMs,
          maxMs,
          lastMs,
          totalMs: sum,
          shareOfFramePct,
          mode,
        });
      }

      ranked.sort((a, b) => {
        if (b.totalMs !== a.totalMs) {
          return b.totalMs - a.totalMs;
        }
        if (b.avgMs !== a.avgMs) {
          return b.avgMs - a.avgMs;
        }
        if (b.maxMs !== a.maxMs) {
          return b.maxMs - a.maxMs;
        }
        if (a.label < b.label) {
          return -1;
        }
        if (a.label > b.label) {
          return 1;
        }
        return 0;
      });

      if (ranked.length > maxResults) {
        ranked.length = maxResults;
      }
      return ranked;
    }

    /**
     * Clears rolling per-pass histories, percentile samples, and budget spike counters
     * while keeping GPU resources intact for subsequent frames.
     */
    reset() {
      this._resetGeneration += 1;
      this._resolvedFrames = 0;
      this._skippedFrames = 0;
      this._droppedPasses = 0;
      this._droppedQueryFrames = 0;
      this._mapErrors = 0;
      this._passesOverBudget = 0;
      this._lastFrameGpuMs = 0;
      this._lastFrameRenderMs = 0;
      this._lastFrameComputeMs = 0;
      this._lastFrameOtherMs = 0;
      this._lastFramePassCount = 0;
      this._renderPassCount = 0;
      this._computePassCount = 0;
      this._otherPassCount = 0;
      this._passHistories.clear();
      this._passDurations.fill(0);
      this._passDurationWriteIdx = 0;
      this._passDurationWindowCount = 0;
      this._sortScratch.fill(0);
      this._activeQueryRingSlice = 0;
      this._queryRingSliceUsed.fill(0);
      this._querySlotRecycles = 0;
      this._querySlotWraps = 0;
      this._framePassCount = 0;
      this._framePassNames.fill(undefined);
      this._framePassKinds.fill(PASS_KIND_UNKNOWN);
      this._framePassHashes.fill(0n);
      this._gpuTimeTable.reset({ preserveSupport: true });
      this._queryCursor = 0;
      this._activePassDepth = 0;
      this._activePassLabels.fill(undefined);
      this._activePassKinds.fill(PASS_KIND_UNKNOWN);
      this._frameTotals.fill(0);
      this._frameTotalSeqs.fill(0);
      this._renderFrameTotals.fill(0);
      this._computeFrameTotals.fill(0);
      this._otherFrameTotals.fill(0);
      this._frameTotalWriteIdx = 0;
      this._frameTotalWindowCount = 0;
      this._frameSeq = 0;
      this._cpuFrameOpen = false;
      this._cpuFrameSeq = 0;
      this._cpuFrameTotalMs = 0;
      this._cpuFrameRenderMs = 0;
      this._cpuFrameComputeMs = 0;
      this._cpuFrameOtherMs = 0;
      this._cpuFramePassCount = 0;

      for (let i = 0; i < this._pendingCount; i += 1) {
        const idx = (this._pendingHead + i) % this._readbackSlotCount;
        const slot = this._pendingRing[idx];
        this._pendingRing[idx] = null;
        if (slot && slot.state === 'pending') {
          slot.state = 'idle';
          slot.passCount = 0;
          slot.passNames.fill(undefined);
          slot.passKinds.fill(PASS_KIND_UNKNOWN);
          slot.queryCount = 0;
          slot.copyBytes = 0;
          slot.generation = this._resetGeneration;
        }
      }
      this._pendingHead = 0;
      this._pendingTail = 0;
      this._pendingCount = 0;
    }

    /**
     * Resets profiler counters, error counts, and rolling pass-duration windows
     * while keeping GPU resources intact, returning the fresh `getStats()` snapshot.
     */
    resetStats() {
      this.reset();
      return this.getStats();
    }

    /**
     * Returns sanitized rolling 30 FPS per-pass GPU timing histogram statistics.
     *
     * - When called with no argument (`label === undefined`), returns a defensive map of
     *   all recorded passes keyed by pass label.
     * - When called with a specific pass label, returns that pass's sanitized histogram
     *   stats object or `null` if no samples have been recorded for it.
     *
     * @param {string} [label]
     * @returns {Record<string, {
     *   count: number,
     *   lastMs: number,
     *   avgMs: number,
     *   p50Ms: number,
     *   p95Ms: number,
     *   maxMs: number,
     *   passesOverBudget: number
     * }> | {
     *   count: number,
     *   lastMs: number,
     *   avgMs: number,
     *   p50Ms: number,
     *   p95Ms: number,
     *   maxMs: number,
     *   passesOverBudget: number
     * } | null}
     */
    getPassStats(label) {
      if (label !== undefined) {
        if (typeof label !== 'string' || label.length === 0) {
          return null;
        }
        const entry = this._passHistories.get(label);
        if (!entry || entry.windowCount === 0) {
          return null;
        }
        const hist = this._computeWindowStats(entry.samples, entry.windowCount);
        return {
          count: entry.count,
          lastMs: entry.lastMs,
          avgMs: hist.avgMs,
          p50Ms: hist.p50Ms,
          p95Ms: hist.p95Ms,
          maxMs: hist.maxMs,
          passesOverBudget: entry.passesOverBudget,
        };
      }

      const passStats = {};
      for (const [name, entry] of this._passHistories.entries()) {
        if (entry.windowCount === 0) {
          continue;
        }
        const hist = this._computeWindowStats(entry.samples, entry.windowCount);
        passStats[name] = {
          count: entry.count,
          lastMs: entry.lastMs,
          avgMs: hist.avgMs,
          p50Ms: hist.p50Ms,
          p95Ms: hist.p95Ms,
          maxMs: hist.maxMs,
          passesOverBudget: entry.passesOverBudget,
        };
      }
      return passStats;
    }

    /**
     * Return the recorded pass category/kind (`'render'`, `'compute'`, or `'unknown'`) for a
     * pass label, or `null` if no samples have been recorded for it.
     *
     * @param {string} label
     * @returns {'render' | 'compute' | 'unknown' | null}
     */
    getPassKind(label) {
      if (typeof label !== 'string' || label.length === 0) {
        return null;
      }
      const entry = this._passHistories.get(label);
      if (!entry || entry.windowCount === 0) {
        return null;
      }
      return passKindCodeToString(entry.kind);
    }

    /**
     * Return per-category (`render`, `compute`, `other`) rolling frame timing summaries.
     * Pass an optional reusable `out` object for zero-allocation reads.
     *
     * @param {object} [out]
     */
    getCategoryStats(out) {
      const target = out && typeof out === 'object' ? out : {};
      const renderTarget =
        target.render && typeof target.render === 'object' ? target.render : (target.render = {});
      const computeTarget =
        target.compute && typeof target.compute === 'object'
          ? target.compute
          : (target.compute = {});
      const otherTarget =
        target.other && typeof target.other === 'object' ? target.other : (target.other = {});

      const windowCount = this._frameTotalWindowCount;
      const scratch = this._windowStatsScratch;

      this._fillWindowStats(this._renderFrameTotals, windowCount, scratch);
      renderTarget.count = this._renderPassCount;
      renderTarget.lastFrameMs = this._lastFrameRenderMs;
      renderTarget.avgFrameMs = scratch.avgMs;
      renderTarget.p50FrameMs = scratch.p50Ms;
      renderTarget.p95FrameMs = scratch.p95Ms;
      renderTarget.maxFrameMs = scratch.maxMs;

      this._fillWindowStats(this._computeFrameTotals, windowCount, scratch);
      computeTarget.count = this._computePassCount;
      computeTarget.lastFrameMs = this._lastFrameComputeMs;
      computeTarget.avgFrameMs = scratch.avgMs;
      computeTarget.p50FrameMs = scratch.p50Ms;
      computeTarget.p95FrameMs = scratch.p95Ms;
      computeTarget.maxFrameMs = scratch.maxMs;

      this._fillWindowStats(this._otherFrameTotals, windowCount, scratch);
      otherTarget.count = this._otherPassCount;
      otherTarget.lastFrameMs = this._lastFrameOtherMs;
      otherTarget.avgFrameMs = scratch.avgMs;
      otherTarget.p50FrameMs = scratch.p50Ms;
      otherTarget.p95FrameMs = scratch.p95Ms;
      otherTarget.maxFrameMs = scratch.maxMs;

      return target;
    }

    /**
     * Return a compact zero-allocation frame timing summary including render/compute/other
     * pass category breakdown, rolling pass percentiles, and readback health counters.
     * Pass an optional reusable `out` object to populate it in-place with zero heap allocations.
     *
     * @param {object} [out]
     */
    getFrameSummary(out) {
      const target = out && typeof out === 'object' ? out : {};
      const rolling = this._fillWindowStats(
        this._passDurations,
        this._passDurationWindowCount,
        this._windowStatsScratch
      );

      target.supported = this._supported;
      target.mode = this._supported ? 'gpu' : 'cpu';
      target.deviceLost = this._deviceLost;
      target.resolvedFrames = this._resolvedFrames;
      target.skippedFrames = this._skippedFrames;
      target.droppedPasses = this._droppedPasses;
      target.droppedQueryFrames = this._droppedQueryFrames;
      target.mapErrors = this._mapErrors;
      target.pendingReadbacks = this._pendingCount;
      target.lastFrameGpuMs = this._lastFrameGpuMs;
      target.lastFrameRenderMs = this._lastFrameRenderMs;
      target.lastFrameComputeMs = this._lastFrameComputeMs;
      target.lastFrameOtherMs = this._lastFrameOtherMs;
      target.lastFramePassCount = this._lastFramePassCount;
      target.p50PassMs = rolling.p50Ms;
      target.p95PassMs = rolling.p95Ms;
      target.maxPassMs = rolling.maxMs;
      target.passBudgetMs = this._passBudgetMs;
      target.passesOverBudget = this._passesOverBudget;
      return target;
    }

    /**
     * Returns a snapshot of overall profiler counters and per-pass rolling stats.
     * Pass an optional reusable `out` object for zero-allocation steady-state reads.
     *
     * @param {object} [out]
     * @returns {{
     *   supported: boolean,
     *   resolvedFrames: number,
     *   skippedFrames: number,
     *   droppedPasses: number,
     *   droppedQueryFrames: number,
     *   mapErrors: number,
     *   lastFrameGpuMs: number,
     *   p50PassMs: number,
     *   p95PassMs: number,
     *   maxPassMs: number,
     *   passBudgetMs: number,
     *   passesOverBudget: number,
     *   querySlotRecycles: number,
     *   querySlotWraps: number,
     *   passStats: Record<string, {
     *     count: number,
     *     lastMs: number,
     *     avgMs: number,
     *     p50Ms: number,
     *     p95Ms: number,
     *     maxMs: number,
     *     passesOverBudget: number
     *   }>
     * }}
     */
    getStats(out) {
      const scratch = this._windowStatsScratch;
      const target = out && typeof out === 'object' ? out : {};
      let passStats;

      if (out && typeof out === 'object') {
        passStats =
          target.passStats && typeof target.passStats === 'object'
            ? target.passStats
            : (target.passStats = {});
        // Remove stale keys if reset() cleared pass histories.
        for (const existingKey in passStats) {
          if (
            Object.prototype.hasOwnProperty.call(passStats, existingKey) &&
            !this._passHistories.has(existingKey)
          ) {
            delete passStats[existingKey];
          }
        }
        for (const [name, entry] of this._passHistories.entries()) {
          if (entry.windowCount === 0) {
            continue;
          }
          this._fillWindowStats(entry.samples, entry.windowCount, scratch);
          let slot = passStats[name];
          if (!slot || typeof slot !== 'object') {
            slot = {};
            passStats[name] = slot;
          }
          slot.count = entry.count;
          slot.lastMs = entry.lastMs;
          slot.avgMs = scratch.avgMs;
          slot.p50Ms = scratch.p50Ms;
          slot.p95Ms = scratch.p95Ms;
          slot.maxMs = scratch.maxMs;
          slot.passesOverBudget = entry.passesOverBudget;
        }
      } else {
        passStats = this.getPassStats();
      }

      this._fillWindowStats(this._passDurations, this._passDurationWindowCount, scratch);

      target.supported = this._supported;
      target.resolvedFrames = this._resolvedFrames;
      target.skippedFrames = this._skippedFrames;
      target.droppedPasses = this._droppedPasses;
      target.droppedQueryFrames = this._droppedQueryFrames;
      target.mapErrors = this._mapErrors;
      target.lastFrameGpuMs = this._lastFrameGpuMs;
      target.p50PassMs = scratch.p50Ms;
      target.p95PassMs = scratch.p95Ms;
      target.maxPassMs = scratch.maxMs;
      target.passBudgetMs = this._passBudgetMs;
      target.passesOverBudget = this._passesOverBudget;
      target.querySlotRecycles = this._querySlotRecycles;
      target.querySlotWraps = this._querySlotWraps;
      target.passStats = passStats;
      return target;
    }

    /**
     * Idempotently destroys the GPUQuerySet and all resolve/readback GPUBuffers.
     */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      if (this._querySet && typeof this._querySet.destroy === 'function') {
        try {
          this._querySet.destroy();
        } catch (_) {
          // ignore cleanup errors
        }
      }
      this._querySet = null;

      if (this._resolveBuffer && typeof this._resolveBuffer.destroy === 'function') {
        try {
          this._resolveBuffer.destroy();
        } catch (_) {
          // ignore cleanup errors
        }
      }
      this._resolveBuffer = null;

      for (const slot of this._readbackSlots) {
        if (slot.buffer && typeof slot.buffer.destroy === 'function') {
          try {
            slot.buffer.destroy();
          } catch (_) {
            // ignore cleanup errors
          }
        }
        slot.buffer = null;
        slot.state = 'destroyed';
        slot.passCount = 0;
        slot.passNames.fill(undefined);
        slot.passHashes.fill(0n);
        slot.queryCount = 0;
        slot.copyBytes = 0;
      }
      this._readbackSlots = [];
      this._timestampWriteDescriptors = [];
      this._pendingRing.fill(null);
      this._pendingHead = 0;
      this._pendingTail = 0;
      this._pendingCount = 0;
      this._completeListeners.clear();
      this.reset();
      this._device = null;
    }
  }

  const exportedApi = {
    DEFAULT_MAX_PASSES_PER_FRAME,
    DEFAULT_READBACK_SLOTS,
    DEFAULT_QUERY_RING_SLOTS,
    DEFAULT_HISTORY_SIZE,
    DEFAULT_FRAME_BUDGET_MS,
    DEFAULT_PASS_BUDGET_MS,
    DEFAULT_30FPS_BUDGET_MS,
    TIMESTAMP_BYTE_SIZE,
    PASS_KINDS,
    GPU_TIME_MAX_KEYS,
    GPU_TIME_SUPPORT,
    BUFFER_USAGE,
    MAP_MODE,
    formatHashHex16,
    normalizeShaderHash,
    computePercentiles,
    GpuTimeTable,
    GpuPassTimer,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6GpuTimer = exportedApi;
  } else if (root) {
    root.PlayGta6GpuTimer = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
