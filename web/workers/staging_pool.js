'use strict';

// Frame-recycled WebGPU staging buffer pool and upload ring.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6StagingPool)
// and Node.js (module.exports). Aligns cleanly with web/workers/wgpu_worker.js and
// web/workers/texture_formats.js (4-byte copyBufferToBuffer & 256-byte copyBufferToTexture).

(function (root) {
  const DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024; // 4 MiB
  const DEFAULT_MAX_WRITE_CHUNK_BYTES = 4 * 1024 * 1024; // 4 MiB cap per queue.writeBuffer / chunked buffer upload
  const DEFAULT_MAX_SCRATCH_BYTES = 16 * 1024 * 1024; // 16 MiB cap for reusable CPU upload scratch
  const DEFAULT_MAX_POOL_BYTES = 64 * 1024 * 1024; // 64 MiB
  const DEFAULT_FRAMES_IN_FLIGHT = 3;
  const DEFAULT_MAX_FRAMES_IN_FLIGHT = DEFAULT_FRAMES_IN_FLIGHT;
  const DEFAULT_MIN_WARM_CHUNKS = 1;
  const DEFAULT_IDLE_EVICT_FRAMES = 0;
  const DEFAULT_IDLE_SHRINK_FRAMES = 120; // ~4 seconds at 30 FPS for transient oversized slabs
  const DEFAULT_LARGE_SLAB_THRESHOLD = 2 * 1024 * 1024; // 2 MiB
  const DEFAULT_MAX_LARGE_SLABS = 12;
  const DEFAULT_POW2_MIN_SLAB_BYTES = 1 * 1024 * 1024; // 1 MiB
  const DEFAULT_POW2_MAX_BUCKET_BYTES = 16 * 1024 * 1024; // 16 MiB
  const DEFAULT_LARGE_SLAB_BUCKETS = Object.freeze([
    2 * 1024 * 1024, // 2 MiB
    3 * 1024 * 1024, // 3 MiB (fits 4K 1920x1080 256B-aligned chroma = 2.11 MiB)
    4 * 1024 * 1024, // 4 MiB (fits 4K NV12 UV = 3.96 MiB)
    6 * 1024 * 1024, // 6 MiB
    8 * 1024 * 1024, // 8 MiB (fits 4K 3840x2160 Y plane = 7.91 MiB)
    12 * 1024 * 1024, // 12 MiB (fits 8.3–12 MiB video / texture slabs)
    16 * 1024 * 1024, // 16 MiB
  ]);
  const POW2_LARGE_SLAB_BUCKETS = Object.freeze([
    1 * 1024 * 1024, // 1 MiB
    2 * 1024 * 1024, // 2 MiB (fits 1920x1080 = 2.07 MB raw chroma)
    4 * 1024 * 1024, // 4 MiB (fits 2.07 MiB / 2048x1080 256B-aligned chroma & 4K NV12 UV)
    8 * 1024 * 1024, // 8 MiB (fits 3840x2160 = 7.91 MiB / 8.29 MB 4K Y plane)
    16 * 1024 * 1024, // 16 MiB (fits 8.3 MiB / 4096x2160 4K Y & 16 MiB RGBA slabs)
  ]);
  const BUCKET_STRATEGY = Object.freeze({
    DEFAULT: 'default',
    POW2: 'pow2',
  });
  const WEBGPU_BUFFER_COPY_ALIGNMENT = 4;
  const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;

  // WebGPU GPUBufferUsage bitmask values with headless/Node fallbacks.
  const BUFFER_USAGE = Object.freeze({
    MAP_WRITE:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.MAP_WRITE === 'number'
        ? GPUBufferUsage.MAP_WRITE
        : 0x0002,
    COPY_SRC:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_SRC === 'number'
        ? GPUBufferUsage.COPY_SRC
        : 0x0004,
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
  });

  // WebGPU GPUMapMode bitmask values with headless/Node fallbacks.
  const MAP_MODE = Object.freeze({
    WRITE:
      typeof GPUMapMode !== 'undefined' && typeof GPUMapMode.WRITE === 'number'
        ? GPUMapMode.WRITE
        : 0x0002,
  });

  const STAGING_MODE = Object.freeze({
    WRITE_BUFFER: 'writeBuffer',
    MAPPED: 'mapped',
  });

  function assertPositiveInteger(value, name) {
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got ${value}`);
    }
  }

  function assertNonNegativeInteger(value, name) {
    if (!Number.isInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative integer, got ${value}`);
    }
  }

  function sanitizeNonNegativeCounter(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return Math.floor(value);
  }

  function clampUnitRate(numerator, denominator) {
    if (
      typeof numerator !== 'number' ||
      typeof denominator !== 'number' ||
      !Number.isFinite(numerator) ||
      !Number.isFinite(denominator) ||
      denominator <= 0 ||
      numerator <= 0
    ) {
      return 0;
    }
    const rate = numerator / denominator;
    if (!Number.isFinite(rate) || rate <= 0) {
      return 0;
    }
    return rate >= 1 ? 1 : rate;
  }

  /**
   * Align a non-negative integer up to the nearest multiple of `alignment`.
   * Safe across the full IEEE-754 integer range (no 32-bit bitwise truncation).
   *
   * @param {number} value
   * @param {number} alignment
   * @returns {number}
   */
  function alignUp(value, alignment) {
    assertNonNegativeInteger(value, 'value');
    assertPositiveInteger(alignment, 'alignment');
    const remainder = value % alignment;
    return remainder === 0 ? value : value + (alignment - remainder);
  }

  /**
   * Align a requested staging buffer size to a WebGPU 4-byte multiple with a
   * minimum size of 4 bytes (`Math.max(4, Math.ceil(size / 4) * 4)`), safe across
   * the full IEEE-754 integer range.
   *
   * @param {number} size
   * @returns {number}
   */
  function alignBufferSize(size) {
    assertNonNegativeInteger(size, 'size');
    if (size === 0) {
      return WEBGPU_BUFFER_COPY_ALIGNMENT;
    }
    return alignUp(size, WEBGPU_BUFFER_COPY_ALIGNMENT);
  }

  function normalizeLargeSlabBuckets(buckets) {
    if (!Array.isArray(buckets)) {
      throw new TypeError('largeSlabBuckets must be an array of positive integers');
    }
    const seen = new Set();
    const normalized = [];
    for (let i = 0; i < buckets.length; i += 1) {
      assertPositiveInteger(buckets[i], `largeSlabBuckets[${i}]`);
      const aligned = alignBufferSize(buckets[i]);
      if (!seen.has(aligned)) {
        seen.add(aligned);
        normalized.push(aligned);
      }
    }
    normalized.sort((a, b) => a - b);
    return Object.freeze(normalized);
  }

  /**
   * Compute the smallest power of two `>= value`, safe across the full IEEE-754
   * safe-integer range (no 32-bit bitwise truncation).
   *
   * @param {number} value
   * @returns {number}
   */
  function nextPowerOfTwo(value) {
    assertPositiveInteger(value, 'value');
    if (value <= 1) {
      return 1;
    }
    let pow2 = 1;
    while (pow2 < value) {
      pow2 *= 2;
    }
    return pow2;
  }

  /**
   * Build a sorted, frozen power-of-two bucket ladder from `minBytes` through
   * `maxBytes` (default 1 MiB .. 16 MiB for 4K Bink2 YUV / RGBA planes).
   *
   * @param {number} [minBytes=DEFAULT_POW2_MIN_SLAB_BYTES]
   * @param {number} [maxBytes=DEFAULT_POW2_MAX_BUCKET_BYTES]
   * @returns {readonly number[]}
   */
  function buildPowerOfTwoBuckets(
    minBytes = DEFAULT_POW2_MIN_SLAB_BYTES,
    maxBytes = DEFAULT_POW2_MAX_BUCKET_BYTES
  ) {
    assertPositiveInteger(minBytes, 'minBytes');
    assertPositiveInteger(maxBytes, 'maxBytes');
    if (minBytes > maxBytes) {
      throw new RangeError(`minBytes (${minBytes}) must be <= maxBytes (${maxBytes})`);
    }
    const start = Math.max(WEBGPU_BUFFER_COPY_ALIGNMENT, nextPowerOfTwo(minBytes));
    const buckets = [];
    for (let size = start; size <= maxBytes; size *= 2) {
      buckets.push(size);
    }
    if (buckets.length === 0) {
      buckets.push(alignBufferSize(maxBytes));
    }
    return Object.freeze(buckets);
  }

  /**
   * Select a power-of-two bucket size `>= alignBufferSize(byteSize)` clamped between
   * `minBucketBytes` and `maxBucketBytes` (default 16 MiB). Requests above
   * `maxBucketBytes` fall back to exact 4-byte-aligned size to avoid doubling huge slabs.
   *
   * @param {number} byteSize
   * @param {number} [minBucketBytes=WEBGPU_BUFFER_COPY_ALIGNMENT]
   * @param {number} [maxBucketBytes=DEFAULT_POW2_MAX_BUCKET_BYTES]
   * @returns {number}
   */
  function selectPowerOfTwoBucket(
    byteSize,
    minBucketBytes = WEBGPU_BUFFER_COPY_ALIGNMENT,
    maxBucketBytes = DEFAULT_POW2_MAX_BUCKET_BYTES
  ) {
    assertPositiveInteger(byteSize, 'byteSize');
    assertPositiveInteger(minBucketBytes, 'minBucketBytes');
    assertPositiveInteger(maxBucketBytes, 'maxBucketBytes');
    if (minBucketBytes > maxBucketBytes) {
      throw new RangeError(
        `minBucketBytes (${minBucketBytes}) must be <= maxBucketBytes (${maxBucketBytes})`
      );
    }
    const aligned = alignBufferSize(byteSize);
    if (aligned > maxBucketBytes) {
      return aligned;
    }
    const floor = Math.max(WEBGPU_BUFFER_COPY_ALIGNMENT, nextPowerOfTwo(minBucketBytes));
    const pow2 = Math.max(floor, nextPowerOfTwo(aligned));
    return pow2 <= maxBucketBytes ? pow2 : aligned;
  }

  /**
   * Deterministic LRU eviction comparator for idle buffers/slabs:
   *  1. Oldest `lastUsedFrame` first (LRU across frames),
   *  2. Largest `size` first on same-frame tie (reclaims oversized VRAM fastest),
   *  3. Oldest monotonic `lastUsedSeq` first on same-frame, same-size tie (LRU within bucket),
   *  4. Lowest `id` as final deterministic tie-breaker.
   */
  function compareEvictionLru(a, b) {
    if (a.lastUsedFrame !== b.lastUsedFrame) {
      return a.lastUsedFrame - b.lastUsedFrame;
    }
    if (a.size !== b.size) {
      return b.size - a.size;
    }
    if (a.lastUsedSeq !== b.lastUsedSeq) {
      return a.lastUsedSeq - b.lastUsedSeq;
    }
    return a.id - b.id;
  }

  /**
   * Select the canonical large-slab bucket byte size for a requested allocation.
   * Returns the smallest bucket `>= alignBufferSize(byteSize)`, or `alignBufferSize(byteSize)`
   * if the request exceeds the largest configured bucket.
   *
   * @param {number} byteSize
   * @param {readonly number[]} [buckets=DEFAULT_LARGE_SLAB_BUCKETS]
   * @returns {number}
   */
  function selectLargeSlabBucket(byteSize, buckets = DEFAULT_LARGE_SLAB_BUCKETS) {
    assertPositiveInteger(byteSize, 'byteSize');
    const aligned = alignBufferSize(byteSize);
    for (let i = 0; i < buckets.length; i += 1) {
      if (buckets[i] >= aligned) {
        return buckets[i];
      }
    }
    return aligned;
  }

  function defaultNowMs() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    if (typeof Date !== 'undefined' && typeof Date.now === 'function') {
      return Date.now();
    }
    return 0;
  }

  function lcmWithBufferAlignment(alignment) {
    if (alignment % WEBGPU_BUFFER_COPY_ALIGNMENT === 0) {
      return alignment;
    }
    if (alignment % 2 === 0) {
      return alignment * 2;
    }
    return alignment * WEBGPU_BUFFER_COPY_ALIGNMENT;
  }

  function isSharedBuffer(buffer) {
    return typeof SharedArrayBuffer !== 'undefined' && buffer instanceof SharedArrayBuffer;
  }

  function toUint8Array(input, label) {
    if (input instanceof Uint8Array) {
      return input;
    }
    if (ArrayBuffer.isView(input)) {
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    if (input instanceof ArrayBuffer) {
      return new Uint8Array(input);
    }
    if (isSharedBuffer(input)) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      `${label} expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`
    );
  }

  /**
   * Sanitize a buffer upload payload for `GPUQueue.writeBuffer()`:
   *  - Copies `SharedArrayBuffer`-backed views into a regular `ArrayBuffer`-backed `Uint8Array`.
   *  - Zero-pads non-4-byte lengths up to a 4-byte multiple (`alignBufferSize`).
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @returns {{ bytes: Uint8Array, originalByteLength: number, alignedByteLength: number, padded: boolean, copiedFromSab: boolean }}
   */
  const EMPTY_UINT8_ARRAY = new Uint8Array(0);

  /**
   * O(1) amortized circular ring queue with O(1) arbitrary removal via per-chunk
   * slot tracking (`chunk._queue` / `chunk._queueSlot`). Preserves strict FIFO
   * insertion order across `push()`, `shift()`, and iteration without
   * `Array.prototype.shift()`, `Array.prototype.splice()`, or per-frame array churn.
   */
  class ChunkRingQueue {
    constructor(initialCapacity = 16) {
      let cap = 16;
      while (cap < initialCapacity) {
        cap *= 2;
      }
      this._buf = new Array(cap);
      this._mask = cap - 1;
      this._head = 0;
      this._tail = 0;
      this._span = 0;
      this._size = 0;
    }

    get size() {
      return this._size;
    }

    push(chunk) {
      if (this._span >= this._buf.length) {
        if (this._size * 2 <= this._buf.length) {
          this._repack(this._buf.length);
        } else {
          this._repack(this._buf.length * 2);
        }
      }
      const slot = this._tail;
      this._buf[slot] = chunk;
      chunk._queue = this;
      chunk._queueSlot = slot;
      this._tail = (slot + 1) & this._mask;
      this._span += 1;
      this._size += 1;
    }

    shift() {
      const buf = this._buf;
      const mask = this._mask;
      while (this._span > 0) {
        const slot = this._head;
        const chunk = buf[slot];
        buf[slot] = undefined;
        this._head = (slot + 1) & mask;
        this._span -= 1;
        if (chunk !== undefined) {
          chunk._queue = null;
          chunk._queueSlot = -1;
          this._size -= 1;
          if (this._size === 0) {
            this._head = 0;
            this._tail = 0;
            this._span = 0;
          }
          return chunk;
        }
      }
      this._head = 0;
      this._tail = 0;
      this._span = 0;
      this._size = 0;
      return null;
    }

    pop() {
      const buf = this._buf;
      const mask = this._mask;
      while (this._span > 0) {
        const slot = (this._tail - 1) & mask;
        const chunk = buf[slot];
        buf[slot] = undefined;
        this._tail = slot;
        this._span -= 1;
        if (chunk !== undefined) {
          chunk._queue = null;
          chunk._queueSlot = -1;
          this._size -= 1;
          if (this._size === 0) {
            this._head = 0;
            this._tail = 0;
            this._span = 0;
          }
          return chunk;
        }
      }
      this._head = 0;
      this._tail = 0;
      this._span = 0;
      this._size = 0;
      return null;
    }

    contains(chunk) {
      return chunk !== null && chunk !== undefined && chunk._queue === this;
    }

    remove(chunk) {
      if (!chunk || chunk._queue !== this) {
        return false;
      }
      const slot = chunk._queueSlot;
      const buf = this._buf;
      buf[slot] = undefined;
      chunk._queue = null;
      chunk._queueSlot = -1;
      this._size -= 1;

      if (this._size === 0) {
        this._head = 0;
        this._tail = 0;
        this._span = 0;
        return true;
      }

      const mask = this._mask;
      while (this._span > 0 && buf[this._head] === undefined) {
        this._head = (this._head + 1) & mask;
        this._span -= 1;
      }
      while (this._span > 0 && buf[(this._tail - 1) & mask] === undefined) {
        this._tail = (this._tail - 1) & mask;
        this._span -= 1;
      }
      return true;
    }

    clear() {
      const buf = this._buf;
      const mask = this._mask;
      let idx = this._head;
      for (let i = 0; i < this._span; i += 1) {
        const chunk = buf[idx];
        if (chunk !== undefined) {
          chunk._queue = null;
          chunk._queueSlot = -1;
          buf[idx] = undefined;
        }
        idx = (idx + 1) & mask;
      }
      this._head = 0;
      this._tail = 0;
      this._span = 0;
      this._size = 0;
    }

    _repack(newCapacity) {
      const oldBuf = this._buf;
      const oldMask = this._mask;
      const newBuf = new Array(newCapacity);
      let idx = this._head;
      let write = 0;
      for (let i = 0; i < this._span; i += 1) {
        const chunk = oldBuf[idx];
        if (chunk !== undefined) {
          newBuf[write] = chunk;
          chunk._queueSlot = write;
          write += 1;
        }
        idx = (idx + 1) & oldMask;
      }
      this._buf = newBuf;
      this._mask = newCapacity - 1;
      this._head = 0;
      this._tail = write & this._mask;
      this._span = write;
      this._size = write;
    }
  }

  function copyTextureRowsPadded(
    src,
    dst,
    rowStart,
    rowCount,
    unalignedBytesPerRow,
    alignedBytesPerRow,
    zeroPad
  ) {
    if (unalignedBytesPerRow <= 64) {
      for (let r = 0; r < rowCount; r += 1) {
        const srcRowStart = (rowStart + r) * unalignedBytesPerRow;
        const dstRowStart = r * alignedBytesPerRow;
        for (let c = 0; c < unalignedBytesPerRow; c += 1) {
          dst[dstRowStart + c] = src[srcRowStart + c];
        }
        if (zeroPad) {
          dst.fill(0, dstRowStart + unalignedBytesPerRow, dstRowStart + alignedBytesPerRow);
        }
      }
      return;
    }
    for (let r = 0; r < rowCount; r += 1) {
      const srcRowStart = (rowStart + r) * unalignedBytesPerRow;
      const dstRowStart = r * alignedBytesPerRow;
      dst.set(src.subarray(srcRowStart, srcRowStart + unalignedBytesPerRow), dstRowStart);
      if (zeroPad) {
        dst.fill(0, dstRowStart + unalignedBytesPerRow, dstRowStart + alignedBytesPerRow);
      }
    }
  }

  function sanitizeWriteBufferPayload(srcBytes) {
    const src = toUint8Array(srcBytes, 'sanitizeWriteBufferPayload');
    const originalByteLength = src.byteLength;
    if (originalByteLength === 0) {
      return {
        bytes: EMPTY_UINT8_ARRAY,
        originalByteLength: 0,
        alignedByteLength: 0,
        padded: false,
        copiedFromSab: false,
      };
    }

    const alignedByteLength = alignBufferSize(originalByteLength);
    const padded = alignedByteLength !== originalByteLength;
    const copiedFromSab = isSharedBuffer(src.buffer);

    if (padded || copiedFromSab) {
      const copy = new Uint8Array(alignedByteLength);
      copy.set(src, 0);
      return {
        bytes: copy,
        originalByteLength,
        alignedByteLength,
        padded,
        copiedFromSab,
      };
    }

    return {
      bytes: src,
      originalByteLength,
      alignedByteLength,
      padded: false,
      copiedFromSab: false,
    };
  }

  function fillNormalizedOriginInto(target, origin) {
    if (origin === undefined || origin === null) {
      target.x = 0;
      target.y = 0;
      target.z = 0;
      return target;
    }
    if (Array.isArray(origin)) {
      const x = origin[0] ?? 0;
      const y = origin[1] ?? 0;
      const z = origin[2] ?? 0;
      assertNonNegativeInteger(x, 'origin[0]');
      assertNonNegativeInteger(y, 'origin[1]');
      assertNonNegativeInteger(z, 'origin[2]');
      target.x = x;
      target.y = y;
      target.z = z;
      return target;
    }
    if (typeof origin === 'object') {
      const x = origin.x ?? 0;
      const y = origin.y ?? 0;
      const z = origin.z ?? 0;
      assertNonNegativeInteger(x, 'origin.x');
      assertNonNegativeInteger(y, 'origin.y');
      assertNonNegativeInteger(z, 'origin.z');
      target.x = x;
      target.y = y;
      target.z = z;
      return target;
    }
    throw new TypeError('origin must be an object { x, y, z } or [x, y, z] array');
  }

  /**
   * Sub-allocating WebGPU staging buffer pool with frame-epoch recycling.
   *
   * Supports both:
   *  - `'writeBuffer'` mode (default): chunks use `COPY_SRC | COPY_DST`, populated via
   *    `device.queue.writeBuffer` and copied via `encoder.copyBufferToBuffer` /
   *    `encoder.copyBufferToTexture`. Recycles synchronously by frame epoch or GPU fence.
   *  - `'mapped'` mode: chunks use `MAP_WRITE | COPY_SRC` (`mappedAtCreation: true`),
   *    exposing a direct `Uint8Array` `view` for zero-extra-copy staging, unmapped on
   *    `prepareSubmit()` / `endFrame()` and remapped via `mapAsync(GPUMapMode.WRITE)`.
   */
  class GpuStagingPool {
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('GpuStagingPool constructor expects an options object');
      }
      const {
        device,
        chunkSize,
        maxPoolBytes,
        maxPooledBytes,
        framesInFlight,
        maxFramesInFlight,
        mode = STAGING_MODE.WRITE_BUFFER,
        minWarmChunks = DEFAULT_MIN_WARM_CHUNKS,
        idleEvictFrames = DEFAULT_IDLE_EVICT_FRAMES,
        idleShrinkFrames = DEFAULT_IDLE_SHRINK_FRAMES,
        maxLargeSlabs = DEFAULT_MAX_LARGE_SLABS,
        maxSlabsPerBucket,
        maxPerBucket,
        bucketStrategy,
        powerOfTwoBuckets,
        maxBucketBytes = DEFAULT_POW2_MAX_BUCKET_BYTES,
        largeSlabMinBytes,
        largeSlabBuckets,
        maxWriteChunkBytes = DEFAULT_MAX_WRITE_CHUNK_BYTES,
        maxScratchBytes = DEFAULT_MAX_SCRATCH_BYTES,
        now,
      } = options;

      if (!device || typeof device.createBuffer !== 'function') {
        throw new TypeError('GpuStagingPool requires a WebGPU device with createBuffer()');
      }

      if (maxPoolBytes !== undefined) {
        assertPositiveInteger(maxPoolBytes, 'maxPoolBytes');
      }
      if (maxPooledBytes !== undefined) {
        assertPositiveInteger(maxPooledBytes, 'maxPooledBytes');
      }
      if (
        maxPoolBytes !== undefined &&
        maxPooledBytes !== undefined &&
        maxPoolBytes !== maxPooledBytes
      ) {
        throw new RangeError(
          `Conflicting maxPoolBytes (${maxPoolBytes}) and maxPooledBytes (${maxPooledBytes})`
        );
      }
      const resolvedMaxPoolBytes =
        maxPoolBytes !== undefined
          ? maxPoolBytes
          : maxPooledBytes !== undefined
            ? maxPooledBytes
            : DEFAULT_MAX_POOL_BYTES;

      if (framesInFlight !== undefined) {
        assertPositiveInteger(framesInFlight, 'framesInFlight');
      }
      if (maxFramesInFlight !== undefined) {
        assertPositiveInteger(maxFramesInFlight, 'maxFramesInFlight');
      }
      if (
        framesInFlight !== undefined &&
        maxFramesInFlight !== undefined &&
        framesInFlight !== maxFramesInFlight
      ) {
        throw new RangeError(
          `Conflicting framesInFlight (${framesInFlight}) and maxFramesInFlight (${maxFramesInFlight})`
        );
      }
      const resolvedFramesInFlight =
        maxFramesInFlight !== undefined
          ? maxFramesInFlight
          : framesInFlight !== undefined
            ? framesInFlight
            : DEFAULT_FRAMES_IN_FLIGHT;

      assertNonNegativeInteger(minWarmChunks, 'minWarmChunks');
      assertNonNegativeInteger(idleEvictFrames, 'idleEvictFrames');
      assertNonNegativeInteger(idleShrinkFrames, 'idleShrinkFrames');
      assertNonNegativeInteger(maxLargeSlabs, 'maxLargeSlabs');
      assertPositiveInteger(maxBucketBytes, 'maxBucketBytes');
      assertPositiveInteger(maxWriteChunkBytes, 'maxWriteChunkBytes');
      assertNonNegativeInteger(maxScratchBytes, 'maxScratchBytes');
      const resolvedMaxWriteChunkBytes = alignBufferSize(maxWriteChunkBytes);

      if (maxSlabsPerBucket !== undefined && maxSlabsPerBucket !== null) {
        assertNonNegativeInteger(maxSlabsPerBucket, 'maxSlabsPerBucket');
      }
      if (maxPerBucket !== undefined && maxPerBucket !== null) {
        assertNonNegativeInteger(maxPerBucket, 'maxPerBucket');
      }
      if (
        maxSlabsPerBucket !== undefined &&
        maxSlabsPerBucket !== null &&
        maxPerBucket !== undefined &&
        maxPerBucket !== null &&
        maxSlabsPerBucket !== maxPerBucket
      ) {
        throw new RangeError(
          `Conflicting maxSlabsPerBucket (${maxSlabsPerBucket}) and maxPerBucket (${maxPerBucket})`
        );
      }
      const resolvedMaxSlabsPerBucket =
        maxSlabsPerBucket !== undefined && maxSlabsPerBucket !== null
          ? maxSlabsPerBucket
          : maxPerBucket !== undefined && maxPerBucket !== null
            ? maxPerBucket
            : null;

      if (powerOfTwoBuckets !== undefined && typeof powerOfTwoBuckets !== 'boolean') {
        throw new TypeError('powerOfTwoBuckets must be a boolean');
      }
      let resolvedBucketStrategy;
      if (bucketStrategy !== undefined) {
        if (
          bucketStrategy !== BUCKET_STRATEGY.DEFAULT &&
          bucketStrategy !== BUCKET_STRATEGY.POW2
        ) {
          throw new RangeError(
            `Unsupported bucketStrategy: ${JSON.stringify(bucketStrategy)} (expected 'default' or 'pow2')`
          );
        }
        if (powerOfTwoBuckets === true && bucketStrategy !== BUCKET_STRATEGY.POW2) {
          throw new RangeError(
            `Conflicting powerOfTwoBuckets=true and bucketStrategy=${JSON.stringify(bucketStrategy)}`
          );
        }
        if (powerOfTwoBuckets === false && bucketStrategy === BUCKET_STRATEGY.POW2) {
          throw new RangeError(
            `Conflicting powerOfTwoBuckets=false and bucketStrategy=${JSON.stringify(bucketStrategy)}`
          );
        }
        resolvedBucketStrategy = bucketStrategy;
      } else {
        resolvedBucketStrategy =
          powerOfTwoBuckets === true ? BUCKET_STRATEGY.POW2 : BUCKET_STRATEGY.DEFAULT;
      }

      if (mode !== STAGING_MODE.WRITE_BUFFER && mode !== STAGING_MODE.MAPPED) {
        throw new RangeError(
          `Unsupported staging mode: ${JSON.stringify(mode)} (expected 'writeBuffer' or 'mapped')`
        );
      }

      if (now !== undefined && typeof now !== 'function') {
        throw new TypeError('now must be a function returning a timestamp in milliseconds');
      }

      let resolvedLargeSlabBuckets;
      let resolvedLargeSlabMinBytes;
      if (largeSlabBuckets !== undefined) {
        resolvedLargeSlabBuckets = normalizeLargeSlabBuckets(largeSlabBuckets);
        if (largeSlabMinBytes !== undefined) {
          assertPositiveInteger(largeSlabMinBytes, 'largeSlabMinBytes');
          resolvedLargeSlabMinBytes = largeSlabMinBytes;
        } else if (resolvedLargeSlabBuckets.length > 0) {
          resolvedLargeSlabMinBytes = resolvedLargeSlabBuckets[0];
        } else {
          resolvedLargeSlabMinBytes = DEFAULT_LARGE_SLAB_THRESHOLD;
        }
      } else if (resolvedBucketStrategy === BUCKET_STRATEGY.POW2) {
        if (largeSlabMinBytes !== undefined) {
          assertPositiveInteger(largeSlabMinBytes, 'largeSlabMinBytes');
          resolvedLargeSlabMinBytes = largeSlabMinBytes;
        } else {
          resolvedLargeSlabMinBytes = Math.min(DEFAULT_POW2_MIN_SLAB_BYTES, maxBucketBytes);
        }
        resolvedLargeSlabBuckets = buildPowerOfTwoBuckets(
          resolvedLargeSlabMinBytes,
          maxBucketBytes
        );
      } else {
        resolvedLargeSlabBuckets = DEFAULT_LARGE_SLAB_BUCKETS;
        if (largeSlabMinBytes !== undefined) {
          assertPositiveInteger(largeSlabMinBytes, 'largeSlabMinBytes');
          resolvedLargeSlabMinBytes = largeSlabMinBytes;
        } else {
          resolvedLargeSlabMinBytes = DEFAULT_LARGE_SLAB_THRESHOLD;
        }
      }

      let resolvedChunkSize;
      let initialAlignPads = 0;
      if (chunkSize === undefined) {
        const clampedMax = Math.max(
          WEBGPU_BUFFER_COPY_ALIGNMENT,
          Math.floor(resolvedMaxPoolBytes / WEBGPU_BUFFER_COPY_ALIGNMENT) *
            WEBGPU_BUFFER_COPY_ALIGNMENT
        );
        resolvedChunkSize = Math.min(DEFAULT_CHUNK_SIZE, clampedMax);
      } else {
        assertPositiveInteger(chunkSize, 'chunkSize');
        resolvedChunkSize = alignBufferSize(chunkSize);
        if (resolvedChunkSize !== chunkSize) {
          initialAlignPads = 1;
        }
      }

      this.device = device;
      this.chunkSize = resolvedChunkSize;
      this.maxPoolBytes = resolvedMaxPoolBytes;
      this.maxPooledBytes = resolvedMaxPoolBytes;
      this.framesInFlight = resolvedFramesInFlight;
      this.maxFramesInFlight = resolvedFramesInFlight;
      this.mode = mode;
      this.minWarmChunks = minWarmChunks;
      this.idleEvictFrames = idleEvictFrames;
      this.idleShrinkFrames = idleShrinkFrames;
      this.maxLargeSlabs = maxLargeSlabs;
      this.maxSlabsPerBucket = resolvedMaxSlabsPerBucket;
      this.bucketStrategy = resolvedBucketStrategy;
      this.maxBucketBytes = maxBucketBytes;
      this.largeSlabMinBytes = resolvedLargeSlabMinBytes;
      this.largeSlabBuckets = resolvedLargeSlabBuckets;
      this.maxWriteChunkBytes = resolvedMaxWriteChunkBytes;
      this.maxScratchBytes = maxScratchBytes;
      this._now = now || defaultNowMs;

      this._destroyed = false;
      this._currentFrame = -1;
      this._inFrame = false;
      this._frameStartedAtLeastOnce = false;
      this._nextChunkId = 1;
      this._lruSeq = 0;

      this._allChunks = new Set();
      this._chunkByBuffer = new Map();
      this._activeChunks = [];
      this._recycledStdQueue = new ChunkRingQueue(16);
      this._recycledOversizedChunks = [];
      this._largeSlabBuckets = new Map();
      this._idleLargeSlabCount = 0;
      this._recycledBytes = 0;
      this._inFlightQueue = new ChunkRingQueue(16);
      this._gpuDoneInFlightCount = 0;
      this._currentChunk = null;
      this._pendingMapPromises = new Set();
      this._lastAcquireFresh = false;
      this._uploadScratch = null;
      this._reclaimScratch = {
        reclaimedChunks: 0,
        reclaimedBytes: 0,
        shrunkSlabs: 0,
        shrunkBytes: 0,
      };
      this._internalSliceScratch = {
        buffer: null,
        offset: 0,
        size: 0,
        alignedSize: 0,
        isLargeSlab: false,
        slabBucketSize: 0,
        view: undefined,
      };
      this._texCopyOrigin = { x: 0, y: 0, z: 0 };
      this._texCopySource = {
        buffer: null,
        offset: 0,
        bytesPerRow: 0,
        rowsPerImage: 0,
      };
      this._texCopyDest = {
        texture: null,
        mipLevel: 0,
        origin: this._texCopyOrigin,
      };
      this._texCopyDestWithAspect = {
        texture: null,
        mipLevel: 0,
        origin: this._texCopyOrigin,
        aspect: 'all',
      };
      this._texCopySize = {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
      };

      // Pool-wide telemetry.
      this._totalChunksAllocated = 0;
      this._totalBytesAllocated = 0;
      this._allocatedBytes = 0;
      this._peakAllocatedBytes = 0;
      this._peakSingleAllocationBytes = 0;
      this._largeSlabBytes = 0;
      this._reclaimedChunks = 0;
      this._reclaimedBytes = 0;
      this._evictedCount = 0;
      this._evictedBytes = 0;
      this._oversizeAllocations = 0;
      this._bytesUploaded = 0;
      this._totalBytesStaged = 0;
      this._bytesReused = 0;
      this._allocationCount = 0;
      this._bufferReuseHits = 0;
      this._bufferReuseMisses = 0;
      this._largeSlabHits = 0;
      this._largeSlabMisses = 0;
      this._largeSlabEvictions = 0;
      this._reusedAllocations = 0;
      this._freshAllocations = 0;
      this._alignPadCount = initialAlignPads;
      this._sabCopyCount = 0;
      this._doubleReleaseIgnored = 0;
      this._subAllocCoalesced = 0;
      this._idleSlabShrinks = 0;
      this._stallAvoidedCount = 0;
      this._zeroAllocAcquires = 0;
      this._sceneTransitionTrimBytes = 0;
      this._chunkedWrites = 0;
      this._trimmedStagingBytes = 0;
      this._destroyedBuffers = 0;

      // Upload throughput & per-frame telemetry.
      this._framesCompleted = 0;
      this._lastFrameBytesUploaded = 0;
      this._peakFrameBytesUploaded = 0;
      this._firstActivityTimeMs = null;
      this._lastActivityTimeMs = null;
      this._frameAllocations = 0;
      this._frameBytesAllocated = 0;
      this._frameBytesUploaded = 0;
    }

    _assertNotDestroyed() {
      if (this._destroyed) {
        throw new Error('GpuStagingPool has been destroyed');
      }
    }

    _ensureInFrame() {
      if (!this._inFrame) {
        if (this._currentFrame < 0) {
          this._currentFrame = 0;
        }
        this._inFrame = true;
        this._frameStartedAtLeastOnce = true;
      }
    }

    _recordActivityTime() {
      const t = this._now();
      if (typeof t === 'number' && Number.isFinite(t)) {
        if (this._firstActivityTimeMs === null) {
          this._firstActivityTimeMs = t;
        }
        if (this._lastActivityTimeMs === null || t >= this._lastActivityTimeMs) {
          this._lastActivityTimeMs = t;
        }
      }
    }

    _getRecycledCount() {
      return (
        this._recycledStdQueue.size +
        this._recycledOversizedChunks.length +
        this._idleLargeSlabCount
      );
    }

    _addToActive(chunk) {
      chunk._activeIdx = this._activeChunks.length;
      this._activeChunks.push(chunk);
    }

    _removeFromActive(chunk) {
      const idx = chunk._activeIdx;
      if (idx < 0) {
        return false;
      }
      const arr = this._activeChunks;
      const lastIdx = arr.length - 1;
      if (idx !== lastIdx) {
        const moved = arr[lastIdx];
        arr[idx] = moved;
        moved._activeIdx = idx;
      }
      arr.pop();
      chunk._activeIdx = -1;
      return true;
    }

    _removeFromOversizedRecycled(chunk) {
      const idx = chunk._oversizedIdx;
      if (idx < 0) {
        return false;
      }
      const arr = this._recycledOversizedChunks;
      const lastIdx = arr.length - 1;
      if (idx !== lastIdx) {
        const moved = arr[lastIdx];
        arr[idx] = moved;
        moved._oversizedIdx = idx;
      }
      arr.pop();
      chunk._oversizedIdx = -1;
      return true;
    }

    _removeFromLargeSlabBucket(chunk) {
      const bucket = this._largeSlabBuckets.get(chunk.slabBucketSize);
      if (!bucket || !bucket.remove(chunk)) {
        return false;
      }
      this._idleLargeSlabCount = Math.max(0, this._idleLargeSlabCount - 1);
      if (bucket.size === 0) {
        this._largeSlabBuckets.delete(chunk.slabBucketSize);
      }
      return true;
    }

    _putIntoInFlight(chunk, frameIndex) {
      chunk.state = 'inFlight';
      chunk.retiredAtFrame = frameIndex;
      chunk.gpuDone = false;
      chunk.readyToRecycle = false;
      this._inFlightQueue.push(chunk);
    }

    _removeFromInFlight(chunk) {
      if (this._inFlightQueue.remove(chunk)) {
        if (chunk.gpuDone) {
          this._gpuDoneInFlightCount = Math.max(0, this._gpuDoneInFlightCount - 1);
        }
        return true;
      }
      return false;
    }

    _resolveLargeSlabBucket(alignedSize) {
      if (this.largeSlabBuckets.length === 0 || alignedSize < this.largeSlabMinBytes) {
        return 0;
      }
      let bucketSize = selectLargeSlabBucket(alignedSize, this.largeSlabBuckets);
      if (bucketSize > this.maxPoolBytes && alignedSize <= this.maxPoolBytes) {
        bucketSize = alignedSize;
      }
      return bucketSize;
    }

    _putIntoRecycled(chunk) {
      chunk.cursor = 0;
      chunk.subAllocCount = 0;
      chunk.retiredAtFrame = -1;
      chunk.gpuDone = false;
      chunk.readyToRecycle = false;
      chunk.state = 'recycled';
      if (chunk.isLargeSlab) {
        let bucket = this._largeSlabBuckets.get(chunk.slabBucketSize);
        if (!bucket) {
          bucket = new ChunkRingQueue(8);
          this._largeSlabBuckets.set(chunk.slabBucketSize, bucket);
        }
        bucket.push(chunk);
        this._idleLargeSlabCount += 1;
      } else if (chunk.oversized) {
        chunk._oversizedIdx = this._recycledOversizedChunks.length;
        this._recycledOversizedChunks.push(chunk);
      } else {
        this._recycledStdQueue.push(chunk);
      }
      this._recycledBytes += chunk.size;
    }

    _enforceLargeSlabCapInto(out) {
      out.reclaimedChunks = 0;
      out.reclaimedBytes = 0;
      if (this._idleLargeSlabCount === 0) {
        return;
      }

      let reclaimedChunks = 0;
      let reclaimedBytes = 0;

      // 1. Enforce optional per-bucket cap via deterministic LRU eviction within each bucket.
      if (this.maxSlabsPerBucket !== null) {
        for (const bucket of this._largeSlabBuckets.values()) {
          if (bucket.size <= this.maxSlabsPerBucket) {
            continue;
          }
          const bucketSlabs = [];
          const bBuf = bucket._buf;
          const bMask = bucket._mask;
          let bIdx = bucket._head;
          for (let i = 0; i < bucket._span; i += 1) {
            const slab = bBuf[bIdx];
            bIdx = (bIdx + 1) & bMask;
            if (slab !== undefined && !slab.destroyed && slab.state === 'recycled') {
              bucketSlabs.push(slab);
            }
          }
          if (bucketSlabs.length <= this.maxSlabsPerBucket) {
            continue;
          }
          bucketSlabs.sort(compareEvictionLru);
          const toEvict = bucketSlabs.length - this.maxSlabsPerBucket;
          for (let i = 0; i < toEvict; i += 1) {
            const slab = bucketSlabs[i];
            if (slab.destroyed || slab.state !== 'recycled') {
              continue;
            }
            const byteSize = slab.size;
            this._destroyChunk(slab);
            reclaimedChunks += 1;
            reclaimedBytes += byteSize;
            this._largeSlabEvictions += 1;
            this._evictedCount += 1;
            this._evictedBytes += byteSize;
          }
        }
      }

      // 2. Enforce global idle large-slab cap via deterministic LRU (largest first on same-frame tie).
      if (this._idleLargeSlabCount > this.maxLargeSlabs) {
        const idleSlabs = [];
        for (const bucket of this._largeSlabBuckets.values()) {
          const bBuf = bucket._buf;
          const bMask = bucket._mask;
          let bIdx = bucket._head;
          for (let i = 0; i < bucket._span; i += 1) {
            const slab = bBuf[bIdx];
            bIdx = (bIdx + 1) & bMask;
            if (slab !== undefined && !slab.destroyed && slab.state === 'recycled') {
              idleSlabs.push(slab);
            }
          }
        }

        idleSlabs.sort(compareEvictionLru);

        for (
          let i = 0;
          i < idleSlabs.length && this._idleLargeSlabCount > this.maxLargeSlabs;
          i += 1
        ) {
          const slab = idleSlabs[i];
          if (slab.destroyed || slab.state !== 'recycled') {
            continue;
          }
          const byteSize = slab.size;
          this._destroyChunk(slab);
          reclaimedChunks += 1;
          reclaimedBytes += byteSize;
          this._largeSlabEvictions += 1;
          this._evictedCount += 1;
          this._evictedBytes += byteSize;
        }
      }

      if (reclaimedChunks > 0) {
        this._reclaimedChunks += reclaimedChunks;
        this._reclaimedBytes += reclaimedBytes;
        this._trimmedStagingBytes += reclaimedBytes;
      }

      out.reclaimedChunks = reclaimedChunks;
      out.reclaimedBytes = reclaimedBytes;
    }

    _enforceLargeSlabCap() {
      this._enforceLargeSlabCapInto(this._reclaimScratch);
      return {
        reclaimedChunks: this._reclaimScratch.reclaimedChunks,
        reclaimedBytes: this._reclaimScratch.reclaimedBytes,
      };
    }

    _createChunk(byteSize, oversized, slabBucketSize = 0) {
      const allocSize = alignBufferSize(byteSize);
      this._ensureBudget(allocSize, byteSize);

      const id = this._nextChunkId++;
      const isLargeSlab = slabBucketSize > 0;
      const isMappedMode = this.mode === STAGING_MODE.MAPPED;
      const label = isLargeSlab ? `playgta6-staging-slab-${id}` : `playgta6-staging-${id}`;
      const descriptor = isMappedMode
        ? {
            label,
            size: allocSize,
            usage: BUFFER_USAGE.MAP_WRITE | BUFFER_USAGE.COPY_SRC,
            mappedAtCreation: true,
          }
        : {
            label,
            size: allocSize,
            usage: BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST,
          };

      const buffer = this.device.createBuffer(descriptor);
      let mappedView = null;
      let isMapped = false;
      if (isMappedMode) {
        if (typeof buffer.getMappedRange !== 'function') {
          if (typeof buffer.destroy === 'function') {
            buffer.destroy();
          }
          throw new TypeError(
            'GpuStagingPool mapped mode requires GPUBuffer.getMappedRange()'
          );
        }
        mappedView = new Uint8Array(buffer.getMappedRange());
        isMapped = true;
      }

      const chunk = {
        id,
        buffer,
        size: allocSize,
        cursor: 0,
        subAllocCount: 0,
        oversized,
        isLargeSlab,
        slabBucketSize,
        state: 'active',
        lastUsedFrame: this._currentFrame < 0 ? 0 : this._currentFrame,
        lastUsedSeq: ++this._lruSeq,
        retiredAtFrame: -1,
        gpuDone: false,
        readyToRecycle: false,
        isMapped,
        mapPending: false,
        mappedView,
        destroyed: false,
        _activeIdx: -1,
        _oversizedIdx: -1,
        _bucketIdx: -1,
        _queue: null,
        _queueSlot: -1,
      };

      this._allChunks.add(chunk);
      this._chunkByBuffer.set(buffer, chunk);
      this._totalChunksAllocated += 1;
      this._totalBytesAllocated += allocSize;
      this._allocatedBytes += allocSize;
      if (isLargeSlab) {
        this._largeSlabBytes += allocSize;
      }
      if (this._allocatedBytes > this._peakAllocatedBytes) {
        this._peakAllocatedBytes = this._allocatedBytes;
      }

      return chunk;
    }

    _destroyChunk(chunk) {
      if (!chunk || chunk.destroyed || chunk.state === 'destroyed') {
        return;
      }
      const wasRecycled = chunk.state === 'recycled';
      chunk.destroyed = true;
      chunk.state = 'destroyed';
      chunk.isMapped = false;
      chunk.mapPending = false;
      chunk.mappedView = null;
      this._allChunks.delete(chunk);
      if (chunk.buffer) {
        this._chunkByBuffer.delete(chunk.buffer);
      }
      if (this._currentChunk === chunk) {
        this._currentChunk = null;
      }
      this._removeFromActive(chunk);
      let removedFromRecycled = false;
      if (this._recycledStdQueue.remove(chunk)) {
        removedFromRecycled = true;
      }
      if (this._removeFromOversizedRecycled(chunk)) {
        removedFromRecycled = true;
      }
      if (chunk.isLargeSlab) {
        if (this._removeFromLargeSlabBucket(chunk)) {
          removedFromRecycled = true;
        }
        this._largeSlabBytes = Math.max(0, this._largeSlabBytes - chunk.size);
      }
      if (removedFromRecycled || wasRecycled) {
        this._recycledBytes = Math.max(0, this._recycledBytes - chunk.size);
      }
      this._removeFromInFlight(chunk);
      this._allocatedBytes = Math.max(0, this._allocatedBytes - chunk.size);
      if (chunk.buffer && typeof chunk.buffer.destroy === 'function') {
        chunk.buffer.destroy();
        this._destroyedBuffers += 1;
      }
    }

    _pickBudgetEvictionCandidate() {
      // 1. Prefer evicting idle non-bucketed oversized burst chunks via deterministic LRU
      //    (oldest frame first, largest first on tie) so steady-state standard chunks stay warm.
      let bestOversized = null;
      const oversizedArr = this._recycledOversizedChunks;
      for (let i = 0; i < oversizedArr.length; i += 1) {
        const candidate = oversizedArr[i];
        if (!candidate || candidate.destroyed || candidate.state !== 'recycled') {
          continue;
        }
        if (!bestOversized || compareEvictionLru(candidate, bestOversized) < 0) {
          bestOversized = candidate;
        }
      }
      if (bestOversized) {
        return bestOversized;
      }

      // 2. Next evict idle large slabs via deterministic LRU (oldest used first, then largest first).
      let bestSlab = null;
      for (const bucket of this._largeSlabBuckets.values()) {
        const bBuf = bucket._buf;
        const bMask = bucket._mask;
        let bIdx = bucket._head;
        for (let i = 0; i < bucket._span; i += 1) {
          const slab = bBuf[bIdx];
          bIdx = (bIdx + 1) & bMask;
          if (slab === undefined || slab.destroyed || slab.state !== 'recycled') {
            continue;
          }
          if (!bestSlab || compareEvictionLru(slab, bestSlab) < 0) {
            bestSlab = slab;
          }
        }
      }
      if (bestSlab) {
        return bestSlab;
      }

      // 3. Finally evict the LRU idle standard chunk.
      let bestStd = null;
      const stdQ = this._recycledStdQueue;
      const stdBuf = stdQ._buf;
      const stdMask = stdQ._mask;
      let idx = stdQ._head;
      for (let i = 0; i < stdQ._span; i += 1) {
        const candidate = stdBuf[idx];
        idx = (idx + 1) & stdMask;
        if (candidate === undefined || candidate.destroyed || candidate.state !== 'recycled') {
          continue;
        }
        if (!bestStd || compareEvictionLru(candidate, bestStd) < 0) {
          bestStd = candidate;
        }
      }
      return bestStd;
    }

    _reclaimCompletedInFlight(enforceCap = true) {
      if (this._destroyed || this._gpuDoneInFlightCount === 0 || this._inFlightQueue.size === 0) {
        return 0;
      }
      let reclaimed = 0;
      const q = this._inFlightQueue;
      const buf = q._buf;
      const mask = q._mask;
      let idx = q._head;
      const span = q._span;
      for (let i = 0; i < span; i += 1) {
        const chunk = buf[idx];
        idx = (idx + 1) & mask;
        if (chunk === undefined) {
          continue;
        }
        if (chunk.destroyed || chunk.state !== 'inFlight') {
          this._removeFromInFlight(chunk);
          continue;
        }
        if (chunk.gpuDone) {
          if (this.mode === STAGING_MODE.MAPPED && !chunk.isMapped) {
            chunk.readyToRecycle = true;
            this._beginRemap(chunk);
            if (chunk.isMapped) {
              this._removeFromInFlight(chunk);
              this._putIntoRecycled(chunk);
              reclaimed += 1;
            }
          } else {
            this._removeFromInFlight(chunk);
            this._putIntoRecycled(chunk);
            reclaimed += 1;
          }
        }
      }
      if (enforceCap && reclaimed > 0) {
        this._enforceLargeSlabCapInto(this._reclaimScratch);
      }
      return reclaimed;
    }

    /**
     * Synchronously move any GPU-completed (`onSubmittedWorkDone`) in-flight chunks
     * into the recycled pool so burst allocations can reuse them immediately.
     *
     * @returns {number} Number of chunks reclaimed into the recycled pool.
     */
    recycleCompleted() {
      this._assertNotDestroyed();
      const recycled = this._reclaimCompletedInFlight(true);
      if (this.idleShrinkFrames > 0) {
        this._shrinkIdleOversizedSlabsInto(this.idleShrinkFrames, this._reclaimScratch);
      }
      if (this.idleEvictFrames > 0) {
        this._reclaimIdleChunksInto(this.idleEvictFrames, this._reclaimScratch);
      }
      return recycled;
    }

    _shrinkIdleOversizedSlabsInto(threshold, out) {
      out.shrunkSlabs = 0;
      out.shrunkBytes = 0;
      if (this._recycledOversizedChunks.length === 0 && this._idleLargeSlabCount === 0) {
        return;
      }

      const currentFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
      let candidates = null;

      const oversizedArr = this._recycledOversizedChunks;
      for (let i = 0; i < oversizedArr.length; i += 1) {
        const chunk = oversizedArr[i];
        if (
          chunk &&
          !chunk.destroyed &&
          chunk.state === 'recycled' &&
          currentFrame - chunk.lastUsedFrame >= threshold
        ) {
          if (candidates === null) candidates = [];
          candidates.push(chunk);
        }
      }

      for (const bucket of this._largeSlabBuckets.values()) {
        const bBuf = bucket._buf;
        const bMask = bucket._mask;
        let bIdx = bucket._head;
        for (let i = 0; i < bucket._span; i += 1) {
          const slab = bBuf[bIdx];
          bIdx = (bIdx + 1) & bMask;
          if (
            slab !== undefined &&
            !slab.destroyed &&
            slab.state === 'recycled' &&
            currentFrame - slab.lastUsedFrame >= threshold
          ) {
            if (candidates === null) candidates = [];
            candidates.push(slab);
          }
        }
      }

      if (candidates === null) {
        return;
      }

      if (candidates.length > 1) {
        candidates.sort(compareEvictionLru);
      }

      let shrunkSlabs = 0;
      let shrunkBytes = 0;
      for (let i = 0; i < candidates.length; i += 1) {
        const chunk = candidates[i];
        if (chunk.destroyed || chunk.state !== 'recycled') {
          continue;
        }
        const byteSize = chunk.size;
        const wasLargeSlab = chunk.isLargeSlab;
        this._destroyChunk(chunk);
        if (wasLargeSlab) {
          this._largeSlabEvictions += 1;
        }
        this._evictedCount += 1;
        this._evictedBytes += byteSize;
        this._idleSlabShrinks += 1;
        shrunkSlabs += 1;
        shrunkBytes += byteSize;
      }

      if (shrunkSlabs > 0) {
        this._reclaimedChunks += shrunkSlabs;
        this._reclaimedBytes += shrunkBytes;
        this._trimmedStagingBytes += shrunkBytes;
      }

      out.shrunkSlabs = shrunkSlabs;
      out.shrunkBytes = shrunkBytes;
    }

    /**
     * Destroy and remove idle recycled oversized / large staging slabs
     * (`chunk.oversized || chunk.isLargeSlab`) whose
     * `currentFrame - chunk.lastUsedFrame >= maxIdleFrames`.
     *
     * Keeps standard steady-state bump-allocation chunks warm while releasing
     * transient boot-time or cutscene oversized slabs to keep GPU process RAM bounded.
     * Never touches active or in-flight slabs.
     *
     * @param {number} [maxIdleFrames]
     * @returns {{ shrunkSlabs: number, shrunkBytes: number }}
     */
    shrinkIdleOversizedSlabs(maxIdleFrames) {
      this._assertNotDestroyed();

      let threshold;
      if (maxIdleFrames === undefined) {
        if (this.idleShrinkFrames <= 0) {
          return { shrunkSlabs: 0, shrunkBytes: 0 };
        }
        threshold = this.idleShrinkFrames;
      } else {
        assertNonNegativeInteger(maxIdleFrames, 'maxIdleFrames');
        threshold = maxIdleFrames;
      }

      this._shrinkIdleOversizedSlabsInto(threshold, this._reclaimScratch);
      return {
        shrunkSlabs: this._reclaimScratch.shrunkSlabs,
        shrunkBytes: this._reclaimScratch.shrunkBytes,
      };
    }

    /**
     * Aggressively destroy idle oversized staging slabs (`slab.size > maxRetainedSlabBytes`,
     * defaulting to `this.chunkSize`) immediately after heavy scene transitions
     * (e.g., 4K Bink2 intro -> 3D Main Menu -> In-Game), without touching active or
     * in-flight slabs or warm standard chunks.
     *
     * @param {{ maxRetainedSlabBytes?: number, includeAllLargeSlabs?: boolean, reclaimCompletedFirst?: boolean, keepOversizedSlabs?: number }} [options]
     * @returns {{ reclaimedSlabs: number, reclaimedBytes: number }}
     */
    trimForSceneTransition(options) {
      this._assertNotDestroyed();

      let maxRetainedSlabBytes = this.chunkSize;
      let includeAllLargeSlabs = false;
      let reclaimCompletedFirst = true;
      let keepOversizedSlabs = 0;

      if (options !== undefined && options !== null) {
        if (typeof options !== 'object' || Array.isArray(options)) {
          throw new TypeError('trimForSceneTransition options must be an object');
        }
        if (options.maxRetainedSlabBytes !== undefined) {
          assertPositiveInteger(options.maxRetainedSlabBytes, 'maxRetainedSlabBytes');
          maxRetainedSlabBytes = options.maxRetainedSlabBytes;
        }
        if (options.includeAllLargeSlabs !== undefined) {
          if (typeof options.includeAllLargeSlabs !== 'boolean') {
            throw new TypeError('includeAllLargeSlabs must be a boolean');
          }
          includeAllLargeSlabs = options.includeAllLargeSlabs;
        }
        if (options.reclaimCompletedFirst !== undefined) {
          if (typeof options.reclaimCompletedFirst !== 'boolean') {
            throw new TypeError('reclaimCompletedFirst must be a boolean');
          }
          reclaimCompletedFirst = options.reclaimCompletedFirst;
        }
        if (options.keepOversizedSlabs !== undefined) {
          assertNonNegativeInteger(options.keepOversizedSlabs, 'keepOversizedSlabs');
          keepOversizedSlabs = options.keepOversizedSlabs;
        }
      }

      const reclaimedInFlight = reclaimCompletedFirst ? this._reclaimCompletedInFlight(false) : 0;

      const candidates = [];
      if (this.chunkSize > maxRetainedSlabBytes && this._recycledStdQueue.size > 0) {
        const stdQ = this._recycledStdQueue;
        const stdBuf = stdQ._buf;
        const stdMask = stdQ._mask;
        let idx = stdQ._head;
        for (let i = 0; i < stdQ._span; i += 1) {
          const chunk = stdBuf[idx];
          idx = (idx + 1) & stdMask;
          if (
            chunk !== undefined &&
            !chunk.destroyed &&
            chunk.state === 'recycled' &&
            chunk.size > maxRetainedSlabBytes
          ) {
            candidates.push(chunk);
          }
        }
      }

      const oversizedArr = this._recycledOversizedChunks;
      for (let i = 0; i < oversizedArr.length; i += 1) {
        const chunk = oversizedArr[i];
        if (
          chunk &&
          !chunk.destroyed &&
          chunk.state === 'recycled' &&
          (chunk.size > maxRetainedSlabBytes || (includeAllLargeSlabs && chunk.isLargeSlab))
        ) {
          candidates.push(chunk);
        }
      }

      for (const bucket of this._largeSlabBuckets.values()) {
        const bBuf = bucket._buf;
        const bMask = bucket._mask;
        let bIdx = bucket._head;
        for (let i = 0; i < bucket._span; i += 1) {
          const slab = bBuf[bIdx];
          bIdx = (bIdx + 1) & bMask;
          if (
            slab !== undefined &&
            !slab.destroyed &&
            slab.state === 'recycled' &&
            (slab.size > maxRetainedSlabBytes || (includeAllLargeSlabs && slab.isLargeSlab))
          ) {
            candidates.push(slab);
          }
        }
      }

      let reclaimedSlabs = 0;
      let reclaimedBytes = 0;

      if (candidates.length > keepOversizedSlabs) {
        candidates.sort(compareEvictionLru);
        const destroyCount = candidates.length - keepOversizedSlabs;
        for (let i = 0; i < destroyCount; i += 1) {
          const chunk = candidates[i];
          if (chunk.destroyed || chunk.state !== 'recycled') {
            continue;
          }
          const byteSize = chunk.size;
          const wasLargeSlab = chunk.isLargeSlab;
          this._destroyChunk(chunk);
          if (wasLargeSlab) {
            this._largeSlabEvictions += 1;
          }
          this._evictedCount += 1;
          this._evictedBytes += byteSize;
          reclaimedSlabs += 1;
          reclaimedBytes += byteSize;
        }
      }

      if (reclaimedSlabs > 0) {
        this._reclaimedChunks += reclaimedSlabs;
        this._reclaimedBytes += reclaimedBytes;
        this._sceneTransitionTrimBytes += reclaimedBytes;
        this._trimmedStagingBytes += reclaimedBytes;
      }
      this._uploadScratch = null;

      if (reclaimedInFlight > 0) {
        this._enforceLargeSlabCapInto(this._reclaimScratch);
      }

      return { reclaimedSlabs, reclaimedBytes };
    }

    _reclaimIdleChunksInto(threshold, out) {
      out.reclaimedChunks = 0;
      out.reclaimedBytes = 0;

      if (this._getRecycledCount() <= this.minWarmChunks) {
        return;
      }

      const currentFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
      let candidates = null;

      const stdQ = this._recycledStdQueue;
      const stdBuf = stdQ._buf;
      const stdMask = stdQ._mask;
      let idx = stdQ._head;
      for (let i = 0; i < stdQ._span; i += 1) {
        const chunk = stdBuf[idx];
        idx = (idx + 1) & stdMask;
        if (
          chunk !== undefined &&
          !chunk.destroyed &&
          chunk.state === 'recycled' &&
          currentFrame - chunk.lastUsedFrame >= threshold
        ) {
          if (candidates === null) candidates = [];
          candidates.push(chunk);
        }
      }

      const oversizedArr = this._recycledOversizedChunks;
      for (let i = 0; i < oversizedArr.length; i += 1) {
        const chunk = oversizedArr[i];
        if (
          chunk &&
          !chunk.destroyed &&
          chunk.state === 'recycled' &&
          currentFrame - chunk.lastUsedFrame >= threshold
        ) {
          if (candidates === null) candidates = [];
          candidates.push(chunk);
        }
      }

      for (const bucket of this._largeSlabBuckets.values()) {
        const bBuf = bucket._buf;
        const bMask = bucket._mask;
        let bIdx = bucket._head;
        for (let i = 0; i < bucket._span; i += 1) {
          const chunk = bBuf[bIdx];
          bIdx = (bIdx + 1) & bMask;
          if (
            chunk !== undefined &&
            !chunk.destroyed &&
            chunk.state === 'recycled' &&
            currentFrame - chunk.lastUsedFrame >= threshold
          ) {
            if (candidates === null) candidates = [];
            candidates.push(chunk);
          }
        }
      }

      if (candidates === null) {
        return;
      }

      if (candidates.length > 1) {
        candidates.sort((a, b) => {
          const aLarge = a.oversized || a.isLargeSlab;
          const bLarge = b.oversized || b.isLargeSlab;
          if (aLarge !== bLarge) {
            return aLarge ? -1 : 1;
          }
          if (aLarge && a.size !== b.size) {
            return b.size - a.size;
          }
          if (a.lastUsedFrame !== b.lastUsedFrame) {
            return a.lastUsedFrame - b.lastUsedFrame;
          }
          if (a.lastUsedSeq !== b.lastUsedSeq) {
            return a.lastUsedSeq - b.lastUsedSeq;
          }
          return a.id - b.id;
        });
      }

      let reclaimedChunks = 0;
      let reclaimedBytes = 0;
      for (let i = 0; i < candidates.length; i += 1) {
        if (this._getRecycledCount() <= this.minWarmChunks) {
          break;
        }
        const chunk = candidates[i];
        if (chunk.destroyed || chunk.state !== 'recycled') {
          continue;
        }
        const byteSize = chunk.size;
        const wasLargeSlab = chunk.isLargeSlab;
        this._destroyChunk(chunk);
        if (wasLargeSlab) {
          this._largeSlabEvictions += 1;
        }
        this._evictedCount += 1;
        this._evictedBytes += byteSize;
        reclaimedChunks += 1;
        reclaimedBytes += byteSize;
      }

      if (reclaimedChunks > 0) {
        this._reclaimedChunks += reclaimedChunks;
        this._reclaimedBytes += reclaimedBytes;
        this._trimmedStagingBytes += reclaimedBytes;
      }

      out.reclaimedChunks = reclaimedChunks;
      out.reclaimedBytes = reclaimedBytes;
    }

    /**
     * Destroy and remove idle recycled (not active, not in-flight) staging chunks
     * whose `currentFrame - chunk.lastUsedFrame >= maxIdleFrames`, while keeping
     * at least `minWarmChunks` warm chunks in the recycled pool.
     *
     * Prefers evicting oversized burst chunks / large slabs (largest first) and
     * oldest chunks first so steady-state standard chunks stay warm.
     *
     * @param {number} [maxIdleFrames]
     * @returns {{ reclaimedChunks: number, reclaimedBytes: number }}
     */
    reclaimIdleChunks(maxIdleFrames) {
      this._assertNotDestroyed();

      let threshold;
      if (maxIdleFrames === undefined) {
        if (this.idleEvictFrames <= 0) {
          return { reclaimedChunks: 0, reclaimedBytes: 0 };
        }
        threshold = this.idleEvictFrames;
      } else {
        assertNonNegativeInteger(maxIdleFrames, 'maxIdleFrames');
        threshold = maxIdleFrames;
      }

      this._reclaimIdleChunksInto(threshold, this._reclaimScratch);
      return {
        reclaimedChunks: this._reclaimScratch.reclaimedChunks,
        reclaimedBytes: this._reclaimScratch.reclaimedBytes,
      };
    }

    _ensureBudget(allocSize, requestedSize) {
      if (this._allocatedBytes + allocSize <= this.maxPoolBytes) {
        return;
      }

      // First reclaim any GPU-completed in-flight chunks so burst eviction can reuse/free them.
      const reclaimedInFlight = this._reclaimCompletedInFlight(false);

      // O(1) check whether evicting idle recycled chunks / idle large slabs can make room.
      const reclaimable = this._recycledBytes;

      if (this._allocatedBytes - reclaimable + allocSize <= this.maxPoolBytes) {
        while (this._allocatedBytes + allocSize > this.maxPoolBytes) {
          const evicted = this._pickBudgetEvictionCandidate();
          if (!evicted) {
            break;
          }
          const wasLargeSlab = evicted.isLargeSlab;
          const byteSize = evicted.size;
          this._destroyChunk(evicted);
          this._evictedCount += 1;
          this._evictedBytes += byteSize;
          if (wasLargeSlab) {
            this._largeSlabEvictions += 1;
          }
        }
        if (reclaimedInFlight > 0) {
          this._enforceLargeSlabCapInto(this._reclaimScratch);
        }
        return;
      }

      if (reclaimedInFlight > 0) {
        this._enforceLargeSlabCapInto(this._reclaimScratch);
      }

      throw new RangeError(
        `GpuStagingPool budget exceeded: cannot allocate ${allocSize}-byte chunk ` +
          `(requested ${requestedSize} bytes; allocated=${this._allocatedBytes}, ` +
          `maxPoolBytes=${this.maxPoolBytes})`
      );
    }

    _takeLargeSlab(bucketSize) {
      const bucket = this._largeSlabBuckets.get(bucketSize);
      if (!bucket || bucket.size === 0) {
        return null;
      }
      while (bucket.size > 0) {
        const slab = bucket.pop();
        this._idleLargeSlabCount = Math.max(0, this._idleLargeSlabCount - 1);
        if (bucket.size === 0) {
          this._largeSlabBuckets.delete(bucketSize);
        }
        if (slab && !slab.destroyed && slab.state === 'recycled') {
          this._recycledBytes = Math.max(0, this._recycledBytes - slab.size);
          slab.cursor = 0;
          slab.subAllocCount = 0;
          slab.lastUsedFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
          slab.lastUsedSeq = ++this._lruSeq;
          slab.retiredAtFrame = -1;
          slab.gpuDone = false;
          slab.readyToRecycle = false;
          slab.state = 'active';
          return slab;
        }
      }
      return null;
    }

    _acquireLargeSlab(bucketSize, oversized) {
      let recycled = this._takeLargeSlab(bucketSize);

      if (!recycled && this._reclaimCompletedInFlight(false) > 0) {
        recycled = this._takeLargeSlab(bucketSize);
        this._enforceLargeSlabCapInto(this._reclaimScratch);
      }

      if (recycled) {
        this._bufferReuseHits += 1;
        this._largeSlabHits += 1;
        this._bytesReused += recycled.size;
        this._addToActive(recycled);
        this._lastAcquireFresh = false;
        if (this._inFlightQueue.size > 0) {
          this._stallAvoidedCount += 1;
        }
        return recycled;
      }

      const created = this._createChunk(bucketSize, oversized, bucketSize);
      this._bufferReuseMisses += 1;
      this._largeSlabMisses += 1;
      this._addToActive(created);
      this._lastAcquireFresh = true;
      if (this._inFlightQueue.size > 0) {
        this._stallAvoidedCount += 1;
      }
      return created;
    }

    _takeRecycledChunk(minSize, oversized) {
      // O(1) FIFO fast path for standard steady-state chunks (zero scan, zero shift).
      if (!oversized && minSize <= this.chunkSize && this._recycledStdQueue.size > 0) {
        while (this._recycledStdQueue.size > 0) {
          const chunk = this._recycledStdQueue.shift();
          if (chunk && !chunk.destroyed && chunk.state === 'recycled' && chunk.size >= minSize) {
            this._recycledBytes = Math.max(0, this._recycledBytes - chunk.size);
            chunk.cursor = 0;
            chunk.subAllocCount = 0;
            chunk.lastUsedFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
            chunk.lastUsedSeq = ++this._lruSeq;
            chunk.retiredAtFrame = -1;
            chunk.gpuDone = false;
            chunk.readyToRecycle = false;
            chunk.state = 'active';
            return chunk;
          }
        }
      }

      const oversizedArr = this._recycledOversizedChunks;
      if (oversizedArr.length === 0) {
        return null;
      }

      let bestIndex = -1;
      let bestSize = Infinity;
      for (let i = 0; i < oversizedArr.length; i += 1) {
        const candidate = oversizedArr[i];
        if (
          !candidate.destroyed &&
          candidate.state === 'recycled' &&
          candidate.size >= minSize &&
          candidate.size < bestSize
        ) {
          bestIndex = i;
          bestSize = candidate.size;
        }
      }

      if (bestIndex === -1) {
        return null;
      }

      const chunk = oversizedArr[bestIndex];
      this._removeFromOversizedRecycled(chunk);
      this._recycledBytes = Math.max(0, this._recycledBytes - chunk.size);
      chunk.cursor = 0;
      chunk.subAllocCount = 0;
      chunk.lastUsedFrame = this._currentFrame < 0 ? 0 : this._currentFrame;
      chunk.lastUsedSeq = ++this._lruSeq;
      chunk.retiredAtFrame = -1;
      chunk.gpuDone = false;
      chunk.readyToRecycle = false;
      chunk.state = 'active';
      return chunk;
    }

    _acquireChunk(requiredSize, oversized) {
      const targetSize = oversized ? alignBufferSize(requiredSize) : this.chunkSize;
      const minRequired = oversized ? targetSize : requiredSize;
      let recycled = this._takeRecycledChunk(minRequired, oversized);

      // During Frame-1 or mid-frame bursts, reclaim GPU-finished chunks on demand
      // before allocating fresh GPUBuffers.
      if (!recycled && this._reclaimCompletedInFlight(false) > 0) {
        recycled = this._takeRecycledChunk(minRequired, oversized);
        this._enforceLargeSlabCapInto(this._reclaimScratch);
      }

      if (recycled) {
        this._bufferReuseHits += 1;
        this._bytesReused += recycled.size;
        this._addToActive(recycled);
        this._lastAcquireFresh = false;
        if (this._inFlightQueue.size > 0) {
          this._stallAvoidedCount += 1;
        }
        return recycled;
      }

      const created = this._createChunk(targetSize, oversized, 0);
      this._bufferReuseMisses += 1;
      this._addToActive(created);
      this._lastAcquireFresh = true;
      if (this._inFlightQueue.size > 0) {
        this._stallAvoidedCount += 1;
      }
      return created;
    }

    _moveInFlightToRecycled(chunk) {
      if (this._destroyed || !chunk || chunk.destroyed || chunk.state !== 'inFlight') {
        return false;
      }
      if (!this._removeFromInFlight(chunk)) {
        return false;
      }
      this._putIntoRecycled(chunk);
      if (chunk.isLargeSlab) {
        this._enforceLargeSlabCapInto(this._reclaimScratch);
      }
      return true;
    }

    _beginRemap(chunk) {
      if (
        this._destroyed ||
        chunk.destroyed ||
        this.mode !== STAGING_MODE.MAPPED ||
        chunk.isMapped ||
        chunk.mapPending
      ) {
        return;
      }

      if (typeof chunk.buffer.mapAsync === 'function') {
        chunk.mapPending = true;
        const mapPromise = Promise.resolve(chunk.buffer.mapAsync(MAP_MODE.WRITE)).then(
          () => {
            chunk.mapPending = false;
            if (this._destroyed || chunk.destroyed) {
              return;
            }
            chunk.mappedView = new Uint8Array(chunk.buffer.getMappedRange());
            chunk.isMapped = true;
            if (chunk.readyToRecycle) {
              this._moveInFlightToRecycled(chunk);
            }
          },
          () => {
            chunk.mapPending = false;
          }
        );
        this._pendingMapPromises.add(mapPromise);
        mapPromise.finally(() => {
          this._pendingMapPromises.delete(mapPromise);
        });
      } else if (typeof chunk.buffer.getMappedRange === 'function') {
        chunk.mappedView = new Uint8Array(chunk.buffer.getMappedRange());
        chunk.isMapped = true;
      }
    }

    _onWorkDoneChunk(chunk) {
      if (
        this._destroyed ||
        !chunk ||
        chunk.destroyed ||
        chunk.state !== 'inFlight' ||
        !this._inFlightQueue.contains(chunk)
      ) {
        return;
      }
      if (!chunk.gpuDone) {
        chunk.gpuDone = true;
        this._gpuDoneInFlightCount += 1;
      }
      if (this.mode === STAGING_MODE.MAPPED && !chunk.isMapped) {
        this._beginRemap(chunk);
      }
    }

    _onWorkDone(retiredChunks) {
      if (this._destroyed) {
        return;
      }
      for (let i = 0; i < retiredChunks.length; i += 1) {
        this._onWorkDoneChunk(retiredChunks[i]);
      }
    }

    _attachWorkDoneFenceSingle(chunk) {
      if (
        chunk &&
        this.device.queue &&
        typeof this.device.queue.onSubmittedWorkDone === 'function'
      ) {
        const workDone = this.device.queue.onSubmittedWorkDone();
        if (workDone && typeof workDone.then === 'function') {
          workDone.then(
            () => {
              this._onWorkDoneChunk(chunk);
            },
            () => {
              // Ignore rejected fence (e.g., lost/destroyed device).
            }
          );
        }
      }
    }

    _attachWorkDoneFence(retiredChunks) {
      if (
        retiredChunks.length > 0 &&
        this.device.queue &&
        typeof this.device.queue.onSubmittedWorkDone === 'function'
      ) {
        const workDone = this.device.queue.onSubmittedWorkDone();
        if (workDone && typeof workDone.then === 'function') {
          workDone.then(
            () => {
              this._onWorkDone(retiredChunks);
            },
            () => {
              // Ignore rejected fence (e.g., lost/destroyed device).
            }
          );
        }
      }
    }

    /**
     * Begin a new frame epoch and recycle any in-flight chunks whose retirement epoch
     * has elapsed (`frameIndex - retiredAtFrame >= framesInFlight`) or whose GPU
     * `onSubmittedWorkDone` fence has resolved.
     *
     * @param {number} [frameIndex]
     * @returns {{ frameIndex: number, recycledCount: number, recycledChunks: number, inFlightChunks: number }}
     */
    beginFrame(frameIndex) {
      this._assertNotDestroyed();

      if (this._inFrame) {
        this.endFrame();
      }

      let nextFrame;
      if (frameIndex === undefined) {
        nextFrame = this._currentFrame + 1;
      } else {
        assertNonNegativeInteger(frameIndex, 'frameIndex');
        nextFrame = frameIndex;
      }

      this._currentFrame = nextFrame;
      this._inFrame = true;
      this._frameStartedAtLeastOnce = true;
      this._recordActivityTime();
      this._frameAllocations = 0;
      this._frameBytesAllocated = 0;
      this._frameBytesUploaded = 0;

      let recycledCount = 0;
      const q = this._inFlightQueue;
      if (q.size > 0) {
        const buf = q._buf;
        const mask = q._mask;
        let idx = q._head;
        const span = q._span;
        const framesInFlight = this.framesInFlight;
        const isMappedMode = this.mode === STAGING_MODE.MAPPED;
        for (let i = 0; i < span; i += 1) {
          const chunk = buf[idx];
          idx = (idx + 1) & mask;
          if (chunk === undefined) {
            continue;
          }
          if (chunk.destroyed || chunk.state !== 'inFlight') {
            this._removeFromInFlight(chunk);
            continue;
          }
          const epochElapsed = nextFrame - chunk.retiredAtFrame >= framesInFlight;
          if (chunk.gpuDone || epochElapsed) {
            if (isMappedMode && !chunk.isMapped) {
              chunk.readyToRecycle = true;
              this._beginRemap(chunk);
              if (chunk.isMapped) {
                this._removeFromInFlight(chunk);
                this._putIntoRecycled(chunk);
                recycledCount += 1;
              }
            } else {
              this._removeFromInFlight(chunk);
              this._putIntoRecycled(chunk);
              recycledCount += 1;
            }
          }
        }
      }

      const scratch = this._reclaimScratch;
      this._enforceLargeSlabCapInto(scratch);
      let reclaimedChunks = scratch.reclaimedChunks;
      let reclaimedBytes = scratch.reclaimedBytes;
      let shrunkSlabs = 0;
      let shrunkBytes = 0;
      if (this.idleShrinkFrames > 0) {
        this._shrinkIdleOversizedSlabsInto(this.idleShrinkFrames, scratch);
        shrunkSlabs = scratch.shrunkSlabs;
        shrunkBytes = scratch.shrunkBytes;
        reclaimedChunks += shrunkSlabs;
        reclaimedBytes += shrunkBytes;
      }
      if (this.idleEvictFrames > 0) {
        this._reclaimIdleChunksInto(this.idleEvictFrames, scratch);
        reclaimedChunks += scratch.reclaimedChunks;
        reclaimedBytes += scratch.reclaimedBytes;
      }

      return {
        frameIndex: this._currentFrame,
        recycledCount,
        recycledChunks: this._getRecycledCount(),
        inFlightChunks: this._inFlightQueue.size,
        reclaimedChunks,
        reclaimedBytes,
        shrunkSlabs,
        shrunkBytes,
      };
    }

    /**
     * Unmap any active mapped staging chunks prior to `device.queue.submit()`.
     * Called automatically by `endFrame()` if not invoked explicitly beforehand.
     */
    prepareSubmit() {
      this._assertNotDestroyed();
      if (this.mode !== STAGING_MODE.MAPPED) {
        return;
      }
      for (let i = 0; i < this._activeChunks.length; i += 1) {
        const chunk = this._activeChunks[i];
        if (!chunk.destroyed && chunk.isMapped) {
          if (typeof chunk.buffer.unmap === 'function') {
            chunk.buffer.unmap();
          }
          chunk.isMapped = false;
          chunk.mappedView = null;
        }
      }
    }

    /**
     * Await any in-progress `GPUBuffer.mapAsync()` operations (useful in `'mapped'` mode).
     */
    async waitForPendingMaps() {
      while (!this._destroyed && this._pendingMapPromises.size > 0) {
        await Promise.all(Array.from(this._pendingMapPromises));
      }
    }

    /**
     * Explicitly retire an active staging slice or GPUBuffer into the in-flight queue.
     * Guarded against double-release: releasing an already-retired, recycled, destroyed,
     * or unknown buffer is a safe no-op returning `false`.
     *
     * @param {{ buffer: GPUBuffer } | GPUBuffer} sliceOrBuffer
     * @returns {boolean} `true` if the active chunk was retired, `false` if ignored.
     */
    release(sliceOrBuffer) {
      this._assertNotDestroyed();
      if (!sliceOrBuffer || typeof sliceOrBuffer !== 'object') {
        return false;
      }
      const buffer = sliceOrBuffer.buffer !== undefined ? sliceOrBuffer.buffer : sliceOrBuffer;
      if (!buffer || typeof buffer !== 'object') {
        return false;
      }
      const chunk = this._chunkByBuffer.get(buffer);
      if (!chunk) {
        return false;
      }
      if (chunk.destroyed || chunk.state !== 'active') {
        this._doubleReleaseIgnored += 1;
        return false;
      }

      this._removeFromActive(chunk);
      if (this._currentChunk === chunk) {
        this._currentChunk = null;
      }

      if (this.mode === STAGING_MODE.MAPPED && chunk.isMapped) {
        if (typeof chunk.buffer.unmap === 'function') {
          chunk.buffer.unmap();
        }
        chunk.isMapped = false;
        chunk.mappedView = null;
      }

      this._putIntoInFlight(chunk, this._currentFrame < 0 ? 0 : this._currentFrame);
      this._attachWorkDoneFenceSingle(chunk);
      return true;
    }

    /**
     * End the current frame, unmap any active mapped chunks, retire active chunks
     * into the in-flight queue for recycling after `framesInFlight` frames (or once
     * `device.queue.onSubmittedWorkDone` resolves), and return per-frame upload stats.
     *
     * @returns {{ frameIndex: number, chunksRetired: number, allocations: number, bytesAllocated: number, bytesUploaded: number, inFlightChunks: number, recycledChunks: number }}
     */
    endFrame() {
      this._assertNotDestroyed();

      if (this._currentFrame < 0) {
        this._currentFrame = 0;
      }

      const wasNotInFrame = !this._inFrame && this._frameStartedAtLeastOnce;

      this.prepareSubmit();

      const active = this._activeChunks;
      const activeLen = active.length;
      this._currentChunk = null;

      const hasWorkDoneFence = Boolean(
        activeLen > 0 &&
          this.device.queue &&
          typeof this.device.queue.onSubmittedWorkDone === 'function'
      );
      let retiredCount = 0;
      let firstRetired = null;
      let retiredList = null;

      for (let i = 0; i < activeLen; i += 1) {
        const chunk = active[i];
        active[i] = undefined;
        if (chunk) {
          chunk._activeIdx = -1;
        }
        if (!chunk || chunk.destroyed || chunk.state !== 'active') {
          this._doubleReleaseIgnored += 1;
          continue;
        }
        this._putIntoInFlight(chunk, this._currentFrame);
        retiredCount += 1;
        if (hasWorkDoneFence) {
          if (retiredCount === 1) {
            firstRetired = chunk;
          } else if (retiredCount === 2) {
            retiredList = [firstRetired, chunk];
          } else {
            retiredList.push(chunk);
          }
        }
      }
      active.length = 0;

      const isDuplicateEnd = wasNotInFrame && retiredCount === 0;
      if (isDuplicateEnd) {
        this._doubleReleaseIgnored += 1;
      } else {
        this._framesCompleted += 1;
        this._lastFrameBytesUploaded = this._frameBytesUploaded;
        if (this._frameBytesUploaded > this._peakFrameBytesUploaded) {
          this._peakFrameBytesUploaded = this._frameBytesUploaded;
        }
        this._recordActivityTime();
      }

      if (hasWorkDoneFence) {
        if (retiredList !== null) {
          this._attachWorkDoneFence(retiredList);
        } else if (firstRetired !== null) {
          this._attachWorkDoneFenceSingle(firstRetired);
        }
      }

      const scratch = this._reclaimScratch;
      this._enforceLargeSlabCapInto(scratch);
      let reclaimedChunks = scratch.reclaimedChunks;
      let reclaimedBytes = scratch.reclaimedBytes;
      let shrunkSlabs = 0;
      let shrunkBytes = 0;
      if (this.idleShrinkFrames > 0) {
        this._shrinkIdleOversizedSlabsInto(this.idleShrinkFrames, scratch);
        shrunkSlabs = scratch.shrunkSlabs;
        shrunkBytes = scratch.shrunkBytes;
        reclaimedChunks += shrunkSlabs;
        reclaimedBytes += shrunkBytes;
      }
      if (this.idleEvictFrames > 0) {
        this._reclaimIdleChunksInto(this.idleEvictFrames, scratch);
        reclaimedChunks += scratch.reclaimedChunks;
        reclaimedBytes += scratch.reclaimedBytes;
      }

      const frameStats = {
        frameIndex: this._currentFrame,
        chunksRetired: retiredCount,
        allocations: this._frameAllocations,
        bytesAllocated: this._frameBytesAllocated,
        bytesUploaded: this._frameBytesUploaded,
        inFlightChunks: this._inFlightQueue.size,
        recycledChunks: this._getRecycledCount(),
        reclaimedChunks,
        reclaimedBytes,
        shrunkSlabs,
        shrunkBytes,
      };

      this._inFrame = false;
      this._frameAllocations = 0;
      this._frameBytesAllocated = 0;
      this._frameBytesUploaded = 0;

      return frameStats;
    }

    _allocateInternal(byteSize, alignment, outTarget, countZeroAlloc) {
      this._assertNotDestroyed();
      assertPositiveInteger(byteSize, 'byteSize');
      assertPositiveInteger(alignment, 'alignment');
      this._ensureInFrame();

      const alignedSize = alignBufferSize(byteSize);
      const effectiveAlignment = lcmWithBufferAlignment(alignment);

      const isOversized = alignedSize > this.chunkSize;
      const slabBucketSize = this._resolveLargeSlabBucket(alignedSize);
      const isLargeSlab = slabBucketSize > 0;
      let chunk;
      let offset;
      let isFreshBuffer = false;

      if (isLargeSlab) {
        // Persistent large-slab bucket (e.g., 2–16 MiB 4K Bink2 YUV planes);
        // preserves _currentChunk for smaller sub-allocations.
        chunk = this._acquireLargeSlab(slabBucketSize, isOversized);
        isFreshBuffer = this._lastAcquireFresh;
        offset = 0;
        chunk.cursor = alignedSize;
        chunk.subAllocCount = 1;
      } else if (isOversized) {
        // Dedicated oversized chunk; preserve _currentChunk for smaller allocations.
        chunk = this._acquireChunk(alignedSize, true);
        isFreshBuffer = this._lastAcquireFresh;
        offset = 0;
        chunk.cursor = alignedSize;
        chunk.subAllocCount = 1;
      } else {
        if (this._currentChunk && !this._currentChunk.destroyed && this._currentChunk.state === 'active') {
          const candidateOffset = alignUp(this._currentChunk.cursor, effectiveAlignment);
          if (candidateOffset + alignedSize <= this._currentChunk.size) {
            chunk = this._currentChunk;
            offset = candidateOffset;
            chunk.cursor = offset + alignedSize;
            chunk.subAllocCount += 1;
            isFreshBuffer = false;
            this._subAllocCoalesced += 1;
          }
        }

        if (!chunk) {
          chunk = this._acquireChunk(alignedSize, false);
          isFreshBuffer = this._lastAcquireFresh;
          offset = 0;
          chunk.cursor = alignedSize;
          chunk.subAllocCount = 1;
          this._currentChunk = chunk;
        }
      }

      chunk.lastUsedFrame = this._currentFrame;
      chunk.lastUsedSeq = ++this._lruSeq;

      if (isOversized) {
        this._oversizeAllocations += 1;
      }

      if (alignedSize !== byteSize) {
        this._alignPadCount += 1;
      }

      if (isFreshBuffer) {
        this._freshAllocations += 1;
      } else {
        this._reusedAllocations += 1;
      }

      if (byteSize > this._peakSingleAllocationBytes) {
        this._peakSingleAllocationBytes = byteSize;
      }

      this._allocationCount += 1;
      this._totalBytesStaged += byteSize;
      this._frameAllocations += 1;
      this._frameBytesAllocated += byteSize;

      if (outTarget !== null) {
        outTarget.buffer = chunk.buffer;
        outTarget.offset = offset;
        outTarget.size = byteSize;
        outTarget.alignedSize = alignedSize;
        if (isLargeSlab) {
          outTarget.isLargeSlab = true;
          outTarget.slabBucketSize = chunk.slabBucketSize;
        } else {
          outTarget.isLargeSlab = false;
          outTarget.slabBucketSize = 0;
        }
        if (chunk.isMapped && chunk.mappedView) {
          outTarget.view = chunk.mappedView.subarray(offset, offset + byteSize);
          if (alignedSize > byteSize) {
            chunk.mappedView.fill(0, offset + byteSize, offset + alignedSize);
          }
        } else {
          outTarget.view = undefined;
        }
        if (countZeroAlloc) {
          this._zeroAllocAcquires += 1;
        }
        return outTarget;
      }

      const slice = {
        buffer: chunk.buffer,
        offset,
        size: byteSize,
        alignedSize,
      };
      if (isLargeSlab) {
        slice.isLargeSlab = true;
        slice.slabBucketSize = chunk.slabBucketSize;
      }
      if (chunk.isMapped && chunk.mappedView) {
        slice.view = chunk.mappedView.subarray(offset, offset + byteSize);
        if (alignedSize > byteSize) {
          chunk.mappedView.fill(0, offset + byteSize, offset + alignedSize);
        }
      }
      return slice;
    }

    /**
     * Sub-allocate an aligned staging slice from the current active chunk, rolling
     * over to a recycled or newly allocated chunk (or a persistent large-slab bucket /
     * dedicated oversized chunk) within `maxPoolBytes`.
     *
     * Enforces WebGPU 4-byte minimum alignment for buffer reservation and offsets
     * while preserving `slice.size === byteSize` for backwards compatibility.
     *
     * Pass an optional reusable `out` scratch object (as the 3rd argument or via
     * `{ alignment, align, out }` options) to populate the slice in place with
     * zero wrapper-object allocations on hot 30 FPS upload loops.
     *
     * @param {number} byteSize
     * @param {number | { alignment?: number, align?: number, out?: object | null }} [alignmentOrOptions=4]
     * @param {object | null} [out]
     * @returns {{ buffer: GPUBuffer, offset: number, size: number, alignedSize: number, isLargeSlab?: boolean, slabBucketSize?: number, view?: Uint8Array }}
     */
    allocate(byteSize, alignmentOrOptions = WEBGPU_BUFFER_COPY_ALIGNMENT, out) {
      let alignment = WEBGPU_BUFFER_COPY_ALIGNMENT;
      let outTarget = null;

      if (
        alignmentOrOptions !== undefined &&
        alignmentOrOptions !== null &&
        typeof alignmentOrOptions === 'object' &&
        !Array.isArray(alignmentOrOptions)
      ) {
        const opts = alignmentOrOptions;
        if (
          opts.alignment !== undefined &&
          opts.align !== undefined &&
          opts.alignment !== opts.align
        ) {
          throw new RangeError(
            `Conflicting alignment (${opts.alignment}) and align (${opts.align})`
          );
        }
        if (opts.alignment !== undefined) {
          alignment = opts.alignment;
        } else if (opts.align !== undefined) {
          alignment = opts.align;
        }
        if (opts.out !== undefined && opts.out !== null) {
          if (typeof opts.out !== 'object' || Array.isArray(opts.out)) {
            throw new TypeError('out must be an object');
          }
          outTarget = opts.out;
        }
        if (out !== undefined && out !== null) {
          if (typeof out !== 'object' || Array.isArray(out)) {
            throw new TypeError('out must be an object');
          }
          if (outTarget !== null && outTarget !== out) {
            throw new RangeError('Conflicting out scratch targets');
          }
          outTarget = out;
        }
      } else {
        if (alignmentOrOptions !== undefined) {
          alignment = alignmentOrOptions;
        }
        if (out !== undefined && out !== null) {
          if (typeof out !== 'object' || Array.isArray(out)) {
            throw new TypeError('out must be an object');
          }
          outTarget = out;
        }
      }

      return this._allocateInternal(byteSize, alignment, outTarget, outTarget !== null);
    }

    /**
     * Acquire a staging slice, supporting both positional `(size, align, out)` and
     * options `(size, { alignment, align, out })` signatures for zero-alloc reuse.
     */
    acquire(byteSize, alignmentOrOptions = WEBGPU_BUFFER_COPY_ALIGNMENT, out) {
      return this.allocate(byteSize, alignmentOrOptions, out);
    }

    /**
     * Acquire an aligned staging slice into an optional reusable `out` scratch target.
     */
    acquireSlice(byteSize, alignmentOrOptions = WEBGPU_BUFFER_COPY_ALIGNMENT, out) {
      return this.allocate(byteSize, alignmentOrOptions, out);
    }

    /**
     * Acquire an aligned upload staging slice into an optional reusable `out` scratch target.
     */
    acquireUploadSlice(byteSize, alignmentOrOptions = WEBGPU_BUFFER_COPY_ALIGNMENT, out) {
      return this.allocate(byteSize, alignmentOrOptions, out);
    }

    _acquireUploadScratch(requiredBytes) {
      if (requiredBytes <= 0) {
        return null;
      }
      if (this._uploadScratch && this._uploadScratch.byteLength >= requiredBytes) {
        return this._uploadScratch;
      }
      if (requiredBytes > this.maxScratchBytes) {
        return null;
      }
      const allocBytes = this._uploadScratch
        ? Math.min(this.maxScratchBytes, Math.max(requiredBytes, this._uploadScratch.byteLength * 2))
        : requiredBytes;
      this._uploadScratch = new Uint8Array(allocBytes);
      return this._uploadScratch;
    }

    _writeBufferChunked(dstBuffer, dstOffset, bytes) {
      const len = bytes.byteLength;
      if (len === 0) {
        return false;
      }
      const chunkMax = this.maxWriteChunkBytes;
      if (len <= chunkMax) {
        this.device.queue.writeBuffer(dstBuffer, dstOffset, bytes);
        return false;
      }
      for (let off = 0; off < len; off += chunkMax) {
        const chunkLen = Math.min(chunkMax, len - off);
        this.device.queue.writeBuffer(dstBuffer, dstOffset + off, bytes.subarray(off, off + chunkLen));
      }
      this._chunkedWrites += 1;
      return true;
    }

    /**
     * Stage `srcBytes` into the pool (with 4-byte aligned length per WebGPU
     * `copyBufferToBuffer` requirements and SharedArrayBuffer-safe copies) and
     * record `encoder.copyBufferToBuffer(...)`.
     *
     * Uploads larger than `maxWriteChunkBytes` (default 4 MiB) are automatically
     * split across bounded staging slices & chunked `queue.writeBuffer` calls so
     * Dawn's internal upload ring and pool chunks never spike to multi-hundred-MiB
     * allocations.
     *
     * Supports both `device.queue.writeBuffer` staging-ring mode and mapped
     * `copyBufferToBuffer` mode transparently. Pass an optional `out` scratch
     * target (or `{ out }`) to avoid per-upload return wrapper allocations.
     *
     * @param {GPUCommandEncoder} encoder
     * @param {GPUBuffer} dstBuffer
     * @param {number} dstOffset
     * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
     * @param {object | { out?: object | null } | null} [outOrOptions]
     */
    stageBufferUpload(encoder, dstBuffer, dstOffset, srcBytes, outOrOptions) {
      this._assertNotDestroyed();
      if (!encoder || typeof encoder.copyBufferToBuffer !== 'function') {
        throw new TypeError('stageBufferUpload requires an encoder with copyBufferToBuffer()');
      }
      if (!dstBuffer || typeof dstBuffer !== 'object') {
        throw new TypeError('stageBufferUpload requires a destination GPUBuffer');
      }
      assertNonNegativeInteger(dstOffset, 'dstOffset');
      if (dstOffset % WEBGPU_BUFFER_COPY_ALIGNMENT !== 0) {
        throw new RangeError(
          `dstOffset (${dstOffset}) must be a multiple of ${WEBGPU_BUFFER_COPY_ALIGNMENT}`
        );
      }

      let outTarget = null;
      if (outOrOptions !== undefined && outOrOptions !== null) {
        if (typeof outOrOptions !== 'object' || Array.isArray(outOrOptions)) {
          throw new TypeError('out must be an object');
        }
        if (Object.prototype.hasOwnProperty.call(outOrOptions, 'out')) {
          if (outOrOptions.out !== undefined && outOrOptions.out !== null) {
            if (typeof outOrOptions.out !== 'object' || Array.isArray(outOrOptions.out)) {
              throw new TypeError('out must be an object');
            }
            outTarget = outOrOptions.out;
          }
        } else {
          outTarget = outOrOptions;
        }
      }

      const src = toUint8Array(srcBytes, 'stageBufferUpload');
      if (src.byteLength === 0) {
        if (outTarget !== null) {
          outTarget.buffer = null;
          outTarget.offset = 0;
          outTarget.size = 0;
          outTarget.srcByteLength = 0;
          return outTarget;
        }
        return {
          buffer: null,
          offset: 0,
          size: 0,
          srcByteLength: 0,
        };
      }

      const copySize = alignBufferSize(src.byteLength);
      const needsPad = copySize !== src.byteLength;
      const fromSab = isSharedBuffer(src.buffer);
      const chunkMax = this.maxWriteChunkBytes;

      if (
        this.mode !== STAGING_MODE.MAPPED &&
        (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function')
      ) {
        throw new TypeError(
          'stageBufferUpload in writeBuffer mode requires device.queue.writeBuffer()'
        );
      }

      let firstBuffer = null;
      let firstOffset = 0;

      if (copySize > chunkMax) {
        for (let off = 0; off < src.byteLength; off += chunkMax) {
          const rawChunkLen = Math.min(chunkMax, src.byteLength - off);
          const alignedChunkLen = alignBufferSize(rawChunkLen);
          const chunkNeedsPad = alignedChunkLen !== rawChunkLen;
          const slice = this._allocateInternal(
            alignedChunkLen,
            WEBGPU_BUFFER_COPY_ALIGNMENT,
            this._internalSliceScratch,
            false
          );
          const sliceBuffer = slice.buffer;
          const sliceOffset = slice.offset;
          const sliceView = slice.view;
          this._internalSliceScratch.buffer = null;
          this._internalSliceScratch.view = undefined;

          if (firstBuffer === null) {
            firstBuffer = sliceBuffer;
            firstOffset = sliceOffset;
          }

          const srcChunk = src.subarray(off, off + rawChunkLen);
          if (sliceView) {
            sliceView.set(srcChunk, 0);
            if (chunkNeedsPad) {
              sliceView.fill(0, rawChunkLen, alignedChunkLen);
            }
          } else if (!fromSab && !chunkNeedsPad) {
            this.device.queue.writeBuffer(sliceBuffer, sliceOffset, srcChunk);
          } else {
            const scratch = this._acquireUploadScratch(alignedChunkLen);
            const dst = scratch
              ? scratch.byteLength === alignedChunkLen
                ? scratch
                : scratch.subarray(0, alignedChunkLen)
              : new Uint8Array(alignedChunkLen);
            dst.set(srcChunk, 0);
            if (chunkNeedsPad) {
              dst.fill(0, rawChunkLen, alignedChunkLen);
            }
            this.device.queue.writeBuffer(sliceBuffer, sliceOffset, dst);
          }

          encoder.copyBufferToBuffer(sliceBuffer, sliceOffset, dstBuffer, dstOffset + off, alignedChunkLen);
        }

        if (needsPad) {
          this._alignPadCount += 1;
        }
        if (fromSab && this.mode !== STAGING_MODE.MAPPED) {
          this._sabCopyCount += 1;
        }
        this._chunkedWrites += 1;
      } else {
        const slice = this._allocateInternal(
          copySize,
          WEBGPU_BUFFER_COPY_ALIGNMENT,
          this._internalSliceScratch,
          false
        );
        firstBuffer = slice.buffer;
        firstOffset = slice.offset;
        const sliceView = slice.view;
        this._internalSliceScratch.buffer = null;
        this._internalSliceScratch.view = undefined;

        if (sliceView) {
          sliceView.set(src, 0);
          if (needsPad) {
            sliceView.fill(0, src.byteLength, copySize);
            this._alignPadCount += 1;
          }
        } else {
          if (!fromSab && !needsPad) {
            this.device.queue.writeBuffer(firstBuffer, firstOffset, src);
          } else {
            const scratch = this._acquireUploadScratch(copySize);
            const dst = scratch
              ? scratch.byteLength === copySize
                ? scratch
                : scratch.subarray(0, copySize)
              : new Uint8Array(copySize);
            dst.set(src, 0);
            if (needsPad) {
              dst.fill(0, src.byteLength, copySize);
              this._alignPadCount += 1;
            }
            if (fromSab) {
              this._sabCopyCount += 1;
            }
            this.device.queue.writeBuffer(firstBuffer, firstOffset, dst);
          }
        }

        encoder.copyBufferToBuffer(firstBuffer, firstOffset, dstBuffer, dstOffset, copySize);
      }

      this._bytesUploaded += copySize;
      this._frameBytesUploaded += copySize;
      this._recordActivityTime();

      if (outTarget !== null) {
        this._zeroAllocAcquires += 1;
        outTarget.buffer = firstBuffer;
        outTarget.offset = firstOffset;
        outTarget.size = copySize;
        outTarget.srcByteLength = src.byteLength;
        return outTarget;
      }

      return {
        buffer: firstBuffer,
        offset: firstOffset,
        size: copySize,
        srcByteLength: src.byteLength,
      };
    }

    /**
     * Stage texture texel/block rows into the pool, enforcing WebGPU's 256-byte
     * `bytesPerRow` alignment, and record `encoder.copyBufferToTexture(...)`.
     *
     * Reuses a bounded internal scratch buffer and pre-allocated copy descriptors
     * for zero per-frame temporary allocations on steady-state uploads, and chunks
     * `queue.writeBuffer` calls at `maxWriteChunkBytes` (default 4 MiB) so Dawn's
     * internal upload ring never spikes on 4K/8K textures.
     *
     * Pass an optional `out` scratch target (via `layout.out` or 5th argument) to
     * avoid per-upload return wrapper allocations.
     *
     * @param {GPUCommandEncoder} encoder
     * @param {GPUTexture} dstTexture
     * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
     * @param {{ width: number, height: number, depthOrLayers?: number, depthOrArrayLayers?: number, bytesPerRow?: number, rowsPerImage?: number, mipLevel?: number, origin?: { x?: number, y?: number, z?: number } | number[], aspect?: string, out?: object | null }} layout
     * @param {object | { out?: object | null } | null} [outOrOptions]
     */
    stageTextureUpload(encoder, dstTexture, srcBytes, layout, outOrOptions) {
      this._assertNotDestroyed();
      if (!encoder || typeof encoder.copyBufferToTexture !== 'function') {
        throw new TypeError('stageTextureUpload requires an encoder with copyBufferToTexture()');
      }
      if (!dstTexture || typeof dstTexture !== 'object') {
        throw new TypeError('stageTextureUpload requires a destination GPUTexture');
      }
      if (!layout || typeof layout !== 'object') {
        throw new TypeError('stageTextureUpload requires a layout options object');
      }

      let outTarget = null;
      if (layout.out !== undefined && layout.out !== null) {
        if (typeof layout.out !== 'object' || Array.isArray(layout.out)) {
          throw new TypeError('out must be an object');
        }
        outTarget = layout.out;
      }
      if (outOrOptions !== undefined && outOrOptions !== null) {
        if (typeof outOrOptions !== 'object' || Array.isArray(outOrOptions)) {
          throw new TypeError('out must be an object');
        }
        let argOut = null;
        if (Object.prototype.hasOwnProperty.call(outOrOptions, 'out')) {
          if (outOrOptions.out !== undefined && outOrOptions.out !== null) {
            if (typeof outOrOptions.out !== 'object' || Array.isArray(outOrOptions.out)) {
              throw new TypeError('out must be an object');
            }
            argOut = outOrOptions.out;
          }
        } else {
          argOut = outOrOptions;
        }
        if (argOut !== null) {
          if (outTarget !== null && outTarget !== argOut) {
            throw new RangeError('Conflicting out scratch targets');
          }
          outTarget = argOut;
        }
      }

      const src = toUint8Array(srcBytes, 'stageTextureUpload');
      const { width, height, mipLevel = 0, origin, aspect } = layout;
      const depthOrLayers = layout.depthOrLayers ?? layout.depthOrArrayLayers ?? 1;
      const rowsPerImage = layout.rowsPerImage ?? height;

      assertPositiveInteger(width, 'width');
      assertPositiveInteger(height, 'height');
      assertPositiveInteger(depthOrLayers, 'depthOrLayers');
      assertPositiveInteger(rowsPerImage, 'rowsPerImage');
      assertNonNegativeInteger(mipLevel, 'mipLevel');
      const normOrigin = fillNormalizedOriginInto(this._texCopyOrigin, origin);

      const totalRows = rowsPerImage * depthOrLayers;
      let unalignedBytesPerRow = layout.bytesPerRow;
      if (unalignedBytesPerRow === undefined) {
        if (src.byteLength > 0 && src.byteLength % totalRows === 0) {
          unalignedBytesPerRow = src.byteLength / totalRows;
        } else {
          throw new RangeError(
            'bytesPerRow is required when srcBytes.byteLength is not evenly divisible by rowsPerImage * depthOrLayers'
          );
        }
      }
      assertPositiveInteger(unalignedBytesPerRow, 'bytesPerRow');

      const requiredSrcBytes = unalignedBytesPerRow * totalRows;
      if (src.byteLength < requiredSrcBytes) {
        throw new RangeError(
          `srcBytes too small (${src.byteLength} bytes; need at least ${requiredSrcBytes} bytes ` +
            `for ${totalRows} rows of ${unalignedBytesPerRow} bytes)`
        );
      }

      const alignedBytesPerRow = alignUp(unalignedBytesPerRow, WEBGPU_BYTES_PER_ROW_ALIGNMENT);
      const stagedBytes = alignedBytesPerRow * totalRows;
      const slice = this._allocateInternal(
        stagedBytes,
        WEBGPU_BYTES_PER_ROW_ALIGNMENT,
        this._internalSliceScratch,
        false
      );
      const sliceBuffer = slice.buffer;
      const sliceOffset = slice.offset;
      const sliceView = slice.view;
      this._internalSliceScratch.buffer = null;
      this._internalSliceScratch.view = undefined;

      if (sliceView) {
        if (alignedBytesPerRow === unalignedBytesPerRow) {
          sliceView.set(src.byteLength === stagedBytes ? src : src.subarray(0, stagedBytes), 0);
        } else {
          copyTextureRowsPadded(
            src,
            sliceView,
            0,
            totalRows,
            unalignedBytesPerRow,
            alignedBytesPerRow,
            true
          );
        }
      } else {
        if (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function') {
          throw new TypeError(
            'stageTextureUpload in writeBuffer mode requires device.queue.writeBuffer()'
          );
        }
        const fromSab = isSharedBuffer(src.buffer);
        const chunkMax = this.maxWriteChunkBytes;
        if (alignedBytesPerRow === unalignedBytesPerRow) {
          if (!fromSab) {
            const payload = src.byteLength === stagedBytes ? src : src.subarray(0, stagedBytes);
            this._writeBufferChunked(sliceBuffer, sliceOffset, payload);
          } else if (stagedBytes <= chunkMax) {
            const scratch = this._acquireUploadScratch(stagedBytes);
            const payload = scratch
              ? scratch.byteLength === stagedBytes
                ? scratch
                : scratch.subarray(0, stagedBytes)
              : new Uint8Array(stagedBytes);
            payload.set(src.byteLength === stagedBytes ? src : src.subarray(0, stagedBytes), 0);
            this.device.queue.writeBuffer(sliceBuffer, sliceOffset, payload);
          } else {
            const scratch = this._acquireUploadScratch(chunkMax);
            for (let off = 0; off < stagedBytes; off += chunkMax) {
              const len = Math.min(chunkMax, stagedBytes - off);
              const payload = scratch
                ? scratch.byteLength === len
                  ? scratch
                  : scratch.subarray(0, len)
                : new Uint8Array(len);
              payload.set(src.subarray(off, off + len), 0);
              this.device.queue.writeBuffer(sliceBuffer, sliceOffset + off, payload);
            }
            this._chunkedWrites += 1;
          }
        } else {
          const rowsPerChunk = Math.max(1, Math.floor(chunkMax / alignedBytesPerRow));
          if (rowsPerChunk >= totalRows) {
            const scratch = this._acquireUploadScratch(stagedBytes);
            const payload = scratch
              ? scratch.byteLength === stagedBytes
                ? scratch
                : scratch.subarray(0, stagedBytes)
              : new Uint8Array(stagedBytes);
            const mustZeroPad = scratch !== null;
            copyTextureRowsPadded(
              src,
              payload,
              0,
              totalRows,
              unalignedBytesPerRow,
              alignedBytesPerRow,
              mustZeroPad
            );
            this._writeBufferChunked(sliceBuffer, sliceOffset, payload);
          } else {
            const maxStripBytes = rowsPerChunk * alignedBytesPerRow;
            const scratch = this._acquireUploadScratch(maxStripBytes);
            const mustZeroPad = scratch !== null;
            for (let rowStart = 0; rowStart < totalRows; rowStart += rowsPerChunk) {
              const count = Math.min(rowsPerChunk, totalRows - rowStart);
              const stripBytes = count * alignedBytesPerRow;
              const dst = scratch
                ? scratch.byteLength === stripBytes
                  ? scratch
                  : scratch.subarray(0, stripBytes)
                : new Uint8Array(stripBytes);
              copyTextureRowsPadded(
                src,
                dst,
                rowStart,
                count,
                unalignedBytesPerRow,
                alignedBytesPerRow,
                mustZeroPad
              );
              this.device.queue.writeBuffer(
                sliceBuffer,
                sliceOffset + rowStart * alignedBytesPerRow,
                dst
              );
            }
            this._chunkedWrites += 1;
          }
        }
        if (fromSab) {
          this._sabCopyCount += 1;
        }
      }

      const source = this._texCopySource;
      source.buffer = sliceBuffer;
      source.offset = sliceOffset;
      source.bytesPerRow = alignedBytesPerRow;
      source.rowsPerImage = rowsPerImage;

      let destination;
      if (aspect !== undefined) {
        destination = this._texCopyDestWithAspect;
        destination.texture = dstTexture;
        destination.mipLevel = mipLevel;
        destination.origin = normOrigin;
        destination.aspect = aspect;
      } else {
        destination = this._texCopyDest;
        destination.texture = dstTexture;
        destination.mipLevel = mipLevel;
        destination.origin = normOrigin;
      }

      const copySize = this._texCopySize;
      copySize.width = width;
      copySize.height = height;
      copySize.depthOrArrayLayers = depthOrLayers;

      encoder.copyBufferToTexture(source, destination, copySize);
      source.buffer = null;
      destination.texture = null;

      this._bytesUploaded += stagedBytes;
      this._frameBytesUploaded += stagedBytes;
      this._recordActivityTime();

      if (outTarget !== null) {
        this._zeroAllocAcquires += 1;
        outTarget.buffer = sliceBuffer;
        outTarget.offset = sliceOffset;
        outTarget.size = stagedBytes;
        outTarget.bytesPerRow = alignedBytesPerRow;
        outTarget.unalignedBytesPerRow = unalignedBytesPerRow;
        outTarget.rowsPerImage = rowsPerImage;
        outTarget.width = width;
        outTarget.height = height;
        outTarget.depthOrArrayLayers = depthOrLayers;
        return outTarget;
      }

      return {
        buffer: sliceBuffer,
        offset: sliceOffset,
        size: stagedBytes,
        bytesPerRow: alignedBytesPerRow,
        unalignedBytesPerRow,
        rowsPerImage,
        width,
        height,
        depthOrArrayLayers: depthOrLayers,
      };
    }

    /**
     * Alias for `stageBufferUpload`.
     */
    writeBuffer(encoder, dstBuffer, dstOffset, srcBytes, outOrOptions) {
      return this.stageBufferUpload(encoder, dstBuffer, dstOffset, srcBytes, outOrOptions);
    }

    /**
     * Alias for `stageTextureUpload`.
     */
    writeTexture(encoder, dstTexture, srcBytes, layout, outOrOptions) {
      return this.stageTextureUpload(encoder, dstTexture, srcBytes, layout, outOrOptions);
    }

    /**
     * Release the reusable CPU upload scratch buffer (`_uploadScratch`) to reclaim
     * transient JS heap memory during idle periods or scene transitions.
     *
     * @returns {number} Bytes released from CPU scratch.
     */
    trimScratch() {
      const released =
        this._uploadScratch && typeof this._uploadScratch.byteLength === 'number'
          ? this._uploadScratch.byteLength
          : 0;
      this._uploadScratch = null;
      return released;
    }

    /**
     * Return concise live GPU & CPU staging memory byte accounting (#1077).
     */
    getMemoryStats() {
      let activeBytes = 0;
      for (let i = 0; i < this._activeChunks.length; i += 1) {
        activeBytes += this._activeChunks[i].size;
      }
      let inFlightBytes = 0;
      const ifQ = this._inFlightQueue;
      if (ifQ.size > 0) {
        const buf = ifQ._buf;
        const mask = ifQ._mask;
        let idx = ifQ._head;
        for (let i = 0; i < ifQ._span; i += 1) {
          const chunk = buf[idx];
          idx = (idx + 1) & mask;
          if (chunk !== undefined) {
            inFlightBytes += chunk.size;
          }
        }
      }
      const recycledBytes = sanitizeNonNegativeCounter(this._recycledBytes);
      const scratchBytes =
        this._uploadScratch && typeof this._uploadScratch.byteLength === 'number'
          ? this._uploadScratch.byteLength
          : 0;
      const allocatedBytes = sanitizeNonNegativeCounter(this._allocatedBytes);
      return {
        stagingPoolBytes: allocatedBytes,
        allocatedBytes,
        peakAllocatedBytes: sanitizeNonNegativeCounter(this._peakAllocatedBytes),
        activeBytes,
        inFlightBytes,
        recycledBytes,
        largeSlabBytes: sanitizeNonNegativeCounter(this._largeSlabBytes),
        scratchBytes,
        maxPoolBytes: this.maxPoolBytes,
      };
    }

    /**
     * Return pool-wide allocation, large-slab reuse, and upload bandwidth statistics.
     */
    getStats() {
      const activeCount = this._activeChunks.length;
      const recycledCount = this._getRecycledCount();
      let activeLargeSlabs = 0;
      let activeBytes = 0;
      for (let i = 0; i < this._activeChunks.length; i += 1) {
        activeBytes += this._activeChunks[i].size;
        if (this._activeChunks[i].isLargeSlab) {
          activeLargeSlabs += 1;
        }
      }
      let inFlightLargeSlabs = 0;
      let inFlightBytes = 0;
      const ifQ = this._inFlightQueue;
      if (ifQ.size > 0) {
        const buf = ifQ._buf;
        const mask = ifQ._mask;
        let idx = ifQ._head;
        for (let i = 0; i < ifQ._span; i += 1) {
          const chunk = buf[idx];
          idx = (idx + 1) & mask;
          if (chunk !== undefined) {
            inFlightBytes += chunk.size;
            if (chunk.isLargeSlab) {
              inFlightLargeSlabs += 1;
            }
          }
        }
      }
      const recycledBytes = sanitizeNonNegativeCounter(this._recycledBytes);
      const largeSlabBucketCounts = {};
      for (const [bucketSize, bucket] of this._largeSlabBuckets.entries()) {
        if (bucket.size > 0) {
          largeSlabBucketCounts[bucketSize] = bucket.size;
        }
      }
      const scratchBytes =
        this._uploadScratch && typeof this._uploadScratch.byteLength === 'number'
          ? this._uploadScratch.byteLength
          : 0;

      const poolHits = sanitizeNonNegativeCounter(this._bufferReuseHits);
      const poolMisses = sanitizeNonNegativeCounter(this._bufferReuseMisses);
      const poolLookups = poolHits + poolMisses;
      const hitRate = clampUnitRate(poolHits, poolLookups);
      const largeSlabHits = sanitizeNonNegativeCounter(this._largeSlabHits);
      const largeSlabMisses = sanitizeNonNegativeCounter(this._largeSlabMisses);
      const largeSlabTotal = largeSlabHits + largeSlabMisses;
      const peakAllocatedBytes = sanitizeNonNegativeCounter(this._peakAllocatedBytes);
      const peakPoolUtilization = clampUnitRate(peakAllocatedBytes, this.maxPoolBytes);
      const lruEvictions = sanitizeNonNegativeCounter(this._evictedCount);
      const oversizedAllocations = sanitizeNonNegativeCounter(this._oversizeAllocations);
      const totalBytesStaged = sanitizeNonNegativeCounter(this._totalBytesStaged);
      const bytesUploaded = sanitizeNonNegativeCounter(this._bytesUploaded);
      const allocationCount = sanitizeNonNegativeCounter(this._allocationCount);
      const reusedAllocations = sanitizeNonNegativeCounter(this._reusedAllocations);
      const freshAllocations = sanitizeNonNegativeCounter(this._freshAllocations);
      const elapsedMs =
        this._firstActivityTimeMs !== null && this._lastActivityTimeMs !== null
          ? Math.max(0, this._lastActivityTimeMs - this._firstActivityTimeMs)
          : 0;
      const uploadBytesPerSec =
        elapsedMs > 0 && bytesUploaded > 0 ? (bytesUploaded * 1000) / elapsedMs : 0;

      return {
        totalChunksAllocated: sanitizeNonNegativeCounter(this._totalChunksAllocated),
        activeChunks: activeCount,
        recycledChunks: recycledCount,
        inFlightChunks: this._inFlightQueue.size,
        activeBuffers: activeCount,
        recycledBuffers: recycledCount,
        activeCount,
        idleCount: recycledCount,
        allocatedBytes: sanitizeNonNegativeCounter(this._allocatedBytes),
        stagingPoolBytes: sanitizeNonNegativeCounter(this._allocatedBytes),
        activeBytes,
        inFlightBytes,
        recycledBytes,
        scratchBytes,
        bytesAllocated: sanitizeNonNegativeCounter(this._totalBytesAllocated),
        bytesReused: sanitizeNonNegativeCounter(this._bytesReused),
        peakBytes: peakAllocatedBytes,
        peakAllocatedBytes,
        peakBytesAllocated: peakAllocatedBytes,
        peakAllocationBytes: peakAllocatedBytes,
        peakSingleAllocationBytes: sanitizeNonNegativeCounter(this._peakSingleAllocationBytes),
        peakPoolUtilization,
        maxPoolBytes: this.maxPoolBytes,
        maxPooledBytes: this.maxPoolBytes,
        reclaimedChunks: sanitizeNonNegativeCounter(this._reclaimedChunks),
        reclaimedBytes: sanitizeNonNegativeCounter(this._reclaimedBytes),
        evictedCount: lruEvictions,
        evictedBytes: sanitizeNonNegativeCounter(this._evictedBytes),
        lruEvictions,
        oversizeAllocations: oversizedAllocations,
        oversizedAllocations,
        bytesUploaded,
        totalBytesUploaded: bytesUploaded,
        totalBytesStaged,
        allocationCount,
        poolHits,
        poolMisses,
        poolHitRate: hitRate,
        hitRate,
        bufferReuseHits: poolHits,
        bufferReuseMisses: poolMisses,
        reusedAllocations,
        freshAllocations,
        reuseRate: clampUnitRate(reusedAllocations, allocationCount),
        largeSlabHits,
        largeSlabMisses,
        largeSlabEvictions: sanitizeNonNegativeCounter(this._largeSlabEvictions),
        largeSlabReuseRate: clampUnitRate(largeSlabHits, largeSlabTotal),
        activeLargeSlabs,
        inFlightLargeSlabs,
        idleLargeSlabs: sanitizeNonNegativeCounter(this._idleLargeSlabCount),
        largeSlabBytes: sanitizeNonNegativeCounter(this._largeSlabBytes),
        maxLargeSlabs: this.maxLargeSlabs,
        maxSlabsPerBucket: this.maxSlabsPerBucket,
        bucketStrategy: this.bucketStrategy,
        maxBucketBytes: this.maxBucketBytes,
        largeSlabBucketCounts,
        framesCompleted: sanitizeNonNegativeCounter(this._framesCompleted),
        lastFrameBytesUploaded: sanitizeNonNegativeCounter(this._lastFrameBytesUploaded),
        peakFrameBytesUploaded: sanitizeNonNegativeCounter(this._peakFrameBytesUploaded),
        avgBytesPerFrame:
          this._framesCompleted > 0 ? bytesUploaded / this._framesCompleted : 0,
        uploadBytesPerSec,
        uploadMiBPerSec: uploadBytesPerSec / (1024 * 1024),
        alignPadCount: sanitizeNonNegativeCounter(this._alignPadCount),
        sabCopyCount: sanitizeNonNegativeCounter(this._sabCopyCount),
        doubleReleaseIgnored: sanitizeNonNegativeCounter(this._doubleReleaseIgnored),
        maxFramesInFlight: this.maxFramesInFlight,
        idleShrinkFrames: this.idleShrinkFrames,
        subAllocCoalesced: sanitizeNonNegativeCounter(this._subAllocCoalesced),
        idleSlabShrinks: sanitizeNonNegativeCounter(this._idleSlabShrinks),
        stallAvoidedCount: sanitizeNonNegativeCounter(this._stallAvoidedCount),
        zeroAllocAcquires: sanitizeNonNegativeCounter(this._zeroAllocAcquires),
        sceneTransitionTrimBytes: sanitizeNonNegativeCounter(this._sceneTransitionTrimBytes),
        chunkedWrites: sanitizeNonNegativeCounter(this._chunkedWrites),
        trimmedStagingBytes: sanitizeNonNegativeCounter(this._trimmedStagingBytes),
        destroyedBuffers: sanitizeNonNegativeCounter(this._destroyedBuffers),
        maxWriteChunkBytes: this.maxWriteChunkBytes,
        destroyed: this._destroyed,
      };
    }

    /**
     * Reset cumulative telemetry counters while preserving live pooled GPUBuffers,
     * active/in-flight/recycled queues, and frame/LRU ordering state.
     * Re-baselines `peakAllocatedBytes` (and `peakPoolUtilization`) to the current
     * live `allocatedBytes` so `peakAllocatedBytes >= allocatedBytes` always holds.
     */
    resetStats() {
      this._totalChunksAllocated = 0;
      this._totalBytesAllocated = 0;
      this._peakAllocatedBytes = this._allocatedBytes;
      this._peakSingleAllocationBytes = 0;
      this._reclaimedChunks = 0;
      this._reclaimedBytes = 0;
      this._evictedCount = 0;
      this._evictedBytes = 0;
      this._oversizeAllocations = 0;
      this._bytesUploaded = 0;
      this._totalBytesStaged = 0;
      this._bytesReused = 0;
      this._allocationCount = 0;
      this._bufferReuseHits = 0;
      this._bufferReuseMisses = 0;
      this._largeSlabHits = 0;
      this._largeSlabMisses = 0;
      this._largeSlabEvictions = 0;
      this._reusedAllocations = 0;
      this._freshAllocations = 0;
      this._alignPadCount = 0;
      this._sabCopyCount = 0;
      this._doubleReleaseIgnored = 0;
      this._subAllocCoalesced = 0;
      this._idleSlabShrinks = 0;
      this._stallAvoidedCount = 0;
      this._zeroAllocAcquires = 0;
      this._sceneTransitionTrimBytes = 0;
      this._chunkedWrites = 0;
      this._trimmedStagingBytes = 0;
      this._destroyedBuffers = 0;
      this._framesCompleted = 0;
      this._lastFrameBytesUploaded = 0;
      this._peakFrameBytesUploaded = 0;
      this._firstActivityTimeMs = null;
      this._lastActivityTimeMs = null;
      this._frameAllocations = 0;
      this._frameBytesAllocated = 0;
      this._frameBytesUploaded = 0;
      return this.getStats();
    }

    /**
     * Destroy all allocated GPUBuffers idempotently and release pool references.
     */
    destroy() {
      if (this._destroyed) {
        this._doubleReleaseIgnored += 1;
        return;
      }
      this._destroyed = true;

      for (const chunk of Array.from(this._allChunks)) {
        this._destroyChunk(chunk);
      }

      this._allChunks.clear();
      this._chunkByBuffer.clear();
      this._largeSlabBuckets.clear();
      this._idleLargeSlabCount = 0;
      this._recycledBytes = 0;
      this._largeSlabBytes = 0;
      this._pendingMapPromises.clear();
      this._activeChunks.length = 0;
      this._recycledStdQueue.clear();
      this._recycledOversizedChunks.length = 0;
      this._inFlightQueue.clear();
      this._gpuDoneInFlightCount = 0;
      this._currentChunk = null;
      this._uploadScratch = null;
      this._internalSliceScratch.buffer = null;
      this._internalSliceScratch.view = undefined;
      this._texCopySource.buffer = null;
      this._texCopyDest.texture = null;
      this._texCopyDestWithAspect.texture = null;
      this._allocatedBytes = 0;
      this._inFrame = false;
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_WRITE_CHUNK_BYTES,
    DEFAULT_MAX_SCRATCH_BYTES,
    DEFAULT_MAX_POOL_BYTES,
    DEFAULT_FRAMES_IN_FLIGHT,
    DEFAULT_MAX_FRAMES_IN_FLIGHT,
    DEFAULT_MIN_WARM_CHUNKS,
    DEFAULT_IDLE_EVICT_FRAMES,
    DEFAULT_IDLE_SHRINK_FRAMES,
    DEFAULT_LARGE_SLAB_THRESHOLD,
    DEFAULT_MAX_LARGE_SLABS,
    DEFAULT_POW2_MIN_SLAB_BYTES,
    DEFAULT_POW2_MAX_BUCKET_BYTES,
    DEFAULT_LARGE_SLAB_BUCKETS,
    POW2_LARGE_SLAB_BUCKETS,
    BUCKET_STRATEGY,
    WEBGPU_BUFFER_COPY_ALIGNMENT,
    WEBGPU_BYTES_PER_ROW_ALIGNMENT,
    BUFFER_USAGE,
    MAP_MODE,
    STAGING_MODE,
    alignUp,
    alignBufferSize,
    nextPowerOfTwo,
    buildPowerOfTwoBuckets,
    selectPowerOfTwoBucket,
    selectLargeSlabBucket,
    isSharedBuffer,
    sanitizeWriteBufferPayload,
    GpuStagingPool,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6StagingPool = exportsObj;
  } else if (root) {
    root.PlayGta6StagingPool = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
