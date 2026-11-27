'use strict';

// 256-byte-aligned WebGPU dynamic uniform buffer ring & per-frame single-flush draw constant batcher.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6UniformRing)
// and Node.js (module.exports). Aligns cleanly with web/workers/wgpu_worker.js and
// web/workers/staging_pool.js.

(function (root) {
  const DEFAULT_UNIFORM_ALIGNMENT = 256;
  const DEFAULT_MIN_UNIFORM_BINDING_SIZE = 16;
  const DEFAULT_BYTE_SIZE = 1024 * 1024; // 1 MiB
  const DEFAULT_MAX_BYTE_SIZE = 16 * 1024 * 1024; // 16 MiB
  const MIN_FRAMES_IN_FLIGHT = 1;
  const MAX_FRAMES_IN_FLIGHT = 4;
  const DEFAULT_FRAMES_IN_FLIGHT = 3;
  const DEFAULT_MAX_BINDING_SIZE = 65536; // 64 KiB WebGPU default maxUniformBufferBindingSize
  const DEFAULT_SHRINK_AFTER_IDLE_FRAMES = 120;
  const DEFAULT_SHRINK_THRESHOLD_RATIO = 0.25;
  const WEBGPU_BUFFER_COPY_ALIGNMENT = 4;
  const MAX_DIRTY_RANGE_POOL_SIZE = 32;
  const EMPTY_UINT8 = new Uint8Array(0);
  const IS_LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

  // WebGPU GPUBufferUsage bitmask values with headless/Node fallbacks.
  const BUFFER_USAGE = Object.freeze({
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
    UNIFORM:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.UNIFORM === 'number'
        ? GPUBufferUsage.UNIFORM
        : 0x0040,
  });

  // WebGPU GPUShaderStage bitmask values with headless/Node fallbacks.
  const SHADER_STAGE = Object.freeze({
    VERTEX:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.VERTEX === 'number'
        ? GPUShaderStage.VERTEX
        : 0x0001,
    FRAGMENT:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.FRAGMENT === 'number'
        ? GPUShaderStage.FRAGMENT
        : 0x0002,
    COMPUTE:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.COMPUTE === 'number'
        ? GPUShaderStage.COMPUTE
        : 0x0004,
  });

  const DEFAULT_SHADER_VISIBILITY = SHADER_STAGE.VERTEX | SHADER_STAGE.FRAGMENT;

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

  function isPowerOfTwo(value) {
    if (!Number.isInteger(value) || value < 1) {
      return false;
    }
    let v = value;
    while (v > 1) {
      if (v % 2 !== 0) {
        return false;
      }
      v /= 2;
    }
    return true;
  }

  function nextPowerOfTwo(value) {
    let p = WEBGPU_BUFFER_COPY_ALIGNMENT;
    while (p < value) {
      p *= 2;
    }
    return p;
  }

  function isArrayBuffer(value) {
    if (value instanceof ArrayBuffer) {
      return true;
    }
    return Object.prototype.toString.call(value) === '[object ArrayBuffer]';
  }

  function isSharedArrayBuffer(value) {
    if (!value || typeof value !== 'object') {
      return false;
    }
    if (value instanceof ArrayBuffer) {
      return false;
    }
    if (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer) {
      return true;
    }
    return Object.prototype.toString.call(value) === '[object SharedArrayBuffer]';
  }

  /**
   * Align a non-negative integer up to the nearest multiple of `alignment`.
   * Safe across the full IEEE-754 safe integer range.
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
   * Pad a uniform slice byte length to a 4-byte multiple and guarantee at least
   * `minBindingSize` bytes (default 16B) so empty or tiny CBVs satisfy WebGPU
   * bind-group validation.
   *
   * @param {number} byteLength
   * @param {number} [minBindingSize=16]
   * @returns {number}
   */
  function alignUniformSliceSize(byteLength, minBindingSize = DEFAULT_MIN_UNIFORM_BINDING_SIZE) {
    assertNonNegativeInteger(byteLength, 'byteLength');
    assertPositiveInteger(minBindingSize, 'minBindingSize');
    const minAligned = Math.max(
      DEFAULT_MIN_UNIFORM_BINDING_SIZE,
      alignUp(minBindingSize, WEBGPU_BUFFER_COPY_ALIGNMENT)
    );
    const aligned4 = alignUp(byteLength, WEBGPU_BUFFER_COPY_ALIGNMENT);
    return Math.max(minAligned, aligned4);
  }

  function toUint8Array(input, label) {
    if (input instanceof Uint8Array) {
      return input;
    }
    if (ArrayBuffer.isView(input)) {
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    if (isArrayBuffer(input) || isSharedArrayBuffer(input)) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      `${label} expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`
    );
  }

  /**
   * Prepare a byte view for `GPUQueue.writeBuffer`:
   * - Always copies `SharedArrayBuffer`-backed data into a regular `ArrayBuffer`-backed
   *   `Uint8Array` (Chrome WebGPU rejects `SharedArrayBuffer` directly in `writeBuffer`).
   * - Copies non-4-byte-aligned views (odd `byteOffset` or `byteLength`) into a fresh
   *   zero-padded, 4-byte-aligned `Uint8Array`.
   * - Zero-copy fast path when already a 4-byte-aligned regular `ArrayBuffer` `Uint8Array`.
   *
   * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} input
   * @param {number} [minBytes=0]
   * @returns {{ bytes: Uint8Array, copiedFromSab: boolean, copiedForAlignment: boolean }}
   */
  function copyForWriteBuffer(input, minBytes = 0) {
    assertNonNegativeInteger(minBytes, 'minBytes');
    const view = toUint8Array(input, 'copyForWriteBuffer');
    const fromSab = isSharedArrayBuffer(view.buffer);
    const targetLen = Math.max(
      alignUp(view.byteLength, WEBGPU_BUFFER_COPY_ALIGNMENT),
      alignUp(minBytes, WEBGPU_BUFFER_COPY_ALIGNMENT)
    );
    const unalignedOffset = view.byteOffset % WEBGPU_BUFFER_COPY_ALIGNMENT !== 0;
    const unalignedLength = view.byteLength !== targetLen;
    const needsAlignCopy = unalignedOffset || unalignedLength;

    if (fromSab || needsAlignCopy) {
      const copy = new Uint8Array(targetLen);
      if (view.byteLength > 0) {
        copy.set(view, 0);
      }
      return {
        bytes: copy,
        copiedFromSab: fromSab,
        copiedForAlignment: needsAlignCopy,
      };
    }

    return {
      bytes: view,
      copiedFromSab: false,
      copiedForAlignment: false,
    };
  }

  function resolveDeviceUniformAlignment(device) {
    const raw =
      device &&
      device.limits &&
      typeof device.limits.minUniformBufferOffsetAlignment === 'number' &&
      Number.isFinite(device.limits.minUniformBufferOffsetAlignment) &&
      device.limits.minUniformBufferOffsetAlignment > 0
        ? Math.floor(device.limits.minUniformBufferOffsetAlignment)
        : 0;

    if (raw < WEBGPU_BUFFER_COPY_ALIGNMENT) {
      return DEFAULT_UNIFORM_ALIGNMENT;
    }
    return isPowerOfTwo(raw) ? raw : nextPowerOfTwo(raw);
  }

  function resolveDeviceMaxBindingSize(device) {
    const raw =
      device &&
      device.limits &&
      typeof device.limits.maxUniformBufferBindingSize === 'number' &&
      Number.isInteger(device.limits.maxUniformBufferBindingSize) &&
      device.limits.maxUniformBufferBindingSize > 0
        ? device.limits.maxUniformBufferBindingSize
        : 0;
    return raw > 0 ? raw : DEFAULT_MAX_BINDING_SIZE;
  }

  function clampUnitRate(numerator, denominator) {
    if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) {
      return 0;
    }
    const ratio = numerator / denominator;
    if (!Number.isFinite(ratio) || ratio <= 0) {
      return 0;
    }
    return ratio >= 1 ? 1 : ratio;
  }

  function isOptionsBag(value) {
    return (
      value !== null &&
      typeof value === 'object' &&
      ('out' in value ||
        'allowEmpty' in value ||
        'wrap' in value ||
        '_deferImmediateFlush' in value ||
        'allowSubAlignment' in value)
    );
  }

  function validateOutTarget(out, label) {
    if (out === undefined || out === null) {
      return null;
    }
    if (typeof out !== 'object') {
      throw new TypeError(`${label} out target must be an object`);
    }
    return out;
  }

  function acquireDirtyRange(pool, start, end, writeCount) {
    if (pool && pool.length > 0) {
      const entry = pool.pop();
      entry.start = start;
      entry.end = end;
      entry.writeCount = writeCount;
      return entry;
    }
    return { start, end, writeCount };
  }

  function releaseDirtyRanges(ranges, pool) {
    if (pool) {
      for (let i = 0; i < ranges.length; i += 1) {
        if (pool.length < MAX_DIRTY_RANGE_POOL_SIZE) {
          pool.push(ranges[i]);
        }
      }
    }
    ranges.length = 0;
  }

  function copySourceToShadow(
    shadow,
    shadowU32,
    shadowI32,
    shadowF32,
    data,
    size,
    offset
  ) {
    if (size === 0) {
      return;
    }
    if (data instanceof Uint8Array || data instanceof Uint8ClampedArray) {
      shadow.set(data, offset);
      return;
    }
    if (shadowF32 && data instanceof Float32Array && (offset & 3) === 0) {
      shadowF32.set(data, offset >>> 2);
      return;
    }
    if (shadowU32 && data instanceof Uint32Array && (offset & 3) === 0) {
      shadowU32.set(data, offset >>> 2);
      return;
    }
    if (shadowI32 && data instanceof Int32Array && (offset & 3) === 0) {
      shadowI32.set(data, offset >>> 2);
      return;
    }
    if (data instanceof DataView) {
      if (IS_LITTLE_ENDIAN && shadowU32 && (offset & 3) === 0) {
        const words = size >>> 2;
        const dstWord = offset >>> 2;
        for (let i = 0; i < words; i += 1) {
          shadowU32[dstWord + i] = data.getUint32(i << 2, true);
        }
        for (let i = words << 2; i < size; i += 1) {
          shadow[offset + i] = data.getUint8(i);
        }
      } else {
        for (let i = 0; i < size; i += 1) {
          shadow[offset + i] = data.getUint8(i);
        }
      }
      return;
    }
    if (ArrayBuffer.isView(data)) {
      shadow.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
      return;
    }
    shadow.set(new Uint8Array(data), offset);
  }

  /**
   * Insert `[start, end)` into a sorted, disjoint array of `{ start, end, writeCount }`
   * dirty byte ranges, coalescing adjacent/overlapping intervals (and intervals separated
   * by at most `maxGapBytes`).
   *
   * O(1) zero-allocation fast path for sequential appends (the dominant per-frame draw path).
   *
   * @param {Array<{ start: number, end: number, writeCount: number }>} ranges
   * @param {number} start
   * @param {number} end
   * @param {number} [writeCount=1]
   * @param {number} [maxGapBytes=0]
   * @param {Array<{ start: number, end: number, writeCount: number }> | null} [pool=null]
   */
  function insertDirtyRange(ranges, start, end, writeCount = 1, maxGapBytes = 0, pool = null) {
    if (end <= start) {
      return;
    }

    const len = ranges.length;
    if (len === 0) {
      ranges.push(acquireDirtyRange(pool, start, end, writeCount));
      return;
    }

    // O(1) fast path: sequential append or overlap with the last range.
    const last = ranges[len - 1];
    if (start >= last.start) {
      if (start <= last.end + maxGapBytes) {
        if (end > last.end) {
          last.end = end;
        }
        last.writeCount += writeCount;
        return;
      }
      ranges.push(acquireDirtyRange(pool, start, end, writeCount));
      return;
    }

    // General sorted insertion + coalescing for wrap-around or out-of-order staged writes.
    let insertIdx = 0;
    while (insertIdx < len && ranges[insertIdx].end + maxGapBytes < start) {
      insertIdx += 1;
    }

    let mergedStart = start;
    let mergedEnd = end;
    let mergedWrites = writeCount;
    let mergeEndIdx = insertIdx;

    while (mergeEndIdx < len && ranges[mergeEndIdx].start <= mergedEnd + maxGapBytes) {
      const existing = ranges[mergeEndIdx];
      if (existing.start < mergedStart) {
        mergedStart = existing.start;
      }
      if (existing.end > mergedEnd) {
        mergedEnd = existing.end;
      }
      mergedWrites += existing.writeCount;
      mergeEndIdx += 1;
    }

    if (mergeEndIdx === insertIdx) {
      const entry = acquireDirtyRange(pool, mergedStart, mergedEnd, mergedWrites);
      ranges.push(entry);
      for (let i = len; i > insertIdx; i -= 1) {
        ranges[i] = ranges[i - 1];
      }
      ranges[insertIdx] = entry;
    } else {
      const keeper = ranges[insertIdx];
      keeper.start = mergedStart;
      keeper.end = mergedEnd;
      keeper.writeCount = mergedWrites;
      const removeCount = mergeEndIdx - insertIdx - 1;
      if (removeCount > 0) {
        if (pool) {
          for (let i = insertIdx + 1; i < mergeEndIdx; i += 1) {
            if (pool.length < MAX_DIRTY_RANGE_POOL_SIZE) {
              pool.push(ranges[i]);
            }
          }
        }
        const newLen = len - removeCount;
        for (let i = insertIdx + 1; i < newLen; i += 1) {
          ranges[i] = ranges[i + removeCount];
        }
        ranges.length = newLen;
      }
    }
  }

  /**
   * Pure helper: sort and coalesce an array of `{ start, end, writeCount? }` byte ranges,
   * merging adjacent/overlapping intervals (and optionally bridging gaps <= `maxGapBytes`).
   *
   * @param {Array<{ start: number, end: number, writeCount?: number }>} inputRanges
   * @param {number} [maxGapBytes=0]
   * @returns {Array<{ start: number, end: number, byteLength: number, writeCount: number }>}
   */
  function coalesceDirtyRanges(inputRanges, maxGapBytes = 0) {
    if (!Array.isArray(inputRanges)) {
      throw new TypeError('coalesceDirtyRanges expects an array of { start, end } ranges');
    }
    assertNonNegativeInteger(maxGapBytes, 'maxGapBytes');
    const merged = [];
    for (let i = 0; i < inputRanges.length; i += 1) {
      const r = inputRanges[i];
      if (!r || typeof r !== 'object') {
        throw new TypeError(`Dirty range at index ${i} must be an object`);
      }
      assertNonNegativeInteger(r.start, 'range.start');
      assertNonNegativeInteger(r.end, 'range.end');
      if (r.end < r.start) {
        throw new RangeError(`Dirty range end (${r.end}) must be >= start (${r.start})`);
      }
      const writes = r.writeCount === undefined ? 1 : r.writeCount;
      assertPositiveInteger(writes, 'range.writeCount');
      insertDirtyRange(merged, r.start, r.end, writes, maxGapBytes);
    }
    const result = new Array(merged.length);
    for (let i = 0; i < merged.length; i += 1) {
      const r = merged[i];
      result[i] = {
        start: r.start,
        end: r.end,
        byteLength: r.end - r.start,
        writeCount: r.writeCount,
      };
    }
    return result;
  }

  /**
   * Multi-buffered WebGPU dynamic uniform buffer ring for per-draw D3D12 root constants / CBVs.
   *
   * Accumulates per-draw uniform payloads into an aligned CPU shadow buffer and uploads the
   * dirty `[0, usedBytes)` prefix in a single `device.queue.writeBuffer(...)` call on `flush()`.
   * Rotates across `framesInFlight` dedicated `GPUBuffer` slots so frame N+1 never overwrites
   * uniform data still being read by the GPU for frame N.
   */
  class DynamicUniformRing {
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('DynamicUniformRing constructor expects an options object');
      }

      const {
        device,
        byteSize,
        capacityBytes,
        alignment,
        minUniformBufferOffsetAlignment,
        maxByteSize,
        maxCapacityBytes,
        framesInFlight = DEFAULT_FRAMES_IN_FLIGHT,
        maxBindingSize,
        minUniformBindingSize = DEFAULT_MIN_UNIFORM_BINDING_SIZE,
        allowEmptyPayloads = false,
        allowOverflow = false,
        allowSpill,
        maxRingBytes,
        multiSlice,
        preserveBuffersOnGrow,
        autoRollover,
        rolloverOnOverflow,
        allowWrap = false,
        wrapOnOverflow = false,
        immediateFlush = false,
        maxCoalesceGapBytes = 0,
        minSlotBytes,
        minCapacityBytes,
        shrinkAfterIdleFrames = DEFAULT_SHRINK_AFTER_IDLE_FRAMES,
        shrinkThresholdRatio = DEFAULT_SHRINK_THRESHOLD_RATIO,
      } = options;

      assertNonNegativeInteger(maxCoalesceGapBytes, 'maxCoalesceGapBytes');

      let resolvedShrinkAfterIdleFrames;
      if (
        shrinkAfterIdleFrames === false ||
        shrinkAfterIdleFrames === null ||
        shrinkAfterIdleFrames === 0
      ) {
        resolvedShrinkAfterIdleFrames = 0;
      } else {
        assertPositiveInteger(shrinkAfterIdleFrames, 'shrinkAfterIdleFrames');
        resolvedShrinkAfterIdleFrames = shrinkAfterIdleFrames;
      }

      if (
        typeof shrinkThresholdRatio !== 'number' ||
        !Number.isFinite(shrinkThresholdRatio) ||
        shrinkThresholdRatio <= 0 ||
        shrinkThresholdRatio >= 1
      ) {
        throw new RangeError(
          `shrinkThresholdRatio must be a finite number in (0, 1), got ${shrinkThresholdRatio}`
        );
      }

      if (!device || typeof device.createBuffer !== 'function') {
        throw new TypeError('DynamicUniformRing requires a WebGPU device with createBuffer()');
      }

      const deviceAlignment = resolveDeviceUniformAlignment(device);
      const rawAlignment =
        alignment !== undefined ? alignment : minUniformBufferOffsetAlignment;
      let resolvedAlignment;
      if (rawAlignment === undefined) {
        resolvedAlignment = deviceAlignment;
      } else {
        if (!isPowerOfTwo(rawAlignment) || rawAlignment < WEBGPU_BUFFER_COPY_ALIGNMENT) {
          throw new RangeError(
            `alignment must be a power-of-two integer >= ${WEBGPU_BUFFER_COPY_ALIGNMENT}, got ${rawAlignment}`
          );
        }
        // Never allow dynamic offsets below the device's minUniformBufferOffsetAlignment.
        resolvedAlignment = Math.max(rawAlignment, deviceAlignment);
      }

      const resolvedMaxBindingSize =
        maxBindingSize === undefined ? resolveDeviceMaxBindingSize(device) : maxBindingSize;

      const rawMaxByteSize =
        maxByteSize !== undefined ? maxByteSize : maxCapacityBytes;
      const hasExplicitMaxByteSize = rawMaxByteSize !== undefined;
      const hasExplicitMaxRingBytes = maxRingBytes !== undefined;
      const rawByteSize = byteSize !== undefined ? byteSize : capacityBytes;
      const resolvedMultiSlice =
        multiSlice !== undefined
          ? Boolean(multiSlice)
          : preserveBuffersOnGrow !== undefined
            ? Boolean(preserveBuffersOnGrow)
            : autoRollover !== undefined
              ? Boolean(autoRollover)
              : rolloverOnOverflow !== undefined
                ? Boolean(rolloverOnOverflow)
                : false;

      if (hasExplicitMaxRingBytes) {
        assertPositiveInteger(maxRingBytes, 'maxRingBytes');
        if (maxRingBytes < resolvedAlignment) {
          throw new RangeError(
            `maxRingBytes (${maxRingBytes}) must be at least alignment (${resolvedAlignment})`
          );
        }
      }

      assertPositiveInteger(framesInFlight, 'framesInFlight');
      const resolvedFramesInFlight = Math.min(
        MAX_FRAMES_IN_FLIGHT,
        Math.max(MIN_FRAMES_IN_FLIGHT, framesInFlight)
      );
      assertPositiveInteger(resolvedMaxBindingSize, 'maxBindingSize');
      assertPositiveInteger(minUniformBindingSize, 'minUniformBindingSize');

      const resolvedMinUniformBindingSize = Math.max(
        DEFAULT_MIN_UNIFORM_BINDING_SIZE,
        alignUp(minUniformBindingSize, WEBGPU_BUFFER_COPY_ALIGNMENT)
      );

      if (resolvedMinUniformBindingSize > resolvedMaxBindingSize) {
        throw new RangeError(
          `minUniformBindingSize (${resolvedMinUniformBindingSize}) exceeds maxBindingSize (${resolvedMaxBindingSize})`
        );
      }

      const spillRequestedByOptions =
        allowSpill === true || (allowSpill === undefined && hasExplicitMaxRingBytes);

      let resolvedMaxByteSize;
      if (hasExplicitMaxByteSize) {
        assertPositiveInteger(rawMaxByteSize, 'maxByteSize');
        if (rawMaxByteSize < resolvedAlignment) {
          throw new RangeError(
            `maxByteSize (${rawMaxByteSize}) must be at least alignment (${resolvedAlignment})`
          );
        }
        resolvedMaxByteSize = rawMaxByteSize;
      } else if (spillRequestedByOptions) {
        resolvedMaxByteSize = null;
      } else if (hasExplicitMaxRingBytes) {
        resolvedMaxByteSize = Math.min(DEFAULT_MAX_BYTE_SIZE, maxRingBytes);
      } else {
        resolvedMaxByteSize = DEFAULT_MAX_BYTE_SIZE;
      }

      let resolvedByteSize;
      if (rawByteSize === undefined) {
        const capBasis =
          resolvedMaxByteSize !== null
            ? resolvedMaxByteSize
            : hasExplicitMaxRingBytes
              ? maxRingBytes
              : DEFAULT_MAX_BYTE_SIZE;
        const clampedMax =
          Math.floor(capBasis / resolvedAlignment) * resolvedAlignment;
        resolvedByteSize = Math.min(
          alignUp(DEFAULT_BYTE_SIZE, resolvedAlignment),
          clampedMax
        );
      } else {
        assertPositiveInteger(rawByteSize, 'byteSize');
        resolvedByteSize = alignUp(rawByteSize, resolvedAlignment);
        if (resolvedMaxByteSize !== null && resolvedByteSize > resolvedMaxByteSize) {
          throw new RangeError(
            `Aligned byteSize (${resolvedByteSize}) exceeds maxByteSize (${resolvedMaxByteSize})`
          );
        }
      }

      if (resolvedMaxByteSize === null) {
        resolvedMaxByteSize = resolvedByteSize;
      }

      const rawMinSlotBytes =
        minSlotBytes !== undefined ? minSlotBytes : minCapacityBytes;
      let resolvedMinSlotBytes;
      if (rawMinSlotBytes === undefined) {
        resolvedMinSlotBytes = resolvedByteSize;
      } else {
        assertPositiveInteger(rawMinSlotBytes, 'minSlotBytes');
        resolvedMinSlotBytes = alignUp(rawMinSlotBytes, resolvedAlignment);
        if (resolvedMinSlotBytes > resolvedMaxByteSize) {
          throw new RangeError(
            `Aligned minSlotBytes (${resolvedMinSlotBytes}) exceeds maxByteSize (${resolvedMaxByteSize})`
          );
        }
        if (rawByteSize !== undefined && resolvedMinSlotBytes > resolvedByteSize) {
          throw new RangeError(
            `Aligned minSlotBytes (${resolvedMinSlotBytes}) exceeds aligned byteSize (${resolvedByteSize})`
          );
        }
        if (rawByteSize === undefined && resolvedMinSlotBytes > resolvedByteSize) {
          resolvedByteSize = resolvedMinSlotBytes;
        }
      }

      if (hasExplicitMaxRingBytes) {
        if (maxRingBytes < resolvedByteSize) {
          throw new RangeError(
            `maxRingBytes (${maxRingBytes}) must be at least aligned byteSize (${resolvedByteSize})`
          );
        }
        if (hasExplicitMaxByteSize && maxRingBytes < resolvedMaxByteSize) {
          throw new RangeError(
            `maxRingBytes (${maxRingBytes}) must be at least maxByteSize (${resolvedMaxByteSize})`
          );
        }
      }

      let resolvedAllowSpill;
      if (allowSpill !== undefined) {
        resolvedAllowSpill = Boolean(allowSpill);
      } else if (hasExplicitMaxRingBytes && maxRingBytes > resolvedMaxByteSize) {
        resolvedAllowSpill = true;
      } else {
        resolvedAllowSpill = false;
      }

      let resolvedMaxRingBytes;
      if (hasExplicitMaxRingBytes) {
        resolvedMaxRingBytes = maxRingBytes;
      } else if (resolvedAllowSpill) {
        resolvedMaxRingBytes = Math.max(DEFAULT_MAX_BYTE_SIZE, resolvedMaxByteSize);
      } else if (allowOverflow) {
        resolvedMaxRingBytes = null;
      } else {
        resolvedMaxRingBytes = resolvedMaxByteSize;
      }

      this.device = device;
      this.alignment = resolvedAlignment;
      this.deviceMinUniformBufferOffsetAlignment = deviceAlignment;
      this.maxByteSize = resolvedMaxByteSize;
      this.maxCapacityBytes = resolvedMaxByteSize;
      this.framesInFlight = resolvedFramesInFlight;
      this.maxBindingSize = resolvedMaxBindingSize;
      this.minUniformBindingSize = resolvedMinUniformBindingSize;
      this.allowEmptyPayloads = Boolean(allowEmptyPayloads);
      this.allowOverflow = Boolean(allowOverflow);
      this.allowSpill = resolvedAllowSpill;
      this.maxRingBytes = resolvedMaxRingBytes;
      this.multiSlice = resolvedMultiSlice;
      this.allowWrap = Boolean(allowWrap || wrapOnOverflow);
      this.immediateFlush = Boolean(immediateFlush);
      this.maxCoalesceGapBytes = maxCoalesceGapBytes;
      this.minSlotBytes = resolvedMinSlotBytes;
      this.shrinkAfterIdleFrames = resolvedShrinkAfterIdleFrames;
      this.shrinkThresholdRatio = shrinkThresholdRatio;

      this._alignmentMask =
        resolvedAlignment <= 0x40000000 ? resolvedAlignment - 1 : -1;
      this._capacity = resolvedByteSize;
      this._shadow = null;
      this._shadowU32 = null;
      this._shadowI32 = null;
      this._shadowF32 = null;
      this._dirtyRanges = [];
      this._dirtyRangePool = [
        { start: 0, end: 0, writeCount: 0 },
        { start: 0, end: 0, writeCount: 0 },
      ];
      this._retiredSlicePool = [];
      this._tmpOverflowAlloc = {
        chunk: null,
        dynamicOffset: 0,
        createdNewPage: false,
      };
      this._tmpFlushResult = {
        uploadedBytes: 0,
        bytesUploaded: 0,
        drawCount: 0,
        stagedWrites: 0,
        writeBufferCalls: 0,
        coalescedWrites: 0,
        dirtyRangesFlushed: 0,
      };
      this._tmpReclaimResult = {
        reclaimedPages: 0,
        reclaimedBytes: 0,
      };
      this._singleDynamicOffset = new Uint32Array(1);
      this._slots = [];
      this._destroyed = false;

      this._frameIndex = -1;
      this._slotIndex = 0;
      this._frameEnded = false;
      this._cursor = 0;
      this._writtenHighWater = 0;
      this._framePrimaryBytes = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._hasFlushedActiveSlot = false;
      this._primaryFrameDrawCount = 0;
      this._frameSliceBytes = 0;
      this._activeSliceIndex = 0;
      this._frameOverflowBytes = 0;
      this._batchDepth = 0;
      this._dirty = false;
      this._consecutiveIdleFrames = 0;
      this._idleWindowHighWaterBytes = 0;

      // Telemetry counters.
      this._allocatedGpuBytes = 0;
      this._peakAllocatedGpuBytes = 0;
      this._peakUsedBytes = 0;
      this._peakFrameBytes = 0;
      this._frameDrawCount = 0;
      this._totalDrawCount = 0;
      this._reusedAllocations = 0;
      this._freshAllocations = 0;
      this._zeroAllocAllocs = 0;
      this._partitionRotations = 0;
      this._totalBytesPushed = 0;
      this._totalBytesUploaded = 0;
      this._alignedBytesWasted = 0;
      this._flushCount = 0;
      this._writeBufferCalls = 0;
      this._coalescedWrites = 0;
      this._stagedWritesFlushed = 0;
      this._growCount = 0;
      this._shrinkCount = 0;
      this._overflowAllocCount = 0;
      this._spillAllocations = 0;
      this._spillPagesCreated = 0;
      this._sabCopyCount = 0;
      this._unalignedCopyCount = 0;
      this._wrapCount = 0;
      this._rolloverCount = 0;
      this._sliceRolloverCount = 0;
      this._spillRolloverCount = 0;
      this._maxSlicesPerFrame = 0;
      this._alignmentViolationsRejected = 0;

      for (let i = 0; i < resolvedFramesInFlight; i += 1) {
        const buffer = this._createSlotBuffer(i, resolvedByteSize);
        const slot = {
          index: i,
          partitionIndex: i,
          buffer,
          byteSize: resolvedByteSize,
          shadow: null,
          shadowU32: null,
          shadowI32: null,
          shadowF32: null,
          cachedFlushView: null,
          cachedFlushStart: -1,
          cachedFlushEnd: -1,
          generation: 1,
          used: false,
          inFlight: false,
          lastFrameIndex: -1,
          endedFrameIndex: -1,
          completedFrameIndex: -1,
          lastFrameBytes: 0,
          fencePromise: null,
          retiredBuffers: [],
          retiredSlices: [],
          overflowChunks: [],
        };
        this._setSlotShadowBuffer(slot, new Uint8Array(resolvedByteSize));
        this._slots.push(slot);
      }
      this._bindSlotShadow(this._slots[0]);
    }

    get currentBuffer() {
      if (this._destroyed || this._slots.length === 0) {
        return null;
      }
      return this._slots[this._slotIndex].buffer;
    }

    get capacity() {
      return this._capacity;
    }

    get capacityBytes() {
      return this._capacity;
    }

    get currentCapacityBytes() {
      return this._capacity;
    }

    get usedBytes() {
      return this._frameSliceBytes + this._framePrimaryBytes + this._frameOverflowBytes;
    }

    get cursor() {
      return this._cursor;
    }

    get batchDepth() {
      return this._batchDepth;
    }

    get isBatchActive() {
      return this._batchDepth > 0;
    }

    get slotIndex() {
      return this._slotIndex;
    }

    get partitionIndex() {
      return this._slotIndex;
    }

    get frameIndex() {
      return this._frameIndex;
    }

    _assertNotDestroyed() {
      if (this._destroyed) {
        throw new Error('DynamicUniformRing has been destroyed');
      }
    }

    _setSlotShadowBuffer(slot, buffer) {
      slot.shadow = buffer;
      slot.cachedFlushView = null;
      slot.cachedFlushStart = -1;
      slot.cachedFlushEnd = -1;
      if (buffer && (buffer.byteLength & 3) === 0) {
        slot.shadowU32 = new Uint32Array(
          buffer.buffer,
          buffer.byteOffset,
          buffer.byteLength >>> 2
        );
        slot.shadowI32 = new Int32Array(
          buffer.buffer,
          buffer.byteOffset,
          buffer.byteLength >>> 2
        );
        slot.shadowF32 = new Float32Array(
          buffer.buffer,
          buffer.byteOffset,
          buffer.byteLength >>> 2
        );
      } else {
        slot.shadowU32 = null;
        slot.shadowI32 = null;
        slot.shadowF32 = null;
      }
    }

    _alignToRingFast(value) {
      const mask = this._alignmentMask;
      if (mask >= 0 && value <= 0x7fffffff - mask) {
        return (value + mask) & ~mask;
      }
      return alignUp(value, this.alignment);
    }

    _getSlotShadowSlice(slot, start, end) {
      const shadow = this._shadow;
      if (start === 0 && end === shadow.byteLength) {
        return shadow;
      }
      if (
        slot.cachedFlushView !== null &&
        slot.cachedFlushStart === start &&
        slot.cachedFlushEnd === end &&
        slot.cachedFlushView.buffer === shadow.buffer
      ) {
        return slot.cachedFlushView;
      }
      const view = shadow.subarray(start, end);
      slot.cachedFlushView = view;
      slot.cachedFlushStart = start;
      slot.cachedFlushEnd = end;
      return view;
    }

    _bindSlotShadow(slot) {
      if (!slot || !slot.shadow) {
        this._shadow = null;
        this._shadowU32 = null;
        this._shadowI32 = null;
        this._shadowF32 = null;
        return;
      }
      this._shadow = slot.shadow;
      this._shadowU32 = slot.shadowU32;
      this._shadowI32 = slot.shadowI32;
      this._shadowF32 = slot.shadowF32;
    }

    _setShadowBuffer(buffer) {
      const activeSlot =
        !this._destroyed && this._slots && this._slots.length > 0
          ? this._slots[this._slotIndex]
          : null;
      if (activeSlot) {
        this._setSlotShadowBuffer(activeSlot, buffer);
        this._bindSlotShadow(activeSlot);
        return;
      }
      this._shadow = buffer;
      if (buffer && (buffer.byteLength & 3) === 0) {
        this._shadowU32 = new Uint32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength >>> 2);
        this._shadowI32 = new Int32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength >>> 2);
        this._shadowF32 = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength >>> 2);
      } else {
        this._shadowU32 = null;
        this._shadowI32 = null;
        this._shadowF32 = null;
      }
    }

    _ensureActiveFrame() {
      if (this._frameIndex < 0) {
        this._frameIndex = 0;
        const slot = this._slots[this._slotIndex];
        slot.used = true;
        slot.inFlight = true;
        slot.lastFrameIndex = 0;
        this._frameEnded = false;
      } else if (this._frameEnded) {
        this.beginFrame(this._frameIndex + 1);
      }
    }

    _computeShrinkTarget(highWaterBytes) {
      const withHeadroom = highWaterBytes > 0 ? highWaterBytes * 2 : 0;
      const minNeeded = Math.max(this.minSlotBytes, alignUp(withHeadroom, this.alignment));
      let target = this.minSlotBytes;
      while (target < minNeeded && target < this._capacity) {
        target *= 2;
      }
      target = alignUp(target, this.alignment);
      if (target < this.minSlotBytes) {
        target = this.minSlotBytes;
      }
      if (target >= this._capacity) {
        return this._capacity;
      }
      return target;
    }

    _evaluateAutoShrink(prevFrameBytes) {
      if (this.shrinkAfterIdleFrames <= 0 || this._capacity <= this.minSlotBytes) {
        this._consecutiveIdleFrames = 0;
        this._idleWindowHighWaterBytes = 0;
        return false;
      }

      const idleThreshold = Math.floor(this._capacity * this.shrinkThresholdRatio);
      if (prevFrameBytes <= idleThreshold) {
        this._consecutiveIdleFrames += 1;
        if (prevFrameBytes > this._idleWindowHighWaterBytes) {
          this._idleWindowHighWaterBytes = prevFrameBytes;
        }
        if (this._consecutiveIdleFrames >= this.shrinkAfterIdleFrames) {
          const targetCapacity = this._computeShrinkTarget(this._idleWindowHighWaterBytes);
          this._consecutiveIdleFrames = 0;
          this._idleWindowHighWaterBytes = 0;
          if (targetCapacity < this._capacity) {
            this._capacity = targetCapacity;
            this._shrinkCount += 1;
            return true;
          }
        }
      } else {
        this._consecutiveIdleFrames = 0;
        this._idleWindowHighWaterBytes = 0;
      }
      return false;
    }

    /**
     * Explicitly request shrinking the ring's target capacity down toward `targetBytes`
     * (default `minSlotBytes`). Any in-flight or already-written slot buffers are preserved
     * and shrink safely via deferred retirement on their next `beginFrame()` rotation.
     *
     * @param {number} [targetBytes]
     * @returns {{ shrunk: boolean, previousCapacity: number, capacity: number }}
     */
    shrink(targetBytes = this.minSlotBytes) {
      this._assertNotDestroyed();
      assertPositiveInteger(targetBytes, 'targetBytes');

      const alignedTarget = alignUp(targetBytes, this.alignment);
      const boundedTarget = Math.max(
        this.minSlotBytes,
        Math.min(this._capacity, alignedTarget)
      );
      const previousCapacity = this._capacity;
      if (boundedTarget >= previousCapacity) {
        return {
          shrunk: false,
          previousCapacity,
          capacity: this._capacity,
        };
      }

      this._capacity = boundedTarget;
      this._shrinkCount += 1;
      this._consecutiveIdleFrames = 0;
      this._idleWindowHighWaterBytes = 0;

      // Only shrink the active slot in place if it has no pending/flushed draws in the
      // current frame; otherwise defer shrinking cleanly until its next beginFrame() wrap.
      const activeSlot = this._slots[this._slotIndex];
      if (
        activeSlot &&
        activeSlot.byteSize > boundedTarget &&
        this._cursor === 0 &&
        !this._hasFlushedActiveSlot &&
        this._frameSliceBytes === 0 &&
        this._framePrimaryBytes === 0
      ) {
        this._resizeSlot(activeSlot, boundedTarget, false);
        this._setShadowBuffer(new Uint8Array(boundedTarget));
      }

      return {
        shrunk: true,
        previousCapacity,
        capacity: this._capacity,
      };
    }

    _createGpuUniformBuffer(label, byteSize) {
      const buffer = this.device.createBuffer({
        label,
        size: byteSize,
        usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
      });
      this._allocatedGpuBytes += byteSize;
      if (this._allocatedGpuBytes > this._peakAllocatedGpuBytes) {
        this._peakAllocatedGpuBytes = this._allocatedGpuBytes;
      }
      return buffer;
    }

    _createSlotBuffer(slotIndex, byteSize) {
      return this._createGpuUniformBuffer(`playgta6-uniform-ring-slot-${slotIndex}`, byteSize);
    }

    _destroyRawBuffer(buffer, byteSize) {
      if (!buffer) {
        return;
      }
      this._allocatedGpuBytes = Math.max(0, this._allocatedGpuBytes - byteSize);
      if (typeof buffer.destroy === 'function') {
        buffer.destroy();
      }
    }

    _destroySlotBuffer(slot) {
      if (!slot || !slot.buffer) {
        return;
      }
      const oldBuffer = slot.buffer;
      const oldSize = slot.byteSize;
      slot.buffer = null;
      slot.byteSize = 0;
      this._destroyRawBuffer(oldBuffer, oldSize);
    }

    _releaseSlotTransientBuffers(slot) {
      if (!slot) {
        return;
      }
      if (slot.retiredBuffers && slot.retiredBuffers.length > 0) {
        for (let i = 0; i < slot.retiredBuffers.length; i += 1) {
          const entry = slot.retiredBuffers[i];
          this._destroyRawBuffer(entry.buffer, entry.byteSize);
        }
        slot.retiredBuffers.length = 0;
      }
      if (slot.retiredSlices && slot.retiredSlices.length > 0) {
        for (let i = 0; i < slot.retiredSlices.length; i += 1) {
          const slice = slot.retiredSlices[i];
          this._destroyRawBuffer(slice.buffer, slice.byteSize);
          if (slice.dirtyRanges && slice.dirtyRanges.length > 0) {
            releaseDirtyRanges(slice.dirtyRanges, this._dirtyRangePool);
          }
          slice.buffer = null;
          slice.shadow = null;
          slice.byteSize = 0;
          slice.cursor = 0;
          slice.flushedCursor = 0;
          slice.drawCount = 0;
          slice.flushedDrawCount = 0;
          slice.dirty = false;
          if (this._retiredSlicePool.length < MAX_DIRTY_RANGE_POOL_SIZE) {
            this._retiredSlicePool.push(slice);
          }
        }
        slot.retiredSlices.length = 0;
      }
      if (slot.overflowChunks && slot.overflowChunks.length > 0) {
        for (let i = 0; i < slot.overflowChunks.length; i += 1) {
          const chunk = slot.overflowChunks[i];
          this._destroyRawBuffer(chunk.buffer, chunk.byteSize);
          if (chunk.dirtyRanges && chunk.dirtyRanges.length > 0) {
            releaseDirtyRanges(chunk.dirtyRanges, this._dirtyRangePool);
          }
          chunk.buffer = null;
          chunk.shadow = null;
          chunk.shadowU32 = null;
          chunk.shadowI32 = null;
          chunk.shadowF32 = null;
        }
        slot.overflowChunks.length = 0;
      }
    }

    _resizeSlot(slot, targetByteSize, deferOldDestroy = false) {
      if (slot.byteSize === targetByteSize && slot.buffer) {
        return;
      }
      if (deferOldDestroy && slot.buffer) {
        // Keep previously flushed buffer alive until this slot is next reset by beginFrame()
        // or destroyed, preventing WebGPU use-after-destroy for earlier draws in the frame.
        slot.retiredBuffers.push({
          buffer: slot.buffer,
          byteSize: slot.byteSize,
        });
        slot.buffer = null;
        slot.byteSize = 0;
      } else {
        this._destroySlotBuffer(slot);
      }
      slot.buffer = this._createSlotBuffer(slot.index, targetByteSize);
      slot.byteSize = targetByteSize;
      slot.generation += 1;
    }

    _grow(requiredBytes) {
      if (requiredBytes > this.maxByteSize) {
        throw new RangeError(
          `DynamicUniformRing capacity exceeded: need ${requiredBytes} bytes, ` +
            `maxByteSize is ${this.maxByteSize} bytes`
        );
      }

      let nextCapacity = this._capacity;
      while (nextCapacity < requiredBytes) {
        nextCapacity *= 2;
      }
      if (nextCapacity > this.maxByteSize) {
        const alignedMax =
          Math.floor(this.maxByteSize / this.alignment) * this.alignment;
        const copyAlignedMax =
          Math.floor(this.maxByteSize / WEBGPU_BUFFER_COPY_ALIGNMENT) *
          WEBGPU_BUFFER_COPY_ALIGNMENT;
        nextCapacity = Math.max(requiredBytes, alignedMax >= requiredBytes ? alignedMax : copyAlignedMax);
      }

      const nextShadow = new Uint8Array(nextCapacity);
      const copyLen = Math.max(this._cursor, this._writtenHighWater);
      if (copyLen > 0) {
        nextShadow.set(this._shadow.subarray(0, copyLen), 0);
      }
      this._setShadowBuffer(nextShadow);
      this._capacity = nextCapacity;
      this._consecutiveIdleFrames = 0;
      this._idleWindowHighWaterBytes = 0;

      // Resize only the active slot now; other slots may still be in flight on the GPU
      // and will resize cleanly when beginFrame() rotates to them.
      // If draws were already flushed into the active slot's previous buffer in this frame,
      // retain the old buffer until the slot rotates back in a later frame.
      const activeSlot = this._slots[this._slotIndex];
      const deferOldDestroy = this._flushedCursor > 0 || this._hasFlushedActiveSlot;
      this._resizeSlot(activeSlot, nextCapacity, deferOldDestroy);

      // The active slot now has a fresh GPUBuffer, so any prior partial flush in this frame
      // must be re-uploaded from offset 0 on the next flush().
      releaseDirtyRanges(this._dirtyRanges, this._dirtyRangePool);
      if (copyLen > 0) {
        insertDirtyRange(
          this._dirtyRanges,
          0,
          copyLen,
          Math.max(1, this._primaryFrameDrawCount),
          this.maxCoalesceGapBytes,
          this._dirtyRangePool
        );
      }
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._hasFlushedActiveSlot = false;
      this._growCount += 1;
      this._overflowAllocCount += 1;
    }

    _getSlotRetiredSliceBytes(slot) {
      let bytes = 0;
      if (slot && slot.retiredSlices) {
        for (let i = 0; i < slot.retiredSlices.length; i += 1) {
          bytes += slot.retiredSlices[i].byteSize;
        }
      }
      return bytes;
    }

    _getSlotPrimaryAndRetiredBytes(slot) {
      if (!slot) {
        return 0;
      }
      return slot.byteSize + this._getSlotRetiredSliceBytes(slot);
    }

    _getSlotSpillBytes(slot) {
      let bytes = 0;
      if (slot && slot.overflowChunks) {
        for (let i = 0; i < slot.overflowChunks.length; i += 1) {
          bytes += slot.overflowChunks[i].byteSize;
        }
      }
      return bytes;
    }

    _getSlotLiveBytes(slot) {
      if (!slot) {
        return 0;
      }
      return (
        slot.byteSize +
        this._getSlotRetiredSliceBytes(slot) +
        this._getSlotSpillBytes(slot)
      );
    }

    _recordFramePeak() {
      const totalUsed = this._frameSliceBytes + this._framePrimaryBytes + this._frameOverflowBytes;
      if (totalUsed > this._peakUsedBytes) {
        this._peakUsedBytes = totalUsed;
      }
      if (totalUsed > this._peakFrameBytes) {
        this._peakFrameBytes = totalUsed;
      }
    }

    /**
     * Non-destructive multi-slice dynamic growth for heavy single-pass frames:
     * preserves the currently-filled slice GPUBuffer (and its dirty CPU shadow range)
     * in `slot.retiredSlices` so earlier draw allocations bound to `alloc.buffer` remain
     * intact and get flushed before GPU submission, then allocates a fresh geometrically
     * larger slice bounded by `maxCapacityBytes` (`maxByteSize`).
     *
     * Returns `false` (without mutating state) if the remaining budget under `maxByteSize`
     * cannot accommodate `alignedSize`.
     *
     * @param {object} slot
     * @param {number} alignedSize
     * @returns {boolean}
     */
    _growMultiSlice(slot, alignedSize) {
      const alignedMax =
        Math.floor(this.maxByteSize / this.alignment) * this.alignment;
      const retiredBytes = this._getSlotRetiredSliceBytes(slot);

      // Fast path when the active slice has zero allocations yet (e.g., first draw of
      // the frame is larger than the current slice size): resize in place cleanly.
      if (this._cursor === 0) {
        const remaining = this.maxByteSize - retiredBytes;
        const maxAlignedRemaining =
          remaining > 0 ? Math.floor(remaining / this.alignment) * this.alignment : 0;
        if (maxAlignedRemaining < alignedSize) {
          return false;
        }
        let nextSize = Math.max(slot.byteSize, this.alignment);
        while (nextSize < alignedSize) {
          nextSize *= 2;
        }
        nextSize = Math.min(alignUp(nextSize, this.alignment), maxAlignedRemaining);
        this._resizeSlot(slot, nextSize, false);
        this._setShadowBuffer(new Uint8Array(nextSize));
        if (nextSize > this._capacity) {
          this._capacity = Math.min(alignedMax, nextSize);
        }
        this._consecutiveIdleFrames = 0;
        this._idleWindowHighWaterBytes = 0;
        this._growCount += 1;
        this._overflowAllocCount += 1;
        return true;
      }

      const committedBytes = retiredBytes + slot.byteSize;
      const remaining = this.maxByteSize - committedBytes;
      const maxAlignedRemaining =
        remaining > 0 ? Math.floor(remaining / this.alignment) * this.alignment : 0;
      if (maxAlignedRemaining < alignedSize) {
        return false;
      }

      let nextSliceSize = Math.max(slot.byteSize * 2, alignedSize);
      nextSliceSize = alignUp(nextSliceSize, this.alignment);
      if (nextSliceSize > maxAlignedRemaining) {
        nextSliceSize = maxAlignedRemaining;
      }

      const sliceDirty = this._dirtyRanges.length > 0;
      let retiredEntry;
      let nextDirtyRanges;
      if (this._retiredSlicePool.length > 0) {
        retiredEntry = this._retiredSlicePool.pop();
        nextDirtyRanges = retiredEntry.dirtyRanges || [];
        nextDirtyRanges.length = 0;
      } else {
        retiredEntry = {
          buffer: null,
          byteSize: 0,
          shadow: null,
          dirtyRanges: null,
          cursor: 0,
          flushedCursor: 0,
          drawCount: 0,
          flushedDrawCount: 0,
          dirty: false,
        };
        nextDirtyRanges = [];
      }
      retiredEntry.buffer = slot.buffer;
      retiredEntry.byteSize = slot.byteSize;
      retiredEntry.shadow = sliceDirty ? this._shadow : null;
      retiredEntry.dirtyRanges = sliceDirty ? this._dirtyRanges : nextDirtyRanges;
      retiredEntry.cursor = this._cursor;
      retiredEntry.flushedCursor = this._flushedCursor;
      retiredEntry.drawCount = this._primaryFrameDrawCount;
      retiredEntry.flushedDrawCount = this._flushedDrawCount;
      retiredEntry.dirty = sliceDirty;
      slot.retiredSlices.push(retiredEntry);

      this._frameSliceBytes += this._framePrimaryBytes;
      this._activeSliceIndex += 1;

      const nextBuffer = this._createGpuUniformBuffer(
        `playgta6-uniform-ring-slot-${slot.index}-slice-${this._activeSliceIndex}`,
        nextSliceSize
      );
      slot.buffer = nextBuffer;
      slot.byteSize = nextSliceSize;
      slot.generation += 1;

      this._setShadowBuffer(new Uint8Array(nextSliceSize));
      this._dirtyRanges = sliceDirty ? nextDirtyRanges : this._dirtyRanges;
      this._dirtyRanges.length = 0;
      this._cursor = 0;
      this._writtenHighWater = 0;
      this._framePrimaryBytes = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._primaryFrameDrawCount = 0;
      this._hasFlushedActiveSlot = false;

      // Raise steady-state per-slot capacity target so subsequent frames fit in a single buffer.
      const totalFrameCapacity = this._getSlotPrimaryAndRetiredBytes(slot);
      let targetCap = this._capacity;
      while (targetCap < totalFrameCapacity && targetCap < alignedMax) {
        targetCap *= 2;
      }
      targetCap = Math.min(alignedMax, Math.max(targetCap, totalFrameCapacity));
      if (targetCap > this._capacity) {
        this._capacity = targetCap;
      }
      this._consecutiveIdleFrames = 0;
      this._idleWindowHighWaterBytes = 0;

      this._growCount += 1;
      this._overflowAllocCount += 1;
      this._rolloverCount += 1;
      this._sliceRolloverCount += 1;
      this._updateMaxSlicesPerFrame(slot);
      return true;
    }

    _updateMaxSlicesPerFrame(slot) {
      if (!slot) {
        return;
      }
      const totalSlices = 1 + slot.retiredSlices.length + slot.overflowChunks.length;
      if (totalSlices > this._maxSlicesPerFrame) {
        this._maxSlicesPerFrame = totalSlices;
      }
    }

    _allocateOverflowSlice(slot, alignedSize) {
      let chunk =
        slot.overflowChunks.length > 0
          ? slot.overflowChunks[slot.overflowChunks.length - 1]
          : null;
      let createdNewPage = false;

      if (!chunk || alignUp(chunk.cursor, this.alignment) + alignedSize > chunk.byteSize) {
        const chunkIndex = slot.overflowChunks.length;
        const desiredPageSize = alignUp(
          Math.max(alignedSize, this._capacity),
          this.alignment
        );
        let chunkByteSize = desiredPageSize;
        if (this.maxRingBytes !== null) {
          const liveBytes = this._getSlotLiveBytes(slot);
          const remainingBytes = this.maxRingBytes - liveBytes;
          const maxAlignedRemaining =
            remainingBytes > 0
              ? Math.floor(remainingBytes / this.alignment) * this.alignment
              : 0;
          if (maxAlignedRemaining < alignedSize) {
            throw new RangeError(
              `DynamicUniformRing maxRingBytes budget exceeded: need ${liveBytes + alignedSize} bytes ` +
                `(live ${liveBytes} + ${alignedSize}), maxRingBytes is ${this.maxRingBytes} bytes`
            );
          }
          chunkByteSize = Math.min(desiredPageSize, maxAlignedRemaining);
        }

        const buffer = this._createGpuUniformBuffer(
          `playgta6-uniform-ring-slot-${slot.index}-overflow-${chunkIndex}`,
          chunkByteSize
        );
        const chunkShadow = new Uint8Array(chunkByteSize);
        const aligned4Shadow = (chunkByteSize & 3) === 0;
        chunk = {
          buffer,
          byteSize: chunkByteSize,
          shadow: chunkShadow,
          shadowU32: aligned4Shadow
            ? new Uint32Array(chunkShadow.buffer, chunkShadow.byteOffset, chunkByteSize >>> 2)
            : null,
          shadowI32: aligned4Shadow
            ? new Int32Array(chunkShadow.buffer, chunkShadow.byteOffset, chunkByteSize >>> 2)
            : null,
          shadowF32: aligned4Shadow
            ? new Float32Array(chunkShadow.buffer, chunkShadow.byteOffset, chunkByteSize >>> 2)
            : null,
          dirtyRanges: [],
          cursor: 0,
          flushedCursor: 0,
          drawCount: 0,
          flushedDrawCount: 0,
          dirty: false,
        };
        slot.overflowChunks.push(chunk);
        this._overflowAllocCount += 1;
        this._spillPagesCreated += 1;
        this._rolloverCount += 1;
        this._spillRolloverCount += 1;
        this._updateMaxSlicesPerFrame(slot);
        createdNewPage = true;
      }

      const dynamicOffset = alignUp(chunk.cursor, this.alignment);
      const out = this._tmpOverflowAlloc;
      out.chunk = chunk;
      out.dynamicOffset = dynamicOffset;
      out.createdNewPage = createdNewPage;
      return out;
    }

    _safeWriteBuffer(buffer, bufferOffset, data) {
      if (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function') {
        throw new TypeError(
          'DynamicUniformRing.flush() requires device.queue.writeBuffer()'
        );
      }
      // Zero-allocation fast path for internal regular ArrayBuffer 4-byte-aligned shadow slices.
      if (
        data instanceof Uint8Array &&
        !isSharedArrayBuffer(data.buffer) &&
        (data.byteOffset & 3) === 0 &&
        (data.byteLength & 3) === 0
      ) {
        this.device.queue.writeBuffer(buffer, bufferOffset, data);
        return data.byteLength;
      }
      const prepared = copyForWriteBuffer(data);
      if (prepared.copiedFromSab) {
        this._sabCopyCount += 1;
      }
      if (prepared.copiedForAlignment) {
        this._unalignedCopyCount += 1;
      }
      this.device.queue.writeBuffer(buffer, bufferOffset, prepared.bytes);
      return prepared.bytes.byteLength;
    }

    _countLiveOverflowBuffers() {
      let count = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        const slot = this._slots[i];
        count +=
          slot.overflowChunks.length +
          slot.retiredBuffers.length +
          slot.retiredSlices.length;
      }
      return count;
    }

    _countLiveRetiredSlices() {
      let count = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        count += this._slots[i].retiredSlices.length;
      }
      return count;
    }

    _countLiveSpillPages() {
      let count = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        count += this._slots[i].overflowChunks.length;
      }
      return count;
    }

    _countLiveSpillBytes() {
      let bytes = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        bytes += this._getSlotSpillBytes(this._slots[i]);
      }
      return bytes;
    }

    _countInFlightPartitions() {
      let count = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        if (this._slots[i].inFlight) {
          count += 1;
        }
      }
      return count;
    }

    _countPendingDirtyRanges() {
      let count = this._dirtyRanges ? this._dirtyRanges.length : 0;
      const slot =
        !this._destroyed && this._slots.length > 0
          ? this._slots[this._slotIndex]
          : null;
      if (slot) {
        for (let i = 0; i < slot.retiredSlices.length; i += 1) {
          const slice = slot.retiredSlices[i];
          if (slice.dirty && slice.dirtyRanges) {
            count += slice.dirtyRanges.length;
          }
        }
        for (let i = 0; i < slot.overflowChunks.length; i += 1) {
          const chunk = slot.overflowChunks[i];
          if (chunk.dirty && chunk.dirtyRanges) {
            count += chunk.dirtyRanges.length;
          }
        }
      }
      return count;
    }

    /**
     * Pad a uniform slice byte length to a 4-byte multiple and at least
     * `this.minUniformBindingSize` (default 16B).
     *
     * @param {number} byteLength
     * @returns {number}
     */
    normalizeSliceSize(byteLength) {
      return alignUniformSliceSize(byteLength, this.minUniformBindingSize);
    }

    /**
     * Begin a new frame: rotate to the next frame-in-flight partition, release transient
     * overflow/retired buffers from that recycled partition, resize the partition's GPUBuffer
     * and CPU shadow slab if the ring grew or shrank on an earlier frame, and reset the write
     * cursor to 0.
     *
     * Pass an optional preallocated `out` object as the 2nd argument to avoid per-frame
     * heap allocations.
     *
     * @param {number} [frameIndex]
     * @param {object} [out]
     * @returns {{ frameIndex: number, slotIndex: number, partitionIndex: number, buffer: GPUBuffer, byteSize: number, generation: number, framesInFlight: number }}
     */
    beginFrame(frameIndex, out) {
      this._assertNotDestroyed();
      const target = validateOutTarget(out, 'beginFrame');

      this._recordFramePeak();

      const hadPriorFrame = this._frameIndex >= 0;
      const prevSlotIndex = this._slotIndex;
      if (hadPriorFrame) {
        const prevFrameBytes =
          this._frameSliceBytes + this._framePrimaryBytes + this._frameOverflowBytes;
        this._slots[prevSlotIndex].lastFrameBytes = prevFrameBytes;
        this._evaluateAutoShrink(prevFrameBytes);
      }

      let nextFrame;
      if (frameIndex === undefined) {
        nextFrame = this._frameIndex + 1;
      } else {
        assertNonNegativeInteger(frameIndex, 'frameIndex');
        nextFrame = frameIndex;
      }

      this._frameIndex = nextFrame;
      this._slotIndex = nextFrame % this.framesInFlight;
      this._frameEnded = false;
      this._cursor = 0;
      this._writtenHighWater = 0;
      this._framePrimaryBytes = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._hasFlushedActiveSlot = false;
      this._primaryFrameDrawCount = 0;
      this._frameSliceBytes = 0;
      this._activeSliceIndex = 0;
      this._frameOverflowBytes = 0;
      this._frameDrawCount = 0;
      this._batchDepth = 0;
      releaseDirtyRanges(this._dirtyRanges, this._dirtyRangePool);
      this._dirty = false;

      if (hadPriorFrame || this._slotIndex !== prevSlotIndex) {
        this._partitionRotations += 1;
      }

      const slot = this._slots[this._slotIndex];
      if (slot.used) {
        this._wrapCount += 1;
        if (slot.endedFrameIndex >= 0) {
          slot.completedFrameIndex = Math.max(slot.completedFrameIndex, slot.endedFrameIndex);
        }
      }
      slot.used = true;
      slot.inFlight = true;
      slot.lastFrameIndex = nextFrame;
      slot.fencePromise = null;

      // This slot is now safe to recycle: destroy any overflow/retired buffers from its
      // previous frame while leaving all other in-flight slots untouched.
      this._releaseSlotTransientBuffers(slot);

      // Lazily grow or shrink this recycled slot to match the current target capacity
      // without touching any other in-flight slots.
      if (slot.byteSize !== this._capacity) {
        this._resizeSlot(slot, this._capacity, false);
      }
      if (!slot.shadow || slot.shadow.byteLength !== slot.byteSize) {
        this._setSlotShadowBuffer(slot, new Uint8Array(slot.byteSize));
      }
      this._bindSlotShadow(slot);

      if (target) {
        target.frameIndex = this._frameIndex;
        target.slotIndex = this._slotIndex;
        target.partitionIndex = this._slotIndex;
        target.buffer = slot.buffer;
        target.byteSize = slot.byteSize;
        target.generation = slot.generation;
        target.framesInFlight = this.framesInFlight;
        return target;
      }

      return {
        frameIndex: this._frameIndex,
        slotIndex: this._slotIndex,
        partitionIndex: this._slotIndex,
        buffer: slot.buffer,
        byteSize: slot.byteSize,
        generation: slot.generation,
        framesInFlight: this.framesInFlight,
      };
    }

    /**
     * Begin a coalesced write batch. Multiple `pushConstants()`, `pushCbv()`, or
     * `writeStaged()` calls inside a batch accumulate in the CPU shadow staging slab and
     * merge adjacent/overlapping dirty ranges so `endBatch()`, `flush()`, or `flushDirty()`
     * emits the minimum number of `device.queue.writeBuffer()` calls.
     *
     * Supports nesting: nested `beginBatch()` / `endBatch()` pairs defer auto-flushing
     * until the outermost batch completes.
     *
     * @param {object} [out]
     * @returns {{ batchDepth: number, frameIndex: number, slotIndex: number, cursor: number }}
     */
    beginBatch(out) {
      this._assertNotDestroyed();
      const target = validateOutTarget(out, 'beginBatch');
      this._ensureActiveFrame();
      this._batchDepth += 1;
      if (target) {
        target.batchDepth = this._batchDepth;
        target.frameIndex = this._frameIndex;
        target.slotIndex = this._slotIndex;
        target.cursor = this._cursor;
        return target;
      }
      return {
        batchDepth: this._batchDepth,
        frameIndex: this._frameIndex,
        slotIndex: this._slotIndex,
        cursor: this._cursor,
      };
    }

    /**
     * End the current write batch. When the outermost batch completes (`batchDepth` returns
     * to 0), flushes all coalesced dirty ranges in the CPU shadow staging slab unless
     * `{ autoFlush: false }` is passed.
     *
     * @param {{ autoFlush?: boolean, out?: object }} [options]
     * @param {object} [out]
     * @returns {{ uploadedBytes: number, bytesUploaded: number, drawCount: number, stagedWrites: number, writeBufferCalls: number, coalescedWrites: number, dirtyRangesFlushed: number, batchDepth: number }}
     */
    endBatch(options, out) {
      this._assertNotDestroyed();
      const target = validateOutTarget(
        out !== undefined ? out : options && options.out,
        'endBatch'
      );
      if (this._batchDepth === 0) {
        if (target) {
          target.uploadedBytes = 0;
          target.bytesUploaded = 0;
          target.drawCount = 0;
          target.stagedWrites = 0;
          target.writeBufferCalls = 0;
          target.coalescedWrites = 0;
          target.dirtyRangesFlushed = 0;
          target.batchDepth = 0;
          return target;
        }
        return {
          uploadedBytes: 0,
          bytesUploaded: 0,
          drawCount: 0,
          stagedWrites: 0,
          writeBufferCalls: 0,
          coalescedWrites: 0,
          dirtyRangesFlushed: 0,
          batchDepth: 0,
        };
      }
      this._batchDepth -= 1;
      const shouldFlush = !(options && options.autoFlush === false);
      if (this._batchDepth === 0 && shouldFlush) {
        const res = this._flushDirtyInternal(target || this._tmpFlushResult);
        if (target) {
          target.batchDepth = 0;
          return target;
        }
        return {
          uploadedBytes: res.uploadedBytes,
          bytesUploaded: res.bytesUploaded,
          drawCount: res.drawCount,
          stagedWrites: res.stagedWrites,
          writeBufferCalls: res.writeBufferCalls,
          coalescedWrites: res.coalescedWrites,
          dirtyRangesFlushed: res.dirtyRangesFlushed,
          batchDepth: 0,
        };
      }
      if (target) {
        target.uploadedBytes = 0;
        target.bytesUploaded = 0;
        target.drawCount = 0;
        target.stagedWrites = 0;
        target.writeBufferCalls = 0;
        target.coalescedWrites = 0;
        target.dirtyRangesFlushed = 0;
        target.batchDepth = this._batchDepth;
        return target;
      }
      return {
        uploadedBytes: 0,
        bytesUploaded: 0,
        drawCount: 0,
        stagedWrites: 0,
        writeBufferCalls: 0,
        coalescedWrites: 0,
        dirtyRangesFlushed: 0,
        batchDepth: this._batchDepth,
      };
    }

    /**
     * Return a defensive snapshot of currently pending coalesced dirty byte ranges in the
     * active slot's CPU shadow staging slab.
     *
     * @returns {Array<{ start: number, end: number, byteLength: number, writeCount: number }>}
     */
    getDirtyRanges() {
      this._assertNotDestroyed();
      const len = this._dirtyRanges.length;
      const out = new Array(len);
      for (let i = 0; i < len; i += 1) {
        const r = this._dirtyRanges[i];
        out[i] = {
          start: r.start,
          end: r.end,
          byteLength: r.end - r.start,
          writeCount: r.writeCount,
        };
      }
      return out;
    }

    _pushInternal(
      data,
      size,
      fromSab,
      isUnalignedInput,
      allowWrap,
      deferImmediateFlush,
      out,
      includeShadow,
      hasData
    ) {
      const sizeAligned4 = size <= 0x7ffffffc ? (size + 3) & ~3 : alignUp(size, WEBGPU_BUFFER_COPY_ALIGNMENT);
      const paddedSize =
        sizeAligned4 > this.minUniformBindingSize ? sizeAligned4 : this.minUniformBindingSize;
      if (paddedSize > this.maxBindingSize) {
        throw new RangeError(
          `pushConstants padded size (${paddedSize} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      const alignedSize = this._alignToRingFast(paddedSize);

      this._ensureActiveFrame();

      const slot = this._slots[this._slotIndex];
      const primaryDynamicOffset = this._alignToRingFast(this._cursor);
      const primaryEndOffset = primaryDynamicOffset + alignedSize;

      let targetBuffer;
      let targetShadow;
      let dynamicOffset;
      let isOverflow = false;
      let wrapped = false;
      let allocationSliceIndex = this._activeSliceIndex;
      let wastedBytes = 0;
      let createdNewBuffer = false;

      if (primaryEndOffset <= slot.byteSize) {
        wastedBytes = (primaryDynamicOffset - this._cursor) + (alignedSize - size);
        if (hasData) {
          copySourceToShadow(
            this._shadow,
            this._shadowU32,
            this._shadowI32,
            this._shadowF32,
            data,
            size,
            primaryDynamicOffset
          );
          if (alignedSize > size) {
            this._shadow.fill(0, primaryDynamicOffset + size, primaryEndOffset);
          }
        } else {
          this._shadow.fill(0, primaryDynamicOffset, primaryEndOffset);
        }
        insertDirtyRange(
          this._dirtyRanges,
          primaryDynamicOffset,
          primaryEndOffset,
          1,
          this.maxCoalesceGapBytes,
          this._dirtyRangePool
        );
        this._cursor = primaryEndOffset;
        if (primaryEndOffset > this._writtenHighWater) {
          this._writtenHighWater = primaryEndOffset;
        }
        this._framePrimaryBytes += alignedSize;
        this._primaryFrameDrawCount += 1;
        targetBuffer = slot.buffer;
        targetShadow = this._shadow;
        dynamicOffset = primaryDynamicOffset;
      } else if (allowWrap && alignedSize <= slot.byteSize) {
        // Wrap cursor cleanly to offset 0 while preserving 256-byte alignment and
        // retaining any unflushed tail dirty range for coalesced multi-range flush.
        wastedBytes = (slot.byteSize - this._cursor) + (alignedSize - size);
        if (hasData) {
          copySourceToShadow(
            this._shadow,
            this._shadowU32,
            this._shadowI32,
            this._shadowF32,
            data,
            size,
            0
          );
          if (alignedSize > size) {
            this._shadow.fill(0, size, alignedSize);
          }
        } else {
          this._shadow.fill(0, 0, alignedSize);
        }
        insertDirtyRange(
          this._dirtyRanges,
          0,
          alignedSize,
          1,
          this.maxCoalesceGapBytes,
          this._dirtyRangePool
        );
        this._cursor = alignedSize;
        if (alignedSize > this._writtenHighWater) {
          this._writtenHighWater = alignedSize;
        }
        this._framePrimaryBytes += alignedSize;
        this._primaryFrameDrawCount += 1;
        this._wrapCount += 1;
        targetBuffer = slot.buffer;
        targetShadow = this._shadow;
        dynamicOffset = 0;
        wrapped = true;
      } else if (this.multiSlice && this._growMultiSlice(slot, alignedSize)) {
        createdNewBuffer = true;
        wastedBytes = alignedSize - size;
        if (hasData) {
          copySourceToShadow(
            this._shadow,
            this._shadowU32,
            this._shadowI32,
            this._shadowF32,
            data,
            size,
            0
          );
          if (alignedSize > size) {
            this._shadow.fill(0, size, alignedSize);
          }
        } else {
          this._shadow.fill(0, 0, alignedSize);
        }
        insertDirtyRange(
          this._dirtyRanges,
          0,
          alignedSize,
          1,
          this.maxCoalesceGapBytes,
          this._dirtyRangePool
        );
        this._cursor = alignedSize;
        if (alignedSize > this._writtenHighWater) {
          this._writtenHighWater = alignedSize;
        }
        this._framePrimaryBytes += alignedSize;
        this._primaryFrameDrawCount += 1;
        targetBuffer = slot.buffer;
        targetShadow = this._shadow;
        dynamicOffset = 0;
        allocationSliceIndex = this._activeSliceIndex;
      } else if (!this.multiSlice && primaryEndOffset <= this.maxByteSize) {
        createdNewBuffer = true;
        wastedBytes = (primaryDynamicOffset - this._cursor) + (alignedSize - size);
        this._grow(primaryEndOffset);
        if (hasData) {
          copySourceToShadow(
            this._shadow,
            this._shadowU32,
            this._shadowI32,
            this._shadowF32,
            data,
            size,
            primaryDynamicOffset
          );
          if (alignedSize > size) {
            this._shadow.fill(0, primaryDynamicOffset + size, primaryEndOffset);
          }
        } else {
          this._shadow.fill(0, primaryDynamicOffset, primaryEndOffset);
        }
        insertDirtyRange(
          this._dirtyRanges,
          primaryDynamicOffset,
          primaryEndOffset,
          1,
          this.maxCoalesceGapBytes,
          this._dirtyRangePool
        );
        this._cursor = primaryEndOffset;
        if (primaryEndOffset > this._writtenHighWater) {
          this._writtenHighWater = primaryEndOffset;
        }
        this._framePrimaryBytes += alignedSize;
        this._primaryFrameDrawCount += 1;
        targetBuffer = slot.buffer;
        targetShadow = this._shadow;
        dynamicOffset = primaryDynamicOffset;
      } else if (allowWrap && !this.multiSlice && alignedSize <= this.maxByteSize) {
        createdNewBuffer = true;
        wastedBytes = alignedSize - size;
        this._grow(alignedSize);
        if (hasData) {
          copySourceToShadow(
            this._shadow,
            this._shadowU32,
            this._shadowI32,
            this._shadowF32,
            data,
            size,
            0
          );
          if (alignedSize > size) {
            this._shadow.fill(0, size, alignedSize);
          }
        } else {
          this._shadow.fill(0, 0, alignedSize);
        }
        insertDirtyRange(
          this._dirtyRanges,
          0,
          alignedSize,
          1,
          this.maxCoalesceGapBytes,
          this._dirtyRangePool
        );
        this._cursor = alignedSize;
        if (alignedSize > this._writtenHighWater) {
          this._writtenHighWater = alignedSize;
        }
        this._framePrimaryBytes += alignedSize;
        this._primaryFrameDrawCount += 1;
        this._wrapCount += 1;
        targetBuffer = slot.buffer;
        targetShadow = this._shadow;
        dynamicOffset = 0;
        wrapped = true;
      } else if (this.allowSpill || this.allowOverflow) {
        const overflowAlloc = this._allocateOverflowSlice(slot, alignedSize);
        const chunk = overflowAlloc.chunk;
        createdNewBuffer = overflowAlloc.createdNewPage;
        const prevChunkCursor = chunk.cursor;
        dynamicOffset = overflowAlloc.dynamicOffset;
        const chunkEndOffset = dynamicOffset + alignedSize;
        wastedBytes = (dynamicOffset - prevChunkCursor) + (alignedSize - size);

        if (hasData) {
          copySourceToShadow(
            chunk.shadow,
            chunk.shadowU32,
            chunk.shadowI32,
            chunk.shadowF32,
            data,
            size,
            dynamicOffset
          );
          if (alignedSize > size) {
            chunk.shadow.fill(0, dynamicOffset + size, chunkEndOffset);
          }
        } else {
          chunk.shadow.fill(0, dynamicOffset, chunkEndOffset);
        }
        insertDirtyRange(
          chunk.dirtyRanges,
          dynamicOffset,
          chunkEndOffset,
          1,
          this.maxCoalesceGapBytes,
          this._dirtyRangePool
        );
        chunk.cursor = chunkEndOffset;
        chunk.drawCount += 1;
        chunk.dirty = true;
        this._frameOverflowBytes += alignedSize;
        this._spillAllocations += 1;

        targetBuffer = chunk.buffer;
        targetShadow = chunk.shadow;
        isOverflow = true;
        allocationSliceIndex = this._activeSliceIndex + slot.overflowChunks.length;
      } else {
        const neededBytes = this.multiSlice
          ? this._getSlotPrimaryAndRetiredBytes(slot) + alignedSize
          : primaryEndOffset;
        throw new RangeError(
          `DynamicUniformRing capacity exceeded: need ${neededBytes} bytes, ` +
            `maxByteSize is ${this.maxByteSize} bytes`
        );
      }

      if (fromSab) {
        this._sabCopyCount += 1;
      }
      if (isUnalignedInput) {
        this._unalignedCopyCount += 1;
      }

      this._dirty = true;
      this._frameDrawCount += 1;
      this._totalDrawCount += 1;
      if (this._maxSlicesPerFrame < 1) {
        this._maxSlicesPerFrame = 1;
      }
      if (createdNewBuffer) {
        this._freshAllocations += 1;
      } else {
        this._reusedAllocations += 1;
      }
      this._totalBytesPushed += size;
      this._alignedBytesWasted += wastedBytes;
      this._recordFramePeak();

      if (this.immediateFlush && this._batchDepth === 0 && !deferImmediateFlush) {
        this._flushDirtyInternal();
      }

      if (out) {
        this._zeroAllocAllocs += 1;
        out.buffer = targetBuffer;
        out.offset = dynamicOffset;
        out.dynamicOffset = dynamicOffset;
        out.size = size;
        out.paddedSize = paddedSize;
        out.alignedSize = alignedSize;
        out.overflow = isOverflow;
        out.isSpill = isOverflow;
        out.wrapped = wrapped;
        out.sliceIndex = allocationSliceIndex;
        if (includeShadow) {
          out.shadow = targetShadow;
        }
        return out;
      }

      if (includeShadow) {
        return {
          buffer: targetBuffer,
          offset: dynamicOffset,
          dynamicOffset,
          size,
          paddedSize,
          alignedSize,
          shadow: targetShadow,
          overflow: isOverflow,
          isSpill: isOverflow,
          wrapped,
          sliceIndex: allocationSliceIndex,
        };
      }

      return {
        buffer: targetBuffer,
        offset: dynamicOffset,
        dynamicOffset,
        size,
        paddedSize,
        alignedSize,
        overflow: isOverflow,
        isSpill: isOverflow,
        wrapped,
        sliceIndex: allocationSliceIndex,
      };
    }

    /**
     * Zero-allocation hot-path uniform slice reservation aligned to `alignment` (default 256B).
     * Pass a reusable `out` object (directly as the 2nd argument or via `{ out }`) to avoid
     * per-call heap allocations. Returns `{ buffer, offset, dynamicOffset, size, paddedSize,
     * alignedSize, shadow, overflow, isSpill, wrapped, sliceIndex }` with `shadow` zero-filled
     * over `[offset, offset + alignedSize)`.
     *
     * @param {number} byteLength
     * @param {object} [outOrOptions]
     */
    allocate(byteLength, outOrOptions) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(byteLength, 'byteLength');

      let allowEmpty = this.allowEmptyPayloads;
      let allowWrap = this.allowWrap;
      let deferImmediateFlush = true;
      let out = null;

      if (outOrOptions !== undefined && outOrOptions !== null) {
        if (typeof outOrOptions !== 'object') {
          throw new TypeError('allocate out/options argument must be an object');
        }
        if (isOptionsBag(outOrOptions)) {
          if (outOrOptions.allowEmpty === true) {
            allowEmpty = true;
          }
          if (outOrOptions.wrap === true) {
            allowWrap = true;
          }
          if (outOrOptions._deferImmediateFlush === false) {
            deferImmediateFlush = false;
          }
          out = validateOutTarget(outOrOptions.out, 'allocate');
        } else {
          out = outOrOptions;
        }
      }

      if (byteLength === 0 && !allowEmpty) {
        throw new RangeError('allocate byteLength must not be empty (got 0 bytes)');
      }
      if (byteLength > this.maxBindingSize) {
        throw new RangeError(
          `allocate size (${byteLength} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      return this._pushInternal(
        EMPTY_UINT8,
        byteLength,
        false,
        false,
        allowWrap,
        deferImmediateFlush,
        out,
        true,
        false
      );
    }

    /**
     * Zero-allocation hot-path aligned uniform/CBV allocation (`alignment`-aligned, default 256B).
     * Writes directly into the caller-supplied `out` object without inspecting option-bag keys.
     *
     * @param {number} byteLength
     * @param {object} out Pre-allocated scratch result object.
     * @param {{ allowEmpty?: boolean, wrap?: boolean }} [options]
     */
    allocateInto(byteLength, out, options) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(byteLength, 'byteLength');
      if (!out || typeof out !== 'object') {
        throw new TypeError('allocateInto requires a preallocated out object');
      }
      const allowEmpty =
        (options && options.allowEmpty === true) || this.allowEmptyPayloads;
      const allowWrap = (options && options.wrap === true) || this.allowWrap;
      const deferImmediateFlush = !(options && options._deferImmediateFlush === false);

      if (byteLength === 0 && !allowEmpty) {
        throw new RangeError('allocate byteLength must not be empty (got 0 bytes)');
      }
      if (byteLength > this.maxBindingSize) {
        throw new RangeError(
          `allocate size (${byteLength} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      return this._pushInternal(
        EMPTY_UINT8,
        byteLength,
        false,
        false,
        allowWrap,
        deferImmediateFlush,
        out,
        true,
        false
      );
    }

    /**
     * Zero-allocation hot-path aligned uniform/CBV allocation (`alignment`-aligned, default 256B).
     * Accepts a reusable `out` scratch object directly (`allocAligned(byteLength, out)`) or via
     * options (`allocAligned(byteLength, { out, allowEmpty, wrap })`) to eliminate per-draw GC
     * churn while preserving the exact default return shape when `out` is omitted.
     *
     * @param {number} byteLength
     * @param {object} [outOrOptions]
     */
    allocAligned(byteLength, outOrOptions) {
      return this.allocate(byteLength, outOrOptions);
    }

    /**
     * Zero-allocation hot-path `pushConstants` writing directly into `out` without
     * allocating an options wrapper object.
     *
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer | null | undefined} data
     * @param {object} out Pre-allocated scratch result object.
     * @param {{ allowEmpty?: boolean, wrap?: boolean }} [options]
     */
    pushConstantsInto(data, out, options) {
      this._assertNotDestroyed();
      if (!out || typeof out !== 'object') {
        throw new TypeError('pushConstantsInto requires a preallocated out object');
      }
      const allowEmpty =
        (options && options.allowEmpty === true) || this.allowEmptyPayloads;
      const allowWrap = (options && options.wrap === true) || this.allowWrap;
      const deferImmediateFlush = Boolean(options && options._deferImmediateFlush === true);
      return this._pushConstantsWithFlags(data, allowEmpty, allowWrap, deferImmediateFlush, out);
    }

    /**
     * Zero-allocation hot-path `pushCbv` writing directly into `out` without
     * allocating an options wrapper object.
     *
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer | null | undefined} data
     * @param {object} out Pre-allocated scratch result object.
     * @param {{ wrap?: boolean }} [options]
     */
    pushCbvInto(data, out, options) {
      this._assertNotDestroyed();
      if (!out || typeof out !== 'object') {
        throw new TypeError('pushCbvInto requires a preallocated out object');
      }
      const allowWrap = (options && options.wrap === true) || this.allowWrap;
      return this._pushConstantsWithFlags(data, true, allowWrap, false, out);
    }

    /**
     * Zero-allocation helper returning the ring's reusable 1-element `Uint32Array` populated
     * with `dynamicOffset` for synchronous `pass.setBindGroup(group, bg, dynamicOffsets)`.
     *
     * @param {number} dynamicOffset
     * @returns {Uint32Array}
     */
    getDynamicOffsetArray(dynamicOffset) {
      this._singleDynamicOffset[0] = dynamicOffset >>> 0;
      return this._singleDynamicOffset;
    }

    /**
     * Write `dynamicOffset` into a caller-provided array/TypedArray (or the ring's reusable
     * 1-element `Uint32Array` when `out` is omitted) at index `index` (default 0).
     *
     * @param {number} dynamicOffset
     * @param {Uint32Array | number[]} [out]
     * @param {number} [index=0]
     * @returns {Uint32Array | number[]}
     */
    fillDynamicOffsets(dynamicOffset, out, index = 0) {
      const target = out || this._singleDynamicOffset;
      target[index] = dynamicOffset >>> 0;
      return target;
    }

    /**
     * Zero-allocation hot-path uniform write.
     * Supports:
     * 1. `write(data, outOrOptions?)` — allocates and copies `data` into the ring aligned to
     *    `alignment` (default 256B); pass a reusable `out` object to avoid heap allocation.
     * 2. `write(offset, data, options?)` — writes into an explicit aligned offset in the
     *    active slot's CPU shadow slab (delegates to `writeStaged`).
     *
     * @param {number | ArrayBufferView | ArrayBuffer | SharedArrayBuffer | null | undefined} offsetOrData
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer | object | null | undefined} [dataOrOutOrOptions]
     * @param {object} [maybeOptions]
     */
    write(offsetOrData, dataOrOutOrOptions, maybeOptions) {
      this._assertNotDestroyed();
      if (typeof offsetOrData === 'number') {
        return this.writeStaged(offsetOrData, dataOrOutOrOptions, maybeOptions);
      }
      if (
        dataOrOutOrOptions !== undefined &&
        dataOrOutOrOptions !== null &&
        typeof dataOrOutOrOptions === 'object' &&
        !isOptionsBag(dataOrOutOrOptions)
      ) {
        return this._pushConstantsWithFlags(
          offsetOrData,
          this.allowEmptyPayloads,
          this.allowWrap,
          false,
          dataOrOutOrOptions
        );
      }
      return this.pushConstants(offsetOrData, dataOrOutOrOptions);
    }

    _pushConstantsWithFlags(data, allowEmpty, allowWrap, deferImmediateFlush, out) {
      let srcData = data;
      let size = 0;
      let fromSab = false;
      let isUnalignedInput = false;

      if ((data === null || data === undefined) && allowEmpty) {
        srcData = EMPTY_UINT8;
        size = 0;
      } else if (ArrayBuffer.isView(data)) {
        size = data.byteLength;
        if (size > 0) {
          fromSab = isSharedArrayBuffer(data.buffer);
          isUnalignedInput =
            (data.byteOffset & 3) !== 0 || (size & 3) !== 0;
        }
      } else if (isArrayBuffer(data)) {
        size = data.byteLength;
        isUnalignedInput = size > 0 && (size & 3) !== 0;
      } else if (isSharedArrayBuffer(data)) {
        size = data.byteLength;
        fromSab = size > 0;
        isUnalignedInput = size > 0 && (size & 3) !== 0;
      } else {
        throw new TypeError(
          'pushConstants expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer'
        );
      }

      if (size === 0 && !allowEmpty) {
        throw new RangeError('pushConstants data must not be empty (got 0 bytes)');
      }
      if (size > this.maxBindingSize) {
        throw new RangeError(
          `pushConstants size (${size} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      return this._pushInternal(
        srcData,
        size,
        fromSab,
        isUnalignedInput,
        allowWrap,
        deferImmediateFlush,
        out,
        false,
        true
      );
    }

    /**
     * Append a per-draw constant payload into the CPU shadow ring aligned to `alignment`
     * (at least `device.limits.minUniformBufferOffsetAlignment`, default 256B).
     * Auto-grows capacity (doubling up to `maxByteSize`) when needed, and optionally
     * allocates transient per-frame overflow buffers when `allowOverflow` is enabled.
     *
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer | null | undefined} data
     * @param {{ allowEmpty?: boolean, wrap?: boolean, out?: object }} [pushOptions]
     * @returns {{ buffer: GPUBuffer, dynamicOffset: number, size: number, paddedSize: number, alignedSize: number, overflow: boolean }}
     */
    pushConstants(data, pushOptions) {
      this._assertNotDestroyed();

      const allowEmpty =
        (pushOptions && pushOptions.allowEmpty === true) || this.allowEmptyPayloads;
      const allowWrap =
        (pushOptions && pushOptions.wrap === true) || this.allowWrap;
      const deferImmediateFlush = Boolean(
        pushOptions && pushOptions._deferImmediateFlush === true
      );
      const out = pushOptions
        ? validateOutTarget(pushOptions.out, 'pushConstants')
        : null;

      return this._pushConstantsWithFlags(
        data,
        allowEmpty,
        allowWrap,
        deferImmediateFlush,
        out
      );
    }

    /**
     * Stage uniform constant bytes into the CPU shadow slab and record their dirty byte
     * range for coalesced upload on `flush()` / `flushDirty()` / `endBatch()`.
     *
     * Supports two call forms:
     * 1. `writeStaged(data, options?)` — allocates the next `alignment`-aligned (default 256B)
     *    slice in the ring (wrapping to offset 0 when `allowWrap` or `options.wrap` is true).
     * 2. `writeStaged(offset, data, options?)` — writes directly at an explicit
     *    `alignment`-aligned (or 4B-aligned when `options.allowSubAlignment` is true) byte
     *    offset in the active slot's CPU shadow slab.
     *
     * Unlike `pushConstants()` under `immediateFlush: true`, `writeStaged()` never auto-flushes
     * before `flushDirty()` / `flush()` / `endBatch()`.
     *
     * @param {number | ArrayBufferView | ArrayBuffer | SharedArrayBuffer | null | undefined} offsetOrData
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer | object | null | undefined} [dataOrOptions]
     * @param {{ allowEmpty?: boolean, wrap?: boolean, allowSubAlignment?: boolean, out?: object }} [maybeOptions]
     */
    writeStaged(offsetOrData, dataOrOptions, maybeOptions) {
      this._assertNotDestroyed();

      if (typeof offsetOrData !== 'number') {
        const opts = dataOrOptions && typeof dataOrOptions === 'object' ? dataOrOptions : null;
        const allowEmpty = (opts && opts.allowEmpty === true) || this.allowEmptyPayloads;
        const allowWrap = (opts && opts.wrap === true) || this.allowWrap;
        const out = opts ? validateOutTarget(opts.out, 'writeStaged') : null;
        const alloc = this._pushConstantsWithFlags(
          offsetOrData,
          allowEmpty,
          allowWrap,
          true,
          out
        );
        alloc.staged = true;
        return alloc;
      }

      const offset = offsetOrData;
      const data = dataOrOptions;
      const options = maybeOptions && typeof maybeOptions === 'object' ? maybeOptions : null;
      assertNonNegativeInteger(offset, 'offset');

      const allowSubAlignment = Boolean(options && options.allowSubAlignment === true);
      const requiredOffsetAlign = allowSubAlignment
        ? WEBGPU_BUFFER_COPY_ALIGNMENT
        : this.alignment;
      if (offset % requiredOffsetAlign !== 0) {
        this._alignmentViolationsRejected += 1;
        throw new RangeError(
          `writeStaged offset (${offset}) must be aligned to ${requiredOffsetAlign} bytes`
        );
      }

      const allowEmpty =
        (options && options.allowEmpty === true) || this.allowEmptyPayloads;
      const out = options ? validateOutTarget(options.out, 'writeStaged') : null;

      let srcData = data;
      let size = 0;
      let fromSab = false;
      let isUnalignedInput = false;

      if ((data === null || data === undefined) && allowEmpty) {
        srcData = EMPTY_UINT8;
        size = 0;
      } else if (ArrayBuffer.isView(data)) {
        size = data.byteLength;
        if (size > 0) {
          fromSab = isSharedArrayBuffer(data.buffer);
          isUnalignedInput =
            (data.byteOffset & 3) !== 0 || (size & 3) !== 0;
        }
      } else if (isArrayBuffer(data)) {
        size = data.byteLength;
        isUnalignedInput = size > 0 && (size & 3) !== 0;
      } else if (isSharedArrayBuffer(data)) {
        size = data.byteLength;
        fromSab = size > 0;
        isUnalignedInput = size > 0 && (size & 3) !== 0;
      } else {
        throw new TypeError(
          'writeStaged expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer'
        );
      }

      if (size === 0 && !allowEmpty) {
        throw new RangeError('writeStaged data must not be empty (got 0 bytes)');
      }
      if (size > this.maxBindingSize) {
        throw new RangeError(
          `writeStaged size (${size} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      const paddedSize = Math.max(
        this.minUniformBindingSize,
        alignUp(size, WEBGPU_BUFFER_COPY_ALIGNMENT)
      );
      if (paddedSize > this.maxBindingSize) {
        throw new RangeError(
          `writeStaged padded size (${paddedSize} bytes) exceeds maxBindingSize (${this.maxBindingSize} bytes)`
        );
      }

      const alignedSize = alignUp(paddedSize, this.alignment);
      const writeSpan = allowSubAlignment ? paddedSize : alignedSize;

      this._ensureActiveFrame();

      const slot = this._slots[this._slotIndex];
      const endOffset = offset + writeSpan;
      if (endOffset > slot.byteSize) {
        throw new RangeError(
          `writeStaged range [${offset}, ${endOffset}) exceeds active slot capacity (${slot.byteSize} bytes)`
        );
      }

      copySourceToShadow(
        this._shadow,
        this._shadowU32,
        this._shadowI32,
        this._shadowF32,
        srcData,
        size,
        offset
      );
      if (writeSpan > size) {
        this._shadow.fill(0, offset + size, endOffset);
      }

      insertDirtyRange(
        this._dirtyRanges,
        offset,
        endOffset,
        1,
        this.maxCoalesceGapBytes,
        this._dirtyRangePool
      );

      if (endOffset > this._cursor) {
        this._cursor = endOffset;
      }
      if (endOffset > this._writtenHighWater) {
        this._writtenHighWater = endOffset;
      }
      this._framePrimaryBytes += writeSpan;
      this._primaryFrameDrawCount += 1;

      if (fromSab) {
        this._sabCopyCount += 1;
      }
      if (isUnalignedInput) {
        this._unalignedCopyCount += 1;
      }

      this._dirty = true;
      this._frameDrawCount += 1;
      this._totalDrawCount += 1;
      this._reusedAllocations += 1;
      this._totalBytesPushed += size;
      this._alignedBytesWasted += writeSpan - size;
      this._recordFramePeak();

      if (out) {
        this._zeroAllocAllocs += 1;
        out.buffer = slot.buffer;
        out.offset = offset;
        out.dynamicOffset = offset;
        out.size = size;
        out.paddedSize = paddedSize;
        out.alignedSize = alignedSize;
        out.overflow = false;
        out.isSpill = false;
        out.wrapped = false;
        out.staged = true;
        out.sliceIndex = this._activeSliceIndex;
        return out;
      }

      return {
        buffer: slot.buffer,
        offset,
        dynamicOffset: offset,
        size,
        paddedSize,
        alignedSize,
        overflow: false,
        isSpill: false,
        wrapped: false,
        staged: true,
        sliceIndex: this._activeSliceIndex,
      };
    }

    /**
     * Convenience wrapper for CBV uploads (including empty or <16B CBVs), automatically
     * padding to a 4-byte multiple of at least `minUniformBindingSize` (16B) and
     * aligning dynamic offsets to `alignment` (default 256B).
     *
     * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer | null | undefined} data
     * @param {{ wrap?: boolean, out?: object }} [pushOptions]
     * @returns {{ buffer: GPUBuffer, dynamicOffset: number, size: number, paddedSize: number, alignedSize: number, overflow: boolean }}
     */
    pushCbv(data, pushOptions) {
      this._assertNotDestroyed();
      const allowWrap =
        (pushOptions && pushOptions.wrap === true) || this.allowWrap;
      const out = pushOptions ? validateOutTarget(pushOptions.out, 'pushCbv') : null;
      return this._pushConstantsWithFlags(data, true, allowWrap, false, out);
    }

    _flushDirtyInternal(out = null) {
      this._assertNotDestroyed();

      const slot = this._slots[this._slotIndex];
      let hasRetiredSlicesDirty = false;
      for (let i = 0; i < slot.retiredSlices.length; i += 1) {
        const slice = slot.retiredSlices[i];
        if (slice.dirty && slice.dirtyRanges && slice.dirtyRanges.length > 0) {
          hasRetiredSlicesDirty = true;
          break;
        }
      }
      const hasPrimaryDirty = this._dirtyRanges.length > 0;
      let hasOverflowDirty = false;
      for (let i = 0; i < slot.overflowChunks.length; i += 1) {
        const chunk = slot.overflowChunks[i];
        if (chunk.dirty && chunk.dirtyRanges && chunk.dirtyRanges.length > 0) {
          hasOverflowDirty = true;
          break;
        }
      }

      if (!this._dirty || (!hasRetiredSlicesDirty && !hasPrimaryDirty && !hasOverflowDirty)) {
        if (out) {
          out.uploadedBytes = 0;
          out.bytesUploaded = 0;
          out.drawCount = 0;
          out.stagedWrites = 0;
          out.writeBufferCalls = 0;
          out.coalescedWrites = 0;
          out.dirtyRangesFlushed = 0;
          return out;
        }
        return {
          uploadedBytes: 0,
          bytesUploaded: 0,
          drawCount: 0,
          stagedWrites: 0,
          writeBufferCalls: 0,
          coalescedWrites: 0,
          dirtyRangesFlushed: 0,
        };
      }

      if (!this.device.queue || typeof this.device.queue.writeBuffer !== 'function') {
        throw new TypeError(
          'DynamicUniformRing.flush() requires device.queue.writeBuffer()'
        );
      }

      let uploadedBytes = 0;
      let stagedWrites = 0;
      let writeBufferCalls = 0;
      let coalescedWrites = 0;

      for (let i = 0; i < slot.retiredSlices.length; i += 1) {
        const slice = slot.retiredSlices[i];
        if (!slice.dirty || !slice.dirtyRanges || slice.dirtyRanges.length === 0) {
          continue;
        }
        for (let rIdx = 0; rIdx < slice.dirtyRanges.length; rIdx += 1) {
          const range = slice.dirtyRanges[rIdx];
          const rangeBytes = range.end - range.start;
          const sliceView =
            range.start === 0 && range.end === slice.shadow.byteLength
              ? slice.shadow
              : slice.shadow.subarray(range.start, range.end);

          this._safeWriteBuffer(slice.buffer, range.start, sliceView);

          writeBufferCalls += 1;
          stagedWrites += range.writeCount;
          coalescedWrites += Math.max(0, range.writeCount - 1);
          uploadedBytes += rangeBytes;
        }
        releaseDirtyRanges(slice.dirtyRanges, this._dirtyRangePool);
        slice.flushedCursor = slice.cursor;
        slice.flushedDrawCount = slice.drawCount;
        slice.dirty = false;
        slice.shadow = null;
      }

      if (hasPrimaryDirty) {
        for (let rIdx = 0; rIdx < this._dirtyRanges.length; rIdx += 1) {
          const range = this._dirtyRanges[rIdx];
          const rangeBytes = range.end - range.start;
          const shadowSlice = this._getSlotShadowSlice(slot, range.start, range.end);

          this._safeWriteBuffer(slot.buffer, range.start, shadowSlice);

          writeBufferCalls += 1;
          stagedWrites += range.writeCount;
          coalescedWrites += Math.max(0, range.writeCount - 1);
          uploadedBytes += rangeBytes;
        }
        releaseDirtyRanges(this._dirtyRanges, this._dirtyRangePool);
        this._flushedCursor = this._cursor;
        this._flushedDrawCount = this._primaryFrameDrawCount;
        this._hasFlushedActiveSlot = true;
      }

      for (let i = 0; i < slot.overflowChunks.length; i += 1) {
        const chunk = slot.overflowChunks[i];
        if (!chunk.dirty || !chunk.dirtyRanges || chunk.dirtyRanges.length === 0) {
          continue;
        }
        for (let rIdx = 0; rIdx < chunk.dirtyRanges.length; rIdx += 1) {
          const range = chunk.dirtyRanges[rIdx];
          const rangeBytes = range.end - range.start;
          const chunkSlice =
            range.start === 0 && range.end === chunk.shadow.byteLength
              ? chunk.shadow
              : chunk.shadow.subarray(range.start, range.end);

          this._safeWriteBuffer(chunk.buffer, range.start, chunkSlice);

          writeBufferCalls += 1;
          stagedWrites += range.writeCount;
          coalescedWrites += Math.max(0, range.writeCount - 1);
          uploadedBytes += rangeBytes;
        }
        releaseDirtyRanges(chunk.dirtyRanges, this._dirtyRangePool);
        chunk.flushedCursor = chunk.cursor;
        chunk.flushedDrawCount = chunk.drawCount;
        chunk.dirty = false;
      }

      this._dirty = false;
      this._flushCount += 1;
      this._totalBytesUploaded += uploadedBytes;
      this._writeBufferCalls += writeBufferCalls;
      this._coalescedWrites += coalescedWrites;
      this._stagedWritesFlushed += stagedWrites;

      if (out) {
        out.uploadedBytes = uploadedBytes;
        out.bytesUploaded = uploadedBytes;
        out.drawCount = stagedWrites;
        out.stagedWrites = stagedWrites;
        out.writeBufferCalls = writeBufferCalls;
        out.coalescedWrites = coalescedWrites;
        out.dirtyRangesFlushed = writeBufferCalls;
        return out;
      }

      return {
        uploadedBytes,
        bytesUploaded: uploadedBytes,
        drawCount: stagedWrites,
        stagedWrites,
        writeBufferCalls,
        coalescedWrites,
        dirtyRangesFlushed: writeBufferCalls,
      };
    }

    /**
     * Upload all coalesced dirty ranges in the CPU shadow staging slab (plus any dirty
     * retired slices or per-frame overflow buffers) to GPU memory using the minimum number
     * of regular `ArrayBuffer`-backed, 4-byte-aligned `device.queue.writeBuffer(...)` calls,
     * returning full coalescing telemetry for this flush.
     *
     * Pass an optional preallocated `out` object to avoid per-flush heap allocation.
     *
     * @param {object} [out]
     * @returns {{ uploadedBytes: number, bytesUploaded: number, drawCount: number, stagedWrites: number, writeBufferCalls: number, coalescedWrites: number, dirtyRangesFlushed: number }}
     */
    flushDirty(out) {
      const target = validateOutTarget(out, 'flushDirty');
      return this._flushDirtyInternal(target);
    }

    /**
     * Upload all coalesced dirty CPU shadow ranges for the current frame (plus any dirty
     * retired slices or per-frame overflow buffers) to GPU memory using regular
     * `ArrayBuffer`-backed, 4-byte-aligned `device.queue.writeBuffer(...)` calls.
     *
     * Preserves the legacy `{ uploadedBytes, drawCount }` return contract for existing
     * callers; pass an optional preallocated `out` object for zero-allocation hot-loop flushes.
     *
     * @param {object} [out]
     * @returns {{ uploadedBytes: number, drawCount: number }}
     */
    flush(out) {
      const target = validateOutTarget(out, 'flush');
      if (target) {
        const res = this._flushDirtyInternal(target);
        return res;
      }
      const res = this._flushDirtyInternal(this._tmpFlushResult);
      return {
        uploadedBytes: res.uploadedBytes,
        drawCount: res.drawCount,
      };
    }

    _reclaimSpillPagesInternal(slotIndex, out) {
      let reclaimedPages = 0;
      let reclaimedBytes = 0;
      const start = slotIndex !== undefined ? slotIndex : 0;
      const end = slotIndex !== undefined ? slotIndex + 1 : this._slots.length;

      for (let i = start; i < end; i += 1) {
        const slot = this._slots[i];
        if (!slot || !slot.overflowChunks || slot.overflowChunks.length === 0) {
          continue;
        }
        for (let j = 0; j < slot.overflowChunks.length; j += 1) {
          const chunk = slot.overflowChunks[j];
          reclaimedPages += 1;
          reclaimedBytes += chunk.byteSize;
          this._destroyRawBuffer(chunk.buffer, chunk.byteSize);
          if (chunk.dirtyRanges && chunk.dirtyRanges.length > 0) {
            releaseDirtyRanges(chunk.dirtyRanges, this._dirtyRangePool);
          }
          chunk.buffer = null;
          chunk.shadow = null;
          chunk.shadowU32 = null;
          chunk.shadowI32 = null;
          chunk.shadowF32 = null;
        }
        slot.overflowChunks.length = 0;
      }

      if (reclaimedPages > 0) {
        const activeSlot = this._slots[this._slotIndex];
        let activeHasDirtySpill = false;
        if (activeSlot && activeSlot.overflowChunks) {
          for (let i = 0; i < activeSlot.overflowChunks.length; i += 1) {
            const chunk = activeSlot.overflowChunks[i];
            if (chunk.dirty && chunk.dirtyRanges && chunk.dirtyRanges.length > 0) {
              activeHasDirtySpill = true;
              break;
            }
          }
        }
        let activeHasDirtyRetiredSlice = false;
        if (activeSlot && activeSlot.retiredSlices) {
          for (let i = 0; i < activeSlot.retiredSlices.length; i += 1) {
            const slice = activeSlot.retiredSlices[i];
            if (slice.dirty && slice.dirtyRanges && slice.dirtyRanges.length > 0) {
              activeHasDirtyRetiredSlice = true;
              break;
            }
          }
        }
        if (
          this._dirtyRanges.length === 0 &&
          !activeHasDirtySpill &&
          !activeHasDirtyRetiredSlice
        ) {
          this._dirty = false;
        }
      }

      if (out) {
        out.reclaimedPages = reclaimedPages;
        out.reclaimedBytes = reclaimedBytes;
        return out;
      }
      return { reclaimedPages, reclaimedBytes };
    }

    /**
     * Reclaim and destroy auxiliary spill-page GPUBuffers across all slots (or a single
     * specified `slotIndex`). Safe and idempotent to call on quiet frames or after GPU work
     * completes.
     *
     * @param {number} [slotIndex]
     * @param {object} [out]
     * @returns {{ reclaimedPages: number, reclaimedBytes: number }}
     */
    reclaimSpillPages(slotIndex, out) {
      this._assertNotDestroyed();
      const target = validateOutTarget(out, 'reclaimSpillPages');
      if (slotIndex !== undefined) {
        assertNonNegativeInteger(slotIndex, 'slotIndex');
        if (slotIndex >= this.framesInFlight) {
          throw new RangeError(
            `slotIndex (${slotIndex}) out of range [0, ${this.framesInFlight - 1}]`
          );
        }
      }
      return this._reclaimSpillPagesInternal(slotIndex, target);
    }

    /**
     * Complete the active (or specified `frameIndex`) frame, updating peak frame utilization
     * telemetry, recording partition fence tracking so in-flight GPU reads are never clobbered,
     * and reclaiming that partition's spill pages unless `{ deferReclaim: true }` or
     * `{ reclaimSpillPages: false }` is passed.
     *
     * Supports:
     * - `endFrame()`
     * - `endFrame(options)`
     * - `endFrame(frameIndex, options?)`
     *
     * @param {number | { deferReclaim?: boolean, reclaimSpillPages?: boolean, fence?: Promise<any>, fencePromise?: Promise<any>, out?: object }} [frameIndexOrOptions]
     * @param {{ deferReclaim?: boolean, reclaimSpillPages?: boolean, fence?: Promise<any>, fencePromise?: Promise<any>, out?: object }} [maybeOptions]
     * @returns {{ frameIndex: number, slotIndex: number, partitionIndex: number, frameBytes: number, reclaimedPages: number, reclaimedBytes: number, inFlight: boolean }}
     */
    endFrame(frameIndexOrOptions, maybeOptions) {
      this._assertNotDestroyed();
      this._recordFramePeak();

      let resolvedFrameIndex;
      let options = null;
      if (typeof frameIndexOrOptions === 'number') {
        assertNonNegativeInteger(frameIndexOrOptions, 'frameIndex');
        resolvedFrameIndex = frameIndexOrOptions;
        if (maybeOptions !== undefined && maybeOptions !== null) {
          if (typeof maybeOptions !== 'object') {
            throw new TypeError('endFrame options argument must be an object');
          }
          options = maybeOptions;
        }
      } else if (frameIndexOrOptions === undefined || frameIndexOrOptions === null) {
        resolvedFrameIndex = this._frameIndex;
      } else if (typeof frameIndexOrOptions === 'object') {
        resolvedFrameIndex = this._frameIndex;
        options = frameIndexOrOptions;
      } else {
        throw new TypeError('endFrame expects an optional frameIndex integer or options object');
      }

      const fence = options
        ? options.fence !== undefined
          ? options.fence
          : options.fencePromise
        : undefined;
      if (fence !== undefined && fence !== null && typeof fence.then !== 'function') {
        throw new TypeError('endFrame fence option must be a Promise/thenable, null, or undefined');
      }
      const target = options ? validateOutTarget(options.out, 'endFrame') : null;

      const targetSlotIndex =
        resolvedFrameIndex >= 0
          ? resolvedFrameIndex % this.framesInFlight
          : this._slotIndex;
      const slot = this._slots[targetSlotIndex];
      const isActiveSlot = targetSlotIndex === this._slotIndex;
      const frameBytes = isActiveSlot
        ? this._frameSliceBytes + this._framePrimaryBytes + this._frameOverflowBytes
        : slot.lastFrameBytes;
      if (isActiveSlot) {
        slot.lastFrameBytes = frameBytes;
        if (resolvedFrameIndex === this._frameIndex || this._frameIndex < 0) {
          this._frameEnded = true;
        }
      }

      if (resolvedFrameIndex >= 0) {
        slot.endedFrameIndex = resolvedFrameIndex;
        if (slot.lastFrameIndex < 0) {
          slot.lastFrameIndex = resolvedFrameIndex;
        }
        slot.inFlight = true;
        if (fence) {
          slot.fencePromise = fence;
          const trackedFrame = resolvedFrameIndex;
          fence.then(
            () => {
              if (
                !this._destroyed &&
                slot.lastFrameIndex === trackedFrame &&
                slot.endedFrameIndex === trackedFrame
              ) {
                slot.completedFrameIndex = Math.max(slot.completedFrameIndex, trackedFrame);
                slot.inFlight = false;
                slot.fencePromise = null;
              }
            },
            () => {
              if (
                !this._destroyed &&
                slot.lastFrameIndex === trackedFrame &&
                slot.endedFrameIndex === trackedFrame
              ) {
                slot.inFlight = false;
                slot.fencePromise = null;
              }
            }
          );
        }
      }

      const shouldReclaim =
        !(options && (options.deferReclaim === true || options.reclaimSpillPages === false));

      let reclaimedPages = 0;
      let reclaimedBytes = 0;
      if (shouldReclaim && slot.overflowChunks && slot.overflowChunks.length > 0) {
        const res = this._reclaimSpillPagesInternal(targetSlotIndex, this._tmpReclaimResult);
        reclaimedPages = res.reclaimedPages;
        reclaimedBytes = res.reclaimedBytes;
      }

      if (target) {
        target.frameIndex = resolvedFrameIndex;
        target.slotIndex = targetSlotIndex;
        target.partitionIndex = targetSlotIndex;
        target.frameBytes = frameBytes;
        target.reclaimedPages = reclaimedPages;
        target.reclaimedBytes = reclaimedBytes;
        target.inFlight = slot.inFlight;
        return target;
      }

      return {
        frameIndex: resolvedFrameIndex,
        slotIndex: targetSlotIndex,
        partitionIndex: targetSlotIndex,
        frameBytes,
        reclaimedPages,
        reclaimedBytes,
        inFlight: slot.inFlight,
      };
    }

    /**
     * Reset per-frame write state and destroy all live spill pages and retired buffers across
     * all slots while preserving primary slot GPUBuffers and lifetime telemetry counters.
     *
     * @param {object} [out]
     * @returns {{ reclaimedPages: number, reclaimedBytes: number }}
     */
    reset(out) {
      this._assertNotDestroyed();
      const target = validateOutTarget(out, 'reset');
      this._recordFramePeak();

      const res = this._reclaimSpillPagesInternal(undefined, this._tmpReclaimResult);
      const reclaimedPages = res.reclaimedPages;
      const reclaimedBytes = res.reclaimedBytes;
      for (let i = 0; i < this._slots.length; i += 1) {
        const slot = this._slots[i];
        this._releaseSlotTransientBuffers(slot);
        slot.inFlight = false;
        slot.fencePromise = null;
        slot.lastFrameBytes = 0;
      }

      this._frameEnded = false;
      this._cursor = 0;
      this._writtenHighWater = 0;
      this._framePrimaryBytes = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._hasFlushedActiveSlot = false;
      this._primaryFrameDrawCount = 0;
      this._frameSliceBytes = 0;
      this._activeSliceIndex = 0;
      this._frameOverflowBytes = 0;
      this._frameDrawCount = 0;
      this._batchDepth = 0;
      releaseDirtyRanges(this._dirtyRanges, this._dirtyRangePool);
      this._dirty = false;
      this._consecutiveIdleFrames = 0;
      this._idleWindowHighWaterBytes = 0;

      if (target) {
        target.reclaimedPages = reclaimedPages;
        target.reclaimedBytes = reclaimedBytes;
        return target;
      }
      return { reclaimedPages, reclaimedBytes };
    }

    /**
     * Build a WebGPU `GPUBindGroupLayoutEntry` for this dynamic uniform buffer
     * (`type: 'uniform'`, `hasDynamicOffset: true`).
     *
     * @param {number} binding
     * @param {number} [visibility]
     * @param {number} [minBindingSize=0]
     */
    createBindGroupLayoutEntry(
      binding,
      visibility = DEFAULT_SHADER_VISIBILITY,
      minBindingSize = 0
    ) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(binding, 'binding');
      assertPositiveInteger(visibility, 'visibility');
      assertNonNegativeInteger(minBindingSize, 'minBindingSize');
      if (minBindingSize > this.maxBindingSize) {
        throw new RangeError(
          `minBindingSize (${minBindingSize}) exceeds maxBindingSize (${this.maxBindingSize})`
        );
      }

      return {
        binding,
        visibility,
        buffer: {
          type: 'uniform',
          hasDynamicOffset: true,
          minBindingSize,
        },
      };
    }

    /**
     * Build a WebGPU `GPUBindGroupEntry` referencing the active (or specified) frame slot's
     * uniform buffer at `offset: 0` with a 4-byte-aligned dynamic window size of at least
     * `minUniformBindingSize` (16B).
     *
     * @param {number} binding
     * @param {number} [sliceSize]
     * @param {number} [slotIndex]
     */
    createBindGroupEntry(binding, sliceSize, slotIndex = this._slotIndex, out) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(binding, 'binding');
      assertNonNegativeInteger(slotIndex, 'slotIndex');
      if (slotIndex >= this.framesInFlight) {
        throw new RangeError(
          `slotIndex (${slotIndex}) out of range [0, ${this.framesInFlight - 1}]`
        );
      }

      const slot = this._slots[slotIndex];
      let resolvedSize;
      if (sliceSize === undefined) {
        resolvedSize = Math.min(this.maxBindingSize, slot.byteSize);
      } else {
        assertPositiveInteger(sliceSize, 'sliceSize');
        if (sliceSize > this.maxBindingSize) {
          throw new RangeError(
            `sliceSize (${sliceSize}) exceeds maxBindingSize (${this.maxBindingSize})`
          );
        }
        if (sliceSize > slot.byteSize) {
          throw new RangeError(
            `sliceSize (${sliceSize}) exceeds slot buffer size (${slot.byteSize})`
          );
        }
        resolvedSize = Math.max(
          this.minUniformBindingSize,
          alignUp(sliceSize, WEBGPU_BUFFER_COPY_ALIGNMENT)
        );
        if (resolvedSize > this.maxBindingSize) {
          throw new RangeError(
            `Padded sliceSize (${resolvedSize}) exceeds maxBindingSize (${this.maxBindingSize})`
          );
        }
        if (resolvedSize > slot.byteSize) {
          throw new RangeError(
            `Padded sliceSize (${resolvedSize}) exceeds slot buffer size (${slot.byteSize})`
          );
        }
      }

      const target = validateOutTarget(out, 'createBindGroupEntry');
      if (target) {
        target.binding = binding;
        if (!target.resource || typeof target.resource !== 'object') {
          target.resource = {
            buffer: slot.buffer,
            offset: 0,
            size: resolvedSize,
          };
        } else {
          target.resource.buffer = slot.buffer;
          target.resource.offset = 0;
          target.resource.size = resolvedSize;
        }
        return target;
      }

      return {
        binding,
        resource: {
          buffer: slot.buffer,
          offset: 0,
          size: resolvedSize,
        },
      };
    }

    /**
     * Build a WebGPU `GPUBindGroupEntry` directly for a `pushConstants()` / `pushCbv()`
     * allocation (works transparently for primary ring slots, rolled-over slices, and
     * spill pages). Pass an optional preallocated `out` object (`{ binding, resource: {} }`)
     * to avoid per-draw heap allocation.
     *
     * @param {number} binding
     * @param {{ buffer: GPUBuffer, size?: number, paddedSize?: number }} allocation
     * @param {object} [out]
     */
    createBindGroupEntryForAllocation(binding, allocation, out) {
      this._assertNotDestroyed();
      assertNonNegativeInteger(binding, 'binding');
      if (!allocation || typeof allocation !== 'object' || !allocation.buffer) {
        throw new TypeError(
          'createBindGroupEntryForAllocation expects an allocation with a GPUBuffer'
        );
      }
      const rawSize =
        allocation.paddedSize !== undefined
          ? allocation.paddedSize
          : allocation.size !== undefined
            ? allocation.size
            : 0;
      const resolvedSize = this.normalizeSliceSize(rawSize);
      if (resolvedSize > this.maxBindingSize) {
        this._alignmentViolationsRejected += 1;
        throw new RangeError(
          `Allocation binding size (${resolvedSize}) exceeds maxBindingSize (${this.maxBindingSize})`
        );
      }
      const target = validateOutTarget(out, 'createBindGroupEntryForAllocation');
      if (target) {
        target.binding = binding;
        if (!target.resource || typeof target.resource !== 'object') {
          target.resource = {
            buffer: allocation.buffer,
            offset: 0,
            size: resolvedSize,
          };
        } else {
          target.resource.buffer = allocation.buffer;
          target.resource.offset = 0;
          target.resource.size = resolvedSize;
        }
        return target;
      }
      return {
        binding,
        resource: {
          buffer: allocation.buffer,
          offset: 0,
          size: resolvedSize,
        },
      };
    }

    /**
     * Return a diagnostic snapshot of all live uniform buffer slices active in the current
     * frame (retired multi-slices + current primary slice + spill pages) in allocation order.
     */
    getActiveFrameSlices() {
      this._assertNotDestroyed();
      const slot = this._slots[this._slotIndex];
      if (!slot) {
        return [];
      }
      const slices = [];
      for (let i = 0; i < slot.retiredSlices.length; i += 1) {
        const s = slot.retiredSlices[i];
        slices.push({
          sliceIndex: i,
          kind: 'retired',
          buffer: s.buffer,
          byteSize: s.byteSize,
          usedBytes: s.cursor,
          drawCount: s.drawCount,
          dirty: s.dirty,
        });
      }
      slices.push({
        sliceIndex: this._activeSliceIndex,
        kind: 'primary',
        buffer: slot.buffer,
        byteSize: slot.byteSize,
        usedBytes: this._cursor,
        drawCount: this._primaryFrameDrawCount,
        dirty: this._dirtyRanges.length > 0,
      });
      for (let i = 0; i < slot.overflowChunks.length; i += 1) {
        const c = slot.overflowChunks[i];
        slices.push({
          sliceIndex: this._activeSliceIndex + 1 + i,
          kind: 'spill',
          buffer: c.buffer,
          byteSize: c.byteSize,
          usedBytes: c.cursor,
          drawCount: c.drawCount,
          dirty: c.dirty,
        });
      }
      return slices;
    }

    /**
     * Return ring configuration, usage, alignment, VRAM shrink, SAB copy, rollover, and overflow diagnostics.
     */
    getStats() {
      const activeSlot =
        !this._destroyed && this._slots.length > 0
          ? this._slots[this._slotIndex]
          : null;
      const peakVramBytes = Math.max(this._allocatedGpuBytes, this._peakAllocatedGpuBytes);
      const activeFrameSliceCount = activeSlot
        ? 1 + activeSlot.retiredSlices.length + activeSlot.overflowChunks.length
        : 0;
      return {
        frameIndex: this._frameIndex,
        slotIndex: this._slotIndex,
        partitionIndex: this._slotIndex,
        framesInFlight: this.framesInFlight,
        partitionRotations: this._partitionRotations,
        inFlightPartitions: this._destroyed ? 0 : this._countInFlightPartitions(),
        alignment: this.alignment,
        minUniformBindingSize: this.minUniformBindingSize,
        byteSize: this._capacity,
        currentCapacityBytes: this._capacity,
        minSlotBytes: this.minSlotBytes,
        maxByteSize: this.maxByteSize,
        maxCapacityBytes: this.maxByteSize,
        maxRingBytes: this.maxRingBytes,
        maxBindingSize: this.maxBindingSize,
        shrinkAfterIdleFrames: this.shrinkAfterIdleFrames,
        shrinkThresholdRatio: this.shrinkThresholdRatio,
        consecutiveIdleFrames: this._consecutiveIdleFrames,
        allowOverflow: this.allowOverflow,
        allowSpill: this.allowSpill,
        multiSlice: this.multiSlice,
        allowWrap: this.allowWrap,
        immediateFlush: this.immediateFlush,
        maxCoalesceGapBytes: this.maxCoalesceGapBytes,
        usedBytes: this._frameSliceBytes + this._framePrimaryBytes + this._frameOverflowBytes,
        peakUsedBytes: this._peakUsedBytes,
        peakFrameBytes: this._peakFrameBytes,
        peakUtilizationRatio: this._capacity > 0 ? this._peakFrameBytes / this._capacity : 0,
        allocatedGpuBytes: this._allocatedGpuBytes,
        peakAllocatedGpuBytes: peakVramBytes,
        peakVramBytes,
        frameDrawCount: this._frameDrawCount,
        totalDrawCount: this._totalDrawCount,
        totalAllocations: this._totalDrawCount,
        reusedAllocations: this._reusedAllocations,
        freshAllocations: this._freshAllocations,
        zeroAllocAllocs: this._zeroAllocAllocs,
        reuseRate: clampUnitRate(this._reusedAllocations, this._totalDrawCount),
        totalBytesPushed: this._totalBytesPushed,
        totalBytesUploaded: this._totalBytesUploaded,
        bytesUploaded: this._totalBytesUploaded,
        alignedBytesWasted: this._alignedBytesWasted,
        alignmentPaddingBytes: this._alignedBytesWasted,
        flushCount: this._flushCount,
        writeBufferCalls: this._writeBufferCalls,
        coalescedWrites: this._coalescedWrites,
        stagedWritesFlushed: this._stagedWritesFlushed,
        dirtyRangeCount: this._destroyed ? 0 : this._countPendingDirtyRanges(),
        batchDepth: this._batchDepth,
        batchActive: this._batchDepth > 0,
        growCount: this._growCount,
        growthCount: this._growCount,
        shrinkCount: this._shrinkCount,
        overflowAllocCount: this._overflowAllocCount,
        overflowBufferCount: this._destroyed ? 0 : this._countLiveOverflowBuffers(),
        liveRetiredSlices: this._destroyed ? 0 : this._countLiveRetiredSlices(),
        spillAllocations: this._spillAllocations,
        spillPagesCreated: this._spillPagesCreated,
        liveSpillPages: this._destroyed ? 0 : this._countLiveSpillPages(),
        liveSpillBytes: this._destroyed ? 0 : this._countLiveSpillBytes(),
        rolloverCount: this._rolloverCount,
        sliceRolloverCount: this._sliceRolloverCount,
        spillRolloverCount: this._spillRolloverCount,
        activeSliceIndex: this._destroyed ? 0 : this._activeSliceIndex,
        activeFrameSliceCount,
        maxSlicesPerFrame: this._maxSlicesPerFrame,
        alignmentViolationsRejected: this._alignmentViolationsRejected,
        sabCopyCount: this._sabCopyCount,
        unalignedCopyCount: this._unalignedCopyCount,
        wrapCount: this._wrapCount,
        bufferGeneration: activeSlot ? activeSlot.generation : 0,
        destroyed: this._destroyed,
      };
    }

    /**
     * Reset cumulative telemetry counters while preserving live slot GPUBuffers, active
     * frame state, and idle shrink aging. Re-baselines `peakVramBytes` (and
     * `peakAllocatedGpuBytes`) to current live VRAM (`allocatedGpuBytes`) so
     * `peakVramBytes >= allocatedGpuBytes` always holds.
     */
    resetStats() {
      const currentUsed =
        this._frameSliceBytes + this._framePrimaryBytes + this._frameOverflowBytes;
      const activeSlot =
        !this._destroyed && this._slots.length > 0
          ? this._slots[this._slotIndex]
          : null;
      this._peakAllocatedGpuBytes = this._allocatedGpuBytes;
      this._peakUsedBytes = currentUsed;
      this._peakFrameBytes = currentUsed;
      this._totalDrawCount = 0;
      this._reusedAllocations = 0;
      this._freshAllocations = 0;
      this._zeroAllocAllocs = 0;
      this._partitionRotations = 0;
      this._totalBytesPushed = 0;
      this._totalBytesUploaded = 0;
      this._alignedBytesWasted = 0;
      this._flushCount = 0;
      this._writeBufferCalls = 0;
      this._coalescedWrites = 0;
      this._stagedWritesFlushed = 0;
      this._growCount = 0;
      this._shrinkCount = 0;
      this._overflowAllocCount = 0;
      this._spillAllocations = 0;
      this._spillPagesCreated = 0;
      this._rolloverCount = 0;
      this._sliceRolloverCount = 0;
      this._spillRolloverCount = 0;
      this._maxSlicesPerFrame =
        activeSlot && currentUsed > 0
          ? 1 + activeSlot.retiredSlices.length + activeSlot.overflowChunks.length
          : 0;
      this._alignmentViolationsRejected = 0;
      this._sabCopyCount = 0;
      this._unalignedCopyCount = 0;
      this._wrapCount = 0;
      return this.getStats();
    }

    /**
     * Destroy all allocated GPUBuffers (primary, retired, and overflow) idempotently and
     * release CPU shadow memory.
     */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      for (let i = 0; i < this._slots.length; i += 1) {
        const slot = this._slots[i];
        this._releaseSlotTransientBuffers(slot);
        this._destroySlotBuffer(slot);
        slot.shadow = null;
        slot.shadowU32 = null;
        slot.shadowI32 = null;
        slot.shadowF32 = null;
        slot.cachedFlushView = null;
        slot.cachedFlushStart = -1;
        slot.cachedFlushEnd = -1;
        slot.inFlight = false;
        slot.fencePromise = null;
      }

      this._slots = [];
      this._setShadowBuffer(null);
      this._dirtyRanges = [];
      this._dirtyRangePool = [];
      this._retiredSlicePool = [];
      this._tmpOverflowAlloc.chunk = null;
      this._allocatedGpuBytes = 0;
      this._frameEnded = false;
      this._cursor = 0;
      this._writtenHighWater = 0;
      this._framePrimaryBytes = 0;
      this._flushedCursor = 0;
      this._flushedDrawCount = 0;
      this._hasFlushedActiveSlot = false;
      this._primaryFrameDrawCount = 0;
      this._frameSliceBytes = 0;
      this._activeSliceIndex = 0;
      this._frameOverflowBytes = 0;
      this._frameDrawCount = 0;
      this._batchDepth = 0;
      this._dirty = false;
      this._consecutiveIdleFrames = 0;
      this._idleWindowHighWaterBytes = 0;
    }
  }

  const exportsObj = Object.freeze({
    DEFAULT_UNIFORM_ALIGNMENT,
    DEFAULT_MIN_UNIFORM_BINDING_SIZE,
    DEFAULT_BYTE_SIZE,
    DEFAULT_MAX_BYTE_SIZE,
    MIN_FRAMES_IN_FLIGHT,
    MAX_FRAMES_IN_FLIGHT,
    DEFAULT_FRAMES_IN_FLIGHT,
    DEFAULT_MAX_BINDING_SIZE,
    DEFAULT_SHRINK_AFTER_IDLE_FRAMES,
    DEFAULT_SHRINK_THRESHOLD_RATIO,
    WEBGPU_BUFFER_COPY_ALIGNMENT,
    BUFFER_USAGE,
    SHADER_STAGE,
    alignUp,
    alignUniformSliceSize,
    copyForWriteBuffer,
    coalesceDirtyRanges,
    DynamicUniformRing,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6UniformRing = exportsObj;
  } else if (root) {
    root.PlayGta6UniformRing = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
