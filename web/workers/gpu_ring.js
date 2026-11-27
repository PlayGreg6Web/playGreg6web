'use strict';

(function (root) {
  // Binary-compatible with playgta6::gpu::GpuRingHeader (runtime/gpu/gpu_ring.h).
  const GPU_RING_MAGIC = 0x47505552; // "GPUR"
  const GPU_RING_VERSION = 1;
  const GPU_RING_HEADER_BYTES = 64;
  const GPU_RING_HEADER_WORDS = 16;
  const DEFAULT_CAPACITY_BYTES = 65536;
  // Binary-compatible with playgta6::GpuPacketHeader (docs/gpu_commands.md):
  //   u16 opcode (0..2), u16 flags (2..4), u32 payload_bytes (4..8).
  const GPU_PACKET_HEADER_BYTES = 8;
  const DEFAULT_BATCH_MAX_PACKETS = 256;
  const DEFAULT_WRAP_SCRATCH_BYTES = 65536;

  // 32-bit word indices within the 64-byte header (offsets 0..64).
  const HEADER_INDEX = Object.freeze({
    MAGIC: 0,           // offset 0
    VERSION: 1,         // offset 4
    CAPACITY_BYTES: 2,  // offset 8
    RESERVED0: 3,       // offset 12
    WRITE_POS: 4,       // offset 16 (monotonic u32 byte counter)
    READ_POS: 5,        // offset 20 (monotonic u32 byte counter)
    DROPPED_PACKETS: 6, // offset 24
    NOTIFY_SEQ: 7,      // offset 28
  });

  function isPowerOfTwo(value) {
    return (
      Number.isInteger(value) &&
      value > 0 &&
      value <= 0x80000000 &&
      (value & (value - 1)) === 0
    );
  }

  function sanitizeNonNegInt(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return Math.floor(value);
  }

  function sanitizeNonNegNumber(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return value;
  }

  function clampUnitRatio(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return value >= 1 ? 1 : value;
  }

  function toUint8Array(input) {
    if (input instanceof Uint8Array) {
      return input;
    }
    if (ArrayBuffer.isView(input)) {
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    if (input instanceof ArrayBuffer) {
      return new Uint8Array(input);
    }
    if (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      'GpuRingBuffer.writeBytes expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer'
    );
  }

  /**
   * Lock-free Single-Producer / Single-Consumer (SPSC) GPU command ring buffer
   * backed by a SharedArrayBuffer:
   *   [0..64)                    : 64-byte atomic control header (GpuRingHeader)
   *   [64..64 + capacity_bytes)  : power-of-two circular byte payload region
   */
  class GpuRingBuffer {
    static MAGIC = GPU_RING_MAGIC;
    static VERSION = GPU_RING_VERSION;
    static HEADER_BYTES = GPU_RING_HEADER_BYTES;
    static HEADER_WORDS = GPU_RING_HEADER_WORDS;
    static HEADER_INDEX = HEADER_INDEX;
    static PACKET_HEADER_BYTES = GPU_PACKET_HEADER_BYTES;
    static DEFAULT_BATCH_MAX_PACKETS = DEFAULT_BATCH_MAX_PACKETS;
    static DEFAULT_WRAP_SCRATCH_BYTES = DEFAULT_WRAP_SCRATCH_BYTES;

    static byteLength(capacityBytes) {
      if (!isPowerOfTwo(capacityBytes)) {
        throw new RangeError(
          `capacityBytes must be a power of two in (0, 2^31], got ${capacityBytes}`
        );
      }
      return GPU_RING_HEADER_BYTES + capacityBytes;
    }

    static create(capacityBytes = DEFAULT_CAPACITY_BYTES) {
      if (typeof SharedArrayBuffer === 'undefined') {
        throw new Error('SharedArrayBuffer is unavailable (requires Cross-Origin Isolation)');
      }
      const totalBytes = GpuRingBuffer.byteLength(capacityBytes);
      const sab = new SharedArrayBuffer(totalBytes);
      return new GpuRingBuffer(sab, { capacityBytes, initialize: true });
    }

    static attach(sab, byteOffset = 0) {
      return new GpuRingBuffer(sab, { byteOffset, initialize: false });
    }

    constructor(sab, options = {}) {
      if (typeof SharedArrayBuffer === 'undefined' || !(sab instanceof SharedArrayBuffer)) {
        throw new TypeError('GpuRingBuffer requires a SharedArrayBuffer');
      }

      const byteOffset = options.byteOffset !== undefined ? options.byteOffset : 0;
      if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || (byteOffset & 3) !== 0) {
        throw new RangeError(`byteOffset must be a non-negative 4-byte aligned integer, got ${byteOffset}`);
      }
      if (byteOffset + GPU_RING_HEADER_BYTES > sab.byteLength) {
        throw new RangeError(
          `SharedArrayBuffer too small (${sab.byteLength} bytes; need at least ${byteOffset + GPU_RING_HEADER_BYTES})`
        );
      }

      this._sab = sab;
      this._byteOffset = byteOffset;
      this._headerI32 = new Int32Array(sab, byteOffset, GPU_RING_HEADER_WORDS);
      this._headerU32 = new Uint32Array(sab, byteOffset, GPU_RING_HEADER_WORDS);

      const existingMagic = Atomics.load(this._headerU32, HEADER_INDEX.MAGIC);
      const shouldInit =
        options.initialize === true ||
        (options.initialize === undefined && existingMagic === 0);

      if (shouldInit) {
        const availableDataBytes = sab.byteLength - byteOffset - GPU_RING_HEADER_BYTES;
        const capacityBytes =
          options.capacityBytes !== undefined ? options.capacityBytes : availableDataBytes;
        if (!isPowerOfTwo(capacityBytes)) {
          throw new RangeError(
            `capacityBytes must be a power of two in (0, 2^31], got ${capacityBytes}`
          );
        }
        if (capacityBytes > availableDataBytes) {
          throw new RangeError(
            `capacityBytes (${capacityBytes}) exceeds available SharedArrayBuffer bytes (${availableDataBytes})`
          );
        }

        for (let i = 0; i < GPU_RING_HEADER_WORDS; i++) {
          Atomics.store(this._headerU32, i, 0);
        }
        Atomics.store(this._headerU32, HEADER_INDEX.VERSION, GPU_RING_VERSION);
        Atomics.store(this._headerU32, HEADER_INDEX.CAPACITY_BYTES, capacityBytes);
        Atomics.store(this._headerU32, HEADER_INDEX.RESERVED0, 0);
        Atomics.store(this._headerU32, HEADER_INDEX.WRITE_POS, 0);
        Atomics.store(this._headerU32, HEADER_INDEX.READ_POS, 0);
        Atomics.store(this._headerU32, HEADER_INDEX.DROPPED_PACKETS, 0);
        Atomics.store(this._headerU32, HEADER_INDEX.NOTIFY_SEQ, 0);
        // Publish magic last so concurrent attachers observe initialized metadata.
        Atomics.store(this._headerU32, HEADER_INDEX.MAGIC, GPU_RING_MAGIC);

        this._capacityBytes = capacityBytes;
      } else {
        if (existingMagic !== GPU_RING_MAGIC) {
          throw new Error(
            `Invalid GpuRingHeader magic: expected 0x${GPU_RING_MAGIC.toString(16)}, got 0x${existingMagic.toString(16)}`
          );
        }
        const version = Atomics.load(this._headerU32, HEADER_INDEX.VERSION);
        if (version !== GPU_RING_VERSION) {
          throw new Error(
            `Unsupported GpuRingHeader version: expected ${GPU_RING_VERSION}, got ${version}`
          );
        }
        const capacityBytes = Atomics.load(this._headerU32, HEADER_INDEX.CAPACITY_BYTES);
        if (!isPowerOfTwo(capacityBytes)) {
          throw new RangeError(`Invalid GpuRingHeader capacity_bytes: ${capacityBytes}`);
        }
        const requiredBytes = byteOffset + GPU_RING_HEADER_BYTES + capacityBytes;
        if (sab.byteLength < requiredBytes) {
          throw new RangeError(
            `SharedArrayBuffer (${sab.byteLength} bytes) is smaller than header requires (${requiredBytes} bytes)`
          );
        }
        this._capacityBytes = capacityBytes;
      }

      this._mask = (this._capacityBytes - 1) >>> 0;
      this._data = new Uint8Array(
        sab,
        byteOffset + GPU_RING_HEADER_BYTES,
        this._capacityBytes
      );
      this._dataView = new DataView(
        sab,
        byteOffset + GPU_RING_HEADER_BYTES,
        this._capacityBytes
      );
      this._headerScratch = new Uint8Array(GPU_PACKET_HEADER_BYTES);
      this._headerScratchView = new DataView(this._headerScratch.buffer);
      // Pre-allocated reusable wrap scratch buffer + cached DataView so wrap-around spans
      // avoid per-batch backing and DataView allocations.
      const initialWrapScratchBytes = Math.min(this._capacityBytes, DEFAULT_WRAP_SCRATCH_BYTES);
      this._wrapScratch = new Uint8Array(initialWrapScratchBytes);
      this._wrapScratchView = new DataView(
        this._wrapScratch.buffer,
        this._wrapScratch.byteOffset,
        this._wrapScratch.byteLength
      );
      this._scratch = this._wrapScratch;
      // Reusable descriptor and result objects for zero-alloc consumer batch draining.
      this._spanDescriptor = {
        bytes: null,
        byteOffset: 0,
        byteLength: 0,
        bufferByteOffset: 0,
        ringOffset: 0,
        packets: 0,
        wrapped: false,
        zeroCopy: false,
        dataView: null,
        viewOffset: 0,
      };
      this._packetDescriptor = {
        opcode: 0,
        flags: 0,
        payloadBytes: 0,
        payload: null,
        bytes: null,
        byteLength: 0,
        wrapped: false,
        dataView: null,
        viewOffset: 0,
      };
      this._drainIntoMeta = {
        opcode: 0,
        flags: 0,
        payloadBytes: 0,
        byteOffset: 0,
        byteLength: 0,
        payloadOffset: 0,
        wrapped: false,
        zeroCopy: false,
        dataView: null,
        viewOffset: 0,
      };
      this._reusableBatchResult = {
        packets: 0,
        bytes: 0,
        remainingBytes: 0,
        wrapped: false,
      };
      this._reusableDrainResult = {
        packets: 0,
        bytes: 0,
        remainingBytes: 0,
        wrapped: false,
        zeroCopy: false,
      };

      // Local throughput & backpressure telemetry (kept outside the 64-byte SAB ABI header).
      this._batchesDrained = 0;
      this._packetsDrained = 0;
      this._maxBatchSize = 0;
      this._backpressureWaits = 0;
      this._backpressureStalls = 0;
      this._writesSucceeded = 0;
      this._peakOccupancyBytes = 0;
      this._drainCalls = 0;
      this._maxDrainBatchBytes = 0;
      this._peakBatchBytes = 0;
      this._zeroAllocDrains = 0;
      this._totalBytesDrained = 0;
      this._emptyDrains = 0;
      this._corruptHeaders = 0;
    }

    _ensureWrapScratch(requiredBytes) {
      if (this._wrapScratch && this._wrapScratch.byteLength >= requiredBytes) {
        return this._wrapScratch;
      }
      let nextSize =
        this._wrapScratch && this._wrapScratch.byteLength > 0
          ? this._wrapScratch.byteLength
          : Math.min(this._capacityBytes, 1024);
      if (nextSize < 64) {
        nextSize = 64;
      }
      while (nextSize < requiredBytes && nextSize < this._capacityBytes) {
        nextSize = Math.min(this._capacityBytes, nextSize * 2);
      }
      if (nextSize < requiredBytes) {
        nextSize = requiredBytes;
      }
      this._wrapScratch = new Uint8Array(nextSize);
      this._wrapScratchView = new DataView(
        this._wrapScratch.buffer,
        this._wrapScratch.byteOffset,
        this._wrapScratch.byteLength
      );
      this._scratch = this._wrapScratch;
      return this._wrapScratch;
    }

    _copyWrappedToScratch(startOffset, byteLength) {
      const cap = this._capacityBytes;
      const clampedLen = byteLength > cap ? cap : byteLength;
      const scratch = this._ensureWrapScratch(clampedLen);
      const data = this._data;
      const firstLen = cap - startOffset;
      const secondLen = clampedLen - firstLen;
      if (firstLen <= 128) {
        for (let i = 0; i < firstLen; i++) {
          scratch[i] = data[startOffset + i];
        }
      } else {
        scratch.set(data.subarray(startOffset, cap), 0);
      }
      if (secondLen <= 128) {
        for (let i = 0; i < secondLen; i++) {
          scratch[firstLen + i] = data[i];
        }
      } else {
        scratch.set(data.subarray(0, secondLen), firstLen);
      }
      return scratch;
    }

    _recordOccupancy(bytes) {
      if (!(bytes > 0) || bytes === Infinity) {
        return;
      }
      const clamped = bytes >= this._capacityBytes ? this._capacityBytes : (bytes | 0);
      const cur = this._peakOccupancyBytes;
      const currentPeak = cur >= 0 && cur < Infinity ? cur : 0;
      if (clamped > currentPeak) {
        this._peakOccupancyBytes = clamped;
      }
    }

    get highWatermarkBytes() {
      return Math.min(this._capacityBytes, sanitizeNonNegInt(this._peakOccupancyBytes));
    }

    get peakOccupancyRatio() {
      if (this._capacityBytes <= 0) {
        return 0;
      }
      return clampUnitRatio(this.highWatermarkBytes / this._capacityBytes);
    }

    get peakUtilizationPct() {
      if (this._capacityBytes <= 0) {
        return 0;
      }
      return Math.min(
        100,
        sanitizeNonNegNumber((this.highWatermarkBytes / this._capacityBytes) * 100)
      );
    }

    get batchesDrained() {
      return sanitizeNonNegInt(this._batchesDrained);
    }

    get packetsDrained() {
      return sanitizeNonNegInt(this._packetsDrained);
    }

    get backpressureWaits() {
      return sanitizeNonNegInt(this._backpressureWaits);
    }

    get backpressureStalls() {
      return sanitizeNonNegInt(this._backpressureStalls);
    }

    get drainCalls() {
      return sanitizeNonNegInt(this._drainCalls);
    }

    get maxDrainBatchBytes() {
      return Math.min(this._capacityBytes, sanitizeNonNegInt(this._maxDrainBatchBytes));
    }

    get peakBatchBytes() {
      return Math.min(this._capacityBytes, sanitizeNonNegInt(this._peakBatchBytes));
    }

    get zeroAllocDrains() {
      return sanitizeNonNegInt(this._zeroAllocDrains);
    }

    get backpressureWaitRate() {
      const waits = sanitizeNonNegInt(this._backpressureWaits);
      const writes = sanitizeNonNegInt(this._writesSucceeded);
      const total = writes + waits;
      if (total <= 0) {
        return 0;
      }
      return clampUnitRatio(waits / total);
    }

    get totalBytesDrained() {
      return sanitizeNonNegInt(this._totalBytesDrained);
    }

    get emptyDrains() {
      return sanitizeNonNegInt(this._emptyDrains);
    }

    get corruptHeaders() {
      return sanitizeNonNegInt(this._corruptHeaders);
    }

    get sab() {
      return this._sab;
    }

    get byteOffset() {
      return this._byteOffset;
    }

    get capacityBytes() {
      return this._capacityBytes;
    }

    get dataView() {
      return this._dataView;
    }

    get data() {
      return this._data;
    }

    get headerI32() {
      return this._headerI32;
    }

    get headerU32() {
      return this._headerU32;
    }

    get wrapScratchView() {
      return this._wrapScratchView;
    }

    get magic() {
      return Atomics.load(this._headerU32, HEADER_INDEX.MAGIC);
    }

    get version() {
      return Atomics.load(this._headerU32, HEADER_INDEX.VERSION);
    }

    get writePos() {
      return Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
    }

    get readPos() {
      return Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
    }

    get droppedPackets() {
      return Atomics.load(this._headerU32, HEADER_INDEX.DROPPED_PACKETS);
    }

    get notifySeq() {
      return Atomics.load(this._headerU32, HEADER_INDEX.NOTIFY_SEQ);
    }

    _snapshotPositions() {
      // Load READ_POS before WRITE_POS so concurrent SPSC progress never produces a
      // wrapped ~4 GiB difference (w - r) when consumer advances between loads.
      let readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      let writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      let avail = (writePos - readPos) >>> 0;
      if (avail > this._capacityBytes) {
        for (let attempt = 0; attempt < 2 && avail > this._capacityBytes; attempt++) {
          readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
          writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
          avail = (writePos - readPos) >>> 0;
        }
        if (avail > 0x80000000) {
          avail = 0;
        } else if (avail > this._capacityBytes) {
          avail = this._capacityBytes;
        }
      }
      return { readPos, writePos, avail };
    }

    availableBytes() {
      let readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      let writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      let avail = (writePos - readPos) >>> 0;
      if (avail > this._capacityBytes) {
        for (let attempt = 0; attempt < 2 && avail > this._capacityBytes; attempt++) {
          readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
          writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
          avail = (writePos - readPos) >>> 0;
        }
        if (avail > 0x80000000) {
          return 0;
        }
        if (avail > this._capacityBytes) {
          avail = this._capacityBytes;
        }
      }
      this._recordOccupancy(avail);
      return avail;
    }

    freeBytes() {
      const used = this.availableBytes();
      return used >= this._capacityBytes ? 0 : this._capacityBytes - used;
    }

    /**
     * Publishes a consumer wakeup notification on NOTIFY_SEQ (paired with writeBytes(..., false)).
     */
    notifyConsumer() {
      Atomics.add(this._headerI32, HEADER_INDEX.NOTIFY_SEQ, 1);
      return Atomics.notify(this._headerI32, HEADER_INDEX.NOTIFY_SEQ, 1);
    }

    /**
     * Race-free consumer wait for queued ring bytes.
     * Snapshots NOTIFY_SEQ before checking WRITE_POS/READ_POS so producer publishes
     * between the emptiness check and Atomics.wait never lose a wakeup.
     * Defaults to a bounded timeout (4 ms) so a non-notifying producer cannot deadlock.
     *
     * @param {number} [timeoutMs=4]
     * @returns {'ok'|'not-equal'|'timed-out'|'unsupported'}
     */
    waitForData(timeoutMs = 4) {
      const seq = Atomics.load(this._headerI32, HEADER_INDEX.NOTIFY_SEQ);
      const readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      const writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      const avail = (writePos - readPos) >>> 0;
      if (avail > 0 && avail <= this._capacityBytes) {
        return 'ok';
      }
      if (typeof Atomics.wait !== 'function') {
        return 'unsupported';
      }
      const boundedTimeout =
        typeof timeoutMs === 'number' && Number.isFinite(timeoutMs) && timeoutMs >= 0
          ? timeoutMs
          : 4;
      try {
        return Atomics.wait(this._headerI32, HEADER_INDEX.NOTIFY_SEQ, seq, boundedTimeout);
      } catch (_err) {
        return 'unsupported';
      }
    }

    /**
     * Race-free producer wait until at least `requiredBytes` are free in the ring.
     *
     * @param {number} [requiredBytes=1]
     * @param {number} [timeoutMs=4]
     * @returns {'ok'|'not-equal'|'timed-out'|'unsupported'}
     */
    waitForSpace(requiredBytes = 1, timeoutMs = 4) {
      const req = sanitizeNonNegInt(requiredBytes);
      if (req === 0) {
        return 'ok';
      }
      if (req > this._capacityBytes) {
        return 'timed-out';
      }
      const readPos = Atomics.load(this._headerI32, HEADER_INDEX.READ_POS);
      const writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      const used = (writePos - (readPos >>> 0)) >>> 0;
      const free = used >= this._capacityBytes ? 0 : this._capacityBytes - used;
      if (free >= req) {
        return 'ok';
      }
      if (typeof Atomics.wait !== 'function') {
        return 'unsupported';
      }
      const boundedTimeout =
        typeof timeoutMs === 'number' && Number.isFinite(timeoutMs) && timeoutMs >= 0
          ? timeoutMs
          : 4;
      try {
        const res = Atomics.wait(this._headerI32, HEADER_INDEX.READ_POS, readPos, boundedTimeout);
        if (res === 'ok' || res === 'timed-out') {
          this.recordBackpressureWait(1);
        }
        return res;
      } catch (_err) {
        return 'unsupported';
      }
    }

    /**
     * Records an external producer backpressure wait (e.g. Atomics.wait / retry backoff).
     *
     * @param {number} [count=1]
     * @returns {number} Updated backpressure wait count.
     */
    recordBackpressureWait(count = 1) {
      const n = sanitizeNonNegInt(count);
      if (n > 0) {
        this._backpressureWaits = sanitizeNonNegInt(this._backpressureWaits) + n;
        this._backpressureStalls = sanitizeNonNegInt(this._backpressureStalls) + n;
      }
      return sanitizeNonNegInt(this._backpressureWaits);
    }

    /**
     * All-or-nothing byte write supporting wrap-around.
     * Returns false (and increments dropped_packets) without modifying write_pos
     * or ring contents when the ring does not have enough free bytes.
     *
     * @param {Uint8Array|ArrayBufferView|ArrayBuffer|SharedArrayBuffer} input
     * @returns {boolean}
     */
    writeBytes(input, notify = true) {
      const src = toUint8Array(input);
      const len = src.byteLength;
      if (len === 0) {
        return true;
      }
      const cap = this._capacityBytes;
      const hdrU32 = this._headerU32;
      if (len > cap) {
        Atomics.add(hdrU32, HEADER_INDEX.DROPPED_PACKETS, 1);
        return false;
      }

      const writePos = Atomics.load(hdrU32, HEADER_INDEX.WRITE_POS);
      const readPos = Atomics.load(hdrU32, HEADER_INDEX.READ_POS);
      const used = (writePos - readPos) >>> 0;
      const free = used >= cap ? 0 : cap - used;
      if (len > free) {
        this._recordOccupancy(used);
        const curWaits = this._backpressureWaits;
        this._backpressureWaits = curWaits >= 0 && curWaits < Infinity ? curWaits + 1 : 1;
        const curStalls = this._backpressureStalls;
        this._backpressureStalls = curStalls >= 0 && curStalls < Infinity ? curStalls + 1 : 1;
        Atomics.add(hdrU32, HEADER_INDEX.DROPPED_PACKETS, 1);
        return false;
      }

      const data = this._data;
      const offset = writePos & this._mask;
      if (offset + len <= cap) {
        if (len <= 64) {
          for (let i = 0; i < len; i++) {
            data[offset + i] = src[i];
          }
        } else {
          data.set(src, offset);
        }
      } else {
        const firstLen = cap - offset;
        const secondLen = len - firstLen;
        if (firstLen <= 128) {
          for (let i = 0; i < firstLen; i++) {
            data[offset + i] = src[i];
          }
        } else {
          data.set(src.subarray(0, firstLen), offset);
        }
        if (secondLen <= 128) {
          for (let i = 0; i < secondLen; i++) {
            data[i] = src[firstLen + i];
          }
        } else {
          data.set(src.subarray(firstLen, len), 0);
        }
      }

      const nextWrite = (writePos + len) >>> 0;
      Atomics.store(hdrU32, HEADER_INDEX.WRITE_POS, nextWrite);
      if (notify !== false) {
        const hdrI32 = this._headerI32;
        Atomics.add(hdrI32, HEADER_INDEX.NOTIFY_SEQ, 1);
        Atomics.notify(hdrI32, HEADER_INDEX.NOTIFY_SEQ, 1);
      }
      const curWrites = this._writesSucceeded;
      this._writesSucceeded = curWrites >= 0 && curWrites < Infinity ? curWrites + 1 : 1;
      this._recordOccupancy(used + len);
      return true;
    }

    /**
     * Zero-alloc framed packet producer: encodes an 8-byte GpuPacketHeader and
     * copies `payload` directly into the ring buffer without allocating a combined
     * intermediate packet buffer.
     *
     * @param {number} opcode
     * @param {number} [flags=0]
     * @param {Uint8Array|ArrayBufferView|ArrayBuffer|SharedArrayBuffer|null} [payload=null]
     * @param {boolean} [notify=true]
     * @returns {boolean}
     */
    writePacket(opcode, flags = 0, payload = null, notify = true) {
      const src = payload ? toUint8Array(payload) : null;
      const payloadLen = src ? src.byteLength : 0;
      const len = GPU_PACKET_HEADER_BYTES + payloadLen;
      const cap = this._capacityBytes;
      const hdrU32 = this._headerU32;
      if (len > cap) {
        Atomics.add(hdrU32, HEADER_INDEX.DROPPED_PACKETS, 1);
        return false;
      }

      const writePos = Atomics.load(hdrU32, HEADER_INDEX.WRITE_POS);
      const readPos = Atomics.load(hdrU32, HEADER_INDEX.READ_POS);
      const used = (writePos - readPos) >>> 0;
      const free = used >= cap ? 0 : cap - used;
      if (len > free) {
        this._recordOccupancy(used);
        const curWaits = this._backpressureWaits;
        this._backpressureWaits = curWaits >= 0 && curWaits < Infinity ? curWaits + 1 : 1;
        const curStalls = this._backpressureStalls;
        this._backpressureStalls = curStalls >= 0 && curStalls < Infinity ? curStalls + 1 : 1;
        Atomics.add(hdrU32, HEADER_INDEX.DROPPED_PACKETS, 1);
        return false;
      }

      const mask = this._mask;
      const data = this._data;
      const offset = writePos & mask;
      if (offset + GPU_PACKET_HEADER_BYTES <= cap) {
        const dv = this._dataView;
        dv.setUint16(offset, opcode & 0xffff, true);
        dv.setUint16(offset + 2, flags & 0xffff, true);
        dv.setUint32(offset + 4, payloadLen >>> 0, true);
      } else {
        const hsv = this._headerScratchView;
        const hs = this._headerScratch;
        hsv.setUint16(0, opcode & 0xffff, true);
        hsv.setUint16(2, flags & 0xffff, true);
        hsv.setUint32(4, payloadLen >>> 0, true);
        const firstHdr = cap - offset;
        for (let i = 0; i < firstHdr; i++) {
          data[offset + i] = hs[i];
        }
        for (let i = firstHdr; i < GPU_PACKET_HEADER_BYTES; i++) {
          data[i - firstHdr] = hs[i];
        }
      }

      if (payloadLen > 0) {
        const payloadOff = (offset + GPU_PACKET_HEADER_BYTES) & mask;
        if (payloadOff + payloadLen <= cap) {
          if (payloadLen <= 64) {
            for (let i = 0; i < payloadLen; i++) {
              data[payloadOff + i] = src[i];
            }
          } else {
            data.set(src, payloadOff);
          }
        } else {
          const firstLen = cap - payloadOff;
          const secondLen = payloadLen - firstLen;
          if (firstLen <= 128) {
            for (let i = 0; i < firstLen; i++) {
              data[payloadOff + i] = src[i];
            }
          } else {
            data.set(src.subarray(0, firstLen), payloadOff);
          }
          if (secondLen <= 128) {
            for (let i = 0; i < secondLen; i++) {
              data[i] = src[firstLen + i];
            }
          } else {
            data.set(src.subarray(firstLen, payloadLen), 0);
          }
        }
      }

      const nextWrite = (writePos + len) >>> 0;
      Atomics.store(hdrU32, HEADER_INDEX.WRITE_POS, nextWrite);
      if (notify !== false) {
        const hdrI32 = this._headerI32;
        Atomics.add(hdrI32, HEADER_INDEX.NOTIFY_SEQ, 1);
        Atomics.notify(hdrI32, HEADER_INDEX.NOTIFY_SEQ, 1);
      }
      const curWrites = this._writesSucceeded;
      this._writesSucceeded = curWrites >= 0 && curWrites < Infinity ? curWrites + 1 : 1;
      this._recordOccupancy(used + len);
      return true;
    }

    /**
     * Drains all currently queued bytes and passes a single contiguous Uint8Array
     * to `callback`.
     * - Zero-copy subarray directly over the SharedArrayBuffer when not wrapping.
     * - Linearized into a reusable scratch Uint8Array on wrap-around so framed
     *   GpuPacketHeader streams are never split or corrupted at the ring boundary.
     * - If `callback` throws, `READ_POS` is left unchanged so the stream remains unread.
     *
     * @param {(bytes: Uint8Array) => void} callback
     * @returns {number} Number of bytes drained.
     */
    drainTo(callback) {
      if (typeof callback !== 'function') {
        throw new TypeError('GpuRingBuffer.drainTo requires a callback function');
      }

      const readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      const writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      const avail = (writePos - readPos) >>> 0;
      const curDrainCalls = this._drainCalls;
      this._drainCalls = curDrainCalls >= 0 && curDrainCalls < Infinity ? curDrainCalls + 1 : 1;
      if (avail === 0) {
        const curEmpty = this._emptyDrains;
        this._emptyDrains = curEmpty >= 0 && curEmpty < Infinity ? curEmpty + 1 : 1;
        return 0;
      }
      const boundedAvail = avail > this._capacityBytes ? this._capacityBytes : avail;
      this._recordOccupancy(boundedAvail);

      const offset = (readPos & this._mask) >>> 0;
      let slice;
      if (offset + boundedAvail <= this._capacityBytes) {
        slice = this._data.subarray(offset, offset + boundedAvail);
      } else {
        const scratch = this._copyWrappedToScratch(offset, boundedAvail);
        slice = scratch.subarray(0, boundedAvail);
      }

      // Only advance READ_POS and telemetry if callback succeeds without throwing.
      callback(slice);

      const nextRead = (readPos + boundedAvail) >>> 0;
      Atomics.store(this._headerU32, HEADER_INDEX.READ_POS, nextRead);
      Atomics.notify(this._headerI32, HEADER_INDEX.READ_POS, 1);
      const curTotal = this._totalBytesDrained;
      this._totalBytesDrained =
        (curTotal >= 0 && curTotal < Infinity ? curTotal : 0) + boundedAvail;
      const curMaxBytes =
        this._maxDrainBatchBytes >= 0 && this._maxDrainBatchBytes < Infinity
          ? this._maxDrainBatchBytes
          : 0;
      if (boundedAvail > curMaxBytes) {
        this._maxDrainBatchBytes = boundedAvail;
      }
      const curPeakBatch =
        this._peakBatchBytes >= 0 && this._peakBatchBytes < Infinity
          ? this._peakBatchBytes
          : 0;
      if (boundedAvail > curPeakBatch) {
        this._peakBatchBytes = boundedAvail;
      }
      return boundedAvail;
    }

    /**
     * Drains up to `maxPackets` complete framed GpuPacketHeader packets in a single
     * bounded batch, reading `write_pos` once via `Atomics.load` and publishing the
     * updated `read_pos` once via `Atomics.store` at batch end (zero extra atomics
     * inside the per-packet loop).
     *
     * Supports both call signatures for ergonomic and backwards-compatible usage:
     *   - `drainBatch(handler, maxPackets = 256)`
     *   - `drainBatch(maxPackets, handler)`
     *
     * - Non-wrapping packets are passed as zero-copy subarrays directly over the SAB.
     * - Only a packet that physically straddles the ring wrap boundary is linearized
     *   into the reusable internal scratch buffer.
     * - Returning `false` from `handler` stops the batch cleanly after committing
     *   the current packet.
     * - If `handler` throws mid-batch, only packets that completed prior to the error
     *   are committed via `read_pos`; the failing packet and all subsequent packets
     *   remain unread in the ring.
     * - Views (`packet.bytes` / `packet.payload`) are only guaranteed valid for the
     *   duration of the `handler` call; copy them if retaining across drains.
     *
     * @param {((packet: {opcode: number, flags: number, payloadBytes: number, payload: Uint8Array, bytes: Uint8Array, byteLength: number, wrapped: boolean}, index: number) => (boolean|void)) | number} handlerOrMaxPackets
     * @param {number | ((packet: {opcode: number, flags: number, payloadBytes: number, payload: Uint8Array, bytes: Uint8Array, byteLength: number, wrapped: boolean}, index: number) => (boolean|void))} [maxPacketsOrHandler]
     * @returns {{packets: number, bytes: number, remainingBytes: number, wrapped: boolean}}
     */
    drainBatch(handlerOrMaxPackets, maxPacketsOrHandler) {
      let handler;
      let maxPacketsArg;
      let reuseResult = false;
      let perPacketCommit = false;
      if (typeof handlerOrMaxPackets === 'function') {
        handler = handlerOrMaxPackets;
        maxPacketsArg = maxPacketsOrHandler;
      } else if (typeof maxPacketsOrHandler === 'function') {
        maxPacketsArg = handlerOrMaxPackets;
        handler = maxPacketsOrHandler;
      } else {
        throw new TypeError('GpuRingBuffer.drainBatch requires a handler function');
      }

      let limit;
      if (maxPacketsArg === undefined || maxPacketsArg === null) {
        limit = DEFAULT_BATCH_MAX_PACKETS;
      } else if (
        typeof handlerOrMaxPackets === 'function' &&
        typeof maxPacketsArg === 'object' &&
        !Array.isArray(maxPacketsArg)
      ) {
        if (maxPacketsArg.maxPackets !== undefined && maxPacketsArg.maxPackets !== null) {
          if (
            typeof maxPacketsArg.maxPackets !== 'number' ||
            !Number.isSafeInteger(maxPacketsArg.maxPackets) ||
            maxPacketsArg.maxPackets < 1
          ) {
            throw new RangeError(
              `GpuRingBuffer.drainBatch maxPackets must be a positive safe integer, got ${maxPacketsArg.maxPackets}`
            );
          }
          limit = maxPacketsArg.maxPackets;
        } else {
          limit = maxPacketsArg.drainAll === true ? Number.MAX_SAFE_INTEGER : DEFAULT_BATCH_MAX_PACKETS;
        }
        if (maxPacketsArg.reuseResult !== undefined) {
          reuseResult = Boolean(maxPacketsArg.reuseResult);
        }
        if (maxPacketsArg.perPacketCommit !== undefined) {
          perPacketCommit = Boolean(maxPacketsArg.perPacketCommit);
        }
      } else if (
        typeof maxPacketsArg !== 'number' ||
        !Number.isSafeInteger(maxPacketsArg) ||
        maxPacketsArg < 1
      ) {
        throw new RangeError(
          `GpuRingBuffer.drainBatch maxPackets must be a positive safe integer, got ${maxPacketsArg}`
        );
      } else {
        limit = maxPacketsArg;
      }

      const hdrU32 = this._headerU32;
      const hdrI32 = this._headerI32;
      const cap = this._capacityBytes;
      const mask = this._mask;
      const readPos = Atomics.load(hdrU32, HEADER_INDEX.READ_POS);
      const writePos = Atomics.load(hdrU32, HEADER_INDEX.WRITE_POS);
      const avail = (writePos - readPos) >>> 0;
      const curDrainCalls = this._drainCalls;
      this._drainCalls = curDrainCalls >= 0 && curDrainCalls < Infinity ? curDrainCalls + 1 : 1;
      if (avail === 0) {
        const curEmpty = this._emptyDrains;
        this._emptyDrains = curEmpty >= 0 && curEmpty < Infinity ? curEmpty + 1 : 1;
        if (reuseResult) {
          const out = this._reusableBatchResult;
          out.packets = 0;
          out.bytes = 0;
          out.remainingBytes = 0;
          out.wrapped = false;
          return out;
        }
        return { packets: 0, bytes: 0, remainingBytes: 0, wrapped: false };
      }

      const boundedAvail = avail > cap ? cap : avail;
      this._recordOccupancy(boundedAvail);

      const data = this._data;
      const dataView = this._dataView;
      const hdrScratch = this._headerScratch;
      const hdrScratchView = this._headerScratchView;
      const pktDesc = this._packetDescriptor;
      const startOffset = readPos & mask;
      let cursor = 0;
      let completedPackets = 0;
      let completedBytes = 0;
      let committedBytes = 0;

      try {
        while (completedPackets < limit) {
          const remaining = boundedAvail - cursor;
          if (remaining < GPU_PACKET_HEADER_BYTES) {
            break;
          }

          const pktOffset = (readPos + cursor) & mask;
          let opcode;
          let flags;
          let payloadBytes;

          if (pktOffset + GPU_PACKET_HEADER_BYTES <= cap) {
            opcode = dataView.getUint16(pktOffset, true);
            flags = dataView.getUint16(pktOffset + 2, true);
            payloadBytes = dataView.getUint32(pktOffset + 4, true);
          } else {
            const firstHdr = cap - pktOffset;
            for (let i = 0; i < firstHdr; i++) {
              hdrScratch[i] = data[pktOffset + i];
            }
            for (let i = firstHdr; i < GPU_PACKET_HEADER_BYTES; i++) {
              hdrScratch[i] = data[i - firstHdr];
            }
            opcode = hdrScratchView.getUint16(0, true);
            flags = hdrScratchView.getUint16(2, true);
            payloadBytes = hdrScratchView.getUint32(4, true);
          }

          if (payloadBytes > remaining - GPU_PACKET_HEADER_BYTES) {
            if (payloadBytes > cap - GPU_PACKET_HEADER_BYTES) {
              const curCorrupt = this._corruptHeaders;
              this._corruptHeaders = curCorrupt >= 0 && curCorrupt < Infinity ? curCorrupt + 1 : 1;
            }
            break;
          }

          const packetLen = GPU_PACKET_HEADER_BYTES + payloadBytes;
          if (pktOffset + packetLen <= cap) {
            pktDesc.bytes = data.subarray(pktOffset, pktOffset + packetLen);
            pktDesc.payload = data.subarray(
              pktOffset + GPU_PACKET_HEADER_BYTES,
              pktOffset + packetLen
            );
            pktDesc.wrapped = false;
            pktDesc.dataView = dataView;
            pktDesc.viewOffset = pktOffset;
          } else {
            const scratch = this._copyWrappedToScratch(pktOffset, packetLen);
            pktDesc.bytes = scratch.subarray(0, packetLen);
            pktDesc.payload = scratch.subarray(GPU_PACKET_HEADER_BYTES, packetLen);
            pktDesc.wrapped = true;
            pktDesc.dataView = this._wrapScratchView;
            pktDesc.viewOffset = 0;
          }

          pktDesc.opcode = opcode;
          pktDesc.flags = flags;
          pktDesc.payloadBytes = payloadBytes;
          pktDesc.byteLength = packetLen;

          const shouldContinue = handler(pktDesc, completedPackets);

          cursor += packetLen;
          completedBytes = cursor;
          completedPackets += 1;
          if (perPacketCommit) {
            const nextRead = (readPos + completedBytes) >>> 0;
            Atomics.store(hdrU32, HEADER_INDEX.READ_POS, nextRead);
            Atomics.notify(hdrI32, HEADER_INDEX.READ_POS, 1);
            committedBytes = completedBytes;
          }
          if (shouldContinue === false) {
            break;
          }
        }
      } finally {
        if (completedBytes > 0) {
          if (committedBytes < completedBytes) {
            const nextRead = (readPos + completedBytes) >>> 0;
            Atomics.store(hdrU32, HEADER_INDEX.READ_POS, nextRead);
            Atomics.notify(hdrI32, HEADER_INDEX.READ_POS, 1);
          }
          const curTotal = this._totalBytesDrained;
          this._totalBytesDrained =
            (curTotal >= 0 && curTotal < Infinity ? curTotal : 0) + completedBytes;
          const curMaxBytes =
            this._maxDrainBatchBytes >= 0 && this._maxDrainBatchBytes < Infinity
              ? this._maxDrainBatchBytes
              : 0;
          if (completedBytes > curMaxBytes) {
            this._maxDrainBatchBytes = completedBytes;
          }
          const curPeakBatch =
            this._peakBatchBytes >= 0 && this._peakBatchBytes < Infinity
              ? this._peakBatchBytes
              : 0;
          if (completedBytes > curPeakBatch) {
            this._peakBatchBytes = completedBytes;
          }
        }
        if (completedPackets > 0) {
          const curBatches = this._batchesDrained;
          this._batchesDrained = curBatches >= 0 && curBatches < Infinity ? curBatches + 1 : 1;
          const curPackets = this._packetsDrained;
          this._packetsDrained =
            (curPackets >= 0 && curPackets < Infinity ? curPackets : 0) + completedPackets;
          const curMaxBatch =
            this._maxBatchSize >= 0 && this._maxBatchSize < Infinity ? this._maxBatchSize : 0;
          if (completedPackets > curMaxBatch) {
            this._maxBatchSize = completedPackets;
          }
        }
      }

      if (completedBytes === 0) {
        const curEmpty = this._emptyDrains;
        this._emptyDrains = curEmpty >= 0 && curEmpty < Infinity ? curEmpty + 1 : 1;
      }

      const batchWrapped = completedBytes > 0 && startOffset + completedBytes > cap;
      const remainingBytes = boundedAvail - completedBytes;

      if (reuseResult) {
        const out = this._reusableBatchResult;
        out.packets = completedPackets;
        out.bytes = completedBytes;
        out.remainingBytes = remainingBytes;
        out.wrapped = batchWrapped;
        return out;
      }

      return {
        packets: completedPackets,
        bytes: completedBytes,
        remainingBytes,
        wrapped: batchWrapped,
      };
    }

    /**
     * Zero-copy batch drain that drains ALL currently available framed GpuPacketHeader
     * packets in a single pass with zero per-packet descriptor object allocations:
     *   - Contiguous (non-wrapping) packets pass a direct `Uint8Array.subarray(start, end)`
     *     view over the SharedArrayBuffer with ZERO memcpy (or `this._data` + `meta.byteOffset`
     *     when `options.directView === true`).
     *   - Only packets that physically wrap across the end of the ring buffer are linearized
     *     into the reusable growable `_wrapScratch` buffer.
     *   - Advances `READ_POS` with a single `Atomics.store` + `Atomics.notify` at batch end
     *     (or per packet when `options.perPacketCommit === true`).
     *
     * @param {(packetView: Uint8Array, meta: {opcode: number, flags: number, payloadBytes: number, byteOffset: number, byteLength: number, payloadOffset: number, wrapped: boolean, zeroCopy: boolean, dataView: DataView, viewOffset: number}, index: number) => (boolean|void)} callback
     * @param {{maxPackets?: number, maxBytes?: number, directView?: boolean, reuseResult?: boolean, perPacketCommit?: boolean}} [options]
     * @returns {{packets: number, bytes: number, remainingBytes: number, wrapped: boolean, zeroCopy: boolean}}
     */
    drainInto(callback, options) {
      if (typeof callback !== 'function') {
        throw new TypeError('GpuRingBuffer.drainInto requires a callback function');
      }

      let maxPackets = Number.MAX_SAFE_INTEGER;
      let maxBytes = this._capacityBytes;
      let directView = false;
      let reuseResult = false;
      let perPacketCommit = false;

      if (options !== undefined && options !== null) {
        if (typeof options === 'number') {
          if (!Number.isSafeInteger(options) || options < 1) {
            throw new RangeError(
              `GpuRingBuffer.drainInto maxPackets must be a positive safe integer, got ${options}`
            );
          }
          maxPackets = options;
        } else if (typeof options === 'object' && !Array.isArray(options)) {
          if (options.maxPackets !== undefined && options.maxPackets !== null) {
            if (
              typeof options.maxPackets !== 'number' ||
              !Number.isSafeInteger(options.maxPackets) ||
              options.maxPackets < 1
            ) {
              throw new RangeError(
                `GpuRingBuffer.drainInto maxPackets must be a positive safe integer, got ${options.maxPackets}`
              );
            }
            maxPackets = options.maxPackets;
          }
          if (options.maxBytes !== undefined && options.maxBytes !== null) {
            if (
              typeof options.maxBytes !== 'number' ||
              !Number.isSafeInteger(options.maxBytes) ||
              options.maxBytes < 1
            ) {
              throw new RangeError(
                `GpuRingBuffer.drainInto maxBytes must be a positive safe integer, got ${options.maxBytes}`
              );
            }
            maxBytes = Math.min(this._capacityBytes, options.maxBytes);
          }
          if (options.directView !== undefined) {
            directView = Boolean(options.directView);
          }
          if (options.reuseResult !== undefined) {
            reuseResult = Boolean(options.reuseResult);
          }
          if (options.perPacketCommit !== undefined) {
            perPacketCommit = Boolean(options.perPacketCommit);
          }
        } else {
          throw new TypeError(
            'GpuRingBuffer.drainInto options must be an options object or positive integer maxPackets'
          );
        }
      }

      const hdrU32 = this._headerU32;
      const hdrI32 = this._headerI32;
      const cap = this._capacityBytes;
      const mask = this._mask;
      const readPos = Atomics.load(hdrU32, HEADER_INDEX.READ_POS);
      const writePos = Atomics.load(hdrU32, HEADER_INDEX.WRITE_POS);
      const avail = (writePos - readPos) >>> 0;
      const curDrainCalls = this._drainCalls;
      this._drainCalls = curDrainCalls >= 0 && curDrainCalls < Infinity ? curDrainCalls + 1 : 1;

      if (avail === 0) {
        const curEmpty = this._emptyDrains;
        this._emptyDrains = curEmpty >= 0 && curEmpty < Infinity ? curEmpty + 1 : 1;
        if (reuseResult) {
          const out = this._reusableDrainResult;
          out.packets = 0;
          out.bytes = 0;
          out.remainingBytes = 0;
          out.wrapped = false;
          out.zeroCopy = false;
          return out;
        }
        return { packets: 0, bytes: 0, remainingBytes: 0, wrapped: false, zeroCopy: false };
      }

      const boundedAvail = avail > cap ? cap : avail;
      this._recordOccupancy(boundedAvail);

      const data = this._data;
      const dataView = this._dataView;
      const hdrScratch = this._headerScratch;
      const hdrScratchView = this._headerScratchView;
      const meta = this._drainIntoMeta;
      const startOffset = readPos & mask;
      let cursor = 0;
      let completedPackets = 0;
      let completedBytes = 0;
      let committedBytes = 0;
      let anyPacketWrapped = false;

      try {
        while (completedPackets < maxPackets) {
          const remaining = boundedAvail - cursor;
          if (remaining < GPU_PACKET_HEADER_BYTES) {
            break;
          }

          const pktOffset = (readPos + cursor) & mask;
          let opcode;
          let flags;
          let payloadBytes;

          if (pktOffset + GPU_PACKET_HEADER_BYTES <= cap) {
            opcode = dataView.getUint16(pktOffset, true);
            flags = dataView.getUint16(pktOffset + 2, true);
            payloadBytes = dataView.getUint32(pktOffset + 4, true);
          } else {
            const firstHdr = cap - pktOffset;
            for (let i = 0; i < firstHdr; i++) {
              hdrScratch[i] = data[pktOffset + i];
            }
            for (let i = firstHdr; i < GPU_PACKET_HEADER_BYTES; i++) {
              hdrScratch[i] = data[i - firstHdr];
            }
            opcode = hdrScratchView.getUint16(0, true);
            flags = hdrScratchView.getUint16(2, true);
            payloadBytes = hdrScratchView.getUint32(4, true);
          }

          if (payloadBytes > remaining - GPU_PACKET_HEADER_BYTES) {
            if (payloadBytes > cap - GPU_PACKET_HEADER_BYTES) {
              const curCorrupt = this._corruptHeaders;
              this._corruptHeaders = curCorrupt >= 0 && curCorrupt < Infinity ? curCorrupt + 1 : 1;
            }
            break;
          }

          const packetLen = GPU_PACKET_HEADER_BYTES + payloadBytes;
          if (completedPackets > 0 && cursor + packetLen > maxBytes) {
            break;
          }

          let packetView;
          if (pktOffset + packetLen <= cap) {
            if (directView) {
              packetView = data;
              meta.byteOffset = pktOffset;
              meta.payloadOffset = pktOffset + GPU_PACKET_HEADER_BYTES;
            } else {
              packetView = data.subarray(pktOffset, pktOffset + packetLen);
              meta.byteOffset = 0;
              meta.payloadOffset = GPU_PACKET_HEADER_BYTES;
            }
            meta.wrapped = false;
            meta.zeroCopy = true;
            meta.dataView = dataView;
            meta.viewOffset = pktOffset;
          } else {
            anyPacketWrapped = true;
            const scratch = this._copyWrappedToScratch(pktOffset, packetLen);
            packetView = directView ? scratch : scratch.subarray(0, packetLen);
            meta.byteOffset = 0;
            meta.payloadOffset = GPU_PACKET_HEADER_BYTES;
            meta.wrapped = true;
            meta.zeroCopy = false;
            meta.dataView = this._wrapScratchView;
            meta.viewOffset = 0;
          }

          meta.opcode = opcode;
          meta.flags = flags;
          meta.payloadBytes = payloadBytes;
          meta.byteLength = packetLen;

          const shouldContinue = callback(packetView, meta, completedPackets);

          cursor += packetLen;
          completedBytes = cursor;
          completedPackets += 1;
          if (perPacketCommit) {
            const nextRead = (readPos + completedBytes) >>> 0;
            Atomics.store(hdrU32, HEADER_INDEX.READ_POS, nextRead);
            Atomics.notify(hdrI32, HEADER_INDEX.READ_POS, 1);
            committedBytes = completedBytes;
          }
          if (shouldContinue === false) {
            break;
          }
        }
      } finally {
        if (completedBytes > 0) {
          if (committedBytes < completedBytes) {
            const nextRead = (readPos + completedBytes) >>> 0;
            Atomics.store(hdrU32, HEADER_INDEX.READ_POS, nextRead);
            Atomics.notify(hdrI32, HEADER_INDEX.READ_POS, 1);
          }
          const curTotal = this._totalBytesDrained;
          this._totalBytesDrained =
            (curTotal >= 0 && curTotal < Infinity ? curTotal : 0) + completedBytes;
          const curMaxBytes =
            this._maxDrainBatchBytes >= 0 && this._maxDrainBatchBytes < Infinity
              ? this._maxDrainBatchBytes
              : 0;
          if (completedBytes > curMaxBytes) {
            this._maxDrainBatchBytes = completedBytes;
          }
          const curPeakBatch =
            this._peakBatchBytes >= 0 && this._peakBatchBytes < Infinity
              ? this._peakBatchBytes
              : 0;
          if (completedBytes > curPeakBatch) {
            this._peakBatchBytes = completedBytes;
          }
          const curZeroAlloc = this._zeroAllocDrains;
          this._zeroAllocDrains = curZeroAlloc >= 0 && curZeroAlloc < Infinity ? curZeroAlloc + 1 : 1;
        }
        if (completedPackets > 0) {
          const curBatches = this._batchesDrained;
          this._batchesDrained = curBatches >= 0 && curBatches < Infinity ? curBatches + 1 : 1;
          const curPackets = this._packetsDrained;
          this._packetsDrained =
            (curPackets >= 0 && curPackets < Infinity ? curPackets : 0) + completedPackets;
          const curMaxBatch =
            this._maxBatchSize >= 0 && this._maxBatchSize < Infinity ? this._maxBatchSize : 0;
          if (completedPackets > curMaxBatch) {
            this._maxBatchSize = completedPackets;
          }
        }
      }

      if (completedBytes === 0) {
        const curEmpty = this._emptyDrains;
        this._emptyDrains = curEmpty >= 0 && curEmpty < Infinity ? curEmpty + 1 : 1;
      }

      const batchWrapped = completedBytes > 0 && startOffset + completedBytes > cap;
      const zeroCopy = completedBytes > 0 && !anyPacketWrapped;
      const remainingBytes = boundedAvail - completedBytes;

      if (reuseResult) {
        const out = this._reusableDrainResult;
        out.packets = completedPackets;
        out.bytes = completedBytes;
        out.remainingBytes = remainingBytes;
        out.wrapped = batchWrapped;
        out.zeroCopy = zeroCopy;
        return out;
      }

      return {
        packets: completedPackets,
        bytes: completedBytes,
        remainingBytes,
        wrapped: batchWrapped,
        zeroCopy,
      };
    }

    /**
     * Zero-alloc consumer batch span drain optimized for high-throughput 30 FPS
     * command-ring consumption (Main Menu & 3D world frames).
     *
     * Scans up to `maxPackets` complete framed GpuPacketHeader packets (bounded by
     * `maxBytes`) without allocating per-packet `Uint8Array` views or wrapper objects,
     * then invokes `handler(bytesView, byteOffset, byteLength, spanDescriptor)` once
     * for the entire drained span:
     *   - Non-wrapping spans pass a direct zero-copy `subarray` view over the SAB.
     *   - Wrap-around spans linearize into the reusable pre-allocated `_wrapScratch`
     *     buffer with zero per-batch backing allocations.
     *   - Reads `write_pos` / `read_pos` once and publishes `read_pos` once on completion.
     *   - If `handler` throws, `read_pos` is left unchanged so the span remains unread.
     *
     * @param {(bytesView: Uint8Array, byteOffset: number, byteLength: number, span: {bytes: Uint8Array, byteOffset: number, byteLength: number, bufferByteOffset: number, ringOffset: number, packets: number, wrapped: boolean, zeroCopy: boolean}) => void} handler
     * @param {number | {maxPackets?: number, maxBytes?: number, framed?: boolean, reuseResult?: boolean}} [options]
     * @returns {{packets: number, bytes: number, remainingBytes: number, wrapped: boolean, zeroCopy: boolean}}
     */
    drainBatchZeroAlloc(handler, options) {
      if (typeof handler !== 'function') {
        throw new TypeError('GpuRingBuffer.drainBatchZeroAlloc requires a handler function');
      }

      let maxPackets = DEFAULT_BATCH_MAX_PACKETS;
      let maxBytes = this._capacityBytes;
      let framed = true;
      let reuseResult = false;
      let directView = false;
      let stopAtWrap = false;

      if (options !== undefined && options !== null) {
        if (typeof options === 'number') {
          if (!Number.isSafeInteger(options) || options < 1) {
            throw new RangeError(
              `GpuRingBuffer.drainBatchZeroAlloc maxPackets must be a positive safe integer, got ${options}`
            );
          }
          maxPackets = options;
        } else if (typeof options === 'object') {
          if (options.maxPackets !== undefined && options.maxPackets !== null) {
            if (
              typeof options.maxPackets !== 'number' ||
              !Number.isSafeInteger(options.maxPackets) ||
              options.maxPackets < 1
            ) {
              throw new RangeError(
                `GpuRingBuffer.drainBatchZeroAlloc maxPackets must be a positive safe integer, got ${options.maxPackets}`
              );
            }
            maxPackets = options.maxPackets;
          }
          if (options.maxBytes !== undefined && options.maxBytes !== null) {
            if (
              typeof options.maxBytes !== 'number' ||
              !Number.isSafeInteger(options.maxBytes) ||
              options.maxBytes < 1
            ) {
              throw new RangeError(
                `GpuRingBuffer.drainBatchZeroAlloc maxBytes must be a positive safe integer, got ${options.maxBytes}`
              );
            }
            maxBytes = Math.min(this._capacityBytes, options.maxBytes);
          }
          if (options.framed !== undefined) {
            framed = Boolean(options.framed);
          }
          if (options.reuseResult !== undefined) {
            reuseResult = Boolean(options.reuseResult);
          }
          if (options.directView !== undefined) {
            directView = Boolean(options.directView);
          }
          if (options.stopAtWrap !== undefined) {
            stopAtWrap = Boolean(options.stopAtWrap);
          }
        } else {
          throw new TypeError(
            'GpuRingBuffer.drainBatchZeroAlloc options must be an options object or positive integer maxPackets'
          );
        }
      }

      const readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      const writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      const avail = (writePos - readPos) >>> 0;
      const curDrainCalls = this._drainCalls;
      this._drainCalls = curDrainCalls >= 0 && curDrainCalls < Infinity ? curDrainCalls + 1 : 1;

      if (avail === 0) {
        const curEmpty = this._emptyDrains;
        this._emptyDrains = curEmpty >= 0 && curEmpty < Infinity ? curEmpty + 1 : 1;
        if (reuseResult) {
          this._reusableDrainResult.packets = 0;
          this._reusableDrainResult.bytes = 0;
          this._reusableDrainResult.remainingBytes = 0;
          this._reusableDrainResult.wrapped = false;
          this._reusableDrainResult.zeroCopy = false;
          return this._reusableDrainResult;
        }
        return { packets: 0, bytes: 0, remainingBytes: 0, wrapped: false, zeroCopy: false };
      }

      const boundedAvail = avail > this._capacityBytes ? this._capacityBytes : avail;
      this._recordOccupancy(boundedAvail);

      const cap = this._capacityBytes;
      const mask = this._mask;
      const data = this._data;
      const dataView = this._dataView;
      const hdrScratch = this._headerScratch;
      const hdrScratchView = this._headerScratchView;
      const startOffset = readPos & mask;
      let spanBytes = 0;
      let packetCount = 0;

      if (framed) {
        while (packetCount < maxPackets) {
          const remaining = boundedAvail - spanBytes;
          if (remaining < GPU_PACKET_HEADER_BYTES) {
            break;
          }

          const pktOffset = (readPos + spanBytes) & mask;
          let payloadBytes;
          if (pktOffset + GPU_PACKET_HEADER_BYTES <= cap) {
            payloadBytes = dataView.getUint32(pktOffset + 4, true);
          } else {
            const firstHdr = cap - pktOffset;
            for (let i = 0; i < firstHdr; i++) {
              hdrScratch[i] = data[pktOffset + i];
            }
            for (let i = firstHdr; i < GPU_PACKET_HEADER_BYTES; i++) {
              hdrScratch[i] = data[i - firstHdr];
            }
            payloadBytes = hdrScratchView.getUint32(4, true);
          }

          if (payloadBytes > remaining - GPU_PACKET_HEADER_BYTES) {
            if (payloadBytes > cap - GPU_PACKET_HEADER_BYTES) {
              const curCorrupt = this._corruptHeaders;
              this._corruptHeaders = curCorrupt >= 0 && curCorrupt < Infinity ? curCorrupt + 1 : 1;
            }
            break;
          }

          const packetLen = GPU_PACKET_HEADER_BYTES + payloadBytes;
          if (packetCount > 0) {
            if (spanBytes + packetLen > maxBytes) {
              break;
            }
            if (stopAtWrap && startOffset + spanBytes + packetLen > cap) {
              break;
            }
          }

          spanBytes += packetLen;
          packetCount += 1;
        }
      } else {
        spanBytes = boundedAvail > maxBytes ? maxBytes : boundedAvail;
      }

      if (spanBytes === 0) {
        const curEmpty = this._emptyDrains;
        this._emptyDrains = curEmpty >= 0 && curEmpty < Infinity ? curEmpty + 1 : 1;
        if (reuseResult) {
          this._reusableDrainResult.packets = 0;
          this._reusableDrainResult.bytes = 0;
          this._reusableDrainResult.remainingBytes = boundedAvail;
          this._reusableDrainResult.wrapped = false;
          this._reusableDrainResult.zeroCopy = false;
          return this._reusableDrainResult;
        }
        return {
          packets: 0,
          bytes: 0,
          remainingBytes: boundedAvail,
          wrapped: false,
          zeroCopy: false,
        };
      }

      let bytesView;
      let viewOffset = 0;
      let spanDataView;
      let wrapped = false;
      if (startOffset + spanBytes <= this._capacityBytes) {
        spanDataView = this._dataView;
        if (directView) {
          bytesView = this._data;
          viewOffset = startOffset;
        } else {
          bytesView = this._data.subarray(startOffset, startOffset + spanBytes);
          viewOffset = 0;
        }
      } else {
        wrapped = true;
        const scratch = this._copyWrappedToScratch(startOffset, spanBytes);
        spanDataView = this._wrapScratchView;
        if (directView) {
          bytesView = scratch;
          viewOffset = 0;
        } else {
          bytesView = scratch.subarray(0, spanBytes);
          viewOffset = 0;
        }
      }

      const zeroCopy = !wrapped;
      const span = this._spanDescriptor;
      span.bytes = bytesView;
      span.byteOffset = viewOffset;
      span.byteLength = spanBytes;
      span.bufferByteOffset = bytesView.byteOffset + viewOffset;
      span.ringOffset = startOffset;
      span.packets = packetCount;
      span.wrapped = wrapped;
      span.zeroCopy = zeroCopy;
      span.dataView = spanDataView;
      span.viewOffset = wrapped ? 0 : startOffset;

      // Only advance READ_POS and telemetry if handler succeeds without throwing.
      handler(bytesView, viewOffset, spanBytes, span);

      const nextRead = (readPos + spanBytes) >>> 0;
      Atomics.store(this._headerU32, HEADER_INDEX.READ_POS, nextRead);
      Atomics.notify(this._headerI32, HEADER_INDEX.READ_POS, 1);

      const curTotal = this._totalBytesDrained;
      this._totalBytesDrained =
        (curTotal >= 0 && curTotal < Infinity ? curTotal : 0) + spanBytes;
      const curMaxBytes =
        this._maxDrainBatchBytes >= 0 && this._maxDrainBatchBytes < Infinity
          ? this._maxDrainBatchBytes
          : 0;
      if (spanBytes > curMaxBytes) {
        this._maxDrainBatchBytes = spanBytes;
      }
      const curPeakBatch =
        this._peakBatchBytes >= 0 && this._peakBatchBytes < Infinity
          ? this._peakBatchBytes
          : 0;
      if (spanBytes > curPeakBatch) {
        this._peakBatchBytes = spanBytes;
      }
      const curZeroAlloc = this._zeroAllocDrains;
      this._zeroAllocDrains = curZeroAlloc >= 0 && curZeroAlloc < Infinity ? curZeroAlloc + 1 : 1;
      const curBatches = this._batchesDrained;
      this._batchesDrained = curBatches >= 0 && curBatches < Infinity ? curBatches + 1 : 1;
      if (packetCount > 0) {
        const curPackets = this._packetsDrained;
        this._packetsDrained =
          (curPackets >= 0 && curPackets < Infinity ? curPackets : 0) + packetCount;
        const curMaxBatch =
          this._maxBatchSize >= 0 && this._maxBatchSize < Infinity ? this._maxBatchSize : 0;
        if (packetCount > curMaxBatch) {
          this._maxBatchSize = packetCount;
        }
      }

      const remainingBytes = boundedAvail - spanBytes;
      if (reuseResult) {
        this._reusableDrainResult.packets = packetCount;
        this._reusableDrainResult.bytes = spanBytes;
        this._reusableDrainResult.remainingBytes = remainingBytes;
        this._reusableDrainResult.wrapped = wrapped;
        this._reusableDrainResult.zeroCopy = zeroCopy;
        return this._reusableDrainResult;
      }
      return {
        packets: packetCount,
        bytes: spanBytes,
        remainingBytes,
        wrapped,
        zeroCopy,
      };
    }

    resetStats() {
      this._batchesDrained = 0;
      this._packetsDrained = 0;
      this._maxBatchSize = 0;
      this._backpressureWaits = 0;
      this._backpressureStalls = 0;
      this._writesSucceeded = 0;
      this._peakOccupancyBytes = 0;
      this._drainCalls = 0;
      this._maxDrainBatchBytes = 0;
      this._peakBatchBytes = 0;
      this._zeroAllocDrains = 0;
      this._totalBytesDrained = 0;
      this._emptyDrains = 0;
      this._corruptHeaders = 0;
    }

    resetTelemetry() {
      this.resetStats();
    }

    reset() {
      Atomics.store(this._headerU32, HEADER_INDEX.WRITE_POS, 0);
      Atomics.store(this._headerU32, HEADER_INDEX.READ_POS, 0);
      Atomics.store(this._headerU32, HEADER_INDEX.DROPPED_PACKETS, 0);
      Atomics.store(this._headerU32, HEADER_INDEX.NOTIFY_SEQ, 0);
      this.resetStats();
    }

    getStats() {
      const { readPos, writePos, avail: availableBytes } = this._snapshotPositions();
      this._recordOccupancy(availableBytes);
      const freeBytes =
        availableBytes >= this._capacityBytes ? 0 : this._capacityBytes - availableBytes;
      const batchesDrained = sanitizeNonNegInt(this._batchesDrained);
      const packetsDrained = sanitizeNonNegInt(this._packetsDrained);
      const maxBatchSize = sanitizeNonNegInt(this._maxBatchSize);
      const avgBatchSize =
        batchesDrained > 0 ? sanitizeNonNegNumber(packetsDrained / batchesDrained) : 0;
      const backpressureWaits = sanitizeNonNegInt(this._backpressureWaits);
      const backpressureStalls = sanitizeNonNegInt(this._backpressureStalls);
      const writesSucceeded = sanitizeNonNegInt(this._writesSucceeded);
      const totalWriteEvents = writesSucceeded + backpressureWaits;
      const backpressureWaitRate =
        totalWriteEvents > 0 ? clampUnitRatio(backpressureWaits / totalWriteEvents) : 0;
      const drainCalls = sanitizeNonNegInt(this._drainCalls);
      const maxDrainBatchBytes = Math.min(
        this._capacityBytes,
        sanitizeNonNegInt(this._maxDrainBatchBytes)
      );
      const peakBatchBytes = Math.min(
        this._capacityBytes,
        sanitizeNonNegInt(this._peakBatchBytes)
      );
      const zeroAllocDrains = sanitizeNonNegInt(this._zeroAllocDrains);
      const totalBytesDrained = sanitizeNonNegInt(this._totalBytesDrained);
      const emptyDrains = sanitizeNonNegInt(this._emptyDrains);
      const peakOccupancyBytes = Math.min(
        this._capacityBytes,
        sanitizeNonNegInt(this._peakOccupancyBytes)
      );
      const highWatermarkBytes = peakOccupancyBytes;
      const peakOccupancyRatio =
        this._capacityBytes > 0
          ? clampUnitRatio(highWatermarkBytes / this._capacityBytes)
          : 0;
      const peakUtilizationPct =
        this._capacityBytes > 0
          ? Math.min(
              100,
              sanitizeNonNegNumber((highWatermarkBytes / this._capacityBytes) * 100)
            )
          : 0;
      const clampedAvailable = Math.min(this._capacityBytes, availableBytes);
      const utilizationPct =
        this._capacityBytes > 0
          ? Math.min(
              100,
              sanitizeNonNegNumber((clampedAvailable / this._capacityBytes) * 100)
            )
          : 0;
      return {
        capacityBytes: this._capacityBytes,
        availableBytes,
        freeBytes,
        writePos,
        readPos,
        droppedPackets: this.droppedPackets,
        notifySeq: this.notifySeq,
        batchesDrained,
        maxBatchSize,
        avgBatchSize,
        packetsDrained,
        backpressureWaits,
        backpressureStalls,
        backpressureWaitRate,
        drainCalls,
        maxDrainBatchBytes,
        peakBatchBytes,
        zeroAllocDrains,
        totalBytesDrained,
        emptyDrains,
        corruptHeaders: sanitizeNonNegInt(this._corruptHeaders),
        peakOccupancyBytes,
        peakOccupancyRatio,
        highWatermarkBytes,
        peakUtilizationPct,
        utilizationPct,
      };
    }
  }

  const exportsObj = {
    GpuRingBuffer,
    GPU_RING_MAGIC,
    GPU_RING_VERSION,
    GPU_RING_HEADER_BYTES,
    GPU_RING_HEADER_WORDS,
    GPU_PACKET_HEADER_BYTES,
    DEFAULT_CAPACITY_BYTES,
    DEFAULT_BATCH_MAX_PACKETS,
    DEFAULT_WRAP_SCRATCH_BYTES,
    HEADER_INDEX,
    isPowerOfTwo,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6GpuRing = exportsObj;
  } else if (root) {
    root.PlayGta6GpuRing = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
