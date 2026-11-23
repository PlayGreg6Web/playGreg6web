'use strict';

(function (root) {
  // Binary-compatible with playgta6::gpu::GpuRingHeader (runtime/gpu/gpu_ring.h).
  const GPU_RING_MAGIC = 0x47505552; // "GPUR"
  const GPU_RING_VERSION = 1;
  const GPU_RING_HEADER_BYTES = 64;
  const GPU_RING_HEADER_WORDS = 16;
  const DEFAULT_CAPACITY_BYTES = 65536;

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
      if (!Number.isInteger(byteOffset) || byteOffset < 0 || (byteOffset & 3) !== 0) {
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
      this._scratch = null;
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

    availableBytes() {
      const writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      const readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      return (writePos - readPos) >>> 0;
    }

    freeBytes() {
      const used = this.availableBytes();
      return used >= this._capacityBytes ? 0 : this._capacityBytes - used;
    }

    /**
     * All-or-nothing byte write supporting wrap-around.
     * Returns false (and increments dropped_packets) without modifying write_pos
     * or ring contents when the ring does not have enough free bytes.
     *
     * @param {Uint8Array|ArrayBufferView|ArrayBuffer|SharedArrayBuffer} input
     * @returns {boolean}
     */
    writeBytes(input) {
      const src = toUint8Array(input);
      const len = src.byteLength;
      if (len === 0) {
        return true;
      }
      if (len > this._capacityBytes) {
        Atomics.add(this._headerU32, HEADER_INDEX.DROPPED_PACKETS, 1);
        return false;
      }

      const writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      const readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      const used = (writePos - readPos) >>> 0;
      const free = this._capacityBytes - used;
      if (len > free) {
        Atomics.add(this._headerU32, HEADER_INDEX.DROPPED_PACKETS, 1);
        return false;
      }

      const offset = (writePos & this._mask) >>> 0;
      const firstLen = Math.min(len, this._capacityBytes - offset);
      this._data.set(src.subarray(0, firstLen), offset);
      if (len > firstLen) {
        this._data.set(src.subarray(firstLen, len), 0);
      }

      const nextWrite = (writePos + len) >>> 0;
      Atomics.store(this._headerU32, HEADER_INDEX.WRITE_POS, nextWrite);
      Atomics.add(this._headerI32, HEADER_INDEX.NOTIFY_SEQ, 1);
      Atomics.notify(this._headerI32, HEADER_INDEX.NOTIFY_SEQ, 1);
      return true;
    }

    /**
     * Drains all currently queued bytes and passes a single contiguous Uint8Array
     * to `callback`.
     * - Zero-copy subarray directly over the SharedArrayBuffer when not wrapping.
     * - Linearized into a reusable scratch Uint8Array on wrap-around so framed
     *   GpuPacketHeader streams are never split or corrupted at the ring boundary.
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
      if (avail === 0) {
        return 0;
      }

      const offset = (readPos & this._mask) >>> 0;
      let slice;
      if (offset + avail <= this._capacityBytes) {
        slice = this._data.subarray(offset, offset + avail);
      } else {
        if (!this._scratch || this._scratch.byteLength < avail) {
          this._scratch = new Uint8Array(avail);
        }
        const firstLen = this._capacityBytes - offset;
        const secondLen = avail - firstLen;
        this._scratch.set(this._data.subarray(offset, this._capacityBytes), 0);
        this._scratch.set(this._data.subarray(0, secondLen), firstLen);
        slice = this._scratch.subarray(0, avail);
      }

      const nextRead = (readPos + avail) >>> 0;
      try {
        callback(slice);
      } finally {
        Atomics.store(this._headerU32, HEADER_INDEX.READ_POS, nextRead);
      }
      return avail;
    }

    reset() {
      Atomics.store(this._headerU32, HEADER_INDEX.WRITE_POS, 0);
      Atomics.store(this._headerU32, HEADER_INDEX.READ_POS, 0);
      Atomics.store(this._headerU32, HEADER_INDEX.DROPPED_PACKETS, 0);
      Atomics.store(this._headerU32, HEADER_INDEX.NOTIFY_SEQ, 0);
    }

    getStats() {
      const writePos = this.writePos;
      const readPos = this.readPos;
      const availableBytes = (writePos - readPos) >>> 0;
      const freeBytes =
        availableBytes >= this._capacityBytes ? 0 : this._capacityBytes - availableBytes;
      return {
        capacityBytes: this._capacityBytes,
        availableBytes,
        freeBytes,
        writePos,
        readPos,
        droppedPackets: this.droppedPackets,
        notifySeq: this.notifySeq,
      };
    }
  }

  const exportsObj = {
    GpuRingBuffer,
    GPU_RING_MAGIC,
    GPU_RING_VERSION,
    GPU_RING_HEADER_BYTES,
    GPU_RING_HEADER_WORDS,
    DEFAULT_CAPACITY_BYTES,
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
