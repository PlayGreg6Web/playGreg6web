'use strict';

(function (root) {
  const DEFAULT_SAMPLE_RATE = 48000;
  const DEFAULT_CHANNELS = 2;
  const DEFAULT_CAPACITY_FRAMES = 4096;

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

  function toFloat32Array(samples) {
    if (samples instanceof Float32Array) {
      return samples;
    }
    if (samples instanceof ArrayBuffer) {
      return new Float32Array(samples);
    }
    if (ArrayBuffer.isView(samples)) {
      return new Float32Array(
        samples.buffer,
        samples.byteOffset,
        Math.floor(samples.byteLength / Float32Array.BYTES_PER_ELEMENT)
      );
    }
    if (Array.isArray(samples)) {
      return Float32Array.from(samples);
    }
    throw new TypeError('PCM samples must be a Float32Array, ArrayBuffer, TypedArray, or number[]');
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

    static create(capacityFrames = DEFAULT_CAPACITY_FRAMES, channels = DEFAULT_CHANNELS) {
      const byteLen = AudioRingBuffer.byteLength(capacityFrames, channels);
      const sab = new SharedArrayBuffer(byteLen);
      return new AudioRingBuffer(sab, { capacityFrames, channels, initialize: true });
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

    get underruns() {
      return Atomics.load(this._ctrl, HDR_UNDERRUNS);
    }

    get underrunFrames() {
      return Atomics.load(this._ctrl, HDR_UNDERRUN_FRAMES);
    }

    get overflows() {
      return Atomics.load(this._ctrl, HDR_OVERFLOWS);
    }

    get droppedFrames() {
      return Atomics.load(this._ctrl, HDR_DROPPED_FRAMES);
    }

    setVolume(volume) {
      this._volume = clampVolume(volume);
      return this._volume;
    }

    setMuted(muted) {
      this._muted = Boolean(muted);
      return this._muted;
    }

    effectiveGain() {
      return this._muted ? 0.0 : this._volume;
    }

    availableFrames() {
      const write = Atomics.load(this._ctrl, HDR_WRITE_CURSOR);
      const read = Atomics.load(this._ctrl, HDR_READ_CURSOR);
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }
      return used;
    }

    freeFrames() {
      return this._capacityFrames - this.availableFrames();
    }

    /**
     * Pushes interleaved PCM float samples into the ring buffer.
     * Applies overflow backpressure: writes as many complete frames as fit,
     * never overwrites unread audio, and increments overflow counters on drop.
     *
     * @param {Float32Array|ArrayBuffer|number[]} samples Interleaved PCM samples.
     * @returns {number} Number of complete frames written.
     */
    pushSamples(samples) {
      const pcm = toFloat32Array(samples);
      const channels = this._channels;
      const inputFrames = Math.floor(pcm.length / channels);
      if (inputFrames <= 0) {
        return 0;
      }

      const cap = this._capacityFrames;
      const write = Atomics.load(this._ctrl, HDR_WRITE_CURSOR);
      const read = Atomics.load(this._ctrl, HDR_READ_CURSOR);
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
      }

      const free = cap - used;
      const framesToWrite = inputFrames <= free ? inputFrames : free;

      if (framesToWrite > 0) {
        const writeSlot = write >= cap ? write - cap : write;
        const firstFrames = Math.min(framesToWrite, cap - writeSlot);
        const secondFrames = framesToWrite - firstFrames;

        const firstSampleCount = firstFrames * channels;
        this._samples.set(
          pcm.subarray(0, firstSampleCount),
          writeSlot * channels
        );

        if (secondFrames > 0) {
          this._samples.set(
            pcm.subarray(firstSampleCount, framesToWrite * channels),
            0
          );
        }

        let nextWrite = write + framesToWrite;
        if (nextWrite >= this._modulus) {
          nextWrite -= this._modulus;
        }
        Atomics.store(this._ctrl, HDR_WRITE_CURSOR, nextWrite);
      }

      if (framesToWrite < inputFrames) {
        Atomics.add(this._ctrl, HDR_OVERFLOWS, 1);
        Atomics.add(this._ctrl, HDR_DROPPED_FRAMES, inputFrames - framesToWrite);
      }

      return framesToWrite;
    }

    /**
     * Pulls interleaved PCM float samples out of the ring buffer into `output`.
     * Applies master volume / mute gain. On underrun, zero-fills remaining
     * frames and increments atomic underrun counters.
     *
     * @param {Float32Array} output Destination buffer (length = frames * channels).
     * @param {number} [gainOverride] Optional explicit gain override in [0, 1].
     * @returns {number} Number of real frames pulled before any zero-fill.
     */
    pullSamples(output, gainOverride) {
      if (!(output instanceof Float32Array)) {
        throw new TypeError('pullSamples target must be a Float32Array');
      }

      const channels = this._channels;
      const requestedFrames = Math.floor(output.length / channels);
      if (requestedFrames <= 0) {
        return 0;
      }

      const cap = this._capacityFrames;
      const read = Atomics.load(this._ctrl, HDR_READ_CURSOR);
      const write = Atomics.load(this._ctrl, HDR_WRITE_CURSOR);
      let used = write - read;
      if (used < 0) {
        used += this._modulus;
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

        if (gain === 0.0) {
          output.fill(0, 0, framesToRead * channels);
        } else if (gain === 1.0) {
          output.set(
            this._samples.subarray(readSlot * channels, readSlot * channels + firstSamples),
            0
          );
          if (secondSamples > 0) {
            output.set(this._samples.subarray(0, secondSamples), firstSamples);
          }
        } else {
          const srcBase = readSlot * channels;
          for (let i = 0; i < firstSamples; i++) {
            output[i] = this._samples[srcBase + i] * gain;
          }
          for (let i = 0; i < secondSamples; i++) {
            output[firstSamples + i] = this._samples[i] * gain;
          }
        }

        let nextRead = read + framesToRead;
        if (nextRead >= this._modulus) {
          nextRead -= this._modulus;
        }
        Atomics.store(this._ctrl, HDR_READ_CURSOR, nextRead);
      }

      if (framesToRead < requestedFrames) {
        output.fill(0, framesToRead * channels, requestedFrames * channels);
        Atomics.add(this._ctrl, HDR_UNDERRUNS, 1);
        Atomics.add(this._ctrl, HDR_UNDERRUN_FRAMES, requestedFrames - framesToRead);
      }

      return framesToRead;
    }

    reset() {
      Atomics.store(this._ctrl, HDR_WRITE_CURSOR, 0);
      Atomics.store(this._ctrl, HDR_READ_CURSOR, 0);
      Atomics.store(this._ctrl, HDR_UNDERRUNS, 0);
      Atomics.store(this._ctrl, HDR_UNDERRUN_FRAMES, 0);
      Atomics.store(this._ctrl, HDR_OVERFLOWS, 0);
      Atomics.store(this._ctrl, HDR_DROPPED_FRAMES, 0);
    }

    getStats() {
      return {
        capacityFrames: this._capacityFrames,
        channels: this._channels,
        availableFrames: this.availableFrames(),
        freeFrames: this.freeFrames(),
        volume: this._volume,
        muted: this._muted,
        effectiveGain: this.effectiveGain(),
        underruns: this.underruns,
        underrunFrames: this.underrunFrames,
        overflows: this.overflows,
        droppedFrames: this.droppedFrames,
      };
    }
  }

  /**
   * Audio mixer worker controller managing initialization, SAB ring buffer I/O,
   * master volume/mute state, and diagnostics messages.
   */
  class AudioMixerWorker {
    constructor(postMessageFn = null) {
      this._postMessage =
        typeof postMessageFn === 'function' ? postMessageFn : () => {};
      this.initialized = false;
      this.sampleRate = DEFAULT_SAMPLE_RATE;
      this.channels = DEFAULT_CHANNELS;
      this.volume = 1.0;
      this.muted = false;
      this.ring = null;
      this.totalPushedFrames = 0;
      this.totalPulledFrames = 0;
    }

    effectiveGain() {
      return this.muted ? 0.0 : this.volume;
    }

    pullSamples(output) {
      if (!this.ring) {
        if (output instanceof Float32Array) {
          output.fill(0);
        }
        return 0;
      }
      const framesRead = this.ring.pullSamples(output);
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
            volume: this.volume,
            muted: this.muted,
            effectiveGain: this.effectiveGain(),
            underruns: 0,
            underrunFrames: 0,
            overflows: 0,
            droppedFrames: 0,
          };

      return {
        initialized: this.initialized,
        sampleRate: this.sampleRate,
        channels: this.channels,
        capacityFrames: ringStats.capacityFrames,
        availableFrames: ringStats.availableFrames,
        freeFrames: ringStats.freeFrames,
        volume: this.volume,
        muted: this.muted,
        effectiveGain: this.effectiveGain(),
        underruns: ringStats.underruns,
        underrunFrames: ringStats.underrunFrames,
        overflows: ringStats.overflows,
        droppedFrames: ringStats.droppedFrames,
        totalPushedFrames: this.totalPushedFrames,
        totalPulledFrames: this.totalPulledFrames,
      };
    }

    handleMessage(msg) {
      if (!msg || typeof msg !== 'object') {
        return;
      }

      if (msg.init) {
        this.sampleRate =
          typeof msg.sampleRate === 'number' && msg.sampleRate > 0
            ? msg.sampleRate
            : DEFAULT_SAMPLE_RATE;
        this.channels =
          Number.isInteger(msg.channels) && msg.channels > 0
            ? msg.channels
            : DEFAULT_CHANNELS;

        if (msg.ringSab instanceof SharedArrayBuffer) {
          this.ring = new AudioRingBuffer(msg.ringSab, {
            channels: this.channels,
            volume: this.volume,
            muted: this.muted,
          });
          this.channels = this.ring.channels;
        } else {
          const cap =
            Number.isInteger(msg.capacityFrames) && msg.capacityFrames > 0
              ? msg.capacityFrames
              : DEFAULT_CAPACITY_FRAMES;
          this.ring = AudioRingBuffer.create(cap, this.channels);
          this.ring.setVolume(this.volume);
          this.ring.setMuted(this.muted);
        }

        this.initialized = true;
        this.totalPushedFrames = 0;
        this.totalPulledFrames = 0;

        this._postMessage({
          ready: true,
          sampleRate: this.sampleRate,
          channels: this.channels,
          capacityFrames: this.ring.capacityFrames,
        });
        return;
      }

      if (msg.pushPcm) {
        if (!this.ring) {
          this._postMessage({
            id: msg.id,
            error: 'AudioMixerWorker not initialized',
          });
          return;
        }
        const pcm = toFloat32Array(msg.samples);
        const inputFrames = Math.floor(pcm.length / this.channels);
        const pushedFrames = this.ring.pushSamples(pcm);
        const droppedFrames = inputFrames - pushedFrames;
        this.totalPushedFrames += pushedFrames;

        if (msg.id !== undefined) {
          this._postMessage({
            id: msg.id,
            pushedFrames,
            droppedFrames,
            availableFrames: this.ring.availableFrames(),
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
        if (msg.id !== undefined) {
          this._postMessage({
            id: msg.id,
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
        if (msg.id !== undefined) {
          this._postMessage({
            id: msg.id,
            volume: this.volume,
            muted: this.muted,
            effectiveGain: this.effectiveGain(),
          });
        }
        return;
      }

      if (msg.stats) {
        this._postMessage({
          id: msg.id,
          stats: this.getStats(),
        });
      }
    }
  }

  const exportsObj = {
    AudioRingBuffer,
    AudioMixerWorker,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    DEFAULT_CAPACITY_FRAMES,
    RING_HEADER_BYTES,
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
    const worker = new AudioMixerWorker((payload, transfer) => {
      if (transfer && transfer.length > 0) {
        self.postMessage(payload, transfer);
      } else {
        self.postMessage(payload);
      }
    });
    self.playGta6AudioWorker = worker;
    self.onmessage = (ev) => {
      worker.handleMessage(ev ? ev.data : null);
    };
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
