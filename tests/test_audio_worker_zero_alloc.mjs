import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  AudioRingBuffer,
  AudioMixerWorker,
  GuestPcmRingReader,
  PcmScratchPool,
  deinterleavePcm,
  interleavePcm,
  extractSafePcmSamples,
  applyFadeRamp,
  downmixToStereo,
  resampleLinear,
  buildAudioTelemetrySnapshot,
} = require('../web/workers/audio_worker.js');

/**
 * Exact reference raised-cosine ramp using Math.cos per frame.
 */
function referenceCosineFadeRamp(samples, channels, frameCount, startGain, endGain) {
  const out = samples instanceof Float32Array ? new Float32Array(samples) : samples.slice();
  if (frameCount <= 0) return out;
  if (frameCount === 1) {
    for (let c = 0; c < channels; c++) {
      const s = out[c];
      out[c] = endGain === 0.0 || !Number.isFinite(s) ? 0.0 : s * endGain;
    }
    return out;
  }
  const gDiff = endGain - startGain;
  const lastIdx = frameCount - 1;
  const invDenom = Math.PI / lastIdx;
  for (let f = 0; f < frameCount; f++) {
    const gain =
      f === 0
        ? startGain
        : f === lastIdx
          ? endGain
          : startGain + gDiff * (0.5 * (1.0 - Math.cos(f * invDenom)));
    const base = f * channels;
    for (let c = 0; c < channels; c++) {
      const s = out[base + c];
      out[base + c] = gain === 0.0 || !Number.isFinite(s) ? 0.0 : s * gain;
    }
  }
  return out;
}

function createMockGuestPcmRing({ capacitySamples = 8192, baseOffset = 64, sampleRate = 48000 } = {}) {
  const dataByteLength = capacitySamples * Float32Array.BYTES_PER_ELEMENT;
  const totalBytes = baseOffset + dataByteLength + 64;
  const sab = new SharedArrayBuffer(totalBytes);
  const readIndexByteOffset = 0;
  const writeIndexByteOffset = 4;
  const readCtrl = new Int32Array(sab, readIndexByteOffset, 1);
  const writeCtrl = new Int32Array(sab, writeIndexByteOffset, 1);
  const data = new Float32Array(sab, baseOffset, capacitySamples);
  const mask = capacitySamples - 1;

  return {
    sab,
    descriptor: {
      sab,
      dataByteOffset: baseOffset,
      readIndexByteOffset,
      writeIndexByteOffset,
      capacitySamples,
      channels: 2,
      sampleRate,
    },
    producerPush(interleaved) {
      let write = Atomics.load(writeCtrl, 0) >>> 0;
      for (let i = 0; i < interleaved.length; i++) {
        data[(write + i) & mask] = interleaved[i];
      }
      write = (write + interleaved.length) >>> 0;
      Atomics.store(writeCtrl, 0, write | 0);
      return interleaved.length;
    },
    readCursor() {
      return Atomics.load(readCtrl, 0) >>> 0;
    },
  };
}

// ---------------------------------------------------------------------------
// 1. Fast Cosine Ramp Accuracy & Trig Call Reduction
// ---------------------------------------------------------------------------

test('fast cosine applyFadeRamp matches Math.cos within < 1e-6 across short and long ramps with exact endpoints', () => {
  const lengths = [1, 2, 3, 4, 16, 128, 1024, 4096, 4097, 8192, 48000];
  const channelCounts = [1, 2, 6];
  const gainPairs = [
    [1.0, 0.0],
    [0.0, 1.0],
    [0.25, 0.9],
    [0.85, 0.15],
  ];

  for (const ch of channelCounts) {
    for (const len of lengths) {
      // Test in Float64 (plain array) for strict math accuracy without Float32 rounding masking drift,
      // and in Float32Array for real audio buffer fidelity.
      const [g0, g1] = gainPairs[(len + ch) % gainPairs.length];
      const f64Input = new Array(len * ch);
      const f32Input = new Float32Array(len * ch);
      for (let f = 0; f < len; f++) {
        for (let c = 0; c < ch; c++) {
          const val = Math.sin(0.031 * f + 0.7 * c) * 0.8 + 0.1 * (c % 2 === 0 ? 1 : -1);
          f64Input[f * ch + c] = val;
          f32Input[f * ch + c] = val;
        }
      }

      const refF64 = referenceCosineFadeRamp(f64Input, ch, len, g0, g1);
      const refF32 = referenceCosineFadeRamp(f32Input, ch, len, g0, g1);

      const origCos = Math.cos;
      let cosCalls = 0;
      Math.cos = (...args) => {
        cosCalls += 1;
        return origCos(...args);
      };
      try {
        applyFadeRamp(f64Input, ch, len, g0, g1);
      } finally {
        Math.cos = origCos;
      }

      // Verify trig call budget: 0 Math.cos calls for <= 4096 frames, <= floor(len / 4096) resyncs above.
      if (len <= 4096) {
        assert.equal(cosCalls, 0, `Expected 0 Math.cos calls for len=${len}, ch=${ch}, got ${cosCalls}`);
      } else {
        const maxAllowed = Math.ceil(len / 4096);
        assert.ok(
          cosCalls <= maxAllowed,
          `Expected <= ${maxAllowed} Math.cos resync calls for len=${len}, ch=${ch}, got ${cosCalls}`
        );
      }

      // Verify exact endpoints.
      if (len === 1) {
        for (let c = 0; c < ch; c++) {
          assert.equal(f64Input[c], refF64[c], `len=1 endpoint mismatch ch=${c}`);
        }
      } else {
        for (let c = 0; c < ch; c++) {
          assert.equal(f64Input[c], refF64[c], `start endpoint mismatch len=${len}, ch=${c}`);
          const endIdx = (len - 1) * ch + c;
          assert.equal(f64Input[endIdx], refF64[endIdx], `end endpoint mismatch len=${len}, ch=${c}`);
        }
      }

      // Verify Float64 max absolute error is far below 1e-6.
      let maxErrF64 = 0;
      for (let i = 0; i < f64Input.length; i++) {
        const err = Math.abs(f64Input[i] - refF64[i]);
        if (err > maxErrF64) maxErrF64 = err;
      }
      assert.ok(
        maxErrF64 < 1e-6,
        `Float64 maxAbsError=${maxErrF64} >= 1e-6 for len=${len}, ch=${ch}`
      );

      // Verify Float32Audio buffer parity is also < 1e-6.
      applyFadeRamp(f32Input, ch, len, g0, g1);
      let maxErrF32 = 0;
      for (let i = 0; i < f32Input.length; i++) {
        const err = Math.abs(f32Input[i] - refF32[i]);
        if (err > maxErrF32) maxErrF32 = err;
      }
      assert.ok(
        maxErrF32 < 1e-6,
        `Float32 maxAbsError=${maxErrF32} >= 1e-6 for len=${len}, ch=${ch}`
      );
    }
  }
});

test('fast cosine ramps in AudioRingBuffer, GuestPcmRingReader, and resampled direct drain match Math.cos < 1e-6 and update fastCosineRamps', () => {
  // A. AudioRingBuffer cosine underrun fade-out + recovery fade-in.
  const ring = AudioRingBuffer.create(256, 2, {
    declickFrames: 32,
    fadeCurve: 'cosine',
  });
  const steady = new Float32Array(64 * 2);
  steady.fill(0.75);
  ring.pushSamples(steady);

  const outFull = new Float32Array(64 * 2);
  assert.equal(ring.pullSamples(outFull), 64);

  const origCos = Math.cos;
  let cosCalls = 0;
  Math.cos = (...args) => {
    cosCalls += 1;
    return origCos(...args);
  };
  const outFadeOut = new Float32Array(64 * 2);
  const outFadeIn = new Float32Array(64 * 2);
  try {
    // Underrun fade-out over 32 frames.
    assert.equal(ring.pullSamples(outFadeOut), 0);
    // Recovery fade-in over 32 frames.
    ring.pushSamples(steady);
    assert.equal(ring.pullSamples(outFadeIn), 64);
  } finally {
    Math.cos = origCos;
  }

  assert.equal(cosCalls, 0, `Expected 0 Math.cos calls for 32-frame ring fades, got ${cosCalls}`);
  assert.equal(ring.fastCosineRamps, 2);
  assert.equal(ring.getStats().fastCosineRamps, 2);

  // Verify fade-out values against exact Math.cos reference.
  for (let i = 0; i < 32; i++) {
    const refGain = i === 31 ? 0.0 : 0.5 * (1.0 + Math.cos(Math.PI * ((i + 1) / 32)));
    const expected = Math.fround(0.75 * refGain);
    assert.ok(
      Math.abs(outFadeOut[i * 2] - expected) < 1e-6,
      `ring fade-out mismatch at frame ${i}: got ${outFadeOut[i * 2]}, expected ${expected}`
    );
  }
  assert.equal(outFadeOut[31 * 2], 0.0);
  assert.equal(outFadeOut[32 * 2], 0.0);

  // Verify fade-in values against exact Math.cos reference.
  for (let i = 0; i < 32; i++) {
    const refGain = 0.5 * (1.0 - Math.cos(Math.PI * (i / 32)));
    const expected = Math.fround(0.75 * refGain);
    assert.ok(
      Math.abs(outFadeIn[i * 2] - expected) < 1e-6,
      `ring fade-in mismatch at frame ${i}: got ${outFadeIn[i * 2]}, expected ${expected}`
    );
  }
  assert.equal(outFadeIn[0], 0.0);
  assert.equal(outFadeIn[32 * 2], 0.75);

  // B. GuestPcmRingReader backlog trim crossfade + underrun/recovery fade.
  const mock = createMockGuestPcmRing({ capacitySamples: 1024, sampleRate: 48000 });
  const reader = new GuestPcmRingReader(mock.descriptor);
  const blockA = new Float32Array(256 * 2).fill(0.4);
  const blockB = new Float32Array(256 * 2).fill(-0.6);
  mock.producerPush(blockA);
  const tmp = new Float32Array(64 * 2);
  reader.readInto(tmp, 64, { declickFrames: 16, fadeCurve: 'cosine' });
  mock.producerPush(blockB);
  // Trim backlog with 32-frame crossfade.
  const trimmed = reader.trimToFrames(128, 32);
  assert.ok(trimmed > 0);

  cosCalls = 0;
  Math.cos = (...args) => {
    cosCalls += 1;
    return origCos(...args);
  };
  const xfadeOut = new Float32Array(128 * 2);
  const guestFadeOut = new Float32Array(64 * 2);
  const guestFadeIn = new Float32Array(64 * 2);
  try {
    reader.readInto(xfadeOut, 128, { declickFrames: 16, fadeCurve: 'cosine' });
    // Now ring is empty -> trigger underrun fade-out.
    reader.readInto(guestFadeOut, 64, { declickFrames: 16, fadeCurve: 'cosine' });
    // Push audio and trigger recovery fade-in.
    mock.producerPush(new Float32Array(64 * 2).fill(0.5));
    reader.readInto(guestFadeIn, 64, { declickFrames: 16, fadeCurve: 'cosine' });
  } finally {
    Math.cos = origCos;
  }

  assert.equal(cosCalls, 0, `Expected 0 Math.cos calls in GuestPcmRingReader fast cosine ramps, got ${cosCalls}`);
  assert.equal(reader.fastCosineRamps, 3); // crossfade + fade-out + fade-in
  for (let f = 0; f < 32; f++) {
    const wNew = 0.5 * (1.0 - Math.cos(Math.PI * ((f + 1) / 33)));
    const expected = Math.fround(0.4 * (1.0 - wNew) + -0.6 * wNew);
    assert.ok(
      Math.abs(xfadeOut[f * 2] - expected) < 1e-6,
      `guest trim xfade mismatch at frame ${f}: got ${xfadeOut[f * 2]}, expected ${expected}`
    );
  }
  assert.equal(guestFadeOut[15 * 2], 0.0);
  assert.equal(guestFadeIn[0], 0.0);
  assert.equal(guestFadeIn[16 * 2], 0.5);
});

// ---------------------------------------------------------------------------
// 2. Steady-State Zero-Allocation Stress Tests (128 & 512 Sample Quanta)
// ---------------------------------------------------------------------------

test('steady-state AudioMixerWorker processes 1000+ alternating 128 and 512 frame quanta with zero scratch or subarray allocations', () => {
  const worker = new AudioMixerWorker(null, {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 16,
    fadeCurve: 'cosine',
  });
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 4096,
    outputState: 'active',
  });

  const sab128 = new SharedArrayBuffer(128 * 2 * Float32Array.BYTES_PER_ELEMENT);
  const view128 = new Float32Array(sab128);
  const sab512 = new SharedArrayBuffer(512 * 2 * Float32Array.BYTES_PER_ELEMENT);
  const view512 = new Float32Array(sab512);
  for (let i = 0; i < view128.length; i++) view128[i] = Math.sin(i * 0.05) * 0.5;
  for (let i = 0; i < view512.length; i++) view512[i] = Math.cos(i * 0.03) * 0.5;

  // Warm up both 512-frame (high-water capacity) and 128-frame view cache entries.
  worker.handleMessage({ pushPcm: true, samples: view512 });
  const warm512 = worker.pullPlanar(512);
  const warm512L = warm512.channels[0];
  const warm512R = warm512.channels[1];
  worker.handleMessage({ pushPcm: true, samples: view128 });
  const warm128 = worker.pullPlanar(128);
  const warm128L = warm128.channels[0];
  const warm128R = warm128.channels[1];

  const allocsAfterWarmup = worker.getScratchStats().allocations;
  const reusesBefore = worker.getStats().scratchBufferReuses;

  // Spy on Float32Array.prototype.subarray to guarantee zero view-wrapper churn during steady state.
  const origSubarray = Float32Array.prototype.subarray;
  let subarrayCalls = 0;
  Float32Array.prototype.subarray = function (...args) {
    subarrayCalls += 1;
    return origSubarray.apply(this, args);
  };

  try {
    for (let q = 0; q < 1000; q++) {
      const is128 = (q & 1) === 0;
      const frames = is128 ? 128 : 512;
      const src = is128 ? view128 : view512;
      worker.handleMessage({ pushPcm: true, samples: src });
      const pulled = worker.pullPlanar(frames);
      assert.equal(pulled.framesRead, frames);
      assert.equal(pulled.channels[0], is128 ? warm128L : warm512L);
      assert.equal(pulled.channels[1], is128 ? warm128R : warm512R);
      // Spot-check bit-exact sample fidelity.
      assert.equal(pulled.channels[0][0], src[0]);
      assert.equal(pulled.channels[1][frames - 1], src[frames * 2 - 1]);
    }
  } finally {
    Float32Array.prototype.subarray = origSubarray;
  }

  assert.equal(subarrayCalls, 0, `Expected 0 subarray() allocations in steady state, got ${subarrayCalls}`);
  const stats = worker.getStats();
  assert.equal(stats.scratchAllocations, allocsAfterWarmup);
  assert.ok(
    stats.scratchBufferReuses >= reusesBefore + 3000,
    `Expected >= ${reusesBefore + 3000} scratchBufferReuses, got ${stats.scratchBufferReuses}`
  );
  worker.destroy();
});

test('steady-state AudioRingBuffer.pullPlanarSamples and GuestPcmRingReader drain 1000+ 128/512 quanta with zero allocations', () => {
  // 1. AudioRingBuffer.pullPlanarSamples alternating 128 and 512 frames.
  const ring = AudioRingBuffer.create(2048, 2);
  const in512 = new Float32Array(512 * 2);
  const in128 = new Float32Array(128 * 2);
  for (let f = 0; f < 512; f++) {
    in512[f * 2] = (f + 1) * 0.001;
    in512[f * 2 + 1] = -(f + 1) * 0.001;
  }
  for (let f = 0; f < 128; f++) {
    in128[f * 2] = (f + 1) * 0.002;
    in128[f * 2 + 1] = -(f + 1) * 0.002;
  }
  const planarOut = [new Float32Array(512), new Float32Array(512)];

  // Warmup pullScratch to 512 frames.
  ring.pushSamples(in512);
  assert.equal(ring.pullPlanarSamples(planarOut, 512), 512);
  const allocsAfterWarmup = ring._pullScratchAllocations;

  const origSubarray = Float32Array.prototype.subarray;
  let subarrayCalls = 0;
  Float32Array.prototype.subarray = function (...args) {
    subarrayCalls += 1;
    return origSubarray.apply(this, args);
  };
  try {
    for (let q = 0; q < 1000; q++) {
      const is128 = (q & 1) === 0;
      const frames = is128 ? 128 : 512;
      const src = is128 ? in128 : in512;
      ring.pushSamples(src);
      assert.equal(ring.pullPlanarSamples(planarOut, frames), frames);
      assert.equal(planarOut[0][0], src[0]);
      assert.equal(planarOut[1][frames - 1], src[frames * 2 - 1]);
    }
  } finally {
    Float32Array.prototype.subarray = origSubarray;
  }

  assert.equal(subarrayCalls, 0);
  assert.equal(ring._pullScratchAllocations, allocsAfterWarmup);
  assert.equal(ring.getStats().scratchBufferReuses, 1000);

  // 2. GuestPcmRingReader + AudioMixerWorker resampled 44.1 kHz -> 48 kHz steady-state drain.
  const worker = new AudioMixerWorker(null, {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 16,
    fadeCurve: 'cosine',
  });
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 4096,
    outputState: 'active',
  });
  const mock = createMockGuestPcmRing({ capacitySamples: 16384, sampleRate: 44100 });
  worker.attachGuestPcmRing({ ...mock.descriptor, quantumFrames: 512 });

  const guestChunk = new Float32Array(512 * 2);
  for (let i = 0; i < guestChunk.length; i++) {
    guestChunk[i] = Math.sin(i * 0.02) * 0.4;
  }
  const directOut = new Float32Array(512 * 2);

  // First drain already pre-warmed both 'guestDrain' and 'resample' slots on attach!
  const allocsAfterAttach = worker.getScratchStats().allocations;
  for (let q = 0; q < 500; q++) {
    mock.producerPush(guestChunk);
    const frames = (q & 1) === 0 ? 128 : 512;
    const res = worker.drainGuestPcmRing(directOut, { maxFrames: frames });
    assert.equal(res.zeroAlloc, true, `Expected zeroAlloc=true on resampled quantum ${q}`);
    assert.equal(res.outputFrames, frames);
  }
  assert.equal(worker.getScratchStats().allocations, allocsAfterAttach);
  worker.destroy();
});

test('downmixToStereo, resampleLinear, and extractSafePcmSamples reuse scratch on SAB/non-Float32 inputs while preserving standalone ownership', () => {
  const pool = new PcmScratchPool({ maxSamples: 65536 });

  // 5.1 SAB-backed input into downmixToStereo with scratch pool.
  const sab51 = new SharedArrayBuffer(128 * 6 * Float32Array.BYTES_PER_ELEMENT);
  const view51 = new Float32Array(sab51);
  for (let f = 0; f < 128; f++) {
    view51[f * 6 + 0] = 0.25; // FL
    view51[f * 6 + 1] = -0.25; // FR
    view51[f * 6 + 2] = 0.2; // FC
    view51[f * 6 + 3] = 0.9; // LFE (0 weight by default)
    view51[f * 6 + 4] = 0.1; // BL
    view51[f * 6 + 5] = -0.1; // BR
  }

  // Standalone call (no scratch) returns independent owned buffers.
  const owned1 = downmixToStereo(view51, 6);
  const owned2 = downmixToStereo(view51, 6);
  assert.notEqual(owned1.buffer, owned2.buffer, 'Standalone downmixToStereo must return independent owned buffers');

  // Scratch-backed calls reuse 'extract' and 'downmix' with zero steady-state allocations.
  const dmWarm = downmixToStereo(view51, 6, { scratch: pool });
  assert.deepEqual(Array.from(dmWarm), Array.from(owned1));
  const allocsAfterDm = pool.allocations;
  for (let i = 0; i < 100; i++) {
    const out = downmixToStereo(view51, 6, { scratch: pool });
    assert.equal(out, dmWarm);
  }
  assert.equal(pool.allocations, allocsAfterDm);

  // Resampling SAB-backed input with scratch pool vs standalone ownership.
  const sabStereo = new SharedArrayBuffer(128 * 2 * Float32Array.BYTES_PER_ELEMENT);
  const viewStereo = new Float32Array(sabStereo);
  for (let i = 0; i < viewStereo.length; i++) viewStereo[i] = Math.sin(i * 0.04) * 0.5;

  const rsOwned1 = resampleLinear(viewStereo, 2, 44100, 48000);
  const rsOwned2 = resampleLinear(viewStereo, 2, 44100, 48000);
  assert.notEqual(rsOwned1.buffer, rsOwned2.buffer, 'Standalone resampleLinear must return independent owned buffers');

  const rsWarm = resampleLinear(viewStereo, 2, 44100, 48000, { scratch: pool });
  assert.deepEqual(Array.from(rsWarm), Array.from(rsOwned1));
  const allocsAfterRs = pool.allocations;
  for (let i = 0; i < 100; i++) {
    const out = resampleLinear(viewStereo, 2, 44100, 48000, { scratch: pool });
    assert.equal(out, rsWarm);
  }
  assert.equal(pool.allocations, allocsAfterRs);

  // Unaligned / non-Uint8Array SAB view (Int16Array) extraction reuses raw byte view & outResult.
  const sabInt16View = new Int16Array(sabStereo);
  const outRes = { ok: false, pcm: null, copiedFromSab: false, sanitizedSamples: 0, error: '' };
  const extOpts = { maxPcmBytes: 65536, scratch: pool, outResult: outRes };
  const r1 = extractSafePcmSamples(sabInt16View, extOpts);
  assert.equal(r1, outRes);
  assert.equal(r1.ok, true);
  const allocsAfterExt = pool.allocations;
  const cachedByteView = pool._lastRawByteView;
  for (let i = 0; i < 100; i++) {
    const r = extractSafePcmSamples(sabInt16View, extOpts);
    assert.equal(r, outRes);
    assert.equal(r.ok, true);
    assert.equal(pool._lastRawByteView, cachedByteView);
  }
  assert.equal(pool.allocations, allocsAfterExt);
});

// ---------------------------------------------------------------------------
// 3. Edge Cases: 0-Frame, 1-Frame, Dynamic Growth, Channel Switches, Lifecycle
// ---------------------------------------------------------------------------

test('edge cases: 0-frame, 1-frame, dynamic frame-size growth, mono/stereo/5.1 channel switches, and voice start/stop/underrun', () => {
  // 0-frame & 1-frame applyFadeRamp.
  const empty = new Float32Array(0);
  assert.equal(applyFadeRamp(empty, 2, 0, 1.0, 0.0), empty);
  const single = new Float32Array([0.6, -0.4]);
  applyFadeRamp(single, 2, 1, 1.0, 0.25);
  assert.ok(Math.abs(single[0] - 0.15) < 1e-6);
  assert.ok(Math.abs(single[1] - -0.1) < 1e-6);
  // NaN sanitization inside applyFadeRamp.
  const nanBuf = new Float32Array([Number.NaN, 0.5, Number.POSITIVE_INFINITY, -0.5, 0.8, -0.8]);
  applyFadeRamp(nanBuf, 2, 3, 1.0, 0.5);
  assert.equal(nanBuf[0], 0.0);
  assert.equal(nanBuf[1], 0.5);
  assert.equal(nanBuf[2], 0.0);
  assert.ok(Math.abs(nanBuf[3] - -0.375) < 1e-6);
  assert.ok(Math.abs(nanBuf[4] - 0.4) < 1e-6);
  assert.ok(Math.abs(nanBuf[5] - -0.4) < 1e-6);

  // Dynamic frame-size growth and channel-count switches (1ch -> 2ch -> 6ch -> 1ch -> 2ch -> 6ch) in PcmScratchPool.
  const pool = new PcmScratchPool({ maxSamples: 65536 });
  const p0 = pool.acquirePlanar(2, 0);
  assert.equal(p0.length, 2);
  assert.equal(p0[0].length, 0);

  const p1 = pool.acquirePlanar(1, 1);
  assert.equal(p1.length, 1);
  assert.equal(p1[0].length, 1);

  // Grow dynamically to 512 frames and 6 channels (5.1).
  const p512_6 = pool.acquirePlanar(6, 512);
  assert.equal(p512_6.length, 6);
  assert.equal(p512_6[5].length, 512);

  // Prime cache entries for mono(128), stereo(128), stereo(512), 5.1(128).
  const p128_1 = pool.acquirePlanar(1, 128);
  const p128_2 = pool.acquirePlanar(2, 128);
  const p512_2 = pool.acquirePlanar(2, 512);
  const p128_6 = pool.acquirePlanar(6, 128);
  const allocsAtHighWater = pool.allocations;

  // Switching back and forth across mono/stereo/5.1 and 128/512 frames allocates ZERO new backing buffers or views.
  const origSubarray = Float32Array.prototype.subarray;
  let subarrayCalls = 0;
  Float32Array.prototype.subarray = function (...args) {
    subarrayCalls += 1;
    return origSubarray.apply(this, args);
  };
  try {
    for (let i = 0; i < 100; i++) {
      assert.equal(pool.acquirePlanar(1, 128), p128_1);
      assert.equal(pool.acquirePlanar(2, 128), p128_2);
      assert.equal(pool.acquirePlanar(2, 512), p512_2);
      assert.equal(pool.acquirePlanar(6, 128), p128_6);
      assert.equal(pool.acquirePlanar(6, 512), p512_6);
    }
  } finally {
    Float32Array.prototype.subarray = origSubarray;
  }
  assert.equal(subarrayCalls, 0);
  assert.equal(pool.allocations, allocsAtHighWater);

  // Voice start -> downmix 5.1 -> stereo -> underrun -> recovery -> stop/reset telemetry check.
  const worker = new AudioMixerWorker(null, {
    outputState: 'active',
    autoDrain: false,
    declickFrames: 16,
    fadeCurve: 'cosine',
  });
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 1024,
    outputState: 'active',
  });

  // 0-frame pullPlanar edge case.
  const zeroPull = worker.pullPlanar(0);
  assert.equal(zeroPull.framesRead, 0);
  assert.equal(zeroPull.frameCount, 0);

  // Push mono, stereo, and 5.1 packets into the stereo worker ring.
  const monoPkt = new Float32Array(64).fill(0.3);
  const stereoPkt = new Float32Array(64 * 2).fill(0.4);
  const surround51Pkt = new Float32Array(64 * 6).fill(0.2);
  worker.handleMessage({ pushPcm: true, channels: 1, samples: monoPkt });
  worker.handleMessage({ pushPcm: true, channels: 2, samples: stereoPkt });
  worker.handleMessage({ pushPcm: true, channels: 6, samples: surround51Pkt });

  // Pull all 192 frames + trigger underrun cosine fade-out on next pull + recovery cosine fade-in.
  assert.equal(worker.pullPlanar(192).framesRead, 192);
  const underrunPull = worker.pullPlanar(64);
  assert.equal(underrunPull.framesRead, 0);
  assert.equal(underrunPull.channels[0][15], 0.0);

  worker.handleMessage({ pushPcm: true, channels: 2, samples: stereoPkt });
  const recoveryPull = worker.pullPlanar(64);
  assert.equal(recoveryPull.framesRead, 64);
  assert.equal(recoveryPull.channels[0][0], 0.0);
  assert.ok(Math.abs(recoveryPull.channels[0][16] - 0.4) < 1e-6);

  // Telemetry snapshot exposes sanitized scratchBufferReuses & fastCosineRamps and sanitizes dirty inputs.
  const liveSnap = buildAudioTelemetrySnapshot(worker, { timestamp: 1234 });
  assert.ok(liveSnap.stats.scratchBufferReuses > 0);
  assert.equal(liveSnap.stats.fastCosineRamps, 2);

  const dirtySnap = buildAudioTelemetrySnapshot(
    {
      scratchBufferReuses: -99,
      fastCosineRamps: Number.NaN,
      hostPath: 'C:\\forbidden\\dump.awc',
    },
    { timestamp: 5678 }
  );
  assert.equal(dirtySnap.stats.scratchBufferReuses, 0);
  assert.equal(dirtySnap.stats.fastCosineRamps, 0);
  assert.equal(JSON.stringify(dirtySnap).includes('dump.awc'), false);

  // Stop & reset zeroes counters cleanly.
  worker.stop();
  const afterStop = worker.getStats();
  assert.equal(afterStop.fastCosineRamps, 0);
  assert.equal(afterStop.scratchBufferReuses, 0);
  worker.destroy();
});
