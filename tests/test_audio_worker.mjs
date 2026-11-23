import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const audioMod = require('../web/workers/audio_worker.js');
const { AudioRingBuffer, AudioMixerWorker } = audioMod;

function makeStereoFrames(startIndex, frameCount) {
  const out = new Float32Array(frameCount * 2);
  for (let f = 0; f < frameCount; f++) {
    const idx = startIndex + f;
    out[f * 2] = (idx + 1) * 0.25; // Left channel
    out[f * 2 + 1] = -(idx + 1) * 0.125; // Right channel
  }
  return out;
}

test('UMD exports expose AudioRingBuffer and AudioMixerWorker on module.exports and globalThis', () => {
  assert.equal(typeof AudioRingBuffer, 'function');
  assert.equal(typeof AudioMixerWorker, 'function');
  assert.equal(globalThis.PlayGta6Audio.AudioRingBuffer, AudioRingBuffer);
  assert.equal(globalThis.PlayGta6Audio.AudioMixerWorker, AudioMixerWorker);
});

test('AudioRingBuffer stereo frame push/pull exactness across shared SAB views', () => {
  const producer = AudioRingBuffer.create(16, 2);
  const consumer = new AudioRingBuffer(producer.sab);

  assert.equal(consumer.capacityFrames, 16);
  assert.equal(consumer.channels, 2);
  assert.equal(producer.availableFrames(), 0);
  assert.equal(consumer.availableFrames(), 0);

  const input = makeStereoFrames(0, 6);
  const written = producer.pushSamples(input);
  assert.equal(written, 6);
  assert.equal(producer.availableFrames(), 6);
  assert.equal(consumer.availableFrames(), 6);
  assert.equal(producer.freeFrames(), 10);

  const output = new Float32Array(6 * 2);
  const read = consumer.pullSamples(output);
  assert.equal(read, 6);
  assert.deepEqual(Array.from(output), Array.from(input));
  assert.equal(consumer.availableFrames(), 0);
  assert.equal(consumer.underruns, 0);
  assert.equal(producer.overflows, 0);
});

test('AudioRingBuffer wrap-around preserves exact sample order across many cycles', () => {
  const ring = AudioRingBuffer.create(8, 2);
  let nextFrameId = 0;

  // Run enough push/pull chunks of size 5 to cross both the 8-frame buffer boundary
  // and the 16-frame double-modulo cursor wrap multiple times.
  for (let round = 0; round < 10; round++) {
    const chunk = makeStereoFrames(nextFrameId, 5);
    assert.equal(ring.pushSamples(chunk), 5);
    assert.equal(ring.availableFrames(), 5);

    const out = new Float32Array(5 * 2);
    assert.equal(ring.pullSamples(out), 5);
    assert.deepEqual(Array.from(out), Array.from(chunk));
    assert.equal(ring.availableFrames(), 0);

    nextFrameId += 5;
  }

  // Also test filling to exact capacity while cursors are mid-buffer.
  const seed = makeStereoFrames(100, 3);
  assert.equal(ring.pushSamples(seed), 3);
  const discard = new Float32Array(3 * 2);
  assert.equal(ring.pullSamples(discard), 3);

  const full = makeStereoFrames(200, 8);
  assert.equal(ring.pushSamples(full), 8);
  assert.equal(ring.availableFrames(), 8);
  assert.equal(ring.freeFrames(), 0);

  const outFull = new Float32Array(8 * 2);
  assert.equal(ring.pullSamples(outFull), 8);
  assert.deepEqual(Array.from(outFull), Array.from(full));
  assert.equal(ring.underruns, 0);
  assert.equal(ring.overflows, 0);
});

test('AudioRingBuffer overflow backpressure drops excess frames without clobbering unread audio', () => {
  const ring = AudioRingBuffer.create(6, 2);

  const first = makeStereoFrames(0, 4);
  assert.equal(ring.pushSamples(first), 4);
  assert.equal(ring.overflows, 0);
  assert.equal(ring.droppedFrames, 0);

  // Only 2 frames free; pushing 5 should write 2 and drop 3.
  const second = makeStereoFrames(4, 5);
  const written = ring.pushSamples(second);
  assert.equal(written, 2);
  assert.equal(ring.availableFrames(), 6);
  assert.equal(ring.overflows, 1);
  assert.equal(ring.droppedFrames, 3);

  // Pushing when completely full writes 0 and increments counters again.
  assert.equal(ring.pushSamples(makeStereoFrames(20, 2)), 0);
  assert.equal(ring.overflows, 2);
  assert.equal(ring.droppedFrames, 5);

  // Verify the 6 readable frames are strictly the original 4 + first 2 of `second`.
  const pulled = new Float32Array(6 * 2);
  assert.equal(ring.pullSamples(pulled), 6);
  const expected = new Float32Array(6 * 2);
  expected.set(first, 0);
  expected.set(second.subarray(0, 4), 8);
  assert.deepEqual(Array.from(pulled), Array.from(expected));
});

test('AudioRingBuffer underrun zero-fills tail and increments underrun counters', () => {
  const ring = AudioRingBuffer.create(8, 2);
  const partial = makeStereoFrames(0, 3);
  assert.equal(ring.pushSamples(partial), 3);

  // Pre-fill destination with sentinel garbage to ensure zero-fill overwrites it.
  const out = new Float32Array(5 * 2);
  out.fill(99.0);

  const framesRead = ring.pullSamples(out);
  assert.equal(framesRead, 3);
  assert.equal(ring.availableFrames(), 0);
  assert.equal(ring.underruns, 1);
  assert.equal(ring.underrunFrames, 2);

  assert.deepEqual(Array.from(out.subarray(0, 6)), Array.from(partial));
  assert.deepEqual(Array.from(out.subarray(6, 10)), [0, 0, 0, 0]);

  // Pulling from an empty buffer zero-fills everything and bumps underrun counters again.
  const emptyOut = new Float32Array(4 * 2);
  emptyOut.fill(-7.5);
  assert.equal(ring.pullSamples(emptyOut), 0);
  assert.deepEqual(Array.from(emptyOut), [0, 0, 0, 0, 0, 0, 0, 0]);
  assert.equal(ring.underruns, 2);
  assert.equal(ring.underrunFrames, 6);
});

test('AudioMixerWorker handles init, pushPcm, setVolume/mute, and stats messages', () => {
  const replies = [];
  const worker = new AudioMixerWorker((msg) => replies.push(msg));

  const ringSab = new SharedArrayBuffer(AudioRingBuffer.byteLength(16, 2));
  worker.handleMessage({
    init: true,
    sampleRate: 48000,
    channels: 2,
    ringSab,
  });

  assert.equal(replies.length, 1);
  assert.deepEqual(replies[0], {
    ready: true,
    sampleRate: 48000,
    channels: 2,
    capacityFrames: 16,
  });

  // Push PCM through worker message and read via shared ring view.
  const pcm = new Float32Array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.8, -0.8]);
  worker.handleMessage({ pushPcm: true, samples: pcm });

  // Set volume to 0.5 and verify scaled output.
  worker.handleMessage({ setVolume: true, volume: 0.5 });
  const outScaled = new Float32Array(2 * 2);
  assert.equal(worker.pullSamples(outScaled), 2);
  assert.deepEqual(Array.from(outScaled), [0.5, -0.5, 0.25, -0.25]);

  // Mute via setVolume message and verify silence while preserving volume.
  worker.handleMessage({ setVolume: true, muted: true });
  const outMuted = new Float32Array(2 * 2);
  outMuted.fill(1.0);
  assert.equal(worker.pullSamples(outMuted), 2);
  assert.deepEqual(Array.from(outMuted), [0, 0, 0, 0]);

  // Trigger an underrun via pullSamples.
  const outUnderrun = new Float32Array(3 * 2);
  assert.equal(worker.pullSamples(outUnderrun), 0);

  // Query stats with message id.
  worker.handleMessage({ stats: true, id: 77 });
  const statsReply = replies[replies.length - 1];
  assert.equal(statsReply.id, 77);
  assert.equal(statsReply.stats.initialized, true);
  assert.equal(statsReply.stats.sampleRate, 48000);
  assert.equal(statsReply.stats.channels, 2);
  assert.equal(statsReply.stats.capacityFrames, 16);
  assert.equal(statsReply.stats.availableFrames, 0);
  assert.equal(statsReply.stats.volume, 0.5);
  assert.equal(statsReply.stats.muted, true);
  assert.equal(statsReply.stats.effectiveGain, 0);
  assert.equal(statsReply.stats.underruns, 1);
  assert.equal(statsReply.stats.underrunFrames, 3);
  assert.equal(statsReply.stats.totalPushedFrames, 4);
  assert.equal(statsReply.stats.totalPulledFrames, 4);

  // Clamp volume bounds cleanly.
  worker.handleMessage({ setVolume: true, volume: 2.5, muted: false, id: 78 });
  assert.equal(replies[replies.length - 1].volume, 1.0);
  assert.equal(replies[replies.length - 1].muted, false);
  worker.handleMessage({ setVolume: true, volume: -0.4, id: 79 });
  assert.equal(replies[replies.length - 1].volume, 0.0);
});

test('AudioMixerWorker default init (loader.js compatibility) works without ringSab', () => {
  const replies = [];
  const worker = new AudioMixerWorker((msg) => replies.push(msg));
  worker.handleMessage({ init: true });

  assert.equal(replies.length, 1);
  assert.equal(replies[0].ready, true);
  assert.equal(replies[0].sampleRate, 48000);
  assert.equal(replies[0].channels, 2);
  assert.ok(replies[0].capacityFrames > 0);
});
