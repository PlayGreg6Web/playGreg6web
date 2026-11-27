import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  GpuRingBuffer,
  GPU_PACKET_HEADER_BYTES,
  DEFAULT_WRAP_SCRATCH_BYTES,
  HEADER_INDEX,
} = require('../web/workers/gpu_ring.js');

const { RangeCache } = require('../web/workers/range_cache.js');

function makePatternBytes(length, seed = 17) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = (i * 31 + seed) & 0xff;
  }
  return out;
}

function createMockRangeFetch(filesMap) {
  const calls = [];

  async function mockFetch(url, init = {}) {
    const rangeHeader = (init.headers && (init.headers.Range || init.headers.range)) || null;
    calls.push({ url, rangeHeader });

    const relPath = url.replace(/^.*\/data\//, '');
    const data = filesMap.get(relPath);
    if (!data) {
      return {
        ok: false,
        status: 404,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    if (!rangeHeader) {
      return {
        ok: true,
        status: 200,
        headers: new Map([['Content-Length', String(data.byteLength)]]),
        async arrayBuffer() {
          return data.slice().buffer;
        },
      };
    }

    const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
    assert.ok(match, `Unexpected Range header: ${rangeHeader}`);
    const reqStart = Number(match[1]);
    const reqEnd = Number(match[2]);

    if (reqStart >= data.byteLength) {
      return {
        ok: false,
        status: 416,
        headers: new Map([['Content-Range', `bytes */${data.byteLength}`]]),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const actualEnd = Math.min(reqEnd, data.byteLength - 1);
    const slice = data.slice(reqStart, actualEnd + 1);
    return {
      ok: true,
      status: 206,
      headers: new Map([
        ['Content-Range', `bytes ${reqStart}-${actualEnd}/${data.byteLength}`],
        ['Content-Length', String(slice.byteLength)],
      ]),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  }

  return { fetchImpl: mockFetch, calls };
}

function createMemoryOpfsDir() {
  const files = new Map();
  return {
    files,
    async getFileHandle(name, opts = {}) {
      if (!files.has(name)) {
        if (!opts.create) {
          const err = new Error(`NotFoundError: ${name}`);
          err.name = 'NotFoundError';
          throw err;
        }
        files.set(name, new Uint8Array(0));
      }
      return {
        async getFile() {
          const cur = files.get(name);
          return {
            size: cur.byteLength,
            async arrayBuffer() {
              return cur.slice().buffer;
            },
          };
        },
        async createWritable() {
          let pending = new Uint8Array(0);
          return {
            async write(chunk) {
              pending = chunk instanceof Uint8Array ? chunk.slice() : new Uint8Array(chunk);
            },
            async close() {
              files.set(name, pending);
            },
            async abort() {},
          };
        },
      };
    },
    async removeEntry(name) {
      files.delete(name);
    },
    async *keys() {
      for (const k of files.keys()) {
        yield k;
      }
    },
  };
}

// =============================================================================
// GpuRingBuffer hyper-optimization & stress tests (#2180)
// =============================================================================

test('GpuRingBuffer caches typed-array & DataView views on the instance across calls', () => {
  const producer = GpuRingBuffer.create(1024);
  const consumer = GpuRingBuffer.attach(producer.sab);

  const hI32 = consumer.headerI32;
  const hU32 = consumer.headerU32;
  const data = consumer.data;
  const dv = consumer.dataView;
  const wrapDv = consumer.wrapScratchView;

  assert.ok(hI32 instanceof Int32Array);
  assert.ok(hU32 instanceof Uint32Array);
  assert.ok(data instanceof Uint8Array);
  assert.ok(dv instanceof DataView);
  assert.ok(wrapDv instanceof DataView);
  assert.equal(hI32.buffer, producer.sab);
  assert.equal(hU32.buffer, producer.sab);
  assert.equal(data.buffer, producer.sab);
  assert.equal(dv.buffer, producer.sab);

  for (let i = 0; i < 64; i += 1) {
    assert.equal(consumer.headerI32, hI32);
    assert.equal(consumer.headerU32, hU32);
    assert.equal(consumer.data, data);
    assert.equal(consumer.dataView, dv);
    assert.equal(consumer.wrapScratchView, wrapDv);
  }
});

test('GpuRingBuffer 25,000-packet stress: writePacket + drainInto zero-copy, wrap scratch reuse, and single atomic publish', () => {
  const TOTAL_PACKETS = 25000;
  const RING_CAP = 4096;
  const producer = GpuRingBuffer.create(RING_CAP);
  const consumer = GpuRingBuffer.attach(producer.sab);

  const initialScratchBuffer = consumer._wrapScratch.buffer;
  assert.equal(consumer._wrapScratch.byteLength, RING_CAP);

  // Reusable producer payload buffers of varying sizes to hit every wrap alignment.
  const payloadSizes = [0, 4, 12, 24, 60, 128, 256, 508];
  const payloadPool = payloadSizes.map((sz) => new Uint8Array(sz));
  const payloadViews = payloadPool.map((buf) => (buf.byteLength >= 4 ? new DataView(buf.buffer) : null));

  const origStore = Atomics.store;
  const origNotify = Atomics.notify;
  let readPosStores = 0;
  let readPosNotifies = 0;
  Atomics.store = (ta, idx, val) => {
    if (idx === HEADER_INDEX.READ_POS) readPosStores += 1;
    return origStore(ta, idx, val);
  };
  Atomics.notify = (ta, idx, count) => {
    if (idx === HEADER_INDEX.READ_POS) readPosNotifies += 1;
    return origNotify(ta, idx, count);
  };

  let produced = 0;
  let consumed = 0;
  let contiguousZeroCopyCount = 0;
  let wrappedScratchCount = 0;
  let firstMetaRef = null;
  let firstResRef = null;

  try {
    while (consumed < TOTAL_PACKETS) {
      while (produced < TOTAL_PACKETS) {
        const sizeIdx = produced % payloadSizes.length;
        const sz = payloadSizes[sizeIdx];
        const buf = payloadPool[sizeIdx];
        if (sz >= 4) {
          payloadViews[sizeIdx].setUint32(0, produced, true);
        }
        for (let b = 4; b < sz; b += 1) {
          buf[b] = (produced * 13 + b * 7) & 0xff;
        }
        const opcode = (produced % 63) + 1;
        const flags = (produced * 37) & 0xffff;
        if (!producer.writePacket(opcode, flags, sz > 0 ? buf : null, false)) {
          break;
        }
        produced += 1;
      }
      producer.notifyConsumer();

      const res = consumer.drainInto(
        (packetView, meta, idx) => {
          if (firstMetaRef === null) {
            firstMetaRef = meta;
          } else {
            assert.equal(meta, firstMetaRef, 'drainInto must reuse packet meta descriptor');
          }

          const expectedSize = payloadSizes[consumed % payloadSizes.length];
          const expectedOpcode = (consumed % 63) + 1;
          const expectedFlags = (consumed * 37) & 0xffff;

          assert.equal(meta.opcode, expectedOpcode);
          assert.equal(meta.flags, expectedFlags);
          assert.equal(meta.payloadBytes, expectedSize);
          assert.equal(meta.byteLength, GPU_PACKET_HEADER_BYTES + expectedSize);
          assert.equal(packetView.byteLength, GPU_PACKET_HEADER_BYTES + expectedSize);
          assert.equal(meta.byteOffset, 0);
          assert.equal(meta.payloadOffset, GPU_PACKET_HEADER_BYTES);

          if (!meta.wrapped) {
            contiguousZeroCopyCount += 1;
            assert.equal(meta.zeroCopy, true);
            assert.equal(packetView.buffer, producer.sab, 'non-wrapping packet must be zero-copy SAB subarray');
            assert.equal(meta.dataView, consumer.dataView);
          } else {
            wrappedScratchCount += 1;
            assert.equal(meta.zeroCopy, false);
            assert.equal(
              packetView.buffer,
              initialScratchBuffer,
              'wrapping packet must reuse pre-allocated scratch buffer without reallocation'
            );
            assert.equal(meta.dataView, consumer.wrapScratchView);
            assert.equal(meta.viewOffset, 0);
          }

          // Verify header bytes via cached DataView + exact payload integrity.
          assert.equal(meta.dataView.getUint16(meta.viewOffset, true), expectedOpcode);
          assert.equal(meta.dataView.getUint16(meta.viewOffset + 2, true), expectedFlags);
          assert.equal(meta.dataView.getUint32(meta.viewOffset + 4, true), expectedSize);
          if (expectedSize >= 4) {
            assert.equal(meta.dataView.getUint32(meta.viewOffset + 8, true), consumed);
          }
          for (let b = 4; b < expectedSize; b += 1) {
            const actual = packetView[GPU_PACKET_HEADER_BYTES + b];
            const expected = (consumed * 13 + b * 7) & 0xff;
            if (actual !== expected) {
              assert.equal(actual, expected, `payload mismatch at packet ${consumed} byte ${b}`);
            }
          }
          assert.ok(idx >= 0);
          consumed += 1;
        },
        { reuseResult: true }
      );

      if (firstResRef === null) {
        firstResRef = res;
      } else {
        assert.equal(res, firstResRef, 'reuseResult: true must reuse the drain result object');
      }
      assert.ok(res.packets > 0, 'drainInto must make forward progress on non-empty ring');
      assert.equal(res.remainingBytes, 0, 'drainInto without cap must drain all complete packets');
    }
  } finally {
    Atomics.store = origStore;
    Atomics.notify = origNotify;
  }

  assert.equal(produced, TOTAL_PACKETS);
  assert.equal(consumed, TOTAL_PACKETS);
  assert.equal(consumer.availableBytes(), 0);
  assert.equal(producer.freeBytes(), RING_CAP);
  assert.ok(contiguousZeroCopyCount > 20000, `expected >20k zero-copy packets, got ${contiguousZeroCopyCount}`);
  assert.ok(wrappedScratchCount > 200, `expected >200 wrap-around packets, got ${wrappedScratchCount}`);

  const consStats = consumer.getStats();
  const prodStats = producer.getStats();
  assert.equal(consStats.packetsDrained, TOTAL_PACKETS);
  assert.equal(readPosStores, consStats.batchesDrained, 'exactly 1 Atomics.store(READ_POS) per batch');
  assert.equal(readPosNotifies, consStats.batchesDrained, 'exactly 1 Atomics.notify(READ_POS) per batch');
  assert.equal(consStats.zeroAllocDrains, consStats.batchesDrained);
  assert.ok(prodStats.backpressureWaits > 0, 'bounded 4 KiB stress must exercise backpressure');
  assert.equal(consumer._wrapScratch.buffer, initialScratchBuffer);
});

test('GpuRingBuffer drainInto & drainBatch options: drainAll, directView, perPacketCommit, early stop, throw safety, and u32 wrap', () => {
  const producer = GpuRingBuffer.create(8192);
  const consumer = GpuRingBuffer.attach(producer.sab);

  // 1) drainBatch(handler, { drainAll: true, reuseResult: true }) drains >256 packets in 1 atomic publish.
  const payload4 = new Uint8Array(4);
  const payload4Dv = new DataView(payload4.buffer);
  for (let i = 0; i < 400; i += 1) {
    payload4Dv.setUint32(0, i, true);
    assert.equal(producer.writePacket(10, i & 0xff, payload4, false), true);
  }
  producer.notifyConsumer();

  const origStore = Atomics.store;
  let stores = 0;
  Atomics.store = (ta, idx, val) => {
    if (idx === HEADER_INDEX.READ_POS) stores += 1;
    return origStore(ta, idx, val);
  };

  let firstPktDesc = null;
  let decoded = 0;
  try {
    const res = consumer.drainBatch(
      (pkt) => {
        if (firstPktDesc === null) {
          firstPktDesc = pkt;
        } else {
          assert.equal(pkt, firstPktDesc, 'drainBatch must reuse packet descriptor object');
        }
        assert.equal(pkt.dataView.getUint32(pkt.viewOffset + 8, true), decoded);
        assert.equal(pkt.bytes.buffer, producer.sab);
        decoded += 1;
      },
      { drainAll: true, reuseResult: true }
    );
    assert.equal(res.packets, 400);
    assert.equal(res.bytes, 400 * 12);
    assert.equal(res.remainingBytes, 0);
    assert.equal(stores, 1, 'drainAll must commit READ_POS once for all 400 packets');

    // 2) perPacketCommit: true commits READ_POS after each packet for strict backpressure safety.
    stores = 0;
    for (let i = 0; i < 5; i += 1) {
      payload4Dv.setUint32(0, 1000 + i, true);
      assert.equal(producer.writePacket(11, 0, payload4, false), true);
    }
    const perPktSeen = [];
    const rPer = consumer.drainInto(
      (view, meta) => {
        // Producer observes freed space after previous packets even mid-batch!
        perPktSeen.push(meta.dataView.getUint32(meta.viewOffset + 8, true));
        assert.equal(view, consumer.data, 'directView: true must pass cached ring Uint8Array directly');
      },
      { directView: true, perPacketCommit: true }
    );
    assert.equal(rPer.packets, 5);
    assert.deepEqual(perPktSeen, [1000, 1001, 1002, 1003, 1004]);
    assert.equal(stores, 5, 'perPacketCommit: true must commit READ_POS once per packet');
  } finally {
    Atomics.store = origStore;
  }

  // 3) Mid-batch throw in drainInto commits only prior completed packets and preserves remaining stream.
  for (let i = 0; i < 6; i += 1) {
    payload4Dv.setUint32(0, 2000 + i, true);
    assert.equal(producer.writePacket(12, i, payload4, false), true);
  }
  const beforeThrow = [];
  assert.throws(() => {
    consumer.drainInto((view, meta, idx) => {
      if (idx === 3) {
        throw new Error('drainInto-mid-boom');
      }
      beforeThrow.push(meta.dataView.getUint32(meta.viewOffset + 8, true));
    });
  }, /drainInto-mid-boom/);
  assert.deepEqual(beforeThrow, [2000, 2001, 2002]);
  assert.equal(consumer.availableBytes(), 3 * 12);

  // Early stop via returning false commits current packet cleanly.
  const afterThrow = [];
  const rEarly = consumer.drainInto((view, meta, idx) => {
    afterThrow.push(meta.dataView.getUint32(meta.viewOffset + 8, true));
    if (idx === 0) return false;
  });
  assert.equal(rEarly.packets, 1);
  assert.equal(rEarly.remainingBytes, 24);
  const rTail = consumer.drainInto((view, meta) => {
    afterThrow.push(meta.dataView.getUint32(meta.viewOffset + 8, true));
  });
  assert.equal(rTail.packets, 2);
  assert.equal(rTail.remainingBytes, 0);
  assert.deepEqual(afterThrow, [2003, 2004, 2005]);

  // 4) 32-bit counter wrap across 0xFFFFFFFF -> 0 with split header + split payload.
  const smallProd = GpuRingBuffer.create(64);
  const smallCons = GpuRingBuffer.attach(smallProd.sab);
  Atomics.store(smallProd.headerU32, HEADER_INDEX.WRITE_POS, 0xfffffffc); // offset 60 -> header splits [60..64) + [0..4)
  Atomics.store(smallProd.headerU32, HEADER_INDEX.READ_POS, 0xfffffffc);
  const pWrap = makePatternBytes(12, 77);
  assert.equal(smallProd.writePacket(42, 0xbeef, pWrap), true);
  assert.equal(smallProd.writePos, 0x00000010);

  let gotBytes = null;
  const rWrap = smallCons.drainInto((view, meta) => {
    assert.equal(meta.opcode, 42);
    assert.equal(meta.flags, 0xbeef);
    assert.equal(meta.payloadBytes, 12);
    assert.equal(meta.wrapped, true);
    assert.equal(meta.zeroCopy, false);
    gotBytes = view.slice(meta.payloadOffset, meta.byteLength);
  });
  assert.equal(rWrap.packets, 1);
  assert.equal(rWrap.wrapped, true);
  assert.equal(smallCons.readPos, 0x00000010);
  assert.deepEqual(gotBytes, pWrap);
});

test('GpuRingBuffer grows wrap scratch once for oversized wrapped packets and reuses it thereafter', () => {
  const RING_CAP = 256 * 1024;
  const producer = GpuRingBuffer.create(RING_CAP);
  const consumer = GpuRingBuffer.attach(producer.sab);

  assert.equal(consumer._wrapScratch.byteLength, DEFAULT_WRAP_SCRATCH_BYTES);

  // Move offset to 200 KiB so a 96 KiB packet wraps across the 256 KiB boundary.
  const pad = new Uint8Array(200 * 1024);
  assert.equal(producer.writeBytes(pad, false), true);
  assert.equal(consumer.drainTo(() => {}), pad.byteLength);

  const bigPayload = makePatternBytes(96 * 1024 - GPU_PACKET_HEADER_BYTES, 91);
  let grownScratchBuffer = null;

  for (let round = 0; round < 12; round += 1) {
    bigPayload[0] = round & 0xff;
    bigPayload[bigPayload.byteLength - 1] = (round * 17) & 0xff;
    assert.equal(producer.writePacket(7, round, bigPayload, false), true);

    const res = consumer.drainInto((view, meta) => {
      assert.equal(meta.opcode, 7);
      assert.equal(meta.flags, round);
      assert.equal(meta.payloadBytes, bigPayload.byteLength);
      if (meta.wrapped) {
        if (grownScratchBuffer === null) {
          grownScratchBuffer = view.buffer;
          assert.ok(grownScratchBuffer.byteLength >= 96 * 1024);
        } else {
          assert.equal(
            view.buffer,
            grownScratchBuffer,
            'wrap scratch must not reallocate once grown to fit max wrapped packet'
          );
        }
      }
      assert.equal(view[meta.payloadOffset], round & 0xff);
      assert.equal(view[meta.byteLength - 1], (round * 17) & 0xff);
    });
    assert.equal(res.packets, 1);
  }
  assert.ok(grownScratchBuffer !== null, 'test must exercise grown wrap scratch');
});

// =============================================================================
// RangeCache O(1) LRU fast paths, zero-copy slice, readInto & stress tests (#2180)
// =============================================================================

test('RangeCache O(1) LRU MRU fast path skips Map.delete/Map.set across 12,000+ lookups (LRU & SLRU)', async () => {
  const KiB = 1024;
  const chunkSize = 64 * KiB;
  const fileA = makePatternBytes(4 * chunkSize, 19);
  const fileB = makePatternBytes(4 * chunkSize, 53);
  const mock = createMockRangeFetch(
    new Map([
      ['x64/hot_a.rpf', fileA],
      ['x64/hot_b.rpf', fileB],
    ])
  );

  // Test SLRU-enabled cache (maxBytes / chunkSize = 16 >= SLRU_MIN_CHUNKS).
  const cache = new RangeCache({
    chunkSize,
    maxBytes: 16 * chunkSize,
    enableReadAhead: false,
    fetchImpl: mock.fetchImpl,
  });

  // Warm chunk 0 of fileA (now at MRU tail).
  const warmBuf = await cache.read('x64/hot_a.rpf', 0, chunkSize);
  assert.deepEqual(warmBuf, fileA.subarray(0, chunkSize));

  // Instrument _memCache.delete & _memCache.set to prove zero redundant Map mutations on MRU tail hits!
  const origDelete = cache._memCache.delete.bind(cache._memCache);
  const origSet = cache._memCache.set.bind(cache._memCache);
  let memDeletes = 0;
  let memSets = 0;
  cache._memCache.delete = (k) => {
    memDeletes += 1;
    return origDelete(k);
  };
  cache._memCache.set = (k, v) => {
    memSets += 1;
    return origSet(k, v);
  };

  const dst = new Uint8Array(4096);
  try {
    // 12,000 lookups hitting the exact same MRU chunk via readCachedRangeInto, getCachedSlice, and readInto!
    for (let i = 0; i < 4000; i += 1) {
      const off = (i * 97) % (chunkSize - 4096);
      const len = 32 + (i % 1024);
      const n = cache.readCachedRangeInto('x64/hot_a.rpf', off, len, dst);
      assert.equal(n, len);
      assert.equal(dst[0], fileA[off]);
      assert.equal(dst[len - 1], fileA[off + len - 1]);
    }

    for (let i = 0; i < 4000; i += 1) {
      const off = (i * 131) % (chunkSize - 1024);
      const len = 64 + (i % 512);
      const slice = cache.getCachedSlice('x64/hot_a.rpf', off, len);
      assert.ok(slice instanceof Uint8Array);
      assert.equal(slice.byteLength, len);
      assert.equal(slice.buffer, cache._memCache.get('x64/hot_a.rpf::0').buffer, 'getCachedSlice must be zero-copy');
      assert.equal(slice[0], fileA[off]);
      assert.equal(slice[len - 1], fileA[off + len - 1]);
    }

    for (let i = 0; i < 4000; i += 1) {
      const off = (i * 53) % (chunkSize - 256);
      const subDst = dst.subarray(0, 128);
      const n = await cache.readInto('x64/hot_a.rpf', off, subDst);
      assert.equal(n, 128);
      assert.equal(subDst[0], fileA[off]);
      assert.equal(subDst[127], fileA[off + 127]);
    }

    assert.equal(memDeletes, 0, 'MRU tail hits must perform 0 Map.delete calls');
    assert.equal(memSets, 0, 'MRU tail hits must perform 0 Map.set calls');
    assert.equal(cache.getStats().mruFastPathHits, 12000);
    assert.equal(mock.calls.length, 1, 'all 12,000 hits must issue 0 network fetches');

    // Now warm chunk 0 of fileB and alternate back to fileA -> exactly 1 promotion, then 0 mutations on repeats.
    await cache.read('x64/hot_b.rpf', 0, 1024);
    memDeletes = 0;
    memSets = 0;

    // First hit back on fileA promotes it once.
    assert.equal(cache.readCachedRangeInto('x64/hot_a.rpf', 0, 64, dst), 64);
    assert.equal(memDeletes, 1);
    assert.equal(memSets, 1);

    // Next 500 hits on fileA are MRU tail hits again -> 0 additional deletes/sets!
    for (let i = 0; i < 500; i += 1) {
      assert.equal(cache.readCachedRangeInto('x64/hot_a.rpf', i * 16, 64, dst), 64);
    }
    assert.equal(memDeletes, 1);
    assert.equal(memSets, 1);
  } finally {
    cache._memCache.delete = origDelete;
    cache._memCache.set = origSet;
  }
});

test('RangeCache single-chunk zero-copy fast path & multi-chunk readInto direct destination copy', async () => {
  const KiB = 1024;
  const chunkSize = 16 * KiB;
  const totalBytes = 4 * chunkSize + 5000; // 4 full chunks + 5000B tail
  const fileBytes = makePatternBytes(totalBytes, 83);
  const mock = createMockRangeFetch(new Map([['game/streaming.rpf', fileBytes]]));
  const opfsDir = createMemoryOpfsDir();

  const cache = new RangeCache({
    chunkSize,
    maxBytes: 16 * chunkSize,
    enableReadAhead: false,
    coalesceAdjacent: true,
    fetchImpl: mock.fetchImpl,
    opfsDir,
  });

  // 1) Cold readInto(file, offset, dst) spanning 3 chunks: fetches & copies directly into caller dst!
  const multiLen = 2 * chunkSize + 3333;
  const multiStart = 4096;
  const multiDst = new Uint8Array(multiLen);
  const nCold = await cache.readInto('game/streaming.rpf', multiStart, multiDst);
  assert.equal(nCold, multiLen);
  assert.deepEqual(multiDst, fileBytes.subarray(multiStart, multiStart + multiLen));
  assert.equal(mock.calls.length, 1, 'coalesced 3-chunk readInto must issue 1 HTTP 206');

  // 2) Warm L1 multi-chunk readInto(url, offset, length, dst, dstOffset) with non-zero dstOffset & guard bytes.
  const callsAfterWarm = mock.calls.length;
  const padded = new Uint8Array(multiLen + 64).fill(0x5a);
  const nWarmMulti = await cache.readInto(
    '/data/game/streaming.rpf',
    multiStart,
    multiLen,
    padded,
    32
  );
  assert.equal(nWarmMulti, multiLen);
  assert.equal(padded[31], 0x5a);
  assert.equal(padded[32 + multiLen], 0x5a);
  assert.deepEqual(padded.subarray(32, 32 + multiLen), fileBytes.subarray(multiStart, multiStart + multiLen));
  assert.equal(mock.calls.length, callsAfterWarm);

  // 3) Single-chunk zero-copy getCachedSlice & read(..., { zeroCopy: true }) vs default owned read().
  const cachedChunk0 = cache._memCache.get('game/streaming.rpf::0');
  const zView = cache.getCachedSlice('game/streaming.rpf', 100, 500);
  assert.equal(zView.buffer, cachedChunk0.buffer, 'getCachedSlice must share backing buffer (zero-copy)');
  assert.deepEqual(zView, fileBytes.subarray(100, 600));

  const zRead = await cache.read('game/streaming.rpf', 200, 300, { zeroCopy: true });
  assert.equal(zRead.buffer, cachedChunk0.buffer, 'read({ zeroCopy: true }) must share backing buffer');
  assert.deepEqual(zRead, fileBytes.subarray(200, 500));

  const ownedRead = await cache.read('game/streaming.rpf', 200, 300);
  assert.notEqual(ownedRead.buffer, cachedChunk0.buffer, 'default read() must return an owned buffer');
  assert.deepEqual(ownedRead, fileBytes.subarray(200, 500));

  // Multi-chunk range returns null from getCachedSlice without failing.
  assert.equal(cache.getCachedSlice('game/streaming.rpf', chunkSize - 10, 20), null);

  // 4) EOF clamping, past-EOF, zero-length, and undersized dst validation in readInto.
  const tailDst = new Uint8Array(chunkSize);
  const nTail = await cache.readInto('game/streaming.rpf', 4 * chunkSize + 1000, tailDst);
  assert.equal(nTail, 4000);
  assert.deepEqual(tailDst.subarray(0, 4000), fileBytes.subarray(4 * chunkSize + 1000, totalBytes));

  assert.equal(await cache.readInto('game/streaming.rpf', totalBytes, tailDst), 0);
  assert.equal(await cache.readInto('game/streaming.rpf', 0, new Uint8Array(0)), 0);
  await assert.rejects(
    () => cache.readInto('game/streaming.rpf', 0, 1024, new Uint8Array(128)),
    RangeError
  );
  await assert.rejects(() => cache.readInto('game/streaming.rpf', 0, new ArrayBuffer(64)), TypeError);

  // 5) OPFS L2 warm reload via readInto with 0 network fetches!
  await cache.flushOpfsWrites();
  const warmMock = createMockRangeFetch(new Map([['game/streaming.rpf', fileBytes]]));
  const warmCache = new RangeCache({
    chunkSize,
    maxBytes: 16 * chunkSize,
    enableReadAhead: false,
    fetchImpl: warmMock.fetchImpl,
    opfsDir,
  });
  const opfsDst = new Uint8Array(multiLen);
  const nOpfs = await warmCache.readInto('game/streaming.rpf', multiStart, opfsDst);
  assert.equal(nOpfs, multiLen);
  assert.deepEqual(opfsDst, fileBytes.subarray(multiStart, multiStart + multiLen));
  assert.equal(warmMock.calls.length, 0, 'warm OPFS L2 readInto must issue 0 network fetches');
  assert.ok(warmCache.getStats().opfsHits >= 3);
});

test('RangeCache bounded eviction under tight low-RAM budget preserves exact byte integrity and O(1) index bounds', async () => {
  const KiB = 1024;
  const chunkSize = 8 * KiB;
  const MAX_CHUNKS = 8; // Tight 64 KiB budget (>= SLRU_MIN_CHUNKS -> exercises 2-segment SLRU!)
  const maxBytes = MAX_CHUNKS * chunkSize;

  const hotFile = makePatternBytes(4 * chunkSize, 11);
  const streamFile = makePatternBytes(64 * chunkSize, 29);
  const randomFile = makePatternBytes(32 * chunkSize, 67);
  const mock = createMockRangeFetch(
    new Map([
      ['game/hot_toc.rpf', hotFile],
      ['movies/stream.bk2', streamFile],
      ['x64/world_lod.rpf', randomFile],
    ])
  );

  const cache = new RangeCache({
    chunkSize,
    maxBytes,
    minReadaheadBytes: chunkSize,
    maxReadaheadBytes: 2 * chunkSize,
    enableReadAhead: true,
    coalesceAdjacent: true,
    fetchImpl: mock.fetchImpl,
  });

  // Promote 4 hot chunks to SLRU Protected segment.
  const tmp = new Uint8Array(chunkSize);
  for (let pass = 0; pass < 2; pass += 1) {
    for (let c = 0; c < 4; c += 1) {
      const n = await cache.readInto('game/hot_toc.rpf', c * chunkSize, tmp);
      assert.equal(n, chunkSize);
      assert.deepEqual(tmp, hotFile.subarray(c * chunkSize, (c + 1) * chunkSize));
    }
  }
  assert.equal(cache._protectedChunks.size, 4);

  // Stream sequentially through 64 chunks (8x cache capacity!) with adaptive readahead active.
  for (let c = 0; c < 64; c += 1) {
    const off = c * chunkSize + (c % 512);
    const len = Math.min(4096, streamFile.byteLength - off);
    const view = tmp.subarray(0, len);
    const n = await cache.readInto('movies/stream.bk2', off, view);
    assert.equal(n, len);
    assert.deepEqual(view, streamFile.subarray(off, off + len));
    await cache.flushPrefetches();

    const stats = cache.getStats();
    assert.ok(stats.bytesCached <= maxBytes, `bytesCached ${stats.bytesCached} exceeded maxBytes ${maxBytes}`);
    assert.ok(stats.entries <= MAX_CHUNKS, `entries ${stats.entries} exceeded ${MAX_CHUNKS}`);
  }

  // Hot protected chunks survived the entire 8x sequential scan with 0 network refetches!
  const callsBeforeHotVerify = mock.calls.length;
  for (let c = 0; c < 4; c += 1) {
    const slice = cache.getCachedSlice('game/hot_toc.rpf', c * chunkSize + 128, 256);
    assert.ok(slice !== null, `protected hot chunk ${c} must survive sequential scan`);
    assert.deepEqual(slice, hotFile.subarray(c * chunkSize + 128, c * chunkSize + 384));
  }
  assert.equal(mock.calls.length, callsBeforeHotVerify);

  // Random seeks across randomFile + tight trimToBytes reclaim under memory pressure.
  let seed = 1337;
  const nextRand = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed;
  };
  for (let i = 0; i < 60; i += 1) {
    const off = nextRand() % (randomFile.byteLength - 2048);
    const len = 1 + (nextRand() % 2048);
    const sub = tmp.subarray(0, len);
    const n = await cache.readInto('x64/world_lod.rpf', off, sub);
    assert.equal(n, len);
    assert.deepEqual(sub, randomFile.subarray(off, off + len));
    assert.ok(cache.getStats().bytesCached <= maxBytes);
  }

  // Reclaim to 50% of budget: probation (including prefetched) evicted first, protected preserved.
  const reclaimed = cache.trimToBytes(4 * chunkSize);
  assert.ok(reclaimed.remainingBytes <= 4 * chunkSize);
  assert.equal(cache._protectedChunks.size, 4, 'protected chunks must be preserved during trimToBytes');

  // Verify internal secondary indices stay strictly bounded & synchronized with _memCache!
  let totalIndexedKeys = 0;
  for (const map of cache._pathChunkKeys.values()) {
    totalIndexedKeys += map.size;
  }
  assert.equal(
    totalIndexedKeys,
    cache._memCache.size,
    '_pathChunkKeys secondary index must stay strictly bounded to resident _memCache entries'
  );
  let totalReadaheadIndexed = 0;
  for (const map of cache._readaheadByPath.values()) {
    totalReadaheadIndexed += map.size;
  }
  assert.equal(
    totalReadaheadIndexed,
    cache._readaheadKeys.size,
    '_readaheadByPath secondary index must stay synchronized with _readaheadKeys'
  );
  assert.ok(cache.getStats().evictions > 50, 'tight budget stress must exercise repeated LRU evictions');
});
