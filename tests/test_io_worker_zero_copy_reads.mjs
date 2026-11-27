import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const ioWorkerMod = require('../web/workers/io_worker.js');
const mailboxMod = require('../web/workers/httpfs_mailbox.js');
const rangeCacheMod = require('../web/workers/range_cache.js');

const {
  IoWorkerController,
  emptyStats,
  emptyStreamingStats,
} = ioWorkerMod;

const {
  HttpfsStatus,
  HttpfsOp,
  HttpfsOffsets,
  createHttpfsMailboxBuffer,
  createHttpfsMailboxPool,
  HttpfsMailboxClient,
} = mailboxMod;

const { RangeCache } = rangeCacheMod;

function makePatternBytes(length, seed = 0x3b) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 131) ^ ((i >> 5) * 47) ^ ((i >> 11) * 19) ^ seed) & 0xff;
  }
  return out;
}

function createMockRangeFetch(urlBytesMap, options = {}) {
  const requests = [];
  const statusOverrides = options.statusOverrides || new Map();
  const throwUrls = options.throwUrls || new Set();

  async function fetchImpl(url, init = {}) {
    const method = (init && init.method) || 'GET';
    const rangeHeader = (init && init.headers && (init.headers.Range || init.headers.range)) || null;
    requests.push({ method, url, rangeHeader });

    if (throwUrls.has(url)) {
      const err = new Error('Simulated network failure');
      err.code = 'ERR_NETWORK';
      throw err;
    }

    const override = statusOverrides.get(url);
    const overrideStatus = typeof override === 'function' ? override(requests.length) : override;
    if (overrideStatus) {
      return {
        ok: false,
        status: overrideStatus,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
        async json() {
          throw new Error(`HTTP ${overrideStatus}`);
        },
      };
    }

    const data = urlBytesMap.get(url);
    if (!data) {
      return {
        ok: false,
        status: 404,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
        async json() {
          throw new Error('HTTP 404');
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
        async json() {
          return JSON.parse(new TextDecoder().decode(data));
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

  return { fetchImpl, requests };
}

function submitRawReadRequest(sab, handle, offsetBig, lengthBig) {
  const statusView = new Int32Array(sab, 0, 1);
  const headerView = new DataView(sab, 0, 40);
  headerView.setUint32(HttpfsOffsets.OP, HttpfsOp.Read, true);
  headerView.setInt32(HttpfsOffsets.HANDLE, handle, true);
  headerView.setBigUint64(HttpfsOffsets.OFFSET, offsetBig, true);
  headerView.setBigUint64(HttpfsOffsets.LENGTH, lengthBig, true);
  headerView.setBigInt64(HttpfsOffsets.RESULT, -1n, true);
  Atomics.store(statusView, 0, HttpfsStatus.RequestReady);
}

function readRawResponse(sab) {
  const statusView = new Int32Array(sab, 0, 1);
  const headerView = new DataView(sab, 0, 40);
  const payloadView = new Uint8Array(sab, 40, sab.byteLength - 40);
  const status = Atomics.load(statusView, 0);
  const result = headerView.getBigInt64(HttpfsOffsets.RESULT, true);
  const bytes =
    status === HttpfsStatus.ResponseReady && result > 0n
      ? payloadView.slice(0, Number(result))
      : new Uint8Array(0);
  Atomics.store(statusView, 0, HttpfsStatus.Idle);
  return { status, result, bytes };
}

async function rawMailboxRead(sab, handle, offsetBig, lengthBig) {
  const statusView = new Int32Array(sab, 0, 1);
  submitRawReadRequest(sab, handle, offsetBig, lengthBig);
  Atomics.notify(statusView, 0, 1);

  let cur = Atomics.load(statusView, 0);
  while (cur === HttpfsStatus.RequestReady) {
    const waitObj = Atomics.waitAsync(statusView, 0, HttpfsStatus.RequestReady, 5000);
    const res = waitObj.async ? await waitObj.value : waitObj.value;
    cur = Atomics.load(statusView, 0);
    if (res === 'timed-out' && cur === HttpfsStatus.RequestReady) {
      throw new Error('Mailbox hung during raw READ');
    }
  }
  return readRawResponse(sab);
}

// ---------------------------------------------------------------------------
// 1. Telemetry defaults & sanitized counter shape
// ---------------------------------------------------------------------------

test('Issue #2192: emptyStats and emptyStreamingStats expose rangeCacheSyncZeroCopyHits and rangeCacheReadIntoHits defaulting to 0', () => {
  const top = emptyStats();
  const stream = emptyStreamingStats();

  assert.equal(top.rangeCacheSyncZeroCopyHits, 0);
  assert.equal(top.rangeCacheReadIntoHits, 0);
  assert.equal(top.streaming.rangeCacheSyncZeroCopyHits, 0);
  assert.equal(top.streaming.rangeCacheReadIntoHits, 0);
  assert.equal(stream.rangeCacheSyncZeroCopyHits, 0);
  assert.equal(stream.rangeCacheReadIntoHits, 0);
});

// ---------------------------------------------------------------------------
// 2. Single-chunk & multi-chunk L1 sync hits + L1 miss async readInto + zero allocs
// ---------------------------------------------------------------------------

test('Issue #2192: RangeCache SAB mailbox reads use async readInto on miss and sync zero-copy L1 on single- & multi-chunk hits with zero slice() or Promise allocations', async () => {
  const chunkSize = 4096;
  const fileSize = 4 * chunkSize + 777; // 17,161 bytes (non-chunk-aligned tail)
  const payload = makePatternBytes(fileSize, 0x6d);

  const mock = createMockRangeFetch(new Map([['/data/stream.bin', payload]]));
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize,
    maxBytes: 256 * 1024,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(32 * 1024);
  const client = new HttpfsMailboxClient(sab);

  const origSlice = Uint8Array.prototype.slice;
  let sliceCalls = 0;

  try {
    await controller.init({ dataBase: '/data/' });

    // Spy on RangeCache methods to confirm read() is NEVER called on SAB mailbox reads!
    const rc = controller.rangeCache;
    let rcReadCalls = 0;
    let rcReadIntoCalls = 0;
    let rcGetCachedSliceCalls = 0;
    let rcReadCachedRangeIntoCalls = 0;

    const origRead = rc.read.bind(rc);
    const origReadInto = rc.readInto.bind(rc);
    const origGetCachedSlice = rc.getCachedSlice.bind(rc);
    const origReadCachedRangeInto = rc.readCachedRangeInto.bind(rc);

    rc.read = (...args) => {
      rcReadCalls += 1;
      return origRead(...args);
    };
    rc.readInto = (...args) => {
      rcReadIntoCalls += 1;
      return origReadInto(...args);
    };
    rc.getCachedSlice = (...args) => {
      rcGetCachedSliceCalls += 1;
      return origGetCachedSlice(...args);
    };
    rc.readCachedRangeInto = (...args) => {
      rcReadCachedRangeIntoCalls += 1;
      return origReadCachedRangeInto(...args);
    };

    // Attach mailbox with probeOnOpen: false so openHandle doesn't pre-warm chunk 0.
    await controller.attachMailbox(sab, { probeOnOpen: false });
    const handle = await client.openAsync('stream.bin');
    assert.ok(handle >= 1);
    assert.equal(controller.getHandleInfo(handle).size, null);

    // A. Cold L1 miss (single chunk 0) -> MUST use async rangeCache.readInto directly into SAB!
    const bufA = new Uint8Array(1024);
    const nA = await client.readAsync(handle, 128n, bufA, 1024);
    assert.equal(nA, 1024);
    assert.deepEqual(bufA, payload.subarray(128, 128 + 1024));
    assert.equal(rcReadCalls, 0, 'Cold single-chunk SAB read must not fall back to rangeCache.read()');
    assert.equal(rcReadIntoCalls, 1);
    assert.equal(controller.getHandleInfo(handle).size, fileSize, 'entry.size must be learned after readInto');

    // B. Cold L1 miss spanning chunks 1..3 -> MUST use async rangeCache.readInto directly into SAB!
    const multiMissOff = chunkSize + 300;
    const multiMissLen = 2 * chunkSize + 500;
    const bufB = new Uint8Array(multiMissLen);
    const nB = await client.readAsync(handle, BigInt(multiMissOff), bufB, multiMissLen);
    assert.equal(nB, multiMissLen);
    assert.deepEqual(bufB, payload.subarray(multiMissOff, multiMissOff + multiMissLen));
    assert.equal(rcReadCalls, 0, 'Cold multi-chunk SAB read must not fall back to rangeCache.read()');
    assert.equal(rcReadIntoCalls, 2);

    // C. Cold L1 miss hitting tail chunk 4 and clamping at EOF -> async rangeCache.readInto!
    const eofClampOff = 4 * chunkSize + 200;
    const bufC = new Uint8Array(2048);
    const nC = await client.readAsync(handle, BigInt(eofClampOff), bufC, 2048);
    assert.equal(nC, fileSize - eofClampOff);
    assert.deepEqual(bufC.subarray(0, nC), payload.subarray(eofClampOff, fileSize));
    assert.equal(rcReadIntoCalls, 3);

    const statsAfterCold = controller.getStats();
    assert.equal(statsAfterCold.rangeCacheReadIntoHits, 3);
    assert.equal(statsAfterCold.streaming.rangeCacheReadIntoHits, 3);
    assert.equal(statsAfterCold.rangeCacheSyncZeroCopyHits, 0);

    // Now ALL 5 chunks (0..4) are resident in RangeCache L1!
    // Instrument Uint8Array.prototype.slice across warm single-chunk (<=64B & >64B),
    // warm multi-chunk, warm EOF clamp, warm past-EOF, and 0-byte reads.
    const reqsBeforeWarm = mock.requests.length;
    const syncHitsBeforeWarm = controller.getStats().syncFastPathHits;
    const asyncHitsBeforeWarm = controller.getStats().mailbox.asyncPathHits;

    Uint8Array.prototype.slice = function instrumentedSlice(...args) {
      sliceCalls += 1;
      return origSlice.apply(this, args);
    };

    try {
      // 1. Warm single-chunk tiny read (<= 64 bytes -> loop copy, zero slice)
      const tinyOut = new Uint8Array(37);
      assert.equal(await client.readAsync(handle, 55n, tinyOut, 37), 37);
      assert.deepEqual(tinyOut, payload.subarray(55, 55 + 37));

      // 2. Warm single-chunk large read (> 64 bytes -> getCachedSlice + payloadView.set)
      const singleOut = new Uint8Array(2048);
      assert.equal(await client.readAsync(handle, 512n, singleOut, 2048), 2048);
      assert.deepEqual(singleOut, payload.subarray(512, 512 + 2048));

      // 3. Warm multi-chunk read spanning chunks 0..3 -> readCachedRangeInto direct copy
      const crossOff = chunkSize - 512;
      const crossLen = 2 * chunkSize + 1024;
      const crossOut = new Uint8Array(crossLen);
      assert.equal(await client.readAsync(handle, BigInt(crossOff), crossOut, crossLen), crossLen);
      assert.deepEqual(crossOut, payload.subarray(crossOff, crossOff + crossLen));

      // 4. Warm multi-chunk read straddling EOF (chunk 3 -> chunk 4 tail)
      const eofMultiOff = 3 * chunkSize + 3500;
      const eofMultiOut = new Uint8Array(4096);
      const nEofMulti = await client.readAsync(handle, BigInt(eofMultiOff), eofMultiOut, 4096);
      assert.equal(nEofMulti, fileSize - eofMultiOff);
      assert.deepEqual(eofMultiOut.subarray(0, nEofMulti), payload.subarray(eofMultiOff, fileSize));

      // 5. Warm past-EOF read -> 0 bytes synchronously
      const pastEofOut = new Uint8Array(64);
      assert.equal(await client.readAsync(handle, BigInt(fileSize + 100), pastEofOut, 64), 0);

      // 6. 0-byte read via client & raw SAB mailbox -> 0 bytes synchronously
      assert.equal(await client.readAsync(handle, 0n, pastEofOut, 0), 0);
      const rawZeroRes = await rawMailboxRead(sab, handle, 0n, 0n);
      assert.equal(rawZeroRes.status, HttpfsStatus.ResponseReady);
      assert.equal(rawZeroRes.result, 0n);

      assert.equal(
        sliceCalls,
        0,
        `Expected ZERO Uint8Array.prototype.slice() calls on warm RangeCache SAB hits, got ${sliceCalls}`,
      );
    } finally {
      Uint8Array.prototype.slice = origSlice;
    }

    assert.equal(mock.requests.length, reqsBeforeWarm, 'Warm L1 reads must issue 0 HTTP requests');
    assert.equal(rcReadCalls, 0, 'rangeCache.read() must never be called');
    assert.equal(rcReadIntoCalls, 3, 'Warm L1 reads must not invoke async readInto()');
    assert.ok(rcGetCachedSliceCalls >= 3, 'Single-chunk warm hits must use getCachedSlice()');
    assert.ok(rcReadCachedRangeIntoCalls >= 2, 'Multi-chunk warm hits must use readCachedRangeInto()');

    const statsAfterWarm = controller.getStats();
    // 5 RangeCache sync L1 hits (tiny, single, cross, eofMulti, pastEof); 0-byte returns before cache lookup.
    assert.equal(statsAfterWarm.rangeCacheSyncZeroCopyHits, 5);
    assert.equal(statsAfterWarm.streaming.rangeCacheSyncZeroCopyHits, 5);
    assert.equal(statsAfterWarm.rangeCacheReadIntoHits, 3);
    assert.equal(
      statsAfterWarm.syncFastPathHits - syncHitsBeforeWarm,
      6,
      'All 6 warm reads (5 L1 sync + 1 zero-byte) must complete on the synchronous mailbox fast path',
    );
    assert.equal(
      statsAfterWarm.mailbox.asyncPathHits,
      asyncHitsBeforeWarm,
      'Zero warm reads may enter the async mailbox path',
    );

    assert.equal(await client.closeAsync(handle), true);
  } finally {
    Uint8Array.prototype.slice = origSlice;
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 3. Direct synchronous _dispatchOnce verification (strictly non-Promise return)
// ---------------------------------------------------------------------------

test('Issue #2192: _dispatchOnce() returns synchronous boolean true (no Promise) on single-chunk & multi-chunk L1 hits', async () => {
  const chunkSize = 1024;
  const payload = makePatternBytes(3 * chunkSize + 333, 0xa5);
  const mock = createMockRangeFetch(new Map([['/data/sync_check.bin', payload]]));

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize,
    maxBytes: 64 * 1024,
    opfsDir: null,
  });

  await controller.init({ dataBase: '/data/' });
  // Pre-warm all chunks in RangeCache L1.
  await controller.rangeCache.read('sync_check.bin', 0, payload.byteLength);

  const sab = createHttpfsMailboxBuffer(8192);
  const { IoWorkerMailboxHost, hostOptions } = controller._createMailboxHostBinding(mailboxMod, {
    probeOnOpen: false,
  });
  const host = new IoWorkerMailboxHost(sab, hostOptions);

  const handle = await controller.openHandle('sync_check.bin', { probeOnOpen: false });
  assert.ok(handle >= 1);

  // Single-chunk L1 hit: _dispatchOnce MUST return strictly `true` (not a Promise/thenable)!
  submitRawReadRequest(sab, handle, 200n, 300n);
  const r1 = host._dispatchOnce();
  assert.strictEqual(r1, true, 'Single-chunk L1 hit must return synchronous boolean true, not a Promise');
  const resp1 = readRawResponse(sab);
  assert.equal(resp1.status, HttpfsStatus.ResponseReady);
  assert.equal(resp1.result, 300n);
  assert.deepEqual(resp1.bytes, payload.subarray(200, 500));

  // Multi-chunk L1 hit spanning chunks 0..2: _dispatchOnce MUST return strictly `true`!
  submitRawReadRequest(sab, handle, 900n, 1600n);
  const r2 = host._dispatchOnce();
  assert.strictEqual(r2, true, 'Multi-chunk L1 hit must return synchronous boolean true, not a Promise');
  const resp2 = readRawResponse(sab);
  assert.equal(resp2.status, HttpfsStatus.ResponseReady);
  assert.equal(resp2.result, 1600n);
  assert.deepEqual(resp2.bytes, payload.subarray(900, 2500));

  // Verify mailbox stats recorded 2 sync hits and 0 async hits.
  const mStats = host.getStats();
  assert.equal(mStats.syncFastPathHits, 2);
  assert.equal(mStats.asyncPathHits, 0);
  assert.equal(controller.getStats().rangeCacheSyncZeroCopyHits, 2);
  assert.equal(controller.getStats().rangeCacheReadIntoHits, 0);
  assert.equal(controller.getHandleInfo(handle).size, payload.byteLength);
});

// ---------------------------------------------------------------------------
// 4. Error handling & errno propagation (404, 403, 500, network throw, sync throw)
// ---------------------------------------------------------------------------

test('Issue #2192: RangeCache zero-copy path propagates HTTP 404/403/500, network errors, and sync throws to exact negative errnos without hanging mailbox', async () => {
  const goodBytes = makePatternBytes(2048, 0x22);
  const mock = createMockRangeFetch(
    new Map([['/data/healthy.bin', goodBytes]]),
    {
      statusOverrides: new Map([
        ['/data/missing.bin', 404],
        ['/data/forbidden.bin', 403],
        ['/data/server_err.bin', 500],
      ]),
      throwUrls: new Set(['/data/net_fail.bin']),
    },
  );

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize: 1024,
    maxBytes: 64 * 1024,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(4096);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.init({ dataBase: '/data/' });
    await controller.attachMailbox(sab, { probeOnOpen: false });

    const hMissing = await client.openAsync('missing.bin');
    const hForbidden = await client.openAsync('forbidden.bin');
    const hServerErr = await client.openAsync('server_err.bin');
    const hNetFail = await client.openAsync('net_fail.bin');
    const hHealthy = await client.openAsync('healthy.bin');

    const res404 = await rawMailboxRead(sab, hMissing, 0n, 256n);
    assert.equal(res404.status, HttpfsStatus.Error);
    assert.equal(res404.result, -2n, 'HTTP 404 must map to -2 (ENOENT)');

    const res403 = await rawMailboxRead(sab, hForbidden, 0n, 256n);
    assert.equal(res403.status, HttpfsStatus.Error);
    assert.equal(res403.result, -13n, 'HTTP 403 must map to -13 (EACCES)');

    const res500 = await rawMailboxRead(sab, hServerErr, 0n, 256n);
    assert.equal(res500.status, HttpfsStatus.Error);
    assert.equal(res500.result, -5n, 'HTTP 500 must map to -5 (EIO)');

    const resNet = await rawMailboxRead(sab, hNetFail, 0n, 256n);
    assert.equal(resNet.status, HttpfsStatus.Error);
    assert.equal(resNet.result, -5n, 'Network error must map to -5 (EIO)');

    // Out-of-bounds offset (> MAX_SAFE_INTEGER) -> -22 (EINVAL)
    const resOob = await rawMailboxRead(sab, hHealthy, BigInt(Number.MAX_SAFE_INTEGER) + 1n, 64n);
    assert.equal(resOob.status, HttpfsStatus.Error);
    assert.equal(resOob.result, -22n);

    // Mailbox remains 100% healthy for subsequent reads!
    const out = new Uint8Array(512);
    assert.equal(await client.readAsync(hHealthy, 100n, out, 512), 512);
    assert.deepEqual(out, goodBytes.subarray(100, 612));

    const stats = controller.getStats();
    assert.ok(stats.errorCount >= 4, `Expected errorCount >= 4, got ${stats.errorCount}`);
    assert.equal(stats.rangeCacheReadIntoHits, 1);

    // Also verify sync throw from a broken custom getCachedSlice maps errno cleanly.
    const origGetSlice = controller.rangeCache.getCachedSlice.bind(controller.rangeCache);
    controller.rangeCache.getCachedSlice = () => {
      const e = new Error('Simulated EACCES sync fault');
      e.errno = -13;
      throw e;
    };
    const resSyncThrow = await rawMailboxRead(sab, hHealthy, 0n, 64n);
    assert.equal(resSyncThrow.status, HttpfsStatus.Error);
    assert.equal(resSyncThrow.result, -13n);

    // Restore and confirm healthy sync hit still works.
    controller.rangeCache.getCachedSlice = origGetSlice;
    const out2 = new Uint8Array(64);
    assert.equal(await client.readAsync(hHealthy, 0n, out2, 64), 64);
    assert.deepEqual(out2, goodBytes.subarray(0, 64));
  } finally {
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 5. Custom mock RangeCache fallback compatibility
// ---------------------------------------------------------------------------

test('Issue #2192: custom mock RangeCache without readInto/getCachedSlice/readCachedRangeInto falls back cleanly to legacy read()', async () => {
  const payload = makePatternBytes(4096, 0x77);
  let legacyReadCalls = 0;

  // Minimal legacy mock exposing ONLY read() and getStats().
  const minimalMockRangeCache = {
    async read(relPath, offset, length) {
      legacyReadCalls += 1;
      if (relPath !== 'legacy.bin') {
        const err = new Error('HTTP 404');
        err.status = 404;
        throw err;
      }
      if (offset >= payload.byteLength) {
        return new Uint8Array(0);
      }
      return payload.slice(offset, Math.min(payload.byteLength, offset + length));
    },
    getStats() {
      return { hits: 0, misses: legacyReadCalls, opfsHits: 0, evictions: 0, bytesCached: 0, entries: 0 };
    },
  };

  const controller = new IoWorkerController({
    rangeCache: minimalMockRangeCache,
    immutableCacheEnabled: false,
  });

  const sab = createHttpfsMailboxBuffer(4096);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.attachMailbox(sab, { probeOnOpen: false });
    const h = await client.openAsync('legacy.bin');
    assert.ok(h >= 1);

    const out = new Uint8Array(512);
    assert.equal(await client.readAsync(h, 256n, out, 512), 512);
    assert.deepEqual(out, payload.subarray(256, 768));

    // EOF clamping on legacy mock
    const tail = new Uint8Array(512);
    const tailLen = await client.readAsync(h, 3900n, tail, 512);
    assert.equal(tailLen, 196);
    assert.deepEqual(tail.subarray(0, tailLen), payload.subarray(3900, 4096));

    assert.equal(legacyReadCalls, 2);
    const stats = controller.getStats();
    assert.equal(stats.rangeCacheSyncZeroCopyHits, 0);
    assert.equal(stats.rangeCacheReadIntoHits, 0);
    assert.equal(await client.closeAsync(h), true);
  } finally {
    await controller.detachMailbox();
  }
});

test('Issue #2192: partial custom mock RangeCache exposing ONLY getCachedSlice or ONLY readInto works cleanly', async () => {
  const payload = makePatternBytes(2048, 0x19);
  let sliceHits = 0;
  let readIntoCalls = 0;
  let warm = false;

  const partialMock = {
    getCachedSlice(_path, offset, length) {
      if (!warm) return null;
      sliceHits += 1;
      if (offset >= payload.byteLength) return new Uint8Array(0);
      return payload.subarray(offset, Math.min(payload.byteLength, offset + length));
    },
    async readInto(_path, offset, length, dst, dstOffset = 0) {
      readIntoCalls += 1;
      warm = true;
      if (offset >= payload.byteLength) return 0;
      const end = Math.min(payload.byteLength, offset + length);
      const len = end - offset;
      dst.set(payload.subarray(offset, end), dstOffset);
      return len;
    },
    async read() {
      throw new Error('Legacy read() should not be called when readInto is available');
    },
    getStats() {
      return { hits: sliceHits, misses: readIntoCalls, opfsHits: 0, evictions: 0, bytesCached: 0, entries: 1 };
    },
  };

  const controller = new IoWorkerController({
    rangeCache: partialMock,
    immutableCacheEnabled: false,
  });

  const sab = createHttpfsMailboxBuffer(4096);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.attachMailbox(sab, { probeOnOpen: false });
    const h = await client.openAsync('partial.bin');

    // First read misses getCachedSlice -> invokes readInto
    const out1 = new Uint8Array(400);
    assert.equal(await client.readAsync(h, 100n, out1, 400), 400);
    assert.deepEqual(out1, payload.subarray(100, 500));
    assert.equal(readIntoCalls, 1);
    assert.equal(sliceHits, 0);

    // Second read hits getCachedSlice synchronously
    const out2 = new Uint8Array(400);
    assert.equal(await client.readAsync(h, 500n, out2, 400), 400);
    assert.deepEqual(out2, payload.subarray(500, 900));
    assert.equal(readIntoCalls, 1);
    assert.equal(sliceHits, 1);

    const stats = controller.getStats();
    assert.equal(stats.rangeCacheReadIntoHits, 1);
    assert.equal(stats.rangeCacheSyncZeroCopyHits, 1);
    assert.equal(await client.closeAsync(h), true);
  } finally {
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 6. Routed mounts (/shaders/) & multi-lane mailbox pool parity
// ---------------------------------------------------------------------------

test('Issue #2192: routed /shaders/ mount and 4-lane HttpfsMailboxPoolHost use RangeCache zero-copy readInto + sync L1 cleanly', async () => {
  const chunkSize = 2048;
  const packBytes = makePatternBytes(3 * chunkSize + 511, 0x8e);
  const mock = createMockRangeFetch(new Map([['/shaders/pipeline_index.bin', packBytes]]));

  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize,
    maxBytes: 128 * 1024,
    opfsDir: null,
  });

  const poolSabs = createHttpfsMailboxPool(4, 8192);
  const clients = poolSabs.map((sab) => new HttpfsMailboxClient(sab));

  try {
    await controller.init({ dataBase: '/data/', shadersBase: '/shaders/' });
    await controller.attachMailboxPool(poolSabs, { probeOnOpen: false });

    const h = await clients[0].openAsync('/shaders/pipeline_index.bin');
    assert.ok(h >= 1);

    // Cold concurrent reads across 4 lanes spanning all chunks
    const offsets = [0, chunkSize - 128, 2 * chunkSize + 64, 3 * chunkSize + 100];
    const lengths = [512, 1024, 1500, 1024];
    const bufs = lengths.map((l) => new Uint8Array(l));
    const gotLens = await Promise.all(
      clients.map((c, i) => c.readAsync(h, BigInt(offsets[i]), bufs[i], lengths[i])),
    );

    for (let i = 0; i < 4; i += 1) {
      const expectedEnd = Math.min(packBytes.byteLength, offsets[i] + lengths[i]);
      const expectedLen = expectedEnd - offsets[i];
      assert.equal(gotLens[i], expectedLen);
      assert.deepEqual(bufs[i].subarray(0, expectedLen), packBytes.subarray(offsets[i], expectedEnd));
    }

    // Warm concurrent reads across all 4 lanes -> all must hit sync L1 zero-copy!
    const syncBefore = controller.getStats().rangeCacheSyncZeroCopyHits;
    const warmBufs = lengths.map((l) => new Uint8Array(l));
    const warmLens = await Promise.all(
      clients.map((c, i) => c.readAsync(h, BigInt(offsets[i]), warmBufs[i], lengths[i])),
    );
    for (let i = 0; i < 4; i += 1) {
      const expectedEnd = Math.min(packBytes.byteLength, offsets[i] + lengths[i]);
      const expectedLen = expectedEnd - offsets[i];
      assert.equal(warmLens[i], expectedLen);
      assert.deepEqual(warmBufs[i].subarray(0, expectedLen), packBytes.subarray(offsets[i], expectedEnd));
    }

    const stats = controller.getStats();
    assert.equal(stats.rangeCacheSyncZeroCopyHits - syncBefore, 4);
    assert.ok(stats.rangeCacheReadIntoHits >= 1);
    assert.equal(await clients[0].closeAsync(h), true);
  } finally {
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 7. High-throughput burst stress test: 1,200+ SAB mailbox reads across chunk boundaries
// ---------------------------------------------------------------------------

test('Issue #2192: high-throughput burst stress (1,200 SAB mailbox reads across chunk boundaries) is 100% byte-exact with zero warm slice() allocations', async () => {
  const chunkSize = 4096;
  const numChunks = 16;
  const fileSize = numChunks * chunkSize + 1234; // 66,770 bytes
  const payload = makePatternBytes(fileSize, 0xc3);

  const mock = createMockRangeFetch(new Map([['/data/burst_asset.bin', payload]]));
  const controller = new IoWorkerController({
    fetchImpl: mock.fetchImpl,
    chunkSize,
    maxBytes: 4 * 1024 * 1024,
    opfsDir: null,
  });

  const sab = createHttpfsMailboxBuffer(32 * 1024);
  const client = new HttpfsMailboxClient(sab);

  const origSlice = Uint8Array.prototype.slice;
  let warmSliceCalls = 0;

  try {
    await controller.init({ dataBase: '/data/' });
    await controller.attachMailbox(sab, { probeOnOpen: false });

    const handle = await client.openAsync('burst_asset.bin');
    assert.ok(handle >= 1);

    // Phase 1: Cold fill via 200 sequential/straddling reads (exercises readInto + sync L1 hits).
    const dst = new Uint8Array(16 * 1024);
    const coldReads = 200;
    for (let i = 0; i < coldReads; i += 1) {
      const off = (i * 357) % (fileSize + 128);
      const len = 1 + ((i * 997) % 10000);
      const got = await client.readAsync(handle, BigInt(off), dst, len);
      const expectedLen = off >= fileSize ? 0 : Math.min(len, fileSize - off);
      assert.equal(got, expectedLen, `Cold read #${i} length mismatch at off=${off} len=${len}`);
      for (let b = 0; b < got; b += 1) {
        if (dst[b] !== payload[off + b]) {
          assert.fail(`Cold read #${i} byte mismatch at offset ${off + b}`);
        }
      }
    }

    // Ensure every chunk is resident before warm stress phase.
    const fullWarm = new Uint8Array(16 * 1024);
    for (let off = 0; off < fileSize; off += 16 * 1024) {
      await client.readAsync(handle, BigInt(off), fullWarm, 16 * 1024);
    }

    const statsBeforeWarm = controller.getStats();
    const reqsBeforeWarm = mock.requests.length;

    // Phase 2: 1,050 warm randomized single-chunk, tiny (<=64B), multi-chunk, and EOF reads.
    const warmReads = 1050;
    Uint8Array.prototype.slice = function instrumentedSlice(...args) {
      warmSliceCalls += 1;
      return origSlice.apply(this, args);
    };

    try {
      let rng = 0x12345678;
      for (let i = 0; i < warmReads; i += 1) {
        rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
        let off;
        let len;
        const mode = i % 5;
        if (mode === 0) {
          // Tiny single-chunk read (<= 64 bytes)
          off = rng % fileSize;
          len = 1 + ((rng >>> 8) % 64);
        } else if (mode === 1) {
          // Single-chunk medium read
          const chunkIdx = rng % numChunks;
          const within = (rng >>> 4) % (chunkSize - 512);
          off = chunkIdx * chunkSize + within;
          len = 65 + ((rng >>> 12) % 400);
        } else if (mode === 2) {
          // Multi-chunk boundary-straddling read (2 to 4 chunks)
          const boundary = (1 + (rng % (numChunks - 2))) * chunkSize;
          const back = 1 + ((rng >>> 8) % 1024);
          off = boundary - back;
          len = back + chunkSize + ((rng >>> 16) % (2 * chunkSize));
        } else if (mode === 3) {
          // EOF-clamped read
          off = fileSize - 1 - (rng % 2048);
          len = 1024 + ((rng >>> 8) % 4096);
        } else {
          // Past-EOF read
          off = fileSize + (rng % 512);
          len = 1 + ((rng >>> 8) % 256);
        }

        const got = await client.readAsync(handle, BigInt(off), dst, len);
        const expectedLen = off >= fileSize ? 0 : Math.min(len, fileSize - off);
        assert.equal(got, expectedLen, `Warm read #${i} length mismatch at off=${off} len=${len}`);
        for (let b = 0; b < got; b += 1) {
          if (dst[b] !== payload[off + b]) {
            assert.fail(`Warm read #${i} byte mismatch at offset ${off + b}`);
          }
        }
      }

      assert.equal(
        warmSliceCalls,
        0,
        `Expected ZERO Uint8Array.prototype.slice() calls across ${warmReads} warm reads, got ${warmSliceCalls}`,
      );
    } finally {
      Uint8Array.prototype.slice = origSlice;
    }

    assert.equal(mock.requests.length, reqsBeforeWarm, 'Warm burst must trigger 0 HTTP fetches');

    const finalStats = controller.getStats();
    assert.equal(
      finalStats.rangeCacheSyncZeroCopyHits - statsBeforeWarm.rangeCacheSyncZeroCopyHits,
      warmReads,
      `All ${warmReads} warm reads must increment rangeCacheSyncZeroCopyHits`,
    );
    assert.equal(
      finalStats.rangeCacheReadIntoHits,
      statsBeforeWarm.rangeCacheReadIntoHits,
      'Warm burst must not increment rangeCacheReadIntoHits',
    );
    assert.equal(
      finalStats.syncFastPathHits - statsBeforeWarm.syncFastPathHits,
      warmReads,
      `All ${warmReads} warm reads must complete on sync mailbox fast path`,
    );
    assert.ok(finalStats.rangeCacheReadIntoHits > 0);
    assert.ok(finalStats.rangeCache.zeroAllocCopyHits >= warmReads - Math.ceil(warmReads / 5));

    assert.equal(await client.closeAsync(handle), true);
  } finally {
    Uint8Array.prototype.slice = origSlice;
    await controller.detachMailbox();
  }
});

// ---------------------------------------------------------------------------
// 8. RangeCache instance passed directly via options.rangeCache works identically
// ---------------------------------------------------------------------------

test('Issue #2192: externally constructed RangeCache instance wired into IoWorkerController uses zero-copy fast paths', async () => {
  const chunkSize = 2048;
  const payload = makePatternBytes(3 * chunkSize + 111, 0x4f);
  const mock = createMockRangeFetch(new Map([['/data/ext.bin', payload]]));

  const externalCache = new RangeCache({
    dataBase: '/data/',
    chunkSize,
    maxBytes: 64 * 1024,
    fetchImpl: mock.fetchImpl,
  });

  const controller = new IoWorkerController({
    rangeCache: externalCache,
    immutableCacheEnabled: false,
  });

  const sab = createHttpfsMailboxBuffer(8192);
  const client = new HttpfsMailboxClient(sab);

  try {
    await controller.attachMailbox(sab, { probeOnOpen: false });
    const h = await client.openAsync('ext.bin');

    // Cold multi-chunk miss -> readInto
    const out1 = new Uint8Array(3000);
    assert.equal(await client.readAsync(h, 1500n, out1, 3000), 3000);
    assert.deepEqual(out1, payload.subarray(1500, 4500));

    // Warm single-chunk hit -> getCachedSlice
    const out2 = new Uint8Array(500);
    assert.equal(await client.readAsync(h, 200n, out2, 500), 500);
    assert.deepEqual(out2, payload.subarray(200, 700));

    // Warm multi-chunk hit -> readCachedRangeInto
    const out3 = new Uint8Array(2500);
    assert.equal(await client.readAsync(h, 1800n, out3, 2500), 2500);
    assert.deepEqual(out3, payload.subarray(1800, 4300));

    const stats = controller.getStats();
    assert.equal(stats.rangeCacheReadIntoHits, 1);
    assert.equal(stats.rangeCacheSyncZeroCopyHits, 2);
    assert.equal(await client.closeAsync(h), true);
  } finally {
    await controller.detachMailbox();
  }
});
