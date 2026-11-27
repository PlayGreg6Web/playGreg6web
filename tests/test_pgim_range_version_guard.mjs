import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const pgimMod = require('../web/runtime/pgim_preloader.js');

const {
  PGIM_MAGIC_BYTES,
  PGIM_VERSION,
  PGIM_HEADER_BYTES,
  PGIM_SECTION_ENTRY_BYTES,
  PGIM_PAGE_SIZE,
  PGIM_FLAGS,
  PGIM_SECTION_FLAGS,
  PGIM_PROT,
  PGIM_OPFS_CACHE_VERSION,
  PGIM_OPFS_PROVENANCE_SCHEMA,
  DEFAULT_MAX_VERSION_RETRIES,
  PgimPreloadError,
  PgimPreloader,
  isStrongEtag,
  isWeakEtag,
  buildPgimOpfsFingerprint,
  pgimFnv1a64,
} = pgimMod;

// ---------------------------------------------------------------------------------------------
// Synthetic PGIM v1 builder + in-memory OPFS store helper
// ---------------------------------------------------------------------------------------------

function alignUp(value, align) {
  return Math.ceil(value / align) * align;
}

function filled(length, seed) {
  const out = new Uint8Array(length);
  let x = (seed >>> 0) || 1;
  for (let i = 0; i < length; i += 1) {
    x = (Math.imul(x, 1103515245) + 12345) >>> 0;
    out[i] = x >>> 24;
  }
  return out;
}

function buildPgim(seed, { textBytes = 0x3800, dataBytes = 0x2400, imageBase = 0x140000000n } = {}) {
  const textVSize = alignUp(textBytes, 0x1000);
  const dataRva = 0x1000 + textVSize;
  const dataVSize = alignUp(dataBytes, 0x1000);
  const sizeOfImage = dataRva + dataVSize;
  const sections = [
    {
      name: '<headers>',
      rva: 0,
      virtualSize: 0x400,
      protection: PGIM_PROT.READ,
      flags: PGIM_SECTION_FLAGS.HEADERS,
      data: filled(0x400, seed),
    },
    {
      name: '.text',
      rva: 0x1000,
      virtualSize: textVSize,
      protection: PGIM_PROT.READ | PGIM_PROT.EXECUTE,
      data: filled(textBytes, seed + 1),
      characteristics: 0x60000020,
    },
    {
      name: '.data',
      rva: dataRva,
      virtualSize: dataVSize,
      protection: PGIM_PROT.READ | PGIM_PROT.WRITE,
      data: filled(dataBytes, seed + 2),
      characteristics: 0xc0000040,
    },
  ];
  const dir64 = [dataRva, dataRva + 0x10];
  const highlow = [dataRva + 0x20];
  const tablesEnd =
    PGIM_HEADER_BYTES + sections.length * PGIM_SECTION_ENTRY_BYTES + (dir64.length + highlow.length) * 4;
  let cursor = tablesEnd;
  const offsets = sections.map((s) => {
    const len = s.data ? s.data.byteLength : 0;
    if (len === 0) return 0;
    const offset = alignUp(cursor, PGIM_PAGE_SIZE);
    cursor = offset + len;
    return offset;
  });
  const fileSize = cursor;
  const out = new Uint8Array(fileSize);
  const view = new DataView(out.buffer);
  out.set(PGIM_MAGIC_BYTES, 0);
  view.setUint32(4, PGIM_VERSION, true);
  view.setUint32(8, PGIM_HEADER_BYTES, true);
  view.setUint32(12, PGIM_SECTION_ENTRY_BYTES, true);
  view.setUint32(16, PGIM_FLAGS.HEADERS_MAPPED, true);
  view.setUint32(20, PGIM_PAGE_SIZE, true);
  view.setBigUint64(24, imageBase, true);
  view.setUint32(32, sizeOfImage, true);
  view.setUint32(36, 0x1000, true);
  view.setUint32(40, 0x1010, true);
  view.setUint32(44, sections.length, true);
  view.setUint32(48, dir64.length, true);
  view.setUint32(52, highlow.length, true);
  view.setBigUint64(56, BigInt(fileSize), true);
  sections.forEach((s, i) => {
    const at = PGIM_HEADER_BYTES + i * PGIM_SECTION_ENTRY_BYTES;
    out.set(new TextEncoder().encode(s.name), at);
    const data = s.data || new Uint8Array(0);
    view.setUint32(at + 16, s.rva, true);
    view.setUint32(at + 20, s.virtualSize, true);
    view.setUint32(at + 24, s.protection, true);
    view.setUint32(at + 28, s.flags || 0, true);
    view.setBigUint64(at + 32, BigInt(offsets[i]), true);
    view.setBigUint64(at + 40, BigInt(data.byteLength), true);
    view.setBigUint64(at + 48, pgimFnv1a64(data), true);
    view.setUint32(at + 56, s.characteristics || 0, true);
    view.setUint32(at + 60, 0, true);
    if (data.byteLength) out.set(data, offsets[i]);
  });
  let r = PGIM_HEADER_BYTES + sections.length * PGIM_SECTION_ENTRY_BYTES;
  for (const rva of [...dir64, ...highlow]) {
    view.setUint32(r, rva, true);
    r += 4;
  }
  return out;
}

function createMemoryOpfsStore(hooks = {}) {
  const entries = new Map();
  const reads = [];
  const writes = [];
  const deletes = [];
  return {
    entries,
    reads,
    writes,
    deletes,
    async read(key, meta) {
      reads.push({ key, meta });
      if (typeof hooks.onRead === 'function') {
        return hooks.onRead(key, meta, entries);
      }
      const val = entries.get(key);
      return val ? val.slice() : null;
    },
    async write(key, bytes, meta) {
      writes.push({ key, byteLength: bytes.byteLength, meta });
      if (typeof hooks.onWrite === 'function') {
        return hooks.onWrite(key, bytes, meta, entries);
      }
      entries.set(key, new Uint8Array(bytes).slice());
      return true;
    },
    async delete(key) {
      deletes.push(key);
      if (typeof hooks.onDelete === 'function') {
        return hooks.onDelete(key, entries);
      }
      return entries.delete(key);
    },
  };
}

function parseRangeHeader(range, total) {
  const m = /^bytes=(\d+)-(\d+)$/.exec(range || '');
  if (!m) return null;
  const start = Number(m[1]);
  const end = Math.min(Number(m[2]), total - 1);
  return { start, end };
}

function decodeMeta(store, metaKey) {
  const raw = store.entries.get(metaKey);
  if (!raw) return null;
  return JSON.parse(new TextDecoder('utf-8').decode(raw));
}

// ---------------------------------------------------------------------------------------------
// 1. ETag validator parsing & option validation
// ---------------------------------------------------------------------------------------------

test('#2080 isStrongEtag and isWeakEtag classify HTTP entity-tags strictly per RFC 9110', () => {
  assert.equal(PGIM_OPFS_PROVENANCE_SCHEMA, 1);
  assert.equal(DEFAULT_MAX_VERSION_RETRIES, 2);

  for (const strong of ['"v1"', '"a1b2-c3d4"', '"0"', '  "trimmed"  ']) {
    assert.equal(isStrongEtag(strong), true, `expected strong: ${strong}`);
    assert.equal(isWeakEtag(strong), false, `expected not weak: ${strong}`);
  }
  for (const weak of ['W/"v1"', 'w/"v1"', 'W/""', '  W/"v2" ']) {
    assert.equal(isStrongEtag(weak), false, `expected not strong: ${weak}`);
    assert.equal(isWeakEtag(weak), true, `expected weak: ${weak}`);
  }
  for (const invalid of [null, undefined, '', '   ', '""', 'unquoted', '"a" "b"', '"bad', 'bad"', 123]) {
    assert.equal(isStrongEtag(invalid), false, `expected invalid/not strong: ${String(invalid)}`);
    assert.equal(isWeakEtag(invalid), false, `expected not weak: ${String(invalid)}`);
  }

  const fetchImpl = async () => new Response(null);
  assert.throws(() => new PgimPreloader({ fetchImpl, maxVersionRetries: -1 }), RangeError);
  assert.throws(() => new PgimPreloader({ fetchImpl, maxVersionRetries: 1.5 }), RangeError);
});

// ---------------------------------------------------------------------------------------------
// 2. Version rotation between HEAD ("v1") and first GET ("v2")
// ---------------------------------------------------------------------------------------------

test('#2080 version rotation between HEAD ("v1") and first GET ("v2") via 200 OK (If-Range), 206 mismatch, and 412', async () => {
  const v1 = buildPgim(10, { textBytes: 0x3800, dataBytes: 0x2400 });
  // v2 has DIFFERENT size to verify stale HEAD Content-Length does not reject coherent 200 OK!
  const v2 = buildPgim(20, { textBytes: 0x4800, dataBytes: 0x2400 });
  assert.notEqual(v1.byteLength, v2.byteLength);

  // 2a. Standard RFC 9110 If-Range server returns 200 OK full coherent body for "v2".
  {
    const store = createMemoryOpfsStore();
    const seen = [];
    let current = { bytes: v1, etag: '"v1"' };
    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      const ifRange = init.headers?.['If-Range'] ?? null;
      seen.push({ method, range, ifRange });
      if (method === 'HEAD') {
        const res = new Response(null, {
          status: 200,
          headers: { 'content-length': String(current.bytes.byteLength), 'accept-ranges': 'bytes', etag: current.etag },
        });
        // Rotate immediately after HEAD completes.
        current = { bytes: v2, etag: '"v2"' };
        return res;
      }
      if (range && ifRange === current.etag) {
        const { start, end } = parseRangeHeader(range, current.bytes.byteLength);
        return new Response(current.bytes.slice(start, end + 1), {
          status: 206,
          headers: {
            'content-range': `bytes ${start}-${end}/${current.bytes.byteLength}`,
            'content-length': String(end - start + 1),
            etag: current.etag,
          },
        });
      }
      // If-Range mismatch -> 200 OK full coherent v2 body.
      return new Response(current.bytes.slice(), {
        status: 200,
        headers: { 'content-length': String(current.bytes.byteLength), etag: current.etag },
      });
    };

    const preloader = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096 });
    const res = await preloader.preloadGuestImage('/boot/guest.pgim', { verifyChecksums: true });
    await preloader.flushOpfsWrites();

    assert.equal(seen[1].ifRange, '"v1"', 'first Range GET must carry If-Range pinned from HEAD');
    assert.deepEqual(res.bytes, v2);
    assert.equal(res.etag, '"v2"');
    assert.equal(res.transport, 'full');
    assert.equal(res.versionVerified, true);
    const fpV1 = buildPgimOpfsFingerprint('/boot/guest.pgim', '"v1"', v1.byteLength);
    const fpV2 = buildPgimOpfsFingerprint('/boot/guest.pgim', '"v2"', v2.byteLength);
    assert.equal(store.entries.has(fpV1.dataKey), false, 'must never cache v1');
    assert.equal(store.entries.has(fpV2.dataKey), true, 'must cache coherent v2');
    assert.deepEqual(store.entries.get(fpV2.dataKey), v2);
    assert.equal(decodeMeta(store, fpV2.metaKey).provenance.verified, true);
  }

  // 2b. Server ignores If-Range and returns 206 with "v2" after HEAD returned "v1" -> clean retry succeeds with "v2".
  {
    const store = createMemoryOpfsStore();
    const v2SameSize = buildPgim(21, { textBytes: 0x3800, dataBytes: 0x2400 });
    assert.equal(v1.byteLength, v2SameSize.byteLength);
    let current = { bytes: v1, etag: '"v1"' };
    let headCount = 0;
    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      if (method === 'HEAD') {
        headCount += 1;
        const res = new Response(null, {
          status: 200,
          headers: { 'content-length': String(current.bytes.byteLength), 'accept-ranges': 'bytes', etag: current.etag },
        });
        if (headCount === 1) current = { bytes: v2SameSize, etag: '"v2"' };
        return res;
      }
      const { start, end } = parseRangeHeader(range, current.bytes.byteLength);
      return new Response(current.bytes.slice(start, end + 1), {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${current.bytes.byteLength}`,
          'content-length': String(end - start + 1),
          etag: current.etag,
        },
      });
    };

    const preloader = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096, concurrency: 2 });
    const res = await preloader.preloadGuestImage('/boot/guest.pgim');
    await preloader.flushOpfsWrites();

    assert.equal(headCount, 2, 'must re-probe HEAD cleanly on HEAD->GET ETag mismatch');
    assert.deepEqual(res.bytes, v2SameSize);
    assert.equal(res.etag, '"v2"');
    assert.equal(res.transport, 'range');
    assert.equal(res.versionVerified, true);
    assert.equal(preloader.getStats().retryCount, 1);
    const fpV1 = buildPgimOpfsFingerprint('/boot/guest.pgim', '"v1"', v1.byteLength);
    const fpV2 = buildPgimOpfsFingerprint('/boot/guest.pgim', '"v2"', v2SameSize.byteLength);
    assert.equal(store.entries.has(fpV1.dataKey), false);
    assert.deepEqual(store.entries.get(fpV2.dataKey), v2SameSize);
  }

  // 2c. Server returns 412 Precondition Failed on first GET -> clean retry succeeds with "v2".
  {
    const store = createMemoryOpfsStore();
    let current = { bytes: v1, etag: '"v1"' };
    let headCount = 0;
    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      const ifRange = init.headers?.['If-Range'] ?? null;
      if (method === 'HEAD') {
        headCount += 1;
        const res = new Response(null, {
          status: 200,
          headers: { 'content-length': String(current.bytes.byteLength), 'accept-ranges': 'bytes', etag: current.etag },
        });
        if (headCount === 1) current = { bytes: v2, etag: '"v2"' };
        return res;
      }
      if (ifRange && ifRange !== current.etag) {
        return new Response(null, { status: 412, headers: { etag: current.etag } });
      }
      const { start, end } = parseRangeHeader(range, current.bytes.byteLength);
      return new Response(current.bytes.slice(start, end + 1), {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${current.bytes.byteLength}`,
          'content-length': String(end - start + 1),
          etag: current.etag,
        },
      });
    };

    const preloader = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096, concurrency: 2 });
    const res = await preloader.preloadGuestImage('/boot/guest.pgim');
    await preloader.flushOpfsWrites();

    assert.equal(headCount, 2);
    assert.deepEqual(res.bytes, v2);
    assert.equal(res.etag, '"v2"');
    assert.equal(res.transport, 'range');
    assert.equal(preloader.getStats().retryCount, 1);
    const fpV2 = buildPgimOpfsFingerprint('/boot/guest.pgim', '"v2"', v2.byteLength);
    assert.deepEqual(store.entries.get(fpV2.dataKey), v2);
  }
});

// ---------------------------------------------------------------------------------------------
// 3. Mid-download same-size rotation (exact #2080 repro: chunk 0 "v1", chunk 1 "v2")
// ---------------------------------------------------------------------------------------------

test('#2080 mid-download same-size rotation (chunk 0 "v1", chunk 1 "v2") is rejected, retried cleanly, and caches only coherent "v2"', async () => {
  const v1 = buildPgim(31, { textBytes: 0x3800, dataBytes: 0x2400 });
  const v2 = buildPgim(32, { textBytes: 0x3800, dataBytes: 0x2400 });
  assert.equal(v1.byteLength, v2.byteLength, 'same byteLength reproduces #2080 silent mixing');
  assert.notDeepEqual(v1, v2);

  // 3a. Server ignores If-Range and serves chunk 0 as "v1", chunk 1+ as "v2" (with verifyChecksums: false!).
  {
    const store = createMemoryOpfsStore();
    const seenIfRange = [];
    let current = { bytes: v1, etag: '"v1"' };
    let rotated = false;

    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      const ifRange = init.headers?.['If-Range'] ?? null;
      if (method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: { 'content-length': String(current.bytes.byteLength), 'accept-ranges': 'bytes', etag: current.etag },
        });
      }
      seenIfRange.push({ range, ifRange });
      const { start, end } = parseRangeHeader(range, current.bytes.byteLength);
      const res = new Response(current.bytes.slice(start, end + 1), {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${current.bytes.byteLength}`,
          'content-length': String(end - start + 1),
          etag: current.etag,
        },
      });
      // Rotate right after chunk 0 of v1 is served!
      if (!rotated && start === 0) {
        rotated = true;
        current = { bytes: v2, etag: '"v2"' };
      }
      return res;
    };

    const preloader = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096, concurrency: 2 });
    const res = await preloader.preloadGuestImage('/boot/guest.pgim', { verifyChecksums: false });
    await preloader.flushOpfsWrites();

    assert.deepEqual(res.bytes, v2, 'must return pure v2 bytes, never mixed v1+v2');
    assert.equal(res.etag, '"v2"');
    assert.equal(res.transport, 'range');
    assert.equal(res.versionVerified, true);
    assert.equal(res.versionProvenance, 'strong-etag-range');
    assert.ok(seenIfRange.some((r) => r.ifRange === '"v1"'), 'initial chunks must carry If-Range: "v1"');
    assert.ok(seenIfRange.some((r) => r.ifRange === '"v2"'), 'retried chunks must carry If-Range: "v2"');

    const fpV1 = buildPgimOpfsFingerprint('/boot/guest.pgim', '"v1"', v1.byteLength);
    const fpV2 = buildPgimOpfsFingerprint('/boot/guest.pgim', '"v2"', v2.byteLength);
    assert.equal(store.entries.has(fpV1.dataKey), false, 'v1 fingerprint must never be written');
    assert.equal(store.entries.has(fpV2.dataKey), true, 'only coherent v2 must be cached');
    assert.deepEqual(store.entries.get(fpV2.dataKey), v2);
    assert.equal(store.writes.length, 2, 'exactly 1 data + 1 meta write after final verified assembly');
    assert.equal(preloader.getStats().bytesLoaded, v2.byteLength);
  }

  // 3b. Standard If-Range server returns 200 OK full coherent v2 mid-download with a signal-bound ReadableStream
  //     (just like real browser/undici fetch): sibling chunks are aborted while the winning 200 OK stream stays intact!
  {
    const store = createMemoryOpfsStore();
    let current = { bytes: v1, etag: '"v1"' };
    let rotated = false;
    let siblingAborted = false;

    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      const ifRange = init.headers?.['If-Range'] ?? null;
      if (method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: { 'content-length': String(current.bytes.byteLength), 'accept-ranges': 'bytes', etag: current.etag },
        });
      }
      if (ifRange && ifRange !== current.etag) {
        const snapshot = current.bytes.slice();
        const reqSignal = init.signal;
        if (range === 'bytes=4096-8191') {
          // Winning 200 OK stream tied to init.signal: errors immediately if its own signal is aborted!
          let pos = 0;
          const stream = new ReadableStream({
            async pull(controller) {
              if (reqSignal?.aborted) {
                controller.error(reqSignal.reason || new DOMException('Aborted', 'AbortError'));
                return;
              }
              await new Promise((r) => setTimeout(r, 2));
              if (reqSignal?.aborted) {
                controller.error(reqSignal.reason || new DOMException('Aborted', 'AbortError'));
                return;
              }
              if (pos >= snapshot.byteLength) {
                controller.close();
                return;
              }
              const nextEnd = Math.min(snapshot.byteLength, pos + 4096);
              controller.enqueue(snapshot.subarray(pos, nextEnd));
              pos = nextEnd;
            },
          });
          return new Response(stream, {
            status: 200,
            headers: { 'content-length': String(snapshot.byteLength), etag: current.etag },
          });
        }
        // Sibling chunk hangs until aborted by the winning 200 OK worker.
        return new Promise((_resolve, reject) => {
          const onAbort = () => {
            siblingAborted = true;
            reject(reqSignal.reason || new DOMException('Aborted', 'AbortError'));
          };
          if (reqSignal?.aborted) onAbort();
          else reqSignal?.addEventListener('abort', onAbort, { once: true });
        });
      }
      const { start, end } = parseRangeHeader(range, current.bytes.byteLength);
      const res = new Response(current.bytes.slice(start, end + 1), {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${current.bytes.byteLength}`,
          'content-length': String(end - start + 1),
          etag: current.etag,
        },
      });
      if (!rotated && start === 0) {
        rotated = true;
        current = { bytes: v2, etag: '"v2"' };
      }
      return res;
    };

    const preloader = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096, concurrency: 3 });
    const res = await preloader.preloadGuestImage('/boot/guest.pgim', { verifyChecksums: false });
    await preloader.flushOpfsWrites();

    assert.equal(siblingAborted, true, 'sibling parallel chunks must be aborted when winning 200 OK fallback arrives');
    assert.deepEqual(res.bytes, v2);
    assert.equal(res.etag, '"v2"');
    assert.equal(res.transport, 'full');
    assert.equal(res.versionVerified, true);
    const fpV1 = buildPgimOpfsFingerprint('/boot/guest.pgim', '"v1"', v1.byteLength);
    const fpV2 = buildPgimOpfsFingerprint('/boot/guest.pgim', '"v2"', v2.byteLength);
    assert.equal(store.entries.has(fpV1.dataKey), false);
    assert.deepEqual(store.entries.get(fpV2.dataKey), v2);
  }

  // 3c. Server responds 412 Precondition Failed mid-download -> discards v1, retries cleanly, caches v2.
  {
    const store = createMemoryOpfsStore();
    let current = { bytes: v1, etag: '"v1"' };
    let rotated = false;

    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      const ifRange = init.headers?.['If-Range'] ?? null;
      if (method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: { 'content-length': String(current.bytes.byteLength), 'accept-ranges': 'bytes', etag: current.etag },
        });
      }
      if (ifRange && ifRange !== current.etag) {
        return new Response(null, { status: 412, headers: { etag: current.etag } });
      }
      const { start, end } = parseRangeHeader(range, current.bytes.byteLength);
      const res = new Response(current.bytes.slice(start, end + 1), {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${current.bytes.byteLength}`,
          'content-length': String(end - start + 1),
          etag: current.etag,
        },
      });
      if (!rotated && start === 0) {
        rotated = true;
        current = { bytes: v2, etag: '"v2"' };
      }
      return res;
    };

    const preloader = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096, concurrency: 2 });
    const res = await preloader.preloadGuestImage('/boot/guest.pgim', { verifyChecksums: false });
    await preloader.flushOpfsWrites();

    assert.deepEqual(res.bytes, v2);
    assert.equal(res.etag, '"v2"');
    assert.equal(res.transport, 'range');
    assert.equal(preloader.getStats().retryCount, 1);
    const fpV2 = buildPgimOpfsFingerprint('/boot/guest.pgim', '"v2"', v2.byteLength);
    assert.deepEqual(store.entries.get(fpV2.dataKey), v2);
  }
});

// ---------------------------------------------------------------------------------------------
// 4. Persistent rotation exhausting retry cap -> rejected cleanly, zero cache writes
// ---------------------------------------------------------------------------------------------

test('#2080 persistent version rotation exhausts retry cap cleanly with zero OPFS writes', async () => {
  const versions = [buildPgim(40), buildPgim(41), buildPgim(42), buildPgim(43), buildPgim(44)];
  const store = createMemoryOpfsStore();
  let versionIdx = 0;

  const fetchImpl = async (url, init = {}) => {
    const method = init.method || 'GET';
    const range = init.headers?.Range ?? null;
    const cur = versions[Math.min(versionIdx, versions.length - 1)];
    const etag = `"v${versionIdx + 1}"`;
    if (method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: { 'content-length': String(cur.byteLength), 'accept-ranges': 'bytes', etag },
      });
    }
    const { start, end } = parseRangeHeader(range, cur.byteLength);
    if (start === 0) {
      // Serve chunk 0 of current version, then immediately rotate so chunk 1 mismatches!
      const res = new Response(cur.slice(start, end + 1), {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${cur.byteLength}`,
          'content-length': String(end - start + 1),
          etag,
        },
      });
      versionIdx += 1;
      return res;
    }
    const next = versions[Math.min(versionIdx, versions.length - 1)];
    const nextEtag = `"v${versionIdx + 1}"`;
    return new Response(next.slice(start, end + 1), {
      status: 206,
      headers: {
        'content-range': `bytes ${start}-${end}/${next.byteLength}`,
        'content-length': String(end - start + 1),
        etag: nextEtag,
      },
    });
  };

  const preloader = new PgimPreloader({
    fetchImpl,
    opfsCache: store,
    chunkSize: 4096,
    concurrency: 1,
    maxVersionRetries: 2,
  });

  await assert.rejects(
    preloader.preloadGuestImage('/boot/guest.pgim'),
    (err) =>
      err instanceof PgimPreloadError &&
      err.code === 'VERSION_MISMATCH' &&
      /exhausted 2 version retry attempts/.test(err.message)
  );
  await preloader.flushOpfsWrites();

  assert.equal(store.writes.length, 0, 'zero OPFS writes allowed when retries exhaust');
  assert.equal(store.entries.size, 0);
  assert.equal(preloader.getStats().retryCount, 2);
  assert.equal(preloader.getStats().opfsWrites, 0);
});

// ---------------------------------------------------------------------------------------------
// 5. Missing or weak (W/"v1") validators -> coherent single-response fallback
// ---------------------------------------------------------------------------------------------

test('#2080 missing or weak validators fall back to coherent single-response full GET (never stitch unverified multi-chunk ranges)', async () => {
  const v1 = buildPgim(50);
  const v2 = buildPgim(51);
  assert.equal(v1.byteLength, v2.byteLength);

  // 5a. Weak ETag on HEAD (W/"v1") -> skips multi-chunk range stitching immediately, never sends weak If-Range, never caches under weak ETag.
  {
    const store = createMemoryOpfsStore();
    const log = [];
    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      const ifRange = init.headers?.['If-Range'] ?? null;
      log.push({ method, range, ifRange });
      if (method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: { 'content-length': String(v1.byteLength), 'accept-ranges': 'bytes', etag: 'W/"v1"' },
        });
      }
      assert.equal(range, null, 'weak ETag on HEAD must fall back directly to full GET without Range');
      assert.equal(ifRange, null, 'must never send weak ETag in If-Range');
      return new Response(v1.slice(), {
        status: 200,
        headers: { 'content-length': String(v1.byteLength), etag: 'W/"v1"' },
      });
    };

    const preloader = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096, concurrency: 3 });
    const res = await preloader.preloadGuestImage('/boot/weak.pgim');
    await preloader.flushOpfsWrites();

    assert.deepEqual(res.bytes, v1);
    assert.equal(res.transport, 'full');
    assert.equal(res.etag, 'W/"v1"');
    assert.equal(res.versionVerified, true);
    assert.equal(res.validatorStrength, 'weak');
    assert.equal(res.versionProvenance, 'single-response');
    assert.equal(store.writes.length, 0, 'weak ETag entry must not be cached in OPFS');
    assert.deepEqual(
      log.map((r) => [r.method, r.range]),
      [
        ['HEAD', null],
        ['GET', null],
      ]
    );
  }

  // 5b. Missing ETag on HEAD + 206 -> detects multi-chunk lacks strong validator, discards chunk 0, fetches coherent full GET!
  {
    const store = createMemoryOpfsStore();
    const log = [];
    let current = v1;
    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      log.push({ method, range });
      if (method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: { 'content-length': String(current.byteLength), 'accept-ranges': 'bytes' },
        });
      }
      if (range) {
        const { start, end } = parseRangeHeader(range, current.byteLength);
        const res = new Response(current.slice(start, end + 1), {
          status: 206,
          headers: {
            'content-range': `bytes ${start}-${end}/${current.byteLength}`,
            'content-length': String(end - start + 1),
          },
        });
        // Rotate right after chunk 0: if preloader naively stitched chunk 1, it would mix v1+v2!
        current = v2;
        return res;
      }
      return new Response(current.slice(), {
        status: 200,
        headers: { 'content-length': String(current.byteLength) },
      });
    };

    const preloader = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096, concurrency: 3 });
    const res = await preloader.preloadGuestImage('/boot/no-etag.pgim', { verifyChecksums: true });
    await preloader.flushOpfsWrites();

    assert.deepEqual(res.bytes, v2, 'coherent full GET fallback must return pure v2 without mixing chunk 0 v1');
    assert.equal(res.transport, 'full');
    assert.equal(res.versionVerified, true);
    assert.equal(res.validatorStrength, 'none');
    assert.equal(
      log.filter((r) => r.method === 'GET' && r.range !== null).length,
      1,
      'must stop ranged fetching after chunk 0 reveals missing strong ETag'
    );
    assert.equal(log.filter((r) => r.method === 'GET' && r.range === null).length, 1);
  }

  // 5c. HEAD 405 + chunk 0 returns weak ETag W/"v1" -> discards chunk 0, falls back to coherent full GET.
  {
    const log = [];
    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      log.push({ method, range });
      if (method === 'HEAD') return new Response(null, { status: 405 });
      if (range) {
        const { start, end } = parseRangeHeader(range, v1.byteLength);
        return new Response(v1.slice(start, end + 1), {
          status: 206,
          headers: {
            'content-range': `bytes ${start}-${end}/${v1.byteLength}`,
            'content-length': String(end - start + 1),
            etag: 'W/"v1"',
          },
        });
      }
      return new Response(v1.slice(), {
        status: 200,
        headers: { 'content-length': String(v1.byteLength), etag: 'W/"v1"' },
      });
    };

    const preloader = new PgimPreloader({ fetchImpl, chunkSize: 4096, concurrency: 2 });
    const res = await preloader.preloadGuestImage('/boot/weak-405.pgim');
    assert.deepEqual(res.bytes, v1);
    assert.equal(res.transport, 'full');
    assert.equal(log.filter((r) => r.method === 'GET' && r.range === null).length, 1);
  }

  // 5d. HEAD omits ETag, chunk 0 returns STRONG ETag -> establishes strong validator from chunk 0 and continues ranged!
  {
    const seenIfRange = [];
    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      const ifRange = init.headers?.['If-Range'] ?? null;
      if (method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: { 'content-length': String(v1.byteLength), 'accept-ranges': 'bytes' },
        });
      }
      seenIfRange.push({ range, ifRange });
      const { start, end } = parseRangeHeader(range, v1.byteLength);
      return new Response(v1.slice(start, end + 1), {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${v1.byteLength}`,
          'content-length': String(end - start + 1),
          etag: '"strong-from-206"',
        },
      });
    };

    const preloader = new PgimPreloader({ fetchImpl, chunkSize: 4096, concurrency: 2 });
    const res = await preloader.preloadGuestImage('/boot/omit-head-etag.pgim');
    assert.deepEqual(res.bytes, v1);
    assert.equal(res.transport, 'range');
    assert.equal(res.etag, '"strong-from-206"');
    assert.equal(res.versionProvenance, 'strong-etag-range');
    assert.equal(seenIfRange[0].ifRange, null);
    assert.ok(seenIfRange.slice(1).every((r) => r.ifRange === '"strong-from-206"'));
  }

  // 5e. Single-chunk PGIM without ETag -> 1 coherent 206 response without unnecessary extra GET.
  {
    const log = [];
    const fetchImpl = async (url, init = {}) => {
      const method = init.method || 'GET';
      const range = init.headers?.Range ?? null;
      log.push({ method, range });
      if (method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: { 'content-length': String(v1.byteLength), 'accept-ranges': 'bytes' },
        });
      }
      const { start, end } = parseRangeHeader(range, v1.byteLength);
      return new Response(v1.slice(start, end + 1), {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${v1.byteLength}`,
          'content-length': String(end - start + 1),
        },
      });
    };

    const preloader = new PgimPreloader({ fetchImpl, chunkSize: v1.byteLength + 4096 });
    const res = await preloader.preloadGuestImage('/boot/single-chunk.pgim');
    assert.deepEqual(res.bytes, v1);
    assert.equal(res.transport, 'range');
    assert.equal(res.chunksFetched, 1);
    assert.equal(res.versionVerified, true);
    assert.equal(res.versionProvenance, 'single-response-range');
    assert.equal(log.filter((r) => r.method === 'GET').length, 1);
  }
});

// ---------------------------------------------------------------------------------------------
// 6. Mismatched Content-Range total length, chunk byte length, or missing chunk ETag
// ---------------------------------------------------------------------------------------------

test('#2080 rejects mismatched Content-Range total, truncated chunk body, and missing chunk ETag without caching', async () => {
  const v1 = buildPgim(60);

  // 6a. Persistent mismatched Content-Range total on chunk 1 -> rejected cleanly, zero cache writes.
  {
    const store = createMemoryOpfsStore();
    const fetchImpl = async (url, init = {}) => {
      if (init.method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: { 'content-length': String(v1.byteLength), 'accept-ranges': 'bytes', etag: '"v1"' },
        });
      }
      const { start, end } = parseRangeHeader(init.headers?.Range, v1.byteLength);
      const bogusTotal = start === 0 ? v1.byteLength : v1.byteLength + 8192;
      return new Response(v1.slice(start, end + 1), {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${bogusTotal}`,
          'content-length': String(end - start + 1),
          etag: '"v1"',
        },
      });
    };

    const preloader = new PgimPreloader({
      fetchImpl,
      opfsCache: store,
      chunkSize: 4096,
      concurrency: 1,
      maxVersionRetries: 1,
    });
    await assert.rejects(
      preloader.preloadGuestImage('/boot/bad-cr-total.pgim'),
      (err) => err instanceof PgimPreloadError && err.code === 'RANGE_MISMATCH'
    );
    await preloader.flushOpfsWrites();
    assert.equal(store.writes.length, 0);
  }

  // 6b. Chunk 1 returns truncated actual body length despite matching headers -> rejected with RANGE_MISMATCH, zero cache writes.
  {
    const store = createMemoryOpfsStore();
    const fetchImpl = async (url, init = {}) => {
      if (init.method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: { 'content-length': String(v1.byteLength), 'accept-ranges': 'bytes', etag: '"v1"' },
        });
      }
      const { start, end } = parseRangeHeader(init.headers?.Range, v1.byteLength);
      const slice = start === 0 ? v1.slice(start, end + 1) : v1.slice(start, end - 15);
      return new Response(slice, {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${v1.byteLength}`,
          etag: '"v1"',
        },
      });
    };

    const preloader = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096, concurrency: 1 });
    await assert.rejects(
      preloader.preloadGuestImage('/boot/short-chunk.pgim'),
      (err) => err instanceof PgimPreloadError && err.code === 'RANGE_MISMATCH'
    );
    await preloader.flushOpfsWrites();
    assert.equal(store.writes.length, 0);
  }

  // 6c. Chunk 1 omits ETag after pinning strong ETag '"v1"' -> rejected with VERSION_MISMATCH, zero cache writes.
  {
    const store = createMemoryOpfsStore();
    const fetchImpl = async (url, init = {}) => {
      if (init.method === 'HEAD') {
        return new Response(null, {
          status: 200,
          headers: { 'content-length': String(v1.byteLength), 'accept-ranges': 'bytes', etag: '"v1"' },
        });
      }
      const { start, end } = parseRangeHeader(init.headers?.Range, v1.byteLength);
      const headers = {
        'content-range': `bytes ${start}-${end}/${v1.byteLength}`,
        'content-length': String(end - start + 1),
      };
      if (start === 0) headers.etag = '"v1"';
      return new Response(v1.slice(start, end + 1), { status: 206, headers });
    };

    const preloader = new PgimPreloader({
      fetchImpl,
      opfsCache: store,
      chunkSize: 4096,
      concurrency: 1,
      maxVersionRetries: 1,
    });
    await assert.rejects(
      preloader.preloadGuestImage('/boot/missing-chunk-etag.pgim'),
      (err) => err instanceof PgimPreloadError && err.code === 'VERSION_MISMATCH'
    );
    await preloader.flushOpfsWrites();
    assert.equal(store.writes.length, 0);
  }
});

// ---------------------------------------------------------------------------------------------
// 7. Cancellation (AbortSignal) mid-download -> aborts cleanly, zero cache writes
// ---------------------------------------------------------------------------------------------

test('#2080 AbortSignal cancellation mid-download aborts cleanly with zero OPFS writes', async () => {
  const v1 = buildPgim(70);
  const store = createMemoryOpfsStore();
  const controller = new AbortController();

  const fetchImpl = async (url, init = {}) => {
    if (init.signal?.aborted) throw init.signal.reason;
    if (init.method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: { 'content-length': String(v1.byteLength), 'accept-ranges': 'bytes', etag: '"cancel-v1"' },
      });
    }
    const { start, end } = parseRangeHeader(init.headers?.Range, v1.byteLength);
    if (start > 0) {
      controller.abort();
      throw init.signal.reason;
    }
    return new Response(v1.slice(start, end + 1), {
      status: 206,
      headers: {
        'content-range': `bytes ${start}-${end}/${v1.byteLength}`,
        'content-length': String(end - start + 1),
        etag: '"cancel-v1"',
      },
    });
  };

  const preloader = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096, concurrency: 2 });
  await assert.rejects(
    preloader.preloadGuestImage('/boot/cancel.pgim', { signal: controller.signal }),
    (err) => err.name === 'AbortError'
  );
  await preloader.flushOpfsWrites();

  assert.equal(store.writes.length, 0, 'cancelled download must never write to OPFS');
  assert.equal(store.entries.size, 0);
  assert.equal(preloader.getStats().state, 'aborted');
  assert.equal(preloader.getStats().opfsWrites, 0);
});

// ---------------------------------------------------------------------------------------------
// 8. Failed cache writes -> non-fatal and leave zero partial/corrupt entries
// ---------------------------------------------------------------------------------------------

test('#2080 failed OPFS data or metadata writes are non-fatal and clean up partial entries', async () => {
  const v1 = buildPgim(80);
  const makeServer = () => async (url, init = {}) => {
    if (init.method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: { 'content-length': String(v1.byteLength), 'accept-ranges': 'bytes', etag: '"cache-v1"' },
      });
    }
    const { start, end } = parseRangeHeader(init.headers?.Range, v1.byteLength);
    return new Response(v1.slice(start, end + 1), {
      status: 206,
      headers: {
        'content-range': `bytes ${start}-${end}/${v1.byteLength}`,
        'content-length': String(end - start + 1),
        etag: '"cache-v1"',
      },
    });
  };
  const fp = buildPgimOpfsFingerprint('/boot/fail-write.pgim', '"cache-v1"', v1.byteLength, PGIM_OPFS_CACHE_VERSION);

  // 8a. Partial dataKey write throws midway -> dataKey is deleted, metaKey never written, preload succeeds!
  {
    const store = createMemoryOpfsStore({
      async onWrite(key, bytes, meta, entries) {
        if (key === fp.dataKey) {
          // Simulate partial file left before write failure.
          entries.set(key, bytes.slice(0, 64));
          throw new DOMException('Disk full mid-write', 'QuotaExceededError');
        }
        entries.set(key, bytes.slice());
        return true;
      },
    });
    const preloader = new PgimPreloader({ fetchImpl: makeServer(), opfsCache: store, chunkSize: 4096 });
    const res = await preloader.preloadGuestImage('/boot/fail-write.pgim');
    await preloader.flushOpfsWrites();

    assert.deepEqual(res.bytes, v1);
    assert.equal(store.entries.size, 0, 'partial dataKey must be deleted after failed write');
    assert.ok(store.deletes.includes(fp.dataKey));
    assert.equal(preloader.getStats().opfsWrites, 0);
  }

  // 8b. metaKey write throws after dataKey succeeded -> both dataKey and metaKey are cleaned up!
  {
    const store = createMemoryOpfsStore({
      async onWrite(key, bytes, meta, entries) {
        if (key === fp.metaKey) {
          entries.set(key, new TextEncoder().encode('{partial'));
          throw new Error('I/O failure writing meta');
        }
        entries.set(key, bytes.slice());
        return true;
      },
    });
    const preloader = new PgimPreloader({ fetchImpl: makeServer(), opfsCache: store, chunkSize: 4096 });
    const res = await preloader.preloadGuestImage('/boot/fail-write.pgim');
    await preloader.flushOpfsWrites();

    assert.deepEqual(res.bytes, v1);
    assert.equal(store.entries.size, 0, 'failed metaKey write must leave neither dataKey nor corrupt metaKey');
    assert.ok(store.deletes.includes(fp.dataKey));
    assert.ok(store.deletes.includes(fp.metaKey));
    assert.equal(preloader.getStats().opfsWrites, 0);
  }

  // 8c. metaKey write returns { stored: false } -> cleans up dataKey, does not count opfsWrite!
  {
    const store = createMemoryOpfsStore({
      async onWrite(key, bytes, meta, entries) {
        if (key === fp.metaKey) return { stored: false };
        entries.set(key, bytes.slice());
        return true;
      },
    });
    const preloader = new PgimPreloader({ fetchImpl: makeServer(), opfsCache: store, chunkSize: 4096 });
    const res = await preloader.preloadGuestImage('/boot/fail-write.pgim');
    await preloader.flushOpfsWrites();

    assert.deepEqual(res.bytes, v1);
    assert.equal(store.entries.size, 0);
    assert.equal(preloader.getStats().opfsWrites, 0);
  }
});

// ---------------------------------------------------------------------------------------------
// 9. Legacy unverified OPFS entries & injected caches
// ---------------------------------------------------------------------------------------------

test('#2080 legacy unverified OPFS entries and unverified injected caches are rejected and replaced with verified entries', async () => {
  const v1 = buildPgim(90);
  const v2 = buildPgim(91);
  assert.equal(v1.byteLength, v2.byteLength);

  // Build a mixed-version PGIM (chunk 0 from v1, rest from v2) — structurally valid header, corrupt payload!
  const mixed = v1.slice();
  mixed.set(v2.subarray(4096), 4096);

  const fpV1 = buildPgimOpfsFingerprint('/boot/legacy.pgim', '"v1"', v1.byteLength, PGIM_OPFS_CACHE_VERSION);
  const fetchImpl = async (url, init = {}) => {
    if (init.method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: { 'content-length': String(v1.byteLength), 'accept-ranges': 'bytes', etag: '"v1"' },
      });
    }
    const { start, end } = parseRangeHeader(init.headers?.Range, v1.byteLength);
    return new Response(v1.slice(start, end + 1), {
      status: 206,
      headers: {
        'content-range': `bytes ${start}-${end}/${v1.byteLength}`,
        'content-length': String(end - start + 1),
        etag: '"v1"',
      },
    });
  };

  // 9a. Legacy OPFS metadata (no provenance field) containing mixed-version bytes -> evicted, refetched, tagged!
  {
    const store = createMemoryOpfsStore();
    store.entries.set(fpV1.dataKey, mixed.slice());
    store.entries.set(
      fpV1.metaKey,
      new TextEncoder().encode(
        JSON.stringify({
          version: PGIM_OPFS_CACHE_VERSION,
          urlPath: fpV1.urlPath,
          etag: '"v1"',
          lastModified: null,
          byteLength: v1.byteLength,
          fingerprint: fpV1.fingerprint,
          dataKey: fpV1.dataKey,
        })
      )
    );

    const p1 = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096 });
    const r1 = await p1.preloadGuestImage('/boot/legacy.pgim', { verifyChecksums: false });
    await p1.flushOpfsWrites();

    assert.equal(r1.opfsCacheHit, false, 'legacy unverified OPFS entry must not be trusted');
    assert.deepEqual(r1.bytes, v1, 'must refetch clean coherent v1 instead of serving mixed legacy bytes');
    assert.equal(p1.getStats().opfsInvalidations, 1);
    assert.equal(p1.getStats().opfsWrites, 1);
    const meta = decodeMeta(store, fpV1.metaKey);
    assert.deepEqual(meta.provenance, {
      schema: PGIM_OPFS_PROVENANCE_SCHEMA,
      verified: true,
      source: 'strong-etag-range',
      validatorStrength: 'strong',
    });

    // Next boot hits the newly verified tagged OPFS entry with 0 GETs!
    let getsOnWarm = 0;
    const p2 = new PgimPreloader({
      fetchImpl: async (url, init) => {
        if ((init?.method || 'GET') === 'GET') getsOnWarm += 1;
        return fetchImpl(url, init);
      },
      opfsCache: store,
      chunkSize: 4096,
    });
    const r2 = await p2.preloadGuestImage('/boot/legacy.pgim');
    assert.equal(r2.opfsCacheHit, true);
    assert.equal(r2.versionVerified, true);
    assert.equal(r2.versionProvenance, 'opfs-cache');
    assert.deepEqual(r2.bytes, v1);
    assert.equal(getsOnWarm, 0);
  }

  // 9b. Injected opfsCache pre-populated with raw dataKey only (no verified metaKey) -> not trusted, replaced cleanly!
  {
    const store = createMemoryOpfsStore();
    store.entries.set(fpV1.dataKey, mixed.slice());

    const p1 = new PgimPreloader({ fetchImpl, opfsCache: store, chunkSize: 4096 });
    const r1 = await p1.preloadGuestImage('/boot/legacy.pgim', { verifyChecksums: false });
    await p1.flushOpfsWrites();

    assert.equal(r1.opfsCacheHit, false, 'meta-less raw injected cache entry must not bypass provenance');
    assert.deepEqual(r1.bytes, v1);
    assert.deepEqual(store.entries.get(fpV1.dataKey), v1);
    assert.equal(decodeMeta(store, fpV1.metaKey).provenance.verified, true);
  }

  // 9c. Injected rangeCache without verifyChecksums is marked unverified and NOT persisted to OPFS;
  //     with verifyChecksums: true, checksum verification upgrades provenance and allows caching.
  {
    const store = createMemoryOpfsStore();
    const rangeCache = {
      async read(path, offset, length) {
        return v1.slice(offset, Math.min(v1.byteLength, offset + length));
      },
    };
    const pUnverified = new PgimPreloader({
      fetchImpl,
      rangeCache,
      opfsCache: store,
      chunkSize: 4096,
    });
    const rUnverified = await pUnverified.preloadGuestImage('/boot/legacy.pgim', { verifyChecksums: false });
    await pUnverified.flushOpfsWrites();

    assert.equal(rUnverified.transport, 'range-cache');
    assert.equal(rUnverified.versionVerified, false);
    assert.equal(rUnverified.versionProvenance, 'range-cache-unverified');
    assert.equal(store.writes.length, 0, 'unverified multi-chunk rangeCache output must not be written to OPFS');

    const rVerified = await pUnverified.preloadGuestImage('/boot/legacy.pgim', { verifyChecksums: true });
    await pUnverified.flushOpfsWrites();
    assert.equal(rVerified.versionVerified, true);
    assert.equal(rVerified.versionProvenance, 'checksum-verified-range-cache');
    assert.equal(store.writes.length, 2, 'checksum-verified rangeCache output may be persisted with provenance');
  }
});
