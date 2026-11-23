import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const rangeCacheMod = require('../web/workers/range_cache.js');
const userdataMod = require('../web/workers/userdata.js');

const { RangeCache, DEFAULT_CHUNK_SIZE, openOpfsCacheDir } = rangeCacheMod;
const { UserdataStore, DEFAULT_DB_NAME, DEFAULT_STORE_NAME, validateUserdataPath } = userdataMod;

function makePatternBytes(length, seed = 17) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = (i * 31 + seed) & 0xff;
  }
  return out;
}

function createMockRangeFetch(filesMap, options = {}) {
  const calls = [];
  let inFlight = 0;
  let maxConcurrent = 0;
  const delayMs = options.delayMs || 0;

  async function mockFetch(url, init = {}) {
    inFlight += 1;
    if (inFlight > maxConcurrent) {
      maxConcurrent = inFlight;
    }

    const rangeHeader = (init.headers && (init.headers.Range || init.headers.range)) || null;
    calls.push({ url, rangeHeader });

    try {
      if (delayMs > 0) {
        await new Promise((r) => setTimeout(r, delayMs));
      }

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
          async json() {
            throw new Error('404');
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
      const headers = new Map([
        ['Content-Range', `bytes ${reqStart}-${actualEnd}/${data.byteLength}`],
        ['Content-Length', String(slice.byteLength)],
      ]);

      return {
        ok: true,
        status: 206,
        headers,
        async arrayBuffer() {
          return slice.buffer;
        },
      };
    } finally {
      inFlight -= 1;
    }
  }

  return {
    fetchImpl: mockFetch,
    calls,
    getMaxConcurrent: () => maxConcurrent,
  };
}

function createMockOpfsDir() {
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
          const current = files.get(name);
          return {
            async arrayBuffer() {
              return current.slice().buffer;
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
          };
        },
      };
    },
  };
}

function createMockIndexedDB(options = {}) {
  const databases = new Map();

  function schedule(fn) {
    queueMicrotask(fn);
  }

  function makeStore(recordsMap) {
    return {
      put(record) {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        schedule(() => {
          const clone = {
            ...record,
            data: record.data instanceof Uint8Array ? record.data.slice() : record.data,
          };
          recordsMap.set(record.path, clone);
          req.result = record.path;
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
      delete(key) {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        schedule(() => {
          recordsMap.delete(key);
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
      getAll() {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        if (options.hangGetAll) {
          return req;
        }
        schedule(() => {
          req.result = Array.from(recordsMap.values()).map((r) => ({
            ...r,
            data: r.data instanceof Uint8Array ? r.data.slice() : r.data,
          }));
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
    };
  }

  return {
    databases,
    open(dbName) {
      const req = {
        result: null,
        error: null,
        onupgradeneeded: null,
        onsuccess: null,
        onerror: null,
      };

      if (options.hangOpen) {
        return req;
      }

      schedule(() => {
        let entry = databases.get(dbName);
        const isNew = !entry;
        if (isNew) {
          entry = { stores: new Map() };
          databases.set(dbName, entry);
        }

        const db = {
          name: dbName,
          objectStoreNames: {
            contains: (name) => entry.stores.has(name),
          },
          createObjectStore(name) {
            if (!entry.stores.has(name)) {
              entry.stores.set(name, new Map());
            }
            return makeStore(entry.stores.get(name));
          },
          transaction(storeName) {
            if (!entry.stores.has(storeName)) {
              throw new Error(`Object store not found: ${storeName}`);
            }
            const tx = {
              error: null,
              oncomplete: null,
              onerror: null,
              onabort: null,
              objectStore(name) {
                const map = entry.stores.get(name);
                const inner = makeStore(map);
                const wrap = (fn) => (...args) => {
                  const innerReq = fn(...args);
                  if (!options.hangGetAll) {
                    const orig = () => {
                      schedule(() => {
                        if (tx.oncomplete) tx.oncomplete({ target: tx });
                      });
                    };
                    schedule(orig);
                  }
                  return innerReq;
                };
                return {
                  put: wrap(inner.put),
                  delete: wrap(inner.delete),
                  getAll: wrap(inner.getAll),
                };
              },
            };
            return tx;
          },
          close() {},
        };

        req.result = db;
        if (isNew && req.onupgradeneeded) {
          req.onupgradeneeded({ target: req });
        }
        if (req.onsuccess) {
          req.onsuccess({ target: req });
        }
      });

      return req;
    },
  };
}

function createMockFsAdapter() {
  const dirs = new Set();
  const files = new Map();
  const mtimes = new Map();

  return {
    dirs,
    files,
    mtimes,
    mkdir(dir) {
      if (dirs.has(dir)) {
        const err = new Error(`EEXIST: ${dir}`);
        err.errno = 20;
        throw err;
      }
      dirs.add(dir);
    },
    writeFile(filePath, data) {
      files.set(filePath, new Uint8Array(data));
    },
    utime(filePath, _atime, mtime) {
      mtimes.set(filePath, mtime);
    },
  };
}

// --- RangeCache tests ---

test('RangeCache UMD export attaches PlayGta6RangeCache on globalThis', () => {
  assert.equal(globalThis.PlayGta6RangeCache.RangeCache, RangeCache);
  assert.equal(DEFAULT_CHUNK_SIZE, 256 * 1024);
  assert.equal(typeof openOpfsCacheDir, 'function');
});

test('RangeCache multi-chunk read and exact byte slicing across chunk boundaries', async () => {
  const fileBytes = makePatternBytes(70, 5);
  const files = new Map([['archives/test.rpf', fileBytes]]);
  const mock = createMockRangeFetch(files);

  const cache = new RangeCache({
    dataBase: '/data/',
    chunkSize: 16,
    maxBytes: 1024,
    fetchImpl: mock.fetchImpl,
  });

  // Read across chunks 0, 1, 2, 3 (offset 7..56 -> 50 bytes).
  const slice = await cache.read('archives/test.rpf', 7, 50);
  assert.deepEqual(slice, fileBytes.subarray(7, 57));
  assert.equal(mock.calls.length, 4);
  assert.deepEqual(
    mock.calls.map((c) => c.rangeHeader),
    ['bytes=0-15', 'bytes=16-31', 'bytes=32-47', 'bytes=48-63'],
  );

  // Mutating returned buffer must not corrupt cached chunks.
  slice.fill(0xff);
  const reread = await cache.read('archives/test.rpf', 7, 50);
  assert.deepEqual(reread, fileBytes.subarray(7, 57));
  assert.equal(mock.calls.length, 4, 'Second read must be served entirely from cache');

  const stats = cache.getStats();
  assert.equal(stats.misses, 4);
  assert.equal(stats.hits, 4);
  assert.equal(stats.entries, 4);
  assert.equal(stats.bytesCached, 64);
});

test('RangeCache handles partial tail chunks and reads past EOF safely', async () => {
  const fileBytes = makePatternBytes(50, 99);
  const files = new Map([['tail.bin', fileBytes]]);
  const mock = createMockRangeFetch(files);

  const cache = new RangeCache({
    chunkSize: 16,
    maxBytes: 512,
    fetchImpl: mock.fetchImpl,
  });

  // Chunks: [0..15], [16..31], [32..47], [48..49] (2-byte tail chunk).
  // Request offset 40, length 30 -> should return exact 10 bytes [40..49].
  const tail = await cache.read('tail.bin', 40, 30);
  assert.deepEqual(tail, fileBytes.subarray(40, 50));
  assert.equal(tail.byteLength, 10);

  // Reading completely beyond EOF returns 0 bytes without throwing.
  const eof = await cache.read('tail.bin', 100, 16);
  assert.equal(eof.byteLength, 0);
});

test('RangeCache deduplicates concurrent in-flight fetches for the same chunk', async () => {
  const fileBytes = makePatternBytes(64, 42);
  const files = new Map([['shared.bin', fileBytes]]);
  const mock = createMockRangeFetch(files, { delayMs: 25 });

  const cache = new RangeCache({
    chunkSize: 32,
    maxBytes: 1024,
    fetchImpl: mock.fetchImpl,
  });

  // Four overlapping concurrent reads all hitting chunk 0 and/or chunk 1.
  const [a, b, c, d] = await Promise.all([
    cache.read('shared.bin', 0, 16),
    cache.read('shared.bin', 4, 20),
    cache.read('shared.bin', 0, 48),
    cache.read('shared.bin', 16, 40),
  ]);

  assert.deepEqual(a, fileBytes.subarray(0, 16));
  assert.deepEqual(b, fileBytes.subarray(4, 24));
  assert.deepEqual(c, fileBytes.subarray(0, 48));
  assert.deepEqual(d, fileBytes.subarray(16, 56));
  assert.equal(mock.calls.length, 2, 'Should fetch chunk 0 and chunk 1 only once each');
  assert.equal(cache.getStats().misses, 2);
});

test('RangeCache evicts least-recently-used chunks when maxBytes is exceeded', async () => {
  const fileBytes = makePatternBytes(64, 7);
  const files = new Map([['lru.bin', fileBytes]]);
  const mock = createMockRangeFetch(files);

  // Budget fits at most 2 chunks (32 bytes).
  const cache = new RangeCache({
    chunkSize: 16,
    maxBytes: 32,
    fetchImpl: mock.fetchImpl,
  });

  // Load chunk 0 and chunk 1. Order (LRU -> MRU): [0, 1].
  await cache.read('lru.bin', 0, 16);
  await cache.read('lru.bin', 16, 16);
  assert.equal(cache.getStats().entries, 2);
  assert.equal(cache.getStats().bytesCached, 32);
  assert.equal(cache.getStats().evictions, 0);

  // Touch chunk 0 so order becomes [1, 0].
  await cache.read('lru.bin', 0, 8);
  assert.equal(cache.getStats().hits, 1);

  // Load chunk 2 -> must evict chunk 1 (not chunk 0).
  await cache.read('lru.bin', 32, 16);
  assert.equal(cache.getStats().evictions, 1);
  assert.equal(cache.getStats().entries, 2);
  assert.equal(cache.getStats().bytesCached, 32);

  const callsBeforeVerify = mock.calls.length;
  // Chunk 0 should still be a hit.
  await cache.read('lru.bin', 0, 4);
  assert.equal(mock.calls.length, callsBeforeVerify);
  assert.equal(cache.getStats().hits, 2);

  // Chunk 1 was evicted and must refetch (evicting chunk 2).
  await cache.read('lru.bin', 16, 4);
  assert.equal(mock.calls.length, callsBeforeVerify + 1);
  assert.equal(cache.getStats().evictions, 2);
});

test('RangeCache persists chunks to OPFS mock and serves warm OPFS hits with zero fetches', async () => {
  const fileBytes = makePatternBytes(48, 123);
  const files = new Map([['levels.rpf', fileBytes]]);
  const mock1 = createMockRangeFetch(files);
  const opfsDir = createMockOpfsDir();

  const coldCache = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: mock1.fetchImpl,
    opfsDir,
  });

  const first = await coldCache.read('levels.rpf', 8, 24);
  assert.deepEqual(first, fileBytes.subarray(8, 32));
  assert.equal(mock1.calls.length, 2);
  assert.equal(opfsDir.files.size, 2);

  // Create a brand new RangeCache instance sharing only the OPFS directory.
  const mock2 = createMockRangeFetch(files);
  const warmCache = new RangeCache({
    chunkSize: 16,
    maxBytes: 128,
    fetchImpl: mock2.fetchImpl,
    opfsDir,
  });

  const second = await warmCache.read('levels.rpf', 8, 24);
  assert.deepEqual(second, fileBytes.subarray(8, 32));
  assert.equal(mock2.calls.length, 0, 'Warm OPFS hit must not issue network fetches');

  const stats = warmCache.getStats();
  assert.equal(stats.opfsHits, 2);
  assert.equal(stats.hits, 2);
  assert.equal(stats.misses, 0);
  assert.equal(stats.entries, 2);
});

test('RangeCache prefetchBootset respects bounded concurrency and warms cache', async () => {
  const fileA = makePatternBytes(64, 11);
  const fileB = makePatternBytes(64, 22);
  const files = new Map([
    ['a.rpf', fileA],
    ['b.rpf', fileB],
  ]);
  const mock = createMockRangeFetch(files, { delayMs: 15 });

  const cache = new RangeCache({
    chunkSize: 16,
    maxBytes: 1024,
    maxPrefetchConcurrency: 2,
    fetchImpl: mock.fetchImpl,
  });

  const result = await cache.prefetchBootset([
    { path: 'a.rpf', offset: 0, length: 48 }, // 3 chunks
    { path: 'b.rpf', offset: 0, length: 32 }, // 2 chunks
    { path: 'missing.rpf', offset: 0, length: 16 }, // 1 failing chunk
  ]);

  assert.equal(result.requestedEntries, 3);
  assert.equal(result.uniqueChunks, 6);
  assert.equal(result.prefetched, 5);
  assert.equal(result.failed, 1);
  assert.ok(mock.getMaxConcurrent() <= 2, `Expected concurrency <= 2, got ${mock.getMaxConcurrent()}`);

  const callsAfterPrefetch = mock.calls.length;
  const outA = await cache.read('a.rpf', 10, 30);
  const outB = await cache.read('b.rpf', 0, 32);
  assert.deepEqual(outA, fileA.subarray(10, 40));
  assert.deepEqual(outB, fileB.subarray(0, 32));
  assert.equal(mock.calls.length, callsAfterPrefetch, 'Prefetched ranges must be cached hits');
});

// --- UserdataStore tests ---

test('UserdataStore UMD export attaches PlayGta6Userdata on globalThis', () => {
  assert.equal(globalThis.PlayGta6Userdata.UserdataStore, UserdataStore);
  assert.equal(DEFAULT_DB_NAME, 'playgta6-userdata');
  assert.equal(DEFAULT_STORE_NAME, 'files');
});

test('UserdataStore enforces strict /userdata/ path prefix and rejects traversal', async () => {
  const invalidPaths = [
    '',
    '/etc/passwd',
    '/userdata',
    '/userdata/',
    'userdata/save.bin',
    '/userdata/../evil',
    '/userdata/sub/../../evil',
    '/userdata/..',
    '/userdata/./save.bin',
    '/userdata/a//b.bin',
    '/userdata\\save.bin',
    '/userdata/sub\\save.bin',
    '/userdata/save.bin/',
    '/userdata/null\0byte',
  ];

  const store = new UserdataStore({ indexedDB: createMockIndexedDB() });
  for (const badPath of invalidPaths) {
    assert.throws(() => validateUserdataPath(badPath), `Expected rejection for ${JSON.stringify(badPath)}`);
    await assert.rejects(() => store.saveFile(badPath, new Uint8Array([1, 2, 3])));
    await assert.rejects(() => store.deleteFile(badPath));
  }

  assert.equal(validateUserdataPath('/userdata/save0.bin'), '/userdata/save0.bin');
  assert.equal(
    validateUserdataPath('/userdata/profiles/slot1/settings.json'),
    '/userdata/profiles/slot1/settings.json',
  );
  store.close();
});

test('UserdataStore save, restore, and delete roundtrip with mock IndexedDB', async () => {
  const mockIdb = createMockIndexedDB();
  const store1 = new UserdataStore({ indexedDB: mockIdb });

  const savePayload = new Uint8Array([0x53, 0x41, 0x56, 0x45, 0x01, 0x02]);
  const settingsPayload = new TextEncoder().encode('{"volume":0.8}');

  await store1.saveFile('/userdata/slot0.sav', savePayload, 1700000001000);
  await store1.saveFile('/userdata/profiles/default/settings.json', settingsPayload, 1700000002000);
  await store1.saveFile('/userdata/temp.bin', new Uint8Array([9, 9]), 1700000003000);
  await store1.deleteFile('/userdata/temp.bin');
  store1.close();

  // Restore into a fresh UserdataStore + mock FS adapter.
  const store2 = new UserdataStore({ indexedDB: mockIdb });
  const fsAdapter = createMockFsAdapter();
  const result = await store2.restoreAll(fsAdapter, { timeoutMs: 2000 });

  assert.equal(result.timedOut, false);
  assert.equal(result.error, null);
  assert.equal(result.restored, 2);
  assert.equal(result.skipped, 0);

  assert.deepEqual(fsAdapter.files.get('/userdata/slot0.sav'), savePayload);
  assert.deepEqual(fsAdapter.files.get('/userdata/profiles/default/settings.json'), settingsPayload);
  assert.equal(fsAdapter.files.has('/userdata/temp.bin'), false);
  assert.equal(fsAdapter.dirs.has('/userdata'), true);
  assert.equal(fsAdapter.dirs.has('/userdata/profiles'), true);
  assert.equal(fsAdapter.dirs.has('/userdata/profiles/default'), true);
  assert.equal(fsAdapter.mtimes.get('/userdata/slot0.sav'), 1700000001000);
  store2.close();
});

test('UserdataStore restoreAll falls back cleanly on timeout when IndexedDB hangs', async () => {
  const hungOpenIdb = createMockIndexedDB({ hangOpen: true });
  const storeOpenHang = new UserdataStore({ indexedDB: hungOpenIdb });
  const fsAdapter1 = createMockFsAdapter();

  const start = Date.now();
  const res1 = await storeOpenHang.restoreAll(fsAdapter1, { timeoutMs: 40 });
  const elapsed = Date.now() - start;

  assert.equal(res1.timedOut, true);
  assert.equal(res1.restored, 0);
  assert.match(res1.error, /timed out/i);
  assert.ok(elapsed < 1000, `Expected prompt timeout fallback, took ${elapsed}ms`);

  // Also test hanging getAll() after DB opens.
  const hungReadIdb = createMockIndexedDB({ hangGetAll: true });
  const storeReadHang = new UserdataStore({ indexedDB: hungReadIdb });
  const fsAdapter2 = createMockFsAdapter();

  const res2 = await storeReadHang.restoreAll(fsAdapter2, { timeoutMs: 40 });
  assert.equal(res2.timedOut, true);
  assert.equal(res2.restored, 0);
  assert.match(res2.error, /timed out/i);
  storeReadHang.close();
});

test('UserdataStore restoreAll degrades gracefully when IndexedDB is unavailable', async () => {
  const store = new UserdataStore({ indexedDB: null });
  const fsAdapter = createMockFsAdapter();
  const res = await store.restoreAll(fsAdapter, { timeoutMs: 100 });
  assert.equal(res.timedOut, false);
  assert.equal(res.restored, 0);
  assert.match(res.error, /unavailable/i);
});

// --- io_worker.js classic Worker integration test ---

test('io_worker handles init (with optional bootset), read, and stats messages', async () => {
  const archive = makePatternBytes(64, 55);
  const manifestJson = new TextEncoder().encode(
    JSON.stringify({ abi_version: 1, file_count: 1, total_bytes: 64, files: [{ path: 'game.rpf', size: 64 }] }),
  );
  const bootsetJson = new TextEncoder().encode(
    JSON.stringify([{ path: 'game.rpf', offset: 0, length: 32 }]),
  );

  const files = new Map([
    ['game.rpf', archive],
    ['data-manifest.json', manifestJson],
    ['bootset.json', bootsetJson],
  ]);
  const mock = createMockRangeFetch(files);

  const posted = [];
  const workerSelf = {
    onmessage: null,
    postMessage(msg) {
      posted.push(msg);
    },
  };

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    fetch: mock.fetchImpl,
    navigator: {},
    Uint8Array,
    ArrayBuffer,
    TextEncoder,
    TextDecoder,
    Map,
    Set,
    Promise,
    Error,
    Number,
    Math,
    importScripts(scriptPath) {
      const resolved = path.resolve(repoRoot, 'web/workers', scriptPath);
      const code = fs.readFileSync(resolved, 'utf8');
      vm.runInContext(code, context, { filename: resolved });
    },
  };
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);

  const ioWorkerPath = path.join(repoRoot, 'web/workers/io_worker.js');
  vm.runInContext(fs.readFileSync(ioWorkerPath, 'utf8'), context, { filename: ioWorkerPath });

  assert.equal(typeof workerSelf.onmessage, 'function');

  // 1. Init worker
  await workerSelf.onmessage({ data: { init: true, dataBase: '/data/', chunkSize: 16, maxBytes: 256 } });
  assert.equal(posted.length, 1);
  assert.equal(posted[0].ready, true);
  assert.equal(posted[0].bootsetEntries, 1);
  assert.match(posted[0].summary, /1 files indexed/);

  // Allow background bootset prefetch to settle.
  await new Promise((r) => setTimeout(r, 20));

  // 2. Read range covered by bootset prefetch
  await workerSelf.onmessage({ data: { read: true, id: 7, path: 'game.rpf', offset: 4, length: 20 } });
  assert.equal(posted[1].id, 7);
  assert.deepEqual(new Uint8Array(posted[1].bytes), archive.subarray(4, 24));

  // 3. Query stats
  await workerSelf.onmessage({ data: { stats: true, id: 8 } });
  assert.equal(posted[2].id, 8);
  assert.equal(posted[2].stats.misses, 2);
  assert.equal(posted[2].stats.hits, 2);
  assert.equal(posted[2].stats.entries, 2);
});

test('io_worker init succeeds gracefully when manifest, bootset, and OPFS are absent', async () => {
  const mock = createMockRangeFetch(new Map());
  const posted = [];
  const workerSelf = {
    onmessage: null,
    postMessage(msg) {
      posted.push(msg);
    },
  };

  const sandbox = {
    self: workerSelf,
    globalThis: null,
    fetch: mock.fetchImpl,
    navigator: {},
    Uint8Array,
    ArrayBuffer,
    TextEncoder,
    TextDecoder,
    Map,
    Set,
    Promise,
    Error,
    Number,
    Math,
    importScripts(scriptPath) {
      const resolved = path.resolve(repoRoot, 'web/workers', scriptPath);
      vm.runInContext(fs.readFileSync(resolved, 'utf8'), context, { filename: resolved });
    },
  };
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);

  const ioWorkerPath = path.join(repoRoot, 'web/workers/io_worker.js');
  vm.runInContext(fs.readFileSync(ioWorkerPath, 'utf8'), context, { filename: ioWorkerPath });

  await workerSelf.onmessage({ data: { init: true, dataBase: '/data/' } });
  assert.equal(posted.length, 1);
  assert.equal(posted[0].ready, true);
  assert.equal(posted[0].opfs, false);
  assert.equal(posted[0].bootsetEntries, 0);
  assert.match(posted[0].summary, /no data-manifest\.json yet/);
});

test('loader.js preserves WebGPU self-test when dist/runtime.js is absent and wires userdata when present', async () => {
  async function runLoaderScenario({ runtimeAvailable, mockIdb }) {
    const progressMessages = [];
    const gpuMessages = [];
    let receivedModuleOpts = null;

    class MockBroadcastChannel {
      postMessage(msg) {
        progressMessages.push(msg);
      }
    }

    const workersByScript = new Map();
    class MockWorker {
      constructor(script) {
        this.script = script;
        this.onmessage = null;
        this.posted = [];
        workersByScript.set(script, this);
      }
      postMessage(msg) {
        this.posted.push(msg);
        if (this.script.includes('wgpu_worker')) {
          gpuMessages.push(msg);
        }
        if (msg.init) {
          queueMicrotask(() => {
            if (this.onmessage) {
              this.onmessage({ data: { ready: true, summary: 'ok', adapter: 'mock-gpu' } });
            }
          });
        }
      }
    }

    const workerSelf = { onmessage: null };
    const sandbox = {
      self: workerSelf,
      globalThis: null,
      BroadcastChannel: MockBroadcastChannel,
      Worker: MockWorker,
      indexedDB: mockIdb,
      Uint8Array,
      ArrayBuffer,
      DataView,
      TextEncoder,
      TextDecoder,
      Map,
      Set,
      Promise,
      Error,
      Number,
      Math,
      setTimeout,
      clearTimeout,
      fetch: async (url) => {
        if (url === './dist/runtime.js') {
          return { ok: runtimeAvailable, status: runtimeAvailable ? 200 : 404 };
        }
        return { ok: false, status: 404 };
      },
      importScripts(scriptPath) {
        if (scriptPath === './dist/runtime.js') {
          workerSelf.createRuntimeModule = async (opts) => {
            receivedModuleOpts = opts;
            return {};
          };
          return;
        }
        const resolved = path.resolve(repoRoot, 'web', scriptPath);
        vm.runInContext(fs.readFileSync(resolved, 'utf8'), context, { filename: resolved });
      },
    };
    sandbox.globalThis = sandbox;
    const context = vm.createContext(sandbox);

    const loaderPath = path.join(repoRoot, 'web/loader.js');
    vm.runInContext(fs.readFileSync(loaderPath, 'utf8'), context, { filename: loaderPath });

    workerSelf.onmessage({
      data: { canvas: { id: 'mock-canvas' }, width: 640, height: 480, dataBase: '/data/' },
    });

    await new Promise((r) => setTimeout(r, 25));
    return { progressMessages, gpuMessages, receivedModuleOpts };
  }

  // Case 1: dist/runtime.js absent -> WebGPU self-test packet posted to wgpu_worker.
  const absent = await runLoaderScenario({ runtimeAvailable: false, mockIdb: createMockIndexedDB() });
  const cmdMsg = absent.gpuMessages.find((m) => m.cmdBuffer instanceof ArrayBuffer);
  assert.ok(cmdMsg, 'Expected WebGPU command-ring self-test packet when dist/runtime.js is absent');
  assert.equal(cmdMsg.cmdBuffer.byteLength, 40);

  // Case 2: dist/runtime.js present -> userdata callbacks and preRun hook wired.
  const mockIdb = createMockIndexedDB();
  const present = await runLoaderScenario({ runtimeAvailable: true, mockIdb });
  assert.ok(present.receivedModuleOpts, 'Expected createRuntimeModule to be invoked');
  assert.equal(typeof present.receivedModuleOpts.hostCallbacks.userdataSave, 'function');
  assert.equal(typeof present.receivedModuleOpts.hostCallbacks.userdataRestore, 'function');
  assert.equal(typeof present.receivedModuleOpts.hostCallbacks.userdataDelete, 'function');
  assert.equal(Array.isArray(present.receivedModuleOpts.preRun), true);
  assert.equal(present.receivedModuleOpts.preRun.length, 1);

  const payload = new Uint8Array([1, 2, 3, 4]);
  await present.receivedModuleOpts.hostCallbacks.userdataSave('/userdata/test.sav', payload, 12345);

  const fsAdapter = createMockFsAdapter();
  const deps = [];
  present.receivedModuleOpts.preRun[0]({
    FS: fsAdapter,
    addRunDependency: (name) => deps.push(`add:${name}`),
    removeRunDependency: (name) => deps.push(`remove:${name}`),
  });
  await new Promise((r) => setTimeout(r, 25));

  assert.deepEqual(deps, ['add:userdata-restore', 'remove:userdata-restore']);
  assert.deepEqual(fsAdapter.files.get('/userdata/test.sav'), payload);
});

