'use strict';

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const require = createRequire(import.meta.url);

const loaderApi = require('../web/loader.js');
const { UserdataStore } = require('../web/workers/userdata.js');
const { bootstrapPlayGta6App } = await import('../web/app_bootstrap.js');

const DIR_MODE = 0o040755;
const FILE_MODE = 0o100644;

function makeErrnoError(code, errno, message) {
  const err = new Error(message || code);
  err.code = code;
  err.errno = errno;
  return err;
}

/**
 * Emscripten-compatible in-memory FS mock supporting nested directories,
 * stat/readdir/readFile/writeFile/mkdirTree/unlink/rmTree, and mtime tracking.
 */
function createMockEmscriptenFS() {
  const dirs = new Set(['/']);
  const files = new Map(); // path -> { bytes: Uint8Array, mtime: number }

  function norm(p) {
    const clean = String(p).replace(/\/+/g, '/');
    return clean.length > 1 && clean.endsWith('/') ? clean.slice(0, -1) : clean;
  }

  function mkdirTree(dirPath) {
    const target = norm(dirPath);
    if (target === '/') return;
    const parts = target.split('/').filter(Boolean);
    let cur = '';
    for (const part of parts) {
      cur += `/${part}`;
      dirs.add(cur);
    }
  }

  function mkdir(dirPath) {
    const target = norm(dirPath);
    if (dirs.has(target)) {
      throw makeErrnoError('EEXIST', 20, `File exists: ${target}`);
    }
    mkdirTree(target);
  }

  function writeFile(filePath, data, mtime = Date.now()) {
    const target = norm(filePath);
    const slash = target.lastIndexOf('/');
    if (slash > 0) {
      mkdirTree(target.slice(0, slash));
    }
    const bytes =
      data instanceof Uint8Array
        ? data.slice()
        : typeof data === 'string'
          ? new TextEncoder().encode(data)
          : new Uint8Array(data);
    files.set(target, {
      bytes,
      mtime: typeof mtime === 'number' && Number.isFinite(mtime) ? mtime : Date.now(),
    });
  }

  function utime(filePath, _atime, mtime) {
    const target = norm(filePath);
    const entry = files.get(target);
    if (entry && typeof mtime === 'number' && Number.isFinite(mtime)) {
      entry.mtime = mtime;
    }
  }

  function unlink(filePath) {
    const target = norm(filePath);
    if (!files.has(target)) {
      throw makeErrnoError('ENOENT', 44, `No such file: ${target}`);
    }
    files.delete(target);
  }

  function rmTree(treePath) {
    const target = norm(treePath);
    const prefix = `${target}/`;
    for (const k of Array.from(files.keys())) {
      if (k === target || k.startsWith(prefix)) {
        files.delete(k);
      }
    }
    for (const d of Array.from(dirs)) {
      if (d === target || d.startsWith(prefix)) {
        dirs.delete(d);
      }
    }
  }

  function stat(p) {
    const target = norm(p);
    if (files.has(target)) {
      const entry = files.get(target);
      return {
        mode: FILE_MODE,
        size: entry.bytes.byteLength,
        mtime: new Date(entry.mtime),
      };
    }
    if (dirs.has(target)) {
      return {
        mode: DIR_MODE,
        size: 0,
        mtime: new Date(0),
      };
    }
    throw makeErrnoError('ENOENT', 44, `No such file or directory: ${target}`);
  }

  function readdir(dirPath) {
    const target = norm(dirPath);
    if (!dirs.has(target)) {
      throw makeErrnoError('ENOENT', 44, `No such directory: ${target}`);
    }
    const prefix = target === '/' ? '/' : `${target}/`;
    const names = new Set(['.', '..']);
    for (const d of dirs) {
      if (d !== target && d.startsWith(prefix)) {
        const rest = d.slice(prefix.length);
        const first = rest.split('/')[0];
        if (first) names.add(first);
      }
    }
    for (const f of files.keys()) {
      if (f.startsWith(prefix)) {
        const rest = f.slice(prefix.length);
        const first = rest.split('/')[0];
        if (first) names.add(first);
      }
    }
    return Array.from(names);
  }

  function readFile(filePath) {
    const target = norm(filePath);
    const entry = files.get(target);
    if (!entry) {
      throw makeErrnoError('ENOENT', 44, `No such file: ${target}`);
    }
    return entry.bytes.slice();
  }

  return {
    dirs,
    files,
    mkdir,
    mkdirTree,
    writeFile,
    utime,
    unlink,
    rmTree,
    stat,
    readdir,
    readFile,
    isDir: (mode) => (mode & 0o170000) === 0o040000,
    isFile: (mode) => (mode & 0o170000) === 0o100000,
  };
}

/**
 * In-memory backend for UserdataStore with optional failure injection and concurrency tracking.
 */
function createInMemoryUserdataBackend(options = {}) {
  const records = new Map();
  let activeWrites = 0;
  let maxConcurrentWrites = 0;
  let writeDelayMs = options.writeDelayMs || 0;
  let shouldFailWrites = Boolean(options.shouldFailWrites);
  let onWriteStarted = typeof options.onWriteStarted === 'function' ? options.onWriteStarted : null;

  return {
    records,
    get maxConcurrentWrites() {
      return maxConcurrentWrites;
    },
    setWriteDelayMs(ms) {
      writeDelayMs = ms;
    },
    setFailWrites(flag) {
      shouldFailWrites = Boolean(flag);
    },
    setOnWriteStarted(fn) {
      onWriteStarted = fn;
    },
    async getAll() {
      return Array.from(records.values()).map((r) => ({
        ...r,
        data: r.data instanceof Uint8Array ? r.data.slice() : r.data,
      }));
    },
    async get(key) {
      const rec = records.get(key);
      if (!rec) return null;
      return { ...rec, data: rec.data instanceof Uint8Array ? rec.data.slice() : rec.data };
    },
    async put(rec) {
      activeWrites += 1;
      if (activeWrites > maxConcurrentWrites) {
        maxConcurrentWrites = activeWrites;
      }
      if (onWriteStarted) {
        onWriteStarted(rec.path);
      }
      try {
        if (writeDelayMs > 0) {
          await new Promise((r) => setTimeout(r, writeDelayMs));
        }
        if (shouldFailWrites) {
          throw new Error('Simulated IndexedDB write failure');
        }
        records.set(rec.path, {
          ...rec,
          data: rec.data instanceof Uint8Array ? rec.data.slice() : rec.data,
        });
      } finally {
        activeWrites -= 1;
      }
    },
    async delete(key) {
      if (shouldFailWrites) {
        throw new Error('Simulated IndexedDB delete failure');
      }
      records.delete(key);
    },
  };
}

/**
 * Creates a mock Wasm runtime module matching runtime/wasm_boot.cpp's
 * `_playgta6_storage_generation` + `_playgta6_storage_take_changed` contract.
 */
function createMockStorageWasmModule({ wasm64 = true, sharedMemory = false, fsObj = null } = {}) {
  const buffer = sharedMemory ? new SharedArrayBuffer(65536) : new ArrayBuffer(65536);
  const heapU8 = new Uint8Array(buffer);
  const mockFs = fsObj || createMockEmscriptenFS();
  const encoder = new TextEncoder();

  let generation = 0n;
  const dirtyPaths = new Set();
  const stringBase = 4096;

  function writeCString(text) {
    const encoded = encoder.encode(text);
    heapU8.set(encoded, stringBase);
    heapU8[stringBase + encoded.byteLength] = 0;
    return wasm64 ? BigInt(stringBase) : stringBase;
  }

  const mod = {
    FS: mockFs,
    HEAPU8: heapU8,
    wasmMemory: { buffer },
    _playgta6_storage_generation() {
      return wasm64 ? generation : Number(generation);
    },
    _playgta6_storage_take_changed() {
      const joined = Array.from(dirtyPaths)
        .map((p) => `${p}\n`)
        .join('');
      dirtyPaths.clear();
      return writeCString(joined);
    },
    markChanged(containerPath) {
      dirtyPaths.add(containerPath);
      generation += 1n;
      return generation;
    },
    getGeneration() {
      return generation;
    },
  };

  return mod;
}

test('readWasmCString handles BigInt Wasm64 pointers, Number Wasm32 pointers, SharedArrayBuffer heaps, and UTF8ToString fallback', () => {
  const sabMod = createMockStorageWasmModule({ wasm64: true, sharedMemory: true });
  const ptr64 = sabMod._playgta6_storage_take_changed();
  assert.equal(typeof ptr64, 'bigint');
  assert.equal(loaderApi.readWasmCString(sabMod, ptr64), '');

  sabMod.markChanged('/userdata/saves/default/local/c-profile');
  const ptrWithData = sabMod._playgta6_storage_take_changed();
  assert.equal(
    loaderApi.readWasmCString(sabMod, ptrWithData),
    '/userdata/saves/default/local/c-profile\n',
  );

  const wasm32Mod = createMockStorageWasmModule({ wasm64: false, sharedMemory: false });
  wasm32Mod.markChanged('/userdata/saves/default/local/c-slot1');
  const ptr32 = wasm32Mod._playgta6_storage_take_changed();
  assert.equal(typeof ptr32, 'number');
  assert.equal(
    loaderApi.readWasmCString(wasm32Mod, ptr32),
    '/userdata/saves/default/local/c-slot1\n',
  );

  // Fallback to UTF8ToString when HEAPU8/wasmMemory are omitted (still passes numeric ptr).
  let seenPtr = null;
  const fallbackMod = {
    UTF8ToString(p) {
      seenPtr = p;
      return '/userdata/saves/default/local/c-fallback\n';
    },
  };
  assert.equal(
    loaderApi.readWasmCString(fallbackMod, 128n),
    '/userdata/saves/default/local/c-fallback\n',
  );
  assert.equal(seenPtr, 128);
  assert.equal(loaderApi.readWasmCString(fallbackMod, 0n), '');
});

test('SaveSyncController polls Wasm64 BigInt and Wasm32 Number generations and persists nested container files while skipping transient swap files', async () => {
  for (const wasm64 of [true, false]) {
    const backend = createInMemoryUserdataBackend();
    const store = new UserdataStore({ backend });
    const mod = createMockStorageWasmModule({ wasm64, sharedMemory: wasm64 });
    const telemetry = [];

    const controller = loaderApi.createSaveSyncController({
      userdataStore: store,
      module: mod,
      reportFn: (msg) => telemetry.push(msg),
      setIntervalFn: null,
    });

    // No changes yet -> generation stays 0, zero writes.
    const idleStats = await controller.pollAndPersist();
    assert.equal(idleStats.lastPersistedGeneration, '0');
    assert.equal(idleStats.persistedFiles, 0);
    assert.equal(telemetry.length, 0);

    // Write nested container layout matching runtime/os/xgamesave_local.cpp under /userdata/saves.
    const profileDir = '/userdata/saves/default/local/c-profile';
    const settingsBytes = new Uint8Array([0x53, 0x45, 0x54, 0x01, 0x02]);
    const calibBytes = new Uint8Array([0x43, 0x41, 0x4c, 0x09]);
    const displayNameBytes = new TextEncoder().encode('Player Settings');
    mod.FS.writeFile(`${profileDir}/b-settings`, settingsBytes, 1700000001000);
    mod.FS.writeFile(`${profileDir}/nested/b-calibration`, calibBytes, 1700000002000);
    mod.FS.writeFile(`${profileDir}/.display-name`, displayNameBytes, 1700000003000);
    // Transient xgamesave temp/swap files must never be persisted.
    mod.FS.writeFile(`${profileDir}/.tmp-0`, new Uint8Array([0xff]), 1700000004000);
    mod.FS.writeFile(`${profileDir}/.new-c-profile-1`, new Uint8Array([0xfe]), 1700000004000);
    mod.FS.writeFile(`${profileDir}/.old-c-profile-1`, new Uint8Array([0xfd]), 1700000004000);
    mod.markChanged(profileDir);

    const stats = await controller.pollAndPersist();
    assert.equal(stats.lastPersistedGeneration, '1');
    assert.equal(stats.persistedFiles, 3);
    assert.equal(stats.deletedFiles, 0);
    assert.equal(stats.errorCount, 0);
    assert.equal(telemetry.length, 1);
    assert.match(telemetry[0], /userdata persisted 3 files, generation 1/);

    const listed = await store.listFiles();
    assert.deepEqual(
      listed.map((f) => f.path).sort(),
      [
        `${profileDir}/.display-name`,
        `${profileDir}/b-settings`,
        `${profileDir}/nested/b-calibration`,
      ],
    );
    assert.deepEqual((await store.getFile(`${profileDir}/b-settings`)).data, settingsBytes);
    assert.deepEqual((await store.getFile(`${profileDir}/nested/b-calibration`)).data, calibBytes);
    assert.deepEqual((await store.getFile(`${profileDir}/.display-name`)).data, displayNameBytes);
  }
});

test('SaveSyncController deletes stale files inside updated containers and removes deleted containers completely', async () => {
  const backend = createInMemoryUserdataBackend();
  const store = new UserdataStore({ backend });
  const mod = createMockStorageWasmModule({ wasm64: true });

  const controller = loaderApi.createSaveSyncController({
    userdataStore: store,
    module: mod,
    setIntervalFn: null,
  });

  const c1 = '/userdata/saves/default/local/c-slot1';
  const c2 = '/userdata/saves/default/local/c-slot2';
  mod.FS.writeFile(`${c1}/b-header`, new Uint8Array([1, 2, 3]), 1000);
  mod.FS.writeFile(`${c1}/b-oldblob`, new Uint8Array([4, 5, 6]), 1000);
  mod.FS.writeFile(`${c2}/b-data`, new Uint8Array([7, 8, 9]), 2000);
  mod.markChanged(c1);
  mod.markChanged(c2);

  await controller.pollAndPersist();
  assert.equal((await store.listFiles()).length, 3);

  // Update c1: replace b-header and remove b-oldblob.
  mod.FS.writeFile(`${c1}/b-header`, new Uint8Array([10, 20, 30, 40]), 3000);
  mod.FS.unlink(`${c1}/b-oldblob`);
  mod.markChanged(c1);

  const afterUpdate = await controller.pollAndPersist();
  assert.equal(afterUpdate.deletedFiles, 1);
  assert.equal(await store.getFile(`${c1}/b-oldblob`), null);
  assert.deepEqual((await store.getFile(`${c1}/b-header`)).data, new Uint8Array([10, 20, 30, 40]));
  assert.notEqual(await store.getFile(`${c2}/b-data`), null);

  // Delete entire c2 container in MEMFS (XGameSaveDeleteContainer).
  mod.FS.rmTree(c2);
  mod.markChanged(c2);

  const afterDelete = await controller.pollAndPersist();
  assert.equal(afterDelete.deletedFiles, 2);
  assert.equal(await store.getFile(`${c2}/b-data`), null);
  const remaining = await store.listFiles();
  assert.deepEqual(
    remaining.map((f) => f.path),
    [`${c1}/b-header`],
  );
});

test('Full reload round-trip: persisted saves restore into a fresh mock FS at the exact /userdata/saves path xgamesave_local reads', async () => {
  const sharedBackend = createInMemoryUserdataBackend();

  // Session 1: first boot writes legal acceptance, calibration, and save slot under /userdata/saves.
  const storeSession1 = new UserdataStore({ backend: sharedBackend });
  const modSession1 = createMockStorageWasmModule({ wasm64: true, sharedMemory: true });
  const sync1 = loaderApi.createSaveSyncController({
    userdataStore: storeSession1,
    module: modSession1,
    setIntervalFn: null,
  });

  const cfgRoot = `${loaderApi.DEFAULT_SAVE_ROOT}/default/local`;
  const legalPath = `${cfgRoot}/c-profile/b-legal_accepted`;
  const calibPath = `${cfgRoot}/c-profile/b-calibration`;
  const slotPath = `${cfgRoot}/c-save0/b-world`;
  const legalPayload = new Uint8Array([1, 0, 0, 0]);
  const calibPayload = new Uint8Array([0x3f, 0x80, 0x00, 0x00]);
  const slotPayload = new Uint8Array([0x47, 0x54, 0x41, 0x36, 0x00, 0x42]);

  modSession1.FS.writeFile(legalPath, legalPayload, 1700001000000);
  modSession1.FS.writeFile(calibPath, calibPayload, 1700001000500);
  modSession1.FS.writeFile(slotPath, slotPayload, 1700001001000);
  modSession1.markChanged(`${cfgRoot}/c-profile`);
  modSession1.markChanged(`${cfgRoot}/c-save0`);

  await sync1.pollAndPersist();

  // Session 2: fresh page reload with a brand-new empty Emscripten MEMFS & fresh UserdataStore instance.
  const storeSession2 = new UserdataStore({ backend: sharedBackend });
  const freshFs = createMockEmscriptenFS();

  // Exercise the exact preRun restore path from buildRuntimeModuleConfig.
  let restoredReport = null;
  let runDepAdded = false;
  let runDepRemoved = false;
  const config = loaderApi.buildRuntimeModuleConfig({
    userdataStore: storeSession2,
    reportFn: (line) => {
      if (line.includes('userdata restored')) {
        restoredReport = line;
      }
    },
  });
  const modSession2 = {
    FS: freshFs,
    addRunDependency(name) {
      if (name === 'userdata-restore') runDepAdded = true;
    },
    removeRunDependency(name) {
      if (name === 'userdata-restore') runDepRemoved = true;
    },
  };
  await new Promise((resolve) => {
    const origRemove = modSession2.removeRunDependency;
    modSession2.removeRunDependency = (name) => {
      origRemove(name);
      if (name === 'userdata-restore') resolve();
    };
    for (const fn of config.preRun) {
      fn(modSession2);
    }
  });

  assert.equal(runDepAdded, true);
  assert.equal(runDepRemoved, true);
  assert.equal(restoredReport, 'userdata restored (3 files)');

  // Verify xgamesave_local sees all restored files at exact paths & bytes!
  assert.deepEqual(freshFs.readFile(legalPath), legalPayload);
  assert.deepEqual(freshFs.readFile(calibPath), calibPayload);
  assert.deepEqual(freshFs.readFile(slotPath), slotPayload);
  assert.equal(freshFs.files.get(legalPath).mtime, 1700001000000);
});

test('SaveSyncController serializes overlapping flushes and runs one trailing pass when generation advances mid-flush', async () => {
  const backend = createInMemoryUserdataBackend({ writeDelayMs: 25 });
  const store = new UserdataStore({ backend });
  const mod = createMockStorageWasmModule({ wasm64: true });

  const container = '/userdata/saves/default/local/c-profile';
  const v1 = new Uint8Array([1, 1, 1]);
  const v2 = new Uint8Array([2, 2, 2, 2]);
  mod.FS.writeFile(`${container}/b-state`, v1, 1000);
  mod.markChanged(container);

  const controller = loaderApi.createSaveSyncController({
    userdataStore: store,
    module: mod,
    setIntervalFn: null,
  });

  let bumpedMidFlush = false;
  let midFlushCallerPromise = null;
  backend.setOnWriteStarted((writtenPath) => {
    if (!bumpedMidFlush && writtenPath === `${container}/b-state`) {
      bumpedMidFlush = true;
      // Guest writes v2 while IndexedDB write for v1 is still in flight!
      mod.FS.writeFile(`${container}/b-state`, v2, 2000);
      mod.markChanged(container);
      // Concurrent flush-userdata arrives mid-flush: must coalesce without overlapping writes.
      midFlushCallerPromise = controller.flush();
    }
  });

  const firstPromise = controller.flush();
  const [stats1, stats2] = await Promise.all([firstPromise, Promise.resolve().then(() => midFlushCallerPromise)]);
  const finalStats = stats2 || stats1;

  assert.equal(bumpedMidFlush, true);
  assert.equal(backend.maxConcurrentWrites, 1, 'IndexedDB writes must never overlap');
  assert.equal(finalStats.lastPersistedGeneration, '2');
  assert.deepEqual((await store.getFile(`${container}/b-state`)).data, v2);
});

test('SaveSyncController recovers cleanly from transient IndexedDB errors and rejects paths outside /userdata/', async () => {
  const backend = createInMemoryUserdataBackend({ shouldFailWrites: true });
  const store = new UserdataStore({ backend });
  const mod = createMockStorageWasmModule({ wasm64: true });

  const controller = loaderApi.createSaveSyncController({
    userdataStore: store,
    module: mod,
    setIntervalFn: null,
  });

  const validContainer = '/userdata/saves/default/local/c-profile';
  mod.FS.writeFile(`${validContainer}/b-cfg`, new Uint8Array([9, 8, 7]), 5000);
  mod.FS.writeFile('/outside/secret.bin', new Uint8Array([0xaa]), 5000);
  mod.markChanged('/outside/secret.bin');
  mod.markChanged('/userdata/../etc/passwd');
  mod.markChanged(validContainer);

  // First flush fails gracefully without throwing and queues retry.
  const failedStats = await controller.pollAndPersist();
  assert.equal(failedStats.errorCount, 1);
  assert.equal(failedStats.lastPersistedGeneration, '0');
  assert.equal(failedStats.pendingPaths, 1);

  // Storage recovers: next poll retries pending path and succeeds.
  backend.setFailWrites(false);
  const okStats = await controller.pollAndPersist();
  assert.equal(okStats.lastPersistedGeneration, '3');
  assert.equal(okStats.pendingPaths, 0);
  const stored = await store.listFiles();
  assert.deepEqual(
    stored.map((f) => f.path),
    [`${validContainer}/b-cfg`],
  );
});

test('buildRuntimeModuleConfig starts SaveSyncController on onRuntimeInitialized and flushes final saves on onExit before clearing boundModule', async () => {
  const backend = createInMemoryUserdataBackend({ writeDelayMs: 5 });
  const store = new UserdataStore({ backend });
  const mod = createMockStorageWasmModule({ wasm64: true });

  let scheduledFn = null;
  let scheduledMs = null;
  let clearedTimerId = null;

  const config = loaderApi.buildRuntimeModuleConfig({
    userdataStore: store,
    saveSyncIntervalMs: 2000,
    saveSyncSetIntervalFn: (fn, ms) => {
      scheduledFn = fn;
      scheduledMs = ms;
      return 77;
    },
    saveSyncClearIntervalFn: (id) => {
      clearedTimerId = id;
    },
  });

  config.onRuntimeInitialized.call(mod);
  assert.equal(scheduledMs, 2000);
  assert.equal(typeof scheduledFn, 'function');

  // Write a save and simulate periodic timer tick.
  const c1 = '/userdata/saves/default/local/c-profile';
  mod.FS.writeFile(`${c1}/b-opts`, new Uint8Array([42]), 1000);
  mod.markChanged(c1);
  scheduledFn();
  await config.flushUserdata();
  assert.deepEqual((await store.getFile(`${c1}/b-opts`)).data, new Uint8Array([42]));

  // Write another save right before exit and invoke onExit (which clears boundModule synchronously).
  mod.FS.writeFile(`${c1}/b-opts`, new Uint8Array([99]), 2000);
  mod.markChanged(c1);
  config.onExit.call(mod, 0);
  assert.equal(clearedTimerId, 77);

  // Wait briefly for the async IDB write of the pre-exit snapshot to finish.
  await new Promise((r) => setTimeout(r, 30));
  assert.deepEqual((await store.getFile(`${c1}/b-opts`)).data, new Uint8Array([99]));
});

test('loader.js coordinator VM handles flush-userdata message and replies with userdata-flushed stats', async () => {
  const loaderSource = fs.readFileSync(path.join(REPO_ROOT, 'web', 'loader.js'), 'utf8');
  const backend = createInMemoryUserdataBackend();
  const sharedStore = new UserdataStore({ backend });
  const mockMod = createMockStorageWasmModule({ wasm64: true });

  const postedToMain = [];
  const createdWorkers = [];

  class FakeWorker {
    constructor(url) {
      this.url = String(url);
      this.onmessage = null;
      this.onerror = null;
      createdWorkers.push(this);
    }
    postMessage() {
      // Simulate immediate worker ready response.
      queueMicrotask(() => {
        if (typeof this.onmessage === 'function') {
          if (this.url.includes('wgpu_worker')) {
            this.onmessage({ data: { ready: true, adapter: 'mock-gpu' } });
          } else {
            this.onmessage({ data: { ready: true } });
          }
        }
      });
    }
    terminate() {}
  }

  const sandbox = {
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    TextEncoder,
    TextDecoder,
    Uint8Array,
    Int32Array,
    ArrayBuffer,
    SharedArrayBuffer,
    URL,
    Promise,
    Date,
    Math,
    JSON,
    Map,
    Set,
    Error,
    TypeError,
    RangeError,
    BigInt,
    Number,
    Object,
    Array,
    String,
    Boolean,
    Worker: FakeWorker,
    PlayGta6Userdata: {
      UserdataStore: function FakeUserdataStoreCtor() {
        return sharedStore;
      },
    },
    importScripts: () => {},
    fetch: async () => ({ ok: true, status: 200, json: async () => ({}) }),
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  sandbox.postMessage = (msg) => postedToMain.push(msg);
  sandbox.createRuntimeModule = async (moduleConfig) => {
    if (typeof moduleConfig.onRuntimeInitialized === 'function') {
      moduleConfig.onRuntimeInitialized.call(mockMod);
    }
    return mockMod;
  };

  vm.runInNewContext(loaderSource, sandbox, { filename: 'web/loader.js' });
  assert.equal(typeof sandbox.self.onmessage, 'function');

  // Boot the coordinator worker so runtimeMod + SaveSyncController attach.
  sandbox.self.onmessage({ data: { canvas: { width: 640, height: 360 } } });
  await new Promise((r) => setTimeout(r, 40));

  // Mutate MEMFS save container and send flush-userdata message from main thread.
  const saveDir = '/userdata/saves/default/local/c-legal';
  mockMod.FS.writeFile(`${saveDir}/b-accepted`, new Uint8Array([1, 2, 3, 4]), 1700005000000);
  mockMod.markChanged(saveDir);

  sandbox.self.onmessage({ data: { type: 'flush-userdata', id: 'req-42', reason: 'pagehide' } });
  await new Promise((r) => setTimeout(r, 30));

  const reply = postedToMain.find((m) => m && m.type === 'userdata-flushed' && m.id === 'req-42');
  assert.ok(reply, 'expected userdata-flushed reply from loader coordinator');
  assert.equal(reply.ok, true);
  assert.equal(reply.stats.lastPersistedGeneration, '1');
  assert.equal(reply.stats.persistedFiles, 1);
  assert.deepEqual((await sharedStore.getFile(`${saveDir}/b-accepted`)).data, new Uint8Array([1, 2, 3, 4]));

  // Clean teardown clears timer cleanly.
  sandbox.self.onmessage({ data: { type: 'destroy' } });
});

test('app_bootstrap.js posts { type: "flush-userdata" } to loaderWorker on hidden visibilitychange, pagehide, and beforeunload and cleans up on destroy', async () => {
  const docListeners = new Map();
  const winListeners = new Map();
  const postedToLoader = [];

  function makeEventTarget(map) {
    return {
      addEventListener(type, fn) {
        if (!map.has(type)) map.set(type, new Set());
        map.get(type).add(fn);
      },
      removeEventListener(type, fn) {
        if (map.has(type)) map.get(type).delete(fn);
      },
      dispatch(type, ev = { type }) {
        const set = map.get(type);
        if (!set) return;
        for (const fn of Array.from(set)) {
          fn(ev);
        }
      },
    };
  }

  const fakeCanvas = {
    width: 640,
    height: 360,
    clientWidth: 640,
    clientHeight: 360,
    dataset: {},
    style: {},
    addEventListener() {},
    removeEventListener() {},
    transferControlToOffscreen() {
      return { width: 640, height: 360 };
    },
  };

  const docTarget = makeEventTarget(docListeners);
  const documentObj = {
    ...docTarget,
    visibilityState: 'visible',
    hidden: false,
    pointerLockElement: null,
    getElementById: (id) => (id === 'game-canvas' ? fakeCanvas : null),
  };

  const winTarget = makeEventTarget(winListeners);
  const windowObj = {
    ...winTarget,
    document: documentObj,
    location: { origin: 'http://127.0.0.1:8080', href: 'http://127.0.0.1:8080/', search: '' },
    innerWidth: 1280,
    innerHeight: 720,
    devicePixelRatio: 1,
    crossOriginIsolated: true,
    navigator: {
      gpu: {
        requestAdapter: async () => ({
          features: new Set(),
          limits: {},
          requestDevice: async () => ({ destroy() {} }),
        }),
      },
    },
    WebAssembly: {
      Memory: class FakeMemory64 {
        constructor(desc) {
          if (typeof desc.initial !== 'bigint') throw new TypeError('need bigint');
        }
        type() {
          return { address: 'i64' };
        }
      },
    },
    Worker: class FakeLoaderWorker {
      constructor() {
        this.onmessage = null;
        this.onerror = null;
      }
      postMessage(msg) {
        postedToLoader.push(msg);
      }
      addEventListener() {}
      removeEventListener() {}
      terminate() {}
    },
    fetch: async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        hasWasmBinary: true,
        runtimeReady: true,
        runtimeUrl: '/dist/runtime.js',
        bootsetReady: true,
      }),
    }),
  };

  const fakeWorker = {
    onmessage: null,
    onerror: null,
    postMessage(msg) {
      postedToLoader.push(msg);
    },
    addEventListener() {},
    removeEventListener() {},
    terminate() {},
  };

  const handle = bootstrapPlayGta6App({
    canvas: fakeCanvas,
    windowObj,
    documentObj,
    navigatorObj: windowObj.navigator,
    fetchImpl: windowObj.fetch,
    storage: null,
    autoStart: false,
    createWasmWorker: () => fakeWorker,
  });

  await handle.start();
  postedToLoader.length = 0;

  // Visible visibilitychange must NOT flush userdata.
  documentObj.visibilityState = 'visible';
  documentObj.hidden = false;
  docTarget.dispatch('visibilitychange');
  assert.equal(postedToLoader.filter((m) => m && m.type === 'flush-userdata').length, 0);

  // Hidden visibilitychange MUST flush userdata.
  documentObj.visibilityState = 'hidden';
  documentObj.hidden = true;
  docTarget.dispatch('visibilitychange');
  const flushEvents1 = postedToLoader.filter((m) => m && m.type === 'flush-userdata');
  assert.equal(flushEvents1.length, 1);
  assert.equal(flushEvents1[0].reason, 'visibilitychange');

  // pagehide MUST flush userdata.
  winTarget.dispatch('pagehide', { type: 'pagehide' });
  const flushEvents2 = postedToLoader.filter((m) => m && m.type === 'flush-userdata');
  assert.equal(flushEvents2.length, 2);
  assert.equal(flushEvents2[1].reason, 'pagehide');

  // beforeunload MUST flush userdata.
  winTarget.dispatch('beforeunload', { type: 'beforeunload' });
  const flushEvents3 = postedToLoader.filter((m) => m && m.type === 'flush-userdata');
  assert.equal(flushEvents3.length, 3);
  assert.equal(flushEvents3[2].reason, 'beforeunload');

  // Cleanup on destroy removes pagehide and beforeunload listeners cleanly.
  await handle.destroy();
  assert.equal(winListeners.get('pagehide')?.size ?? 0, 0);
  assert.equal(winListeners.get('beforeunload')?.size ?? 0, 0);
});
