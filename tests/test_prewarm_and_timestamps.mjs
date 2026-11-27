import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const prewarmMod = require('../web/workers/prewarm_bridge.js');
const gpuTimerMod = require('../web/workers/gpu_timer.js');
const loaderMod = require('../web/loader.js');

const {
  PREWARM_MAGIC,
  PREWARM_LIST_VERSION_MODULES,
  PREWARM_LIST_VERSION_PIPELINES,
  PREWARM_LIST_HEADER_BYTES,
  PREWARM_LIST_ENTRY_BYTES,
  PREWARM_TASK_BYTES,
  PREWARM_PROGRESS_BYTES,
  PREWARM_KIND_GRAPHICS,
  PREWARM_KIND_COMPUTE,
  PREWARM_HARD_MAX_IN_FLIGHT,
  PREWARM_MSG_REQUEST,
  PREWARM_MSG_RESULT,
  formatShaderKeyHex,
  parseWarmList,
  serializeWarmList,
  decodePrewarmTasks,
  decodePrewarmProgress,
  PrewarmListStore,
  PrewarmBridge,
  executePrewarmTask,
  attachPrewarmBridge,
} = prewarmMod;

const {
  GPU_TIME_MAX_KEYS,
  GPU_TIME_SUPPORT,
  MAP_MODE,
  formatHashHex16,
  GpuTimeTable,
  GpuPassTimer,
} = gpuTimerMod;

const {
  buildRuntimeModuleConfig,
  createRuntimePrewarmBridge,
  loadPrewarmWarmList,
  readPrewarmProgress,
  savePrewarmWarmList,
} = loaderMod;

/**
 * In-memory faithful simulation of the C++ `playgta6::gpu::PipelinePrewarm` Wasm exports
 * (`runtime/wasm_boot.cpp` & `runtime/gpu/pipeline_prewarm.{h,cpp}`) over a real
 * `WebAssembly.Memory`, supporting both Wasm64 (`BigInt` pointers/u64) and Wasm32 (`Number` pointers).
 */
function createMockPrewarmWasmRuntime({ wasm64 = true, maxInFlight = 4 } = {}) {
  const memory = new WebAssembly.Memory({ initial: 4 });
  let bump = 1024;
  const liveAllocs = new Set();

  // Mirror C++ PipelinePrewarm state
  const slots = [];
  const queue = [];
  const observed = [];
  let inFlightCount = 0;
  let queuedCount = 0;
  let doneCount = 0;
  let failedCount = 0;
  let skippedCount = 0;
  let longestNs = 0n;
  let totalNs = 0n;

  const completeCalls = [];
  const takeCalls = [];
  const loadCalls = [];
  const saveCalls = [];

  function toOff(ptr) {
    if (wasm64 && typeof ptr !== 'bigint') {
      throw new TypeError(`Wasm64 expected BigInt pointer, got ${typeof ptr}`);
    }
    return Number(ptr);
  }

  function _malloc(size) {
    const n = Number(size);
    const aligned = Math.max(8, Math.ceil(n / 8) * 8);
    const off = bump;
    bump += aligned;
    if (bump > memory.buffer.byteLength) {
      const neededPages = Math.ceil((bump - memory.buffer.byteLength) / 65536);
      memory.grow(neededPages);
    }
    liveAllocs.add(off);
    return wasm64 ? BigInt(off) : off;
  }

  function _free(ptr) {
    const off = toOff(ptr);
    liveAllocs.delete(off);
  }

  function _playgta6_prewarm_load_list(dataPtr, size) {
    const off = toOff(dataPtr);
    const byteLen = Number(size);
    loadCalls.push({ off, byteLen });
    const slice = new Uint8Array(memory.buffer, off, byteLen).slice();
    const parsed = parseWarmList(slice);
    if (!parsed.ok) {
      return -1;
    }
    let added = 0;
    for (const e of parsed.entries) {
      const id = slots.length + 1;
      slots.push({
        id,
        kind: e.kind,
        key0: e.key0,
        key1: e.key1,
        state: 'queued',
      });
      queue.push(id);
      queuedCount += 1;
      added += 1;
    }
    return added;
  }

  function _playgta6_prewarm_take(outPtr, max) {
    const off = toOff(outPtr);
    const reqMax = Number(max);
    takeCalls.push({ off, reqMax });
    const dv = new DataView(memory.buffer, off, reqMax * PREWARM_TASK_BYTES);
    let written = 0;
    while (written < reqMax && inFlightCount < maxInFlight && queue.length > 0) {
      const id = queue.shift();
      const s = slots[id - 1];
      if (s.state !== 'queued') continue;
      s.state = 'in_flight';
      queuedCount -= 1;
      inFlightCount += 1;
      const base = written * PREWARM_TASK_BYTES;
      dv.setUint32(base, s.id, true);
      dv.setUint32(base + 4, s.kind, true);
      dv.setBigUint64(base + 8, s.key0, true);
      dv.setBigUint64(base + 16, s.key1, true);
      written += 1;
    }
    return written;
  }

  function _playgta6_prewarm_complete(id, ok, buildNs) {
    const taskId = Number(id);
    const okBool = Number(ok) !== 0;
    const ns = typeof buildNs === 'bigint' ? buildNs : BigInt(buildNs);
    completeCalls.push({ id: taskId, ok: okBool, buildNs: ns });
    if (taskId < 1 || taskId > slots.length) {
      return 0;
    }
    const s = slots[taskId - 1];
    if (s.state !== 'in_flight') {
      return 0;
    }
    inFlightCount -= 1;
    if (okBool) {
      s.state = 'done';
      doneCount += 1;
    } else {
      s.state = 'failed';
      failedCount += 1;
    }
    if (ns > longestNs) {
      longestNs = ns;
    }
    totalNs += ns;
    return 1;
  }

  function _playgta6_prewarm_progress(outPtr) {
    const off = toOff(outPtr);
    const dv = new DataView(memory.buffer, off, PREWARM_PROGRESS_BYTES);
    dv.setUint32(0, slots.length, true);
    dv.setUint32(4, queuedCount, true);
    dv.setUint32(8, inFlightCount, true);
    dv.setUint32(12, doneCount, true);
    dv.setUint32(16, failedCount, true);
    dv.setUint32(20, skippedCount, true);
    dv.setUint32(24, observed.length, true);
    dv.setUint32(28, 0, true);
    dv.setBigUint64(32, longestNs, true);
    dv.setBigUint64(40, totalNs, true);
  }

  function _playgta6_prewarm_save_list(outPtr, cap) {
    const off = toOff(outPtr);
    const capNum = Number(cap);
    saveCalls.push({ off, cap: capNum });
    const bytes = serializeWarmList(observed);
    if (off !== 0 && capNum >= bytes.byteLength) {
      new Uint8Array(memory.buffer, off, bytes.byteLength).set(bytes);
    }
    return wasm64 ? BigInt(bytes.byteLength) : bytes.byteLength;
  }

  function noteTitlePipeline(kind, key0, key1 = 0n) {
    observed.push({ kind, key0, key1 });
  }

  function noteSkipped(count = 1) {
    skippedCount += count;
  }

  const mod = {
    wasmMemory: memory,
    _malloc,
    _free,
    _playgta6_prewarm_load_list,
    _playgta6_prewarm_take,
    _playgta6_prewarm_complete,
    _playgta6_prewarm_progress,
    _playgta6_prewarm_save_list,
  };

  return {
    mod,
    memory,
    liveAllocs,
    completeCalls,
    takeCalls,
    loadCalls,
    saveCalls,
    noteTitlePipeline,
    noteSkipped,
  };
}

function createMockTimestampWebGpu({ supported = true, deferMap = false } = {}) {
  const querySets = [];
  const buffers = [];
  const resolveCalls = [];
  const copyCalls = [];
  const pendingMaps = [];
  let nextTimestamps = [];

  const device = {
    features: new Set(supported ? ['timestamp-query'] : []),
    createQuerySet(desc) {
      const qs = {
        desc: { ...desc },
        values: new BigUint64Array(desc.count),
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      querySets.push(qs);
      return qs;
    },
    createBuffer(desc) {
      const storage = new ArrayBuffer(desc.size);
      const buf = {
        size: desc.size,
        usage: desc.usage,
        _storage: storage,
        mapState: 'unmapped',
        destroyed: false,
        mapAsync(mode) {
          if (mode !== MAP_MODE.READ) {
            return Promise.reject(new Error('bad map mode'));
          }
          this.mapState = 'pending';
          if (deferMap) {
            return new Promise((resolve, reject) => {
              pendingMaps.push({
                resolve: () => {
                  this.mapState = 'mapped';
                  resolve();
                },
                reject: (err) => {
                  this.mapState = 'unmapped';
                  reject(err);
                },
              });
            });
          }
          return Promise.resolve().then(() => {
            this.mapState = 'mapped';
          });
        },
        getMappedRange(offset = 0, size = this.size - offset) {
          return this._storage.slice(offset, offset + size);
        },
        unmap() {
          this.mapState = 'unmapped';
        },
        destroy() {
          this.destroyed = true;
        },
      };
      buffers.push(buf);
      return buf;
    },
    createCommandEncoder() {
      return {
        resolveQuerySet(querySet, firstQuery, queryCount, destination, destinationOffset) {
          resolveCalls.push({ querySet, firstQuery, queryCount, destination, destinationOffset });
          for (let i = 0; i < queryCount; i += 1) {
            querySet.values[firstQuery + i] =
              i < nextTimestamps.length ? BigInt(nextTimestamps[i]) : 0n;
          }
          const dst = new BigUint64Array(destination._storage, destinationOffset, queryCount);
          for (let i = 0; i < queryCount; i += 1) {
            dst[i] = querySet.values[firstQuery + i];
          }
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          copyCalls.push({ source, sourceOffset, destination, destinationOffset, size });
          new Uint8Array(destination._storage, destinationOffset, size).set(
            new Uint8Array(source._storage, sourceOffset, size),
          );
        },
      };
    },
  };

  return {
    device,
    querySets,
    buffers,
    resolveCalls,
    copyCalls,
    pendingMaps,
    setTimestamps(list) {
      nextTimestamps = list.slice();
    },
  };
}

// ============================================================================
// 1. PG6W Binary Codec & Malformed-List Handling (#1613)
// ============================================================================

test('PG6W v1 serialize/parse round-trips exact 16-byte header + 24-byte entries in first-use order', () => {
  const entries = [
    { kind: PREWARM_KIND_GRAPHICS, key0: 0xfedcba9876543210n, key1: 0x0123456789abcdefn },
    { kind: PREWARM_KIND_COMPUTE, key0: 0x1122334455667788n, key1: 0n },
    { kind: 'graphics', key0: '0x00000000deadbeef', key1: 0n },
  ];

  const bytes = serializeWarmList(entries);
  assert.equal(
    bytes.byteLength,
    PREWARM_LIST_HEADER_BYTES + entries.length * PREWARM_LIST_ENTRY_BYTES,
  );
  assert.equal(String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]), PREWARM_MAGIC);

  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  assert.equal(dv.getUint32(4, true), PREWARM_LIST_VERSION_MODULES);
  assert.equal(dv.getUint32(8, true), 3);
  assert.equal(dv.getUint32(12, true), 0);

  // Verify entry 0 binary layout: u8 kind @ +0, u8[7] zero @ +1..7, u64 key0 @ +8, u64 key1 @ +16.
  const e0Off = PREWARM_LIST_HEADER_BYTES;
  assert.equal(bytes[e0Off], PREWARM_KIND_GRAPHICS);
  for (let b = 1; b < 8; b += 1) {
    assert.equal(bytes[e0Off + b], 0);
  }
  assert.equal(dv.getBigUint64(e0Off + 8, true), 0xfedcba9876543210n);
  assert.equal(dv.getBigUint64(e0Off + 16, true), 0x0123456789abcdefn);

  const parsed = parseWarmList(bytes);
  assert.equal(parsed.ok, true);
  assert.equal(parsed.version, PREWARM_LIST_VERSION_MODULES);
  assert.equal(parsed.count, 3);
  assert.deepEqual(parsed.entries, [
    { kind: 0, key0: 0xfedcba9876543210n, key1: 0x0123456789abcdefn, detail: null },
    { kind: 1, key0: 0x1122334455667788n, key1: 0n, detail: null },
    { kind: 0, key0: 0xdeadbeefn, key1: 0n, detail: null },
  ]);
});

test('PG6W v2 serialize/parse preserves opaque pipeline detail bytes and rejects malformed lists', () => {
  const detail = new Uint8Array([0xaa, 0xbb, 0xcc, 0xdd, 0xee]);
  const bytes = serializeWarmList([
    { kind: PREWARM_KIND_GRAPHICS, key0: 0x101n, key1: 0x202n, detail },
    { kind: PREWARM_KIND_COMPUTE, key0: 0x303n, key1: 0n },
  ]);
  const parsed = parseWarmList(bytes);
  assert.equal(parsed.ok, true);
  assert.equal(parsed.version, PREWARM_LIST_VERSION_PIPELINES);
  assert.equal(parsed.count, 2);
  assert.deepEqual(Array.from(parsed.entries[0].detail), [0xaa, 0xbb, 0xcc, 0xdd, 0xee]);
  assert.equal(parsed.entries[1].detail, null);

  // Malformed cases matching C++ PipelinePrewarm::load_warm_list checks:
  assert.equal(parseWarmList(new Uint8Array(8)).ok, false); // too short

  const badMagic = bytes.slice();
  badMagic[0] = 0x58; // 'X'
  assert.deepEqual(parseWarmList(badMagic), { ok: false, error: 'warm list bad magic' });

  const badVersion = serializeWarmList([{ kind: 0, key0: 1n }]);
  new DataView(badVersion.buffer).setUint32(4, 99, true);
  assert.deepEqual(parseWarmList(badVersion), { ok: false, error: 'warm list unsupported version' });

  const truncated = serializeWarmList([{ kind: 0, key0: 1n }, { kind: 1, key0: 2n }]).slice(0, 30);
  assert.deepEqual(parseWarmList(truncated), { ok: false, error: 'warm list truncated' });

  const badKind = serializeWarmList([{ kind: 0, key0: 1n }]);
  badKind[PREWARM_LIST_HEADER_BYTES] = 7;
  assert.deepEqual(parseWarmList(badKind), { ok: false, error: 'warm list bad entry' });

  const zeroKey = serializeWarmList([{ kind: 0, key0: 1n }]);
  new DataView(zeroKey.buffer).setBigUint64(PREWARM_LIST_HEADER_BYTES + 8, 0n, true);
  assert.deepEqual(parseWarmList(zeroKey), { ok: false, error: 'warm list bad entry' });

  const truncatedV2Detail = bytes.slice(0, bytes.byteLength - 2);
  assert.deepEqual(parseWarmList(truncatedV2Detail), { ok: false, error: 'warm list truncated' });
});

// ============================================================================
// 2. PrewarmListStore OPFS / IndexedDB / Memory Fallback Persistence
// ============================================================================

test('PrewarmListStore supports OPFS, IndexedDB, and graceful in-memory fallback + rejects corrupt bytes', async () => {
  const sample = serializeWarmList([
    { kind: PREWARM_KIND_GRAPHICS, key0: 0xaaaabbbbccccddddn, key1: 0x1111222233334444n },
  ]);

  // 1) Graceful in-memory fallback when neither OPFS nor IndexedDB exists.
  const memStore = new PrewarmListStore({ navigator: null, indexedDB: null });
  assert.equal(await memStore.load('title'), null);
  const saveMem = await memStore.save('title', sample);
  assert.equal(saveMem.ok, true);
  assert.equal(saveMem.backend, 'memory');
  const loadedMem = await memStore.load('title');
  assert.deepEqual(Array.from(loadedMem), Array.from(sample));

  // Malformed save is refused without clobbering existing valid entry.
  const badSave = await memStore.save('title', new Uint8Array([1, 2, 3, 4]));
  assert.equal(badSave.ok, false);
  assert.deepEqual(Array.from(await memStore.load('title')), Array.from(sample));

  // 2) Mock OPFS backend.
  const opfsFiles = new Map();
  const mockNavigator = {
    storage: {
      async getDirectory() {
        return {
          async getDirectoryHandle(dirName) {
            return {
              async getFileHandle(fileName, { create } = {}) {
                const full = `${dirName}/${fileName}`;
                if (!create && !opfsFiles.has(full)) {
                  throw new Error('NotFoundError');
                }
                return {
                  async getFile() {
                    const data = opfsFiles.get(full);
                    return { async arrayBuffer() { return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength); } };
                  },
                  async createWritable() {
                    let written = null;
                    return {
                      async write(bytes) { written = new Uint8Array(bytes); },
                      async close() { opfsFiles.set(full, written); },
                    };
                  },
                };
              },
              async removeEntry(fileName) {
                opfsFiles.delete(`${dirName}/${fileName}`);
              },
            };
          },
        };
      },
    },
  };
  const opfsStore = new PrewarmListStore({ navigator: mockNavigator, indexedDB: null });
  const saveOpfs = await opfsStore.save('gta6', sample);
  assert.equal(saveOpfs.ok, true);
  assert.equal(saveOpfs.backend, 'opfs');
  const loadedOpfs = await opfsStore.load('gta6');
  assert.deepEqual(Array.from(loadedOpfs), Array.from(sample));
  assert.equal(opfsStore.lastBackend, 'opfs');

  // Corrupt payload in storage returns null and increments corruptReads.
  opfsFiles.set('playgta6-prewarm/gta6.pg6w', new Uint8Array([0x00, 0x01, 0x02]));
  assert.equal(await opfsStore.load('gta6'), null);
  assert.equal(opfsStore.corruptReads, 1);

  // 3) Mock IndexedDB backend when OPFS is absent.
  const idbData = new Map();
  const mockIndexedDB = {
    open() {
      const req = {
        result: {
          objectStoreNames: { contains: () => true },
          createObjectStore() {},
          close() {},
          transaction() {
            const tx = {
              oncomplete: null,
              onerror: null,
              onabort: null,
              objectStore() {
                return {
                  get(key) {
                    const r = { result: idbData.get(key), onsuccess: null, onerror: null };
                    queueMicrotask(() => r.onsuccess && r.onsuccess());
                    return r;
                  },
                  put(val, key) {
                    idbData.set(key, val);
                    queueMicrotask(() => tx.oncomplete && tx.oncomplete());
                  },
                  delete(key) {
                    idbData.delete(key);
                    queueMicrotask(() => tx.oncomplete && tx.oncomplete());
                  },
                };
              },
            };
            return tx;
          },
        },
        onupgradeneeded: null,
        onsuccess: null,
        onerror: null,
      };
      queueMicrotask(() => {
        if (req.onupgradeneeded) req.onupgradeneeded();
        if (req.onsuccess) req.onsuccess();
      });
      return req;
    },
  };
  const idbStore = new PrewarmListStore({ navigator: null, indexedDB: mockIndexedDB });
  const saveIdb = await idbStore.save('slot1', sample);
  assert.equal(saveIdb.ok, true);
  assert.equal(saveIdb.backend, 'indexeddb');
  const loadedIdb = await idbStore.load('slot1');
  assert.deepEqual(Array.from(loadedIdb), Array.from(sample));
});

// ============================================================================
// 3. PrewarmBridge Concurrency Cap (<= 4), Graphics/Compute/Errors, Progress & Save
// ============================================================================

test('PrewarmBridge drains playgta6_prewarm_take with <= 4 concurrency, reports ok/fail + build_ns, decodes 48-byte progress, and saves PG6W list', async () => {
  const rt = createMockPrewarmWasmRuntime({ wasm64: true, maxInFlight: 4 });

  // Build a 7-entry PG6W v1 warm list (mix of graphics and compute).
  const initialList = serializeWarmList([
    { kind: PREWARM_KIND_GRAPHICS, key0: 0x1001n, key1: 0x2001n },
    { kind: PREWARM_KIND_COMPUTE, key0: 0x3001n, key1: 0n },
    { kind: PREWARM_KIND_GRAPHICS, key0: 0x1002n, key1: 0x2002n },
    { kind: PREWARM_KIND_COMPUTE, key0: 0x3002n, key1: 0n },
    { kind: PREWARM_KIND_GRAPHICS, key0: 0x1003n, key1: 0n }, // VS-only
    { kind: PREWARM_KIND_COMPUTE, key0: 0x3003n, key1: 0n }, // will fail via exception
    { kind: PREWARM_KIND_GRAPHICS, key0: 0x1004n, key1: 0x2004n }, // will fail via { ok: false }
  ]);

  let currentInFlight = 0;
  let observedMaxInFlight = 0;
  let isIdle = false;
  let fakeNowMs = 1000;
  const pendingResolvers = [];

  const bridge = new PrewarmBridge({
    module: rt.mod,
    maxInFlight: 12, // Must still clamp to hard cap <= 4!
    now: () => fakeNowMs,
    isIdle: () => isIdle,
    storeOptions: { navigator: null, indexedDB: null },
    buildTask: (task) => {
      currentInFlight += 1;
      if (currentInFlight > observedMaxInFlight) {
        observedMaxInFlight = currentInFlight;
      }
      return new Promise((resolve, reject) => {
        pendingResolvers.push({ task, resolve, reject });
      });
    },
  });

  assert.equal(bridge.available, true);
  assert.equal(bridge.maxInFlight, PREWARM_HARD_MAX_IN_FLIGHT);

  // Malformed warm list returns -1 without throwing.
  const badLoad = bridge.loadWarmList(new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]));
  assert.equal(badLoad.ok, false);
  assert.equal(badLoad.queued, -1);

  // Valid PG6W v1 load queues all 7 tasks.
  const goodLoad = bridge.loadWarmList(initialList);
  assert.equal(goodLoad.ok, true);
  assert.equal(goodLoad.queued, 7);

  // While busy (isIdle === false), pump() dispatches 0 tasks.
  assert.equal(bridge.pump(), 0);
  assert.equal(bridge.inFlightCount, 0);

  // Once idle, first pump takes exactly 4 tasks (bounded concurrency cap).
  isIdle = true;
  assert.equal(bridge.pump(), 4);
  assert.equal(bridge.inFlightCount, 4);
  // A second pump while 4 are in flight takes 0 additional tasks.
  assert.equal(bridge.pump(), 0);

  // Flush microtasks so buildTask promises start.
  await Promise.resolve();
  assert.equal(observedMaxInFlight, 4);
  assert.equal(pendingResolvers.length, 4);

  // Complete first 4 tasks with deterministic durations (5ms, 12ms, 3ms, 8ms).
  const durationsMs = [5, 12, 3, 8];
  for (let i = 0; i < 4; i += 1) {
    const item = pendingResolvers.shift();
    fakeNowMs += durationsMs[i];
    currentInFlight -= 1;
    item.resolve({ ok: true, buildNs: durationsMs[i] * 1_000_000 });
  }
  await new Promise((r) => setTimeout(r, 0));

  // Pump remaining 3 tasks (ids 5, 6, 7).
  assert.equal(bridge.pump(), 3);
  await Promise.resolve();
  assert.equal(pendingResolvers.length, 3);

  // Task 5 (VS-only graphics): succeeds in 2ms.
  const t5 = pendingResolvers.shift();
  assert.equal(t5.task.kind, PREWARM_KIND_GRAPHICS);
  assert.equal(t5.task.key1, 0n);
  currentInFlight -= 1;
  t5.resolve({ ok: true, buildNs: 2_000_000 });

  // Task 6 (compute): throws an exception -> must call complete(id, 0, ns).
  const t6 = pendingResolvers.shift();
  assert.equal(t6.task.kind, PREWARM_KIND_COMPUTE);
  fakeNowMs += 4;
  currentInFlight -= 1;
  t6.reject(new Error('WGSL compile error'));

  // Task 7 (graphics): returns { ok: false, buildNs: 1_500_000 }.
  const t7 = pendingResolvers.shift();
  currentInFlight -= 1;
  t7.resolve({ ok: false, buildNs: 1_500_000, error: 'pipeline validation failed' });

  await new Promise((r) => setTimeout(r, 0));

  assert.equal(observedMaxInFlight, 4);
  assert.equal(bridge.inFlightCount, 0);
  assert.equal(rt.completeCalls.length, 7);
  assert.equal(rt.completeCalls.filter((c) => c.ok).length, 5);
  assert.equal(rt.completeCalls.filter((c) => !c.ok).length, 2);

  // Record observed title pipelines + skipped count and verify 48-byte progress struct.
  rt.noteTitlePipeline(PREWARM_KIND_GRAPHICS, 0x1001n, 0x2001n);
  rt.noteTitlePipeline(PREWARM_KIND_COMPUTE, 0x3001n, 0n);
  rt.noteSkipped(1);

  const progress = bridge.readProgress();
  assert.deepEqual(progress, {
    total: 7,
    queued: 0,
    in_flight: 0,
    done: 5,
    failed: 2,
    skipped: 1,
    observed: 2,
    reserved: 0,
    longest_ns: 12_000_000,
    total_ns: rt.completeCalls.reduce((acc, c) => acc + Number(c.buildNs), 0),
  });
  // Verify JSON-safe serialization for loader diagnostics.
  assert.doesNotThrow(() => JSON.stringify(bridge.getDiagnostics()));

  // Save PG6W warm list via playgta6_prewarm_save_list and persist to store.
  const saveRes = await bridge.saveWarmList({ persist: true, key: 'run-1' });
  assert.equal(saveRes.ok, true);
  assert.equal(saveRes.persisted, true);
  assert.equal(saveRes.version, PREWARM_LIST_VERSION_MODULES);
  assert.equal(saveRes.entryCount, 2);
  assert.equal(saveRes.byteLength, 16 + 2 * 24);

  // Re-load persisted warm list into a fresh runtime instance (next-boot round-trip).
  const rt2 = createMockPrewarmWasmRuntime({ wasm64: true });
  const bridge2 = new PrewarmBridge({
    module: rt2.mod,
    store: bridge.store,
    storeKey: 'run-1',
  });
  const loaded = await bridge2.loadPersistedList();
  assert.equal(loaded.ok, true);
  assert.equal(loaded.found, true);
  assert.equal(loaded.queued, 2);

  // Zero leaked scratch allocations in Wasm linear memory.
  assert.equal(rt.liveAllocs.size, 0);
  assert.equal(rt2.liveAllocs.size, 0);

  bridge.destroy();
  bridge2.destroy();
});

// ============================================================================
// 4. Worker Message Contract (attachPrewarmBridge) & web/loader.js Integration
// ============================================================================

test('attachPrewarmBridge + loader.js buildRuntimeModuleConfig wire end-to-end graphics/compute prewarm & lifecycle', async () => {
  const rt = createMockPrewarmWasmRuntime({ wasm64: false });
  const createdShaders = [];
  const createdComputePipelines = [];

  const shaderPack = new Map([
    [formatShaderKeyHex(0xaaa1n), { wgsl: '@vertex fn main() -> @builtin(position) vec4f { return vec4f(0.0); }' }],
    [formatShaderKeyHex(0xbbb1n), { wgsl: '@fragment fn main() -> @location(0) vec4f { return vec4f(1.0); }' }],
    [formatShaderKeyHex(0xccc1n), { wgsl: '@compute @workgroup_size(1) fn main() {}' }],
  ]);

  const mockDevice = {
    createShaderModule(desc) {
      createdShaders.push(desc.label);
      return {
        label: desc.label,
        async getCompilationInfo() {
          return { messages: [] };
        },
      };
    },
    async createComputePipelineAsync(desc) {
      createdComputePipelines.push(desc.label);
      return { label: desc.label };
    },
  };

  // Wire a mock MessagePort pair between loader config and worker-side attachPrewarmBridge.
  const workerResponses = [];
  let configRef = null;
  const workerPrewarm = attachPrewarmBridge(
    { device: mockDevice, shaderPack },
    {
      maxInFlight: 4,
      postMessage: (reply) => {
        workerResponses.push(reply);
        if (configRef) {
          configRef.handlePrewarmMessage(reply);
        }
      },
    },
  );

  const mockGpuWorker = {
    postMessage(msg) {
      workerPrewarm.handleMessage(msg);
    },
  };

  // Pre-seed store with 2 tasks (1 graphics VS+FS, 1 compute).
  const store = new PrewarmListStore({ navigator: null, indexedDB: null });
  await store.save(
    'boot-key',
    serializeWarmList([
      { kind: PREWARM_KIND_GRAPHICS, key0: 0xaaa1n, key1: 0xbbb1n },
      { kind: PREWARM_KIND_COMPUTE, key0: 0xccc1n, key1: 0n },
    ]),
  );

  const config = buildRuntimeModuleConfig({
    gpuWorker: mockGpuWorker,
    prewarmStore: store,
    prewarmStoreKey: 'boot-key',
    prewarmPollIntervalMs: 10,
    prewarmIsIdle: () => true,
  });
  configRef = config;

  // Trigger onRuntimeInitialized -> attaches PrewarmBridge, loads persisted PG6W list, dispatches to worker.
  config.onRuntimeInitialized.call(rt.mod);

  // Wait for async load -> dispatch -> worker compile -> prewarm-task-result -> complete().
  for (let i = 0; i < 40 && rt.completeCalls.length < 2; i += 1) {
    await new Promise((r) => setTimeout(r, 5));
  }

  assert.equal(rt.completeCalls.length, 2);
  assert.equal(rt.completeCalls[0].ok, true);
  assert.equal(rt.completeCalls[1].ok, true);
  assert.equal(createdShaders.length, 3); // VS + FS + CS
  assert.equal(createdComputePipelines.length, 1);
  assert.equal(workerResponses.length, 2);
  assert.equal(workerResponses[0].type, PREWARM_MSG_RESULT);

  const progress = config.getPrewarmProgress();
  assert.equal(progress.total, 2);
  assert.equal(progress.done, 2);
  assert.equal(progress.failed, 0);

  const diag = config.getPrewarmDiagnostics();
  assert.equal(diag.available, true);
  assert.equal(diag.tasksCompleted, 2);
  assert.equal(diag.tasksSucceeded, 2);

  // Record a new observed pipeline and trigger onExit -> persists updated warm list.
  rt.noteTitlePipeline(PREWARM_KIND_COMPUTE, 0xddd1n, 0n);
  config.onExit();
  await new Promise((r) => setTimeout(r, 10));

  const savedOnExit = await store.load('boot-key');
  assert.ok(savedOnExit);
  const parsedExit = parseWarmList(savedOnExit);
  assert.equal(parsedExit.ok, true);
  assert.equal(parsedExit.count, 1);
  assert.equal(parsedExit.entries[0].key0, 0xddd1n);

  // Direct loader.js helper smoke checks.
  const rtDirect = createMockPrewarmWasmRuntime({ wasm64: true });
  assert.equal(loadPrewarmWarmList(rtDirect.mod, savedOnExit), 1);
  assert.equal(readPrewarmProgress(rtDirect.mod).queued, 1);
  rtDirect.noteTitlePipeline(PREWARM_KIND_GRAPHICS, 0x9999n, 0x8888n);
  const directSaved = savePrewarmWarmList(rtDirect.mod);
  assert.ok(directSaved);
  assert.equal(parseWarmList(directSaved).entries[0].key0, 0x9999n);

  const directBridge = createRuntimePrewarmBridge(rtDirect.mod);
  assert.equal(directBridge.available, true);
  directBridge.destroy();
  workerPrewarm.detach();
});

// ============================================================================
// 5. gpu_timer.js Timestamp Telemetry Bridge (#1608)
// ============================================================================

test('GpuPassTimer generates timestampWrites, resolves async readbacks, tracks gpu_time per-pipeline stats, and fires onComplete', async () => {
  const env = createMockTimestampWebGpu({ supported: true, deferMap: true });
  const completions = [];

  const timer = new GpuPassTimer({
    device: env.device,
    maxPassesPerFrame: 8,
    readbackSlots: 3,
    onComplete: (event) => completions.push(event),
  });

  let extraCalls = 0;
  const unsubscribe = timer.onComplete(() => {
    extraCalls += 1;
  });

  timer.beginFrame();
  const csDesc = timer.getPassTimestampWrites('cull_compute', 'compute', 0x00000000cafebaben);
  const rsDesc = timer.beginPass({
    label: 'gbuffer_render',
    kind: 'render',
    hash: 0x00000000deadbeefn,
  });
  timer.endPass();

  assert.deepEqual(csDesc, {
    querySet: env.querySets[0],
    beginningOfPassWriteIndex: 0,
    endOfPassWriteIndex: 1,
  });
  assert.deepEqual(rsDesc, {
    querySet: env.querySets[0],
    beginningOfPassWriteIndex: 2,
    endOfPassWriteIndex: 3,
  });

  // Frame timestamps: compute = 1.5ms (1_500_000 ns), render = 3.25ms (3_250_000 ns)
  env.setTimestamps([
    10_000_000n,
    11_500_000n,
    20_000_000n,
    23_250_000n,
  ]);

  const encoder = env.device.createCommandEncoder();
  assert.equal(timer.resolveFrame(encoder), true);

  // Start non-blocking async flush; frame loop does not stall.
  const flushPromise = timer.flushReadbacks();
  assert.equal(env.pendingMaps.length, 1);
  assert.equal(completions.length, 0);

  env.pendingMaps.shift().resolve();
  const flushed = await flushPromise;
  assert.equal(flushed.length, 1);
  assert.equal(flushed[0].totalGpuMs, 4.75);

  // Verify onComplete hook payload (ns + ms + kind + hex16 shader hash).
  assert.equal(completions.length, 1);
  assert.equal(extraCalls, 1);
  assert.equal(completions[0].mode, 'gpu');
  assert.equal(completions[0].totalGpuNs, 4_750_000);
  assert.equal(completions[0].totalGpuMs, 4.75);
  assert.deepEqual(completions[0].passes, [
    {
      name: 'cull_compute',
      kind: 'compute',
      hash: '0x00000000cafebabe',
      durationNs: 1_500_000,
      durationMs: 1.5,
    },
    {
      name: 'gbuffer_render',
      kind: 'render',
      hash: '0x00000000deadbeef',
      durationNs: 3_250_000,
      durationMs: 3.25,
    },
  ]);

  // Verify runtime/gpu/gpu_time.h GpuTimeTable::json() parity.
  const gpuTime = timer.getGpuTimeStats();
  assert.deepEqual(gpuTime, {
    available: true,
    support: GPU_TIME_SUPPORT.AVAILABLE,
    frames_measured: 1,
    frames_skipped: 0,
    passes_skipped: 0,
    frame_gpu_ns: 4_750_000,
    total_ns: 4_750_000,
    tracked: 2,
    other: { passes: 0, total_ns: 0 },
    top: [
      {
        hash: '0x00000000deadbeef',
        kind: 'render',
        passes: 1,
        total_ns: 3_250_000,
        max_ns: 3_250_000,
        avg_ns: 3_250_000,
      },
      {
        hash: '0x00000000cafebabe',
        kind: 'compute',
        passes: 1,
        total_ns: 1_500_000,
        max_ns: 1_500_000,
        avg_ns: 1_500_000,
      },
    ],
  });
  assert.deepEqual(JSON.parse(timer.getGpuTimeJson()), gpuTime);

  // Unsubscribe prevents further extraCalls on direct complete().
  unsubscribe();
  const directRes = timer.complete(
    new BigUint64Array([
      100n,
      600n, // 500 ticks * 2.0 period = 1000 ns
      0n,
      500n, // zero start -> skipped (matches gpu_time.cpp!)
      900n,
      800n, // backwards -> skipped
    ]),
    {
      periodNs: 2.0,
      passes: [
        { name: 'copy_pass', kind: 'copy', hash: 0x42n },
        { name: 'unwritten', kind: 'render', hash: 0x43n },
        { name: 'backwards', kind: 'compute', hash: 0x44n },
      ],
    },
  );
  assert.equal(extraCalls, 1);
  assert.equal(completions.length, 2);
  assert.equal(directRes.measuredPasses, 1);
  assert.equal(directRes.skippedPasses, 2);
  assert.equal(directRes.totalGpuNs, 1000);

  const afterDirect = timer.getGpuTimeStats();
  assert.equal(afterDirect.frames_measured, 2);
  assert.equal(afterDirect.passes_skipped, 2);
  assert.equal(afterDirect.tracked, 3);
  assert.equal(afterDirect.frame_gpu_ns, 4_751_000);

  timer.destroy();
});

test('GpuPassTimer and GpuTimeTable degrade gracefully when timestamp-query is unavailable without inventing numbers', async () => {
  const env = createMockTimestampWebGpu({ supported: false });
  const completions = [];
  const timer = new GpuPassTimer({
    device: env.device,
    onComplete: (e) => completions.push(e),
  });

  assert.equal(timer.supported, false);
  assert.equal(timer.getPassTimestampWrites('shadow', 'render', 0x1234n), undefined);
  assert.equal(timer.resolveFrame(env.device.createCommandEncoder()), false);
  assert.deepEqual(await timer.flushReadbacks(), []);
  assert.equal(completions.length, 0);

  const report = timer.getGpuTimeStats();
  assert.equal(report.available, false);
  assert.equal(report.support, GPU_TIME_SUPPORT.UNAVAILABLE);
  assert.equal(report.reason, 'adapter lacks timestamp-query');
  assert.equal(report.frames_measured, 0);
  assert.equal(report.total_ns, 0);
  assert.deepEqual(report.top, []);

  // Verify GpuTimeTable bounded key cap overflows into `other`.
  const table = new GpuTimeTable({ maxKeys: 2 });
  assert.equal(GPU_TIME_MAX_KEYS, 1024);
  assert.equal(formatHashHex16(0xabn), '0x00000000000000ab');
  table.record(1n, 'render', 1000);
  table.record(2n, 'compute', 2000);
  table.record(3n, 'render', 500); // overflows into other
  const snap = table.getSnapshot();
  assert.equal(snap.tracked, 2);
  assert.deepEqual(snap.other, { passes: 1, total_ns: 500 });
  assert.equal(snap.total_ns, 3500);

  timer.destroy();
});

test('PrewarmBridge and attachPrewarmBridge yield behind live title submits/compiles and report yieldMs (#2054)', async () => {
  const rt = createMockPrewarmWasmRuntime({ wasm64: true });
  let fakeNow = 100;
  let idle = false;
  const built = [];

  const bridge = new PrewarmBridge({
    module: rt.mod,
    isIdle: () => idle,
    now: () => fakeNow,
    buildTask: async (task) => {
      built.push(task.id);
      return { ok: true, buildNs: 250_000 };
    },
  });

  const warmBytes = serializeWarmList([
    { kind: PREWARM_KIND_COMPUTE, key0: 0xfeedface00000001n, key1: 0n },
  ]);
  assert.equal(bridge.loadWarmList(warmBytes).queued, 1);

  assert.equal(bridge.pump(), 0);
  fakeNow = 145;
  assert.equal(bridge.pump(), 0);
  assert.equal(built.length, 0);
  assert.equal(bridge.getDiagnostics().yieldCount, 2);
  assert.equal(bridge.getDiagnostics().yieldMs, 45);

  idle = true;
  fakeNow = 160;
  assert.equal(bridge.pump(), 1);
  await new Promise((r) => setTimeout(r, 10));
  assert.deepEqual(built, [1]);
  assert.equal(bridge.getDiagnostics().yieldCount, 2);
  assert.equal(bridge.getDiagnostics().yieldMs, 60);
  bridge.destroy();

  // Worker-side attachPrewarmBridge defaults to yielding while processor has live submits/compiles.
  let workerNow = 500;
  const fakeProcessor = {
    _hasPendingUploadCommands: true,
    _passCommands: [{ type: 'draw' }],
    _inFlightPipelineCompiles: 1,
    device: {
      createShaderModule: ({ code }) => ({ code }),
    },
    shaderPack: {
      getShader: (key) => ({ code: `// ${key.toString(16)}`, entryPoint: 'main' }),
    },
  };
  const workerBridge = attachPrewarmBridge(fakeProcessor, {
    now: () => workerNow,
    setTimeout: () => null,
  });

  const replies = [];
  const taskPromise = workerBridge.executeTask(
    { id: 7, kind: PREWARM_KIND_COMPUTE, key0: 0x42n, key1: 0n },
    (m) => replies.push(m),
  );
  assert.equal(workerBridge.getStats().queued, 1);
  assert.equal(workerBridge.getStats().inFlight, 0);
  assert.equal(workerBridge.getStats().yieldCount, 1);

  workerNow = 535;
  assert.equal(workerBridge.getStats().yieldMs, 35);

  // Clear live title work and resume draining.
  fakeProcessor._hasPendingUploadCommands = false;
  fakeProcessor._passCommands = [];
  fakeProcessor._inFlightPipelineCompiles = 0;
  workerNow = 550;
  workerBridge.pump();

  const res = await taskPromise;
  assert.equal(res.ok, true);
  assert.equal(replies.length, 1);
  assert.equal(workerBridge.getStats().completed, 1);
  assert.equal(workerBridge.getStats().yieldCount, 1);
  assert.equal(workerBridge.getStats().yieldMs, 50);
  workerBridge.detach();
});
