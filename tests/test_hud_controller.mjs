import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  HudController,
  HUD_ACTIONS,
  DEFAULT_FILENAMES,
} from '../web/ui/hud_controller.js';
import {
  BrowserStorageManager,
  DEFAULT_OPFS_CACHE_DIR,
  SNAPSHOT_FORMAT,
  SNAPSHOT_VERSION,
} from '../web/storage/storage_manager.js';
import { CrashReporter } from '../web/telemetry/crash_report.js';
import { PerfMonitor } from '../web/telemetry/perf_monitor.js';
import { InputProfileManager } from '../web/input/input_profiles.js';
import {
  RuntimeSessionController,
  SESSION_STATE,
  SESSION_MODE,
} from '../web/runtime/session_controller.js';

const require = createRequire(import.meta.url);
const { GpuFrameRecorder, OPCODES } = require('../web/workers/gpu_recorder.js');
const { GpuBootPanelController, EXPORT_FORMAT } = require('../web/ui/gpu_boot_panel.js');

// --- Lightweight Mocks for Browser APIs (DOM, OPFS, IndexedDB, localStorage, fetch) ---

function createMockHud() {
  const logs = [];
  const statuses = [];
  const metrics = [];
  return {
    logs,
    statuses,
    metrics,
    appendLog(message, level = 'info') {
      logs.push({ message, level });
    },
    setStatus(text, level = 'info') {
      statuses.push({ text, level });
    },
    updateMetrics(payload) {
      metrics.push({ ...payload });
    },
  };
}

function createMockOpfsDirHandle(name = '', initialEntries = null) {
  const entries = new Map();
  if (initialEntries) {
    for (const [k, v] of Object.entries(initialEntries)) {
      entries.set(k, v instanceof Uint8Array ? v.slice() : v);
    }
  }

  return {
    kind: 'directory',
    name,
    _entries: entries,
    async getDirectoryHandle(dirName, opts = {}) {
      const existing = entries.get(dirName);
      if (existing && existing.kind === 'directory') {
        return existing;
      }
      if (!opts.create) {
        const err = new Error(`NotFoundError: ${dirName}`);
        err.name = 'NotFoundError';
        throw err;
      }
      const child = createMockOpfsDirHandle(dirName);
      entries.set(dirName, child);
      return child;
    },
    async *entries() {
      for (const [entryName, value] of entries.entries()) {
        if (value && value.kind === 'directory') {
          yield [entryName, value];
        } else {
          yield [
            entryName,
            {
              kind: 'file',
              name: entryName,
              async getFile() {
                return {
                  size: value.byteLength,
                  async arrayBuffer() {
                    return value.slice().buffer;
                  },
                };
              },
            },
          ];
        }
      }
    },
    async removeEntry(entryName, opts = {}) {
      if (!entries.has(entryName)) {
        const err = new Error(`NotFoundError: ${entryName}`);
        err.name = 'NotFoundError';
        throw err;
      }
      const target = entries.get(entryName);
      if (target && target.kind === 'directory' && target._entries.size > 0 && !opts.recursive) {
        const err = new Error(`InvalidModificationError: ${entryName} is not empty`);
        err.name = 'InvalidModificationError';
        throw err;
      }
      entries.delete(entryName);
    },
  };
}

function createMockIndexedDB() {
  const databases = new Map();

  function makeStore(recordsMap) {
    return {
      put(record) {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        queueMicrotask(() => {
          recordsMap.set(record.path, {
            ...record,
            data: record.data instanceof Uint8Array ? record.data.slice() : record.data,
          });
          req.result = record.path;
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
      delete(key) {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        queueMicrotask(() => {
          recordsMap.delete(key);
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
      getAll() {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        queueMicrotask(() => {
          req.result = Array.from(recordsMap.values()).map((r) => ({
            ...r,
            data: r.data instanceof Uint8Array ? r.data.slice() : r.data,
          }));
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
      clear() {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        queueMicrotask(() => {
          recordsMap.clear();
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
      queueMicrotask(() => {
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
          },
          transaction(storeName) {
            if (!entry.stores.has(storeName)) {
              entry.stores.set(storeName, new Map());
            }
            const store = makeStore(entry.stores.get(storeName));
            return {
              objectStore: () => store,
            };
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

function createMockLocalStorage() {
  const map = new Map();
  return {
    _map: map,
    getItem(key) {
      return map.has(key) ? map.get(key) : null;
    },
    setItem(key, value) {
      map.set(key, String(value));
    },
    removeItem(key) {
      map.delete(key);
    },
  };
}

function createMockElement(initialValue = '') {
  const listeners = new Map();
  const attributes = new Map();
  return {
    value: initialValue,
    dataset: {},
    _listeners: listeners,
    _attributes: attributes,
    addEventListener(type, fn) {
      if (!listeners.has(type)) {
        listeners.set(type, new Set());
      }
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      const set = listeners.get(type);
      if (set) {
        set.delete(fn);
      }
    },
    setAttribute(name, val) {
      attributes.set(name, String(val));
    },
    getAttribute(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
    listenerCount(type) {
      const set = listeners.get(type);
      return set ? set.size : 0;
    },
    dispatch(type, extra = {}) {
      const set = listeners.get(type);
      if (!set) return;
      const event = { type, target: this, ...extra };
      for (const fn of [...set]) {
        fn(event);
      }
    },
    click() {
      this.dispatch('click');
    },
  };
}

function buildPacket(opcode, payloadBytes = new Uint8Array(0), flags = 0) {
  const out = new Uint8Array(8 + payloadBytes.byteLength);
  const view = new DataView(out.buffer);
  view.setUint16(0, opcode, true);
  view.setUint16(2, flags, true);
  view.setUint32(4, payloadBytes.byteLength, true);
  out.set(payloadBytes, 8);
  return out;
}

function buildSampleGpuFrame() {
  const begin = buildPacket(OPCODES.BEGIN_FRAME);
  const clearPayload = new Uint8Array(16);
  const clearView = new DataView(clearPayload.buffer);
  clearView.setFloat32(0, 0.1, true);
  clearView.setFloat32(4, 0.2, true);
  clearView.setFloat32(8, 0.3, true);
  clearView.setFloat32(12, 1.0, true);
  const clear = buildPacket(OPCODES.CLEAR_COLOR, clearPayload);
  const present = buildPacket(OPCODES.PRESENT);

  const frame = new Uint8Array(begin.byteLength + clear.byteLength + present.byteLength);
  frame.set(begin, 0);
  frame.set(clear, begin.byteLength);
  frame.set(present, begin.byteLength + clear.byteLength);
  return frame;
}

function createFullCapabilities() {
  return {
    crossOriginIsolated: true,
    sharedArrayBuffer: true,
    webgpu: true,
    wasm64: true,
    opfs: true,
    gamepad: true,
    audioWorklet: true,
  };
}

function createMockFetch(statusPayload = {}) {
  return async () => ({
    ok: true,
    status: 200,
    async json() {
      return {
        ok: true,
        coopCoep: true,
        dataDirConfigured: true,
        hasManifest: true,
        hasBootset: true,
        hasWasmBinary: false,
        shaderPackCount: 2,
        ...statusPayload,
      };
    },
  });
}

async function flushAsync(ticks = 6) {
  for (let i = 0; i < ticks; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

// --- Tests ---

test('clearOpfsCache clears OPFS entries via real BrowserStorageManager and logs freed bytes', async () => {
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, {
    'chunk_000.bin': new Uint8Array(2048),
    'chunk_001.bin': new Uint8Array(1024),
  });
  const opfsRoot = createMockOpfsDirHandle('', {
    [DEFAULT_OPFS_CACHE_DIR]: cacheDir,
  });

  const storageManager = new BrowserStorageManager({
    storageObj: {
      async estimate() {
        return { usage: 3072, quota: 1024 * 1024 };
      },
      async persisted() {
        return true;
      },
      async getDirectory() {
        return opfsRoot;
      },
    },
    opfsRoot,
    indexedDBFactory: createMockIndexedDB(),
  });

  const hud = createMockHud();
  const controller = new HudController({ hud, storageManager });

  const before = await storageManager.inspectOpfsCache();
  assert.equal(before.entryCount, 2);
  assert.equal(before.totalBytes, 3072);

  const res = await controller.clearOpfsCache();
  assert.equal(res.ok, true);
  assert.equal(res.action, HUD_ACTIONS.CLEAR_OPFS_CACHE);
  assert.equal(res.removedCount, 2);
  assert.equal(res.freedBytes, 3072);
  assert.match(res.summary, /3 KiB/);
  assert.equal(hud.logs.length, 1);
  assert.equal(hud.logs[0].level, 'ok');
  assert.match(hud.logs[0].message, /3072 B/);

  const after = await storageManager.inspectOpfsCache();
  assert.equal(after.entryCount, 0);
  assert.equal(after.totalBytes, 0);

  const state = controller.getState();
  assert.equal(state.lastAction.action, HUD_ACTIONS.CLEAR_OPFS_CACHE);
  assert.equal(state.lastAction.ok, true);
});

test('exportSaves and importSaves round-trip userdata backup through real BrowserStorageManager', async () => {
  const mockIdb = createMockIndexedDB();
  const storageManager = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: null,
    indexedDBFactory: mockIdb,
    now: () => 1700000000000,
  });

  // Seed two /userdata/ save files using importUserdataSnapshot.
  const payloadA = new Uint8Array([1, 2, 3, 4, 5, 6]);
  const payloadB = new TextEncoder().encode('{"slot":1,"progress":42}');
  await storageManager.importUserdataSnapshot({
    format: SNAPSHOT_FORMAT,
    version: SNAPSHOT_VERSION,
    files: [
      {
        path: '/userdata/saves/slot0.sav',
        encoding: 'base64',
        dataBase64: Buffer.from(payloadA).toString('base64'),
        byteLength: payloadA.byteLength,
        updatedAt: 1700000000000,
      },
      {
        path: '/userdata/profile.json',
        encoding: 'base64',
        dataBase64: Buffer.from(payloadB).toString('base64'),
        byteLength: payloadB.byteLength,
        updatedAt: 1700000000000,
      },
    ],
  });

  const downloads = [];
  const hud = createMockHud();
  const controller = new HudController({
    hud,
    storageManager,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
  });

  // 1. Export saves and confirm JSON backup download.
  const exp = await controller.exportSaves();
  assert.equal(exp.ok, true);
  assert.equal(exp.action, HUD_ACTIONS.EXPORT_SAVES);
  assert.equal(exp.downloaded, true);
  assert.equal(exp.fileCount, 2);
  assert.equal(exp.totalBytes, payloadA.byteLength + payloadB.byteLength);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, DEFAULT_FILENAMES.USERDATA_BACKUP);
  assert.equal(downloads[0].mimeType, 'application/json');
  assert.equal(downloads[0].data, exp.json);

  // 2. Wipe IndexedDB and re-import from the exported JSON string.
  const cleared = await storageManager.clearUserdata();
  assert.equal(cleared.clearedCount, 2);
  assert.equal((await storageManager.listUserdataFiles()).fileCount, 0);

  const imp = await controller.importSaves(exp.json, { overwrite: true });
  assert.equal(imp.ok, true);
  assert.equal(imp.action, HUD_ACTIONS.IMPORT_SAVES);
  assert.equal(imp.importedCount, 2);
  assert.equal(imp.skippedCount, 0);
  assert.equal(imp.importedBytes, payloadA.byteLength + payloadB.byteLength);

  // 3. Re-importing with overwrite: false skips existing files.
  const impSkip = await controller.importSaves(exp.json, { overwrite: false });
  assert.equal(impSkip.ok, true);
  assert.equal(impSkip.importedCount, 0);
  assert.equal(impSkip.skippedCount, 2);

  // 4. Verify restored snapshot content matches original export.
  const reExport = await storageManager.exportUserdataSnapshot();
  assert.deepEqual(reExport.files, exp.snapshot.files);

  // 5. Malformed JSON & path traversal fail gracefully without throwing.
  const badJson = await controller.importSaves('{not valid json');
  assert.equal(badJson.ok, false);
  assert.equal(badJson.action, HUD_ACTIONS.IMPORT_SAVES);
  assert.ok(badJson.error);

  const traversalJson = JSON.stringify({
    format: SNAPSHOT_FORMAT,
    version: SNAPSHOT_VERSION,
    files: [
      {
        path: '/userdata/../etc/passwd',
        encoding: 'base64',
        dataBase64: 'AAAA',
      },
    ],
  });
  const badPath = await controller.importSaves(traversalJson);
  assert.equal(badPath.ok, false);
  assert.match(badPath.error, /Invalid userdata path segment/i);
});

test('toggleGpuRecording and exportGpuTrace record and download .pg6trace via real GpuFrameRecorder', async () => {
  const gpuRecorder = new GpuFrameRecorder({ maxFrames: 8, autoStart: false });
  const downloads = [];
  const hud = createMockHud();

  const controller = new HudController({
    hud,
    gpuRecorder,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
  });

  assert.equal(controller.getState().isRecording, false);

  // Exporting before any frames are recorded returns ok: false without downloading empty bytes.
  const emptyExp = await controller.exportGpuTrace();
  assert.equal(emptyExp.ok, false);
  assert.equal(emptyExp.byteLength, 0);
  assert.equal(downloads.length, 0);

  // Toggle ON
  const startRes = await controller.toggleGpuRecording();
  assert.equal(startRes.ok, true);
  assert.equal(startRes.action, HUD_ACTIONS.TOGGLE_GPU_RECORDING);
  assert.equal(startRes.isRecording, true);
  assert.equal(gpuRecorder.isRecording(), true);
  assert.equal(controller.getState().isRecording, true);

  // Record 2 complete frames
  const frameBytes = buildSampleGpuFrame();
  gpuRecorder.recordChunk(frameBytes);
  gpuRecorder.recordChunk(frameBytes);

  // Toggle OFF
  const stopRes = await controller.toggleGpuRecording();
  assert.equal(stopRes.ok, true);
  assert.equal(stopRes.isRecording, false);
  assert.equal(gpuRecorder.isRecording(), false);
  assert.match(stopRes.summary, /2 frames/);

  // Export .pg6trace binary stream
  const traceRes = await controller.exportGpuTrace();
  assert.equal(traceRes.ok, true);
  assert.equal(traceRes.action, HUD_ACTIONS.EXPORT_GPU_TRACE);
  assert.equal(traceRes.filename, DEFAULT_FILENAMES.GPU_TRACE);
  assert.equal(traceRes.downloaded, true);
  assert.equal(traceRes.frameCount, 2);
  assert.equal(traceRes.byteLength, frameBytes.byteLength * 2);
  assert.ok(traceRes.bytes instanceof Uint8Array);

  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, 'playgta6-capture.pg6trace');
  assert.equal(downloads[0].mimeType, 'application/octet-stream');
  assert.deepEqual(downloads[0].data.subarray(0, frameBytes.byteLength), frameBytes);
});

test('toggleGpuRecording and exportGpuTrace also work via smokeDemoRunner when gpuRecorder is omitted', async () => {
  const innerRecorder = new GpuFrameRecorder({ maxFrames: 4, autoStart: false });
  const frameBytes = buildSampleGpuFrame();
  const smokeDemoRunner = {
    running: false,
    recorder: innerRecorder,
    start() {
      this.running = true;
      return true;
    },
    stop() {
      this.running = false;
      return true;
    },
    startRecording() {
      innerRecorder.start();
      innerRecorder.recordChunk(frameBytes);
    },
    stopRecording() {
      innerRecorder.stop();
    },
    exportTraceBytes(opts) {
      return innerRecorder.exportTraceBytes(opts);
    },
    exportTraceMetadata() {
      return innerRecorder.exportMetadata();
    },
  };

  const downloads = [];
  const controller = new HudController({
    hud: createMockHud(),
    smokeDemoRunner,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
  });

  const onRes = await controller.toggleGpuRecording();
  assert.equal(onRes.ok, true);
  assert.equal(onRes.isRecording, true);

  const offRes = await controller.toggleGpuRecording();
  assert.equal(offRes.ok, true);
  assert.equal(offRes.isRecording, false);

  const expRes = await controller.exportGpuTrace();
  assert.equal(expRes.ok, true);
  assert.equal(expRes.frameCount, 1);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, 'playgta6-capture.pg6trace');
});

test('exportCrashReport produces sanitized JSON bundle via real CrashReporter, PerfMonitor, StorageManager, and SessionController', async () => {
  const crashReporter = new CrashReporter({
    appVersion: '0.2.0-test',
    now: () => 1700000123000,
  });
  crashReporter.recordError(
    'io_error',
    new Error('Failed reading C:\\Users\\secretuser\\GTA6\\data.rpf with Bearer super-secret-token-999'),
    { path: '/home/secretuser/private/key.bin' }
  );
  crashReporter.recordLog('warn', 'hud', 'Loaded from file:///Users/secretuser/local.wasm?token=abc123');

  const perfMonitor = new PerfMonitor({ now: () => 1000 });
  perfMonitor.recordFrameTimeMs(16.6);
  perfMonitor.recordGpuBatch({ packets: 12, drawCalls: 2, bytes: 512, durationMs: 16.6 });

  const storageManager = new BrowserStorageManager({
    storageObj: {
      async estimate() {
        return { usage: 4096, quota: 1024 * 1024 };
      },
      async persisted() {
        return false;
      },
    },
    opfsRoot: null,
    indexedDBFactory: createMockIndexedDB(),
  });

  const sessionController = new RuntimeSessionController({
    capabilityProbe: async () => createFullCapabilities(),
    fetchImpl: createMockFetch({ hasWasmBinary: false, shaderPackCount: 5 }),
    startSmokeDemo: async () => ({ stop() {} }),
  });
  await sessionController.start({ preferSmoke: true });

  const downloads = [];
  const clipboardWrites = [];
  const hud = createMockHud();

  const controller = new HudController({
    hud,
    crashReporter,
    perfMonitor,
    storageManager,
    sessionController,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
    copyToClipboard(text) {
      clipboardWrites.push(text);
    },
  });

  const res = await controller.exportCrashReport({ copyToClipboard: true });
  assert.equal(res.ok, true);
  assert.equal(res.action, HUD_ACTIONS.EXPORT_CRASH_REPORT);
  assert.equal(res.filename, DEFAULT_FILENAMES.CRASH_REPORT);
  assert.equal(res.downloaded, true);
  assert.equal(res.copied, true);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, 'playgta6-diagnostic-report.json');
  assert.equal(downloads[0].mimeType, 'application/json');
  assert.equal(clipboardWrites.length, 1);

  // Verify sanitization: zero host usernames, paths, or secrets leaked.
  assert.doesNotMatch(res.json, /secretuser/);
  assert.doesNotMatch(res.json, /super-secret-token-999/);
  assert.doesNotMatch(res.json, /token=abc123/);
  assert.match(res.json, /\[REDACTED_PATH\]/);
  assert.match(res.json, /\[REDACTED\]/);

  const parsed = JSON.parse(res.json);
  assert.equal(parsed.schemaVersion, 1);
  assert.equal(parsed.appVersion, '0.2.0-test');
  assert.equal(parsed.errors.length, 1);
  assert.equal(parsed.perf.gpu.totalGpuPackets, 12);
  assert.equal(parsed.storage.quota.usageBytes, 4096);
  assert.equal(parsed.serverStatus.shaderPackCount, 5);

  await sessionController.stop();
});

test('switchInputProfile switches active profile and persists via real InputProfileManager', async () => {
  const mockStorage = createMockLocalStorage();
  const inputProfiles = new InputProfileManager({ storage: mockStorage });
  const hud = createMockHud();
  const controller = new HudController({ hud, inputProfiles });

  assert.equal(controller.getState().activeProfileId, 'default');

  const okRes = await controller.switchInputProfile('fps');
  assert.equal(okRes.ok, true);
  assert.equal(okRes.action, HUD_ACTIONS.SWITCH_INPUT_PROFILE);
  assert.equal(okRes.profileId, 'fps');
  assert.equal(okRes.persisted, true);
  assert.equal(inputProfiles.getActiveProfile().id, 'fps');
  assert.equal(controller.getState().activeProfileId, 'fps');

  // Verify persistence by loading into a fresh InputProfileManager instance.
  const reloaded = new InputProfileManager({ storage: mockStorage, autoLoad: true });
  assert.equal(reloaded.getActiveProfile().id, 'fps');

  // Unknown profile ID fails gracefully without throwing.
  const badRes = await controller.switchInputProfile('nonexistent-profile');
  assert.equal(badRes.ok, false);
  assert.equal(badRes.action, HUD_ACTIONS.SWITCH_INPUT_PROFILE);
  assert.match(badRes.error, /Unknown input profile/i);
  assert.equal(inputProfiles.getActiveProfile().id, 'fps');
});

test('toggleSmokeDemo and restartSession drive real RuntimeSessionController lifecycle', async () => {
  let wasmStarts = 0;
  let wasmStops = 0;
  let smokeStarts = 0;
  let smokeStops = 0;

  const sessionController = new RuntimeSessionController({
    capabilityProbe: async () => createFullCapabilities(),
    fetchImpl: createMockFetch({ hasWasmBinary: true }),
    startWasmRuntime: async () => {
      wasmStarts += 1;
      return {
        stop() {
          wasmStops += 1;
        },
      };
    },
    startSmokeDemo: async () => {
      smokeStarts += 1;
      return {
        stop() {
          smokeStops += 1;
        },
      };
    },
  });

  const hud = createMockHud();
  const controller = new HudController({ hud, sessionController });

  assert.equal(controller.getState().sessionState, SESSION_STATE.IDLE);

  // 1. Toggle smoke demo ON from idle -> running_smoke.
  const startSmoke = await controller.toggleSmokeDemo();
  assert.equal(startSmoke.ok, true);
  assert.equal(startSmoke.action, HUD_ACTIONS.TOGGLE_SMOKE_DEMO);
  assert.equal(startSmoke.running, true);
  assert.equal(sessionController.getState(), SESSION_STATE.RUNNING_SMOKE);
  assert.equal(controller.getState().sessionState, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(smokeStarts, 1);

  // 2. Toggle smoke demo OFF -> stopped.
  const stopSmoke = await controller.toggleSmokeDemo();
  assert.equal(stopSmoke.ok, true);
  assert.equal(stopSmoke.running, false);
  assert.equal(sessionController.getState(), SESSION_STATE.STOPPED);
  assert.equal(smokeStops, 1);

  // 3. Restart session preferring Wasm64 (preferSmoke: false) -> running_wasm.
  const restartWasm = await controller.restartSession({ preferSmoke: false });
  assert.equal(restartWasm.ok, true);
  assert.equal(restartWasm.action, HUD_ACTIONS.RESTART_SESSION);
  assert.equal(restartWasm.state, SESSION_STATE.RUNNING_WASM);
  assert.equal(restartWasm.mode, SESSION_MODE.WASM);
  assert.equal(wasmStarts, 1);

  // 4. Restart session with preferSmoke: true -> stops Wasm64 and boots smoke demo.
  const restartSmoke = await controller.restartSession({ preferSmoke: true });
  assert.equal(restartSmoke.ok, true);
  assert.equal(restartSmoke.state, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(restartSmoke.mode, SESSION_MODE.SMOKE);
  assert.equal(wasmStops, 1);
  assert.equal(smokeStarts, 2);

  await sessionController.stop();
  assert.equal(smokeStops, 2);
});

test('bindButtons wires DOM controls cleanly, avoids duplicate listeners, and cleans up on destroy', async () => {
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, {
    'block.bin': new Uint8Array(512),
  });
  const opfsRoot = createMockOpfsDirHandle('', {
    [DEFAULT_OPFS_CACHE_DIR]: cacheDir,
  });
  const storageManager = new BrowserStorageManager({
    storageObj: {
      async estimate() {
        return { usage: 512, quota: 1024 * 1024 };
      },
      async persisted() {
        return false;
      },
      async getDirectory() {
        return opfsRoot;
      },
    },
    opfsRoot,
    indexedDBFactory: createMockIndexedDB(),
  });
  const gpuRecorder = new GpuFrameRecorder({ autoStart: false });
  const crashReporter = new CrashReporter({ now: () => 1700000000000 });
  const inputProfiles = new InputProfileManager({ storage: createMockLocalStorage() });
  const sessionController = new RuntimeSessionController({
    capabilityProbe: async () => createFullCapabilities(),
    fetchImpl: createMockFetch({ hasWasmBinary: false }),
    startSmokeDemo: async () => ({ stop() {} }),
  });

  const downloads = [];
  const hud = createMockHud();
  const controller = new HudController({
    hud,
    storageManager,
    gpuRecorder,
    crashReporter,
    inputProfiles,
    sessionController,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
  });

  const buttons = {
    clearCacheBtn: createMockElement(),
    exportSavesBtn: createMockElement(),
    recordTraceBtn: createMockElement(),
    exportTraceBtn: createMockElement(),
    exportCrashBtn: createMockElement(),
    restartBtn: createMockElement(),
    profileSelect: createMockElement('default'),
  };

  controller.bindButtons(buttons);
  // Re-binding replaces previous listeners rather than duplicating them.
  controller.bindButtons(buttons);

  assert.equal(controller.getState().boundControlCount, 7);
  assert.equal(buttons.clearCacheBtn.listenerCount('click'), 1);
  assert.equal(buttons.profileSelect.listenerCount('change'), 1);
  assert.equal(buttons.recordTraceBtn.dataset.recording, 'false');
  assert.equal(buttons.recordTraceBtn.getAttribute('aria-pressed'), 'false');

  // Fire each bound control
  buttons.clearCacheBtn.click();
  await flushAsync();
  assert.equal((await storageManager.inspectOpfsCache()).entryCount, 0);

  buttons.exportSavesBtn.click();
  await flushAsync();
  assert.ok(downloads.some((d) => d.filename === DEFAULT_FILENAMES.USERDATA_BACKUP));

  buttons.recordTraceBtn.click();
  await flushAsync();
  assert.equal(gpuRecorder.isRecording(), true);
  assert.equal(buttons.recordTraceBtn.dataset.recording, 'true');
  assert.equal(buttons.recordTraceBtn.getAttribute('aria-pressed'), 'true');

  gpuRecorder.recordChunk(buildSampleGpuFrame());
  buttons.exportTraceBtn.click();
  await flushAsync();
  assert.ok(downloads.some((d) => d.filename === DEFAULT_FILENAMES.GPU_TRACE));

  buttons.exportCrashBtn.click();
  await flushAsync();
  assert.ok(downloads.some((d) => d.filename === DEFAULT_FILENAMES.CRASH_REPORT));

  buttons.restartBtn.click();
  await flushAsync();
  assert.equal(sessionController.getState(), SESSION_STATE.RUNNING_SMOKE);

  buttons.profileSelect.value = 'southpaw';
  buttons.profileSelect.dispatch('change');
  await flushAsync();
  assert.equal(inputProfiles.getActiveProfile().id, 'southpaw');

  const logCountBeforeDestroy = hud.logs.length;
  const downloadCountBeforeDestroy = downloads.length;

  // Destroy cleans up every listener and is idempotent.
  controller.destroy();
  controller.destroy();

  assert.equal(controller.getState().destroyed, true);
  assert.equal(controller.getState().boundControlCount, 0);
  for (const [name, el] of Object.entries(buttons)) {
    const eventType = name === 'profileSelect' ? 'change' : 'click';
    assert.equal(el.listenerCount(eventType), 0, `${name} should have 0 listeners after destroy()`);
  }

  // Firing elements after destroy must be complete no-ops.
  buttons.clearCacheBtn.click();
  buttons.exportSavesBtn.click();
  buttons.recordTraceBtn.click();
  buttons.exportTraceBtn.click();
  buttons.exportCrashBtn.click();
  buttons.restartBtn.click();
  buttons.profileSelect.value = 'default';
  buttons.profileSelect.dispatch('change');
  await flushAsync();

  assert.equal(hud.logs.length, logCountBeforeDestroy);
  assert.equal(downloads.length, downloadCountBeforeDestroy);
  assert.equal(inputProfiles.getActiveProfile().id, 'southpaw');

  await sessionController.stop();
});

test('gracefully handles absent subsystems and failing download/HUD callbacks without unhandled rejections', async () => {
  const controller = new HudController();

  assert.deepEqual(controller.getState(), {
    isRecording: false,
    sessionState: SESSION_STATE.IDLE,
    sessionMode: SESSION_MODE.NONE,
    activeProfileId: null,
    lastAction: null,
    boundControlCount: 0,
    destroyed: false,
  });

  const results = [
    await controller.clearOpfsCache(),
    await controller.exportSaves(),
    await controller.importSaves('{"files":[]}'),
    await controller.toggleGpuRecording(),
    await controller.exportGpuTrace(),
    await controller.exportCrashReport(),
    await controller.switchInputProfile('fps'),
    await controller.toggleSmokeDemo(),
    await controller.restartSession({ preferSmoke: true }),
  ];

  for (const res of results) {
    assert.equal(res.ok, false);
    assert.equal(typeof res.action, 'string');
    assert.equal(typeof res.summary, 'string');
    assert.equal(typeof res.error, 'string');
  }

  // Failing downloadBlob and throwing hud.appendLog never reject.
  const crashReporter = new CrashReporter({ now: () => 1700000000000 });
  const failingController = new HudController({
    hud: {
      appendLog() {
        throw new Error('HUD DOM detached');
      },
    },
    crashReporter,
    downloadBlob() {
      throw new Error('Download blocked by browser');
    },
  });

  const failRes = await failingController.exportCrashReport();
  assert.equal(failRes.ok, false);
  assert.equal(failRes.action, HUD_ACTIONS.EXPORT_CRASH_REPORT);
  assert.match(failRes.error, /Download blocked by browser/);
});

function createPanelMockDocument() {
  function makeNode(tagName) {
    const listeners = new Map();
    const attributes = new Map();
    const node = {
      tagName: tagName.toUpperCase(),
      className: '',
      textContent: '',
      hidden: false,
      style: {},
      children: [],
      parentNode: null,
      setAttribute(name, val) {
        attributes.set(name, String(val));
      },
      getAttribute(name) {
        return attributes.has(name) ? attributes.get(name) : null;
      },
      appendChild(child) {
        if (child && typeof child === 'object') {
          child.parentNode = node;
        }
        node.children.push(child);
        return child;
      },
      removeChild(child) {
        const idx = node.children.indexOf(child);
        if (idx !== -1) {
          node.children.splice(idx, 1);
        }
        if (child && typeof child === 'object') {
          child.parentNode = null;
        }
        return child;
      },
      addEventListener(type, fn) {
        if (!listeners.has(type)) listeners.set(type, new Set());
        listeners.get(type).add(fn);
      },
      removeEventListener(type, fn) {
        listeners.get(type)?.delete(fn);
      },
    };
    return node;
  }

  return {
    head: makeNode('head'),
    body: makeNode('body'),
    createElement(tag) {
      return makeNode(tag);
    },
    querySelector() {
      return null;
    },
  };
}

test('updateGpuBootTelemetry and exportMissingShaders wire cleanly to real GpuBootPanelController and DOM button', async () => {
  const doc = createPanelMockDocument();
  const container = doc.createElement('div');
  const gpuBootPanel = new GpuBootPanelController({
    container,
    documentObj: doc,
    now: () => 1700000555000,
  });

  const downloads = [];
  const clipboardWrites = [];
  const hud = createMockHud();
  const controller = new HudController({
    hud,
    gpuBootPanel,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
    copyToClipboard(text) {
      clipboardWrites.push(text);
    },
  });

  // Update telemetry via HudController helper.
  const panelState = controller.updateGpuBootTelemetry({
    bootSummary: { stage: 'FIRST_PRESENT', milestonesReached: 9, totalMilestones: 9 },
    frameStats: { framesPresented: 12, droppedFrames: 1 },
    fallbackStats: { totalLookups: 10, packHits: 8, fallbackHits: 2, builtinHits: 0 },
    missingShaders: [
      { hash: '0x0000000000001234', stage: 'vertex', hits: 3, rawPath: 'C:\\Users\\alice\\shader.dxil' },
      { hash: '0x0000000000005678', stage: 'fragment', hits: 7 },
    ],
  });

  assert.ok(panelState);
  assert.equal(panelState.boot.stage, 'FIRST_PRESENT');
  assert.equal(panelState.boot.readyForSplash, true);
  assert.equal(panelState.frameStats.framesPresented, 12);
  assert.equal(panelState.totalMissingShaders, 2);

  // Export missing shaders programmatically with clipboard copy.
  const res = await controller.exportMissingShaders({ copyToClipboard: true });
  assert.equal(res.ok, true);
  assert.equal(res.action, HUD_ACTIONS.EXPORT_MISSING_SHADERS);
  assert.equal(res.filename, DEFAULT_FILENAMES.MISSING_SHADERS);
  assert.equal(res.missingShaderCount, 2);
  assert.equal(res.downloaded, true);
  assert.equal(res.copied, true);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, 'playgta6-missing-shaders.json');
  assert.equal(downloads[0].mimeType, 'application/json');
  assert.equal(downloads[0].data, res.json);
  assert.equal(clipboardWrites.length, 1);
  assert.doesNotMatch(res.json, /alice/i);

  const parsed = JSON.parse(res.json);
  assert.equal(parsed.format, EXPORT_FORMAT);
  assert.equal(parsed.missingShaderCount, 2);
  assert.equal(parsed.missingShaders[0].hash, '0x0000000000005678');
  assert.equal(parsed.missingShaders[0].hits, 7);

  // Bind #btn-export-missing-shaders DOM button and verify click + destroy cleanup.
  const exportMissingBtn = createMockElement();
  controller.bindButtons({ exportMissingShadersBtn: exportMissingBtn });
  assert.equal(controller.getState().boundControlCount, 1);
  assert.equal(exportMissingBtn.listenerCount('click'), 1);

  exportMissingBtn.click();
  await flushAsync();
  assert.equal(downloads.length, 2);

  controller.destroy();
  assert.equal(exportMissingBtn.listenerCount('click'), 0);
  assert.equal(controller.updateGpuBootTelemetry({ frameStats: { framesPresented: 99 } }), null);
  assert.equal(gpuBootPanel.getState().frameStats.framesPresented, 12);

  gpuBootPanel.destroy();
});

test('HudController resolves app.gpuBootPanel fallback and preserves backwards compatibility when absent or failing', async () => {
  // 1. Absent gpuBootPanel: safe no-op for telemetry and graceful failure for export.
  const emptyController = new HudController({ hud: createMockHud() });
  assert.equal(
    emptyController.updateGpuBootTelemetry({
      bootSummary: { stage: 'PGIM_LOAD' },
      missingShaders: [{ hash: '0x1', stage: 'vertex', hits: 1 }],
    }),
    null
  );
  const absentRes = await emptyController.exportMissingShaders();
  assert.equal(absentRes.ok, false);
  assert.equal(absentRes.action, HUD_ACTIONS.EXPORT_MISSING_SHADERS);
  assert.match(absentRes.error, /not configured/i);

  // 2. Fallback via options.app.gpuBootPanel (including late attachment).
  const doc = createPanelMockDocument();
  const appStub = { gpuBootPanel: null };
  const downloads = [];
  const appController = new HudController({
    app: appStub,
    hud: createMockHud(),
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
  });
  assert.equal(appController.updateGpuBootTelemetry({ bootSummary: { stage: 'PGIM_LOAD' } }), null);

  appStub.gpuBootPanel = new GpuBootPanelController({
    container: doc.createElement('div'),
    documentObj: doc,
    now: () => 1700000999000,
  });

  const updated = appController.updateGpuBootTelemetry({
    bootSummary: { stage: 'BINK_OPEN' },
    missingShaders: [{ hash: '0xabcd', stage: 'compute', hits: 2 }],
  });
  assert.equal(updated.boot.stage, 'BINK_OPEN');
  assert.equal(updated.totalMissingShaders, 1);

  const exp = await appController.exportMissingShaders();
  assert.equal(exp.ok, true);
  assert.equal(exp.missingShaderCount, 1);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, DEFAULT_FILENAMES.MISSING_SHADERS);

  // 3. Throwing gpuBootPanel methods never crash caller.
  const throwingController = new HudController({
    hud: createMockHud(),
    gpuBootPanel: {
      update() {
        throw new Error('panel update boom');
      },
      exportMissingShadersJson() {
        throw new Error('panel export boom');
      },
    },
  });
  assert.equal(throwingController.updateGpuBootTelemetry({ bootSummary: { stage: 'PGIM_LOAD' } }), null);
  const errRes = await throwingController.exportMissingShaders();
  assert.equal(errRes.ok, false);
  assert.equal(errRes.action, HUD_ACTIONS.EXPORT_MISSING_SHADERS);
  assert.match(errRes.error, /panel export boom/);

  appStub.gpuBootPanel.destroy();
});
