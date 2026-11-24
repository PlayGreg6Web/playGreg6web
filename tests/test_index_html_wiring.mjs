import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import { bootstrapPlayGta6App } from '../web/app_bootstrap.js';
import { HudController, HUD_ACTIONS, DEFAULT_FILENAMES } from '../web/ui/hud_controller.js';
import { SESSION_STATE, SESSION_MODE } from '../web/runtime/session_controller.js';

const require = createRequire(import.meta.url);
const { GpuBootPanelController, EXPORT_FORMAT } = require('../web/ui/gpu_boot_panel.js');

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const INDEX_HTML_PATH = path.join(REPO_ROOT, 'web', 'index.html');

const indexHtml = fs.readFileSync(INDEX_HTML_PATH, 'utf8');

function extractModuleScript(html) {
  const match = /<script\s+type="module">([\s\S]*?)<\/script>/i.exec(html);
  assert.ok(match, 'web/index.html must contain a <script type="module"> block');
  return match[1];
}

function createClock(startMs = 1000) {
  let current = startMs;
  return {
    now: () => current,
    advance(deltaMs) {
      current += deltaMs;
      return current;
    },
  };
}

function createMockEventTarget(extra = {}) {
  const listeners = new Map();
  const attributes = new Map();
  return {
    dataset: {},
    value: '',
    textContent: '',
    children: [],
    ...extra,
    setAttribute(name, value) {
      attributes.set(name, String(value));
    },
    getAttribute(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    addEventListener(type, fn, opts) {
      if (!listeners.has(type)) {
        listeners.set(type, new Set());
      }
      listeners.get(type).add({ fn, once: Boolean(opts && opts.once) });
    },
    removeEventListener(type, fn) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of set) {
        if (entry.fn === fn) {
          set.delete(entry);
        }
      }
    },
    dispatch(type, event = {}) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of [...set]) {
        if (entry.once) {
          set.delete(entry);
        }
        entry.fn({ type, target: this, ...event });
      }
    },
    listenerCount(type) {
      if (type) {
        return listeners.get(type)?.size ?? 0;
      }
      let total = 0;
      for (const set of listeners.values()) {
        total += set.size;
      }
      return total;
    },
  };
}

function createMemoryStorage() {
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
    async removeEntry(entryName) {
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
          objectStoreNames: { contains: (name) => entry.stores.has(name) },
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
            return { objectStore: () => store };
          },
          close() {},
        };
        req.result = db;
        if (isNew && req.onupgradeneeded) req.onupgradeneeded({ target: req });
        if (req.onsuccess) req.onsuccess({ target: req });
      });
      return req;
    },
  };
}

function createFakeRaf(clock) {
  let nextId = 1;
  const scheduled = new Map();
  return {
    raf(cb) {
      const id = nextId++;
      scheduled.set(id, cb);
      return id;
    },
    cancelRaf(id) {
      scheduled.delete(id);
    },
    flushFrame(deltaMs = 16) {
      if (scheduled.size === 0) return 0;
      const batch = [...scheduled.entries()];
      scheduled.clear();
      const ts = clock.advance(deltaMs);
      for (const [, cb] of batch) {
        cb(ts);
      }
      return batch.length;
    },
  };
}

function createFakeTimers() {
  let nextId = 1;
  const active = new Map();
  return {
    setInterval(fn, ms) {
      const id = nextId++;
      active.set(id, { fn, ms });
      return id;
    },
    clearInterval(id) {
      active.delete(id);
    },
  };
}

function createMockHud() {
  const logs = [];
  const statuses = [];
  const metrics = [];
  return {
    logs,
    statuses,
    metrics,
    setStatus(text, level = 'info') {
      statuses.push({ text, level });
    },
    appendLog(message, level = 'info') {
      logs.push({ message, level });
    },
    updateMetrics(payload) {
      metrics.push({ ...payload });
    },
    refreshServerStatus() {},
    setCollapsed() {},
  };
}

function createMockWebGpuEnv() {
  let nextHandle = 1;
  let swapchainCounter = 0;

  const context = {
    configure() {},
    unconfigure() {},
    getCurrentTexture() {
      const id = `swapchain-view-${swapchainCounter++}`;
      return {
        createView() {
          return { kind: 'GPUTextureView', id };
        },
      };
    },
  };

  const canvas = createMockEventTarget({
    width: 640,
    height: 360,
    clientWidth: 640,
    clientHeight: 360,
    getContext(kind) {
      return kind === 'webgpu' ? context : null;
    },
  });

  const device = createMockEventTarget({
    limits: { maxTextureDimension2D: 8192 },
    lost: new Promise(() => {}),
    destroy() {},
    createBuffer(desc) {
      return {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroy() {},
      };
    },
    createTexture(desc) {
      const handle = nextHandle++;
      return {
        kind: 'GPUTexture',
        handle,
        desc: { ...desc, size: { ...desc.size } },
        createView() {
          return { kind: 'GPUTextureView', textureHandle: handle, handle: nextHandle++ };
        },
        destroy() {},
      };
    },
    createSampler(desc) {
      return { kind: 'GPUSampler', handle: nextHandle++, desc: { ...desc } };
    },
    createShaderModule(desc) {
      return { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
    },
    createRenderPipeline(desc) {
      return {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index = 0) {
          return { kind: 'GPUBindGroupLayout', index, handle: nextHandle++ };
        },
      };
    },
    createComputePipeline(desc) {
      return {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index = 0) {
          return { kind: 'GPUBindGroupLayout', index, handle: nextHandle++ };
        },
      };
    },
    createBindGroup(desc) {
      return { kind: 'GPUBindGroup', handle: nextHandle++, desc };
    },
    createCommandEncoder() {
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            setViewport() {},
            setScissorRect() {},
            draw() {},
            drawIndexed() {},
            end() {},
          };
        },
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        copyBufferToBuffer() {},
        finish() {
          return { kind: 'GPUCommandBuffer', handle: nextHandle++ };
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {},
    },
  });

  const adapter = {
    async requestDevice() {
      return device;
    },
  };

  return { canvas, device, adapter };
}

async function flushMicrotasks(rounds = 8) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

test('web/index.html preserves all HUD IDs and adds Controls & Diagnostics elements without duplicates', () => {
  const requiredIds = [
    // Viewport & HUD frame
    'viewport',
    'hud',
    'hud-status-dot',
    'hud-status',
    'hud-toggle',
    'hud-body',
    'hud-refresh-status',
    'hud-clear-log',
    'hud-log',
    // Capability badges
    'cap-coi',
    'cap-webgpu',
    'cap-wasm64',
    'cap-opfs',
    'cap-gamepad',
    // Server readiness cards
    'srv-manifest',
    'srv-bootset',
    'srv-wasm',
    'srv-shaders',
    // Telemetry cards
    'metric-fps',
    'metric-gpu-packets',
    'metric-opfs-hits',
    'metric-pointer-lock',
    'metric-gamepad',
    // Controls & Diagnostics (W30)
    'btn-smoke-demo',
    'btn-restart-session',
    'btn-record-trace',
    'btn-export-trace',
    'btn-clear-cache',
    'btn-export-saves',
    'btn-export-crash',
    'btn-export-missing-shaders',
    'select-input-profile',
    // Frame-1 GPU & missing-shader diagnostics panel (W55)
    'gpu-boot-panel-root',
  ];

  for (const id of requiredIds) {
    assert.match(
      indexHtml,
      new RegExp(`id="${id}"`),
      `web/index.html must define element with id="${id}"`
    );
  }

  // Verify no duplicate HTML id attributes exist.
  const idMatches = [...indexHtml.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
  const seen = new Set();
  for (const id of idMatches) {
    assert.equal(seen.has(id), false, `Duplicate element id="${id}" found in web/index.html`);
    seen.add(id);
  }

  // Verify the profile selector is a <select> element.
  assert.match(indexHtml, /<select\b[^>]*\bid="select-input-profile"/);
});

test('web/index.html inline ES module has valid syntax, imports app_bootstrap & hud_controller, and exposes global hooks', () => {
  const moduleScript = extractModuleScript(indexHtml);

  // Verify exact ES module imports and that the referenced files exist on disk.
  assert.match(
    moduleScript,
    /import\s*\{\s*bootstrapPlayGta6App\s*\}\s*from\s*['"]\.\/app_bootstrap\.js['"]/
  );
  assert.match(
    moduleScript,
    /import\s*\{\s*HudController\s*\}\s*from\s*['"]\.\/ui\/hud_controller\.js['"]/
  );
  assert.equal(fs.existsSync(path.join(REPO_ROOT, 'web', 'app_bootstrap.js')), true);
  assert.equal(fs.existsSync(path.join(REPO_ROOT, 'web', 'ui', 'hud_controller.js')), true);

  // Legacy hardcoded worker boot block must be gone.
  assert.doesNotMatch(moduleScript, /new\s+Worker\(\s*['"]\.\/loader\.js['"]\s*\)/);

  // Required global hooks and HUD API methods.
  assert.match(moduleScript, /window\.playgta6Hud\s*=\s*hudApi/);
  assert.match(moduleScript, /window\.playgta6App\s*=\s*app/);
  assert.match(moduleScript, /window\.playgta6HudController\s*=\s*hudController/);
  for (const fnName of [
    'setStatus',
    'appendLog',
    'updateMetrics',
    'refreshServerStatus',
    'setCollapsed',
  ]) {
    assert.match(moduleScript, new RegExp(`\\b${fnName}\\b`));
  }

  // F9 shortcut, capability guards, profile population, bindButtons, and pagehide cleanup.
  assert.match(moduleScript, /ev\.code\s*===\s*['"]F9['"]/);
  assert.match(moduleScript, /Missing COOP\/COEP/);
  assert.match(moduleScript, /WebGPU unavailable/);
  assert.match(moduleScript, /listProfiles\(\)/);
  assert.match(moduleScript, /hudController\.bindButtons\(/);
  assert.match(moduleScript, /addEventListener\(\s*['"]pagehide['"]/);
  assert.match(moduleScript, /hudController\.destroy\(\)/);
  assert.match(moduleScript, /app\.destroy\(\)/);

  // Validate syntax using Node's native ESM parser (`node --input-type=module --check`).
  const check = spawnSync(process.execPath, ['--input-type=module', '--check'], {
    input: moduleScript,
    encoding: 'utf8',
  });
  assert.equal(
    check.status,
    0,
    `Inline <script type="module"> failed syntax check:\n${check.stderr}`
  );
});

test('integration: bootstrapPlayGta6App + HudController wired to web/index.html control IDs handles smoke toggle, recording/export, profile switch, diagnostics, and destroy', async () => {
  const clock = createClock(5000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const hud = createMockHud();
  const localStorage = createMemoryStorage();
  const indexedDB = createMockIndexedDB();
  const opfsRoot = createMockOpfsDirHandle('', {
    'playgta6-cache': createMockOpfsDirHandle('playgta6-cache', {
      'chunk_0.bin': new Uint8Array([1, 2, 3, 4]),
    }),
  });
  const { canvas, adapter } = createMockWebGpuEnv();
  const downloads = [];

  const mockElements = {
    'btn-smoke-demo': createMockEventTarget({ textContent: 'Smoke Demo' }),
    'btn-restart-session': createMockEventTarget({ textContent: 'Restart' }),
    'btn-record-trace': createMockEventTarget({ textContent: 'Record Trace' }),
    'btn-export-trace': createMockEventTarget({ textContent: 'Export .pg6trace' }),
    'btn-clear-cache': createMockEventTarget({ textContent: 'Clear Cache' }),
    'btn-export-saves': createMockEventTarget({ textContent: 'Export Saves' }),
    'btn-export-crash': createMockEventTarget({ textContent: 'Export Report' }),
    'select-input-profile': createMockEventTarget({ value: 'default' }),
  };

  // Ensure every mock element ID actually exists in web/index.html.
  for (const id of Object.keys(mockElements)) {
    assert.match(indexHtml, new RegExp(`id="${id}"`), `Expected id="${id}" in web/index.html`);
  }

  const windowObj = createMockEventTarget({
    crossOriginIsolated: true,
    SharedArrayBuffer,
    devicePixelRatio: 1,
    innerWidth: 640,
    innerHeight: 360,
    location: { origin: 'http://127.0.0.1:8080' },
    localStorage,
    indexedDB,
    requestAnimationFrame: raf.raf,
    cancelAnimationFrame: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  const documentObj = createMockEventTarget({
    pointerLockElement: null,
    getElementById(id) {
      if (id === 'viewport') return canvas;
      return mockElements[id] || null;
    },
    createElement(tag) {
      return createMockEventTarget({ tagName: tag.toUpperCase() });
    },
  });

  const navigatorObj = {
    gpu: {
      requestAdapter: async () => adapter,
      getPreferredCanvasFormat: () => 'bgra8unorm',
    },
    storage: {
      getDirectory: async () => opfsRoot,
      estimate: async () => ({ usage: 4, quota: 1024 * 1024 }),
    },
    getGamepads: () => [],
  };

  const fetchImpl = async (url) => {
    assert.equal(url, '/api/status');
    return {
      ok: true,
      status: 200,
      async json() {
        return {
          ok: true,
          coopCoep: true,
          dataDirConfigured: true,
          hasManifest: false,
          hasBootset: false,
          hasWasmBinary: false,
          shaderPackCount: 0,
        };
      },
    };
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl,
    canvas,
    hud,
    dataBase: 'http://127.0.0.1:8080/data/',
    now: clock.now,
    epochNow: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  // Wire HudController with the same dynamic smoke runner adapter as web/index.html.
  const smokeRunnerProxy = {
    get recorder() {
      return app.getSmokeRunner()?.recorder || null;
    },
    get running() {
      return Boolean(app.getSmokeRunner()?.running);
    },
    start() {
      const runner = app.getSmokeRunner();
      return runner ? runner.start() : false;
    },
    stop() {
      const runner = app.getSmokeRunner();
      return runner ? runner.stop() : false;
    },
    startRecording() {
      const runner = app.getSmokeRunner();
      if (!runner) {
        throw new Error('Smoke demo runner is not active');
      }
      return runner.startRecording();
    },
    stopRecording() {
      const runner = app.getSmokeRunner();
      if (!runner) {
        throw new Error('Smoke demo runner is not active');
      }
      return runner.stopRecording();
    },
    exportTraceBytes(opts) {
      const runner = app.getSmokeRunner();
      return runner ? runner.exportTraceBytes(opts) : new Uint8Array(0);
    },
    exportTraceMetadata() {
      const runner = app.getSmokeRunner();
      return runner ? runner.exportTraceMetadata() : null;
    },
  };

  const hudController = new HudController({
    sessionController: app.session,
    smokeDemoRunner: smokeRunnerProxy,
    storageManager: app.storageManager,
    crashReporter: app.crashReporter,
    perfMonitor: app.perfMonitor,
    inputProfiles: app.inputProfiles,
    hud,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
    now: clock.now,
  });

  // Populate #select-input-profile from app.inputProfiles.listProfiles().
  const profileSelect = documentObj.getElementById('select-input-profile');
  const profiles = app.inputProfiles.listProfiles();
  assert.ok(profiles.length >= 4, 'Expected built-in input profiles to be available');
  for (const profile of profiles) {
    const opt = documentObj.createElement('option');
    opt.value = profile.id;
    opt.textContent = profile.name;
    profileSelect.appendChild(opt);
  }
  profileSelect.value = app.inputProfiles.getActiveProfile().id;

  hudController.bindButtons({
    smokeDemoBtn: documentObj.getElementById('btn-smoke-demo'),
    restartBtn: documentObj.getElementById('btn-restart-session'),
    recordTraceBtn: documentObj.getElementById('btn-record-trace'),
    exportTraceBtn: documentObj.getElementById('btn-export-trace'),
    clearCacheBtn: documentObj.getElementById('btn-clear-cache'),
    exportSavesBtn: documentObj.getElementById('btn-export-saves'),
    exportCrashBtn: documentObj.getElementById('btn-export-crash'),
    profileSelect,
  });

  assert.equal(hudController.getState().boundControlCount, 8);

  // 1. Boot app: dist/runtime.wasm is absent, so it should auto-start G5 WebGPU Smoke Demo.
  const startResult = await app.start();
  assert.equal(startResult.ok, true);
  assert.equal(app.session.getState(), SESSION_STATE.RUNNING_SMOKE);
  assert.equal(app.session.getMode(), SESSION_MODE.SMOKE);
  assert.ok(app.getSmokeRunner(), 'SmokeDemoRunner must be active');

  // 2. Click Record Trace -> flush 2 frames -> stop recording -> click Export .pg6trace.
  mockElements['btn-record-trace'].dispatch('click');
  await flushMicrotasks();
  assert.equal(hudController.getState().isRecording, true);
  assert.equal(mockElements['btn-record-trace'].dataset.recording, 'true');

  raf.flushFrame(16);
  raf.flushFrame(16);

  mockElements['btn-record-trace'].dispatch('click');
  await flushMicrotasks();
  assert.equal(hudController.getState().isRecording, false);
  assert.equal(mockElements['btn-record-trace'].dataset.recording, 'false');

  mockElements['btn-export-trace'].dispatch('click');
  await flushMicrotasks();
  const traceDownload = downloads.find((d) => d.filename === DEFAULT_FILENAMES.GPU_TRACE);
  assert.ok(traceDownload, 'Export .pg6trace button must trigger trace download');
  assert.ok(traceDownload.data instanceof Uint8Array);
  assert.ok(traceDownload.data.byteLength > 8);
  const traceView = new DataView(
    traceDownload.data.buffer,
    traceDownload.data.byteOffset,
    traceDownload.data.byteLength
  );
  assert.equal(
    traceView.getUint16(0, true),
    1,
    'Exported .pg6trace stream must start with BEGIN_FRAME (opcode 1) packet header'
  );

  // 3. Click Smoke Demo toggle to stop, then click again to restart smoke demo.
  mockElements['btn-smoke-demo'].dispatch('click');
  await flushMicrotasks();
  assert.equal(app.session.getState(), SESSION_STATE.STOPPED);
  assert.equal(app.getSmokeRunner(), null);

  mockElements['btn-smoke-demo'].dispatch('click');
  await flushMicrotasks();
  assert.equal(app.session.getState(), SESSION_STATE.RUNNING_SMOKE);
  assert.ok(app.getSmokeRunner(), 'Clicking Smoke Demo again must restart SmokeDemoRunner');

  // 4. Switch input profile via #select-input-profile change event.
  profileSelect.value = 'fps';
  profileSelect.dispatch('change', { target: profileSelect });
  await flushMicrotasks();
  assert.equal(app.inputProfiles.getActiveProfile().id, 'fps');
  assert.equal(hudController.getState().activeProfileId, 'fps');
  assert.ok(localStorage._map.size > 0, 'Switching profile must persist to storage');

  // 5. Click Export Report (#btn-export-crash) and verify sanitized JSON download.
  mockElements['btn-export-crash'].dispatch('click');
  await flushMicrotasks();
  const reportDownload = downloads.find((d) => d.filename === DEFAULT_FILENAMES.CRASH_REPORT);
  assert.ok(reportDownload, 'Export Report button must trigger diagnostic JSON download');
  const parsedReport = JSON.parse(reportDownload.data);
  assert.equal(typeof parsedReport.schemaVersion, 'number');
  assert.ok(parsedReport.serverStatus);

  // 6. Click Clear Cache and Export Saves to confirm storage wiring.
  mockElements['btn-clear-cache'].dispatch('click');
  await flushMicrotasks();
  assert.equal(hudController.getState().lastAction?.ok, true);

  mockElements['btn-export-saves'].dispatch('click');
  await flushMicrotasks();
  const savesDownload = downloads.find((d) => d.filename === DEFAULT_FILENAMES.USERDATA_BACKUP);
  assert.ok(savesDownload, 'Export Saves button must trigger userdata backup JSON download');

  // 7. Click Restart (#btn-restart-session).
  mockElements['btn-restart-session'].dispatch('click');
  await flushMicrotasks();
  assert.equal(app.session.getState(), SESSION_STATE.RUNNING_SMOKE);

  // 8. Clean teardown (mirrors pagehide handler in web/index.html).
  hudController.destroy();
  await app.destroy();
  assert.equal(app.destroyed, true);
  assert.equal(hudController.getState().destroyed, true);
  assert.equal(hudController.getState().boundControlCount, 0);
  for (const [id, el] of Object.entries(mockElements)) {
    assert.equal(
      el.listenerCount(),
      0,
      `Expected zero remaining event listeners on #${id} after hudController.destroy()`
    );
  }
});

test('web/index.html wires W45 splash overlay root, UMD boot scripts, and global hooks', () => {
  assert.match(indexHtml, /id="splash-overlay-root"/);
  assert.match(indexHtml, /<script\s+src="\.\/runtime\/pgim_preloader\.js"><\/script>/);
  assert.match(indexHtml, /<script\s+src="\.\/ui\/splash_overlay\.js"><\/script>/);
  assert.match(indexHtml, /<script\s+src="\.\/runtime\/boot_orchestrator\.js"><\/script>/);
  assert.equal(fs.existsSync(path.join(REPO_ROOT, 'web', 'runtime', 'pgim_preloader.js')), true);
  assert.equal(fs.existsSync(path.join(REPO_ROOT, 'web', 'ui', 'splash_overlay.js')), true);
  assert.equal(fs.existsSync(path.join(REPO_ROOT, 'web', 'runtime', 'boot_orchestrator.js')), true);

  const moduleScript = extractModuleScript(indexHtml);
  assert.match(moduleScript, /getElementById\(\s*['"]splash-overlay-root['"]\s*\)/);
  assert.match(moduleScript, /splashContainer:\s*splashOverlayRoot/);
  assert.match(moduleScript, /window\.playgta6SplashOverlay\s*=\s*app\.splashOverlay/);
  assert.match(moduleScript, /window\.playgta6BootOrchestrator\s*=\s*app\.bootOrchestrator/);
});

test('web/index.html wires W55 GpuBootPanelController script, #gpu-boot-panel-root container, bootstrap option, and HUD export control', async () => {
  assert.match(indexHtml, /<script\s+src="\.\/ui\/gpu_boot_panel\.js"><\/script>/);
  assert.equal(fs.existsSync(path.join(REPO_ROOT, 'web', 'ui', 'gpu_boot_panel.js')), true);

  // #gpu-boot-panel-root must live inside #hud-body.
  const hudBodyIdx = indexHtml.indexOf('id="hud-body"');
  const panelRootIdx = indexHtml.indexOf('id="gpu-boot-panel-root"');
  const scriptIdx = indexHtml.indexOf('<script src="./ui/gpu_boot_panel.js">');
  assert.ok(hudBodyIdx !== -1 && panelRootIdx > hudBodyIdx && panelRootIdx < scriptIdx);
  assert.match(indexHtml, /id="btn-export-missing-shaders"/);

  const moduleScript = extractModuleScript(indexHtml);
  assert.match(moduleScript, /getElementById\(\s*['"]gpu-boot-panel-root['"]\s*\)/);
  assert.match(moduleScript, /getElementById\(\s*['"]btn-export-missing-shaders['"]\s*\)/);
  assert.match(moduleScript, /bootstrapPlayGta6App\(\{[\s\S]*\bgpuBootPanelRoot\b[\s\S]*\}\)/);
  assert.match(moduleScript, /PlayGta6GpuBootPanel\?\.\s*GpuBootPanelController/);
  assert.match(moduleScript, /onExportMissingShaders\s*\(/);
  assert.match(moduleScript, /playgta6-missing-shaders\.json/);
  assert.match(moduleScript, /new\s+HudController\(\{[\s\S]*\bgpuBootPanel\b[\s\S]*\}\)/);
  assert.match(moduleScript, /exportMissingShadersBtn/);
  assert.match(moduleScript, /window\.playgta6GpuBootPanel\s*=\s*gpuBootPanel/);
  assert.match(moduleScript, /gpuBootPanel\.destroy\(\)/);

  // Integration check: mount real GpuBootPanelController into mock #gpu-boot-panel-root
  // and wire it with HudController using the same guard pattern as web/index.html.
  function createPanelMockElement(tagName = 'DIV') {
    const el = createMockEventTarget({
      tagName: tagName.toUpperCase(),
      className: '',
      hidden: false,
      style: {},
      parentNode: null,
    });
    const baseAppend = el.appendChild.bind(el);
    el.appendChild = (child) => {
      if (child && typeof child === 'object') {
        child.parentNode = el;
      }
      return baseAppend(child);
    };
    el.removeChild = (child) => {
      const idx = el.children.indexOf(child);
      if (idx !== -1) {
        el.children.splice(idx, 1);
      }
      if (child && typeof child === 'object') {
        child.parentNode = null;
      }
      return child;
    };
    return el;
  }

  const head = createPanelMockElement('HEAD');
  const panelRoot = createPanelMockElement('DIV');
  const exportBtn = createMockEventTarget({ textContent: 'Missing Shaders' });
  const panelDoc = {
    head,
    body: createPanelMockElement('BODY'),
    createElement(tag) {
      return createPanelMockElement(tag);
    },
    querySelector() {
      return null;
    },
  };

  const downloads = [];
  const hud = createMockHud();
  let hudController = null;

  const gpuBootPanel = new GpuBootPanelController({
    container: panelRoot,
    documentObj: panelDoc,
    collapsed: true,
    now: () => 1700000000000,
    onExportMissingShaders(jsonText) {
      if (hudController && hudController.isExportingMissingShaders) {
        return;
      }
      downloads.push({
        source: 'panel',
        filename: DEFAULT_FILENAMES.MISSING_SHADERS,
        data: jsonText,
        mimeType: 'application/json',
      });
    },
  });

  assert.equal(panelRoot.children.length, 1, 'GpuBootPanelController must mount into #gpu-boot-panel-root');

  hudController = new HudController({
    hud,
    gpuBootPanel,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ source: 'hud', filename, data, mimeType });
    },
  });

  hudController.bindButtons({ exportMissingShadersBtn: exportBtn });

  const state = hudController.updateGpuBootTelemetry({
    bootSummary: { stage: 'D3D12_CREATE_DEVICE', milestonesReached: 7, totalMilestones: 9 },
    frameStats: { framesPresented: 2, droppedFrames: 0 },
    fallbackStats: { totalLookups: 4, packHits: 3, fallbackHits: 1, builtinHits: 0 },
    missingShaders: [
      { hash: '0x00000000deadbeef', stage: 'fragment', hits: 5, hostPath: 'C:\\secret\\path.dxil' },
    ],
  });

  assert.equal(state.boot.stage, 'D3D12_CREATE_DEVICE');
  assert.equal(state.totalMissingShaders, 1);

  // Clicking #btn-export-missing-shaders should download exactly once (via HudController).
  exportBtn.dispatch('click');
  await flushMicrotasks();
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].source, 'hud');
  assert.equal(downloads[0].filename, DEFAULT_FILENAMES.MISSING_SHADERS);
  assert.equal(downloads[0].mimeType, 'application/json');
  assert.doesNotMatch(downloads[0].data, /secret/);

  const parsed = JSON.parse(downloads[0].data);
  assert.equal(parsed.format, EXPORT_FORMAT);
  assert.equal(parsed.missingShaderCount, 1);
  assert.equal(parsed.missingShaders[0].hash, '0x00000000deadbeef');
  assert.equal(hudController.getState().lastAction?.action, HUD_ACTIONS.EXPORT_MISSING_SHADERS);
  assert.equal(hudController.getState().lastAction?.ok, true);

  // Direct export on panel triggers panel's own onExportMissingShaders download helper.
  gpuBootPanel.exportMissingShadersJson();
  assert.equal(downloads.length, 2);
  assert.equal(downloads[1].source, 'panel');

  hudController.destroy();
  gpuBootPanel.destroy();
  assert.equal(exportBtn.listenerCount('click'), 0);
  assert.equal(panelRoot.children.length, 0);
});
