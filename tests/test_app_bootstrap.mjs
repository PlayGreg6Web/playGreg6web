import test from 'node:test';
import assert from 'node:assert/strict';

import bootstrapDefault, {
  DEFAULT_LOADER_URL,
  PlayGta6AppHandle,
  bootstrapPlayGta6App,
  detectWasm64,
  extractBootAssetUrls,
  resolveDefaultGameBase,
} from '../web/app_bootstrap.js';
import { SESSION_MODE, SESSION_STATE } from '../web/runtime/session_controller.js';
import { SmokeDemoRunner } from '../web/demo/run_smoke_demo.js';
import { INPUT_BUTTONS } from '../web/input/input_bridge.js';
import { InputProfileManager } from '../web/input/input_profiles.js';

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

function createFakeEventTarget(extra = {}) {
  const listeners = new Map();
  return {
    ...extra,
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
        entry.fn({ type, ...event });
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

function createMemoryStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem(key) {
      return map.has(key) ? map.get(key) : null;
    },
    setItem(key, value) {
      map.set(key, String(value));
    },
    removeItem(key) {
      map.delete(key);
    },
    clear() {
      map.clear();
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
    pendingCount() {
      return scheduled.size;
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
    tickAll() {
      for (const entry of [...active.values()]) {
        entry.fn();
      }
    },
    activeCount() {
      return active.size;
    },
  };
}

function createMockHud() {
  const statuses = [];
  const logs = [];
  const metricsUpdates = [];
  return {
    statuses,
    logs,
    metricsUpdates,
    setStatus(text, level = 'info') {
      statuses.push({ text, level });
    },
    appendLog(message, level = 'info') {
      logs.push({ message, level });
    },
    updateMetrics(metrics) {
      metricsUpdates.push({ ...metrics });
    },
  };
}

function createMockFetch(payload, { ok = true, status = 200, throwError = null } = {}) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    if (throwError) {
      throw throwError;
    }
    return {
      ok,
      status,
      async json() {
        return typeof payload === 'function' ? payload() : payload;
      },
    };
  };
  fetchImpl.calls = calls;
  return fetchImpl;
}

function createMockWebGpuEnv() {
  let nextHandle = 1;
  let swapchainViewCounter = 0;
  const configureCalls = [];
  let unconfigureCount = 0;
  let deviceDestroyed = false;

  let resolveLost;
  const lost = new Promise((resolve) => {
    resolveLost = resolve;
  });

  const context = {
    configure(config) {
      configureCalls.push({ ...config });
    },
    unconfigure() {
      unconfigureCount += 1;
    },
    getCurrentTexture() {
      const id = `swapchain-view-${swapchainViewCounter++}`;
      const view = { kind: 'GPUTextureView', id };
      return {
        createView() {
          return view;
        },
      };
    },
  };

  const canvas = createFakeEventTarget({
    width: 640,
    height: 360,
    clientWidth: 640,
    clientHeight: 360,
    getContext(kind) {
      return kind === 'webgpu' ? context : null;
    },
  });

  const device = createFakeEventTarget({
    limits: { maxTextureDimension2D: 8192 },
    lost,
    triggerLost(info = { reason: 'unknown', message: 'simulated GPU device loss' }) {
      resolveLost(info);
    },
    destroy() {
      deviceDestroyed = true;
    },
    createBuffer(desc) {
      return {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        writes: [],
        destroy() {
          this.destroyed = true;
        },
      };
    },
    createTexture(desc) {
      const handle = nextHandle++;
      return {
        kind: 'GPUTexture',
        handle,
        desc: { ...desc, size: { ...desc.size } },
        destroyed: false,
        createView() {
          return { kind: 'GPUTextureView', textureHandle: handle, handle: nextHandle++ };
        },
        destroy() {
          this.destroyed = true;
        },
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
        getBindGroupLayout(index) {
          return { kind: 'GPUBindGroupLayout', pipelineKind: 'render', index };
        },
      };
    },
    createComputePipeline(desc) {
      return {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index) {
          return { kind: 'GPUBindGroupLayout', pipelineKind: 'compute', index };
        },
      };
    },
    createBindGroup(desc) {
      return { kind: 'GPUBindGroup', handle: nextHandle++, desc };
    },
    createCommandEncoder() {
      const commandBuffer = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
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
        finish() {
          return commandBuffer;
        },
      };
    },
    queue: {
      writeBuffer(buffer, dstOffset, data) {
        buffer.writes.push({
          dstOffset,
          bytes: new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice(),
        });
      },
      writeTexture() {},
      submit() {},
    },
  });

  let requestAdapterCalls = 0;
  let requestDeviceCalls = 0;

  const adapter = {
    async requestDevice() {
      requestDeviceCalls += 1;
      return device;
    },
  };

  const gpu = {
    getPreferredCanvasFormat() {
      return 'bgra8unorm';
    },
    async requestAdapter() {
      requestAdapterCalls += 1;
      return adapter;
    },
  };

  return {
    gpu,
    adapter,
    device,
    context,
    canvas,
    configureCalls,
    get unconfigureCount() {
      return unconfigureCount;
    },
    get deviceDestroyed() {
      return deviceDestroyed;
    },
    get requestAdapterCalls() {
      return requestAdapterCalls;
    },
    get requestDeviceCalls() {
      return requestDeviceCalls;
    },
  };
}

function createMockBrowserHost({ clock, raf, timers, gpu, localStorage }) {
  const windowObj = createFakeEventTarget({
    crossOriginIsolated: true,
    innerWidth: 800,
    innerHeight: 600,
    devicePixelRatio: 1,
    location: { origin: 'https://playgta6.local' },
    localStorage: localStorage || createMemoryStorage(),
    requestAnimationFrame: raf.raf,
    cancelAnimationFrame: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  const documentObj = createFakeEventTarget({
    pointerLockElement: null,
    hidden: false,
    visibilityState: 'visible',
  });

  const navigatorObj = {
    gpu: gpu ?? null,
    getGamepads() {
      return [];
    },
    storage: {
      async estimate() {
        return { usage: 1024, quota: 1024 * 1024 };
      },
      async persisted() {
        return false;
      },
      async getDirectory() {
        return {
          async getDirectoryHandle() {
            const err = new Error('Not found');
            err.name = 'NotFoundError';
            throw err;
          },
        };
      },
    },
  };

  return { windowObj, documentObj, navigatorObj, clock };
}

test('exports and argument validation', () => {
  assert.equal(bootstrapDefault, bootstrapPlayGta6App);
  assert.equal(typeof DEFAULT_LOADER_URL, 'string');
  assert.ok(DEFAULT_LOADER_URL.endsWith('/loader.js'));

  assert.throws(() => bootstrapPlayGta6App(null), TypeError);
  assert.throws(() => bootstrapPlayGta6App({ webgpuAdapterFactory: 'bad' }), TypeError);
  assert.throws(() => bootstrapPlayGta6App({ createWasmWorker: 123 }), TypeError);
});

test('end-to-end smoke demo fallback when /api/status reports hasWasmBinary: false', async () => {
  const clock = createClock(2000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  // Persist a custom active input profile ('lefty-kb') in mocked localStorage beforehand.
  const storage = createMemoryStorage();
  const seedProfiles = new InputProfileManager({ storage });
  seedProfiles.setActiveProfile('lefty-kb');
  seedProfiles.save();

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
    localStorage: storage,
  });

  const fetchImpl = createMockFetch({
    ok: true,
    coopCoep: true,
    dataDirConfigured: false,
    hasManifest: false,
    hasBootset: false,
    hasWasmBinary: false,
    shaderPackCount: 0,
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl,
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  assert.ok(app instanceof PlayGta6AppHandle);
  assert.equal(app.destroyed, false);
  assert.equal(app.getSmokeRunner(), null);
  assert.equal(app.getWasmWorker(), null);
  assert.equal(app.crashReporter.isGlobalHandlersAttached, true);

  // Saved input profile was loaded and applied to InputBridge.
  assert.equal(app.inputProfiles.getActiveProfile().id, 'lefty-kb');
  windowObj.dispatch('keydown', { code: 'KeyI' });
  const inputSnap = app.inputBridge.getSnapshot();
  assert.ok(inputSnap);
  assert.notEqual(inputSnap.buttons & INPUT_BUTTONS.MOVE_FORWARD, 0);
  windowObj.dispatch('keyup', { code: 'KeyI' });

  // Start session -> auto-starts SmokeDemoRunner because hasWasmBinary is false.
  const startResult = await app.start();
  assert.equal(startResult.ok, true);
  assert.equal(startResult.mode, SESSION_MODE.SMOKE);
  assert.equal(startResult.state, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(webgpu.requestAdapterCalls, 1);
  assert.equal(webgpu.requestDeviceCalls, 1);

  const smokeRunner = app.getSmokeRunner();
  assert.ok(smokeRunner instanceof SmokeDemoRunner);
  assert.equal(smokeRunner.initialized, true);
  assert.equal(smokeRunner.running, true);
  assert.equal(smokeRunner.inputSab, app.inputBridge.sab);

  // Step frames via RAF loop and verify PerfMonitor + HUD receive live updates.
  const metricsBefore = hud.metricsUpdates.length;
  assert.ok(raf.flushFrame(16) >= 1);
  assert.ok(raf.flushFrame(16) >= 1);
  assert.ok(raf.flushFrame(16) >= 1);

  assert.equal(smokeRunner.frameIndex, 3);
  const perfSnap = app.perfMonitor.getSnapshot();
  assert.ok(perfSnap.frames.totalFrames >= 3);
  assert.ok(perfSnap.gpu.totalGpuPackets > 0);
  assert.ok(perfSnap.gpu.totalDrawCalls > 0);
  assert.ok(hud.metricsUpdates.length > metricsBefore);

  // Periodic PerfMonitor HUD timer tick also works.
  const countBeforeTick = hud.metricsUpdates.length;
  timers.tickAll();
  assert.equal(hud.metricsUpdates.length, countBeforeTick + 1);

  // Programmatic resize() reconfigures the smoke demo swapchain.
  const resized = app.resize(1280, 720, 2);
  assert.equal(resized, true);
  assert.equal(webgpu.canvas.width, 2560);
  assert.equal(webgpu.canvas.height, 1440);

  // Window resize event also routes through handle.resize().
  webgpu.canvas.clientWidth = 1024;
  webgpu.canvas.clientHeight = 576;
  windowObj.devicePixelRatio = 1;
  windowObj.dispatch('resize');
  assert.equal(webgpu.canvas.width, 1024);
  assert.equal(webgpu.canvas.height, 576);

  // Global error, unhandledrejection, and WebGPU validation/device-lost errors record into CrashReporter.
  windowObj.dispatch('error', {
    error: new Error('Boom at C:\\Users\\alice\\secret\\game.js'),
    filename: 'file:///C:/Users/alice/app.js',
    lineno: 42,
    colno: 7,
  });
  windowObj.dispatch('unhandledrejection', {
    reason: new Error('Rejected promise from /home/alice/data'),
  });
  webgpu.device.dispatch('uncapturederror', {
    error: { name: 'GPUValidationError', message: 'Invalid bind group layout' },
  });
  webgpu.device.triggerLost({ reason: 'unknown', message: 'Adapter reset' });
  await Promise.resolve();

  const errors = app.crashReporter.getErrors();
  assert.equal(errors.length, 4);
  assert.ok(!errors[0].message.includes('alice'));
  assert.equal(errors[1].category, 'unhandled_rejection');
  assert.ok(!errors[1].message.includes('/home/alice'));
  assert.equal(errors[2].category, 'webgpu_validation');
  assert.equal(errors[3].category, 'webgpu_device_lost');

  // Destroy cleanly tears down every subsystem and leaks zero listeners or timers.
  await app.destroy();
  assert.equal(app.destroyed, true);
  assert.equal(app.getSmokeRunner(), null);
  assert.equal(smokeRunner.destroyed, true);
  assert.equal(app.session.getState(), SESSION_STATE.STOPPED);
  assert.equal(app.inputBridge.disposed, true);
  assert.equal(app.perfMonitor.isHudAttached, false);
  assert.equal(app.crashReporter.isGlobalHandlersAttached, false);
  assert.equal(app.audioEngine.state, 'closed');
  assert.equal(timers.activeCount(), 0);
  assert.equal(windowObj.listenerCount(), 0);
  assert.equal(documentObj.listenerCount(), 0);
  assert.equal(webgpu.canvas.listenerCount(), 0);
  assert.equal(webgpu.device.listenerCount(), 0);

  // Destroy is idempotent and start() after destroy returns cleanly.
  await app.destroy();
  const afterDestroy = await app.start();
  assert.equal(afterDestroy.ok, false);
  assert.equal(afterDestroy.destroyed, true);
});

test('Wasm64 runtime path when /api/status reports hasWasmBinary: true invokes createWasmWorker', async () => {
  const clock = createClock(5000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const offscreenCanvas = { kind: 'MockOffscreenCanvas' };
  let transferCalls = 0;
  webgpu.canvas.transferControlToOffscreen = () => {
    transferCalls += 1;
    return offscreenCanvas;
  };

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  const workerCalls = [];
  const postedMessages = [];
  let terminatedCount = 0;

  const mockWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      postedMessages.push({ payload, transfer });
    },
    terminate() {
      terminatedCount += 1;
    },
  });

  const fetchImpl = createMockFetch({
    ok: true,
    coopCoep: true,
    dataDirConfigured: true,
    hasManifest: true,
    hasBootset: true,
    hasWasmBinary: true,
    shaderPackCount: 3,
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl,
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      workerCalls.push(args);
      return mockWorker;
    },
  });

  const startResult = await app.start();
  assert.equal(startResult.ok, true);
  assert.equal(startResult.mode, SESSION_MODE.WASM);
  assert.equal(startResult.state, SESSION_STATE.RUNNING_WASM);
  assert.equal(app.getSmokeRunner(), null);
  assert.equal(app.getWasmWorker(), mockWorker);
  assert.equal(webgpu.requestAdapterCalls, 0);

  assert.equal(workerCalls.length, 1);
  assert.equal(workerCalls[0].url, DEFAULT_LOADER_URL);
  assert.equal(workerCalls[0].inputSab, app.inputBridge.sab);
  assert.equal(workerCalls[0].audioSab, app.audioEngine.sab);
  assert.equal(workerCalls[0].dataBase, 'https://playgta6.local/data/');
  assert.equal(transferCalls, 1);

  assert.equal(postedMessages.length, 1);
  assert.equal(postedMessages[0].payload.canvas, offscreenCanvas);
  assert.equal(postedMessages[0].payload.width, 640);
  assert.equal(postedMessages[0].payload.height, 360);
  assert.equal(postedMessages[0].payload.inputSab, app.inputBridge.sab);
  assert.equal(postedMessages[0].payload.audioSab, app.audioEngine.sab);
  assert.deepEqual(postedMessages[0].transfer, [offscreenCanvas]);

  // resize() forwards pixel dimensions to the Wasm64 worker.
  const resized = app.resize(1920, 1080, 1.5);
  assert.equal(resized, true);
  assert.equal(postedMessages.length, 2);
  assert.deepEqual(postedMessages[1].payload, {
    resize: true,
    width: 2880,
    height: 1620,
  });

  // Worker error events are captured by CrashReporter and reported to HUD.
  mockWorker.dispatch('error', { message: 'wasm trap in worker' });
  const errs = app.crashReporter.getErrors();
  assert.equal(errs.length, 1);
  assert.equal(errs[0].category, 'worker_error');
  assert.match(errs[0].message, /wasm trap in worker/);

  // Stopping terminates worker and detaches its listeners.
  const stopResult = await app.stop();
  assert.equal(stopResult.ok, true);
  assert.equal(terminatedCount, 1);
  assert.equal(app.getWasmWorker(), null);
  assert.equal(mockWorker.listenerCount(), 0);

  await app.destroy();
  assert.equal(windowObj.listenerCount(), 0);
  assert.equal(documentObj.listenerCount(), 0);
  assert.equal(webgpu.canvas.listenerCount(), 0);
});

test('preferSmoke: true selects SmokeDemoRunner via injectable webgpuAdapterFactory even when hasWasmBinary: true', async () => {
  const clock = createClock(1000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: null, // Verified that injectable webgpuAdapterFactory works even without navigator.gpu
  });

  let wasmCalled = 0;
  let factoryCalled = 0;

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    webgpuAdapterFactory: async () => {
      factoryCalled += 1;
      return webgpu.adapter;
    },
    createWasmWorker: () => {
      wasmCalled += 1;
      return { postMessage() {}, terminate() {} };
    },
  });

  const res = await app.start({ preferSmoke: true });
  assert.equal(res.ok, true);
  assert.equal(res.mode, SESSION_MODE.SMOKE);
  assert.equal(wasmCalled, 0);
  assert.equal(factoryCalled, 1);
  assert.ok(app.getSmokeRunner() instanceof SmokeDemoRunner);

  // Stop and restart smoke demo cleanly.
  const firstRunner = app.getSmokeRunner();
  await app.stop();
  assert.equal(firstRunner.destroyed, true);
  assert.equal(app.getSmokeRunner(), null);

  const res2 = await app.start({ preferSmoke: true });
  assert.equal(res2.ok, true);
  assert.equal(res2.mode, SESSION_MODE.SMOKE);
  assert.notEqual(app.getSmokeRunner(), firstRunner);

  await app.destroy();
  assert.equal(windowObj.listenerCount(), 0);
  assert.equal(documentObj.listenerCount(), 0);
  assert.equal(webgpu.canvas.listenerCount(), 0);
});

test('graceful handling of corrupt persisted input profiles and session probe/launcher failures', async () => {
  const clock = createClock(1000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  // Corrupt JSON in localStorage should be recorded in CrashReporter, not crash bootstrap.
  const corruptStorage = createMemoryStorage({
    'playgta6.inputProfiles.v1': '{not valid json',
  });

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: {
      async requestAdapter() {
        return null; // Simulate missing GPU adapter
      },
    },
    localStorage: corruptStorage,
  });

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  // Default profile active despite corrupt storage; load error captured.
  assert.equal(app.inputProfiles.getActiveProfile().id, 'default');
  assert.equal(app.crashReporter.errorCount, 1);
  assert.equal(app.crashReporter.getErrors()[0].category, 'io_error');

  // Starting smoke demo fails cleanly when GPU adapter is null and records the error.
  const failed = await app.start();
  assert.equal(failed.ok, false);
  assert.equal(failed.state, SESSION_STATE.ERROR);
  assert.equal(app.getSmokeRunner(), null);
  assert.equal(app.crashReporter.errorCount, 2);
  assert.match(app.crashReporter.getErrors()[1].message, /GPUAdapter/i);

  await app.destroy();
  assert.equal(windowObj.listenerCount(), 0);
  assert.equal(documentObj.listenerCount(), 0);
  assert.equal(webgpu.canvas.listenerCount(), 0);
});

test('W45 wires BootTracer, SplashOverlayController, and BootOrchestrator through /api/status discovery and FIRST_PRESENT dismissal', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { SplashOverlayController } = require('../web/ui/splash_overlay.js');
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');

  const clock = createClock(1000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  function makeMockNode(tag = 'div') {
    const attrs = new Map();
    const styles = new Map();
    const listeners = new Map();
    const node = {
      tagName: String(tag).toUpperCase(),
      className: '',
      hidden: false,
      textContent: '',
      innerHTML: '',
      parentNode: null,
      children: [],
      style: {
        setProperty(k, v) {
          styles.set(k, String(v));
        },
        getPropertyValue(k) {
          return styles.get(k) || '';
        },
      },
      classList: {
        add(c) {
          const set = new Set(String(node.className || '').split(/\s+/).filter(Boolean));
          set.add(c);
          node.className = [...set].join(' ');
        },
        remove(c) {
          const set = new Set(String(node.className || '').split(/\s+/).filter(Boolean));
          set.delete(c);
          node.className = [...set].join(' ');
        },
        contains(c) {
          return String(node.className || '').split(/\s+/).includes(c);
        },
      },
      setAttribute(k, v) {
        attrs.set(k, String(v));
      },
      getAttribute(k) {
        return attrs.has(k) ? attrs.get(k) : null;
      },
      removeAttribute(k) {
        attrs.delete(k);
      },
      appendChild(child) {
        child.parentNode = node;
        node.children.push(child);
        return child;
      },
      removeChild(child) {
        const idx = node.children.indexOf(child);
        if (idx >= 0) node.children.splice(idx, 1);
        child.parentNode = null;
        return child;
      },
      addEventListener(type, fn) {
        if (!listeners.has(type)) listeners.set(type, new Set());
        listeners.get(type).add(fn);
      },
      removeEventListener(type, fn) {
        listeners.get(type)?.delete(fn);
      },
      querySelector() {
        return null;
      },
    };
    return node;
  }

  const splashContainer = makeMockNode('div');
  const head = makeMockNode('head');
  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
  });
  documentObj.head = head;
  documentObj.createElement = (tag) => makeMockNode(tag);
  documentObj.querySelector = () => null;

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      hasWasmBinary: false,
      splashImage: '/game/xbs_splash.png',
    }),
    canvas: webgpu.canvas,
    splashContainer,
    SplashOverlayCtor: SplashOverlayController,
    BootOrchestratorCtor: BootOrchestrator,
    splashOverlayOptions: { fadeMs: 0 },
    hud,
    webgpuAdapterFactory: async () => webgpu.adapter,
    now: clock.now,
    raf: raf.raf,
    cancelRaf: raf.cancelRaf,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
  });

  assert.ok(app.bootTracer, 'bootTracer must be exposed');
  assert.ok(app.splashOverlay, 'splashOverlay must be mounted when splashContainer is provided');
  assert.ok(app.bootOrchestrator, 'bootOrchestrator must be wired when splashOverlay is active');
  assert.equal(app.splashOverlay.getState().visible, true);

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.SMOKE);

  // Splash image discovered from /api/status must be applied and dismissed on smoke FIRST_PRESENT.
  const splashState = app.splashOverlay.getState();
  assert.equal(splashState.splashImageUrl, '/game/xbs_splash.png');
  assert.equal(splashState.dismissed, true);
  assert.equal(splashState.visible, false);
  assert.equal(app.bootTracer.getSummary().readyForSplash, true);

  await app.destroy();
  assert.equal(app.splashOverlay.getState().destroyed, true);
  assert.equal(app.bootOrchestrator.getState().destroyed, true);
});

test('W49 startWasmRuntime forwards bootBase, shaderPackUrl, reflectionManifestUrl, and pgimUrl from /api/status', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const { BootOrchestrator } = require('../web/runtime/boot_orchestrator.js');
  const { createWorkerBootInitPayloads } = require('../web/loader.js');

  const clock = createClock(7000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const offscreenCanvas = { kind: 'MockOffscreenCanvas' };
  webgpu.canvas.transferControlToOffscreen = () => offscreenCanvas;

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  const workerCalls = [];
  const postedMessages = [];
  const mockWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      postedMessages.push({ payload, transfer });
    },
    terminate() {},
  });

  const statusPayload = {
    ok: true,
    coopCoep: true,
    dataDirConfigured: true,
    hasManifest: true,
    hasBootset: true,
    hasWasmBinary: true,
    shaderPackCount: 1,
    shaderPackUrl: '/shaders/fallback.wgsl.pack',
    reflectionManifestUrl: '/shaders/reflection-manifest.json',
    pgimUrl: '/boot/guest.pgim',
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch(statusPayload),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      workerCalls.push(args);
      return mockWorker;
    },
  });

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.WASM);
  assert.equal(workerCalls.length, 1);
  assert.equal(postedMessages.length, 1);

  assert.equal(workerCalls[0].bootBase, 'https://playgta6.local/boot/');
  assert.equal(workerCalls[0].shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(workerCalls[0].reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(workerCalls[0].pgimUrl, '/boot/guest.pgim');

  const initPayload = postedMessages[0].payload;
  assert.equal(initPayload.bootBase, 'https://playgta6.local/boot/');
  assert.equal(initPayload.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(initPayload.reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(initPayload.pgimUrl, '/boot/guest.pgim');

  // Verify W47 loader init payload forwarding consumes the exact fields posted by app_bootstrap.
  const workerInits = createWorkerBootInitPayloads(initPayload);
  assert.equal(workerInits.ioInit.bootBase, 'https://playgta6.local/boot/');
  assert.equal(workerInits.ioInit.pgimUrl, '/boot/guest.pgim');
  assert.equal(workerInits.gpuInit.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(workerInits.gpuInit.reflectionManifestUrl, '/shaders/reflection-manifest.json');

  await app.destroy();

  // Also verify BootOrchestrator discovery path and default '/boot/' fallback when location.origin is absent.
  const orchWorkerCalls = [];
  const orchPostedMessages = [];
  const orchWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      orchPostedMessages.push({ payload, transfer });
    },
    terminate() {},
  });
  const hostNoOrigin = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  hostNoOrigin.windowObj.location = { origin: 'null' };

  const orchApp = bootstrapPlayGta6App({
    windowObj: hostNoOrigin.windowObj,
    documentObj: hostNoOrigin.documentObj,
    navigatorObj: hostNoOrigin.navigatorObj,
    fetchImpl: createMockFetch(statusPayload),
    canvas: webgpu.canvas,
    hud,
    enableBootOrchestrator: true,
    BootOrchestratorCtor: BootOrchestrator,
    pgimPreloader: {
      async preloadBootBundle() {
        return { pgim: new Uint8Array(4), shaderPackBytes: null, reflectionManifest: null, bytesLoaded: 4 };
      },
    },
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      orchWorkerCalls.push(args);
      return orchWorker;
    },
  });

  const orchStarted = await orchApp.start();
  assert.equal(orchStarted.ok, true);
  assert.equal(orchStarted.mode, SESSION_MODE.WASM);
  assert.equal(orchWorkerCalls.length, 1);
  assert.equal(orchPostedMessages.length, 1);

  assert.equal(orchWorkerCalls[0].bootBase, '/boot/');
  assert.equal(orchWorkerCalls[0].shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(orchWorkerCalls[0].reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(orchWorkerCalls[0].pgimUrl, '/boot/guest.pgim');

  assert.equal(orchPostedMessages[0].payload.bootBase, '/boot/');
  assert.equal(orchPostedMessages[0].payload.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(orchPostedMessages[0].payload.reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(orchPostedMessages[0].payload.pgimUrl, '/boot/guest.pgim');

  await orchApp.destroy();
});

test('W54 resolveDefaultGameBase and extractBootAssetUrls discover gameBase and programManifestUrl', () => {
  assert.equal(
    resolveDefaultGameBase({ gameBase: 'https://cdn.local/custom-game/' }, { location: { origin: 'https://playgta6.local' } }),
    'https://cdn.local/custom-game/'
  );
  assert.equal(
    resolveDefaultGameBase({}, { location: { origin: 'https://playgta6.local' } }),
    'https://playgta6.local/game/'
  );
  assert.equal(resolveDefaultGameBase({}, { location: { origin: 'null' } }), '/game/');
  assert.equal(resolveDefaultGameBase({}, null), '/game/');

  assert.deepEqual(
    extractBootAssetUrls({
      bootBase: '/boot/',
      gameBase: '/game-alt/',
      shaderPackUrl: '/shaders/pack.wgsl',
      reflectionManifestUrl: '/shaders/reflection.json',
      pgimUrl: '/boot/guest.pgim',
      programManifestUrl: '/boot/program-manifest.json',
    }),
    {
      bootBase: '/boot/',
      gameBase: '/game-alt/',
      shaderPackUrl: '/shaders/pack.wgsl',
      reflectionManifestUrl: '/shaders/reflection.json',
      pgimUrl: '/boot/guest.pgim',
      programManifestUrl: '/boot/program-manifest.json',
    }
  );

  assert.deepEqual(
    extractBootAssetUrls({
      programManifest: { url: '/boot/alt-program-manifest.json' },
    }),
    {
      programManifestUrl: '/boot/alt-program-manifest.json',
    }
  );
});

test('W54 startWasmRuntime forwards gameBase and programManifestUrl into createWasmWorker and worker.postMessage', async () => {
  const clock = createClock(8000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const offscreenCanvas = { kind: 'MockOffscreenCanvas' };
  webgpu.canvas.transferControlToOffscreen = () => offscreenCanvas;

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  const workerCalls = [];
  const postedMessages = [];
  const mockWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      postedMessages.push({ payload, transfer });
    },
    terminate() {},
  });

  const statusPayload = {
    ok: true,
    coopCoep: true,
    dataDirConfigured: true,
    hasManifest: true,
    hasBootset: true,
    hasWasmBinary: true,
    shaderPackCount: 2,
    shaderPackUrl: '/shaders/fallback.wgsl.pack',
    reflectionManifestUrl: '/shaders/reflection-manifest.json',
    pgimUrl: '/boot/guest.pgim',
    programManifestUrl: '/boot/program-manifest.json',
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch(statusPayload),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      workerCalls.push(args);
      return mockWorker;
    },
  });

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.WASM);
  assert.equal(workerCalls.length, 1);
  assert.equal(postedMessages.length, 1);

  assert.equal(workerCalls[0].bootBase, 'https://playgta6.local/boot/');
  assert.equal(workerCalls[0].gameBase, 'https://playgta6.local/game/');
  assert.equal(workerCalls[0].shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(workerCalls[0].reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(workerCalls[0].pgimUrl, '/boot/guest.pgim');
  assert.equal(workerCalls[0].programManifestUrl, '/boot/program-manifest.json');

  const initPayload = postedMessages[0].payload;
  assert.equal(initPayload.bootBase, 'https://playgta6.local/boot/');
  assert.equal(initPayload.gameBase, 'https://playgta6.local/game/');
  assert.equal(initPayload.shaderPackUrl, '/shaders/fallback.wgsl.pack');
  assert.equal(initPayload.reflectionManifestUrl, '/shaders/reflection-manifest.json');
  assert.equal(initPayload.pgimUrl, '/boot/guest.pgim');
  assert.equal(initPayload.programManifestUrl, '/boot/program-manifest.json');

  await app.destroy();

  // Verify discovered gameBase + programManifest alias and '/game/' fallback when origin is 'null'.
  const altWorkerCalls = [];
  const altPostedMessages = [];
  const altWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      altPostedMessages.push({ payload, transfer });
    },
    terminate() {},
  });
  const hostNoOrigin = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  hostNoOrigin.windowObj.location = { origin: 'null' };

  const altApp = bootstrapPlayGta6App({
    windowObj: hostNoOrigin.windowObj,
    documentObj: hostNoOrigin.documentObj,
    navigatorObj: hostNoOrigin.navigatorObj,
    fetchImpl: createMockFetch({
      ok: true,
      hasWasmBinary: true,
      gameBase: '/custom-game-root/',
      programManifest: '/boot/alias-program-manifest.json',
      pgimUrl: '/boot/guest.pgim',
    }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      altWorkerCalls.push(args);
      return altWorker;
    },
  });

  const altStarted = await altApp.start();
  assert.equal(altStarted.ok, true);
  assert.equal(altWorkerCalls[0].gameBase, '/custom-game-root/');
  assert.equal(altWorkerCalls[0].programManifestUrl, '/boot/alias-program-manifest.json');
  assert.equal(altPostedMessages[0].payload.gameBase, '/custom-game-root/');
  assert.equal(altPostedMessages[0].payload.programManifestUrl, '/boot/alias-program-manifest.json');

  await altApp.destroy();

  // Verify '/game/' default fallback when neither options.gameBase, status.gameBase, nor valid origin is set.
  const fallbackWorkerCalls = [];
  const fallbackPostedMessages = [];
  const fallbackWorker = createFakeEventTarget({
    postMessage(payload, transfer) {
      fallbackPostedMessages.push({ payload, transfer });
    },
    terminate() {},
  });

  const fallbackApp = bootstrapPlayGta6App({
    windowObj: hostNoOrigin.windowObj,
    documentObj: hostNoOrigin.documentObj,
    navigatorObj: hostNoOrigin.navigatorObj,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker(args) {
      fallbackWorkerCalls.push(args);
      return fallbackWorker;
    },
  });

  const fallbackStarted = await fallbackApp.start();
  assert.equal(fallbackStarted.ok, true);
  assert.equal(fallbackWorkerCalls[0].gameBase, '/game/');
  assert.equal(fallbackPostedMessages[0].payload.gameBase, '/game/');
  assert.equal(fallbackWorkerCalls[0].programManifestUrl, undefined);
  assert.equal(fallbackPostedMessages[0].payload.programManifestUrl, undefined);

  await fallbackApp.destroy();
});

test('W54 connects BootTracer updates and worker gpu-telemetry events to injected gpuBootPanel and cleans up on destroy', async () => {
  const clock = createClock(9000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  const panelUpdates = [];
  let panelDestroyCalls = 0;
  const mockGpuBootPanel = {
    update(snapshot) {
      panelUpdates.push(snapshot);
    },
    destroy() {
      panelDestroyCalls += 1;
    },
  };

  const app = bootstrapPlayGta6App({
    windowObj,
    documentObj,
    navigatorObj,
    fetchImpl: createMockFetch({ ok: true, hasWasmBinary: true }),
    canvas: webgpu.canvas,
    hud,
    gpuBootPanel: mockGpuBootPanel,
    now: clock.now,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    createWasmWorker() {
      return mockWorker;
    },
  });

  assert.equal(app.gpuBootPanel, mockGpuBootPanel);
  assert.equal(panelUpdates.length, 0);

  const started = await app.start();
  assert.equal(started.ok, true);
  assert.equal(started.mode, SESSION_MODE.WASM);
  assert.equal(mockWorker.listenerCount('message'), 1);

  // BootTracer milestone updates flow into gpuBootPanel.update({ bootSummary }).
  app.bootTracer.markMilestone('PGIM_LOAD');
  assert.equal(panelUpdates.length, 1);
  assert.equal(panelUpdates[0].bootSummary.stage, 'PGIM_LOAD');
  assert.equal(panelUpdates[0].bootSummary.milestonesReached, 1);

  // BootTracer ingestLogLine updates flow once (without duplicate updates).
  app.bootTracer.ingestLogLine('[boot] D3D12_CREATE_DEVICE');
  assert.equal(panelUpdates.length, 2);
  assert.equal(panelUpdates[1].bootSummary.stage, 'D3D12_CREATE_DEVICE');

  // Unrelated log lines do not emit spurious panel updates.
  app.bootTracer.ingestLogLine('ordinary non-milestone message');
  assert.equal(panelUpdates.length, 2);

  // Worker gpu-telemetry messages forward bootSummary, frameStats, fallbackStats, missingShaders.
  const frameStats = { framesPresented: 4, droppedFrames: 1 };
  const fallbackStats = { totalLookups: 10, packHits: 8, fallbackHits: 2, builtinHits: 0 };
  const missingShaders = [{ hash: '0x00000000deadbeef', stage: 'fragment', hits: 2 }];

  mockWorker.dispatch('message', {
    data: {
      type: 'gpu-telemetry',
      frameStats,
      fallbackStats,
      missingShaders,
    },
  });

  assert.equal(panelUpdates.length, 3);
  assert.equal(panelUpdates[2].bootSummary.stage, 'D3D12_CREATE_DEVICE');
  assert.deepEqual(panelUpdates[2].frameStats, frameStats);
  assert.deepEqual(panelUpdates[2].fallbackStats, fallbackStats);
  assert.deepEqual(panelUpdates[2].missingShaders, missingShaders);

  // Non-telemetry worker messages are ignored.
  mockWorker.dispatch('message', { data: { type: 'other-event', ok: true } });
  assert.equal(panelUpdates.length, 3);

  // destroy() removes worker and tracer hooks, and does NOT destroy an externally-owned gpuBootPanel.
  await app.destroy();
  assert.equal(mockWorker.listenerCount('message'), 0);
  assert.equal(mockWorker.listenerCount(), 0);
  assert.equal(panelDestroyCalls, 0);

  // Subsequent worker telemetry or BootTracer calls after destroy() must not invoke gpuBootPanel.update().
  mockWorker.dispatch('message', {
    data: { type: 'gpu-telemetry', frameStats: { framesPresented: 99, droppedFrames: 0 } },
  });
  app.bootTracer.markMilestone('FIRST_PRESENT');
  assert.equal(panelUpdates.length, 3);
});

test('W54 constructs owned GpuBootPanelController from gpuBootPanelRoot and globalThis auto-discovery and destroys on teardown', async () => {
  const { createRequire } = await import('node:module');
  const require = createRequire(import.meta.url);
  const gpuBootPanelApi = require('../web/ui/gpu_boot_panel.js');
  const { GpuBootPanelController } = gpuBootPanelApi;

  const clock = createClock(10000);
  const raf = createFakeRaf(clock);
  const timers = createFakeTimers();
  const webgpu = createMockWebGpuEnv();
  const hud = createMockHud();

  function makeMockDomNode(tag = 'div') {
    const attrs = new Map();
    const listeners = new Map();
    const node = {
      tagName: String(tag).toUpperCase(),
      className: '',
      hidden: false,
      textContent: '',
      parentNode: null,
      children: [],
      style: {},
      setAttribute(k, v) {
        attrs.set(k, String(v));
      },
      getAttribute(k) {
        return attrs.has(k) ? attrs.get(k) : null;
      },
      appendChild(child) {
        child.parentNode = node;
        node.children.push(child);
        return child;
      },
      removeChild(child) {
        const idx = node.children.indexOf(child);
        if (idx >= 0) node.children.splice(idx, 1);
        child.parentNode = null;
        return child;
      },
      addEventListener(type, fn) {
        if (!listeners.has(type)) listeners.set(type, new Set());
        listeners.get(type).add(fn);
      },
      removeEventListener(type, fn) {
        listeners.get(type)?.delete(fn);
      },
      querySelector() {
        return null;
      },
    };
    return node;
  }

  const panelRoot = makeMockDomNode('div');
  const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
    clock,
    raf,
    timers,
    gpu: webgpu.gpu,
  });
  documentObj.head = makeMockDomNode('head');
  documentObj.createElement = (tag) => makeMockDomNode(tag);
  documentObj.querySelector = () => null;

  const mockWorker = createFakeEventTarget({
    postMessage() {},
    terminate() {},
  });

  // Auto-discover via globalThis.PlayGta6GpuBootPanel.GpuBootPanelController (no GpuBootPanelCtor passed).
  const prevGlobal = globalThis.PlayGta6GpuBootPanel;
  globalThis.PlayGta6GpuBootPanel = gpuBootPanelApi;
  let app = null;
  try {
    app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, hasWasmBinary: true }),
      canvas: webgpu.canvas,
      hud,
      gpuBootPanelRoot: panelRoot,
      now: clock.now,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
      createWasmWorker() {
        return mockWorker;
      },
    });

    assert.ok(app.gpuBootPanel instanceof GpuBootPanelController);
    assert.equal(panelRoot.children.length, 1);

    const started = await app.start();
    assert.equal(started.ok, true);

    app.bootTracer.markMilestone('FIRST_PRESENT');
    mockWorker.dispatch('message', {
      data: {
        type: 'gpu-telemetry',
        frameStats: { framesPresented: 7, droppedFrames: 0 },
        fallbackStats: { totalLookups: 20, packHits: 18, fallbackHits: 2, builtinHits: 0 },
        missingShaders: [{ hash: '0x1234', stage: 'vertex', hits: 2 }],
      },
    });

    const state = app.gpuBootPanel.getState();
    assert.equal(state.boot.stage, 'FIRST_PRESENT');
    assert.equal(state.boot.readyForSplash, true);
    assert.equal(state.frameStats.framesPresented, 7);
    assert.equal(state.fallbackStats.packHits, 18);
    assert.equal(state.totalMissingShaders, 1);
    assert.equal(state.missingShaders[0].hash, '0x0000000000001234');

    await app.destroy();
    assert.equal(app.gpuBootPanel.getState().destroyed, true);
    assert.equal(panelRoot.children.length, 0);
    assert.equal(mockWorker.listenerCount(), 0);
  } finally {
    if (app && !app.destroyed) {
      await app.destroy();
    }
    if (prevGlobal === undefined) {
      delete globalThis.PlayGta6GpuBootPanel;
    } else {
      globalThis.PlayGta6GpuBootPanel = prevGlobal;
    }
  }
});

test('W58 detectWasm64 supports spec-standard address: i64, legacy index: i64, and rejects non-Wasm64 engines', () => {
  // Simulates WebIDL ToNumber coercion on i32 memory limits, which throws TypeError on BigInt.
  function assertI32Limit(value) {
    return +value;
  }

  // 1. Modern Chrome/V8/Firefox/Node engine accepting only `{ address: 'i64' }` with BigInt limits.
  const modernCalls = [];
  const modernWasmApi = {
    Memory: class MockModernMemory {
      constructor(desc) {
        modernCalls.push(desc);
        if (desc.address === 'i64') {
          if (typeof desc.initial !== 'bigint' || typeof desc.maximum !== 'bigint') {
            throw new TypeError('Expected BigInt limits for i64 memory');
          }
          return;
        }
        assertI32Limit(desc.initial);
      }
    },
  };
  assert.equal(detectWasm64(modernWasmApi), true);
  assert.equal(modernCalls.length, 1);
  assert.equal(modernCalls[0].address, 'i64');
  assert.equal(modernCalls[0].initial, 1n);
  assert.equal(modernCalls[0].maximum, 2n);

  // 2. Older memory64 proposal engine accepting only legacy `{ index: 'i64' }`.
  const legacyCalls = [];
  const legacyWasmApi = {
    Memory: class MockLegacyMemory {
      constructor(desc) {
        legacyCalls.push(desc);
        if (desc.index === 'i64') {
          if (typeof desc.initial !== 'bigint' || typeof desc.maximum !== 'bigint') {
            throw new TypeError('Expected BigInt limits for i64 memory');
          }
          return;
        }
        // Unknown `address` key is ignored -> treated as i32 -> BigInt throws TypeError.
        assertI32Limit(desc.initial);
      }
    },
  };
  assert.equal(detectWasm64(legacyWasmApi), true);
  assert.equal(legacyCalls.length, 2);
  assert.equal(legacyCalls[0].address, 'i64');
  assert.equal(legacyCalls[1].index, 'i64');

  // 3. Non-Wasm64 engine that ignores both keys and throws on BigInt initial/maximum.
  const wasm32Calls = [];
  const wasm32OnlyApi = {
    Memory: class MockWasm32Memory {
      constructor(desc) {
        wasm32Calls.push(desc);
        assertI32Limit(desc.initial);
      }
    },
  };
  assert.equal(detectWasm64(wasm32OnlyApi), false);
  assert.equal(wasm32Calls.length, 2);
  assert.equal(wasm32Calls[0].address, 'i64');
  assert.equal(wasm32Calls[1].index, 'i64');

  // 4. Defensive edge cases: missing API or mock whose type reflection reports i32.
  assert.equal(detectWasm64(null), false);
  assert.equal(detectWasm64(undefined), false);
  assert.equal(detectWasm64({}), false);
  assert.equal(
    detectWasm64({
      Memory: class MockI32ReflectingMemory {
        type() {
          return { minimum: 1, maximum: 2, address: 'i32' };
        }
      },
    }),
    false
  );
});

test('W58 bootstrapPlayGta6App capability probe reports wasm64 for address: i64, legacy index: i64, and non-Wasm64 engines', async () => {
  async function probeWithWebAssembly(mockWebAssembly) {
    const clock = createClock(11000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const webgpu = createMockWebGpuEnv();
    const hud = createMockHud();
    const { windowObj, documentObj, navigatorObj } = createMockBrowserHost({
      clock,
      raf,
      timers,
      gpu: webgpu.gpu,
    });
    windowObj.WebAssembly = mockWebAssembly;

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: createMockFetch({ ok: true, hasWasmBinary: false }),
      canvas: webgpu.canvas,
      hud,
      now: clock.now,
      raf: raf.raf,
      cancelRaf: raf.cancelRaf,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });

    try {
      const env = await app.session.probeEnvironment();
      return env.capabilities.wasm64;
    } finally {
      await app.destroy();
    }
  }

  // Modern engine accepting only `{ address: 'i64' }`.
  const modernResult = await probeWithWebAssembly({
    Memory: class MockModernMemory {
      constructor(desc) {
        if (desc.address !== 'i64') {
          return +desc.initial;
        }
      }
    },
  });
  assert.equal(modernResult, true);

  // Legacy engine accepting only `{ index: 'i64' }`.
  const legacyResult = await probeWithWebAssembly({
    Memory: class MockLegacyMemory {
      constructor(desc) {
        if (desc.index !== 'i64') {
          return +desc.initial;
        }
      }
    },
  });
  assert.equal(legacyResult, true);

  // Non-Wasm64 engine rejecting BigInt/i64 memory.
  const wasm32Result = await probeWithWebAssembly({
    Memory: class MockWasm32Memory {
      constructor(desc) {
        return +desc.initial;
      }
    },
  });
  assert.equal(wasm32Result, false);

  // Explicitly absent WebAssembly API on windowObj.
  const absentResult = await probeWithWebAssembly(null);
  assert.equal(absentResult, false);
});
