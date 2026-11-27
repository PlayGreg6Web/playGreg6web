import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const swapchainMod = require('../web/workers/wgpu_swapchain.js');
const {
  BLIT_FULLSCREEN_WGSL,
  BLIT_SRGB_ENCODE_WGSL,
  BLIT_TONEMAP_WGSL,
  BUFFER_USAGE_COPY_DST,
  BUFFER_USAGE_UNIFORM,
  COMPATIBLE_VIEW_FORMATS,
  DEFAULT_DEPTH_FORMAT,
  DEFAULT_DROPPED_FRAME_THRESHOLD_MS,
  DEFAULT_EXPOSURE,
  DEFAULT_TONE_MAP_MODE,
  HDR_STREAM_FORMATS,
  TARGET_30FPS_BUDGET_MS,
  DEFAULT_MAX_ERROR_HISTORY,
  DEFAULT_USAGE,
  DROPPED_FRAME_CADENCE_MULTIPLIER,
  INTERMEDIATE_TEXTURE_USAGE,
  MAX_VALIDATION_ERROR_MESSAGE_LENGTH,
  SUPPORTED_DEPTH_FORMATS,
  TEXTURE_USAGE_COPY_SRC,
  TEXTURE_USAGE_RENDER_ATTACHMENT,
  TEXTURE_USAGE_TEXTURE_BINDING,
  TONE_MAP_MODES,
  VALID_CANVAS_FORMATS,
  WgpuSwapchain,
  getCompatibleViewFormats,
  isHdrStreamFormat,
  isSrgbFormat,
  isValidCanvasFormat,
  normalizeDisplayBounds,
  clampToDisplayBounds,
  normalizeExposure,
  normalizeToneMapMode,
} = swapchainMod;

function createMockWebGpuEnv({ limits, withUnconfigure = true } = {}) {
  const events = [];
  let nextTextureId = 1;
  let nextSwapViewId = 1;

  function makeDevice(label = 'device-0', deviceLimits = limits) {
    let resolveLost;
    const lost = new Promise((resolve) => {
      resolveLost = resolve;
    });

    const createdTextures = [];
    const device = {
      label,
      limits: deviceLimits,
      lost,
      createdTextures,
      triggerLost(info = { reason: 'unknown', message: 'simulated GPU reset' }) {
        resolveLost(info);
      },
      createTexture(desc) {
        const id = nextTextureId++;
        const view = { kind: 'GPUTextureView', id: `depth-view-${id}`, textureId: id };
        const texture = {
          kind: 'GPUTexture',
          id,
          deviceLabel: label,
          desc: {
            ...desc,
            size: { ...desc.size },
          },
          view,
          destroyed: false,
          destroyCalls: 0,
          createView() {
            events.push({ op: 'depthTexture.createView', id, deviceLabel: label });
            return view;
          },
          destroy() {
            this.destroyed = true;
            this.destroyCalls += 1;
            events.push({ op: 'depthTexture.destroy', id, deviceLabel: label });
          },
        };
        createdTextures.push(texture);
        events.push({ op: 'device.createTexture', id, deviceLabel: label, desc: texture.desc });
        return texture;
      },
    };
    return device;
  }

  const context = {
    configurations: [],
    unconfigureCalls: 0,
    configure(cfg) {
      this.configurations.push({ ...cfg });
      events.push({ op: 'context.configure', cfg: { ...cfg } });
    },
    getCurrentTexture() {
      const view = { kind: 'GPUTextureView', id: `color-view-${nextSwapViewId++}` };
      events.push({ op: 'context.getCurrentTexture', viewId: view.id });
      return {
        createView() {
          return view;
        },
      };
    },
  };

  if (withUnconfigure) {
    context.unconfigure = function unconfigure() {
      this.unconfigureCalls += 1;
      events.push({ op: 'context.unconfigure' });
    };
  }

  const canvas = {
    width: 0,
    height: 0,
    getContext(kind) {
      return kind === 'webgpu' ? context : null;
    },
  };

  const device = makeDevice('device-0', limits);

  return { canvas, context, device, events, makeDevice };
}

test('UMD export attaches PlayGta6WgpuSwapchain on globalThis and exposes depth constants', () => {
  assert.equal(globalThis.PlayGta6WgpuSwapchain.WgpuSwapchain, WgpuSwapchain);
  assert.equal(DEFAULT_DEPTH_FORMAT, 'depth24plus');
  assert.equal(globalThis.PlayGta6WgpuSwapchain.DEFAULT_DEPTH_FORMAT, 'depth24plus');
  assert.ok(SUPPORTED_DEPTH_FORMATS.includes('depth24plus'));
  assert.ok(SUPPORTED_DEPTH_FORMATS.includes('depth32float'));
  assert.ok(SUPPORTED_DEPTH_FORMATS.includes('depth24plus-stencil8'));
  assert.ok(Object.isFrozen(SUPPORTED_DEPTH_FORMATS));
});

test('initial configure applies DPR scaling, clamps to [1, maxDimension], and lazily creates depth texture', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    depthFormat: 'depth32float',
    alphaMode: 'opaque',
    maxDimension: 4096,
  });

  // Constructor must not allocate a depth texture before configure().
  assert.equal(device.createdTextures.length, 0);
  assert.equal(swapchain.isConfigured, false);

  // DPR scaling within limits: 1280x720 @ 2x -> 2560x1440.
  const cfg = swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 2 });
  assert.deepEqual(cfg, {
    width: 2560,
    height: 1440,
    format: 'bgra8unorm',
    depthFormat: 'depth32float',
    devicePixelRatio: 2,
  });
  assert.equal(canvas.width, 2560);
  assert.equal(canvas.height, 1440);
  assert.equal(context.configurations.length, 1);
  assert.deepEqual(context.configurations[0], {
    device,
    format: 'bgra8unorm',
    alphaMode: 'opaque',
    viewFormats: ['bgra8unorm-srgb'],
  });
  assert.equal(device.createdTextures.length, 1);
  assert.deepEqual(device.createdTextures[0].desc, {
    label: 'playgta6-swapchain-depth',
    size: { width: 2560, height: 1440, depthOrArrayLayers: 1 },
    format: 'depth32float',
    usage: TEXTURE_USAGE_RENDER_ATTACHMENT,
  });
  assert.equal(swapchain.getTelemetry().resizeCount, 0);

  // Clamping: oversized width clamped to maxDimension (4096), zero/negative height clamped to 1.
  swapchain.configure({ width: 5000, height: -20, devicePixelRatio: 2 });
  assert.equal(canvas.width, 4096);
  assert.equal(canvas.height, 1);
  assert.equal(device.createdTextures[0].destroyed, true);
  assert.equal(device.createdTextures.length, 2);
  assert.deepEqual(device.createdTextures[1].desc.size, {
    width: 4096,
    height: 1,
    depthOrArrayLayers: 1,
  });
  assert.equal(swapchain.getTelemetry().resizeCount, 1);
});

test('resize no-ops when pixel dimensions are unchanged and destroys old depth texture before creating new one on change', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
  });

  swapchain.configure({ width: 640, height: 360, devicePixelRatio: 2 });
  assert.equal(canvas.width, 1280);
  assert.equal(canvas.height, 720);
  assert.equal(device.createdTextures.length, 1);

  const firstDepthTex = device.createdTextures[0];

  // Exact same CSS size + DPR -> no-op.
  assert.equal(swapchain.resize(640, 360, 2), false);
  // Different CSS size + DPR resolving to the exact same pixel dimensions -> still a no-op.
  assert.equal(swapchain.resize(1280, 720, 1), false);
  assert.equal(context.configurations.length, 1);
  assert.equal(device.createdTextures.length, 1);
  assert.equal(firstDepthTex.destroyed, false);
  assert.equal(swapchain.getTelemetry().resizeCount, 0);

  // Clear recorded events to assert ordering on actual resize.
  events.length = 0;

  assert.equal(swapchain.resize(960, 540, 2), true);
  assert.equal(canvas.width, 1920);
  assert.equal(canvas.height, 1080);
  assert.equal(context.configurations.length, 2);
  assert.equal(device.createdTextures.length, 2);
  assert.equal(firstDepthTex.destroyed, true);
  assert.equal(firstDepthTex.destroyCalls, 1);
  assert.equal(swapchain.getTelemetry().resizeCount, 1);

  // Verify previous depth texture was destroyed strictly BEFORE creating the replacement.
  const destroyIdx = events.findIndex((e) => e.op === 'depthTexture.destroy');
  const createIdx = events.findIndex((e) => e.op === 'device.createTexture');
  assert.ok(destroyIdx !== -1, 'expected old depth texture destroy event');
  assert.ok(createIdx !== -1, 'expected new depth texture create event');
  assert.ok(
    destroyIdx < createIdx,
    `expected depthTexture.destroy (${destroyIdx}) before device.createTexture (${createIdx})`
  );
});

test('acquireRenderPassDescriptor builds color-only, color+depth, and color+depth-stencil descriptors', () => {
  // 1. Color-only mode (enableDepth: false)
  {
    const { canvas, context, device } = createMockWebGpuEnv();
    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      enableDepth: false,
    });
    swapchain.configure({ width: 800, height: 600 });

    assert.equal(device.createdTextures.length, 0);
    assert.equal(swapchain.depthFormat, null);
    assert.equal(swapchain.getTelemetry().depthFormat, null);

    const desc = swapchain.acquireRenderPassDescriptor({
      clearColor: { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
      loadOp: 'clear',
      storeOp: 'store',
    });

    assert.equal(desc.colorAttachments.length, 1);
    assert.equal(desc.colorAttachments[0].view.kind, 'GPUTextureView');
    assert.equal(desc.colorAttachments[0].loadOp, 'clear');
    assert.equal(desc.colorAttachments[0].storeOp, 'store');
    assert.deepEqual(desc.colorAttachments[0].clearValue, { r: 0.1, g: 0.2, b: 0.3, a: 1.0 });
    assert.equal('depthStencilAttachment' in desc, false);
  }

  // 2. Color + depth24plus (default)
  {
    const { canvas, context, device } = createMockWebGpuEnv();
    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      enableDepth: true,
    });
    swapchain.configure({ width: 800, height: 600 });

    const desc = swapchain.acquireRenderPassDescriptor({
      clearColor: { r: 0.25, g: 0.5, b: 0.75, a: 1.0 },
      depthClearValue: 0.0,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
    });

    assert.equal(desc.colorAttachments.length, 1);
    assert.deepEqual(desc.colorAttachments[0].clearValue, { r: 0.25, g: 0.5, b: 0.75, a: 1.0 });
    assert.deepEqual(desc.depthStencilAttachment, {
      view: device.createdTextures[0].view,
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
      depthClearValue: 0.0,
    });
  }

  // 3. Color + depth24plus-stencil8 includes stencil load/store/clear fields
  {
    const { canvas, context, device } = createMockWebGpuEnv();
    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      depthFormat: 'depth24plus-stencil8',
    });
    swapchain.configure({ width: 640, height: 480 });

    const desc = swapchain.acquireRenderPassDescriptor({
      loadOp: 'load',
      depthLoadOp: 'load',
      depthStoreOp: 'store',
      stencilClearValue: 7,
    });

    assert.equal('clearValue' in desc.colorAttachments[0], false);
    assert.deepEqual(desc.depthStencilAttachment, {
      view: device.createdTextures[0].view,
      depthLoadOp: 'load',
      depthStoreOp: 'store',
      stencilLoadOp: 'clear',
      stencilStoreOp: 'store',
      stencilClearValue: 7,
    });
  }
});

test('rolling FPS and frame-time telemetry uses injectable clock and evicts old samples beyond window', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  let clockMs = 1000;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    now: () => clockMs,
  });
  swapchain.configure({ width: 1280, height: 720 });

  // Before any frames.
  let t = swapchain.getTelemetry();
  assert.deepEqual(t, {
    fps: 0,
    avgFrameTimeMs: 0,
    maxFrameTimeMs: 0,
    totalFrames: 0,
    presentCount: 0,
    droppedFrameCount: 0,
    lastFrameMs: 0,
    avgFrameMs: 0,
    maxFrameMs: 0,
    resizeCount: 0,
    reconfigureCount: 1,
    skippedRedundantReconfigures: 0,
    lastPresentIntervalMs: 0,
    avgPresentIntervalMs: 0,
    width: 1280,
    height: 720,
    format: 'bgra8unorm',
    depthFormat: 'depth24plus',
    isDeviceLost: false,
    hasVisibleContent: false,
    firstVisiblePresentIndex: null,
    firstVisiblePresentMs: null,
    consecutiveBlackPresents: 0,
    blackPresentsTotal: 0,
    visiblePresentsTotal: 0,
  });

  // First presented frame establishes baseline timestamp (no delta yet).
  swapchain.recordFramePresented();
  t = swapchain.getTelemetry();
  assert.equal(t.totalFrames, 1);
  assert.equal(t.avgFrameTimeMs, 0);
  assert.equal(t.maxFrameTimeMs, 0);
  assert.equal(t.fps, 0);

  // Frame 2: +20ms, Frame 3: +50ms spike, Frame 4: +20ms -> avg = 30ms, max = 50ms, fps = 1000/30.
  clockMs += 20;
  swapchain.recordFramePresented();
  clockMs += 50;
  swapchain.recordFramePresented();
  clockMs += 20;
  swapchain.recordFramePresented();

  t = swapchain.getTelemetry();
  assert.equal(t.totalFrames, 4);
  assert.equal(t.avgFrameTimeMs, 30);
  assert.equal(t.maxFrameTimeMs, 50);
  assert.ok(Math.abs(t.fps - 1000 / 30) < 1e-9);

  // Present 60 steady 16ms frames so the earlier 50ms spike rolls completely out of the 60-frame window.
  for (let i = 0; i < 60; i += 1) {
    clockMs += 16;
    swapchain.recordFramePresented();
  }

  t = swapchain.getTelemetry();
  assert.equal(t.totalFrames, 64);
  assert.equal(t.avgFrameTimeMs, 16);
  assert.equal(t.maxFrameTimeMs, 16);
  assert.equal(t.fps, 62.5);
});

test('device.lost handler destroys stale depth texture and recover(newDevice) restores swapchain cleanly', async () => {
  const { canvas, context, device, makeDevice } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    depthFormat: 'depth24plus',
  });
  swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 1 });

  const oldDepthTex = device.createdTextures[0];
  const lostEvents = [];
  swapchain.attachDeviceLostHandler((info, sc) => {
    lostEvents.push({ info, sameInstance: sc === swapchain });
  });

  // Trigger device loss and await promise microtask flush.
  device.triggerLost({ reason: 'unknown', message: 'GPU driver crash' });
  await device.lost;
  await Promise.resolve();

  assert.equal(swapchain.isDeviceLost, true);
  assert.equal(swapchain.getTelemetry().isDeviceLost, true);
  assert.equal(oldDepthTex.destroyed, true);
  assert.equal(swapchain.depthTexture, null);
  assert.equal(lostEvents.length, 1);
  assert.deepEqual(lostEvents[0], {
    info: { reason: 'unknown', message: 'GPU driver crash' },
    sameInstance: true,
  });

  // Operations while lost should reject cleanly.
  assert.throws(() => swapchain.acquireRenderPassDescriptor(), /lost/i);
  assert.throws(() => swapchain.resize(800, 600), /lost/i);

  // Recover with a new GPUDevice and updated canvas format.
  const newDevice = makeDevice('device-1');
  const recovered = swapchain.recover(newDevice, 'rgba8unorm');

  assert.deepEqual(recovered, {
    width: 1280,
    height: 720,
    format: 'rgba8unorm',
    depthFormat: 'depth24plus',
    isDeviceLost: false,
  });
  assert.equal(swapchain.device, newDevice);
  assert.equal(swapchain.canvas, canvas);
  assert.equal(swapchain.context, context);
  assert.equal(swapchain.isDeviceLost, false);
  assert.equal(swapchain.getTelemetry().resizeCount, 0);

  // Context was reconfigured with the new device and format.
  const lastConfig = context.configurations[context.configurations.length - 1];
  assert.deepEqual(lastConfig, {
    device: newDevice,
    format: 'rgba8unorm',
    alphaMode: 'opaque',
    viewFormats: ['rgba8unorm-srgb'],
  });

  // Fresh depth texture created on the new device and usable in render pass descriptors.
  assert.equal(newDevice.createdTextures.length, 1);
  const newDepthTex = newDevice.createdTextures[0];
  assert.deepEqual(newDepthTex.desc.size, { width: 1280, height: 720, depthOrArrayLayers: 1 });

  const passDesc = swapchain.acquireRenderPassDescriptor();
  assert.equal(passDesc.depthStencilAttachment.view, newDepthTex.view);

  // Registered onLost callback remains attached to the recovered device too.
  newDevice.triggerLost({ reason: 'destroyed', message: 'second loss' });
  await newDevice.lost;
  await Promise.resolve();
  assert.equal(lostEvents.length, 2);
  assert.equal(lostEvents[1].info.message, 'second loss');
  assert.equal(newDepthTex.destroyed, true);
});

test('destroy() destroys depth texture, unconfigures context, clears refs, and is idempotent', () => {
  const { canvas, context, device } = createMockWebGpuEnv({ withUnconfigure: true });
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
  });
  swapchain.configure({ width: 640, height: 480 });

  const depthTex = device.createdTextures[0];
  assert.equal(depthTex.destroyed, false);
  assert.equal(context.unconfigureCalls, 0);

  swapchain.destroy();
  assert.equal(swapchain.isDestroyed, true);
  assert.equal(swapchain.isConfigured, false);
  assert.equal(depthTex.destroyed, true);
  assert.equal(depthTex.destroyCalls, 1);
  assert.equal(context.unconfigureCalls, 1);
  assert.equal(swapchain.depthTexture, null);
  assert.equal(swapchain.device, null);
  assert.equal(swapchain.context, null);
  assert.equal(swapchain.canvas, null);

  // Second destroy() must be a safe no-op (idempotent).
  swapchain.destroy();
  assert.equal(depthTex.destroyCalls, 1);
  assert.equal(context.unconfigureCalls, 1);

  // Post-destroy operations throw cleanly while getTelemetry remains safe.
  assert.throws(() => swapchain.configure({ width: 320, height: 240 }), /destroyed/i);
  assert.throws(() => swapchain.resize(320, 240), /destroyed/i);
  assert.throws(() => swapchain.acquireRenderPassDescriptor(), /destroyed/i);
  assert.doesNotThrow(() => swapchain.getTelemetry());

  // Also works safely when context does not implement unconfigure().
  const noUnconfigEnv = createMockWebGpuEnv({ withUnconfigure: false });
  const swapchain2 = new WgpuSwapchain({
    canvas: noUnconfigEnv.canvas,
    device: noUnconfigEnv.device,
    context: noUnconfigEnv.context,
    format: 'bgra8unorm',
  });
  swapchain2.configure({ width: 320, height: 240 });
  assert.doesNotThrow(() => swapchain2.destroy());
  assert.doesNotThrow(() => swapchain2.destroy());
  assert.equal(noUnconfigEnv.device.createdTextures[0].destroyed, true);
});

test('constructor validates invalid options cleanly', () => {
  const { canvas, context, device } = createMockWebGpuEnv();

  assert.throws(() => new WgpuSwapchain({ device, context }), TypeError);
  assert.throws(() => new WgpuSwapchain({ canvas, context }), TypeError);
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, depthFormat: 'invalid-depth' }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, alphaMode: 'bogus' }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, maxDimension: 0 }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, onValidationError: 'not-a-fn' }),
    TypeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, maxErrorHistory: 0 }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, maxErrorHistory: -5 }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, maxErrorHistory: 2.5 }),
    RangeError
  );
});

function attachMockEventTarget(device, { withRemove = true } = {}) {
  const listeners = new Map();
  const addCalls = [];
  const removeCalls = [];

  device.addEventListener = function addEventListener(type, fn) {
    addCalls.push({ type, fn });
    let set = listeners.get(type);
    if (!set) {
      set = new Set();
      listeners.set(type, set);
    }
    set.add(fn);
  };

  if (withRemove) {
    device.removeEventListener = function removeEventListener(type, fn) {
      removeCalls.push({ type, fn });
      const set = listeners.get(type);
      if (set) {
        set.delete(fn);
      }
    };
  }

  device.dispatchUncapturedError = function dispatchUncapturedError(error, extra = {}) {
    const event = { type: 'uncapturederror', error, ...extra };
    const set = listeners.get('uncapturederror');
    if (set) {
      for (const fn of Array.from(set)) {
        fn(event);
      }
    }
    return event;
  };

  device.listenerCount = function listenerCount(type = 'uncapturederror') {
    const set = listeners.get(type);
    return set ? set.size : 0;
  };

  device.addCalls = addCalls;
  device.removeCalls = removeCalls;
  return device;
}

test('G35 uncapturederror listener attaches, captures normalized errors, updates getStats(), and invokes onValidationError safely', () => {
  assert.equal(DEFAULT_MAX_ERROR_HISTORY, 32);
  assert.equal(MAX_VALIDATION_ERROR_MESSAGE_LENGTH, 512);

  const { canvas, context, device } = createMockWebGpuEnv();
  attachMockEventTarget(device);

  let clockMs = 5000;
  const callbackRecords = [];

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    now: () => clockMs,
    onValidationError(record) {
      callbackRecords.push({ ...record });
      // Mutate received object to prove internal state is isolated.
      record.message = 'MUTATED_BY_CALLBACK';
      if (record.seq === 2) {
        throw new Error('user callback failure should be isolated');
      }
    },
  });

  assert.equal(device.listenerCount('uncapturederror'), 1);
  assert.equal(device.addCalls.length, 1);
  assert.equal(device.addCalls[0].type, 'uncapturederror');

  let stats = swapchain.getStats();
  assert.equal(stats.validationErrorCount, 0);
  assert.equal(stats.validationErrorHistorySize, 0);
  assert.equal(stats.maxErrorHistory, 32);
  assert.equal(stats.isDestroyed, false);
  assert.deepEqual(swapchain.getValidationErrors(), []);

  class GPUValidationErrorMock {
    constructor(message) {
      this.message = message;
    }
  }
  Object.defineProperty(GPUValidationErrorMock, 'name', { value: 'GPUValidationError' });

  class GPUOutOfMemoryErrorMock {
    constructor(message) {
      this.message = message;
    }
  }
  Object.defineProperty(GPUOutOfMemoryErrorMock, 'name', { value: 'GPUOutOfMemoryError' });

  // 1. Validation error
  clockMs = 5010;
  device.dispatchUncapturedError(
    new GPUValidationErrorMock('BindGroupLayout mismatch on pipeline 0x123456789abcdef0')
  );

  // 2. Out-of-memory error (callback throws, must not disrupt capture)
  clockMs = 5025;
  assert.doesNotThrow(() => {
    device.dispatchUncapturedError(
      new GPUOutOfMemoryErrorMock('Allocation of 256MB texture failed')
    );
  });

  // 3. Internal error via explicit type
  clockMs = 5040;
  device.dispatchUncapturedError({ type: 'internal', message: 'Dawn internal backend error' });

  // 4. Unknown error object
  clockMs = 5055;
  device.dispatchUncapturedError({ message: 'Unclassified GPU failure' });

  stats = swapchain.getStats();
  assert.equal(stats.validationErrorCount, 4);
  assert.equal(stats.validationErrorHistorySize, 4);

  const history = swapchain.getValidationErrors();
  assert.deepEqual(history, [
    {
      seq: 1,
      type: 'validation',
      message: 'BindGroupLayout mismatch on pipeline 0x123456789abcdef0',
      timestampMs: 5010,
    },
    {
      seq: 2,
      type: 'out-of-memory',
      message: 'Allocation of 256MB texture failed',
      timestampMs: 5025,
    },
    {
      seq: 3,
      type: 'internal',
      message: 'Dawn internal backend error',
      timestampMs: 5040,
    },
    {
      seq: 4,
      type: 'unknown',
      message: 'Unclassified GPU failure',
      timestampMs: 5055,
    },
  ]);

  assert.equal(callbackRecords.length, 4);
  assert.equal(callbackRecords[0].seq, 1);
  assert.equal(callbackRecords[1].type, 'out-of-memory');

  // Mutating returned shallow copies must not corrupt internal history.
  history[0].message = 'CORRUPTED';
  history.push({ seq: 999, type: 'validation', message: 'fake', timestampMs: 0 });
  const freshHistory = swapchain.getValidationErrors();
  assert.equal(freshHistory.length, 4);
  assert.equal(
    freshHistory[0].message,
    'BindGroupLayout mismatch on pipeline 0x123456789abcdef0'
  );
});

test('G35 ring evicts oldest entries at maxErrorHistory and clearValidationErrors() resets ring while keeping monotonic seq/count', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  attachMockEventTarget(device);

  let clockMs = 100;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    now: () => clockMs,
    maxErrorHistory: 3,
  });

  for (let i = 1; i <= 5; i += 1) {
    clockMs = 100 + i * 10;
    device.dispatchUncapturedError({
      name: 'GPUValidationError',
      message: `validation error #${i}`,
    });
  }

  assert.equal(swapchain.getStats().validationErrorCount, 5);
  assert.equal(swapchain.getStats().validationErrorHistorySize, 3);
  assert.equal(swapchain.getStats().maxErrorHistory, 3);

  const retained = swapchain.getValidationErrors();
  assert.deepEqual(
    retained.map((r) => r.seq),
    [3, 4, 5]
  );
  assert.deepEqual(
    retained.map((r) => r.message),
    ['validation error #3', 'validation error #4', 'validation error #5']
  );

  // Clear history ring; cumulative count stays intact and next seq continues monotonically.
  swapchain.clearValidationErrors();
  assert.deepEqual(swapchain.getValidationErrors(), []);
  assert.equal(swapchain.getStats().validationErrorHistorySize, 0);
  assert.equal(swapchain.getStats().validationErrorCount, 5);

  clockMs = 200;
  device.dispatchUncapturedError({
    name: 'GPUValidationError',
    message: 'post-clear error',
  });
  const afterClear = swapchain.getValidationErrors();
  assert.equal(afterClear.length, 1);
  assert.deepEqual(afterClear[0], {
    seq: 6,
    type: 'validation',
    message: 'post-clear error',
    timestampMs: 200,
  });
  assert.equal(swapchain.getStats().validationErrorCount, 6);
});

test('G35 sanitizes host paths, long buffers, control characters, and caps message length to 512 chars', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  attachMockEventTarget(device);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    now: () => 42,
  });

  // 1. Host paths (Windows drive, quoted path with spaces, UNC, file://, Unix paths)
  device.dispatchUncapturedError({
    name: 'GPUValidationError',
    message:
      'Shader failed at C:\\Users\\alice\\local-data\\gta6\\shader.wgsl and "D:\\My Games\\GTA6\\x.rpf" and \\\\host01\\share\\dump.bin and file:///Users/alice/secret/a.wgsl and /home/alice/dev/pipeline.wgsl (format rgba8unorm/bgra8unorm)',
  });

  // 2. Long buffers (hex dump, byte array, base64 data URI, raw base64)
  const hexBlob = 'ab'.repeat(40);
  const spacedHex = Array.from({ length: 24 }, (_, i) => i.toString(16).padStart(2, '0')).join(' ');
  const byteArray = Array.from({ length: 24 }, (_, i) => String(i)).join(', ');
  const b64Blob = 'QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFB';
  device.dispatchUncapturedError({
    name: 'GPUValidationError',
    message: `Buffer mismatch:\nhex=${hexBlob}\tbytes=[${byteArray}] dump=${spacedHex} uri=data:application/octet-stream;base64,${b64Blob} raw=${b64Blob}\x00\x07`,
  });

  // 3. Overly long message capped strictly to 512 characters
  const longText = `Validation failure: ${'texture_binding_mismatch '.repeat(40)}`;
  assert.ok(longText.length > 512);
  device.dispatchUncapturedError({
    name: 'GPUValidationError',
    message: longText,
  });

  const errors = swapchain.getValidationErrors();
  assert.equal(errors.length, 3);

  // Verify host paths stripped while preserving WebGPU format slash token.
  const pathMsg = errors[0].message;
  assert.doesNotMatch(pathMsg, /alice/i);
  assert.doesNotMatch(pathMsg, /C:\\/);
  assert.doesNotMatch(pathMsg, /D:\\/);
  assert.doesNotMatch(pathMsg, /\\\\host01/);
  assert.doesNotMatch(pathMsg, /file:\/\//);
  assert.doesNotMatch(pathMsg, /\/home\//);
  assert.match(pathMsg, /<path>/);
  assert.match(pathMsg, /rgba8unorm\/bgra8unorm/);

  // Verify long buffers & control chars stripped.
  const bufMsg = errors[1].message;
  assert.doesNotMatch(bufMsg, new RegExp(hexBlob));
  assert.doesNotMatch(bufMsg, new RegExp(b64Blob));
  assert.doesNotMatch(bufMsg, /[\x00-\x1F\x7F]/);
  assert.match(bufMsg, /<buffer>/);

  // Verify 512-char cap.
  const cappedMsg = errors[2].message;
  assert.equal(cappedMsg.length, MAX_VALIDATION_ERROR_MESSAGE_LENGTH);
  assert.ok(cappedMsg.endsWith('...'));
});

test('G35 removes uncapturederror listener cleanly on recover(newDevice) and destroy() with zero leaks', () => {
  const { canvas, context, device, makeDevice } = createMockWebGpuEnv();
  attachMockEventTarget(device);

  const seen = [];
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    onValidationError: (rec) => seen.push(rec),
  });
  swapchain.configure({ width: 640, height: 360 });

  assert.equal(device.listenerCount('uncapturederror'), 1);
  const firstRegisteredFn = device.addCalls[0].fn;

  // Re-bind to a new device via recover(): old listener must be removed, new one attached.
  const newDevice = attachMockEventTarget(makeDevice('device-1'));
  swapchain.recover(newDevice);

  assert.equal(device.listenerCount('uncapturederror'), 0);
  assert.equal(device.removeCalls.length, 1);
  assert.equal(device.removeCalls[0].type, 'uncapturederror');
  assert.equal(device.removeCalls[0].fn, firstRegisteredFn);

  assert.equal(newDevice.listenerCount('uncapturederror'), 1);
  const secondRegisteredFn = newDevice.addCalls[0].fn;

  // Dispatching on old device does nothing; dispatching on new device is captured.
  device.dispatchUncapturedError({ name: 'GPUValidationError', message: 'stale device error' });
  assert.equal(swapchain.getStats().validationErrorCount, 0);

  newDevice.dispatchUncapturedError({ name: 'GPUValidationError', message: 'recovered device error' });
  assert.equal(swapchain.getStats().validationErrorCount, 1);
  assert.equal(seen.length, 1);
  assert.equal(seen[0].message, 'recovered device error');

  // Destroy removes listener from active device and is idempotent.
  swapchain.destroy();
  assert.equal(newDevice.listenerCount('uncapturederror'), 0);
  assert.equal(newDevice.removeCalls.length, 1);
  assert.equal(newDevice.removeCalls[0].type, 'uncapturederror');
  assert.equal(newDevice.removeCalls[0].fn, secondRegisteredFn);
  assert.equal(swapchain.getStats().isDestroyed, true);

  // Calling destroy again does not double-remove.
  swapchain.destroy();
  assert.equal(newDevice.removeCalls.length, 1);

  // Events fired after destroy are ignored; previously captured history remains inspectable.
  newDevice.dispatchUncapturedError({ name: 'GPUValidationError', message: 'post-destroy error' });
  assert.equal(swapchain.getStats().validationErrorCount, 1);
  assert.equal(swapchain.getValidationErrors().length, 1);

  // Also test device without removeEventListener: stale handler still guards on destroyed/device mismatch.
  const env2 = createMockWebGpuEnv();
  attachMockEventTarget(env2.device, { withRemove: false });
  const sc2 = new WgpuSwapchain({
    canvas: env2.canvas,
    device: env2.device,
    context: env2.context,
    format: 'bgra8unorm',
  });
  sc2.destroy();
  env2.device.dispatchUncapturedError({ name: 'GPUValidationError', message: 'ignored after destroy' });
  assert.equal(sc2.getStats().validationErrorCount, 0);
});

function attachMockBlitCapabilities(device, events) {
  let nextShaderId = 1;
  let nextSamplerId = 1;
  let nextPipelineId = 1;
  let nextBindGroupId = 1;

  device.createdShaderModules = [];
  device.createdSamplers = [];
  device.createdRenderPipelines = [];
  device.createdBindGroups = [];

  device.createShaderModule = function createShaderModule(desc) {
    const mod = { kind: 'GPUShaderModule', id: nextShaderId++, deviceLabel: device.label, desc };
    device.createdShaderModules.push(mod);
    events.push({ op: 'device.createShaderModule', id: mod.id, deviceLabel: device.label });
    return mod;
  };

  device.createSampler = function createSampler(desc) {
    const sampler = { kind: 'GPUSampler', id: nextSamplerId++, deviceLabel: device.label, desc };
    device.createdSamplers.push(sampler);
    events.push({ op: 'device.createSampler', id: sampler.id, deviceLabel: device.label });
    return sampler;
  };

  device.createRenderPipeline = function createRenderPipeline(desc) {
    const id = nextPipelineId++;
    const pipeline = {
      kind: 'GPURenderPipeline',
      id,
      deviceLabel: device.label,
      desc,
      getBindGroupLayout(index) {
        return { kind: 'GPUBindGroupLayout', pipelineId: id, index };
      },
    };
    device.createdRenderPipelines.push(pipeline);
    events.push({
      op: 'device.createRenderPipeline',
      id,
      deviceLabel: device.label,
      targetFormat: desc.fragment && desc.fragment.targets && desc.fragment.targets[0]?.format,
    });
    return pipeline;
  };

  device.createBindGroup = function createBindGroup(desc) {
    const bg = { kind: 'GPUBindGroup', id: nextBindGroupId++, deviceLabel: device.label, desc };
    device.createdBindGroups.push(bg);
    events.push({ op: 'device.createBindGroup', id: bg.id, deviceLabel: device.label });
    return bg;
  };

  return device;
}

function createMockCommandEncoder() {
  const passes = [];
  return {
    passes,
    beginRenderPass(desc) {
      const record = {
        desc,
        pipelines: [],
        bindGroups: [],
        draws: [],
        ended: false,
      };
      passes.push(record);
      return {
        setPipeline(pipeline) {
          record.pipelines.push(pipeline);
        },
        setBindGroup(index, bindGroup) {
          record.bindGroups.push({ index, bindGroup });
        },
        draw(vertexCount, instanceCount, firstVertex, firstInstance) {
          record.draws.push({ vertexCount, instanceCount, firstVertex, firstInstance });
        },
        end() {
          record.ended = true;
        },
      };
    },
  };
}

test('W112 direct WebGPU canvas formats (rgba8unorm, bgra8unorm, rgba16float) allocate zero intermediate textures and blitToCanvasIfNeeded is a no-op', () => {
  assert.deepEqual(VALID_CANVAS_FORMATS, ['bgra8unorm', 'rgba8unorm', 'rgba16float']);
  assert.equal(isValidCanvasFormat('rgba8unorm'), true);
  assert.equal(isValidCanvasFormat('bgra8unorm'), true);
  assert.equal(isValidCanvasFormat('rgba16float'), true);
  assert.equal(isValidCanvasFormat('rgb10a2unorm'), false);
  assert.equal(isValidCanvasFormat('rg11b10ufloat'), false);
  assert.equal(isValidCanvasFormat('rgba8unorm-srgb'), false);
  assert.equal(
    INTERMEDIATE_TEXTURE_USAGE,
    TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_TEXTURE_BINDING | TEXTURE_USAGE_COPY_SRC
  );
  assert.ok(typeof BLIT_FULLSCREEN_WGSL === 'string' && BLIT_FULLSCREEN_WGSL.includes('vs_main'));
  assert.ok(BLIT_FULLSCREEN_WGSL.includes('fs_main'));

  for (const directFormat of ['rgba8unorm', 'bgra8unorm', 'rgba16float']) {
    const { canvas, context, device, events } = createMockWebGpuEnv();
    attachMockBlitCapabilities(device, events);

    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      enableDepth: false,
    });
    swapchain.configure({ width: 1280, height: 720 });
    swapchain.configureForStreamFormat(directFormat);

    assert.equal(swapchain.usesBlitFallback, false);
    assert.equal(swapchain.streamFormat, directFormat);
    assert.equal(swapchain.canvasFormat, directFormat);
    assert.equal(swapchain.format, directFormat);
    assert.equal(swapchain.intermediateTexture, null);
    assert.equal(swapchain.intermediateTextureView, null);
    assert.equal(device.createdTextures.length, 0);

    const lastCfg = context.configurations[context.configurations.length - 1];
    assert.equal(lastCfg.format, directFormat);

    // Direct render-target view comes straight from context.getCurrentTexture().
    const rtView = swapchain.getRenderTargetView();
    assert.match(rtView.id, /^color-view-/);

    // blitToCanvasIfNeeded & present must be strict no-ops with zero GPU pipeline allocations.
    const encoder = createMockCommandEncoder();
    assert.equal(swapchain.blitToCanvasIfNeeded(encoder, rtView), false);
    assert.equal(swapchain.present(encoder, rtView), false);
    assert.equal(encoder.passes.length, 0);
    assert.equal(device.createdShaderModules.length, 0);
    assert.equal(device.createdSamplers.length, 0);
    assert.equal(device.createdRenderPipelines.length, 0);
    assert.equal(device.createdBindGroups.length, 0);
    assert.equal(swapchain.blitCount, 0);

    const stats = swapchain.getStats();
    assert.equal(stats.usesBlitFallback, false);
    assert.equal(stats.streamFormat, directFormat);
    assert.equal(stats.canvasFormat, directFormat);
    assert.equal(stats.blitCount, 0);
  }
});

test('W112 non-canvas stream format (rgb10a2unorm) allocates intermediate texture, keeps canvas in preferred format, and records fullscreen-triangle blit pass', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: true,
  });
  swapchain.configure({ width: 1280, height: 720 });

  assert.equal(device.createdTextures.length, 1); // depth only
  assert.equal(context.configurations.length, 1);

  // Switch to non-canvas format rgb10a2unorm.
  assert.equal(swapchain.configureForStreamFormat('rgb10a2unorm'), true);
  // Repeating the same stream format is a no-op.
  assert.equal(swapchain.configureForStreamFormat('rgb10a2unorm'), false);

  assert.equal(swapchain.usesBlitFallback, true);
  assert.equal(swapchain.streamFormat, 'rgb10a2unorm');
  assert.equal(swapchain.canvasFormat, 'bgra8unorm');
  assert.equal(swapchain.format, 'bgra8unorm');

  // Canvas context stayed configured with valid preferred format (no invalid rgb10a2unorm configure).
  assert.equal(context.configurations.length, 1);
  assert.equal(context.configurations[0].format, 'bgra8unorm');

  // Intermediate rgb10a2unorm texture allocated with RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC.
  assert.equal(device.createdTextures.length, 2);
  const intermediateTex = device.createdTextures[1];
  assert.equal(swapchain.intermediateTexture, intermediateTex);
  assert.equal(swapchain.intermediateTextureView, intermediateTex.view);
  assert.deepEqual(intermediateTex.desc, {
    label: 'playgta6-swapchain-intermediate',
    size: { width: 1280, height: 720, depthOrArrayLayers: 1 },
    format: 'rgb10a2unorm',
    usage: INTERMEDIATE_TEXTURE_USAGE,
  });

  // getRenderTargetView() and acquireRenderPassDescriptor() transparently target intermediate view.
  assert.equal(swapchain.getRenderTargetView(), intermediateTex.view);
  const passDesc = swapchain.acquireRenderPassDescriptor({
    clearColor: { r: 0.2, g: 0.4, b: 0.6, a: 1.0 },
  });
  assert.equal(passDesc.colorAttachments[0].view, intermediateTex.view);
  assert.equal(passDesc.depthStencilAttachment.view, device.createdTextures[0].view);

  // Record blit pass onto explicit canvas texture view.
  const encoder1 = createMockCommandEncoder();
  const explicitCanvasView = { kind: 'GPUTextureView', id: 'explicit-canvas-view-1' };
  assert.equal(swapchain.blitToCanvasIfNeeded(encoder1, explicitCanvasView), true);

  assert.equal(device.createdShaderModules.length, 1);
  assert.equal(device.createdSamplers.length, 1);
  assert.equal(device.createdRenderPipelines.length, 1);
  assert.equal(device.createdBindGroups.length, 1);
  assert.equal(
    device.createdRenderPipelines[0].desc.fragment.targets[0].format,
    'bgra8unorm'
  );
  assert.equal(device.createdBindGroups[0].desc.entries[0].resource, intermediateTex.view);
  assert.equal(device.createdBindGroups[0].desc.entries[1].resource, device.createdSamplers[0]);

  assert.equal(encoder1.passes.length, 1);
  const blitPass1 = encoder1.passes[0];
  assert.equal(blitPass1.desc.colorAttachments[0].view, explicitCanvasView);
  assert.equal(blitPass1.desc.colorAttachments[0].loadOp, 'clear');
  assert.equal(blitPass1.desc.colorAttachments[0].storeOp, 'store');
  assert.deepEqual(blitPass1.pipelines, [device.createdRenderPipelines[0]]);
  assert.deepEqual(blitPass1.bindGroups, [{ index: 0, bindGroup: device.createdBindGroups[0] }]);
  assert.equal(blitPass1.draws.length, 1);
  assert.equal(blitPass1.draws[0].vertexCount, 3);
  assert.equal(blitPass1.ended, true);
  assert.equal(swapchain.blitCount, 1);

  // Second blit (via present() with 1-arg auto canvas view) reuses cached pipeline & bindGroup!
  const encoder2 = createMockCommandEncoder();
  assert.equal(swapchain.present(encoder2), true);
  assert.equal(device.createdShaderModules.length, 1);
  assert.equal(device.createdSamplers.length, 1);
  assert.equal(device.createdRenderPipelines.length, 1);
  assert.equal(device.createdBindGroups.length, 1);
  assert.equal(encoder2.passes.length, 1);
  assert.match(encoder2.passes[0].desc.colorAttachments[0].view.id, /^color-view-/);
  assert.equal(encoder2.passes[0].draws[0].vertexCount, 3);
  assert.equal(swapchain.blitCount, 2);

  const stats = swapchain.getStats();
  assert.equal(stats.usesBlitFallback, true);
  assert.equal(stats.streamFormat, 'rgb10a2unorm');
  assert.equal(stats.canvasFormat, 'bgra8unorm');
  assert.equal(stats.blitCount, 2);
});

test('W112 constructor with non-canvas format allocates intermediate lazily on configure and supports format changes', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'rg11b10ufloat',
    enableDepth: false,
  });

  // Constructor must not allocate textures before configure().
  assert.equal(device.createdTextures.length, 0);
  assert.equal(swapchain.usesBlitFallback, true);
  assert.equal(swapchain.streamFormat, 'rg11b10ufloat');
  assert.equal(swapchain.canvasFormat, 'bgra8unorm');

  swapchain.configure({ width: 800, height: 600 });
  assert.equal(context.configurations[0].format, 'bgra8unorm');
  assert.equal(device.createdTextures.length, 1);
  assert.equal(device.createdTextures[0].desc.format, 'rg11b10ufloat');
  assert.deepEqual(device.createdTextures[0].desc.size, {
    width: 800,
    height: 600,
    depthOrArrayLayers: 1,
  });

  const enc1 = createMockCommandEncoder();
  assert.equal(swapchain.blitToCanvasIfNeeded(enc1), true);
  assert.equal(device.createdBindGroups.length, 1);

  // Switch to another non-canvas format (rgba8unorm-srgb): destroys old intermediate & invalidates bind group.
  const firstTex = device.createdTextures[0];
  assert.equal(swapchain.configureForStreamFormat('rgba8unorm-srgb'), true);
  assert.equal(firstTex.destroyed, true);
  assert.equal(device.createdTextures.length, 2);
  assert.equal(device.createdTextures[1].desc.format, 'rgba8unorm-srgb');

  const enc2 = createMockCommandEncoder();
  assert.equal(swapchain.blitToCanvasIfNeeded(enc2), true);
  assert.equal(device.createdBindGroups.length, 2);
  assert.equal(device.createdRenderPipelines.length, 1); // canvasFormat unchanged -> pipeline reused
  assert.equal(device.createdBindGroups[1].desc.entries[0].resource, device.createdTextures[1].view);
});

test('W112 resize destroys old intermediate texture before creating replacement and invalidates bind group', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: false,
  });
  swapchain.configure({ width: 640, height: 360 });
  swapchain.configureForStreamFormat('rgb10a2unorm');

  const firstIntermediate = device.createdTextures[0];
  assert.deepEqual(firstIntermediate.desc.size, {
    width: 640,
    height: 360,
    depthOrArrayLayers: 1,
  });

  const enc1 = createMockCommandEncoder();
  assert.equal(swapchain.blitToCanvasIfNeeded(enc1), true);
  assert.equal(device.createdRenderPipelines.length, 1);
  assert.equal(device.createdBindGroups.length, 1);

  // No-op resize keeps intermediate texture and bind group intact.
  assert.equal(swapchain.resize(640, 360, 1), false);
  assert.equal(firstIntermediate.destroyed, false);
  assert.equal(device.createdTextures.length, 1);

  // Clear events to check destroy-before-create ordering on resize.
  events.length = 0;

  assert.equal(swapchain.resize(960, 540, 2), true);
  assert.equal(firstIntermediate.destroyed, true);
  assert.equal(firstIntermediate.destroyCalls, 1);
  assert.equal(device.createdTextures.length, 2);

  const secondIntermediate = device.createdTextures[1];
  assert.equal(swapchain.intermediateTexture, secondIntermediate);
  assert.deepEqual(secondIntermediate.desc.size, {
    width: 1920,
    height: 1080,
    depthOrArrayLayers: 1,
  });
  assert.equal(secondIntermediate.desc.format, 'rgb10a2unorm');

  const destroyIdx = events.findIndex((e) => e.op === 'depthTexture.destroy' && e.id === firstIntermediate.id);
  const createIdx = events.findIndex((e) => e.op === 'device.createTexture' && e.id === secondIntermediate.id);
  assert.ok(destroyIdx !== -1 && createIdx !== -1 && destroyIdx < createIdx);

  // Next blit must allocate a fresh bind group for the resized view while reusing the blit pipeline.
  const enc2 = createMockCommandEncoder();
  assert.equal(swapchain.blitToCanvasIfNeeded(enc2), true);
  assert.equal(device.createdRenderPipelines.length, 1);
  assert.equal(device.createdBindGroups.length, 2);
  assert.equal(device.createdBindGroups[1].desc.entries[0].resource, secondIntermediate.view);
  assert.equal(enc2.passes[0].bindGroups[0].bindGroup, device.createdBindGroups[1]);
  assert.equal(swapchain.blitCount, 2);
});

test('W112 switching back to direct canvas format (rgba8unorm) destroys intermediate texture and disables blit fallback', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: false,
  });
  swapchain.configure({ width: 1280, height: 720 });
  swapchain.configureForStreamFormat('rgb10a2unorm');

  const intermediateTex = device.createdTextures[0];
  const enc1 = createMockCommandEncoder();
  assert.equal(swapchain.blitToCanvasIfNeeded(enc1), true);
  assert.equal(swapchain.blitCount, 1);

  // Switch back to direct rgba8unorm.
  assert.equal(swapchain.configureForStreamFormat('rgba8unorm'), true);

  assert.equal(intermediateTex.destroyed, true);
  assert.equal(intermediateTex.destroyCalls, 1);
  assert.equal(swapchain.intermediateTexture, null);
  assert.equal(swapchain.intermediateTextureView, null);
  assert.equal(swapchain.usesBlitFallback, false);
  assert.equal(swapchain.streamFormat, 'rgba8unorm');
  assert.equal(swapchain.canvasFormat, 'rgba8unorm');
  assert.equal(swapchain.format, 'rgba8unorm');
  assert.equal(context.configurations[context.configurations.length - 1].format, 'rgba8unorm');

  // Direct view now returns canvas view and blitToCanvasIfNeeded is a no-op.
  const directView = swapchain.getRenderTargetView();
  assert.match(directView.id, /^color-view-/);

  const enc2 = createMockCommandEncoder();
  assert.equal(swapchain.blitToCanvasIfNeeded(enc2, directView), false);
  assert.equal(enc2.passes.length, 0);
  assert.equal(swapchain.blitCount, 1);

  // Re-enabling blit fallback later builds fresh resources (confirming old cache was cleared).
  assert.equal(swapchain.configureForStreamFormat('rgb10a2unorm'), true);
  assert.equal(swapchain.usesBlitFallback, true);
  assert.equal(swapchain.canvasFormat, 'rgba8unorm');
  const enc3 = createMockCommandEncoder();
  assert.equal(swapchain.blitToCanvasIfNeeded(enc3), true);
  assert.equal(device.createdRenderPipelines.length, 2);
  assert.equal(device.createdRenderPipelines[1].desc.fragment.targets[0].format, 'rgba8unorm');
  assert.equal(device.createdBindGroups.length, 2);
  assert.equal(swapchain.blitCount, 2);
});

test('W112 device.lost and recover(newDevice) destroy stale intermediate texture and rebuild blit resources cleanly', async () => {
  const { canvas, context, device, events, makeDevice } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: false,
  });
  swapchain.configure({ width: 1280, height: 720 });
  swapchain.configureForStreamFormat('rgb10a2unorm');

  const oldIntermediate = device.createdTextures[0];
  const enc1 = createMockCommandEncoder();
  assert.equal(swapchain.blitToCanvasIfNeeded(enc1), true);
  assert.equal(swapchain.blitCount, 1);

  device.triggerLost({ reason: 'unknown', message: 'simulated loss' });
  await device.lost;
  await Promise.resolve();

  assert.equal(oldIntermediate.destroyed, true);
  assert.equal(swapchain.intermediateTexture, null);
  assert.equal(swapchain.blitToCanvasIfNeeded(createMockCommandEncoder()), false);

  const newDevice = attachMockBlitCapabilities(makeDevice('device-1'), events);
  swapchain.recover(newDevice, 'rgba8unorm');

  assert.equal(swapchain.usesBlitFallback, true);
  assert.equal(swapchain.streamFormat, 'rgb10a2unorm');
  assert.equal(swapchain.canvasFormat, 'rgba8unorm');
  assert.equal(newDevice.createdTextures.length, 1);
  assert.equal(newDevice.createdTextures[0].desc.format, 'rgb10a2unorm');

  const enc2 = createMockCommandEncoder();
  assert.equal(swapchain.blitToCanvasIfNeeded(enc2), true);
  assert.equal(newDevice.createdShaderModules.length, 1);
  assert.equal(newDevice.createdSamplers.length, 1);
  assert.equal(newDevice.createdRenderPipelines.length, 1);
  assert.equal(newDevice.createdRenderPipelines[0].desc.fragment.targets[0].format, 'rgba8unorm');
  assert.equal(newDevice.createdBindGroups.length, 1);
  assert.equal(newDevice.createdBindGroups[0].desc.entries[0].resource, newDevice.createdTextures[0].view);
  assert.equal(swapchain.blitCount, 2);
});

test('W112 destroy() cleanly destroys intermediate texture, is idempotent, and prevents further blits', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: true,
  });
  swapchain.configure({ width: 800, height: 600 });
  swapchain.configureForStreamFormat('rgb10a2unorm');

  const depthTex = device.createdTextures[0];
  const intermediateTex = device.createdTextures[1];
  const enc = createMockCommandEncoder();
  assert.equal(swapchain.blitToCanvasIfNeeded(enc), true);

  swapchain.destroy();
  assert.equal(depthTex.destroyed, true);
  assert.equal(depthTex.destroyCalls, 1);
  assert.equal(intermediateTex.destroyed, true);
  assert.equal(intermediateTex.destroyCalls, 1);
  assert.equal(swapchain.intermediateTexture, null);
  assert.equal(swapchain.intermediateTextureView, null);

  // Idempotent second destroy().
  swapchain.destroy();
  assert.equal(intermediateTex.destroyCalls, 1);

  // Post-destroy guards.
  assert.throws(() => swapchain.getRenderTargetView(), /destroyed/i);
  assert.equal(swapchain.blitToCanvasIfNeeded(createMockCommandEncoder()), false);
  assert.equal(swapchain.configureForStreamFormat('rgba8unorm'), false);
  assert.equal(swapchain.getStats().blitCount, 1);
});

// ---------------------------------------------------------------------------
// W169 (#876): 1080p <-> 4K plane transitions, sRGB/linear viewFormats,
// graceful viewFormats fallback, and present-cadence telemetry
// ---------------------------------------------------------------------------

test('W169 compatible linear/sRGB sibling viewFormats for bgra8unorm/rgba8unorm and omitted for rgba16float', () => {
  assert.ok(Object.isFrozen(COMPATIBLE_VIEW_FORMATS));
  assert.deepEqual(getCompatibleViewFormats('bgra8unorm'), ['bgra8unorm-srgb']);
  assert.deepEqual(getCompatibleViewFormats('rgba8unorm'), ['rgba8unorm-srgb']);
  assert.deepEqual(getCompatibleViewFormats('bgra8unorm-srgb'), ['bgra8unorm']);
  assert.deepEqual(getCompatibleViewFormats('rgba8unorm-srgb'), ['rgba8unorm']);
  assert.deepEqual(getCompatibleViewFormats('rgba16float'), []);
  assert.deepEqual(getCompatibleViewFormats(''), []);
  assert.deepEqual(getCompatibleViewFormats(null), []);

  const { canvas, context, device } = createMockWebGpuEnv();
  const createdViewDescriptors = [];
  context.getCurrentTexture = () => ({
    createView(desc) {
      createdViewDescriptors.push(desc);
      return { kind: 'GPUTextureView', format: desc ? desc.format : 'default' };
    },
  });

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
  });

  swapchain.configure({ width: 1920, height: 1080 });
  assert.deepEqual(context.configurations[0], {
    device,
    format: 'bgra8unorm',
    alphaMode: 'opaque',
    viewFormats: ['bgra8unorm-srgb'],
  });
  assert.deepEqual(swapchain.viewFormats, ['bgra8unorm-srgb']);
  assert.deepEqual(swapchain.getStats().viewFormats, ['bgra8unorm-srgb']);
  assert.equal(swapchain.getStats().viewFormatsFallbackCount, 0);

  // Acquiring an sRGB view directly works seamlessly.
  const srgbView = swapchain.getRenderTargetView({ format: 'bgra8unorm-srgb' });
  assert.equal(srgbView.format, 'bgra8unorm-srgb');
  assert.deepEqual(createdViewDescriptors[0], { format: 'bgra8unorm-srgb' });

  // Switching to rgba8unorm automatically configures rgba8unorm-srgb sibling viewFormat.
  assert.equal(swapchain.setFormat('rgba8unorm'), true);
  assert.deepEqual(context.configurations[1], {
    device,
    format: 'rgba8unorm',
    alphaMode: 'opaque',
    viewFormats: ['rgba8unorm-srgb'],
  });
  assert.deepEqual(swapchain.viewFormats, ['rgba8unorm-srgb']);

  // Redundant setFormat with the same effective sibling viewFormats is skipped.
  assert.equal(swapchain.setFormat('rgba8unorm', ['rgba8unorm-srgb']), false);
  assert.equal(context.configurations.length, 2);
  assert.equal(swapchain.getTelemetry().skippedRedundantReconfigures, 1);

  // Switching to rgba16float omits viewFormats (no sRGB sibling).
  assert.equal(swapchain.setFormat('rgba16float'), true);
  assert.deepEqual(context.configurations[2], {
    device,
    format: 'rgba16float',
    alphaMode: 'opaque',
  });
  assert.equal('viewFormats' in context.configurations[2], false);
  assert.deepEqual(swapchain.viewFormats, []);
});

test('W169 gracefully retries context.configure without viewFormats when mock or device rejects them', () => {
  const { canvas, context, device, makeDevice } = createMockWebGpuEnv();
  const attempts = [];
  let rejectViewFormats = true;

  context.configure = function configureWithOptionalReject(cfg) {
    attempts.push({ ...cfg, viewFormats: cfg.viewFormats ? cfg.viewFormats.slice() : undefined });
    if (rejectViewFormats && Array.isArray(cfg.viewFormats) && cfg.viewFormats.length > 0) {
      throw new TypeError('Simulated device rejecting canvas viewFormats');
    }
    this.configurations.push({ ...cfg });
  };

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
  });

  // Initial configure attempts viewFormats, catches rejection, retries cleanly without viewFormats.
  const result = swapchain.configure({ width: 1920, height: 1080 });
  assert.equal(result.width, 1920);
  assert.equal(result.height, 1080);
  assert.equal(swapchain.isConfigured, true);
  assert.equal(attempts.length, 2);
  assert.deepEqual(attempts[0].viewFormats, ['bgra8unorm-srgb']);
  assert.equal('viewFormats' in attempts[1] && attempts[1].viewFormats !== undefined, false);
  assert.equal(context.configurations.length, 1);
  assert.deepEqual(context.configurations[0], {
    device,
    format: 'bgra8unorm',
    alphaMode: 'opaque',
  });
  assert.deepEqual(swapchain.viewFormats, []);
  assert.equal(swapchain.getStats().viewFormatsFallbackCount, 1);
  assert.equal(swapchain.getTelemetry().reconfigureCount, 1);

  // Redundant configure after fallback is still recognized and skipped without calling configure() again.
  swapchain.configure({ width: 1920, height: 1080, format: 'bgra8unorm' });
  assert.equal(attempts.length, 2);
  assert.equal(context.configurations.length, 1);
  assert.equal(swapchain.getTelemetry().reconfigureCount, 1);
  assert.equal(swapchain.getTelemetry().skippedRedundantReconfigures, 1);

  // Recovering onto a new device that supports viewFormats restores sibling viewFormats.
  rejectViewFormats = false;
  const newDevice = makeDevice('device-with-srgb-views');
  swapchain.recover(newDevice, 'rgba8unorm');
  assert.deepEqual(swapchain.viewFormats, ['rgba8unorm-srgb']);
  assert.deepEqual(context.configurations[context.configurations.length - 1], {
    device: newDevice,
    format: 'rgba8unorm',
    alphaMode: 'opaque',
    viewFormats: ['rgba8unorm-srgb'],
  });

  // If retry without viewFormats also throws, the error propagates cleanly.
  const env2 = createMockWebGpuEnv();
  env2.context.configure = () => {
    throw new Error('Fatal canvas configure error');
  };
  const failingSwapchain = new WgpuSwapchain({
    canvas: env2.canvas,
    device: env2.device,
    context: env2.context,
    format: 'bgra8unorm',
  });
  assert.throws(
    () => failingSwapchain.configure({ width: 1280, height: 720 }),
    /Fatal canvas configure error/
  );
});

test('W169 1080p UI plane 0 <-> 4K (3840x2160) Bink2/scene plane 1 transitions skip redundant reconfigures, avoid blanking, and preserve presented frames', () => {
  const { context, device, events } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  // Track canvas.width/height setter writes to verify zero redundant backing-store clears.
  let widthWrites = 0;
  let heightWrites = 0;
  let backingWidth = 0;
  let backingHeight = 0;
  const canvas = {
    get width() {
      return backingWidth;
    },
    set width(v) {
      widthWrites += 1;
      backingWidth = v;
    },
    get height() {
      return backingHeight;
    },
    set height(v) {
      heightWrites += 1;
      backingHeight = v;
    },
    getContext(kind) {
      return kind === 'webgpu' ? context : null;
    },
  };

  let clockMs = 1000;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    nowFn: () => clockMs,
  });

  // Plane 0: 1080p UI boot.
  swapchain.configure({ width: 1920, height: 1080, devicePixelRatio: 1 });
  assert.equal(canvas.width, 1920);
  assert.equal(canvas.height, 1080);
  assert.equal(widthWrites, 1);
  assert.equal(heightWrites, 1);
  assert.equal(context.configurations.length, 1);
  assert.equal(swapchain.getTelemetry().reconfigureCount, 1);
  assert.equal(swapchain.getTelemetry().resizeCount, 0);

  // Present 3 frames on 1080p UI plane 0.
  swapchain.recordFramePresented();
  clockMs += 16.5;
  swapchain.recordFramePresented();
  clockMs += 16.5;
  swapchain.recordFramePresented();
  assert.equal(swapchain.getTelemetry().totalFrames, 3);

  // Redundant configure() and resize() at 1080p must be skipped with zero context.configure calls.
  swapchain.configure({ width: 1920, height: 1080, devicePixelRatio: 1, format: 'bgra8unorm' });
  assert.equal(swapchain.resize(1920, 1080, 1), false);
  assert.equal(swapchain.resize(960, 540, 2), false);
  assert.equal(context.configurations.length, 1);
  assert.equal(widthWrites, 1);
  assert.equal(heightWrites, 1);
  assert.equal(swapchain.getTelemetry().reconfigureCount, 1);
  assert.equal(swapchain.getTelemetry().skippedRedundantReconfigures, 3);
  assert.equal(swapchain.getTelemetry().resizeCount, 0);

  // Transition to Plane 1: 4K (3840x2160) Bink2 intro video / 3D scene.
  events.length = 0;
  const depth1080p = device.createdTextures[0];
  assert.equal(swapchain.resize(3840, 2160, 1), true);
  assert.equal(canvas.width, 3840);
  assert.equal(canvas.height, 2160);
  assert.equal(widthWrites, 2);
  assert.equal(heightWrites, 2);
  assert.equal(depth1080p.destroyed, true);
  assert.equal(device.createdTextures.length, 2);
  assert.deepEqual(device.createdTextures[1].desc.size, {
    width: 3840,
    height: 2160,
    depthOrArrayLayers: 1,
  });

  // Verify old 1080p depth texture was destroyed BEFORE allocating the 4K depth texture.
  const destroy1080Idx = events.findIndex((e) => e.op === 'depthTexture.destroy' && e.id === depth1080p.id);
  const create4kIdx = events.findIndex((e) => e.op === 'device.createTexture' && e.id === device.createdTextures[1].id);
  assert.ok(destroy1080Idx !== -1 && create4kIdx !== -1 && destroy1080Idx < create4kIdx);

  // DPR-equivalent 4K calls (1920x1080 @ 2x) are skipped as redundant.
  assert.equal(swapchain.resize(1920, 1080, 2), false);
  swapchain.configure({ width: 3840, height: 2160, devicePixelRatio: 1 });
  assert.equal(context.configurations.length, 2);
  assert.equal(swapchain.getTelemetry().reconfigureCount, 2);
  assert.equal(swapchain.getTelemetry().skippedRedundantReconfigures, 5);
  assert.equal(swapchain.getTelemetry().resizeCount, 1);

  // Present 2 frames at 4K; totalFrames must continue seamlessly without dropping counts.
  clockMs += 33.0;
  swapchain.recordFramePresented();
  clockMs += 33.0;
  swapchain.recordFramePresented();
  assert.equal(swapchain.getTelemetry().totalFrames, 5);

  // Format-only change at 4K does not needlessly rewrite canvas.width / canvas.height.
  assert.equal(swapchain.setFormat('rgba8unorm'), true);
  assert.equal(widthWrites, 2);
  assert.equal(heightWrites, 2);
  assert.equal(swapchain.getTelemetry().reconfigureCount, 3);
  assert.equal(swapchain.getTelemetry().resizeCount, 1);

  // Transition back to Plane 0: 1080p UI.
  const depth4k = device.createdTextures[1];
  assert.equal(swapchain.resize(1920, 1080, 1), true);
  assert.equal(depth4k.destroyed, true);
  assert.equal(canvas.width, 1920);
  assert.equal(canvas.height, 1080);
  assert.equal(swapchain.getTelemetry().reconfigureCount, 4);
  assert.equal(swapchain.getTelemetry().resizeCount, 2);
  assert.equal(swapchain.getTelemetry().totalFrames, 5);

  // Present another frame after transitioning back to 1080p.
  clockMs += 16.5;
  const t = swapchain.recordFramePresented();
  assert.equal(t.totalFrames, 6);
  assert.equal(t.resizeCount, 2);
  assert.equal(t.lastPresentIntervalMs, 16.5);
});

test('W169 resize safely rejects invalid/zero/negative/NaN dimensions and clamps oversized dimensions without dropping state', () => {
  const { canvas, context, device } = createMockWebGpuEnv({
    limits: { maxTextureDimension2D: 4096 },
  });
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    maxDimension: 8192,
  });

  // Before initial configure, invalid resize returns false safely.
  assert.equal(swapchain.resize(0, 0), false);
  assert.equal(swapchain.isConfigured, false);

  // Configure valid 4K (3840x2160) within 4096 device limit.
  swapchain.configure({ width: 3840, height: 2160 });
  const activeDepth = device.createdTextures[0];
  assert.equal(swapchain.width, 3840);
  assert.equal(swapchain.height, 2160);
  assert.equal(swapchain.getTelemetry().reconfigureCount, 1);

  // Invalid / zero / negative / NaN / non-number resize inputs must be safely rejected (no blanking).
  for (const [w, h] of [
    [0, 0],
    [0, 2160],
    [3840, 0],
    [-1920, 1080],
    [1920, -1080],
    [NaN, 1080],
    [1920, NaN],
    [-Infinity, 1080],
    [undefined, 1080],
    ['1920', 1080],
  ]) {
    assert.equal(swapchain.resize(w, h), false, `expected resize(${w}, ${h}) to be rejected`);
  }

  assert.equal(swapchain.width, 3840);
  assert.equal(swapchain.height, 2160);
  assert.equal(canvas.width, 3840);
  assert.equal(canvas.height, 2160);
  assert.equal(activeDepth.destroyed, false);
  assert.equal(context.configurations.length, 1);
  assert.equal(swapchain.getTelemetry().resizeCount, 0);
  assert.equal(swapchain.getTelemetry().reconfigureCount, 1);

  // Non-positive / NaN DPR falls back safely to 1 instead of collapsing 1920x1080 to 1x1.
  assert.equal(swapchain.resize(1920, 1080, 0), true);
  assert.equal(swapchain.width, 1920);
  assert.equal(swapchain.height, 1080);
  assert.equal(swapchain.resize(1920, 1080, -2), false);
  assert.equal(swapchain.resize(1920, 1080, NaN), false);
  assert.equal(swapchain.width, 1920);
  assert.equal(swapchain.height, 1080);

  // Oversized positive dimensions clamp safely to device.limits.maxTextureDimension2D (4096).
  assert.equal(swapchain.resize(16384, 9216, 1), true);
  assert.equal(swapchain.width, 4096);
  assert.equal(swapchain.height, 4096);
  assert.equal(swapchain.getTelemetry().resizeCount, 2);
});

test('W169 present-cadence telemetry tracks lastPresentIntervalMs, avgPresentIntervalMs, frameIntervalUs, and injectable nowFn', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  let clockMs = 500;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    nowFn: () => clockMs,
    frameWindowSize: 4,
  });
  swapchain.configure({ width: 1920, height: 1080 });

  // frameIntervalUs is omitted when not provided.
  let t = swapchain.getTelemetry();
  assert.equal('frameIntervalUs' in t, false);
  assert.equal(t.reconfigureCount, 1);
  assert.equal(t.skippedRedundantReconfigures, 0);
  assert.equal(t.lastPresentIntervalMs, 0);
  assert.equal(t.avgPresentIntervalMs, 0);

  // Frame 1 establishes baseline timestamp with declared 60Hz target interval (16667 us).
  t = swapchain.recordFramePresented({ frameIntervalUs: 16667 });
  assert.equal(t.totalFrames, 1);
  assert.equal(t.lastPresentIntervalMs, 0);
  assert.equal(t.avgPresentIntervalMs, 0);
  assert.equal(t.frameIntervalUs, 16667);
  assert.equal(swapchain.frameIntervalUs, 16667);

  // Frame 2 (+16ms) and Frame 3 (+20ms) -> last=20, avg=18.
  clockMs += 16;
  t = swapchain.recordFramePresented();
  assert.equal(t.lastPresentIntervalMs, 16);
  assert.equal(t.avgPresentIntervalMs, 16);

  clockMs += 20;
  t = swapchain.recordFramePresented();
  assert.equal(t.lastPresentIntervalMs, 20);
  assert.equal(t.avgPresentIntervalMs, 18);
  assert.equal(t.frameIntervalUs, 16667);

  // Update declared frameIntervalUs to 30Hz (33333 us) during 4K transition via recordFramePresented metadata.
  clockMs += 34;
  t = swapchain.recordFramePresented(undefined, { frameIntervalUs: 33333 });
  assert.equal(t.lastPresentIntervalMs, 34);
  assert.ok(Math.abs(t.avgPresentIntervalMs - (16 + 20 + 34) / 3) < 1e-9);
  assert.equal(t.frameIntervalUs, 33333);
  assert.equal(swapchain.getStats().frameIntervalUs, 33333);

  // Explicit timestamp + numeric frameIntervalUs + rolling window eviction (window=4).
  t = swapchain.recordFramePresented(clockMs + 30, 33333); // deltas: [16, 20, 34, 30] -> avg 25
  assert.equal(t.lastPresentIntervalMs, 30);
  assert.equal(t.avgPresentIntervalMs, 25);

  t = swapchain.recordFramePresented(clockMs + 62); // +32ms -> deltas: [20, 34, 30, 32] -> avg 29
  assert.equal(t.lastPresentIntervalMs, 32);
  assert.equal(t.avgPresentIntervalMs, 29);

  // Clearing frameIntervalUs removes it from telemetry again.
  swapchain.setFrameIntervalUs(null);
  assert.equal('frameIntervalUs' in swapchain.getTelemetry(), false);
  assert.equal(swapchain.frameIntervalUs, null);

  // Constructor validates nowFn, frameIntervalUs, and viewFormats cleanly.
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, nowFn: 'bad' }),
    TypeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, frameIntervalUs: -1 }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, frameIntervalUs: NaN }),
    RangeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, viewFormats: 'bgra8unorm-srgb' }),
    TypeError
  );
});

// ---------------------------------------------------------------------------
// W197: Swapchain configuration elision across (width, height, format, usage,
// alphaMode, device) & zero-allocation frame-pacing telemetry
// ---------------------------------------------------------------------------

test('W197 elides redundant context.configure calls when (width, height, format, usage, alphaMode, device) are unchanged and reconfigures on any tuple change', () => {
  assert.equal(DEFAULT_USAGE, TEXTURE_USAGE_RENDER_ATTACHMENT);
  assert.ok(Math.abs(DEFAULT_DROPPED_FRAME_THRESHOLD_MS - 1000 / 30) < 1e-9);
  assert.equal(DROPPED_FRAME_CADENCE_MULTIPLIER, 1.5);

  const { context, device } = createMockWebGpuEnv();
  let widthWrites = 0;
  let heightWrites = 0;
  let backingWidth = 0;
  let backingHeight = 0;
  const canvas = {
    get width() {
      return backingWidth;
    },
    set width(v) {
      widthWrites += 1;
      backingWidth = v;
    },
    get height() {
      return backingHeight;
    },
    set height(v) {
      heightWrites += 1;
      backingHeight = v;
    },
    getContext(kind) {
      return kind === 'webgpu' ? context : null;
    },
  };

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    alphaMode: 'opaque',
  });

  // Initial configure: default usage is TEXTURE_USAGE_RENDER_ATTACHMENT and omitted from cfg when unspecified.
  swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 1 });
  assert.equal(context.configurations.length, 1);
  assert.equal('usage' in context.configurations[0], false);
  assert.equal(swapchain.usage, TEXTURE_USAGE_RENDER_ATTACHMENT);
  assert.equal(swapchain.reconfigureCount, 1);
  assert.equal(device.createdTextures.length, 1);
  assert.equal(widthWrites, 1);
  assert.equal(heightWrites, 1);

  // Repeated configure with identical tuple — including explicit default usage, device, alphaMode,
  // format, and DPR-equivalent pixel size — must be completely elided.
  swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 1 });
  swapchain.configure({
    width: 640,
    height: 360,
    devicePixelRatio: 2,
    format: 'bgra8unorm',
    usage: TEXTURE_USAGE_RENDER_ATTACHMENT,
    alphaMode: 'opaque',
    device,
  });
  assert.equal(swapchain.resize(1280, 720, 1), false);
  assert.equal(swapchain.setDevice(device), false);

  assert.equal(context.configurations.length, 1);
  assert.equal(swapchain.reconfigureCount, 1);
  assert.equal(swapchain.getTelemetry().skippedRedundantReconfigures, 4);
  assert.equal(device.createdTextures.length, 1);
  assert.equal(widthWrites, 1);
  assert.equal(heightWrites, 1);

  // 1. Changing ONLY usage triggers reconfigure with explicit usage; repeating elides.
  const copyAndRenderUsage = TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC;
  swapchain.configure({ usage: copyAndRenderUsage });
  assert.equal(context.configurations.length, 2);
  assert.equal(context.configurations[1].usage, copyAndRenderUsage);
  assert.equal(swapchain.usage, copyAndRenderUsage);
  assert.equal(swapchain.getStats().usage, copyAndRenderUsage);
  assert.equal(swapchain.reconfigureCount, 2);
  assert.equal(device.createdTextures.length, 1); // depth texture preserved
  assert.equal(widthWrites, 1);
  assert.equal(heightWrites, 1);

  swapchain.configure({ width: 1280, height: 720, usage: copyAndRenderUsage });
  assert.equal(context.configurations.length, 2);
  assert.equal(swapchain.reconfigureCount, 2);
  assert.equal(swapchain.getTelemetry().skippedRedundantReconfigures, 5);

  // Switching usage back to RENDER_ATTACHMENT triggers reconfigure; repeating elides.
  swapchain.configure({ usage: TEXTURE_USAGE_RENDER_ATTACHMENT });
  assert.equal(context.configurations.length, 3);
  assert.equal(context.configurations[2].usage, TEXTURE_USAGE_RENDER_ATTACHMENT);
  assert.equal(swapchain.reconfigureCount, 3);
  swapchain.configure({ usage: TEXTURE_USAGE_RENDER_ATTACHMENT });
  assert.equal(context.configurations.length, 3);

  // 2. Changing ONLY alphaMode triggers reconfigure; repeating elides.
  swapchain.configure({ alphaMode: 'premultiplied' });
  assert.equal(context.configurations.length, 4);
  assert.equal(context.configurations[3].alphaMode, 'premultiplied');
  assert.equal(swapchain.reconfigureCount, 4);
  swapchain.configure({ alphaMode: 'premultiplied' });
  assert.equal(context.configurations.length, 4);

  // 3. Changing ONLY format triggers reconfigure; repeating elides.
  swapchain.configure({ format: 'rgba8unorm' });
  assert.equal(context.configurations.length, 5);
  assert.equal(context.configurations[4].format, 'rgba8unorm');
  assert.equal(swapchain.reconfigureCount, 5);
  swapchain.configure({ format: 'rgba8unorm' });
  assert.equal(context.configurations.length, 5);

  // 4. Leaving blit fallback via setFormat() when canvas format/viewFormats didn't change elides context.configure.
  assert.equal(swapchain.configureForStreamFormat('rgb10a2unorm'), true);
  assert.equal(context.configurations.length, 5);
  assert.equal(swapchain.setFormat('rgba8unorm'), true);
  assert.equal(swapchain.usesBlitFallback, false);
  assert.equal(context.configurations.length, 5);
  assert.equal(swapchain.reconfigureCount, 5);

  // 5. Validation guards for invalid usage / device / droppedFrameThresholdMs.
  assert.throws(() => new WgpuSwapchain({ canvas, device, context, usage: 0 }), RangeError);
  assert.throws(() => new WgpuSwapchain({ canvas, device, context, usage: -4 }), RangeError);
  assert.throws(() => new WgpuSwapchain({ canvas, device, context, usage: 1.5 }), RangeError);
  assert.throws(() => swapchain.configure({ usage: 0 }), RangeError);
  assert.throws(() => swapchain.configure({ device: null }), TypeError);
  assert.throws(() => swapchain.setDevice(null), TypeError);
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, droppedFrameThresholdMs: -1 }),
    RangeError
  );
  assert.throws(() => swapchain.configure({ droppedFrameThresholdMs: NaN }), RangeError);
  assert.throws(() => swapchain.recordFrameDropped(-1), RangeError);
  assert.throws(() => swapchain.recordFrameDropped(1.5), RangeError);
});

test('W197 cleanly reconfigures on resize and live/lost device replacement via configure({ device }) and setDevice()', async () => {
  const { canvas, context, device, events, makeDevice } = createMockWebGpuEnv({
    limits: { maxTextureDimension2D: 4096 },
  });
  attachMockEventTarget(device);
  attachMockBlitCapabilities(device, events);

  const customUsage = TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    usage: customUsage,
    enableDepth: true,
  });

  swapchain.configure({ width: 3840, height: 2160 });
  assert.equal(context.configurations.length, 1);
  assert.equal(context.configurations[0].device, device);
  assert.equal(context.configurations[0].usage, customUsage);
  assert.equal(device.createdTextures.length, 1);
  assert.equal(device.listenerCount('uncapturederror'), 1);

  const oldDepthTex = device.createdTextures[0];

  // Replace device via configure({ device: device2 }) with smaller maxTextureDimension2D (2048).
  const device2 = attachMockEventTarget(
    attachMockBlitCapabilities(makeDevice('device-2', { maxTextureDimension2D: 2048 }), events)
  );
  const res2 = swapchain.configure({ device: device2 });
  assert.equal(res2.width, 2048);
  assert.equal(res2.height, 2048);
  assert.equal(swapchain.device, device2);
  assert.equal(oldDepthTex.destroyed, true);
  assert.equal(device.listenerCount('uncapturederror'), 0);
  assert.equal(device2.listenerCount('uncapturederror'), 1);
  assert.equal(context.configurations.length, 2);
  assert.equal(context.configurations[1].device, device2);
  assert.equal(context.configurations[1].usage, customUsage);
  assert.equal(device2.createdTextures.length, 1);
  assert.equal(swapchain.reconfigureCount, 2);
  assert.equal(swapchain.getTelemetry().resizeCount, 1);

  // Subsequent configure with device2 and unchanged size is elided.
  swapchain.configure({ device: device2, width: 2048, height: 2048, usage: customUsage });
  assert.equal(context.configurations.length, 2);
  assert.equal(swapchain.reconfigureCount, 2);

  // Resize on device2 reconfigures cleanly and destroys previous depth texture.
  const depthOnDevice2 = device2.createdTextures[0];
  assert.equal(swapchain.resize(1280, 720, 1), true);
  assert.equal(depthOnDevice2.destroyed, true);
  assert.equal(context.configurations.length, 3);
  assert.equal(swapchain.reconfigureCount, 3);
  assert.equal(swapchain.getTelemetry().resizeCount, 2);

  // Trigger device loss on device2, then replace directly via setDevice(device3).
  device2.triggerLost({ reason: 'unknown', message: 'simulated crash' });
  await device2.lost;
  await Promise.resolve();
  assert.equal(swapchain.isDeviceLost, true);

  // Configuring with the SAME lost device still rejects.
  assert.throws(() => swapchain.configure({ device: device2, width: 1280, height: 720 }), /lost/i);

  const device3 = attachMockEventTarget(makeDevice('device-3', { maxTextureDimension2D: 4096 }));
  assert.equal(swapchain.setDevice(device3), true);
  assert.equal(swapchain.isDeviceLost, false);
  assert.equal(swapchain.device, device3);
  assert.equal(device2.listenerCount('uncapturederror'), 0);
  assert.equal(device3.listenerCount('uncapturederror'), 1);
  assert.equal(context.configurations.length, 4);
  assert.equal(context.configurations[3].device, device3);
  assert.equal(context.configurations[3].usage, customUsage);
  assert.equal(swapchain.reconfigureCount, 4);

  // Repeating setDevice(device3) on live already-configured device is elided.
  assert.equal(swapchain.setDevice(device3), false);
  assert.equal(context.configurations.length, 4);
  assert.equal(swapchain.reconfigureCount, 4);

  // Replacing a lost device via configure({ device: device4 }) also recovers cleanly.
  device3.triggerLost({ reason: 'destroyed', message: 'lost again' });
  await device3.lost;
  await Promise.resolve();
  assert.equal(swapchain.isDeviceLost, true);

  const device4 = attachMockEventTarget(makeDevice('device-4', { maxTextureDimension2D: 4096 }));
  swapchain.configure({ device: device4, width: 1920, height: 1080 });
  assert.equal(swapchain.isDeviceLost, false);
  assert.equal(swapchain.device, device4);
  assert.equal(swapchain.width, 1920);
  assert.equal(swapchain.height, 1080);
  assert.equal(context.configurations.length, 5);
  assert.equal(context.configurations[4].device, device4);
  assert.equal(swapchain.reconfigureCount, 5);
});

test('W197 tracks zero-allocation frame-pacing telemetry (presentCount, reconfigureCount, droppedFrameCount, lastFrameMs, avgFrameMs, maxFrameMs)', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  let clockMs = 1000;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    nowFn: () => clockMs,
    frameWindowSize: 4,
    frameIntervalUs: 16667, // 60Hz -> 1.5x drop threshold = 25.0005ms
  });

  // Ring buffer is pre-allocated as a fixed-size Float64Array.
  assert.ok(swapchain._frameDeltaRing instanceof Float64Array);
  assert.equal(swapchain._frameDeltaRing.length, 4);

  swapchain.configure({ width: 1280, height: 720 });

  // Initial pacing stats before presenting any frame.
  const pacingOut = {};
  const initPacing = swapchain.getFramePacingStats(pacingOut);
  assert.strictEqual(initPacing, pacingOut);
  assert.deepEqual(initPacing, {
    presentCount: 0,
    reconfigureCount: 1,
    droppedFrameCount: 0,
    lastFrameMs: 0,
    avgFrameMs: 0,
    maxFrameMs: 0,
  });

  // Frame 1 (baseline) + Frame 2 (+16ms) + Frame 3 (+40ms drop!) + Frame 4 (+20ms) + Frame 5 (+16ms).
  const t1 = swapchain.recordFramePresented();
  clockMs += 16;
  const t2 = swapchain.recordFramePresented();
  clockMs += 40; // > 25.0005ms -> droppedFrameCount += 1
  const t3 = swapchain.recordFramePresented();
  clockMs += 20;
  const t4 = swapchain.recordFramePresented();
  clockMs += 16;
  const t5 = swapchain.recordFramePresented();

  // Zero per-frame allocation check: recordFramePresented reuses the exact same telemetry object & Float64Array ring.
  assert.strictEqual(t1, t2);
  assert.strictEqual(t2, t3);
  assert.strictEqual(t3, t4);
  assert.strictEqual(t4, t5);

  // Rolling window of 4 deltas: [16, 40, 20, 16] -> avg = 23, max = 40, last = 16, dropped = 1.
  assert.equal(t5.presentCount, 5);
  assert.equal(t5.totalFrames, 5);
  assert.equal(t5.reconfigureCount, 1);
  assert.equal(t5.droppedFrameCount, 1);
  assert.equal(t5.lastFrameMs, 16);
  assert.equal(t5.avgFrameMs, 23);
  assert.equal(t5.maxFrameMs, 40);

  // Direct scalar getters match telemetry & getStats() & getFramePacingStats().
  assert.equal(swapchain.presentCount, 5);
  assert.equal(swapchain.reconfigureCount, 1);
  assert.equal(swapchain.droppedFrameCount, 1);
  assert.equal(swapchain.lastFrameMs, 16);
  assert.equal(swapchain.avgFrameMs, 23);
  assert.equal(swapchain.maxFrameMs, 40);

  const reusableTelemetry = {};
  const filled = swapchain.getTelemetry(reusableTelemetry);
  assert.strictEqual(filled, reusableTelemetry);
  assert.equal(filled.presentCount, 5);
  assert.equal(filled.droppedFrameCount, 1);
  assert.equal(filled.lastFrameMs, 16);
  assert.equal(filled.avgFrameMs, 23);
  assert.equal(filled.maxFrameMs, 40);

  // No-arg getTelemetry() returns fresh independent snapshots.
  const snapA = swapchain.getTelemetry();
  const snapB = swapchain.getTelemetry();
  assert.notStrictEqual(snapA, snapB);
  assert.deepEqual(snapA, snapB);

  const stats = swapchain.getStats();
  assert.equal(stats.presentCount, 5);
  assert.equal(stats.reconfigureCount, 1);
  assert.equal(stats.droppedFrameCount, 1);
  assert.equal(stats.lastFrameMs, 16);
  assert.equal(stats.avgFrameMs, 23);
  assert.equal(stats.maxFrameMs, 40);

  // Explicit dropped frame API + explicit { dropped: true } metadata.
  assert.equal(swapchain.recordFrameDropped(), 2);
  assert.equal(swapchain.recordFrameDropped(2), 4);
  clockMs += 16;
  swapchain.recordFramePresented({ timestampMs: clockMs, dropped: true });
  assert.equal(swapchain.droppedFrameCount, 5);
  assert.equal(swapchain.presentCount, 6);

  // High-frequency steady-state loop: 1000 frames at 16ms evicts the 40ms spike and never reallocates the ring.
  const ringRef = swapchain._frameDeltaRing;
  for (let i = 0; i < 1000; i += 1) {
    clockMs += 16;
    const live = swapchain.recordFramePresented(clockMs);
    assert.strictEqual(live, t1);
  }
  assert.strictEqual(swapchain._frameDeltaRing, ringRef);
  assert.equal(swapchain._frameDeltaRing.length, 4);

  swapchain.getFramePacingStats(pacingOut);
  assert.deepEqual(pacingOut, {
    presentCount: 1006,
    reconfigureCount: 1,
    droppedFrameCount: 5,
    lastFrameMs: 16,
    avgFrameMs: 16,
    maxFrameMs: 16,
  });

  // Explicit droppedFrameThresholdMs override (including 0 to disable automatic drop detection).
  swapchain.configure({ droppedFrameThresholdMs: 10 });
  clockMs += 12; // > 10ms -> dropped
  swapchain.recordFramePresented(clockMs);
  assert.equal(swapchain.droppedFrameCount, 6);

  swapchain.configure({ droppedFrameThresholdMs: 0 });
  clockMs += 200; // auto-detection disabled -> not counted
  swapchain.recordFramePresented(clockMs);
  assert.equal(swapchain.droppedFrameCount, 6);
  assert.equal(swapchain.lastFrameMs, 200);
  assert.equal(swapchain.maxFrameMs, 200);
});

// ---------------------------------------------------------------------------
// W231 (#1031): Zero-GPU-readback visible-content readiness & black-loading
// present telemetry
// ---------------------------------------------------------------------------

test('W231 tracks black-loading presents, first visible frame readiness, and later black transitions via markFrameContent + recordPresent', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  let clockMs = 1000;
  const firstVisibleEvents = [];

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    nowFn: () => clockMs,
    onFirstVisibleFrame(info) {
      firstVisibleEvents.push({ ...info });
      // Mutate received info to verify internal state isolation.
      info.firstVisiblePresentIndex = 999999;
      info.hasVisibleContent = false;
    },
  });
  swapchain.configure({ width: 1280, height: 720 });

  // Initial state before any present.
  assert.equal(swapchain.hasVisibleContent, false);
  assert.equal(swapchain.firstVisiblePresentIndex, null);
  assert.equal(swapchain.firstVisiblePresentMs, null);
  assert.equal(swapchain.consecutiveBlackPresents, 0);
  assert.equal(swapchain.blackPresentsTotal, 0);
  assert.equal(swapchain.visiblePresentsTotal, 0);
  assert.equal(firstVisibleEvents.length, 0);

  const initStats = swapchain.getStats();
  assert.equal(initStats.hasVisibleContent, false);
  assert.equal(initStats.firstVisiblePresentIndex, null);
  assert.equal(initStats.firstVisiblePresentMs, null);
  assert.equal(initStats.consecutiveBlackPresents, 0);
  assert.equal(initStats.blackPresentsTotal, 0);
  assert.equal(initStats.visiblePresentsTotal, 0);

  // Present 3 black/clear-only boot frames.
  assert.equal(
    swapchain.markFrameContent({ hasDraws: false, realPackDraws: 0, nonClearDraw: false }),
    false
  );
  swapchain.recordPresent();
  clockMs += 16;
  swapchain.recordPresent(); // unmarked frame defaults to black
  clockMs += 16;
  swapchain.markFrameContent({ hasDraws: 0, realPackDraws: 0, nonClearDraw: 0 });
  swapchain.recordPresent();

  assert.equal(swapchain.presentCount, 3);
  assert.equal(swapchain.hasVisibleContent, false);
  assert.equal(swapchain.firstVisiblePresentIndex, null);
  assert.equal(swapchain.firstVisiblePresentMs, null);
  assert.equal(swapchain.consecutiveBlackPresents, 3);
  assert.equal(swapchain.blackPresentsTotal, 3);
  assert.equal(swapchain.visiblePresentsTotal, 0);
  assert.equal(firstVisibleEvents.length, 0);

  // Frame 4: multi-pass frame where pass 1 has real draws and pass 2 is clear-only.
  clockMs = 1500;
  assert.equal(swapchain.markFrameContent({ hasDraws: false, realPackDraws: 0 }), false);
  assert.equal(
    swapchain.markFrameContent({ hasDraws: true, realPackDraws: 5, nonClearDraw: true }),
    true
  );
  assert.equal(
    swapchain.markFrameContent({ hasDraws: false, realPackDraws: 0, nonClearDraw: false }),
    true
  );

  // Before present commits the frame, readiness counters have not fired yet.
  assert.equal(swapchain.hasVisibleContent, false);
  assert.equal(firstVisibleEvents.length, 0);

  const t4 = swapchain.recordPresent();
  assert.equal(t4.presentCount, 4);
  assert.equal(t4.hasVisibleContent, true);
  assert.equal(t4.firstVisiblePresentIndex, 4);
  assert.equal(t4.firstVisiblePresentMs, 1500);
  assert.equal(t4.consecutiveBlackPresents, 0);
  assert.equal(t4.blackPresentsTotal, 3);
  assert.equal(t4.visiblePresentsTotal, 1);

  assert.equal(firstVisibleEvents.length, 1);
  assert.deepEqual(firstVisibleEvents[0], {
    presentIndex: 4,
    firstVisiblePresentIndex: 4,
    timestampMs: 1500,
    firstVisiblePresentMs: 1500,
    blackPresentsTotal: 3,
    consecutiveBlackPresents: 0,
    visiblePresentsTotal: 1,
    realPackDraws: 5,
    hasVisibleContent: true,
    width: 1280,
    height: 720,
  });

  // Internal state unaffected by callback mutation.
  assert.equal(swapchain.firstVisiblePresentIndex, 4);
  assert.equal(swapchain.hasVisibleContent, true);

  // Frame 5: another visible frame -> callback must NOT re-fire; firstVisible* stays latched.
  clockMs = 1516;
  swapchain.markFrameContent({ nonClearDraw: true });
  swapchain.recordPresent();
  assert.equal(firstVisibleEvents.length, 1);
  assert.equal(swapchain.firstVisiblePresentIndex, 4);
  assert.equal(swapchain.firstVisiblePresentMs, 1500);
  assert.equal(swapchain.visiblePresentsTotal, 2);
  assert.equal(swapchain.blackPresentsTotal, 3);
  assert.equal(swapchain.consecutiveBlackPresents, 0);

  // Frames 6 & 7: subsequent black loading screen frames increment consecutiveBlackPresents
  // while preserving hasVisibleContent=true and latched firstVisible*.
  clockMs = 1532;
  swapchain.recordPresent();
  clockMs = 1548;
  swapchain.recordPresent();
  assert.equal(swapchain.hasVisibleContent, true);
  assert.equal(swapchain.firstVisiblePresentIndex, 4);
  assert.equal(swapchain.firstVisiblePresentMs, 1500);
  assert.equal(swapchain.consecutiveBlackPresents, 2);
  assert.equal(swapchain.blackPresentsTotal, 5);
  assert.equal(swapchain.visiblePresentsTotal, 2);
  assert.equal(swapchain.blackPresentsTotal + swapchain.visiblePresentsTotal, swapchain.presentCount);

  // Frame 8: visible again -> consecutiveBlackPresents resets to 0.
  clockMs = 1564;
  swapchain.markFrameContent({ realPackDraws: 2 });
  const t8 = swapchain.recordPresent();
  assert.equal(t8.consecutiveBlackPresents, 0);
  assert.equal(t8.blackPresentsTotal, 5);
  assert.equal(t8.visiblePresentsTotal, 3);
  assert.equal(t8.presentCount, 8);
  assert.equal(firstVisibleEvents.length, 1);

  // getStats() and getTelemetry() expose identical sanitized readiness counters.
  const stats = swapchain.getStats();
  const telem = swapchain.getTelemetry();
  for (const key of [
    'hasVisibleContent',
    'firstVisiblePresentIndex',
    'firstVisiblePresentMs',
    'consecutiveBlackPresents',
    'blackPresentsTotal',
    'visiblePresentsTotal',
  ]) {
    assert.equal(stats[key], telem[key], `mismatch on ${key}`);
  }
});

test('W231 supports visible-content metadata directly in present() and recordPresent(), including blit fallback and timestamp precedence', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  let clockMs = 2000;
  const fired = [];
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: false,
    nowFn: () => clockMs,
    onFirstVisibleFrame: (info) => fired.push(info),
  });
  swapchain.configure({ width: 800, height: 600 });
  swapchain.configureForStreamFormat('rgb10a2unorm');

  // 1. present(encoder, canvasView, meta) with black frame -> blits and counts as black.
  const enc1 = createMockCommandEncoder();
  const explicitView = { kind: 'GPUTextureView', id: 'canvas-view-w231' };
  assert.equal(
    swapchain.present(enc1, explicitView, { hasVisibleContent: false, realPackDraws: 0 }),
    true
  );
  assert.equal(swapchain.blitCount, 1);
  assert.equal(swapchain.presentCount, 1);
  assert.equal(swapchain.blackPresentsTotal, 1);
  assert.equal(swapchain.consecutiveBlackPresents, 1);
  assert.equal(swapchain.hasVisibleContent, false);

  // 2. present(encoder, meta) 2-arg form with realPackDraws > 0 triggers first visible frame.
  clockMs = 2033;
  const enc2 = createMockCommandEncoder();
  assert.equal(swapchain.present(enc2, { realPackDraws: 4, timestampMs: 2050 }), true);
  assert.equal(swapchain.blitCount, 2);
  assert.equal(swapchain.presentCount, 2);
  assert.equal(swapchain.hasVisibleContent, true);
  assert.equal(swapchain.firstVisiblePresentIndex, 2);
  assert.equal(swapchain.firstVisiblePresentMs, 2050);
  assert.equal(swapchain.visiblePresentsTotal, 1);
  assert.equal(swapchain.consecutiveBlackPresents, 0);
  assert.equal(fired.length, 1);
  assert.equal(fired[0].realPackDraws, 4);
  assert.equal(fired[0].firstVisiblePresentMs, 2050);

  // 3. Metadata-only present({ hasVisibleContent: true }) without encoder returns false and records present.
  clockMs = 2066;
  assert.equal(swapchain.present({ hasVisibleContent: true, realPackDraws: 2 }), false);
  assert.equal(swapchain.presentCount, 3);
  assert.equal(swapchain.visiblePresentsTotal, 2);
  assert.equal(swapchain.blitCount, 2);

  // 4. Explicit hasVisibleContent: false on recordPresent overrides prior markFrameContent({ hasDraws: true }).
  swapchain.markFrameContent({ hasDraws: true, realPackDraws: 1 });
  swapchain.recordPresent(2100, { hasVisibleContent: false });
  assert.equal(swapchain.presentCount, 4);
  assert.equal(swapchain.blackPresentsTotal, 2);
  assert.equal(swapchain.consecutiveBlackPresents, 1);
  assert.equal(swapchain.visiblePresentsTotal, 2);

  // 5. markFrameContent({ timestampMs }) is used when recordPresent() omits explicit timestampMs.
  swapchain.resetStats();
  clockMs = 9000;
  swapchain.markFrameContent({ hasDraws: true, timestampMs: 4321 });
  swapchain.recordPresent();
  assert.equal(swapchain.firstVisiblePresentIndex, 1);
  assert.equal(swapchain.firstVisiblePresentMs, 4321);

  // 6. Explicit recordPresent({ timestampMs }) overrides markFrameContent({ timestampMs }).
  swapchain.reset();
  swapchain.markFrameContent({ hasDraws: true, timestampMs: 4321 });
  swapchain.recordPresent({ timestampMs: 7777 });
  assert.equal(swapchain.firstVisiblePresentIndex, 1);
  assert.equal(swapchain.firstVisiblePresentMs, 7777);
});

test('W231 isolates onFirstVisibleFrame callback errors and re-arms cleanly on resetStats() and reset()', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  attachMockEventTarget(device);

  let clockMs = 100;
  let shouldThrow = true;
  const calls = [];

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    nowFn: () => clockMs,
    onFirstVisibleFrame(info) {
      calls.push(info);
      if (shouldThrow) {
        throw new Error('UI splash transition callback boom');
      }
    },
  });
  swapchain.configure({ width: 1920, height: 1080 });
  const depthTexBeforeReset = device.createdTextures[0];

  // Record a validation error to verify resetStats() clears stats while keeping GPU resources alive.
  device.dispatchUncapturedError({ name: 'GPUValidationError', message: 'pre-reset error' });
  assert.equal(swapchain.getStats().validationErrorCount, 1);

  // Black frame then first visible frame whose callback throws.
  swapchain.recordPresent();
  clockMs = 250;
  assert.doesNotThrow(() => {
    swapchain.recordPresent({ hasVisibleContent: true, realPackDraws: 7 });
  });

  // Callback error was isolated; counters are intact and callback is NOT retried on next visible frame.
  assert.equal(calls.length, 1);
  assert.equal(calls[0].firstVisiblePresentIndex, 2);
  assert.equal(calls[0].firstVisiblePresentMs, 250);
  assert.equal(swapchain.hasVisibleContent, true);
  assert.equal(swapchain.firstVisiblePresentIndex, 2);
  assert.equal(swapchain.firstVisiblePresentMs, 250);
  assert.equal(swapchain.visiblePresentsTotal, 1);
  assert.equal(swapchain.blackPresentsTotal, 1);

  clockMs = 266;
  swapchain.recordPresent({ hasVisibleContent: true });
  assert.equal(calls.length, 1);
  assert.equal(swapchain.visiblePresentsTotal, 2);

  // Add a pending mark before resetStats() to ensure resetStats() clears pending frame content too.
  swapchain.markFrameContent({ hasDraws: true, realPackDraws: 3, timestampMs: 999 });
  swapchain.resetStats();

  // Configured GPU state preserved; all telemetry and readiness counters reset cleanly.
  assert.equal(swapchain.isConfigured, true);
  assert.equal(swapchain.width, 1920);
  assert.equal(swapchain.height, 1080);
  assert.equal(depthTexBeforeReset.destroyed, false);

  const resetStats = swapchain.getStats();
  assert.equal(resetStats.totalFrames, 0);
  assert.equal(resetStats.presentCount, 0);
  assert.equal(resetStats.droppedFrameCount, 0);
  assert.equal(resetStats.validationErrorCount, 0);
  assert.equal(resetStats.validationErrorHistorySize, 0);
  assert.equal(resetStats.hasVisibleContent, false);
  assert.equal(resetStats.firstVisiblePresentIndex, null);
  assert.equal(resetStats.firstVisiblePresentMs, null);
  assert.equal(resetStats.consecutiveBlackPresents, 0);
  assert.equal(resetStats.blackPresentsTotal, 0);
  assert.equal(resetStats.visiblePresentsTotal, 0);

  // Stale pre-reset markFrameContent was cleared -> unmarked present counts as black.
  clockMs = 300;
  swapchain.recordPresent();
  assert.equal(swapchain.hasVisibleContent, false);
  assert.equal(swapchain.blackPresentsTotal, 1);
  assert.equal(calls.length, 1);

  // Callback is re-armed after resetStats() and fires once on the new session's first visible frame.
  shouldThrow = false;
  clockMs = 400;
  swapchain.recordPresent({ realPackDraws: 2 });
  assert.equal(calls.length, 2);
  assert.equal(calls[1].firstVisiblePresentIndex, 2);
  assert.equal(calls[1].firstVisiblePresentMs, 400);
  assert.equal(calls[1].blackPresentsTotal, 1);

  // reset() alias also re-arms callback cleanly.
  swapchain.reset();
  clockMs = 500;
  swapchain.recordPresent({ hasDraws: true });
  assert.equal(calls.length, 3);
  assert.equal(calls[2].firstVisiblePresentIndex, 1);
  assert.equal(calls[2].firstVisiblePresentMs, 500);
});

test('W231 validates onFirstVisibleFrame option, sanitizes invalid inputs, and handles destroy/device-lost safely', async () => {
  const { canvas, context, device } = createMockWebGpuEnv();

  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, onFirstVisibleFrame: 'nope' }),
    TypeError
  );
  assert.throws(
    () => new WgpuSwapchain({ canvas, device, context, onFirstVisibleFrame: 42 }),
    TypeError
  );

  let clockMs = 500;
  let firedCount = 0;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    nowFn: () => clockMs,
    onFirstVisibleFrame: () => {
      firedCount += 1;
    },
  });
  swapchain.configure({ width: 640, height: 360 });

  // Invalid / negative / NaN / Infinity inputs never mark visible or corrupt counters.
  assert.equal(swapchain.markFrameContent(), false);
  assert.equal(swapchain.markFrameContent(null), false);
  assert.equal(swapchain.markFrameContent({}), false);
  assert.equal(
    swapchain.markFrameContent({
      hasDraws: -1,
      realPackDraws: -10,
      nonClearDraw: NaN,
      timestampMs: Infinity,
    }),
    false
  );
  assert.equal(
    swapchain.markFrameContent({
      hasDraws: NaN,
      realPackDraws: Infinity,
      nonClearDraw: -3,
      timestampMs: NaN,
    }),
    false
  );

  const live1 = swapchain.recordPresent();
  assert.equal(live1.hasVisibleContent, false);
  assert.equal(live1.blackPresentsTotal, 1);
  assert.equal(live1.consecutiveBlackPresents, 1);
  assert.equal(live1.visiblePresentsTotal, 0);
  assert.equal(live1.firstVisiblePresentIndex, null);
  assert.equal(live1.firstVisiblePresentMs, null);

  // Zero-allocation reuse of live telemetry object on subsequent recordPresent calls.
  clockMs += 16;
  const live2 = swapchain.recordPresent({ realPackDraws: 3.9 });
  assert.strictEqual(live1, live2);
  assert.equal(live2.hasVisibleContent, true);
  assert.equal(live2.firstVisiblePresentIndex, 2);
  assert.equal(live2.firstVisiblePresentMs, 516);
  assert.equal(firedCount, 1);

  // No-arg present() is a safe no-op returning false.
  assert.equal(swapchain.present(), false);
  assert.equal(swapchain.presentCount, 2);

  // When device is lost, present() returns false without recording a present.
  device.triggerLost({ reason: 'unknown', message: 'lost' });
  await device.lost;
  await Promise.resolve();
  assert.equal(swapchain.present({ hasVisibleContent: true }), false);
  assert.equal(swapchain.presentCount, 2);

  // After destroy(), markFrameContent() and present() are safe no-ops and getStats() stays readable.
  swapchain.destroy();
  assert.equal(swapchain.markFrameContent({ hasDraws: true }), false);
  assert.equal(swapchain.present({ hasVisibleContent: true }), false);
  assert.equal(swapchain.getStats().hasVisibleContent, true);
  assert.equal(swapchain.getStats().firstVisiblePresentIndex, 2);
});

// ---------------------------------------------------------------------------
// W286 (#1177): Zero-alloc present blit descriptor reuse, (view, sampler) bind
// group caching/invalidation, and 30 FPS compliance/jitter/stall telemetry
// ---------------------------------------------------------------------------

test('W286 reuses pre-allocated blit pass & bind-group descriptors with zero per-frame descriptor allocations in steady-state present()', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  let clockMs = 1000;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: false,
    nowFn: () => clockMs,
  });
  swapchain.configure({ width: 1280, height: 720 });
  swapchain.configureForStreamFormat('rgb10a2unorm');

  const encoder = createMockCommandEncoder();
  const view1 = { kind: 'GPUTextureView', id: 'canvas-view-1' };
  const view2 = { kind: 'GPUTextureView', id: 'canvas-view-2' };

  assert.equal(swapchain.present(encoder, view1), true);
  const firstPassDesc = encoder.passes[0].desc;
  const firstAttachments = firstPassDesc.colorAttachments;
  const firstColorAttachment = firstAttachments[0];
  const firstClearValue = firstColorAttachment.clearValue;
  const firstBindGroup = encoder.passes[0].bindGroups[0].bindGroup;
  const firstBindGroupDesc = device.createdBindGroups[0].desc;

  assert.strictEqual(firstColorAttachment.view, view1);
  assert.equal(device.createdBindGroups.length, 1);

  // Run 100 steady-state presents at 30 FPS (33ms): every pass reuses exact descriptor objects & bind group.
  for (let i = 0; i < 100; i += 1) {
    clockMs += 33;
    const frameView = i % 2 === 0 ? view2 : view1;
    assert.equal(swapchain.present(encoder, frameView), true);
    const pass = encoder.passes[encoder.passes.length - 1];
    assert.strictEqual(pass.desc, firstPassDesc);
    assert.strictEqual(pass.desc.colorAttachments, firstAttachments);
    assert.strictEqual(pass.desc.colorAttachments[0], firstColorAttachment);
    assert.strictEqual(pass.desc.colorAttachments[0].clearValue, firstClearValue);
    assert.strictEqual(pass.desc.colorAttachments[0].view, frameView);
    assert.strictEqual(pass.bindGroups[0].bindGroup, firstBindGroup);
  }

  assert.equal(device.createdBindGroups.length, 1);
  assert.equal(device.createdRenderPipelines.length, 1);
  assert.equal(swapchain.blitCount, 101);

  // Redundant resize/configure no-ops must preserve cached bind group with zero rebuilds.
  assert.equal(swapchain.resize(1280, 720, 1), false);
  swapchain.configure({ width: 1280, height: 720, format: 'rgb10a2unorm' });
  clockMs += 33;
  assert.equal(swapchain.present(encoder, view1), true);
  assert.equal(device.createdBindGroups.length, 1);

  // Real resize invalidates bind group, clears stale view ref, and reuses pre-allocated descriptor containers.
  assert.equal(swapchain.resize(1920, 1080, 1), true);
  assert.equal(firstColorAttachment.view, null);
  assert.equal(swapchain._blitBindGroup, null);
  assert.equal(swapchain._blitBindGroupCache.size, 0);

  clockMs += 33;
  assert.equal(swapchain.present(encoder, view2), true);
  assert.equal(device.createdBindGroups.length, 2);
  assert.strictEqual(device.createdBindGroups[1].desc, firstBindGroupDesc);
  assert.strictEqual(encoder.passes[encoder.passes.length - 1].desc, firstPassDesc);
  assert.strictEqual(firstColorAttachment.view, view2);
});

test('W286 keys cached blit bind groups by (sourceView, sampler) identity and invalidates cleanly on reconfigure, device-loss, and destroy()', async () => {
  const { canvas, context, device, events, makeDevice } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: false,
  });
  swapchain.configure({ width: 1280, height: 720 });
  swapchain.configureForStreamFormat('rgb10a2unorm');

  // Prime pipeline and primary bind group.
  const enc = createMockCommandEncoder();
  assert.equal(swapchain.present(enc), true);
  assert.equal(device.createdBindGroups.length, 1);

  const viewA = swapchain.intermediateTextureView;
  const sampler1 = swapchain._blitSampler;
  const viewB = { kind: 'GPUTextureView', id: 'alt-source-view-b' };
  const sampler2 = { kind: 'GPUSampler', id: 'alt-sampler-2' };

  // Alternating (sourceView, sampler) identities cache independently and hit without rebuilding.
  const bgA1 = swapchain._ensureBlitBindGroup(viewA, sampler1);
  const bgB1 = swapchain._ensureBlitBindGroup(viewB, sampler1);
  const bgA2 = swapchain._ensureBlitBindGroup(viewA, sampler2);
  assert.equal(device.createdBindGroups.length, 3);
  assert.notStrictEqual(bgA1, bgB1);
  assert.notStrictEqual(bgA1, bgA2);
  assert.notStrictEqual(bgB1, bgA2);

  // Returning to previously seen (view, sampler) pairs hits cache with zero createBindGroup calls.
  assert.strictEqual(swapchain._ensureBlitBindGroup(viewA, sampler1), bgA1);
  assert.strictEqual(swapchain._ensureBlitBindGroup(viewB, sampler1), bgB1);
  assert.strictEqual(swapchain._ensureBlitBindGroup(viewA, sampler2), bgA2);
  assert.equal(device.createdBindGroups.length, 3);

  // Reconfigure (alphaMode change without resize) cleanly invalidates bind-group cache and target view.
  swapchain.configure({ alphaMode: 'premultiplied' });
  assert.equal(swapchain._blitBindGroup, null);
  assert.equal(swapchain._blitBindGroupCache.size, 0);
  assert.equal(swapchain._blitColorAttachment.view, null);

  assert.equal(swapchain.present(enc), true);
  assert.equal(device.createdBindGroups.length, 4);

  // Device loss invalidates bind-group cache, pipeline, and stale view/descriptor references.
  device.triggerLost({ reason: 'unknown', message: 'simulated gpu reset' });
  await device.lost;
  await Promise.resolve();
  assert.equal(swapchain._blitBindGroup, null);
  assert.equal(swapchain._blitBindGroupCache.size, 0);
  assert.equal(swapchain._blitColorAttachment.view, null);
  assert.equal(swapchain._blitBindGroupEntry0.resource, null);
  assert.equal(swapchain._blitBindGroupEntry1.resource, null);

  // Recovery rebuilds cleanly on the new device while reusing pre-allocated pass descriptor.
  const passDescRef = swapchain._blitPassDescriptor;
  const newDevice = attachMockBlitCapabilities(makeDevice('device-1'), events);
  swapchain.recover(newDevice, 'rgba8unorm');
  assert.equal(swapchain.present(enc), true);
  assert.equal(newDevice.createdBindGroups.length, 1);
  assert.strictEqual(enc.passes[enc.passes.length - 1].desc, passDescRef);

  // destroy() clears all cached bind groups and descriptor resource/view references.
  swapchain.destroy();
  assert.equal(swapchain._blitBindGroup, null);
  assert.equal(swapchain._blitBindGroupCache.size, 0);
  assert.equal(swapchain._blitColorAttachment.view, null);
  assert.equal(swapchain._blitBindGroupEntry0.resource, null);
  assert.equal(swapchain._blitBindGroupEntry1.resource, null);
});

test('W286 tracks 30 FPS compliance rate, rolling frameJitterMs, and longestStallMs in getStats() with clean resetStats()', () => {
  assert.ok(Math.abs(TARGET_30FPS_BUDGET_MS - 1000 / 30) < 1e-12);

  const { canvas, context, device } = createMockWebGpuEnv();
  let clockMs = 1000;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    nowFn: () => clockMs,
    frameWindowSize: 4,
  });
  swapchain.configure({ width: 1280, height: 720 });

  // Initial state before any frames.
  let stats = swapchain.getStats();
  assert.equal(stats.fps30ComplianceRate, 0);
  assert.equal(stats.frameJitterMs, 0);
  assert.equal(stats.longestStallMs, 0);
  assert.equal(swapchain.fps30ComplianceRate, 0);
  assert.equal(swapchain.frameJitterMs, 0);
  assert.equal(swapchain.longestStallMs, 0);

  // Frame 1 establishes baseline timestamp (still no inter-frame interval).
  swapchain.markFrameContent({ hasDraws: true, realPackDraws: 3 });
  swapchain.recordFramePresented();
  stats = swapchain.getStats();
  assert.equal(stats.fps30ComplianceRate, 0);
  assert.equal(stats.frameJitterMs, 0);
  assert.equal(stats.longestStallMs, 0);
  assert.equal(stats.hasVisibleContent, true);

  // Steady 30 FPS boundary intervals (33.333ms and 1000/30ms): 100% compliant, zero stalls.
  clockMs += 33.333;
  swapchain.recordFramePresented();
  clockMs += 1000 / 30;
  swapchain.recordFramePresented();
  stats = swapchain.getStats();
  assert.equal(stats.fps30ComplianceRate, 1);
  assert.equal(stats.longestStallMs, 0);
  assert.ok(stats.frameJitterMs >= 0 && stats.frameJitterMs < 1e-3);

  // Reset and feed a known 4-frame window: [16, 33.333, 40, 80] -> 2/4 compliant = 0.5, longestStall = 80.
  swapchain.resetStats();
  stats = swapchain.getStats();
  assert.equal(stats.fps30ComplianceRate, 0);
  assert.equal(stats.frameJitterMs, 0);
  assert.equal(stats.longestStallMs, 0);
  assert.equal(stats.hasVisibleContent, false);

  const deltas = [16, 33.333, 40, 80];
  swapchain.recordFramePresented(clockMs);
  for (const dt of deltas) {
    clockMs += dt;
    swapchain.recordFramePresented(clockMs);
  }

  const expectedMean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  const expectedStdDev = Math.sqrt(
    deltas.reduce((acc, dt) => acc + (dt - expectedMean) ** 2, 0) / deltas.length
  );

  stats = swapchain.getStats();
  assert.equal(stats.fps30ComplianceRate, 0.5);
  assert.ok(Math.abs(stats.frameJitterMs - expectedStdDev) < 1e-9);
  assert.equal(stats.longestStallMs, 80);
  assert.equal(swapchain.fps30ComplianceRate, 0.5);
  assert.ok(Math.abs(swapchain.frameJitterMs - expectedStdDev) < 1e-9);
  assert.equal(swapchain.longestStallMs, 80);

  // Just-over-30fps-budget boundary (33.334ms) is non-compliant; a 120ms spike updates longestStallMs.
  clockMs += 33.334; // ring: [33.333, 40, 80, 33.334] -> 1/4 compliant = 0.25
  swapchain.recordFramePresented(clockMs);
  assert.equal(swapchain.getStats().fps30ComplianceRate, 0.25);
  assert.equal(swapchain.getStats().longestStallMs, 80);

  clockMs += 120; // ring: [40, 80, 33.334, 120] -> 0/4 compliant = 0, longestStall = 120
  swapchain.recordFramePresented(clockMs);
  stats = swapchain.getStats();
  assert.equal(stats.fps30ComplianceRate, 0);
  assert.equal(stats.longestStallMs, 120);

  // Rolling eviction: 4 steady 16ms frames recover compliance to 1 and jitter to 0, while longestStallMs stays 120.
  for (let i = 0; i < 4; i += 1) {
    clockMs += 16;
    swapchain.recordFramePresented(clockMs);
  }
  stats = swapchain.getStats();
  assert.equal(stats.fps30ComplianceRate, 1);
  assert.equal(stats.frameJitterMs, 0);
  assert.equal(stats.maxFrameMs, 16);
  assert.equal(stats.longestStallMs, 120);

  // resetStats() clears longestStallMs, compliance, and jitter back to 0 cleanly.
  swapchain.resetStats();
  stats = swapchain.getStats();
  assert.equal(stats.fps30ComplianceRate, 0);
  assert.equal(stats.frameJitterMs, 0);
  assert.equal(stats.longestStallMs, 0);

  // New session after resetStats(): 2 steady 20ms intervals -> compliance 1, jitter 0, longestStall 0.
  clockMs += 100;
  swapchain.recordFramePresented(clockMs);
  clockMs += 20;
  swapchain.recordFramePresented(clockMs);
  clockMs += 20;
  swapchain.recordFramePresented(clockMs);
  stats = swapchain.getStats();
  assert.equal(stats.fps30ComplianceRate, 1);
  assert.equal(stats.frameJitterMs, 0);
  assert.equal(stats.longestStallMs, 0);
});

test('W348 preserves COPY_SRC usage across resize, configure, setFormat, configureForStreamFormat, and recover, and rolls back failed configure', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);

  const copySrcUsage = TEXTURE_USAGE_RENDER_ATTACHMENT | TEXTURE_USAGE_COPY_SRC;
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: false,
  });

  // Initial configure enables RENDER_ATTACHMENT | COPY_SRC.
  swapchain.configure({ width: 1280, height: 720, usage: copySrcUsage });
  assert.equal(swapchain.usage, copySrcUsage);
  assert.equal(swapchain.hasCopySrcUsage, true);
  assert.equal(context.configurations[0].usage, copySrcUsage);

  // Subsequent resize() and configure({ width, height }) without usage MUST preserve COPY_SRC.
  assert.equal(swapchain.resize(1920, 1080, 1), true);
  assert.equal(swapchain.usage, copySrcUsage);
  assert.equal(swapchain.hasCopySrcUsage, true);
  assert.equal(context.configurations[context.configurations.length - 1].usage, copySrcUsage);

  swapchain.configure({ width: 1280, height: 720 });
  assert.equal(swapchain.usage, copySrcUsage);
  assert.equal(context.configurations[context.configurations.length - 1].usage, copySrcUsage);

  // Format changes (direct and blit fallback) MUST preserve COPY_SRC.
  swapchain.setFormat('rgba8unorm');
  assert.equal(swapchain.usage, copySrcUsage);
  assert.equal(context.configurations[context.configurations.length - 1].usage, copySrcUsage);

  swapchain.configureForStreamFormat('rgb10a2unorm');
  assert.equal(swapchain.usesBlitFallback, true);
  assert.equal(swapchain.usage, copySrcUsage);

  swapchain.configureForStreamFormat('bgra8unorm');
  assert.equal(swapchain.usesBlitFallback, false);
  assert.equal(swapchain.usage, copySrcUsage);
  assert.equal(context.configurations[context.configurations.length - 1].usage, copySrcUsage);

  // Device recovery MUST preserve COPY_SRC.
  const env2 = createMockWebGpuEnv();
  swapchain.recover(env2.device);
  assert.equal(swapchain.usage, copySrcUsage);
  assert.equal(context.configurations[context.configurations.length - 1].usage, copySrcUsage);

  // A throwing configure({ usage }) must roll back stored usage rather than corrupting it.
  const origConfigure = context.configure.bind(context);
  context.configure = () => {
    throw new Error('simulated configure failure');
  };
  assert.throws(
    () => swapchain.configure({ usage: TEXTURE_USAGE_RENDER_ATTACHMENT }),
    /simulated configure failure/
  );
  assert.equal(swapchain.usage, copySrcUsage);
  assert.equal(swapchain.hasCopySrcUsage, true);
  context.configure = origConfigure;
  swapchain.destroy();
});

test('W348 confirmVisibleFrame transitions gated black presents to visible without double-counting presentCount', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  let clockMs = 1000;
  const fired = [];
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: false,
    nowFn: () => clockMs,
    onFirstVisibleFrame: (info) => fired.push({ ...info }),
  });
  swapchain.configure({ width: 1280, height: 720 });

  // 3 gated black presents: hasVisibleContent stays false, visiblePresentsTotal stays 0.
  for (let i = 0; i < 3; i += 1) {
    clockMs += 100;
    swapchain.markFrameContent({ hasVisibleContent: false, realPackDraws: 0, timestampMs: clockMs });
    swapchain.recordFramePresented(clockMs);
  }
  assert.equal(swapchain.presentCount, 3);
  assert.equal(swapchain.hasVisibleContent, false);
  assert.equal(swapchain.visiblePresentsTotal, 0);
  assert.equal(swapchain.blackPresentsTotal, 3);
  assert.equal(swapchain.firstVisiblePresentIndex, null);
  assert.equal(swapchain.firstVisiblePresentMs, null);
  assert.equal(fired.length, 0);

  // Async readback confirms non-black pixels on frame 3 at t=4200ms without incrementing presentCount.
  assert.equal(swapchain.confirmVisibleFrame({ timestampMs: 4200, realPackDraws: 5 }), true);
  assert.equal(swapchain.presentCount, 3);
  assert.equal(swapchain.hasVisibleContent, true);
  assert.equal(swapchain.visiblePresentsTotal, 1);
  assert.equal(swapchain.blackPresentsTotal, 2);
  assert.equal(swapchain.consecutiveBlackPresents, 0);
  assert.equal(swapchain.firstVisiblePresentIndex, 3);
  assert.equal(swapchain.firstVisiblePresentMs, 4200);
  assert.equal(fired.length, 1);
  assert.equal(fired[0].firstVisiblePresentIndex, 3);
  assert.equal(fired[0].firstVisiblePresentMs, 4200);
  assert.equal(fired[0].realPackDraws, 5);

  // Repeated confirmation is a safe one-shot no-op.
  assert.equal(swapchain.confirmVisibleFrame({ timestampMs: 9999, realPackDraws: 1 }), false);
  assert.equal(swapchain.presentCount, 3);
  assert.equal(swapchain.firstVisiblePresentMs, 4200);
  assert.equal(fired.length, 1);
  swapchain.destroy();
});

// ---------------------------------------------------------------------------
// W380 (#1763): HDR Tone-Map Blit Pipeline (Passthrough / Reinhard / ACES),
//               Preallocated Uniform Buffer & Zero-Alloc blitToCanvas
// ---------------------------------------------------------------------------

function attachMockToneMapBufferSupport(device, events) {
  let nextBufId = 1;
  device.createdBuffers = [];
  device.queueWrites = [];
  device.createBuffer = function createBuffer(desc) {
    const buf = {
      kind: 'GPUBuffer',
      id: nextBufId++,
      deviceLabel: device.label,
      desc: { ...desc },
      destroyed: false,
      destroyCalls: 0,
      destroy() {
        this.destroyed = true;
        this.destroyCalls += 1;
        events.push({ op: 'buffer.destroy', id: this.id });
      },
    };
    device.createdBuffers.push(buf);
    events.push({ op: 'device.createBuffer', id: buf.id, desc: buf.desc });
    return buf;
  };
  device.queue = {
    writeBuffer(buffer, bufferOffset, data) {
      const bytes =
        data instanceof ArrayBuffer
          ? new Uint8Array(data.slice(0))
          : new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
      device.queueWrites.push({ buffer, bufferOffset, bytes });
      events.push({ op: 'queue.writeBuffer', bufferId: buffer.id, byteLength: bytes.byteLength });
    },
  };
  return device;
}

test('W380 exposes BLIT_TONEMAP_WGSL, HDR stream format detection, and tone-map option validators', () => {
  assert.deepEqual(TONE_MAP_MODES, ['none', 'passthrough', 'reinhard', 'aces']);
  assert.equal(DEFAULT_TONE_MAP_MODE, 'none');
  assert.equal(DEFAULT_EXPOSURE, 1.0);
  assert.ok(typeof BLIT_TONEMAP_WGSL === 'string');
  assert.ok(BLIT_TONEMAP_WGSL.includes('vs_main'));
  assert.ok(BLIT_TONEMAP_WGSL.includes('fs_main'));
  assert.ok(BLIT_TONEMAP_WGSL.includes('tonemap_reinhard'));
  assert.ok(BLIT_TONEMAP_WGSL.includes('tonemap_aces'));
  assert.ok(BLIT_TONEMAP_WGSL.includes('linear_to_srgb'));

  for (const hdrFmt of ['rgba16float', 'rg11b10ufloat', 'r16float', 'rgba32float', 'rgb10a2unorm']) {
    assert.ok(HDR_STREAM_FORMATS.includes(hdrFmt));
    assert.equal(isHdrStreamFormat(hdrFmt), true);
  }
  assert.equal(isHdrStreamFormat('bgra8unorm'), false);
  assert.equal(isHdrStreamFormat('rgba8unorm-srgb'), false);
  assert.equal(isHdrStreamFormat(null), false);

  assert.equal(normalizeToneMapMode(undefined), 'none');
  assert.equal(normalizeToneMapMode('ACES'), 'aces');
  assert.equal(normalizeToneMapMode('reinhard'), 'reinhard');
  assert.equal(normalizeToneMapMode('passthrough'), 'passthrough');
  assert.throws(() => normalizeToneMapMode('filmic-unknown'), RangeError);
  assert.throws(() => normalizeToneMapMode(42), TypeError);

  assert.equal(normalizeExposure(undefined), 1.0);
  assert.equal(normalizeExposure(2.5), 2.5);
  assert.throws(() => normalizeExposure(0), RangeError);
  assert.throws(() => normalizeExposure(-1), RangeError);
  assert.throws(() => normalizeExposure(Number.NaN), RangeError);
  assert.throws(() => normalizeExposure(Number.POSITIVE_INFINITY), RangeError);
});

test('W380 ACES/Reinhard tone-map blit updates preallocated uniform buffer only on param change with zero per-frame allocs', () => {
  const { canvas, context, device, events } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);
  attachMockToneMapBufferSupport(device, events);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    streamFormat: 'rgba16float',
    toneMapMode: 'aces',
    exposure: 1.5,
    enableDepth: false,
  });
  swapchain.configure({ width: 1280, height: 720 });

  // HDR rgba16float stream with ACES tone-mapping onto SDR bgra8unorm keeps canvas SDR and activates blit fallback!
  assert.equal(swapchain.usesBlitFallback, true);
  assert.equal(swapchain.canvasFormat, 'bgra8unorm');
  assert.equal(swapchain.streamFormat, 'rgba16float');
  assert.equal(swapchain.isHdrStream, true);
  assert.equal(swapchain.toneMapMode, 'aces');
  assert.equal(swapchain.exposure, 1.5);
  assert.equal(swapchain.applySrgbEncode, true); // auto-enabled for linear HDR -> SDR non-sRGB canvas
  assert.equal(swapchain.toneMapActive, true);

  const encoder = createMockCommandEncoder();
  const canvasView = { kind: 'GPUTextureView', id: 'sdr-canvas-view' };

  // First present creates tone-map shader, pipeline, 16B uniform buffer, 3-entry bind group, and uploads uniform ONCE.
  assert.equal(swapchain.present(encoder, canvasView), true);
  assert.equal(device.createdBuffers.length, 1);
  assert.equal(device.createdBuffers[0].desc.size, 16);
  assert.equal(
    device.createdBuffers[0].desc.usage,
    BUFFER_USAGE_UNIFORM | BUFFER_USAGE_COPY_DST
  );
  assert.equal(device.createdShaderModules[0].desc.code, BLIT_TONEMAP_WGSL);
  assert.equal(device.createdBindGroups.length, 1);
  assert.equal(device.createdBindGroups[0].desc.entries.length, 3);
  assert.equal(device.createdBindGroups[0].desc.entries[2].resource.buffer, device.createdBuffers[0]);
  assert.equal(device.queueWrites.length, 1);

  // Verify exact uniform payload: mode=2 (aces), exposure=1.5f, apply_srgb=1.
  const u32_0 = new Uint32Array(device.queueWrites[0].bytes.buffer);
  const f32_0 = new Float32Array(device.queueWrites[0].bytes.buffer);
  assert.equal(u32_0[0], 2);
  assert.equal(f32_0[1], 1.5);
  assert.equal(u32_0[2], 1);

  // 50 steady-state frames: ZERO extra uniform uploads, ZERO extra bind groups, ZERO extra pipelines, exact descriptor reuse!
  const firstPassDesc = encoder.passes[0].desc;
  const firstBg = encoder.passes[0].bindGroups[0].bindGroup;
  for (let i = 0; i < 50; i += 1) {
    assert.equal(swapchain.present(encoder, canvasView), true);
    const p = encoder.passes[encoder.passes.length - 1];
    assert.strictEqual(p.desc, firstPassDesc);
    assert.strictEqual(p.bindGroups[0].bindGroup, firstBg);
  }
  assert.equal(device.queueWrites.length, 1);
  assert.equal(device.createdBindGroups.length, 1);
  assert.equal(device.createdRenderPipelines.length, 1);
  assert.equal(swapchain.toneMapBlitCount, 51);
  assert.equal(swapchain.toneMapUniformUploads, 1);

  // Changing tone-map options via setToneMapOptions marks dirty and triggers exactly 1 new upload on next blit.
  const updated = swapchain.setToneMapOptions({
    mode: 'reinhard',
    exposure: 2.0,
    applySrgbEncode: false,
  });
  assert.deepEqual(updated, {
    toneMapMode: 'reinhard',
    exposure: 2.0,
    applySrgbEncode: false,
    toneMapActive: true,
  });
  assert.equal(swapchain.present(encoder, canvasView), true);
  assert.equal(device.queueWrites.length, 2);
  assert.equal(device.createdBindGroups.length, 1); // same uniform buffer -> same bind group reused!

  const u32_1 = new Uint32Array(device.queueWrites[1].bytes.buffer);
  const f32_1 = new Float32Array(device.queueWrites[1].bytes.buffer);
  assert.equal(u32_1[0], 1); // reinhard
  assert.equal(f32_1[1], 2.0);
  assert.equal(u32_1[2], 0); // applySrgbEncode overridden to false

  // Calling setToneMapOptions with identical values does NOT re-upload.
  swapchain.setToneMapOptions({ mode: 'reinhard', exposure: 2.0, applySrgbEncode: false });
  assert.equal(swapchain.present(encoder, canvasView), true);
  assert.equal(device.queueWrites.length, 2);

  // Telemetry exposed cleanly in getStats() and reset cleanly in resetStats().
  const stats = swapchain.getStats();
  assert.equal(stats.toneMapMode, 'reinhard');
  assert.equal(stats.exposure, 2.0);
  assert.equal(stats.applySrgbEncode, false);
  assert.equal(stats.isHdrStream, true);
  assert.equal(stats.toneMapActive, true);
  assert.equal(stats.toneMapBlitCount, 53);
  assert.equal(stats.toneMapUniformUploads, 2);

  swapchain.resetStats();
  assert.equal(swapchain.getStats().toneMapBlitCount, 0);
  assert.equal(swapchain.getStats().toneMapUniformUploads, 0);

  // destroy() destroys the preallocated uniform buffer cleanly.
  const uniformBuf = device.createdBuffers[0];
  swapchain.destroy();
  assert.equal(uniformBuf.destroyed, true);
  assert.equal(uniformBuf.destroyCalls, 1);
});

test('W380 blitToCanvas blits arbitrary external HDR RT view onto SDR canvas and survives device.lost + recover()', async () => {
  const { canvas, context, device, events, makeDevice } = createMockWebGpuEnv();
  attachMockBlitCapabilities(device, events);
  attachMockToneMapBufferSupport(device, events);

  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: false,
    toneMapMode: 'aces',
  });
  swapchain.configure({ width: 1920, height: 1080 });
  assert.equal(swapchain.usesBlitFallback, false); // direct SDR swapchain

  const externalHdrView = { kind: 'GPUTextureView', id: 'rt-pool-hdr-rgba16f' };
  const canvasView = { kind: 'GPUTextureView', id: 'swapchain-bgra8' };
  const enc = createMockCommandEncoder();

  // Explicit blitToCanvas with sourceFormat='rgba16float' auto-enables sRGB encode & ACES tone-mapping.
  assert.equal(swapchain.blitToCanvas(enc, externalHdrView, canvasView, 'rgba16float'), true);
  assert.equal(swapchain.blitCount, 1);
  assert.equal(swapchain.toneMapBlitCount, 1);
  assert.equal(device.queueWrites.length, 1);
  const u32 = new Uint32Array(device.queueWrites[0].bytes.buffer);
  assert.equal(u32[0], 2); // aces
  assert.equal(u32[2], 1); // auto apply_srgb for rgba16float -> bgra8unorm

  // Switching to passthrough/none + exposure 1.0 uses zero-uniform passthrough blit for pre-encoded SDR (`rgba8unorm`),
  // while `rgba8unorm-srgb` (hardware sRGB->linear on sample) MUST engage exact sRGB re-encoding (apply_srgb = 1).
  swapchain.setToneMapOptions({ mode: 'passthrough', exposure: 1.0, applySrgbEncode: null });
  assert.equal(swapchain.toneMapActive, false);
  assert.equal(swapchain.blitToCanvas(enc, externalHdrView, canvasView, 'rgba8unorm'), true);
  assert.equal(swapchain.blitCount, 2);
  assert.equal(swapchain.toneMapBlitCount, 1);

  assert.equal(swapchain.blitToCanvas(enc, externalHdrView, canvasView, 'rgba8unorm-srgb'), true);
  assert.equal(swapchain.blitCount, 3);
  assert.equal(swapchain.toneMapBlitCount, 2);
  const u32Srgb = new Uint32Array(device.queueWrites[device.queueWrites.length - 1].bytes.buffer);
  assert.equal(u32Srgb[0], 0); // passthrough (no tonemap compression)
  assert.equal(u32Srgb[2], 1); // exact linear -> sRGB re-encode prevents ~x^4.84 double-gamma crush

  // Device loss destroys old uniform buffer; recover() rebuilds cleanly on new device.
  const oldBuf = device.createdBuffers[0];
  device.triggerLost({ reason: 'unknown', message: 'gpu reset' });
  await device.lost;
  await Promise.resolve();
  assert.equal(oldBuf.destroyed, true);
  assert.equal(swapchain.blitToCanvas(enc, externalHdrView, canvasView, 'rgba16float'), false);

  const newDevice = attachMockToneMapBufferSupport(
    attachMockBlitCapabilities(makeDevice('device-1'), events),
    events
  );
  swapchain.recover(newDevice, 'rgba8unorm');
  swapchain.setToneMapOptions({ mode: 'aces', exposure: 1.25 });
  assert.equal(swapchain.blitToCanvas(enc, externalHdrView, canvasView, 'rg11b10ufloat'), true);
  assert.equal(newDevice.createdBuffers.length, 1);
  assert.equal(newDevice.queueWrites.length, 1);

  // Validation errors on invalid setToneMapOptions inputs.
  assert.throws(() => swapchain.setToneMapOptions(null), TypeError);
  assert.throws(() => swapchain.setToneMapOptions({ mode: 'invalid' }), RangeError);
  assert.throws(() => swapchain.setToneMapOptions({ exposure: -2 }), RangeError);
  assert.throws(() => swapchain.setToneMapOptions({ applySrgbEncode: 'yes' }), TypeError);

  swapchain.destroy();
});

test('preserveCopySrcUntilVisible blocks ensureCopySrcUsage() after FIRST_VISIBLE_FRAME and drops COPY_SRC on resize', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    enableDepth: false,
    preserveCopySrcUntilVisible: true,
  });

  swapchain.configure({ width: 1280, height: 720 });
  assert.equal(swapchain.hasCopySrcUsage, true);
  assert.equal(swapchain.ensureCopySrcUsage(), true);

  // Confirm first visible frame -> ensureCopySrcUsage must no longer re-enable COPY_SRC.
  swapchain.confirmVisibleFrame({ timestampMs: 1234, realPackDraws: 1 });
  assert.equal(swapchain.hasVisibleContent, true);
  assert.equal(swapchain.getStats().hasVisibleContent, true);
  assert.equal(swapchain.ensureCopySrcUsage(), false);

  // Subsequent resize drops COPY_SRC back to RENDER_ATTACHMENT (direct scanout fast path).
  const configuresBefore = context.configurations.length;
  swapchain.resize(1920, 1080);
  assert.equal(context.configurations.length, configuresBefore + 1);
  assert.equal(swapchain.hasCopySrcUsage, false);
  assert.equal(swapchain.ensureCopySrcUsage(), false);
  assert.equal(
    context.configurations.length,
    configuresBefore + 1,
    'ensureCopySrcUsage after FIRST_VISIBLE_FRAME must not trigger canvas reconfigure'
  );

  swapchain.destroy();
});

test('sRGB stream formats (bgra8unorm-srgb, rgba8unorm-srgb) apply exact single IEC 61966-2-1 linear->sRGB encode on SDR canvas without double-gamma crush', () => {
  assert.equal(isSrgbFormat('bgra8unorm-srgb'), true);
  assert.equal(isSrgbFormat('rgba8unorm-srgb'), true);
  assert.equal(isSrgbFormat('BGRA8UNORM-SRGB'), true);
  assert.equal(isSrgbFormat('bgra8unorm'), false);
  assert.equal(isSrgbFormat('rgba8unorm'), false);
  assert.equal(isSrgbFormat('rgba16float'), false);
  assert.equal(isSrgbFormat(null), false);

  assert.ok(typeof BLIT_SRGB_ENCODE_WGSL === 'string');
  assert.ok(BLIT_SRGB_ENCODE_WGSL.includes('linear_to_srgb_channel'));
  assert.ok(BLIT_SRGB_ENCODE_WGSL.includes('0.0031308'));
  assert.ok(BLIT_SRGB_ENCODE_WGSL.includes('12.92'));
  assert.ok(BLIT_SRGB_ENCODE_WGSL.includes('1.055'));
  assert.ok(BLIT_SRGB_ENCODE_WGSL.includes('1.0 / 2.4'));
  assert.ok(BLIT_SRGB_ENCODE_WGSL.includes('0.055'));

  // 1) Real WebGPU / buffer-capable device: uses BLIT_TONEMAP_WGSL with mode=0, exposure=1.0, apply_srgb=1.
  for (const srgbStreamFormat of ['bgra8unorm-srgb', 'rgba8unorm-srgb']) {
    for (const sdrCanvasFormat of ['bgra8unorm', 'rgba8unorm']) {
      const { canvas, context, device, events } = createMockWebGpuEnv();
      attachMockBlitCapabilities(device, events);
      attachMockToneMapBufferSupport(device, events);

      const swapchain = new WgpuSwapchain({
        canvas,
        device,
        context,
        format: sdrCanvasFormat,
        enableDepth: false,
      });
      swapchain.configure({ width: 1280, height: 720 });
      assert.equal(swapchain.configureForStreamFormat(srgbStreamFormat), true);

      assert.equal(swapchain.usesBlitFallback, true);
      assert.equal(swapchain.streamFormat, srgbStreamFormat);
      assert.equal(swapchain.canvasFormat, sdrCanvasFormat);
      assert.equal(swapchain.applySrgbEncode, true);
      assert.equal(swapchain.toneMapActive, true);
      assert.equal(swapchain.getStats().applySrgbEncode, true);
      assert.equal(swapchain.getStats().toneMapActive, true);

      const encoder = createMockCommandEncoder();
      const canvasView = { kind: 'GPUTextureView', id: `canvas-${sdrCanvasFormat}` };
      assert.equal(swapchain.present(encoder, canvasView), true);

      assert.equal(swapchain.blitCount, 1);
      assert.equal(swapchain.toneMapBlitCount, 1);
      assert.equal(device.createdShaderModules.length, 1);
      assert.equal(device.createdShaderModules[0].desc.code, BLIT_TONEMAP_WGSL);
      assert.equal(device.queueWrites.length, 1);

      const u32 = new Uint32Array(device.queueWrites[0].bytes.buffer);
      const f32 = new Float32Array(device.queueWrites[0].bytes.buffer);
      assert.equal(u32[0], 0); // mode: none (no tonemap distortion on SDR sRGB)
      assert.equal(f32[1], 1.0); // exposure: 1.0
      assert.equal(u32[2], 1); // apply_srgb: 1 (exact IEC 61966-2-1 re-encode after hardware sRGB decode)

      swapchain.destroy();
    }
  }

  // 2) Buffer-less minimal device fallback: uses zero-uniform 2-binding BLIT_SRGB_ENCODE_WGSL (never raw BLIT_FULLSCREEN_WGSL).
  {
    const { canvas, context, device, events } = createMockWebGpuEnv();
    attachMockBlitCapabilities(device, events);

    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      enableDepth: false,
    });
    swapchain.configure({ width: 1280, height: 720 });
    swapchain.configureForStreamFormat('bgra8unorm-srgb');

    assert.equal(swapchain.applySrgbEncode, true);
    assert.equal(swapchain.toneMapActive, false); // no createBuffer on mock device

    const encoder = createMockCommandEncoder();
    assert.equal(swapchain.present(encoder), true);
    assert.equal(device.createdShaderModules.length, 1);
    assert.equal(device.createdShaderModules[0].desc.code, BLIT_SRGB_ENCODE_WGSL);
    assert.equal(device.createdBindGroups.length, 1);
    assert.equal(device.createdBindGroups[0].desc.entries.length, 2);
    swapchain.destroy();
  }

  // 3) Numerical verification: IEC 61966-2-1 sRGB decode -> linear_to_srgb round-trips every 8-bit value
  //    bit-for-bit (0..255), keeping midtone 0.5 linear at 188/255 (not crushed 128/255) and dark grays non-zero.
  const srgbToLinear = (s) => (s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4));
  const linearToSrgb = (v) => {
    const c = Math.min(1.0, Math.max(0.0, v));
    return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
  };
  for (let byte = 0; byte <= 255; byte += 1) {
    const linear = srgbToLinear(byte / 255);
    const reencodedByte = Math.round(linearToSrgb(linear) * 255);
    assert.equal(reencodedByte, byte, `sRGB byte ${byte} must round-trip without crush or drift`);
  }
  assert.equal(Math.round(linearToSrgb(0.5) * 255), 188);
  assert.ok(Math.round(linearToSrgb(srgbToLinear(10 / 255)) * 255) === 10);
});

test('HDR linear stream formats (rgba16float, rgb10a2unorm, rg11b10ufloat) activate auto sRGB encode + tonemapping cleanly without dead code', () => {
  // 1) rgb10a2unorm & rg11b10ufloat with default toneMapMode='none' and exposure=1.0 MUST activate
  //    _shouldUseToneMapPipeline via _resolveEffectiveApplySrgbEncode(sourceFormat) -> apply_srgb = 1.
  for (const hdrFormat of ['rgb10a2unorm', 'rg11b10ufloat']) {
    const { canvas, context, device, events } = createMockWebGpuEnv();
    attachMockBlitCapabilities(device, events);
    attachMockToneMapBufferSupport(device, events);

    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      enableDepth: false,
    });
    swapchain.configure({ width: 1920, height: 1080 });
    swapchain.configureForStreamFormat(hdrFormat);

    assert.equal(swapchain.usesBlitFallback, true);
    assert.equal(swapchain.applySrgbEncode, true);
    assert.equal(swapchain.toneMapActive, true);

    const encoder = createMockCommandEncoder();
    assert.equal(swapchain.present(encoder), true);
    assert.equal(swapchain.toneMapBlitCount, 1);
    assert.equal(device.createdShaderModules[0].desc.code, BLIT_TONEMAP_WGSL);
    const u32 = new Uint32Array(device.queueWrites[0].bytes.buffer);
    assert.equal(u32[0], 0); // mode: none
    assert.equal(u32[2], 1); // apply_srgb: 1
    swapchain.destroy();
  }

  // 2) rgba16float via configure({ streamFormat: 'rgba16float' }) with ACES / Reinhard / custom exposure / explicit applySrgbEncode
  //    keeps SDR canvas and runs tonemap + single sRGB encode; default toneMapMode='none' keeps direct rgba16float canvas.
  {
    const { canvas, context, device, events } = createMockWebGpuEnv();
    attachMockBlitCapabilities(device, events);
    attachMockToneMapBufferSupport(device, events);

    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      toneMapMode: 'aces',
      enableDepth: false,
    });
    swapchain.configure({ width: 1280, height: 720, streamFormat: 'rgba16float' });
    assert.equal(swapchain.usesBlitFallback, true);
    assert.equal(swapchain.canvasFormat, 'bgra8unorm');
    assert.equal(swapchain.streamFormat, 'rgba16float');
    assert.equal(swapchain.applySrgbEncode, true);
    assert.equal(swapchain.toneMapActive, true);

    const encoder = createMockCommandEncoder();
    assert.equal(swapchain.present(encoder), true);
    const u32 = new Uint32Array(device.queueWrites[0].bytes.buffer);
    assert.equal(u32[0], 2); // aces
    assert.equal(u32[2], 1); // apply_srgb: 1
    swapchain.destroy();
  }
});

test('pre-encoded SDR passthrough (bgra8unorm, rgba8unorm) and HDR rgba16float canvas targets are NEVER double-encoded', () => {
  // 1) Pre-encoded SDR sources onto SDR canvas: applySrgbEncode=false, uses raw BLIT_FULLSCREEN_WGSL (0x extra encode).
  for (const sdrSource of ['bgra8unorm', 'rgba8unorm']) {
    const { canvas, context, device, events } = createMockWebGpuEnv();
    attachMockBlitCapabilities(device, events);
    attachMockToneMapBufferSupport(device, events);

    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      enableDepth: false,
    });
    swapchain.configure({ width: 1280, height: 720 });
    assert.equal(swapchain.applySrgbEncode, false);
    assert.equal(swapchain.toneMapActive, false);

    const encoder = createMockCommandEncoder();
    const externalSdrView = { kind: 'GPUTextureView', id: `sdr-${sdrSource}` };
    const canvasView = { kind: 'GPUTextureView', id: 'canvas-bgra8' };
    assert.equal(swapchain.blitToCanvas(encoder, externalSdrView, canvasView, sdrSource), true);

    assert.equal(swapchain.blitCount, 1);
    assert.equal(swapchain.toneMapBlitCount, 0);
    assert.equal(device.createdBuffers.length, 0);
    assert.equal(device.createdShaderModules.length, 1);
    assert.equal(device.createdShaderModules[0].desc.code, BLIT_FULLSCREEN_WGSL);
    swapchain.destroy();
  }

  // 2) HDR rgba16float canvas target: NEVER sRGB-encodes linear/sRGB sources (compositor expects linear float).
  {
    const { canvas, context, device, events } = createMockWebGpuEnv();
    attachMockBlitCapabilities(device, events);
    attachMockToneMapBufferSupport(device, events);

    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'rgba16float',
      enableDepth: false,
    });
    swapchain.configure({ width: 1280, height: 720 });
    assert.equal(swapchain.canvasFormat, 'rgba16float');
    assert.equal(swapchain.applySrgbEncode, false);
    assert.equal(swapchain.toneMapActive, false);

    const encoder = createMockCommandEncoder();
    const srcView = { kind: 'GPUTextureView', id: 'srgb-src' };
    const dstView = { kind: 'GPUTextureView', id: 'hdr-canvas-view' };
    for (const srcFmt of ['bgra8unorm-srgb', 'rgba8unorm-srgb', 'rgb10a2unorm', 'rgba16float']) {
      assert.equal(swapchain.blitToCanvas(encoder, srcView, dstView, srcFmt), true);
    }
    assert.equal(swapchain.blitCount, 4);
    assert.equal(swapchain.toneMapBlitCount, 0);
    assert.equal(device.createdBuffers.length, 0);
    assert.equal(device.createdShaderModules[0].desc.code, BLIT_FULLSCREEN_WGSL);
    swapchain.destroy();
  }
});

// ---------------------------------------------------------------------------
// Issue #2161: Display-native physical pixel clamping (full native sharpness,
//              zero wasted unseen 4K backing buffers on 1080p / 1680x1050)
// ---------------------------------------------------------------------------

test('display-native clamping caps 4K requests cleanly to 1080p and aspect-preserving 1680x945 on 1680x1050 monitors', () => {
  // 1) 4K (3840x2160) request on a 1080p (1920x1080 @ 1.0 DPR) monitor -> 1920x1080
  {
    const { canvas, context, device } = createMockWebGpuEnv();
    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      maxDisplayWidth: 1920,
      maxDisplayHeight: 1080,
    });

    const cfg = swapchain.configure({ width: 3840, height: 2160, devicePixelRatio: 1 });
    assert.deepEqual(cfg, {
      width: 1920,
      height: 1080,
      format: 'bgra8unorm',
      depthFormat: 'depth24plus',
      devicePixelRatio: 1,
    });
    assert.equal(canvas.width, 1920);
    assert.equal(canvas.height, 1080);
    assert.deepEqual(device.createdTextures[0].desc.size, {
      width: 1920,
      height: 1080,
      depthOrArrayLayers: 1,
    });

    const stats = swapchain.getStats();
    assert.equal(stats.clampedToDisplayCount, 1);
    assert.equal(stats.requestedWidth, 3840);
    assert.equal(stats.requestedHeight, 2160);
    assert.equal(stats.backingWidth, 1920);
    assert.equal(stats.backingHeight, 1080);
    assert.deepEqual(stats.displayBounds, {
      maxDisplayWidth: 1920,
      maxDisplayHeight: 1080,
      screenWidth: 1920,
      screenHeight: 1080,
      devicePixelRatio: 1,
    });

    // Repeated resize with same 4K request clamps to identical 1920x1080 -> strict no-op!
    assert.equal(swapchain.resize(3840, 2160, 1), false);
    assert.equal(context.configurations.length, 1);
    assert.equal(device.createdTextures.length, 1);
    assert.equal(swapchain.clampedToDisplayCount, 1);
    assert.equal(swapchain.getTelemetry().skippedRedundantReconfigures, 1);

    swapchain.destroy();
  }

  // 2) 4K 16:9 (3840x2160) request on a 1680x1050 (16:10) monitor -> aspect-preserving 1680x945
  {
    const { canvas, context, device } = createMockWebGpuEnv();
    const swapchain = new WgpuSwapchain({
      canvas,
      device,
      context,
      format: 'bgra8unorm',
      screenWidth: 1680,
      screenHeight: 1050,
      dpr: 1,
    });

    swapchain.configure({ width: 3840, height: 2160, devicePixelRatio: 1 });
    assert.equal(canvas.width, 1680);
    assert.equal(canvas.height, 945);
    assert.equal(swapchain.backingWidth, 1680);
    assert.equal(swapchain.backingHeight, 945);
    assert.equal(swapchain.requestedWidth, 3840);
    assert.equal(swapchain.requestedHeight, 2160);
    assert.deepEqual(device.createdTextures[0].desc.size, {
      width: 1680,
      height: 945,
      depthOrArrayLayers: 1,
    });
    assert.equal(swapchain.getStats().clampedToDisplayCount, 1);
    swapchain.destroy();
  }
});

test('display-native clamping never upscales small canvases and never downscales below native display pixels', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    displayBounds: { maxDisplayWidth: 3840, maxDisplayHeight: 2160 },
  });

  // 1080p request on a 4K monitor -> untouched 1920x1080 (never upscaled, never clamped)
  swapchain.configure({ width: 1920, height: 1080, devicePixelRatio: 1 });
  assert.equal(canvas.width, 1920);
  assert.equal(canvas.height, 1080);
  let stats = swapchain.getStats();
  assert.equal(stats.requestedWidth, 1920);
  assert.equal(stats.requestedHeight, 1080);
  assert.equal(stats.backingWidth, 1920);
  assert.equal(stats.backingHeight, 1080);
  assert.equal(stats.clampedToDisplayCount, 0);

  // Small 800x600 canvas -> untouched 800x600
  assert.equal(swapchain.resize(800, 600, 1), true);
  assert.equal(canvas.width, 800);
  assert.equal(canvas.height, 600);
  assert.equal(swapchain.clampedToDisplayCount, 0);

  // Exact native 4K request on 4K monitor -> untouched full 3840x2160
  assert.equal(swapchain.resize(3840, 2160, 1), true);
  assert.equal(canvas.width, 3840);
  assert.equal(canvas.height, 2160);
  stats = swapchain.getStats();
  assert.equal(stats.requestedWidth, 3840);
  assert.equal(stats.requestedHeight, 2160);
  assert.equal(stats.backingWidth, 3840);
  assert.equal(stats.backingHeight, 2160);
  assert.equal(stats.clampedToDisplayCount, 0);

  swapchain.destroy();
});

test('display-native clamping handles HiDPI (1280x720 @ 1.5 DPR = 1920x1080) and PerfProfile computeDisplayNativeBounds integration', () => {
  const { computeDisplayNativeBounds } = require('../web/workers/perf_profile.js');
  const nativeBounds = computeDisplayNativeBounds({
    screenWidth: 1280,
    screenHeight: 720,
    devicePixelRatio: 1.5,
    maxTextureDimension2D: 8192,
  });
  assert.equal(nativeBounds.maxDisplayWidth, 1920);
  assert.equal(nativeBounds.maxDisplayHeight, 1080);

  // Passing computeDisplayNativeBounds directly must NOT double-multiply by DPR!
  const { canvas, context, device } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
    displayBounds: nativeBounds,
  });

  // Native 1280x720 @ 1.5 DPR request -> exact 1920x1080, unclamped
  swapchain.configure({ width: 1280, height: 720, devicePixelRatio: 1.5 });
  assert.equal(canvas.width, 1920);
  assert.equal(canvas.height, 1080);
  assert.equal(swapchain.getStats().clampedToDisplayCount, 0);

  // Oversized 4K request (2560x1440 @ 1.5 DPR = 3840x2160) -> clamped to exact 1920x1080
  // Since backing size stays 1920x1080, resize() elides redundant GPU reconfigure!
  assert.equal(swapchain.resize(2560, 1440, 1.5), false);
  assert.equal(canvas.width, 1920);
  assert.equal(canvas.height, 1080);
  assert.equal(swapchain.requestedWidth, 3840);
  assert.equal(swapchain.requestedHeight, 2160);

  // Starting fresh with screenWidth/screenHeight/dpr constructor shorthand on 1280x720 @ 1.5 DPR
  const env2 = createMockWebGpuEnv();
  const sc2 = new WgpuSwapchain({
    canvas: env2.canvas,
    device: env2.device,
    context: env2.context,
    format: 'bgra8unorm',
    screenWidth: 1280,
    screenHeight: 720,
    dpr: 1.5,
  });
  sc2.configure({ width: 3840, height: 2160, devicePixelRatio: 1 });
  assert.equal(env2.canvas.width, 1920);
  assert.equal(env2.canvas.height, 1080);
  assert.equal(sc2.getStats().clampedToDisplayCount, 1);
  assert.deepEqual(sc2.displayBounds, {
    maxDisplayWidth: 1920,
    maxDisplayHeight: 1080,
    screenWidth: 1280,
    screenHeight: 720,
    devicePixelRatio: 1.5,
  });

  swapchain.destroy();
  sc2.destroy();
});

test('setDisplayBounds dynamically updates backing resolution across monitor changes and supports clearing/recovery', async () => {
  const { canvas, context, device, makeDevice } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
  });

  // Start on 4K monitor requesting 3840x2160 -> full 3840x2160
  assert.equal(swapchain.setDisplayBounds(3840, 2160, 1), false); // unconfigured yet
  swapchain.configure({ width: 3840, height: 2160, devicePixelRatio: 1 });
  assert.equal(canvas.width, 3840);
  assert.equal(canvas.height, 2160);
  assert.equal(swapchain.clampedToDisplayCount, 0);

  const depthTex4k = device.createdTextures[0];

  // Drag window to 1680x1050 @ 1.0 DPR monitor -> clamps immediately to 1680x945
  assert.equal(swapchain.setDisplayBounds(1680, 1050, 1), true);
  assert.equal(canvas.width, 1680);
  assert.equal(canvas.height, 945);
  assert.equal(depthTex4k.destroyed, true);
  assert.deepEqual(device.createdTextures[1].desc.size, {
    width: 1680,
    height: 945,
    depthOrArrayLayers: 1,
  });
  assert.equal(swapchain.requestedWidth, 3840);
  assert.equal(swapchain.requestedHeight, 2160);
  assert.equal(swapchain.clampedToDisplayCount, 1);
  assert.equal(swapchain.getTelemetry().resizeCount, 1);

  // Calling setDisplayBounds with equivalent HiDPI bounds yielding same backing size -> no-op
  const configCountBefore = context.configurations.length;
  assert.equal(swapchain.setDisplayBounds(840, 525, 2), false);
  assert.equal(context.configurations.length, configCountBefore);

  // Drag window onto a 1080p HiDPI laptop (1280x720 @ 1.5 DPR -> 1920x1080) -> re-clamps to 1920x1080
  assert.equal(swapchain.setDisplayBounds(1280, 720, 1.5), true);
  assert.equal(canvas.width, 1920);
  assert.equal(canvas.height, 1080);
  assert.equal(swapchain.clampedToDisplayCount, 2);
  assert.equal(swapchain.getTelemetry().resizeCount, 2);

  // Drag window back to 4K monitor -> restores full 3840x2160 automatically from remembered requested dims!
  assert.equal(swapchain.setDisplayBounds(3840, 2160, 1), true);
  assert.equal(canvas.width, 3840);
  assert.equal(canvas.height, 2160);
  assert.equal(swapchain.clampedToDisplayCount, 2); // unclamped restore

  // Re-clamp to 1080p, verify resetStats() resets clampedToDisplayCount while keeping displayBounds,
  // and verify recover() preserves display-native clamping on new GPUDevice.
  assert.equal(swapchain.setDisplayBounds(1920, 1080, 1), true);
  assert.equal(canvas.width, 1920);
  assert.equal(canvas.height, 1080);
  assert.equal(swapchain.clampedToDisplayCount, 3);

  swapchain.resetStats();
  assert.equal(swapchain.getStats().clampedToDisplayCount, 0);
  assert.equal(swapchain.displayBounds.maxDisplayWidth, 1920);
  assert.equal(swapchain.backingWidth, 1920);
  assert.equal(swapchain.requestedWidth, 3840);

  device.triggerLost({ reason: 'unknown', message: 'reset' });
  await device.lost;
  await Promise.resolve();

  const newDevice = makeDevice('device-1');
  swapchain.recover(newDevice);
  assert.equal(canvas.width, 1920);
  assert.equal(canvas.height, 1080);
  assert.deepEqual(newDevice.createdTextures[0].desc.size, {
    width: 1920,
    height: 1080,
    depthOrArrayLayers: 1,
  });

  // Clearing displayBounds with null restores full requested 3840x2160
  assert.equal(swapchain.setDisplayBounds(null), true);
  assert.equal(swapchain.displayBounds, null);
  assert.equal(swapchain.getStats().displayBounds, null);
  assert.equal(canvas.width, 3840);
  assert.equal(canvas.height, 2160);

  swapchain.destroy();
});

test('display-native clamping preserves full backward compatibility when unset and validates invalid inputs', () => {
  const { canvas, context, device } = createMockWebGpuEnv();
  const swapchain = new WgpuSwapchain({
    canvas,
    device,
    context,
    format: 'bgra8unorm',
  });

  assert.equal(swapchain.displayBounds, null);
  swapchain.configure({ width: 3840, height: 2160, devicePixelRatio: 1 });
  assert.equal(canvas.width, 3840);
  assert.equal(canvas.height, 2160);

  const stats = swapchain.getStats();
  assert.equal(stats.displayBounds, null);
  assert.equal(stats.clampedToDisplayCount, 0);
  assert.equal(stats.requestedWidth, 3840);
  assert.equal(stats.requestedHeight, 2160);
  assert.equal(stats.backingWidth, 3840);
  assert.equal(stats.backingHeight, 2160);

  // Pure helper edge checks
  assert.deepEqual(clampToDisplayBounds(3840, 2160, null), {
    width: 3840,
    height: 2160,
    clamped: false,
  });
  assert.deepEqual(
    clampToDisplayBounds(3840, 2160, { maxDisplayWidth: 1680, maxDisplayHeight: 1050 }),
    { width: 1680, height: 945, clamped: true }
  );
  assert.deepEqual(
    clampToDisplayBounds(2560, 2560, { maxDisplayWidth: 1920, maxDisplayHeight: 1080 }),
    { width: 1080, height: 1080, clamped: true }
  );

  // Validation errors on malformed bounds
  assert.throws(() => normalizeDisplayBounds('1920x1080'), TypeError);
  assert.throws(() => normalizeDisplayBounds({}), RangeError);
  assert.throws(() => normalizeDisplayBounds({ maxDisplayWidth: -1, maxDisplayHeight: 1080 }), RangeError);
  assert.throws(() => swapchain.setDisplayBounds(0, 1080, 1), RangeError);
  assert.throws(() => swapchain.setDisplayBounds(1920, 1080, -1), RangeError);

  swapchain.destroy();
  assert.throws(() => swapchain.setDisplayBounds(1920, 1080, 1), /destroyed/i);
});
