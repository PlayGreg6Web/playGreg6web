'use strict';

// Dedicated OffscreenCanvas WebGPU worker consuming binary GpuPacketHeader streams.
// Opcode contract: docs/gpu_commands.md & runtime/abi/playgta6_abi.h.
const OPCODES = Object.freeze({
  NOP: 0,
  BEGIN_FRAME: 1,
  CLEAR_COLOR: 2,
  CREATE_BUFFER: 3,
  WRITE_BUFFER: 4,
  CREATE_SHADER_MODULE: 5,
  DRAW: 6,
  PRESENT: 7,
  CREATE_RENDER_PIPELINE: 8,
  SET_PIPELINE: 9,
  SET_VERTEX_BUFFER: 10,
  DESTROY_RESOURCE: 11,
  CREATE_TEXTURE: 12,
  WRITE_TEXTURE: 13,
  CREATE_SAMPLER: 14,
  CREATE_BIND_GROUP: 15,
  SET_BIND_GROUP: 16,
  SET_INDEX_BUFFER: 17,
  DRAW_INDEXED: 18,
  CREATE_COMPUTE_PIPELINE: 19,
  SET_COMPUTE_PIPELINE: 20,
  DISPATCH_COMPUTE: 21,
  CREATE_BIND_GROUP_LAYOUT: 22,
  CREATE_PIPELINE_LAYOUT: 23,
  CREATE_RENDER_PIPELINE2: 24,
  CREATE_COMPUTE_PIPELINE2: 25,
  BEGIN_RENDER_PASS: 26,
  END_RENDER_PASS: 27,
  SET_VIEWPORT: 28,
  SET_SCISSOR_RECT: 29,
  SET_BLEND_CONSTANT: 30,
  SET_STENCIL_REFERENCE: 31,
  CREATE_BIND_GROUP2: 32,
  SET_BIND_GROUP2: 33,
  CREATE_SAMPLER2: 34,
  CREATE_TEXTURE2: 35,
});

const SHADER_STAGE = Object.freeze({
  VERTEX: 0,
  FRAGMENT: 1,
  COMPUTE: 2,
});

const PRIMITIVE_TOPOLOGY = Object.freeze({
  TRIANGLE_LIST: 0,
  TRIANGLE_STRIP: 1,
  LINE_LIST: 2,
  LINE_STRIP: 3,
  POINT_LIST: 4,
});

const TOPOLOGY_NAMES = Object.freeze([
  'triangle-list',
  'triangle-strip',
  'line-list',
  'line-strip',
  'point-list',
]);

const BUFFER_USAGE = Object.freeze({
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
});

const TEXTURE_USAGE = Object.freeze({
  COPY_SRC: 0x0001,
  COPY_DST: 0x0002,
  TEXTURE_BINDING: 0x0004,
  STORAGE_BINDING: 0x0008,
  RENDER_ATTACHMENT: 0x0010,
});

const TEXTURE_FORMAT = Object.freeze({
  RGBA8_UNORM: 0,
  BGRA8_UNORM: 1,
  RGBA16_FLOAT: 2,
  DEPTH24_PLUS: 3,
  DEPTH32_FLOAT: 4,
  BC7_RGBA_UNORM: 5,
});

const TEXTURE_FORMAT_NAMES = Object.freeze([
  'rgba8unorm',
  'bgra8unorm',
  'rgba16float',
  'depth24plus',
  'depth32float',
  'bc7-rgba-unorm',
]);

const TEXTURE_DIMENSION = Object.freeze({
  DIM_1D: 0,
  DIM_2D: 1,
  DIM_3D: 2,
});

const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);

const ADDRESS_MODE = Object.freeze({
  CLAMP_TO_EDGE: 0,
  REPEAT: 1,
  MIRROR_REPEAT: 2,
});

const ADDRESS_MODE_NAMES = Object.freeze(['clamp-to-edge', 'repeat', 'mirror-repeat']);

const FILTER_MODE = Object.freeze({
  NEAREST: 0,
  LINEAR: 1,
});

const FILTER_MODE_NAMES = Object.freeze(['nearest', 'linear']);

const BIND_RESOURCE_TYPE = Object.freeze({
  BUFFER: 0,
  TEXTURE_VIEW: 1,
  SAMPLER: 2,
});

const INDEX_FORMAT = Object.freeze({
  UINT16: 0,
  UINT32: 1,
});

const INDEX_FORMAT_NAMES = Object.freeze(['uint16', 'uint32']);

const DEFAULT_CLEAR_COLOR = Object.freeze({ r: 0.05, g: 0.07, b: 0.11, a: 1.0 });

const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
const FNV1A_64_PRIME = 0x00000100000001b3n;
const U64_MASK = 0xffffffffffffffffn;

const TEXTURE_FORMAT_BPP = Object.freeze({
  rgba8unorm: 4,
  bgra8unorm: 4,
  rgba16float: 8,
  depth24plus: 4,
  depth32float: 4,
});

function fnv1a64Hex(text) {
  const bytes = new TextEncoder().encode(String(text));
  let hash = FNV1A_64_OFFSET_BASIS;
  for (let i = 0; i < bytes.length; i += 1) {
    hash ^= BigInt(bytes[i]);
    hash = (hash * FNV1A_64_PRIME) & U64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, '0')}`;
}

function estimateTextureBytes(width, height, depthOrArrayLayers, mipLevelCount, format, dimension) {
  const w0 = Math.max(1, width | 0);
  const h0 = Math.max(1, height | 0);
  const d0 = Math.max(1, depthOrArrayLayers | 0);
  const mips = Math.max(1, mipLevelCount | 0);
  const is1D = dimension === '1d';
  const is3D = dimension === '3d';
  const isBc7 = format === 'bc7-rgba-unorm';
  const bpp = TEXTURE_FORMAT_BPP[format] || 4;

  let total = 0;
  for (let mip = 0; mip < mips; mip += 1) {
    const scale = mip >= 31 ? Infinity : Math.pow(2, mip);
    const mw = scale === Infinity ? 1 : Math.max(1, Math.floor(w0 / scale));
    const mh = is1D ? 1 : scale === Infinity ? 1 : Math.max(1, Math.floor(h0 / scale));
    const md = is3D ? (scale === Infinity ? 1 : Math.max(1, Math.floor(d0 / scale))) : d0;
    if (isBc7) {
      total += Math.ceil(mw / 4) * Math.ceil(mh / 4) * 16 * md;
    } else {
      total += mw * mh * md * bpp;
    }
  }
  return total;
}

function extractWgslEntryPoint(code, stage) {
  if (typeof code !== 'string' || code.length === 0) {
    return null;
  }
  const attr =
    stage === SHADER_STAGE.VERTEX
      ? 'vertex'
      : stage === SHADER_STAGE.FRAGMENT
        ? 'fragment'
        : stage === SHADER_STAGE.COMPUTE
          ? 'compute'
          : null;
  if (attr) {
    const re = new RegExp(`@${attr}\\b[\\s\\S]*?\\bfn\\s+([A-Za-z_][A-Za-z0-9_]*)`);
    const m = re.exec(code);
    if (m) return m[1];
  }
  const fallback = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/.exec(code);
  return fallback ? fallback[1] : null;
}

function stageCodeToName(stage) {
  if (stage === SHADER_STAGE.VERTEX) return 'vertex';
  if (stage === SHADER_STAGE.FRAGMENT) return 'fragment';
  if (stage === SHADER_STAGE.COMPUTE) return 'compute';
  return null;
}

function resolveSubsystemCtor(injectedCtor, globalNamespace, exportName, scriptRelPath) {
  if (typeof injectedCtor === 'function') {
    return injectedCtor;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis[globalNamespace] &&
    typeof globalThis[globalNamespace][exportName] === 'function'
  ) {
    return globalThis[globalNamespace][exportName];
  }
  if (typeof importScripts === 'function') {
    try {
      importScripts(scriptRelPath);
      if (
        typeof globalThis !== 'undefined' &&
        globalThis[globalNamespace] &&
        typeof globalThis[globalNamespace][exportName] === 'function'
      ) {
        return globalThis[globalNamespace][exportName];
      }
    } catch (_) {
      // Optional helper script unavailable in this worker environment.
    }
  }
  if (typeof require === 'function') {
    try {
      const mod = require(scriptRelPath);
      if (mod && typeof mod[exportName] === 'function') {
        return mod[exportName];
      }
    } catch (_) {
      // Optional helper module unavailable.
    }
  }
  return null;
}

function deviceSupportsTimestampQuery(device) {
  if (!device || typeof device !== 'object') {
    return false;
  }
  const features = device.features;
  if (!features || typeof features.has !== 'function') {
    return false;
  }
  try {
    if (!features.has('timestamp-query')) {
      return false;
    }
  } catch (_) {
    return false;
  }
  return typeof device.createQuerySet === 'function' && typeof device.createBuffer === 'function';
}

function resolveGpuRingBuffer(injectedCtor) {
  if (typeof injectedCtor === 'function') {
    return injectedCtor;
  }
  if (
    typeof globalThis !== 'undefined' &&
    globalThis.PlayGta6GpuRing &&
    typeof globalThis.PlayGta6GpuRing.GpuRingBuffer === 'function'
  ) {
    return globalThis.PlayGta6GpuRing.GpuRingBuffer;
  }
  if (typeof importScripts === 'function') {
    importScripts('./gpu_ring.js');
    if (
      typeof globalThis !== 'undefined' &&
      globalThis.PlayGta6GpuRing &&
      typeof globalThis.PlayGta6GpuRing.GpuRingBuffer === 'function'
    ) {
      return globalThis.PlayGta6GpuRing.GpuRingBuffer;
    }
  }
  if (typeof require === 'function') {
    const mod = require('./gpu_ring.js');
    if (mod && typeof mod.GpuRingBuffer === 'function') {
      return mod.GpuRingBuffer;
    }
  }
  throw new Error('GpuRingBuffer unavailable; load ./gpu_ring.js or inject GpuRingBuffer');
}

class WgpuCommandProcessor {
  constructor(device, ctx, options = {}) {
    this.device = device;
    this.ctx = ctx;
    this.format = options.format || 'bgra8unorm';
    this.buffers = new Map();
    this.textures = new Map();
    this.textureViews = new Map();
    this.samplers = new Map();
    this.bindGroups = new Map();
    this._bindGroupKinds = new Map();
    this.shaderModules = new Map();
    this.pipelines = new Map();
    this.computePipelines = new Map();
    this._pipelineBindGroupLayouts = new Map();
    this._computePipelineBindGroupLayouts = new Map();
    this._shaderReflectionKeys = new Map();
    this.frameCount = 0;
    this.ring = null;
    this._GpuRingBuffer = typeof options.GpuRingBuffer === 'function' ? options.GpuRingBuffer : null;
    this._decoder = new TextDecoder('utf-8');
    this._destroyed = false;

    // Telemetry counters
    this._packetCount = 0;
    this._drawCallCount = 0;
    this._computeDispatchCount = 0;
    this._bytesProcessed = 0;

    // Pending staging upload encoder & frame lifecycle tracking
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._frameBegunForIndex = -1;
    this._lastPassTimings = null;
    this._pendingTimerReadbacks = new Set();

    // Optional WebGpuPipelineCache for reflection layout deduplication
    this.pipelineCache = options.pipelineCache || null;

    // Saved subsystem options / constructors for lazy construction where supported
    this._reflectionRegistryOptions = options.reflectionRegistryOptions || null;
    this._ReflectionRegistryCtor = options.ReflectionRegistry || null;
    this._rtPoolOptions = options.rtPoolOptions || null;
    this._RenderTargetPoolCtor = options.RenderTargetPool || null;
    this._rtPoolExplicitlyDisabled = options.enableRtPool === false;

    // 1. GpuStagingPool (G14)
    this._ownsStagingPool = false;
    this.stagingPool = null;
    if (options.stagingPool && typeof options.stagingPool === 'object') {
      this.stagingPool = options.stagingPool;
    } else if (
      options.enableStagingPool === true &&
      device &&
      typeof device.createBuffer === 'function'
    ) {
      const StagingCtor = resolveSubsystemCtor(
        options.GpuStagingPool,
        'PlayGta6StagingPool',
        'GpuStagingPool',
        './staging_pool.js'
      );
      if (StagingCtor) {
        this.stagingPool = new StagingCtor({
          ...(options.stagingPoolOptions || {}),
          device,
        });
        this._ownsStagingPool = true;
      }
    }

    // 2. DynamicUniformRing (G18)
    this._ownsUniformRing = false;
    this.uniformRing = null;
    if (options.uniformRing && typeof options.uniformRing === 'object') {
      this.uniformRing = options.uniformRing;
    } else if (
      options.enableUniformRing === true &&
      device &&
      typeof device.createBuffer === 'function'
    ) {
      const UniformRingCtor = resolveSubsystemCtor(
        options.DynamicUniformRing,
        'PlayGta6UniformRing',
        'DynamicUniformRing',
        './uniform_ring.js'
      );
      if (UniformRingCtor) {
        this.uniformRing = new UniformRingCtor({
          ...(options.uniformRingOptions || {}),
          device,
        });
        this._ownsUniformRing = true;
      }
    }

    // 3. GpuPassTimer (G16)
    this._ownsPassTimer = false;
    this.passTimer = null;
    if (options.passTimer && typeof options.passTimer === 'object') {
      this.passTimer = options.passTimer;
    } else if (
      options.enablePassTimer === true ||
      (options.enablePassTimer !== false && deviceSupportsTimestampQuery(device))
    ) {
      const PassTimerCtor = resolveSubsystemCtor(
        options.GpuPassTimer,
        'PlayGta6GpuTimer',
        'GpuPassTimer',
        './gpu_timer.js'
      );
      if (PassTimerCtor) {
        this.passTimer = new PassTimerCtor({
          ...(options.passTimerOptions || {}),
          device,
        });
        this._ownsPassTimer = true;
      }
    }

    // 4. ReflectionRegistry (G17)
    this._ownsReflectionRegistry = false;
    this.reflectionRegistry = null;
    if (options.reflectionRegistry && typeof options.reflectionRegistry === 'object') {
      this.reflectionRegistry = options.reflectionRegistry;
    } else if (options.enableReflectionRegistry === true) {
      const ReflectionCtor = resolveSubsystemCtor(
        options.ReflectionRegistry,
        'PlayGta6ReflectionLoader',
        'ReflectionRegistry',
        './reflection_loader.js'
      );
      if (ReflectionCtor) {
        this.reflectionRegistry = new ReflectionCtor(options.reflectionRegistryOptions || {});
        this._ownsReflectionRegistry = true;
      }
    }

    // 5. GpuResourceRegistry (G21)
    this._ownsResourceRegistry = false;
    this.resourceRegistry = null;
    if (options.resourceRegistry && typeof options.resourceRegistry === 'object') {
      this.resourceRegistry = options.resourceRegistry;
    } else if (options.enableResourceRegistry === true) {
      const RegistryCtor = resolveSubsystemCtor(
        options.GpuResourceRegistry,
        'PlayGta6ResourceRegistry',
        'GpuResourceRegistry',
        './resource_registry.js'
      );
      if (RegistryCtor) {
        this.resourceRegistry = new RegistryCtor(options.resourceRegistryOptions || {});
        this._ownsResourceRegistry = true;
      }
    }

    // 6. RenderTargetPool (G22)
    this._ownsRtPool = false;
    this.rtPool = null;
    if (options.rtPool && typeof options.rtPool === 'object') {
      this.rtPool = options.rtPool;
    } else if (
      options.enableRtPool === true &&
      device &&
      typeof device.createTexture === 'function'
    ) {
      const RtPoolCtor = resolveSubsystemCtor(
        options.RenderTargetPool,
        'PlayGta6RenderTargetPool',
        'RenderTargetPool',
        './rt_pool.js'
      );
      if (RtPoolCtor) {
        this.rtPool = new RtPoolCtor({
          ...(options.rtPoolOptions || {}),
          device,
        });
        this._ownsRtPool = true;
      }
    }

    if (options.reflectionManifest !== undefined && options.reflectionManifest !== null) {
      this.loadReflectionManifest(options.reflectionManifest);
    }

    // Packet ABI v2 handler ({ handlePacket, destroyResource }), see render_state_packets.js.
    this.extension = null;
    this._resetFrameState();
  }

  get destroyed() {
    return this._destroyed;
  }

  _resetFrameState() {
    this._clearColor = { ...DEFAULT_CLEAR_COLOR };
    this._passCommands = [];
  }

  _ensurePendingEncoder() {
    if (!this._pendingEncoder && this.device && typeof this.device.createCommandEncoder === 'function') {
      this._pendingEncoder = this.device.createCommandEncoder();
      this._hasPendingUploadCommands = false;
    }
    return this._pendingEncoder;
  }

  _flushPendingUploads(retireStaging = true) {
    if (!this._pendingEncoder) {
      return false;
    }
    const encoder = this._pendingEncoder;
    this._pendingEncoder = null;
    const hadCommands = this._hasPendingUploadCommands;
    this._hasPendingUploadCommands = false;

    if (!hadCommands || !this.device || !this.device.queue) {
      return false;
    }

    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.prepareSubmit === 'function'
    ) {
      this.stagingPool.prepareSubmit();
    }
    this.device.queue.submit([encoder.finish()]);
    if (
      retireStaging &&
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.endFrame === 'function'
    ) {
      this.stagingPool.endFrame();
    }
    return true;
  }

  _beginSubsystemsForFrame(frameIndex) {
    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.beginFrame === 'function'
    ) {
      this.stagingPool.beginFrame(frameIndex);
    }
    if (
      this.uniformRing &&
      !this.uniformRing._destroyed &&
      typeof this.uniformRing.beginFrame === 'function'
    ) {
      this.uniformRing.beginFrame(frameIndex);
    }
    if (
      this.passTimer &&
      !this.passTimer.isDestroyed &&
      typeof this.passTimer.beginFrame === 'function'
    ) {
      this.passTimer.beginFrame();
    }
    if (
      this.resourceRegistry &&
      typeof this.resourceRegistry.beginFrame === 'function'
    ) {
      const current =
        typeof this.resourceRegistry.getStats === 'function'
          ? this.resourceRegistry.getStats().currentFrame
          : 0;
      const targetFrame = Math.max(frameIndex, current);
      this.resourceRegistry.beginFrame(targetFrame);
    }
    this._frameBegunForIndex = frameIndex;
  }

  _ensureFrameBegun() {
    if (this._frameBegunForIndex !== this.frameCount) {
      this._beginSubsystemsForFrame(this.frameCount);
    }
  }

  _endSubsystemsForFrame(workDoneFence = null) {
    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.endFrame === 'function'
    ) {
      this.stagingPool.endFrame();
    }
    if (
      this.resourceRegistry &&
      typeof this.resourceRegistry.endFrame === 'function'
    ) {
      this.resourceRegistry.endFrame(workDoneFence);
    }
    if (
      this.rtPool &&
      !this.rtPool.isDestroyed &&
      typeof this.rtPool.endFrame === 'function'
    ) {
      this.rtPool.endFrame();
    }
    this._frameBegunForIndex = -1;
  }

  _getWorkDoneFence() {
    if (
      this.device &&
      this.device.queue &&
      typeof this.device.queue.onSubmittedWorkDone === 'function'
    ) {
      try {
        const fence = this.device.queue.onSubmittedWorkDone();
        if (fence && typeof fence.then === 'function') {
          return fence;
        }
      } catch (_) {
        // Ignore unavailable/lost queue fence.
      }
    }
    return null;
  }

  /**
   * Begin a frame epoch across enabled subsystems (staging pool, uniform ring,
   * pass timer, and deferred resource GC) and reset transient pass state.
   *
   * @param {number} [frameIndex]
   */
  beginFrame(frameIndex) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    // Submit any out-of-frame staged uploads (e.g. init buffer writes) before rotating epochs.
    if (this._pendingEncoder && this._frameBegunForIndex === -1) {
      this._flushPendingUploads(true);
    }
    this._resetFrameState();
    const targetFrame = frameIndex !== undefined ? frameIndex : this.frameCount;
    this._beginSubsystemsForFrame(targetFrame);
  }

  /**
   * End the current frame epoch cleanly: flushes any uncommitted upload/uniform data,
   * retires staging chunks and deferred resources, and advances the RT pool epoch.
   */
  endFrame() {
    if (this._destroyed) {
      return;
    }
    if (
      this.uniformRing &&
      !this.uniformRing._destroyed &&
      typeof this.uniformRing.flush === 'function'
    ) {
      this.uniformRing.flush();
    }
    this._flushPendingUploads(false);
    const workDoneFence = this._getWorkDoneFence();
    this._endSubsystemsForFrame(workDoneFence);
  }

  /**
   * Append a per-draw constant payload into the active DynamicUniformRing slot.
   * Flushed automatically in a single queue.writeBuffer before command submission.
   *
   * @param {ArrayBufferView | ArrayBuffer | SharedArrayBuffer} data
   * @returns {{ buffer: GPUBuffer, dynamicOffset: number, size: number, alignedSize: number }}
   */
  pushDrawConstants(data) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.uniformRing || this.uniformRing._destroyed) {
      throw new Error(
        'DynamicUniformRing is not enabled; pass { uniformRing } or { enableUniformRing: true }'
      );
    }
    this._ensureFrameBegun();
    return this.uniformRing.pushConstants(data);
  }

  /**
   * Return the most recently resolved GPU pass timings from GpuPassTimer, or null
   * when timestamp queries are unsupported or no readback has completed yet.
   *
   * @returns {{ totalGpuMs: number, passes: Array<{ name: string, durationMs: number }> } | null}
   */
  getLastPassTimings() {
    if (!this._lastPassTimings) {
      return null;
    }
    return {
      totalGpuMs: this._lastPassTimings.totalGpuMs,
      passes: this._lastPassTimings.passes.map((p) => ({ name: p.name, durationMs: p.durationMs })),
    };
  }

  /**
   * Await any in-flight GPU pass timer readbacks (and collect one pending slot if queued),
   * updating and returning the latest pass timings snapshot.
   */
  async collectPassTimings() {
    if (this._destroyed || !this.passTimer || !this.passTimer.supported) {
      return this.getLastPassTimings();
    }
    if (this._pendingTimerReadbacks.size > 0) {
      await Promise.all(Array.from(this._pendingTimerReadbacks));
    }
    if (
      !this._destroyed &&
      typeof this.passTimer.collectCompletedReadback === 'function'
    ) {
      const timing = await this.passTimer.collectCompletedReadback();
      if (timing && !this._destroyed) {
        this._lastPassTimings = timing;
      }
    }
    return this.getLastPassTimings();
  }

  /**
   * Load and index an S6 shader reflection manifest (JSON string or parsed object).
   * Lazily initializes ReflectionRegistry if one was not injected in the constructor.
   *
   * @param {string | object} jsonOrObj
   * @returns {object} Parsed reflection manifest
   */
  loadReflectionManifest(jsonOrObj) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.reflectionRegistry) {
      const ReflectionCtor = resolveSubsystemCtor(
        this._ReflectionRegistryCtor,
        'PlayGta6ReflectionLoader',
        'ReflectionRegistry',
        './reflection_loader.js'
      );
      if (!ReflectionCtor) {
        throw new Error(
          'ReflectionRegistry unavailable; load ./reflection_loader.js or inject reflectionRegistry'
        );
      }
      this.reflectionRegistry = new ReflectionCtor(this._reflectionRegistryOptions || {});
      this._ownsReflectionRegistry = true;
    }
    return this.reflectionRegistry.loadManifest(jsonOrObj);
  }

  /**
   * Explicitly associate a command-stream shader ID with a reflection hash or name key.
   *
   * @param {number} shaderId
   * @param {bigint | string | number} hashOrName
   */
  registerShaderReflectionKey(shaderId, hashOrName) {
    this._shaderReflectionKeys.set(shaderId, hashOrName);
  }

  /**
   * Resolve the GPUBindGroupLayout for a pipeline handle and group index, preferring
   * S6-reflected layouts built during pipeline creation and falling back to
   * `pipeline.getBindGroupLayout(groupIndex)`.
   *
   * @param {number} pipelineId
   * @param {number} groupIndex
   * @returns {GPUBindGroupLayout | null}
   */
  getBindGroupLayout(pipelineId, groupIndex) {
    const reflectedRender = this._pipelineBindGroupLayouts.get(pipelineId);
    if (reflectedRender && reflectedRender[groupIndex]) {
      return reflectedRender[groupIndex];
    }
    const reflectedCompute = this._computePipelineBindGroupLayouts.get(pipelineId);
    if (reflectedCompute && reflectedCompute[groupIndex]) {
      return reflectedCompute[groupIndex];
    }
    const pipeline = this.pipelines.get(pipelineId) || this.computePipelines.get(pipelineId);
    if (pipeline && typeof pipeline.getBindGroupLayout === 'function') {
      return pipeline.getBindGroupLayout(groupIndex);
    }
    return null;
  }

  /**
   * Acquire a transient render-target texture + view from RenderTargetPool.
   *
   * @param {object} desc
   * @param {string} [label]
   */
  acquireRenderTarget(desc, label) {
    if (this._destroyed) {
      throw new Error('WgpuCommandProcessor has been destroyed');
    }
    if (!this.rtPool) {
      if (this._rtPoolExplicitlyDisabled) {
        throw new Error('RenderTargetPool is disabled on this WgpuCommandProcessor');
      }
      const RtPoolCtor = resolveSubsystemCtor(
        this._RenderTargetPoolCtor,
        'PlayGta6RenderTargetPool',
        'RenderTargetPool',
        './rt_pool.js'
      );
      if (!RtPoolCtor || !this.device || typeof this.device.createTexture !== 'function') {
        throw new Error(
          'RenderTargetPool unavailable; load ./rt_pool.js or inject rtPool'
        );
      }
      this.rtPool = new RtPoolCtor({
        ...(this._rtPoolOptions || {}),
        device: this.device,
      });
      this._ownsRtPool = true;
    }
    return this.rtPool.acquire(desc, label);
  }

  /**
   * Release a previously acquired transient render-target back into RenderTargetPool
   * for same-frame aliasing or cross-frame reuse.
   *
   * @param {object | number} entry
   * @returns {boolean}
   */
  releaseRenderTarget(entry) {
    if (this._destroyed || !this.rtPool || this.rtPool.isDestroyed) {
      return false;
    }
    return this.rtPool.release(entry);
  }

  /**
   * Notify the processor of a swapchain/viewport resize so idle transient render targets
   * in RenderTargetPool are invalidated cleanly.
   *
   * @param {number} [width]
   * @param {number} [height]
   * @returns {number} Number of invalidated idle render targets
   */
  onResize(_width, _height) {
    if (
      !this._destroyed &&
      this.rtPool &&
      !this.rtPool.isDestroyed &&
      typeof this.rtPool.invalidateAll === 'function'
    ) {
      return this.rtPool.invalidateAll();
    }
    return 0;
  }

  handleResize(width, height) {
    return this.onResize(width, height);
  }

  _resolveShaderReflection(shaderId, shaderEntry, stageName) {
    if (
      !this.reflectionRegistry ||
      typeof this.reflectionRegistry.getShaderReflection !== 'function'
    ) {
      return null;
    }
    if (typeof this.reflectionRegistry.getStats === 'function') {
      const stats = this.reflectionRegistry.getStats();
      if (stats && stats.shaderCount === 0) {
        return null;
      }
    }

    const explicitKey = this._shaderReflectionKeys.get(shaderId);
    if (explicitKey !== undefined) {
      const hit = this.reflectionRegistry.getShaderReflection(explicitKey);
      if (hit) return hit;
    }

    if (shaderEntry && shaderEntry.hashHex) {
      const byHash = this.reflectionRegistry.getShaderReflection(shaderEntry.hashHex);
      if (byHash) return byHash;
    }

    if (shaderEntry && shaderEntry.entryPoint && stageName) {
      const byStageName = this.reflectionRegistry.getShaderReflection(
        `${stageName}:${shaderEntry.entryPoint}`
      );
      if (byStageName) return byStageName;
    }

    return null;
  }

  _buildReflectedPipelineLayout(reflections, label) {
    if (
      !this.reflectionRegistry ||
      !Array.isArray(reflections) ||
      reflections.length === 0 ||
      !this.device ||
      typeof this.device.createPipelineLayout !== 'function'
    ) {
      return null;
    }

    const cache = this.pipelineCache;
    const hasCacheBgl =
      Boolean(cache) &&
      (typeof cache.createBindGroupLayoutFromReflection === 'function' ||
        typeof cache.getOrCreateBindGroupLayout === 'function');
    if (!hasCacheBgl && typeof this.device.createBindGroupLayout !== 'function') {
      return null;
    }

    const capturedBgls = [];
    const collectorCache = {
      createBindGroupLayoutFromReflection: (desc) => {
        let bgl;
        if (cache && typeof cache.createBindGroupLayoutFromReflection === 'function') {
          bgl = cache.createBindGroupLayoutFromReflection(desc);
        } else if (cache && typeof cache.getOrCreateBindGroupLayout === 'function') {
          bgl = cache.getOrCreateBindGroupLayout(desc);
        } else {
          bgl = this.device.createBindGroupLayout(desc);
        }
        capturedBgls.push(bgl);
        return bgl;
      },
    };

    const layoutOptions = label ? { label } : {};
    const pipelineLayout = this.reflectionRegistry.buildPipelineLayout(
      this.device,
      collectorCache,
      reflections,
      layoutOptions
    );
    return {
      pipelineLayout,
      bindGroupLayouts: capturedBgls,
    };
  }

  _destroyBufferIfPresent(id) {
    const existing = this.buffers.get(id);
    if (existing) {
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        const retired = this.resourceRegistry.retire('buffer', id);
        if (!retired && typeof existing.destroy === 'function') {
          existing.destroy();
        }
      } else if (typeof existing.destroy === 'function') {
        existing.destroy();
      }
      this.buffers.delete(id);
    }
  }

  _destroyTextureIfPresent(id) {
    const existing = this.textures.get(id);
    if (existing) {
      if (this.resourceRegistry && typeof this.resourceRegistry.retire === 'function') {
        const retiredTex = this.resourceRegistry.retire('texture', id);
        this.resourceRegistry.retire('textureView', id);
        if (!retiredTex && typeof existing.destroy === 'function') {
          existing.destroy();
        }
      } else if (typeof existing.destroy === 'function') {
        existing.destroy();
      }
      this.textures.delete(id);
    }
    this.textureViews.delete(id);
  }

  _handleCreateBuffer(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const usage = view.getUint32(payloadOffset + 4, true);
    const byteSize = Number(view.getBigUint64(payloadOffset + 8, true));
    this._destroyBufferIfPresent(id);
    const buffer = this.device.createBuffer({
      size: byteSize,
      usage,
    });
    this.buffers.set(id, buffer);
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('buffer', id, buffer, {
        byteSize: Math.max(0, Math.trunc(byteSize)),
        label: `buffer:${id}`,
      });
    }
  }

  _handleWriteBuffer(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const dstOffset = view.getUint32(payloadOffset + 4, true);
    const dataLen = view.getUint32(payloadOffset + 8, true);
    if (16 + dataLen > payloadBytes) return;

    const buffer = this.buffers.get(id);
    if (!buffer) return;

    const data = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 16, dataLen);

    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.stageBufferUpload === 'function' &&
      dataLen > 0 &&
      dstOffset % 4 === 0
    ) {
      const rem = dataLen % 4;
      const copySize = rem === 0 ? dataLen : dataLen + (4 - rem);
      const fitsDst = typeof buffer.size !== 'number' || dstOffset + copySize <= buffer.size;
      if (fitsDst) {
        const encoder = this._ensurePendingEncoder();
        if (encoder && typeof encoder.copyBufferToBuffer === 'function') {
          this.stagingPool.stageBufferUpload(encoder, buffer, dstOffset, data);
          this._hasPendingUploadCommands = true;
          return;
        }
        if (!this._hasPendingUploadCommands) {
          this._pendingEncoder = null;
        }
      }
    }

    this.device.queue.writeBuffer(buffer, dstOffset, data);
  }

  _handleCreateShaderModule(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const stage = view.getUint32(payloadOffset + 4, true);
    const codeLen = view.getUint32(payloadOffset + 8, true);
    if (16 + codeLen > payloadBytes) return;

    const rawBytes = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 16, codeLen);
    const isShared = typeof SharedArrayBuffer !== 'undefined' && view.buffer instanceof SharedArrayBuffer;
    const code = this._decoder.decode(isShared ? rawBytes.slice() : rawBytes);
    const module = this.device.createShaderModule({ code });
    const hashHex = fnv1a64Hex(code);
    const entryPoint = extractWgslEntryPoint(code, stage);
    this.shaderModules.set(id, { module, stage, code, hashHex, entryPoint });
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('shaderModule', id, module, {
        byteSize: Math.max(0, Math.trunc(codeLen)),
        label: `shaderModule:${id}`,
      });
    }
  }

  _handleCreateRenderPipeline(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const vsShaderId = view.getUint32(payloadOffset + 4, true);
    const fsShaderId = view.getUint32(payloadOffset + 8, true);
    const vertexStride = view.getUint32(payloadOffset + 12, true);
    const topologyCode = view.getUint32(payloadOffset + 16, true);
    // Reserved pipeline flags at payloadOffset + 20 for future blend/depth state.

    const vsEntry = this.shaderModules.get(vsShaderId);
    if (!vsEntry) return;
    const fsEntry = fsShaderId !== 0 ? this.shaderModules.get(fsShaderId) : null;

    const vsRefl = this._resolveShaderReflection(vsShaderId, vsEntry, 'vertex');
    const fsRefl = fsEntry ? this._resolveShaderReflection(fsShaderId, fsEntry, 'fragment') : null;
    const matchedRefls = [];
    if (vsRefl) matchedRefls.push(vsRefl);
    if (fsRefl) matchedRefls.push(fsRefl);

    const reflected = this._buildReflectedPipelineLayout(matchedRefls, `render-pipeline-${id}`);
    if (reflected) {
      this._pipelineBindGroupLayouts.set(id, reflected.bindGroupLayouts);
    } else {
      this._pipelineBindGroupLayouts.delete(id);
    }

    const topology = TOPOLOGY_NAMES[topologyCode] || 'triangle-list';
    let vertexBuffers = vertexStride > 0
      ? [
          {
            arrayStride: vertexStride,
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: vertexStride >= 12 ? 'float32x3' : vertexStride >= 8 ? 'float32x2' : 'float32',
              },
            ],
          },
        ]
      : [];

    if (
      vsRefl &&
      vertexStride > 0 &&
      Array.isArray(vsRefl.vertexInputs) &&
      vsRefl.vertexInputs.length > 0 &&
      this.reflectionRegistry &&
      typeof this.reflectionRegistry.buildVertexBufferLayouts === 'function'
    ) {
      try {
        const reflectedVbl = this.reflectionRegistry.buildVertexBufferLayouts(vsRefl, vertexStride);
        if (Array.isArray(reflectedVbl) && reflectedVbl.length > 0) {
          vertexBuffers = reflectedVbl;
        }
      } catch (_) {
        // Keep default single-attribute vertex layout when stride/format overrides don't match.
      }
    }

    const descriptor = {
      layout: reflected ? reflected.pipelineLayout : 'auto',
      vertex: {
        module: vsEntry.module,
        entryPoint: vsRefl && vsRefl.entryPoint ? vsRefl.entryPoint : 'vs_main',
        buffers: vertexBuffers,
      },
      primitive: {
        topology,
      },
    };

    if (fsEntry) {
      descriptor.fragment = {
        module: fsEntry.module,
        entryPoint: fsRefl && fsRefl.entryPoint ? fsRefl.entryPoint : 'fs_main',
        targets: [{ format: this.format }],
      };
    }

    const pipeline = this.device.createRenderPipeline(descriptor);
    this.pipelines.set(id, pipeline);
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('renderPipeline', id, pipeline, {
        byteSize: 0,
        label: `renderPipeline:${id}`,
      });
    }
  }

  _handleCreateTexture(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const width = view.getUint32(payloadOffset + 4, true);
    const height = view.getUint32(payloadOffset + 8, true);
    const depthOrArrayLayers = Math.max(1, view.getUint32(payloadOffset + 12, true));
    const mipLevelCount = Math.max(1, view.getUint32(payloadOffset + 16, true));
    const formatCode = view.getUint32(payloadOffset + 20, true);
    const usage = view.getUint32(payloadOffset + 24, true);
    const dimensionCode = view.getUint32(payloadOffset + 28, true);

    const format = TEXTURE_FORMAT_NAMES[formatCode] || 'rgba8unorm';
    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode] || '2d';

    this._destroyTextureIfPresent(id);
    const texture = this.device.createTexture({
      size: {
        width,
        height,
        depthOrArrayLayers,
      },
      mipLevelCount,
      format,
      usage,
      dimension,
    });
    this.textures.set(id, texture);
    let textureView = null;
    if (texture && typeof texture.createView === 'function') {
      textureView = texture.createView();
      this.textureViews.set(id, textureView);
    }
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      const byteSize = estimateTextureBytes(
        width,
        height,
        depthOrArrayLayers,
        mipLevelCount,
        format,
        dimension
      );
      this.resourceRegistry.register('texture', id, texture, {
        byteSize,
        label: `texture:${id}`,
      });
      if (textureView) {
        this.resourceRegistry.register('textureView', id, textureView, {
          byteSize: 0,
          label: `textureView:${id}`,
        });
      }
    }
  }

  _handleWriteTexture(view, payloadOffset, payloadBytes) {
    const textureId = view.getUint32(payloadOffset, true);
    const mipLevel = view.getUint32(payloadOffset + 4, true);
    const originX = view.getUint32(payloadOffset + 8, true);
    const originY = view.getUint32(payloadOffset + 12, true);
    const originZ = view.getUint32(payloadOffset + 16, true);
    const width = view.getUint32(payloadOffset + 20, true);
    const height = view.getUint32(payloadOffset + 24, true);
    const depthOrArrayLayers = Math.max(1, view.getUint32(payloadOffset + 28, true));
    const bytesPerRow = view.getUint32(payloadOffset + 32, true);
    const rowsPerImage = view.getUint32(payloadOffset + 36, true);
    const dataLen = view.getUint32(payloadOffset + 40, true);
    if (48 + dataLen > payloadBytes) return;

    const texture = this.textures.get(textureId);
    if (!texture) return;

    const data = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 48, dataLen);

    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.stageTextureUpload === 'function' &&
      width > 0 &&
      height > 0 &&
      dataLen > 0
    ) {
      const effectiveRowsPerImage = rowsPerImage > 0 ? rowsPerImage : height;
      const totalRows = effectiveRowsPerImage * depthOrArrayLayers;
      const canStage =
        bytesPerRow > 0
          ? dataLen >= bytesPerRow * totalRows
          : totalRows > 0 && dataLen % totalRows === 0;
      if (canStage) {
        const encoder = this._ensurePendingEncoder();
        if (encoder && typeof encoder.copyBufferToTexture === 'function') {
          const layout = {
            width,
            height,
            depthOrArrayLayers,
            mipLevel,
            origin: { x: originX, y: originY, z: originZ },
          };
          if (bytesPerRow > 0) {
            layout.bytesPerRow = bytesPerRow;
          }
          if (rowsPerImage > 0) {
            layout.rowsPerImage = rowsPerImage;
          }
          this.stagingPool.stageTextureUpload(encoder, texture, data, layout);
          this._hasPendingUploadCommands = true;
          return;
        }
        if (!this._hasPendingUploadCommands) {
          this._pendingEncoder = null;
        }
      }
    }

    const dataLayout = { offset: 0 };
    if (bytesPerRow > 0) {
      dataLayout.bytesPerRow = bytesPerRow;
    }
    if (rowsPerImage > 0) {
      dataLayout.rowsPerImage = rowsPerImage;
    }
    this.device.queue.writeTexture(
      {
        texture,
        mipLevel,
        origin: { x: originX, y: originY, z: originZ },
      },
      data,
      dataLayout,
      {
        width,
        height,
        depthOrArrayLayers,
      }
    );
  }

  _handleCreateSampler(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const addressModeU = ADDRESS_MODE_NAMES[view.getUint32(payloadOffset + 4, true)] || 'clamp-to-edge';
    const addressModeV = ADDRESS_MODE_NAMES[view.getUint32(payloadOffset + 8, true)] || 'clamp-to-edge';
    const addressModeW = ADDRESS_MODE_NAMES[view.getUint32(payloadOffset + 12, true)] || 'clamp-to-edge';
    const magFilter = FILTER_MODE_NAMES[view.getUint32(payloadOffset + 16, true)] || 'nearest';
    const minFilter = FILTER_MODE_NAMES[view.getUint32(payloadOffset + 20, true)] || 'nearest';
    const mipmapFilter = FILTER_MODE_NAMES[view.getUint32(payloadOffset + 24, true)] || 'nearest';
    const maxAnisotropy = Math.max(1, view.getUint32(payloadOffset + 28, true));

    const sampler = this.device.createSampler({
      addressModeU,
      addressModeV,
      addressModeW,
      magFilter,
      minFilter,
      mipmapFilter,
      maxAnisotropy,
    });
    this.samplers.set(id, sampler);
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('sampler', id, sampler, {
        byteSize: 0,
        label: `sampler:${id}`,
      });
    }
  }

  _handleCreateComputePipeline(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const csShaderId = view.getUint32(payloadOffset + 4, true);
    // Reserved flags at payloadOffset + 8 and reserved at payloadOffset + 12.

    const csEntry = this.shaderModules.get(csShaderId);
    if (!csEntry) return;

    const csRefl = this._resolveShaderReflection(csShaderId, csEntry, 'compute');
    const reflected = csRefl
      ? this._buildReflectedPipelineLayout([csRefl], `compute-pipeline-${id}`)
      : null;
    if (reflected) {
      this._computePipelineBindGroupLayouts.set(id, reflected.bindGroupLayouts);
    } else {
      this._computePipelineBindGroupLayouts.delete(id);
    }

    const pipeline = this.device.createComputePipeline({
      layout: reflected ? reflected.pipelineLayout : 'auto',
      compute: {
        module: csEntry.module,
        entryPoint: csRefl && csRefl.entryPoint ? csRefl.entryPoint : 'cs_main',
      },
    });
    this.computePipelines.set(id, pipeline);
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('computePipeline', id, pipeline, {
        byteSize: 0,
        label: `computePipeline:${id}`,
      });
    }
  }

  _handleCreateBindGroup(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const pipelineId = view.getUint32(payloadOffset + 4, true);
    const groupIndex = view.getUint32(payloadOffset + 8, true);
    const entryCount = view.getUint32(payloadOffset + 12, true);
    if (16 + entryCount * 32 > payloadBytes) return;

    const renderPipeline = this.pipelines.get(pipelineId);
    const computePipeline = this.computePipelines.get(pipelineId);
    const pipeline = renderPipeline || computePipeline;
    const layout = this.getBindGroupLayout(pipelineId, groupIndex);
    if (!pipeline || !layout) return;

    const entries = [];
    for (let i = 0; i < entryCount; i++) {
      const entryOffset = payloadOffset + 16 + i * 32;
      const binding = view.getUint32(entryOffset, true);
      const resourceType = view.getUint32(entryOffset + 4, true);
      const resourceId = view.getUint32(entryOffset + 8, true);
      const offset = Number(view.getBigUint64(entryOffset + 16, true));
      const size = Number(view.getBigUint64(entryOffset + 24, true));

      if (resourceType === BIND_RESOURCE_TYPE.BUFFER) {
        const buffer = this.buffers.get(resourceId);
        if (!buffer) return;
        const resource = { buffer, offset };
        if (size > 0) {
          resource.size = size;
        }
        entries.push({ binding, resource });
      } else if (resourceType === BIND_RESOURCE_TYPE.TEXTURE_VIEW) {
        let textureView = this.textureViews.get(resourceId);
        if (!textureView) {
          const texture = this.textures.get(resourceId);
          if (texture && typeof texture.createView === 'function') {
            textureView = texture.createView();
            this.textureViews.set(resourceId, textureView);
            if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
              this.resourceRegistry.register('textureView', resourceId, textureView, {
                byteSize: 0,
                label: `textureView:${resourceId}`,
              });
            }
          }
        }
        if (!textureView) return;
        entries.push({ binding, resource: textureView });
      } else if (resourceType === BIND_RESOURCE_TYPE.SAMPLER) {
        const sampler = this.samplers.get(resourceId);
        if (!sampler) return;
        entries.push({ binding, resource: sampler });
      } else {
        return;
      }
    }

    const bindGroup = this.device.createBindGroup({
      layout,
      entries,
    });
    this.bindGroups.set(id, bindGroup);
    this._bindGroupKinds.set(id, computePipeline && !renderPipeline ? 'compute' : 'render');
    if (this.resourceRegistry && typeof this.resourceRegistry.register === 'function') {
      this.resourceRegistry.register('bindGroup', id, bindGroup, {
        byteSize: 0,
        label: `bindGroup:${id}`,
      });
    }
  }

  _handleDestroyResource(id) {
    this._destroyBufferIfPresent(id);
    this._destroyTextureIfPresent(id);
    if (this.resourceRegistry && typeof this.resourceRegistry.retireHandle === 'function') {
      this.resourceRegistry.retireHandle(id);
    }
    this.samplers.delete(id);
    this.bindGroups.delete(id);
    this._bindGroupKinds.delete(id);
    this.shaderModules.delete(id);
    this.pipelines.delete(id);
    this.computePipelines.delete(id);
    this._pipelineBindGroupLayouts.delete(id);
    this._computePipelineBindGroupLayouts.delete(id);
    this._shaderReflectionKeys.delete(id);
    if (this.extension) this.extension.destroyResource(id);
  }

  _flushPresent() {
    const encoder = this._pendingEncoder || this.device.createCommandEncoder();
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;

    const swapchainView = this.ctx.getCurrentTexture().createView();
    let activePass = null;
    let activePassKind = null;
    let renderPassOpened = false;
    let renderPassIdx = 0;
    let computePassIdx = 0;

    const timerSupported = Boolean(this.passTimer && this.passTimer.supported);

    const endActivePass = () => {
      if (activePass) {
        activePass.end();
        activePass = null;
        activePassKind = null;
      }
    };

    const ensureRenderPass = () => {
      if (activePassKind === 'render') return activePass;
      endActivePass();
      const attachment = {
        view: swapchainView,
        loadOp: renderPassOpened ? 'load' : 'clear',
        storeOp: 'store',
      };
      if (!renderPassOpened) {
        attachment.clearValue = { ...this._clearColor };
      }
      const passDesc = {
        colorAttachments: [attachment],
      };
      if (timerSupported && typeof this.passTimer.getPassTimestampWrites === 'function') {
        const tsWrites = this.passTimer.getPassTimestampWrites(`render_${renderPassIdx}`);
        if (tsWrites) {
          passDesc.timestampWrites = tsWrites;
        }
      }
      renderPassIdx += 1;
      activePass = encoder.beginRenderPass(passDesc);
      activePassKind = 'render';
      renderPassOpened = true;
      return activePass;
    };

    const ensureComputePass = () => {
      if (activePassKind === 'compute') return activePass;
      endActivePass();
      if (timerSupported && typeof this.passTimer.getPassTimestampWrites === 'function') {
        const tsWrites = this.passTimer.getPassTimestampWrites(`compute_${computePassIdx}`);
        computePassIdx += 1;
        activePass = tsWrites
          ? encoder.beginComputePass({ timestampWrites: tsWrites })
          : encoder.beginComputePass();
      } else {
        computePassIdx += 1;
        activePass = encoder.beginComputePass();
      }
      activePassKind = 'compute';
      return activePass;
    };

    // Pass access for queued extension commands ({ type, run(passControl) }).
    const passControl = {
      swapchainView,
      get pass() { return activePass; },
      get passKind() { return activePassKind; },
      endPass: endActivePass,
      beginRenderPass: (descriptor) => {
        endActivePass();
        if (timerSupported && typeof this.passTimer.getPassTimestampWrites === 'function') {
          const tsWrites = this.passTimer.getPassTimestampWrites(`render_${renderPassIdx}`);
          if (tsWrites) descriptor = { ...descriptor, timestampWrites: tsWrites };
        }
        renderPassIdx += 1;
        activePass = encoder.beginRenderPass(descriptor);
        activePassKind = 'render';
        renderPassOpened = true;
        return activePass;
      },
    };

    for (const cmd of this._passCommands) {
      switch (cmd.type) {
        case 'setPipeline': {
          const pass = ensureRenderPass();
          const pipeline = this.pipelines.get(cmd.pipelineId);
          if (pipeline) {
            pass.setPipeline(pipeline);
          }
          break;
        }
        case 'setVertexBuffer': {
          const pass = ensureRenderPass();
          const buffer = this.buffers.get(cmd.bufferId);
          if (buffer) {
            pass.setVertexBuffer(cmd.slot, buffer, cmd.offset);
          }
          break;
        }
        case 'setIndexBuffer': {
          const pass = ensureRenderPass();
          const buffer = this.buffers.get(cmd.bufferId);
          if (buffer) {
            pass.setIndexBuffer(buffer, cmd.indexFormat, cmd.offset);
          }
          break;
        }
        case 'setBindGroup': {
          const bindGroup = this.bindGroups.get(cmd.bindGroupId);
          if (bindGroup) {
            const pass =
              activePass ||
              (this._bindGroupKinds.get(cmd.bindGroupId) === 'compute'
                ? ensureComputePass()
                : ensureRenderPass());
            pass.setBindGroup(cmd.groupIndex, bindGroup);
          }
          break;
        }
        case 'draw': {
          const pass = ensureRenderPass();
          pass.draw(cmd.vertexCount, cmd.instanceCount, cmd.firstVertex, cmd.firstInstance);
          this._drawCallCount += 1;
          break;
        }
        case 'drawIndexed': {
          const pass = ensureRenderPass();
          pass.drawIndexed(
            cmd.indexCount,
            cmd.instanceCount,
            cmd.firstIndex,
            cmd.baseVertex,
            cmd.firstInstance
          );
          this._drawCallCount += 1;
          break;
        }
        case 'setComputePipeline': {
          const pass = ensureComputePass();
          const pipeline = this.computePipelines.get(cmd.pipelineId);
          if (pipeline) {
            pass.setPipeline(pipeline);
          }
          break;
        }
        case 'dispatchCompute': {
          const pass = ensureComputePass();
          pass.dispatchWorkgroups(cmd.workgroupX, cmd.workgroupY, cmd.workgroupZ);
          this._computeDispatchCount += 1;
          break;
        }
        default:
          if (typeof cmd.run === 'function') cmd.run(passControl);
          break;
      }
    }

    endActivePass();
    if (!renderPassOpened) {
      ensureRenderPass();
      endActivePass();
    }

    let timerResolved = false;
    if (timerSupported && typeof this.passTimer.resolveFrame === 'function') {
      timerResolved = Boolean(this.passTimer.resolveFrame(encoder));
    }

    // Flush batched per-draw dynamic uniform constants once before queue submission.
    if (
      this.uniformRing &&
      !this.uniformRing._destroyed &&
      typeof this.uniformRing.flush === 'function'
    ) {
      this.uniformRing.flush();
    }

    // Unmap any mapped staging chunks before queue submission.
    if (
      this.stagingPool &&
      !this.stagingPool._destroyed &&
      typeof this.stagingPool.prepareSubmit === 'function'
    ) {
      this.stagingPool.prepareSubmit();
    }

    this.device.queue.submit([encoder.finish()]);
    this.frameCount += 1;

    const workDoneFence = this._getWorkDoneFence();
    this._endSubsystemsForFrame(workDoneFence);

    if (
      timerResolved &&
      this.passTimer &&
      typeof this.passTimer.collectCompletedReadback === 'function'
    ) {
      const readbackPromise = Promise.resolve(this.passTimer.collectCompletedReadback()).then(
        (timing) => {
          if (timing && !this._destroyed) {
            this._lastPassTimings = timing;
          }
          return timing;
        },
        () => null
      );
      this._pendingTimerReadbacks.add(readbackPromise);
      readbackPromise.finally(() => {
        this._pendingTimerReadbacks.delete(readbackPromise);
      });
    }

    this._resetFrameState();
  }

  attachRing(ringOrSab, options = {}) {
    if (ringOrSab && typeof ringOrSab.drainTo === 'function') {
      this.ring = ringOrSab;
      return this.ring;
    }
    const RingCtor = resolveGpuRingBuffer(options.GpuRingBuffer || this._GpuRingBuffer);
    const ctorOptions = {};
    if (options.byteOffset !== undefined) {
      ctorOptions.byteOffset = options.byteOffset;
    }
    if (options.initialize !== undefined) {
      ctorOptions.initialize = options.initialize;
    }
    if (options.capacityBytes !== undefined) {
      ctorOptions.capacityBytes = options.capacityBytes;
    }
    this.ring = new RingCtor(ringOrSab, ctorOptions);
    return this.ring;
  }

  drainRing() {
    if (this._destroyed || !this.ring || !this.device || !this.ctx) {
      return { frameCount: this.frameCount, presents: 0, bytesDrained: 0 };
    }
    let presents = 0;
    const bytesDrained = this.ring.drainTo((bytes) => {
      const result = this.executeCommandBuffer(bytes);
      presents += result.presents;
    });
    return { frameCount: this.frameCount, presents, bytesDrained };
  }

  executeCommandBuffer(input) {
    if (this._destroyed || !this.device || !this.ctx) {
      return { frameCount: this.frameCount, presents: 0 };
    }

    let view;
    if (
      input instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer)
    ) {
      view = new DataView(input);
    } else if (ArrayBuffer.isView(input)) {
      view = new DataView(input.buffer, input.byteOffset, input.byteLength);
    } else {
      throw new TypeError(
        'WgpuCommandProcessor.executeCommandBuffer expects an ArrayBuffer, SharedArrayBuffer, or ArrayBufferView'
      );
    }

    let offset = 0;
    let presents = 0;

    while (offset + 8 <= view.byteLength) {
      const opcode = view.getUint16(offset, true);
      const payloadBytes = view.getUint32(offset + 4, true);
      const payloadOffset = offset + 8;
      if (payloadOffset + payloadBytes > view.byteLength) break;

      this._packetCount += 1;
      this._bytesProcessed += 8 + payloadBytes;

      switch (opcode) {
        case OPCODES.NOP:
          break;
        case OPCODES.BEGIN_FRAME:
          if (this._pendingEncoder && this._frameBegunForIndex === -1) {
            this._flushPendingUploads(true);
          }
          this._resetFrameState();
          this._ensureFrameBegun();
          break;
        case OPCODES.CLEAR_COLOR:
          if (payloadBytes >= 16) {
            this._clearColor = {
              r: view.getFloat32(payloadOffset, true),
              g: view.getFloat32(payloadOffset + 4, true),
              b: view.getFloat32(payloadOffset + 8, true),
              a: view.getFloat32(payloadOffset + 12, true),
            };
          }
          break;
        case OPCODES.CREATE_BUFFER:
          if (payloadBytes >= 16) {
            this._handleCreateBuffer(view, payloadOffset);
          }
          break;
        case OPCODES.WRITE_BUFFER:
          if (payloadBytes >= 16) {
            this._handleWriteBuffer(view, payloadOffset, payloadBytes);
          }
          break;
        case OPCODES.CREATE_SHADER_MODULE:
          if (payloadBytes >= 16) {
            this._handleCreateShaderModule(view, payloadOffset, payloadBytes);
          }
          break;
        case OPCODES.CREATE_RENDER_PIPELINE:
          if (payloadBytes >= 24) {
            this._handleCreateRenderPipeline(view, payloadOffset);
          }
          break;
        case OPCODES.SET_PIPELINE:
          if (payloadBytes >= 4) {
            this._passCommands.push({
              type: 'setPipeline',
              pipelineId: view.getUint32(payloadOffset, true),
            });
          }
          break;
        case OPCODES.SET_VERTEX_BUFFER:
          if (payloadBytes >= 16) {
            this._passCommands.push({
              type: 'setVertexBuffer',
              slot: view.getUint32(payloadOffset, true),
              bufferId: view.getUint32(payloadOffset + 4, true),
              offset: Number(view.getBigUint64(payloadOffset + 8, true)),
            });
          }
          break;
        case OPCODES.DRAW:
          if (payloadBytes >= 16) {
            this._passCommands.push({
              type: 'draw',
              vertexCount: view.getUint32(payloadOffset, true),
              instanceCount: view.getUint32(payloadOffset + 4, true),
              firstVertex: view.getUint32(payloadOffset + 8, true),
              firstInstance: view.getUint32(payloadOffset + 12, true),
            });
          }
          break;
        case OPCODES.PRESENT:
          this._flushPresent();
          presents += 1;
          break;
        case OPCODES.DESTROY_RESOURCE:
          if (payloadBytes >= 4) {
            this._handleDestroyResource(view.getUint32(payloadOffset, true));
          }
          break;
        case OPCODES.CREATE_TEXTURE:
          if (payloadBytes >= 32) {
            this._handleCreateTexture(view, payloadOffset);
          }
          break;
        case OPCODES.WRITE_TEXTURE:
          if (payloadBytes >= 48) {
            this._handleWriteTexture(view, payloadOffset, payloadBytes);
          }
          break;
        case OPCODES.CREATE_SAMPLER:
          if (payloadBytes >= 32) {
            this._handleCreateSampler(view, payloadOffset);
          }
          break;
        case OPCODES.CREATE_BIND_GROUP:
          if (payloadBytes >= 16) {
            this._handleCreateBindGroup(view, payloadOffset, payloadBytes);
          }
          break;
        case OPCODES.SET_BIND_GROUP:
          if (payloadBytes >= 8) {
            this._passCommands.push({
              type: 'setBindGroup',
              groupIndex: view.getUint32(payloadOffset, true),
              bindGroupId: view.getUint32(payloadOffset + 4, true),
            });
          }
          break;
        case OPCODES.SET_INDEX_BUFFER:
          if (payloadBytes >= 16) {
            const formatCode = view.getUint32(payloadOffset + 4, true);
            this._passCommands.push({
              type: 'setIndexBuffer',
              bufferId: view.getUint32(payloadOffset, true),
              indexFormat: INDEX_FORMAT_NAMES[formatCode] || 'uint16',
              offset: Number(view.getBigUint64(payloadOffset + 8, true)),
            });
          }
          break;
        case OPCODES.DRAW_INDEXED:
          if (payloadBytes >= 20) {
            this._passCommands.push({
              type: 'drawIndexed',
              indexCount: view.getUint32(payloadOffset, true),
              instanceCount: view.getUint32(payloadOffset + 4, true),
              firstIndex: view.getUint32(payloadOffset + 8, true),
              baseVertex: view.getInt32(payloadOffset + 12, true),
              firstInstance: view.getUint32(payloadOffset + 16, true),
            });
          }
          break;
        case OPCODES.CREATE_COMPUTE_PIPELINE:
          if (payloadBytes >= 16) {
            this._handleCreateComputePipeline(view, payloadOffset);
          }
          break;
        case OPCODES.SET_COMPUTE_PIPELINE:
          if (payloadBytes >= 4) {
            this._passCommands.push({
              type: 'setComputePipeline',
              pipelineId: view.getUint32(payloadOffset, true),
            });
          }
          break;
        case OPCODES.DISPATCH_COMPUTE:
          if (payloadBytes >= 12) {
            this._passCommands.push({
              type: 'dispatchCompute',
              workgroupX: view.getUint32(payloadOffset, true),
              workgroupY: view.getUint32(payloadOffset + 4, true),
              workgroupZ: view.getUint32(payloadOffset + 8, true),
            });
          }
          break;
        default:
          if (this.extension) this.extension.handlePacket(opcode, view, payloadOffset, payloadBytes);
          break;
      }

      offset = payloadOffset + payloadBytes;
    }

    return { frameCount: this.frameCount, presents };
  }

  execute(input) {
    return this.executeCommandBuffer(input);
  }

  /**
   * Return aggregated processor and subsystem telemetry (packet counts, frame count,
   * staging pool, dynamic uniform ring, GPU pass timings, VRAM resource registry,
   * render-target pool hit rate, and reflection shader count).
   */
  getStats() {
    const stagingPool =
      this.stagingPool && typeof this.stagingPool.getStats === 'function'
        ? this.stagingPool.getStats()
        : null;
    const uniformRing =
      this.uniformRing && typeof this.uniformRing.getStats === 'function'
        ? this.uniformRing.getStats()
        : null;
    const passTimerBase =
      this.passTimer && typeof this.passTimer.getStats === 'function'
        ? this.passTimer.getStats()
        : null;
    const lastPassTimings = this.getLastPassTimings();
    const passTimer = passTimerBase
      ? { ...passTimerBase, lastTimings: lastPassTimings }
      : null;
    const resourceRegistry =
      this.resourceRegistry && typeof this.resourceRegistry.getStats === 'function'
        ? this.resourceRegistry.getStats()
        : null;
    const rtPool =
      this.rtPool && typeof this.rtPool.getStats === 'function'
        ? this.rtPool.getStats()
        : null;
    const reflectionRegistry =
      this.reflectionRegistry && typeof this.reflectionRegistry.getStats === 'function'
        ? this.reflectionRegistry.getStats()
        : null;

    return {
      frameCount: this.frameCount,
      packetCount: this._packetCount,
      drawCalls: this._drawCallCount,
      computeDispatches: this._computeDispatchCount,
      bytesProcessed: this._bytesProcessed,
      destroyed: this._destroyed,
      resources: {
        buffers: this.buffers.size,
        textures: this.textures.size,
        textureViews: this.textureViews.size,
        samplers: this.samplers.size,
        bindGroups: this.bindGroups.size,
        shaderModules: this.shaderModules.size,
        pipelines: this.pipelines.size,
        computePipelines: this.computePipelines.size,
      },
      stagingPool,
      uniformRing,
      passTimer,
      lastPassTimings,
      resourceRegistry,
      vramBytes: resourceRegistry ? resourceRegistry.totalBytes : 0,
      rtPool,
      rtPoolHitRate: rtPool ? rtPool.hitRate : 0,
      reflectionRegistry,
      reflectionShaderCount: reflectionRegistry ? reflectionRegistry.shaderCount : 0,
    };
  }

  /**
   * Idempotently destroy all tracked GPU resources and wired subsystems.
   */
  destroy() {
    if (this._destroyed) {
      return;
    }
    this._destroyed = true;
    this._pendingEncoder = null;
    this._hasPendingUploadCommands = false;
    this._passCommands = [];
    this._frameBegunForIndex = -1;

    // 1. Resource registry destroys all registered active & pending-retirement resources.
    const destroyedByRegistry = new Set();
    if (this.resourceRegistry && typeof this.resourceRegistry.destroyAll === 'function') {
      if (typeof this.resourceRegistry.list === 'function') {
        const activeEntries = this.resourceRegistry.list();
        for (let i = 0; i < activeEntries.length; i += 1) {
          destroyedByRegistry.add(activeEntries[i].resource);
        }
      }
      this.resourceRegistry.destroyAll();
    }

    // Destroy any buffers/textures not managed by resourceRegistry.
    for (const buffer of this.buffers.values()) {
      if (buffer && !destroyedByRegistry.has(buffer) && typeof buffer.destroy === 'function') {
        buffer.destroy();
      }
    }
    for (const texture of this.textures.values()) {
      if (texture && !destroyedByRegistry.has(texture) && typeof texture.destroy === 'function') {
        texture.destroy();
      }
    }

    this.buffers.clear();
    this.textures.clear();
    this.textureViews.clear();
    this.samplers.clear();
    this.bindGroups.clear();
    this._bindGroupKinds.clear();
    this.shaderModules.clear();
    this.pipelines.clear();
    this.computePipelines.clear();
    this._pipelineBindGroupLayouts.clear();
    this._computePipelineBindGroupLayouts.clear();
    this._shaderReflectionKeys.clear();

    // 2. Tear down auxiliary subsystems idempotently.
    if (this.stagingPool && typeof this.stagingPool.destroy === 'function') {
      this.stagingPool.destroy();
    }
    if (this.uniformRing && typeof this.uniformRing.destroy === 'function') {
      this.uniformRing.destroy();
    }
    if (this.passTimer && typeof this.passTimer.destroy === 'function') {
      this.passTimer.destroy();
    }
    if (this.rtPool && typeof this.rtPool.destroy === 'function') {
      this.rtPool.destroy();
    }
    if (
      this.reflectionRegistry &&
      this._ownsReflectionRegistry &&
      typeof this.reflectionRegistry.clear === 'function'
    ) {
      this.reflectionRegistry.clear();
    }

    this._pendingTimerReadbacks.clear();
    this._lastPassTimings = null;
    this.ring = null;
  }
}

function createWgpuWorkerHandler(options = {}) {
  const postMessage =
    typeof options.postMessage === 'function' ? options.postMessage : () => {};
  const injectedRingCtor =
    typeof options.GpuRingBuffer === 'function' ? options.GpuRingBuffer : null;
  let canvas = options.canvas || null;
  let processor = options.processor || null;
  let attachedRing = processor && processor.ring ? processor.ring : null;

  async function defaultInitGpu(offscreen, width, height, initPayload = {}) {
    canvas = offscreen;
    canvas.width = width;
    canvas.height = height;

    if (typeof navigator === 'undefined' || !navigator.gpu) {
      postMessage({ error: 'navigator.gpu unavailable in worker' });
      return null;
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      postMessage({ error: 'no WebGPU adapter available' });
      return null;
    }

    const device = await adapter.requestDevice();
    const format = navigator.gpu.getPreferredCanvasFormat();
    const ctx = canvas.getContext('webgpu');
    ctx.configure({ device, format, alphaMode: 'opaque' });
    processor = new WgpuCommandProcessor(device, ctx, {
      format,
      GpuRingBuffer: injectedRingCtor,
      stagingPool: options.stagingPool,
      uniformRing: options.uniformRing,
      passTimer: options.passTimer,
      reflectionRegistry: options.reflectionRegistry,
      resourceRegistry: options.resourceRegistry,
      rtPool: options.rtPool,
      pipelineCache: options.pipelineCache,
      enableStagingPool: initPayload.enableStagingPool ?? options.enableStagingPool,
      enableUniformRing: initPayload.enableUniformRing ?? options.enableUniformRing,
      enablePassTimer: initPayload.enablePassTimer ?? options.enablePassTimer,
      enableReflectionRegistry:
        initPayload.enableReflectionRegistry ?? options.enableReflectionRegistry,
      enableResourceRegistry:
        initPayload.enableResourceRegistry ?? options.enableResourceRegistry,
      enableRtPool: initPayload.enableRtPool ?? options.enableRtPool,
      reflectionManifest: initPayload.reflectionManifest ?? options.reflectionManifest,
    });
    try {
      if (!globalThis.PlayGta6RenderState && typeof importScripts === 'function') {
        importScripts('./render_state_packets.js');
      }
      if (globalThis.PlayGta6RenderState) globalThis.PlayGta6RenderState.installRenderStatePackets(processor);
      if (!globalThis.PlayGta6TexturePackets && typeof importScripts === 'function') {
        importScripts('./texture_packets.js');
      }
      if (globalThis.PlayGta6TexturePackets) globalThis.PlayGta6TexturePackets.installTexturePackets(processor);
    } catch (err) {
      postMessage({ error: `packet ABI v2 unavailable: ${String(err)}` });
    }
    if (attachedRing) {
      processor.attachRing(attachedRing);
    }

    const info = adapter.info
      ? `${adapter.info.vendor || 'gpu'} ${adapter.info.architecture || ''}`.trim()
      : 'WebGPU';
    postMessage({ ready: true, adapter: info });
    return processor;
  }

  const initGpu = typeof options.initGpu === 'function' ? options.initGpu : defaultInitGpu;

  function attachRingFromMessage(msg) {
    const ringSource = msg.ring || msg.ringSab;
    if (!ringSource) {
      throw new Error('attachRing requires ringSab or ring');
    }
    const RingCtor =
      ringSource && typeof ringSource.drainTo === 'function'
        ? null
        : resolveGpuRingBuffer(msg.GpuRingBuffer || injectedRingCtor);
    attachedRing = RingCtor
      ? new RingCtor(ringSource, { byteOffset: msg.byteOffset || 0 })
      : ringSource;
    if (processor) {
      processor.attachRing(attachedRing);
    }
    postMessage({
      ringAttached: true,
      capacityBytes: attachedRing.capacityBytes,
    });
    return attachedRing;
  }

  async function handleMessage(msg) {
    const payload = msg || {};
    try {
      if (payload.init && payload.canvas) {
        const created = await initGpu(
          payload.canvas,
          payload.width || 1280,
          payload.height || 720,
          payload
        );
        if (created && created !== processor) {
          processor = created;
        }
        if (payload.ringSab || payload.ring) {
          attachRingFromMessage(payload);
        } else if (attachedRing && processor) {
          processor.attachRing(attachedRing);
        }
      } else if (payload.resize && canvas) {
        const nextW = Math.max(1, payload.width | 0);
        const nextH = Math.max(1, payload.height | 0);
        canvas.width = nextW;
        canvas.height = nextH;
        if (processor && typeof processor.onResize === 'function') {
          processor.onResize(nextW, nextH);
        }
      } else if (payload.attachRing) {
        attachRingFromMessage(payload);
      } else if (payload.loadReflectionManifest && processor) {
        processor.loadReflectionManifest(payload.loadReflectionManifest);
        const stats = processor.getStats();
        postMessage({
          id: payload.id,
          reflectionLoaded: true,
          shaderCount: stats.reflectionShaderCount,
        });
      } else if (payload.stats) {
        postMessage({
          id: payload.id,
          stats: processor && typeof processor.getStats === 'function' ? processor.getStats() : null,
        });
      } else if (payload.destroy && processor) {
        if (typeof processor.destroy === 'function') {
          processor.destroy();
        }
        postMessage({ id: payload.id, destroyed: true });
      } else if (payload.drainRing && processor) {
        const result = processor.drainRing();
        if (result.presents > 0) {
          postMessage({ frame: result.frameCount });
        }
      } else if (payload.cmdBuffer && processor) {
        const result = processor.executeCommandBuffer(payload.cmdBuffer);
        if (result.presents > 0) {
          postMessage({ frame: result.frameCount });
        }
      }
    } catch (err) {
      postMessage({ error: String(err) });
    }
  }

  return {
    handleMessage,
    get processor() {
      return processor;
    },
    get ring() {
      return attachedRing;
    },
  };
}

const exportedApi = {
  WgpuCommandProcessor,
  createWgpuWorkerHandler,
  resolveGpuRingBuffer,
  estimateTextureBytes,
  fnv1a64Hex,
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
  TEXTURE_USAGE,
  TEXTURE_FORMAT,
  TEXTURE_DIMENSION,
  ADDRESS_MODE,
  FILTER_MODE,
  BIND_RESOURCE_TYPE,
  INDEX_FORMAT,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Wgpu = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  const workerHandler = createWgpuWorkerHandler({
    postMessage: (data) => self.postMessage(data),
  });
  self.onmessage = async (ev) => {
    await workerHandler.handleMessage(ev ? ev.data : null);
  };
}
