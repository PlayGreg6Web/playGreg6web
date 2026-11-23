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

const DEFAULT_CLEAR_COLOR = Object.freeze({ r: 0.05, g: 0.07, b: 0.11, a: 1.0 });

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
    this.shaderModules = new Map();
    this.pipelines = new Map();
    this.frameCount = 0;
    this.ring = null;
    this._GpuRingBuffer = typeof options.GpuRingBuffer === 'function' ? options.GpuRingBuffer : null;
    this._decoder = new TextDecoder('utf-8');
    this._resetFrameState();
  }

  _resetFrameState() {
    this._clearColor = { ...DEFAULT_CLEAR_COLOR };
    this._passCommands = [];
  }

  _destroyBufferIfPresent(id) {
    const existing = this.buffers.get(id);
    if (existing) {
      if (typeof existing.destroy === 'function') {
        existing.destroy();
      }
      this.buffers.delete(id);
    }
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
  }

  _handleWriteBuffer(view, payloadOffset, payloadBytes) {
    const id = view.getUint32(payloadOffset, true);
    const dstOffset = view.getUint32(payloadOffset + 4, true);
    const dataLen = view.getUint32(payloadOffset + 8, true);
    if (16 + dataLen > payloadBytes) return;

    const buffer = this.buffers.get(id);
    if (!buffer) return;

    const data = new Uint8Array(view.buffer, view.byteOffset + payloadOffset + 16, dataLen);
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
    this.shaderModules.set(id, { module, stage });
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

    const topology = TOPOLOGY_NAMES[topologyCode] || 'triangle-list';
    const vertexBuffers = vertexStride > 0
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

    const descriptor = {
      layout: 'auto',
      vertex: {
        module: vsEntry.module,
        entryPoint: 'vs_main',
        buffers: vertexBuffers,
      },
      primitive: {
        topology,
      },
    };

    if (fsEntry) {
      descriptor.fragment = {
        module: fsEntry.module,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      };
    }

    const pipeline = this.device.createRenderPipeline(descriptor);
    this.pipelines.set(id, pipeline);
  }

  _handleDestroyResource(id) {
    this._destroyBufferIfPresent(id);
    this.shaderModules.delete(id);
    this.pipelines.delete(id);
  }

  _flushPresent() {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.ctx.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { ...this._clearColor },
        },
      ],
    });

    for (const cmd of this._passCommands) {
      if (cmd.type === 'setPipeline') {
        const pipeline = this.pipelines.get(cmd.pipelineId);
        if (pipeline) {
          pass.setPipeline(pipeline);
        }
      } else if (cmd.type === 'setVertexBuffer') {
        const buffer = this.buffers.get(cmd.bufferId);
        if (buffer) {
          pass.setVertexBuffer(cmd.slot, buffer, cmd.offset);
        }
      } else if (cmd.type === 'draw') {
        pass.draw(cmd.vertexCount, cmd.instanceCount, cmd.firstVertex, cmd.firstInstance);
      }
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.frameCount += 1;
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
    if (!this.ring || !this.device || !this.ctx) {
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
    if (!this.device || !this.ctx) {
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

      switch (opcode) {
        case OPCODES.NOP:
          break;
        case OPCODES.BEGIN_FRAME:
          this._resetFrameState();
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
        default:
          break;
      }

      offset = payloadOffset + payloadBytes;
    }

    return { frameCount: this.frameCount, presents };
  }

  execute(input) {
    return this.executeCommandBuffer(input);
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

  async function defaultInitGpu(offscreen, width, height) {
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
    });
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
          payload.height || 720
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
        canvas.width = Math.max(1, payload.width | 0);
        canvas.height = Math.max(1, payload.height | 0);
      } else if (payload.attachRing) {
        attachRingFromMessage(payload);
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
  OPCODES,
  SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
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
