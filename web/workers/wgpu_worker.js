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

  _destroyTextureIfPresent(id) {
    const existing = this.textures.get(id);
    if (existing) {
      if (typeof existing.destroy === 'function') {
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
    if (texture && typeof texture.createView === 'function') {
      this.textureViews.set(id, texture.createView());
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
  }

  _handleCreateComputePipeline(view, payloadOffset) {
    const id = view.getUint32(payloadOffset, true);
    const csShaderId = view.getUint32(payloadOffset + 4, true);
    // Reserved flags at payloadOffset + 8 and reserved at payloadOffset + 12.

    const csEntry = this.shaderModules.get(csShaderId);
    if (!csEntry) return;

    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: csEntry.module,
        entryPoint: 'cs_main',
      },
    });
    this.computePipelines.set(id, pipeline);
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
    if (!pipeline || typeof pipeline.getBindGroupLayout !== 'function') return;

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

    const layout = pipeline.getBindGroupLayout(groupIndex);
    const bindGroup = this.device.createBindGroup({
      layout,
      entries,
    });
    this.bindGroups.set(id, bindGroup);
    this._bindGroupKinds.set(id, computePipeline && !renderPipeline ? 'compute' : 'render');
  }

  _handleDestroyResource(id) {
    this._destroyBufferIfPresent(id);
    this._destroyTextureIfPresent(id);
    this.samplers.delete(id);
    this.bindGroups.delete(id);
    this._bindGroupKinds.delete(id);
    this.shaderModules.delete(id);
    this.pipelines.delete(id);
    this.computePipelines.delete(id);
  }

  _flushPresent() {
    const encoder = this.device.createCommandEncoder();
    const swapchainView = this.ctx.getCurrentTexture().createView();
    let activePass = null;
    let activePassKind = null;
    let renderPassOpened = false;

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
      activePass = encoder.beginRenderPass({
        colorAttachments: [attachment],
      });
      activePassKind = 'render';
      renderPassOpened = true;
      return activePass;
    };

    const ensureComputePass = () => {
      if (activePassKind === 'compute') return activePass;
      endActivePass();
      activePass = encoder.beginComputePass();
      activePassKind = 'compute';
      return activePass;
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
          break;
        }
        default:
          break;
      }
    }

    endActivePass();
    if (!renderPassOpened) {
      ensureRenderPass();
      endActivePass();
    }

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
