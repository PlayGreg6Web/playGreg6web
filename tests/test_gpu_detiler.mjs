import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const detilerMod = require('../web/workers/gpu_detiler.js');

const {
  WEBGPU_BYTES_PER_ROW_ALIGNMENT,
  DETILE_WORKGROUP_SIZE_X,
  DETILE_WORKGROUP_SIZE_Y,
  DETILE_UNIFORM_U32_COUNT,
  DETILE_UNIFORM_BYTE_SIZE,
  GFX10_SWIZZLE_MODE,
  BUFFER_USAGE,
  SHADER_STAGE,
  bppToLog2,
  alignUp,
  alignBytesPerRow,
  decodeD3d12Layout,
  isGfx10SwizzleSupported,
  getGfx10TileShape2D,
  applyGfx10Xor64k,
  computeGfx10TiledByteOffset2D,
  computeGfx10TiledFootprint2D,
  computeDetileDispatchGrid,
  detile2DReference,
  tile2DReference,
  buildGfx10DetileWgsl,
  GpuTextureDetiler,
} = detilerMod;

// Deterministic synthetic byte generator (matches tests/test_gfx10_detile.cpp xorshift32 pattern).
function makeSyntheticLinear(size, seed) {
  const data = new Uint8Array(size);
  let x = (seed ^ 0x9e3779b9) >>> 0;
  for (let i = 0; i < size; i += 1) {
    x = (x ^ ((x << 13) >>> 0)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ ((x << 5) >>> 0)) >>> 0;
    data[i] = (x ^ Math.imul(i, 131)) & 0xff;
  }
  return data;
}

/**
 * Faithful software interpreter of the exact WGSL compute kernel in `buildGfx10DetileWgsl()`.
 * Runs every thread `(gid.x, gid.y)` across `(workgroupsX * 8, workgroupsY * 8)`, verifies that
 * ZERO `dstWords[idx]` indices are ever written by more than one thread (proving race-freedom
 * for 1B, 2B, 4B, 8B, and 16B formats!), and writes exact u32 words into `dstWords`.
 */
function executeWgslDetileKernel(uniformU32, srcU32, dstU32, workgroupsX, workgroupsY) {
  const blocksX = uniformU32[0] >>> 0;
  const blocksY = uniformU32[1] >>> 0;
  const bytesPerBlock = uniformU32[2] >>> 0;
  const pitchInBlocks = uniformU32[3] >>> 0;
  const swizzleMode = uniformU32[4] >>> 0;
  const pipeBankXor = uniformU32[5] >>> 0;
  const srcByteOffset = uniformU32[6] >>> 0;
  const dstByteOffset = uniformU32[7] >>> 0;
  const dstRowPitchBytes = uniformU32[8] >>> 0;
  const srcByteLimit = uniformU32[9] >>> 0;
  const paramTilesPerRow = uniformU32[10] >>> 0;

  function readSrcByte(absOff) {
    if (absOff >= srcByteLimit) return 0;
    const wordIdx = absOff >>> 2;
    if (wordIdx >= srcU32.length) return 0;
    return (srcU32[wordIdx] >>> ((absOff & 3) * 8)) & 0xff;
  }

  function readSrcU16(absOff) {
    if (srcByteLimit < 2 || absOff > srcByteLimit - 2) return 0;
    const byteRem = absOff & 3;
    const wordIdx = absOff >>> 2;
    if (byteRem <= 2) {
      if (wordIdx >= srcU32.length) return 0;
      return (srcU32[wordIdx] >>> (byteRem * 8)) & 0xffff;
    }
    if (wordIdx + 1 >= srcU32.length) return 0;
    const lo = (srcU32[wordIdx] >>> 24) & 0xff;
    const hi = srcU32[wordIdx + 1] & 0xff;
    return (lo | (hi << 8)) >>> 0;
  }

  function readSrcU32(absOff) {
    if (srcByteLimit < 4 || absOff > srcByteLimit - 4) return 0;
    const wordIdx = absOff >>> 2;
    const byteRem = absOff & 3;
    if (byteRem === 0) {
      if (wordIdx >= srcU32.length) return 0;
      return srcU32[wordIdx] >>> 0;
    }
    if (wordIdx + 1 >= srcU32.length) return 0;
    const lo = srcU32[wordIdx] >>> 0;
    const hi = srcU32[wordIdx + 1] >>> 0;
    const shift = byteRem * 8;
    return ((lo >>> shift) | ((hi << (32 - shift)) >>> 0)) >>> 0;
  }

  function computeIntra256Wgsl(isDisplay, bpb, lx, ly) {
    if (bpb === 1) {
      if (!isDisplay) {
        return ((lx & 0xf) | ((ly & 0xf) << 4)) >>> 0;
      }
      return (
        (lx & 7) |
        (((ly >>> 1) & 1) << 3) |
        ((ly & 1) << 4) |
        (((ly >>> 2) & 1) << 5) |
        (((lx >>> 3) & 1) << 6) |
        (((ly >>> 3) & 1) << 7)
      ) >>> 0;
    }
    if (bpb === 2) {
      return (((lx & 7) << 1) | ((ly & 7) << 4) | (((lx >>> 3) & 1) << 7)) >>> 0;
    }
    if (bpb === 4) {
      if (!isDisplay) {
        return (((lx & 3) << 2) | ((ly & 7) << 4) | (((lx >>> 2) & 1) << 7)) >>> 0;
      }
      return (
        ((lx & 3) << 2) |
        ((ly & 3) << 4) |
        (((lx >>> 2) & 1) << 6) |
        (((ly >>> 2) & 1) << 7)
      ) >>> 0;
    }
    if (bpb === 8) {
      if (!isDisplay) {
        return (((lx & 1) << 3) | ((ly & 3) << 4) | (((lx >>> 1) & 3) << 6)) >>> 0;
      }
      return (
        ((lx & 1) << 3) |
        ((ly & 1) << 4) |
        (((lx >>> 1) & 3) << 5) |
        (((ly >>> 1) & 1) << 7)
      ) >>> 0;
    }
    if (!isDisplay) {
      return (((ly & 3) << 4) | ((lx & 3) << 6)) >>> 0;
    }
    return (
      ((lx & 1) << 4) |
      ((ly & 1) << 5) |
      (((lx >>> 1) & 1) << 6) |
      (((ly >>> 1) & 1) << 7)
    ) >>> 0;
  }

  function computeTiledOffsetWgsl(bx, by) {
    if (swizzleMode === 0) {
      return ((by * pitchInBlocks + bx) * bytesPerBlock) >>> 0;
    }
    const isDisplay =
      swizzleMode === 2 ||
      swizzleMode === 6 ||
      swizzleMode === 10 ||
      swizzleMode === 14 ||
      swizzleMode === 26 ||
      swizzleMode === 30;
    const isXor =
      swizzleMode === 25 || swizzleMode === 26 || swizzleMode === 29 || swizzleMode === 30;
    let blkSizeLog2 = 16;
    if (swizzleMode === 1 || swizzleMode === 2) {
      blkSizeLog2 = 8;
    } else if (swizzleMode === 5 || swizzleMode === 6) {
      blkSizeLog2 = 12;
    }
    let w256 = 2;
    let h256 = 2;
    if (bytesPerBlock === 1) {
      w256 = 4;
      h256 = 4;
    } else if (bytesPerBlock === 2) {
      w256 = 4;
      h256 = 3;
    } else if (bytesPerBlock === 4) {
      w256 = 3;
      h256 = 3;
    } else if (bytesPerBlock === 8) {
      w256 = 3;
      h256 = 2;
    }
    const extraLog2 = (blkSizeLog2 - 8) >>> 1;
    const wLog2 = w256 + extraLog2;
    const hLog2 = h256 + extraLog2;
    const tileWMask = (1 << wLog2) - 1;
    const tileHMask = (1 << hLog2) - 1;

    const tileX = bx >>> wLog2;
    const tileY = by >>> hLog2;
    const lx = bx & tileWMask;
    const ly = by & tileHMask;

    let intra = computeIntra256Wgsl(isDisplay, bytesPerBlock, lx, ly);
    if (blkSizeLog2 >= 12) {
      const hx = lx >>> w256;
      const hy = ly >>> h256;
      intra =
        (intra |
          ((hy & 1) << 8) |
          ((hx & 1) << 9) |
          (((hy >>> 1) & 1) << 10) |
          (((hx >>> 1) & 1) << 11)) >>>
        0;
      if (blkSizeLog2 === 16) {
        intra =
          (intra |
            (((hy >>> 2) & 1) << 12) |
            (((hx >>> 2) & 1) << 13) |
            (((hy >>> 3) & 1) << 14) |
            (((hx >>> 3) & 1) << 15)) >>>
          0;
      }
    }
    if (isXor) {
      const hi = (intra >>> 12) & 0xf;
      const rev = ((hi & 1) << 3) | ((hi & 2) << 1) | ((hi & 4) >>> 1) | ((hi & 8) >>> 3);
      const seed = (pipeBankXor & 0xff) << 8;
      intra = (intra ^ (rev << 8) ^ seed) >>> 0;
    }
    let tilesPerRow = paramTilesPerRow;
    if (tilesPerRow === 0) {
      const tileW = 1 << wLog2;
      const effPitch = pitchInBlocks > 0 ? pitchInBlocks : tileW;
      tilesPerRow = (effPitch + tileW - 1) >>> wLog2;
    }
    const tileIndex = tileY * tilesPerRow + tileX;
    return (((tileIndex << blkSizeLog2) >>> 0) + intra) >>> 0;
  }

  const writtenWordIndices = new Set();
  let duplicateWrites = 0;

  function writeDstU32(wordIdx, val) {
    if (wordIdx < dstU32.length) {
      if (writtenWordIndices.has(wordIdx)) {
        duplicateWrites += 1;
      }
      writtenWordIndices.add(wordIdx);
      dstU32[wordIdx] = val >>> 0;
    }
  }

  const totalX = workgroupsX * DETILE_WORKGROUP_SIZE_X;
  const totalY = workgroupsY * DETILE_WORKGROUP_SIZE_Y;

  for (let gidY = 0; gidY < totalY; gidY += 1) {
    for (let gidX = 0; gidX < totalX; gidX += 1) {
      const by = gidY;
      let bx = gidX;
      if (bytesPerBlock === 1) bx = gidX * 4;
      else if (bytesPerBlock === 2) bx = gidX * 2;
      if (bx >= blocksX || by >= blocksY) continue;

      const dstRowWordBase = (dstByteOffset >>> 2) + by * (dstRowPitchBytes >>> 2);
      if (bytesPerBlock === 1) {
        let packed = 0;
        for (let i = 0; i < 4; i += 1) {
          const cx = bx + i;
          if (cx < blocksX) {
            const off = computeTiledOffsetWgsl(cx, by);
            const bVal = readSrcByte(srcByteOffset + off);
            packed = (packed | ((bVal << (i * 8)) >>> 0)) >>> 0;
          }
        }
        writeDstU32(dstRowWordBase + gidX, packed);
      } else if (bytesPerBlock === 2) {
        let packed = 0;
        for (let i = 0; i < 2; i += 1) {
          const cx = bx + i;
          if (cx < blocksX) {
            const off = computeTiledOffsetWgsl(cx, by);
            const hVal = readSrcU16(srcByteOffset + off);
            packed = (packed | ((hVal << (i * 16)) >>> 0)) >>> 0;
          }
        }
        writeDstU32(dstRowWordBase + gidX, packed);
      } else if (bytesPerBlock === 4) {
        const off = computeTiledOffsetWgsl(bx, by);
        writeDstU32(dstRowWordBase + bx, readSrcU32(srcByteOffset + off));
      } else if (bytesPerBlock === 8) {
        const off = computeTiledOffsetWgsl(bx, by);
        const srcBase = srcByteOffset + off;
        const dstWord = dstRowWordBase + bx * 2;
        writeDstU32(dstWord, readSrcU32(srcBase));
        writeDstU32(dstWord + 1, readSrcU32(srcBase + 4));
      } else {
        const off = computeTiledOffsetWgsl(bx, by);
        const srcBase = srcByteOffset + off;
        const dstWord = dstRowWordBase + bx * 4;
        writeDstU32(dstWord, readSrcU32(srcBase));
        writeDstU32(dstWord + 1, readSrcU32(srcBase + 4));
        writeDstU32(dstWord + 2, readSrcU32(srcBase + 8));
        writeDstU32(dstWord + 3, readSrcU32(srcBase + 12));
      }
    }
  }

  return { writtenWords: writtenWordIndices.size, duplicateWrites };
}

/**
 * Mock WebGPU compute device & encoder harness that records every resource creation,
 * uniform upload, bind group, compute dispatch, and copyBufferToTexture call, and optionally
 * executes `executeWgslDetileKernel` live during `dispatchWorkgroups`!
 */
function createMockWebGpuCompute({
  executeKernel = true,
  withCopyBufferToTexture = true,
  limits = {},
} = {}) {
  const shaderModules = [];
  const bindGroupLayouts = [];
  const pipelineLayouts = [];
  const pipelines = [];
  const buffers = [];
  const bindGroups = [];
  const writeBufferCalls = [];
  const dispatches = [];
  const copyBufferToTextureCalls = [];

  const device = {
    limits: {
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      ...limits,
    },
    queue: {
      writeBuffer(buffer, bufferOffset, data) {
        const srcBytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        const dstBytes = new Uint8Array(buffer._storage, bufferOffset, srcBytes.byteLength);
        dstBytes.set(srcBytes);
        writeBufferCalls.push({
          buffer,
          bufferOffset,
          byteLength: srcBytes.byteLength,
          words: Array.from(new Uint32Array(srcBytes.slice().buffer)),
        });
      },
    },
    createShaderModule(desc) {
      const mod = { kind: 'GPUShaderModule', label: desc.label, code: desc.code };
      shaderModules.push(mod);
      return mod;
    },
    createBindGroupLayout(desc) {
      const bgl = { kind: 'GPUBindGroupLayout', label: desc.label, entries: desc.entries.slice() };
      bindGroupLayouts.push(bgl);
      return bgl;
    },
    createPipelineLayout(desc) {
      const pl = { kind: 'GPUPipelineLayout', label: desc.label, bindGroupLayouts: desc.bindGroupLayouts.slice() };
      pipelineLayouts.push(pl);
      return pl;
    },
    createComputePipeline(desc) {
      const p = { kind: 'GPUComputePipeline', label: desc.label, layout: desc.layout, compute: { ...desc.compute } };
      pipelines.push(p);
      return p;
    },
    createBuffer(desc) {
      const storage = new ArrayBuffer(desc.size);
      const buf = {
        kind: 'GPUBuffer',
        label: desc.label || '',
        size: desc.size,
        usage: desc.usage,
        _storage: storage,
        destroyed: false,
        destroyCalls: 0,
        destroy() {
          this.destroyed = true;
          this.destroyCalls += 1;
        },
      };
      buffers.push(buf);
      return buf;
    },
    createBindGroup(desc) {
      const bg = {
        kind: 'GPUBindGroup',
        label: desc.label,
        layout: desc.layout,
        entries: desc.entries.map((e) => ({ binding: e.binding, resource: { buffer: e.resource.buffer } })),
      };
      bindGroups.push(bg);
      return bg;
    },
    createCommandEncoder() {
      const encoder = {
        beginComputePass(desc = {}) {
          let boundPipeline = null;
          let boundGroup = null;
          return {
            setPipeline(pipeline) {
              boundPipeline = pipeline;
            },
            setBindGroup(index, bindGroup) {
              assert.equal(index, 0);
              boundGroup = bindGroup;
            },
            dispatchWorkgroups(wx, wy, wz = 1) {
              assert.ok(boundPipeline, 'pipeline must be bound before dispatch');
              assert.ok(boundGroup, 'bindGroup must be bound before dispatch');
              const uniformBuf = boundGroup.entries[0].resource.buffer;
              const srcBuf = boundGroup.entries[1].resource.buffer;
              const dstBuf = boundGroup.entries[2].resource.buffer;
              let kernelResult = null;
              if (executeKernel) {
                kernelResult = executeWgslDetileKernel(
                  new Uint32Array(uniformBuf._storage, 0, DETILE_UNIFORM_U32_COUNT),
                  new Uint32Array(srcBuf._storage),
                  new Uint32Array(dstBuf._storage),
                  wx,
                  wy
                );
              }
              dispatches.push({
                label: desc.label,
                pipeline: boundPipeline,
                bindGroup: boundGroup,
                workgroupsX: wx,
                workgroupsY: wy,
                workgroupsZ: wz,
                kernelResult,
              });
            },
            end() {},
          };
        },
      };
      if (withCopyBufferToTexture) {
        encoder.copyBufferToTexture = (source, destination, copySize) => {
          copyBufferToTextureCalls.push({
            source: { ...source },
            destination: {
              texture: destination.texture,
              mipLevel: destination.mipLevel,
              origin: { ...destination.origin },
              aspect: destination.aspect,
            },
            copySize: { ...copySize },
          });
        };
      }
      return encoder;
    },
  };

  return {
    device,
    shaderModules,
    bindGroupLayouts,
    pipelineLayouts,
    pipelines,
    buffers,
    bindGroups,
    writeBufferCalls,
    dispatches,
    copyBufferToTextureCalls,
  };
}

test('UMD export attaches PlayGta6GpuDetiler on globalThis and exposes constants', () => {
  assert.equal(globalThis.PlayGta6GpuDetiler.GpuTextureDetiler, GpuTextureDetiler);
  assert.equal(globalThis.PlayGta6GpuDetiler.computeGfx10TiledByteOffset2D, computeGfx10TiledByteOffset2D);
  assert.equal(globalThis.PlayGta6GpuDetiler.detile2DReference, detile2DReference);
  assert.equal(globalThis.PlayGta6GpuDetiler.buildGfx10DetileWgsl, buildGfx10DetileWgsl);

  assert.equal(WEBGPU_BYTES_PER_ROW_ALIGNMENT, 256);
  assert.equal(DETILE_WORKGROUP_SIZE_X, 8);
  assert.equal(DETILE_WORKGROUP_SIZE_Y, 8);
  assert.equal(DETILE_UNIFORM_U32_COUNT, 12);
  assert.equal(DETILE_UNIFORM_BYTE_SIZE, 48);

  assert.equal(GFX10_SWIZZLE_MODE.LINEAR, 0);
  assert.equal(GFX10_SWIZZLE_MODE.SW_256B_S, 1);
  assert.equal(GFX10_SWIZZLE_MODE.SW_256B_D, 2);
  assert.equal(GFX10_SWIZZLE_MODE.SW_4KB_S, 5);
  assert.equal(GFX10_SWIZZLE_MODE.SW_4KB_D, 6);
  assert.equal(GFX10_SWIZZLE_MODE.SW_64KB_S, 9);
  assert.equal(GFX10_SWIZZLE_MODE.SW_64KB_D, 10);
  assert.equal(GFX10_SWIZZLE_MODE.SW_64KB_S_X, 25);
  assert.equal(GFX10_SWIZZLE_MODE.SW_64KB_D_X, 26);
  assert.equal(GFX10_SWIZZLE_MODE.UNSUPPORTED, 255);

  assert.equal(bppToLog2(1), 0);
  assert.equal(bppToLog2(2), 1);
  assert.equal(bppToLog2(4), 2);
  assert.equal(bppToLog2(8), 3);
  assert.equal(bppToLog2(16), 4);
  assert.equal(bppToLog2(3), -1);
  assert.equal(bppToLog2(12), -1);
  assert.equal(alignBytesPerRow(1), 256);
  assert.equal(alignBytesPerRow(256), 256);
  assert.equal(alignBytesPerRow(257), 512);
  assert.equal(alignUp(100, 4), 100);
  assert.equal(alignUp(101, 4), 104);
  assert.equal(BUFFER_USAGE.STORAGE, 0x0080);
  assert.equal(SHADER_STAGE.COMPUTE, 0x0004);
});

test('decodeD3d12Layout matches runtime/gpu/gfx10_detile.cpp for all supported and unsupported layouts', () => {
  assert.equal(decodeD3d12Layout(0), GFX10_SWIZZLE_MODE.LINEAR);
  assert.equal(decodeD3d12Layout(1), GFX10_SWIZZLE_MODE.LINEAR);
  assert.equal(decodeD3d12Layout(0x100), GFX10_SWIZZLE_MODE.LINEAR);
  assert.equal(decodeD3d12Layout(0x101), GFX10_SWIZZLE_MODE.SW_256B_S);
  assert.equal(decodeD3d12Layout(0x102), GFX10_SWIZZLE_MODE.SW_256B_D);
  assert.equal(decodeD3d12Layout(0x105), GFX10_SWIZZLE_MODE.SW_4KB_S);
  assert.equal(decodeD3d12Layout(0x106), GFX10_SWIZZLE_MODE.SW_4KB_D);
  assert.equal(decodeD3d12Layout(0x109), GFX10_SWIZZLE_MODE.SW_64KB_S);
  assert.equal(decodeD3d12Layout(0x10a), GFX10_SWIZZLE_MODE.SW_64KB_D);
  assert.equal(decodeD3d12Layout(0x10d), GFX10_SWIZZLE_MODE.SW_VAR_S);
  assert.equal(decodeD3d12Layout(0x10e), GFX10_SWIZZLE_MODE.SW_VAR_D);
  assert.equal(decodeD3d12Layout(0x119), GFX10_SWIZZLE_MODE.SW_64KB_S_X);
  assert.equal(decodeD3d12Layout(0x11a), GFX10_SWIZZLE_MODE.SW_64KB_D_X);
  assert.equal(decodeD3d12Layout(0x11d), GFX10_SWIZZLE_MODE.SW_VAR_S_X);
  assert.equal(decodeD3d12Layout(0x11e), GFX10_SWIZZLE_MODE.SW_VAR_D_X);

  // Unrecognized, _R, or out-of-range layouts must return UNSUPPORTED.
  for (const bad of [-1, 2, 3, 0x103, 0x104, 0x107, 0x108, 0x10b, 0x110, 0x118, 0x11b, 0x120, 0x200, 0xffffffff]) {
    assert.equal(decodeD3d12Layout(bad), GFX10_SWIZZLE_MODE.UNSUPPORTED, `layout 0x${bad.toString(16)}`);
  }

  assert.equal(isGfx10SwizzleSupported(GFX10_SWIZZLE_MODE.LINEAR), true);
  assert.equal(isGfx10SwizzleSupported(GFX10_SWIZZLE_MODE.SW_64KB_D_X), true);
  assert.equal(isGfx10SwizzleSupported(GFX10_SWIZZLE_MODE.UNSUPPORTED), false);
  assert.equal(isGfx10SwizzleSupported(3), false);
});

test('computeGfx10TiledByteOffset2D matches exact golden AMD addrlib RB+ vectors from test_gfx10_detile.cpp', () => {
  // Exact golden vectors from test_r32g32b32a32_2x2_sw256bd():
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 0, 0, 16, 2), 0);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 1, 0, 16, 2), 16);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 0, 1, 16, 2), 32);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 1, 1, 16, 2), 48);

  // Exact golden vectors from test_addrlib_equations_2d():
  // SW_256B_D / SW_256B_S, 4 B elements:
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 1, 1, 4, 8), 4 + 16);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 2, 0, 4, 8), 8);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 0, 2, 4, 8), 32);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 4, 0, 4, 8), 64);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 0, 4, 4, 8), 128);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_S, 4, 0, 4, 8), 128);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_S, 0, 4, 4, 8), 64);

  // SW_256B_D, 1 B: X0 X1 X2 Y1 Y0 Y2 X3 Y3
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 0, 1, 1, 16), 16);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 0, 2, 1, 16), 8);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 8, 0, 1, 16), 64);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 0, 8, 1, 16), 128);

  // SW_256B_D, 8 B: 0 0 0 X0 Y0 X1 X2 Y1
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 1, 0, 8, 8), 8);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 0, 1, 8, 8), 16);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 2, 0, 8, 8), 32);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 4, 0, 8, 8), 64);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 0, 2, 8, 8), 128);

  // SW_4KB_D, 16 B: 0 0 0 0 X0 Y0 X1 Y1 Y2 X2 Y3 X3
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_4KB_D, 0, 4, 16, 16), 256);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_4KB_D, 4, 0, 16, 16), 512);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_4KB_D, 0, 8, 16, 16), 1024);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_4KB_D, 8, 0, 16, 16), 2048);

  // SW_64KB_D, 4 B: 0 0 X0 X1 Y0 Y1 X2 Y2 Y3 X3 Y4 X4 Y5 X5 Y6 X6
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_64KB_D, 4, 4, 4, 128), 64 + 128);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_64KB_D, 0, 8, 4, 128), 256);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_64KB_D, 8, 0, 4, 128), 512);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_64KB_D, 0, 64, 4, 128), 16384);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_64KB_D, 64, 0, 4, 128), 32768);

  // SW_4KB_S, 4 B: 0 0 X0 X1 Y0 Y1 Y2 X2 Y3 X3 Y4 X4
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_4KB_S, 0, 8, 4, 32), 256);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_4KB_S, 8, 0, 4, 32), 512);

  // Non-XOR modes ignore pipeBankXor completely.
  assert.equal(
    computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_64KB_S, 5, 7, 4, 128, 0x5a),
    computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_64KB_S, 5, 7, 4, 128, 0)
  );
  assert.equal(
    computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_64KB_D, 5, 7, 16, 64, 0xa5),
    computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_64KB_D, 5, 7, 16, 64, 0)
  );

  // Invalid inputs return -1.
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.UNSUPPORTED, 0, 0, 4, 16), -1);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, 0, 0, 3, 16), -1);
  assert.equal(computeGfx10TiledByteOffset2D(GFX10_SWIZZLE_MODE.SW_256B_D, -1, 0, 4, 16), -1);
});

test('complete single-tile bijectivity across all supported modes, all block sizes, and pipeBankXor seeds', () => {
  const modes = [
    GFX10_SWIZZLE_MODE.SW_256B_S,
    GFX10_SWIZZLE_MODE.SW_256B_D,
    GFX10_SWIZZLE_MODE.SW_4KB_S,
    GFX10_SWIZZLE_MODE.SW_4KB_D,
    GFX10_SWIZZLE_MODE.SW_64KB_S,
    GFX10_SWIZZLE_MODE.SW_64KB_D,
    GFX10_SWIZZLE_MODE.SW_64KB_S_X,
    GFX10_SWIZZLE_MODE.SW_64KB_D_X,
  ];
  const bpps = [1, 2, 4, 8, 16];

  for (const mode of modes) {
    const isXor = mode === GFX10_SWIZZLE_MODE.SW_64KB_S_X || mode === GFX10_SWIZZLE_MODE.SW_64KB_D_X;
    const seeds = isXor ? [0, 0x05, 0x3c, 0xa7, 0xff] : [0];

    for (const pbx of seeds) {
      for (const bpp of bpps) {
        const shape = getGfx10TileShape2D(mode, bpp);
        assert.equal(shape.valid, true);
        const elemCount = shape.tileW * shape.tileH;
        const seen = new Uint8Array(elemCount);

        for (let y = 0; y < shape.tileH; y += 1) {
          for (let x = 0; x < shape.tileW; x += 1) {
            const off = computeGfx10TiledByteOffset2D(mode, x, y, bpp, shape.tileW, pbx);
            assert.ok(off >= 0 && off < shape.blockBytes, `offset ${off} out of tile range`);
            assert.equal(off % bpp, 0, `offset ${off} not aligned to ${bpp}`);
            const slot = off / bpp;
            assert.equal(seen[slot], 0, `duplicate slot ${slot} for mode=${mode} bpp=${bpp} pbx=${pbx}`);
            seen[slot] = 1;
          }
        }
        for (let i = 0; i < elemCount; i += 1) {
          assert.equal(seen[i], 1);
        }

        // For _X modes with non-zero pbx: origin moves and low 256B pipe interleave bits stay identical.
        if (isXor && pbx !== 0) {
          const off0 = computeGfx10TiledByteOffset2D(mode, 0, 0, bpp, shape.tileW, 0);
          const offx = computeGfx10TiledByteOffset2D(mode, 0, 0, bpp, shape.tileW, pbx);
          assert.notEqual(offx, off0);
          assert.equal(offx & 0xff, off0 & 0xff);
        }
      }
    }
  }

  // Verify applyGfx10Xor64k matches zero-seed involution property.
  for (let addr = 0; addr < 65536; addr += 257) {
    const x1 = applyGfx10Xor64k(addr, 0);
    const x2 = applyGfx10Xor64k(x1, 0);
    assert.equal(x2, addr);
  }
});

test('detile2DReference round-trips all modes, block sizes (1/2/4/8/16B), NPOT sizes, custom pitches, offsets, and 256B row alignment', () => {
  const cases = [
    // R8 (1B), NPOT width & height, 256B-aligned dstRowPitchBytes
    { mode: GFX10_SWIZZLE_MODE.SW_256B_S, bx: 17, by: 13, bpb: 1, pbx: 0, align256: true },
    { mode: GFX10_SWIZZLE_MODE.SW_256B_D, bx: 35, by: 19, bpb: 1, pbx: 0, align256: true },
    { mode: GFX10_SWIZZLE_MODE.SW_4KB_S, bx: 67, by: 35, bpb: 1, pbx: 0, align256: false },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_D_X, bx: 50, by: 30, bpb: 1, pbx: 0x6d, align256: true },
    // R16 / RG8 (2B), odd widths
    { mode: GFX10_SWIZZLE_MODE.SW_256B_D, bx: 19, by: 11, bpb: 2, pbx: 0, align256: true },
    { mode: GFX10_SWIZZLE_MODE.SW_4KB_D, bx: 67, by: 35, bpb: 2, pbx: 0, align256: false },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_S_X, bx: 45, by: 23, bpb: 2, pbx: 0xa5, align256: true },
    // RGBA8 (4B), NPOT & custom pitchInBlocks
    { mode: GFX10_SWIZZLE_MODE.LINEAR, bx: 20, by: 10, bpb: 4, pitch: 64, pbx: 0, align256: true },
    { mode: GFX10_SWIZZLE_MODE.SW_256B_S, bx: 33, by: 17, bpb: 4, pbx: 0, align256: false },
    { mode: GFX10_SWIZZLE_MODE.SW_4KB_S, bx: 128, by: 72, bpb: 4, pbx: 0, align256: true },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_D, bx: 100, by: 60, bpb: 4, pbx: 0, align256: false },
    // BC1 / BC4 / RGBA16F (8B)
    { mode: GFX10_SWIZZLE_MODE.SW_256B_D, bx: 25, by: 15, bpb: 8, pbx: 0, align256: true },
    { mode: GFX10_SWIZZLE_MODE.SW_4KB_D, bx: 37, by: 19, bpb: 8, pbx: 0, align256: false },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_S_X, bx: 48, by: 24, bpb: 8, pbx: 0x3b, align256: true },
    // BC2 / BC3 / BC5 / BC6H / BC7 / RGBA32F (16B)
    { mode: GFX10_SWIZZLE_MODE.SW_256B_D, bx: 2, by: 2, bpb: 16, pbx: 0, align256: true },
    { mode: GFX10_SWIZZLE_MODE.SW_4KB_S, bx: 25, by: 15, bpb: 16, pbx: 0, align256: true },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_S, bx: 32, by: 16, bpb: 16, pbx: 0, align256: false },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_D_X, bx: 27, by: 19, bpb: 16, pbx: 0xc9, align256: true },
  ];

  let seed = 4200;
  for (const c of cases) {
    const tightRow = c.bx * c.bpb;
    const linearIn = makeSyntheticLinear(tightRow * c.by, seed++);
    const srcByteOffset = 128;
    const dstByteOffset = 64;
    const dstRowPitchBytes = c.align256 ? alignBytesPerRow(tightRow) : tightRow;

    const footprint = computeGfx10TiledFootprint2D(c.mode, c.bx, c.by, c.bpb, c.pitch || 0);
    assert.ok(footprint > 0);
    const tiledBuf = new Uint8Array(srcByteOffset + footprint + 37);
    tiledBuf.fill(0xcc);

    const tileRes = tile2DReference(
      {
        swizzleMode: c.mode,
        blocksX: c.bx,
        blocksY: c.by,
        bytesPerBlock: c.bpb,
        pitchInBlocks: c.pitch || 0,
        pipeBankXor: c.pbx,
        tiledByteOffset: srcByteOffset,
      },
      linearIn,
      tiledBuf
    );
    assert.equal(tileRes.ok, true);

    const detiledBuf = new Uint8Array(dstByteOffset + dstRowPitchBytes * c.by);
    detiledBuf.fill(0xee);
    const detileRes = detile2DReference(
      {
        swizzleMode: c.mode,
        blocksX: c.bx,
        blocksY: c.by,
        bytesPerBlock: c.bpb,
        pitchInBlocks: c.pitch || 0,
        pipeBankXor: c.pbx,
        srcByteOffset,
        dstByteOffset,
        dstRowPitchBytes,
        srcByteLimit: srcByteOffset + footprint,
      },
      tiledBuf,
      detiledBuf
    );
    assert.equal(detileRes.ok, true);
    assert.equal(detileRes.rowPitchBytes, dstRowPitchBytes);
    assert.equal(detileRes.bytesWritten, tightRow * c.by);

    for (let y = 0; y < c.by; y += 1) {
      const actualRow = detiledBuf.subarray(
        dstByteOffset + y * dstRowPitchBytes,
        dstByteOffset + y * dstRowPitchBytes + tightRow
      );
      const expectedRow = linearIn.subarray(y * tightRow, (y + 1) * tightRow);
      assert.deepEqual(actualRow, expectedRow, `row ${y} mismatch for mode=${c.mode} bpb=${c.bpb}`);
    }
  }
});

test('WGSL source validation: standard WGSL v1.0, workgroup_size(8, 8, 1), zero vendor extensions, race-free 1B/2B packing, bounds guards', () => {
  const wgsl = buildGfx10DetileWgsl();
  assert.equal(typeof wgsl, 'string');
  assert.ok(wgsl.length > 500);

  // Zero vendor `enable` directives (100% cross-GPU standard WGSL).
  assert.equal(/^\s*enable\s+/m.test(wgsl), false, 'WGSL must not use vendor enable extensions');

  // Zero atomic operations (1B/2B race freedom via word packing, not atomics).
  assert.equal(/\batomic\w*\b/.test(wgsl), false, 'WGSL must not rely on atomics');

  // Required entry point & workgroup size.
  assert.ok(wgsl.includes('@compute @workgroup_size(8, 8, 1)'));
  assert.ok(wgsl.includes('fn main(@builtin(global_invocation_id) gid: vec3<u32>)'));

  // Required storage/uniform bindings.
  assert.ok(wgsl.includes('@group(0) @binding(0) var<uniform> params: DetileParams;'));
  assert.ok(wgsl.includes('@group(0) @binding(1) var<storage, read> srcWords: array<u32>;'));
  assert.ok(wgsl.includes('@group(0) @binding(2) var<storage, read_write> dstWords: array<u32>;'));

  // Required bounds clamping & race-free 4x1B / 2x2B horizontal packing.
  assert.ok(wgsl.includes('if (bx >= params.blocksX || by >= params.blocksY)'));
  assert.ok(wgsl.includes('bx = gid.x * 4u;'));
  assert.ok(wgsl.includes('bx = gid.x * 2u;'));
  assert.ok(wgsl.includes('params.srcByteLimit'));
  assert.ok(wgsl.includes('params.dstRowPitchBytes'));
  assert.ok(wgsl.includes('params.pipeBankXor'));

  // Balanced delimiters sanity check.
  let braceDepth = 0;
  let parenDepth = 0;
  for (const ch of wgsl) {
    if (ch === '{') braceDepth += 1;
    else if (ch === '}') braceDepth -= 1;
    else if (ch === '(') parenDepth += 1;
    else if (ch === ')') parenDepth -= 1;
    assert.ok(braceDepth >= 0, 'unbalanced closing brace in WGSL');
    assert.ok(parenDepth >= 0, 'unbalanced closing paren in WGSL');
  }
  assert.equal(braceDepth, 0, 'unclosed brace in WGSL');
  assert.equal(parenDepth, 0, 'unclosed paren in WGSL');
});

test('simulated WGSL kernel matches detile2DReference bit-for-bit with ZERO write races across 1B/2B/4B/8B/16B & all modes', () => {
  const env = createMockWebGpuCompute({ executeKernel: true });
  const detiler = new GpuTextureDetiler(env.device);

  const testCases = [
    // Odd & NPOT 1B (R8) with 256B-aligned row pitch (tests 4x1B word packing + partial trailing word!)
    { mode: GFX10_SWIZZLE_MODE.SW_256B_S, bx: 1, by: 1, bpb: 1, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_256B_D, bx: 5, by: 7, bpb: 1, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_4KB_S, bx: 19, by: 17, bpb: 1, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_D_X, bx: 67, by: 35, bpb: 1, pbx: 0x5d },
    // Odd & NPOT 2B (R16) with 256B-aligned row pitch (tests 2x2B word packing + partial trailing word!)
    { mode: GFX10_SWIZZLE_MODE.SW_256B_D, bx: 3, by: 5, bpb: 2, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_4KB_D, bx: 33, by: 19, bpb: 2, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_S_X, bx: 41, by: 23, bpb: 2, pbx: 0xb3 },
    // 4B (RGBA8)
    { mode: GFX10_SWIZZLE_MODE.LINEAR, bx: 20, by: 10, bpb: 4, pitch: 64, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_256B_S, bx: 15, by: 11, bpb: 4, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_4KB_D, bx: 37, by: 29, bpb: 4, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_S, bx: 128, by: 64, bpb: 4, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_D_X, bx: 73, by: 41, bpb: 4, pbx: 0x77 },
    // 8B (BC1 / BC4 / RGBA16F)
    { mode: GFX10_SWIZZLE_MODE.SW_256B_D, bx: 13, by: 9, bpb: 8, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_4KB_S, bx: 35, by: 17, bpb: 8, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_D_X, bx: 29, by: 21, bpb: 8, pbx: 0x42 },
    // 16B (BC7 / BC3 / RGBA32F)
    { mode: GFX10_SWIZZLE_MODE.SW_256B_S, bx: 7, by: 5, bpb: 16, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_4KB_D, bx: 25, by: 15, bpb: 16, pbx: 0 },
    { mode: GFX10_SWIZZLE_MODE.SW_64KB_S_X, bx: 19, by: 13, bpb: 16, pbx: 0xe1 },
  ];

  let seed = 9001;
  for (const tc of testCases) {
    const tightRow = tc.bx * tc.bpb;
    const dstRowPitchBytes = alignBytesPerRow(tightRow);
    const srcByteOffset = 256;
    const dstByteOffset = 64;
    const linearIn = makeSyntheticLinear(tightRow * tc.by, seed++);
    const footprint = computeGfx10TiledFootprint2D(tc.mode, tc.bx, tc.by, tc.bpb, tc.pitch || 0);

    const srcBuf = env.device.createBuffer({
      size: alignUp(srcByteOffset + footprint, 4),
      usage: BUFFER_USAGE.STORAGE,
    });
    const dstBuf = env.device.createBuffer({
      size: dstByteOffset + dstRowPitchBytes * tc.by,
      usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
    });

    const tiledRes = tile2DReference(
      {
        swizzleMode: tc.mode,
        blocksX: tc.bx,
        blocksY: tc.by,
        bytesPerBlock: tc.bpb,
        pitchInBlocks: tc.pitch || 0,
        pipeBankXor: tc.pbx,
        tiledByteOffset: srcByteOffset,
      },
      linearIn,
      new Uint8Array(srcBuf._storage)
    );
    assert.equal(tiledRes.ok, true);

    const encoder = env.device.createCommandEncoder();
    const res = detiler.detileBufferToBuffer(encoder, {
      srcBuffer: srcBuf,
      dstBuffer: dstBuf,
      swizzleMode: tc.mode,
      blocksX: tc.bx,
      blocksY: tc.by,
      bytesPerBlock: tc.bpb,
      pitchInBlocks: tc.pitch || 0,
      pipeBankXor: tc.pbx,
      srcByteOffset,
      dstByteOffset,
      dstRowPitchBytes,
      srcByteLimit: srcByteOffset + footprint,
    });

    assert.equal(res.ok, true);
    assert.equal(res.fallback, false);

    const lastDispatch = env.dispatches[env.dispatches.length - 1];
    assert.equal(
      lastDispatch.kernelResult.duplicateWrites,
      0,
      `DATA RACE detected for mode=${tc.mode} bpb=${tc.bpb} bx=${tc.bx}`
    );

    const gpuOut = new Uint8Array(dstBuf._storage);
    const refRes = detile2DReference(
      {
        swizzleMode: tc.mode,
        blocksX: tc.bx,
        blocksY: tc.by,
        bytesPerBlock: tc.bpb,
        pitchInBlocks: tc.pitch || 0,
        pipeBankXor: tc.pbx,
        srcByteOffset,
        dstByteOffset,
        dstRowPitchBytes,
        srcByteLimit: srcByteOffset + footprint,
      },
      new Uint8Array(srcBuf._storage)
    );
    assert.equal(refRes.ok, true);

    for (let y = 0; y < tc.by; y += 1) {
      const rowStart = dstByteOffset + y * dstRowPitchBytes;
      const actual = gpuOut.subarray(rowStart, rowStart + tightRow);
      const expected = linearIn.subarray(y * tightRow, (y + 1) * tightRow);
      const refRow = refRes.dstBytes.subarray(rowStart, rowStart + tightRow);
      assert.deepEqual(actual, expected, `GPU vs linear mismatch mode=${tc.mode} bpb=${tc.bpb} row=${y}`);
      assert.deepEqual(actual, refRow, `GPU vs reference mismatch mode=${tc.mode} bpb=${tc.bpb} row=${y}`);
    }
  }

  // Pipeline, shader module, BGL, layout, and uniform buffer compiled lazily ONCE across all 18 dispatches!
  assert.equal(env.shaderModules.length, 1);
  assert.equal(env.bindGroupLayouts.length, 1);
  assert.equal(env.pipelineLayouts.length, 1);
  assert.equal(env.pipelines.length, 1);
  assert.equal(detiler.getStats().gpuDetileDispatches, testCases.length);
  assert.equal(detiler.getStats().gpuDetileFallbackCount, 0);
  assert.equal(detiler.getStats().pipelineCompiled, true);

  detiler.destroy();
});

test('GpuTextureDetiler lazy pipeline caching, uniform buffer reuse, and zero steady-state allocations', () => {
  const env = createMockWebGpuCompute({ executeKernel: true });
  const detiler = new GpuTextureDetiler({ device: env.device, labelPrefix: 'test-detile' });

  // Nothing allocated in constructor (strictly lazy).
  assert.equal(detiler.pipelineCompiled, false);
  assert.equal(env.shaderModules.length, 0);
  assert.equal(env.bindGroupLayouts.length, 0);
  assert.equal(env.pipelines.length, 0);
  assert.equal(env.buffers.length, 0);

  const srcBuf = env.device.createBuffer({ size: 65536, usage: BUFFER_USAGE.STORAGE });
  const dstBuf = env.device.createBuffer({ size: 65536, usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC });
  const buffersBeforeDispatch = env.buffers.length;

  // Dispatch 5 times with the same (srcBuf, dstBuf) pair.
  for (let i = 0; i < 5; i += 1) {
    const encoder = env.device.createCommandEncoder();
    const res = detiler.detileBufferToBuffer(encoder, {
      srcBuffer: srcBuf,
      dstBuffer: dstBuf,
      d3d12Layout: 0x10a, // SW_64KB_D
      blocksX: 64,
      blocksY: 64,
      bytesPerBlock: 16,
    });
    assert.equal(res.ok, true);
    assert.equal(res.workgroupsX, 8);
    assert.equal(res.workgroupsY, 8);
  }

  assert.equal(detiler.pipelineCompiled, true);
  assert.equal(env.shaderModules.length, 1);
  assert.equal(env.bindGroupLayouts.length, 1);
  assert.equal(env.pipelineLayouts.length, 1);
  assert.equal(env.pipelines.length, 1);
  // Only 1 uniform buffer created by the detiler across all 5 dispatches!
  assert.equal(env.buffers.length - buffersBeforeDispatch, 1);
  assert.equal(env.buffers[buffersBeforeDispatch].size, DETILE_UNIFORM_BYTE_SIZE);
  assert.equal(env.buffers[buffersBeforeDispatch].usage, BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST);
  // Same (srcBuf, dstBuf) reuses the cached bind group!
  assert.equal(env.bindGroups.length, 1);
  assert.equal(env.writeBufferCalls.length, 5);
  assert.equal(env.dispatches.length, 5);

  const stats = detiler.getStats();
  assert.equal(stats.gpuDetileDispatches, 5);
  assert.equal(stats.gpuDetileBytes, 5 * 64 * 64 * 16);
  assert.equal(stats.gpuDetileFallbackCount, 0);
  assert.equal(stats.pipelineCompiled, true);

  // Defensive stats snapshot: mutating returned object does not mutate internal state.
  stats.gpuDetileDispatches = 9999;
  assert.equal(detiler.getStats().gpuDetileDispatches, 5);

  detiler.destroy();
});

test('detileToTexture aligns bytesPerRow to 256, reuses staging buffer, and encodes copyBufferToTexture for uncompressed & BC formats', () => {
  const env = createMockWebGpuCompute({ executeKernel: true });
  const detiler = new GpuTextureDetiler(env.device);

  // 1) BC7 100x60 texture => 25x15 4x4 blocks @ 16B/block = 400B tight row -> aligned to 512B bytesPerRow.
  const bc7Footprint = computeGfx10TiledFootprint2D(GFX10_SWIZZLE_MODE.SW_4KB_S, 25, 15, 16);
  const srcBc7 = env.device.createBuffer({ size: bc7Footprint, usage: BUFFER_USAGE.STORAGE });
  const linearBc7 = makeSyntheticLinear(25 * 15 * 16, 777);
  tile2DReference(
    { swizzleMode: GFX10_SWIZZLE_MODE.SW_4KB_S, blocksX: 25, blocksY: 15, bytesPerBlock: 16 },
    linearBc7,
    new Uint8Array(srcBc7._storage)
  );
  const mockBc7Texture = { kind: 'GPUTexture', label: 'bc7-tex', format: 'bc7-rgba-unorm' };

  const enc1 = env.device.createCommandEncoder();
  const res1 = detiler.detileToTexture(enc1, {
    srcBuffer: srcBc7,
    dstTexture: mockBc7Texture,
    d3d12Layout: 0x105, // SW_4KB_S
    blocksX: 25,
    blocksY: 15,
    bytesPerBlock: 16,
    blockWidth: 4,
    blockHeight: 4,
    mipLevel: 1,
    arrayLayer: 2,
  });

  assert.equal(res1.ok, true);
  assert.equal(res1.bytesPerRow, 512);
  assert.equal(res1.rowsPerImage, 15);
  assert.equal(res1.copiedToTexture, true);
  assert.equal(env.copyBufferToTextureCalls.length, 1);

  const copy1 = env.copyBufferToTextureCalls[0];
  assert.equal(copy1.source.buffer, res1.stagingBuffer);
  assert.equal(copy1.source.offset, 0);
  assert.equal(copy1.source.bytesPerRow, 512);
  assert.equal(copy1.source.rowsPerImage, 15);
  assert.equal(copy1.destination.texture, mockBc7Texture);
  assert.equal(copy1.destination.mipLevel, 1);
  assert.deepEqual(copy1.destination.origin, { x: 0, y: 0, z: 2 });
  assert.deepEqual(copy1.copySize, { width: 100, height: 60, depthOrArrayLayers: 1 });

  // Verify detiled bytes in staging buffer match original BC7 linear blocks bit-for-bit!
  const stagingBytes = new Uint8Array(res1.stagingBuffer._storage);
  for (let y = 0; y < 15; y += 1) {
    assert.deepEqual(
      stagingBytes.subarray(y * 512, y * 512 + 400),
      linearBc7.subarray(y * 400, (y + 1) * 400)
    );
  }

  // 2) Second smaller texture reuses the EXACT SAME internal staging buffer (zero new allocations!).
  const buffersAfterFirst = env.buffers.length;
  const srcSmall = env.device.createBuffer({ size: 256, usage: BUFFER_USAGE.STORAGE });
  const enc2 = env.device.createCommandEncoder();
  const res2 = detiler.detileToTexture(enc2, {
    srcBuffer: srcSmall,
    // Omit dstTexture => staging-only mode (no copyBufferToTexture call).
    swizzleMode: GFX10_SWIZZLE_MODE.SW_256B_D,
    blocksX: 8,
    blocksY: 8,
    bytesPerBlock: 4,
  });

  assert.equal(res2.ok, true);
  assert.equal(res2.bytesPerRow, 256);
  assert.equal(res2.copiedToTexture, false);
  assert.equal(res2.stagingBuffer, res1.stagingBuffer);
  assert.equal(env.buffers.length, buffersAfterFirst + 1); // only srcSmall created by test
  assert.equal(env.copyBufferToTextureCalls.length, 1);

  // 3) Caller-provided staging buffer is used directly and NOT destroyed by detiler.destroy().
  const callerStaging = env.device.createBuffer({
    size: 4096,
    usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
  });
  const enc3 = env.device.createCommandEncoder();
  const res3 = detiler.detileToTexture(enc3, {
    srcBuffer: srcSmall,
    stagingBuffer: callerStaging,
    dstTexture: { kind: 'GPUTexture', label: 'rgba8-tex' },
    swizzleMode: GFX10_SWIZZLE_MODE.SW_256B_D,
    blocksX: 8,
    blocksY: 8,
    bytesPerBlock: 4,
  });
  assert.equal(res3.ok, true);
  assert.equal(res3.stagingBuffer, callerStaging);
  assert.equal(res3.copiedToTexture, true);

  detiler.destroy();
  assert.equal(callerStaging.destroyed, false, 'caller-owned staging buffer must not be destroyed');
  assert.equal(res1.stagingBuffer.destroyed, true, 'internal staging buffer must be destroyed');
  assert.equal(res1.stagingBuffer.destroyCalls, 1);
});

test('canDetile and graceful CPU fallback on 3D, unsupported/unknown modes, unaligned pitches, bad sizes, and destroy()', () => {
  const env = createMockWebGpuCompute({ executeKernel: true });
  const detiler = new GpuTextureDetiler(env.device);

  assert.equal(
    detiler.canDetile({ swizzleMode: GFX10_SWIZZLE_MODE.SW_64KB_D, blocksX: 32, blocksY: 32, bytesPerBlock: 4 }),
    true
  );
  // 3D textures rejected
  assert.equal(
    detiler.canDetile({ swizzleMode: GFX10_SWIZZLE_MODE.SW_64KB_S, blocksX: 16, blocksY: 16, depth: 4, bytesPerBlock: 4 }),
    false
  );
  assert.equal(
    detiler.canDetile({ swizzleMode: GFX10_SWIZZLE_MODE.SW_64KB_S, blocksX: 16, blocksY: 16, dimension: '3d', bytesPerBlock: 4 }),
    false
  );
  // Unknown / _R / unsupported swizzle modes rejected
  assert.equal(detiler.canDetile({ d3d12Layout: 0x10b, blocksX: 16, blocksY: 16, bytesPerBlock: 4 }), false);
  assert.equal(detiler.canDetile({ d3d12Layout: 0x104, blocksX: 16, blocksY: 16, bytesPerBlock: 4 }), false);
  assert.equal(detiler.canDetile({ swizzleMode: 255, blocksX: 16, blocksY: 16, bytesPerBlock: 4 }), false);
  // Invalid block sizes / dimensions / alignments rejected
  assert.equal(detiler.canDetile({ swizzleMode: 9, blocksX: 16, blocksY: 16, bytesPerBlock: 12 }), false);
  assert.equal(detiler.canDetile({ swizzleMode: 9, blocksX: 0, blocksY: 16, bytesPerBlock: 4 }), false);
  assert.equal(detiler.canDetile({ swizzleMode: 9, blocksX: 16, blocksY: 16, bytesPerBlock: 4, dstByteOffset: 2 }), false);
  assert.equal(detiler.canDetile({ swizzleMode: 9, blocksX: 5, blocksY: 5, bytesPerBlock: 1, dstRowPitchBytes: 5 }), false);

  const srcBuf = env.device.createBuffer({ size: 4096, usage: BUFFER_USAGE.STORAGE });
  const dstBuf = env.device.createBuffer({ size: 4096, usage: BUFFER_USAGE.STORAGE });
  const encoder = env.device.createCommandEncoder();

  // Fallback on 3D texture
  const fb3d = detiler.detileBufferToBuffer(encoder, {
    srcBuffer: srcBuf,
    dstBuffer: dstBuf,
    swizzleMode: GFX10_SWIZZLE_MODE.SW_4KB_S,
    blocksX: 16,
    blocksY: 16,
    depth: 4,
    bytesPerBlock: 4,
  });
  assert.equal(fb3d.ok, false);
  assert.equal(fb3d.fallback, true);
  assert.equal(typeof fb3d.reason, 'string');

  // Fallback on unsupported layout
  const fbUnknown = detiler.detileBufferToBuffer(encoder, {
    srcBuffer: srcBuf,
    dstBuffer: dstBuf,
    d3d12Layout: 0x10b,
    blocksX: 16,
    blocksY: 16,
    bytesPerBlock: 4,
  });
  assert.equal(fbUnknown.ok, false);
  assert.equal(fbUnknown.fallback, true);

  // Fallback on unaligned tight 1B row in detileBufferToBuffer (prevents cross-row u32 race!)
  const fbUnaligned = detiler.detileBufferToBuffer(encoder, {
    srcBuffer: srcBuf,
    dstBuffer: dstBuf,
    swizzleMode: GFX10_SWIZZLE_MODE.SW_256B_S,
    blocksX: 5,
    blocksY: 5,
    bytesPerBlock: 1,
  });
  assert.equal(fbUnaligned.ok, false);
  assert.equal(fbUnaligned.fallback, true);

  // Fallback on too-small dstBuffer
  const tinyDst = env.device.createBuffer({ size: 16, usage: BUFFER_USAGE.STORAGE });
  const fbSmall = detiler.detileBufferToBuffer(encoder, {
    srcBuffer: srcBuf,
    dstBuffer: tinyDst,
    swizzleMode: GFX10_SWIZZLE_MODE.SW_256B_S,
    blocksX: 16,
    blocksY: 16,
    bytesPerBlock: 4,
  });
  assert.equal(fbSmall.ok, false);
  assert.equal(fbSmall.fallback, true);

  // Pipeline was never even compiled because every call fell back cleanly!
  assert.equal(detiler.getStats().pipelineCompiled, false);
  assert.equal(detiler.getStats().gpuDetileDispatches, 0);
  assert.equal(detiler.getStats().gpuDetileFallbackCount, 4);

  // Idempotent destroy() and post-destroy fallback.
  detiler.destroy();
  detiler.destroy();
  assert.equal(detiler.isDestroyed, true);
  assert.equal(detiler.canDetile({ swizzleMode: 9, blocksX: 16, blocksY: 16, bytesPerBlock: 4 }), false);
  const fbDestroyed = detiler.detileBufferToBuffer(encoder, {
    srcBuffer: srcBuf,
    dstBuffer: dstBuf,
    swizzleMode: 9,
    blocksX: 16,
    blocksY: 16,
    bytesPerBlock: 4,
  });
  assert.equal(fbDestroyed.ok, false);
  assert.equal(fbDestroyed.fallback, true);
  assert.equal(detiler.getStats().gpuDetileFallbackCount, 5);

  // Missing device degrades cleanly without throwing.
  const noDev = new GpuTextureDetiler(null);
  assert.equal(noDev.canDetile({ swizzleMode: 9, blocksX: 16, blocksY: 16, bytesPerBlock: 4 }), false);
  const fbNoDev = noDev.detileBufferToBuffer(encoder, {
    srcBuffer: srcBuf,
    dstBuffer: dstBuf,
    swizzleMode: 9,
    blocksX: 16,
    blocksY: 16,
    bytesPerBlock: 4,
  });
  assert.equal(fbNoDev.ok, false);
  assert.equal(fbNoDev.fallback, true);
  noDev.destroy();
});

test('srcByteLimit bounds clamping zeroes out-of-range source reads identically in JS reference and WGSL kernel', () => {
  const env = createMockWebGpuCompute({ executeKernel: true });
  const detiler = new GpuTextureDetiler(env.device);

  const srcBuf = env.device.createBuffer({ size: 256, usage: BUFFER_USAGE.STORAGE });
  new Uint8Array(srcBuf._storage).fill(0xab);
  const dstBuf = env.device.createBuffer({ size: 256, usage: BUFFER_USAGE.STORAGE });
  new Uint8Array(dstBuf._storage).fill(0x55);

  // Truncate srcByteLimit to 64 bytes of a 256-byte SW_256B_D 8x8 RGBA8 tile.
  const encoder = env.device.createCommandEncoder();
  const res = detiler.detileBufferToBuffer(encoder, {
    srcBuffer: srcBuf,
    dstBuffer: dstBuf,
    swizzleMode: GFX10_SWIZZLE_MODE.SW_256B_D,
    blocksX: 8,
    blocksY: 8,
    bytesPerBlock: 4,
    srcByteLimit: 64,
  });
  assert.equal(res.ok, true);

  const ref = detile2DReference(
    {
      swizzleMode: GFX10_SWIZZLE_MODE.SW_256B_D,
      blocksX: 8,
      blocksY: 8,
      bytesPerBlock: 4,
      srcByteLimit: 64,
    },
    new Uint8Array(srcBuf._storage)
  );
  assert.equal(ref.ok, true);
  assert.deepEqual(new Uint8Array(dstBuf._storage), ref.dstBytes);

  // Verify non-zero bytes in first 64B of tile and zeroes past srcByteLimit.
  let nonZeroCount = 0;
  let zeroCount = 0;
  for (const b of new Uint8Array(dstBuf._storage)) {
    if (b === 0xab) nonZeroCount += 1;
    else if (b === 0) zeroCount += 1;
  }
  assert.equal(nonZeroCount, 64);
  assert.equal(zeroCount, 192);

  detiler.destroy();
});

test('computeDetileDispatchGrid packs 4x1B and 2x2B horizontal texels per thread', () => {
  assert.deepEqual(computeDetileDispatchGrid(1, 1, 1), { threadsX: 1, threadsY: 1, workgroupsX: 1, workgroupsY: 1 });
  assert.deepEqual(computeDetileDispatchGrid(32, 16, 1), { threadsX: 8, threadsY: 16, workgroupsX: 1, workgroupsY: 2 });
  assert.deepEqual(computeDetileDispatchGrid(33, 17, 1), { threadsX: 9, threadsY: 17, workgroupsX: 2, workgroupsY: 3 });
  assert.deepEqual(computeDetileDispatchGrid(16, 8, 2), { threadsX: 8, threadsY: 8, workgroupsX: 1, workgroupsY: 1 });
  assert.deepEqual(computeDetileDispatchGrid(17, 9, 2), { threadsX: 9, threadsY: 9, workgroupsX: 2, workgroupsY: 2 });
  assert.deepEqual(computeDetileDispatchGrid(64, 64, 4), { threadsX: 64, threadsY: 64, workgroupsX: 8, workgroupsY: 8 });
  assert.deepEqual(computeDetileDispatchGrid(25, 15, 8), { threadsX: 25, threadsY: 15, workgroupsX: 4, workgroupsY: 2 });
  assert.deepEqual(computeDetileDispatchGrid(25, 15, 16), { threadsX: 25, threadsY: 15, workgroupsX: 4, workgroupsY: 2 });
});
