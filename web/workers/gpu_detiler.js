// Cross-GPU WebGPU compute-shader AMD GFX10 / RDNA2 2D texture detiler (Issue #2234).
// 100% lossless bit-exact match with runtime/gpu/gfx10_detile.cpp (AMD addrlib RB+ equations).
// Standard WebGPU WGSL v1.0 only (zero vendor extensions; works on NVIDIA, AMD, Intel, Apple Silicon).
// Zero external dependencies; runs in Web Workers (importScripts), browsers, and Node.js.

(function (root) {
  'use strict';

  const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;
  const DETILE_WORKGROUP_SIZE_X = 8;
  const DETILE_WORKGROUP_SIZE_Y = 8;
  const DETILE_UNIFORM_U32_COUNT = 12;
  const DETILE_UNIFORM_BYTE_SIZE = DETILE_UNIFORM_U32_COUNT * 4; // 48 bytes (16-byte aligned)
  const DEFAULT_MAX_WORKGROUPS_PER_DIM = 65535;
  const DEFAULT_STAGING_GRANULARITY = 4096;

  /**
   * Documented AMD GFX10 2D thin SWIZZLE_MODE_ENUM values matching runtime/gpu/gfx10_detile.h.
   * Only verified addrlib RB+ _S / _D modes (plus Linear and 64KB/Var _X XOR modes) are enabled
   * on the GPU compute path; all other modes return UNSUPPORTED for graceful CPU fallback.
   */
  const GFX10_SWIZZLE_MODE = Object.freeze({
    LINEAR: 0,
    SW_256B_S: 1,
    SW_256B_D: 2,
    SW_4KB_S: 5,
    SW_4KB_D: 6,
    SW_64KB_S: 9,
    SW_64KB_D: 10,
    SW_VAR_S: 13,
    SW_VAR_D: 14,
    SW_64KB_S_X: 25,
    SW_64KB_D_X: 26,
    SW_VAR_S_X: 29,
    SW_VAR_D_X: 30,
    UNSUPPORTED: 255,
  });

  // WebGPU GPUBufferUsage & GPUShaderStage constants with headless/Node fallbacks.
  const BUFFER_USAGE = Object.freeze({
    COPY_SRC:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_SRC === 'number'
        ? GPUBufferUsage.COPY_SRC
        : 0x0004,
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
    UNIFORM:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.UNIFORM === 'number'
        ? GPUBufferUsage.UNIFORM
        : 0x0040,
    STORAGE:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.STORAGE === 'number'
        ? GPUBufferUsage.STORAGE
        : 0x0080,
  });

  const SHADER_STAGE = Object.freeze({
    COMPUTE:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.COMPUTE === 'number'
        ? GPUShaderStage.COMPUTE
        : 0x0004,
  });

  // 256B micro-tile coordinate widths/heights (log2 in elements/blocks) indexed by bppLog2 [0..4].
  const W256_LOG2 = Object.freeze([4, 4, 3, 3, 2]);
  const H256_LOG2 = Object.freeze([4, 3, 3, 2, 2]);

  // Exact AMD addrlib GFX10 RB+ 2D thin bit equations (mirrors kEqStandard / kEqDisplay in
  // runtime/gpu/gfx10_detile.cpp). Entry i encodes bit source: 0 = intra-element byte bit,
  // 0x10|b = X bit b, 0x20|b = Y bit b.
  const EQ_X = 0x10;
  const EQ_Y = 0x20;
  const X = (b) => EQ_X | b;
  const Y = (b) => EQ_Y | b;

  const EQ_STANDARD = Object.freeze([
    Object.freeze([X(0), X(1), X(2), X(3), Y(0), Y(1), Y(2), Y(3), Y(4), X(4), Y(5), X(5), Y(6), X(6), Y(7), X(7)]),
    Object.freeze([0, X(0), X(1), X(2), Y(0), Y(1), Y(2), X(3), Y(3), X(4), Y(4), X(5), Y(5), X(6), Y(6), X(7)]),
    Object.freeze([0, 0, X(0), X(1), Y(0), Y(1), Y(2), X(2), Y(3), X(3), Y(4), X(4), Y(5), X(5), Y(6), X(6)]),
    Object.freeze([0, 0, 0, X(0), Y(0), Y(1), X(1), X(2), Y(2), X(3), Y(3), X(4), Y(4), X(5), Y(5), X(6)]),
    Object.freeze([0, 0, 0, 0, Y(0), Y(1), X(0), X(1), Y(2), X(2), Y(3), X(3), Y(4), X(4), Y(5), X(5)]),
  ]);

  const EQ_DISPLAY = Object.freeze([
    Object.freeze([X(0), X(1), X(2), Y(1), Y(0), Y(2), X(3), Y(3), Y(4), X(4), Y(5), X(5), Y(6), X(6), Y(7), X(7)]),
    Object.freeze([0, X(0), X(1), X(2), Y(0), Y(1), Y(2), X(3), Y(3), X(4), Y(4), X(5), Y(5), X(6), Y(6), X(7)]),
    Object.freeze([0, 0, X(0), X(1), Y(0), Y(1), X(2), Y(2), Y(3), X(3), Y(4), X(4), Y(5), X(5), Y(6), X(6)]),
    Object.freeze([0, 0, 0, X(0), Y(0), X(1), X(2), Y(1), Y(2), X(3), Y(3), X(4), Y(4), X(5), Y(5), X(6)]),
    Object.freeze([0, 0, 0, 0, X(0), Y(0), X(1), Y(1), Y(2), X(2), Y(3), X(3), Y(4), X(4), Y(5), X(5)]),
  ]);

  /**
   * Convert bytesPerBlock in {1, 2, 4, 8, 16} to log2 in [0..4], or -1 if invalid.
   *
   * @param {number} bytesPerBlock
   * @returns {number}
   */
  function bppToLog2(bytesPerBlock) {
    switch (bytesPerBlock) {
      case 1:
        return 0;
      case 2:
        return 1;
      case 4:
        return 2;
      case 8:
        return 3;
      case 16:
        return 4;
      default:
        return -1;
    }
  }

  /**
   * Align a non-negative integer upward to a power-of-two alignment.
   *
   * @param {number} value
   * @param {number} alignment
   * @returns {number}
   */
  function alignUp(value, alignment) {
    if (!Number.isFinite(value) || value <= 0) {
      return 0;
    }
    if (!Number.isInteger(alignment) || alignment <= 1 || (alignment & (alignment - 1)) !== 0) {
      return Math.ceil(value);
    }
    const rem = value % alignment;
    return rem === 0 ? value : value + (alignment - rem);
  }

  /**
   * Align a row byte width up to WebGPU's 256-byte `copyBufferToTexture` bytesPerRow requirement.
   *
   * @param {number} rowBytes
   * @returns {number}
   */
  function alignBytesPerRow(rowBytes) {
    return alignUp(rowBytes, WEBGPU_BYTES_PER_ROW_ALIGNMENT);
  }

  /**
   * Decode a D3D12_TEXTURE_LAYOUT value into a supported GFX10_SWIZZLE_MODE.
   * Matches decode_d3d12_layout() in runtime/gpu/gfx10_detile.cpp for supported 2D thin modes.
   *
   * @param {number} d3d12Layout
   * @returns {number}
   */
  function decodeD3d12Layout(d3d12Layout) {
    if (!Number.isInteger(d3d12Layout) || d3d12Layout < 0) {
      return GFX10_SWIZZLE_MODE.UNSUPPORTED;
    }
    if (d3d12Layout === 0 || d3d12Layout === 1) {
      return GFX10_SWIZZLE_MODE.LINEAR;
    }
    if (d3d12Layout >= 0x100 && d3d12Layout <= 0x11f) {
      switch (d3d12Layout & 0x1f) {
        case 0:
          return GFX10_SWIZZLE_MODE.LINEAR;
        case 1:
          return GFX10_SWIZZLE_MODE.SW_256B_S;
        case 2:
          return GFX10_SWIZZLE_MODE.SW_256B_D;
        case 5:
          return GFX10_SWIZZLE_MODE.SW_4KB_S;
        case 6:
          return GFX10_SWIZZLE_MODE.SW_4KB_D;
        case 9:
          return GFX10_SWIZZLE_MODE.SW_64KB_S;
        case 10:
          return GFX10_SWIZZLE_MODE.SW_64KB_D;
        case 13:
          return GFX10_SWIZZLE_MODE.SW_VAR_S;
        case 14:
          return GFX10_SWIZZLE_MODE.SW_VAR_D;
        case 25:
          return GFX10_SWIZZLE_MODE.SW_64KB_S_X;
        case 26:
          return GFX10_SWIZZLE_MODE.SW_64KB_D_X;
        case 29:
          return GFX10_SWIZZLE_MODE.SW_VAR_S_X;
        case 30:
          return GFX10_SWIZZLE_MODE.SW_VAR_D_X;
        default:
          return GFX10_SWIZZLE_MODE.UNSUPPORTED;
      }
    }
    return GFX10_SWIZZLE_MODE.UNSUPPORTED;
  }

  function classifySwizzleMode(mode) {
    switch (mode) {
      case GFX10_SWIZZLE_MODE.LINEAR:
        return { valid: true, isLinear: true, isDisplay: false, blkSizeLog2: 0, isXor: false };
      case GFX10_SWIZZLE_MODE.SW_256B_S:
        return { valid: true, isLinear: false, isDisplay: false, blkSizeLog2: 8, isXor: false };
      case GFX10_SWIZZLE_MODE.SW_256B_D:
        return { valid: true, isLinear: false, isDisplay: true, blkSizeLog2: 8, isXor: false };
      case GFX10_SWIZZLE_MODE.SW_4KB_S:
        return { valid: true, isLinear: false, isDisplay: false, blkSizeLog2: 12, isXor: false };
      case GFX10_SWIZZLE_MODE.SW_4KB_D:
        return { valid: true, isLinear: false, isDisplay: true, blkSizeLog2: 12, isXor: false };
      case GFX10_SWIZZLE_MODE.SW_64KB_S:
      case GFX10_SWIZZLE_MODE.SW_VAR_S:
        return { valid: true, isLinear: false, isDisplay: false, blkSizeLog2: 16, isXor: false };
      case GFX10_SWIZZLE_MODE.SW_64KB_D:
      case GFX10_SWIZZLE_MODE.SW_VAR_D:
        return { valid: true, isLinear: false, isDisplay: true, blkSizeLog2: 16, isXor: false };
      case GFX10_SWIZZLE_MODE.SW_64KB_S_X:
      case GFX10_SWIZZLE_MODE.SW_VAR_S_X:
        return { valid: true, isLinear: false, isDisplay: false, blkSizeLog2: 16, isXor: true };
      case GFX10_SWIZZLE_MODE.SW_64KB_D_X:
      case GFX10_SWIZZLE_MODE.SW_VAR_D_X:
        return { valid: true, isLinear: false, isDisplay: true, blkSizeLog2: 16, isXor: true };
      default:
        return { valid: false, isLinear: false, isDisplay: false, blkSizeLog2: 0, isXor: false };
    }
  }

  /**
   * True if `mode` is supported by the GPU compute detiler.
   *
   * @param {number} mode
   * @returns {boolean}
   */
  function isGfx10SwizzleSupported(mode) {
    return Number.isInteger(mode) && classifySwizzleMode(mode).valid;
  }

  /**
   * Return 2D tile geometry for `(mode, bytesPerBlock)`.
   *
   * @param {number} mode
   * @param {number} bytesPerBlock
   */
  function getGfx10TileShape2D(mode, bytesPerBlock) {
    const traits = classifySwizzleMode(mode);
    const bppLog2 = bppToLog2(bytesPerBlock);
    if (!traits.valid || bppLog2 < 0) {
      return {
        valid: false,
        isLinear: false,
        isDisplay: false,
        isXor: false,
        blkSizeLog2: 0,
        blockBytes: 0,
        wLog2: 0,
        hLog2: 0,
        tileW: 0,
        tileH: 0,
      };
    }
    if (traits.isLinear) {
      return {
        valid: true,
        isLinear: true,
        isDisplay: false,
        isXor: false,
        blkSizeLog2: 0,
        blockBytes: bytesPerBlock,
        wLog2: 0,
        hLog2: 0,
        tileW: 1,
        tileH: 1,
      };
    }
    const extraLog2 = (traits.blkSizeLog2 - 8) >> 1; // 0 (256B), 2 (4KB), 4 (64KB/Var)
    const wLog2 = W256_LOG2[bppLog2] + extraLog2;
    const hLog2 = H256_LOG2[bppLog2] + extraLog2;
    return {
      valid: true,
      isLinear: false,
      isDisplay: traits.isDisplay,
      isXor: traits.isXor,
      blkSizeLog2: traits.blkSizeLog2,
      blockBytes: 1 << traits.blkSizeLog2,
      wLog2,
      hLog2,
      tileW: 1 << wLog2,
      tileH: 1 << hLog2,
    };
  }

  /**
   * Exact GFX10 64KB/Var _X pipe-bank XOR matching apply_gfx10_xor_64k() in
   * runtime/gpu/gfx10_detile.cpp.
   *
   * @param {number} intra
   * @param {number} [pipeBankXor=0]
   * @returns {number}
   */
  function applyGfx10Xor64k(intra, pipeBankXor = 0) {
    const hi = (intra >>> 12) & 0xf;
    const rev = ((hi & 1) << 3) | ((hi & 2) << 1) | ((hi & 4) >>> 1) | ((hi & 8) >>> 3);
    const seed = (pipeBankXor & 0xff) << 8;
    return (intra ^ (rev << 8) ^ seed) >>> 0;
  }

  /**
   * Compute the exact byte offset of 2D block/texel `(bx, by)` in a GFX10 2D thin surface.
   * Matches `tiled_byte_offset_2d_xor()` in `runtime/gpu/gfx10_detile.cpp` bit-for-bit.
   * Returns -1 if `mode`, `bytesPerBlock`, or coordinates are invalid.
   *
   * @param {number} mode
   * @param {number} bx
   * @param {number} by
   * @param {number} bytesPerBlock
   * @param {number} pitchInBlocks
   * @param {number} [pipeBankXor=0]
   * @returns {number}
   */
  function computeGfx10TiledByteOffset2D(
    mode,
    bx,
    by,
    bytesPerBlock,
    pitchInBlocks,
    pipeBankXor = 0
  ) {
    const traits = classifySwizzleMode(mode);
    const bppLog2 = bppToLog2(bytesPerBlock);
    if (
      !traits.valid ||
      bppLog2 < 0 ||
      !Number.isInteger(bx) ||
      !Number.isInteger(by) ||
      bx < 0 ||
      by < 0 ||
      !Number.isInteger(pitchInBlocks) ||
      pitchInBlocks < 0
    ) {
      return -1;
    }

    if (traits.isLinear) {
      return (by * pitchInBlocks + bx) * bytesPerBlock;
    }

    const extraLog2 = (traits.blkSizeLog2 - 8) >> 1;
    const wLog2 = W256_LOG2[bppLog2] + extraLog2;
    const hLog2 = H256_LOG2[bppLog2] + extraLog2;
    const tileW = 1 << wLog2;
    const tileH = 1 << hLog2;

    const effPitch = pitchInBlocks > 0 ? pitchInBlocks : tileW;
    const tilesPerRow = Math.floor((effPitch + tileW - 1) / tileW);

    const tileX = Math.floor(bx / tileW);
    const tileY = Math.floor(by / tileH);
    const localX = bx & (tileW - 1);
    const localY = by & (tileH - 1);

    const eq = traits.isDisplay ? EQ_DISPLAY[bppLog2] : EQ_STANDARD[bppLog2];
    let intra = 0;
    for (let i = 0; i < traits.blkSizeLog2; i += 1) {
      const term = eq[i];
      if (term === 0) {
        continue;
      }
      const axis = term >>> 4;
      const bit = term & 0xf;
      const coord = axis === 1 ? localX : localY;
      if (((coord >>> bit) & 1) !== 0) {
        intra |= 1 << i;
      }
    }

    if (traits.isXor) {
      intra = applyGfx10Xor64k(intra, pipeBankXor);
    }

    const blockBytes = 1 << traits.blkSizeLog2;
    const tileIndex = tileY * tilesPerRow + tileX;
    return tileIndex * blockBytes + intra;
  }

  /**
   * Compute required source footprint in bytes for a 2D GFX10 tiled or linear subresource.
   * Returns 0 if any parameter is invalid.
   *
   * @param {number} mode
   * @param {number} blocksX
   * @param {number} blocksY
   * @param {number} bytesPerBlock
   * @param {number} [pitchInBlocks=0]
   * @returns {number}
   */
  function computeGfx10TiledFootprint2D(
    mode,
    blocksX,
    blocksY,
    bytesPerBlock,
    pitchInBlocks = 0
  ) {
    if (
      !Number.isInteger(blocksX) ||
      !Number.isInteger(blocksY) ||
      blocksX < 1 ||
      blocksY < 1 ||
      !Number.isInteger(pitchInBlocks) ||
      pitchInBlocks < 0
    ) {
      return 0;
    }
    const shape = getGfx10TileShape2D(mode, bytesPerBlock);
    if (!shape.valid) {
      return 0;
    }
    const effPitch = pitchInBlocks > 0 ? pitchInBlocks : blocksX;
    if (effPitch < blocksX) {
      return 0;
    }
    if (shape.isLinear) {
      return effPitch * bytesPerBlock * blocksY;
    }
    const tilesX = Math.floor((effPitch + shape.tileW - 1) / shape.tileW);
    const tilesY = Math.floor((blocksY + shape.tileH - 1) / shape.tileH);
    return tilesX * tilesY * shape.blockBytes;
  }

  /**
   * Compute workgroup dispatch dimensions for `(blocksX, blocksY, bytesPerBlock)` matching the
   * race-free word-packed WGSL compute kernel (4×1B or 2×2B texels per thread, 1 block per thread
   * for 4B/8B/16B).
   *
   * @param {number} blocksX
   * @param {number} blocksY
   * @param {number} bytesPerBlock
   */
  function computeDetileDispatchGrid(blocksX, blocksY, bytesPerBlock) {
    const threadsX =
      bytesPerBlock === 1
        ? Math.ceil(blocksX / 4)
        : bytesPerBlock === 2
          ? Math.ceil(blocksX / 2)
          : blocksX;
    const threadsY = blocksY;
    return {
      threadsX,
      threadsY,
      workgroupsX: Math.ceil(threadsX / DETILE_WORKGROUP_SIZE_X),
      workgroupsY: Math.ceil(threadsY / DETILE_WORKGROUP_SIZE_Y),
    };
  }

  function toUint8View(buf) {
    if (buf instanceof Uint8Array) {
      return buf;
    }
    if (ArrayBuffer.isView(buf)) {
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    }
    if (buf instanceof ArrayBuffer) {
      return new Uint8Array(buf);
    }
    return null;
  }

  function resolveSwizzleMode(params) {
    if (!params || typeof params !== 'object') {
      return GFX10_SWIZZLE_MODE.UNSUPPORTED;
    }
    if (params.swizzleMode !== undefined) {
      return Number.isInteger(params.swizzleMode)
        ? params.swizzleMode
        : GFX10_SWIZZLE_MODE.UNSUPPORTED;
    }
    if (params.d3d12Layout !== undefined) {
      return decodeD3d12Layout(params.d3d12Layout);
    }
    return GFX10_SWIZZLE_MODE.UNSUPPORTED;
  }

  /**
   * Pure JS exact reference 2D detiler matching the WGSL compute shader and
   * `runtime/gpu/gfx10_detile.cpp` bit-for-bit.
   *
   * @param {object} params
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer} srcBytes
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer} [dstBytes]
   */
  function detile2DReference(params, srcBytes, dstBytes) {
    if (!params || typeof params !== 'object') {
      return { ok: false, error: 'missing params', bytesWritten: 0, rowPitchBytes: 0, dstBytes: null };
    }
    const src = toUint8View(srcBytes);
    if (!src) {
      return { ok: false, error: 'invalid srcBytes', bytesWritten: 0, rowPitchBytes: 0, dstBytes: null };
    }

    const mode = resolveSwizzleMode(params);
    const blocksX = params.blocksX;
    const blocksY = params.blocksY;
    const bytesPerBlock = params.bytesPerBlock;
    const shape = getGfx10TileShape2D(mode, bytesPerBlock);

    if (!shape.valid) {
      return {
        ok: false,
        error: 'unsupported swizzleMode or bytesPerBlock',
        bytesWritten: 0,
        rowPitchBytes: 0,
        dstBytes: null,
      };
    }
    if (!Number.isInteger(blocksX) || !Number.isInteger(blocksY) || blocksX < 1 || blocksY < 1) {
      return {
        ok: false,
        error: 'blocksX and blocksY must be positive integers',
        bytesWritten: 0,
        rowPitchBytes: 0,
        dstBytes: null,
      };
    }

    const pitchInBlocks =
      params.pitchInBlocks !== undefined && params.pitchInBlocks !== 0
        ? params.pitchInBlocks
        : blocksX;
    if (!Number.isInteger(pitchInBlocks) || pitchInBlocks < blocksX) {
      return {
        ok: false,
        error: 'pitchInBlocks must be an integer >= blocksX',
        bytesWritten: 0,
        rowPitchBytes: 0,
        dstBytes: null,
      };
    }

    const srcByteOffset = params.srcByteOffset !== undefined ? params.srcByteOffset : 0;
    const dstByteOffset = params.dstByteOffset !== undefined ? params.dstByteOffset : 0;
    if (
      !Number.isInteger(srcByteOffset) ||
      srcByteOffset < 0 ||
      !Number.isInteger(dstByteOffset) ||
      dstByteOffset < 0
    ) {
      return {
        ok: false,
        error: 'srcByteOffset and dstByteOffset must be non-negative integers',
        bytesWritten: 0,
        rowPitchBytes: 0,
        dstBytes: null,
      };
    }

    const tightRowBytes = blocksX * bytesPerBlock;
    const dstRowPitchBytes =
      params.dstRowPitchBytes !== undefined && params.dstRowPitchBytes !== 0
        ? params.dstRowPitchBytes
        : tightRowBytes;
    if (!Number.isInteger(dstRowPitchBytes) || dstRowPitchBytes < tightRowBytes) {
      return {
        ok: false,
        error: 'dstRowPitchBytes must be an integer >= blocksX * bytesPerBlock',
        bytesWritten: 0,
        rowPitchBytes: 0,
        dstBytes: null,
      };
    }

    const rawSrcLimit =
      params.srcByteLimit !== undefined ? params.srcByteLimit : src.byteLength;
    if (!Number.isInteger(rawSrcLimit) || rawSrcLimit < 0) {
      return {
        ok: false,
        error: 'srcByteLimit must be a non-negative integer',
        bytesWritten: 0,
        rowPitchBytes: 0,
        dstBytes: null,
      };
    }
    const srcByteLimit = Math.min(rawSrcLimit, src.byteLength);
    const pipeBankXor = shape.isXor ? (params.pipeBankXor || 0) & 0xff : 0;

    const requiredDstSize = dstByteOffset + dstRowPitchBytes * blocksY;
    let dst = dstBytes !== undefined && dstBytes !== null ? toUint8View(dstBytes) : null;
    if (dstBytes !== undefined && dstBytes !== null && !dst) {
      return { ok: false, error: 'invalid dstBytes', bytesWritten: 0, rowPitchBytes: 0, dstBytes: null };
    }
    if (!dst) {
      dst = new Uint8Array(requiredDstSize);
    } else if (dst.byteLength < requiredDstSize) {
      return {
        ok: false,
        error: 'dstBytes too small for output footprint',
        bytesWritten: 0,
        rowPitchBytes: 0,
        dstBytes: null,
      };
    }

    for (let by = 0; by < blocksY; by += 1) {
      const dstRowBase = dstByteOffset + by * dstRowPitchBytes;
      for (let bx = 0; bx < blocksX; bx += 1) {
        const tiledOff = computeGfx10TiledByteOffset2D(
          mode,
          bx,
          by,
          bytesPerBlock,
          pitchInBlocks,
          pipeBankXor
        );
        const absSrcOff = srcByteOffset + tiledOff;
        const dstOff = dstRowBase + bx * bytesPerBlock;
        if (tiledOff >= 0 && absSrcOff + bytesPerBlock <= srcByteLimit) {
          for (let b = 0; b < bytesPerBlock; b += 1) {
            dst[dstOff + b] = src[absSrcOff + b];
          }
        } else {
          for (let b = 0; b < bytesPerBlock; b += 1) {
            dst[dstOff + b] = 0;
          }
        }
      }
    }

    return {
      ok: true,
      error: null,
      bytesWritten: tightRowBytes * blocksY,
      rowPitchBytes: dstRowPitchBytes,
      blocksX,
      blocksY,
      dstBytes: dst,
    };
  }

  /**
   * Pure JS inverse encoder for synthetic round-trip testing: tiles linear 2D blocks into the
   * exact GFX10 2D tiled layout consumed by `detile2DReference` and the WGSL compute shader.
   *
   * @param {object} params
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer} linearBytes
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer} [tiledOutBytes]
   */
  function tile2DReference(params, linearBytes, tiledOutBytes) {
    if (!params || typeof params !== 'object') {
      return { ok: false, error: 'missing params', tiledBytes: null, footprintBytes: 0 };
    }
    const lin = toUint8View(linearBytes);
    if (!lin) {
      return { ok: false, error: 'invalid linearBytes', tiledBytes: null, footprintBytes: 0 };
    }

    const mode = resolveSwizzleMode(params);
    const blocksX = params.blocksX;
    const blocksY = params.blocksY;
    const bytesPerBlock = params.bytesPerBlock;
    const pitchInBlocks =
      params.pitchInBlocks !== undefined && params.pitchInBlocks !== 0
        ? params.pitchInBlocks
        : blocksX;
    const footprintBytes = computeGfx10TiledFootprint2D(
      mode,
      blocksX,
      blocksY,
      bytesPerBlock,
      pitchInBlocks
    );
    if (footprintBytes <= 0) {
      return {
        ok: false,
        error: 'invalid layout, format, or dimensions',
        tiledBytes: null,
        footprintBytes: 0,
      };
    }

    const tightRowBytes = blocksX * bytesPerBlock;
    const linearRowPitchBytes =
      params.linearRowPitchBytes !== undefined && params.linearRowPitchBytes !== 0
        ? params.linearRowPitchBytes
        : tightRowBytes;
    const linearByteOffset = params.linearByteOffset !== undefined ? params.linearByteOffset : 0;
    const tiledByteOffset = params.tiledByteOffset !== undefined ? params.tiledByteOffset : 0;
    if (
      !Number.isInteger(linearRowPitchBytes) ||
      linearRowPitchBytes < tightRowBytes ||
      !Number.isInteger(linearByteOffset) ||
      linearByteOffset < 0 ||
      !Number.isInteger(tiledByteOffset) ||
      tiledByteOffset < 0
    ) {
      return { ok: false, error: 'invalid offsets or row pitch', tiledBytes: null, footprintBytes: 0 };
    }
    if (lin.byteLength < linearByteOffset + linearRowPitchBytes * blocksY) {
      return { ok: false, error: 'linearBytes truncated', tiledBytes: null, footprintBytes: 0 };
    }

    const shape = getGfx10TileShape2D(mode, bytesPerBlock);
    const pipeBankXor = shape.isXor ? (params.pipeBankXor || 0) & 0xff : 0;
    const requiredTiledSize = tiledByteOffset + footprintBytes;
    let out = tiledOutBytes !== undefined && tiledOutBytes !== null ? toUint8View(tiledOutBytes) : null;
    if (!out) {
      out = new Uint8Array(requiredTiledSize);
    } else if (out.byteLength < requiredTiledSize) {
      return { ok: false, error: 'tiledOutBytes too small', tiledBytes: null, footprintBytes: 0 };
    }

    for (let by = 0; by < blocksY; by += 1) {
      const linRow = linearByteOffset + by * linearRowPitchBytes;
      for (let bx = 0; bx < blocksX; bx += 1) {
        const tiledOff = computeGfx10TiledByteOffset2D(
          mode,
          bx,
          by,
          bytesPerBlock,
          pitchInBlocks,
          pipeBankXor
        );
        const dstOff = tiledByteOffset + tiledOff;
        const srcOff = linRow + bx * bytesPerBlock;
        for (let b = 0; b < bytesPerBlock; b += 1) {
          out[dstOff + b] = lin[srcOff + b];
        }
      }
    }

    return {
      ok: true,
      error: null,
      tiledBytes: out,
      footprintBytes,
    };
  }

  // Cached immutable WGSL source string (100% standard WebGPU WGSL v1.0, zero vendor extensions).
  const GFX10_DETILE_WGSL = /* wgsl */ `
// Cross-GPU AMD GFX10 / RDNA2 2D thin texture detiler (standard WebGPU WGSL v1.0).
// Matches runtime/gpu/gfx10_detile.cpp AMD addrlib RB+ equations bit-for-bit.
// Race-free on 1B and 2B formats: each thread packs 4x1B or 2x2B linear texels into one u32 write.

struct DetileParams {
  blocksX: u32,
  blocksY: u32,
  bytesPerBlock: u32,
  pitchInBlocks: u32,
  swizzleMode: u32,
  pipeBankXor: u32,
  srcByteOffset: u32,
  dstByteOffset: u32,
  dstRowPitchBytes: u32,
  srcByteLimit: u32,
  tilesPerRow: u32,
  _reserved: u32,
}

@group(0) @binding(0) var<uniform> params: DetileParams;
@group(0) @binding(1) var<storage, read> srcWords: array<u32>;
@group(0) @binding(2) var<storage, read_write> dstWords: array<u32>;

fn readSrcByte(absOff: u32) -> u32 {
  if (absOff >= params.srcByteLimit) {
    return 0u;
  }
  let wordIdx = absOff >> 2u;
  if (wordIdx >= arrayLength(&srcWords)) {
    return 0u;
  }
  return (srcWords[wordIdx] >> ((absOff & 3u) * 8u)) & 0xFFu;
}

fn readSrcU16(absOff: u32) -> u32 {
  if (params.srcByteLimit < 2u || absOff > params.srcByteLimit - 2u) {
    return 0u;
  }
  let byteRem = absOff & 3u;
  let wordIdx = absOff >> 2u;
  let totalWords = arrayLength(&srcWords);
  if (byteRem <= 2u) {
    if (wordIdx >= totalWords) {
      return 0u;
    }
    return (srcWords[wordIdx] >> (byteRem * 8u)) & 0xFFFFu;
  }
  if (wordIdx + 1u >= totalWords) {
    return 0u;
  }
  let lo = (srcWords[wordIdx] >> 24u) & 0xFFu;
  let hi = srcWords[wordIdx + 1u] & 0xFFu;
  return lo | (hi << 8u);
}

fn readSrcU32(absOff: u32) -> u32 {
  if (params.srcByteLimit < 4u || absOff > params.srcByteLimit - 4u) {
    return 0u;
  }
  let wordIdx = absOff >> 2u;
  let byteRem = absOff & 3u;
  let totalWords = arrayLength(&srcWords);
  if (byteRem == 0u) {
    if (wordIdx >= totalWords) {
      return 0u;
    }
    return srcWords[wordIdx];
  }
  if (wordIdx + 1u >= totalWords) {
    return 0u;
  }
  let lo = srcWords[wordIdx];
  let hi = srcWords[wordIdx + 1u];
  let shift = byteRem * 8u;
  return (lo >> shift) | (hi << (32u - shift));
}

fn writeDstU32(wordIdx: u32, value: u32) {
  if (wordIdx < arrayLength(&dstWords)) {
    dstWords[wordIdx] = value;
  }
}

fn computeIntra256(isDisplay: bool, bpb: u32, lx: u32, ly: u32) -> u32 {
  if (bpb == 1u) {
    if (!isDisplay) {
      // SW_*_S 1B: X0 X1 X2 X3 Y0 Y1 Y2 Y3
      return (lx & 0xFu) | ((ly & 0xFu) << 4u);
    }
    // SW_*_D 1B: X0 X1 X2 Y1 Y0 Y2 X3 Y3
    return (lx & 7u)
      | (((ly >> 1u) & 1u) << 3u)
      | ((ly & 1u) << 4u)
      | (((ly >> 2u) & 1u) << 5u)
      | (((lx >> 3u) & 1u) << 6u)
      | (((ly >> 3u) & 1u) << 7u);
  }
  if (bpb == 2u) {
    // SW_*_S and SW_*_D 2B share bits 0..7: 0 X0 X1 X2 Y0 Y1 Y2 X3
    return ((lx & 7u) << 1u)
      | ((ly & 7u) << 4u)
      | (((lx >> 3u) & 1u) << 7u);
  }
  if (bpb == 4u) {
    if (!isDisplay) {
      // SW_*_S 4B: 0 0 X0 X1 Y0 Y1 Y2 X2
      return ((lx & 3u) << 2u)
        | ((ly & 7u) << 4u)
        | (((lx >> 2u) & 1u) << 7u);
    }
    // SW_*_D 4B: 0 0 X0 X1 Y0 Y1 X2 Y2
    return ((lx & 3u) << 2u)
      | ((ly & 3u) << 4u)
      | (((lx >> 2u) & 1u) << 6u)
      | (((ly >> 2u) & 1u) << 7u);
  }
  if (bpb == 8u) {
    if (!isDisplay) {
      // SW_*_S 8B: 0 0 0 X0 Y0 Y1 X1 X2
      return ((lx & 1u) << 3u)
        | ((ly & 3u) << 4u)
        | (((lx >> 1u) & 3u) << 6u);
    }
    // SW_*_D 8B: 0 0 0 X0 Y0 X1 X2 Y1
    return ((lx & 1u) << 3u)
      | ((ly & 1u) << 4u)
      | (((lx >> 1u) & 3u) << 5u)
      | (((ly >> 1u) & 1u) << 7u);
  }
  // 16B
  if (!isDisplay) {
    // SW_*_S 16B: 0 0 0 0 Y0 Y1 X0 X1
    return ((ly & 3u) << 4u) | ((lx & 3u) << 6u);
  }
  // SW_*_D 16B: 0 0 0 0 X0 Y0 X1 Y1
  return ((lx & 1u) << 4u)
    | ((ly & 1u) << 5u)
    | (((lx >> 1u) & 1u) << 6u)
    | (((ly >> 1u) & 1u) << 7u);
}

fn applyGfx10Xor64k(intra: u32, pipeBankXor: u32) -> u32 {
  let hi = (intra >> 12u) & 0xFu;
  let rev = ((hi & 1u) << 3u) | ((hi & 2u) << 1u) | ((hi & 4u) >> 1u) | ((hi & 8u) >> 3u);
  let seed = (pipeBankXor & 0xFFu) << 8u;
  return intra ^ (rev << 8u) ^ seed;
}

fn computeGfx10TiledOffset2D(bx: u32, by: u32) -> u32 {
  let mode = params.swizzleMode;
  let bpb = params.bytesPerBlock;
  if (mode == 0u) {
    return (by * params.pitchInBlocks + bx) * bpb;
  }

  let isDisplay = (mode == 2u) || (mode == 6u) || (mode == 10u) || (mode == 14u) || (mode == 26u) || (mode == 30u);
  let isXor = (mode == 25u) || (mode == 26u) || (mode == 29u) || (mode == 30u);

  var blkSizeLog2: u32 = 16u;
  if (mode == 1u || mode == 2u) {
    blkSizeLog2 = 8u;
  } else if (mode == 5u || mode == 6u) {
    blkSizeLog2 = 12u;
  }

  var w256: u32 = 2u;
  var h256: u32 = 2u;
  if (bpb == 1u) {
    w256 = 4u;
    h256 = 4u;
  } else if (bpb == 2u) {
    w256 = 4u;
    h256 = 3u;
  } else if (bpb == 4u) {
    w256 = 3u;
    h256 = 3u;
  } else if (bpb == 8u) {
    w256 = 3u;
    h256 = 2u;
  }

  let extraLog2 = (blkSizeLog2 - 8u) >> 1u;
  let wLog2 = w256 + extraLog2;
  let hLog2 = h256 + extraLog2;
  let tileWMask = (1u << wLog2) - 1u;
  let tileHMask = (1u << hLog2) - 1u;

  let tileX = bx >> wLog2;
  let tileY = by >> hLog2;
  let lx = bx & tileWMask;
  let ly = by & tileHMask;

  var intra = computeIntra256(isDisplay, bpb, lx, ly);
  if (blkSizeLog2 >= 12u) {
    let hx = lx >> w256;
    let hy = ly >> h256;
    intra = intra
      | ((hy & 1u) << 8u)
      | ((hx & 1u) << 9u)
      | (((hy >> 1u) & 1u) << 10u)
      | (((hx >> 1u) & 1u) << 11u);
    if (blkSizeLog2 == 16u) {
      intra = intra
        | (((hy >> 2u) & 1u) << 12u)
        | (((hx >> 2u) & 1u) << 13u)
        | (((hy >> 3u) & 1u) << 14u)
        | (((hx >> 3u) & 1u) << 15u);
    }
  }
  if (isXor) {
    intra = applyGfx10Xor64k(intra, params.pipeBankXor);
  }

  var tilesPerRow = params.tilesPerRow;
  if (tilesPerRow == 0u) {
    let tileW = 1u << wLog2;
    let effPitch = select(tileW, params.pitchInBlocks, params.pitchInBlocks > 0u);
    tilesPerRow = (effPitch + tileW - 1u) >> wLog2;
  }

  let tileIndex = tileY * tilesPerRow + tileX;
  return (tileIndex << blkSizeLog2) + intra;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let by = gid.y;
  let bpb = params.bytesPerBlock;
  var bx = gid.x;
  if (bpb == 1u) {
    bx = gid.x * 4u;
  } else if (bpb == 2u) {
    bx = gid.x * 2u;
  }

  if (bx >= params.blocksX || by >= params.blocksY) {
    return;
  }

  let dstRowWordBase = (params.dstByteOffset >> 2u) + by * (params.dstRowPitchBytes >> 2u);

  if (bpb == 1u) {
    var packed: u32 = 0u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let cx = bx + i;
      if (cx < params.blocksX) {
        let tiledOff = computeGfx10TiledOffset2D(cx, by);
        let bVal = readSrcByte(params.srcByteOffset + tiledOff);
        packed = packed | (bVal << (i * 8u));
      }
    }
    writeDstU32(dstRowWordBase + gid.x, packed);
    return;
  }

  if (bpb == 2u) {
    var packed: u32 = 0u;
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
      let cx = bx + i;
      if (cx < params.blocksX) {
        let tiledOff = computeGfx10TiledOffset2D(cx, by);
        let hVal = readSrcU16(params.srcByteOffset + tiledOff);
        packed = packed | (hVal << (i * 16u));
      }
    }
    writeDstU32(dstRowWordBase + gid.x, packed);
    return;
  }

  let tiledOff = computeGfx10TiledOffset2D(bx, by);
  let srcBase = params.srcByteOffset + tiledOff;

  if (bpb == 4u) {
    writeDstU32(dstRowWordBase + bx, readSrcU32(srcBase));
    return;
  }

  if (bpb == 8u) {
    let dstWord = dstRowWordBase + bx * 2u;
    writeDstU32(dstWord, readSrcU32(srcBase));
    writeDstU32(dstWord + 1u, readSrcU32(srcBase + 4u));
    return;
  }

  // bpb == 16u (RGBA32F, BC2/BC3/BC5/BC6H/BC7)
  let dstWord = dstRowWordBase + bx * 4u;
  writeDstU32(dstWord, readSrcU32(srcBase));
  writeDstU32(dstWord + 1u, readSrcU32(srcBase + 4u));
  writeDstU32(dstWord + 2u, readSrcU32(srcBase + 8u));
  writeDstU32(dstWord + 3u, readSrcU32(srcBase + 12u));
}
`.trim();

  /**
   * Return the standard WebGPU WGSL compute shader source for GFX10 2D texture detiling.
   *
   * @returns {string}
   */
  function buildGfx10DetileWgsl() {
    return GFX10_DETILE_WGSL;
  }

  function hasComputeDeviceCapabilities(device) {
    if (!device || typeof device !== 'object') {
      return false;
    }
    return (
      typeof device.createShaderModule === 'function' &&
      typeof device.createBindGroupLayout === 'function' &&
      typeof device.createPipelineLayout === 'function' &&
      typeof device.createComputePipeline === 'function' &&
      typeof device.createBuffer === 'function' &&
      typeof device.createBindGroup === 'function' &&
      device.queue !== null &&
      device.queue !== undefined &&
      typeof device.queue.writeBuffer === 'function'
    );
  }

  /**
   * Cross-GPU WebGPU compute-shader GFX10 2D texture detiler.
   *
   * Lazily compiles and caches a single standard-WGSL compute pipeline, reuses a 48-byte
   * uniform `GPUBuffer` + `Uint32Array(12)` scratch (zero steady-state per-dispatch allocations),
   * and provides `detileBufferToBuffer` + `detileToTexture` (256-byte `bytesPerRow` staging ready).
   */
  class GpuTextureDetiler {
    /**
     * @param {GPUDevice | { device: GPUDevice, labelPrefix?: string, maxStagingBytes?: number }} deviceOrOptions
     */
    constructor(deviceOrOptions) {
      let device = null;
      let labelPrefix = 'playgta6-gfx10-detile';
      let maxStagingBytes = 64 * 1024 * 1024; // 64 MiB safety cap for internal staging buffer

      if (deviceOrOptions && typeof deviceOrOptions === 'object') {
        if ('device' in deviceOrOptions) {
          device = deviceOrOptions.device || null;
          if (typeof deviceOrOptions.labelPrefix === 'string' && deviceOrOptions.labelPrefix.length > 0) {
            labelPrefix = deviceOrOptions.labelPrefix;
          }
          if (
            Number.isInteger(deviceOrOptions.maxStagingBytes) &&
            deviceOrOptions.maxStagingBytes > 0
          ) {
            maxStagingBytes = deviceOrOptions.maxStagingBytes;
          }
        } else {
          device = deviceOrOptions;
        }
      }

      this._device = device;
      this._labelPrefix = labelPrefix;
      this._maxStagingBytes = maxStagingBytes;
      this._destroyed = false;

      // Lazily created WebGPU pipeline & reusable uniform/staging resources.
      this._shaderModule = null;
      this._bindGroupLayout = null;
      this._pipelineLayout = null;
      this._pipeline = null;
      this._uniformBuffer = null;
      this._pipelineCompiled = false;

      this._stagingBuffer = null;
      this._stagingBufferSize = 0;

      // Single-slot bind group cache (avoids redundant createBindGroup calls on repeated buffer pairs).
      this._lastSrcBuffer = null;
      this._lastDstBuffer = null;
      this._lastBindGroup = null;

      // Pre-allocated scratch & descriptor objects (zero allocations per dispatch in steady state).
      this._uniformScratch = new Uint32Array(DETILE_UNIFORM_U32_COUNT);
      this._computePassDesc = { label: `${this._labelPrefix}-pass` };
      this._bgEntry0 = { binding: 0, resource: { buffer: null } };
      this._bgEntry1 = { binding: 1, resource: { buffer: null } };
      this._bgEntry2 = { binding: 2, resource: { buffer: null } };
      this._bgDesc = {
        label: `${this._labelPrefix}-bg`,
        layout: null,
        entries: [this._bgEntry0, this._bgEntry1, this._bgEntry2],
      };
      this._copySrcDesc = {
        buffer: null,
        offset: 0,
        bytesPerRow: 0,
        rowsPerImage: 0,
      };
      this._copyOrigin = { x: 0, y: 0, z: 0 };
      this._copyDstDesc = {
        texture: null,
        mipLevel: 0,
        origin: this._copyOrigin,
        aspect: 'all',
      };
      this._copyExtent = {
        width: 0,
        height: 0,
        depthOrArrayLayers: 1,
      };
      this._toTextureDispatchParams = {
        srcBuffer: null,
        dstBuffer: null,
        swizzleMode: 0,
        blocksX: 0,
        blocksY: 0,
        bytesPerBlock: 0,
        pitchInBlocks: 0,
        pipeBankXor: 0,
        srcByteOffset: 0,
        dstByteOffset: 0,
        dstRowPitchBytes: 0,
        srcByteLimit: undefined,
      };

      // Sanitized telemetry counters.
      this._gpuDetileDispatches = 0;
      this._gpuDetileBytes = 0;
      this._gpuDetileFallbackCount = 0;
    }

    get isDestroyed() {
      return this._destroyed;
    }

    get pipelineCompiled() {
      return this._pipelineCompiled;
    }

    _getMaxWorkgroupsPerDimension() {
      const lim =
        this._device &&
        this._device.limits &&
        typeof this._device.limits.maxComputeWorkgroupsPerDimension === 'number' &&
        this._device.limits.maxComputeWorkgroupsPerDimension > 0
          ? this._device.limits.maxComputeWorkgroupsPerDimension
          : DEFAULT_MAX_WORKGROUPS_PER_DIM;
      return lim;
    }

    _validateDesc(desc, options) {
      const requireBufferRowAlignment = Boolean(options && options.requireBufferRowAlignment);
      const force256RowAlign = Boolean(options && options.force256RowAlign);

      if (this._destroyed) {
        return { ok: false, reason: 'detiler destroyed' };
      }
      if (!hasComputeDeviceCapabilities(this._device)) {
        return { ok: false, reason: 'device lacks WebGPU compute capabilities' };
      }
      if (!desc || typeof desc !== 'object') {
        return { ok: false, reason: 'missing descriptor' };
      }
      if (desc.depth !== undefined && desc.depth !== 1) {
        return { ok: false, reason: '3D textures are not supported by 2D thin GPU detiler' };
      }
      if (desc.dimension !== undefined && desc.dimension !== '2d' && desc.dimension !== 2) {
        return { ok: false, reason: 'only 2D thin textures are supported' };
      }

      const swizzleMode = resolveSwizzleMode(desc);
      const bytesPerBlock = desc.bytesPerBlock;
      const shape = getGfx10TileShape2D(swizzleMode, bytesPerBlock);
      if (!shape.valid) {
        return { ok: false, reason: 'unsupported swizzleMode or bytesPerBlock' };
      }

      const blocksX = desc.blocksX;
      const blocksY = desc.blocksY;
      if (!Number.isInteger(blocksX) || !Number.isInteger(blocksY) || blocksX < 1 || blocksY < 1) {
        return { ok: false, reason: 'blocksX and blocksY must be positive integers' };
      }

      const grid = computeDetileDispatchGrid(blocksX, blocksY, bytesPerBlock);
      const maxWg = this._getMaxWorkgroupsPerDimension();
      if (grid.workgroupsX > maxWg || grid.workgroupsY > maxWg) {
        return { ok: false, reason: 'dispatch workgroup dimensions exceed device limit' };
      }

      const pitchInBlocks =
        desc.pitchInBlocks !== undefined && desc.pitchInBlocks !== 0
          ? desc.pitchInBlocks
          : blocksX;
      if (!Number.isInteger(pitchInBlocks) || pitchInBlocks < blocksX) {
        return { ok: false, reason: 'pitchInBlocks must be an integer >= blocksX' };
      }

      const srcByteOffset = desc.srcByteOffset !== undefined ? desc.srcByteOffset : 0;
      const dstByteOffset = desc.dstByteOffset !== undefined ? desc.dstByteOffset : 0;
      if (
        !Number.isInteger(srcByteOffset) ||
        srcByteOffset < 0 ||
        !Number.isInteger(dstByteOffset) ||
        dstByteOffset < 0
      ) {
        return { ok: false, reason: 'srcByteOffset and dstByteOffset must be non-negative integers' };
      }
      if ((dstByteOffset & 3) !== 0) {
        return { ok: false, reason: 'dstByteOffset must be 4-byte aligned' };
      }

      const tightRowBytes = blocksX * bytesPerBlock;
      let dstRowPitchBytes;
      if (desc.dstRowPitchBytes !== undefined && desc.dstRowPitchBytes !== 0) {
        dstRowPitchBytes = desc.dstRowPitchBytes;
      } else if (force256RowAlign) {
        dstRowPitchBytes = alignBytesPerRow(tightRowBytes);
      } else {
        dstRowPitchBytes = tightRowBytes;
      }

      if (!Number.isInteger(dstRowPitchBytes) || dstRowPitchBytes < tightRowBytes) {
        return { ok: false, reason: 'dstRowPitchBytes must be an integer >= blocksX * bytesPerBlock' };
      }
      if (force256RowAlign && (dstRowPitchBytes % WEBGPU_BYTES_PER_ROW_ALIGNMENT) !== 0) {
        return { ok: false, reason: 'dstRowPitchBytes must be 256-byte aligned for texture copy' };
      }
      if (
        (requireBufferRowAlignment ||
          (desc.dstRowPitchBytes !== undefined && desc.dstRowPitchBytes !== 0)) &&
        (dstRowPitchBytes & 3) !== 0
      ) {
        return { ok: false, reason: 'dstRowPitchBytes must be 4-byte aligned for race-free u32 writes' };
      }

      const pipeBankXor = shape.isXor ? (desc.pipeBankXor || 0) & 0xff : 0;
      const tilesPerRow = shape.isLinear
        ? 0
        : Math.floor((pitchInBlocks + shape.tileW - 1) / shape.tileW);

      const srcFootprintBytes = computeGfx10TiledFootprint2D(
        swizzleMode,
        blocksX,
        blocksY,
        bytesPerBlock,
        pitchInBlocks
      );
      const dstFootprintBytes = dstRowPitchBytes * blocksY;
      if (srcFootprintBytes <= 0 || dstFootprintBytes <= 0) {
        return { ok: false, reason: 'invalid surface byte footprint' };
      }
      if (srcByteOffset + srcFootprintBytes > 0xffffffff || dstByteOffset + dstFootprintBytes > 0xffffffff) {
        return { ok: false, reason: 'buffer offset exceeds 32-bit WGSL storage address space' };
      }

      let srcByteLimit = desc.srcByteLimit;
      if (srcByteLimit !== undefined) {
        if (!Number.isInteger(srcByteLimit) || srcByteLimit < 0 || srcByteLimit > 0xffffffff) {
          return { ok: false, reason: 'srcByteLimit must be a valid uint32 byte count' };
        }
      }

      return {
        ok: true,
        reason: null,
        swizzleMode,
        bytesPerBlock,
        blocksX,
        blocksY,
        pitchInBlocks,
        pipeBankXor,
        srcByteOffset,
        dstByteOffset,
        dstRowPitchBytes,
        srcByteLimit,
        tilesPerRow,
        workgroupsX: grid.workgroupsX,
        workgroupsY: grid.workgroupsY,
        tightRowBytes,
        srcFootprintBytes,
        dstFootprintBytes,
      };
    }

    /**
     * Check whether `desc` can be detiled on the GPU compute path without falling back to CPU.
     *
     * @param {object} desc
     * @returns {boolean}
     */
    canDetile(desc) {
      return this._validateDesc(desc, { requireBufferRowAlignment: false }).ok;
    }

    _ensurePipeline() {
      if (this._pipelineCompiled) {
        return true;
      }
      if (this._destroyed || !hasComputeDeviceCapabilities(this._device)) {
        return false;
      }

      const device = this._device;
      let uniformBuffer = null;
      try {
        const shaderModule = device.createShaderModule({
          label: `${this._labelPrefix}-wgsl`,
          code: GFX10_DETILE_WGSL,
        });
        const bindGroupLayout = device.createBindGroupLayout({
          label: `${this._labelPrefix}-bgl`,
          entries: [
            {
              binding: 0,
              visibility: SHADER_STAGE.COMPUTE,
              buffer: { type: 'uniform' },
            },
            {
              binding: 1,
              visibility: SHADER_STAGE.COMPUTE,
              buffer: { type: 'read-only-storage' },
            },
            {
              binding: 2,
              visibility: SHADER_STAGE.COMPUTE,
              buffer: { type: 'storage' },
            },
          ],
        });
        const pipelineLayout = device.createPipelineLayout({
          label: `${this._labelPrefix}-layout`,
          bindGroupLayouts: [bindGroupLayout],
        });
        const pipeline = device.createComputePipeline({
          label: `${this._labelPrefix}-pipeline`,
          layout: pipelineLayout,
          compute: {
            module: shaderModule,
            entryPoint: 'main',
          },
        });
        uniformBuffer = device.createBuffer({
          label: `${this._labelPrefix}-uniform`,
          size: DETILE_UNIFORM_BYTE_SIZE,
          usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
        });

        if (!shaderModule || !bindGroupLayout || !pipelineLayout || !pipeline || !uniformBuffer) {
          throw new Error('WebGPU compute pipeline initialization returned empty resource');
        }

        this._shaderModule = shaderModule;
        this._bindGroupLayout = bindGroupLayout;
        this._pipelineLayout = pipelineLayout;
        this._pipeline = pipeline;
        this._uniformBuffer = uniformBuffer;
        this._bgEntry0.resource.buffer = uniformBuffer;
        this._bgDesc.layout = bindGroupLayout;
        this._pipelineCompiled = true;
        return true;
      } catch (_) {
        if (uniformBuffer && typeof uniformBuffer.destroy === 'function') {
          try {
            uniformBuffer.destroy();
          } catch (__) {
            // ignore cleanup error
          }
        }
        return false;
      }
    }

    _getOrCreateBindGroup(srcBuffer, dstBuffer) {
      if (
        this._lastBindGroup !== null &&
        this._lastSrcBuffer === srcBuffer &&
        this._lastDstBuffer === dstBuffer
      ) {
        return this._lastBindGroup;
      }
      this._bgEntry1.resource.buffer = srcBuffer;
      this._bgEntry2.resource.buffer = dstBuffer;
      const bindGroup = this._device.createBindGroup(this._bgDesc);
      this._lastSrcBuffer = srcBuffer;
      this._lastDstBuffer = dstBuffer;
      this._lastBindGroup = bindGroup;
      return bindGroup;
    }

    _ensureStagingBuffer(requiredBytes) {
      if (requiredBytes > this._maxStagingBytes) {
        return null;
      }
      if (this._stagingBuffer !== null && this._stagingBufferSize >= requiredBytes) {
        return this._stagingBuffer;
      }
      const nextSize = Math.min(
        this._maxStagingBytes,
        Math.max(
          alignUp(requiredBytes, DEFAULT_STAGING_GRANULARITY),
          this._stagingBufferSize * 2
        )
      );
      const nextBuffer = this._device.createBuffer({
        label: `${this._labelPrefix}-staging`,
        size: nextSize,
        usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
      });
      if (!nextBuffer) {
        return null;
      }
      if (this._stagingBuffer && typeof this._stagingBuffer.destroy === 'function') {
        try {
          this._stagingBuffer.destroy();
        } catch (_) {
          // ignore cleanup error
        }
      }
      if (this._lastDstBuffer === this._stagingBuffer) {
        this._lastDstBuffer = null;
        this._lastBindGroup = null;
      }
      this._stagingBuffer = nextBuffer;
      this._stagingBufferSize = nextSize;
      return nextBuffer;
    }

    _recordFallback(reason) {
      this._gpuDetileFallbackCount += 1;
      return {
        ok: false,
        fallback: true,
        reason,
      };
    }

    /**
     * Encode a GPU compute pass that detiles a 2D GFX10 tiled source buffer into a linear
     * destination buffer. Zero steady-state allocations.
     *
     * @param {GPUCommandEncoder} encoder
     * @param {object} params
     */
    detileBufferToBuffer(encoder, params) {
      if (!encoder || typeof encoder.beginComputePass !== 'function') {
        return this._recordFallback('invalid GPUCommandEncoder');
      }
      const checked = this._validateDesc(params, { requireBufferRowAlignment: true });
      if (!checked.ok) {
        return this._recordFallback(checked.reason);
      }

      const srcBuffer = params.srcBuffer;
      const dstBuffer = params.dstBuffer;
      if (!srcBuffer || !dstBuffer) {
        return this._recordFallback('missing srcBuffer or dstBuffer');
      }

      const dstRequiredSize = checked.dstByteOffset + checked.dstFootprintBytes;
      if (typeof dstBuffer.size === 'number' && dstBuffer.size < dstRequiredSize) {
        return this._recordFallback('dstBuffer smaller than required linear output footprint');
      }

      const maxStorageSize =
        this._device &&
        this._device.limits &&
        typeof this._device.limits.maxStorageBufferBindingSize === 'number'
          ? this._device.limits.maxStorageBufferBindingSize
          : 0;
      if (maxStorageSize > 0) {
        if (
          (typeof srcBuffer.size === 'number' && srcBuffer.size > maxStorageSize) ||
          (typeof dstBuffer.size === 'number' && dstBuffer.size > maxStorageSize)
        ) {
          return this._recordFallback('buffer size exceeds maxStorageBufferBindingSize');
        }
      }

      const srcBufferSize =
        typeof srcBuffer.size === 'number' && Number.isInteger(srcBuffer.size) && srcBuffer.size >= 0
          ? srcBuffer.size
          : checked.srcByteOffset + checked.srcFootprintBytes;
      const srcByteLimit =
        checked.srcByteLimit !== undefined
          ? Math.min(checked.srcByteLimit, srcBufferSize)
          : srcBufferSize;

      if (!this._ensurePipeline()) {
        return this._recordFallback('failed to compile WebGPU detile compute pipeline');
      }

      const u = this._uniformScratch;
      u[0] = checked.blocksX >>> 0;
      u[1] = checked.blocksY >>> 0;
      u[2] = checked.bytesPerBlock >>> 0;
      u[3] = checked.pitchInBlocks >>> 0;
      u[4] = checked.swizzleMode >>> 0;
      u[5] = checked.pipeBankXor >>> 0;
      u[6] = checked.srcByteOffset >>> 0;
      u[7] = checked.dstByteOffset >>> 0;
      u[8] = checked.dstRowPitchBytes >>> 0;
      u[9] = srcByteLimit >>> 0;
      u[10] = checked.tilesPerRow >>> 0;
      u[11] = 0;

      this._device.queue.writeBuffer(this._uniformBuffer, 0, u);

      const bindGroup = this._getOrCreateBindGroup(srcBuffer, dstBuffer);
      const pass = encoder.beginComputePass(this._computePassDesc);
      pass.setPipeline(this._pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(checked.workgroupsX, checked.workgroupsY, 1);
      pass.end();

      const bytesDetiled = checked.tightRowBytes * checked.blocksY;
      this._gpuDetileDispatches += 1;
      this._gpuDetileBytes += bytesDetiled;

      return {
        ok: true,
        fallback: false,
        workgroupsX: checked.workgroupsX,
        workgroupsY: checked.workgroupsY,
        dstRowPitchBytes: checked.dstRowPitchBytes,
        srcByteLimit,
        bytesDetiled,
      };
    }

    /**
     * Dispatch compute detiling into a 256-byte-row-aligned linear staging buffer, and optionally
     * encode `copyBufferToTexture` into `params.dstTexture`.
     *
     * @param {GPUCommandEncoder} encoder
     * @param {object} params
     */
    detileToTexture(encoder, params) {
      if (!encoder || typeof encoder.beginComputePass !== 'function') {
        return this._recordFallback('invalid GPUCommandEncoder');
      }
      const checked = this._validateDesc(params, {
        requireBufferRowAlignment: true,
        force256RowAlign: true,
      });
      if (!checked.ok) {
        return this._recordFallback(checked.reason);
      }
      if (!params.srcBuffer) {
        return this._recordFallback('missing srcBuffer');
      }

      const dstTexture = params.dstTexture || null;
      if (dstTexture && typeof encoder.copyBufferToTexture !== 'function') {
        return this._recordFallback('encoder lacks copyBufferToTexture');
      }

      const requiredStagingBytes = checked.dstByteOffset + checked.dstFootprintBytes;
      let stagingBuffer = params.stagingBuffer || null;
      if (stagingBuffer) {
        if (typeof stagingBuffer.size === 'number' && stagingBuffer.size < requiredStagingBytes) {
          return this._recordFallback('provided stagingBuffer is too small');
        }
      } else {
        stagingBuffer = this._ensureStagingBuffer(requiredStagingBytes);
        if (!stagingBuffer) {
          return this._recordFallback('failed to allocate internal staging buffer');
        }
      }

      // Reuse pre-allocated dispatch params object (zero steady-state allocations).
      const dp = this._toTextureDispatchParams;
      dp.srcBuffer = params.srcBuffer;
      dp.dstBuffer = stagingBuffer;
      dp.swizzleMode = checked.swizzleMode;
      dp.blocksX = checked.blocksX;
      dp.blocksY = checked.blocksY;
      dp.bytesPerBlock = checked.bytesPerBlock;
      dp.pitchInBlocks = checked.pitchInBlocks;
      dp.pipeBankXor = checked.pipeBankXor;
      dp.srcByteOffset = checked.srcByteOffset;
      dp.dstByteOffset = checked.dstByteOffset;
      dp.dstRowPitchBytes = checked.dstRowPitchBytes;
      dp.srcByteLimit = checked.srcByteLimit;

      const dispatchRes = this.detileBufferToBuffer(encoder, dp);
      dp.srcBuffer = null;
      dp.dstBuffer = null;
      if (!dispatchRes.ok) {
        return dispatchRes;
      }

      let copiedToTexture = false;
      if (dstTexture) {
        const blockWidth =
          Number.isInteger(params.blockWidth) && params.blockWidth > 0 ? params.blockWidth : 1;
        const blockHeight =
          Number.isInteger(params.blockHeight) && params.blockHeight > 0 ? params.blockHeight : 1;
        const copyWidth =
          Number.isInteger(params.copyWidth) && params.copyWidth > 0
            ? params.copyWidth
            : checked.blocksX * blockWidth;
        const copyHeight =
          Number.isInteger(params.copyHeight) && params.copyHeight > 0
            ? params.copyHeight
            : checked.blocksY * blockHeight;
        const origin = params.origin;

        this._copySrcDesc.buffer = stagingBuffer;
        this._copySrcDesc.offset = checked.dstByteOffset;
        this._copySrcDesc.bytesPerRow = checked.dstRowPitchBytes;
        this._copySrcDesc.rowsPerImage = checked.blocksY;

        this._copyOrigin.x = origin && Number.isInteger(origin.x) ? origin.x : 0;
        this._copyOrigin.y = origin && Number.isInteger(origin.y) ? origin.y : 0;
        this._copyOrigin.z =
          origin && Number.isInteger(origin.z)
            ? origin.z
            : Number.isInteger(params.arrayLayer) && params.arrayLayer >= 0
              ? params.arrayLayer
              : 0;

        this._copyDstDesc.texture = dstTexture;
        this._copyDstDesc.mipLevel =
          Number.isInteger(params.mipLevel) && params.mipLevel >= 0 ? params.mipLevel : 0;
        this._copyDstDesc.aspect = typeof params.aspect === 'string' ? params.aspect : 'all';

        this._copyExtent.width = copyWidth;
        this._copyExtent.height = copyHeight;
        this._copyExtent.depthOrArrayLayers = 1;

        encoder.copyBufferToTexture(this._copySrcDesc, this._copyDstDesc, this._copyExtent);
        copiedToTexture = true;
      }

      return {
        ok: true,
        fallback: false,
        stagingBuffer,
        bytesPerRow: checked.dstRowPitchBytes,
        rowsPerImage: checked.blocksY,
        copiedToTexture,
        workgroupsX: dispatchRes.workgroupsX,
        workgroupsY: dispatchRes.workgroupsY,
        bytesDetiled: dispatchRes.bytesDetiled,
      };
    }

    /**
     * Sanitized telemetry snapshot (defensive plain object).
     */
    getStats() {
      return {
        gpuDetileDispatches: this._gpuDetileDispatches,
        gpuDetileBytes: this._gpuDetileBytes,
        gpuDetileFallbackCount: this._gpuDetileFallbackCount,
        pipelineCompiled: this._pipelineCompiled,
        stagingBufferBytes: this._stagingBufferSize,
        destroyed: this._destroyed,
      };
    }

    /**
     * Idempotent GPU resource cleanup. Destroys owned uniform & staging buffers without touching
     * caller-owned buffers.
     */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      if (this._uniformBuffer && typeof this._uniformBuffer.destroy === 'function') {
        try {
          this._uniformBuffer.destroy();
        } catch (_) {
          // ignore cleanup error
        }
      }
      if (this._stagingBuffer && typeof this._stagingBuffer.destroy === 'function') {
        try {
          this._stagingBuffer.destroy();
        } catch (_) {
          // ignore cleanup error
        }
      }

      this._shaderModule = null;
      this._bindGroupLayout = null;
      this._pipelineLayout = null;
      this._pipeline = null;
      this._uniformBuffer = null;
      this._stagingBuffer = null;
      this._stagingBufferSize = 0;
      this._pipelineCompiled = false;
      this._lastSrcBuffer = null;
      this._lastDstBuffer = null;
      this._lastBindGroup = null;
      this._bgEntry0.resource.buffer = null;
      this._bgEntry1.resource.buffer = null;
      this._bgEntry2.resource.buffer = null;
      this._bgDesc.layout = null;
      this._copySrcDesc.buffer = null;
      this._copyDstDesc.texture = null;
      this._device = null;
    }
  }

  const exportedApi = {
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
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6GpuDetiler = exportedApi;
  } else if (root) {
    root.PlayGta6GpuDetiler = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
