// DXGI / Xbox Series texture format mapper, WebGPU 256-byte mip-chain layout &
// upload alignment helpers, and CPU fallback BC1/BC3/BC4 block decoders.
// Zero external dependencies; runs in Web Workers (importScripts), browsers, and Node.js.
// Designed for runtime/gpu <-> web/workers/wgpu_worker.js (docs/gpu_commands.md).

(function (globalScope) {
  'use strict';

  const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;
  const TEXTURE_COMPRESSION_BC_FEATURE = 'texture-compression-bc';
  const DEPTH32FLOAT_STENCIL8_FEATURE = 'depth32float-stencil8';

  /**
   * Standard DirectX / Xbox Series DXGI_FORMAT numeric enum values (dxgiformat.h).
   */
  const DXGI_FORMAT = Object.freeze({
    UNKNOWN: 0,
    R32G32B32A32_FLOAT: 2,
    R16G16B16A16_FLOAT: 10,
    D32_FLOAT_S8X24_UINT: 20,
    R10G10B10A2_UNORM: 24,
    R11G11B10_FLOAT: 26,
    R8G8B8A8_UNORM: 28,
    R8G8B8A8_UNORM_SRGB: 29,
    R8G8B8A8_UINT: 30,
    R8G8B8A8_SNORM: 31,
    R8G8B8A8_SINT: 32,
    R16G16_FLOAT: 34,
    D32_FLOAT: 40,
    R32_FLOAT: 41,
    R32_UINT: 42,
    R32_SINT: 43,
    D24_UNORM_S8_UINT: 45,
    R8G8_UNORM: 49,
    R16_FLOAT: 54,
    D16_UNORM: 55,
    R8_UNORM: 61,
    BC1_UNORM: 71,
    BC1_UNORM_SRGB: 72,
    BC2_UNORM: 74,
    BC2_UNORM_SRGB: 75,
    BC3_UNORM: 77,
    BC3_UNORM_SRGB: 78,
    BC4_UNORM: 80,
    BC4_SNORM: 81,
    BC5_UNORM: 83,
    BC5_SNORM: 84,
    B8G8R8A8_UNORM: 87,
    B8G8R8A8_UNORM_SRGB: 91,
    BC6H_UF16: 95,
    BC6H_SF16: 96,
    BC7_UNORM: 98,
    BC7_UNORM_SRGB: 99,
  });

  const DXGI_FORMAT_NAMES = Object.freeze(
    Object.fromEntries(
      Object.entries(DXGI_FORMAT).map(([name, code]) => [code, name])
    )
  );

  function makeEntry(
    dxgiFormat,
    webgpuFormat,
    blockWidth,
    blockHeight,
    bytesPerBlock,
    options = {}
  ) {
    return Object.freeze({
      dxgiFormat,
      name: DXGI_FORMAT_NAMES[dxgiFormat],
      webgpuFormat,
      blockWidth,
      blockHeight,
      bytesPerBlock,
      isCompressed: Boolean(options.isCompressed),
      isDepth: Boolean(options.isDepth),
      isSrgb: Boolean(options.isSrgb),
      requiresFeature: options.requiresFeature || null,
    });
  }

  // Canonical DXGI_FORMAT -> WebGPU GPUTextureFormat descriptor table.
  const FORMAT_TABLE = Object.freeze({
    [DXGI_FORMAT.R32G32B32A32_FLOAT]: makeEntry(DXGI_FORMAT.R32G32B32A32_FLOAT, 'rgba32float', 1, 1, 16),
    [DXGI_FORMAT.R16G16B16A16_FLOAT]: makeEntry(DXGI_FORMAT.R16G16B16A16_FLOAT, 'rgba16float', 1, 1, 8),
    [DXGI_FORMAT.D32_FLOAT_S8X24_UINT]: makeEntry(
      DXGI_FORMAT.D32_FLOAT_S8X24_UINT,
      'depth32float-stencil8',
      1,
      1,
      8,
      { isDepth: true, requiresFeature: DEPTH32FLOAT_STENCIL8_FEATURE }
    ),
    [DXGI_FORMAT.R10G10B10A2_UNORM]: makeEntry(DXGI_FORMAT.R10G10B10A2_UNORM, 'rgb10a2unorm', 1, 1, 4),
    [DXGI_FORMAT.R11G11B10_FLOAT]: makeEntry(DXGI_FORMAT.R11G11B10_FLOAT, 'rg11b10ufloat', 1, 1, 4),
    [DXGI_FORMAT.R8G8B8A8_UNORM]: makeEntry(DXGI_FORMAT.R8G8B8A8_UNORM, 'rgba8unorm', 1, 1, 4),
    [DXGI_FORMAT.R8G8B8A8_UNORM_SRGB]: makeEntry(
      DXGI_FORMAT.R8G8B8A8_UNORM_SRGB,
      'rgba8unorm-srgb',
      1,
      1,
      4,
      { isSrgb: true }
    ),
    [DXGI_FORMAT.R8G8B8A8_UINT]: makeEntry(DXGI_FORMAT.R8G8B8A8_UINT, 'rgba8uint', 1, 1, 4),
    [DXGI_FORMAT.R8G8B8A8_SNORM]: makeEntry(DXGI_FORMAT.R8G8B8A8_SNORM, 'rgba8snorm', 1, 1, 4),
    [DXGI_FORMAT.R8G8B8A8_SINT]: makeEntry(DXGI_FORMAT.R8G8B8A8_SINT, 'rgba8sint', 1, 1, 4),
    [DXGI_FORMAT.R16G16_FLOAT]: makeEntry(DXGI_FORMAT.R16G16_FLOAT, 'rg16float', 1, 1, 4),
    [DXGI_FORMAT.D32_FLOAT]: makeEntry(DXGI_FORMAT.D32_FLOAT, 'depth32float', 1, 1, 4, { isDepth: true }),
    [DXGI_FORMAT.R32_FLOAT]: makeEntry(DXGI_FORMAT.R32_FLOAT, 'r32float', 1, 1, 4),
    [DXGI_FORMAT.R32_UINT]: makeEntry(DXGI_FORMAT.R32_UINT, 'r32uint', 1, 1, 4),
    [DXGI_FORMAT.R32_SINT]: makeEntry(DXGI_FORMAT.R32_SINT, 'r32sint', 1, 1, 4),
    [DXGI_FORMAT.D24_UNORM_S8_UINT]: makeEntry(
      DXGI_FORMAT.D24_UNORM_S8_UINT,
      'depth24plus-stencil8',
      1,
      1,
      4,
      { isDepth: true }
    ),
    [DXGI_FORMAT.R8G8_UNORM]: makeEntry(DXGI_FORMAT.R8G8_UNORM, 'rg8unorm', 1, 1, 2),
    [DXGI_FORMAT.R16_FLOAT]: makeEntry(DXGI_FORMAT.R16_FLOAT, 'r16float', 1, 1, 2),
    [DXGI_FORMAT.D16_UNORM]: makeEntry(DXGI_FORMAT.D16_UNORM, 'depth16unorm', 1, 1, 2, { isDepth: true }),
    [DXGI_FORMAT.R8_UNORM]: makeEntry(DXGI_FORMAT.R8_UNORM, 'r8unorm', 1, 1, 1),
    [DXGI_FORMAT.B8G8R8A8_UNORM]: makeEntry(DXGI_FORMAT.B8G8R8A8_UNORM, 'bgra8unorm', 1, 1, 4),
    [DXGI_FORMAT.B8G8R8A8_UNORM_SRGB]: makeEntry(
      DXGI_FORMAT.B8G8R8A8_UNORM_SRGB,
      'bgra8unorm-srgb',
      1,
      1,
      4,
      { isSrgb: true }
    ),

    // BC1..BC7 4x4 block-compressed formats (require 'texture-compression-bc' in WebGPU).
    [DXGI_FORMAT.BC1_UNORM]: makeEntry(DXGI_FORMAT.BC1_UNORM, 'bc1-rgba-unorm', 4, 4, 8, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC1_UNORM_SRGB]: makeEntry(DXGI_FORMAT.BC1_UNORM_SRGB, 'bc1-rgba-unorm-srgb', 4, 4, 8, {
      isCompressed: true,
      isSrgb: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC2_UNORM]: makeEntry(DXGI_FORMAT.BC2_UNORM, 'bc2-rgba-unorm', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC2_UNORM_SRGB]: makeEntry(DXGI_FORMAT.BC2_UNORM_SRGB, 'bc2-rgba-unorm-srgb', 4, 4, 16, {
      isCompressed: true,
      isSrgb: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC3_UNORM]: makeEntry(DXGI_FORMAT.BC3_UNORM, 'bc3-rgba-unorm', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC3_UNORM_SRGB]: makeEntry(DXGI_FORMAT.BC3_UNORM_SRGB, 'bc3-rgba-unorm-srgb', 4, 4, 16, {
      isCompressed: true,
      isSrgb: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC4_UNORM]: makeEntry(DXGI_FORMAT.BC4_UNORM, 'bc4-r-unorm', 4, 4, 8, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC4_SNORM]: makeEntry(DXGI_FORMAT.BC4_SNORM, 'bc4-r-snorm', 4, 4, 8, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC5_UNORM]: makeEntry(DXGI_FORMAT.BC5_UNORM, 'bc5-rg-unorm', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC5_SNORM]: makeEntry(DXGI_FORMAT.BC5_SNORM, 'bc5-rg-snorm', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC6H_UF16]: makeEntry(DXGI_FORMAT.BC6H_UF16, 'bc6h-rgb-ufloat', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC6H_SF16]: makeEntry(DXGI_FORMAT.BC6H_SF16, 'bc6h-rgb-float', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC7_UNORM]: makeEntry(DXGI_FORMAT.BC7_UNORM, 'bc7-rgba-unorm', 4, 4, 16, {
      isCompressed: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
    [DXGI_FORMAT.BC7_UNORM_SRGB]: makeEntry(DXGI_FORMAT.BC7_UNORM_SRGB, 'bc7-rgba-unorm-srgb', 4, 4, 16, {
      isCompressed: true,
      isSrgb: true,
      requiresFeature: TEXTURE_COMPRESSION_BC_FEATURE,
    }),
  });

  function resolveDxgiFormatCode(dxgiFormat) {
    if (typeof dxgiFormat === 'number') {
      if (Number.isInteger(dxgiFormat) && Object.prototype.hasOwnProperty.call(FORMAT_TABLE, dxgiFormat)) {
        return dxgiFormat;
      }
      throw new RangeError(`Unsupported DXGI_FORMAT value: ${dxgiFormat}`);
    }
    if (typeof dxgiFormat === 'string') {
      const normalized = dxgiFormat.trim().toUpperCase().replace(/^DXGI_FORMAT_/, '');
      if (Object.prototype.hasOwnProperty.call(DXGI_FORMAT, normalized)) {
        const code = DXGI_FORMAT[normalized];
        if (Object.prototype.hasOwnProperty.call(FORMAT_TABLE, code)) {
          return code;
        }
      }
      throw new RangeError(`Unsupported DXGI_FORMAT name: ${JSON.stringify(dxgiFormat)}`);
    }
    throw new TypeError(`dxgiFormat must be a DXGI_FORMAT integer or name string`);
  }

  function hasSupportedFeature(supportedFeatures, featureName) {
    if (!supportedFeatures || !featureName) {
      return false;
    }
    if (typeof supportedFeatures.has === 'function') {
      return Boolean(supportedFeatures.has(featureName));
    }
    if (Array.isArray(supportedFeatures)) {
      return supportedFeatures.includes(featureName);
    }
    if (typeof supportedFeatures[Symbol.iterator] === 'function') {
      for (const item of supportedFeatures) {
        if (item === featureName) {
          return true;
        }
      }
      return false;
    }
    if (typeof supportedFeatures === 'object') {
      return Boolean(supportedFeatures[featureName]);
    }
    return false;
  }

  function toUint8Array(input, label) {
    if (input instanceof Uint8Array) {
      return input;
    }
    if (ArrayBuffer.isView(input)) {
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    if (input instanceof ArrayBuffer) {
      return new Uint8Array(input);
    }
    if (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      `${label} expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`
    );
  }

  function assertPositiveInteger(value, name) {
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got ${value}`);
    }
  }

  function assertNonNegativeInteger(value, name) {
    if (!Number.isInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative integer, got ${value}`);
    }
  }

  /**
   * Map a DXGI / Xbox texture format to WebGPU GPUTextureFormat metadata.
   *
   * When a BCn format is requested and `supportedFeatures` does not contain
   * `'texture-compression-bc'`, `needsSoftwareDecode` is set to `true` and
   * `webgpuFormat` falls back to `'rgba8unorm'` or `'rgba8unorm-srgb'`.
   *
   * @param {number | string} dxgiFormat
   * @param {Set<string> | string[] | Iterable<string> | Record<string, boolean> | null} [supportedFeatures]
   */
  function mapDxgiFormatToWebGpu(dxgiFormat, supportedFeatures) {
    const code = resolveDxgiFormatCode(dxgiFormat);
    const entry = FORMAT_TABLE[code];

    let webgpuFormat = entry.webgpuFormat;
    let needsSoftwareDecode = false;

    if (entry.requiresFeature === TEXTURE_COMPRESSION_BC_FEATURE) {
      const hasBc = hasSupportedFeature(supportedFeatures, TEXTURE_COMPRESSION_BC_FEATURE);
      if (!hasBc) {
        needsSoftwareDecode = true;
        webgpuFormat = entry.isSrgb ? 'rgba8unorm-srgb' : 'rgba8unorm';
      }
    } else if (
      entry.requiresFeature === DEPTH32FLOAT_STENCIL8_FEATURE &&
      !hasSupportedFeature(supportedFeatures, DEPTH32FLOAT_STENCIL8_FEATURE)
    ) {
      webgpuFormat = 'depth24plus-stencil8';
    }

    return {
      dxgiFormat: entry.dxgiFormat,
      name: entry.name,
      webgpuFormat,
      nativeWebgpuFormat: entry.webgpuFormat,
      blockWidth: entry.blockWidth,
      blockHeight: entry.blockHeight,
      bytesPerBlock: entry.bytesPerBlock,
      isCompressed: entry.isCompressed,
      isDepth: entry.isDepth,
      isSrgb: entry.isSrgb,
      requiresFeature: entry.requiresFeature,
      needsSoftwareDecode,
    };
  }

  /**
   * Compute mip-level dimensions clamped down to 1x1.
   * By default `depthOrLayers` is treated as array layers (unchanged across mips);
   * pass `is3D = true` for 3D volume textures whose depth halves per mip level.
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrLayers=1]
   * @param {number} [mipLevel=0]
   * @param {boolean} [is3D=false]
   * @returns {{ width: number, height: number, depthOrLayers: number }}
   */
  function calcMipSize(width, height, depthOrLayers = 1, mipLevel = 0, is3D = false) {
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');
    assertPositiveInteger(depthOrLayers, 'depthOrLayers');
    assertNonNegativeInteger(mipLevel, 'mipLevel');

    const scale = mipLevel >= 31 ? Infinity : Math.pow(2, mipLevel);
    const mipWidth = scale === Infinity ? 1 : Math.max(1, Math.floor(width / scale));
    const mipHeight = scale === Infinity ? 1 : Math.max(1, Math.floor(height / scale));
    const mipDepth = is3D
      ? (scale === Infinity ? 1 : Math.max(1, Math.floor(depthOrLayers / scale)))
      : depthOrLayers;

    return {
      width: mipWidth,
      height: mipHeight,
      depthOrLayers: mipDepth,
    };
  }

  /**
   * Compute block grid counts, unaligned row stride, WebGPU-aligned `bytesPerRow`
   * (default 256-byte alignment), `rowsPerImage`, and `totalBytes` for a mip level.
   *
   * @param {object | number | string} formatInfo
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrLayers=1]
   * @param {number} [alignBytesPerRow=256]
   */
  function calcMipLayout(
    formatInfo,
    width,
    height,
    depthOrLayers = 1,
    alignBytesPerRow = WEBGPU_BYTES_PER_ROW_ALIGNMENT
  ) {
    let info = formatInfo;
    if (typeof formatInfo === 'number' || typeof formatInfo === 'string') {
      info = FORMAT_TABLE[resolveDxgiFormatCode(formatInfo)];
    }
    if (!info || typeof info !== 'object') {
      throw new TypeError('formatInfo must be a format descriptor object or DXGI_FORMAT');
    }

    assertPositiveInteger(info.blockWidth, 'formatInfo.blockWidth');
    assertPositiveInteger(info.blockHeight, 'formatInfo.blockHeight');
    assertPositiveInteger(info.bytesPerBlock, 'formatInfo.bytesPerBlock');
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');
    assertPositiveInteger(depthOrLayers, 'depthOrLayers');
    assertPositiveInteger(alignBytesPerRow, 'alignBytesPerRow');

    const blocksX = Math.ceil(width / info.blockWidth);
    const blocksY = Math.ceil(height / info.blockHeight);
    const unalignedBytesPerRow = blocksX * info.bytesPerBlock;
    const bytesPerRow = Math.ceil(unalignedBytesPerRow / alignBytesPerRow) * alignBytesPerRow;
    const rowsPerImage = blocksY;
    const totalBytes = bytesPerRow * rowsPerImage * depthOrLayers;

    return {
      width,
      height,
      depthOrLayers,
      blocksX,
      blocksY,
      unalignedBytesPerRow,
      bytesPerRow,
      rowsPerImage,
      totalBytes,
    };
  }

  /**
   * Pad (or tightly copy) texture rows so every row starts on an `alignedBytesPerRow`
   * boundary (256 bytes for WebGPU `copyBufferToTexture` / staging uploads).
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} unalignedBytesPerRow
   * @param {number} alignedBytesPerRow
   * @param {number} rowCount
   * @returns {Uint8Array}
   */
  function padTextureRowsForWriteTexture(
    srcBytes,
    unalignedBytesPerRow,
    alignedBytesPerRow,
    rowCount
  ) {
    const src = toUint8Array(srcBytes, 'padTextureRowsForWriteTexture');
    assertNonNegativeInteger(unalignedBytesPerRow, 'unalignedBytesPerRow');
    assertNonNegativeInteger(alignedBytesPerRow, 'alignedBytesPerRow');
    assertNonNegativeInteger(rowCount, 'rowCount');

    if (alignedBytesPerRow < unalignedBytesPerRow) {
      throw new RangeError(
        `alignedBytesPerRow (${alignedBytesPerRow}) must be >= unalignedBytesPerRow (${unalignedBytesPerRow})`
      );
    }

    const requiredSrcBytes = unalignedBytesPerRow * rowCount;
    if (src.byteLength < requiredSrcBytes) {
      throw new RangeError(
        `srcBytes too small (${src.byteLength} bytes; need at least ${requiredSrcBytes} bytes for ${rowCount} rows)`
      );
    }

    const dstBytes = alignedBytesPerRow * rowCount;
    if (alignedBytesPerRow === unalignedBytesPerRow) {
      return src.slice(0, dstBytes);
    }

    const dst = new Uint8Array(dstBytes);
    for (let row = 0; row < rowCount; row += 1) {
      const srcOffset = row * unalignedBytesPerRow;
      dst.set(src.subarray(srcOffset, srcOffset + unalignedBytesPerRow), row * alignedBytesPerRow);
    }
    return dst;
  }

  function decodeRgb565(c, outPalette, baseOffset) {
    const r5 = (c >>> 11) & 0x1f;
    const g6 = (c >>> 5) & 0x3f;
    const b5 = c & 0x1f;
    outPalette[baseOffset] = (r5 << 3) | (r5 >>> 2);
    outPalette[baseOffset + 1] = (g6 << 2) | (g6 >>> 4);
    outPalette[baseOffset + 2] = (b5 << 3) | (b5 >>> 2);
    outPalette[baseOffset + 3] = 255;
  }

  // Decode an 8-byte BC4 / BC3-alpha interpolated endpoint table into `outTable` (length 8).
  function decodeBc4Palette(endpoint0, endpoint1, outTable) {
    outTable[0] = endpoint0;
    outTable[1] = endpoint1;
    if (endpoint0 > endpoint1) {
      outTable[2] = Math.floor((6 * endpoint0 + 1 * endpoint1 + 3) / 7);
      outTable[3] = Math.floor((5 * endpoint0 + 2 * endpoint1 + 3) / 7);
      outTable[4] = Math.floor((4 * endpoint0 + 3 * endpoint1 + 3) / 7);
      outTable[5] = Math.floor((3 * endpoint0 + 4 * endpoint1 + 3) / 7);
      outTable[6] = Math.floor((2 * endpoint0 + 5 * endpoint1 + 3) / 7);
      outTable[7] = Math.floor((1 * endpoint0 + 6 * endpoint1 + 3) / 7);
    } else {
      outTable[2] = Math.floor((4 * endpoint0 + 1 * endpoint1 + 2) / 5);
      outTable[3] = Math.floor((3 * endpoint0 + 2 * endpoint1 + 2) / 5);
      outTable[4] = Math.floor((2 * endpoint0 + 3 * endpoint1 + 2) / 5);
      outTable[5] = Math.floor((1 * endpoint0 + 4 * endpoint1 + 2) / 5);
      outTable[6] = 0;
      outTable[7] = 255;
    }
  }

  /**
   * CPU fallback BC1 / DXT1 decoder (4x4 RGB565 4-color opaque and
   * 3-color + 1-bit alpha punchthrough modes) -> tightly packed RGBA8 Uint8Array.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} width
   * @param {number} height
   * @returns {Uint8Array}
   */
  function decodeBc1ToRgba8(srcBytes, width, height) {
    const src = toUint8Array(srcBytes, 'decodeBc1ToRgba8');
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');

    const blocksX = Math.ceil(width / 4);
    const blocksY = Math.ceil(height / 4);
    const requiredBytes = blocksX * blocksY * 8;
    if (src.byteLength < requiredBytes) {
      throw new RangeError(
        `BC1 input too small (${src.byteLength} bytes; need ${requiredBytes} bytes for ${width}x${height})`
      );
    }

    const out = new Uint8Array(width * height * 4);
    const palette = new Uint8Array(16);

    let blockOffset = 0;
    for (let by = 0; by < blocksY; by += 1) {
      const baseY = by * 4;
      for (let bx = 0; bx < blocksX; bx += 1) {
        const baseX = bx * 4;
        const color0 = src[blockOffset] | (src[blockOffset + 1] << 8);
        const color1 = src[blockOffset + 2] | (src[blockOffset + 3] << 8);
        const indices =
          (src[blockOffset + 4] |
            (src[blockOffset + 5] << 8) |
            (src[blockOffset + 6] << 16) |
            (src[blockOffset + 7] << 24)) >>> 0;
        blockOffset += 8;

        decodeRgb565(color0, palette, 0);
        decodeRgb565(color1, palette, 4);

        const r0 = palette[0];
        const g0 = palette[1];
        const b0 = palette[2];
        const r1 = palette[4];
        const g1 = palette[5];
        const b1 = palette[6];

        if (color0 > color1) {
          // 4-color opaque interpolation mode
          palette[8] = Math.floor((2 * r0 + r1 + 1) / 3);
          palette[9] = Math.floor((2 * g0 + g1 + 1) / 3);
          palette[10] = Math.floor((2 * b0 + b1 + 1) / 3);
          palette[11] = 255;

          palette[12] = Math.floor((r0 + 2 * r1 + 1) / 3);
          palette[13] = Math.floor((g0 + 2 * g1 + 1) / 3);
          palette[14] = Math.floor((b0 + 2 * b1 + 1) / 3);
          palette[15] = 255;
        } else {
          // 3-color + 1-bit transparent black punchthrough mode
          palette[8] = (r0 + r1) >>> 1;
          palette[9] = (g0 + g1) >>> 1;
          palette[10] = (b0 + b1) >>> 1;
          palette[11] = 255;

          palette[12] = 0;
          palette[13] = 0;
          palette[14] = 0;
          palette[15] = 0;
        }

        for (let py = 0; py < 4; py += 1) {
          const y = baseY + py;
          if (y >= height) {
            break;
          }
          const rowOutBase = y * width * 4;
          const rowShiftBase = py * 8;
          for (let px = 0; px < 4; px += 1) {
            const x = baseX + px;
            if (x >= width) {
              break;
            }
            const palIdx = ((indices >>> (rowShiftBase + px * 2)) & 0x3) * 4;
            const dstOff = rowOutBase + x * 4;
            out[dstOff] = palette[palIdx];
            out[dstOff + 1] = palette[palIdx + 1];
            out[dstOff + 2] = palette[palIdx + 2];
            out[dstOff + 3] = palette[palIdx + 3];
          }
        }
      }
    }

    return out;
  }

  /**
   * CPU fallback BC3 / DXT5 decoder (8-level / 6-level interpolated alpha block
   * + 4-color RGB565 block) -> tightly packed RGBA8 Uint8Array.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} width
   * @param {number} height
   * @returns {Uint8Array}
   */
  function decodeBc3ToRgba8(srcBytes, width, height) {
    const src = toUint8Array(srcBytes, 'decodeBc3ToRgba8');
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');

    const blocksX = Math.ceil(width / 4);
    const blocksY = Math.ceil(height / 4);
    const requiredBytes = blocksX * blocksY * 16;
    if (src.byteLength < requiredBytes) {
      throw new RangeError(
        `BC3 input too small (${src.byteLength} bytes; need ${requiredBytes} bytes for ${width}x${height})`
      );
    }

    const out = new Uint8Array(width * height * 4);
    const alphaTable = new Uint8Array(8);
    const colorPalette = new Uint8Array(16);

    let blockOffset = 0;
    for (let by = 0; by < blocksY; by += 1) {
      const baseY = by * 4;
      for (let bx = 0; bx < blocksX; bx += 1) {
        const baseX = bx * 4;

        // Bytes 0..7: interpolated alpha block (identical layout to BC4)
        const alpha0 = src[blockOffset];
        const alpha1 = src[blockOffset + 1];
        const alphaBitsLo =
          src[blockOffset + 2] | (src[blockOffset + 3] << 8) | (src[blockOffset + 4] << 16);
        const alphaBitsHi =
          src[blockOffset + 5] | (src[blockOffset + 6] << 8) | (src[blockOffset + 7] << 16);
        decodeBc4Palette(alpha0, alpha1, alphaTable);

        // Bytes 8..15: RGB565 color block (always 4-color mode in BC3)
        const color0 = src[blockOffset + 8] | (src[blockOffset + 9] << 8);
        const color1 = src[blockOffset + 10] | (src[blockOffset + 11] << 8);
        const colorIndices =
          (src[blockOffset + 12] |
            (src[blockOffset + 13] << 8) |
            (src[blockOffset + 14] << 16) |
            (src[blockOffset + 15] << 24)) >>> 0;
        blockOffset += 16;

        decodeRgb565(color0, colorPalette, 0);
        decodeRgb565(color1, colorPalette, 4);

        const r0 = colorPalette[0];
        const g0 = colorPalette[1];
        const b0 = colorPalette[2];
        const r1 = colorPalette[4];
        const g1 = colorPalette[5];
        const b1 = colorPalette[6];

        colorPalette[8] = Math.floor((2 * r0 + r1 + 1) / 3);
        colorPalette[9] = Math.floor((2 * g0 + g1 + 1) / 3);
        colorPalette[10] = Math.floor((2 * b0 + b1 + 1) / 3);

        colorPalette[12] = Math.floor((r0 + 2 * r1 + 1) / 3);
        colorPalette[13] = Math.floor((g0 + 2 * g1 + 1) / 3);
        colorPalette[14] = Math.floor((b0 + 2 * b1 + 1) / 3);

        for (let py = 0; py < 4; py += 1) {
          const y = baseY + py;
          if (y >= height) {
            break;
          }
          const rowOutBase = y * width * 4;
          const pixelRowBase = py * 4;
          for (let px = 0; px < 4; px += 1) {
            const x = baseX + px;
            if (x >= width) {
              break;
            }
            const pixelIndex = pixelRowBase + px;
            const colorPalOff = ((colorIndices >>> (pixelIndex * 2)) & 0x3) * 4;
            const alphaIdx =
              pixelIndex < 8
                ? (alphaBitsLo >>> (pixelIndex * 3)) & 0x7
                : (alphaBitsHi >>> ((pixelIndex - 8) * 3)) & 0x7;

            const dstOff = rowOutBase + x * 4;
            out[dstOff] = colorPalette[colorPalOff];
            out[dstOff + 1] = colorPalette[colorPalOff + 1];
            out[dstOff + 2] = colorPalette[colorPalOff + 2];
            out[dstOff + 3] = alphaTable[alphaIdx];
          }
        }
      }
    }

    return out;
  }

  /**
   * CPU fallback BC4 UNORM single-channel red block decoder (used for roughness,
   * ambient occlusion, and height maps) -> tightly packed R8 Uint8Array.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} width
   * @param {number} height
   * @returns {Uint8Array}
   */
  function decodeBc4ToR8(srcBytes, width, height) {
    const src = toUint8Array(srcBytes, 'decodeBc4ToR8');
    assertPositiveInteger(width, 'width');
    assertPositiveInteger(height, 'height');

    const blocksX = Math.ceil(width / 4);
    const blocksY = Math.ceil(height / 4);
    const requiredBytes = blocksX * blocksY * 8;
    if (src.byteLength < requiredBytes) {
      throw new RangeError(
        `BC4 input too small (${src.byteLength} bytes; need ${requiredBytes} bytes for ${width}x${height})`
      );
    }

    const out = new Uint8Array(width * height);
    const redTable = new Uint8Array(8);

    let blockOffset = 0;
    for (let by = 0; by < blocksY; by += 1) {
      const baseY = by * 4;
      for (let bx = 0; bx < blocksX; bx += 1) {
        const baseX = bx * 4;

        const red0 = src[blockOffset];
        const red1 = src[blockOffset + 1];
        const bitsLo =
          src[blockOffset + 2] | (src[blockOffset + 3] << 8) | (src[blockOffset + 4] << 16);
        const bitsHi =
          src[blockOffset + 5] | (src[blockOffset + 6] << 8) | (src[blockOffset + 7] << 16);
        blockOffset += 8;

        decodeBc4Palette(red0, red1, redTable);

        for (let py = 0; py < 4; py += 1) {
          const y = baseY + py;
          if (y >= height) {
            break;
          }
          const rowOutBase = y * width;
          const pixelRowBase = py * 4;
          for (let px = 0; px < 4; px += 1) {
            const x = baseX + px;
            if (x >= width) {
              break;
            }
            const pixelIndex = pixelRowBase + px;
            const redIdx =
              pixelIndex < 8
                ? (bitsLo >>> (pixelIndex * 3)) & 0x7
                : (bitsHi >>> ((pixelIndex - 8) * 3)) & 0x7;
            out[rowOutBase + x] = redTable[redIdx];
          }
        }
      }
    }

    return out;
  }

  const exportsObj = Object.freeze({
    DXGI_FORMAT,
    DXGI_FORMAT_NAMES,
    WEBGPU_BYTES_PER_ROW_ALIGNMENT,
    TEXTURE_COMPRESSION_BC_FEATURE,
    mapDxgiFormatToWebGpu,
    calcMipSize,
    calcMipLayout,
    padTextureRowsForWriteTexture,
    decodeBc1ToRgba8,
    decodeBc3ToRgba8,
    decodeBc4ToR8,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6TextureFormats = exportsObj;
  } else if (globalScope) {
    globalScope.PlayGta6TextureFormats = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
