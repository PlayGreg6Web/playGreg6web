// GPU packet ABI v1/v2 texture handling for WgpuCommandProcessor:
// - ABI v2 CreateTexture2 (opcode 35, 40-byte payload, docs/gpu_commands.md "CreateTexture2")
//   with safe mipLevels and sampleCount clamping alongside ABI v1 CreateTexture (opcode 12),
//   automatically attaching compatible sRGB/linear sibling `viewFormats` (`rgba8unorm`,
//   `bgra8unorm`, `bc1..bc7`) and promoting `1d` LUT textures to WebGPU `2d` (`width x 1 x 1`)
//   when `promote1DTo2D` is enabled (default `true`, W341 / #1460) for `texture_2d` compatibility.
// - Zero-copy direct `queue.writeTexture` fast path when source row pitch is already 256-byte
//   aligned (e.g., 4K Bink2 `3840x2160` `r8unorm` Y plane).
// - WebGPU 256-byte bytesPerRow alignment & row-pitch repacking helpers (`alignBytesPerRow256`,
//   `repackRowsTo256`) backed by a bounded persistent per-handler scratch `Uint8Array` (e.g.,
//   reused across 4K Bink2 `1920x1080` `r8unorm` Cr/Cb planes where `1920 -> 2048`).
// - Zero-alloc WebGPU upload descriptor reuse: pre-allocated `GPUTexelCopyTextureInfo`,
//   `GPUTexelCopyBufferLayout` / `GPUTexelCopyBufferInfo`, `GPUExtent3DDict`, and staging layout
//   structs are reused across hot `writeTexture` / `copyBufferToTexture` / `handleWriteTexture`
//   calls with every field cleanly overwritten on each upload (W303).
// - Sub-region dirty-rect bounds clamping & validation against mip dimensions (respecting 4x4
//   physical mip sizes for BC1..BC7 tail mips) to prevent WebGPU out-of-bounds validation errors.
// - Direct SharedArrayBuffer writes when WebGPU accepts them; cached legacy TypeError
//   fallback and huge/chunked uploads retain bounded snapshots before queue writes.
// - Sanitized telemetry counters (`createTexture2Count`, `rowPitchRepackCount`, `sabCopyCount`,
//   `uploadCount`, `uploadsCount`, `uploadBytes`, `uploadBytesTotal`, `reusedDescriptorCount`,
//   `clampedSubRegionCount`, `zeroCopyUploads`, `scratchRepackUploads`,
//   `directZeroCopyUploads`, `repackedUploads`, `scratchBufferReuses`, `r8unormUploads`,
//   `bytesUploaded`, `srgbSiblingViewFormatsAdded`, `promoted1DTo2DCount`) exposed via
//   `getStats()`, `getUploadStats()`, the `uploadStats` getter, and cleanly reset via
//   `resetStats()` / `reset()` / `destroy()`.
//
// Install with installTexturePackets(processor) after the v2 handler (render_state_packets.js):
// it chains in front of processor.extension and forwards every other opcode to it.

(function (globalScope) {
  'use strict';

  const CREATE_TEXTURE = 12;
  const CREATE_TEXTURE_BYTES = 32;
  const WRITE_TEXTURE = 13;
  const WRITE_TEXTURE_HEADER_BYTES = 48;
  const CREATE_TEXTURE2 = 35;
  const CREATE_TEXTURE2_BYTES = 40;
  const OPCODE_COPY_TEXTURE = 62;
  const COPY_TEXTURE_PAYLOAD_BYTES = 56;
  const MAX_VIEW_FORMATS = 8;
  const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;
  const DEFAULT_MAX_SCRATCH_BYTES = 16 * 1024 * 1024; // 16 MiB cap for persistent row-repack / SAB scratch
  const DEFAULT_MAX_WRITE_TEXTURE_CHUNK_BYTES = 8 * 1024 * 1024; // 8 MiB strip cap (keeps 4K Bink2 Y 7.91 MiB single-call while chunking >=32 MiB 4K RGBA/array uploads)
  const DEFAULT_MAX_WRITE_BUFFER_CHUNK_BYTES = 4 * 1024 * 1024; // 4 MiB cap per queue.writeBuffer fallback
  const GPU_BUFFER_USAGE_COPY_SRC_DST = 0x0004 | 0x0008;

  // Compatible WebGPU sRGB <-> linear viewFormat sibling pairs.
  const SRGB_SIBLING_FORMATS = Object.freeze({
    rgba8unorm: 'rgba8unorm-srgb',
    'rgba8unorm-srgb': 'rgba8unorm',
    bgra8unorm: 'bgra8unorm-srgb',
    'bgra8unorm-srgb': 'bgra8unorm',
    'bc1-rgba-unorm': 'bc1-rgba-unorm-srgb',
    'bc1-rgba-unorm-srgb': 'bc1-rgba-unorm',
    'bc2-rgba-unorm': 'bc2-rgba-unorm-srgb',
    'bc2-rgba-unorm-srgb': 'bc2-rgba-unorm',
    'bc3-rgba-unorm': 'bc3-rgba-unorm-srgb',
    'bc3-rgba-unorm-srgb': 'bc3-rgba-unorm',
    'bc7-rgba-unorm': 'bc7-rgba-unorm-srgb',
    'bc7-rgba-unorm-srgb': 'bc7-rgba-unorm',
  });

  // v1 GpuTextureFormat enum names (runtime/abi/playgta6_abi.h).
  const V1_TEXTURE_FORMAT_NAMES = Object.freeze([
    'rgba8unorm',
    'bgra8unorm',
    'rgba16float',
    'depth24plus',
    'depth32float',
    'bc7-rgba-unorm',
  ]);

  // v2 texture format codes ("WebGPU enum codes" in docs/gpu_commands.md).
  const TEXTURE_FORMAT_NAMES = Object.freeze([
    'undefined', 'r8unorm', 'r8snorm', 'r8uint', 'r8sint', 'r16unorm', 'r16snorm', 'r16uint',
    'r16sint', 'r16float', 'rg8unorm', 'rg8snorm', 'rg8uint', 'rg8sint', 'r32uint', 'r32sint',
    'r32float', 'rg16unorm', 'rg16snorm', 'rg16uint', 'rg16sint', 'rg16float', 'rgba8unorm',
    'rgba8unorm-srgb', 'rgba8snorm', 'rgba8uint', 'rgba8sint', 'bgra8unorm', 'bgra8unorm-srgb',
    'rgb9e5ufloat', 'rgb10a2uint', 'rgb10a2unorm', 'rg11b10ufloat', 'rg32uint', 'rg32sint',
    'rg32float', 'rgba16unorm', 'rgba16snorm', 'rgba16uint', 'rgba16sint', 'rgba16float',
    'rgba32uint', 'rgba32sint', 'rgba32float', 'stencil8', 'depth16unorm', 'depth24plus',
    'depth24plus-stencil8', 'depth32float', 'depth32float-stencil8', 'bc1-rgba-unorm',
    'bc1-rgba-unorm-srgb', 'bc2-rgba-unorm', 'bc2-rgba-unorm-srgb', 'bc3-rgba-unorm',
    'bc3-rgba-unorm-srgb', 'bc4-r-unorm', 'bc4-r-snorm', 'bc5-rg-unorm', 'bc5-rg-snorm',
    'bc6h-rgb-ufloat', 'bc6h-rgb-float', 'bc7-rgba-unorm', 'bc7-rgba-unorm-srgb',
  ]);

  const TEXTURE_DIMENSION_NAMES = Object.freeze(['1d', '2d', '3d']);
  const BC_SLICED_3D_FEATURE = 'texture-compression-bc-sliced-3d';
  const BGRA8_STORAGE_FEATURE = 'bgra8unorm-storage';
  const RG11B10_RENDERABLE_FEATURE = 'rg11b10ufloat-renderable';
  const GPU_TEXTURE_USAGE_COPY_SRC = 0x01;
  const GPU_TEXTURE_USAGE_STORAGE_BINDING = 0x08;
  const GPU_TEXTURE_USAGE_RENDER_ATTACHMENT = 0x10;

  // Block geometry for all WebGPU texture formats in TEXTURE_FORMAT_NAMES.
  const WEBGPU_FORMAT_BLOCK_INFO = Object.freeze({
    r8unorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 1, isCompressed: false }),
    r8snorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 1, isCompressed: false }),
    r8uint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 1, isCompressed: false }),
    r8sint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 1, isCompressed: false }),
    r16unorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false }),
    r16snorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false }),
    r16uint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false }),
    r16sint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false }),
    r16float: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false }),
    rg8unorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false }),
    rg8snorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false }),
    rg8uint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false }),
    rg8sint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false }),
    r32uint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    r32sint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    r32float: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rg16unorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rg16snorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rg16uint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rg16sint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rg16float: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rgba8unorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    'rgba8unorm-srgb': Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rgba8snorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rgba8uint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rgba8sint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    bgra8unorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    'bgra8unorm-srgb': Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rgb9e5ufloat: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rgb10a2uint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rgb10a2unorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rg11b10ufloat: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    rg32uint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false }),
    rg32sint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false }),
    rg32float: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false }),
    rgba16unorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false }),
    rgba16snorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false }),
    rgba16uint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false }),
    rgba16sint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false }),
    rgba16float: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false }),
    rgba32uint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 16, isCompressed: false }),
    rgba32sint: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 16, isCompressed: false }),
    rgba32float: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 16, isCompressed: false }),
    stencil8: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 1, isCompressed: false }),
    depth16unorm: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 2, isCompressed: false }),
    depth24plus: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    'depth24plus-stencil8': Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    depth32float: Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 4, isCompressed: false }),
    'depth32float-stencil8': Object.freeze({ blockWidth: 1, blockHeight: 1, bytesPerBlock: 8, isCompressed: false }),
    'bc1-rgba-unorm': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 8, isCompressed: true }),
    'bc1-rgba-unorm-srgb': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 8, isCompressed: true }),
    'bc2-rgba-unorm': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true }),
    'bc2-rgba-unorm-srgb': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true }),
    'bc3-rgba-unorm': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true }),
    'bc3-rgba-unorm-srgb': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true }),
    'bc4-r-unorm': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 8, isCompressed: true }),
    'bc4-r-snorm': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 8, isCompressed: true }),
    'bc5-rg-unorm': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true }),
    'bc5-rg-snorm': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true }),
    'bc6h-rgb-ufloat': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true }),
    'bc6h-rgb-float': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true }),
    'bc7-rgba-unorm': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true }),
    'bc7-rgba-unorm-srgb': Object.freeze({ blockWidth: 4, blockHeight: 4, bytesPerBlock: 16, isCompressed: true }),
  });

  const HOOK_ORIG_CREATE_TEXTURE = '__playgta6OrigHandleCreateTexture';
  const HOOK_ORIG_WRITE_TEXTURE = '__playgta6OrigHandleWriteTexture';
  const HOOK_ORIG_DESTROY = '__playgta6OrigTexturePacketsDestroy';

  function isBcCompressedFormat(format) {
    return typeof format === 'string' && format.startsWith('bc');
  }

  /**
   * Look up WebGPU block geometry (`blockWidth`, `blockHeight`, `bytesPerBlock`,
   * `isCompressed`) for a WebGPU texture format name, or `null` when unknown.
   *
   * @param {string | object | null | undefined} format
   */
  function getTextureFormatBlockInfo(format) {
    if (typeof format === 'string') {
      const normalized = format.trim().toLowerCase();
      if (Object.prototype.hasOwnProperty.call(WEBGPU_FORMAT_BLOCK_INFO, normalized)) {
        return WEBGPU_FORMAT_BLOCK_INFO[normalized];
      }
      return null;
    }
    if (
      format &&
      typeof format === 'object' &&
      Number.isInteger(format.blockWidth) &&
      format.blockWidth >= 1 &&
      Number.isInteger(format.blockHeight) &&
      format.blockHeight >= 1 &&
      Number.isInteger(format.bytesPerBlock) &&
      format.bytesPerBlock >= 1
    ) {
      return format;
    }
    return null;
  }

  /**
   * Align a WebGPU texture copy size up to the format's block dimensions
   * (`4x4` for BC1..BC7) so small tail mips (`1x1`, `2x2`, `3x3`) and
   * non-multiple-of-4 sub-mips pass WebGPU copy-extent validation.
   *
   * @param {string | object | null | undefined} format
   * @param {{ width: number, height?: number, depthOrArrayLayers?: number } | number[]} copySize
   * @returns {{ width: number, height: number, depthOrArrayLayers: number }}
   */
  function alignCopySizeForFormat(format, copySize) {
    const norm = normalizeCopySize(copySize);
    const info = getTextureFormatBlockInfo(format);
    if (!info || (info.blockWidth <= 1 && info.blockHeight <= 1)) {
      return norm;
    }
    const alignedWidth = Math.ceil(norm.width / info.blockWidth) * info.blockWidth;
    const alignedHeight = Math.ceil(norm.height / info.blockHeight) * info.blockHeight;
    return {
      width: alignedWidth,
      height: alignedHeight,
      depthOrArrayLayers: norm.depthOrArrayLayers,
    };
  }

  /**
   * Compute exact WebGPU mip upload layout for BC1..BC7 compressed and
   * uncompressed formats, including 4x4 block-clamped copy extents for small
   * tail mips (`1x1`, `2x2`, `3x3` -> `4x4`) and 256-byte `bytesPerRow`.
   *
   * @param {string | object} format
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrLayers=1]
   * @param {number} [alignBytesPerRow=256]
   */
  function calcMipUploadLayout(
    format,
    width,
    height,
    depthOrLayers = 1,
    alignBytesPerRow = WEBGPU_BYTES_PER_ROW_ALIGNMENT
  ) {
    const info = getTextureFormatBlockInfo(format);
    if (!info) {
      throw new RangeError(`calcMipUploadLayout: unsupported WebGPU texture format ${JSON.stringify(format)}`);
    }
    if (!Number.isInteger(width) || width < 1) {
      throw new RangeError(`calcMipUploadLayout: width must be a positive integer, got ${width}`);
    }
    if (!Number.isInteger(height) || height < 1) {
      throw new RangeError(`calcMipUploadLayout: height must be a positive integer, got ${height}`);
    }
    if (!Number.isInteger(depthOrLayers) || depthOrLayers < 1) {
      throw new RangeError(
        `calcMipUploadLayout: depthOrLayers must be a positive integer, got ${depthOrLayers}`
      );
    }
    if (!Number.isInteger(alignBytesPerRow) || alignBytesPerRow < 1) {
      throw new RangeError(
        `calcMipUploadLayout: alignBytesPerRow must be a positive integer, got ${alignBytesPerRow}`
      );
    }

    const blocksX = Math.ceil(width / info.blockWidth);
    const blocksY = Math.ceil(height / info.blockHeight);
    const copyWidth = blocksX * info.blockWidth;
    const copyHeight = blocksY * info.blockHeight;
    const unalignedBytesPerRow = blocksX * info.bytesPerBlock;
    const bytesPerRow = Math.ceil(unalignedBytesPerRow / alignBytesPerRow) * alignBytesPerRow;
    const rowsPerImage = blocksY;
    const tightTotalBytes = unalignedBytesPerRow * rowsPerImage * depthOrLayers;
    const totalBytes = bytesPerRow * rowsPerImage * depthOrLayers;
    const isCompressed = Boolean(info.isCompressed || info.blockWidth > 1 || info.blockHeight > 1);

    return {
      width,
      height,
      copyWidth,
      copyHeight,
      depthOrLayers,
      blocksX,
      blocksY,
      blockWidth: info.blockWidth,
      blockHeight: info.blockHeight,
      bytesPerBlock: info.bytesPerBlock,
      isCompressed,
      isTailMip: isCompressed && (width < info.blockWidth || height < info.blockHeight),
      wasBlockClamped: copyWidth !== width || copyHeight !== height,
      unalignedBytesPerRow,
      bytesPerRow,
      rowsPerImage,
      tightTotalBytes,
      totalBytes,
      needsRowPadding: bytesPerRow !== unalignedBytesPerRow,
    };
  }

  /**
   * Return the WebGPU-compatible sRGB <-> linear sibling format for `format`
   * (`rgba8unorm`, `bgra8unorm`, `bc1/bc2/bc3/bc7` unorm <-> `-srgb`), or `null`
   * if `format` has no sRGB sibling.
   *
   * @param {string} format
   * @returns {string | null}
   */
  function getSrgbSiblingFormat(format) {
    if (typeof format !== 'string') {
      return null;
    }
    const normalized = format.trim().toLowerCase();
    if (Object.prototype.hasOwnProperty.call(SRGB_SIBLING_FORMATS, normalized)) {
      return SRGB_SIBLING_FORMATS[normalized];
    }
    return null;
  }

  /**
   * Build a deduplicated, WebGPU-valid `viewFormats` array for a texture of `format`:
   * - Filters any caller-supplied `requestedViewFormats` down to formats that are
   *   view-format-compatible with `format` (`format` itself or its sRGB/linear sibling).
   * - Automatically appends the compatible sRGB/linear sibling when one exists so
   *   sRGB/linear texture views never fail WebGPU validation.
   *
   * @param {string} format
   * @param {string[] | null | undefined} [requestedViewFormats]
   * @returns {{ viewFormats: string[] | undefined, siblingAdded: boolean }}
   */
  function resolveCompatibleViewFormats(format, requestedViewFormats, options = null) {
    if (typeof format !== 'string' || !format) {
      return { viewFormats: undefined, siblingAdded: false };
    }
    const baseFormat = format.trim().toLowerCase();
    const hasStorageBinding = Boolean(
      options &&
        (options.hasStorageBinding === true ||
          (typeof options.usage === 'number' &&
            (options.usage & GPU_TEXTURE_USAGE_STORAGE_BINDING) !== 0))
    );
    const sibling = getSrgbSiblingFormat(baseFormat);
    const allowSibling = sibling !== null && !(hasStorageBinding && sibling.endsWith('-srgb'));
    if (!Array.isArray(requestedViewFormats) || requestedViewFormats.length === 0) {
      return allowSibling
        ? { viewFormats: [sibling], siblingAdded: true }
        : { viewFormats: undefined, siblingAdded: false };
    }
    const out = [];
    const seen = new Set();

    for (let i = 0; i < requestedViewFormats.length; i += 1) {
      const raw = requestedViewFormats[i];
      if (typeof raw !== 'string') {
        continue;
      }
      const vf = raw.trim().toLowerCase();
      if (!vf || seen.has(vf)) {
        continue;
      }
      if (hasStorageBinding && vf.endsWith('-srgb')) {
        continue;
      }
      if (vf === baseFormat || (sibling !== null && vf === sibling)) {
        seen.add(vf);
        out.push(vf);
      }
    }

    let siblingAdded = false;
    if (allowSibling && !seen.has(sibling)) {
      seen.add(sibling);
      out.push(sibling);
      siblingAdded = true;
    }

    return {
      viewFormats: out.length > 0 ? out : undefined,
      siblingAdded,
    };
  }

  function deviceHasFeature(device, featureName) {
    const features = device && device.features;
    if (!features) {
      return false;
    }
    if (typeof features.has === 'function') {
      return Boolean(features.has(featureName));
    }
    if (Array.isArray(features)) {
      return features.includes(featureName);
    }
    if (typeof features[Symbol.iterator] === 'function') {
      for (const item of features) {
        if (item === featureName) {
          return true;
        }
      }
      return false;
    }
    if (typeof features === 'object') {
      return Boolean(features[featureName]);
    }
    return false;
  }

  function deviceSupportsBcSliced3d(device) {
    return deviceHasFeature(device, BC_SLICED_3D_FEATURE);
  }

  function isDepthOrStencilFormat(format) {
    return (
      typeof format === 'string' &&
      (format.startsWith('depth') || format === 'stencil8')
    );
  }

  function formatSupportsStorageBinding(format, device) {
    if (typeof format !== 'string' || !format) {
      return false;
    }
    if (
      format.endsWith('-srgb') ||
      isBcCompressedFormat(format) ||
      isDepthOrStencilFormat(format) ||
      format === 'rgb9e5ufloat' ||
      format === 'rg11b10ufloat' ||
      format === 'rgb10a2unorm' ||
      format === 'rgb10a2uint' ||
      format === 'r8unorm' ||
      format === 'r8snorm' ||
      format === 'r8uint' ||
      format === 'r8sint' ||
      format === 'rg8unorm' ||
      format === 'rg8snorm' ||
      format === 'rg8uint' ||
      format === 'rg8sint' ||
      format === 'r16unorm' ||
      format === 'r16snorm' ||
      format === 'r16uint' ||
      format === 'r16sint' ||
      format === 'r16float' ||
      format === 'rg16unorm' ||
      format === 'rg16snorm' ||
      format === 'rg16uint' ||
      format === 'rg16sint' ||
      format === 'rg16float' ||
      format === 'rgba16unorm' ||
      format === 'rgba16snorm'
    ) {
      return false;
    }
    if (format === 'bgra8unorm') {
      return device && device.features ? deviceHasFeature(device, BGRA8_STORAGE_FEATURE) : true;
    }
    return true;
  }

  function formatSupportsRenderAttachment(format, device) {
    if (typeof format !== 'string' || !format) {
      return false;
    }
    if (
      isBcCompressedFormat(format) ||
      format === 'rgb9e5ufloat' ||
      format === 'r8snorm' ||
      format === 'rg8snorm' ||
      format === 'rgba8snorm' ||
      format === 'r16snorm' ||
      format === 'rg16snorm' ||
      format === 'rgba16snorm'
    ) {
      return false;
    }
    if (format === 'rg11b10ufloat') {
      return device && device.features ? deviceHasFeature(device, RG11B10_RENDERABLE_FEATURE) : true;
    }
    return true;
  }

  function formatSupportsMultisample(format, device) {
    if (!formatSupportsRenderAttachment(format, device)) {
      return false;
    }
    if (
      format === 'r32uint' ||
      format === 'r32sint' ||
      format === 'rg32uint' ||
      format === 'rg32sint' ||
      format === 'rg32float' ||
      format === 'rgba32uint' ||
      format === 'rgba32sint' ||
      format === 'rgba32float'
    ) {
      return false;
    }
    return true;
  }

  /**
   * Compute WebGPU §22.5.1 minimum required byte length for a texel copy where the
   * final row of the final slice may be tightly packed (`lastRowBytes <= bytesPerRow`).
   */
  function calcMinTexelCopyBytes(bytesPerRow, rowsPerImage, rowsInSlice, depthOrLayers, lastRowBytes) {
    if (rowsInSlice <= 0 || depthOrLayers <= 0 || bytesPerRow <= 0) {
      return 0;
    }
    const finalRow =
      Number.isInteger(lastRowBytes) && lastRowBytes > 0 && lastRowBytes < bytesPerRow
        ? lastRowBytes
        : bytesPerRow;
    const rpi =
      Number.isInteger(rowsPerImage) && rowsPerImage >= rowsInSlice ? rowsPerImage : rowsInSlice;
    return (depthOrLayers - 1) * rpi * bytesPerRow + (rowsInSlice - 1) * bytesPerRow + finalRow;
  }

  function toNonNegativeInt(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      return 0;
    }
    return Math.floor(value);
  }

  function isSharedBuffer(buf) {
    if (!buf || typeof buf !== 'object') {
      return false;
    }
    // O(1) V8 fast path for standard non-shared ArrayBuffers (avoids Object.prototype.toString).
    if (buf instanceof ArrayBuffer) {
      return false;
    }
    if (typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer) {
      return true;
    }
    return Object.prototype.toString.call(buf) === '[object SharedArrayBuffer]';
  }

  /**
   * Returns `true` if `input` is a `SharedArrayBuffer` or an `ArrayBufferView`
   * backed by a `SharedArrayBuffer` (cross-realm safe).
   *
   * @param {unknown} input
   * @returns {boolean}
   */
  function isSharedArrayBufferView(input) {
    if (!input || typeof input !== 'object') {
      return false;
    }
    if (isSharedBuffer(input)) {
      return true;
    }
    if (ArrayBuffer.isView(input)) {
      return isSharedBuffer(input.buffer);
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
    if (input instanceof ArrayBuffer || isSharedBuffer(input)) {
      return new Uint8Array(input);
    }
    if (
      input &&
      typeof input === 'object' &&
      typeof input.byteLength === 'number' &&
      Object.prototype.toString.call(input) === '[object ArrayBuffer]'
    ) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      `${label} expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`
    );
  }

  /**
   * Bounded reusable non-shared `Uint8Array` scratch buffer pool for row-pitch
   * repacking, SAB copying, and `#973` `A8_UNORM -> rgba8unorm` expansion.
   *
   * - Requests exceeding `maxBytes` allocate a standalone one-off `Uint8Array`
   *   so huge one-off uploads never pin giant RAM.
   * - Callers retaining slices across frames can pass `{ copy: true }` or call
   *   `snapshot(view)` to avoid aliasing with subsequent scratch writes.
   */
  class TextureScratchPool {
    constructor(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const rawMaxBytes =
        opts.maxBytes !== undefined ? opts.maxBytes : opts.maxScratchBytes;
      this.maxBytes =
        Number.isInteger(rawMaxBytes) && rawMaxBytes >= 0
          ? rawMaxBytes
          : DEFAULT_MAX_SCRATCH_BYTES;
      const rawMaxSlots = opts.maxSlots;
      this.maxSlots =
        Number.isInteger(rawMaxSlots) && rawMaxSlots >= 1 ? rawMaxSlots : 1;
      this._slots = new Array(this.maxSlots).fill(null);
      this._allocations = 0;
      this._reuses = 0;
      this._oversizedFallbacks = 0;
    }

    get currentBuffer() {
      return this._slots[0] || null;
    }

    get retainedBytes() {
      let total = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        if (this._slots[i]) {
          total += this._slots[i].byteLength;
        }
      }
      return total;
    }

    acquire(requiredBytes, options = {}) {
      if (!Number.isInteger(requiredBytes) || requiredBytes < 0) {
        throw new RangeError(
          `TextureScratchPool.acquire: requiredBytes must be a non-negative integer, got ${requiredBytes}`
        );
      }
      const opts = options && typeof options === 'object' ? options : {};
      if (requiredBytes === 0) {
        const empty = new Uint8Array(0);
        return {
          buffer: empty,
          view: empty,
          reused: false,
          pooled: false,
          oversized: false,
          copied: Boolean(opts.copy),
        };
      }
      if (opts.copy === true) {
        const standalone = new Uint8Array(requiredBytes);
        return {
          buffer: standalone,
          view: standalone,
          reused: false,
          pooled: false,
          oversized: false,
          copied: true,
        };
      }

      const slotIndex =
        Number.isInteger(opts.slot) && opts.slot >= 0 && opts.slot < this.maxSlots
          ? opts.slot
          : 0;
      const existing = this._slots[slotIndex];
      if (existing && existing.byteLength >= requiredBytes) {
        this._reuses += 1;
        return {
          buffer: existing,
          view: existing.subarray(0, requiredBytes),
          reused: true,
          pooled: true,
          oversized: false,
          copied: false,
        };
      }

      const otherSlotsBytes = this.retainedBytes - (existing ? existing.byteLength : 0);
      const maxForSlot = this.maxBytes - otherSlotsBytes;
      if (requiredBytes > this.maxBytes || requiredBytes > maxForSlot) {
        this._oversizedFallbacks += 1;
        const standalone = new Uint8Array(requiredBytes);
        return {
          buffer: standalone,
          view: standalone,
          reused: false,
          pooled: false,
          oversized: true,
          copied: false,
        };
      }

      // First allocation sizes tightly to requiredBytes; subsequent growths expand
      // geometrically (2x clamped to maxForSlot) to amortize ramping upload sizes.
      const allocBytes = existing
        ? Math.min(maxForSlot, Math.max(requiredBytes, existing.byteLength * 2))
        : requiredBytes;
      const allocated = new Uint8Array(allocBytes);
      this._slots[slotIndex] = allocated;
      this._allocations += 1;
      return {
        buffer: allocated,
        view: allocBytes === requiredBytes ? allocated : allocated.subarray(0, requiredBytes),
        reused: false,
        pooled: true,
        oversized: false,
        copied: false,
      };
    }

    /**
     * Zero-allocation fast path for slot 0 acquisition used by hot TexturePackets
     * uploads (writes result fields directly into `out`).
     */
    acquireSlot0Into(out, requiredBytes) {
      if (requiredBytes <= 0 || requiredBytes > this.maxBytes) {
        if (requiredBytes > this.maxBytes) {
          this._oversizedFallbacks += 1;
        }
        out.buffer = null;
        out.reused = false;
        out.pooled = false;
        out.oversized = requiredBytes > this.maxBytes;
        return out;
      }
      const existing = this._slots[0];
      if (existing && existing.byteLength >= requiredBytes) {
        this._reuses += 1;
        out.buffer = existing;
        out.reused = true;
        out.pooled = true;
        out.oversized = false;
        return out;
      }
      const otherSlotsBytes =
        this.maxSlots === 1 ? 0 : this.retainedBytes - (existing ? existing.byteLength : 0);
      const maxForSlot = this.maxBytes - otherSlotsBytes;
      if (requiredBytes > maxForSlot) {
        this._oversizedFallbacks += 1;
        out.buffer = null;
        out.reused = false;
        out.pooled = false;
        out.oversized = true;
        return out;
      }
      const allocBytes = existing
        ? Math.min(maxForSlot, Math.max(requiredBytes, existing.byteLength * 2))
        : requiredBytes;
      const allocated = new Uint8Array(allocBytes);
      this._slots[0] = allocated;
      this._allocations += 1;
      out.buffer = allocated;
      out.reused = false;
      out.pooled = true;
      out.oversized = false;
      return out;
    }

    snapshot(slice) {
      const src = toUint8Array(slice, 'TextureScratchPool.snapshot');
      const copy = new Uint8Array(src.byteLength);
      copy.set(src, 0);
      return copy;
    }

    getStats() {
      let activeSlots = 0;
      let retainedBytes = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        if (this._slots[i]) {
          activeSlots += 1;
          retainedBytes += this._slots[i].byteLength;
        }
      }
      return {
        allocations: this._allocations,
        reuses: this._reuses,
        oversizedFallbacks: this._oversizedFallbacks,
        retainedBytes,
        maxBytes: this.maxBytes,
        maxSlots: this.maxSlots,
        activeSlots,
      };
    }

    trim(maxRetainedBytes = 0) {
      const cap =
        Number.isInteger(maxRetainedBytes) && maxRetainedBytes >= 0 ? maxRetainedBytes : 0;
      let trimmedSlots = 0;
      let trimmedBytes = 0;
      for (let i = 0; i < this._slots.length; i += 1) {
        const slot = this._slots[i];
        if (slot && slot.byteLength > cap) {
          trimmedBytes += slot.byteLength;
          trimmedSlots += 1;
          this._slots[i] = null;
        }
      }
      return { trimmedSlots, trimmedBytes };
    }

    reset() {
      this._slots.fill(null);
      this._allocations = 0;
      this._reuses = 0;
      this._oversizedFallbacks = 0;
    }

    destroy() {
      this.reset();
    }
  }

  function resolveScratchDestination(requiredBytes, scratchOrOptions) {
    if (requiredBytes === 0) {
      return {
        view: new Uint8Array(0),
        usingScratch: false,
        reusedScratch: false,
        pooledScratch: false,
        oversized: false,
        forceCopy: false,
      };
    }
    if (!scratchOrOptions) {
      return {
        view: new Uint8Array(requiredBytes),
        usingScratch: false,
        reusedScratch: false,
        pooledScratch: false,
        oversized: false,
        forceCopy: false,
      };
    }
    if (scratchOrOptions instanceof Uint8Array) {
      if (!isSharedBuffer(scratchOrOptions.buffer) && scratchOrOptions.byteLength >= requiredBytes) {
        return {
          view: scratchOrOptions.subarray(0, requiredBytes),
          usingScratch: true,
          reusedScratch: true,
          pooledScratch: false,
          oversized: false,
          forceCopy: false,
        };
      }
      return {
        view: new Uint8Array(requiredBytes),
        usingScratch: false,
        reusedScratch: false,
        pooledScratch: false,
        oversized: false,
        forceCopy: false,
      };
    }
    if (typeof scratchOrOptions === 'object' && typeof scratchOrOptions.acquire === 'function') {
      const acquired = scratchOrOptions.acquire(requiredBytes);
      return {
        view: acquired.view,
        usingScratch: acquired.pooled,
        reusedScratch: acquired.reused,
        pooledScratch: acquired.pooled,
        oversized: acquired.oversized,
        forceCopy: false,
      };
    }
    if (typeof scratchOrOptions === 'object') {
      const forceCopy = Boolean(scratchOrOptions.copyResult === true || scratchOrOptions.copy === true);
      if (forceCopy) {
        return {
          view: new Uint8Array(requiredBytes),
          usingScratch: false,
          reusedScratch: false,
          pooledScratch: false,
          oversized: false,
          forceCopy: true,
        };
      }
      if (
        scratchOrOptions.scratchPool &&
        typeof scratchOrOptions.scratchPool.acquire === 'function'
      ) {
        const acquired = scratchOrOptions.scratchPool.acquire(requiredBytes, {
          slot: scratchOrOptions.slot,
        });
        return {
          view: acquired.view,
          usingScratch: acquired.pooled,
          reusedScratch: acquired.reused,
          pooledScratch: acquired.pooled,
          oversized: acquired.oversized,
          forceCopy: false,
        };
      }
      const buf = scratchOrOptions.scratchBuffer;
      if (buf instanceof Uint8Array && !isSharedBuffer(buf.buffer) && buf.byteLength >= requiredBytes) {
        return {
          view: buf.subarray(0, requiredBytes),
          usingScratch: true,
          reusedScratch: true,
          pooledScratch: false,
          oversized: false,
          forceCopy: false,
        };
      }
    }
    return {
      view: new Uint8Array(requiredBytes),
      usingScratch: false,
      reusedScratch: false,
      pooledScratch: false,
      oversized: false,
      forceCopy: false,
    };
  }

  /**
   * Safely converts `input` to a `Uint8Array` backed by a non-shared `ArrayBuffer`.
   * If `input` is backed by a `SharedArrayBuffer`, copies into an optional reusable
   * non-shared `scratchBuffer` / `TextureScratchPool` (when large enough) or allocates
   * a standalone copy so WebGPU `queue.writeTexture` / `queue.writeBuffer` /
   * `copyBufferToTexture` calls never reject shared buffers or race with concurrent
   * guest writes.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} input
   * @param {Uint8Array | TextureScratchPool | { scratchBuffer?: Uint8Array, scratchPool?: TextureScratchPool, slot?: number, copyResult?: boolean } | null} [scratchOrOptions=null]
   * @returns {{ data: Uint8Array, copiedFromSab: boolean, reusedScratch: boolean }}
   */
  function copyFromSharedIfNeeded(input, scratchOrOptions = null) {
    const src = toUint8Array(input, 'copyFromSharedIfNeeded');
    if (isSharedBuffer(src.buffer)) {
      const dest = resolveScratchDestination(src.byteLength, scratchOrOptions);
      dest.view.set(src, 0);
      return { data: dest.view, copiedFromSab: true, reusedScratch: dest.reusedScratch };
    }
    if (
      scratchOrOptions &&
      typeof scratchOrOptions === 'object' &&
      !(scratchOrOptions instanceof Uint8Array) &&
      (scratchOrOptions.copyResult === true || scratchOrOptions.copy === true)
    ) {
      const copy = new Uint8Array(src.byteLength);
      copy.set(src, 0);
      return { data: copy, copiedFromSab: false, reusedScratch: false };
    }
    return { data: src, copiedFromSab: false, reusedScratch: false };
  }

  /**
   * Align a texture row stride (`bytesPerRow`) up to WebGPU's required 256-byte
   * `copyBufferToTexture` boundary without 32-bit bitwise truncation.
   *
   * @param {number} bytesPerRow
   * @returns {number}
   */
  function alignBytesPerRow256(bytesPerRow) {
    if (!Number.isInteger(bytesPerRow) || bytesPerRow < 0) {
      throw new RangeError(
        `alignBytesPerRow256 expects a non-negative integer, got ${bytesPerRow}`
      );
    }
    if (bytesPerRow === 0) {
      return 0;
    }
    const remainder = bytesPerRow % WEBGPU_BYTES_PER_ROW_ALIGNMENT;
    return remainder === 0
      ? bytesPerRow
      : bytesPerRow + (WEBGPU_BYTES_PER_ROW_ALIGNMENT - remainder);
  }

  /**
   * Repack texture rows into a 256-byte-aligned `Uint8Array` suitable for WebGPU
   * `copyBufferToTexture`, zero-padding each row's trailing alignment bytes and
   * safely copying out of `SharedArrayBuffer`-backed views.
   *
   * When `bytesPerRow` is already a multiple of 256 and `srcBytes` is not backed
   * by a `SharedArrayBuffer`, returns a zero-copy view with `repacked: false`
   * (unless `{ copyResult: true }` is explicitly requested for safe retention).
   *
   * When an optional non-shared `scratchBuffer` (`Uint8Array`), `TextureScratchPool`,
   * or `{ scratchBuffer, scratchPool, slot, copyResult }` is provided and
   * copying/repacking is required, writes into the bounded scratch buffer instead
   * of allocating a new buffer and strictly zeroes each row's trailing padding bytes.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} bytesPerRow
   * @param {number} rowCount
   * @param {number} [depthOrLayers=1]
   * @param {Uint8Array | TextureScratchPool | object | null} [scratchOrOptions=null]
   */
  function copyAndPadRepackedRows(src, dst, bytesPerRow, alignedBytesPerRow, totalRows, zeroPad) {
    const srcLen = src.byteLength;
    if (totalRows === 1) {
      const copyLen = Math.min(bytesPerRow, srcLen);
      if (copyLen === srcLen && copyLen > 0) {
        dst.set(src, 0);
      } else if (copyLen <= 32) {
        for (let i = 0; i < copyLen; i += 1) {
          dst[i] = src[i];
        }
      } else {
        dst.set(src.subarray(0, copyLen), 0);
      }
      if ((zeroPad || copyLen < bytesPerRow) && copyLen < alignedBytesPerRow) {
        dst.fill(0, copyLen, alignedBytesPerRow);
      }
      return;
    }
    const lastRowIndex = totalRows - 1;
    if (bytesPerRow <= 32) {
      for (let row = 0; row < totalRows; row += 1) {
        const srcOffset = row * bytesPerRow;
        const dstOffset = row * alignedBytesPerRow;
        const copyLen =
          row === lastRowIndex ? Math.min(bytesPerRow, Math.max(0, srcLen - srcOffset)) : bytesPerRow;
        for (let i = 0; i < copyLen; i += 1) {
          dst[dstOffset + i] = src[srcOffset + i];
        }
        if ((zeroPad || copyLen < bytesPerRow) && copyLen < alignedBytesPerRow) {
          dst.fill(0, dstOffset + copyLen, dstOffset + alignedBytesPerRow);
        }
      }
      return;
    }
    for (let row = 0; row < totalRows; row += 1) {
      const srcOffset = row * bytesPerRow;
      const dstOffset = row * alignedBytesPerRow;
      const copyLen =
        row === lastRowIndex ? Math.min(bytesPerRow, Math.max(0, srcLen - srcOffset)) : bytesPerRow;
      if (copyLen > 0) {
        dst.set(src.subarray(srcOffset, srcOffset + copyLen), dstOffset);
      }
      if ((zeroPad || copyLen < bytesPerRow) && copyLen < alignedBytesPerRow) {
        dst.fill(0, dstOffset + copyLen, dstOffset + alignedBytesPerRow);
      }
    }
  }

  function repackRowsTo256(
    srcBytes,
    bytesPerRow,
    rowCount,
    depthOrLayers = 1,
    scratchOrOptions = null
  ) {
    const src = toUint8Array(srcBytes, 'repackRowsTo256');
    if (!Number.isInteger(bytesPerRow) || bytesPerRow < 1) {
      throw new RangeError(`repackRowsTo256: bytesPerRow must be a positive integer, got ${bytesPerRow}`);
    }
    if (!Number.isInteger(rowCount) || rowCount < 0) {
      throw new RangeError(`repackRowsTo256: rowCount must be a non-negative integer, got ${rowCount}`);
    }
    if (!Number.isInteger(depthOrLayers) || depthOrLayers < 1) {
      throw new RangeError(
        `repackRowsTo256: depthOrLayers must be a positive integer, got ${depthOrLayers}`
      );
    }

    const totalRows = rowCount * depthOrLayers;
    const alignedBytesPerRow = alignBytesPerRow256(bytesPerRow);
    const fromSab = isSharedBuffer(src.buffer);
    const isOptsObj =
      scratchOrOptions &&
      typeof scratchOrOptions === 'object' &&
      !(scratchOrOptions instanceof Uint8Array);
    const forceCopy = Boolean(
      isOptsObj && (scratchOrOptions.copyResult === true || scratchOrOptions.copy === true)
    );
    const optLastRowBytes =
      isOptsObj &&
      Number.isInteger(scratchOrOptions.lastRowBytes) &&
      scratchOrOptions.lastRowBytes > 0
        ? Math.min(bytesPerRow, scratchOrOptions.lastRowBytes)
        : bytesPerRow;

    if (totalRows === 0) {
      return {
        data: new Uint8Array(0),
        bytesPerRow: alignedBytesPerRow,
        originalBytesPerRow: bytesPerRow,
        rowCount: 0,
        repacked: false,
        copiedFromSab: false,
        reusedScratch: false,
      };
    }

    const fullSrcBytes = bytesPerRow * totalRows;
    const minRequiredSrcBytes = (totalRows - 1) * bytesPerRow + optLastRowBytes;
    if (src.byteLength < minRequiredSrcBytes) {
      throw new RangeError(
        `repackRowsTo256: srcBytes too small (${src.byteLength} bytes; need at least ${minRequiredSrcBytes} bytes for ${totalRows} rows of ${bytesPerRow} bytes)`
      );
    }

    const needsLastRowPad = src.byteLength < fullSrcBytes;
    if (alignedBytesPerRow === bytesPerRow && !needsLastRowPad) {
      const slice = src.byteLength === fullSrcBytes ? src : src.subarray(0, fullSrcBytes);
      if (fromSab || forceCopy) {
        const dest = resolveScratchDestination(fullSrcBytes, scratchOrOptions);
        dest.view.set(slice, 0);
        return {
          data: dest.view,
          bytesPerRow: alignedBytesPerRow,
          originalBytesPerRow: bytesPerRow,
          rowCount: totalRows,
          repacked: false,
          copiedFromSab: fromSab,
          reusedScratch: dest.reusedScratch,
        };
      }
      return {
        data: slice,
        bytesPerRow: alignedBytesPerRow,
        originalBytesPerRow: bytesPerRow,
        rowCount: totalRows,
        repacked: false,
        copiedFromSab: false,
        reusedScratch: false,
      };
    }

    const totalDstBytes = alignedBytesPerRow * totalRows;
    const dest = resolveScratchDestination(totalDstBytes, scratchOrOptions);
    const dst = dest.view;

    copyAndPadRepackedRows(
      src,
      dst,
      bytesPerRow,
      alignedBytesPerRow,
      totalRows,
      dest.usingScratch && dest.reusedScratch
    );

    return {
      data: dst,
      bytesPerRow: alignedBytesPerRow,
      originalBytesPerRow: bytesPerRow,
      rowCount: totalRows,
      repacked: true,
      copiedFromSab: fromSab,
      reusedScratch: dest.reusedScratch,
    };
  }

  /**
   * Expand `#973` `DXGI_FORMAT_A8_UNORM` (1 Bpp alpha-only font/mask atlas) into
   * `rgba8unorm` texels `(0, 0, 0, a)` in a single pass, with optional 256-byte
   * row-pitch padding (default `alignBytesPerRow = 256`) and bounded scratch
   * buffer reuse (`scratchBuffer` / `scratchPool`).
   *
   * By default (when no scratch buffer/pool is passed, or when `copyResult: true`),
   * returns an independent non-shared `Uint8Array` so callers retaining slices
   * across frames are never aliased.
   *
   * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} srcBytes
   * @param {number} width
   * @param {number} height
   * @param {{
   *   depthOrLayers?: number,
   *   srcBytesPerRow?: number,
   *   srcRowsPerImage?: number,
   *   alignBytesPerRow?: number,
   *   scratchBuffer?: Uint8Array | null,
   *   scratchPool?: TextureScratchPool | null,
   *   slot?: number,
   *   copyResult?: boolean,
   * }} [options]
   */
  function expandA8ToRgba8(srcBytes, width, height, options = {}) {
    const src = toUint8Array(srcBytes, 'expandA8ToRgba8');
    if (!Number.isInteger(width) || width < 1) {
      throw new RangeError(`expandA8ToRgba8: width must be a positive integer, got ${width}`);
    }
    if (!Number.isInteger(height) || height < 1) {
      throw new RangeError(`expandA8ToRgba8: height must be a positive integer, got ${height}`);
    }

    const opts = options && typeof options === 'object' ? options : {};
    const depthOrLayers = opts.depthOrLayers !== undefined ? opts.depthOrLayers : 1;
    const srcBytesPerRow = opts.srcBytesPerRow !== undefined ? opts.srcBytesPerRow : width;
    const srcRowsPerImage = opts.srcRowsPerImage !== undefined ? opts.srcRowsPerImage : height;
    const alignBytesPerRow =
      opts.alignBytesPerRow !== undefined
        ? opts.alignBytesPerRow
        : WEBGPU_BYTES_PER_ROW_ALIGNMENT;

    if (!Number.isInteger(depthOrLayers) || depthOrLayers < 1) {
      throw new RangeError(
        `expandA8ToRgba8: depthOrLayers must be a positive integer, got ${depthOrLayers}`
      );
    }
    if (!Number.isInteger(srcBytesPerRow) || srcBytesPerRow < width) {
      throw new RangeError(
        `expandA8ToRgba8: srcBytesPerRow (${srcBytesPerRow}) must be an integer >= width (${width})`
      );
    }
    if (!Number.isInteger(srcRowsPerImage) || srcRowsPerImage < height) {
      throw new RangeError(
        `expandA8ToRgba8: srcRowsPerImage (${srcRowsPerImage}) must be an integer >= height (${height})`
      );
    }
    if (!Number.isInteger(alignBytesPerRow) || alignBytesPerRow < 1) {
      throw new RangeError(
        `expandA8ToRgba8: alignBytesPerRow must be a positive integer, got ${alignBytesPerRow}`
      );
    }

    const srcSliceBytes = srcBytesPerRow * srcRowsPerImage;
    const minRequiredSrcBytes =
      (depthOrLayers - 1) * srcSliceBytes + (height - 1) * srcBytesPerRow + width;
    if (src.byteLength < minRequiredSrcBytes) {
      throw new RangeError(
        `expandA8ToRgba8: srcBytes too small (${src.byteLength} bytes; need at least ${minRequiredSrcBytes} bytes for ${width}x${height}x${depthOrLayers})`
      );
    }

    const unalignedBytesPerRow = width * 4;
    const bytesPerRow = Math.ceil(unalignedBytesPerRow / alignBytesPerRow) * alignBytesPerRow;
    const rowsPerImage = height;
    const totalBytes = bytesPerRow * rowsPerImage * depthOrLayers;
    const dest = resolveScratchDestination(totalBytes, opts);
    const dst = dest.view;
    const hasTrailingPad = bytesPerRow > unalignedBytesPerRow;

    for (let z = 0; z < depthOrLayers; z += 1) {
      const srcSliceBase = z * srcSliceBytes;
      const dstSliceBase = z * rowsPerImage * bytesPerRow;
      for (let y = 0; y < height; y += 1) {
        const srcRow = srcSliceBase + y * srcBytesPerRow;
        const dstRow = dstSliceBase + y * bytesPerRow;
        for (let x = 0; x < width; x += 1) {
          const outOff = dstRow + (x << 2);
          dst[outOff] = 0;
          dst[outOff + 1] = 0;
          dst[outOff + 2] = 0;
          dst[outOff + 3] = src[srcRow + x];
        }
        if (dest.usingScratch && dest.reusedScratch && hasTrailingPad) {
          dst.fill(0, dstRow + unalignedBytesPerRow, dstRow + bytesPerRow);
        }
      }
    }

    return {
      data: dst,
      width,
      height,
      depthOrLayers,
      unalignedBytesPerRow,
      bytesPerRow,
      rowsPerImage,
      totalBytes,
      repacked: bytesPerRow !== unalignedBytesPerRow,
      copiedFromSab: isSharedBuffer(src.buffer),
      reusedScratch: dest.reusedScratch,
      pooledScratch: dest.pooledScratch,
      oversized: dest.oversized,
    };
  }

  /**
   * Compute the maximum valid WebGPU mip level count for texture dimensions.
   * Per WebGPU spec:
   * - `'1d'` textures only allow 1 mip level.
   * - `'2d'` textures allow `floor(log2(max(width, height))) + 1`.
   * - `'3d'` textures allow `floor(log2(max(width, height, depth))) + 1`.
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {string} [dimension='2d']
   * @returns {number}
   */
  function computeMaxMipLevels(width, height, depthOrArrayLayers = 1, dimension = '2d') {
    if (dimension === '1d') {
      return 1;
    }
    const w = Math.max(1, Math.floor(Number(width) || 1));
    const h = Math.max(1, Math.floor(Number(height) || 1));
    const d = Math.max(1, Math.floor(Number(depthOrArrayLayers) || 1));
    const maxDim = dimension === '3d' ? Math.max(w, h, d) : Math.max(w, h);
    return Math.floor(Math.log2(maxDim)) + 1;
  }

  /**
   * Clamp a requested `mipLevelCount` safely into `[1, maxMipLevels]`.
   *
   * @param {number} requestedMipLevels
   * @param {number} width
   * @param {number} height
   * @param {number} [depthOrArrayLayers=1]
   * @param {string} [dimension='2d']
   * @returns {number}
   */
  function clampMipLevels(
    requestedMipLevels,
    width,
    height,
    depthOrArrayLayers = 1,
    dimension = '2d'
  ) {
    const maxMips = computeMaxMipLevels(width, height, depthOrArrayLayers, dimension);
    const n = Number(requestedMipLevels);
    if (!Number.isFinite(n) || n < 1) {
      return 1;
    }
    return Math.min(maxMips, Math.max(1, Math.floor(n)));
  }

  /**
   * Clamp a requested `sampleCount` safely to WebGPU's legal `{1, 4}` set.
   * Forces `1` whenever multisampling is disallowed by WebGPU texture geometry
   * (`dimension !== '2d'`, `mipLevelCount > 1`, or `depthOrArrayLayers > 1`).
   *
   * @param {number} requestedSampleCount
   * @param {{ mipLevelCount?: number, dimension?: string, depthOrArrayLayers?: number }} [context]
   * @returns {number}
   */
  function clampSampleCount(requestedSampleCount, context = {}) {
    const n = Number(requestedSampleCount);
    if (!Number.isFinite(n) || n <= 1) {
      return 1;
    }
    const mipLevelCount = context.mipLevelCount !== undefined ? context.mipLevelCount : 1;
    const dimension = context.dimension || '2d';
    const depthOrArrayLayers =
      context.depthOrArrayLayers !== undefined ? context.depthOrArrayLayers : 1;

    const geometryAllowsMsaa =
      dimension === '2d' && mipLevelCount <= 1 && depthOrArrayLayers === 1;

    if (n >= 4 && geometryAllowsMsaa) {
      return 4;
    }
    return 1;
  }

  function attachNonEnumerableDecodePool(obj, pool) {
    Object.defineProperty(obj, '_pool', {
      value: pool,
      enumerable: false,
      configurable: false,
    });
    return pool;
  }

  function createTextureDecodeScratch(isV2) {
    const size = { width: 1, height: 1, depthOrArrayLayers: 1 };
    const descriptor = isV2
      ? {
          size,
          mipLevelCount: 1,
          sampleCount: 1,
          dimension: '2d',
          format: 'rgba8unorm',
          usage: 0,
        }
      : {
          size,
          mipLevelCount: 1,
          format: 'rgba8unorm',
          usage: 0,
          dimension: '2d',
        };
    const out = { id: 0, descriptor };
    attachNonEnumerableDecodePool(out, { descriptor, size, viewFormats: [] });
    return out;
  }

  /**
   * Decodes a v1 `CreateTexture` (opcode 12, 32-byte `GpuCreateTexturePayload`)
   * into `{ id, descriptor }` with safely clamped dimensions and `mipLevelCount`,
   * or `{ id, error }` when truncated. Supports an optional pre-allocated `out`
   * scratch object (4th arg, or 3rd arg when `payloadBytes` is omitted).
   *
   * @param {DataView} view
   * @param {number} [offset=0]
   * @param {number} [payloadBytes]
   * @param {object | null} [out=null]
   */
  function decodeCreateTexture(view, offset = 0, payloadBytes, out = null) {
    let targetOut = out;
    let explicitPayloadBytes = payloadBytes;
    if (typeof payloadBytes === 'object' && payloadBytes !== null && out === null) {
      targetOut = payloadBytes;
      explicitPayloadBytes = undefined;
    }
    const available = explicitPayloadBytes !== undefined ? explicitPayloadBytes : view.byteLength - offset;
    if (offset < 0 || available < CREATE_TEXTURE_BYTES || offset + CREATE_TEXTURE_BYTES > view.byteLength) {
      const error = `CreateTexture: payload ${available} < ${CREATE_TEXTURE_BYTES} bytes`;
      if (targetOut) {
        targetOut.id = 0;
        targetOut.error = error;
        if ('descriptor' in targetOut) delete targetOut.descriptor;
        return targetOut;
      }
      return { id: 0, error };
    }

    const id = view.getUint32(offset, true);
    const rawWidth = view.getUint32(offset + 4, true);
    const rawHeight = view.getUint32(offset + 8, true);
    const rawDepth = view.getUint32(offset + 12, true);
    const rawMipLevels = view.getUint32(offset + 16, true);
    const formatCode = view.getUint32(offset + 20, true);
    const usage = view.getUint32(offset + 24, true);
    const dimensionCode = view.getUint32(offset + 28, true);

    const format = V1_TEXTURE_FORMAT_NAMES[formatCode] || 'rgba8unorm';
    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode] || '2d';
    const width = Math.max(1, rawWidth);
    const height = dimension === '1d' ? 1 : Math.max(1, rawHeight);
    const depthOrArrayLayers = dimension === '1d' ? 1 : Math.max(1, rawDepth);
    const mipLevelCount = clampMipLevels(
      rawMipLevels,
      width,
      height,
      depthOrArrayLayers,
      dimension
    );
    const hasStorageBinding = (usage & GPU_TEXTURE_USAGE_STORAGE_BINDING) !== 0;
    const sibling = getSrgbSiblingFormat(format);
    const attachSibling = sibling !== null && !(hasStorageBinding && sibling.endsWith('-srgb'));

    if (targetOut) {
      let pool = targetOut._pool;
      if (!pool) {
        const sizeObj =
          targetOut.descriptor && targetOut.descriptor.size && typeof targetOut.descriptor.size === 'object'
            ? targetOut.descriptor.size
            : { width: 1, height: 1, depthOrArrayLayers: 1 };
        const descObj =
          targetOut.descriptor && typeof targetOut.descriptor === 'object'
            ? targetOut.descriptor
            : { size: sizeObj, mipLevelCount: 1, format: 'rgba8unorm', usage: 0, dimension: '2d' };
        descObj.size = sizeObj;
        pool = attachNonEnumerableDecodePool(targetOut, {
          descriptor: descObj,
          size: sizeObj,
          viewFormats: Array.isArray(descObj.viewFormats) ? descObj.viewFormats : [],
        });
      }
      const size = pool.size;
      size.width = width;
      size.height = height;
      size.depthOrArrayLayers = depthOrArrayLayers;
      const descriptor = pool.descriptor;
      descriptor.size = size;
      descriptor.mipLevelCount = mipLevelCount;
      descriptor.format = format;
      descriptor.usage = usage;
      descriptor.dimension = dimension;
      if ('sampleCount' in descriptor) delete descriptor.sampleCount;
      const viewFormats = pool.viewFormats;
      if (attachSibling) {
        viewFormats.length = 1;
        viewFormats[0] = sibling;
        descriptor.viewFormats = viewFormats;
      } else {
        viewFormats.length = 0;
        if ('viewFormats' in descriptor) delete descriptor.viewFormats;
      }
      targetOut.id = id;
      targetOut.descriptor = descriptor;
      if ('error' in targetOut) delete targetOut.error;
      return targetOut;
    }

    const descriptor = {
      size: { width, height, depthOrArrayLayers },
      mipLevelCount,
      format,
      usage,
      dimension,
    };
    if (attachSibling) {
      descriptor.viewFormats = [sibling];
    }

    return {
      id,
      descriptor,
    };
  }

  /**
   * Decodes an ABI v2 `CreateTexture2` payload (opcode 35, 40-byte
   * `GpuCreateTexture2Payload`) into `{ id, descriptor }` (`GPUTextureDescriptor`)
   * with safely clamped `mipLevelCount` and `sampleCount`, or `{ id, error }` when
   * the payload is truncated or contains invalid codes. Supports an optional
   * pre-allocated `out` scratch object (4th arg, or 3rd arg when `payloadBytes` is omitted).
   *
   * @param {DataView} view
   * @param {number} [offset=0]
   * @param {number} [payloadBytes]
   * @param {object | null} [out=null]
   */
  function decodeCreateTexture2(view, offset = 0, payloadBytes, out = null) {
    let targetOut = out;
    let explicitPayloadBytes = payloadBytes;
    if (typeof payloadBytes === 'object' && payloadBytes !== null && out === null) {
      targetOut = payloadBytes;
      explicitPayloadBytes = undefined;
    }
    const available = explicitPayloadBytes !== undefined ? explicitPayloadBytes : view.byteLength - offset;
    if (
      offset < 0 ||
      available < CREATE_TEXTURE2_BYTES ||
      offset + CREATE_TEXTURE2_BYTES > view.byteLength
    ) {
      const error = `CreateTexture2: payload ${available} < ${CREATE_TEXTURE2_BYTES} bytes`;
      if (targetOut) {
        targetOut.id = 0;
        targetOut.error = error;
        if ('descriptor' in targetOut) delete targetOut.descriptor;
        return targetOut;
      }
      return { id: 0, error };
    }

    const id = view.getUint32(offset, true);
    const rawWidth = view.getUint32(offset + 4, true);
    const rawHeight = view.getUint32(offset + 8, true);
    const rawDepth = view.getUint32(offset + 12, true);
    const rawMipLevels = view.getUint32(offset + 16, true);
    const rawSampleCount = view.getUint32(offset + 20, true);
    const usage = view.getUint32(offset + 24, true);
    const dimensionCode = view.getUint8(offset + 28);
    const formatCode = view.getUint8(offset + 29);
    const viewFormatCount = view.getUint8(offset + 30);

    const format = TEXTURE_FORMAT_NAMES[formatCode];
    if (formatCode === 0 || format === undefined) {
      const error = `CreateTexture2 ${id}: invalid texture format code ${formatCode}`;
      if (targetOut) {
        targetOut.id = id;
        targetOut.error = error;
        if ('descriptor' in targetOut) delete targetOut.descriptor;
        return targetOut;
      }
      return { id, error };
    }
    const dimension = TEXTURE_DIMENSION_NAMES[dimensionCode];
    if (dimension === undefined) {
      const error = `CreateTexture2 ${id}: invalid dimension ${dimensionCode}`;
      if (targetOut) {
        targetOut.id = id;
        targetOut.error = error;
        if ('descriptor' in targetOut) delete targetOut.descriptor;
        return targetOut;
      }
      return { id, error };
    }
    if (viewFormatCount > MAX_VIEW_FORMATS) {
      const error = `CreateTexture2 ${id}: view_format_count ${viewFormatCount} > ${MAX_VIEW_FORMATS}`;
      if (targetOut) {
        targetOut.id = id;
        targetOut.error = error;
        if ('descriptor' in targetOut) delete targetOut.descriptor;
        return targetOut;
      }
      return { id, error };
    }

    let pool = targetOut ? targetOut._pool : null;
    if (targetOut && !pool) {
      const sizeObj =
        targetOut.descriptor && targetOut.descriptor.size && typeof targetOut.descriptor.size === 'object'
          ? targetOut.descriptor.size
          : { width: 1, height: 1, depthOrArrayLayers: 1 };
      const descObj =
        targetOut.descriptor && typeof targetOut.descriptor === 'object'
          ? targetOut.descriptor
          : {
              size: sizeObj,
              mipLevelCount: 1,
              sampleCount: 1,
              dimension: '2d',
              format: 'rgba8unorm',
              usage: 0,
            };
      descObj.size = sizeObj;
      pool = attachNonEnumerableDecodePool(targetOut, {
        descriptor: descObj,
        size: sizeObj,
        viewFormats: Array.isArray(descObj.viewFormats) ? descObj.viewFormats : [],
      });
    }

    const viewFormats = pool ? pool.viewFormats : [];
    viewFormats.length = 0;
    for (let i = 0; i < viewFormatCount; i += 1) {
      const code = view.getUint8(offset + 32 + i);
      const name = TEXTURE_FORMAT_NAMES[code];
      if (code === 0 || name === undefined) {
        viewFormats.length = 0;
        const error = `CreateTexture2 ${id}: invalid view format code ${code}`;
        if (targetOut) {
          targetOut.id = id;
          targetOut.error = error;
          if ('descriptor' in targetOut) delete targetOut.descriptor;
          return targetOut;
        }
        return { id, error };
      }
      viewFormats.push(name);
    }

    const width = Math.max(1, rawWidth);
    const height = dimension === '1d' ? 1 : Math.max(1, rawHeight);
    const depthOrArrayLayers = dimension === '1d' ? 1 : Math.max(1, rawDepth);
    const mipLevelCount = clampMipLevels(
      rawMipLevels,
      width,
      height,
      depthOrArrayLayers,
      dimension
    );
    const geometryAllowsMsaa =
      dimension === '2d' && mipLevelCount <= 1 && depthOrArrayLayers === 1;
    const sampleCount =
      Number.isFinite(rawSampleCount) && rawSampleCount >= 4 && geometryAllowsMsaa ? 4 : 1;

    if (targetOut) {
      const size = pool.size;
      size.width = width;
      size.height = height;
      size.depthOrArrayLayers = depthOrArrayLayers;
      const descriptor = pool.descriptor;
      descriptor.size = size;
      descriptor.mipLevelCount = mipLevelCount;
      descriptor.sampleCount = sampleCount;
      descriptor.dimension = dimension;
      descriptor.format = format;
      descriptor.usage = usage;
      if (viewFormats.length > 0) {
        descriptor.viewFormats = viewFormats;
      } else if ('viewFormats' in descriptor) {
        delete descriptor.viewFormats;
      }
      targetOut.id = id;
      targetOut.descriptor = descriptor;
      if ('error' in targetOut) delete targetOut.error;
      return targetOut;
    }

    const descriptor = {
      size: { width, height, depthOrArrayLayers },
      mipLevelCount,
      sampleCount,
      dimension,
      format,
      usage,
    };
    if (viewFormats.length > 0) descriptor.viewFormats = viewFormats;
    return { id, descriptor };
  }

  function normalizeCopySize(copySize) {
    if (Array.isArray(copySize)) {
      return {
        width: Math.max(1, Math.floor(Number(copySize[0]) || 1)),
        height: Math.max(1, Math.floor(Number(copySize[1] !== undefined ? copySize[1] : 1) || 1)),
        depthOrArrayLayers: Math.max(
          1,
          Math.floor(Number(copySize[2] !== undefined ? copySize[2] : 1) || 1)
        ),
      };
    }
    if (copySize && typeof copySize === 'object') {
      return {
        width: Math.max(1, Math.floor(Number(copySize.width) || 1)),
        height: Math.max(1, Math.floor(Number(copySize.height !== undefined ? copySize.height : 1) || 1)),
        depthOrArrayLayers: Math.max(
          1,
          Math.floor(
            Number(
              copySize.depthOrArrayLayers !== undefined
                ? copySize.depthOrArrayLayers
                : copySize.depthOrLayers !== undefined
                  ? copySize.depthOrLayers
                  : 1
            ) || 1
          )
        ),
      };
    }
    throw new TypeError('copySize must be a GPUExtent3D object or [width, height, depth] array');
  }

  function calcMipDimension(baseDim, mipLevel) {
    const dim = Number.isFinite(baseDim) ? Math.floor(baseDim) : 1;
    if (dim <= 1) {
      return 1;
    }
    const mip = Number.isFinite(mipLevel) ? Math.floor(mipLevel) : 0;
    if (mip <= 0) {
      return dim;
    }
    if (mip >= 32) {
      return 1;
    }
    return Math.max(1, Math.floor(dim / Math.pow(2, mip)));
  }

  function extractTextureGeometry(texture, trackedMeta) {
    if (trackedMeta) {
      return trackedMeta;
    }
    if (!texture || typeof texture !== 'object') {
      return null;
    }
    const desc = texture.desc && typeof texture.desc === 'object' ? texture.desc : null;
    const size = desc && desc.size && typeof desc.size === 'object' ? desc.size : null;
    const width =
      Number.isInteger(texture.width) && texture.width >= 1
        ? texture.width
        : size && Number.isInteger(size.width) && size.width >= 1
          ? size.width
          : 0;
    const height =
      Number.isInteger(texture.height) && texture.height >= 1
        ? texture.height
        : size && Number.isInteger(size.height) && size.height >= 1
          ? size.height
          : 0;
    if (width < 1 || height < 1) {
      return null;
    }
    const depthOrArrayLayers =
      Number.isInteger(texture.depthOrArrayLayers) && texture.depthOrArrayLayers >= 1
        ? texture.depthOrArrayLayers
        : size && Number.isInteger(size.depthOrArrayLayers) && size.depthOrArrayLayers >= 1
          ? size.depthOrArrayLayers
          : 1;
    const dimension =
      typeof texture.dimension === 'string' && texture.dimension
        ? texture.dimension
        : desc && typeof desc.dimension === 'string' && desc.dimension
          ? desc.dimension
          : '2d';
    const mipLevelCount =
      Number.isInteger(texture.mipLevelCount) && texture.mipLevelCount >= 1
        ? texture.mipLevelCount
        : desc && Number.isInteger(desc.mipLevelCount) && desc.mipLevelCount >= 1
          ? desc.mipLevelCount
          : computeMaxMipLevels(width, height, depthOrArrayLayers, dimension);
    const format =
      typeof texture.format === 'string' && texture.format
        ? texture.format
        : desc && typeof desc.format === 'string' && desc.format
          ? desc.format
          : null;
    return { width, height, depthOrArrayLayers, mipLevelCount, dimension, format };
  }

  /**
   * Zero-alloc in-place sub-region dirty-rect bounds clamping & validation
   * against mip dimensions (using physical 4x4 block-aligned mip sizes for
   * BC1..BC7 compressed textures so W265 1x1/2x2/3x3 tail mips stay valid).
   */
  function clampSubRegionToMipBoundsInto(
    target,
    texMeta,
    mipLevel,
    originX,
    originY,
    originZ,
    width,
    height,
    depthOrArrayLayers,
    blockInfo
  ) {
    const isCompressed = Boolean(blockInfo && blockInfo.isCompressed);
    const bw = isCompressed ? blockInfo.blockWidth : 1;
    const bh = isCompressed ? blockInfo.blockHeight : 1;

    let clamped = false;
    let ox = Number.isFinite(originX) ? Math.floor(originX) : 0;
    let oy = Number.isFinite(originY) ? Math.floor(originY) : 0;
    let oz = Number.isFinite(originZ) ? Math.floor(originZ) : 0;
    if (ox < 0) {
      ox = 0;
      clamped = true;
    }
    if (oy < 0) {
      oy = 0;
      clamped = true;
    }
    if (oz < 0) {
      oz = 0;
      clamped = true;
    }

    if (isCompressed) {
      const alignedOx = Math.floor(ox / bw) * bw;
      const alignedOy = Math.floor(oy / bh) * bh;
      if (alignedOx !== ox) {
        ox = alignedOx;
        clamped = true;
      }
      if (alignedOy !== oy) {
        oy = alignedOy;
        clamped = true;
      }
    }

    let w = Number.isFinite(width) ? Math.floor(width) : 0;
    let h = Number.isFinite(height) ? Math.floor(height) : 0;
    let d = Number.isFinite(depthOrArrayLayers) ? Math.floor(depthOrArrayLayers) : 1;

    if (w <= 0 || h <= 0 || d <= 0) {
      target.valid = false;
      target.clamped = clamped || w < 0 || h < 0 || d < 0;
      target.outOfBounds = true;
      target.invalidMipLevel = false;
      target.mipLevel = Number.isFinite(mipLevel) ? Math.floor(mipLevel) : 0;
      target.originX = ox;
      target.originY = oy;
      target.originZ = oz;
      target.width = 0;
      target.height = 0;
      target.depthOrArrayLayers = 0;
      target.mipWidth = 0;
      target.mipHeight = 0;
      target.mipDepth = 0;
      return target;
    }

    if (isCompressed) {
      w = Math.ceil(w / bw) * bw;
      h = Math.ceil(h / bh) * bh;
    }

    const mip = Number.isFinite(mipLevel) ? Math.floor(mipLevel) : -1;
    if (!texMeta) {
      if (mip < 0) {
        target.valid = false;
        target.clamped = false;
        target.outOfBounds = true;
        target.invalidMipLevel = true;
        target.mipLevel = mip;
        target.originX = ox;
        target.originY = oy;
        target.originZ = oz;
        target.width = 0;
        target.height = 0;
        target.depthOrArrayLayers = 0;
        target.mipWidth = 0;
        target.mipHeight = 0;
        target.mipDepth = 0;
        return target;
      }
      target.valid = true;
      target.clamped = clamped;
      target.outOfBounds = false;
      target.invalidMipLevel = false;
      target.mipLevel = mip;
      target.originX = ox;
      target.originY = oy;
      target.originZ = oz;
      target.width = w;
      target.height = h;
      target.depthOrArrayLayers = d;
      target.mipWidth = ox + w;
      target.mipHeight = oy + h;
      target.mipDepth = oz + d;
      return target;
    }

    const mipCount =
      Number.isInteger(texMeta.mipLevelCount) && texMeta.mipLevelCount >= 1
        ? texMeta.mipLevelCount
        : 1;
    if (mip < 0 || mip >= mipCount) {
      target.valid = false;
      target.clamped = false;
      target.outOfBounds = true;
      target.invalidMipLevel = true;
      target.mipLevel = mip;
      target.originX = ox;
      target.originY = oy;
      target.originZ = oz;
      target.width = 0;
      target.height = 0;
      target.depthOrArrayLayers = 0;
      target.mipWidth = 0;
      target.mipHeight = 0;
      target.mipDepth = 0;
      return target;
    }

    const dimension = texMeta.dimension || '2d';
    const baseW = Math.max(1, Math.floor(Number(texMeta.width) || 1));
    const baseH = dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(texMeta.height) || 1));
    const baseD =
      dimension === '1d' ? 1 : Math.max(1, Math.floor(Number(texMeta.depthOrArrayLayers) || 1));
    const logicalMipW = calcMipDimension(baseW, mip);
    const logicalMipH = dimension === '1d' ? 1 : calcMipDimension(baseH, mip);
    const mipD =
      dimension === '1d'
        ? 1
        : dimension === '3d'
          ? calcMipDimension(baseD, mip)
          : baseD;
    const mipW = isCompressed ? Math.ceil(logicalMipW / bw) * bw : logicalMipW;
    const mipH = isCompressed ? Math.ceil(logicalMipH / bh) * bh : logicalMipH;

    if (ox >= mipW || oy >= mipH || oz >= mipD) {
      target.valid = false;
      target.clamped = true;
      target.outOfBounds = true;
      target.invalidMipLevel = false;
      target.mipLevel = mip;
      target.originX = ox;
      target.originY = oy;
      target.originZ = oz;
      target.width = 0;
      target.height = 0;
      target.depthOrArrayLayers = 0;
      target.mipWidth = mipW;
      target.mipHeight = mipH;
      target.mipDepth = mipD;
      return target;
    }

    const maxW = mipW - ox;
    const maxH = mipH - oy;
    const maxD = mipD - oz;

    if (w > maxW) {
      w = isCompressed ? Math.floor(maxW / bw) * bw : maxW;
      clamped = true;
    }
    if (h > maxH) {
      h = isCompressed ? Math.floor(maxH / bh) * bh : maxH;
      clamped = true;
    }
    if (d > maxD) {
      d = maxD;
      clamped = true;
    }

    if (w <= 0 || h <= 0 || d <= 0) {
      target.valid = false;
      target.clamped = true;
      target.outOfBounds = true;
      target.invalidMipLevel = false;
      target.mipLevel = mip;
      target.originX = ox;
      target.originY = oy;
      target.originZ = oz;
      target.width = 0;
      target.height = 0;
      target.depthOrArrayLayers = 0;
      target.mipWidth = mipW;
      target.mipHeight = mipH;
      target.mipDepth = mipD;
      return target;
    }

    target.valid = true;
    target.clamped = clamped;
    target.outOfBounds = false;
    target.invalidMipLevel = false;
    target.mipLevel = mip;
    target.originX = ox;
    target.originY = oy;
    target.originZ = oz;
    target.width = w;
    target.height = h;
    target.depthOrArrayLayers = d;
    target.mipWidth = mipW;
    target.mipHeight = mipH;
    target.mipDepth = mipD;
    return target;
  }

  /**
   * Pure helper: validate and clamp a texture sub-region dirty rect against mip
   * dimensions, returning a fresh result object.
   *
   * @param {{ width: number, height?: number, depthOrArrayLayers?: number, mipLevelCount?: number, dimension?: string, format?: string }} texMeta
   * @param {number} mipLevel
   * @param {{ x?: number, y?: number, z?: number } | number[] | null | undefined} origin
   * @param {{ width: number, height?: number, depthOrArrayLayers?: number } | number[]} size
   * @param {string | object | null | undefined} [formatOrBlockInfo]
   */
  function clampSubRegionToMipBounds(texMeta, mipLevel, origin, size, formatOrBlockInfo) {
    let ox = 0;
    let oy = 0;
    let oz = 0;
    if (Array.isArray(origin)) {
      ox = origin[0] !== undefined ? origin[0] : 0;
      oy = origin[1] !== undefined ? origin[1] : 0;
      oz = origin[2] !== undefined ? origin[2] : 0;
    } else if (origin && typeof origin === 'object') {
      ox = origin.x !== undefined ? origin.x : 0;
      oy = origin.y !== undefined ? origin.y : 0;
      oz = origin.z !== undefined ? origin.z : 0;
    }

    let w = 0;
    let h = 1;
    let d = 1;
    if (Array.isArray(size)) {
      w = size[0] !== undefined ? size[0] : 0;
      h = size[1] !== undefined ? size[1] : 1;
      d = size[2] !== undefined ? size[2] : 1;
    } else if (size && typeof size === 'object') {
      w = size.width !== undefined ? size.width : 0;
      h = size.height !== undefined ? size.height : 1;
      d =
        size.depthOrArrayLayers !== undefined
          ? size.depthOrArrayLayers
          : size.depthOrLayers !== undefined
            ? size.depthOrLayers
            : 1;
    } else {
      throw new TypeError('size must be a GPUExtent3D object or [width, height, depth] array');
    }

    const fmt = formatOrBlockInfo !== undefined ? formatOrBlockInfo : texMeta && texMeta.format;
    const blockInfo = getTextureFormatBlockInfo(fmt);
    const out = {
      valid: false,
      clamped: false,
      outOfBounds: false,
      invalidMipLevel: false,
      mipLevel: 0,
      originX: 0,
      originY: 0,
      originZ: 0,
      width: 0,
      height: 0,
      depthOrArrayLayers: 0,
      mipWidth: 0,
      mipHeight: 0,
      mipDepth: 0,
    };
    return clampSubRegionToMipBoundsInto(
      out,
      texMeta || null,
      mipLevel,
      ox,
      oy,
      oz,
      w,
      h,
      d,
      blockInfo
    );
  }

  class TexturePackets {
    constructor(processor, next, options = {}) {
      this.processor = processor;
      this.next = next || null;
      const opts = options && typeof options === 'object' ? options : {};
      // W341 (#1460): Promote 1D textures (width x 1) to WebGPU 2D (width x 1 x 1) by
      // default so shaders with texture_2d bindings never fail 1D/2D view validation.
      if (opts.promote1DTo2D !== undefined) {
        this.promote1DTo2D = opts.promote1DTo2D !== false;
      } else if (opts.promote1DTexturesTo2D !== undefined) {
        this.promote1DTo2D = opts.promote1DTexturesTo2D !== false;
      } else {
        this.promote1DTo2D = true;
      }
      this._bc3dFallbackTextures = new Set();
      this._bc3dFallbackTextureObjs = new WeakSet();
      this._textureFormats = new Map();
      this._textureObjFormats = new WeakMap();
      this._textureMeta = new Map();
      this._textureObjMeta = new WeakMap();
      const rawMaxScratch = opts.maxScratchBytes;
      this._maxScratchBytes =
        Number.isInteger(rawMaxScratch) && rawMaxScratch >= 0
          ? rawMaxScratch
          : DEFAULT_MAX_SCRATCH_BYTES;
      const rawMaxWriteChunk = opts.maxWriteChunkBytes;
      this._maxWriteChunkBytes =
        Number.isInteger(rawMaxWriteChunk) && rawMaxWriteChunk >= WEBGPU_BYTES_PER_ROW_ALIGNMENT
          ? rawMaxWriteChunk
          : DEFAULT_MAX_WRITE_TEXTURE_CHUNK_BYTES;
      this._scratchPool = new TextureScratchPool({ maxBytes: this._maxScratchBytes });

      // Pre-allocated WebGPU descriptor structs reused on hot writeTexture /
      // copyBufferToTexture / stageTextureUpload paths (zero per-upload heap churn).
      this._reusableOrigin = { x: 0, y: 0, z: 0 };
      this._reusableCopyTexture = {
        texture: null,
        mipLevel: 0,
        origin: this._reusableOrigin,
        aspect: 'all',
      };
      this._reusableDataLayout = {
        offset: 0,
        bytesPerRow: undefined,
        rowsPerImage: undefined,
      };
      this._reusableBufferSource = {
        buffer: null,
        offset: 0,
        bytesPerRow: undefined,
        rowsPerImage: undefined,
      };
      this._reusableCopySize = {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
      };
      this._reusableStageOrigin = { x: 0, y: 0, z: 0 };
      this._reusableStageLayout = {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
        mipLevel: 0,
        origin: this._reusableStageOrigin,
        bytesPerRow: 0,
        rowsPerImage: 1,
        aspect: undefined,
      };
      this._reusableClampResult = {
        valid: true,
        clamped: false,
        outOfBounds: false,
        invalidMipLevel: false,
        mipLevel: 0,
        originX: 0,
        originY: 0,
        originZ: 0,
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
        mipWidth: 1,
        mipHeight: 1,
        mipDepth: 1,
      };
      this._reusableScratchAcquire = {
        buffer: null,
        reused: false,
        pooled: false,
        oversized: false,
      };
      this._reusablePreparedData = {
        data: null,
        copiedFromSab: false,
        reusedScratch: false,
      };
      this._reusableRepackResult = {
        data: null,
        bytesPerRow: 0,
        originalBytesPerRow: 0,
        rowCount: 0,
        repacked: false,
        copiedFromSab: false,
        reusedScratch: false,
        usedPooledScratch: false,
      };
      this._reusableDispatchResult = {
        copiedFromSab: false,
        reusedScratch: false,
        chunked: false,
      };
      this._reusableDecodeCreate1 = createTextureDecodeScratch(false);
      this._reusableDecodeCreate2 = createTextureDecodeScratch(true);
      this._pendingStagedTextures = new Set();
      this._sharedDirectQueue = null;
      this._sharedDirectSupported = null;

      this.stats = this._createEmptyStats();
      this.lastError = null;
    }

    get _scratchBuffer() {
      return this._scratchPool.currentBuffer;
    }

    set _scratchBuffer(value) {
      this._scratchPool._slots[0] = value || null;
    }

    _createEmptyStats() {
      return {
        created: 0,
        errors: 0,
        createTexture2Count: 0,
        rowPitchRepackCount: 0,
        sabCopyCount: 0,
        bc3dFallbacks: 0,
        bc3dFallbackUploadsSkipped: 0,
        uploadCount: 0,
        uploadsCount: 0,
        uploadBytes: 0,
        uploadBytesTotal: 0,
        reusedDescriptorCount: 0,
        clampedSubRegionCount: 0,
        zeroCopyUploads: 0,
        scratchRepackUploads: 0,
        directZeroCopyUploads: 0,
        repackedUploads: 0,
        scratchBufferReuses: 0,
        scratchBufferAllocations: 0,
        r8unormUploads: 0,
        bytesUploaded: 0,
        srgbSiblingViewFormatsAdded: 0,
        promoted1DTo2DCount: 0,
        chunkedWrites: 0,
        trimmedStagingBytes: 0,
        destroyedTextures: 0,
      };
    }

    _recordUploadCount() {
      this.stats.uploadCount += 1;
      this.stats.uploadsCount += 1;
    }

    _recordUploadBytes(byteLength) {
      const bytes = toNonNegativeInt(byteLength);
      this.stats.bytesUploaded += bytes;
      this.stats.uploadBytes += bytes;
      this.stats.uploadBytesTotal += bytes;
    }

    _clearReusableDescriptorRefs() {
      this._reusableOrigin.x = 0;
      this._reusableOrigin.y = 0;
      this._reusableOrigin.z = 0;
      this._reusableCopyTexture.texture = null;
      this._reusableCopyTexture.mipLevel = 0;
      this._reusableCopyTexture.origin = this._reusableOrigin;
      this._reusableCopyTexture.aspect = 'all';
      this._reusableDataLayout.offset = 0;
      this._reusableDataLayout.bytesPerRow = undefined;
      this._reusableDataLayout.rowsPerImage = undefined;
      this._reusableBufferSource.buffer = null;
      this._reusableBufferSource.offset = 0;
      this._reusableBufferSource.bytesPerRow = undefined;
      this._reusableBufferSource.rowsPerImage = undefined;
      this._reusableCopySize.width = 1;
      this._reusableCopySize.height = 1;
      this._reusableCopySize.depthOrArrayLayers = 1;
      this._reusableStageOrigin.x = 0;
      this._reusableStageOrigin.y = 0;
      this._reusableStageOrigin.z = 0;
      this._reusableStageLayout.width = 1;
      this._reusableStageLayout.height = 1;
      this._reusableStageLayout.depthOrArrayLayers = 1;
      this._reusableStageLayout.mipLevel = 0;
      this._reusableStageLayout.origin = this._reusableStageOrigin;
      this._reusableStageLayout.bytesPerRow = 0;
      this._reusableStageLayout.rowsPerImage = 1;
      this._reusableStageLayout.aspect = undefined;
      this._reusableScratchAcquire.buffer = null;
      this._reusableScratchAcquire.reused = false;
      this._reusableScratchAcquire.pooled = false;
      this._reusableScratchAcquire.oversized = false;
      this._reusablePreparedData.data = null;
      this._reusablePreparedData.copiedFromSab = false;
      this._reusablePreparedData.reusedScratch = false;
      this._reusableRepackResult.data = null;
      this._reusableRepackResult.bytesPerRow = 0;
      this._reusableRepackResult.originalBytesPerRow = 0;
      this._reusableRepackResult.rowCount = 0;
      this._reusableRepackResult.repacked = false;
      this._reusableRepackResult.copiedFromSab = false;
      this._reusableRepackResult.reusedScratch = false;
      this._reusableRepackResult.usedPooledScratch = false;
      this._reusableDispatchResult.copiedFromSab = false;
      this._reusableDispatchResult.reusedScratch = false;
      this._reusableDispatchResult.chunked = false;
      this._reusableDecodeCreate1.id = 0;
      if ('error' in this._reusableDecodeCreate1) delete this._reusableDecodeCreate1.error;
      this._reusableDecodeCreate1._pool.viewFormats.length = 0;
      this._reusableDecodeCreate2.id = 0;
      if ('error' in this._reusableDecodeCreate2) delete this._reusableDecodeCreate2.error;
      this._reusableDecodeCreate2._pool.viewFormats.length = 0;
    }

    _fillCopyTexture(texture, mipLevel, originX, originY, originZ, aspect) {
      const origin = this._reusableOrigin;
      origin.x = originX;
      origin.y = originY;
      origin.z = originZ;
      const dst = this._reusableCopyTexture;
      dst.texture = texture;
      dst.mipLevel = mipLevel;
      dst.origin = origin;
      dst.aspect = aspect !== undefined ? aspect : 'all';
      return dst;
    }

    _fillDataLayout(offset, bytesPerRow, rowsPerImage) {
      const layout = this._reusableDataLayout;
      layout.offset = offset;
      layout.bytesPerRow = bytesPerRow;
      layout.rowsPerImage = rowsPerImage;
      return layout;
    }

    _fillBufferSource(buffer, offset, bytesPerRow, rowsPerImage) {
      const src = this._reusableBufferSource;
      src.buffer = buffer;
      src.offset = offset;
      src.bytesPerRow = bytesPerRow;
      src.rowsPerImage = rowsPerImage;
      return src;
    }

    _fillCopySize(width, height, depthOrArrayLayers) {
      const sz = this._reusableCopySize;
      sz.width = width;
      sz.height = height;
      sz.depthOrArrayLayers = depthOrArrayLayers;
      return sz;
    }

    _fillStageLayout(
      width,
      height,
      depthOrArrayLayers,
      mipLevel,
      originX,
      originY,
      originZ,
      bytesPerRow,
      rowsPerImage,
      aspect
    ) {
      const origin = this._reusableStageOrigin;
      origin.x = originX;
      origin.y = originY;
      origin.z = originZ;
      const layout = this._reusableStageLayout;
      layout.width = width;
      layout.height = height;
      layout.depthOrArrayLayers = depthOrArrayLayers;
      layout.mipLevel = mipLevel;
      layout.origin = origin;
      layout.bytesPerRow = bytesPerRow;
      layout.rowsPerImage = rowsPerImage;
      layout.aspect = aspect;
      return layout;
    }

    /**
     * Return a sanitized snapshot of texture packet counters (finite non-negative
     * integers only; no host paths, raw buffers, or mutable internal references).
     */
    getStats() {
      return {
        created: toNonNegativeInt(this.stats.created),
        errors: toNonNegativeInt(this.stats.errors),
        createTexture2Count: toNonNegativeInt(this.stats.createTexture2Count),
        rowPitchRepackCount: toNonNegativeInt(this.stats.rowPitchRepackCount),
        sabCopyCount: toNonNegativeInt(this.stats.sabCopyCount),
        bc3dFallbacks: toNonNegativeInt(this.stats.bc3dFallbacks),
        bc3dFallbackUploadsSkipped: toNonNegativeInt(this.stats.bc3dFallbackUploadsSkipped),
        uploadCount: toNonNegativeInt(this.stats.uploadCount),
        uploadsCount: toNonNegativeInt(this.stats.uploadsCount),
        uploadBytes: toNonNegativeInt(this.stats.uploadBytes),
        uploadBytesTotal: toNonNegativeInt(this.stats.uploadBytesTotal),
        reusedDescriptorCount: toNonNegativeInt(this.stats.reusedDescriptorCount),
        clampedSubRegionCount: toNonNegativeInt(this.stats.clampedSubRegionCount),
        zeroCopyUploads: toNonNegativeInt(this.stats.zeroCopyUploads),
        scratchRepackUploads: toNonNegativeInt(this.stats.scratchRepackUploads),
        directZeroCopyUploads: toNonNegativeInt(this.stats.directZeroCopyUploads),
        repackedUploads: toNonNegativeInt(this.stats.repackedUploads),
        scratchBufferReuses: toNonNegativeInt(this.stats.scratchBufferReuses),
        scratchBufferAllocations: toNonNegativeInt(this.stats.scratchBufferAllocations),
        scratchBufferBytes: toNonNegativeInt(
          this._scratchBuffer ? this._scratchBuffer.byteLength : 0
        ),
        r8unormUploads: toNonNegativeInt(this.stats.r8unormUploads),
        bytesUploaded: toNonNegativeInt(this.stats.bytesUploaded),
        srgbSiblingViewFormatsAdded: toNonNegativeInt(this.stats.srgbSiblingViewFormatsAdded),
        promoted1DTo2DCount: toNonNegativeInt(this.stats.promoted1DTo2DCount),
        chunkedWrites: toNonNegativeInt(this.stats.chunkedWrites),
        trimmedStagingBytes: toNonNegativeInt(this.stats.trimmedStagingBytes),
        destroyedTextures: toNonNegativeInt(this.stats.destroyedTextures),
      };
    }

    /**
     * Release retained scratch buffers larger than `maxRetainedBytes` (default 0,
     * freeing all idle scratch memory after boot / scene transitions) and record
     * the reclaimed byte total in `trimmedStagingBytes`.
     *
     * @param {number} [maxRetainedBytes=0]
     * @returns {{ trimmedSlots: number, trimmedBytes: number }}
     */
    trimScratch(maxRetainedBytes = 0) {
      const res = this._scratchPool.trim(maxRetainedBytes);
      if (res.trimmedBytes > 0) {
        this.stats.trimmedStagingBytes += res.trimmedBytes;
      }
      return res;
    }

    /**
     * Reset all telemetry counters to zero while preserving live texture
     * metadata, fallback tracking, and reusable scratch/descriptor pools.
     */
    resetStats() {
      this.stats = this._createEmptyStats();
      this.lastError = null;
    }

    /**
     * Return a lightweight sanitized snapshot of WriteTexture / texture upload
     * telemetry (`uploadCount`, `uploadBytes`, `zeroCopyUploads`,
     * `scratchRepackUploads`, and scratch/SAB counters).
     */
    getUploadStats() {
      return {
        uploadCount: toNonNegativeInt(this.stats.uploadCount),
        uploadBytes: toNonNegativeInt(this.stats.uploadBytes),
        zeroCopyUploads: toNonNegativeInt(this.stats.zeroCopyUploads),
        scratchRepackUploads: toNonNegativeInt(this.stats.scratchRepackUploads),
        scratchBufferAllocations: toNonNegativeInt(this.stats.scratchBufferAllocations),
        scratchBufferReuses: toNonNegativeInt(this.stats.scratchBufferReuses),
        scratchBufferBytes: toNonNegativeInt(
          this._scratchBuffer ? this._scratchBuffer.byteLength : 0
        ),
        sabCopyCount: toNonNegativeInt(this.stats.sabCopyCount),
      };
    }

    get uploadStats() {
      return this.getUploadStats();
    }

    /**
     * Reset all telemetry counters, cached texture format/fallback tracking, and
     * release the persistent scratch buffer. Handler remains usable after reset().
     */
    reset() {
      this._bc3dFallbackTextures.clear();
      this._bc3dFallbackTextureObjs = new WeakSet();
      this._textureFormats.clear();
      this._textureObjFormats = new WeakMap();
      this._textureMeta.clear();
      this._textureObjMeta = new WeakMap();
      this._pendingStagedTextures.clear();
      this._sharedDirectQueue = null;
      this._sharedDirectSupported = null;
      this._scratchPool.reset();
      this._clearReusableDescriptorRefs();
      this.stats = this._createEmptyStats();
      this.lastError = null;
      if (this.next && typeof this.next.reset === 'function') {
        this.next.reset();
      }
    }

    /**
     * Release internal resources (including the persistent scratch buffer) and
     * reset all telemetry counters. Idempotent.
     */
    destroy() {
      this._bc3dFallbackTextures.clear();
      this._bc3dFallbackTextureObjs = new WeakSet();
      this._textureFormats.clear();
      this._textureObjFormats = new WeakMap();
      this._textureMeta.clear();
      this._textureObjMeta = new WeakMap();
      this._pendingStagedTextures.clear();
      this._sharedDirectQueue = null;
      this._sharedDirectSupported = null;
      this._scratchPool.destroy();
      this._clearReusableDescriptorRefs();
      this.stats = this._createEmptyStats();
      this.lastError = null;
      if (this.next && typeof this.next.destroy === 'function') {
        this.next.destroy();
      }
    }

    _acquireScratchSlot0(requiredBytes) {
      const out = this._reusableScratchAcquire;
      if (
        !Number.isInteger(requiredBytes) ||
        requiredBytes <= 0 ||
        requiredBytes > this._maxScratchBytes
      ) {
        out.buffer = null;
        out.reused = false;
        out.pooled = false;
        out.oversized = requiredBytes > this._maxScratchBytes;
        return out;
      }
      this._scratchPool.acquireSlot0Into(out, requiredBytes);
      if (!out.pooled) {
        return out;
      }
      if (out.reused) {
        this.stats.scratchBufferReuses += 1;
      } else {
        this.stats.scratchBufferAllocations += 1;
      }
      return out;
    }

    _acquireScratchBuffer(requiredBytes) {
      const acquired = this._acquireScratchSlot0(requiredBytes);
      return { buffer: acquired.buffer, reused: acquired.reused };
    }

    _prepareDirectWriteDataInto(out, input) {
      const src = toUint8Array(input, 'writeTexture');
      if (!isSharedBuffer(src.buffer)) {
        out.data = src;
        out.copiedFromSab = false;
        out.reusedScratch = false;
        return out;
      }
      if (src.byteLength === 0) {
        out.data = new Uint8Array(0);
        out.copiedFromSab = true;
        out.reusedScratch = false;
        return out;
      }
      const scratch = this._acquireScratchSlot0(src.byteLength);
      if (scratch.buffer) {
        const dst =
          scratch.buffer.byteLength === src.byteLength
            ? scratch.buffer
            : scratch.buffer.subarray(0, src.byteLength);
        dst.set(src, 0);
        out.data = dst;
        out.copiedFromSab = true;
        out.reusedScratch = scratch.reused;
        return out;
      }
      const copy = new Uint8Array(src.byteLength);
      copy.set(src, 0);
      out.data = copy;
      out.copiedFromSab = true;
      out.reusedScratch = false;
      return out;
    }

    _prepareDirectWriteData(input) {
      const res = this._prepareDirectWriteDataInto(this._reusablePreparedData, input);
      return {
        data: res.data,
        copiedFromSab: res.copiedFromSab,
        reusedScratch: res.reusedScratch,
      };
    }

    _repackRowsWithScratchInto(
      out,
      srcBytes,
      bytesPerRow,
      rowCount,
      depthOrLayers = 1,
      lastRowBytes = bytesPerRow
    ) {
      const src = toUint8Array(srcBytes, 'repackRowsTo256');
      if (!Number.isInteger(bytesPerRow) || bytesPerRow < 1) {
        throw new RangeError(
          `repackRowsTo256: bytesPerRow must be a positive integer, got ${bytesPerRow}`
        );
      }
      if (!Number.isInteger(rowCount) || rowCount < 0) {
        throw new RangeError(
          `repackRowsTo256: rowCount must be a non-negative integer, got ${rowCount}`
        );
      }
      if (!Number.isInteger(depthOrLayers) || depthOrLayers < 1) {
        throw new RangeError(
          `repackRowsTo256: depthOrLayers must be a positive integer, got ${depthOrLayers}`
        );
      }

      const totalRows = rowCount * depthOrLayers;
      const alignedBytesPerRow = alignBytesPerRow256(bytesPerRow);
      const fromSab = isSharedBuffer(src.buffer);

      if (totalRows === 0) {
        out.data = new Uint8Array(0);
        out.bytesPerRow = alignedBytesPerRow;
        out.originalBytesPerRow = bytesPerRow;
        out.rowCount = 0;
        out.repacked = false;
        out.copiedFromSab = false;
        out.reusedScratch = false;
        out.usedPooledScratch = false;
        return out;
      }

      const effLastRowBytes =
        Number.isInteger(lastRowBytes) && lastRowBytes > 0
          ? Math.min(bytesPerRow, lastRowBytes)
          : bytesPerRow;
      const fullSrcBytes = bytesPerRow * totalRows;
      const minRequiredSrcBytes = (totalRows - 1) * bytesPerRow + effLastRowBytes;
      if (src.byteLength < minRequiredSrcBytes) {
        throw new RangeError(
          `repackRowsTo256: srcBytes too small (${src.byteLength} bytes; need at least ${minRequiredSrcBytes} bytes for ${totalRows} rows of ${bytesPerRow} bytes)`
        );
      }

      const needsLastRowPad = src.byteLength < fullSrcBytes;
      if (alignedBytesPerRow === bytesPerRow && !needsLastRowPad) {
        const slice = src.byteLength === fullSrcBytes ? src : src.subarray(0, fullSrcBytes);
        if (fromSab) {
          const scratch = this._acquireScratchSlot0(fullSrcBytes);
          let dst;
          if (scratch.buffer) {
            dst =
              scratch.buffer.byteLength === fullSrcBytes
                ? scratch.buffer
                : scratch.buffer.subarray(0, fullSrcBytes);
          } else {
            dst = new Uint8Array(fullSrcBytes);
          }
          dst.set(slice, 0);
          out.data = dst;
          out.bytesPerRow = alignedBytesPerRow;
          out.originalBytesPerRow = bytesPerRow;
          out.rowCount = totalRows;
          out.repacked = false;
          out.copiedFromSab = true;
          out.reusedScratch = scratch.buffer !== null;
          out.usedPooledScratch = scratch.buffer !== null;
          return out;
        }
        out.data = slice;
        out.bytesPerRow = alignedBytesPerRow;
        out.originalBytesPerRow = bytesPerRow;
        out.rowCount = totalRows;
        out.repacked = false;
        out.copiedFromSab = false;
        out.reusedScratch = false;
        out.usedPooledScratch = false;
        return out;
      }

      const totalDstBytes = alignedBytesPerRow * totalRows;
      const scratch = this._acquireScratchSlot0(totalDstBytes);
      let dst;
      let mustZeroPad = false;
      let usedPooled = false;
      if (scratch.buffer) {
        usedPooled = true;
        mustZeroPad = scratch.reused;
        dst =
          scratch.buffer.byteLength === totalDstBytes
            ? scratch.buffer
            : scratch.buffer.subarray(0, totalDstBytes);
      } else {
        dst = new Uint8Array(totalDstBytes);
      }

      copyAndPadRepackedRows(src, dst, bytesPerRow, alignedBytesPerRow, totalRows, mustZeroPad);

      out.data = dst;
      out.bytesPerRow = alignedBytesPerRow;
      out.originalBytesPerRow = bytesPerRow;
      out.rowCount = totalRows;
      out.repacked = true;
      out.copiedFromSab = fromSab;
      out.reusedScratch = usedPooled;
      out.usedPooledScratch = usedPooled;
      return out;
    }

    _repackRowsWithScratch(srcBytes, bytesPerRow, rowCount, depthOrLayers = 1) {
      const res = this._repackRowsWithScratchInto(
        this._reusableRepackResult,
        srcBytes,
        bytesPerRow,
        rowCount,
        depthOrLayers
      );
      return {
        data: res.data,
        bytesPerRow: res.bytesPerRow,
        originalBytesPerRow: res.originalBytesPerRow,
        rowCount: res.rowCount,
        repacked: res.repacked,
        copiedFromSab: res.copiedFromSab,
        reusedScratch: res.reusedScratch,
      };
    }

    _getTextureFormat(texture, textureId) {
      if (textureId !== undefined && this._textureFormats.has(textureId)) {
        return this._textureFormats.get(textureId);
      }
      if (texture && typeof texture === 'object') {
        const mapped = this._textureObjFormats.get(texture);
        if (mapped) {
          return mapped;
        }
        if (typeof texture.format === 'string' && texture.format) {
          return texture.format;
        }
        if (texture.desc && typeof texture.desc.format === 'string' && texture.desc.format) {
          return texture.desc.format;
        }
      }
      return null;
    }

    _getTextureMeta(texture, textureId) {
      if (textureId !== undefined && this._textureMeta.has(textureId)) {
        return this._textureMeta.get(textureId);
      }
      if (texture && typeof texture === 'object') {
        const tracked = this._textureObjMeta.get(texture);
        if (tracked) {
          return tracked;
        }
        return extractTextureGeometry(texture, null);
      }
      return null;
    }

    _clampSubRegion(
      texture,
      textureId,
      mipLevel,
      originX,
      originY,
      originZ,
      width,
      height,
      depthOrArrayLayers,
      blockInfo
    ) {
      const meta = this._getTextureMeta(texture, textureId);
      return clampSubRegionToMipBoundsInto(
        this._reusableClampResult,
        meta,
        mipLevel,
        originX,
        originY,
        originZ,
        width,
        height,
        depthOrArrayLayers,
        blockInfo
      );
    }

    _isR8UnormTexture(texture, textureId) {
      return this._getTextureFormat(texture, textureId) === 'r8unorm';
    }

    handlePacket(opcode, view, payloadOffset, payloadBytes) {
      if (opcode === CREATE_TEXTURE2) {
        if (payloadBytes < CREATE_TEXTURE2_BYTES) {
          this._error(`CreateTexture2: payload ${payloadBytes} < ${CREATE_TEXTURE2_BYTES} bytes`);
          return true;
        }
        const decoded = decodeCreateTexture2(
          view,
          payloadOffset,
          payloadBytes,
          this._reusableDecodeCreate2
        );
        if (decoded.error) {
          this._error(decoded.error);
          return true;
        }
        this._create(decoded.id, decoded.descriptor, true);
        return true;
      }

      if (opcode === CREATE_TEXTURE) {
        if (payloadBytes < CREATE_TEXTURE_BYTES) {
          this._error(`CreateTexture: payload ${payloadBytes} < ${CREATE_TEXTURE_BYTES} bytes`);
          return true;
        }
        const decoded = decodeCreateTexture(
          view,
          payloadOffset,
          payloadBytes,
          this._reusableDecodeCreate1
        );
        if (decoded.error) {
          this._error(decoded.error);
          return true;
        }
        this._create(decoded.id, decoded.descriptor, false);
        return true;
      }

      if (opcode === WRITE_TEXTURE) {
        if (payloadBytes < WRITE_TEXTURE_HEADER_BYTES) {
          this._error(
            `WriteTexture: payload ${payloadBytes} < ${WRITE_TEXTURE_HEADER_BYTES} bytes`
          );
          return true;
        }
        this.handleWriteTexture(view, payloadOffset, payloadBytes);
        return true;
      }

      return this.next ? this.next.handlePacket(opcode, view, payloadOffset, payloadBytes) : false;
    }

    destroyResource(id) {
      const hadTracked =
        this._textureMeta.has(id) ||
        this._textureFormats.has(id) ||
        this._bc3dFallbackTextures.has(id);
      this._bc3dFallbackTextures.delete(id);
      this._textureFormats.delete(id);
      this._textureMeta.delete(id);
      if (hadTracked) {
        this.stats.destroyedTextures += 1;
      }
      if (this.next && typeof this.next.destroyResource === 'function') {
        this.next.destroyResource(id);
      }
    }

    _writeBufferChunked(device, buffer, bufferOffset, bytes) {
      const len = bytes.byteLength;
      if (len === 0) {
        return;
      }
      const chunkMax = Math.max(
        4,
        Math.floor(Math.min(this._maxWriteChunkBytes, DEFAULT_MAX_WRITE_BUFFER_CHUNK_BYTES) / 4) * 4
      );
      if (len <= chunkMax) {
        device.queue.writeBuffer(buffer, bufferOffset, bytes);
        return;
      }
      for (let off = 0; off < len; off += chunkMax) {
        const chunkLen = Math.min(chunkMax, len - off);
        device.queue.writeBuffer(buffer, bufferOffset + off, bytes.subarray(off, off + chunkLen));
      }
      this.stats.chunkedWrites += 1;
    }

    /**
     * Dispatch a direct `queue.writeTexture` upload, automatically splitting huge
     * uploads (> `_maxWriteChunkBytes`, default 8 MiB) into bounded row-strips or
     * depth-slice batches so Dawn's upload ring never spikes and SAB-backed huge
     * textures copy through the bounded reusable scratch buffer without giant
     * transient allocations.
     */
    _dispatchDirectWriteTexture(
      queue,
      texture,
      textureFormat,
      blockInfo,
      effMipLevel,
      effOx,
      effOy,
      effOz,
      rawData,
      layoutOffset,
      layoutBytesPerRow,
      layoutRowsPerImage,
      effW,
      effH,
      effD,
      dstAspect,
      hasSize,
      rawSizeFallback
    ) {
      const src = toUint8Array(rawData, 'writeTexture');
      // Prevent queue.writeTexture from overtaking an earlier staged copyBufferToTexture to
      // the same texture recorded on processor._pendingEncoder (WebGPU queue timeline order).
      const proc = this.processor;
      if (proc) {
        if (!proc._hasPendingUploadCommands) {
          if (this._pendingStagedTextures.size > 0) {
            this._pendingStagedTextures.clear();
          }
        } else if (
          proc._pendingEncoder &&
          this._pendingStagedTextures.has(texture) &&
          typeof proc._flushPendingUploads === 'function'
        ) {
          this._pendingStagedTextures.clear();
          proc._flushPendingUploads(false);
        }
      }
      const chunkMax = this._maxWriteChunkBytes;
      const srcAvailBytes = Math.max(0, src.byteLength - layoutOffset);

      if (
        hasSize &&
        srcAvailBytes > chunkMax &&
        Number.isInteger(layoutBytesPerRow) &&
        layoutBytesPerRow > 0 &&
        effW > 0 &&
        effH > 0 &&
        effD >= 1
      ) {
        const bh = blockInfo && blockInfo.isCompressed ? blockInfo.blockHeight : 1;
        const blocksY = Math.ceil(effH / bh);
        const rpi =
          Number.isInteger(layoutRowsPerImage) && layoutRowsPerImage > 0
            ? layoutRowsPerImage
            : blocksY;
        const sliceBytes = layoutBytesPerRow * rpi;
        const fromSab = isSharedBuffer(src.buffer);
        const canBatchSlices = effD > 1 && sliceBytes > 0 && sliceBytes <= chunkMax;
        const blockRowsPerChunk = Math.max(1, Math.floor(chunkMax / layoutBytesPerRow));
        const canSplitRows = effD > 1 || blockRowsPerChunk < blocksY;

        if (canBatchSlices || canSplitRows) {
          let totalUploadedBytes = 0;
          let anyReusedScratch = false;
          let chunkCount = 0;

          if (canBatchSlices) {
            const slicesPerChunk = Math.max(1, Math.floor(chunkMax / sliceBytes));
            for (let zStart = 0; zStart < effD; zStart += slicesPerChunk) {
              const zCount = Math.min(slicesPerChunk, effD - zStart);
              const byteStart = layoutOffset + zStart * sliceBytes;
              if (byteStart >= src.byteLength) {
                break;
              }
              const byteLen = Math.min(zCount * sliceBytes, src.byteLength - byteStart);
              const rawSlice = src.subarray(byteStart, byteStart + byteLen);
              const prepared = this._prepareDirectWriteDataInto(this._reusablePreparedData, rawSlice);
              if (prepared.reusedScratch) {
                anyReusedScratch = true;
              }
              const reusedDest = this._fillCopyTexture(
                texture,
                effMipLevel,
                effOx,
                effOy,
                effOz + zStart,
                dstAspect
              );
              const reusedLayout = this._fillDataLayout(0, layoutBytesPerRow, rpi);
              const reusedSize = this._fillCopySize(effW, effH, zCount);
              queue.writeTexture(reusedDest, prepared.data, reusedLayout, reusedSize);
              totalUploadedBytes += prepared.data.byteLength;
              prepared.data = null;
              chunkCount += 1;
            }
          } else {
            for (let z = 0; z < effD; z += 1) {
              const sliceBase = layoutOffset + z * sliceBytes;
              for (let blockRow = 0; blockRow < blocksY; blockRow += blockRowsPerChunk) {
                const stripBlockRows = Math.min(blockRowsPerChunk, blocksY - blockRow);
                const yOff = blockRow * bh;
                const stripH = Math.min(effH - yOff, stripBlockRows * bh);
                const byteStart = sliceBase + blockRow * layoutBytesPerRow;
                if (byteStart >= src.byteLength) {
                  break;
                }
                const byteLen = Math.min(
                  stripBlockRows * layoutBytesPerRow,
                  src.byteLength - byteStart
                );
                const rawStrip = src.subarray(byteStart, byteStart + byteLen);
                const prepared = this._prepareDirectWriteDataInto(
                  this._reusablePreparedData,
                  rawStrip
                );
                if (prepared.reusedScratch) {
                  anyReusedScratch = true;
                }
                const reusedDest = this._fillCopyTexture(
                  texture,
                  effMipLevel,
                  effOx,
                  effOy + yOff,
                  effOz + z,
                  dstAspect
                );
                const reusedLayout = this._fillDataLayout(0, layoutBytesPerRow, stripBlockRows);
                const reusedSize = this._fillCopySize(effW, stripH, 1);
                queue.writeTexture(reusedDest, prepared.data, reusedLayout, reusedSize);
                totalUploadedBytes += prepared.data.byteLength;
                prepared.data = null;
                chunkCount += 1;
              }
            }
          }

          if (chunkCount > 0) {
            if (fromSab) {
              this.stats.sabCopyCount += 1;
            }
            if (chunkCount > 1) {
              this.stats.chunkedWrites += 1;
            }
            this.stats.reusedDescriptorCount += 1;
            this._recordUploadCount();
            this._recordUploadBytes(totalUploadedBytes);
            if (!fromSab) {
              this.stats.directZeroCopyUploads += 1;
              this.stats.zeroCopyUploads += 1;
            }
            if (textureFormat === 'r8unorm') {
              this.stats.r8unormUploads += 1;
            }
            const result = this._reusableDispatchResult;
            result.copiedFromSab = fromSab;
            result.reusedScratch = anyReusedScratch;
            result.chunked = chunkCount > 1;
            return result;
          }
        }
      }

      const reusedDest = this._fillCopyTexture(
        texture,
        effMipLevel,
        effOx,
        effOy,
        effOz,
        dstAspect
      );
      const reusedLayout = this._fillDataLayout(layoutOffset, layoutBytesPerRow, layoutRowsPerImage);
      const reusedSize = hasSize ? this._fillCopySize(effW, effH, effD) : rawSizeFallback;

      // https://www.w3.org/TR/webgpu/#dom-gpuqueue-writetexture (§19.2)
      // accepts AllowSharedBufferSource and copies its contents on
      // the content timeline, before this call returns. Probe using the actual
      // write; older implementations synchronously reject SAB views with TypeError.
      // Huge/chunked uploads retain their existing bounded scratch path above.
      if (queue !== this._sharedDirectQueue) {
        this._sharedDirectQueue = queue;
        this._sharedDirectSupported = null;
      }
      let wroteShared = false;
      if (src.byteLength <= chunkMax && isSharedBuffer(src.buffer) && this._sharedDirectSupported !== false) {
        try {
          queue.writeTexture(reusedDest, src, reusedLayout, reusedSize);
          this._sharedDirectSupported = true;
          wroteShared = true;
        } catch (error) {
          if (!(error instanceof TypeError) && (!error || error.name !== 'TypeError')) throw error;
          this._sharedDirectSupported = false;
        }
      }
      const prepared = this._reusablePreparedData;
      if (wroteShared) {
        prepared.data = src;
        prepared.copiedFromSab = false;
        prepared.reusedScratch = false;
      } else {
        this._prepareDirectWriteDataInto(prepared, src);
        if (prepared.copiedFromSab) this.stats.sabCopyCount += 1;
        queue.writeTexture(reusedDest, prepared.data, reusedLayout, reusedSize);
      }

      this.stats.reusedDescriptorCount += 1;
      this._recordUploadCount();
      this._recordUploadBytes(prepared.data.byteLength);
      if (!prepared.copiedFromSab) {
        this.stats.directZeroCopyUploads += 1;
        this.stats.zeroCopyUploads += 1;
      }
      if (textureFormat === 'r8unorm') {
        this.stats.r8unormUploads += 1;
      }

      const result = this._reusableDispatchResult;
      result.copiedFromSab = prepared.copiedFromSab;
      result.reusedScratch = prepared.reusedScratch;
      result.chunked = false;
      prepared.data = null;
      return result;
    }

    /**
     * Safe `queue.writeTexture` wrapper that takes a zero-copy direct upload path
     * for non-shared and supported nonchunked shared views, with a reusable
     * snapshot fallback for legacy queues. Clamps sub-region dirty rects
     * against mip bounds, and reuses pre-allocated WebGPU descriptor structs.
     */
    writeTexture(destination, data, dataLayout, size, textureId) {
      const texture = destination && destination.texture ? destination.texture : null;
      if (texture && this._bc3dFallbackTextureObjs.has(texture)) {
        this.stats.bc3dFallbackUploadsSkipped += 1;
        return { uploaded: false, copiedFromSab: false, skippedBc3dFallback: true };
      }

      const p = this.processor;
      if (!p || !p.device || !p.device.queue || typeof p.device.queue.writeTexture !== 'function') {
        this._error('writeTexture: device.queue.writeTexture is unavailable');
        return { uploaded: false, copiedFromSab: false };
      }

      // Snapshot caller fields into locals first so passing reusable structs is aliasing-safe.
      const dstMipLevel =
        destination && destination.mipLevel !== undefined ? destination.mipLevel : 0;
      const dstAspect = destination ? destination.aspect : undefined;
      const dstOrigin = destination ? destination.origin : null;
      let ox = 0;
      let oy = 0;
      let oz = 0;
      if (Array.isArray(dstOrigin)) {
        ox = dstOrigin[0] !== undefined ? dstOrigin[0] : 0;
        oy = dstOrigin[1] !== undefined ? dstOrigin[1] : 0;
        oz = dstOrigin[2] !== undefined ? dstOrigin[2] : 0;
      } else if (dstOrigin && typeof dstOrigin === 'object') {
        ox = dstOrigin.x !== undefined ? dstOrigin.x : 0;
        oy = dstOrigin.y !== undefined ? dstOrigin.y : 0;
        oz = dstOrigin.z !== undefined ? dstOrigin.z : 0;
      }

      let rawW = 0;
      let rawH = 1;
      let rawD = 1;
      let hasSize = false;
      if (Array.isArray(size)) {
        hasSize = true;
        rawW = size[0] !== undefined ? size[0] : 0;
        rawH = size[1] !== undefined ? size[1] : 1;
        rawD = size[2] !== undefined ? size[2] : 1;
      } else if (size && typeof size === 'object') {
        hasSize = true;
        rawW = size.width !== undefined ? size.width : 0;
        rawH = size.height !== undefined ? size.height : 1;
        rawD =
          size.depthOrArrayLayers !== undefined
            ? size.depthOrArrayLayers
            : size.depthOrLayers !== undefined
              ? size.depthOrLayers
              : 1;
      }

      const layoutOffset =
        dataLayout && dataLayout.offset !== undefined ? dataLayout.offset : 0;
      const layoutBytesPerRow =
        dataLayout && dataLayout.bytesPerRow !== undefined ? dataLayout.bytesPerRow : undefined;
      let layoutRowsPerImage =
        dataLayout && dataLayout.rowsPerImage !== undefined ? dataLayout.rowsPerImage : undefined;

      const textureFormat = this._getTextureFormat(texture, textureId);
      const blockInfo = getTextureFormatBlockInfo(textureFormat);

      let effMipLevel = dstMipLevel;
      let effOx = ox;
      let effOy = oy;
      let effOz = oz;
      let effW = rawW;
      let effH = rawH;
      let effD = rawD;
      let clampedSubRegion = false;

      if (hasSize) {
        const clamped = this._clampSubRegion(
          texture,
          textureId,
          dstMipLevel,
          ox,
          oy,
          oz,
          rawW,
          rawH,
          rawD,
          blockInfo
        );
        if (clamped.invalidMipLevel) {
          this._error(`writeTexture: mipLevel ${dstMipLevel} out of bounds`);
          return {
            uploaded: false,
            copiedFromSab: false,
            outOfBounds: true,
            error: this.lastError,
          };
        }
        if (clamped.clamped) {
          clampedSubRegion = true;
          this.stats.clampedSubRegionCount += 1;
        }
        if (!clamped.valid) {
          return {
            uploaded: false,
            copiedFromSab: false,
            clamped: clampedSubRegion,
            outOfBounds: true,
          };
        }
        effMipLevel = clamped.mipLevel;
        effOx = clamped.originX;
        effOy = clamped.originY;
        effOz = clamped.originZ;
        effW = clamped.width;
        effH = clamped.height;
        effD = clamped.depthOrArrayLayers;

        if (effD === 1 && layoutRowsPerImage !== undefined) {
          const bh = blockInfo && blockInfo.isCompressed ? blockInfo.blockHeight : 1;
          const clampedBlocksY = Math.ceil(effH / bh);
          if (layoutRowsPerImage > clampedBlocksY) {
            layoutRowsPerImage = clampedBlocksY;
          }
        }
      }

      const dispatched = this._dispatchDirectWriteTexture(
        p.device.queue,
        texture,
        textureFormat,
        blockInfo,
        effMipLevel,
        effOx,
        effOy,
        effOz,
        data,
        layoutOffset,
        layoutBytesPerRow,
        layoutRowsPerImage,
        effW,
        effH,
        effD,
        dstAspect,
        hasSize,
        size
      );

      return {
        uploaded: true,
        copiedFromSab: dispatched.copiedFromSab,
        zeroCopy: !dispatched.copiedFromSab,
        reusedScratch: dispatched.reusedScratch,
        clamped: clampedSubRegion,
      };
    }

    /**
     * Internal zero-alloc direct queue.writeTexture fast path called by
     * `handleWriteTexture` after sub-region clamping and format lookup have
     * already been completed (avoids double clamping and throwaway return objects).
     */
    _writeTextureDirectPreclamped(
      texture,
      textureFormat,
      blockInfo,
      effMipLevel,
      effOx,
      effOy,
      effOz,
      rawData,
      layoutBytesPerRow,
      layoutRowsPerImage,
      effW,
      effH,
      effD
    ) {
      const p = this.processor;
      if (!p || !p.device || !p.device.queue || typeof p.device.queue.writeTexture !== 'function') {
        this._error('writeTexture: device.queue.writeTexture is unavailable');
        return;
      }

      this._dispatchDirectWriteTexture(
        p.device.queue,
        texture,
        textureFormat,
        blockInfo,
        effMipLevel,
        effOx,
        effOy,
        effOz,
        rawData,
        0,
        layoutBytesPerRow,
        layoutRowsPerImage,
        effW,
        effH,
        effD,
        'all',
        true,
        null
      );
    }

    /**
     * Expand `#973` `A8_UNORM` font/mask atlas data (`1 Bpp`) into `rgba8unorm`
     * `(0, 0, 0, a)` using the handler's bounded persistent scratch buffer and
     * upload to `destination.texture` via `stagingPool` (256-byte row-pitch
     * aligned) or direct `queue.writeTexture`.
     *
     * @param {{ texture: object, mipLevel?: number, origin?: { x?: number, y?: number, z?: number }, aspect?: string }} destination
     * @param {Uint8Array | ArrayBufferView | ArrayBuffer | SharedArrayBuffer} a8Bytes
     * @param {{ width: number, height?: number, depthOrArrayLayers?: number } | number[]} size
     * @param {{
     *   srcBytesPerRow?: number,
     *   srcRowsPerImage?: number,
     *   alignTo256?: boolean,
     *   preferDirect?: boolean,
     *   textureId?: number,
     * }} [options]
     */
    writeA8Texture(destination, a8Bytes, size, options = {}) {
      if (!destination || typeof destination !== 'object' || !destination.texture) {
        throw new TypeError('writeA8Texture requires a destination with a texture');
      }
      if (this._bc3dFallbackTextureObjs.has(destination.texture)) {
        this.stats.bc3dFallbackUploadsSkipped += 1;
        return { uploaded: false, copiedFromSab: false, skippedBc3dFallback: true };
      }

      const p = this.processor;
      if (!p || !p.device) {
        this._error('writeA8Texture: device is unavailable');
        return { uploaded: false, copiedFromSab: false, error: this.lastError };
      }

      const opts = options && typeof options === 'object' ? options : {};
      let rawW;
      let rawH;
      let rawD;
      if (Array.isArray(size)) {
        rawW = Math.max(1, Math.floor(Number(size[0]) || 1));
        rawH = Math.max(1, Math.floor(Number(size[1] !== undefined ? size[1] : 1) || 1));
        rawD = Math.max(1, Math.floor(Number(size[2] !== undefined ? size[2] : 1) || 1));
      } else if (size && typeof size === 'object') {
        rawW = Math.max(1, Math.floor(Number(size.width) || 1));
        rawH = Math.max(1, Math.floor(Number(size.height !== undefined ? size.height : 1) || 1));
        rawD = Math.max(
          1,
          Math.floor(
            Number(
              size.depthOrArrayLayers !== undefined
                ? size.depthOrArrayLayers
                : size.depthOrLayers !== undefined
                  ? size.depthOrLayers
                  : 1
            ) || 1
          )
        );
      } else {
        throw new TypeError('copySize must be a GPUExtent3D object or [width, height, depth] array');
      }

      const dstMipLevel = destination.mipLevel !== undefined ? destination.mipLevel : 0;
      const dstAspect = destination.aspect;
      const dstOrigin = destination.origin;
      let ox = 0;
      let oy = 0;
      let oz = 0;
      if (Array.isArray(dstOrigin)) {
        ox = dstOrigin[0] !== undefined ? dstOrigin[0] : 0;
        oy = dstOrigin[1] !== undefined ? dstOrigin[1] : 0;
        oz = dstOrigin[2] !== undefined ? dstOrigin[2] : 0;
      } else if (dstOrigin && typeof dstOrigin === 'object') {
        ox = dstOrigin.x !== undefined ? dstOrigin.x : 0;
        oy = dstOrigin.y !== undefined ? dstOrigin.y : 0;
        oz = dstOrigin.z !== undefined ? dstOrigin.z : 0;
      }

      const clamped = this._clampSubRegion(
        destination.texture,
        opts.textureId,
        dstMipLevel,
        ox,
        oy,
        oz,
        rawW,
        rawH,
        rawD,
        null
      );
      if (clamped.invalidMipLevel) {
        this._error(`writeA8Texture: mipLevel ${dstMipLevel} out of bounds`);
        return { uploaded: false, copiedFromSab: false, outOfBounds: true, error: this.lastError };
      }
      if (clamped.clamped) {
        this.stats.clampedSubRegionCount += 1;
      }
      if (!clamped.valid) {
        return {
          uploaded: false,
          copiedFromSab: false,
          clamped: clamped.clamped,
          outOfBounds: true,
        };
      }

      const effW = clamped.width;
      const effH = clamped.height;
      const effD = clamped.depthOrArrayLayers;
      const effMip = clamped.mipLevel;
      const effOx = clamped.originX;
      const effOy = clamped.originY;
      const effOz = clamped.originZ;
      const clampedSubRegion = clamped.clamped;

      const useStaging = Boolean(
        !opts.preferDirect &&
        p.stagingPool &&
        !p.stagingPool._destroyed &&
        typeof p.stagingPool.stageTextureUpload === 'function' &&
        typeof p._ensurePendingEncoder === 'function'
      );
      const alignBytesPerRow =
        useStaging || opts.alignTo256 ? WEBGPU_BYTES_PER_ROW_ALIGNMENT : 1;
      const unalignedDstBpr = effW * 4;
      const alignedDstBpr = Math.ceil(unalignedDstBpr / alignBytesPerRow) * alignBytesPerRow;
      const totalDstBytes = alignedDstBpr * effH * effD;

      const acquired = this._acquireScratchSlot0(totalDstBytes);
      const scratchBuf = acquired.buffer;
      const scratchReused = acquired.reused;
      const expanded = expandA8ToRgba8(a8Bytes, effW, effH, {
        depthOrLayers: effD,
        srcBytesPerRow: opts.srcBytesPerRow !== undefined ? opts.srcBytesPerRow : rawW,
        srcRowsPerImage: opts.srcRowsPerImage !== undefined ? opts.srcRowsPerImage : rawH,
        alignBytesPerRow,
        scratchBuffer: scratchBuf,
      });
      acquired.buffer = null;

      if (expanded.copiedFromSab) {
        this.stats.sabCopyCount += 1;
      }

      if (useStaging) {
        const encoder = p._ensurePendingEncoder();
        if (encoder && typeof encoder.copyBufferToTexture === 'function') {
          if (expanded.repacked) {
            this.stats.rowPitchRepackCount += 1;
            this.stats.repackedUploads += 1;
            if (scratchBuf !== null) {
              this.stats.scratchRepackUploads += 1;
            }
          }
          const stageLayout = this._fillStageLayout(
            effW,
            effH,
            effD,
            effMip,
            effOx,
            effOy,
            effOz,
            expanded.bytesPerRow,
            expanded.rowsPerImage,
            dstAspect
          );
          p.stagingPool.stageTextureUpload(encoder, destination.texture, expanded.data, stageLayout);
          p._hasPendingUploadCommands = true;
          this.stats.reusedDescriptorCount += 1;
          this._recordUploadCount();
          this._recordUploadBytes(expanded.data.byteLength);
          return {
            uploaded: true,
            staged: true,
            width: effW,
            height: effH,
            depthOrArrayLayers: effD,
            bytesPerRow: expanded.bytesPerRow,
            unalignedBytesPerRow: expanded.unalignedBytesPerRow,
            rowsPerImage: expanded.rowsPerImage,
            repacked: expanded.repacked,
            copiedFromSab: expanded.copiedFromSab,
            reusedScratch: scratchReused,
            pooledScratch: scratchBuf !== null,
            clamped: clampedSubRegion,
          };
        }
        if (!p._hasPendingUploadCommands) {
          p._pendingEncoder = null;
        }
      }

      if (!p.device.queue || typeof p.device.queue.writeTexture !== 'function') {
        this._error('writeA8Texture: device.queue.writeTexture is unavailable');
        return { uploaded: false, copiedFromSab: expanded.copiedFromSab, error: this.lastError };
      }

      if (expanded.repacked) {
        this.stats.rowPitchRepackCount += 1;
        this.stats.repackedUploads += 1;
        if (scratchBuf !== null) {
          this.stats.scratchRepackUploads += 1;
        }
      }

      const reusedDest = this._fillCopyTexture(
        destination.texture,
        effMip,
        effOx,
        effOy,
        effOz,
        dstAspect
      );
      const reusedLayout = this._fillDataLayout(0, expanded.bytesPerRow, expanded.rowsPerImage);
      const reusedSize = this._fillCopySize(effW, effH, effD);
      p.device.queue.writeTexture(reusedDest, expanded.data, reusedLayout, reusedSize);
      this.stats.reusedDescriptorCount += 1;
      this._recordUploadCount();
      this._recordUploadBytes(expanded.data.byteLength);

      return {
        uploaded: true,
        staged: false,
        width: effW,
        height: effH,
        depthOrArrayLayers: effD,
        bytesPerRow: expanded.bytesPerRow,
        unalignedBytesPerRow: expanded.unalignedBytesPerRow,
        rowsPerImage: expanded.rowsPerImage,
        repacked: expanded.repacked,
        copiedFromSab: expanded.copiedFromSab,
        reusedScratch: scratchReused,
        pooledScratch: scratchBuf !== null,
        clamped: clampedSubRegion,
      };
    }

    /**
     * Hardened `encoder.copyBufferToTexture` wrapper:
     * - Repacks CPU texture data (`sourceBytes` or `source.data`) via `repackRowsTo256`
     *   whenever `bytesPerRow` is not a multiple of 256 bytes, incrementing
     *   `stats.rowPitchRepackCount`.
     * - Safely copies `SharedArrayBuffer`-backed views out before staging/uploading,
     *   incrementing `stats.sabCopyCount`.
     * - When copying from an existing `GPUBuffer` with an unaligned 4-byte-multiple
     *   `bytesPerRow`, performs a GPU-side row-by-row repack via `copyBufferToBuffer`
     *   into a 256-byte-aligned staging buffer before `copyBufferToTexture`.
     */
    copyBufferToTexture(encoder, source, destination, copySize, sourceBytes) {
      if (!encoder || typeof encoder.copyBufferToTexture !== 'function') {
        throw new TypeError('copyBufferToTexture requires an encoder with copyBufferToTexture()');
      }
      if (!source || typeof source !== 'object') {
        throw new TypeError('copyBufferToTexture requires a source descriptor object');
      }
      if (!destination || typeof destination !== 'object' || !destination.texture) {
        throw new TypeError('copyBufferToTexture requires a destination with a texture');
      }

      if (this._bc3dFallbackTextureObjs.has(destination.texture)) {
        this.stats.bc3dFallbackUploadsSkipped += 1;
        return {
          repacked: false,
          copiedFromSab: false,
          bytesPerRow: 0,
          originalBytesPerRow: Number.isInteger(source.bytesPerRow) ? source.bytesPerRow : 0,
          staged: false,
          skippedBc3dFallback: true,
        };
      }

      const textureFormat = this._getTextureFormat(destination.texture);
      const blockInfo = getTextureFormatBlockInfo(textureFormat);
      const isCompressed = Boolean(blockInfo && blockInfo.isCompressed);
      const bw = isCompressed ? blockInfo.blockWidth : 1;
      const bh = isCompressed ? blockInfo.blockHeight : 1;

      // Extract raw copySize without allocating temporary objects.
      let rawW;
      let rawH;
      let rawD;
      if (Array.isArray(copySize)) {
        rawW = Math.max(1, Math.floor(Number(copySize[0]) || 1));
        rawH = Math.max(1, Math.floor(Number(copySize[1] !== undefined ? copySize[1] : 1) || 1));
        rawD = Math.max(1, Math.floor(Number(copySize[2] !== undefined ? copySize[2] : 1) || 1));
      } else if (copySize && typeof copySize === 'object') {
        rawW = Math.max(1, Math.floor(Number(copySize.width) || 1));
        rawH = Math.max(1, Math.floor(Number(copySize.height !== undefined ? copySize.height : 1) || 1));
        rawD = Math.max(
          1,
          Math.floor(
            Number(
              copySize.depthOrArrayLayers !== undefined
                ? copySize.depthOrArrayLayers
                : copySize.depthOrLayers !== undefined
                  ? copySize.depthOrLayers
                  : 1
            ) || 1
          )
        );
      } else {
        throw new TypeError('copySize must be a GPUExtent3D object or [width, height, depth] array');
      }

      const dstTexture = destination.texture;
      const dstMipLevel = destination.mipLevel !== undefined ? destination.mipLevel : 0;
      const dstAspect = destination.aspect;
      const dstOrigin = destination.origin;
      let ox = 0;
      let oy = 0;
      let oz = 0;
      if (Array.isArray(dstOrigin)) {
        ox = dstOrigin[0] !== undefined ? dstOrigin[0] : 0;
        oy = dstOrigin[1] !== undefined ? dstOrigin[1] : 0;
        oz = dstOrigin[2] !== undefined ? dstOrigin[2] : 0;
      } else if (dstOrigin && typeof dstOrigin === 'object') {
        ox = dstOrigin.x !== undefined ? dstOrigin.x : 0;
        oy = dstOrigin.y !== undefined ? dstOrigin.y : 0;
        oz = dstOrigin.z !== undefined ? dstOrigin.z : 0;
      }

      // Preserve source layout fields before any reusable struct mutations.
      const srcBuffer = source.buffer;
      const srcOffset = source.offset !== undefined ? source.offset : 0;
      const srcBytesPerRow = source.bytesPerRow;
      const srcRowsPerImage = source.rowsPerImage;

      // Compute original source row geometry before clamping copy extent.
      const origBlocksX = isCompressed ? Math.ceil(rawW / bw) : rawW;
      const origBlocksY = isCompressed ? Math.ceil(rawH / bh) : rawH;
      const origMinRowBytes = isCompressed ? origBlocksX * blockInfo.bytesPerBlock : 0;

      const cpuInput = sourceBytes !== undefined ? sourceBytes : source.data;
      const rawBytesForCheck =
        cpuInput !== undefined && cpuInput !== null
          ? toUint8Array(cpuInput, 'copyBufferToTexture')
          : null;

      let origRowsPerImage;
      if (Number.isInteger(srcRowsPerImage) && srcRowsPerImage > 0) {
        if (
          isCompressed &&
          srcRowsPerImage === rawH &&
          srcRowsPerImage > origBlocksY &&
          rawBytesForCheck
        ) {
          const candidateBpr =
            Number.isInteger(srcBytesPerRow) && srcBytesPerRow > 0
              ? srcBytesPerRow
              : origMinRowBytes;
          const minTightBlocksBytes = calcMinTexelCopyBytes(
            candidateBpr,
            origBlocksY,
            origBlocksY,
            rawD,
            origMinRowBytes
          );
          if (
            candidateBpr > 0 &&
            rawBytesForCheck.byteLength < candidateBpr * srcRowsPerImage * rawD &&
            rawBytesForCheck.byteLength >= minTightBlocksBytes
          ) {
            origRowsPerImage = origBlocksY;
          } else {
            origRowsPerImage = srcRowsPerImage;
          }
        } else {
          origRowsPerImage = isCompressed ? Math.max(origBlocksY, srcRowsPerImage) : srcRowsPerImage;
        }
      } else {
        origRowsPerImage = isCompressed ? origBlocksY : rawH;
      }
      const origTotalRows = origRowsPerImage * rawD;

      // Validate and clamp sub-region dirty rect against mip dimensions.
      const clamped = this._clampSubRegion(
        dstTexture,
        undefined,
        dstMipLevel,
        ox,
        oy,
        oz,
        rawW,
        rawH,
        rawD,
        blockInfo
      );
      if (clamped.invalidMipLevel) {
        this._error(`copyBufferToTexture: mipLevel ${dstMipLevel} out of bounds`);
        return {
          repacked: false,
          copiedFromSab: false,
          bytesPerRow: 0,
          originalBytesPerRow: Number.isInteger(srcBytesPerRow) ? srcBytesPerRow : 0,
          staged: false,
          outOfBounds: true,
          error: this.lastError,
        };
      }
      if (clamped.clamped) {
        this.stats.clampedSubRegionCount += 1;
      }
      if (!clamped.valid) {
        return {
          repacked: false,
          copiedFromSab: false,
          bytesPerRow: 0,
          originalBytesPerRow: Number.isInteger(srcBytesPerRow) ? srcBytesPerRow : 0,
          staged: false,
          clamped: clamped.clamped,
          outOfBounds: true,
        };
      }

      const effW = clamped.width;
      const effH = clamped.height;
      const effD = clamped.depthOrArrayLayers;
      const effMip = clamped.mipLevel;
      const effOx = clamped.originX;
      const effOy = clamped.originY;
      const effOz = clamped.originZ;
      const clampedSubRegion = clamped.clamped;

      const blocksX = isCompressed ? Math.ceil(effW / bw) : effW;
      const blocksY = isCompressed ? Math.ceil(effH / bh) : effH;
      const minRowBytes = isCompressed ? blocksX * blockInfo.bytesPerBlock : 0;

      let rowsPerImage = origRowsPerImage;
      if (effD === 1 && rowsPerImage > blocksY) {
        rowsPerImage = blocksY;
      } else if (rowsPerImage < blocksY) {
        rowsPerImage = blocksY;
      }

      const totalRows = rowsPerImage * effD;
      const p = this.processor;
      const device = p && p.device ? p.device : null;

      if (rawBytesForCheck !== null) {
        const rawBytes = rawBytesForCheck;
        let rawBytesPerRow = srcBytesPerRow;
        if (!Number.isInteger(rawBytesPerRow) || rawBytesPerRow <= 0) {
          if (isCompressed && origTotalRows > 0 && rawBytes.byteLength >= origMinRowBytes * origTotalRows) {
            rawBytesPerRow =
              rawBytes.byteLength % origTotalRows === 0 &&
              rawBytes.byteLength / origTotalRows >= origMinRowBytes
                ? rawBytes.byteLength / origTotalRows
                : origMinRowBytes;
          } else if (
            origTotalRows > 0 &&
            rawBytes.byteLength > 0 &&
            rawBytes.byteLength % origTotalRows === 0
          ) {
            rawBytesPerRow = rawBytes.byteLength / origTotalRows;
          } else {
            this._error('copyBufferToTexture: unable to infer positive bytesPerRow');
            return {
              repacked: false,
              copiedFromSab: false,
              bytesPerRow: 0,
              originalBytesPerRow: 0,
              error: this.lastError,
            };
          }
        }
        if (isCompressed && rawBytesPerRow < minRowBytes) {
          this._error(
            `copyBufferToTexture: compressed bytesPerRow ${rawBytesPerRow} < minimum ${minRowBytes}`
          );
          return {
            repacked: false,
            copiedFromSab: false,
            bytesPerRow: rawBytesPerRow,
            originalBytesPerRow: rawBytesPerRow,
            error: this.lastError,
          };
        }

        const tightLastRowBytes =
          blockInfo && blocksX > 0 ? Math.min(rawBytesPerRow, blocksX * blockInfo.bytesPerBlock) : rawBytesPerRow;
        const packed = this._repackRowsWithScratchInto(
          this._reusableRepackResult,
          rawBytes,
          rawBytesPerRow,
          rowsPerImage,
          effD,
          tightLastRowBytes
        );
        if (packed.repacked) {
          this.stats.rowPitchRepackCount += 1;
          this.stats.repackedUploads += 1;
          if (packed.reusedScratch) {
            this.stats.scratchRepackUploads += 1;
          }
        }
        if (packed.copiedFromSab) {
          this.stats.sabCopyCount += 1;
        }
        const isR8 = this._isR8UnormTexture(dstTexture);

        if (
          p &&
          p.stagingPool &&
          !p.stagingPool._destroyed &&
          typeof p.stagingPool.stageTextureUpload === 'function'
        ) {
          const stageLayout = this._fillStageLayout(
            effW,
            effH,
            effD,
            effMip,
            effOx,
            effOy,
            effOz,
            packed.bytesPerRow,
            rowsPerImage,
            dstAspect
          );
          const packedData = packed.data;
          const packedLen = packedData.byteLength;
          const packedRepacked = packed.repacked;
          const packedCopiedFromSab = packed.copiedFromSab;
          const packedReusedScratch = packed.reusedScratch;
          const packedBpr = packed.bytesPerRow;
          packed.data = null;
          p.stagingPool.stageTextureUpload(encoder, dstTexture, packedData, stageLayout);
          if (p._pendingEncoder && encoder === p._pendingEncoder) {
            this._pendingStagedTextures.add(dstTexture);
          }
          this.stats.reusedDescriptorCount += 1;
          this._recordUploadCount();
          this._recordUploadBytes(packedLen);
          if (!packedRepacked && !packedCopiedFromSab) {
            this.stats.zeroCopyUploads += 1;
          }
          if (isR8) {
            this.stats.r8unormUploads += 1;
          }
          return {
            repacked: packedRepacked,
            copiedFromSab: packedCopiedFromSab,
            reusedScratch: packedReusedScratch,
            bytesPerRow: packedBpr,
            originalBytesPerRow: rawBytesPerRow,
            staged: true,
            clamped: clampedSubRegion,
          };
        }

        if (!packed.repacked && srcBuffer) {
          const packedData = packed.data;
          const packedLen = packedData.byteLength;
          const packedCopiedFromSab = packed.copiedFromSab;
          const packedReusedScratch = packed.reusedScratch;
          const packedBpr = packed.bytesPerRow;
          packed.data = null;
          if (device && device.queue && typeof device.queue.writeBuffer === 'function') {
            this._writeBufferChunked(device, srcBuffer, srcOffset, packedData);
          }
          const reusedSrc = this._fillBufferSource(
            srcBuffer,
            srcOffset,
            packedBpr,
            rowsPerImage
          );
          const reusedDest = this._fillCopyTexture(
            dstTexture,
            effMip,
            effOx,
            effOy,
            effOz,
            dstAspect
          );
          const reusedSize = this._fillCopySize(effW, effH, effD);
          encoder.copyBufferToTexture(reusedSrc, reusedDest, reusedSize);
          this.stats.reusedDescriptorCount += 1;
          this._recordUploadCount();
          this._recordUploadBytes(packedLen);
          if (!packedCopiedFromSab) {
            this.stats.zeroCopyUploads += 1;
          }
          if (isR8) {
            this.stats.r8unormUploads += 1;
          }
          return {
            repacked: false,
            copiedFromSab: packedCopiedFromSab,
            reusedScratch: packedReusedScratch,
            bytesPerRow: packedBpr,
            originalBytesPerRow: rawBytesPerRow,
            staged: false,
            clamped: clampedSubRegion,
          };
        }

        if (
          !device ||
          typeof device.createBuffer !== 'function' ||
          !device.queue ||
          typeof device.queue.writeBuffer !== 'function'
        ) {
          packed.data = null;
          this._error(
            'copyBufferToTexture: device.createBuffer/queue.writeBuffer required for repacking'
          );
          return {
            repacked: packed.repacked,
            copiedFromSab: packed.copiedFromSab,
            bytesPerRow: packed.bytesPerRow,
            originalBytesPerRow: rawBytesPerRow,
            error: this.lastError,
          };
        }

        const packedData = packed.data;
        const packedLen = packedData.byteLength;
        const packedRepacked = packed.repacked;
        const packedCopiedFromSab = packed.copiedFromSab;
        const packedReusedScratch = packed.reusedScratch;
        const packedBpr = packed.bytesPerRow;
        packed.data = null;
        const stagingBuffer = device.createBuffer({
          label: 'playgta6-texture-repack-staging',
          size: Math.max(4, packedLen),
          usage: GPU_BUFFER_USAGE_COPY_SRC_DST,
        });
        this._writeBufferChunked(device, stagingBuffer, 0, packedData);
        const reusedSrc = this._fillBufferSource(
          stagingBuffer,
          0,
          packedBpr,
          rowsPerImage
        );
        const reusedDest = this._fillCopyTexture(
          dstTexture,
          effMip,
          effOx,
          effOy,
          effOz,
          dstAspect
        );
        const reusedSize = this._fillCopySize(effW, effH, effD);
        encoder.copyBufferToTexture(reusedSrc, reusedDest, reusedSize);
        this.stats.reusedDescriptorCount += 1;
        this._recordUploadCount();
        this._recordUploadBytes(packedLen);
        if (!packedRepacked && !packedCopiedFromSab) {
          this.stats.zeroCopyUploads += 1;
        }
        if (isR8) {
          this.stats.r8unormUploads += 1;
        }
        return {
          repacked: packedRepacked,
          copiedFromSab: packedCopiedFromSab,
          reusedScratch: packedReusedScratch,
          bytesPerRow: packedBpr,
          originalBytesPerRow: rawBytesPerRow,
          stagingBuffer,
          staged: true,
          clamped: clampedSubRegion,
        };
      }

      // GPU-buffer-only copy path (no CPU bytes supplied).
      const rawBytesPerRow = Number.isInteger(srcBytesPerRow) ? srcBytesPerRow : 0;
      const isR8 = this._isR8UnormTexture(dstTexture);
      if (rawBytesPerRow === 0 || rawBytesPerRow % WEBGPU_BYTES_PER_ROW_ALIGNMENT === 0) {
        const reusedSrc = this._fillBufferSource(
          srcBuffer,
          srcOffset,
          srcBytesPerRow,
          srcRowsPerImage !== undefined ? rowsPerImage : undefined
        );
        const reusedDest = this._fillCopyTexture(
          dstTexture,
          effMip,
          effOx,
          effOy,
          effOz,
          dstAspect
        );
        const reusedSize = this._fillCopySize(effW, effH, effD);
        encoder.copyBufferToTexture(reusedSrc, reusedDest, reusedSize);
        this.stats.reusedDescriptorCount += 1;
        this._recordUploadCount();
        if (rawBytesPerRow > 0) {
          this._recordUploadBytes(rawBytesPerRow * totalRows);
        }
        if (isR8) {
          this.stats.r8unormUploads += 1;
        }
        return {
          repacked: false,
          copiedFromSab: false,
          bytesPerRow: rawBytesPerRow,
          originalBytesPerRow: rawBytesPerRow,
          staged: false,
          clamped: clampedSubRegion,
        };
      }

      const alignedBytesPerRow = alignBytesPerRow256(rawBytesPerRow);
      const repackedTotalBytes = alignedBytesPerRow * totalRows;
      if (
        srcBuffer &&
        rawBytesPerRow % 4 === 0 &&
        typeof encoder.copyBufferToBuffer === 'function' &&
        device &&
        typeof device.createBuffer === 'function'
      ) {
        let dstBuffer;
        let dstOffset = 0;
        if (
          p &&
          p.stagingPool &&
          !p.stagingPool._destroyed &&
          typeof p.stagingPool.allocate === 'function'
        ) {
          const slice = p.stagingPool.allocate(
            Math.max(4, repackedTotalBytes),
            WEBGPU_BYTES_PER_ROW_ALIGNMENT
          );
          dstBuffer = slice.buffer;
          dstOffset = slice.offset;
        } else {
          dstBuffer = device.createBuffer({
            label: 'playgta6-texture-repack-gpu',
            size: Math.max(4, repackedTotalBytes),
            usage: GPU_BUFFER_USAGE_COPY_SRC_DST,
          });
        }

        const srcBufSize =
          typeof srcBuffer.size === 'number' && Number.isFinite(srcBuffer.size) && srcBuffer.size > 0
            ? srcBuffer.size
            : Infinity;
        for (let row = 0; row < totalRows; row += 1) {
          const rowSrcOff = srcOffset + row * rawBytesPerRow;
          let copySize = rawBytesPerRow;
          if (row === totalRows - 1 && rowSrcOff + copySize > srcBufSize) {
            const avail = Math.max(0, srcBufSize - rowSrcOff);
            copySize = Math.floor(avail / 4) * 4;
          }
          if (copySize > 0) {
            encoder.copyBufferToBuffer(
              srcBuffer,
              rowSrcOff,
              dstBuffer,
              dstOffset + row * alignedBytesPerRow,
              copySize
            );
          }
        }
        const reusedSrc = this._fillBufferSource(
          dstBuffer,
          dstOffset,
          alignedBytesPerRow,
          rowsPerImage
        );
        const reusedDest = this._fillCopyTexture(
          dstTexture,
          effMip,
          effOx,
          effOy,
          effOz,
          dstAspect
        );
        const reusedSize = this._fillCopySize(effW, effH, effD);
        encoder.copyBufferToTexture(reusedSrc, reusedDest, reusedSize);
        this.stats.reusedDescriptorCount += 1;
        this._recordUploadCount();
        this.stats.rowPitchRepackCount += 1;
        this.stats.repackedUploads += 1;
        this._recordUploadBytes(repackedTotalBytes);
        if (isR8) {
          this.stats.r8unormUploads += 1;
        }
        return {
          repacked: true,
          copiedFromSab: false,
          bytesPerRow: alignedBytesPerRow,
          originalBytesPerRow: rawBytesPerRow,
          stagingBuffer: dstBuffer,
          staged: true,
          clamped: clampedSubRegion,
        };
      }

      this._error(
        `copyBufferToTexture: unaligned bytesPerRow ${rawBytesPerRow} cannot be repacked without CPU bytes or 4-byte-aligned copyBufferToBuffer`
      );
      return {
        repacked: false,
        copiedFromSab: false,
        bytesPerRow: rawBytesPerRow,
        originalBytesPerRow: rawBytesPerRow,
        error: this.lastError,
      };
    }

    /**
     * Hardened `WriteTexture` (opcode 13) handler:
     * - When `processor.stagingPool` is active, repacks unaligned rows to 256-byte
     *   `bytesPerRow` via `repackRowsTo256` (incrementing `rowPitchRepackCount`) and
     *   copies `SharedArrayBuffer`-backed views safely (incrementing `sabCopyCount`)
     *   before recording `copyBufferToTexture`.
     * - When uploading directly via `device.queue.writeTexture`, copies
     *   `SharedArrayBuffer`-backed views safely (incrementing `sabCopyCount`).
     */
    handleWriteTexture(view, payloadOffset, payloadBytes) {
      const p = this.processor;
      if (!p || !p.device) {
        return;
      }

      const textureId = view.getUint32(payloadOffset, true);
      const mipLevel = view.getUint32(payloadOffset + 4, true);
      const originX = view.getUint32(payloadOffset + 8, true);
      const originY = view.getUint32(payloadOffset + 12, true);
      const originZ = view.getUint32(payloadOffset + 16, true);
      const width = view.getUint32(payloadOffset + 20, true);
      const rawHeight = view.getUint32(payloadOffset + 24, true);
      const depthOrArrayLayers = Math.max(1, view.getUint32(payloadOffset + 28, true));
      const bytesPerRow = view.getUint32(payloadOffset + 32, true);
      const rowsPerImage = view.getUint32(payloadOffset + 36, true);
      const dataLen = view.getUint32(payloadOffset + 40, true);
      if (WRITE_TEXTURE_HEADER_BYTES + dataLen > payloadBytes) {
        return;
      }

      const texture = p.textures.get(textureId);
      if (!texture) {
        return;
      }

      if (this._bc3dFallbackTextures.has(textureId)) {
        this.stats.bc3dFallbackUploadsSkipped += 1;
        return;
      }

      const texMeta = this._getTextureMeta(texture, textureId);
      const is1DOrPromoted = Boolean(
        texMeta && (texMeta.dimension === '1d' || texMeta.requestedDimension === '1d')
      );
      const height = rawHeight === 0 && is1DOrPromoted && width > 0 ? 1 : rawHeight;

      const rawData = new Uint8Array(
        view.buffer,
        view.byteOffset + payloadOffset + WRITE_TEXTURE_HEADER_BYTES,
        dataLen
      );

      const textureFormat = this._getTextureFormat(texture, textureId);
      const blockInfo = getTextureFormatBlockInfo(textureFormat);
      const isCompressed = Boolean(blockInfo && blockInfo.isCompressed);
      const blockWidth = isCompressed ? blockInfo.blockWidth : 1;
      const blockHeight = isCompressed ? blockInfo.blockHeight : 1;
      const origBlocksX = width > 0 ? Math.ceil(width / blockWidth) : 0;
      const origBlocksY = height > 0 ? Math.ceil(height / blockHeight) : 0;
      const origMinRowBytes = isCompressed ? origBlocksX * blockInfo.bytesPerBlock : 0;

      let effectiveRowsPerImage;
      if (rowsPerImage > 0) {
        if (isCompressed && rowsPerImage === height && rowsPerImage > origBlocksY) {
          const candidateBpr = bytesPerRow > 0 ? bytesPerRow : origMinRowBytes;
          const minTightBlocksBytes = calcMinTexelCopyBytes(
            candidateBpr,
            origBlocksY,
            origBlocksY,
            depthOrArrayLayers,
            origMinRowBytes
          );
          if (
            candidateBpr > 0 &&
            dataLen < candidateBpr * rowsPerImage * depthOrArrayLayers &&
            dataLen >= minTightBlocksBytes
          ) {
            effectiveRowsPerImage = origBlocksY;
          } else {
            effectiveRowsPerImage = rowsPerImage;
          }
        } else {
          effectiveRowsPerImage = isCompressed ? Math.max(origBlocksY, rowsPerImage) : rowsPerImage;
        }
      } else {
        effectiveRowsPerImage = isCompressed ? origBlocksY : height;
      }

      const origTotalRows = effectiveRowsPerImage * depthOrArrayLayers;
      let effectiveBytesPerRow = 0;
      if (bytesPerRow > 0) {
        effectiveBytesPerRow = bytesPerRow;
      } else if (origTotalRows > 0 && dataLen > 0) {
        if (isCompressed) {
          if (dataLen % origTotalRows === 0 && dataLen / origTotalRows >= origMinRowBytes) {
            effectiveBytesPerRow = dataLen / origTotalRows;
          } else if (dataLen >= origMinRowBytes * origTotalRows) {
            effectiveBytesPerRow = origMinRowBytes;
          }
        } else if (dataLen % origTotalRows === 0) {
          effectiveBytesPerRow = dataLen / origTotalRows;
        }
      }

      // Clamp sub-region dirty rect against mip dimensions before dispatching GPU work.
      const clamped = this._clampSubRegion(
        texture,
        textureId,
        mipLevel,
        originX,
        originY,
        originZ,
        width,
        height,
        depthOrArrayLayers,
        blockInfo
      );
      if (clamped.invalidMipLevel) {
        this._error(`WriteTexture ${textureId}: mipLevel ${mipLevel} out of bounds`);
        return;
      }
      if (clamped.clamped) {
        this.stats.clampedSubRegionCount += 1;
      }
      if (!clamped.valid) {
        return;
      }

      const effMipLevel = clamped.mipLevel;
      const effOriginX = clamped.originX;
      const effOriginY = clamped.originY;
      const effOriginZ = clamped.originZ;
      const copyWidth = clamped.width;
      const copyHeight = clamped.height;
      const effDepth = clamped.depthOrArrayLayers;
      const clampedBlocksX = Math.ceil(copyWidth / blockWidth);
      const clampedBlocksY = Math.ceil(copyHeight / blockHeight);
      const minRowBytes = isCompressed ? clampedBlocksX * blockInfo.bytesPerBlock : 0;

      if (effDepth === 1 && effectiveRowsPerImage > clampedBlocksY) {
        effectiveRowsPerImage = clampedBlocksY;
      } else if (effectiveRowsPerImage < clampedBlocksY) {
        effectiveRowsPerImage = clampedBlocksY;
      }
      const totalRows = effectiveRowsPerImage * effDepth;

      if (isCompressed && effectiveBytesPerRow > 0 && minRowBytes > 0 && effectiveBytesPerRow < minRowBytes) {
        this._error(
          `WriteTexture ${textureId}: compressed bytesPerRow ${effectiveBytesPerRow} < minimum ${minRowBytes}`
        );
        return;
      }

      const isRowAligned256 =
        effectiveBytesPerRow > 0 &&
        effectiveBytesPerRow % WEBGPU_BYTES_PER_ROW_ALIGNMENT === 0;
      const hasDirectQueueWrite = Boolean(
        p.device.queue && typeof p.device.queue.writeTexture === 'function'
      );

      // Fast path: when source row pitch is already 256-byte aligned (e.g., 4K Bink2
      // 3840x2160 r8unorm Y plane where 3840 % 256 === 0, or 64x64 BC7 / 128x128 BC1),
      // upload directly via queue.writeTexture without staging pool churn, double
      // sub-region clamping, or scratch repacking.
      if (isRowAligned256 && hasDirectQueueWrite && copyWidth > 0 && copyHeight > 0 && dataLen > 0) {
        const directRpi =
          rowsPerImage > 0 || totalRows > 1 ? effectiveRowsPerImage : undefined;
        this._writeTextureDirectPreclamped(
          texture,
          textureFormat,
          blockInfo,
          effMipLevel,
          effOriginX,
          effOriginY,
          effOriginZ,
          rawData,
          effectiveBytesPerRow,
          directRpi,
          copyWidth,
          copyHeight,
          effDepth
        );
        return;
      }

      if (
        p.stagingPool &&
        !p.stagingPool._destroyed &&
        typeof p.stagingPool.stageTextureUpload === 'function' &&
        copyWidth > 0 &&
        copyHeight > 0 &&
        dataLen > 0
      ) {
        const tightLastRowBytes =
          blockInfo && clampedBlocksX > 0
            ? Math.min(effectiveBytesPerRow, clampedBlocksX * blockInfo.bytesPerBlock)
            : effectiveBytesPerRow;
        const minTightSrcBytes = calcMinTexelCopyBytes(
          effectiveBytesPerRow,
          effectiveRowsPerImage,
          clampedBlocksY,
          effDepth,
          tightLastRowBytes
        );
        const canStage =
          effectiveBytesPerRow > 0 && minTightSrcBytes > 0 && dataLen >= minTightSrcBytes;

        if (canStage) {
          const encoder =
            typeof p._ensurePendingEncoder === 'function' ? p._ensurePendingEncoder() : null;
          if (encoder && typeof encoder.copyBufferToTexture === 'function') {
            const packed = this._repackRowsWithScratchInto(
              this._reusableRepackResult,
              rawData,
              effectiveBytesPerRow,
              effectiveRowsPerImage,
              effDepth,
              tightLastRowBytes
            );
            if (packed.repacked) {
              this.stats.rowPitchRepackCount += 1;
              this.stats.repackedUploads += 1;
              if (packed.reusedScratch) {
                this.stats.scratchRepackUploads += 1;
              }
            }
            if (packed.copiedFromSab) {
              this.stats.sabCopyCount += 1;
            }

            const stageLayout = this._fillStageLayout(
              copyWidth,
              copyHeight,
              effDepth,
              effMipLevel,
              effOriginX,
              effOriginY,
              effOriginZ,
              packed.bytesPerRow,
              effectiveRowsPerImage,
              undefined
            );
            const packedData = packed.data;
            const packedLen = packedData.byteLength;
            packed.data = null;
            p.stagingPool.stageTextureUpload(encoder, texture, packedData, stageLayout);
            p._hasPendingUploadCommands = true;
            this._pendingStagedTextures.add(texture);
            this.stats.reusedDescriptorCount += 1;
            this._recordUploadCount();
            this._recordUploadBytes(packedLen);
            if (!packed.repacked && !packed.copiedFromSab) {
              this.stats.zeroCopyUploads += 1;
            }
            if (textureFormat === 'r8unorm') {
              this.stats.r8unormUploads += 1;
            }
            return;
          }
          if (!p._hasPendingUploadCommands) {
            p._pendingEncoder = null;
          }
        }
      }

      const fallbackBpr =
        bytesPerRow > 0
          ? bytesPerRow
          : effectiveBytesPerRow > 0 && (totalRows > 1 || isCompressed)
            ? effectiveBytesPerRow
            : undefined;
      const fallbackRpi =
        rowsPerImage > 0
          ? effectiveRowsPerImage
          : effDepth > 1 && effectiveRowsPerImage > 0
            ? effectiveRowsPerImage
            : undefined;

      this._writeTextureDirectPreclamped(
        texture,
        textureFormat,
        blockInfo,
        effMipLevel,
        effOriginX,
        effOriginY,
        effOriginZ,
        rawData,
        fallbackBpr,
        fallbackRpi,
        copyWidth,
        copyHeight,
        effDepth
      );
    }

    _create(id, descriptor, isV2 = true) {
      const p = this.processor;
      const hadExisting = Boolean(p.textures && p.textures.has(id));
      this._bc3dFallbackTextures.delete(id);
      this._textureFormats.delete(id);
      this._textureMeta.delete(id);
      if (typeof p._destroyTextureIfPresent === 'function') {
        p._destroyTextureIfPresent(id);
      }
      if (hadExisting) {
        this.stats.destroyedTextures += 1;
      }
      // Cached v2 views of a previous texture with this id must not survive.
      if (this.next && typeof this.next.destroyResource === 'function') {
        this.next.destroyResource(id);
      }

      let effectiveDescriptor = descriptor;
      let bc3dFallback = false;
      let srgbSiblingAdded = false;
      let promoted1DTo2D = false;
      const requestedDimension =
        descriptor && typeof descriptor.dimension === 'string' && descriptor.dimension
          ? descriptor.dimension
          : '2d';
      if (
        descriptor &&
        descriptor.dimension === '3d' &&
        isBcCompressedFormat(descriptor.format) &&
        !deviceSupportsBcSliced3d(p && p.device)
      ) {
        // W108 (#734): WebGPU rejects 3D BC-compressed textures without texture-compression-bc-sliced-3d.
        // Preserve sRGB color transfer when falling back from 3D BC*-srgb formats.
        bc3dFallback = true;
        const fallbackFormat = descriptor.format.endsWith('-srgb') ? 'rgba8unorm-srgb' : 'rgba8unorm';
        effectiveDescriptor = {
          ...descriptor,
          size: descriptor.size ? { ...descriptor.size } : descriptor.size,
          format: fallbackFormat,
        };
        if (Array.isArray(descriptor.viewFormats)) {
          const filteredViewFormats = descriptor.viewFormats.filter(
            (vf) => !isBcCompressedFormat(vf)
          );
          if (filteredViewFormats.length > 0) {
            effectiveDescriptor.viewFormats = filteredViewFormats;
          } else {
            delete effectiveDescriptor.viewFormats;
          }
        }
      }

      // Sanitize usage before resolving viewFormats so STORAGE_BINDING compatibility is exact
      // (WebGPU forbids `-srgb` entries in `viewFormats` when STORAGE_BINDING is present), and
      // include COPY_SRC on every created texture so CopyTexture (opcode 62) can copy from any texture.
      if (
        effectiveDescriptor &&
        typeof effectiveDescriptor.format === 'string' &&
        typeof effectiveDescriptor.usage === 'number'
      ) {
        let sanitizedUsage = (effectiveDescriptor.usage >>> 0) | GPU_TEXTURE_USAGE_COPY_SRC;
        const fmt = effectiveDescriptor.format;
        const dev = p && p.device ? p.device : null;
        if (
          (sanitizedUsage & GPU_TEXTURE_USAGE_STORAGE_BINDING) !== 0 &&
          !formatSupportsStorageBinding(fmt, dev)
        ) {
          sanitizedUsage &= ~GPU_TEXTURE_USAGE_STORAGE_BINDING;
        }
        if (
          (sanitizedUsage & GPU_TEXTURE_USAGE_RENDER_ATTACHMENT) !== 0 &&
          !formatSupportsRenderAttachment(fmt, dev)
        ) {
          sanitizedUsage &= ~GPU_TEXTURE_USAGE_RENDER_ATTACHMENT;
        }
        if (sanitizedUsage !== effectiveDescriptor.usage) {
          effectiveDescriptor = {
            ...effectiveDescriptor,
            size: effectiveDescriptor.size ? { ...effectiveDescriptor.size } : effectiveDescriptor.size,
            usage: sanitizedUsage,
          };
        }
      }

      const hasStorageBinding = Boolean(
        effectiveDescriptor &&
          typeof effectiveDescriptor.usage === 'number' &&
          (effectiveDescriptor.usage & GPU_TEXTURE_USAGE_STORAGE_BINDING) !== 0
      );

      if (bc3dFallback) {
        if (hasStorageBinding && Array.isArray(effectiveDescriptor.viewFormats)) {
          const filteredViewFormats = effectiveDescriptor.viewFormats.filter(
            (vf) => typeof vf !== 'string' || !vf.endsWith('-srgb')
          );
          if (filteredViewFormats.length > 0) {
            effectiveDescriptor.viewFormats = filteredViewFormats;
          } else {
            delete effectiveDescriptor.viewFormats;
          }
        }
      } else if (effectiveDescriptor && typeof effectiveDescriptor.format === 'string') {
        const resolved = resolveCompatibleViewFormats(
          effectiveDescriptor.format,
          effectiveDescriptor.viewFormats,
          hasStorageBinding ? { hasStorageBinding: true } : null
        );
        const origViewFormats = Array.isArray(effectiveDescriptor.viewFormats)
          ? effectiveDescriptor.viewFormats
          : null;
        const resolvedViewFormats = resolved.viewFormats || null;
        const changed =
          resolved.siblingAdded ||
          (origViewFormats !== null &&
            (resolvedViewFormats === null ||
              origViewFormats.length !== resolvedViewFormats.length ||
              origViewFormats.some((vf, idx) => vf !== resolvedViewFormats[idx])));
        if (changed) {
          effectiveDescriptor = {
            ...effectiveDescriptor,
            size: effectiveDescriptor.size ? { ...effectiveDescriptor.size } : effectiveDescriptor.size,
          };
          if (resolvedViewFormats && resolvedViewFormats.length > 0) {
            effectiveDescriptor.viewFormats = resolvedViewFormats;
          } else {
            delete effectiveDescriptor.viewFormats;
          }
          srgbSiblingAdded = resolved.siblingAdded;
        } else if (!isV2 && resolvedViewFormats && resolvedViewFormats.length > 0) {
          // decodeCreateTexture (v1) already attached the compatible sibling viewFormat.
          srgbSiblingAdded = true;
        }
      }

      if (this.promote1DTo2D && effectiveDescriptor && effectiveDescriptor.dimension === '1d') {
        // W341 (#1460): Promote 1D LUT textures to WebGPU 2D (width x 1 x 1) so default
        // and explicit 2D texture views match shader texture_2d bindings cleanly.
        const rawSize = effectiveDescriptor.size;
        const rawWidth = Array.isArray(rawSize)
          ? rawSize[0]
          : rawSize && typeof rawSize === 'object'
            ? rawSize.width
            : 1;
        const width = Math.max(1, Math.floor(Number(rawWidth) || 1));
        effectiveDescriptor = {
          ...effectiveDescriptor,
          dimension: '2d',
          size: { width, height: 1, depthOrArrayLayers: 1 },
          mipLevelCount: 1,
        };
        if (effectiveDescriptor.sampleCount !== undefined) {
          effectiveDescriptor.sampleCount = 1;
        }
        promoted1DTo2D = true;
      }

      if (
        effectiveDescriptor === this._reusableDecodeCreate1.descriptor ||
        effectiveDescriptor === this._reusableDecodeCreate2.descriptor
      ) {
        const copy = {
          size: {
            width: effectiveDescriptor.size.width,
            height: effectiveDescriptor.size.height,
            depthOrArrayLayers: effectiveDescriptor.size.depthOrArrayLayers,
          },
          mipLevelCount: effectiveDescriptor.mipLevelCount,
          format: effectiveDescriptor.format,
          usage: effectiveDescriptor.usage,
          dimension: effectiveDescriptor.dimension,
        };
        if (effectiveDescriptor.sampleCount !== undefined) {
          copy.sampleCount = effectiveDescriptor.sampleCount;
        }
        if (Array.isArray(effectiveDescriptor.viewFormats) && effectiveDescriptor.viewFormats.length > 0) {
          copy.viewFormats = effectiveDescriptor.viewFormats.slice();
        }
        effectiveDescriptor = copy;
      } else if (
        effectiveDescriptor &&
        Array.isArray(effectiveDescriptor.viewFormats) &&
        (effectiveDescriptor.viewFormats === this._reusableDecodeCreate1._pool.viewFormats ||
          effectiveDescriptor.viewFormats === this._reusableDecodeCreate2._pool.viewFormats)
      ) {
        effectiveDescriptor.viewFormats = effectiveDescriptor.viewFormats.slice();
      }

      let texture;
      try {
        texture = p.device.createTexture(effectiveDescriptor);
      } catch (err) {
        const label = isV2 ? 'CreateTexture2' : 'CreateTexture';
        this._error(`${label} ${id}: ${String(err && err.message ? err.message : err)}`);
        return;
      }
      p.textures.set(id, texture);
      if (effectiveDescriptor && typeof effectiveDescriptor.format === 'string') {
        this._textureFormats.set(id, effectiveDescriptor.format);
        if (texture && typeof texture === 'object') {
          this._textureObjFormats.set(texture, effectiveDescriptor.format);
        }
      }
      if (effectiveDescriptor && effectiveDescriptor.size) {
        const meta = {
          width: Math.max(1, Math.floor(Number(effectiveDescriptor.size.width) || 1)),
          height: Math.max(1, Math.floor(Number(effectiveDescriptor.size.height) || 1)),
          depthOrArrayLayers: Math.max(
            1,
            Math.floor(Number(effectiveDescriptor.size.depthOrArrayLayers) || 1)
          ),
          mipLevelCount: Math.max(1, Math.floor(Number(effectiveDescriptor.mipLevelCount) || 1)),
          dimension: effectiveDescriptor.dimension || '2d',
          requestedDimension,
          promoted1DTo2D,
          format: effectiveDescriptor.format || null,
          usage: typeof effectiveDescriptor.usage === 'number' ? effectiveDescriptor.usage : 0,
          viewFormats: Array.isArray(effectiveDescriptor.viewFormats)
            ? effectiveDescriptor.viewFormats
            : null,
        };
        this._textureMeta.set(id, meta);
        if (texture && typeof texture === 'object') {
          this._textureObjMeta.set(texture, meta);
        }
      }
      if (srgbSiblingAdded) {
        this.stats.srgbSiblingViewFormatsAdded += 1;
      }
      if (promoted1DTo2D) {
        this.stats.promoted1DTo2DCount += 1;
      }
      if (bc3dFallback) {
        this._bc3dFallbackTextures.add(id);
        if (texture && typeof texture === 'object') {
          this._bc3dFallbackTextureObjs.add(texture);
        }
        this.stats.bc3dFallbacks += 1;
      }
      if (
        (bc3dFallback || promoted1DTo2D) &&
        this.next &&
        this.next.textureMeta &&
        typeof this.next.textureMeta.set === 'function'
      ) {
        // Propagate fallback / 1D->2D promotion metadata to chained RenderStatePackets so
        // CreateBindGroup2 / BeginRenderPass views see the effective format and dimension.
        this.next.textureMeta.set(id, {
          id,
          requestedFormat: descriptor.format,
          format: effectiveDescriptor.format,
          requestedDimension,
          dimension: effectiveDescriptor.dimension,
          promoted1DTo2D,
          bcFallback: bc3dFallback,
          formatFallback: bc3dFallback,
          usage: effectiveDescriptor.usage,
          sampleCount: effectiveDescriptor.sampleCount || 1,
          width: effectiveDescriptor.size ? effectiveDescriptor.size.width : 1,
          height: effectiveDescriptor.size ? effectiveDescriptor.size.height : 1,
          depthOrArrayLayers: effectiveDescriptor.size
            ? effectiveDescriptor.size.depthOrArrayLayers
            : 1,
          viewFormats: Array.isArray(effectiveDescriptor.viewFormats)
            ? effectiveDescriptor.viewFormats
            : null,
        });
      }
      let textureView = null;
      if (texture && typeof texture.createView === 'function') {
        textureView = texture.createView();
        p.textureViews.set(id, textureView);
      }
      if (p.resourceRegistry && typeof p.resourceRegistry.register === 'function') {
        p.resourceRegistry.register('texture', id, texture, {
          byteSize: 0,
          label: `texture:${id}`,
        });
        if (textureView) {
          p.resourceRegistry.register('textureView', id, textureView, {
            byteSize: 0,
            label: `textureView:${id}`,
          });
        }
      }
      this.stats.created += 1;
      if (isV2) {
        this.stats.createTexture2Count += 1;
      }
    }

    _error(message) {
      this.stats.errors += 1;
      this.lastError = message;
    }
  }

  function installTexturePackets(processor, options = {}) {
    const handler = new TexturePackets(processor, processor.extension || null, options);
    processor.extension = handler;

    if (processor && typeof processor === 'object') {
      if (
        typeof processor._handleCreateTexture === 'function' &&
        !processor[HOOK_ORIG_CREATE_TEXTURE]
      ) {
        processor[HOOK_ORIG_CREATE_TEXTURE] = processor._handleCreateTexture;
      }
      if (typeof processor._handleCreateTexture === 'function') {
        processor._handleCreateTexture = function hardenedHandleCreateTexture(view, payloadOffset) {
          const decoded = decodeCreateTexture(
            view,
            payloadOffset,
            undefined,
            handler._reusableDecodeCreate1
          );
          if (decoded.error) {
            handler._error(decoded.error);
            return;
          }
          handler._create(decoded.id, decoded.descriptor, false);
        };
      }

      if (
        typeof processor._handleWriteTexture === 'function' &&
        !processor[HOOK_ORIG_WRITE_TEXTURE]
      ) {
        processor[HOOK_ORIG_WRITE_TEXTURE] = processor._handleWriteTexture;
      }
      if (typeof processor._handleWriteTexture === 'function') {
        processor._handleWriteTexture = function hardenedHandleWriteTexture(
          view,
          payloadOffset,
          payloadBytes
        ) {
          handler.handleWriteTexture(view, payloadOffset, payloadBytes);
        };
      }

      if (typeof processor.destroy === 'function' && !processor[HOOK_ORIG_DESTROY]) {
        processor[HOOK_ORIG_DESTROY] = processor.destroy;
      }
      if (typeof processor[HOOK_ORIG_DESTROY] === 'function') {
        const origDestroy = processor[HOOK_ORIG_DESTROY];
        processor.destroy = function hardenedDestroy() {
          const result = origDestroy.apply(this, arguments);
          handler.destroy();
          return result;
        };
      }
    }

    return handler;
  }

  const exportedApi = {
    TexturePackets,
    TextureScratchPool,
    installTexturePackets,
    decodeCreateTexture,
    decodeCreateTexture2,
    computeMaxMipLevels,
    calcMipDimension,
    clampMipLevels,
    clampSampleCount,
    clampSubRegionToMipBounds,
    alignBytesPerRow256,
    repackRowsTo256,
    expandA8ToRgba8,
    copyFromSharedIfNeeded,
    isSharedArrayBufferView,
    isBcCompressedFormat,
    getTextureFormatBlockInfo,
    alignCopySizeForFormat,
    calcMipUploadLayout,
    deviceSupportsBcSliced3d,
    getSrgbSiblingFormat,
    resolveCompatibleViewFormats,
    WEBGPU_FORMAT_BLOCK_INFO,
    SRGB_SIBLING_FORMATS,
    DEFAULT_MAX_SCRATCH_BYTES,
    DEFAULT_MAX_WRITE_TEXTURE_CHUNK_BYTES,
    DEFAULT_MAX_WRITE_BUFFER_CHUNK_BYTES,
    BC_SLICED_3D_FEATURE,
    CREATE_TEXTURE,
    CREATE_TEXTURE_BYTES,
    WRITE_TEXTURE,
    WRITE_TEXTURE_HEADER_BYTES,
    CREATE_TEXTURE2,
    CREATE_TEXTURE2_BYTES,
    OPCODE_COPY_TEXTURE,
    COPY_TEXTURE_PAYLOAD_BYTES,
    MAX_VIEW_FORMATS,
    WEBGPU_BYTES_PER_ROW_ALIGNMENT,
    TEXTURE_FORMAT_NAMES,
    V1_TEXTURE_FORMAT_NAMES,
    TEXTURE_DIMENSION_NAMES,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6TexturePackets = exportedApi;
  } else if (globalScope) {
    globalScope.PlayGta6TexturePackets = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
